# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Engine of the replay of the VIC-WUR 3.0 groundwater coupling
(tests/test_vicwur_groundwater.py).

The reference runs are tools/vicwur/gw_runs.py's (<data root>/verification/
vicwur_4/gruns/R1):

  G11   GWM, VIC only: the dump build's continuous C run; its OUT_GWRECHARGE
        run month by month through VIC-WUR 3.0's own MODFLOW driver code and
        MF6 6.7.0 (mf_ref.npz).
  G11c  GWM + routing + water use + non-renewable: VIC-WUR 3.0's coupled
        loop, the C restarted every month with the MODFLOW baseflow as its
        DISCHARGE forcing and the capillary rise in its state (dump.bin, the
        months' mf/out_*.npz, loop.npz).

The replay runs the run's connectors (groundwater with profile vicwur3;
G11c also vicwur_routing, vicwur_wateruse_local, vicwur_nonrenew, all with
gwm) as one WurGroup through the coupling runtime, with VIC replaced by a
window connector that gives the group what the C had each day: the dumped
soil and Cv, OUT_GWRECHARGE (double) as the view's `gwrecharge`, and (G11c)
OUT_RUNOFF / OUT_BASEFLOW and the routing runoff, as the replay of
tests/vicwur_l2_replay.py does. It records, per MODFLOW step, the heads,
the exchange per cell and the soil edit (the capillary rise with its
cascade).
"""
import glob
import os
import _paths  # noqa: E402
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'vicwur'))

import vicwur_l2_replay as L2                                       # noqa
from torchvic.coupling import TorchVIC, Connector, Balance, Proposal  # noqa
from torchvic.coupling import registry                               # noqa
from torchvic.coupling.connectors import vicwur_common as VC         # noqa
from torchvic.vicwur.view import StepView                            # noqa
import gw_runs as GW                                                 # noqa

VER = _paths.env_path(('TORCHVIC_VICWUR_VERIFY',), 'verification')
RUNS = os.environ.get('VICWUR_4_GRUNS', VER + '/vicwur_4/gruns/R1')
DAY = 86400.0
F64 = torch.float64
START = '2003-07-01'


def run_dir(run):
    return os.path.join(RUNS, run)


def available(run):
    d = run_dir(run)
    ok = os.path.exists(os.path.join(d, 'dump.bin'))
    if run == 'G11':
        ok = ok and os.path.exists(os.path.join(d, 'mf_ref.npz'))
    else:
        ok = ok and os.path.exists(os.path.join(d, 'loop.npz'))
    return ok


def domain(case):
    g = case.grid
    la, lo = np.meshgrid(g.lat, g.lon, indexing='ij')
    sv = GW.static_values()
    return dict(lat=g.take(la), lon=g.take(lo), area=g.area_active), sv


class GwCase(object):
    """G11: the dump (soil, Cv) and the history files, no routing."""

    def __init__(self, run='G11', nmax=None):
        from torchvic.vicwur import params as P
        self.run = run
        self.rd = run_dir(run)
        self.c = dict(L2.GR.BASE, **L2.GR.runs()[run])
        gpath = os.path.join(self.rd, 'global_W3.txt')
        self.g = P.parse_global(gpath)
        self.inp = P.load_vicwur_inputs(gpath)
        self.grid = grid = self.inp.grid
        self.N = grid.n
        dims, rec = L2.read_dump(os.path.join(self.rd, 'dump.bin'))
        if nmax:
            rec = {k: v[:nmax] for k, v in rec.items()}
        N, NT, NB, NL = (int(x) for x in dims[:4])
        self.NT, self.NB, self.NL = NT, NB, NL
        T = self.T = rec[1].shape[0]
        self.S = L2.soil_arrays(self.g, grid)
        soil = rec[2].reshape(T, N, NT, NB, NL, 2)
        cvd = rec[3].reshape(T, N, NT)
        self.cv = np.nan_to_num(cvd)
        self.moist = np.nan_to_num(soil[..., 0])
        self.ice0 = np.nan_to_num(soil[..., 1])
        self.gw = L2.read_out(self.rd, 'OUT_GWRECHARGE', grid)[:T]


def load(run, nmax=None):
    """The case of a run: GwCase (G11) or the vicwur_l2_replay Case (G11c, read
    from this replay's run directory) with OUT_GWRECHARGE."""
    if run == 'G11':
        return GwCase(run, nmax)
    old = L2.GRUNS
    L2.GRUNS = RUNS
    try:
        c = L2.Case(run, nmax)
    finally:
        L2.GRUNS = old
    c.gw = L2.read_out(c.rd, 'OUT_GWRECHARGE', c.grid)[:c.T]
    return c


class GwVic(L2.DumpVic):
    """DumpVic (tests/vicwur_l2_replay.py) that also hands the group the
    day's OUT_GWRECHARGE and records the groundwater steps. Without a
    router (G11) only the soil and the recharge are given."""

    def __init__(self, case, device, use, group, days=30):
        if isinstance(case, GwCase):
            Connector.__init__(self)
            self.case, self.phase_group = case, group
            self.interval_s = days * DAY
            self.day_coupled = list(use)
            self.dev = torch.device(device)
            self.t_step = 0
            self.rec = {}
            self.local_calls, self.nr_calls = [], []
            self.hooked = False
            self.cur = 0
            t = lambda a: torch.as_tensor(a, dtype=F64, device=self.dev)
            self.mm = t(case.S['mm'])
            self.af = t(case.S['af'])
            self.view_bf = self.view_ro = None
        else:
            super().__init__(case, device, use, group, days)
        self.view_gw = torch.as_tensor(case.gw, dtype=F64, device=self.dev)
        self.gwrec = {}

    def has_router(self):
        return self.phase_group.router is not None

    def hook(self):
        if self.has_router():
            super().hook()

    def view_pre(self, d):
        c = self.case
        if self.has_router():
            v = super().view_pre(d)
        else:
            v = StepView(self._t(c.moist[d]), self.mm, self._t(c.cv[d]),
                         self.af, ice=self._t(c.ice0[d]))
        v.gwrecharge = self.view_gw[d]
        return v

    def between_day(self, d, t_post, pre=True):
        grp, c = self.phase_group, self.case
        self.cur = d
        self.local_calls, self.nr_calls = [], []
        gw = grp.roles['groundwater']
        n0 = gw.nstep
        v = self.view_pre(d)
        base = v.moist.clone()
        delta = grp.between(v, t_post, None, pre=False)
        if self.has_router():
            self.record(d, v)
            self.snap_residues(d)
        if gw.nstep > n0:
            L = gw.last
            self.gwrec[d] = dict(
                head=L['head'].cpu().numpy(), RIV=L['RIV'].cpu().numpy(),
                DRN=L['DRN'].cpu().numpy(), EVT=L['EVT'].cpu().numpy(),
                RCH=L['RCH'].cpu().numpy(), WEL=L['WEL'].cpu().numpy(),
                force=gw.force.cpu().numpy(), cpr=gw.cpr_mm.cpu().numpy(),
                base=base.cpu().numpy(), delta=delta.cpu().numpy())
        if pre and d + 1 < c.T:
            grp.between(self.view_post(d), None, t_post + DAY, post=False)


def connector_cfgs(case, group, mode='exact', period=None):
    dom, sv = domain(case)
    cfg = {}
    if case.run != 'G11':
        cfg = L2.connector_cfgs(case, group)
        cfg['vicwur_routing']['gwm'] = True
        cfg['vicwur_wateruse_local']['gwm'] = True
        cfg['vicwur_nonrenew']['gwm'] = True
    cfg['groundwater'] = dict(
        profile='vicwur3', domain=dom, static=sv, start_date=START,
        mode=mode, period=period, group=group)
    return cfg


def replay(case, device='cpu', days=30, mode='exact', period=None,
           until=None, checkpoint=None, restart=None, tag=''):
    """Run the case through the runtime (ledger on); returns the window and
    the model. `checkpoint` (path) saves at `until` and stops there;
    `restart` (path) continues from such a file."""
    group = 'tv4_%s_%s_%s_%d%s' % (case.run, device, mode, os.getpid(), tag)
    cfgs = connector_cfgs(case, group, mode, period)
    win = GwVic(case, device, list(cfgs), group, days=days)
    nm = 'tv4_window_%s_%s%s' % (case.run, device, tag)
    registry.register(nm, lambda **c: win, 'test', replace=True)
    mods = {'vic': dict(connector=nm)}
    for k, v in cfgs.items():
        mods[k] = dict(connector='groundwater' if k == 'groundwater' else k,
                       **v)
    cfg = dict(run=dict(device=device, log=False), modules=mods,
               connections=[])
    if restart:
        cfg['checkpoint'] = dict(restart_from=restart)
    m = TorchVIC.from_dict(cfg)
    try:
        with torch.inference_mode():
            m.initialize()
            if checkpoint:
                m.run(until=until)
                m.save_checkpoint(checkpoint)
            else:
                m.run(until)
                m.finalize(write=False)
    finally:
        win.unhook()
    return win, m
