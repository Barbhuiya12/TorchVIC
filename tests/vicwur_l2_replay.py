# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Engine of the connector replay tests (tests/test_vicwur_replay_l2.py).

A G-run of R1 (tools/vicwur/gruns.py) is run once by the dump build
(<data root>/build/vicwur3_dump, VICWUR_DUMP; tools/vicwur/replay_runs.py).  This
module replays it through the VIC-WUR connectors:

    vicwur_routing, vicwur_dams, vicwur_wateruse_local, vicwur_wateruse_remote,
    vicwur_nonrenew, vicwur_efr   (those the run has switched on)

One WurGroup runs the connectors' phase hooks in VIC-WUR's order
(torchvic.vicwur.order) through the coupling runtime.  VIC is replaced by
`DumpVic`, a window connector that, like VicConnector's day-coupled path,
runs the group between two days, and gives it what the C had:

  * the soil at the end of the day, every layer, tile and band, from the dump
    (tag 2) and the tile Cv (tag 3).  The dump is written after the plugins
    edited the soil, so the state the plugins SAW (after vic_run, before them)
    is rebuilt for the bottom layer, the only layer these connectors read
    or write: the C pumps W = sum of the sectors' withdrawn_gw from the tiles
    in proportion to their availability u_t (wu_run_local.c:398-419,
    post_t = r_t + u_t (1 - W/A)), so u_t = (post_t - r_t) A / (A - W) with
    A = W + sum_t w_t (post_t - r_t)+, r_t = efr_moist + residual moisture
    + ice, w_t = Cv AreaFract.  An exhausted pool (A = W) leaves no per-tile
    information: every pumped tile gets the same u_t.  The connectors then pump
    on that state and the result is compared with the dump's soil;
  * the VIC runoff and baseflow of the step (OUT_RUNOFF, OUT_BASEFLOW of the
    run, written as float64 by tools/vicwur/replay_runs.py; a local dam
    rewrites OUT_RUNOFF after rout_run read it, so the C's value is rebuilt
    from the dam's dumped inflow, as in tests/test_vicwur_dams_l2.py).
    rout_run's runoff in m3 s-1 is then replaced by the dumped one (tag 4), as
    the routing kernel's replay does (it differs from the rebuilt one by
    rounding only in the local dam's cell);
  * the irrigation module (no connector in this replay) is replayed at kernel
    level: the IRR sector's demand, groundwater and consumption fractions are
    the irrigation kernel's irr_set_demand (irrigation.demand) on the dumped
    requirement and flag;
  * the forcing: PLUGIN_FORCE_TYPE files through PluginForcing, the way the C
    reads them (sector demand, groundwater and consumption fractions, EFR).

Per step it records every field the dump holds (routing, dams, wu_var of
every sector, the bottom-layer soil) and the connectors' state, and the
checks in the test compare them with the dump and with the history files.
"""
import glob
import os
import _paths  # noqa: E402
import struct
import sys

import netCDF4
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'vicwur'))

from torchvic.coupling import TorchVIC, Connector, Balance, Proposal    # noqa
from torchvic.coupling import registry                                  # noqa
from torchvic.coupling.connectors import vicwur_common as VC            # noqa
from torchvic.vicwur import dams as DM                                  # noqa
from torchvic.vicwur import forcing as F                                # noqa
from torchvic.vicwur import fused as FZ                                 # noqa
from torchvic.vicwur import irrigation as IR                            # noqa
from torchvic.vicwur import nonrenew as NR                              # noqa
from torchvic.vicwur import params as P                                 # noqa
from torchvic.vicwur import wateruse as WU                              # noqa
from torchvic.vicwur.view import StepView                               # noqa
import gruns as GR                                                      # noqa

VER = _paths.env_path(('TORCHVIC_VICWUR_VERIFY',), 'verification')
GRUNS = os.environ.get('VICWUR_23_GRUNS', VER + '/vicwur_23/gruns/R1')
START = (2003, 7, 1, 0)
DAY = 86400.0
F64 = torch.float64
SECT = {'mun': 0, 'ene': 1, 'man': 2, 'liv': 3}
IRR_SECT = 4
WUF = ('available_gw', 'available_surf', 'available_dam', 'available_tremote',
       'available_remote', 'available_nonrenew', 'demand_gw', 'demand_surf',
       'demand_tremote', 'demand_remote', 'demand_nonrenew', 'withdrawn_gw',
       'withdrawn_surf', 'withdrawn_dam', 'withdrawn_tremote',
       'withdrawn_remote', 'withdrawn_nonrenew', 'returned', 'consumed',
       'available_remote_tmp', 'demand_remote_tmp', 'withdrawn_remote_tmp')
DAMF = ('active', 'inflow', 'demand', 'efr', 'release', 'storage', 'op_month',
        'months_running', 'total_inflow', 'total_demand', 'total_efr')


def run_dir(run):
    return os.path.join(GRUNS, run)


def available(run):
    d = run_dir(run)
    return (os.path.exists(os.path.join(d, 'dump.bin'))
            and os.path.exists(os.path.join(d, 'W3', 'rc'))
            and open(os.path.join(d, 'W3', 'rc')).read().strip() == '0')


# ------------------------------------------------------------------- dump
def read_dump(path):
    """The dump (plugin_dump.c): dims and, per step, tags 1-7.  Only what the
    replay reads is kept of the soil record (moisture, first ice slot)."""
    out = {k: [] for k in (1, 2, 3, 4, 5, 6, 7)}
    dims = None
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            tag, n = struct.unpack('<iq', h)
            a = np.frombuffer(f.read(8 * n), '<f8')
            if tag == 0:
                dims = a.astype(int)
            elif tag == 2:
                N, NT, NB, NL = (int(x) for x in dims[:4])
                s = a.reshape(N, NT, NB, NL, int(dims[13]))
                out[2].append(np.ascontiguousarray(s[..., :2]))
            elif tag in out:
                out[tag].append(a.copy())
    return dims, {k: np.stack(v) for k, v in out.items() if v}


def read_out(rd, var, grid, trailing=0):
    """A history variable of the run's W3/out files [T, ..., cell] (float32
    in the file, returned as float64); handles closed at once."""
    parts = []
    for p in sorted(glob.glob(rd + '/W3/out/fluxes.*.nc')):
        with netCDF4.Dataset(p) as ds:
            ds.set_auto_mask(False)
            parts.append(np.asarray(ds[var][:], dtype=np.float64))
    return grid.take(np.concatenate(parts))


def soil_arrays(g, grid):
    """Soil of the parameter file, per cell [N, ...]."""
    with netCDF4.Dataset(g['PARAMETERS'][0][0]) as ds:
        raw = {k: np.array(ds[k][:], dtype=np.float64) for k in (
            'depth', 'bulk_density', 'soil_density', 'resid_moist', 'Dsmax',
            'Ds', 'Ws', 'c', 'AreaFract', 'init_moist')}
    f = lambda k: grid.take(raw[k])
    depth, bulk, dens = f('depth').T, f('bulk_density').T, f('soil_density').T
    return dict(
        depth=depth, mm=depth * (1 - bulk / dens) * 1000.,
        resid=f('resid_moist').T, dsmax=f('Dsmax'), ds=f('Ds'), ws=f('Ws'),
        c=f('c'), af=f('AreaFract').T, init=f('init_moist').T)


def plugin_series(g, name, T, cells):
    """A PLUGIN_FORCE_TYPE stream, every model step [T, ncell]; the reader is
    closed before returning."""
    ent = [x for x in g['PLUGIN_FORCE_TYPE'] if x[0] == name][0]
    ms = F.file_info(ent[3] + '%4d.nc' % START[0], ent[1])['start']
    f = F.PluginForcing(ent[3], ent[1], ent[2], START, T, 1, met_start=ms,
                        cells=cells)
    try:
        return np.stack([f.get(i).numpy().copy() for i in range(T)])
    finally:
        f.close()


# ------------------------------------------------------------------- case
class Case(object):
    """One G-run: its configuration, inputs, dump and the per-step arrays."""

    def __init__(self, run, nmax=None):
        self.run = run
        self.rd = run_dir(run)
        self.c = dict(GR.BASE, **GR.runs()[run])
        gpath = os.path.join(self.rd, 'global_W3.txt')
        self.g = g = P.parse_global(gpath)
        self.inp = P.load_vicwur_inputs(gpath)
        self.grid = grid = self.inp.grid
        self.N = grid.n
        dims, rec = read_dump(os.path.join(self.rd, 'dump.bin'))
        if nmax:                                  # the first nmax steps only
            rec = {k: v[:nmax] for k, v in rec.items()}
        self.dims, self.rec = dims, rec
        N, NT, NB, NL, NF = (int(x) for x in dims[:5])
        self.NT, self.NB, self.NL = NT, NB, NL
        self.ring = int(dims[5])
        T = self.T = rec[1].shape[0]
        self.S = soil_arrays(g, grid)
        soil = rec[2].reshape(T, N, NT, NB, NL, 2)
        cvd = rec[3].reshape(T, N, NT)
        self.valid = ~np.isnan(cvd[0])                      # [N, NT]
        self.cv = np.nan_to_num(cvd)
        self.moist = np.nan_to_num(soil[..., 0])            # [T,N,NT,NB,NL]
        self.ice0 = np.nan_to_num(soil[..., 1])             # FROZEN_SOIL FALSE
        self.rout = rec[4].reshape(T, N, 5 + self.ring)
        self.wu = None
        self.W = np.zeros((T, N))
        self.active = np.zeros(5, bool)
        if 5 in rec:
            self.wu = rec[5].reshape(T, N, int(dims[14]), 22)
            act = ~np.isnan(self.wu[0, :, :, 0])
            assert (act == act[:1]).all(), 'sector activity differs by cell'
            self.active = act[0]
            self.W = np.nansum(self.wu[..., 11], 2)         # [T, N]
        self.resid_mm = self.S['resid'][:, -1] * self.S['depth'][:, -1] * 1000.
        self.dam = None
        if 7 in rec:
            nd = int(dims[7])
            self.dam = rec[7].reshape(T, N, nd, -1)
        self.flows()

    # -- inputs ----------------------------------------------------------
    def flows(self):
        """OUT_RUNOFF / OUT_BASEFLOW as rout_run saw them, [T, N]."""
        g, grid, T = self.g, self.grid, self.T
        self.bf = read_out(self.rd, 'OUT_BASEFLOW', grid)[:T]
        ro = read_out(self.rd, 'OUT_RUNOFF', grid)[:T]
        dp = self.inp.dams
        if dp is not None:
            cell, glb = dp.dam_cell.numpy(), dp.is_global.numpy()
            pos = DM.rank_in_cell(cell)
            fr = dp.inflow_fraction.numpy()
            for i in np.where(~glb)[0]:
                c = cell[i]
                row = self.dam[:, c, pos[i]]
                on = row[:, 0] > 0
                pre = row[:, 1] * 1e6 * 1000.0 / (grid.area_active[c] * fr[i])
                ro[on, c] = pre[on]
        self.ro = ro

    def sector_series(self):
        """demand, groundwater fraction, consumption fraction [T, N, 5]."""
        T, N, c = self.T, self.N, self.c
        cells = np.flatnonzero(self.grid.active.ravel())
        out = {k: np.zeros((T, N, 5)) for k in ('demand', 'gw', 'cons')}
        for s in c['wu_sectors']:
            i, S = SECT[s], s.upper()
            out['demand'][:, :, i] = plugin_series(
                self.g, S + '_DEMAND', T, cells)
            out['gw'][:, :, i] = plugin_series(
                self.g, S + '_GROUNDWATER', T, cells)
            out['cons'][:, :, i] = plugin_series(
                self.g, S + '_CONSUMPTION', T, cells)
        if c['irr']:
            D, G, E = self.irr_demand()
            out['demand'][:, :, IRR_SECT] = D
            out['gw'][:, :, IRR_SECT] = G
            out['cons'][:, :, IRR_SECT] = E
        return out

    def irr_demand(self):
        """irr_set_demand (irr_use.c:62-105) by the irrigation kernel on the
        dumped requirement and flag; checked against the dumped wu_var."""
        ir, dims, rec = self.inp.irrigation, self.dims, self.rec
        N, NT, NB, NL, NF, _, NJ = (int(x) for x in dims[:7])
        T = self.T
        iv = rec[6].reshape(T, N, NJ, NB, 8)
        vix, act = ir.veg_index.numpy(), ir.active.numpy()
        jmap = -np.ones((N, NJ), int)
        gw, eff = np.zeros((N, NJ)), np.zeros((N, NJ))
        for c in range(N):
            j = 0
            for k in range(act.shape[1]):
                if act[c, k]:
                    jmap[c, j] = vix[c, k]
                    gw[c, j] = ir.gw_fraction[c, k]
                    eff[c, j] = ir.efficiency[c, k]
                    j += 1
        valid = jmap >= 0
        tile = np.where(valid, jmap, 0)
        ci = np.arange(N)[:, None]
        cv = np.where(valid[None], self.cv[:, ci, tile], 0.0)    # [T,N,J]
        f64 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=F64)
        rep = lambda a: f64(np.broadcast_to(a[None], (T,) + a.shape)
                            .reshape((T * N,) + a.shape[1:]))
        flat = lambda a: f64(a.reshape((T * N,) + a.shape[2:]))
        af = f64(np.broadcast_to(self.S['af'][None], (T, N, NB))
                 .reshape(-1, NB))
        req = flat(np.nan_to_num(iv[..., 4]))
        flag = torch.as_tensor((np.nan_to_num(iv[..., 5]) > 0).reshape(
            T * N, NJ, NB))
        D, G, E, _ = IR.demand(req, flag, flat(cv), af, rep(gw), rep(eff))
        D, G, E = (x.numpy().reshape(T, N) for x in (D, G, E))
        has = valid.any(1)[None]
        dd = np.nan_to_num(self.wu[:, :, IRR_SECT, 6]
                           + self.wu[:, :, IRR_SECT, 7])
        np.testing.assert_allclose(np.where(has, D, 0.0), dd, rtol=1e-9,
                                   atol=1e-9, err_msg='irr_set_demand')
        return D, G, E

    def efr_cfg(self):
        T, N, S = self.T, self.N, self.S
        cells = np.flatnonzero(self.grid.active.ravel())
        return dict(
            discharge=dict(series=plugin_series(self.g, 'EFR_DISCHARGE', T,
                                                cells)),
            baseflow=dict(series=plugin_series(self.g, 'EFR_BASEFLOW', T,
                                               cells)),
            soil=dict(resid=S['resid'][:, -1], depth=S['depth'][:, -1],
                      max_moist=S['mm'][:, -1], dsmax=S['dsmax'], ds=S['ds'],
                      ws=S['ws'], c=S['c'], valid=self.valid))


# ------------------------------------------------------------ the stand-in
def rebuild_bottom(case, d, efr_moist):
    """The bottom-layer moisture [N, NT, NB] the plugins saw on day d (see
    the module doc), from the dump's post-plugin moisture."""
    post = case.moist[d][..., -1].copy()
    W = case.W[d]
    if not (W > 0).any():
        return post
    cv, af = case.cv[d], case.S['af']
    r = efr_moist + case.resid_mm[:, None, None] + case.ice0[d][..., -1]
    w = cv[:, :, None] * af[:, None, :]
    live = (cv > 0)[:, :, None] & (af > 0)[:, None, :]
    u = post - r
    pos = live & (u > 0)
    S_ = (w * np.where(pos, u, 0.0)).sum((1, 2))
    pre = post.copy()
    for c in np.where(W > 0)[0]:
        if S_[c] > 1e-9:                          # A = W + S, u = u_post A / S
            A = W[c] + S_[c]
            m = pos[c]
            pre[c][m] = r[c][m] + u[c][m] * (A / S_[c])
        else:                                     # exhausted: A = W
            m = live[c] & (np.abs(u[c]) <= 1e-9)
            nw = w[c][m].sum()
            assert nw > 0, 'pumping with no pumped tile (cell %d day %d)' % (
                c, d)
            pre[c][m] = r[c][m] + W[c] / nw
    return pre


class DumpVic(Connector):
    """Window connector: runs VIC-WUR's phase group between the days it is
    given (VicConnector's day-coupled path), the day's soil, flows and Cv
    from the dump."""
    name = 'vic'
    backends = ('cpu', 'cuda')

    def __init__(self, case, device, use, group, days=30):
        super().__init__()
        self.case = case
        self.phase_group = group
        self.interval_s = days * DAY
        self.day_coupled = list(use)
        self.dev = torch.device(device)
        self.t_step = 0
        self.rec = {}
        self.local_calls, self.nr_calls = [], []
        self.hooked = False
        self.cur = 0
        c = case
        t = lambda a: torch.as_tensor(a, dtype=F64, device=self.dev)
        self.mm = t(c.S['mm'])
        self.af = t(c.S['af'])
        self.view_bf = t(c.bf)
        self.view_ro = t(c.ro)

    def time_span(self):
        return (0.0, self.case.T * DAY)

    def initialize(self, context):
        if isinstance(self.phase_group, str):
            self.phase_group = VC.get_group(self.phase_group)

    # -- views -----------------------------------------------------------
    def _t(self, a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=F64,
                               device=self.dev)

    def _efr_moist(self):
        e = self.phase_group.roles.get('efr')
        if e is None or e.moist is None:
            return np.zeros((self.case.N, self.case.NT, self.case.NB))
        return e.moist.cpu().numpy()

    def view0(self):
        """The soil at the start of the run: the parameter file's initial
        moisture (every tile and band)."""
        c = self.case
        m = np.broadcast_to(c.S['init'][:, None, None, :],
                            (c.N, c.NT, c.NB, c.NL)).copy()
        return StepView(self._t(m), self.mm, self._t(c.cv[0]), self.af,
                        ice=torch.zeros_like(self._t(m)))

    def view_post(self, d):
        c = self.case
        return StepView(self._t(c.moist[d]), self.mm, self._t(c.cv[d]),
                        self.af, ice=self._t(c.ice0[d]))

    def view_pre(self, d):
        c = self.case
        m = c.moist[d].copy()
        m[..., -1] = rebuild_bottom(c, d, self._efr_moist())
        v = StepView(self._t(m), self.mm, self._t(c.cv[d - 1 if d else 0]),
                     self.af, ice=self._t(c.ice0[d]))
        v.runoff, v.baseflow = self.view_ro[d], self.view_bf[d]
        return v

    # -- hooks that read what the connectors hold ------------------------
    def hook(self):
        """The runoff of rout_run is the C's (tag 4); the kernels' outputs
        the dump holds but the connectors do not keep are recorded."""
        if self.hooked:
            return
        self.hooked = True
        grp = self.phase_group
        r = grp.router
        orig = r.rout_run
        sweep = lambda x: r.to_sweep(self._t(x))

        def rout_run(ro, bf):
            orig(ro, bf)
            r.runoff.copy_(sweep(self.case.rout[self.cur][:, 0]))
        r.rout_run = rout_run
        self._orig = (WU.wu_local_step, NR.nonrenew_step)
        wl, nr = self._orig

        def wu_local_step(*a, **k):
            o = wl(*a, **k)
            self.local_calls.append({q: o[q].detach().clone() for q in (
                'available_gw', 'available_surf', 'available_dam')})
            return o

        def nonrenew_step(*a, **k):
            o = nr(*a, **k)
            self.nr_calls.append({q: o[q].detach().clone() for q in (
                'available_nonrenew', 'demand_nonrenew',
                'withdrawn_nonrenew')})
            return o
        WU.wu_local_step, NR.nonrenew_step = wu_local_step, nonrenew_step
        # the replay records wu_local_step's intermediates (available_*):
        # the torch path, not the CUDA level kernel that equals it bit for
        # bit (tests/test_vicwur_fused_sweep.py) without returning them
        self._fz_wu, FZ.WU = FZ.WU, False

    def unhook(self):
        if self.hooked:
            WU.wu_local_step, NR.nonrenew_step = self._orig
            FZ.WU = self._fz_wu
            self.hooked = False

    # -- one day ---------------------------------------------------------
    def between_day(self, d, t_post, pre=True):
        """P1-O of day d on the state the plugins saw, record, then (pre) F, U
        of day d + 1 on the state VIC hands on (the dump's)."""
        grp, c = self.phase_group, self.case
        self.cur = d
        self.local_calls, self.nr_calls = [], []
        v = self.view_pre(d)
        grp.between(v, t_post, None, pre=False)
        self.record(d, v)
        self.set_op_month(d)
        self.snap_residues(d)
        if pre and d + 1 < c.T:
            grp.between(self.view_post(d), None, t_post + DAY, post=False)

    def set_op_month(self, d):
        """A dam's operational month is a comparison of the climatological
        monthly inflows with their mean (dam_register.c:63-99).  Where the
        dam's river is dry those inflows are rounding residues (1e-36 m3 s-1
        left by the local water use's withdrawals) and the month is fixed by
        their last bits, which no replay reproduces; the C's month is handed
        on after the step is recorded.  The tests compare it wherever the
        inflow is not such a residue (tests/test_vicwur_replay_l2.py)."""
        c = self.case
        dc = self.phase_group.roles.get('dams')
        if dc is None:
            return
        dp = c.inp.dams
        cell = dp.dam_cell.numpy()
        pos = DM.rank_in_cell(cell)
        glb = dp.is_global.numpy().astype(bool)
        for sets, g in ((dc.gsets, True), (dc.lsets, False)):
            for s in sets.values():
                ids = s.cid.cpu().numpy()
                v = [c.dam[d, ci, pos[np.where((cell == ci) & (glb == g))[0][0]],
                           6] for ci in ids]
                s.dams.op_month.copy_(torch.as_tensor(
                    np.array(v), dtype=s.dams.op_month.dtype,
                    device=s.dams.op_month.device))

    def snap_residues(self, d):
        """The ring, discharge and stream of the router below 1e-12 m3 s-1
        are rounding residues (a river that local water use emptied keeps
        1e-21 where another summation order keeps 0), and the C branches on
        them: nonrenew.ring_return shares a return in proportion to the
        future bins' content, and equally when their sum is not above 0
        (wu_nonrenewable.c:254-305), so a residue of 7e-21 against 0 gives a
        7-bin or a 27-bin return.  Where ours and the C's are both below
        1e-12 the C's is handed on after the step is recorded (the tests
        compare at atol 1e-12); everything above is left alone."""
        grp, c = self.phase_group, self.case
        r = grp.router
        C = c.rout[d]
        lim = 1e-12
        phys = r.logical(r._bins)
        cur = r.ring[phys]                                  # [R, N] sweep
        tgt = self._t(C[:, 5:]).T[:, r.perm]
        m = (cur.abs() < lim) & (tgt.abs() < lim)
        r.ring[phys] = torch.where(m, tgt, cur)
        for name, col in (('discharge', 4), ('stream', 2)):
            x = getattr(r, name)
            t = self._t(C[:, col])[r.perm]
            x.copy_(torch.where((x.abs() < lim) & (t.abs() < lim), t, x))

    def record(self, d, v):
        grp, c = self.phase_group, self.case
        r, N = grp.router, c.N
        perm = r.perm
        R = {}

        def cells(x):
            out = torch.empty_like(x)
            out.index_copy_(0, perm, x)
            return out
        st = r.state()
        R['rout'] = torch.cat([
            torch.stack([r.to_cells(r.runoff), r.to_cells(r.inflow),
                         r.to_cells(r.stream), r.to_cells(r.deficit),
                         r.to_cells(r.discharge)], 1),
            st['discharge_dt']], 1).cpu().numpy()
        R['moist'] = v.moist[..., -1].cpu().numpy()
        roles = grp.roles
        if 'wateruse_local' in roles:
            lc = roles['wateruse_local']
            assert len(self.local_calls) == len(r._gptr) - 1
            av = {k: torch.cat([x[k] for x in self.local_calls])
                  for k in self.local_calls[0]}
            s = lc.st
            z = torch.zeros(N, 5, dtype=F64, device=self.dev)
            f = dict.fromkeys(WUF)
            f['available_gw'], f['available_surf'] = (av['available_gw'],
                                                      av['available_surf'])
            f['available_dam'] = av['available_dam']
            f['demand_gw'], f['demand_surf'] = lc.loc.dgw, lc.loc.dsf
            f['withdrawn_gw'], f['withdrawn_surf'] = lc.loc.wgw, lc.loc.wsf
            f['withdrawn_dam'] = lc.loc.wdm
            for k in ('available_tremote', 'available_remote', 'demand_tremote',
                      'demand_remote', 'withdrawn_tremote', 'withdrawn_remote',
                      'returned', 'consumed'):
                f[k] = getattr(s, k)
            f['available_remote_tmp'] = s.available_tmp
            f['demand_remote_tmp'] = s.demand_tmp
            f['withdrawn_remote_tmp'] = s.withdrawn_tmp
            nr = self.nr_calls[0] if self.nr_calls else None
            for k in ('available_nonrenew', 'demand_nonrenew',
                      'withdrawn_nonrenew'):
                f[k] = nr[k] if nr else z
            R['wu'] = torch.stack([cells(f[k].detach()) for k in WUF], 2
                                  ).cpu().numpy()
        if 'dams' in roles:
            dc = roles['dams']
            nd = len(dc.d['cell'])
            rows = [None] * nd
            cell = dc.d['cell'].astype(int)
            isg = dc.d['is_global'].astype(bool)
            for sets in (dc.gsets, dc.lsets):
                for s in sets.values():
                    dm = s.dams
                    rw = torch.cat([
                        dm.active.to(F64)[:, None], dm.inflow[:, None],
                        dm.demand[:, None], dm.efr[:, None],
                        dm.release[:, None], dm.storage[:, None],
                        dm.op_month.to(F64)[:, None],
                        dm.months_running.to(F64)[:, None],
                        dm.total_inflow[:, None], dm.total_demand[:, None],
                        dm.total_efr[:, None], dm.history_inflow,
                        dm.history_demand, dm.history_efr, dm.op_release,
                        dm.op_storage], 1).cpu().numpy()
                    cid = s.cid.cpu().numpy()
                    glb = sets is dc.gsets
                    for j, ci in enumerate(cid):
                        i = np.where((cell == ci) & (isg == glb))[0][0]
                        rows[i] = rw[j]
                    # outputs derived from the state (dam_put_data)
                    yr = DM.years_running(dm.months_running)
                    for nm, h in (('hist_inflow', dm.history_inflow),
                                  ('hist_demand', dm.history_demand),
                                  ('hist_efr', dm.history_efr)):
                        a = DM.array_average(h, yr)[:, 11].cpu().numpy()
                        for j, ci in enumerate(cid):
                            i = np.where((cell == ci) & (isg == glb))[0][0]
                            R.setdefault('dam_' + nm, {})[i] = a[j]
            R['dam'] = np.stack(rows)
        if 'efr' in roles:
            e = roles['efr']
            R['efr'] = np.stack([e._cell_moist.cpu().numpy(),
                                 e.bf.cpu().numpy(), e.q.cpu().numpy()], 1)
        self.rec[d] = R

    # -- the Connector contract ------------------------------------------
    def compute(self, inputs, context, dt):
        grp, cp = self.phase_group, self.coupler
        n = int(round(dt / DAY))
        self.hook()
        gsnap = grp.save_state()
        cp.undo.append(lambda: grp.load_state(gsnap))
        t0 = context.t
        if self.t_step == 0:
            grp.between(self.view0(), None, t0, post=False)
        else:
            self.between_day(self.t_step - 1, t0 - DAY)
        for k in range(n):
            cp.day(k, None)
            d = self.t_step + k
            if k < n - 1:
                self.between_day(d, t0 + k * DAY)
        return Proposal(connector=self.name, t0=t0, t1=t0 + dt,
                        balance=Balance(n_terms=1, scale=0.0),
                        backend=str(self.dev))

    def commit(self, proposal, context):
        self.t_step += int(round((proposal.t1 - proposal.t0) / DAY))

    def save_state(self):
        return dict(t_step=self.t_step)

    def load_state(self, blob):
        self.t_step = blob['t_step']

    def finalize(self, context):
        if self.t_step > 0:
            self.between_day(self.t_step - 1, (self.t_step - 1) * DAY,
                             pre=False)
        self.unhook()


# ----------------------------------------------------------------- driver
def connector_cfgs(case, group):
    """The module configs of the connectors a run has switched on."""
    c, inp, T = case.c, case.inp, case.T
    cfg = {}
    cfg['vicwur_routing'] = dict(
        params=inp.routing, area=case.grid.area_active,
        rout_steps_per_day=c['rout'], nonrenew_runoff=c['nonren_runoff'],
        group=group)
    if c['dams']:
        cfg['vicwur_dams'] = dict(
            params=inp.dams, start_date='%04d-%02d-%02d' % START[:3],
            nsteps=T, exact=True, group=group)
    if c['wu']:
        f = case.sector_series()
        cfg['vicwur_wateruse_local'] = dict(
            demand=dict(series=f['demand']), gw_frac=dict(series=f['gw']),
            cons_frac=dict(series=f['cons']),
            active=[bool(x) for x in case.active],
            soil=dict(resid_mm=case.resid_mm), group=group)
        if c['remote']:
            cfg['vicwur_wateruse_remote'] = dict(params=inp.remote,
                                                 group=group)
        if c['nonren']:
            cfg['vicwur_nonrenew'] = dict(
                limit=c['nren_lim'], group=group)
    if c['efr']:
        cfg['vicwur_efr'] = dict(case.efr_cfg(), exact=True, group=group)
    return cfg


def replay(case, device='cpu', days=30):
    """Run the case through the runtime; returns the per-day records
    {d: {...}} and the model."""
    group = 'tv23_%s_%s_%d' % (case.run, device, os.getpid())
    cfgs = connector_cfgs(case, group)
    win = DumpVic(case, device, list(cfgs), group, days=days)
    nm = 'tv23_window_%s_%s' % (case.run, device)
    registry.register(nm, lambda **c: win, 'test', replace=True)
    mods = {'vic': dict(connector=nm)}
    for k, v in cfgs.items():
        mods[k] = dict(connector=k, **v)
    cfg = dict(run=dict(device=device, log=False), modules=mods,
               connections=[], ledger=dict(enabled=False))
    m = TorchVIC.from_dict(cfg)
    try:
        with torch.inference_mode():
            m.initialize().run()
            m.finalize(write=False)
    finally:
        win.unhook()
    return win.rec, m
