# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The connectors vicwur_routing, vicwur_dams, vicwur_wateruse_local,
vicwur_wateruse_remote, vicwur_nonrenew and vicwur_efr through the coupling
runtime.

A small R1-shaped network (6 cells: chain c0-c1-c2-c3, tributary c4-c2,
delta receiver c5 of source c3; two global dams, one local dam) is run for 60
days from 2003-12-15 through the runtime. VIC is replaced by a window
connector that supplies a deterministic soil, runoff and baseflow each day
and runs VIC-WUR's phase group between the days exactly as VicConnector
does (VicConnector's runoff/baseflow for the view are tested in
tests/test_vicwur_connectors_b.py).

  * registration and config validation;
  * the direct kernel replay: the same days, the same kernels called serially
    cell by cell in the C order, without connectors, group or runtime, must
    give bit-for-bit the same ring, deficit, dams, remote totals, soil;
  * restart equals continuous, bitwise (checkpoint through the runtime);
  * the ledger of every module and of the coupled domain closes, and the
    modules' exchange terms cancel (ring, dams, deficit paybacks);
  * the same on a GPU (marked gpu).
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from torchvic.coupling import TorchVIC, Connector, Balance, Proposal   # noqa
from torchvic.coupling import registry                                 # noqa
from torchvic.coupling.spec import CouplingError                       # noqa
from torchvic.coupling.connectors import vicwur_common as VC           # noqa
from torchvic.model.routing_cascade import CascadeRouter, _div         # noqa
from torchvic.vicwur import dams as DM                                 # noqa
from torchvic.vicwur import efr as EF                                  # noqa
from torchvic.vicwur import nonrenew as NR                             # noqa
from torchvic.vicwur import wateruse as WU                             # noqa
from torchvic.vicwur import wateruse_remote as WR                      # noqa
from torchvic.vicwur.forcing import step_dates                         # noqa
from torchvic.vicwur.view import StepView                              # noqa

NAMES = ('vicwur_routing', 'vicwur_dams', 'vicwur_wateruse_local',
         'vicwur_wateruse_remote', 'vicwur_nonrenew', 'vicwur_efr')
N, V, B, NL = 6, 2, 1, 3
DOWN = [1, 2, 3, 3, 2, 5]
EDGES = {'src': [3], 'dst': [5]}
AREA = 2e8
NSUB = 2
DAY = 86400.0
START = '2003-12-15'
NDAYS = 60
WINDOW = 5
F64 = torch.float64


def uh():
    u = np.array([0.4, 0.3, 0.2, 0.1])
    return np.repeat(u[:, None], N, 1)


class Sim(object):
    """Deterministic stand-in for VIC: soil, runoff and baseflow of day d."""

    def __init__(self, device):
        self.dev = torch.device(device)
        self.max_moist = torch.full((N, NL), 200.0, dtype=F64,
                                    device=self.dev)
        self.cv = torch.tensor([[0.6, 0.4]] * N, dtype=F64, device=self.dev)
        self.af = torch.ones(N, B, dtype=F64, device=self.dev)
        g = torch.Generator().manual_seed(7)
        self.moist0 = (60 + 80 * torch.rand(N, V, B, NL, generator=g,
                                            dtype=F64)).to(self.dev)

    def day(self, d, moist):
        g = torch.Generator().manual_seed(1000 + d)
        r = lambda *s: torch.rand(*s, generator=g, dtype=F64).to(self.dev)
        new = (moist * 0.97 + 3.0 * r(N, V, B, NL)).clamp(max=190.0)
        return new, 4.0 * r(N), 1.5 * r(N)

    def view(self, moist, runoff, baseflow):
        v = StepView(moist, self.max_moist, self.cv, self.af,
                     ice=0.1 * moist)
        v.runoff, v.baseflow = runoff, baseflow
        return v


class Window(Connector):
    """A window connector that runs the VIC-WUR phase group like
    VicConnector's day-coupled path (vic.py _compute_day_coupled)."""
    name = 'vic'
    backends = ('cpu', 'cuda')

    def __init__(self, device, use, group='tv21a', days=WINDOW,
                 ndays=NDAYS):
        super().__init__()
        self.phase_group = group
        self.interval_s = days * DAY
        self.ndays = ndays
        self.sim = Sim(device)
        self.day_coupled = list(use)
        self.moist = self.runoff = self.baseflow = None
        self.t_step = 0

    def time_span(self):
        return (0.0, self.ndays * DAY)

    def initialize(self, context):
        if isinstance(self.phase_group, str):
            self.phase_group = VC.get_group(self.phase_group)
        self.moist = self.sim.moist0.clone()

    def compute(self, inputs, context, dt):
        grp, cp, sim = self.phase_group, self.coupler, self.sim
        n = int(round(dt / DAY))
        gsnap = grp.save_state()
        cp.undo.append(lambda: grp.load_state(gsnap))
        moist, ro, bf = self.moist.clone(), self.runoff, self.baseflow
        t0 = context.t
        if self.t_step == 0:
            z = torch.zeros(N, dtype=F64, device=sim.dev)
            grp.between(sim.view(moist, z, z), None, t0, post=False)
        else:
            delta = grp.between(sim.view(moist, ro, bf), t0 - DAY, t0)
            moist = moist + delta
        for k in range(n):
            cp.day(k, None)
            d = self.t_step + k
            moist, ro, bf = sim.day(d, moist)
            if k < n - 1:
                delta = grp.between(sim.view(moist, ro, bf), t0 + k * DAY,
                                    t0 + (k + 1) * DAY)
                moist = moist + delta
        self.stage = (moist, ro, bf)
        return Proposal(connector=self.name, t0=t0, t1=t0 + dt,
                        balance=Balance(n_terms=1, scale=0.0),
                        backend=str(sim.dev))

    def commit(self, proposal, context):
        self.moist, self.runoff, self.baseflow = self.stage
        self.t_step += int(round((proposal.t1 - proposal.t0) / DAY))

    def save_state(self):
        return dict(moist=self.moist.clone(), t_step=self.t_step,
                    ro=None if self.runoff is None else self.runoff.clone(),
                    bf=None if self.baseflow is None
                    else self.baseflow.clone())

    def load_state(self, blob):
        self.moist = blob['moist'].to(self.sim.dev)
        self.t_step = blob['t_step']
        self.runoff = None if blob['ro'] is None \
            else blob['ro'].to(self.sim.dev)
        self.baseflow = None if blob['bf'] is None \
            else blob['bf'].to(self.sim.dev)

    def finalize(self, context):
        # C runs P1-O of the last day too (its soil edits have no step left)
        if self.t_step > 0:
            self.phase_group.between(
                self.sim.view(self.moist, self.runoff, self.baseflow),
                (self.t_step - 1) * DAY, None, pre=False)


# ------------------------------------------------------------------ config
def forcing_arrays(nd=NDAYS + 3):
    g = np.random.default_rng(3)
    dem = np.zeros((nd, N, 5))
    dem[:, :, :4] = g.uniform(0.0, 2.5, (nd, N, 4))
    return dict(demand=dem, gw_frac=np.full((nd, N, 5), 0.3),
                cons_frac=np.full((nd, N, 5), 0.4))


def module_cfgs(use=NAMES, group='tv21a'):
    f = forcing_arrays()
    cfg = {}
    cfg['vicwur_routing'] = dict(
        downstream=DOWN, uh_inflow=uh().tolist(), uh_runoff=uh().tolist(),
        area=[AREA] * N, rout_steps_per_day=NSUB, group=group)
    cfg['vicwur_dams'] = dict(
        dams=dict(cell=[1, 2, 4], capacity=[20.0, 30.0, 6.0],
                  inflow_frac=[0.8, 0.9, 0.3], year=[2000, 2000, 2003],
                  is_global=[True, True, False],
                  service=[[0, 1], [2, 3], [-1, -1]],
                  service_frac=[[0.5, 0.5], [0.7, 0.3], [0.0, 0.0]]),
        start_date=START, nsteps=NDAYS, exact=True, group=group)
    cfg['vicwur_wateruse_local'] = dict(
        demand={'series': f['demand'].tolist()},
        gw_frac={'series': f['gw_frac'].tolist()},
        cons_frac={'series': f['cons_frac'].tolist()},
        soil=dict(resid_mm=[30.0] * N), group=group)
    cfg['vicwur_wateruse_remote'] = dict(edges=EDGES, group=group)
    cfg['vicwur_nonrenew'] = dict(group=group)
    cfg['vicwur_efr'] = dict(
        discharge=0.5, baseflow=0.2,
        soil=dict(resid=[0.05] * N, depth=[0.5] * N, max_moist=[200.0] * N,
                  dsmax=[10.0] * N, ds=[0.01] * N, ws=[0.8] * N,
                  c=[2.0] * N), exact=True, group=group)
    return {k: v for k, v in cfg.items() if k in use}


def build(device, use=NAMES, group='tv21a', ndays=NDAYS, restart=None,
          ledger=True):
    win = Window(device, use, group, ndays=ndays)
    registry.register('tv21a_window', lambda **c: win, 'test',
                      replace=True)
    mods = {'vic': dict(connector='tv21a_window')}
    for k, c in module_cfgs(use, group).items():
        mods[k] = dict(connector=k, **c)
    cfg = dict(run=dict(device=device, log=False), modules=mods,
               connections=[])
    if restart:
        cfg['checkpoint'] = dict(restart_from=restart)
    return TorchVIC.from_dict(cfg), win


def run(device, **kw):
    m, win = build(device, **kw)
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    return m, win


def states(m):
    return {k: c.save_state() for k, c in m.runtime.connectors.items()}


def same(a, b, path=''):
    if torch.is_tensor(a):
        return [path] if not (torch.is_tensor(b) and torch.equal(
            a.cpu(), b.cpu())) else []
    if isinstance(a, dict):
        out = []
        for k, v in a.items():
            out += same(v, b[k], path + '/' + str(k))
        return out
    if isinstance(a, (list, tuple)):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += same(x, y, path + '/%d' % i)
        return out
    return [] if a == b else [path]


# ----------------------------------------------------------- registration
def test_registration():
    av = registry.available()
    for n in NAMES:
        assert n in av
        assert registry.get(n).status == 'available'
    # the lumped connectors are still there
    assert 'irrigation' in av and 'reservoir' in av


def test_connectors_declare_a_daily_interval_and_checkpoint():
    for n, c in module_cfgs().items():
        conn = registry.get(n).make(**c)
        i = conn.info()
        assert i.time.interval_s == DAY and i.checkpoint == 'full'
        assert 'cuda' in i.backends


# ------------------------------------------------------ config validation
def _bad(name, **over):
    c = module_cfgs()[name]
    c = dict(c, group=VC.WurGroup('x'), **over)
    with pytest.raises(CouplingError):
        registry.get(name).make(**c)


def test_config_validation():
    _bad('vicwur_routing', area=[1.0] * (N - 1))
    _bad('vicwur_routing', rout_steps_per_day=7)
    _bad('vicwur_routing', downstream=[1, 2, 3, 3, 2, 9])
    _bad('vicwur_routing', bogus=1)
    _bad('vicwur_dams', alpha=1.0)
    _bad('vicwur_dams', dams=dict(cell=[1, 1], capacity=[1.0, 1.0],
                                  inflow_frac=[.5, .5], year=[2000, 2000],
                                  is_global=[True, True]))
    _bad('vicwur_dams', start_date='15/12/2003')
    _bad('vicwur_wateruse_remote', edges=dict(src=[3], dst=[3]))
    _bad('vicwur_wateruse_remote', edges=dict(src=[3, 3], dst=[5, 5]))
    _bad('vicwur_nonrenew', limit=-1.0)
    _bad('vicwur_efr', soil=dict(resid=[0.1]))
    _bad('vicwur_wateruse_local', soil=dict())
    with pytest.raises(CouplingError):
        VC.Series({'constant': 1, 'series': 2})


def test_missing_peer_is_refused():
    g = VC.WurGroup('y')
    c = registry.get('vicwur_wateruse_remote').make(edges=EDGES, group=g)
    with pytest.raises(CouplingError, match='routing'):
        g.build()
    del c


def test_a_group_joins_each_role_once():
    g = VC.WurGroup('z')
    registry.get('vicwur_efr').make(group=g, **{
        k: v for k, v in module_cfgs()['vicwur_efr'].items()
        if k != 'group'})
    with pytest.raises(CouplingError, match='already'):
        registry.get('vicwur_efr').make(group=g, **{
            k: v for k, v in module_cfgs()['vicwur_efr'].items()
            if k != 'group'})


def test_plan_follows_the_connectors_present():
    full = VC.WurGroup('p1')
    for n, c in module_cfgs(group=full).items():
        registry.get(n).make(**c)
    hooks = [(c.phase, c.kind, c.hook) for c in full.plan]
    p2 = [h[2] for h in hooks if h[0] == 'P2']
    assert p2 == ['sweep:local_dam', 'sweep:route', 'sweep:wu_local',
                  'sweep:global_dam', 'sweep:wu_remote']
    assert ('P3', 'wateruse', 'pass3_nonrenew') in hooks
    part = VC.WurGroup('p2')
    for n in ('vicwur_routing', 'vicwur_efr'):
        registry.get(n).make(**module_cfgs(group=part)[n])
    assert [h[2] for h in [(c.phase, c.kind, c.hook) for c in part.plan]
            if h[0] == 'P2'] == ['sweep:route']


# ------------------------------------------------------------ the replay
def replay(device):
    """The same days through the kernels, serially cell by cell (the C
    order), with no connector, group or runtime."""
    dev = torch.device(device)
    sim = Sim(device)
    cfg = module_cfgs()
    f = forcing_arrays()
    ser = lambda k: torch.as_tensor(f[k], dtype=F64, device=dev)
    rcv = [[] for _ in range(N)]
    rcv[3].append(5)
    r = CascadeRouter(DOWN, uh(), uh(), np.full(N, AREA), DAY, NSUB,
                      receiving=rcv, device=dev)
    perm = r.perm
    ys, ms, _ = step_dates((2003, 12, 15, 0), NDAYS + 3, 1,
                           'proleptic_gregorian')
    cal = DM.make_calendar(ys, ms, 'proleptic_gregorian', 1, dev)
    # one kernel set per dam, in cell order (C serves one cell at a time)
    mk = lambda cap, fr, yr: DM.GlobalDams([cap], [fr], [yr], 0.85, 0.6, 5.0,
                                           1, True, dev)
    gd = {1: mk(20.0, 0.8, 2000), 2: mk(30.0, 0.9, 2000)}
    ld = {4: DM.LocalDams([6.0], [0.3], [2003], 0.85, 0.6, 5.0, 1, True,
                          dev)}
    svc = {1: ([0, 1], [0.5, 0.5]), 2: ([2, 3], [0.7, 0.3])}
    soil = {k: torch.full((N,), v, dtype=F64, device=dev) for k, v in dict(
        resid=0.05, depth=0.5, max_moist=200.0, dsmax=10.0, ds=0.01,
        ws=0.8, c=2.0).items()}
    resid_mm = torch.full((N,), 30.0, dtype=F64, device=dev)
    act = torch.tensor([True] * 4 + [False], device=dev).expand(N, 5)
    st = WR.RemoteState(N, dev)
    L = WR.LocalUse(*[torch.zeros(N, 5, dtype=F64, device=dev)
                      for _ in range(5)],
                    torch.zeros(N, 5, dtype=F64, device=dev), act)
    plan = WR.plan_for_router(r, [3], [5])
    slot = WR.RemoteSlot(plan, r, st, L)
    sweep = lambda x: x.index_select(0, perm)
    moist = sim.moist0.clone()
    delta = None
    efr = {}
    ones = lambda m: torch.ones_like(m[:, 0, 0]).unsqueeze(1)

    def F(d, view):
        t = min(d, NDAYS + 2)
        m = view.moist[..., -1]
        ice = view.ice[..., -1].unsqueeze(-1)
        s = EF.efr_step(m, ice, ones(m), soil['resid'], soil['depth'],
                        soil['max_moist'], soil['dsmax'], soil['ds'],
                        soil['ws'], soil['c'], view.cv, view.af,
                        torch.full((N,), 0.2, dtype=F64, device=dev),
                        torch.full((N,), 0.5, dtype=F64, device=dev),
                        steps_per_day=1, bug=True)
        efr['moist'], efr['q'] = s.efr_moist, s.discharge
        L.cons_frac = sweep(ser('cons_frac')[t])
        L.dem, L.gfr = sweep(ser('demand')[t]), sweep(ser('gw_frac')[t])
        return t

    def P(d, view):
        st_ = cal.at(min(d, NDAYS + 2))
        t = min(d, NDAYS + 2)
        # wu_run
        st.reset()
        st.returned = torch.zeros(N, 5, dtype=F64, device=dev)
        st.consumed = torch.zeros(N, 5, dtype=F64, device=dev)
        L.dgw, L.dsf = WU.demand(L.dem, L.gfr, act, False)
        L.wgw = torch.zeros(N, 5, dtype=F64, device=dev)
        L.wsf, L.wdm = torch.zeros_like(L.wgw), torch.zeros_like(L.wgw)
        # rout_run + begin
        r.rout_run(view.runoff, view.baseflow)
        r.begin()
        qs = r.to_sweep(efr['q'])
        slot.efr_q = qs
        slot.begin()
        for g in range(len(r._gptr) - 1):
            gc = r.group_cells(g).tolist()
            for c in gc:                                   # local dams
                cells = torch.tensor([c], device=dev)
                ci = perm.index_select(0, cells)
                if int(ci) in ld:
                    ld[int(ci)].run(st_, view.runoff.index_select(0, ci),
                                    r.area.index_select(0, cells))
            r.route_group(g)                               # the level
            for c in gc:
                cells = torch.tensor([c], device=dev)
                ci = perm.index_select(0, cells)
                cid = int(ci)
                # local water use
                row = lambda x: x.index_select(0, cells)
                moist_ = view.layer_at(-1, ci)
                ice_ = view.ice_at(-1, ci).unsqueeze(-1)
                af, cv = view.af.index_select(0, ci), \
                    view.cv.index_select(0, ci)
                area = row(r.area)
                kw = {}
                if cid in ld:
                    d_ = ld[cid]
                    kw = dict(dam_storage=d_.storage[:, None],
                              dam_active=d_.active[:, None],
                              dam_local=torch.ones(1, 1, dtype=torch.bool,
                                                   device=dev))
                kw.update(efr_moist=efr['moist'].index_select(0, ci),
                          efr_q=efr['q'].index_select(0, ci),
                          discharge=row(r.discharge))
                ring = r.ring_logical().index_select(1, cells).T
                out = WU.wu_local_step(
                    row(L.dem), row(L.gfr), row(L.cons_frac), moist_, ice_,
                    ones(moist_), resid_mm.index_select(0, ci), af, cv, ring,
                    DAY, area, active=row(act), deficit=row(r.deficit),
                    **kw)
                for k, tgt in (('demand_gw', L.dgw), ('demand_surf', L.dsf),
                               ('withdrawn_gw', L.wgw),
                               ('withdrawn_surf', L.wsf),
                               ('withdrawn_dam', L.wdm)):
                    tgt.index_copy_(0, cells, out[k])
                st.returned.index_copy_(0, cells, out['returned'])
                st.consumed.index_copy_(0, cells, out['consumed'])
                r.deficit.index_copy_(0, cells, out['deficit'])
                dd, _ = WU.pump_depth(out['av_gw'], out['A_gw'],
                                      out['cell_gw'], af, cv)
                view.withdraw_cells(-1, ci, dd, floor='zero')
                ws, rr = out['cell_surf'], out['returned_to_river']
                gate = (ws > 0) | (rr > 0)
                qw = _div(_div(ws, 1000.) * area, DAY)
                qr = _div(_div(rr, 1000.) * area, DAY)
                d0, s0 = row(r.discharge), row(r.stream)
                r.scale_prop(cells, withdraw=qw, ret=qr, lo=0)
                r.recompute(cells)
                r.discharge.index_copy_(0, cells, torch.where(
                    gate, row(r.discharge), d0))
                r.stream.index_copy_(0, cells, torch.where(
                    gate, row(r.stream), s0))
                if cid in ld:
                    ld[cid].storage = out['dam_storage'][:, 0]
                if cid in gd:                              # global dam
                    sv, fr = svc[cid]
                    sp = r.pos[torch.tensor(sv, device=dev)]
                    dem = DM.service_demand(
                        L.dsf, act, sp[None], torch.tensor(
                            [fr], dtype=F64, device=dev),
                        torch.ones(1, 2, dtype=torch.bool, device=dev),
                        row(r.area))
                    phys = r.logical(torch.arange(NSUB, device=dev))
                    rg = r.ring[phys].index_select(1, cells).T.contiguous()
                    g_ = gd[cid]
                    rg2, dis2 = g_.run(st_, rg, row(r.discharge), DAY, dem,
                                       efr['q'].index_select(0, ci))
                    r.ring[phys[:, None], cells[None, :]] = rg2.T
                    r.discharge.index_copy_(0, cells, dis2)
            slot(r, r.group_cells(g))
        # wu_nonrenew
        allc = torch.arange(N, device=dev)
        out = NR.nonrenew_step(
            L.dgw, L.dsf, L.wgw, L.wsf, L.wdm, L.cons_frac, r.deficit, None,
            act, st.withdrawn_tremote, st.returned, st.consumed, False,
            r.ring_logical().T, r.area, DAY, r.nsub, r.L, r.discharge,
            r.stream)
        r.ring[r.logical(r._bins)] = out['dt_discharge'].T
        r.discharge.copy_(out['discharge'])
        r.stream.copy_(out['stream'])
        r.deficit.copy_(out['deficit'])
        st.returned, st.consumed = out['returned'], out['consumed']

    z = torch.zeros(N, dtype=F64, device=dev)
    F(0, sim.view(moist, z, z))
    for d in range(NDAYS):
        moist, ro, bf = sim.day(d, moist)
        v = sim.view(moist, ro, bf)
        P(d, v)
        if d < NDAYS - 1:
            F(d + 1, v)
            moist = moist + v.commit()
    return dict(r=r, st=st, L=L, gd=gd, ld=ld, efr=efr, moist=moist)


def _replay_vs_runtime(device):
    m, win = run(device)
    ref = replay(device)
    c = m.runtime.connectors
    r = c['vicwur_routing'].router
    bad = []
    chk = lambda name, a, b: bad.append(name) if not torch.equal(
        a.cpu(), b.cpu()) else None
    chk('ring', r.ring, ref['r'].ring)
    chk('head', r.head, ref['r'].head)
    chk('discharge', r.discharge, ref['r'].discharge)
    chk('stream', r.stream, ref['r'].stream)
    chk('deficit', r.deficit, ref['r'].deficit)
    st = c['vicwur_wateruse_local'].st
    for k in WR.RemoteState.TOTALS + WR.RemoteState.TMPS:
        chk('st.' + k, getattr(st, k), getattr(ref['st'], k))
    for cell, d_ in ref['gd'].items():
        s = [s for s in c['vicwur_dams'].gsets.values()
             if int(s.cid[0]) == cell][0]
        for k, v in d_.state_dict().items():
            chk('gdam%d.%s' % (cell, k), s.dams.state_dict()[k], v)
    s = list(c['vicwur_dams'].lsets.values())[0]
    for k, v in ref['ld'][4].state_dict().items():
        chk('ldam.' + k, s.dams.state_dict()[k], v)
    chk('efr.moist', c['vicwur_efr'].moist, ref['efr']['moist'])
    chk('soil', win.moist, ref['moist'])
    assert not bad, bad
    return m


def test_runtime_equals_the_direct_kernel_replay():
    m = _replay_vs_runtime('cpu')
    # the run did something: water was withdrawn, a deficit built up, the
    # local dam became active and the global dams planned
    c = m.runtime.connectors
    assert float(c['vicwur_routing'].router.deficit.max()) > 0
    ld = list(c['vicwur_dams'].lsets.values())[0].dams
    assert bool(ld.active.all())
    assert float(c['vicwur_wateruse_local'].loc.wsf.sum()) > 0


# ------------------------------------------------------------------ ledger
def _rows(m, mod):
    return [r for r in m.ledger.balances if r['module'] == mod]


def check_ledger(m):
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    for n in NAMES:
        assert len(_rows(m, n)) == NDAYS          # a row per day
    # the exchange terms cancel between the modules, day by day
    for i in range(1, NDAYS):
        g = lambda mod, k: _rows(m, mod)[i].get('detail_' + k, 0.0)
        rt, lo, rm = ('vicwur_routing', 'vicwur_wateruse_local',
                      'vicwur_wateruse_remote')
        net_ring = g(rt, 'ring_in') - g(rt, 'ring_out')
        expect = (g(lo, 'ring_returned') - g(lo, 'ring_withdrawn')
                  + g(rm, 'ring_returned') - g(rm, 'ring_withdrawn')
                  + g('vicwur_nonrenew', 'ring_return')
                  + g('vicwur_dams', 'global_release')
                  - g('vicwur_dams', 'global_inflow'))
        scale = abs(g(rt, 'ring_in')) + abs(g(rt, 'ring_out')) + 1.0
        assert abs(net_ring - expect) <= 1e-9 * scale, (i, net_ring, expect)
        # the paybacks into the deficit are what the non-renewable module
        # books as inflow
        pb = (g(rt, 'payback') + g(lo, 'payback') + g(rm, 'payback'))
        assert abs(pb - g('vicwur_nonrenew', 'payback')) <= 1e-6 * (
            abs(pb) + 1.0), (i, pb)


def test_ledger_closes_and_exchanges_cancel():
    m, _ = run('cpu')
    check_ledger(m)
    tot = m.ledger.totals('vicwur_wateruse_local')
    assert tot['I_ext'] > 0 and tot['E_other'] > 0


def test_a_broken_ring_edit_opens_the_ledger():
    # take water out of the ring behind the ledger's back: the router's
    # balance must not close
    from torchvic.coupling.runtime import StepRejected
    m, win = build('cpu')
    m.initialize()
    c = m.runtime.connectors['vicwur_routing']
    orig = c.sweep_slot

    def bad(lv, slot, cells):
        orig(lv, slot, cells)
        if slot == 'route' and lv == 1:
            c.router.ring.mul_(0.5)
    c.sweep_slot = bad
    with pytest.raises(StepRejected, match='water balance'):
        with torch.inference_mode():
            m.run()


# ----------------------------------------------------------------- restart
def test_restart_equals_continuous(tmp_path):
    full, wf = run('cpu')
    ck = str(tmp_path / 'ck.pt')
    m1, w1 = build('cpu')
    with torch.inference_mode():
        m1.initialize().run(until=30 * DAY)
        m1.save_checkpoint(ck)
    VC._GROUPS.clear()
    m2, w2 = build('cpu', restart=ck)
    with torch.inference_mode():
        m2.initialize().run()
        m2.finalize(write=False)
    assert not same(states(full), states(m2))
    assert not same(wf.save_state(), w2.save_state())
    for n in NAMES:
        a = [r for r in full.ledger.balances if r['module'] == n]
        b = [r for r in m2.ledger.balances if r['module'] == n]
        assert [r['residual'] for r in a] == [r['residual'] for r in b]
    check_ledger(m2)


def test_save_load_state_roundtrip():
    m, _ = run('cpu')
    a = states(m)
    with torch.inference_mode():
        for c in m.runtime.connectors.values():
            c.load_state(c.save_state())
    assert not same(a, states(m))


def test_rollback_of_a_rejected_window_restores_the_group():
    # a window that fails in its last day leaves every connector where the
    # window started
    m, win = build('cpu')
    m.initialize()
    with torch.inference_mode():
        m.run(until=10 * DAY)
    before = states(m)
    c = m.runtime.connectors['vicwur_nonrenew']
    orig = c.pass3_nonrenew
    n = [0]

    def boom():
        n[0] += 1
        if n[0] == 3:
            raise RuntimeError('boom')
        orig()
    c.pass3_nonrenew = boom
    with pytest.raises(Exception):
        with torch.inference_mode():
            m.step()
    assert not same(before, states(m))


# --------------------------------------------------------------------- GPU
def _close(a, b, path=''):
    """Every tensor of two nested states equal to rounding (CUDA reduces
    batches in another order than the CPU, so not bitwise)."""
    bad = []
    if torch.is_tensor(a):
        a, b = a.cpu(), b.cpu()
        if a.is_floating_point():
            ok = torch.allclose(a, b, rtol=1e-9, atol=1e-9)
        else:
            ok = torch.equal(a, b)
        return [] if ok else [path]
    if isinstance(a, dict):
        for k, v in a.items():
            bad += _close(v, b[k], path + '/' + str(k))
    return bad


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_matches_the_cpu_and_the_ledger_closes():
    mc, _ = run('cpu')
    mg, _ = run('cuda')
    assert not _close(states(mc), states(mg))
    check_ledger(mg)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_restart_equals_continuous(tmp_path):
    full, wf = run('cuda')
    ck = str(tmp_path / 'ck.pt')
    m1, w1 = build('cuda')
    with torch.inference_mode():
        m1.initialize().run(until=30 * DAY)
        m1.save_checkpoint(ck)
    VC._GROUPS.clear()
    m2, w2 = build('cuda', restart=ck)
    with torch.inference_mode():
        m2.initialize().run()
        m2.finalize(write=False)
    assert not same(states(full), states(m2))
    assert not same(wf.save_state(), w2.save_state())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_hooks_do_no_host_sync():
    m, win = build('cuda')
    m.initialize()
    with torch.inference_mode():
        m.run(until=5 * DAY)          # builds, warms the kernels
    grp = win.phase_group
    sim = win.sim
    moist, ro, bf = sim.day(100, win.moist)
    v = sim.view(moist, ro, bf)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        with torch.inference_mode():
            grp._set_view(v)
            grp._set_time(5 * DAY)
            grp._phases(('P1', 'P2', 'P3', 'O'))
            grp._set_time(6 * DAY)
            grp._phases(('F', 'U'))
    finally:
        torch.cuda.set_sync_debug_mode('default')


def test_series_forms():
    class Obj(object):
        def get(self, step):
            return [float(step)] * 3
    s = VC.Series({'forcing': Obj()}, 'f').to('cpu')
    assert s.get(4).tolist() == [4.0, 4.0, 4.0]
    s = VC.Series({'series': [[1.0], [2.0]]}).to('cpu')
    assert s.get(0).tolist() == [1.0] and s.get(9).tolist() == [2.0]
    assert VC.Series(3.0).to('cpu').get(5).item() == 3.0
