"""The river-system connectors (routing, reservoirs, quality) in the coupled
runtime: discharge equals the standalone engine bit for bit, the ledgers
close, a rejected step leaves no trace, a restart equals the continuous run.

Part 1 uses a synthetic runoff source (no VIC) on the random river network
of tests/test_river_network.py; part 2 couples VIC (Stehekin) to the router.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from torchvic.coupling import (TorchVIC, Connector, Balance, Proposal,   # noqa: E402
                               StepRejected, VarSpec, registry)
from torchvic.coupling.spec import GridSpec                                # noqa: E402
from torchvic.coupling.connectors import (RoutingConnector,                # noqa: E402
                                          ReservoirConnector,
                                          QualityConnector)
from torchvic.model import river_network as rn                             # noqa: E402
from test_river_network import make_net                                    # noqa: E402

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
DAY = 86400.0
GRID = 'vic_cells'


@pytest.fixture(autouse=True)
def _on_cpu():
    was = torch.empty(0).device
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(was)


class Source(Connector):
    """Land surface stand-in: daily runoff (mm) from a fixed table; it
    closes its own balance (P = Q_out)."""
    name = 'src'
    backends = ('cpu', 'cuda')
    interval_s = DAY
    exported_fluxes = ('total_runoff',)
    outputs = (VarSpec('total_runoff', 'mm', ('cell',), kind='flux',
                       grid=GRID),)

    def __init__(self, table, area, interval_s=DAY):
        super().__init__()
        self.table = table                  # [days, cell] mm
        self.area = area
        self.interval_s = interval_s
        self.k = 0
        self.wanted = set()

    def grids(self):
        return [GridSpec(GRID, self.table.shape[1],
                         area=torch.as_tensor(self.area))]

    def initialize(self, context):
        self.dev = context.device

    def time_span(self):
        return (0.0, self.table.shape[0] * DAY)

    def compute(self, inputs, context, dt):
        n = int(round(dt / DAY))
        k0 = int(round(context.t / DAY))
        x = self.table[k0:k0 + n].sum(0).to(self.dev)
        v = (x * torch.as_tensor(self.area, dtype=x.dtype,
                                 device=self.dev)).sum() * 1e-3
        return Proposal(self.name, context.t, context.t + dt, state=k0 + n,
                        exports=dict(total_runoff=x),
                        balance=Balance(P=v, Q_out=v, n_terms=x.numel(),
                                        scale=2 * v.abs()))

    def commit(self, proposal, context):
        self.k = proposal.state

    def save_state(self):
        return dict(k=self.k)

    def load_state(self, blob):
        self.k = blob['k']


def network(seed=0, nt=40, device='cpu'):
    net, cells = make_net()
    gen = torch.Generator().manual_seed(seed)
    area = (torch.rand(net.n, generator=gen, dtype=torch.float64) * 1e7
            + 2e7).numpy()
    table = torch.rand(nt, net.n, generator=gen, dtype=torch.float64) * 5
    gauges = np.unique([int(net.order[-1]), int(net.order[-40]),
                        int(net.order[len(net.order) // 2])])
    return net, gauges, area, table


def run_river(device, net, gauges, area, table, routing_kw=None, modules=None,
              connections=None, extra=None, steps_check=True, **rt):
    nt = table.shape[0]
    src = Source(table, area)
    rk = dict(net=net, gauge_cells=gauges, area=area, cell_uh=[0.3, 0.5, 0.2])
    rk.update(routing_kw or {})
    mods = dict(src=dict(), routing=dict(rk))
    mods.update(modules or {})
    conns = [dict({'from': 'src.total_runoff', 'to': 'routing.inflow'})]
    conns += list(connections or [])
    conn_objs = dict(src=src)
    conn_objs.update(extra or {})
    m = TorchVIC.from_dict(
        dict(run=dict(device=device, log=False), modules={
            k: (dict(connector=('routing' if k == 'routing' else 'two_store'
                                if k == 'src' else v.pop('connector', k)),
                     **v)) for k, v in mods.items()},
            connections=conns, **rt), connectors=conn_objs)
    return m


def standalone(device, net, gauges, area, table, **kw):
    r, info = rn.build_step_router(
        net, gauges, area, DAY, cell_uh=torch.tensor([0.3, 0.5, 0.2],
                                                     dtype=torch.float64,
                                                     device=device),
        device=device, **kw)
    if device == 'cuda':
        from torchvic.model import routing_fast
        r = routing_fast.FastStepRouter.from_router(r)
    return r, info


@pytest.mark.parametrize('device', DEVICES)
def test_routing_connector_equals_standalone_and_ledger_closes(device):
    net, gauges, area, table = network()
    m = run_river(device, net, gauges, area, table)
    m.initialize().run()
    rc = m.connector('routing')
    got = m.outputs('routing')['discharge']
    ref, _ = standalone(device, net, gauges, area, table)
    want = torch.stack([ref.step(table[t].to(device))[0]
                        for t in range(table.shape[0])])
    assert torch.equal(got.cpu(), want.cpu())
    assert torch.equal(rc.router.ring.cpu(), ref.ring.cpu())
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    tot = m.ledger.totals('routing')
    assert abs(tot['residual']) <= 1e-9 * abs(tot['I_ext'])
    assert float(rc.ledger().abs().max()) < 1e-9 * float(
        rc.router.in_runoff.max())


class User(Connector):
    """A water user at two river sites: asks for `want` m3 a day (a state,
    seen by the router the next step), stores what it is given and returns
    `ret` of its store to the river the next day (an injection)."""
    name = 'user'
    backends = ('cpu', 'cuda')
    interval_s = DAY
    imported_fluxes = ('received',)
    exported_fluxes = ('return_flow',)
    dependencies = ('routing',)

    def __init__(self, want, ret=0.5, reject_at=None):
        super().__init__()
        self.want = torch.as_tensor(want, dtype=torch.float64)
        self.ret, self.reject_at = ret, reject_at
        self.n = len(want)
        self.store = torch.zeros(self.n, dtype=torch.float64)
        self.wanted = set()
        self.inputs = (VarSpec('received', 'm3', ('site',), kind='flux',
                               grid='river_sites'),)
        self.outputs = (
            VarSpec('request', 'm3', ('site',), kind='state',
                    grid='river_sites'),
            VarSpec('return_flow', 'm3', ('site',), kind='flux',
                    grid='river_sites'))

    def initialize(self, context):
        self.dev = context.device
        self.store = self.store.to(self.dev)
        self.want = self.want.to(self.dev)

    def compute(self, inputs, context, dt):
        rec = inputs.get('received')
        rec = torch.zeros_like(self.store) if rec is None else rec
        out = self.ret * self.store
        s1 = self.store + rec - out
        return Proposal(
            self.name, context.t, context.t + dt, state=s1,
            exports=dict(request=self.want.clone(), return_flow=out),
            balance=Balance(I_ext=rec.sum(), Q_out=out.sum(),
                            dS=(s1 - self.store).sum(), n_terms=3 * self.n,
                            scale=(rec.sum() + out.sum() + s1.sum()
                                   + self.store.sum())))

    def validate(self, proposal, context):
        if self.reject_at is not None and context.t >= self.reject_at:
            return ['user: told to reject']
        return []

    def commit(self, proposal, context):
        self.store = proposal.state

    def save_state(self):
        return dict(store=self.store.cpu())

    def load_state(self, blob):
        self.store = blob['store'].to(self.dev)


USER_CONNS = [
    {'from': 'user.request', 'to': 'routing.withdrawal_request', 'lag': 1},
    {'from': 'routing.withdrawn', 'to': 'user.received'},
    {'from': 'user.return_flow', 'to': 'routing.injection', 'lag': 1}]


def user_run(device, net, gauges, area, table, sites, want, **kw):
    user = User(want, **kw)
    m = run_river(device, net, gauges, area, table,
                  routing_kw=dict(site_cells=sites),
                  modules=dict(user=dict(connector='two_store')),
                  connections=USER_CONNS, extra=dict(user=user))
    return m, user


@pytest.mark.parametrize('device', DEVICES)
def test_withdraw_and_inject_services_equal_the_standalone_hooks(device):
    net, gauges, area, table = network(nt=30)
    sites, want = [7, 60], [2.0e5, 1.0e8]
    m, user = user_run(device, net, gauges, area, table, sites, want)
    m.initialize().run()
    rc = m.connector('routing')
    # the standalone loop: the request seen at step k is the one the user
    # put out at step k-1; the injection is the previous step's return
    ref, _ = standalone(device, net, gauges, area, table, site_cells=sites)
    store = torch.zeros(2, dtype=torch.float64, device=device)
    w = torch.as_tensor(want, dtype=torch.float64, device=device)
    req = inj = None
    qs = []
    for t in range(table.shape[0]):
        q, got = ref.step(table[t].to(device), withdraw=req, inject=inj)
        qs.append(q)
        got = torch.zeros(2, dtype=torch.float64, device=device) \
            if got is None else got
        out = 0.5 * store
        store = store + got - out
        req, inj = w.clone(), out
    assert torch.equal(m.outputs('routing')['discharge'].cpu(),
                       torch.stack(qs).cpu())
    assert torch.equal(rc.router.ring.cpu(), ref.ring.cpu())
    assert torch.equal(user.store.cpu(), store.cpu())
    # the big request is capped by what passes the site
    assert float(rc.router.withdrawn[1]) < 1.0e8 * table.shape[0]
    assert float(rc.router.withdrawn.sum()) > 0
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    # nothing negative anywhere, the engine's own node ledger closes
    assert float(rc.router.ring.min()) >= -1e-9
    assert float(rc.ledger().abs().max()) < 1e-9 * float(
        rc.router.in_runoff.max())


def test_rejected_step_leaves_the_river_where_it_was():
    net, gauges, area, table = network(nt=20)
    m, user = user_run('cpu', net, gauges, area, table, [7, 60],
                       [2.0e5, 1.0e8], reject_at=6 * DAY)
    m.initialize()
    m.run(until=6 * DAY)
    rc = m.connector('routing')
    before = [x.clone() for x in rc._live()]
    with pytest.raises(StepRejected):
        m.step()
    assert m.runtime.time.t == 6 * DAY
    # the rejected compute changed the ring in place and left an undo
    # snapshot; the next access restores it
    rc._rollback()
    assert all(torch.equal(a, b) for a, b in zip(before, rc._live()))
    user.reject_at = None
    m.run()
    ref, _ = m_ref = user_run('cpu', net, gauges, area, table, [7, 60],
                              [2.0e5, 1.0e8])
    ref.initialize().run()
    assert torch.equal(rc.router.ring, ref.connector('routing').router.ring)
    assert torch.equal(rc.router.withdrawn,
                       ref.connector('routing').router.withdrawn)


def test_restart_from_a_checkpoint_equals_the_continuous_run(tmp_path):
    net, gauges, area, table = network(nt=24)

    def build():
        return user_run('cpu', net, gauges, area, table, [7, 60],
                        [2.0e5, 1.0e8])
    a, ua = build()
    a.initialize().run()
    b, ub = build()
    b.initialize().run(until=9 * DAY)
    ck = b.save_checkpoint(str(tmp_path / 'ck.pt'))
    c, uc = build()
    c.initialize().load_checkpoint(ck)
    c.run()
    ra, rcn = a.connector('routing'), c.connector('routing')
    for k in ra.router._STATE:
        assert torch.equal(getattr(ra.router, k), getattr(rcn.router, k)), k
    assert torch.equal(ua.store, uc.store)
    assert len(c.ledger.balances) == len(a.ledger.balances)


# ---------------------------------------------------------------------------
# reservoirs
# ---------------------------------------------------------------------------
def dam_cells(net):
    """Two dams on one river: a big-catchment cell and the cell downstream
    of it that is itself well fed."""
    up = np.ones(net.n)
    for c in net.order:
        if net.down[c] >= 0:
            up[net.down[c]] += up[c]
    gset = {int(net.order[-1]), int(net.order[-40]),
            int(net.order[len(net.order) // 2])}

    def gauged(c):                      # the dams must drain to a gauge
        while c >= 0:
            if int(c) in gset:
                return True
            c = net.down[c]
        return False
    cand = [c for c in net.order if up[c] > 25 and net.down[c] >= 0
            and up[net.down[c]] > up[c] and net.down[net.down[c]] >= 0
            and gauged(net.down[net.down[c]]) and int(c) not in gset
            and int(net.down[net.down[c]]) not in gset]
    c1 = int(cand[0])
    return c1, int(net.down[net.down[c1]]), up


def reservoir_config(net, start='2000-01-25'):
    c1, c2, up = dam_cells(net)
    q = np.array([40.0, 60.0, 50.0, 30.0, 20.0, 25.0, 35.0, 45.0, 50.0,
                  40.0, 30.0, 25.0])
    dams = [dict(cell=c1, capacity_m3=3.0e7, rule='target',
                 target_frac=[0.9] * 12, init_frac=0.8, dead_m3=1e6,
                 q_env=5.0, area_max=1.0e7),
            dict(cell=c2, capacity_m3=2.0e8, rule='hanasaki', init_frac=0.5,
                 irrigation=True, q_env=2.0, build_year=1990)]
    return dams, dict(start_date=start, relax_steps=5.0, evap_m=0.004,
                      inflow_clim=q, demand_clim=0.5 * q), (c1, c2)


def reservoir_run(device, net, gauges, area, table, start='2000-01-25',
                  user=True, modules=None):
    dams, rk, cells = reservoir_config(net, start)
    mods = dict(reservoir=dict(connector='reservoir', dams=dams, **rk))
    mods.update(modules or {})
    conns, extra = [], {}
    kw = dict(routing_kw=dict(reservoirs='reservoir',
                              site_cells=[7] if user else []),
              modules=mods, connections=conns, extra=extra)
    m = run_river(device, net, gauges, area, table, **kw)
    return m, dams, rk, cells


def reservoir_standalone(device, net, gauges, area, table, dams, rk, cells,
                         sites=(7,)):
    from torchvic.reservoir import Reservoirs, ReservoirParams, RULE_TARGET
    from torchvic.reservoir import RULE_HANASAKI
    import datetime
    allsites = list(sites) + [c for c in cells if c not in sites]
    r, info = standalone(device, net, gauges, area, table,
                         site_cells=allsites)
    tf = np.ones((2, 12))
    tf[0] = 0.9
    p = ReservoirParams(
        site=np.array([allsites.index(c) for c in cells]),
        capacity=np.array([d['capacity_m3'] for d in dams]),
        dead=np.array([d.get('dead_m3', 0.0) for d in dams]),
        q_env=np.array([d.get('q_env', 0.0) for d in dams]),
        q_max=np.full(2, np.inf),
        build_year=np.array([d.get('build_year', -1e9) for d in dams]),
        init_frac=np.array([d['init_frac'] for d in dams]),
        area_max=np.array([d.get('area_max', 0.0) for d in dams]),
        area_exp=np.full(2, 1.5),
        irrigation=np.array([d.get('irrigation', False) for d in dams]),
        rule=np.array([RULE_TARGET, RULE_HANASAKI]), target_frac=tf,
        relax_steps=rk['relax_steps'], inflow_clim=rk['inflow_clim'],
        demand_clim=rk['demand_clim'], clim_years=1.0)
    res = Reservoirs(p, r)
    start = datetime.datetime.strptime(rk['start_date'], '%Y-%m-%d')
    qs, last = [], None
    for t in range(table.shape[0]):
        d = start + datetime.timedelta(days=t)
        if last != (d.year, d.month):
            res.new_month(d.month - 1, d.year)
            last = (d.year, d.month)
        qs.append(res.step(table[t].to(device),
                           evap=torch.full((2,), rk['evap_m'],
                                           dtype=torch.float64,
                                           device=device))[0])
    return r, res, torch.stack(qs)


@pytest.mark.parametrize('device', DEVICES)
def test_reservoir_connector_equals_the_engine_and_conserves(device):
    net, gauges, area, table = network(nt=40)
    table = table * 3.0
    m, dams, rk, cells = reservoir_run(device, net, gauges, area, table)
    m.initialize().run()
    r, res, want = reservoir_standalone(device, net, gauges, area, table,
                                        dams, rk, cells)
    rc, rv = m.connector('routing'), m.connector('reservoir')
    assert torch.equal(m.outputs('routing')['discharge'].cpu(), want.cpu())
    assert torch.equal(rv.engine.storage.cpu(), res.storage.cpu())
    assert torch.equal(m.outputs('reservoir')['storage'][-1].cpu(),
                       res.storage.cpu())
    # water: every module row and the coupled rows close
    assert all(x['ok'] for x in m.ledger.balances)
    assert all(x['ok'] for x in m.ledger.coupled)
    assert {x['module'] for x in m.ledger.balances} == {'src', 'routing',
                                                        'reservoir'}
    # the engine's storage identity and the router's node ledger
    assert float(rv.volume_balance().abs().max()) < 1e-6
    assert float(rc.ledger().abs().max()) < 1e-9 * float(
        rc.router.in_runoff.max())
    # the reservoirs worked: evaporation, an environmental release and
    # (the small upstream dam, a wet period) a spill
    cum = rv.engine.cum
    assert float(cum['evap'][0]) > 0 and float(cum['env'].min()) > 0
    assert float(cum['spill'].sum()) > 0 and float(cum['release'].sum()) > 0


@pytest.mark.parametrize('device', DEVICES)
def test_reservoir_release_and_spill_reach_the_outlets(device):
    """With dams, the water that left a dam's catchment arrives downstream:
    after the river drains, the outlet gauge below the dams has received
    exactly the dry-run volume minus what stayed in the dams (storage
    change, evaporation, supplied)."""
    net, gauges, area, table = network(nt=40)
    nt = table.shape[0]
    table = torch.cat([table * 3.0, torch.zeros(160, net.n,
                                                dtype=torch.float64)])
    m, dams, rk, cells = reservoir_run(device, net, gauges, area, table)
    m.initialize().run()
    dry = run_river(device, net, gauges, area, table,
                    routing_kw=dict(site_cells=[7]))
    dry.initialize().run()
    rc, rd = m.connector('routing'), dry.connector('routing')
    rv = m.connector('reservoir')
    assert float(rc.router.ring.abs().max()) < 1e-6        # drained
    cum = rv.engine.cum
    kept = (rv.engine.storage - rv.engine.init_frac * rv.engine.cap).sum() \
        + cum['evap'].sum() + cum['withdrawn'].sum()
    out_m = rc.router.out[rc.outlet_idx].sum()
    out_d = rd.router.out[rd.outlet_idx].sum()
    # both dams sit upstream of the same outlet
    assert abs(float(out_d - out_m - kept)) < 1e-9 * float(out_d)
    assert float(kept) != 0.0


def test_reservoir_restart_and_rollback_cross_a_month_boundary(tmp_path):
    net, gauges, area, table = network(nt=24)
    table = table * 3.0

    def build():
        return reservoir_run('cpu', net, gauges, area, table)[0]
    a = build().initialize()
    a.run()
    b = build().initialize()
    b.run(until=9 * DAY)                      # 2000-01-25 + 9 d: in February
    ck = b.save_checkpoint(str(tmp_path / 'ck.pt'))
    c = build().initialize().load_checkpoint(ck)
    c.run()
    ea, ec = a.connector('reservoir').engine, c.connector('reservoir').engine
    for k in ea._STATE:
        assert torch.equal(getattr(ea, k), getattr(ec, k)), k
    for k in a.connector('routing').router._STATE:
        assert torch.equal(getattr(a.connector('routing').router, k),
                           getattr(c.connector('routing').router, k)), k
    assert a.connector('reservoir')._cal == c.connector('reservoir')._cal
    assert len(c.ledger.balances) == len(a.ledger.balances)
    # a rejected step (the reservoir is told to fail) leaves the state
    d = build().initialize()
    d.run(until=8 * DAY)                      # the next step is 2000-02-02
    rv = d.connector('reservoir')
    eng = rv.engine
    snap = {k: getattr(eng, k).clone() for k in eng._STATE}
    orig = rv.validate
    rv.validate = lambda p, ctx: ['told to reject']
    with pytest.raises(StepRejected):
        d.step()
    rv.validate = orig
    d.connector('routing')._rollback()
    for k, v in snap.items():
        assert torch.equal(getattr(eng, k), v), k
    d.run()                                    # the same step now succeeds
    for k in ea._STATE:
        assert torch.equal(getattr(ea, k), getattr(eng, k)), k


# ---------------------------------------------------------------------------
# water quality and temperature
# ---------------------------------------------------------------------------
QUAL = dict(constituents={'n': 2.0e-7, 'tracer': 0.0},
            temperature=dict(depth=1.0), loads={'n': 3.0, 'tracer': 1.5},
            met=dict(t_air=12.0, vp=0.9, wind=2.0, sw=180.0),
            t_runoff=4.0)


def quality_run(device, net, gauges, area, table, sites=(7, 60), **q):
    user = User([2.0e5, 1.0e8])
    cfg = dict(QUAL)
    cfg.update(q)
    m = run_river(device, net, gauges, area, table,
                  routing_kw=dict(site_cells=list(sites), quality='quality'),
                  modules=dict(user=dict(connector='two_store'),
                               quality=dict(connector='quality', **cfg)),
                  connections=USER_CONNS, extra=dict(user=user))
    return m, user


def quality_standalone(device, net, gauges, area, table, sites=(7, 60)):
    from torchvic.quality import WaterQuality
    r, info = standalone(device, net, gauges, area, table,
                         site_cells=list(sites))
    wq = WaterQuality(r, area, {'n': 2.0e-7, 'tracer': 0.0},
                      temperature=dict(depth=1.0), site_cells=list(sites))
    n = net.n
    f = lambda v: torch.full((n,), v, dtype=torch.float64, device=device)
    met = dict(t_air=f(12.0), vp=f(0.9), wind=f(2.0), sw=f(180.0))
    w = torch.tensor([2.0e5, 1.0e8], dtype=torch.float64, device=device)
    store = torch.zeros(2, dtype=torch.float64, device=device)
    req = inj = None
    outs = []
    for t in range(table.shape[0]):
        o = wq.step(table[t].to(device), loads={'n': f(3.0), 'tracer': f(1.5)},
                    met=met, t_runoff=f(4.0), withdraw=req, inject=inj)
        outs.append(o)
        got = o['got'][:2]
        out = 0.5 * store
        store = store + got - out
        req, inj = w.clone(), out
    return r, wq, outs


@pytest.mark.parametrize('device', DEVICES)
def test_quality_connector_equals_the_engine_and_the_ledgers_close(device):
    net, gauges, area, table = network(nt=30)
    m, user = quality_run(device, net, gauges, area, table)
    m.initialize().run()
    r, wq, outs = quality_standalone(device, net, gauges, area, table)
    rc, qc = m.connector('routing'), m.connector('quality')
    series = m.outputs('quality')
    for k, key in (('conc_n', 'n'), ('conc_tracer', 'tracer')):
        want = torch.stack([o['conc'][key] for o in outs])
        assert torch.equal(series[k].cpu(), want.cpu()), k
    assert torch.equal(series['temperature'].cpu(),
                       torch.stack([o['t'] for o in outs]).cpu())
    assert torch.equal(m.outputs('routing')['discharge'].cpu(),
                       torch.stack([o['q'] for o in outs]).cpu())
    assert torch.equal(rc.router.ring.cpu(), r.ring.cpu())
    # water ledger
    assert all(x['ok'] for x in m.ledger.balances)
    assert all(x['ok'] for x in m.ledger.coupled)
    # mass and heat ledgers: node by node, in_runoff + in_hooks - out -
    # stored, for water, each constituent and heat
    led = qc.mass_ledger()
    sc = {'n': wq.ch['n'].in_runoff, 'tracer': wq.ch['tracer'].in_runoff,
          'heat': wq.temp[0].in_runoff + wq.temp[1].in_runoff,
          'water': r.in_runoff}
    for k, v in sc.items():
        assert float(led[k].abs().max()) < 1e-9 * float(v.abs().max()), k
    # the decayed mass is a sink: loaded = delivered + stored + decayed,
    # added up over the outlets (loads enter each routed cell once)
    o = rc.outlet_idx
    for key, k in (('n', 2.0e-7), ('tracer', 0.0)):
        ch = qc.wq.ch[key]
        assert float((ch.in_runoff[o] + ch.in_hooks[o] - ch.out[o]
                      - ch.ring.sum(0)[o] * ch.dt).abs().max()) \
            < 1e-9 * float(ch.in_runoff[o].max())
    assert float(led['decayed_n'][o].sum()) > 0
    assert float(led['decayed_tracer'].abs().max()) < 1e-12 * float(
        qc.wq.ch['tracer'].in_runoff.max())
    assert float(led['atmosphere_heat'].max()) > 0
    # mixing: the conservative tracer is a mass-weighted mean of its parts,
    # so its concentration is bounded by the per-cell load / runoff ratio
    c = series['conc_tracer'][-1]
    q = m.outputs('routing')['discharge'][-1]
    ok = q > 1e-6
    assert bool((c[ok] >= 0).all()) and bool(torch.isfinite(c).all())


def test_quality_restart_and_a_rejected_step(tmp_path):
    net, gauges, area, table = network(nt=20)

    def build():
        return quality_run('cpu', net, gauges, area, table)
    a, ua = build()
    a.initialize().run()
    b, ub = build()
    b.initialize().run(until=7 * DAY)
    ck = b.save_checkpoint(str(tmp_path / 'ck.pt'))
    c, uc = build()
    c.initialize().load_checkpoint(ck)
    c.run()
    qa, qc_ = a.connector('quality'), c.connector('quality')
    for x, y in zip(qa.state_tensors(), qc_.state_tensors()):
        assert torch.equal(x, y)
    assert torch.equal(a.connector('routing').router.ring,
                       c.connector('routing').router.ring)
    # a rejected step: the engine state is the one before it
    d, ud = build()
    d.initialize().run(until=5 * DAY)
    qd = d.connector('quality')
    snap = [x.clone() for x in qd.state_tensors()]
    ud.reject_at = 5 * DAY
    with pytest.raises(StepRejected):
        d.step()
    d.connector('routing')._rollback()
    assert all(torch.equal(x, y) for x, y in zip(snap, qd.state_tensors()))
    ud.reject_at = None
    d.run()
    for x, y in zip(qa.state_tensors(), qd.state_tensors()):
        assert torch.equal(x, y)


def combined_run(device, net, gauges, area, table, build_year=None,
                 request=2.0e5):
    dams, rk, cells = reservoir_config(net)
    if build_year is not None:
        for d in dams:
            d['build_year'] = build_year
    rk = dict(rk, request_m3=request)
    q = dict(QUAL, loads={'n': 3.0, 'tracer': 1.5})
    m = run_river(device, net, gauges, area, table,
                  routing_kw=dict(reservoirs='reservoir', quality='quality'),
                  modules=dict(reservoir=dict(connector='reservoir',
                                              dams=dams, **rk),
                               quality=dict(connector='quality', **q)))
    return m, dams, rk, cells


@pytest.mark.parametrize('device', DEVICES)
def test_reservoirs_and_quality_share_one_sweep_and_conserve(device):
    net, gauges, area, table = network(nt=40)
    table = table * 3.0
    m, dams, rk, cells = combined_run(device, net, gauges, area, table)
    m.initialize().run()
    rc, rv, qc = (m.connector(k) for k in ('routing', 'reservoir', 'quality'))
    # water: the reservoir-only router, bit for bit
    ref = run_river(device, net, gauges, area, table,
                    routing_kw=dict(reservoirs='reservoir'),
                    modules=dict(reservoir=dict(connector='reservoir',
                                                dams=dams, **rk)))
    ref.initialize().run()
    assert torch.equal(m.outputs('routing')['discharge'].cpu(),
                       ref.outputs('routing')['discharge'].cpu())
    assert torch.equal(rv.engine.storage.cpu(),
                       ref.connector('reservoir').engine.storage.cpu())
    assert all(x['ok'] for x in m.ledger.balances)
    assert all(x['ok'] for x in m.ledger.coupled)
    cum = rv.engine.cum
    assert float(cum['spill'].sum()) > 0 and float(cum['withdrawn'].sum()) > 0
    # mass and heat: every node's ledger, and the dams' store identity
    led = qc.mass_ledger()
    w = qc.wq
    sc = {'n': w.ch['n'].in_runoff, 'tracer': w.ch['tracer'].in_runoff,
          'heat': w.temp[0].in_runoff + w.temp[1].in_runoff}
    for k, v in sc.items():
        assert float(led[k].abs().max()) < 1e-9 * float(v.abs().max()), k
    store, c = w.store, w.cum
    assert float(store[:, :].abs().max()) > 0
    assert float(led['reservoir_store'].abs().max()) < 1e-9 * float(
        c[0].abs().max())
    # the conservative tracer, all of it: loaded on the routed cells =
    # delivered at the outlets + in the river + in the dams + supplied
    o = rc.outlet_idx
    ch = w.ch['tracer']
    loaded = 1.5 * int(rc.routed.sum()) * table.shape[0]
    held = float((ch.out[o] + ch.ring.sum(0)[o] * ch.dt).sum()) \
        + float(store[1].sum()) + float(c[2, 1].sum())
    assert abs(loaded - held) < 1e-9 * loaded
    assert float(c[2, 1].sum()) > 0          # the supplied water took mass
    # quality is a passenger of the dams: a dam holds mass back
    assert float(store[1].sum()) > 0


@pytest.mark.parametrize('device', DEVICES)
def test_dams_not_yet_built_pass_mass_through(device):
    """Before their build year the dams only pass water and mass: the
    combined sweep equals the quality-only router."""
    net, gauges, area, table = network(nt=20)
    m, dams, rk, cells = combined_run(device, net, gauges, area, table,
                                      build_year=3000, request=0.0)
    m.initialize().run()
    # quality-only router with the same sites
    qo = run_river(device, net, gauges, area, table,
                   routing_kw=dict(quality='quality',
                                   site_cells=[int(c) for c in cells]),
                   modules=dict(quality=dict(connector='quality',
                                             **dict(QUAL, loads={
                                                 'n': 3.0, 'tracer': 1.5}))))
    qo.initialize().run()
    for k in ('conc_tracer', 'conc_n', 'temperature'):
        a, b = m.outputs('quality')[k], qo.outputs('quality')[k]
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-12), k
    assert torch.allclose(m.outputs('routing')['discharge'],
                          qo.outputs('routing')['discharge'], rtol=1e-12)
    assert float(m.connector('quality').wq.store.abs().max()) == 0.0


def test_combined_restart_and_rollback(tmp_path):
    net, gauges, area, table = network(nt=24)
    table = table * 3.0

    def build():
        return combined_run('cpu', net, gauges, area, table)[0]
    a = build().initialize()
    a.run()
    b = build().initialize()
    b.run(until=9 * DAY)
    ck = b.save_checkpoint(str(tmp_path / 'ck.pt'))
    c = build().initialize().load_checkpoint(ck)
    c.run()
    for x, y in zip(a.connector('quality').state_tensors(),
                    c.connector('quality').state_tensors()):
        assert torch.equal(x, y)
    # a rejected step undoes the stores too
    d = build().initialize()
    d.run(until=5 * DAY)
    qd = d.connector('quality')
    snap = [x.clone() for x in qd.state_tensors()]
    qd.validate = lambda p, ctx: ['told to reject']
    with pytest.raises(StepRejected):
        d.step()
    del qd.validate
    d.connector('routing')._rollback()
    assert all(torch.equal(x, y) for x, y in zip(snap, qd.state_tensors()))
    d.run()
    for x, y in zip(a.connector('quality').state_tensors(),
                    qd.state_tensors()):
        assert torch.equal(x, y)


# ---------------------------------------------------------------------------
# VIC (Stehekin) + routing
# ---------------------------------------------------------------------------
import _paths                                                           # noqa: E402
from torchvic import case as _case                                      # noqa: E402

STEHE = _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                set_options={'SNOW_BAND': 'FALSE'})
STEHE_DATA = os.path.join(_paths.SAMPLES, 'image', 'Stehekin')


@pytest.fixture(autouse=True)
def _stehe_env(monkeypatch):
    import gc
    monkeypatch.setenv('test_data_dir', _paths.SAMPLES)
    gc.collect()                  # close the previous test's HDF5 files now
    yield
    gc.collect()


def _need_stehe():
    if not (os.path.exists(STEHE) and os.path.isdir(STEHE_DATA)):
        pytest.skip("VIC's Stehekin sample is not present")


def stehe_network(case):
    """A river network on Stehekin's 16 cells: the sample ships no flow
    directions and a priority flood on its elevations gives almost only
    mouths, so the cells are ordered by elevation (lowest first, from the
    parameter file) and cell k drains to cell (k - 1) // 2 -- a binary tree
    whose mouth is the lowest cell -- with great-circle channel lengths. A
    stand-in network for the connector tests, not a validated river. Returns
    (network, cell of position k, area m2 x land fraction, position -> cell)."""
    n = case.n_cells
    lat = np.asarray(case.dom.lat, dtype=np.float64)[:n]     # per cell
    lon = np.asarray(case.dom.lon, dtype=np.float64)[:n]
    z = np.asarray(case.batch.elevation.cpu(), dtype=np.float64)[:n]
    pos = np.argsort(z, kind='stable')               # position -> cell
    where = np.empty(n, np.int64)
    where[pos] = np.arange(n)
    down = -np.ones(n, np.int64)
    length = np.zeros(n)
    for k in range(1, n):
        c, dc = pos[k], pos[(k - 1) // 2]
        down[c] = dc
        length[c] = rn._great_circle(lat[c], lon[c], lat[dc], lon[dc])
    net = rn.RiverNetwork(down, length)
    gauges = np.unique([int(pos[0]), int(pos[1])])
    area = np.asarray(case.dom.area, dtype=np.float64)[:n] \
        * np.asarray(case.dom.frac, dtype=np.float64)[:n]
    return net, gauges, area, pos


_CACHE = {}


def vic_river(device, vic=None, routing=None, extra=None, var='total_runoff',
              **rt):
    if 'net' not in _CACHE:         # one object: a checkpoint's digest
        c = _case.load(STEHE, device='cpu', log=lambda *a: None)
        net, gauges, area, pos = stehe_network(c)   # names the config
        net.pos = pos
        _CACHE['net'] = (net, gauges, area)
        del c
        import gc
        gc.collect()
    net, gauges, area = _CACHE['net']
    rk = dict(connector='routing', net=net, gauge_cells=gauges, area=area,
              cell_uh=[0.3, 0.5, 0.2])
    rk.update(routing or {})
    mods = dict(vic=dict(dict(global_param=STEHE, exchange='day'),
                         **(vic or {})), routing=rk)
    if extra is not None:
        mods.update(extra(net))
    m = TorchVIC.from_dict(
        dict(run=dict(device=device, log=False), modules=mods,
             connections=[{'from': 'vic.' + var,
                           'to': 'routing.inflow'}], **rt))
    return m, None, (net, gauges, area)


def vic_direct_daily(c, window_days=1):
    """VIC's own per-step runoff + baseflow, as the exports form it: the
    window sums of runoff and of baseflow, added."""
    out = c.run()
    n = c.model_steps * window_days
    ro, bf = out['runoff'], out['baseflow']
    return [ro[i:i + n].sum(0) + bf[i:i + n].sum(0)
            for i in range(0, ro.shape[0], n)], out


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
def test_vic_plus_routing_gauge_discharge_equals_routing_vic_standalone(
        device):
    _need_stehe()
    with torch.inference_mode():
        m, _, (net, gauges, area) = vic_river(device)
        m.initialize().run()
        c = _case.load(STEHE, device=device, log=lambda *a: None)
        sums, out = vic_direct_daily(c)
        r, _ = standalone(device, net, gauges, area,
                          torch.zeros(1, net.n, dtype=torch.float64))
        want = torch.stack([r.step(s.to(torch.float64))[0] for s in sums])
        got = m.outputs('routing')['discharge']
        assert got.shape == want.shape and float(got.abs().max()) > 0
        assert torch.equal(got.cpu(), want.cpu())
        assert torch.equal(m.connector('routing').router.ring.cpu(),
                           r.ring.cpu())
        assert all(x['ok'] for x in m.ledger.balances)
        assert all(x['ok'] for x in m.ledger.coupled)
        _global_budget(m)


def _global_budget(m, rel=1e-9):
    """P - ET - Q_out(outlets, withdrawals) - E_unrouted - dS - closure = 0
    over the whole run, from the module totals (the buffers' content in
    transit is part of dS)."""
    v, rt = m.ledger.totals('vic'), m.ledger.totals('routing')
    dS_buf = sum(x['dS_buffers'] for x in m.ledger.coupled)
    resid = (v['P'] + rt['P'] - v['ET'] - rt['Q_out'] - rt['E_other']
             - v['dS'] - rt['dS'] - dS_buf - v['closure'])
    assert abs(resid) <= rel * abs(v['P']), (resid, v['P'])
    return resid


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
def test_vic_windows_spread_over_routing_steps_conserve(device):
    """VIC in 5-day windows feeds the daily router: each day gets 1/5 of the
    window's runoff (the runtime's exchange buffer), the router equals the
    standalone one fed the same, and the coupled ledger closes."""
    _need_stehe()
    with torch.inference_mode():
        m, _, (net, gauges, area) = vic_river(
            device, vic=dict(exchange='window', window_days=5))
        m.initialize().run()
        c = _case.load(STEHE, device=device, log=lambda *a: None)
        sums, out = vic_direct_daily(c, window_days=5)
        r, _ = standalone(device, net, gauges, area,
                          torch.zeros(1, net.n, dtype=torch.float64))
        frac = (DAY) / (5 * DAY)
        want = []
        for s in sums:
            for _ in range(5):
                want.append(r.step(s.to(torch.float64) * frac)[0])
        got = m.outputs('routing')['discharge']
        assert torch.equal(got.cpu(), torch.stack(want).cpu())
        assert all(x['ok'] for x in m.ledger.balances)
        assert all(x['ok'] for x in m.ledger.coupled)
        _global_budget(m)
        # the water VIC sent is the water the router received
        rows = [x for x in m.ledger.transfers if x['quantity'] ==
                'total_runoff']
        sent = sum(x['amount'] for x in rows if x['leg'] == 'sent')
        dlv = sum(x['amount'] for x in rows if x['leg'] == 'delivered')
        assert abs(sent - dlv) <= 1e-9 * sent


def stehe_dams(net, **kw):
    """Two nested dams: positions 3 and 1 of the stand-in tree (3 -> 1 -> 0)."""
    p = net.pos
    d1 = dict(cell=int(p[3]), capacity_m3=1.5e5, rule='target',
              target_frac=[0.9] * 12, init_frac=0.8, dead_m3=1e4, q_env=0.02,
              area_max=2.0e5)
    d2 = dict(cell=int(p[1]), capacity_m3=6.0e5, rule='target',
              target_frac=[0.5] * 12, init_frac=0.5, q_env=0.05)
    return [d1, d2]


def vic_reservoir_river(device, **kw):
    def extra(net):
        return dict(reservoir=dict(connector='reservoir',
                                   dams=stehe_dams(net),
                                   start_date='1949-01-01', relax_steps=3.0,
                                   evap_m=0.002))
    m, c, nw = vic_river(device, routing=dict(reservoirs='reservoir'),
                         extra=extra, **kw)
    return m, c, nw, None


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
def test_vic_reservoir_routing_conserves_and_dams_pass_water_down(device):
    _need_stehe()
    with torch.inference_mode():
        m, c, (net, gauges, area), dams = vic_reservoir_river(device)
        m.initialize().run()
        rv, rc = m.connector('reservoir'), m.connector('routing')
        assert all(x['ok'] for x in m.ledger.balances)
        assert all(x['ok'] for x in m.ledger.coupled)
        assert {x['module'] for x in m.ledger.balances} == {
            'vic', 'routing', 'reservoir'}
        # reservoir water is in the coupled budget: P - ET (land and
        # reservoir surface) - outlets - unrouted - dS (land, river, dams,
        # buffers) - VIC's own closure = 0
        v, rt, rs = (m.ledger.totals(k) for k in ('vic', 'routing',
                                                  'reservoir'))
        dS_buf = sum(x['dS_buffers'] for x in m.ledger.coupled)
        resid = (v['P'] - v['ET'] - rs['ET'] - rt['Q_out'] - rs['Q_out']
                 + rt['I_ext'] - v['Q_out'] + rs['I_ext']
                 - rt['E_other'] - v['dS'] - rt['dS'] - rs['dS'] - dS_buf
                 - v['closure'])
        # (rt Q_out/I_ext include the take / release transfers with the
        # dams, rs Q_out/I_ext the same ones seen from the dams: they
        # cancel; rt I_ext and v Q_out are the VIC -> river transfer)
        assert abs(resid) <= 1e-9 * abs(v['P']), (resid, v['P'])
        cum = rv.engine.cum
        assert float(cum['release'].sum() + cum['env'].sum()) > 0
        # what the dams released arrives at the mouth gauge
        q = m.outputs('routing')['discharge']
        assert float(q.max()) > 0
        assert float(rv.volume_balance().abs().max()) < 1e-6


@pytest.mark.reference
def test_vic_reservoir_routing_restart_equals_the_continuous_run(tmp_path):
    _need_stehe()
    with torch.inference_mode():
        a = vic_reservoir_river('cpu')[0].initialize()
        a.run()
        b = vic_reservoir_river('cpu')[0].initialize()
        b.run(until=4 * DAY)
        ck = b.save_checkpoint(str(tmp_path / 'ck.pt'))
        c = vic_reservoir_river('cpu')[0].initialize().load_checkpoint(ck)
        c.run()
        ea, ec = a.connector('reservoir').engine, c.connector(
            'reservoir').engine
        for k in ea._STATE:
            assert torch.equal(getattr(ea, k), getattr(ec, k)), k
        ra, rcn = a.connector('routing').router, c.connector(
            'routing').router
        for k in ra._STATE:
            assert torch.equal(getattr(ra, k), getattr(rcn, k)), k
        # the discharge after the restart is the continuous run's
        qa = a.outputs('routing')['discharge'][4:]
        qc = c.outputs('routing')['discharge']
        assert torch.equal(qa, qc)
        assert torch.equal(a.connector('vic').state['moist'],
                           c.connector('vic').state['moist'])


_FILES_SCRIPT = r"""
import sys, os, numpy as np, torch, netCDF4
sys.path.insert(0, os.path.join(os.environ['REPO'], 'tests'))
from torchvic.model import river_network as rn
from torchvic.coupling import TorchVIC
from test_coupling_routing import Source, DAY
d = sys.argv[1]
g = np.random.default_rng(5)
ny, nx = 6, 7
elev = g.random((ny, nx)) * 100 + np.linspace(0, 300, nx)[None, :]
mask = np.ones((ny, nx), bool); mask[0, :2] = False
lat = 20 + np.arange(ny) * 0.0625; lon = 70 + np.arange(nx) * 0.0625
net0, cells = rn.derive_d8_from_elevation(elev, mask, lat, lon)
inv = {v: k for k, v in rn.ESRI_D8.items()}
code = np.zeros((ny, nx), np.int16)
for i, (r, c) in enumerate(cells):
    if net0.down[i] >= 0:
        dr, dc = cells[net0.down[i]] - (r, c)
        code[r, c] = inv[(int(dr), int(dc))]
area = g.random((ny, nx)) * 1e7 + 2e7
frac = 0.5 + 0.5 * g.random((ny, nx))
def put(path, **v):
    ds = netCDF4.Dataset(path, 'w')
    ds.createDimension('lat', ny); ds.createDimension('lon', nx)
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    for k, a in v.items():
        ds.createVariable(k, a.dtype, ('lat', 'lon'))[:] = a
    ds.close()
put(d + '/fdir.nc', fdir=code)
put(d + '/domain.nc', mask=mask.astype(np.int8), area=area, frac=frac)
net, cells2 = rn.d8_downstream(code, mask, lat, lon)
low = [int(net.order[-1]), int(net.order[len(net.order) // 3])]
with open(d + '/gauges.csv', 'w') as fh:
    fh.write('lat,lng\n')
    for c in low:
        fh.write('%.10f,%.10f\n' % (lat[cells2[c, 0]], lon[cells2[c, 1]]))
a = (area * frac)[cells2[:, 0], cells2[:, 1]]
nt = 12
table = torch.rand(nt, net.n, generator=torch.Generator().manual_seed(1),
                   dtype=torch.float64) * 4
def run(**kw):
    m = TorchVIC.from_dict(
        dict(run=dict(device='cpu', log=False),
             modules=dict(src=dict(connector='two_store'),
                          routing=dict(connector='routing', **kw)),
             connections=[{'from': 'src.total_runoff', 'to': 'routing.inflow'}]),
        connectors=dict(src=Source(table, a)))
    m.initialize().run()
    return m
m1 = run(fdir=d + '/fdir.nc:fdir', domain=d + '/domain.nc',
         gauges=d + '/gauges.csv')
m2 = run(net=net, gauge_cells=np.array(sorted(low)), area=a)
q1, q2 = m1.outputs('routing')['discharge'], m2.outputs('routing')['discharge']
assert torch.equal(q1, q2) and float(q1.abs().max()) > 0
assert all(r['ok'] for r in m1.ledger.balances)
print('files ok', q1.shape)
"""


def test_the_network_from_files_equals_the_network_from_objects(tmp_path):
    import subprocess
    env = dict(os.environ, REPO=os.path.dirname(HERE),
               PYTHONPATH=os.path.dirname(HERE), HDF5_USE_FILE_LOCKING='FALSE')
    r = subprocess.run([sys.executable, '-c', _FILES_SCRIPT, str(tmp_path)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0 and 'files ok' in r.stdout, r.stderr[-2000:]


def test_the_shipped_configurations_validate(monkeypatch, tmp_path):
    from torchvic.coupling import validate_config
    for k in ('GLOBAL', 'FDIR', 'DOMAIN', 'GAUGES', 'DAMS'):
        f = tmp_path / k.lower()
        f.write_text('x')
        monkeypatch.setenv('TORCHVIC_' + k, str(f))
    cfgs = os.path.join(os.path.dirname(HERE), 'configs')
    for n in ('vic_routing', 'vic_reservoir_routing', 'vic_quality_routing'):
        problems = validate_config(os.path.join(cfgs, n + '.yaml'))
        assert problems == [], (n, problems)


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
def test_vic_windows_routed_day_by_day_from_vics_daily_series(device):
    """VIC in 5-day windows, the router fed `total_runoff_daily`: the
    discharge is, bit for bit, the standalone router fed VIC's own daily
    series -- the window total is not spread evenly -- so the flood peak of
    a day is the peak VIC computed. The ledger closes."""
    _need_stehe()
    with torch.inference_mode():
        m, _, (net, gauges, area) = vic_river(
            device, vic=dict(exchange='window', window_days=5),
            var='total_runoff_daily')
        m.initialize().run()
        c = _case.load(STEHE, device=device, log=lambda *a: None)
        sums, out = vic_direct_daily(c, window_days=1)
        r, _ = standalone(device, net, gauges, area,
                          torch.zeros(1, net.n, dtype=torch.float64))
        want = torch.stack([r.step(s.to(torch.float64))[0] for s in sums])
        got = m.outputs('routing')['discharge']
        assert torch.equal(got.cpu(), want.cpu())
        assert torch.equal(m.connector('routing').router.ring.cpu(),
                           r.ring.cpu())
        assert all(x['ok'] for x in m.ledger.balances)
        assert all(x['ok'] for x in m.ledger.coupled)
        _global_budget(m)
        # the even spreading smooths the peak the daily series keeps
        u, _, _ = vic_river(device, vic=dict(exchange='window',
                                             window_days=5))
        u.initialize().run()
        qu = u.outputs('routing')['discharge']
        assert float(got.max()) >= float(qu.max()) * (1 - 1e-12)
        assert not torch.equal(got, qu)
