"""Two-way flood coupling (docs/connectors/flood.md): the routing
connector hands the river over at the entry site, the flood engine owns it
inside the box and returns what leaves through its outlet to the routing
connector downstream; optional floodplain infiltration into VIC's soil
(soil_add, capped) and open-water evaporation (E_other).

A synthetic channel: 12 cells in a chain (5 km links), cells 4-7 form the
flood box (their runoff goes to the flood model), the entry site is cell 3,
the return site cell 8, the gauge the mouth (cell 11). The hydraulic grid
is a valley falling east with a channel row; the river enters at the west
channel cells and leaves through the open east edge (the return outlet)."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from torchvic.coupling import (TorchVIC, Connector, Balance, Proposal,  # noqa: E402
                               StepRejected, VarSpec, CouplingError,
                               registry)
from torchvic.coupling.spec import GridSpec                                # noqa: E402
from torchvic.coupling.connectors import flood as F                        # noqa: E402
from torchvic.flood import FloodEngine                                     # noqa: E402
from torchvic.model import river_network as rn                             # noqa: E402

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
DAY = 86400.0
NC = 12                       # river cells
BOX = [4, 5, 6, 7]            # VIC cells inside the flood box
ENTRY, RETURN, MOUTH = 3, 8, 11
NY, NX, DX = 12, 32, 500.0
CH = [5, 6]                   # channel rows


@pytest.fixture(autouse=True)
def _on_cpu():
    was = torch.empty(0).device
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(was)


def valley():
    """Falls east 1e-3; the channel rows sit 1.5 m below the floodplain,
    which rises 0.05 m per row away from the channel."""
    x = np.arange(NX)[None, :] * DX
    y = np.abs(np.arange(NY)[:, None] - 5.5)
    z = 20.0 - 1e-3 * x + 0.05 * y * np.ones((1, NX))
    z[CH, :] -= 1.5
    return z


_NET = {}


def network():
    """One object per session (a checkpoint's digest names it)."""
    if 'net' not in _NET:
        down = np.append(np.arange(1, NC), -1)
        length = np.append(np.full(NC - 1, 5000.0), 0.0)
        _NET['net'] = (rn.RiverNetwork(down, length), np.full(NC, 4.0e7))
    return _NET['net']


def runoff_table(nt=10, wet=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(nt, NC, generator=g, dtype=torch.float64) * 30.0
    t[wet:] = 0.0
    return t


class Source(Connector):
    """Land surface stand-in: daily runoff (mm) from a table (P = Q_out)."""
    name = 'src'
    backends = ('cpu', 'cuda')
    interval_s = DAY
    exported_fluxes = ('total_runoff',)

    def __init__(self, table, area, **c):
        super().__init__(**c)
        self.table, self.area = table, area
        self.outputs = (VarSpec('total_runoff', 'mm', ('cell',), kind='flux',
                                grid='vic_cells'),)

    def grids(self):
        return [GridSpec('vic_cells', NC, area=torch.as_tensor(self.area))]

    def initialize(self, context):
        self.dev = context.device

    def time_span(self):
        return (0.0, self.table.shape[0] * DAY)

    def compute(self, inputs, context, dt):
        k = int(round(context.t / DAY))
        x = self.table[k].to(self.dev)
        v = (x * torch.as_tensor(self.area, device=self.dev)).sum() * 1e-3
        return Proposal(self.name, context.t, context.t + dt, state=None,
                        exports=dict(total_runoff=x),
                        balance=Balance(P=v, Q_out=v, n_terms=NC,
                                        scale=2 * v.abs()))

    def commit(self, proposal, context):
        pass


class Split(Connector):
    """Each VIC cell's water goes whole to routing or to the flood box."""
    name = 'split'
    backends = ('cpu', 'cuda')
    interval_s = DAY
    imported_fluxes = ('runoff',)
    exported_fluxes = ('routed', 'box')

    def __init__(self, area, reject_at=None, **c):
        super().__init__(**c)
        self.area = area
        self.reject_at = reject_at
        self.inputs = (VarSpec('runoff', 'mm', ('cell',), kind='flux',
                               grid='vic_cells'),)
        self.outputs = (VarSpec('routed', 'mm', ('cell',), kind='flux',
                                grid='vic_cells'),
                        VarSpec('box', 'mm', ('cell',), kind='flux',
                                grid='box_cells'))

    def grids(self):
        return [GridSpec('box_cells', len(BOX), area=torch.as_tensor(
            self.area[BOX]))]

    def compute(self, inputs, context, dt):
        x = inputs['runoff']
        bi = torch.as_tensor(BOX, device=x.device)
        inbox = torch.zeros_like(x, dtype=torch.bool)
        inbox[bi] = True
        box = x[bi].clone()
        routed = torch.where(inbox, torch.zeros_like(x), x)
        a = torch.as_tensor(self.area, device=x.device)
        v = (x * a).sum() * 1e-3
        return Proposal(self.name, context.t, context.t + dt,
                        exports=dict(routed=routed, box=box),
                        balance=Balance(I_ext=v, Q_out=v, n_terms=NC,
                                        scale=2 * v.abs()))

    def validate(self, proposal, context):
        if self.reject_at is not None and context.t >= self.reject_at:
            return ['split: told to reject']
        return []

    def commit(self, proposal, context):
        pass


def flood_points():
    return [(r, 1) for r in CH]


def build(device='cpu', table=None, two_way=True, reject_at=None,
          flood_kw=None):
    net, area = network()
    table = runoff_table() if table is None else table
    src = Source(table, area)
    split = Split(area, reject_at=reject_at)
    pts = flood_points()
    # box VIC cell k <- hydraulic columns 8k..8k+7 (equal shares)
    pairs = [[c // 8, r * NX + c, 1.0 / (8 * NY)]
             for r in range(NY) for c in range(NX)]
    fk = dict(connector='flood', dem=valley(), dx=DX, interval_s=DAY,
              backend='engine', free_edges=[['E', 1e-3]],
              inflow_points=[list(p) for p in pts], max_dt=120.0)
    conns = [{'from': 'src.total_runoff', 'to': 'split.runoff'},
             {'from': 'split.routed', 'to': 'routing.inflow'},
             {'from': 'split.box', 'to': 'flood.runoff', 'mapping': 'box'}]
    maps = dict(box=dict(source='box_cells', target='flood_cells',
                         pairs=pairs))
    rk = dict(connector='routing', net=net, gauge_cells=[MOUTH], area=area,
              cell_uh=[0.3, 0.5, 0.2], site_cells=[ENTRY, RETURN])
    if two_way:
        fk.update(hand_over=True, return_outlets=[['E']],
                  dependencies=['routing'])
        conns += [
            {'from': 'flood.river_request',
             'to': 'routing.withdrawal_request', 'lag': 1,
             'mapping': 'pt_site'},
            {'from': 'routing.withdrawn', 'to': 'flood.inflow',
             'mapping': 'site_pt'},
            {'from': 'flood.returned', 'to': 'routing.injection', 'lag': 1,
             'mapping': 'out_site'}]
        maps.update(
            pt_site=dict(source='flood_inflow', target='river_sites',
                         pairs=[[i, 0, 1.0] for i in range(len(pts))]),
            # site 0 (entry) feeds the points equally; site 1 (return)
            # hands nothing over and maps to point 0
            site_pt=dict(source='river_sites', target='flood_inflow',
                         pairs=[[0, i, 1.0 / len(pts)]
                                for i in range(len(pts))] + [[1, 0, 1.0]]),
            out_site=dict(source='flood_outlets', target='river_sites',
                          pairs=[[0, 1, 1.0]]))
    else:
        fk['inflow_points'] = [list(p) for p in pts]
    fk.update(flood_kw or {})
    cfg = dict(run=dict(device=device, log=False),
               modules=dict(src=dict(connector='two_store'),
                            split=dict(connector='two_store'),
                            routing=rk, flood=fk),
               connections=conns, mappings=maps,
               ledger=dict(enabled=True, on_violation='reject'))
    return TorchVIC.from_dict(cfg, connectors=dict(src=src, split=split))


def combined_residual(m):
    """P + I_boundary - ET - Q_boundary - E_other - dS - closure over the
    coupled system (dS includes the buffers in transit), whole run."""
    led = m.ledger
    c = led.coupled
    s = sum(r['P'] + r['I_boundary'] - r['ET'] - r['Q_boundary'] - r['dS']
            for r in c)
    t = led.totals()
    return s - t['E_other'] - t['closure'], sum(r['P'] for r in c)


# ---------------------------------------------------------------------------
@pytest.mark.parametrize('device', DEVICES)
def test_engine_edge_tracking_changes_nothing(device):
    """track_edges on or off: the same depth, momentum and ledger bit for
    bit; the per-face outflows add up to the ledger's outflow."""
    z = valley()
    res = []
    for tr in (False, True):
        e = FloodEngine(z, DX, free_edges={'E': 1e-3}, max_dt=120.0,
                        inflow_points=flood_points(), device=device,
                        track_edges=tr)
        outs = []
        for k in range(3):
            o = e.run(6 * 3600.0, inflow=np.array([4.0e5, 4.0e5]),
                      runoff=np.full(NY * NX, 0.5))
            outs.append(o)
        res.append((e, outs))
    (a, oa), (b, ob) = res
    for k in ('h', 'Qx', 'Qy', 'qx', 'qy'):
        assert torch.equal(getattr(a, k), getattr(b, k)), k
    for x, y in zip(oa, ob):
        lx = {k: v for k, v in x['ledger'].items()}
        ly = {k: v for k, v in y['ledger'].items() if k != 'edge_outflow'}
        assert lx == ly
        eo = y['edge_out']
        assert float(eo.sum()) == y['ledger']['edge_outflow']
        assert abs(y['ledger']['edge_outflow'] - y['outflow']) <= \
            1e-12 * abs(y['outflow'])
        # only the open east faces carry water out
        e_idx = b.edge_index('E')
        assert float(eo[e_idx].sum()) == pytest.approx(float(eo.sum()),
                                                       rel=1e-15)
    assert ob[-1]['outflow'] > 0


def test_one_way_unchanged_bit_for_bit():
    """Without two-way options the connector declares the one-way variables
    only and its engine does not track edges; with them, the one-way path's
    depth is reached by the same engine arithmetic."""
    f = F.FloodConnector(dem=valley(), dx=DX, backend='engine',
                         free_edges=[('E', 1e-3)], max_dt=120.0,
                         inflow_points=flood_points())
    assert not f.two_way
    assert {v.name for v in f.info().outputs} == {
        'depth', 'max_depth', 'speed', 'outflow'}
    assert [v.name for v in f.info().inputs] == ['runoff', 'inflow']
    assert f.exported_fluxes == ('outflow',)
    # the same intervals through a one-way and a two-way connector (no
    # losses): identical depth, momentum and stored volume
    from torchvic.coupling.connector import Context
    g = F.FloodConnector(dem=valley(), dx=DX, backend='engine',
                         free_edges=[('E', 1e-3)],
                         inflow_points=flood_points(), max_dt=120.0,
                         return_outlets=[['E']])
    ctx = Context(t=0.0, device='cpu', dtype=torch.float64)
    f.initialize(ctx)
    g.initialize(ctx)
    assert not f.engine.track_edges and g.engine.track_edges
    for k in range(3):
        ctx.t = k * DAY
        inp = dict(inflow=torch.tensor([3.0e5, 2.0e5], dtype=torch.float64),
                   runoff=torch.full((NY * NX,), 2.0, dtype=torch.float64))
        pf, pg = f.compute(inp, ctx, DAY), g.compute(inp, ctx, DAY)
        f.commit(pf, ctx)
        g.commit(pg, ctx)
        assert torch.equal(pf.exports['depth'], pg.exports['depth'])
        assert pf.info['stored1'] == pg.info['stored1']
        assert abs(pg.balance.residual()) <= 1e-12 * pg.balance.magnitude()
        assert float(pg.exports['returned'].sum()) == pytest.approx(
            pf.balance.Q_out, rel=1e-12)
    for k in ('h', 'Qx', 'Qy', 'qx', 'qy'):
        assert torch.equal(getattr(f.engine, k), getattr(g.engine, k))


@pytest.mark.parametrize('device', DEVICES)
def test_closed_loop_volume_conservation(device):
    """Rain on every cell for 4 days, then dry until the river and the
    floodplain drain: every module row and every coupled row closes, and
    the whole-system balance (P - Q_boundary - E_other - dS) closes to
    float64 rounding; P = outlet water + water still stored."""
    m = build(device, table=runoff_table(nt=14, wet=4))
    m.initialize().run()
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    R, P = combined_residual(m)
    assert abs(R) <= 1e-12 * P, (R, P)
    fl = m.connector('flood')
    rec = fl.volume_record()
    assert rec['inflow'] if 'inflow' in rec else True
    assert fl.totals['inflow'] > 0 and fl.totals['returned'] > 0
    assert abs(rec['residual']) <= 1e-12 * rec['handed']
    # the water left the system at the mouth (routing's outlet) only
    c = m.ledger.coupled
    Qb = sum(r['Q_boundary'] for r in c)
    rc = m.connector('routing')
    out_mouth = float(rc._totals()[0])
    stored = (sum(r['dS'] for r in c))
    assert abs(Qb - out_mouth - fl.totals['lost']) <= 1e-9 * P
    assert abs(P - Qb - stored - m.ledger.totals()['E_other']) <= 1e-12 * P


def test_water_leaves_and_reenters_the_river():
    """The river leaves routing at the entry site, crosses the floodplain
    in the hydraulic model and re-enters routing at the return site the
    next day: what the router delivered is what the flood model received,
    what the flood model returned is what the router injected (one day
    later), and the mouth sees, after draining, the water routing alone
    would have delivered less what is still on the floodplain."""
    table = runoff_table(nt=14, wet=4)
    m = build('cpu', table=table)
    m.initialize().run()
    tr = m.ledger.transfers

    def legs(q):
        rows = [x for x in tr if x['quantity'] == q]
        return (sum(x['amount'] for x in rows if x.get('leg') == 'sent'),
                sum(x['amount'] for x in rows if x.get('leg') == 'delivered'),
                rows)
    s_w, d_w, _ = legs('withdrawn')
    s_r, d_r, rows_r = legs('returned')
    assert s_w > 0 and abs(s_w - d_w) <= 1e-12 * s_w
    fl, rc = m.connector('flood'), m.connector('routing')
    assert fl.totals['inflow'] == pytest.approx(d_w, rel=1e-12)
    # returned on day k is injected on day k + 1
    assert s_r > 0 and d_r <= s_r
    assert float(rc.router.injected[1]) == pytest.approx(d_r, rel=1e-12)
    assert float(rc.router.withdrawn[0]) == pytest.approx(s_w, rel=1e-12)
    assert float(rc.router.withdrawn[1]) == 0.0
    # the reference: everything routed, no flood box
    net, area = network()
    r, _ = rn.build_step_router(net, [MOUTH], area, DAY,
                                cell_uh=torch.tensor([0.3, 0.5, 0.2],
                                                     dtype=torch.float64))
    for k in range(table.shape[0]):
        r.step(table[k])
    q_ref = float(r.out.sum())
    q_two = float(rc._totals()[0])
    in_transit = float(rc._totals()[1]) + (s_r - d_r)
    on_plain = fl.totals['stored'] - fl.totals['stored_start']
    assert q_two < q_ref
    # all of the reference outlet water is accounted for (mouth + stored)
    assert abs(q_ref - float(r.ring.sum()) * DAY
               - (q_two + in_transit + on_plain + fl.totals['lost'])) \
        <= 1e-9 * q_ref
    # the router's own node ledger still closes with the hooks
    assert float(rc.ledger().abs().max()) <= 1e-9 * q_ref


def test_restart_during_a_flood_is_bit_identical(tmp_path):
    a = build('cpu')
    a.initialize().run()
    b = build('cpu')
    b.initialize().run(until=3 * DAY)
    fb = b.connector('flood')
    assert float(fb.depth.sum()) > 0 and fb.totals['inflow'] > 0
    ck = b.save_checkpoint(str(tmp_path / 'ck.pt'))
    c = build('cpu')
    c.initialize().load_checkpoint(ck)
    c.run()
    fa, fc = a.connector('flood'), c.connector('flood')
    for k in ('h', 'Qx', 'Qy', 'qx', 'qy'):
        assert torch.equal(getattr(fa.engine, k), getattr(fc.engine, k)), k
    assert fa.totals == fc.totals
    ra, rr = a.connector('routing').router, c.connector('routing').router
    for k in ra._STATE:
        assert torch.equal(getattr(ra, k), getattr(rr, k)), k
    assert [r['residual'] for r in a.ledger.balances] == \
        [r['residual'] for r in c.ledger.balances]


def test_rejected_step_rolls_back_flood_and_routing():
    m = build('cpu', reject_at=3 * DAY)
    m.initialize()
    m.run(until=3 * DAY)
    fl, rc = m.connector('flood'), m.connector('routing')
    eng0 = {k: v.clone() for k, v in fl.engine.state_dict().items()
            if torch.is_tensor(v)}
    tot0 = dict(fl.totals)
    rc._rollback()
    ring0 = [x.clone() for x in rc._live()]
    with pytest.raises(StepRejected):
        m.step()
    assert m.runtime.time.t == 3 * DAY
    s1 = fl.engine.state_dict()
    assert all(torch.equal(v, s1[k]) for k, v in eng0.items())
    assert fl.totals == tot0
    rc._rollback()
    assert all(torch.equal(a, b) for a, b in zip(ring0, rc._live()))
    m.connector('split').reject_at = None
    m.run()
    ref = build('cpu')
    ref.initialize().run()
    assert torch.equal(fl.engine.h, ref.connector('flood').engine.h)
    assert torch.equal(rc.router.ring, ref.connector('routing').router.ring)
    assert fl.totals == ref.connector('flood').totals


def test_open_water_evaporation_is_E_other_and_never_overdraws():
    """A closed pond: evaporation takes min(depth, E dt) from cells deeper
    than the threshold, is ledgered as E_other and the depth stays >= 0."""
    from torchvic.coupling.connector import Context
    n = 10
    z = np.full((n, n), 5.0)
    h0 = np.zeros((n, n))
    h0[3:7, 3:7] = 0.004                 # shallower than one day's loss
    h0[4:6, 4:6] = 0.5
    f = F.FloodConnector(dem=z, dx=10.0, backend='engine',
                         open_water_evap_mm_day=5.0, initial_depth=h0,
                         flood_threshold=0.001)
    assert f.two_way and f.losses
    ctx = Context(t=0.0, device='cpu', dtype=torch.float64)
    f.initialize(ctx)
    p = f.compute({}, ctx, DAY)
    f.commit(p, ctx)
    b = p.balance
    assert b.E_other > 0 and b.Q_out == 0.0
    assert abs(b.residual()) <= 1e-12 * b.magnitude()
    d = f.depth.cpu().numpy()
    assert (d >= 0).all()
    # the water spreads inside the interval, so compare volumes: the loss is
    # at most 5 mm over the wet area
    assert b.E_other <= 5e-3 * 100.0 * n * n
    rec = f.volume_record()
    assert rec['evaporation'] == pytest.approx(b.E_other)
    assert abs(rec['residual']) <= 1e-12 * float(h0.sum()) * 100.0


def test_two_way_refusals():
    with pytest.raises(CouplingError, match='engine backend'):
        F.FloodConnector(dem=valley(), dx=DX, solver='acc_cpu',
                         free_edges=[('E', 1e-3)], return_outlets=[['E']])
    with pytest.raises(CouplingError, match='not open'):
        F.FloodConnector(dem=valley(), dx=DX, backend='engine',
                         free_edges=[('E', 1e-3)], return_outlets=[['W']])
    with pytest.raises(CouplingError, match='vic_map'):
        F.FloodConnector(dem=valley(), dx=DX, backend='engine',
                         infiltration_mm_day=10.0)
    with pytest.raises(CouplingError, match='inflow_points'):
        F.FloodConnector(dem=valley(), dx=DX, backend='engine',
                         hand_over=True)


# ---------------------------------------------------------------------------
# VIC (Stehekin) <-> flood: floodplain infiltration through soil_add
# ---------------------------------------------------------------------------
@pytest.mark.reference
def test_floodplain_infiltration_enters_vic_soil_and_balances(monkeypatch):
    import _paths
    stehe = _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                    set_options={'SNOW_BAND': 'FALSE'})
    if not (os.path.exists(stehe) and os.path.isdir(
            os.path.join(_paths.SAMPLES, 'image', 'Stehekin'))):
        pytest.skip("VIC's Stehekin sample is not present")
    monkeypatch.setenv('test_data_dir', _paths.SAMPLES)
    nv = 16
    n = 16                                  # 4 x 4 hydraulic cells per VIC cell
    pairs = []
    for r in range(n):
        for c in range(n):
            k = (r // 4) * 4 + (c // 4)
            pairs.append([k, r * n + c, 1.0 / 16])
    yy, xx = np.mgrid[0:n, 0:n]
    # 1 km cells: the floodplain is about half of each VIC cell's area, so
    # a day's infiltration exceeds the top layer's room (cap -> unmet)
    dem = 10.0 + 1e-4 * np.hypot(xx - n / 2, yy - n / 2) * 1000.0
    h0 = np.where(np.hypot(xx - n / 2, yy - n / 2) < 6, 3.0, 0.0)
    ident = [[k, k, 1.0] for k in range(nv)]
    cfg = dict(
        run=dict(device='cpu', log=False),
        modules=dict(
            vic=dict(global_param=stehe, exchange='day',
                     soil_add_mode='cap', exports=['soil_add_unmet']),
            flood=dict(connector='flood', dem=dem, dx=1000.0,
                       interval_s=DAY, max_dt=300.0,
                       backend='engine', initial_depth=h0,
                       infiltration_mm_day=1000.0,
                       open_water_evap_mm_day=3.0, vic_map=pairs)),
        connections=[
            {'from': 'vic.total_runoff', 'to': 'flood.runoff',
             'mapping': 'v2f'},
            {'from': 'flood.infiltration', 'to': 'vic.soil_add', 'lag': 1,
             'mapping': 'fv2v'},
            {'from': 'vic.soil_add_unmet', 'to': 'flood.infiltration_unmet',
             'mapping': 'v2fv'}],
        mappings=dict(
            v2f=dict(source='vic_cells', target='flood_cells', pairs=pairs),
            fv2v=dict(source='flood_vic', target='vic_cells', pairs=ident),
            v2fv=dict(source='vic_cells', target='flood_vic', pairs=ident)),
        ledger=dict(enabled=True, on_violation='reject'))
    with torch.inference_mode():
        m = TorchVIC.from_dict(cfg).initialize()
        m.run()
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    fl = m.connector('flood')
    T = fl.totals
    assert T['infiltration'] > 0 and T['evaporation'] > 0
    assert T['unmet'] > 0                   # VIC's cap acted and handed back
    vic = m.ledger.totals('vic')
    assert vic['I_ext'] > 0                 # soil_add reached VIC
    R, P = combined_residual(m)
    store = fl.totals['stored_start']
    print('infiltration test: infiltration %.6e unmet %.6e evaporation '
          '%.6e m3, R %.3e of P %.6e + store %.6e'
          % (T['infiltration'], T['unmet'], T['evaporation'], R, P, store))
    assert abs(R) <= 1e-12 * (abs(P) + store), (R, P)
