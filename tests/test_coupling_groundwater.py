"""The groundwater connector: VIC's baseflow as recharge, pumping with
limits, river-aquifer exchange and groundwater ET, in the coupled runtime.
Conservation (module and coupled ledger, and the storage change against the
heads), restart, transactional failure and the ownership rule. The aquifer
is synthetic (no real parameters for these domains): the tests check the
coupling, not a catchment.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                                            # noqa: E402
from torchvic import case as _case                       # noqa: E402
from torchvic.coupling import (TorchVIC, Connector, Balance,  # noqa: E402
                               Proposal, StepRejected, VarSpec, GridSpec)
from torchvic.coupling.connectors.groundwater import (    # noqa: E402
    GroundwaterConnector, ownership_problems)
from torchvic.coupling.toy import TwoStoreConnector      # noqa: E402
from torchvic.coupling.vic import GRID, CaseDriver, VicConnector  # noqa: E402

STEHE = _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                set_options={'SNOW_BAND': 'FALSE'})
STEHE_DATA = os.path.join(_paths.SAMPLES, 'image', 'Stehekin')
DAY = 86400.0
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
NY, NX = 4, 5
N = NY * NX
AREA = 25.0e6                      # m2 of land per cell


def _domain():
    lat, lon = np.meshgrid(40.0 + 0.25 * np.arange(NY),
                           -100.0 + 0.25 * np.arange(NX), indexing='ij')
    return dict(lat=lat.ravel(), lon=lon.ravel(),
                area=np.full(N, AREA), frac=np.ones(N))


class Source(Connector):
    """Stands in for VIC: a constant baseflow (mm per interval) as a flux
    and a pumping request (m3) as a state, every day."""
    name = 'src'
    version = '1'
    backends = ('cpu', 'cuda')
    exported_fluxes = ('baseflow', 'runoff')
    owned_state = ('src_nothing',)

    def __init__(self, baseflow=2.0, runoff=1.0, request=0.0,
                 interval_days=1.0):
        super().__init__()
        self.bf, self.ro, self.req = baseflow, runoff, request
        self.interval_s = interval_days * DAY
        self.outputs = (
            VarSpec('baseflow', 'mm', ('cell',), kind='flux', grid=GRID),
            VarSpec('runoff', 'mm', ('cell',), kind='flux', grid=GRID),
            VarSpec('request', 'm3', ('cell',), kind='state', grid=GRID))
        self.i = 0

    def grids(self):
        return [GridSpec(GRID, N, area=torch.full((N,), AREA,
                                                  dtype=torch.float64))]

    def initialize(self, ctx):
        self.dev = ctx.device

    def compute(self, inputs, ctx, dt):
        f = dt / DAY
        n = self.i
        bf = torch.full((N,), self.bf * f, dtype=torch.float64,
                        device=self.dev)
        ro = torch.full((N,), self.ro * f, dtype=torch.float64,
                        device=self.dev)
        req = self.req(n) if callable(self.req) else self.req
        rq = torch.full((N,), float(req), dtype=torch.float64,
                        device=self.dev)
        a = AREA / 1000.0
        # P = baseflow + runoff leave as Q_out; nothing stored
        bal = Balance(P=float((bf + ro).sum()) * a,
                      Q_out=float((bf + ro).sum()) * a, n_terms=2 * N)
        return Proposal(self.name, ctx.t, ctx.t + dt, state=n + 1,
                        exports=dict(baseflow=bf, runoff=ro, request=rq),
                        balance=bal, backend=str(self.dev))

    def commit(self, proposal, ctx):
        self.i = proposal.state

    def save_state(self):
        return dict(i=self.i)

    def load_state(self, b):
        self.i = b['i']


def _run(days=20, device='cpu', gw=None, src=None, sink=False, conn=None,
         ledger=None, extra_modules=None, **runopts):
    gwcfg = dict(connector='groundwater', domain=_domain())
    gwcfg.update(gw or {})
    mods = dict(src=dict(connector='two_store'), groundwater=gwcfg)
    cons = [{'from': 'src.baseflow', 'to': 'groundwater.recharge'},
            {'from': 'src.request', 'to': 'groundwater.pumping_request'}]
    conns = dict(src=Source(**(src or {})))
    if sink:
        mods['sink'] = dict(connector='two_store')
        conns['sink'] = TwoStoreConnector(
            n=N, interval_s=DAY, grid=GRID, area_m2=AREA, precip=0.0,
            k_ab=1.0, k_out=1.0, a0=0.0, b0=0.0, name='sink')
        cons.append({'from': 'groundwater.pumping', 'to': 'sink.inflow'})
    if conn is not None:
        cons = conn(cons)
    cfg = dict(run=dict(device=device, log=False, duration_days=days),
               modules=mods, connections=cons)
    if ledger:
        cfg['ledger'] = ledger
    return TorchVIC.from_dict(cfg, connectors=conns)


def _gw(m):
    return m.connector('groundwater')


def _heads(m):
    c = _gw(m)
    return c.model.h[0].reshape(-1)[c._flat].clone()


def _storage(c, h):
    """Drainable storage (m3) from the heads: Sy * A * (h - bottom) for
    heads inside the aquifer (independent of the engine's budget)."""
    sy = c.model.sy[0].reshape(-1)[c._flat]
    return float((sy * c.area_d * (h - c.bot)).sum())


def _closes(m):
    for r in m.ledger.balances:
        assert r['ok'], r
    for r in m.ledger.coupled:
        assert r['ok'], r


# ---------------------------------------------------------------------------
@pytest.mark.parametrize('device', DEVICES)
def test_recharge_discharge_storage_close(device):
    # recharge enough to lift the water table into the drains
    gw = dict(aquifer=dict(k=1.0, sy=0.1, ss=0.0, thickness=50.0,
                           depth_to_water=3.0), drain=dict(cond=1.0e6))
    m = _run(20, device, gw=gw, src=dict(baseflow=8.0),
             ledger=dict(on_violation='reject'))
    c = _gw(m)
    m.initialize()
    h0 = _heads(m)
    s0 = _storage(c, h0)
    m.run()
    _closes(m)
    t = m.ledger.totals('groundwater')
    rech = 8.0 * 20 / 1000.0 * AREA * N
    # all the recharge the source sent arrived (internal transfer, counted once)
    assert t['I_ext'] == pytest.approx(rech, rel=1e-12)
    # the source's runoff is the only boundary outflow besides discharge
    ds_heads = _storage(c, _heads(m)) - s0
    assert t['dS'] == pytest.approx(ds_heads, rel=1e-9, abs=1e-6)
    # recharge = storage gain + discharge to rivers (no pumping, no ET)
    assert t['I_ext'] - t['Q_out'] == pytest.approx(t['dS'], rel=1e-8)
    assert t['Q_out'] > 0.05 * rech           # the drains discharge
    cp = m.ledger.coupled[-1]
    assert abs(cp['residual']) <= cp['tolerance']


def test_runoff_is_not_routed_twice_rule():
    cfg = dict(modules=dict(vic={}, groundwater={}, routing={}),
               connections=[
                   {'from': 'vic.baseflow', 'to': 'groundwater.recharge'},
                   {'from': 'vic.total_runoff', 'to': 'routing.inflow'}])
    assert any('counted twice' in p for p in ownership_problems(cfg))
    cfg['connections'][1]['from'] = 'vic.runoff'
    cfg['connections'].append({'from': 'groundwater.discharge',
                               'to': 'routing.inflow'})
    assert ownership_problems(cfg) == []
    cfg['connections'][0]['from'] = 'vic.total_runoff'
    assert any('recharge candidate' in p for p in ownership_problems(cfg))


@pytest.mark.parametrize('device', DEVICES)
def test_pumping_is_limited_by_the_water_above_the_bottom(device):
    # thin aquifer, a request far beyond what it holds, every day
    big = 5.0e8
    gw = dict(aquifer=dict(k=1.0, sy=0.1, thickness=10.0, depth_to_water=2.0))
    m = _run(15, device, gw=gw, src=dict(baseflow=0.5, request=big),
             sink=True)
    c = _gw(m)
    m.initialize()
    delivered = 0.0
    for _ in range(15):
        m.step()
        h = _heads(m)
        assert bool((h >= c.bot - 1e-6).all())          # never below bottom
    _closes(m)
    t = m.ledger.totals('groundwater')
    req = big * N * 15
    out = m.outputs('groundwater')
    pumped = float(out['pumping'].sum())
    unmet = float(out['pumping_unmet'].sum())
    assert pumped > 0 and pumped < 0.5 * req
    assert pumped + unmet == pytest.approx(req, rel=1e-9)
    # what was delivered arrived at the sink (internal transfer)
    snk = m.ledger.totals('sink')
    assert snk['I_ext'] == pytest.approx(pumped, rel=1e-9)
    # it cannot exceed the water it had: start storage + recharge
    s0 = N * 0.1 * AREA * (10.0 - 2.0)
    assert pumped <= s0 + 0.5 / 1000.0 * AREA * N * 15 + 1e-6
    # a modest request is delivered in full
    m2 = _run(10, device, gw=gw, src=dict(baseflow=0.5, request=1.0e4),
              sink=True)
    m2.run()
    _closes(m2)
    o2 = m2.outputs('groundwater')
    assert float(o2['pumping_unmet'].sum()) == pytest.approx(0.0, abs=1e-3)
    assert float(o2['pumping'].sum()) == pytest.approx(1.0e4 * N * 10,
                                                       rel=1e-6)


def test_river_aquifer_exchange_and_et_close():
    # a river 1 m below the surface: gains where the water table is higher,
    # loses where lower; with groundwater ET above the water table
    riv = dict(cond=2.0e4, stage=-6.0, bottom=-8.0)
    gw = dict(aquifer=dict(k=2.0, sy=0.1, thickness=40.0, depth_to_water=2.0),
              river=riv, evt=dict(rate=2.0e-3, depth=3.0))
    m = _run(25, 'cpu', gw=gw, src=dict(baseflow=1.0))
    m.run()
    _closes(m)
    t = m.ledger.totals('groundwater')
    assert t['ET'] > 0                              # ET took water
    row = [r for r in m.ledger.balances if r['module'] == 'groundwater'][0]
    assert row['detail_river_out'] > 0 or row['detail_river_in'] > 0
    # signed discharge = drains + river gains - river losses
    out = m.outputs('groundwater')
    assert torch.isfinite(out['discharge']).all()
    # the water table moved and the balance still holds with dS from heads
    c = _gw(m)
    assert t['I_ext'] - t['Q_out'] - t['ET'] == pytest.approx(
        t['dS'], rel=1e-8, abs=1e-6)


@pytest.mark.parametrize('device', DEVICES)
def test_restart_is_the_continuous_run(device, tmp_path):
    def mk():
        return _run(12, device, gw=dict(aquifer=dict(
            k=1.0, sy=0.1, thickness=30.0, depth_to_water=3.0)),
            src=dict(baseflow=1.0, request=2.0e4), sink=True)
    a = mk()
    a.run()
    b = mk()
    b.initialize()
    for _ in range(5):
        b.step()
    p = str(tmp_path / 'ck.pt')
    b.save_checkpoint(p)
    c = mk()
    c.initialize()
    c.runtime.load_checkpoint(p)
    c.run()
    assert torch.equal(_heads(a), _heads(c))
    assert torch.equal(a.outputs('groundwater')['pumping'][-1],
                       c.outputs('groundwater')['pumping'][-1])
    assert a.ledger.totals('groundwater') == pytest.approx(
        c.ledger.totals('groundwater'), rel=0, abs=1e-6)


def test_a_failed_step_changes_nothing():
    gw = dict(aquifer=dict(k=1.0, sy=0.1, thickness=30.0),
              solver=dict(outer_maximum=1, inner_maximum=1))
    m = _run(5, 'cpu', gw=gw)
    m.initialize()
    h0 = _heads(m)
    with pytest.raises(StepRejected):
        m.step()
    assert torch.equal(_heads(m), h0)
    assert m.runtime.time.t == 0 and not m.ledger.balances


def test_compute_does_not_change_the_accepted_state():
    m = _run(3, 'cpu')
    m.initialize()
    c = _gw(m)
    h0 = _heads(m)
    from torchvic.coupling import Context
    ctx = Context(t=0.0, device=torch.device('cpu'), dtype=torch.float64)
    rec = torch.full((N,), 2.0, dtype=torch.float64)
    p1 = c.compute(dict(recharge=rec), ctx, DAY)
    assert torch.equal(_heads(m), h0)
    p2 = c.compute(dict(recharge=rec), ctx, DAY)
    assert torch.equal(p1.state['h'], p2.state['h'])
    c.commit(p1, ctx)
    assert torch.equal(_heads(m), p1.state['h'][0].reshape(-1)[c._flat])


def test_synthetic_parameters_are_announced():
    msgs = []
    m = TorchVIC.from_dict(
        dict(run=dict(device='cpu', log=False, duration_days=1),
             modules=dict(src=dict(connector='two_store'),
                          groundwater=dict(connector='groundwater',
                                           domain=_domain())),
             connections=[{'from': 'src.baseflow',
                           'to': 'groundwater.recharge'}]),
        connectors=dict(src=Source()), log=msgs.append)
    m.initialize()
    assert any('SYNTHETIC' in str(x) for x in msgs)
    assert _gw(m).synthetic


def test_a_losing_river_gives_water_to_the_aquifer():
    gw = dict(aquifer=dict(k=2.0, sy=0.1, thickness=40.0, depth_to_water=6.0),
              river=dict(cond=2.0e4, stage=-1.0, bottom=-2.0))
    m = _run(10, 'cpu', gw=gw, src=dict(baseflow=0.0))
    m.run()
    _closes(m)
    row = [r for r in m.ledger.balances if r['module'] == 'groundwater']
    assert sum(r['detail_river_in'] for r in row) > 0
    assert float(m.outputs('groundwater')['discharge'].sum()) < 0
    t = m.ledger.totals('groundwater')
    assert t['dS'] > 0                       # the river fed the aquifer


# ---------------------------------------------------------------------------
# VIC + groundwater on VIC's Stehekin sample (16 cells, 10 days)
# ---------------------------------------------------------------------------
@pytest.fixture
def _stehe_env(monkeypatch):
    monkeypatch.setenv('test_data_dir', _paths.SAMPLES)
    if not (os.path.exists(STEHE) and os.path.isdir(STEHE_DATA)):
        pytest.skip("VIC's Stehekin sample is not present")


def _vic_gw(device, window_days=5, gw=None, routing_runoff=True,
            sink=False, **vic):
    c = _case.load(STEHE, device=device, log=lambda *a: None)
    mods = dict(vic=dict(dict(global_param=STEHE, window_days=window_days),
                         **vic),
                groundwater=dict(dict(connector='groundwater', domain=c.dom,
                                      interval_days=1.0), **(gw or {})))
    conns = dict(vic=VicConnector(driver=CaseDriver(c), window_days=window_days,
                                  **vic))
    cons = [{'from': 'vic.baseflow', 'to': 'groundwater.recharge'}]
    if sink:
        a = torch.as_tensor(c.dom.area * c.dom.frac, dtype=torch.float64)
        mods['sink'] = dict(connector='two_store')
        conns['sink'] = TwoStoreConnector(
            n=c.n_cells, interval_s=DAY, grid=GRID, area_m2=1.0, precip=0.0,
            k_ab=1.0, k_out=1.0, a0=0.0, b0=0.0, name='sink')
        conns['sink'].grids = lambda a=a, n=c.n_cells: [
            GridSpec(GRID, n, area=a)]
        cons.append({'from': 'groundwater.discharge', 'to': 'sink.inflow'})
    m = TorchVIC.from_dict(dict(run=dict(device=device, log=False),
                                modules=mods, connections=cons),
                           connectors=conns)
    return c, m


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
def test_vic_with_groundwater_on_stehekin(device, _stehe_env):
    gw = dict(aquifer=dict(k=2.0, sy=0.1, ss=0.0, thickness=60.0,
                           depth_to_water=4.0), drain=dict(cond=5.0e4))
    c, m = _vic_gw(device, gw=gw, sink=True)
    m.initialize()
    g = _gw(m)
    s0 = _storage(g, _heads(m))
    m.run()
    # VIC does not feel the aquifer: the run is VIC's own, value for value
    alone = _case.load(STEHE, device=device, log=lambda *a: None)
    ref = alone.run(return_state=False)
    out = m.outputs('vic')
    for k in ('runoff', 'baseflow', 'et'):
        assert torch.allclose(out[k], ref[k], rtol=0, atol=1e-9), k
    # every ledger row closed, VIC's, the aquifer's and the coupled domain's
    _closes(m)
    area = torch.as_tensor(c.dom.area * c.dom.frac, dtype=torch.float64)
    bf = float((out['baseflow'].cpu().sum(0) * area).sum()) / 1000.0
    t = m.ledger.totals('groundwater')
    # the baseflow VIC computed is the recharge the aquifer received, once
    assert t['I_ext'] == pytest.approx(bf, rel=1e-12)
    # and the sink received the aquifer's discharge, once
    sk = m.ledger.totals('sink')
    assert sk['I_ext'] == pytest.approx(t['Q_out'], rel=1e-9, abs=1e-6)
    ds_heads = _storage(g, _heads(m)) - s0
    assert t['dS'] == pytest.approx(ds_heads, rel=1e-8, abs=1e-6)
    assert t['I_ext'] - t['Q_out'] == pytest.approx(t['dS'], rel=1e-8,
                                                    abs=1e-6)
    assert bf > 0
    # VIC's ledger: baseflow left VIC to the aquifer (a transfer), runoff
    # left the domain
    cp = m.ledger.coupled
    assert cp and all(r['ok'] for r in cp)
    sent = sum(x['amount'] for x in m.ledger.transfers
               if x['source'] == 'vic' and x['leg'] == 'sent')
    assert sent == pytest.approx(bf, rel=1e-12)


@pytest.mark.reference
def test_vic_gw_window_and_daily_exchange_agree_in_total(_stehe_env):
    gw = dict(aquifer=dict(k=2.0, sy=0.1, ss=0.0, thickness=60.0,
                           depth_to_water=4.0), drain=dict(cond=5.0e4))
    tot = []
    for kw in (dict(window_days=5), dict(window_days=0, exchange='day')):
        c, m = _vic_gw('cpu', gw=gw, **kw)
        m.run()
        _closes(m)
        tot.append(m.ledger.totals('groundwater')['I_ext'])
    assert tot[0] == pytest.approx(tot[1], rel=1e-12)


def test_yaml_validates_and_names_the_baseflow_once():
    import yaml
    from torchvic.coupling import validate_config
    cfgp = os.path.join(_paths.REPO, 'configs', 'vic_groundwater.yaml')
    if os.path.exists(_paths.SAMPLE_DATA):
        os.environ.setdefault('TORCHVIC_SAMPLE_DATA', _paths.SAMPLE_DATA)
    raw = yaml.safe_load(open(cfgp))
    assert ownership_problems(raw) == []
    problems = validate_config(cfgp)
    assert not [p for p in problems if 'groundwater' in p], problems


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
@pytest.mark.parametrize('every', [1, 3])
def test_deferred_convergence_redoes_a_step_that_needs_more_chunks(every):
    """After the synchronous warm-up steps the engine launches the chunks
    the hardest step needed without reading the flag; a step that needs more
    is redone synchronously by the runtime and the result equals the
    all-synchronous run."""
    gw = dict(aquifer=dict(k=1.0, sy=0.1, ss=0.0, thickness=50.0,
                           depth_to_water=3.0), drain=dict(cond=1.0e6),
              solver=dict(outer_per_check=1))
    led = dict(check_every=every)
    ref = _run(10, 'cuda', gw=dict(gw, async_convergence=False),
               src=dict(baseflow=6.0), ledger=led)
    ref.initialize().run()
    m = _run(10, 'cuda', gw=gw, src=dict(baseflow=6.0), ledger=led)
    m.initialize()
    m.run(until=4 * DAY)
    c = _gw(m)
    assert c.model._ahead >= 1
    c.model._ahead = 1                    # fewer chunks than the step needs
    m.run()
    assert getattr(c, 'redone', 0) >= 1     # the redo path was taken
    assert torch.equal(_heads(m).cpu(), _heads(ref).cpu())
    assert len(m.ledger.balances) == len(ref.ledger.balances)
    for r, q in zip(m.ledger.balances, ref.ledger.balances):
        assert r['ok'] and r['residual'] == q['residual']
