"""The flood connector (one way): VIC runoff and routed inflow into the
TorchVIC flood engine (default backend, in process) or LISFLOOD-FP as an
external process (backend lisflood); the generic cases run for both. Controlled volumes, depth -> volume
conversion with the contributing area, the VIC -> hydraulic mapping,
boundary-flow accounting and restart equality. The CPU solver runs
everywhere the binary exists; the CUDA (ACC GPU) cases need a GPU."""
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('pandas')
pytest.importorskip('netCDF4')

from torchvic.coupling import (CouplingError, Connector, TorchVIC,  # noqa: E402
                               VarSpec, GridSpec, registry)
from torchvic.coupling.connector import Balance, Context, Proposal  # noqa: E402
from torchvic.coupling.connectors import flood as F             # noqa: E402

HAS_BIN = os.path.exists(F.DEFAULT_BINARY)
needs_bin = pytest.mark.skipif(not HAS_BIN, reason='LISFLOOD-FP binary not '
                               'built (LISFLOOD_BIN)')
HAS_GPU = torch.cuda.is_available()
DAY = 86400.0
N, DX = 24, 20.0


def bowl(n=N, slope=0.002):
    yy, xx = np.mgrid[0:n, 0:n]
    return 10.0 + slope * np.hypot(xx - n / 2, yy - n / 2) * DX


def plane(n=N, slope=0.002):
    """falls to the east (columns), so a FREE east edge drains it"""
    xx = np.arange(n)[None, :] * np.ones((n, 1))
    return 10.0 - slope * xx * DX


class Source(Connector):
    """Stand-in for VIC: constant mm per interval on a 'vic_cells' grid
    with real cell areas, balanced as precipitation that leaves as runoff."""
    name = 'src'
    imported_fluxes = ()
    exported_fluxes = ('runoff',)
    backends = ('cpu', 'cuda')

    def __init__(self, mm, area, interval_s=DAY, **c):
        super().__init__(**c)
        self.mm = torch.as_tensor(mm, dtype=torch.float64)
        self.area = torch.as_tensor(area, dtype=torch.float64)
        self.interval_s = interval_s
        self.outputs = (VarSpec('runoff', 'mm', ('cell',), kind='flux',
                                grid='vic_cells'),)

    def grids(self):
        return [GridSpec('vic_cells', len(self.mm), area=self.area)]

    def compute(self, inputs, context, dt):
        v = float((self.mm * 1e-3 * self.area).sum())
        return Proposal('src', context.t, context.t + dt, state=None,
                        exports=dict(runoff=self.mm.clone()),
                        balance=Balance(P=v, Q_out=v, n_terms=len(self.mm),
                                        scale=2 * v),
                        backend='cpu')

    def commit(self, proposal, context):
        pass


@pytest.fixture(params=['engine', 'lisflood'])
def bk(request):
    if request.param == 'lisflood' and not HAS_BIN:
        pytest.skip('LISFLOOD-FP binary not built (LISFLOOD_BIN)')
    return request.param


def _flood(dem=None, interval_s=DAY, backend='lisflood', **kw):
    dem = bowl() if dem is None else dem
    if backend == 'engine':
        return F.FloodConnector(dem=dem, dx=DX, interval_s=interval_s,
                                backend='engine', **kw)
    return F.FloodConnector(dem=dem, dx=DX, interval_s=interval_s,
                            solver=kw.pop('solver', 'acc_cpu'), **kw)


def _cfg_flood(bk):
    return (dict(backend='engine') if bk == 'engine'
            else dict(solver='acc_cpu'))


def _ctx(t=0.0, device='cpu'):
    return Context(t=t, device=device, dtype=torch.float64)


def _run(flood, V=None, Vp=None, dt=200.0, steps=1, device='cpu'):
    ctx = _ctx(device=device)
    flood.initialize(ctx)
    props = []
    for k in range(steps):
        ctx.t = k * dt
        inp = {}
        if V is not None:
            inp['runoff'] = torch.as_tensor(V, dtype=torch.float64)
        if Vp is not None:
            inp['inflow'] = torch.as_tensor(Vp, dtype=torch.float64)
        p = flood.compute(inp, ctx, dt)
        from torchvic.coupling.connector import resolve_problems
        assert resolve_problems(flood.validate(p, ctx)) == []
        flood.commit(p, ctx)
        props.append(p)
    return props


# ---------------------------------------------------------------------------
def test_registered_and_declares_its_variables(bk):
    assert 'flood' in registry.available()
    f = _flood(inflow_points=[(3, 3)], backend=bk)
    assert f.backend == bk
    i = f.info()
    assert [v.name for v in i.inputs] == ['runoff', 'inflow']
    assert {v.name for v in i.outputs} == {'depth', 'max_depth', 'speed', 'outflow'}
    assert i.input('runoff').units == 'm3' and i.input('runoff').kind == 'flux'
    assert i.output('depth').kind == 'state'
    assert set(i.imported_fluxes) == {'runoff', 'inflow'}
    g = {x.name: x for x in f.grids()}
    assert g['flood_cells'].n == N * N
    assert float(g['flood_cells'].area[0]) == DX * DX


def test_controlled_volume_closed_basin_known_to_rounding(bk):
    """Known volume in (distributed + point), closed edges: everything is
    stored. Checked against the rasters and the ledger balance."""
    rng = np.random.default_rng(1)
    V = rng.uniform(0.0, 20.0, N * N)
    Vp = np.array([5000.0])
    f = _flood(inflow_points=[(N // 2, N // 2)], backend=bk)
    (p,) = _run(f, V, Vp, dt=300.0)
    exp = V.sum() + Vp.sum()
    stored = float(p.exports['depth'].sum()) * DX * DX
    assert abs(stored - exp) <= 1e-9 * exp
    assert abs(p.balance.residual()) <= 1e-9 * exp
    assert p.balance.Q_out == 0.0
    if bk == 'lisflood':
        # the .mass record saw the handed point inflow
        assert abs(p.info['mass_qin'] - Vp.sum()) <= 1e-4 * Vp.sum()
    else:
        assert abs(p.balance.residual()) <= 1e-13 * exp
    rec = f.volume_record()
    assert abs(rec['handed'] - exp) <= 1e-12 * exp
    assert abs(rec['residual']) <= 1e-9 * exp


@needs_bin
def test_inflow_unit_is_m3_per_s_over_cell_width(tmp_path):
    """P x y QFIX is per unit width: the connector writes Q / dx."""
    f = _flood(inflow_points=[(5, 7)], keep_work=True,
               workdir=str(tmp_path))
    _run(f, None, np.array([1200.0]), dt=100.0)
    bci = (tmp_path / 'step_00000' / 'case.bci').read_text().split()
    assert bci[0] == 'P'
    x, y, kind, q = float(bci[1]), float(bci[2]), bci[3], float(bci[4])
    assert kind == 'QFIX' and abs(q - (1200.0 / 100.0) / DX) < 1e-12
    # cell (row 5 from the top, col 7): x = (7+.5) dx, y from the bottom
    assert (x, y) == ((7 + .5) * DX, (N - 1 - 5 + .5) * DX)


def test_point_source_and_rain_land_in_the_right_cells(bk):
    V = np.zeros(N * N)
    V[2 * N + 20] = 50.0                         # row 2, col 20
    flood_rc = (N - 3, 4)
    f = _flood(dem=np.full((N, N), 10.0), inflow_points=[flood_rc],
               backend=bk)
    (p,) = _run(f, V, np.array([300.0]), dt=60.0)
    d = p.exports['depth'].reshape(N, N).cpu().numpy()
    assert np.unravel_index(np.argmax(d), d.shape) in ((2, 20), flood_rc)
    assert d[2, 20] > 0 and d[flood_rc] > 0
    assert d[N - 3, 20] == 0 and d[2, 4] == 0
    assert abs(d.sum() * DX * DX - 350.0) < 1e-8


def test_vic_runoff_depth_to_volume_through_the_runtime(tmp_path, bk):
    """VIC-like mm on 4 cells with different areas -> volume with each
    cell's contributing area -> hydraulic cells by the conservative mapping
    -> stored volume = sum(mm x area)."""
    mm = np.array([1.5, 0.0, 4.0, 10.0])
    area = np.array([4.1e6, 4.0e6, 3.9e6, 4.05e6])
    n = 16
    dem = np.full((n, n), 20.0)
    # hydraulic cell -> VIC cell by quadrant; equal share inside a quadrant
    pairs = []
    quad = lambda r, c: (r // 8) * 2 + (c // 8)
    cnt = np.bincount([quad(r, c) for r in range(n) for c in range(n)])
    for r in range(n):
        for c in range(n):
            pairs.append([quad(r, c), r * n + c, 1.0 / cnt[quad(r, c)]])
    cfg = dict(
        run=dict(device='cpu', log=False, duration_days=2),
        modules=dict(
            vic=dict(connector='src'),
            flood=dict(connector='flood', dem=dem, dx=DX,
                       workdir=str(tmp_path), **_cfg_flood(bk))),
        connections=[{'from': 'vic.runoff', 'to': 'flood.runoff',
                      'mapping': 'to_flood'}],
        mappings=dict(to_flood=dict(source='vic_cells',
                                    target='flood_cells', pairs=pairs)))
    reg = registry
    reg.register('src', lambda **k: Source(mm, area, **k), 'test',
                 replace=True)
    try:
        m = TorchVIC.from_dict(cfg).initialize()
        m.run()
        fl = m.connector('flood')
        exp = float((mm * 1e-3 * area).sum()) * 2          # two days
        stored = float(fl.depth.sum()) * DX * DX
        assert abs(stored - exp) <= 1e-9 * exp, (stored, exp)
        assert m.ledger.balances and all(r['ok'] for r in m.ledger.balances)
        assert all(r['ok'] for r in m.ledger.coupled)
        rec = fl.volume_record()
        assert abs(rec['handed'] - exp) <= 1e-9 * exp
    finally:
        reg.unregister('src')


def test_mapping_helpers_are_conservative_and_refuse_empty_cells():
    n = 40
    meta = dict(nx=n, ny=n, dx=500.0, lon0=91.0, lat0=26.0,
                m_lon=100000.0, m_lat=110000.0)
    vlon = np.array([91.03125, 91.09375])
    vlat = np.array([26.03125, 26.03125])
    s, t, fr = F.vic_pairs(meta, vlon, vlat)
    tot = np.bincount(s, weights=fr, minlength=2)
    assert np.allclose(tot, 1.0, atol=1e-14, rtol=0)
    assert len(np.unique(t)) == len(t)            # one owner per cell
    # a VIC cell that no hydraulic cell falls in is refused
    with pytest.raises(CouplingError, match='no hydraulic cell'):
        F.vic_pairs(meta, np.array([91.03125, 91.59375]),
                    np.array([26.03125, 26.5]))


def test_engine_boundary_outflow_from_its_ledger():
    """Open east edge, engine backend: the outflow is the engine's sum of
    the boundary face fluxes, and the interval balance closes to float64
    rounding; closed edges let nothing out."""
    f = _flood(dem=plane(), inflow_points=[(N // 2, 2)],
               free_edges=[('E', 0.002)], backend='engine')
    props = _run(f, None, np.array([4000.0]), dt=400.0, steps=4)
    rec = f.volume_record()
    assert rec['outflow'] > 0.03 * rec['handed']
    assert abs(rec['residual']) <= 1e-12 * rec['handed']
    for p in props:
        assert p.info['outflow_mode'] == 'ledger'
        assert abs(p.balance.residual()) <= 1e-13 * p.balance.magnitude()
    g = _flood(dem=plane(), inflow_points=[(N // 2, 2)], backend='engine')
    (pg,) = _run(g, None, np.array([4000.0]), dt=400.0)
    assert pg.balance.Q_out == 0.0
    assert abs(pg.balance.residual()) <= 1e-12 * 4000.0


def test_engine_momentum_kept_across_intervals():
    """The engine keeps the face momentum between intervals (LISFLOOD-FP
    restarts from rest): two 200 s intervals stay closer to one 400 s run
    than the same two intervals restarted from rest."""
    kw = dict(dem=plane(), inflow_points=[(N // 2, 2)],
              free_edges=[('E', 0.002)], backend='engine')
    a = _flood(**kw)
    _run(a, None, np.array([4000.0]), dt=400.0)
    b = _flood(**kw)
    _run(b, None, np.array([2000.0]), dt=200.0, steps=2)
    assert float(b.engine.qx.abs().max()) > 0
    c = _flood(**kw)
    ctx = _ctx()
    c.initialize(ctx)
    for k in range(2):
        ctx.t = 200.0 * k
        for x in (c.engine.qx, c.engine.qy):
            x.zero_()                          # restart from rest
        p = c.compute(dict(inflow=torch.tensor([2000.0])), ctx, 200.0)
        c.commit(p, ctx)
    kept = float((a.depth - b.depth).abs().max())
    reset = float((a.depth - c.depth).abs().max())
    assert kept < reset


def test_engine_rejected_step_leaves_state():
    """compute alone (a step the runtime rejects) changes nothing."""
    f = _flood(inflow_points=[(3, 3)], backend='engine')
    _run(f, None, np.array([1000.0]), dt=100.0)
    s0 = {k: v.clone() for k, v in f.engine.state_dict().items()
          if torch.is_tensor(v)}
    f.compute(dict(inflow=torch.tensor([5000.0])), _ctx(t=100.0), 100.0)
    s1 = f.engine.state_dict()
    assert all(torch.equal(v, s1[k]) for k, v in s0.items())


def test_upstream_discharge_from_routing_nodes():
    """m3 s-1 at the listed gauge nodes x dt, shared by the inflow points."""
    f = _flood(dem=np.full((N, N), 10.0), inflow_points=[(5, 5), (6, 5)],
               inflow_nodes=[1], backend='engine')
    assert [v.name for v in f.info().inputs][-1] == 'upstream_discharge'
    ctx = _ctx()
    f.initialize(ctx)
    p = f.compute(dict(upstream_discharge=torch.tensor([7.0, 3.0, 9.0])),
                  ctx, 100.0)
    assert p.info['I_inflow'] == pytest.approx(300.0, rel=1e-15)
    assert float(p.exports['depth'].sum()) * DX * DX == pytest.approx(
        300.0, rel=1e-12)


@needs_bin
def test_boundary_flow_accounting_free_edge():
    """Open east edge: water leaves; the outflow derived from the rasters
    agrees with the CPU solver's own .mass record (checked in validate and
    here), and the run's volumes close."""
    f = _flood(dem=plane(), inflow_points=[(N // 2, 2)],
               free_edges=[('E', 0.002)])
    props = _run(f, None, np.array([4000.0]), dt=400.0, steps=4)
    rec = f.volume_record()
    assert rec['outflow'] > 0.03 * rec['handed']
    assert abs(rec['residual']) <= 1e-9 * rec['handed']
    mass_out = sum(p.info['mass_qout'] for p in props)
    assert abs(mass_out - rec['outflow']) <= 5e-3 * rec['handed']
    for p in props:
        assert p.info['outflow_mode'] == 'derived'
        assert abs(p.balance.residual()) <= 1e-9 * p.balance.I_ext
    # closed edges: nothing may leave and the balance is a real check
    g = _flood(dem=plane(), inflow_points=[(N // 2, 2)])
    (pg,) = _run(g, None, np.array([4000.0]), dt=400.0)
    assert pg.info['outflow_mode'] == 'closed' and pg.balance.Q_out == 0.0
    assert abs(pg.balance.residual()) <= 1e-9 * 4000.0


def test_negative_or_bad_water_is_refused_and_state_untouched(bk):
    f = _flood(inflow_points=[(3, 3)], backend=bk)
    ctx = _ctx()
    f.initialize(ctx)
    before = f.depth.clone()
    for bad in (dict(runoff=-torch.ones(N * N)),
                dict(runoff=torch.full((N * N,), float('nan'))),
                dict(inflow=torch.tensor([-1.0]))):
        if bk == 'engine':
            # device flags: refused at validate (nothing is committed)
            from torchvic.coupling.connector import resolve_problems
            prop = f.compute(bad, ctx, 100.0)
            pr = resolve_problems(list(f.validate(prop, ctx))
                                  + list(prop.flags))
            assert any('finite and >= 0' in x for x in pr), pr
            continue
        with pytest.raises(Exception, match='finite and >= 0'):
            f.compute(bad, ctx, 100.0)
    assert torch.equal(f.depth, before)
    with pytest.raises(CouplingError, match='outside'):
        F.FloodConnector(dem=bowl(), dx=DX, inflow_points=[(N, 0)])
    with pytest.raises(CouplingError, match='two inflow points'):
        F.FloodConnector(dem=bowl(), dx=DX, inflow_points=[(1, 1), (1, 1)])


def test_restart_during_a_flood_equals_the_continuous_run(tmp_path, bk):
    """Checkpoint with water on the ground and a flood still growing,
    resume in a new process-like object: depth identical bit for bit."""
    def build():
        mm = np.array([2.0, 3.0, 0.5, 6.0])
        area = np.full(4, 2.0e6)
        n = 12
        dem = np.full((n, n), 20.0) + 0.01 * np.arange(n)[None, :] * DX
        quad = lambda r, c: (r // 6) * 2 + (c // 6)
        cnt = np.bincount([quad(r, c) for r in range(n) for c in range(n)])
        pairs = [[quad(r, c), r * n + c, 1.0 / cnt[quad(r, c)]]
                 for r in range(n) for c in range(n)]
        registry.register('src', lambda **k: Source(mm, area, **k), 'test',
                          replace=True)
        cfg = dict(run=dict(device='cpu', log=False, duration_days=4),
                   modules=dict(
                       vic=dict(connector='src', interval_s=DAY),
                       flood=dict(connector='flood', dem=dem, dx=DX,
                                  interval_s=DAY,
                                  workdir=str(tmp_path / 'w'),
                                  **_cfg_flood(bk))),
                   connections=[{'from': 'vic.runoff', 'to': 'flood.runoff',
                                 'mapping': 'm'}],
                   mappings=dict(m=dict(source='vic_cells',
                                        target='flood_cells', pairs=pairs)))
        return TorchVIC.from_dict(cfg).initialize()
    try:
        a = build()
        a.run()
        b = build()
        b.run(until=2 * DAY)
        mid = b.connector('flood').depth.clone()
        assert float(mid.sum()) > 0
        ck = b.save_checkpoint(str(tmp_path / 'ck.pt'))
        c = build().load_checkpoint(ck)
        assert torch.equal(c.connector('flood').depth, mid)
        c.run()
        assert torch.equal(a.connector('flood').depth,
                           c.connector('flood').depth)
        assert a.connector('flood').totals == c.connector('flood').totals
        assert len(c.ledger.balances) == len(a.ledger.balances)
    finally:
        registry.unregister('src')


def test_hydraulics_do_not_feed_back_one_way(bk):
    """Nothing in the connector's declaration can modify VIC."""
    f = _flood(backend=bk)
    outs = {v.name for v in f.info().outputs}
    assert outs == {'depth', 'max_depth', 'speed', 'outflow'}
    assert 'soil_add' not in outs and 'precip_add' not in outs


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason='needs a GPU')
def test_gpu_engine_controlled_volume_and_cpu_agreement():
    """The engine on the GPU (fused Triton step): closed-basin volume to
    rounding and the CPU engine's depth field to round-off."""
    rng = np.random.default_rng(5)
    V = rng.uniform(0.0, 20.0, N * N)
    Vp = np.array([5000.0])
    g = _flood(inflow_points=[(N // 2, N // 2)], backend='engine')
    c = _flood(inflow_points=[(N // 2, N // 2)], backend='engine')
    (pg,) = _run(g, V, Vp, dt=300.0, device='cuda')
    (pc,) = _run(c, V, Vp, dt=300.0)
    exp = V.sum() + Vp.sum()
    assert pg.exports['depth'].device.type == 'cuda'
    assert abs(float(pg.exports['depth'].sum()) * DX * DX - exp) <= 1e-12 * exp
    assert float((pg.exports['depth'].cpu() - pc.exports['depth'])
                 .abs().max()) <= 1e-10


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason='needs a GPU')
def test_gpu_engine_mixed_controlled_volume_and_float64_agreement():
    """dtype 'mixed' (float32 faces, float64 depth): the closed-basin volume
    to rounding and the float64 engine's depth to 1e-6 m."""
    rng = np.random.default_rng(5)
    V = rng.uniform(0.0, 20.0, N * N)
    Vp = np.array([5000.0])
    g = _flood(inflow_points=[(N // 2, N // 2)], backend='engine',
               dtype='mixed')
    c = _flood(inflow_points=[(N // 2, N // 2)], backend='engine')
    (pg,) = _run(g, V, Vp, dt=300.0, device='cuda')
    (pc,) = _run(c, V, Vp, dt=300.0, device='cuda')
    exp = V.sum() + Vp.sum()
    assert g.engine.mixed and pg.balance is not None
    assert abs(float(pg.exports['depth'].sum()) * DX * DX - exp) <= 1e-12 * exp
    assert float((pg.exports['depth'] - pc.exports['depth'])
                 .abs().max()) <= 1e-6


@needs_bin
@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason='needs a GPU')
def test_gpu_acc_controlled_volume_and_cpu_agreement():
    """LISFLOOD-FP's CUDA ACC solver: closed-basin volume to rounding, and
    the same depth field as the CPU ACC solver (same code path)."""
    rng = np.random.default_rng(5)
    V = rng.uniform(0.0, 20.0, N * N)
    Vp = np.array([5000.0])
    g = _flood(inflow_points=[(N // 2, N // 2)], solver='acc_gpu')
    c = _flood(inflow_points=[(N // 2, N // 2)])
    (pg,) = _run(g, V, Vp, dt=300.0, device='cuda')
    (pc,) = _run(c, V, Vp, dt=300.0)
    exp = V.sum() + Vp.sum()
    assert pg.exports['depth'].device.type == 'cuda'
    assert abs(float(pg.exports['depth'].sum()) * DX * DX - exp) <= 1e-9 * exp
    dg = pg.exports['depth'].cpu().numpy()
    dc = pc.exports['depth'].numpy()
    # same ACC scheme, different time stepping: fields agree to ~1 percent
    assert np.max(np.abs(dg - dc)) <= 0.02 * dc.max()


@needs_bin
@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason='needs a GPU')
def test_gpu_free_edge_water_leaves_and_is_derived_from_rasters():
    f = _flood(dem=plane(), inflow_points=[(N // 2, 2)],
               free_edges=[('E', 0.002)], solver='acc_gpu')
    props = _run(f, None, np.array([4000.0]), dt=400.0, steps=4,
                 device='cuda')
    assert props[0].info['outflow_mode'] == 'derived'
    rec = f.volume_record()
    assert rec['outflow'] > 0 and abs(rec['residual']) <= 1e-9 * rec['handed']


def test_speed_is_exported_and_zero_for_still_water(bk):
    f = _flood(inflow_points=[(N // 2, N // 2)], backend=bk)
    (p,) = _run(f, None, np.array([3000.0]), dt=200.0)
    sp = p.exports['speed'].reshape(N, N)
    d = p.exports['depth'].reshape(N, N)
    assert float(sp.max()) > 0 and bool(torch.isfinite(sp).all())
    assert float(sp[d == 0].max()) == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason='needs a GPU')
@pytest.mark.parametrize('backend', ['engine', 'engine-mixed', 'lisflood'])
def test_gpu_restart_equals_the_continuous_run(backend):
    """save_state / load_state in the middle of a growing flood on the GPU:
    the resumed depth equals the uninterrupted one bit for bit."""
    if backend == 'lisflood' and not HAS_BIN:
        pytest.skip('LISFLOOD-FP binary not built')
    rng = np.random.default_rng(7)
    Vs = [rng.uniform(0.0, 30.0, N * N) for _ in range(4)]
    extra = {}
    if backend == 'engine-mixed':
        backend, extra = 'engine', dict(dtype='mixed')
    kw = dict(dem=plane(), inflow_points=[(N // 2, 2)], free_edges=[('E', 0.002)],
              backend=backend, **extra)
    if backend == 'lisflood':
        kw['solver'] = 'acc_gpu'

    def run(f, ks, start=0):
        ctx = _ctx(device='cuda')
        if f.depth is None:
            f.initialize(ctx)
        for k in ks:
            ctx.t = 300.0 * k
            p = f.compute(dict(runoff=torch.as_tensor(Vs[k]),
                               inflow=torch.tensor([2500.0])), ctx, 300.0)
            f.commit(p, ctx)
    a = _flood(**kw)
    run(a, range(4))
    b = _flood(**kw)
    run(b, range(2))
    blob = b.save_state()
    assert float(b.depth.sum()) > 0
    c = _flood(**kw)
    c.initialize(_ctx(device='cuda'))
    c.load_state(blob)
    run(c, range(2, 4))
    assert torch.equal(a.depth, c.depth)
    assert a.totals == c.totals


def test_upstream_discharge_inflow_map():
    f = _flood(dem=np.full((N, N), 10.0),
               inflow_points=[(5, 5), (6, 5), (20, 20)],
               inflow_map=[[0, [0, 1]], [2, [2]]], backend='engine')
    ctx = _ctx()
    f.initialize(ctx)
    p = f.compute(dict(upstream_discharge=torch.tensor([4.0, 3.0, 1.0])),
                  ctx, 100.0)
    d = p.exports['depth'].reshape(N, N)
    assert p.info['I_inflow'] == pytest.approx(500.0, rel=1e-15)
    assert float(d[13:, 13:].sum()) * DX * DX == pytest.approx(100.0,
                                                                   rel=1e-9)
