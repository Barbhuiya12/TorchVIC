"""TorchVIC flood engine (torchvic/flood): conservation, restart, schemes,
DEM conditioning, and agreement with the LISFLOOD-FP binary when present."""
import os
import _paths  # noqa: E402
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from torchvic.flood import FloodEngine, dem as D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'flood'))
BIN = _paths.env_path(('TORCHVIC_LISFLOOD_BIN', 'LISFLOOD_BIN'), 'static/upstream/lisflood-fp-8.2-zenodo/build_cuda/lisflood')
HAVE_BIN = Path(BIN).exists()
CUDA = torch.cuda.is_available()


def slope_case():
    ny, nx, dx = 30, 80, 10.0
    x = np.arange(nx)[None, :] * np.ones((ny, 1))
    y = np.arange(ny)[:, None] * np.ones((1, nx))
    z = 5.0 - 0.002 * x * dx + 0.05 * np.cos(y / 4.0)
    return z, dx


def bowl(ny=40, nx=44, dx=20.0, k=1e-5):
    y, x = np.mgrid[0:ny, 0:nx]
    r2 = ((x - nx / 2 + 0.3) ** 2 + (y - ny / 2 - 0.7) ** 2) * dx * dx
    return k * r2 + 0.1 * np.sin(x / 5.0) * np.cos(y / 7.0), dx


def test_closed_basin_ledger_exact():
    z, dx = bowl()
    e = FloodEngine(z, dx, 0.03, inflow_points=[(20, 22), (5, 5)],
                    device='cpu', block=16)
    tot = 0.0
    for k in range(3):
        o = e.run(1200.0, runoff=np.full(z.shape, 0.5),
                  inflow=[1500.0, 300.0 * k])
        L = o['ledger']
        assert L['outflow'] == 0.0
        assert L['clamp'] == 0.0
        assert abs(L['rel_residual']) < 1e-14
        assert L['t_end'] == 1200.0
        tot += 0.5 * z.size + 1500.0 + 300.0 * k
    assert e.storage() == pytest.approx(tot, rel=1e-13)


def test_free_edge_outflow_and_ledger():
    z, dx = slope_case()
    e = FloodEngine(z, dx, 0.035, free_edges={'E': 0.002}, device='cpu')
    for _ in range(2):
        o = e.run(900.0, rate=np.full(z.shape, 30.0 / 3.6e6))
        L = o['ledger']
        assert abs(L['rel_residual']) < 1e-14
    assert L['outflow'] > 0
    assert o['speed'].max() > 0


def test_lake_at_rest():
    z, dx = bowl()
    h0 = np.clip(3.0 - z, 0.0, None)
    e = FloodEngine(z, dx, 0.03, device='cpu')
    e.reset(depth=h0)
    e.run(600.0)
    wet = h0 > 0
    assert np.abs(e.h.numpy() - h0)[wet].max() == 0.0
    assert float(e.Qx.abs().max()) == 0.0 and float(e.Qy.abs().max()) == 0.0


def test_restart_bitwise():
    z, dx = slope_case()
    kw = dict(free_edges={'E': 0.002}, inflow_points=[(15, 3)], device='cpu',
              block=8)
    a = FloodEngine(z, dx, 0.035, **kw)
    b = FloodEngine(z, dx, 0.035, **kw)
    r = np.full(z.shape, 20.0 / 3.6e6)
    for _ in range(2):
        a.run(400.0, rate=r, discharge=[3.0])
        b.run(400.0, rate=r, discharge=[3.0])
    s = b.state_dict()
    c = FloodEngine(z, dx, 0.035, **kw)
    c.load_state_dict(s)
    a.run(400.0, rate=r, discharge=[3.0])
    c.run(400.0, rate=r, discharge=[3.0])
    assert torch.equal(a.h, c.h) and torch.equal(a.qx, c.qx)
    assert torch.equal(a.qy, c.qy)


def test_momentum_kept_across_intervals():
    """Two half intervals equal one interval up to the clipped step at the
    split (no restart from rest)."""
    z, dx = slope_case()
    kw = dict(free_edges={'E': 0.002}, device='cpu')
    r = np.full(z.shape, 20.0 / 3.6e6)
    a = FloodEngine(z, dx, 0.035, **kw)
    a.run(1000.0, rate=r)
    b = FloodEngine(z, dx, 0.035, **kw)
    b.run(500.0, rate=r)
    assert float(b.qx.abs().max()) > 0
    b.run(500.0, rate=r)
    assert float((a.h - b.h).abs().max()) < 1e-3 * float(a.h.max())


def test_float32_mode():
    z, dx = slope_case()
    r = np.full(z.shape, 30.0 / 3.6e6)
    a = FloodEngine(z, dx, 0.035, free_edges={'E': 0.002}, device='cpu')
    b = FloodEngine(z, dx, 0.035, free_edges={'E': 0.002}, device='cpu',
                    dtype=torch.float32)
    a.run(900.0, rate=r)
    o = b.run(900.0, rate=r)
    assert abs(o['ledger']['rel_residual']) < 1e-5
    assert float((a.h - b.h.double()).abs().max()) < 1e-4


def test_schemes_agree_without_limiter():
    """torchvic and the LISFLOOD-FP CPU order are the same arithmetic where
    the wet/dry limiter does not act (slope case, overshoot end)."""
    z, dx = slope_case()
    r = np.full(z.shape, 25.0 / 3.6e6)
    a = FloodEngine(z, dx, 0.035, free_edges={'E': 0.002}, device='cpu',
                    end='overshoot')
    b = FloodEngine(z, dx, 0.035, free_edges={'E': 0.002}, device='cpu',
                    scheme='lisflood_cpu')
    a.run(900.0, rate=r)
    b.run(900.0, rate=r)
    assert float((a.h - b.h).abs().max()) < 1e-14


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
def test_gpu_equals_cpu():
    z, dx = slope_case()
    r = np.full(z.shape, 25.0 / 3.6e6)
    kw = dict(free_edges={'E': 0.002}, inflow_points=[(15, 3)], block=16)
    a = FloodEngine(z, dx, 0.035, device='cpu', **kw)
    b = FloodEngine(z, dx, 0.035, device='cuda', **kw)
    for _ in range(2):
        oa = a.run(600.0, rate=r, discharge=[2.0])
        ob = b.run(600.0, rate=r, discharge=[2.0])
    assert oa['ledger']['steps'] == ob['ledger']['steps']
    assert float((a.h - b.h.cpu()).abs().max()) < 1e-10
    assert abs(ob['ledger']['rel_residual']) < 1e-13


# ---------------------------------------------------------------------------
# LISFLOOD-FP binary
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_BIN, reason='LISFLOOD-FP binary not built')
@pytest.mark.parametrize('name', ['flat', 'slope'])
def test_lisflood_cpu_compat(tmp_path, name):
    import validate_engine as V
    c = V.cases([name])[name]
    ref = V.run_lisflood(tmp_path / name, c, 'acc_cpu')
    eng = V.run_engine(c, 'lisflood_cpu', 'cpu')
    assert np.abs(eng['h'] - ref['h']).max() < 1e-12


@pytest.mark.gpu
@pytest.mark.skipif(not (HAVE_BIN and CUDA), reason='needs CUDA + binary')
@pytest.mark.parametrize('name', ['flat', 'slope', 'basin'])
def test_lisflood_gpu_compat(tmp_path, name):
    import validate_engine as V
    c = V.cases([name])[name]
    ref = V.run_lisflood(tmp_path / name, c, 'acc_gpu')
    # A fresh interpreter: importing torchvic.core.implicit/snow/blowing (any
    # other test file in the same pytest session) runs fusion.setup(), which
    # switches inductor to eager-identical (no FMA) arithmetic process-wide and
    # exports TRITON_DEFAULT_FP_FUSION=0 (dropped from the child env below).
    # The engine's compiled step then differs from the LISFLOOD-FP CUDA
    # binary (nvcc contracts to FMA) by ~2e-5 on 'basin' -- run-to-run
    # deterministic, but order-dependent within a session.
    out = tmp_path / 'eng.npy'
    code = ('import sys, numpy as np; sys.path.insert(0, %r); '
            'import validate_engine as V; c = V.cases([%r])[%r]; '
            'np.save(%r, V.run_engine(c, "lisflood_gpu", "cuda")["h"])'
            % (str(ROOT / 'tools' / 'flood'), name, name, str(out)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(ROOT)] + os.environ.get('PYTHONPATH', '').split(os.pathsep)))
    env.pop('TRITON_DEFAULT_FP_FUSION', None)   # fusion.setup() exports 0
    subprocess.run([sys.executable, '-c', code], check=True, env=env,
                   cwd=tmp_path)
    assert np.abs(np.load(out) - ref['h']).max() < 1e-9


# ---------------------------------------------------------------------------
# DEM conditioning
# ---------------------------------------------------------------------------
def _drains(z, edges='NESW'):
    """True if every cell has a non-ascending 8-path to an open edge."""
    full = D.fill_depressions(z, edges)
    return np.allclose(full, z)


def test_fill_and_breach_remove_pits():
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 5, (30, 40)) + np.arange(40)[None, :] * 0.1
    z[10, 10] = -50.0                       # deep one-cell pit
    f = D.fill_depressions(z)
    assert _drains(f) and (f >= z).all()
    b = D.breach_depressions(z)
    assert _drains(b) and (b <= z).all()
    p = D.fill_pits(z)
    assert p[10, 10] >= np.sort(z[9:12, 9:12].ravel())[1]


def test_condition_keeps_shallow_depressions():
    z = np.full((20, 20), 10.0)
    z[0, :] = z[-1, :] = z[:, 0] = z[:, -1] = 9.0
    z[5, 5] = 9.8                           # 0.2 m hollow (kept)
    z[14, 14] = -40.0                       # 50 m pit (filled)
    out, info = D.condition(z, 'fill', min_depth=1.0)
    assert out[5, 5] == 9.8 and out[14, 14] == 10.0
    assert info['depressions'] == 1
    out, info = D.condition(z, 'breach', min_depth=1.0)
    assert out[5, 5] == 9.8 and out[14, 14] == -40.0
    diff = D.fill_depressions(out) != out
    assert diff.sum() == 1 and diff[5, 5]


def test_epsilon_fill_drains_flats():
    z = np.full((15, 15), 5.0)
    z[0, :] = 4.0
    z[7, 7] = 1.0
    f = D.fill_depressions(z, 'N', epsilon=1e-3)
    assert f[7, 7] > 5.0
    # every interior cell has a strictly lower neighbour
    for r in range(1, 14):
        for c in range(1, 14):
            assert f[r - 1:r + 2, c - 1:c + 2].min() < f[r, c]


def test_burn_rivers_path_non_increasing():
    ny, nx = 60, 60
    y, x = np.mgrid[0:ny, 0:nx]
    z = 10.0 + 0.01 * x + 2.0 * np.sin(y / 3.0)
    z[30, 35] = 3.0                         # pit on the river line

    def rc(lon, lat):                       # 1 'degree' = 20 cells
        return (ny - lat * 20.0, lon * 20.0)
    rivers = [(2.5, 1.5, 1.5, 1.5, 100.0), (1.5, 1.5, 0.5, 1.5, 200.0)]
    out, burnt = D.burn_rivers(z, rc, rivers, dvic=1.0)
    assert burnt.sum() > 20
    assert out[30, 35] > 3.0               # the pit is raised, not carried


def test_monotone_profile():
    v = [10.0, 9.0, 2.0, 8.5, 8.0, 9.5, 7.0]
    p = D.monotone_profile(v)
    assert (np.diff(p) <= 0).all()
    assert p.sum() == pytest.approx(sum(v))
    assert (D.monotone_profile([5.0, 4.0, 3.0]) == [5.0, 4.0, 3.0]).all()


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
@pytest.mark.parametrize('limiter', [True, False])
def test_triton_equals_torch(limiter):
    """Fused Triton kernels against the torch code of the same scheme, with
    point inflows, a free edge and wet/dry limiting (bowl edge)."""
    z, dx = bowl(60, 70, 20.0, 1e-5)
    kw = dict(free_edges={'E': -1.0, 'N': 0.001}, inflow_points=[(30, 35)],
              device='cuda', block=16, drycheck=limiter)
    a = FloodEngine(z, dx, 0.03, backend='torch', **kw)
    b = FloodEngine(z, dx, 0.03, backend='triton', **kw)
    r = np.full(z.shape, 40.0 / 3.6e6)
    for _ in range(3):
        oa = a.run(300.0, rate=r, discharge=[20.0])
        ob = b.run(300.0, rate=r, discharge=[20.0])
    La, Lb = oa['ledger'], ob['ledger']
    assert abs(Lb['rel_residual']) < 1e-13
    assert La['steps'] == Lb['steps']
    assert float((a.h - b.h).abs().max()) < 1e-9
    assert Lb['outflow'] == pytest.approx(La['outflow'], rel=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
def test_triton_float32_and_restart():
    z, dx = slope_case()
    r = np.full(z.shape, 30.0 / 3.6e6)
    kw = dict(free_edges={'E': 0.002}, device='cuda', block=8)
    a = FloodEngine(z, dx, 0.035, dtype=torch.float32, **kw)
    o = a.run(900.0, rate=r)
    assert abs(o['ledger']['rel_residual']) < 1e-5
    b = FloodEngine(z, dx, 0.035, **kw)
    c = FloodEngine(z, dx, 0.035, **kw)
    b.run(500.0, rate=r)
    c.load_state_dict(b.state_dict())
    b.run(500.0, rate=r)
    c.run(500.0, rate=r)
    assert torch.equal(b.h, c.h) and torch.equal(b.qx, c.qx)


# ---------------------------------------------------------------------------
# run-to-run bitwise determinism of the engine
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
@pytest.mark.parametrize('name', ['flat', 'basin'])
def test_engine_bitwise_deterministic(name):
    import validate_engine as V
    c = V.cases([name])[name]
    a = V.run_engine(c, 'lisflood_gpu', 'cuda')
    b = V.run_engine(c, 'lisflood_gpu', 'cuda')
    assert np.array_equal(a['h'], b['h'])
    assert np.array_equal(a['max'], b['max'])
    assert a['nsteps'] == b['nsteps']


# ---------------------------------------------------------------------------
# run-to-run determinism (bitwise)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
@pytest.mark.parametrize('scheme,backend', [('torchvic', 'triton'),
                                            ('torchvic', 'torch'),
                                            ('lisflood_gpu', 'torch')])
def test_engine_bitwise_deterministic_cuda(scheme, backend):
    z, dx = bowl(ny=48, nx=52)
    # four inflow points on one cell: an atomic index_add_ would round
    # differently from run to run
    pts = [(20, 22)] * 4 + [(5, 5), (40, 30)]

    def once():
        e = FloodEngine(z, dx, 0.03, inflow_points=pts, scheme=scheme,
                        backend=backend, device='cuda', block=16)
        outs = []
        for k in range(3):
            o = e.run(900.0, runoff=np.full(z.shape, 0.4 + 0.1 * k),
                      inflow=[1.1e3, 0.7e3, 3.3e2, 9.1e2, 5e2, 2e2])
            outs.append((o['depth'].clone(), o['max_depth'].clone(),
                         o['speed'].clone(), o['ledger']))
        return e.h.clone(), e.Qx.clone(), e.Qy.clone(), outs

    a, b = once(), once()
    for x, y in zip(a[:3], b[:3]):
        assert torch.equal(x, y)
    for (d1, m1, s1, l1), (d2, m2, s2, l2) in zip(a[3], b[3]):
        assert torch.equal(d1, d2) and torch.equal(m1, m2)
        assert torch.equal(s1, s2)
        assert l1 == l2


# ---------------------------------------------------------------------------
# fused Triton step (default) against the five-kernel reference, bit for bit;
# mixed precision (option) against float64
# ---------------------------------------------------------------------------
STATE = ('h', 'qx', 'qy', 'Qx', 'Qy', 'hmax', 'acc', 'eacc')


def _dual_case(n_raster, block, graphs, drycheck=True):
    rng = np.random.default_rng(0)
    ny, nx = 61, 77
    y, x = np.mgrid[0:ny, 0:nx]
    z = 2e-3 * ((x - 38) ** 2 + (y - 30) ** 2) + 0.3 * rng.random((ny, nx))
    man = 0.03 + 0.01 * rng.random((ny, nx)) if n_raster else 0.03
    kw = dict(free_edges={'E': -1.0, 'N': 0.001, 'W': 1e-3},
              inflow_points=[(30, 35), (30, 35), (5, 70)], device='cuda',
              block=block, graphs=graphs, track_edges=True,
              drycheck=drycheck)
    return z, man, kw


def _drive(e, z, n=4, T=700.0):
    out = None
    for k in range(n):
        out = e.run(T, rate=np.full(z.shape, (40.0 + 10 * k) / 3.6e6),
                    discharge=[20.0, 5.0, 3.0])
    return out


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
@pytest.mark.parametrize('fp_fusion', ['1', '0'])
@pytest.mark.parametrize('n_raster,block,graphs,drycheck', [
    (False, 16, True, True), (True, 7, True, True), (False, 5, False, True),
    (True, 16, True, False), (False, 64, True, True)])
def test_fused_triton_step_equals_reference_bitwise(n_raster, block, graphs,
                                                    drycheck, fp_fusion,
                                                    monkeypatch):
    """The default (fused, three launches per step) path against the
    five-kernel reference path: every state tensor and the ledger are
    identical, with edges, repeated inflow cells, a Manning raster, odd
    blocks, graphs on and off and per-face edge tracking; with Triton's FMA
    contraction on (its default) and off (what importing torchvic.core.fusion
    sets for the whole process)."""
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', fp_fusion)
    z, man, kw = _dual_case(n_raster, block, graphs, drycheck)
    a = FloodEngine(z, 20.0, man, triton_cfg=dict(fused=False), **kw)
    b = FloodEngine(z, 20.0, man, **kw)
    oa, ob = _drive(a, z), _drive(b, z)
    for k in STATE:
        assert torch.equal(getattr(a, k), getattr(b, k)), k
    assert oa['ledger'] == ob['ledger']
    assert torch.equal(oa['edge_out'], ob['edge_out'])


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
@pytest.mark.parametrize('cfg', [dict(compact=False), dict(wfuse=True),
                                 dict(compact=False, skip_dry=False),
                                 dict(ublock=128, npm=256, cblock=256,
                                      cprog=7),
                                 dict(seg=True), dict(seg=True, wfuse=True),
                                 dict(seg=True, sblock=64, sch=32, snw=2),
                                 dict(seg=True, quiet=True),
                                 dict(seg=True, quiet=True, wfuse=True),
                                 dict(seg=True, quiet=True, ublock=32,
                                      sblock=64, sch=32, snw=1),
                                 dict(fclock=True),
                                 dict(fclock=True, wfuse=True),
                                 dict(fclock=True, seg=True, quiet=True),
                                 dict(wflag=True),
                                 dict(wflag=True, fclock=True, block=512,
                                      num_warps=8),
                                 dict(pms=True), dict(pms=True, fclock=True,
                                                      wflag=True),
                                 dict(qflag=True),
                                 dict(qflag=True, pms=True, block=512,
                                      num_warps=8)])
@pytest.mark.parametrize('fp_fusion', ['1', '0'])
def test_fused_triton_variants_bitwise(cfg, fp_fusion, monkeypatch):
    """Tuning options of the fused path change no bit, with Triton's FMA
    contraction on and off."""
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', fp_fusion)
    z, man, kw = _dual_case(False, 16, True)
    a = FloodEngine(z, 20.0, man, **kw)
    b = FloodEngine(z, 20.0, man, triton_cfg=cfg, **kw)
    oa, ob = _drive(a, z), _drive(b, z)
    for k in STATE:
        assert torch.equal(getattr(a, k), getattr(b, k)), k
    assert oa['ledger'] == ob['ledger']


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
def test_mixed_precision_ledger_and_error_bound():
    """mixed=True: the water balance closes to float64 round-off (float32
    fluxes are used once per face by both cells, the continuity is a
    float32 pair); the depth stays within 1e-3 m of the float64 engine on
    this case and the stored volume within 1e-5 (the open edges carry the
    float32 rounding of the outflow: 3.6e-6 here)."""
    z, man, kw = _dual_case(True, 16, True)
    a = FloodEngine(z, 20.0, man, **kw)
    m = FloodEngine(z, 20.0, man, mixed=True, **kw)
    m2 = FloodEngine(z, 20.0, man, mixed=True, triton_cfg=dict(mskip=True),
                     **kw)
    oa, om = _drive(a, z), _drive(m, z)
    om2 = _drive(m2, z)
    for k in STATE:
        assert torch.equal(getattr(m, k), getattr(m2, k)), k
    assert om['ledger'] == om2['ledger']
    L = om['ledger']
    assert abs(L['rel_residual']) < 1e-13 and L['clamp'] == 0.0
    assert m.h.dtype == torch.float64 and m.qx.dtype == torch.float32
    assert float((m.h - a.h).abs().max()) < 1e-3
    assert abs(L['stored1'] - oa['ledger']['stored1']) <= (
        1e-5 * oa['ledger']['stored1'])
    # outflow through the three open edges (one with the local slope):
    # 8.5e-4 relative here
    assert om['edge_out'].sum().item() == pytest.approx(
        oa['edge_out'].sum().item(), rel=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
def test_mixed_precision_restart_and_lake_at_rest():
    z, dx = slope_case()
    r = np.full(z.shape, 20.0 / 3.6e6)
    kw = dict(free_edges={'E': 0.002}, inflow_points=[(15, 3)],
              device='cuda', block=8, mixed=True)
    a, b = FloodEngine(z, dx, 0.035, **kw), FloodEngine(z, dx, 0.035, **kw)
    for _ in range(2):
        a.run(400.0, rate=r, discharge=[3.0])
        b.run(400.0, rate=r, discharge=[3.0])
    c = FloodEngine(z, dx, 0.035, **kw)
    c.load_state_dict(b.state_dict())
    a.run(400.0, rate=r, discharge=[3.0])
    c.run(400.0, rate=r, discharge=[3.0])
    for k in ('h', 'qx', 'qy', 'hmax'):
        assert torch.equal(getattr(a, k), getattr(c, k)), k
    zb, dxb = bowl()
    h0 = np.clip(3.0 - zb, 0.0, None)
    e = FloodEngine(zb, dxb, 0.03, device='cuda', mixed=True)
    e.reset(depth=h0)
    e.run(600.0)
    wet = h0 > 0
    assert np.abs(e.h.cpu().numpy() - h0)[wet].max() < 1e-12
    assert float(e.qx.abs().max()) < 1e-9 and float(e.qy.abs().max()) < 1e-9


def test_mixed_precision_needs_triton():
    z, dx = slope_case()
    with pytest.raises(ValueError, match='mixed precision'):
        FloodEngine(z, dx, 0.03, device='cpu', mixed=True)
    if CUDA:
        with pytest.raises(ValueError, match='mixed precision'):
            FloodEngine(z, dx, 0.03, device='cuda', mixed=True,
                        dtype=torch.float32)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
@pytest.mark.parametrize('fp_fusion', ['1', '0'])
@pytest.mark.parametrize('cfg', [dict(seg=True, quiet=True, ublock=16),
                                 dict(seg=True, quiet=True, ublock=32,
                                      wfuse=True)])
def test_quiet_tiles_bitwise(cfg, fp_fusion, monkeypatch):
    """Dry-tile skipping on a domain with dry land, a wetting front and
    rain on part of it only (tiles settle, wake up and stay static): every
    state tensor and the ledger equal the default path's."""
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', fp_fusion)
    rng = np.random.default_rng(3)
    ny, nx = 70, 150
    y, x = np.mgrid[0:ny, 0:nx]
    z = 0.02 * x + 1e-3 * (y - 35) ** 2 + 0.05 * rng.random((ny, nx))
    kw = dict(free_edges={'W': 1e-3, 'S': -1.0}, inflow_points=[(35, 10)],
              device='cuda', block=7, track_edges=True)
    a = FloodEngine(z, 30.0, 0.03, **kw)
    b = FloodEngine(z, 30.0, 0.03, triton_cfg=cfg, **kw)
    r = np.zeros((ny, nx))
    r[:, :40] = 30.0 / 3.6e6
    r[10:20, 100:120] = 60.0 / 3.6e6
    for k in range(4):
        oa = a.run(900.0, rate=r * (k % 2), discharge=[8.0 + 4 * k])
        ob = b.run(900.0, rate=r * (k % 2), discharge=[8.0 + 4 * k])
        for s in STATE:
            assert torch.equal(getattr(a, s), getattr(b, s)), (k, s)
        assert oa['ledger'] == ob['ledger']
        assert torch.equal(oa['edge_out'], ob['edge_out'])
    st = b._stepper
    assert int(st.dry[0].sum()) > 0 and int(st.qst.sum()) > 0


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
@pytest.mark.parametrize('fp_fusion', ['1', '0'])
@pytest.mark.parametrize('cfg', [dict(), dict(fclock=True),
                                 dict(seg=True, quiet=True),
                                 dict(wflag=True, fclock=True),
                                 dict(pms=True, wflag=True),
                                 dict(qflag=True, pms=True)])
def test_signed_zero_depth_bitwise(cfg, fp_fusion, monkeypatch):
    """A start depth with -0.0 cells (and +0.0, tiny and wet cells): the
    default and option paths equal the reference path bit for bit (the
    sign-bit compares of the limiter and update treat -0 as 0)."""
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', fp_fusion)
    z, man, kw = _dual_case(False, 16, True)
    h0 = np.where(np.arange(z.size).reshape(z.shape) % 3 == 0, -0.0, 0.0)
    h0[20:30, 30:45] = 0.4
    h0[5, 5] = 1e-310
    a = FloodEngine(z, 20.0, man, triton_cfg=dict(fused=False), **kw)
    b = FloodEngine(z, 20.0, man, triton_cfg=cfg, **kw)
    a.reset(depth=h0)
    b.reset(depth=h0)
    assert bool(torch.signbit(b.h).any())
    oa, ob = _drive(a, z, n=2), _drive(b, z, n=2)
    for k in STATE:
        assert torch.equal(getattr(a, k), getattr(b, k)), k
    assert oa['ledger'] == ob['ledger']
