"""GRND_FLUX_TYPE GF_406: the VIC 4.0.6 ground heat flux is the single
surface term alone, and the default GF_410 rows do not move.

func_surf_energy_bal.c:385-391 (QUICK_FLUX, the path TorchVIC runs;
5.0.1:411-416 identical): under GF_406 the flux is the attenuation
factor times kappa1/D1 (T1 - TMean) -- no kappa2/D2 deep term, no
division by 2. The T1 estimate above it (:379) is outside the branch
and is the same under both values.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

from torchvic.core import energy as E  # noqa: E402
from torchvic.io.global_param import GlobalParam  # noqa: E402

import _paths  # noqa: E402

sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
matrix = pytest.importorskip('matrix')

CPU = torch.device('cpu')
D = torch.float64


def _args(n, sw):
    t = lambda v: torch.full((n,), float(v), dtype=D, device=CPU)
    def evap_of(net_rad, ra):
        return torch.clamp(net_rad, min=0.0) * 0.0004 * 3.0
    return (t(18.0), t(0.1), t(0.1), t(0.8), t(0.9), t(2.5e6), t(2.6e6), t(4.0),
            t(10800.0), t(0.65), sw, t(320.0), t(40.0), t(5.0), t(0.402),
            t(0.0738), t(2.85), t(1.1), t(2.45e6), evap_of)


def test_gf406_is_the_single_surface_term():
    n = 16
    ts = torch.linspace(-10, 25, n, dtype=D, device=CPU)
    tair = ts + 2.0
    t0, t1_old = ts - 1.0, ts - 3.0
    sw = torch.linspace(0, 600, n, dtype=D, device=CPU)
    args = (t0,) + _args(n, sw)
    _, terms = E._residual(ts, tair, t1_old, *args, grnd_flux_type='GF_406')
    (t2, d1, d2, kappa1, kappa2, cs1, cs2, dp, dt, atten) = args[1:11]
    t1 = E.estimate_T1(ts, t1_old, t2, d1, d2, kappa1, kappa2, cs2, dp, dt)
    assert torch.equal(terms['t1'], t1)
    assert torch.equal(terms['grnd'], atten * (kappa1 / d1 * (t1 - ts)))


def test_gf406_leaves_t1_and_every_other_term_alone():
    # The branch is below estimate_T1 and touches only grnd_flux; the T1,
    # the storage change and the residual's other terms are the GF_410
    # run's, and only the flux and what carries it differ.
    n = 16
    ts = torch.linspace(-10, 25, n, dtype=D, device=CPU)
    tair = ts + 2.0
    t0, t1_old = ts - 1.0, ts - 3.0
    sw = torch.linspace(0, 600, n, dtype=D, device=CPU)
    args = (t0,) + _args(n, sw)
    _, tm410 = E._residual(ts, tair, t1_old, *args)
    _, tm406 = E._residual(ts, tair, t1_old, *args, grnd_flux_type='GF_406')
    assert torch.equal(tm406['t1'], tm410['t1'])
    assert torch.equal(tm406['delta_h'], tm410['delta_h'])
    assert not torch.equal(tm406['grnd'], tm410['grnd'])
    (t2, d1, d2, kappa1, kappa2, cs1, cs2, dp, dt, atten) = args[1:11]
    deep = kappa2 / d2 * (1.0 - torch.exp(-d1 / dp)) * (t2 - tm410['t1'])
    assert torch.equal(tm410['grnd'], atten * (kappa1 / d1
                                               * (tm410['t1'] - ts) + deep)
                        / 2.0)


def test_gf410_default_is_bit_identical_with_or_without_the_flag():
    n = 16
    tair = torch.linspace(-10, 35, n, dtype=D, device=CPU)
    t0, t1_old = tair - 2.0, tair - 4.0
    sw = torch.linspace(0, 600, n, dtype=D, device=CPU)
    args = (t0,) + _args(n, sw)
    ts, _ = E.solve_surface_temperature(tair, t0, t1_old, args)
    r0, tm0 = E._residual(ts, tair, t1_old, *args)
    r1, tm1 = E._residual(ts, tair, t1_old, *args, grnd_flux_type='GF_410')
    assert torch.equal(r0, r1)
    for k in tm0:
        # terms of the QUICK_FLUX FALSE profile are None on this path
        if tm0[k] is None or tm1[k] is None:
            assert tm0[k] is None and tm1[k] is None, k
            continue
        assert torch.equal(tm0[k], tm1[k]), k


def test_water_balance_gf406_is_the_single_surface_term():
    n = 8
    tair = torch.linspace(-5, 30, n, dtype=D, device=CPU)
    t0, t1_old = tair - 1.0, tair - 3.0
    t2 = torch.full((n,), 18.0, dtype=D, device=CPU)
    d1 = torch.full((n,), 0.1, dtype=D, device=CPU)
    kap1 = torch.full((n,), 0.8, dtype=D, device=CPU)
    kap2 = torch.full((n,), 0.9, dtype=D, device=CPU)
    cs1 = torch.full((n,), 2.5e6, dtype=D, device=CPU)
    cs2 = torch.full((n,), 2.6e6, dtype=D, device=CPU)
    dp = torch.full((n,), 4.0, dtype=D, device=CPU)
    dt = torch.full((n,), 10800.0, dtype=D, device=CPU)
    atten = torch.full((n,), 0.65, dtype=D, device=CPU)
    a = (tair, t0, t1_old, t2, d1, kap1, kap2, cs1, cs2, dp, dt, atten)
    g, dh, t1 = E.water_balance_ground_flux(*a, grnd_flux_type='GF_406')
    t1x = E.estimate_T1(tair, t1_old, t2, d1, d1, kap1, kap2, cs2, dp, dt)
    assert torch.equal(t1, t1x)
    assert torch.equal(g, atten * (kap1 / d1 * (t1 - tair)))
    g0, dh0, t10 = E.water_balance_ground_flux(*a)
    g1, dh1, t11 = E.water_balance_ground_flux(*a, grnd_flux_type='GF_410')
    assert torch.equal(g0, g1) and torch.equal(dh0, dh1)
    assert torch.equal(t10, t11) and torch.equal(dh, dh0)


def test_image_options_accepts_gf406_and_defaults_to_gf410(tmp_path):
    from torchvic import case as C
    r = matrix.row('stehe', 'EB', GRND_FLUX_TYPE='GF_406')
    g = tmp_path / 'g_gf406.txt'
    g.write_text(matrix.global_text(r, str(tmp_path), str(tmp_path),
                                    ['OUT_GRND_FLUX']))
    assert C.image_options(GlobalParam(str(g)))['grnd_flux_type'] == 'GF_406'
    r = matrix.row('stehe', 'EB')
    g = tmp_path / 'g_gf410.txt'
    g.write_text(matrix.global_text(r, str(tmp_path), str(tmp_path),
                                    ['OUT_GRND_FLUX']))
    assert C.image_options(GlobalParam(str(g)))['grnd_flux_type'] == 'GF_410'
