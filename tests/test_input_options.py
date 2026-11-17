"""ORGANIC_FRACT, BULK_DENSITY_COMB, MAX_SNOW_ALBEDO and CORRPREC: the
image driver's inputs and the physics they reach.

Each is checked against the C rewritten as a scalar Python loop, straight
from the source (VIC 5.1.0): vic_init.c:572-627 and :735-737, correct_precip.c
and surface_fluxes.c:524-535. The runs against vic_image.exe are the
matrix rows for these options (verification/matrix.py).
"""
import math
import random

import numpy as np
import pytest
import torch

from torchvic.core import energy, frozen
from torchvic.io.params import ImageParams
from torchvic.model import basin

torch.set_default_dtype(torch.float64)


def c_correct_precip(wind, wind_h, rough, snow_rough, gauge_height=1.0):
    """correct_precip.c:19-37."""
    gw = wind * (math.log((gauge_height + rough) / rough)
                 / math.log(wind_h / rough))
    rain = 100. / math.exp(4.606 - 0.041 * math.pow(gw, 0.69))
    gw = wind * (math.log((gauge_height + snow_rough) / snow_rough)
                 / math.log(wind_h / snow_rough))
    snow = 100. / math.exp(4.606 - 0.036 * math.pow(gw, 1.75))
    return rain, snow


def test_gauge_correction_matches_c():
    rng = random.Random(3)
    for _ in range(200):
        wind = rng.uniform(0.0, 15.0)
        wind_h = rng.choice([10.0, 2.0, 5.0])
        rough = rng.uniform(0.005, 0.2)
        srough = rng.uniform(0.005, 0.1)
        want = c_correct_precip(wind, wind_h, rough, srough)
        got = basin.gauge_correction(torch.tensor(wind), wind_h, 1.0,
                                     torch.tensor(rough), torch.tensor(srough))
        assert float(got[0]) == pytest.approx(want[0], rel=1e-14)
        assert float(got[1]) == pytest.approx(want[1], rel=1e-14)


def test_gauge_correction_is_an_undercatch_fix():
    # more wind, more undercatch: the factor grows with the wind and
    # exceeds 1 (the gauge catches less than falls)
    w = torch.tensor([0.5, 2.0, 6.0])
    r = torch.full_like(w, 0.01)
    rain, snow = basin.gauge_correction(w, 10.0, 1.0, r, 3 * r)
    assert bool((rain > 1.0).all()) and bool((snow > 1.0).all())
    assert bool((rain[1:] > rain[:-1]).all())
    assert bool((snow[1:] > snow[:-1]).all())


def _params(**soil):
    p = object.__new__(ImageParams)
    p.soil = {k: np.asarray(v, np.float64) for k, v in soil.items()}
    return p


def _soil(n=4, layers=3, seed=0):
    r = np.random.RandomState(seed)
    return dict(bulk_density=r.uniform(1300, 1600, (layers, n)),
                soil_density=r.uniform(2600, 2700, (layers, n)),
                organic=r.uniform(0, 0.5, (layers, n)),
                bulk_density_org=r.uniform(100, 400, (layers, n)),
                soil_density_org=r.uniform(1200, 1400, (layers, n)),
                bulk_density_comb=r.uniform(900, 1500, (layers, n)))


def test_density_options_off_touches_nothing():
    p = _params(**_soil())
    before = {k: v.copy() for k, v in p.soil.items()}
    p.apply_density_options(False, False)
    assert set(p.soil) == set(before)
    for k in before:
        assert np.array_equal(p.soil[k], before[k])


def test_organic_mix_matches_c():
    s = _soil()
    p = _params(**s)
    p.apply_density_options(True, False)
    L, N = s['organic'].shape
    for j in range(L):
        for i in range(N):
            o = s['organic'][j, i]
            bulk = (1 - o) * s['bulk_density'][j, i] \
                + o * s['bulk_density_org'][j, i]          # vic_init.c:621
            soil = (1 - o) * s['soil_density'][j, i] \
                + o * s['soil_density_org'][j, i]          # :735
            assert p.soil['bulk_density'][j, i] == bulk
            assert p.soil['soil_density'][j, i] == soil
    assert np.array_equal(p.soil['bulk_dens_min'], s['bulk_density'])
    assert np.array_equal(p.soil['soil_dens_min'], s['soil_density'])
    assert np.array_equal(p.soil['organic'], s['organic'])


def test_bulk_density_comb_is_read_not_mixed():
    s = _soil()
    p = _params(**s)
    p.apply_density_options(False, True)
    assert np.array_equal(p.soil['bulk_density'], s['bulk_density_comb'])
    # no organic fraction: the particle density is the mineral one
    assert np.array_equal(p.soil['soil_density'], s['soil_density'])
    assert not p.soil['organic'].any()
    assert np.array_equal(p.soil['bulk_dens_min'], s['bulk_density'])


def test_comb_with_organic_mixes_only_the_particle_density():
    s = _soil()
    del s['bulk_density_org']                # not read with COMB (:583)
    p = _params(**s)
    p.apply_density_options(True, True)
    assert np.array_equal(p.soil['bulk_density'], s['bulk_density_comb'])
    o = s['organic']
    assert np.array_equal(p.soil['soil_density'],
                          (1 - o) * s['soil_density']
                          + o * s['soil_density_org'])


@pytest.mark.parametrize('org,comb,gone', [
    (True, False, 'organic'), (True, False, 'bulk_density_org'),
    (True, False, 'soil_density_org'), (False, True, 'bulk_density_comb')])
def test_missing_file_variable_is_named(org, comb, gone):
    s = _soil()
    del s[gone]
    with pytest.raises(ValueError, match=gone):
        _params(**s).apply_density_options(org, comb)


def test_thermal_properties_use_mineral_densities_for_kdry():
    rng = np.random.RandomState(1)
    n = 5
    t = lambda a: torch.as_tensor(a)
    depth = t(rng.uniform(0.1, 0.5, (n, 2)))
    bmin = t(rng.uniform(1300, 1600, (n, 2)))
    smin = t(rng.uniform(2600, 2700, (n, 2)))
    org = t(rng.uniform(0.05, 0.4, (n, 2)))
    bulk = (1 - org) * bmin + org * 300.0
    soil = (1 - org) * smin + org * 1300.0
    quartz = t(rng.uniform(0.2, 0.6, (n, 2)))
    moist = depth * 1000.0 * t(rng.uniform(0.1, 0.4, (n, 2)))
    kap, cs = energy.thermal_properties(
        moist, depth, bulk, soil, quartz, organic=org,
        bulk_dens_min=bmin, soil_dens_min=smin)
    frac = moist / depth / 1000.0
    want_k = frozen.soil_conductivity(frac, frac, smin, bmin, quartz, soil,
                                      bulk, org)
    want_c = frozen.volumetric_heat_capacity(bulk / soil, frac,
                                             torch.zeros_like(frac), org)
    assert torch.equal(kap, want_k) and torch.equal(cs, want_c)
    # and it differs from treating the mixed pair as the mineral one
    kap0, _ = energy.thermal_properties(moist, depth, bulk, soil, quartz,
                                        organic=org)
    assert not torch.equal(kap, kap0)
    # the old call, organic and mineral arguments absent, is unchanged
    k1, c1 = energy.thermal_properties(moist, depth, bmin, smin, quartz)
    assert torch.equal(k1, frozen.soil_conductivity(
        frac, frac, smin, bmin, quartz, smin, bmin, torch.zeros_like(frac)))


def test_new_snow_albedo_per_tile_rule():
    # surface_fluxes.c:524-535: a vegetation tile takes its class's value,
    # the bare-soil tile (iveg == Nveg) the run's
    from torchvic.core.vegetation import VegMosaic
    cv = torch.tensor([[0.5, 0.3, 0.2]])
    m = VegMosaic(cv, torch.zeros(1, 3, 12), cv, cv, cv, torch.zeros(1, 3, 3),
                  torch.float64, 'cpu',
                  bare_tile=torch.tensor([[False, False, True]]),
                  max_snow_albedo=torch.tensor([[0.7, 0.8, 0.9]]))
    assert m.select(torch.tensor([0])).max_snow_albedo.shape == (1, 3)
    assert m.tile(2).max_snow_albedo.shape == (2, 3)
