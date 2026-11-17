"""The snow pack's energy terms as VIC 5.0.1/5.1.0 report them.

Each test pins one term of put_data.c's snow rows (OUT_ADVECTION,
OUT_RFRZ_ENERGY, OUT_MELT_ENERGY, OUT_LATENT_SUB, ...) to the C line that
forms it, on a small hand-built case. The whole-model agreement is the
verification matrix's (verification/), these are the regression guards.
"""
import torch

from torchvic.core import snow

F64 = torch.float64


def _c(v, n=1):
    return torch.full((n,), float(v), dtype=F64)


def _forcing(air_temp, rain=0.0, snowfall=0.0, dt=10800.0):
    return dict(t_grnd=_c(0.0), wind=_c(2.0), net_short=_c(5.0),
                long_snow_in=_c(250.0), air_dens=_c(1.2),
                eact_air=_c(400.0), lv=_c(2.5e6), press=_c(85000.0),
                vpd=_c(150.0), rain=_c(rain), dt=_c(dt), z=_c(2.0),
                z0=_c(0.001), ra=_c(80.0), snow_depth=_c(0.0),
                snow_density=_c(0.0), air_temp=_c(air_temp),
                snowfall=_c(snowfall))


def test_thin_pack_keeps_rain_advection():
    """snow_melt.c 5.1.0:172 is a thin pack's last balance call (no
    root_brent under SNOW_MIN_SWQ_EB_THRES, :266-305), so its advection is
    CPFW RHOFW Tair Rain / dt (SnowPackEnergyBalance.c 5.1.0:224-229), and
    surface_fluxes.c 5.1.0:593, :874, :911-929 report it. Walnut Gulch
    2008-01-03 cell 14 reported 0 here against VIC's -3.7e-4 W m-2."""
    tair, rain, sf, dt = -0.3895502589, 2.4749e-6, 1.9932e-5, 10800.0
    st = dict(ice=_c(0.0), surf_water=_c(rain), pack_water=_c(0.0),
              surf_temp=_c(0.0), pack_temp=_c(0.0))
    new, _ = snow.snow_step(st, _forcing(tair, rain, sf, dt),
                            thin_to_surface=True)
    assert bool(new['thin'].all())
    want = snow.CONST_CPFW * snow.CONST_RHOFW * tair * rain / dt
    assert torch.allclose(new['energy']['advection'], _c(want), rtol=1e-14,
                          atol=0.0)
    # the frozen-branch terms of a thin pack stay the surface solve's
    assert float(new['energy']['latent_sub']) == 0.0


def _thin_residual(adv=0.0, thin=True):
    from torchvic.core import energy
    c = _c
    pack = dict(thin=torch.tensor([thin]), coverage=c(1.0), swq=c(2e-5),
                surf_water=c(7e-7), old_t_surf=c(0.0), density=c(100.0),
                net_short_snow=c(12.0), eact_air=c(500.0), press=c(85000.0),
                vpd=c(100.0), kappa_snow=c(400.0), t_snow_surf=c(-0.4),
                advection=c(adv))
    ts = c(-0.9)
    return energy._residual(
        ts, c(-0.39), c(2.0), c(3.0), c(8.0), c(0.1), c(0.1), c(1.5),
        c(1.4), c(2e6), c(2e6), c(4.0), 10800.0, c(0.6), c(0.0), c(270.0),
        c(80.0), c(10.0), c(0.0), c(0.01), c(2.0), c(1.1), c(2.5e6),
        lambda nbr, ra, ns=None, **k: torch.zeros_like(nbr),
        pack, c(0.0))


def test_thin_pack_surface_balance_terms():
    """func_surf_energy_bal.c 5.1.0 with INCLUDE_SNOW: the whole net
    longwave (:633-636, :662), the sensible heat not scaled by the bare
    fraction (:838-847), the pack's rain advection in the balance (:855).
    Walnut Gulch EB 2008-01-03: TorchVIC had no longwave and no sensible
    heat under a thin pack, so it sublimated 82 W m-2 where VIC had 66."""
    from torchvic.core import energy
    r0, t0 = _thin_residual(0.0)
    r1, _ = _thin_residual(0.5)
    long_out = snow.calc_outgoing_longwave(_c(-0.9) + energy.CONST_TKFRZ,
                                           energy.EMISS_GRND)
    assert torch.allclose(t0['net_long'], 270.0 - long_out, rtol=1e-14)
    sens = snow.calc_sensible_heat(_c(1.1), _c(-0.39), _c(-0.9),
                                   t0['ra_used'])
    assert torch.allclose(t0['sensible'], sens, rtol=1e-14)
    assert float(t0['sensible']) != 0.0
    assert abs(float(r1 - r0) - 0.5) < 1e-9


def _matrix_row_against_vic(tmp_path, rid, days):
    """One verification-matrix row, cut to `days`, through pristine VIC and
    this checkout's TorchVIC; returns verification/compare.py's result."""
    import os
    import sys
    import pytest
    import _paths
    sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
    M = pytest.importorskip('matrix')
    R = pytest.importorskip('run_matrix')
    C = pytest.importorskip('compare')
    r = [x for x in M.MATRIX if x['id'] == rid][0]
    c = M.CASES[r['case']]
    if not (os.path.exists(M.VIC[r['core']])
            and os.path.exists(c['params'])):
        pytest.skip('pristine VIC or the %s data is not present' % r['case'])
    d = R.prepare(r, days, root=str(tmp_path))
    rc_v, _ = R.run_vic(r, d)
    env_pp = os.environ.get('PYTHONPATH')
    os.environ['PYTHONPATH'] = _paths.REPO
    try:
        rc_t, _ = R.run_tv(r, d, 'cpu', 2, days)
    finally:
        if env_pp is None:
            os.environ.pop('PYTHONPATH')
        else:
            os.environ['PYTHONPATH'] = env_pp
    assert rc_v == 0 and rc_t == 0, open(os.path.join(d, 'tv.log')).read()[-2000:]
    return C.compare_dirs(os.path.join(d, 'vic'), os.path.join(d, 'tv'), r,
                          dict(vic_rc=0, tv_rc=0))


def _worst(res):
    return sorted(((x['score'], x['var']) for x in res['vars']
                   if x['group'] != 'diag'), reverse=True)[:5]


def test_stehekin_full_energy_snow_rows_are_vic51s(tmp_path):
    """Stehekin, 10 hourly January days, FULL_ENERGY TRUE, VIC 5.1.0: every
    water and energy row passes (score <= 1e-4), the snow energy terms
    (OUT_LATENT_SUB, OUT_ADVECTION, OUT_RFRZ_ENERGY, OUT_DELTACC,
    OUT_SNOW_FLUX, OUT_MELT_ENERGY) included. Before these fixes OUT_LATENT
    was 2.1e2 and OUT_LATENT_SUB 1.7e-1 off from the first step."""
    res = _matrix_row_against_vic(tmp_path, 'stehe_EB_510_1h', 10)
    assert res['grade'] == 'pass', _worst(res)


def test_walnut_gulch_full_energy_thin_packs_are_vic51s(tmp_path):
    """Walnut Gulch, the first 20 days of 2008 at 3 h, FULL_ENERGY TRUE,
    VIC 5.1.0: the trace packs of 3-4 January (INCLUDE_SNOW) pass. Were
    OUT_REFREEZE 2.6e9 and OUT_LATENT_SUB 0.23 off."""
    res = _matrix_row_against_vic(tmp_path, 'wg_EB_510_3h', 20)
    assert res['grade'] == 'pass', _worst(res)


def test_gangotri_snow_rows_are_vic51s(tmp_path):
    """ISC Gangotri (3450-5600 m), the first 60 days of 2004 at 3 h, VIC
    5.1.0, both modes: overstory canopies under snow (OUT_LATENT_SUB takes
    the canopy's AtmosLatentSub, OUT_ADVECTION/OUT_RFRZ_ENERGY its advection
    and refreeze), packs that melt out (OUT_MELT_ENERGY, delta_coverage 1,
    surf_atten 1), the two-layer pack's cold content (PackCC from before the
    melt, snow_melt.c 5.1.0:101-127, :390-418) and the ground's emission
    under a canopy (LongUnderIn - NetLongSnow). Were MELT_ENERGY 0.98 and
    LATENT_SUB 0.05 (WB), LATENT_SUB 41 vs 66 W m-2 at step 0 (EB)."""
    for rid in ('himal_WB_510_3h', 'himal_EB_510_3h'):
        res = _matrix_row_against_vic(tmp_path / rid, rid, 60)
        assert res['grade'] == 'pass', (rid, _worst(res))


def test_frozen_soil_under_and_after_snow_is_vics(tmp_path):
    """ISC Gangotri frozen soil (QUICK_FLUX), 20 days, VIC 5.0.1: once the
    first packs are gone, bare soil and transpiration take the liquid
    moisture, moist - ice (arno_evap.c :50-54, canopy_evap.c :201-241).
    Was OUT_EVAP 0.092 off from the first snow-free step."""
    res = _matrix_row_against_vic(tmp_path, 'himal_FS_501_3h', 20)
    assert res['grade'] == 'pass', _worst(res)
