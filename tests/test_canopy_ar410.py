"""AERO_RESIST_CANSNOW AR_410: the intercepted-snow resistance is Ra[1]
divided by the stability correction at the trial foliage temperature
(func_canopy_energy_bal.c:178-188), and rows without it are bit-identical."""

import torch

from torchvic.core import canopy as C
from torchvic.core import snow as S
from torchvic.core.devconst import device_constant

D = torch.float64
CPU = torch.device("cpu")


def _st(n, aero_410, aero_406=True, aero_406_full=True):
    g = torch.Generator(device=CPU).manual_seed(11)
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g, dtype=D,
                                                  device=CPU)
    return dict(
        has_snow=torch.ones(n, dtype=torch.bool, device=CPU),
        net_short_over=r(0.0, 300.0), long_over_in=r(180.0, 340.0),
        long_under_out=r(250.0, 360.0), air_dens=r(1.0, 1.3),
        eact_air=r(100.0, 900.0), press=r(70000.0, 101300.0),
        ra0=r(30.0, 120.0), ra1=r(10.0, 40.0), t_canopy=r(-12.0, 2.0),
        vpd=r(0.0, 400.0), rainfall=torch.zeros(n, dtype=D, device=CPU),
        int_rain=r(0.0, 0.0006),
        dt=torch.full((n,), 3600.0, dtype=D, device=CPU),
        lv=torch.full((n,), 2.5e6, dtype=D, device=CPU),
        aero_406=torch.full((n,), aero_406, dtype=torch.bool, device=CPU),
        aero_406_full=torch.full((n,), aero_406_full, dtype=torch.bool,
                                 device=CPU),
        aero_410=torch.full((n,), aero_410, dtype=torch.bool, device=CPU),
        wind1=r(0.0, 6.0), ref_height1=r(8.0, 15.0),
        displacement1=r(1.0, 6.0), roughness1=r(0.1, 1.5))


def test_ar410_uses_stability_corrected_resistance():
    n = 300
    st = _st(n, True, aero_406=False, aero_406_full=True)
    g = torch.Generator(device=CPU).manual_seed(12)
    t = -15.0 + 15.0 * torch.rand(n, generator=g, dtype=D, device=CPU)
    rest, terms = C.canopy_residual(t, st, None, return_terms=True)
    corr = S.stability_correction(st["ref_height1"], st["displacement1"], t,
                                  st["t_canopy"], st["wind1"],
                                  st["roughness1"])
    expect_ra = torch.where(st["wind1"] > 0.0, st["ra1"] / corr,
                            torch.full_like(corr, C.HUGE_RESIST))
    assert torch.equal(terms["ra_used1"], expect_ra)
    # The vapour flux and the sensible heat both see it (:190-191, :238).
    es = S.svp(t)
    # IEEE division, as the implementation: `python_float / tensor` rounds
    # 1 ulp off for some operands (torchvic.core.snow's note).
    _eps = device_constant(S.CONST_EPS, st["press"].dtype,
                           st["press"].device)
    dens_eps = st["air_dens"] * (_eps / st["press"])
    vmf = dens_eps * (st["eact_air"] - es) / expect_ra / S.CONST_RHOFW
    vmf = torch.where((st["vpd"] == 0.0) & (vmf < 0.0),
                      torch.zeros_like(vmf), vmf)
    assert torch.equal(terms["vapor_mass_flux"], vmf)
    sens = S.calc_sensible_heat(st["air_dens"], st["t_canopy"], t, expect_ra)
    assert torch.equal(terms["sensible"], sens)
    assert torch.isfinite(rest).all()


def test_no_410_is_bit_identical_with_or_without_the_flag():
    n = 200
    st0 = _st(n, False)
    del st0["aero_410"], st0["wind1"], st0["ref_height1"]
    del st0["displacement1"], st0["roughness1"]
    st1 = _st(n, False)  # flag present but false, geometry present
    g = torch.Generator(device=CPU).manual_seed(13)
    t = -15.0 + 15.0 * torch.rand(n, generator=g, dtype=D, device=CPU)
    r0, tm0 = C.canopy_residual(t, st0, None, return_terms=True)
    r1, tm1 = C.canopy_residual(t, st1, None, return_terms=True)
    assert torch.equal(r0, r1)
    for k in tm0:
        assert torch.equal(tm0[k], tm1[k]), k
