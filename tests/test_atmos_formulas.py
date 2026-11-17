"""torchvic.core.atmos, measured against VIC 5.0.1 itself.

The numbers below were not read out of a paper or off a header. They were
printed by a binary compiled against the **pristine** upstream VIC 5.0.1 tree
(`reference/VIC-VIC.5.0.1/vic`), by a shim that calls the same functions VIC calls:

    calc_scale_height            vic_run/src/physics.c:76
    calc_latent_heat_of_vaporization              physics.c:49
    svp, svp_slope               vic_run/src/svp.c:35, :56
    air_density                  drivers/shared_all/src/forcing_utils.c:88
    CONST_PSTD * exp(-elev / h)  vic_run/src/penman.c:206
    0.003486 * pz / (275 + t)    vic_run/src/penman.c:218
    1628.6 * pz / lv             vic_run/src/penman.c:213
    CONST_MINDECL * cos(...)     vic_run/src/compute_coszen.c:60

They are pasted here at 17 significant digits so the comparison survives
without a compiler: a test that only runs where VIC has been built is a test
that stops running. `tests/oracle/build_ref_soil.sh` is the pattern for
regenerating them.

What each test would have shown had the fix not been made is written on the
test, because that is the question to ask of every measurement.
"""
import math

import torch


from torchvic.core import atmos  # noqa: E402
from torchvic.core.snow import svp  # noqa: E402
import pytest

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')

D = torch.float64


def _t(x):
    # Pinned to the CPU on purpose, and not left to the ambient default
    # device: this is a bit-exactness claim against a double-precision C
    # binary, and a CUDA kernel is entitled to round the last bit elsewhere.
    # Other tests in this directory set the default device to cuda, which also
    # made the module-level tensors below disagree with tensors built inside a
    # test on where they lived.
    return torch.tensor(x, dtype=D, device='cpu')


# tair (C), elevation (m) -> VIC's answers, printed by the shim.
#   scale_height, pressure, latent_heat, r_air_penman, density_forcing,
#   svp, svp_slope, gamma
VIC_STATES = [
    (-40.0, 3000.0, 6539.2826915889473, 64043.802201301929, 2595440.0,
     0.9500284871222916, 0.9569643721831248, 12.499070297976381,
     1.3157957153643742, 40.186533406682614),
    (-20.0, 0.0, 7410.1137572772695, 101325.0, 2548220.0,
     1.3851723529411766, 1.3944172998765458, 102.49283462648772,
     8.8948571870608557, 64.758103695913221),
    (0.0, 1000.0, 7900.4136009841404, 89278.216400927704, 2501000.0,
     1.1317231359041235, 1.1386713198090286, 610.77999999999997,
     44.448208259587005, 58.136146833486947),
    (20.0, 2000.0, 8390.7134446910113, 79835.991073283221, 2453780.0,
     0.94341784705581466, 0.94877428197905422, 2338.0229641467558,
     144.7218574205433, 52.988000171958795),
    (35.0, 500.0, 8972.48713983573, 95833.016331774081, 2418365.0,
     1.0776577255889175, 1.0834454927057418, 5621.7744901824153,
     310.70129652909549, 64.536846339542322),
]

_TAIR = _t([s[0] for s in VIC_STATES])
_ELEV = _t([s[1] for s in VIC_STATES])


def _col(i):
    return _t([s[i] for s in VIC_STATES])


def _max_rel(got, ref):
    return float(((got - ref).abs() / ref.abs().clamp(min=1e-30)).max())


# ---------------------------------------------------------------------------
# The routines that were already right. Measured, not assumed.
# ---------------------------------------------------------------------------
def test_scale_height_is_bit_exact_against_vic():
    got = atmos.scale_height(_TAIR, _ELEV)
    assert torch.equal(got, _col(2))


def test_air_pressure_is_bit_exact_against_vic():
    # penman.c:206. VIC's image driver reads PRESSURE from forcing instead, so
    # this is the fallback path, but the formula is still VIC's.
    got = atmos.air_pressure(_TAIR, _ELEV)
    assert torch.equal(got, _col(3))


def test_latent_heat_of_vaporization_is_bit_exact_against_vic():
    got = atmos.latent_heat_of_vaporization(_TAIR)
    assert torch.equal(got, _col(4))


def test_svp_slope_matches_vic_to_one_ulp():
    # svp.c:56. Evaluated in pure Python the expression is bit-identical to
    # the C; through torch it lands one ulp away on some states, because a
    # zero-dimensional tensor and a Python scalar dispatch to different
    # kernels. That is the effect test_params_s0.py documents, not a
    # difference in the formula, so the tolerance is a few ulp and not 1e-9.
    ref = _col(8)
    got = atmos.svp_slope(_TAIR)
    assert _max_rel(got, ref) < 4.0 * 2.220446049250313e-16
    assert torch.equal(svp(_TAIR), _col(7))


# ---------------------------------------------------------------------------
# Defect 1: air density. VIC has two, and we were returning the wrong one.
# ---------------------------------------------------------------------------
def test_air_density_is_the_ideal_gas_forcing_density():
    """forcing_utils.c:88 -- the density every driver puts in force[].density.

    Before the fix `air_density` returned penman.c:218's local `r_air`, which
    is 0.5-0.7 per cent smaller. This assertion would have failed by up to
    1.10e-2 kg/m3 (7.2e-3 relative), and that error goes straight into
    calc_sensible_heat and latent_heat_from_snow, so it is an energy-balance
    error and not a rounding one.
    """
    got = atmos.air_density(_TAIR, atmos.air_pressure(_TAIR, _ELEV))
    assert torch.equal(got, _col(6))


def test_penman_air_density_is_kept_under_its_own_name():
    """penman.c:218, Handbook of Hydrology 4.2.4.

    VIC still uses this one, but only inside penman(). Keeping it means the
    fix above is a renaming with a correction, not a deletion.
    """
    got = atmos.air_density_penman(_TAIR, atmos.air_pressure(_TAIR, _ELEV))
    assert torch.equal(got, _col(5))


def test_the_two_air_densities_actually_differ():
    """The guard against "fixed" by aliasing one to the other."""
    pz = atmos.air_pressure(_TAIR, _ELEV)
    rel = _max_rel(atmos.air_density(_TAIR, pz),
                   atmos.air_density_penman(_TAIR, pz))
    assert rel > 5e-3


# ---------------------------------------------------------------------------
# Defect 2: the psychrometric constant had no home.
# ---------------------------------------------------------------------------
def test_psychrometric_constant_matches_penman():
    """penman.c:213, Handbook of Hydrology 4.2.28.

    VIC inlines `gamma = 1628.6 * pz / lv`, so there was no routine to port
    and the coefficient sat as a bare literal at each call site, where
    params.py could not install it as a tensor. Without the function this test
    fails with AttributeError.
    """
    pz = atmos.air_pressure(_TAIR, _ELEV)
    lv = atmos.latent_heat_of_vaporization(_TAIR)
    assert torch.equal(atmos.psychrometric_constant(pz, lv), _col(9))


def test_psychrometric_coefficient_is_vics_literal_not_a_derivation():
    # 1013 / 0.622 is 1628.6045..., and VIC writes 1628.6. Deriving it would
    # disagree with VIC in the fifth digit.
    assert atmos.PSYCHROMETRIC_COEF == 1628.6
    assert abs(atmos.PSYCHROMETRIC_COEF - 1013.0 / 0.622) > 4e-3


# ---------------------------------------------------------------------------
# Defect 3: solar declination used a textbook phase, not VIC's.
# ---------------------------------------------------------------------------
# compute_coszen.c:60 evaluated at these days of the year.
VIC_DECL = {
    1: -0.40021369237635768,
    15: -0.36820162033043657,
    100: 0.13813242804924331,
    182: 0.40229488659440077,
    300: -0.24621152137305929,
    365: -0.40163493106148224,
}


def _our_declination(doy):
    return (atmos.CONST_MINDECL
            * torch.cos((doy + atmos.CONST_DAYSOFF) * atmos.CONST_RADPERDAY))


def test_declination_constants_are_vics():
    assert atmos.CONST_MINDECL == -0.4092797     # :73
    assert atmos.CONST_DAYSOFF == 11.25          # :74
    assert atmos.CONST_RADPERDAY == 0.017214     # :71


def test_potential_shortwave_uses_vics_declination():
    """The old `0.4093 sin(2 pi j / 365 - 1.405)` is 0.028 rad out of phase.

    That is 0.65 degrees of declination at worst -- about 1.6 days -- and it
    moves the daily-mean top-of-atmosphere flux by up to 4.3 W/m2 at 32 N and
    5.8 W/m2 at 48 N, which moves the cloud index and therefore the estimated
    longwave. This test would have failed by 1.14e-2 rad against VIC.
    """
    doy = _t(sorted(VIC_DECL))
    ref = _t([VIC_DECL[int(d)] for d in doy])
    assert _max_rel(_our_declination(doy), ref) < 1e-15

    old = 0.4093 * torch.sin(2.0 * math.pi * doy / 365.0 - 1.405)
    assert float((old - ref).abs().max()) > 1e-3     # the form that was there


def test_potential_shortwave_is_finite_and_seasonal():
    # The flux around VIC's geometry is ours -- VIC 5 has no solar constant --
    # so only its shape can be asserted: positive, and larger at midsummer.
    lat = _t([32.0, 48.0])
    summer = atmos.potential_shortwave(_t([182.0]).unsqueeze(1),
                                       lat.unsqueeze(0))
    winter = atmos.potential_shortwave(_t([1.0]).unsqueeze(1),
                                       lat.unsqueeze(0))
    assert torch.isfinite(summer).all() and torch.isfinite(winter).all()
    assert bool((summer > winter).all())
    assert bool((winter > 0).all())


# ---------------------------------------------------------------------------
# The constants themselves.
# ---------------------------------------------------------------------------
VIC_CONSTANTS = {
    'CONST_BOLTZ': 1.38065e-23,
    'CONST_AVOGAD': 6.02214e26,
    'CONST_RGAS': 8314.4675910000005,
    'CONST_MWDAIR': 28.966,
    'CONST_RDAIR': 287.0423113650487,
    'CONST_G': 9.80616,
    'CONST_PSTD': 101325.0,
    'CONST_LATVAP': 2501000.0,
    'CONST_STEBOL': 5.67e-08,
    'CONST_TKFRZ': 273.15,
    'LAPSE_RATE': -0.0065,
    'SEC_PER_DAY': 86400.0,
    'LATENT_HEAT_SLOPE': 2361.0,
}


def test_every_atmos_constant_equals_vics():
    for name, want in VIC_CONSTANTS.items():
        # A constant atmos *uses* but does not *define* is read through the
        # module that owns it -- CONST_STEBOL lives in snow.py and atmos
        # reaches it as _snow.CONST_STEBOL, so that one installed tensor is
        # what every module computes with. The test follows the same path.
        got = getattr(atmos, name, None)
        if got is None:
            got = getattr(atmos._snow, name)
        assert got == want, '%s: ours %.17g, VIC %.17g' % (name, got, want)


def test_svp_slope_defaults_are_vics_svp_b_and_svp_c():
    # initialize_parameters.c:101-102
    d = dict(zip(('svp_b', 'svp_c'), atmos.svp_slope.__defaults__))
    assert d['svp_b'] == 17.269 and d['svp_c'] == 237.3
