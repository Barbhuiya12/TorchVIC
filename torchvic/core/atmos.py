"""Atmospheric quantities the energy balance needs beyond the raw forcing:
air pressure, air density and the latent heat of vaporisation.

Where VIC has its own routine, that routine is what is reproduced, and
tests/test_atmos_formulas.py measures this module against VIC 5.0.1's
compiled code (the pristine upstream tree) over a grid of temperatures and
elevations:

    scale_height                vic_run/src/physics.c calc_scale_height
    air_pressure                vic_run/src/penman.c:206, hypsometric
    latent_heat_of_vaporization vic_run/src/physics.c:115
    air_density                 drivers/shared_all/src/forcing_utils.c:88
    air_density_penman          vic_run/src/penman.c:218, Handbook 4.2.4
    psychrometric_constant      vic_run/src/penman.c:213, Handbook 4.2.28
    svp, svp_slope              vic_run/src/svp.c (svp lives in snow.py)

**Two air densities, because VIC has two.** `forcing_utils.c` builds the
`force[].density` the energy balance consumes from the ideal gas law,
p / (R_d * T); `penman.c` computes a *local* r_air from the Handbook of
Hydrology's 0.003486 p / (275 + T) and uses it nowhere else. They disagree by
up to 0.72 per cent, which is not noise in a sensible-heat flux, so each is
named for where VIC uses it. `air_density` is the forcing one because that is
what every driver writes into `force[].density`.

**Where VIC takes the quantity from forcing.** The image driver reads PRESSURE
(kPa) and LONGWAVE from the forcing file and never derives either. So
`air_pressure` is a fallback for a dataset that lacks pressure -- VIC's own
hypsometric formula, but applied where VIC would have read a number -- and
`longwave_prata` is not VIC at all. Neither is a defect against VIC; both are
inputs VIC assumes it has been given.

Three quantities have no VIC routine to copy and are marked as what they are.

**Shortwave.** A daylight-period mean flux (as some gridded datasets publish
`srad`) becomes a daily mean as `srad * dayl / 86400`.

**Longwave.** VIC 5 does not derive it. `longwave_prata` is the
Prata (1996) clear-sky emissivity with a cloud correction driven by the ratio
of observed to potential shortwave. It is a parameterisation, not VIC code,
and it is the weakest link in this forcing chain; it is kept in one function
so it can be replaced by a reanalysis product without touching anything else.

**Potential shortwave and cloud fraction.** VIC 5 has neither: MTCLIM went
with VIC 4, and nothing in `vic_run/` or `drivers/` computes a clear-sky flux
or a cloud index. What VIC 5 does still carry is the *solar geometry* --
`compute_coszen.c` declines the sun with CONST_MINDECL, CONST_DAYSOFF and
CONST_RADPERDAY -- so the geometry here is VIC's even though the flux around
it is not.
"""
import math

import torch

from . import snow as _snow
from .snow import svp, CONST_TKFRZ
from .devconst import device_constant

# VIC's own values. Every one of these was printed out of a binary linked
# against the pristine VIC 5.0.1 tree and compared digit for digit; the file
# and line is the definition it was taken from.
CONST_BOLTZ = 1.38065e-23          # vic_physical_constants.h:80
CONST_AVOGAD = 6.02214e26          # vic_physical_constants.h:81
CONST_RGAS = CONST_AVOGAD * CONST_BOLTZ            # :90, derived as VIC does
CONST_MWDAIR = 28.966              # vic_physical_constants.h:84
CONST_RDAIR = CONST_RGAS / CONST_MWDAIR            # :91, 287.0423113650487
CONST_G = 9.80616                  # vic_physical_constants.h:78
CONST_PSTD = 101325.0              # vic_physical_constants.h:98
CONST_LATVAP = 2.501e6             # vic_physical_constants.h:115
# The temperature slope of the latent heat of vaporisation. VIC hard-codes it
# as a bare 2361. in the expression rather than giving it a param slot
# (physics.c:115), so there is no VIC name to borrow; it is named here only so
# that params.py can install it as a tensor and autograd can rank it.
LATENT_HEAT_SLOPE = 2361.0         # vic_run/src/physics.c:115
# CONST_STEBOL is defined once, in snow.py, and read through the module at
# call time (`_snow.CONST_STEBOL`) rather than copied here. A copy would be
# discovered by params.py as a second constant and installed as a second
# leaf tensor, so the functions in this module and the ones in snow.py would
# differentiate with respect to two different objects for one VIC symbol --
# and a sensitivity ranking off the raw gradients would under-state it.
# param.LAPSE_RATE, initialize_parameters.c:39
LAPSE_RATE = -0.0065
SEC_PER_DAY = 86400.0              # vic_physical_constants.h:41

# Handbook of Hydrology eqn 4.2.28, as VIC spells it: gamma = 1628.6 * p / lv.
# The number is c_p / 0.622 with c_p = 1013 J/kg/K, but VIC hard-codes the
# quotient rather than deriving it, so the quotient is what is carried here --
# deriving it from CONST_CPMAIR would land on 1628.6045, not 1628.6, and the
# psychrometric constant would then differ from VIC's in the fifth digit.
PSYCHROMETRIC_COEF = 1628.6        # vic_run/src/penman.c:213

# Solar declination, from VIC's own geometry rather than the textbook sine
# form. compute_coszen.c:60 is the only place VIC 5 still declines the sun,
# and its phase is 0.028 rad -- about 1.6 days -- away from the
# `0.4093 sin(2 pi j / 365 - 1.405)` this module used to carry. That is 0.65
# degrees of declination at worst, which moves the top-of-atmosphere flux by
# up to 5.8 W/m2 at 48 N and so moves the cloud index and the longwave with
# it. Small, and free to remove.
CONST_MINDECL = -0.4092797         # vic_physical_constants.h:73
CONST_DAYSOFF = 11.25              # vic_physical_constants.h:74
CONST_RADPERDAY = 0.017214         # vic_physical_constants.h:71


def scale_height(tair, elevation):
    """Atmospheric scale height, m.  physics.c:76 calc_scale_height."""
    return (CONST_RDAIR / CONST_G
            * ((tair + CONST_TKFRZ) + 0.5 * elevation * LAPSE_RATE))


def air_pressure(tair, elevation):
    """Surface air pressure, Pa.  penman.c:206, hypsometric.

    A fallback, not a step VIC takes in the image driver: there PRESSURE is a
    forcing variable read in kPa and multiplied by PA_PER_KPA. VIC uses this
    formula only inside penman() and canopy_assimilation(), where it needs a
    pressure and has not been handed one -- which is exactly the situation a
    dataset without a pressure field puts us in.
    """
    return CONST_PSTD * torch.exp(-elevation / scale_height(tair, elevation))


def latent_heat_of_vaporization(temp):
    """J/kg.  physics.c:115 calc_latent_heat_of_vaporization.

    Handbook of Hydrology eqn 4.2.1, with Ts taken as Tair as VIC takes it.
    """
    return CONST_LATVAP - LATENT_HEAT_SLOPE * temp


def air_density(tair, pressure):
    """kg/m3.  drivers/shared_all/src/forcing_utils.c:88 air_density().

    This is the density every VIC driver writes into `force[].density`, and
    therefore the density the whole energy balance sees -- calc_sensible_heat,
    latent_heat_from_snow, the canopy vapour flux, the lake module. It is the
    ideal gas law with the dry-air gas constant.

    It is *not* penman.c's `r_air`, which this function used to return. That
    one is local to penman() and is 0.5-0.7 per cent smaller across the range
    of states VIC runs over; measured against the compiled VIC 5.0.1 the old
    form was wrong by up to 1.10e-2 kg/m3 (7.2e-3 relative) at -40 C, which
    propagates straight into the sensible heat flux. Where the Penman form is
    genuinely wanted, call `air_density_penman`.
    """
    return pressure / (CONST_RDAIR * (CONST_TKFRZ + tair))


def air_density_penman(tair, pressure):
    """kg/m3.  penman.c:218, Handbook of Hydrology eqn 4.2.4.

    Kept separate because VIC keeps it separate: this expression appears once
    in VIC, inside penman(), and is not the density any other routine uses.
    """
    return 0.003486 * pressure / (275.0 + tair)


def psychrometric_constant(pressure, lv):
    """Psychrometric constant, Pa/C.  penman.c:213, Handbook eqn 4.2.28.

    `pressure` in Pa, `lv` in J/kg. VIC inlines this as a bare expression in
    penman() rather than giving it a routine, which is why it was missing
    here: there was no function to port. It is carried now so the coefficient
    is a named module constant that params.py can install as a tensor, instead
    of a literal repeated at each call site where autograd cannot reach it.
    """
    return PSYCHROMETRIC_COEF * pressure / lv


def svp_slope(temp, svp_b=17.269, svp_c=237.3):
    """d(svp)/dT.  svp.c svp_slope."""
    # svp.c 5.1.0:40-41. The constant over a tensor is formed as a tensor
    # division: torch evaluates `number / tensor` as reciprocal-then-
    # multiply, which rounds an ulp from the C's quotient.
    num = (torch.full_like(temp, svp_b * svp_c) if torch.is_tensor(temp)
           else svp_b * svp_c)
    return num / ((svp_c + temp) * (svp_c + temp)) * svp(temp)


def daily_mean_shortwave(srad, dayl):
    """Daylight-average shortwave -> daily mean, W/m2.

    Not a modelling choice: `srad` is defined over the daylight period only.
    """
    return srad * dayl / SEC_PER_DAY


def coszen(latitude_deg, longitude_deg, time_zone_lng_deg, day_in_year,
           second):
    """Cosine of the solar zenith angle.  compute_coszen.c:31-79.

    VIC's own, line for line, including the fact that `coshss` -- the cosine
    of the half sunset hour angle -- is computed, clamped, and then never
    used: the return is `cosegeom * cosh + sinegeom` regardless. The clamp
    is reproduced for the same reason the rest is, and left unused for the
    same reason.

    The value goes negative at night, which is what `faparl` and
    `calc_Nscale_factors` test against PHOTO_ZENITHMINPAR. `day_in_year` is
    1-based and `second` is the second of the day.
    """
    lat = torch.as_tensor(latitude_deg) * math.pi / 180.0
    hour = torch.as_tensor(second, dtype=lat.dtype) / 3600.0
    j = torch.as_tensor(day_in_year, dtype=lat.dtype)
    decl = CONST_MINDECL * torch.cos((j + CONST_DAYSOFF) * CONST_RADPERDAY)
    cosegeom = torch.cos(lat) * torch.cos(decl)
    sinegeom = torch.sin(lat) * torch.sin(decl)
    hour_offset = ((torch.as_tensor(time_zone_lng_deg, dtype=lat.dtype)
                    - torch.as_tensor(longitude_deg, dtype=lat.dtype))
                   * 24.0 / 360.0)
    cosh = torch.cos((hour + hour_offset - 12.0) * math.pi / 12.0)
    return cosegeom * cosh + sinegeom


def potential_shortwave(day_of_year, latitude_deg):
    """Clear-sky daily mean shortwave at the surface, W/m2.

    Top-of-atmosphere insolation from the standard solar geometry, reduced by
    a bulk clear-sky transmissivity. Used only as the denominator of the cloud
    index in `longwave_prata`, so it needs to be the right shape rather than
    the right absolute value.

    The *geometry* is VIC's -- the declination is compute_coszen.c:60 with
    VIC's own CONST_MINDECL/CONST_DAYSOFF/CONST_RADPERDAY. The flux around it
    is not: VIC 5 has no solar constant and no clear-sky transmissivity,
    because MTCLIM left with VIC 4, so `s0` and the 0.75 are ours and cannot
    be checked against VIC. `day_of_year` is 1-based, as VIC's day_in_year is.
    """
    lat = torch.as_tensor(latitude_deg) * math.pi / 180.0
    j = torch.as_tensor(day_of_year, dtype=torch.get_default_dtype()
                        if not torch.is_floating_point(
                            torch.as_tensor(day_of_year))
                        else None)
    # compute_coszen.c:60. VIC cosines the declination off the winter
    # solstice; the equivalent sine form differs only in its phase, and the
    # phase VIC uses is the one that has to be reproduced.
    decl = CONST_MINDECL * torch.cos((j + CONST_DAYSOFF) * CONST_RADPERDAY)
    ws = torch.acos(torch.clamp(-torch.tan(lat) * torch.tan(decl),
                                min=-1.0, max=1.0))
    dr = 1.0 + 0.033 * torch.cos(2.0 * math.pi * j / 365.0)
    # Daily mean top-of-atmosphere flux, W/m2
    s0 = 1367.0
    toa = (s0 / math.pi) * dr * (ws * torch.sin(lat) * torch.sin(decl)
                                 + torch.cos(lat) * torch.cos(decl)
                                 * torch.sin(ws))
    return 0.75 * toa


def longwave_prata(tair, vp, cloud_fraction):
    """Downwelling longwave, W/m2. Prata (1996) clear sky, cloud-corrected.

    NOT a VIC routine and not measurable against one: VIC 5 takes longwave as
    an input forcing and a dataset may not supply it. `tair` is in C, `vp` in
    Pa, `cloud_fraction` in [0, 1].
    """
    tk = tair + CONST_TKFRZ
    # Prata's w: precipitable water proxy, from vapour pressure in hPa
    w = 46.5 * (vp / 100.0) / tk
    eps_clear = 1.0 - (1.0 + w) * torch.exp(-(1.2 + 3.0 * w) ** 0.5)
    eps = eps_clear * (1.0 - cloud_fraction) + cloud_fraction
    return eps * _snow.CONST_STEBOL * tk ** 4


def cloud_fraction_from_shortwave(sw_obs, sw_pot, floor=0.0):
    """Cloud fraction from the shortfall of observed against potential sun."""
    ratio = torch.clamp(sw_obs / torch.clamp(sw_pot, min=1.0), min=0.0, max=1.0)
    # maximum/minimum rather than clamp: clamp refuses to mix a Python number
    # with a tensor bound, and `floor` must be allowed to be a tensor so it
    # can carry a gradient.
    lo = device_constant(floor, ratio.dtype, ratio.device)
    return torch.minimum(torch.maximum(1.0 - ratio, lo),
                         torch.ones_like(ratio))
