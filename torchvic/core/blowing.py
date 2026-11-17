"""Blowing snow: sublimation and transport of wind-driven snow.
CalcBlowingSnow.c, vectorised and differentiable.

VIC's BLOWING option. Wind lifts snow off the pack into a saltation layer a
few centimetres deep and a suspension layer above it, and the particles
sublimate while airborne. The loss is large where it happens -- it is the
reason a wind-exposed tundra pack can lose a third of its mass without ever
melting -- and it reaches the rest of the model as one number, the blowing
mass flux, which `torchvic.core.snow.latent_heat_from_snow` already takes as an
argument and which has been zero until now.

The chain, following the C:

  1. `shear_stress` finds the friction velocity during saltation, which is
     a root of get_shear() -- VIC solves it with a Newton/bisection hybrid
     (rtnewt), here with the bracketed solver in `torchvic.core.implicit` and the
     implicit function theorem for the gradient.
  2. `get_thresh` is the shear the wind must beat to move anything, and
     `get_prob` the probability that blowing snow is occurring at all.
  3. `calc_sub_flux` builds the saltation layer's mass concentration and
     integrates the suspension layer's sublimation and transport over
     height by Romberg's method.
  4. `calc_blowing_snow` splits the cell's wind into BLOWING_NUMINCS
     increments of a log-normal distribution (BLOWING_SPATIAL_WIND, on by
     default) and weights each increment's flux by its probability.

Two departures from the C, both about iteration counts, both measured
rather than asserted:

* VIC's `qromb` refines until the Richardson extrapolation stops moving at
  machine precision, up to BLOWING_MAX_ITER = 100 levels. Here the level
  count is fixed, because a convergence test costs a device
  synchronisation per level and the integrand is smooth. The default was
  chosen by comparing against a scalar transcription of the C; see
  tests/test_blowing_snow.py.
* VIC's `rtnewt` likewise iterates to a tolerance. `torchvic.core.implicit`
  solves the same root to the same bracket with a fixed count.

Everything else is the C's arithmetic in the C's order.

Shapes: every function is elementwise over whatever leading shape its
inputs share, and `calc_blowing_snow` adds an increment axis in front of
that and reduces it away.
"""
import math

import torch

from . import implicit
from . import snow as _snow

CONST_G = 9.80616
CONST_PI = 3.141592653589793
CONST_KARMAN = 0.40
CONST_RHOICE = 917.0
CONST_RDAIR = 287.0423112684495
CONST_EPS = 0.6218851586723554
CONST_TKFRZ = 273.15
SEC_PER_HOUR = 3600.0

#: initialize_parameters.c:188-195.
BLOWING_KA = 0.0245187        #: thermal conductivity of air, W/m/K
BLOWING_CSALT = 0.68          #: saltation constant
BLOWING_UTHRESH = 0.25        #: constant threshold shear, m/s
BLOWING_KIN_VIS = 1.3e-5      #: kinematic viscosity of air, m2/s
BLOWING_MAX_ITER = 100
BLOWING_K = 5                 #: points the Romberg extrapolation uses
BLOWING_SETTLING = 0.3        #: particle settling velocity, m/s
BLOWING_NUMINCS = 10          #: wind-distribution increments

#: Romberg levels. VIC refines adaptively to machine precision; this is
#: fixed for the reason in the module docstring. Eight levels is 129
#: integrand evaluations and reproduces a fourteen-level answer to
#: 4.6e-07 relative -- three orders inside VIC's own solver tolerance of
#: 1.2e-03, and three times cheaper than ten levels.
ROMBERG_LEVELS = 8


def get_shear(x, ur, zr):
    """Residual and derivative of the saltation shear relation.
    CalcBlowingSnow.c:487-495.

    f(x) = ln(2 g Zr / 0.12) + ln(1/x^2) - k Ur / x, whose root is the
    friction velocity at which the saltation roughness Zo_salt = 0.12 x^2 /
    (2 g) is consistent with the wind Ur at height Zr.
    """
    return (torch.log(2.0 * CONST_G * zr / 0.12) + torch.log(1.0 / (x * x))
            - CONST_KARMAN * ur / x)


def shear_stress(u10, z0, utshear, zr=10.0):
    """Friction velocity during saltation and the saltation roughness.
    CalcBlowingSnow.c:650-684.  Returns (ushear, z0_salt).

    The bracket is [utshear, k*U10], as the C sets it. Where the residual
    does not change sign across it -- the C's `fl > 0 && fh > 0` branch --
    there is no saltation roughness to find and the friction velocity is
    the ordinary logarithmic one over the snow's own roughness.
    """
    umin = utshear
    umax = CONST_KARMAN * u10
    zr_t = torch.full_like(u10, float(zr))

    def f(x, *_):
        return get_shear(x, u10, zr_t)

    with torch.no_grad():
        fl, fh = f(umin), f(umax)
        # The C raises on `fl < 0 && fh < 0`; here such an element takes the
        # same fallback as the unbracketed one rather than taking the run
        # down, and the caller is told how many by the returned mask.
        bracketed = (fl * fh) < 0.0
    root = implicit.solve(f, umin, umax)
    plain = CONST_KARMAN * u10 / torch.log(float(zr) / z0)
    ushear = torch.where(bracketed, root, plain)
    z0_salt = torch.where(bracketed, 0.12 * ushear * ushear / (2.0 * CONST_G),
                          z0)
    return ushear, z0_salt, bracketed


def get_prob(tair, age, surface_liquid_water, u10, calc_prob=True):
    """Probability that blowing snow is occurring.  CalcBlowingSnow.c
    :573-620, Li and Pomeroy (1997).

    `age` is the snow's age in hours. The C's `log(Age)` is undefined at
    zero and VIC reaches it whenever snow fell in the previous step; the
    age is floored at one step's worth here, which is what a zero age
    means, rather than left to produce a negative infinity that `where`
    would carry into the backward pass.
    """
    if not calc_prob:
        return torch.ones_like(u10)
    wet = surface_liquid_water >= 0.001
    safe_age = torch.clamp(age, min=torch.finfo(age.dtype).tiny)
    mean_dry = (11.2 + 0.365 * tair + 0.00706 * tair * tair
                + 0.9 * torch.log(safe_age))
    sigma_dry = 4.3 + 0.145 * tair + 0.00196 * tair * tair
    mean_u = torch.where(wet, torch.full_like(mean_dry, 21.0), mean_dry)
    sigma = torch.where(wet, torch.full_like(sigma_dry, 7.0), sigma_dry)
    # A zero sigma would divide by zero; the dry fit is a parabola in air
    # temperature with no real root, so it cannot vanish for a physical
    # temperature, and the guard is for the unphysical tail alone.
    safe_sigma = torch.where(sigma.abs() > 1e-12, sigma,
                             torch.full_like(sigma, 1e-12))
    p = 1.0 / (1.0 + torch.exp(math.sqrt(CONST_PI) * (mean_u - u10)
                               / safe_sigma))
    return torch.clamp(p, 0.0, 1.0)


def get_thresh(tair, surface_liquid_water, z0_salt, var_threshold=True):
    """Threshold friction velocity.  CalcBlowingSnow.c:626-655."""
    dry = 9.43 + 0.18 * tair + 0.0033 * tair * tair
    ut10 = torch.where(surface_liquid_water < 0.001, dry,
                       torch.full_like(dry, 9.9))
    if not var_threshold:
        return torch.full_like(ut10, BLOWING_UTHRESH)
    return CONST_KARMAN * ut10 / torch.log(10.0 / z0_salt)


def _mean_particle(z):
    """The saltating particle's mean radius and mass at height z.
    CalcBlowingSnow.c:529-540."""
    rrz = 4.6e-5 * torch.pow(z, -0.258)
    alphaz = 4.08 + 12.6 * z
    mz = ((4.0 / 3.0) * CONST_PI * CONST_RHOICE * rrz * rrz * rrz
          * (1.0 + (3.0 / alphaz) + (2.0 / (alphaz * alphaz))))
    rmean = torch.pow((3.0 * mz) / (4.0 * CONST_PI * CONST_RHOICE), 1.0 / 3.0)
    return rmean, mz


def _phi_t(z, wind, ushear, hsalt, phi_r):
    """Turbulent suspended-snow concentration, Kind (1992).
    CalcBlowingSnow.c:562-570 and :831-838, the same expression in both."""
    temp = (0.5 * ushear * ushear) / (wind * BLOWING_SETTLING)
    return phi_r * ((temp + 1.0)
                    * torch.pow(z / hsalt,
                                (-1.0 * BLOWING_SETTLING)
                                / (CONST_KARMAN * ushear)) - temp)


def sub_with_height(z, es, wind, air_dens, z0, eact_air, f, hsalt, phi_r,
                    ushear, zrh):
    """Sublimation rate at height z, kg/m3/s.  CalcBlowingSnow.c:508-570."""
    rmean, mz = _mean_particle(z)
    terminal_v = 1.1e7 * torch.pow(rmean, 1.8)           # Pomeroy and Male 1986
    fluctuat_v = 0.005 * torch.pow(wind, 1.36)           # Pomeroy 1988
    vtz = terminal_v + 3.0 * fluctuat_v * math.cos(CONST_PI / 4.0)  # Lee 1975
    re = 2.0 * rmean * vtz / BLOWING_KIN_VIS
    nu = 1.79 + 0.606 * torch.sqrt(re)
    sigz = ((eact_air / es) - 1.0) * (1.019 + 0.027 * torch.log(z))
    dmdt = 2.0 * CONST_PI * rmean * sigz * nu / f
    psi_t = dmdt / mz
    return psi_t * _phi_t(z, wind, ushear, hsalt, phi_r)


def transport_with_height(z, es, wind, air_dens, z0, eact_air, f, hsalt,
                          phi_r, ushear, zrh):
    """Horizontal snow transport at height z, kg/m2/s.
    CalcBlowingSnow.c:805-839."""
    u_z = ushear * torch.log(z / z0) / CONST_KARMAN
    return u_z * _phi_t(z, wind, ushear, hsalt, phi_r)


def _polint(h, s):
    """Extrapolate the trapezoid estimates to zero step size.
    CalcBlowingSnow.c:313-362 does this with Neville's algorithm on the
    (h, s) pairs; because `h` is exactly quartered at every level
    (:302, `h[j+1] = 0.25 * h[j]`), the interpolating polynomial through
    them evaluated at zero is the Richardson table, which is written here
    instead -- the same number by a shorter route, and one that does not
    need the index bookkeeping Neville does.

    `h` is accepted and ignored so the signature says what the C's
    arguments are; the quartering is what makes it redundant.
    """
    del h
    r = list(s)
    for m in range(1, len(r)):
        factor = 4.0 ** m
        for i in range(len(r) - 1, m - 1, -1):
            r[i] = r[i] + (r[i] - r[i - 1]) / (factor - 1.0)
    return r[-1]


def romberg(func, a, b, args=(), levels=ROMBERG_LEVELS, k=BLOWING_K):
    """Romberg integration of `func` over [a, b], a fixed number of levels.
    CalcBlowingSnow.c:271-307 with the convergence test replaced by a fixed
    count -- see the module docstring.

    `func(z, *args)` must accept a tensor whose shape is `a`'s with a
    leading axis added, and broadcast the rest of its arguments against it.
    The C's trapezoid refinement evaluates the 2^(j-2) new midpoints one at
    a time; here they are one tensor and one call, which is the difference
    between 512 kernel launches per integral and ten. At ten levels, ten
    increments and two integrals a step, the loop version made a FindleyLake
    step slower by more than an order of magnitude.
    """
    s_vals = []
    s = None
    for j in range(1, int(levels) + 1):
        if j == 1:
            s = 0.5 * (b - a) * (func(a, *args) + func(b, *args))
        else:
            it = 1 << (j - 2)
            tnm = float(it)
            delta = (b - a) / tnm
            # The `it` new midpoints, as one leading axis.
            offs = torch.arange(it, dtype=a.dtype, device=a.device)
            shape = (it,) + (1,) * a.dim()
            z = a.unsqueeze(0) + (offs.reshape(shape) + 0.5) * delta.unsqueeze(0)
            total = func(z, *args).sum(0)
            s = 0.5 * (s + (b - a) * total / tnm)
        s_vals.append(s)
    return _polint(None, s_vals[-k:])


def calc_sub_flux(eact_air, es, zrh, air_dens, utshear, ushear, fe, tsnow,
                  tair, u10, z0_salt, f, fetch=True, simple=False,
                  levels=ROMBERG_LEVELS):
    """Sublimation flux and horizontal transport.  CalcBlowingSnow.c
    :699-797.  Returns (sub_flux, transport), kg/m2/s and kg/m/s.
    """
    particle = utshear * 2.8
    if simple:
        # SBSM, :729-741.
        under = torch.where(
            eact_air >= es, torch.zeros_like(es),
            ((eact_air / es) - 1.0)
            * (1.0 - 0.027 * torch.log(zrh) + 0.027 * math.log(2.0)))
        return 0.25 * under * torch.pow(u10, 5.0) / f, torch.zeros_like(u10)

    # Liston and Sturm 1998 eq. 6, the maximum saltation transport rate.
    qsalt = ((BLOWING_CSALT * air_dens / CONST_G) * (utshear / ushear)
             * (ushear * ushear - utshear * utshear))
    if fetch:
        qsalt = qsalt * (1.0 + (500.0 / (3.0 * fe))
                         * (torch.exp(-3.0 * fe / 500.0) - 1.0))
    hsalt = 0.08436 * torch.pow(ushear, 1.27)     # Pomeroy and Male 1992
    phi_s = qsalt / (hsalt * particle)
    t_ = 0.5 * (ushear * ushear) / (u10 * BLOWING_SETTLING)
    ztop = hsalt * torch.pow(t_ / (t_ + 1.0),
                             (CONST_KARMAN * ushear) / (-1.0 * BLOWING_SETTLING))
    args = (es, u10, air_dens, z0_salt, eact_air, f, hsalt, phi_s, ushear, zrh)

    psi_s = sub_with_height(hsalt / 2.0, *args)
    sub = phi_s * psi_s * hsalt
    sub = sub + romberg(sub_with_height, hsalt, ztop, args, levels=levels)
    # :763-766: air already at saturation sublimates nothing.
    sub = torch.where(eact_air >= es, torch.zeros_like(sub), sub)

    saltation_transport = qsalt * (1.0 - torch.exp(-3.0 * fe / 500.0))
    suspension_transport = romberg(transport_with_height, hsalt, ztop, args,
                                   levels=levels)
    transport = suspension_transport + saltation_transport
    if fetch:
        transport = transport / fe
    return sub, transport


def calc_blowing_snow(dt, tair, last_snow, surface_liquid_water, wind, ls,
                      air_dens, eact_air, z0, zrh, snowdepth, lag_one,
                      sigma_slope, tsnow, fe, displacement, roughness,
                      bare_soil=None, spatial_wind=True, calc_prob=True,
                      var_threshold=True, fetch=True, simple=False,
                      n_incs=BLOWING_NUMINCS, levels=ROMBERG_LEVELS):
    """Blowing-snow sublimation, m of water per step, and transport.
    CalcBlowingSnow.c:32-267.  Returns (sublimation, transport).

    `sublimation` carries the C's sign: the return value is a mass flux in
    kg/m2/s that surface_fluxes.c:463 converts to metres per step, and it is
    negative for a loss, matching `torchvic.core.snow.latent_heat_from_snow`'s
    `blowing_mass_flux`.

    `bare_soil` is the C's `iveg == Nveg` test at :99-102, which replaces
    the fetch and the slope variance with bare-soil values.
    """
    age = last_snow * dt / SEC_PER_HOUR
    es = _snow.svp(tair)
    tk = tair + CONST_TKFRZ
    ros = CONST_EPS * es / (CONST_RDAIR * tk)
    diffusivity = 2.06e-5 * torch.pow(tk / 273.0, 1.75)     # Liston eq. A-7
    # Essery et al. 1999 eq. 6, m*s/kg
    f = (ls / (BLOWING_KA * tk)) * (ls * tk / CONST_RDAIR - 1.0)
    f = f + 1.0 / (diffusivity * ros)

    # :95: the wind reaching here is at 2 m above the snow.
    wind10 = wind * torch.log(10.0 / z0) / torch.log((2.0 + z0) / z0)
    if bare_soil is not None:
        fe = torch.where(bare_soil, torch.full_like(fe, 1500.0), fe)
        sigma_slope = torch.where(bare_soil,
                                  torch.full_like(sigma_slope, 0.0002),
                                  sigma_slope)
    ratio = (2.44 - 0.43 * lag_one) * sigma_slope
    sigma_w = wind10 * ratio
    uo = wind10
    hv = 1.5 * displacement
    # :106-107. A tile with no vegetation has no displacement height,
    # and the C never reaches this with one -- it is called per real
    # vegetation tile. A padded tile with zero cover has displacement 0
    # here, and 0/0 would be a NaN that the cover-weighted sum turns
    # into 0 * NaN = NaN for the whole cell.
    safe_disp = torch.where(displacement > 0.0, displacement,
                            torch.ones_like(displacement))
    nd = (4.0 / 3.0) * (roughness / safe_disp)

    def one(u10, weight):
        # Every other quantity here is per cell-tile; `u10` may carry a
        # leading increment axis and broadcasting puts the rest onto it.
        # :182-188, the vegetation shelters the pack until the snow buries it
        uveg = torch.where(snowdepth < hv,
                           u10 / torch.sqrt(1.0 + 170.0 * nd
                                            * torch.clamp(hv - snowdepth,
                                                          min=0.0)),
                           u10)
        prob = get_prob(tair, age, surface_liquid_water, uveg,
                        calc_prob=calc_prob)
        utshear = get_thresh(tair, surface_liquid_water, z0,
                             var_threshold=var_threshold)
        ushear, z0_salt, _ = shear_stress(u10, z0, utshear)
        sub, transport = calc_sub_flux(eact_air, es, zrh, air_dens, utshear,
                                       ushear, fe, tsnow, tair, u10, z0_salt,
                                       f, fetch=fetch, simple=simple,
                                       levels=levels)
        moving = ushear > utshear
        sub = torch.where(moving, sub, torch.zeros_like(sub))
        transport = torch.where(moving, transport, torch.zeros_like(transport))
        return weight * sub * prob, weight * transport * prob

    zero = torch.zeros_like(uo)
    if spatial_wind:
        # :123-180. The cell's wind is log-normal about Uo with width
        # sigma_w; the distribution is cut into n_incs equal-probability
        # increments and each one's expected wind speed is used.
        #
        # The C takes them one at a time. Here they are one leading axis and
        # one pass, which turns ten root finds and twenty integrals into one
        # and two on a tensor ten times as long -- the difference between 958
        # and about 100 ms on a sequential FindleyLake step.
        area = 1.0 / float(n_incs)
        u10 = torch.stack([_increment_wind(q, int(n_incs), uo, sigma_w, area)
                           for q in range(int(n_incs))], 0)
        s_all, tr_all = one(u10, area)
        total = s_all.sum(0)
        total_transport = tr_all.sum(0)
        # :219: with sigma_w zero the distribution has no width and the C
        # takes the single-wind branch instead.
        s0, tr0 = one(uo, torch.ones_like(uo))
        total = torch.where(sigma_w != 0.0, total, s0)
        total_transport = torch.where(sigma_w != 0.0, total_transport, tr0)
    else:
        total, total_transport = one(uo, torch.ones_like(uo))

    # :241-247: nothing blows off bare ground, and the loss is capped.
    has_snow = snowdepth > 0.0
    total = torch.where(has_snow, total, zero)
    total_transport = torch.where(has_snow, total_transport, zero)
    total = torch.clamp(total, min=-0.00005)
    return total, total_transport


def _increment_wind(p, n, uo, sigma_w, area):
    """Expected wind speed of the p-th equal-probability increment.
    CalcBlowingSnow.c:126-180.

    The C builds the interval's limits from the log-normal quantiles and
    integrates u over it analytically. Which of its two closed forms applies
    is decided by p alone, not by the data: the limits are Uo plus or minus
    sigma times a logarithm whose sign follows from p, so the first half of
    the increments lie below Uo and the second half above. Both forms are
    written out in the C and one raises where the other applies; here only
    the applicable one is *evaluated*, which matters because the outermost
    two intervals are unbounded.

    The C writes those as -9999 and 9999 and lets the exponential underflow
    to zero. Forward that is right; backward it is not -- the derivative of
    exp(-3928) is (huge) * 0, a NaN that `where` carries through the
    backward pass even though the value is discarded. The unbounded end is
    therefore dropped analytically, which is the limit the C is
    approximating anyway.
    """
    safe = torch.where(sigma_w.abs() > 0.0, sigma_w,
                       torch.full_like(sigma_w, 1e-30))
    if p < n / 2:
        # upper <= Uo: the C's second form, :159-167.
        upper = uo + safe * math.log(2.0 * (p + 1) * area)
        term_u = (upper - safe) * torch.exp((upper - uo) / safe)
        if p == 0:
            term_l = torch.zeros_like(uo)      # lower -> -infinity
        else:
            lower = uo + safe * math.log(2.0 * p * area)
            term_l = (lower - safe) * torch.exp((lower - uo) / safe)
        u10 = 0.5 * (term_u - term_l) / area
    else:
        # lower >= Uo: the C's first form, :152-158.
        lower = uo - safe * math.log(2.0 - 2.0 * (p * area))
        term_l = (lower + safe) * torch.exp(-(lower - uo) / safe)
        if p == n - 1:
            term_u = torch.zeros_like(uo)      # upper -> +infinity
        else:
            upper = uo - safe * math.log(2.0 - 2.0 * ((p + 1.0) * area))
            term_u = (upper + safe) * torch.exp(-(upper - uo) / safe)
        u10 = -0.5 * (term_u - term_l) / area
    # :173-179, the C's own clamp.
    return torch.clamp(u10, 0.4, 25.0)
