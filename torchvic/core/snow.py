"""Differentiable PyTorch implementation of VIC's snow pack.

This is a *reimplementation*, not a wrapper, for the same reason as
`torchvic.core.soil`. Verification
against the C is therefore the central obligation, and it is done twice over:
tests/test_vic_snow.py checks the individual functions against the
`snow` mode of tests/oracle/ref_soil.c, and
tests/test_snow_formulas.py checks them, plus `snow_step` as a whole,
against tests/oracle/ref_snow.c -- both oracles link *pristine* VIC trees
(5.1.0 and 5.0.1), so that a defect introduced in a modified copy of the C
cannot appear as agreement.

Scope. The closed-form functions -- new snow density, pack compaction and
depth, albedo decay -- are direct transcriptions. snow_melt.c and
SnowPackEnergyBalance.c are here too: the surface energy balance is a
residual that `torchvic.core.implicit` roots and differentiates through the
implicit function theorem rather than by unrolling VIC's `root_brent`, and
the rest of snow_melt() around it is transcribed line for line. See
`snow_step` for the list of what is deliberately absent.

Every `file:line` citation in this module is against that pristine tree, at
reference/VIC-VIC.5.0.1/vic/. The `vic/` directory in this repository has a
twenty-line-shorter header on several of these files and its own changes
besides, so its line numbers do not agree and neither, in at least one place,
does its arithmetic.

Not here, and not approximated anywhere: snow_intercept.c and massrelease.c
(canopy snow, which belongs with `torchvic.core.canopy`), CalcBlowingSnow.c (the
BLOWING option), and ice_melt.c, which is the lake model's pack and not this
one. SPATIAL_SNOW *is* here -- `snow_coverage` is calc_snow_coverage.c and
`advected_sensible_heat` is its own file -- but it is off unless
`basin.simulate` is asked for it, and with it off solve_snow.c:401-407 sets
the cover fraction to 1 wherever there is a pack and 0 where there is not,
which is what the pack functions below assume.

Shapes: every function is batched over a leading cell axis, matching
`torchvic.core.soil`.

Parameter defaults are VIC's, from
drivers/shared_all/src/initialize_parameters.c, checked digit for digit
against the pristine tree rather than read off ours. They are arguments
rather than constants so they can be varied per run.

A note on `last_snow`. It is an integer count of time steps since the last
snowfall, and it enters the albedo decay as `last_snow ** b` with b < 1. The
derivative of that is unbounded at zero. `last_snow` is a discrete counter,
not a run parameter, so no gradient is expected to flow through it
and none does -- but anything that later makes it continuous must revisit
this.
"""
import torch

from . import version as core_version
from . import xmath
from .devconst import device_constant
from . import fusion, graphsafe


def _like(ref, value):
    """`value` broadcast to the shape of `ref`, number or tensor alike.

    torch.full_like insists on a Python number, which silently makes every
    constant it touches undifferentiable: the tensor is read for its value and
    then dropped from the graph. Broadcasting costs the same and keeps the
    parameter connected.
    """
    return device_constant(value, ref.dtype,
                           ref.device) * torch.ones_like(ref)


# VIC 5.1.0 parameters with no 5.0.1 counterpart (initialize_parameters.c
# 5.1.0:133-134); read only under core_version 5.1.0.
SNOW_NEW_SNOW_DENS_MAX = 400.0
SNOW_DEPTH_THRES = 1.e-8

# vic/vic_run/include/vic_physical_constants.h
SEC_PER_DAY = 86400.0
GRAMS_PER_KG = 1000.0


def c_to_f(t):
    """Celsius to Fahrenheit, matching the C_TO_F macro."""
    return t * 9.0 / 5.0 + 32.0


def new_snow_density(air_temp, method='bras', dens_max=None,
                     snow_density=50.0,
                     bras_denom=100.0,
                     snt_c1=67.92, snt_c2=51.25, snt_c3=2.59):
    """Density of newly fallen snow, kg/m3.  snow_utility.c:207-239.

    `method` selects VIC's SNOW_DENSITY option. 'bras' is VIC's default
    (initialize_options.c:68) and is equation 6.2 of Bras (1990); 'snthrm'
    is Hedstrom and Pomeroy (1998).

    The C branches on air temperature in Fahrenheit; here both sides are
    evaluated and selected with `where`, so the result is the same and the
    expression stays differentiable.

    `dens_max` caps the result and defaults to no cap, which is VIC 5.0.1:
    snow_utility.c:226-232 returns the Bras quadratic unbounded and
    snow_utility.c:219-220 the SNTHRM exponential unbounded. This used to
    clamp unconditionally at 400 kg/m3, which is `param.SNOW_NEW_SNOW_DENS_MAX`
    -- a parameter the `vic/` tree in this repository carries and pristine
    VIC 5.0.1 does not have at all. That is the whole reason the audit oracle
    (tests/oracle/ref_snow.c) links upstream rather than our own copy of the
    C: verified against the wrong authority, the cap looked like agreement.

    The size of the difference depends entirely on how warm it is allowed to
    snow. Against build/ref_snow at +30 C the cap gave 400 where VIC 5.0.1
    gives 789.6 -- 49% low, and on the SNTHRM branch a factor of 1.4e4 -- but
    with VIC's own precipitation partition (param.SNOW_MAX_SNOW_TEMP = 0.5 C)
    the warmest snowfall reaches only 158 kg/m3 on the Bras branch, so the
    cap does not bind in the configuration VIC actually runs. It is a real
    disagreement with 5.0.1 and an inert one; both halves of that belong in
    the record.
    """
    if method == 'bras':
        tf = c_to_f(air_temp)
        # The C writes this as (t/d) * (t/d), not (t/d)**2. Same value in
        # exact arithmetic, different rounding in floating point; matching the
        # operation order makes the comparison bit-exact rather than merely
        # close.
        ratio = tf / bras_denom
        warm = snow_density + GRAMS_PER_KG * ratio * ratio
        density = torch.where(tf > 0.0, warm,
                              _like(air_temp, snow_density))
    elif method == 'snthrm':
        density = snt_c1 + snt_c2 * torch.exp(air_temp / snt_c3)
    else:
        raise ValueError('unknown method: %r' % (method,))

    if dens_max is None and core_version.at_least('5.1.0'):
        # VIC 5.1.0 caps it, snow_utility.c 5.1.0:218-222, at
        # param.SNOW_NEW_SNOW_DENS_MAX = 400 (initialize_parameters.c
        # 5.1.0:133), everywhere new snow gets a density (snow_density,
        # solve_snow.c:352).
        dens_max = SNOW_NEW_SNOW_DENS_MAX
    if dens_max is not None:
        density = torch.clamp(density, max=dens_max)
    return density


def snow_albedo(new_snow, new_snow_albedo, swq, cold_content, dt, last_snow,
                melting, trace_snow=0.03,
                accum_a=0.94, accum_b=0.58,
                thaw_a=0.82, thaw_b=0.46):
    """Snow surface albedo.  snow_utility.c:248.

    US Army Corps of Engineers decay, as a function of snow age and whether
    the pack is accumulating or melting.

    `dt` is the time step in seconds; `last_snow` counts time steps since
    the last snowfall; `melting` is VIC's MELTING flag. The C's three-way
    branch becomes two nested `where` calls over the same conditions, in the
    same order.

    `trace_snow` defaults to the C's own 0.03, which is a number whose unit
    is written down nowhere in VIC: it is compared against whatever units
    the caller's `new_snow` is in, and solve_snow.c's are millimetres --
    snow_melt.c:103 is the proof, `SnowFall = snowfall / MM_PER_M`. A
    caller working in metres must pass `TRACE_SNOW_M`, and `torchvic.model.basin`
    does.
    """
    age_days = last_snow * dt / SEC_PER_DAY

    # pow(x, b) has an unbounded derivative at x = 0; clamp only the value fed
    # to the power, not the result, so the forward value is unchanged for
    # age_days = 0 (0 ** b == 0) while the backward pass stays finite.
    safe_age = torch.clamp(age_days, min=torch.finfo(age_days.dtype).tiny)

    accum = new_snow_albedo * xmath.pow(accum_a, xmath.pow(safe_age, accum_b))
    thaw = new_snow_albedo * xmath.pow(thaw_a, xmath.pow(safe_age, thaw_b))

    aged = torch.where(
        (cold_content < 0.0) & (~melting), accum, thaw)
    have_pack = torch.where(swq > 0.0, aged, torch.zeros_like(aged))

    fresh = (new_snow > trace_snow) & (cold_content < 0.0)
    return torch.where(fresh, new_snow_albedo, have_pack)


#: param.SNOW_TRACESNOW, initialize_parameters.c:182, in metres.
#:
#: The C's value is 0.03 and its unit appears nowhere, which is how this
#: came to be wrong. solve_snow.c:138 forms `*snowfall` from `prec`, which
#: surface_fluxes.c:386 takes from force->prec in MILLIMETRES, and
#: snow_melt.c:103 converts with `SnowFall = snowfall / MM_PER_M`. So
#: SNOW_TRACESNOW is 0.03 mm, and in the metres this model carries
#: snowfall in, 3e-5 m.
#:
#: Read as 0.03 m it was a threshold of 30 mm of water equivalent in one
#: step. FindleyLake never sees that in an hour, so nothing ever reset the
#: albedo clock and the snow albedo decayed all winter -- 0.29 against
#: VIC's 0.81, averaged over the days with a pack.
TRACE_SNOW_M = 3.0e-5
MM_PER_M = 1000.0
MM_PER_IN = 25.4
IN_PER_M = MM_PER_M / MM_PER_IN
CONST_G = 9.80616
CONST_TKFRZ = 273.15
CONST_RHOFW = 1.000e3
CONST_RHOICE = 917.0        # vic_physical_constants.h:103, density of ice


def snow_density_bras(depth0, swq0, surf_temp, new_snow, sswq, air_temp, dt,
                      dens_denom=10.0, dens_exp=0.35, max_change=0.9,
                      eta0=3.6e6, c5=0.08, c6=0.021, **new_snow_kwargs):
    """Snow pack density after compaction, kg/m3.  snow_utility.c:41, DENS_BRAS.

    Two effects in sequence, following Bras (1990) p.257 and the SNTHRM89
    densification of Jordan (1991):

    1. compaction of the existing pack by the weight of new snowfall
    2. densification of the whole pack under its own overburden

    `depth0` and `swq0` are the pack state on entry (`snow->depth`,
    `snow->swq`); `sswq` is the snow water equivalent the caller passes
    separately, which the C uses in preference to `snow->swq`.

    This is the DENS_BRAS option, VIC's default (initialize_options.c:68).
    DENS_SNTHRM is `snow_density_snthrm`, a different and longer branch,
    and is not approximated here.

    Two behaviours of the C are reproduced deliberately rather than corrected:

    * When `new_snow <= 0` the C divides by `snow->depth`, the *entry* depth,
      not the running local one. With an empty pack that is a division by
      zero and yields an infinity. Returning a finite number instead would
      disagree with the model, so the division is written as the C has it and
      callers must exclude empty packs, exactly as they must in the C.
    * Both `delta_depth` limits clamp against `max_change * depth` using the
      depth current at that point in the sequence, which differs between the
      two clamps.
    """
    density_new = torch.where(
        new_snow > 0.0,
        new_snow_density(air_temp, **new_snow_kwargs),
        torch.zeros_like(air_temp))

    Tavg = surf_temp + CONST_TKFRZ

    # --- compaction by the weight of new snow -----------------------------
    # Guard only the denominators of the branches not taken, so the selected
    # branch is bit-identical to the C while the unselected one stays finite
    # and cannot poison the gradient with a NaN.
    safe_depth0 = torch.where(depth0 > 0.0, depth0, torch.ones_like(depth0))
    safe_sswq = torch.where(sswq > 0.0, sswq, torch.ones_like(sswq))

    delta_depth = (((new_snow / MM_PER_IN) * (safe_depth0 * IN_PER_M)) /
                   (safe_sswq * IN_PER_M) *
                   xmath.pow((safe_depth0 * IN_PER_M) / dens_denom, dens_exp)
                   ) / IN_PER_M
    delta_depth = torch.minimum(delta_depth, max_change * safe_depth0)

    depth_new = new_snow / torch.where(density_new > 0.0, density_new,
                                       torch.ones_like(density_new))

    # new snow onto an existing pack
    depth_a = safe_depth0 - delta_depth + depth_new
    swq_a = sswq + new_snow / MM_PER_M
    # new snow onto bare ground: the pack takes the new snow's density
    swq_b = sswq + new_snow / MM_PER_M
    depth_b = CONST_RHOFW * swq_b / torch.where(
        density_new > 0.0, density_new, torch.ones_like(density_new))

    has_pack = depth0 > 0.0
    snowing = new_snow > 0.0

    depth = torch.where(snowing, torch.where(has_pack, depth_a, depth_b),
                        depth0)
    swq = torch.where(snowing, torch.where(has_pack, swq_a, swq_b), sswq)
    # The no-snowfall branch divides by the entry depth; see the docstring.
    # `safe_depth0`, not `depth0`. torch.where evaluates BOTH branches, so
    # the no-snowfall expression is computed on every tile including the
    # snow-free ones, where depth0 is 0 and sswq is 0: 0/0 is NaN. The
    # forward result is masked out by the `snowing` selector and by the
    # `depth > 0` selector at the end of this function, which is why the
    # value looked clean and the water balance closed to 1e-13 -- but a NaN
    # in an unselected branch is still a NaN in the DERIVATIVE, and
    # MulBackward0 propagated it out through the surface energy balance
    # (energy.py:324, the refreeze term reads this density).
    #
    # On the Indian 6 km data that killed the gradient of five parameters
    # outright -- pcf, new_snow_albedo, alb_accum_a, alb_accum_b and
    # lai_scale -- leaving 1 of 16 usable. A forward run that conserves is
    # not evidence the model's gradients are usable.
    #
    # safe_depth0 is already defined for this reason at the top of the
    # function and used by delta_depth; this line was the one that missed
    # it. The selected branch is unchanged, so no value moves.
    density = torch.where(snowing,
                          CONST_RHOFW * swq / torch.where(
                              depth > 0.0, depth, torch.ones_like(depth)),
                          CONST_RHOFW * sswq / safe_depth0)

    # --- densification under overburden -----------------------------------
    overburden = 0.5 * CONST_G * CONST_RHOFW * swq
    viscosity = eta0 * torch.exp(-c5 * (Tavg - CONST_TKFRZ) + c6 * density)
    dd2 = overburden / viscosity * depth * dt
    dd2 = torch.minimum(dd2, max_change * depth)
    depth_aged = depth - dd2
    density_aged = CONST_RHOFW * swq / torch.where(
        depth_aged > 0.0, depth_aged, torch.ones_like(depth_aged))

    return torch.where(depth > 0.0, density_aged, density)


#: vic_def.h 5.1.0:53, the liquid fraction of the depth above which a pack
#: counts as wet snow in the SNTHRM densification.
MIN_SNOW_WETFRAC = 0.01


def snow_density_snthrm(density, depth, surf_temp, new_snow, sswq, air_temp,
                        dt, surf_water, pack_water,
                        dm_limit=100.0, dm_limit_factor=1.15,
                        c1=0.04, c2=2.778e-6, c3=1.0, c3_const=-0.046,
                        c4=1.0, c4wet=2.0, c5=0.08, c6=0.021, f=0.6,
                        eta0=3.6e6, wetfrac=MIN_SNOW_WETFRAC,
                        **new_snow_kwargs):
    """Snow pack density after compaction, kg/m3.  snow_utility.c 5.1.0:61-130,
    DENS_SNTHRM.

    The SNTHERM89 densification of Jordan (1991), adjusted for VIC's
    essentially single-layer pack: settling by destructive metamorphism
    plus compaction under the overburden of the new snow and a share `f`
    of the standing pack, applied as a rate over the step.

    `density` and `depth` are the pack state on entry (`snow->density`,
    `snow->depth`); unlike the Bras branch this one seeds from the carried
    density rather than recomputing it from depth and water equivalent, so
    a caller that does not carry the density forward gets a pack that
    forgets its age every step. `sswq` is the water equivalent the caller
    passes separately, in metres; `new_snow` is in millimetres, as the C
    is handed it. `surf_water` and `pack_water` are the pack's liquid
    stores, in metres, for the wet-snow check.

    Parameter defaults are VIC 5.1.0's, from
    drivers/shared_all/src/initialize_parameters.c:135-147. `new_snow_kwargs`
    go to `new_snow_density`, whose method defaults to 'snthrm' here: the
    C reads the same option for both ends of the branch.

    Two behaviours of the C are reproduced deliberately rather than corrected:

    * With no snowfall the density is the carried one grown by the
      metamorphism rate alone; the overburden term is exactly zero
      (:117-125), not merely small.
    * The wet check divides the liquid water by the entry depth only where
      a pack stands (:98-104). The divisor is guarded elsewhere and masked
      out, so the selected branch matches the C while no gradient passes
      through a division by zero.
    """
    new_snow_kwargs.setdefault('method', 'snthrm')
    density_new = torch.where(
        new_snow > 0.0,
        new_snow_density(air_temp, **new_snow_kwargs),
        torch.zeros_like(air_temp))

    tavg = surf_temp + CONST_TKFRZ

    snowing = new_snow > 0.0
    has_pack = depth > 0.0
    # The seed (:62-72): the carried density, except new snow onto bare
    # ground, which takes the new snow's. This branch never divides by the
    # entry depth, so an empty pack with no snowfall returns 0 -- the
    # carried density grown by nothing -- where the Bras branch divides by
    # zero and callers must exclude it.
    density = torch.where(snowing,
                          torch.where(has_pack, density, density_new),
                          density)

    dexpf = torch.exp(-c1 * (CONST_TKFRZ - tavg))

    # --- settling by destructive metamorphism ------------------------------
    scaled = dm_limit_factor * density_new
    dm = torch.where(snowing & (density_new > 0.0),
                     torch.where(scaled > dm_limit, scaled,
                                 _like(density_new, dm_limit)),
                     _like(density_new, dm_limit))
    c3v = torch.where(density <= dm, _like(density, c3),
                      torch.exp(c3_const * (density - dm)))
    safe_depth = torch.where(has_pack, depth, torch.ones_like(depth))
    wet = has_pack & ((surf_water + pack_water) / safe_depth > wetfrac)
    c4v = torch.where(wet, _like(density, c4wet), _like(density, c4))
    ddz1 = -c2 * c3v * c4v * dexpf

    # --- compaction under overburden ---------------------------------------
    swq = new_snow / MM_PER_M + f * sswq
    ps = 0.5 * CONST_G * CONST_RHOFW * swq
    # The exponent nested as the C nests it (:119-121): the negation of
    # (-C5 * (Tavg - TKFRZ) + C6 * density). That is C5 * (Tavg - TKFRZ)
    # - C6 * density, but written the C's way so the rounding matches.
    ddz2 = torch.where(
        snowing,
        -ps / eta0 * torch.exp(-(-c5 * (tavg - CONST_TKFRZ)
                                 + c6 * density)),
        torch.zeros_like(swq))

    cr = -ddz1 - ddz2
    return density * (1.0 + cr * dt)


# Further physical constants, vic_physical_constants.h
CONST_STEBOL = 5.67e-8      # Stefan-Boltzmann, W/m2/K4
CONST_CPMAIR = 1013.0       # specific heat of moist air, J/kg/K


def calc_outgoing_longwave(temp_k, emiss):
    """Outgoing longwave, W/m2.  physics.c:62.

    The C writes temp*temp*temp*temp rather than pow(temp, 4); the products
    are reproduced in that order so the rounding matches.
    """
    return emiss * CONST_STEBOL * temp_k * temp_k * temp_k * temp_k


def calc_sensible_heat(atmos_density, t1, t0, ra):
    """Sensible heat flux, W/m2.  physics.c:93."""
    return CONST_CPMAIR * atmos_density * (t1 - t0) / ra


def advected_sensible_heat(snow_cover_fract, air_dens, t_air, t_grnd, ra):
    """Sensible heat advected from bare patches onto the pack, W/m2.
    advected_sensible_heat.c:38-70.  SPATIAL_SNOW only.

    When only part of a tile carries snow, the bare part of it runs warm and
    hands some of its sensible heat sideways to the pack. VIC computes the
    bare patch's own sensible heat flux from the *ground* temperature, takes
    the share Fs of it that reaches the snow, and spreads what the bare
    fraction produced over the covered fraction that receives it.

    Fs is Liston's (1995) step-wise fit, four pieces in cover fraction. The
    C's chain of `if` is a `where` here so that the function is defined, and
    differentiable, on the whole range rather than only on the piece the
    state happens to fall in.

    `snow_cover_fract` must be strictly positive; the caller is the one that
    tests it (SnowPackEnergyBalance.c:201-211 returns zero at zero cover),
    because at zero cover the division below is the C's own 0/0.
    """
    q_bare = calc_sensible_heat(air_dens, t_air, t_grnd, ra)
    f = snow_cover_fract
    fs = torch.where(
        f > 0.6, torch.ones_like(f),
        torch.where(f > 0.5, xmath.pow(_like(f, 10.0), 3.0 * f - 1.8),
                    torch.where(f > 0.2,
                                xmath.pow(_like(f, 10.0), 5.6667 * f - 3.1333),
                                _like(f, 0.01))))
    safe = torch.where(f > 0.0, f, torch.ones_like(f))
    return torch.where(f > 0.0, q_bare * (1.0 - f) / safe * fs,
                       torch.zeros_like(f))


def stability_invariants(z, d, t_air, wind, z0):
    """stability_correction's terms that do not depend on the surface
    temperature, for a caller that evaluates it at many."""
    return dict(safe_wind=torch.where(wind > 0.0, wind, torch.ones_like(wind)),
                t_air_k=t_air + CONST_TKFRZ, z_d=z - d,
                log_z=torch.log((z - d) / z0) + 5)


def stability_correction(z, d, t_surf, t_air, wind, z0, ri_cr=0.2, inv=None):
    """Richardson-number stability correction.  StabilityCorrection.c:34.

    Returns the factor the aerodynamic resistance is *divided* by in
    SnowPackEnergyBalance, so a small value means very stable air and a large
    resistance.

    The C returns exactly 1.0 when TSurf == Tair, short-circuiting before the
    Richardson number is formed. That branch is kept, because the expression
    it guards divides by (Tair - TSurf) implicitly through Ri and the neutral
    case must not depend on how that limit is approached.

    All branches are evaluated and selected with `where`, so unselected sides
    must stay finite: the wind term is guarded below, since Wind = 0 gives an
    infinite Ri in the C and is handled by the caller, which substitutes
    HUGE_RESIST instead of calling this at all.
    """
    if inv is None:
        inv = stability_invariants(z, d, t_air, wind, z0)
    safe_wind = inv['safe_wind']

    t_mean_k = (inv['t_air_k'] + (t_surf + CONST_TKFRZ)) / 2.0
    ri = (CONST_G * (t_air - t_surf) * inv['z_d'] /
          (t_mean_k * safe_wind * safe_wind))
    ri_limit = inv['t_air_k'] / (t_mean_k * inv['log_z'])

    ri = torch.minimum(ri, ri_limit)

    stable = (1 - ri / ri_cr) * (1 - ri / ri_cr)
    # The unstable branch is selected only where ri <= 0, and the C clamps ri
    # at -0.5 there. Clamping at both ends keeps 1 - 16*ri in [1, 9] for every
    # element, including those where the other branch is selected. Without the
    # upper clamp, ri > 1/16 makes this sqrt of a negative number: `where`
    # discards the value, but a NaN in an unselected branch still propagates
    # through the backward pass and NaNs the gradient. That happened in 30 of
    # 240 energy-balance cases before the clamp was added.
    unstable = xmath.sqrt(1 - 16 * torch.clamp(ri, min=-0.5, max=0.0))
    correction = torch.where(ri > 0.0, stable, unstable)

    return torch.where(t_surf != t_air, correction, torch.ones_like(correction))


PA_PER_KPA = 1000.0
JOULES_PER_CAL = 4.1868
CONST_MWWV = 18.016
CONST_MWAIR = 28.97
CONST_EPS = CONST_MWWV / CONST_MWAIR


def svp(temp, svp_a=0.61078, svp_b=17.269, svp_c=237.3):
    """Saturated vapour pressure, Pa.  svp.c.

    The sub-zero correction is a multiplicative polynomial the C applies only
    below 0 C; both sides are evaluated here and selected, which is the same
    value.
    """
    base = svp_a * xmath.exp((svp_b * temp) / (svp_c + temp))
    cold = base * (1.0 + 0.00972 * temp + 0.000042 * temp * temp)
    return torch.where(temp < 0.0, cold, base) * PA_PER_KPA


def calc_latent_heat_of_sublimation(temp):
    """Latent heat of sublimation, J/kg.  physics.c:35."""
    return (677.0 - 0.07 * temp) * JOULES_PER_CAL * GRAMS_PER_KG


def latent_heat_from_snow(air_dens, eact_air, lv, press, ra, t_mean, vpd,
                          blowing_mass_flux, dens_eps=None, vpd_zero=None):
    """Latent heat and vapour mass fluxes from a snow surface.
    latent_heat_from_snow.c.

    Returns (latent_heat, latent_heat_sub, vapor_mass_flux,
             surface_mass_flux).

    The C writes its results through pointers and takes BlowingMassFlux as an
    in-out; here it is an input and the rest are returned, because a tensor
    written through in place breaks autograd's version counter -- the same
    reason torchvic.core.column keeps layers as a list rather than writing rows
    of a tensor.

    Above freezing the pack is melting and the latent heat uses vaporisation,
    with the sublimation term zero; below, the reverse. The C's branch is
    exclusive, so exactly one of the two returned heats is non-zero, and that
    is preserved.
    """
    es_snow = svp(t_mean)

    # CONST_EPS is wrapped as a tensor rather than left a Python float.
    # torch does not round `python_float / tensor` the way IEEE division
    # requires -- it is 1 ulp off for some operands, apparently computing
    # scalar * (1/tensor) -- while `tensor / tensor` and `tensor / float` are
    # both exact. With the float form this expression disagreed with the C in
    # 58 of 192 test cases at 3.8e-16 relative. Never write
    # `python_float / tensor` anywhere in torchvic.
    if dens_eps is None:
        eps = device_constant(CONST_EPS, press.dtype, press.device)
        dens_eps = air_dens * (eps / press)
    surface_mass_flux = dens_eps * (eact_air - es_snow) / ra
    # The C zeroes the surface flux when the air is saturated and the flux
    # would be into the pack; condensation onto a saturated surface is not
    # represented.
    surface_mass_flux = torch.where(
        (vpd == 0.0 if vpd_zero is None else vpd_zero)
        & (surface_mass_flux < 0.0),
        torch.zeros_like(surface_mass_flux), surface_mass_flux)

    vapor_mass_flux = surface_mass_flux + blowing_mass_flux

    melting = t_mean >= 0.0
    ls = calc_latent_heat_of_sublimation(t_mean)
    zero = torch.zeros_like(vapor_mass_flux)
    latent_heat = torch.where(melting, lv * vapor_mass_flux, zero)
    latent_heat_sub = torch.where(melting, zero, ls * vapor_mass_flux)

    return latent_heat, latent_heat_sub, vapor_mass_flux, surface_mass_flux


CONST_CPFW = 4.188e3
CONST_CPICE = 2.11727e3
CONST_RHOFW_ = CONST_RHOFW
CONST_VCPICE_WQ = CONST_CPICE * CONST_RHOFW
CONST_LATICE = 3.337e5
HUGE_RESIST = 1e20
EMISS_SNOW = 0.97
SNOW_CONDUCT = 2.9302e-6


#: As energy.COMPILE, for the pack's residual: compiled only where no
#: derivative passes through it (see energy._residual_fast). Off by default.
COMPILE = False
_compiled_pack = None


#: The pack's surface-temperature search: 'brent' is snow_melt.c 5.1.0
#: :266-305 -- root_brent from [surf_temp - SNOW_DT, surf_temp + SNOW_DT],
#: the bracket widened as root_brent.c 5.1.0:147-219 does, its iteration and
#: stopping rule (implicit.root_brent), OldTSurf where it fails (TFALLBACK)
#: -- so the temperature is VIC's to the rounding of the residual;
#: 'illinois' solves [-80, 0] to ~1e-13 W m-2, up to ROOT_BRENT_T from VIC's.
SNOW_SOLVER = 'brent'
#: snow_melt.c's bracket half-width, K (initialize_parameters.c 5.1.0:183).
SNOW_DT = 5.0


@fusion.opaque
def _solve_pack_brent(t_mixed, old_t_surf, eb):
    """snow_melt.c 5.1.0:266-305 for every element, with gradients.

    `t_mixed` is snow->surf_temp after the snowfall was mixed in (:125-130),
    the centre of the C's bracket. Elements whose bracket holds no sign
    change after ROOT_BRENT_MAXTRIES widenings take OldTSurf, as VIC's
    TFALLBACK default does (:289-293)."""
    from . import implicit
    from .energy import ROOT_BRENT_TSTEP, ROOT_BRENT_MAXTRIES

    def f(ts, st):
        return _pack_residual_fast(ts, st)

    with torch.no_grad():
        lo = t_mixed - SNOW_DT
        hi = t_mixed + SNOW_DT
        flo, fhi = f(lo, eb), f(hi, eb)
        if graphsafe.ACTIVE:
            lo, hi, flo, fhi = graphsafe.widen(f, lo, hi, flo, fhi, (eb,),
                                               ROOT_BRENT_TSTEP,
                                               ROOT_BRENT_MAXTRIES)
        else:
            for _ in range(ROOT_BRENT_MAXTRIES):
                bad = (flo * fhi) >= 0.0
                if not bool(bad.any()):
                    break
                lo = torch.where(bad, lo - ROOT_BRENT_TSTEP, lo)
                hi = torch.where(bad, hi + ROOT_BRENT_TSTEP, hi)
                flo = torch.where(bad, f(lo, eb), flo)
                fhi = torch.where(bad, f(hi, eb), fhi)
        # a failed element is solved on a dummy bracket and discarded
        failed, lo, hi, flo, fhi = _pack_bracket(lo, hi, flo, fhi)
        r0 = implicit.root_brent(f, lo, hi, flo, fhi, args=(eb,),
                                 t_tol=implicit.ROOT_BRENT_T)
    if torch.is_grad_enabled():
        rg = implicit.root_with_grad(f, r0, args=(eb,))
        root = r0 + (rg - rg.detach())
    else:
        root = r0
    SIDE['last'] = failed
    return _where3(failed, old_t_surf, root)


def _pack_bracket_(lo, hi, flo, fhi):
    """No sign change after the widenings (or a non-finite residual):
    failed, and the dummy bracket (-1, 0), residuals (-1, 1) it is solved
    on instead."""
    failed = (((flo * fhi) >= 0.0) | ~torch.isfinite(flo)
              | ~torch.isfinite(fhi))
    lo = torch.where(failed, torch.full_like(lo, -1.0), lo)
    hi = torch.where(failed, torch.zeros_like(hi), hi)
    flo = torch.where(failed, -torch.ones_like(flo), flo)
    fhi = torch.where(failed, torch.ones_like(fhi), fhi)
    return failed, lo, hi, flo, fhi


def _where3_(c, a, b):
    return torch.where(c, a, b)


# fused regions on the CUDA graph path (core.fusion), around the solver
_pack_bracket = fusion.region(_pack_bracket_)
_where3 = fusion.region(_where3_)


#: SIDE['last']: the elements the last `_solve_pack_brent` found no root
#: for; SIDE['pack']: the subset of them that VIC's snow_melt.c solves at all
#: (melting, thin and unstable packs never reach root_brent): the caller
#: reads it after `snow_step` for TFALLBACK (core/tfallback.py).
SIDE = {'last': None, 'pack': None}


def _pack_residual_fast(t_surf, st):
    """snow_pack_energy_balance(t_surf, st), compiled under no_grad and
    without a forward-mode tangent -- the solver's iterations -- and eager
    wherever a derivative is taken."""
    global _compiled_pack
    import torch.autograd.forward_ad as fwad
    if (COMPILE and not torch.is_grad_enabled()
            and fwad.unpack_dual(t_surf).tangent is None):
        if _compiled_pack is None:
            from . import fusion
            fusion.ensure_setup()
            _compiled_pack = torch.compile(snow_pack_energy_balance,
                                           dynamic=False)
        return _compiled_pack(t_surf, st)
    return _pack_residual_fused(t_surf, st)


def snow_pack_energy_balance(t_surf, st, return_refreeze=False,
                             return_terms=False):
    """Snowpack surface energy balance residual, W/m2.
    SnowPackEnergyBalance.c.

    `t_surf` is the surface temperature in C being tested; `st` is a dict of
    everything else the balance needs, held fixed while the root is sought.
    Returns the residual that VIC's root_brent drives to zero, or, with
    `return_refreeze`, the triple (residual, refreeze energy, vapour flux).

    The vapour flux is in metres of water per time step and carries the C's
    sign: negative is mass leaving the pack. SnowPackEnergyBalance.c:217-235
    converts it to kg/m2/s on the way into latent_heat_from_snow() and back
    again on the way out, and it is the only route by which sublimation
    reaches the mass balance -- snow_melt.c:217-228 and 361-374 add it to the
    surface liquid water or to the surface ice. Returning only the residual,
    as this function used to, silently discards VIC's OUT_SUB_SNOW.

    The advected sensible heat term is SPATIAL_SNOW's: a caller that passes
    a `snow_cover_fract` in `st` gets it (advected_sensible_heat.c), and a
    caller that leaves it out gets zero, which is what the C sets it to with
    SPATIAL_SNOW off -- VIC's default, initialize_options.c:70.

    This is the function torchvic.core.implicit solves and differentiates. It is
    written so that every operation is differentiable in `t_surf` and in the
    parameters, which is why the branches are `where` rather than `if`: the
    root solver needs a residual defined on the whole bracket, not only on the
    side the state happens to fall.
    """
    t_mean = t_surf
    density = CONST_RHOFW
    # What does not depend on the surface temperature, formed once per `st`
    # (the solver evaluates this a dozen times) with the operations written
    # out below, so the same numbers.
    inv = st.get('_invariant')
    if inv is None:
        _z = st['z']
        _d = torch.zeros_like(_z)
        _thres = SNOW_DEPTH_THRES if core_version.at_least('5.1.0') else 0.0
        inv = dict(
            stab=stability_invariants(_z, _d, st['t_air'], st['wind'],
                                      st['z0']),
            wind_pos=st['wind'] > 0.0,
            dens_eps=(st['air_dens']
                      * (device_constant(CONST_EPS, st['press'].dtype,
                                         st['press'].device)
                         / st['press'])),
            vpd_zero=st['vpd'] == 0.0,
            adv=(CONST_CPFW * CONST_RHOFW * st['t_air'] * st['rain'])
            / st['dt'],
            deep=st['snow_depth'] > _thres,
            # SnowPackEnergyBalance.c 5.1.0:237, SNOW_CONDUCT *
            # pow(SnowDensity, 2.): the square first
            cond=SNOW_CONDUCT * (st['snow_density'] * st['snow_density']),
            depth=torch.where(st['snow_depth'] > 0.0, st['snow_depth'],
                              torch.ones_like(st['snow_depth'])),
            cc=CONST_VCPICE_WQ * st['swe_surface'],
            refreeze=(st['surface_liquid_water'] * CONST_LATICE * density
                      / st['dt']))
        if '_invariant' in st and not torch.is_grad_enabled():
            st['_invariant'] = inv       # the caller's dict allows it

    # Stability-corrected aerodynamic resistance. The C substitutes
    # HUGE_RESIST outright when there is no wind rather than dividing.
    correction = stability_correction(None, None, t_mean, st['t_air'],
                                      None, None, inv=inv['stab'])
    ra_used = torch.where(inv['wind_pos'], st['ra'] / correction,
                          _like(correction, HUGE_RESIST))

    net_long = st['long_snow_in'] - calc_outgoing_longwave(
        t_mean + CONST_TKFRZ, EMISS_SNOW)
    net_rad = st['net_short'] + net_long

    sensible = calc_sensible_heat(st['air_dens'], st['t_air'], t_mean,
                                  ra_used)

    # SnowPackEnergyBalance.c:200-214: with SPATIAL_SNOW the bare part of a
    # partly covered tile hands sensible heat to the pack, computed from the
    # *ground* temperature and the same stability-corrected resistance the
    # pack itself uses. With the option off, and at zero cover, it is zero.
    cover_f = st.get('snow_cover_fract')
    if cover_f is None:
        adv_sensible = torch.zeros_like(t_mean)
    else:
        adv_sensible = advected_sensible_heat(
            cover_f, st['air_dens'], st['t_air'], st['t_grnd'], ra_used)

    latent, latent_sub, vapor_mass_flux, _ = latent_heat_from_snow(
        st['air_dens'], st['eact_air'], st['lv'], st['press'], ra_used,
        t_mean, st['vpd'], st['blowing_mass_flux'],
        dens_eps=inv['dens_eps'], vpd_zero=inv['vpd_zero'])

    # kg/m2/s back to metres of water per time step.
    # SnowPackEnergyBalance.c:233.
    vapor_flux = vapor_mass_flux * st['dt'] / density

    # Rain falling on a melting pack advects heat; the C tests TMean == 0
    # exactly, so the comparison is reproduced exactly rather than widened.
    advected = torch.where(t_mean == 0.0, inv['adv'],
                           torch.zeros_like(t_mean))

    delta_cc = (inv['cc'] *
                (t_surf - st['old_t_surf']) / st['dt'])

    # SnowPackEnergyBalance.c:252 tests `SnowDepth > 0.` and nothing wider.
    # This used to test against 1e-8 m, which is a guard the C does not have:
    # between 0 and 1e-8 m of snow VIC computes a conduction term that this
    # zeroed. The division is guarded separately just below, so the guard
    # buys nothing that costs agreement.
    # VIC 5.1.0 does have that guard: `SnowDepth > param.SNOW_DEPTH_THRES`,
    # 1e-8 m (SnowPackEnergyBalance.c 5.1.0:236, initialize_parameters.c
    # 5.1.0:134).
    ground_flux = torch.where(
        inv['deep'],
        inv['cond'] *
        (st['t_grnd'] - t_mean) /
        inv['depth'] / st['dt'],
        torch.zeros_like(t_mean))
    delta_cc = delta_cc - ground_flux

    rest = (net_rad + sensible + latent + latent_sub + advected +
            ground_flux - delta_cc + adv_sensible)

    refreeze = inv['refreeze']
    # The C writes RefreezeEnergy through a pointer, and snow_melt() needs it:
    # at 0 C the residual is forced to zero and the surplus energy is handed
    # back through that parameter instead. Discarding it makes the pack
    # unable to melt, because the caller then sees only a zero residual with
    # no indication of how much energy was available.
    melting = (t_surf == 0.0) & (rest > -refreeze)
    out_refreeze = torch.where(melting, -rest, refreeze)
    out_rest = torch.where(melting, torch.zeros_like(rest), rest + refreeze)
    if return_terms:
        # The terms SnowPackEnergyBalance.c writes through its pointers and
        # snow_melt.c:500-508 saves into the energy struct: advection,
        # deltaCC (after the ground flux was subtracted from it, as the C
        # does), latent, latent_sub, sensible. grnd_flux goes to a local of
        # solve_snow.c (tmp_grnd_flux) and is never reported; it is returned
        # here only so the closure of the terms can be tested.
        terms = dict(net_rad=net_rad, net_short=st['net_short'],
                     net_long=net_rad - st['net_short'],
                     sensible=sensible, latent=latent,
                     latent_sub=latent_sub, advection=advected,
                     delta_cc=delta_cc, grnd_flux=ground_flux,
                     advected_sensible=adv_sensible)
        return out_rest, out_refreeze, vapor_flux, terms
    if return_refreeze:
        return out_rest, out_refreeze, vapor_flux
    return out_rest


SNOW_LIQUID_WATER_CAPACITY = 0.035
SNOW_MAX_SURFACE_SWE = 0.125
#: param.SNOW_MIN_SWQ_EB_THRES, initialize_parameters.c:172. A frozen pack
#: whose surface layer holds no more than this is not solved by the pack
#: (snow_melt.c:266, :322-324 set surf_temp = 999) but by the soil surface
#: with the pack included (surface_fluxes.c:585-591).
SNOW_MIN_SWQ_EB_THRES = 0.001


#: the residual as one fused region on the CUDA graph path (core.fusion)
_pack_residual_fused = fusion.region(snow_pack_energy_balance)


def pack_refreeze(surf_water, pack_water, surface_swq, pack_swq, pack_cc,
                  pack_temp, liquid_capacity=SNOW_LIQUID_WATER_CAPACITY):
    """Route liquid water between snow layers and refreeze it.
    snow_melt.c, the block following the surface energy balance.

    Returns (surf_water, pack_water, pack_swq, ice_gain, pack_temp, melt),
    where `ice_gain` is what must be added to the pack's ice content and
    `melt` is the outflow leaving the pack.

    The sequence is the C's: surface liquid above holding capacity drains to
    the pack, the pack refreezes what its cold content allows, and whatever
    the pack cannot hold becomes melt outflow.

    Cold content is negative or zero. Where `PackCC < -PackRefreezeEnergy`
    the pack has cold to spare and freezes all its water; otherwise it freezes
    exactly enough to exhaust the cold content and warms to 0 C.
    """
    # Surface layer: drain above holding capacity.
    max_liquid_surf = liquid_capacity * surface_swq
    drains = surf_water > max_liquid_surf
    surf_out = torch.where(drains, surf_water - max_liquid_surf,
                           torch.zeros_like(surf_water))
    surf_water = torch.where(drains, max_liquid_surf, surf_water)

    # Pack layer: take the surface outflow, then refreeze.
    pack_water = pack_water + surf_out
    refreeze_energy = pack_water * CONST_LATICE * CONST_RHOFW

    # Branch A: cold content not fully depleted -- freeze everything.
    pack_swq_a = pack_swq + pack_water
    ice_gain_a = pack_water
    has_pack = pack_swq_a > 0.0
    safe_swq_a = torch.where(has_pack, pack_swq_a, torch.ones_like(pack_swq_a))
    cc_a = pack_swq_a * CONST_VCPICE_WQ * pack_temp + refreeze_energy
    temp_a = torch.where(has_pack,
                         torch.clamp(cc_a / (CONST_VCPICE_WQ * safe_swq_a),
                                     max=0.0),
                         torch.zeros_like(pack_temp))
    water_a = torch.zeros_like(pack_water)

    # Branch B: cold content exhausted -- freeze exactly enough, pack ripens.
    delta_swq = -pack_cc / (CONST_LATICE * CONST_RHOFW)
    water_b = pack_water - delta_swq
    pack_swq_b = pack_swq + delta_swq
    ice_gain_b = delta_swq
    temp_b = torch.zeros_like(pack_temp)

    spare_cold = pack_cc < -refreeze_energy
    pack_water = torch.where(spare_cold, water_a, water_b)
    pack_swq = torch.where(spare_cold, pack_swq_a, pack_swq_b)
    ice_gain = torch.where(spare_cold, ice_gain_a, ice_gain_b)
    pack_temp = torch.where(spare_cold, temp_a, temp_b)

    # Pack layer: drain above holding capacity. This is the melt that leaves.
    max_liquid_pack = liquid_capacity * pack_swq
    drains = pack_water > max_liquid_pack
    melt = torch.where(drains, pack_water - max_liquid_pack,
                       torch.zeros_like(pack_water))
    pack_water = torch.where(drains, max_liquid_pack, pack_water)

    return surf_water, pack_water, pack_swq, ice_gain, pack_temp, melt


def rebalance_layers(ice, surface_swq, pack_swq, surf_temp, pack_temp,
                     pack_water, surf_water, max_surface_swe=SNOW_MAX_SURFACE_SWE):
    """Move ice between the surface and pack layers to restore the surface
    layer's maximum water equivalent.  snow_melt.c:455.

    Returns (surface_swq, pack_swq, surf_temp, pack_temp, swq).

    Cold content moves with the ice, so the layer temperatures are
    recomputed from the redistributed cold contents rather than carried.

    Two sequencing details of the C are reproduced rather than tidied:

    * In the branch where the surface layer is too small, `SurfaceCC` is
      incremented using `PackCC` *after* `PackCC` has already been decremented
      on the line above, so the transferred amount is `PackCC*(1-f)*f` rather
      than `PackCC*f`. Writing it the tidier way changes the answer.
    * Where the total ice is at or below the surface maximum there is no pack
      layer at all, and the C zeroes its water equivalent and temperature
      rather than leaving them.
    """
    surface_cc = CONST_VCPICE_WQ * surf_temp * surface_swq
    pack_cc = CONST_VCPICE_WQ * pack_temp * pack_swq

    # Guard the denominators of unselected branches; selected values are
    # unaffected. See the note in stability_correction on why an unselected
    # NaN still matters.
    safe_surface_swq = torch.where(surface_swq > 0.0, surface_swq,
                                   torch.ones_like(surface_swq))
    safe_pack_swq = torch.where(pack_swq > 0.0, pack_swq,
                                torch.ones_like(pack_swq))

    # Surface layer too large: push the excess down into the pack.
    # The C multiplies before dividing -- SurfaceCC * (Swq - MAX) / Swq -- so
    # the association is kept rather than factoring out the fraction, which
    # rounds differently and cost 1 ulp when it was written the tidy way.
    excess = surface_swq - max_surface_swe
    pack_cc_a = pack_cc + surface_cc * excess / safe_surface_swq
    surface_cc_a = surface_cc - surface_cc * excess / safe_surface_swq
    pack_swq_a = pack_swq + excess
    surface_swq_a = torch.full_like(surface_swq, 0.0) + max_surface_swe

    # Surface layer too small: pull ice up from the pack.
    deficit = max_surface_swe - surface_swq
    pack_cc_b = pack_cc - pack_cc * deficit / safe_pack_swq
    # uses the updated PackCC, as the C does
    surface_cc_b = surface_cc + pack_cc_b * deficit / safe_pack_swq
    pack_swq_b = pack_swq - deficit
    surface_swq_b = torch.full_like(surface_swq, 0.0) + max_surface_swe

    too_big = surface_swq > max_surface_swe
    too_small = surface_swq < max_surface_swe

    new_pack_cc = torch.where(too_big, pack_cc_a,
                              torch.where(too_small, pack_cc_b, pack_cc))
    new_surface_cc = torch.where(too_big, surface_cc_a,
                                 torch.where(too_small, surface_cc_b,
                                             surface_cc))
    new_pack_swq = torch.where(too_big, pack_swq_a,
                               torch.where(too_small, pack_swq_b, pack_swq))
    new_surface_swq = torch.where(too_big, surface_swq_a,
                                  torch.where(too_small, surface_swq_b,
                                              surface_swq))

    s_pack = torch.where(new_pack_swq != 0.0, new_pack_swq,
                         torch.ones_like(new_pack_swq))
    s_surf = torch.where(new_surface_swq != 0.0, new_surface_swq,
                         torch.ones_like(new_surface_swq))
    layered_pack_temp = new_pack_cc / (CONST_VCPICE_WQ * s_pack)
    layered_surf_temp = new_surface_cc / (CONST_VCPICE_WQ * s_surf)

    layered = ice > max_surface_swe
    out_pack_swq = torch.where(layered, new_pack_swq,
                               torch.zeros_like(new_pack_swq))
    out_surface_swq = torch.where(layered, new_surface_swq, surface_swq)
    out_pack_temp = torch.where(layered, layered_pack_temp,
                                torch.zeros_like(pack_temp))
    out_surf_temp = torch.where(layered, layered_surf_temp, surf_temp)

    swq = ice + pack_water + surf_water
    empty = swq == 0.0
    zero = torch.zeros_like(swq)
    out_surf_temp = torch.where(empty, zero, out_surf_temp)
    out_pack_temp = torch.where(empty, zero, out_pack_temp)

    return out_surface_swq, out_pack_swq, out_surf_temp, out_pack_temp, swq


def surface_melt(refreeze_energy, surf_water, surface_swq, pack_swq,
                 pack_water, ice, dt, vapor_flux=None):
    """Melt or refreeze at the snow surface when the pack is at 0 C.
    snow_melt.c:189, the Qnet == 0 branch.

    Returns (snow_melt, surf_water, surface_swq, pack_swq, pack_water, ice,
             melt_energy, surf_temp, pack_temp, vapor_flux).

    Entered when the surface energy balance is satisfied at 0 C, which means
    the surface temperature cannot fall and the residual energy goes into
    phase change instead. Positive `refreeze_energy` freezes surface liquid;
    negative melts ice.

    Melt is drawn from the pack layer first and only then from the surface
    layer, which is the C's order and matters because the two layers carry
    different temperatures.

    `vapor_flux` is the surface vapour exchange in metres of water over the
    step, with the C's sign -- negative is mass leaving the pack. It is
    applied to the surface liquid water between the melt computation and the
    routing, which is where snow_melt.c:217-228 puts it and not a detail:
    the melt that leaves the pack in the same step is drawn from the store
    the flux has already adjusted. Passing None means no exchange, which is
    what the earlier version of this function did unconditionally, and cost
    up to 5.6% of the step's melt in the reference cases.

    The returned vapour flux is the one actually applied. VIC caps it at the
    available liquid (snow_melt.c:218-227) and rescales blowing_flux and
    surface_flux alongside; only the total is rescaled here, because
    BLOWING is not implemented and the split is then trivially all-surface.
    """
    latent_vol = CONST_LATICE * CONST_RHOFW

    # --- refreezing (RefreezeEnergy >= 0) ---------------------------------
    refrozen = refreeze_energy / latent_vol * dt
    capped = refrozen > surf_water
    refrozen = torch.where(capped, surf_water, refrozen)
    energy_r = torch.where(capped, surf_water * latent_vol / dt,
                           refreeze_energy)
    surf_water_r = torch.clamp(surf_water - refrozen, min=0.0)
    surface_swq_r = surface_swq + refrozen
    ice_r = ice + refrozen
    melt_r = torch.zeros_like(surf_water)

    # --- melting (RefreezeEnergy < 0) -------------------------------------
    melt_m = torch.abs(refreeze_energy) / latent_vol * dt

    freezing = refreeze_energy >= 0.0
    snow_melt = torch.where(freezing, melt_r, melt_m)
    melt_energy = torch.where(freezing, energy_r, refreeze_energy)
    surf_water = torch.where(freezing, surf_water_r, surf_water)
    surface_swq = torch.where(freezing, surface_swq_r, surface_swq)
    ice = torch.where(freezing, ice_r, ice)

    # --- vapour exchange with the surface liquid --------------------------
    # snow_melt.c:217-228. The pack cannot sublimate liquid it does not have,
    # so the flux is capped at the store and the capped value is what gets
    # reported as OUT_SUB_SNOW.
    if vapor_flux is None:
        vapor_flux = torch.zeros_like(surf_water)
    exhausted = surf_water < -vapor_flux
    applied_vapor = torch.where(exhausted, -surf_water, vapor_flux)
    surf_water = torch.where(exhausted, torch.zeros_like(surf_water),
                             surf_water + vapor_flux)

    # --- route the melt --------------------------------------------------
    # Melt smaller than the pack layer: take it all from the pack.
    sw_a = surf_water + snow_melt
    pk_a = pack_swq - snow_melt
    ice_a = ice - snow_melt
    pw_a = pack_water

    # Melt exceeds the pack but not the total ice: empty the pack, take the
    # rest from the surface layer.
    # snow_melt.c 5.1.0:219, surf_water += SnowMelt + pack_water
    sw_b = surf_water + (snow_melt + pack_water)
    pw_b = torch.zeros_like(pack_water)
    pk_b = torch.zeros_like(pack_swq)
    ice_b = ice - snow_melt
    ss_b = ice_b

    # Melt exceeds the whole pack: everything melts.
    #
    # The C's `melt_energy -= RefreezeEnergy` here uses the refreeze energy
    # *after* the capping above reduced it, not the value passed in. Since
    # melt_energy accumulated exactly that updated value and started at zero,
    # melt_energy is the updated energy, and subtracting it leaves zero before
    # the sign term is added. Using the original argument instead gives -50
    # where the C gives 0 whenever the cap bit.
    re_updated = melt_energy
    melt_c = ice
    sw_c = surf_water + ice
    nonzero = re_updated != 0.0
    safe_re = torch.where(nonzero, re_updated, torch.ones_like(re_updated))
    energy_c = torch.where(
        nonzero,
        re_updated - re_updated +
        re_updated / torch.abs(safe_re) * melt_c * latent_vol / dt,
        re_updated)

    within_ice = snow_melt < ice
    within_pack = snow_melt <= pack_swq

    out_melt = torch.where(within_ice, snow_melt, melt_c)
    out_sw = torch.where(within_ice, torch.where(within_pack, sw_a, sw_b),
                         sw_c)
    # The C leaves pack_water untouched when everything melts; only the
    # intermediate branch empties it into the surface layer.
    out_pw = torch.where(within_ice, torch.where(within_pack, pw_a, pw_b),
                         pack_water)
    out_pk = torch.where(within_ice, torch.where(within_pack, pk_a, pk_b),
                         torch.zeros_like(pack_swq))
    out_ice = torch.where(within_ice, torch.where(within_pack, ice_a, ice_b),
                          torch.zeros_like(ice))
    out_ss = torch.where(within_ice,
                         torch.where(within_pack, surface_swq, ss_b),
                         torch.zeros_like(surface_swq))
    out_energy = torch.where(within_ice, melt_energy, energy_c)

    zero = torch.zeros_like(surf_water)
    return (out_melt, out_sw, out_ss, out_pk, out_pw, out_ice, out_energy,
            zero, torch.where(within_ice, zero, zero), applied_vapor)


DBL_EPSILON = 2.220446049250313e-16


def accumulate_snowfall(ice, surf_temp, pack_temp, snowfall, air_temp,
                        max_surface_swe=SNOW_MAX_SURFACE_SWE, cold=False):
    """Split the pack into layers and add new snowfall with its cold content.
    snow_melt.c:114.

    Returns (surface_swq, pack_swq, surf_temp, pack_temp).

    The surface layer is capped at `max_surface_swe` and the remainder is the
    pack. New snow lands on the surface layer; if it would overfill it, the
    excess and a proportional share of cold content move down to the pack.

    New snow arriving warmer than freezing carries no cold content, which is
    the C's branch on air temperature rather than on the snow's own state.

    The guard on the overfill branch is the C's: it requires the remaining
    surface capacity to exceed DBL_EPSILON, not merely to be positive, so a
    surface layer already at capacity takes the simple path.
    """
    surface_swq = torch.minimum(ice, _like(ice, max_surface_swe))
    pack_swq = ice - surface_swq
    surface_cc = CONST_VCPICE_WQ * surface_swq * surf_temp
    pack_cc = CONST_VCPICE_WQ * pack_swq * pack_temp

    snowfall_cc = torch.where(air_temp > 0.0, torch.zeros_like(air_temp),
                              CONST_VCPICE_WQ * snowfall * air_temp)

    capacity = max_surface_swe - surface_swq
    overfills = (snowfall > capacity) & (capacity > DBL_EPSILON)

    safe_snowfall = torch.where(snowfall > 0.0, snowfall,
                                torch.ones_like(snowfall))
    safe_surface_swq = torch.where(surface_swq > 0.0, surface_swq,
                                   torch.ones_like(surface_swq))

    delta_pack_swq = surface_swq + snowfall - max_surface_swe
    # Which share of cold content follows the ice down depends on whether the
    # transfer exceeds the whole surface layer.
    delta_pack_cc = torch.where(
        delta_pack_swq > surface_swq,
        surface_cc + (snowfall - max_surface_swe) / safe_snowfall *
        snowfall_cc,
        delta_pack_swq / safe_surface_swq * surface_cc)

    surface_swq_o = _like(surface_swq, max_surface_swe)
    # :125, SurfaceCC += SnowFallCC - DeltaPackCC: the difference first
    surface_cc_o = surface_cc + (snowfall_cc - delta_pack_cc)
    pack_swq_o = pack_swq + delta_pack_swq
    pack_cc_o = pack_cc + delta_pack_cc

    surface_swq = torch.where(overfills, surface_swq_o,
                              surface_swq + snowfall)
    surface_cc = torch.where(overfills, surface_cc_o,
                             surface_cc + snowfall_cc)
    pack_swq = torch.where(overfills, pack_swq_o, pack_swq)
    pack_cc = torch.where(overfills, pack_cc_o, pack_cc)

    s_surf = torch.where(surface_swq > 0.0, surface_swq,
                         torch.ones_like(surface_swq))
    s_pack = torch.where(pack_swq > 0.0, pack_swq, torch.ones_like(pack_swq))
    surf_temp = torch.where(surface_swq > 0.0,
                            surface_cc / (CONST_VCPICE_WQ * s_surf),
                            torch.zeros_like(surf_temp))
    pack_temp = torch.where(pack_swq > 0.0,
                            pack_cc / (CONST_VCPICE_WQ * s_pack),
                            torch.zeros_like(pack_temp))
    if cold:
        # SurfaceCC and PackCC as the C carries them on (:101-131), not
        # rebuilt from the temperatures just divided out of them, which
        # rounds an ulp apart
        return surface_swq, pack_swq, surf_temp, pack_temp, surface_cc, pack_cc
    return surface_swq, pack_swq, surf_temp, pack_temp


def snow_coverage(store_snow, max_slope, old_coverage, swq, old_swq,
                  depth, old_depth, melt, max_snow_depth, snowfall,
                  store_swq, snow_distrib_slope, store_coverage):
    """Sub-grid snow cover fraction.  calc_snow_coverage.c:38-160.

    VIC's SPATIAL_SNOW: instead of a tile being all snow or no snow, the pack
    is taken to be spread over the tile with a linear depth distribution, so
    that as it melts the shallow end goes bare first and the covered fraction
    falls continuously from 1 to 0. `max_slope` is the soil file's
    `max_snow_distrib_slope`, the deepest such distribution the tile allows.

    The distribution has memory. Once melt has exposed part of a tile, a thin
    later snowfall sits on top of that pattern rather than erasing it: the C
    parks the buried cover fraction in `store_coverage` and the new snow's
    water equivalent in `store_swq`, and restores them when that thin layer
    melts off again. A snowfall deep enough to bury the pattern entirely
    (`depth >= max_slope / 2`) does erase it, and the tile starts uniform.

    Every argument and every returned quantity is a tensor of the same shape,
    one entry per tile. `store_snow` is boolean; the rest are floating point.
    Returns a dict with keys

        coverage, store_snow, store_swq, snow_distrib_slope,
        max_snow_depth, store_coverage

    The C's three-way `if (snowfall > 0) / else if (melt > 0) / else` and the
    nested tests inside them are `where` here, so all three arms are
    evaluated and the result selected. That is what makes the function
    differentiable, and it is safe because none of the arms can produce a
    non-finite value: the one division, `-max_snow_depth / snow_distrib_slope`
    at :133, is guarded.

    That guard is the single deliberate departure from the C. The slope is
    set at :122-127 to `-2 * old_depth` when the pack is shallower than
    `max_slope / 2`, so a tile that melts with `old_depth` exactly zero gives
    the C a division by zero and an infinite cover fraction, which then fails
    the `> 1` clamp at :135 because the sign is negative. Here such a tile
    keeps the cover fraction it had. It is reachable only for a tile that
    reports melt with no depth at all.
    """
    zero = torch.zeros_like(swq)
    one = torch.ones_like(swq)
    half = max_slope / 2.0
    true_ = torch.ones_like(store_snow)
    false_ = torch.zeros_like(store_snow)

    snowing = snowfall > 0.0
    # The C's `else if`: melt is only consulted when nothing fell.
    melting = (~snowing) & (melt > 0.0)

    # ---- snowfall > 0: continued accumulation, :60-87 ----------------------
    # The buried cover fraction is stored the first step new snow lands on a
    # partial pack, which the C detects as store_swq still being zero (:67-72)
    # -- so the test is made before store_swq is advanced, not after.
    first = store_snow & (store_swq == 0.0)
    a_store_cov = torch.where(
        first, torch.where(old_coverage < 1.0, old_coverage, one),
        store_coverage)
    a_store_swq = torch.where(store_snow, store_swq + (swq - old_swq),
                              store_swq)
    a_store_snow = store_snow
    a_slope = snow_distrib_slope
    # :77-84, accumulation deep enough to remove the memory of the old melt
    # distribution.
    buried = store_snow & (depth >= half)
    a_store_snow = torch.where(buried, false_, a_store_snow)
    a_store_swq = torch.where(buried, zero, a_store_swq)
    a_slope = torch.where(buried, zero, a_slope)
    a_store_cov = torch.where(buried, one, a_store_cov)
    # :85-88, the first snow to fall on a pattern that is not yet stored.
    opening = (~store_snow) & (old_coverage < 1.0)
    a_store_snow = torch.where(opening, true_, a_store_snow)
    a_store_swq = torch.where(opening, swq - old_swq, a_store_swq)

    # ---- melt > 0: the pack begins or continues melting, :89-146 -----------
    # :96-107, melt the thin later snowfall off the stored distribution.
    shedding = (store_swq > 0.0) & (swq < old_swq)
    b_store_swq = torch.where(shedding, store_swq + (swq - old_swq), store_swq)
    cleared = shedding & (b_store_swq <= 0.0)
    b_old_cov = torch.where(cleared, store_coverage, old_coverage)
    b_store_cov = torch.where(cleared, one, store_coverage)
    b_store_swq = torch.where(cleared, zero, b_store_swq)

    # :109, the stored layer is gone and the old distribution is exposed.
    exposed = b_store_swq == 0.0
    # :112-127, build the distribution the first time melt exposes it.
    opens = exposed & (snow_distrib_slope == 0.0)
    b_slope = torch.where(
        opens, torch.where(old_depth > half, -max_slope, -2.0 * old_depth),
        snow_distrib_slope)
    b_store_snow = torch.where(opens, true_, store_snow)
    # :129-134: old_max_snow_depth is read after the assignment at :126, so
    # on the step the distribution is built it is the slope's magnitude.
    old_max = torch.where(opens, -b_slope, max_snow_depth)
    b_max_depth = torch.where(exposed, 2.0 * depth, max_snow_depth)
    # :135-137, melt has occurred and the cover fraction falls with it;
    # otherwise rain or sublimation raised swq and the fraction is unchanged.
    shrunk = exposed & ((b_max_depth < old_max) | (old_max == 0.0))
    usable = b_slope < 0.0
    safe_slope = torch.where(usable, b_slope, -one)
    fell = torch.clamp(-b_max_depth / safe_slope, max=1.0)
    b_cov = torch.where(shrunk & usable, fell, b_old_cov)

    # ---- select the arm, :147-152 ------------------------------------------
    def pick(a, b, c):
        return torch.where(snowing, a, torch.where(melting, b, c))

    return dict(
        coverage=pick(one, b_cov, old_coverage),
        store_snow=pick(a_store_snow, b_store_snow, store_snow),
        store_swq=pick(a_store_swq, b_store_swq, store_swq),
        snow_distrib_slope=pick(a_slope, b_slope, snow_distrib_slope),
        max_snow_depth=pick(max_snow_depth, b_max_depth, max_snow_depth),
        store_coverage=pick(a_store_cov, b_store_cov, store_coverage))


def snow_step(state, forcing, thin_to_surface=False, p=None,
              unstable=False, snow_density='DENS_BRAS'):
    """Advance the snowpack one time step.  snow_melt.c, assembled.

    The analogue of `torchvic.core.column.step` for snow: it chains the verified
    pieces of this module in the order snow_melt() runs them, so the whole
    step is differentiable end to end.

    `state` is a dict with keys
        ice, surf_water, pack_water, surf_temp, pack_temp
    `forcing` is a dict with keys
        snowfall, air_temp, t_grnd, wind, net_short, long_snow_in, air_dens,
        eact_air, lv, press, vpd, rain, dt, z, z0, ra, snow_depth,
        snow_density
    and optionally `t_canopy`: the canopy air temperature the pack's own
    balance runs at. snow_melt() hands SnowPackEnergyBalance Tcanopy as
    its air temperature (snow_melt.c:159, :257, :288, :316) while the
    cold content of new snow, the density and the ageing still run at
    the air's (air_temp). Without the key the two coincide, as they do
    with CLOSE_ENERGY false.

    `unstable` (a bool or a bool tensor shaped like the state) is
    surface_fluxes.c's UNSTABLE_SNOW: the pack skips the melt and the
    solve alike and is left for the surface (snow_melt.c:169, :353-357),
    which is the thin path below, forced.

    Rainfall must already be in `state['surf_water']`: snow_melt.c:168 adds it
    to the surface liquid store before the balance is formed, and
    `forcing['rain']` is used only for the heat the rain advects.

    Returns (new_state, melt), where `melt` is the outflow leaving the pack
    in the same units as the water stores. The new state carries, besides the
    stores and temperatures, `sublimation` -- the mass the pack lost to the
    air over the step, positive for a loss, in the same units. That is VIC's
    OUT_SUB_SNOW and its sign convention (put_data.c:627, after the flip at
    snow_melt.c:499). It also carries `depth` and `density`, updated as
    solve_snow.c:358-372 updates them, but only when `state` carried a
    `depth` on the way in: see the comment at that step for why a caller has
    to opt in rather than have a placeholder compacted on its behalf.

    Order, following the C:

    1. split the pack into layers and add snowfall with its cold content
    2. solve the surface energy balance for surface temperature
    3. melt or refreeze at the surface with whatever energy is left, and
       exchange vapour with the air
    4. refreeze pack water and route liquid between the layers
    5. rebalance the layers and recompute their temperatures
    6. compact and age the pack, and recompute its depth

    `snow_density` is VIC's SNOW_DENSITY option, 'DENS_BRAS' (VIC's default)
    or 'DENS_SNTHRM', selecting both the compaction branch of step 6 and
    the new-snow formula wherever one is taken.

    Step 2 is the only one that is not a direct transcription: the surface
    temperature comes from a root find, differentiated with the implicit
    function theorem rather than by unrolling the solver
    (`torchvic.core.implicit`). Its value agrees with a root of VIC's own residual
    to solver tolerance rather than bit for bit, for the reason recorded
    there; every other step is bit-exact.

    Not implemented, and not approximated: canopy interception
    (snow_intercept.c), the blowing-snow vapour fluxes (CalcBlowingSnow.c),
    SPATIAL_SNOW and with it calc_snow_coverage.c, and the thin-pack
    handoff. That last one is the one to watch: snow_melt.c:266 solves the
    surface temperature only when the surface layer exceeds
    param.SNOW_MIN_SWQ_EB_THRES = 0.001 m, and otherwise sets surf_temp to
    the flag value 999 so that calc_surf_energy_bal.c solves the snow and the
    ground together. This function solves the snow layer alone at every
    depth, which is a real difference below 1 mm of surface water
    equivalent, not a tolerance.
    """
    from . import implicit

    ice = state['ice']
    surf_water = state['surf_water']
    pack_water = state['pack_water']
    # The water equivalent the pack had before this step, which is what
    # solve_snow.c:292 stores as `old_swq` and hands to snow_density(). The
    # rain is subtracted back out because the caller has already added it to
    # the surface liquid store (see the docstring) whereas VIC adds it at
    # snow_melt.c:168, which is after old_swq was taken.
    old_swq = ice + surf_water + pack_water - forcing['rain']
    sw_pre = state.get('sw_pre')
    if sw_pre is not None:
        # VIC carries the pack as snow->swq = Ice + pack_water + surf_water
        # (snow_melt.c 5.1.0:464) and takes the ice back out of it on the
        # next call, Ice = swq - pack_water - surf_water (:90), before the
        # rain reaches surf_water (:148). Those round, so the ice is re-
        # derived the C's way from the liquid the step started with.
        old_swq = ice + pack_water + sw_pre
        swq_c = state.get('swq_c')
        if swq_c is not None:
            # the swq the C carried, where it is the one this ice came from
            # (a thin pack's, updated in place by calc_surf_energy_bal.c
            # 5.1.0:637-665, is not ice + water re-summed)
            old_swq = torch.where((swq_c - pack_water) - sw_pre == ice,
                                  swq_c, old_swq)
        ice = old_swq - pack_water - sw_pre

    # snow_melt.c:107 records OldTSurf from snow->surf_temp *before* the fresh
    # snowfall is distributed at 132-164, and that distribution overwrites
    # snow->surf_temp with the mixed temperature of the old surface layer and
    # the new snow. Taking OldTSurf after the mixing instead -- which is what
    # this used to do -- changes the change-in-cold-content term
    # CONST_VCPICE_WQ * SweSurfaceLayer * (TSurf - OldTSurf) / Dt by the whole
    # temperature of the new snow. Against build/ref_snow, 8 mm of snow at
    # -6 C onto bare ground came out at -12.28 C where VIC gives -11.51 C.
    old_t_surf = state['surf_temp']

    (surface_swq, pack_swq, surf_temp, pack_temp, surface_cc0,
     pack_cc0) = accumulate_snowfall(
        ice, state['surf_temp'], state['pack_temp'],
        forcing['snowfall'], forcing['air_temp'], cold=True)
    # snow_melt.c 5.1.0:147, Ice += SnowFall
    ice = (surface_swq + pack_swq if sw_pre is None
           else ice + forcing['snowfall'])
    # PackCC as snow_melt.c 5.1.0:101-127 leaves it once the snowfall is
    # distributed. Nothing updates it again before the pack refreeze at
    # :390-418 reads it -- in particular not the melt that :217-224 took out
    # of PackSwq -- so it is the cold content of the pack as it stood here.
    # SurfaceCC as snow_melt.c 5.1.0:103-131 leaves it once the snowfall is
    # distributed. It is not touched again unless the pack ends the step
    # above SNOW_MAX_SURFACE_SWE (:433-456), and :478 stores it as
    # snow->coldcontent -- see `cold_content` at the end.

    eb = dict(forcing)
    # snow_pack_energy_balance names the air temperature t_air; the forcing
    # dict uses air_temp throughout for consistency with accumulate_snowfall.
    # With CLOSE_ENERGY the balance runs at the canopy air temperature,
    # which the caller carries separately (snow_melt.c:159); without the
    # key it is the air's, as before, to the object.
    eb['t_air'] = (forcing['air_temp'] if 't_canopy' not in forcing
                   else forcing['t_canopy'])
    eb['swe_surface'] = surface_swq
    eb['old_t_surf'] = old_t_surf
    eb['surface_liquid_water'] = surf_water
    # snow_pack_energy_balance may keep its temperature-free terms here:
    # nothing in `eb` changes between its evaluations below
    eb['_invariant'] = None
    # CalcBlowingSnow's flux where the caller ran it (surface_fluxes.c
    # :443-463), zero otherwise -- which is what VIC leaves
    # snow->blowing_flux at with the option off.
    _blowing_on = 'blowing_mass_flux' in eb
    if not _blowing_on:
        eb['blowing_mass_flux'] = torch.zeros_like(surf_water)

    # The C tests the balance at 0 C *before* solving (snow_melt.c:263). If it
    # is positive the pack cannot be colder than freezing, so the surface sits
    # at 0 C and the surplus goes into melt; only when it is non-positive is
    # there a sub-freezing root to find.
    #
    # Skipping this test and solving unconditionally is wrong twice over: the
    # residual then has no sign change in the bracket, so bisection returns a
    # bound rather than a root, and the melt branch is never entered. That
    # produced a pack which would not melt under any forcing.
    # snow_melt.c:172 evaluates the balance at 0 C first. That call returns a
    # residual of exactly zero when the pack is melting -- the balance forces
    # it to zero and hands the surplus energy back through RefreezeEnergy --
    # so `Qnet == 0.0` at line 191 is the melting test, not a positive
    # residual. Only when it is non-zero is there a sub-freezing root.
    zero_t = torch.zeros_like(surf_temp)
    qnet_at_zero, refreeze_energy, vapor_flux_melt, terms_melt = \
        snow_pack_energy_balance(zero_t, eb, return_terms=True)
    melting = qnet_at_zero == 0.0
    _unst = None
    if torch.is_tensor(unstable):
        _unst = unstable
    elif unstable:
        _unst = torch.ones_like(melting)
    if _unst is not None:
        # An unstable pack skips the melt and the solve alike and is left
        # for the surface at surf_temp 999 (snow_melt.c:169, :353-357) --
        # the thin path, forced, since what it skips is exactly what the
        # thin pack skips (:304-324): no melt application, no frozen
        # application, the 0 C terms, then the liquid update below.
        _unst = _unst & (ice > 0.0)
        melting = melting & ~_unst

    if SNOW_SOLVER == 'brent':
        t_solved = _solve_pack_brent(surf_temp, old_t_surf, eb)
    else:
        SIDE['last'] = None
        lo = torch.full_like(surf_temp, -80.0)
        hi = zero_t
        t_solved = implicit.solve(_pack_residual_fast, lo, hi, args=(eb,),
                                  tol=implicit.ROOT_BRENT_T)
    # snow_melt.c:329 calls the balance again at the solved temperature, and
    # it is that call which leaves snow->vapor_flux at the value the frozen
    # branch then applies to the ice. The vapour flux is not a by-product of
    # the root find; it is a state variable the balance writes, and which of
    # the two calls wrote it last decides how much mass the pack loses.
    _, refreeze_frozen, vapor_flux_frozen, terms_frozen = \
        snow_pack_energy_balance(t_solved, eb, return_terms=True)
    vapor_flux = torch.where(melting, vapor_flux_melt, vapor_flux_frozen)
    # The energy terms at the temperature the pack actually takes this step:
    # 0 C on the melting branch, the root otherwise. snow_melt.c evaluates
    # the balance once at the branch it takes; both are evaluated here and
    # selected, so the terms belong to the selected branch.
    energy_terms = {k: torch.where(melting, terms_melt[k], terms_frozen[k])
                    for k in terms_melt}
    # RefreezeEnergy as the C leaves it: on the frozen branch the liquid's
    # latent heat (:353-358, where the surface water refreezes); on the
    # melting branch the surplus, unless the pack melts completely, when
    # :256-258 rescales it to the melt that was actually available,
    # keeping its sign.
    refreeze_reported = torch.where(melting, refreeze_energy, refreeze_frozen)

    # snow_melt.c:266 and :322-324: a *frozen* pack whose surface layer is
    # at or under SNOW_MIN_SWQ_EB_THRES gets no solve here -- surf_temp is
    # set to 999 and the block that applies the vapour flux and refreezes
    # the surface water is skipped -- and the soil surface solves it with
    # the pack included. The caller asks for that with `thin_to_surface`
    # and reads `thin` back; a melting thin pack stays the pack's own.
    # `thin_to_surface` may also be a bool tensor shaped like the state,
    # true where the caller evaluates thin packs itself and false where it
    # wants them solved here, as before.
    if torch.is_tensor(thin_to_surface):
        thin = ((~melting) & (surface_swq <= SNOW_MIN_SWQ_EB_THRES)
                & (ice > 0.0) & thin_to_surface)
    elif thin_to_surface:
        thin = (~melting) & (surface_swq <= SNOW_MIN_SWQ_EB_THRES) & (ice > 0.0)
    else:
        thin = torch.full_like(melting, False)
    if _unst is not None:
        thin = thin | _unst
    SIDE['pack'] = (None if SIDE['last'] is None
                    else SIDE['last'] & ~melting & ~thin & (ice > 0.0))
    vapor_flux = torch.where(thin, torch.zeros_like(vapor_flux), vapor_flux)
    vapor_flux_pre = vapor_flux
    t_new = torch.where(melting, zero_t, t_solved)
    (melt_s, surf_water_m, surface_swq_m, pack_swq_m, pack_water_m, ice_m,
     _, _, _, vapor_applied_m) = surface_melt(
        refreeze_energy, surf_water, surface_swq, pack_swq, pack_water, ice,
        forcing['dt'], vapor_flux)

    zero = torch.zeros_like(surf_water)

    # --- the sub-freezing branch, snow_melt.c:352-374 ---------------------
    # The surface temperature came out below freezing, so every drop of
    # liquid in the surface layer has frozen; the C moves it into the ice
    # outright. Leaving it liquid, as this used to, both loses the phase
    # change and leaves the layer water equivalents wrong, which then feeds
    # the wrong cold content into the rebalance: a 0.5 m pack at -12 C
    # holding 2 mm of surface liquid came out with a pack temperature of
    # -14.000 against VIC's -14.0155.
    # :353-357 refreeze the surface water into the frozen surface layer;
    # a thin pack skips that block (:322-324) and keeps its liquid.
    surface_swq_f = torch.where(thin, surface_swq, surface_swq + surf_water)
    ice_f = torch.where(thin, ice, ice + surf_water)

    # ... and then the vapour exchange comes off the ice rather than off the
    # liquid, capped at the ice that is there (snow_melt.c:362-374). Note the
    # C sets Ice = PackSwq when the flux exhausts the surface layer, rather
    # than subtracting, because the surface layer is gone entirely.
    exhausted = surface_swq_f < -vapor_flux
    vapor_applied_f = torch.where(exhausted, -surface_swq_f, vapor_flux)
    surface_swq_f = torch.where(exhausted, torch.zeros_like(surface_swq_f),
                                surface_swq_f + vapor_flux)
    ice_f = torch.where(exhausted, pack_swq, ice_f + vapor_flux)

    complete = melting & (ice_m == 0.0) & (refreeze_energy != 0.0)
    safe_abs = torch.where(refreeze_energy != 0.0, refreeze_energy.abs(),
                           torch.ones_like(refreeze_energy))
    refreeze_reported = torch.where(
        complete,
        refreeze_energy / safe_abs * melt_s * CONST_LATICE * CONST_RHOFW
        / forcing['dt'],
        refreeze_reported)
    energy_terms['refreeze_energy'] = refreeze_reported

    surf_water = torch.where(melting, surf_water_m,
                             torch.where(thin, surf_water, zero))
    surface_swq = torch.where(melting, surface_swq_m, surface_swq_f)
    pack_swq = torch.where(melting, pack_swq_m, pack_swq)
    pack_water = torch.where(melting, pack_water_m, pack_water)
    ice = torch.where(melting, ice_m, ice_f)
    # a thin pack's temperature is the surface solve's to set
    # (calc_surf_energy_bal.c:682); until then it keeps the mixed value
    surf_temp = torch.where(melting, zero, torch.where(thin, surf_temp, t_new))
    energy_terms = {k: torch.where(thin, torch.zeros_like(v), v)
                    for k, v in energy_terms.items()}
    # A thin pack's last SnowPackEnergyBalance call is the one at 0 C
    # (snow_melt.c 5.1.0:172, no root_brent below :266), and its
    # AdvectedEnergy, CPFW RHOFW Tair Rain / dt (SnowPackEnergyBalance.c
    # 5.1.0:224-229), survives the hand-off: surface_fluxes.c 5.1.0:593
    # copies it to the soil energy, :874 back to the pack, and :911-929
    # report it as OUT_ADVECTION.
    energy_terms['advection'] = torch.where(thin, terms_melt['advection'],
                                            energy_terms['advection'])
    refreeze_reported = torch.where(thin, torch.zeros_like(refreeze_reported),
                                    refreeze_reported)
    vapor_flux = torch.where(melting, vapor_applied_m, vapor_applied_f)

    pack_cc = pack_cc0
    (surf_water, pack_water, pack_swq, ice_gain, pack_temp,
     melt) = pack_refreeze(surf_water, pack_water, surface_swq, pack_swq,
                           pack_cc, pack_temp)
    ice = ice + ice_gain
    # snow_melt.c:454 rebuilds Ice from the two layers rather than carrying
    # the running total. The two agree here, but writing it the C's way means
    # a future change to either layer cannot put them out of step silently.
    ice = surface_swq + pack_swq
    big = ice > SNOW_MAX_SURFACE_SWE

    surface_swq, pack_swq, surf_temp, pack_temp, swq = rebalance_layers(
        ice, surface_swq, pack_swq, surf_temp, pack_temp, pack_water,
        surf_water)

    # --- compaction, ageing and depth, solve_snow.c:358-372 ---------------
    # snow_melt() itself never touches snow->density or snow->depth; its
    # caller does, after the mass balance is settled, using the *entry* depth
    # and the water equivalent the pack had before the step. Both feed the
    # ground conduction term of the next step's energy balance, so leaving
    # them frozen -- which is what the caller does when it supplies a
    # constant density and a depth of zero -- pins that term at zero forever.
    #
    # solve_snow.c:366-367 has a second branch, for a pack whose surface is
    # above freezing, which takes the new snow's density outright. It cannot
    # be reached from here: `surf_temp` leaving this function is either
    # exactly 0 (melting) or the sub-freezing root, never positive.
    # The update runs only when the caller carries the depth in its state,
    # which is how it says it intends to carry it forward. A caller that puts
    # a placeholder in `forcing['snow_depth']` for the conduction term and
    # nothing in the state -- which is what `torchvic.model.basin` does today, with a
    # depth of zero and a fixed density -- gets no depth and no density back,
    # rather than a number computed from a placeholder. Under DENS_BRAS
    # `snow_density_bras` divides by the entry depth when nothing has fallen,
    # exactly as snow_utility.c:182 does, so a zero placeholder would come
    # back infinite.
    depth0 = state.get('depth')
    if depth0 is None:
        density = None
        depth = None
    else:
        # solve_snow.c 5.1.0:346 hands snow_density the snowfall in mm as
        # it has it, not the metres snow_melt divided out of it
        _sf_mm = forcing.get('snowfall_mm')
        if _sf_mm is None:
            _sf_mm = forcing['snowfall'] * MM_PER_M
        if snow_density == 'DENS_BRAS':
            density = snow_density_bras(
                depth0, old_swq, surf_temp,
                _sf_mm, old_swq,
                forcing['air_temp'], forcing['dt'])
            new_snow_method = 'bras'
        elif snow_density == 'DENS_SNTHRM':
            # solve_snow.c 5.1.0:343-347 hands snow_density() the entry
            # depth and density with the post-melt liquid stores;
            # `surf_water` and `pack_water` here are those stores,
            # `state['density']` the entry density (VIC cold-starts it at
            # 0, initialize_snow.c 5.1.0:24).
            dens0 = state.get('density')
            if dens0 is None:
                dens0 = torch.zeros_like(depth0)
            density = snow_density_snthrm(
                dens0, depth0, surf_temp,
                _sf_mm, old_swq,
                forcing['air_temp'], forcing['dt'], surf_water, pack_water)
            new_snow_method = 'snthrm'
        else:
            raise ValueError("snow_density must be 'DENS_BRAS' or "
                             "'DENS_SNTHRM'; got %r" % (snow_density,))
        # An empty pack has neither: solve_snow.c:471-472.
        # solve_snow.c 5.1.0:343-353: snow_melt leaves a thin pack's
        # surf_temp at 999 (snow_melt.c 5.1.0:302-304), so `surf_temp <= 0`
        # fails and the pack is neither compacted nor aged: it takes
        # new_snow_density(air_temp) on a step with last_snow == 0 (snow from
        # the sky, or a pack just begun) and keeps its density otherwise.
        # The same `else` takes a pack whose surface the layer rebalance
        # (snow_melt.c 5.1.0:433-456) left above 0 C: ISC Gangotri
        # 2004-04-22, 0.0208 C, where VIC 5.0.1 gave the pack new snow's
        # 109 kg m-3 and a depth of 2.44 m.
        warm = torch.as_tensor(thin, device=surf_temp.device) | (surf_temp > 0.0)
        # graph-safe: taken always; with no warm pack it selects `density`
        if graphsafe.ACTIVE or bool(warm.any()):
            last0 = forcing.get('last_snow_zero')
            if last0 is None:
                last0 = forcing['snowfall'] > 0.0
            dens_prev = state.get('density', density)
            # The same option reads at both ends of solve_snow.c:343-349:
            # a warm pack takes new snow's density under either method.
            density = torch.where(
                warm, torch.where(last0, new_snow_density(
                    forcing['air_temp'], method=new_snow_method),
                    dens_prev), density)
        has_pack = swq > 0.0
        density = torch.where(has_pack, density, torch.zeros_like(density))

        # The guard is on the DENSITY, not on whether there is a pack.
        # Those are not the same condition, and assuming they were left
        # both an infinite depth and a NaN gradient in the model.
        #
        # `snow_density_bras` returns zero when it is given no entry depth,
        # no standing water equivalent and no snowfall -- its no-snowfall
        # branch is CONST_RHOFW * sswq / depth0 with sswq zero. A tile can
        # reach that and still satisfy `swq > 0` at this point, because
        # `swq` here is the water equivalent AFTER the step while bras was
        # given the values from before it. Measured on the Indian 6 km
        # data: 119 elements per step with has_pack true and density
        # exactly 0.
        #
        # With the old guard those elements divided by zero. The forward
        # value was +inf -- an infinite snow depth, which no output
        # reported because `depth` is consumed by the ground conduction
        # term (basin.py:1907) as density**2 / depth, and 0/inf is 0. The
        # gradient was 0/0 = NaN, and that NaN reached every parameter
        # whose gradient flows through the snow pack: on this dataset it
        # left 1 of 16 parameters with a usable gradient. A forward run
        # that conserves to 1e-13 is not evidence a model's gradients are
        # usable.
        #
        # A pack with no density has no depth to report, which is the same
        # answer an empty pack gets, so the selector carries the condition
        # and the divisor is never zero in either branch.
        live = has_pack & (density > 0.0)
        safe_density = torch.where(live, density, torch.ones_like(density))
        depth = torch.where(live, CONST_RHOFW * swq / safe_density,
                            torch.zeros_like(swq))

    # Only the pack outflow leaves the snowpack. `melt_s` from surface_melt
    # is ice converted to liquid *within* the pack: it is added to surf_water,
    # which then routes through pack_refreeze to the outlet. Returning their
    # sum counts the same water twice, which showed up as melt of 0.2 from a
    # pack holding 0.1.
    #
    # `sublimation` is reported with VIC's OUT_SUB_SNOW sign: snow_melt.c:499
    # flips snow->vapor_flux on the way out so that put_data.c:627 reports a
    # positive number when the pack loses mass to the air.
    # The water equivalent carried forward is the C's `Ice`, the total taken
    # at snow_melt.c:454 *before* the layers are rebalanced, and it is what
    # snow->swq is built from at :485. It is not the sum of the two layers
    # after the rebalance: in the Ice <= SNOW_MAX_SURFACE_SWE branch the C
    # discards PackSwq without adding it to SurfaceSwq (:476-480), which is
    # harmless there because the layers are locals re-derived from Ice at
    # :114-120 on the next call -- accumulate_snowfall does the same here --
    # but summing the layers after that branch dropped the pack's ice on the
    # floor. It fired only when a pack within a few mm of 125 mm was being
    # drawn down through the layer maximum by sublimation: 3.84e-5 m in one
    # step of a synthetic case, and 1.5e-2 mm on 1977-01-24 at FindleyLake
    # (SWE 124.9 mm), which was the whole decade's water-balance residual.
    out = dict(ice=ice, surf_water=surf_water,
               pack_water=pack_water, surf_temp=surf_temp,
               pack_temp=pack_temp, swq=swq,
               surface_swq=surface_swq, pack_swq=pack_swq,
               sublimation=-vapor_flux, swq_c=swq)
    # snow->coldcontent (snow_melt.c 5.1.0:478) is SurfaceCC, which the C
    # recomputes from the rebalanced surface layer only when the pack ends
    # the step above SNOW_MAX_SURFACE_SWE (:433-456). Below that -- every
    # pack under 125 mm -- it is the cold content the surface layer had at
    # the START of the step, after the snowfall was mixed in (:103-131),
    # whatever the step then did. Its sign picks the albedo curve
    # (snow_utility.c 5.1.0:246-262) and sets MELTING (solve_snow.c 5.1.0
    # :360-370), so a thin pack that melts this step is still "cold" to
    # both until the next step. Reading it as `surf_temp >= 0` put ISC's
    # Tibetan packs on the melt-season albedo curve a step early.
    out['cold_content'] = torch.where(
        big, CONST_VCPICE_WQ * surf_temp * surface_swq, surface_cc0)
    if density is not None:
        out['density'] = density
        out['depth'] = depth
    out['energy'] = energy_terms
    out['thin'] = thin
    # snow_melt.c:201, :345 (calc_surf_energy_bal.c:631): when the vapour
    # flux exceeds what the layer holds, VIC scales blowing_flux by the same
    # factor as vapor_flux (-(supply) / vapor_flux); this is that factor
    # (1 where nothing was capped), for OUT_SUB_BLOWING
    if _blowing_on:
        _pre = vapor_flux_pre
        out['blow_scale'] = torch.where(
            (vapor_flux != _pre) & (_pre != 0.0),
            vapor_flux / torch.where(_pre != 0.0, _pre,
                                     torch.ones_like(_pre)),
            torch.ones_like(vapor_flux))
    return out, melt


# the step as fused regions on the CUDA graph path (core.fusion), the
# pack's root find between them
snow_step = fusion.region(snow_step)
