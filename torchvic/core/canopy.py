"""Differentiable canopy snow and rain interception.

`vic/vic_run/src/snow_intercept.c` does two separable things: it partitions
snowfall and rainfall between the canopy and the ground, and it solves a
canopy energy balance for foliage temperature with `root_brent`. This module
covers the first -- the interception mass balance, which is closed form. The
energy balance is a root find and belongs with the machinery in
`torchvic.core.implicit`; it is not approximated here.

The interception capacity for snow follows Kobayashi (1986) via a piecewise
function of foliage temperature: near freezing the canopy holds several times
more snow than when it is cold, because wet snow adheres. Windy conditions
below -3 C blow some of it off again (Schmidt and Troendle 1992).

Conventions follow the C. Depths are in metres inside the routine, `F` is the
canopy fraction, `LAI` the leaf area index, `MaxInt` the maximum liquid
interception capacity. Everything is batched over a leading cell axis.
"""
import torch

from .devconst import device_constant
from . import fusion, graphsafe

DBL_EPSILON = 2.220446049250313e-16
VEG_LAI_SNOW_MULTIPLIER = 0.0005
SNOW_LIQUID_WATER_CAPACITY = 0.035
SNOW_MIN_SWQ_EB_THRES = 0.0010


def max_snow_interception(t_foliage, lai,
                          multiplier=VEG_LAI_SNOW_MULTIPLIER):
    """Maximum snow the canopy can hold, m.  snow_intercept.c:149.

    Piecewise in foliage temperature, from Kobayashi (1986) figure 4. Warm
    foliage holds four times its cold capacity because wet snow sticks; the
    band between -3 and -1 C interpolates linearly.

    The C's branch order matters and is preserved: the middle band is tested
    first, so a temperature above -1 C reaches the `> -1` branch only by
    failing the first test.
    """
    mid = t_foliage * 3.0 / 2.0 + 11.0 / 2.0
    warm = torch.full_like(t_foliage, 4.0)
    cold = torch.ones_like(t_foliage)
    in_band = (t_foliage < -1.0) & (t_foliage > -3.0)
    m = torch.where(in_band, mid, torch.where(t_foliage > -1.0, warm, cold))
    return m * multiplier * lai


def intercept(t_foliage, lai, snowfall, rainfall, int_snow, int_rain, wind,
              canopy_fraction, max_int,
              multiplier=VEG_LAI_SNOW_MULTIPLIER,
              liquid_capacity=SNOW_LIQUID_WATER_CAPACITY,
              min_swq_thres=SNOW_MIN_SWQ_EB_THRES):
    """Partition snowfall and rainfall between canopy and ground.
    snow_intercept.c:140-270.

    Returns (int_snow, int_rain, snow_throughfall, rain_throughfall,
             blown_snow).

    The order of operations is the C's, and it is load-bearing rather than
    incidental:

    1. snow interception is capped by the temperature-dependent capacity
    2. wind blows some of the new interception off, below -3 C
    3. the structural limit `Imax1` can zero the interception entirely --
       note it zeroes `DeltaSnowInt`, not the excess, so a canopy already near
       its limit intercepts nothing rather than a little
    4. a canopy holding too little to solve an energy balance for drops its
       load through
    5. rain is intercepted up to a capacity that itself depends on how much
       snow is held
    6. overloading sloughs rain and snow off in proportion to their shares
    """
    imax1 = 4.0 * multiplier * lai
    f = canopy_fraction

    max_snow_int = max_snow_interception(t_foliage, lai, multiplier)

    # 1. Interception, capped by capacity.
    safe_max = torch.where(max_snow_int > 0.0, max_snow_int,
                           torch.ones_like(max_snow_int))
    d_snow = (1 - int_snow / safe_max) * snowfall
    d_snow = torch.minimum(d_snow, max_snow_int - int_snow)
    d_snow = torch.clamp(d_snow, min=0.0)
    d_snow = torch.where(max_snow_int > 0.0, d_snow,
                         torch.zeros_like(d_snow))

    # 2. Wind removes part of the new interception when cold.
    blown = (0.2 * wind - 0.2) * d_snow
    blown = torch.minimum(blown, d_snow)
    blows = (t_foliage < -3.0) & (d_snow > 0.0) & (wind > 1.0)
    blown = torch.where(blows, blown, torch.zeros_like(blown))
    d_snow = d_snow - blown

    # 3. Structural limit: the C zeroes the increment, not the excess.
    d_snow = torch.where(int_snow + d_snow > imax1,
                         torch.zeros_like(d_snow), d_snow)

    snow_through = (snowfall - d_snow) * f + snowfall * (1 - f)

    # 4. Too little snow in the canopy to solve an energy balance for.
    thin_snow = (snowfall == 0) & (int_snow < min_swq_thres)
    snow_through = torch.where(thin_snow, snow_through + int_snow,
                               snow_through)
    d_snow = torch.where(thin_snow, d_snow - int_snow, d_snow)
    int_snow = int_snow + d_snow
    # :231: a store below DBL_EPSILON is zeroed, not carried
    int_snow = torch.where(int_snow < DBL_EPSILON, torch.zeros_like(int_snow),
                           int_snow)

    # 5. Rain, with a capacity that depends on the snow load.
    max_water_int = liquid_capacity * int_snow + max_int
    over = int_rain + rainfall > max_water_int
    rain_through = torch.where(
        over,
        (int_rain + rainfall - max_water_int) * f + rainfall * (1 - f),
        rainfall * (1 - f))
    # :243-245 adds RainFall to IntRain *unscaled*: the store is per canopy
    # area (divided by F at :152) and the C adds the pixel depth as it is.
    # This used to add rainfall * f, the same only at F = 1.
    int_rain = torch.where(over, max_water_int, int_rain + rainfall)

    thin_rain = (rainfall == 0) & (int_rain < min_swq_thres)
    rain_through = torch.where(thin_rain, rain_through + int_rain,
                               rain_through)
    int_rain = torch.where(thin_rain, torch.zeros_like(int_rain), int_rain)

    # 6. Overloading sloughs both off in proportion to their shares.
    total = int_rain + int_snow
    safe_total = torch.where(total > 0.0, total, torch.ones_like(total))
    overload = total - imax1
    rain_fract = int_rain / safe_total
    snow_fract = int_snow / safe_total
    loaded = total > imax1
    rain_through = torch.where(loaded,
                               rain_through + overload * rain_fract * f,
                               rain_through)
    snow_through = torch.where(loaded,
                               snow_through + overload * snow_fract * f,
                               snow_through)
    int_rain = torch.where(loaded, int_rain - overload * rain_fract, int_rain)
    int_snow = torch.where(loaded, int_snow - overload * snow_fract, int_snow)

    return int_snow, int_rain, snow_through, rain_through, blown


#: initialize_parameters.c:66: the canopy holds this much snow (m) before
#: mass release can happen (MassRelease.c, snow_intercept.c:467-473).
VEG_MIN_INTERCEPTION_STORAGE = 0.005
#: initialize_parameters.c:201: half-width of the foliage temperature
#: bracket (snow_intercept.c:311-324).
SNOW_DT = 5.0
#: initialize_parameters.c:177: the albedo of snow held in the canopy
#: (snow_intercept.c:293).
SNOW_NEW_SNOW_ALB = 0.85


def mass_release(int_snow, temp_int_storage, released, drip,
                 min_storage=VEG_MIN_INTERCEPTION_STORAGE, max_iter=64):
    """MassRelease.c, vectorised. Returns (int_snow, temp_int_storage,
    released, drip).

    The C recurses: while the melt accumulated in TempInterceptionStorage
    is at least a tenth of the intercepted snow, a tenth drips, up to 17%
    more is released as snow (never below the minimum storage), and it
    tries again on what is left; below the threshold, or once the store is
    under the minimum, the accumulated melt drips as far as the snow goes.
    Each pass removes at least a tenth of the store, so the recursion is
    bounded; here it is a masked loop, each element stopping on its own.
    """
    zero = torch.zeros_like(int_snow)
    active = torch.ones_like(int_snow, dtype=torch.bool)
    for _ in range(max_iter):
        big = int_snow > min_storage
        threshold = 0.10 * int_snow
        max_release = 0.17 * int_snow
        release_branch = active & big & (temp_int_storage >= threshold)
        # :30-45, the release branch, then recurse
        drip_r = drip + threshold
        snow_r = int_snow - threshold
        temp_r = temp_int_storage - threshold
        rel = torch.where(snow_r < min_storage, zero,
                          torch.minimum(snow_r - min_storage, max_release))
        released_r = released + rel
        snow_r = snow_r - rel
        # :46-52, big store but melt below the threshold: drip what there is
        tdrip = torch.minimum(temp_int_storage, int_snow)
        drip_b = drip + tdrip
        snow_b = int_snow - tdrip
        # :56-62, small store: only melt, and the accumulator is emptied
        drip_s = drip + tdrip
        snow_s = int_snow - tdrip
        temp_s = zero
        small = active & ~big
        stop_b = active & big & ~release_branch
        drip = torch.where(release_branch, drip_r,
                           torch.where(stop_b, drip_b,
                                       torch.where(small, drip_s, drip)))
        int_snow = torch.where(release_branch, snow_r,
                               torch.where(stop_b, snow_b,
                                           torch.where(small, snow_s, int_snow)))
        temp_int_storage = torch.where(release_branch, temp_r,
                                       torch.where(small, temp_s,
                                                   temp_int_storage))
        released = torch.where(release_branch, released_r, released)
        active = release_branch
        # The exit test copies from the device and stalls the queue, and the
        # loop is short; testing every eighth pass costs at most seven
        # no-op passes and removes seven eighths of the stalls.
        if (_ % 8) == 7:
            if graphsafe.ACTIVE:
                # eight passes, as the eager loop's first test; a store
                # still releasing after them is recorded (graphsafe)
                graphsafe.need(active, graphsafe.MASS_RELEASE)
                break
            if not bool(active.any()):
                break
    return int_snow, temp_int_storage, released, drip


def melt_and_release(t_foliage, refreeze_energy, vapor_mass_flux, int_snow,
                     int_rain, temp_int_storage, snow_through, initial_int_snow,
                     max_int, dt, liquid_capacity=SNOW_LIQUID_WATER_CAPACITY,
                     min_storage=VEG_MIN_INTERCEPTION_STORAGE,
                     vic_canopy_rounding=False):
    """snow_intercept.c:421-535: after the canopy's temperature is known,
    what its snow and rain do with the energy and the vapour.

    Inputs per canopy area in metres (as the C's stores are after :151-152),
    `vapor_mass_flux` in m per step (the C multiplies the flux by Dt at
    :430) and `refreeze_energy` in J/m2 over the step, with the
    balance's sign -- the C multiplies that by Dt too, at :419, and
    every use of it below is of the multiplied value. `snow_through`
    the snow that already fell through (pixel depth), `initial_int_snow`
    the store at the start of the step (:145).

    Returns (int_snow, int_rain, temp_int_storage, vapor_mass_flux,
    melt_energy, drip, released), all per canopy area, melt_energy in W/m2.

    At 0 C (:433-495): the vapour leaves the liquid first, capped at it; a
    negative refreeze energy melts up to the snow held and MeltEnergy
    carries the heat; the melt goes to the liquid up to MaxWaterInt and the
    rest either drips at once (snowing onto a canopy that started nearly
    empty, :467-473) or accumulates for MassRelease (:479-487); liquid above
    MaxWaterInt drips (:491-494). Below 0 C (:497-535): the accumulator is
    reset, the refreeze energy freezes liquid into snow -- all of it if the
    energy allows, else as much as the energy pays for -- with the heat into
    MeltEnergy, and the vapour then leaves the snow, capped at it.
    """
    zero = torch.zeros_like(int_snow)
    max_water_int = liquid_capacity * int_snow + max_int
    melting = t_foliage == 0.0

    # ---- melting canopy ---------------------------------------------------
    v = vapor_mass_flux
    vap_m = torch.where(-v > int_rain, -int_rain, v)
    rain_m = torch.where(-v > int_rain, zero, int_rain + v)
    # snow_intercept.c:433-437 divides by CONST_LATICE and CONST_RHOFW
    # in two steps and multiplies the melt back in C association; one
    # division by the product rounds 1 ulp off. Gated on the caller's
    # flag so runs without it keep their established rounding bit for
    # bit.
    if vic_canopy_rounding:
        pot_melt = torch.where(refreeze_energy < 0.0,
                               torch.minimum(-refreeze_energy / _LATICE()
                                             / _RHOFW(), int_snow),
                               zero)
        melt_energy_m = -((_LATICE() * pot_melt) * _RHOFW()) / dt
    else:
        lat = _lat_vol()
        pot_melt = torch.where(refreeze_energy < 0.0,
                               torch.minimum(-refreeze_energy / lat, int_snow),
                               zero)
        melt_energy_m = -(lat * pot_melt) / dt
    fits = (rain_m + pot_melt) <= max_water_int
    snow_a = int_snow - pot_melt
    rain_a = rain_m + pot_melt
    excess = pot_melt + rain_m - max_water_int
    snow_b = torch.clamp(int_snow - (max_water_int - rain_m), min=0.0)
    rain_b = max_water_int
    quick = (snow_through > 0.0) & (initial_int_snow <= min_storage)
    drip_b = torch.where(quick, excess, zero)
    snow_b = torch.where(quick, torch.clamp(snow_b - excess, min=0.0), snow_b)
    temp_b = torch.where(quick, temp_int_storage, temp_int_storage + excess)
    snow_b, temp_b, rel_b, drip_b = mass_release(snow_b, temp_b, zero, drip_b,
                                                 min_storage)
    snow_m = torch.where(fits, snow_a, snow_b)
    rain_m2 = torch.where(fits, rain_a, rain_b)
    temp_m = torch.where(fits, temp_int_storage, temp_b)
    drip_m = torch.where(fits, zero, drip_b)
    rel_m = torch.where(fits, zero, rel_b)
    mwi2 = liquid_capacity * snow_m + max_int
    over = rain_m2 > mwi2
    # snow_intercept.c 5.1.0:492, Drip += *IntRain - MaxWaterInt
    drip_m = torch.where(over, drip_m + (rain_m2 - mwi2), drip_m)
    rain_m2 = torch.where(over, mwi2, rain_m2)

    # ---- frozen canopy ----------------------------------------------------
    # :499-519. Note the C compares energies per unit *without* the water
    # density: -RefreezeEnergy > -IntRain * CONST_LATICE, and then converts
    # with / CONST_LATICE alone (metres of water per J/kg -- a factor of
    # CONST_RHOFW off from the melt branch). Reproduced as written.
    all_freezes = -refreeze_energy > -int_rain * _LATICE()
    frz = refreeze_energy.abs() / _LATICE()
    snow_f = torch.where(all_freezes, int_snow + frz, int_snow + int_rain)
    rain_f = torch.where(all_freezes, int_rain - frz, zero)
    # :499-501, C association for the frozen branch's heat too,
    # under the same gate.
    if vic_canopy_rounding:
        melt_energy_f = torch.where(
            all_freezes, refreeze_energy.abs() * _RHOFW() / dt,
            ((_LATICE() * int_rain) * _RHOFW()) / dt)
    else:
        melt_energy_f = torch.where(all_freezes, refreeze_energy.abs() * _RHOFW() / dt,
                                    _lat_vol() * int_rain / dt)
    vap_f = torch.where(-v > snow_f, -snow_f, v)
    snow_f = torch.where(-v > snow_f, zero, snow_f + v)

    int_snow_out = torch.where(melting, snow_m, snow_f)
    int_rain_out = torch.where(melting, rain_m2, rain_f)
    temp_out = torch.where(melting, temp_m, zero)
    vap_out = torch.where(melting, vap_m, vap_f)
    melt_energy = torch.where(melting, melt_energy_m, melt_energy_f)
    drip = torch.where(melting, drip_m, zero)
    released = torch.where(melting, rel_m, zero)
    return (int_snow_out, int_rain_out, temp_out, vap_out, melt_energy, drip,
            released)


def _LATICE():
    from . import snow as _snow
    return _snow.CONST_LATICE


def _RHOFW():
    from . import snow as _snow
    return _snow.CONST_RHOFW


def _lat_vol():
    return _LATICE() * _RHOFW()


EMISS_VEG = 0.97
HUGE_RESIST = 1e20


def canopy_energy_balance(t_foliage, st, return_refreeze=False):
    """Canopy energy balance residual for a snow-laden canopy, W/m2.
    func_canopy_energy_bal.c.

    This is what VIC drives to zero with root_brent to find foliage
    temperature. `st` holds everything fixed while the root is sought:
    long_over_in, long_under_out, net_short_over, air_dens, eact_air, press,
    ra, t_canopy, rainfall, int_rain, wind, dt.

    **Only the intercepted-snow branch (`IntSnow > 0`) is implemented.** The
    snow-free branch calls `canopy_evap`, 500 lines of transpiration physics
    that this module does not cover, and approximating it would be inventing
    numbers. A snow-free canopy must not use this function.

    That restriction costs little: the snow-laden case is the one the snow
    model needs, and it is the branch whose dependencies -- `svp`,
    `calc_outgoing_longwave`, `calc_sensible_heat`,
    `calc_latent_heat_of_sublimation` -- are already verified bit-exact
    against VIC.

    Outgoing longwave is subtracted **twice**: the canopy radiates from both
    faces. That is the C's `- 2 * LongOverOut`, not a transcription slip.

    Like the snowpack balance, the residual is forced to zero at 0 C when
    there is energy to melt with, and the surplus is returned through the
    refreeze term rather than the residual. A caller that discards it cannot
    melt anything -- the mistake that made `snow_step` unable to melt snow
    until it was found.
    """
    from . import snow as _snow

    long_over_out = _snow.calc_outgoing_longwave(
        t_foliage + _snow.CONST_TKFRZ, EMISS_VEG)
    net_radiation = (st['net_short_over'] + st['long_over_in'] +
                     st['long_under_out'] - 2 * long_over_out)

    ra = st['ra']
    es_snow = _snow.svp(t_foliage)
    eps = device_constant(_snow.CONST_EPS, ra.dtype, ra.device)
    vapor_mass_flux = (st['air_dens'] * (eps / st['press']) *
                       (st['eact_air'] - es_snow) / ra / _snow.CONST_RHOFW)
    vapor_mass_flux = torch.where(
        (st.get('vpd', torch.ones_like(ra)) == 0.0) & (vapor_mass_flux < 0.0),
        torch.zeros_like(vapor_mass_flux), vapor_mass_flux)

    ls = _snow.calc_latent_heat_of_sublimation(t_foliage)
    latent_heat_sub = ls * vapor_mass_flux * _snow.CONST_RHOFW
    latent_heat = torch.zeros_like(latent_heat_sub)

    sensible = _snow.calc_sensible_heat(st['air_dens'], st['t_canopy'],
                                        t_foliage, ra)
    advected = (_snow.CONST_CPFW * _snow.CONST_RHOFW * st['t_canopy'] *
                st['rainfall']) / st['dt']

    rest = (sensible + latent_heat + latent_heat_sub + net_radiation +
            advected)

    refreeze = (st['int_rain'] * _snow.CONST_LATICE * _snow.CONST_RHOFW /
                st['dt'])
    melting = (t_foliage == 0.0) & (rest > -refreeze)
    out_refreeze = torch.where(melting, -rest, refreeze)
    out_rest = torch.where(melting, torch.zeros_like(rest), rest + refreeze)

    if return_refreeze:
        return out_rest, out_refreeze
    return out_rest


def canopy_residual(t_foliage, st, evap_of=None, return_terms=False):
    """func_canopy_energy_bal.c, both branches, W/m2.

    `st` carries per-tile tensors: has_snow (IntSnow > 0), net_short_over,
    long_over_in, long_under_out, air_dens, eact_air, press, ra0, ra1,
    t_canopy, vpd, rainfall (m/step), int_rain (m), dt, lv, aero_406
    (bool: AERO_RESIST_CANSNOW is AR_406 / AR_406_LS / AR_406_FULL, the
    resistance to intercepted snow is ten times the canopy's, :129-135 and
    :150-152), aero_410 (bool: AR_410, the resistance is Ra[1] divided by
    the stability correction at the trial foliage temperature, :178-188)
    with wind1, ref_height1, displacement1, roughness1 (Wind[1] and the
    canopy's reference height, displacement and roughness).

    With intercepted snow (:120-152): sublimation from the snow at
    Tfoliage through Ra_used[1], no evaporation. Without (:153-168):
    Ra_used[1] is Ra[1] under AR_406_FULL/AR_410 and Ra[0] otherwise, and
    the latent heat is Le * Evap * RHOFW with Evap from canopy_evap at
    Tfoliage -- `evap_of(t_foliage, net_radiation, net_short_over)` in
    m/s, the caller's canopy evaporation. The C adds that latent heat with
    a positive sign (:167), the opposite of func_surf_energy_bal.c:816;
    reproduced as written. Sensible heat through Ra_used[1] against the
    canopy air (:170), rain advection (:171-172), and at 0 C with
    intercepted snow the surplus becomes RefreezeEnergy with a zero
    residual (:176-186).
    """
    from . import snow as _snow
    zero = torch.zeros_like(t_foliage)
    long_over_out = _snow.calc_outgoing_longwave(t_foliage + _snow.CONST_TKFRZ,
                                                 EMISS_VEG)
    net_rad = (st['net_short_over'] + st['long_over_in'] + st['long_under_out']
               - 2.0 * long_over_out)
    net_long_over = st['long_over_in'] - long_over_out
    has_snow = st['has_snow']
    # The terms that do not depend on Tfoliage, formed once per `st` (the
    # solver evaluates this at a few dozen trial temperatures) with the
    # operations written out below, so the same numbers.
    inv = st.get('_invariant')
    if inv is None or inv['shape'] != tuple(t_foliage.shape):
        _ra_snow = torch.where(st['aero_406'], st['ra1'] * 10.0, st['ra1'])
        _ra_free = torch.where(st['aero_406_full'], st['ra1'], st['ra0'])
        _ra_sens_snow = torch.where(st.get('aero_406_exact',
                                           torch.zeros_like(has_snow)),
                                    st['ra1'], _ra_snow)
        _a410 = st.get('aero_410')
        if _a410 is None:
            _a410 = torch.zeros_like(has_snow)
        # whether any tile runs AR_410: the option's own value when the
        # caller gives it (a Python bool, decided at set-up: asking the
        # device inside a captured CUDA graph is refused), else asked
        _has_410 = st.get('aero_410_on')
        _has_410 = (bool(_a410.any()) if _has_410 is None
                    else bool(_has_410) and _a410 is not None)
        _stab = (None if not _has_410 else _snow.stability_invariants(
            st['ref_height1'], st['displacement1'], st['t_canopy'],
            st['wind1'], st['roughness1']))
        # func_canopy_energy_bal.c:190-191 forms AirDens * (EPS / Press)
        # with an IEEE division; EPS unwrapped is a Python float and
        # `python_float / tensor` rounds 1 ulp off it for some operands
        # (torchvic.core.snow's note), so it is wrapped as a tensor.
        _eps = device_constant(_snow.CONST_EPS, st['press'].dtype,
                               st['press'].device)
        inv = dict(
            shape=tuple(t_foliage.shape), ra_snow=_ra_snow,
            dens_eps=st['air_dens'] * (_eps / st['press']),
            vpd_zero=st['vpd'] == 0.0,
            aero_410=_a410, has_410=_has_410, stab=_stab,
            wind_pos=(st['wind1'] > 0.0) if _has_410 else None,
            ra_used1=torch.where(has_snow, _ra_sens_snow, _ra_free),
            advected=(_snow.CONST_CPFW * _snow.CONST_RHOFW * st['t_canopy']
                      * st['rainfall']) / st['dt'],
            refreeze=torch.where(has_snow, st['int_rain']
                                 * _snow.CONST_LATICE
                                 * _snow.CONST_RHOFW / st['dt'], zero))
        if '_invariant' in st and not torch.is_grad_enabled():
            st['_invariant'] = inv       # the caller's dict allows it
    # --- intercepted snow: sublimation, no evaporation ----------------
    # func_canopy_energy_bal.c:186-189 multiplies Ra_used[1] by ten for
    # AR_406, AR_406_LS and AR_406_FULL, 'to account for smoothing by
    # intercepted snow', and the vapour flux at :209-211 uses that. Then
    # :225-227 divides it back by ten again -- for AR_406 alone -- before
    # the sensible heat at :255 reads the same array. So under AR_406,
    # which is what both sample global files ask for, the canopy sublimes
    # through ten times the resistance and exchanges sensible heat
    # through one times it. Using the multiplied value for both decoupled
    # the canopy from the air by a factor of ten and let it run cold: on
    # FindleyLake it then never reached 0 C, never melted its snow, and
    # kept a 0.85 albedo all winter.
    ra_snow = inv['ra_snow']
    # func_canopy_energy_bal.c:178-188, AR_410 with intercepted snow: no
    # tenfold smoothing; Ra_used[1] is Ra[1] divided by the stability
    # correction at the trial foliage temperature, HUGE_RESIST without
    # wind. The correction moves with Tfoliage, so it is formed on every
    # evaluation, with the temperature-free terms from `inv['stab']`.
    use_410 = None
    if inv['has_410']:
        use_410 = inv['aero_410'] & has_snow
        corr = _snow.stability_correction(None, None, t_foliage,
                                          st['t_canopy'], None, None,
                                          inv=inv['stab'])
        ra_410 = torch.where(inv['wind_pos'], st['ra1'] / corr,
                             torch.full_like(corr, HUGE_RESIST))
        ra_snow = torch.where(use_410, ra_410, ra_snow)
    es = _snow.svp(t_foliage)
    vmf = (inv['dens_eps']
           * (st['eact_air'] - es) / ra_snow / _snow.CONST_RHOFW)
    vmf = torch.where(inv['vpd_zero'] & (vmf < 0.0), zero, vmf)
    ls = _snow.calc_latent_heat_of_sublimation(t_foliage)
    latent_sub_s = ls * vmf * _snow.CONST_RHOFW
    # --- no intercepted snow: canopy evaporation at Tfoliage -----------
    # :229-238, no intercepted snow: AR_406_FULL and AR_410 keep Ra[1],
    # AR_406 and AR_406_LS take Ra[0], and the evaporation and the
    # sensible heat both use it.
    if evap_of is None:
        evap = zero
    else:
        evap = evap_of(t_foliage, net_rad, st['net_short_over'])   # m/s
    latent_free = st['lv'] * evap * _snow.CONST_RHOFW
    # :225-227 again: the resistance the sensible heat sees is the
    # vapour flux's, divided back by ten under AR_406 exactly. Under
    # AR_410 it is the same stability-corrected resistance (no
    # divide-back: :205-207 is AR_406 alone).
    ra_used1 = inv['ra_used1']
    if use_410 is not None:
        ra_used1 = torch.where(use_410, ra_410, ra_used1)
    latent = torch.where(has_snow, zero, latent_free)
    latent_sub = torch.where(has_snow, latent_sub_s, zero)
    vmf = torch.where(has_snow, vmf, zero)
    sensible = _snow.calc_sensible_heat(st['air_dens'], st['t_canopy'],
                                        t_foliage, ra_used1)
    advected = inv['advected']
    rest = sensible + latent + latent_sub + net_rad + advected
    refreeze = inv['refreeze']
    melting = has_snow & (t_foliage == 0.0) & (rest > -refreeze)
    refreeze = torch.where(melting, -rest, refreeze)
    rest = torch.where(melting, zero, torch.where(has_snow, rest + refreeze, rest))
    if return_terms:
        return rest, dict(refreeze_energy=refreeze, vapor_mass_flux=vmf,
                          ra_used1=ra_used1,
                          latent=latent, latent_sub=latent_sub,
                          sensible=sensible, advected=advected,
                          net_radiation=net_rad, net_long_over=net_long_over,
                          long_over_out=long_over_out, evap=evap)
    return rest


#: canopy_residual as one fused region on the CUDA graph path (core.fusion)
_residual_fused = fusion.region(canopy_residual)


def _residual_trial(t, st, evap_of):
    """canopy_residual at a trial temperature, fused once `st` holds its
    temperature-free terms (formed eagerly at the first evaluation)."""
    if st.get('_invariant') is None:
        return canopy_residual(t, st, evap_of)
    return _residual_fused(t, st, evap_of)


#: Tfoliage's search: 'brent' is root_brent.c's (VIC's root to the rounding
#: of the residual), 'illinois' the tighter Illinois solve (up to
#: ROOT_BRENT_T from VIC's).
CANOPY_SOLVER = 'brent'
#: how many widenings the eager solves needed (tries -> calls), for
#: choosing graph-safe counts; maxtries itself counts the calls that ran out
WIDEN_TRIES = {}


@fusion.opaque
def _widen_and_solve(f, lo, hi, t_old, tol, tstep=10.0, maxtries=5,
                     args=()):
    """root_brent.c's bracket widening on fa*fb >= 0, then the implicit
    solver; a bracket that never holds a sign change falls back to `t_old`
    (TFALLBACK, initialize_options.c:71) and is flagged.

    The state goes in `args`, not in a closure over `f`, so that if
    implicit._local_dfdt ever falls back to reverse mode it can detach it:
    a closure's tensors it cannot reach, and in a run with gradients they are
    connected to every earlier step. (`evap_of` is itself such a closure;
    forward mode is what makes that harmless.) See implicit._local_dfdt."""
    from . import implicit
    with torch.no_grad():
        flo, fhi = f(lo, *args), f(hi, *args)
        if graphsafe.ACTIVE:
            lo, hi, flo, fhi = graphsafe.widen(f, lo, hi, flo, fhi, args,
                                               tstep, maxtries)
        else:
            for _ in range(maxtries):
                bad = (flo * fhi) >= 0.0
                if not bool(bad.any()):
                    WIDEN_TRIES[_] = WIDEN_TRIES.get(_, 0) + 1
                    break
                lo = torch.where(bad, lo - tstep, lo)
                hi = torch.where(bad, hi + tstep, hi)
                flo = torch.where(bad, f(lo, *args), flo)
                fhi = torch.where(bad, f(hi, *args), fhi)
        failed = _bracket_failed(flo, fhi)
    if CANOPY_SOLVER == 'brent':
        # snow_intercept.c 5.1.0:318-338: root_brent's own iteration and
        # stop (implicit.root_brent) from the widened bracket, so Tfoliage
        # is VIC's to the rounding of the residual; the gradient from the
        # implicit-function step, added and taken away again. A failed
        # element is solved on a dummy bracket and replaced below.
        with torch.no_grad():
            lo_, hi_, flo_, fhi_ = _dummy_bracket(failed, lo, hi, flo, fhi)
            r0 = implicit.root_brent(f, lo_, hi_, flo_, fhi_,
                                     args=args, t_tol=tol)
        if torch.is_grad_enabled():
            rg = implicit.root_with_grad(f, r0, args=args)
            root = r0 + (rg - rg.detach())
        else:
            root = r0
    else:
        root = implicit.solve(f, lo, hi, args=args, tol=tol)
    return _root_or_fallback(failed, root, t_old)


def _bracket_failed_(flo, fhi):
    return ((flo * fhi) >= 0.0) | ~torch.isfinite(flo) | ~torch.isfinite(fhi)


def _dummy_bracket_(failed, lo, hi, flo, fhi):
    """A failed element's dummy bracket (-1, 0) and residuals (-1, 1)."""
    one = torch.ones_like(flo)
    return (torch.where(failed, -one, lo),
            torch.where(failed, torch.zeros_like(hi), hi),
            torch.where(failed, -one, flo),
            torch.where(failed, one, fhi))


def _root_or_fallback_(failed, root, t_old):
    failed = failed | ~torch.isfinite(root)
    return torch.where(failed, t_old, root), failed


# fused regions on the CUDA graph path (core.fusion), around the solver
_bracket_failed = fusion.region(_bracket_failed_)
_dummy_bracket = fusion.region(_dummy_bracket_)
_root_or_fallback = fusion.region(_root_or_fallback_)


def canopy_step(state, forcing, evap_of=None, snow_dt=SNOW_DT,
                vic_canopy_rounding=False):
    """snow_intercept.c for a set of overstory tiles, one step.

    `state`: int_snow, int_rain (m, per canopy area), t_foliage (C),
    temp_int_storage (m). `forcing`: rainfall, snowfall (m per step, pixel
    depth), lai, max_int (m), wind1 (U[1]), ra0, ra1, air_dens, eact_air,
    press, lv, t_canopy, vpd, long_over_in, long_under_out, short_over_in,
    bare_albedo, dt, aero_406, aero_406_full (bools), aero_410 (bool) with
    ref_height1, displacement1, roughness1. `evap_of` as in
    canopy_residual.

    Returns (new_state, out) with out: rain_through, snow_through (m,
    pixel depth, :540-548, drip and released mass included), vapor_flux
    (m per step, the C's sign flipped at :550: positive is loss), latent,
    latent_sub, sensible, advected, net_long_over, net_short_over,
    long_over_out (the ground's LongUnderIn), albedo_over,
    canopy_refreeze (the C's MeltEnergy = RefreezeEnergy / Dt, :555),
    melt_energy (what the melt block accumulated), fallback.

    Sequence: (a)-(b) intercept at the old foliage temperature; the
    canopy's albedo is the new-snow albedo with snow held or falling and
    the bare albedo otherwise (:292-294, :331-333); with snow the balance
    is tried at 0 C first and, if not closed, solved on [Told - SNOW_DT, 0]
    (:296-329), without snow on [Told - SNOW_DT, Told + SNOW_DT]
    (:334-336); the terms are read once more at the root (:376-380);
    (d) melt_and_release; (e) the closing lines.

    `vic_canopy_rounding` (True) selects snow_intercept.c's own division
    order for the melt over the established product division. It
    defaults off (False).
    """
    from . import snow as _snow
    f_ = forcing
    int_snow0 = state['int_snow']
    t_old = state['t_foliage']
    int_snow, int_rain, snow_through, rain_through, _ = intercept(
        t_old, f_['lai'], f_['snowfall'], f_['rainfall'], int_snow0,
        state['int_rain'], f_['wind1'], torch.ones_like(t_old), f_['max_int'])
    # :285-289: a canopy with nothing on it takes the canopy air temperature
    t_old = torch.where(int_rain + int_snow < DBL_EPSILON, f_['t_canopy'], t_old)
    has_snow = (int_snow > 0.0) | (f_['snowfall'] > 0.0)
    albedo_over = torch.where(has_snow,
                              f_.get('new_snow_albedo',
                                     torch.full_like(t_old, SNOW_NEW_SNOW_ALB)),
                              f_['bare_albedo'])
    st = dict(has_snow=has_snow,
              net_short_over=(1.0 - albedo_over) * f_['short_over_in'],
              long_over_in=f_['long_over_in'],
              long_under_out=f_['long_under_out'], air_dens=f_['air_dens'],
              eact_air=f_['eact_air'], press=f_['press'], ra0=f_['ra0'],
              ra1=f_['ra1'], t_canopy=f_['t_canopy'], vpd=f_['vpd'],
              rainfall=f_['rainfall'],
              # snow_intercept.c 5.1.0:129, :289, :330 hand
              # func_canopy_energy_bal IntRainOrg, the liquid store as the
              # step opened (per plant), for its `IntRain`: RefreezeEnergy
              # (func_canopy_energy_bal.c 5.1.0:257) is the opening store's
              # latent heat, not the store after this step's rain was
              # intercepted. With the latter, rain caught by a cold snowy
              # canopy froze at once; in VIC it stays liquid that step.
              int_rain=state['int_rain'], dt=f_['dt'],
              lv=f_['lv'], aero_406=f_['aero_406'],
              aero_406_full=f_['aero_406_full'],
              aero_406_exact=f_.get(
                  'aero_406_exact',
                  torch.zeros_like(f_['aero_406'])),
              aero_410=f_.get('aero_410',
                              torch.zeros_like(f_['aero_406'])),
              aero_410_on=f_.get('aero_410_on'),
              wind1=f_.get('wind1'), ref_height1=f_.get('ref_height1'),
              displacement1=f_.get('displacement1'),
              roughness1=f_.get('roughness1'))
    # canopy_residual may keep its temperature-free terms in `st`: nothing
    # in it changes between the evaluations below
    st['_invariant'] = None
    f = _residual_trial
    zero = torch.zeros_like(t_old)
    with torch.no_grad():
        q0 = f(zero, st, evap_of)
    melting = has_snow & (q0 == 0.0)
    lo = torch.where(has_snow,
                     torch.where(t_old <= 0.0, t_old - snow_dt,
                                 torch.full_like(t_old, -snow_dt)),
                     t_old - snow_dt)
    hi = torch.where(has_snow, zero, t_old + snow_dt)
    from . import implicit
    root, failed = _widen_and_solve(f, lo, hi, t_old, implicit.ROOT_BRENT_T,
                                    args=(st, evap_of))
    t_new = torch.where(melting, zero, root)
    failed = failed & ~melting
    _, terms = canopy_residual(t_new, st, evap_of, return_terms=True)
    vmf_step = terms['vapor_mass_flux'] * f_['dt']                    # :430
    # snow_intercept.c 5.1.0:539 reports MeltEnergy = RefreezeEnergy / Dt,
    # the balance's own W m-2 (:400 multiplied it by Dt), except where the
    # frozen canopy's energy refroze only part of the liquid and :492 set
    # it to 0. (This was the balance divided by Dt once more.)
    _rf_j = terms['refreeze_energy'] * f_['dt']
    canopy_refreeze = torch.where(
        (t_new != 0.0) & (-_rf_j > -int_rain * _LATICE()),
        torch.zeros_like(_rf_j), terms['refreeze_energy'])
    (int_snow, int_rain, temp_int, vmf_applied, melt_energy, drip,
     released) = melt_and_release(
        # snow_intercept.c:419, `RefreezeEnergy *= Dt`: everything from
        # there on treats it as an energy per unit area over the step,
        # not a flux. Passing the flux made the canopy melt 3600 times
        # too slowly at an hourly step -- it could not shed intercepted
        # snow at all, and so kept a 0.85 albedo through a whole winter.
        t_new, terms['refreeze_energy'] * f_['dt'], vmf_step, int_snow,
        int_rain,
        state['temp_int_storage'], snow_through, int_snow0, f_['max_int'],
        f_['dt'], vic_canopy_rounding=vic_canopy_rounding)
    # :540-548, F = 1
    spill = (int_snow == 0.0) & (int_rain > f_['max_int'])
    # snow_intercept.c 5.1.0:524, RainThroughFall += *IntRain - MaxInt
    rain_through = torch.where(spill,
                               rain_through + (int_rain - f_['max_int']),
                               rain_through)
    int_rain = torch.where(spill, f_['max_int'], int_rain)
    new_state = dict(int_snow=int_snow, int_rain=int_rain, t_foliage=t_new,
                     temp_int_storage=temp_int)
    out = dict(rain_through=rain_through + drip,
               snow_through=snow_through + released,
               vapor_flux=-vmf_applied,
               latent=terms['latent'], latent_sub=terms['latent_sub'],
               sensible=terms['sensible'], advected=terms['advected'],
               net_long_over=terms['net_long_over'],
               net_short_over=st['net_short_over'],
               long_over_out=terms['long_over_out'], albedo_over=albedo_over,
               canopy_refreeze=canopy_refreeze,
               # Ra_used[1] as func_canopy_energy_bal.c leaves it. It is
               # what surface_fluxes.c:243 hands put_data as
               # cell.aero_resist[1], which is the resistance an overstory
               # tile reports (put_data.c:676-685) and the one
               # compute_pot_evap.c:71 uses for its vegetated Penman.
               ra_used1=terms['ra_used1'],
               melt_energy=melt_energy, fallback=failed,
               # which branch of func_canopy_energy_bal.c ran (snow held or
               # falling, :179), and its NetRadiation at the root, for a
               # caller that takes the liquid branch's canopy_evap mass
               has_snow=has_snow, net_radiation=terms['net_radiation'])
    return new_state, out


# the step as fused regions on the CUDA graph path (core.fusion), the
# foliage temperature's root find between them
canopy_step = fusion.region(canopy_step)
