"""The surface energy balance, for the configuration VIC actually runs here.

VIC's full energy balance is a nest of iterations: an overstory/understory
coupling loop, a canopy-air temperature solve, a snowpack solve, a foliage
solve, and a finite-difference soil thermal profile with its own Newton
iteration. Almost none of it runs at Walnut Gulch, and reading which switches
turn which parts off is what makes this module small:

    CLOSE_ENERGY false   the over/under loop runs exactly once
                         (surface_fluxes.c:228-233 sets MAX_ITER to 0)
    FROZEN_SOIL  false   fusion = 0, ice = 0, no node profile
    QUICK_FLUX   true    T1 from a closed form (estimate_T1.c), no
                         finite-difference soil solve
    overstory    false   on every tile with cover, so no canopy solve, no
                         snow interception, no atmosphere solve
    snow         absent  so INCLUDE_SNOW is false and the pack terms vanish

What is left is **one root find in one unknown** -- the surface temperature
Ts -- over a residual that is smooth and monotone in Ts, because both the
outgoing longwave and the sensible heat decrease as Ts rises. That residual is
transcribed below from func_surf_energy_bal.c, term by term, with the line
cited at each.

Two things a port gets wrong here if it is not careful, and both are
measured facts from the reference run rather than cautions:

* The initial bracket is too narrow 86 per cent of the time. VIC starts at
  0.5 (T[0] + Tair) +/- SURF_DT with SURF_DT = 1.0 and widens by
  ROOT_BRENT_TSTEP = 10 up to ROOT_BRENT_MAXTRIES = 5 times
  (root_brent.c:169-179). The widening is the common path, not the
  exception, and a solver that bisects on +/-1 fails on most steps.
* Transpiration is driven by the *neutral* aerodynamic resistance and the
  sensible heat by the *stability-corrected* one, in the same step for the
  same tile. func_surf_energy_bal.c:757 hands canopy_evap `Ra_veg[1]`, which
  the no-overstory branch sets equal to the neutral `Ra[0]` and which
  StabilityCorrection never touches; only `Ra_veg[0]` is corrected (:674-684)
  and only it reaches calc_sensible_heat (:847). Using the corrected one for
  both is the obvious improvement and it is not VIC.

Sign convention, because it is easy to get backwards and the closure check
cannot tell: inside VIC every flux is positive *toward* the surface and the
residual is `NetBareRad + sensible + latent = 0`. put_data.c then writes the
outputs negated -- `out[OUT_LATENT] -= latent` (:913) and likewise sensible
(:919), grnd_flux (:922), deltaH (:925) -- so the *reported* fluxes are
positive away from the surface, and the reported ground flux is positive
into the ground. Net radiation is reported as it is, positive downward. The
energy-error check uses the internal signs, so it reads zero whichever way
round the outputs are written. This module returns the reported convention.

Beyond that configuration the same residual also carries, when the caller
passes them: snow on the surface (INCLUDE_SNOW, the `pack` argument), bare soil
under a partial canopy (fcanopy < 1, `bare`) and the finite-difference soil
profile (QUICK_FLUX false, `fd`). An overstory is not solved in this module:
torchvic.model.basin runs the canopy stage (torchvic.core.canopy) and the
overstory aerodynamic resistances (torchvic.core.vegetation) for such a tile,
and this balance then solves the ground under it.
"""
import math

import torch
import torch.autograd.forward_ad as _fwAD

from . import atmos, frozen, graphsafe, snow, soiltemp, xmath
from . import implicit as _implicit
from .implicit import solve as implicit_solve
from .implicit import root_brent as _root_brent, root_with_grad
from .snow import CONST_TKFRZ
from .vegetation import CONST_KARMAN

# ---------------------------------------------------------------------------
# VIC's parameters and constants, cited. initialize_parameters.c unless noted.
# ---------------------------------------------------------------------------
#: Half-width of the first bracket around 0.5 (T[0] + Tair).  :202
SURF_DT = 1.0
#: Widening applied to both ends when the bracket holds no sign change.  :228
ROOT_BRENT_TSTEP = 10.0
#: How many widenings are tried before VIC gives up.  :226
ROOT_BRENT_MAXTRIES = 5
#: VIC's own convergence tolerance for root_brent, K.  :229
ROOT_BRENT_T = 1.0e-7
#: Half-width of the canopy-air-temperature bracket, C.  :186
CANOPY_DT = 1.0
#: Ground and overstory loop tolerances, W/m2 and C.  :190-191
TOL_GRND = 0.001
TOL_OVER = 0.001
#: Passes of the ground/canopy iteration with CLOSE_ENERGY.  :202
#: vic_init.c:1556 resets it to 0 without the option.
MAX_ITER_GRND_CANOPY = 10
#: The surface solve's search: 'brent' is root_brent.c's own iteration and
#: stopping rule (implicit.root_brent), so the surface temperature is VIC's
#: to the rounding of the residual; 'illinois' solves the balance to ~1e-13
#: W m-2, tighter than VIC (whose root sits up to ROOT_BRENT_T away).
SURFACE_SOLVER = 'brent'
#: Emissivity of the snow-free ground.  :51
EMISS_GRND = 0.97
#: Height of the vegetation is displacement / this.  :67
VEG_RATIO_DH_HEIGHT = 0.67
#: Resistance used in place of a division by zero wind.  :45
HUGE_RESIST = 1.0e20
#: Critical Richardson number in StabilityCorrection.c:44.
RI_CR = 0.2
#: Density of fresh water, kg/m3, for the latent heat.  vic_physical_constants.h
CONST_RHOFW = 1000.0
#: Specific heat of air, J/kg/K.  vic_physical_constants.h
CONST_CPMAIR = 1013.0
#: Gravity, m/s2.
CONST_G = 9.80616   # vic_physical_constants.h:78 -- 9.81 was 4.5e-4 off in the correction
MM_PER_M = 1000.0


# ---------------------------------------------------------------------------
# Thermal properties of the top two soil layers.  prepare_full_energy.c and
# soil_conduction.c:704-740.
# ---------------------------------------------------------------------------
def thermal_properties(moist_mm, depth, bulk_density, soil_density, quartz,
                       organic=None, ice_mm=None, bulk_dens_min=None,
                       soil_dens_min=None):
    """Conductivity kappa (W/m/K) and heat capacity Cs (J/m3/K) per layer.

    `moist_mm` is the layer moisture in mm, [cell, layer]; the rest are the
    parameter file's own, [cell, layer]. Ice is zero with FROZEN_SOIL off.

    compute_soil_layer_thermal_properties converts the moisture to a volume
    fraction first (:722, `moist / depth / MM_PER_M`) and hands the mineral
    bulk and particle densities in separately. The image driver reads those
    straight from the file's `bulk_density` and `soil_density` variables
    (vic_init.c:557-575) and the classic driver reads them as their own
    columns, so both drivers give soil_conductivity the same two numbers and
    with organic = 0 they *are* bulk_density and soil_density.

    With ORGANIC_FRACT or BULK_DENSITY_COMB the file's mineral pair is
    `bulk_dens_min` / `soil_dens_min` and `bulk_density` / `soil_density`
    are the combined ones (ImageParams.apply_density_options); left None,
    the mineral pair is the density pair, as before.
    """
    if organic is None:
        organic = torch.zeros_like(moist_mm)
    frac = moist_mm / depth / MM_PER_M
    # soil_conduction.c:711-716: the ice is a volume fraction too, and
    # it enters the conductivity as the *unfrozen* water `moist - ice`
    # and the heat capacity as its own term. Frozen soil conducts better
    # and stores less heat, which is most of why a frozen profile runs
    # colder than an unfrozen one under the same forcing.
    ice = (torch.zeros_like(frac) if ice_mm is None
           else ice_mm / depth / MM_PER_M)
    kappa = frozen.soil_conductivity(
        frac, frac - ice,
        soil_density if soil_dens_min is None else soil_dens_min,
        bulk_density if bulk_dens_min is None else bulk_dens_min,
        quartz, soil_density, bulk_density, organic)
    cs = frozen.volumetric_heat_capacity(bulk_density / soil_density,
                                         frac - ice, ice, organic)
    return kappa, cs


# ---------------------------------------------------------------------------
# The closed-form pieces of the QUICK_FLUX profile.
# ---------------------------------------------------------------------------
def estimate_T1(ts, t1_old, t2, d1, d2, kappa1, kappa2, cs2, dp, dt):
    """Temperature at the base of the top layer.  estimate_T1.c:52-58.

    Affine in `ts`, which is what makes the whole residual monotone. `t2` is
    not the temperature of layer 2, nor of node 2, nor at depth dp -- VIC
    says so in a comment at func_surf_energy_bal.c:401-403 -- it is the
    constant deep temperature the profile is asymptotic to, `avg_temp` from
    the parameter file. And `d2` is depth[0] again, not depth[1]
    (calc_surf_energy_bal.c:198-199, with VIC's own comment). This is the
    root-zones-as-layers mistake waiting to happen a second time.
    """
    c1 = cs2 * dp / d2 * (1.0 - torch.exp(-d2 / dp))
    c2 = -(1.0 - torch.exp(d1 / dp)) * torch.exp(-d2 / dp)
    c3 = kappa1 / d1 - kappa2 / d1 + kappa2 / d1 * torch.exp(-d1 / dp)
    num = (kappa1 / 2.0 / d1 / d2 * ts + c1 / dt * t1_old
           + (2.0 * c2 - 1.0 + torch.exp(-d1 / dp))
           * kappa2 / 2.0 / d1 / d2 * t2)
    den = c1 / dt + kappa2 / d1 / d2 * c2 + c3 / 2.0 / d2
    return num / den


def layer_temperatures_quick_flux(tsurf, t1, tp, depth, dp, lsum=None):
    """Layer-average soil temperatures.  soil_conduction.c:598-628.

    Linear in the top layer, exponential below, anchored at the deep
    temperature `tp`. `depth` is [.., layer]. Verified by the reference
    audit against VIC's own sub-daily output to 0.0000 over 3720 cell-steps.
    `lsum`, if given, is torch.cumsum(depth, -1) made once by the caller
    (the depths do not change; on a GPU the scan of a 3-long row is a slow
    kernel to run every step).
    """
    if lsum is None:
        lsum = torch.cumsum(depth, dim=-1)                   # Lsum[1..N]
    lsum0 = torch.cat([torch.zeros_like(lsum[..., :1]), lsum], dim=-1)
    out = [0.5 * (tsurf + t1)]
    for l in range(1, depth.shape[-1]):
        top, bot = lsum0[..., l], lsum0[..., l + 1]
        out.append(tp - dp / depth[..., l] * (t1 - tp)
                   * (torch.exp(-(bot - lsum0[..., 1]) / dp)
                      - torch.exp(-(top - lsum0[..., 1]) / dp)))
    return torch.stack(out, dim=-1)


def stability_correction(z, d, t_surf, t_air, wind, z0):
    """Factor the resistance is *divided* by.  StabilityCorrection.c:34-80.

    Richardson-number form: bulk Ri from the surface-air temperature
    difference, capped at a limit that keeps the stable branch positive,
    floored at -0.5 on the unstable side. Neutral when the temperatures are
    equal, exactly, which the C tests with `!=`.
    """
    tk_air = t_air + CONST_TKFRZ
    tk_surf = t_surf + CONST_TKFRZ
    tmean = 0.5 * (tk_air + tk_surf)
    # :34-35 divides by (TMean * Wind) * Wind, left to right; TMean * Wind^2
    # rounds differently and moved thin-pack roots by an ulp (himal day 0).
    w = torch.clamp(wind, min=1e-6)
    ri = CONST_G * (t_air - t_surf) * (z - d) / (tmean * w * w)
    ri_limit = tk_air / (tmean * (torch.log((z - d) / z0) + 5.0))
    ri = torch.minimum(ri, ri_limit)
    stable = (1.0 - ri / RI_CR) * (1.0 - ri / RI_CR)  # :47, a product
    # The unstable form is selected only for Ri < 0, so clamping its argument
    # at zero changes nothing where it is used. Where it is NOT used -- the
    # stable side, Ri > 1/16 -- sqrt(1 - 16 Ri) is the square root of a
    # negative number. The forward `where` never sees that; autograd does,
    # because the gradient through an unselected branch is 0 * d(NaN) = NaN,
    # and it reached the implicit-function step as a NaN derivative on every
    # zero-cover tile of a warm afternoon. VIC has an `if`; this needs both
    # branches finite.
    unstable = xmath.sqrt(1.0 - 16.0 * torch.clamp(ri, min=-0.5, max=0.0))
    corr = torch.where(ri > 0.0, stable, unstable)
    return torch.where(t_surf != t_air, corr, torch.ones_like(corr))


# ---------------------------------------------------------------------------
# The residual and its solution.
# ---------------------------------------------------------------------------
def _residual(ts, tair, t1_old, ts_old, t2, d1, d2, kappa1, kappa2, cs1, cs2,
              dp, dt, surf_atten, net_short, long_in, ra_neutral, ref_height,
              displacement, roughness, wind, air_dens, le, evap_of,
              pack=None, bare_frac=None, frozen_soil=None, bare=None,
              fd=None, grnd_flux_type='GF_410'):
    """func_surf_energy_bal.c for the no-overstory case, with or without a
    thin pack folded in (INCLUDE_SNOW).

    Returns the residual and every term, so the solved state can be read
    back with one more call rather than recomputed by hand. All in W/m2 with
    VIC's internal sign (positive toward the surface).

    `pack`, when given, carries per-tile tensors for the cell's pack:
    `thin` (bool: surface_fluxes.c:585 INCLUDE_SNOW), `swq`, `surf_water`,
    `old_t_surf`, `density`, `net_short_snow` = (1 - snow albedo) * SW,
    `eact_air`, `press`, `vpd`. Where `thin` is false every term reduces
    to the snow-free case.

    `bare`, when given (the VIC 5.1.0 core with a canopy fraction), carries
    per-tile `fc`, the bare soil's neutral resistance `ra` and the
    reference height, displacement, roughness and wind its stability
    correction takes (func_surf_energy_bal.c 5.1.0:677-718): Ra_used[0] is
    then the conductance average of Ra_veg[0] and that Ra_bare wherever
    fcanopy < 1, it is what the sensible heat, a thin pack and arno_evap
    see, and `evap_of` is handed it as `ra_used`.

    `fd`, when given, selects the finite-difference soil temperature
    profile (QUICK_FLUX FALSE, :404-445) in place of the two-node
    closed form: the profile is solved at this trial Ts and T1, the
    ground flux, the storage change and the fusion are read from it
    (torchvic.core.soiltemp). Everything past those four terms is the
    same code on both paths.
    """
    thin = None if pack is None else pack['thin']
    # NetShortBare as the caller formed it (calc_surf_energy_bal.c
    # 5.1.0:211-213): zero under a full pack, the snow-free part's under a
    # partial one (SPATIAL_SNOW)
    net_short_bare = net_short
    if thin is not None:
        # calc_surf_energy_bal.c:231-238 with coverage 1: the snow-free
        # area sees no radiation, the pack's NetShortSnow stands in, and the
        # longwave is LongSnowIn - LongBareOut with emissivity 1 (:213).
        net_short = torch.where(thin, pack['net_short_snow'], net_short)
    att = surf_atten
    if pack is not None and 'coverage' in pack:
        att = pack['coverage'] + (1.0 - pack['coverage']) * surf_atten
    t_new = None
    if fd is None:
        t1 = estimate_T1(ts, t1_old, t2, d1, d2, kappa1, kappa2, cs2, dp, dt)
        # Ground heat flux, GF_410 form (:419-429), snow_coverage = 0.
        # func_surf_energy_bal.c 5.1.0:394-401: the factor is snow_coverage +
        # (1 - snow_coverage) surf_atten, so a covered tile's soil sees the
        # whole flux whatever its canopy. Under GF_406 (:385-391) the
        # flux is the surface term alone: no kappa2/D2 deep term, no
        # division by 2.
        if grnd_flux_type == 'GF_406':
            grnd = att * (kappa1 / d1 * (t1 - ts))
        else:
            grnd = att * (kappa1 / d1 * (t1 - ts)
                          + kappa2 / d2 * (1.0 - torch.exp(-d1 / dp))
                          * (t2 - t1)) / 2.0
        # Change in heat storage between the surface and the layer base
        # (:531).
        delta_h = cs1 * ((ts_old + t1_old) - (ts + t1)) * d1 / dt / 2.0
    else:
        # :404-445: the profile at this trial surface temperature, and
        # T1, the ground flux, the storage change and the fusion from
        # it (:447-613).
        t_new = soiltemp.solve_profile(
            ts, fd['t_node'], fd['moist_node'], fd['ice_node'],
            fd['kappa_node'], fd['cs_node'], fd['porosity_node'],
            fd['bubble_node'], fd['expt_node'], fd['alpha'], fd['beta'],
            fd['gamma'], fd['zsum'], dt, fd['dp'], fd['bulk_dens_min'],
            fd['soil_dens_min'], fd['quartz'], fd['bulk_density'],
            fd['soil_density'], fd['organic'], fd['depth'], fd['nlayers'],
            fd['nnodes'], fd['implicit'], fd['noflux'], fd['exp_trans'],
            fd['fs_active'], fd['frozen_soil'],
            tmp_nnodes=fd.get('tmp_nnodes'), cols=fd.get('solve_cols'),
            cache=fd.setdefault('_profile_cache', {}))
        t1, grnd, delta_h, fusion = soiltemp.profile_ground_terms(
            ts, t_new, fd['t_node'], t1_old, ts_old, fd['ice0'],
            fd['moist0'], fd['porosity0'], fd['bubble0'], fd['expt0'],
            kappa1, kappa2, cs1, d1, dt, fd['zsum'], att,
            fd['exp_trans'], fd['fs_active'], fd['frozen_soil'],
            grnd_flux_type=grnd_flux_type)
    # Longwave out and net (:658-662).
    long_out = snow.calc_outgoing_longwave(ts + CONST_TKFRZ, EMISS_GRND)
    # func_surf_energy_bal.c:662: NetLongBare = LongBareIn - (1 - coverage) *
    # LongBareOut, and calc_surf_energy_bal.c:232 has LongBareIn itself as
    # (1 - coverage) * LongUnderIn. Under a full pack both are zero: the
    # snow-free surface has no sky above it to trade longwave with. Scaling
    # only the incoming -- which is what this did first -- left the surface
    # radiating 300 W/m2 into nothing and put OUT_LWNET at -231 against
    # VIC's -26 on 90 winter days of FindleyLake.
    bf = 1.0 if bare_frac is None else bare_frac
    net_long = bf * (long_in - long_out)
    if thin is not None:
        # :633-636, :662 with INCLUDE_SNOW: NetLongBare = LongBareIn - (1 -
        # c) LongBareOut with LongBareIn = (1 - c) LongUnderIn
        # (calc_surf_energy_bal.c 5.1.0:214) and NetLongSnow = LongSnowIn -
        # c LongBareOut with LongSnowIn = c LongUnderIn (:218); together the
        # whole of LongUnderIn - LongBareOut whatever the cover, summed as
        # the C sums them (:599, NetLongUnder = NetLongBare + NetLongSnow).
        _c = pack['coverage'] if 'coverage' in pack else torch.ones_like(ts)
        net_long_bare = (1.0 - _c) * long_in - (1.0 - _c) * long_out
        net_long_snow = _c * long_in - _c * long_out
        net_long = torch.where(thin, net_long_bare + net_long_snow,
                               net_long)
    # Net radiation available to the snow-free surface (:663). Under a
    # thin pack the snow-free area is nothing: NetShortBare and NetLongBare
    # are 0 and NetBareRad is the soil terms alone, which is what canopy_evap
    # is then handed (:757), with NetShortBare = 0 for calc_rc.
    # func_surf_energy_bal.c:561-574, the latent heat of fusion released
    # or absorbed by the top layer as its ice content changes over the
    # step. The ice is evaluated at the mean of the surface and the base
    # of the layer, (TMean + T1)/2, so it is a function of the
    # temperature being solved for and belongs inside the residual.
    # :664 adds it to NetBareRad. On the finite-difference path the
    # fusion came with the profile above (:535-613).
    if fd is None and frozen_soil is None:
        fusion = torch.zeros_like(ts)
    elif fd is None:
        fz = frozen_soil
        t_mid = 0.5 * (ts + t1)
        ice_now = torch.clamp(
            fz['moist'] - frozen.maximum_unfrozen_water(
                t_mid, fz['max_moist'], fz['bubble'], fz['expt']),
            min=0.0)
        fusion = -(snow.CONST_RHOICE * snow.CONST_LATICE
                   * (fz['ice0'] - ice_now) * fz['d1'] / dt)
        fusion = torch.where(fz['active'], fusion,
                             torch.zeros_like(fusion))
    net_bare_rad_full = net_short + net_long + grnd + delta_h + fusion
    net_bare_rad = (net_bare_rad_full if thin is None else
                    torch.where(thin, (((net_short_bare + net_long_bare)
                                        + grnd) + delta_h) + fusion,
                                net_bare_rad_full))
    # Stability-corrected resistance for sensible heat only (:674-684).
    corr = stability_correction(ref_height, displacement, ts, tair, wind,
                                roughness)
    ra_used = torch.where(wind > 0.0, ra_neutral / corr,
                          torch.full_like(ra_neutral, HUGE_RESIST))
    ra_veg0 = ra_used
    if bare is not None:
        # func_surf_energy_bal.c 5.1.0:677-718 (5.0.1:703-745): below a
        # closed canopy the resistance is the conductance average of the
        # understory's Ra_veg[0] and the bare soil's, CalcAerodynamic's
        # no-overstory form over the soil roughness at 2 m, itself
        # stability-corrected at this surface temperature; 0 where either
        # is 0. Every division sees a positive denominator in both
        # branches of every where.
        fc = bare['fc']
        corr_b = stability_correction(bare['ref_height'],
                                      bare['displacement'], ts, tair,
                                      bare['wind'], bare['roughness'])
        ra_b = bare['ra'] / corr_b
        pos = (ra_used > 0.0) & (ra_b > 0.0)
        one = torch.ones_like(ra_used)
        ga = (fc * (1.0 / torch.where(ra_used > 0.0, ra_used, one))
              + (1.0 - fc) * (1.0 / torch.where(ra_b > 0.0, ra_b, one)))
        blended = torch.where(pos, 1.0 / torch.where(pos, ga, one),
                              torch.zeros_like(ga))
        ra_used = torch.where(fc < 1.0, blended, ra_used)
    # Evaporation from the canopy, driven by NetBareRad and the *neutral*
    # resistance (:757-761). `evap_of` returns mm per step; canopy_evap.c:160
    # accumulates m/s, and :816 wants m/s.
    kw = {} if bare is None else dict(ra_used=ra_used)
    if thin is None:
        evap_mm = evap_of(net_bare_rad, ra_neutral, **kw)
    else:
        evap_mm = evap_of(net_bare_rad, ra_neutral,
                          torch.where(thin, net_short_bare, net_short), **kw)
    if isinstance(evap_mm, tuple):
        # the caller's Evap in m/s as func_surf_energy_bal.c 5.1.0:734-751
        # forms it, beside the mm the water balance takes
        evap_mm, evap_ms = evap_mm
    else:
        evap_ms = evap_mm / MM_PER_M / dt
    # :838 and :845-851 scale both turbulent terms by the snow-free
    # fraction as well, for the same reason.
    latent = -CONST_RHOFW * le * evap_ms * bf                           # :816
    sensible = snow.calc_sensible_heat(air_dens, tair, ts, ra_used)
    if thin is None:
        sensible = sensible * bf                                         # :847
    else:
        # :838-847: an included pack's tile is not scaled by the bare part
        sensible = torch.where(thin, sensible, sensible * bf)
    zero = torch.zeros_like(ts)
    latent_sub, vapor_flux, delta_cc, refreeze = zero, zero, zero, zero
    if pack is not None and 'coverage' not in pack:
        pack = None                    # a thin-pack dict without the cover
    # calc_surf_energy_bal.c:379-384 and func_surf_energy_bal.c:868: with a
    # pack over the soil and the pack solving its own balance, the surface
    # exchanges heat with it by conduction,
    #     snow_flux = kappa_snow * (Tsnow_surf - Tsurf),
    # and that term enters the residual weighted by the cover. Without it a
    # fully covered surface has only the ground flux and its own heat
    # storage to balance and drifts warm: 2.3 C against VIC's -2.3 on 90
    # winter days of FindleyLake.
    # :386-389 zeroes the flux for a pack folded into this solve -- the pack
    # then *has* the surface's temperature and cannot conduct against it.
    snow_flux = zero if pack is None else torch.where(
        (pack['coverage'] > 0.0) & ~thin,
        pack['kappa_snow'] * (pack['t_snow_surf'] - ts), zero)
    melt_e = zero if pack is None else pack.get('melt_energy', zero)
    # :853-857 without INCLUDE_SNOW: + Advection - deltaCC as the energy
    # struct holds them, which a snow sub-step after a thin-pack one still
    # carries (surface_fluxes.c:225-226, :496, :578)
    stale = None
    if pack is not None and pack.get('stale_adv') is not None:
        # func_surf_energy_bal.c 5.1.0:828-829, `+ Advection - *deltaCC`:
        # added, then subtracted, one at a time (their difference first
        # rounds an ulp apart)
        stale = (pack['stale_adv'], pack['stale_dcc'])
    if thin is None:
        resid = net_bare_rad + sensible + latent + snow_flux + melt_e  # :859
        if stale is not None:
            resid = (resid + stale[0]) - stale[1]
    else:
        # :819-834: the thin pack's latent heat at this surface temperature,
        # added to the canopy's; its sublimation, in m per step over ice.
        l_s, ls_s, vmf, _ = snow.latent_heat_from_snow(
            air_dens, pack['eact_air'], le, pack['press'], ra_used, ts,
            pack['vpd'], zero)
        # :780-781: both weighted by the cover (1 without SPATIAL_SNOW)
        _cs = pack['coverage']
        latent = torch.where(thin, latent + l_s * _cs, latent)
        latent_sub = torch.where(thin, ls_s * _cs, zero)
        vapor_flux = torch.where(thin, vmf * dt / snow.CONST_RHOICE, zero)
        # :643-655: the pack's change in cold content at this temperature,
        # clipped at 0 C, and the latent heat of its liquid at the *snow*
        # density, which is the C's factor.
        ts0 = torch.clamp(ts, max=0.0)
        # :627-628 scale both by the cover fraction
        delta_cc = torch.where(
            thin, snow.CONST_VCPICE_WQ * (pack['swq'] - pack['surf_water'])
            * (ts0 - pack['old_t_surf']) / dt * _cs, zero)
        refreeze = torch.where(
            thin, pack['surf_water'] * snow.CONST_LATICE * pack['density']
            / dt * _cs, zero)
        # :855-866: NetBareRad + NetShortSnow + NetLongSnow + sensible +
        # latent + latent_sub + Advection - deltaCC; snow_flux and
        # melt_energy are 0 for an included pack (:386-389, surface_fluxes.c
        # :590). NetBareRad keeps the fusion term (:664).
        rad = torch.where(thin, net_short + net_long + grnd + delta_h
                          + fusion, net_bare_rad)
        # :868-876 carries snow_flux * coverage in the error whichever
        # branch a tile is in; it is zero for a thin pack by the line above.
        # func_surf_energy_bal.c:815 weights the conduction by the cover
        # fraction -- the pack's heat reaches only the covered ground -- so
        # a tile at 0.04 cover conducts 4% of kappa * dT, not all of it.
        # Without the option the cover is 0 or 1 and this changes nothing.
        # :853-857: + Advection, the rain advection snow_melt's 0 C call
        # left (surface_fluxes.c 5.1.0:593)
        adv = pack.get('advection')
        adv = zero if adv is None else torch.where(thin, adv, zero)
        # :854 melt_energy, which surface_fluxes.c 5.1.0:597 zeroes for an
        # included pack
        error = (rad + sensible + latent + latent_sub - delta_cc
                 + snow_flux * pack['coverage']
                 + adv + torch.where(thin, zero, melt_e))
        # An included pack sums in the C's order, NetBareRad first and the
        # snow terms after it (func_surf_energy_bal.c 5.1.0:810-829); folding
        # the snow longwave into rad first rounds 1 ulp off at some Brent
        # iterates, which moves the root. NetShortBare and NetLongBare are
        # +0.0 at cover 1 (calc_surf_energy_bal.c 5.1.0:211-214),
        # NetShortGrnd +0.0 (solve_snow.c 5.1.0:271, untouched for a folded
        # pack), emissivity 1 (:193); the +0.0 terms below are load-bearing.
        _cov_e = pack['coverage']
        # Under a partial pack (SPATIAL_SNOW) NetShortBare and NetLongBare
        # are the snow-free part's and NetLongSnow the covered part's.
        _nbr = ((((net_short_bare + net_long_bare) + grnd) + delta_h)
                + fusion)
        thin_err = (((((((((_nbr + 0.0) + net_short) + net_long_snow)
                          + sensible) + (latent + latent_sub))
                        + snow_flux * _cov_e)
                       + torch.where(thin, zero, melt_e)) + adv)
                    - delta_cc)
        error = torch.where(thin, thin_err, error)
        if stale is not None:
            error = torch.where(thin, error,
                                (error + stale[0]) - stale[1])
        # :868-876: at 0 C with a surplus the surplus is handed back as
        # refreeze energy and the residual is 0; otherwise the liquid's
        # latent heat is part of the balance.
        melting = thin & (ts == 0.0) & (error > -refreeze)
        refreeze = torch.where(melting, -error, refreeze)
        resid = torch.where(melting, zero,
                            torch.where(thin, error + refreeze, error))
    return resid, dict(t1=t1, grnd=grnd, delta_h=delta_h, net_long=net_long,
                       fusion=fusion, t_new=t_new,
                       long_out=long_out, net_bare_rad=net_bare_rad,
                       ra_used=ra_used, ra_veg0=ra_veg0,
                       latent=latent, sensible=sensible,
                       evap_mm=evap_mm, latent_sub=latent_sub,
                       vapor_flux=vapor_flux, delta_cc=delta_cc,
                       refreeze_energy=refreeze, snow_flux=snow_flux)


#: torch.compile the residual where no derivative passes through it. Off
#: by default: compiled arithmetic agrees with eager to ~1e-15 relative but
#: not bit for bit, and the VIC-comparison tests are bit-exact. Scripts turn
#: it on with enable_compile().
COMPILE = False
_compiled = None
_column_eager = None


#: compile the soil column's step with its gradient too. The compiled
#: backward used to return NaN for soil.expt on some batches: AOTAutograd
#: gives every output a zero cotangent, including the step's diagnostics
#: that nothing reads, and the lumped water-table curve among them has an
#: infinite local derivative, so 0 * inf = NaN. The compiled step now
#: returns its diagnostics detached (enable_compile), which leaves the
#: forward bit-identical and the gradient equal to eager's (1e-13 x max|g|
#: on 4,096 random rows, expt 3-30; the full model to 1e-16).
COLUMN_COMPILE_GRAD = True


def enable_compile(on=True):
    """Compile the surface and snow-pack residuals where no derivative
    passes through them. See _residual_fast."""
    global COMPILE, _column_eager
    COMPILE = bool(on)
    snow.COMPILE = bool(on)
    # The soil column step as well: no root find, no forward-mode tangent,
    # so it compiles WITH its gradient. On India basin 37 (daily water
    # balance, 180 days, 100 cells, gradients on) forward 11.4 -> 3.2 s and
    # backward 12.2 -> 1.1 s, gradients equal to 1.5e-16 relative.
    from . import column
    if on and _column_eager is None:
        _column_eager = column.step
        eager = _column_eager

        def _detached_diagnostics(*a, **k):
            out = eager(*a, **k)
            if len(out) < 4:              # diagnostics=False: nothing extra
                return out
            return out[:3] + ({n: (v.detach() if torch.is_tensor(v) else v)
                               for n, v in out[3].items()},) + out[4:]
        from . import fusion
        fusion.ensure_setup()
        compiled = torch.compile(_detached_diagnostics, dynamic=False)

        def step(*a, **k):
            # With gradients, compiled only when COLUMN_COMPILE_GRAD (see
            # there for the NaN the undetached diagnostics caused).
            if torch.is_grad_enabled() and not COLUMN_COMPILE_GRAD:
                return _column_eager(*a, **k)
            return compiled(*a, **k)
        column.step = step
    elif not on and _column_eager is not None:
        column.step = _column_eager
        _column_eager = None
    if on:
        # One compiled variant per month (the evaporation closure carries
        # the month index) and per cache state; torch's default of 8 was
        # reached in a 6-month window, after which every later call ran
        # eager -- silently, at the old speed.
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = max(
            _dynamo.config.cache_size_limit, 64)
        # A compile that fails (an inductor backend error on a new input
        # pattern -- seen once in a multi-basin training run: "Event device
        # type CUDA does not match blocking stream's device type CPU")
        # runs that call eager instead of killing the run. Eager and
        # compiled agree to ~1e-15 (test_compiled_solvers.py).
        _dynamo.config.suppress_errors = True
        # The partitioner's far-apart ban walks the fusible users of every
        # saved node of the joint graph, and on the unrolled column step
        # it took hours per variant (an AOT min-cut call that did not
        # return in 2 h on an H200). Off, the min cut still runs, only
        # without that heuristic.
        import torch._functorch.config as _fc
        _fc.ban_recompute_used_far_apart = False


def _residual_fast(ts, *args, grnd_flux_type='GF_410'):
    """`_residual`, compiled when that cannot touch a gradient.

    Per hourly step the surface solve evaluates the residual ~16 times --
    the bracket search and the Illinois iterations -- and each evaluation is
    several hundred tiny kernels (the whole per-tile evaporation). That was
    101 of 191 ms per step on India basin 37: the GPU idle between launches.
    Compiled, those fuse into a few kernels.

    Only where no derivative is taken: grad mode off (the bracket and the
    iterations run under no_grad; only the root matters there) AND no
    forward-mode tangent on T (implicit._local_dfdt passes a dual tensor
    under no_grad, and a compiled graph dropped its tangent: gradients came
    out 15-53% wrong when the whole residual was compiled). The evaluation
    that carries the parameter gradient stays eager, so gradients are the
    same formula as without compilation.
    """
    global _compiled
    if (COMPILE and not torch.is_grad_enabled()
            and _fwAD.unpack_dual(ts).tangent is None):
        if _compiled is None:
            from . import fusion
            fusion.ensure_setup()
            _compiled = torch.compile(_residual, dynamic=False)
        return _compiled(ts, *args, grnd_flux_type=grnd_flux_type)
    return _residual(ts, *args, grnd_flux_type=grnd_flux_type)


def solve_surface_temperature(tair, t0_old, t1_old, args, tol=ROOT_BRENT_T,
                            grnd_flux_type='GF_410'):
    """Ts such that the surface energy balance closes, with gradients.

    Bracket as VIC does (calc_surf_energy_bal.c:273-274) and widen as VIC
    does (root_brent.c:169-179), then hand the bracketed root to the
    implicit-function-theorem solver. The widening is done for every element
    at once: a per-element bracket that already holds a sign change is left
    alone, and one that does not is grown by ROOT_BRENT_TSTEP on each side,
    up to ROOT_BRENT_MAXTRIES times.

    Returns the root and a mask of elements whose bracket never held a sign
    change. VIC falls back to Ts_old on those (TFALLBACK, default true) and
    counts them in OUT_SURFT_FBFLAG. The same is done here and the count is
    the caller's to report -- a solver that raises diverges from VIC on
    exactly the steps that matter, and one that falls back silently hides
    them.
    """
    def f_(ts):
        return _residual_fast(ts, tair, t1_old, *args,
                              grnd_flux_type=grnd_flux_type)[0]

    def f(ts):
        # on the CUDA graph path one fused region (core.fusion)
        return _fusion.run(f_, ts, compact=True)

    mid = 0.5 * (t0_old + tair)
    lo = mid - SURF_DT
    hi = mid + SURF_DT
    # The pack dict, by what it carries rather than by position: the
    # frozen-soil inputs, the bare-soil inputs and the finite-difference
    # profile are dicts at the end of `args` too, and none carries
    # 'thin'.
    pack = next((a for a in args
                 if isinstance(a, dict) and 'thin' in a), None)
    if pack is not None:
        # calc_surf_energy_bal.c:265-268: with the pack included the bracket
        # is [T[0] - SURF_DT, 0]; root_brent widens from there on fa*fb >= 0
        # exactly as the loop below does.
        lo = torch.where(pack['thin'], t0_old - SURF_DT, lo)
        hi = torch.where(pack['thin'], torch.zeros_like(hi), hi)
    # The implicit profile solve can skip the columns whose residual is
    # not kept (soiltemp.only): the widened brackets' and the stopped
    # search elements'.
    fd = next((a for a in args if isinstance(a, dict)
               and a.get('implicit')), None)
    live_ctx = soiltemp.only if fd is not None else None
    with torch.no_grad():
        flo, fhi = f(lo), f(hi)
        if graphsafe.ACTIVE:
            # the CUDA graph path: the same tries, host-decided between
            # graph replays (graphsafe.widen); every element's residual
            # is evaluated (live_ctx only saves the work of the ones
            # whose value is not kept)
            lo, hi, flo, fhi = graphsafe.widen(
                lambda ts_: f(ts_), lo, hi, flo, fhi, (), ROOT_BRENT_TSTEP,
                ROOT_BRENT_MAXTRIES, compact=True)
        for _ in range(0 if graphsafe.ACTIVE else ROOT_BRENT_MAXTRIES):
            bad = (flo * fhi) >= 0.0
            if _implicit.LIVE_PROBE:
                _h = _implicit.LIVE_HIST.setdefault(
                    ('widen', live_ctx is not None, int(bad.numel())), {})
                _h.setdefault(_, []).append(int(bad.sum()))
            if not bool(bad.any()):
                break
            lo = torch.where(bad, lo - ROOT_BRENT_TSTEP, lo)
            hi = torch.where(bad, hi + ROOT_BRENT_TSTEP, hi)
            if live_ctx is not None:
                with live_ctx(bad):
                    flo = torch.where(bad, f(lo), flo)
                    fhi = torch.where(bad, f(hi), fhi)
            else:
                flo = torch.where(bad, f(lo), flo)
                fhi = torch.where(bad, f(hi), fhi)
        # A residual that is not a number is not a bracket either. NaN
        # compares False against everything, so `(flo * fhi) >= 0` lets a
        # NaN element straight through to bisection, which returns NaN, and
        # the fallback never fires. Non-finite endpoints are failures.
        failed = (((flo * fhi) >= 0.0)
                  | ~torch.isfinite(flo) | ~torch.isfinite(fhi))

    # tair and t1_old go through `args`, not the closure: t1_old is the last
    # step's soil temperature, live back to the start of the run, and a
    # closure would hide it from the detaching in implicit._local_dfdt's
    # reverse-mode fallback. See implicit._local_dfdt.
    def g_(ts, tair_, t1_, *a):
        return _residual_fast(ts, tair_, t1_, *a,
                              grnd_flux_type=grnd_flux_type)[0]

    def g(ts, tair_, t1_, *a):
        # on the CUDA graph path one fused region (core.fusion)
        return _fusion.run(g_, ts, tair_, t1_, *a, compact=True)

    if SURFACE_SOLVER == 'brent':
        # calc_surf_energy_bal.c 5.1.0:279 (5.0.1:301): root_brent from
        # the bracket above, its iteration and stopping rule as the C's
        # (root_brent.c 5.1.0:227-294). The implicit-function-theorem
        # Newton step supplies the gradient only: its value is added and
        # taken away again, so the root is root_brent's to the bit with
        # gradients on or off (a Newton step at a residual of VIC's ~1e-6
        # W m-2 would move it by up to ROOT_BRENT_T).
        gargs = (tair, t1_old) + tuple(args)
        with torch.no_grad():
            r0 = _root_brent(g, lo, hi, flo, fhi, args=gargs, t_tol=tol,
                             live_ctx=live_ctx, compact=True)
        if torch.is_grad_enabled():
            rg = root_with_grad(g, r0, args=gargs)
            root = r0 + (rg - rg.detach())
        else:
            root = r0
    else:
        root = implicit_solve(g, lo, hi, args=(tair, t1_old) + tuple(args),
                              tol=tol)
    # implicit_solve returns NaN when the residual's derivative at the root
    # is NaN -- the Newton correction divides by it -- even though the root
    # itself was found and the residual there is zero. That is not a
    # converged result and must not be carried as one: it falls back and is
    # counted with the bracket failures, so a future `where` with a
    # non-finite unselected branch is caught here rather than poisoning the
    # soil column on the next step.
    failed = failed | ~torch.isfinite(root)
    root = torch.where(failed, t0_old, root)
    return root, failed


def solve_canopy_air_temp(in_sensible, ra, tair, air_dens):
    """The canopy air temperature with CLOSE_ENERGY, with gradients.

    calc_atmos_energy_bal.c:73-102 (5.0.1:93-122 identical): Tcanopy is
    root_brent's root of the atmospheric energy balance over
    Tair +- CANOPY_DT, widened as VIC widens (root_brent.c:169-179); the
    reported sensible heat is the balance at the solved temperature
    (func_atmos_energy_bal.c:34-40). A bracket that never holds a sign
    change falls back to Tair (TFALLBACK, which TorchVIC always honours)
    and is flagged, as :86-91 does.

    Returns (tcanopy, sensible_heat, fbflag). The residual is linear, so
    the gradient is the closed form's: the root_brent value is kept to
    the bit and the closed form only carries the backward pass.
    """
    def f(tc):
        return (in_sensible
                - snow.calc_sensible_heat(air_dens, tair, tc, ra))

    lo = tair - CANOPY_DT
    hi = tair + CANOPY_DT
    with torch.no_grad():
        flo, fhi = f(lo), f(hi)
        for _ in range(ROOT_BRENT_MAXTRIES):
            bad = (flo * fhi) >= 0.0
            if not bool(bad.any()):
                break
            lo = torch.where(bad, lo - ROOT_BRENT_TSTEP, lo)
            hi = torch.where(bad, hi + ROOT_BRENT_TSTEP, hi)
            flo = torch.where(bad, f(lo), flo)
            fhi = torch.where(bad, f(hi), fhi)
        failed = (((flo * fhi) >= 0.0)
                  | ~torch.isfinite(flo) | ~torch.isfinite(fhi))
        r0 = _root_brent(f, lo, hi, flo, fhi, t_tol=ROOT_BRENT_T)
    if torch.is_grad_enabled():
        closed = tair - in_sensible * ra / (CONST_CPMAIR * air_dens)
        root = r0 + (closed - closed.detach())
    else:
        root = r0
    tcanopy = torch.where(failed, tair, root)
    sensible = snow.calc_sensible_heat(air_dens, tair, tcanopy, ra)
    return tcanopy, sensible, failed


def _surface_step(tair, t0_old, t1_old, t2, depth0, kappa1, kappa2, cs1, cs2,
                 dp, dt, surf_atten, net_short, long_in, ra_neutral,
                 ref_height, displacement, roughness, wind, air_dens, le,
                 evap_of, depth_all, pack=None, bare_frac=None,
                 frozen_soil=None, bare=None, fd=None,
                 grnd_flux_type='GF_410', depth_lsum=None):
    """One step of the surface energy balance for a set of tiles.

    Returns a dict of the reported quantities (put_data.c sign convention)
    and the new thermal state. Everything is elementwise over whatever
    leading shape the inputs share -- [cell, tile] in the basin loop.

    `fd`, when given, is the finite-difference profile's state and
    parameters (torchvic.core.soiltemp.solve_profile's inputs as a
    dict, plus prepare_full_energy's `ice0`/`moist0`, the top layer's
    `porosity0`/`bubble0`/`expt0`, the layer `moist`/`max_moist`/
    `expt`/`bubble` for the layer ice, and `tmp_nnodes` for
    QUICK_SOLVE's reduced trials): the residual solves the profile
    at each trial Ts, the nodes for the next step are the solved
    profile, and `layer_ice` carries the profile-integrated layer ice
    (None on the QUICK_FLUX path).
    """
    args = (t0_old, t2, depth0, depth0, kappa1, kappa2, cs1, cs2, dp, dt,
            surf_atten, net_short, long_in, ra_neutral, ref_height,
            displacement, roughness, wind, air_dens, le, evap_of,
            pack, bare_frac, frozen_soil)
    if bare is not None or fd is not None:
        # Positional: fd must not land in bare's slot.
        args = args + (bare,)
    if fd is not None:
        args = args + (fd,)
    ts, failed = solve_surface_temperature(tair, t0_old, t1_old, args,
                                           grnd_flux_type=grnd_flux_type)
    if fd is not None and fd.get('tmp_nnodes') is not None:
        # calc_surf_energy_bal.c:397-431: QUICK_SOLVE bracketed on the
        # reduced column, and the surface crossed freezing -- solve
        # again on the full column, from the same bracket.
        fd_full = dict(fd, tmp_nnodes=None)
        args_full = args[:-1] + (fd_full,)
        redo = t0_old * ts < 0.0
        if bool(redo.any()):
            ts2, failed2 = solve_surface_temperature(
                tair, t0_old, t1_old, args_full,
                grnd_flux_type=grnd_flux_type)
            ts = torch.where(redo, ts2, ts)
            failed = torch.where(redo, failed2, failed)
        args = args_full
    # The residual at the solved temperature is the surface balance's own
    # error (func_surf_energy_bal.c returns `error`, calc_surf_energy_bal.c
    # stores it as energy->error), so it is reported rather than rebuilt:
    # as OUT_ENERGY_ERROR with energy_error='solver', the default. It is
    # not what VIC 5 writes there: put_data.c:538-552 forms OUT_ENERGY_ERROR
    # from the cell's reported terms, which energy_error='vic' reproduces
    # (model/outvars.py).
    # calc_surf_energy_bal.c:532 evaluates it on the full column, which
    # is what `args` now holds under QUICK_SOLVE too.
    def at_root(ts_):
        return _residual(ts_, tair, t1_old, *args,
                         grnd_flux_type=grnd_flux_type)
    resid_at_root, terms = _fusion.run(at_root, ts, compact=True)
    t1 = terms['t1']
    if pack is not None and 'thin' in pack:
        # calc_surf_energy_bal.c 5.1.0:598-601: with INCLUDE_SNOW the
        # understory's NetShortUnder is NetShortBare + NetShortSnow, the
        # shortwave the residual used (put_data.c 5.1.0:871 via
        # NetShortAtmos)
        net_short = torch.where(pack['thin'],
                                (net_short + pack['net_short_snow']) + 0.0,
                                net_short)
    layer_ice = None
    if fd is None:
        # Node temperatures for the next step
        # (calc_surf_energy_bal.c:593-599). Zsum_node[2] is dp
        # (vic_init.c:825-827), so the third node is avg_temp + (T1 -
        # avg_temp) exp(-(dp - D1)/dp).
        t_node2 = t2 + (t1 - t2) * torch.exp(-(dp - depth0) / dp)
        layer_t = layer_temperatures_quick_flux(ts, t1, t2, depth_all, dp,
                                                lsum=depth_lsum)
        t_node = torch.stack([ts, t1, t_node2], dim=-1)
    else:
        # The solved profile is the next step's
        # (calc_layer_average_thermal_props, frozen_soil.c:47-49), and
        # the layer temperatures and ice are integrated from it
        # (soil_conduction.c). One frost area with the whole weight
        # and zero slope: SPATIAL_FROST is refused (vic_init.c:712-714
        # with Nfrost 1, and frost_slope unread).
        t_node = terms['t_new']
        fr1 = torch.ones(ts.shape + (1,), dtype=ts.dtype, device=ts.device)
        fs0 = torch.zeros_like(ts)
        tmp_t, tmp_z = soiltemp.frost_temperature_and_depth(
            fd['zsum'], t_node, fd['depth'], fr1, fs0, fd['nnodes'],
            fd['nlayers'])
        layer_t = soiltemp.layer_temperature(
            tmp_t, tmp_z, fd['zsum'], fd['depth'], fd['nnodes'],
            fd['nlayers'])
        layer_ice = soiltemp.layer_ice_content(
            tmp_t, tmp_z, fd['zsum'], fd['depth'], fd['max_moist'],
            fd['expt'], fd['bubble'], fd['nnodes'], fd['nlayers'],
            fd['fs_active'], fd['frozen_soil'], fd['moist'])[..., 0]
    return dict(
        t_surf=ts, t_node=t_node,
        layer_temp=layer_t, layer_ice=layer_ice, fallback=failed,
        # Reported convention: latent, sensible, ground flux and storage
        # change are negated on output (put_data.c:913-925).
        latent=-terms['latent'], sensible=-terms['sensible'],
        grnd_flux=-terms['grnd'], delta_h=-terms['delta_h'],
        net_short=net_short, net_long=terms['net_long'],
        net_rad=net_short + terms['net_long'], long_out=terms['long_out'],
        aero_resist=terms['ra_used'], aero_resist_veg=terms['ra_veg0'],
        evap_mm=terms['evap_mm'],
        snow_flux=terms['snow_flux'],
        latent_sub=terms['latent_sub'], vapor_flux=terms['vapor_flux'],
        fusion=terms['fusion'],
        delta_cc=terms['delta_cc'], refreeze_energy=terms['refreeze_energy'],
        # The internal net radiation the canopy was driven with, VIC's sign.
        # Returned so a caller re-evaluating at the root uses the same number
        # the residual used, instead of rebuilding it from three reported
        # quantities with their signs flipped back -- which is right, and is
        # exactly the mistake this module's docstring says not to invite.
        net_bare_rad=terms['net_bare_rad'],
        # Internal closure, which reads zero when the solve converged.
        # Not net_bare_rad + sensible + latent: that was the whole residual
        # once, but the balance has since gained the conduction against the
        # pack and, for a thin pack, its latent, cold-content and refreeze
        # terms. Reporting the old three left 0.365 W/m2 of "error" on the
        # Walnut Gulch year where VIC reports 2.7e-05 -- the terms the
        # solver had driven to zero were simply missing from the report.
        energy_error=resid_at_root,
    )


def surface_step(*args, **kwargs):
    """_surface_step inside a hoisting scope (core.compact): the part of
    the residual that does not depend on the surface temperature is
    evaluated once for the bracket, the widening, the search and the
    residual at the root."""
    with graphsafe.hoist_scope():
        return _surface_step(*args, **kwargs)


surface_step.__doc__ = _surface_step.__doc__ + surface_step.__doc__


def water_balance_ground_flux(tair, t0_old, t1_old, t2, d1, kappa1, kappa2,
                              cs1, cs2, dp, dt, surf_atten,
                              grnd_flux_type='GF_410'):
    """The QUICK_FLUX ground heat flux and heat-storage change at Tsurf = Tair.

    VIC's water-balance mode (FULL_ENERGY and FROZEN_SOIL both FALSE) does
    not solve for a surface temperature: calc_surf_energy_bal.c:542-545 sets
    Tsurf = Tair and evaluates func_surf_energy_bal once there (:552). That
    evaluation still runs the Liang et al. (1999) quick flux, which
    get_global_param.c (image :943-949) requires in this mode:

        T1      = estimate_T1(Tair, T1_old, T2, D1, D2 = D1, ...)   (:405)
        grnd    = surf_atten (kappa1/D1 (T1 - Tair)
                  + kappa2/D2 (1 - exp(-D1/dp)) (T2 - T1)) / 2      (:419-429)
        deltaH  = Cs1 ((Ts_old + T1_old) - (Tair + T1)) D1 / dt / 2 (:530-531)

    and both enter NetBareRad (:663-664), which is what canopy_evap and
    arno_evap are driven with. The state it leaves is T[0] = Tair and
    T[1] = T1 (calc_surf_energy_bal.c:593-602 into
    calc_layer_average_thermal_props, frozen_soil.c:66-69), which the next
    step reads as Ts_old and T1_old and the snow pack reads as its ground
    temperature (surface_fluxes.c:389). Every node starts at 0 C in this
    mode, not at avg_temp (generate_default_state.c:110-114), which is why
    deltaH is 86.9 W/m2 on VIC's first day of basin 37.

    Closed form -- no root find -- and differentiable: every term is
    smooth in the soil moisture that sets kappa and Cs. `surf_atten` is the
    factor the C writes as (snow_coverage + (1 - snow_coverage) surf_atten),
    GF_410 (initialize_options.c:39) is the default form; GF_406
    (func_surf_energy_bal.c:385-391) is the surface term alone. Returns
    (grnd, deltaH, T1) in VIC's internal sign, positive toward the surface.
    """
    t1 = estimate_T1(tair, t1_old, t2, d1, d1, kappa1, kappa2, cs2, dp, dt)
    if grnd_flux_type == 'GF_406':
        grnd = surf_atten * (kappa1 / d1 * (t1 - tair))
    else:
        grnd = surf_atten * (kappa1 / d1 * (t1 - tair)
                             + kappa2 / d1 * (1.0 - torch.exp(-d1 / dp))
                             * (t2 - t1)) / 2.0
    delta_h = cs1 * ((t0_old + t1_old) - (tair + t1)) * d1 / dt / 2.0
    return grnd, delta_h, t1


def reference_height(wind_h_veg, displacement, roughness):
    """vic_run.c:259-264. The *library's* wind_h, not the global file's."""
    return torch.where(displacement < wind_h_veg, wind_h_veg,
                       displacement + wind_h_veg + roughness)


def surface_attenuation(lai, rad_atten, fcanopy=1.0):
    """vic_run.c:222-225.  (1 - fcanopy) + fcanopy exp(-rad_atten LAI).

    Applied to the ground flux and to the ARNO radiation -- and NOT to the
    net shortwave, which the reference audit verified holds SWNET =
    SWDOWN (1 - albedo) to 6e-5 at every cell-step. Attenuating the shortwave
    is the physically natural thing to do and would be wrong by up to 35 per
    cent in July.
    """
    return (1.0 - fcanopy) + fcanopy * xmath.exp(-rad_atten * lai)


# fused regions on the CUDA graph path (core.fusion)
from . import fusion as _fusion                                # noqa: E402
thermal_properties = _fusion.region(thermal_properties)
water_balance_ground_flux = _fusion.region(water_balance_ground_flux)
