"""The VIC 5 time loop: snow, vegetation, soil, energy balance and routing.

Every physical kernel here has been measured against VIC's own compiled code
and the measurements are recorded next to each one:

    torchvic.core.column    soil column, 2.6e-14 against runoff()
    torchvic.core.snow      energy-balance snowpack, 12,201 values bit-exact,
                        mass conserving to 4.16e-17
    torchvic.core.atmos     pressure, density, latent heat bit-identical
    torchvic.model.routing   bit-identical to rout_convolution.c for one source,
                        1.5e-15 for 57 sources and 8 outlets

`simulate` runs every cell of a domain as one [time, cell] problem: the
sequential dimension is time, and the cells are a batch, which is what makes
the GPU pay on a large domain. A case is assembled from VIC 5 image-driver
inputs by `torchvic.case.load`; `simulate` itself only sees tensors.

Vegetation tiling, the energy balance (FULL_ENERGY), FROZEN_SOIL, BLOWING,
CARBON, SPATIAL_SNOW and SHARE_LAYER_MOIST are all switched by arguments.
"""
import copy
import inspect
import math
import os

import torch

from ..core import ordered as _ordered

from ..core import (atmos, canopy, column, version as core_version, energy,
                    graphsafe, snow, soiltemp, vegetation)
from ..core import tfallback as _tfx
from ..core import fusion
from ..core import fusion as _fusion   # `fusion` is also a local in the step
# `implicit` is VIC's IMPLICIT flag in simulate's signature, so the
# gradients module of the same name is imported aliased.
from ..core import implicit as _implicit_mod
from . import routing
from ..core.snow import CONST_TKFRZ
from ..core.vegetation import VON_KARMAN_SQ

#: float.h DBL_EPSILON, which put_data.c compares the resistances against
DBL_EPSILON = 2.220446049250313e-16
from .state import (export_state, restore_state, from_vic_file,
                    CARRIED as _CARRIED_STATE)
from . import lanes as _lanes_mod
from . import outvars as _outvars
from . import wbenergy as _wbenergy
from ..core.devconst import device_constant

SEC_PER_DAY = 86400.0
# VIC's CONST_REARTH (vic_physical_constants.h:77) is 6.37122e6, not the
# round 6371000 this used. 220 m of radius is 7e-5 of it, which matters only
# where a cell area is compared against VIC's own -- and the domain file
# supplies those, so this is the fallback path.
EARTH_RADIUS_M = 6.37122e6
# torchvic.core.snow works in metres of water equivalent, as VIC's snow_melt.c
# does: param.SNOW_MAX_SURFACE_SWE is 0.125, which is 125 mm of surface layer,
# and surface_melt divides energy by CONST_LATICE * CONST_RHOFW to get metres.
# Forcing and the soil column are in millimetres. Feeding millimetres to the
# snow module treats a 200 mm pack as 200 m, which then melts at 0.04 m a day
# and is indistinguishable from a pack that never melts: discharge peaks in
# winter instead of June and evapotranspiration collapses because the cell
# never leaves its snow albedo.
MM_PER_M = 1000.0
CM_PER_M = 100.0
#: vic_def.h:45, the most frost sub-areas SPATIAL_FROST allows.
MAX_FROST_AREAS = 10
# VIC's default when a wind forcing is absent, m/s.
DEFAULT_WIND = 1.5
# A guard on the wind speed, m/s, against an exactly-zero (or negative)
# forcing value only. VIC 5.1.0 has no wind floor in the image driver or in
# vic_run: any positive wind goes into CalcAerodynamic unchanged
# (vic_run.c:171, CalcAerodynamic.c:177-186), and only wind == 0 is special
# (Ra = HUGE_RESIST). This used to be 0.1 m/s, which silently replaced every
# calmer 3-hourly value and moved the stable-snow aerodynamic resistance and
# the snow surface temperature of a cold, still, deep pack (ISC cell y451 x118,
# 2004 day 341: forcing 0.0992 m/s, AERO_RESIST 0.86 % off VIC, snow surface
# temperature 2.8e-4 K off). Positive winds below this guard are not touched.
MIN_WIND = 1e-12

#: param.VEG_RATIO_RL_HEIGHT, initialize_parameters.c:68: a surface's
#: height is its roughness over this, and the bare-soil resistance
#: func_surf_energy_bal.c:712-713 forms under a fractional canopy takes its
#: displacement from that height.
VEG_RATIO_RL_HEIGHT = 0.123

#: param.SOIL_WINDH, initialize_parameters.c:61 (5.1.0:41), 10 m in both: the
#: reference height func_surf_energy_bal.c 5.1.0:689 gives the bare soil's
#: CalcAerodynamic, 10 m, whose wind and Richardson number its stability
#: correction then take (the neutral resistance does not depend on it).
SOIL_WINDH = 10.0

#: param.SOIL_RARC, initialize_parameters.c:58. The architectural
#: resistance compute_pot_evap.c:79-81 gives the bare-soil Penman.
SOIL_RARC = 100.0

#: solve_snow.c:139-141, a literal: `if (*snowfall < 1e-5) *snowfall = 0.;`,
#: mm per step, after the gauge correction and before out_prec, out_snow and
#: the snow->snow test at :164 read it. A trace below it is not snow and is
#: not precipitation either: VIC drops it from out_prec, so its water balance
#: never sees it. India's daily cache, 2000-2009, default ramp: 25,745
#: cell-days with 0 < snowfall < 1e-5 mm in 7,814 of 104,841 cells, each of
#: which gated a whole day's evaporation off in the water-balance path.
SNOWFALL_TRACE_MM = 1.0e-5
#: Whether the water-balance thin-pack branch (_wb_thin_pack) also runs on
#: overstory tiles, as VIC's does (snow_melt.c 5.1.0:302-304).
WB_THIN_ON_OVERSTORY = True
# Snow surface roughness and measurement height, m: VIC's snow defaults.
SNOW_Z0 = 0.03
MEAS_HEIGHT = 2.0

#: Skip snow_step on a step where no tile holds ice and no snow falls; its
#: outputs are all masked away then (see simulate). A switch so the skip can
#: be checked against the full routine.
SNOW_FREE_FAST_PATH = True
#: On the graph path, a step with snow on no tile under a graph made for
#: snow (graphsafe SNOW_ANY) is not a broken assumption: the graph's
#: snow_step runs on an all-false mask and _mask_tree zeroes every output,
#: which is the snow-free branch's cached zeros bit for bit (checked in
#: tests/test_graph_snow_adapt.py). Without it a day with snow on some of
#: its steps only (high-altitude summer snowfall) ran again eagerly, ~8x a
#: graph day (~40 % of a full ISC no-grad lane run). Instead the step notes
#: on the device whether it saw snow (graphsafe.seen); a day that saw none
#: switches the next day to the snow-free graph. Found by the calibration
#: work. TORCHVIC_GRAPH_SNOW_ADAPT=0 restores the redo.
GRAPH_SNOW_ADAPT = os.environ.get('TORCHVIC_GRAPH_SNOW_ADAPT', '1') != '0'
#: The canopy stage (snow interception over overstory tiles) runs on the
#: tiles where it is active only -- gathered, solved, scattered back -- not
#: on every tile of every cell with the result masked away. Bit-identical:
#: every operation in it is per tile and every result is read through a mask
#: of those same tiles. A switch so the two can be compared.
CANOPY_COMPACT = True
#: Likewise snow_step, on the tiles that have a pack or snow falling on them
#: (with a pack per tile, no SPATIAL_SNOW): everything downstream reads its
#: outputs through has_snow -- or multiplies them by a cover that is 0
#: without one -- and the packs are independent of each other.
SNOW_COMPACT = True
#: On the graph path (GRAPHS), the snow and canopy stages on a fixed subset
#: of the lanes for the whole simulate call (one window of forcing): every
#: lane that could need either stage in the window -- a pack, liquid in it
#: or snow on its canopy when the window opens, or a step of the window
#: with precipitation below MAX_SNOW_TEMP -- padded to a bucket size
#: (_SNOW_SET_STEP), in place of every lane (every overstory lane for the
#: canopy) with the results masked. The stages run on a lane only where
#: solve_snow.c:146-150 (snow->snow) and :170-171 (the canopy) say they
#: do; snow falls only where :117-141 leave snowfall > 0, which needs
#: precipitation and Tair < MAX_SNOW_TEMP; a pack or canopy snow arises
#: only from snowfall; so a lane outside the set holds zero in every value
#: the masked full-lane stage gives it, and inside it the same values (each
#: operation is per lane). Each step checks the subset on the device
#: (_SNOW_SET_BIT) and a day that broke it runs again eagerly, the rest of
#: the call on every lane. TORCHVIC_SNOW_SET=0 turns it off.
SNOW_SET = os.environ.get('TORCHVIC_SNOW_SET', '1') != '0'
#: bucket ladder of the subset's size: n_lanes * _SNOW_SET_STEP**-k (a
#: new size compiles the stages' regions anew)
_SNOW_SET_STEP = 2.0 ** 0.5
#: the ladder's smallest size (TORCHVIC_SNOW_SET_MIN: a check on small
#: cases); TORCHVIC_SNOW_SET_SAY=1 prints each window's subset sizes
_SNOW_SET_MIN = int(os.environ.get('TORCHVIC_SNOW_SET_MIN', '32'))
_SNOW_SET_SAY = os.environ.get('TORCHVIC_SNOW_SET_SAY', '0') != '0'
#: the subset sizes this process has run, by (stage, domain size): a later
#: window takes the smallest of them that holds its lanes before a new
#: (smaller) size, whose compile costs more than its lanes save (a size
#: new to the process re-traces the stages' regions, ~2 s; on the H200 a
#: lane less saves ~3 ns a step)
_SNOW_SET_USED = {}
#: graphsafe.need's bit for the subset's assumption (graphsafe's own run 0-4)
_SNOW_SET_BIT = 5
#: The water-balance loop on the covered tiles only (torchvic.model.lanes):
#: each covered (cell, tile) pair is one lane, and a sum over a cell's
#: tiles is formed on the padded grid as before, so every output is the
#: padded run's bit for bit. Without gradients, on the water-balance path
#: without SPATIAL_SNOW, BLOWING, CARBON or FROZEN_SOIL. A switch so the
#: two can be compared (TORCHVIC_LANES=0 in the environment turns it off).
#: CUDA only (see simulate).
LANE_COMPACT = os.environ.get('TORCHVIC_LANES', '1') != '0'
#: The energy balance (FULL_ENERGY) on the covered tiles as well, and so on
#: the CUDA graph path: every cell sum of its step goes through the same
#: tile sum (Lanes.tsum) as the water balance's. Not with CLOSE_ENERGY
#: (its iteration re-runs the whole step a data-dependent number of
#: times), LAKES or the daily step's sub-steps, which stay on the padded
#: grid, eagerly. TORCHVIC_EB_LANES=0 keeps every energy run there.
EB_LANES = os.environ.get('TORCHVIC_EB_LANES', '1') != '0'


#: QUICK_FLUX FALSE (the finite-difference profile, explicit or IMPLICIT)
#: on the lanes too; with FROZEN_SOIL or QUICK_SOLVE it stays eager (see
#: graph_mode in simulate)
EB_LANES_FD = os.environ.get('TORCHVIC_EB_LANES_FD', '1') != '0'
#: FROZEN_SOIL with the energy balance on the QUICK_FLUX path, on the lanes
EB_LANES_FS = os.environ.get('TORCHVIC_EB_LANES_FS', '1') != '0'


def _eb_lanes_ok(quick_flux, close_energy, lakes, subdaily):
    return (EB_LANES and (quick_flux or EB_LANES_FD) and not close_energy
            and lakes is None and subdaily is None)


#: On the covered tiles on a CUDA device, the step replayed as a CUDA graph
#: (see simulate). TORCHVIC_GRAPHS=0 turns it off.
GRAPHS = os.environ.get('TORCHVIC_GRAPHS', '1') != '0'
#: VIC-WUR's vic_run hooks (vicwur_core) on the covered tiles and in the
#: CUDA graphs as well (see simulate's _vw_host); TORCHVIC_VW_LANES=0
#: keeps a run with hooks on the padded grid, eagerly, as before.
VW_LANES = os.environ.get('TORCHVIC_VW_LANES', '1') != '0'
#: what the last graph run did: graphs captured, steps run eagerly to
#: capture them, days run again eagerly
GRAPH_STATS = {}
#: profiling aid: the graph-safe step run eagerly every step, never captured
GRAPHS_NO_CAPTURE = os.environ.get('TORCHVIC_GRAPHS_NO_CAPTURE', '0') != '0'
#: check aid: the graph path, every step run under graphsafe.SyncCheck in
#: place of a capture, on a CPU as on a GPU (the covered tiles too on a
#: CPU); what it finds -- every place in the step that would wait for the
#: device inside a capture -- goes to SYNC_FOUND ('file:line' -> operation)
SYNC_CHECK = os.environ.get('TORCHVIC_SYNC_CHECK', '0') != '0'
SYNC_FOUND = {}
# Snow-free surface albedo and emissivity. Single values because this model has
# no vegetation tiling to take them from.
LAND_ALBEDO = 0.20
# Smallest soil-moisture availability the ET power may see. At 1e-4 and the
# lowest permitted et_beta of 0.2, the steepest derivative is about 1.6e3,
# which is large but finite; at zero it is infinite.
ET_AVAIL_FLOOR = 1e-4
SURFACE_EMISSIVITY = 0.97


def _mineral_kw(th, two):
    """thermal_properties' ORGANIC_FRACT / BULK_DENSITY_COMB arguments: the
    mineral densities and the organic fraction of the top two layers, where
    ImageParams.apply_density_options set them (else none, and the
    combined pair is the mineral pair)."""
    if 'organic' not in th:
        return {}
    return dict(organic=two(th['organic'][:, :2]),
                bulk_dens_min=two(th['bulk_dens_min'][:, :2]),
                soil_dens_min=two(th['soil_dens_min'][:, :2]))


def gauge_correction(wind, wind_h, gauge_height, rough, snow_rough):
    """Gauge undercatch factors (rain, snow).  correct_precip.c:19-37.

    The wind at the gauge height from the log profile over the soil's
    roughness (rain) or the snow's (snow), then Bras Figure 4.16's fits.
    `wind` is the step's forcing wind, `rough` and `snow_rough` the soil
    file's, all broadcastable to one another."""
    gw = wind * (torch.log((gauge_height + rough) / rough)
                 / torch.log(wind_h / rough))
    rain = 100. / torch.exp(4.606 - 0.041 * torch.pow(gw, 0.69))
    gw = wind * (torch.log((gauge_height + snow_rough) / snow_rough)
                 / torch.log(wind_h / snow_rough))
    snow = 100. / torch.exp(4.606 - 0.036 * torch.pow(gw, 1.75))
    return rain, snow


def partition_precipitation(prcp, tair, t_max_snow, t_min_rain, check=True):
    """Split precipitation into rain and snow over a temperature range.

    VIC's MAX_SNOW_TEMP / MIN_RAIN_TEMP linear ramp: all snow at or below
    `t_min_rain`, all rain at or above `t_max_snow`, linear between. A hard
    threshold has zero gradient with respect to the threshold almost
    everywhere and so cannot be fitted; the ramp is both what VIC does and
    what is differentiable.

    The ordering is checked because getting it backwards does not raise: the
    denominator changes sign and the ramp silently inverts, turning warm days
    into snow and cold days into rain. That produced a snowpack of 1,842 mm
    that never melted, and it looked like a snow physics problem rather than a
    swapped pair of arguments.

    `check=False` skips the test, for a caller that has made it already:
    reading the two parameters off a GPU is a host synchronisation, and
    simulate's time loop would otherwise make it every step for parameters
    that do not change within the run (it checks once before the loop).
    """
    if check:
        check_snow_range(t_max_snow, t_min_rain)
    frac_snow = torch.clamp((t_max_snow - tair) / (t_max_snow - t_min_rain),
                            0.0, 1.0)
    return prcp * (1.0 - frac_snow), prcp * frac_snow


def check_snow_range(t_max_snow, t_min_rain):
    """partition_precipitation's ordering check (see there). Either
    threshold may be a scalar (VIC 5's global parameter) or a [cell] tensor
    (a parameter extension: per-cell thresholds, found by the calibration
    work); the check holds on every cell."""
    if not (torch.is_tensor(t_max_snow) and t_max_snow.dim() > 0
            or torch.is_tensor(t_min_rain) and t_min_rain.dim() > 0):
        if float(t_max_snow) <= float(t_min_rain):
            raise ValueError('t_max_snow (%g) must exceed t_min_rain (%g); '
                             'the ramp inverts silently otherwise'
                             % (float(t_max_snow), float(t_min_rain)))
        return
    gap = torch.as_tensor(t_max_snow) - torch.as_tensor(t_min_rain)
    if not bool((gap > 0.0).all()):
        i = int(torch.argmin(torch.where(gap > 0.0, gap, gap.new_full(
            (), -float('inf')))).reshape(-1)[0]) if gap.dim() else 0
        raise ValueError('t_max_snow must exceed t_min_rain on every cell '
                         '(cell %d: gap %g); the ramp inverts silently '
                         'otherwise' % (i, float(gap.reshape(-1)[i])))


def penman_monteith_pet(atm, net_rad, rs, ra):
    """Potential evapotranspiration, mm/day.  penman.c, same algebra.

    `net_rad` is in W/m2, `rs` and `ra` in s/m.
    """
    slope = atmos.svp_slope(atm['tair'])
    gamma = 1628.6 * atm['press'] / atm['lv']
    num = (slope * net_rad
           + atm['air_dens'] * 1013.0 * atm['vpd'] / ra)
    den = slope + gamma * (1.0 + rs / ra)
    # W/m2 -> mm/day
    return torch.clamp(num / den / atm['lv'] * SEC_PER_DAY, min=0.0)


def _like(ref, value):
    """`value` broadcast to the shape of `ref`, whether it is a number or a
    tensor.

    torch.full_like insists on a Python number, which makes every constant it
    touches undifferentiable. Broadcasting instead costs nothing and lets any
    of these be a run parameter rather than a fixed number.
    """
    return device_constant(value, ref.dtype,
                           ref.device) * torch.ones_like(ref)


def _bcast(x, ref):
    """A cell or tile quantity expanded onto `ref`'s shape.

    The loop carries three shapes -- [cell], [cell, tile] and, for the snow
    state, [cell, tile] again -- and a term built from two of them has to
    agree. This adds trailing axes and expands; it never reduces, so a
    mismatch that is not a broadcast is still an error.
    """
    if not torch.is_tensor(x):
        return torch.full_like(ref, float(x))
    while x.dim() < ref.dim():
        x = x.unsqueeze(-1)
    return x.expand_as(ref)


def snow_forcing_at(atm, t, rain, snowfall, albedo, wind=None,
                    meas_height=None, dt=SEC_PER_DAY, depth=None,
                    density=None, t_grnd=None, shortwave=None, longwave=None,
                    per_tile=False, ra_snow=None, wind_snow=None,
                    z0_snow=None):
    """Assemble the forcing dict `torchvic.core.snow.snow_step` expects.

    `wind` defaults to the forcing's own field when the atmosphere carries
    one, and to DEFAULT_WIND only when it does not. It used to default to
    DEFAULT_WIND unconditionally *in the signature*, which bound 1.5 m/s at
    import time -- changing the module constant afterwards did nothing,
    because `__defaults__` already held the old value.

    `meas_height` is the global file's WIND_H. Hard-wiring 2 m where a file
    declares 10 m understates the aerodynamic resistance ln(z/z0)^2 by a
    factor of 1.91, which nearly doubles turbulent exchange.
    """
    # With a pack per vegetation tile, everything the cell supplies -- air
    # temperature, humidity, pressure, wind -- is the same for every tile of
    # that cell and is simply given a tile axis to broadcast on. What differs
    # between tiles is what the canopy above them did: the snowfall that got
    # through, the radiation that got through, and the pack each one has.
    def cell(x):
        return x.unsqueeze(-1) if (per_tile and isinstance(x, torch.Tensor)
                                   and x.dim() == 1) else x

    tair = cell(atm['tair'][t])
    if wind is None:
        wind = atm['wind'][t] if 'wind' in atm else DEFAULT_WIND
    wind = cell(wind) if isinstance(wind, torch.Tensor) else wind
    z = MEAS_HEIGHT if meas_height is None else meas_height
    # torch.full_like wants a Python number, so wind is broadcast instead of
    # filled. That also lets wind be a per-cell tensor rather than a
    # constant.
    w = torch.clamp(_like(tair, wind), min=MIN_WIND)   # see MIN_WIND above
    # CalcAerodynamic.c:97-98, the snow term of the no-overstory branch:
    #     Ra[2] = log((2 + z0) / z0) * log(zref / z0) / K2      then / wind
    # This used log(zref/z0) squared, which is not the same thing: VIC's two
    # logarithms have different numerators, one anchored 2 m above the snow
    # roughness and one at the reference height. At z = 10 m, z0 = 0.03 the
    # squared form gives 33.75 where VIC gives 4.21 * 5.81 = 24.46 -- 38 per
    # cent high, and in the same direction for every snow-covered step.
    # (then `/ wind` as an IEEE division: a Python number over a tensor is
    # reciprocal-then-multiply in torch, an ulp off the C's quotient)
    ra = (torch.full_like(w, math.log((2.0 + SNOW_Z0) / SNOW_Z0)
                          * math.log(z / SNOW_Z0) / VON_KARMAN_SQ) / w)
    # CalcAerodynamic.c:42-45: the pack's balance is handed the wind
    # brought down to 2 m over the snow, U[2] = wind * ln((2 + z0s)/z0s) /
    # ln(zref/z0s), and its reference height ref_height[2] = 2 + z0s;
    # SnowPackEnergyBalance.c:183 corrects Ra with exactly those two
    # (StabilityCorrection(Z, 0, TMean, Tair, Wind, Z0[2])). The forcing's
    # wind and the measurement height went in here before.
    w_snow = math.log((2.0 + SNOW_Z0) / SNOW_Z0) / math.log(z / SNOW_Z0) * w
    # Under an overstory none of that applies. solve_snow.c:242 sets
    # UnderStory = 2 where there is ground snow, and snow_melt.c is then
    # handed aero_resist[2], wind[2], ref_height[2] and roughness[2] --
    # the snow-surface entries CalcAerodynamic filled from the *canopy*
    # wind profile (:73-104), where the wind at the trunk space is a
    # fraction of the open wind. Using the open-field resistance under a
    # forest gives the pack far too much turbulent exchange: it sublimates
    # and melts too fast. The caller supplies those entries per tile.
    zs = SNOW_Z0 if z0_snow is None else z0_snow
    if ra_snow is not None:
        ra = ra_snow
    if wind_snow is not None:
        w_snow = wind_snow
    return dict(
        snowfall=snowfall, rain=rain, air_temp=tair,
        # surface_fluxes.c:389: Tgrnd = energy->T[0], the top soil node as
        # the previous step left it, which the conduction term through the
        # pack sees (SnowPackEnergyBalance.c:252). A caller without a soil
        # thermal solution has no node and gets 0 C, the old placeholder.
        t_grnd=(torch.zeros_like(tair) if t_grnd is None else t_grnd),
        # solve_snow.c:156-157 starts the understory radiation at the sky's
        # and :185 replaces the shortwave with the canopy-attenuated one
        # under an overstory, while snow_intercept writes the canopy's own
        # emission into LongUnderIn. A pack under a forest that is given the
        # whole sky melts and sublimates far too fast: measured on 90 winter
        # days of FindleyLake, snow melt +152%, sublimation +86%, SWE -31%.
        wind=w_snow,
        net_short=((cell(atm['shortwave'][t]) if shortwave is None
                    else shortwave) * (1.0 - albedo)),
        long_snow_in=(cell(atm['longwave'][t]) if longwave is None
                      else longwave),
        air_dens=cell(atm['air_dens'][t]),
        eact_air=cell(atm['eact_air'][t]), lv=cell(atm['lv'][t]),
        press=cell(atm['press'][t]), vpd=cell(atm['vpd'][t]), dt=dt,
        z=(_like(tair, 2.0 + SNOW_Z0) if z0_snow is None else 2.0 + zs),
        z0=(_like(tair, SNOW_Z0) if z0_snow is None else zs), ra=ra,
        # The pack's depth and density *on entry*, which is what snow_melt.c:177
        # hands SnowPackEnergyBalance for the conduction term; solve_snow.c
        # updates both only afterwards (:360-372). A caller that carries
        # neither gets the old placeholders, and no depth back.
        snow_depth=(torch.zeros_like(tair) if depth is None else depth),
        snow_density=(_like(tair, 300.0) if density is None else density))


def gamma_unit_hydrograph(nuh, shape, scale, dtype=torch.float64, device=None):
    """A unit hydrograph normalised to sum to one, so routing conserves mass.

    [nuh] for scalar shape and scale; [nuh, cell] when either is given per
    cell (an ensemble gives each member's cells their member's value)."""
    k = torch.arange(1, nuh + 1, dtype=dtype, device=device)
    per_cell = any(torch.is_tensor(v) and v.dim() == 1 for v in (shape, scale))
    if per_cell:
        k = k.unsqueeze(1)
    uh = k ** (shape - 1.0) * torch.exp(-k / scale)
    return uh / uh.sum(0, keepdim=per_cell)


def _take_tiles(x, shape, rows, cols):
    """x broadcast to `shape` ([cell, tile]) at (rows, cols): [n, ...].
    Numbers and 0-d tensors pass through; dicts are taken entry by entry."""
    if isinstance(x, dict):
        return {k: _take_tiles(v, shape, rows, cols) for k, v in x.items()}
    if not torch.is_tensor(x) or x.dim() == 0:
        return x
    return torch.broadcast_to(x, tuple(shape) + tuple(x.shape[2:])
                              if x.dim() > 2 else shape)[rows, cols]


def _put_tiles(x, shape, rows, cols):
    """The inverse of _take_tiles: [n, ...] back to [cell, tile, ...] with
    zeros (False) on the tiles not taken."""
    if isinstance(x, dict):
        return {k: _put_tiles(v, shape, rows, cols) for k, v in x.items()}
    if not torch.is_tensor(x) or x.dim() == 0:
        return x
    out = torch.zeros(tuple(shape) + tuple(x.shape[1:]), dtype=x.dtype,
                      device=x.device)
    out[rows, cols] = x
    return out


#: The VegMosaic fields that are [cell, tile, step] under a vegetation
#: history (torchvic.io.veghist.apply).
_STEP_FIELDS = ('lai', 'fcanopy', 'albedo', 'displacement', 'veg_rough')


class _DayView(object):
    """What simulate's day_hook sees between two days: `moist`, the soil
    moisture the next step starts from in the loop's layout ([cell, tile,
    layer], or [lane, 1, layer] on the covered tiles of `lanes`), and
    `prcp`, the [step, cell] precipitation forcing the coming steps read
    (mm per step; the hook may add to it in place). Read-only otherwise.

    `day_fluxes()` (VIC-WUR's plugins read them, docs/connectors/vicwur.md
    2.1b) returns {'runoff', 'baseflow'}: the day just ended's OUT_RUNOFF
    and OUT_BASEFLOW summed over its steps, [column] mm; and, with a
    vicwur_core carrying the stored water stress, 'water_stress' (the last
    step's cell->water_stress per tile, [column, tile]). Computed only when
    called, so a hook that does not ask changes nothing."""
    __slots__ = ('moist', 'prcp', 'lanes', 'soil_ice', '_fluxes',
                 '_outputs', 'graphs')

    def __init__(self, moist, prcp, lanes, soil_ice=None, fluxes=None,
                 outputs=None, graphs=False):
        self.moist, self.prcp, self.lanes = moist, prcp, lanes
        # the ice store the next step starts from (read-only; None without
        # FROZEN_SOIL)
        self.soil_ice = soil_ice
        self._fluxes = fluxes
        self._outputs = outputs
        # the loop replays CUDA graphs (what GRAPH_STATS says after a call)
        self.graphs = graphs

    def day_outputs(self):
        """simulate(day_outputs=True): the outputs of the day just ended,
        with its state (return_state), as a call over that day alone from
        the state the day started on returns them."""
        if self._outputs is None:
            raise ValueError('this day view carries no outputs '
                             '(simulate(day_outputs=True))')
        return self._outputs()

    def day_fluxes(self):
        if self._fluxes is None:
            raise ValueError('this day view carries no fluxes')
        return self._fluxes()


class _Fixed(object):
    """atm[k][t] for any t: the step's buffer (a graph's forcing). At
    module level so that every simulate call's is one type (a fused
    region's guards check it)."""
    __slots__ = ('v',)

    def __init__(self, v):
        self.v = v

    def __getitem__(self, t_):
        return self.v


#: every index reads this one snow step's forcing
_At = _Fixed


class _AnyIndex(object):
    """x[i] is the same value for every i."""
    __slots__ = ('v',)

    def __init__(self, v):
        self.v = v

    def __getitem__(self, i):
        return self.v


class _Phases(object):
    """Wall time of simulate's set-up, loop and outputs, device
    synchronised, when TORCHVIC_PHASES=1 (else nothing is measured)."""

    def __init__(self, device):
        self.on = os.environ.get('TORCHVIC_PHASES', '0') != '0'
        self.cuda = torch.device(device).type == 'cuda'
        self.times = {}
        GRAPH_STATS.clear()
        if self.on:
            import time
            self._time = time.time
            self._sync()
            self.t = self._time()

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def mark(self, name):
        if self.on:
            self._sync()
            t = self._time()
            self.times[name] = t - self.t
            self.t = t


def _mask_tree(x, mask):
    """x where `mask`, zeros (False) elsewhere, through dicts: what
    _put_tiles leaves on the tiles it was not given."""
    if isinstance(x, dict):
        return {k: _mask_tree(v, mask) for k, v in x.items()}
    if not torch.is_tensor(x) or x.dim() == 0:
        return x
    if x.dim() < mask.dim():
        x = torch.broadcast_to(x, mask.shape)
    m = mask.reshape(tuple(mask.shape) + (1,) * (x.dim() - mask.dim()))
    return torch.where(m, x, torch.zeros_like(x))


# a fused region on the CUDA graph path (core.fusion)
_mask_tree = fusion.region(_mask_tree)


def _sub_gather_(d, r, c, shape):
    """A dict's [cell, tile]-broadcastable tensors at the (r, c) subset,
    [n]; other entries as they are."""
    return {k: (torch.broadcast_to(v, shape)[r, c] if torch.is_tensor(v)
                else v) for k, v in d.items()}


def _sub_scatter_(d, pos, valid, keep):
    """A dict's [n] subset tensors back on the [cell, tile] grid, zero
    off the subset and where `keep` (if given) is false: the subset's
    entry at `pos` (its index in the subset) where `valid`."""
    out = {}
    for k, v in d.items():
        if not torch.is_tensor(v) or v.dim() == 0:
            out[k] = v
            continue
        if keep is not None:
            v = torch.where(keep, v, torch.zeros_like(v))
        x = v[pos]
        out[k] = torch.where(valid, x, torch.zeros_like(x))
    return out


_sub_gather = fusion.region(_sub_gather_)


def _set_take_(x, shape, rows, cols):
    """_take_tiles of a tree (the snow step's state and forcing) at the
    snow subset (SNOW_SET); a tuple entry by entry."""
    if isinstance(x, tuple):
        return tuple(_take_tiles(v, shape, rows, cols) for v in x)
    return _take_tiles(x, shape, rows, cols)


_set_take = fusion.region(_set_take_)


def _set_put_(x, pos, valid, keep):
    """The snow subset's results (a tree of [n, ...]) on the [lane, 1, ...]
    grid as _mask_tree(_put_tiles(x), mask) leaves them: the subset's entry
    at `pos` where `valid` and `keep` (the subset's mask) hold, zero
    (False) elsewhere; 0-d entries as they are."""
    if isinstance(x, dict):
        return {k: _set_put_(v, pos, valid, keep) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_set_put_(v, pos, valid, keep) for v in x)
    if not torch.is_tensor(x) or x.dim() == 0:
        return x
    k = keep.reshape(tuple(keep.shape) + (1,) * (x.dim() - keep.dim()))
    x = torch.where(k, x, torch.zeros_like(x))
    y = x[pos]
    m = valid.reshape(tuple(valid.shape) + (1,) * (y.dim() - valid.dim()))
    return torch.where(m, y, torch.zeros_like(y))


_set_put = fusion.region(_set_put_)


def _snow_bucket(n_need, n_all):
    """The smallest size of the ladder n_all * _SNOW_SET_STEP**-k that
    holds n_need lanes (n_all itself at the top)."""
    b = n_all
    while True:
        nb = int(math.ceil(b / _SNOW_SET_STEP))
        if nb < n_need or nb < _SNOW_SET_MIN or nb >= b:
            return b
        b = nb


def _snow_window_set(prcp, tair, t_max_snow, lane_cell, stores, over):
    """The lanes the snow and canopy stages may run on in a window (see
    SNOW_SET), as two padded index sets, or None where the set would be
    every lane. `prcp`, `tair`: the window's [time, cell] forcing;
    `stores`: [lane, 1(, ...)] snow and canopy stores at its start;
    `over`: [lane, 1] overstory, or None. Host-side, once per call."""
    n = lane_cell.shape[0]
    # solve_snow.c:117-141: snowfall > 0 needs precipitation and
    # Tair < MAX_SNOW_TEMP (partition_precipitation's ramp, any gauge
    # factor); NaN counts as possible
    t_wet = torch.where(prcp != 0.0, tair,
                        torch.full_like(tair, float('inf'))).amin(0)
    t_l = t_wet.index_select(0, lane_cell)
    tmx = torch.as_tensor(t_max_snow, dtype=t_l.dtype, device=t_l.device)
    if tmx.dim() > 0:
        # a per-cell threshold arrives gathered to the lanes (Lanes.params)
        tmx = tmx.reshape(-1)
        if tmx.numel() != 1 and tmx.shape[0] != n:
            raise ValueError('t_max_snow: %d values for %d lanes'
                             % (tmx.shape[0], n))
    snowy = ~(t_l >= tmx)
    for s_ in stores:
        if s_ is not None:
            snowy = snowy | (s_.reshape(n, -1) != 0.0).any(1)
    sets = []
    for stage, dom in (('snow', None), ('canopy', over)):
        d_ = (torch.ones(n, dtype=torch.bool, device=snowy.device)
              if dom is None else dom.reshape(n))
        n_dom = int(d_.sum())
        need_ = snowy & d_
        n_need = int(need_.sum())
        used = _SNOW_SET_USED.setdefault((stage, n_dom), set())
        fit = [u for u in used if u >= n_need]
        b = min(fit) if fit else _snow_bucket(n_need, n_dom)
        used.add(b)
        if b >= n_dom:
            sets.append(None)
            continue
        # padded with the first lanes of the domain outside the set
        pad = (d_ & ~need_).nonzero()[:b - n_need, 0]
        idx = torch.sort(torch.cat([need_.nonzero()[:, 0], pad]))[0]
        sets.append(idx)
    return sets, snowy

#: a name a fused block (core.fusion.run) left unassigned
_UNSET = object()


def _bound(local_vars, names):
    """The values of `names` in a function's locals(), _UNSET for one not
    bound there (a fused block's inputs that the eager code before it binds
    on some paths only). The dict is not kept: the frame's locals() dict
    holding itself would keep every tensor of the step alive until the
    cyclic garbage collector ran (measured: 15-18 GB on the ISC domain)."""
    return tuple(local_vars.get(n, _UNSET) for n in names)


def _cell_sums_(plain, snowc, cv, start, mask, n_tiles, lane_cell):
    """The step's cover-weighted cell sums at the lanes, many in one fused
    region: `plain[k]` -> tsum(x * cv) (to_cell), `snowc[k]` = (x, hp) ->
    the pack-cover mean (to_snow_cell), bit for bit (Lanes.tsum's tree:
    each cell's once, `start`, `mask` per cell, then at its lanes)."""
    from .tsum_kernel import tsum_gather

    def g(x, s_, m_, n_):
        return tsum_gather(x, s_, m_, n_).index_select(0, lane_cell)
    out = {}
    for k, x in plain.items():
        out[k] = g(x * cv, start, mask, n_tiles)
    for k, (x, hp) in snowc.items():
        cvs = g(cv * hp.to(x.dtype), start, mask, n_tiles)
        tot = g(torch.where(hp, x, torch.zeros_like(x)) * cv, start, mask,
                n_tiles)
        out[k] = torch.where(cvs > 0.0,
                             tot / torch.where(cvs > 0.0, cvs,
                                               torch.ones_like(cvs)), tot)
    return out


_cell_sums = fusion.region(_cell_sums_)


def _fill_lanes(srcs, dsts, t, lane_cell, season_t, season_buf):
    """The graph's step inputs at device step index `t`: each [time, cell]
    forcing field's row t at the lanes into its buffer (the gather the
    host made per field), and the season flag. Eager (a compiled region
    would be compiled again for every window length): one gather per
    field from the flattened field, at one index computed once."""
    n = int(srcs[0].shape[1]) if srcs else 0
    idx = t * n + lane_cell
    for s_, d_ in zip(srcs, dsts):
        torch.index_select(s_.view(-1), 0, idx, out=d_)
    season_buf.copy_(season_t.index_select(0, t.reshape(1))[0])
_sub_scatter = fusion.region(_sub_scatter_)


def _clone_tree(x):
    """A copy of `x` (dicts of tensors) with every tensor cloned: a day's
    state handed out while the loop goes on writing into its own."""
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone_tree(v) for k, v in x.items()}
    return x


def _same(x):
    return x


def _zeros_like_tree(x):
    """Zeros with the structure of a snow_step output (tensors in dicts)."""
    if torch.is_tensor(x):
        return torch.zeros_like(x, requires_grad=False)
    if isinstance(x, dict):
        return {k: _zeros_like_tree(v) for k, v in x.items()}
    return x


def _scale_layers(soil_params, init_moist, top=None, bottom=None,
                  middle=None):
    """A copy of `soil_params` (and `init_moist`, [cell, layer] mm) with
    every length of the top, the middle (layers 1..n-2) and/or the bottom
    layer multiplied by a factor.

    Depth, maximum and residual moisture, the critical and wilting points
    and the starting moisture scale together, so the layer stays the same
    soil, only thicker or thinner -- and so does the thermal copy of the
    depths the energy balance reads (`thermal['depth']`), which converts
    moisture to volumetric content for the conductivity.
    """
    import copy
    sp = copy.copy(soil_params)

    def scaled(x):
        if x is None or not torch.is_tensor(x) or x.dim() < 2:
            return x
        f = torch.ones_like(x)
        # Per cell in an ensemble ([cell] against [cell, layer]): cellwise
        # puts the value on the cell axis, not the layer axis.
        t_ = vegetation.cellwise(top if top is not None else 1.0, x)
        b_ = vegetation.cellwise(bottom if bottom is not None else 1.0, x)
        m_ = vegetation.cellwise(middle if middle is not None else 1.0, x)
        parts = [f[..., :1] * t_, f[..., 1:-1] * m_, f[..., -1:] * b_]
        return x * torch.cat(parts, dim=-1)
    for name in ('depth', 'max_moist', 'resid_moist', 'w_crit_mm',
                 'w_wilt_mm'):
        if getattr(sp, name, None) is not None:
            setattr(sp, name, scaled(getattr(sp, name)))
    th = getattr(sp, 'thermal', None)
    if isinstance(th, dict) and 'depth' in th:
        sp.thermal = dict(th, depth=scaled(th['depth']))
    return sp, scaled(init_moist)


def initial_states(n_cells, soil_params, dtype=torch.float64, device=None,
                   init_moist=None):
    """Initial state. `init_moist` is the parameter file's own, in mm.

    VIC starts each layer at `soil_con->init_moist[lidx]`, clipped at
    `max_moist` (generate_default_state.c:75-84). It does not start at a
    fraction of capacity, and the difference is not small: Walnut Gulch
    declares 20.9 / 62.8 / 209.3 mm where 0.7 of capacity is 30.1 / 90.4 /
    301.2, so the old constant put 128.7 mm of water in the column that the
    file never asked for -- 31 per cent of that year's precipitation, in a
    basin whose whole annual evaporation is the question.

    It survived because the runner computed its own opening store with the
    same 0.7 constant, so the water balance was self-consistent by
    construction and closed to 1e-15 around the invented water.

    The fraction remains as the fallback for a dataset that carries no
    `init_moist`, and the caller is told which was used.
    """
    kw = dict(dtype=dtype, device=device)
    if init_moist is not None:
        moist = torch.as_tensor(init_moist, **kw).expand(
            n_cells, soil_params.max_moist.shape[-1]).clone()
        moist = torch.minimum(moist, soil_params.max_moist.expand_as(moist))
    else:
        moist = 0.7 * soil_params.max_moist.expand(n_cells, soil_params.max_moist.shape[-1]).clone()
    zero = torch.zeros(n_cells, **kw)
    snow_state = dict(ice=zero.clone(), surf_water=zero.clone(),
                      pack_water=zero.clone(),
                      # snow->swq as the C carries it (snow_step `sw_pre`)
                      swq_c=zero.clone(),
                      surf_temp=zero.clone(), pack_temp=zero.clone(),
                      # solve_snow.c:471-472: no pack, no depth, no density.
                      # Carried so snow_step compacts the pack and returns
                      # its depth (solve_snow.c:358-372), which is
                      # OUT_SNOW_DEPTH and the conduction term's input.
                      depth=zero.clone(), density=zero.clone(),
                      # snow->coldcontent, 0 at the start
                      # (initialize_snow.c 5.1.0:24); see snow_step
                      cold_content=zero.clone())
    return snow_state, moist.to(**kw)



def _soil_carbon_step(carbon_state, t_node, zwt_lumped_cm, soil_params,
                      npp, dt_seconds, per_second=False):
    """One step of the three soil carbon pools.  soil_carbon_balance.c:40-186,
    called from surface_fluxes.c:1074.

    The arithmetic is `torchvic.core.carbon.soil_carbon_balance`, which is audited
    against VIC's own compiled function. What happens here is the part that
    is about the step loop: choosing which soil thermal nodes lie inside the
    hydrologic column, and giving each one its temperature and its relative
    moisture.

    Under QUICK_FLUX -- VIC's default, and the only thermal path here -- the
    profile is exactly three nodes, at the surface, the first layer's base,
    and the damping depth (read_soilparam.c:716-726). VIC keeps the ones
    inside the column (:63-72): the third sits at the damping depth, which is
    normally metres below the soil, so that is one node or two. Which, is
    decided per cell from the column's own depth rather than assumed.

    `carbon_state` holds the pools and the annual production; it is replaced
    in place, and the step's fluxes are returned.
    """
    from ..core import carbon as _carbon

    th = soil_params.thermal
    depth = th['depth']                                  # [cell, layer], m
    d_tot = depth.sum(-1)
    d1 = depth[..., 0]
    dp = th['dp']

    # read_soilparam.c:716-726, the QUICK_FLUX node grid.
    dz_mm = torch.stack([d1, d1], -1) * 1000.0           # nodes 0 and 1
    dz_cum = torch.cumsum(dz_mm, -1)
    # :63-72 -- node 2 is at the damping depth; it joins only if the column
    # reaches that far, and node 1 only if the column passes the first layer.
    two = dp <= d_tot
    dz_tot = torch.where(two, dz_cum[..., 1], dz_cum[..., 0])

    def node(x, j, like):
        return _bc(x[..., j] if x.dim() > 1 else x, like)

    nodes = lambda f: torch.stack([f(0), f(1)], 0)
    ref = t_node[..., 0]
    dz_n = nodes(lambda j: node(dz_mm, j, ref))
    # A node outside the column contributes no thickness, so its respiration
    # drops out of the sum without a data-dependent shape.
    dz_n = torch.stack([dz_n[0], torch.where(_bc(two, ref), dz_n[1],
                                             torch.zeros_like(dz_n[1]))], 0)
    w = _carbon.node_relative_moisture(
        zwt_lumped_cm,
        nodes(lambda j: node(dz_cum, j, ref)),
        nodes(lambda j: node(soil_params.bubble, 0, ref)),
        nodes(lambda j: node(soil_params.expt, 0, ref)))
    t_nodes = torch.stack([t_node[..., 0], t_node[..., 1]], 0)

    out = _carbon.soil_carbon_balance(
        dz_n, nodes(lambda j: node(dz_cum, j, ref)), _bc(dz_tot, ref),
        dt_seconds, t_nodes, w, carbon_state['c_litter'],
        carbon_state['c_inter'], carbon_state['c_slow'],
        carbon_state['annual_npp_prev'], per_second=per_second)

    for pool in ('CLitter', 'CInter', 'CSlow'):
        carbon_state[{'CLitter': 'c_litter', 'CInter': 'c_inter',
                      'CSlow': 'c_slow'}[pool]] = out[pool]
    # surface_fluxes.c:1077-1080 -- only the steps that fixed carbon count.
    carbon_state['annual_npp'] = (
        carbon_state['annual_npp']
        + torch.clamp(npp, min=0.0) * _carbon.CONST_MWC / 1000.0 * dt_seconds)
    out['NEE'] = npp - out['RhTot']
    return out


def _bc(x, like):
    """A per-cell quantity broadcast onto a per-cell-and-tile one."""
    while x.dim() < like.dim():
        x = x.unsqueeze(-1)
    return x.expand_as(like)


#: read_soilparam.c:1038-1041 -- the soil's albedo in the PAR band, derived
#: from its broadband bare-soil albedo and floored. With VIC's defaults
#: (param.ALBEDO_BARE_SOIL 0.2, param.PHOTO_ALBSOIPARMIN 0) that is
#: 0.92 * 0.2 - 0.015.
ALBEDO_PAR = 0.92 * 0.2 - 0.015


def _wb_thin_pack_(thin, new_state, snow_state, old_t_surf, albedo, tair_t,
                  atm, t, tile, ground_heat, mosaic, month, wind, th, over_t,
                  ov, fc, ra_bare, dt, water_in, swes, snow_depths,
                  snow_covers, sublimations, snow_eb, tile_rows, to_cell,
                  sub_tile, sw_under=None, lw_under=None, melts=None,
                  melt_tile=None, spc=None):
    """A thin pack in water-balance mode: calc_surf_energy_bal at Tsurf=Tair.

    snow_melt.c:322-324 leaves a frozen pack whose surface layer is under
    SNOW_MIN_SWQ_EB_THRES (1 mm) unsolved (surf_temp = 999), and
    surface_fluxes.c:585-591 folds it into the soil surface balance
    (INCLUDE_SNOW). In water-balance mode that balance is not solved: it is
    evaluated once at Tsurf = Tair (calc_surf_energy_bal.c:542-552), so
    func_surf_energy_bal.c gives the pack

        vapour flux   latent_heat_from_snow at TMean = Tair with Ra_used[0]
                      = aero_resist[UnderStory = 2], the snow surface's
                      neutral resistance (:667-684; StabilityCorrection is
                      1 at Tsurf = Tair; the fcanopy average of :703-745)
        deltaCC       CONST_VCPICE_WQ (swq - water) (min(Tair, 0) - OldTSurf)
                      / dt (:643-651)
        refreeze      water * CONST_LATICE * snow density / dt (:652), the
                      surplus instead only when Tair is exactly 0 C and the
                      balance is positive (:871-879)

    and calc_surf_energy_bal.c:643-700 applies them: the vapour flux capped
    at the pack, refreezing up to the liquid there is, melt (to the soil,
    surface_fluxes.c:637-640) only on a negative refreeze energy, the pack's
    surface at min(Tair, 0). Evaporation and transpiration are zero on the
    tile (SNOWING), so nothing else in the balance changes the pack.

    Walnut Gulch, 2008, 3-hourly water balance: VIC sublimates each step's
    trace snowfall off in the step it falls (OUT_SUB_SNOW = OUT_SNOWF on the
    snow steps of day 25). Solving the pack in snow_step instead, with
    everything else in place, left a basin-mean pack of up to 0.065 mm at
    the end of days where VIC's was gone, and a tile with a pack does not
    evaporate: canopy evaporation -0.35%, snow sublimation +5.6% and ET
    -0.024% for the year against VIC, where this gives all three to
    0.0000%.

    `thin` is per tile. `sw_under` and `lw_under` are what reaches the pack
    under an overstory that ran its snow canopy this step (solve_snow.c:190,
    ShortUnderIn *= surf_atten; snow_intercept's LongUnderIn), per tile, the
    sky's elsewhere; None means the sky's everywhere. Updates `snow_state`
    and the step's reported rows in place and returns `water_in` with any
    melt added.
    """
    z = torch.zeros_like(tair_t)
    sel = lambda a, b: torch.where(thin, a, b)
    ice0 = new_state['ice']
    sw0 = new_state['surf_water']
    pw = new_state['pack_water']
    swq = ice0 + pw + sw0  # snow_melt.c 5.1.0:464
    dens = new_state['density']
    safe_dens = torch.where(dens > 0.0, dens, torch.ones_like(dens))
    # CalcAerodynamic.c:97-98 with vic_run.c:245-263's reference height and
    # the soil file's snow roughness; the overstory branch's own Ra[2].
    z0 = mosaic.veg_rough[:, :, month]
    rough = th.get('rough') if th is not None else None
    if rough is not None:
        z0 = torch.where(z0 > 0.0, z0, tile(rough).expand_as(z0))
    d = mosaic.displacement[:, :, month]
    ref_h = energy.reference_height(mosaic.wind_h, d, z0)
    z0s = (th.get('snow_rough') if th is not None else None)
    z0s = (torch.full_like(tair_t, SNOW_Z0) if z0s is None
           else tile(z0s).expand_as(tair_t))
    ra2 = (torch.log((2.0 + z0s) / z0s) * torch.log(ref_h / z0s)
           / VON_KARMAN_SQ / tile(wind))
    if ov is not None and over_t is not None:
        ra2 = torch.where(over_t, ov['ra'][..., 2], ra2)
    if fc is not None and ra_bare is not None:
        # :703 averages only `if (veg_var->fcanopy < 1)`; at exactly 1 the
        # reciprocal round trip would move ra2 by an ulp.
        ra2 = torch.where(fc < 1.0,
                          1.0 / (fc / ra2 + (1.0 - fc) / ra_bare), ra2)
    l_s, ls_s, vmf, _ = snow.latent_heat_from_snow(
        tile(atm['air_dens'][t]), tile(atm['eact_air'][t]),
        tile(atm['lv'][t]), tile(atm['press'][t]), ra2, tair_t,
        tile(atm['vpd'][t]), z)
    v = vmf * dt / snow.CONST_RHOICE
    t_cc = torch.where(tair_t > 0.0, z, tair_t)
    dcc = snow.CONST_VCPICE_WQ * (swq - sw0) * (t_cc - old_t_surf) / dt
    rf = sw0 * snow.CONST_LATICE * dens / dt
    # :859-879 with coverage 1, emissivity 1 (calc_surf_energy_bal.c:213),
    # no sensible heat (Tsurf = Tair), no snow_flux or melt_energy for an
    # included pack, and advection CPFW RHOFW Tair rain / dt, which is 0
    # exactly where the test can pass (Tsnow_surf = Tair = 0).
    long_out = snow.calc_outgoing_longwave(tair_t + CONST_TKFRZ,
                                           energy.EMISS_GRND)
    sw_in = tile(atm['shortwave'][t]) if sw_under is None else tile(sw_under)
    lw_in = tile(atm['longwave'][t]) if lw_under is None else tile(lw_under)
    if spc is None:
        err = ((ground_heat if ground_heat is not None else z)
               + sw_in * (1.0 - albedo) + (lw_in - long_out) + l_s + ls_s
               - dcc)
    else:
        # SPATIAL_SNOW: func_surf_energy_bal.c 5.1.0:617-628 weights the
        # pack's deltaCC and refreeze energy by snow_coverage (the cover
        # solve_snow.c:376 just left, surface_fluxes.c:606), and :779-781
        # its latent heats; the vapour flux is not weighted (:784). The
        # balance (:808-829) adds the bare part's shortwave
        # (calc_surf_energy_bal.c:211-213) to the pack's NetShortSnow,
        # which solve_snow.c:307 formed from AlbedoUnder.
        c_, d_ = spc['c'], spc['d']
        dcc = dcc * c_
        rf = rf * c_
        l_s, ls_s = l_s * c_, ls_s * c_
        ns_bare = (sw_in * (1.0 - (c_ + d_)) * (1.0 - spc['bare_alb'])
                   + sw_in * d_ * (1.0 - albedo))
        err = ((ground_heat if ground_heat is not None else z)
               + sw_in * (1.0 - spc['alb_under']) + ns_bare
               + (lw_in - long_out) + l_s + ls_s - dcc)
    rf = torch.where((tair_t == 0.0) & (err > -rf), -err, rf)
    lat_vol = snow.CONST_LATICE * snow.CONST_RHOFW
    v = torch.maximum(v, -swq)
    swq1 = swq + v
    sw1 = torch.clamp(sw0 + v, min=0.0)
    want = rf / lat_vol * dt
    refrozen = torch.where(rf >= 0.0, torch.minimum(want, sw1), z)
    rf = torch.where((rf >= 0.0) & (want > sw1),
                     refrozen * lat_vol / dt, rf)
    sw2 = torch.clamp(sw1 - refrozen, min=0.0)
    melt = torch.where(rf < 0.0, rf.abs() / lat_vol * dt, z)
    swq2 = swq1 - melt
    melt = torch.where(swq2 < 0.0, melt + swq2, melt)
    swq2 = torch.clamp(swq2, min=0.0)
    gone = swq2 <= 0.0
    keep = lambda x: torch.where(gone, z, x)
    upd = dict(ice=keep(torch.clamp(swq2 - sw2 - pw, min=0.0)),
               surf_water=keep(sw2), pack_water=keep(pw),
               surf_temp=keep(torch.clamp(tair_t, max=0.0)),
               pack_temp=keep(new_state['pack_temp']),
               depth=keep(snow.CONST_RHOFW * swq2 / safe_dens),
               density=keep(dens),
               # calc_surf_energy_bal.c 5.1.0:668-672
               # (a pack that is gone keeps snow_melt's SurfaceCC, which
               # nothing reads before the next snowfall replaces it)
               cold_content=torch.where(
                   gone, snow_state['cold_content']
                   if 'cold_content' in snow_state else z,
                   snow.CONST_VCPICE_WQ * torch.clamp(tair_t, max=0.0)
                   * swq2))
    for k, val in upd.items():
        if k in snow_state:
            snow_state[k] = sel(val, snow_state[k])
    swe_t = (snow_state['ice'] + snow_state['surf_water']
             + snow_state['pack_water'])
    swes[-1] = to_cell(swe_t) * MM_PER_M
    if snow_depths and 'depth' in snow_state:
        snow_depths[-1] = to_cell(snow_state['depth']) * CM_PER_M
    if spc is None:
        snow_covers[-1] = to_cell((swe_t > 0.0).to(swe_t.dtype))
    else:
        # calc_surf_energy_bal.c 5.1.0:671-716: a surviving pack's cover is
        # recomputed from the one solve_snow.c:376 left, with melt plus
        # sublimation (`*melt - vapor_flux`, the flux still signed as a
        # gain) and the zero snowfall solve_snow.c:468 left behind; a pack
        # that is gone has cover 0 and no stored layer, the rest of the
        # distribution untouched (the call is not reached).
        sp = spc['sp']
        sp2 = snow.snow_coverage(
            store_snow=sp['store_snow'], max_slope=spc['slope'],
            old_coverage=spc['c'], swq=swq2, old_swq=swq,
            depth=torch.where(gone, z, snow.CONST_RHOFW * swq2 / safe_dens),
            old_depth=spc['depth'], melt=melt - v,
            max_snow_depth=sp['max_snow_depth'], snowfall=z,
            store_swq=sp['store_swq'],
            snow_distrib_slope=sp['snow_distrib_slope'],
            store_coverage=sp['store_coverage'])
        tg = thin & gone
        tk = thin & ~gone
        for k in sp2:
            sp[k] = torch.where(tk, sp2[k], sp[k])
        sp['coverage'] = torch.where(tg, z, sp['coverage'])
        sp['store_swq'] = torch.where(tg, z, sp['store_swq'])
        snow_covers[-1] = to_cell(sp['coverage'])
        # surface_fluxes.c 5.1.0:910-933: deltaCC, refreeze energy,
        # advection (snow_melt's, :578) and the stale advected sensible
        # heat (:871) reach the record weighted again, by coverage +
        # delta_coverage with the recomputed cover, or by the step-open
        # cover (1 if that was 0 and there was precipitation) where the
        # pack is gone.
        lc = torch.where((spc['cov_prev'] == 0.0) & (spc['prcp'] > 0.0),
                         torch.ones_like(z), spc['cov_prev'])
        thin_f = torch.where(gone, lc, sp['coverage'] + spc['d'])
        tile_rows['advection'] = sel(spc['advection'] * thin_f,
                                     tile_rows['advection'])
        snow_eb['advection'][-1] = to_cell(tile_rows['advection'])
        if 'advected_sensible' in tile_rows:
            adv_t = spc['adv_prev'] * thin_f
            sp['adv_sens'] = sel(adv_t, sp['adv_sens'])
            as_ = -adv_t
            if over_t is not None:
                as_ = torch.where(over_t, z, as_)
            tile_rows['advected_sensible'] = sel(
                as_, tile_rows['advected_sensible'])
            snow_eb['advected_sensible'][-1] = to_cell(
                tile_rows['advected_sensible'])
        dcc = dcc * thin_f
        rf = rf * thin_f
    # put_data.c reports the vapour flux as a loss, OUT_SUB_SNOW.
    sublimations[-1] = to_cell(sel(-v * MM_PER_M, sub_tile))
    if melts is not None and melt_tile is not None:
        # OUT_SNOW_MELT of a thin pack is calc_surf_energy_bal's melt, which
        # overwrites solve_snow's step_melt (surface_fluxes.c 5.1.0:610,
        # :860, :951-952; calc_surf_energy_bal.c 5.1.0:652-665); the rain
        # that ran through the pack is in water_in but is not melt.
        melts[-1] = to_cell(sel(melt * MM_PER_M, melt_tile))
    snow_eb['latent_sub'][-1] = to_cell(sel(-ls_s, tile_rows['latent_sub']))
    snow_eb['deltacc'][-1] = to_cell(sel(dcc, tile_rows['deltacc']))
    snow_eb['rfrz_energy'][-1] = to_cell(sel(rf, tile_rows['rfrz_energy']))
    snow_eb['refreeze'][-1] = to_cell(sel(
        rf / snow.CONST_LATICE * dt,
        tile_rows['rfrz_energy'] / snow.CONST_LATICE * dt))
    tile_rows['deltacc'] = sel(dcc, tile_rows['deltacc'])
    tile_rows['rfrz_energy'] = sel(rf, tile_rows['rfrz_energy'])
    # the pack's latent heat of the evaluation (func_surf_energy_bal.c:779,
    # weighted by the cover), which energy->latent carries: a row only the
    # energy rows (wbenergy) read
    tile_rows['thin_latent'] = sel(l_s, z)
    return water_in + sel(melt * MM_PER_M, z)


_wb_thin_pack_fused = fusion.region(_wb_thin_pack_)
_WB_THIN_SIG = inspect.signature(_wb_thin_pack_)
#: the forcing _wb_thin_pack reads at step t
_WB_THIN_ATM = ('air_dens', 'eact_air', 'lv', 'press', 'vpd', 'shortwave',
                'longwave')
#: its report lists, of which it rewrites the last row
_WB_THIN_ROWS = ('swes', 'snow_depths', 'snow_covers', 'sublimations',
                 'melts')


def _wb_thin_pack(*args, **kwargs):
    """_wb_thin_pack_, on the CUDA graph path one fused region
    (core.fusion). The region sees the report lists as their last row and
    the forcing at step t alone: it would otherwise be compiled anew for
    every list length and every t."""
    if not fusion.wants(args[:1]):
        return _wb_thin_pack_(*args, **kwargs)
    ba = _WB_THIN_SIG.bind(*args, **kwargs)
    ar = ba.arguments
    atm, t = ar['atm'], ar['t']
    ar['atm'] = {k: (atm[k][t],) for k in _WB_THIN_ATM if k in atm}
    ar['t'] = 0
    lists = {}
    for k in _WB_THIN_ROWS:
        if ar.get(k) is not None:
            lists[k] = ar[k]
            ar[k] = [ar[k][-1]] if ar[k] else []
    eb = ar['snow_eb']
    ar['snow_eb'] = {k: ([v[-1]] if v else []) for k, v in eb.items()}
    out = _wb_thin_pack_fused(*ba.args, **ba.kwargs)
    for k, lst in lists.items():
        if lst:
            lst[-1] = ar[k][0]
    for k, v in eb.items():
        if v:
            v[-1] = ar['snow_eb'][k][0]
    return out


def _carbon_fluxes(mosaic, month, lai_t, rc_t, dry_t, snow_t, moist_top,
                   w_max_top, atm_t, elevation, press_t, ncanopy=3,
                   albedo_par=None, rgas=None, rs_t=None):
    """One step of VIC's carbon cycle for every tile of the mosaic.
    surface_fluxes.c:402-438 and :732-790, canopy_evap.c:300-317.

    VIC computes this after the surface energy balance has converged, from
    the canopy resistance that balance left behind. With RC_MODE at its
    default of RC_JARVIS -- which is what VIC 5 ships -- the resistance is
    still Jarvis's, so the carbon step reads the hydrology and never
    changes it: turning CARBON on cannot move a drop of water. That is
    worth stating plainly, because it is the reason this can be added to a
    step loop that already agrees with VIC on every water and energy row.

    `atm_t` supplies par, Catm, fdir and coszen -- the four fields VIC 5
    demands as forcing columns and `torchvic.forcing.daily` derives.

    Returns a dict of per-tile fluxes in gC/m2/s, plus aPAR and Ci.
    """
    from ..core import carbon as _carbon

    if albedo_par is None:
        albedo_par = ALBEDO_PAR
    lib = mosaic.carbon
    dev, dt_ = lai_t.device, lai_t.dtype
    # read_vegparam.c:124-128 -- the canopy is split into equal shares of
    # leaf area, so the bounds are just (cidx + 1) / Ncanopy.
    bnd = ((torch.arange(int(ncanopy), device=dev, dtype=dt_) + 1.0)
           / float(ncanopy)).reshape((int(ncanopy),) + (1,) * lai_t.dim())

    def tile(x):
        return x.unsqueeze(1).expand_as(lai_t) if x.dim() == 1 else x

    coszen, fdir = tile(atm_t['coszen']), tile(atm_t['fdir'])
    par, catm = tile(atm_t['par']), tile(atm_t['Catm'])
    sw, tair = tile(atm_t['shortwave']), tile(atm_t['tair'])

    lai_layer = _carbon.canopy_layer_lai(lai_t, bnd)
    fap = _carbon.fapar(lai_t, bnd, coszen, albedo_par, fdir)
    apar = _carbon.apar_per_leaf(par, fap, lai_layer)
    nscale = _carbon.nscale_factors(lib['NscaleFlag'] > 0.5, bnd, lai_t,
                                    coszen)
    # RC_JARVIS: canopy_evap.c:279-292 divides the whole-canopy resistance
    # by LAI; RC_PHOTO: calc_rc_ps's own per-layer resistances (`rs_t`).
    rs_layer = (_carbon.rs_layer_from_rc(rc_t, lai_t, ncanopy)
                if rs_t is None else rs_t)

    out = _carbon.canopy_assimilation(
        lib['Ctype'], lib['MaxCarboxRate'], lib['MaxETransport'],
        lib['CO2Specificity'], nscale, tair, sw, apar, tile(elevation),
        catm, bnd, lai_t, 'rs', rs_layer=rs_layer, press=press_t,
        **({} if rgas is None else {'rgas': rgas}))

    # canopy_evap.c:258-275 then surface_fluxes.c:758-776: the dry fraction
    # of the canopy, further inhibited when the top soil layer is wet.
    npp_f = _carbon.npp_factor(moist_top, w_max_top, lib['Wnpp_inhib'],
                              lib['NPPfactor_sat'])
    scale = dry_t * npp_f
    fc = (torch.ones_like(lai_t) if mosaic.fcanopy is None
          else mosaic.fcanopy[:, :, month])
    # :733 -- `iveg < Nveg && !step_snow.snow && dryFrac > 0`. A snow-covered
    # or bare or wet-through tile assimilates nothing and its resistances go
    # to HUGE_RESIST, which is what the else branch at :777-788 writes.
    live = (mosaic.cv > 0.0) & (lai_t > 0.0) & (~snow_t) & (dry_t > 0.0)
    scale = torch.where(live, scale * fc, torch.zeros_like(scale))

    res = {}
    for k in ('GPP', 'Rdark', 'Rphoto', 'Rmaint', 'Rgrowth', 'Raut', 'NPP'):
        res[k] = out[k] * scale
    res['aPAR'] = torch.where(live, (par * fap / lai_layer).sum(0),
                              torch.zeros_like(lai_t))
    res['Ci'] = torch.where(live, out['Ci'], torch.zeros_like(lai_t))
    # In "rs" mode canopy_assimilation.c takes the resistance as given and
    # leaves `rc` alone, so there is none to report; surface_fluxes.c:1059
    # overwrites veg_var->rc from the step's own conductances regardless.
    res['NPPfactor'] = npp_f
    return res


def simulate(batch, atm, p, soil_params,
             mosaic=None, wind_h=None, init_moist=None, steps_per_day=1,
             full_energy=False, log_every=0, log=None,
             allow_overstory=True, aero_resist_cansnow='AR_406_FULL',
             grnd_flux_type='GF_410', close_energy=False,
             spatial_snow=False, frozen_soil=False, blowing=False,
             blowing_options=None, carbon=False, carbon_options=None,
             n_frost=1,
             share_layer_moist=True, snow_density='DENS_BRAS',
             progress=False,
             progress_desc='forward', initial_state=None,
             return_state=False, routing_prefix=None, allow_gap=False,
             subdaily=None, snow_steps=None, band_tfactor=None,
             corrprec=None, tfallback=True,
             quick_flux=True, implicit=False, exp_trans=False,
             noflux=False, nnodes=3, quick_solve=False, lakes=None,
             day_hook=None, vicwur_core=None, day_outputs=False,
             tail_from=0, defer_day_check=False, energy_error='solver',
             out_vars=None):
    """Run the whole batch: snow, soil, routing. Returns a dict of outputs.

    `out_vars` names the VIC output variables beyond the ones this function
    always returned that the call forms: None (the default) none, 'all'
    every one, or a set of OUT_* names (a global file's OUTVAR lines; the
    CLI, `Case.run` and the ISC driver pass them). Each extra row costs a
    [step, cell] array for the whole call, so nothing is formed that no name
    asks for; a requested row is the same bit for bit whichever others are
    requested.

    `energy_error` is what out['eb_energy_error'] (OUT_ENERGY_ERROR) holds
    on a full-energy run. 'solver' (the default) is the surface balance's
    residual at the temperature the solver stopped on. 'vic' forms it as
    VIC 5 does (put_data.c:538-552, calc_energy_balance_error): from the
    cell's reported rows, so it is the open balance VIC's own output
    carries (up to ~100 W/m2 where its snow terms do not close). The two
    differ only in the diagnostic; no flux or state changes.

    `return_state=True` adds out['state'], everything the time loop carries
    from one step to the next (see `export_state` below for the list), and
    `initial_state` starts the loop from such a state instead of VIC's cold
    start. A run split into A and B, with B given A's state and B's
    `batch` and `atm` covering B's steps, reproduces the single run A+B bit
    for bit on the CPU in every output over B (tests/
    test_state_carry.py). The state's tensors are used as given: detached
    (`detach_state`) they end the graph there, as after a no-grad warm-up
    (`warm_up`); left attached, B's gradient reaches back through A.
    `init_moist` is ignored when a state is given.

    Discharge is continuous too: the state carries the tail of the routed
    runoff (`routing_prefix`, [steps, cell] of runoff + baseflow in mm),
    which is prepended before the convolution and cut off after it, so B's
    discharge includes what A's runoff was still delivering. Passing
    `routing_prefix` explicitly overrides the state's. The tail is as long
    as A's unit hydrograph, which depends on A's routing parameters (the
    Lohmann length follows route_velocity): a state reused under a slower
    channel than the one that exported it treats the missing history as
    zero. warm_up exports and consumes with the same parameters, so it is
    exact; a spin-up state shared across runs with other parameters is not.

    A state continues from the step after the one it was exported at, and
    a first date that does not follow is refused; `allow_gap=True` says a
    state is being reused deliberately (one spin-up for several windows).

    `p` carries the run parameters as tensors broadcastable over cells:
        pcf         precipitation correction factor      [-]
        t_max_snow  warmest temperature giving snow      [C]
        t_min_rain  coldest temperature giving rain      [C]
        new_snow_albedo                albedo of fresh snow   [-]
        alb_accum_a, alb_accum_b       decay while accumulating [-]
        alb_thaw_a, alb_thaw_b         decay while thawing      [-]
        rs         surface resistance                  [s/m]
        et_beta    soil moisture exponent on ET          [-]
        uh_shape, uh_scale   routing unit hydrograph     [-]

    `corrprec` is `options.CORRPREC` with its inputs, or None (VIC's
    default): {wind_h: the global file's WIND_H, m; gauge_height: m (VIC's
    GAUGE_HEIGHT, 1.0)}. The rain and the snow of every step are scaled by
    the gauge undercatch factors of correct_precip.c from the step's wind
    and the soil file's rough and snow_rough (vic_run.c:106-114,
    solve_snow.c:117-120).

    `share_layer_moist` is `options.SHARE_LAYER_MOIST`, VIC's default TRUE:
    a layer above its critical moisture makes up the shortfall of a rooted
    layer that is below it (canopy_evap.c:287-362). Both cases have always
    been implemented; the flag was not reachable, so a file that turned the
    sharing off still got it. Turning it off is a real difference -- against
    VIC's own year of Walnut Gulch the dry-season transpiration was 1.30 of
    VIC's without it and 1.00 with it.

    Mass bookkeeping follows VIC's snow_melt.c: rain is added to the surface
    liquid store before the energy balance, and leaves as pack outflow. When a
    cell carries no snow the energy balance is skipped and rain reaches the
    soil directly, which is both what VIC does and cheaper than solving a
    balance for a pack that does not exist.

    `tfallback` is `options.TFALLBACK` (initialize_options.c:51, default
    TRUE): a temperature solve that finds no root keeps its previous
    temperature, and out['tfallback'] counts those per solve (canopy
    foliage, snow pack, surface, canopy air; the CUDA-graph path counts
    none). FALSE stops where VIC stops: the first step on which any solve
    fails raises core.tfallback.TemperatureSolveError naming the step, the
    lowest cell and the solves (see core/tfallback.py for the C lines).

    `band_tfactor` ([cell], C) lapses the initial foliage temperature per
    column (generate_default_state.c:106); a snow-band case passes its
    Tfactor, every other run leaves it at None.

    `quick_flux=False` selects the finite-difference soil temperature
    profile (torchvic.core.soiltemp) in place of the two-node closed
    form; `implicit`, `exp_trans`, `noflux`, `nnodes` and `quick_solve`
    are VIC's IMPLICIT, EXP_TRANS, NOFLUX, NODES and QUICK_SOLVE. The
    global-file checks live in torchvic.io.options; what cannot run at
    all is refused here too.

    `vicwur_core` (torchvic.vicwur.core.VicwurCore, default None) carries
    VIC-WUR's per-tile fields of vic_run: layer Ksat and Wcr, the
    vegetation a plugin writes, the b_co2 factor and the stored water
    stress (returned as out['vicwur']). None leaves every path of the
    model exactly as it is; given, it runs on the padded water-balance
    path and is refused where it is not wired.
    """
    # options.CARBON. VIC 5 runs the carbon cycle only on vegetated tiles and
    # only with the four forcing columns it refuses to derive; both are
    # checked here rather than discovered several frames down. With RC_MODE at
    # its VIC default of RC_JARVIS, the carbon step reads the canopy
    # resistance the hydrology already computed and writes nothing back, so
    # switching it on cannot change a single water or energy row.
    _phase = _Phases(atm['tair'].device)
    # which of the out-vars rows to form (torchvic.model.outvars.Wanted)
    _ov = _outvars.Wanted(out_vars)
    _w_aero = _ov(*_outvars.AERO_OUT)
    _w_tnode = _ov('OUT_SOIL_TNODE', 'OUT_SOIL_TNODE_WL')
    _w_band = {k: _ov(n) for k, n in (
        ('adv', 'OUT_ADVECTION_BAND'), ('rfz', 'OUT_RFRZ_ENERGY_BAND'),
        ('lat', 'OUT_LATENT_BAND'), ('lsub', 'OUT_LATENT_SUB_BAND'),
        ('sen', 'OUT_SENSIBLE_BAND'), ('surft', 'OUT_SNOW_SURFT_BAND'),
        ('packt', 'OUT_SNOW_PACKT_BAND'))}
    _w_frost = _ov('OUT_SURF_FROST_FRAC')
    _w_fronts = _ov('OUT_FDEPTH', 'OUT_TDEPTH')
    _w_blow = _ov('OUT_SUB_BLOWING', 'OUT_SUB_SURFACE')
    _w_zwtl = _ov('OUT_ZWT_LUMPED')
    _w_wb = _ov(*_outvars.WB_ENERGY_OUT)
    co = dict(carbon_options or {})
    carbon_ncanopy = int(co.pop('ncanopy', 3))
    carbon_albedo_par = co.pop('albedo_par', ALBEDO_PAR)
    # options.RC_MODE. RC_PHOTO takes the canopy resistance from
    # photosynthetic demand (canopy_evap.c:296-307, penman.c calc_rc_ps), so
    # unlike RC_JARVIS it changes the water and energy balance.
    _rcm = str(co.pop('rc_mode', 'RC_JARVIS')).upper()
    if _rcm not in ('RC_JARVIS', 'RC_PHOTO'):
        raise ValueError("carbon_options['rc_mode'] must be 'RC_JARVIS' or "
                         "'RC_PHOTO'; got %r" % (_rcm,))
    rc_photo = _rcm == 'RC_PHOTO'
    if mosaic is not None:
        mosaic.photo = None
        mosaic.rs_last = None
        mosaic.vicwur = None
    if vicwur_core is not None:
        # torchvic.vicwur.core: wired on the padded water-balance step only
        _why = [w for w, bad in (
            ('no vegetation mosaic', mosaic is None),
            ('FULL_ENERGY TRUE', bool(full_energy)),
            ('sub-daily snow steps of a daily step', subdaily is not None),
            ('LAKES', lakes is not None),
            ('CLOSE_ENERGY', bool(close_energy)),
            ('FROZEN_SOIL', bool(frozen_soil)),
            ('RC_PHOTO', rc_photo)) if bad]
        if _why:
            raise NotImplementedError(
                'vicwur_core hooks are wired on the water-balance path only; '
                'this run has %s' % ', '.join(_why))
    if rc_photo and not carbon:
        # get_global_param.c:849-853 falls back to RC_JARVIS with a warning;
        # a caller here asked for it by name, so say no instead
        raise ValueError('RC_MODE RC_PHOTO needs CARBON (VIC resets it to '
                         'RC_JARVIS without: get_global_param.c:849-853)')
    # VIC 5 substituted a per-kilomole gas constant into photosynth.c's
    # per-mole formulas in all nine places (see torchvic.core.carbon.photosynth).
    # The default reproduces VIC 5, because that is this model's contract;
    # 'per_mol' is VIC 4.2.d's arithmetic and the units the activation
    # energies are stated in.
    # VIC reads the three pools and last year's production from its state
    # file (read_initial_model_state.c); a run that starts from bare ground
    # has them at zero (initialize_soil.c:46-48) and grows no litter until a
    # year has passed. Both are legitimate starting points and neither is a
    # default anyone should get by accident, so the zero start is the
    # default and a restart says so.
    _init_pools = co.pop('initial_pools', None)
    _init_npp_prev = co.pop('annual_npp_prev', None)
    _rg = co.pop('gas_constant', 'vic5')
    if _rg not in ('vic5', 'per_mol'):
        raise ValueError("carbon_options['gas_constant'] must be 'vic5' or "
                         "'per_mol'; got %r" % (_rg,))
    # compute_soil_resp.c:180-192 divides the litter pool's turnover time by
    # SEC_PER_DAY/dt and the other two by HOURS_PER_DAY/dt, with dt in
    # seconds. Reproduced by default; 'per_second' is what the litter line
    # implies for all three.
    _sr = co.pop('soil_respiration', 'vic5')
    if _sr not in ('vic5', 'per_second'):
        raise ValueError("carbon_options['soil_respiration'] must be 'vic5' "
                         "or 'per_second'; got %r" % (_sr,))
    carbon_resp_per_second = _sr == 'per_second'
    from ..core import carbon as _carbon_mod
    carbon_rgas = (None if _rg == 'vic5' else _carbon_mod.PHOTO_RGAS_PER_MOL)
    if co:
        raise ValueError('unknown carbon_options: %s' % ', '.join(sorted(co)))
    # options.SNOW_DENSITY, initialize_options.c 5.1.0:48. snow_step raises
    # on an unknown value too, but only on a step that compacts a pack;
    # refusing here fails fast whatever the weather.
    if snow_density not in ('DENS_BRAS', 'DENS_SNTHRM'):
        raise ValueError("snow_density must be 'DENS_BRAS' or 'DENS_SNTHRM'; "
                         'got %r' % (snow_density,))
    do_carbon = bool(carbon)
    carbon_out = {}
    carbon_state = None
    # options.CARBON also carries three soil pools. They need a soil thermal
    # node temperature and the lumped water table, which only the energy
    # balance and a soil file with `bubble` provide; a run without them gets
    # the vegetation fluxes and no pools, said rather than silently skipped.
    do_soil_carbon = False
    if do_carbon:
        if mosaic is None or getattr(mosaic, 'carbon', None) is None:
            raise ValueError('CARBON needs the vegetation library\'s carbon '
                             'columns (Ctype, MaxCarboxRate, MaxETransport, '
                             'CO2Specificity, NscaleFlag, Wnpp_inhib, '
                             'NPPfactor_sat); read the library with '
                             'VEGLIB_PHOTO on (read_veglib.c:162-192)')
        missing = [k for k in ('par', 'Catm', 'fdir', 'coszen')
                   if k not in atm]
        if missing:
            raise ValueError('CARBON needs %s in the forcing '
                             '(vic_force.c:165-178). VIC 5 refuses to derive '
                             'them; torchvic.forcing.daily.daily_to_subdaily'
                             '(carbon=True) builds them as VIC 4.2.d did'
                             % ', '.join(missing))

    nt, nc = atm['tair'].shape
    dtype, device = atm['tair'].dtype, atm['tair'].device
    # VIC 5 has no configuration that reads a daily forcing *file* -- the snow
    # step is never 1 and the forcing is consumed at it -- but it does run a
    # daily *model* step: MODEL_STEPS_PER_DAY 1, water balance only, the
    # image driver averaging the day's records into one (vic_force.c:404-412)
    # and canopy_evap.c:108-110 letting that day's rain evaporate off the
    # canopy as it lands. That is a different model from VIC at 8 or 24 steps
    # a day (Walnut Gulch canopy evaporation 180 mm a year daily against 49
    # 3-hourly, both VIC 5.0.1 in water balance), not a different code path:
    # steps_per_day = 1 here is VIC's daily water balance, measured against
    # vic_image.exe at +0.0006% ET on basin 37, 2001, and pinned on Walnut
    # Gulch by tests/test_wb_vs_vic.py. A day with a pack, canopy snow or
    # snowflag is not that average: VIC loops hidx 0..NF-1 at snow_dt and
    # calls runoff once on the summed throughfall (surface_fluxes.c:258-269,
    # :954, :1080). `subdaily` is those records, [day, snow step, cell].
    # Without it, and on every snow-free day, the step is the daily mean.
    step_seconds = SEC_PER_DAY / float(steps_per_day)
    if subdaily is not None:
        _sd_t = subdaily.get('tair')
        if (not torch.is_tensor(_sd_t) or _sd_t.dim() != 3
                or _sd_t.shape[0] != nt or _sd_t.shape[2] != nc):
            raise ValueError('subdaily["tair"] must be [day, snow step, '
                             'cell] with %d days and %d cells; got %s'
                             % (nt, nc, None if _sd_t is None
                                else tuple(_sd_t.shape)))
        _nf = int(_sd_t.shape[1])
        if snow_steps is not None and int(snow_steps) != _nf:
            raise ValueError('snow_steps %s does not match subdaily\'s %d'
                             % (snow_steps, _nf))
        if 'prcp' not in subdaily:
            raise ValueError('subdaily has no prcp; will_it_snow reads it '
                             '(forcing_utils.c:86-100)')
    if subdaily is not None and mosaic is not None:
        # The image driver steps the vegetation at veg_hist[NR], the NF
        # sub-steps' mean of the month's value: vic_force.c:247-259 fills
        # NF identical copies and :466-474 averages them with
        # forcing_utils.c:13-31's sequential sum over n. The mean of NF
        # identical doubles is not the value itself -- 8 copies of
        # fcanopy 0.599 average to 0.59900000000000009 -- so a daily
        # step runs LAI, fcanopy, albedo, displacement and roughness a
        # ulp off the file, and the per-plant LAI, MaxInt, aerodynamic
        # resistance and interception downstream with them (himal
        # 2004-01-01 cell 30 tile 5: LAI 3.686143572621035 against
        # 3.6861435726210354 un-averaged). The mean is elementwise over
        # the month table, so it is formed once per run, in the C's
        # order. With one forcing record per model step average()
        # returns ar[0] untouched, so sub-daily runs never reach this.
        _nfavg = int(subdaily['tair'].shape[1])
        if _nfavg > 1:
            mosaic = copy.copy(mosaic)
            for _k in ('lai', 'fcanopy', 'albedo', 'displacement',
                       'veg_rough'):
                _v = getattr(mosaic, _k, None)
                if _v is None or not torch.is_tensor(_v):
                    continue
                _a = torch.zeros_like(_v)
                for _ in range(_nfavg):
                    _a = _a + _v
                setattr(mosaic, _k, _a / float(_nfavg))
    # Interception capacity per unit LAI, mm (VIC's VEG_LAI_WATER_FACTOR,
    # 0.1 by default). A run parameter so it can be varied: on one Indian
    # basin the canopy evaporated 0.97 mm/day of intercepted rain in the
    # 2002 monsoon against GLEAM's 0.14, and nothing else could reduce it
    # without also cutting transpiration, which already matched.
    lwf = p.get('lai_water_factor', vegetation.VEG_LAI_WATER_FACTOR)
    _snow_zero = {}                 # see the snow-free fast path below
    # The bottom soil layer's thickness, as a multiple of the file's (VIC's
    # d3). Everything that is a length of
    # that layer scales with it -- depth, maximum and residual moisture, the
    # critical and wilting points, the starting moisture -- so the layer
    # stays the same soil, only deeper. On one Indian basin the model's 1.0 m
    # column dried out within weeks of the 2002 monsoon (ET ~1 mm/day in
    # October against GLEAM's ~2) while it ran off water a deeper store
    # would have kept. Absent, nothing changes.
    # And the top layer's (d1): ESA CCI senses ~2-5 cm and the file's top
    # layer is 10 cm, and a thicker layer wets and dries more slowly --
    # run4's top-layer moisture varied 0.68x as much as the satellite's.
    if ('bottom_depth_scale' in p or 'top_depth_scale' in p
            or 'middle_depth_scale' in p):
        soil_params, init_moist = _scale_layers(
            soil_params, init_moist, top=p.get('top_depth_scale'),
            bottom=p.get('bottom_depth_scale'),
            middle=p.get('middle_depth_scale'))
    # LANE_COMPACT: the loop runs on the covered tiles, each a "cell" of one
    # tile (torchvic.model.lanes). The cell-axis objects are kept for the
    # routing and the outputs after the loop.
    lanes = None
    # the cell-level soil parameters, for the outputs the tail forms on the
    # cell axis (the lanes replace soil_params with the per-tile ones)
    cell_soil_params = soil_params
    # On a CUDA device only: there every elementwise kernel computes an
    # element the same way wherever it sits in the tensor. The CPU's
    # vectorised transcendentals (SLEEF) do not -- a vector's lanes and the
    # scalar tail round differently -- so moving an element changes it.
    if (LANE_COMPACT and mosaic is not None
            and (vicwur_core is None or VW_LANES)
            and (not full_energy or _eb_lanes_ok(quick_flux, close_energy,
                                                 lakes, subdaily))
            and (torch.device(device).type == 'cuda' or SYNC_CHECK)
            and not spatial_snow and not blowing and not carbon
            and (not frozen_soil
                 or (full_energy and quick_flux and EB_LANES_FS))
            and not torch.is_grad_enabled()):
        _ln = _lanes_mod.Lanes(mosaic.cv)
        _lm = _ln.mosaic(mosaic) if _ln.complete else None
        _ls = _ln.soil(soil_params) if _lm is not None else None
        _lp = _ln.params(p) if _ls is not None else None
        if _lp is not None and (init_moist is None
                                or _ln.is_cell(init_moist)):
            lanes = _ln
            cell_batch, cell_atm, cell_mosaic, cell_nc = batch, atm, mosaic, nc
            batch = lanes.batch(batch)
            atm = lanes.atmosphere(atm)
            mosaic, soil_params, p = _lm, _ls, _lp
            if init_moist is not None:
                init_moist = lanes.gather_cell(init_moist)
            if band_tfactor is not None:
                # per column (cell) -> per lane: each covered tile takes its
                # cell's Tfactor, as the padded grid broadcasts it over the
                # cell's tiles (generate_default_state.c:106)
                _tfb = torch.as_tensor(band_tfactor, dtype=dtype,
                                       device=device)
                if _tfb.dim() != 1 or _tfb.shape[0] != nc:
                    raise ValueError(
                        'band_tfactor must be [%d columns]; got %s'
                        % (nc, tuple(_tfb.shape)))
                band_tfactor = lanes.gather_cell(_tfb)
            nc = lanes.n_lanes
            lwf = p.get('lai_water_factor', vegetation.VEG_LAI_WATER_FACTOR)
            if initial_state is not None:
                _il = initial_state.get('lanes')
                if _il is None:
                    initial_state = dict(initial_state)
                    for _k in _CARRIED_STATE:
                        initial_state[_k] = lanes.state_to_lanes(
                            initial_state.get(_k))
                elif int(_il['n_lanes']) != lanes.n_lanes:
                    raise ValueError(
                        'initial_state is on %d lanes (covered tiles); this '
                        'run has %d' % (_il['n_lanes'], lanes.n_lanes))
    if lanes is None and initial_state is not None \
            and initial_state.get('lanes') is not None:
        raise ValueError(
            'initial_state was exported by a run on the covered tiles only '
            '(LANE_COMPACT), and this run is on the padded tile grid; run '
            'both the same way')
    _tsum = (lambda x: x.sum(1)) if lanes is None else lanes.tsum
    snow_state, moist = initial_states(nc, soil_params, dtype, device,
                                       init_moist=init_moist)
    # VIC keeps a snow pack per vegetation tile (snow[iveg][band] in
    # vic_run.c), not one per grid cell. It matters wherever the tiles do
    # not all see the same snow: under a forest the canopy intercepts what
    # an open tile receives in full, and one shared pack gives the open
    # ground the forest's shading and the forest the open ground's melt.
    # Measured on 90 winter days of FindleyLake with one pack per cell:
    # sublimation +167% and evaporation +88% against VIC.
    #
    # The soil column is already per tile, so tile k's melt now reaches
    # tile k's column, which is what VIC does. Where every tile sees the
    # same weather -- no overstory, uniform precipitation -- the packs stay
    # identical and the cover-weighted mean is what one pack gave.
    per_tile_snow = mosaic is not None
    if per_tile_snow:
        ntl_s = mosaic.n_tiles
        snow_state = {k: v.unsqueeze(-1).expand(nc, ntl_s).clone()
                      for k, v in snow_state.items()}
    # The height the global file declares its wind at (WIND_H). Falling back
    # to 2 m where a file says 10 m understates ln(z/z0)^2 by 1.91.
    z_wind = MEAS_HEIGHT if wind_h is None else wind_h

    # Snow age in time steps since the last snowfall, and VIC's MELTING flag.
    # Both are state: a pack's albedo depends on how long since it last snowed
    # and on whether the pack is warming or cooling, and neither can be
    # recovered from the current step alone.
    # The albedo's age and melting flags follow the pack, so they take its
    # shape: one per tile when the packs are per tile.
    _snow_shape = ((nc, mosaic.n_tiles) if per_tile_snow else (nc,))
    # The albedo of fresh snow. MAX_SNOW_ALBEDO gives every vegetation tile
    # its class's max_snow_albedo and the bare-soil tile (iveg == Nveg) the
    # run's value (surface_fluxes.c:524-534); otherwise one number for all.
    _gauge = None
    if corrprec:
        if subdaily is not None:
            raise ValueError(
                'CORRPREC with a daily step over a sub-daily forcing is not '
                'implemented: the snow sub-steps would need their own wind')
        _th_g = getattr(soil_params, 'thermal', None) or {}
        if _th_g.get('rough') is None or _th_g.get('snow_rough') is None \
                or 'wind' not in atm:
            raise ValueError(
                'CORRPREC needs the soil file\'s rough and snow_rough and a '
                'wind forcing (vic_run.c:106-114)')
        _gauge = dict(wind_h=float(corrprec['wind_h']),
                      height=float(corrprec.get('gauge_height', 1.0)),
                      rough=_th_g['rough'], snow_rough=_th_g['snow_rough'])
    new_snow_alb = p['new_snow_albedo']
    if mosaic is not None and mosaic.max_snow_albedo is not None:
        _msa = mosaic.max_snow_albedo
        new_snow_alb = (_msa if mosaic.bare_tile is None else torch.where(
            mosaic.bare_tile, torch.zeros_like(_msa) + new_snow_alb, _msa))
    # SPATIAL_SNOW (options.SPATIAL_SNOW, off by default in VIC too --
    # initialize_options.c:70). With it on, a tile's pack is spread over the
    # tile with a linear depth distribution rather than covering all of it,
    # and the covered fraction falls continuously as the shallow end melts
    # out. `sp` is the six-variable state snow_data_struct carries for it,
    # all zero and store_snow false at the start (initialize_snow.c:45-63).
    # FROZEN_SOIL, the QUICK_FLUX path. With SPATIAL_FROST off -- VIC's
    # default -- Nfrost is 1, frost_fract[0] is 1 and frost_slope is 0, so
    # soil_conduction.c:659-688 collapses to one line per layer:
    #     ice = clamp(moist - maximum_unfrozen_water(T_layer), 0, moist)
    # with T_layer the quick-flux layer temperature frozen_soil.c:75-81
    # computes. Everything else FROZEN_SOIL does -- the ice reducing both
    # the capacity and the residual the liquid moves between -- is in
    # column.step.
    # BLOWING. surface_fluxes.c:443-463 calls CalcBlowingSnow on a tile with
    # no overstory and a pack, and its mass flux joins the pack's own
    # sublimation in SnowPackEnergyBalance.c:217-235. The three parameters
    # it needs -- the slope's standard deviation, the topography's lag-one
    # autocorrelation and the fetch -- are the three fields read_vegparam.c
    # :444-456 adds to every tile line under the option.
    # BLOWING's five sub-options, at VIC's own defaults
    # (initialize_options.c:45-49) unless the caller passes the global
    # file's. A runner that reads them hands them through here.
    _blow_opt = dict(var_threshold=True, calc_prob=True, simple=False,
                     fetch=True, spatial_wind=True)
    if blowing_options:
        unknown = set(blowing_options) - set(_blow_opt)
        if unknown:
            raise ValueError('unknown blowing option(s): %s; the five are '
                             '%s' % (', '.join(sorted(unknown)),
                                     ', '.join(sorted(_blow_opt))))
        _blow_opt.update(blowing_options)
    if blowing:
        if mosaic is None or mosaic.sigma_slope is None:
            raise ValueError(
                'blowing=True needs the vegetation file\'s sigma_slope, '
                'lag_one and fetch, the three fields read_vegparam.c'
                ':444-456 adds to each tile line under BLOWING. The classic '
                'reader supplies them when the global file sets BLOWING '
                'TRUE; without them there is no topography to blow snow '
                'over.')
    soil_ice = None
    t_node = None
    wb_t0 = wb_t1 = None
    # QUICK_FLUX FALSE: the static node grid (fd_static, set once below)
    # and the node moisture, ice, conductivity and heat capacity
    # (node_props, redistributed after every column step).
    fd_static = None
    node_props = None
    if frozen_soil:
        if mosaic is None:
            raise ValueError(
                'frozen_soil=True needs a vegetation mosaic: the layer '
                'temperatures it freezes against come from it '
                '(frozen_soil.c:75-81).')
        if getattr(soil_params, 'bubble', None) is None:
            raise ValueError(
                'frozen_soil=True needs soil_params.bubble, the bubbling '
                'pressure the unfrozen-water curve is built from '
                '(soil_conduction.c:788). A parameter file without it '
                'cannot run FROZEN_SOIL.')
    # SPATIAL_FROST (get_global_param.c:221-229, :833-843): the sub-grid
    # frost distribution over n_frost areas. With one area -- VIC's
    # default -- or without frozen soil every area is the same column
    # and the run below is the one-area run exactly.
    if (not isinstance(n_frost, int) or isinstance(n_frost, bool)
            or n_frost < 1 or n_frost > MAX_FROST_AREAS):
        raise ValueError(
            'n_frost must be an integer 1..%d (SPATIAL_FROST TRUE n, '
            'get_global_param.c:833-843); got %r'
            % (MAX_FROST_AREAS, n_frost))
    _frost = None
    if n_frost > 1:
        _fslope = getattr(soil_params, 'frost_slope', None)
        if _fslope is None:
            raise ValueError(
                'n_frost > 1 needs soil_params.frost_slope, the per-cell '
                'width (C) of the soil-temperature distribution the frost '
                'areas read (vic_init.c:703-710). The image reader '
                'supplies it when the parameter file carries frost_slope.')
        _fslope = torch.as_tensor(_fslope, dtype=dtype, device=device)
        if bool((_fslope < 0.0).any()):
            raise ValueError(
                'frost_slope must be >= 0 everywhere (VIC stops, '
                'vic_init.c:808-811)')
        from ..core import frozen as _frzmod
        _frost = dict(n=n_frost, slope=_fslope,
                      fract=_frzmod.frost_fractions(n_frost),
                      fract_t=torch.as_tensor(
                          _frzmod.frost_fractions(n_frost), dtype=dtype,
                          device=device))

    def _ice_eff():
        """The ice the surface sees: per-area ice frost-weighted
        (canopy_evap.c:205-206, arno_evap.c:50-54,
        soil_conduction.c:699-704, put_data.c:672-682), or the ice
        itself with one area."""
        if _frost is None or soil_ice is None:
            return soil_ice
        return (soil_ice * _frost['fract_t']).sum(-1)
    sp = None
    if spatial_snow:
        _slope = getattr(soil_params, 'max_snow_distrib_slope', None)
        if _slope is None:
            raise ValueError(
                'spatial_snow=True needs soil_params.max_snow_distrib_slope, '
                'the soil-file column read_soilparam.c:542-552 reads under '
                'SPATIAL_SNOW (classic) or the parameter file variable '
                'vic_init.c:691-700 reads (image). Both readers supply it '
                'when the global file sets SPATIAL_SNOW TRUE; without it '
                'there is no distribution to spread the pack over.')
        _slope = torch.as_tensor(_slope, dtype=dtype, device=device)
        while _slope.dim() < len(_snow_shape):
            _slope = _slope.unsqueeze(-1)
        sp_slope = _slope.expand(_snow_shape)
        _z = torch.zeros(_snow_shape, dtype=dtype, device=device)
        # adv_sens: energy->advected_sensible as the last step left it
        # (surface_fluxes.c 5.1.0:977), which a pack folded into the next
        # step's surface solve reports again (:236, :871); zero at a cold
        # start and after a VIC restart (initialize_energy)
        sp = dict(coverage=_z.clone(), store_swq=_z.clone(),
                  snow_distrib_slope=_z.clone(), max_snow_depth=_z.clone(),
                  store_coverage=_z.clone(), adv_sens=_z.clone(),
                  store_snow=torch.zeros(_snow_shape, dtype=torch.bool,
                                         device=device))
    last_snow = torch.zeros(_snow_shape, dtype=dtype, device=device)
    # One month index per model step. `batch.dates` is whatever axis the
    # forcing is on, so this is right for a daily run and for a sub-daily one;
    # repeating a daily month list would silently apply January's leaf area to
    # every step of a sub-daily February.
    months = [d.month - 1 for d in batch.dates]
    # The day of year, for the one thing in the step loop that turns over on
    # the first of January: vic_run.c:301-306 rolls the running annual net
    # primary production into last year's, which feeds the next year's
    # litterfall.
    ydays = [d.timetuple().tm_yday for d in batch.dates]
    if len(months) != nt:
        raise ValueError('the date axis has %d entries and the forcing %d '
                         'steps; they must be the same axis'
                         % (len(months), nt))
    # A vegetation history (FORCING2, *_SRC FROM_VEGHIST) is one value per
    # model step, not per month: the fields are [cell, tile, step] and the
    # step is the index (vic_force.c:264-363, update_step_vars.c:35-44).
    step_axis = bool(getattr(mosaic, 'step_axis', False))
    if step_axis:
        months = list(range(nt))
    # Canopy storage per tile, carried across steps. Dry at the start.
    wdew = mosaic.initial_wdew() if mosaic is not None else None
    # The canopy snow of covered overstory tiles (snow_intercept.c via
    # canopy.canopy_step). State per tile; the intercepted rain is the canopy's
    # Wdew, which the C shares with the rain interception (veg_var->Wdew), so
    # it is `wdew` here too.
    over_t = (((mosaic.overstory > 0.5) & (mosaic.cv > 0.0))
              if mosaic is not None and getattr(mosaic, 'overstory', None)
              is not None else None)
    any_over = over_t is not None and bool(over_t.any())
    if any_over:
        missing = [k for k in ('veg_rough', 'displacement', 'wind_atten',
                               'trunk_ratio', 'albedo')
                   if getattr(mosaic, k, None) is None]
        if missing:
            # A mosaic without the canopy's inputs (a test's bare mosaic;
            # every parameter file carries them) runs as before, without
            # the canopy stage, and says so.
            (log or print)('overstory tiles present but the mosaic carries no %s: '
                 'canopy snow (snow_intercept.c) not run' % ', '.join(missing))
            any_over = False
    ground_long_out = None
    canopy_state = None
    if any_over:
        ntl0 = mosaic.n_tiles
        canopy_state = dict(
            int_snow=torch.zeros(nc, ntl0, dtype=dtype, device=device),
            t_foliage=atm['tair'][0].unsqueeze(1).expand(nc, ntl0).clone(),
            temp_int_storage=torch.zeros(nc, ntl0, dtype=dtype, device=device),
            # energy->canopy_advection / canopy_refreeze as the last step
            # left them (surface_fluxes.c:235, :495, :978)
            carry_adv=torch.zeros(nc, ntl0, dtype=dtype, device=device),
            carry_rfz=torch.zeros(nc, ntl0, dtype=dtype, device=device))
        ground_long_out = None        # the previous step's ground emission
        aero_406 = aero_resist_cansnow in ('AR_406', 'AR_406_LS', 'AR_406_FULL')
        aero_406_full = aero_resist_cansnow in ('AR_406_FULL', 'AR_410')
        # AR_406 alone undoes the tenfold resistance before the
        # canopy's sensible heat (func_canopy_energy_bal.c:225-227).
        aero_406_exact = aero_resist_cansnow == 'AR_406'
        # AR_410 stability-corrects the snow branch's resistance at the
        # trial foliage temperature (func_canopy_energy_bal.c:178-188).
        aero_410 = aero_resist_cansnow == 'AR_410'
    canopy_long_out = None
    snow_canopies, sub_canops, canopy_terms = [], [], []
    _tfx.start(tfallback)

    # Soil thermal state for the surface energy balance, per tile: the three
    # node temperatures. generate_default_state.c:110-111 starts every node
    # at avg_temp when FULL_ENERGY is on -- not at zero, and not at an
    # invented fraction of anything. Ts_old is T[0] (calc_surf_energy_bal.c
    # :179), so the surface starts there too.
    if do_carbon:
        do_soil_carbon = bool(full_energy) and soil_params.bubble is not None
    use_eb = full_energy and mosaic is not None
    if close_energy and not use_eb:
        # VIC stops: CLOSE_ENERGY TRUE needs FULL_ENERGY TRUE
        # (get_global_param.c 5.1.0:961-964).
        raise ValueError('close_energy needs the surface energy balance '
                         '(full_energy with a mosaic); VIC stops on '
                         'CLOSE_ENERGY TRUE with FULL_ENERGY FALSE')
    if close_energy and spatial_snow:
        raise ValueError('close_energy with SPATIAL_SNOW is not run: the '
                         'partial cover the iteration re-weights is not '
                         'carried')
    # The previous step's per-tile (snow_flux, grnd_flux, deltaH, fusion)
    # in VIC's internal signs, which seeds the ground/canopy iteration's
    # snow_flux (surface_fluxes.c:225-230). Zeros at cold start
    # (initialize_energy.c:73-98), carried only with CLOSE_ENERGY.
    _vic_aux = (dict(initial_state.get('vic_aux') or {})
                if initial_state is not None else {})
    ce_seed = None
    if close_energy and use_eb:
        ce_seed = torch.zeros(nc, mosaic.n_tiles, 4, dtype=dtype,
                              device=device)
    if any_over:
        # generate_default_state.c 5.1.0:94-107: every node starts at
        # avg_temp with FULL_ENERGY or FROZEN_SOIL and at 0 C otherwise,
        # Tfoliage = T[0] + Tfactor (Tfactor 0 without bands), and the
        # ground's LongUnderOut the first snow_intercept sees is
        # calc_outgoing_longwave(T[0] + CONST_TKFRZ, EMISS_SNOW). This
        # started the canopy at the first step's air temperature and the
        # ground's emission at Tair.
        _th0 = getattr(soil_params, 'thermal', None)
        if ((full_energy or frozen_soil) and _th0 is not None
                and _th0.get('avg_temp') is not None):
            _t0 = torch.as_tensor(_th0['avg_temp'], dtype=dtype,
                                  device=device).unsqueeze(1).expand(
                nc, mosaic.n_tiles)
        else:
            _t0 = torch.zeros(nc, mosaic.n_tiles, dtype=dtype, device=device)
        if band_tfactor is not None:
            # generate_default_state.c:106: the foliage temperature
            # starts at T[0] plus the band's Tfactor, per band. The
            # ground's emission above stays at T[0], as VIC's does.
            # (on the lanes, band_tfactor is already per lane: one tile)
            _tfb = torch.as_tensor(band_tfactor, dtype=dtype,
                                   device=device)
            if _tfb.dim() != 1 or _tfb.shape[0] != nc:
                raise ValueError(
                    'band_tfactor must be [%d columns]; got %s'
                    % (nc, tuple(_tfb.shape)))
            canopy_state['t_foliage'] = _t0 + _tfb.reshape(nc, 1).expand(
                nc, mosaic.n_tiles)
        else:
            canopy_state['t_foliage'] = _t0.clone()
        ground_long_out = snow.calc_outgoing_longwave(
            _t0 + CONST_TKFRZ, snow.EMISS_SNOW)
    if use_eb:
        # The surface energy balance in torchvic.core.energy is the no-overstory
        # branch: no canopy energy solve, no canopy wind profile, no snow
        # interception. A forest tile with cover would be solved as if it
        # were grass, and every balance would still close. FindleyLake's
        # dominant tile is 82 per cent evergreen with overstory = 1. A
        # configuration with overstory is supported: its covered tiles take the
        # overstory aerodynamic resistances and the canopy stage (canopy snow
        # via canopy.canopy_step) in addition to this balance. With
        # allow_overstory=False the run refuses by name instead of running the
        # wrong physics quietly.
        if getattr(mosaic, 'overstory', None) is not None and not allow_overstory:
            # the guard, kept for a caller that asks for the refusal
            bad = (mosaic.overstory > 0.5) & (mosaic.cv > 0.0)
            if bool(bad.any()):
                cells = torch.nonzero(bad.any(1)).flatten().tolist()
                tiles = torch.nonzero(bad.any(0)).flatten().tolist()
                raise NotImplementedError(
                    'FULL_ENERGY on a tile with overstory: %d cell(s) '
                    '(e.g. %s) carry cover on overstory class(es) %s, and '
                    'the surface energy balance here solves only the '
                    'no-overstory branch. Run with the water-balance path '
                    '(--no-energy) or wait for the overstory phase; the '
                    'answer would otherwise be a forest computed as grass '
                    'with every balance closing.'
                    % (len(cells), cells[:5], tiles))
        th = soil_params.thermal
        if not quick_flux:
            # get_global_param.c:939-945: QUICK_FLUX FALSE with neither
            # FULL_ENERGY nor FROZEN_SOIL stops VIC; the frozen-soil
            # water balance is refused separately below, so reaching
            # here without the energy balance cannot run.
            if not use_eb:
                raise ValueError(
                    'quick_flux=False needs full_energy=True: VIC stops '
                    'on QUICK_FLUX FALSE in the water balance without '
                    'frozen soil (get_global_param.c:939-945), and the '
                    'frozen-soil water balance is not implemented.')
            if (getattr(soil_params, 'bubble', None) is None
                    or getattr(soil_params, 'expt', None) is None):
                raise ValueError(
                    'quick_flux=False needs soil_params.bubble and expt, '
                    'the unfrozen-water curve the node ice is built from '
                    '(soil_conduction.c:246-265). The parameter readers '
                    'supply both from the file VIC reads them from.')
            if implicit and not noflux and (nnodes == 3 or quick_solve):
                raise ValueError(
                    'IMPLICIT with NOFLUX FALSE solves for a single node '
                    'on a 3-node column (QUICK_SOLVE can shrink any '
                    'column to 3), where VIC reads T_2[1] past its '
                    'unknowns (frozen_soil.c:793-796): uninitialized '
                    'memory, not physics. Use NOFLUX TRUE, NODES >= 4, '
                    'or IMPLICIT FALSE.')
        _nn = nnodes if (use_eb and not quick_flux) else 3
        t_node = th['avg_temp'][:, None, None].expand(
            nc, mosaic.n_tiles, _nn).clone()
        # the layers' cumulative depths the QUICK_FLUX layer temperatures
        # read (energy.layer_temperatures_quick_flux), once per run
        _eb_lsum = torch.cumsum(th['depth'].unsqueeze(1).expand(
            nc, mosaic.n_tiles, 3), dim=-1)
        if use_eb and not quick_flux:
            # The thermal node grid and the soil properties on it, once
            # per run (vic_init.c:833-945, set_node_parameters): they
            # depend on the parameters alone, never on the state.
            _depth = th['depth']
            _nl = _depth.shape[-1]
            _zsum, _dz = soiltemp.node_grid(_depth, th['dp'], nnodes,
                                             exp_trans)
            # vic_init.c:741-742, exactly as written.
            _por = 1.0 - th['bulk_density'] / th['soil_density']
            _pn, _en, _bn, _al, _be, _ga = soiltemp.node_parameters(
                _zsum, _depth, _por, soil_params.expt, soil_params.bubble,
                nnodes, _nl, noflux)
            _fs = getattr(soil_params, 'fs_active', None)
            if _fs is None:
                _fs = torch.ones(nc, dtype=torch.bool, device=device)
            else:
                _fs = torch.as_tensor(_fs, dtype=torch.bool, device=device)
            fd_static = dict(
                zsum=_zsum, dz=_dz, porosity=_por, porosity_node=_pn,
                expt_node=_en, bubble_node=_bn, alpha=_al, beta=_be,
                gamma=_ga, fs_active=_fs, nlayers=_nl,
                # ORGANIC_FRACT / BULK_DENSITY_COMB (vic_init.c:572-617):
                # th's bulk/soil densities are then the combined pair and
                # the mineral pair rides along; without them the mineral
                # pair is the pair and nothing is organic.
                organic=(th['organic'] if 'organic' in th
                         else torch.zeros_like(_depth)),
                bulk_dens_min=th.get('bulk_dens_min', th['bulk_density']),
                soil_dens_min=th.get('soil_dens_min', th['soil_density']))
        if do_soil_carbon:
            # initialize_soil.c:46-48 -- the pools start empty and fill from
            # litterfall. `annual_npp_prev` is zero for the first year, so
            # nothing enters the litter until a year has been simulated,
            # which is VIC's own behaviour and not a spin-up shortcut.
            _z = torch.zeros(nc, mosaic.n_tiles, dtype=dtype, device=device)
            def _pool(v):
                if v is None:
                    return _z.clone()
                v = torch.as_tensor(v, dtype=dtype, device=device)
                return (v.expand_as(_z).clone() if v.dim() < _z.dim()
                        else v.clone())
            _p3 = _init_pools or (None, None, None)
            carbon_state = dict(c_litter=_pool(_p3[0]), c_inter=_pool(_p3[1]),
                                c_slow=_pool(_p3[2]), annual_npp=_z.clone(),
                                annual_npp_prev=_pool(_init_npp_prev))
        eb_out = {k: [] for k in ('net_short', 'net_long', 'net_rad',
                                  'latent', 'sensible', 'grnd_flux',
                                  'delta_h', 't_surf', 'aero_resist',
                                  'energy_error', 'layer_temp', 'fallback')}
        # put_data.c rows that are functions of the per-tile balance rather
        # than cover-weighted means of it. Stacked beside eb_out; units are
        # in the run script's EB_UNITS.
        eb_extra = {k: [] for k in ('aero_cond', 'rad_temp', 'vegt',
                                    'snow_flux', 'fusion', 'baresoilt',
                                    # out-vars: put_data.c's per-surface
                                    # conductances and the soil node
                                    # temperatures, from the same tiles
                                    'aero_cond1', 'aero_cond2', 'tnode')
                    if (k in ('aero_cond', 'rad_temp', 'vegt', 'snow_flux',
                              'fusion', 'baresoilt')
                        or (k in ('aero_cond1', 'aero_cond2') and _w_aero)
                        or (k == 'tnode' and _w_tnode))}
        pets = []
    def _carbon_here(t, m, rc_t, dry_t, t_node, rs_t=None):
        """The carbon step, at the point surface_fluxes.c:732 takes it.

        Called from both paths -- the energy balance, where the canopy
        resistance comes from the converged solve, and the water balance,
        where it comes from the transpiration -- so the carbon cycle is
        wired in one place rather than two that can drift.

        `t_node` is the soil thermal profile, which only the energy balance
        has. Without it the vegetation fluxes are computed and the soil
        pools are not, and their rows are absent rather than zero.
        """
        lai_c = (mosaic.lai[:, :, m]
                 * vegetation.cellwise(p['lai_scale'], mosaic.cv))
        ice = snow_state['ice']
        snow_t = (ice > 0.0) if per_tile_snow else (
            (ice > 0.0).unsqueeze(1).expand_as(lai_c))
        cf = _carbon_fluxes(
            mosaic, m, lai_c, rc_t, dry_t, snow_t, moist[:, :, 0],
            soil_params.max_moist[..., 0].unsqueeze(-1),
            {k: atm[k][t] for k in ('par', 'Catm', 'fdir', 'coszen',
                                    'shortwave', 'tair')},
            batch.elevation,
            press_t=atm['press'][t].unsqueeze(1).expand_as(lai_c),
            ncanopy=carbon_ncanopy, albedo_par=carbon_albedo_par,
            rgas=carbon_rgas, rs_t=rs_t)
        for _k, _v in cf.items():
            carbon_out.setdefault(_k, []).append((_v * mosaic.cv).sum(1))
        if not (do_soil_carbon and t_node is not None):
            return
        # surface_fluxes.c:1074 runs soil_carbon_balance *before* runoff()
        # at :1087, so the water table the pools see is the one the column
        # holds on entry to the step -- which is `moist` here, before this
        # step's infiltration.
        zl = column.compute_zwt_lumped(
            moist.reshape(-1, column.NLAYER),
            soil_params_t, curve=(zwt_curves or {}).get('lumped')
        ).view(nc, mosaic.n_tiles)
        sc = _soil_carbon_step(carbon_state, t_node, zl, soil_params,
                               cf['NPP'], step_seconds,
                               per_second=carbon_resp_per_second)
        for _k, _v in sc.items():
            carbon_out.setdefault(_k, []).append((_v * mosaic.cv).sum(1))
        # vic_run.c:301-306 -- on the first day of each year the running
        # total becomes last year's, which feeds the litter for the next
        # twelve months.
        if t + 1 < len(batch.dates) and ydays[t + 1] == 1 and ydays[t] != 1:
            carbon_state['annual_npp_prev'] = carbon_state['annual_npp']
            carbon_state['annual_npp'] = torch.zeros_like(
                carbon_state['annual_npp'])

    melting = torch.zeros(_snow_shape, dtype=torch.bool, device=device)

    # VIC's water-balance mode (FULL_ENERGY FALSE) with a mosaic. VIC still
    # carries each tile's soil thermal nodes there: Tsurf = Tair every step
    # (calc_surf_energy_bal.c:542-545), T1 from the QUICK_FLUX closed form,
    # and both stored as energy->T[0] and T[1] (frozen_soil.c:66-69). They
    # feed two things: the ground heat flux and heat-storage change that
    # enter NetBareRad (energy.water_balance_ground_flux), and T[0] as the
    # ground temperature under a pack (surface_fluxes.c:389). Every node
    # starts at 0 C in this mode (generate_default_state.c:110-114).
    #
    # Measured against pristine vic_image.exe, annual basin totals: basin 37
    # 2001 at 1, 8 and 24 steps a day, ET, transpiration, canopy evaporation,
    # runoff and baseflow all within 0.005% (before this block, up to +7.5%
    # on transpiration and -16.0% on canopy evaporation at sub-daily steps);
    # Walnut Gulch 2008 3-hourly, every water flux within 0.0001%.
    #
    # A daily step with snow is sub-stepped when `subdaily` is passed:
    # SNOW_STEPS_PER_DAY surface steps on the sub-daily records, the column
    # once on the summed throughfall (surface_fluxes.c:258-269, :1080).
    # Snowfall is then partitioned on each sub-step's temperature, not on
    # the daily mean. A snow-free day is still that one mean, so June-August
    # at Walnut Gulch stays the daily path.
    #
    # An overstory under snow is VIC's too. VIC's canopy air temperature is
    # the air's in every configuration the image driver can run: it never
    # reads CLOSE_ENERGY (initialize_options.c 5.1.0:31 leaves it false), so
    # calc_atmos_energy_bal returns Tair (calc_atmos_energy_bal.c 5.1.0
    # :73-103) and vic_init.c 5.1.0:1558 sets MAX_ITER_GRND_CANOPY to 0.
    # What differed was the canopy's opening state, the ground's longwave,
    # the refreeze energy's store, the saturated vapour pressure, the thin
    # pack, and the albedo age (see where each is set). Stehekin, VIC's own
    # water-balance sample (hourly, January 1949, ten days, SNOW_BAND
    # FALSE): ET, snow and canopy sublimation, canopy evaporation and SWE
    # within 0.005% of vic_image.exe (were ET +11%, canopy sublimation
    # +55%).
    if energy_error not in ('solver', 'vic'):
        raise ValueError("energy_error must be 'solver' or 'vic'; got %r"
                         % (energy_error,))
    wb_mode = mosaic is not None and not use_eb
    wb_quick_flux = False
    if wb_mode and not quick_flux:
        # get_global_param.c:939-945 stops VIC on QUICK_FLUX FALSE in the
        # water balance without frozen soil; with it VIC runs the profile
        # in the water balance, which TorchVIC does not: the
        # water-balance ground flux below is the QUICK_FLUX form only.
        raise ValueError(
            'quick_flux=False needs full_energy=True: without frozen soil '
            'VIC stops the same way (get_global_param.c:939-945); the '
            'water-balance finite-difference profile with frozen soil is '
            'not implemented')
    if not quick_flux and not tfallback:
        # frozen_soil.c:301-515: with TFALLBACK FALSE VIC stops at a failed
        # node solve of the profile; the profile here always falls back.
        raise ValueError(
            'quick_flux=False with tfallback=False is not implemented: VIC '
            'stops at a failed profile node solve (frozen_soil.c:301-515)')
    if not quick_flux and _frost is not None:
        # The profile's layer ice is integrated per frost area
        # (soil_conduction.c:469-615, estimate_layer_ice_content); the
        # finite-difference path here integrates one area only.
        raise ValueError(
            'quick_flux=False with n_frost > 1 (SPATIAL_FROST) is not '
            'implemented: the per-frost-area profile ice '
            '(soil_conduction.c:469-615) is not')
    if wb_mode:
        _th = getattr(soil_params, 'thermal', None)
        wb_quick_flux = _th is not None and all(
            _th.get(k) is not None for k in ('avg_temp', 'dp', 'depth',
                                              'bulk_density', 'soil_density',
                                              'quartz'))
        if not wb_quick_flux:
            if frozen_soil:
                raise ValueError(
                    'frozen_soil=True in water-balance mode needs the soil '
                    'parameters\' thermal properties (avg_T, dp, bulk and '
                    'particle density, quartz): the layer temperatures the '
                    'ice is estimated against come from the QUICK_FLUX '
                    'profile (frozen_soil.c:75-81).')
            (log or print)('water balance: the soil parameters carry no '
                           'thermal properties (avg_T, dp, bulk and particle '
                           'density, quartz), so the ground heat flux VIC '
                           'adds to NetBareRad (func_surf_energy_bal.c'
                           ':398-429, :663-664) is not formed')
        # generate_default_state.c:95-99: every node starts at avg_temp
        # with FULL_ENERGY or FROZEN_SOIL, at 0 C otherwise.
        if frozen_soil:
            _tavg = torch.as_tensor(_th['avg_temp'], dtype=dtype,
                                    device=device).unsqueeze(1).expand(
                nc, mosaic.n_tiles).clone()
            wb_t0, wb_t1 = _tavg, _tavg.clone()
        else:
            wb_t0 = torch.zeros(nc, mosaic.n_tiles, dtype=dtype,
                                device=device)
            wb_t1 = torch.zeros_like(wb_t0)
        wb_grnd, wb_delta_h = [], []
        # put_data.c:928 reports OUT_FUSION negated in this mode too: the
        # fusion term is formed (func_surf_energy_bal.c:535-548) and joins
        # NetBareRad (:663-664) even with Tsurf = Tair.
        wb_fusion = []
        # OUT_SOIL_TEMP, the quick-flux layer temperatures
        # (soil_conduction.c:598-628) cover-weighted over tiles
        # (put_data.c:693).
        wb_layer_temp = []
        # out-vars-wb: the energy rows VIC reports in this mode
        # (torchvic.model.wbenergy), appended outside the fused regions
        wb_e = {k: [] for k in ('net_short', 'net_long', 'net_rad', 'latent',
                                'sensible', 't_surf', 'aero_resist',
                                'layer_temp') if _w_wb}
        wb_x = {k: [] for k in ('aero_cond', 'rad_temp', 'vegt', 'snow_flux',
                                'aero_cond1', 'aero_cond2', 'tnode')
                if _w_wb}
        # rows nothing asked for take their appends and keep nothing
        for _d in (wb_e, wb_x):
            for _k in list(_d):
                if not _ov(*_outvars.WB_ENERGY_ROWS[_k]):
                    _d[_k] = _outvars.NullList()
        wb_pets = ([] if _ov('OUT_PET') else _outvars.NullList()) \
            if _w_wb else []
    # Where the water-balance path takes a thin frozen pack from snow_step
    # and evaluates it at Tsurf = Tair (_wb_thin_pack), as VIC's INCLUDE_SNOW
    # branch does (snow_melt.c:322-324, calc_surf_energy_bal.c:646-739).
    # False keeps snow_step solving it, as it did before that branch, and
    # three places keep that until what they need is carried:
    #   * a snow-free daily step: the forcing is the day's mean, and a thin
    #     pack evaluated at that mean moved Walnut Gulch 2008 daily (VIC's
    #     file day) away from VIC. At 3-hourly, where the branch was
    #     validated, every flux is within 0.0001%. A snow day below is
    #     sub-stepped at the sub-daily temperature, and wb_thin is on for
    #     those steps only;
    #   (SPATIAL_SNOW's partial cover is carried: `spc` in _wb_thin_pack.)
    # Overstory tiles take it too: their surface and thin pack are evaluated
    # at Tcanopy = Tair like any other (surface_fluxes.c 5.1.0:365, :595-616
    # with CLOSE_ENERGY false). WB_THIN_ON_OVERSTORY False restores the
    # snow_step solve they had before, for measuring.
    wb_thin = (wb_mode and per_tile_snow and steps_per_day > 1)
    if wb_thin and any_over and not WB_THIN_ON_OVERSTORY:
        wb_thin = ~over_t
    evap_bares = []

    # VIC reports transpiration, bare-soil evaporation and canopy evaporation
    # as three separate variables, and the split is the diagnosis: a total
    # that matches for the wrong reasons is not agreement. Keeping them apart
    # costs one append each and makes the model comparable to VIC's own
    # output rather than only to its sum.
    runoffs, baseflows, swes, ets = [], [], [], []
    snow_depths = []
    # 'advected_sensible' is filled only under SPATIAL_SNOW; with the option
    # off VIC sets the term to zero (SnowPackEnergyBalance.c:212-215) and the
    # list stays empty rather than carrying a column of zeros.
    snow_eb = {k: [] for k in ('advected_sensible', 'latent_sub',
                               'advection', 'deltacc',
                               'rfrz_energy', 'melt_energy', 'refreeze')}
    transps, canopy_evaps, sublimations = [], [], []
    # VIC's OUT_ALBEDO is AlbedoUnder: the cover-weighted snow-free albedo of
    # the tiles (calc_surf_energy_bal.c:633-635, snow_coverage 0). The mosaic
    # has carried the file's per-class monthly albedo since the canopy took
    # it; this returns the weighted value so the row can be compared instead
    # of reading "a fixed land albedo is used", which stopped being true then.
    albedos = []
    in_longs = []                                               # IN_LONG
    # Everything below is a quantity the loop already computes and used to drop
    # on the floor. Each list records a value at the point it exists; none of
    # them feeds back into the state, so a run with and without these lists is
    # the same run to the digit. The names are VIC's OUT_* with the prefix
    # removed. put_data.c is the authority for what each one is and how it is
    # weighted over tiles.
    rains, snowfalls, prcps, melts = [], [], [], []           # RAINF SNOWF SNOW_MELT
    snow_covers, salbedos = [], []                            # SNOW_COVER SALBEDO
    snow_surf_temps, snow_pack_temps = [], []                 # SNOW_SURF/PACK_TEMP
    wdews, lais, rootmoists = [], [], []                      # WDEW LAI ROOTMOIST
    soil_liqs, asats, inflows, soil_wets, zwts = [], [], [], [], []
    zwt_lumpeds = []                                          # ZWT_LUMPED
    surf_frost_fracs, fdepths, tdepths = [], [], []   # SURF_FROST_FRAC FDEPTH TDEPTH
    sub_blowings = []                                  # SUB_BLOWING (BLOWING)
    # the OUT_*_BAND rows that are not a cell-level row (put_data.c:858-924:
    # the understory's own advection, refreeze, latent, sublimation and
    # sensible heat, and the pack's temperatures, each times Cv)
    band_rows = {k: [] for k in ('adv', 'rfz', 'lat', 'lsub', 'sen',
                                 'surft', 'packt') if _w_band[k]}
    soil_ices = []
    withhelds = []
    # put_data.c:482-547 forms every DEL* term and the water error against
    # the *previous* step's totals held in save_data, and vic_init_output.c
    # seeds save_data from the initial state, so the first step's delta is
    # against the state the run opened in. That opening state is kept here.
    if mosaic is not None:
        # VIC keeps a soil column per vegetation tile
        # (cell[iveg][band].layer, vic_run.c:148 loops iveg over the tiles
        # and each carries its own layer moisture). One column per cell,
        # which this model kept until now, was measured as the whole of the
        # remaining ET disagreement at Walnut Gulch: with the grass tile
        # alone at Cv = 1 transpiration was 0.998..1.013 of VIC's every
        # month, with the mosaic 1.30 in June, because the grass tile
        # (rmin 50) dries its own column past wilting point days before the
        # cell mean does.
        # Every tile opens in the cell's initial moisture, as VIC's do.
        moist = moist.unsqueeze(1).expand(nc, mosaic.n_tiles, moist.shape[-1]).clone()
        soil_params_t = column.expand_params(soil_params, mosaic.n_tiles)
        cvl = mosaic.cv.unsqueeze(-1)
    if _frost is not None and frozen_soil:
        # generate_default_state.c:140-163: the initial ice from layer
        # temperatures of 0 C, which is what the calloc'd layers hold
        # before compute_derived_state_vars. With one area that is always
        # zero and soil_ice stays None until the first step; with more
        # the cold end of the distribution can already hold ice.
        from ..core import frozen as _frz0
        _z0 = torch.zeros_like(moist)
        soil_ice = _frz0.layer_ice_quick_flux(
            moist,
            soil_params.max_moist.unsqueeze(1).expand_as(moist),
            soil_params.bubble.unsqueeze(1).expand_as(moist),
            soil_params.expt.unsqueeze(1).expand_as(moist),
            _z0, _frost['slope'], n_frost=_frost['n'],
            fs_active=getattr(soil_params, 'fs_active', None))
    # A warm start. Everything above built the cold start; a state from an
    # earlier run replaces each carried quantity with what that run's last
    # step left, after checking it has this run's shape. What is carried is
    # listed in export_state, and must be all of it: one quantity missed
    # and the split run is a different model from the single one.
    _prev_rows = None
    if initial_state is not None:
        _cold = dict(
            moist=moist, snow=snow_state, last_snow=last_snow,
            melting=melting, spatial_snow=sp, wdew=wdew,
            canopy=canopy_state if any_over else None,
            ground_long_out=ground_long_out if any_over else None,
            t_node=t_node if use_eb else None, soil_ice=soil_ice,
            wb_t=dict(t0=wb_t0, t1=wb_t1) if wb_mode else None,
            carbon=carbon_state, snow_zero=_snow_zero, ce_seed=ce_seed)
        if initial_state.get('source') == 'vic_file':
            # INIT_STATE: VIC's state file laid over the cold start
            _r = from_vic_file(initial_state, _cold, mosaic.cv,
                               frozen_soil=bool(frozen_soil),
                               n_frost=1 if _frost is None else _frost['n'],
                               any_over=any_over)
        else:
            _r = restore_state(
                initial_state, _cold, steps_per_day=steps_per_day,
                frozen_soil=bool(frozen_soil),
                n_frost=1 if _frost is None else _frost['n'])
        if not allow_gap:
            # A state continues the run that exported it, from the next step.
            # A window cut at the wrong index would otherwise be accepted and
            # silently start from a different day's stores (and the 1 January
            # carbon turnover assumes contiguity). Reusing one spin-up state
            # for other windows is legitimate, but must be said: allow_gap.
            import datetime as _dtm
            _last = initial_state.get('last_date')
            if _last is not None and len(batch.dates):
                _want = _last + _dtm.timedelta(seconds=SEC_PER_DAY
                                               / float(steps_per_day))
                if batch.dates[0] != _want:
                    raise ValueError(
                        'initial_state ends at %s, so this run should start at '
                        '%s, but its first date is %s; pass allow_gap=True to '
                        'reuse a state deliberately' % (_last, _want,
                                                        batch.dates[0]))
        if _r.get('vic_aux'):
            _vic_aux.update(_r['vic_aux'])
        moist, snow_state = _r['moist'], _r['snow']
        if 'swq_c' not in snow_state:
            # a state saved without it: snow->swq as snow_melt.c 5.1.0:464
            snow_state = dict(snow_state)
            snow_state['swq_c'] = (snow_state['ice'] + snow_state['pack_water']
                                   + snow_state['surf_water'])
        last_snow, melting, sp = _r['last_snow'], _r['melting'], _r['spatial_snow']
        wdew, soil_ice, _snow_zero = _r['wdew'], _r['soil_ice'], _r['snow_zero']
        if any_over:
            canopy_state = _r['canopy']
            for k_ in ('carry_adv', 'carry_rfz'):
                if k_ not in canopy_state:
                    canopy_state[k_] = torch.zeros_like(
                        canopy_state['int_snow'])
            ground_long_out = _r['ground_long_out']
        if use_eb:
            t_node = _r['t_node']
        if close_energy and use_eb:
            ce_seed = _r['ce_seed']
        if wb_mode:
            wb_t0, wb_t1 = _r['wb_t']['t0'], _r['wb_t']['t1']
        carbon_state = _r['carbon']
        _prev_rows = _r['report']
        # vic_run.c:301-306 turns the year over after the last step of 31
        # December (the check in _carbon_here looks one date ahead). A run
        # that ended there cannot see the next date, so a run that starts
        # on 1 January does it before its first step, as the single run
        # did between the two.
        if (carbon_state is not None and ydays[0] == 1
                and _r['last_yday'] != 1):
            carbon_state['annual_npp_prev'] = carbon_state['annual_npp']
            carbon_state['annual_npp'] = torch.zeros_like(
                carbon_state['annual_npp'])
    if soil_ice is None and initial_state is None and use_eb \
            and not quick_flux and frozen_soil:
        # generate_default_state.c:174-196: with QUICK_FLUX FALSE the
        # cold start integrates the layer ice over the opening profile
        # (estimate_layer_ice_content at avg_temp), so a frozen cell
        # opens with ice in its layers. prepare_full_energy's kappa/Cs
        # and the first step's liquid (moist - ice) all see it. With
        # QUICK_FLUX TRUE the same cold start runs the quick-flux
        # estimate on layer temperatures still at their calloc zero
        # (soil_conduction.c:615-672), which clamps to nothing -- ice
        # None (zero) is already that, hence the gate. A warm start
        # carries its own ice and never comes here.
        _fr1 = torch.ones(t_node.shape[:-1] + (1,), dtype=dtype,
                          device=device)
        _fs0 = torch.zeros(t_node.shape[:-1], dtype=dtype, device=device)
        _tt, _tz = soiltemp.frost_temperature_and_depth(
            fd_static['zsum'], t_node, th['depth'], _fr1, _fs0, nnodes,
            fd_static['nlayers'])
        soil_ice = soiltemp.layer_ice_content(
            _tt, _tz, fd_static['zsum'], th['depth'],
            soil_params.max_moist, soil_params.expt, soil_params.bubble,
            nnodes, fd_static['nlayers'], fd_static['fs_active'],
            frozen_soil, moist)[..., 0]

    def _distribute(moist_, t_node_):
        # distribute_node_moisture_properties (soil_conduction.c:186):
        # the node moisture, ice, conductivity and heat capacity from
        # the tiles' layer moisture and the current profile. VIC runs
        # it at setup (compute_derived_state_vars.c:132) and after
        # every runoff() (runoff.c:381); a warm start recomputes it
        # from the restored stores, which is the same pure function of
        # the same state, so it needs no slot in the carried state.
        _m, _i, _k, _c = soiltemp.distribute_node_moisture(
            moist_, t_node_, fd_static['zsum'], fd_static['porosity_node'],
            fd_static['expt_node'], fd_static['bubble_node'], th['depth'],
            fd_static['soil_dens_min'], fd_static['bulk_dens_min'],
            th['quartz'], th['soil_density'], th['bulk_density'],
            fd_static['organic'],
            nnodes, fd_static['nlayers'], fd_static['fs_active'],
            frozen_soil)
        return dict(moist=_m, ice=_i, kappa=_k, cs=_c)

    if use_eb and not quick_flux:
        node_props = _distribute(moist, t_node)
    moist0 = moist
    # The cell's opening moisture for the DEL* terms: cover-weighted over
    # the tiles' columns, as put_data.c weights OUT_SOIL_MOIST.
    moist0_cell = _tsum(moist0 * cvl) if mosaic is not None else moist0
    wdew0 = (_tsum(wdew * mosaic.cv) if mosaic is not None
             else torch.zeros(nc, dtype=dtype, device=device))
    # A ten-year hourly run is 87,648 silent iterations. One was killed by a
    # four-hour timeout minutes before it would have finished, and nothing
    # had said how far along it was. `log_every` > 0 prints step, elapsed
    # and a linear estimate of the remainder at that interval; the default
    # of 0 keeps the loop silent, so no existing caller changes behaviour.
    import time as _time
    _t0 = _time.time()
    _say = log or print
    # `progress` puts a tqdm bar on this loop, the part that runs for
    # minutes. Off by default
    # for the same reason `log_every` is 0: hundreds of tests call this.
    # The bar observes and changes nothing; tests/test_progress.py holds
    # the outputs bit-identical with it on and off.
    from .. import progress as _progress
    # The soil's two water-table curves depend on its parameters alone, so
    # they are built once here rather than in column.step every step (on
    # the ISC domain that rebuild was a large share of the step's device
    # time). Without gradients only: a run with them keeps the per-step
    # graph it always had, and with it the order its gradients sum in.
    zwt_curves = None
    if not torch.is_grad_enabled():
        zwt_curves = column.zwt_curves(soil_params_t if mosaic is not None
                                       else soil_params)
    # partition_precipitation's ordering check, once for the run rather
    # than once a step (it reads both parameters back from the device).
    check_snow_range(p['t_max_snow'], p['t_min_rain'])
    _steps = _progress.bar(range(nt), total=nt, desc=progress_desc,
                           enable=bool(progress), unit='step',
                           leave=False)
    # Lane mode: every per-step output is a cell-level value each lane of the
    # cell holds; _flush_lanes takes one lane per cell, after each step.
    _out_lists = [runoffs, baseflows, swes, ets, snow_depths, transps,
                  canopy_evaps, sublimations, albedos, rains, snowfalls,
                  prcps, melts, snow_covers, salbedos, snow_surf_temps,
                  snow_pack_temps, wdews, lais, rootmoists, soil_liqs, asats,
                  inflows, soil_wets, zwts, zwt_lumpeds, soil_ices, withhelds,
                  surf_frost_fracs, fdepths, tdepths, sub_blowings, evap_bares,
                  *band_rows.values(),
                  snow_canopies, sub_canops, in_longs] + list(snow_eb.values())
    if wb_mode:
        _out_lists += [wb_grnd, wb_delta_h, wb_fusion, wb_layer_temp]
        _out_lists += list(wb_e.values()) + list(wb_x.values()) + [wb_pets]
    if use_eb:
        # the energy rows are cell-level too (EB_LANES)
        _out_lists += list(eb_out.values()) + list(eb_extra.values()) + [pets]
    _flushed = [0] * len(_out_lists)
    # One name per output list so a snow day can sum the fluxes VIC sums
    # (melt, vapour, rain, evaporation) and average the ones it divides by
    # N_steps (the W/m2 terms, surface_fluxes.c:960-988). Anything else is
    # the store at the last snow step.
    _named = dict(
        runoff=runoffs, baseflow=baseflows, swe=swes, et=ets,
        snow_depth=snow_depths, transp=transps, canopy_evap=canopy_evaps,
        sublimation=sublimations, albedo=albedos, rain=rains,
        snowfall=snowfalls, prcp=prcps, melt=melts, snow_cover=snow_covers,
        salbedo=salbedos, snow_surf_temp=snow_surf_temps,
        snow_pack_temp=snow_pack_temps, wdew=wdews, lai=lais,
        rootmoist=rootmoists, soil_liq=soil_liqs, asat=asats,
        inflow=inflows, soil_wet=soil_wets, zwt=zwts, zwt_lumped=zwt_lumpeds,
        surf_frost_frac=surf_frost_fracs, fdepth=fdepths, tdepth=tdepths,
        sub_blowing=sub_blowings,
        **{'band_' + k_: v_ for k_, v_ in band_rows.items()},
        soil_ice=soil_ices,
        withheld=withhelds, evap_bare=evap_bares, snow_canopy=snow_canopies,
        sub_canop=sub_canops, in_long=in_longs)
    for _ek, _ev in snow_eb.items():
        _named['eb_' + _ek] = _ev
    if wb_mode:
        _named['wb_grnd'] = wb_grnd
        _named['wb_delta_h'] = wb_delta_h
        _named['wb_fusion'] = wb_fusion
        _named['wb_layer_temp'] = wb_layer_temp
        for _ek, _ev in wb_e.items():
            _named['ebout_' + _ek] = _ev
        for _ek, _ev in wb_x.items():
            _named['ebex_' + _ek] = _ev
        _named['pet'] = wb_pets
    # transp and evap_bare are the last snow step's: surface_fluxes.c
    # :1020-1021 copies that step's layers and overwrites only evap with the
    # sum, and put_data.c:599-602 reports layer esoil and transp.
    _SUM = {'sub_blowing', 'et', 'canopy_evap', 'sublimation', 'rain', 'snowfall',
            'prcp', 'melt', 'sub_canop', 'eb_refreeze'}
    _MEAN = {'albedo', 'in_long', 'eb_latent_sub', 'eb_advection', 'eb_deltacc',
             'eb_rfrz_energy', 'eb_melt_energy', 'eb_advected_sensible',
             'wb_grnd', 'wb_delta_h', 'wb_fusion'}
    if wb_mode:
        # the water-balance path's energy rows combine over the snow steps as
        # the energy path's (surface_fluxes.c:960-988, :1023-1041, :934)
        _SUM.add('pet')
        _MEAN.update('ebout_' + k for k in ('net_short', 'net_long',
                                            'net_rad', 'latent', 'sensible'))
        _MEAN.update('ebex_' + k for k in ('aero_cond', 'snow_flux',
                                           'aero_cond1', 'aero_cond2'))
    if use_eb:
        # The energy rows are W/m2 averaged over the snow steps
        # (surface_fluxes.c:960-988) except the temperatures, which are the
        # state the last snow step left, and PET, which is an amount.
        for _ek, _ev in eb_out.items():
            _named['ebout_' + _ek] = _ev
        for _ek, _ev in eb_extra.items():
            _named['ebex_' + _ek] = _ev
        _named['pet'] = pets
        _SUM.add('pet')
        _MEAN.update('ebout_' + k for k in (
            'net_short', 'net_long', 'net_rad', 'latent', 'sensible',
            'grnd_flux', 'delta_h', 'energy_error'))
        _MEAN.update('ebex_' + k for k in ('aero_cond', 'snow_flux', 'fusion',
                                           'aero_cond1', 'aero_cond2'))
    _MEAN.update('band_' + k_ for k_ in ('adv', 'rfz', 'lat', 'lsub', 'sen'))
    # skip: surface and snow run, the column waits until the last snow
    # step. add_*: throughfall and layer evaporation already accumulated.
    # divide_fcan: vic_run.c:219-225 divides Wdew and snow_canopy by
    # fcanopy once, before surface_fluxes. solve_snow.c:250-251
    # multiplies them back at the end of every snow sub-step, and the
    # next sub-step reads those values as they stand (snow_intercept.c
    # :132-133 divides by F, and the call passes F = 1). A later snow
    # sub-step of the same model step must not divide again.
    _sub = dict(skip=False, add_tf=None, add_ev=None, cap_tf=None,
                cap_ev=None, flush=True, divide_fcan=True,
                pin_tgrnd=False, t_grnd=None, atten_one=None, eb_rows=[],
                can_rows=[], alb_rows=[], soil_adv=None, soil_dcc=None,
                model_dt=None, tile_w=None, cov_star=None,
                # CLOSE_ENERGY's iteration: the pass's per-tile canopy air
                # temperature (None: the air's, as without the option) and
                # UNSTABLE_SNOW, the pass's stashed outputs for the driver,
                # whether this pass commits the closed sensible heat, and
                # the snow_flux/UNSTABLE_SNOW a snow sub-step hands the next
                # (surface_fluxes.c locals, which persist across hidx).
                tcanopy=None, unstable=None, ce_pass=None, ce_final=False,
                ce_carry=None, ce_seed_rows=[], understory=None)

    def _canopy_carry(on_, over_, adv_, rfz_):
        # A step whose snow branch does not run leaves energy->canopy_*
        # as it stood (solve_snow.c:146-253 alone sets them), and
        # surface_fluxes.c:495, :898 adds that value again. On a single
        # step put_data.c:919-931 reports only a step with snow, so
        # outside the snow steps the reported value is unchanged.
        c_a = torch.where(on_, adv_, canopy_state['carry_adv'])
        c_r = torch.where(on_, rfz_, canopy_state['carry_rfz'])
        canopy_state['carry_adv'], canopy_state['carry_rfz'] = c_a, c_r
        if not _sub['pin_tgrnd']:
            return adv_, rfz_
        z_ = torch.zeros_like(c_a)
        c_a = torch.where(over_, c_a, z_)
        c_r = torch.where(over_, c_r, z_)
        _sub['can_rows'].append(dict(adv=c_a, rfz=c_r, on=on_))
        return c_a, c_r

    def _cells_fused():
        # the step's cell sums can go through one fused region
        # (_cell_sums): lanes on the graph path with fusion on
        return (lanes is not None and mosaic is not None
                and _fusion.wants((mosaic.cv,)))

    def _out_named(key):
        # an output list by name ('eb.<k>' for snow_eb[k])
        if key.startswith('eb.'):
            return snow_eb[key[3:]]
        return dict(sublimations=sublimations, melts=melts,
                    snow_depths=snow_depths, swes=swes,
                    snow_covers=snow_covers, salbedos=salbedos,
                    snow_surf_temps=snow_surf_temps,
                    snow_pack_temps=snow_pack_temps)[key]

    def _flush_lanes():
        if graphsafe.ACTIVE:
            # the graph path: every output of the step to the cell axis in
            # one fused region (the same index_select, gathered together)
            at_ = [(i_, j_) for i_, l_ in enumerate(_out_lists)
                   for j_ in range(_flushed[i_], len(l_))
                   if torch.is_tensor(l_[j_])]
            if at_:
                got_ = _lanes_mod.to_cell_many(
                    [_out_lists[i_][j_] for i_, j_ in at_], lanes.rep)
                for (i_, j_), v_ in zip(at_, got_):
                    _out_lists[i_][j_] = v_
            for i_, l_ in enumerate(_out_lists):
                _flushed[i_] = len(l_)
            return
        for i_, l_ in enumerate(_out_lists):
            for j_ in range(_flushed[i_], len(l_)):
                if torch.is_tensor(l_[j_]):
                    l_[j_] = lanes.to_cell(l_[j_])
            _flushed[i_] = len(l_)

    # LAKES TRUE (torchvic.model.lakecell): the land step leaves its
    # per-tile runoff, baseflow, leaf area and energy terms here for the
    # lake step that follows it (vic_run.c 5.1.0:368-466).
    _lake_tap = {} if lakes is not None else None
    # the graph path's overstory subset maps (_sub_scatter), by grid shape
    _sub_maps = {}
    # the graph path's snow subsets (SNOW_SET): 'snow' and 'canopy', each
    # None or its index maps; set per graph window
    _snow_set = {}
    # the step's critical and wilting points (constant over the run)
    _wcw_cache = {}

    def _thin_snapshot(snow_state_, tile_rows_, tf_t_):
        # the energy balance's thin-pack branch on the graph path: every
        # value it may replace, as it stands before it
        return ([(l_, l_[-1]) for l_ in _out_lists
                 if l_ and torch.is_tensor(l_[-1])],
                dict(snow_state_), dict(tile_rows_), tf_t_)

    def _thin_blend(any_, pre_, snow_state_, tile_rows_, tf_t_):
        # calc_surf_energy_bal.c 5.1.0:643-700 runs where some tile's pack
        # is thin (the eager step's `if`): what the branch wrote where
        # `any_` (a device bool) holds, what it found elsewhere -- the
        # eager step's values either way, without asking the device
        lists_, ss_, tr_, tf0_ = pre_
        for l_, v_ in lists_:
            if l_[-1] is not v_:
                l_[-1] = torch.where(any_, l_[-1], v_)
        for d_, d0_ in ((snow_state_, ss_), (tile_rows_, tr_)):
            for k_, v_ in d0_.items():
                if torch.is_tensor(v_) and d_.get(k_) is not v_:
                    d_[k_] = torch.where(any_, d_[k_], v_)
        return tf_t_ if tf_t_ is tf0_ else torch.where(any_, tf_t_, tf0_)

    def _step_inner(t):
        nonlocal _k, _ls, _r, _th, _v, canopy_long_out, ground_long_out, k, \
                 last_snow, melting, moist, node_props, share, snow_state, \
                 soil_ice, sp, t_node, th, v, wb_t0, wb_t1, wdew
        # The step's time index where it reads the forcing and the month. In a
        # graph (graphsafe.ACTIVE) both are the step's buffers whatever the
        # index, and a constant index keeps the fused blocks below
        # (core.fusion.run) from specialising on the step.
        _t1 = 0 if graphsafe.ACTIVE else t
        # the day of year (the MELTING season below); a graph reads its
        # buffer instead
        _doy = (None if graphsafe.ACTIVE and _season_buf is not None
                else float(batch.day_of_year[t]) if hasattr(batch, 'day_of_year')
                else float(batch.dates[t].timetuple().tm_yday))
        if rc_photo:
            # surface_fluxes.c:375-407 computes aPARLayer before the solve;
            # canopy_evap's calc_rc_ps reads it with the step's CO2 and
            # shortwave.
            mosaic.photo = dict(
                {k_: atm[k_][_t1] for k_ in ('par', 'Catm', 'fdir', 'coszen',
                                           'shortwave')},
                albedo_par=carbon_albedo_par, ncanopy=carbon_ncanopy,
                rgas=carbon_rgas)
        if lanes is not None and t and _sub['flush']:
            _flush_lanes()
        if log_every and t and t % log_every == 0:
            el = _time.time() - _t0
            _say('  step %d/%d  %.0f s elapsed  ~%.0f s to go  (%.0f ms/step)'
                 % (t, nt, el, el * (nt - t) / t, 1000.0 * el / t))
        _tfx.set_step(t)
        # (core.fusion.run: this block one fused region on the graph path)
        def _step_head():
            prcp = rain = rain_m = rain_mm0 = sky_snow_m = snowfall = snowfall_m = trace = wb_int_snow0 = wb_sky_snow = wb_swq0 = _UNSET
            prcp = atm['prcp'][_t1] * p['pcf']
            rain, snowfall = partition_precipitation(prcp, atm['tair'][_t1],
                                                     p['t_max_snow'],
                                                     p['t_min_rain'],
                                                     check=False)
            # solve_snow.c:139-143: a snowfall under SNOWFALL_TRACE_MM is set to
            # zero before anything reads it, and out_prec is then rain plus what
            # is left, so the trace leaves the water balance with it. Only those
            # entries change; everywhere else `prcp` and `snowfall` are the
            # partition's own, bit for bit.
            if _gauge is not None:
                # vic_run.c:106-114: factors of 1 on a step without precipitation
                # (none matter there); solve_snow.c:117-120 scales the snow and
                # the rain, and :139-141's trace test sees the scaled snow
                g_rain, g_snow = gauge_correction(
                    atm['wind'][_t1], _gauge['wind_h'], _gauge['height'],
                    _gauge['rough'], _gauge['snow_rough'])
                g_on = prcp > 0.0
                snowfall = torch.where(g_on, g_snow * (prcp - rain), snowfall)
                rain = torch.where(g_on, g_rain * rain, rain)
            trace = (snowfall > 0.0) & (snowfall < SNOWFALL_TRACE_MM)
            snowfall = torch.where(trace, torch.zeros_like(snowfall), snowfall)
            prcp = torch.where(trace, rain, prcp)
            if _gauge is not None:
                # out_prec = snowfall + rainfall (solve_snow.c:142)
                prcp = snowfall + rain
            # Into the snow module in metres, out of it in millimetres. The snow
            # state is carried in metres throughout; see MM_PER_M above.
            rain_m = rain / MM_PER_M
            snowfall_m = snowfall / MM_PER_M
            rain_mm0 = rain   # the sky's rain in mm, before any canopy re-sum
            # the sky's snowfall, before a canopy replaces snowfall_m with its
            # throughfall (store_snowfall, solve_snow.c 5.1.0:127)
            sky_snow_m = snowfall_m
            if wb_mode or use_eb:
                # solve_snow.c:164-171 decides snow->snow from the state the
                # step opens with and the snow falling from the sky, before any
                # canopy has intercepted it; the canopy stage below overwrites
                # snowfall_m with its throughfall, so both are kept here.
                wb_sky_snow = snowfall_m
                wb_swq0 = (snow_state['ice'] + snow_state['surf_water']
                           + snow_state['pack_water'])
                wb_int_snow0 = canopy_state['int_snow'] if any_over else None
            return dict(prcp=prcp, rain=rain, rain_m=rain_m, rain_mm0=rain_mm0, sky_snow_m=sky_snow_m, snowfall=snowfall, snowfall_m=snowfall_m, trace=trace, wb_int_snow0=wb_int_snow0, wb_sky_snow=wb_sky_snow, wb_swq0=wb_swq0)
        _blk_res = _fusion.run(_step_head)
        if _blk_res['prcp'] is not _UNSET:
            prcp = _blk_res['prcp']
        if _blk_res['rain'] is not _UNSET:
            rain = _blk_res['rain']
        if _blk_res['rain_m'] is not _UNSET:
            rain_m = _blk_res['rain_m']
        if _blk_res['rain_mm0'] is not _UNSET:
            rain_mm0 = _blk_res['rain_mm0']
        if _blk_res['sky_snow_m'] is not _UNSET:
            sky_snow_m = _blk_res['sky_snow_m']
        if _blk_res['snowfall'] is not _UNSET:
            snowfall = _blk_res['snowfall']
        if _blk_res['snowfall_m'] is not _UNSET:
            snowfall_m = _blk_res['snowfall_m']
        if _blk_res['trace'] is not _UNSET:
            trace = _blk_res['trace']
        if _blk_res['wb_int_snow0'] is not _UNSET:
            wb_int_snow0 = _blk_res['wb_int_snow0']
        if _blk_res['wb_sky_snow'] is not _UNSET:
            wb_sky_snow = _blk_res['wb_sky_snow']
        if _blk_res['wb_swq0'] is not _UNSET:
            wb_swq0 = _blk_res['wb_swq0']
        # solve_snow.c:143-145 records the partition as out_rain / out_snow
        # and put_data.c:140,150 echoes them as OUT_RAINF / OUT_SNOWF, in mm
        # over the cell. `prcp` is kept beside them so the per-cell water
        # error below is formed from the same number the loop consumed.
        rains.append(rain)
        snowfalls.append(snowfall)
        prcps.append(prcp)

        # Snow albedo from VIC's own decay rather than a constant. A
        # sensitivity study measured albedo as the third most important parameter for snow water
        # equivalent, behind only the precipitation correction and snow
        # emissivity, which is what three independent studies report for cold
        # radiation-driven basins. One number cannot express it: the albedo of
        # fresh snow, and the rates it decays under accumulation and under
        # thaw, are four separate quantities and VIC has always kept them
        # separate. snow_albedo() was already ported and bit-exact against
        # snow_utility.c; it simply was never called.
        canopy_long_out = None
        canopy_lat = canopy_sen = canopy_lsub = None
        canopy_adv = canopy_rfz = None
        vegt_canopy = None
        # The pack sees the sky unless a canopy stands over it, and takes the
        # cell's precipitation unless a canopy intercepted some of it.
        under_short = under_long = None
        rain_tile = snow_tile = snow_tile_mm = None
        canopy_nl = canopy_ns = canopy_ran = None
        alb_over_t = None
        ra_veg1_t = None
        ov_step = None
        # water balance: the canopy stage's liquid evaporation, mm per tile
        wb_canopy_liquid = None
        if any_over:
            # solve_snow.c:170-171: the canopy runs when the cell has a
            # pack, snow is falling, or the canopy holds snow. The pack is
            # per cell and the canopy per tile: a named structural
            # difference (VIC has a pack per tile).
            mm_ = months[_t1]
            alb_c = mosaic.albedo[:, :, mm_]
            tile_c = lambda x: x.unsqueeze(1).expand_as(alb_c)
            # (core.fusion.run: this block one fused region on the graph path)
            def _canopy_pre():
                atten_c = canopy_on = fc_c = fcd_c = ground_pack = lai_c = ov_c = ov_step = _UNSET
                z0_c = mosaic.veg_rough[:, :, mm_]
                z0_c = torch.where(z0_c > 0.0, z0_c,
                                   tile_c(soil_params.thermal['rough']))
                d_c = mosaic.displacement[:, :, mm_]
                zref_c = energy.reference_height(mosaic.wind_h, d_c, z0_c)
                lai_c = mosaic.lai[:, :, mm_] * vegetation.cellwise(p['lai_scale'], mosaic.cv)
                # vic_run.c:136-140: with a fractional canopy the whole step
                # runs per plant, not per tile. LAI, the canopy's liquid and
                # its snow are divided by fcanopy on the way in and every
                # canopy output is multiplied back by it on the way out
                # (solve_snow.c:255-277). :222-225 then builds surf_atten
                # from the *divided* LAI, so a sparse canopy of thick
                # plants shades its ground less than a closed one of the
                # same tile-mean leaf area. FindleyLake's conifer tile is
                # fcanopy 0.62 to 0.96 by month and this was reading 1.
                fc_c = (torch.ones_like(alb_c) if mosaic.fcanopy is None
                        else mosaic.fcanopy[:, :, mm_])
                # VIC 5.1.0 divides only where fcanopy > 0 (vic_run.c 5.1.0
                # :219-226; ShortOverIn at solve_snow.c 5.1.0:173-175): 1 in
                # place of a 0 leaves LAI, the stores and ShortOverIn as the C
                # leaves them and keeps 0/0 out of both branches of every where.
                fcd_c = (torch.where(fc_c > 0.0, fc_c, torch.ones_like(fc_c))
                         if core_version.at_least('5.1.0') else fc_c)
                lai_c = lai_c / fcd_c
                atten_c = energy.surface_attenuation(lai_c, mosaic.rad_atten,
                                                     fc_c)
                thc = soil_params.thermal
                z0s_c = (thc['snow_rough'] if thc.get('snow_rough') is not None
                         else torch.full_like(thc['rough'], SNOW_Z0))
                u_c = (atm['wind'][_t1] if 'wind' in atm
                       else torch.full_like(atm['tair'][_t1], DEFAULT_WIND))
                u_c = torch.clamp(u_c, min=MIN_WIND)
                ov_c = vegetation.aerodynamic_resistance_overstory(
                    d_c / energy.VEG_RATIO_DH_HEIGHT, mosaic.trunk_ratio,
                    tile_c(z0s_c), tile_c(thc['rough']), mosaic.wind_atten,
                    z0_c, d_c, zref_c, tile_c(u_c))
                # The overstory profile on a tile that has none: a class with no
                # displacement has height 0, and CalcAerodynamic's canopy terms
                # (h / (n (Zw - d)), (d + z0) / h) give NaN for Ra[1] and U[1]
                # there -- 16 tile entries per step on Stehekin, one per cell.
                # Every reader selects overstory tiles with torch.where, so the
                # forward never saw them; the backward did. canopy_step runs on
                # every tile and masks afterwards, and its sensible heat,
                # C rho (Tcanopy - Tfoliage) / Ra_used[1] (physics.c:93), sent
                # 0 * NaN into Tfoliage's gradient: pcf, t_max_snow, t_min_rain,
                # new_snow_albedo, lai_scale and lai_water_factor all came back
                # NaN through 240 hourly Stehekin water-balance steps. A finite
                # stand-in where there is no overstory (a non-conducting canopy,
                # still air) changes no selected value and makes both branches
                # of every where finite.
                no_ov = ~over_t.unsqueeze(-1)
                ov_c = dict(ov_c)
                ov_c['ra'] = torch.where(
                    no_ov, torch.full_like(ov_c['ra'], vegetation.HUGE_RESIST),
                    ov_c['ra'])
                ov_c['u'] = torch.where(no_ov, torch.zeros_like(ov_c['u']),
                                        ov_c['u'])
                ov_step = ov_c                      # reused by the energy block
                # solve_snow.c:170-171: the canopy runs where the tile has a
                # pack, snow is falling on it, or it is already holding snow.
                # With a pack per tile the first test is per tile too.
                # snow->swq > 0 (solve_snow.c 5.1.0:164): liquid included
                ground_pack = ((snow_state['ice'] + snow_state['surf_water']
                                + snow_state['pack_water']) > 0.0)
                if not per_tile_snow:
                    ground_pack = ground_pack.unsqueeze(1)
                canopy_on = over_t & (
                    ground_pack | (snowfall_m > 0.0).unsqueeze(1)
                    | (canopy_state['int_snow'] > 0.0))
                return dict(atten_c=atten_c, canopy_on=canopy_on, fc_c=fc_c, fcd_c=fcd_c, ground_pack=ground_pack, lai_c=lai_c, ov_c=ov_c, ov_step=ov_step)
            _blk_res = _fusion.run(_canopy_pre)
            if _blk_res['atten_c'] is not _UNSET:
                atten_c = _blk_res['atten_c']
            if _blk_res['canopy_on'] is not _UNSET:
                canopy_on = _blk_res['canopy_on']
            if _blk_res['fc_c'] is not _UNSET:
                fc_c = _blk_res['fc_c']
            if _blk_res['fcd_c'] is not _UNSET:
                fcd_c = _blk_res['fcd_c']
            if _blk_res['ground_pack'] is not _UNSET:
                ground_pack = _blk_res['ground_pack']
            if _blk_res['lai_c'] is not _UNSET:
                lai_c = _blk_res['lai_c']
            if _blk_res['ov_c'] is not _UNSET:
                ov_c = _blk_res['ov_c']
            if _blk_res['ov_step'] is not _UNSET:
                ov_step = _blk_res['ov_step']
            # The canopy stage computes only on the tiles where it runs
            # (CANOPY_COMPACT): everything it returns is read through
            # `sel`/`liquid` below, both of which select canopy_on tiles,
            # and every operation in it is per tile, so the tiles it skips
            # could not change a selected value. Finding them is the one
            # host synchronisation the `.any()` test made before.
            # (Bit-identical only while the solver's iteration count does
            # not depend on the batch: Illinois, not bisect.)
            _canopy_compact = (CANOPY_COMPACT
                               and _implicit_mod.DEFAULT_METHOD == 'illinois')
            _con_g = None
            if graphsafe.ACTIVE:
                # graph-safe: on every covered overstory tile (a fixed set),
                # results masked to canopy_on below as the scatter of the
                # compacted stage leaves them; a step on which no tile runs
                # its canopy takes the eager branch (recorded).
                # A step on which no tile runs its canopy takes the eager
                # step's other branch; every value the rest of the step
                # reads from this one is then the same, except Tfoliage off
                # the overstory, selected below.
                _on_rc = _over_rc
                _cs_ = _snow_set.get('canopy')
                if _cs_ is not None:
                    # SNOW_SET: the window's snow lanes of the overstory;
                    # a canopy running outside them breaks the day
                    _on_rc = _cs_['rc']
                    graphsafe.need(canopy_on & ~_cs_['in'], _SNOW_SET_BIT)
                _canopy_any = True
            elif _canopy_compact:
                _on_rc = canopy_on.nonzero()
                _canopy_any = _on_rc.shape[0] > 0
            else:
                _canopy_any = bool(canopy_on.any())
            if not graphsafe.ACTIVE:
                graphsafe.LAST['canopy_any'] = bool(_canopy_any)
            if _canopy_any:
                if _canopy_compact or graphsafe.ACTIVE:
                    _on_r, _on_c = _on_rc[:, 0], _on_rc[:, 1]
                    _cshape = canopy_on.shape
                    if graphsafe.ACTIVE:
                        _con_g = canopy_on[_on_r, _on_c]

                    def g(x):
                        # a [cell, tile]-broadcastable tensor at the
                        # canopy's tiles, [n]
                        if not torch.is_tensor(x):
                            return x
                        return torch.broadcast_to(x, _cshape)[_on_r, _on_c]

                    def s_(v):
                        # back to [cell, tile]; the tiles the canopy did not
                        # run on hold zeros, which nothing selects
                        if not torch.is_tensor(v) or v.dim() == 0:
                            return v
                        out = torch.zeros(_cshape, dtype=v.dtype,
                                          device=v.device)
                        if _con_g is not None:
                            v = torch.where(_con_g, v, torch.zeros_like(v))
                        out[_on_r, _on_c] = v
                        return out

                    g_tree = lambda d: {k_: g(v_) for k_, v_ in d.items()}
                    s_tree = lambda d: {k_: s_(v_) for k_, v_ in d.items()}
                    if graphsafe.ACTIVE:
                        # the fixed overstory set of the graph path: every
                        # gather of a dict, and every scatter back, one
                        # fused region (a gather through the set's inverse
                        # map in place of zeros + index_put), the same
                        # values
                        _pk = (tuple(_cshape), _on_rc.shape[0])
                        _pv = _sub_maps.get(_pk)
                        if _pv is None:
                            _pos = torch.zeros(_cshape, dtype=torch.long,
                                               device=canopy_on.device)
                            _pos[_on_r, _on_c] = torch.arange(
                                _on_r.shape[0], device=canopy_on.device)
                            _val = torch.zeros(_cshape, dtype=torch.bool,
                                               device=canopy_on.device)
                            _val[_on_r, _on_c] = True
                            _pv = _sub_maps[_pk] = (_pos, _val)
                        g_tree = lambda d: _sub_gather(d, _on_r, _on_c,
                                                       _cshape)
                        s_tree = lambda d: _sub_scatter(d, _pv[0], _pv[1],
                                                        _con_g)
                else:
                    g = s_ = lambda x: x
                    g_tree = s_tree = lambda d: d
                # (core.fusion.run: this block one fused region on the graph path)
                def _canopy_in():
                    cf = cl_elev = cl_fc = cl_lai = cl_lwf = cl_ra = cl_rain = cl_rarc = cl_rgl = cl_vpd = cl_wdew = cs = _UNSET
                    tk = atm['tair'][_t1] + CONST_TKFRZ
                    lw_ground = (ground_long_out if ground_long_out is not None
                                 else tile_c(snow.calc_outgoing_longwave(
                                     tk, energy.EMISS_GRND)))
                    lv_t = atmos.latent_heat_of_vaporization(atm['tair'][_t1])
                    ra_wet = torch.where(torch.full_like(canopy_on, aero_406_full),
                                         ov_c['ra'][..., 1], ov_c['ra'][..., 0])
                    cf = dict(
                        rainfall=tile_c(rain_m), snowfall=tile_c(snowfall_m),
                        lai=lai_c,
                        max_int=vegetation.cellwise(lwf, lai_c) * lai_c / MM_PER_M,
                        wind1=ov_c['u'][..., 1], ra0=ov_c['ra'][..., 0],
                        ra1=ov_c['ra'][..., 1], air_dens=tile_c(atm['air_dens'][_t1]),
                        eact_air=tile_c(atm['eact_air'][_t1]),
                        press=tile_c(atm['press'][_t1]), lv=tile_c(lv_t),
                        t_canopy=(_sub['tcanopy']
                                   if _sub['tcanopy'] is not None
                                   else tile_c(atm['tair'][_t1])),
                        vpd=tile_c(atm['vpd'][_t1]),
                        long_over_in=tile_c(atm['longwave'][_t1]),
                        long_under_out=lw_ground,
                        # solve_snow.c:192, ShortOverIn /= fcanopy: what the
                        # canopy intercepts, spread over the plants alone.
                        short_over_in=((1.0 - atten_c)
                                       * tile_c(atm['shortwave'][_t1]) / fcd_c),
                        bare_albedo=alb_c,
                        dt=torch.full_like(alb_c, step_seconds),
                        aero_406=torch.full_like(canopy_on, aero_406),
                        aero_406_full=torch.full_like(canopy_on, aero_406_full),
                        aero_406_exact=torch.full_like(canopy_on,
                                                       aero_406_exact),
                        aero_410=torch.full_like(canopy_on, aero_410),
                        aero_410_on=bool(aero_410),
                        # func_canopy_energy_bal.c:180-183, AR_410: the
                        # canopy's own reference height, displacement and
                        # roughness (CalcAerodynamic.c rewrites [1] to them).
                        ref_height1=ov_c['ref_height'][..., 1],
                        displacement1=ov_c['displacement'][..., 1],
                        roughness1=ov_c['roughness'][..., 1],
                        # not float(): that would detach the parameter and
                        # p.new_snow_albedo would see no gradient through the
                        # canopy at all.
                        new_snow_albedo=(torch.zeros_like(alb_c)
                                         + new_snow_alb))
                    # Once per model step (divide_fcan). A later snow sub-step
                    # already holds solve_snow.c:250-251's ground-area store.
                    _fc_in = (fcd_c if _sub['divide_fcan']
                              else torch.ones_like(fcd_c))
                    cs = dict(int_snow=canopy_state['int_snow'] / _fc_in,
                              # vic_run.c 5.1.0:219-226 divides Wdew by
                              # fcanopy, then snow_intercept.c 5.1.0:127 by
                              # MM_PER_M: that order, to the last bit
                              int_rain=wdew / _fc_in / MM_PER_M,
                              t_foliage=canopy_state['t_foliage'],
                              temp_int_storage=canopy_state['temp_int_storage'])
                    cf = g_tree(cf)
                    cs = g_tree(cs)
                    # what canopy_liquid reads, at the canopy's tiles
                    _cl = g_tree(dict(
                        wdew=wdew / _fc_in, rain=tile_c(rain_m) * MM_PER_M,
                        lai=lai_c, rgl=mosaic.rgl, vpd=tile_c(atm['vpd'][_t1]),
                        ra=ra_wet, rarc=mosaic.rarc,
                        elev=tile_c(batch.elevation), fc=fc_c))
                    cl_wdew, cl_rain = _cl['wdew'], _cl['rain']
                    cl_lai, cl_rgl = _cl['lai'], _cl['rgl']
                    cl_vpd = _cl['vpd']
                    cl_ra, cl_rarc = _cl['ra'], _cl['rarc']
                    cl_elev = _cl['elev']
                    cl_fc = _cl['fc']
                    cl_lwf = (g(torch.broadcast_to(
                        vegetation.cellwise(lwf, lai_c), lai_c.shape))
                        if _canopy_compact and torch.is_tensor(lwf) else lwf)
                    return dict(cf=cf, cl_elev=cl_elev, cl_fc=cl_fc, cl_lai=cl_lai, cl_lwf=cl_lwf, cl_ra=cl_ra, cl_rain=cl_rain, cl_rarc=cl_rarc, cl_rgl=cl_rgl, cl_vpd=cl_vpd, cl_wdew=cl_wdew, cs=cs)
                _blk_res = _fusion.run(_canopy_in)
                if _blk_res['cf'] is not _UNSET:
                    cf = _blk_res['cf']
                if _blk_res['cl_elev'] is not _UNSET:
                    cl_elev = _blk_res['cl_elev']
                if _blk_res['cl_fc'] is not _UNSET:
                    cl_fc = _blk_res['cl_fc']
                if _blk_res['cl_lai'] is not _UNSET:
                    cl_lai = _blk_res['cl_lai']
                if _blk_res['cl_lwf'] is not _UNSET:
                    cl_lwf = _blk_res['cl_lwf']
                if _blk_res['cl_ra'] is not _UNSET:
                    cl_ra = _blk_res['cl_ra']
                if _blk_res['cl_rain'] is not _UNSET:
                    cl_rain = _blk_res['cl_rain']
                if _blk_res['cl_rarc'] is not _UNSET:
                    cl_rarc = _blk_res['cl_rarc']
                if _blk_res['cl_rgl'] is not _UNSET:
                    cl_rgl = _blk_res['cl_rgl']
                if _blk_res['cl_vpd'] is not _UNSET:
                    cl_vpd = _blk_res['cl_vpd']
                if _blk_res['cl_wdew'] is not _UNSET:
                    cl_wdew = _blk_res['cl_wdew']
                if _blk_res['cs'] is not _UNSET:
                    cs = _blk_res['cs']

                def canopy_liquid(tf, net_rad, net_short_over):
                    # func_canopy_energy_bal.c:229-247: with no intercepted
                    # snow the canopy's liquid is canopy_evap with CALC_EVAP
                    # false -- evaporation, the store it leaves and the
                    # throughfall, all mm. The energy path keeps the
                    # arguments its bit-exact tests were recorded with:
                    # the tile's store and Tfoliage. The water-balance path
                    # hands canopy_evap what :239-243 hands it: the store
                    # per plant (vic_run.c:137 divided Wdew by fcanopy) and
                    # the canopy air temperature, `Tcanopy`, which is the air
                    # temperature (CLOSE_ENERGY false: calc_atmos_energy_bal.c 5.1.0:73-103).
                    return vegetation.canopy_interception(
                        cl_wdew, cl_rain, cl_lai, cl_rgl,
                        net_short_over, cf['t_canopy'],
                        cl_vpd, cl_ra, cl_rarc, net_rad, cl_elev,
                        dt=step_seconds, lai_water_factor=cl_lwf,
                        # VIC's own wetness on the water-balance path; the
                        # energy path keeps the value its bit-exact tests
                        # were recorded with (vegetation.canopy_wetness).
                        exact_wetness=True)

                v51_c = core_version.at_least('5.1.0')

                _evap_prep = {}

                def evap_of_canopy(tf, net_rad, net_short_over):
                    # the latent heat's evaporation (:245-246), m/s
                    if use_eb or torch.is_grad_enabled():
                        ce = canopy_liquid(tf, net_rad, net_short_over)[0]
                    else:
                        # water balance: only the net radiation changes from
                        # one trial temperature to the next (the canopy air
                        # is Tair), so the rest is formed once a step
                        if _evap_prep.get('ns') is not net_short_over:
                            _evap_prep['ns'] = net_short_over
                            _evap_prep['fn'] = \
                                vegetation.canopy_interception_evap(
                                    cl_wdew, cl_rain, cl_lai, cl_rgl,
                                    net_short_over, cf['t_canopy'], cl_vpd,
                                    cl_ra, cl_rarc, cl_elev, dt=step_seconds,
                                    lai_water_factor=cl_lwf,
                                    exact_wetness=True)
                        ce = _evap_prep['fn'](net_rad)
                    if v51_c:
                        # func_canopy_energy_bal.c 5.1.0:228-229, "account
                        # for fractional canopy cover": *Evap *= fcanopy
                        # before the latent heat is formed from it.
                        ce = ce * cl_fc
                    return ce / MM_PER_M / step_seconds

                graphsafe.MASK = _con_g
                try:
                    # The canopy keeps the established product division
                    # on both paths: snow_intercept.c, massrelease.c
                    # and canopy_evap.c have no SPATIAL_SNOW branch,
                    # and VIC's TRUE/FALSE canopy outputs are
                    # bit-identical through step 558 (first difference
                    # at 559 via ground feedback). Gating the C-order
                    # division on SPATIAL_SNOW forked our TRUE canopy
                    # from our FALSE at step 444 instead.
                    # snow_intercept.c 5.1.0:425-427's own division order
                    # for the melt, -RefreezeEnergy / LATICE / RHOFW
                    new_cs, cout = canopy.canopy_step(
                        cs, cf, evap_of_canopy, vic_canopy_rounding=True)
                finally:
                    graphsafe.MASK = None
                if True:   # both modes (func_canopy_energy_bal.c 5.1.0:220-230)
                    # The liquid branch's mass. canopy_step spends the
                    # evaporation above on the canopy's energy only: its
                    # store and throughfall are snow_intercept.c:243-260's
                    # interception, which VIC overwrites on this branch --
                    # *Wdew is IntRain's pointer (:309, :724), reset from
                    # IntRainOrg and passed through canopy_evap, and
                    # RainThroughFall becomes veg_var->throughfall (:404-406)
                    # -- and veg_var->canopyevap is what put_data.c:639
                    # reports as OUT_EVAP_CANOP. So on a canopy that ran
                    # without snow the liquid evaporated in VIC and not here,
                    # and the water-balance path reported no canopy
                    # evaporation under snow where VIC did (Stehekin, days
                    # 5-6, cell sums 0.02285 and 0.03131 mm).
                    ce_l, wd_l, tf_l, _ = canopy_liquid(
                        new_cs['t_foliage'], cout['net_radiation'],
                        cout['net_short_over'])
                    if v51_c:
                        # func_canopy_energy_bal.c 5.1.0:230, *Wdew *=
                        # fcanopy on the store it hands back (IntRain), and
                        # solve_snow.c:251 multiplies it by fcanopy again
                        # below: VIC 5.1.0 keeps fcanopy^2 of the plants'
                        # store on this branch.
                        wd_l = wd_l * cl_fc
                    _lq = s_tree(dict(ce=ce_l, wd=wd_l, tf=tf_l))
                    ce_l, wd_l, tf_l = _lq['ce'], _lq['wd'], _lq['tf']
                new_cs = s_tree(new_cs)
                cout = s_tree(cout)
                # snow_intercept.c:342, only where the canopy held snow
                # (the liquid branch solves no temperature)
                _tfx.note('canopy foliage',
                          canopy_on & cout['has_snow'] & cout['fallback'])
                # From here to the canopy's outputs: one fused region on the
                # graph path (core.fusion.run), the same code eagerly.
                _tair_t, _lw_t, _sw_t = (atm['tair'][_t1], atm['longwave'][_t1],
                                         atm['shortwave'][_t1])
                under_short = under_long = None

                def _canopy_post(wdew, rain_m, snowfall_m, rain, snowfall,
                                 under_short, under_long):
                    if True:   # both modes
                        liquid = canopy_on & ~cout['has_snow']
                        new_cs['int_rain'] = torch.where(
                            liquid, wd_l / MM_PER_M, new_cs['int_rain'])
                        cout['rain_through'] = torch.where(
                            liquid, tf_l / MM_PER_M, cout['rain_through'])
                        # per tile, mm, back to the whole tile (solve_snow.c:269)
                        wb_canopy_liquid = torch.where(
                            liquid, ce_l * fc_c, torch.zeros_like(ce_l))
                    sel = lambda a, b: torch.where(canopy_on, a, b)
                    # solve_snow.c:255-277 scales every canopy result back
                    # from per-plant to per-tile.
                    canopy_state['int_snow'] = sel(
                        new_cs['int_snow'] * fc_c, canopy_state['int_snow'])
                    _tf_old = canopy_state['t_foliage']
                    canopy_state['t_foliage'] = sel(new_cs['t_foliage'],
                                                    tile_c(_tair_t))
                    if graphsafe.ACTIVE:
                        # the eager step with no canopy anywhere (below)
                        canopy_state['t_foliage'] = torch.where(
                            canopy_on.any(), canopy_state['t_foliage'],
                            torch.where(over_t, tile_c(_tair_t),
                                        _tf_old))
                    canopy_state['temp_int_storage'] = sel(
                        new_cs['temp_int_storage'], canopy_state['temp_int_storage'])
                    wdew = sel(new_cs['int_rain'] * MM_PER_M * fc_c, wdew)
                    # :258-265, the throughfall rescale: the bare fraction of
                    # the tile passes its precipitation straight through.
                    # In mm, as the C blends them: snow_intercept hands the
                    # throughfall back in mm (snow_intercept.c 5.1.0:534-535),
                    # solve_snow.c:241-248 weights it against the sky's mm, and
                    # snow_melt.c:83-84 divides by MM_PER_M after. Blending in
                    # metres rounds an ulp apart.
                    rain_tile = sel(((1.0 - fc_c) * tile_c(rain)
                                     + fc_c * (cout['rain_through'] * MM_PER_M))
                                    / MM_PER_M,
                                    tile_c(rain_m))
                    snow_tile_mm = sel((1.0 - fc_c) * tile_c(snowfall)
                                       + fc_c * (cout['snow_through'] * MM_PER_M),
                                       tile_c(snowfall))
                    snow_tile = sel(snow_tile_mm / MM_PER_M, tile_c(snowfall_m))
                    # The cell's totals still stand for the water balance and the
                    # reported OUT_RAINF / OUT_SNOWF; the per-tile values go to
                    # the per-tile packs.
                    rain_m = _tsum(rain_tile * mosaic.cv)
                    snowfall_m = _tsum(snow_tile * mosaic.cv)
                    rain = rain_m * MM_PER_M
                    snowfall = snowfall_m * MM_PER_M
                    canopy_long_out = sel(cout['long_over_out'],
                                          tile_c(_lw_t))
                    # What the ground under the canopy sees, cover-weighted to
                    # the cell the pack lives on: the shortwave attenuated by
                    # the canopy on overstory tiles (solve_snow.c:185) and the
                    # canopy's emission in place of the sky's (snow_intercept
                    # writes LongUnderIn). Tiles without an overstory keep the
                    # sky, as the C does.
                    # solve_snow.c:161 sets ShortUnderIn to the full
                    # shortwave and :190 multiplies it by surf_atten only
                    # inside the overstory branch, which runs only when
                    # there is snow somewhere (:164-165). An overstory
                    # tile with no snow gives its ground the whole sun in
                    # VIC -- surf_atten reaches the ground flux and the
                    # ARNO radiation there, never the net shortwave.
                    # Gating on `over_t` instead shaded every forested
                    # tile on a day when any other tile still had a pack.
                    atten_used = torch.where(canopy_on, atten_c,
                                             torch.ones_like(atten_c))
                    short_t = tile_c(_sw_t) * atten_used
                    if per_tile_snow:
                        # each tile's pack sees what got through its own canopy
                        under_short, under_long = short_t, canopy_long_out
                    else:
                        under_short = _tsum(short_t * mosaic.cv)
                        under_long = _tsum(canopy_long_out * mosaic.cv)
                    # calc_atmos_energy_bal.c:78-80, with snow under an
                    # overstory: NetLongAtmos is the *canopy's* net longwave
                    # alone (F = 1), and NetShortAtmos is the canopy's plus the
                    # understory's. Reporting the ground's longwave instead put
                    # OUT_LWNET at -10.1 W/m2 against VIC's -26.4 over a winter
                    # at FindleyLake. Where no canopy ran, surface_fluxes.c
                    # :686-689 takes the understory's, which is what the
                    # energy block already reports.
                    # put_data.c:903-909 reports AlbedoOver on an
                    # overstory tile and AlbedoUnder elsewhere, and
                    # snow_intercept.c:298 and :332 set AlbedoOver to the
                    # new-snow albedo when the canopy holds snow and to
                    # the class albedo when it does not. Where the snow
                    # branch did not run at all it is zero
                    # (solve_snow.c:490, :508) -- a zero VIC really does
                    # report, which is why its OUT_ALBEDO over a forested
                    # winter is 0.49 and not the 0.85 of the snow itself.
                    # put_data.c:676-685 reports an overstory tile's
                    # aerodynamic resistance from cell.aero_resist[1], which
                    # surface_fluxes.c:243 aliases to aero_resist_used --
                    # the array func_canopy_energy_bal.c wrote Ra_used[1]
                    # into. Where the canopy did not run this step it is
                    # CalcAerodynamic's own second entry.
                    ra_veg1_t = sel(cout['ra_used1'], ov_c['ra'][..., 1])
                    # snow_intercept.c 5.1.0:292-294, :331-333 decide on the
                    # store after interception (a store under
                    # SNOW_MIN_SWQ_EB_THRES has fallen through by then) and the
                    # sky's snowfall -- canopy_step's own albedo_over. Testing
                    # the step's end store and the throughfall instead kept
                    # 0.85 on the step a thin canopy load dropped.
                    alb_over_t = sel(cout['albedo_over'], torch.zeros_like(alb_c))
                    # solve_snow.c 5.1.0:470-476: with no pack on the ground and
                    # no snow reaching it, AlbedoOver is reset to 0 after the
                    # canopy ran.
                    _gp = ground_pack if per_tile_snow else ground_pack.expand_as(alb_c)
                    alb_over_t = torch.where(_gp | (snow_tile > 0.0), alb_over_t,
                                             torch.zeros_like(alb_over_t))
                    canopy_nl = sel(cout['net_long_over'] * fc_c,
                                    torch.zeros_like(alb_c))
                    canopy_ns = sel(cout['net_short_over'] * fc_c,
                                    torch.zeros_like(alb_c))
                    canopy_ran = canopy_on
                    # calc_atmos_energy_bal.c 5.1.0:64-66: the canopy's latent
                    # heat joins AtmosLatent (OUT_LATENT) and its latent heat of
                    # sublimation AtmosLatentSub (OUT_LATENT_SUB, put_data.c
                    # 5.1.0:893-896), separately; surface_fluxes.c 5.1.0:633-655
                    # runs it on an overstory tile with snow in either mode.
                    canopy_lat = sel(cout['latent'] * fc_c,
                                     torch.zeros_like(alb_c))
                    canopy_lsub = sel(cout['latent_sub'] * fc_c,
                                      torch.zeros_like(alb_c))
                    # put_data.c 5.1.0:918-930: on an overstory tile with snow
                    # OUT_ADVECTION and OUT_RFRZ_ENERGY add the canopy's
                    # advection and refreeze energy, which solve_snow.c 5.1.0
                    # :253-257 scales by fcanopy; OUT_REFREEZE is formed from
                    # the sum (:493-494).
                    canopy_adv = sel(cout['advected'] * fc_c,
                                     torch.zeros_like(alb_c))
                    canopy_rfz = sel(cout['canopy_refreeze'] * fc_c,
                                     torch.zeros_like(alb_c))
                    canopy_adv, canopy_rfz = _canopy_carry(
                        canopy_on, over_t, canopy_adv, canopy_rfz)
                    canopy_sen = sel(cout['sensible'] * fc_c,
                                     torch.zeros_like(alb_c))
                    _sub_canop = (
                        _tsum(sel(cout['vapor_flux'] * fc_c,
                                  torch.zeros_like(alb_c)) * mosaic.cv)
                        * MM_PER_M)
                    return dict(_sub_canop=_sub_canop, liquid=liquid, wb_canopy_liquid=wb_canopy_liquid, wdew=wdew, rain_tile=rain_tile, snow_tile_mm=snow_tile_mm, snow_tile=snow_tile, rain_m=rain_m, snowfall_m=snowfall_m, rain=rain, snowfall=snowfall, canopy_long_out=canopy_long_out, under_short=under_short, under_long=under_long, ra_veg1_t=ra_veg1_t, alb_over_t=alb_over_t, canopy_nl=canopy_nl, canopy_ns=canopy_ns, canopy_ran=canopy_ran, canopy_lat=canopy_lat, canopy_lsub=canopy_lsub, canopy_adv=canopy_adv, canopy_rfz=canopy_rfz, canopy_sen=canopy_sen)
                _cp = _fusion.run(_canopy_post, wdew, rain_m, snowfall_m,
                                  rain, snowfall, under_short, under_long)
                liquid = _cp['liquid']
                wb_canopy_liquid = _cp['wb_canopy_liquid']
                wdew = _cp['wdew']
                rain_tile = _cp['rain_tile']
                snow_tile_mm = _cp['snow_tile_mm']
                snow_tile = _cp['snow_tile']
                rain_m = _cp['rain_m']
                snowfall_m = _cp['snowfall_m']
                rain = _cp['rain']
                snowfall = _cp['snowfall']
                canopy_long_out = _cp['canopy_long_out']
                under_short = _cp['under_short']
                under_long = _cp['under_long']
                ra_veg1_t = _cp['ra_veg1_t']
                alb_over_t = _cp['alb_over_t']
                canopy_nl = _cp['canopy_nl']
                canopy_ns = _cp['canopy_ns']
                canopy_ran = _cp['canopy_ran']
                canopy_lat = _cp['canopy_lat']
                canopy_lsub = _cp['canopy_lsub']
                canopy_adv = _cp['canopy_adv']
                canopy_rfz = _cp['canopy_rfz']
                canopy_sen = _cp['canopy_sen']
                # (appended out here: a list a fused region appended to
                # would be guarded on its length, which grows every step)
                sub_canops.append(_cp['_sub_canop'])
            else:
                alb_over_t = torch.zeros_like(over_t, dtype=dtype)
                canopy_state['t_foliage'] = torch.where(
                    over_t, tile_c(atm['tair'][_t1]), canopy_state['t_foliage'])
                sub_canops.append(torch.zeros_like(rain_m))
                if _sub['pin_tgrnd']:
                    _z = torch.zeros_like(canopy_state['int_snow'])
                    canopy_adv, canopy_rfz = _canopy_carry(
                        canopy_on, over_t, _z, _z)
                    canopy_lsub = _z
            snow_canopies.append(_tsum(canopy_state['int_snow'] * mosaic.cv)
                                 * MM_PER_M)
            vegt_canopy = canopy_state['t_foliage']

        # (core.fusion.run: this block one fused region on the graph path)
        def _snow_prep(wdew, last_snow):
            _ls = _on = _th = alb_under = albedo = bare_alb = cold_content = cov_prev = cov_use = f = ra_snow = rain_p = rain_past_pack = snow_p = sp_old_depth = sp_old_swq = st = t_grnd = wind_snow = z0_snow_t = _UNSET
            # What reaches each tile's pack: the canopy's throughfall where a
            # canopy stood over it this step, the cell's precipitation where
            # none did. `rain_tile`/`snow_tile` are set by the canopy stage.
            if per_tile_snow:
                rain_p = (rain_tile if rain_tile is not None
                          else rain_m.unsqueeze(-1).expand(nc, ntl_s))
                snow_p = (snow_tile if snow_tile is not None
                          else snowfall_m.unsqueeze(-1).expand(nc, ntl_s))
                # the same in mm, as solve_snow.c hands *snowfall to
                # snow_density (:346) before snow_melt divides it (:83)
                snow_p_mm = (snow_tile_mm if snow_tile_mm is not None
                             else snowfall.unsqueeze(-1).expand(nc, ntl_s))
                if (core_version.at_least('5.1.0') and mosaic is not None
                        and mosaic.fcanopy is not None):
                    # solve_snow.c 5.1.0:236-248: on every vegetated tile with
                    # snow (iveg != Nveg, snow->snow) the precipitation is
                    # rescaled to the tile, (1 - fc) out + fc through, even
                    # where no overstory changed it -- not an identity in
                    # floating point. Overstory tiles were blended above; the
                    # Wdew dump below keeps its own form.
                    _fc = mosaic.fcanopy[:, :, months[_t1]]
                    _veg = torch.ones_like(_fc, dtype=torch.bool)
                    if mosaic.bare_tile is not None:
                        _veg = ~mosaic.bare_tile
                    _on = ((snow_state['ice'] + snow_state['pack_water']
                            + snow_state['surf_water']) > 0.0) | (snow_p_mm > 0.0)
                    _bl = _veg & _on
                    if over_t is not None:
                        _bl = _bl & ~over_t
                    _sbl = (1.0 - _fc) * snow_p_mm + _fc * snow_p_mm
                    snow_p = torch.where(_bl, _sbl / MM_PER_M, snow_p)
                    snow_p_mm = torch.where(_bl, _sbl, snow_p_mm)
                    _r_mm = rain_mm0.unsqueeze(-1).expand(nc, ntl_s)
                    _nodump = ~((snow_p > 0.0) & (wdew > 0.0) & (_fc > 0.0))
                    rain_p = torch.where(
                        _bl & _nodump,
                        ((1.0 - _fc) * _r_mm + _fc * _r_mm) / MM_PER_M, rain_p)
            else:
                rain_p, snow_p = rain_m, snowfall_m
                snow_p_mm = snowfall

            # snow->coldcontent as the previous step left it (solve_snow.c
            # 5.1.0:298-302 hands it to snow_albedo); a state without it falls
            # back to the surface temperature's sign.
            cold_content = (snow_state['cold_content']
                            if 'cold_content' in snow_state
                            else snow.CONST_VCPICE_WQ * snow_state['surf_temp']
                            * snow_state['ice'])
            # solve_snow.c:311-327. The age advances, and the decay curve is
            # consulted, only on a step with no snowfall from the sky. Any sky
            # snowfall at all takes the else branch at :323-327, which zeroes
            # the age and sets the albedo to the new-snow value outright -- it
            # does not call snow_albedo() and tests no trace threshold. The
            # test is on `store_snowfall`, what the sky delivered (:146), not
            # on what reached the ground through the canopy; the C's own
            # comment at :312-313 says so.
            #
            # This used to advance the age unless the *throughfall* exceeded
            # TRACE_SNOW, and with TRACE_SNOW misread as 0.03 m rather than
            # 0.03 mm the reset essentially never fired.
            # The sky's snowfall: snowfall_m is the canopy's throughfall by now
            # where a canopy ran, and a canopy dropping its load onto the pack
            # restarted the albedo decay where VIC ages it. And snow->swq is
            # the whole pack, liquid included (snow_melt.c 5.1.0:464), not its
            # ice (solve_snow.c 5.1.0:296).
            sky_fall = (sky_snow_m.unsqueeze(-1).expand_as(last_snow)
                        if per_tile_snow else sky_snow_m)
            aging = (((snow_state['ice'] + snow_state['surf_water']
                       + snow_state['pack_water']) > 0.0) & (sky_fall <= 0.0))
            last_snow = torch.where(aging, last_snow + 1.0,
                                    torch.zeros_like(last_snow))
            albedo = torch.where(
                aging,
                snow.snow_albedo(
                    snow_p, new_snow_alb,
                    snow_state['ice'] + snow_state['surf_water']
                    + snow_state['pack_water'], cold_content, step_seconds, last_snow, melting,
                    trace_snow=snow.TRACE_SNOW_M,
                    accum_a=p['alb_accum_a'], accum_b=p['alb_accum_b'],
                    thaw_a=p['alb_thaw_a'], thaw_b=p['alb_thaw_b']),
                torch.zeros_like(cold_content) + new_snow_alb)

            # ---- SPATIAL_SNOW, the part that runs before the pack solves -------
            # solve_snow.c:176 takes the cover fraction the previous step left,
            # and everything from here to the pack's own solve is conditioned on
            # it. `alb_under` is AlbedoUnder: with the option off it is the snow
            # albedo under a pack and the class albedo without one, which is what
            # the rest of this loop already assumed.
            alb_under = albedo
            rain_past_pack = None
            if spatial_snow:
                cov_prev = sp['coverage']
                # surface_fluxes.c:234 sets the mid-step cover `coverage` once a
                # call; its address goes to every sub-step's solve_snow, which
                # writes it only in the delta_coverage branches
                # (solve_snow.c:408,423,427,434). A daily step's later snow
                # sub-steps therefore see the previous sub-step's mid-step
                # cover in snow_melt (:318,:333) and the albedo mix (:304),
                # not the cover it left (a 3-hourly step re-reads it, :234).
                cov_use = (cov_prev if (_sub['flush'] or _sub['cov_star'] is None)
                           else _sub['cov_star'])
                sp_old_swq = (snow_state['ice'] + snow_state['surf_water']
                              + snow_state['pack_water'])
                sp_old_depth = snow_state['depth']
                # :297-300, new snow makes the pack uniform at its mean depth.
                cov_a = torch.where(snow_p > 0.0, torch.ones_like(cov_prev),
                                    cov_prev)
                # :301-307, rain falls evenly over the tile, so the part of it
                # that lands on bare ground never enters the pack at all. That
                # water is the C's *ppt and goes straight to the soil.
                split = (cov_a > 0.0) & (snow_p <= 0.0) & (cov_a < 1.0)
                rain_past_pack = torch.where(split, rain_p * (1.0 - cov_a),
                                             torch.zeros_like(rain_p))
                rain_p = torch.where(split, rain_p * cov_a, rain_p)
                # solve_snow.c:294-312. The albedo branch tests
                # `store_snowfall` -- what the sky delivered, not what reached
                # the ground through the canopy: with no sky snowfall the aged
                # albedo mixes with the bare ground at the previous step's
                # cover, even while the canopy drips snow onto the pack. A
                # throughfall test here unmixed it on exactly those steps and
                # starved the pack of 3.5 W/m2 (March 10: VIC 57.25, TV 53.71
                # NetShortSnow). The rain split above correctly tests the
                # throughfall (:284-290), so it keeps `cov_a`. The mix also
                # needs a pack: with swq at zero the C takes the else arm and
                # AlbedoUnder is the new snow albedo unmixed, even with no sky
                # snowfall (canopy drip onto bare ground).
                mix_alb = (sp_old_swq > 0.0) & (sky_fall <= 0.0)
                if mosaic is not None and mosaic.albedo is not None:
                    bare_alb = mosaic.albedo[:, :, months[_t1]]
                else:
                    bare_alb = torch.full_like(cov_a, LAND_ALBEDO)
                alb_under = torch.where(
                    mix_alb,
                    cov_use * albedo + (1.0 - cov_use) * bare_alb, albedo)

            if per_tile_snow:
                # (both modes: solve_snow.c 5.1.0:222-231 is not gated on
                # FULL_ENERGY)
                # solve_snow.c:222-231: snow falling on a tile without an
                # overstory empties the canopy store into the rain that reaches
                # the pack (rainfall += Wdew; Wdew = 0), in tile units after the
                # fcanopy rescale at :244-253. OUT_RAINF was recorded above,
                # before the dump, as VIC's out_rain is.
                dump = (snow_p > 0.0) & (wdew > 0.0)
                if over_t is not None:
                    dump = dump & ~over_t
                if (core_version.at_least('5.1.0') and mosaic is not None
                        and mosaic.fcanopy is not None):
                    # With fcanopy 0 (VIC 5.1.0, a leafless month) the rescale
                    # at :242-251 gives rainfall = out_rain and Wdew = 0: the
                    # store is deleted rather than dumped. mosaic.step zeroes
                    # it (Wdew *= fcanopy under snow).
                    dump = dump & (mosaic.fcanopy[:, :, months[_t1]] > 0.0)
                _dumped = wdew
                if (not _sub['divide_fcan'] and core_version.at_least('5.1.0')
                        and mosaic is not None and mosaic.fcanopy is not None):
                    # a later snow step: Wdew is read per plant as it stands
                    # and :241-243 rescales the rain by fcanopy
                    _dumped = mosaic.fcanopy[:, :, months[_t1]] * wdew
                rain_p = rain_p + torch.where(dump, _dumped / MM_PER_M,
                                              torch.zeros_like(wdew))
                wdew = torch.where(dump, torch.zeros_like(wdew), wdew)
            # VIC adds rainfall to the surface liquid store before the balance.
            st = dict(snow_state)
            st['sw_pre'] = st['surf_water']
            st['surf_water'] = st['surf_water'] + rain_p
            # surface_fluxes.c:389: Tgrnd is the tile's own top soil node, which
            # with a pack per tile is now exactly that rather than a
            # cover-weighted stand-in for one shared pack.
            if use_eb:
                t_grnd = (t_node[:, :, 0] if per_tile_snow
                          else _tsum(t_node[:, :, 0] * mosaic.cv))
            elif wb_mode and per_tile_snow:
                # In water-balance mode T[0] is the previous step's Tsurf, which
                # is the previous step's Tair (0 C before the first step), not
                # the 0 C this used to hand every pack.
                t_grnd = wb_t0
            else:
                t_grnd = None
            if _sub['pin_tgrnd'] and t_grnd is not None:
                # surface_fluxes.c:362 reads Tgrnd from energy->T[0], and
                # `energy` is written only after the snow steps (:959), so every
                # snow step of a model step sees the node it opened with.
                if _sub['t_grnd'] is None:
                    _sub['t_grnd'] = t_grnd.clone()
                t_grnd = _sub['t_grnd']
            # The pack under a canopy takes CalcAerodynamic's snow-surface
            # entries (index 2), not the open-field ones.
            ra_snow = wind_snow = z0_snow_t = None
            if per_tile_snow:
                # The open-field (no-overstory) pack: CalcAerodynamic.c 5.1.0
                # :97-104 with the cell's own snow roughness Z0_SNOW =
                # soil_con->snow_rough (vic_run.c 5.1.0:196-201) and the tile's
                # own reference height ref_height[0] -- its class's wind_h, or
                # d + wind_h + z0 above a tall canopy (vic_run.c 5.1.0:168,
                # 186-191) -- not one constant roughness (0.03 m) and the global
                # file's WIND_H for every tile. snow_melt gets Ra[2], U[2],
                # ref_height[2] = 2 + Z0_SNOW and roughness[2] = Z0_SNOW
                # (solve_snow.c 5.1.0:315-322; SnowPackEnergyBalance.c 5.1.0:167
                # corrects Ra with those). On ISC's Tibetan grassland (wind_h
                # 3 m, snow_rough 0.0005 m) the old form gave a resistance 3x
                # too small, and the pack sublimated away.
                u_open = (atm['wind'][_t1] if 'wind' in atm
                          else torch.full_like(atm['tair'][_t1], DEFAULT_WIND))
                u_open = torch.clamp(u_open, min=MIN_WIND).unsqueeze(-1)
                _th = soil_params.thermal
                z0s_cell = (_th['snow_rough'] if _th.get('snow_rough') is not None
                            else torch.full_like(atm['tair'][_t1], SNOW_Z0))
                z0_snow_t = z0s_cell.unsqueeze(-1).expand(nc, ntl_s)
                if mosaic.wind_h is not None:
                    _m = months[_t1]
                    z0_v = mosaic.veg_rough[:, :, _m]
                    if _th.get('rough') is not None:
                        z0_v = torch.where(z0_v > 0.0, z0_v,
                                           _th['rough'].unsqueeze(-1)
                                           .expand_as(z0_v))
                    z_ref_t = energy.reference_height(
                        mosaic.wind_h, mosaic.displacement[:, :, _m], z0_v)
                else:
                    z_ref_t = torch.full_like(z0_snow_t, float(z_wind))
                lr = torch.log((2.0 + z0_snow_t) / z0_snow_t)
                lz = torch.log(z_ref_t / z0_snow_t)
                # CalcAerodynamic.c 5.1.0:76-81 then :178-185, in the C's
                # order: Ra[2] = lr lz / K2, then /= wind; U[2] = lr / lz, then
                # *= wind (the other groupings round an ulp apart).
                ra_snow = lr * lz / VON_KARMAN_SQ / u_open
                wind_snow = lr / lz * u_open
                if any_over and ov_step is not None:
                    ra_snow = torch.where(over_t, ov_step['ra'][..., 2], ra_snow)
                    # CalcAerodynamic.c 5.1.0 U[2] as it comes: the forcing is
                    # floored (MIN_WIND, ours) but the canopy-attenuated wind at
                    # the pack is not; flooring it again put 0.1 m/s in the
                    # pack's stability correction where VIC had 0.07 (Stehekin)
                    wind_snow = torch.where(over_t, ov_step['u'][..., 2],
                                            wind_snow)
            blow_flux = None
            if blowing:
                _thc = soil_params.thermal
                z0s_snow = (_thc['snow_rough']
                            if _thc.get('snow_rough') is not None
                            else torch.full_like(_thc['rough'], SNOW_Z0))
                # surface_fluxes.c:443-463. The wind, roughness and reference
                # height are CalcAerodynamic's snow-surface entries, index 2,
                # the same ones the pack's own balance uses; the displacement
                # and roughness passed separately are the vegetation's, index 1
                # (:456-458). `iveg == Nveg` at CalcBlowingSnow.c:99 is the
                # bare-soil tile, which takes a fixed fetch and slope variance.
                from ..core import blowing as _blow
                _u = (wind_snow if wind_snow is not None
                      else _bcast(torch.clamp(
                          atm['wind'][_t1] if 'wind' in atm
                          else torch.full_like(atm['tair'][_t1], DEFAULT_WIND),
                          min=MIN_WIND), snow_state['ice']))
                _z0 = _bcast(z0s_snow, snow_state['ice'])
                _lst = _bcast(last_snow, snow_state['ice'])
                _ls = snow.calc_latent_heat_of_sublimation(
                    snow_state['surf_temp'])
                blow_flux, _ = _blow.calc_blowing_snow(
                    dt=torch.full_like(snow_state['ice'], step_seconds),
                    tair=_bcast(atm['tair'][_t1], snow_state['ice']),
                    last_snow=_lst,
                    surface_liquid_water=snow_state['surf_water'],
                    wind=_u, ls=_ls,
                    air_dens=_bcast(atm['air_dens'][_t1], snow_state['ice']),
                    eact_air=_bcast(atm['eact_air'][_t1], snow_state['ice']),
                    z0=_z0, zrh=2.0 + _z0,   # ref_height[2] (:425)
                    snowdepth=snow_state['depth'],
                    lag_one=_bcast(mosaic.lag_one, snow_state['ice']),
                    sigma_slope=_bcast(mosaic.sigma_slope, snow_state['ice']),
                    tsnow=snow_state['surf_temp'],
                    fe=_bcast(mosaic.fetch, snow_state['ice']),
                    displacement=_bcast(mosaic.displacement[:, :, months[_t1]],
                                        snow_state['ice']),
                    roughness=_bcast(mosaic.veg_rough[:, :, months[_t1]],
                                     snow_state['ice']),
                    bare_soil=_bcast(mosaic.overstory <= 0.5,
                                     snow_state['ice']),
                    **_blow_opt)
                # :443, no blowing snow under an overstory, and none on a
                # padded tile with no cover -- that one has no vegetation
                # geometry to shelter the pack and its zero cover would turn
                # any non-finite value into a NaN for the whole cell.
                _off = _bcast(mosaic.cv <= 0.0, blow_flux)
                if any_over:
                    _off = _off | _bcast(over_t, blow_flux)
                blow_flux = torch.where(_off, torch.zeros_like(blow_flux),
                                        blow_flux)
            f = snow_forcing_at(atm, _t1, rain_p, snow_p, alb_under,
                                meas_height=z_wind, dt=step_seconds,
                                depth=snow_state.get('depth'),
                                density=snow_state.get('density'),
                                t_grnd=t_grnd, shortwave=under_short,
                                longwave=under_long, per_tile=per_tile_snow,
                                ra_snow=ra_snow, wind_snow=wind_snow,
                                z0_snow=z0_snow_t)
            f['snowfall_mm'] = snow_p_mm
            if _sub['tcanopy'] is not None:
                # CLOSE_ENERGY: the pack's own balance runs at the canopy air
                # temperature (snow_melt.c:159); the snowfall cold content,
                # density and ageing stay at the air's (snow_step).
                f['t_canopy'] = _sub['tcanopy']
            # snow->last_snow == 0 this step (solve_snow.c 5.1.0:296-309), for
            # a thin pack's density (snow.snow_step)
            f['last_snow_zero'] = ~aging
            if blow_flux is not None:
                # SnowPackEnergyBalance.c:217-235 adds it to the pack's own
                # vapour flux. The round trip surface_fluxes.c:463 makes --
                # kg/m2/s to metres per step and back again at :218 -- is the
                # identity, so the flux goes in as CalcBlowingSnow returns it.
                f['blowing_mass_flux'] = blow_flux
            if spatial_snow:
                # SnowPackEnergyBalance.c:200-211: the bare part of a partly
                # covered tile advects sensible heat onto the pack. The C passes
                # snow_melt the caller's cover fraction (solve_snow.c:333), which
                # is the one the previous step left.
                f['snow_cover_fract'] = cov_use
            return dict(_ls=_ls, _on=_on, _th=_th, alb_under=alb_under, albedo=albedo, bare_alb=bare_alb, cold_content=cold_content, cov_prev=cov_prev, cov_use=cov_use, f=f, last_snow=last_snow, ra_snow=ra_snow, rain_p=rain_p, rain_past_pack=rain_past_pack, snow_p=snow_p, sp_old_depth=sp_old_depth, sp_old_swq=sp_old_swq, st=st, t_grnd=t_grnd, wdew=wdew, wind_snow=wind_snow, z0_snow_t=z0_snow_t)
        _blk_res = _fusion.run(_snow_prep, wdew, last_snow)
        if _blk_res['_ls'] is not _UNSET:
            _ls = _blk_res['_ls']
        if _blk_res['_on'] is not _UNSET:
            _on = _blk_res['_on']
        if _blk_res['_th'] is not _UNSET:
            _th = _blk_res['_th']
        if _blk_res['alb_under'] is not _UNSET:
            alb_under = _blk_res['alb_under']
        if _blk_res['albedo'] is not _UNSET:
            albedo = _blk_res['albedo']
        if _blk_res['bare_alb'] is not _UNSET:
            bare_alb = _blk_res['bare_alb']
        if _blk_res['cold_content'] is not _UNSET:
            cold_content = _blk_res['cold_content']
        if _blk_res['cov_prev'] is not _UNSET:
            cov_prev = _blk_res['cov_prev']
        if _blk_res['cov_use'] is not _UNSET:
            cov_use = _blk_res['cov_use']
        if _blk_res['f'] is not _UNSET:
            f = _blk_res['f']
        last_snow = _blk_res['last_snow']
        if _blk_res['ra_snow'] is not _UNSET:
            ra_snow = _blk_res['ra_snow']
        if _blk_res['rain_p'] is not _UNSET:
            rain_p = _blk_res['rain_p']
        if _blk_res['rain_past_pack'] is not _UNSET:
            rain_past_pack = _blk_res['rain_past_pack']
        if _blk_res['snow_p'] is not _UNSET:
            snow_p = _blk_res['snow_p']
        if _blk_res['sp_old_depth'] is not _UNSET:
            sp_old_depth = _blk_res['sp_old_depth']
        if _blk_res['sp_old_swq'] is not _UNSET:
            sp_old_swq = _blk_res['sp_old_swq']
        if _blk_res['st'] is not _UNSET:
            st = _blk_res['st']
        if _blk_res['t_grnd'] is not _UNSET:
            t_grnd = _blk_res['t_grnd']
        wdew = _blk_res['wdew']
        if _blk_res['wind_snow'] is not _UNSET:
            wind_snow = _blk_res['wind_snow']
        if _blk_res['z0_snow_t'] is not _UNSET:
            z0_snow_t = _blk_res['z0_snow_t']
        # With the energy block on, a frozen pack whose surface layer is
        # under SNOW_MIN_SWQ_EB_THRES is left for the surface solve
        # (snow_melt.c:322-324, surface_fluxes.c:585-591); snow_step marks
        # it `thin` and skips its own solve for it.
        # No ice on any tile and no snow falling: every output of snow_step
        # is then discarded below -- the state is zeroed where ~has_snow,
        # water_in takes the rain, sublimation and melt are masked, energy
        # terms are multiplied by a cover of 0 -- so the step is skipped and
        # a zero-valued copy of its outputs stands in, bit-identical in
        # everything downstream. It was 52 of 82 ms per daily step on a
        # tropical basin that never sees snow. (It does compute a vapour
        # flux over a pack of zero mass, up to 4.4 mm/day, which the masks
        # below have always dropped.) The first snow-free step still runs
        # the real routine, to learn the shape of what it returns.
        snow_rc = None
        snow_mask = None
        if graphsafe.ACTIVE:
            # graph-safe: snow_step on every tile, its outputs zeroed off the
            # tiles the compacted step would take (as _put_tiles leaves
            # them); a snow-free step takes the eager branch (recorded).
            snow_mask = ((snow_state['ice'] + snow_p > 1e-11)
                         | (st['ice'] > 0.0) | (snow_p > 0.0))
            if graphsafe.MODES.get('snow_any', True) or not _snow_zero:
                if GRAPH_SNOW_ADAPT:
                    # a snow-free step here equals the snow-free branch
                    # (see GRAPH_SNOW_ADAPT); only noted, for the next day
                    graphsafe.seen(snow_mask.any())
                else:
                    graphsafe.need(~snow_mask.any(), graphsafe.SNOW_ANY)
                snow_free = False
            else:
                # snow-free everywhere: the cached zeros, as eagerly
                graphsafe.need(snow_mask, graphsafe.SNOW_NONE)
                snow_free = True
                snow_mask = None
        elif (SNOW_COMPACT and per_tile_snow and not spatial_snow
                and _implicit_mod.DEFAULT_METHOD == 'illinois'):
            # The tiles whose snow_step output is read: has_snow below, and
            # every tile with ice or snowfall at all (a superset). With none
            # of them the step is snow-free, which is the test below
            # without a second synchronisation.
            snow_rc = ((snow_state['ice'] + snow_p > 0.0)
                       | (st['ice'] > 0.0) | (snow_p > 0.0)).nonzero()
            snow_free = SNOW_FREE_FAST_PATH and snow_rc.shape[0] == 0
        else:
            snow_free = (SNOW_FREE_FAST_PATH and not spatial_snow
                         and not bool((st['ice'] > 0.0).any()
                                      | (snow_p > 0.0).any()))
        if not graphsafe.ACTIVE:
            graphsafe.LAST['snow_any'] = not snow_free
        # CLOSE_ENERGY's UNSTABLE_SNOW for this pass (None: stable, as
        # without the option, whose single pass never sets it).
        _unst = (False if _sub['unstable'] is None else _sub['unstable'])
        _ss = _snow_set.get('snow') if snow_mask is not None else None
        if _ss is not None:
            # SNOW_SET: on the window's snow lanes only; a lane outside
            # them that needs the step breaks the day (run again eagerly)
            graphsafe.need(snow_mask & ~_ss['in'], _SNOW_SET_BIT)
            _sh = st['ice'].shape
            _r, _c = _ss['r'], _ss['c']
            _st_s, _f_s, _thin_s, _un_s, _m_s = _set_take(
                (st, f, True if use_eb else wb_thin, _unst, snow_mask),
                _sh, _r, _c)
            graphsafe.MASK = _m_s           # the tiles whose results count
            try:
                new_state, melt_m = snow.snow_step(
                    _st_s, _f_s, thin_to_surface=_thin_s, unstable=_un_s,
                    snow_density=snow_density)
            finally:
                graphsafe.MASK = None
            new_state, melt_m = _set_put((new_state, melt_m), _ss['pos'],
                                         _ss['valid'], _m_s)
        elif snow_mask is not None:
            graphsafe.MASK = snow_mask      # the tiles whose results count
            try:
                new_state, melt_m = snow.snow_step(
                    st, f, thin_to_surface=True if use_eb else wb_thin,
                    unstable=_unst, snow_density=snow_density)
            finally:
                graphsafe.MASK = None
            new_state = _mask_tree(new_state, snow_mask)
            melt_m = _mask_tree(melt_m, snow_mask)
        elif snow_free and _snow_zero:
            new_state, melt_m = _snow_zero['state'], _snow_zero['melt']
        elif snow_rc is not None and snow_rc.shape[0] > 0:
            _sh = st['ice'].shape
            _r, _c = snow_rc[:, 0], snow_rc[:, 1]
            new_state, melt_m = snow.snow_step(
                _take_tiles(st, _sh, _r, _c), _take_tiles(f, _sh, _r, _c),
                thin_to_surface=_take_tiles(True if use_eb else wb_thin,
                                            _sh, _r, _c),
                unstable=_take_tiles(_unst, _sh, _r, _c),
                snow_density=snow_density)
            if snow.SIDE['pack'] is not None:
                _pf = torch.zeros(_sh, dtype=torch.bool, device=device)
                _pf[_r, _c] = snow.SIDE['pack']
                _tfx.note('snow pack', _pf)
            new_state = _put_tiles(new_state, _sh, _r, _c)
            melt_m = _put_tiles(melt_m, _sh, _r, _c)
        else:
            # snow_melt.c:322-324 hands a thin frozen pack to the soil
            # surface in water-balance mode too: calc_surf_energy_bal then
            # evaluates it at Tsurf = Tair (the WB branch below) where
            # `wb_thin` says the inputs for that are carried.
            new_state, melt_m = snow.snow_step(
                st, f, thin_to_surface=True if use_eb else wb_thin,
                unstable=_unst, snow_density=snow_density)
            _tfx.note('snow pack', snow.SIDE['pack'])
            if snow_free:
                _snow_zero['state'] = _zeros_like_tree(new_state)
                _snow_zero['melt'] = torch.zeros_like(melt_m)
        if (snow_mask is not None and GRAPH_SNOW_ADAPT and not _snow_zero
                and graphsafe.SEGMENTER is None):
            # GRAPH_SNOW_ADAPT: no eager redo may come to fill the
            # snow-free branch's zeros (which the snow-free graph needs),
            # so the eager step before a capture does: zeros shaped as
            # the masked outputs, what the graph-safe branch gives on a
            # snow-free step
            _snow_zero['state'] = _zeros_like_tree(new_state)
            _snow_zero['melt'] = _zeros_like_tree(melt_m)
        old_snow_surf_temp = st['surf_temp']

        # put_data.c weights every tile's contribution by its cover; with a
        # pack per tile that is now a real sum over tiles rather than one
        # pack's value repeated.
        def to_cell(x):
            return _tsum(x * mosaic.cv) if per_tile_snow else x

        def to_snow_cell(x, hp):
            # put_data.c 5.1.0:199-201, :433-437: the pack's albedo and
            # temperatures are averaged over the tiles that HAVE a pack
            # (cv_snow), not over the whole cell.
            if not per_tile_snow:
                return torch.where(hp, x, torch.zeros_like(x))
            cvs = _tsum(mosaic.cv * hp.to(x.dtype))
            tot = _tsum(torch.where(hp, x, torch.zeros_like(x)) * mosaic.cv)
            return torch.where(cvs > 0.0,
                               tot / torch.where(cvs > 0.0, cvs,
                                                 torch.ones_like(cvs)), tot)

        if blowing and per_tile_snow and _w_blow:
            # put_data.c:610: OUT_SUB_BLOWING, snow->blowing_flux (m, the
            # sign CalcBlowingSnow gives it: negative is mass leaving) times
            # Cv, scaled as snow_melt.c rescales it where the vapour flux
            # exhausts the layer
            sub_blowings.append(to_cell(
                f['blowing_mass_flux'] * (step_seconds / snow.CONST_RHOFW)
                * new_state['blow_scale'] * MM_PER_M))
        # (core.fusion.run: this block one fused region on the graph path)
        def _snow_post(snow_state, sp, melting):
            _adv_prev = _alb_in = _as = _dq = _k = _v = alb_end = cov_eff = cov_new = cover = delta_cov = gone = has_p = has_snow = melt_energy_t = melt_tile = se = sp_depth = sp_swq = sub_tile = tile_rows = water_in = _UNSET
            # Sublimation is mass leaving the pack as vapour. snow_step computes
            # it -- SnowPackEnergyBalance.c:217-235 writes the flux and
            # snow_melt.c:217-228 and :361-374 apply it -- and it used to be
            # dropped here, so the pack lost water that the water balance never
            # saw. VIC reports it as OUT_SUB_SNOW, 1.249 mm a year at this site,
            # and the residual that appeared when it started being computed was
            # 0.77 mm: the same quantity, uncounted.
            subl_m = new_state.get('sublimation', torch.zeros_like(melt_m))

            # snow->snow (solve_snow.c 5.1.0:146-150): any pack or any snowfall,
            # down to the 1.6e-16 mm a canopy can release (ISC Gangotri,
            # 2004-04-08), which then reports its melt energy and advection
            has_snow = (snow_state['ice'] + snow_p) > 0.0
            # ---- SPATIAL_SNOW, the part that runs after the pack solves -------
            # solve_snow.c:389-400 then :414-453. `cov_new` is snow->coverage for
            # the step just taken; `cov_mid` and `delta_cov` are the pair the C
            # hands calc_surf_energy_bal, whose sum is the fraction every energy
            # term is weighted by.
            if spatial_snow:
                sp_swq = (new_state['ice'] + new_state['surf_water']
                          + new_state['pack_water'])
                sp_depth = new_state.get('depth', sp_old_depth)
                # :396 passes melt + vapor_flux, both positive for mass leaving
                # the pack: snow_melt.c:499 flips the vapour flux's sign on the
                # way out, which is the sign `sublimation` already carries.
                sp_out = snow.snow_coverage(
                    store_snow=sp['store_snow'], max_slope=sp_slope,
                    old_coverage=cov_prev, swq=sp_swq, old_swq=sp_old_swq,
                    depth=sp_depth, old_depth=sp_old_depth,
                    melt=melt_m + subl_m, max_snow_depth=sp['max_snow_depth'],
                    snowfall=snow_p, store_swq=sp['store_swq'],
                    snow_distrib_slope=sp['snow_distrib_slope'],
                    store_coverage=sp['store_coverage'])
                gone = sp_swq <= 0.0
                # :340/:393-395 run the coverage update only when the pack
                # survived the step; a pack that melted out keeps coverage 0
                # and the C never calls calc_snow_coverage, so the stored
                # distribution is left alone (:454-469 reset only the slope
                # and the store flag). Calling it anyway parks a buried
                # fraction of 0 over bare ground at every melt-out.
                cov_new = torch.where(gone, torch.zeros_like(sp_swq),
                                      sp_out['coverage'])
                sp = dict(
                    adv_sens=sp.get('adv_sens', torch.zeros_like(cov_new)),
                    coverage=cov_new,
                    store_swq=torch.where(gone, sp['store_swq'],
                                          sp_out['store_swq']),
                    max_snow_depth=torch.where(gone, sp['max_snow_depth'],
                                               sp_out['max_snow_depth']),
                    store_coverage=torch.where(gone, sp['store_coverage'],
                                               sp_out['store_coverage']),
                    snow_distrib_slope=torch.where(
                        gone, torch.zeros_like(sp_swq),
                        sp_out['snow_distrib_slope']),
                    store_snow=torch.where(gone,
                                           torch.ones_like(sp_out['store_snow']),
                                           sp_out['store_snow']))
                # :414-453. Where melt shrank the cover the C keeps the *old*
                # fraction as the mid-step one and carries the difference in
                # delta_coverage; where the cover grew, or a pack appeared and
                # melted within one step, it does not.
                delta = cov_prev - cov_new
                shrank = delta > 0.0
                one_within = (cov_prev == 0.0) & (cov_new == 0.0)
                cov_mid = torch.where(shrank, cov_prev, cov_new)
                cov_mid = torch.where(one_within, torch.zeros_like(cov_mid),
                                      cov_mid)
                delta_cov = torch.where(shrank, delta, torch.zeros_like(delta))
                delta_cov = torch.where(one_within, torch.ones_like(delta_cov),
                                        delta_cov)
                # solve_snow.c:268 lets a tile into the pack block on the full
                # step-open swq -- ice plus both liquid stores -- or falling
                # snow; ice alone misses a pack that is all liquid. The bare
                # branches (:480, :513) keep delta_coverage 0, so the (0, 0)
                # branch at :431-439 is snow that fell and melted within the
                # step, not bare all along. Without the gate a tile pastes the
                # pack's shortwave onto its bare ground
                # (calc_surf_energy_bal.c:211-213) on every step after melt-out.
                # The same branches reset the stored distribution (:485-487,
                # :514-516): a stale store_swq would otherwise be mistaken for
                # snow buried over a partial pack at the next snowfall.
                sp_pack = (sp_old_swq + snow_p) > 0.0
                delta_cov = torch.where(sp_pack, delta_cov,
                                        torch.zeros_like(delta_cov))
                # calc_surf_energy_bal.c:633-634 and :931-941 of surface_fluxes.c
                # both weight by this sum, which is the old fraction while the
                # cover is shrinking and the new one otherwise. Bare tiles with
                # no snow at all (sp_pack false) get 0: the C reaches this
                # block only when snow is present or falling (solve_snow.c
                # :146-150), so old==0 and new==0 there means snow fell and
                # melted, not bare all along (which never enters solve_snow).
                cov_eff = torch.where(sp_pack, cov_new + delta_cov,
                                      torch.zeros_like(cov_new))
                # solve_snow.c:401-435: `coverage` is the old cover where it
                # shrank, the new one where it grew, 0 where the pack formed
                # and melted in the step, and untouched where it did not move
                # or no snow was present (:480-516).
                _grew = cov_new > cov_prev
                _cs = torch.where(shrank, cov_prev,
                                  torch.where(_grew, cov_new, cov_use))
                _cs = torch.where(one_within, torch.zeros_like(_cs), _cs)
                _sub['cov_star'] = torch.where(sp_pack, _cs, cov_use)
                sp['store_swq'] = torch.where(sp_pack, sp['store_swq'],
                                              torch.zeros_like(cov_new))
                sp['store_coverage'] = torch.where(
                    sp_pack, sp['store_coverage'], torch.ones_like(cov_new))
                alb_end = cov_eff * albedo + (1.0 - cov_eff) * bare_alb
            # The pack's energy terms as put_data.c reports them, with coverage
            # 1 under a pack (no SPATIAL_SNOW) and 0 without one, per
            # surface_fluxes.c:918-940: OUT_LATENT_SUB -= latent_sub (:916),
            # OUT_ADVECTION += advection (:942), OUT_DELTACC += deltaCC (:936),
            # OUT_RFRZ_ENERGY += refreeze_energy (:952), OUT_REFREEZE =
            # OUT_RFRZ_ENERGY / CONST_LATICE * dt in mm (:509-510).
            # OUT_MELT_ENERGY is solve_snow.c:130 zero except when snow falls and
            # melts within the step (:445-452), where it is the sum of the pack
            # terms; the SPATIAL_SNOW branch (:430) is not run here. OUT_SNOW_FLUX
            # is the soil balance's residual under the pack
            # (calc_surf_energy_bal.c:721), not the pack's conduction, and is not
            # formed until the soil node under the pack replaces t_grnd = 0.
            se = new_state['energy']
            zero_w = torch.zeros_like(melt_m)
            cover = torch.where(has_snow, torch.ones_like(melt_m), zero_w)
            if spatial_snow:
                # surface_fluxes.c:931-941 weights the pack's energy terms by
                # (coverage + delta_coverage) rather than by a 0/1 flag.
                cover = cov_eff
            had_pack = snow_state['ice'] > 0.0
            one_step = (~had_pack) & (snow_p > 0.0) & (new_state['ice'] <= 0.0)

            # On the graph path the block's cell sums are deferred to one fused
            # region (_cell_sums) just before `alb`, the first value read back;
            # each list still gets its one append of this block, in order.
            _dq = ([] if (per_tile_snow and lanes is not None
                          and _fusion.wants((mosaic.cv,))) else None)

            def _put(key, x, scale=None):
                if _dq is None or x.dim() != 2:
                    v_ = to_cell(x)
                    _out_named(key).append(v_ if scale is None else v_ * scale)
                else:
                    _dq.append((key, x, None, scale))

            def _put_snow(key, x, hp):
                if _dq is None or x.dim() != 2:
                    _out_named(key).append(to_snow_cell(x, hp))
                else:
                    _dq.append((key, x, hp, None))

            # Kept per tile as well as cover-weighted, because the thin-pack
            # branch of the surface solve replaces some of them later in the
            # step and has to do that before the weighting, not after.
            tile_rows = dict(latent_sub=-se['latent_sub'] * cover,
                             advection=se['advection'] * cover,
                             deltacc=se['delta_cc'] * cover,
                             rfrz_energy=se['refreeze_energy'] * cover)
            _adv_prev = None
            if spatial_snow and 'advected_sensible' in se:
                # surface_fluxes.c:923-924 and :932-933, weighted the same way;
                # put_data.c 5.1.0:938-940 reports it as OUT_ADV_SENS with the
                # sign flipped and only where the tile has no overstory.
                sp['adv_sens'], _adv_prev = (se['advected_sensible'] * cover,
                                             sp.get('adv_sens'))
                if _adv_prev is None:
                    _adv_prev = torch.zeros_like(sp['adv_sens'])
                _as = -sp['adv_sens']
                if over_t is not None:
                    _as = torch.where(over_t, torch.zeros_like(_as), _as)
                tile_rows['advected_sensible'] = _as
            for _k, _v in tile_rows.items():
                _put('eb.' + _k, _v)
            if not spatial_snow:
                # solve_snow.c 5.1.0:401-438 without SPATIAL_SNOW: coverage is 1
                # with a pack and 0 without, so delta_coverage is 1 whenever the
                # step ends with no pack -- a pack that melted out (:405-418) as
                # well as snow that fell and melted within the step (:428-438)
                # -- and melt_energy is then the pack's whole term sum, the
                # advected sensible heat included. A thin pack handed to the
                # surface reports 0 (surface_fluxes.c 5.1.0:597).
                _thin_se = new_state.get('thin')
                one_step = has_snow & ((new_state['ice'] + new_state['surf_water']
                                        + new_state['pack_water']) <= 0.0)
                if _thin_se is not None:
                    one_step = one_step & ~_thin_se
                melt_energy_t = torch.where(
                    one_step,
                    se['advection'] - se['delta_cc'] + se['latent']
                    + se['latent_sub'] + se['sensible'] + se['refreeze_energy']
                    + se.get('advected_sensible', zero_w),
                    zero_w)
            else:
                # solve_snow.c 5.1.0:399-437 with SPATIAL_SNOW: melt_energy is
                # delta_coverage times the pack's term sum where the cover shrank
                # (:405-425), the sum where snow fell and melted within the step
                # (delta 1, :426-437), 0 otherwise; thin packs report 0
                # (surface_fluxes.c 5.1.0:597). Bare tiles with no snow at all
                # (has_snow false) report 0: the C reaches this block only when
                # snow is present or falling (:146-150), so old==0 and new==0
                # there means snow fell and melted, not bare all along.
                _sum = (se['advection'] - se['delta_cc'] + se['latent']
                        + se['latent_sub'] + se['sensible'] + se['refreeze_energy']
                        + se.get('advected_sensible', zero_w))
                _thin_se = new_state.get('thin')
                if _thin_se is not None:
                    _sum = torch.where(_thin_se, zero_w, _sum)
                melt_energy_t = torch.where(has_snow, delta_cov * _sum, zero_w)
            _put('eb.melt_energy', melt_energy_t)
            _put('eb.refreeze', se['refreeze_energy'] * cover
                 / snow.CONST_LATICE * step_seconds)
            snow_state = {k: torch.where(has_snow, new_state[k],
                                         torch.zeros_like(new_state[k]))
                          for k in ('ice', 'surf_water', 'pack_water',
                                    'surf_temp', 'pack_temp', 'depth', 'density',
                                    'cold_content', 'swq_c')
                          if k in new_state}
            # What reaches each tile's canopy and soil: its own pack's outflow
            # where it has a pack, its own rain where it has none.
            water_in = torch.where(has_snow, melt_m * MM_PER_M,
                                   rain_p * MM_PER_M)
            if rain_past_pack is not None:
                # solve_snow.c:304 -- rain that fell on the bare part of a partly
                # covered tile is already on its way to the soil, pack or no
                # pack. Dropping it is a hole in the water balance the size of
                # the bare fraction's rainfall.
                water_in = water_in + rain_past_pack * MM_PER_M
            # Kept per tile as well, because the thin-pack branch below replaces
            # it for some tiles and must not lose the rest.
            sub_tile = torch.where(has_snow, subl_m * MM_PER_M,
                                   torch.zeros_like(melt_m))
            _put('sublimations', sub_tile)
            # OUT_SNOW_MELT is snow->melt, which surface_fluxes.c:961 sets to the
            # outflow solve_snow.c:351 also adds to ppt -- rain that passed
            # through the pack included, since snow_melt.c:168 put it in the
            # surface store first. That is `melt_m` here, in mm. Without a pack
            # solve_snow's `melt` stays at its line-125 zero.
            # A thin pack folded into the surface balance reports that
            # balance's melt instead (below).
            melt_tile = torch.where(has_snow, melt_m * MM_PER_M,
                                    torch.zeros_like(melt_m))
            _put('melts', melt_tile)

            # solve_snow.c:375-385, VIC's MELTING flag. It is seasonal and
            # sticky, not a statement about this step: it turns on once the
            # pack has no cold content left, during the melt half of the year,
            # and turns off again only when snow deeper than SNOW_TRACESNOW
            # falls on it. It chooses which albedo decay curve the pack is on,
            # so reading it as `this step produced outflow` -- which is what
            # this used to do -- left the pack on the accumulation curve
            # through the whole melt season.
            #
            # snow->coldcontent is SurfaceCC (snow_melt.c 5.1.0:478), which is
            # the surface layer's cold content at the START of the step unless
            # the pack ends it above SNOW_MAX_SURFACE_SWE (:433-456) -- not
            # `surf_temp >= 0`, which flagged a thin pack as melting on the
            # step it first reached 0 C. snow_step returns it.
            doy = _doy
            _lat = getattr(batch, 'latitude', None)
            if _lat is None:
                north = torch.ones_like(melting)
            else:
                north = _lat >= 0.0
                while north.dim() < melting.dim():
                    north = north.unsqueeze(-1)
                north = north.expand_as(melting)
            # ~1 March to ~1 October north of the equator, the complement
            # south of it.
            if graphsafe.ACTIVE and _season_buf is not None:
                # in a graph the date is the step's buffer, not a Python value
                season = torch.where(_season_buf, north, ~north)
            else:
                season = north if (60.0 < doy < 273.0) else ~north
            isothermal = (new_state['cold_content'] >= 0.0
                          if 'cold_content' in new_state
                          else new_state['surf_temp'] >= 0.0)
            melting = torch.where(
                season & isothermal, torch.ones_like(melting),
                torch.where(melting & (snow_p > snow.TRACE_SNOW_M),
                            torch.zeros_like(melting), melting))
            # solve_snow.c 5.1.0:465, :487 and :517: a tile that ends the step
            # with no pack -- melted out, or never had one -- has MELTING false.
            melting = melting & ((snow_state['ice'] + snow_state['surf_water']
                                  + snow_state['pack_water']) > 0.0)
            # put_data.c:724: snow.depth in cm, reported at end of step.
            if 'depth' in snow_state:
                _put('snow_depths', snow_state['depth'], CM_PER_M)
            _put('swes', snow_state['ice'] + snow_state['surf_water']
                 + snow_state['pack_water'], MM_PER_M)
            # The pack's own diagnostics, all conditioned on the pack that the
            # step *leaves behind*. With SPATIAL_SNOW off, snow->coverage is 1
            # where swq > 0 and 0 otherwise (calc_surf_energy_bal.c:712-717), and
            # put_data.c:727-731 accumulates albedo and the two temperatures only
            # from tiles with swq > 0, dividing by that cover at :450-452 -- so
            # a cell without a pack reports zero for all three, not a land albedo
            # and not an air temperature. `albedo` is the value this step's
            # energy balance used, which is what snow->albedo holds at :728.
            # With a pack per tile these are per tile too, and put_data.c weights
            # each by its cover: OUT_SNOW_COVER becomes the covered *fraction* of
            # the cell rather than a flag, which is what VIC reports.
            has_p = snow_state['ice'] + snow_state['surf_water'] \
                + snow_state['pack_water'] > 0.0
            _put('snow_covers', cov_new if spatial_snow else has_p.to(dtype))
            _put_snow('salbedos', albedo, has_p)
            _put_snow('snow_surf_temps', snow_state['surf_temp'], has_p)
            _put_snow('snow_pack_temps', snow_state['pack_temp'], has_p)
            if _lake_tap is not None and per_tile_snow:
                _lake_tap['snow_cover'] = (cov_new if spatial_snow
                                           else has_p.to(dtype))
                _lake_tap['snow_ratio'] = dict(
                    salbedo=albedo, snow_surf_temp=snow_state['surf_temp'],
                    snow_pack_temp=snow_state['pack_temp'])

            # Net radiation at the surface. Albedo is the snow value where there
            # is snow and a fixed land value where there is not; this model has no
            # vegetation library to take a per-tile albedo from.
            # This cell-level net radiation drives the no-mosaic path and the
            # non-energy vegetation path, so the albedo is the cover-weighted one
            # when the packs are per tile.
            _alb_in = (alb_end if spatial_snow
                       else torch.where(has_snow, albedo,
                                        torch.full_like(albedo, LAND_ALBEDO)))
            return dict(_adv_prev=_adv_prev, _alb_in=_alb_in, _as=_as, _dq=_dq, _k=_k, _v=_v, alb_end=alb_end, cov_eff=cov_eff, cov_new=cov_new, cover=cover, delta_cov=delta_cov, gone=gone, has_p=has_p, has_snow=has_snow, melt_energy_t=melt_energy_t, melt_tile=melt_tile, melting=melting, se=se, snow_state=snow_state, sp=sp, sp_depth=sp_depth, sp_swq=sp_swq, sub_tile=sub_tile, tile_rows=tile_rows, water_in=water_in)
        _blk_res = _fusion.run(_snow_post, snow_state, sp, melting)
        if _blk_res['_adv_prev'] is not _UNSET:
            _adv_prev = _blk_res['_adv_prev']
        if _blk_res['_alb_in'] is not _UNSET:
            _alb_in = _blk_res['_alb_in']
        if _blk_res['_as'] is not _UNSET:
            _as = _blk_res['_as']
        if _blk_res['_dq'] is not _UNSET:
            _dq = _blk_res['_dq']
        if _blk_res['_k'] is not _UNSET:
            _k = _blk_res['_k']
        if _blk_res['_v'] is not _UNSET:
            _v = _blk_res['_v']
        if _blk_res['alb_end'] is not _UNSET:
            alb_end = _blk_res['alb_end']
        if _blk_res['cov_eff'] is not _UNSET:
            cov_eff = _blk_res['cov_eff']
        if _blk_res['cov_new'] is not _UNSET:
            cov_new = _blk_res['cov_new']
        if _blk_res['cover'] is not _UNSET:
            cover = _blk_res['cover']
        if _blk_res['delta_cov'] is not _UNSET:
            delta_cov = _blk_res['delta_cov']
        if _blk_res['gone'] is not _UNSET:
            gone = _blk_res['gone']
        if _blk_res['has_p'] is not _UNSET:
            has_p = _blk_res['has_p']
        if _blk_res['has_snow'] is not _UNSET:
            has_snow = _blk_res['has_snow']
        if _blk_res['melt_energy_t'] is not _UNSET:
            melt_energy_t = _blk_res['melt_energy_t']
        if _blk_res['melt_tile'] is not _UNSET:
            melt_tile = _blk_res['melt_tile']
        melting = _blk_res['melting']
        if _blk_res['se'] is not _UNSET:
            se = _blk_res['se']
        snow_state = _blk_res['snow_state']
        sp = _blk_res['sp']
        if _blk_res['sp_depth'] is not _UNSET:
            sp_depth = _blk_res['sp_depth']
        if _blk_res['sp_swq'] is not _UNSET:
            sp_swq = _blk_res['sp_swq']
        if _blk_res['sub_tile'] is not _UNSET:
            sub_tile = _blk_res['sub_tile']
        if _blk_res['tile_rows'] is not _UNSET:
            tile_rows = _blk_res['tile_rows']
        if _blk_res['water_in'] is not _UNSET:
            water_in = _blk_res['water_in']
        if _dq is None:
            alb = to_cell(_alb_in)
        else:
            _plain = {'alb': _alb_in}
            _snowc = {}
            for i_, (l_, x_, hp_, sc_) in enumerate(_dq):
                if hp_ is None:
                    _plain['q%d' % i_] = x_
                else:
                    _snowc['q%d' % i_] = (x_, hp_)
            _res = _cell_sums(_plain, _snowc, mosaic.cv, lanes.cell_start,
                              lanes.cell_mask, lanes.n_tiles,
                              lanes.lane_cell)
            for i_, (l_, x_, hp_, sc_) in enumerate(_dq):
                v_ = _res['q%d' % i_]
                _out_named(l_).append(v_ if sc_ is None else v_ * sc_)
            alb = _res['alb']
        net_rad = (atm['shortwave'][_t1] * (1.0 - alb) + atm['longwave'][_t1]
                   - SURFACE_EMISSIVITY * snow.CONST_STEBOL
                   * (atm['tair'][_t1] + CONST_TKFRZ) ** 4)
        # Aerodynamic resistance for the vegetated surface. The wind is the
        # forcing's own where there is one; it used to be the constant
        # DEFAULT_WIND for every cell and every day even though the forcing
        # carried a measured field, which fixed ra at 69.95 s/m all year
        # against VIC's 17-43 s/m at this site's mean wind.
        #
        # Still approximate in a way worth naming: VIC's CalcAerodynamic.c
        # builds this from each tile's own veg_rough, displacement and wind_h,
        # and this uses one snow roughness for every surface. That is the next
        # thing to fix here, not something the file fails to supply.
        u = atm['wind'][_t1] if 'wind' in atm else _like(prcp, DEFAULT_WIND)
        u = torch.clamp(u, min=MIN_WIND)
        ra = (math.log(z_wind / SNOW_Z0) ** 2) / (VON_KARMAN_SQ * u)
        if mosaic is not None:
            # Set only on the water-balance path with a fractional canopy:
            # the (1 - fcanopy) bare-soil evaporation, per cell, and the
            # per-tile layer evaporation the columns then take.
            bare_cell = col_evap_t = None
            if mosaic.albedo is not None:
                # AlbedoUnder mixes the snow-free tile albedo with the snow
                # albedo by coverage, 0 or 1 with SPATIAL_SNOW off
                # (calc_surf_energy_bal.c:633-635). Reporting the class
                # albedo alone was right at Walnut Gulch by accident of
                # climate -- four days of trace snow -- and wrong at
                # FindleyLake by construction: 0.12 against VIC's 0.42 on a
                # cell under snow half the year.
                # Per tile: each tile reports its own snow albedo where it
                # has a pack and its class albedo where it does not, and the
                # cell's value is the cover-weighted sum -- which is what
                # calc_surf_energy_bal.c:633-635 forms as AlbedoUnder.
                # (core.fusion.run: this block one fused region on the graph path)
                def _alb_blk():
                    _alb_cell = _z = per_t = under_t = _UNSET
                    if spatial_snow:
                        under_t = alb_end
                    elif per_tile_snow:
                        under_t = torch.where(
                            has_snow, albedo, mosaic.albedo[:, :, months[_t1]])
                    else:
                        under_t = torch.where(
                            has_snow.unsqueeze(1) if has_snow.dim() == 1
                            else has_snow,
                            albedo.unsqueeze(1) if albedo.dim() == 1
                            else albedo,
                            mosaic.albedo[:, :, months[_t1]])
                    # put_data.c:903-909: an overstory tile reports the
                    # canopy's albedo, not the ground's. Reporting the
                    # snow's for a forested cell gave OUT_ALBEDO 0.74
                    # against VIC's 0.49 over a FindleyLake winter, and
                    # the same error sits in OUT_SWNET, which is the
                    # canopy's absorption plus the ground's.
                    if alb_over_t is not None:
                        # put_data.c 5.1.0:885-890: AlbedoOver only where
                        # snow.snow (solve_snow.c 5.1.0:164-168, the tiles the
                        # canopy ran on); AlbedoUnder elsewhere, overstory or not.
                        _ran = (canopy_ran if canopy_ran is not None
                                else torch.zeros_like(over_t))
                        per_t = torch.where(over_t & _ran, alb_over_t, under_t)
                        if _sub['pin_tgrnd']:
                            _on = over_t & _ran
                            _sky = tile_c(atm['longwave'][_t1])
                            _lu = (_sky if canopy_long_out is None
                                   else torch.where(_on, canopy_long_out, _sky))
                            _z = torch.zeros_like(_sky)
                            _sub['alb_rows'].append(dict(
                                on=_on, alb_over=torch.where(_on, alb_over_t, _z),
                                alb_under=under_t,
                                lw_over=torch.where(_on, _sky, _z),
                                lw_under=_lu))
                    else:
                        per_t = under_t
                    _alb_cell = _tsum(per_t * mosaic.cv)
                    return dict(_alb_cell=_alb_cell, _z=_z, per_t=per_t, under_t=under_t)
                _blk_res = _fusion.run(_alb_blk)
                if _blk_res['_alb_cell'] is not _UNSET:
                    _alb_cell = _blk_res['_alb_cell']
                if _blk_res['_z'] is not _UNSET:
                    _z = _blk_res['_z']
                if _blk_res['per_t'] is not _UNSET:
                    per_t = _blk_res['per_t']
                if _blk_res['under_t'] is not _UNSET:
                    under_t = _blk_res['under_t']
                albedos.append(_alb_cell)
                if return_state:
                    # calc_gridcell_avg_albedo.c (STATE_AVG_ALBEDO) reads
                    # the ground's albedo per tile
                    _vic_aux['alb_under'] = (
                        under_t if under_t.dim() == mosaic.cv.dim()
                        else under_t.unsqueeze(1).expand_as(mosaic.cv))
                in_longs.append(atm['longwave'][_t1])
                if _lake_tap is not None:
                    _lake_tap['alb'] = per_t
            # Vegetation tiles. A cell's demand is the area-weighted sum over
            # its tiles, each with its own leaf area for this month, its own
            # stomatal resistance and its own root profile. This is what `rs`
            # and `et_beta` were standing in for, and a sensitivity study
            # measured those two stand-ins as the leading controls on evapotranspiration
            # precisely because they were carrying all of it.
            # Thresholds from the prior when it supplies them, so the
            # stress function uses the same soil the column does.
            # the same every step: formed once (on a graph's warm-up step)
            # and kept, unless gradients flow through the parameters
            _wcw = (_wcw_cache.get('v') if not torch.is_grad_enabled()
                    else None)
            if _wcw is None:
                w_crit = getattr(soil_params, 'w_crit_mm', None)
                w_wilt = getattr(soil_params, 'w_wilt_mm', None)
                if w_crit is None:
                    w_crit = soil_params.max_moist * vegetation.cellwise(
                        p['w_crit'], soil_params.max_moist)
                    w_wilt = soil_params.max_moist * vegetation.cellwise(
                        p['w_wilt'], soil_params.max_moist)
                # Multiples of the file's critical and wilting points (VIC's
                # Wcr_FRACT, Wpwp_FRACT). On one Indian basin the
                # file's Wcr of 0.74 of saturation stressed the crops as soon as
                # the monsoon soil drained: ET fell to 1.4 mm/day in Sep-Oct
                # 2002 where GLEAM kept 2.4. The wilting point is held below the
                # critical point, which VIC requires (Wpwp <= Wcr).
                if 'wcr_scale' in p:
                    w_crit = w_crit * vegetation.cellwise(p['wcr_scale'], w_crit)
                if 'wpwp_scale' in p:
                    w_wilt = w_wilt * vegetation.cellwise(p['wpwp_scale'], w_wilt)
                if 'wcr_scale' in p or 'wpwp_scale' in p:
                    w_wilt = torch.minimum(w_wilt, 0.99 * w_crit)
                if not torch.is_grad_enabled():
                    _wcw_cache['v'] = (w_crit, w_wilt)
            else:
                w_crit, w_wilt = _wcw
            # The canopy sees the water first. Only what falls through it
            # reaches the ground, and only the dry part of the leaf
            # transpires. Snow interception is not implemented, so snowfall
            # passes the canopy untouched; a configuration needing
            # snow_intercept.c must not use this path.
            if use_eb:
                # ---- the surface energy balance, one root find per tile ----
                # (core.fusion.run: this block one fused region on the graph path)
                def _eb_pre(th, cs):
                    m = alb_t = tile = cov_end_t = dcov_t = bare_frac = sw_in = ns_t = z0_t = d_t = ra_t = zref_t = u_surf = ra_canopy = z0_solve = d_solve = ov = u_surf0 = us2 = lai_t = fc_t = lai_d = atten_t = _sn_no_over = atten_one_eb = ntl = two = kap = kap1 = kap2 = cs1 = cs2 = le = cache = tair_now = vpd_now = bare_eb = bare_in = z0_s = d_s = top_s = pos_u = safe_u = ra_b = u_b = d0_b = snowing_eb = _UNSET
                    th = soil_params.thermal
                    m = months[_t1]
                    # Per-tile radiation and resistance at this step, from the
                    # tile's own albedo, roughness and displacement.
                    alb_t = mosaic.albedo[:, :, m]
                    # A cell quantity gets the tile axis; one that already
                    # has it -- with a pack per tile, most of the snow
                    # state -- passes through.
                    tile = (lambda x: x.expand_as(alb_t) if x.dim() >= 2
                            else x.unsqueeze(1).expand_as(alb_t))
                    # calc_surf_energy_bal.c:231-232: the snow-free area of the
                    # cell is what the bare surface sees, so a cell under a pack
                    # (snow_coverage 1 without SPATIAL_SNOW) gives it neither
                    # shortwave nor sky longwave -- the radiation is the pack's,
                    # and the pack reflects it with the snow albedo. Giving the
                    # soil surface the whole sun under 400 mm of snow put
                    # OUT_SWNET 158% above VIC's on 90 winter days of
                    # FindleyLake, because it was absorbing at the vegetation's
                    # albedo what VIC reflects at the snow's.
                    # calc_surf_energy_bal.c:234, LongBareIn: the longwave and
                    # the turbulent fluxes see the fraction left bare at the end
                    # of the step.
                    # Without SPATIAL_SNOW that is snow->coverage as solve_snow
                    # leaves it (solve_snow.c 5.1.0:381-397): 1 where a pack is
                    # left, 0 where none is -- including where the step's snow
                    # fell and melted, or a pack melted out, which then has
                    # delta_coverage 1 (:401-438).
                    cov_end_t = has_p.to(dtype)
                    dcov_t = (has_snow & ~has_p).to(dtype)
                    bare_frac = tile(1.0 - (cov_new if spatial_snow
                                            else cov_end_t))
                    sw_in = (atm['shortwave'][_t1] if under_short is None
                             else under_short)
                    if spatial_snow:
                        # :231-233, NetShortBare: the ground that was bare all
                        # step absorbs at its own albedo, and the strip the cover
                        # uncovered during the step absorbs at the snow's. With
                        # the option off delta_coverage is zero and the second
                        # term vanishes, which is the expression above.
                        ns_t = (tile(sw_in) * (1.0 - alb_t) * tile(1.0 - cov_eff)
                                + tile(sw_in) * tile(delta_cov)
                                * (1.0 - tile(albedo)))
                    else:
                        # calc_surf_energy_bal.c 5.1.0:229-231 with
                        # delta_coverage 1 on a tile whose pack is gone: the
                        # strip absorbs at the snow's albedo
                        ns_t = (tile(sw_in) * (1.0 - alb_t)
                                * (bare_frac - tile(dcov_t))
                                + tile(sw_in) * tile(dcov_t)
                                * (1.0 - tile(albedo)))
                    z0_t = mosaic.veg_rough[:, :, m]
                    # vic_run.c:245: a tile with zero roughness takes the soil's.
                    z0_t = torch.where(z0_t > 0.0, z0_t,
                                       th['rough'].unsqueeze(1).expand_as(z0_t))
                    d_t = mosaic.displacement[:, :, m]
                    ra_t = vegetation.aerodynamic_resistance(z0_t, d_t,
                                                             u.unsqueeze(1))
                    zref_t = energy.reference_height(mosaic.wind_h, d_t, z0_t)
                    # CalcAerodynamic.c:32-33 (no overstory): the wind the
                    # surface balance sees is the forcing's brought down to 2 m
                    # over the displacement, U[0] = wind * ln((2 + z0)/z0) /
                    # ln((zref - d)/z0), and vic_run.c hands that array
                    # (tmp_wind, :271 then :334) to surface_fluxes, whose
                    # stability correction takes wind[UnderStory]
                    # (func_surf_energy_bal.c:674-680). The forcing's wind went
                    # in here before: 6-22% too much wind in the Richardson
                    # number at this site's roughnesses, weakest correction on
                    # the stable nights that set OUT_AERO_RESIST.
                    # (:55-56 forms the log ratio, :178-179 scales it by the
                    # wind afterwards; that order, to the ulp)
                    u_surf = (torch.log((2.0 + z0_t) / z0_t)
                              / torch.log((zref_t - d_t) / z0_t) * u.unsqueeze(1))
                    # A covered overstory tile takes CalcAerodynamic.c's
                    # overstory branch. The surface solve under the canopy uses
                    # Ra[0], U[0] and the rewritten understory references (2 m
                    # over the soil roughness, no displacement, :105-112);
                    # canopy evaporation uses Ra[1] as canopy_evap is handed
                    # Ra_veg[1]. The canopy itself is at the air temperature
                    # (Tfoliage = Tair) except where the canopy stage
                    # (snow_intercept.c via canopy.canopy_step) runs and gives
                    # it a balance.
                    # `over_t` and `any_over` were decided once before the loop:
                    # the mask is a property of the mosaic, not of the step, and
                    # `bool(x.any())` inside the loop copies from the device and
                    # stalls the queue every step.
                    ra_canopy = ra_t
                    z0_solve, d_solve = z0_t, d_t
                    if any_over:
                        ov = ov_step                # computed once, at step start
                        ra_canopy = torch.where(over_t, ov['ra'][..., 1], ra_t)
                        ra_t = torch.where(over_t, ov['ra'][..., 0], ra_t)
                        u_surf = torch.where(over_t, ov['u'][..., 0], u_surf)
                        zref_t = torch.where(over_t, ov['ref_height'][..., 0], zref_t)
                        z0_solve = torch.where(over_t, ov['roughness'][..., 0], z0_t)
                        d_solve = torch.where(over_t, ov['displacement'][..., 0], d_t)
                    # The bare soil between plants takes the tile's own wind[0]
                    # whatever the snow (func_surf_energy_bal.c 5.1.0:686-697).
                    u_surf0 = u_surf
                    if per_tile_snow and ra_snow is not None:
                        # solve_snow.c 5.1.0:133-139 and :287: UnderStory = 2
                        # where the tile opens with a pack, snow falls from the
                        # sky, or snow reaches the ground through a canopy, and
                        # func_surf_energy_bal.c 5.1.0:641-656 then takes
                        # ra[2], wind[2], ref_height[2] = 2 + Z0_SNOW,
                        # roughness[2] = Z0_SNOW and no displacement
                        # (CalcAerodynamic.c 5.1.0:76-81, :167-170) for the
                        # surface -- and so for a thin pack's vapour flux and
                        # sensible heat. Index 0 was used under snow too.
                        us2 = ((wb_swq0 > 0.0) | tile(sky_snow_m > 0.0)
                               | (snow_p > 0.0))
                        if _sub['understory'] is not None:
                            # CLOSE_ENERGY: UnderStory is set once per
                            # over-iteration (:133-139) and then only ever set
                            # to 2 where the pack branch runs (:276), so a pass
                            # whose throughfall dried up still solves at the
                            # snow level its over-iteration reached.
                            us2 = us2 | (_sub['understory'] == 2)
                        ra_t = torch.where(us2, ra_snow, ra_t)
                        u_surf = torch.where(us2, wind_snow, u_surf)
                        zref_t = torch.where(us2, 2.0 + z0_snow_t, zref_t)
                        z0_solve = torch.where(us2, z0_snow_t, z0_solve)
                        d_solve = torch.where(us2, torch.zeros_like(d_solve),
                                              d_solve)
                    lai_t = mosaic.lai[:, :, m] * vegetation.cellwise(p['lai_scale'], mosaic.cv)
                    # vic_run.c:136 divides LAI by fcanopy for the whole step,
                    # so the soil path sees the same per-plant leaf area the
                    # canopy did, and :222-225 builds surf_atten from it.
                    fc_t = (torch.ones_like(lai_t) if mosaic.fcanopy is None
                            else mosaic.fcanopy[:, :, m])
                    # VIC 5.1.0: only where fcanopy > 0 (vic_run.c 5.1.0:219-226)
                    lai_d = lai_t / (torch.where(fc_t > 0.0, fc_t,
                                                 torch.ones_like(fc_t))
                                     if core_version.at_least('5.1.0') else fc_t)
                    if (mosaic.bare_tile is not None and mosaic.fcanopy is not None
                            and not core_version.at_least('5.1.0')):
                        # vic_run.c 5.0.1:132-141 divides for iveg < Nveg only;
                        # the bare-soil tile's surf_atten (:222-225) takes its
                        # own LAI (as on the water-balance path).
                        lai_d = torch.where(mosaic.bare_tile, lai_t, lai_d)
                    lai_t = lai_d
                    atten_t = energy.surface_attenuation(lai_t,
                                                         mosaic.rad_atten, fc_t)
                    # solve_snow.c 5.1.0:146-153: with snow present or falling a
                    # tile without an overstory sets surf_atten = 1, which the
                    # ground flux then sees (func_surf_energy_bal.c 5.1.0
                    # :394-401) even where the pack melted out within the step
                    _sn_no_over = tile(has_snow)
                    if any_over:
                        _sn_no_over = _sn_no_over & ~over_t
                    atten_one_eb = None
                    if _sub['pin_tgrnd']:
                        # surf_atten is surface_fluxes' own argument (:19,
                        # :558): it stays 1 for the later snow steps.
                        if _sub['atten_one'] is not None:
                            _sn_no_over = _sn_no_over | _sub['atten_one']
                        _sub['atten_one'] = _sn_no_over
                        atten_one_eb = _sn_no_over
                    atten_t = torch.where(_sn_no_over, torch.ones_like(atten_t),
                                          atten_t)
                    # Thermal properties of the top two layers from the cell's
                    # moisture. VIC has a soil column per tile; this model has one
                    # per cell, so the two layers' kappa and Cs are shared across
                    # tiles. That is a named structural difference, not a slip.
                    # per tile now: each tile's own column sets its two layers'
                    # conductivity and heat capacity, as VIC's does.
                    ntl = mosaic.n_tiles
                    two = lambda x: x.unsqueeze(1).expand(nc, ntl, 2).reshape(-1, 2)
                    kap, cs = energy.thermal_properties(
                        moist[:, :, :2].reshape(-1, 2), two(th['depth'][:, :2]),
                        two(th['bulk_density'][:, :2]),
                        two(th['soil_density'][:, :2]), two(th['quartz'][:, :2]),
                        **_mineral_kw(th, two),
                        # soil_conduction.c:711-716 builds the two upper
                        # layers' conductivity and heat capacity from the ice
                        # the previous step left, which is what
                        # calc_layer_average_thermal_props passes on
                        # (frost-weighted over the sub-areas with
                        # SPATIAL_FROST).
                        ice_mm=(None if soil_ice is None
                                else _ice_eff()[:, :, :2].reshape(-1, 2)))
                    kap1 = kap[:, 0].view(nc, ntl)
                    kap2 = kap[:, 1].view(nc, ntl)
                    cs1 = cs[:, 0].view(nc, ntl)
                    cs2 = cs[:, 1].view(nc, ntl)
                    # A cell quantity gets the tile axis; one that already
                    # has it -- with a pack per tile, most of the snow
                    # state -- passes through.
                    tile = (lambda x: x.expand_as(alb_t) if x.dim() >= 2
                            else x.unsqueeze(1).expand_as(alb_t))
                    le = atmos.latent_heat_of_vaporization(atm['tair'][_t1])

                    # The evaporation the residual calls, per tile, at the net
                    # radiation it is solving for. It caches its last full result
                    # so the state can be advanced from the solved temperature
                    # without a recomputation the residual did not see.
                    cache = {}
                    # Sliced here, not inside the closure: a closure that reads
                    # the step index `t` is, to torch.compile, a different
                    # function every step (it guards on t's value) and was
                    # recompiled every hour until the cache gave up.
                    tair_now, vpd_now = atm['tair'][_t1], atm['vpd'][_t1]
                    if _sub['tcanopy'] is not None:
                        # CLOSE_ENERGY: the ground balance runs at the canopy
                        # air temperature (calc_surf_energy_bal.c:28, "T of
                        # canopy or air"), so canopy_evap and arno_evap do too
                        # (func_surf_energy_bal.c:735-750).
                        tair_now = _sub['tcanopy']

                    # VIC 5.1.0 with a canopy fraction (a leafless month's tile
                    # has fcanopy 0): func_surf_energy_bal.c 5.1.0:677-761 on
                    # this path too -- Ra_used[0] the conductance average with
                    # the bare soil's stability-corrected resistance (energy.
                    # _residual's `bare`), and (1 - fcanopy) of the tile ARNO
                    # bare soil at that resistance (evap_per_tile's
                    # `bare_soil`). The bare soil's resistance is
                    # CalcAerodynamic's no-overstory form over the soil
                    # roughness with the tile's wind[0] and SOIL_WINDH as the
                    # reference height (:684-693), and its stability correction
                    # takes that height and the wind the call leaves in
                    # tmp_wind[0], U[0] = wind[0] ln((2 + z0)/z0) /
                    # ln((SOIL_WINDH - d)/z0). (At 2 m instead, the bare soil's
                    # resistance at a hot July noon on Walnut Gulch came out
                    # 67.2 s/m against VIC's 40.1: Ri -0.09 where VIC's is at
                    # its -0.5 floor.)
                    bare_eb = bare_in = None
                    # VIC 5.0.1 forms the same average and bare soil at
                    # func_surf_energy_bal.c 5.0.1:703-745 and :762-805; its
                    # arno_evap takes the residual as a fraction (resid_mm
                    # None, as on the water-balance path).
                    if mosaic.fcanopy is not None:
                        z0_s = tile(th['rough'])
                        d_s = energy.VEG_RATIO_DH_HEIGHT * (z0_s
                                                            / VEG_RATIO_RL_HEIGHT)
                        top_s = 2.0 + vegetation.DISPL_FACTOR * d_s
                        pos_u = u_surf0 > 0.0
                        safe_u = torch.where(pos_u, u_surf0,
                                             torch.ones_like(u_surf0))
                        ra_b = torch.where(
                            pos_u,
                            torch.log(top_s / z0_s)
                            * torch.log(top_s / (0.1 * z0_s))
                            / VON_KARMAN_SQ / safe_u,
                            torch.full_like(u_surf0, vegetation.HUGE_RESIST))
                        # CalcAerodynamic.c 5.1.0:55-56 then :178-179: the log
                        # ratio first, the wind after it, as the C rounds it
                        u_b = (torch.log((2.0 + z0_s) / z0_s)
                               / torch.log((SOIL_WINDH - d_s) / z0_s) * u_surf0)
                        bare_eb = dict(fc=mosaic.fcanopy[:, :, m], ra=ra_b,
                                       ref_height=(torch.zeros_like(u_surf0)
                                                   + SOIL_WINDH),
                                       displacement=d_s, roughness=z0_s,
                                       wind=u_b)
                        d0_b = tile(th['depth'][:, 0])
                        # func_surf_energy_bal.c 5.1.0:732: `if (!SNOWING)`
                        # gates the bare soil as it gates everything else
                        # (solve_snow.c:164-171's snow->snow, as on the
                        # water-balance path).
                        snowing_eb = (tile(has_snow) | tile(wb_swq0 > 0.0)
                                      | tile(wb_sky_snow > 0.0))
                        if any_over:
                            snowing_eb = snowing_eb | (over_t
                                                       & (wb_int_snow0 > 0.0))
                        bare_in = dict(
                            # arno_evap.c 5.1.0:50-54: the liquid, moist - ice
                            # (frost-weighted with SPATIAL_FROST)
                            moist=(moist[:, :, 0] if soil_ice is None
                                   else moist[:, :, 0] - _ice_eff()[:, :, 0]),
                            max_moist=tile(soil_params.max_moist[:, 0]),
                            b_infilt=tile(soil_params.b_infilt), depth=d0_b,
                            resid_frac=(tile(soil_params.resid_moist[:, 0])
                                        / (d0_b * MM_PER_M)),
                            resid_mm=(tile(soil_params.resid_moist[:, 0])
                                      if core_version.at_least('5.1.0')
                                      else None))
                    return dict(_sn_no_over=_sn_no_over, alb_t=alb_t, atten_one_eb=atten_one_eb, atten_t=atten_t, bare_eb=bare_eb, bare_frac=bare_frac, bare_in=bare_in, cache=cache, cov_end_t=cov_end_t, cs=cs, cs1=cs1, cs2=cs2, d0_b=d0_b, d_s=d_s, d_solve=d_solve, d_t=d_t, dcov_t=dcov_t, fc_t=fc_t, kap=kap, kap1=kap1, kap2=kap2, lai_d=lai_d, lai_t=lai_t, le=le, m=m, ns_t=ns_t, ntl=ntl, ov=ov, pos_u=pos_u, ra_b=ra_b, ra_canopy=ra_canopy, ra_t=ra_t, safe_u=safe_u, snowing_eb=snowing_eb, sw_in=sw_in, tair_now=tair_now, th=th, tile=tile, top_s=top_s, two=two, u_b=u_b, u_surf=u_surf, u_surf0=u_surf0, us2=us2, vpd_now=vpd_now, z0_s=z0_s, z0_solve=z0_solve, z0_t=z0_t, zref_t=zref_t)
                _blk_res = _fusion.run(_eb_pre, *_bound(locals(), ('th', 'cs', )))
                if _blk_res['_sn_no_over'] is not _UNSET:
                    _sn_no_over = _blk_res['_sn_no_over']
                if _blk_res['alb_t'] is not _UNSET:
                    alb_t = _blk_res['alb_t']
                if _blk_res['atten_one_eb'] is not _UNSET:
                    atten_one_eb = _blk_res['atten_one_eb']
                if _blk_res['atten_t'] is not _UNSET:
                    atten_t = _blk_res['atten_t']
                if _blk_res['bare_eb'] is not _UNSET:
                    bare_eb = _blk_res['bare_eb']
                if _blk_res['bare_frac'] is not _UNSET:
                    bare_frac = _blk_res['bare_frac']
                if _blk_res['bare_in'] is not _UNSET:
                    bare_in = _blk_res['bare_in']
                if _blk_res['cache'] is not _UNSET:
                    cache = _blk_res['cache']
                if _blk_res['cov_end_t'] is not _UNSET:
                    cov_end_t = _blk_res['cov_end_t']
                cs = _blk_res['cs']
                if _blk_res['cs1'] is not _UNSET:
                    cs1 = _blk_res['cs1']
                if _blk_res['cs2'] is not _UNSET:
                    cs2 = _blk_res['cs2']
                if _blk_res['d0_b'] is not _UNSET:
                    d0_b = _blk_res['d0_b']
                if _blk_res['d_s'] is not _UNSET:
                    d_s = _blk_res['d_s']
                if _blk_res['d_solve'] is not _UNSET:
                    d_solve = _blk_res['d_solve']
                if _blk_res['d_t'] is not _UNSET:
                    d_t = _blk_res['d_t']
                if _blk_res['dcov_t'] is not _UNSET:
                    dcov_t = _blk_res['dcov_t']
                if _blk_res['fc_t'] is not _UNSET:
                    fc_t = _blk_res['fc_t']
                if _blk_res['kap'] is not _UNSET:
                    kap = _blk_res['kap']
                if _blk_res['kap1'] is not _UNSET:
                    kap1 = _blk_res['kap1']
                if _blk_res['kap2'] is not _UNSET:
                    kap2 = _blk_res['kap2']
                if _blk_res['lai_d'] is not _UNSET:
                    lai_d = _blk_res['lai_d']
                if _blk_res['lai_t'] is not _UNSET:
                    lai_t = _blk_res['lai_t']
                if _blk_res['le'] is not _UNSET:
                    le = _blk_res['le']
                if _blk_res['m'] is not _UNSET:
                    m = _blk_res['m']
                if _blk_res['ns_t'] is not _UNSET:
                    ns_t = _blk_res['ns_t']
                if _blk_res['ntl'] is not _UNSET:
                    ntl = _blk_res['ntl']
                if _blk_res['ov'] is not _UNSET:
                    ov = _blk_res['ov']
                if _blk_res['pos_u'] is not _UNSET:
                    pos_u = _blk_res['pos_u']
                if _blk_res['ra_b'] is not _UNSET:
                    ra_b = _blk_res['ra_b']
                if _blk_res['ra_canopy'] is not _UNSET:
                    ra_canopy = _blk_res['ra_canopy']
                if _blk_res['ra_t'] is not _UNSET:
                    ra_t = _blk_res['ra_t']
                if _blk_res['safe_u'] is not _UNSET:
                    safe_u = _blk_res['safe_u']
                if _blk_res['snowing_eb'] is not _UNSET:
                    snowing_eb = _blk_res['snowing_eb']
                if _blk_res['sw_in'] is not _UNSET:
                    sw_in = _blk_res['sw_in']
                if _blk_res['tair_now'] is not _UNSET:
                    tair_now = _blk_res['tair_now']
                th = _blk_res['th']
                if _blk_res['tile'] is not _UNSET:
                    tile = _blk_res['tile']
                if _blk_res['top_s'] is not _UNSET:
                    top_s = _blk_res['top_s']
                if _blk_res['two'] is not _UNSET:
                    two = _blk_res['two']
                if _blk_res['u_b'] is not _UNSET:
                    u_b = _blk_res['u_b']
                if _blk_res['u_surf'] is not _UNSET:
                    u_surf = _blk_res['u_surf']
                if _blk_res['u_surf0'] is not _UNSET:
                    u_surf0 = _blk_res['u_surf0']
                if _blk_res['us2'] is not _UNSET:
                    us2 = _blk_res['us2']
                if _blk_res['vpd_now'] is not _UNSET:
                    vpd_now = _blk_res['vpd_now']
                if _blk_res['z0_s'] is not _UNSET:
                    z0_s = _blk_res['z0_s']
                if _blk_res['z0_solve'] is not _UNSET:
                    z0_solve = _blk_res['z0_solve']
                if _blk_res['z0_t'] is not _UNSET:
                    z0_t = _blk_res['z0_t']
                if _blk_res['zref_t'] is not _UNSET:
                    zref_t = _blk_res['zref_t']

                def evap_of(net_bare_rad, ra_neutral, net_short=None,
                            ra_used=None):
                    # `net_short` is what calc_rc sees (NetShortBare): the
                    # tile's own, or 0 under a thin pack (:231, :757). The
                    # canopy's resistance is Ra_veg[1]: the tile's own, or
                    # the overstory's canopy value where there is one.
                    ra_veg1 = torch.where(over_t, ra_canopy, ra_neutral)
                    bs = (None if bare_in is None or ra_used is None
                          else {**bare_in, 'ra': ra_used})
                    got = mosaic.evap_per_tile(
                        m, wdew, water_in, moist, w_crit, w_wilt,
                        ns_t if net_short is None else net_short,
                        tair_now, vpd_now, ra_veg1,
                        net_bare_rad, batch.elevation,
                        lai_scale=p['lai_scale'], rmin_scale=p['rmin_scale'],
                        lai_water_factor=lwf,
                        dt=step_seconds, return_rc=do_carbon,
                        share_layer_moist=share_layer_moist, bare_soil=bs,
                        wilting_clamp=True, ice=_ice_eff(),
                        atten_one=atten_one_eb,
                        wdew_per_plant=not _sub['divide_fcan'])
                    n_out = 4 if bs is None else 5
                    if bs is not None:
                        got = tuple(got[:4]) + (torch.where(
                            snowing_eb, torch.zeros_like(got[4]), got[4]),
                        ) + tuple(got[5:])
                    cache['last'] = got[:n_out]
                    # func_surf_energy_bal.c's Evap in m/s, formed the C's way
                    cache['ms'] = getattr(mosaic, 'last_evap_ms', None)
                    if do_carbon:
                        # surface_fluxes.c:736-754 reads the canopy
                        # resistance the converged balance left, so what is
                        # kept here is the last evaluation's, which after the
                        # re-evaluation at :552 is the root's.
                        cache['rc'] = got[n_out]
                        cache['dry'] = got[n_out + 1]
                        if rc_photo:
                            cache['rs'] = mosaic.rs_last
                    tr, ce = got[0], got[1]
                    if bs is not None:
                        return tr.sum(-1) + ce + got[4]
                    return tr.sum(-1) + ce

                # func_surf_energy_bal.c 5.1.0:730-761: `if (!SNOWING)`
                # gates canopy_evap and arno_evap alike. A tile with
                # snow->snow (a pack the step opened with, snow from the
                # sky, or an overstory holding snow: solve_snow.c 5.1.0
                # :164-168) transpires and evaporates nothing in the
                # balance (latent_heat = -RHOFW Le Evap with Evap = 0,
                # :766), keeps its canopy store, and its throughfall is
                # what reached it (calc_surf_energy_bal.c 5.1.0:605-612
                # skips the assignment). Only the bare soil was gated here,
                # so an overstory under snow evaporated the liquid its
                # canopy had just melted.
                snowing_all = (tile(has_snow) | tile(wb_swq0 > 0.0)
                               | tile(wb_sky_snow > 0.0))
                if any_over:
                    snowing_all = snowing_all | (over_t & (wb_int_snow0 > 0.0))
                _evap_of_raw = evap_of

                def evap_of(net_bare_rad, ra_neutral, net_short=None,
                            ra_used=None):
                    _e = _evap_of_raw(net_bare_rad, ra_neutral, net_short,
                                      ra_used)
                    got = list(cache['last'])
                    z_ = torch.zeros_like(got[1])
                    got[0] = torch.where(snowing_all.unsqueeze(-1),
                                         torch.zeros_like(got[0]), got[0])
                    got[1] = torch.where(snowing_all, z_, got[1])
                    got[2] = torch.where(snowing_all,
                                         water_in.expand_as(got[2]), got[2])
                    kept = wdew.expand_as(got[3])
                    if mosaic.fcanopy is not None and core_version.at_least('5.1.0'):
                        # solve_snow.c 5.1.0:251, Wdew *= fcanopy under snow
                        # on every tile but the bare-soil one
                        _gone = mosaic.fcanopy[:, :, m] <= 0.0
                        if mosaic.bare_tile is not None:
                            _gone = _gone & ~mosaic.bare_tile
                        kept = torch.where(_gone, torch.zeros_like(kept), kept)
                        if not _sub['divide_fcan']:
                            # a later snow step: the store is read per plant
                            # as it stands; an overstory tile's comes from
                            # the canopy stage, which applied :251 already
                            _sc = mosaic.fcanopy[:, :, m] * kept
                            if mosaic.bare_tile is not None:
                                _sc = torch.where(mosaic.bare_tile, kept, _sc)
                            if any_over:
                                _sc = torch.where(over_t, kept, _sc)
                            kept = _sc
                    got[3] = torch.where(snowing_all, kept, got[3])
                    if len(got) > 4:
                        got[4] = torch.where(snowing_all,
                                             torch.zeros_like(got[4]), got[4])
                    cache['last'] = tuple(got)
                    _ret = torch.where(snowing_all, torch.zeros_like(_e), _e)
                    if cache.get('ms') is not None:
                        return _ret, torch.where(snowing_all,
                                                 torch.zeros_like(_e),
                                                 cache['ms'])
                    return _ret

                # The cell's pack, per tile, for the surface solve to include
                # where it is thin (surface_fluxes.c:585-591). The pack is
                # per cell and the surface per tile, so every tile of a
                # thin-pack cell folds the same pack in -- VIC has a pack
                # per tile; a named structural difference.
                # (core.fusion.run: this block one fused region on the graph path)
                def _eb_pack():
                    thin_c = swq_c = dep_c = kappa_snow = _dep_mid = safe_d = pt = pack = fz_in = _frz = _d1 = _mm = _m0 = _mx0 = _bb0 = _ex0 = _t_old = _ice0 = _fs = _act = fd = _t1o = _fd_d1 = _fd_ice0 = _fd_moist0 = _UNSET
                    thin_c = new_state['thin'] & has_snow
                    # snow_melt.c 5.1.0:464: Ice + pack_water + surf_water
                    swq_c = (new_state['ice'] + new_state['pack_water']
                             + new_state['surf_water'])
                    # calc_surf_energy_bal.c:232-239: the pack's conductivity,
                    # SNOW_CONDUCT * density^2 / depth, and the temperature the
                    # pack's own solve left, for the conduction term the soil
                    # surface balances against.
                    dep_c = snow_state.get('depth')
                    if dep_c is None:
                        kappa_snow = torch.zeros_like(prcp)
                    else:
                        # surface_fluxes.c 5.1.0:603 passes snow_depth =
                        # (step_snow.depth + iter_snow.depth) / 2, the mean of
                        # the step's opening and closing depths, and
                        # calc_surf_energy_bal.c 5.1.0:205-211 divides by it
                        # where the new depth is positive
                        _dep_mid = dep_c
                        if st.get('depth') is not None:
                            _dep_mid = 0.5 * (st['depth'] + dep_c)
                        safe_d = torch.where(dep_c > 0.0, _dep_mid,
                                             torch.ones_like(dep_c))
                        kappa_snow = torch.where(
                            dep_c > 0.0,
                            # calc_surf_energy_bal.c 5.1.0:203, left to right
                            snow.SNOW_CONDUCT * new_state['density']
                            * new_state['density'] / safe_d,
                            torch.zeros_like(dep_c))
                    # Per tile already when the packs are; `tile` is then the
                    # identity for them and only the cell's weather is spread.
                    pt = (lambda x: x) if per_tile_snow else tile
                    pack = dict(
                        # calc_surf_energy_bal's snow_coverage is
                        # iter_snow.coverage (surface_fluxes.c 5.1.0:602)
                        coverage=pt(cov_new if spatial_snow else cov_end_t),
                        # solve_snow's melt_energy, the energy that went into
                        # the cover lost this step (func_surf_energy_bal.c
                        # 5.1.0:854)
                        melt_energy=pt(melt_energy_t),
                        kappa_snow=pt(kappa_snow),
                        t_snow_surf=pt(new_state['surf_temp']),
                        thin=pt(thin_c), swq=pt(swq_c),
                        surf_water=pt(new_state['surf_water']),
                        old_t_surf=pt(old_snow_surf_temp),
                        density=pt(new_state['density']),
                        net_short_snow=(
                            # solve_snow.c:458 scales NetShortSnow by the cover
                            # fraction before calc_surf_energy_bal sees it.
                            # NetShortSnow is (1-AlbedoUnder)*ShortUnderIn
                            # (:312), where ShortUnderIn is the canopy-attenuated
                            # shortwave where a canopy ran (:173), not the sky's.
                            pt((under_short
                                if (per_tile_snow and under_short is not None)
                                else (atm['shortwave'][_t1].unsqueeze(-1)
                                      if per_tile_snow
                                      else atm['shortwave'][_t1])))
                            * (1.0 - alb_under) * cov_new
                            if spatial_snow else
                            # solve_snow.c 5.1.0:312, (1 - AlbedoUnder)
                            # ShortUnderIn: under a canopy that ran, the
                            # attenuated shortwave (:173), not the sky's
                            ((tile(atm['shortwave'][_t1]) if under_short is None
                              else under_short) * (1.0 - albedo)
                             if per_tile_snow
                             else tile(atm['shortwave'][_t1] * (1.0 - albedo)))),
                        eact_air=tile(atm['eact_air'][_t1]),
                        press=tile(atm['press'][_t1]), vpd=tile(atm['vpd'][_t1]),
                        # a thin pack's rain advection (snow_step), which
                        # func_surf_energy_bal.c 5.1.0:855 adds to its balance
                        advection=pt(se['advection']))
                    if _sub['pin_tgrnd'] and _sub['soil_adv'] is not None:
                        pack['stale_adv'] = _sub['soil_adv']
                        pack['stale_dcc'] = _sub['soil_dcc']
                    # func_surf_energy_bal.c:561-574 and prepare_full_energy.c
                    # :56-77. Both work in volume fractions of the top layer:
                    # moist0 = layer[0].moist / (depth[0] * MM_PER_M), and ice0
                    # is what the unfrozen-water curve will not hold at the mean
                    # of the two OLD node temperatures.
                    fz_in = None
                    if frozen_soil:
                        from ..core import frozen as _frz
                        _d1 = tile(th['depth'][:, 0])
                        _mm = _d1 * MM_PER_M
                        _m0 = moist[:, :, 0] / _mm
                        _mx0 = tile(soil_params.max_moist[:, 0]) / _mm
                        _bb0 = tile(soil_params.bubble[:, 0])
                        _ex0 = tile(soil_params.expt[:, 0])
                        _t_old = 0.5 * (t_node[:, :, 0] + t_node[:, :, 1])
                        _ice0 = torch.clamp(
                            _m0 - _frz.maximum_unfrozen_water(_t_old, _mx0, _bb0,
                                                              _ex0), min=0.0)
                        _fs = getattr(soil_params, 'fs_active', None)
                        if _fs is None:
                            _act = torch.ones_like(_m0, dtype=torch.bool)
                        else:
                            _act = _fs
                            while _act.dim() < _m0.dim():
                                _act = _act.unsqueeze(-1)
                            _act = _act.expand_as(_m0)
                        fz_in = dict(moist=_m0, max_moist=_mx0, bubble=_bb0,
                                     expt=_ex0, ice0=_ice0, d1=_d1, active=_act)
                    fd = None
                    _t1o = t_node[:, :, 1]
                    if not quick_flux:
                        # QUICK_FLUX FALSE: the residual solves the profile at
                        # each trial Ts. The node properties are the step's
                        # (redistributed after the last column step); the
                        # profile, the prepare outputs and the reduced count
                        # are this solve's.
                        _fd_d1 = tile(th['depth'][:, 0])
                        if exp_trans:
                            # calc_surf_energy_bal.c:165-177: T1_old is the
                            # old profile interpolated to depth[0].
                            _t1o = soiltemp.t1_old_exp_trans(
                                t_node, fd_static['zsum'], _fd_d1)
                        if fz_in is not None:
                            _fd_ice0, _fd_moist0 = fz_in['ice0'], fz_in['moist']
                        else:
                            _fd_moist0 = moist[:, :, 0] / (_fd_d1 * MM_PER_M)
                            _fd_ice0 = torch.zeros_like(_fd_moist0)
                        fd = dict(
                            t_node=t_node, moist_node=node_props['moist'],
                            ice_node=node_props['ice'],
                            kappa_node=node_props['kappa'], cs_node=node_props['cs'],
                            porosity_node=fd_static['porosity_node'],
                            bubble_node=fd_static['bubble_node'],
                            expt_node=fd_static['expt_node'],
                            alpha=fd_static['alpha'], beta=fd_static['beta'],
                            gamma=fd_static['gamma'], zsum=fd_static['zsum'],
                            dp=th['dp'], bulk_dens_min=fd_static['bulk_dens_min'],
                            soil_dens_min=fd_static['soil_dens_min'],
                            quartz=th['quartz'],
                            bulk_density=th['bulk_density'],
                            soil_density=th['soil_density'],
                            organic=fd_static['organic'], depth=th['depth'],
                            max_moist=soil_params.max_moist, expt=soil_params.expt,
                            bubble=soil_params.bubble, moist=moist,
                            nlayers=fd_static['nlayers'], nnodes=nnodes,
                            implicit=implicit, noflux=noflux, exp_trans=exp_trans,
                            fs_active=fd_static['fs_active'],
                            frozen_soil=frozen_soil,
                            tmp_nnodes=(soiltemp.quick_solve_nodes(t_node, nnodes)
                                        if quick_solve else None),
                            ice0=_fd_ice0, moist0=_fd_moist0,
                            porosity0=tile(fd_static['porosity'][:, 0]),
                            bubble0=tile(soil_params.bubble[:, 0]),
                            expt0=tile(soil_params.expt[:, 0]),
                            # the implicit solve skips padding tiles (Cv 0)
                            solve_cols=mosaic.cv > 0.0)
                    return dict(_act=_act, _bb0=_bb0, _d1=_d1, _dep_mid=_dep_mid, _ex0=_ex0, _fd_d1=_fd_d1, _fd_ice0=_fd_ice0, _fd_moist0=_fd_moist0, _frz=_frz, _fs=_fs, _ice0=_ice0, _m0=_m0, _mm=_mm, _mx0=_mx0, _t1o=_t1o, _t_old=_t_old, dep_c=dep_c, fd=fd, fz_in=fz_in, kappa_snow=kappa_snow, pack=pack, pt=pt, safe_d=safe_d, swq_c=swq_c, thin_c=thin_c)
                _blk_res = _fusion.run(_eb_pack)
                if _blk_res['_act'] is not _UNSET:
                    _act = _blk_res['_act']
                if _blk_res['_bb0'] is not _UNSET:
                    _bb0 = _blk_res['_bb0']
                if _blk_res['_d1'] is not _UNSET:
                    _d1 = _blk_res['_d1']
                if _blk_res['_dep_mid'] is not _UNSET:
                    _dep_mid = _blk_res['_dep_mid']
                if _blk_res['_ex0'] is not _UNSET:
                    _ex0 = _blk_res['_ex0']
                if _blk_res['_fd_d1'] is not _UNSET:
                    _fd_d1 = _blk_res['_fd_d1']
                if _blk_res['_fd_ice0'] is not _UNSET:
                    _fd_ice0 = _blk_res['_fd_ice0']
                if _blk_res['_fd_moist0'] is not _UNSET:
                    _fd_moist0 = _blk_res['_fd_moist0']
                if _blk_res['_frz'] is not _UNSET:
                    _frz = _blk_res['_frz']
                if _blk_res['_fs'] is not _UNSET:
                    _fs = _blk_res['_fs']
                if _blk_res['_ice0'] is not _UNSET:
                    _ice0 = _blk_res['_ice0']
                if _blk_res['_m0'] is not _UNSET:
                    _m0 = _blk_res['_m0']
                if _blk_res['_mm'] is not _UNSET:
                    _mm = _blk_res['_mm']
                if _blk_res['_mx0'] is not _UNSET:
                    _mx0 = _blk_res['_mx0']
                if _blk_res['_t1o'] is not _UNSET:
                    _t1o = _blk_res['_t1o']
                if _blk_res['_t_old'] is not _UNSET:
                    _t_old = _blk_res['_t_old']
                if _blk_res['dep_c'] is not _UNSET:
                    dep_c = _blk_res['dep_c']
                if _blk_res['fd'] is not _UNSET:
                    fd = _blk_res['fd']
                if _blk_res['fz_in'] is not _UNSET:
                    fz_in = _blk_res['fz_in']
                if _blk_res['kappa_snow'] is not _UNSET:
                    kappa_snow = _blk_res['kappa_snow']
                if _blk_res['pack'] is not _UNSET:
                    pack = _blk_res['pack']
                if _blk_res['pt'] is not _UNSET:
                    pt = _blk_res['pt']
                if _blk_res['safe_d'] is not _UNSET:
                    safe_d = _blk_res['safe_d']
                if _blk_res['swq_c'] is not _UNSET:
                    swq_c = _blk_res['swq_c']
                if _blk_res['thin_c'] is not _UNSET:
                    thin_c = _blk_res['thin_c']
                eb = energy.surface_step(
                    # CLOSE_ENERGY: "T of canopy or air"
                    # (calc_surf_energy_bal.c:28).
                    (_sub['tcanopy'] if _sub['tcanopy'] is not None
                     else tile(atm['tair'][_t1])),
                    t_node[:, :, 0], _t1o,
                    tile(th['avg_temp']), tile(th['depth'][:, 0]),
                    kap1, kap2, cs1, cs2, tile(th['dp']), step_seconds,
                    atten_t, ns_t,
                    (tile(atm['longwave'][_t1]) if canopy_long_out is None
                     else canopy_long_out), ra_t, zref_t,
                    d_solve, z0_solve, u_surf,
                    tile(atm['air_dens'][_t1]), tile(le), evap_of,
                    th['depth'].unsqueeze(1).expand(nc, mosaic.n_tiles, -1),
                    pack=pack, bare_frac=bare_frac,
                    frozen_soil=fz_in, bare=bare_eb, fd=fd,
                    grnd_flux_type=grnd_flux_type, depth_lsum=_eb_lsum)
                # calc_surf_energy_bal.c:552 evaluates func_surf_energy_bal
                # once more at the solved temperature, so the state advances
                # from the root and not from the last bisection trial: that
                # evaluation is surface_step's residual at the root
                # (`at_root`), which calls evap_of last with the very
                # arguments (net_bare_rad, the neutral resistance, the
                # blended one) this once repeated, so cache['last'] is the
                # root's already
                tr, ce, tf, nw = cache['last'][:4]
                if bare_eb is not None:
                    # arno_evap.c:192 writes layer 0 only (as on the
                    # water-balance path)
                    bare_t = cache['last'][4]
                    col_evap_t = torch.cat(
                        [tr[..., :1] + bare_t.unsqueeze(-1), tr[..., 1:]],
                        dim=-1)
                    bare_cell = _tsum(bare_t * mosaic.cv)
                    evap_bares.append(bare_cell)
                t_node = eb['t_node']
                if _sub['pin_tgrnd']:
                    # surface_fluxes.c:578 and func_surf_energy_bal.c
                    # :617-627: a thin-pack step leaves the snow's advection
                    # and its deltaCC in soil_energy for the next sub-step
                    _z = torch.zeros_like(eb['grnd_flux'])
                    _pa = (_z if _sub['soil_adv'] is None
                           else _sub['soil_adv'])
                    _pd = (_z if _sub['soil_dcc'] is None
                           else _sub['soil_dcc'])
                    _thin_t = pack['thin']
                    _sub['soil_adv'] = torch.where(
                        _thin_t, pack['advection'], _pa)
                    _sub['soil_dcc'] = torch.where(
                        _thin_t, eb['delta_cc'], _pd)
                if do_carbon:
                    _carbon_here(t, m, cache['rc'], cache['dry'], t_node,
                                 cache.get('rs'))
                cv = mosaic.cv
                tr_t, tf_t = tr, tf                 # per tile, for the columns
                _thin_dev = _thin_pre = None
                if graphsafe.ACTIVE:
                    # graph-safe: the branch on every step, and what it
                    # wrote kept only where some tile is thin -- as the
                    # eager step runs it -- the values it found elsewhere
                    # (_thin_blend)
                    _thin_any = True
                    _thin_dev = thin_c.any()
                    _thin_pre = _thin_snapshot(snow_state, tile_rows,
                                               tf_t)
                else:
                    _thin_any = bool(thin_c.any())
                if _thin_any:
                    # calc_surf_energy_bal.c:643-700, on the cell's pack with
                    # the tiles' solve cover-weighted: the vapour flux capped
                    # at the pack, applied to swq and the surface liquid;
                    # a non-negative refreeze energy refreezes liquid up to
                    # what there is, a negative one melts ice up to what
                    # there is and the melt reaches the soil (surface_fluxes.c
                    # :640-643, step_ppt += step_melt); the pack takes the
                    # surface temperature clipped at 0 C and its depth from
                    # the density it has.
                    # With a pack per tile the surface solve's terms are
                    # already that tile's; with one per cell they have to be
                    # cover-weighted onto it first.
                    wsum1 = ((lambda x: x) if per_tile_snow
                             else (lambda x: _tsum(x * cv)))
                    v = wsum1(eb['vapor_flux'])
                    rf = wsum1(eb['refreeze_energy'])
                    ts_c = wsum1(eb['t_surf'])
                    lat_vol = snow.CONST_LATICE * snow.CONST_RHOFW
                    sw0 = new_state['surf_water']
                    v = torch.maximum(v, -swq_c)
                    swq1 = swq_c + v
                    sw1 = torch.clamp(sw0 + v, min=0.0)
                    refrozen = torch.where(
                        rf >= 0.0,
                        torch.minimum(rf / lat_vol * step_seconds, sw1),
                        torch.zeros_like(rf))
                    # calc_surf_energy_bal.c 5.1.0:650-656: refreezing more
                    # than the liquid left after the vapour flux reports
                    # only the energy of what did refreeze
                    rf = torch.where(
                        (rf >= 0.0)
                        & (rf / lat_vol * step_seconds > sw1),
                        refrozen * lat_vol / step_seconds, rf)
                    sw2 = sw1 - refrozen
                    melt_thin = torch.where(rf < 0.0, rf.abs() / lat_vol
                                            * step_seconds,
                                            torch.zeros_like(rf))
                    swq2 = swq1 - melt_thin
                    melt_thin = torch.where(swq2 < 0.0, melt_thin + swq2,
                                            melt_thin)
                    swq2 = torch.clamp(swq2, min=0.0)
                    # snow_melt.c 5.1.0:90 on the next call, Ice = swq -
                    # pack_water - surf_water
                    ice2 = torch.clamp(swq2 - new_state['pack_water'] - sw2,
                                       min=0.0)
                    dens = new_state['density']
                    safe_dens = torch.where(dens > 0.0, dens,
                                            torch.ones_like(dens))
                    gone = swq2 <= 0.0
                    sel = lambda a, b: torch.where(thin_c, a, b)
                    snow_state['ice'] = sel(ice2, snow_state['ice'])
                    snow_state['surf_water'] = sel(sw2, snow_state['surf_water'])
                    if 'swq_c' in snow_state:
                        # the pack's swq as calc_surf_energy_bal.c 5.1.0
                        # :637-665 leaves it, which the next snow_melt reads
                        snow_state['swq_c'] = sel(swq2, snow_state['swq_c'])
                    snow_state['surf_temp'] = sel(
                        torch.where(gone, torch.zeros_like(ts_c),
                                    torch.clamp(ts_c, max=0.0)),
                        snow_state['surf_temp'])
                    if 'cold_content' in snow_state:
                        # calc_surf_energy_bal.c 5.1.0:668-672; a pack that
                        # is gone keeps the C's SurfaceCC, which nothing
                        # reads before the next snowfall replaces it
                        snow_state['cold_content'] = sel(
                            torch.where(gone, snow_state['cold_content'],
                                        snow.CONST_VCPICE_WQ
                                        * torch.clamp(ts_c, max=0.0) * swq2),
                            snow_state['cold_content'])
                    if 'depth' in snow_state:
                        snow_state['depth'] = sel(
                            torch.where(gone, torch.zeros_like(swq2),
                                        snow.CONST_RHOFW * swq2 / safe_dens),
                            snow_state['depth'])
                        snow_state['density'] = sel(
                            torch.where(gone, torch.zeros_like(dens), dens),
                            snow_state['density'])
                    # the melt goes to the soil, the sublimation to the
                    # record (put_data.c:626-627, loss positive)
                    thin_melt = sel(melt_thin * MM_PER_M,
                                    torch.zeros_like(melt_thin))
                    if spatial_snow:
                        # calc_surf_energy_bal.c:680-689. A pack thin enough
                        # to be folded into the surface solve does its melting
                        # here, so the cover fraction is recomputed here too,
                        # from the fraction solve_snow.c:376 just left. The
                        # C's melt argument is `*melt - snow->vapor_flux`
                        # with the flux still negative for a loss (:718 flips
                        # it), which is melt plus sublimation as `thin_sub`
                        # already carries it. The snowfall argument is the
                        # zero solve_snow.c:468 left behind -- all falling
                        # snow has been added to the pack -- passed through
                        # surface_fluxes.c:612, so this call never takes the
                        # snowfall arm however hard it is snowing.
                        thin_swq = torch.clamp(swq2, min=0.0)
                        thin_depth = torch.where(
                            gone, torch.zeros_like(swq2),
                            snow.CONST_RHOFW * swq2 / safe_dens)
                        sp2 = snow.snow_coverage(
                            store_snow=sp['store_snow'], max_slope=sp_slope,
                            old_coverage=cov_new, swq=thin_swq,
                            old_swq=sp_swq, depth=thin_depth,
                            old_depth=sp_depth,
                            melt=melt_thin - v,
                            max_snow_depth=sp['max_snow_depth'],
                            snowfall=torch.zeros_like(snow_p),
                            store_swq=sp['store_swq'],
                            snow_distrib_slope=sp['snow_distrib_slope'],
                            store_coverage=sp['store_coverage'])
                        # :706-716: a pack that melted out in the thin branch
                        # leaves no cover and no stored layer, but the C
                        # only reaches calc_snow_coverage when swq survived
                        # (:671), so the rest of the distribution -- slope,
                        # store flag, maximum depth, buried fraction -- is
                        # the solve's, not the recompute's.
                        _sp_pre = dict(sp)
                        for _k in sp2:
                            sp[_k] = torch.where(thin_c, sp2[_k], sp[_k])
                        _tg = thin_c & gone
                        sp['coverage'] = torch.where(
                            _tg, torch.zeros_like(sp['coverage']),
                            sp['coverage'])
                        sp['store_swq'] = torch.where(
                            _tg, torch.zeros_like(sp['store_swq']),
                            sp['store_swq'])
                        sp['store_coverage'] = torch.where(
                            _tg, _sp_pre['store_coverage'],
                            sp['store_coverage'])
                        sp['max_snow_depth'] = torch.where(
                            _tg, _sp_pre['max_snow_depth'],
                            sp['max_snow_depth'])
                        sp['snow_distrib_slope'] = torch.where(
                            _tg, _sp_pre['snow_distrib_slope'],
                            sp['snow_distrib_slope'])
                        sp['store_snow'] = torch.where(
                            _tg, _sp_pre['store_snow'], sp['store_snow'])
                        snow_covers[-1] = to_cell(sp['coverage'])
                        # surface_fluxes.c 5.1.0:910-933: the folded pack's
                        # advection, deltaCC and refreeze energy (already
                        # cover-weighted in func_surf_energy_bal.c:627-628)
                        # reach the record weighted again, by coverage +
                        # delta_coverage with the cover this recompute
                        # left, or by the step-open cover (1 if that was 0
                        # and it rained or snowed) where the pack is gone.
                        _lc = torch.where(
                            (cov_prev == 0.0)
                            & (tile(atm['prcp'][_t1]) > 0.0),
                            torch.ones_like(cov_prev), cov_prev)
                        _thin_f = torch.where(gone, _lc,
                                              sp['coverage'] + delta_cov)
                        if 'advected_sensible' in tile_rows:
                            # :871, :914-924: the folded pack reports the
                            # stale soil_energy.advected_sensible, the last
                            # step's, weighted the same way
                            _adv_t = _adv_prev * _thin_f
                            sp['adv_sens'] = torch.where(
                                thin_c, _adv_t, sp['adv_sens'])
                            _as = -_adv_t
                            if over_t is not None:
                                _as = torch.where(over_t,
                                                  torch.zeros_like(_as), _as)
                            tile_rows['advected_sensible'] = torch.where(
                                thin_c, _as, tile_rows['advected_sensible'])
                            snow_eb['advected_sensible'][-1] = to_cell(
                                tile_rows['advected_sensible'])
                    # The pack these tiles END the step with is the one this
                    # branch just wrote, not the one snow_step left: the
                    # surface solve applied the vapour flux to it, refroze
                    # or melted part of it, and the state carried to the next
                    # step is `snow_state` as amended above. `swes` and
                    # `snow_depths` were appended ~400 lines earlier, before
                    # any of that, so the reported series was the pre-solve
                    # pack while the model ran on the post-solve one.
                    #
                    # That is a water-balance hole and only on this path:
                    # the residual is exactly the SWE the two disagree by --
                    # -2.06e-02 mm per cell per step on Walnut Gulch, where
                    # the water-only path closes at 1.1e-13. It shows up
                    # nowhere else, because every other output is taken from
                    # the amended state and only the record was stale.
                    #
                    # `snow_covers[-1]` and `sublimations[-1]` were already
                    # being patched here for the same reason; these two were
                    # missing from that list.
                    swes[-1] = to_cell(
                        snow_state['ice'] + snow_state['surf_water']
                        + snow_state['pack_water']) * MM_PER_M
                    if snow_depths and 'depth' in snow_state:
                        snow_depths[-1] = (to_cell(snow_state['depth'])
                                           * CM_PER_M)
                    if not spatial_snow:
                        # calc_surf_energy_bal.c:712-717: cover follows the
                        # pack, and the pack just changed.
                        _has_p = (snow_state['ice']
                                  + snow_state['surf_water']
                                  + snow_state['pack_water']) > 0.0
                        snow_covers[-1] = to_cell(_has_p.to(dtype))
                    if per_tile_snow:
                        # put_data.c 5.1.0:727-731, as on the water-balance
                        # path: the snow albedo and pack temperatures of
                        # the pack the step ends with, which a thin pack
                        # sublimated away does not have
                        _end_p = (snow_state['ice'] + snow_state['surf_water']
                                  + snow_state['pack_water']) > 0.0
                        salbedos[-1] = to_snow_cell(albedo, _end_p)
                        snow_surf_temps[-1] = to_snow_cell(
                            snow_state['surf_temp'], _end_p)
                        snow_pack_temps[-1] = to_snow_cell(
                            snow_state['pack_temp'], _end_p)
                    tf_t = tf_t + (thin_melt if per_tile_snow
                                   else thin_melt.unsqueeze(1))
                    # Only the thin tiles take the surface solve's vapour;
                    # the rest keep the sublimation their own pack computed.
                    # Substituting zeros for them lost that mass and opened
                    # a 8.1e-07 relative hole in the water balance.
                    thin_sub = sel(-v * MM_PER_M,
                                   sub_tile if per_tile_snow
                                   else sublimations[-1])
                    sublimations[-1] = (to_cell(thin_sub) if per_tile_snow
                                        else thin_sub)
                    # surface_fluxes.c 5.1.0:610 hands step_melt to
                    # calc_surf_energy_bal, which overwrites it with the
                    # thin pack's own melt (calc_surf_energy_bal.c 5.1.0
                    # :652-665), and that is what :860 and :951-952 report
                    # as snow->melt. The rain that ran through the pack in
                    # snow_melt reached the soil but is not OUT_SNOW_MELT.
                    thin_mrep = sel(melt_thin * MM_PER_M,
                                    melt_tile if per_tile_snow
                                    else melts[-1])
                    melts[-1] = (to_cell(thin_mrep) if per_tile_snow
                                 else thin_mrep)
                    # the six snow rows for a thin pack are the surface
                    # solve's (put_data.c:916, :936, :952, :509-510)
                    _dcc_rep = wsum1(eb['delta_cc'])
                    _rf_rep = rf
                    if spatial_snow:
                        _dcc_rep = _dcc_rep * _thin_f
                        _rf_rep = rf * _thin_f
                        tile_rows['advection'] = sel(
                            pack['advection'] * _thin_f,
                            tile_rows['advection'])
                        snow_eb['advection'][-1] = to_cell(
                            tile_rows['advection'])
                    snow_eb['latent_sub'][-1] = to_cell(sel(
                        -wsum1(eb['latent_sub']), tile_rows['latent_sub']))
                    snow_eb['deltacc'][-1] = to_cell(sel(
                        _dcc_rep, tile_rows['deltacc']))
                    snow_eb['rfrz_energy'][-1] = to_cell(sel(
                        _rf_rep, tile_rows['rfrz_energy']))
                    snow_eb['refreeze'][-1] = to_cell(sel(
                        _rf_rep / snow.CONST_LATICE * step_seconds,
                        tile_rows['rfrz_energy'] / snow.CONST_LATICE
                        * step_seconds))
                    if per_tile_snow:
                        tile_rows['deltacc'] = sel(
                            eb['delta_cc'] if not spatial_snow
                            else _dcc_rep, tile_rows['deltacc'])
                        tile_rows['rfrz_energy'] = sel(
                            _rf_rep, tile_rows['rfrz_energy'])
                if _thin_dev is not None:
                    tf_t = _thin_blend(_thin_dev, _thin_pre, snow_state,
                                       tile_rows, tf_t)
                if _sub['pin_tgrnd'] and per_tile_snow:
                    _erows = {k_: tile_rows[k_] for k_ in
                              ('advection', 'deltacc', 'rfrz_energy')}
                    # surface_fluxes.c:990-994 reports the ground flux
                    # the same way; the idle-tile fix below needs the
                    # steps' own per-tile soil values.
                    _erows['snow_flux'] = eb['snow_flux']
                    if spatial_snow:
                        # as reported (surface_fluxes.c:928-930): weighted
                        # by coverage + delta_coverage, so the idle-tile
                        # fix subtracts the mean that was accumulated
                        _erows['snow_flux'] = eb['snow_flux'] * tile(cov_eff)
                    _sub['eb_rows'].append(dict(rows=_erows,
                                                active=snowing_all,
                                                thin=thin_c))
                # out-vars: OUT_ADVECTION_BAND, OUT_RFRZ_ENERGY_BAND and
                # OUT_LATENT_SUB_BAND are the understory's terms
                # (put_data.c:889-905: energy.advection, .refreeze_energy,
                # .latent_sub, without the canopy's), which the rows hold
                # before the canopy is added below
                if 'adv' in band_rows:
                    band_rows['adv'].append(snow_eb['advection'][-1])
                if 'rfz' in band_rows:
                    band_rows['rfz'].append(snow_eb['rfrz_energy'][-1])
                if 'lsub' in band_rows:
                    band_rows['lsub'].append(-snow_eb['latent_sub'][-1])
                if canopy_lsub is not None:
                    # put_data.c 5.1.0:896, OUT_LATENT_SUB -= AtmosLatentSub
                    snow_eb['latent_sub'][-1] = (snow_eb['latent_sub'][-1]
                                                 - _tsum(canopy_lsub * cv))
                    _cvc = mosaic.cv
                    snow_eb['advection'][-1] = (snow_eb['advection'][-1]
                                                + _tsum(canopy_adv * _cvc))
                    snow_eb['rfrz_energy'][-1] = (snow_eb['rfrz_energy'][-1]
                                                  + _tsum(canopy_rfz * _cvc))
                    snow_eb['refreeze'][-1] = (snow_eb['refreeze'][-1]
                                               + _tsum(canopy_rfz * _cvc)
                                               / snow.CONST_LATICE * step_seconds)
                # (core.fusion.run: this block one fused region on the graph path)
                def _eb_fluxes(ce, wdew, ra_veg1_t, eb, ground_long_out):
                    evap = canopy_evap = throughfall = pack_ns_t = pack_nl_t = pack_ns = pack_nl = ra_rep = cond = pack_w = _cov_rep_in = cov_rep = _ts_end = ts_snow_rep = _ov_here = _snow_snow = _gate = _ins = _ra_atm = _tc_sol = _sh_sol = _tc_fail = _endp = _pack_long_out = _lw = _nlb = _nls = _UNSET
                    if wb_canopy_liquid is not None:
                        # an overstory canopy without snow evaporates its liquid
                        # (func_canopy_energy_bal.c 5.1.0:220-230,
                        # veg_var->canopyevap, put_data.c :619) on a tile the
                        # surface balance left alone (SNOWING)
                        ce = ce + wb_canopy_liquid
                    evap = _tsum(tr * cv.unsqueeze(-1))
                    canopy_evap = _tsum(ce * cv)
                    throughfall = _tsum(tf * cv)
                    wdew = nw
                    # put_data.c:671-688 accumulates the *conductance* over
                    # tiles, and :471-476 reports the resistance as its inverse.
                    # A cover-weighted mean of the resistances is a different
                    # number -- 4 per cent different on ten January days while
                    # the latent and sensible heat that use the resistance
                    # agreed to 0.0 and 0.7 -- because tiles with very different
                    # ra do not commute with the weighting.
                    # put_data.c:891-894 reports NetShortAtmos and NetLongAtmos,
                    # which under a pack are the pack's terms, not the bare
                    # surface's zeros (calc_atmos_energy_bal.c:64-80 sums over
                    # and under).
                    if spatial_snow:
                        # solve_snow.c:457-458, both scaled by the cover.
                        pack_ns_t = se['net_short'] * cov_new
                        pack_nl_t = se['net_long'] * cov_new
                    else:
                        # solve_snow.c 5.1.0:440-442: NetShortSnow and
                        # NetLongSnow times the cover the pack ends with
                        pack_ns_t = torch.where(has_p, se['net_short'],
                                                torch.zeros_like(melt_m))
                        pack_nl_t = torch.where(has_p, se['net_long'],
                                                torch.zeros_like(melt_m))
                    pack_ns, pack_nl = to_cell(pack_ns_t), to_cell(pack_nl_t)
                    # put_data.c:668-688: the conductance is 1/aero_resist[0]
                    # on a tile without an overstory and 1/aero_resist[1] on
                    # one with, and OUT_AERO_RESIST is the reciprocal of the
                    # cover-weighted sum. Using [0] everywhere reported 39.2
                    # s/m at FindleyLake against VIC's 28.9.
                    ra_rep = eb['aero_resist']
                    # Where no canopy ran this step anywhere in the batch,
                    # ra_veg1_t was never formed; an overstory tile then
                    # reports CalcAerodynamic's own aero_resist[1] (vic_run.c
                    # 5.1.0:249-250 loads cell.aero_resist from it and only the
                    # canopy's snow balance overwrites it). It reported the
                    # understory's Ra_used[0]: OUT_AERO_COND 0.0191 against
                    # VIC's 0.0195 from step 1 on the snow-free forest window.
                    if ra_veg1_t is None and ov_step is not None:
                        ra_veg1_t = ov_step['ra'][..., 1]
                    if ra_veg1_t is not None:
                        ra_rep = torch.where(over_t, ra_veg1_t, ra_rep)
                    cond = _tsum(cv / torch.clamp(ra_rep, min=1e-12))
                    if _lake_tap is not None:
                        _lake_tap['ra'] = ra_rep
                    # calc_surf_energy_bal.c:628-630: with a pack over the soil
                    # that solved its own balance, the *pack's* latent,
                    # latent_sub and sensible heat are added to the surface's
                    # before anything reports them. The surface's own share is
                    # scaled by the snow-free fraction (func_surf_energy_bal.c
                    # :838, :845-851), which under a full pack is zero -- so
                    # leaving the pack's out reports nothing at all for a
                    # snow-covered tile. On 181 days of FindleyLake that put
                    # OUT_LATENT at 0.24 W/m2 against VIC's 6.89 and OUT_SENSIBLE
                    # at -8.02 against +12.88.
                    #
                    # :620-624 is the other branch and adds nothing: a pack thin
                    # enough to be folded into this solve is already in `eb`.
                    # `latent`/`sensible` are reported negated (energy.surface_step),
                    # so the C's `+=` is a `-=` here, which is the same sign
                    # convention snow_eb['latent_sub'] already uses.
                    pack_w = cover * (~thin_c).to(dtype)
                    eb = dict(eb)
                    # calc_surf_energy_bal.c:617-618:
                    #   energy->Tsurf = snow->coverage * snow->surf_temp
                    #                 + (1 - snow->coverage) * Tsurf
                    # so the temperature a covered tile reports is the
                    # *pack's* surface, not the soil's underneath it. Reporting
                    # the soil's put OUT_SURF_TEMP at 1.88 C against VIC's
                    # 0.37 over a FindleyLake half-year, and through
                    # put_data.c:841-844 the same error is in OUT_RAD_TEMP.
                    # With SPATIAL_SNOW the weight is the fractional cover the
                    # pack solve left, not a 0/1 flag: a tile at 0.04 cover
                    # reports its warm soil, not its 0 C pack.
                    _cov_rep_in = cov_new if spatial_snow else has_p.to(dtype)
                    cov_rep = (pt(_cov_rep_in) if per_tile_snow
                               else tile(_cov_rep_in))
                    # A thin pack (INCLUDE_SNOW) has not been given its new
                    # temperature when :638 runs: surface_fluxes.c 5.1.0:594 put
                    # back the step's opening snow->surf_temp, and
                    # calc_surf_energy_bal.c sets the new one only below (:682).
                    _ts_end = torch.where(thin_c, old_snow_surf_temp,
                                          snow_state['surf_temp'])
                    ts_snow_rep = (pt(_ts_end) if per_tile_snow
                                   else tile(_ts_end))
                    eb['t_surf'] = (cov_rep * ts_snow_rep
                                    + (1.0 - cov_rep) * eb['t_surf'])
                    eb['latent'] = eb['latent'] - se['latent'] * pack_w
                    eb['sensible'] = eb['sensible'] - se['sensible'] * pack_w
                    if _sub['tcanopy'] is not None:
                        # CLOSE_ENERGY: this pass's atmospheric balance and the
                        # outputs the iteration driver reads back. InSensible is
                        # the canopy's sensible heat plus the ground's
                        # (calc_atmos_energy_bal.c:54-55), both internal signs:
                        # eb['sensible'] is still the reported ground share here.
                        _ov_here = (over_t if over_t is not None
                                    else torch.zeros_like(eb['sensible'],
                                                          dtype=torch.bool))
                        _snow_snow = ((wb_swq0 > 0.0) | tile(wb_sky_snow > 0.0)
                                      | ((_ov_here & (wb_int_snow0 > 0.0))
                                         if wb_int_snow0 is not None
                                         else torch.zeros_like(_ov_here)))
                        _gate = _ov_here & _snow_snow
                        _ins = -eb['sensible']
                        if canopy_sen is not None:
                            _ins = _ins + canopy_sen
                        _ra_atm = ra_veg1_t
                        if _ra_atm is None:
                            if ov_step is not None:
                                _ra_atm = ov_step['ra'][..., 1]
                            else:
                                # No overstory anywhere: the gate below is
                                # empty and the solve's values are discarded;
                                # a finite stand-in keeps it defined.
                                _ra_atm = torch.ones_like(_ins)
                        _tc_sol, _sh_sol, _tc_fail = energy.solve_canopy_air_temp(
                            _ins, _ra_atm, tile(atm['tair'][_t1]),
                            tile(atm['air_dens'][_t1]))
                        # calc_atmos_energy_bal.c:87
                        _tfx.note('canopy air', _tc_fail & _gate)
                        _sub['ce_pass'] = dict(
                            snow_flux=eb['snow_flux'], thin=thin_c, has_p=has_p,
                            dcov=dcov_t, gate=_gate, snow_snow=_snow_snow,
                            pack_branch=((wb_swq0 > 0.0) | (snow_p > 0.0)),
                            tcanopy=_tc_sol, atmos_sensible=_sh_sol,
                            seed=torch.stack(
                                (eb['snow_flux'], -eb['grnd_flux'],
                                 -eb['delta_h'], eb['fusion']), dim=-1))
                    # calc_atmos_energy_bal.c:64-80: the atmosphere sees the
                    # canopy's latent and sensible heat added to the ground's
                    # OUT_LATENT_BAND and OUT_SENSIBLE_BAND (put_data.c:915-922)
                    # are energy.latent and energy.sensible, the ground's
                    # (and pack's) own, before the canopy's join them
                    if 'lat' in band_rows:
                        band_rows['lat'].append(_tsum(eb['latent'] * mosaic.cv))
                    if 'sen' in band_rows:
                        band_rows['sen'].append(_tsum(eb['sensible'] * mosaic.cv))
                    if canopy_lat is not None:
                        eb['latent'] = eb['latent'] - canopy_lat   # VIC sign: OUT_LATENT -= AtmosLatent
                        eb['sensible'] = eb['sensible'] - canopy_sen
                        # ...but AtmosSensible is not that sum where the canopy
                        # ran: calc_atmos_energy_bal.c 5.1.0:103-104 re-evaluates
                        # it at the returned Tcanopy, which without CLOSE_ENERGY
                        # is Tair (:98-100), so func_atmos_energy_bal.c 5.1.0:52
                        # leaves calc_sensible_heat(Tair, Tair) = 0 and
                        # put_data.c 5.1.0:899 reports 0 for the tile.
                        if _sub['ce_final']:
                            # CLOSE_ENERGY: the closed sensible heat at the
                            # solved canopy air temperature, on the tiles the
                            # atmosphere balance ran on (surface_fluxes.c
                            # :637-660, the overstory with snow).
                            eb['sensible'] = torch.where(
                                _sub['ce_pass']['gate'],
                                -_sub['ce_pass']['atmos_sensible'],
                                eb['sensible'])
                        else:
                            eb['sensible'] = torch.where(
                                over_t & canopy_ran,
                                torch.zeros_like(eb['sensible']),
                                eb['sensible'])
                    # the ground's emission for the next step's canopy,
                    # LongUnderIn - NetLongUnder (calc_surf_energy_bal.c 5.1.0
                    # :612): the surface's at Tsurf, or with a pack snow_melt
                    # solved the pack's at its own surface temperature
                    # (SnowPackEnergyBalance.c 5.1.0:176-177)
                    ground_long_out = eb['long_out']
                    if per_tile_snow:
                        _endp = ((snow_state['ice'] + snow_state['surf_water']
                                  + snow_state['pack_water']) > 0.0) & ~thin_c
                        # LongUnderIn - NetLongSnow (calc_surf_energy_bal.c
                        # 5.1.0:612, :605): the emission at the temperature of
                        # snow_melt's last balance call, before the layers are
                        # rebalanced (snow_melt.c 5.1.0:433-456 moves surf_temp)
                        _pack_long_out = (f['long_snow_in'] - se['net_long'])
                        if spatial_snow:
                            # :612 is LongUnderIn - NetLongUnder, and NetLongUnder
                            # is NetLongBare + NetLongSnow with each side already
                            # carrying its own fraction -- (1 - coverage) of the
                            # soil's emission at Tsurf and coverage of the pack's
                            # -- so the ground the next step's canopy sees is the
                            # cover-weighted mix, not all pack or all soil. A
                            # 0/1 switch here swings the canopy's incoming
                            # longwave by the whole soil-pack difference on every
                            # fractional tile.
                            # In the C's own association: NetLongBare =
                            # (1 - c) LongUnderIn - (1 - c) LongBareOut
                            # (calc_surf_energy_bal.c:214, func_surf_energy_bal.c
                            # :636), NetLongSnow *= c (solve_snow.c:442), and
                            # LongUnderOut = LongUnderIn - (NetLongBare +
                            # NetLongSnow) (calc_surf_energy_bal.c:605, :612);
                            # the canopy's root moves with its last bit.
                            # A thin pack folded into the surface has
                            # NetLongSnow = c LongUnderIn - c LongBareOut
                            # (calc_surf_energy_bal.c:218, func_surf_energy_bal.c
                            # :634) in the same sum.
                            _lw = f['long_snow_in']
                            _nlb = ((1.0 - cov_new) * _lw
                                    - (1.0 - cov_new) * eb['long_out'])
                            _nls = torch.where(
                                thin_c, cov_new * _lw - cov_new * eb['long_out'],
                                se['net_long'] * cov_new)
                            ground_long_out = _lw - (_nlb + _nls)
                        else:
                            ground_long_out = torch.where(
                                _endp, _pack_long_out, ground_long_out)
                    return dict(_cov_rep_in=_cov_rep_in, _endp=_endp, _gate=_gate, _ins=_ins, _lw=_lw, _nlb=_nlb, _nls=_nls, _ov_here=_ov_here, _pack_long_out=_pack_long_out, _ra_atm=_ra_atm, _sh_sol=_sh_sol, _snow_snow=_snow_snow, _tc_fail=_tc_fail, _tc_sol=_tc_sol, _ts_end=_ts_end, canopy_evap=canopy_evap, ce=ce, cond=cond, cov_rep=cov_rep, eb=eb, evap=evap, ground_long_out=ground_long_out, pack_nl=pack_nl, pack_nl_t=pack_nl_t, pack_ns=pack_ns, pack_ns_t=pack_ns_t, pack_w=pack_w, ra_rep=ra_rep, ra_veg1_t=ra_veg1_t, throughfall=throughfall, ts_snow_rep=ts_snow_rep, wdew=wdew)
                _blk_res = _fusion.run(_eb_fluxes, *_bound(locals(), ('ce', 'wdew', 'ra_veg1_t', 'eb', 'ground_long_out', )))
                if _blk_res['_cov_rep_in'] is not _UNSET:
                    _cov_rep_in = _blk_res['_cov_rep_in']
                if _blk_res['_endp'] is not _UNSET:
                    _endp = _blk_res['_endp']
                if _blk_res['_gate'] is not _UNSET:
                    _gate = _blk_res['_gate']
                if _blk_res['_ins'] is not _UNSET:
                    _ins = _blk_res['_ins']
                if _blk_res['_lw'] is not _UNSET:
                    _lw = _blk_res['_lw']
                if _blk_res['_nlb'] is not _UNSET:
                    _nlb = _blk_res['_nlb']
                if _blk_res['_nls'] is not _UNSET:
                    _nls = _blk_res['_nls']
                if _blk_res['_ov_here'] is not _UNSET:
                    _ov_here = _blk_res['_ov_here']
                if _blk_res['_pack_long_out'] is not _UNSET:
                    _pack_long_out = _blk_res['_pack_long_out']
                if _blk_res['_ra_atm'] is not _UNSET:
                    _ra_atm = _blk_res['_ra_atm']
                if _blk_res['_sh_sol'] is not _UNSET:
                    _sh_sol = _blk_res['_sh_sol']
                if _blk_res['_snow_snow'] is not _UNSET:
                    _snow_snow = _blk_res['_snow_snow']
                if _blk_res['_tc_fail'] is not _UNSET:
                    _tc_fail = _blk_res['_tc_fail']
                if _blk_res['_tc_sol'] is not _UNSET:
                    _tc_sol = _blk_res['_tc_sol']
                if _blk_res['_ts_end'] is not _UNSET:
                    _ts_end = _blk_res['_ts_end']
                if _blk_res['canopy_evap'] is not _UNSET:
                    canopy_evap = _blk_res['canopy_evap']
                ce = _blk_res['ce']
                if _blk_res['cond'] is not _UNSET:
                    cond = _blk_res['cond']
                if _blk_res['cov_rep'] is not _UNSET:
                    cov_rep = _blk_res['cov_rep']
                eb = _blk_res['eb']
                if _blk_res['evap'] is not _UNSET:
                    evap = _blk_res['evap']
                ground_long_out = _blk_res['ground_long_out']
                if _blk_res['pack_nl'] is not _UNSET:
                    pack_nl = _blk_res['pack_nl']
                if _blk_res['pack_nl_t'] is not _UNSET:
                    pack_nl_t = _blk_res['pack_nl_t']
                if _blk_res['pack_ns'] is not _UNSET:
                    pack_ns = _blk_res['pack_ns']
                if _blk_res['pack_ns_t'] is not _UNSET:
                    pack_ns_t = _blk_res['pack_ns_t']
                if _blk_res['pack_w'] is not _UNSET:
                    pack_w = _blk_res['pack_w']
                if _blk_res['ra_rep'] is not _UNSET:
                    ra_rep = _blk_res['ra_rep']
                ra_veg1_t = _blk_res['ra_veg1_t']
                if _blk_res['throughfall'] is not _UNSET:
                    throughfall = _blk_res['throughfall']
                if _blk_res['ts_snow_rep'] is not _UNSET:
                    ts_snow_rep = _blk_res['ts_snow_rep']
                wdew = _blk_res['wdew']
                if return_state:
                    # What VIC's state file holds that the loop does not
                    # carry (vic_store.c 5.1.0: STATE_ENERGY_SNOW_FLUX,
                    # STATE_ENERGY_LONGUNDEROUT): the last step's per-tile
                    # values, read by torchvic.io.state_file only.
                    _vic_aux['snow_flux'] = eb['snow_flux']
                    if spatial_snow:
                        # surface_fluxes.c 5.1.0:922-935, :1003: the stored
                        # flux is soil_energy.snow_flux weighted by
                        # coverage + delta_coverage (1 without the option)
                        _ce = tile(cov_eff)
                        _vic_aux['snow_flux'] = torch.where(
                            _ce > 0.0, eb['snow_flux'] * _ce,
                            eb['snow_flux'])
                    _vic_aux['long_under_out'] = ground_long_out
                    _vic_aux['tair_t'] = tile(atm['tair'][_t1])
                    _vic_aux['snow_on'] = snowing_all
                    # snow->snow (solve_snow.c:146-152) gates two stores the
                    # C leaves alone on a step without snow: the cold content
                    # snow_melt.c:478 left (stale once the pack is gone) and,
                    # on the bare-soil tile, the foliage temperature
                    # (solve_snow.c:146-220 never reaches it; :514 sets it to
                    # Tcanopy = Tair only when there is no snow).
                    _cc0 = _vic_aux.get('cold_content')
                    if _cc0 is None:
                        _cc0 = torch.zeros_like(snow_state['ice'])
                    # snow_melt (which sets coldcontent) runs only with a
                    # pack or snowfall reaching the ground (solve_snow.c:
                    # 264), not for canopy snow alone
                    _vic_aux['cold_content'] = torch.where(
                        tile(has_snow) | tile(wb_swq0 > 0.0),
                        snow_state['cold_content'], _cc0)
                    _tf0 = _vic_aux.get('tfoliage_bare')
                    if _tf0 is None:
                        _th_ = soil_params.thermal
                        _tf0 = (torch.as_tensor(
                            _th_['avg_temp'], dtype=dtype, device=device
                        ).unsqueeze(1).expand_as(snow_state['ice']).clone()
                            if (full_energy or frozen_soil)
                            else torch.zeros_like(snow_state['ice']))
                        if band_tfactor is not None:
                            # generate_default_state.c:106: + Tfactor[band]
                            _tf0 = _tf0 + torch.as_tensor(
                                band_tfactor, dtype=dtype, device=device
                            ).reshape(-1, 1)
                    _vic_aux['tfoliage_bare'] = torch.where(
                        snowing_all, _tf0, tile(atm['tair'][_t1]))
                # the step's energy rows, appended to their lists after the
                # fused block below (a list read inside one would be guarded)
                _rows_eb, _rows_ex = {}, {}
                # (core.fusion.run: this block one fused region on the graph path)
                def _eb_rows(k, v, soil_ice):
                    pns = pnl = ns_atm = nl_atm = pick = _live = _fv = _frozen = tl = _mx = _bb = _ex = ice_new = _sflux_t = t_fol = t_rad = t_k = _bt = _cvb = t_veg = cv_v = ns_pot = nl_pot = rc_pot = ra_pot = epot_veg = epot_soil = fc_pot = _dt_pot = _rows_pet = _UNSET
                    for k in eb_out:
                        v = eb[k]
                        if k in ('net_short', 'net_long', 'net_rad'):
                            pns = tile(pack_ns_t) if per_tile_snow else pack_ns
                            pnl = tile(pack_nl_t) if per_tile_snow else pack_nl
                            ns_atm = eb['net_short'] + pns
                            nl_atm = eb['net_long'] + pnl
                            if canopy_ran is not None:
                                ns_atm = ns_atm + torch.where(
                                    canopy_ran, canopy_ns, torch.zeros_like(ns_atm))
                                nl_atm = torch.where(canopy_ran, canopy_nl, nl_atm)
                            pick = {'net_short': ns_atm, 'net_long': nl_atm,
                                    'net_rad': ns_atm + nl_atm}[k]
                            _rows_eb[k] = (_tsum(pick * cv))
                            continue
                        if k == 'layer_temp':
                            _rows_eb[k] = (_tsum(v * cv.unsqueeze(-1)))
                        elif k == 'fallback':
                            # calc_surf_energy_bal.c:311: the tiles that ran
                            # (cover, and the sub-pass's own tiles)
                            _live = cv > 0.0
                            if _sub['tile_w'] is not None:
                                _live = _live & _sub['tile_w']
                            _tfx.note('surface', v & _live)
                            _fv = v.to(dtype)
                            if _sub['tile_w'] is not None:
                                _fv = _fv * _sub['tile_w'].to(dtype)
                            _rows_eb[k] = (_tsum(_fv))
                        elif k == 'aero_resist':
                            _rows_eb[k] = (1.0 / torch.clamp(cond, min=1e-30))
                        else:
                            _rows_eb[k] = (_tsum(v * cv))
                    if frozen_soil:
                        # soil_conduction.c:659-688: the ice per frost sub-area
                        # (one line per layer with Nfrost 1), against the
                        # entering moisture. frozen_soil.c:75-81 takes the
                        # temperature from
                        # estimate_layer_temperature_quick_flux, which is what
                        # eb['layer_temp'] already is. On the finite-difference
                        # path the surface step already integrated the ice over
                        # the profile (estimate_layer_ice_content), active-gated
                        # and clamped from below only, as the C is.
                        if eb.get('layer_ice') is not None:
                            soil_ice = eb['layer_ice']
                        else:
                            from ..core import frozen as _frozen
                            tl = eb['layer_temp']
                            _mx = soil_params.max_moist.unsqueeze(1).expand_as(tl)
                            _bb = soil_params.bubble.unsqueeze(1).expand_as(tl)
                            _ex = soil_params.expt.unsqueeze(1).expand_as(tl)
                            ice_new = _frozen.layer_ice_quick_flux(
                                moist, _mx, _bb, _ex, tl,
                                None if _frost is None else _frost['slope'],
                                n_frost=1 if _frost is None else _frost['n'],
                                fs_active=getattr(soil_params, 'fs_active', None))
                            soil_ice = (ice_new if _frost is not None
                                        else ice_new[..., 0])
                    _rows_ex['aero_cond'] = (cond)
                    if frozen_soil:
                        # put_data.c:928 reports OUT_FUSION negated, the same
                        # convention as the latent and sensible heat.
                        _rows_ex['fusion'] = (_tsum(-eb['fusion'] * cv))
                    # put_data.c:945: OUT_SNOW_FLUX is the conduction between
                    # the pack and the soil surface, cover-weighted over tiles.
                    # surface_fluxes.c:928-930 weights each tile's flux by its
                    # (coverage + delta_coverage) too: the strip the cover
                    # uncovered during the step conducted all step. Without the
                    # option that weight is 1 wherever the flux is nonzero.
                    _sflux_t = eb['snow_flux']
                    if spatial_snow:
                        _sflux_t = _sflux_t * tile(cov_eff)
                    _rows_ex['snow_flux'] = (_tsum(_sflux_t * cv))
                    # put_data.c:932-933 and :456: the radiative temperature is
                    # the fourth root of the cover-weighted fourth power of the
                    # tile skin temperature in Kelvin. Without an overstory every
                    # tile's rad_temp is its Tsurf (:844).
                    # put_data.c 5.1.0:820-825 (5.0.1:840-845): the radiative
                    # temperature is Tfoliage on an overstory tile *with* snow
                    # (snow.snow, solve_snow.c 5.1.0:146-152: a pack, snow
                    # falling or snow in the canopy -- `snowing_all`), Tsurf
                    # everywhere else. This had the condition inverted, taking
                    # Tfoliage (Tair, solve_snow.c 5.1.0:499) on every snow-free
                    # overstory tile: OUT_RAD_TEMP 0.03-0.35 K off from step 1 on
                    # the forest and tntip windows.
                    t_fol = (tile(atm['tair'][_t1]) if vegt_canopy is None
                             else vegt_canopy)
                    t_rad = torch.where(over_t & snowing_all, t_fol,
                                        eb['t_surf'])
                    t_k = t_rad + CONST_TKFRZ
                    if mosaic.bare_tile is not None:
                        # put_data.c 5.1.0:156-161, :190-195, :424-426, :832-834:
                        # OUT_BARESOILT is the bare-soil tile's rad_temp (its
                        # Tsurf: it has no overstory) over that tile's cover,
                        # 0 where it has none.
                        _bt = mosaic.bare_tile.to(dtype).expand_as(cv) * cv
                        _cvb = _tsum(_bt)
                        _rows_ex['baresoilt'] = (torch.where(
                            _cvb > 0.0, _tsum(t_rad * _bt)
                            / torch.where(_cvb > 0.0, _cvb, torch.ones_like(_cvb)),
                            torch.zeros_like(_cvb)))
                    _rows_ex['rad_temp'] = (
                        _tsum((t_k * t_k * t_k * t_k) * cv) ** 0.25)
                    # put_data.c 5.1.0:836-844 and :443-445: OUT_VEGT is
                    # Tfoliage on an overstory tile without snow and rad_temp
                    # elsewhere -- Tfoliage on every overstory tile -- over the
                    # vegetated tiles divided by their cover. Every tile of the
                    # mosaic is vegetated, so the divisor is the coverage.
                    t_veg = torch.where(over_t, t_fol, eb['t_surf'])
                    # put_data.c 5.1.0:190-195, :427-429: the bare tile is not
                    # vegetation (HasVeg false) and is in neither sum
                    cv_v = (cv if mosaic.bare_tile is None
                            else torch.where(mosaic.bare_tile.expand_as(cv),
                                             torch.zeros_like(cv), cv))
                    _rows_ex['vegt'] = (_tsum(t_veg * cv_v)
                                            / torch.clamp(_tsum(cv_v), min=1e-12))
                    if _lake_tap is not None:
                        _lake_tap['t_veg'] = t_veg
                        _lake_tap['veg_tile'] = cv_v > 0.0
                    # compute_pot_evap.c, called at surface_fluxes.c:797 with the
                    # library rmin, RGL and rarc, the tile albedo, the incoming
                    # shortwave, the solved NetLongAtmos, the forcing Tair and
                    # VPD (:385, :394) and iter_aero_resist_veg[0]. That array is
                    # loaded with the *neutral* resistance at :539, but it is the
                    # same array func_surf_energy_bal.c receives as Ra_veg and
                    # overwrites in place at :675-684 -- Ra_veg[0] = ra /
                    # StabilityCorrection(...); Ra_used[0] = Ra_veg[0] -- so by the
                    # PET call it holds the stability-corrected value, which is
                    # what surface_step returns as aero_resist. Handing it the
                    # neutral resistance instead put PET 14 per cent high over ten
                    # January days and 1.6x high at midday on the calm ones
                    # (wind 0.76 m/s), where rc >> ra makes Penman grow with ra;
                    # rebuilt from VIC's own step outputs at one such step, the
                    # neutral resistance gave 0.307 mm and the corrected 0.175
                    # against VIC's 0.189. calc_rc runs unstressed, gsm_inv = 1.
                    # fcanopy is 1 -- the mosaic carries none, as the fcanopy
                    # output already says -- so the soil-side Penman that
                    # (1 - fcanopy) would weight contributes nothing and is not
                    # formed. penman is mm/day; :43 divides by
                    # model_steps_per_day.
                    # :63-65: net_short is the tile's own albedo against the
                    # *incoming* shortwave, not the coverage-weighted bare
                    # surface's, and the longwave is NetLongAtmos.
                    ns_pot = tile(atm['shortwave'][_t1]) * (1.0 - alb_t)
                    # surface_fluxes.c:788-791 hands compute_pot_evap
                    # NetLongAtmos, which without an overstory is NetLongUnder
                    # (:676-679): the bare ground's plus the pack's
                    # (calc_surf_energy_bal.c:606). The bare ground's alone is 0
                    # under a full pack, which is what a sub-step with no canopy
                    # anywhere passed here.
                    nl_pot = nl_atm
                    rc_pot = vegetation.calc_rc(
                        mosaic.rmin * vegetation.cellwise(p['rmin_scale'], mosaic.rmin), ns_pot, mosaic.rgl,
                        tile(atm['tair'][_t1]), tile(atm['vpd'][_t1]), lai_t,
                        torch.ones_like(lai_t))
                    # :68-73: an overstory tile's vegetated Penman uses
                    # aero_resist[1], everything else aero_resist[0].
                    # (compute_pot_evap gets iter_aero_resist_veg, whose [0] is
                    # Ra_veg[0] before the fcanopy average, and Ra_used[0] for
                    # the soil: the same number unless fcanopy < 1 under 5.1.0.)
                    ra_pot = eb['aero_resist_veg']
                    if ra_veg1_t is not None:
                        ra_pot = torch.where(over_t, ra_veg1_t, ra_pot)
                    epot_veg = vegetation.penman(
                        tile(atm['tair'][_t1]), tile(batch.elevation),
                        ns_pot + nl_pot, tile(atm['vpd'][_t1]),
                        ra_pot, rc_pot, mosaic.rarc)
                    # :79-84: the bare fraction of the tile evaporates as
                    # soil -- no canopy resistance, SOIL_RARC, and the
                    # surface's own aerodynamic resistance. With fcanopy
                    # below 1 this is most of the potential evaporation, and
                    # it was not formed at all: PET was 50.8 mm against
                    # VIC's 227.4 over half a year at FindleyLake.
                    epot_soil = vegetation.penman(
                        tile(atm['tair'][_t1]), tile(batch.elevation),
                        ns_pot + nl_pot, tile(atm['vpd'][_t1]),
                        eb['aero_resist'], torch.zeros_like(rc_pot),
                        torch.full_like(rc_pot, SOIL_RARC))
                    fc_pot = (torch.ones_like(alb_t) if mosaic.fcanopy is None
                              else mosaic.fcanopy[:, :, m])
                    epot_veg = fc_pot * epot_veg + (1.0 - fc_pot) * epot_soil
                    # compute_pot_evap.c:58-62 divides by model_steps_per_day
                    # on a snow sub-step too, and surface_fluxes.c:934 sums
                    # the sub-steps
                    _dt_pot = (_sub['model_dt'] if _sub['pin_tgrnd']
                               else step_seconds)
                    _rows_pet = (_tsum(epot_veg * cv)
                                * (_dt_pot / vegetation.CONST_CDAY))
                    return dict(_bb=_bb, _bt=_bt, _cvb=_cvb, _dt_pot=_dt_pot, _ex=_ex, _frozen=_frozen, _fv=_fv, _live=_live, _mx=_mx, _rows_pet=_rows_pet, _sflux_t=_sflux_t, cv_v=cv_v, epot_soil=epot_soil, epot_veg=epot_veg, fc_pot=fc_pot, ice_new=ice_new, k=k, nl_atm=nl_atm, nl_pot=nl_pot, ns_atm=ns_atm, ns_pot=ns_pot, pick=pick, pnl=pnl, pns=pns, ra_pot=ra_pot, rc_pot=rc_pot, soil_ice=soil_ice, t_fol=t_fol, t_k=t_k, t_rad=t_rad, t_veg=t_veg, tl=tl, v=v)
                _blk_res = _fusion.run(_eb_rows, *_bound(locals(), ('k', 'v', 'soil_ice', )))
                if _blk_res['_bb'] is not _UNSET:
                    _bb = _blk_res['_bb']
                if _blk_res['_bt'] is not _UNSET:
                    _bt = _blk_res['_bt']
                if _blk_res['_cvb'] is not _UNSET:
                    _cvb = _blk_res['_cvb']
                if _blk_res['_dt_pot'] is not _UNSET:
                    _dt_pot = _blk_res['_dt_pot']
                if _blk_res['_ex'] is not _UNSET:
                    _ex = _blk_res['_ex']
                if _blk_res['_frozen'] is not _UNSET:
                    _frozen = _blk_res['_frozen']
                if _blk_res['_fv'] is not _UNSET:
                    _fv = _blk_res['_fv']
                if _blk_res['_live'] is not _UNSET:
                    _live = _blk_res['_live']
                if _blk_res['_mx'] is not _UNSET:
                    _mx = _blk_res['_mx']
                if _blk_res['_rows_pet'] is not _UNSET:
                    _rows_pet = _blk_res['_rows_pet']
                if _blk_res['_sflux_t'] is not _UNSET:
                    _sflux_t = _blk_res['_sflux_t']
                if _blk_res['cv_v'] is not _UNSET:
                    cv_v = _blk_res['cv_v']
                if _blk_res['epot_soil'] is not _UNSET:
                    epot_soil = _blk_res['epot_soil']
                if _blk_res['epot_veg'] is not _UNSET:
                    epot_veg = _blk_res['epot_veg']
                if _blk_res['fc_pot'] is not _UNSET:
                    fc_pot = _blk_res['fc_pot']
                if _blk_res['ice_new'] is not _UNSET:
                    ice_new = _blk_res['ice_new']
                k = _blk_res['k']
                if _blk_res['nl_atm'] is not _UNSET:
                    nl_atm = _blk_res['nl_atm']
                if _blk_res['nl_pot'] is not _UNSET:
                    nl_pot = _blk_res['nl_pot']
                if _blk_res['ns_atm'] is not _UNSET:
                    ns_atm = _blk_res['ns_atm']
                if _blk_res['ns_pot'] is not _UNSET:
                    ns_pot = _blk_res['ns_pot']
                if _blk_res['pick'] is not _UNSET:
                    pick = _blk_res['pick']
                if _blk_res['pnl'] is not _UNSET:
                    pnl = _blk_res['pnl']
                if _blk_res['pns'] is not _UNSET:
                    pns = _blk_res['pns']
                if _blk_res['ra_pot'] is not _UNSET:
                    ra_pot = _blk_res['ra_pot']
                if _blk_res['rc_pot'] is not _UNSET:
                    rc_pot = _blk_res['rc_pot']
                soil_ice = _blk_res['soil_ice']
                if _blk_res['t_fol'] is not _UNSET:
                    t_fol = _blk_res['t_fol']
                if _blk_res['t_k'] is not _UNSET:
                    t_k = _blk_res['t_k']
                if _blk_res['t_rad'] is not _UNSET:
                    t_rad = _blk_res['t_rad']
                if _blk_res['t_veg'] is not _UNSET:
                    t_veg = _blk_res['t_veg']
                if _blk_res['tl'] is not _UNSET:
                    tl = _blk_res['tl']
                v = _blk_res['v']
                for k_, v_ in _rows_eb.items():
                    eb_out[k_].append(v_)
                for k_, v_ in _rows_ex.items():
                    eb_extra[k_].append(v_)
                pets.append(_rows_pet)
                # out-vars (put_data.c 5.1.0:653-672, :828): OUT_AERO_COND1 is
                # the sum over tiles of (1 / aero_resist[0]) * AreaFactor,
                # OUT_AERO_COND2 the same over overstory tiles with
                # aero_resist[1] (normalised by their cover in the tail), a
                # resistance at or below DBL_EPSILON counting as HUGE_RESIST;
                # OUT_SOIL_TNODE the cover-weighted node temperatures the
                # step left. Formed outside the fused regions, so no row that
                # existed is touched.
                if _w_aero:
                    _lv_t = cv > 0.0
                    _ra0_t = eb['aero_resist']
                    eb_extra['aero_cond1'].append(_tsum(torch.where(
                        _lv_t, torch.where(_ra0_t > DBL_EPSILON,
                                           (1.0 / _ra0_t) * cv,
                                           torch.full_like(cv, vegetation.HUGE_RESIST)),
                        torch.zeros_like(cv))))
                    if over_t is not None and ra_veg1_t is not None:
                        _ra1_t = ra_veg1_t
                        eb_extra['aero_cond2'].append(_tsum(torch.where(
                            over_t & _lv_t, torch.where(
                                _ra1_t > DBL_EPSILON, (1.0 / _ra1_t) * cv,
                                torch.full_like(cv, vegetation.HUGE_RESIST)),
                            torch.zeros_like(cv))))
                    else:
                        eb_extra['aero_cond2'].append(torch.zeros_like(
                            eb_extra['aero_cond1'][-1]))
                if _w_tnode:
                    eb_extra['tnode'].append(_tsum(t_node * cv.unsqueeze(-1)))
                if _lake_tap is not None:
                    _lake_tap['eb'] = eb
                    _lake_tap['t_rad_k'] = t_k
                    _lake_tap['t_node'] = t_node
                    _lake_tap['pet'] = epot_veg * (_dt_pot
                                                   / vegetation.CONST_CDAY)
            else:
                # ---- VIC's water-balance mode, per tile ------------------
                m = months[_t1]
                ntl = mosaic.n_tiles
                cv = mosaic.cv
                _wbq = None
                tile2 = (lambda x: x.unsqueeze(1).expand(nc, ntl)
                         if x.dim() == 1 else x)
                # (core.fusion.run: this block one fused region on the graph path)
                def _wb_pre(wb_t0, wb_t1):
                    _m0 = _wbq = atten_one = bare_soil = fc_m = ground_heat = ra_bare = ra_over = rough = snowing = tair_t = th = _UNSET
                    tair_t = tile2(atm['tair'][_t1])
                    # solve_snow.c:164-171, snow->snow: a pack at the start of
                    # the step, snow falling from the sky, or snow held by an
                    # overstory. func_surf_energy_bal.c:756/:798/:807-809 then
                    # skip every evaporation on the tile.
                    snowing = (has_snow | (wb_swq0 > 0.0)
                               | tile2(wb_sky_snow > 0.0))
                    if any_over:
                        snowing = snowing | (over_t & (wb_int_snow0 > 0.0))
                    fc_m = (None if mosaic.fcanopy is None
                            else mosaic.fcanopy[:, :, m])
                    th = soil_params.thermal
                    ground_heat = None
                    # solve_snow.c:154-156 sets surf_atten = 1 on a tile with no
                    # overstory while snow is present, and surf_atten is
                    # surface_fluxes' own argument (:19, :558): nothing resets it
                    # for the later snow steps of the same model step.
                    atten_one = snowing if over_t is None else snowing & ~over_t
                    if _sub['pin_tgrnd']:
                        if _sub['atten_one'] is not None:
                            atten_one = atten_one | _sub['atten_one']
                        _sub['atten_one'] = atten_one
                    if wb_quick_flux:
                        # prepare_full_energy.c:95-103, run every step whatever
                        # the mode (vic_run.c:228): the top two layers' kappa
                        # and Cs from each tile's own moisture on entry.
                        two = (lambda x: x.unsqueeze(1).expand(nc, ntl, 2)
                               .reshape(-1, 2))
                        kap, cs = energy.thermal_properties(
                            moist[:, :, :2].reshape(-1, 2),
                            two(th['depth'][:, :2]),
                            two(th['bulk_density'][:, :2]),
                            two(th['soil_density'][:, :2]),
                            two(th['quartz'][:, :2]),
                            **_mineral_kw(th, two),
                            # soil_conduction.c:711-716 builds the two upper
                            # layers' conductivity and heat capacity from the
                            # ice the previous step left, as on the energy
                            # path above (frost-weighted with SPATIAL_FROST).
                            ice_mm=(None if soil_ice is None
                                    else _ice_eff()[:, :, :2].reshape(-1, 2)))
                        kap1, kap2 = kap[:, 0].view(nc, ntl), kap[:, 1].view(nc, ntl)
                        cs1, cs2 = cs[:, 0].view(nc, ntl), cs[:, 1].view(nc, ntl)
                        # vic_run.c:222-225 from the per-plant LAI; solve_snow.c
                        # :175-177 sets it to 1 on a tile without an overstory
                        # while snow is present. The C's ground-flux factor is
                        # (coverage + (1 - coverage) surf_atten), coverage the
                        # pack the step leaves (0 or 1, SPATIAL_SNOW off).
                        lai_q = (mosaic.lai[:, :, m]
                                 * vegetation.cellwise(p['lai_scale'], cv))
                        fc_q = torch.ones_like(lai_q) if fc_m is None else fc_m
                        rad_att = (mosaic.rad_atten
                                   if mosaic.rad_atten is not None
                                   else torch.full_like(lai_q, 0.5))
                        if core_version.at_least('5.1.0'):
                            # vic_run.c 5.1.0:219-226: divided only where
                            # fcanopy > 0 (a leafless tile has 0 and LAI 0),
                            # and on the bare-soil tile too; surf_atten is then
                            # per band (:232), which is one value here.
                            lai_d = lai_q / torch.where(fc_q > 0.0, fc_q,
                                                        torch.ones_like(fc_q))
                        else:
                            lai_d = lai_q / fc_q
                        if (mosaic.bare_tile is not None
                                and not core_version.at_least('5.1.0')):
                            # vic_run.c:133 divides LAI by fcanopy for iveg <
                            # Nveg only; the bare-soil tile keeps its own.
                            lai_d = torch.where(mosaic.bare_tile, lai_q, lai_d)
                        atten = energy.surface_attenuation(lai_d, rad_att, fc_q)
                        atten = torch.where(atten_one, torch.ones_like(atten), atten)
                        cov_end = has_p.to(dtype)
                        if spatial_snow:
                            # with SPATIAL_SNOW the factor's coverage is
                            # snow_coverage, iter_snow.coverage as solve_snow.c
                            # :376 left it (surface_fluxes.c 5.1.0:606,
                            # func_surf_energy_bal.c 5.1.0:384-398)
                            cov_end = torch.where(has_p, cov_new,
                                                  torch.zeros_like(cov_new))
                        grnd, d_h, t1_new = energy.water_balance_ground_flux(
                            tair_t, wb_t0, wb_t1, tile2(th['avg_temp']),
                            tile2(th['depth'][:, 0]), kap1, kap2, cs1, cs2,
                            tile2(th['dp']), step_seconds,
                            cov_end + (1.0 - cov_end) * atten,
                            grnd_flux_type=grnd_flux_type)
                        fusion = torch.zeros_like(grnd)
                        if frozen_soil:
                            # func_surf_energy_bal.c:535-548 (!EXP_TRANS) with
                            # TMean = Tair: the latent heat of the top layer's
                            # ice change over the step, which joins NetBareRad
                            # (:663-664) in water-balance mode too. ice0 is
                            # prepare_full_energy.c:35-53, at the mean of the
                            # two OLD node temperatures; the new ice at the
                            # mean of Tair and the T1 just formed. Volume
                            # fractions of the top layer, as fz_in above.
                            from ..core import frozen as _frz
                            _d1 = tile2(th['depth'][:, 0])
                            _mm = _d1 * MM_PER_M
                            _m0 = moist[:, :, 0] / _mm
                            _mx0 = tile2(soil_params.max_moist[:, 0]) / _mm
                            _bb0 = tile2(soil_params.bubble[:, 0])
                            _ex0 = tile2(soil_params.expt[:, 0])
                            _i0 = torch.clamp(
                                _m0 - _frz.maximum_unfrozen_water(
                                    0.5 * (wb_t0 + wb_t1), _mx0, _bb0, _ex0),
                                min=0.0)
                            _inow = torch.clamp(
                                _m0 - _frz.maximum_unfrozen_water(
                                    0.5 * (tair_t + t1_new), _mx0, _bb0, _ex0),
                                min=0.0)
                            fusion = (-(snow.CONST_RHOICE * snow.CONST_LATICE
                                        * (_i0 - _inow) * _d1 / step_seconds))
                            _fs = getattr(soil_params, 'fs_active', None)
                            if _fs is not None:
                                while _fs.dim() < fusion.dim():
                                    _fs = _fs.unsqueeze(-1)
                                fusion = torch.where(
                                    _fs.expand_as(fusion), fusion,
                                    torch.zeros_like(fusion))
                        ground_heat = grnd + d_h + fusion
                        wb_t1 = t1_new
                        # put_data.c:922, :925, :928 report all three negated
                        # (appended after the fused block below)
                        _wbq = (-_tsum(grnd * cv), -_tsum(d_h * cv),
                                -_tsum(fusion * cv))
                    wb_t0 = tair_t
                    ra_over = (ov_step['ra'][..., 1]
                               if any_over and ov_step is not None else None)
                    bare_soil = None
                    ra_bare = None
                    if fc_m is not None:
                        # func_surf_energy_bal.c:703-745 at Tsurf = Tair, where
                        # StabilityCorrection is 1: Ra_used[0] is the conductance
                        # average of the understory's Ra_veg[0] and a bare-soil
                        # resistance from the soil roughness at the understory's
                        # own wind, CalcAerodynamic's no-overstory form with
                        # d = 0.67 * rough / 0.123.
                        z0_b = mosaic.veg_rough[:, :, m]
                        rough = (th['rough'] if th is not None
                                 and th.get('rough') is not None else None)
                        if rough is not None:
                            z0_b = torch.where(z0_b > 0.0, z0_b,
                                               tile2(rough).expand_as(z0_b))
                        d_b = mosaic.displacement[:, :, m]
                        ra_veg0 = vegetation.aerodynamic_resistance(
                            z0_b, d_b, u.unsqueeze(1))
                        zref_b = energy.reference_height(mosaic.wind_h, d_b, z0_b)
                        u0 = (torch.log((2.0 + z0_b) / z0_b)
                              / torch.log((zref_b - d_b) / z0_b) * u.unsqueeze(1))
                        if any_over and ov_step is not None:
                            ra_veg0 = torch.where(over_t, ov_step['ra'][..., 0],
                                                  ra_veg0)
                            u0 = torch.where(over_t, ov_step['u'][..., 0], u0)
                        z0_s = tile2(rough if rough is not None
                                     else torch.full_like(atm['tair'][_t1],
                                                          SNOW_Z0))
                        d_s = energy.VEG_RATIO_DH_HEIGHT * (z0_s
                                                            / VEG_RATIO_RL_HEIGHT)
                        top_s = 2.0 + vegetation.DISPL_FACTOR * d_s
                        safe_u0 = torch.where(u0 > 0.0, u0, torch.ones_like(u0))
                        ra_bare = torch.where(
                            u0 > 0.0,
                            torch.log(top_s / z0_s) * torch.log(top_s / (0.1 * z0_s))
                            / VON_KARMAN_SQ / safe_u0,
                            torch.full_like(u0, vegetation.HUGE_RESIST))
                        # :703 averages only where fcanopy < 1 (a bare-soil
                        # tile with fcanopy 1 keeps Ra_veg[0] to the bit).
                        ra_used0 = torch.where(
                            fc_m < 1.0,
                            1.0 / (fc_m / ra_veg0 + (1.0 - fc_m) / ra_bare),
                            ra_veg0)
                        d0 = tile2(th['depth'][:, 0] if th is not None
                                   else soil_params.depth[:, 0])
                        bare_soil = dict(
                            # arno_evap.c 5.1.0:50-54: the liquid, moist - ice,
                            # as the energy path's bare_in (frost-weighted
                            # with SPATIAL_FROST).
                            moist=(moist[:, :, 0] if soil_ice is None
                                   else moist[:, :, 0] - _ice_eff()[:, :, 0]),
                            max_moist=tile2(soil_params.max_moist[:, 0]),
                            b_infilt=tile2(soil_params.b_infilt),
                            depth=d0,
                            resid_frac=(tile2(soil_params.resid_moist[:, 0])
                                        / (d0 * MM_PER_M)),
                            # VIC 5.1.0 passes the residual in mm as it is
                            # (func_surf_energy_bal.c 5.1.0:749-752)
                            resid_mm=(tile2(soil_params.resid_moist[:, 0])
                                      if core_version.at_least('5.1.0')
                                      else None),
                            ra=ra_used0)
                    return dict(_m0=_m0, _wbq=_wbq, atten_one=atten_one, bare_soil=bare_soil, fc_m=fc_m, ground_heat=ground_heat, ra_bare=ra_bare, ra_over=ra_over, rough=rough, snowing=snowing, tair_t=tair_t, th=th, wb_t0=wb_t0, wb_t1=wb_t1)
                _blk_res = _fusion.run(_wb_pre, wb_t0, wb_t1)
                if _blk_res['_m0'] is not _UNSET:
                    _m0 = _blk_res['_m0']
                if _blk_res['_wbq'] is not _UNSET:
                    _wbq = _blk_res['_wbq']
                if _blk_res['atten_one'] is not _UNSET:
                    atten_one = _blk_res['atten_one']
                if _blk_res['bare_soil'] is not _UNSET:
                    bare_soil = _blk_res['bare_soil']
                if _blk_res['fc_m'] is not _UNSET:
                    fc_m = _blk_res['fc_m']
                if _blk_res['ground_heat'] is not _UNSET:
                    ground_heat = _blk_res['ground_heat']
                if _blk_res['ra_bare'] is not _UNSET:
                    ra_bare = _blk_res['ra_bare']
                if _blk_res['ra_over'] is not _UNSET:
                    ra_over = _blk_res['ra_over']
                if _blk_res['rough'] is not _UNSET:
                    rough = _blk_res['rough']
                if _blk_res['snowing'] is not _UNSET:
                    snowing = _blk_res['snowing']
                if _blk_res['tair_t'] is not _UNSET:
                    tair_t = _blk_res['tair_t']
                if _blk_res['th'] is not _UNSET:
                    th = _blk_res['th']
                wb_t0 = _blk_res['wb_t0']
                wb_t1 = _blk_res['wb_t1']
                if _wbq is not None:
                    wb_grnd.append(_wbq[0])
                    wb_delta_h.append(_wbq[1])
                    wb_fusion.append(_wbq[2])
                thin_w = new_state.get('thin')
                thin_w = None if thin_w is None else thin_w & has_snow
                # graph-safe: taken always; with no thin pack every row it
                # rewrites is rewritten to the value it had
                if thin_w is not None and (graphsafe.ACTIVE
                                           or bool(thin_w.any())):
                    water_in = _wb_thin_pack(
                        thin_w, new_state, snow_state, old_snow_surf_temp,
                        albedo, tair_t, atm, t, tile2, ground_heat,
                        mosaic, m, u, th, over_t,
                        ov_step if any_over else None, fc_m, ra_bare,
                        step_seconds, water_in, swes, snow_depths,
                        snow_covers, sublimations, snow_eb, tile_rows,
                        to_cell, sub_tile, sw_under=under_short,
                        lw_under=under_long, melts=melts,
                        melt_tile=melt_tile,
                        spc=(dict(c=cov_new, d=delta_cov, sp=sp,
                                  cov_prev=cov_prev, slope=sp_slope,
                                  depth=sp_depth, prcp=tile2(atm['prcp'][_t1]),
                                  bare_alb=bare_alb, alb_under=alb_under,
                                  advection=se['advection'],
                                  adv_prev=_adv_prev)
                             if spatial_snow else None))
                    # put_data.c:727-731 reports the snow albedo and the two
                    # pack temperatures from the pack the step *ends* with,
                    # which is the one the thin-pack balance just amended: a
                    # trace pack it sublimated away (Walnut Gulch, January)
                    # has none to report, and one it kept has its surface at
                    # min(Tair, 0). These rows were appended before it ran.
                    end_p = (snow_state['ice'] + snow_state['surf_water']
                             + snow_state['pack_water']) > 0.0
                    if _cells_fused():
                        # the three pack-cover means in one fused region
                        _r3 = _cell_sums(
                            {}, dict(a=(albedo, end_p),
                                     s=(snow_state['surf_temp'], end_p),
                                     p=(snow_state['pack_temp'], end_p)),
                            mosaic.cv, lanes.cell_start, lanes.cell_mask,
                            lanes.n_tiles, lanes.lane_cell)
                        salbedos[-1] = _r3['a']
                        snow_surf_temps[-1] = _r3['s']
                        snow_pack_temps[-1] = _r3['p']
                    else:
                        salbedos[-1] = to_snow_cell(albedo, end_p)
                        snow_surf_temps[-1] = to_snow_cell(
                            snow_state['surf_temp'], end_p)
                        snow_pack_temps[-1] = to_snow_cell(
                            snow_state['pack_temp'], end_p)
                # OUT_SNOW_FLUX of the evaluation: func_surf_energy_bal.c
                # :353-366 conducts through the pack against the surface held
                # at Tair (zero for a thin pack folded into the surface), and
                # surface_fluxes.c:928-930 weights it by coverage +
                # delta_coverage (1 without SPATIAL_SNOW)
                _wb_dep = snow_state.get('depth')
                if _wb_dep is None:
                    _wb_kap = torch.zeros_like(tair_t)
                else:
                    _wb_dm = _wb_dep
                    if st.get('depth') is not None:
                        _wb_dm = 0.5 * (st['depth'] + _wb_dep)
                    _wb_sd = torch.where(_wb_dep > 0.0, _wb_dm,
                                         torch.ones_like(_wb_dep))
                    _wb_kap = torch.where(
                        _wb_dep > 0.0,
                        snow.SNOW_CONDUCT * new_state['density']
                        * new_state['density'] / _wb_sd,
                        torch.zeros_like(_wb_dep))
                _wb_thin = (new_state['thin'] & has_snow
                            if new_state.get('thin') is not None
                            else torch.zeros_like(has_snow))
                _wb_sflux = torch.where(
                    ((cov_new if spatial_snow else
                      ((snow_state['ice'] + snow_state['surf_water']
                        + snow_state['pack_water']) > 0.0).to(dtype)) > 0.0)
                    & ~_wb_thin,
                    _wb_kap * (new_state['surf_temp'] - tair_t),
                    torch.zeros_like(tair_t))
                if spatial_snow:
                    _wb_sflux = _wb_sflux * cov_eff
                if _sub['pin_tgrnd'] and per_tile_snow:
                    _sub['eb_rows'].append(dict(
                        rows=dict({k_: tile_rows[k_] for k_ in
                                   ('advection', 'deltacc', 'rfrz_energy')},
                                  snow_flux=_wb_sflux),
                        active=snowing,
                        thin=(thin_w if thin_w is not None
                              else torch.zeros_like(snowing))))
                if 'adv' in band_rows:
                    band_rows['adv'].append(snow_eb['advection'][-1])
                if 'rfz' in band_rows:
                    band_rows['rfz'].append(snow_eb['rfrz_energy'][-1])
                if 'lsub' in band_rows:
                    band_rows['lsub'].append(-snow_eb['latent_sub'][-1])
                if canopy_lsub is not None:
                    # put_data.c 5.1.0:896: OUT_LATENT_SUB -= AtmosLatentSub,
                    # which calc_atmos_energy_bal.c 5.1.0:66 makes the
                    # canopy's plus the ground's on an overstory tile with
                    # snow, in water-balance mode as well
                    # (surface_fluxes.c 5.1.0:633-655).
                    if _cells_fused():
                        _c3 = _cell_sums(
                            dict(ls=canopy_lsub, adv=canopy_adv,
                                 rfz=canopy_rfz), {}, mosaic.cv,
                            lanes.cell_start, lanes.cell_mask, lanes.n_tiles,
                            lanes.lane_cell)
                        _ls_c, _adv_c, _rfz_c = _c3['ls'], _c3['adv'], _c3['rfz']
                    else:
                        _ls_c = _tsum(canopy_lsub * mosaic.cv)
                        _adv_c = _tsum(canopy_adv * mosaic.cv)
                        _rfz_c = _tsum(canopy_rfz * mosaic.cv)
                    snow_eb['latent_sub'][-1] = (
                        snow_eb['latent_sub'][-1] - _ls_c)
                    snow_eb['advection'][-1] = (snow_eb['advection'][-1]
                                                + _adv_c)
                    snow_eb['rfrz_energy'][-1] = (snow_eb['rfrz_energy'][-1]
                                                  + _rfz_c)
                    snow_eb['refreeze'][-1] = (snow_eb['refreeze'][-1]
                                               + _rfz_c
                                               / snow.CONST_LATICE * step_seconds)
                if (any_over or return_state) and per_tile_snow:
                    # The ground's LongUnderOut for the next step's canopy
                    # (surface_fluxes.c 5.1.0:520, :967): LongUnderIn -
                    # NetLongUnder (calc_surf_energy_bal.c 5.1.0:612). With
                    # Tsurf = Tair (:542-545) the bare part emits
                    # EMISS_GRND Tair^4 (func_surf_energy_bal.c 5.1.0
                    # :632-636), and so does a thin pack folded into the
                    # surface (INCLUDE_SNOW, :633-634); a pack snow_melt
                    # solved emits EMISS_SNOW Ts^4 from its own surface
                    # (SnowPackEnergyBalance.c 5.1.0:176-177, times its
                    # coverage at solve_snow.c 5.1.0:443). This was Tair's
                    # emission every step.
                    _endp = (snow_state['ice'] + snow_state['surf_water']
                             + snow_state['pack_water']) > 0.0
                    _thin = new_state.get('thin')
                    if _thin is not None:
                        _endp = _endp & ~_thin
                    _glo_wb = snow.calc_outgoing_longwave(
                        tair_t + CONST_TKFRZ, snow.EMISS_SNOW)
                    # a pack snow_melt solved: LongUnderIn - NetLongSnow
                    # (calc_surf_energy_bal.c 5.1.0:605, :612), its emission
                    # at the temperature of its last balance call, before
                    # snow_melt.c 5.1.0:433-456 rebalances the layers
                    if spatial_snow:
                        # SPATIAL_SNOW: NetLongUnder = NetLongBare +
                        # NetLongSnow, the bare part (1 - c)(LongUnderIn -
                        # its emission at Tair) (func_surf_energy_bal.c
                        # 5.1.0:636, calc_surf_energy_bal.c 5.1.0:214) and
                        # the pack's scaled by c = snow->coverage
                        # (solve_snow.c 5.1.0:442)
                        _c = cov_new
                        _nlb = ((1.0 - _c) * f['long_snow_in']
                                - (1.0 - _c) * _glo_wb)
                        _glo_sp = (f['long_snow_in']
                                   - (_nlb + se['net_long'] * _c))
                        _glo_wb = torch.where(_endp, _glo_sp, _glo_wb)
                    else:
                        _glo_wb = torch.where(
                            _endp, f['long_snow_in'] - se['net_long'],
                            _glo_wb)
                    if any_over:
                        ground_long_out = _glo_wb
                    if return_state:
                        _vic_aux['long_under_out'] = _glo_wb
                if return_state:
                    # calc_surf_energy_bal.c:561, func_surf_energy_bal.c
                    # :354-365: the conduction through the pack against the
                    # surface held at Tair
                    _dep_c = snow_state.get('depth')
                    if _dep_c is None:
                        _kap = torch.zeros_like(tair_t)
                    else:
                        # calc_surf_energy_bal.c:203-211 as the energy path
                        # forms it (surface_fluxes.c:603's mean depth)
                        _dm = _dep_c
                        if st.get('depth') is not None:
                            _dm = 0.5 * (st['depth'] + _dep_c)
                        _sd = torch.where(_dep_c > 0.0, _dm,
                                          torch.ones_like(_dep_c))
                        _kap = torch.where(
                            _dep_c > 0.0,
                            snow.SNOW_CONDUCT * new_state['density']
                            * new_state['density'] / _sd,
                            torch.zeros_like(_dep_c))
                    _thin_w = new_state.get('thin')
                    _thin_w = (torch.zeros_like(has_snow) if _thin_w is None
                               else _thin_w & has_snow)
                    _vic_aux['snow_flux'] = torch.where(
                        ((cov_new if spatial_snow else
                          ((snow_state['ice'] + snow_state['surf_water']
                            + snow_state['pack_water']) > 0.0)
                          .to(dtype)) > 0.0)
                        & ~_thin_w,
                        _kap * (new_state['surf_temp'] - tair_t),
                        torch.zeros_like(tair_t))
                    _vic_aux['tair_t'] = tair_t
                    _vic_aux['snow_on'] = snowing
                    _cc0 = _vic_aux.get('cold_content')
                    if _cc0 is None:
                        _cc0 = torch.zeros_like(snow_state['ice'])
                    _vic_aux['cold_content'] = torch.where(
                        has_snow | (wb_swq0 > 0.0), snow_state['cold_content'],
                        _cc0)
                    _tf0 = _vic_aux.get('tfoliage_bare')
                    if _tf0 is None:
                        _th_ = soil_params.thermal
                        _tf0 = (torch.as_tensor(
                            _th_['avg_temp'], dtype=dtype, device=device
                        ).unsqueeze(1).expand_as(snow_state['ice']).clone()
                            if (full_energy or frozen_soil)
                            else torch.zeros_like(snow_state['ice']))
                        if band_tfactor is not None:
                            _tf0 = _tf0 + torch.as_tensor(
                                band_tfactor, dtype=dtype, device=device
                            ).reshape(-1, 1)
                    _vic_aux['tfoliage_bare'] = torch.where(
                        snowing, _tf0, tair_t)
                stepped = mosaic.step(
                    m, wdew, water_in, moist, w_crit, w_wilt,
                    atm['shortwave'][_t1], atm['tair'][_t1], atm['vpd'][_t1],
                    _like(prcp, ra), net_rad, batch.elevation,
                    lai_scale=p['lai_scale'], rmin_scale=p['rmin_scale'],
                    lai_water_factor=lwf,
                    dt=step_seconds,
                    share_layer_moist=share_layer_moist,
                    wind=u, shortwave=atm['shortwave'][_t1],
                    longwave=atm['longwave'][_t1], per_tile=True,
                    return_rc=do_carbon, ground_heat=ground_heat,
                    ra_over=ra_over, over=over_t,
                    soil_rough=(th.get('rough') if th is not None else None),
                    snowing=snowing, wilting_clamp=True, exact_wetness=True,
                    bare_soil=bare_soil, atten_one=atten_one,
                    wdew_per_plant=not _sub['divide_fcan'],
                    # canopy_evap.c:201-241 transpires the liquid, against
                    # the ice the step opened with (calc_surf_energy_bal.c
                    # :532 runs the balance before :581 re-estimates it;
                    # frost-weighted with SPATIAL_FROST).
                    ice=_ice_eff())
                # (core.fusion.run: this block one fused region on the graph path)
                def _veg_out(stepped):
                    bare_cell = canopy_evap = col_evap_t = dry_t = evap = rc_t = tf_t = tr_t = wdew = _UNSET
                    stepped = list(stepped)
                    tr_t, ce_t, tf_t, wdew = stepped[:4]
                    if bare_soil is not None:
                        bare_t = stepped[4]
                        # arno_evap.c:192 writes layer 0 only; the tile's column
                        # gives up (1 - fcanopy) of it there (:773-775).
                        col_evap_t = torch.cat(
                            [tr_t[..., :1] + bare_t.unsqueeze(-1), tr_t[..., 1:]],
                            dim=-1)
                        bare_cell = _tsum(bare_t * cv)
                    if do_carbon:
                        rc_t, dry_t = stepped[-2], stepped[-1]
                    evap = _tsum(tr_t * mosaic.cv.unsqueeze(-1))
                    if wb_canopy_liquid is not None:
                        # An overstory tile whose canopy ran is under snow, so
                        # the step above left it alone (SNOWING); its canopy
                        # evaporation is the canopy stage's liquid branch,
                        # veg_var->canopyevap (put_data.c:639).
                        ce_t = ce_t + wb_canopy_liquid
                    canopy_evap = _tsum(ce_t * mosaic.cv)
                    throughfall = _tsum(tf_t * mosaic.cv)
                    return dict(bare_cell=bare_cell, canopy_evap=canopy_evap, col_evap_t=col_evap_t, dry_t=dry_t, evap=evap, rc_t=rc_t, tf_t=tf_t, tr_t=tr_t, wdew=wdew)
                _blk_res = _fusion.run(_veg_out, stepped)
                if _blk_res['bare_cell'] is not _UNSET:
                    bare_cell = _blk_res['bare_cell']
                if _blk_res['canopy_evap'] is not _UNSET:
                    canopy_evap = _blk_res['canopy_evap']
                if _blk_res['col_evap_t'] is not _UNSET:
                    col_evap_t = _blk_res['col_evap_t']
                if _blk_res['dry_t'] is not _UNSET:
                    dry_t = _blk_res['dry_t']
                if _blk_res['evap'] is not _UNSET:
                    evap = _blk_res['evap']
                if _blk_res['rc_t'] is not _UNSET:
                    rc_t = _blk_res['rc_t']
                if _blk_res['tf_t'] is not _UNSET:
                    tf_t = _blk_res['tf_t']
                if _blk_res['tr_t'] is not _UNSET:
                    tr_t = _blk_res['tr_t']
                if _blk_res['wdew'] is not _UNSET:
                    wdew = _blk_res['wdew']
                if bare_soil is not None:
                    evap_bares.append(bare_cell)
                if do_carbon:
                    _carbon_here(t, months[_t1], rc_t, dry_t, None,
                                 mosaic.rs_last if rc_photo else None)
                if frozen_soil and wb_quick_flux:
                    # calc_layer_average_thermal_props on the QUICK_FLUX
                    # path (frozen_soil.c:75-81, calc_surf_energy_bal.c
                    # :581), after the balance as VIC runs it: the layer
                    # temperatures from the new nodes
                    # (soil_conduction.c:598-628) and the ice from the
                    # unfrozen-water curve against the ENTERING moisture
                    # (soil_conduction.c:659-688, Nfrost 1), which the
                    # column below has not moved yet. Per surface
                    # evaluation, so a later snow sub-step reads this
                    # one's ice, as VIC's does.
                    from ..core import frozen as _frz
                    tl = energy.layer_temperatures_quick_flux(
                        wb_t0, wb_t1, tile2(th['avg_temp']),
                        th['depth'].unsqueeze(1).expand(nc, ntl, -1),
                        tile2(th['dp']))
                    _mx = soil_params.max_moist.unsqueeze(1).expand_as(tl)
                    _bb = soil_params.bubble.unsqueeze(1).expand_as(tl)
                    _ex = soil_params.expt.unsqueeze(1).expand_as(tl)
                    ice_new = _frz.layer_ice_quick_flux(
                        moist, _mx, _bb, _ex, tl,
                        None if _frost is None else _frost['slope'],
                        n_frost=1 if _frost is None else _frost['n'],
                        fs_active=getattr(soil_params, 'fs_active', None))
                    soil_ice = (ice_new if _frost is not None
                                else ice_new[..., 0])
                    wb_layer_temp.append(_tsum(tl * cv.unsqueeze(-1)))
                if (_w_wb and wb_quick_flux and mosaic.albedo is not None
                        and mosaic.veg_rough is not None
                        and mosaic.displacement is not None):
                    # ---- out-vars-wb: the energy rows of the Tsurf = Tair
                    # evaluation (torchvic.model.wbenergy), formed outside the
                    # fused regions from what this step holds; no state or
                    # existing row reads them.
                    _cvw = mosaic.cv
                    _zf = torch.zeros_like(tair_t)
                    _lw_sky = tile2(atm['longwave'][_t1])
                    _sw_sky = tile2(atm['shortwave'][_t1])
                    _lw_in = _lw_sky if canopy_long_out is None else canopy_long_out
                    _sw_in = _sw_sky if under_short is None else under_short
                    _thin_e = (thin_w if thin_w is not None
                               else torch.zeros_like(has_snow))
                    if spatial_snow:
                        _cov_e, _dcov_e = cov_new, delta_cov
                        _pns_e = se['net_short'] * cov_new
                        _pnl_e = se['net_long'] * cov_new
                        _alb_u_e = tile2(alb_under)
                    else:
                        _cov_e = has_p.to(dtype)
                        _dcov_e = (has_snow & ~has_p).to(dtype)
                        _pns_e = torch.where(has_p, se['net_short'], _zf)
                        _pnl_e = torch.where(has_p, se['net_long'], _zf)
                        _alb_u_e = _zf
                    _m_e = months[_t1]
                    _alb_t_e = mosaic.albedo[:, :, _m_e]
                    # Ra_veg[0] at corr == 1: the understory level's neutral
                    # resistance (the snow level's under a pack or snowfall,
                    # vic_run.c / solve_snow.c 5.1.0:133-139)
                    _z0e = mosaic.veg_rough[:, :, _m_e]
                    _rgh = th.get('rough') if th is not None else None
                    if _rgh is not None:
                        _z0e = torch.where(_z0e > 0.0, _z0e,
                                           tile2(_rgh).expand_as(_z0e))
                    _ra0_e = vegetation.aerodynamic_resistance(
                        _z0e, mosaic.displacement[:, :, _m_e], u.unsqueeze(1))
                    if any_over and ov_step is not None:
                        _ra0_e = torch.where(over_t, ov_step['ra'][..., 0],
                                             _ra0_e)
                    if ra_snow is not None:
                        _us2_e = ((wb_swq0 > 0.0) | tile2(sky_snow_m > 0.0)
                                  | (snow_p > 0.0))
                        _ra0_e = torch.where(_us2_e, ra_snow, _ra0_e)
                    _ev_mm = stepped[0].sum(-1) + stepped[1]
                    if bare_soil is not None:
                        _ev_mm = _ev_mm + stepped[4]
                    _thin_lat = tile_rows.get('thin_latent')
                    if _thin_lat is None:
                        _thin_lat = _zf
                    _r = _wbenergy.surface_rows(
                        tair=tair_t, lw_in=_lw_in, sw_in=_sw_in, cov=_cov_e,
                        dcov=_dcov_e, spatial=spatial_snow,
                        cov_eff=(cov_eff if spatial_snow else None),
                        thin=_thin_e, alb_t=_alb_t_e, albedo=albedo,
                        alb_under=_alb_u_e, pack_ns=_pns_e, pack_nl=_pnl_e,
                        pack_lat=se['latent'], pack_sen=se['sensible'],
                        pack_w=cover * (~_thin_e).to(dtype),
                        evap_ms=_ev_mm / MM_PER_M / step_seconds,
                        thin_latent=_thin_lat, ra_veg0=_ra0_e,
                        ra_bare=ra_bare, fc=fc_m,
                        air_dens=tile2(atm['air_dens'][_t1]),
                        le=tile2(atmos.latent_heat_of_vaporization(
                            atm['tair'][_t1])))
                    _ns_a, _nl_a = _r['net_short'], _r['net_long']
                    if canopy_ran is not None:
                        _ns_a = _ns_a + torch.where(canopy_ran, canopy_ns,
                                                    torch.zeros_like(_ns_a))
                        _nl_a = torch.where(canopy_ran, canopy_nl, _nl_a)
                    if 'lat' in band_rows:
                        band_rows['lat'].append(_tsum(_r['latent'] * _cvw))
                    if 'sen' in band_rows:
                        band_rows['sen'].append(_tsum(_r['sensible'] * _cvw))
                    _lat_a, _sen_a = _r['latent'], _r['sensible']
                    if canopy_lat is not None:
                        _lat_a = _lat_a - canopy_lat
                        _sen_a = _sen_a - canopy_sen
                        _sen_a = torch.where(over_t & canopy_ran,
                                             torch.zeros_like(_sen_a), _sen_a)
                    wb_e['net_short'].append(_tsum(_ns_a * _cvw))
                    wb_e['net_long'].append(_tsum(_nl_a * _cvw))
                    wb_e['net_rad'].append(_tsum((_ns_a + _nl_a) * _cvw))
                    wb_e['latent'].append(_tsum(_lat_a * _cvw))
                    wb_e['sensible'].append(_tsum(_sen_a * _cvw))
                    # calc_surf_energy_bal.c:617-618: the pack's surface (the
                    # step's opening one on a thin pack) weights the soil's
                    # Tsurf = Tair by the cover
                    _cr_e = cov_new if spatial_snow else has_p.to(dtype)
                    _ts_e = torch.where(_thin_e, old_snow_surf_temp,
                                        snow_state['surf_temp'])
                    _tsurf_e = _cr_e * _ts_e + (1.0 - _cr_e) * tair_t
                    wb_e['t_surf'].append(_tsum(_tsurf_e * _cvw))
                    # put_data.c:471-492, :653-688 as the energy path forms
                    # them
                    _ra_rep = _r['aero_resist']
                    _ra1_e = ra_veg1_t
                    if _ra1_e is None and ov_step is not None:
                        _ra1_e = ov_step['ra'][..., 1]
                    if _ra1_e is not None:
                        _ra_rep = torch.where(over_t, _ra1_e, _ra_rep)
                    _cond_e = _tsum(_cvw / torch.clamp(_ra_rep, min=1e-12))
                    wb_x['aero_cond'].append(_cond_e)
                    wb_e['aero_resist'].append(
                        1.0 / torch.clamp(_cond_e, min=1e-30))
                    _lv_e = _cvw > 0.0
                    _ra0u = _r['aero_resist']
                    _c1_e = _tsum(torch.where(
                        _lv_e, torch.where(
                            _ra0u > DBL_EPSILON, (1.0 / _ra0u) * _cvw,
                            torch.full_like(_cvw, vegetation.HUGE_RESIST)),
                        torch.zeros_like(_cvw)))
                    wb_x['aero_cond1'].append(_c1_e)
                    if over_t is not None and _ra1_e is not None:
                        wb_x['aero_cond2'].append(_tsum(torch.where(
                            over_t & _lv_e, torch.where(
                                _ra1_e > DBL_EPSILON, (1.0 / _ra1_e) * _cvw,
                                torch.full_like(_cvw,
                                                vegetation.HUGE_RESIST)),
                            torch.zeros_like(_cvw))))
                    else:
                        wb_x['aero_cond2'].append(torch.zeros_like(_c1_e))
                    # put_data.c 5.1.0:820-844, :443-456
                    _t_fol_e = (tair_t if vegt_canopy is None else vegt_canopy)
                    _t_rad_e = (torch.where(over_t & snowing, _t_fol_e,
                                            _tsurf_e)
                                if over_t is not None else _tsurf_e)
                    _t_k_e = _t_rad_e + CONST_TKFRZ
                    wb_x['rad_temp'].append(
                        _tsum((_t_k_e * _t_k_e * _t_k_e * _t_k_e) * _cvw)
                        ** 0.25)
                    _t_veg_e = (torch.where(over_t, _t_fol_e, _tsurf_e)
                                if over_t is not None else _tsurf_e)
                    _cv_v = (_cvw if mosaic.bare_tile is None else
                             torch.where(mosaic.bare_tile.expand_as(_cvw),
                                         torch.zeros_like(_cvw), _cvw))
                    wb_x['vegt'].append(_tsum(_t_veg_e * _cv_v)
                                        / torch.clamp(_tsum(_cv_v), min=1e-12))
                    wb_x['snow_flux'].append(_tsum(_wb_sflux * _cvw))
                    # OUT_SOIL_TEMP, OUT_SOIL_TNODE: Tnew_node = (Tsurf, T1,
                    # avg_T + (T1 - avg_T) exp(-(Zsum_node[2] - D1) / dp))
                    # (calc_surf_energy_bal.c:593-599), layer temperatures from
                    # them (frozen_soil.c:75-81)
                    _avg_e = tile2(th['avg_temp'])
                    _d1_e = tile2(th['depth'][:, 0])
                    _dp_e = tile2(th['dp'])
                    _tl_e = energy.layer_temperatures_quick_flux(
                        wb_t0, wb_t1, _avg_e,
                        th['depth'].unsqueeze(1).expand(nc, ntl, -1), _dp_e)
                    wb_e['layer_temp'].append(
                        _tsum(_tl_e * _cvw.unsqueeze(-1)))
                    _T2_e = _avg_e + (wb_t1 - _avg_e) * torch.exp(
                        -(_dp_e - _d1_e) / _dp_e)
                    wb_x['tnode'].append(_tsum(
                        torch.stack([wb_t0, wb_t1, _T2_e], -1)
                        * _cvw.unsqueeze(-1)))
                    # compute_pot_evap.c (surface_fluxes.c:788-798)
                    _lai_e = (mosaic.lai[:, :, _m_e]
                              * vegetation.cellwise(p['lai_scale'], _cvw))
                    _fcq_e = (torch.ones_like(_lai_e) if fc_m is None
                              else fc_m)
                    if core_version.at_least('5.1.0'):
                        _lai_e = _lai_e / torch.where(
                            _fcq_e > 0.0, _fcq_e, torch.ones_like(_fcq_e))
                    else:
                        _lai_e = _lai_e / _fcq_e
                        if mosaic.bare_tile is not None:
                            _lai_e = torch.where(
                                mosaic.bare_tile,
                                mosaic.lai[:, :, _m_e]
                                * vegetation.cellwise(p['lai_scale'], _cvw),
                                _lai_e)
                    _ra_pot = _ra0_e
                    if _ra1_e is not None:
                        _ra_pot = torch.where(over_t, _ra1_e, _ra_pot)
                    _epot = _wbenergy.potential_evap(
                        tair=tair_t, vpd=tile2(atm['vpd'][_t1]),
                        elev=tile2(batch.elevation), sw_sky=_sw_sky,
                        alb_t=_alb_t_e, nl_atm=_nl_a,
                        rmin=mosaic.rmin * vegetation.cellwise(
                            p['rmin_scale'], mosaic.rmin),
                        rgl=mosaic.rgl, rarc=mosaic.rarc, lai=_lai_e,
                        ra_veg=_ra_pot, ra_used=_r['aero_resist'], fc=fc_m)
                    _dtp_e = (_sub['model_dt'] if _sub['pin_tgrnd']
                              else step_seconds)
                    wb_pets.append(_tsum(_epot * _cvw)
                                   * (_dtp_e / vegetation.CONST_CDAY))
            _sub_last = sublimations[-1]
            _subc_last = sub_canops[-1] if sub_canops else 0.0
            # (core.fusion.run: this block one fused region on the graph path)
            def _flux_out():
                _oq = _UNSET
                et_t = (evap.sum(1) + canopy_evap + _sub_last
                        + _subc_last)
                if bare_cell is not None:
                    et_t = et_t + bare_cell          # put_data.c:612-622
                _oq = [('et', et_t), ('transp', evap.sum(1)),
                       ('canopy_evap', canopy_evap)]
                # put_data.c:660 and :664 weight each tile's Wdew and LAI by its
                # cover. Wdew is the store the step leaves on the leaves, so it is
                # read after the canopy has run; LAI is this month's leaf area at
                # the scale the canopy actually used, which is what veg_var.LAI
                # holds by the time put_data reads it.
                _oq.append(('wdew', _tsum(wdew * mosaic.cv)))
                _oq.append(('lai', _tsum(mosaic.lai[:, :, months[_t1]]
                                         * vegetation.cellwise(p['lai_scale'],
                                                               mosaic.cv)
                                         * mosaic.cv)))
                return dict(_oq=_oq)
            _blk_res = _fusion.run(_flux_out)
            if _blk_res['_oq'] is not _UNSET:
                _oq = _blk_res['_oq']
            for _n, _v in _oq:
                _named[_n].append(_v)
            # diagnostics=True changes nothing about the moisture the step
            # returns; it adds the dict of what runoff() computes for the
            # history file, read from the state the step leaves behind.
            # Each tile's column takes its own throughfall and transpiration
            # (vic_run.c loops surface_fluxes over iveg); the cell's runoff,
            # baseflow, moisture and diagnostics are the cover-weighted sums
            # over tiles, as put_data.c forms them (AreaFactor).
            ntl = mosaic.n_tiles
            # surface_fluxes.c:954 and :1080: ppt is the sum of the snow
            # steps' throughfall, and runoff() runs once for the model step.
            _tf_col = tf_t
            _ev_col = tr_t if col_evap_t is None else col_evap_t
            if _sub['add_tf'] is not None:
                _tf_col = _tf_col + _sub['add_tf']
                _ev_col = _ev_col + _sub['add_ev']
            if _sub['skip']:
                _sub['cap_tf'] = _tf_col
                _sub['cap_ev'] = _ev_col
                return
            if _frost is not None and soil_ice is not None:
                # runoff.c per frost sub-area: each area runs its own
                # column with its own ice and its share of the layer
                # evaporation (:86-140), and the moisture, runoff,
                # baseflow and saturated fraction are the area-weighted
                # sums (:356-368). The water table and wetness read the
                # summed moisture, as wrap_compute_zwt does (:370-372).
                _ev_f = column.distribute_evap_frost(
                    _ev_col.reshape(nc * ntl, -1),
                    moist.reshape(nc * ntl, -1),
                    soil_ice.reshape(nc * ntl, -1, _frost['n']),
                    soil_params_t, _frost['fract'])
                _fr = _frost['fract']
                _m0 = moist.reshape(nc * ntl, -1)
                _p0 = _tf_col.reshape(-1)
                m_f = torch.zeros_like(_m0)
                ro_f = torch.zeros_like(_p0)
                bf_f = torch.zeros_like(_p0)
                _asat = torch.zeros_like(_p0)
                _with = torch.zeros_like(_p0)
                _gwr = torch.zeros_like(_p0)
                _rch = torch.zeros_like(_p0)
                for _fi, _w in enumerate(_fr):
                    _mf, _ro, _bf, _dg = column.step(
                        _m0, _p0, _ev_f[..., _fi].reshape(nc * ntl, -1),
                        (soil_params_t if _vw_ksat[0] is None
                         else _vw_ksat[0]), diagnostics=True,
                        ice=soil_ice[..., _fi].reshape(nc * ntl, -1),
                        curves=zwt_curves, gwm=_vw_gwm)
                    m_f = m_f + _mf * _w
                    ro_f = ro_f + _ro * _w
                    bf_f = bf_f + _bf * _w
                    _asat = _asat + _dg['asat'] * _w
                    _with = _with + _dg['evap_withheld'] * _w
                    if _vw_gwm is not None:
                        _gwr = _gwr + _dg['gwrecharge'] * _w
                        _rch = _rch + _dg['recharge'] * _w
                diag = dict(
                    gwrecharge=_gwr, recharge=_rch,
                    asat=_asat, inflow=_p0, evap_withheld=_with,
                    soil_wet=column.soil_wetness(m_f, soil_params_t),
                    rootmoist=(None if soil_params_t.root is None else
                               column.root_zone_moisture(m_f,
                                                         soil_params_t.root)),
                    zwt=None, zwt_layer=None, zwt_lumped=None)
                if (soil_params_t.depth is not None
                        and soil_params_t.bubble is not None):
                    _zc = zwt_curves or {}
                    diag['zwt'], diag['zwt_layer'] = \
                        column.wrap_compute_zwt(m_f, soil_params_t,
                                                curve=_zc.get('layer'))
                    diag['zwt_lumped'] = column.compute_zwt_lumped(
                        m_f, soil_params_t, curve=_zc.get('lumped'))
            else:
                m_f, ro_f, bf_f, diag = column.step(
                    moist.reshape(nc * ntl, -1), _tf_col.reshape(-1),
                    _ev_col.reshape(nc * ntl, -1),
                    (soil_params_t if _vw_ksat[0] is None else _vw_ksat[0]),
                    diagnostics=True,
                    ice=(None if soil_ice is None
                         else soil_ice.reshape(nc * ntl, -1)),
                    curves=zwt_curves, gwm=_vw_gwm)
            # (core.fusion.run: this block one fused region on the graph path)
            def _col_out():
                _oq2 = moist = node_props = _UNSET
                moist = m_f.view(nc, ntl, -1)
                if use_eb and not quick_flux:
                    # runoff.c:381: the column left new moisture, so the
                    # node properties are redistributed from it and the
                    # current profile. Once per model step: on a skipped
                    # snow sub-step the column above did not run.
                    node_props = _distribute(moist, t_node)
                wsum = lambda x: _tsum(x.view(nc, ntl) * mosaic.cv)
                _oq2 = [('runoff', wsum(ro_f)), ('baseflow', wsum(bf_f))]
                if _vw_gwm is not None:
                    # VIC-WUR 3.0 put_data.c:653-656, Cv-weighted like runoff
                    _oq2.append(('gwrecharge', wsum(diag['gwrecharge'])))
                    _oq2.append(('recharge', wsum(diag['recharge'])))
                if _lake_tap is not None:
                    _lake_tap['ro'] = ro_f.view(nc, ntl)
                    _lake_tap['bf'] = bf_f.view(nc, ntl)
                    _lake_tap['asat'] = diag['asat'].view(nc, ntl)
                    _lake_tap['soil_wet'] = (None if diag['soil_wet'] is None
                                             else diag['soil_wet'].view(nc, ntl))
                    _lake_tap['lai'] = (mosaic.lai[:, :, months[_t1]]
                                        * vegetation.cellwise(p['lai_scale'],
                                                              mosaic.cv))
                # put_data.c:691-703: OUT_SOIL_MOIST is the total,
                # OUT_SOIL_LIQ is the total less the ice, OUT_SOIL_ICE the
                # ice. With FROZEN_SOIL off the last two are the total and
                # zero, which is what this reduces to.
                # `soil_liqs` carries the TOTAL, because that is the store the
                # water balance closes against -- ice is water, and reporting
                # the liquid here opened a 700 mm hole the first step a
                # synthetic column froze. OUT_SOIL_LIQ is formed from it and
                # the ice at the end.
                _sl = _tsum(moist * mosaic.cv.unsqueeze(-1))
                _oq2.append(('soil_liq', _sl))
                _oq2.append(('soil_ice', torch.zeros_like(_sl)
                             if soil_ice is None
                             else _tsum(_ice_eff() * mosaic.cv.unsqueeze(-1))))
                _oq2.append(('asat', wsum(diag['asat'])))
                _oq2.append(('withheld', wsum(diag['evap_withheld'])))
                _oq2.append(('inflow', wsum(diag['inflow'])))
                _oq2.append(('soil_wet', None if diag['soil_wet'] is None
                             else wsum(diag['soil_wet'])))
                _oq2.append(('zwt', None if diag['zwt'] is None
                             else wsum(diag['zwt'])))
                # put_data.c 5.1.0:667-668: OUT_ZWT_LUMPED, the column's
                # lumped water table, over tiles like OUT_ZWT
                if _w_zwtl:
                    _oq2.append(('zwt_lumped',
                                 None if diag['zwt_lumped'] is None
                                 else wsum(diag['zwt_lumped'])))
                # vic_run.c:358-364 sums, per tile, the whole moisture of every
                # layer that tile's roots reach; put_data.c:705 then weights the
                # tiles by cover.
                _rf = mosaic.root_frac
                if mosaic.bare_tile is not None:
                    # calc_root_fraction.c 5.1.0:52 fills veg < Nveg only: the
                    # bare tile's roots stay calloc's zeros, so vic_run.c 5.1.0
                    # :335-340 gives it no root-zone moisture
                    _rf = torch.where(mosaic.bare_tile.unsqueeze(-1), torch.zeros_like(_rf), _rf)
                _oq2.append(('rootmoist', _tsum(column.root_zone_moisture(
                    moist, _rf) * mosaic.cv)))
                return dict(_oq2=_oq2, moist=moist, node_props=node_props)
            _blk_res = _fusion.run(_col_out)
            if _blk_res['_oq2'] is not _UNSET:
                _oq2 = _blk_res['_oq2']
            if _blk_res['moist'] is not _UNSET:
                moist = _blk_res['moist']
            if _blk_res['node_props'] is not _UNSET:
                node_props = _blk_res['node_props']
            for _n, _v in _oq2:
                _named[_n].append(_v)
            # out-vars: the frozen-soil diagnostics put_data.c reports. The
            # frost cover of the top layer (collect_eb_terms :800-806: the
            # frost areas whose top layer holds ice, their fractions summed,
            # over tiles by AreaFactor) and the freezing and thawing fronts
            # the finite nodes hold (find_0_degree_fronts, run from
            # calc_layer_average_thermal_props.c:39-41 on the nodes the step
            # ended with; :788-798 reports them in cm over tiles)
            _cvt = mosaic.cv
            if _w_frost:
                if soil_ice is None:
                    _sff = torch.zeros_like(_tsum(_cvt))
                elif _frost is None:
                    _sff = _tsum((soil_ice[..., 0] > 0.0).to(dtype) * _cvt)
                else:
                    _sff = _tsum(((soil_ice[:, :, 0, :] > 0.0).to(dtype)
                                  * _frost['fract_t']).sum(-1) * _cvt)
                _named['surf_frost_frac'].append(_sff)
            # OUT_SNOW_SURFT_BAND / OUT_SNOW_PACKT_BAND (put_data.c:906-912):
            # the pack's temperatures times Cv, with no snow gate
            for _bk, _sk in (('surft', 'surf_temp'), ('packt', 'pack_temp')):
                if _bk not in band_rows:
                    continue
                _x = snow_state[_sk]
                _x = _x.unsqueeze(1) if _x.dim() == 1 else _x
                band_rows[_bk].append(_tsum(_x * _cvt))
            if _w_fronts and frozen_soil and (use_eb or wb_quick_flux):
                _th_f = soil_params.thermal
                if use_eb:
                    _Tn = t_node
                    _Zn = (fd_static['zsum'].unsqueeze(1)
                           if not quick_flux else None)
                else:
                    _avg = _th_f['avg_temp'].unsqueeze(1)
                    _d1f = _th_f['depth'][:, 0].unsqueeze(1)
                    _dpf = _th_f['dp'].unsqueeze(1)
                    _T2 = _avg + (wb_t1 - _avg) * torch.exp(
                        -(_dpf - _d1f) / _dpf)
                    _Tn = torch.stack([wb_t0, wb_t1, _T2], -1)
                    _Zn = None
                if _Zn is None:
                    _Zn = torch.stack([
                        torch.zeros_like(_th_f['dp']),
                        _th_f['depth'][:, 0], _th_f['dp']], -1).unsqueeze(1)
                _fd, _td = _outvars.find_0_degree_fronts(_Tn, _Zn)
                _fsa = getattr(soil_params, 'fs_active', None)
                if _fsa is not None:
                    # FS_ACTIVE false (frozen_soil.c:39): the fronts are
                    # never found and stay at initialize_energy.c's 0
                    _ka = _fsa.reshape(-1, 1, 1)
                    _fd = torch.where(_ka, _fd, torch.zeros_like(_fd))
                    _td = torch.where(_ka, _td, torch.zeros_like(_td))
                _named['fdepth'].append(_tsum(_outvars.front_rows(_fd, _cvt)))
                _named['tdepth'].append(_tsum(_outvars.front_rows(_td, _cvt)))
            return

        pet = penman_monteith_pet({k: atm[k][_t1] for k in
                                   ('tair', 'press', 'lv', 'air_dens', 'vpd')},
                                  net_rad, p['rs'], ra) \
            * (step_seconds / SEC_PER_DAY)
        # Soil moisture limitation, layer by layer, on the two upper layers.
        evap = []
        for lay in range(3):
            cap = soil_params.max_moist[..., lay]
            res = soil_params.resid_moist[..., lay]
            # The base of the power is floored well away from zero. The
            # derivative of avail**beta is beta * avail**(beta - 1), which is
            # infinite at avail = 0 for any beta below 1, and et_beta is
            # allowed down to 0.2. A cell that dries out completely would then
            # emit an infinite gradient, which is what collapsed training: the
            # guard skipped every update in an epoch and the run reported
            # zero loss rather than failing.
            #
            # Flooring the base is the fix rather than masking the result with
            # `where`, because `where` still evaluates the unselected branch
            # and its NaN propagates through the backward pass regardless.
            avail = torch.clamp((moist[:, lay] - res)
                                / torch.clamp(cap - res, min=1e-6),
                                ET_AVAIL_FLOOR, 1.0)
            share = (0.6, 0.4, 0.0)[lay]
            evap.append(pet * share * avail ** p['et_beta'])
        evap = torch.stack(evap, dim=1)
        ets.append(evap.sum(1) + sublimations[-1])   # put_data.c:626

        _wi, _evs = water_in, evap
        if _sub['add_tf'] is not None:
            _wi = _wi + _sub['add_tf']
            _evs = _evs + _sub['add_ev']
        if _sub['skip']:
            _sub['cap_tf'] = _wi
            _sub['cap_ev'] = _evs
            return
        moist, ro, bf, diag = column.step(moist, _wi, _evs, soil_params,
                                          diagnostics=True, curves=zwt_curves)
        runoffs.append(ro)
        baseflows.append(bf)
        soil_liqs.append(moist)
        asats.append(diag['asat'])
        withhelds.append(diag['evap_withheld'])
        inflows.append(diag['inflow'])
        soil_wets.append(diag['soil_wet'])
        zwts.append(diag['zwt'])
        if _w_zwtl:
            zwt_lumpeds.append(diag['zwt_lumped'])

    def _ce_step(t_):
        """One step with CLOSE_ENERGY: _step_inner iterated to closure.

        surface_fluxes.c:442-720 (5.0.1:469-730 the same loop): each tile
        runs its own overstory/understory iteration -- the understory
        re-solving the snow and ground balances at the averaged canopy
        air temperature until the snow_flux tolerance stills, the
        overstory until the canopy temperature stills -- up to
        MAX_ITER_GRND_CANOPY passes each, and keeps its last pass. A
        pass here is a whole _step_inner with the pass's per-tile
        Tcanopy (and UNSTABLE_SNOW); a tile whose loops have exited
        freezes its Tcanopy, so re-running it reproduces its last pass
        to the bit while the others converge, and the committing pass
        runs the column once on every tile's own last pass (:954,
        :1080). Tcanopy is solved inside the pass, from that pass's
        canopy and ground sensible heat (calc_atmos_energy_bal.c).
        """
        nonlocal ce_seed
        ntl = mosaic.n_tiles
        snap0 = _snap()
        lens0 = {k_: len(v_) for k_, v_ in _named.items()}
        skip0 = _sub['skip']
        soil_adv0, soil_dcc0 = _sub['soil_adv'], _sub['soil_dcc']
        rows0 = (len(_sub['eb_rows']), len(_sub['can_rows']),
                 len(_sub['alb_rows']))
        over_cls = (over_t if over_t is not None
                    else torch.zeros(nc, ntl, dtype=torch.bool,
                                     device=device))
        # surface_fluxes.c:365-374: Tcanopy opens at the air's every snow
        # step; the snow_flux seed and UNSTABLE_SNOW persist across the
        # snow steps of a model step (:225-230 is outside the hidx loop).
        tcanopy = atm['tair'][t_].unsqueeze(1).expand(nc, ntl).to(dtype)
        tcanopy = tcanopy.clone()
        last_tcanopy = torch.full_like(tcanopy, 999.0)
        if _sub['pin_tgrnd'] and _sub['ce_carry'] is not None:
            snow_flux = _sub['ce_carry']['snow_flux']
            unstable = _sub['ce_carry']['unstable']
        else:
            swq_open = ((snow_state['ice'] + snow_state['surf_water']
                         + snow_state['pack_water']) > 0.0)
            snow_flux = torch.where(
                swq_open, ce_seed[..., 0],
                -(ce_seed[..., 1] + ce_seed[..., 2] + ce_seed[..., 3]))
            unstable = torch.zeros(nc, ntl, dtype=torch.bool, device=device)
        last_snow_flux = torch.full_like(tcanopy, 999.0)
        f_prev = torch.zeros_like(tcanopy)
        include = torch.zeros(nc, ntl, dtype=torch.bool, device=device)
        over_iter = torch.zeros(nc, ntl, dtype=torch.int64, device=device)
        under_iter = torch.zeros(nc, ntl, dtype=torch.int64, device=device)
        tol_over = torch.full_like(tcanopy, 999.0)
        last_tol_over = torch.full_like(tcanopy, 999.0)
        tol_under = torch.full_like(tcanopy, 999.0)
        last_tol_under = torch.full_like(tcanopy, 999.0)
        store = torch.full_like(tcanopy, 999.0)
        a_tol = torch.full_like(tcanopy, 999.0)
        unstable_cnt = torch.zeros(nc, ntl, dtype=torch.int64, device=device)
        done = torch.zeros(nc, ntl, dtype=torch.bool, device=device)
        understory = torch.full((nc, ntl), 999, dtype=torch.int64,
                                 device=device)
        reopen = torch.zeros(nc, ntl, dtype=torch.bool, device=device)
        maxit = energy.MAX_ITER_GRND_CANOPY
        # The first over-iteration opens as :446-459 does.
        over_iter = over_iter + 1
        try:
            _sub['skip'] = True
            n_pass = 0
            while True:
                active = ~done
                # An under pass opens (:465-491): the pass counts, the
                # averaged Tcanopy, the snow_flux bisection unless the
                # last pass stepped the wrong way or left INCLUDE_SNOW.
                last_tol_under = torch.where(active, tol_under,
                                             last_tol_under)
                under_iter = torch.where(active, under_iter + 1, under_iter)
                avg = active & (last_tcanopy != 999.0)
                tcanopy = torch.where(
                    avg, (last_tcanopy + tcanopy) / 2.0, tcanopy)
                last_tcanopy = torch.where(active, tcanopy, last_tcanopy)
                live = active & (last_snow_flux != 999.0)
                grow = (((store.abs() > a_tol.abs()) & (a_tol != 999.0)
                         & ((store - a_tol).abs() > 1.0))
                        | (tol_under < 0.0))
                hit = live & grow
                unstable_cnt = torch.where(hit, unstable_cnt + 1,
                                           unstable_cnt)
                unstable = torch.where(
                    hit & ((unstable_cnt > 3) | (tol_under < 0.0)),
                    torch.ones_like(unstable), unstable)
                bis = live & ~grow & ~include
                snow_flux = torch.where(
                    bis, (last_snow_flux + f_prev) / 2.0, snow_flux)
                # Outside the live guard, as :489 is outside the if.
                last_snow_flux = torch.where(active, snow_flux,
                                             last_snow_flux)
                a_tol = torch.where(active, store, a_tol)
                _restore(snap0)
                understory = torch.where(
                    reopen, torch.full_like(understory, 999), understory)
                _sub['tcanopy'] = tcanopy
                _sub['unstable'] = unstable
                _sub['understory'] = understory
                _sub['soil_adv'] = soil_adv0
                _sub['soil_dcc'] = soil_dcc0
                _step_inner(t_)
                passed = _sub['ce_pass']
                understory = torch.where(
                    active & passed['pack_branch'],
                    torch.full_like(understory, 2), understory)
                _harvest(lens0)
                del _sub['eb_rows'][rows0[0]:]
                del _sub['can_rows'][rows0[1]:]
                del _sub['alb_rows'][rows0[2]:]
                # The pass's tolerances (:690-712).
                f_prev = torch.where(active, passed['snow_flux'], f_prev)
                include = torch.where(active, passed['thin'], include)
                zero_tol = (passed['thin'] | (~passed['has_p']
                                              & (passed['dcov'] == 0.0)))
                store_new = snow_flux - passed['snow_flux']
                store = torch.where(
                    active, torch.where(zero_tol, torch.zeros_like(store),
                                        store_new), store)
                tol_under = torch.where(
                    active, torch.where(
                        zero_tol, torch.zeros_like(tol_under),
                        store_new.abs()), tol_under)
                stall = (((tol_under - last_tol_under).abs()
                          < energy.TOL_GRND) & (tol_under > 1.0))
                tol_under = torch.where(
                    active & stall, torch.full_like(tol_under, -999.0),
                    tol_under)
                gate = passed['gate']
                tol_over = torch.where(
                    active, torch.where(
                        gate, (passed['tcanopy'] - last_tcanopy).abs(),
                        torch.zeros_like(tol_over)), tol_over)
                under_go = (((tol_under - last_tol_under).abs()
                             > energy.TOL_GRND) & (tol_under != 0.0)
                            & (under_iter < maxit))
                leave_under = active & ~under_go
                over_go = (((tol_over - last_tol_over).abs()
                            > energy.TOL_OVER) & over_cls
                           & (tol_over != 0.0) & (over_iter < maxit))
                new_over = leave_under & over_go
                # A new over-iteration re-opens the under loop (:446-459);
                # last_Tcanopy, last_snow_flux, snow_flux, Tcanopy and
                # UNSTABLE_SNOW persist, as the C's locals do.
                over_iter = torch.where(new_over, over_iter + 1, over_iter)
                last_tol_over = torch.where(new_over, tol_over,
                                            last_tol_over)
                under_iter = torch.where(new_over,
                                         torch.zeros_like(under_iter),
                                         under_iter)
                tol_under = torch.where(
                    new_over, torch.full_like(tol_under, 999.0), tol_under)
                a_tol = torch.where(
                    new_over, torch.full_like(a_tol, 999.0), a_tol)
                store = torch.where(
                    new_over, torch.full_like(store, 999.0), store)
                unstable_cnt = torch.where(
                    new_over, torch.zeros_like(unstable_cnt), unstable_cnt)
                reopen = new_over
                done = done | (leave_under & ~over_go)
                # VIC calls the atmosphere balance only on the overstory
                # with snow (:637); elsewhere Tcanopy stays the average.
                # A tile that just exited keeps its last pass's Tcanopy:
                # the committing pass re-runs that pass, not a new one at
                # the solved temperature.
                tcanopy = torch.where(active & gate & ~done,
                                      passed['tcanopy'], tcanopy)
                n_pass += 1
                if bool(done.all()):
                    break
                if n_pass > maxit * maxit:
                    raise RuntimeError(
                        'CLOSE_ENERGY iteration did not converge in %d '
                        'passes' % (maxit * maxit))
            # The committing pass: every tile reproduces its own last
            # pass, and the column runs once on those.
            _restore(snap0)
            _sub['tcanopy'] = tcanopy
            _sub['unstable'] = unstable
            _sub['understory'] = torch.where(
                reopen, torch.full_like(understory, 999), understory)
            _sub['soil_adv'] = soil_adv0
            _sub['soil_dcc'] = soil_dcc0
            _sub['ce_final'] = True
            _sub['skip'] = skip0
            _step_inner(t_)
            passed = _sub['ce_pass']
            if _sub['pin_tgrnd']:
                _sub['ce_seed_rows'].append(dict(
                    seed=passed['seed'],
                    snow_or_include=(passed['snow_snow']
                                     | passed['thin'])))
                _sub['ce_carry'] = dict(snow_flux=snow_flux,
                                        unstable=unstable)
            else:
                ce_seed = passed['seed']
        finally:
            _sub['tcanopy'] = None
            _sub['unstable'] = None
            _sub['understory'] = None
            _sub['ce_pass'] = None
            _sub['ce_final'] = False
            _sub['skip'] = skip0

    # torchvic.vicwur.core (VIC-WUR's vic_run hooks). _vw is None on every
    # run that does not ask for them, and then nothing below runs.
    _vw = (vicwur_core if vicwur_core is not None and vicwur_core.active()
           else None)
    _vw_ksat = [None]            # the step's SoilParams with layer Ksat
    _vw_out = {}
    _vw_tf = [None]              # tile_fluxes: the last step's
    _vw_ws = None
    # the step's hook values (_vw_host), read by the step (_vw_dev); on the
    # lanes each is a buffer written in place (a CUDA graph reads it)
    _vw_now = {}
    # on the graph path the host part runs before the replay (_vw_host),
    # the step's own (_vw_dev, _vw_after) inside it
    _vw_split = [False]
    # the graph path's state sync (set there): the carried states a
    # state_fn replaced into the graphs' buffers
    _vw_sync = [None]
    # the graph path's month buffers (g_mosaic) and its monthly fields
    _vw_gm = [None, ()]
    # GWM (VIC-WUR 3.0 runoff_GWM): OUT_GWRECHARGE and OUT_RECHARGE per step
    _vw_gwm = getattr(_vw, 'gwm', None) if _vw is not None else None
    gwrecharges, recharges = [], []
    if _vw_gwm is not None:
        if mosaic is None:
            raise NotImplementedError('vicwur_core gwm is wired on the '
                                      'mosaic water-balance path only')
        _named['gwrecharge'] = gwrecharges
        _named['recharge'] = recharges
        if lanes is not None:
            # cell-level rows (Cv-weighted tile sums) like runoff's: one
            # lane per cell after the step (_flush_lanes)
            _out_lists.extend([gwrecharges, recharges])
            _flushed.extend([0, 0])
    # lanes: the hooks' [column, tile, ...] values at the lanes. A value
    # handed again (the same tensor, unmodified) is not gathered again.
    _vw_gc = {}

    def _vw_ver(x_):
        # the tensor's version counter (None for an inference tensor, which
        # keeps none: such a value is gathered again every step)
        try:
            return x_._version
        except RuntimeError:
            return None

    def _vw_lane(x_, tail_, key=None, cell=False):
        # x_ broadcast to the padded [column, tile, *tail_] (or [column]
        # with cell=True) as the padded step broadcasts it, at the lanes
        if key is not None:
            c_ = _vw_gc.get(key)
            if c_ is not None and c_[0] is x_ and c_[1] is not None \
                    and c_[1] == _vw_ver(x_):
                return c_[2]
        x0_ = x_
        x_ = x_.to(device)
        if cell:
            y_ = lanes.gather_cell(x_.expand((lanes.n_cells,) + tuple(tail_)))
        else:
            y_ = lanes.gather_tile(x_.expand((lanes.n_cells, lanes.n_tiles)
                                             + tuple(tail_)))
        if key is not None:
            _vw_gc[key] = (x0_, _vw_ver(x0_), y_)
        return y_

    def _vw_cols(x_, steps=False):
        # a lane value [(step,) lane, 1, ...] on the columns [(step,)
        # column, tile, ...], zero on the tiles that are not lanes
        if lanes is None or x_ is None:
            return x_
        if not steps:
            return lanes.scatter(x_)
        full_ = torch.zeros((x_.shape[0], lanes.n_cells, lanes.n_tiles)
                            + tuple(x_.shape[3:]), dtype=x_.dtype,
                            device=x_.device)
        full_[:, lanes.lane_cell, lanes.lane_tile] = x_[:, :, 0]
        return full_

    def _vw_put(name_, x_, tail_):
        # the lanes' buffer `name_` holding x_ (gathered once per value,
        # copied in only when it changed)
        y_ = _vw_lane(x_, tail_, key=name_)
        b_ = _vw_buf.get(name_)
        if b_ is None:
            b_ = _vw_buf[name_] = y_.clone()
            _vw_src[name_] = y_
        elif _vw_src.get(name_) is not y_:
            b_.copy_(y_)
            _vw_src[name_] = y_
        _vw_now[name_] = b_
        return b_
    _vw_buf, _vw_src = {}, {}
    _vw_lmask = [None]           # lanes: [column, tile] True on a lane
    _vw_cvsrc = [None]
    if _vw is not None:
        from ..vicwur import core as _vwcore
        _vw_months = months
        _vw_veg = (_vwcore._VegOverride(
            mosaic, list(_vw.veg),
            to_lanes=None if lanes is None else _vw_lane)
                   if _vw.veg else None)
        # padded: the case's mosaic, whose Cv a cv hook replaces until the
        # call's end; lanes: the lanes' Cv is a private buffer written in
        # place, the case's (the coupling reads it) replaced as padded
        _vw_cv0 = (mosaic if lanes is None else cell_mosaic).cv
        if lanes is not None and _vw.cv is not None:
            mosaic.cv = mosaic.cv.clone()
            _lm_ = torch.zeros(lanes.n_cells, lanes.n_tiles,
                               dtype=torch.bool, device=device)
            _lm_[lanes.lane_cell, lanes.lane_tile] = True
            _vw_lmask[0] = _lm_
        if _vw.water_stress:
            # VIC-WUR 3.0 initialize_soil.c:62: layer[l].water_stress = 0
            _vw_ws = (torch.zeros(nc, mosaic.n_tiles, moist.shape[-1], dtype=dtype,
                                  device=device)
                      if _vw.stress_init is None else
                      (torch.as_tensor(_vw.stress_init, dtype=dtype,
                                       device=device).clone()
                       if lanes is None else _vw_lane(torch.as_tensor(
                           _vw.stress_init, dtype=dtype, device=device),
                           (moist.shape[-1],)).clone()))
            _vw_out.update(water_stress=[])
            if getattr(_vw, 'stress_history', True):
                _vw_out.update(water_stress_layer=[])
        if _vw.diagnostics:
            _vw_out.update(transp_tile=[], esoil_tile=[])
    # the hooks' per-step rows (the graph path harvests them like the
    # outputs)
    _vw_lists = list(_vw_out.values())
    # VIC-CropSyst's Wpwp floor (vicwur_core wpwp_floor): the soil
    # parameters carrying min_liq, built once; the step's soil parameters
    # (_vw_sp0) are these, or the plain ones
    _vw_sp0 = soil_params_t if mosaic is not None else None
    if _vw is not None and _vw.vcs_drain is not None:
        # VIC-CropSyst's drainage law (runoff.c:575-621): Field_Capacity
        # per layer and the irrigated-field flag per row
        if mosaic is None:
            raise NotImplementedError('vicwur_core vcs_drain needs the '
                                      'mosaic path')
        vd_ = _vw.vcs_drain
        fc_ = getattr(soil_params_t, 'vcs_fc_mm', None)
        if fc_ is None:
            raise NotImplementedError('vicwur_core vcs_drain needs the '
                                      'Field_Capacity (saxton_soil_params'
                                      '(vcs_drainage=True))')
        ir_ = vd_.get('irrigated')
        if ir_ is not None:
            ir_ = torch.as_tensor(ir_, dtype=torch.bool, device=device)
            ir_ = (_vw_lane(ir_, ()).reshape(-1, 1) if lanes is not None
                   else ir_.reshape(-1, 1))
        _vw_sp0 = object.__new__(type(soil_params_t))
        for name_ in type(soil_params_t).__slots__:
            if hasattr(soil_params_t, name_):
                setattr(_vw_sp0, name_, getattr(soil_params_t, name_))
        _vw_sp0.vcs_fc = fc_
        _vw_sp0.vcs_irrig = ir_
    if _vw is not None and _vw.wpwp_floor is not None:
        if mosaic is None or soil_params_t.w_wilt_mm is None:
            raise NotImplementedError('vicwur_core wpwp_floor needs the '
                                      "mosaic path and the soil's Wpwp")
        wf_ = _vw.wpwp_floor
        if wf_ is True:
            mk_ = torch.ones(nc * mosaic.n_tiles, 1, dtype=torch.bool,
                             device=device)
        else:
            wf_ = torch.as_tensor(wf_, dtype=torch.bool, device=device)
            if lanes is not None:
                mk_ = _vw_lane(wf_, ()).reshape(-1, 1)
            else:
                mk_ = wf_.reshape(-1, 1)
        rs_ = soil_params_t.resid_moist
        wp_ = soil_params_t.w_wilt_mm.to(rs_.dtype)
        # runoff.c:301-304: layer 0 keeps the residual, the rest Wpwp
        wp_ = torch.cat([rs_[:, :1], wp_[:, 1:]], dim=-1)
        _vw_sp1 = object.__new__(type(soil_params_t))
        for name_ in type(soil_params_t).__slots__:
            if hasattr(_vw_sp0, name_):
                setattr(_vw_sp1, name_, getattr(_vw_sp0, name_))
        _vw_sp0 = _vw_sp1
        _vw_sp0.min_liq = torch.where(mk_, wp_, rs_)

    def _vw_state_get(name_):
        if name_.startswith('snow.'):
            return snow_state.get(name_[5:])
        if name_.startswith('canopy.'):
            return (canopy_state or {}).get(name_[7:])
        return dict(moist=moist, wdew=wdew, last_snow=last_snow,
                    melting=melting, soil_ice=soil_ice, wb_t0=wb_t0,
                    wb_t1=wb_t1, t_node=t_node,
                    ground_long_out=ground_long_out).get(name_)

    def _vw_state_set(name_, v_):
        nonlocal moist, wdew, last_snow, melting, soil_ice, wb_t0, wb_t1, \
            t_node, ground_long_out
        old_ = _vw_state_get(name_)
        if old_ is None:
            raise KeyError('vicwur_core state %r: not carried by this run'
                           % name_)
        v_ = torch.as_tensor(v_, dtype=old_.dtype, device=old_.device)
        if lanes is not None:
            # the columns' value, at the lanes
            shp_ = (lanes.n_cells, lanes.n_tiles) + tuple(old_.shape[2:])
            if tuple(v_.shape) != shp_:
                raise ValueError('vicwur_core state %r: shape %s, the run '
                                 'carries %s' % (name_, tuple(v_.shape),
                                                 shp_))
            v_ = lanes.gather_tile(v_)
        if tuple(v_.shape) != tuple(old_.shape):
            raise ValueError('vicwur_core state %r: shape %s, the run '
                             'carries %s' % (name_, tuple(v_.shape),
                                             tuple(old_.shape)))
        v_ = v_.clone()
        if name_.startswith('snow.'):
            snow_state[name_[5:]] = v_
        elif name_.startswith('canopy.'):
            canopy_state[name_[7:]] = v_
        elif name_ == 'moist':
            moist = v_
        elif name_ == 'wdew':
            wdew = v_
        elif name_ == 'last_snow':
            last_snow = v_
        elif name_ == 'melting':
            melting = v_
        elif name_ == 'soil_ice':
            soil_ice = v_
        elif name_ == 'wb_t0':
            wb_t0 = v_
        elif name_ == 'wb_t1':
            wb_t1 = v_
        elif name_ == 't_node':
            t_node = v_
        elif name_ == 'ground_long_out':
            ground_long_out = v_

    def _vw_state_cols(name_):
        # what a state_fn reads: the columns' value (lanes scattered)
        x_ = _vw_state_get(name_)
        if lanes is None or x_ is None:
            return x_
        return lanes.scatter(x_)

    def _vw_state_names():
        n_ = [k_ for k_ in ('moist', 'wdew', 'last_snow', 'melting',
                            'soil_ice', 'wb_t0', 'wb_t1', 't_node',
                            'ground_long_out')
              if _vw_state_get(k_) is not None]
        n_ += ['snow.' + k_ for k_, v_ in snow_state.items()
               if torch.is_tensor(v_)]
        n_ += ['canopy.' + k_ for k_, v_ in (canopy_state or {}).items()
               if torch.is_tensor(v_)]
        return n_

    def _vw_host(t_):
        # the step's hook values: the plugins' callables, the state edits
        # and the Cv, then each value where the step reads it
        if _vw.step_fn is not None:
            _vw.step_fn(t_, _vw)
        if _vw_veg is not None:
            m_ = _vw_months[t_]
            _vw_veg.apply(_vw, t_, m_)
            if _vw_gm[0] is not None:
                # the graph's month buffers take the step's tables
                for k_ in _vw_veg.tables():
                    if k_ in _vw_gm[1]:
                        getattr(_vw_gm[0], k_).copy_(
                            getattr(_vw_veg.mosaic, k_)[:, :, m_:m_ + 1])
        if _vw.state_fn is not None:
            # the carried tile states before step t, [column, tile, ...]
            _vw.state_fn(t_, _vwcore.TileStates(
                _vw_state_cols if lanes is not None else _vw_state_get,
                _vw_state_set, _vw_state_names))
            if _vw_sync[0] is not None:
                _vw_sync[0]()
        cv_ = _vw.cv_at(t_)
        if cv_ is not None:
            # veg_con[iveg].Cv, the tile weights from this step on (the
            # land-use plugin writes them between steps, VIC-WUR 3.0
            # lu_apply.c:995-1029)
            if lanes is None:
                mosaic.cv = torch.as_tensor(cv_, dtype=dtype, device=device)
            else:
                v_ = _vw_ver(cv_) if torch.is_tensor(cv_) else None
                old_ = _vw_cvsrc[0]
                same_ = old_ is not None and old_[0] is cv_
                if not (same_ and v_ is not None and old_[1] == v_):
                    c_ = torch.as_tensor(cv_, dtype=dtype, device=device)
                    if not same_ and bool(((c_ > 0.0)
                                           & ~_vw_lmask[0]).any()):
                        # (a new Cv: one host read)
                        raise NotImplementedError(
                            'vicwur_core cv covers a tile with no cover at '
                            "the call's start (not one of its lanes); run "
                            'it on the padded grid (TORCHVIC_VW_LANES=0)')
                    mosaic.cv.copy_(lanes.gather_tile(c_))
                    cell_mosaic.cv = c_
                    _vw_cvsrc[0] = (cv_, v_)
        if lanes is None:
            _vw_now.clear()
        wc_ = _vw.wcr_at(t_)
        if wc_ is not None:
            wc_ = torch.as_tensor(wc_, dtype=dtype, device=device)
            if lanes is None:
                _vw_now['wcr'] = wc_
            else:
                _vw_put('wcr', wc_, tuple(wc_.shape[2:]))
        else:
            _vw_now.pop('wcr', None)
        if lanes is None:
            cf_ = _vw.co2factor_at(t_, mosaic.cv)
            if cf_ is not None:
                _vw_now['co2factor'] = cf_
        elif _vw.b_co2 is not None:
            # co2factor_at at the lanes: each lane's tile b_co2 and its
            # column's Catm, elementwise as on the columns
            b_ = _vw_lane(torch.as_tensor(_vw.b_co2, device=device), (),
                          key='b_co2')
            c_ = _vw._at(_vw.catm, t_)
            if c_ is None:
                ppm_ = torch.full_like(mosaic.cv[:, 0], _vw.co2ref
                                       * _vwcore._co2.PPM_TO_MIXRATIO)
            else:
                c_ = torch.as_tensor(c_, dtype=mosaic.cv.dtype,
                                     device=device)
                ppm_ = lanes.gather_cell(c_[t_] if c_.dim() == 2 else c_)
            ppm_ = _vwcore._co2.ppm_from_catm(ppm_)
            cf_ = _vwcore._co2.co2factor(b_, ppm_.unsqueeze(-1), _vw.co2ref)
            b_ = _vw_buf.get('co2factor')
            if b_ is None:
                b_ = _vw_buf['co2factor'] = cf_
            else:
                b_.copy_(cf_)
            _vw_now['co2factor'] = b_
        tv_ = _vw.transp_at(t_)
        if tv_ is None:
            _vw_now.pop('transp_ov', None)
            _vw_now.pop('transp_mask', None)
        else:
            # the override and its mask where the step reads them
            tv_ = torch.as_tensor(tv_, dtype=dtype, device=device)
            tm_ = _vw.transp_mask
            tm_ = (torch.ones(1, 1, dtype=torch.bool, device=device)
                   if tm_ is None else
                   torch.as_tensor(tm_, dtype=torch.bool, device=device))
            if lanes is None:
                _vw_now['transp_ov'] = tv_
                _vw_now['transp_mask'] = tm_
            else:
                _vw_put('transp_ov', tv_, tuple(tv_.shape[2:]))
                _vw_put('transp_mask', tm_, ())
        if _vw.canopy_crop is not None:
            # VIC-CropSyst P7: the tiles whose store the crop evaporates and
            # what it took the day before
            cm_ = torch.as_tensor(_vw.canopy_crop['mask'], dtype=torch.bool,
                                  device=device)
            ct_ = _vw.canopy_take_at(t_)
            if lanes is None:
                _vw_now['canopy_mask'] = cm_
            else:
                _vw_put('canopy_mask', cm_, ())
            if ct_ is None:
                _vw_now.pop('canopy_take', None)
            else:
                ct_ = torch.as_tensor(ct_, dtype=dtype, device=device)
                if lanes is None:
                    _vw_now['canopy_take'] = ct_
                else:
                    _vw_put('canopy_take', ct_, ())
        if _vw.crop_soil_evap is not None:
            cs_ = _vw.crop_soil_evap
            for k_, n_ in (('mask', 'se_mask'), ('cover_green', 'se_green'),
                           ('cover_total', 'se_total')):
                x_ = _vw._at(cs_[k_], t_)
                x_ = torch.as_tensor(x_, dtype=(torch.bool if k_ == 'mask'
                                                else dtype), device=device)
                if lanes is None:
                    _vw_now[n_] = x_
                else:
                    _vw_put(n_, x_, ())
        ks_ = _vw.ksat_at(t_)
        if ks_ is None:
            if _vw_sp0 is not soil_params_t:
                _vw_now['ksat'] = _vw_sp0
            else:
                _vw_now.pop('ksat', None)
        else:
            # VIC-WUR 3.0 runoff.c:156-157 reads layer[l].Ksat of the tile's
            # own column; the column rows are tile k of column c at c*n + k
            ks_ = torch.as_tensor(ks_, dtype=dtype, device=device)
            if lanes is not None:
                ks_ = _vw_put('ksat_t', ks_, tuple(ks_.shape[2:]))
            sp_ = _vw_buf.get('ksat_sp')
            if sp_ is None or lanes is None:
                sp_ = object.__new__(type(soil_params_t))
                for name_ in type(soil_params_t).__slots__:
                    if hasattr(_vw_sp0, name_):
                        setattr(sp_, name_, getattr(_vw_sp0, name_))
                sp_.Ksat = ks_.reshape(nc * mosaic.n_tiles, -1)
                if lanes is not None:
                    _vw_buf['ksat_sp'] = sp_
            _vw_now['ksat'] = sp_

    def _vw_key():
        # what a captured step reads of the hooks
        return tuple(sorted(_vw_now)) if _vw is not None else ()

    def _vw_dev():
        d = {k_: _vw_now[k_] for k_ in ('wcr', 'co2factor', 'transp_ov',
                                         'transp_mask', 'canopy_mask',
                                         'canopy_take', 'se_mask',
                                         'se_green', 'se_total')
             if k_ in _vw_now}
        if _vw.pet_short:
            d['pet_short'] = True
        if _vw_ws is not None:
            d['stress'] = {}
        if _vw.diagnostics:
            d['diag'] = {}
        if _vw.tile_fluxes:
            d['tflux'] = {}
        if _vw.throughfall == 'exact':
            d['w3_tf'] = True
        mosaic.vicwur = d
        _vw_ksat[0] = _vw_now.get('ksat')

    def _vw_before(t_):
        if not _vw_split[0]:
            _vw_host(t_)
        _vw_dev()

    def _vw_after(t_):
        nonlocal _vw_ws
        d = mosaic.vicwur or {}
        if _vw_ws is not None:
            st_ = d.get('stress') or {}
            if st_.get('gsm') is not None:
                ran_ = st_['ran'].unsqueeze(-1)
                _vw_ws = torch.where(ran_, st_['gsm'].expand_as(_vw_ws),
                                     _vw_ws)
            # VIC-WUR 3.0 surface_fluxes.c:1046-1049, in the C's order
            rt_ = mosaic.root_frac.expand_as(_vw_ws)
            cws_ = torch.zeros_like(_vw_ws[..., 0])
            for l_ in range(_vw_ws.shape[-1]):
                cws_ = cws_ + _vw_ws[..., l_] * rt_[..., l_]
            if 'water_stress_layer' in _vw_out:
                _vw_out['water_stress_layer'].append(_vw_ws)
            _vw_out['water_stress'].append(cws_)
        if _vw.diagnostics:
            dg_ = d.get('diag') or {}
            _vw_out['transp_tile'].append(dg_.get('transp'))
            _vw_out['esoil_tile'].append(dg_.get('bare'))
        if _vw.tile_fluxes:
            # the step's per-tile pot_evap, transp, esoil, canopyevap
            # (VegMosaic.step; vicwur.core tile_fluxes)
            _vw_tf[0] = {'tf_' + k_: v_ for k_, v_ in
                         (d.get('tflux') or {}).items()}
        mosaic.vicwur = None
        _vw_ksat[0] = None

    def _step(t_):
        if _vw is not None:
            _vw_before(t_)
        if close_energy and use_eb:
            _ce_step(t_)
        else:
            _step_inner(t_)
        if _vw is not None:
            _vw_after(t_)

    def _harvest(lens):
        # Carbon fluxes are created on the first step that runs them, and
        # VIC reports the average over the snow steps (surface_fluxes.c:1053).
        for k_, lst_ in carbon_out.items():
            nk_ = 'carbon_' + k_
            if nk_ not in _named:
                _named[nk_] = lst_
                _MEAN.add(nk_)
        got = {}
        for k_, lst_ in _named.items():
            n0_ = lens.get(k_, 0)
            if len(lst_) > n0_:
                got[k_] = list(lst_[n0_:])
                del lst_[n0_:]
        return got

    def _combine(parts):
        keys_ = []
        for p_ in parts:
            for k_ in p_:
                if k_ not in keys_:
                    keys_.append(k_)
        for k_ in keys_:
            rows_ = [p_[k_] for p_ in parts if k_ in p_]
            n_ = len(rows_[0])
            if any(len(r_) != n_ for r_ in rows_):
                raise RuntimeError('snow sub-step changed how many %s '
                                   'values a step records' % k_)
            # The column runs on the last snow step only, so its rows
            # exist in that part alone and are already the day's.
            if len(rows_) != len(parts):
                _named[k_].extend(rows_[-1])
                continue
            kind_ = ('sum' if k_ in _SUM else
                     'mean' if k_ in _MEAN else 'last')
            for i_ in range(n_):
                xs_ = [r_[i_] for r_ in rows_]
                if kind_ == 'last' or not all(torch.is_tensor(x_)
                                              for x_ in xs_):
                    _named[k_].append(xs_[-1])
                    continue
                acc_ = xs_[0]
                for x_ in xs_[1:]:
                    acc_ = acc_ + x_
                if kind_ == 'mean':
                    acc_ = acc_ / float(len(xs_))
                _named[k_].append(acc_)

    def _ce_seed_after(rows_, nf_):
        # surface_fluxes.c:983-985, :990-994: the step's closing energy is
        # the snow steps' mean, except snow_flux, which is the mean only
        # where the last snow step had snow->snow or INCLUDE_SNOW and the
        # last step's own elsewhere. That closing energy seeds the next
        # step's iteration (:225-230).
        nonlocal ce_seed
        n_ = float(nf_)
        mean_ = rows_[0]['seed']
        for r_ in rows_[1:]:
            mean_ = mean_ + r_['seed']
        mean_ = mean_ / n_
        snow_ = torch.where(rows_[-1]['snow_or_include'],
                            mean_[..., 0], rows_[-1]['seed'][..., 0])
        ce_seed = torch.stack((snow_, mean_[..., 1], mean_[..., 2],
                               mean_[..., 3]), dim=-1)

    def _snow_terms_after(rows_, nf_, day_dt_):
        # surface_fluxes.c:990-994: advection, deltaCC and refreeze_energy
        # are the snow steps' mean only where the last snow step had
        # snow->snow or INCLUDE_SNOW. Elsewhere they are soil_energy's as
        # the last snow step left them: what the last thin-pack step set
        # (func_surf_energy_bal.c:617-628, surface_fluxes.c:578), or the 0
        # of :225-233. put_data.c:493-494 forms OUT_REFREEZE from it.
        if len(rows_) != nf_ or not nf_:
            return
        idle_ = ~rows_[-1]['active']
        if not (graphsafe.ACTIVE or bool(idle_.any())):
            return
        n_ = float(nf_)
        for k_, nk_ in (('advection', 'eb_advection'),
                        ('deltacc', 'eb_deltacc'),
                        ('rfrz_energy', 'eb_rfrz_energy')):
            if nk_ not in _named or not _named[nk_]:
                continue
            mean_ = rows_[0]['rows'][k_]
            for r_ in rows_[1:]:
                mean_ = mean_ + r_['rows'][k_]
            mean_ = mean_ / n_
            kept_ = torch.zeros_like(mean_)
            for r_ in rows_:
                kept_ = torch.where(r_['thin'], r_['rows'][k_], kept_)
            d_ = _tsum(torch.where(idle_, kept_ - mean_,
                                   torch.zeros_like(mean_)) * mosaic.cv)
            _named[nk_][-1] = _named[nk_][-1] + d_
            # the understory's own OUT_ADVECTION_BAND / OUT_RFRZ_ENERGY_BAND
            # (put_data.c:889-905) take the same correction, and none of the
            # canopy's (that is _canopy_after's)
            _bk = {'advection': 'band_adv', 'rfrz_energy': 'band_rfz'}.get(k_)
            if _bk is not None and _named.get(_bk):
                _named[_bk][-1] = _named[_bk][-1] + d_
            if k_ == 'rfrz_energy' and _named.get('eb_refreeze'):
                _named['eb_refreeze'][-1] = (_named['eb_refreeze'][-1]
                                             + d_ / snow.CONST_LATICE
                                             * day_dt_)
        if ('snow_flux' in rows_[-1]['rows'] and 'ebex_snow_flux' in _named
                and _named['ebex_snow_flux']):
            # surface_fluxes.c:990-994 covers the ground flux too: where
            # the last snow step has no snow->snow (and no INCLUDE_SNOW)
            # the reported value is the soil flux that step left, not
            # the steps' mean. A snow-free step leaves 0
            # (func_surf_energy_bal.c:353-366 sets it on every ground
            # evaluation; calc_surf_energy_bal.c:700-703 only fires
            # with a pack).
            mean_ = rows_[0]['rows']['snow_flux']
            for r_ in rows_[1:]:
                mean_ = mean_ + r_['rows']['snow_flux']
            mean_ = mean_ / n_
            last_ = rows_[-1]['rows']['snow_flux']
            d_ = _tsum(torch.where(idle_, last_ - mean_,
                                   torch.zeros_like(mean_)) * mosaic.cv)
            _named['ebex_snow_flux'][-1] = _named['ebex_snow_flux'][-1] + d_

    def _canopy_after(rows_, nf_, day_dt_):
        # surface_fluxes.c:978-982: energy->canopy_* become the snow steps'
        # mean, which the next step carries; put_data.c:919-931 reports
        # them only where the last snow step ran the snow branch.
        if len(rows_) != nf_ or not nf_:
            return
        n_ = float(nf_)
        adv_ = rows_[0]['adv']
        rfz_ = rows_[0]['rfz']
        for r_ in rows_[1:]:
            adv_ = adv_ + r_['adv']
            rfz_ = rfz_ + r_['rfz']
        adv_ = adv_ / n_
        rfz_ = rfz_ / n_
        canopy_state['carry_adv'] = adv_
        canopy_state['carry_rfz'] = rfz_
        idle_ = ~rows_[-1]['on']
        if not (graphsafe.ACTIVE or bool(idle_.any())):
            return
        z_ = torch.zeros_like(adv_)
        d_a = _tsum(torch.where(idle_, adv_, z_) * mosaic.cv)
        d_r = _tsum(torch.where(idle_, rfz_, z_) * mosaic.cv)
        if _named.get('eb_advection'):
            _named['eb_advection'][-1] = _named['eb_advection'][-1] - d_a
        if _named.get('eb_rfrz_energy'):
            _named['eb_rfrz_energy'][-1] = _named['eb_rfrz_energy'][-1] - d_r
        if _named.get('eb_refreeze'):
            _named['eb_refreeze'][-1] = (_named['eb_refreeze'][-1]
                                         - d_r / snow.CONST_LATICE * day_dt_)

    def _radiation_after(rows_, nf_):
        # surface_fluxes.c:960-965 average AlbedoOver, AlbedoUnder,
        # LongOverIn and LongUnderIn over the snow steps (solve_snow.c
        # :475, :501-504 zero the overstory pair on a step without snow);
        # put_data.c:877-890 reports the overstory pair where the last
        # snow step had snow, the understory pair elsewhere.
        if len(rows_) != nf_ or not nf_:
            return
        n_ = float(nf_)
        m_ = {}
        for k_ in ('alb_over', 'alb_under', 'lw_over', 'lw_under'):
            a_ = rows_[0][k_]
            for r_ in rows_[1:]:
                a_ = a_ + r_[k_]
            m_[k_] = a_ / n_
        on_ = rows_[-1]['on']
        if _named.get('albedo'):
            _named['albedo'][-1] = _tsum(torch.where(
                on_, m_['alb_over'], m_['alb_under']) * mosaic.cv)
        if _named.get('in_long'):
            _named['in_long'][-1] = _tsum(torch.where(
                on_, m_['lw_over'], m_['lw_under']) * mosaic.cv)

    def _clone(x_):
        if torch.is_tensor(x_):
            return x_.clone()
        if isinstance(x_, dict):
            return {k_: _clone(v_) for k_, v_ in x_.items()}
        return x_

    def _snap():
        return dict(
            moist=_clone(moist), last_snow=_clone(last_snow),
            melting=_clone(melting), wdew=_clone(wdew),
            snow_state=_clone(snow_state), soil_ice=_clone(soil_ice),
            sp=_clone(sp), t_node=_clone(t_node),
            wb_t0=_clone(wb_t0), wb_t1=_clone(wb_t1),
            ground_long_out=_clone(ground_long_out),
            canopy_long_out=_clone(canopy_long_out),
            canopy=_clone(canopy_state) if any_over else None,
            carbon=_clone(carbon_state), ce_seed=_clone(ce_seed),
            snow_zero=_clone(_snow_zero))

    def _restore(s_):
        nonlocal moist, last_snow, melting, wdew, snow_state, soil_ice, \
            sp, t_node, wb_t0, wb_t1, ground_long_out, canopy_long_out, \
            ce_seed
        moist, last_snow = s_['moist'], s_['last_snow']
        ce_seed = s_['ce_seed']
        melting, wdew = s_['melting'], s_['wdew']
        snow_state = s_['snow_state']
        soil_ice, sp, t_node = s_['soil_ice'], s_['sp'], s_['t_node']
        wb_t0, wb_t1 = s_['wb_t0'], s_['wb_t1']
        ground_long_out = s_['ground_long_out']
        canopy_long_out = s_['canopy_long_out']
        if any_over and s_['canopy'] is not None:
            canopy_state.clear()
            canopy_state.update(s_['canopy'])
        if s_['carbon'] is not None and carbon_state is not None:
            carbon_state.clear()
            carbon_state.update(s_['carbon'])
        _snow_zero.clear()
        _snow_zero.update(s_['snow_zero'])

    def _where_cell(mask_, a_, b_):
        if not torch.is_tensor(a_) or not torch.is_tensor(b_):
            return a_ if a_ is not None else b_
        if a_.shape != b_.shape or a_.shape[:1] != mask_.shape[:1]:
            raise RuntimeError('snow-day blend: %s against %s, mask %s'
                               % (tuple(a_.shape), tuple(b_.shape),
                                  tuple(mask_.shape)))
        if mask_.dim() == 2:
            # a per-tile choice (surface_fluxes.c:258-269 decides per tile);
            # a cell-level value follows the cell's snow tiles
            if a_.dim() >= 2 and a_.shape[1] == mask_.shape[1]:
                m_ = mask_.view(mask_.shape + (1,) * (a_.dim() - 2))
                return torch.where(m_, a_, b_)
            mask_ = mask_.any(1)
        m_ = mask_.view(mask_.shape + (1,) * (a_.dim() - 1))
        return torch.where(m_, a_, b_)

    def _blend_state(mask_, snow_s_):
        nonlocal moist, last_snow, melting, wdew, snow_state, soil_ice, \
            sp, t_node, wb_t0, wb_t1, ground_long_out, canopy_long_out, \
            ce_seed
        moist = _where_cell(mask_, snow_s_['moist'], moist)
        if ce_seed is not None and snow_s_['ce_seed'] is not None:
            ce_seed = _where_cell(mask_, snow_s_['ce_seed'], ce_seed)
        last_snow = _where_cell(mask_, snow_s_['last_snow'], last_snow)
        melting = _where_cell(mask_, snow_s_['melting'], melting)
        wdew = _where_cell(mask_, snow_s_['wdew'], wdew)
        snow_state = {k_: _where_cell(mask_, snow_s_['snow_state'][k_], v_)
                      for k_, v_ in snow_state.items()}
        if soil_ice is not None:
            soil_ice = _where_cell(mask_, snow_s_['soil_ice'], soil_ice)
        if sp is not None and snow_s_['sp'] is not None:
            sp = {k_: _where_cell(mask_, snow_s_['sp'][k_], v_)
                  for k_, v_ in sp.items()}
        if t_node is not None:
            t_node = _where_cell(mask_, snow_s_['t_node'], t_node)
        if wb_t0 is not None:
            wb_t0 = _where_cell(mask_, snow_s_['wb_t0'], wb_t0)
            wb_t1 = _where_cell(mask_, snow_s_['wb_t1'], wb_t1)
        if ground_long_out is not None:
            ground_long_out = _where_cell(
                mask_, snow_s_['ground_long_out'], ground_long_out)
        if canopy_long_out is not None:
            canopy_long_out = _where_cell(
                mask_, snow_s_['canopy_long_out'], canopy_long_out)
        if any_over and snow_s_['canopy'] is not None:
            for k_ in canopy_state:
                canopy_state[k_] = _where_cell(
                    mask_, snow_s_['canopy'][k_], canopy_state[k_])

    def _blend_outputs(mask_, snow_rows_):
        for k_, rows_ in snow_rows_.items():
            lst_ = _named[k_]
            n_ = len(rows_)
            for i_ in range(n_):
                d_ = lst_[-n_ + i_]
                s_ = rows_[i_]
                if torch.is_tensor(s_) and torch.is_tensor(d_):
                    lst_[-n_ + i_] = _where_cell(mask_, s_, d_)

    def _sd_at(k_, *idx_):
        # The sub-daily records are [day, snow step, cell]. On the lanes
        # they are read as the other forcing is: the step's row gathered to
        # the lanes (Lanes.gather_cell, as LaneAtmosphere.at), so each tile
        # takes its cell's record (surface_fluxes.c:258-269 reads the
        # cell's snowflag[NR] for every tile).
        x_ = subdaily[k_][idx_]
        return x_ if lanes is None else x_.index_select(-1, lanes.lane_cell)

    def _snow_substeps(t_):
        """NF surface steps, one column step (surface_fluxes.c:258-269)."""
        nonlocal step_seconds, wb_thin, ground_long_out
        saved_dt_ = step_seconds
        saved_thin_ = wb_thin
        nf_ = int(subdaily['tair'].shape[1])
        sum_tf_ = None
        sum_ev_ = None
        parts_ = []
        glo_ = []
        pushed_ = None
        sf_acc_ = None
        sf_prev_ = _vic_aux.get('snow_flux') if return_state else None
        try:
            # The thin-pack branch was measured at a sub-daily temperature.
            # A snow-free day leaves wb_thin as the daily-mean decision.
            wb_thin = (wb_mode and per_tile_snow)
            _sub['pin_tgrnd'] = True
            _sub['t_grnd'] = None
            _sub['atten_one'] = None
            _sub['eb_rows'] = []
            _sub['can_rows'] = []
            _sub['alb_rows'] = []
            _sub['soil_adv'] = _sub['soil_dcc'] = None
            _sub['model_dt'] = saved_dt_
            _sub['ce_seed_rows'] = []
            _sub['ce_carry'] = None
            for h_ in range(nf_):
                # on the lanes atm[k] is a LaneAtmosphere series, not a tensor
                over_ = {k_: _sd_at(k_, t_, h_) for k_ in subdaily
                         if k_ in atm and torch.is_tensor(subdaily[k_])
                         and (torch.is_tensor(atm[k_]) if lanes is None
                              else atm.has_series(k_))}
                pushed_ = {}
                for k_, v_ in over_.items():
                    pushed_[k_] = atm[k_]
                    atm[k_] = _At(v_)
                step_seconds = SEC_PER_DAY / float(nf_)
                _sub['flush'] = (h_ == 0)
                _sub['divide_fcan'] = (h_ == 0)
                last_ = h_ == nf_ - 1
                _sub['add_tf'] = None if not last_ else sum_tf_
                _sub['add_ev'] = None if not last_ else sum_ev_
                _sub['skip'] = not last_
                lens_ = {k_: len(v_) for k_, v_ in _named.items()}
                _step(t_)
                parts_.append(_harvest(lens_))
                glo_.append(ground_long_out)
                if return_state and 'snow_flux' in _vic_aux:
                    sf_acc_ = (_vic_aux['snow_flux'] if sf_acc_ is None
                               else sf_acc_ + _vic_aux['snow_flux'])
                if not last_:
                    if _sub['cap_tf'] is None:
                        raise RuntimeError('a snow sub-step did not reach '
                                           'the soil column')
                    sum_tf_ = (_sub['cap_tf'] if sum_tf_ is None
                               else sum_tf_ + _sub['cap_tf'])
                    sum_ev_ = (_sub['cap_ev'] if sum_ev_ is None
                               else sum_ev_ + _sub['cap_ev'])
                for k_, v_ in pushed_.items():
                    atm[k_] = v_
                pushed_ = None
            _combine(parts_)
            _snow_terms_after(_sub['eb_rows'], nf_, saved_dt_)
            if any_over:
                _canopy_after(_sub['can_rows'], nf_, saved_dt_)
                _radiation_after(_sub['alb_rows'], nf_)
            if glo_ and all(torch.is_tensor(x_) for x_ in glo_):
                # surface_fluxes.c:888, :967: the next model step's first
                # snow step reads LongUnderOut from energy, the snow steps'
                # mean (:520 with soil_energy = *energy, :236).
                acc_ = glo_[0]
                for x_ in glo_[1:]:
                    acc_ = acc_ + x_
                ground_long_out = acc_ / float(len(glo_))
                if return_state:
                    _vic_aux['long_under_out'] = ground_long_out
            if return_state and sf_acc_ is not None:
                # surface_fluxes.c:922-935, :1003-1006: the mean over the
                # snow steps, kept only where the last of them had snow
                _mean_ = sf_acc_ / float(nf_)
                _vic_aux['snow_flux'] = (
                    _mean_ if sf_prev_ is None else
                    torch.where(_vic_aux['snow_on'], _mean_, sf_prev_))
            if use_eb and eb_extra['aero_cond'] and eb_out['aero_resist']:
                # surface_fluxes.c:1023-1025: the resistance is the inverse
                # of the snow steps' mean conductance, not the mean of the
                # inverses.
                _c = eb_extra['aero_cond'][-1]
                eb_out['aero_resist'][-1] = 1.0 / torch.clamp(_c, min=1e-30)
            if wb_mode and wb_x and wb_x['aero_cond'] and wb_e['aero_resist']:
                # the same inverse of the snow steps' mean conductance
                wb_e['aero_resist'][-1] = 1.0 / torch.clamp(
                    wb_x['aero_cond'][-1], min=1e-30)
            if (close_energy and use_eb
                    and len(_sub['ce_seed_rows']) == nf_ and nf_):
                _ce_seed_after(_sub['ce_seed_rows'], nf_)
        finally:
            if pushed_:
                for k_, v_ in pushed_.items():
                    atm[k_] = v_
            step_seconds = saved_dt_
            wb_thin = saved_thin_
            _sub['flush'] = True
            _sub['skip'] = False
            _sub['divide_fcan'] = True
            _sub['pin_tgrnd'] = False
            _sub['t_grnd'] = None
            _sub['atten_one'] = None
            _sub['eb_rows'] = []
            _sub['can_rows'] = []
            _sub['alb_rows'] = []
            _sub['soil_adv'] = _sub['soil_dcc'] = None
            _sub['add_tf'] = None
            _sub['add_ev'] = None
            _sub['ce_seed_rows'] = []
            _sub['ce_carry'] = None

    def _snow_mask(t_):
        # forcing_utils.c:86-100, snowflag[NR], plus a pack or canopy snow
        # already on the ground (surface_fluxes.c:258-262). t_offset is 0
        # with SNOW_BAND FALSE (vic_force.c:389-390).
        ta_ = _sd_at('tair', t_)
        pr_ = _sd_at('prcp', t_)
        flag_ = ((ta_ < p['t_max_snow']) & (pr_ > 0)).any(0)
        swq_ = (snow_state['ice'] + snow_state['surf_water']
                + snow_state['pack_water'])
        if swq_.dim() > 1:
            swq_ = swq_.reshape(swq_.shape[0], -1).sum(-1)
        has_ = swq_ > 0
        if any_over:
            cs_ = canopy_state['int_snow']
            if cs_.dim() > 1:
                cs_ = cs_.reshape(cs_.shape[0], -1).sum(-1)
            has_ = has_ | (cs_ > 0)
        return has_ | flag_

    _tile_split = (mosaic is not None and per_tile_snow)

    def _cell_any(x_):
        # a [cell, tile] mask reduced to its cell (on the lanes: each lane
        # holds its cell's value, as tsum hands a cell sum back)
        if lanes is None:
            return x_.any(1)
        return lanes.tsum(x_.to(dtype)) > 0.0


    def _tile_mask(t_):
        # surface_fluxes.c:258-269, per tile: snow->swq, snow->snow_canopy,
        # or the cell's snowflag[NR]
        ta_ = _sd_at('tair', t_)
        pr_ = _sd_at('prcp', t_)
        flag_ = ((ta_ < p['t_max_snow']) & (pr_ > 0)).any(0)
        has_ = (snow_state['ice'] + snow_state['surf_water']
                + snow_state['pack_water']) > 0
        if any_over:
            has_ = has_ | (canopy_state['int_snow'] > 0)
        return (has_ | flag_.unsqueeze(1)) & (mosaic.cv > 0.0)

    def _split_rows(rows_a, rows_b, w_a, w_b, hp_a, hp_b):
        # The cell's rows from the two runs, each made over its own tiles'
        # Cv (put_data.c sums tile by tile with AreaFactor). Rows that are a
        # cell value, or a ratio of tile sums, are recombined from their
        # weights; a cell all on one side keeps that side's row as it is.
        cv_ = mosaic.cv
        ws_ = {}

        def _w(key_, sel_a, sel_b):
            if key_ not in ws_:
                ws_[key_] = (_tsum(w_a * sel_a.to(w_a.dtype)),
                             _tsum(w_b * sel_b.to(w_b.dtype)))
            return ws_[key_]

        bare_ = (mosaic.bare_tile.expand_as(cv_)
                 if mosaic.bare_tile is not None
                 else torch.zeros_like(cv_, dtype=torch.bool))
        all_ = torch.ones_like(cv_, dtype=torch.bool)
        cell_valued = {'rain', 'snowfall', 'prcp'}
        if not any_over:
            cell_valued.add('in_long')
        ratio = {'salbedo': ('hp', hp_a, hp_b),
                 'snow_surf_temp': ('hp', hp_a, hp_b),
                 'snow_pack_temp': ('hp', hp_a, hp_b),
                 'ebex_vegt': ('veg', ~bare_, ~bare_),
                 'ebex_baresoilt': ('bare', bare_, bare_)}
        sa_, sb_ = _w('all', all_, all_)
        for k_ in set(rows_a) | set(rows_b):
            ra_ = rows_a.get(k_, [])
            rb_ = rows_b.get(k_, [])
            n_ = max(len(ra_), len(rb_))
            for i_ in range(n_):
                a_ = ra_[i_] if i_ < len(ra_) else None
                b_ = rb_[i_] if i_ < len(rb_) else None
                if not (torch.is_tensor(a_) and torch.is_tensor(b_)):
                    _named[k_].append(a_ if a_ is not None else b_)
                    continue
                if k_ in cell_valued or k_ == 'in_long':
                    # in_long from the snow steps is already a tile sum
                    # where an overstory made it one (_radiation_after)
                    if k_ in cell_valued:
                        mix_ = a_ * sa_ + b_ * sb_
                    else:
                        mix_ = a_ * sa_ + b_
                    wa_, wb_ = sa_, sb_
                elif k_ in ratio:
                    nm_, s_a, s_b = ratio[k_]
                    wa_, wb_ = _w(nm_, s_a, s_b)
                    tot_ = wa_ + wb_
                    mix_ = (a_ * wa_ + b_ * wb_) / torch.where(
                        tot_ > 0.0, tot_, torch.ones_like(tot_))
                elif k_ == 'ebex_rad_temp':
                    wa_, wb_ = sa_, sb_
                    mix_ = (a_ ** 4 + b_ ** 4) ** 0.25
                elif k_ == 'ebout_aero_resist':
                    wa_, wb_ = sa_, sb_
                    c_a = rows_a['ebex_aero_cond'][i_]
                    c_b = rows_b['ebex_aero_cond'][i_]
                    mix_ = 1.0 / torch.clamp(c_a + c_b, min=1e-30)
                else:
                    _named[k_].append(a_ + b_)
                    continue
                _named[k_].append(torch.where(
                    wb_ <= 0.0, a_, torch.where(wa_ <= 0.0, b_, mix_)))

    def _blend_aux(mask_, aux_b_):
        # the state file's last-step stores follow the same tile/cell split
        # as the state: the snow tiles take the sub-stepped day's
        if not return_state:
            return
        for k_, vb_ in aux_b_.items():
            va_ = _vic_aux.get(k_)
            if (torch.is_tensor(vb_) and torch.is_tensor(va_)
                    and va_.shape == vb_.shape):
                m_ = mask_
                while m_.dim() < va_.dim():
                    m_ = m_.unsqueeze(-1)
                _vic_aux[k_] = torch.where(m_, vb_, va_)

    def _one_day(t_):
        # VIC decides per tile (surface_fluxes.c:258-269). A day no tile
        # snows on is the daily-mean step, bit for bit; a day every tile
        # does is only the sub-step. A day with cells that hold both runs
        # each over its own tiles' Cv and adds the cells' rows.
        if _tile_split:
            tm_ = _tile_mask(t_)
            live_ = mosaic.cv > 0.0
            if not bool(tm_.any()):
                _step(t_)
                return
            if bool((tm_ | ~live_).all()):
                _snow_substeps(t_)
                return
            mixed_ = _cell_any(tm_) & _cell_any(live_ & ~tm_)
            if bool(mixed_.any()):
                cv0_ = mosaic.cv
                opening_ = _snap()
                lens_ = {k_: len(v_) for k_, v_ in _named.items()}
                try:
                    mosaic.cv = cv0_ * tm_.to(cv0_.dtype)
                    _sub['tile_w'] = tm_
                    _m0 = _tfx.mark()
                    _snow_substeps(t_)
                    _tfx.restrict(_m0, tm_)
                    snow_end_ = _snap()
                    aux_b_ = dict(_vic_aux)
                    rows_b = _harvest(lens_)
                    w_b = mosaic.cv
                    _restore(opening_)
                    mosaic.cv = cv0_ * (live_ & ~tm_).to(cv0_.dtype)
                    _sub['tile_w'] = live_ & ~tm_
                    _m0 = _tfx.mark()
                    _step(t_)
                    _tfx.restrict(_m0, live_ & ~tm_)
                    rows_a = _harvest(lens_)
                    w_a = mosaic.cv
                    hp_a = (snow_state['ice'] + snow_state['surf_water']
                            + snow_state['pack_water']) > 0.0
                    _sb = snow_end_['snow_state']
                    hp_b = (_sb['ice'] + _sb['surf_water']
                            + _sb['pack_water']) > 0.0
                finally:
                    mosaic.cv = cv0_
                    _sub['tile_w'] = None
                _split_rows(rows_a, rows_b, w_a, w_b, hp_a, hp_b)
                _blend_state(tm_, snow_end_)
                _blend_aux(tm_, aux_b_)
                return
            mask_ = _cell_any(tm_)
        else:
            mask_ = _snow_mask(t_)
        if not bool(mask_.any()):
            _step(t_)
            return
        if bool(mask_.all()):
            _snow_substeps(t_)
            return
        opening_ = _snap()
        lens_ = {k_: len(v_) for k_, v_ in _named.items()}
        _m0 = _tfx.mark()
        _snow_substeps(t_)
        _tfx.restrict(_m0, mask_)       # the cells whose snow steps count
        snow_end_ = _snap()
        aux_b_ = dict(_vic_aux)
        snow_rows_ = _harvest(lens_)
        _restore(opening_)
        _m0 = _tfx.mark()
        _step(t_)
        _tfx.restrict(_m0, ~mask_)
        _blend_outputs(mask_, snow_rows_)
        _blend_state(mask_, snow_end_)
        _blend_aux(mask_.unsqueeze(-1) if mosaic is not None else mask_,
                   aux_b_)

    graph_mode = (GRAPHS and lanes is not None
                  and (device.type == 'cuda' or SYNC_CHECK)
                  and CANOPY_COMPACT and SNOW_COMPACT and per_tile_snow
                  and _implicit_mod.DEFAULT_METHOD == 'illinois'
                  and not progress and nt > 1 and subdaily is None
                  # The iteration re-runs the step a data-dependent number
                  # of times per tile; a captured graph cannot hold it.
                  and not close_energy
                  # a failed solve is asked of the device, which a graph
                  # cannot do
                  and tfallback
                  # QUICK_FLUX FALSE: the profile's frozen nodes are a root
                  # search inside every trial of the surface's (nested
                  # host-decided loops), and QUICK_SOLVE groups the
                  # columns by a node count read from the device: both
                  # stay eager
                  and not (use_eb and not quick_flux
                           and (frozen_soil or quick_solve))
                  # the hooks' host part (the plugins' callables, a
                  # state_fn) runs once a day, before the day's snapshot
                  and (_vw is None or steps_per_day == 1))
    _season_buf = None
    _phase.mark('setup_s')
    _lake_baresoilt = []

    def _lake_after(t_):
        # vic_run.c 5.1.0:368-466 after the land step, then put_data.c
        # 5.1.0:148-400's lake and wetland terms on the step's rows
        nonlocal moist, wdew, snow_state, canopy_state
        _lake_tap['lat'] = batch.latitude
        land_, P_ = lakes.after(
            t_, _lake_tap, dict(moist=moist, wdew=wdew, snow_state=snow_state,
                                canopy_state=canopy_state if any_over
                                else None, soil_ice=soil_ice),
            atm, p, soil_params, wind_h_lake, ydays[t_], steps_per_day,
            mosaic, months[t_])
        moist, wdew, snow_state = (land_['moist'], land_['wdew'],
                                   land_['snow_state'])
        if any_over:
            canopy_state = land_['canopy_state']
        ren_ = {'eb_net_short': 'ebout_net_short',
                'eb_net_long': 'ebout_net_long',
                'eb_net_rad': 'ebout_net_rad', 'eb_latent': 'ebout_latent',
                'eb_sensible': 'ebout_sensible',
                'eb_grnd_flux': 'ebout_grnd_flux',
                'eb_delta_h': 'ebout_delta_h', 'eb_t_surf': 'ebout_t_surf',
                'eb_layer_temp': 'ebout_layer_temp',
                'eb_snow_flux': 'ebex_snow_flux',
                '_aero_cond': 'ebex_aero_cond', 'vegt': 'ebex_vegt',
                'refreeze': 'eb_refreeze'}
        for k_, d_ in P_.items():
            if k_.startswith('%'):
                # a ratio row re-weighted: (old denominator, new
                # denominator, numerator term to add)
                lst_ = _named.get(ren_.get(k_[1:], k_[1:]))
                if lst_:
                    d0_, d1_, a_ = d_
                    r_ = lst_[-1]
                    n_ = torch.where(d0_ > 0.0, r_ * d0_, r_) + a_
                    lst_[-1] = torch.where(d1_ > 0.0, n_ / torch.where(
                        d1_ > 0.0, d1_, torch.ones_like(d1_)), n_)
                continue
            if k_ in ('_rad4', '_baresoilt'):
                continue
            lst_ = _named.get(ren_.get(k_, k_))
            if lst_ is None or not lst_ or lst_[-1] is None:
                continue
            lst_[-1] = lst_[-1] + d_
        if '_aero_cond' in P_ and _named.get('ebout_aero_resist'):
            _named['ebout_aero_resist'][-1] = 1.0 / torch.clamp(
                _named['ebex_aero_cond'][-1], min=1e-30)
        if '_rad4' in P_ and _named.get('ebex_rad_temp'):
            r_ = _named['ebex_rad_temp'][-1]
            _named['ebex_rad_temp'][-1] = (r_ * r_ * r_ * r_
                                           + P_['_rad4']) ** 0.25
        if '_baresoilt' in P_:
            _lake_baresoilt.append(P_['_baresoilt'])
        # put_data.c 5.1.0:113-121: OUT_PREC/RAINF/SNOWF are force->out_*,
        # which vic_run.c:438-441 adds the lake's share to, unsplit by the
        # land's trace-snow cut (solve_snow.c:139-143)
        dg_ = lakes.last['diag']
        cvf_ = torch.zeros(nc, dtype=dtype, device=device)
        cvf_[lakes.cells] = lakes.cv_lake * lakes.last['lakefrac']
        for nm_, v_ in (('rain', dg_['rainfall']), ('snowfall',
                                                     dg_['snowfall'])):
            full_ = _named[nm_][-1].clone()
            full_[lakes.cells] = v_
            _named[nm_][-1] = _named[nm_][-1] + cvf_ * (full_
                                                        - _named[nm_][-1])
        _named['prcp'][-1] = _named['rain'][-1] + _named['snowfall'][-1]

    if lakes is not None:
        lakes.reset()
        if initial_state is not None:
            # vic_restore.c 5.1.0 lake section / TorchVIC's own state
            if initial_state.get('lake') is not None:
                lakes.set_state(initial_state['lake'])
            elif initial_state.get('lake_vic') is not None:
                lakes.set_state(lakes.state_from_vic(initial_state['lake_vic']))
            else:
                raise ValueError('initial_state holds no lake state but this '
                                 'run has lakes')
        if not use_eb:
            raise ValueError('LAKES TRUE needs FULL_ENERGY TRUE '
                             '(get_global_param.c 5.1.0:980)')
        if subdaily is not None or graph_mode or lanes is not None:
            raise ValueError('LAKES TRUE runs at the sub-daily step on the '
                             'eager path only')
        wind_h_lake = float(lakes.wind_h)
    # simulate's outputs, as a function of the steps they cover (see
    # _tail): the whole call at its end; with day_outputs also each day,
    # from the day hook, as a one-day call would give it
    _tail_rows, _tail_pre, _tail_uh, _tail_st = {}, {}, {}, {}
    # the graph path's failed-assumption flag (defer_day_check)
    _flag_ref = [None]

    def _tail_static(key, fn):
        # a check on the call's fixed inputs (one host read a call)
        if key not in _tail_st:
            _tail_st[key] = fn()
        return _tail_st[key]

    def _tail(lo, hi, batch, atm, mosaic, nc, cvl, moist0_cell, wdew0,
              routing_prefix, copy_state=False):
        """The outputs of steps [lo, hi) of this call (the whole call: lo =
        0, hi = nt), from the per-step lists and the loop's state at hi.
        A part of the call (day_outputs: one day, from the day hook) gives
        what a call over that part alone, started from the state the call
        had at lo, gives: its DEL terms and water error open on the rows
        of step lo - 1, its routing on the tail the steps before lo left,
        and its state is copied (the loop goes on writing into its own)."""
        full = lo == 0 and hi == nt
        S = (lambda l_: l_) if full else (lambda l_: l_[lo:hi])
        if _vw is not None:
            # torchvic.vicwur.core: the file's vegetation back in the mosaic,
            # and the hooks' rows, [step, column, tile(, layer)]
            if _vw_veg is not None:
                _vw_veg.restore()
            mosaic.vicwur = None
            _vw_res = {}
            for k_, v_ in _vw_out.items():
                if v_ and all(x_ is not None for x_ in v_):
                    _vw_res[k_] = _vw_cols(torch.stack(v_), steps=True)
            if _vw_ws is not None:
                _vw_res['water_stress_layer_end'] = _vw_cols(_vw_ws)
            if _vw_tf[0]:
                _vw_res['tile_fluxes_end'] = {k_: _vw_cols(v_) for k_, v_
                                              in _vw_tf[0].items()}
        if lanes is not None:
            # Back to the cell axis for the routing and the outputs.
            _flush_lanes()
            soil_moisture_end = lanes.to_cell(_tsum(moist * cvl))
            canopy_storage_end = lanes.to_cell(_tsum(wdew * mosaic.cv))
            moist_tile_end = lanes.scatter(moist)
            moist0_cell = lanes.to_cell(moist0_cell)
            wdew0 = lanes.to_cell(wdew0)
            batch, atm, mosaic, nc = cell_batch, cell_atm, cell_mosaic, cell_nc
            cvl = mosaic.cv.unsqueeze(-1)
        else:
            soil_moisture_end = ((moist * cvl).sum(1) if mosaic is not None
                                 else moist)
            canopy_storage_end = ((wdew * mosaic.cv).sum(1)
                                  if mosaic is not None else None)
            moist_tile_end = moist
        runoff = torch.stack(S(runoffs))
        baseflow = torch.stack(S(baseflows))
        if _tail_uh:
            uh = _tail_uh['uh']
        else:
            uh = _tail_uh['uh'] = gamma_unit_hydrograph(
                60, p['uh_shape'], p['uh_scale'], dtype, device)
        # route() scatters with index_add, which sums the sources feeding an
        # outlet. That is right for a volume flux and wrong for a depth: summing
        # 13 cells of mm gives 13 times the basin-mean depth. The cells of a
        # 1/16 degree grid are equal area, so the basin mean is the sum over the
        # count. A gauge measures a basin-integrated depth, so this is what the
        # discharge observation is comparable with.
        # route() scatters with index_add, which sums the sources feeding an
        # outlet. That is right for a volume flux and wrong for a depth. The
        # basin-mean depth is the area-weighted mean, not the plain mean: cells on
        # a lat/lon grid are not equal area, and treating them as equal biases a
        # basin mean by up to 0.5 per cent here. Weighting each source by its
        # share of its own basin's area turns the sum route() already performs
        # into exactly that weighted mean.
        if 'share' in _tail_uh:
            share, uh_cells = _tail_uh['share'], _tail_uh['uh_cells']
            fdist = False
        else:
            w = torch.zeros(batch.n_outlets, dtype=dtype, device=device)
            # (core.ordered: the CPU's sums on a GPU, the same every run)
            w = _ordered.index_add(w, 0, batch.source2outlet,
                                   batch.cell_area)
            share = batch.cell_area / w[batch.source2outlet]
            fdist = getattr(batch, 'flow_distance', None)
        if fdist is False:
            pass
        elif fdist is None:
            # No flow network: one lumped unit hydrograph for every cell.
            uh_cells = uh if uh.dim() == 2 else uh.unsqueeze(1).expand(-1, nc)
        else:
            # A flow network: the within-cell response `uh` followed by the
            # Lohmann channel response over each cell's own flow length, so a
            # far cell arrives later and flatter than a near one.
            dflt = (routing.LOHMANN_VELOCITY, routing.LOHMANN_DIFFUSION)
            v = p.get('route_velocity', dflt[0])
            D = p.get('route_diffusion', dflt[1])
            uh_cells = routing.lohmann_unit_hydrographs(
                fdist, v, D, uh, dt=step_seconds)
        _tail_uh['share'], _tail_uh['uh_cells'] = share, uh_cells
        # The runoff an earlier run was still routing, prepended so the
        # convolution sees the same history it would inside one long run and
        # cut off again afterwards. conv1d computes each output as the same dot
        # product whatever the series' length, so the result is bit-identical
        # to the long run's (scratch conv_probe.py: five shapes, max diff 0.0).
        q_cells = runoff + baseflow
        if lo > 0:
            # (taken: each day's tail is read once, by the next day)
            routing_prefix = _tail_pre.pop(lo)
        elif routing_prefix is None and initial_state is not None:
            routing_prefix = initial_state.get('routing_prefix')
        n_pre = 0
        if routing_prefix is not None:
            if routing_prefix.dim() != 2 or routing_prefix.shape[1] != nc:
                raise ValueError('routing_prefix must be [steps, %d cells] of '
                                 'runoff + baseflow; got %s'
                                 % (nc, tuple(routing_prefix.shape)))
            n_pre = int(routing_prefix.shape[0])
            q_cells = torch.cat([routing_prefix.to(q_cells.dtype), q_cells], 0)
        _tail_pre[hi] = q_cells[max(int(q_cells.shape[0])
                                    - (int(uh_cells.shape[0]) - 1), 0):]
        discharge = routing.route(q_cells * share.unsqueeze(0),
                                  uh_cells, batch.source2outlet,
                                  batch.n_outlets, skip=n_pre)
        out = dict(discharge_mm=discharge, runoff=runoff, baseflow=baseflow,
                   swe=torch.stack(S(swes)), et=torch.stack(S(ets)),
                   **({'snow_depth': torch.stack(S(snow_depths))} if S(snow_depths)
                      else {}),
                   **{('refreeze' if k == 'refreeze' else 'eb_' + k):
                      torch.stack(S(v)) for k, v in snow_eb.items() if S(v)},
                   soil_moisture=soil_moisture_end,
                   **({'soil_moisture_tile': moist_tile_end}
                      if mosaic is not None else {}),
                   # VIC's OUT_SUB_SNOW. A flux out of the basin like evaporation,
                   # and it must appear in any balance that closes.
                   sub_snow=torch.stack(S(sublimations)))
        if mosaic is not None:
            # Canopy storage is a store like any other and must appear in a water
            # balance. Leaving it out made a year-long run miss by 0.05 mm, which
            # is small, real, and exactly the size of the water left on the leaves.
            out['canopy_storage'] = canopy_storage_end
            out['transp'] = torch.stack(S(transps))
            out['evap_canop'] = torch.stack(S(canopy_evaps))
            # put_data.c:612-622: OUT_EVAP_BARE is the soil evaporation of the
            # bare tile, cover 1 - sum(Cv). A mosaic that covers the cell has no
            # bare tile and VIC's accumulator stays at exactly 0; a mosaic that
            # does not cover it has a bare tile this model does not simulate,
            # and the row is left absent rather than reported as zero.
            if _tail_static('covered', lambda: bool(
                    (mosaic.cv.sum(1) >= 1.0 - 1e-6).all())):
                # With a fractional canopy the vegetated tiles have bare gaps
                # of their own, and :612-622 counts their ARNO evaporation
                # (bare_evap_frac, func_surf_energy_bal.c:776-783) here too;
                # the water-balance path forms it.
                out['evap_bare'] = (torch.stack(S(evap_bares)) if S(evap_bares)
                                    else torch.zeros_like(out['transp']))
            if wb_mode and wb_e:
                # out-vars-wb: the energy rows of the Tsurf = Tair evaluation
                # (torchvic.model.wbenergy), under the energy path's names
                for k, v in wb_e.items():
                    if S(v):
                        out['eb_' + k] = torch.stack(S(v))
                for k, v in wb_x.items():
                    if S(v):
                        out['eb_' + k] = torch.stack(S(v))
                if S(wb_pets):
                    out['pet'] = torch.stack(S(wb_pets))
            if wb_mode and wb_grnd:
                # OUT_GRND_FLUX and OUT_DELTAH, which VIC reports in
                # water-balance mode too: the quick flux at Tsurf = Tair,
                # cover-weighted and negated as put_data.c:922, :925 write them.
                out['eb_grnd_flux'] = torch.stack(S(wb_grnd))
                out['eb_delta_h'] = torch.stack(S(wb_delta_h))
                if frozen_soil:
                    # OUT_FUSION (put_data.c:928, negated) and OUT_SOIL_TEMP
                    # (put_data.c:693), which VIC reports in this mode too.
                    # Absent without frozen soil, as before.
                    out['eb_fusion'] = torch.stack(S(wb_fusion))
                    out['eb_layer_temp'] = torch.stack(S(wb_layer_temp))
            # solve_snow.c:181-193: snow_intercept runs only `if (overstory)`,
            # and vic_run.c loops only tiles with cover; a mosaic whose overstory
            # classes all have Cv = 0 (Walnut Gulch: six classes, none covering
            # any cell) has no canopy snow and VIC's OUT_SNOW_CANOPY and
            # OUT_SUB_CANOP are exactly 0. With a *covered* overstory tile the
            # rows are left absent: interception is not simulated. The first
            # version tested the class flag alone and counted padding.
            if not _tail_static('overstory', lambda: bool(
                    ((mosaic.overstory > 0) & (mosaic.cv > 0)).any())):
                out['snow_canopy'] = torch.zeros_like(out['transp'])
                out['sub_canop'] = torch.zeros_like(out['transp'])
            elif S(snow_canopies):
                # put_data.c: OUT_SNOW_CANOPY is snow.snow_canopy cover-weighted
                # (a store, end of step), OUT_SUB_CANOP the canopy's vapour flux
                out['snow_canopy'] = torch.stack(S(snow_canopies))
                out['sub_canop'] = torch.stack(S(sub_canops))
            if carbon_out:
                # put_data.c's OUT_GPP, OUT_RAUT, OUT_NPP and the rest, in VIC's
                # own units: the fluxes are gC/m2/s per unit ground area,
                # cover-weighted over the mosaic.
                for _k, _v in carbon_out.items():
                    out['carbon_' + _k] = torch.stack(S(_v))
            if S(albedos):
                out['albedo'] = torch.stack(S(albedos))
            if use_eb:
                # The surface energy balance, cover-weighted over tiles, in
                # put_data.c's reported convention: latent, sensible, ground flux
                # and storage change positive away from the surface / into the
                # ground; net radiation positive downward.
                for k, v in eb_out.items():
                    out['eb_' + k] = torch.stack(S(v))
                for k, v in eb_extra.items():
                    # 'fusion' is filled only under FROZEN_SOIL; with the
                    # option off the list stays empty and the zero series
                    # below stands in, as VIC's own OUT_FUSION does.
                    if S(v):
                        out['eb_' + k] = torch.stack(S(v))
                # func_surf_energy_bal.c:560: fusion is written only inside
                # `if (FS_ACTIVE && options.FROZEN_SOIL)`; an unfrozen run leaves
                # it at 0 and OUT_FUSION reports that 0 (put_data.c). This model
                # has no frozen soil (GlobalParam refuses FROZEN_SOIL TRUE), so
                # the row is VIC's 0 under VIC's condition, not an omission.
                out['eb_fusion'] = (torch.stack(S(eb_extra['fusion']))
                                    if S(eb_extra.get('fusion') or [])
                                    else torch.zeros_like(out['eb_t_surf']))
                out['pet'] = torch.stack(S(pets))
                # put_data.c:853 and :441: OUT_BARESOILT accumulates over the
                # bare tile only and is normalised by its cover if that cover is
                # positive; a mosaic that covers the cell leaves the accumulator
                # at exactly 0, and so does this. A mosaic that does not cover
                # the cell has a bare tile this model does not simulate, and the
                # row is then left absent rather than reported as zero.
                if ('eb_baresoilt' not in out
                        and _tail_static('covered', lambda: bool(
                            (mosaic.cv.sum(1) >= 1.0 - 1e-6).all()))):
                    out['eb_baresoilt'] = torch.zeros_like(out['eb_t_surf'])
                # put_data.c:538-552, :1071-1082: OUT_ENERGY_ERROR is formed from
                # the step's reported rows, after the snow steps are combined
                _ee = ('eb_net_short', 'eb_net_long', 'eb_latent', 'eb_sensible',
                       'eb_grnd_flux', 'eb_delta_h', 'eb_fusion', 'eb_advection',
                       'eb_deltacc', 'eb_snow_flux', 'eb_rfrz_energy')
                # CLOSE_ENERGY joins the subdaily runs here: the mean surface
                # residual the loop leaves is the open balance's error, while
                # the closed one reports the put_data combination, which does
                # not vanish where the iteration re-weighted the terms.
                # SPATIAL_SNOW likewise: a pack folded into the surface solve
                # under a step-open temperature of 0 C (a warm start's first
                # step) leaves the rows apart from the root's residual, and
                # put_data reports the rows (0.71 W/m2 on himal's first
                # restarted step).
                if (subdaily is not None or close_energy or spatial_snow) \
                        and all(k in out for k in _ee):
                    _z = torch.zeros_like(out['eb_net_short'])
                    _g = lambda k: out.get(k, _z)
                    out['eb_energy_error'] = (
                        (_g('eb_net_short') + _g('eb_net_long'))
                        - (_g('eb_latent') + _g('eb_latent_sub'))
                        - (_g('eb_sensible') + _g('eb_advected_sensible'))
                        - (_g('eb_grnd_flux') + _g('eb_delta_h') + _g('eb_fusion'))
                        + (_g('eb_advection') - _g('eb_deltacc')
                           + _g('eb_snow_flux') + _g('eb_rfrz_energy')))

        # ---- The outputs the loop computed and used to drop -----------------
        # Every array here is [time, cell] (soil_liq and soil_ice [time, cell,
        # layer]) at the model step, in VIC's history units, and is assembled
        # from the lists above and the stacked fluxes -- no physics runs after
        # the loop. Where VIC forms a term from two steps' totals (the DEL*
        # terms, the water error) the previous step's total is the series
        # shifted by one with the opening state in front, which is what
        # save_data holds when put_data.c runs for step t.
        prcp_s = torch.stack(S(prcps))
        swe_s = out['swe']
        # put_data.c:502-504: the pack and the canopy's snow are one store
        # (OUT_DELSWE = OUT_SWE + OUT_SNOW_CANOPY - save.swe)
        canopy_swe = (out['snow_canopy'] if 'snow_canopy' in out
                      else torch.zeros_like(swe_s))
        soil_liq = torch.stack(S(soil_liqs))                      # [time, cell, layer]
        soil_ice_s = (torch.stack(S(soil_ices)) if S(soil_ices)
                      else torch.zeros_like(soil_liq))
        total_moist = soil_liq.sum(-1)
        zero_cell = torch.zeros(nc, dtype=dtype, device=device)

        def previous(series, first, injected=False):
            p_ = torch.cat([first.unsqueeze(0), series[:-1]], 0)
            if injected and _day_inj:
                # a day_hook's soil addition moves the total the next step's
                # DEL term and water error open on (as a restart would)
                for t_, v_ in _day_inj:
                    if lo <= t_ < hi:
                        p_[t_ - lo] = p_[t_ - lo] + v_
            return p_

        def opening(name, cold):
            # What save_data held before this run's first step: the cold
            # start's totals, or, after a warm start, the row the earlier run's
            # last step reported. A cold start has no pack, so its SWE is 0.
            if lo > 0:
                # a day of a session (day_outputs): the rows the day before
                # ended on, as its state's report hands them to a one-day run
                return _tail_rows[lo][name]
            return cold if _prev_rows is None else _prev_rows[name]

        # Canopy storage is zero everywhere on the no-mosaic path -- there is no
        # canopy -- and that is a real zero, not a placeholder: the water error
        # below must see the same store the balance in the runner sees.
        wdew_s = torch.stack(S(wdews)) if S(wdews) else torch.zeros_like(swe_s)
        out.update(
            rainf=torch.stack(S(rains)), snowf=torch.stack(S(snowfalls)),
            snow_melt=torch.stack(S(melts)),
            snow_cover=torch.stack(S(snow_covers)), salbedo=torch.stack(S(salbedos)),
            snow_surf_temp=torch.stack(S(snow_surf_temps)),
            snow_pack_temp=torch.stack(S(snow_pack_temps)),
            # put_data.c:683-697: OUT_SOIL_LIQ is layer moisture less its ice,
            # OUT_SOIL_ICE the ice. This column runs with FROZEN_SOIL off and
            # Nfrost = 1, under which cell.layer[].ice is identically zero (see
            # torchvic.core.column's header), so the liquid is the whole moisture and
            # the ice is a genuine zero rather than an unimplemented quantity.
            # put_data.c:691-703: OUT_SOIL_MOIST is the total, OUT_SOIL_LIQ
            # the total less the ice, OUT_SOIL_ICE the ice.
            soil_liq=soil_liq - soil_ice_s, soil_ice=soil_ice_s,
            asat=torch.stack(S(asats)), inflow=torch.stack(S(inflows)),
            # put_data.c:482-504, each a store's total now minus the total the
            # previous step ended with. OUT_DELSWE adds OUT_SNOW_CANOPY on both
            # sides; there is no canopy snow here, so it drops out exactly.
            delsoilmoist=total_moist - previous(
                total_moist, opening('total_moist', moist0_cell.sum(-1)),
                injected=True),
            delswe=(swe_s + canopy_swe) - previous(
                swe_s + canopy_swe, opening('swe_canopy', zero_cell)),
            delintercept=wdew_s - previous(wdew_s, opening('wdew', wdew0)),
            # OUT_SURF_COND is declared (vic_driver_shared_all.h:245) and given
            # metadata (history_metadata.c:1138-1143) and is assigned nowhere in
            # the pristine 5.0.1 tree: put_data.c never touches it, so VIC
            # writes the zero out_data was initialised to. Zero is therefore
            # VIC's value, not a stand-in for one.
            surf_cond=torch.zeros_like(swe_s),
        )
        # put_data.c:531-544: inflow - outflow - (storage - last_storage), per
        # cell per step. OUT_EVAP there carries the snow sublimation
        # (put_data.c:626: tmp_evap += snow.vapor_flux), and so does `et` here,
        # so the outflow is OUT_EVAP + RUNOFF + BASEFLOW exactly as :532 has it.
        # Storage is the soil water, the pack and the canopy store; surfstor and
        # canopy snow are zero on this path. Routing is not a store of the cell
        # -- VIC has none -- so runoff and baseflow leave here, and the runner's
        # basin residual is the one that carries what is still in transit.
        storage = total_moist + swe_s + wdew_s + canopy_swe
        _bf_wb = baseflow
        if _vw_gwm is not None:
            # VIC-WUR 3.0 put_data.c:537-538: OUT_GWRECHARGE leaves the cell
            out['gwrecharge'] = torch.stack(S(gwrecharges))
            out['recharge'] = torch.stack(S(recharges))
            _bf_wb = baseflow + out['gwrecharge']
        _tail_rows[hi] = dict(total_moist=total_moist[-1],
                              swe_canopy=(swe_s + canopy_swe)[-1],
                              wdew=wdew_s[-1], storage=storage[-1])
        out['water_error'] = (prcp_s
                              - (out['et'] + runoff + _bf_wb)
                              - (storage - previous(
                                  storage, opening('storage',
                                                   moist0_cell.sum(-1) + wdew0),
                                  injected=True)))
        # These two are None when the soil parameters did not carry what they
        # need -- w_wilt_mm for the wetness, depth and bubble for the water
        # table -- and column.step says so rather than guessing. They are then
        # left out of the output rather than filled with a number that would
        # agree with VIC only by accident.
        # The evaporation the column refused because the layer had no liquid
        # above residual, which VIC withholds and still reports (runoff.c
        # :108-133). OUT_WATER_ERROR is short by exactly this, in VIC as
        # here; with FROZEN_SOIL off it is zero except on a column dried to
        # residual, which is why nothing noticed before.
        if S(withhelds):
            out['evap_withheld'] = torch.stack(S(withhelds))
        if S(soil_wets) and S(soil_wets)[0] is not None:
            out['soil_wet'] = torch.stack(S(soil_wets))
        if S(zwts) and S(zwts)[0] is not None:
            out['zwt'] = torch.stack(S(zwts))
        if S(zwt_lumpeds) and S(zwt_lumpeds)[0] is not None:
            out['zwt_lumped'] = torch.stack(S(zwt_lumpeds))
        for _k, _v in band_rows.items():
            if S(_v):
                out['band_' + _k] = torch.stack(S(_v))
        if S(sub_blowings):
            out['sub_blowing'] = torch.stack(S(sub_blowings))
        if S(surf_frost_fracs):
            out['surf_frost_frac'] = torch.stack(S(surf_frost_fracs))
        if S(fdepths):
            out['fdepth'] = torch.stack(S(fdepths))
            out['tdepth'] = torch.stack(S(tdepths))
        if mosaic is not None:
            out['wdew'] = wdew_s
            out['rootmoist'] = torch.stack(S(rootmoists))
            out['lai'] = torch.stack(S(lais))
            # No fcanopy on the mosaic: FCAN_SRC FROM_DEFAULT with no cover in
            # the last class, where vic_init.c:262-273 gives every covered class
            # 1.0, which is what VIC reports at put_data.c:667.
            if mosaic is not None and mosaic.fcanopy is not None:
                fc_series = torch.stack(
                    [(mosaic.fcanopy[:, :, months[i]] * mosaic.cv).sum(1)
                     for i in range(lo, hi)])
                out['fcanopy'] = fc_series
            else:
                out['fcanopy'] = torch.ones_like(swe_s)
        # The forcing as the model saw it, put_data.c:133-153. VIC echoes the
        # model-step value force->x[NR]; `atm` is already at the model step. The
        # three pressures are divided by PA_PER_KPA (1000) for output at :137
        # and :151-152 -- `atm` carries them in Pa, as vic_image.py's UNITS says
        # -- and relative humidity is FRACT_TO_PERCENT * vp / (vp + vpd) at
        # :141-142, a percentage that history_metadata.c labels "1". Wind is the
        # forcing's own where there is one, not the MIN_WIND-clamped value the
        # resistances use: VIC reports force->wind, and the clamp is ours.
        if lakes is not None:
            # put_data.c 5.1.0:305-400 (OUT_LAKE_*, OUT_SURFSTOR), and the
            # lake's OUT_BARESOILT term (see lakecell)
            out.update(lakes.outputs(nc, nt))
            if _lake_baresoilt:
                _lb = torch.stack(_lake_baresoilt)
                out['eb_baresoilt'] = (out['eb_baresoilt'] + _lb
                                       if 'eb_baresoilt' in out else _lb)
            # put_data.c 5.1.0:525-541: the storage the water error closes
            # against includes the lake (OUT_SURFSTOR)
            _ss = out['surfstor']
            _ss0 = torch.zeros_like(_ss[0])
            _ss0[lakes.cells] = (lakes.initial_volume * 1000.0
                                 / lakes.cell_area)
            out['water_error'] = out['water_error'] - (_ss - previous(_ss, _ss0))
            # put_data.c 5.1.0:730: OUT_DELSURFSTOR, against save_data
            out['delsurfstor'] = _ss - previous(_ss, _ss0)
            # Measured, not derived: pristine VIC 5.1.0 with LAKES TRUE writes
            # NaN for the first step's OUT_DELSOILMOIST, OUT_DELSWE,
            # OUT_DELINTERCEPT and OUT_WATER_ERROR (Findley Lake 1970; finite
            # from step 2 on). They are the terms formed against save_data,
            # which initialize_save_data fills with a put_data call on the
            # initial state (put_data.c 5.1.0:1033-1048); that call evidently
            # yields NaN totals with a lake. Reproduced on the lake cells.
            for _k in ('delsoilmoist', 'delswe', 'delintercept', 'water_error'):
                if _k in out:
                    _x = out[_k].clone()
                    _x[0, lakes.cells] = float('nan')
                    out[_k] = _x
        if not full:
            atm = {k_: (v_[lo:hi] if torch.is_tensor(v_) and v_.dim() >= 1
                        and int(v_.shape[0]) == nt else v_)
                   for k_, v_ in atm.items()}
        out['air_temp'] = atm['tair']
        out['swdown'] = atm['shortwave']
        out['lwdown'] = atm['longwave']
        # put_data.c:877-882: LongOverIn or LongUnderIn, the forcing's own
        # outside the snow steps
        out['in_long'] = (torch.stack(S(in_longs)) if len(S(in_longs)) == hi - lo
                          else atm['longwave'])
        out['pressure'] = atm['press'] / 1000.0
        out['vp'] = atm['eact_air'] / 1000.0
        out['vpd'] = atm['vpd'] / 1000.0
        out['rel_humid'] = 100.0 * atm['eact_air'] / (atm['eact_air'] + atm['vpd'])
        out['density'] = atm['air_dens']
        out['wind'] = (atm['wind'] if 'wind' in atm
                       else _like(atm['tair'], DEFAULT_WIND))
        # the VIC variables formed from the rows above (torchvic.model.outvars)
        _outvars.derive(out, atm, depth=getattr(cell_soil_params, 'depth', None),
                        carbon=bool(carbon), lakes=lakes is not None,
                        blowing=bool(blowing), energy_balance=bool(use_eb),
                        vic_energy_error=(energy_error == 'vic'),
                        dt_seconds=step_seconds, spatial_snow=bool(spatial_snow),
                        any_overstory=bool(any_over),
                        fallback_free=(_ov(*_outvars.FBFLAGS)
                                       and not any(_tfx.counts().values())),
                        want=_ov,
                        nnode=(3 if quick_flux else nnodes),
                        cv_overstory=(
                            None if mosaic is None or mosaic.overstory is None
                            else (mosaic.cv * ((mosaic.overstory > 0.5)
                                               & (mosaic.cv > 0.0))).sum(1)))
        if copy_state:
            # the end-of-day stores may be the loop's own tensors
            for k_ in ('soil_moisture', 'soil_moisture_tile',
                       'canopy_storage'):
                if torch.is_tensor(out.get(k_)):
                    out[k_] = out[k_].clone()
        if return_state:
            # Everything the loop would carry into a next step, as the last one
            # left it, plus the report rows the next run's first DEL* terms and
            # water error are formed against, and the routing tail.
            n_tail = int(uh_cells.shape[0]) - 1
            out['state'] = (_clone_tree if copy_state else _same)(export_state(
                dict(moist=moist, snow=snow_state, last_snow=last_snow,
                     melting=melting, spatial_snow=sp, wdew=wdew,
                     canopy=canopy_state if any_over else None,
                     ground_long_out=ground_long_out if any_over else None,
                     t_node=t_node if use_eb else None, soil_ice=soil_ice,
                     wb_t=dict(t0=wb_t0, t1=wb_t1) if wb_mode else None,
                     carbon=carbon_state, snow_zero=_snow_zero,
                     ce_seed=ce_seed),
                report=dict(total_moist=total_moist[-1],
                            swe_canopy=(swe_s + canopy_swe)[-1],
                            wdew=wdew_s[-1], storage=storage[-1]),
                routing_prefix=q_cells[max(int(q_cells.shape[0]) - n_tail, 0):],
                steps_per_day=steps_per_day, last_date=batch.dates[hi - 1],
                last_yday=ydays[hi - 1]))
            out['state_aux'] = (_clone_tree if copy_state else _same)(
                dict(_vic_aux))
            if lakes is not None:
                out['state']['lake'] = lakes.export_state()
            out['state']['vic_aux'] = out['state_aux']
            if lanes is not None:
                # carried per covered tile (see LANE_COMPACT)
                out['state']['lanes'] = dict(n_lanes=lanes.n_lanes,
                                             n_cells=lanes.n_cells,
                                             n_tiles=lanes.n_tiles)
        if _vw is not None:
            out['vicwur'] = _vw_res
            if mosaic.cv is not _vw_cv0:
                # the last step's Cv override formed the outputs above; the
                # case keeps its file's Cv
                out['vicwur']['cv_end'] = mosaic.cv
                mosaic.cv = _vw_cv0
        return out

    # day_hook (coupling, torchvic.coupling.vic): called between days of
    # the run, after the last step of every day but the run's last, with a
    # _DayView of the loop's state; it may return (new soil moisture in the
    # loop's layout, the cell total added, mm) and may add to the coming
    # steps' precipitation forcing in place. No equation of the step
    # changes: the soil moisture the next step starts from and the forcing
    # it reads are the hook's, as a run cut at that day boundary and
    # restarted with the addition would have them (the DEL terms and the
    # water error open on the moved totals, as after a restart).
    _day_inj = []
    s_in = None
    if day_outputs and (day_hook is None or _vw is not None
                        or not return_state):
        raise ValueError('day_outputs needs a day_hook, return_state and no '
                         'vicwur_core')
    if defer_day_check and not day_outputs:
        raise ValueError('defer_day_check needs day_outputs')
    if not 0 <= int(tail_from) < nt or (tail_from and not day_outputs):
        raise ValueError('tail_from must be a step of the call (with '
                         'day_outputs); got %r' % (tail_from,))
    if day_hook is not None and (subdaily is not None or lakes is not None):
        raise ValueError('day_hook runs on the plain model-step loop only '
                         '(no sub-daily snow steps, no lakes)')

    def _call_hook(t_):
        nonlocal moist
        if graph_mode:
            buf_ = _flat[_fk.index('prcp')]
        elif lanes is not None:
            buf_ = cell_atm['prcp']
        else:
            buf_ = atm['prcp']
        def _fluxes_():
            # the day's steps t_-steps_per_day .. t_-1 (VIC-WUR's rout_run
            # reads OUT_RUNOFF, OUT_BASEFLOW: VIC-WUR 3.0 rout_run.c:19-26)
            if lanes is not None:
                _flush_lanes()
            d_ = dict(runoff=torch.stack(runoffs[-steps_per_day:]).sum(0),
                      baseflow=torch.stack(baseflows[-steps_per_day:])
                      .sum(0))
            if _vw is not None and _vw_out.get('water_stress'):
                d_['water_stress'] = _vw_cols(_vw_out['water_stress'][-1])
            if _vw is not None and _vw_tf[0]:
                # the day's (last) step: VIC-WUR's crop runs daily
                d_.update({k_: _vw_cols(v_) for k_, v_ in _vw_tf[0].items()})
            if _vw_gwm is not None:
                d_['gwrecharge'] = torch.stack(
                    gwrecharges[-steps_per_day:]).sum(0)
            return d_
        _outs_ = None
        if day_outputs:
            def _outs_():
                lo_ = t_ - steps_per_day
                for l_ in _out_lists + list(carbon_out.values()):
                    if len(l_) not in (0, t_):
                        # a row list not filled once a step: its rows
                        # are not the day's steps
                        raise ValueError('day_outputs: an output list has '
                                         '%d rows at step %d' % (len(l_), t_))
                o_ = _tail(lo_, t_, batch, atm, mosaic, nc, cvl,
                           moist0_cell, wdew0, routing_prefix,
                           copy_state=True)
                if defer_day_check and graph_mode:
                    o_['_day_flag'] = _flag_ref[0].clone()
                # the day's rows are in o_ now: the lists keep their
                # length (the loop indexes them by step), not the tensors
                for l_ in _out_lists:
                    for j_ in range(max(lo_ - 1, 0), min(t_ - 1, len(l_))):
                        l_[j_] = None
                return o_
        r_ = day_hook(t_, _DayView(moist, buf_, lanes, soil_ice,
                                   fluxes=_fluxes_, outputs=_outs_,
                                   graphs=graph_mode))
        if r_ is None:
            return
        new_, cell_ = r_
        if s_in is not None and moist is s_in['moist']:
            moist.copy_(new_)         # the graphs' own buffer
        else:
            moist = new_
        _day_inj.append((t_, cell_))

    if not graph_mode:
        if subdaily is None and lakes is not None:
            for t in _steps:
                lakes.before(mosaic)
                _step(t)
                _lake_after(t)
                _tfx.end_step(t)
            lakes.restore(mosaic)
        elif subdaily is None:
            for t in _steps:
                _step(t)
                _tfx.end_step(t)
                if day_hook is not None and (t + 1) % steps_per_day == 0 \
                        and t + 1 < nt:
                    _call_hook(t + 1)
        else:
            for t in _steps:
                _one_day(t)
                _tfx.end_step(t)
    else:
        # CUDA graphs (GRAPHS): one graph per (month, season) -- the two
        # Python values a step reads from its date -- captured on the
        # step's graph-safe path (torchvic.core.graphsafe) and replayed on
        # fixed buffers: the carried state, the step's forcing at the
        # lanes, and the outputs, cloned after each replay. The graph-safe
        # path is the eager one bit for bit while its assumptions hold;
        # they are checked once a day of steps, and a day that broke one
        # is run again eagerly from the state it started with.
        def _get_state():
            d = dict(moist=moist, last_snow=last_snow, melting=melting,
                     wdew=wdew)
            for k_, v_ in snow_state.items():
                d['snow.' + k_] = v_
            if any_over:
                for k_, v_ in canopy_state.items():
                    d['canopy.' + k_] = v_
                d['ground_long_out'] = ground_long_out
            if wb_mode:
                d['wb_t0'], d['wb_t1'] = wb_t0, wb_t1
            if use_eb:
                d['t_node'] = t_node
            if torch.is_tensor(soil_ice):
                d['soil_ice'] = soil_ice
            if _vw_ws is not None:
                # the stored layer stress (vicwur_core water_stress)
                d['vw.ws'] = _vw_ws
            if node_props is not None:
                # QUICK_FLUX FALSE: the node properties the column step
                # redistributed, which the next step's profile reads
                for k_, v_ in node_props.items():
                    d['node.' + k_] = v_
            # the state file's extra stores: a step reads some of them
            # (cold_content, tfoliage_bare: kept where no snow), so they
            # are carried like the rest -- a graph reading the tensor the
            # step before its capture left there would read freed memory
            for k_, v_ in _vic_aux.items():
                if torch.is_tensor(v_):
                    d['aux.' + k_] = v_
            return d

        def _set_state(d):
            nonlocal moist, last_snow, melting, wdew, snow_state, \
                ground_long_out, wb_t0, wb_t1, t_node, soil_ice, node_props, \
                _vw_ws
            if 'vw.ws' in d:
                _vw_ws = d['vw.ws']
            moist, last_snow = d['moist'], d['last_snow']
            melting, wdew = d['melting'], d['wdew']
            snow_state = {k_[5:]: v_ for k_, v_ in d.items()
                          if k_.startswith('snow.')}
            if any_over:
                for k_, v_ in d.items():
                    if k_.startswith('canopy.'):
                        canopy_state[k_[7:]] = v_
                ground_long_out = d['ground_long_out']
            if wb_mode:
                wb_t0, wb_t1 = d['wb_t0'], d['wb_t1']
            if use_eb:
                t_node = d['t_node']
            # None until the first step makes it (one frost area)
            soil_ice = d.get('soil_ice')
            if node_props is not None:
                node_props = {k_[5:]: v_ for k_, v_ in d.items()
                              if k_.startswith('node.')}
            for k_, v_ in d.items():
                if k_.startswith('aux.'):
                    _vic_aux[k_[4:]] = v_

        lane_atm = atm
        satm = {k_: torch.empty(lanes.n_lanes, dtype=v_.dtype, device=device)
                for k_, v_ in cell_atm.items()
                if torch.is_tensor(v_) and v_.dim() == 2}
        fixed_atm = {k_: _Fixed(v_) for k_, v_ in satm.items()}

        # The step's index on the device: a graph reads its forcing (and
        # season) at it, so a replay needs only the index set
        # (_fill_lanes, recorded in the graph) in place of a gather per
        # forcing field launched from the host.
        _t_dev = torch.zeros((), dtype=torch.long, device=device)
        _fk = list(satm)

        def _doy_of(t_):
            return (float(batch.day_of_year[t_])
                    if hasattr(batch, 'day_of_year')
                    else float(batch.dates[t_].timetuple().tm_yday))
        _season_t = torch.tensor([60.0 < _doy_of(t_) < 273.0
                                  for t_ in range(nt)], dtype=torch.bool,
                                 device=device)

        # each field flat, once per call (a window handed over as a view
        # would otherwise be copied whole at every step)
        _flat = [cell_atm[k_].contiguous() for k_ in _fk]

        def _fill_dev():
            _fill_lanes(_flat, [satm[k_] for k_ in _fk],
                        _t_dev, lanes.lane_cell, _season_t, _season_buf)

        # The date enters the step twice: the month indexes the mosaic's
        # monthly fields, and the season decides the melting flag. In a
        # graph both are buffers: the monthly fields at the step's month
        # (as [lane, 1, 1], read at month 0) and the season as a flag.
        _monthly = [k_ for k_, v_ in vars(mosaic).items()
                    if torch.is_tensor(v_) and v_.dim() == 3
                    and (k_ in _STEP_FIELDS if step_axis
                         else v_.shape[2] == 12)]
        lane_mosaic, lane_months = mosaic, months
        g_mosaic = copy.copy(mosaic)
        for k_ in _monthly:
            setattr(g_mosaic, k_, torch.empty_like(
                getattr(mosaic, k_)[:, :, :1]))
        g_months = _AnyIndex(0)
        cur_month = [None]
        if _vw is not None:
            # the hooks' host part writes the veg tables' step values into
            # the month buffers, and a state_fn's states into the graphs'
            _vw_gm[0], _vw_gm[1] = g_mosaic, frozenset(_monthly)

            def _vw_sync_fn():
                if s_in is None:
                    return
                for k_, v_ in _get_state().items():
                    if k_ in s_in and v_ is not s_in[k_]:
                        s_in[k_].copy_(v_)
                _set_state(s_in)
            _vw_sync[0] = _vw_sync_fn
            _vw_split[0] = True

        def _set_date(t_):
            m_ = lane_months[t_]      # `months` is g_months while graphed
            if m_ != cur_month[0]:
                for k_ in _monthly:
                    getattr(g_mosaic, k_).copy_(
                        getattr(lane_mosaic, k_)[:, :, m_:m_ + 1])
                cur_month[0] = m_

        def _key(t_):
            return ()

        def _mark_flushed():
            for i_, l_ in enumerate(_out_lists):
                _flushed[i_] = len(l_)

        if any_over:
            _over_rc = over_t.nonzero()
        _snow_set.clear()
        _snow_set['ver'] = 0
        if SNOW_SET and per_tile_snow:
            _stores = [snow_state.get(k_) for k_ in
                       ('ice', 'surf_water', 'pack_water', 'swq')]
            if any_over:
                _stores.append(canopy_state.get('int_snow'))
            (_ix_s, _ix_c), _ = _snow_window_set(
                cell_atm['prcp'][:nt], cell_atm['tair'][:nt],
                p['t_max_snow'], lanes.lane_cell, _stores,
                over_t if any_over else None)
            _shp = snow_state['ice'].shape
            for nm_, ix_ in (('snow', _ix_s), ('canopy', _ix_c)):
                if ix_ is None:
                    _snow_set[nm_] = None
                    continue
                in_ = torch.zeros(_shp, dtype=torch.bool, device=device)
                in_[ix_, 0] = True
                pos_ = torch.zeros(_shp, dtype=torch.long, device=device)
                pos_[ix_, 0] = torch.arange(ix_.shape[0], device=device)
                c_ = torch.zeros_like(ix_)
                _snow_set[nm_] = {'in': in_, 'valid': in_, 'pos': pos_,
                                  'r': ix_, 'c': c_,
                                  'rc': torch.stack([ix_, c_], 1)}
            # lanes in each subset (all: every lane), per window
            if _SNOW_SET_SAY:
                print('snow set: %d lanes, snow %s, canopy %s' % (
                    nc, *('all' if _snow_set[k_] is None
                          else int(_snow_set[k_]['r'].shape[0])
                          for k_ in ('snow', 'canopy'))), flush=True)
            GRAPH_STATS['snow_set'] = '%s/%s ' % tuple(
                'all' if _snow_set[k_] is None
                else str(int(_snow_set[k_]['r'].shape[0]))
                for k_ in ('snow', 'canopy'))
        flag = torch.zeros(1, dtype=torch.long, device=device)
        _flag_ref[0] = flag
        # GRAPH_SNOW_ADAPT: snow seen by a day's steps (read with the flag)
        seen = (torch.zeros(1, dtype=torch.long, device=device)
                if GRAPH_SNOW_ADAPT and not defer_day_check else None)
        modes = dict(snow_any=True)
        _season_buf = torch.zeros((), dtype=torch.bool, device=device)
        redo_bits = {}
        graphs = {}
        _chk_seen = set()
        # Earlier windows' graph pools are freed blocks the allocator keeps
        # cached; on a small device hand them back before capturing anew
        # (on a large one this would only cost the next window its
        # allocations again).
        if device.type == 'cuda' and not SYNC_CHECK:
            _total = torch.cuda.get_device_properties(device).total_memory
            if torch.cuda.memory_reserved(device) > 0.5 * _total:
                torch.cuda.empty_cache()
            pool = torch.cuda.graph_pool_handle()
            cap_stream = torch.cuda.Stream(device)
        s_in = None
        n_eager = n_redo = 0
        capture_s = record_s = 0.0
        n_loop = 0
        t = 0
        while t < nt:
            t_end = min(t + steps_per_day, nt)
            if _vw is not None:
                # the day's (one step's) hook values, before its snapshot:
                # a day run again starts from the states a state_fn left
                _vw_host(t)
            snap = {k_: v_.clone() for k_, v_ in _get_state().items()}
            lens = [len(l_) for l_ in _out_lists]
            vlens = [len(l_) for l_ in _vw_lists]
            graphsafe.FLAG = flag
            graphsafe.SEEN = seen
            graphsafe.MODES = modes
            graphsafe.ACTIVE = True
            atm, mosaic, months = fixed_atm, g_mosaic, g_months
            try:
                for tt in range(t, t_end):
                    _t_dev.fill_(tt)
                    _set_date(tt)
                    key = _key(tt) + (modes['snow_any'], _snow_set['ver'],
                                      _vw_key())
                    g_ = graphs.get(key)
                    if g_ is None:
                        _fill_dev()
                    if g_ is None and SYNC_CHECK and key in _chk_seen:
                        # every step as its capture would run it, checked
                        chk_ = graphsafe.SyncCheck()
                        graphsafe.SEGMENTER = graphsafe.CheckSegmenter(chk_)
                        try:
                            with chk_:
                                _step(tt)
                                _flush_lanes()
                        finally:
                            graphsafe.SEGMENTER = None
                        SYNC_FOUND.update(chk_.found)
                        n_eager += 1
                        continue
                    if g_ is None and SYNC_CHECK:
                        # the first step of a kind warms what it makes
                        # lazily, as the eager step before a capture does
                        _chk_seen.add(key)
                        _step(tt)
                        _flush_lanes()
                        n_eager += 1
                        continue
                    if g_ is None:
                        # this step eagerly, on the graph-safe path (which
                        # also warms every lazily made constant), then the
                        # capture, which runs nothing. The carried state is
                        # first made distinct tensors, as the capture's
                        # buffers are: a fused region (core.fusion) compiled
                        # here assumes which of its inputs share memory,
                        # and the capture must find the same
                        _keys0 = set(_get_state())
                        _set_state({k_: v_.clone()
                                    for k_, v_ in _get_state().items()})
                        _step(tt)
                        _flush_lanes()
                        n_eager += 1
                        if GRAPHS_NO_CAPTURE or any(
                                k_ == 'soil_ice' or k_.startswith('node.')
                                for k_ in set(_get_state()) - _keys0):
                            # (a carried value the step made that the step
                            # reads, the soil ice of a cold start: the next
                            # step, which reads it, is the one captured)
                            continue
                        cur = _get_state()
                        if s_in is None:
                            s_in = {k_: v_.clone() for k_, v_ in cur.items()}
                        else:
                            if set(cur) != set(s_in):
                                raise RuntimeError('graph state keys changed')
                            for k_, v_ in cur.items():
                                s_in[k_].copy_(v_)
                        _set_state(s_in)
                        n0 = [len(l_) for l_ in _out_lists]
                        vn0 = [len(l_) for l_ in _vw_lists]
                        tf0 = _vw_tf[0]
                        _tc = _time.time()
                        # torch.cuda.graph() would also collect garbage and
                        # empty the allocator's cache first, which in a long
                        # run costs more than the capture
                        cap_stream.wait_stream(torch.cuda.current_stream())
                        gr = graphsafe.Segmenter(pool)
                        with torch.cuda.stream(cap_stream), \
                                graphsafe.capture_ops():
                            gr.begin()
                            graphsafe.SEGMENTER = gr
                            try:
                                _fill_dev()
                                _step(tt)
                                _flush_lanes()
                                s_out = _get_state()
                                if set(s_out) != set(s_in):
                                    raise RuntimeError(
                                        'graph state keys changed')
                                for k_, v_ in s_out.items():
                                    if v_ is not s_in[k_]:
                                        s_in[k_].copy_(v_)
                                _tr = _time.time()
                            finally:
                                graphsafe.SEGMENTER = None
                                gr.finish()
                        torch.cuda.current_stream().wait_stream(cap_stream)
                        record_s += _tr - _tc
                        outs = []
                        for i_, l_ in enumerate(_out_lists):
                            if len(l_) > n0[i_]:
                                outs.append((i_, list(l_[n0[i_]:])))
                                del l_[n0[i_]:]
                        _mark_flushed()
                        vouts = []
                        for i_, l_ in enumerate(_vw_lists):
                            if len(l_) > vn0[i_]:
                                vouts.append((i_, list(l_[vn0[i_]:])))
                                del l_[vn0[i_]:]
                        # the replays' tile fluxes (read before the next
                        # replay: _fluxes_, _tail copy them)
                        gtf = _vw_tf[0] if _vw_tf[0] is not tf0 else None
                        _vw_tf[0] = tf0
                        _set_state(s_in)
                        graphs[key] = g_ = (gr, outs, vouts, gtf)
                        capture_s += _time.time() - _tc
                        continue
                    n_loop += g_[0].replay()
                    for i_, ts_ in g_[1]:
                        _out_lists[i_].extend(
                            x_.clone() if torch.is_tensor(x_) else x_
                            for x_ in ts_)
                    _mark_flushed()
                    for i_, ts_ in g_[2]:
                        _vw_lists[i_].extend(
                            x_.clone() if torch.is_tensor(x_) else x_
                            for x_ in ts_)
                    if g_[3] is not None:
                        _vw_tf[0] = g_[3]
            finally:
                graphsafe.ACTIVE = False
                graphsafe.FLAG = None
                graphsafe.SEEN = None
                graphsafe.MODES = {}
                atm, mosaic, months = lane_atm, lane_mosaic, lane_months
            if defer_day_check:
                # read by the caller with its own checks (day_outputs'
                # '_day_flag'): a day whose assumption failed is computed
                # again there, from the state it started on; no host read
                # here
                bits = 0
            elif seen is not None and modes['snow_any']:
                # one host read for both
                bits, seen_ = torch.cat([flag, seen]).tolist()
                seen.zero_()
                if not bits and not seen_:
                    # no snow on any step of the day: the next day on the
                    # snow-free graph (a step with snow breaks it, as ever)
                    modes['snow_any'] = False
            else:
                bits = int(flag)
            if bits:
                # an assumption of the graph-safe path failed on this day:
                # the day again, eagerly, from the state it started with
                n_redo += 1
                redo_bits[bits] = redo_bits.get(bits, 0) + 1
                if bits & (1 << _SNOW_SET_BIT):
                    # the window's snow subset missed a lane: every lane
                    # for the rest of the call
                    _snow_set.update(snow=None, canopy=None,
                                     ver=_snow_set['ver'] + 1)
                flag.zero_()
                for l_, n_ in zip(_out_lists, lens):
                    del l_[n_:]
                _mark_flushed()
                for l_, n_ in zip(_vw_lists, vlens):
                    del l_[n_:]
                # the stores the day opened without (made by its first
                # step) are taken away again, as the day found them
                for k_ in [k_ for k_, v_ in _vic_aux.items()
                           if torch.is_tensor(v_) and 'aux.' + k_ not in snap]:
                    del _vic_aux[k_]
                _set_state({k_: v_.clone() for k_, v_ in snap.items()})
                day_snow = False
                for tt in range(t, t_end):
                    _step(tt)
                    _tfx.end_step(tt)
                    _flush_lanes()
                    day_snow = day_snow or graphsafe.LAST.get('snow_any',
                                                              True)
                if s_in is not None:
                    for k_, v_ in _get_state().items():
                        s_in[k_].copy_(v_)
                    _set_state(s_in)
                # the next day assumes the branches this day ended on
                modes.update({k_: graphsafe.LAST.get(k_, True)
                              for k_ in modes})
                if GRAPH_SNOW_ADAPT:
                    # the snow graph takes snow-free steps too: snow on any
                    # step of the day keeps it
                    modes['snow_any'] = day_snow
            t = t_end
            if day_hook is not None and t < nt:
                _call_hook(t)
        _vw_split[0] = False
        _vw_sync[0] = None
        _vw_gm[0] = None
        GRAPH_STATS.update(graphs=len(graphs), eager_steps=n_eager,
                           redone_days=n_redo, steps=nt,
                           redo_bits=str(redo_bits),
                           capture_s=capture_s, record_s=record_s,
                           loop_replays=n_loop)
        del graphs

    if False:
        # names the code before bound only here (nested blocks declare
        # them nonlocal)
        k = share = v = None
    _phase.mark('loop_s')
    out = _tail(tail_from, nt, batch, atm, mosaic, nc, cvl,
                moist0_cell, wdew0, routing_prefix)
    if defer_day_check and graph_mode:
        out['_day_flag'] = _flag_ref[0].clone()
    _phase.mark('post_s')
    GRAPH_STATS.update(_phase.times)
    return out
