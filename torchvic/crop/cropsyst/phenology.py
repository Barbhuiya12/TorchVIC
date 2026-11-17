"""E6 of the VIC-CropSyst port: thermal time and phenology (docs/connectors/cropsyst.md).

What the shipped build runs (VIC_CropSyst/build/library/.../Makefile_Kamiak links
thermal_time_common.o, thermal_time_daily.o, phenology_A.o and phenology_2013.o; there is no
phenology_2018.o, thermal_time_hourly.o, vernalization.o or photoperiod.o, and neither
PHENOLOGY_VERSION nor THERMAL_TIME_OBSOLETE_2020 is defined, so every `#if` takes the 2013 /
non-obsolete branch):

- One crop thermal time accumulator, Thermal_time_daily_linear or _nonlinear
  (crop_cropsyst.cpp:276-295). Its accumulated degree days (GDDs) drive the stage events
  (thermal_time_matchs, crop_cropsyst.cpp:2957-2964). The plant temperature is the
  stress-adjusted Tmax (crop_cropsyst.cpp:119-141, 2645-2653) and the air Tmin
  (thermal_time_daily.cpp:22-36, 55-86).
- Phenology_2013 (phenology_2013.cpp) holds the growth stage and four Period_thermal objects:
  accrescence, culminescence, senescence and root elongation (phenology_A.cpp:172-375). Each
  period owns a separate thermal time accumulator with the same parameters, started the day the
  period is activated, and expires when its own sum exceeds its duration (phenology_A.cpp:342-345).
  Periods drive the canopy stages: accrescence expiry -> culminescence, culminescence expiry ->
  senescence (crop_cropsyst.cpp:2987-3001).
- Vernalization and photoperiod limit the daily thermal time by min(photo, vern)
  (thermal_time_common.cpp:158-195), all season long (no growth-stage gating in this build).

Reference behaviour ported as is (each is checked by the oracle,
tests/oracle/ref_cropsyst_phenology.cpp, which drives the compiled reference objects):

- Vernalization::update runs TWICE per accumulation: once in accumulate_degree_days and once in
  calc_limited_growing_degree_day (thermal_time_common.cpp:208-226, 178). Each call adds the day's
  vernalization day again. The factor of the second call is the one used.
- The vernalization status never becomes ACTIVE (nothing sets it), so it never ends and
  `vern_days` is never reset (thermal_time_common.cpp:69-82). Before `start` is reached the factor
  is 1; after, min + (1-min)(v-start)/(end-start), clamped to [0, 1] (std::min/std::max order).
- The crop accumulator and the phenology own SEPARATE Vernalization objects
  (crop_cropsyst.cpp:265-270 and 303-307). The phenology one is shared by all its periods, so it
  advances two vernalization days per active period per day, in the period order accrescence,
  culminescence, senescence, root elongation (phenology_2013.cpp:68-71), plus two more on the
  emergence day, when create_roots starts the root elongation period at once
  (crop_cropsyst.cpp:1534-1535).
- The vernalization temperature is the daily mean (Tmax+Tmin)/2 of the unadjusted air temperature
  (temperature_with_est_night, crop_cropsyst.cpp:2640-2641; VIC_land_unit_simulation.cpp:252).
- calc_stress_adjusted_temperature's `started_maturity` argument receives !is_matured()
  (crop_cropsyst.cpp:2652), so the senescence reduction factor is used only AFTER maturity.
- The LAI-based canopy's know_senescence(0) returns false (canopy_growth_LAI_based.h:235-236), so
  Crop_complete::culminate_senescence stops before phenology.culminate_senescence
  (crop_cropsyst.cpp:2367-2371): the expired senescence period stays ("senescence(expired)" in the
  crop csv). `culminate_senescence=True` in the parameters gives the cover-based behaviour.
- Period durations come from the legacy keys (crop_param_class.cpp:922-937, 955-962): accrescence =
  peak_LAI - emergence, culminescence = senescence - peak_LAI, senescence = full_senescence -
  senescence; zero durations become 10; root elongation defaults to 1040 (:199).
- Parameter adjustments ([adjustment:n] in the .crp, event_scheduler_crop.cpp:120-150) overwrite a
  thermal time temperature on (synchronised stage day + offset), after that day's accumulation
  (Crop_common::start_day runs after the thermal time, crop_cropsyst.cpp:1997-2001). Adjustments of
  other parameters are not part of E6 and are ignored here.

Scope: the thermal-time events of annual crops and of fruit trees (Crop_orchard_fruit::
thermal_time_event: fructescence, veraison, rapid fruit growth). The perennial life cycle
(consider_inactive_period, restart_after_inactive_period, dormancy) is orchard.py; the
`quiescence` stage stops the events, as in the reference. The response to
clipping (thermal time reset, periods restarted, new accrescence) is clipping.py.
Hydrothermal emergence and the hourly thermal time (not compiled) are not ported.

Layout: every tensor has a leading batch shape [...] (e.g. [cell, crop tile]); stage-indexed
arrays add a last axis of NGS_COUNT. Stage switches are boolean masks applied with torch.where,
there are no Python loops over cells, no host syncs and no data-dependent control flow, so `step`
traces under torch.compile and CUDA-graph capture. Days are integer day indices (any origin).
The equations are re-derived from the shipped C++ (Academic Free License; no code text copied).
"""
from __future__ import annotations

import math
import re
from typing import NamedTuple

import torch

from .readers import IniTable

# Normal_crop_event_sequence of the CS5 build (growth_stages.h:23-55; values printed by the oracle)
NGS_NONE = 0
NGS_PLANTING = 1
NGS_GERMINATION = 2
NGS_EMERGENCE = 3
NGS_RESTART = 4
NGS_ACCRESCENCE = 5          # NGS_ACTIVE_GROWTH
NGS_END_CANOPY_GROWTH = 6
NGS_ANTHESIS = 7
NGS_FILLING = 8
NGS_RIPENING = 9
FGS_RAPID_FRUIT_DEVELOPMENT = 10
NGS_MATURITY = 11
NGS_HARVESTABLE = 12
NGS_HARVEST = 13
NGS_POSTHARVEST = 14
NGS_QUIESCENCE = 15
NGS_TERMINATED = 16
NGS_COUNT = 17

# normal_crop_event_sequence_label_table (growth_stages.c:7-30), for event_phenologic
STAGE_LABELS = ["none", "sowing", "germination", "emergence", "restart", "active_growth",
                "end_canopy_growth", "flowering", "filling", "ripening",
                "reserved_for_future_use_2", "maturity", "harvestable", "harvest", "postharvest",
                "dormant", "terminated"]
# get_growth_stage_description (growth_stages.c:66-96), spaces removed as in the crop csv
STAGE_PHRASE = {NGS_NONE: "preplantingfallow", NGS_GERMINATION: "germination",
                NGS_EMERGENCE: "emergence", NGS_RESTART: "restart",
                NGS_ACCRESCENCE: "Activegrowth", NGS_END_CANOPY_GROWTH: "canopy(culmination)",
                NGS_ANTHESIS: "flowering(anthesis)", NGS_FILLING: "yieldformation",
                NGS_RIPENING: "ripening", FGS_RAPID_FRUIT_DEVELOPMENT: "fructescence(rapid)",
                NGS_MATURITY: "maturity", NGS_HARVESTABLE: "harvestable",
                NGS_HARVEST: "harvest", NGS_QUIESCENCE: "quiescent",
                NGS_TERMINATED: "terminated"}

# Photoperiodism (crop_types.h:188-194; labels crop_types.c:162-168)
PHOTOPERIODISM = ["none", "short-day", "long-day", "intermediate-day", "day-neutral"]

# thermal time temperatures that an adjustment may set (index in the state's `temps`)
ADJ_PARAM = {"base_temp": 0, "cutoff_temp": 1, "optimal_temp": 2, "opt_temp": 2,
             "maximum_temp": 3, "max_temp": 3}

VERN_DAY_SLOPE = 7.0           # thermal_time_common.cpp:62-66
IS_ZERO_TOL = 0.0000001        # corn/math/compare.hpp:41,50
STRESS_VPD_COEF = 1.5          # crop_cropsyst.cpp:134

# defaults of the parameter classes when a key is absent
DEFAULT_TEMPS = (3.0, 22.0, 22.0, 35.0)                     # crop_param_class.cpp:44-48
DEFAULT_PHENOLOGY = dict(emergence=100.0, tuber_init=1000.0, flowering=1000.0,
                         filling=1020.0, maturity=1500.0)   # crop_param_class.cpp:177-181
DEFAULT_DUR_CULMINESCENCE = 100.0                           # crop_param_class.cpp:197
DEFAULT_DUR_SENESCENCE = 500.0                              # crop_param_class.cpp:198
DEFAULT_DUR_ROOT_ELONGATION = 1040.0                        # crop_param_class.cpp:199
ZERO_DURATION_REPLACEMENT = 10.0                            # crop_param_class.cpp:955-962
DEFAULT_VERNALIZATION = dict(low_temp=3.0, high_temp=10.0, start=10.0, end=50.0,
                             min_factor=0.0)                # crop_param_class.cpp:158-165
DEFAULT_SENS_WATER_STRESS = 1.0                             # crop_param_class.cpp:209


def _cmax(a, b):
    """std::max(a, b) = (a < b) ? b : a  (corn/math/compare.hpp)."""
    return torch.where(a < b, b, a)


def _cmin(a, b):
    """std::min(a, b) = (b < a) ? b : a."""
    return torch.where(b < a, b, a)


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(x.dtype)


# ----------------------------------------------------------------------------- pure functions

def gdd_linear(tmax_adj, tmin, base, cutoff):
    """Thermal_time_daily_linear::calc_growing_degree_day_daily_resolution
    (thermal_time_daily.cpp:22-36): mean plant temperature clipped to [base, cutoff], minus base."""
    avg = (tmax_adj + tmin) / 2.0
    avg = torch.where(avg < base, base, avg)
    avg = torch.where(avg > cutoff, cutoff, avg)
    return avg - base


def gdd_nonlinear(tmax_adj, tmin, base, opt, tmax_card):
    """Thermal_time_daily_nonlinear (thermal_time_daily.cpp:55-86): Wang & Engel (1998) beta
    function with beta = 1, times (Topt - Tmin); zero outside (Tmin, Tmax)."""
    avg = (tmax_adj + tmin) / 2.0
    inside = (avg > base) & (avg < tmax_card)
    y = opt - base
    alpha = math.log(2.0) / torch.log((tmax_card - base) / y)
    x = torch.where(inside, avg - base, torch.ones_like(avg))
    xa = torch.pow(x, alpha)
    num = 2.0 * xa * torch.pow(y, alpha) - torch.pow(x, 2.0 * alpha)
    wedr = num / torch.pow(y, 2.0 * alpha)
    return torch.where(inside, wedr, torch.zeros_like(wedr)) * y


def growing_degree_day(tmax_adj, tmin, temps, nonlinear):
    """Daily thermal time not limited by vernalization or photoperiod; `temps` [..., 4] =
    (base, cutoff, opt, max)."""
    base, cutoff, opt, tmx = temps.unbind(-1)
    lin = gdd_linear(tmax_adj, tmin, base, cutoff)
    nonlin = gdd_nonlinear(tmax_adj, tmin, base, opt, tmx)
    return torch.where(nonlinear, nonlin, lin)


def vernalization_day(tavg, low, high):
    """Vernalization day of one update (thermal_time_common.cpp:62-67)."""
    v = torch.where(tavg > high, 1.0 - (tavg - high) / VERN_DAY_SLOPE,
                    torch.where(tavg < low, 1.0 - (low - tavg) / VERN_DAY_SLOPE,
                                torch.ones_like(tavg)))
    return torch.where(v < 0.0, torch.zeros_like(v), v)


def vernalization_factor(vern_days, start, end, min_factor):
    """Vfactor after an update (thermal_time_common.cpp:55, 69-72, 83); the END branch (:78-82) is
    unreachable because the status never becomes ACTIVE."""
    f = min_factor + ((1.0 - min_factor) * (vern_days - start)) / (end - start)
    f = torch.where(vern_days >= start, f, torch.ones_like(f))
    return _cmin(_cmax(f, torch.zeros_like(f)), torch.ones_like(f))


def photoperiod_factor(day_length, stop, unconstrained):
    """Photoperiodization::update (thermal_time_common.cpp:95-130); long-day when
    unconstrained >= stop, short-day otherwise; 1 when stop == unconstrained."""
    one, zero = torch.ones_like(day_length), torch.zeros_like(day_length)
    ramp = 1.0 - ((day_length - unconstrained) / (stop - unconstrained))
    long_day = torch.where(day_length <= stop, zero,
                           torch.where(day_length >= unconstrained, one, ramp))
    short_day = torch.where(day_length >= stop, zero,
                            torch.where(day_length <= unconstrained, one, ramp))
    f = torch.where(unconstrained >= stop, long_day, short_day)
    f = _cmin(_cmax(f, zero), one)
    return torch.where(stop == unconstrained, one, f)


def stress_adjusted_temperature(sens_water_stress, tmax, senescence_reduction_factor,
                                water_stress_index, vpd_max, matured, has_leaf_senescence):
    """calc_stress_adjusted_temperature (crop_cropsyst.cpp:119-141) as called by
    update_temperatures (:2645-2653), whose `started_maturity` argument is !is_matured().
    `water_stress_index` is the crop's water_stress_stats.index of the previous day (1 - the
    transpiration ratio, sample_cumulative_T.h:55)."""
    srf = senescence_reduction_factor
    nonzero = torch.abs(0.0 - srf) > IS_ZERO_TOL           # !CORN::is_zero(srf)
    red = torch.where(has_leaf_senescence & matured & nonzero, srf, torch.ones_like(srf))
    red = _cmax(torch.zeros_like(red), _cmin(red, torch.ones_like(red)))
    return tmax + ((1.0 + STRESS_VPD_COEF * vpd_max * red) * water_stress_index) * sens_water_stress


# ----------------------------------------------------------------------------- parameters

class PhenologyParams(NamedTuple):
    """Per-crop E6 parameters, float32-rounded like the reference structs (crop_param_struct.h),
    any batch shape [...] (gather by crop id to [cell, tile])."""
    temps: torch.Tensor                  # [..., 4] base, cutoff, opt, max  [deg C]
    nonlinear: torch.Tensor              # bool, thermal_response == nonlinear
    emergence: torch.Tensor              # thresholds on the crop GDDs [deg C day]
    tuberization: torch.Tensor
    flowering: torch.Tensor
    filling: torch.Tensor
    maturity: torch.Tensor
    dur_accrescence: torch.Tensor        # period durations [deg C day]
    dur_culminescence: torch.Tensor
    dur_senescence: torch.Tensor
    dur_root_elongation: torch.Tensor
    maturity_significant: torch.Tensor   # bool
    has_leaf_senescence: torch.Tensor    # bool (crop_param_class.cpp:595-600)
    is_tuber: torch.Tensor               # bool, harvested_part == tuber
    fruit_harvest: torch.Tensor          # bool, fruit tree harvested for fruit (no filling event)
    culminate_senescence: torch.Tensor   # bool, False for the LAI-based canopy (see module doc)
    vern_enabled: torch.Tensor           # bool
    vern_low: torch.Tensor
    vern_high: torch.Tensor
    vern_start: torch.Tensor
    vern_end: torch.Tensor
    vern_min: torch.Tensor
    photo_crop: torch.Tensor             # bool, photoperiodism > none (crop accumulator, :272)
    photo_phen: torch.Tensor             # bool, photoperiodism != none (phenology, :308)
    photo_stop: torch.Tensor             # [h]
    photo_unconstrained: torch.Tensor    # [h]
    sens_water_stress: torch.Tensor      # phenology water_stress_sensitivity
    adj_stage: torch.Tensor              # [..., A] long, synchronising NGS stage (-1 unused)
    adj_offset: torch.Tensor             # [..., A] long, days after the stage
    adj_param: torch.Tensor              # [..., A] long, index into temps (-1 = not E6)
    adj_value: torch.Tensor              # [..., A] float32-rounded new value
    is_fruit: torch.Tensor               # bool, crop model fruit (Crop_orchard_fruit)
    rapid_fruit: torch.Tensor            # thermal time of rapid fruit growth [deg C day]
    grape: torch.Tensor                  # bool, veraison before rapid fruit growth


def phenology_params(crops: IniTable, device=None, dtype=torch.float64) -> PhenologyParams:
    """[n_crop] E6 parameters from a crop table (readers.crop_table), with the reference defaults
    for absent keys (crop_param_class.cpp) and the legacy period composition
    (crop_param_class.cpp:922-937, 955-962)."""
    n = crops.n
    dev = device if device is not None else next(iter(crops.params.values())).device

    def get(key, default):
        """Float parameter; NaN (float) or -1 (int) marks an absent key (readers.MISSING_*)."""
        t = crops.params.get(key)
        if t is None:
            return torch.full((n,), float(default), dtype=dtype, device=dev)
        t = t.to(dev)
        missing = torch.isnan(t) if t.dtype.is_floating_point else t == -1
        return torch.where(missing, torch.full((n,), float(default), dtype=dtype, device=dev),
                           t.to(dtype))

    def get_bool(key, default):
        t = crops.params.get(key)
        if t is None:
            return torch.full((n,), bool(default), device=dev)
        return t.to(dev)

    def enum_is(key, word):
        t = crops.params.get(key)
        if t is None or word not in crops.vocab.get(key, []):
            return torch.zeros(n, dtype=torch.bool, device=dev)
        return t.to(dev) == crops.code(key, word)

    temps = torch.stack([get("phenology.base_temp", DEFAULT_TEMPS[0]),
                         get("phenology.cutoff_temp", DEFAULT_TEMPS[1]),
                         get("phenology.optimal_temp", DEFAULT_TEMPS[2]),
                         get("phenology.maximum_temp", DEFAULT_TEMPS[3])], -1)
    emergence = get("phenology.emergence", DEFAULT_PHENOLOGY["emergence"])
    peak = get("phenology.peak_LAI", 0.0)
    sen0 = get("phenology.senescence", 0.0)
    sen1 = get("phenology.full_senescence", 0.0)
    emergence, peak, sen0, sen1 = _f32(emergence), _f32(peak), _f32(sen0), _f32(sen1)
    zero = torch.zeros_like(peak)
    # crop_param_class.cpp:922-937: accrescence starts at emergence
    dur_acc = torch.where(peak > 0, _f32(peak - emergence), zero)
    has_sen = sen0 > 0
    dur_culm = torch.where(has_sen, _f32(sen0 - peak), torch.full_like(peak, DEFAULT_DUR_CULMINESCENCE))
    dur_sen = torch.where(has_sen, _f32(sen1 - sen0), torch.full_like(peak, DEFAULT_DUR_SENESCENCE))

    def nz(d):                                              # :955-962 (CORN::is_zero<float32>)
        return torch.where(torch.abs(0.0 - d) <= IS_ZERO_TOL,
                           torch.full_like(d, ZERO_DURATION_REPLACEMENT), d)

    life_annual = enum_is("crop.life_cycle", "annual")
    deciduous = enum_is("crop.abscission", "deciduous")
    senesces = get_bool("leaf.senesces", False)
    is_fruit_model = enum_is("crop.model", "fruit")
    harvest_fruit = enum_is("crop.harvested_part", "fruit")
    photo = crops.params.get("photoperiod.consideration")
    if photo is None:
        photo_code = torch.zeros(n, dtype=torch.long, device=dev)
    else:
        words = crops.vocab["photoperiod.consideration"]
        lut = torch.tensor([PHOTOPERIODISM.index(w) if w in PHOTOPERIODISM else 0 for w in words],
                           dtype=torch.long, device=dev)
        photo = photo.to(dev)
        photo_code = torch.where(photo >= 0, lut[photo.clamp(min=0)], torch.zeros_like(photo))

    # adjustments that set a thermal time temperature (event_scheduler_crop.cpp:120-150)
    A = 0
    if "adjustment.value" in crops.params:
        A = crops.params["adjustment.value"].shape[1]
    stage = torch.full((n, max(A, 1)), -1, dtype=torch.long)
    offset = torch.zeros((n, max(A, 1)), dtype=torch.long)
    param = torch.full((n, max(A, 1)), -1, dtype=torch.long)
    value = torch.zeros((n, max(A, 1)), dtype=dtype)
    if A:
        valid = crops.valid["adjustment"].cpu()
        P = crops.params
        V = crops.vocab
        for i in range(n):                                  # setup-time parsing, host side
            for j in range(A):
                if not bool(valid[i, j]):
                    continue
                if "adjustment.enabled" in P and not bool(P["adjustment.enabled"][i, j]):
                    continue
                sync = V["adjustment.event_synchronization"][int(P["adjustment.event_synchronization"][i, j])]
                if sync != "after_normal_crop_growth_stage":
                    continue
                phen = V["adjustment.event_phenologic"][int(P["adjustment.event_phenologic"][i, j])]
                text = crops.text["adjustment.parameter"][i][j] or ""
                m = re.search(r"\[([^\]]*)\]", text)
                key = m.group(1) if m else text.strip()
                if key not in ADJ_PARAM or phen not in STAGE_LABELS:
                    continue
                stage[i, j] = STAGE_LABELS.index(phen)
                offset[i, j] = int(P["adjustment.event_offset"][i, j])
                param[i, j] = ADJ_PARAM[key]
                value[i, j] = float(P["adjustment.value"][i, j])

    def t(x):
        return x.to(device=dev)

    return PhenologyParams(
        temps=_f32(temps), nonlinear=torch.zeros(n, dtype=torch.bool, device=dev),
        emergence=emergence,
        tuberization=_f32(get("phenology.tuber_init", DEFAULT_PHENOLOGY["tuber_init"])),
        flowering=_f32(get("phenology.flowering", DEFAULT_PHENOLOGY["flowering"])),
        filling=_f32(get("phenology.filling", DEFAULT_PHENOLOGY["filling"])),
        maturity=_f32(get("phenology.maturity", DEFAULT_PHENOLOGY["maturity"])),
        dur_accrescence=nz(dur_acc), dur_culminescence=nz(dur_culm), dur_senescence=nz(dur_sen),
        dur_root_elongation=torch.full_like(peak, DEFAULT_DUR_ROOT_ELONGATION),
        maturity_significant=get_bool("phenology.maturity_significant", True),
        has_leaf_senescence=deciduous | senesces | life_annual,
        is_tuber=enum_is("crop.harvested_part", "tuber"),
        fruit_harvest=is_fruit_model & harvest_fruit,
        culminate_senescence=torch.zeros(n, dtype=torch.bool, device=dev),
        vern_enabled=get_bool("vernalization.enabled", False),
        vern_low=_f32(get("vernalization.low_temp", DEFAULT_VERNALIZATION["low_temp"])),
        vern_high=_f32(get("vernalization.high_temp", DEFAULT_VERNALIZATION["high_temp"])),
        vern_start=_f32(get("vernalization.start", DEFAULT_VERNALIZATION["start"])),
        vern_end=_f32(get("vernalization.end", DEFAULT_VERNALIZATION["end"])),
        vern_min=_f32(get("vernalization.min_factor", DEFAULT_VERNALIZATION["min_factor"])),
        photo_crop=photo_code > 0, photo_phen=photo_code != 0,
        photo_stop=_f32(get("photoperiod.stop", 0.0)),
        photo_unconstrained=_f32(get("photoperiod.unconstrain", 0.0)),
        sens_water_stress=_f32(get("phenology.water_stress_sensitivity", DEFAULT_SENS_WATER_STRESS)),
        adj_stage=t(stage), adj_offset=t(offset), adj_param=t(param), adj_value=t(_f32(value)),
        is_fruit=is_fruit_model, rapid_fruit=_f32(get("phenology.rapid_fruit_growth", 0.0)),
        grape=get_bool("fruit.grape", False))


def gather_params(p: PhenologyParams, index: torch.Tensor) -> PhenologyParams:
    """Per-tile parameters from per-crop ones: index [...] of crop ids."""
    return PhenologyParams(*[x[index] for x in p])


# ----------------------------------------------------------------------------- state

class PhenologyState(NamedTuple):
    present: torch.Tensor        # bool, a crop is on the tile (sown, not yet harvested)
    stage: torch.Tensor          # long, Phenology_2013::growth_stage
    gdd: torch.Tensor            # growing_degree_day of the day (not limited)
    gdds: torch.Tensor           # crop GDDs (Accum_DD of the crop csv)
    gdds_yesterday: torch.Tensor
    vern_crop: torch.Tensor      # vern_days of the crop accumulator's Vernalization
    vern_phen: torch.Tensor      # vern_days of the phenology's (shared by the periods)
    vfac_crop: torch.Tensor      # last Vfactor of each
    vfac_phen: torch.Tensor
    period_on: torch.Tensor      # [..., 4] bool: accrescence, culminescence, senescence, root
    period_tt: torch.Tensor      # [..., 4] period thermal time
    period_rel_elapsed: torch.Tensor     # [..., 4] (phenology_A.cpp:347-353)
    period_rel_remaining: torch.Tensor   # [..., 4] float32, as the getter returns it (phenology_A.h:78)
    canopy: torch.Tensor         # bool, canopy object exists (after the first emergence)
    temps: torch.Tensor          # [..., 4] current thermal time temperatures (adjustable)
    adj_done: torch.Tensor       # [..., A] bool
    init_day: torch.Tensor       # [..., NGS_COUNT] long: initialization_date32 as day index, -1
    sync_day: torch.Tensor       # [..., NGS_COUNT] long: trigger_synchronization day, -1
    tuber_day: torch.Tensor      # long, start_date_tuber_initiation, -1
    season_days: torch.Tensor    # long, season_duration_days
    tmax_adj: torch.Tensor       # the day's stress-adjusted temperature


ACC, CULM, SEN, ROOT = 0, 1, 2, 3


def initial_state(params: PhenologyParams) -> PhenologyState:
    """No crop on any tile; shapes follow `params` (already gathered to the tile batch)."""
    shape = params.emergence.shape
    dev, dt = params.emergence.device, params.emergence.dtype
    z = torch.zeros(shape, dtype=dt, device=dev)
    zl = torch.zeros(shape, dtype=torch.long, device=dev)
    zb = torch.zeros(shape, dtype=torch.bool, device=dev)
    A = params.adj_stage.shape[-1]
    return PhenologyState(
        present=zb, stage=zl, gdd=z, gdds=z, gdds_yesterday=z, vern_crop=z, vern_phen=z,
        vfac_crop=torch.ones_like(z), vfac_phen=torch.ones_like(z),
        period_on=torch.zeros(shape + (4,), dtype=torch.bool, device=dev),
        period_tt=torch.zeros(shape + (4,), dtype=dt, device=dev),
        period_rel_elapsed=torch.zeros(shape + (4,), dtype=dt, device=dev),
        period_rel_remaining=torch.zeros(shape + (4,), dtype=torch.float32, device=dev),
        canopy=zb, temps=params.temps.clone(),
        adj_done=torch.zeros(shape + (A,), dtype=torch.bool, device=dev),
        init_day=torch.full(shape + (NGS_COUNT,), -1, dtype=torch.long, device=dev),
        sync_day=torch.full(shape + (NGS_COUNT,), -1, dtype=torch.long, device=dev),
        tuber_day=zl - 1, season_days=zl, tmax_adj=z)


def sow(state: PhenologyState, params: PhenologyParams, mask: torch.Tensor,
        day: torch.Tensor) -> PhenologyState:
    """A new crop on the masked tiles (Crop_complete::plant/start, crop_cropsyst.cpp:619, 652):
    fresh accumulators and periods, growth stage germination, planting initiated today. Call it
    before `step` on the sowing day: the reference accumulates thermal time on the sowing day."""
    m = mask
    mk = m[..., None]
    day = torch.as_tensor(day, dtype=torch.long, device=m.device)
    k = torch.arange(NGS_COUNT, device=m.device)
    fresh_days = torch.where(k == NGS_PLANTING, day[..., None] if day.dim() else day,
                             torch.full_like(state.init_day, -1))
    z = torch.zeros_like(state.gdds)
    w = lambda new, old: torch.where(m, new, old)
    return PhenologyState(
        present=m | state.present, stage=w(torch.full_like(state.stage, NGS_GERMINATION), state.stage),
        gdd=w(z, state.gdd), gdds=w(z, state.gdds), gdds_yesterday=w(z, state.gdds_yesterday),
        vern_crop=w(z, state.vern_crop), vern_phen=w(z, state.vern_phen),
        vfac_crop=w(z + 1.0, state.vfac_crop), vfac_phen=w(z + 1.0, state.vfac_phen),
        period_on=state.period_on & ~mk,
        period_tt=torch.where(mk, torch.zeros_like(state.period_tt), state.period_tt),
        period_rel_elapsed=torch.where(mk, torch.zeros_like(state.period_tt), state.period_rel_elapsed),
        period_rel_remaining=torch.where(mk, torch.zeros_like(state.period_rel_remaining),
                                         state.period_rel_remaining),
        canopy=state.canopy & ~m, temps=torch.where(mk, params.temps, state.temps),
        adj_done=state.adj_done & ~mk,
        init_day=torch.where(mk, fresh_days, state.init_day),
        sync_day=torch.where(mk, fresh_days, state.sync_day),
        tuber_day=w(torch.full_like(state.tuber_day, -1), state.tuber_day),
        season_days=w(torch.zeros_like(state.season_days), state.season_days),
        tmax_adj=state.tmax_adj)


def terminate(state: PhenologyState, mask: torch.Tensor) -> PhenologyState:
    """Remove the crop from the masked tiles (harvest with termination); the phenology stops."""
    return state._replace(present=state.present & ~mask)


def _vern_update(v, vd, m, p):
    """One masked Vernalization::update: v += vd, then the factor."""
    v = torch.where(m, v + vd, v)
    return v, vernalization_factor(v, p.vern_start, p.vern_end, p.vern_min)


def step(state: PhenologyState, params: PhenologyParams, day, tmax, tmin, day_length=None,
         water_stress_index=None, vpd_max=None, senescence_reduction_factor=None
         ) -> PhenologyState:
    """One day of E6 for every tile, in the order of Crop_complete::start_day and end_day
    (crop_cropsyst.cpp:1938-1948, 2026-2029). Inputs broadcast over the batch shape:
    tmax, tmin [deg C] air temperatures; day_length [h] (photoperiod only); water_stress_index of
    the previous day, vpd_max [kPa] and the canopy's senescence reduction factor (stress-adjusted
    temperature; defaults give no adjustment). Tiles without a crop are left unchanged."""
    p = state
    P = params
    on = p.present
    dt = p.gdds.dtype
    tmax = torch.as_tensor(tmax, dtype=dt, device=on.device).expand_as(p.gdds)
    tmin = torch.as_tensor(tmin, dtype=dt, device=on.device).expand_as(p.gdds)
    zero = torch.zeros_like(p.gdds)
    dl = zero + 12.0 if day_length is None else torch.as_tensor(day_length, dtype=dt, device=on.device) + zero
    wsi = zero if water_stress_index is None else torch.as_tensor(water_stress_index, dtype=dt, device=on.device) + zero
    vpd = zero if vpd_max is None else torch.as_tensor(vpd_max, dtype=dt, device=on.device) + zero
    srf = zero + 1.0 if senescence_reduction_factor is None else torch.as_tensor(senescence_reduction_factor, dtype=dt, device=on.device) + zero
    day = torch.as_tensor(day, dtype=torch.long, device=on.device)

    # --- update_temperatures (crop_cropsyst.cpp:2638-2672)
    tavg = (tmax + tmin) / 2.0                                   # VIC_land_unit_simulation.cpp:252
    matured = p.stage >= NGS_MATURITY                            # is_matured, phenology_I.h:44-45
    tadj = stress_adjusted_temperature(P.sens_water_stress, tmax, srf, wsi, vpd, matured,
                                       P.has_leaf_senescence)
    gdd = growing_degree_day(tadj, tmin, p.temps, P.nonlinear)  # same for every accumulator today
    vd = vernalization_day(tavg, P.vern_low, P.vern_high)
    photo = photoperiod_factor(dl, P.photo_stop, P.photo_unconstrained)
    one = torch.ones_like(gdd)
    photo_c = torch.where(P.photo_crop, photo, one)
    photo_p = torch.where(P.photo_phen, photo, one)
    # crop accumulator: two vernalization updates (thermal_time_common.cpp:208-226)
    mv = on & P.vern_enabled
    vc, _ = _vern_update(p.vern_crop, vd, mv, P)
    vc, fc = _vern_update(vc, vd, mv, P)
    fc = torch.where(P.vern_enabled, fc, one)
    vfac_crop = torch.where(mv, fc, p.vfac_crop)
    gdds = torch.where(on, p.gdds + gdd * _cmin(photo_c, fc), p.gdds)
    gdd_out = torch.where(on, gdd, p.gdd)

    # --- phenology.start_day: periods accumulate in order (phenology_2013.cpp:68-71)
    vp = p.vern_phen
    vfac_phen = p.vfac_phen
    pon = p.period_on & on[..., None]
    tts = list(p.period_tt.unbind(-1))
    durs = [P.dur_accrescence, P.dur_culminescence, P.dur_senescence, P.dur_root_elongation]
    for k in range(4):                    # four fixed period slots, not a loop over cells
        m = pon[..., k]
        mvk = m & P.vern_enabled
        vp, _ = _vern_update(vp, vd, mvk, P)
        vp, f = _vern_update(vp, vd, mvk, P)
        vfac_phen = torch.where(mvk, f, vfac_phen)
        f = torch.where(P.vern_enabled, f, one)
        tts[k] = torch.where(m, tts[k] + gdd * _cmin(photo_p, f), tts[k])

    # --- thermal_time_event (crop_cropsyst.cpp:2968-3015)
    y, t = p.gdds_yesterday, gdds

    def match(x):
        return (y <= x) & (t >= x)

    stage = p.stage
    act = on & (stage != NGS_QUIESCENCE)
    k_ar = torch.arange(NGS_COUNT, device=on.device)
    init_day, sync_day = p.init_day, p.sync_day

    dayk = day[..., None] if day.dim() else day

    def mark(arr, m, code):
        return torch.where(m[..., None] & (k_ar == code), dayk, arr)

    pon = list(pon.unbind(-1))
    # orchard fruit events before the common ones (Crop_orchard_fruit::thermal_time_event,
    # crop_fruit.cpp:27-39): fruit growth (fructescence, the filling stage) and rapid fruit growth
    # (veraison first for grapes, crop_fruit.cpp:310-331; phenology_2013.h:116-118)
    ff = act & P.is_fruit & match(P.filling)
    stage = torch.where(ff, torch.full_like(stage, NGS_FILLING), stage)
    sync_day = mark(sync_day, ff, NGS_FILLING)
    fr = act & P.is_fruit & match(P.rapid_fruit)
    gv = fr & P.grape
    stage = torch.where(gv, torch.full_like(stage, NGS_RIPENING), stage)
    sync_day = mark(sync_day, gv, NGS_RIPENING)
    stage = torch.where(fr, torch.full_like(stage, FGS_RAPID_FRUIT_DEVELOPMENT), stage)
    sync_day = mark(sync_day, fr, FGS_RAPID_FRUIT_DEVELOPMENT)
    # emergence -> initiate_accrescence (:2269-2310)
    e = act & (stage == NGS_GERMINATION) & match(P.emergence)
    canopy = p.canopy | e
    newroot = e & ~pon[ROOT]                                     # create_roots (:1534-1535)
    mvr = newroot & P.vern_enabled
    vp, _ = _vern_update(vp, vd, mvr, P)
    vp, f = _vern_update(vp, vd, mvr, P)
    vfac_phen = torch.where(mvr, f, vfac_phen)
    f = torch.where(P.vern_enabled, f, one)
    tts[ROOT] = torch.where(newroot, gdd * _cmin(photo_p, f), tts[ROOT])
    pon[ROOT] = pon[ROOT] | newroot
    stage = torch.where(e, torch.full_like(stage, NGS_ACCRESCENCE), stage)
    for code in (NGS_EMERGENCE, NGS_ACCRESCENCE):
        init_day = mark(init_day, e, code)
        sync_day = mark(sync_day, e, code)
    tts[ACC] = torch.where(e, zero, tts[ACC])
    pon[ACC] = pon[ACC] | e
    pon[CULM] = pon[CULM] & ~e                                   # phenology_A.cpp:79-94
    pon[SEN] = pon[SEN] & ~e
    newp = [e, torch.zeros_like(e), torch.zeros_like(e), newroot]   # created today

    # tuber initiation (:2319-2328)
    tb = act & P.has_leaf_senescence & match(P.tuberization) & P.is_tuber
    tuber_day = torch.where(tb, day, p.tuber_day)
    init_day = mark(init_day, tb, NGS_FILLING)

    # flowering (:2331-2346)
    fl = act & canopy & (P.flowering != 0) & match(P.flowering)
    sync_day = mark(sync_day, fl, NGS_ANTHESIS)
    fl_stage = fl & (stage != NGS_FILLING)
    stage = torch.where(fl_stage, torch.full_like(stage, NGS_ANTHESIS), stage)
    init_day = mark(init_day, fl_stage, NGS_ANTHESIS)

    # culminescence expired -> senescence (:2359-2366; phenology_A.cpp:110-124)
    se = act & canopy & pon[CULM] & (tts[CULM] > P.dur_culminescence)
    pon[SEN] = pon[SEN] | se
    tts[SEN] = torch.where(se, zero, tts[SEN])
    newp[SEN] = se
    pon[ACC] = pon[ACC] & ~se
    pon[CULM] = pon[CULM] & ~se

    # accrescence expired -> culminescence (:2374-2381; phenology_A.cpp:96-108)
    cu = act & canopy & pon[ACC] & (tts[ACC] > P.dur_accrescence)
    init_day = mark(init_day, cu, NGS_END_CANOPY_GROWTH)
    sync_day = mark(sync_day, cu, NGS_END_CANOPY_GROWTH)
    pon[CULM] = pon[CULM] | cu
    tts[CULM] = torch.where(cu, zero, tts[CULM])
    newp[CULM] = cu
    pon[ACC] = pon[ACC] & ~cu

    # senescence expired -> culminate_senescence (no-op for the LAI canopy, see module doc)
    cs = act & canopy & pon[SEN] & (tts[SEN] > P.dur_senescence) & P.culminate_senescence
    pon[ACC] = pon[ACC] & ~cs
    pon[CULM] = pon[CULM] & ~cs
    pon[SEN] = pon[SEN] & ~cs

    # filling (:2383-2396)
    fi = act & ~P.fruit_harvest & (P.filling != 0) & match(P.filling)
    stage = torch.where(fi, torch.full_like(stage, NGS_FILLING), stage)
    init_day = mark(init_day, fi, NGS_FILLING)
    sync_day = mark(sync_day, fi, NGS_FILLING)

    # maturity (:2397-2420)
    ma = act & P.maturity_significant & match(P.maturity)
    stage = torch.where(ma, torch.full_like(stage, NGS_MATURITY), stage)
    init_day = mark(init_day, ma, NGS_MATURITY)
    sync_day = mark(sync_day, ma, NGS_MATURITY)

    # period relative thermal time, set by each period's start_day (phenology_A.cpp:323-324,
    # 347-367); a period created today keeps 0 unless it accumulated (root on emergence day)
    rel_e = list(p.period_rel_elapsed.unbind(-1))
    rel_r = list(p.period_rel_remaining.unbind(-1))
    started = [p.period_on[..., k] & on & ~newp[k] for k in range(4)]   # accumulated in start_day
    started[ROOT] = started[ROOT] | newroot
    for k in range(4):
        el = tts[k] / durs[k]
        # float32 tensor, not a float64 round trip (torch.compile may drop a double cast pair)
        rr = _cmax((durs[k] - tts[k]) / durs[k], zero).to(torch.float32)
        rel_e[k] = torch.where(started[k], el, torch.where(newp[k], zero, rel_e[k]))
        rel_r[k] = torch.where(started[k], rr, torch.where(newp[k], torch.zeros_like(rr), rel_r[k]))

    # parameter adjustments (event_scheduler_crop.cpp:120-150), after the accumulation
    temps = p.temps
    A = P.adj_stage.shape[-1]
    sd = sync_day.gather(-1, P.adj_stage.clamp(min=0))
    fire = on[..., None] & (P.adj_param >= 0) & ~p.adj_done & (sd >= 0) & \
        (day == sd + P.adj_offset)
    tl = list(temps.unbind(-1))
    for a in range(A):                    # adjustment slots of the crop file (static count)
        for q in range(4):
            tl[q] = torch.where(fire[..., a] & (P.adj_param[..., a] == q), P.adj_value[..., a], tl[q])
    temps = torch.stack(tl, -1)

    # --- end_day: thermal_time->end_day, phenology.end_day (:2027-2029)
    gdds_y = torch.where(on, gdds, p.gdds_yesterday)
    season = torch.where(on, p.season_days + 1, p.season_days)
    return PhenologyState(
        present=on, stage=stage, gdd=gdd_out, gdds=gdds, gdds_yesterday=gdds_y,
        vern_crop=vc, vern_phen=vp, vfac_crop=vfac_crop, vfac_phen=vfac_phen,
        period_on=torch.where(on[..., None], torch.stack(pon, -1), p.period_on),
        period_tt=torch.stack(tts, -1),
        period_rel_elapsed=torch.stack(rel_e, -1), period_rel_remaining=torch.stack(rel_r, -1),
        canopy=canopy, temps=temps, adj_done=p.adj_done | fire, init_day=init_day,
        sync_day=sync_day, tuber_day=tuber_day, season_days=season,
        tmax_adj=torch.where(on, tadj, p.tmax_adj))


# ----------------------------------------------------------------------------- reporting

def stage_label(stage: int, period_on, sen_expired: bool, planting=False, harvest=False) -> str:
    """Grow_Stage of the crop csv for one tile-day (Crop_complete::describe_periods,
    crop_cropsyst.cpp:665-692; spaces dropped and items joined with '&' as in the csv).
    Host-side helper for reports and tests, not for the device loop."""
    items = [STAGE_PHRASE.get(int(stage), "preplantingfallow")]
    if planting:
        items.append("planting")
    if harvest:
        items.append("harvest")
    acc, culm, sen = (bool(period_on[0]), bool(period_on[1]), bool(period_on[2]))
    if acc:
        items.append("accrescence")
    if culm:
        items.append("culminescence")
    if sen:
        items.append("senescence(expired)" if sen_expired else "senescence")
    return "&".join(items)
