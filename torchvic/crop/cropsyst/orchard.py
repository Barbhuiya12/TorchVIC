"""Orchard and perennial fruit crops of the VIC-CropSyst port: the parts of
Crop_orchard_fruit (crop_fruit.cpp) and the perennial life cycle of Crop_complete
(crop_cropsyst.cpp) that the annual port did not have.

What the shipped build runs for a crop file with `model=fruit` (perform_op.cpp:430-480;
CROP_ORCHARD defined; the CRB files have no canopy architecture; with `[morphology]
canopy_architecture=true` the tree canopy extent and the orchard interception models of
orchard_arch.py run at the start of each process_day):

- An established crop (veg code without a rotation, VIC_land_unit_simulation.cpp:800-840) is sown
  on the first simulation day (land_unit_sim.cpp:1026-1029). Crop_orchard_fruit::start
  (crop_fruit.cpp:191-226): when the start date is in winter (Seasons::is_winter_time,
  seasons.cpp:50-57) the tree is taken as dormant: a chill requirement (chill_requirement units)
  and a catch-all quiescence end requirement are set up. Crop_complete::start
  (crop_cropsyst.cpp:597-640): a perennial gets its roots at once, at the maximum length
  (is_established, :2843-2845; create_roots :1514-1548), and a quiescence START requirement only
  when no end requirement was set (:605-611).
- Chill (Fruit_chill_requirement, crop_fruit.cpp:17-31): each day of process_day the 24 hourly air
  temperatures of the land unit (CS::Weather_hours_calculated, a Temperature_estimator on Tmax and
  Tmin, weather_hours.cpp:23-41, temperature_est.cpp, min_max_to_timestep_estimator.cpp) are turned
  into chill units (<1.4: 0, <2.4: 0.5, <9.1: 1, <12.4: 0.5, <15.9: 0, <18: -0.5, else -1)
  and deducted (float32) from the hours remaining; the requirement is met when the remainder is
  < 0 (crop_fruit.h:28). In spring/summer a positive remainder is set to -1 (crop_fruit.cpp:
  119-125, the 230707LML fix). The next day bud break is evaluated: the crop thermal time (it
  keeps accumulating while dormant) reaching deg_day_bud_break_chill_req_sat (the "not
  satisfied" threshold is unreachable: the chill object is gone by then, crop_fruit.cpp:58-68).
- Bud break (crop_fruit.cpp:58-93) or the catch-all end requirement (7 consecutive days with the
  mean temperature >= the inducement temperature, quiescence.cpp:51-65; crop_cropsyst.cpp:
  2122-2142) restarts the tree: Crop_orchard_fruit::restart_after_inactive_period
  (crop_fruit.cpp:228-276) -> Crop_complete::restart_after_inactive_period (crop_cropsyst.cpp:
  2179-2284) -> initiate_accrescence (:2286-2330): new season statistics, a new quiescence start
  requirement, phenology restart and accrescence, thermal time cleared, a new canopy at
  initial_GAI, emergence and accrescence again; then the fruit side (max fruit load reset, roots
  updated once with an anticipated canopy of 2.5 kg/m2, harvested and reserves cleared).
- Fruit (crop_fruit.cpp:146-268, 317-360): from the filling thermal time (fructescence) a
  fraction of each day's growth goes to the fruit (initial_growth_fraction_to_fruit; from
  rapid_fruit_growth on rapid_growth_fraction_to_fruit) up to the maximum dry fruit load
  (max_fruit_load * fract_total_solids); once it is reached, or the fruit is harvested, all growth
  goes to the reserves. Leaf growth stops once the canopy LAI exceeds max_LAI. Trees keep growing
  during maturity (crop_cropsyst.cpp:1633-1640).
- Harvest: the management event synchronised after a fruit growth stage (after the crop's
  start_day, before its process_day: Simulation_unit::process_day, CS_simulation_unit.cpp:113-125,
  perform_op.cpp:1063-1170: only when the canopy LAI > min_LAI and the canopy biomass >=
  min_biomass), or the latest harvest date of the fruit harvest requirements
  (land_unit_sim.cpp:1534-1544, after the crop's process_day). Crop_orchard_fruit::harvest
  (crop_fruit.cpp:462-479) records the fruit as the yield and clears it; the canopy stays.
- Dormancy (crop_cropsyst.cpp:2143-2176): once the tree has a canopy, is past accrescence, is
  matured or in rapid fruit growth or culminescence, and the day of year is past 182: 7
  consecutive days with the mean temperature < the inducement temperature, or day 349. Then
  senesce_at_dormancy (fruit: harvested cleared, the canopy forgets its senescence period,
  crop_fruit.cpp:131-136), a new chill requirement, and start_inactive_period (:737-777): with
  dormancy senesce = 100 % the leaves drop (the canopy object is deleted), thermal time cleared,
  growth stage dormant.

Reference quirks ported as is:
- The phenology constructor gets is_fruit_tree and is_perennial swapped (crop_cropsyst.cpp:
  319-321 vs phenology_2013.cpp:9-10); harmless, both are true for these crops.
- The life cycle label perennial_woody is read as perennial; abscission is never read (the label
  table has no terminator, every file gives "none"), so there is no sloughing; "grape=grape" is
  false (only true/1 are true).
- The ET crop coefficient of a fruit tree is not adjusted by LAI (crop_cropsyst.cpp:3612-3613), on
  every day including dormancy.
- The peak LAI of a fruit tree stays 0 (only the reported peak follows), so its senescence
  reduction factor is 0 and the stress-adjusted temperature never uses it.
- Hourly air temperatures: the estimator is created when the crop is sown (in process_day, after
  the land unit's weather update of that day) with Tmax yesterday = Tmax today and Tmin tomorrow
  = Tmin today, so on the day after sowing "Tmin today" is the sowing day's Tmin.

Not ported (raise at setup): dormancy that keeps part of the canopy (dormancy senesce < 100, or
continuation mode multiple: end_season moves the canopy to effete portions), perennial crops that
are not fruit trees (alfalfa-type dormancy with biomass fate and dormant roots), vernalization or
photoperiod with perennial crops, hardiness (the VIC build multiplies leaf damage by a snow
protection factor of 0 and aborts in Phenology_2013::is_in_fructescence for fruit damage outside
anthesis).

Everything is batched over the tile axis with masks; no host syncs, no data-dependent control
flow (static loops over the 24 hours and the 6 season slots only).
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from . import canopy as cn
from . import orchard_arch as oa
from . import phenology as ph
from . import root as rt
from . import yield_ as yd
from .growth import round_f32
from .readers import IniTable, InputError

# fruit_tree_event_sequence_label_table (growth_stages.c:33-56), index = Normal_crop_event_sequence
FRUIT_STAGE_LABELS = ["none", "planting", "<reserved>", "<reserved>", "bud_break", "active_growth",
                      "end_canopy_growth", "flowering", "fruit_development", "veraison",
                      "rapid_fruit_development", "maturity", "harvestable", "harvest",
                      "postharvest", "dormant", "terminated"]
CONTINUATION = ["single", "multiple", "dormancy"]     # continuation_mode_label_table (crop_types.c)
HARVEST_MODES = ["management_scheduled", "consider_fixed_date", "consider_min_temperature",
                 "consider_avg_temperature"]           # crop_types.c:135-140
SINGLE, MULTIPLE, DORMANCY = 0, 1, 2

QSTART_DOY_AFTER = 182          # crop_cropsyst.cpp:2151 (DOY > 182)
QSTART_DOY_LATEST = 349         # crop_cropsyst.cpp:2153 (or DOY >= 349)
FRUIT_HARVEST_MIN = 1e-6        # crop_fruit.cpp:346
RESTART_ROOT_CANOPY = 2.5       # crop_fruit.cpp:262 anticipated canopy biomass for the roots
LINGER_DAYS = 5                 # crop_fruit.cpp:91, 110 (describe_periods labels)
CHILL_EDGES = (1.4, 2.4, 9.1, 12.4, 15.9, 18.0)       # crop_fruit.cpp:20-26
CHILL_UNITS = (0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0)
DEFAULTS = dict(                  # crop_param_class.cpp constructors
    chill_hours=400.0, senesce=100.0, bb_sat=100.0, bb_not_sat=150.0,     # :298-304
    q_temp=10.0, q_days=7,                                               # :286-289
    fts=0.35, mfl=20000.0, h_temp=15.0, h_days=5, h_early=258, h_late=275,  # :455-468
    igf=0.9, rgf=0.65)                                                  # :473-474


def _f32(x):
    from .growth import round_f32_bits
    return round_f32_bits(x.contiguous())


def _rep(nt, **kw):
    """NamedTuple._replace without collections internals (traceable by torch.compile)."""
    return type(nt)(*[kw[f] if f in kw else getattr(nt, f) for f in type(nt)._fields])


def is_winter(doy, south):
    """Seasons::is_winter_time (seasons.cpp:50-57)."""
    return torch.where(south, (doy > 150) & (doy < 270), (doy > 330) | (doy < 90))


def is_spring_or_summer(doy, south):
    """CS::is_spring_or_summer_time (seasons.cpp:6-13)."""
    return torch.where(south, (doy > 270) | (doy < 90), (doy > 90) & (doy < 270))


# ----------------------------------------------------------------------------- parameters

class OrchardParams(NamedTuple):
    """Per-crop orchard and perennial parameters (float32-rounded like the reference structs)."""
    is_fruit: torch.Tensor          # bool, crop model fruit
    perennial: torch.Tensor         # bool, life cycle perennial*
    continuation: torch.Tensor      # long, SINGLE / MULTIPLE / DORMANCY
    senesce: torch.Tensor           # % of the canopy that senesces at dormancy
    chill_hours: torch.Tensor
    bb_sat: torch.Tensor            # thermal time to bud break, chill requirement satisfied
    bb_not_sat: torch.Tensor
    q_temp: torch.Tensor            # inducement temperature of the inactive period
    q_days: torch.Tensor            # long, consecutive days
    fts: torch.Tensor               # fraction of total solids
    mfl: torch.Tensor               # max fruit load, fresh kg/ha
    igf: torch.Tensor               # initial growth fraction to fruit
    rgf: torch.Tensor               # rapid growth fraction to fruit
    h_temp: torch.Tensor            # fruit harvest requirements (crop_fruit.cpp:44-79)
    h_days: torch.Tensor            # long
    h_early: torch.Tensor           # long DOY
    h_late: torch.Tensor            # long DOY (also the fixed date, crop_param_struct.h:680)
    h_mode: torch.Tensor            # long, HARVEST_MODES index
    season_adj: torch.Tensor        # [..., 6] biomass adjustment by season (emergence count)
    arch: Optional[object] = None   # orchard_arch.ArchParams when a crop has canopy_architecture=true


def orchard_params(crops: IniTable, device=None, dtype=torch.float64) -> OrchardParams:
    """[n_crop] parameters from a crop table, with the reference defaults for absent keys."""
    n = crops.n
    dev = device if device is not None else next(iter(crops.params.values())).device

    def get(key, default):
        t = crops.params.get(key)
        full = torch.full((n,), float(default), dtype=dtype, device=dev)
        if t is None:
            return full
        t = t.to(dev)
        missing = torch.isnan(t) if t.dtype.is_floating_point else t == -1
        return torch.where(missing, full, t.to(dtype))

    def word(key, i, default):
        t = crops.params.get(key)
        if t is None or int(t[i]) < 0:
            return default
        return crops.vocab[key][int(t[i])]

    is_fruit = torch.tensor([word("crop.model", i, "crop") == "fruit" for i in range(n)], device=dev)
    # life_cycle_label_table (crop_types.c:30-40, CS5): the label is matched by its prefix
    perennial = torch.tensor([word("crop.life_cycle", i, "annual").startswith("perennial")
                              for i in range(n)], device=dev)
    cont = []
    for i in range(n):
        w = word("crop.season_dormancy_mode", i, None) or word("crop.continuation_mode", i, "single")
        if w not in CONTINUATION:
            raise InputError(f"cropsyst: continuation mode {w!r} unknown")
        cont.append(CONTINUATION.index(w))
    modes = []
    for i in range(n):
        w = word("fruit.harvest_condition", i, "management_scheduled")
        if w not in HARVEST_MODES:
            raise InputError(f"cropsyst: fruit harvest_condition {w!r} unknown")
        modes.append(HARVEST_MODES.index(w))
    adj = torch.ones((n, 6), dtype=dtype, device=dev)
    for k in range(2, 6):
        adj[:, k] = get(f"water_use_efficiency.season{k}_adj", 1.0)
    L = lambda x: torch.tensor(x, dtype=torch.long, device=dev)
    return OrchardParams(
        is_fruit=is_fruit, perennial=perennial, continuation=L(cont),
        senesce=_f32(get("dormancy.senesce", DEFAULTS["senesce"])),
        chill_hours=_f32(get("dormancy.chill_requirement", DEFAULTS["chill_hours"])),
        bb_sat=_f32(get("dormancy.deg_day_bud_break_chill_req_sat", DEFAULTS["bb_sat"])),
        bb_not_sat=_f32(get("dormancy.deg_day_bud_break_chill_req_not_sat", DEFAULTS["bb_not_sat"])),
        q_temp=_f32(get("inactive_period.inducement_temperature", DEFAULTS["q_temp"])),
        q_days=get("inactive_period.consider_inactive_days", DEFAULTS["q_days"]).long(),
        fts=_f32(get("fruit.fract_total_solids", DEFAULTS["fts"])),
        mfl=_f32(get("fruit.max_fruit_load", DEFAULTS["mfl"])),
        igf=_f32(get("fruit.initial_growth_fraction_to_fruit", DEFAULTS["igf"])),
        rgf=_f32(get("fruit.rapid_growth_fraction_to_fruit", DEFAULTS["rgf"])),
        h_temp=_f32(get("fruit.harvest_temperature", DEFAULTS["h_temp"])),
        h_days=get("fruit.harvest_temperature_days", DEFAULTS["h_days"]).long(),
        h_early=get("fruit.earliest_harvest", DEFAULTS["h_early"]).long(),
        h_late=get("fruit.latest_harvest", DEFAULTS["h_late"]).long(),
        h_mode=L(modes), season_adj=_f32(adj))


def check_supported(crops: IniTable, OP: OrchardParams, PP: ph.PhenologyParams):
    """Refuse (at setup) the perennial options this port does not model (module docstring)."""
    for i in range(crops.n):
        name = crops.paths[i]
        per, fruit = bool(OP.perennial[i]), bool(OP.is_fruit[i])
        tol = crops.params.get("hardiness.enabled")
        if (per or fruit) and tol is not None and bool(tol[i]):
            raise NotImplementedError(f"cropsyst: {name}: hardiness of perennial/fruit crops is not "
                                      "ported")
        if not (per or fruit):
            continue
        if not fruit:
            raise NotImplementedError(f"cropsyst: {name}: perennial crops that are not fruit trees "
                                      "(dormancy with biomass fate, dormant roots) are not ported")
        cont = int(OP.continuation[i])
        if cont == MULTIPLE or (cont == DORMANCY and abs(float(OP.senesce[i]) - 100.0) > 0.00001):
            raise NotImplementedError(f"cropsyst: {name}: an inactive period that keeps the canopy "
                                      "(effete portions) is not ported")
        if bool(PP.vern_enabled[i]) or bool(PP.photo_phen[i]):
            raise NotImplementedError(f"cropsyst: {name}: vernalization/photoperiod with a perennial "
                                      "crop is not ported")


def gather(OP: OrchardParams, idx) -> OrchardParams:
    g = lambda x: None if x is None else (type(x)(*[g(v) for v in x]) if isinstance(x, tuple) else x[idx])
    return g(OP)


class OrchardStatic(NamedTuple):
    OP: OrchardParams               # [n_crop]
    h_min_lai: torch.Tensor         # [n_mgt, H] harvest event constraint (perform_op.cpp:1145)
    h_min_bm: torch.Tensor          # [n_mgt, H] kg/m2 (perform_op.cpp:1147)
    south: torch.Tensor             # [T] bool, southern hemisphere tile
    arch: Optional[object] = None   # orchard_arch.ArchStatic (canopy architecture)


# ----------------------------------------------------------------------------- state

class OrchardState(NamedTuple):
    fruit_bm: torch.Tensor          # fruit dry biomass kg/m2
    reserves: torch.Tensor          # reserves biomass kg/m2
    latest_yield: torch.Tensor      # latest_harvest_fruit_dry_biomass
    mfl_adj: torch.Tensor           # damage-adjusted max fruit load (kg/ha)
    max_load: torch.Tensor          # bool max_fruit_load_achieved
    limit_lai: torch.Tensor         # bool limit_max_LAI
    harvested: torch.Tensor         # bool
    harvestable: torch.Tensor       # bool
    chill_on: torch.Tensor          # bool, a chill requirement exists
    chill_rem: torch.Tensor         # float32 value, chill hours remaining
    bb_on: torch.Tensor             # bool, a bud break requirement exists
    qend_on: torch.Tensor           # bool
    qend_n: torch.Tensor            # long consecutive days
    qstart_on: torch.Tensor
    qstart_n: torch.Tensor
    fh_on: torch.Tensor             # bool, fruit harvest requirements exist
    fh_n: torch.Tensor              # long consecutive days below the harvest temperature
    em_count: torch.Tensor          # long emergence (season) count
    a_tmax: torch.Tensor            # hourly air temperature estimator (land unit)
    a_tmax_y: torch.Tensor
    a_tmin: torch.Tensor
    a_tmin_t: torch.Tensor
    a_on: torch.Tensor              # bool estimator exists
    lg_chill: torch.Tensor          # long linger counters (labels only)
    lg_bb: torch.Tensor
    arch: Optional[object] = None   # orchard_arch.ArchState (canopy architecture)


def initial_state(T, dtype=torch.float64, device=None) -> OrchardState:
    z = torch.zeros(T, dtype=dtype, device=device)
    b = torch.zeros(T, dtype=torch.bool, device=device)
    n = torch.zeros(T, dtype=torch.long, device=device)
    return OrchardState(z, z, z, z, b, b, b, b, b, z, b, b, n, b, n, b, n, n, z, z, z, z, b,
                        n - 1, n - 1)


def _w(m, new, old):
    return torch.where(m, new, old)


def max_fruit_dry(o: OrchardState, OP: OrchardParams):
    """max_fruit_dry_biomass = per_ha_to_per_m2(max load) * fract_total_solids (crop_fruit.cpp:140-142)."""
    return (o.mfl_adj / 10000.0) * OP.fts


def sow_state(o: OrchardState, OP: OrchardParams, mask, doy, south, tmax, tmin) -> OrchardState:
    """Crop_orchard_fruit::initialize and start (crop_fruit.cpp:174-226) and the requirement set-up
    of Crop_complete::start (crop_cropsyst.cpp:605-611); the land unit's hourly temperature
    estimator is created (reset with Tmax yesterday = Tmax today, Tmin tomorrow = Tmin today,
    subdaily.cpp:120-140; VIC's Weather_spanning with the fixed provided dates)."""
    m = mask
    z = torch.zeros_like(o.fruit_bm)
    zb = torch.zeros_like(o.max_load)
    zn = torch.zeros_like(o.qend_n)
    winter = is_winter(doy, south) & OP.is_fruit
    qend = winter
    qstart = ~qend & (OP.continuation > SINGLE)
    return OrchardState(
        fruit_bm=_w(m, z, o.fruit_bm), reserves=_w(m, z, o.reserves), latest_yield=_w(m, z, o.latest_yield),
        mfl_adj=_w(m, OP.mfl + z, o.mfl_adj), max_load=o.max_load & ~m, limit_lai=o.limit_lai & ~m,
        harvested=o.harvested & ~m, harvestable=o.harvestable & ~m,
        chill_on=_w(m, winter, o.chill_on), chill_rem=_w(m, _w(winter, OP.chill_hours + z, z), o.chill_rem),
        bb_on=o.bb_on & ~m, qend_on=_w(m, qend, o.qend_on), qend_n=_w(m, zn, o.qend_n),
        qstart_on=_w(m, qstart, o.qstart_on), qstart_n=_w(m, zn, o.qstart_n),
        fh_on=o.fh_on & ~m, fh_n=_w(m, zn, o.fh_n), em_count=_w(m, zn, o.em_count),
        a_tmax=_w(m, tmax + z, o.a_tmax), a_tmax_y=_w(m, tmax + z, o.a_tmax_y),
        a_tmin=_w(m, tmin + z, o.a_tmin), a_tmin_t=_w(m, tmin + z, o.a_tmin_t), a_on=o.a_on | m,
        lg_chill=_w(m, zn - 1, o.lg_chill), lg_bb=_w(m, zn - 1, o.lg_bb), arch=o.arch)


def start_day(o: OrchardState, sown, tmax, tmin_tomorrow) -> OrchardState:
    """Land unit start of day: the weather update shifts the hourly estimator
    (Temperature_estimator::set_max_today_min_tomorrow, temperature_est.cpp:27-33; subdaily.cpp:
    22-30) on every day after the one it was created; the linger counters count down
    (crop_fruit.cpp:47-52)."""
    upd = o.a_on & ~sown
    z = torch.zeros_like(o.a_tmax)
    return _rep(o, a_tmax_y=_w(upd, o.a_tmax, o.a_tmax_y), a_tmin=_w(upd, o.a_tmin_t, o.a_tmin),
                      a_tmax=_w(upd, tmax + z, o.a_tmax), a_tmin_t=_w(upd, tmin_tomorrow + z, o.a_tmin_t),
                      lg_chill=o.lg_chill - 1, lg_bb=o.lg_bb - 1)


def air_hours(o: OrchardState):
    """[..., 24] float32 hourly air temperatures (min_max_to_timestep_estimator.cpp:22-38,
    temperature_est.cpp:8-24)."""
    twf, before, after = yd._hour_tables(o.a_tmax.device, o.a_tmax.dtype)
    tn = torch.where(before, o.a_tmin[..., None], torch.where(after, o.a_tmin_t[..., None], o.a_tmin[..., None]))
    tx = torch.where(before, o.a_tmax_y[..., None], o.a_tmax[..., None])
    return _f32(twf * tx + (1.0 - twf) * tn)


def chill_units(t):
    """Fruit_chill_requirement::add_hour_temperature (crop_fruit.cpp:17-31)."""
    u = torch.full_like(t, CHILL_UNITS[-1])
    for edge, unit in zip(reversed(CHILL_EDGES), reversed(CHILL_UNITS[:-1])):
        u = torch.where(t < edge, torch.full_like(t, unit), u)
    return u


def add_chill(rem, hours):
    """24 sequential float32 deductions (chill_hours_remaining is float32)."""
    u = chill_units(hours)
    for h in range(24):                 # the hours of the day (static)
        rem = _f32(rem - u[..., h])
    return rem


def fruit_harvest_check(o: OrchardState, OP: OrchardParams, doy, tmin, tavg):
    """Fruit_harvest_requirements::is_satisfied (crop_fruit.cpp:44-79) for the tiles that have the
    requirements: returns (satisfied, new consecutive-day count)."""
    latest = doy == OP.h_late
    window = (doy >= OP.h_early) & (doy <= OP.h_late)
    below_min = window & (tmin <= OP.h_temp)
    below_avg = window & (tavg <= OP.h_temp)
    below = torch.where(OP.h_mode == 2, below_min, below_avg)
    counting = (OP.h_mode >= 2) & ~latest
    n = torch.where(counting, torch.where(below, o.fh_n + 1, torch.zeros_like(o.fh_n)), o.fh_n)
    sat = latest | ((OP.h_mode == 1) & (doy == OP.h_late)) | (counting & below & (n >= OP.h_days))
    n = torch.where(o.fh_on, n, o.fh_n)
    return o.fh_on & sat, n


def distribute(o: OrchardState, OP: OrchardParams, stage, growth, lai, max_lai, mask):
    """Crop_orchard_fruit::distribute_todays_non_leaf_stem_growth (crop_fruit.cpp:146-268) on the
    masked fruit tiles: returns (leaf and branch growth, state). `lai` is the canopy LAI after the
    day's die-back (get_LAI(true), crop_fruit.cpp:205-208)."""
    m = mask & OP.is_fruit
    to_res = o.max_load | o.harvested
    frac = torch.where(stage == ph.NGS_FILLING, OP.igf,
                       torch.where((stage == ph.NGS_RIPENING) | (stage == ph.FGS_RAPID_FRUIT_DEVELOPMENT),
                                   OP.rgf, torch.zeros_like(OP.rgf)))
    today = growth * frac
    fb = o.fruit_bm + today
    mx = max_fruit_dry(o, OP)
    over = fb > mx
    today = torch.where(over, torch.zeros_like(today), today)
    fb = torch.where(over, mx, fb)
    leaf = torch.where(to_res, torch.zeros_like(growth), growth - today)
    res = torch.where(to_res, o.reserves + growth, o.reserves)
    fb = torch.where(to_res, o.fruit_bm, fb)
    lim = o.limit_lai | (lai > max_lai)
    return torch.where(m, leaf, growth), _rep(o, 
        fruit_bm=_w(m, fb, o.fruit_bm), reserves=_w(m, res, o.reserves), limit_lai=_w(m, lim, o.limit_lai))


# ----------------------------------------------------------------------------- life cycle events

def _mark(arr, m, code, day):
    k = torch.arange(arr.shape[-1], device=arr.device)
    return torch.where(m[..., None] & (k == code), day, arr)


def restart(phn: ph.PhenologyState, g, o: OrchardState, OP: OrchardParams, PP: ph.PhenologyParams, GP,
            m, day, thickness):
    """Crop_orchard_fruit::restart_after_inactive_period on the masked tiles (crop_fruit.cpp:228-276,
    crop_cropsyst.cpp:2179-2330); returns (phenology state, growth state, orchard state)."""
    from .daily import _rep as grep
    dayk = torch.as_tensor(day, dtype=torch.long, device=m.device)
    mk = m[..., None]
    k = torch.arange(ph.NGS_COUNT, device=m.device)
    # phenology: activate_restart / restart_season, accrescence, reset(EMERGENCE..), reset(RESTART..),
    # thermal time cleared, then initiate_accrescence: emergence, accrescence (phenology_2013.cpp)
    init = torch.where(mk & (k >= ph.NGS_EMERGENCE), torch.full_like(phn.init_day, -1), phn.init_day)
    init = _mark(init, m, ph.NGS_EMERGENCE, dayk)
    init = _mark(init, m, ph.NGS_ACCRESCENCE, dayk)
    sync = phn.sync_day
    for code in (ph.NGS_RESTART, ph.NGS_EMERGENCE, ph.NGS_ACCRESCENCE):
        sync = _mark(sync, m, code, dayk)
    j = torch.arange(4, device=m.device)
    acc = j == ph.ACC
    pon = torch.where(mk & acc, torch.ones_like(phn.period_on), phn.period_on)
    pon = torch.where(mk & ((j == ph.CULM) | (j == ph.SEN)), torch.zeros_like(pon), pon)
    z = torch.zeros_like(phn.gdds)
    phn = ph.PhenologyState(
        present=phn.present, stage=torch.where(m, torch.full_like(phn.stage, ph.NGS_ACCRESCENCE), phn.stage),
        gdd=phn.gdd, gdds=torch.where(m, z, phn.gdds), gdds_yesterday=torch.where(m, z, phn.gdds_yesterday),
        vern_crop=phn.vern_crop, vern_phen=phn.vern_phen, vfac_crop=phn.vfac_crop, vfac_phen=phn.vfac_phen,
        period_on=pon, period_tt=torch.where(mk & acc, torch.zeros_like(phn.period_tt), phn.period_tt),
        period_rel_elapsed=torch.where(mk & acc, torch.zeros_like(phn.period_rel_elapsed), phn.period_rel_elapsed),
        period_rel_remaining=torch.where(mk & acc, torch.zeros_like(phn.period_rel_remaining),
                                         phn.period_rel_remaining),
        canopy=phn.canopy | m, temps=phn.temps, adj_done=phn.adj_done & ~mk, init_day=init, sync_day=sync,
        tuber_day=phn.tuber_day, season_days=torch.where(m, torch.ones_like(phn.season_days), phn.season_days),
        tmax_adj=phn.tmax_adj)
    # growth: new canopy at initial_GAI (provide_canopy + restart_with), roots told the initial
    # biomass, then (fruit) updated once with an anticipated canopy of 2.5 kg/m2 and the water stress
    # of the cleared statistics (1.0)
    c = cn.restart(g.canopy, m, GP.initial_gai, GP.sla, GP.kc)
    b0 = (GP.initial_gai.to(torch.float32) / GP.sla.to(torch.float32)).to(z.dtype)
    r = g.root
    one = torch.ones_like(z)
    rel = phn.period_rel_remaining[..., ph.ROOT].to(z.dtype)
    exp_ = phn.period_tt[..., ph.ROOT] > PP.dur_root_elongation
    ru = rt.update(r, one, RESTART_ROOT_CANOPY + z, phn.period_tt[..., ph.ROOT], PP.dur_root_elongation, rel,
                   exp_, thickness, GP.root_max_depth, GP.root_surface_density, GP.root_curvature,
                   GP.root_sensitivity, GP.root_length_emergence_cm, GP.root_shoot_emergence,
                   GP.root_shoot_full)
    rf = grep(r, above_ground=torch.where(m, b0, r.above_ground))
    ru = rt.RootState(*[torch.where(OP.is_fruit.reshape(OP.is_fruit.shape + (1,) * (a.dim() - OP.is_fruit.dim())),
                                    a, b) for a, b in zip(ru, rf)])
    r = rt.RootState(*[torch.where(m.reshape(m.shape + (1,) * (a.dim() - m.dim())), a, b) for a, b in zip(ru, r)])
    g = grep(g, canopy=c, root=r, growth_on=g.growth_on | m, ws=torch.where(m, one, g.ws),
             ws_index=torch.where(m, z, g.ws_index), gsp_done=g.gsp_done & ~mk)
    o = _rep(o, mfl_adj=_w(m, OP.mfl + z, o.mfl_adj), bb_on=o.bb_on & ~m, chill_on=o.chill_on & ~m,
             qstart_on=o.qstart_on | m, qstart_n=_w(m, torch.zeros_like(o.qstart_n), o.qstart_n),
             em_count=o.em_count + m.to(torch.long), limit_lai=o.limit_lai & ~m, max_load=o.max_load & ~m,
             harvested=o.harvested & ~m, reserves=_w(m, z, o.reserves), harvestable=o.harvestable & ~m)
    if o.arch is not None:                  # set_deg_days_begin_canopy_development (crop_fruit.cpp:262-270)
        o = _rep(o, arch=oa.restart(o.arch, m & OP.is_fruit, phn.gdds))
    return phn, g, o


def dormancy(phn: ph.PhenologyState, g, o: OrchardState, OP: OrchardParams, m, day):
    """Start of dormancy on the masked tiles: senesce_at_dormancy (crop_fruit.cpp:131-136), a new chill
    requirement (crop_fruit.cpp:277-288), start_inactive_period with the leaves dropped
    (crop_cropsyst.cpp:737-777)."""
    from .daily import _rep as grep
    dayk = torch.as_tensor(day, dtype=torch.long, device=m.device)
    z = torch.zeros_like(phn.gdds)
    phn = _rep(phn, stage=torch.where(m, torch.full_like(phn.stage, ph.NGS_QUIESCENCE), phn.stage),
               gdds=torch.where(m, z, phn.gdds), gdds_yesterday=torch.where(m, z, phn.gdds_yesterday),
               canopy=phn.canopy & ~m, init_day=_mark(phn.init_day, m, ph.NGS_QUIESCENCE, dayk),
               sync_day=_mark(phn.sync_day, m, ph.NGS_QUIESCENCE, dayk))
    c = cn.drop(g.canopy, m)
    g = grep(g, canopy=c, intercepted=torch.where(m, z, g.intercepted))
    o = _rep(o, harvested=o.harvested & ~m, chill_on=o.chill_on | m,
             chill_rem=_w(m, OP.chill_hours + z, o.chill_rem), qstart_on=o.qstart_on & ~m)
    return phn, g, o


def harvest(o: OrchardState, m):
    """Crop_orchard_fruit::harvest (crop_fruit.cpp:462-479)."""
    return _rep(o, harvested=o.harvested | m, harvestable=o.harvestable & ~m,
                latest_yield=_w(m, o.fruit_bm, o.latest_yield),
                fruit_bm=_w(m, torch.zeros_like(o.fruit_bm), o.fruit_bm))


def mark_harvest(phn: ph.PhenologyState, m, day):
    """phenology.activate_harvest (phenology_2013.cpp activate_stage(NGS_HARVEST))."""
    dayk = torch.as_tensor(day, dtype=torch.long, device=m.device)
    return _rep(phn, init_day=_mark(phn.init_day, m, ph.NGS_HARVEST, dayk))


def inactive_period(phn: ph.PhenologyState, g, o: OrchardState, OP: OrchardParams, PP: ph.PhenologyParams,
                    GP, on, day, doy, tavg, south, thickness):
    """Crop_orchard_fruit::consider_inactive_period (crop_fruit.cpp:54-130) with
    Crop_complete::consider_inactive_period (crop_cropsyst.cpp:2119-2178) inside, for the fruit tiles
    `on`. Returns (phenology, growth, orchard state, restarted mask, dormancy-start mask)."""
    t32 = _f32(tavg)
    hours = air_hours(o)
    # bud break (evaluated from the day after the chill requirement was met)
    thr = torch.where(~o.chill_on, OP.bb_sat, OP.bb_not_sat)
    bb = on & o.bb_on & (_f32(phn.gdds) >= thr)
    phn = _rep(phn, sync_day=_mark(phn.sync_day, bb, ph.NGS_RESTART,
                                   torch.as_tensor(day, dtype=torch.long, device=on.device)))
    o = _rep(o, qend_on=o.qend_on & ~bb, latest_yield=_w(bb, torch.zeros_like(o.latest_yield), o.latest_yield),
             lg_bb=_w(bb, torch.full_like(o.lg_bb, LINGER_DAYS), o.lg_bb))
    # quiescence end requirement (catch-all of the first winter): checked before today's temperature
    qe = on & o.qend_on
    sat_e = qe & (o.qend_n >= OP.q_days)
    cnt = torch.where(t32 >= OP.q_temp, o.qend_n + 1, torch.zeros_like(o.qend_n))
    o = _rep(o, qend_on=o.qend_on & ~sat_e, qend_n=torch.where(qe & ~sat_e, cnt, o.qend_n))
    rs = bb | sat_e
    phn, g, o = restart(phn, g, o, OP, PP, GP, rs, day, thickness)
    # quiescence start requirement (start of dormancy)
    pon, tt = phn.period_on, phn.period_tt
    in_acc = pon[..., ph.ACC]
    in_culm = pon[..., ph.CULM] & ~(tt[..., ph.CULM] > PP.dur_culminescence)
    matured = phn.stage >= ph.NGS_MATURITY
    rapid = phn.stage == ph.FGS_RAPID_FRUIT_DEVELOPMENT
    gate = on & o.qstart_on & g.canopy.exists & ~(OP.is_fruit & in_acc) & \
        (matured | rapid | in_culm | in_acc) & (doy > QSTART_DOY_AFTER)
    sat_s = gate & ((o.qstart_n >= OP.q_days) | (doy >= QSTART_DOY_LATEST))
    cnt = torch.where(t32 < OP.q_temp, o.qstart_n + 1, torch.zeros_like(o.qstart_n))
    o = _rep(o, qstart_n=torch.where(gate & ~sat_s, cnt, o.qstart_n))
    g = _rep(g, canopy=cn._rep(g.canopy, sen_known=g.canopy.sen_known & ~sat_s))
    phn, g, o = dormancy(phn, g, o, OP, sat_s, day)
    # chill requirement
    ch = on & o.chill_on
    met = ch & (o.chill_rem < 0.0)
    rem = add_chill(o.chill_rem, hours)
    rem = torch.where(is_spring_or_summer(doy, south) & (rem > 0.0), torch.full_like(rem, -1.0), rem)
    o = _rep(o, chill_on=o.chill_on & ~met, bb_on=o.bb_on | met,
             lg_chill=_w(met, torch.full_like(o.lg_chill, LINGER_DAYS), o.lg_chill),
             chill_rem=_w(ch & ~met, rem, o.chill_rem))
    return phn, g, o, rs, sat_s


# ----------------------------------------------------------------------------- the day of a run

def season_factor(o: OrchardState, OP: OrchardParams):
    """Biomass_growth_RUE_TUE: perennials use season_biomass_adjustment[min(emergence count, 5)]
    (biomass_growth_RUE_TUE.cpp:126-132), others the continuous cultivation factor (1)."""
    k = o.em_count.clamp(min=0, max=5)
    f = OP.season_adj.gather(-1, k[..., None])[..., 0]
    return torch.where(OP.perennial, f, torch.ones_like(f))


def day_begin(o: OrchardState, OP: OrchardParams, phen: ph.PhenologyState, grow, PP: ph.PhenologyParams, GP,
              sow, day, doy, south, tmax, tmin, tmin_tom, thick, AS=None):
    """Before the crop's start_day: the land unit's weather update (hourly estimator) and, on the tiles
    sown today with a perennial crop, Crop_orchard_fruit::start and Crop_complete::start
    (crop_cropsyst.cpp:597-640): roots at once (create_roots, crop_cropsyst.cpp:1514-1548; at the
    maximum length because the crop is established), the root elongation period started (its first
    accumulation sees the crop's stress-adjusted temperature still at 0), the plant at its maximum
    height."""
    from .daily import _rep as grep
    o = start_day(o, sow, tmax, tmin_tom)
    per = sow & OP.perennial
    o = sow_state(o, OP, sow & OP.is_fruit, doy, south, tmax, tmin)
    if o.arch is not None:                  # the crop's Orchard_common (perform_op.cpp:430-436)
        o = _rep(o, arch=oa.sow(o.arch, OP.arch, AS, sow & OP.is_fruit))
    z = torch.zeros_like(phen.gdds)
    r = rt.create(grow.root, per, GP.root_length_emergence_cm, thick)
    est = per & (OP.continuation > SINGLE)             # is_established (crop_cropsyst.cpp:2843-2845)
    l0 = GP.root_max_depth + z                         # float32 parameter (crop_cropsyst.cpp:1537-1538)
    r = grep(r, length=torch.where(est, l0, r.length), unstr_today=torch.where(est, l0, r.unstr_today),
             unstr_yday=torch.where(est, l0, r.unstr_yday), depth=torch.where(est, l0, r.depth))
    c = grep(grow.canopy, height=torch.where(sow & (OP.continuation > SINGLE), GP.max_height + z,
                                             grow.canopy.height))
    grow = grep(grow, root=r, canopy=c)
    tt0 = ph.growing_degree_day(z, tmin + z, phen.temps, PP.nonlinear)
    dur = PP.dur_root_elongation
    j = torch.arange(4, device=sow.device)
    pm = per[..., None] & (j == ph.ROOT)
    rel_r = round_f32(torch.where((dur - tt0) / dur < 0.0, z, (dur - tt0) / dur)).to(torch.float32)
    phen = _rep(phen, period_on=phen.period_on | pm,
                period_tt=torch.where(pm, tt0[..., None], phen.period_tt),
                period_rel_elapsed=torch.where(pm, (tt0 / dur)[..., None], phen.period_rel_elapsed),
                period_rel_remaining=torch.where(pm, rel_r[..., None], phen.period_rel_remaining))
    return o, phen, grow


def after_start(phn: ph.PhenologyState, g, o: OrchardState, OP: OrchardParams, PP, GP, day, doy, tmin, tavg,
                south, thick, h_stage, h_offset, h_min_lai, h_min_bm, AS=None, srad=None):
    """Between the crop's start_day and the first half of its process_day: the side effects of the
    fruit thermal-time events (initiate_fruit_growth, initiate_rapid_fruit_growth, crop_fruit.cpp:
    298-341), the emergence count of initiate_accrescence, the harvest event of the management
    (performed by the land unit before the crop's process_day), the fruit harvest requirements
    (crop_fruit.cpp:343-347) and the inactive period. h_*: the harvest events of each tile's
    management [T, H]. Returns (phenology, growth, orchard, restarted, harvested by an event,
    season biomass factor)."""
    on = phn.present & OP.is_fruit
    dk = torch.as_tensor(day, dtype=torch.long, device=on.device)
    hit = lambda code: phn.present & (phn.sync_day[..., code] == dk)
    o = _rep(o, em_count=o.em_count + hit(ph.NGS_EMERGENCE).to(torch.long))
    ff = on & hit(ph.NGS_FILLING)
    o = _rep(o, fruit_bm=_w(ff, torch.zeros_like(o.fruit_bm), o.fruit_bm), harvestable=o.harvestable & ~ff)
    fr = on & hit(ph.FGS_RAPID_FRUIT_DEVELOPMENT)
    o = _rep(o, fh_on=o.fh_on | fr, fh_n=_w(fr, torch.zeros_like(o.fh_n), o.fh_n))
    # harvest event (perform_op.cpp:1140-1150: canopy LAI > min_LAI, canopy biomass >= min_biomass)
    c = g.canopy
    lai = torch.where(c.exists, cn.lai(c), torch.zeros_like(o.fruit_bm))
    bm = torch.where(c.exists, cn.biomass_vital(c), torch.zeros_like(o.fruit_bm))
    sd = phn.sync_day.gather(-1, h_stage.clamp(min=0))
    fire = on[..., None] & (h_stage >= 0) & (sd >= 0) & (dk == sd + h_offset)
    fire = fire & (lai[..., None] > h_min_lai) & (bm[..., None] >= h_min_bm)
    hv = fire.any(-1)
    o = harvest(o, hv)
    # the land unit harvests a fruit crop its process_day of yesterday made harvestable, before the
    # crop's process_day (land_unit_sim.cpp:1534-1544 precedes :1831; default operation: LAI > 0)
    hv2 = on & o.harvestable & (lai > 0.0) & (bm >= 0.0)
    o = harvest(o, hv2)
    hv = hv | hv2
    phn = mark_harvest(phn, hv, day)
    # Crop_orchard_fruit::process_day: canopy architecture (expand, interception), fruit harvest
    # requirements, then consider_inactive_period
    if o.arch is not None:
        o = _rep(o, arch=oa.update(o.arch, OP.arch, AS, on, phn.gdds, doy, srad))
    sat, fh_n = fruit_harvest_check(o, OP, doy, tmin, tavg)
    o = _rep(o, fh_n=torch.where(on, fh_n, o.fh_n),
             harvestable=o.harvestable | (on & sat & (o.fruit_bm > FRUIT_HARVEST_MIN)))
    phn, g, o, rs, ds = inactive_period(phn, g, o, OP, PP, GP, on, day, doy, tavg, south, thick)
    return phn, g, o, rs, ds, hv, season_factor(o, OP)


def after_actual(phn: ph.PhenologyState, g, o: OrchardState, OP: OrchardParams, day, thick):
    """After the crop's process_day: max fruit load (crop_fruit.cpp:350-351) and the end_day root
    depth (crop_cropsyst.cpp:2094-2101). Returns (phenology, growth, orchard)."""
    from .daily import _rep as grep
    on = phn.present & OP.is_fruit
    o = _rep(o, max_load=o.max_load | (on & (o.fruit_bm >= max_fruit_dry(o, OP))))
    r = g.root
    d = rt.apparent_depth(r.length, r.start_depth, thick)
    g = grep(g, root=grep(r, depth=torch.where(on & r.exists, d, r.depth)))
    return phn, g, o


class OrchardDayOutput(NamedTuple):
    fruit_bm: torch.Tensor          # kg/m2 after the day
    chill_remaining: torch.Tensor   # get_fruit_chill_hours_remaining (0 without a chill requirement)
    reserves: torch.Tensor
    harvest: torch.Tensor           # bool, a fruit harvest today
    restart: torch.Tensor           # bool, bud break / restart today
    dormant: torch.Tensor           # bool, stage dormant after the day
    interception: Optional[torch.Tensor] = None   # solid canopy interception (canopy architecture)
    width: Optional[torch.Tensor] = None          # canopy width across the row (m)
    breadth: Optional[torch.Tensor] = None        # canopy breadth along the row (m)
    height: Optional[torch.Tensor] = None         # canopy height of the architecture (m)


def day_output(o: OrchardState, phn: ph.PhenologyState, hv, rs):
    a = o.arch
    return OrchardDayOutput(o.fruit_bm, torch.where(o.chill_on, o.chill_rem, torch.zeros_like(o.chill_rem)),
                            o.reserves, hv, rs, phn.stage == ph.NGS_QUIESCENCE,
                            *((None,) * 4 if a is None else (a.interception, a.width, a.breadth, a.height)))
