"""Clipping of the CropSyst crop backend: the forage cutting of alfalfa (lucerne) hay
and other clipped crops, by date, by condition and by an automatic clipping period.

Sources (CS = VIC-CropSyst-Package/CropSyst/source; every equation re-derived, no code copied):

  management             management/management_param_V4.cpp:231-319 (harvest and automatic
                         clipping keys: min_LAI, min_biomass, reserve_biomass, remove_amount,
                         min_retain_GAI, the fate percents, adj_rel_growth, max_LAI, max_biomass,
                         after_flowering; float32 fields, int16 percents),
                         management_param_class.cpp:170-219 (defaults: biomass_fate harvest_once,
                         harvest_amount_mode crop_param, return_as_manure 80, max_biomass 4000),
                         mgmt_types.cpp:39-73 / mgmt_types.h:53-110 (modes, fates, causes)
  scheduling             common/simulation/event.cpp:409-445 (an event is dated once, the first
                         time its growth stage is triggered: later triggers do not move it),
                         synchronization.cpp:282-311 (stage + offset, actual date),
                         CS_event_scheduler.cpp:441-470 (aftermath: a failed event with contingency
                         'postpone' is retried the next day; 'none', 'abort', 'stop' eat it: the VIC
                         land unit has no run loop that reads the aborted/viable flags),
                         CS_simulation_unit.cpp:114-137 (events are performed at the start of the
                         land unit's process_day, after the crop's start_day and before its
                         process_day, land_unit_sim.cpp:1503-1505, 1831)
  harvest / clipping     perform_op.cpp:1063-1258 (perform_harvest_or_clipping: fixed amount
                         demand and reserve, accept_less, LAI > min_LAI and biomass >= min_biomass
                         or terminate; event status FAIL 32768, limited commodity 1024, out of
                         season 16; last clipping: no more clippings, end_season_if_no_dormancy)
                         crop/crop_cropsyst.cpp:2430-2507 (Crop_complete::harvest: cause, the
                         harvest-index route harvest_crop :1830-1880 when the crop applies a
                         harvest index (crop_param_class.cpp:588-593: model crop, part below canopy,
                         not PASTURE) or the mode is crop_param; terminate, else respond_to_clipping)
  biomass fate           crop/crop_cropsyst.cpp:3210-3482 (process_biomass_fate: the yield first,
                         newest first; then the mode's removal; fixed: the fates of what was
                         removed; others: the fates of the unharvested canopy; terminate: all the
                         rest to standing stubble and all roots dead)
  canopy removal         crop/canopy_growth_portioned.cpp:103-160 (Canopy_accumulation::
                         remove_biomass), :272-325 (remove_all_biomass, remove_biomass),
                         canopy_growth_LAI_based.cpp:196-211 (the self-shading LAI becomes the GAI),
                         :295-300 (respond_to_clipping), :302-326 (restart_with), :425-432
                         (terminate), :111-114 (dec_living)
  regrowth               crop/crop_cropsyst.cpp:2510-2594 (respond_to_clipping), :2286-2323
                         (initiate_accrescence), thermal_time_common.cpp:293-306 (reset_to),
                         phenology_2013.cpp:217-234, phenology_A.cpp:377-395 (period response),
                         phenology_2013.cpp:264-282 (activate emergence, accrescence),
                         phenology_A.cpp:72-84 (new accrescence deletes culminescence, senescence)
  automatic clipping     perform_op.cpp:1262-1268 (period start), :290-303 (period end: clip now
                         or never), CS_event_scheduler.cpp:256-300, event.cpp:603-620 (the end is
                         performed the day after the end date), :1703-1757 (process_any_automatic_clipping), land_unit_sim.cpp:
                         1828-1835 (after the crop's process_day), crop_cropsyst.cpp:3658-3815
                         (check_for_automatic_clipping, clip_biomass)
  nitrogen               crop/crop_N_common.cpp:1126-1145 (remove_canopy_nitrogen_for),
                         land_unit_sim.cpp:2193-2270 (distribute_any_crop_biomass_fate at end_day:
                         chaff flat, stubble standing, dead roots; residue N concentration)

Reference facts ported as is:
- Canopy_accumulation::remove_biomass pops a portion it may not remove (the living index would
  fall below the GAI to retain) and never puts it back: the portion leaves the canopy without
  being counted as removed (its N stays in the canopy N mass). With min_retain_GAI the canopy is
  therefore not kept at that GAI; most portions vanish.
- A terminating harvest appends the untouchable restart portion to the tail of the vital list
  without invalidating the list's cached totals (canopy_growth_LAI_based.cpp:425-432): the canopy
  biomass and the living index the first removal sees exclude it (as n_engine.harvest_fate).
- The harvest-index route takes the yield newest first whatever clip_newest says
  (crop_cropsyst.cpp:3242-3248), with the reserve biomass ignored (harvest_crop passes 0).
- respond_to_clipping resets the crop thermal time to the emergence threshold, but each active
  period (accrescence, culminescence, senescence) starts again from 0 (phenology_A.cpp:380-382):
  initiate_accrescence then gives the crop a new accrescence period and a NEW canopy
  (provide_canopy always deletes the old one, crop_cropsyst.cpp:779-832) at the initial GAI and
  biomass: whatever the clipping left standing vanishes, unaccounted, and its N stays in the
  crop's canopy N mass. Only crops without clipping_resets (or clipped with no active period)
  regrow from the stubble.
- Without clipping_resets a clipping in flowering or filling sets the crop's flowering and filling
  thresholds to 0 (crop_cropsyst.cpp:2563-2569): neither event happens again for that crop.
- The end of an automatic clipping period synchronised to sowing is never dated: the period's
  end date stays 0 (Period::update_end_period returns at once, event.cpp:603-607; seen in the
  reference dump of the cs-clip case), so the mode runs until the crop ends. An end synchronised to
  another stage (emergence) is dated and performed the day after the end date.
- The fate of an automatic clipping is deleted, not distributed (crop_cropsyst.cpp:3770-3784):
  with nitrogen the clipped biomass leaves the field whatever the percents say; its yield + useful
  still shows in the day's Yield_kg_m2 (record_biomass_fate).
- The automatic clipping reads the self-shading LAI (get_LAI(false), crop_cropsyst.cpp:3672),
  min_LAI with >= (the scheduled events use LAI(canopy) > min_LAI, perform_op.cpp:1145).
- Yield_kg_m2 of a clipping day is yield + useful of the day's fates (crop_cropsyst.h:851,
  VIC_land_unit_C_interface.cpp:666).

Fruit trees (orchard.py): a harvest or clipping event of a fruit tree is its fruit harvest
(Crop_orchard_fruit::harvest, crop_fruit.cpp:792-806: no biomass fate); the land unit's checks and
the last-clipping rules apply. The automatic clipping does apply its biomass fate to a fruit
tree's canopy and responds to it (clip_biomass calls respond_to_clipping for every crop).

Grazing: the grazed biomass and its manure share are computed (fates); the manure the land unit
holds until the end of the growing season (land_unit_sim.cpp:2041-2085, a biomatter application)
is not applied: the port refuses biomatter/organic applications with nitrogen on (n_engine).

Everything is a masked tensor operation over the tiles [T] (and [T, P] portions, [T, H] events);
the in-order canopy removal is a per-tile walk over the portions (torch loop on the CPU, one
Triton lane per tile on CUDA, fate_kernel style): no host syncs, no data-dependent control flow
on CUDA, capturable in a CUDA graph.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from . import canopy as cn
from . import phenology as ph

IS_ZERO_TOL = 0.0000001
ALL = 999999.0                                  # canopy_growth_portioned.cpp:276-277
LOW_GAI_RATE = 0.05                             # crop_cropsyst.cpp:3663
ES_FAIL, ES_LIMITED, ES_OUT_OF_SEASON, ES_COMPLETE = 32768, 1024, 16, 0
NEVER = -29999                                  # phenology_2013.cpp:30

# Harvest_amount_mode (mgmt_types.h:55-60) and the fate modes / causes (mgmt_types.h:62-110)
MODE_WORDS = ["crop_param", "remove_biomass", "percent_biomass", "retain_GAI", "other"]
CROP_PARAM, FIXED, PERCENT, RETAIN_GAI, OTHER = range(5)
FATE_WORDS = ["harvest_once", "clipping", "last_clipping", "grazing", "defoliation", "harvest_index"]
HARVEST_ONCE, CLIPPING, LAST_CLIPPING, GRAZE = 0, 1, 2, 3
CLIP_NOW_OR_NEVER, INSUFFICIENT, LOW_GROWTH, MAX_BIOMASS, MAX_LAI, AFTER_FLOWERING = 6, 7, 8, 9, 10, 11
NO_CAUSE = 99
# biomass fate percents, int16 (mgmt_types.h:112-140); index in the pct vector
PCT_KEYS = ["use", "graze", "residue", "dispose", "dead", "live", "root_dead", "root_loss", "root_live",
            "return_as_manure"]
USE, GRAZE_P, RESIDUE, DISPOSE, DEAD, LIVE, ROOT_DEAD, ROOT_LOSS, ROOT_LIVE, MANURE = range(10)
PCT_DEFAULT = [0, 0, 0, 0, 100, 100, 100, 100, 100, 80]          # management_param_V4.cpp:239-248
SYNC_STAGE, SYNC_FRUIT, SYNC_DATE, SYNC_NONE = 0, 1, 2, 3


def _f32(x):
    return x.to(torch.float32).to(x.dtype)


def _rep(nt, **kw):
    return type(nt)(*[kw[f] if f in kw else getattr(nt, f) for f in type(nt)._fields])


def _is_zero(x):
    d = 0.0 - x
    return torch.where(d < 0.0, -d, d) <= IS_ZERO_TOL


def _cmin(a, b):
    return torch.where(b < a, b, a)


def _cmax(a, b):
    return torch.where(a < b, b, a)


# ----------------------------------------------------------------------------- static data

class ClipEvents(NamedTuple):
    """Harvest/clipping events that take the clipping path, [n_mgt, H] (or gathered [T, H])."""
    on: torch.Tensor            # bool
    sync: torch.Tensor          # long SYNC_*
    stage: torch.Tensor         # long growth stage (normal or fruit sequence index)
    offset: torch.Tensor        # long days
    date: torch.Tensor          # long run day of an actual date (-1)
    terminate: torch.Tensor     # bool
    fate: torch.Tensor          # long FATE (biomass_fate)
    mode: torch.Tensor          # long MODE
    remove: torch.Tensor        # kg/m2 (float32 kg/ha / 10000)
    reserve: torch.Tensor       # kg/m2
    min_lai: torch.Tensor       # float32
    min_bm: torch.Tensor        # kg/m2
    retain: torch.Tensor        # min_retain_GAI float32
    accept: torch.Tensor        # bool accept_less
    postpone: torch.Tensor      # bool contingency postpone
    pct: torch.Tensor           # [..., 10] percents


class AutoClip(NamedTuple):
    """Automatic clipping periods, [n_mgt, K] (or gathered [T, K])."""
    on: torch.Tensor
    b_sync: torch.Tensor
    b_stage: torch.Tensor
    b_offset: torch.Tensor
    b_date: torch.Tensor
    e_sync: torch.Tensor
    e_stage: torch.Tensor
    e_offset: torch.Tensor
    e_date: torch.Tensor
    mode: torch.Tensor
    remove: torch.Tensor
    reserve: torch.Tensor
    min_lai: torch.Tensor
    min_bm: torch.Tensor
    retain: torch.Tensor
    accept: torch.Tensor
    pct: torch.Tensor           # [..., 10]
    adj: torch.Tensor           # adj_rel_growth float32
    max_lai: torch.Tensor       # LAI forcing clipping, float32 (0 off)
    max_bm: torch.Tensor        # kg/m2 (0 off)
    flower: torch.Tensor        # long days after flowering forcing clipping (0 off)


class ClipStatic(NamedTuple):
    ev: ClipEvents              # [n_mgt, H]
    au: AutoClip                # [n_mgt, K]
    resets: torch.Tensor        # [n_crop] bool phenology clipping_resets
    newest: torch.Tensor        # [n_crop] bool leaf clip_newest
    applies_hi: torch.Tensor    # [n_crop] bool
    dormancy: torch.Tensor      # [n_crop] bool continuation mode dormancy
    exact: bool = False         # True: the reference's loss of the portions kept for the GAI to retain


def _word(table, key, i, j=None, default=None):
    t = table.params.get(key)
    if t is None:
        return default
    v = int(t[i] if j is None else t[i, j])
    if v < 0:
        return default
    return table.vocab[key][v]


def _num(table, key, i, j, default):
    t = table.params.get(key)
    if t is None:
        return default
    v = float(t[i, j])
    return default if v != v else v


def is_classic_harvest(mgts, i, j) -> bool:
    """The harvest events the existing harvest path keeps (all the shipped CRB ones): a harvest_once
    in crop_param mode without grazing or postponement that terminates the crop after a growth
    stage, or a fruit tree's harvest after a fruit growth stage. Every other harvest or clipping
    event takes the clipping path."""
    fate = _word(mgts, "harvest.biomass_fate", i, j, "harvest_once")
    mode = _word(mgts, "harvest.harvest_amount_mode", i, j, "crop_param")
    sync = _word(mgts, "harvest.event_synchronization", i, j, None)
    cont = _word(mgts, "harvest.contingency", i, j, "none")
    graze = _num(mgts, "harvest.graze", i, j, 0.0)
    term = mgts.params.get("harvest.terminate_crop")
    term = True if term is None else bool(term[i, j])
    if fate != "harvest_once" or mode != "crop_param" or cont == "postpone" or graze != 0.0:
        return False
    return (sync == "after_normal_crop_growth_stage" and term) or sync == "after_fruit_tree_growth_stage"


def _sync(table, sec, prefix, i, j, run_days):
    """(sync code, stage, offset, run day) of one event's synchronization (synchronization.cpp)."""
    from .daily import STAGE_WORDS
    from .orchard import FRUIT_STAGE_LABELS
    w = _word(table, f"{sec}.{prefix}synchronization", i, j, None)
    off = table.params.get(f"{sec}.{prefix}offset")
    off = 0 if off is None or int(off[i, j]) < 0 and int(off[i, j]) == -1 else int(off[i, j])
    if w == "after_normal_crop_growth_stage":
        return SYNC_STAGE, STAGE_WORDS[_word(table, f"{sec}.{prefix}phenologic", i, j)], off, -1
    if w == "after_fruit_tree_growth_stage":
        return SYNC_FRUIT, FRUIT_STAGE_LABELS.index(_word(table, f"{sec}.{prefix}phenologic(fruit)", i, j)), off, -1
    if w == "actual_date":
        d = table.params.get(f"{sec}.{prefix}date")
        if d is None or int(d[i, j]) < 0:
            raise ValueError(f"cropsyst: {sec} event {j + 1} of {table.paths[i]}: actual_date without a date")
        y, doy = divmod(int(d[i, j]), 1000)
        return SYNC_DATE, -1, 0, run_days.get((y, doy), -2)
    if w is None:
        return SYNC_NONE, -1, 0, -1
    from .engine import NotYetImplemented
    raise NotYetImplemented(f"cropsyst: {sec} event synchronised {w!r} ({table.paths[i]}) is not ported "
                            "(clipping events after a growth stage or on an actual date)")


def _ops(table, sec, i, j):
    """Amount mode, the float32 constraint values and the percents of a harvest-like section."""
    f32 = lambda v: float(torch.tensor(v, dtype=torch.float32))
    mode = MODE_WORDS.index(_word(table, f"{sec}.harvest_amount_mode", i, j, "crop_param"))
    v = dict(remove=f32(_num(table, f"{sec}.remove_amount", i, j, 0.0)) / 10000.0,
             reserve=f32(_num(table, f"{sec}.reserve_biomass", i, j, 0.0)) / 10000.0,
             min_lai=f32(_num(table, f"{sec}.min_LAI", i, j, 0.0)),
             min_bm=f32(_num(table, f"{sec}.min_biomass", i, j, 0.0)) / 10000.0,
             retain=f32(_num(table, f"{sec}.min_retain_GAI", i, j, 0.0)))
    acc = table.params.get(f"{sec}.accept_less")
    pct = [_num(table, f"{sec}.{k}", i, j, float(d)) for k, d in zip(PCT_KEYS, PCT_DEFAULT)]
    return mode, v, (False if acc is None else bool(acc[i, j])), pct


def build_static(rs, year, doy, device, dtype=torch.float64, clip_mode="fixed") -> Optional[ClipStatic]:
    """ClipStatic from the rotation set, or None when no management clips (the run keeps the
    existing harvest path only)."""
    mgts, crops = rs.mgts, rs.crops
    n = mgts.n
    run_days = {(int(y), int(j)): k for k, (y, j) in enumerate(zip(year, doy))}
    hv = mgts.valid.get("harvest")
    H = 0 if hv is None or hv.dim() < 2 else hv.shape[1]
    av = mgts.valid.get("auto_clip_biomass")
    K = 0 if av is None or av.dim() < 2 else av.shape[1]
    rows = []
    for i in range(n):
        r = []
        for j in range(H):
            if not bool(hv[i, j]) or is_classic_harvest(mgts, i, j):
                r.append(None)
                continue
            if "harvest.enabled" in mgts.params and not bool(mgts.params["harvest.enabled"][i, j]):
                r.append(None)
                continue
            if "harvest.checked" in mgts.params and not bool(mgts.params["harvest.checked"][i, j]):
                r.append(None)                   # cs_event.cpp:290: unchecked events are not scheduled
                continue
            s, st, off, dd = _sync(mgts, "harvest", "event_", i, j, run_days)
            mode, v, acc, pct = _ops(mgts, "harvest", i, j)
            term = mgts.params.get("harvest.terminate_crop")
            r.append(dict(sync=s, stage=st, offset=off, date=dd,
                          terminate=False if term is None else bool(term[i, j]),
                          fate=FATE_WORDS.index(_word(mgts, "harvest.biomass_fate", i, j, "harvest_once")),
                          mode=mode, accept=acc, postpone=_word(mgts, "harvest.contingency", i, j, "none") == "postpone",
                          pct=pct, **v))
        rows.append(r)
    arows = []
    for i in range(n):
        r = []
        for j in range(K):
            if not bool(av[i, j]):
                r.append(None)
                continue
            bs, bst, boff, bd = _sync(mgts, "auto_clip_biomass", "begin_", i, j, run_days)
            es, est, eoff, ed = _sync(mgts, "auto_clip_biomass", "end_", i, j, run_days)
            if es == SYNC_STAGE and est == ph.NGS_PLANTING:
                es = SYNC_NONE                     # never dated in the reference (module notes)
            mode, v, acc, pct = _ops(mgts, "auto_clip_biomass", i, j)
            f32 = lambda x: float(torch.tensor(x, dtype=torch.float32))
            fl = mgts.params.get("auto_clip_biomass.after_flowering")
            r.append(dict(b_sync=bs, b_stage=bst, b_offset=boff, b_date=bd, e_sync=es, e_stage=est,
                          e_offset=eoff, e_date=ed, mode=mode, accept=acc, pct=pct,
                          adj=f32(_num(mgts, "auto_clip_biomass.adj_rel_growth", i, j, 1.0)),
                          max_lai=f32(_num(mgts, "auto_clip_biomass.max_LAI", i, j, 0.0)),
                          max_bm=f32(_num(mgts, "auto_clip_biomass.max_biomass", i, j, 4000.0)) / 10000.0,
                          flower=0 if fl is None or int(fl[i, j]) < 0 else int(fl[i, j]), **v))
        arows.append(r)
    if not any(e is not None for r in rows for e in r) and not any(e is not None for r in arows for e in r):
        return None
    ev = _table(ClipEvents, rows, max(H, 1), dtype, device)
    au = _table(AutoClip, arows, max(K, 1), dtype, device)
    nc = crops.n
    def cword(key, i, default):
        return _word(crops, key, i, None, default)
    resets = torch.tensor([bool(crops.params["phenology.clipping_resets"][i]) if "phenology.clipping_resets"
                           in crops.params else False for i in range(nc)], device=device)
    newest = torch.tensor([bool(crops.params["leaf.clip_newest"][i]) if "leaf.clip_newest" in crops.params
                           else True for i in range(nc)], device=device)    # crop_param_class.cpp:408
    parts = ["grain", "root", "tuber", "bulb", "leaf", "fruit", "canopy", "complete", "unknown"]
    def part_index(i):
        w = cword("crop.harvested_part", i, "grain")
        return parts.index(w) if w in parts else 0          # 'seed' (V4 label): grain
    hi = torch.tensor([cword("crop.model", i, "crop") == "crop" and part_index(i) < parts.index("canopy")
                       and cword("crop.land_use", i, "ROW_CROPS") != "PASTURE" for i in range(nc)], device=device)
    dorm = torch.tensor([(cword("crop.season_dormancy_mode", i, None) or cword("crop.continuation_mode", i, "single"))
                         == "dormancy" for i in range(nc)], device=device)
    if clip_mode not in ("fixed", "exact"):
        raise ValueError("cropsyst: clip_mode must be 'fixed' or 'exact'")
    return ClipStatic(ev, au, resets, newest, hi, dorm, clip_mode == "exact")


def _table(cls, rows, W, dtype, device):
    n = len(rows)
    out = {}
    for f in cls._fields:
        if f == "on":
            t = torch.zeros((n, W), dtype=torch.bool)
        elif f == "pct":
            t = torch.zeros((n, W, 10), dtype=dtype)
        elif f in ("terminate", "accept", "postpone"):
            t = torch.zeros((n, W), dtype=torch.bool)
        elif f in ("sync", "b_sync", "e_sync"):
            t = torch.full((n, W), SYNC_NONE, dtype=torch.long)
        elif f in ("stage", "date", "b_stage", "b_date", "e_stage", "e_date"):
            t = torch.full((n, W), -1, dtype=torch.long)
        elif f in ("offset", "fate", "mode", "b_offset", "e_offset", "flower"):
            t = torch.zeros((n, W), dtype=torch.long)
        else:
            t = torch.zeros((n, W), dtype=dtype)
        for i, r in enumerate(rows):
            for j, e in enumerate(r):
                if e is None:
                    continue
                if f == "on":
                    t[i, j] = True
                elif f == "pct":
                    t[i, j] = torch.tensor(e["pct"], dtype=dtype)
                else:
                    t[i, j] = e[f]
        out[f] = t.to(device)
    return cls(**out)


def gather_events(E, mid):
    m = mid.clamp(min=0)
    return type(E)(*[x[m] for x in E])


# ----------------------------------------------------------------------------- state

class ClipState(NamedTuple):
    ev_date: torch.Tensor       # [T, H] long day the event is due (-1 not yet synchronised)
    ev_live: torch.Tensor       # [T, H] bool not yet performed or eaten
    b_date: torch.Tensor        # [T, K] period begin / end dates
    e_date: torch.Tensor
    b_live: torch.Tensor        # [T, K]
    e_live: torch.Tensor
    active: torch.Tensor        # [T] long automatic clipping period in force (-1 none)
    no_more: torch.Tensor       # [T] bool no_more_clippings_this_season
    low_days: torch.Tensor      # [T] long consecutive_low_growth_days_for_clipping
    max_rate: torch.Tensor      # [T] max_daily_GAI_rate
    flower_off: torch.Tensor    # [T] bool flowering threshold set to 0 by a clipping
    fill_off: torch.Tensor      # [T] bool filling threshold set to 0


def initial_state(CS: ClipStatic, T, dtype=torch.float64, device=None) -> ClipState:
    H = CS.ev.on.shape[1]
    K = CS.au.on.shape[1]
    L = lambda *s: torch.full(s, -1, dtype=torch.long, device=device)
    B = lambda *s: torch.zeros(s, dtype=torch.bool, device=device)
    return ClipState(L(T, H), B(T, H), L(T, K), L(T, K), B(T, K), B(T, K), L(T),
                     B(T), torch.zeros(T, dtype=torch.long, device=device),
                     torch.zeros(T, dtype=dtype, device=device), B(T), B(T))


def sow(cs: ClipState, CS: ClipStatic, mid, mask) -> ClipState:
    """A new crop and its management's events (the land unit's automatic clipping mode is
    cleared at the start of the fallow, land_unit_sim.cpp:2150)."""
    ev = gather_events(CS.ev, mid)
    au = gather_events(CS.au, mid)
    m2 = mask[:, None]
    w = lambda a, b: torch.where(m2, a, b)
    return ClipState(
        ev_date=w(torch.where(ev.sync == SYNC_DATE, ev.date, torch.full_like(cs.ev_date, -1)), cs.ev_date),
        ev_live=w(ev.on, cs.ev_live),
        b_date=w(torch.where(au.b_sync == SYNC_DATE, au.b_date, torch.full_like(cs.b_date, -1)), cs.b_date),
        e_date=w(torch.where(au.e_sync == SYNC_DATE, au.e_date, torch.full_like(cs.e_date, -1)), cs.e_date),
        b_live=w(au.on, cs.b_live), e_live=w(au.on & (au.e_sync != SYNC_NONE), cs.e_live),
        active=torch.where(mask, torch.full_like(cs.active, -1), cs.active),
        no_more=cs.no_more & ~mask, low_days=torch.where(mask, torch.zeros_like(cs.low_days), cs.low_days),
        max_rate=torch.where(mask, torch.zeros_like(cs.max_rate), cs.max_rate),
        flower_off=cs.flower_off & ~mask, fill_off=cs.fill_off & ~mask)


def end_crop(cs: ClipState, mask) -> ClipState:
    """The crop leaves the tile: its events go with it and the clipping mode is cleared
    (land_unit_sim.cpp:1996, 2116, 2150)."""
    return _rep(cs, ev_live=cs.ev_live & ~mask[:, None], b_live=cs.b_live & ~mask[:, None],
                e_live=cs.e_live & ~mask[:, None], active=torch.where(mask, torch.full_like(cs.active, -1), cs.active))


def adjust_params(PP: ph.PhenologyParams, cs: Optional[ClipState]):
    """The flowering / filling thresholds a clipping set to 0 (crop_cropsyst.cpp:2563-2569)."""
    if cs is None:
        return PP
    z = torch.zeros_like(PP.flowering)
    return PP._replace(flowering=torch.where(cs.flower_off, z, PP.flowering),
                       filling=torch.where(cs.fill_off, z, PP.filling))


def _date(sync, stage, offset, cur, sync_day, present):
    """Event.synchronize: dated once, the first time the stage is triggered."""
    sd = sync_day.gather(-1, stage.clamp(min=0))
    new = present[:, None] & (cur < 0) & ((sync == SYNC_STAGE) | (sync == SYNC_FRUIT)) & (stage >= 0) & (sd >= 0)
    return torch.where(new, sd + offset, cur)


def schedule(cs: ClipState, CS: ClipStatic, mid, phn) -> ClipState:
    ev = gather_events(CS.ev, mid)
    au = gather_events(CS.au, mid)
    return _rep(cs, ev_date=_date(ev.sync, ev.stage, ev.offset, cs.ev_date, phn.sync_day, phn.present),
                b_date=_date(au.b_sync, au.b_stage, au.b_offset, cs.b_date, phn.sync_day, phn.present),
                e_date=_date(au.e_sync, au.e_stage, au.e_offset, cs.e_date, phn.sync_day, phn.present))


# ----------------------------------------------------------------------------- canopy removal

def _walk_torch(pg, pd, pb, pgb, cnt, amount, retain, avail, newest, mask, lose=True):
    """Canopy_accumulation::remove_biomass (canopy_growth_portioned.cpp:103-160) over the list
    slots [0, cnt) of each masked tile, newest first (tail) or oldest first (head). Slots of
    portions removed or popped-and-dropped become 0 (a zero slot is a no-op in every sum).
    Returns (pg, pd, pb, pgb, removed, margin: the smallest |living left - portion living -
    retain| of the decisions taken, a diagnostic of knife-edge keep/drop decisions)."""
    T, R = pb.shape
    go = mask & ~_is_zero(amount)
    more = amount
    act = torch.zeros_like(amount)
    margin = torch.full_like(amount, float("inf"))   # closest keep/drop decision (diagnostic)
    alive = go & (amount > 0.0)                    # the loop tests more > 0 before each pop
    ar = torch.arange(R, device=pb.device)
    if pb.device.type == "cpu":                    # host checks are free on the CPU
        if not bool(alive.any()):
            return pg, pd, pb, pgb, act, margin
        steps = int(cnt[alive].max())
    else:
        steps = R
    for k in range(steps):
        idx = torch.where(newest, cnt - 1 - k, torch.full_like(cnt, k))
        valid = alive & (idx >= 0) & (idx < cnt)
        i = idx.clamp(0, R - 1)[:, None]
        b = pb.gather(1, i)[:, 0]
        L = pg.gather(1, i)[:, 0]
        keep = (avail - L) < retain
        dm = (avail - L) - retain
        dm = torch.where(dm < 0.0, -dm, dm)
        margin = torch.where(valid & (dm < margin), dm, margin)
        take = valid & ~keep
        drop = valid & keep
        whole = take & (more > b)
        part = take & ~(more > b)
        lt = _f32(L * (more / torch.where(part, b, torch.ones_like(b))))
        dec = torch.where(b > more, more, b)
        avail = torch.where(whole, avail - L, torch.where(part, avail - lt, avail))
        act = torch.where(whole, act + b, torch.where(part, act + dec, act))
        more = torch.where(whole, more - b, torch.where(part, more - dec, more))
        hit = (ar[None, :] == i) & (whole | (drop & lose))[:, None]
        ph_ = (ar[None, :] == i) & part[:, None]
        z = torch.zeros_like(pb)
        pg = torch.where(hit, z, torch.where(ph_, (L - lt)[:, None], pg))
        pb = torch.where(hit, z, torch.where(ph_, (b - dec)[:, None], pb))
        pd = torch.where(hit, z, pd)
        pgb = torch.where(hit, z, pgb)
        alive = valid & (more > 0.0)
    return pg, pd, pb, pgb, act, margin


def walk(pg, pd, pb, pgb, cnt, amount, retain, avail, newest, mask, lose=True):
    if pb.device.type == "cuda":
        from . import clip_kernel
        if clip_kernel.enabled():
            return clip_kernel.walk(pg, pd, pb, pgb, cnt, amount, retain, avail, newest, mask, lose)
    return _walk_torch(pg, pd, pb, pgb, cnt, amount, retain, avail, newest, mask, lose)


class Fate(NamedTuple):
    """Crop_mass_fate of the day's clippings / harvests on the clipping path [T]."""
    yield_: torch.Tensor
    useful: torch.Tensor
    manure: torch.Tensor         # grazing_to_manure
    gdisp: torch.Tensor          # grazing_to_disposal
    disposal: torch.Tensor
    chaff: torch.Tensor          # chaff_or_leaf_litter
    stubble: torch.Tensor        # stubble_dead
    uncut: torch.Tensor
    removed: torch.Tensor        # total_act_biomass_removed (float32 accumulation)
    margin: torch.Tensor         # closest keep/drop decision of the removals (inf: none)


def zero_fate(like):
    z = torch.zeros_like(like)
    return Fate(*([z] * (len(Fate._fields) - 1)), torch.full_like(z, float("inf")))


def _living(pg, u_gai, u_on):
    return cn._seqsum(pg) + torch.where(u_on, u_gai, torch.zeros_like(u_gai))


def biomass_fate(c: cn.CanopyState, canopy_n, conc, mask, mode, remove, retain_gai, retain_bm, pct,
                 terminate, yield_now, newest, exact=False):
    """Crop_complete::process_biomass_fate (crop_cropsyst.cpp:3210-3482) on the masked tiles, with
    the canopy removals of canopy_growth_portioned.cpp (see the module notes). canopy_n, conc: the
    crop's vital canopy N mass and concentration (nitrogen on), else None. Returns (canopy,
    canopy N, Fate, killed: the terminate mask)."""
    T, P = c.p_bm.shape
    dt = c.p_bm.dtype
    term = mask & terminate
    zc = torch.zeros((T, 1), dtype=dt, device=c.p_bm.device)
    ext = lambda x: torch.cat([x, zc], -1)
    pg, pd, pb, pgb = ext(c.p_gai), ext(c.p_dead), ext(c.p_bm), ext(c.p_gbm)
    # Canopy_growth_leaf_area_index_based::terminate: the untouchable portion to the list's tail
    ar = torch.arange(P + 1, device=pb.device)
    at = (ar[None, :] == c.n[:, None]) & term[:, None]
    pg = torch.where(at, c.u_gai[:, None], pg)
    pd = torch.where(at, c.u_dead[:, None], pd)
    pb = torch.where(at, c.u_bm[:, None], pb)
    pgb = torch.where(at, c.u_gbm[:, None], pgb)
    cnt = c.n + term.to(torch.long)
    u_on = ~term                                   # the untouchable pointer
    # canopy biomass: the cached vital total (stale when terminating) + the untouchable portion
    curr = cn._seqsum(c.p_bm) + torch.where(u_on, c.u_bm, torch.zeros_like(c.u_bm))
    unh = curr
    nn = canopy_n
    fresh = ~term                                  # the cached living total is current
    margins = []

    def removal(pg, pd, pb, pgb, amount, retain, newest_, on, fresh):
        av = torch.where(fresh, _living(pg, c.u_gai, u_on), cn._seqsum(c.p_gai))
        pg, pd, pb, pgb, act, mg = walk(pg, pd, pb, pgb, cnt, amount, retain, av, newest_, on, exact)
        eff = on & ~_is_zero(amount)               # an effective call invalidates the totals
        margins.append(mg)
        return pg, pd, pb, pgb, act, fresh | eff

    def n_out(nn, act, on):
        if nn is None:
            return None
        r = _cmin(act * conc, nn)
        return torch.where(on, nn - r, nn)

    # the yield first, newest first (crop_cropsyst.cpp:3242-3249)
    ty = mask & (yield_now > 0.0000001)
    unh = torch.where(ty, unh - yield_now, unh)
    pg, pd, pb, pgb, act1, fresh = removal(pg, pd, pb, pgb, torch.where(ty, yield_now, torch.zeros_like(unh)),
                                           retain_gai, torch.ones_like(term), ty, fresh)
    nn = n_out(nn, act1, ty)
    total = _f32(torch.where(ty, act1, torch.zeros_like(act1)))
    dead_i = torch.where(term, _cmin(pct[..., DEAD] + pct[..., LIVE], torch.full_like(pct[..., DEAD], 100.0)),
                         pct[..., DEAD])
    live_i = torch.where(term, torch.zeros_like(pct[..., LIVE]), pct[..., LIVE])
    removable = unh - retain_bm
    f = lambda k: _f32(pct[..., k])
    fixed = mode == FIXED
    # default branch (crop_param, percent, retain_GAI, other)
    useful = unh * f(USE) / 100.0
    manure = unh * f(GRAZE_P) / 100.0 * (f(MANURE) / 100.0)
    gdisp = unh * f(GRAZE_P) / 100.0 * (1.0 - f(MANURE) / 100.0)
    disposal = unh * f(DISPOSE) / 100.0
    chaff = unh * f(RESIDUE) / 100.0
    to_rm = (((((0.0 + useful) + manure) + gdisp) + disposal) + chaff)
    killed = unh * _f32(dead_i) / 100.0
    to_rm = to_rm + killed
    uncut = unh * _f32(live_i) / 100.0
    low = uncut < retain_bm
    uncut = torch.where(low, retain_bm, uncut)
    to_rm = torch.where(low, removable, to_rm)
    amount = torch.where(fixed, _cmin(remove, removable), to_rm)
    newest_c = newest
    pg, pd, pb, pgb, act2, fresh = removal(pg, pd, pb, pgb, amount, retain_gai, newest_c, mask, fresh)
    nn = n_out(nn, act2, mask)
    # fixed (and retain_GAI): the fates of what was removed (crop_cropsyst.cpp:3325-3353)
    post = fixed | (mode == RETAIN_GAI)
    # the default branch adds the removal as float32 (crop_cropsyst.cpp:3310-3314), the fixed one not
    act2 = torch.where(fixed, act2, _f32(act2))
    rg = _f32(act2 * pct[..., GRAZE_P] / 100.0)
    useful = torch.where(post, act2 * f(USE) / 100.0, useful)
    manure_p = rg * (f(MANURE) / 100.0)
    manure = torch.where(post, manure_p, manure)
    gdisp = torch.where(post, rg - manure_p, gdisp)
    disposal = torch.where(post, act2 * f(DISPOSE) / 100.0, disposal)
    chaff = torch.where(post, act2 * f(RESIDUE) / 100.0, chaff)
    stubble = torch.where(post, act2 * dead_i / 100.0, killed)
    uncut = torch.where(post, unh - act2, uncut)
    total = _f32(total + _f32(act2))
    # terminate: everything left to standing stubble (remove_all_biomass)
    big = torch.full_like(unh, ALL)
    pg, pd, pb, pgb, act3, _ = walk(pg, pd, pb, pgb, cnt, big, torch.zeros_like(unh), big, torch.ones_like(term), term)
    act3 = 0.0 + act3
    stubble = torch.where(term, stubble + act3, stubble)
    uncut = torch.where(term, torch.zeros_like(uncut), uncut)
    if nn is not None:
        nn = torch.where(term, nn - _cmin(ALL * conc, nn), nn)
    total = torch.where(term, _f32(total + _f32(act3)), total)
    # the canopy after: the list back to its slots; the LAI wrapper's self-shading LAI
    gai_after = cn._seqsum(pg) + torch.where(u_on, c.u_gai + c.u_dead, torch.zeros_like(c.u_gai))
    mp = mask[:, None]
    c2 = _rep(c, p_gai=torch.where(mp, pg[:, :P], c.p_gai), p_dead=torch.where(mp, pd[:, :P], c.p_dead),
              p_bm=torch.where(mp, pb[:, :P], c.p_bm), p_gbm=torch.where(mp, pgb[:, :P], c.p_gbm),
              lai_ss=torch.where(mask, gai_after, c.lai_ss))
    # a terminated crop's canopy is gone (the crop object leaves at the end of the day)
    empty = cn.initial_state(term.shape, P, dt, c.p_bm.device)
    c2 = cn.CanopyState(*[torch.where(term.reshape(term.shape + (1,) * (a.dim() - 1)), e, a)
                          for a, e in zip(c2, empty)])
    c2 = _rep(c2, height=c.height)
    z = torch.zeros_like(unh)
    w = lambda x: torch.where(mask, x, z)
    margin = _cmin(margins[0], margins[1])
    fate = Fate(w(torch.where(ty, yield_now, z)), w(useful), w(manure), w(gdisp), w(disposal), w(chaff),
                w(stubble), w(uncut), w(total), torch.where(mask, margin, torch.full_like(z, float("inf"))))
    return c2, nn, fate, term


def add_fate(a: Fate, b: Fate) -> Fate:
    return Fate(*[getattr(a, f) + getattr(b, f) for f in Fate._fields if f != "margin"],
                _cmin(a.margin, b.margin))


# ----------------------------------------------------------------------------- regrowth

def _mark(arr, m, code, day):
    k = torch.arange(arr.shape[-1], device=arr.device)
    return torch.where(m[..., None] & (k == code), day, arr)


def canopy_respond(c: cn.CanopyState, m, ia, initial_gai, sla, kc) -> cn.CanopyState:
    """The canopy side of respond_to_clipping on the clipped tiles m: where initiate_accrescence runs
    (ia) a new canopy at the initial GAI and biomass (provide_canopy + restart_with,
    crop_cropsyst.cpp:2296-2307, 779-832); then Canopy_growth_leaf_area_index_based::
    respond_to_clipping (canopy_growth_LAI_based.cpp:295-300): peak LAI 0, cover updated."""
    c = cn.restart(c, ia, initial_gai, sla, kc)
    return _rep(c, peak_lai=torch.where(m, torch.zeros_like(c.peak_lai), c.peak_lai),
                cover_total=torch.where(m & c.exists, cn.cover_fract(c.lai_ss, kc), c.cover_total))


def respond(phn: ph.PhenologyState, c: cn.CanopyState, PP, GP, m, resets, day):
    """Crop_complete::respond_to_clipping (crop_cropsyst.cpp:2510-2594) on the masked tiles.
    Returns (phenology, canopy, initiated: initiate_accrescence ran, flowering/filling disabled)."""
    dk = torch.as_tensor(day, dtype=torch.long, device=m.device)
    sync = _mark(phn.sync_day, m, ph.NGS_HARVEST, dk)
    not_mat = m & (phn.stage < ph.NGS_MATURITY)
    rs = not_mat & resets
    gdds = torch.where(rs, PP.emergence, phn.gdds)
    gdds_y = torch.where(rs, PP.emergence, phn.gdds_yesterday)
    stage = phn.stage
    ana = not_mat & ~resets & ((stage == ph.NGS_ANTHESIS) | (stage == ph.NGS_FILLING))
    ag = rs | ana
    stage = torch.where(ag, torch.full_like(stage, ph.NGS_ACCRESCENCE), stage)
    sync = _mark(sync, ag, ph.NGS_ACCRESCENCE, dk)
    # each period that clipping resets starts again (phenology_A.cpp:377-395)
    pon = phn.period_on
    pr = rs[:, None] & pon[:, :3]
    tt = phn.period_tt.clone()
    tt[:, :3] = torch.where(pr, torch.zeros_like(tt[:, :3]), tt[:, :3])
    ia = rs & pr.any(-1)
    # initiate_accrescence (crop_cropsyst.cpp:2286-2323): provide_canopy deletes the canopy and
    # makes a new one (:779-832), restart_with gives it the initial GAI and biomass; the biomass of
    # the old canopy leaves the books (its N stays in the crop's canopy N mass)
    stage = torch.where(ia, torch.full_like(stage, ph.NGS_ACCRESCENCE), stage)
    init = _mark(_mark(phn.init_day, ia, ph.NGS_EMERGENCE, dk), ia, ph.NGS_ACCRESCENCE, dk)
    sync = _mark(_mark(sync, ia, ph.NGS_EMERGENCE, dk), ia, ph.NGS_ACCRESCENCE, dk)
    pon = pon.clone()
    pon[:, ph.ACC] = pon[:, ph.ACC] | ia
    pon[:, ph.CULM] = pon[:, ph.CULM] & ~ia
    pon[:, ph.SEN] = pon[:, ph.SEN] & ~ia
    tt[:, ph.ACC] = torch.where(ia, torch.zeros_like(tt[:, ph.ACC]), tt[:, ph.ACC])
    rel_e = phn.period_rel_elapsed.clone()
    rel_r = phn.period_rel_remaining.clone()
    rel_e[:, ph.ACC] = torch.where(ia, torch.zeros_like(rel_e[:, ph.ACC]), rel_e[:, ph.ACC])
    rel_r[:, ph.ACC] = torch.where(ia, torch.zeros_like(rel_r[:, ph.ACC]), rel_r[:, ph.ACC])
    c = canopy_respond(c, m, ia, GP.initial_gai, GP.sla, GP.kc)
    phn = _rep(phn, gdds=gdds, gdds_yesterday=gdds_y, stage=stage, sync_day=sync, init_day=init, period_on=pon,
               period_tt=tt, period_rel_elapsed=rel_e, period_rel_remaining=rel_r)
    return phn, c, ia, ana


# ----------------------------------------------------------------------------- the day

class ClipDay(NamedTuple):
    """What the day's clipping hands back to the engine."""
    fate: Fate                  # event fates (distributed at end_day with nitrogen)
    auto_fate: Fate             # automatic clipping fates (deleted: not distributed)
    performed: torch.Tensor     # [T] bool an event clipping or harvest today
    auto: torch.Tensor          # [T] bool an automatic clipping today
    kill: torch.Tensor          # [T] bool a terminating event today
    ev_index: torch.Tensor      # [T] long the last performed event (residue operation)
    lwp_reset: torch.Tensor     # [T] bool a new transpiration object (initiate_accrescence)
    fruit_harvest: torch.Tensor  # [T] bool a fruit tree's event harvest


def _yield_hi(g, GP):
    from . import daily as dl
    _, yl, _ = dl.harvest(g, GP)
    return yl


def morning(cs: ClipState, CS: ClipStatic, cid, mid, phn, g, PP, GP, day, canopy_n=None, conc=None,
            fruit=None, tavg=None):
    """The land unit's performable operations of the day on the clipping path, between the crop's
    start_day and its process_day: the harvest / clipping events due today (in their management
    order), the start and the end (clip now or never) of an automatic clipping period. Returns
    (clip state, phenology, growth state, canopy N or None, ClipDay)."""
    from . import daily as dl
    T = cid.shape[0]
    dk = torch.as_tensor(day, dtype=torch.long, device=cid.device)
    cs = schedule(cs, CS, mid, phn)
    ev = gather_events(CS.ev, mid)
    au = gather_events(CS.au, mid)
    resets = CS.resets[cid]
    newest = CS.newest[cid]
    ahi = CS.applies_hi[cid]
    dorm = CS.dormancy[cid]
    fruit = torch.zeros_like(resets) if fruit is None else fruit
    c = g.canopy
    fate = zero_fate(c.u_bm)
    afate = zero_fate(c.u_bm)
    performed = torch.zeros_like(resets)
    kill = torch.zeros_like(resets)
    fh = torch.zeros_like(resets)
    lwp_reset = torch.zeros_like(resets)
    auto = torch.zeros_like(resets)
    ev_index = torch.full_like(cid, -1)
    nn = canopy_n
    H = ev.on.shape[1]
    ev_date, ev_live, no_more = cs.ev_date, cs.ev_live, cs.no_more
    flower_off, fill_off = cs.flower_off, cs.fill_off
    for j in range(H):                          # management event slots (static count)
        e = type(ev)(*[x[:, j] for x in ev])
        due = ev_live[:, j] & (ev_date[:, j] == dk) & phn.present & ~kill
        c = g.canopy
        bm = torch.where(c.exists, cn.biomass_vital(c), torch.zeros_like(c.u_bm))
        lai = torch.where(c.exists, cn.lai(c), torch.zeros_like(c.u_bm))
        # perform_harvest_or_clipping (perform_op.cpp:1092-1150)
        fixed = e.mode == FIXED
        all_ = torch.where(fixed, (bm - e.remove) >= e.reserve, torch.ones_like(due))
        some = torch.where(fixed, bm >= e.reserve, torch.ones_like(due))
        lim = fixed & ~all_ & some & e.accept
        accept_amt = torch.where(fixed, torch.where(lim, bm - e.reserve, e.remove), bm)
        status = torch.where(lim, torch.full_like(cid, ES_LIMITED), torch.full_like(cid, ES_FAIL))
        ok = some & ~(~all_ & ~e.accept)
        suff = (lai > e.min_lai) & (bm >= e.min_bm)
        go = due & ok & (suff | e.terminate)
        status = torch.where(due & ok & ~(suff | e.terminate) & no_more, torch.full_like(status, ES_OUT_OF_SEASON), status)
        # fruit trees: the fruit harvest (no biomass fate)
        fgo = go & fruit
        fh = fh | fgo
        bgo = go & ~fruit
        # Crop_complete::harvest
        cause = torch.where(e.terminate, torch.full_like(cid, LAST_CLIPPING), torch.full_like(cid, CLIPPING))
        cause = torch.where(e.pct[:, GRAZE_P] > 0, torch.full_like(cid, GRAZE), cause)
        hi_route = (ahi & (cause != GRAZE)) | (e.mode == CROP_PARAM)
        retain_gai = torch.where(fixed, e.retain, torch.zeros_like(e.retain))
        yl = torch.where(hi_route & (e.mode == CROP_PARAM), _yield_hi(g, GP), torch.zeros_like(bm))
        retain_bm = torch.where(hi_route, torch.zeros_like(e.reserve), e.reserve)
        c2, nn, fj, killed = biomass_fate(c, nn, conc, bgo, e.mode, accept_amt, retain_gai, retain_bm, e.pct,
                                          e.terminate, yl, newest, CS.exact)
        fate = add_fate(fate, fj)
        # harvest_crop: phenology.activate_harvest, trigger HARVEST (harvest() sets the stage sync below)
        g = dl._rep(g, canopy=c2)
        resp = bgo & ~e.terminate
        phn, c3, ia, ana = respond(phn, g.canopy, PP, GP, resp, resets, dk)
        phn = _rep(phn, sync_day=_mark(phn.sync_day, bgo | fgo, ph.NGS_HARVEST, dk))
        g = dl._rep(g, canopy=c3)
        flower_off = flower_off | ana
        fill_off = fill_off | ana
        lwp_reset = lwp_reset | ia
        kill = kill | killed                    # a fruit tree's harvest never terminates it
        performed = performed | go
        ev_index = torch.where(go, torch.full_like(ev_index, j), ev_index)
        # last clipping: no more clippings, end_season_if_no_dormancy (QUIESCENCE sync)
        last = due & (e.fate == LAST_CLIPPING)
        no_more = no_more | last
        phn = _rep(phn, sync_day=_mark(phn.sync_day, last & ~dorm, ph.NGS_QUIESCENCE, dk))
        status = torch.where(go, torch.full_like(status, ES_COMPLETE), status)
        retry = due & (status == ES_FAIL) & e.postpone
        ev_date = ev_date.clone()
        ev_date[:, j] = torch.where(retry, ev_date[:, j] + 1, ev_date[:, j])
        ev_live = ev_live.clone()
        ev_live[:, j] = ev_live[:, j] & ~(due & ~retry)
    cs = _rep(cs, ev_date=ev_date, ev_live=ev_live, no_more=no_more, flower_off=flower_off, fill_off=fill_off)
    # automatic clipping periods: start, then end with a clip now or never (perform_op.cpp:290-303)
    K = au.on.shape[1]
    active = cs.active
    b_live, e_live = cs.b_live, cs.e_live
    for k in range(K):
        bd = b_live[:, k] & (cs.b_date[:, k] == dk) & phn.present
        active = torch.where(bd, torch.full_like(active, k), active)
        b_live = b_live.clone()
        b_live[:, k] = b_live[:, k] & ~bd
        # Period::update_end_period (event.cpp:603-620): the day AFTER the end date, not on a day the
        # period begins (CS_event_scheduler.cpp:276-300: an initiated event is not ended)
        ed = e_live[:, k] & (cs.e_date[:, k] >= 0) & (dk > cs.e_date[:, k]) & ~bd & phn.present
        b_live[:, k] = b_live[:, k] & ~ed
        cs = _rep(cs, active=active)
        cs, phn, g, nn, ok, af, ia = check(cs, CS, cid, mid, phn, g, PP, GP, dk, ed & (active >= 0), True, nn, conc,
                                           tavg)
        afate = add_fate(afate, af)
        auto = auto | ok
        lwp_reset = lwp_reset | ia
        active = torch.where(ed, torch.full_like(active, -1), cs.active)
        e_live = e_live.clone()
        e_live[:, k] = e_live[:, k] & ~ed
    cs = _rep(cs, active=active, b_live=b_live, e_live=e_live)
    cs = end_crop(cs, kill)
    return cs, phn, g, nn, ClipDay(fate, afate, performed, auto, kill, ev_index, lwp_reset, fh)


def check(cs: ClipState, CS: ClipStatic, cid, mid, phn, g, PP, GP, day, mask, now_or_never, canopy_n=None,
          conc=None, tavg=None):
    """process_any_automatic_clipping + Crop_complete::check_for_automatic_clipping
    (perform_op.cpp:1703-1757, crop_cropsyst.cpp:3658-3815) with the period in force on the masked
    tiles. Returns (state, phenology, growth, canopy N, clipped, Fate, initiated)."""
    from . import daily as dl
    au = gather_events(CS.au, mid)
    k = cs.active.clamp(min=0)[:, None]
    a = type(au)(*[(x.gather(1, k)[:, 0] if x.dim() == 2 else x.gather(1, k[:, :, None].expand(-1, 1, x.shape[-1]))[:, 0])
                   for x in au])
    c = g.canopy
    bm = torch.where(c.exists, cn.biomass_vital(c), torch.zeros_like(c.u_bm))
    gai = torch.where(c.exists, cn.gai_vital(c), torch.zeros_like(bm))
    lai = torch.where(c.exists, c.lai_ss, torch.zeros_like(bm))
    fixed = a.mode == FIXED
    all_ = (bm - a.remove) >= a.reserve
    some = bm >= a.reserve
    amt = torch.where(fixed & ~all_ & some & a.accept, bm - a.reserve, a.remove)
    pc = a.pct
    m = mask & ((pc[:, USE] + pc[:, DISPOSE] + pc[:, RESIDUE]) > 0)
    max_rate = torch.where(m, _cmax(cs.max_rate, gai), cs.max_rate)
    base = phn.temps[..., 0]
    tavg = torch.zeros_like(bm) if tavg is None else tavg
    low = (gai <= LOW_GAI_RATE * a.adj) & (gai < max_rate) & (tavg > base)
    low_days = torch.where(m, cs.low_days + low.to(torch.long), cs.low_days)
    cause = torch.full_like(cid, NO_CAUSE)
    s_bm = bm >= a.min_bm
    s_lai = lai >= a.min_lai
    if now_or_never:
        cause = torch.where(s_bm, torch.full_like(cause, CLIP_NOW_OR_NEVER), cause)
    lg = low_days > 14
    cause = torch.where(lg, torch.where(s_bm & s_lai, torch.full_like(cause, LOW_GROWTH),
                                        torch.full_like(cause, INSUFFICIENT)), cause)
    cont = m & s_bm & s_lai
    cause = torch.where((a.max_lai > 0.0000001) & (lai > a.max_lai), torch.full_like(cause, MAX_LAI), cause)
    cause = torch.where((a.max_bm > 0.0000001) & (bm > a.max_bm), torch.full_like(cause, MAX_BIOMASS), cause)
    sd = phn.sync_day[..., ph.NGS_ANTHESIS]
    daf = torch.where(sd >= 0, day - sd, torch.full_like(sd, NEVER))
    cause = torch.where((a.flower != 0) & (daf >= a.flower), torch.full_like(cause, AFTER_FLOWERING), cause)
    clip = cont & (cause != NO_CAUSE) & (cause != INSUFFICIENT)
    c2, nn, fate, _ = biomass_fate(c, canopy_n, conc, clip, a.mode, amt, a.retain, a.reserve, pc,
                                   torch.zeros_like(clip), torch.zeros_like(bm), CS.newest[cid], CS.exact)
    g = dl._rep(g, canopy=c2)
    phn, c3, ia, ana = respond(phn, g.canopy, PP, GP, clip, CS.resets[cid], day)
    g = dl._rep(g, canopy=c3)
    low_days = torch.where(clip, torch.zeros_like(low_days), low_days)
    max_rate = torch.where(clip, torch.zeros_like(max_rate), max_rate)
    cs = _rep(cs, low_days=low_days, max_rate=max_rate, flower_off=cs.flower_off | ana, fill_off=cs.fill_off | ana)
    return cs, phn, g, nn, clip, fate, ia


def evening(cs: ClipState, CS: ClipStatic, cid, mid, phn, g, PP, GP, day, canopy_n=None, conc=None, tavg=None,
            mask=None):
    """The automatic clipping after the crop's process_day (land_unit_sim.cpp:1833)."""
    dk = torch.as_tensor(day, dtype=torch.long, device=cid.device)
    on = (cs.active >= 0) & phn.present
    if mask is not None:
        on = on & mask
    cs, phn, g, nn, clip, fate, ia = check(cs, CS, cid, mid, phn, g, PP, GP, dk, on, False, canopy_n, conc, tavg)
    return cs, phn, g, nn, clip, fate, ia


class ClipDayOutput(NamedTuple):
    """The day's clipping in the day record [T]."""
    harvested: torch.Tensor     # yield + useful of the day's fates (Yield_kg_m2 of the crop csv)
    yield_: torch.Tensor
    useful: torch.Tensor
    chaff: torch.Tensor
    stubble: torch.Tensor
    removed: torch.Tensor       # canopy biomass removed (float32 accumulation)
    event: torch.Tensor         # bool a harvest / clipping event performed
    auto: torch.Tensor          # bool an automatic clipping
    killed: torch.Tensor        # bool the event terminated the crop
    margin: torch.Tensor        # closest keep/drop decision of the day's removals (inf: none)


def day_output(cday: ClipDay, afate: Fate, auto) -> ClipDayOutput:
    f = add_fate(cday.fate, add_fate(cday.auto_fate, afate))
    return ClipDayOutput(f.yield_ + f.useful, f.yield_, f.useful, f.chaff, f.stubble, f.removed,
                         cday.performed, cday.auto | auto, cday.kill, f.margin)


def _where_tree(mask, a, b):
    if a is None:
        return None
    if isinstance(a, tuple):
        return type(a)(*[_where_tree(mask, x, y) for x, y in zip(a, b)])
    return torch.where(mask.reshape(mask.shape + (1,) * (a.dim() - mask.dim())), a, b)
