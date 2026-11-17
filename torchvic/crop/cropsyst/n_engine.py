"""Nitrogen in the CropSyst engine: the day of a crop tile with crop and soil nitrogen,
the management events that act on soil N and residue, and their setup from the rotation's
management files. engine.day_step hands the day to day_step_n when the run was built with
nitrogen=True; the nitrogen-off path is not touched.

The day (CS/land_unit_sim.cpp:1281-2038, the order of tests/cropsyst_n_l2.py, which matches the
original reference over whole CRB seasons):

  1. sowing (engine), new crop N (nitrogen.sow), real wilting point of the crop's wilt leaf water
     potential (Soil_base::reinitialize_for_crop, soil_base.cpp:194-220)
  2. daily.potential (crop start_day and the first half of process_day)
  3. organic-matter start_day
  4. the day's management events, at most SLOTS per tile (a tile with more sets n_overflow):
     yesterday's harvest residue operation, the scenario events (before planting, relative to the
     simulation begin, actual date) and the crop's events (after a growth stage); each event is an
     NRCS field operation response and/or inorganic N, soil N sampling or a split automatic N
     application (perform_op.cpp:104-269, 1282-1355, 1491-1700)
  5. plant available water, abiotic environment, organic-matter process_day
  6. crop N update (nitrogen.crop_potential), the N-limited potential transpiration, root
     uptake (or the replayed water stress), actual growth with the N-limited growth and the
     N leaf stress; skipped on the harvest day (the harvest operation terminates the crop before
     its process_day)
  7. hourly transformations, residue balance, soil disturbance
  8. harvest: yield, biomass fate to residue (chaff and stubble flat/standing, dead roots), real
     wilting point at -1500 J/kg for the fallow, the harvest's NRCS residue operation for the next
     day when it flattens residue (perform_op.cpp:1176-1199)

Event rules taken from the reference (cs_event.cpp, cs_mgmt.cpp, synchronization.cpp:274-310) and
checked against the operations the reference performed (dump Q records, tests/test_cropsyst_n_engine):
events with checked=false are not scheduled (cs_event.cpp:290); before-planting events are
counted back from the scheduled sowing date (the first attempt of a computed sowing); crop events
(after a growth stage) exist only while the crop stands, so events after the 'harvest' stage of a
crop that the harvest terminates and the relative-to-sowing-date events never happen in the CRB
runs; a soil N sampling sets the automatic N mode and total that the split applications use.

Not ported (refused at setup with NotYetImplemented): a sowing with an NRCS operation, automatic N
without split applications, events synchronised in other modes. Dust mulch (finite-difference
infiltration only) and supplemental automatic N (read, never used) have no effect in the reference. A tillage with terminate_crop on a standing crop terminates it without a biomass fate (no
CRB case exercises this).
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import torch

from ...core import xscan
from . import canopy as cn
from . import daily as dl
from . import n_management as nm
from . import nitrogen as nt
from . import organic_matter as om
from . import om_kernels as omk
from . import phenology as ph
from . import soil_n as sn
from . import uptake as up
from .readers import InputError

N_NONE, N_INORGANIC, N_AUTO_APPL, N_SAMPLING = 0, 2, 3, 4
NEVER = -2                       # stage code of a crop event that never fires (see module notes)
HARVEST_STAGE = ph.STAGE_LABELS.index("harvest")
SLOTS = 2
INJECTION_DEPTH_DEFAULT_CM = 10.0
WILT_FALLOW = -1500.0            # land_unit_sim.cpp:2147-2160


class NotPorted(InputError, NotImplementedError):
    pass


class Events(NamedTuple):
    """Management events [..., K] (per management row, or gathered per tile)."""
    stage: torch.Tensor          # long: growth stage of a crop event, -1 scenario event, NEVER
    offset: torch.Tensor         # long days after the stage
    reps: torch.Tensor           # long: NRCS field operation responses (0, 1, 2 for Seedbed finisher)
    sdr: torch.Tensor            # [..., K, 6] depth, mixing, rating, effect, flatten, burial
    has_ox: torch.Tensor         # bool, tillage oxidation effect
    ox: torch.Tensor             # [..., K, 2] float32 sand, clay oxidation
    terminate: torch.Tensor      # bool
    nkind: torch.Tensor          # long N_*
    no3: torch.Tensor            # inorganic: kg/ha; split application: % of the total
    nh4: torch.Tensor
    fract: torch.Tensor          # inorganic: NH4 fraction to the soil (not volatilized)
    inj: torch.Tensor            # inorganic: injection depth cm
    pct: torch.Tensor            # split application: % of the total
    mode: torch.Tensor           # sampling: nm.AUTO_* mode
    depth: torch.Tensor          # sampling depth m
    target: torch.Tensor         # kg/ha
    est: torch.Tensor
    eff: torch.Tensor            # %
    crit: torch.Tensor           # kg/ha


class NStatic(NamedTuple):
    NP: nt.NParams               # [n_crop]
    RP: om.ResidueParams         # [n_crop]
    layers: sn.SoilLayers        # [T, L]
    sand0: torch.Tensor          # [T, L] %
    clay0: torch.Tensor
    crop_ev: Events              # [n_mgt, K]
    scen_ev: Events              # [n_mgt, Ks]
    harv_ev: Events              # [n_mgt, H] the harvest's next-day residue operation
    harv_fate: torch.Tensor      # [n_mgt, H, 7] use, graze, return_as_manure, dispose, residue, dead, live %
    day_mgt: torch.Tensor        # [n_day, T, Kd] scenario events of the day: management row
    day_k: torch.Tensor          # [n_day, T, Kd] and event index (-1 none)
    S: int                       # residue list capacities
    B: int
    max_reps: int                # 2 when a "Seedbed finisher" event exists (two responses)
    auto_plant: torch.Tensor = None   # [n_mgt] bool, [management] automatic_nitrogen (to the plant)


class NEState(NamedTuple):
    soil: nm.SoilState
    crop: nt.NState
    real_wilt: torch.Tensor      # [T, L] volumetric
    auto_total: torch.Tensor     # [T] kg/m2 of the last soil N sampling
    auto_mode: torch.Tensor      # [T] long
    harvest_day: torch.Tensor    # [T] long, -1
    pend_mgt: torch.Tensor       # [T] long: tomorrow's harvest residue operation (-1 none)
    pend_h: torch.Tensor
    overflow: torch.Tensor       # [T] bool: more events in a day than SLOTS, or a residue list full


class NDayOutput(NamedTuple):
    n_stress: torch.Tensor       # [T] leaf N stress factor (1 none)
    canopy_n: torch.Tensor       # [T] kg/m2
    no3_n: torch.Tensor          # [T, L] kg/m2 elemental
    nh4_n: torch.Tensor          # [T, L] (NaN where residue N mineralization reached it, as the reference)
    overflow: torch.Tensor       # [T] bool


# ----------------------------------------------------------------------------- setup (host)

def _f32(x):
    return float(np.float32(x))


def _events_from_rows(rows, n_mgt, dtype, device):
    """rows[i] = list of dicts (one per event) -> Events [n_mgt, K]."""
    K = max([len(r) for r in rows] + [1])
    def col(key, fill, dt):
        t = torch.full((n_mgt, K), fill, dtype=dt)
        for i, r in enumerate(rows):
            for j, e in enumerate(r):
                t[i, j] = e.get(key, fill)
        return t.to(device)
    sdr = torch.zeros((n_mgt, K, 6), dtype=dtype)
    ox = torch.zeros((n_mgt, K, 2), dtype=dtype)
    for i, r in enumerate(rows):
        for j, e in enumerate(r):
            if "sdr" in e:
                sdr[i, j] = torch.tensor(e["sdr"], dtype=dtype)
            if "ox" in e:
                ox[i, j] = torch.tensor(e["ox"], dtype=dtype)
    L, F, B = torch.long, dtype, torch.bool
    return Events(col("stage", -1, L), col("offset", 0, L), col("reps", 0, L), sdr.to(device),
                  col("has_ox", False, B), ox.to(device), col("terminate", False, B), col("nkind", 0, L),
                  col("no3", 0.0, F), col("nh4", 0.0, F), col("fract", 1.0, F), col("inj", 0.0, F),
                  col("pct", 0.0, F), col("mode", 0, L), col("depth", 0.0, F), col("target", 0.0, F),
                  col("est", 0.0, F), col("eff", 100.0, F), col("crit", 0.0, F))


def _sdr_fields(desc, sdr):
    op = sdr[desc]
    return dict(reps=2 if desc == "Seedbed finisher" else 1,
                sdr=[op["depth_m"], op["mixing"], float(op["rating"]), float(op["effect"]), op["flatten"][0],
                     op["burial"][0]])


def _management_events(mgts, run_days):
    """Host parse of the management table: (crop events, scenario events, harvest residue
    operations, harvest fates) as lists per management row. run_days: {(year, doy): run day}."""
    P, V, X = mgts.params, mgts.vocab, mgts.text
    sdr = sn.sdr_operations()
    n = mgts.n

    def val(name, i, j=None, default=None):
        t = P.get(name)
        if t is None:
            return default
        v = t[i] if j is None else t[i, j]
        return v.item()

    def word(name, i, j=None):
        code = val(name, i, j)
        if code is None or code < 0:
            return None
        return V[name][int(code)]

    def text(name, i, j):
        t = X.get(name)
        if t is None or t[i] is None or j >= len(t[i]):
            return ""
        return (t[i][j] or "").strip()

    crop, scen, harv, fate = [], [], [], []
    for i in range(n):
        c_rows, s_rows, h_rows, f_rows = [], [], [], []
        auto_mode = word("automatic_nitrogen.automatic_N_mode", i) or "disabled"
        mode = {"disabled": nm.AUTO_DISABLED, "balance_based": nm.AUTO_BALANCE,
                "local_calibration": nm.AUTO_LOCAL}.get(auto_mode)
        if mode is None:
            raise NotPorted(f"cropsyst nitrogen: automatic N mode {auto_mode!r} is not ported")
        # automatic_nitrogen supplemental.* is read but used nowhere in the model (perform_op.cpp:1586)
        n_split = int(mgts.valid["auto_N_application"][i].sum()) if "auto_N_application" in mgts.valid else 0
        if mode != nm.AUTO_DISABLED and n_split == 0:
            raise NotPorted("cropsyst nitrogen: automatic N without split applications is not ported")
        for sec in ("tillage", "residue", "fertilization", "auto_N_application"):
            if sec not in mgts.valid:
                continue
            valid = mgts.valid[sec][i]
            for j in range(valid.shape[0]):
                if not bool(valid[j]):
                    continue
                if val(f"{sec}.enabled", i, j, True) is False or val(f"{sec}.checked", i, j, True) is False:
                    continue                                    # cs_event.cpp:290
                e = {}
                desc = text(f"{sec}.NRCS_field_op", i, j)
                if sec != "auto_N_application" and desc in sdr:
                    e.update(_sdr_fields(desc, sdr))
                if sec == "residue" and desc in sdr and sdr[desc]["number"] in (15, 16, 17):
                    raise NotPorted("cropsyst nitrogen: residue burning operations are not ported")
                if sec in ("tillage", "residue"):
                    # a residue operation is a tillage (perform_op.cpp:936-970, Residue_operation); dust mulch applies only with finite-difference infiltration (perform_op.cpp:1048-1058);
                    # the VIC build runs the cascade, so it never acts
                    s_ox = val(f"{sec}.sand_oxidation", i, j)
                    c_ox = val(f"{sec}.clay_oxidation", i, j)
                    s_ox = nm.OXIDATION_SAND_DEFAULT if s_ox is None or s_ox != s_ox else s_ox
                    c_ox = nm.OXIDATION_CLAY_DEFAULT if c_ox is None or c_ox != c_ox else c_ox
                    e.update(has_ox=True, ox=[_f32(s_ox), _f32(c_ox)])
                elif sec == "fertilization":
                    if word("fertilization.operation_type", i, j) not in (None, "INORGANIC_NITROGEN"):
                        raise NotPorted("cropsyst nitrogen: fertilization other than inorganic N is not ported")
                    src = word("fertilization.ammonium_source", i, j) or "urea"
                    meth = word("fertilization.NH4_appl_method", i, j) or "surface_broadcast"
                    calc = bool(val("fertilization.NH4_volatilization_calculated", i, j, False))
                    vol = _f32(val("fertilization.NH4_volatilization", i, j, 0.0))
                    fr = sn.fract_inorganic_nh4_to_soil(sn.AMMONIUM_SOURCES.index(src),
                                                        sn.NH4_APPL_METHODS.index(meth), calc, vol, 20.0, 7.0)
                    e.update(nkind=N_INORGANIC, no3=_f32(val("fertilization.NO3_N", i, j, 0.0)),
                             nh4=_f32(val("fertilization.NH4_N", i, j, 0.0)), fract=fr,
                             inj=INJECTION_DEPTH_DEFAULT_CM)
                else:
                    e.update(nkind=N_AUTO_APPL, pct=float(val("auto_N_application.percent", i, j, 100)),
                             no3=_f32(val("auto_N_application.NO3_N", i, j, 0.0)),
                             nh4=_f32(val("auto_N_application.NH4_N", i, j, 0.0)))
                # only a tillage terminates the crop (perform_op.cpp:990-1000); the flag of the
                # other operations is not read (308200's split N applications carry it)
                e["terminate"] = sec == "tillage" and bool(val(f"{sec}.terminate_crop", i, j, False))
                if "reps" not in e and e.get("nkind", 0) == 0:
                    continue                                    # nothing that touches soil N or residue
                sync = word(f"{sec}.event_synchronization", i, j)
                off = int(val(f"{sec}.event_offset", i, j, 0) or 0)
                if sync == "after_normal_crop_growth_stage":
                    e.update(stage=dl.STAGE_WORDS[word(f"{sec}.event_phenologic", i, j)], offset=off)
                    c_rows.append(e)
                elif sync == "relative_to_sow_date":
                    e.update(stage=NEVER)                       # see module notes
                    c_rows.append(e)
                elif sync == "before_planting":
                    e.update(sync="before", offset=off)
                    s_rows.append(e)
                elif sync == "actual_date":
                    e.update(sync="date", date=int(val(f"{sec}.event_date", i, j)))
                    s_rows.append(e)
                else:
                    raise NotPorted(f"cropsyst nitrogen: {sec} event synchronised {sync!r} is not ported")
        # soil N sampling (one per management; starts the automatic N mode)
        if "soil_N_sampling" in mgts.valid and bool(mgts.valid["soil_N_sampling"][i]):
            sync = word("soil_N_sampling.synchronization", i)
            e = dict(nkind=N_SAMPLING, mode=mode, depth=_f32(val("soil_N_sampling.sampling_depth", i, default=0.0)),
                     target=_f32(val("automatic_nitrogen.target_yield_N_uptake", i, default=0.0)),
                     est=_f32(val("automatic_nitrogen.estimated_mineralization", i, default=0.0)),
                     eff=_f32(val("automatic_nitrogen.fertilizer_use_efficiency", i, default=100.0)),
                     crit=_f32(val("automatic_nitrogen.critical_soil_N_for_no_response", i, default=0.0)))
            off = int(val("soil_N_sampling.offset", i, default=0) or 0)
            if sync == "before_planting":
                e.update(sync="before", offset=off)
            elif sync == "relative_to_simulation_event":
                ev = word("soil_N_sampling.relative_to_simulation_event", i)
                if ev != "begin":
                    raise NotPorted(f"cropsyst nitrogen: soil N sampling relative to {ev!r} is not ported")
                e.update(sync="begin", offset=off)
            else:
                raise NotPorted(f"cropsyst nitrogen: soil N sampling synchronised {sync!r} is not ported")
            s_rows.append(e)
        # harvests: next-day residue operation and biomass fate
        if "harvest" in mgts.valid:
            for j in range(mgts.valid["harvest"].shape[1]):
                desc = text("harvest.NRCS_field_op", i, j)
                e = {}
                if desc in sdr and sdr[desc]["flatten"][0] > 0.000001:
                    e.update(_sdr_fields(desc, sdr), has_ox=True,
                             ox=[_f32(nm.OXIDATION_SAND_DEFAULT), _f32(nm.OXIDATION_CLAY_DEFAULT)])
                h_rows.append(e)
                pc = lambda k: float(val(f"harvest.{k}", i, j, 0) or 0)
                mode_w = word("harvest.harvest_amount_mode", i, j)
                from .clipping import is_classic_harvest
                if mode_w not in (None, "crop_param") and bool(mgts.valid["harvest"][i, j]) \
                        and is_classic_harvest(mgts, i, j):
                    raise NotPorted(f"cropsyst nitrogen: harvest amount mode {mode_w!r} is not ported")
                f_rows.append([pc("use"), pc("graze"), pc("return_as_manure"), pc("dispose"), pc("residue"),
                               pc("dead"), pc("live")])
        crop.append(c_rows)
        scen.append(s_rows)
        harv.append(h_rows)
        fate.append(f_rows)
    return crop, scen, harv, fate


def _day_tables(scen, sow_mgt, year, doy, run_days, n_day, T):
    """[n_day, T, Kd] (management row, event index) of the scenario events from the scheduled
    sowings (sow_mgt [n_day, T], -1 none)."""
    lists = {}
    sm = sow_mgt.cpu().numpy()
    for t in range(T):
        for d in np.nonzero(sm[:, t] >= 0)[0]:
            m = int(sm[d, t])
            for k, e in enumerate(scen[m]):
                if e["sync"] == "before":
                    dd = int(d) - e["offset"]
                elif e["sync"] == "begin":
                    dd = e["offset"]
                else:
                    y, j = divmod(e["date"], 1000)
                    dd = run_days.get((y, j), -1)
                if 0 <= dd < n_day:
                    lst = lists.setdefault((dd, t), [])
                    if e.get("nkind") == N_SAMPLING:
                        # every scheduled sowing transfers its sampling (cs_event.cpp:170-184): on one
                        # day the samplings see the same soil and the last sets the mode and total
                        lst[:] = [(m2, k2) for m2, k2 in lst if scen[m2][k2].get("nkind") != N_SAMPLING]
                    lst.append((m, k))
    Kd = max([len(v) for v in lists.values()] + [1])
    mg = torch.full((n_day, T, Kd), -1, dtype=torch.long)
    kk = torch.full((n_day, T, Kd), -1, dtype=torch.long)
    for (dd, t), v in lists.items():
        for q, (m, k) in enumerate(v):
            mg[dd, t, q], kk[dd, t, q] = m, k
    return mg, kk


def build_n_static(rs, soil: dict, thick, max_moist, sow_mgt, year, doy, device, dtype=torch.float64,
                   S=96, B=192) -> NStatic:
    """NStatic from the rotation set, the per-tile soil ([T, L]) and the sowing day table.
    soil: quartz and clay fractions; bulk_density kg/m3 (or bulk_density_g_cm3); wcr and wpwp mm
    (or field_capacity and wilting_point, volumetric); optional saturation (volumetric, default
    max_moist / thickness). The volumetric forms take the values a VIC soil hands CropSyst as they
    are (VIC_soil.cpp:346-374)."""
    need = [("quartz",), ("clay",), ("bulk_density", "bulk_density_g_cm3"), ("wcr", "field_capacity"),
            ("wpwp", "wilting_point")]
    for alts in need:
        if not any(k in soil for k in alts):
            raise InputError(f"cropsyst: nitrogen=True needs soil {' or '.join(alts)} (per layer)")
    t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    if "sowing.NRCS_field_op" in rs.rot.text and any(
            (x or "").strip() for r in rs.rot.text["sowing.NRCS_field_op"] for x in (r or [])):
        raise NotPorted("cropsyst nitrogen: a sowing with an NRCS field operation is not ported")
    lay = sn.soil_layers(t(thick), t(soil["bulk_density"]) if "bulk_density" in soil else t(thick),
                         t(max_moist), t(soil["wcr"]) if "wcr" in soil else t(thick),
                         t(soil["wpwp"]) if "wpwp" in soil else t(thick))
    lay = lay._replace(**{f: t(soil[k]) for f, k in (("bulk_density", "bulk_density_g_cm3"), ("fc", "field_capacity"),
                                                    ("pwp", "wilting_point"), ("sat", "saturation")) if k in soil})
    year = [int(y) for y in year]
    doy = [int(x) for x in doy]
    run_days = {(y, j): i for i, (y, j) in enumerate(zip(year, doy))}
    crop, scen, harv, fate = _management_events(rs.mgts, run_days)
    n_mgt = rs.mgts.n
    H = max([len(f) for f in fate] + [1])
    hf = torch.zeros((n_mgt, H, 7), dtype=dtype)
    for i, f in enumerate(fate):
        for j, v in enumerate(f):
            hf[i, j] = torch.tensor(v, dtype=dtype)
    T = int(thick.shape[0])
    mg, kk = _day_tables(scen, sow_mgt, year, doy, run_days, len(year), T)
    return NStatic(nt.n_params_from_table(rs.crops, device=device, dtype=dtype),
                   om.residue_params_from_table(rs.crops, device=device, dtype=dtype), lay,
                   t(soil["quartz"]) * 100.0, t(soil["clay"]) * 100.0,
                   _events_from_rows(crop, n_mgt, dtype, device), _events_from_rows(scen, n_mgt, dtype, device),
                   _events_from_rows(harv, n_mgt, dtype, device), hf.to(device), mg.to(device), kk.to(device), S, B,
                   max([e.get("reps", 0) for rows in (crop, scen, harv) for r in rows for e in r] + [1]),
                   _auto_plant(rs.mgts, device))


def _auto_plant(mgts, device):
    """[management] automatic_nitrogen of each management row (perform_op.cpp:526-528)."""
    t = mgts.params.get("management.automatic_nitrogen")
    if t is None:
        return torch.zeros(mgts.n, dtype=torch.bool, device=device)
    return t.to(device).bool()


def initial_state(N: NStatic, T, dtype=torch.float64, device=None) -> NEState:
    L = N.layers.thickness.shape[-1]
    dev = device if device is not None else N.layers.thickness.device
    z = torch.zeros(T, dtype=dtype, device=dev)
    neg = torch.full((T,), -1, dtype=torch.long, device=dev)
    soil = nm.SoilState(sn.initial_state((T,), L, dtype, dev), om.initial_state((T,), N.S, N.B, dtype, dev),
                        N.sand0.clone(), N.clay0.clone(), nm.initial_disturbance((T,), L, dtype, dev))
    return NEState(soil, nt.initial_state((T,), dtype, dev), torch.zeros((T, L), dtype=dtype, device=dev), z,
                   torch.zeros(T, dtype=torch.long, device=dev), neg, neg.clone(), neg.clone(),
                   torch.zeros(T, dtype=torch.bool, device=dev))


# ----------------------------------------------------------------------------- harvest biomass fate

def _remove_newest_first(bm, amount):
    """Canopy_accumulation::remove_biomass (canopy_growth_portioned.cpp:103-160) with no GAI to
    retain: portions [..., M] in list order (the head first), taken from the tail while the amount
    left exceeds a whole portion, the last one in part. Returns the portions left."""
    tail_first = torch.flip(bm, [-1])
    before = xscan.cumsum(tail_first) - tail_first             # taken before this portion
    left = amount[..., None] - before
    whole = left > tail_first
    part = (~whole) & (left > 0.0)
    rem = torch.where(whole, torch.zeros_like(bm), torch.where(part, tail_first - left, tail_first))
    return torch.flip(rem, [-1])


def harvest_fate(c, yl, pct, terminate, mask=None):
    """Crop_complete::process_biomass_fate for a harvest with harvest_amount_mode crop_param
    (crop_cropsyst.cpp:3208-3400): pct [..., 7] the management's use, graze, return_as_manure,
    dispose, residue, dead, live percents. Returns (chaff, stubble), as the reference computes
    them (they turn negative when the yield exceeds the vital portions, and are printed so):
    - a terminating harvest first appends the untouchable restart portion to the TAIL of the vital
      list (Canopy_growth_leaf_area_index_based::terminate, canopy_growth_LAI_based.cpp:425-432)
      without invalidating the list's cached totals, so the unharvested biomass is the vital
      portions' sum without it, and the living area the removal starts from is the portions'
      living GAI without it;
    - the yield, then the fates, are removed newest portion first (canopy_growth_portioned.cpp:
      103-160): the untouchable portion comes first; if its living GAI exceeds the cached living
      total ((living - its living) < 0 retained GAI) it is popped and neither removed nor put
      back, so its biomass leaves the books;
    - a terminating harvest then moves everything left to the standing stubble (remove_all_biomass,
      crop_cropsyst.cpp:3378-3393).
    mask: the tiles whose stubble is used (the harvest); given, the portion removal runs as the
    per-tile kernel of fate_kernel (CUDA) and the stubble of the other tiles is not computed."""
    from . import canopy as cn
    f32 = lambda x: x.to(torch.float32).to(x.dtype)
    X = cn._seqsum(c.p_bm)
    take_yield = yl > 0.0000001
    unh = torch.where(take_yield, X - yl, X)
    use, graze, ret, disp, resid = (pct[..., i] for i in range(5))
    dead_p = torch.where(terminate, torch.clamp(pct[..., 5] + pct[..., 6], max=100.0), pct[..., 5])
    live_p = torch.where(terminate, torch.zeros_like(pct[..., 6]), pct[..., 6])
    useful = unh * f32(use) / 100.0
    manure = unh * f32(graze) / 100.0 * (f32(ret) / 100.0)
    gdisp = unh * f32(graze) / 100.0 * (1.0 - f32(ret) / 100.0)
    disposal = unh * f32(disp) / 100.0
    chaff = unh * f32(resid) / 100.0
    killed = unh * f32(dead_p) / 100.0
    to_remove = (((((useful + manure) + gdisp) + disposal) + chaff) + killed)
    uncut = unh * f32(live_p) / 100.0
    to_remove = torch.where(uncut < 0.0, unh, to_remove)
    first_removal = take_yield | (to_remove > IS_ZERO_TOL_7)
    dropped = terminate & ((cn._seqsum(c.p_gai) - c.u_gai) < 0.0) & first_removal
    u = torch.where(terminate & ~dropped, c.u_bm, torch.zeros_like(c.u_bm))
    a1 = torch.where(take_yield, yl, torch.zeros_like(yl))
    a2 = torch.where(to_remove > IS_ZERO_TOL_7, to_remove, torch.zeros_like(to_remove))
    if mask is not None:
        from . import fate_kernel
        left = fate_kernel.harvest_left(c.p_bm, u, a1, a2, mask)
    else:
        portions = torch.cat([c.p_bm, u[..., None]], -1)
        portions = _remove_newest_first(portions, a1)
        portions = _remove_newest_first(portions, a2)
        left = cn._seqsum(portions)
    stubble = torch.where(terminate, killed + left, killed)
    return chaff, stubble


IS_ZERO_TOL_7 = 0.0000001


# ----------------------------------------------------------------------------- the day

def _where_tree(mask, a, b):
    if a is None:                       # an absent part (no canopy architecture)
        return None
    if isinstance(a, tuple):
        return type(a)(*[_where_tree(mask, x, y) for x, y in zip(a, b)])
    return torch.where(mask.reshape(mask.shape + (1,) * (a.dim() - mask.dim())), a, b)


def _gather_events(E: Events, mgt, k):
    """Events of rows mgt [T, C] and indices k [T, C] (both clamped by the caller's mask)."""
    m = mgt.clamp(min=0)
    j = k.clamp(min=0)
    return Events(*[x[m, j] for x in E])


def _cat_events(parts):
    return Events(*[torch.cat([getattr(p, f) for p in parts], 1) for f in Events._fields])


def _pick(E: Events, sel):
    """The event of each tile where sel [T, C] is True (exactly one or none)."""
    idx = sel.to(torch.long).argmax(-1)
    g = lambda x: (x.gather(1, idx.view(-1, 1, *([1] * (x.dim() - 2))).expand(-1, 1, *x.shape[2:]))[:, 0])
    return Events(*[g(x) for x in E])


def _apply_event(soil: nm.SoilState, lay, e: Events, on, auto_total, auto_mode, max_reps=2, residue_tillage=None):
    for rep in range(1, max_reps + 1):
        m = on & (e.reps >= rep)
        if rep == 1:
            soil = nm.field_operation_t(soil, m, lay, e.sdr, e.ox[:, 0], e.ox[:, 1], e.has_ox,
                                        residue_tillage=residue_tillage)
        else:
            soil = nm.field_operation_t(soil, m, lay, e.sdr, residue_tillage=residue_tillage)
    m = on & (e.nkind == N_INORGANIC)
    soil = soil._replace(n=sn.fertilizer(soil.n, m, lay, e.no3, e.nh4, e.fract, e.inj))
    m = on & (e.nkind == N_SAMPLING)
    sampled = nm.sampled_soil_n(soil.n, lay.thickness, e.depth)
    tot = nm.auto_n_total_t(sampled, e.target, e.est, e.eff, e.mode, e.crit)
    auto_total = torch.where(m, tot, auto_total)
    auto_mode = torch.where(m, e.mode, auto_mode)
    go, a3, a4 = nm.auto_n_split_t(auto_total, e.pct, e.no3, e.nh4)
    m = on & (e.nkind == N_AUTO_APPL) & (auto_mode != nm.AUTO_DISABLED) & go
    soil = soil._replace(n=sn.auto_n_application(soil.n, m, (a3, a4)))
    return soil, auto_total, auto_mode


def day_step_n(st, S, d, moist_mm, ice_mm, soil_t, icpt_m=None):
    """engine.day_step with nitrogen (module docstring). Returns (state, DayOutput)."""
    from . import engine as eg
    N: NStatic = S.n
    ns: NEState = st.n
    lay = N.layers
    if moist_mm is None:
        raise ValueError("day_step (nitrogen): layer moisture needed (liquid + ice, mm)")
    if soil_t is None:
        raise ValueError("day_step (nitrogen): layer soil temperature needed (deg C)")
    ice_mm = torch.zeros_like(moist_mm) if ice_mm is None else ice_mm
    # 1. sowing
    sched = d.sow_crop >= 0
    pc = torch.where(sched, d.sow_crop, st.pend_crop)
    pm = torch.where(sched, d.sow_mgt, st.pend_mgt)
    pmode = torch.where(sched, d.sow_mode, st.pend_mode)
    pthr = torch.where(sched, d.sow_thr, st.pend_thr)
    sow = (pc >= 0) & ((pmode != 0) | (d.warm15 >= pthr))
    cid = torch.where(sow, pc, st.cid)
    mid = torch.where(sow, pm, st.mid)
    pc = torch.where(sow, torch.full_like(pc, -1), pc)
    PP = ph.gather_params(S.pp, cid)
    GP = dl.gather_params(S.gp, cid)
    TP = up.TranspirationParams(*[x[cid] for x in S.tp])
    NP = nt.gather_params(N.NP, cid)
    RP = om.gather_residue_params(N.RP, cid)
    clip = S.clip
    if clip is not None:                            # clipping (clipping.py)
        from . import clipping as cl
        csx = cl.sow(st.clip, clip, mid, sow)
        PP = cl.adjust_params(PP, csx)
    phen = ph.sow(st.phen, PP, sow, st.day)
    grow = dl.sow(st.grow, sow, d.tmax, d.tmin_tom)
    lwp = torch.where(sow, torch.zeros_like(st.lwp), st.lwp)
    crop_n = nt.sow(ns.crop, sow, NP)
    rw = lay.sat * torch.pow(TP.lwp_wilt_point[:, None] / S.air_entry, -1.0 / S.campbell_b)
    real_wilt = torch.where(sow[:, None], rw, ns.real_wilt)
    orc = S.orc
    perennial = None
    if orc is not None:                             # perennial and fruit crops (orchard.py)
        from . import orchard as oc
        OP = oc.gather(orc.OP, cid)
        perennial = OP.perennial
        o, phen, grow = oc.day_begin(st.orc, OP, phen, grow, PP, GP, sow, st.day, d.doy, orc.south,
                                     d.tmax, d.tmin, d.tmin_tom, S.thick, AS=orc.arch)
    # 2. potential
    et_used = torch.where(torch.isnan(st.et_prev), d.et_ref, st.et_prev)
    vpd_max = dl.gr.vpd_max_from_rhmin(d.tmax, d.rhmin)
    g_before = grow
    phn, g1, ctx = dl.potential_start(phen, grow, PP, GP, st.day, d.tmax, d.tmin, d.tmin_tom, d.srad,
                                      vpd_max, et_used, d.co2, d.delta, d.gamma, d.ra, S.thick,
                                      co2_on=S.co2_on)
    ccif, n_started = 1.0, None
    if orc is not None:
        # the crop's N start_day comes before the inactive period of its process_day
        n_started = nt.crop_start_day(crop_n, NP, PP, GP, phn, g_before, g1, lay)
        root_bm = g1.root.biomass
        phn, g1, o, restarted, dormant, hv_ev, ccif = oc.after_start(
            phn, g1, o, OP, PP, GP, st.day, d.doy, d.tmin, (d.tmax + d.tmin) / 2.0, orc.south, S.thick,
            S.h_stage[mid], S.h_offset[mid], S.h_min_lai[mid], S.h_min_bm[mid], AS=orc.arch, srad=d.srad)
        n_started = nt.dormancy_and_restart(n_started, NP, dormant, restarted, root_bm)
        lwp = torch.where(restarted, torch.zeros_like(lwp), lwp)
    ckill = None
    if clip is not None:
        # the crop's N start_day precedes the land unit's operations (crop_cropsyst.cpp:1974-1976)
        if n_started is None:
            n_started = nt.crop_start_day(crop_n, NP, PP, GP, phn, g_before, g1, lay)
        csx, phn, g1, cn_n, cday = cl.morning(csx, clip, cid, mid, phn, g1, PP, GP, st.day, n_started.canopy_n,
                                              n_started.canopy_conc, fruit=None if orc is None else OP.is_fruit,
                                              tavg=(d.tmax + d.tmin) / 2.0)
        n_started = n_started._replace(canopy_n=cn_n)
        lwp = torch.where(cday.lwp_reset, torch.zeros_like(lwp), lwp)
        ckill = cday.kill
        if orc is not None:
            o = oc._rep(o, em_count=o.em_count + cday.lwp_reset.to(torch.long))
            o = oc.harvest(o, cday.fruit_harvest)
            hv_ev = hv_ev | cday.fruit_harvest
    g = dl.potential_process(phn, g1, PP, GP, ctx, S.thick, ccif)
    g_proc = g
    # harvest events of the day (decided by the phenology of the day)
    hs, ho, ht = S.h_stage[mid], S.h_offset[mid], S.h_terminate[mid]
    sd = phn.sync_day.gather(-1, hs.clamp(min=0))
    fire = phn.present[..., None] & (hs >= 0) & (sd >= 0) & (st.day == sd + ho)
    if orc is not None:                             # fruit trees: harvested in after_start
        fire = fire & ~OP.is_fruit[..., None]
    harvest = fire.any(-1)
    terminate = (fire & ht).any(-1)
    hj = fire.to(torch.long).argmax(-1)
    thick = S.thick
    liquid = moist_mm / (thick * 1000.0)
    ice_v = ice_mm / (thick * 1000.0)
    # 3. organic-matter start_day (on CUDA the per-tile kernels of om_kernels: new lists, which the
    # day's tillage then changes in place)
    kern = omk.enabled(moist_mm.device)
    soil = ns.soil._replace(residues=(omk.start_day if kern else om.start_day)(ns.soil.residues))
    # 4. events: yesterday's harvest residue operation, scenario events, crop events
    T = moist_mm.shape[0]
    pend = _gather_events(N.harv_ev, ns.pend_mgt[:, None], ns.pend_h[:, None])
    pend_on = ns.pend_mgt[:, None] >= 0
    dm = torch.index_select(N.day_mgt, 0, st.day.reshape(1))[0]
    dk = torch.index_select(N.day_k, 0, st.day.reshape(1))[0]
    scen = _gather_events(N.scen_ev, dm, dk)
    scen_on = dm >= 0
    K = N.crop_ev.stage.shape[1]
    ce = _gather_events(N.crop_ev, mid[:, None].expand(T, K), torch.arange(K, device=mid.device)[None, :].expand(T, K))
    cst = ce.stage
    csync = torch.where(cst == HARVEST_STAGE, ns.harvest_day[:, None].expand_as(cst),
                        phn.sync_day.gather(-1, cst.clamp(min=0)))
    crop_on = phn.present[:, None] & (cst >= 0) & (csync >= 0) & (st.day == csync + ce.offset) & \
        (mid[:, None] >= 0)
    E = _cat_events([pend, scen, ce])
    on = torch.cat([pend_on, scen_on, crop_on], 1)
    rank = xscan.cumsum(on.to(moist_mm.dtype)).to(torch.long)     # counts: exact
    auto_total, auto_mode = ns.auto_total, ns.auto_mode
    kill = torch.zeros_like(harvest)
    for s in range(SLOTS):
        sel = on & (rank == s + 1)
        has = sel.any(-1)
        e = _pick(E, sel)
        soil, auto_total, auto_mode = _apply_event(soil, lay, e, has, auto_total, auto_mode, N.max_reps,
                                                   omk.tillage_ if kern else None)
        kill = kill | (has & e.terminate & phn.present)
    overflow = ns.overflow | (rank[:, -1] > SLOTS)
    # 5. plant available water (VIC_soil.cpp:129-143, real wilting point), organic matter
    paw = sn._macro_between((liquid - real_wilt) / (lay.fc - real_wilt), torch.zeros_like(liquid),
                            torch.ones_like(liquid))
    env = sn.abiotic_update(liquid, ice_v, soil_t, lay)
    if kern and omk.PROCESS_KERNEL:             # in place, sums in list order (om_kernels.process_)
        res, sN, dec = omk.process_(soil.residues, env, soil.n)
    else:
        res, sN, dec = om.process(soil.residues, env, soil.n)
    soil = soil._replace(residues=res, n=sN)
    # 6. crop: N update, transpiration, actual growth (not on the harvest day)
    crop_day = ~harvest & ~kill
    if clip is not None:
        crop_day = crop_day & ~ckill
    n2, sN2, n_lim, n_pot = nt.crop_potential(crop_n, NP, PP, GP, phn, g_before, g, lay, soil.n, paw, soil_t[:, 1],
                                              started=n_started, perennial=perennial,
                                              auto_plant=None if N.auto_plant is None else N.auto_plant[mid])
    canopy_on = phn.present & g.canopy.exists & crop_day
    n_sd = nt.crop_start_day(crop_n, NP, PP, GP, phn, g_before, g, lay)
    if clip is not None:
        crop_n = _where_tree(ckill, n_started, crop_n)
    crop_n = _where_tree(crop_day, n2, _where_tree(harvest, n_sd, crop_n))
    soil = soil._replace(n=sn.SoilN(torch.where(crop_day[:, None], sN2.no3, soil.n.no3),
                                    torch.where(crop_day[:, None], sN2.nh4, soil.n.nh4)))
    g = g._replace(limited_pot=torch.where(canopy_on, n_pot, g.limited_pot))
    crop_n = _where_tree(crop_day, nt.crop_after_actual(crop_n, phn, g, None if orc is None else OP.is_fruit),
                         crop_n)   # in the crop's process_day
    icpt_used = None
    if d.act_frac is not None:
        act = d.act_frac * g.limited_pot
        lw = d.lwp
        uptake = torch.zeros_like(thick)
        lwp_next = lw
    else:
        psi = up.soil_water_potential(moist_mm, ice_mm, thick, S.max_moist, S.air_entry, S.campbell_b)
        r = up.root_uptake(g.limited_pot, cn.interception_green(g.canopy, GP.kc),
                           torch.zeros_like(lwp) if icpt_m is None else icpt_m,
                           g.root.fract, psi, torch.full_like(psi, eg.PSI_AT_FC), TP, lwp,
                           co2_transp_factor=g.transp_factor, active=g.canopy.exists & phn.present & crop_day)
        act = r.act_transp_m
        lw = r.leaf_water_pot
        uptake = r.uptake_m
        lwp_next = r.leaf_water_pot
        if icpt_m is not None:
            icpt_used = icpt_m - r.interception_m
    stress = nt.leaf_stress_factor(crop_n, NP)
    if orc is None:
        g_act = dl.actual(phn, g, PP, GP, act, lw, d.sunrise, n_limited_growth=n_lim, n_leaf_stress=stress)
    else:
        g_act, o_act = dl.actual(phn, g, PP, GP, act, lw, d.sunrise, n_limited_growth=n_lim, n_leaf_stress=stress,
                                 orchard=(OP, o))
        o = _where_tree(crop_day, o_act, o)
    g = _where_tree(crop_day, g_act, g)
    clip_out = None
    if clip is not None:
        g = _where_tree(ckill, dl._rep(g1, et_max=g_proc.et_max), g)
        act = torch.where(ckill, torch.zeros_like(act), act)
        csx, phn, g, cn_n, aclip, afate, ia2 = cl.evening(csx, clip, cid, mid, phn, g, PP, GP, st.day,
                                                          crop_n.canopy_n, crop_n.canopy_conc,
                                                          tavg=(d.tmax + d.tmin) / 2.0, mask=~ckill)
        crop_n = crop_n._replace(canopy_n=cn_n)
        lwp_next = torch.where(ia2, torch.zeros_like(lwp_next), lwp_next)
        if orc is not None:
            o = oc._rep(o, em_count=o.em_count + ia2.to(torch.long))
        clip_out = cl.day_output(cday, afate, aclip)
    # 7. transformations, residue balance, disturbance
    sN, _, _ = sn.transformations(soil.n, lay, env, soil_t, dec.co2)
    soil = soil._replace(n=sN, residues=om.balance(soil.residues))
    aw = sn._macro_between((liquid - lay.pwp) / (lay.fc - lay.pwp), torch.zeros_like(liquid), torch.ones_like(liquid))
    soil = soil._replace(disturbance=nm.process_disturbance(soil.disturbance, aw, soil.clay))
    # 8. harvest: yield, biomass fate, fallow wilting point, tomorrow's residue operation
    bfh, yl, hi = dl.harvest(g, GP)
    res = soil.residues
    conc = nt.residue_concentration(crop_n, NP)
    chaff, stubble = harvest_fate(g.canopy, yl, N.harv_fate[mid.clamp(min=0), hj], terminate,
                                  mask=harvest if kern else None)
    roots = g.root.fract * g.root.biomass[:, None]
    rmask = harvest
    if clip is not None:                         # the clipping events' fates (land_unit_sim.cpp:2193-2270)
        fz = torch.zeros_like(chaff)
        chaff = torch.where(harvest, chaff, fz) + cday.fate.chaff
        stubble = torch.where(harvest, stubble, fz) + cday.fate.stubble
        harvest = harvest | cday.performed
        rmask = rmask | ckill
    if kern:                                     # the day's own lists, changed in place
        res = omk.harvest_residue_(res, harvest & (chaff > 0.0000001), chaff, harvest & (stubble > 0.0000001),
                                   stubble, rmask & (roots.sum(-1) > 0.0000001), roots, conc, crop_n.root_conc, RP)
    else:
        res = om.add_surface(res, harvest & (chaff > 0.0000001), om.FLAT, chaff, conc, RP)
        res = om.add_surface(res, harvest & (stubble > 0.0000001), om.STUBBLE, stubble, conc, RP)
        res = om.add_subsurface(res, rmask & (roots.sum(-1) > 0.0000001), roots, crop_n.root_conc, RP)
    soil = soil._replace(residues=res)
    wf = lay.sat * torch.pow(torch.full_like(S.air_entry, WILT_FALLOW) / S.air_entry, -1.0 / S.campbell_b)
    real_wilt = torch.where(rmask[:, None], wf, real_wilt)
    if clip is not None:
        hj = torch.where(cday.performed & ~rmask, cday.ev_index.clamp(min=0), hj)
        hj = torch.where(ckill, cday.ev_index.clamp(min=0), hj)
    has_op = harvest & (N.harv_ev.reps[mid.clamp(min=0), hj] > 0)
    neg = torch.full_like(mid, -1)
    pend_mgt = torch.where(has_op, mid, neg)
    pend_h = torch.where(has_op, hj, neg)
    if clip is not None:                         # the first harvest trigger of the season dates its events
        first = harvest & (ns.harvest_day < phn.init_day[:, ph.NGS_PLANTING])
        harvest_day = torch.where(rmask | first, st.day.expand_as(mid), ns.harvest_day)
    else:
        harvest_day = torch.where(harvest, st.day.expand_as(mid), ns.harvest_day)
    overflow = overflow | res.overflow
    rec_present = phn.present
    root_rec = g.root.depth
    orc_out = None
    if orc is not None:
        phn, g, o = oc.after_actual(phn, g, o, OP, st.day, S.thick)
        hv = hv_ev
        yl = torch.where(hv, o.latest_yield, yl)
        hi = torch.where(hv, torch.zeros_like(hi), hi)
        harvest = harvest | hv
        orc_out = oc.day_output(o, phn, hv, restarted)
    if clip is not None:
        cut = (clip_out.event & ~cday.fruit_harvest) | clip_out.auto
        yl = torch.where(harvest & ~cday.performed, yl, torch.zeros_like(yl)) + clip_out.harvested
        hi = torch.where(harvest & ~cday.performed, hi, torch.zeros_like(hi))
        harvest = harvest | cut
        terminate = terminate | ckill
        csx = cl.end_crop(csx, terminate | kill)
    phn_out = ph.terminate(phn, terminate | kill)
    c = g.canopy
    nout = NDayOutput(stress, crop_n.canopy_n, sn.no3_n(soil.n), sn.nh4_n(soil.n), overflow)
    out = eg.DayOutput(
        gai=cn.gai_vital(c), biomass=cn.biomass_vital(c),
        root_depth=torch.where(rec_present, root_rec, torch.zeros_like(root_rec)),
        green_cover=cn.interception_green(c, GP.kc), et_max=g.et_max, limited_pot=g.limited_pot,
        act_transp=act, water_stress=g.ws, lwp=lw, uptake=uptake,
        yield_=torch.where(harvest, yl, torch.zeros_like(yl)), hi=torch.where(harvest, hi, torch.zeros_like(hi)),
        harvest=harvest, sown=sow, stage=phn.stage,
        lai=torch.where(phn_out.present & c.exists, cn.lai(c), torch.zeros_like(g.ws)),
        present=phn_out.present, active_record=rec_present, n=nout, orc=orc_out, clip=clip_out, icpt_used=icpt_used,
        cover_total=torch.where(phn_out.present & c.exists, c.cover_total, torch.zeros_like(c.cover_total)))
    nn = NEState(soil, crop_n, real_wilt, auto_total, auto_mode, harvest_day, pend_mgt, pend_h, overflow)
    new = eg.CropSystState(st.day + 1, phn_out, g, cid, mid, lwp_next, d.et_ref, pc, pm, pmode, pthr, nn,
                           o if orc is not None else None, csx if clip is not None else None)
    return new, out
