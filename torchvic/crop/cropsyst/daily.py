"""The crop day of the VIC-CropSyst port around E4-E9: phenology (E6) plus
potential transpiration, growth, canopy, roots and yield, in the order of the shipped build.

The day is split at the water uptake so that E1 (uptake.py) can run in between:

  potential(...)  Land_unit_simulation_VX::start_day ET max (land_unit_sim.cpp:1296-1322, on
                  yesterday's canopy), Crop_complete::start_day (crop_cropsyst.cpp:1937-2003:
                  temperatures, phenology, thermal-time events with their initiate_* side effects
                  :2286-2427, CO2 update, roots start_day, grain stress period events), and the
                  first half of process_day (:2715-2785: root update, canopy update, CO2
                  adjustments, intercepted radiation, attainable growth, limited potential
                  transpiration). Returns the state and the limited potential transpiration [m],
                  the green cover and the root fractions E1 needs.
  actual(...)     process_transpiration_m's bookkeeping (crop_common.cpp) with the E1 result
                  (actual transpiration [m], leaf water potential [J/kg]), actual_growth
                  (crop_cropsyst.cpp:1622-1738) and end_day (:2005-2117).

Inputs the coupled run must supply each day: see growth.py (ET_ref of the previous day, VPD max
= es(Tmax)(1 - RHmin/100), radiation, CO2, delta, gamma, ra), Tmin of tomorrow (hourly
temperatures), the sunrise hour (yield_.sunrise_hour) and the layer thicknesses.

Nitrogen is disabled (CROP_NITROGEN_DISABLED). The CRB scenarios run with
run_nitrogen=true; when nitrogen does not limit, the reference path reduces to this one
(crop_N_V5.cpp:34-71: stress factor 1, TUE reduction 1, N-limited growth = attainable).

Event side effects (masks on the phenology state; several may fire on one day, in this order):
emergence -> new canopy (restart_with initial_GAI/SLA), new roots, growth starts; culminescence
expiry -> canopy know_senescence; senescence expiry -> know_senescence(0) (die-back stops); filling -> biomass at filling; maturity -> biomass at maturity,
attainable growth stops (biomass_growth deleted). Grain stress periods start on (stage day +
offset) after the day's accumulation, like the phenology parameter adjustments.

Everything is batched over the leading shape (e.g. [cell, crop tile]); no host syncs, no
data-dependent control flow; static loops only over parameter slots.
"""
from __future__ import annotations

import re
from typing import NamedTuple

import torch

from . import canopy as cn
from . import growth as gr
from . import phenology as ph
from . import root as rt
from . import yield_ as yd
from .growth import round_f32
from .readers import IniTable

STAGE_WORDS = {w: i for i, w in enumerate(ph.STAGE_LABELS)}



def _rep(nt, **kw):
    """NamedTuple._replace without collections internals (traceable by torch.compile)."""
    return type(nt)(*[kw[f] if f in kw else getattr(nt, f) for f in type(nt)._fields])

def _f32(x):
    return x.to(torch.float32).to(x.dtype)


class GrowthParams(NamedTuple):
    """Per-crop E4-E9 parameters (float32-rounded like the reference structs), any batch shape."""
    tue_1kpa: torch.Tensor
    tue_scaling: torch.Tensor
    rue: torch.Tensor                    # g/MJ
    tl_min: torch.Tensor
    tl_low: torch.Tensor
    tl_high: torch.Tensor
    tl_max: torch.Tensor
    et_coef: torch.Tensor
    kc: torch.Tensor                     # light extinction coefficient
    reduce_lwp: torch.Tensor
    stop_lwp: torch.Tensor
    max_height: torch.Tensor
    initial_gai: torch.Tensor
    regrowth_gai: torch.Tensor
    max_lai: torch.Tensor
    sla: torch.Tensor
    fract_lai_mature: torch.Tensor
    stem_leaf_partition: torch.Tensor
    root_max_depth: torch.Tensor
    root_surface_density: torch.Tensor
    root_curvature: torch.Tensor
    root_sensitivity: torch.Tensor
    root_length_emergence_cm: torch.Tensor
    root_shoot_emergence: torch.Tensor
    root_shoot_full: torch.Tensor
    hi_unstressed: torch.Tensor
    hi_adjust: torch.Tensor              # bool
    hi_extreme: torch.Tensor             # bool, temperature extreme response
    hi_kind: torch.Tensor                # long, yield model of the harvested part (yield_.KIND_*)
    translocation_max: torch.Tensor
    co2_tue_conc: torch.Tensor
    co2_rue_conc: torch.Tensor
    co2_alpha: torch.Tensor
    co2_theta: torch.Tensor
    co2_gra: torch.Tensor
    gsp_stage: torch.Tensor              # [..., G] long NGS stage (-1 unused)
    gsp_offset: torch.Tensor             # [..., G] long
    gsp: torch.Tensor                    # [..., G, 10] cold(no,max,sens,init,hours) heat(...)


GROWTH_DEFAULTS = dict(            # crop_param_class.cpp constructors (CS_VERSION 5)
    tue_1kpa=5.0, tue_scaling=0.45, rue=4.0, tl_min=3.0, tl_low=10.0, tl_high=25.0, tl_max=45.0,
    et_coef=1.0, kc=0.5, reduce_lwp=-800.0, stop_lwp=-1200.0, max_height=1.5,
    initial_gai=0.011, regrowth_gai=0.011, max_lai=5.0, sla=22.0, fract_lai_mature=0.8,
    stem_leaf_partition=3.0, root_max_depth=1.5, root_surface_density=3.0, root_curvature=0.1,
    root_sensitivity=0.0, root_length_emergence_cm=1.2, root_shoot_emergence=2.0,
    root_shoot_full=0.1, hi_unstressed=0.5, translocation_max=0.4, co2_tue_conc=350.0,
    co2_rue_conc=350.0, co2_alpha=0.0032725, co2_theta=0.941574, co2_gra=1.2771338)
# .crp key of each parameter as the shipped V4 parser maps it (verified against the reference
# parser in tests/test_cropsyst_growth.py); RUE and the temperature limitation have no key in a
# CS_VERSION 5 build (crop_param_class.cpp:136-153).
GROWTH_KEYS = dict(
    tue_1kpa="growth.TUE_at_1pKa_VPD", tue_scaling="growth.TUE_scaling_coef",
    et_coef="transpiration.ET_crop_coef", kc="transpiration.kc",
    reduce_lwp="growth.LWP_reduces_canopy_expansion", stop_lwp="growth.LWP_stops_canopy_expansion",
    max_height="morphology.max_canopy_height", initial_gai="leaf.initial_GAI",
    regrowth_gai="leaf.regrowth_GAI", max_lai="leaf.max_LAI", sla="leaf.specific_leaf_area",
    fract_lai_mature="leaf.fract_LAI_mature", stem_leaf_partition="leaf.stem_leaf_partition",
    root_max_depth="root.max_root_depth", root_surface_density="root.surface_root_density",
    root_curvature="root.root_density_distribution_curvature",
    root_sensitivity="root.root_sensitivity_water_stress",
    root_length_emergence_cm="root.root_length_at_emergence",
    root_shoot_emergence="root.root_shoot_emergence_ratio", root_shoot_full="root.root_shoot_full_ratio",
    hi_unstressed="harvest_index.unstressed", translocation_max="harvest_index.translocation_max",
    co2_tue_conc="CO2.TUE_reference_conc", co2_rue_conc="CO2.RUE_reference_conc",
    co2_alpha="CO2.alpha", co2_theta="CO2.theta", co2_gra="CO2.growth_ratio_asymptotic")
GSP_KEYS = ["cold_stress_no_stress_threshold_temperature", "cold_stress_max_stress_threshold_temperature",
            "cold_stress_sensitivity", "cold_stress_exposure_initial_hour", "cold_stress_exposure_damaging_hours",
            "heat_stress_no_stress_threshold_temperature", "heat_stress_max_stress_threshold_temperature",
            "heat_stress_sensitivity", "heat_stress_exposure_initial_hour", "heat_stress_exposure_damaging_hours"]
GSP_DEFAULTS = [-0.5, -2.0, 1.0, 0, 4.0, 31.0, 44.0, 1.0, 99, 4.0]   # crop_param_struct.h set_default


def _yield_kinds(crops, n, dev):
    """Yield model per crop from `crop.harvested_part` (crop_cropsyst.cpp:388-466): grain, tuber,
    canopy, complete, else the simple harvest index (root, bulb, leaf). "seed" is the V4 label of
    the grain part; fruit (orchards) is not handled by the port and keeps the grain model. A tuber crop with a nonzero
    tuber sensitivity (water/temperature stress statistics of the tuber periods) is not ported."""
    kind = torch.zeros(n, dtype=torch.long, device=dev)
    t = crops.params.get("crop.harvested_part")
    if t is None:
        return kind
    words = crops.vocab.get("crop.harvested_part", [])
    codes = {w: i for i, w in enumerate(words)}
    for i in range(n):
        w = words[int(t[i])] if 0 <= int(t[i]) < len(words) else "grain"
        k = {"grain": yd.KIND_GRAIN, "seed": yd.KIND_GRAIN, "fruit": yd.KIND_GRAIN, "tuber": yd.KIND_TUBER,
             "canopy": yd.KIND_CANOPY, "complete": yd.KIND_COMPLETE}.get(w, yd.KIND_SIMPLE)
        kind[i] = k
        if k == yd.KIND_TUBER:
            for key in ("harvest_index.tuber_init_sensitivity", "harvest_index.tuber_growth_sensitivity"):
                v = crops.params.get(key)
                if v is not None and float(v[i]) == float(v[i]) and abs(float(v[i])) > 1e-12:
                    raise NotImplementedError(f"cropsyst: {key} != 0 (tuber stress statistics) is not ported")
    return kind


def growth_params(crops: IniTable, device=None, dtype=torch.float64) -> GrowthParams:
    """[n_crop] E4-E9 parameters from a crop table (readers.crop_table)."""
    n = crops.n
    dev = device if device is not None else next(iter(crops.params.values())).device

    def get(key, default):
        t = crops.params.get(key) if key else None
        full = torch.full((n,), float(default), dtype=dtype, device=dev)
        if t is None:
            return full
        t = t.to(dev)
        missing = torch.isnan(t) if t.dtype.is_floating_point else t == -1
        return torch.where(missing, full, t.to(dtype))

    def get_bool(key, default):
        t = crops.params.get(key)
        return torch.full((n,), bool(default), device=dev) if t is None else t.to(dev)

    vals = {k: _f32(get(GROWTH_KEYS.get(k), v)) for k, v in GROWTH_DEFAULTS.items()}
    G = crops.params["grain_stress_period.event_offset"].shape[1] if "grain_stress_period.event_offset" in crops.params else 0
    stage = torch.full((n, max(G, 1)), -1, dtype=torch.long)
    offset = torch.zeros((n, max(G, 1)), dtype=torch.long)
    gsp = torch.tensor(GSP_DEFAULTS, dtype=dtype).repeat(n, max(G, 1), 1)
    if G:
        P, V = crops.params, crops.vocab
        valid = crops.valid["grain_stress_period"].cpu()
        for i in range(n):                         # setup-time parsing, host side
            for j in range(G):
                if not bool(valid[i, j]):
                    continue
                if "grain_stress_period.enabled" in P and not bool(P["grain_stress_period.enabled"][i, j]):
                    continue
                sync = V["grain_stress_period.event_synchronization"][int(P["grain_stress_period.event_synchronization"][i, j])]
                if sync != "after_normal_crop_growth_stage":
                    continue
                stage[i, j] = STAGE_WORDS[V["grain_stress_period.event_phenologic"][int(P["grain_stress_period.event_phenologic"][i, j])]]
                offset[i, j] = int(P["grain_stress_period.event_offset"][i, j])
                for q, key in enumerate(GSP_KEYS):
                    t = P.get(f"grain_stress_period.{key}")
                    if t is not None:
                        x = float(t[i, j])
                        if not (x != x or (not t.dtype.is_floating_point and x == -1)):
                            gsp[i, j, q] = x
    return GrowthParams(
        **vals, hi_adjust=get_bool("harvest_index.adjust_HI", True),
        hi_extreme=get_bool("harvest_index.heat_extreme", False),
        hi_kind=_yield_kinds(crops, n, dev),
        gsp_stage=stage.to(dev), gsp_offset=offset.to(dev), gsp=_f32(gsp).to(dev))


def gather_params(p: GrowthParams, index) -> GrowthParams:
    return GrowthParams(*[x[index] for x in p])


class GrowthState(NamedTuple):
    canopy: cn.CanopyState
    root: rt.RootState
    yld: yd.YieldState
    growth_on: torch.Tensor          # bool, biomass_growth object exists (emergence..maturity)
    ws: torch.Tensor                 # water stress factor (current), float32 value
    ws_index: torch.Tensor           # 1 - ws
    co2_tue_adjustment: torch.Tensor
    tue_co2_adjusted: torch.Tensor   # mutable TUE of the last evaluation (init 1)
    et_max: torch.Tensor
    pot_unadj: torch.Tensor
    pot_co2: torch.Tensor
    transp_factor: torch.Tensor
    intercepted: torch.Tensor
    attainable: torch.Tensor
    limited_pot: torch.Tensor
    act_transp: torch.Tensor
    act_growth: torch.Tensor
    gai_rel_growth: torch.Tensor
    wcef: torch.Tensor
    overall_stress: torch.Tensor
    srf: torch.Tensor                # senescence reduction factor used today
    gsp_done: torch.Tensor           # [..., G] bool


def initial_state(shape, L, G, P=400, dtype=torch.float64, device=None) -> GrowthState:
    z = torch.zeros(shape, dtype=dtype, device=device)
    return GrowthState(cn.initial_state(shape, P, dtype, device), rt.initial_state(shape, L, dtype, device),
                       yd.initial_state(shape, G, dtype, device),
                       torch.zeros(shape, dtype=torch.bool, device=device), z + 1.0, z, z + 1.0, z + 1.0,
                       z, z, z, z + 1.0, z, z, z, z, z, z, z - 1.0, z, z + 1.0,
                       torch.zeros(tuple(shape) + (max(G, 1),), dtype=torch.bool, device=device))


def sow(g: GrowthState, mask, tmax, tmin_tomorrow) -> GrowthState:
    """A new Crop_complete object on the masked tiles (call with phenology.sow, before the day)."""
    m = mask
    z = torch.zeros_like(g.ws)
    w = lambda new, old: torch.where(m, new, old)
    zc = cn.initial_state(m.shape, g.canopy.p_gai.shape[-1], z.dtype, z.device)
    c = cn.CanopyState(*[torch.where(m.reshape(m.shape + (1,) * (a.dim() - m.dim())), b, a)
                         for a, b in zip(g.canopy, zc)])
    zr = rt.initial_state(m.shape, g.root.fract.shape[-1], z.dtype, z.device)
    r = rt.RootState(*[torch.where(m.reshape(m.shape + (1,) * (a.dim() - m.dim())), b, a)
                       for a, b in zip(g.root, zr)])
    return _rep(g, canopy=c, root=r, yld=yd.sow(g.yld, m, tmax, tmin_tomorrow),
                      growth_on=g.growth_on & ~m, ws=w(z + 1.0, g.ws), ws_index=w(z, g.ws_index),
                      co2_tue_adjustment=w(z + 1.0, g.co2_tue_adjustment),
                      tue_co2_adjusted=w(z + 1.0, g.tue_co2_adjusted), intercepted=w(z, g.intercepted),
                      limited_pot=w(z, g.limited_pot), act_transp=w(z, g.act_transp),
                      gsp_done=g.gsp_done & ~m[..., None])


class StartContext(NamedTuple):
    """Values potential_start hands to potential_process (the crop start_day's results that the
    first half of process_day uses)."""
    tmax: torch.Tensor
    tmin: torch.Tensor
    srad: torch.Tensor
    vpd_max: torch.Tensor
    etm: torch.Tensor
    pot: torch.Tensor
    srf: torch.Tensor
    tf: torch.Tensor
    tue_adj: torch.Tensor
    gain_rue: torch.Tensor
    bmv: torch.Tensor
    co2_on: torch.Tensor


def potential(phs: ph.PhenologyState, g: GrowthState, PP: ph.PhenologyParams, GP: GrowthParams,
              day, tmax, tmin, tmin_tomorrow, srad, vpd_max, et_ref, co2_ppm, delta, gamma, ra,
              thickness, sunrise_hour, co2_on=True, day_length=None):
    """First half of the day; returns (phenology state, growth state)."""
    phn, g1, ctx = potential_start(phs, g, PP, GP, day, tmax, tmin, tmin_tomorrow, srad, vpd_max, et_ref,
                                   co2_ppm, delta, gamma, ra, thickness, co2_on=co2_on,
                                   day_length=day_length)
    return phn, potential_process(phn, g1, PP, GP, ctx, thickness)


def potential_start(phs: ph.PhenologyState, g: GrowthState, PP: ph.PhenologyParams, GP: GrowthParams,
                    day, tmax, tmin, tmin_tomorrow, srad, vpd_max, et_ref, co2_ppm, delta, gamma, ra,
                    thickness, co2_on=True, day_length=None):
    """ET max on yesterday's canopy and Crop_complete::start_day (crop_cropsyst.cpp:1937-2003):
    temperatures, phenology and its thermal-time events with their side effects, CO2 update, roots
    start_day, grain stress periods. Returns (phenology state, growth state, StartContext).
    Orchard crops (PP.is_fruit): the ET crop coefficient is not adjusted by LAI (crop_cropsyst.cpp:
    3612-3613), fruit growth replaces grain filling (no biomass at filling), and the biomass growth
    object survives maturity (crop_cropsyst.cpp:2412-2416)."""
    dt = g.ws.dtype
    T = lambda x: torch.as_tensor(x, dtype=dt, device=g.ws.device) + torch.zeros_like(g.ws)
    tmax, tmin, tmin_tomorrow, srad, vpd_max, et_ref = map(T, (tmax, tmin, tmin_tomorrow, srad, vpd_max, et_ref))
    delta, gamma, ra = map(T, (delta, gamma, ra))
    co2_on = torch.as_tensor(co2_on, device=g.ws.device) & torch.ones_like(g.growth_on)
    on = phs.present
    c = g.canopy
    # E4 on yesterday's canopy
    kadj = torch.where(c.exists, gr.adjusted_et_coef(c.lai_ss, GP.et_coef), torch.ones_like(c.lai_ss))
    kfruit = torch.where(GP.et_coef == 0.0, torch.ones_like(kadj), GP.et_coef + torch.zeros_like(kadj))
    kadj = torch.where(PP.is_fruit, kfruit, kadj)
    etm = gr.et_max(et_ref, kadj)
    pot = cn.interception_green(c, GP.kc) * etm
    # Crop_complete::start_day
    srf = cn.senescence_reduction_factor(c)
    phn = ph.step(phs, PP, day, tmax, tmin, day_length=day_length, water_stress_index=g.ws_index,
                  vpd_max=vpd_max, senescence_reduction_factor=srf)
    y = yd.hourly_temperatures(g.yld, on, tmax, tmin_tomorrow, phn.tmax_adj)
    dayl = torch.as_tensor(day, device=on.device)
    dk = dayl[..., None] if dayl.dim() else dayl
    hit = lambda code: on & (phn.sync_day[..., code] == dayl)
    emerg = hit(ph.NGS_EMERGENCE)
    c = cn.restart(c, emerg, GP.initial_gai, GP.sla, GP.kc)
    r = rt.create(g.root, emerg & ~g.root.exists, GP.root_length_emergence_cm, thickness)
    b0 = (GP.initial_gai.to(torch.float32) / GP.sla.to(torch.float32)).to(dt)   # float32 division (crop_cropsyst.cpp:2302)
    r = _rep(r, above_ground=torch.where(emerg, b0 + torch.zeros_like(r.above_ground), r.above_ground))
    growth_on = g.growth_on | emerg
    sen = on & c.exists & phn.period_on[..., ph.SEN] & ~phs.period_on[..., ph.SEN]
    c = cn.know_senescence(c, sen, PP.dur_senescence)
    # senescence expired -> culminate_senescence -> know_senescence(0): the canopy forgets the
    # senescence period and the die-back stops (crop_cropsyst.cpp:2368-2372, 2997-3001)
    sen_exp = on & c.exists & phn.period_on[..., ph.SEN] & (phn.period_tt[..., ph.SEN] > PP.dur_senescence)
    c = _rep(c, sen_known=c.sen_known & ~sen_exp)
    bmv = torch.where(c.exists, cn.biomass_vital(c), torch.zeros_like(c.u_bm))
    fill = hit(ph.NGS_FILLING) & ~PP.fruit_harvest
    mat = hit(ph.NGS_MATURITY)
    y = _rep(y, bm_filling=torch.where(fill, bmv, y.bm_filling), bm_maturity=torch.where(mat, bmv, y.bm_maturity))
    growth_on = growth_on & ~(mat & ~PP.is_fruit)
    # CO2 update (crop_cropsyst.cpp:1959-1964)
    cond, gain_tue, gain_rue = gr.co2_gains(torch.as_tensor(co2_ppm, dtype=dt, device=on.device), GP.co2_alpha,
                                            GP.co2_theta, GP.co2_gra, GP.co2_tue_conc, GP.co2_rue_conc)
    tf, tue_adj = gr.co2_daily(cond, gain_tue, delta, gamma, ra)
    tf = torch.where(co2_on, tf, torch.ones_like(tf))
    r = rt.start_day(r)
    # grain stress periods (event_scheduler_crop.cpp:96-112)
    sd = phn.sync_day.gather(-1, GP.gsp_stage.clamp(min=0))
    fire = on[..., None] & (GP.gsp_stage >= 0) & ~g.gsp_done & (sd >= 0) & (dk == sd + GP.gsp_offset)
    fire = fire & GP.hi_extreme[..., None]
    y = yd.start_periods(y, fire)
    g1 = _rep(g, canopy=c, root=r, yld=y, growth_on=growth_on, gsp_done=g.gsp_done | fire)
    return phn, g1, StartContext(tmax, tmin, srad, vpd_max, etm, pot, srf, tf, tue_adj, gain_rue, bmv, co2_on)


def potential_process(phn: ph.PhenologyState, g1: GrowthState, PP: ph.PhenologyParams,
                      GP: GrowthParams, ctx: StartContext, thickness, ccif=1.0):
    """The first half of Crop_complete::process_day (crop_cropsyst.cpp:2715-2785) up to the limited
    potential transpiration: roots, canopy, CO2 adjustments, intercepted radiation, attainable growth.
    g1: the growth state after potential_start (and any inactive-period changes in between). Orchard crops (PP.is_fruit): the root length is the maximum depth and is
    not updated (crop_cropsyst.cpp:2719-2720), no RUE temperature limitation
    (biomass_growth_RUE_TUE.cpp:78-79); ccif is the perennial season biomass adjustment
    (biomass_growth_RUE_TUE.cpp:126-132)."""
    g = g1
    dt = g.ws.dtype
    on = phn.present
    c, r, y, growth_on = g1.canopy, g1.root, g1.yld, g1.growth_on
    tmax, tmin, srad, vpd_max = ctx.tmax, ctx.tmin, ctx.srad, ctx.vpd_max
    etm, pot, srf, tf, tue_adj, gain_rue, bmv, co2_on = (ctx.etm, ctx.pot, ctx.srf, ctx.tf, ctx.tue_adj,
                                                          ctx.gain_rue, ctx.bmv, ctx.co2_on)
    # process_day: roots
    above = torch.where(c.exists, cn.biomass_vital(c), torch.zeros_like(bmv))
    r_rel = phn.period_rel_remaining[..., ph.ROOT].to(dt)
    r_exp = phn.period_tt[..., ph.ROOT] > PP.dur_root_elongation
    r = rt.update(r, g.ws, above, phn.period_tt[..., ph.ROOT], PP.dur_root_elongation, r_rel, r_exp, thickness,
                  GP.root_max_depth, GP.root_surface_density, GP.root_curvature, GP.root_sensitivity,
                  GP.root_length_emergence_cm, GP.root_shoot_emergence, GP.root_shoot_full)
    rfruit = _rep(g1.root, length=torch.where(g1.root.exists, round_f32(GP.root_max_depth), g1.root.length))
    r = rt.RootState(*[torch.where(PP.is_fruit.reshape(PP.is_fruit.shape + (1,) * (a.dim() - PP.is_fruit.dim())),
                                   b, a) for a, b in zip(r, rfruit)])
    # canopy
    ce = c.exists & on
    c = cn.process_day(c, GP.kc, GP.max_height, GP.max_lai)
    green = cn.interception_green(c, GP.kc)
    co2_adj = torch.where(ce & co2_on, torch.where(green > gr.GREEN_TUE_THRESHOLD, tue_adj, torch.ones_like(tue_adj)),
                          g.co2_tue_adjustment)
    pot_co2 = torch.where(ce & co2_on, pot * tf, pot)
    intercepted = torch.where(ce, srad * green, g.intercepted)
    tsf = gr.rue_temperature_limitation((tmax + tmin) / 2.0, GP.tl_min, GP.tl_low, GP.tl_high, GP.tl_max)
    tsf = torch.where(PP.is_fruit, torch.ones_like(tsf), tsf)
    gain_rue = torch.where(co2_on, gain_rue, torch.ones_like(gain_rue))
    tue_un = gr.tue_unadjusted(gr.vpd_daytime(vpd_max), GP.tue_1kpa, GP.tue_scaling)
    att, tue = gr.attainable_growth(intercepted, GP.rue, gain_rue, tsf, pot_co2, tue_un, co2_adj, ccif)
    run = ce & growth_on
    att = torch.where(run, att, torch.zeros_like(att))
    tue_co2 = torch.where(run, tue, g.tue_co2_adjusted)
    limited = torch.where(ce, gr.limited_pot_transpiration(att, tue_co2), g.limited_pot)
    g2 = _rep(g, canopy=c, root=r, yld=y, growth_on=growth_on, co2_tue_adjustment=co2_adj,
                    tue_co2_adjusted=tue_co2, et_max=torch.where(on, etm, g.et_max),
                    pot_unadj=torch.where(on, pot, g.pot_unadj), pot_co2=torch.where(on, pot_co2, g.pot_co2),
                    transp_factor=tf, intercepted=intercepted, attainable=torch.where(on, att, g.attainable),
                    limited_pot=limited, srf=torch.where(on, srf, g.srf),
                    act_transp=torch.where(on, torch.zeros_like(att), g.act_transp),
                    act_growth=torch.where(on, torch.zeros_like(att), g.act_growth),
                    gai_rel_growth=torch.where(on, torch.zeros_like(att), g.gai_rel_growth),
                    wcef=torch.where(on, torch.full_like(att, -1.0), g.wcef),
                    overall_stress=torch.where(on, torch.ones_like(att), g.overall_stress),
                    gsp_done=g1.gsp_done)
    return g2


def actual(phs: ph.PhenologyState, g: GrowthState, PP: ph.PhenologyParams, GP: GrowthParams,
           act_transp, lwp, sunrise_hour, n_limited_growth=None, n_leaf_stress=None, orchard=None):
    """Second half of the day: transpiration bookkeeping, actual growth, canopy, end_day.
    With nitrogen on pass the N-limited growth and the N leaf stress factor of the day
    (nitrogen.py); g.limited_pot must then be the N-limited potential transpiration.
    orchard=(OrchardParams, OrchardState) gathered to the tiles: fruit trees grow on through
    maturity (crop_cropsyst.cpp:1633-1640), part of the growth goes to the fruit or the reserves
    (Crop_orchard_fruit::distribute_todays_non_leaf_stem_growth, crop_fruit.cpp:146-268), leaf
    growth stops once the LAI passed max_LAI, and the self-shading LAI is the GAI
    (canopy_growth_LAI_based.cpp:222-228). Returns (growth state, orchard state) then."""
    dt = g.ws.dtype
    T = lambda x: torch.as_tensor(x, dtype=dt, device=g.ws.device) + torch.zeros_like(g.ws)
    act_transp, lwp, sunrise_hour = map(T, (act_transp, lwp, sunrise_hour))
    c = g.canopy
    tr = phs.present & c.exists
    act = torch.where(act_transp < 0.0, torch.zeros_like(act_transp), act_transp)
    ws, wsi = gr.water_stress(act, g.limited_pot)
    ws = torch.where(tr, ws, g.ws)
    wsi = torch.where(tr, wsi, g.ws_index)
    not_mat = phs.stage < ph.NGS_MATURITY
    if n_limited_growth is None:
        ag, gai_rel, wcef, overall = gr.actual_growth(g.attainable, g.limited_pot, act, lwp, GP.reduce_lwp, GP.stop_lwp)
    else:   # Crop_nitrogen_V5 path of calc_act_biomass_growth (crop_cropsyst.cpp:2847-2947)
        from . import nitrogen as nt
        ag, gai_rel, wcef, overall = nt.actual_growth(g.attainable, g.limited_pot, n_limited_growth, act, lwp,
                                                      GP.reduce_lwp, GP.stop_lwp, n_leaf_stress)
    fruit = PP.is_fruit
    grow = tr & (not_mat | ((phs.stage == ph.NGS_MATURITY) & fruit))
    ag = torch.where(grow, ag, torch.zeros_like(ag))
    gai_rel = torch.where(grow, gai_rel, torch.zeros_like(ag))
    wcef = torch.where(grow, wcef, g.wcef)
    overall = torch.where(grow, overall, g.overall_stress)
    # yield know_stress (yield.cpp:205-212)
    in_acc = phs.period_on[..., ph.ACC]
    upd = tr & GP.hi_extreme & (in_acc | (phs.stage == ph.NGS_ANTHESIS))
    gs = GP.gsp
    y = yd.update_extreme(g.yld, upd, sunrise_hour, *[gs[..., q] for q in range(10)])
    # canopy: senescence die-back, new leaf, self-shading LAI
    c = cn.age_leaf(c, tr, phs.gdd)
    quies = phs.stage == ph.NGS_QUIESCENCE
    leaf, cont, orc = ag, in_acc, None
    if orchard is not None:
        from . import orchard as oc
        OP, orc = orchard
        leaf, orc = oc.distribute(orc, OP, phs.stage, ag, cn.lai(c), GP.max_lai, tr & ~quies)
        cont = (in_acc | ~PP.has_leaf_senescence) & ~orc.limit_lai
    c = cn.develop_leaf(c, tr & ~quies, cont, leaf, gai_rel, GP.sla, GP.stem_leaf_partition, GP.regrowth_gai)
    tt, dur = phs.period_tt, [PP.dur_accrescence, PP.dur_culminescence, PP.dur_senescence]
    pon = phs.period_on
    expired_acc = pon[..., ph.ACC] & (tt[..., ph.ACC] > dur[0])
    in_culm = pon[..., ph.CULM] & ~(tt[..., ph.CULM] > dur[1])
    in_sen = pon[..., ph.SEN] & ~(tt[..., ph.SEN] > dur[2])
    matured = phs.stage >= ph.NGS_MATURITY
    c = cn.update_lai_self_shading(c, tr, expired_acc | in_culm | in_sen, ~matured,
                                   phs.period_rel_remaining[..., ph.SEN].to(dt), GP.fract_lai_mature,
                                   fruit=fruit)
    g = _rep(g, canopy=c, yld=y, ws=ws, ws_index=wsi,
                      act_transp=torch.where(tr, act, g.act_transp), act_growth=ag, gai_rel_growth=gai_rel,
                      wcef=wcef, overall_stress=overall)
    return g if orchard is None else (g, orc)


def harvest(g: GrowthState, GP: GrowthParams):
    """harvest_crop's yield (crop_cropsyst.cpp:1849-1866): (biomass for harvest, yield, HI)."""
    c = g.canopy
    bfh = g.yld.bm_maturity
    zero = torch.abs(0.0 - bfh) <= cn.IS_ZERO_TOL
    bfh = torch.where(zero, torch.where(c.exists, cn.biomass_vital(c), torch.zeros_like(bfh)), bfh)
    yl, hi = yd.grain_yield(bfh, g.yld.bm_filling, g.yld.bm_maturity, GP.hi_unstressed, GP.hi_adjust,
                            GP.translocation_max, yd.extreme_factor(g.yld))
    yl, hi = yd.other_yield(GP.hi_kind, bfh, GP.hi_unstressed, yl, hi)
    return bfh, yl, hi
