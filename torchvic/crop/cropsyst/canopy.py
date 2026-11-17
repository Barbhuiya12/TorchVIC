"""E7 of the VIC-CropSyst port (docs/connectors/cropsyst.md): the
LAI-based canopy the CRB crops use (canopy_growth=leaf_area_index_V5 ->
Canopy_growth_leaf_area_index_based_V5, crop_cropsyst.cpp:817-827).

Sources: canopy_growth_LAI_based_V5.cpp (age_leaf by senescence die-back, know_senescence),
canopy_growth_LAI_based.cpp (portions, develop_leaf, restart_with, cover fractions, self-shading
LAI, plant height, ET coefficient), canopy_growth_portioned.cpp (take_portion, die_back_to_GAI,
the untouchable portion).

Layout: the reference keeps a linked list of daily canopy portions. Here the vital portions are a
fixed-capacity slot axis [..., P] filled in order (slot = number of portions so far); the
"untouchable" restart portion is separate. Sums over portions are reductions over the slot axis,
the in-order die-back is a cumulative sum with masks. No host syncs, no data-dependent control
flow.

Reference quirks ported as is:
- V5 never ages portions by leaf duration (age_by is empty, canopy_growth_LAI_based_V5.cpp:26-29);
  leaves die only by the senescence die-back, GAI(t) -= GAI_at_senescence_start /
  senescence_duration * GDD, until the senescence period expires: culminate_senescence then
  calls know_senescence(0), which clears the canopy's period pointer and stops the die-back
  (crop_cropsyst.cpp:2368-2372); the GAI left (often the untouchable portion's) stays.
- die_back_to_GAI writes dead_AI = the GAI removed TODAY (it overwrites, it does not add), and a
  portion that dies writes dead_AI = its remaining GAI; portions after the die-back front get
  dead_AI = 0 (canopy_growth_portioned.cpp:377-419, canopy_growth_LAI_based.cpp:96-106). The LAI
  (GAI + dead AI) is therefore not conserved.
- The untouchable portion contributes its LAI (GAI + dead AI) to get_GAI(vital)
  (canopy_growth_LAI_based.cpp:220-227), so the dead area of the restart portion keeps counting
  as green area index.
- New growth whose GAI is smaller than what the untouchable portion still lacks to reach
  regrowth_GAI is merged into it (canopy_growth_portioned.cpp:482-492).
- A portion is kept only if its biomass is not zero within 1e-7 (canopy_growth_portioned.cpp:30).
- After the senescence period expires is_in_senescence() is false, so the self-shading LAI
  returns to max(GAI, previous) (canopy_growth_LAI_based.cpp:188-221).

Fruit trees (orchard.py): the self-shading LAI is the GAI and the peak LAI stays
0 (update_lai_self_shading `fruit`), the canopy is deleted at dormancy (drop) and a new one is
started at the restart (restart).

Clipping (clipping.py) removes portions from the slots in list order (a removed
portion leaves a zero slot, which no sum or walk sees) and replaces the canopy on regrowth (restart).

Not ported: leaf-duration ageing (V4), effete portions (an inactive period that keeps the canopy;
abscission is never read by the shipped parser, so there is no sloughing), the aliasing of the
first portion when restart_with gets a biomass below 1e-7 (initial_GAI/SLA,
canopy_growth_portioned.cpp:480-498). The orchard interception models (canopy_architecture) leave
the canopy's cover as it is (orchard_arch.py).
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from ...core import xscan

IS_ZERO_TOL = 0.0000001          # corn/math/compare.hpp:44,50
CLUMP_FACTOR = 0.75              # canopy_growth_LAI_based.cpp:244
COVER_XAI_MIN = 0.00001          # canopy_growth_LAI_based.cpp:262
GREEN_XAI_FLOOR = 0.00001        # canopy_growth_LAI_based.cpp:687-693
LAI_SS_INITIAL = 0.00001         # canopy_growth_LAI_based.cpp:153
HEIGHT_FACTOR = 1.15             # canopy_growth_LAI_based.cpp:300



def _rep(nt, **kw):
    """NamedTuple._replace without collections internals (traceable by torch.compile)."""
    return type(nt)(*[kw[f] if f in kw else getattr(nt, f) for f in type(nt)._fields])

def _cmax(a, b):
    return torch.where(a < b, b, a)


def _cmin(a, b):
    return torch.where(b < a, b, a)


class CanopyState(NamedTuple):
    exists: torch.Tensor        # bool, canopy object exists (from emergence)
    p_gai: torch.Tensor         # [..., P] vital portions: green area index
    p_dead: torch.Tensor        # [..., P] dead area index
    p_bm: torch.Tensor          # [..., P] biomass (current == produced: no removal ported)
    p_gbm: torch.Tensor         # [..., P] GAI-related biomass
    n: torch.Tensor             # long, number of vital portions
    u_gai: torch.Tensor         # untouchable portion
    u_dead: torch.Tensor
    u_bm: torch.Tensor
    u_gbm: torch.Tensor
    lai_ss: torch.Tensor        # LAI_for_self_shading_response
    peak_lai: torch.Tensor
    cover_total: torch.Tensor   # cover_total_fract
    sen_known: torch.Tensor     # bool, senescence period known (know_senescence)
    sen_rate: torch.Tensor      # senesce_LAI_unit_per_deg_day
    height: torch.Tensor        # plant height [m] (crop member, calc_plant_development)


def initial_state(shape, P=400, dtype=torch.float64, device=None) -> CanopyState:
    z = torch.zeros(shape, dtype=dtype, device=device)
    zp = torch.zeros(tuple(shape) + (P,), dtype=dtype, device=device)
    return CanopyState(torch.zeros(shape, dtype=torch.bool, device=device), zp, zp, zp, zp,
                       torch.zeros(shape, dtype=torch.long, device=device), z, z, z, z,
                       z + LAI_SS_INITIAL, z, z, torch.zeros(shape, dtype=torch.bool, device=device),
                       z, z)


def clumping(xai, area_factor):
    return CLUMP_FACTOR + (1.0 - CLUMP_FACTOR) * (1.0 - torch.exp(-area_factor * xai))


def cover_fract(xai, kc):
    """calc_cover_fract, global radiation, with clumping (canopy_growth_LAI_based.cpp:248-268)."""
    c = 1.0 - torch.exp(-kc * xai * clumping(xai, 0.35))
    return torch.where(xai > COVER_XAI_MIN, c, torch.zeros_like(c))


def _seqsum(x):
    """Sum over the portion axis in list order, as the accumulator does (0 + p1 + p2 + ...);
    cumsum is sequential on the CPU (on CUDA: torch.cumsum's own order, core.xscan)."""
    return xscan.cumsum_last(x)


def gai_vital(c: CanopyState):
    """get_GAI(include_vital): vital portions + the untouchable portion's LAI."""
    return _seqsum(c.p_gai) + (c.u_gai + c.u_dead)


gai_vital_effete = gai_vital          # no effete portions for annual crops


def lai(c: CanopyState):
    return (_seqsum(c.p_gai) + _seqsum(c.p_dead)) + (c.u_gai + c.u_dead)


def biomass_vital(c: CanopyState):
    return _seqsum(c.p_bm) + c.u_bm


def gai_related_biomass(c: CanopyState):
    return _seqsum(c.p_gbm) + c.u_gbm


def interception_green(c: CanopyState, kc):
    """calc_interception_global_green (canopy_growth_LAI_based.cpp:687-693); 0 without canopy."""
    g = gai_vital_effete(c)
    g = torch.where(g < GREEN_XAI_FLOOR, torch.full_like(g, GREEN_XAI_FLOOR), g)
    return torch.where(c.exists, cover_fract(g, kc), torch.zeros_like(g))


def senescence_reduction_factor(c: CanopyState):
    """get_senescence_reduction_factor; 1 when there is no canopy (crop_cropsyst.cpp:2642-2645)."""
    g = gai_vital_effete(c)
    zero_peak = torch.abs(0.0 - c.peak_lai) <= IS_ZERO_TOL
    f = torch.where(zero_peak, torch.zeros_like(g), g / torch.where(zero_peak, torch.ones_like(g), c.peak_lai))
    return torch.where(c.exists, f, torch.ones_like(f))


def restart(c: CanopyState, mask, initial_gai, sla, kc) -> CanopyState:
    """New canopy at emergence: provide_canopy + restart_with(initial_GAI/SLA, initial_GAI)
    (crop_cropsyst.cpp:779-832, 2303-2307; canopy_growth_LAI_based.cpp:283-309), then update()."""
    m = mask
    mp = m[..., None]
    b0 = (initial_gai.to(torch.float32) / sla.to(torch.float32)).to(initial_gai.dtype)   # float32 / float32
    z = torch.zeros_like(c.u_gai)
    zp = torch.zeros_like(c.p_gai)
    w = lambda new, old: torch.where(m, new, old)
    lai_ss = w(initial_gai + z, c.lai_ss)
    return CanopyState(
        exists=c.exists | m, p_gai=torch.where(mp, zp, c.p_gai), p_dead=torch.where(mp, zp, c.p_dead),
        p_bm=torch.where(mp, zp, c.p_bm), p_gbm=torch.where(mp, zp, c.p_gbm),
        n=torch.where(m, torch.zeros_like(c.n), c.n),
        u_gai=w(initial_gai + z, c.u_gai), u_dead=w(z, c.u_dead), u_bm=w(b0 + z, c.u_bm),
        u_gbm=w(b0 + z, c.u_gbm), lai_ss=lai_ss, peak_lai=w(z, c.peak_lai),
        cover_total=w(cover_fract(lai_ss, kc), c.cover_total),
        sen_known=c.sen_known & ~m, sen_rate=w(z, c.sen_rate), height=c.height)


def drop(c: CanopyState, mask) -> CanopyState:
    """Crop_complete::drop_leaves (crop_cropsyst.cpp:837-841): the canopy object is deleted at the
    start of dormancy when all of it senesces; the plant height (a crop member) stays."""
    m = mask
    z = initial_state(m.shape, c.p_gai.shape[-1], c.u_gai.dtype, c.u_gai.device)
    names = [f for f in CanopyState._fields if f != "height"]     # no tuple slicing (torch.compile)
    new = {f: torch.where(m.reshape(m.shape + (1,) * (getattr(c, f).dim() - m.dim())), getattr(z, f),
                          getattr(c, f)) for f in names}
    return CanopyState(height=c.height, **new)


def know_senescence(c: CanopyState, mask, dur_senescence) -> CanopyState:
    """V5 know_senescence (canopy_growth_LAI_based_V5.cpp:47-63)."""
    rate = gai_vital(c) / dur_senescence
    return _rep(c, sen_known=c.sen_known | mask, sen_rate=torch.where(mask, rate, c.sen_rate))


def process_day(c: CanopyState, kc, max_height, max_lai, overall_stress=1.0) -> CanopyState:
    """Canopy process_day -> update() -> update_fraction_of_canopy_cover, then
    calc_plant_development (crop_cropsyst.cpp:2725-2737; canopy_growth_LAI_based.cpp:239-242,
    292-302). The crop's overall_growth_stress is 1 here (cleared in clear_day)."""
    ct = cover_fract(c.lai_ss, kc)
    f = c.lai_ss / max_lai
    f = _cmin(f, torch.ones_like(f))
    dev = _cmax(c.height, _cmin(torch.pow(f, overall_stress) * max_height, HEIGHT_FACTOR * f * max_height))
    return _rep(c, cover_total=torch.where(c.exists, ct, c.cover_total),
                      height=torch.where(c.exists, dev, c.height))


def age_leaf(c: CanopyState, mask, gdd) -> CanopyState:
    """V5 age_leaf: senescence die-back to GAI - rate*GDD (canopy_growth_LAI_based_V5.cpp:30-46,
    canopy_growth_portioned.cpp:377-419), oldest portions first, then the untouchable portion."""
    on = mask & c.sen_known
    old = gai_vital_effete(c)
    target = old - c.sen_rate * gdd
    r0 = gai_vital_effete(c) - target
    l = c.p_gai
    living = l > 0.0
    lv = torch.where(living, l, torch.zeros_like(l))
    cum = xscan.cumsum(lv)
    r_i = r0[..., None] - (cum - lv)
    die = living & (r_i > l)
    alive = living & ~die
    alive_before = xscan.cumsum(alive.to(l.dtype)).to(torch.long) - alive.to(torch.long)   # counts: exact
    front = alive & (alive_before == 0)
    rr = torch.where(front, r_i, torch.zeros_like(r_i))
    new_g = l - rr
    p_gai = torch.where(die, torch.zeros_like(l), torch.where(alive, new_g, l))
    p_dead = torch.where(die, l, torch.where(alive, l - new_g, c.p_dead))
    r_after = torch.where(alive.any(-1), torch.zeros_like(r0), r0 - cum[..., -1])
    ru = r_after > 0.0
    u_die = ru & (r_after > c.u_gai)
    u_new = c.u_gai - r_after
    u_gai = torch.where(u_die, torch.zeros_like(u_new), torch.where(ru, u_new, c.u_gai))
    u_dead = torch.where(u_die, c.u_gai, torch.where(ru, c.u_gai - u_new, c.u_dead))
    onp = on[..., None]
    return _rep(c, p_gai=torch.where(onp, p_gai, c.p_gai), p_dead=torch.where(onp, p_dead, c.p_dead),
                      u_gai=torch.where(on, u_gai, c.u_gai), u_dead=torch.where(on, u_dead, c.u_dead))


def develop_leaf(c: CanopyState, mask, continue_growth, act_growth, gai_rel_growth, sla, slp,
                 regrowth_gai) -> CanopyState:
    """develop_leaf + take_portion (canopy_growth_LAI_based.cpp:311-330,
    canopy_growth_portioned.cpp:466-504)."""
    gbm_new = gai_related_biomass(c) + gai_rel_growth
    d = slp * gbm_new + 1.0
    daily = torch.where(continue_growth, gai_rel_growth * sla / (d * d), torch.zeros_like(gbm_new))
    valid = mask & ~(torch.abs(0.0 - act_growth) <= IS_ZERO_TOL)
    more = regrowth_gai - c.u_gai
    absorb = valid & (more > 0.0) & (daily < more)
    app = valid & ~absorb
    P = c.p_gai.shape[-1]
    slot = (torch.arange(P, device=c.n.device) == c.n[..., None]) & app[..., None]
    return _rep(c, 
        p_gai=torch.where(slot, daily[..., None], c.p_gai),
        p_dead=torch.where(slot, torch.zeros_like(c.p_dead), c.p_dead),
        p_bm=torch.where(slot, act_growth[..., None], c.p_bm),
        p_gbm=torch.where(slot, gai_rel_growth[..., None], c.p_gbm),
        n=c.n + app.to(torch.long),
        u_gai=torch.where(absorb, c.u_gai + daily, c.u_gai),
        u_bm=torch.where(absorb, c.u_bm + act_growth, c.u_bm),
        u_gbm=torch.where(absorb, c.u_gbm + gai_rel_growth, c.u_gbm))


def update_lai_self_shading(c: CanopyState, mask, culm_initiated, before_maturity,
                            sen_rel_remaining, fract_lai_mature, fruit=None) -> CanopyState:
    """update_LAI_for_self_shading_response for crops with leaf senescence
    (canopy_growth_LAI_based.cpp:175-223). `sen_rel_remaining` is the float32 senescence period
    relative thermal time remaining (phenology state). Fruit trees (`fruit` mask) take the GAI and keep
    the peak LAI at its value (only the reported peak follows, canopy_growth_LAI_based.cpp:
    222-228), so their senescence reduction factor stays 0."""
    g = gai_vital_effete(c)
    peak = c.peak_lai
    a = torch.where(before_maturity & c.sen_known,
                    peak - (peak - peak * fract_lai_mature) * sen_rel_remaining,
                    peak * fract_lai_mature)
    b = _cmax(g, c.lai_ss)
    peak_b = _cmax(b, peak)
    l = torch.where(culm_initiated, a, b)
    peak_n = torch.where(culm_initiated, peak, peak_b)
    if fruit is not None:
        l = torch.where(fruit, g, l)
        peak_n = torch.where(fruit, peak, peak_n)
    l = torch.where(l < 0.0, torch.zeros_like(l), l)
    l = _cmax(l, g)
    return _rep(c, lai_ss=torch.where(mask, l, c.lai_ss), peak_lai=torch.where(mask, peak_n, c.peak_lai))
