"""Management operations that act on soil nitrogen and residue in the VIC-CropSyst port:
the NRCS field operation response (tillage, sowing, fertilizer and harvest-residue operations),
the soil disturbance that follows tillage, and inorganic fertilizer.

Sources (CS = VIC-CropSyst-Package/CropSyst/source):
  operation dispatch          CS/perform_op.cpp:104-269 (each operation with an NRCS description
                              calls respond_to_field_operation: sowing :614-617, tillage :1046-1047,
                              inorganic N :1292-1293; a harvest schedules a residue tillage
                              operation for the next day when the harvest operation flattens
                              residue, :1176-1199)
  field operation response    CS/perform_op.cpp:1618-1700 (soil tillage effect, then residues and
                              SOM; "Seedbed finisher" twice)
  soil disturbance            CS/soil/disturbance.cpp:14-92, CS/soil/tillage_effect.cpp:52-90,
                              365-369, CS/soil.cpp:124-131 (processed every day)
  inorganic N                 CS/perform_op.cpp:1282-1355 (soil_n.fertilizer)
  automatic N                 CS/perform_op.cpp:1491-1613 (soil_n.auto_n_application)

The NRCS operation parameters come from soil_n.sdr_operations() (extracted from the reference).
Operation lists are host-side setup data (descriptions, amounts); the day functions take per-tile
masks and tensors.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from . import soil_n as sn
from . import organic_matter as om

OXIDATION_SAND_DEFAULT, OXIDATION_CLAY_DEFAULT = 1.5, 0.5     # management_param_struct.h:114-117
K_CLAY = 5.5                                                  # disturbance.cpp:80
K_TILLAGE = 1.0 / 40.0                                        # disturbance.cpp:81
DECAY_RATE = 0.02                                             # disturbance.cpp:89


class Disturbance(NamedTuple):
    exists: torch.Tensor          # bool, Soil::Disturbance created (first tillage)
    rating: torch.Tensor          # [..., L]
    factor: torch.Tensor          # [..., L] decomposition adjustment factor
    deepest: torch.Tensor         # long, deepest_layer_affected
    ox_sand: torch.Tensor         # float32 values
    ox_clay: torch.Tensor


def initial_disturbance(shape, L, dtype=torch.float64, device=None) -> Disturbance:
    z = torch.zeros(tuple(shape) + (L,), dtype=dtype, device=device)
    s = torch.zeros(tuple(shape), dtype=dtype, device=device)
    return Disturbance(torch.zeros(tuple(shape), dtype=torch.bool, device=device), z, z + 1.0,
                       torch.zeros(tuple(shape), dtype=torch.long, device=device),
                       s + sn.round_f32(torch.tensor(OXIDATION_SAND_DEFAULT, dtype=dtype)),
                       s + sn.round_f32(torch.tensor(OXIDATION_CLAY_DEFAULT, dtype=dtype)))


def add_rating(d: Disturbance, mask, rating, last, ox_sand=None, ox_clay=None) -> Disturbance:
    """Soil_tillage_effect::respond_to_field_operation tail (tillage_effect.cpp:365-369) and
    Disturbance::add_rating (disturbance.cpp:26-31)."""
    L = d.rating.shape[-1]
    idx = torch.arange(1, L + 1, device=d.rating.device)
    inl = mask[..., None] & (idx <= last[..., None])
    if torch.is_tensor(rating) and rating.dim():
        add = rating.to(d.rating.dtype)[..., None]
    else:
        add = float(rating)
    rt = torch.where(inl, d.rating + add, d.rating)
    deepest = torch.where(mask, torch.maximum(d.deepest, last), d.deepest)
    sand, clay = d.ox_sand, d.ox_clay
    if ox_sand is not None:
        sand = torch.where(mask, ox_sand + torch.zeros_like(sand), sand)
        clay = torch.where(mask, ox_clay + torch.zeros_like(clay), clay)
    return Disturbance(d.exists | mask, rt, d.factor, deepest, sand, clay)


def process_disturbance(d: Disturbance, available_water, clay_pct) -> Disturbance:
    """Disturbance::process (disturbance.cpp:33-65), run by Soil_components::process_day while a
    disturbance exists. available_water [..., L]: calc_available_water_fraction(layer, false)
    (FC/PWP based, VIC_soil.cpp:129-143)."""
    L = d.rating.shape[-1]
    idx = torch.arange(1, L + 1, device=d.rating.device)
    inl = d.exists[..., None] & (idx <= d.deepest[..., None])
    r = d.rating * (0.98 * torch.exp(-DECAY_RATE * available_water))
    r = torch.where(r < 1.0, torch.zeros_like(r), r)
    term1 = d.ox_clay[..., None] + (d.ox_sand - d.ox_clay)[..., None] * torch.exp(-K_CLAY * (clay_pct / 100.0))
    f = 1.0 + term1 * (1.0 - torch.exp(-K_TILLAGE * r))
    rating = torch.where(inl, r, d.rating)
    factor = torch.where(inl, f, d.factor)
    dl = (d.deepest - 1).clamp(min=0)
    fd = torch.gather(factor, -1, dl[..., None])[..., 0]
    last_off = d.exists & (d.deepest > 0) & om._approx(fd, torch.ones_like(fd), 0.0001)
    at = last_off[..., None] & (idx == d.deepest[..., None])
    factor = torch.where(at, torch.ones_like(factor), factor)
    rating = torch.where(at, torch.zeros_like(rating), rating)
    deepest = torch.where(last_off, d.deepest - 1, d.deepest)
    return Disturbance(d.exists, rating, factor, deepest, d.ox_sand, d.ox_clay)


class SoilState(NamedTuple):
    n: sn.SoilN
    residues: om.Residues
    sand: torch.Tensor            # [..., L] %
    clay: torch.Tensor            # [..., L] %
    disturbance: Disturbance


def field_operation(st: SoilState, mask, layers: sn.SoilLayers, op: dict, oxidation=None) -> SoilState:
    """Land_unit_simulation_VX::respond_to_field_operation (perform_op.cpp:1618-1700) for one
    NRCS operation (op = soil_n.sdr_operations()[name]) on the masked tiles. oxidation: the
    tillage operation's (sand, clay) oxidation effect, None for other operations."""
    reps = 2 if op.get("name") == "Seedbed finisher" else 1
    for _ in range(reps):
        dt = layers.thickness.dtype
        shape = mask.shape
        depth = torch.full(shape, op["depth_m"], dtype=dt, device=mask.device)     # float32 -> float64
        mixing = torch.full(shape, op["mixing"], dtype=dt, device=mask.device)
        # Soil_tillage_effect::respond_to_field_operation (tillage_effect.cpp:240-370)
        n2, sand, clay, last = sn.tillage_mixing(st.n, layers, st.sand, st.clay, depth, mixing)
        active = mask & (last > 0)
        n = sn.SoilN(torch.where(active[..., None], n2.no3, st.n.no3), torch.where(active[..., None], n2.nh4, st.n.nh4))
        sand = torch.where(active[..., None], sand, st.sand)
        clay = torch.where(active[..., None], clay, st.clay)
        ox_s = ox_c = None
        if oxidation is not None:
            ox_s = sn.round_f32(torch.tensor(oxidation[0], dtype=dt, device=mask.device))
            ox_c = sn.round_f32(torch.tensor(oxidation[1], dtype=dt, device=mask.device))
        dist = add_rating(st.disturbance, active, op["rating"], last, ox_s, ox_c)
        # residues (residue_V4_common.cpp:394-651); SOM is empty in the VIC build (organic_matter.py)
        res = om.tillage(st.residues, mask, layers.thickness, depth, torch.full(shape, op["effect"], device=mask.device),
                         torch.full(shape, op["flatten"][0], dtype=dt, device=mask.device),
                         torch.full(shape, op["burial"][0], dtype=dt, device=mask.device))
        st = SoilState(n, res, sand, clay, dist)
    return st


def field_operation_t(st: SoilState, mask, layers: sn.SoilLayers, sdr, ox_sand=None, ox_clay=None,
                      has_ox=None, residue_tillage=None) -> SoilState:
    """field_operation with per-tile operation parameters (the engine's form, no host data):
    sdr [..., 6] = depth [m], mixing fraction, disturbance rating, soil disturbance effect code,
    residue flatten ratio and burial ratio (index 0) of soil_n.sdr_operations(); ox_sand/ox_clay
    [...] the tillage oxidation effect, applied where has_ox. One response; the caller repeats
    it for "Seedbed finisher" (the second time without the oxidation effect, perform_op.cpp:
    1662-1680). residue_tillage: the residue response (default organic_matter.tillage; the
    nitrogen day on CUDA passes om_kernels.tillage_, which changes the day's lists in place)."""
    depth, mixing, rating = sdr[..., 0], sdr[..., 1], sdr[..., 2]
    effect = sdr[..., 3].round().to(torch.long)
    n2, sand, clay, last = sn.tillage_mixing(st.n, layers, st.sand, st.clay, depth, mixing)
    active = mask & (last > 0)
    n = sn.SoilN(torch.where(active[..., None], n2.no3, st.n.no3), torch.where(active[..., None], n2.nh4, st.n.nh4))
    sand = torch.where(active[..., None], sand, st.sand)
    clay = torch.where(active[..., None], clay, st.clay)
    dist = add_rating(st.disturbance, active, rating, last)
    if ox_sand is not None:
        m = active & has_ox
        dist = dist._replace(ox_sand=torch.where(m, ox_sand, dist.ox_sand), ox_clay=torch.where(m, ox_clay, dist.ox_clay))
    res = (residue_tillage or om.tillage)(st.residues, mask, layers.thickness, depth, effect, sdr[..., 4],
                                          sdr[..., 5])
    return SoilState(n, res, sand, clay, dist)


def _f32(x):
    import numpy as np
    return float(np.float32(x))


def auto_n_total(soil_n_kg_m2, target_kg_ha, estimated_mineralization_kg_ha, efficiency_pct, mode="balance_based",
                 critical_kg_ha=0.0):
    """N_application_soil_observation_mode_class::determine_total_automatic_N_application
    (perform_op.cpp:1491-1520) at the soil N sampling, host-side for one tile: soil_n_kg_m2 is the
    NO3-N plus NH4-N to the sampling depth (sum_NO3_to_depth + sum_NH4_to_depth, molecular sums
    converted, chemicals_profile.cpp:715-731; NaN when NH4 is NaN, then nothing is applied).
    The management values are float32 (management_param_class.h)."""
    t, e, eff = _f32(target_kg_ha), _f32(estimated_mineralization_kg_ha), _f32(efficiency_pct)
    if mode == "balance_based":
        a = t / 10000.0 - soil_n_kg_m2 - e / 10000.0
    elif mode == "local_calibration":
        c = _f32(critical_kg_ha) / 10000.0
        a = 0.0 if soil_n_kg_m2 > c else c - soil_n_kg_m2
    else:
        a = 0.0
    a = a / (eff / 100.0)
    return a if not (a < 0.0) else 0.0               # must_be_0_or_greater: NaN stays NaN


def auto_n_split(total_kg_m2, percent, no3_pct, nh4_pct):
    """perform_automatic_N_application (perform_op.cpp:1522-1550): the split application's NO3-N
    and NH4-N, stored as float32 (management_param_class.h:314-315); None when total <= 1e-8."""
    if not (total_kg_m2 > 0.00000001):
        return None
    no3 = _f32(total_kg_m2 * (int(percent) / 100.0) * (_f32(no3_pct) / 100.0))
    nh4 = _f32(total_kg_m2 * (int(percent) / 100.0) * (_f32(nh4_pct) / 100.0))
    return no3, nh4


def sampled_soil_n(s: sn.SoilN, thickness, sampling_depth):
    """sum_NO3_to_depth + sum_NH4_to_depth (chemicals_profile.cpp:715-731,
    chemical_mass_profile.cpp:43-53): molecular sums over layers 1..layer at depth (all layers for
    a 0 depth), converted to N; batched, [...] -> [...]."""
    L = thickness.shape[-1]
    last = torch.where(sn._is_zero(sampling_depth), torch.full_like(sampling_depth, L, dtype=torch.long),
                       sn.layer_at_depth(thickness, sampling_depth))
    inl = torch.arange(1, L + 1, device=thickness.device) <= last[..., None]
    z = torch.zeros_like(s.no3)
    m3 = torch.zeros_like(sampling_depth)
    m4 = torch.zeros_like(sampling_depth)
    for l in range(L):                                    # sequential, as the layer loop
        m3 = m3 + torch.where(inl[..., l], s.no3[..., l], z[..., l])
        m4 = m4 + torch.where(inl[..., l], s.nh4[..., l], z[..., l])
    return m3 * sn.NO3_TO_N + m4 * sn.NH4_TO_N


AUTO_DISABLED, AUTO_BALANCE, AUTO_LOCAL = 0, 1, 2


def auto_n_total_t(soil_n_kg_m2, target_kg_ha, estimated_mineralization_kg_ha, efficiency_pct, mode=None,
                   critical_kg_ha=None):
    """Tensor form of auto_n_total (management values float32). mode: AUTO_* codes per tile
    (default balance based)."""
    f = sn.round_f32
    a = f(target_kg_ha) / 10000.0 - soil_n_kg_m2 - f(estimated_mineralization_kg_ha) / 10000.0
    if mode is not None:
        c = f(critical_kg_ha) / 10000.0
        loc = torch.where(soil_n_kg_m2 > c, torch.zeros_like(a), c - soil_n_kg_m2)
        a = torch.where(mode == AUTO_BALANCE, a, torch.where(mode == AUTO_LOCAL, loc, torch.zeros_like(a)))
    a = a / (f(efficiency_pct) / 100.0)
    return torch.where(a < 0.0, torch.zeros_like(a), a)


def auto_n_split_t(total_kg_m2, percent, no3_pct, nh4_pct):
    """Tensor form of auto_n_split: (apply mask, NO3-N, NH4-N), amounts rounded to float32."""
    f = sn.round_f32
    go = total_kg_m2 > 0.00000001
    no3 = f(total_kg_m2 * (percent / 100.0) * (f(no3_pct) / 100.0))
    nh4 = f(total_kg_m2 * (percent / 100.0) * (f(nh4_pct) / 100.0))
    return go, no3, nh4
