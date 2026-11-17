"""E1-E3 of the VIC-CropSyst port (docs/connectors/cropsyst.md).

E1  root water uptake: RAF, root/top conductances, leaf water potential, per-layer uptake,
    water stress ratio   CropSyst-5 CS/crop/transpiration.cpp:41-389, crop_common.cpp:124-200
E2  soil water potential per VIC layer (Campbell, air entry and b from the soil file)
                         VIC_CropSyst/agronomic/VIC_soil.cpp:81-127, 239-266, 346-374
E3  uptake into the VIC layers and the layer evap the VIC water balance sees
                         VIC_soil.cpp:225-236, canopy_evap.c:1008-1009, runoff.c:370-374

The equations are re-derived from the shipped C++ (Academic Free License; no code text is copied).
Everything is elementwise or a reduction over the last (layer) axis, so any leading batch shape
works, e.g. [cell, crop] for scalars and [cell, crop, layer] for layer fields. There are no Python
loops over cells or layers, no host syncs and no data-dependent control flow: every C++ branch is a
`torch.where`, so the functions trace cleanly under torch.compile and CUDA-graph capture.

Branch semantics follow the C++ exactly, including its comparison rules for NaN: `std::max(a, b)`
is `a < b ? b : a` and `std::min(a, b)` is `b < a ? b : a` (corn/math/compare.hpp:25-38); they are
spelled that way below rather than with torch.clamp, whose NaN handling differs.

Reference facts this module relies on (all checked against the compiled reference through the oracle
tests/oracle/ref_cropsyst_uptake.cpp):
- CropSyst layer n is VIC layer n-1; the VIC soil serves one sublayer per VIC layer (VIC_soil.cpp:83).
- Salinity is off in VIC-CropSyst (the VIC soil provides no salt profile), so the salt stress factor
  is 1 and the osmotic potential is 0 (transpiration.cpp:215-224, 262-264, 308-310).
- RAF exponent `dry_soil_root_activity_coef` defaults to 8.0 (cs_scenario.cpp:268); the CRB
  scenarios do not override it ([transpiration] section of .CropSyst_scenario, cs_scenario.cpp:512-516).
  The wet-side RAF is commented out in the reference (transpiration.cpp:203-213) and is not ported.
- Crop parameters LWP_stomatal_closure, LWP_wilt_point, normal_max_water_uptake_mm and the RAF
  coefficient are float32 in the reference (crop_param_struct.h:293-295, cs_scenario.h:206-207);
  `transpiration_params` rounds them through float32 for parity.
- Uptake is extracted from CropSyst layer 2 down (crop_common.cpp:193, start_layer 2): the VIC top
  layer never receives crop uptake. Roots start in CropSyst layer 2 as well
  (rooting_layer = 2, crop_root.cpp:49, :174), so layer-1 uptake is zero in a coupled run; for
  arbitrary root fractions it would count in the reported transpiration but leave VIC unchanged,
  exactly as in the reference.
- `leaf_water_pot` and the VIC top layer's `transpiration_daily` are not rewritten on every path of
  the reference; they carry over from the previous day, so the caller passes them in.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from .readers import IniTable

# Named constants of the reference (values in the cited lines).
SWP_FOR_MAX_PLANT_HYDRAUL_COND = 0.0     # transpiration.cpp:137
ROOT_COND_SHARE = 0.65                   # transpiration.cpp:176
TOP_COND_SHARE = 0.35                    # transpiration.cpp:177
ROOT_COND_ADJ_MIN = 0.00001              # transpiration.cpp:234
MAX_UPTAKE_ZERO_TOL = 0.00000007         # transpiration.cpp:282 (CORN::is_zero tolerance)
WC_MIN = 0.0000001                       # VIC_soil.cpp:122
PHC_ADJ_BASE = 0.5                       # transpiration.cpp:369
SEC_PER_DAY = 86400.0                    # canopy_evap.c:1009
UPTAKE_START_LAYER = 2                   # crop_common.cpp:193 (CropSyst 1-based layer)
DEFAULT_DRY_SOIL_ROOT_ACTIVITY_COEF = 8.0    # cs_scenario.cpp:268
# crop_param_class.cpp:60-63, used when a .crp file lacks the key
DEFAULT_MAX_WATER_UPTAKE_MM = 10.0
DEFAULT_LWP_WILT_POINT = -1600.0
DEFAULT_LWP_STOMATAL_CLOSURE = -700.0


def _cmax(a, b):
    """std::max(a, b) = (a < b) ? b : a."""
    return torch.where(a < b, b, a)


def _cmin(a, b):
    """std::min(a, b) = (b < a) ? b : a."""
    return torch.where(b < a, b, a)


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(x.dtype)


# ----------------------------------------------------------------------------- parameters

class TranspirationParams(NamedTuple):
    """Per-crop E1 parameters, float32-rounded like the reference, any batch shape."""
    max_water_uptake_mm: torch.Tensor     # normal_max_water_uptake_mm  [mm/day]
    lwp_stomatal_closure: torch.Tensor    # LWP_stomatal_closure        [J/kg]
    lwp_wilt_point: torch.Tensor          # LWP_wilt_point              [J/kg]
    dry_soil_root_activity_coef: torch.Tensor


def transpiration_params(crops: IniTable,
                         dry_soil_root_activity_coef: float = DEFAULT_DRY_SOIL_ROOT_ACTIVITY_COEF
                         ) -> TranspirationParams:
    """[n_crop] E1 parameters from a crop table (readers.crop_table), with the reference defaults
    for missing keys and the float32 storage of the reference."""
    def get(key, default):
        t = crops.params.get(f"transpiration.{key}")
        if t is None:
            return torch.full((crops.n,), default, dtype=torch.float64)
        return torch.where(torch.isnan(t), torch.full_like(t, default), t)
    mwu = get("max_water_uptake", DEFAULT_MAX_WATER_UPTAKE_MM)
    clo = get("stomatal_closure_leaf_water_pot", DEFAULT_LWP_STOMATAL_CLOSURE)
    wil = get("wilt_leaf_water_pot", DEFAULT_LWP_WILT_POINT)
    raf = torch.full_like(mwu, float(dry_soil_root_activity_coef))
    return TranspirationParams(_f32(mwu), _f32(clo), _f32(wil), _f32(raf))


# ----------------------------------------------------------------------------- E2

def soil_water_potential(moist_mm: torch.Tensor, ice_mm: torch.Tensor, depth_m: torch.Tensor,
                         max_moist_mm: torch.Tensor, air_entry_pot: torch.Tensor,
                         campbell_b: torch.Tensor) -> torch.Tensor:
    """E2: matric potential [J/kg] of each VIC layer, VIC::Soil_hydrology::get_water_potential.

    All inputs broadcast over [..., layer]. `air_entry_pot` and `campbell_b` are soil_con.VCS.AE and
    VCS.b_campbell (Saxton, read_soilparam.c:746-748; E14).

    VIC_soil.cpp:84    water content     wc  = moist / (1000 depth)
    VIC_soil.cpp:97    ice content       ice = ice / (1000 depth)
    VIC_soil.cpp:373   saturation        sat = (max_moist / 1000) / depth
    VIC_soil.cpp:239-266 air entry with ice: the ice-adjusted saturation the reference asks for is
                       the plain saturation (current ice is ignored, :371), so the Campbell factor
                       is (sat/sat)^-b = 1 and air entry is AE, or 0 when ice > sat (:256-257).
    VIC_soil.cpp:119-125 psi = ae (wc/(sat - ice))^-b when wc < sat, wc > 1e-7 and sat - ice > 0,
                       else ae.
    """
    depth_mm = depth_m * 1000.0
    wc = moist_mm / depth_mm
    ice = ice_mm / depth_mm
    sat = (max_moist_mm / 1000.0) / depth_m
    ae_ice = air_entry_pot * torch.pow(sat / sat, -campbell_b)
    ae = torch.where(ice > 0.0, torch.where(ice > sat, torch.zeros_like(ae_ice), ae_ice),
                     air_entry_pot)
    sat_liq = sat - ice
    ok = (wc < sat) & (wc > WC_MIN) & (sat_liq > 0.0)
    safe = torch.where(ok, sat_liq, torch.ones_like(sat_liq))
    return torch.where(ok, ae * torch.pow(wc / safe, -campbell_b), ae)


# ----------------------------------------------------------------------------- E1

class UptakeResult(NamedTuple):
    act_transp_m: torch.Tensor           # [...]   uptake + intercepted water used [m/day]
    uptake_act_m: torch.Tensor           # [...]   sum of layer uptake (incl. layer 1)
    uptake_m: torch.Tensor               # [..., L] per layer, CropSyst layer l+1 [m/day]
    leaf_water_pot: torch.Tensor         # [...]   [J/kg]
    interception_m: torch.Tensor         # [...]   canopy interception store left over [m]
    water_stress: torch.Tensor           # [...]   act/pot, crop_common.cpp:160-164


def root_uptake(pot_transp_m: torch.Tensor, fract_green_canopy: torch.Tensor,
                interception_m: torch.Tensor, fract_root_length: torch.Tensor,
                soil_water_pot: torch.Tensor, water_pot_at_fc: torch.Tensor,
                params: TranspirationParams, leaf_water_pot_prev: torch.Tensor,
                co2_transp_factor: Optional[torch.Tensor] = None,
                active: Optional[torch.Tensor] = None) -> UptakeResult:
    """E1: one day of crop water uptake, Crop_transpiration_2::calc_transpiration and uptake_actual.

    Scalars broadcast over [...] (e.g. [cell, crop]); `fract_root_length`, `soil_water_pot` (E2) and
    `water_pot_at_fc` (soil_con.VCS.water_pot_at_FC, -33 J/kg, read_soilparam.c:779) over [..., L].
    `pot_transp_m` is the N-limited, CO2-adjusted potential transpiration [m] (E4),
    `fract_green_canopy` yesterday's green-canopy solar interception fraction, `interception_m` the
    crop's stored intercepted precipitation [m]. `active` False reproduces a crop with no
    transpiration object (crop_common.cpp:150-156): zero transpiration, nothing else changes.

    transpiration.cpp:65-78   intercepted water serves the demand first; the rest is Tp [mm]
    :81-86                    uptake capacity Up = max_uptake * fract_green, CO2-scaled
    :91-94                    demand  M = min(Tp, Up * co2)
    :96-104                   uptake only when M > 0
    :170-177                  K_plant = Up / (0 - LWP_close); K_root = K/0.65; K_top = K/0.35
    :194-214                  RAF_l = 0 if psi_l < LWP_wilt, else 1 - ((psi_l-fc)/(wilt-fc))^c when
                              psi_l <= fc, else 1; clamped to [0, 1]
    :225-229                  a_l = RAF_l f_l; A = sum a_l; K_root,l = K_root a_l
    :234                      everything below only when A > 1e-5
    :236-244                  K_top,l = K_top a_l / A; K_l = K_root,l K_top,l / (K_root,l + K_top,l)
                              when a_l > 0 and the sum > 0, else 0
    :246-258                  K_root *= A; K = K_root K_top / (K_root + K_top) (0 if sum <= 0)
    :260-268                  psi_avg = sum psi_l a_l / A
    :269                      below only when K > 0
    :352-382                  leaf potential: psi_leaf = psi_avg - M/K; under stress (psi_leaf <
                              LWP_close) K is scaled by 0.5 + 0.5 (psi_leaf-wilt)/(close-wilt) and
                              psi_leaf solved again; then psi_leaf = max(psi_leaf, wilt)
    :279-295                  ratio = (M (psi_leaf - wilt)/(close - wilt)) / M when psi_leaf < close
                              and |M| > 7e-8, else 1
    :305-321                  u_l = K_l (psi_l - psi_leaf) ratio [mm] / 1000 (may be negative)
    :105-106                  act = sum u_l + intercepted water used
    """
    dt = soil_water_pot.dtype
    pot = pot_transp_m
    icpt = interception_m
    co2 = torch.ones_like(pot) if co2_transp_factor is None else co2_transp_factor
    mwu, closure, wilt, raf_c = params
    closure_l = closure.unsqueeze(-1)
    wilt_l = wilt.unsqueeze(-1)

    # transpiration.cpp:65-78
    pot_gt = pot > icpt
    icpt_used = torch.where(pot_gt, icpt, pot)
    icpt_left = torch.where(pot_gt, torch.zeros_like(icpt),
                            torch.where(pot > icpt, torch.zeros_like(icpt), icpt - pot))
    tp_m = torch.where(pot_gt, pot - icpt, torch.zeros_like(pot))
    tp_mm = tp_m * 1000.0
    # :81-94
    up_mm = mwu * fract_green_canopy
    up_co2 = up_mm * co2
    max_mm = _cmin(tp_mm, up_co2)
    do_uptake = max_mm > 0

    # :170-177
    k_plant0 = up_mm / (SWP_FOR_MAX_PLANT_HYDRAUL_COND - closure)
    k_root = k_plant0 / ROOT_COND_SHARE
    k_top = k_plant0 / TOP_COND_SHARE
    # :194-214
    psi = soil_water_pot
    fc = water_pot_at_fc
    raf_dry = 1.0 - torch.pow((psi - fc) / (wilt_l - fc), raf_c.unsqueeze(-1))
    raf_dry = torch.where(psi < wilt_l, torch.zeros_like(raf_dry), raf_dry)
    raf = torch.where(psi <= fc, raf_dry, torch.ones_like(raf_dry))
    raf = _cmin(_cmax(raf, torch.zeros_like(raf)), torch.ones_like(raf))
    # :225-229 (salinity factor 1)
    adj = raf * fract_root_length
    a_sum = adj.sum(-1)
    k_root_l = k_root.unsqueeze(-1) * adj
    have_roots = a_sum > ROOT_COND_ADJ_MIN
    a_safe = torch.where(have_roots, a_sum, torch.ones_like(a_sum)).unsqueeze(-1)
    # :236-244
    k_top_l = k_top.unsqueeze(-1) * adj / a_safe
    k_sum_l = k_root_l + k_top_l
    lay_ok = (adj > 0) & (k_sum_l > 0.0)
    k_l = torch.where(lay_ok, k_root_l * k_top_l / torch.where(lay_ok, k_sum_l, torch.ones_like(k_sum_l)),
                      torch.zeros_like(k_sum_l))
    # :246-258
    k_root_tot = k_root * torch.where(have_roots, a_sum, torch.ones_like(a_sum))
    k_rt = k_root_tot + k_top
    k_ok = k_rt > 0.0
    k_plant = torch.where(k_ok, (k_root_tot * k_top) / torch.where(k_ok, k_rt, torch.ones_like(k_rt)),
                          torch.zeros_like(k_rt))
    # :260-268 (osmotic potential 0)
    psi_avg = ((psi + 0.0) * adj / a_safe).sum(-1)
    # :269, :352-382
    solve = have_roots & (k_plant > 0.0)
    k_safe = torch.where(solve, k_plant, torch.ones_like(k_plant))
    lwp = psi_avg - max_mm / k_safe
    stressed = lwp < closure
    phc_adj = PHC_ADJ_BASE + (0.5 * (lwp - wilt)) / (closure - wilt)
    k_adj = phc_adj * k_safe
    lwp_s = (k_adj * psi_avg * (closure - wilt) + wilt * max_mm) / (k_adj * (closure - wilt) + max_mm)
    lwp = torch.where(stressed, lwp_s, lwp)
    lwp = _cmax(lwp, wilt)
    # :279-295
    ratio_stress = (lwp < closure) & ~((max_mm.abs()) <= MAX_UPTAKE_ZERO_TOL)
    attainable = max_mm * (lwp - wilt) / (closure - wilt)
    ratio = torch.where(ratio_stress, attainable / max_mm, torch.ones_like(attainable))
    # :305-321
    u_mm = k_l * (psi + 0.0 - lwp.unsqueeze(-1)) * ratio.unsqueeze(-1)
    u_m = u_mm / 1000.0

    run = do_uptake & solve
    if active is not None:
        run = run & active
    uptake_m = torch.where(run.unsqueeze(-1), u_m, torch.zeros_like(u_m))
    uptake_act = torch.where(run, u_m.sum(-1), torch.zeros_like(pot))
    lwp_out = torch.where(run, lwp, leaf_water_pot_prev.to(dt))
    act = uptake_act + icpt_used
    icpt_out = icpt_left
    if active is not None:
        act = torch.where(active, act, torch.zeros_like(act))
        icpt_out = torch.where(active, icpt_left, icpt)
    # crop_common.cpp:157-164
    act0 = _cmax(act, torch.zeros_like(act))
    stress = torch.where(pot > 0.0, act0 / torch.where(pot > 0.0, pot, torch.ones_like(pot)),
                         torch.ones_like(pot))
    return UptakeResult(act, uptake_act, uptake_m, lwp_out, icpt_out, stress)


# ----------------------------------------------------------------------------- E3

class LayerEvap(NamedTuple):
    transp_daily_mm: torch.Tensor        # [..., L] VIC layer.VCS.transpiration_daily [mm/day]
    extracted_m: torch.Tensor            # [...]   total moved to VIC (layers >= start) [m]
    layer_evap_mm: torch.Tensor          # [..., L] layer.evap of the crop tile for the step [mm]
    top_evap_mm: torch.Tensor            # [...]   evap the top layer loses in runoff() [mm]


def uptake_to_vic(uptake_m: torch.Tensor, transp_daily_prev_mm: torch.Tensor,
                  evap_bare_soil_mm: torch.Tensor, dt_sec: float = SEC_PER_DAY,
                  start_layer: int = UPTAKE_START_LAYER) -> LayerEvap:
    """E3: Soil_hydrology::extract_water(uptake, start_layer) and the crop-tile layer evap.

    VIC_soil.cpp:229-233  layers from CropSyst `start_layer` down get transpiration_daily = 1000 u;
                          earlier layers keep yesterday's value; total = sum(td / 1000)
    canopy_evap.c:1008    layer evap = transpiration_daily * dt / 86400
    runoff.c:370-374      top-layer evap in runoff() = evap_bare_soil + layer evap
    """
    L = uptake_m.shape[-1]
    idx = torch.arange(L, device=uptake_m.device)
    written = idx >= (start_layer - 1)
    td = torch.where(written, uptake_m * 1000.0, transp_daily_prev_mm)
    extracted = torch.where(written, td / 1000.0, torch.zeros_like(td)).sum(-1)
    evap = td * dt_sec / SEC_PER_DAY
    top = evap_bare_soil_mm + evap[..., 0]
    return LayerEvap(td, extracted, evap, top)
