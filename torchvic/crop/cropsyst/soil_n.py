"""N2 of the VIC-CropSyst port (docs/connectors/cropsyst.md): soil
mineral nitrogen (NO3 and NH4 per layer), the soil abiotic environment the N and organic-matter
processes use, nitrification, denitrification, crop uptake bookkeeping, fertilizer and the
tillage mixing of NO3, NH4 and texture.

Sources (CS = VIC-CropSyst-Package/CropSyst/source, the shipped build):
  NO3/NH4 pools          CS/soil/chemical_profile.cpp:29-111 (mass in molecular kg/m2),
                         CS/soil/chemical_mass_profile.cpp:28-118, chemical_mass_profile.h:40-48
                         (layer based mass, dec_layer = min(request, mass))
  conversions            CS/soil/nitrogen_I.h:7-11
  initial profile        CS/soil/chemicals_profile.cpp:127-239 with the VIC soil
                         VIC_CropSyst/agronomic/VIC_soil.h:411-418 (distribute_amount is a no-op)
  abiotic environment    CS/soil/abiotic_environment.cpp:11-116, abiotic_environment.h:25-35
  denitrification        CS/soil/nitrogen_profile.cpp:282-617 (NO3_with_CO2_Profile)
  nitrification          CS/soil/nitrogen_profile.cpp:157-201, 619-718 (NH4_Profile_with_pH)
  transformations order  CS/soil/chemicals_profile.cpp:358-456 (NO3 first, then NH4, hourly)
  uptake                 CS/soil/chemical_profile.cpp:314-336, chemicals_profile.cpp:701-713
  fertilizer             CS/perform_op.cpp:1282-1355, 1552-1580; CS/inorganic_NH4.cpp:6-175;
                         chemical_profile.cpp:408-441; nitrogen_profile.cpp:720-731
  tillage mixing         CS/soil/tillage_effect.cpp:240-370
  VIC soil hydrology     VIC_CropSyst/agronomic/VIC_soil.cpp:81-224, 346-374;
                         CS/soil/hydrology_A.cpp:263-271 (hourly water content)

Layers: [..., L], index l is CropSyst/VIC-CropSyst layer l+1 (one CropSyst layer per VIC layer,
VIC_soil.cpp:83). All masses are kg/m2; NO3 and NH4 are stored as molecular masses like the
reference (NO3 = 62/14 N, NH4 = 18/14 N).

Reference facts of the shipped build that this module reproduces:
- The scenario's initial NO3/NH4 horizons never reach the soil: VIC::Soil_components::
  distribute_amount returns 0 without writing (VIC_soil.h:411-418). The NO3 layer array is cleared
  first (chemicals_profile.cpp:129) so NO3 starts at 0; the NH4 array is not cleared (:181) and
  holds stack garbage. In the original CRB runs that garbage is inert: a dump build that clears it
  reproduces the original crop output byte for byte in all three replayed CRB cases,
  while the plain binary run from another directory does not (the garbage is path dependent).
  The port therefore starts both pools at 0.
- The VIC soil reports pH 7 in every layer (VIC_soil.h:423), so the nitrification pH function is
  (7 - 3.5)/(6.5 - 3.5) = 7/6, not clipped (nitrogen_profile.cpp:195-199).
- The infiltration model of the CRB scenarios is cascade_hourly, so denitrification uses the
  hourly moisture function (nitrogen_profile.cpp:474, 489-492); the daily cascade branch
  (:284-313) is ported for completeness (`cascade_daily=True`).
- The VIC soil has no hourly water content (water_content_hourly_available is never set), so the
  water-filled porosity of every hour is liquid/saturation (hydrology_A.cpp:263-271), while the
  daily (hour 24) value includes VIC ice (VIC_soil.cpp:205-222). The soil temperature is the VIC
  layer temperature for every hour (VIC_soil.h:345-346).
- NaN behaves as in the reference: comparisons with NaN are false, std::min/std::max are
  `b < a ? b : a` / `a < b ? b : a` (corn/math/compare.hpp), the CORN_* macros differ
  (compare.h:7-9). NH4 turns NaN wherever residue N mineralization is NaN (organic_matter.py);
  the arithmetic below propagates it exactly like the C++ (e.g. a NaN layer is skipped by crop
  uptake because `NO3 + NH4 > 0` is false).
- Nitrification uses the NH4 mass of the start of the day in every hour (nitrogen_profile.cpp:
  645, 659-663) and 1% of the oxidized N is lost as N2O (:671); denitrification respiration comes
  from the organic-matter CO2 loss of the day (residues and SOM, OM_pools_common.cpp:568).

Everything is batched over the leading shape; the hourly loops are static 24-step loops with
masks (no host syncs, no data-dependent control flow).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from ...core import xscan

from .growth import round_f32

NO3_TO_N = 14.0 / 62.0               # nitrogen_I.h:10
N_TO_NO3 = 62.0 / 14.0               # nitrogen_I.h:11
NH4_TO_N = 14.0 / 18.0               # nitrogen_I.h:7
N_TO_NH4 = 18.0 / 14.0               # nitrogen_I.h:8
SECONDS_PER_DAY = 86400
TRANSFORMATION_STEP = 3600           # land_unit_sim.cpp:1838-1847 (always hourly)
IS_ZERO_TOL = 0.0000001              # corn/math/compare.hpp:50
# abiotic_environment.cpp:11-13, 46-48 (all three temperature-function modes are the same)
T_MIN, T_OPT, T_MAX = -5.0, 35.0, 50.0
MIN_TEMPERATURE_FUNCTION = 0.0000001  # abiotic_environment.cpp:28
WFP_MIN, WFP_LOW, WFP_HIGH = 0.1, 0.5, 0.7
OM_MOISTURE_AT_SATURATION = 0.6      # OM_const.h:4
NITRIFICATION_MOISTURE_AT_SATURATION = 0.5   # nitrogen_profile.cpp:653
MAX_RESIDUE_HOLD_M3_KG = 0.003       # residue_const.h:6
# nitrification, nitrogen_profile.cpp:637-638, 669-671
NITRIFICATION_CONSTANT_HOURLY = 0.00208
NITRIFICATION_NO3_NH4_RATIO = 8.0
NITRIFICATION_N2O_FRACTION = 0.01
PH_MIN, PH_MAX = 3.5, 6.5            # chemicals_profile.cpp:216-217
VIC_SOIL_PH = 7.0                    # VIC_soil.h:423
# denitrification, nitrogen_profile.cpp:420-453
DENITRIFICATION_POT_MAX = 24.0       # mg N / kg soil / hour
NITRATE_MIN = 0.00000001             # :434
DENITRIFICATION_MIN_HOURLY = 0.0000001   # :516
EXHAUSTED_TOL = 0.0000000001         # :522-523
UPPER_WFP_THRESHOLD = 0.9            # :335
# Denitrification_parameters defaults (soil/soil_N_param.h:15-19); the CRB scenarios do not set them
DENITRIFICATION_SCALE = 0.0
DENITRIFICATION_RATE_MAX = 1.0
DENITRIFICATION_ALPHA = 1.0


def _cmax(a, b):
    """std::max(a, b) = (a < b) ? b : a  (corn/math/compare.hpp:25-30)."""
    return torch.where(a < b, b, a)


def _cmin(a, b):
    """std::min(a, b) = (b < a) ? b : a."""
    return torch.where(b < a, b, a)


def _between(v, lo, hi):
    """CORN::must_be_between = std::min(std::max(v, lo), hi) (compare.hpp:37-38); NaN stays NaN."""
    return _cmin(_cmax(v, lo), hi)


def _macro_between(v, lo, hi):
    """CORN_must_be_between macro: CORN_min(CORN_max(v, lo), hi) with CORN_max(x, y) = x > y ? x : y
    and CORN_min(x, y) = x < y ? x : y (corn/math/compare.h:7-9); NaN becomes `hi`."""
    m = torch.where(v > lo, v, lo + torch.zeros_like(v))
    return torch.where(m < hi, m, hi + torch.zeros_like(m))


def _is_zero(x, tol=IS_ZERO_TOL):
    """CORN::is_zero (compare.hpp:41-52): |0 - x| <= tol; NaN is never zero."""
    d = 0.0 - x
    d = torch.where(d < 0.0, -d, d)
    return d <= tol


class SoilN(NamedTuple):
    """Mineral N pools, molecular kg/m2, [..., L]."""
    no3: torch.Tensor
    nh4: torch.Tensor


class SoilLayers(NamedTuple):
    """Static per-layer soil properties, [..., L] (from the VIC soil, VIC_soil.cpp:346-374)."""
    thickness: torch.Tensor          # m (VIC depth)
    bulk_density: torch.Tensor       # g/cm3 = VIC bulk_density [kg/m3] * 0.001 (VIC_land_unit_simulation.cpp:210,
                                     # structure_common.cpp:59-63, kg_m3_to_g_cm3 multiplies by 0.001)
    sat: torch.Tensor                # max_moist / depth / 1000 (volumetric)
    fc: torch.Tensor                 # Wcr / depth / 1000
    pwp: torch.Tensor                # Wpwp / depth / 1000
    ph_function: torch.Tensor        # (pH - 3.5)/(6.5 - 3.5), pH = 7 from the VIC soil


def initial_state(shape, L, dtype=torch.float64, device=None) -> SoilN:
    """Chemicals_profile::enable_nitrogen on a VIC soil: both pools start at 0 (see module notes)."""
    z = torch.zeros(tuple(shape) + (L,), dtype=dtype, device=device)
    return SoilN(z, z.clone())


def soil_layers(thickness, bulk_density_kg_m3, max_moist_mm, wcr_mm, wpwp_mm, ph=VIC_SOIL_PH) -> SoilLayers:
    """Per-layer properties from the VIC soil parameters ([..., L] tensors)."""
    t = thickness
    ph_f = (torch.as_tensor(ph, dtype=t.dtype, device=t.device) - PH_MIN) / (PH_MAX - PH_MIN) + torch.zeros_like(t)
    return SoilLayers(t, bulk_density_kg_m3 * 0.001, (max_moist_mm / 1000.0) / t, (wcr_mm / 1000.0) / t,
                      (wpwp_mm / 1000.0) / t, ph_f)


def soil_mass(layers: SoilLayers):
    """Soil_interface::get_mass_kg_m2 (soil_I.cpp:5-15): bulk density [kg/m3] (= g/cm3 * 1000.0,
    structure_common.cpp:97-99) times thickness."""
    return (layers.bulk_density * 1000.0) * layers.thickness


def depths(thickness):
    """Soil_layers::set_depth_by_thickness (VIC_soil.cpp:295-306): running sum of thicknesses."""
    return xscan.cumsum(thickness)


def layer_at_depth(thickness, depth):
    """Soil_layers::get_layer_at_depth (CS/soil/layers.cpp:6-14), 1-based: the first layer whose
    running bottom depth is not below `depth`, at most the layer count."""
    L = thickness.shape[-1]
    below = depths(thickness) < depth[..., None]          # do-while continues while layerdepth < depth
    run = torch.ones_like(below[..., 0], dtype=torch.int64)
    n = torch.zeros_like(run)
    for l in range(L):                                    # the leading run of True (cumprod(...).sum(-1))
        run = run * below[..., l].to(torch.int64)
        n = n + run
    n = n + 1
    return torch.clamp(n, max=L)


def no3_n(s: SoilN):
    return s.no3 * NO3_TO_N


def nh4_n(s: SoilN):
    return s.nh4 * NH4_TO_N


# ----------------------------------------------------------------------------- abiotic environment

def temperature_function(t):
    """calc_temperature_function (abiotic_environment.cpp:20-40)."""
    q = (T_MIN - T_OPT) / (T_OPT - T_MAX)
    inside = (t >= T_MIN) & (t <= T_MAX)
    tt = torch.where(inside, t, torch.zeros_like(t) + T_OPT)
    f = (torch.pow(tt - T_MIN, q) * (T_MAX - tt)) / (math.pow(T_OPT - T_MIN, q) * (T_MAX - T_OPT))
    f = torch.where(inside, f, torch.full_like(t, MIN_TEMPERATURE_FUNCTION))
    return _between(f, torch.full_like(f, MIN_TEMPERATURE_FUNCTION), torch.ones_like(f))


def moisture_function(wfp, value_at_saturation):
    """calc_moisture_function (abiotic_environment.cpp:42-61)."""
    a = (wfp >= WFP_MIN) & (wfp < WFP_LOW)
    b = (wfp >= WFP_LOW) & (wfp <= WFP_HIGH)
    c = (wfp > WFP_HIGH) & (wfp <= 1.0)
    f = torch.where(a, (wfp - WFP_MIN) / (WFP_LOW - WFP_MIN),
        torch.where(b, torch.ones_like(wfp),
        torch.where(c, value_at_saturation + (1.0 - value_at_saturation)
                    * ((1.0 - wfp) / (1.0 - WFP_HIGH)) * ((1.0 - wfp) / (1.0 - WFP_HIGH)), torch.zeros_like(wfp))))
    return torch.where(f < 0.0, torch.full_like(f, 0.00001), f)


class Abiotic(NamedTuple):
    """Abiotic_environment_layer after Abiotic_environment_profile::update, [..., L]. The hourly
    values of the VIC soil are the same in every hour (no hourly water content or temperature)."""
    moisture_daily: torch.Tensor          # moisture_function (from the hour-24 water-filled porosity)
    tf_hour: torch.Tensor                 # temperature function of every hour (float32 sample values)
    tf_daily_mean: torch.Tensor           # float32(sum of the 24 float32 values / 24)
    minimized_tm: torch.Tensor            # sum over 24 hours of min(tf, moisture_hour)/24 (float64)
    wfp_hour: torch.Tensor                # water-filled porosity of hours 0-23 (liquid / saturation)
    wfp_daily: torch.Tensor               # hour 24: (liquid + ice) / thickness / saturation


def water_filled_porosity(liquid_wc, ice_wc, sat):
    """Soil_hydrology::get_water_filled_porosity (VIC_soil.cpp:216-222): every hour reads
    get_water_plus_ice_content_volumetric (the VIC override of the hourly getter, VIC_soil.h:116-118),
    (VIC ice + VIC moist) / thickness, over the saturation. Returns (hours 0-23, hour 24); the CRB
    runs have no soil ice, the oracle's SOIL mode checks days with ice."""
    w = (liquid_wc + ice_wc) / sat
    return w, w


def abiotic_update(liquid_wc, ice_wc, soil_t, layers: SoilLayers) -> Abiotic:
    """Abiotic_environment_profile::update (abiotic_environment.cpp:74-116) for layers 1..L.
    liquid_wc = VIC moist/(1000 depth), ice_wc = VIC ice/(1000 depth), soil_t = VIC layer T."""
    wfp_h, wfp_d = water_filled_porosity(liquid_wc, ice_wc, layers.sat)
    md = moisture_function(wfp_d, OM_MOISTURE_AT_SATURATION)
    mh = moisture_function(wfp_h, OM_MOISTURE_AT_SATURATION)
    tf = round_f32(temperature_function(soil_t))           # appended to a float32 sample (abiotic_environment.h:28)
    s = torch.zeros_like(tf)
    for _ in range(24):                                     # Sample_dataset_cowl::recompute, float64 sum
        s = s + tf
    mean = round_f32(s / 24.0)                             # arithmetic_mean is float32 (sample_dataset_T.h:196)
    m = torch.zeros_like(tf)
    v = _cmin(tf, mh) / 24.0
    for _ in range(24):                                     # :105-110
        m = m + v
    return Abiotic(md, tf, mean, m, wfp_h, wfp_d)


# ----------------------------------------------------------------------------- transformations

def denitrification(s: SoilN, layers: SoilLayers, wfp_hour, co2_c_loss, denitrification_event=None,
                    scale=DENITRIFICATION_SCALE, rate_max=DENITRIFICATION_RATE_MAX,
                    alpha=DENITRIFICATION_ALPHA, cascade_daily=False):
    """NO3_with_CO2_Profile::transformation for every layer (nitrogen_profile.cpp:402-617), the
    hourly loop of 24 one-hour intervals stopping when the NO3 is exhausted.
    co2_c_loss [..., L]: CO2-C lost from residues and SOM in the layer today (kg/m2,
    OM_pools_common.cpp:568). Returns (state, denitrified N per layer, N2O-N per layer)."""
    dt = s.no3.dtype
    dev = s.no3.device                                      # fills on the device: no host copies (CUDA graphs)
    c = lambda x: x.to(dt) if torch.is_tensor(x) else torch.full((), x, dtype=dt, device=dev)
    alpha32 = round_f32(c(alpha))                           # Denitrification_parameters are float32
    rate_max = round_f32(c(rate_max))
    scale = round_f32(c(scale))
    std_resp = round_f32(c(math.pow(DENITRIFICATION_POT_MAX / 0.1, 1.0 / 1.3)))  # float32 (:419-422)
    soil_mass = layers.bulk_density * 1000.0 * layers.thickness
    nitrate = NO3_TO_N * s.no3
    nitrate = torch.where(nitrate < NITRATE_MIN, torch.zeros_like(nitrate), nitrate)
    ppm = (nitrate / soil_mass) * 1000000.0
    thr = math.pow(DENITRIFICATION_POT_MAX / 1.15, 1 / 0.57)
    nf1 = torch.pow(ppm / thr, 0.57)
    nf2 = 1.0 / (1.0 + (((1.0 / 0.001) - 1.0) * torch.exp(-1.015 * ppm)))
    nitrate_f = _cmin(nf1, nf2)
    resp_mg = (co2_c_loss / soil_mass) * 1000000.0
    resp_f = torch.pow(resp_mg / std_resp, 1.3)
    resp_f = _cmin(resp_f, torch.ones_like(resp_f))
    co2_ha = co2_c_loss * 10000.0
    n2n2o_resp = 13.0 + (30.78 * torch.atan(math.pi * 0.07 * (co2_ha - 13.0))) / math.pi
    n2n2o_no3 = (1.0 - (0.5 + torch.atan(math.pi * 0.01 * (ppm - 190.0)) / math.pi)) * 25.0
    # moisture functions (:316-362); the hourly one does not depend on the hour on the VIC soil
    wfp_thr = (layers.fc * alpha32 + layers.pwp * (1.0 - alpha32)) / layers.sat
    def mf_hourly(thr_):
        return torch.where(wfp_hour > thr_,
                           torch.where(wfp_hour < UPPER_WFP_THRESHOLD,
                                       torch.pow((wfp_hour - thr_) / (UPPER_WFP_THRESHOLD - thr_), 2.0),
                                       torch.ones_like(wfp_hour)), torch.zeros_like(wfp_hour))
    if cascade_daily:
        ev = denitrification_event if denitrification_event is not None else torch.zeros_like(s.no3, dtype=torch.bool)
        mf, mf_part = torch.where(ev, 0.6, 0.0) + 0 * s.no3, torch.where(ev, 0.5, 0.0) + 0 * s.no3
    else:
        mf, mf_part = mf_hourly(wfp_thr), mf_hourly(wfp_thr + (-0.05))
    pot_rate = torch.where(wfp_hour < 0.1, rate_max + torch.zeros_like(wfp_hour),
                           scale + (0.4622 * wfp_hour * wfp_hour - 0.0893 * wfp_hour + 0.0523))
    factor = _cmin(resp_f, nitrate_f)
    tot_pot = ((pot_rate / 1000.0) * soil_mass * factor) * mf / 1000.0      # mg -> g -> kg
    ratio = _cmin(n2n2o_no3, n2n2o_resp) * mf_part
    n2o_frac = 1.0 / (1.0 + ratio)
    mass = s.no3
    exhausted = _is_zero(nitrate)
    den = torch.zeros_like(mass)
    n2o = torch.zeros_like(mass)
    for _ in range(SECONDS_PER_DAY // TRANSFORMATION_STEP):
        go = ~exhausted & (tot_pot > DENITRIFICATION_MIN_HOURLY)
        pot_m = N_TO_NO3 * tot_pot
        act_m = _cmin(pot_m, mass)
        act_m = torch.where(go, act_m, torch.zeros_like(act_m))
        mass = mass - act_m
        exhausted = exhausted | (go & _is_zero(act_m, EXHAUSTED_TOL))
        act_n = NO3_TO_N * act_m
        den = den + act_n
        n2o = n2o + torch.where(go, act_n * n2o_frac, torch.zeros_like(act_n))
    return SoilN(mass, s.nh4), den, n2o


def nitrification(s: SoilN, layers: SoilLayers, wfp_hour, tf_nitr_hour, soil_t):
    """NH4_Profile_with_pH::transformation for every layer (nitrogen_profile.cpp:623-718), run
    after denitrification (chemicals_profile.cpp:362-440, NO3 is first in the profile list).
    tf_nitr_hour: the hourly nitrification temperature function (float32 values, Abiotic.tf_hour).
    Returns (state, NO3-N formed per layer, N2O-N per layer)."""
    nitrate = NO3_TO_N * s.no3
    ammonium = NH4_TO_N * s.nh4
    exhausted = _is_zero(ammonium)
    mf = moisture_function(wfp_hour, NITRIFICATION_MOISTURE_AT_SATURATION)
    k = 1.0 - torch.exp(-NITRIFICATION_CONSTANT_HOURLY * layers.ph_function * tf_nitr_hour)
    nh4 = s.nh4
    to_no3 = torch.zeros_like(nh4)
    n2o = torch.zeros_like(nh4)
    for _ in range(24):
        pot = torch.where((nitrate / ammonium) < NITRIFICATION_NO3_NH4_RATIO,
                          (ammonium - (nitrate / NITRIFICATION_NO3_NH4_RATIO)) * k * mf, torch.zeros_like(nh4))
        act_m = _cmin(pot * N_TO_NH4, nh4)
        act_m = torch.where(exhausted, torch.zeros_like(act_m), act_m)
        nh4 = nh4 - act_m
        act = NH4_TO_N * act_m
        loss = act * NITRIFICATION_N2O_FRACTION
        n2o = n2o + loss
        hour_no3 = act - loss
        nitrate = nitrate + hour_no3
        to_no3 = to_no3 + hour_no3
    no3 = s.no3 + N_TO_NO3 * to_no3          # Chemical_pool::take_elemental_from_transformation (chemical_profile.cpp:1150-1157)
    return SoilN(no3, nh4), to_no3, n2o


def transformations(s: SoilN, layers: SoilLayers, env: Abiotic, soil_t, co2_c_loss, **kw):
    """Chemicals_profile::chemical_transformations with an hourly step (land_unit_sim.cpp:1837-1847):
    denitrification of every layer, then nitrification of every layer."""
    s, den, n2o_d = denitrification(s, layers, env.wfp_hour, co2_c_loss, **kw)
    s, nit, n2o_n = nitrification(s, layers, env.wfp_hour, env.tf_hour, soil_t)
    return s, den, nit


def uptake(s: SoilN, no3_n_request, nh4_n_request):
    """Chemical_uptake_profile::uptake_elemental for NO3 and NH4 (chemical_profile.cpp:314-336):
    actual = min-by-comparison(request, available); returns (state, NO3-N taken, NH4-N taken)."""
    a3 = NO3_TO_N * s.no3
    t3 = torch.where(no3_n_request > a3, a3, no3_n_request)
    no3 = s.no3 - _cmin(N_TO_NO3 * t3, s.no3)
    a4 = NH4_TO_N * s.nh4
    t4 = torch.where(nh4_n_request > a4, a4, nh4_n_request)
    nh4 = s.nh4 - _cmin(N_TO_NH4 * t4, s.nh4)
    return SoilN(no3, nh4), t3, t4


# ----------------------------------------------------------------------------- operations

# Ammonium_source / NH4_Appl_method label tables (N_sources.cpp:3-17), in enum order
AMMONIUM_SOURCES = ["urea", "urea_ammonium_nitrate", "ammonium_sulfate", "ammonium_nitrate", "anhydrous_ammonia",
                    "ammonium_nitrosulfate"]
NH4_APPL_METHODS = ["surface_broadcast", "surface_dribble", "incorporate", "inject"]

NH4_LOSS_TABLES = {   # inorganic_NH4.cpp:6-114, [ammonium source][application method], float32
    ('dry', 'MAX', 'above'): [[0.4, 0.3, 0.1, 0.01], [0.4, 0.3, 0.1, 0.01], [0.6, 0.5, 0.3, 0.01], [0.3, 0.25, 0.2, 0.01], [0.05, 0.05, 0.05, 0.05], [0.5, 0.3, 0.25, 0.01]],
    ('dry', 'MAX', 'below'): [[0.4, 0.3, 0.02, 0.01], [0.2, 0.15, 0.02, 0.01], [0.02, 0.02, 0.02, 0.01], [0.02, 0.02, 0.02, 0.01], [0.02, 0.02, 0.02, 0.01], [0.02, 0.02, 0.02, 0.01]],
    ('dry', 'MIN', 'above'): [[0.02, 0.02, 0.01, 0.01], [0.02, 0.02, 0.01, 0.01], [0.05, 0.04, 0.01, 0.01], [0.05, 0.04, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.05, 0.04, 0.01, 0.01]],
    ('dry', 'MIN', 'below'): [[0.05, 0.02, 0.01, 0.01], [0.02, 0.02, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01]],
    ('humid', 'MAX', 'above'): [[0.2, 0.15, 0.1, 0.01], [0.2, 0.15, 0.1, 0.01], [0.4, 0.3, 0.1, 0.01], [0.2, 0.15, 0.1, 0.01], [0.02, 0.02, 0.02, 0.02], [0.3, 0.2, 0.1, 0.01]],
    ('humid', 'MAX', 'below'): [[0.05, 0.05, 0.0, 0.01], [0.05, 0.05, 0.0, 0.01], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ('humid', 'MIN', 'above'): [[0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01]],
    ('humid', 'MIN', 'below'): [[0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ('subhumid', 'MAX', 'above'): [[0.3, 0.2, 0.1, 0.01], [0.3, 0.2, 0.1, 0.01], [0.5, 0.4, 0.2, 0.01], [0.25, 0.2, 0.15, 0.01], [0.03, 0.03, 0.03, 0.03], [0.3, 0.3, 0.3, 0.01]],
    ('subhumid', 'MAX', 'below'): [[0.3, 0.2, 0.02, 0.01], [0.15, 0.1, 0.02, 0.01], [0.02, 0.02, 0.02, 0.01], [0.02, 0.02, 0.02, 0.01], [0.02, 0.02, 0.02, 0.01], [0.02, 0.02, 0.02, 0.01]],
    ('subhumid', 'MIN', 'above'): [[0.02, 0.02, 0.01, 0.01], [0.02, 0.02, 0.02, 0.01], [0.02, 0.01, 0.01, 0.01], [0.02, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.02, 0.01, 0.01, 0.01]],
    ('subhumid', 'MIN', 'below'): [[0.05, 0.02, 0.01, 0.01], [0.02, 0.02, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01]],
}
NH4_SOURCES = ["urea", "UAN", "ammonium_sulfate", "ammonium_nitrate", "anhydrous_ammonia",
               "ammonium_nitrate_sulfate"]          # Ammonium_source order of the table rows
NH4_METHODS = ["surface_broadcast", "surface_dribble", "incorporate", "inject"]


def fract_inorganic_nh4_to_soil(source, method, calculated, volatilization_pct, cec, ph, wes2_mm=0.0,
                                wes7_mm=0.0):
    """Land_unit_simulation_VX::fract_inorganic_NH4_to_soil (inorganic_NH4.cpp:116-175) for one
    operation (host-side scalars, evaluated when the management events are set up). The VIC
    build never appends to `wettings` (the block is under SOIL_INFILTRATION, land_unit_sim.cpp:
    1733-1736), so the water entering the soil over 2 and 7 days is 0: the dry tables apply."""
    import numpy as np
    if not calculated:
        return 1 - (volatilization_pct / 100)
    f_cec = 1.0 - (cec - 10.0) / 15.0
    f_cec = f_cec if f_cec > 0.0 else 0.0              # CORN_must_be_between macro (compare.h:9)
    f_cec = f_cec if f_cec < 1.0 else 1.0
    side = "above" if ph > 7.00000001 else "below"
    rng, mn = np.float32(0), np.float32(0)
    for clim, cond in (("dry", wes7_mm < 3.0), ("subhumid", wes7_mm >= 3.0), ("humid", wes2_mm > 12)):
        if cond:
            hi = np.float32(NH4_LOSS_TABLES[(clim, "MAX", side)][source][method])
            lo = np.float32(NH4_LOSS_TABLES[(clim, "MIN", side)][source][method])
            rng, mn = np.float32(hi - lo), lo
    return 1 - (float(mn) + float(rng) * f_cec)


def fertilizer(s: SoilN, mask, layers: SoilLayers, no3_n_kg_ha, nh4_n_kg_ha, fract_to_soil, injection_depth_cm):
    """Land_unit_simulation_VX::perform_inorganic_nitrogen (perform_op.cpp:1282-1355): NO3 goes to
    the layer at the injection depth (application_at_layer, chemical_profile.cpp:429-441), the
    non-volatilized NH4 to the same layer (NH4_Profile::volatilized_application ->
    application split into 1 layer, nitrogen_profile.cpp:720-731, chemical_profile.cpp:408-427).
    Amounts in kg/ha are converted with over_ha_to_m2 = 0.0001 (corn units, :1313)."""
    L = s.no3.shape[-1]
    layer = layer_at_depth(layers.thickness, injection_depth_cm / 100.0)
    at = mask[..., None] & (torch.arange(1, L + 1, device=s.no3.device) == layer[..., None])
    nh4_m2 = nh4_n_kg_ha * 0.0001
    non_vol = nh4_m2 * fract_to_soil
    no3_add = N_TO_NO3 * (no3_n_kg_ha * 0.0001)
    nh4_add = (N_TO_NH4 * non_vol) / 1
    return SoilN(torch.where(at, s.no3 + no3_add[..., None], s.no3), torch.where(at, s.nh4 + nh4_add[..., None], s.nh4))


def application_split(mass, mask, addition_n, layer, split, to_molecular):
    """Chemical_profile::application (chemical_profile.cpp:408-427): the molecular amount split
    equally over `split` layers from `layer` (1-based), reduced while it would run past the bottom."""
    L = mass.shape[-1]
    split = torch.minimum(split, L - layer + 1)
    add_m = to_molecular * addition_n
    idx = torch.arange(1, L + 1, device=mass.device)
    inl = mask[..., None] & (idx >= layer[..., None]) & (idx < (layer + split)[..., None])
    return torch.where(inl, mass + (add_m / split)[..., None], mass)


def auto_n_application(s: SoilN, mask, total_n):
    """Land_unit_simulation_VX::commit_auto_N (perform_op.cpp:1552-1580) for a split application
    whose NO3 and NH4 parts are given in total_n = (NO3-N, NH4-N) kg/m2: half to layer 1, half
    over layers 2-5."""
    no3_n, nh4_n = total_n
    one = torch.ones_like(no3_n, dtype=torch.long)
    h3, h4 = no3_n / 2.0, nh4_n / 2.0
    no3 = application_split(s.no3, mask, h3, one, one, N_TO_NO3)
    no3 = application_split(no3, mask, h3, one * 2, one * 4, N_TO_NO3)
    nh4 = application_split(s.nh4, mask, h4, one, one, N_TO_NH4)
    nh4 = application_split(nh4, mask, h4, one * 2, one * 4, N_TO_NH4)
    return SoilN(no3, nh4)


def tillage_mixing(s: SoilN, layers: SoilLayers, sand_pct, clay_pct, depth, mixing):
    """Soil_tillage_effect::respond_to_field_operation (tillage_effect.cpp:240-370): mixes
    NO3, NH4 and the texture of the tilled layers. depth, mixing [...]: the NRCS operation's
    average disturbance depth [m] and mixing fraction; a zero depth or fraction leaves the soil
    unchanged (:247-249). The water content change is a no-op on the VIC soil (patched stub,
    VIC_soil.cpp:338-343). Returns (state, sand %, clay %, last tilled layer (1-based, 0 = none))."""
    L = layers.thickness.shape[-1]
    active = ~(_is_zero(depth) | _is_zero(mixing))
    last = torch.where(active, layer_at_depth(layers.thickness, depth), torch.zeros_like(depth, dtype=torch.long))
    mass = soil_mass(layers)
    bottom = depths(layers.thickness)
    clay_f, sand_f = clay_pct / 100.0, sand_pct / 100.0
    no3c = (NO3_TO_N * s.no3) / mass
    nh4c = (NH4_TO_N * s.nh4) / mass
    tot = torch.zeros_like(depth)
    mc = torch.zeros_like(depth); ms = torch.zeros_like(depth)
    m3 = torch.zeros_like(depth); m4 = torch.zeros_like(depth)
    above = torch.zeros_like(depth)
    mixed = []
    remain = []
    for l in range(L):
        inl = active & (l + 1 <= last)
        m = mass[..., l]
        b = bottom[..., l]
        full = b <= depth
        part = (b > depth) & (above < depth)
        mx = torch.where(full, m * mixing, torch.where(part, mixing * m * (depth - above) / layers.thickness[..., l],
                                                         torch.zeros_like(m)))
        rm = torch.where(full, (1.0 - mixing) * m, torch.where(part, m - mx, m))
        den = tot + mx
        mc = torch.where(inl, (tot * mc + mx * clay_f[..., l]) / den, mc)
        ms = torch.where(inl, (tot * ms + mx * sand_f[..., l]) / den, ms)
        m3 = torch.where(inl, (tot * m3 + mx * no3c[..., l]) / den, m3)
        m4 = torch.where(inl, (tot * m4 + mx * nh4c[..., l]) / den, m4)
        tot = torch.where(inl, tot + mx, tot)
        above = torch.where(inl, b, above)
        mixed.append(mx)
        remain.append(rm)
    mixed = torch.stack(mixed, -1)
    remain = torch.stack(remain, -1)
    inl = active[..., None] & (torch.arange(1, L + 1, device=depth.device) <= last[..., None])
    new_clay = (clay_f * remain + mc[..., None] * mixed) / mass * 100.0
    new_sand = (sand_f * remain + ms[..., None] * mixed) / mass * 100.0
    new_no3 = ((no3c * remain + m3[..., None] * mixed) / mass) * mass
    new_nh4 = ((nh4c * remain + m4[..., None] * mixed) / mass) * mass
    no3 = torch.where(inl, N_TO_NO3 * new_no3, s.no3)
    nh4 = torch.where(inl, N_TO_NH4 * new_nh4, s.nh4)
    return SoilN(no3, nh4), torch.where(inl, new_sand, sand_pct), torch.where(inl, new_clay, clay_pct), last


# ----------------------------------------------------------------------------- NRCS operations

_SDR = None


def sdr_operations():
    """The NRCS/RUSLE2 SDR field operations by description (sdr_field_ops.txt, extracted from the
    compiled reference with tests/oracle/ref_cropsyst_nitrogen.cpp): name -> dict(depth_m,
    mixing, rating, effect, flatten[6], burial[6]); depth and mixing are the float32 values the
    reference passes on (SDR_field_ops.cpp:1086-1127), rating the nat8 sum of the disturbance
    ratings (:1145-1166), effect with the unhandled effects mapped to NEGLIGIBLE (:1129-1143).
    The first operation of a name wins (lookup_operation_number, :1042-1066)."""
    global _SDR
    if _SDR is None:
        import os
        _SDR = {}
        for line in open(os.path.join(os.path.dirname(__file__), "sdr_field_ops.txt")):
            if not line.startswith("OP "):
                continue
            a = line.rstrip("\n").split(" ", 18)
            name = a[18]
            if name in _SDR:
                continue
            f = [float.fromhex(x) for x in a[6:18]]
            _SDR[name] = dict(number=int(a[1]), depth_m=float.fromhex(a[2]), mixing=float.fromhex(a[3]),
                              rating=int(a[4]), effect=int(a[5]), flatten=f[:6], burial=f[6:])
    return _SDR
