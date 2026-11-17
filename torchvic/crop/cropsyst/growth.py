"""E4 and E5 of the VIC-CropSyst port (docs/connectors/cropsyst.md):
potential transpiration with the CO2 response, and biomass growth limited by radiation-use and
transpiration-use efficiency.

E4  ET max and potential transpiration      crop_common.cpp (update_evapotranspiration_max),
                                            crop_cropsyst.cpp:3606-3619,
                                            canopy_growth_LAI_based.cpp (calc_adjusted_ET_crop_coefficient),
                                            land_unit_sim.cpp:1296-1322 (called before the crop's
                                            start_day, on yesterday's canopy)
    CO2 response                            CO2_response.cpp:16-128, crop_cropsyst.cpp:2752-2761
E5  attainable growth min(RUE, TUE)         biomass_growth_RUE_TUE.cpp:67-170
    TUE(VPD)                                transpiration_dependent_growth_V4.cpp:24-56,
                                            transpiration_dependent_growth.cpp:16-34
    limited potential transpiration         crop_cropsyst.cpp:2829-2842 (nitrogen disabled)
    actual growth and leaf-expansion factor crop_cropsyst.cpp:2847-2947

What the coupled run must supply each day (inputs of the functions below):
- ET_ref [m]: the shipped V3 build uses CropSyst's own reference ET (land_unit_sim.cpp:1296-1305),
  and of the PREVIOUS day: the ET max is taken before the reference ET is updated (the crop
  csv's crop_evapotransp_max equals yesterday's CropSyst_refET times the coefficient, as the replay
  shows); the coupled run uses VIC PET_SHORT instead.
- VPD max [kPa] = es(Tmax)(1 - RHmin/100) (WC_vapor_pressure_deficit.cpp:53-76, over water,
  es Murray 1966 = 0.6108 exp(17.2693882 T/(T + 273.15 - 35.86)), `saturation_vp_kpa`); the daytime VPD is 0.666666 of it, floored at 1e-5
  (WC_vapor_pressure_deficit.cpp:183-191). `vpd_daytime` below computes it.
- Solar radiation [MJ/m2/day], the mean air temperature (Tmax+Tmin)/2.
- CO2 [ppm] (vic_control CO2_PPM), the slope of the saturation vapour pressure curve delta
  [kPa/C] at the mean temperature (WC_vapor_pressure_saturation_slope.cpp:9-18), the psychrometric
  constant gamma [kPa/C] and the aerodynamic resistance of the reference crop ra [day/m] of the
  ET reference model (CO2_response.cpp:107-125).

Reference quirks ported as is:
- RUE_global, min/max_tolerable_temp and the threshold temperatures of the [growth] section are
  read only when CS_VERSION < 5 (crop_param_class.cpp:136-153); the shipped build is CS_VERSION 5,
  so RUE = 4.0 g/MJ and the temperature limitation 3/10/25/45 C are the class defaults for every
  CRB crop (checked against the reference parser, tests/test_cropsyst_growth.py).
- The RUE temperature limitation has an always-true middle test (`T >= low || T <= high`,
  biomass_growth_RUE_TUE.cpp:156-158); the later ramps override it; 1e-8 below 0.01.
- The TUE-dependent growth is floored at 0.00001 kg/m2 (transpiration_dependent_growth.cpp:30),
  so on the emergence day, when the potential transpiration (computed before emergence) is 0,
  attainable growth is 1e-5 kg/m2.
- CO2_TUE_adjustment is reset to 1 on days with green cover <= 1e-7 (crop_cropsyst.cpp:2754-2757).
- The CO2 transpiration factor multiplies the potential transpiration while TUE is divided by
  it, so it cancels in TUE-limited growth but not in the limited potential transpiration.
- The transpiration-use efficiency used for the limited potential transpiration is the value of
  the last TUE evaluation (a mutable member): after maturity it is no longer refreshed.
"""
from __future__ import annotations

import math

import torch

RC_FAO56 = 0.00081                   # CO2_response.cpp:35 (day/m)
CO2_SATURATION = 1100.0              # CO2_response.cpp:14
CO2_FAO_1998 = 359.0                 # CO2_response.cpp:48
VPD_DAYTIME_FRACTION = 0.666666      # WC_vapor_pressure_deficit.cpp:183
VPD_FLOOR = 0.00001                  # WC_vapor_pressure_deficit.cpp:187
TUE_VPD_FLOOR = 0.00001              # transpiration_dependent_growth_V4.cpp:26
TUE_GROWTH_FLOOR = 0.00001           # transpiration_dependent_growth.cpp:30
GREEN_TUE_THRESHOLD = 0.0000001      # crop_cropsyst.cpp:2755
GROWTH_STRESS_EPS = 0.000001         # crop_cropsyst.cpp:2945


def round_f32(x):
    """Round float64 values to the nearest float32 value, staying in float64 (the reference
    stores these quantities in float32). Veltkamp splitting with 2^29 + 1 keeps 24 significant
    bits; unlike x.to(float32).to(float64) it is not rewritten by torch.compile (inductor may
    evaluate the producing expression in float32 when the result is cast to float32).
    Ties round away from the even neighbour in rare cases; the rounded quantities here are
    quotients and sums, never exact ties in practice."""
    c = x * 536870913.0
    return c - (c - x)


def round_f32_bits(x):
    """Round float64 values to the nearest float32 value (ties to even), staying in float64, with
    integer operations on the bits: unlike the Veltkamp split of round_f32, torch.compile on CUDA
    cannot contract it into an FMA and drop the rounding (seen with the hourly temperatures).
    For finite values in the float32 normal range."""
    b = x.view(torch.int64)
    keep = (b >> 29) & 1
    b = (b + (0x0FFFFFFF + keep)) & ~0x1FFFFFFF
    return b.view(torch.float64)


def _cmax(a, b):
    """std::max(a, b) = (a < b) ? b : a."""
    return torch.where(a < b, b, a)


def _cmin(a, b):
    """std::min(a, b) = (b < a) ? b : a."""
    return torch.where(b < a, b, a)


def saturation_vp_kpa(t):
    """es over water [kPa], Murray (1966) as Physical::calc_saturation_kPa
    (common/physics/property/P_vapor_pressure.cpp:7-27): 0.6108 exp(17.2693882 T / (T + 273.15
    - 35.86)); VPD max uses OVER_WATER_ONLY (WC_vapor_pressure_deficit.cpp:60-62). Not the FAO
    17.27/237.3 form (that form biases TUE growth by ~2e-6 per day, found in the replay)."""
    return 0.6108 * torch.exp((17.2693882 * t) / ((t + 273.15) - 35.86))


def vpd_max_from_rhmin(tmax, rhmin_pct):
    return saturation_vp_kpa(tmax) * (1.0 - rhmin_pct / 100.0)


def vpd_daytime(vpd_max):
    """Vapor_pressure_deficit_daytime_calculator::estimate_from_VPD_max."""
    v = vpd_max * VPD_DAYTIME_FRACTION
    return torch.where(v <= 0.0, torch.full_like(v, VPD_FLOOR), v)


# ----------------------------------------------------------------------------- E4

def adjusted_et_coef(lai_ss, et_coef):
    """Canopy_growth_leaf_area_index_based::calc_adjusted_ET_crop_coefficient on the self-shading
    LAI; Crop_complete::update_adjusted_ET_crop_coefficient gives 1 when ET_coef == 0."""
    adj = torch.where((et_coef > 1.0) & (lai_ss < 3.0), 1.0 + (et_coef - 1.0) * lai_ss / 3.0,
                      et_coef + torch.zeros_like(lai_ss))
    return torch.where(et_coef == 0.0, torch.ones_like(adj), adj)


def et_max(et_ref, kc_adj):
    e = et_ref * kc_adj
    return torch.where(e < 0.0, torch.zeros_like(e), e)


def canopy_conductance(c):
    """Allen (1990), CO2_response.cpp:38-44."""
    return 0.0485 - 0.00007 * c + 0.000000034 * (c * c)


def _growth_ratio(alpha, theta, gra, c):
    ac = alpha * c + gra
    return (ac - torch.sqrt(ac * ac - 4.0 * theta * alpha * c * gra)) / (2.0 * theta)


def co2_gains(co2_ppm, alpha, theta, gra, tue_conc, rue_conc):
    """Crop_CO2_response::initialize + update_with_new_atmphospheric_CO2 (CO2_response.cpp:45-104):
    returns (canopy conductance adjustment vs FAO56, TUE gain ratio, RUE gain ratio)."""
    c = co2_ppm + torch.zeros_like(alpha)
    g_tue = _growth_ratio(alpha, theta, gra, tue_conc)
    g_rue = _growth_ratio(alpha, theta, gra, rue_conc)
    ac = alpha * c + gra
    plus = ac * ac
    minus = 4.0 * theta * alpha * c * gra
    rel = (alpha * c + gra - torch.sqrt(plus - minus)) / (2.0 * theta)
    g359 = canopy_conductance(torch.full_like(c, CO2_FAO_1998))
    gsat = canopy_conductance(torch.full_like(c, CO2_SATURATION))
    cond = torch.where(c >= CO2_SATURATION, gsat, canopy_conductance(c)) / g359
    return cond, rel / g_tue, rel / g_rue


def co2_daily(cond_adj, gain_tue, delta, gamma, ra):
    """Crop_CO2_response::update (CO2_response.cpp:106-128): transpiration adjustment factor and
    TUE adjustment for today's delta [kPa/C], gamma [kPa/C] and ra [day/m]."""
    rc_adj = RC_FAO56 / cond_adj
    f = (delta + gamma * (RC_FAO56 + ra) / ra) / (delta + gamma * (rc_adj + ra) / ra)
    return f, gain_tue / f


# ----------------------------------------------------------------------------- E5

def rue_temperature_limitation(tavg, tmin_, tlow, thigh, tmax_):
    """Biomass_growth_RUE_TUE::calc_RUE_temperature_limitation (biomass_growth_RUE_TUE.cpp:143-170)."""
    one = torch.ones_like(tavg)
    tsf = torch.where((tavg <= tmin_) | (tavg >= tmax_), torch.full_like(tavg, 0.00000001),
                      torch.where((tavg >= tlow) | (tavg <= thigh), one, one))
    tsf = torch.where((tavg < tlow) & (tavg > tmin_), 1.0 - ((tlow - tavg) / (tlow - tmin_)), tsf)
    tsf = torch.where((tavg > thigh) & (tavg < tmax_), (tmax_ - tavg) / (tmax_ - thigh), tsf)
    tsf = torch.where(tsf > 1.0, one, tsf)
    return torch.where(tsf <= 0.01, torch.full_like(tsf, 0.00000001), tsf)


def tue_unadjusted(vpd_day, tue_1kpa, scaling):
    """get_transpiration_use_efficiency_unadj (transpiration_dependent_growth_V4.cpp:24-37)
    [kg biomass / kg water]."""
    v = torch.where(vpd_day < TUE_VPD_FLOOR, torch.full_like(vpd_day, TUE_VPD_FLOOR), vpd_day)
    return (tue_1kpa / 1000.0) * torch.pow(v, -scaling)


def attainable_growth(intercepted_rad, rue_g_mj, gain_rue, tsf, pot_transp_co2_m, tue_unadj,
                      co2_tue_adjustment, ccif):
    """calc_attainable_top_growth, TUE efficiency model (biomass_growth_RUE_TUE.cpp:67-140).
    Returns (attainable growth [kg/m2], TUE CO2-adjusted)."""
    rue = (rue_g_mj / 1000.0) * gain_rue
    rad_growth = intercepted_rad * rue
    t_corr = rad_growth * tsf
    tue = tue_unadj * co2_tue_adjustment
    pt = (pot_transp_co2_m * 1000.0) * tue
    pt = torch.where(pt <= 0.0, torch.full_like(pt, TUE_GROWTH_FLOOR), pt)
    return _cmin(t_corr, pt) * ccif, tue


def limited_pot_transpiration(attainable, tue_co2_adjusted):
    """no_nitrogen_calculate_limited_pot_transpiration (crop_cropsyst.cpp:2829-2842) [m]."""
    return (attainable / tue_co2_adjusted) / 1000.0


def actual_growth(attainable, limited_pot, act_transp, lwp, reduce_lwp, stop_lwp):
    """calc_act_biomass_growth with nitrogen disabled (crop_cropsyst.cpp:2847-2947):
    returns (actual growth, GAI-related growth, water canopy expansion factor, overall stress)."""
    pos = limited_pot > 0.0
    ratio = act_transp / torch.where(pos, limited_pot, torch.ones_like(limited_pot))
    tlg = attainable * ratio
    act = torch.where(pos, _cmin(tlg, attainable), torch.zeros_like(attainable))
    w = (lwp - stop_lwp) / (reduce_lwp - stop_lwp)
    w = _cmin(_cmax(w, torch.zeros_like(w)), torch.ones_like(w))
    gai_rel = torch.where(pos, act * w, torch.zeros_like(act))
    overall = torch.where(attainable > GROWTH_STRESS_EPS,
                          act / torch.where(attainable > GROWTH_STRESS_EPS, attainable,
                                            torch.ones_like(attainable)),
                          torch.ones_like(attainable))
    return act, gai_rel, torch.where(pos, w, torch.full_like(w, -1.0)), overall


def water_stress(act_transp, limited_pot):
    """process_transpiration_m (crop_common.cpp): the ratio appended as float32 to the
    float64 Sample_cumulative; index = 1 - value (sample_cumulative_T.h:49-55)."""
    pos = limited_pot > 0.0
    ws = torch.where(pos, act_transp / torch.where(pos, limited_pot, torch.ones_like(limited_pot)),
                     torch.ones_like(limited_pot))
    ws = round_f32(ws)
    return ws, 1.0 - ws
