"""E9 of the VIC-CropSyst port (docs/connectors/cropsyst.md): grain
yield with the source-limited harvest index and the extreme-temperature response of the CRB
grain crops (Yield_grain, crop_cropsyst.cpp:388-419).

Sources: yield.cpp:118-216 (Yield_grain::calculate, know_stress), harvest_grain_asymptotic.c
(calc_source_limited_harvest_index), yield_temperature_extreme_response.cpp/.h (periods,
temperature stress function, factor), event_scheduler_crop.cpp:96-112 (grain stress periods
start on stage + offset), crop_cropsyst.cpp:1849-1866 (harvest_crop's biomass for harvest),
the hourly stress-adjusted temperatures (crop_cropsyst.cpp:2655-2662, crop_cropsyst.h:106-119,
temperature_est.cpp, min_max_to_timestep_estimator.cpp).

Reference quirks ported as is:
- The hourly estimator is reset when the crop is created (sowing day) and shifted the same day,
  so on the sowing day "yesterday's" Tmax is today's and "today's" Tmin is tomorrow's
  (crop_cropsyst.cpp:378-387 then temperature_est.cpp:31-38).
- Hourly values and the time weights are float32; hours 1-5 use yesterday's (unadjusted) Tmax
  and today's Tmin, hours after 14:00 tomorrow's Tmin, the others the stress-adjusted Tmax.
- The temperature response accrues only on days in accrescence or with growth stage ==
  anthesis (yield.cpp:205-212), into the most recently started period only.
- Per-period means are float32 means of float32 daily means of float32 hourly factors
  (Sample_dataset recompute, sample_dataset_T.h:153-195); a period with hours but no cold (or
  heat) samples has a 0 mean, so its factor is 0.
- Heat exposure starting hour > 24 means the sunrise hour (yield_temperature_extreme_response.cpp:
  38-40); hours wrap after 23.
- VIC build: when biomass at maturity is below biomass at filling, the current canopy biomass is
  used for the yield-formation fraction (yield.cpp:166-172, VIC_CROPSYST_VERSION).
- The asymptote iteration stops after 100 iterations or when the change < 1e-4 or the decay
  coefficient < 0.1 of its maximum; at 100 iterations the sink-strength HI is kept.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

TWF = [float(__import__("numpy").float32(0.44 - 0.46 * math.sin(i * math.pi / 12 + 0.9)
                                         + 0.11 * math.sin(2.0 * i * math.pi / 12 + 0.9)))
       for i in range(1, 25)]                      # min_max_to_timestep_estimator.cpp:24-27
BEFORE_5AM = [i <= 5 for i in range(1, 25)]
AFTER_2PM = [i > 14 for i in range(1, 25)]
KIND_GRAIN, KIND_TUBER, KIND_CANOPY, KIND_SIMPLE, KIND_COMPLETE = 0, 1, 2, 3, 4
MAX_HOURS = 48                                     # static unroll of the exposure-hour loop
HI_ITERS = 100
FRACTION_NEAR_ZERO = 0.00001



def _rep(nt, **kw):
    """NamedTuple._replace without collections internals (traceable by torch.compile)."""
    return type(nt)(*[kw[f] if f in kw else getattr(nt, f) for f in type(nt)._fields])

def _f32(x):
    from .growth import round_f32_bits  # float32 rounding that survives torch.compile on CUDA
    return round_f32_bits(x)


def _cmin(a, b):
    return torch.where(b < a, b, a)


class YieldState(NamedTuple):
    max_today: torch.Tensor        # hourly estimator state (unadjusted Tmax, Tmin)
    max_yday: torch.Tensor
    min_today: torch.Tensor
    min_tom: torch.Tensor
    hourly: torch.Tensor           # [..., 24] float32 values (as float64)
    bm_filling: torch.Tensor
    bm_maturity: torch.Tensor
    cur: torch.Tensor              # long, current period slot (-1 none)
    started: torch.Tensor          # [..., G] bool, period started (in the factor list)
    cold_sum: torch.Tensor         # [..., G] sum of float32 daily means
    cold_n: torch.Tensor           # [..., G]
    heat_sum: torch.Tensor
    heat_n: torch.Tensor


def initial_state(shape, G, dtype=torch.float64, device=None) -> YieldState:
    z = torch.zeros(shape, dtype=dtype, device=device)
    zg = torch.zeros(tuple(shape) + (max(G, 1),), dtype=dtype, device=device)
    return YieldState(z, z, z, z, torch.zeros(tuple(shape) + (24,), dtype=dtype, device=device), z, z,
                      torch.full(shape, -1, dtype=torch.long, device=device),
                      torch.zeros(tuple(shape) + (max(G, 1),), dtype=torch.bool, device=device),
                      zg, zg, zg, zg)


def sow(y: YieldState, mask, tmax, tmin_tomorrow) -> YieldState:
    """New crop: estimator reset (only max_today and min_tomorrow matter, see module doc),
    no periods, no filling/maturity biomass."""
    w = lambda new, old: torch.where(mask, new, old)
    z = torch.zeros_like(y.max_today)
    mg = mask[..., None]
    return _rep(y, max_today=w(tmax + z, y.max_today), min_tom=w(tmin_tomorrow + z, y.min_tom),
                      bm_filling=w(z, y.bm_filling), bm_maturity=w(z, y.bm_maturity),
                      cur=torch.where(mask, torch.full_like(y.cur, -1), y.cur),
                      started=y.started & ~mg,
                      cold_sum=torch.where(mg, torch.zeros_like(y.cold_sum), y.cold_sum),
                      cold_n=torch.where(mg, torch.zeros_like(y.cold_n), y.cold_n),
                      heat_sum=torch.where(mg, torch.zeros_like(y.heat_sum), y.heat_sum),
                      heat_n=torch.where(mg, torch.zeros_like(y.heat_n), y.heat_n))


_TABLES = {}


def _hour_tables(device, dtype):
    """Time weights and hour masks, created once per device (no host copies inside a captured
    CUDA graph: call the day step once before capturing)."""
    key = (str(device), dtype)
    if key not in _TABLES:
        _TABLES[key] = (torch.tensor(TWF, dtype=dtype, device=device),
                        torch.tensor(BEFORE_5AM, device=device), torch.tensor(AFTER_2PM, device=device))
    return _TABLES[key]


def hourly_temperatures(y: YieldState, mask, tmax, tmin_tomorrow, tmax_adj) -> YieldState:
    """set_max_today_min_tomorrow + get_temperature_by_interval (update_temperatures)."""
    max_yday = y.max_today
    min_today = y.min_tom
    dev = tmax_adj.device
    twf, before, after = _hour_tables(dev, tmax_adj.dtype)
    tn = torch.where(before, min_today[..., None], torch.where(after, (tmin_tomorrow + torch.zeros_like(min_today))[..., None], min_today[..., None]))
    tx = torch.where(before, max_yday[..., None], tmax_adj[..., None])
    h = _f32(twf * tx + (1.0 - twf) * tn)
    w = lambda new, old: torch.where(mask, new, old)
    z = torch.zeros_like(y.max_today)
    return _rep(y, max_yday=w(max_yday, y.max_yday), min_today=w(min_today, y.min_today),
                      max_today=w(tmax + z, y.max_today), min_tom=w(tmin_tomorrow + z, y.min_tom),
                      hourly=torch.where(mask[..., None], h, y.hourly))


def start_periods(y: YieldState, fire) -> YieldState:
    """add_period for the grain stress periods firing today, in slot order ([..., G] bool)."""
    G = fire.shape[-1]
    cur = y.cur
    for g in range(G):                 # static count of periods in the crop file
        cur = torch.where(fire[..., g], torch.full_like(cur, g), cur)
    return _rep(y, cur=cur, started=y.started | fire)


def temperature_stress_function(t, no_stress, max_stress, sens):
    """Temperature_factors::temperature_stress_function (yield_temperature_extreme_response.h)."""
    one, zero = torch.ones_like(t), torch.zeros_like(t)
    heat = torch.where(t <= no_stress, one, torch.where(t >= max_stress, zero,
                       1.0 - (t - no_stress) / (max_stress - no_stress)))
    cold = torch.where(t >= no_stress, one, torch.where(t <= max_stress, zero,
                       1.0 - (no_stress - t) / (no_stress - max_stress)))
    tsf = torch.where(max_stress > no_stress, heat, cold)
    return torch.where(torch.abs(0.0 - sens) <= 0.00001, one, torch.pow(tsf, sens))


def _exposure_mean(hourly, init_hour, dmg_hours, no_s, max_s, sens):
    j = torch.arange(MAX_HOURS, device=hourly.device)
    n = torch.floor(dmg_hours).clamp(min=0)            # int8 counter 1..dmg_hours
    h0 = torch.where(init_hour > 23, torch.zeros_like(init_hour), init_hour)
    hrs = torch.remainder(h0[..., None] + j, 24)
    t = torch.gather(hourly, -1, hrs.long())
    tsf = temperature_stress_function(t, no_s[..., None], max_s[..., None], sens[..., None])
    tsf = _f32(tsf)
    use = j < n[..., None]
    s = torch.where(use, tsf, torch.zeros_like(tsf)).sum(-1)
    return torch.where(n > 0, _f32(s / torch.where(n > 0, n, torch.ones_like(n))), torch.zeros_like(s))


def update_extreme(y: YieldState, mask, sunrise_hour, cold_no, cold_max, cold_sens, cold_init,
                   cold_hours, heat_no, heat_max, heat_sens, heat_init, heat_hours) -> YieldState:
    """Yield_grain_temperature_extreme_response::update for the current period; the period
    parameters are [..., G] and gathered at the current slot."""
    on = mask & (y.cur >= 0)
    k = y.cur.clamp(min=0)[..., None]
    g = lambda x: torch.gather(x, -1, k)[..., 0]
    cm = _exposure_mean(y.hourly, g(cold_init), g(cold_hours), g(cold_no), g(cold_max), g(cold_sens))
    hi = g(heat_init)
    hi = torch.where(hi > 24, sunrise_hour + torch.zeros_like(hi), hi)
    hm = _exposure_mean(y.hourly, hi, g(heat_hours), g(heat_no), g(heat_max), g(heat_sens))
    sel = (torch.arange(y.cold_sum.shape[-1], device=k.device) == k) & on[..., None]
    return _rep(y, cold_sum=torch.where(sel, y.cold_sum + cm[..., None], y.cold_sum),
                      cold_n=torch.where(sel, y.cold_n + 1.0, y.cold_n),
                      heat_sum=torch.where(sel, y.heat_sum + hm[..., None], y.heat_sum),
                      heat_n=torch.where(sel, y.heat_n + 1.0, y.heat_n))


def extreme_factor(y: YieldState):
    """get_factor: product over started periods of min(cold mean, heat mean)."""
    def mean(s, n):
        return torch.where(n > 0, _f32(s / torch.where(n > 0, n, torch.ones_like(n))), torch.zeros_like(s))
    eff = _cmin(mean(y.cold_sum, y.cold_n), mean(y.heat_sum, y.heat_n))
    eff = torch.where(y.started & ((y.cold_n + y.heat_n) > 0), eff, torch.ones_like(eff))
    f = torch.ones_like(y.max_today)
    for g in range(eff.shape[-1]):
        f = f * eff[..., g]
    return f


def source_limited_harvest_index(frac, attainable_hi, transl):
    """calc_source_limited_harvest_index (harvest_grain_asymptotic.c), 100 unrolled iterations."""
    tmin = transl / 3.0
    big = attainable_hi > transl
    den = torch.where(big, attainable_hi - transl, torch.ones_like(attainable_hi))
    maxf = (attainable_hi - tmin) / (1.0 - tmin)
    kmax = (1.0 - transl) / den
    hi0 = FRACTION_NEAR_ZERO + (1.0 - FRACTION_NEAR_ZERO) * transl
    k = kmax
    asym = torch.zeros_like(k)
    searching = big.clone()
    iters = torch.zeros_like(k)
    for _ in range(HI_ITERS):
        a = (attainable_hi - transl * torch.exp(-k * maxf)) / (1.0 - torch.exp(-k * maxf))
        nk = -torch.log((a - hi0) / (a - transl)) / FRACTION_NEAR_ZERO
        err = torch.abs(nk - k)
        asym = torch.where(searching, a, asym)
        k_new = torch.where(searching, nk, k)
        iters = torch.where(searching, iters + 1.0, iters)
        stop = (err < 0.0001) | (nk < 0.1 * kmax)
        searching = searching & ~stop
        k = k_new
    hi = asym - (asym - transl) * torch.exp(-k * frac)
    hi = torch.where(big & (iters < HI_ITERS), hi, attainable_hi)
    return torch.where(attainable_hi < 0.0001, attainable_hi, hi)


def grain_yield(biomass_for_harvest, bm_filling, bm_maturity, hi_unstressed, adjust, transl, factor):
    """Yield_grain::calculate (yield.cpp:136-180); returns (yield [kg/m2], reported HI)."""
    stress = _cmin(factor, torch.ones_like(factor))
    sink = hi_unstressed * stress
    pos = bm_maturity > 0.000001
    frac = torch.where(pos, (bm_maturity - bm_filling) / torch.where(pos, bm_maturity, torch.ones_like(bm_maturity)),
                       torch.zeros_like(bm_maturity))
    low = bm_maturity < bm_filling
    harv = torch.where(low, biomass_for_harvest, bm_maturity)
    frac = torch.where(low, (harv - bm_filling) / harv, frac)
    hi = source_limited_harvest_index(frac, sink, transl)
    y_adj = biomass_for_harvest * hi
    hi = torch.where(adjust, hi, hi_unstressed + torch.zeros_like(hi))
    return torch.where(adjust, y_adj, biomass_for_harvest * hi_unstressed), hi


def sunrise_hour(lat_deg, lon_deg, doy):
    """Integer sunrise hour of Sun::provide_sun_rise_set_times with the declination of
    Sun::calc_angles (common/weather/sun.cpp:40-55, 143-209; time_A.cpp:24-25). This is what the
    heat exposure of a grain stress period with initial hour > 24 starts from; the coupled run
    supplies it per day (host or device tensor)."""
    lat = torch.as_tensor(lat_deg, dtype=torch.float64) * (math.pi / 180.0)
    lon = torch.as_tensor(lon_deg, dtype=torch.float64)
    d = torch.as_tensor(doy, dtype=torch.float64)
    sdec = 0.39785 * torch.sin(4.869 + 0.0172 * d + 0.03345 * torch.sin(6.224 + 0.0172 * d))
    dec = torch.asin(sdec)
    coshs = (-torch.sin(lat) * torch.sin(dec)) / (torch.cos(lat) * torch.cos(dec))
    sinhs = torch.sqrt(1.0 - coshs * coshs)
    half = math.pi / 2 - torch.atan(coshs / sinhs)
    f = (279.575 + 0.9856 * d) * math.pi / 180.0
    et = (-104.7 * torch.sin(f) + 596.2 * torch.sin(2.0 * f) + 4.3 * torch.sin(3.0 * f)
          - 12.7 * torch.sin(4 * f) - 429.3 * torch.cos(f) - 2.0 * torch.cos(2.0 * f)
          + 19.3 * torch.cos(3.0 * f)) / 3600.0
    noon = 12.0 - (torch.trunc(lon) - lon) / 15.0 - et
    rise = noon - (half * 180.0 / math.pi) / 15.0
    sec = torch.trunc(rise * 3600.0)
    return torch.remainder(torch.div(sec, 3600, rounding_mode="trunc"), 24)


def other_yield(kind, bfh, hi_unstressed, y_grain, hi_grain):
    """The other yield models of crop_cropsyst.cpp:388-466 with zero stress sensitivities (the
    shipped CRB crops): Yield_tuber::calculate (yield.cpp:76-96, bulking/initiation effects are 1.0
    for a zero sensitivity, sensitivity_effect_B), Yield_canopy (all biomass, reported HI stays 0),
    Yield_complete (all biomass, HI 1) and Yield_simple_harvest_index (yield.cpp:98-120, min of
    biomass * HI and the stressed value, equal for a zero sensitivity; HI printed only for
    biomass > 0). Grain crops pass through."""
    y_hi = bfh * hi_unstressed
    pos = bfh > 0.00001
    safe = torch.where(pos, bfh, torch.ones_like(bfh))
    tuber_hi = torch.where(pos, y_hi / safe, torch.zeros_like(bfh))
    simple_hi = torch.where(bfh > 0.0, y_hi / torch.where(bfh > 0.0, bfh, torch.ones_like(bfh)), torch.zeros_like(bfh))
    zero = torch.zeros_like(bfh)
    yl = torch.where(kind == KIND_TUBER, y_hi, torch.where(kind == KIND_SIMPLE, y_hi, torch.where(
        (kind == KIND_CANOPY) | (kind == KIND_COMPLETE), bfh, y_grain)))
    hi = torch.where(kind == KIND_TUBER, tuber_hi, torch.where(kind == KIND_SIMPLE, simple_hi, torch.where(
        kind == KIND_CANOPY, zero, torch.where(kind == KIND_COMPLETE, torch.ones_like(bfh), hi_grain))))
    return yl, hi
