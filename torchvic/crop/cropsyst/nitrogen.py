"""N1 of the VIC-CropSyst port (docs/connectors/cropsyst.md): crop
nitrogen (Crop_nitrogen_V5 of the shipped build) and how it limits growth.

Sources (CS = VIC-CropSyst-Package/CropSyst/source):
  construction, initialize   CS/crop/crop_N_common.cpp:48-114, 143-216; crop_N_V5.cpp:5-32;
                             crop_param_struct.h:742-770, crop_param_struct.cpp:13-24,
                             crop_param_class.cpp:697-757 (defaults, slope)
  season, day                crop_N_common.cpp:246-283 (start_season, start_day),
                             :460-545 (find_concentration_range, compute_dilution)
  N-limited growth           crop_N_common.cpp:546-781 (demand, potential and actual uptake,
                             nitrogen_depend_growth), :797-948 (Fract_N_demand,
                             update_limited_growth); crop_N_V5.cpp:33-72
  stresses                   crop_N_common.cpp:318-346 (leaf stress), crop_N_V5.cpp:33-54 (TUE),
                             crop_cropsyst.cpp:2781-2808, 2847-2947 (use in the crop day),
                             :1636-1650 (gaseous loss after maturity), crop_N_common.cpp:782-796
  roots per layer            crop_root.cpp:102-133, 160-185 (root densities and lengths)
  harvest residue N          crop_N_common.cpp:126-136 (residue concentration), 980-1040

Facts of the shipped build reproduced here:
- Crop_nitrogen_common::restart is never called for a new crop (initiate_accrescence,
  crop_cropsyst.cpp:2286-2323, does not call it; only dormancy recovery does, :2259-2260), so the
  canopy N mass and concentration start at 0 at emergence. The leaf stress factor is then 0.001
  (numerator < 1e-5, :336-342) until N is taken up: GAI growth is cut by 1000.
- PHENOLOGY_VERSION 2013: the concentration range is updated while the growth stage is in
  [RESTART, QUIESCENCE) (crop_N_common.cpp:264-270) from the crop's accumulated degree days.
- The N stress used by TUE and by the yield is the float32 current value of the stress sample
  (Sample_cumulative<float32>, crop_N_common.h:172); the TUE reduction uses the float64 member
  `stress_factor` (crop_N_V5.cpp:36-44), the yield ignores it (Yield_grain::know_stress,
  yield.cpp:230-237), so N limits yield only through biomass.
- Uptake does not change the canopy N computed from the expected concentration (the plant N is
  set before the soil N is removed, :730-777); uptake is shared out over the layers in proportion
  to each layer's potential uptake (find_act_uptake).
- Layers with a NaN NH4 (organic_matter.py) are skipped by the potential uptake because
  `NO3 + NH4 > 0` is false (:395-445), NO3 there included.

Layers: [..., L], index l = CropSyst layer l+1. Masses kg/m2 (N), concentrations kg N/kg biomass.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from .growth import round_f32
from . import soil_n as sn

C3, C4 = 0, 1
# crop_param_struct.cpp:13-24 (float32 arrays), index by photosynthetic pathway
N_MATURE_MAX_TO_CRIT = (0.64680851, 0.65, 0.65)
N_MATURE_MAX_TO_MIN = (0.27659574, 0.2777777, 0.2777777)
N_DILUTION_SLOPE_DEFAULT = (0.45, 0.38, 0.38)
MIN_N_CONC_SCALING = (0.69, 0.69, 0.69)          # crop_N_common.cpp:44 (float64)
IS_ZERO_TOL = 0.0000001
UPTAKE_ZERO_TOL = 0.000000001                    # crop_N_common.cpp:550-551
MAX_FIXATION = 0.0006                            # :595
STORE_TOL = 0.000000001                          # :752
LEAF_STRESS_FLOOR = 0.001                        # :340
LEAF_NUMERATOR_MIN = 0.00001                     # :339
GASEOUS_FRACTION = 0.1                           # :788
STRESS_ATTAINABLE_MIN = 0.000001                 # crop_N_V5.cpp:62
NGS_RESTART, NGS_ACCRESCENCE, NGS_ANTHESIS, NGS_FILLING, NGS_RIPENING = 4, 5, 7, 8, 9
FGS_RAPID_FRUIT_DEVELOPMENT, NGS_MATURITY, NGS_QUIESCENCE = 10, 11, 15


def _cmax(a, b):
    return torch.where(a < b, b, a)


def _cmin(a, b):
    return torch.where(b < a, b, a)


def _is_zero(x, tol=IS_ZERO_TOL):
    d = 0.0 - x
    d = torch.where(d < 0.0, -d, d)
    return d <= tol


def _f32(x):
    return round_f32(x)


class NParams(NamedTuple):
    """Per-crop [nitrogen] parameters (float32 in the reference), any batch shape."""
    fixating: torch.Tensor                 # bool
    legume_fraction: torch.Tensor
    max_uptake_kg_ha: torch.Tensor
    stubble_max_conc: torch.Tensor
    root_conc: torch.Tensor
    emergence_max_conc: torch.Tensor
    emergence_crit_conc: torch.Tensor
    paw_decrease: torch.Tensor             # PAW_where_N_uptake_rate_decreases, floored at 1e-5
    residual_ppm: torch.Tensor             # residual_N_not_available_for_uptake
    soil_n_conc_decrease: torch.Tensor     # soil_N_conc_where_N_uptake_decreases
    mature_max_conc: torch.Tensor          # dilution_curve_max_conc_mature
    dilution_slope: torch.Tensor           # provide_concentration_slope (positive)
    end_dilution_deg_day: torch.Tensor
    leaf_expansion_sensitivity: torch.Tensor
    pathway: torch.Tensor                  # long, C3 = 0


N_DEFAULTS = dict(      # crop_param_class.cpp:697-723 (C3 defaults)
    legume_fraction=1.0, max_uptake_daily_rate=5.0, stubble_max_conc=0.007, root_conc=0.007,
    emergence_max_conc=0.007, emergence_crit_conc=0.007, leaf_expansion_sensitivity=1.0,
    residual_N_not_available_for_uptake=1.0, soil_N_conc_where_N_uptake_decreases=5.0,
    PAW_where_N_uptake_rate_decreases=0.5, N_mature_max_conc=0.0235, dilution_curve_slope=0.45,
    end_dilution_curve=0.0)


def n_params(values: dict, pathway=C3, fixating=False, dtype=torch.float64, device=None) -> NParams:
    """From a dict of [nitrogen] keys of one crop (missing keys take the defaults)."""
    v = {k: values.get(k, d) for k, d in N_DEFAULTS.items()}
    f = lambda x: _f32(torch.as_tensor(float(x), dtype=dtype, device=device))
    slope = f(v["dilution_curve_slope"])
    slope = torch.where(_is_zero(slope), f(N_DILUTION_SLOPE_DEFAULT[pathway]), slope)   # :742-757
    return NParams(torch.as_tensor(bool(fixating), device=device), f(v["legume_fraction"]),
                   f(v["max_uptake_daily_rate"]), f(v["stubble_max_conc"]), f(v["root_conc"]),
                   f(v["emergence_max_conc"]), f(v["emergence_crit_conc"]),
                   _cmax(f(v["PAW_where_N_uptake_rate_decreases"]), f(0.00001)),
                   f(v["residual_N_not_available_for_uptake"]), f(v["soil_N_conc_where_N_uptake_decreases"]),
                   f(v["N_mature_max_conc"]), slope, torch.as_tensor(float(v["end_dilution_curve"]), dtype=dtype, device=device),
                   f(v["leaf_expansion_sensitivity"]), torch.as_tensor(pathway, device=device))


def stack_params(ps):
    return NParams(*[torch.stack(list(x)) for x in zip(*ps)])


def gather_params(p: NParams, index) -> NParams:
    return NParams(*[x[index] for x in p])


class NState(NamedTuple):
    canopy_n: torch.Tensor          # canopy_N_mass_vital
    root_n: torch.Tensor            # root_N_mass_vital
    canopy_conc: torch.Tensor       # canopy_concentration_vital
    root_conc: torch.Tensor         # root_concentration_vital
    store: torch.Tensor             # store_persistent
    n_max: torch.Tensor             # N_max_concentration (current)
    n_crit: torch.Tensor
    n_min: torch.Tensor
    max_change: torch.Tensor        # N_*_conc_at_change
    crit_change: torch.Tensor
    min_change: torch.Tensor
    max_mature: torch.Tensor        # N_*_conc_at_maturity
    crit_mature: torch.Tensor
    min_mature: torch.Tensor
    before_change: torch.Tensor     # bool
    limited_growth: torch.Tensor    # N_limited_growth [kg/m2]
    limited_pot: torch.Tensor       # N_limited_pot_transpiration [m]
    stress: torch.Tensor            # stress_factor (float64 member)
    stress32: torch.Tensor          # stress_factor_stats current (float32 value)
    max_uptake_per_length: torch.Tensor
    uptake: torch.Tensor            # daily uptake (soil N actually removed)
    fixation: torch.Tensor
    gaseous: torch.Tensor


def initial_state(shape, dtype=torch.float64, device=None) -> NState:
    z = torch.zeros(shape, dtype=dtype, device=device)
    return NState(z, z, z, z, z, z, z, z, z, z, z, z, z, z, torch.ones(shape, dtype=torch.bool, device=device),
                  z, z, z + 1.0, z + 1.0, z, z, z, z)


def mature_concentrations(P: NParams):
    """N_*_conc_at_maturity (crop_N_common.cpp:107-109, crop_param_struct.h:742-751): float32."""
    mx = P.mature_max_conc
    mult = lambda t: torch.where(P.pathway == C3, t[0], t[1]).to(mx.dtype)
    crit = _f32(mx * _f32(mult(N_MATURE_MAX_TO_CRIT)))
    mn = _f32(mx * _f32(mult(N_MATURE_MAX_TO_MIN)))
    return mx, crit, mn


def sow(n: NState, mask, P: NParams) -> NState:
    """A new crop: constructor + initialize + start_season (crop_N_common.cpp:48-114, 143-216,
    246-257)."""
    z = torch.zeros_like(n.canopy_n)
    mx, crit, mn = mature_concentrations(P)
    fresh = NState(z, z, z, z, z, z, z, z, z, z, z, mx + z, crit + z, mn + z,
                   torch.ones_like(mask), z, z, z + 1.0, z + 1.0, z, z, z, z)
    return NState(*[torch.where(mask, a, b) for a, b in zip(fresh, n)])


def _emergence_concentrations(P: NParams):
    mx = P.emergence_max_conc
    crit = P.emergence_crit_conc
    mn = crit * MIN_N_CONC_SCALING[0]
    return mx, crit, mn


def _dilution(n: NState, P: NParams, canopy_biomass_vital):
    """compute_dilution (crop_N_common.cpp:529-545): only when the biomass is not 0."""
    t_ha = canopy_biomass_vital * 10.0
    nz = t_ha != 0.0
    mx, crit, mn = _emergence_concentrations(P)
    slope = -P.dilution_slope
    pw = torch.pow(torch.where(nz, t_ha, torch.ones_like(t_ha)), slope)
    d = lambda e: _cmin(e + torch.zeros_like(t_ha), e * pw)
    return (torch.where(nz, d(mx), n.n_max), torch.where(nz, d(crit), n.n_crit), torch.where(nz, d(mn), n.n_min))


def start_day(n: NState, P: NParams, on, stage, gdds, maturity_dd, maturity_significant, canopy_biomass_vital,
              roots_exist, total_root_lengths) -> NState:
    """Crop_nitrogen_common::start_day (crop_N_common.cpp:259-283) for the tiles with a crop
    (`on`): the concentration range while in the growing season, the maximum uptake per root
    length. maturity_dd: phenology initiation.maturity; gdds: accumulated degree days."""
    season = on & (stage >= NGS_RESTART) & (stage < NGS_QUIESCENCE)
    w = lambda a, b: torch.where(season, a, b)
    # find_concentration_range (:460-528)
    early = n.before_change                                    # start_change_today is always false here
    start_change = early & (P.end_dilution_deg_day != 0.0) & (gdds >= P.end_dilution_deg_day)
    d_max, d_crit, d_min = _dilution(n, P, canopy_biomass_vital)
    mx_m, crit_m, mn_m = mature_concentrations(P)
    begin = P.end_dilution_deg_day
    rel = 1.0 - (maturity_dd - gdds) / (maturity_dd - begin)
    lin = lambda ch, mat: torch.where(ch > mat, ch - (ch - mat) * rel, ch)
    late = ~early & maturity_significant
    n_max = torch.where(early, d_max, torch.where(late, lin(n.max_change, n.max_mature), d_max))
    n_crit = torch.where(early, d_crit, torch.where(late, lin(n.crit_change, n.crit_mature), d_crit))
    n_min = torch.where(early, d_min, torch.where(late, lin(n.min_change, n.min_mature), d_min))
    max_change = torch.where(start_change, n_max, n.max_change)
    crit_change = torch.where(start_change, n_crit, n.crit_change)
    min_change = torch.where(start_change, n_min, n.min_change)
    max_mature = torch.where(start_change, mx_m + torch.zeros_like(n_max), n.max_mature)
    crit_mature = torch.where(start_change, crit_m + torch.zeros_like(n_max), n.crit_mature)
    min_mature = torch.where(start_change, mn_m + torch.zeros_like(n_max), n.min_mature)
    before = n.before_change & ~start_change
    rate_m2 = P.max_uptake_kg_ha / 10000.0
    mupl = torch.where(roots_exist, rate_m2 / total_root_lengths, torch.zeros_like(total_root_lengths))
    z = torch.zeros_like(n.canopy_n)
    return n._replace(n_max=w(n_max, n.n_max), n_crit=w(n_crit, n.n_crit), n_min=w(n_min, n.n_min),
                      max_change=w(max_change, n.max_change), crit_change=w(crit_change, n.crit_change),
                      min_change=w(min_change, n.min_change), max_mature=w(max_mature, n.max_mature),
                      crit_mature=w(crit_mature, n.crit_mature), min_mature=w(min_mature, n.min_mature),
                      before_change=w(before, n.before_change),
                      max_uptake_per_length=torch.where(on, mupl, n.max_uptake_per_length),
                      uptake=torch.where(on, z, n.uptake), fixation=torch.where(on, z, n.fixation),
                      gaseous=torch.where(on, z, n.gaseous))


def root_lengths(root_length, root_fract, thickness, max_depth, start_depth, surface_density):
    """Crop_root_vital::update_lengths (crop_root.cpp:160-185) with the densities of
    update_root_densities (:102-121): per-layer root lengths [m root/m2] and their total.
    root_fract [..., L] are the total_fract_root_length (layer index 0 = CropSyst layer 1)."""
    L = thickness.shape[-1]
    mrd = surface_density * root_length / (max_depth - start_depth)
    f2 = root_fract[..., 1]
    remaining = root_length
    lengths = []
    total = torch.zeros_like(root_length)
    zero = torch.zeros_like(root_length)
    for l in range(L):
        if l == 0:
            lengths.append(zero)
            continue
        go = remaining > 0.0
        th = thickness[..., l]
        lr = torch.where(remaining > th, th, remaining)
        dens = mrd * root_fract[..., l] / f2
        rl = (dens * 10000) * lr
        lengths.append(torch.where(go, rl, zero))
        total = torch.where(go, total + rl, total)
        remaining = torch.where(go, remaining - lr, remaining)
    return torch.stack(lengths, -1), total


def root_activity_factor(after_accrescence, perennial, peak_lai, gai_vital):
    """Canopy_growth_leaf_area_index_based::get_influence_on_root_activity_factor
    (canopy_growth_LAI_based.cpp:537-550)."""
    use = ~perennial & after_accrescence & (peak_lai > 0.0)
    r = gai_vital / torch.where(peak_lai > 0.0, peak_lai, torch.ones_like(peak_lai))
    return torch.where(use, _cmin(r, torch.ones_like(r)), torch.ones_like(r))


def fract_n_demand(P: NParams, perennial, stage, gdds, flowering_dd, filling_dd, maturity_dd, paw, thickness,
                   root_fract, no3_n, nh4_n, soil_t2):
    """Fract_N_demand (crop_N_common.cpp:797-891): the legume fixation share of the demand;
    0 for non-fixating crops."""
    dev = torch.ones_like(gdds)
    dev = torch.where(~perennial & (stage == NGS_ACCRESCENCE), gdds / flowering_dd, dev)
    end = (maturity_dd + filling_dd) / 2.0
    fill = (stage == NGS_FILLING) | (stage == NGS_RIPENING) | (stage == FGS_RAPID_FRUIT_DEVELOPMENT)
    dev = torch.where(~perennial & fill, torch.where(gdds <= end, 1.0 - (gdds - filling_dd) / (end - filling_dd),
                                                    torch.zeros_like(gdds)), dev)
    L = thickness.shape[-1]
    depth = thickness[..., 0]
    cnt = torch.zeros_like(gdds)
    s = torch.zeros_like(gdds)
    go = torch.ones_like(gdds, dtype=torch.bool)
    for l in range(L):                      # sublayer = 1..L while depth < 0.5 (layer 1 excluded)
        go = go & (depth < 0.5)
        if l != 0:
            cnt = torch.where(go, cnt + 1.0, cnt)
            s = torch.where(go, s + paw[..., l], s)
        depth = torch.where(go, depth + thickness[..., l], depth)
    avg = torch.where(cnt > 0, s / torch.where(cnt > 0, cnt, torch.ones_like(cnt)), torch.zeros_like(cnt))
    water = _cmin(torch.ones_like(avg), _cmax(torch.zeros_like(avg), (avg - 0.3) / 0.3))
    contrib = torch.where(root_fract > 0.0, (root_fract / thickness) * (no3_n + nh4_n), torch.zeros_like(root_fract))
    idx = torch.zeros_like(gdds)
    for l in range(L):
        idx = idx + contrib[..., l]
    nitro = torch.where(idx > 0.03, torch.zeros_like(idx), torch.where(idx < 0.01, torch.ones_like(idx),
                                                                        1.0 - ((idx - 0.01) / 0.02)))
    t = soil_t2
    temp = torch.where(t > 36, torch.full_like(t, 0.7), torch.where(t > 30, torch.ones_like(t),
                       torch.where(t < 0, torch.zeros_like(t), t / 30.0)))
    fnd = dev * _cmin(_cmin(water, nitro), temp)
    return torch.where(P.fixating, fnd, torch.zeros_like(fnd))


def _allocate(to_alloc, mass, demand):
    """allocate_available_to_plant_part_as_demanded (crop_N_common.cpp:1055-1075)."""
    more = to_alloc > demand
    give = torch.where(more, demand, to_alloc)
    rest = torch.where(more, to_alloc - demand, to_alloc - to_alloc)
    dem = torch.where(more, torch.zeros_like(demand), demand - to_alloc)
    return rest, mass + give, dem


def depend_growth(n: NState, P: NParams, attainable, raf, fnd, canopy_biomass_vital, root_biomass,
                  root_growth_pot, lengths, paw, layers: sn.SoilLayers, s: sn.SoilN, auto_plant=None):
    """Crop_nitrogen_common::nitrogen_depend_growth (crop_N_common.cpp:693-781) on every tile
    passed (the caller masks). Returns (state, soil N after uptake, N-limited growth).
    auto_plant: automatic_nitrogen_to_plant, the management's automatic_nitrogen set at sowing
    (perform_op.cpp:526-528, crop_N_common.h:195): a canopy below the critical concentration is
    brought to it without uptake (:734-741)."""
    canopy_y = n.canopy_n
    # find_plant_demand (:546-626)
    root_max = P.root_conc
    n_root_growth = root_max * root_growth_pot
    rb_before = root_biomass - root_growth_pot
    root_conc = torch.where(~_is_zero(rb_before), n.root_n / torch.where(_is_zero(rb_before), torch.ones_like(rb_before),
                                                                         rb_before), torch.zeros_like(rb_before))
    top_new = n.n_max * attainable + 0.0
    top_def = _cmax((n.n_max - n.canopy_conc) * (canopy_biomass_vital - 0.0), torch.zeros_like(top_new))
    canopy_dem = top_def + top_new
    root_def = _cmax((root_max - root_conc) * rb_before, torch.zeros_like(top_new))
    root_dem = root_def + n_root_growth
    plant = canopy_dem + root_dem
    fix_on = fnd > 0.0
    fixation = torch.where(fix_on, _cmin(plant * fnd * P.legume_fraction, torch.full_like(plant, MAX_FIXATION)),
                           torch.zeros_like(plant))
    rest, root_n, root_dem2 = _allocate(fixation, n.root_n, root_dem)
    rest, canopy_n, canopy_dem2 = _allocate(rest, n.canopy_n, canopy_dem)
    root_n = torch.where(fix_on, root_n, n.root_n)
    canopy_n = torch.where(fix_on, canopy_n, n.canopy_n)
    root_dem = torch.where(fix_on, root_dem2, root_dem)
    canopy_dem = torch.where(fix_on, canopy_dem2, canopy_dem)
    plant = torch.where(fix_on, canopy_dem + root_dem, plant)
    more = n.store > plant
    store = torch.where(more, n.store - plant, torch.zeros_like(plant))
    plant = torch.where(more, torch.zeros_like(plant), plant - n.store)
    # find_pot_uptake (:357-450)
    th, bd = layers.thickness, layers.bulk_density
    waf = 1.0 - torch.exp(-(5.259 * torch.pow(P.paw_decrease, -1.0246))[..., None] * paw)
    resid = P.residual_ppm[..., None]
    resid_mass = (resid / 1000.0) * (th * bd)
    no3 = sn.no3_n(s)
    nh4 = sn.nh4_n(s)
    no3_ppm = (no3 * 1000.0) / (th * bd)
    nh4_ppm = (nh4 * 1000.0) / (th * bd)
    soil = no3 + nh4
    has = soil > 0.0
    soil_safe = torch.where(has, soil, torch.ones_like(soil))
    mpl = n.max_uptake_per_length[..., None]
    x = P.soil_n_conc_decrease - P.residual_ppm
    x = _f32(x)                                               # float32 difference (:192-194)
    no3_adj = torch.where(x > 0.00001, 4.9259 * torch.pow(torch.where(x > 0.00001, x, torch.ones_like(x)), -0.9821),
                          torch.zeros_like(x))[..., None]
    nh4_adj = (4.9259 * torch.pow(P.soil_n_conc_decrease, -0.9821))[..., None]
    raf_ = raf[..., None]
    p3 = lengths * _cmax(1.0 - torch.exp(-(no3_ppm - resid) * no3_adj), torch.zeros_like(no3)) * waf \
        * (mpl * (no3 / soil_safe)) * raf_
    avail3 = no3 - resid_mass
    p3 = torch.where(p3 > avail3, avail3, p3)
    p3 = torch.where(has & (no3_ppm > resid), p3, torch.zeros_like(p3))
    p4 = lengths * _cmax(1.0 - torch.exp(-(nh4_ppm) * nh4_adj), torch.zeros_like(nh4)) * waf \
        * (mpl * (nh4 / soil_safe)) * raf_
    p4 = torch.where(p4 > nh4, nh4, p4)
    p4 = torch.where(has & (nh4_ppm > 0.0), p4, torch.zeros_like(p4))
    first = torch.arange(p3.shape[-1], device=p3.device) >= 1              # soil_rooting_sublayer = 2
    p3 = torch.where(first, p3, torch.zeros_like(p3))
    p4 = torch.where(first, p4, torch.zeros_like(p4))
    pot = torch.zeros_like(attainable)
    for l in range(1, p3.shape[-1]):
        pot = torch.where(has[..., l], pot + (p3[..., l] + p4[..., l]), pot)
    # expected concentration (:712-760)
    new_bm = canopy_biomass_vital + attainable
    expected = _cmin(plant, pot)
    unable = expected - root_dem
    unable = torch.where(unable < 0, torch.zeros_like(unable), unable)
    conc_exp = (canopy_n + unable) / new_bm
    root_n = root_n + root_dem
    low = conc_exp < n.n_crit
    high = ~low & (conc_exp > n.n_max + STORE_TOL)
    stored = new_bm * (conc_exp - n.n_max)
    store = torch.where(high, store + stored, store)
    canopy_conc = torch.where(high, n.n_max, conc_exp)
    canopy_n = torch.where(high, new_bm * n.n_max, new_bm * canopy_conc)
    required = torch.zeros_like(canopy_n)
    if auto_plant is not None:
        fert = low & auto_plant
        required = torch.where(fert, new_bm * (n.n_crit - conc_exp), required)
        canopy_n = torch.where(fert, new_bm * n.n_crit, canopy_n)
        canopy_conc = torch.where(fert, n.n_crit, canopy_conc)
    act_dem = canopy_n - canopy_y + root_dem - required
    # find_act_uptake (:629-664)
    do = ~(_is_zero(pot, UPTAKE_ZERO_TOL) | _is_zero(act_dem, UPTAKE_ZERO_TOL))
    ratio = act_dem / torch.where(do, pot, torch.ones_like(pot))
    req3 = torch.where(do[..., None] & first & (pot > 0.0)[..., None], p3 * ratio[..., None], torch.zeros_like(p3))
    req4 = torch.where(do[..., None] & first & (pot > 0.0)[..., None], p4 * ratio[..., None], torch.zeros_like(p4))
    s2, t3, t4 = sn.uptake(s, req3, req4)
    take = do[..., None] & first & (pot > 0.0)[..., None]
    s2 = sn.SoilN(torch.where(take, s2.no3, s.no3), torch.where(take, s2.nh4, s.nh4))
    up3 = torch.zeros_like(pot)
    up4 = torch.zeros_like(pot)
    for l in range(1, p3.shape[-1]):
        up3 = torch.where(take[..., l], up3 + t3[..., l], up3)
        up4 = torch.where(take[..., l], up4 + t4[..., l], up4)
    uptake = torch.where(do, up4 + up3, torch.zeros_like(pot))
    # find_actual_growth (:666-691)
    ok = n.n_crit != n.n_min
    rf = 1.0 - (n.n_crit - canopy_conc) / torch.where(ok, n.n_crit - n.n_min, torch.ones_like(canopy_conc))
    rf = _cmin(_cmax(rf, torch.zeros_like(rf)), torch.ones_like(rf))
    lim = torch.where(canopy_conc >= n.n_crit, torch.ones_like(rf), rf)
    growth = torch.where(ok, attainable * lim, torch.zeros_like(attainable))
    n2 = n._replace(canopy_n=canopy_n, root_n=root_n, canopy_conc=canopy_conc, root_conc=root_conc, store=store,
                    uptake=uptake, fixation=torch.where(fix_on, fixation, n.fixation))
    return n2, s2, growth


def gaseous_losses(n: NState, mask, canopy_biomass_vital) -> NState:
    """process_any_gaseous_losses (crop_N_common.cpp:782-796)."""
    over = n.canopy_conc > n.n_max
    pot = canopy_biomass_vital * (n.canopy_conc - n.n_max)
    act = _cmax(pot * GASEOUS_FRACTION, torch.zeros_like(pot))
    act = torch.where(act > n.canopy_n, torch.zeros_like(act), act)
    act = torch.where(over, act, torch.zeros_like(act))
    cn = torch.where(over, n.canopy_n - act, n.canopy_n)
    cc = torch.where(over, cn / canopy_biomass_vital, n.canopy_conc)
    w = lambda a, b: torch.where(mask, a, b)
    return n._replace(canopy_n=w(cn, n.canopy_n), canopy_conc=w(cc, n.canopy_conc), gaseous=w(act, n.gaseous))


def update_limited_growth(n: NState, P: NParams, on, attainable, raf, perennial, maturity_significant, gdds,
                          maturity_dd, fnd, canopy_biomass_vital, root_biomass, root_growth_pot, lengths, paw,
                          layers: sn.SoilLayers, s: sn.SoilN, tue_co2_adjusted, auto_plant=None):
    """Crop_nitrogen_V5::update_limited_growth + update_limited_pot_transpiration
    (crop_N_V5.cpp:33-72, crop_N_common.cpp:892-948, crop_cropsyst.cpp:2781-2795) for the tiles
    `on` (canopy and transpiration exist). Returns (state, soil N, N-limited growth,
    N-limited potential transpiration [m])."""
    run = on & (attainable > 0.0) & (perennial | ~maturity_significant | (maturity_significant & (gdds < maturity_dd)))
    n2, s2, g = depend_growth(n, P, attainable, raf, fnd, canopy_biomass_vital, root_biomass, root_growth_pot,
                              lengths, paw, layers, s, auto_plant)
    w = lambda a, b: torch.where(run, a, b)
    n3 = NState(*[w(a, b) for a, b in zip(n2, n)])
    s3 = sn.SoilN(torch.where(run[..., None], s2.no3, s.no3), torch.where(run[..., None], s2.nh4, s.nh4))
    limited = torch.where(run, g, attainable)
    gas = on & ~run & ~maturity_significant
    n3 = gaseous_losses(n3, gas, canopy_biomass_vital)
    stress = torch.where(attainable > STRESS_ATTAINABLE_MIN, limited / torch.where(attainable > STRESS_ATTAINABLE_MIN,
                                                                                  attainable, torch.ones_like(attainable)),
                         torch.ones_like(attainable))
    red = torch.where(stress < 1.0, _cmax(-0.201 * stress * stress + 0.3258 * stress + 0.8604, torch.zeros_like(stress)),
                      torch.ones_like(stress))
    pot = (limited / (tue_co2_adjusted * red)) / 1000.0
    n3 = n3._replace(limited_growth=torch.where(on, limited, n.limited_growth),
                     stress=torch.where(on, stress, n.stress), stress32=torch.where(on, _f32(stress), n.stress32),
                     limited_pot=torch.where(on, pot, n.limited_pot))
    return n3, s3, limited, pot


def leaf_stress_factor(n: NState, P: NParams):
    """calc_leaf_stress_factor (crop_N_common.cpp:318-346)."""
    den = n.n_crit - n.n_min
    num = n.canopy_conc - n.n_min
    floor = _is_zero(den) | (num < LEAF_NUMERATOR_MIN)
    f = torch.pow(num / torch.where(floor, torch.ones_like(den), den), P.leaf_expansion_sensitivity)
    f = torch.where(floor, torch.full_like(f, LEAF_STRESS_FLOOR), f)
    return _cmin(_cmax(f, torch.zeros_like(f)), torch.ones_like(f))


def actual_growth(attainable, n_limited_pot, n_limited_growth, act_transp, lwp, reduce_lwp, stop_lwp, leaf_stress):
    """calc_act_biomass_growth with Crop_nitrogen_V5 and the TUE model (crop_cropsyst.cpp:
    2847-2947): returns (actual growth, GAI-related growth, water canopy expansion factor,
    overall stress)."""
    pos = n_limited_pot > 0.0
    wsf = torch.where(pos, act_transp / torch.where(pos, n_limited_pot, torch.ones_like(n_limited_pot)),
                      torch.ones_like(n_limited_pot))
    act = torch.where(pos, wsf * n_limited_growth, torch.zeros_like(attainable))
    w = (lwp - stop_lwp) / (reduce_lwp - stop_lwp)
    w = _cmin(_cmax(w, torch.zeros_like(w)), torch.ones_like(w))
    gai_rel = torch.where(pos, (act * w) * leaf_stress, torch.zeros_like(act))
    eps = 0.000001
    overall = torch.where(attainable > eps, act / torch.where(attainable > eps, attainable, torch.ones_like(attainable)),
                          torch.ones_like(attainable))
    return act, gai_rel, torch.where(pos, w, torch.full_like(w, -1.0)), overall


def residue_concentration(n: NState, P: NParams):
    """get_residue_concentration (crop_N_common.cpp:126-136)."""
    c = P.stubble_max_conc * n.canopy_conc / n.max_mature
    return _cmin(c, P.stubble_max_conc + torch.zeros_like(c))


# [nitrogen] key of each NParams field as the .crp files spell it (crop_param_V4 parser)
N_KEYS = dict(legume_fraction="legume_fraction", max_uptake_kg_ha="max_uptake_daily_rate",
              stubble_max_conc="stubble_max_conc", root_conc="root_conc", emergence_max_conc="emergence_max_conc",
              emergence_crit_conc="emergence_crit_conc", paw_decrease="PAW_where_N_uptake_rate_decreases",
              residual_ppm="residual_N_not_available_for_uptake",
              soil_n_conc_decrease="soil_N_conc_where_N_uptake_decreases", mature_max_conc="N_mature_max_conc",
              dilution_slope="dilution_curve_slope", end_dilution_deg_day="end_dilution_curve",
              leaf_expansion_sensitivity="leaf_expansion_sensitivity")


def n_params_from_table(crops, device=None, dtype=torch.float64) -> NParams:
    """[n_crop] NParams from a crop table (readers.crop_table); missing keys take the defaults of
    crop_param_class.cpp:697-723."""
    n = crops.n
    dev = device if device is not None else next(iter(crops.params.values())).device
    defaults = dict(N_DEFAULTS, max_uptake_daily_rate=5.0)
    def get(key):
        d = defaults[key]
        t = crops.params.get(f"nitrogen.{key}")
        full = torch.full((n,), float(d), dtype=dtype, device=dev)
        if t is None:
            return full
        t = t.to(dev)
        miss = torch.isnan(t) if t.dtype.is_floating_point else t == -1
        return torch.where(miss, full, t.to(dtype))
    v = {f: get(k) for f, k in N_KEYS.items()}
    pw = torch.zeros(n, dtype=torch.long, device=dev)
    if "crop.photosynthetic_pathway" in crops.params:
        voc = crops.vocab["crop.photosynthetic_pathway"]
        codes = crops.params["crop.photosynthetic_pathway"].to(dev)
        c4 = [i for i, w in enumerate(voc) if w == "C4"]
        if c4:
            pw = torch.where(codes == c4[0], torch.ones_like(pw), pw)
    fix = crops.params.get("nitrogen.fixation")
    fix = torch.zeros(n, dtype=torch.bool, device=dev) if fix is None else (fix.to(dev) == 1)
    f32 = {k: _f32(x) for k, x in v.items() if k != "end_dilution_deg_day"}
    slope = torch.where(_is_zero(f32["dilution_slope"]),
                        _f32(torch.where(pw == C3, N_DILUTION_SLOPE_DEFAULT[0], N_DILUTION_SLOPE_DEFAULT[1]).to(dtype)),
                        f32["dilution_slope"])
    return NParams(fix, f32["legume_fraction"], f32["max_uptake_kg_ha"], f32["stubble_max_conc"], f32["root_conc"],
                   f32["emergence_max_conc"], f32["emergence_crit_conc"],
                   _cmax(f32["paw_decrease"], torch.full_like(f32["paw_decrease"], _f32(torch.tensor(0.00001, dtype=dtype)))),
                   f32["residual_ppm"], f32["soil_n_conc_decrease"], f32["mature_max_conc"], slope,
                   v["end_dilution_deg_day"].trunc(), f32["leaf_expansion_sensitivity"], pw)


# ----------------------------------------------------------------------------- the crop day

def crop_start_day(n: NState, P: NParams, PP, GP, phn, g_before, g, layers: sn.SoilLayers) -> NState:
    """Crop_nitrogen_common::start_day alone (crop_cropsyst.cpp:1974-1976): it also runs on a
    harvest day, whose harvest operation then ends the crop before its process_day."""
    from . import canopy as cn
    c = g.canopy
    bm = torch.where(c.exists, cn.biomass_vital(c), torch.zeros_like(n.canopy_n))
    rb = g_before.root
    len_y, tot_y = root_lengths(rb.length, rb.fract, layers.thickness, GP.root_max_depth, rb.start_depth,
                                GP.root_surface_density)
    tot_y = torch.where(rb.exists, tot_y, torch.full_like(tot_y, 0.000001))
    return start_day(n, P, phn.present, phn.stage, phn.gdds, PP.maturity, PP.maturity_significant, bm,
                     g.root.exists, tot_y)


def crop_potential(n: NState, P: NParams, PP, GP, phn, g_before, g, layers: sn.SoilLayers, s: sn.SoilN, paw,
                   soil_t2, started=None, perennial=None, auto_plant=None):
    """The nitrogen part of the crop's start_day and process_day (crop_cropsyst.cpp:1974-1976,
    2779-2795), applied after daily.potential: Crop_nitrogen_common::start_day with the canopy
    biomass of the start of the day and yesterday's root lengths (1e-6 for roots created today,
    crop_root.cpp:52), then update_limited_growth / update_limited_pot_transpiration on today's
    roots. g_before / g: the growth states before / after daily.potential; phn: the phenology
    after it. Returns (state, soil N, N-limited growth, N-limited potential transpiration [m]);
    the caller replaces g.limited_pot by the last one where the canopy exists.
    started: the state after crop_start_day when the caller ran it already (orchard crops: before
    their inactive period); perennial: the crops' is_perennial (None: all annual)."""
    from . import canopy as cn
    from . import phenology as ph
    present = phn.present
    c = g.canopy
    bm = torch.where(c.exists, cn.biomass_vital(c), torch.zeros_like(n.canopy_n))
    th = layers.thickness
    n = crop_start_day(n, P, PP, GP, phn, g_before, g, layers) if started is None else started
    r = g.root
    lengths, _ = root_lengths(r.length, r.fract, th, GP.root_max_depth, r.start_depth, GP.root_surface_density)
    on = present & c.exists
    perennial = torch.zeros_like(on) if perennial is None else perennial
    in_acc = phn.period_on[..., ph.ACC]
    raf = root_activity_factor(~in_acc, perennial, c.peak_lai, cn.gai_vital(c))
    fnd = fract_n_demand(P, perennial, phn.stage, phn.gdds, PP.flowering, PP.filling, PP.maturity, paw, th,
                         r.fract, sn.no3_n(s), sn.nh4_n(s), soil_t2)
    n, s, lim, pot = update_limited_growth(n, P, on, g.attainable, raf, perennial, PP.maturity_significant,
                                           phn.gdds, PP.maturity, fnd, bm, r.biomass, r.growth_biomass, lengths, paw,
                                           layers, s, g.tue_co2_adjusted, auto_plant)
    return n, s, lim, pot


def crop_after_actual(n: NState, phs, g, fruit=None):
    """Gaseous N loss on days without growth after maturity (crop_cropsyst.cpp:1636-1650): call
    with the growth state BEFORE daily.actual (the loss is computed before the day's leaf ageing).
    Fruit trees (`fruit` mask) still grow during maturity (:1633-1640): no loss then."""
    from . import canopy as cn
    from . import phenology as ph
    c = g.canopy
    mask = phs.present & c.exists & (phs.stage >= ph.NGS_MATURITY) & (phs.stage != ph.NGS_QUIESCENCE)
    if fruit is not None:
        mask = mask & ~(fruit & (phs.stage == ph.NGS_MATURITY))
    bm = torch.where(c.exists, cn.biomass_vital(c), torch.zeros_like(n.canopy_n))
    return gaseous_losses(n, mask, bm)


def dormancy_and_restart(n: NState, P: NParams, dormant, restarted, root_biomass) -> NState:
    """The crop N side of the perennial inactive period (crop_cropsyst.cpp:762, 1906, 2259-2261):
    start_dormancy moves the vital canopy and root N to effete pools (crop_N_common.cpp:305-316;
    the effete pools are not tracked: nothing in the crop or soil N day reads them in this build);
    at the restart start_season (:246-257: the dilution curve starts again, stress statistics
    cleared to 1) and restart after dormancy (:218-245, with the null guard of patch P4 for the
    canopy the orchard dropped): canopy concentration at the emergence maximum, the vital roots get
    their N back at the root concentration (root_biomass: the roots before the fruit tree's
    restart update)."""
    z = torch.zeros_like(n.canopy_n)
    w = lambda m, a, b: torch.where(m, a, b)
    n = n._replace(canopy_n=w(dormant, z, n.canopy_n), root_n=w(dormant, z, n.root_n))
    mx, _, _ = _emergence_concentrations(P)
    r = restarted
    return n._replace(before_change=n.before_change | r, stress32=w(r, z + 1.0, n.stress32),
                      canopy_conc=w(r, mx + z, n.canopy_conc),
                      root_n=w(r, n.root_n + root_biomass * n.root_conc, n.root_n))
