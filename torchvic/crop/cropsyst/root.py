"""E8 of the VIC-CropSyst port (docs/connectors/cropsyst.md): root
length, apparent depth, root fractions per soil layer and root biomass (Crop_root_vital_V5).

Sources: crop_root.cpp:28-206 (Crop_root_vital: update_length, update_root_depth,
update_root_densities, update_fractions), crop_root_V5.cpp:8-44 (calc_biomass),
crop_cropsyst.cpp:1514-1548 (create_roots), :2715-2724 (update in process_day), :2021-2022 and
:2094-2101 (reported depth, end_day update_root_depth).

Layers: [..., L] VIC layers; VIC layer l is CropSyst layer l+1 (VIC_land_unit_simulation.cpp:
208-222), thickness = VIC depth. Roots start in CropSyst layer 2 (rooting_layer = 2,
crop_root.cpp:49) below a start depth equal to the top layer's thickness (:50), so the fraction
of VIC layer 0 is always 0 (uptake from layer 1 is never extracted anyway, see uptake.py).

Reference quirks ported as is:
- The initial root length cm_to_m(root_length_at_emergence) is passed as float32
  (crop_root_V5.h, crop_root.cpp:35) and the deprecated `initial_depth` key has no effect: its
  branch writes a shadowing local (crop_cropsyst.cpp:1519-1533).
- The vital root biomass starts at 1e-5 and never decreases (max with yesterday's,
  crop_root_V5.cpp:21, :37).
- Growth uses the water stress factor of the PREVIOUS day (get_water_stress before today's
  transpiration, crop_cropsyst.cpp:2722); factors below 1e-6 mean no adjustment (crop_root.cpp:71).
- Once the root elongation period has expired (its thermal time > duration) the length is frozen
  and so are the unstressed lengths (crop_root.cpp:59).
- The root density distribution measures depth from 0 at the top of the rooting layer
  (crop_root.cpp:117-131), not from the surface.
- Fractions are renormalised only when the front is below the middle of the bottom layer or the
  sum differs from 1 by more than 1e-9 (crop_root.cpp:136-158); is_established() is false for
  single-season crops (`!mode == single_season`, crop_cropsyst.cpp:2843-2845), so roots start
  short.
Not ported: root densities and lengths per layer (update_lengths, used only by the disabled
nitrogen model), dormant roots.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from ...core import xscan

from .growth import round_f32

ROOT_BIOMASS_INITIAL = 0.00001       # crop_root_V5.cpp:21
STRESS_MIN = 0.000001                # crop_root.cpp:71
SUM_ZERO_TOL = 1e-12                 # crop_root.cpp:139
IS_ZERO_TOL = 0.0000001
SUM_ONE_TOL = 0.000000001            # crop_root.cpp:152
PROFILE_MIN = 0.00001                # crop_root.cpp:93



def _rep(nt, **kw):
    """NamedTuple._replace without collections internals (traceable by torch.compile)."""
    return type(nt)(*[kw[f] if f in kw else getattr(nt, f) for f in type(nt)._fields])

def _cmax(a, b):
    return torch.where(a < b, b, a)


def _cmin(a, b):
    return torch.where(b < a, b, a)


class RootState(NamedTuple):
    exists: torch.Tensor
    length: torch.Tensor            # root_length [m] below the start depth
    unstr_today: torch.Tensor
    unstr_yday: torch.Tensor
    norm_today: torch.Tensor
    norm_yday: torch.Tensor
    depth: torch.Tensor             # apparent_root_depth [m]
    biomass: torch.Tensor           # [kg/m2]
    biomass_yday: torch.Tensor
    growth_biomass: torch.Tensor
    above_ground: torch.Tensor
    start_depth: torch.Tensor       # start_rooting_depth = thickness of the top layer
    fract: torch.Tensor             # [..., L] total_fract_root_length per VIC layer


def initial_state(shape, L, dtype=torch.float64, device=None) -> RootState:
    z = torch.zeros(shape, dtype=dtype, device=device)
    return RootState(torch.zeros(shape, dtype=torch.bool, device=device), z, z, z, z, z, z, z, z, z,
                     z, z, torch.zeros(tuple(shape) + (L,), dtype=dtype, device=device))


def create(r: RootState, mask, length_at_emergence_cm, thickness) -> RootState:
    """create_roots + Crop_root_vital_V5 constructor."""
    l0 = round_f32(length_at_emergence_cm / 100.0) + torch.zeros_like(r.length)   # float32 argument
    w = lambda new, old: torch.where(mask, new, old)
    z = torch.zeros_like(r.length)
    return RootState(r.exists | mask, w(l0, r.length), w(l0, r.unstr_today), w(l0, r.unstr_yday),
                     w(z, r.norm_today), w(z, r.norm_yday), w(l0, r.depth),
                     w(z + ROOT_BIOMASS_INITIAL, r.biomass), w(z, r.biomass_yday), w(z, r.growth_biomass),
                     w(z, r.above_ground), w(thickness[..., 0] + z, r.start_depth),
                     torch.where(mask[..., None], torch.zeros_like(r.fract), r.fract))


def start_day(r: RootState) -> RootState:
    """Crop_root_vital::start_day (crop_root.cpp:82-87)."""
    e = r.exists
    return _rep(r, norm_yday=torch.where(e, r.norm_today, r.norm_yday),
                      unstr_yday=torch.where(e, r.unstr_today, r.unstr_yday),
                      biomass_yday=torch.where(e, r.biomass, r.biomass_yday))


def apparent_depth(length, start_depth, thickness):
    """update_root_depth (crop_root.cpp:88-100)."""
    d = torch.where(length > 0.0, length + start_depth, torch.zeros_like(length))
    lim = xscan.cumsum_last(thickness)
    return torch.where(lim > PROFILE_MIN, _cmin(d, lim), d)


def update(r: RootState, stress, above_ground, period_tt, dur, rel_remaining, expired, thickness,
           max_depth, surface_density, curvature, sensitivity, length_at_emergence_cm,
           rs_emergence, rs_full) -> RootState:
    """know_above_ground_biomass + Crop_root_vital::update (crop_root.cpp:186-200)."""
    e = r.exists
    # update_length (crop_root.cpp:54-81)
    norm = period_tt / dur
    len_em = length_at_emergence_cm / 100.0
    gain_max = (max_depth - r.start_depth) - len_em
    unstr = len_em + gain_max * norm
    adj = torch.where(stress < STRESS_MIN, torch.ones_like(stress), torch.pow(stress, sensitivity))
    gain = (unstr - r.unstr_yday) * adj
    gain = torch.where(gain < 0.0, torch.zeros_like(gain), gain)
    rl = r.length + gain
    rl = _cmin(rl, max_depth - r.start_depth)
    grow = e & ~expired
    rl = torch.where(grow, rl, r.length)
    unstr = torch.where(grow, unstr, r.unstr_today)
    norm = torch.where(grow, norm, r.norm_today)
    depth = torch.where(e, apparent_depth(rl, r.start_depth, thickness), r.depth)
    # update_root_densities (crop_root.cpp:102-133): layers 2..L, top depth from 0
    L = thickness.shape[-1]
    mrl = max_depth - r.start_depth
    mrd = surface_density * rl / mrl
    th = thickness[..., 1:]
    top = torch.cat([torch.zeros_like(th[..., :1]), xscan.cumsum(th)[..., :-1]], -1)
    ex = torch.exp(curvature[..., None] * (rl[..., None] - top) / surface_density[..., None])
    dist = -mrd[..., None] * (1.0 - ex)
    dist = torch.where(dist < 0.0, torch.zeros_like(dist), dist)
    s = xscan.cumsum_last(dist)                  # sequential, as the layer loop
    # update_fractions (crop_root.cpp:134-160)
    nz = ~(torch.abs(0.0 - s) <= SUM_ZERO_TOL)
    fr = dist / torch.where(nz, s, torch.ones_like(s))[..., None]
    sf = xscan.cumsum_last(fr)
    sf = torch.where(torch.abs(0.0 - sf) <= IS_ZERO_TOL, torch.ones_like(sf), sf)
    depth_l = xscan.cumsum_last(thickness)
    lim = depth_l - thickness[..., -1] / 2.0
    renorm = ((rl + thickness[..., 0]) > lim) | ~(torch.abs(1.0 - sf) <= SUM_ONE_TOL)
    fr = torch.where(renorm[..., None], fr * (1.0 / sf)[..., None], fr)
    fr = torch.where(nz[..., None], fr, torch.zeros_like(fr))
    fract = torch.cat([torch.zeros_like(fr[..., :1]), fr], -1) if L > 1 else torch.zeros_like(thickness)
    # calc_biomass V5 (crop_root_V5.cpp:24-44)
    ratio = torch.where(expired, rs_full + torch.zeros_like(rl),
                        rs_full + (rs_emergence.to(torch.float32) - rs_full.to(torch.float32)).to(rl.dtype)   # float32 arithmetic
                    * torch.pow(rel_remaining, 3.0))
    calc = above_ground * ratio
    bm = _cmax(calc, r.biomass)
    gb = torch.where(bm > r.biomass_yday, bm - r.biomass_yday, torch.zeros_like(bm))
    w = lambda new, old: torch.where(e, new, old)
    return RootState(e, w(rl, r.length), w(unstr, r.unstr_today), r.unstr_yday, w(norm, r.norm_today),
                     r.norm_yday, depth, w(bm, r.biomass), r.biomass_yday, w(gb, r.growth_biomass),
                     w(above_ground, r.above_ground), r.start_depth,
                     torch.where(e[..., None], fract, r.fract))
