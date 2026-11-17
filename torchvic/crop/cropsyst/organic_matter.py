"""N3 of the VIC-CropSyst port (docs/connectors/cropsyst.md): crop
residues and soil organic matter (multiple_SOM_multiple_residues), as the shipped VIC build
runs them.

Sources (CS = VIC-CropSyst-Package/CropSyst/source, RES = .../common/residue):
  pools list, day          CS/organic_matter/OM_pools_common.cpp:276-350 (start_day, detrition),
                           :310-317 (process), :618-699 (potential/actual decomposition),
                           :556-616 (carbon transfer), :296-308, 363-382 (soil N update),
                           :701-744 (reduction factor)
  residue pool             CS/organic_matter/OM_common.cpp:47-112, 183-296, 348-491,
                           multiple_pool/OM_multiple_pool.cpp:130-274
  residues                 RES/residues_cycling.cpp:42-206 (new plant residue), 377-409 (exhaust),
                           429-468 (receivers), RES/residue_V4_common.cpp:27-62, 369-381,
                           384-666 (tillage), RES/residues_common.cpp:115-155 (cover, contact)
  mass arithmetic          common/biomatter/biomass.cpp:5-17, biomass_abstract.cpp:60-102,
                           common/physics/property_cowl_T.h:67-80, mass.cpp:84-110
  parameters               RES/residue_decomposition_param.cpp:48-207, residue_const.h,
                           common/biomatter/decomposition_const.h:5, OM_const.h
  land unit                CS/land_unit_sim.cpp:1412-1413, 1825-1826, 1886-1887, 2191-2271

What the shipped VIC build does (and this module reproduces):
- Soil organic matter is created in Land_unit_simulation::initialize, before the VIC soil sets the
  bulk density, texture and organic fraction (setup_soil_property_from_VIC runs in start(),
  VIC_land_unit_simulation.cpp:1047-1053), with the scenario's [%OM] (0 in the CRB scenarios).
  The SOM pools therefore start empty, and an empty pool can never gain mass: Biomass::add_mass is
  a no-op when the mass is 0 (biomass.cpp:9-17) and receive_carbon goes through it
  (OM_common.cpp:230-245). The SOM pools are inert for the whole run; they only serve as the
  (empty) receivers of residue carbon with their C/N ratios (scenario microbial 8, metastable 15).
  The dump of the N-on runs confirms empty SOM pools every day. They are not stored here.
- The build does not define CROPSYST_PROPER, so the residue humification fraction is NaN
  (residues_cycling.cpp:433-447). Every decomposing residue pool then has a NaN net N
  mineralization (OM_common.cpp:400-439: assimilated N = NaN * C), which is added to the NH4 of its
  layer (layer 1 for surface residues): NH4 turns NaN there for good. The carbon to deduct is
  CO2 + NaN, so `carbon_to_deduct > 0` is false and the residue pools never lose carbon to
  decomposition (OM_multiple_pool.cpp:264-274); the 55 % to CO2 is still reported and drives
  denitrification (soil_n.denitrification). Immobilization demand stays 0 (NaN < 0 is false), so
  the decomposition reduction factor is 1.
- All surface residue pools point at Residues_common::flat_surface_contact_fraction; adding a
  standing-stubble pool assigns the stubble contact fraction (0 after clear_all) to that shared
  variable through a reference (residues_cycling.cpp:84-91), so after a harvest the surface
  contact is 0 until the next daily balance (residues_common.cpp:144-153). Subsurface pools use
  the pools' subsurface contact fraction 1 (OM_pools_common.cpp:622-626).
- All pools are created with RUSLE2 type UNKNOWN (OM_common.cpp:148 ignores the argument), so
  the tillage flatten and burial ratios of index 0 apply.
- Detritis of surface residue goes to soil layer 2 with the abiotic environment of layer 2;
  buried flat residue keeps the layer-1 environment of the surface pool it was cloned from;
  pools merge only when type, position, layer, cycling and C fraction, C/N, decomposition and
  detrition constants agree within 1e-6 (OM_common.cpp:64-77). Tillage sifting moves the last
  tilled layer's outflow nowhere (it is dropped, residue_V4_common.cpp:465-503) and a layer
  without a pool passes the incoming clone on unchanged, keeping the layer it came from (:497-503).

Layout: two ordered pool lists per tile, surface (positions flat/stubble, [..., S]) and
subsurface ([..., B]). The C++ list interleaves them, but every match/merge/order-dependent
operation only involves one of the two lists; only floating sums mix them (their order differs
from the C++ list by a permutation, i.e. at the 1e-16 level). New pools go to the end of their
list; deleted slots are compacted at the start of each day (stable). A static loop over the S
surface slots implements the order-dependent detrition and tillage loops.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from .growth import round_f32
from . import soil_n as sn

STRAW, MANURE = 0, 1                     # OM_types.h:30-36
SUBSURFACE, FLAT, STUBBLE, ATTACHED = 0, 1, 2, 3   # OM_types.h:85-91
FAST, SLOW, LIGNIFIED = 1, 2, 3          # OM_types.h:68-75
MICROBIAL_CN_DEFAULT = 8.0               # OM_params.cpp:65 (scenario microbial_CN_ratio)
METASTABLE_CN_DEFAULT = 15.0             # scenario metastable_active_SOM_CN_ratio
C_FRACTION_RESIDUE_TO_CO2 = 0.55         # residues_cycling.cpp:439
HUMIFICATION_FRACTION = float("nan")     # residues_cycling.cpp:443-447 without CROPSYST_PROPER
NEGLIGIBLE = 0.0000001                   # OM_pools_common.cpp:286, residues_cycling.cpp:384
DETRITION_MIN = 0.000000001              # OM_common.cpp:102
APPROX = 0.000001                        # OM_common.cpp:72-75, property_cowl_T.h:80
DETRITIS_LAYER = 2                       # OM_common.cpp:105
STRAW_CN_DEFAULT = 50.0                  # residue_const.h:14 (straw_carbon_nitrogen_ratio_V4_4)
FLAT_CONTACT_BASE, FLAT_CONTACT_RANGE = 0.3, 0.7   # residues_common.cpp:152-153
STUBBLE_CONTACT_AFTER_CLEAR = 0.0        # residues_common.cpp:96 (clear_all at initialize_residues)


def _cmin(a, b):
    return torch.where(b < a, b, a)


def _approx(a, b, tol):
    """CORN::is_approximately (compare.hpp:41-47)."""
    d = b - a
    d = torch.where(d < 0.0, -d, d)
    return d <= tol


class Pools(NamedTuple):
    """An ordered residue pool list, every field [..., N]; n [...] is the next free slot."""
    active: torch.Tensor
    type: torch.Tensor
    position: torch.Tensor
    layer: torch.Tensor
    cycling: torch.Tensor
    env: torch.Tensor          # 1-based layer of the abiotic environment the pool decomposes in
    mass: torch.Tensor
    cfrac: torch.Tensor
    cn: torch.Tensor
    k: torch.Tensor
    kdet: torch.Tensor
    apm: torch.Tensor          # area per mass (above ground only)
    n: torch.Tensor


class Residues(NamedTuple):
    sur: Pools
    sub: Pools
    flat_contact: torch.Tensor     # Residues_common::flat_surface_contact_fraction
    overflow: torch.Tensor         # bool: a list ran out of slots (result invalid)


class ResidueParams(NamedTuple):
    """Residue_decomposition_parameters of a crop ([residue] section, float32 in the reference),
    any batch shape."""
    top: torch.Tensor          # [..., 4] top fractions by cycling (index 1..3)
    root: torch.Tensor         # [..., 4] root fractions by cycling
    halflife: torch.Tensor     # [..., 4] days to 50 % decomposition by cycling
    detrition_halflife: torch.Tensor
    carbon_fraction: torch.Tensor
    stubble_apm: torch.Tensor
    flat_apm: torch.Tensor


RESIDUE_DEFAULTS = dict(       # residue_decomposition_param.cpp:55-81
    fast_cycling_top_fraction=0.15, slow_cycling_top_fraction=0.7, lignified_top_fraction=0.15,
    fast_cycling_root_fraction=0.15, slow_cycling_root_fraction=0.7, lignified_root_fraction=0.15,
    fast_cycling_decomposition_time_50=3.0, slow_cycling_decomposition_time_50=6.0,
    lignified_decomposition_time_50=30.0, detrition_time_50=720.0, carbon_fraction=0.46,
    stubble_area_covered_to_mass_ratio=2.0, area_covered_to_mass_ratio=4.0)


def residue_params(values: dict, dtype=torch.float64, device=None) -> ResidueParams:
    """From a dict of [residue] keys (missing keys take the constructor defaults); every value is
    rounded to float32 like the reference struct."""
    v = {k: values.get(k, d) for k, d in RESIDUE_DEFAULTS.items()}
    f = lambda x: round_f32(torch.as_tensor(x, dtype=dtype, device=device))
    z = f(0.0)
    top = torch.stack([z, f(v["fast_cycling_top_fraction"]), f(v["slow_cycling_top_fraction"]),
                       f(v["lignified_top_fraction"])], -1)
    root = torch.stack([z, f(v["fast_cycling_root_fraction"]), f(v["slow_cycling_root_fraction"]),
                        f(v["lignified_root_fraction"])], -1)
    hl = torch.stack([z, f(v["fast_cycling_decomposition_time_50"]), f(v["slow_cycling_decomposition_time_50"]),
                      f(v["lignified_decomposition_time_50"])], -1)
    return ResidueParams(top, root, hl, f(v["detrition_time_50"]), f(v["carbon_fraction"]),
                         f(v["stubble_area_covered_to_mass_ratio"]), f(v["area_covered_to_mass_ratio"]))


def halflife_to_constant(h):
    """convert_halflife_to_constant (decomposition_const.h:5): -ln(0.5)/h, 0 when h <= 0."""
    return torch.where(h > 0.0, -math.log(0.5) / torch.where(h > 0.0, h, torch.ones_like(h)), torch.zeros_like(h))


def empty_pools(shape, N, dtype=torch.float64, device=None) -> Pools:
    sh = tuple(shape) + (N,)
    zi = torch.zeros(sh, dtype=torch.long, device=device)
    zf = torch.zeros(sh, dtype=dtype, device=device)
    return Pools(torch.zeros(sh, dtype=torch.bool, device=device), zi, zi.clone(), zi.clone(), zi.clone(),
                 zi.clone(), zf, zf.clone(), zf.clone(), zf.clone(), zf.clone(), zf.clone(),
                 torch.zeros(tuple(shape), dtype=torch.long, device=device))


def initial_state(shape, S=96, B=192, dtype=torch.float64, device=None) -> Residues:
    """After initialize_residues (land_unit_sim.cpp:1062-1163): no residue (the CRB scenarios set
    no initial residue); flat contact 0.3 (residues_common.cpp:97)."""
    return Residues(empty_pools(shape, S, dtype, device), empty_pools(shape, B, dtype, device),
                    torch.full(tuple(shape), 0.3, dtype=dtype, device=device),
                    torch.zeros(tuple(shape), dtype=torch.bool, device=device))


def _dev(v, like):
    """v as a tensor on like's device and dtype; a Python number becomes a fill (no host copy,
    so the call stays capturable in a CUDA graph)."""
    if torch.is_tensor(v):
        return v.to(like.dtype) if v.dtype != like.dtype else v
    return torch.full((), v, dtype=like.dtype, device=like.device)


def _rep(nt, **kw):
    return type(nt)(*[kw[f] if f in kw else getattr(nt, f) for f in type(nt)._fields])


def compact(p: Pools) -> Pools:
    """Remove deleted slots keeping the order (the list after delete_current)."""
    N = p.active.shape[-1]
    idx = torch.arange(N, device=p.active.device)
    key = torch.where(p.active, idx, idx + N)
    order = torch.argsort(key, dim=-1, stable=True)
    g = lambda x: torch.gather(x, -1, order)
    n = p.active.sum(-1)
    return Pools(g(p.active), g(p.type), g(p.position), g(p.layer), g(p.cycling), g(p.env), g(p.mass),
                 g(p.cfrac), g(p.cn), g(p.k), g(p.kdet), g(p.apm), n)


def append(p: Pools, mask, type_, position, layer, cycling, env, mass, cfrac, cn, k, kdet, apm):
    """Unidirectional_list::append at the next free slot; returns (pools, overflow mask)."""
    N = p.active.shape[-1]
    over = mask & (p.n >= N)
    at = mask[..., None] & (torch.arange(N, device=p.n.device) == p.n[..., None])
    def w(field, v):
        v = _dev(v, field)
        return torch.where(at, v[..., None] if v.dim() == mask.dim() else v, field)
    q = Pools(p.active | at, w(p.type, type_), w(p.position, position), w(p.layer, layer), w(p.cycling, cycling),
              w(p.env, env), w(p.mass, mass), w(p.cfrac, cfrac), w(p.cn, cn), w(p.k, k), w(p.kdet, kdet),
              w(p.apm, apm), p.n + mask.to(p.n.dtype))
    return q, over


def append_many(p: Pools, on, type_, position, layer, cycling, env, mass, cfrac, cn, k, kdet, apm):
    """M appends at once: candidates on [..., M] in list order (field values [..., M] or
    broadcastable) land at the next free slots, as M successive `append` calls would put them;
    returns (pools, overflow)."""
    N = p.active.shape[-1]
    rank = torch.cumsum(on.to(p.n.dtype), -1) - 1 + p.n[..., None]
    ok = on & (rank < N)
    over = (on & (rank >= N)).any(-1)
    pos = torch.where(ok, rank, torch.full_like(rank, N))

    def w(field, v):
        v = torch.broadcast_to(_dev(v, field), on.shape)
        ext = torch.cat([field, field[..., :1]], -1)
        return ext.scatter(-1, pos, v)[..., :N]
    q = Pools(w(p.active, torch.ones_like(on)), w(p.type, type_), w(p.position, position), w(p.layer, layer),
              w(p.cycling, cycling), w(p.env, env), w(p.mass, mass), w(p.cfrac, cfrac), w(p.cn, cn), w(p.k, k),
              w(p.kdet, kdet), w(p.apm, apm), p.n + on.to(p.n.dtype).sum(-1))
    return q, over


def put_many(x, hit, idx, v):
    """_put for several distinct slots at once: idx [..., M] (distinct where hit)."""
    N = x.shape[-1]
    pos = torch.where(hit, idx, torch.full_like(idx, N))
    ext = torch.cat([x, x[..., :1]], -1)
    return ext.scatter(-1, pos, torch.broadcast_to(v.to(x.dtype), pos.shape))[..., :N]


def first_index(cond):
    """First True along the last axis: (found, index)."""
    found = cond.any(-1)
    idx = torch.argmax(cond.to(torch.int8), -1)
    return found, idx


def _take(x, idx):
    return torch.gather(x, -1, idx[..., None])[..., 0]


def _put(x, mask, idx, v):
    N = x.shape[-1]
    at = mask[..., None] & (torch.arange(N, device=x.device) == idx[..., None])
    v = _dev(v, x)
    return torch.where(at, v[..., None] if v.dim() == mask.dim() else v, x)


def add_mass(mass, x):
    """Biomass::add_mass (biomass.cpp:9-17): scaled by (m + x)/m, a no-op on an empty pool."""
    nz = mass != 0.0
    return torch.where(nz, mass * ((mass + x) / torch.where(nz, mass, torch.ones_like(mass))), mass)


def is_empty(mass):
    """Property_cowl::is_empty (property_cowl_T.h:79-80)."""
    return _approx(mass, torch.zeros_like(mass), APPROX)


def reset_mass(mass, new):
    """Biomass_abstract::reset_mass (biomass_abstract.cpp:66-74)."""
    e = is_empty(mass)
    safe = torch.where(e, torch.ones_like(mass), mass)
    return torch.where(e, add_mass(mass, new), mass * (1.0 + ((new - mass) / safe)))


def merge_intensive(augend_q, addend_q, addend_mass, augend_mass):
    """Physical::Mass::merge_intensive_property (mass.cpp:84-110); augend_mass is the pool's
    mass at the call (in add_similar_mass that is already the merged mass)."""
    s = addend_mass + augend_mass
    sz = _approx(s, torch.zeros_like(s), APPROX)
    safe = torch.where(sz, torch.ones_like(s), s)
    w = addend_q * torch.where(sz, torch.zeros_like(s), addend_mass / safe) \
        + augend_q * torch.where(sz, torch.zeros_like(s), augend_mass / safe)
    w = torch.where(_approx(augend_mass, torch.zeros_like(s), APPROX), addend_q, w)
    w = torch.where(_approx(addend_mass, torch.zeros_like(s), APPROX), augend_q, w)
    return torch.where(addend_q != augend_q, w, augend_q)


def add_different_mass(mass, cfrac, cn, k, apm, above, a_mass, a_cfrac, a_cn, a_k, a_apm):
    """Organic_biomatter_common::add_different_mass -> Biomass_abstract::add_similar_mass
    (OM_common.cpp:79-97, biomass_abstract.cpp:76-102) on matching tensors.
    Returns (mass, cfrac, cn, k, apm)."""
    go = a_mass > NEGLIGIBLE
    cur_c = mass * cfrac
    add_c = a_mass * a_cfrac
    cur_n = cur_c / cn
    add_n = add_c / a_cn
    m2 = mass + a_mass
    c_tot = cur_c + add_c
    n_tot = cur_n + add_n
    cn2 = c_tot / n_tot
    cf2 = torch.where(is_empty(m2), torch.zeros_like(m2), c_tot / m2)
    k2 = merge_intensive(k, a_k, a_mass, m2)
    mass_n = torch.where(go, m2, mass)
    cfrac_n = torch.where(go, cf2, cfrac)
    cn_n = torch.where(go, cn2, cn)
    k_n = torch.where(go, k2, k)
    apm_n = torch.where(above, (mass * apm + a_mass * a_apm) / (mass + a_mass), apm)
    return mass_n, cfrac_n, cn_n, k_n, apm_n


def _bc(v, f):
    """Broadcast a per-tile value (scalar or [...]) against a slot field [..., N]."""
    v = v if torch.is_tensor(v) else torch.full((), v, dtype=f.dtype, device=f.device)
    return v[..., None] if v.dim() == f.dim() - 1 else v


def comparable(p: Pools, type_, position, layer, cycling, cfrac, cn, k, kdet):
    """Organic_biomatter_common::is_comparable (OM_common.cpp:64-77) of every slot against one
    pool per tile, [..., N]."""
    a = lambda f, v: _approx(f, _bc(v, f).to(f.dtype) + torch.zeros_like(f), APPROX)
    return (p.active & (p.type == _bc(type_, p.type)) & (p.position == _bc(position, p.position))
            & (p.layer == _bc(layer, p.layer)) & (p.cycling == _bc(cycling, p.cycling))
            & a(p.cfrac, cfrac) & a(p.cn, cn) & a(p.k, k) & a(p.kdet, kdet))


# ----------------------------------------------------------------------------- the day

def start_day(r: Residues) -> Residues:
    """Organic_matter_pools_common::start_day (OM_pools_common.cpp:276-294): detrition of the
    surface residue (to layer 2), then every negligible (< 1e-7 kg/m2) residue pool is deleted."""
    sur, sub = compact(r.sur), compact(r.sub)
    over = r.overflow
    S = sur.active.shape[-1]
    for j in range(S):
        on = sur.active[..., j] & (sur.position[..., j] != SUBSURFACE) & (sur.kdet[..., j] > DETRITION_MIN)
        m = sur.mass[..., j]
        keep = torch.exp(-sur.kdet[..., j])
        dm = m * (1.0 - keep)
        sur = _rep(sur, mass=torch.where(on[..., None] & (torch.arange(S, device=m.device) == j), (m * keep)[..., None], sur.mass))
        t, c, cf, cn, k = sur.type[..., j], sur.cycling[..., j], sur.cfrac[..., j], sur.cn[..., j], sur.k[..., j]
        z = torch.zeros_like(m)
        cmp = comparable(sub, t, SUBSURFACE, DETRITIS_LAYER, c, cf, cn, k, z)
        found, idx = first_index(cmp)
        hit = on & found
        sub = _rep(sub, mass=_put(sub.mass, hit, idx, add_mass(_take(sub.mass, idx), dm)))
        sub, o = append(sub, on & ~found, t, SUBSURFACE, DETRITIS_LAYER, c, DETRITIS_LAYER, dm, cf, cn, k, z,
                        sur.apm[..., j])
        over = over | o
    neg_s = sur.active & (sur.mass < NEGLIGIBLE)
    neg_b = sub.active & (sub.mass < NEGLIGIBLE)
    return Residues(_rep(sur, active=sur.active & ~neg_s), _rep(sub, active=sub.active & ~neg_b), r.flat_contact, over)


class Decomposition(NamedTuple):
    co2: torch.Tensor              # [..., L] CO2-C loss of the day by soil layer (surface -> layer 1)
    mineralized: torch.Tensor      # [..., L] residue N mineralization by layer (NaN where residue decomposes)
    demand: torch.Tensor           # [..., L] immobilization demand by layer
    act_sur: torch.Tensor          # [..., S] actual C decomposition
    act_sub: torch.Tensor          # [..., B]


def _potential(p: Pools, contact, env: sn.Abiotic, receiver_cn, humification):
    """Organic_biomatter_pool_common::potential_C_decomposition for residue pools (OM_common.cpp:
    348-398, OM_multiple_pool.cpp:153-252). Returns (exists, pot, N mineralization, demand)."""
    L = env.tf_daily_mean.shape[-1]
    carbon = p.mass * p.cfrac
    exists = p.active & (p.mass > 0.0) & (carbon > 0.0)
    e = (p.env - 1).clamp(0, L - 1)
    tf = torch.gather(env.tf_daily_mean, -1, e)
    mf = torch.gather(env.moisture_daily, -1, e)
    pot = carbon * contact * (1.0 - torch.exp(-p.k * tf)) * mf
    ncd = torch.where(p.cn == 0.0, torch.zeros_like(p.cn), 1.0 / torch.where(p.cn == 0.0, torch.ones_like(p.cn), p.cn))
    dec_n = pot * ncd
    rcn = receiver_cn                                            # CN of the receiving SOM pool
    ncr = torch.where(rcn == 0.0, torch.zeros_like(rcn), 1.0 / torch.where(rcn == 0.0, torch.ones_like(rcn), rcn))
    assim = (humification * pot) * ncr
    net = dec_n - assim
    cn_ok = ~(p.cn < 1.0)
    demand = torch.where(cn_ok & (net < 0.0), -net, torch.zeros_like(net))
    nmin = torch.where(cn_ok & ~(net < 0.0), net, torch.zeros_like(net))
    z = torch.zeros_like(pot)
    return exists, torch.where(exists, pot, z), torch.where(exists, nmin, z), torch.where(exists, demand, z)


def _scatter_layers(v, layer1, L):
    """Sum v [..., N] into 1-based layers -> [..., L], in slot order on every device
    (core.xscan.segment_sum: the CPU's scatter_add order; CUDA's scatter_add is not reproducible)."""
    from ...core import xscan
    return xscan.segment_sum(v, (layer1 - 1).clamp(0, L - 1), L)


def reduction_factor(demand, no3_n, nh4_n):
    """Organic_matter_pools_common::determine_reduction_factor (OM_pools_common.cpp:701-744), per
    layer. Returns (factor, immobilized NH4-N, immobilized NO3-N)."""
    pos = demand > 0.0
    d = demand
    i4 = torch.where(d < nh4_n, d, nh4_n)
    d1 = torch.where(d < nh4_n, torch.zeros_like(d), d - i4)
    i3 = torch.where(d1 > 0.0, torch.where(d1 < no3_n, d1, no3_n), torch.zeros_like(d))
    d2 = torch.where(d1 > 0.0, torch.where(d1 < no3_n, torch.zeros_like(d1), d1 - i3), d1)
    safe = torch.where(pos, d, torch.ones_like(d))
    rf = torch.where(pos, (d - d2) / safe, torch.ones_like(d))
    z = torch.zeros_like(d)
    return rf, torch.where(pos, i4, z), torch.where(pos, i3, z)


def process(r: Residues, env: sn.Abiotic, s: sn.SoilN, microbial_cn=MICROBIAL_CN_DEFAULT,
            metastable_cn=METASTABLE_CN_DEFAULT, humification=HUMIFICATION_FRACTION):
    """Organic_matter_residues_profile_common::process_day (OM_residues_profile_common.cpp:246-
    257): potential decomposition, reduction factor, actual decomposition and carbon transfer,
    soil N update (tally_immobilization), then exhaust_all. `env` must be today's abiotic update.
    Returns (residues, soil N, Decomposition)."""
    L = s.no3.shape[-1]
    dt = s.no3.dtype
    def recv(p):
        return torch.where(p.cycling == LIGNIFIED, metastable_cn, microbial_cn).to(dt)
    one = torch.ones_like(r.sub.mass)
    ex_s, pot_s, nm_s, dm_s = _potential(r.sur, r.flat_contact[..., None], env, recv(r.sur), humification)
    ex_b, pot_b, nm_b, dm_b = _potential(r.sub, one, env, recv(r.sub), humification)
    lay_s = torch.ones_like(r.sur.layer)
    mineral = _scatter_layers(nm_s, lay_s, L) + _scatter_layers(nm_b, r.sub.layer, L)
    demand = _scatter_layers(dm_s, lay_s, L) + _scatter_layers(dm_b, r.sub.layer, L)
    rf, im4, im3 = reduction_factor(demand, sn.no3_n(s), sn.nh4_n(s))
    def actual(p, ex, pot, lay):
        c = p.mass * p.cfrac
        a = pot * torch.gather(rf, -1, (lay - 1).clamp(0, L - 1))
        a = _cmin(a, c)
        a = torch.where(ex, a, torch.zeros_like(a))
        co2 = a * C_FRACTION_RESIDUE_TO_CO2
        pend = co2 + a * humification
        go = ex & (pend > 0.0)
        removed = pend / torch.where(go, p.cfrac, torch.ones_like(p.cfrac))
        m = torch.where(go, add_mass(p.mass, -removed), p.mass)
        return _rep(p, mass=m), a, co2
    sur, a_s, co2_s = actual(r.sur, ex_s, pot_s, lay_s)
    sub, a_b, co2_b = actual(r.sub, ex_b, pot_b, r.sub.layer)
    co2 = _scatter_layers(co2_s, lay_s, L) + _scatter_layers(co2_b, r.sub.layer, L)
    # tally_immobilization / update_soil_N (OM_pools_common.cpp:296-308, 363-382)
    no3 = s.no3 - _cmin(im3 * sn.N_TO_NO3, s.no3)
    nh4 = s.nh4 - _cmin(im4 * sn.N_TO_NH4, s.nh4)
    nh4 = nh4 + mineral * sn.N_TO_NH4
    # Residues_cycling::exhaust_all (residues_cycling.cpp:377-409)
    ex = lambda p: p.active & _approx(p.mass, torch.zeros_like(p.mass), NEGLIGIBLE) & (p.position != ATTACHED)
    sur = _rep(sur, active=sur.active & ~ex(sur))
    sub = _rep(sub, active=sub.active & ~ex(sub))
    return Residues(sur, sub, r.flat_contact, r.overflow), sn.SoilN(no3, nh4), Decomposition(co2, mineral, demand, a_s, a_b)


def balance(r: Residues) -> Residues:
    """Residues_V4_common::balance -> Residues_common::update_totals_all (residues_common.cpp:
    115-155): flat ground cover and the shared flat-surface contact fraction for tomorrow."""
    p = r.sur
    flat = p.active & (p.position == FLAT)
    straw_flat = flat & (p.type == STRAW)
    ai = torch.where(flat, p.mass * p.apm, torch.zeros_like(p.mass)).sum(-1)
    mass = torch.where(straw_flat, p.mass, torch.zeros_like(p.mass)).sum(-1)
    ratio = torch.where(mass > 0.000001, ai / torch.where(mass > 0.000001, mass, torch.ones_like(mass)), torch.zeros_like(mass))
    cover = 1.0 - torch.exp(-0.6 * (ratio * mass))
    return _rep(r, flat_contact=FLAT_CONTACT_BASE + FLAT_CONTACT_RANGE * torch.exp(-2.0 * cover))


# ----------------------------------------------------------------------------- new residue

def calc_cn(biomass, n_conc, cfrac):
    """Residues_V4_common::calc_carbon_nitrogen_ratio (residue_V4_common.cpp:369-381)."""
    c = biomass * cfrac
    n = biomass * n_conc
    return torch.where(n > 0.000001, c / torch.where(n > 0.000001, n, torch.ones_like(n)), torch.full_like(c, STRAW_CN_DEFAULT))


def add_surface(r: Residues, mask, position, biomass, n_conc, P: ResidueParams) -> Residues:
    """Residues_cycling::add_surface_plant_pool (residues_cycling.cpp:42-127): one pool per
    cycling with mass > 0, appended (never merged); a stubble pool sets the shared flat contact
    fraction to the stubble value (0)."""
    sur, over, fc = r.sur, r.overflow, r.flat_contact
    kdet = halflife_to_constant(P.detrition_halflife)
    for c in (FAST, SLOW, LIGNIFIED):
        comp = biomass * P.top[..., c]
        on = mask & (comp > 0.0)
        k = halflife_to_constant(P.halflife[..., c])
        cn = calc_cn(comp, n_conc, P.carbon_fraction)
        apm = P.stubble_apm if position == STUBBLE else P.flat_apm
        if position == STUBBLE:
            fc = torch.where(on, torch.full_like(fc, STUBBLE_CONTACT_AFTER_CLEAR), fc)
        sur, o = append(sur, on, STRAW, position, 0, c, 1, comp, P.carbon_fraction, cn, k, kdet, apm)
        over = over | o
    return Residues(sur, r.sub, fc, over)


def add_subsurface(r: Residues, mask, root_biomass, n_conc, P: ResidueParams) -> Residues:
    """Residues_cycling::add_subsurface_plant_pool (residues_cycling.cpp:129-206): dead roots of
    each layer (> 1e-5 kg/m2) by cycling, merged into the first comparable pool or appended.
    root_biomass [..., L] by layer (layer 1 = index 0)."""
    sub, over = r.sub, r.overflow
    L = root_biomass.shape[-1]
    lay_ids = torch.arange(1, L + 1, device=root_biomass.device)
    ap = lambda f, v: _approx(f, v + torch.zeros_like(f), APPROX)
    for c in (FAST, SLOW, LIGNIFIED):
        # the pools of one cycling go to distinct layers; a pool appended for one (cycling, layer)
        # never matches another, so the layers are done together and appended in layer order
        k = halflife_to_constant(P.halflife[..., c])
        on = mask[..., None] & (root_biomass > 0.00001)
        comp = root_biomass * P.root[..., c][..., None]
        cn = calc_cn(comp, n_conc[..., None], P.carbon_fraction[..., None])
        cond = (sub.active & (sub.type == STRAW) & (sub.position == SUBSURFACE) & (sub.cycling == c)
                & ap(sub.cfrac, P.carbon_fraction[..., None]) & ap(sub.k, k[..., None])
                & ap(sub.kdet, torch.zeros_like(sub.kdet)))[..., None, :] \
            & (sub.layer[..., None, :] == lay_ids[:, None]) & ap(sub.cn[..., None, :], cn[..., None])
        found, idx = first_index(cond)
        hit = on & found
        sub = _rep(sub, mass=put_many(sub.mass, hit, idx, add_mass(torch.gather(sub.mass, -1, idx), comp)))
        sub, o = append_many(sub, on & ~found, STRAW, SUBSURFACE, lay_ids, c, lay_ids, comp,
                             P.carbon_fraction[..., None], cn, k[..., None], torch.zeros_like(comp),
                             torch.zeros_like(comp))
        over = over | o
    return Residues(r.sur, sub, r.flat_contact, over)


# ----------------------------------------------------------------------------- tillage

INVERSION_SOME_MIXING, MIXING_SOME_INVERSION, MIXING = 1, 2, 3   # SDR_field_ops.h:12-25


def tillage_layers(mask, thickness, depth, effect):
    """The per-layer part of `tillage` that does not depend on the pools: (mix [...], last tilled
    layer [...] 1-based, sifting coefficient [..., L] (:438-504), burial fraction [..., L]
    (:506-648))."""
    mix = mask & (effect >= INVERSION_SOME_MIXING) & (effect <= MIXING) & (depth > 0.0)
    L = thickness.shape[-1]
    last = sn.layer_at_depth(thickness, depth)
    bottom = sn.depths(thickness)
    coef = []
    above = torch.zeros_like(depth)
    for l in range(L):
        b = bottom[..., l]
        ratio = ((b + above) / 2.0) / torch.where(depth > 0.0, depth, torch.ones_like(depth))
        cf_ = torch.where(effect == INVERSION_SOME_MIXING, torch.full_like(ratio, 0.6),
              torch.where(effect == MIXING_SOME_INVERSION, 1.0 - (0.7539 * ratio + 0.2413),
              torch.where(effect == MIXING, 1 - (0.55152 * ratio + 0.44667), torch.zeros_like(ratio))))
        coef.append(cf_)
        above = b
    factor = torch.where(effect == MIXING_SOME_INVERSION, torch.full_like(depth, 0.5), torch.full_like(depth, 0.3))
    safe_d = torch.where(depth > 0.0, depth, torch.ones_like(depth))
    frac = []
    prev = torch.zeros_like(depth)
    top_acc = torch.zeros_like(depth)
    for l in range(L):
        b = bottom[..., l]
        is_last = (l + 1) == last
        before_last = (l + 1) < last
        f_mix = torch.where(before_last, torch.pow(b / safe_d, factor) - torch.pow(prev / safe_d, factor),
                            torch.where(is_last, 1.0 - torch.pow(prev / safe_d, factor), torch.zeros_like(b)))
        ratio = b / safe_d
        cum = torch.where(ratio <= 0.6, 0.28 * torch.exp(1.83 * ratio - 1.0),
                          1 - 0.441 * torch.pow(((1 - ratio) / 0.4), 1.4))
        f_inv = torch.where(before_last, cum - top_acc, torch.where(is_last, 1.0 - top_acc, torch.zeros_like(b)))
        frac.append(torch.where(effect == INVERSION_SOME_MIXING, f_inv, f_mix))
        prev = torch.where(before_last, b, prev)
        top_acc = torch.where(before_last, cum, top_acc)
    return mix, last, torch.stack(coef, -1), torch.stack(frac, -1)


def tillage(r: Residues, mask, thickness, depth, effect, flatten_ratio, burial_ratio) -> Residues:
    """Residues_V4_common::respond_to_field_operation (residue_V4_common.cpp:394-651) for the
    masked tiles. depth [m] (float32 in the reference), effect: Soil_disturbance_effect code,
    flatten_ratio / burial_ratio: the RUSLE2 ratios of residue type index 0 (every pool is
    UNKNOWN, see module notes)."""
    sur, sub, over = r.sur, r.sub, r.overflow
    S = sur.active.shape[-1]
    dt = sur.mass.dtype
    # 1. standing stubble partly flattened (:401-430), clones appended after the loop in order
    on_s = mask[..., None] & sur.active & (sur.position == STUBBLE)
    m = sur.mass
    fr_ = flatten_ratio[..., None] if torch.is_tensor(flatten_ratio) and flatten_ratio.dim() else flatten_ratio
    sur0 = sur
    sur = _rep(sur, mass=torch.where(on_s, m * (1.0 - fr_), m))
    sur, o = append_many(sur, on_s, sur0.type, FLAT, 0, sur0.cycling, sur0.env, m * fr_, sur0.cfrac, sur0.cn,
                         sur0.k, sur0.kdet, sur0.apm * 2.0)
    over = over | o
    L = thickness.shape[-1]
    mix, last, coef, fr = tillage_layers(mask, thickness, depth, effect)
    coef = [coef[..., l] for l in range(L)]
    # 2. sifting of subsurface residue (:438-504)
    moved = []
    for t in (STRAW, MANURE):
        for c in (0, FAST, SLOW, LIGNIFIED):
            rin = None
            for l in range(L):
                inl = mix & (l + 1 <= last)
                cond = sub.active & (sub.type == t) & (sub.position == SUBSURFACE) & (sub.cycling == c) & (sub.layer == l + 1)
                found, idx = first_index(cond)
                hit = inl & found
                g = lambda f: _take(f, idx)
                mass, cfrac, cn, k, apm = g(sub.mass), g(sub.cfrac), g(sub.cn), g(sub.k), g(sub.apm)
                out = mass * coef[l]
                if rin is not None:
                    r_on, r_f = rin
                    mrg = hit & r_on
                    m2, cf2, cn2, k2, apm2 = add_different_mass(mass, cfrac, cn, k, apm, torch.zeros_like(hit),
                                                               r_f["mass"], r_f["cfrac"], r_f["cn"], r_f["k"], r_f["apm"])
                    mass = torch.where(mrg, m2, mass); cfrac = torch.where(mrg, cf2, cfrac)
                    cn = torch.where(mrg, cn2, cn); k = torch.where(mrg, k2, k)
                    # no pool here: the incoming clone joins the list unchanged (:497-500)
                    moved.append((inl & ~found & r_on, r_f))
                sub_out = hit & (out > 0.0)
                mass = torch.where(sub_out, add_mass(mass, -out), mass)
                sub = _rep(sub, mass=_put(sub.mass, hit, idx, mass), cfrac=_put(sub.cfrac, hit, idx, cfrac),
                           cn=_put(sub.cn, hit, idx, cn), k=_put(sub.k, hit, idx, k))
                new_on = hit & ~_approx(out, torch.zeros_like(out), APPROX)
                clone = dict(mass=reset_mass(mass, out), cfrac=cfrac, cn=cn, k=k, apm=apm, type=g(sub.type),
                             cycling=g(sub.cycling), layer=g(sub.layer), env=g(sub.env), kdet=g(sub.kdet))
                if rin is None:
                    rin = (new_on, clone)
                else:
                    r_on, r_f = rin
                    keep_old = ~(inl & (found | r_on))         # untouched when the layer is not processed
                    nf = {key: torch.where(new_on, clone[key], r_f[key]) for key in clone}
                    rin = (torch.where(keep_old, r_on, new_on), nf)
    if moved:
        st_ = lambda key: torch.stack([f[key] for _, f in moved], -1)
        sub, o = append_many(sub, torch.stack([on for on, _ in moved], -1), st_("type"), SUBSURFACE, st_("layer"),
                             st_("cycling"), st_("env"), st_("mass"), st_("cfrac"), st_("cn"), st_("k"), st_("kdet"),
                             st_("apm"))
        over = over | o
    # 3. burial of flat residue (:506-648)
    # the subsurface list does not change membership during the burial (clones are appended
    # after it), so the first comparable pool of (type, cycling, layer) is fixed; the merges of one
    # flat pool go to distinct layers and are done together, the flat pools in list order
    lay_ids = torch.arange(1, L + 1, device=depth.device)
    base = sub.active & (sub.position == SUBSURFACE)
    lay_match = sub.layer[..., None, :] == lay_ids[:, None]
    tgt = []
    for t in (STRAW, MANURE):
        for c in (0, FAST, SLOW, LIGNIFIED):
            tgt.append((t, c) + first_index((base & (sub.type == t) & (sub.cycling == c))[..., None, :] & lay_match))
    inl = lay_ids <= last[..., None]
    br = burial_ratio[..., None] if torch.is_tensor(burial_ratio) and burial_ratio.dim() else burial_ratio
    b_on, b_t, b_c, b_e, b_m, b_cf, b_cn, b_k, b_apm = [], [], [], [], [], [], [], [], []
    for j in range(S):
        on = mix & sur.active[..., j] & (sur.position[..., j] == FLAT)
        m0 = sur.mass[..., j]
        t, c = sur.type[..., j], sur.cycling[..., j]
        go = on[..., None] & inl & (fr > 0.0)
        mt = m0[..., None] * br * fr
        cm = reset_mass(m0[..., None].expand_as(mt), mt)
        found = torch.zeros_like(go)
        idx = torch.zeros(go.shape, dtype=torch.long, device=go.device)
        for (tt, cc, f_, i_) in tgt:
            sel = ((t == tt) & (c == cc))[..., None]
            found = torch.where(sel, f_, found)
            idx = torch.where(sel, i_, idx)
        hit = go & found
        g = lambda f: torch.gather(f, -1, idx)
        sj = lambda f: f[..., j, None]
        m2, cf2, cn2, k2, _ = add_different_mass(g(sub.mass), g(sub.cfrac), g(sub.cn), g(sub.k), g(sub.apm),
                                                 torch.zeros_like(hit), cm, sj(sur.cfrac), sj(sur.cn), sj(sur.k),
                                                 sj(sur.apm))
        sub = _rep(sub, mass=put_many(sub.mass, hit, idx, m2), cfrac=put_many(sub.cfrac, hit, idx, cf2),
                   cn=put_many(sub.cn, hit, idx, cn2), k=put_many(sub.k, hit, idx, k2))
        e = lambda v: torch.broadcast_to(v[..., None], go.shape)
        b_on.append(go & ~found)
        b_t.append(e(t)); b_c.append(e(c)); b_e.append(e(sur.env[..., j])); b_m.append(cm)
        b_cf.append(e(sur.cfrac[..., j])); b_cn.append(e(sur.cn[..., j])); b_k.append(e(sur.k[..., j]))
        b_apm.append(e(sur.apm[..., j]))
        sur = _rep(sur, mass=torch.where(on[..., None] & (torch.arange(S, device=m0.device) == j),
                                         (m0 * (1.0 - burial_ratio))[..., None], sur.mass))
    cat = lambda xs: torch.cat(xs, -1)
    sub, o = append_many(sub, cat(b_on), cat(b_t), SUBSURFACE, lay_ids.repeat(S), cat(b_c), cat(b_e), cat(b_m),
                         cat(b_cf), cat(b_cn), cat(b_k), torch.zeros_like(cat(b_m)), cat(b_apm))
    over = over | o
    return Residues(sur, sub, r.flat_contact, over)


def residue_params_from_table(crops, device=None, dtype=torch.float64) -> ResidueParams:
    """[n_crop] ResidueParams from a crop table (readers.crop_table), defaults of
    residue_decomposition_param.cpp:55-81 for missing keys."""
    n = crops.n
    dev = device if device is not None else next(iter(crops.params.values())).device
    def get(key):
        t = crops.params.get(f"residue.{key}")
        full = torch.full((n,), float(RESIDUE_DEFAULTS[key]), dtype=dtype, device=dev)
        if t is None:
            return full
        t = t.to(dev)
        return torch.where(torch.isnan(t), full, t.to(dtype))
    f = lambda k: round_f32(get(k))
    z = torch.zeros(n, dtype=dtype, device=dev)
    return ResidueParams(
        torch.stack([z, f("fast_cycling_top_fraction"), f("slow_cycling_top_fraction"), f("lignified_top_fraction")], -1),
        torch.stack([z, f("fast_cycling_root_fraction"), f("slow_cycling_root_fraction"), f("lignified_root_fraction")], -1),
        torch.stack([z, f("fast_cycling_decomposition_time_50"), f("slow_cycling_decomposition_time_50"),
                     f("lignified_decomposition_time_50")], -1),
        f("detrition_time_50"), f("carbon_fraction"), f("stubble_area_covered_to_mass_ratio"),
        f("area_covered_to_mass_ratio"))


def gather_residue_params(p: ResidueParams, index) -> ResidueParams:
    return ResidueParams(*[x[index] for x in p])
