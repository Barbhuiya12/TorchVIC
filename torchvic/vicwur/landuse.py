# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR land use (C11): time-varying Cv and the redistribution of the
tile states when Cv changes.  Pure tensor kernels.

Batched over cells (C), vegetation tiles (T, the veg_con index), snow bands
(B), soil layers (L), frost areas (F), thermal nodes (N) and irrigated tiles
(I, irr_con order).  No loop over cells and no host read: the Python loops
run over tiles, layers and irrigated tiles only (they keep the C summation
order, so the sums are deterministic and bitwise those of the C on the CPU),
and every branch is on an option.

C references are VIC-WUR 3.0 plugins/landuse
(<data root>/build/vicwur3/src_tree/vic/plugins/landuse;
MINCOVERAGECHANGE = 1e-5 is include/landuse.h:32):
  lu_force.c:70-91       forced Cv per veg class onto the tiles (vidx); a
                         class without a tile but forced Cv > 0 is fatal
  lu_force.c:94          lu_apply on every forcing record (forcerun: the
                         first step and each new MONTH/YEAR record)
  lu_apply.c:995-1029    Cv_new = forced / sum(forced) (normalised BEFORE
                         the <= 0 check), Cv_change = Cv_new - Cv_old
  lu_apply.c:1032-1033   only bands with AreaFract > 0
  lu_apply.c:1035-1044   derived water states, old heat capacities/energies
  lu_apply.c:251-481     water: Wdew, pack_water, surf_water, swq,
                         snow_canopy, moist[l], ice[l][f]; shrinking tiles
                         (dCv < -1e-5) form a pool weighted by -dCv, each
                         growing tile (dCv > 1e-5) gets
                         X Cv_old/Cv_new + pool (dCv/Cv_avail)/Cv_new
  lu_apply.c:33-245,308  paddy first: with IRRIGATION and Pbare !=
                         NVEGTYPES, the changed paddy tiles and the Pbare
                         tile exchange among themselves and then leave the
                         general pass (Cv_change_tmp = 0)
  lu_apply.c:383-433     post-fixes of a growing tile: canopy snow to swq
                         (no overstory, LAI <= 0 or the bare tile), Wdew to
                         layer 0 (LAI <= 0 or bare), soil above max_moist
                         cascades down from layer 0; what is left in a full
                         column is warned about and LOST
  lu_apply.c:486-602     carbon: AnnualNPP, AnnualNPPPrev, CLitter, CInter,
                         CSlow, the same pool (no paddy step)
  lu_apply.c:607-781     energy: heat-capacity weighted temperatures,
                         E = capacity T; pools of E, then T = E/capacity_new;
                         positive snow-pack E without capacity goes to the
                         surface layer, positive surface E without capacity
                         to node 0, negative E without capacity is dropped
  lu_apply.c:787-904     irrigation leftover of the changed irrigated tiles,
                         sum(leftover Cv_old), to the ring bins
                         [rps, UH + rps - 1) / (UH - 1) each, clipped at 0,
                         per band and NOT weighted by AreaFract
  lu_support.c:33-90     heat capacities (CONST_VCPICE_WQ swq parts,
                         Cs_node dz_node; zero where Cv = 0)
  lu_support.c:95-279    derived water states (snow density, depth, MELTING;
                         Cs_node through distribute_node_moisture_properties)
  lu_support.c:284-356   derived energy states (coldcontent)
  lu_apply.c:1101-1131   conservation warnings (max |.| nv 1e-5); the
                         per-function checks use integer abs (:440, 571, 883)

Two modes:
  exact   the C, quirks included: the 1e-5 threshold (tiles changing less
          keep their per-area states, so their water is created or lost),
          pools without a receiver are lost, soil-overflow water is lost,
          negative energy without capacity is lost, the leftover goes to the
          ring once per band without AreaFract, the new-snow density uses
          cell 0's air temperature and Tfactor (force->air_temp[NR] and
          soil_con->Tfactor, lu_support.c:227-228).
  fixed   threshold 0; a pool without a receiver in its group goes to the
          other group's receivers; overflow returned as runoff;
          energy folded whatever its sign; leftover weighted by AreaFract;
          per-cell air temperature.  Water and energy then close exactly.
Every term the C loses or creates is returned (`terms`), so the ledger
closes in both modes: see `ledger`.

Units are the C's: moist, ice, Wdew, leftover mm; swq, pack_water,
surf_water, snow_canopy m; temperatures C; capacities J m-2 K-1.  The cell
water in mm is Wdew + sum(moist) + 1000 (swq + snow_canopy) (ice is part of
moist, pack and surface water are part of swq).
"""
import torch

MINCOVERAGECHANGE = 0.00001        # landuse.h:32
AREA_SUM_ERROR_THRESH = 1e-5       # vic_driver_shared_image.h:37
CONST_RHOFW = 1.000e3              # vic_physical_constants.h:104
CONST_CPICE = 2.11727e3            # :112
CONST_VCPICE_WQ = CONST_CPICE * CONST_RHOFW   # :114
MM_PER_M = 1000.0

WATER = ('wdew', 'pack_water', 'surf_water', 'swq', 'snow_canopy')
CARBON = ('annual_npp', 'annual_npp_prev', 'clitter', 'cinter', 'cslow')


def _col(x, t, nd):
    """[C, T] -> column t shaped [C, 1, ...] for an nd-dimensional [C, ...]
    (tile already removed)."""
    return x[:, t].reshape((x.shape[0],) + (1,) * (nd - 1))


def _w(m, x):
    """x where m else 0 (NaN-safe: masked entries never enter a sum)."""
    return torch.where(m, x, torch.zeros_like(x))


# --------------------------------------------------------------- Cv update
def map_cv_forcing(forced, tile_class):
    """lu_force.c:70-91.  forced [C, K] per veg class (K = NVEGTYPES);
    tile_class [C, T] the 0-based class of each tile, -1 for VIC's own bare
    tile (veg_class == NVEGTYPES, never in the forcing: its lu_force.Cv
    stays 0, lu_init_library.c:40) and for padding.
    Returns (cv [C, T], fatal [C]): fatal where a class without a tile has
    forced Cv > 0 (log_err).

    A forced Cv below 0 (the -1e-16 rounding residue of a regridded fraction
    map: 427 ISC cells) is taken as 0: it made the tile's moisture NaN at the
    first step of the ISC land-use run (ISC campaign, S5L); a valid forcing
    is unchanged."""
    K = forced.shape[1]
    has = tile_class >= 0
    cv = _w(has, forced.gather(1, tile_class.clamp(min=0)).clamp(min=0))
    k = torch.arange(K, device=forced.device)
    present = (tile_class.unsqueeze(-1) == k).any(1)            # [C, K]
    fatal = (~present & (forced > 0)).any(1)
    return cv, fatal


def cv_update(cv_forced, cv_old, active):
    """lu_apply.c:995-1029.  cv_forced, cv_old [C, T]; active [C, T] the
    tiles < nv_active.  Returns (cv_new, dcv, cv_sum, fatal, warn): fatal
    where sum <= 0 (log_err; cv_new is then kept at cv_old), warn where
    |sum - 1| > 1e-5 (assert_close_double, comparisons.c:48-56)."""
    f = _w(active, cv_forced)
    s = f[:, 0]
    for t in range(1, f.shape[1]):
        s = s + f[:, t]
    fatal = ~(s > 0)
    warn = ~((s - 1.0).abs() <= AREA_SUM_ERROR_THRESH)
    safe = torch.where(fatal, torch.ones_like(s), s)
    cv_new = torch.where(active & ~fatal[:, None], f / safe[:, None], cv_old)
    return cv_new, cv_new - cv_old, s, fatal, warn


# --------------------------------------------------------------- capacities
def snow_capacities(swq, surf_water, pack_water, cv, max_surface_swe=0.125):
    """get_heat_capacities, lu_support.c:60-73,81-83: [C, T, B] surface and
    pack capacities (J m-2 K-1); zero where Cv <= 0 (cv [C, T])."""
    ice = swq - surf_water - pack_water
    surf = ice.clamp(max=max_surface_swe)  # (ice > max) ? max : ice
    pack = ice - surf
    on = (cv > 0).unsqueeze(-1)
    return (_w(on, CONST_VCPICE_WQ * surf), _w(on, CONST_VCPICE_WQ * pack))


def node_capacities(cs_node, dz_node, cv):
    """lu_support.c:75-79,85-87: Cs_node dz_node [C, T, B, N], zero where
    Cv <= 0.  dz_node [C, N]."""
    on = (cv > 0)[:, :, None, None]
    return _w(on, cs_node * dz_node[:, None, None, :])


def node_heat_capacity(moist, t_node, soil, frozen_soil=False):
    """Cs_node [C, T, B, N] of distribute_node_moisture_properties
    (soil_conduction.c:229-316, called at lu_support.c:253-271) from layer
    moisture [C, T, B, L] and node temperatures [C, T, B, N], through the
    TorchVIC port (torchvic.core.soiltemp.distribute_node_moisture).
    `soil`: zsum_node, max_moist_node, expt_node, bubble_node [C, N]; depth,
    soil_dens_min, bulk_dens_min, quartz, soil_density, bulk_density,
    organic [C, L]; fs_active [C]."""
    from torchvic.core import soiltemp
    nn, nl = t_node.shape[-1], moist.shape[-1]
    ex = lambda x: x[:, None, None, :].expand(moist.shape[:3] + x.shape[-1:])
    fs = soil.get('fs_active')
    if fs is None:
        fs = torch.zeros(moist.shape[0], dtype=torch.bool, device=moist.device)
    fs = fs[:, None, None].expand(moist.shape[:3])
    return soiltemp.distribute_node_moisture(
        moist, t_node, ex(soil['zsum_node']), ex(soil['max_moist_node']),
        ex(soil['expt_node']), ex(soil['bubble_node']), ex(soil['depth']),
        ex(soil['soil_dens_min']), ex(soil['bulk_dens_min']),
        ex(soil['quartz']), ex(soil['soil_density']),
        ex(soil['bulk_density']), ex(soil['organic']), nn, nl, fs,
        frozen_soil)[3]


def layer_heat_capacity(moist, ice, soil):
    """layer[l].Cs of compute_soil_layer_thermal_properties (soil_conduction
    .c:721-737): [C, T, B, L] from moist [C, T, B, L] and ice
    [C, T, B, L, F] (frost-weighted with soil['frost_fract'] [C, F])."""
    from torchvic.core import frozen
    dep = soil['depth'][:, None, None, :]
    ff = soil['frost_fract']
    m = moist / dep / MM_PER_M
    ic = torch.zeros_like(m)
    for f in range(ice.shape[-1]):
        ic = ic + ice[..., f] / dep / MM_PER_M * ff[:, None, None, None, f]
    bd = soil['bulk_density'][:, None, None, :]
    sd = soil['soil_density'][:, None, None, :]
    return frozen.volumetric_heat_capacity(bd / sd, m - ic, ic,
                                           soil['organic'][:, None, None, :])


def lu_heat_capacity_fn(soil, mode='fixed', frozen_soil=False):
    """cs_fn(moist, ice, t_node) -> the Cs_node lu_apply's energy step uses.

    exact: calculate_derived_water_states writes energy.Cs[2] =
    layer[1].Cs (lu_support.c:275) past the end of `double Cs[2]`
    (vic_def.h:856), i.e. into Cs_node[0] (vic_def.h:857), after
    distribute_node_moisture_properties has set it; both capacity
    computations of lu_apply (:1036, :1067) read that.  The node-0 heat
    capacity is therefore layer 1's (and kappa_node[0] is layer 1's kappa,
    which the redistribution does not use).  fixed: Cs_node[0] as computed.
    The Cs_node lu_apply leaves behind is the correct one (recomputed by
    calculate_derived_energy_states, lu_support.c:332-350)."""
    def cs_fn(moist, ice, t_node):
        cs = node_heat_capacity(moist, t_node, soil, frozen_soil)
        if mode == 'fixed' or moist.shape[-1] < 2:
            return cs
        lcs = layer_heat_capacity(moist, ice, soil)
        return torch.cat([lcs[..., 1:2], cs[..., 1:]], -1)
    return cs_fn


# ------------------------------------------------------------ derived snow
def derived_snow(swq, snow_canopy, coldcontent, density, depth, last_snow,
                 overstory, doy, lat, new_density):
    """The snow part of calculate_derived_water_states, lu_support.c:207-232.
    [C, T, B] states; overstory [C, T]; doy (VIC day_in_year) int or
    0-d tensor; lat [C]; new_density [C, B] the new-snow density.
    Returns (snow, melting, density, depth)."""
    pos = swq > 0
    snow = pos | ((snow_canopy > 0) & overstory.unsqueeze(-1))
    north = (doy > 60) & (doy < 273)       # python bools or 0-d tensors:
    south = (doy < 60) | (doy > 273)       # no host-to-device copy
    ge = lat >= 0
    season = ((ge & north) | (~ge & south))[:, None, None]
    melting = pos & (coldcontent >= 0) & season
    renew = pos & ((last_snow == 0) | (density <= 0))
    density = torch.where(renew, new_density[:, None, :].expand_as(density),
                          density)
    depth = torch.where(pos, CONST_RHOFW * swq / torch.where(
        pos, density, torch.ones_like(density)), depth)
    return snow, melting, density, depth


def band_new_density(air_temp, tfactor, method='bras', dens_max=400.0):
    """new_snow_density(air_temp + Tfactor[band]) [C, B]
    (snow_utility.c:208-245, VIC-WUR 3.0 caps at
    SNOW_NEW_SNOW_DENS_MAX = 400, initialize_parameters.c:154).  air_temp [C]
    (force->air_temp[NR]), tfactor [C, B]."""
    from torchvic.core import snow
    return snow.new_snow_density(air_temp[:, None] + tfactor, method=method,
                                 dens_max=dens_max)


# ------------------------------------------------------------- the passes
def _pool(X, give, w):
    """sum_t X[:, t] w[:, t] over the giving tiles, in tile order
    (lu_apply.c:337-354).  X [C, T, ...], give/w [C, T]."""
    nd = X.dim() - 1
    acc = torch.zeros_like(X[:, 0])
    for t in range(X.shape[1]):
        acc = acc + _w(_col(give, t, nd), X[:, t] * _col(w, t, nd))
    return acc


def _avail(dcv, recv):
    """Cv_avail: sum of the receivers' dCv in tile order (:318-322)."""
    a = torch.zeros_like(dcv[:, 0])
    for t in range(dcv.shape[1]):
        a = a + _w(recv[:, t], dcv[:, t])
    return a


def _fracs(cv_old, cv_new, dcv, recv, avail, red_one=None):
    """red_frac, add_frac [C, T] of the receivers (:359-360)."""
    one = torch.ones_like(cv_new)
    den = torch.where(recv, cv_new, one)
    av = torch.where(recv, avail[:, None].expand_as(dcv), one)
    red = cv_old / den
    if red_one is not None:
        red = torch.where(red_one, one, red)
    return red, (dcv / av) / den


def _set(X, pool, recv, red, add):
    """X red + pool add on the receivers (:362-381).  X [C, T, B, ...],
    pool [C, B, ...], recv/red/add [C, T]."""
    nd = X.dim()
    sh = recv.shape + (1,) * (nd - 2)
    return torch.where(recv.reshape(sh), X * red.reshape(sh)
                       + pool.unsqueeze(1) * add.reshape(sh), X)


def _post_fixes(st, recv, P):
    """lu_apply.c:383-433 on the receivers (recv [C, T]): canopy snow to
    swq, Wdew to layer 0, soil overflow cascade.  Returns the water left in
    a full column (mm over the tile, [C, T, B])."""
    rb = recv.unsqueeze(-1)
    lai0 = P['lai'] <= 0.
    bare = P['bare'].unsqueeze(-1)
    fs = rb & (~P['overstory'].unsqueeze(-1) | lai0 | bare)
    st['swq'] = torch.where(fs, st['swq'] + st['snow_canopy'], st['swq'])
    st['snow_canopy'] = _w(~fs, st['snow_canopy'])
    fd = rb & (lai0 | bare)
    m = st['moist']
    L = m.shape[-1]
    lay = [m[..., l] for l in range(L)]
    lay[0] = torch.where(fd, lay[0] + st['wdew'], lay[0])
    st['wdew'] = _w(~fd, st['wdew'])
    mx = P['max_moist']                                     # [C, L]
    inflow = torch.zeros_like(lay[0])
    for l in range(L):                                      # :400-409
        ml = mx[:, l, None, None]
        ex = rb & (lay[l] > ml)
        inflow = torch.where(ex, inflow + (lay[l] - ml), inflow)
        lay[l] = torch.where(ex, ml.expand_as(lay[l]), lay[l])
    go = rb & (inflow > 0)
    carry = _w(go, inflow)
    for l in range(L):                                      # :411-427
        ml = mx[:, l, None, None]
        v = lay[l] + carry
        over = go & (v > ml)
        lay[l] = torch.where(go, torch.where(over, ml.expand_as(v), v), lay[l])
        carry = torch.where(over, v - ml, torch.zeros_like(v))
    st['moist'] = torch.stack(lay, -1)
    return carry


def _water_pass(st, give, recv, w, dcv, cv_old, cv_new, P, avail=None,
                red_one=None):
    """One pool-and-set over the water states (:324-435 / :116-238)."""
    if avail is None:
        avail = _avail(dcv, recv)
    red, add = _fracs(cv_old, cv_new, dcv, recv, avail, red_one)
    for k in WATER + ('moist', 'ice'):
        st[k] = _set(st[k], _pool(st[k], give, w), recv, red, add)
    return _post_fixes(st, recv, P), avail


def _paddy_pool(X, giv, w, order):
    """The paddy group's pool in the C order: paddy tiles by irr_con index,
    then the Pbare tile (lu_apply.c:92-106,129-149).  order [C, I + 1] tile
    index or -1."""
    nd = X.dim() - 1
    acc = torch.zeros_like(X[:, 0])
    for j in range(order.shape[1]):
        t = order[:, j]
        ok = t >= 0
        tc = t.clamp(min=0)
        idx = tc.reshape((-1, 1) + (1,) * (X.dim() - 2)).expand(
            (X.shape[0], 1) + X.shape[2:])
        xt = X.gather(1, idx).squeeze(1)
        g = (ok & giv.gather(1, tc[:, None]).squeeze(1)).reshape(
            (-1,) + (1,) * (nd - 1))
        wt = w.gather(1, tc[:, None]).squeeze(1).reshape(
            (-1,) + (1,) * (nd - 1))
        acc = acc + _w(g, xt * wt)
    return acc


def _paddy_avail(dcv, recv, order):
    a = torch.zeros_like(dcv[:, 0])
    for j in range(order.shape[1]):
        t = order[:, j]
        tc = t.clamp(min=0)
        r = (t >= 0) & recv.gather(1, tc[:, None]).squeeze(1)
        a = a + _w(r, dcv.gather(1, tc[:, None]).squeeze(1))
    return a


# ----------------------------------------------------------------- energy
def _tsum(x):
    """Sum over the tile axis (dim 1) in tile order."""
    acc = torch.zeros_like(x[:, 0])
    for t in range(x.shape[1]):
        acc = acc + x[:, t]
    return acc


def _energy(st, E0, cap1, dcv, cv_old, cv_new, thr, mode):
    """distribute_energy_balance_terms, lu_apply.c:657-737, and the
    recomputation of get_energy_terms / calculate_derived_energy_states
    (:1082-1088, lu_support.c:551-559,353-354).  E0 = (surf, pack, T) old
    energies; cap1 = (surf, pack, node) new capacities.  Returns the new
    energies and the energy dropped (J m-2 over the tile, [C, T, B])."""
    give = dcv < -thr
    recv = dcv > thr
    w = -dcv
    avail = _avail(dcv, recv)
    red, add = _fracs(cv_old, cv_new, dcv, recv, avail)
    Es, Ep, Et = [_set(e, _pool(e, give, w), recv, red, add) for e in E0]
    # the C's check totals over the changed tiles (:652-655, :705-708)
    chg = (give | recv).unsqueeze(-1)
    c_tot = []
    for (a, b, n), cv in ((E0, cv_old), ((Es, Ep, Et), cv_new)):
        x = a + b
        for k in range(n.shape[-1]):
            x = x + n[..., k]
        c_tot.append(_tsum(_w(chg, x * cv.unsqueeze(-1))))
    cs, cp, cn = cap1
    signed = mode == 'fixed'
    zero = torch.zeros_like(Ep)
    # :711-719 pack -> surface where the pack has no capacity
    pk = cp > 0
    mv = ~pk & ((Ep != 0) if signed else (Ep > 0))
    lost = _w(~pk & ~mv, Ep)
    Es = torch.where(mv, Es + Ep, Es)
    st['pack_temp'] = torch.where(pk, Ep / torch.where(pk, cp, 1.0 + zero),
                                  st['pack_temp'])
    # :721-729 surface -> node 0 where the surface has no capacity
    sf = cs > 0
    mv = ~sf & ((Es != 0) if signed else (Es > 0))
    lost = lost + _w(~sf & ~mv, Es)
    Et0 = torch.where(mv, Et[..., 0] + Es, Et[..., 0])
    Et = torch.cat([Et0.unsqueeze(-1), Et[..., 1:]], -1)
    st['surf_temp'] = torch.where(sf, Es / torch.where(sf, cs, 1.0 + zero),
                                  st['surf_temp'])
    # :731-736 nodes
    nd = cn > 0
    lost = lost + _node_lost(Et, nd)
    st['t_node'] = torch.where(nd, Et / torch.where(nd, cn, torch.ones_like(
        cn)), st['t_node'])
    # get_energy_terms with the new capacities
    E1 = (st['surf_temp'] * cs, st['pack_temp'] * cp, st['t_node'] * cn)
    st['coldcontent'] = cs * st['surf_temp']
    return E1, lost, c_tot


def _node_lost(Et, nd):
    acc = torch.zeros_like(Et[..., 0])
    for n in range(Et.shape[-1]):
        acc = acc + _w(~nd[..., n], Et[..., n])
    return acc


def energy_total(E, cv, band=None):
    """calculate_total_energy summed over tiles (lu_support.c:459-469 over
    every tile, not only the changed ones): [C, B]."""
    Es, Ep, Et = E
    tot = Es + Ep
    for n in range(Et.shape[-1]):
        tot = tot + Et[..., n]
    acc = torch.zeros_like(tot[:, 0])
    for t in range(tot.shape[1]):
        acc = acc + tot[:, t] * cv[:, t, None]
    return acc


def water_column(st):
    """Water of every tile and band in mm: Wdew + sum(moist) + 1000 (swq +
    snow_canopy) [C, T, B]."""
    m = st['moist']
    tot = st['wdew'] + MM_PER_M * (st['swq'] + st['snow_canopy'])
    for l in range(m.shape[-1]):
        tot = tot + m[..., l]
    return tot




def _c_water(st, cv, chg):
    """calculate_total_water, lu_support.c:384-400: over the changed tiles,
    in the C's mixed units (pack and surface water counted again) [C, B]."""
    x = (st['wdew'] + st['pack_water'] + st['surf_water'] + st['swq']
         + st['snow_canopy'])
    for l in range(st['moist'].shape[-1]):
        x = x + st['moist'][..., l]
    return _tsum(_w(chg.unsqueeze(-1), x * cv.unsqueeze(-1)))


# ----------------------------------------------------- irrigation leftover
def leftover_to_ring(leftover, irr_tile, dcv, cv_old, area_fract, ring, area,
                     dt, rps, uh_length, thr, mode='fixed'):
    """distribute_irrigation_balance_terms, lu_apply.c:815-878, for all bands
    (lu_apply.c:1032-1138).  leftover [C, I, B] mm; irr_tile [C, I] tile
    index (-1 pad); ring [C, R] dt_discharge (m3 s-1); area [C] m2.  Every
    changed irrigated tile's leftover is zeroed and sum(leftover Cv_old)
    goes to bins [rps, uh_length + rps - 1), / (uh_length - 1) each, and a
    bin below 0 is set to 0.  exact: once per band with AreaFract > 0,
    unweighted; fixed: weighted by AreaFract.
    Returns (leftover, ring, discharge, stream, parts) with parts in mm over
    the cell [C]: removed (AreaFract-weighted leftover taken from the
    tiles), routed (what the ring got), clip (ring gain from the clip at 0).
    """
    tc = irr_tile.clamp(min=0)
    d = dcv.gather(1, tc)
    ch = (irr_tile >= 0) & ((d < -thr) | (d > thr))          # [C, I]
    cvo = cv_old.gather(1, tc)
    R = ring.shape[1]
    sel = torch.zeros(R, dtype=torch.bool, device=ring.device)
    sel[rps:uh_length + rps - 1] = True
    to_mm = dt / area * MM_PER_M
    routed = torch.zeros_like(area)
    removed = torch.zeros_like(area)
    clip = torch.zeros_like(area)
    cols = []
    for b in range(leftover.shape[2]):
        on = area_fract[:, b] > 0
        lo = torch.zeros_like(area)
        for i in range(leftover.shape[1]):                   # :834-840
            lo = lo + _w(ch[:, i], leftover[:, i, b] * cvo[:, i])
        removed = removed + _w(on, lo * area_fract[:, b])
        if mode == 'fixed':
            lo = lo * area_fract[:, b]
        lo = _w(on, lo)
        cols.append(_w(~(ch & on[:, None]), leftover[:, :, b]))
        q = lo / MM_PER_M * area / dt                         # :852-854
        add = on[:, None] & sel
        r = ring + _w(add, (q / (uh_length - 1))[:, None].expand_as(ring))
        neg = add & (r < 0)                                   # :861-863
        clip = clip - _tsum(_w(neg, r)) * to_mm
        ring = torch.where(neg, torch.zeros_like(r), r)
        routed = routed + lo
    leftover = torch.stack(cols, 2)
    dis = torch.zeros_like(area)
    stream = torch.zeros_like(area)
    for k in range(uh_length + rps - 1):                      # :867-878
        if k < rps:
            dis = dis + ring[:, k]
        else:
            stream = stream + ring[:, k]
    return leftover, ring, dis, stream, dict(removed=removed, routed=routed,
                                             clip=clip)


# ------------------------------------------------------------------ apply
def _orphan(W, give, recv, d):
    """Water (or energy) of a pool that has givers but no receiver: the C
    adds nothing anywhere, the givers just shrink.  [C, B]."""
    none = ~recv.any(1) & give.any(1)
    return _w(none[:, None], _tsum(_w(give.unsqueeze(-1), W * (-d)
                                       .unsqueeze(-1))))


def apply(st, cv_old, cv_new, P, mode='fixed', paddy_exchange=False,
          route_leftover=False, cs_fn=None, full_energy=False):
    """lu_apply's band loop (lu_apply.c:1032-1140) for every cell and band.

    st: dict of [C, T, B] tensors wdew, swq, pack_water, surf_water,
        snow_canopy, the CARBON pools (optional), surf_temp, pack_temp,
        coldcontent, density, depth, last_snow, melting; moist [C, T, B, L];
        ice [C, T, B, L, F]; t_node [C, T, B, N]; with route_leftover also
        leftover [C, I, B] and ring [C, R].
    cv_old, cv_new [C, T] (cv_update).
    P: area_fract [C, B]; max_moist [C, L]; overstory, bare (veg_class ==
       NVEGTYPES), paddy (a paddy irrigated tile), pbare (the Pbare tile)
       [C, T] bool; lai [C, T, B]; dz_node [C, N]; max_surface_swe;
       doy, lat [C], new_density [C, B] (derived_snow); with
       paddy_exchange: paddy_order [C, I + 1] (paddy tiles in irr_con
       order, then the Pbare tile, -1 pad); with route_leftover: irr_tile
       [C, I], area [C], dt, rps, uh_length; optional active [C, T] (the
       tiles < nv_active).
    cs_fn(moist, ice, t_node) -> the Cs_node of the energy step
    (lu_heat_capacity_fn).
    paddy_exchange = IRRIGATION and Pbare != NVEGTYPES (lu_apply.c:308-309;
    irr_start.c:43-51 allows Pbare = NVEGTYPES only when NBARE = 1, so the
    paddy step needs NBARE > 1).  route_leftover = IRRIGATION and ROUTING
    (:1133).

    Returns (new state, terms).  terms ([C, B] sums over the tiles weighted
    by Cv, i.e. mm or J m-2 over the band; see `ledger`): water_before,
    water_after, lost (soil overflow of a full column), orphan (pools
    without a receiver), subthreshold (dCv X of the tiles changing within
    +-1e-5), the same energy_* terms, overflow [C, T, B] (mm per tile, the
    runoff of `fixed`), fatal [C] (the C's log_err checks), and with
    route_leftover discharge, stream and the leftover parts [C].
    """
    thr = 0.0 if mode == 'fixed' else MINCOVERAGECHANGE
    old = st
    st = dict(st)
    af = P['area_fract']
    on_b = af > 0                                             # [C, B]
    dcv = cv_new - cv_old
    chg = (dcv < -thr) | (dcv > thr)
    lat, doy = P['lat'], P['doy']
    msw = P.get('max_surface_swe', 0.125)
    nd_ = P['new_density']
    if mode != 'fixed':                 # force->air_temp, soil_con->Tfactor
        nd_ = nd_[:1].expand_as(nd_)    # are cell 0's (lu_support.c:227)

    # :1035 derived water states on the old state; old capacities, energies
    _, _, st['density'], st['depth'] = derived_snow(
        st['swq'], st['snow_canopy'], st['coldcontent'], st['density'],
        st['depth'], st['last_snow'], P['overstory'], doy, lat, nd_)
    cs0 = cs_fn(st['moist'], st['ice'], st['t_node'])
    sc0 = snow_capacities(st['swq'], st['surf_water'], st['pack_water'],
                          cv_old, msw)
    nc0 = node_capacities(cs0, P['dz_node'], cv_old)
    E0 = (st['surf_temp'] * sc0[0], st['pack_temp'] * sc0[1],
          st['t_node'] * nc0)
    W0 = water_column(st)
    c_before = _c_water(st, cv_old, chg)

    # water, the paddy group first (:308-314, :33-245)
    dtmp = dcv
    lost_w = torch.zeros_like(st['wdew'])
    orphan = torch.zeros_like(af)
    if paddy_exchange:
        order = P['paddy_order']
        tc = order.clamp(min=0)
        ar = torch.arange(chg.shape[1], device=chg.device)
        inord = torch.zeros_like(chg)
        for j in range(order.shape[1]):
            inord = inord | ((order[:, j:j + 1] >= 0) & (ar == tc[:, j:j + 1]))
        grp = inord & ((P['paddy'] & chg) | P['pbare'])
        g_give = grp & (dcv < -thr)
        g_recv = grp & (dcv > thr)
        av_p = _paddy_avail(dcv, g_recv, order)
        none_p = ~g_recv.any(1) & g_give.any(1)
        red, add = _fracs(cv_old, cv_new, dcv, g_recv, av_p)
        for k in WATER + ('moist', 'ice'):
            pool = _paddy_pool(st[k], g_give, -dcv, order)
            st[k] = _set(st[k], pool, g_recv, red, add)
        lost_w = lost_w + _post_fixes(st, g_recv, P)
        if mode == 'fixed':
            # a paddy pool without a paddy receiver joins the general pass
            dtmp = _w(~(grp & ~none_p[:, None]), dcv)
        else:
            orphan = orphan + _orphan(W0, g_give, g_recv, dcv)
            dtmp = _w(~grp, dcv)
    give = dtmp < -thr
    recv = dtmp > thr
    av = _avail(dtmp, recv)
    if mode == 'fixed' and paddy_exchange:
        # a general pool without a general receiver goes to the paddy
        # group's receivers (already diluted there: red_frac = 1)
        alt = ~recv.any(1) & give.any(1) & g_recv.any(1)
        recv2 = torch.where(alt[:, None], g_recv, recv)
        av2 = torch.where(alt, av_p, av)
        dd = torch.where(alt[:, None] & g_recv, dcv, dtmp)
        lw, _ = _water_pass(st, give, recv2, -dtmp, dd, cv_old, cv_new, P,
                            avail=av2, red_one=alt[:, None] & recv2)
        orphan = orphan + _orphan(W0, give, recv2, dtmp)
    else:
        lw, _ = _water_pass(st, give, recv, -dtmp, dtmp, cv_old, cv_new, P,
                            avail=av)
        orphan = orphan + _orphan(W0, give, recv, dtmp)
    lost_w = lost_w + lw
    c_after = _c_water(st, cv_new, chg)

    # :1060 derived water states on the new water
    _, st['melting'], st['density'], st['depth'] = derived_snow(
        st['swq'], st['snow_canopy'], st['coldcontent'], st['density'],
        st['depth'], st['last_snow'], P['overstory'], doy, lat, nd_)

    # carbon (:1063, :519-566)
    cfatal = torch.zeros_like(on_b)
    if all(k in st for k in CARBON):
        cg, cr = dcv < -thr, dcv > thr
        ca = _avail(dcv, cr)
        red, add = _fracs(cv_old, cv_new, dcv, cr, ca)
        cb = sum(_tsum(_w(chg.unsqueeze(-1), st[k] * cv_old.unsqueeze(-1)))
                 for k in CARBON)
        for k in CARBON:
            st[k] = _set(st[k], _pool(st[k], cg, -dcv), cr, red, add)
        caf = sum(_tsum(_w(chg.unsqueeze(-1), st[k] * cv_new.unsqueeze(-1)))
                  for k in CARBON)
        cfatal = (cb - caf).trunc().abs() > 0                 # :571 int abs

    # energy (:1067-1088)
    cs1 = cs_fn(st['moist'], st['ice'], st['t_node'])
    sc1 = snow_capacities(st['swq'], st['surf_water'], st['pack_water'],
                          cv_new, msw)
    nc1 = node_capacities(cs1, P['dz_node'], cv_new)
    E1, lost_e, (ceb, cea) = _energy(st, E0, (sc1[0], sc1[1], nc1), dcv,
                                     cv_old, cv_new, thr, mode)

    # bands with AreaFract <= 0 and padding tiles (>= nv_active) are not
    # touched (:1033; every loop runs to nv_active)
    act = P.get('active')
    tb = on_b.unsqueeze(1) if act is None else \
        (on_b.unsqueeze(1) & act.unsqueeze(-1))               # [C, T, B]

    def band(v, o):
        return torch.where(tb.reshape(tb.shape + (1,) * (v.dim() - 3)), v, o)
    for k in list(st):
        if k in ('leftover', 'ring'):
            continue
        st[k] = band(st[k], old[k])
    E1 = tuple(band(e, z) for e, z in zip(E1, E0))
    W1 = water_column(st)

    cvo, cvn = cv_old.unsqueeze(-1), cv_new.unsqueeze(-1)
    sub = ~chg & (dcv != 0)
    Et0 = E0[0] + E0[1]
    for n in range(E0[2].shape[-1]):
        Et0 = Et0 + E0[2][..., n]
    eg, er = dcv < -thr, dcv > thr
    terms = dict(
        water_before=_w(on_b, _tsum(W0 * cvo)),
        water_after=_w(on_b, _tsum(W1 * cvn)),
        lost=_w(on_b, _tsum(lost_w * cvn)),
        orphan=_w(on_b, orphan),
        subthreshold=_w(on_b, _tsum(_w(sub.unsqueeze(-1), W0) * dcv
                                    .unsqueeze(-1))),
        overflow=band(lost_w, torch.zeros_like(lost_w)),
        energy_before=_w(on_b, energy_total(E0, cv_old)),
        energy_after=_w(on_b, energy_total(E1, cv_new)),
        energy_lost=_w(on_b, _tsum(lost_e * cvn)),
        energy_orphan=_w(on_b, _orphan(Et0, eg, er, dcv)),
        energy_subthreshold=_w(on_b, _tsum(_w(sub.unsqueeze(-1), Et0)
                                           * dcv.unsqueeze(-1))))
    # the C's fatal checks: integer abs of the water/carbon differences
    # (:440, :571), the energy check under FULL_ENERGY (:740-744)
    wf = (c_before - c_after).trunc().abs() > 0
    fatal = (on_b & (wf | cfatal)).any(1)
    if full_energy:
        mx = torch.maximum(torch.maximum(ceb.abs(), cea.abs()),
                           torch.full_like(ceb, 2.220446049250313e-16))
        nv = (P['active'].sum(1) if 'active' in P else
              torch.full_like(cv_old[:, 0], cv_old.shape[1])).to(ceb.dtype)
        fatal = fatal | (on_b & ((ceb - cea).abs() > mx * nv[:, None]
                                 * MINCOVERAGECHANGE)).any(1)
    terms['fatal'] = fatal
    if route_leftover:
        lo, ring, dis, stream, parts = leftover_to_ring(
            old['leftover'], P['irr_tile'], dcv, cv_old, af, old['ring'],
            P['area'], P['dt'], P['rps'], P['uh_length'], thr, mode)
        st['leftover'], st['ring'] = lo, ring
        terms['discharge'], terms['stream'] = dis, stream
        tc = P['irr_tile'].clamp(min=0)
        dl = dcv.gather(1, tc)
        subl = (P['irr_tile'] >= 0) & ~((dl < -thr) | (dl > thr)) & (dl != 0)
        terms['leftover_subthreshold'] = (
            _w(subl.unsqueeze(-1), old['leftover']) * dl.unsqueeze(-1)
            * af.unsqueeze(1)).sum((1, 2))
        for k, v in parts.items():
            terms['leftover_' + k] = v
    return st, terms


def storage(st, cv, area_fract, P=None):
    """Cell water in mm over the cell [C]: sum_b AF_b sum_t Cv_t
    water_column, plus (with P and a leftover/ring in st) the leftover
    sum_b AF_b sum_i Cv leftover and the ring sum(dt_discharge) dt / area."""
    S = (_tsum(water_column(st) * cv.unsqueeze(-1)) * area_fract).sum(1)
    if P is not None and 'leftover' in st:
        tc = P['irr_tile'].clamp(min=0)
        ok = (P['irr_tile'] >= 0).unsqueeze(-1)
        S = S + (_w(ok, st['leftover']) * cv.gather(1, tc).unsqueeze(-1)
                 * area_fract.unsqueeze(1)).sum((1, 2))
        S = S + st['ring'].sum(1) * (P['dt'] / P['area'] * MM_PER_M)
    return S


def ledger(st0, st1, cv_old, cv_new, terms, area_fract, P=None):
    """Water ledger of one land-use step, mm over the cell [C]:

        storage_after - storage_before
            = subthreshold - lost - orphan                    (soil, snow)
              + leftover_subthreshold + leftover_routed
              - leftover_removed + leftover_clip               (irrigation)

    Each right-hand term is computed on its own (not as a difference of
    storages), so `residual` is round-off when the kernel conserves what it
    claims.  In `fixed` mode subthreshold = orphan = 0, routed = removed,
    and `lost` is the runoff the caller sends to the river.  Energy (the
    C's capacity-weighted temperatures, J m-2): energy_residual likewise."""
    af = area_fract
    S0 = storage(st0, cv_old, af, P)
    S1 = storage(st1, cv_new, af, P)
    rhs = ((terms['subthreshold'] - terms['lost'] - terms['orphan'])
           * af).sum(1)
    if 'leftover_routed' in terms:
        rhs = rhs + (terms['leftover_subthreshold'] + terms['leftover_routed']
                     - terms['leftover_removed'] + terms['leftover_clip'])
    eb = (terms['energy_before'] * af).sum(1)
    ea = (terms['energy_after'] * af).sum(1)
    erhs = ((terms['energy_subthreshold'] - terms['energy_lost']
             - terms['energy_orphan']) * af).sum(1)
    return dict(storage_before=S0, storage_after=S1, rhs=rhs,
                residual=(S1 - S0) - rhs, energy_before=eb, energy_after=ea,
                energy_residual=(ea - eb) - erhs)
