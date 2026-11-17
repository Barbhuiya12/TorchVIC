# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR's crop feedback into VIC and irrigation (docs/connectors/vicwur.md):
pure tensor kernels.

C paths are VIC-WUR 3.0, <data root>/build/vicwur3/src_tree/vic/plugins:

  crops/src/crop_update_step_vars.c:4-256   `update_step_vars`: the crops'
        LAI, cover, albedo, height, roots and Wcr into VIC's vegetation of
        the crop tiles (U phase, before vic_run)
  wofost/src/Sweaf.c:21-32                  `sweaf`
  crops/src/crop_adjust_requirement.c:7-97  `adjust_requirement`: the
        irrigation veto of tiles whose crops are not cultivating (P1)
  irrigation/src/irr_use.c:531-644          `wofost_water`: the IRRTAB water
        of irr_wofost (P3)
  crops/src/crop_output.c:1003-1032         `evap_outputs`: OUT_CROP_EVAP,
        _EVAP_BARE, _EVAP_CANOP, _TRANSP_VEG (O)

Layout. Units are WofostNPK's lanes [N, B, NC] (cell, band, crop class); a
crop class sits on tile `tile_of[n, c]` (its vegetation class - 1). VIC's
per-tile fields come back on cells [N, T] (veg_hist and veg_lib are one per
cell and tile) and per band [N, T, B, L] (veg_var root and the layer Wcr are
per band). The crop state is the C's float (WofostNPK float32); every
expression is evaluated as C evaluates it: a float promoted to double where
it meets a double, sums in the C order (a loop over bands, then the crop
list, then layers). No host sync, no data-dependent branch.

C quirks reproduced (docs/connectors/vicwur.md):
  * `root[1] += (1 - root[0]) * af` reads the running root[0]
    (crop_update_step_vars.c:127-133);
  * the root normalisation and the no-crop root of the Transform loop act on
    one stale veg_var, the one the Set loop left (the last unit of the last
    band: the cell's highest active crop class, band SNOW_BAND - 1), once per
    crop vegetation class in ascending order (:183-184, :208-214); every
    other band keeps its accumulated, unnormalised roots;
  * the layer Wcr of every band is divided by the cover summed over all
    bands (:216-223): the band scaling of Wcr;
  * the cover is floored at MIN_FCANOPY after the LAI < 1 rule (:227-249).
Where the stale band has no area C divides 0 by 0 (NaN on a band nothing
runs or reads); here that root stays finite.
"""
import numpy as np
import torch

#: VIC-WUR 3.0 initialize_parameters.c:48,58,61,67-68 and vic_def.h:74
ALBEDO_BARE_SOIL = 0.2
SOIL_RARC = 100.0
SOIL_WINDH = 10.0
VEG_RATIO_DH_HEIGHT = 0.67
VEG_RATIO_RL_HEIGHT = 0.123
MIN_FCANOPY = 0.0001
CM_PER_M = 100.0
MM_PER_CM = 10.0
DAYS_PER_YEAR = 365

#: veg_hist fields (cell, tile) and veg_lib fields, as the vicwur_core veg
#: hook names them (torchvic.vicwur.core TABLE_FIELDS, TILE_FIELDS)
HIST = ('lai', 'fcanopy', 'albedo', 'displacement', 'veg_rough')
LIB = ('trunk_ratio', 'wind_atten', 'wind_h', 'rad_atten', 'rgl', 'rarc',
       'rmin')


def _f32(x):
    """A C float value (round to float32, carried as float64)."""
    return x.to(torch.float32).to(torch.float64)


#: 0.10 and 0.95 as C floats (Sweaf's limit)
_SWEAF_LO = float(np.float32(0.10))
_SWEAF_HI = float(np.float32(0.95))


def sweaf(cgn, pet):
    """Sweaf.c:21-32, float arguments and result: [..] float32 tensors
    (cgn the CropGroupNumber, pet the PotEvapotransp, cm/day) -> the float
    value as float64."""
    c = cgn.to(torch.float64)
    p = pet.to(torch.float64)
    s = _f32(1. / (0.76 + 1.5 * p) - (5. - c) * 0.10)
    s = torch.where(c < 3., _f32(s + (p - 0.6) / (c * (c + 3.))), s)
    # limit(0.10, 0.95, sweaf), Functions.c:4-18, float arguments
    lo, hi = _SWEAF_LO, _SWEAF_HI
    return torch.where(s < lo, torch.full_like(s, lo),
                       torch.where(s <= hi, s, torch.full_like(s, hi)))


def stale_unit(active):
    """The (crop class [N], has [N]) whose veg_var the Set loop of
    crop_update_step_vars leaves in `cveg_var`: the last unit of the last
    band's list, i.e. the cell's highest active class (the lists are built
    in class order, wofost_alloc_free.c:21-40, on every band)."""
    NC = active.shape[1]
    idx = torch.arange(NC, device=active.device).expand_as(active)
    last = torch.where(active, idx, torch.full_like(idx, -1)).max(1).values
    return last.clamp(min=0), last >= 0


def update_step_vars(st, prm, cc, af, active, tile_of, crop_tiles,
                     crop_tile_list, soil, n_tile, cols=None):
    """crop_update_step_vars.c:4-256 for every cell.

    st          crop state: st_LAI, st_Development, st_RootDepth,
                met_PotEvaptrans [N, B, NC] (float32)
    prm         [NC] float32: Fcanopy MaxHeight Albedo TrunkRatio
                WindAttenuation RadiationAttenuation RGL
                MaxArchitecturalResistance MinStomatalResistance
                CropGroupNumber
    cc          [N, NC] Cc of the month (double)
    af          [N, B] AreaFract (double)
    active      [N, NC] bool (cidx != NODATA: a unit on every band)
    tile_of     [N, NC] long, the class's tile (vegetation class - 1)
    crop_tiles  [N, T] bool, the tiles some crop class maps to (cleared and
                transformed, :42-82, :154-255)
    crop_tile_list  the tiles that are a crop tile in some cell, ascending
                (host ints: the Transform loop's vegetation classes)
    soil        dict depth0 [N] m, wpwp, wfc, wcr [N, L] mm, rough [N] m
    cols        (host ints, ascending, optional) every tile an active class
                maps to and every tile of crop_tile_list: the only columns
                the C writes (the others stay 0); the sums run on them
    Returns dict: the HIST and LIB fields [N, T] (valid on crop_tiles),
    root and wcr [N, T, B, L].
    """
    cols = tuple(range(int(n_tile))) if cols is None else tuple(cols)
    if _fusable(cc):
        return _usv_compiled()(st, prm, cc, af, active, tile_of, crop_tiles,
                               tuple(crop_tile_list), soil, int(n_tile),
                               cols)
    return _usv(st, prm, cc, af, active, tile_of, crop_tiles,
                tuple(crop_tile_list), soil, int(n_tile), cols)


def _fusable(x):
    """update_step_vars as one fused region (core.fusion, eager's
    arithmetic): CUDA, fusion on, no gradient, not exact-math mode."""
    from ..core import fusion, xmath
    return (x.is_cuda and fusion.FUSION and not xmath.EXACT
            and not torch.is_grad_enabled())


_COMPILED = [None]


def _usv_compiled():
    if _COMPILED[0] is None:
        from ..crop.wofost_npk.fuse import compile_region
        _COMPILED[0] = compile_region(_usv)
    return _COMPILED[0]


def _usv(st, prm, cc, af, active, tile_of, crop_tiles, crop_tile_list, soil,
         n_tile, cols):
    """update_step_vars on whole [N, Tc] columns (the tiles `cols`; the
    others stay 0), without writes into slices (the same operations
    element for element as the C loops)."""
    dev = cc.device
    f64 = torch.float64
    N, B, NC = st['st_LAI'].shape
    TT = int(n_tile)
    T = len(cols)
    pos = {v: j for j, v in enumerate(cols)}
    L = soil['wpwp'].shape[1]
    z = lambda *s: torch.zeros(*s, dtype=f64, device=dev)
    acc = {k: z(N, T) for k in HIST + LIB}
    # root[:, :, b, l] and wcr[:, :, b, l] as [N, T] columns
    root = [[z(N, T) for _ in range(L)] for _ in range(B)]
    wcr = [[z(N, T) for _ in range(L)] for _ in range(B)]
    # [Tc] built on the device (no host copy: capturable)
    tix = torch.stack([torch.full((), v, dtype=torch.long, device=dev)
                       for v in cols])
    d0cm = soil['depth0'].to(f64) * CM_PER_M
    wpwp, wfc = soil['wpwp'].to(f64), soil['wfc'].to(f64)
    P = {k: v.to(f64) for k, v in prm.items()}
    # MaxHeight + 1: float + int, a float (:116)
    mh1 = _f32(prm['MaxHeight'].to(torch.float32) + 1)
    one = torch.ones((), dtype=f64, device=dev)
    # Set (:84-151): bands, then the crop list (classes in order)
    for b in range(B):
        for c in range(NC):
            on = active[:, c]                                    # [N]
            sel = (tix[None, :] == tile_of[:, c:c + 1]) & on[:, None]
            a = cc[:, c] * af[:, b]
            a = a * P['Fcanopy'][c]
            dev_ = st['st_Development'][:, b, c].to(f64)
            tmin = torch.where(dev_ < 1.0, dev_, one)
            h = P['MaxHeight'][c] * tmin
            terms = dict(
                fcanopy=a,
                lai=st['st_LAI'][:, b, c].to(f64) * a,
                albedo=P['Albedo'][c] * a,
                displacement=h * VEG_RATIO_DH_HEIGHT * a,
                veg_rough=h * VEG_RATIO_RL_HEIGHT * a,
                trunk_ratio=P['TrunkRatio'][c] * a,
                wind_atten=P['WindAttenuation'][c] * a,
                wind_h=mh1[c] * a,
                rad_atten=P['RadiationAttenuation'][c] * a,
                rgl=P['RGL'][c] * a,
                rarc=P['MaxArchitecturalResistance'][c] * tmin * a,
                rmin=P['MinStomatalResistance'][c] * a)
            for k, x in terms.items():
                acc[k] = torch.where(sel, acc[k] + x[:, None], acc[k])
            # roots, layers 0-1 only (:127-136)
            rd = st['st_RootDepth'][:, b, c].to(f64)
            deep = rd > d0cm
            rds = torch.where(deep, rd, one)
            r0 = root[b][0]
            r1 = root[b][1]
            n0 = torch.where(deep[:, None], r0 + ((d0cm / rds) * a)[:, None],
                             r0 + a[:, None])
            n1 = torch.where(deep[:, None], r1 + (1 - n0) * a[:, None], r1)
            root[b][0] = torch.where(sel, n0, r0)
            root[b][1] = torch.where(sel, n1, r1)
            # Wcr (:138-147)
            # 1 - sweaf(): int - float, a float (:138-141)
            frac = _f32(1 - sweaf(prm['CropGroupNumber'][c].expand(N),
                                  st['met_PotEvaptrans'][:, b, c]))
            for l in range(L):
                w = (wpwp[:, l] + (wfc[:, l] - wpwp[:, l]) * frac) * a
                cur = wcr[b][l]
                wcr[b][l] = torch.where(sel, cur + w[:, None], cur)
    # Transform (:153-255), crop vegetation classes in ascending order
    sc, has = stale_unit(active)
    st_t = tile_of.gather(1, sc[:, None])[:, 0]                  # [N]
    st_sel = (tix[None, :] == st_t[:, None]) & has[:, None]     # [N, T]
    for v in crop_tile_list:
        j = pos[v]
        col = (tix == v)[None, :]                                # [1, Tc]
        on = crop_tiles[:, v]
        fc, lai = acc['fcanopy'], acc['lai']
        fcv, laiv = fc[:, j], lai[:, j]
        nocrop = (fcv == 0) | (laiv == 0)
        dv = torch.where(nocrop, one, fcv)
        sel = col & on[:, None]
        upd = {}
        for k, dflt in (('rarc', SOIL_RARC), ('trunk_ratio', 0.0),
                        ('wind_atten', 0.5), ('wind_h', SOIL_WINDH),
                        ('rmin', 0.0), ('rad_atten', 0.5), ('rgl', 100.0)):
            upd[k] = torch.where(nocrop, torch.full_like(fcv, dflt),
                                 acc[k][:, j] / dv)
        for k in ('displacement', 'veg_rough'):
            upd[k] = torch.where(nocrop, soil['rough'].to(f64),
                                 acc[k][:, j] / dv)
        for k, x in upd.items():
            acc[k] = torch.where(sel, x[:, None], acc[k])
        # the stale veg_var's root (:183-184, :208-214)
        sr = root[B - 1]                                      # L x [N, T]
        rs = torch.zeros_like(sr[0])
        for l in range(L):
            rs = rs + sr[l]
        rsd = torch.where(rs != 0, rs, torch.ones_like(rs))
        m = st_sel & on[:, None]
        nc = nocrop[:, None]
        for l in range(L):
            nrm = sr[l] / rsd
            ncr = (torch.ones_like(sr[l]) if l == 0 else
                   torch.zeros_like(sr[l]) if l == 1 else sr[l])
            root[B - 1][l] = torch.where(m, torch.where(nc, ncr, nrm), sr[l])
        # every band's Wcr of the tile (:186-193, :216-223)
        wsoil = soil['wcr'].to(f64)
        for b in range(B):
            for l in range(L):
                wv = wcr[b][l][:, j]
                wnew = torch.where(nocrop, wsoil[:, l], wv / dv)
                wcr[b][l] = torch.where(sel, wnew[:, None], wcr[b][l])
        # LAI < 1, albedo, clamps (:226-249)
        lo = laiv / torch.where(fcv != 0, fcv, one)
        lo = torch.where(fcv != 0, lo, torch.full_like(lo, float('nan')))
        f2 = torch.where(lo < 1., laiv, fcv)
        alb = f2 * acc['albedo'][:, j] + (1 - f2) * ALBEDO_BARE_SOIL
        f3 = torch.where(f2 < 0, torch.zeros_like(f2),
                         torch.where(f2 > 1, torch.ones_like(f2),
                                     torch.where(f2 < MIN_FCANOPY,
                                                 torch.full_like(
                                                     f2, MIN_FCANOPY), f2)))
        acc['albedo'] = torch.where(sel, alb[:, None], acc['albedo'])
        acc['fcanopy'] = torch.where(sel, f3[:, None], acc['fcanopy'])
    full = lambda x: (x if T == TT else torch.zeros(
        (N, TT) + tuple(x.shape[2:]), dtype=x.dtype, device=dev).index_copy(
            1, tix, x))
    out = {k: full(x) for k, x in acc.items()}
    out['root'] = full(torch.stack([torch.stack(r, -1) for r in root], 2))
    out['wcr'] = full(torch.stack([torch.stack(w, -1) for w in wcr], 2))
    return out


def adjust_requirement(flag, cv, af, act, irr_tile, offset_max, active,
                       tile_of, cultivating, start_doy, doy, leap):
    """crop_adjust_requirement.c:7-97: clear the irrigation flag of an
    irrigated tile and band (Cv > 0, AreaFract > 0, flagged) where a crop
    of the tile's vegetation class is in the band's list, none of them
    cultivates and none is to be sown within the tile's offset days.

    flag [N, NI, B] bool; cv [N, NI]; af [N, B]; act [N, NI] (ni_active);
    irr_tile [NI] long; offset_max [N, NI] (irr_con offset, days); active,
    tile_of [N, NC]; cultivating [N, B, NC] (int); start_doy [N, NC]
    (cgrid->start.day_in_year); doy, leap 0-d long tensors of the step
    (dmy day_in_year, leap_year(year)). Returns the new flag."""
    N, NI, B = flag.shape
    NC = active.shape[1]
    out = flag.clone()
    for i in range(NI):
        crop = torch.zeros(N, B, dtype=torch.bool, device=flag.device)
        cult = torch.zeros_like(crop)
        offs = torch.zeros_like(crop)
        for c in range(NC):
            m = (active[:, c] & (tile_of[:, c] == irr_tile[i]))[:, None]
            crop = crop | m
            cult = cult | (m & (cultivating[:, :, c] > 0))
            pd = start_doy[:, c]
            pd = torch.where(pd < doy, pd + DAYS_PER_YEAR + leap, pd)
            offs = offs | (m & ((pd - doy) < offset_max[:, i])[:, None])
        ok = (act[:, i] & (cv[:, i] > 0))[:, None] & (af > 0) \
            & out[:, i]
        clear = ok & crop & ~cult & ~offs
        out[:, i] = out[:, i] & ~clear
    return out


def wofost_water(amount, cc, active, tile_of, irr_tile, act, af, n_band):
    """irr_wofost's water (irr_use.c:568-603): per irrigated entry and band,
    the sum over the band's crop list of List(Irrigation) x Cc x MM_PER_CM
    for the crops of the entry's tile (the same on every band; get_withdrawn
    applies it where Cv > 0 and AreaFract > 0). amount [NC] float32 (List of
    the day, List.c:9-24), cc [N, NC]. Returns (water [N, NI, B], bad [N]):
    `bad` is the check of :606-640, a crop with management irrigation on a
    tile the irrigation module does not irrigate (log_err in C)."""
    N, NC = cc.shape
    NI = irr_tile.shape[0]
    dev = cc.device
    am = amount.to(torch.float64)
    out = torch.zeros(N, NI, dtype=torch.float64, device=dev)
    for i in range(NI):
        acc = torch.zeros(N, dtype=torch.float64, device=dev)
        for c in range(NC):
            m = active[:, c] & (tile_of[:, c] == irr_tile[i])
            acc = torch.where(m, acc + am[c] * cc[:, c] * MM_PER_CM, acc)
        out[:, i] = acc
    bad = torch.zeros(N, dtype=torch.bool, device=dev)
    anyband = (af > 0).any(1)
    for c in range(NC):
        irrd = torch.zeros(N, dtype=torch.bool, device=dev)
        for i in range(NI):
            irrd = irrd | (act[:, i] & (tile_of[:, c] == irr_tile[i]))
        bad = bad | (active[:, c] & anyband & ~irrd
                     & (am[c] * cc[:, c] * MM_PER_CM > 0))
    return out[:, :, None].expand(N, NI, n_band).contiguous(), bad


def evap_outputs(esoil, transp, canop, cc, af, growing, active, tile_of):
    """OUT_CROP_EVAP_BARE, _TRANSP_VEG, _EVAP_CANOP and _EVAP of
    crop_put_rate_data (crop_output.c:1003-1032), [N, NC] each, in the C
    order: bands, the crop list, layers; EVAP adds the three running sums of
    its class after each unit (:1025-1031).

    esoil [N, T, B] (layer[0].esoil; the other layers' are 0), transp
    [N, T, B, L], canop [N, T, B] (veg_var canopyevap): VIC's fluxes of the
    step on the crop's tile, 0 on a band without area; growing
    [N, B, NC] int after crop_run."""
    if _fusable(cc):
        return _evap_compiled()(esoil, transp, canop, cc, af, growing,
                                active, tile_of)
    return _evap(esoil, transp, canop, cc, af, growing, active, tile_of)


def _evap(esoil, transp, canop, cc, af, growing, active, tile_of):
    """evap_outputs with each class's running sums as their own [N]
    tensors (no writes into columns)."""
    N, NC = cc.shape
    B = af.shape[1]
    L = transp.shape[-1]
    dev = cc.device
    z = lambda: torch.zeros(N, dtype=torch.float64, device=dev)
    bare = [z() for _ in range(NC)]
    tr = [z() for _ in range(NC)]
    can = [z() for _ in range(NC)]
    ev = [z() for _ in range(NC)]
    for b in range(B):
        for c in range(NC):
            on = active[:, c]
            t = tile_of[:, c:c + 1]
            g = growing[:, b, c].to(torch.float64)
            w = lambda x: ((x * af[:, b]) * cc[:, c]) * g
            e0 = esoil[:, :, b].gather(1, t)[:, 0]
            tl = transp[:, :, b, :].gather(
                1, t[:, :, None].expand(N, 1, L))[:, 0]
            cv = canop[:, :, b].gather(1, t)[:, 0]
            nb, nt = bare[c], tr[c]
            for l in range(L):
                nb = nb + w(e0 if l == 0 else torch.zeros_like(e0))
                nt = nt + w(tl[:, l])
            nc_ = can[c] + w(cv)
            ne = ev[c] + ((nb + nt) + nc_)
            bare[c] = torch.where(on, nb, bare[c])
            tr[c] = torch.where(on, nt, tr[c])
            can[c] = torch.where(on, nc_, can[c])
            ev[c] = torch.where(on, ne, ev[c])
    st = lambda x: torch.stack(x, 1)
    return {'OUT_CROP_EVAP': st(ev), 'OUT_CROP_EVAP_BARE': st(bare),
            'OUT_CROP_EVAP_CANOP': st(can), 'OUT_CROP_TRANSP_VEG': st(tr)}


_COMPILED_EVAP = [None]


def _evap_compiled():
    if _COMPILED_EVAP[0] is None:
        from ..crop.wofost_npk.fuse import compile_region
        _COMPILED_EVAP[0] = compile_region(_evap)
    return _COMPILED_EVAP[0]
