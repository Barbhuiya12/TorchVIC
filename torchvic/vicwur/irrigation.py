# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR irrigation (C8) per-tile kernels: pure tensor functions.

Every function takes tensors and returns tensors, batched over cells (C),
irrigated tiles (I, the compact index of irr_con_map), snow bands (B),
layers (L) and frost areas (F).  No loop over cells and no host sync; the
only Python loops run over the layers (3), frost areas and the tile/band
sums, to keep the C summation order.

C references are VIC-WUR 3.0 plugins/irrigation/src
(<data root>/build/vicwur3/src_tree/vic/plugins/irrigation/src):
  irr_run.c:39-70     irr_update_step_vars   paddy Ksat before vic_run
  irr_run.c:107-221   irr_run_requirement    offset counter, requirement, flag
  irr_run.c:260-372   irr_run_shortage       shortage and water stress
  irr_use.c:62-105    irr_set_demand         D0, G, E, D
  irr_use.c:217-269   irr_apply              application into layer 0
  irr_use.c:274-324   irr_leftover           re-apply the leftover first
  irr_use.c:329-380   irr_potential          potential mode
  irr_use.c:386-526   irr_wateruse           pro-rata share of the consumed water
  irr_use.c:531-608   irr_wofost             IRRTAB water (the amount comes
                                             from crop_feedback.wofost_water)
  irr_use.c:650-687   irr_get_withdrawn      the order of the above
irr_return_leftover (irr_use.c:112-212) is commented out in the driver
(irr_use.c:686) and is not here.

Conventions.  A tile/band is "active" when Cv > 0 and AreaFract > 0
(irr_run.c:140,148); inactive entries keep their previous leftover (frozen,
so the leftover is lost while Cv is 0) and get zero
requirement/flag/received.  The C macros are reproduced: `max(a, b)` is
`(a > b) ? a : b` (vic_def.h:92).  Moisture is mm of the layer's total water
(ice included); liquid is moist - ice weighted by frost_fract.  The paddy
flood height is 0 (irrigation.h:33).  Ksat is in mm/d, as in the C code
(pow(Ksat, 0.33) is unit dependent).
"""
import torch

PADDY_FLOOD_HEIGHT = 0.0
KSAT_EXPT = 0.33          # plugin_init_library.c:116


def _cmax(a, b):
    """C macro max(a, b) = (a > b) ? a : b."""
    return torch.where(a > b, a, b)


def active_mask(cv, area_fract):
    """[C,I] x [C,B] -> [C,I,B]: Cv > 0 and AreaFract > 0."""
    return (cv > 0).unsqueeze(-1) & (area_fract > 0).unsqueeze(1)


def paddy_ksat(ksat0, paddy, nbands, active=None, expt=KSAT_EXPT):
    """Layer-0 Ksat of every tile/band (irr_run.c:62-67 and :118,153-156).

    ksat0 [C]; paddy [C,I] bool; returns [C,I,B].  active=None is the
    VIC-WUR 3.0 pre-vic_run reset (irr_update_step_vars: every band, no
    guard); with `active` [C,I,B] it is the guarded reset of
    irr_run_requirement.
    """
    k = ksat0[:, None, None].expand(paddy.shape + (nbands,))
    sel = paddy.unsqueeze(-1).expand_as(k)
    if active is not None:
        sel = sel & active
    return torch.where(sel, torch.pow(k, expt), k)


def offset_step(offset, prev_cv, cv, offset_max, new_day):
    """Daily season-offset counter (irr_run.c:123-138).

    offset [C,I,B] (any numeric dtype), prev_cv [C,I,B], cv [C,I],
    offset_max [C,I], new_day scalar bool tensor (day_in_year differs from
    the previous step, or step 0).  Runs for every tile and band, with no
    Cv or AreaFract guard.  Returns (offset, prev_cv).
    """
    cvb = cv.unsqueeze(-1)
    changed = cvb != prev_cv
    off = torch.where(changed, torch.zeros_like(offset), offset)
    off = torch.where(off < offset_max.unsqueeze(-1).to(off.dtype), off + 1,
                      off)
    pcv = cvb.expand_as(prev_cv)
    return (torch.where(new_day, off, offset),
            torch.where(new_day, pcv, prev_cv))


def liquid_layers(moist, ice, frost_fract):
    """moist[k] = sum_l (moist_k - ice_k,l) * frost_l (irr_run.c:166-171).

    moist [C,I,B,L]; ice [C,I,B,L,F]; frost_fract [C,F].  Sequential in l.
    """
    liq = torch.zeros_like(moist)
    for l in range(ice.shape[-1]):
        liq = liq + (moist - ice[..., l]) * frost_fract[:, None, None, None, l]
    return liq


def requirement(moist, ice, frost_fract, root, wcr, wfc, max_moist0,
                leftover, offset, offset_max, paddy, cv, area_fract,
                flood=PADDY_FLOOD_HEIGHT, hold=None):
    """irr_run_requirement without the offset update (irr_run.c:140-217).

    moist [C,I,B,L] end-of-step layer moisture; ice [C,I,B,L,F];
    frost_fract [C,F]; root [C,I,B,L] root fractions; wcr [C,I,B,L] dynamic
    layer Wcr; wfc [C,L] static Wfc; max_moist0 [C]; leftover, offset
    [C,I,B]; offset_max [C,I]; paddy [C,I] bool; cv [C,I]; area_fract
    [C,B].  Returns (req [C,I,B], flag [C,I,B] bool); both 0/False on
    inactive tiles.  The requirement is never clamped.

    hold [C,I,B] bool (the previous day's flag), or None: VIC-WUR 3.0's
    flag needs the root zone below Wcr (irr_run.c:165-177), so an
    application that lifts it above Wcr closes the flag for the next day
    and the next day's ET opens it again: an on/off pulse on alternate days
    wherever the supply is short of the requirement (VIC-WUR 3.0 itself
    does this, e.g. R1 G4e).  With `hold` the flag of those tiles stays on
    while the requirement is positive; the connector passes the tiles
    flagged and short of supply the day before (a fully supplied tile keeps
    VIC-WUR 3.0's flag); None is VIC-WUR 3.0 exactly.
    """
    act = active_mask(cv, area_fract)
    liq = liquid_layers(moist, ice, frost_fract)
    zero = torch.zeros_like(leftover)
    tm, tw, tf = zero, zero, zero
    for k in range(moist.shape[-1]):
        r = root[..., k] > 0.
        tm = tm + torch.where(r, liq[..., k], zero)
        tw = tw + torch.where(r, wcr[..., k], zero)
        tf = tf + torch.where(r, wfc[:, None, None, k].expand_as(zero), zero)
    mm0 = max_moist0[:, None, None]
    pad = paddy.unsqueeze(-1)
    lo = leftover
    req = torch.where(pad, (mm0 + flood) - (liq[..., 0] + lo),
                      tf - (tm + lo))
    open_ = offset >= offset_max.unsqueeze(-1)
    f_pad = pad & ((mm0 - (liq[..., 0] + lo)) > 0) & (req > 0)
    f_cv = ((tw - (tm + lo)) > 0) & (req > 0)
    on = f_pad | f_cv
    if hold is not None:
        on = on | (hold & (req > 0))
    flag = open_ & on & act
    return torch.where(act, req, zero), flag


def demand(req, flag, cv, area_fract, gw_frac, eff):
    """irr_set_demand (irr_use.c:62-105).

    req, flag [C,I,B]; cv [C,I]; area_fract [C,B]; gw_frac, eff [C,I].
    Returns (D, G, E, D0): D = D0 / E (the wu_force demand, mm), G the
    demand-weighted groundwater fraction, E the demand-weighted efficiency
    (the wu_force consumption_frac) and D0 the crop demand.  With D0 = 0 all
    four are 0.  E = 0 with D0 > 0 gives inf, as in C.  The sum runs over
    tiles, then bands, in C order.
    """
    act = active_mask(cv, area_fract) & flag
    z = torch.zeros_like(cv[:, 0])
    d0, g, e = z, z, z
    for i in range(req.shape[1]):
        for b in range(req.shape[2]):
            dc = torch.where(act[:, i, b],
                             req[:, i, b] * cv[:, i] * area_fract[:, b], z)
            d0 = d0 + dc
            g = g + dc * gw_frac[:, i]
            e = e + dc * (eff[:, i, b] if eff.dim() == 3 else eff[:, i])
    pos = d0 > 0
    one = torch.ones_like(d0)
    g = torch.where(pos, g / torch.where(pos, d0, one), z)
    e = torch.where(pos, e / torch.where(pos, d0, one), z)
    d = torch.where(pos, d0 * (1 / torch.where(pos, e, one)), z)
    return d, g, e, d0


def apply(moist0, received, wfc0, max_moist0, paddy, efficient):
    """irr_apply (irr_use.c:245-269) for any matching shapes.

    moist0 [C,I,B] layer-0 total moisture; received [C,I,B]; wfc0,
    max_moist0 [C]; paddy [C,I] bool; efficient: bool (EFFICIENT_IRRIGATION).
    Returns (new_moist0, applied, leftover) with applied and leftover the
    increments of this call (zero where received <= 0).
    """
    pad = paddy.unsqueeze(-1)
    if efficient:
        cap = torch.where(pad, max_moist0[:, None, None],
                          wfc0[:, None, None]) - moist0
    else:
        cap = max_moist0[:, None, None] - moist0
    cap = _cmax(torch.zeros_like(cap), cap)
    go = received > 0.
    app = torch.where(received < cap, received, cap)
    app = torch.where(go, app, torch.zeros_like(app))
    left = torch.where(go, received - app, torch.zeros_like(app))
    return moist0 + app, app, left


def leftover_reapply(moist0, leftover, applied, active, wfc0, max_moist0,
                     paddy, efficient):
    """irr_leftover (irr_use.c:299-323): re-apply last step's leftover.

    Returns (moist0, leftover, applied); tiles with leftover <= 0 or
    inactive are unchanged.
    """
    sel = active & (leftover > 0)
    tmp = torch.where(sel, leftover, torch.zeros_like(leftover))
    m, a, l = apply(moist0, tmp, wfc0, max_moist0, paddy, efficient)
    new_left = torch.where(sel, torch.zeros_like(leftover), leftover) + l
    return m, new_left, applied + a


def get_withdrawn(moist0, leftover, req, flag, active, wfc0, max_moist0,
                  paddy, efficient, cv, area_fract, potential=False,
                  available=None, wofost=None):
    """irr_get_withdrawn (irr_use.c:650-687): one P3 irrigation step.

    Resets received/applied, re-applies the leftover, then irrigates with
    `potential` (irr_potential: received = requirement on flagged tiles) or
    with `available` [C] = wu_var[IRR].consumed (irr_wateruse: share
    req * available / D0 on flagged tiles; skipped when available <= 0), and
    then with `wofost` [C,I,B] mm (irr_wofost, applied on every active tile
    and band with no flag).  The gating of the three modes by the options
    (POTENTIAL || WATERUSE || WOFOST) is the caller's.

    Returns (moist0, leftover, received, applied).
    """
    z = torch.zeros_like(leftover)
    received, applied = z, z
    moist0, leftover, applied = leftover_reapply(
        moist0, leftover, applied, active, wfc0, max_moist0, paddy,
        efficient)
    if potential:
        rec = torch.where(flag & active, req, z)
        moist0, a, l = apply(moist0, rec, wfc0, max_moist0, paddy, efficient)
        applied, leftover, received = applied + a, leftover + l, received + rec
    else:
        if available is not None:
            d0 = z[:, 0, 0]
            for i in range(req.shape[1]):
                for b in range(req.shape[2]):
                    d0 = d0 + torch.where(
                        flag[:, i, b] & active[:, i, b],
                        req[:, i, b] * cv[:, i] * area_fract[:, b],
                        z[:, 0, 0])
            pos = available > 0
            frac = torch.where(pos, available / d0, torch.zeros_like(d0))
            rec = torch.where(flag & active & (frac > 0)[:, None, None],
                              req * frac[:, None, None], z)
            moist0, a, l = apply(moist0, rec, wfc0, max_moist0, paddy,
                                 efficient)
            applied, leftover, received = (applied + a, leftover + l,
                                           received + rec)
        if wofost is not None:
            rec = torch.where(active, wofost, z)
            moist0, a, l = apply(moist0, rec, wfc0, max_moist0, paddy,
                                 efficient)
            applied, leftover, received = (applied + a, leftover + l,
                                           received + rec)
    return moist0, leftover, received, applied


def balance_residual(received, leftover_prev, applied, leftover_new):
    """received + leftover_prev - applied - leftover_new, per tile and band.

    Zero (to rounding) on every active tile: the full water invariant of the
    irrigation application (the C check irr_use.c:512-515 covers only the
    irr_wateruse path).
    """
    return received + leftover_prev - applied - leftover_new


def shortage_stress(moist, ice, frost_fract, root, wcr, wpwp, cv, area_fract,
                    flag, share_layer_moist, water_stress_in=None):
    """irr_run_shortage (irr_run.c:260-372), run after the application.

    moist [C,I,B,L] post-application; wcr [C,I,B,L]; wpwp [C,L]; flag
    [C,I,B] bool; share_layer_moist bool (SHARE_LAYER_MOIST);
    water_stress_in [C,I,B,L] keeps the previous layer stress on inactive
    tiles (default zeros).  Shortage sums only the layers above the lowest
    one; the lowest layer feeds the SHARE_LAYER_MOIST test.
    Returns (shortage [C,I,B], layer_stress [C,I,B,L],
    cell_stress [C,I,B]); cell_stress is 0 on inactive tiles.
    """
    act = active_mask(cv, area_fract)
    L = moist.shape[-1]
    liq = liquid_layers(moist, ice, frost_fract)
    z = torch.zeros_like(liq[..., 0])
    tm, tw = z, z
    for k in range(L - 1):
        r = root[..., k] > 0.
        tm = tm + torch.where(r, liq[..., k], z)
        tw = tw + torch.where(r, wcr[..., k], z)
    lm = liq[..., L - 1]
    lw = wcr[..., L - 1]
    lr = root[..., L - 1]
    sh = torch.where((tm < tw) & flag, tw - tm, z)
    sh = torch.where(sh < 0.0, z, sh)
    share = ((tm >= tw) & (lm >= lw) & (tw > 0.)) | \
            ((tm >= tw) & ((1 - lr) >= 0.5)) | ((lm >= lw) & (lr >= 0.5))
    if not share_layer_moist:
        share = torch.zeros_like(share)
    wp = wpwp[:, None, None, :]
    ramp = (liq - wp) / (wcr - wp)
    ws = torch.where(liq >= wcr, torch.ones_like(liq),
                     torch.where((liq >= wp) & (liq < wcr), ramp,
                                 torch.zeros_like(liq)))
    ws = torch.where(share.unsqueeze(-1), torch.ones_like(ws), ws)
    prev = torch.zeros_like(ws) if water_stress_in is None else water_stress_in
    ws = torch.where(act.unsqueeze(-1), ws, prev)
    cs = z
    for k in range(L):
        cs = cs + ws[..., k] * root[..., k]
    cs = torch.where(act, cs, z)
    return torch.where(act, sh, z), ws, cs
