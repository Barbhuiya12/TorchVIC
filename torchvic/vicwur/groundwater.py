# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR 3.0 groundwater coupling (C15, docs/connectors/vicwur.md).

VIC-WUR 3.0 couples VIC to MODFLOW 6 offline, month by month, through the
Python driver `plugins/modflow/*.py` (paths under
`<data root>/build/vicwur3/src_tree/vic/`). This module is that exchange for
the TorchVIC groundwater engine (torchvic.groundwater, the MODFLOW 6 GWF
equations): the VIC-WUR 3.0 package set built from its static grids, and the
per-period exchange kernels. The connector is the `vicwur3` profile of
`torchvic/coupling/connectors/groundwater.py`.

What VIC-WUR 3.0 exchanges (one MODFLOW stress period per calendar month,
one time step of `days` days; `vic_global_online_full.py:41-84`):

  VIC -> MF  recharge  OUT_GWRECHARGE (runoff_GWM's Q12 out of the bottom
             layer, `vic_run/src/runoff.c:709-737`), the month's sum in mm,
             / days / 1000 = m/d (`vic_runner.py:257`), times
             cellarea / (delr delc) (`config_module.py:293-310`); negative
             values become 0 (:298).
  routing -> MF  river stage  the month's mean OUT_DISCHARGE (m3 s-1,
             AGG_TYPE_AVG, `rout_output.c:111`) through Manning
             (`config_module.py:312-370`).
  MF -> routing  baseflow  -(RIV + DRN) of the month (`mf_run.py:271-322`),
             m3 s-1, written as the DISCHARGE plugin forcing of the NEXT month
             (`mf_run.py:324-346`; held, `rout_force.c`), the routing's forced
             inflow (`rout_basin_run.c:88-110`); month 1 gets 0
             (`vic_runner.py:15-65`).
  MF -> VIC  capillary rise  the EVT outflow (FOC coupling) as mm added to
             soil layer 3 (index 2) in the state file VIC restarts the next
             month from, with a saturation cascade upward
             (`mf_run.py:349-377`, `vic_runner.py:267-321`).
  pumping    none: `ts_gwabstract` is 0 (`mf_run.py:21,37`). With GWM the C
             gives every sector's demand to surface water (`wu_run.c:94-97`),
             takes nothing from the soil (`wu_run_local.c:96,398`), and the
             non-renewable withdrawal books no deficit (`wu_nonrenewable.c:
             211,220`): its water has no source. In mode `fixed` TorchVIC
             takes it from MODFLOW wells.

Modes (the connector's `mode`):
  exact  VIC-WUR 3.0's exchange: monthly period, held rate, negative
         recharge clipped, the Manning precedence of config_module.py:333,
         the cascade's global `.any()` stop and the loss of the top layer's
         excess, no wells.
  fixed  the same packages, with: signed recharge, Manning as written in
         the comment ((n Q / (W sqrt S))^(3/5)), the baseflow volume handed on
         whole (not a held rate), the cascade per cell with the top excess
         handed back to the aquifer, the non-renewable withdrawal as wells.
  Either mode steps monthly or daily (`period`); daily is recommended.

Fixed in every mode (Python-driver bugs): the 1/days factor on the baseflow
(`mf_run.py:319`: the budget q is a rate in m3/d), the EVT rate taken as a
monthly volume and truncated to int (`mf_run.py:369-375`), the Cv^2 state
edit (`vic_runner.py:275`, cpr * Cv added to each tile's mm), the edit of
band 0 only (`vic_runner.py:289`), the float32 head file between months
(`mf_run.py:239`), and the month-loop file plumbing. The `[:-1]` of
`np.unique(ldd)` (`config_module.py:383`) drops VIC-WUR 3.0's no-data id;
here the ids are the cells' own (no no-data id among VIC cells).

Static grids (VIC-WUR 3.0 file -> key here): dem_ini dem, aqdepth_ini aqdepth,
ksat_log, ksat_l1_conf_log, ksat_l2_conf_log, cell_area cellarea,
bankfull_discharge qbank, riv_slope, Z0_floodplain z0, min_dem, KQ3 kq3,
spe_yi_inp sy, landmask, initial head head0 [2], ibound, ldd, capillary rise
cpsurface, cprate, cpdistinct, max_moist [layer].

All set-up is numpy on the host (once); the per-period kernels are torch on
the device with no host read and no data-dependent branch.
"""
import numpy as np
import torch

RHO, MU, G = 1000, 0.001, 9.81                 # config_module.py:246
MANNING, RESISTANCE = 0.045, 1.0                # config_module.py:315
KQ3_MIN = 1.0e-4                                # config_module.py:378
INVALID_BASE = -900000.0                        # config_module.py:380
DEG_M = 111 * 1000                              # config_module.py:143-144

STATIC_KEYS = ('dem', 'aqdepth', 'ksat_log', 'ksat_l1_conf_log',
               'ksat_l2_conf_log', 'cellarea', 'qbank', 'riv_slope', 'z0',
               'min_dem', 'kq3', 'sy', 'landmask', 'head0', 'ibound', 'ldd',
               'cpsurface', 'cprate', 'cpdistinct', 'max_moist')


# ---------------------------------------------------------------------------
# VIC-WUR 3.0 package set (config_module.py), numpy on the full (R, C) grid
# ---------------------------------------------------------------------------
def aqdepth(lm, aq):
    """cal_aqdepth (:211-215)."""
    a = np.where(np.isnan(lm), 200, aq)
    a = np.where(np.isnan(a), 200, a)
    return np.where(a >= 0, a, 200)


def top(lm, dem):
    """config.dem (:146-148) = cal_toplayer_elevation (:217-221)."""
    return np.where(np.isnan(lm), 0, dem)


def botm(tp, aq):
    """cal_botlayer_elevation (:223-233): layer 1 is 10 % of the aquifer."""
    b1 = tp - (aq * 0.1)
    b2 = tp - aq
    return [np.where(np.isnan(b1), 0, b1), np.where(np.isnan(b2), 0, b2)]


def npf(s, aq, delr, delc):
    """get_npf_param (:243-291): k_hor, k_ver, stor (two layers each), in the
    C operation order. `delr`, `delc` are config.delrow, config.delcol."""
    lm = s['landmask']
    d1i = aq * 0.1
    d2i = aq - d1i
    d1 = np.where(np.isnan(lm), np.nan, d1i.copy())
    d2 = np.where(np.isnan(lm), np.nan, d2i.copy())
    ksat = RHO * (10 ** s['ksat_log']) * (G / MU) * 24.0 * 3600
    k2c = RHO * (10 ** s['ksat_l2_conf_log']) * (G / MU) * 24.0 * 3600
    k1c = RHO * (10 ** s['ksat_l1_conf_log']) * (G / MU) * 24.0 * 3600
    kh2i = np.where(~np.isnan(k2c), k2c, ksat)
    kh1i = np.where(~np.isnan(k1c), k1c, ksat)
    kh2i = np.clip(kh2i, a_min=0.01, a_max=None)
    kh1i = np.clip(kh1i, a_min=0.01, a_max=None)
    ca = s['cellarea']
    kv1i = np.where(kh1i > -999, kh1i * ca / (delc * delr),
                    (10.0 * ca / (delc * delr)))
    kv2 = np.where(kh2i > -999, (10.0 * ca / (delc * delr)),
                   (10.0 * ca / (delc * delr)))
    kv1 = np.maximum(d1 / 5000, kv1i)
    kd1 = np.maximum(30, kh1i * d1)
    kd2 = np.maximum(30, kh2i * d2)
    kh1 = kd1 / d1
    kh2 = kd2 / d2
    kh1[kh1 < 0] = 20
    kh2[kh2 < 0] = 190
    kv1[kv1 < 0] = 1e10
    kv2[kv2 < 0] = 1e10
    missing = s['aqdepth'].flat[0]                 # :138, aqdepth_ini[0][0]
    sy = np.where(lm >= 1, s['sy'], missing)
    sy = np.maximum(0.01, sy)
    sy = np.minimum(1, sy)
    sy = np.where(aq > -999.9, np.maximum(sy, 0.11), sy)
    fill = lambda k, v: np.where(np.isnan(k), v, k)
    return ([fill(kh1, 20), fill(kh2, 20)], [fill(kv1, 20), fill(kv2, 20)],
            [fill(sy, 0.1), fill(sy, 0.1)])


def riv_static(s, tp):
    """The discharge-independent part of get_riv_param (:312-358). Returns a
    dict of (R, C) arrays; `stage(...)` adds the discharge."""
    lm, ca = s['landmask'], s['cellarea']
    min_dem2 = np.where(s['min_dem'] < 0, 0, s['min_dem'])
    z1 = np.maximum(min_dem2, s['z0'])
    z2 = np.where(s['z0'] < 0, z1, s['z0'])
    with np.errstate(invalid='ignore'):
        width = 4.8 * (s['qbank'] ** 0.5)
        slope = np.where(width > 0, s['riv_slope'], np.nan)
        slope_used = np.where(slope > 0.00005, slope, np.nan)
        head_ini = np.where(width > 30, z2, tp)
        head_ini = np.where(np.isnan(lm), 0, head_ini)
        d_bkfl = ((MANNING * (s['qbank']) ** 0.5)
                  / (4.8 * slope_used ** 0.5)) ** (3.0 / 5.0)
        bot_bkfl = min_dem2 - d_bkfl
        den = np.where(width > 30, width * (ca * 2) ** 0.5, 0)
        riv_cond = 1 / RESISTANCE * den
        riv_cond = np.where(np.isnan(lm), 0, riv_cond)
        drn_width = np.where(riv_cond == 0, np.maximum(10, width), 0)
        drn_cond = np.where(riv_cond == 0, (1 / RESISTANCE) * drn_width
                            * ((ca * 2) ** 0.5), 0)
        bot = np.where(riv_cond > 0, bot_bkfl, head_ini)
        bot = np.where(np.isinf(bot), 0, bot)
        bot = np.where(np.isnan(lm), np.nan, bot)
        cond = np.where(riv_cond > 0, riv_cond, drn_cond)
        cond = np.where(np.isnan(lm), np.nan, cond)
    return dict(width=width, slope_used=slope_used, head_ini=head_ini,
                bot_bkfl=bot_bkfl, d_bkfl=d_bkfl, riv=riv_cond > 0, rbot=bot, cond=cond,
                lm=lm)


def riv_records(r):
    """Cells with an RIV record (:363-369): stage, rbot, cond finite, cond
    != 0, land. The stage is finite wherever rbot is (stage() maps NaN to 0
    before the Q < 0 rule), so the set does not depend on the discharge."""
    st0 = stage_np(r, np.zeros_like(r['rbot']), exact=True)
    ok = ~(np.isnan(st0) | np.isnan(r['rbot']) | np.isnan(r['cond'])
           | (np.nan_to_num(r['cond']) == 0) | np.isnan(r['lm']))
    return ok


def stage_np(r, q, exact=True):
    """The RIV stage for discharge q (m3 s-1), numpy (:333-356)."""
    q = np.where(np.isfinite(q), q, 0.0)                 # set_ts_discharge
    with np.errstate(invalid='ignore', divide='ignore'):
        if exact:
            d1 = (MANNING * q) / (r['width'] * r['slope_used'] ** 0.5) \
                ** (3.0 / 5.0)
        else:
            d1 = ((MANNING * q) / (r['width'] * r['slope_used'] ** 0.5)) \
                ** (3.0 / 5.0)
        d = np.where(d1 < 0.01, 0, d1)
        d = np.where(np.isnan(d), 0, d)
        h = r['bot_bkfl'] + d
        h2 = np.where(np.isnan(h), r['head_ini'], h)
        hc = np.where(r['riv'], h2, r['head_ini'])
        hc = np.where(np.isnan(hc), 0, hc)
        hc = np.where(np.isnan(r['lm']), np.nan, hc)
        return np.where(q < 0, r['rbot'], hc)


def drn(s, sy0):
    """get_drn_param (:372-405): (elevation, conductance) on the grid; the
    conductance is 0 where there is no record. `sy0` is stor[0] of npf()."""
    lm, z0 = s['landmask'], s['z0']
    ldd = s['ldd']
    valid = (lm == 1)
    base = np.full_like(z0, np.nan)
    ids = np.unique(ldd[~np.isnan(ldd)]) if np.issubdtype(
        np.asarray(ldd).dtype, np.floating) else np.unique(ldd)
    for uid in ids:
        sel = (ldd == uid) & valid
        if sel.any():
            base[ldd == uid] = z0[sel].min()
    with np.errstate(invalid='ignore'):
        aq_mask = s['aqdepth_c'] > -9999.9
        base = np.where(aq_mask, np.maximum(z0, base), base)
        used = np.where(valid, base, INVALID_BASE)
        kq3 = np.clip(s['kq3'], KQ3_MIN, 1.0)
        kq3_sy = np.where(valid, kq3 * sy0, 0.0)
        ar = np.where(used != INVALID_BASE, kq3_sy * s['cellarea'], 0.0)
        ar = np.where(np.isnan(lm), 0, ar)
    on = ~(np.isnan(ar) | (ar == 0))
    return used, np.where(on, ar, 0.0)


def evt(s):
    """get_cpr_param (:427-442): EVT records (surface, rate, depth) where
    all three and the land mask are finite and the rate is not 0."""
    sf, rt, dp = s['cpsurface'], s['cprate'], s['cpdistinct']
    on = ~(np.isnan(sf) | np.isnan(rt) | np.isnan(dp)
           | np.isnan(s['landmask'])) & (np.nan_to_num(rt) != 0)
    return on, np.where(on, sf, 0.0), np.where(on, rt, 0.0), \
        np.where(on, dp, 1.0)


def chd_cells(s):
    """get_chd_input (:407-425): ibound == 2, both layers, head 0."""
    return np.asarray(s['ibound']) == 2


def initial_head(s):
    """get_initial_head (:235-241)."""
    h = np.asarray(s['head0'], float)
    return [np.where(np.isnan(h[0]), 0, h[0]), np.where(np.isnan(h[1]), 0,
                                                         h[1])]


# ---------------------------------------------------------------------------
# per-period kernels (torch, device, no host read)
# ---------------------------------------------------------------------------
def stage(r, q, exact=True):
    """`stage_np` on tensors: r holds the record cells' static fields as
    tensors (width, slope_used, bot_bkfl, head_ini, riv, rbot), q [n] the
    period's mean discharge on those cells."""
    q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
    num = MANNING * q
    den = r['width'] * r['slope_used'] ** 0.5
    d1 = (num / den ** (3.0 / 5.0) if exact
          else (num / den) ** (3.0 / 5.0))
    z = torch.zeros_like(q)
    d = torch.where(d1 < 0.01, z, d1)
    d = torch.where(torch.isnan(d), z, d)
    cap = r.get('d_bkfl')
    if cap is not None and not exact:
        # fixed mode: the stage does not rise above the bank (bankfull
        # depth); above it the river spreads on the floodplain. Unbounded,
        # a flood wave of 1e5 m3 s-1 on a 1e-4 slope is 40 m deep, the
        # river pushes that head into the aquifer and the next day's
        # discharge swings negative (ISC 2005)
        d = torch.minimum(d, torch.nan_to_num(cap, nan=1e30))
    h = r['bot_bkfl'] + d
    h2 = torch.where(torch.isnan(h), r['head_ini'], h)
    hc = torch.where(r['riv'], h2, r['head_ini'])
    hc = torch.where(torch.isnan(hc), z, hc)
    return torch.where(q < 0, r['rbot'], hc)


def recharge_rate(gw_mm, days, cellarea, delr, delc, exact=True):
    """The RCH rate (m/d over delr*delc) of a period whose OUT_GWRECHARGE
    sum is `gw_mm` (vic_runner.py:257; config_module.py:293-310). exact
    clips negative values to 0 (:298)."""
    r = gw_mm / days / 1000
    r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
    r = (r * cellarea) / (delc * delr)
    if exact:
        r = torch.where(r < 0, torch.zeros_like(r), r)
    return r


def capillary_rise(moist, cv, live, cpr, max_moist, layer=2, exact=True):
    """The capillary-rise state edit (vic_runner.py:267-321) on the tile
    state moist [cell, tile, band, layer] (mm), cv [cell, tile], live
    [cell, tile, band] (Cv > 0 and AreaFract > 0), cpr [cell] (mm; or
    [cell, tile, band] per column, which reproduces VIC-WUR 3.0's cpr * Cv
    on band 0), max_moist [cell, layer] (the cell's mm).

    `cpr` is added to `layer` of every live tile (VIC-WUR 3.0 adds
    cpr * Cv: the Cv^2 bug, not ported). Then, from `layer` up to 0, where
    the cell's Cv weighted moisture of a band is above max_moist the band's
    tiles are set to max_moist and the excess is added to every tile of the
    layer above. exact: the cascade stops at the first layer with no
    saturated column in the whole domain (VIC-WUR 3.0's
    `if checksaturation.any()` / `break`), and the top layer's excess is
    lost. fixed: per column, no global stop; the top excess is returned.
    Returns (new moist, top excess [cell, band] mm)."""
    m = moist.clone()
    lv = live.to(m.dtype)
    add = cpr[:, None, None] if cpr.dim() == 1 else cpr
    m[..., layer] = m[..., layer] + add * lv
    w = cv[:, :, None]
    go = torch.ones((), dtype=torch.bool, device=m.device)
    lost = torch.zeros(m.shape[0], m.shape[2], dtype=m.dtype,
                       device=m.device)
    for ly in range(layer, -1, -1):
        tot = (m[..., ly] * w).sum(1)                       # [cell, band]
        mx = max_moist[:, ly][:, None]
        sat = tot > mx
        if exact:
            go = go & sat.any()
            sat = sat & go
        exc = torch.where(sat, (tot - mx).clamp_min(0.0),
                          torch.zeros_like(tot))
        on = sat[:, None, :] & live
        m[..., ly] = torch.where(on, mx[:, :, None].expand_as(m[..., ly]),
                                 m[..., ly])
        if ly > 0:
            m[..., ly - 1] = m[..., ly - 1] + torch.where(
                live, exc[:, None, :], torch.zeros_like(m[..., ly - 1]))
        else:
            lost = exc
    return m, lost
