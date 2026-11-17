#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Smoke-test inputs of the VIC-WUR stack on ISC.

Generates what the ISC data set (<data root>/vicwur/isc) does not hold, under
<data root>/vicwur/isc/smoke/ (README.md there states method and source):

  rout_n<S>.nc          routing parameters (read_routing format) from the ISC
                        D8 flow directions
  forcing/efr_{q,b}.Y.nc  EFR forcing placeholders (monthly)
  gw_static.npz         the vicwur3 MODFLOW static values per active cell
  global_isc_<Y>.txt    the VIC-WUR 3.0 global file

None of it is science input; all of it exists to run the stack at 112,825
cells. Usage: smoke_inputs.py [--year 2005] [--rout-steps 4]
"""
import argparse
import os
import sys

import numpy as np
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ISC, ROOT, month_starts, yearly_file   # noqa

OUT = ROOT + '/vicwur/isc/smoke'
ESRI = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1),
        32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
DEG_KM = 111.0
SECTORS = ('mun', 'liv', 'ene', 'man')
OUTVARS = """OUT_PREC OUT_EVAP OUT_RUNOFF OUT_BASEFLOW OUT_SWE OUT_SOIL_MOIST
OUT_TRANSP_VEG OUT_EVAP_CANOP OUT_EVAP_BARE OUT_AIR_TEMP OUT_DISCHARGE
OUT_STREAM_INFLOW OUT_STREAM_RUNOFF OUT_STREAM_MOIST OUT_ROUTING_ERROR
OUT_NONREN_DEFICIT OUT_GDAM_INFLOW OUT_GDAM_DEMAND OUT_GDAM_EFR
OUT_GDAM_STORAGE OUT_GDAM_RELEASE OUT_LDAM_INFLOW OUT_LDAM_DEMAND
OUT_LDAM_EFR OUT_LDAM_STORAGE OUT_LDAM_RELEASE OUT_DE_GW_SECT
OUT_DE_SURF_SECT OUT_DE_NREN_SECT OUT_DE_REM_SECT OUT_DE_TREM_SECT
OUT_WI_GW_SECT OUT_WI_SURF_SECT OUT_WI_DAM_SECT OUT_WI_REM_SECT
OUT_WI_TREM_SECT OUT_WI_NREN_SECT OUT_WITHDRAWN OUT_DEMAND OUT_RETURNED
OUT_CONSUMED OUT_SHORTAGE OUT_REQUIREMENT OUT_RECEIVED OUT_LEFTOVER
OUT_APPLIED OUT_EFR_DISCHARGE OUT_EFR_BASEFLOW OUT_EFR_MOIST OUT_CV
OUT_GWRECHARGE OUT_RECHARGE""".split()


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ network
def network(g):
    """downstream index per active cell (own index = outlet), cell-to-
    downstream distance (m), basin id (outlet index + 1), upstream area
    (km2) and the D8 slope-ready elevation drop."""
    with nc.Dataset(ROOT + '/static/isc_fdir_grit.nc') as d:
        fdir = np.array(np.ma.filled(d['fdir'][:].astype(np.int32), -1))
    idx = -np.ones((g.ny, g.nx), np.int64)
    cells = np.argwhere(g.active)
    n = len(cells)
    idx[g.active] = np.arange(n)
    r, c = cells[:, 0], cells[:, 1]
    down = np.arange(n)
    for code, (dr, dc) in ESRI.items():
        m = np.flatnonzero(fdir[r, c] == code)
        rr, cc = r[m] + dr, c[m] + dc
        ok = (rr >= 0) & (rr < g.ny) & (cc >= 0) & (cc < g.nx)
        t = np.full(len(m), -1)
        t[ok] = idx[rr[ok], cc[ok]]
        sel = t >= 0
        down[m[sel]] = t[sel]
    # a cycle never reaches an outlet: its cells become outlets
    p = down.copy()
    for _ in range(40):
        p = p[p]
    outl = down == np.arange(n)
    bad = ~outl[p]
    down[bad] = np.flatnonzero(bad)
    log('network: %d cells, %d outlets, %d cycle cells broken'
        % (n, int((down == np.arange(n)).sum()), int(bad.sum())))
    p = down.copy()
    for _ in range(40):
        p = p[p]
    basin = (p + 1).astype(np.float64)
    lat = np.asarray(g.lat)[r]
    dy = DEG_KM / 16 * 1e3
    dx = DEG_KM / 16 * 1e3 * np.cos(np.deg2rad(lat))
    dr_, dc_ = r[down] - r, c[down] - c
    dist = np.hypot(dr_ * dy, dc_ * dx)
    dist = np.where(down == np.arange(n), dy, dist)
    # upstream area by topological accumulation (km2)
    indeg = np.bincount(down[down != np.arange(n)], minlength=n)
    acc = g.area[g.active] / 1e6
    stack = list(np.flatnonzero(indeg == 0))
    while stack:
        i = stack.pop()
        d = down[i]
        if d != i:
            acc[d] += acc[i]
            indeg[d] -= 1
            if indeg[d] == 0:
                stack.append(d)
    return cells, down, dist, basin, acc


def gamma_uh(n, dt_s, mean_s, shape):
    t = (np.arange(n) + 0.5) * dt_s
    pdf = t ** (shape - 1) * np.exp(-t / (mean_s / shape))
    return pdf / pdf.sum()


def write_routing(g, cells, down, dist, steps, path):
    n = 6 * steps
    dt = 86400 // steps
    N = len(cells)
    ids = (np.arange(g.ny * g.nx) + 1).reshape(g.ny, g.nx).astype('i4')
    did = ids.copy()
    flat = np.flatnonzero(g.active.reshape(-1))
    did.reshape(-1)[flat] = ids.reshape(-1)[flat][down]
    uh_r = np.zeros((n, g.ny, g.nx))
    uh_i = np.zeros((n, g.ny, g.nx))
    # uh_runoff: one gamma (mean 1 day, shape 2) for all cells
    ur = gamma_uh(n, dt, 86400.0, 2.0)
    uh_r[:, g.active] = ur[:, None]
    # uh_inflow: gamma (shape 3) of mean = cell-to-downstream distance / 1 m/s
    tt = dist / 1.0
    t = (np.arange(n)[:, None] + 0.5) * dt
    th = tt[None, :] / 3.0
    pdf = t ** 2 * np.exp(-t / th)
    pdf /= pdf.sum(0)
    uh_i[:, g.active] = pdf
    with nc.Dataset(path, 'w') as ds:
        ds.title = 'ISC smoke-test routing parameters (not science input)'
        ds.comment = ('D8 from <data root>/static/isc_fdir_grit.nc; uh_runoff '
                      'gamma mean 1 d shape 2; uh_inflow gamma shape 3, mean '
                      'distance/1 m/s; ROUT_STEPS_PER_DAY %d; outlets point '
                      'to themselves' % steps)
        ds.createDimension('lat', g.ny)
        ds.createDimension('lon', g.nx)
        ds.createDimension('time', n)
        for k, a in (('lat', g.lat), ('lon', g.lon)):
            v = ds.createVariable(k, 'f8', (k,))
            v[:] = a
        v = ds.createVariable('time', 'i4', ('time',))
        v.units = 'seconds'
        v[:] = np.arange(n) * dt
        for k, a in (('downstream_id', ids), ('downstream', did)):
            ds.createVariable(k, 'i4', ('lat', 'lon'))[:] = a
        for k, a in (('uh_inflow', uh_i), ('uh_runoff', uh_r)):
            ds.createVariable(k, 'f8', ('time', 'lat', 'lon'), zlib=True,
                              complevel=2)[:] = a
    log('routing: wrote %s (uh length %d)' % (path, n))


# --------------------------------------------------------------------- EFR
def write_efr(g, years, cells, acc, outdir):
    qb = 0.05 * acc ** 0.9                       # bankfull m3/s, see static
    full = lambda a: _grid(g, a)
    for y in years:
        ms, dm = month_starts(y)
        sea = 0.5 + 0.5 * np.sin(2 * np.pi * (ms + 15 - 120) / 365.0)
        q = 0.2 * qb[None, :] * (0.5 + sea[:, None])
        b = np.full((12, len(qb)), 0.05) * (0.5 + sea[:, None])
        yearly_file('%s/efr_q.%d.nc' % (outdir, y), y, g, 'efr_q',
                    np.stack([full(x) for x in q]), 'm3/s', ms,
                    attrs=dict(note='PLACEHOLDER: 0.2 x bankfull x seasonal'))
        yearly_file('%s/efr_b.%d.nc' % (outdir, y), y, g, 'efr_b',
                    np.stack([full(x) for x in b]), 'mm/day', ms,
                    attrs=dict(note='PLACEHOLDER: 0.05 mm/day x seasonal'))
    log('efr: wrote %s' % years)


def _grid(g, a, fill=-1.0e30):
    o = np.full((g.ny, g.nx), fill)
    o[g.active] = a
    return o


# -------------------------------------------------------------- groundwater
def static_values(g, cells, down, dist, basin, acc):
    """The vicwur3 MODFLOW static values per active cell (keys of
    tools/vicwur/gw_runs.py static_values). Rules in the README."""
    N = len(cells)
    r, c = cells[:, 0], cells[:, 1]
    take = lambda a: np.asarray(np.ma.filled(a, np.nan), float)[g.active]
    with nc.Dataset(ROOT + '/vic_params_isc/domain.nc') as d:
        dem = np.array(d['elev'][:], float)
    dem_c = dem[g.active]
    with nc.Dataset(ROOT + '/vicwur/isc/params/modflow_static.nc') as m:
        logk = take(m['log10_k'][:])
        aqd = take(m['aqdepth'][:])
        sy = take(m['Sy'][:])
        h0 = take(m['initial_head'][:])
    # soil moisture capacity [3, N] (vicwur_setup.soil_statics)
    with nc.Dataset(ROOT + '/vicwur/isc/params/params_wur.nc') as p:
        dep = np.array(p['depth'][:], float)[:, g.active]
        bd = np.array(p['bulk_density'][:], float)[:, g.active]
        sd = np.array(p['soil_density'][:], float)[:, g.active]
    max_moist = dep * (1 - bd / sd) * 1000.
    outlet = down == np.arange(N)
    # river slope: elevation drop to the downstream cell / distance
    slope = np.clip((dem_c - dem_c[down]) / dist, 1e-4, 0.05)
    slope[outlet] = 1e-4
    qbank = 0.05 * acc ** 0.9
    # min elevation of the 3x3 neighbourhood among active cells
    pad = np.full((g.ny + 2, g.nx + 2), np.inf)
    pad[1:-1, 1:-1] = np.where(g.active, dem, np.inf)
    mn = np.full((g.ny, g.nx), np.inf)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            mn = np.minimum(mn, pad[dr:dr + g.ny, dc:dc + g.nx])
    min_dem = np.minimum(mn[g.active], dem_c)
    # The 3x3 neighbourhood minimum is a DEM-relief value: on 5 km cells in
    # the Himalaya it lies up to 3 km below the cell, far under the aquifer
    # (20-790 m), so drains and river beds sat hundreds of metres below the
    # water table and released ~800 km3 in the first day. Drain
    # and river levels are held inside the layer they act on: not below the
    # base of layer 1 (VIC-WUR 3.0 botm: 10 % of the aquifer depth).
    b1 = dem_c - 0.1 * aqd
    min_dem = np.maximum(min_dem, b1)
    z0 = min_dem + 0.5 * (dem_c - min_dem)
    # CHD (ibound 2): outlets at or below 20 m (sea outlets)
    ibound = np.where(outlet & (dem_c <= 20.0), 2, 1).astype(float)
    botm2 = dem_c - aqd
    head = np.where(np.isfinite(h0) & (np.abs(h0) < 1e6), h0, dem_c - 5.0)
    head = np.clip(head, botm2 + 0.5, dem_c)
    # cold start without spin-up: the aquifer starts filled to the drain /
    # river level (never above the land surface), so the first month
    # releases the recharge, not stored water the file's water table sits
    # above the drains
    head = np.clip(np.maximum(head, z0), botm2 + 0.5, dem_c)
    # river cells: the cell-mean head starts at the stage of a low flow
    # (10 % of bankfull), not above it: the routing starts empty, and a head
    # 10-50 m over the dry-river stage released ~110 km3 d-1 on day 1
    from torchvic.vicwur import groundwater as W
    c1 = lambda a: np.asarray(a, float).reshape(-1, 1)
    sd = dict(landmask=c1(np.ones(N)), cellarea=c1(g.area[g.active]),
              min_dem=c1(min_dem), z0=c1(z0), qbank=c1(qbank),
              riv_slope=c1(slope))
    rv = W.riv_static(sd, c1(dem_c))
    stg = W.stage_np(rv, 0.1 * c1(qbank), exact=False)[:, 0]
    isr = rv['riv'][:, 0] & np.isfinite(stg)
    head = np.where(isr, np.minimum(head, np.maximum(stg, botm2 + 0.5)),
                    head)
    head = np.where(ibound == 2, 0.0, head)
    cprate = np.where(ibound == 2, np.nan, 0.002)
    v = dict(
        dem=dem_c, aqdepth=aqd, ksat_log=logk,
        ksat_l1_conf_log=np.full(N, np.nan),
        ksat_l2_conf_log=np.full(N, np.nan),
        cellarea=g.area[g.active], qbank=qbank, riv_slope=slope, z0=z0,
        min_dem=min_dem, kq3=np.full(N, 0.01), sy=sy, landmask=np.ones(N),
        ibound=ibound, ldd=basin, head0=np.stack([head, head - 0.1]),
        cpsurface=dem_c - 1.0, cprate=cprate,
        cpdistinct=np.full(N, 3.0), max_moist=max_moist)
    for k, a in v.items():
        assert a.shape[-1] == N, k
        if k not in ('ksat_l1_conf_log', 'ksat_l2_conf_log', 'cprate'):
            assert np.isfinite(a).all(), k
    return v


# -------------------------------------------------------------------- global
def global_text(year, steps, outdir, calendar=True):
    R = ROOT
    W = R + '/vicwur/isc'
    F = W + '/forcing'
    L = ['# VIC-WUR smoke test on ISC (tools/vicwur/isc/smoke_inputs.py)']
    a = L.append
    a('MODEL_STEPS_PER_DAY 1\nSNOW_STEPS_PER_DAY 1\nRUNOFF_STEPS_PER_DAY 1')
    a('STARTYEAR %d\nSTARTMONTH 1\nSTARTDAY 1' % year)
    a('ENDYEAR %d\nENDMONTH 12\nENDDAY 31' % year)
    a('CALENDAR PROLEPTIC_GREGORIAN\nFULL_ENERGY FALSE\nFROZEN_SOIL FALSE')
    a('DOMAIN %s/vic_params_isc/domain.nc' % R)
    for k, n in (('LAT', 'lat'), ('LON', 'lon'), ('MASK', 'mask'),
                 ('AREA', 'area'), ('FRAC', 'frac'), ('YDIM', 'lat'),
                 ('XDIM', 'lon')):
        a('DOMAIN_TYPE %s %s' % (k, n))
    # the VIC-WUR image driver has no TMAX/TMIN: daily mean tair
    # (tools/vicwur/isc/smoke_met.py, as R1's ref_domains.py)
    fp = W + '/smoke/forcing/met.'
    for t, v in (('AIR_TEMP', 'tair'), ('PREC', 'prcp'),
                 ('PRESSURE', 'sp'), ('SWDOWN', 'swdown'),
                 ('LWDOWN', 'lwdown'), ('VP', 'vp'), ('WIND', 'wind')):
        a('FORCE_TYPE %s %s %s' % (t, v, fp))
    a('WIND_H 10.0')
    # vegetated tiles without roots filled (smoke_params.py)
    a('PARAMETERS %s/smoke/params_wur_roots.nc' % W)
    a('SNOW_BAND TRUE\nBASEFLOW ARNO\nJULY_TAVG_SUPPLIED FALSE\n'
      'ORGANIC_FRACT FALSE')
    a('LAI_SRC FROM_VEGLIB\nFCAN_SRC FROM_VEGLIB\nALB_SRC FROM_VEGLIB')
    a('SHARE_LAYER_MOIST TRUE\nBCO2_SRC FROM_VEGPARAM\nGWM TRUE')
    a('ROUTING TRUE\nROUT_STEPS_PER_DAY %d' % steps)
    a('ROUTING_PARAMETERS %s/rout_n%d.nc' % (outdir, steps))
    a('DECOMPOSITION BASIN')
    a('DAMS TRUE\nDAMS_PARAMETERS %s/dams/dams.nc' % W)
    a('WATERUSE TRUE')
    a('LOCAL_WITHDRAWAL TRUE\nREMOTE_WITHDRAWAL FALSE\n'
      'COMPENSATION_WITHDRAWAL FALSE\nNONRENEWABLE_WITHDRAWAL TRUE\n'
      'NONRENEWABLE_RUNOFF FALSE')
    # per-step cap on the non-renewable (MODFLOW well) withdrawal, mm/day
    # per cell (GWM books no deficit, so availability = NONRENEWABLE_LIMIT
    # each step, wu_nonrenewable.c:79-84). Unlimited, a cold start pumped
    # 1.4e12 m3 on day 1. 10 mm/d: CGWB, Dynamic Ground Water Resources of
    # India 2022: gross extraction ~240 bcm/yr over ~3.3e6 km2 = ~73 mm/yr
    # on average (0.2 mm/d), 60 % stage; the most stressed states (Punjab,
    # Haryana, Rajasthan; stage 130-165 %) extract ~1 m/yr, ~2.5 mm/d, and
    # irrigation concentrates it in ~4 months (~7-10 mm/d peak)
    # VIC-WUR 3.0 reads NONRENEWABLE_LIMIT from the CONSTANTS (parameter)
    # file only (wu_get_global_param.c:158-174); in the global file it is an
    # unrecognized option and the run is unlimited
    a('CONSTANTS %s/constants_isc.txt' % outdir)
    for s in SECTORS:
        S = s.upper()
        a('PLUGIN_FORCE_TYPE %s_DEMAND %s_demand MONTH %s/%s_demand.'
          % (S, s, F, s))
        a('PLUGIN_FORCE_TYPE %s_GROUNDWATER %s_gw MONTH %s/%s_gw.'
          % (S, s, F, s))
        a('PLUGIN_FORCE_TYPE %s_CONSUMPTION %s_cons MONTH %s/%s_cons.'
          % (S, s, F, s))
    a('IRRIGATION TRUE\nIRRIGATION_PARAMETERS %s/params/irr.nc' % W)
    a('EFFICIENT_IRRIGATION TRUE\nPBARE 20')   # irrigation_offset is not in irr.nc (OFFSET_IRRIGATION off = 0)
    a('EFR TRUE')
    a('PLUGIN_FORCE_TYPE EFR_DISCHARGE efr_q MONTH %s/forcing/efr_q.'
      % outdir)
    a('PLUGIN_FORCE_TYPE EFR_BASEFLOW efr_b MONTH %s/forcing/efr_b.'
      % outdir)
    a('LANDUSE_FORCE TRUE')
    # irrigation season: monthly Cv (make_calendar.py); the tile is irrigated
    # only in the months it has Cv > 0 (irr_run.c: veg_fract > 0)
    if calendar:
        a('PLUGIN_FORCE_TYPE CV cv MONTH %s/cv.' % F.replace('/forcing', '/forcing_cal'))
    else:
        a('PLUGIN_FORCE_TYPE CV cv YEAR %s/cv.' % F)
    a('PLUGIN_FORCE_TYPE CO2 co2 YEAR %s/co2.' % F)
    a('WOFOST TRUE\nWOFOST_STEPS_PER_DAY 1')
    a('CROP_PARAMETERS %s/wofost/crop_params_VICWOFOST_isc.%d.nc' % (W, year))
    a('WOFOST_TEXT_PARAMETERS %s/wofost/wofost_params_VICWOFOST_isc.txt' % W)
    a('WOFOST_DIST_SEASON TRUE\nWOFOST_DIST_TSUM TRUE\n'
      'WOFOST_DIST_FERTILIZER TRUE\nWOFOST_DIST_MINERALIZATION TRUE')
    a('RESULT_DIR %s/out/\nLOG_DIR %s/log/' % (outdir, outdir))
    a('OUTFILE fluxes\nAGGFREQ NDAYS 1\nHISTFREQ NYEARS 1\n'
      'OUT_FORMAT NETCDF4')
    for v in OUTVARS:
        a('OUTVAR %s' % v)
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2005)
    ap.add_argument('--rout-steps', type=int, default=4)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out + '/forcing', exist_ok=True)
    g = ISC()
    cells, down, dist, basin, acc = network(g)
    write_routing(g, cells, down, dist, a.rout_steps,
                  '%s/rout_n%d.nc' % (a.out, a.rout_steps))
    write_efr(g, [a.year], cells, acc, a.out + '/forcing')
    v = static_values(g, cells, down, dist, basin, acc)
    np.savez(a.out + '/gw_static.npz', **v)
    log('groundwater static: %d basins, %d CHD cells'
        % (len(np.unique(basin)), int((v['ibound'] == 2).sum())))
    open('%s/constants_isc.txt' % a.out, 'w').write(
        '# W3 parameters of the ISC smoke runs (tools/vicwur/isc/smoke_inputs.py)\n'
        'NONRENEWABLE_LIMIT 10\n')
    open('%s/global_isc_%d.txt' % (a.out, a.year), 'w').write(
        global_text(a.year, a.rout_steps, a.out))
    log('wrote global_isc_%d.txt' % a.year)


if __name__ == '__main__':
    main()
