#!/usr/bin/env python
"""VIC-WUR dam inputs for the ISC domain from Global Dam Watch v1.0 (open
replacement for GRanD) plus the Indian dams of the CWC/NDSA National Register
of Specified Dams (NRSD, April 2025) that GDW lacks.

Output <data root>/vicwur/isc/dams/dams.nc (format of torchvic.vicwur.params.
read_dams) and dams_table.csv (every dam with area, purpose and flags).

Method, following VIC_support-VIC-WUR Scripts/Generate/dams (R):
  inflow fraction  catchment area of the dam / upstream area of the model
                   cell (capped at 1); dams of one cell are merged (capacity
                   summed, year minimum, fractions summed); a cell with
                   fraction >= 1 is a global dam (type 1), else local (0).
  service area     only global irrigation dams: walk downstream cell by cell,
                   fraction = Q(dam cell) / Q(cell), stop below 0.25 or at
                   the outlet (serviceDam.R).  Q is proxied by upstream
                   area (no calibrated discharge at the ISC grid); where a
                   cell is served by several dams the fractions are scaled
                   to sum to 1.
  catchment        GDW CATCH_SKM; unknown ones from a power law fitted to
                   the dams that have both capacity and catchment.
  NRSD match      within 0.12 deg of a GDW dam and (within 0.02 deg, or name
                  similarity >= 0.85, or <= 0.06 deg and similarity >= 0.7, or
                  capacity ratio 0.7-1.4); matched dams only fill GDW gaps
                  (year, purpose, irrigation, capacity), the rest are added.
  cell choice      the containing cell, or a neighbour (5 x 5) whose upstream
                   area best matches the catchment (the GDW point is on the
                   river but the 1/16 deg D8 river can be one cell away).
Dams with capacity below MIN_CAP_MCM are not used (reader cost and
relevance; GRanD itself starts at 100 MCM).
"""
import difflib
import os
import re
import subprocess
import sys
import numpy as np
import netCDF4 as nc
import pandas as pd
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa
import parse_nrsd               # noqa

MIN_CAP_MCM = 10.0
UNKNOWN_YEAR = 1900
SERVICE_MIN = 0.25
NRSD_PDF = RAW + '/nrld_cwc/NRSD_2025.pdf'
GDW = WORK + '/gdw/GDW_v1_0_shp/GDW_barriers_v1_0.shp'
OUTD = OUT + '/dams'
ESRI = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1),
        32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
GENERIC = {'dam', 'reservoir', 'project', 'barrage', 'weir', 'irrigation',
           'tank', 'lake', 'scheme', 'sagar', 'the', 'of', 'and', 'pond',
           'anicut', 'balancing', 'major', 'medium', 'minor', 'power',
           'house', 'hydro', 'hydel', 'main', 'ii', 'i', 'iii', 'stage'}


def tokens(s):
    t = re.sub(r'[^a-z ]', ' ', str(s).lower()).split()
    return {w for w in t if w not in GENERIC and len(w) > 2}


def name_sim(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    best = 0.0
    for x in ta:
        for y in tb:
            best = max(best, difflib.SequenceMatcher(None, x, y).ratio())
    return best


def network(g):
    fd = nc.Dataset(FDIR)
    fdir = np.array(fd['fdir'][:])
    up_given = np.array(fd['upstream_area'][:], float)
    fd.close()
    idx = -np.ones((g.ny, g.nx), int)
    cells = np.argwhere(g.active)
    idx[g.active] = np.arange(len(cells))
    down = -np.ones(len(cells), int)
    r, c = cells[:, 0], cells[:, 1]
    for code, (dr, dc) in ESRI.items():
        m = fdir[r, c] == code
        rr, cc = r[m] + dr, c[m] + dc
        ok = (rr >= 0) & (rr < g.ny) & (cc >= 0) & (cc < g.nx)
        t = -np.ones(m.sum(), int)
        t[ok] = idx[rr[ok], cc[ok]]
        down[np.nonzero(m)[0]] = t
    # upstream area by topological accumulation (m2)
    area = g.area[g.active]
    indeg = np.bincount(down[down >= 0], minlength=len(cells))
    acc = area.copy()
    stack = list(np.nonzero(indeg == 0)[0])
    while stack:
        i = stack.pop()
        d = down[i]
        if d >= 0:
            acc[d] += acc[i]
            indeg[d] -= 1
            if indeg[d] == 0:
                stack.append(d)
    return idx, cells, down, acc, up_given[g.active]


def main():
    g = ISC()
    os.makedirs(OUTD, exist_ok=True)
    rows = []
    idx, cells, down, acc, up_given = network(g)
    A_km2 = acc / 1e6
    rows.append('upstream area: accumulated vs isc_fdir upstream_area, median ratio %.3f (given units as stored)'
                % np.median(A_km2[A_km2 > 100] / np.maximum(up_given[A_km2 > 100], 1e-9)))
    # ---- GDW dams in the domain
    gd = gpd.read_file(GDW)
    x, y = gd.geometry.x.values, gd.geometry.y.values
    r0 = np.round((y - g.lat[0]) / (g.lat[1] - g.lat[0])).astype(int)
    c0 = np.round((x - g.lon[0]) / (g.lon[1] - g.lon[0])).astype(int)
    ok = (r0 >= 0) & (r0 < g.ny) & (c0 >= 0) & (c0 < g.nx)
    ok[ok] = g.active[r0[ok], c0[ok]]
    gd = gd[ok].copy()
    gd['lon'], gd['lat'] = x[ok], y[ok]
    rows.append('GDW barriers in the ISC active domain: %d (%s)' % (
        len(gd), gd.COUNTRY.value_counts().head(5).to_dict()))
    irr_g = gd[['USE_IRRI']].notna().any(axis=1) | (gd.MAIN_USE == 'Irrigation')
    D = pd.DataFrame({
        'source': 'GDW', 'src_id': gd.GDW_ID.astype(str).values,
        'name': gd.DAM_NAME.fillna(gd.RES_NAME).fillna('').values,
        'country': gd.COUNTRY.values, 'lat': gd.lat.values, 'lon': gd.lon.values,
        'cap_mcm': gd.CAP_MCM.where(gd.CAP_MCM > 0).values,
        'area_km2': gd.AREA_SKM.where(gd.AREA_SKM > 0).values,
        'year': gd.YEAR_DAM.where(gd.YEAR_DAM > 0).values,
        'purpose': gd.MAIN_USE.fillna('').values,
        'irrigation': irr_g.values,
        'catch_km2': gd.CATCH_SKM.where(gd.CATCH_SKM > 0).values,
        'height_m': gd.DAM_HGT_M.where(gd.DAM_HGT_M > 0).values})
    D['flags'] = ''
    # ---- NRSD
    csv = OUTD + '/nrsd_parsed.csv'
    if not os.path.exists(csv):          # pdftotext is login-node only
        parse_nrsd.main(NRSD_PDF, csv)
    ns = pd.read_csv(csv)
    rows.append('NRSD 2025 dams parsed: %d' % len(ns))
    gin = D[D.country == 'India']
    gx = np.radians(gin.lon.values)
    gy = np.radians(gin.lat.values)
    matched = np.zeros(len(ns), bool)
    add = []
    nmatch_name = nmatch_loc = 0
    for k, r in ns.iterrows():
        dd = np.hypot((gin.lon.values - r.lon) * np.cos(np.radians(r.lat)),
                      gin.lat.values - r.lat)
        j = np.argsort(dd)[:5]
        best = None
        for jj in j:
            if dd[jj] > 0.12:
                break
            sim = name_sim(r['name'], gin.name.values[jj]) if isinstance(r['name'], str) else 0
            gc_ = gin.cap_mcm.values[jj]
            rc = gc_ / r.gross_mcm if (gc_ > 0 and r.gross_mcm > 0) else 0.0
            if dd[jj] <= 0.02 or sim >= 0.85 or (dd[jj] <= 0.06 and sim >= 0.7) \
                    or (0.7 <= rc <= 1.4):
                best = (dd[jj], jj, sim)
                break
        if best is not None:
            matched[k] = True
            gi = gin.index[best[1]]
            if best[2] >= 0.85:
                nmatch_name += 1
            else:
                nmatch_loc += 1
            # fill the gaps of the GDW record from NRSD
            if pd.isna(D.at[gi, 'year']) and pd.notna(r.year):
                D.at[gi, 'year'] = r.year
                D.at[gi, 'flags'] += 'year_nrsd;'
            if pd.isna(D.at[gi, 'cap_mcm']) and pd.notna(r.gross_mcm):
                D.at[gi, 'cap_mcm'] = r.gross_mcm
                D.at[gi, 'flags'] += 'cap_nrsd;'
            if not D.at[gi, 'purpose'] and isinstance(r.purpose, str) and r.purpose:
                D.at[gi, 'purpose'] = 'NRSD: ' + r.purpose
                D.at[gi, 'flags'] += 'purpose_nrsd;'
            if isinstance(r.purpose, str) and re.search('Irrig|Multi', r.purpose):
                if not D.at[gi, 'irrigation']:
                    D.at[gi, 'flags'] += 'irr_nrsd;'
                D.at[gi, 'irrigation'] = True
        else:
            add.append(k)
    rows.append('NRSD matched to GDW: %d (by name+location %d, by location %d); not in GDW: %d'
                % (matched.sum(), nmatch_name, nmatch_loc, len(add)))
    A = ns.loc[add]
    inbox = ((A.lat > g.lat[0]) & (A.lat < g.lat[-1]) & (A.lon > g.lon[0]) &
             (A.lon < g.lon[-1]))
    A = A[inbox]
    rr = np.round((A.lat.values - g.lat[0]) / (g.lat[1] - g.lat[0])).astype(int)
    cc = np.round((A.lon.values - g.lon[0]) / (g.lon[1] - g.lon[0])).astype(int)
    okk = g.active[np.clip(rr, 0, g.ny - 1), np.clip(cc, 0, g.nx - 1)]
    A = A[okk]
    N = pd.DataFrame({
        'source': 'NRSD', 'src_id': A.pic.values,
        'name': [re.sub(r'\b[A-Z]{2}\d{2}[A-Z]{2}\d{4}\b', '', str(s)).strip() for s in A['name']],
        'country': 'India', 'lat': A.lat.values, 'lon': A.lon.values,
        'cap_mcm': A.gross_mcm.where(A.gross_mcm > 0).values,
        'area_km2': np.nan, 'year': A.year.values,
        'purpose': ['NRSD: ' + str(p) if isinstance(p, str) else '' for p in A.purpose],
        'irrigation': [bool(re.search('Irrig|Multi', str(p))) for p in A.purpose],
        'catch_km2': np.nan, 'height_m': A.height_m.values, 'flags': 'nrsd_added;'})
    rows.append('NRSD dams added (inside active ISC cells): %d' % len(N))
    D = pd.concat([D, N], ignore_index=True)
    # ---- catchment power law from dams with both capacity and catchment
    both = D[(D.cap_mcm > 0) & (D.catch_km2 > 0)]
    b, a = np.polyfit(np.log(both.cap_mcm), np.log(both.catch_km2), 1)
    res = np.log(both.catch_km2) - (a + b * np.log(both.cap_mcm))
    rows.append('catchment ~ cap^b: log(catch km2) = %.3f + %.3f log(cap MCM); n %d, resid std %.2f (ln)'
                % (a, b, len(both), res.std()))
    est = D.catch_km2.isna() & (D.cap_mcm > 0)
    D.loc[est, 'catch_km2'] = np.exp(a + b * np.log(D.loc[est, 'cap_mcm']))
    D.loc[est, 'flags'] += 'catch_est;'
    # ---- floor, cell choice
    D = D[D.cap_mcm >= MIN_CAP_MCM].reset_index(drop=True)
    rows.append('dams with capacity >= %g MCM: %d' % (MIN_CAP_MCM, len(D)))
    r0 = np.round((D.lat.values - g.lat[0]) / (g.lat[1] - g.lat[0])).astype(int)
    c0 = np.round((D.lon.values - g.lon[0]) / (g.lon[1] - g.lon[0])).astype(int)
    cell = np.zeros(len(D), int)
    moved = 0
    for i in range(len(D)):
        best, bs = None, 1e9
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr_, cc_ = r0[i] + dr, c0[i] + dc
                if not (0 <= rr_ < g.ny and 0 <= cc_ < g.nx) or not g.active[rr_, cc_]:
                    continue
                k = idx[rr_, cc_]
                sc = abs(np.log(A_km2[k] / D.catch_km2[i])) + 0.15 * np.hypot(dr, dc)
                if sc < bs:
                    best, bs = k, sc
        cell[i] = best
        if best != idx[r0[i], c0[i]]:
            moved += 1
    D['cell'] = cell
    D['frac_raw'] = D.catch_km2.values / A_km2[cell]
    rows.append('dams moved to a neighbouring cell for the best area match: %d' % moved)
    # ---- merge per cell
    recs = []
    for k, grp in D.groupby('cell'):
        fr = min(1.0, grp.frac_raw.sum())
        yrs = grp.year.dropna()
        recs.append(dict(
            cell=k, row=cells[k][0], col=cells[k][1], n_dams=len(grp),
            capacity_mcm=grp.cap_mcm.sum(), area_km2=grp.area_km2.sum(min_count=1),
            year=int(yrs.min()) if len(yrs) else UNKNOWN_YEAR,
            year_known=len(yrs) > 0, inflow_fraction=fr,
            is_global=int(fr >= 1.0), irrigation=bool(grp.irrigation.any()),
            purpose='; '.join(sorted({p for p in grp.purpose if p})),
            names='; '.join(n for n in grp.name if n)[:200],
            sources='+'.join(sorted(set(grp.source))),
            ids=','.join(grp.src_id.astype(str)), lat=grp.lat.mean(), lon=grp.lon.mean()))
    M = pd.DataFrame(recs).sort_values(['row', 'col']).reset_index(drop=True)
    M['id'] = np.arange(1, len(M) + 1)
    nd = len(M)
    rows.append('merged dam cells (dam classes): %d  global %d local %d  irrigation %d  year unknown %d'
                % (nd, M.is_global.sum(), (1 - M.is_global).sum(), M.irrigation.sum(), (~M.year_known).sum()))
    # ---- service areas
    svc = -np.ones((nd, g.ny, g.nx), 'i4')
    sfr = -np.ones((nd, g.ny, g.nx))
    nserv = np.zeros(nd, 'i4')
    for j, m in M.iterrows():
        if not (m.is_global and m.irrigation):
            continue
        cur = int(m.cell)
        while True:
            fr = acc[int(m.cell)] / acc[cur]
            if fr < SERVICE_MIN:
                break
            rr_, cc_ = cells[cur]
            svc[j, rr_, cc_] = m.id
            sfr[j, rr_, cc_] = fr
            nserv[j] += 1
            if down[cur] < 0 or down[cur] == cur:
                break
            cur = down[cur]
    tot = np.where(svc > 0, sfr, 0).sum(0)
    over = tot > 1.0
    rows.append('cells served by dams: %d, with summed fraction > 1 (scaled to 1): %d'
                % (int((tot > 0).sum()), int(over.sum())))
    scale = np.where(over, 1.0 / np.where(over, tot, 1), 1.0)
    sfr = np.where(svc > 0, sfr * scale[None], -1.0)
    M['n_service'] = nserv
    # ---- write
    with nc.Dataset(OUTD + '/dams.nc', 'w') as ds:
        coords(ds, g)
        ds.createDimension('dam_class', nd)

        def var(n_, t, d, a, u=None, fill=None, **at):
            v = ds.createVariable(n_, t, d, fill_value=fill, zlib=True, complevel=2)
            if u:
                v.units = u
            for k_, a_ in at.items():
                v.setncattr(k_, a_)
            v[:] = a
        idmap = -np.ones((g.ny, g.nx), 'i4')
        idmap[M.row.values, M.col.values] = M.id.values
        var('dam_class', 'i4', ('dam_class',), np.arange(nd))
        var('id_map', 'i4', ('lat', 'lon'), idmap, 'ID', -1)
        var('id', 'i4', ('dam_class',), M.id.values, 'ID', -1)
        var('type', 'i4', ('dam_class',), M.is_global.values, '1 global, 0 local', -1)
        var('year', 'i4', ('dam_class',), M.year.values, 'year AD', -1,
            comment='%d where unknown (%d dams)' % (UNKNOWN_YEAR, int((~M.year_known).sum())))
        var('capacity', 'f8', ('dam_class',), M.capacity_mcm.values, 'hm3')
        var('inflow_fraction', 'f8', ('dam_class',), M.inflow_fraction.values, 'fraction')
        var('Nservice', 'i4', ('dam_class',), nserv)
        var('service', 'i4', ('dam_class', 'lat', 'lon'), svc, fill=-1)
        var('service_fraction', 'f8', ('dam_class', 'lat', 'lon'), sfr, fill=-1.0)
        ds.title = 'VIC-WUR dam parameters, ISC: GDW v1.0 + NRSD 2025'
    M.to_csv(OUTD + '/dams_table.csv', index=False)
    d_ = M.capacity_mcm
    rows.append('capacity MCM: min %.1f median %.1f max %.1f total %.0f' % (
        d_.min(), d_.median(), d_.max(), d_.sum()))
    rows.append('inflow fraction: min %.3f median %.3f' % (M.inflow_fraction.min(), M.inflow_fraction.median()))
    rows.append('service cells per irrigation global dam: median %d max %d' % (
        M[M.n_service > 0].n_service.median() if (M.n_service > 0).any() else 0, M.n_service.max()))
    for r in rows:
        log('CHECK ' + r)
    log('dams done')


if __name__ == '__main__':
    main()
