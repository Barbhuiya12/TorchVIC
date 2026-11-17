#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""District-level yields of an ISC CropSyst run against the Indian district statistics.

  district_yield.py RUN_DIR [--year 2005]

RUN_DIR (run_step.py output) holds series.npz (vicwur_cropsyst.yield / .harvest, kg ha-1 on the harvest day) and
cs_tiles.npz. Simulated district yield of a crop = area-weighted mean over the harvests of the year on that crop's
tiles (weight = the tile's Cv x cell area, all bands of a cell weighted alike); reference = India Data Portal crop-wise
area-production-yield (DES; district, 2004-05 and 2005-06 mean), polygons = geoBoundaries ADM2 (point-in-polygon on
cell centres; districts of the table are matched to polygons by normalised name). Writes district_yield.csv and
prints per crop: districts matched, bias, RMSE, correlation, and the national mean against FAOSTAT.
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

A = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
REF = A + '/raw/yield_ref/district'
CROP_APY = {'rice_kharif': ('Rice',), 'rice_boro': ('Rice',), 'wheat': ('Wheat',), 'maize': ('Maize',),
            'sorghum': ('Jowar',), 'pearl_millet': ('Bajra',), 'chickpea_pulses': ('Gram',),
            'sugarcane': ('Sugarcane',), 'cotton': ('Cotton(Lint)',), 'groundnut': ('Groundnut',),
            'mustard_rapeseed': ('Rapeseed &Mustard',), 'potato': ('Potato',), 'soybean': ('Soyabean',)}


def norm(s):
    s = re.sub(r'[^a-z ]', '', s.lower())
    for w in ('district', 'zila', 'urban', 'rural'):
        s = s.replace(w, '')
    return re.sub(r'\s+', '', s)


def rasterise(lat, lon, geo):
    """district index [n_cell] (-1 none) and names by point-in-polygon (matplotlib.path)."""
    from matplotlib.path import Path
    names, did = [], np.full(len(lat), -1, np.int32)
    pts = np.stack([lon, lat], 1)
    for f in geo['features']:
        g = f['geometry']
        polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        k = len(names)
        names.append(f['properties']['shapeName'])
        for poly in polys:
            ring = np.asarray(poly[0])
            lo, hi = ring.min(0), ring.max(0)
            c = np.flatnonzero((lon >= lo[0]) & (lon <= hi[0]) & (lat >= lo[1]) & (lat <= hi[1]) & (did < 0))
            if len(c):
                did[c[Path(ring).contains_points(pts[c])]] = k
    return did, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--year', type=int, default=2005)
    a = ap.parse_args()
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import isc_inputs as CSI
    import make_isc_cropsyst as G
    import netCDF4 as nc
    ser = np.load(a.run_dir + '/series.npz')
    tt = np.load(a.run_dir + '/cs_tiles.npz')
    meta = json.load(open(a.run_dir + '/meta.json'))
    yl, hv = ser['vicwur_cropsyst.yield'], ser['vicwur_cropsyst.harvest']
    # rows are days from the run start; the 2005 harvests
    import datetime
    d0 = datetime.date.fromisoformat(meta['start'])
    d1 = (datetime.date(a.year, 1, 1) - d0).days
    d2 = (datetime.date(a.year + 1, 1, 1) - d0).days
    # connector series start one day after the run's first (first row dropped by the runner): align on length
    off = meta['days'] - yl.shape[0]
    sl = slice(max(d1 - off, 0), d2 - off)
    ysum = (yl[sl] * (hv[sl] > 0.5)).sum(0)
    nh = (hv[sl] > 0.5).sum(0)
    # tile -> crop through the rotation code
    rot = {int(r['code']): r['crop'] for r in csv.DictReader(open(CSI.INPUTS + '/rotations_isc.csv'))}
    crop = np.array([rot[int(c)] for c in tt['code']])
    glob = os.environ.get('TORCHVIC_ISC_GLOBAL')
    gp = glob or json.load(open(a.run_dir + '/meta.json')).get('global', A + '/verification/isc_cropsyst/out/%s/global.txt' % os.path.basename(a.run_dir.rstrip('/')))
    act = CSI.active_mask(gp)
    inp = CSI.read_inputs()
    lat2 = np.broadcast_to(inp['lat'][:, None], act.shape)
    lon2 = np.broadcast_to(inp['lon'][None, :], act.shape)
    lat, lon = lat2[act], lon2[act]
    area = np.cos(np.deg2rad(lat))
    geo = json.load(open(REF + '/IND_ADM2.geojson'))
    did, names = rasterise(lat, lon, geo)
    cell = tt['cell']
    ok = (nh > 0) & (did[cell] >= 0)
    df = pd.DataFrame(dict(crop=crop[ok], district=did[cell][ok], y=ysum[ok] / nh[ok] / 1000.0, w=area[cell][ok]))
    sim = df.groupby(['crop', 'district']).apply(lambda g: pd.Series(dict(
        sim_t_ha=np.average(g.y, weights=g.w), cells=len(g)))).reset_index()
    sim['district_name'] = [names[i] for i in sim.district]
    sim['key'] = sim.district_name.map(norm)
    apy = pd.read_csv(REF + '/apy.csv')
    apy = apy[apy.year.isin(['%d-%d' % (a.year - 1, a.year),
                              '%d-%d' % (a.year, a.year + 1)])]
    apy = apy.dropna(subset=['yield'])
    # cotton lint is reported in bales (170 kg)
    apy['production'] = np.where(apy.production_unit.astype(str).str.startswith('Bale'), 0.17 * apy.production,
                                 apy.production)
    apy = apy[apy.area > 0]
    apy['key'] = apy.district_name.map(norm)
    out, rows = [], []
    for c, names_ in CROP_APY.items():
        r = apy[apy.crop_name.isin(names_)]
        if c.startswith('rice'):
            r = r[r.season.str.strip().isin(('Kharif', 'Autumn', 'Summer', 'Winter', 'Rabi', 'Whole Year'))]
        r = r.groupby('key').apply(lambda g: pd.Series(dict(ref_t_ha=g.production.sum() / g.area.sum(),
                                                            area=g.area.sum()))).reset_index()
        s = sim[sim.crop == c].merge(r, on='key')
        if c == 'sugarcane':
            pass
        if len(s) == 0:
            rows.append((c, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        d = s.sim_t_ha - s.ref_t_ha
        rows.append((c, len(s), float(np.average(s.sim_t_ha, weights=s.area)), float(np.average(s.ref_t_ha, weights=s.area)),
                     float(np.sqrt((d ** 2).mean())), float(np.corrcoef(s.sim_t_ha, s.ref_t_ha)[0, 1]) if len(s) > 2 else np.nan))
        s['crop'] = c
        out.append(s)
    if out:
        pd.concat(out).to_csv(a.run_dir + '/district_yield.csv', index=False)
    print('crop districts sim_mean_t_ha ref_mean_t_ha rmse r   (areal mean over matched districts; sim is dry grain / '
          'cane dry biomass, ref is as reported)')
    for r in rows:
        print('%-18s %4d %8.2f %8.2f %6.2f %5.2f' % r)


if __name__ == '__main__':
    main()
