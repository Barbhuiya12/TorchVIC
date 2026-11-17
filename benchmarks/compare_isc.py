#!/usr/bin/env python3
"""Compare two ISC runs' 2005 history: annual totals per cell and domain mean.

    compare_isc.py REF_RUN TEST_RUN [--label NAME] [--year 2005]
        [--csv OUT.csv] [--map OUT.png]

REF_RUN and TEST_RUN are run directories (<data root>/benchmark_isc/runs/NAME)
holding out/fluxes.YYYY-01-01.nc. For each variable: the domain mean
(weighted by area x frac) of both runs and its relative difference, and the
per-cell differences: median and 99th percentile of |relative difference|
over cells whose reference value is above a floor, the largest absolute
difference, and the number of cells off by more than 1%.
Fluxes (OUT_PREC, OUT_EVAP, OUT_RUNOFF, OUT_BASEFLOW) are annual sums, mm;
states (OUT_SWE, OUT_SOIL_MOIST summed over layers) annual means, mm.
"""
import argparse
import csv
import os
import sys

import numpy as np
import xarray as xr

DOMAIN = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'vic_params_isc/domain.nc')
VARS = (('OUT_PREC', 'sum', 1.0), ('OUT_EVAP', 'sum', 1.0),
        ('OUT_RUNOFF', 'sum', 1.0), ('OUT_BASEFLOW', 'sum', 1.0),
        ('OUT_SWE', 'mean', 1.0), ('OUT_SOIL_MOIST', 'mean', 1.0))


def annual(run, year):
    path = os.path.join(run, 'out', 'fluxes.%d-01-01.nc' % year)
    ds = xr.open_dataset(path, mask_and_scale=True)
    out = {}
    for v, how, floor in VARS:
        x = ds[v].astype('f8')
        if 'nlayer' in x.dims:
            x = x.sum('nlayer', skipna=False)
        out[v] = (x.sum('time', skipna=False) if how == 'sum'
                  else x.mean('time', skipna=False)).values
    out['ndays'] = ds.sizes['time']
    out['swe_max'] = ds['OUT_SWE'].max('time').values
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ref')
    ap.add_argument('test')
    ap.add_argument('--label', default=None)
    ap.add_argument('--year', type=int, default=2005)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--map', default=None)
    ap.add_argument('--domain', default=DOMAIN)
    ap.add_argument('--added', default=None,
                    help='CSV (gridcell,lat,lon) of cells to report as their own group')
    a = ap.parse_args(argv)
    dom = xr.open_dataset(a.domain)
    mask = dom['mask'].values > 0
    w = (dom['area'] * dom['frac']).values[mask]
    R, T = annual(a.ref, a.year), annual(a.test, a.year)
    assert R['ndays'] == T['ndays'], (R['ndays'], T['ndays'])
    label = a.label or '%s vs %s' % (os.path.basename(a.test.rstrip('/')),
                                     os.path.basename(a.ref.rstrip('/')))
    rows = []
    # cells that hold snow at some time in either run, in this or the
    # previous year (a pack carried into the year counts)
    snow = (np.nan_to_num(R['swe_max']) > 0) | (np.nan_to_num(T['swe_max']) > 0)
    try:
        R0, T0 = annual(a.ref, a.year - 1), annual(a.test, a.year - 1)
        snow |= (np.nan_to_num(R0['swe_max']) > 0) | (np.nan_to_num(T0['swe_max']) > 0)
    except (OSError, FileNotFoundError, KeyError):
        pass
    snow = snow[mask]
    groups = [('all', np.ones(mask.sum(), bool)),
              ('snow-free', ~snow), ('snow', snow)]
    if a.added:
        import pandas as pd
        ad = pd.read_csv(a.added)
        lat, lon = dom['lat'].values, dom['lon'].values
        if lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        key = set(zip(np.round(ad['lat'].values, 4), np.round(ad['lon'].values, 4)))
        hit = np.array([(round(float(y), 4), round(float(x), 4)) in key
                        for y, x in zip(lat[mask], lon[mask])])
        assert hit.sum() == len(ad), (hit.sum(), len(ad))
        groups.append(('added coastal', hit))
    for cells, sel in groups:
        rows += table(R, T, mask, w, sel, cells, label, a.year)
    if a.csv:
        new = not os.path.exists(a.csv)
        with open(a.csv, 'a', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new:
                wr.writeheader()
            wr.writerows(rows)
    if a.map:
        draw(R, T, mask, dom, label, a)
    return 0


def table(R, T, mask, w0, sel, cells, label, year):
    rows = []
    w = w0[sel]
    print('%s, %d, %s cells (%d)' % (label, year, cells, sel.sum()))
    print('%-15s %12s %12s %10s %10s %10s %12s %8s'
          % ('variable', 'ref mean', 'test mean', 'rel diff%', 'med|rd|%',
             'p99|rd|%', 'max|d| mm', 'n>1%'))
    for v, how, _ in VARS:
        r, t = R[v][mask][sel], T[v][mask][sel]
        ok = np.isfinite(r) & np.isfinite(t)
        nbad = int((~ok).sum())
        rm = float((r[ok] * w[ok]).sum() / w[ok].sum())
        tm = float((t[ok] * w[ok]).sum() / w[ok].sum())
        rd_dom = 100.0 * (tm - rm) / rm if rm else float('nan')
        d = t[ok] - r[ok]
        floor = max(1.0, 1e-3 * np.abs(rm))      # mm: ignore near-zero cells
        big = np.abs(r[ok]) > floor
        rd = 100.0 * np.abs(d[big]) / np.abs(r[ok][big])
        if not d.size:
            continue
        row = dict(label=label, year=year, cells=cells, variable=v,
                   aggregation=how,
                   ref_domain_mean_mm=rm, test_domain_mean_mm=tm,
                   domain_rel_diff_pct=rd_dom,
                   cells_compared=int(big.sum()), floor_mm=floor,
                   median_abs_rel_diff_pct=float(np.median(rd)) if rd.size else np.nan,
                   p99_abs_rel_diff_pct=float(np.percentile(rd, 99)) if rd.size else np.nan,
                   max_abs_diff_mm=float(np.abs(d).max()),
                   cells_over_1pct=int((rd > 1.0).sum()),
                   cells_nonfinite=nbad)
        rows.append(row)
        print('%-15s %12.4f %12.4f %10.4f %10.4f %10.4f %12.3f %8d%s'
              % (v, rm, tm, rd_dom, row['median_abs_rel_diff_pct'],
                 row['p99_abs_rel_diff_pct'], row['max_abs_diff_mm'],
                 row['cells_over_1pct'],
                 '  (%d non-finite)' % nbad if nbad else ''))
    print()
    return rows


def draw(R, T, mask, dom, label, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lat, lon = dom['lat'].values, dom['lon'].values
    names = ('OUT_EVAP', 'OUT_RUNOFF', 'OUT_BASEFLOW', 'OUT_SWE')
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
    for ax, v in zip(axs, names):
        d = np.where(mask, T[v] - R[v], np.nan)
        lim = np.nanpercentile(np.abs(d), 99.5) or 1e-9
        im = ax.pcolormesh(lon, lat, d, cmap='RdBu_r', vmin=-lim, vmax=lim,
                           shading='auto')
        ax.set_title('%s %d: test - ref, mm\nmax |d| %.3g mm'
                     % (v.replace('OUT_', ''), a.year,
                        np.nanmax(np.abs(d))), fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('%s (annual totals; SWE: annual mean)' % label, fontsize=10)
    fig.savefig(a.map, dpi=110)
    print('wrote %s' % a.map)


if __name__ == '__main__':
    sys.exit(main())
