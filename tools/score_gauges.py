#!/usr/bin/env python3
"""Score routed VIC discharge against observed discharge at the ISC gauges.

  score_gauges.py --sim c2_gauges.csv --obs-dir <data root>/discharge/final_basins/discharge \
                  --index <data root>/discharge/final_basins/basin_index.csv \
                  --out-dir OUT [--start 2004-01-01 --end 2004-12-31]

`--sim` is the gauge discharge (m3 s-1) written by tools/route_isc.py (CSV: date, then one
column per clean_id) from a VIC + routing run, e.g. the ISC combination 2 (configs/vic_routing.yaml).
Observed series are `<obs-dir>/<clean_id>.csv` (Date, Q in m3 s-1, daily, gaps absent).
Scores come from torchvic.metrics (KGE 2009, NSE, PBIAS in % with + meaning the model is too high,
plus r, alpha and beta from the KGE decomposition) over the days of the window with an observation.
The 4 canal gauges (regulated diversions, not natural flow) are skipped, as are gauges with fewer
than --min-days observed days in the window.

Writes OUT/gauge_scores.csv, OUT/summary.txt and OUT/gauge_map.png.
A sensitivity pass with the first --skip-days days dropped (the routed river starts empty) is
reported in the summary and as *_skip columns.
"""
import argparse
import csv
import os
import sys
from datetime import date, timedelta

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from torchvic import metrics  # noqa: E402

CANALS = {478, 480, 482, 484}


def read_sim(path, sim_start):
    """(dates, ids, Q[time, gauge]); the dates are rebuilt from `sim_start` if the file's first date differs
    by the old route_isc calendar offset (up to 3 d), never silently shifted otherwise."""
    with open(path) as fh:
        r = csv.reader(fh)
        head = next(r)
        ids = [int(x) for x in head[1:]]
        rows = list(r)
    dates = [date.fromisoformat(x[0]) for x in rows]
    q = np.array([[float(v) for v in x[1:]] for x in rows])
    if dates[0] != sim_start:
        off = (sim_start - dates[0]).days
        if abs(off) > 3:
            raise SystemExit('sim starts %s, expected %s (--sim-start)' % (dates[0], sim_start))
        print('WARNING: sim dates start %s, re-dated by %+d d to %s' % (dates[0], off, sim_start))
        dates = [d + timedelta(days=off) for d in dates]
    return dates, ids, q


def read_obs(path):
    out = {}
    with open(path) as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            try:
                out[date.fromisoformat(row[0])] = float(row[1])
            except (ValueError, IndexError):
                pass
    return out


def one(sim, obs):
    s = torch.tensor(sim, dtype=torch.float64)[:, None]
    o = torch.tensor(obs, dtype=torch.float64)[:, None]
    k, comp = metrics.kge(s, o, components=True)
    return dict(kge=float(k), nse=float(metrics.nse(s, o)), pbias=float(metrics.pbias(s, o)),
                r=float(metrics.pearson_r(s, o)), alpha=float(comp['alpha']), beta=float(comp['beta']))


def dist(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if not len(x):
        return 'n=0'
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return 'n=%d  p10 %.2f  p25 %.2f  median %.2f  p75 %.2f  p90 %.2f' % ((len(x),) + tuple(q))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sim', required=True)
    ap.add_argument('--obs-dir', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--start', default='2004-01-01', help='first day of the scoring window')
    ap.add_argument('--sim-start', default='2004-01-01', help='date of the first row of --sim')
    ap.add_argument('--end', default='2004-12-31')
    ap.add_argument('--min-days', type=int, default=180)
    ap.add_argument('--skip-days', type=int, default=90)
    ap.add_argument('--all-gauges', action='store_true',
                    help='also score gauges not flagged usable == 1.0 (default: the reviewed set only)')
    ap.add_argument('--domain', help='domain.nc, drawn as the map background')
    ap.add_argument('--label', default='VIC + routing, uncalibrated')
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    os.makedirs(a.out_dir, exist_ok=True)

    dates, ids, q = read_sim(a.sim, date.fromisoformat(a.sim_start))
    col = {g: i for i, g in enumerate(ids)}
    di = {d: i for i, d in enumerate(dates)}
    win = [d for d in dates if start <= d <= end]
    skip_from = start + timedelta(days=a.skip_days)

    with open(a.index) as fh:
        index = {int(r['clean_id']): r for r in csv.DictReader(fh)}

    rows, notes = [], {'canal': 0, 'unreviewed': 0, 'not_routed': 0, 'no_obs_file': 0, 'short': 0}
    for g, r in sorted(index.items()):
        if g in CANALS:
            notes['canal'] += 1
            continue
        if not a.all_gauges and r.get('usable', '') != '1.0':
            notes['unreviewed'] += 1
            continue
        if g not in col:
            notes['not_routed'] += 1
            continue
        f = os.path.join(a.obs_dir, '%d.csv' % g)
        if not os.path.exists(f):
            notes['no_obs_file'] += 1
            continue
        ob = read_obs(f)
        days = [d for d in win if d in ob and np.isfinite(ob[d])]
        if len(days) < a.min_days:
            notes['short'] += 1
            continue
        sim = np.array([q[di[d], col[g]] for d in days])
        obs = np.array([ob[d] for d in days])
        row = dict(clean_id=g, name=r['name'], lat=float(r['lat']), lon=float(r['lng']),
                   area_km2=r.get('poly_area_km2') or r['area'], usable=r.get('usable', ''),
                   wb_class=r.get('wb_class', ''), n_days=len(days),
                   obs_mean=obs.mean(), sim_mean=sim.mean())
        row.update(one(sim, obs))
        keep = [i for i, d in enumerate(days) if d >= skip_from]
        if len(keep) >= a.min_days // 2:
            sk = one(sim[keep], obs[keep])
            row.update({k + '_skip': v for k, v in sk.items()})
        rows.append(row)

    fields = ['clean_id', 'name', 'lat', 'lon', 'area_km2', 'usable', 'wb_class', 'n_days',
              'obs_mean', 'sim_mean', 'kge', 'nse', 'pbias', 'r', 'alpha', 'beta',
              'kge_skip', 'nse_skip', 'pbias_skip', 'r_skip', 'alpha_skip', 'beta_skip']
    with open(os.path.join(a.out_dir, 'gauge_scores.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('%.5g' % v if isinstance(v, float) else v) for k, v in r.items()})

    def col_(k, sub=None):
        return [r[k] for r in rows if (sub is None or sub(r)) and k in r]

    L = ['Gauge scores: %s, %s to %s' % (a.label, a.start, a.end), '',
         'gauges in the index %d; skipped: %d canal, %d not flagged usable == 1.0, %d not routed, '
         '%d without observation file, %d with < %d observed days in the window; scored %d'
         % (len(index), notes['canal'], notes['unreviewed'], notes['not_routed'], notes['no_obs_file'], notes['short'],
            a.min_days, len(rows)), '']
    subsets = [('usable == 1.0 (the reviewed gauges)', lambda r: r['usable'] == '1.0')]
    if a.all_gauges:
        subsets.insert(0, ('all scored gauges', None))
    for title, sub in subsets:
        n = len([r for r in rows if sub is None or sub(r)])
        L.append('%s (%d gauges)' % (title, n))
        for k in ('kge', 'nse', 'pbias'):
            L.append('  %-6s %s' % (k, dist(col_(k, sub))))
            if rows and k + '_skip' in rows[0]:
                L.append('  %-6s %s   [first %d days dropped]' % (k, dist(col_(k + '_skip', sub)), a.skip_days))
        kk = np.array(col_('kge', sub))
        nn = np.array(col_('nse', sub))
        pb = np.array(col_('pbias', sub))
        L.append('  KGE > 0.5: %d, KGE > 0: %d, KGE > -0.41 (beats the mean flow): %d of %d'
                 % ((kk > 0.5).sum(), (kk > 0).sum(), (kk > -0.41).sum(), len(kk)))
        L.append('  NSE > 0: %d;  |PBIAS| < 25%%: %d;  model too high (PBIAS > 0): %d, too low: %d'
                 % ((nn > 0).sum(), (np.abs(pb) < 25).sum(), (pb > 0).sum(), (pb < 0).sum()))
        L.append('')
    if not rows:
        L += ['no gauge scored']
    L.append('Reading: VIC is uncalibrated here (parameters from the ISC set, 3-hourly forcing disaggregated from ERA5 daily '
             'by MTCLIM, one year, no routing spin-up, no reservoirs, irrigation or abstraction in this combination). '
             'These are baseline scores, not a calibrated skill claim.')
    txt = '\n'.join(L)
    open(os.path.join(a.out_dir, 'summary.txt'), 'w').write(txt + '\n')
    print(txt)

    if rows:
        make_map(rows, a)


def make_map(rows, a):
    try:
        import matplotlib
    except ImportError:               # the map is optional: the scores and CSV do not need it
        print('matplotlib not installed: map skipped', file=sys.stderr)
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    lon = np.array([r['lon'] for r in rows])
    lat = np.array([r['lat'] for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.4), constrained_layout=True)
    bg = None
    if a.domain:
        import netCDF4
        d = netCDF4.Dataset(a.domain)
        bg = (d['lon'][:], d['lat'][:], np.ma.filled(d['mask'][:], 0))
    panels = [('kge', 'KGE', [-1e9, -0.41, 0, 0.3, 0.5, 0.7, 1e9], ['#7f2704', '#d95f0e', '#fdae6b', '#c7e9b4', '#41b6c4', '#225ea8']),
              ('nse', 'NSE', [-1e9, -1, 0, 0.3, 0.5, 0.7, 1e9], ['#7f2704', '#d95f0e', '#fdae6b', '#c7e9b4', '#41b6c4', '#225ea8']),
              ('pbias', 'PBIAS (%), + = model too high', [-1e9, -50, -25, 25, 50, 100, 1e9],
               ['#2166ac', '#92c5de', '#d9d9d9', '#f4a582', '#d6604d', '#8c1d1d'])]
    for ax, (k, title, edges, cols) in zip(axs, panels):
        if bg is not None:
            ax.pcolormesh(bg[0], bg[1], np.where(bg[2] > 0, 1.0, np.nan), cmap=ListedColormap(['#e6e6e6']), shading='auto')
        v = np.array([r[k] for r in rows])
        cmap = ListedColormap(cols)
        norm = BoundaryNorm(edges, cmap.N)
        sc = ax.scatter(lon, lat, c=v, cmap=cmap, norm=norm, s=26, edgecolor='k', linewidth=0.3)
        ax.set_title('%s  (median %.2f, n=%d)' % (title, np.median(v), len(v)))
        ax.set_xlabel('longitude')
        ax.set_aspect('equal')
        fig.colorbar(sc, ax=ax, shrink=0.7, ticks=edges[1:-1])
    axs[0].set_ylabel('latitude')
    fig.suptitle('%s, %s to %s: %d gauges' % (a.label, a.start, a.end, len(rows)))
    fig.savefig(os.path.join(a.out_dir, 'gauge_map.png'), dpi=130)


if __name__ == '__main__':
    main()
