#!/usr/bin/env python3
"""Route TorchVIC/VIC runoff + baseflow over the ISC grid to gauges.

  route_isc.py --out-dir RUN/out --domain domain.nc --gauges basin_index.csv
               [--fdir fdir.nc:VAR | --derive-dem] --device cuda --result X.nc

Flow directions: `--fdir file:var` (ESRI D8 codes on the domain grid).
`--derive-dem` builds a stand-in network from domain.nc's elevations
(priority flood) -- for timing only; it is NOT an observed river network.
Writes gauge discharge (m3/s) as NetCDF and CSV, and prints timings.

Gauges (with --fdir and --snap iou, the default there; polygons from
<data root>/discharge/final_basins/full_shapefile):
  channel gauge   snapped to the cell within 3 cells whose grid catchment has the best IoU with the
                  polygon (tools/gauge_review.py); routed along the D8 network.
  sub-cell gauge  polygon smaller than --subcell-cells ISC cells (default 4, gauge_review.SUBCELL_CELLS),
                  or treatment 'subcell' in tools/data/gauge_corrections.csv: discharge is
                  sum over the cells covered by the polygon of (cell runoff+baseflow x fraction of the
                  cell's area inside the polygon), routed with the cell unit hydrograph only (no
                  channel routing inside one cell).  Fractions are computed from the polygon.
  skipped         treatment 'canal' (regulated diversion, not a natural catchment: needs a water-use
                  module), 'outofmask' (polygon outside the ISC mask).
"""
import argparse
import csv
import glob
import os
import sys
import time

import netCDF4
import numpy as np
import torch

from torchvic.model import river_network as rn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gauge_review as gr  # noqa: E402


def sync(dev):
    if dev == 'cuda':
        torch.cuda.synchronize()


def step_check(a, net, gc, sub, canal, runoff_np, area, q, dev):
    """Stepping router vs the whole-series result, ledger, timings."""
    rng = np.random.default_rng(0)
    sites = rng.choice(net.n, a.step_sites, replace=False) if a.step_sites else []
    ts = time.time()
    r, info = rn.build_step_router(net, gc, area, a.dt, subcell=sub,
                                   n_canal=len(canal), site_cells=sites,
                                   device=dev)
    sync(dev)
    print('step router built %.1fs: nodes %d (channel %d, extra sites %d, sub-cell %d, canal %d), L %d, pairs %d'
          % (time.time() - ts, info['n_node'], info['n_channel'], info['n_site_extra'],
             info['n_subcell'], info['n_canal'], r.L, r.n_pair), flush=True)
    R = torch.as_tensor(runoff_np, device=dev)
    nt = R.shape[0]
    S = len(sites)
    zero = torch.zeros(max(S, 1), dtype=R.dtype, device=dev)
    out = torch.empty(nt, r.n_node, dtype=R.dtype, device=dev)
    for t in range(2):                       # warm-up on a scratch state
        r.step(R[t])
    r.reset()
    sync(dev)
    ts = time.time()
    for t in range(nt):
        out[t] = r.step(R[t], withdraw=zero if S else None)[0]
    sync(dev)
    tt = time.time() - ts
    cols = np.r_[np.arange(info['n_channel']),
                 info['n_channel'] + info['n_site_extra'] + np.arange(info['n_subcell'])]
    got = out[:, cols].cpu().numpy()
    scale = np.abs(q).max(0)
    scale[scale == 0] = 1
    rel = (np.abs(got - q).max(0) / scale)
    print('step vs whole series: %d gauges, max relative difference %.3e (worst gauge col %d); max |diff| %.3e m3/s'
          % (got.shape[1], rel.max(), int(rel.argmax()), np.abs(got - q).max()), flush=True)
    print('step loop: %.3fs for %d steps = %.2f ms/step%s'
          % (tt, nt, 1e3 * tt / nt, ' (with %d-site withdraw hook)' % S if S else ''), flush=True)
    bal = r.balance().abs().max().item() / r.in_runoff.max().item()
    print('ledger: max |in - out - stored| / max in = %.3e  (in %.6e m3, out %.6e m3, stored %.6e m3, node max)'
          % (bal, r.in_runoff.max().item(), r.out.max().item(), (r.ring.sum(0) * a.dt).max().item()), flush=True)
    if S:
        print('hook (zero requests): withdrew %.3e m3' % r.withdrawn.sum().item(), flush=True)
        r.reset()
        req = torch.full((S,), 2e5, dtype=R.dtype, device=dev)
        inj = torch.full((S,), 1e5, dtype=R.dtype, device=dev)
        sync(dev)
        ts = time.time()
        for t in range(nt):
            r.step(R[t], withdraw=req, inject=inj)
        sync(dev)
        tt = time.time() - ts
        bal = r.balance().abs().max().item() / r.in_runoff.max().item()
        print('hooks (request 2e5 m3, inject 1e5 m3 at %d sites): %.2f ms/step; delivered %.4e of %.4e m3 requested; injected %.4e m3; ledger %.3e'
              % (S, 1e3 * tt / nt, r.withdrawn.sum().item(), 2e5 * S * nt, r.injected.sum().item(), bal), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--domain', required=True)
    ap.add_argument('--gauges', required=True)
    ap.add_argument('--fdir')
    ap.add_argument('--derive-dem', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dt', type=float, default=86400.0)
    ap.add_argument('--result', required=True)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--also-cpu', action='store_true')
    ap.add_argument('--step', action='store_true',
                    help='also run the stepping router (routing.StepRouter) and compare with the whole series')
    ap.add_argument('--step-sites', type=int, default=0,
                    help='with --step: this many random cells as withdrawal/injection sites (timing and ledger test)')
    ap.add_argument('--snap', choices=['iou', 'nearest'], default=None)
    ap.add_argument('--subcell-cells', type=float, default=gr.SUBCELL_CELLS)
    ap.add_argument('--corrections', default=gr.CORRECTIONS)
    a = ap.parse_args()
    t0 = time.time()

    d = netCDF4.Dataset(a.domain)
    lat, lon = d['lat'][:].data, d['lon'][:].data
    mask = np.asarray(d['mask'][:]) > 0
    area_g = np.asarray(d['area'][:], dtype=np.float64)
    elev = np.asarray(d['elev'][:], dtype=np.float64)
    if a.fdir:
        f, v = a.fdir.split(':')
        fdir = np.asarray(netCDF4.Dataset(f)[v][:])
        net, cells = rn.d8_downstream(fdir, mask, lat, lon)
    elif a.derive_dem:
        net, cells = rn.derive_d8_from_elevation(elev, mask, lat, lon)
    else:
        raise SystemExit('no flow directions: give --fdir or --derive-dem')
    print('cells %d, network built %.1fs, longest path %.0f km'
          % (net.n, time.time() - t0, net.dist_to_mouth.max() / 1e3), flush=True)

    rows = list(csv.DictReader(open(a.gauges)))
    snap = a.snap or ('iou' if a.fdir else 'nearest')
    cl, cc = lat[cells[:, 0]], lon[cells[:, 1]]
    cell_index = -np.ones(mask.shape, dtype=np.int64)
    cell_index[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
    gc, names, sub, sub_names, skipped, kinds = [], [], [], [], [], []
    seen = {}
    canal = []
    if snap == 'iou':
        g = gr.Grid(a.fdir.split(':')[0], a.domain)
        _, shp = gr.load_gauges()
        cor = gr.read_corrections(a.corrections)
        for r in rows:                                  # moved coordinates (table), as in load_gauges
            c = cor.get(int(r['clean_id']))
            if c and c.get('new_lat'):
                r['lat'], r['lng'] = c['new_lat'], c['new_lon']
        for rec in gr.snap_all(g, rows, shp):
            gid = int(rec['id'])
            treat = cor.get(gid, {}).get('treatment', '')
            if not rec.get('ok'):
                skipped.append((gid, 'outside the domain')); continue
            if treat in ('canal', 'outofmask'):
                skipped.append((gid, treat))
                if treat == 'canal':
                    canal.append(gid)
                continue
            rings = gr.read_polygon(shp[gid])
            ncell = rec['poly_km2'] / g.carea[rec['cell_r'], rec['cell_c']]
            if treat == 'subcell' or ncell < a.subcell_cells:
                idx, fr, pa, _ = gr.subcell_weights(g, rings)
                sub.append((cell_index.ravel()[idx], fr)); sub_names.append((str(gid), rec['name']))
                continue
            i = int(cell_index[rec['cell_r'], rec['cell_c']])
            if i in seen:
                skipped.append((gid, 'shares a cell with gauge %s' % seen[i])); continue
            seen[i] = gid
            gc.append(i); names.append((str(gid), rec['name']))
    else:
        for r in rows:
            la, lo = float(r['lat']), float(r['lng'])
            i = int(np.argmin((cl - la) ** 2 + ((cc - lo) * np.cos(np.radians(la))) ** 2))
            if i in seen:
                continue
            seen[i] = r['clean_id']
            gc.append(i)
            names.append((r['clean_id'], r['name']))
    t1 = time.time()
    pairs = rn.build_pairs(net, np.array(gc))
    if sub:
        pairs = rn.add_subcell_gauges(pairs, sub)
    names = names + sub_names
    kinds = [0] * len(gc) + [1] * len(sub)
    print('gauges %d channel + %d sub-cell + %d skipped (of %d listed), pairs %d, built %.1fs'
          % (len(gc), len(sub), len(skipped), len(rows), pairs.n_pair, time.time() - t1), flush=True)
    for gid, why in skipped:
        print('  skipped gauge %d: %s' % (gid, why))

    files = sorted(glob.glob(a.out_dir + '/fluxes.*.nc'))
    R, times = [], []
    for fn in files:
        ds = netCDF4.Dataset(fn)
        ro = ds['OUT_RUNOFF'][:]
        bf = ds['OUT_BASEFLOW'][:]
        # cells of the domain the run did not simulate are masked (fill 9.97e36): no runoff
        tot = (np.ma.filled(np.ma.asarray(ro, dtype=np.float64), 0.0)
               + np.ma.filled(np.ma.asarray(bf, dtype=np.float64), 0.0))
        R.append(tot[:, cells[:, 0], cells[:, 1]])
        tv = ds['time']
        times += list(netCDF4.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'),
                                         only_use_cftime_datetimes=False))
    runoff_np = np.concatenate(R)
    if a.max_steps:
        runoff_np = runoff_np[:a.max_steps]
        times = times[:a.max_steps]
    runoff_np = np.nan_to_num(runoff_np)
    area = area_g[cells[:, 0], cells[:, 1]]
    print('runoff %s read %.1fs' % (runoff_np.shape, time.time() - t0), flush=True)

    res = {}
    devs = [a.device] + (['cpu'] if a.also_cpu and a.device != 'cpu' else [])
    for dev in devs:
        sync(dev)
        ts = time.time()
        r = torch.as_tensor(runoff_np, device=dev)
        ar = torch.as_tensor(area, device=dev)
        cuh = torch.ones(1, dtype=torch.float64, device=dev)
        uh = rn.pair_unit_hydrographs(
            torch.as_tensor(pairs.distance, device=dev), 1.5, 800.0, cuh, a.dt)
        sync(dev)
        tu = time.time() - ts
        ts = time.time()
        q = rn.route_network(r, pairs, uh, ar, a.dt)
        sync(dev)
        tr = time.time() - ts
        print('%s: unit hydrographs %.2fs (%d x %d), routing %.2fs'
              % (dev, tu, uh.shape[0], uh.shape[1], tr), flush=True)
        res[dev] = q.cpu().numpy()
    if len(res) == 2:
        e = np.abs(res['cpu'] - res[a.device]).max() / np.abs(res['cpu']).max()
        print('GPU vs CPU max relative difference %.3e' % e, flush=True)
    q = res[a.device]
    if a.step:
        step_check(a, net, np.array(gc), sub, canal, runoff_np, area, q,
                   a.device)

    nc = netCDF4.Dataset(a.result, 'w')
    nc.createDimension('time', q.shape[0])
    nc.createDimension('gauge', q.shape[1])
    tv = nc.createVariable('time', 'f8', ('time',))
    tv.units = 'days since 1900-01-01'
    tv[:] = netCDF4.date2num(times, tv.units)
    gv = nc.createVariable('clean_id', 'i4', ('gauge',))
    gv[:] = [int(n[0]) for n in names]
    kv = nc.createVariable('sub_cell', 'i1', ('gauge',))
    kv[:] = kinds
    kv.comment = '1: area-weighted sub-cell gauge (no channel routing), 0: routed along the D8 network'
    qv = nc.createVariable('discharge', 'f8', ('time', 'gauge'))
    qv.units = 'm3 s-1'
    qv[:] = q
    nc.description = ('Gauge discharge from OUT_RUNOFF+OUT_BASEFLOW, '
                      'Lohmann channel UH, flow network: %s'
                      % ('file ' + a.fdir if a.fdir else
                         'DEM priority-flood stand-in (timing only)'))
    nc.close()
    with open(a.result.replace('.nc', '.csv'), 'w') as fh:
        w = csv.writer(fh)
        w.writerow(['date'] + [n[0] for n in names])
        for i, t in enumerate(times):
            w.writerow([t.strftime('%Y-%m-%d')] + ['%.6g' % x for x in q[i]])
    print('done %.1fs total' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
