#!/usr/bin/env python3
"""Hand-check of single gauges against the native-resolution HydroSHEDS 15 s network.

  gauge_handcheck.py --ids 376,401,518,539,549 --maps hs_ihu=a.nc,hs_eam=b.nc,... --out-dir D

Per gauge: the 15 s catchment traced upstream from the 15 s cell (within RADIUS deg of the
gauge point) whose catchment has the best IoU with the polygon (area misfit as a second term),
the polygon outlet (polygon 15 s cell with the largest upstream area), distances gauge ->
traced cell / polygon outlet, and for every ISC map the IoU-snapped catchment.  The panel shows
the gauge point, the polygon, the 15 s rivers (upstream area, log), the 15 s catchment and the
ISC-map catchments.  Needs pyflwdir (tools/requirements_fdir.txt).  Writes one CSV of numbers
and one PNG per gauge.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gauge_review as gr  # noqa: E402
import fdir_sources as fs  # noqa: E402

RADIUS = 0.12      # deg, search radius for the native-resolution gauge cell


def km_between(lat1, lon1, lat2, lon2):
    return float(np.hypot((lat2 - lat1) * 111.19, (lon2 - lon1) * 111.19 * np.cos(np.radians(lat1))))


def native(gid, row, rings, pad=1.0):
    """15 s window around the polygon, flow raster, polygon mask, upstream area."""
    import pyflwdir
    from affine import Affine
    a, dx = fs.read_hs15()                       # north-up, LAT1..LAT0
    lo = np.min([r.min(0) for r in rings], 0)
    hi = np.max([r.max(0) for r in rings], 0)
    x0, x1 = lo[0] - pad, hi[0] + pad
    y0, y1 = lo[1] - pad, hi[1] + pad
    c0, c1 = int((x0 - fs.LON0) / dx), int((x1 - fs.LON0) / dx)
    r0, r1 = int((fs.LAT1 - y1) / dx), int((fs.LAT1 - y0) / dx)
    sub = a[r0:r1, c0:c1]
    tr = Affine(dx, 0, fs.LON0 + c0 * dx, 0, -dx, fs.LAT1 - r0 * dx)
    flw = pyflwdir.from_array(sub, ftype='d8', transform=tr, latlon=True, check_ftype=False)
    ny, nx = sub.shape
    lon = fs.LON0 + (c0 + np.arange(nx) + 0.5) * dx
    lat = fs.LAT1 - (r0 + np.arange(ny) + 0.5) * dx
    from matplotlib.path import Path
    X, Y = np.meshgrid(lon, lat)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    ins = np.zeros(len(P), bool)
    for ring in rings:
        ins ^= Path(ring).contains_points(P)
    pm = ins.reshape(ny, nx)
    up = flw.upstream_area(unit='km2')
    return flw, sub, lon, lat, pm, up, dx


def trace(flw, ny, nx, idx):
    return flw.basins(idxs=np.array([idx])).reshape(ny, nx) > 0


def handcheck(gid, row, shp, grids, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    rings = gr.read_polygon(shp[gid])
    pa = gr.polygon_area_km2(rings)
    glat, glon = float(row['lat']), float(row['lng'])
    flw, sub, lon, lat, pm, up, dx = native(gid, row, rings)
    ny, nx = sub.shape
    cell_km2 = up.copy()
    # polygon outlet: polygon cell with the largest upstream area
    upm = np.where(pm & (sub != 247), up, -1)
    ro, co = np.unravel_index(np.argmax(upm), upm.shape)
    # native gauge cell: best IoU within RADIUS of the gauge point
    gi, gj = int(round((fs.LAT1 - glat) / dx - 0.5)) - int(round((fs.LAT1 - lat[0]) / dx - 0.5)), \
        int(round((glon - lon[0]) / dx))
    n = int(RADIUS / dx)
    best = None
    pw = pm.sum()
    for di in range(-n, n + 1):
        for dj in range(-n, n + 1):
            i, j = gi + di, gj + dj
            if not (0 <= i < ny and 0 <= j < nx) or sub[i, j] == 247 or up[i, j] < 0.3 * pa:
                continue
            if up[i, j] > 3 * pa:
                continue
            b = trace(flw, ny, nx, i * nx + j)
            inter = (b & pm).sum()
            iou = inter / max((b | pm).sum(), 1)
            lr = min(abs(np.log(up[i, j] / pa)), 1.0)
            sc = iou - gr.W_AREA * lr - 1e-4 * np.hypot(di, dj)
            if best is None or sc > best[0]:
                best = (sc, iou, i, j, b)
    # river cell at the gauge point: largest upstream area within 2 km (3 cells) of the gauge point
    ag = [(up[gi + di, gj + dj], gi + di, gj + dj) for di in range(-3, 4) for dj in range(-3, 4)
          if 0 <= gi + di < ny and 0 <= gj + dj < nx and sub[gi + di, gj + dj] != 247]
    ua, ig, jg = max(ag)
    bg_ = trace(flw, ny, nx, ig * nx + jg)
    cover = float((bg_ & pm).sum() / max(pm.sum(), 1))
    res = dict(id=gid, name=row['name'], gauge_lat=glat, gauge_lon=glon, poly_km2=round(pa, 1))
    if best is None:
        res.update(native_note='no 15 s cell with 0.3-3x polygon area within %.2f deg' % RADIUS)
        best = (0, np.nan, ro, co, trace(flw, ny, nx, ro * nx + co))
    _, iou, i, j, nb = best
    res.update(polygon_outlet_lat=round(lat[ro], 4), polygon_outlet_lon=round(lon[co], 4),
               polygon_outlet_up_km2=round(up[ro, co]), gauge_to_poly_outlet_km=round(km_between(glat, glon, lat[ro], lon[co]), 1),
               native_cell_lat=round(lat[i], 4), native_cell_lon=round(lon[j], 4), native_up_km2=round(up[i, j]),
               native_err_pct=round((up[i, j] / pa - 1) * 100, 1), native_iou=round(float(iou), 3),
               gauge_to_native_km=round(km_between(glat, glon, lat[i], lon[j]), 1))
    res.update(at_gauge_up_km2=round(float(ua)), at_gauge_polygon_coverage=round(cover, 3),
               at_gauge_extra_km2=round(float(ua - pa)))
    nb_poly_out = (nb & ~pm).sum() * dx * dx * 111.19 ** 2 * np.cos(np.radians(glat))
    res['native_catch_outside_poly_km2'] = round(float(nb_poly_out))
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    lo = np.min([r.min(0) for r in rings], 0)
    hi = np.max([r.max(0) for r in rings], 0)
    pd = max(0.12, 0.25 * max(hi - lo))
    ext = [lon[0] - dx / 2, lon[-1] + dx / 2, lat[-1] - dx / 2, lat[0] + dx / 2]
    cols = dict(zip(grids, ['tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']))
    for k, a in enumerate(ax):
        a.imshow(np.where(sub != 247, np.maximum(up, 0.1), np.nan), extent=ext, cmap='Blues', norm=LogNorm(0.5, 5e4), zorder=1)
        a.contour(lon, lat, nb.astype(float), [0.5], colors='cyan', linewidths=1.6, zorder=4)
        a.contour(lon, lat, pm.astype(float), [0.5], colors='k', linewidths=1.4, zorder=5)
        a.plot(glon, glat, '*', ms=15, color='gold', mec='k', zorder=9)
        a.plot(lon[j], lat[i], 's', ms=8, mfc='none', mec='cyan', mew=2, zorder=9)
        a.plot(lon[co], lat[ro], 'o', ms=8, mfc='none', mec='lime', mew=2, zorder=9)
        a.set_xlim(lo[0] - pd, hi[0] + pd); a.set_ylim(lo[1] - pd, hi[1] + pd); a.set_aspect('equal')
    ax[0].set_title('15 s HydroSHEDS rivers (log km2), polygon (black), traced 15 s catchment (cyan)')
    info = []
    for nm, g in grids.items():
        rec = gr.snap_all(g, [row], shp)[0]
        if not rec.get('ok'):
            continue
        idx = g.catchment(rec['cell_r'], rec['cell_c'])
        m = np.zeros(g.mask.shape); m[np.divmod(idx, g.nx)] = 1
        ax[1].contour(g.lon, g.lat, m, [0.5], colors=cols[nm], linewidths=1.3, zorder=6)
        ax[1].plot([], [], color=cols[nm], label='%s %.0f km2 IoU %.2f' % (nm, rec['grid_km2'], rec['iou']))
        res['%s_km2' % nm] = round(rec['grid_km2']); res['%s_iou' % nm] = round(float(rec['iou']), 3)
        info.append(nm)
    ax[1].legend(loc='lower left', fontsize=8)
    ax[1].set_title('ISC-map catchments (IoU-snapped): ' + ', '.join(info))
    fig.suptitle('%d %s | poly %.0f km2 | 15 s cell %.0f km2 (%+.0f%%, IoU %.2f) | gauge->poly outlet %.1f km, gauge->15 s cell %.1f km\n'
                 'star gauge, green circle polygon outlet, cyan square traced 15 s cell' % (
                     gid, row['name'], pa, res['native_up_km2'], res['native_err_pct'], res['native_iou'],
                     res['gauge_to_poly_outlet_km'], res['gauge_to_native_km']), fontsize=10)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig('%s/%03d_handcheck.png' % (out_dir, gid), dpi=85)
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', required=True)
    ap.add_argument('--maps', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    grids = {}
    for m in a.maps.split(','):
        n, f = m.split('=')
        grids[n] = gr.Grid(f)
    rows, shp = gr.load_gauges()
    ids = {int(x) for x in a.ids.split(',')}
    out = []
    for row in rows:
        gid = int(row['clean_id'])
        if gid in ids:
            r = handcheck(gid, row, shp, grids, a.out_dir)
            out.append(r)
            print(r, flush=True)
    keys = sorted({k for r in out for k in r}, key=lambda k: (k not in out[0], k))
    with open(a.out_dir + '/handcheck.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(out)


if __name__ == '__main__':
    main()
