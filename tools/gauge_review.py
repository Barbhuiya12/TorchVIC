#!/usr/bin/env python3
"""Gauge review for the ISC GRIT flow map: polygon areas from the shapefiles,
IoU-based snapping, per-gauge panels, validation before/after.

  gauge_review.py stats    --fdir F --out-csv X.csv [--flags flags.csv]
  gauge_review.py panels   --fdir F --ids 482,23,... --out-dir D [--gpkg G]

Polygon areas are recomputed from the .shp files (spherical equal-area,
WGS84 authalic radius), not read from basin_index.csv.  A gauge is snapped to
the cell within 3 cells whose grid catchment has the best IoU with the
polygon (polygons smaller than one cell fall back to the area ratio).
Pure numpy/scipy/matplotlib; the shapefile is parsed directly (untrusted data).
"""
import argparse
import csv
import glob
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_isc_fdir_grit as bg  # noqa: E402

ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
W_AREA = float(os.environ.get("GR_WAREA", 3.0))          # weight of the |ln area ratio| term next to the IoU
R_AUTH = 6371007.2  # authalic radius, m


# ------------------------------------------------------------- shapefile
def read_polygon(path):
    """List of rings [(n,2) lon/lat arrays] of the first polygon record, with
    the shapefile orientation (clockwise = outer, counter-clockwise = hole)."""
    b = open(path, 'rb').read()
    st = struct.unpack_from('<i', b, 100 + 8)[0]
    assert st in (5, 15, 25), st
    o = 100 + 8
    nprt, np_ = struct.unpack_from('<ii', b, o + 36)
    parts = list(struct.unpack_from('<%di' % nprt, b, o + 44))
    pts = np.frombuffer(b, '<f8', count=np_ * 2, offset=o + 44 + 4 * nprt).reshape(np_, 2)
    parts.append(np_)
    return [pts[parts[i]:parts[i + 1]] for i in range(nprt)]


def ring_area_km2(r):
    lam, phi = np.radians(r[:, 0]), np.radians(r[:, 1])
    s = np.sin(phi)
    a = 0.5 * np.sum((lam[1:] - lam[:-1]) * (s[1:] + s[:-1]))
    return a * R_AUTH ** 2 / 1e6          # signed: positive = counter-clockwise


def polygon_area_km2(rings):
    """Outer rings share the orientation of the largest ring, holes the opposite."""
    a = np.array([ring_area_km2(r) for r in rings])
    return float(np.sign(a[np.argmax(np.abs(a))]) * a.sum())


def polygon_fraction(rings, lat, lon, r0, r1, c0, c1, sub=4):
    """Fraction of each cell [r0:r1, c0:c1] inside the polygon (sub x sub
    sampling, even-odd rule over all rings)."""
    from matplotlib.path import Path
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    off = (np.arange(sub) + 0.5) / sub - 0.5
    yy = lat[r0] + (np.arange(r1 - r0)[:, None] + off[None, :]).ravel() * dlat
    xx = lon[c0] + (np.arange(c1 - c0)[:, None] + off[None, :]).ravel() * dlon
    X, Y = np.meshgrid(xx, yy)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    ins = np.zeros(len(P), bool)
    for ring in rings:
        lo, hi = ring.min(0), ring.max(0)
        m = (P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1])
        if m.any():
            ins[m] ^= Path(ring).contains_points(P[m])
    ins = ins.reshape(r1 - r0, sub, c1 - c0, sub)
    return ins.mean(axis=(1, 3))


# ----------------------------------------------------------------- grid
class Grid:
    def __init__(self, fdir_file, domain=ROOT + '/vic_params_isc/domain.nc'):
        import netCDF4
        from scipy.sparse import csr_matrix
        d = netCDF4.Dataset(domain)
        self.lat, self.lon = d['lat'][:].data, d['lon'][:].data
        self.mask = np.asarray(d['mask'][:]) > 0
        self.carea = np.asarray(d['area'][:], dtype=np.float64) / 1e6
        f = netCDF4.Dataset(fdir_file)
        self.fd = np.asarray(f['fdir'][:])
        self.ny, self.nx = self.mask.shape
        tgt = bg._target(np.where(self.mask, self.fd, 0), self.mask)
        self.tgt = tgt
        n = self.ny * self.nx
        src = np.nonzero(tgt >= 0)[0]
        # reverse graph (downstream -> upstream)
        self.rev = csr_matrix((np.ones(len(src)), (tgt[src], src)), shape=(n, n))
        self.up = bg.accumulate(np.where(self.mask, self.fd, 0), self.mask, self.carea)

    def edited(self, edits):
        """Copy of the grid with {(r,c): ESRI code} direction edits applied."""
        import copy
        from scipy.sparse import csr_matrix
        h = copy.copy(self)
        fd = self.fd.copy()
        for (r, c), code in edits.items():
            fd[r, c] = code
        h.fd = fd
        tgt = bg._target(np.where(self.mask, fd, 0), self.mask)
        h.tgt = tgt
        n = self.ny * self.nx
        src = np.nonzero(tgt >= 0)[0]
        h.rev = csr_matrix((np.ones(len(src)), (tgt[src], src)), shape=(n, n))
        return h

    def area_of(self, idx):
        return self.carea.ravel()[idx].sum()

    def catchment(self, r, c):
        from scipy.sparse.csgraph import breadth_first_order
        idx = breadth_first_order(self.rev, r * self.nx + c, directed=True,
                                  return_predecessors=False)
        return idx


def iou_at(g, rings, bbox, frac, r, c):
    r0, r1, c0, c1 = bbox
    idx = g.catchment(r, c)
    rr, cc = np.divmod(idx, g.nx)
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    m = np.zeros((r1 - r0, c1 - c0))
    m[rr[keep] - r0, cc[keep] - c0] = 1.0
    out_area = g.carea.ravel()[idx[~keep]].sum()   # catchment outside the polygon bbox
    w = g.carea[r0:r1, c0:c1]
    inter = (np.minimum(m, frac) * w).sum()
    union = (np.maximum(m, frac) * w).sum() + out_area
    return inter / union if union > 0 else 0.0


def poly_bbox(g, rings, pad=3):
    lo = np.min([r.min(0) for r in rings], 0)
    hi = np.max([r.max(0) for r in rings], 0)
    dlat, dlon = g.lat[1] - g.lat[0], g.lon[1] - g.lon[0]
    r0 = max(int(np.floor((lo[1] - g.lat[0]) / dlat)) - pad, 0)
    r1 = min(int(np.ceil((hi[1] - g.lat[0]) / dlat)) + pad + 1, g.ny)
    c0 = max(int(np.floor((lo[0] - g.lon[0]) / dlon)) - pad, 0)
    c1 = min(int(np.ceil((hi[0] - g.lon[0]) / dlon)) + pad + 1, g.nx)
    return r0, r1, c0, c1


CORRECTIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gauge_corrections.csv')
SUBCELL_CELLS = 4.0     # a polygon smaller than this many ISC cells is a sub-cell gauge (area-weighted cell runoff)


def read_corrections(path=CORRECTIONS):
    """{gauge id: row} of tools/data/gauge_corrections.csv (decision table of the hand-checks)."""
    if not path or not os.path.exists(path):
        return {}
    return {int(r['gauge_id']): r for r in csv.DictReader(open(path))}


def load_gauges(corrections=True):
    """basin_index rows and shapefile paths; gauge coordinates moved where the correction table
    says treatment 'coords' (new_lat/new_lon)."""
    rows = list(csv.DictReader(open(ROOT + '/discharge/final_basins/basin_index.csv')))
    if corrections:
        cor = read_corrections()
        for r in rows:
            c = cor.get(int(r['clean_id']))
            if c and c.get('new_lat'):
                r['lat'], r['lng'] = c['new_lat'], c['new_lon']
    shp = {}
    for p in glob.glob(ROOT + '/discharge/final_basins/full_shapefile/*.shp'):
        shp[int(os.path.basename(p).split('_')[0])] = p
    return rows, shp


def subcell_weights(g, rings, sub=16):
    """Cells covered by a small polygon and the fraction of each cell's area inside it.

    Returns (flat indices of mask cells, fractions, polygon area km2, cell-area-weighted check).
    Fractions come from sub x sub point sampling per cell; they are scaled by one factor so that
    sum(f * cell area) over ALL covered cells (in or out of the mask) equals the polygon area from
    the shapefile (removes the sampling error); cells outside the mask are then dropped."""
    pa = polygon_area_km2(rings)
    r0, r1, c0, c1 = poly_bbox(g, rings, 1)
    frac = polygon_fraction(rings, g.lat, g.lon, r0, r1, c0, c1, sub=sub)
    w = g.carea[r0:r1, c0:c1]
    tot = float((frac * w).sum())
    if tot > 0:
        frac = frac * (pa / tot)
    rr, cc = np.nonzero(frac > 0)
    keep = g.mask[rr + r0, cc + c0]
    idx = (rr[keep] + r0) * g.nx + (cc[keep] + c0)
    return idx, frac[rr[keep], cc[keep]], pa, float((frac[rr[keep], cc[keep]] * w[rr[keep], cc[keep]]).sum())


def snap_all(g, rows, shp, radius=3, verbose=False):
    """Per gauge: polygon area, nearest cell, area-snap cell (old rule, +-2),
    IoU-snap cell (+-radius)."""
    res = []
    for k, row in enumerate(rows):
        gid = int(row['clean_id'])
        la, lo = float(row['lat']), float(row['lng'])
        rings = read_polygon(shp[gid])
        pa = polygon_area_km2(rings)
        r0 = int(round((la - g.lat[0]) / (g.lat[1] - g.lat[0])))
        c0 = int(round((lo - g.lon[0]) / (g.lon[1] - g.lon[0])))
        rec = dict(id=gid, name=row['name'], lat=la, lon=lo, poly_km2=pa,
                   csv_poly_km2=float(row['area_poly_km2'] or 'nan'),
                   stated_km2=float(row['area'] or 'nan'))
        if not (0 <= r0 < g.ny and 0 <= c0 < g.nx):
            res.append(rec | dict(ok=False)); continue
        bbox = poly_bbox(g, rings)
        frac = polygon_fraction(rings, g.lat, g.lon, *bbox)
        small = pa < g.carea[r0, c0]
        cand = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < g.ny and 0 <= c < g.nx and g.mask[r, c]):
                    continue
                iou = np.nan if small else iou_at(g, rings, bbox, frac, r, c)
                cand.append((r, c, np.hypot(dr, dc), g.up[r, c], iou))
        if not cand:
            res.append(rec | dict(ok=False)); continue
        cand = np.array(cand)
        near = (r0, c0) if g.mask[r0, c0] else tuple(cand[np.argmin(cand[:, 2]), :2].astype(int))
        # old rule: area match only, within 2 cells
        m2 = np.array([abs(x[0] - r0) <= 2 and abs(x[1] - c0) <= 2 for x in cand])
        sc_old = np.abs(np.log(cand[:, 3] / pa)) + 1e-6 * cand[:, 2]
        sc_old[~m2] = np.inf
        io = int(np.argmin(sc_old))
        # new rule: IoU of the grid catchment with the polygon, area misfit as a
        # secondary term, then distance; polygons < 1 cell: area only
        lr = np.abs(np.log(cand[:, 3] / pa))
        if small:
            sc = -lr
        else:
            sc = cand[:, 4] - W_AREA * np.minimum(lr, 1.0)
        sc = sc - 1e-4 * cand[:, 2]
        i = int(np.argmax(sc))
        rec.update(ok=True, small=small, cell_r=int(cand[i, 0]), cell_c=int(cand[i, 1]),
                   grid_km2=cand[i, 3], iou=cand[i, 4],
                   old_r=int(cand[io, 0]), old_c=int(cand[io, 1]),
                   old_grid_km2=cand[io, 3], old_iou=cand[io, 4],
                   near_km2=g.up[near],
                   dist_cells=float(cand[i, 2]))
        res.append(rec)
        if verbose and k % 50 == 0:
            print('snapped', k, flush=True)
    return res


def err_stats(vals, ref, label, iou=None):
    ae = np.abs(vals - ref) / ref * 100
    s = ('%s: n=%d median|err|=%.1f%% p90|err|=%.1f%% within10=%d within20=%d'
         % (label, len(ae), np.median(ae), np.percentile(ae, 90), (ae <= 10).sum(), (ae <= 20).sum()))
    if iou is not None:
        s += ' meanIoU=%.3f' % np.nanmean(iou)
    return s


def read_flags(path):
    if not path or not os.path.exists(path):
        return {}
    return {int(r['id']): r for r in csv.DictReader(open(path))}


def cmd_stats(a):
    g = Grid(a.fdir)
    rows, shp = load_gauges()
    res = [r for r in snap_all(g, rows, shp, verbose=True) if r['ok']]
    flags = read_flags(a.flags)
    nonroute = {i for i, r in flags.items() if r['class'] in ('b', 'd')}
    with open(a.out_csv, 'w', newline='') as f:
        keys = list(res[0])
        w = csv.DictWriter(f, fieldnames=keys + ['flag'], extrasaction='ignore')
        w.writeheader()
        for r in res:
            w.writerow(r | {'flag': flags.get(r['id'], {}).get('class', '')})
    sel = [r for r in res if not r['small']]
    for nm, sub in (('all with polygon >= 1 cell', sel),
                    ('excluding flagged b/d', [r for r in sel if r['id'] not in nonroute])):
        ref = np.array([r['poly_km2'] for r in sub])
        print(nm)
        print('  ', err_stats(np.array([r['near_km2'] for r in sub]), ref, 'nearest cell'))
        print('  ', err_stats(np.array([r['old_grid_km2'] for r in sub]), ref, 'area-snap +-2 (old)',
                              [r['old_iou'] for r in sub]))
        print('  ', err_stats(np.array([r['grid_km2'] for r in sub]), ref, 'IoU-snap +-3 (new)',
                              [r['iou'] for r in sub]))
    d = np.array([r['poly_km2'] / r['csv_poly_km2'] - 1 for r in res])
    print('polygon area vs csv area_poly_km2: max|diff|=%.2f%% n>1%%=%d' %
          (100 * np.nanmax(np.abs(d)), int((np.abs(d) > .01).sum())))
    print('small polygons (< 1 cell): %d' % sum(r['small'] for r in res))


GPKG = ROOT + '/static/grit/GRITv06_segments_AS_EPSG4326.gpkg'


def panel(g, rec, shp, segs_cache, out_png, note=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm
    gid = rec['id']
    rings = read_polygon(shp[gid])
    pad = 6
    r0, r1, c0, c1 = poly_bbox(g, rings, pad)
    r0 = max(r0 - 0, 0)
    ext = [g.lon[c0] - 0.03125, g.lon[c1 - 1] + 0.03125, g.lat[r0] - 0.03125, g.lat[r1 - 1] + 0.03125]
    segs = bg.read_segments(GPKG, (ext[0], ext[2], ext[1], ext[3]))
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    up = np.where(g.mask, g.up, np.nan)[r0:r1, c0:c1]
    r, c = rec['cell_r'], rec['cell_c']
    ro, co = rec['old_r'], rec['old_c']
    idx = g.catchment(r, c)
    rr, cc = np.divmod(idx, g.nx)
    gm = np.zeros(g.mask.shape); gm[rr, cc] = 1
    gm = gm[r0:r1, c0:c1]
    bbox = (r0, r1, c0, c1)
    frac = polygon_fraction(rings, g.lat, g.lon, *bbox)
    for k, a in enumerate(ax):
        a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3]); a.set_aspect('equal')
        for ring in rings:
            a.plot(ring[:, 0], ring[:, 1], color='k', lw=1.2, zorder=6)
        a.plot(rec['lon'], rec['lat'], marker='*', ms=14, color='gold', mec='k', zorder=9, ls='')
        a.plot(g.lon[c], g.lat[r], marker='s', ms=9, mfc='none', mec='red', mew=2, zorder=9, ls='')
        if (ro, co) != (r, c):
            a.plot(g.lon[co], g.lat[ro], marker='D', ms=8, mfc='none', mec='magenta', mew=1.5, zorder=9, ls='')
    a = ax[0]
    im = a.imshow(up, origin='lower', extent=ext, cmap='viridis', norm=LogNorm(10, max(np.nanmax(up), 100)), zorder=1)
    lc = []; lw = []
    for sg in segs:
        lc.append(sg['xy']); lw.append(0.3 + 0.35 * np.log10(max(sg['aout'], 1.0)))
    a.add_collection(LineCollection(lc, linewidths=lw, colors='w', alpha=0.8, zorder=3))
    # D8 arrows
    ys, xs = np.mgrid[r0:r1, c0:c1]
    dr = np.zeros(up.shape); dc = np.zeros(up.shape)
    for code, (a_, b_) in bg.ESRI.items():
        m = g.fd[r0:r1, c0:c1] == code
        dr[m], dc[m] = a_, b_
    ok = np.isfinite(up) & (up > 200)
    a.quiver(g.lon[xs][ok], g.lat[ys][ok], dc[ok], dr[ok], color='r', angles='xy', scale_units='xy',
             scale=1 / 0.8 * 16, width=0.002, zorder=4)
    a.set_title('upstream area (log km2), GRIT reaches (white), D8 arrows (>200 km2)')
    plt.colorbar(im, ax=a, fraction=0.04)
    a = ax[1]
    a.imshow(np.where(g.mask[r0:r1, c0:c1], g.up[r0:r1, c0:c1] > 0, np.nan), origin='lower', extent=ext, cmap='Greys', vmin=0, vmax=3)
    ov = np.zeros(up.shape + (4,))
    ov[(gm > 0) & (frac >= .5)] = (0.1, 0.7, 0.1, 0.8)
    ov[(gm > 0) & (frac < .5)] = (0.9, 0.1, 0.1, 0.8)
    ov[(gm == 0) & (frac >= .5)] = (0.1, 0.3, 0.9, 0.8)
    a.imshow(ov, origin='lower', extent=ext, zorder=2)
    a.set_title('green: grid catchment & polygon; red: grid only; blue: polygon only')
    a = ax[2]
    a.imshow(np.where(g.mask[r0:r1, c0:c1], 1.0, np.nan), origin='lower', extent=ext, cmap='Greys', vmin=0, vmax=3)
    lc = [sg['xy'] for sg in segs]
    a.add_collection(LineCollection(lc, linewidths=[0.4 + 0.5 * np.log10(max(sg['aout'], 1)) for sg in segs],
                                    colors='#1f5fa8', zorder=3))
    for sg in segs:
        pass
    a.set_title('GRIT reaches (width ~ log drainage area)')
    s = ('%d %s  poly %.0f km2 | grid %.0f (IoU %.2f) | old-rule cell %.0f (IoU %.2f) | snap %.1f cells\n'
         'star gauge, red square snapped cell, magenta diamond old area-snap cell. %s'
         % (gid, rec['name'], rec['poly_km2'], rec['grid_km2'], rec['iou'], rec['old_grid_km2'],
            rec['old_iou'], rec['dist_cells'], note))
    fig.suptitle(s, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def cmd_panels(a):
    g = Grid(a.fdir)
    rows, shp = load_gauges()
    ids = {int(x) for x in a.ids.split(',')}
    rows = [r for r in rows if int(r['clean_id']) in ids]
    res = [r for r in snap_all(g, rows, shp) if r['ok']]
    notes = {}
    if a.flags:
        for k, v in read_flags(a.flags).items():
            notes[k] = '[%s] %s' % (v['class'], v['reason'])
    os.makedirs(a.out_dir, exist_ok=True)
    for r in res:
        out = '%s/%03d_%s.png' % (a.out_dir, r['id'], r['name'].replace('/', '_'))
        panel(g, r, shp, None, out, notes.get(r['id'], ''))
        print(out, flush=True)


def diagnose(g, rec, shp, top=4):
    """Polygon exits (in-polygon cells draining out of it) and foreign inflows
    (outside cells draining into it), largest upstream area first."""
    rings = read_polygon(shp[rec['id']])
    r0, r1, c0, c1 = poly_bbox(g, rings, 2)
    frac = polygon_fraction(rings, g.lat, g.lon, r0, r1, c0, c1)
    inside = np.zeros(g.mask.shape, bool)
    inside[r0:r1, c0:c1] = frac >= 0.5
    ii = np.nonzero(inside.ravel() & g.mask.ravel())[0]
    t = g.tgt[ii]
    ex = ii[(t < 0) | ~inside.ravel()[np.maximum(t, 0)]]
    ex = ex[np.argsort(-g.up.ravel()[ex])][:top]
    src = np.nonzero((g.tgt >= 0) & ~inside.ravel())[0]
    fin = src[inside.ravel()[g.tgt[src]]]
    fin = fin[np.argsort(-g.up.ravel()[fin])][:top]
    f = lambda L: ['(%d,%d)%.0f->%s' % (k // g.nx, k % g.nx, g.up.ravel()[k],
                   'out' if g.tgt[k] < 0 else '(%d,%d)' % divmod(int(g.tgt[k]), g.nx)) for k in L]
    return f(ex), f(fin), int(inside.sum())


def cmd_diag(a):
    g = Grid(a.fdir)
    rows, shp = load_gauges()
    ids = {int(x) for x in a.ids.split(',')}
    rows = [r for r in rows if int(r['clean_id']) in ids]
    for r in [r for r in snap_all(g, rows, shp) if r['ok']]:
        ex, fin, n = diagnose(g, r, shp)
        print('%d %s poly=%.0f cells=%d snap=(%d,%d) grid=%.0f iou=%.2f gauge=(%.3f,%.3f)' % (
            r['id'], r['name'], r['poly_km2'], n, r['cell_r'], r['cell_c'], r['grid_km2'], r['iou'], r['lat'], r['lon']))
        print('   exits:', ex)
        print('   foreign inflows:', fin)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('stats')
    s.add_argument('--fdir', required=True)
    s.add_argument('--out-csv', required=True)
    s.add_argument('--flags')
    p = sp.add_parser('panels')
    p.add_argument('--fdir', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--flags')
    d = sp.add_parser('diag')
    d.add_argument('--fdir', required=True)
    d.add_argument('--ids', required=True)
    a = ap.parse_args()
    {'stats': cmd_stats, 'panels': cmd_panels, 'diag': cmd_diag}[a.cmd](a)


if __name__ == '__main__':
    main()
