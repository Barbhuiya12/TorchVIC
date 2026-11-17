#!/usr/bin/env python3
"""Score several ISC flow-direction maps at all gauges, build a combined map,
and report out-of-mask catchments / canal diversion sources.

  fdir_multi.py score   --maps name=file.nc,... --out-dir D          per-map gauge CSV + table
  fdir_multi.py combine --maps name=file.nc,... --base NAME --out F  transplant per gauge polygon
  fdir_multi.py extend  --fdir F --out-dir D                          cells to add to the mask per gauge
  fdir_multi.py canals  --fdir F --out CSV                            Gurdaspur canal gauges -> source cell

Scoring is tools/gauge_review.py: IoU-based snapping (+-3 cells), polygons from the .shp files.
Python: numpy, scipy, netCDF4, matplotlib (scratch venv, tools/requirements_fdir.txt).
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_isc_fdir_grit as bg  # noqa: E402
import gauge_review as gr  # noqa: E402

ROOT = gr.ROOT
FLAGS = ROOT + '/static/isc_gauge_flags.csv'


# ---------------------------------------------------------------- scoring
def score_map(name_path):
    name, path = name_path
    g = gr.Grid(path)
    rows, shp = gr.load_gauges()
    res = [r for r in gr.snap_all(g, rows, shp) if r['ok']]
    return name, res


def stat_line(res, label, key='grid_km2'):
    sel = [r for r in res if not r['small']]
    ref = np.array([r['poly_km2'] for r in sel])
    ae = np.abs(np.array([r[key] for r in sel]) - ref) / ref * 100
    iou = np.array([r['iou'] for r in sel])
    return dict(map=label, n=len(sel), median=np.median(ae), p90=np.percentile(ae, 90),
                w10=int((ae <= 10).sum()), w20=int((ae <= 20).sum()), miou=float(np.nanmean(iou)))


def fmt(s):
    return '%-14s n=%d median|err|=%.1f%% p90=%.1f%% within10=%d within20=%d meanIoU=%.3f' % (
        s['map'], s['n'], s['median'], s['p90'], s['w10'], s['w20'], s['miou'])


def write_csv(path, res):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]), extrasaction='ignore')
        w.writeheader()
        w.writerows(res)


def cmd_score(a):
    maps = [m.split('=') for m in a.maps.split(',')]
    os.makedirs(a.out_dir, exist_ok=True)
    with Pool(min(len(maps), 8)) as p:
        out = p.map(score_map, maps)
    flags = gr.read_flags(FLAGS)
    nonroute = {i for i, r in flags.items() if r['class'] in ('b', 'd')}
    for name, res in out:
        write_csv('%s/%s.csv' % (a.out_dir, name), res)
        print(fmt(stat_line(res, name)))
        print(fmt(stat_line([r for r in res if r['id'] not in nonroute], name + '(excl b/d)')))


# ------------------------------------------------------------ combination
class Gauge:
    """Per-gauge cached geometry for fast re-evaluation on edited maps."""

    def __init__(self, g, row, shp):
        self.id = int(row['clean_id'])
        self.lat, self.lon = float(row['lat']), float(row['lng'])
        self.rings = gr.read_polygon(shp[self.id])
        self.pa = gr.polygon_area_km2(self.rings)
        self.r0 = int(round((self.lat - g.lat[0]) / (g.lat[1] - g.lat[0])))
        self.c0 = int(round((self.lon - g.lon[0]) / (g.lon[1] - g.lon[0])))
        self.valid = 0 <= self.r0 < g.ny and 0 <= self.c0 < g.nx
        self.small = self.valid and self.pa < g.carea[self.r0, self.c0]
        if self.valid:
            self.bbox = gr.poly_bbox(g, self.rings)
            self.frac = gr.polygon_fraction(self.rings, g.lat, g.lon, *self.bbox)
            self.region = self.frac >= 0.5


def evaluate(g, G, radius=3):
    """Best snapped cell for gauge G on grid g: (score, iou, cell_r, cell_c, grid_km2)."""
    best = None
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = G.r0 + dr, G.c0 + dc
            if not (0 <= r < g.ny and 0 <= c < g.nx and g.mask[r, c]):
                continue
            ar = g.up[r, c]
            lr = min(abs(np.log(ar / G.pa)), 1.0)
            if G.small:
                iou, sc = np.nan, -lr
            else:
                iou = gr.iou_at(g, G.rings, G.bbox, G.frac, r, c)
                sc = iou - gr.W_AREA * lr
            sc -= 1e-4 * np.hypot(dr, dc)
            if best is None or sc > best[0]:
                best = (sc, iou, r, c, ar)
    return best


def _regrid(g0, fd):
    import copy
    from scipy.sparse import csr_matrix
    g = copy.copy(g0)
    g.fd = fd
    fdm = np.where(g.mask, fd, 0)
    g.tgt = bg._target(fdm, g.mask)
    n = g.ny * g.nx
    src = np.nonzero(g.tgt >= 0)[0]
    g.rev = csr_matrix((np.ones(len(src)), (g.tgt[src], src)), shape=(n, n))
    g.up = bg.accumulate(fdm, g.mask, g.carea)
    return g


def has_loop(g):
    fdm = np.where(g.mask, g.fd, 0)
    return bool((g.mask & ~bg.drains_to_terminal(fdm, g.mask)).any())


def cmd_combine(a):
    names, files = zip(*[m.split('=') for m in a.maps.split(',')])
    files = dict(zip(names, files))
    grids = {n: gr.Grid(f) for n, f in files.items()}
    rows, shp = gr.load_gauges()
    base = grids[a.base]
    gl = [Gauge(base, r, shp) for r in rows]
    gl = [G for G in gl if G.valid]
    # per-map evaluation of every gauge
    ev = {n: {G.id: evaluate(grids[n], G) for G in gl} for n in names}
    flags = gr.read_flags(FLAGS)
    skip = {i for i, r in flags.items() if r['class'] in ('b', 'd')}      # canal / polygon-domain gauges
    cur = base
    fd = base.fd.copy()
    src = np.full(fd.shape, 1 + list(names).index(a.base), np.uint8)
    cur_ev = dict(ev[a.base])
    fixed = []
    log = []
    order = sorted([G for G in gl if not G.small and G.id not in skip], key=lambda G: -G.pa)
    for G in order:
        sc0 = cur_ev[G.id][0]
        cands = sorted([(ev[n][G.id][0], n) for n in names if n != a.base], reverse=True)
        for scm, n in cands:
            if scm < sc0 + a.margin:
                break
            r0, r1, c0, c1 = G.bbox
            reg = np.zeros(fd.shape, bool)
            reg[r0:r1, c0:c1] = G.region
            reg &= base.mask
            fd2 = fd.copy()
            take = reg & (grids[n].fd != 255)
            fd2[take] = grids[n].fd[take]
            if (fd2 == fd).all():
                continue
            try:
                g2 = _regrid(cur, fd2)
            except RuntimeError:                      # accumulate() found a loop
                g2 = None
            if g2 is None:
                log.append('%d: transplant from %s rejected (loop)' % (G.id, n))
                continue
            new = evaluate(g2, G)
            if new[0] <= sc0 + a.margin / 2 or new[1] < cur_ev[G.id][1] - 0.01:
                continue                              # needs a better score and no IoU loss
            # no overlapping protected gauge (already fixed, or good now) may get worse
            worse = []
            for H in gl:
                if H.id == G.id or H.small:
                    continue
                if not (H.bbox[0] < r1 and H.bbox[1] > r0 and H.bbox[2] < c1 and H.bbox[3] > c0):
                    continue
                if H.id in fixed or cur_ev[H.id][1] >= 0.8:
                    e = evaluate(g2, H)
                    if e[0] < cur_ev[H.id][0] - a.tol:
                        worse.append(H.id)
            if worse:
                log.append('%d: transplant from %s rejected (worsens %s)' % (G.id, n, worse))
                continue
            fd, cur = fd2, g2
            src[take & (fd == grids[n].fd)] = 1 + list(names).index(n)
            for H in gl:
                if H.bbox[0] < r1 and H.bbox[1] > r0 and H.bbox[2] < c1 and H.bbox[3] > c0:
                    cur_ev[H.id] = evaluate(cur, H)
            fixed.append(G.id)
            log.append('%d: %s score %.2f -> %.2f (IoU %.2f -> %.2f), %d cells' % (
                G.id, n, sc0, cur_ev[G.id][0], ev[a.base][G.id][1], cur_ev[G.id][1], int(take.sum())))
            break
    for l in log:
        print(l, flush=True)
    write_combined(a.out, names, a.base, cur, src, fixed)
    print('transplanted gauges:', len(fixed))


def write_combined(path, names, base, g, src, fixed):
    import fdir_sources as fs
    fs.write_nc(path, 'ISC 1/16 deg D8 flow directions: combined multi-source map',
                'base %s; per-gauge polygon transplants from %s; tools/fdir_multi.py; transplanted gauge ids: %s' % (
                    base, ','.join(names), ' '.join(map(str, fixed))),
                g.lat, g.lon, g.mask, g.fd, src, g.up,
                'source index: ' + ' '.join('%d=%s' % (i + 1, n) for i, n in enumerate(names)))


# ------------------------------------------------- per-gauge cell overrides
OVERRIDES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'isc_fdir_overrides_v2.csv')
EDIT_CAP = 14          # max direction edits per gauge
UPHILL_M = 20.0        # a reach may not be sent uphill by more than this (cell mean elevation)


class Editor:
    """Fast single-cell direction edits on a D8 tree (catchment by set algebra).

    A direction change of cell e moves the whole sub-tree of e: for a gauge cell X,
    catch'(X) = catch(X) - sub(e), plus sub(e) when the new downstream path of e reaches X."""

    def __init__(self, g, G, X):
        self.g, self.G = g, G
        self.r0, self.r1, self.c0, self.c1 = G.bbox
        self.X = X
        self.sub = {}
        idx = g.catchment(*X)
        self.base = np.zeros(g.ny * g.nx, bool)
        self.base[idx] = True
        self.w = g.carea.ravel()
        self.inbb = np.zeros((g.ny, g.nx), bool)
        self.inbb[self.r0:self.r1, self.c0:self.c1] = True
        self.inbb = self.inbb.ravel()
        self.fr = G.frac
        self.wb = g.carea[self.r0:self.r1, self.c0:self.c1]

    def subtree(self, e):
        if e not in self.sub:
            self.sub[e] = self.g.catchment(e // self.g.nx, e % self.g.nx)
        return self.sub[e]

    def score(self, base_mask):
        g = self.g
        bb = base_mask.reshape(g.ny, g.nx)[self.r0:self.r1, self.c0:self.c1]
        out = (self.w * (base_mask & ~self.inbb)).sum()
        inter = (np.minimum(bb, self.fr) * self.wb).sum()
        union = (np.maximum(bb, self.fr) * self.wb).sum() + out
        area = (self.w * base_mask).sum()
        iou = inter / union if union > 0 else 0.0
        return iou - gr.W_AREA * min(abs(np.log(max(area, 1e-9) / self.G.pa)), 1.0), iou, area

    def path_hits(self, e, t):
        """Downstream cells of e after sending it to t (None if that makes a loop)."""
        g = self.g
        path = [t]
        cur = t
        while True:
            if cur == e:
                return None
            nxt = g.tgt[cur]
            if nxt < 0 or len(path) > 5000:
                return set(path)
            path.append(int(nxt))
            cur = int(nxt)

    def trial(self, e, t):
        """Score of the catchment of X if cell e is sent to cell t."""
        hit = self.path_hits(e, t)
        if hit is None:
            return None
        sub = self.subtree(e)
        m = self.base.copy()
        if self.base[e]:
            m[sub] = False                   # e's sub-tree leaves X's catchment ...
        if self.X[0] * self.g.nx + self.X[1] in hit or (self.X == divmod(e, self.g.nx)):
            m[sub] = True                    # ... and joins it again if the new path reaches X
        return self.score(m)


def optimize_gauge(g, G, others, skip_pairs=(), log=print):
    """Greedy single-cell direction edits that raise IoU - W_AREA*|ln area ratio| of the gauge's
    catchment with its polygon, subject to: no loop, no uphill beyond UPHILL_M, and no
    overlapping already-good gauge losing more than `tol` score."""
    import netCDF4
    elev = None
    edits = []
    banned = set()
    cur = g
    ev = evaluate(cur, G)
    for _ in range(EDIT_CAP):
        X = (ev[2], ev[3])
        E = Editor(cur, G, X)
        sc0, iou0, a0 = E.score(E.base)
        r0, r1, c0, c1 = G.bbox
        best = None
        for r in range(r0, r1):
            for c in range(c0, c1):
                e = r * cur.nx + c
                if not cur.mask[r, c] or cur.fd[r, c] == 255:
                    continue
                for code, (dr, dc) in bg.ESRI.items():
                    rr, cc = r + dr, c + dc
                    if code == cur.fd[r, c] or not (0 <= rr < cur.ny and 0 <= cc < cur.nx) or not cur.mask[rr, cc]:
                        continue
                    if (e, code) in banned or (r, c) in {(a, b) for a, b, _, _ in edits}:
                        continue
                    if cur.elev[rr, cc] > cur.elev[r, c] + UPHILL_M:
                        continue
                    res = E.trial(e, rr * cur.nx + cc)
                    if res is None:
                        continue
                    if best is None or res[0] > best[0]:
                        best = (res[0], res[1], res[2], r, c, code)
        if best is None or best[0] < sc0 + 0.005:
            break
        sc, iou, area, r, c, code = best
        fd2 = cur.fd.copy()
        fd2[r, c] = code
        try:
            g2 = _regrid(cur, fd2)
        except RuntimeError:
            banned.add((r * cur.nx + c, code)); continue
        bad = [H.id for H in others if H.id != G.id and not H.small and
               H.bbox[0] < r1 and H.bbox[1] > r0 and H.bbox[2] < c1 and H.bbox[3] > c0 and
               evaluate(g2, H)[0] < evaluate(cur, H)[0] - 0.02 and evaluate(cur, H)[1] >= 0.8]
        if bad:
            banned.add((r * cur.nx + c, code))
            log('  %d: edit (%d,%d)->%d rejected, worsens good gauge(s) %s' % (G.id, r, c, code, bad))
            continue
        edits.append((r, c, int(cur.fd[r, c]), code))
        cur, ev = g2, evaluate(g2, G)
        log('  %d: edit (%d,%d) %d->%d  score %.3f IoU %.3f grid %.0f km2 (poly %.0f)' % (
            G.id, r, c, edits[-1][2], code, ev[0], ev[1], ev[4], G.pa))
    return cur, edits, ev


def cmd_optimize(a):
    g = gr.Grid(a.fdir)
    import netCDF4
    g.elev = np.asarray(netCDF4.Dataset(gr.ROOT + '/vic_params_isc/domain.nc')['elev'][:], dtype=np.float64)
    rows, shp = gr.load_gauges()
    ids = [int(x) for x in a.ids.split(',')]
    gl = [Gauge(g, r, shp) for r in rows]
    gl = [G for G in gl if G.valid]
    byid = {G.id: G for G in gl}
    out = []
    for gid in ids:
        G = byid[gid]
        before = evaluate(g, G)
        print('%d %s: before IoU %.3f grid %.0f km2 poly %.0f' % (gid, G.id, before[1], before[4], G.pa), flush=True)
        g, edits, ev = optimize_gauge(g, G, gl)
        for r, c, f, t in edits:
            out.append(dict(gauge_id=gid, row=r, col=c, lat='%.5f' % g.lat[r], lon='%.5f' % g.lon[c], from_code=f, to_code=t,
                            iou_before=round(float(before[1]), 3), iou_after=round(float(ev[1]), 3),
                            area_before=round(float(before[4])), area_after=round(float(ev[4])), poly_km2=round(G.pa)))
        print('%d: %d edits, IoU %.3f -> %.3f, area %.0f -> %.0f (poly %.0f)' % (
            gid, len(edits), before[1], ev[1], before[4], ev[4], G.pa), flush=True)
    if out:
        write_csv(a.out, out)


def read_overrides(path=OVERRIDES):
    return [r for r in csv.DictReader(open(path))] if os.path.exists(path) else []


def cmd_apply(a):
    """Apply the per-cell overrides (by lat/lon) to a map; refuses loops; writes a new map file."""
    import netCDF4
    import fdir_sources as fs
    g = gr.Grid(a.fdir)
    rows = read_overrides(a.overrides)
    fd = g.fd.copy()
    n = 0
    for r in rows:
        i = int(round((float(r['lat']) - g.lat[0]) / (g.lat[1] - g.lat[0])))
        j = int(round((float(r['lon']) - g.lon[0]) / (g.lon[1] - g.lon[0])))
        if not (0 <= i < g.ny and 0 <= j < g.nx and g.mask[i, j]):
            print('skip (outside mask): ', r); continue
        if fd[i, j] != int(r['from_code']) and fd[i, j] != int(r['to_code']):
            print('WARNING: cell (%d,%d) has code %d, table expects %s -> %s' % (i, j, fd[i, j], r['from_code'], r['to_code']))
        fd[i, j] = int(r['to_code']); n += 1
    g2 = _regrid(g, fd)
    if has_loop(g2):
        raise SystemExit('overrides create a loop')
    src = np.asarray(netCDF4.Dataset(a.fdir)['source'][:])
    for r in rows:
        i = int(round((float(r['lat']) - g.lat[0]) / (g.lat[1] - g.lat[0])))
        j = int(round((float(r['lon']) - g.lon[0]) / (g.lon[1] - g.lon[0])))
        if 0 <= i < g.ny and 0 <= j < g.nx and g.mask[i, j]:
            src[i, j] = 9
    fs.write_nc(a.out, 'ISC 1/16 deg D8 flow directions: combined multi-source map + per-gauge cell overrides',
                'base %s; %d cell overrides from %s (tools/fdir_multi.py apply); gauges: %s' % (
                    a.fdir, n, a.overrides, ' '.join(sorted({r['gauge_id'] for r in rows}, key=int))),
                g2.lat, g2.lon, g2.mask, g2.fd, src, g2.up,
                'source index as in the base map; 9=per-gauge override')
    print('applied %d cell overrides -> %s' % (n, a.out))


# ------------------------------------------------------- out-of-mask cells
def cmd_extend(a):
    g = gr.Grid(a.fdir)
    rows, shp = gr.load_gauges()
    dlat, dlon = g.lat[1] - g.lat[0], g.lon[1] - g.lon[0]
    os.makedirs(a.out_dir, exist_ok=True)
    allcells = {}
    out = []
    for row in rows:
        gid = int(row['clean_id'])
        rings = gr.read_polygon(shp[gid])
        lo = np.min([r.min(0) for r in rings], 0)
        hi = np.max([r.max(0) for r in rings], 0)
        # virtual grid extended beyond the ISC domain (same lattice)
        r0 = int(np.floor((lo[1] - g.lat[0]) / dlat + 0.5)) - 1
        r1 = int(np.ceil((hi[1] - g.lat[0]) / dlat + 0.5)) + 2
        c0 = int(np.floor((lo[0] - g.lon[0]) / dlon + 0.5)) - 1
        c1 = int(np.ceil((hi[0] - g.lon[0]) / dlon + 0.5)) + 2
        lat = g.lat[0] + dlat * np.arange(r0, r1)
        lon = g.lon[0] + dlon * np.arange(c0, c1)
        frac = gr.polygon_fraction(rings, lat, lon, 0, len(lat), 0, len(lon))
        rr, cc = np.nonzero(frac >= 0.5)
        rr, cc = rr + r0, cc + c0
        inm = (rr >= 0) & (rr < g.ny) & (cc >= 0) & (cc < g.nx)
        inm[inm] = g.mask[rr[inm], cc[inm]]
        n_in, n_out = int(inm.sum()), int((~inm).sum())
        # cell area ~ equal-area correction by latitude
        ar = g.carea[g.ny // 2, 0] * np.cos(np.radians(g.lat[0] + dlat * rr)) / np.cos(np.radians(g.lat[g.ny // 2]))
        out_km2 = float(ar[~inm].sum())
        if n_out:
            for r, c in zip(rr[~inm], cc[~inm]):
                allcells.setdefault((int(r), int(c)), []).append(gid)
        out.append(dict(id=gid, name=row['name'], poly_km2=gr.polygon_area_km2(rings), cells_in_mask=n_in,
                        cells_outside=n_out, outside_km2=out_km2,
                        outside_frac=n_out / max(n_in + n_out, 1)))
    write_csv(a.out_dir + '/out_of_mask_by_gauge.csv', out)
    sel = [o for o in out if o['cells_outside'] >= 1]
    print('gauges with >=1 polygon cell (frac>=.5) outside the mask: %d' % len(sel))
    for o in sorted(sel, key=lambda o: -o['cells_outside'])[:25]:
        print('  %d %-28s outside %4d of %5d cells (%.0f km2, %.0f%%)' % (
            o['id'], o['name'][:28], o['cells_outside'], o['cells_in_mask'] + o['cells_outside'],
            o['outside_km2'], 100 * o['outside_frac']))
    sig = {o['id'] for o in out if o['outside_frac'] >= 0.02 and o['cells_outside'] >= 3}
    cells = {k: v for k, v in allcells.items() if set(v) & sig}
    print('significant gauges (>=3 cells and >=2%% outside): %d' % len(sig))
    print('union of cells to add for them: %d (all gauges with any outside cell: %d)' % (len(cells), len(allcells)))
    with open(a.out_dir + '/cells_to_add.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row', 'col', 'lat', 'lon', 'gauge_ids'])
        for (r, c), ids in sorted(cells.items()):
            w.writerow([r, c, '%.5f' % (g.lat[0] + dlat * r), '%.5f' % (g.lon[0] + dlon * c), ' '.join(map(str, sorted(set(ids) & sig)))])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    ext = [g.lon[0] - dlon / 2, g.lon[-1] + dlon / 2, g.lat[0] - dlat / 2, g.lat[-1] + dlat / 2]
    ax.imshow(np.where(g.mask, 1.0, np.nan), origin='lower', extent=ext, cmap='Greys', vmin=0, vmax=3)
    if cells:
        la = np.array([g.lat[0] + dlat * r for r, c in cells])
        lo_ = np.array([g.lon[0] + dlon * c for r, c in cells])
        ax.scatter(lo_, la, s=4, c='red', marker='s', label='cells to add (%d)' % len(cells))
    for o in out:
        if o['id'] in sig:
            ax.plot(*[float(row[k]) for row in rows if int(row['clean_id']) == o['id'] for k in ('lng', 'lat')],
                    'b*', ms=8)
    ax.legend(loc='upper right')
    ax.set_title('ISC mask (grey) and polygon cells outside it for %d gauges (blue stars)' % len(sig))
    fig.savefig(a.out_dir + '/out_of_mask.png', dpi=120, bbox_inches='tight')


# ------------------------------------------------------------------ canals
def cmd_canals(a):
    g = gr.Grid(a.fdir)
    rows, shp = gr.load_gauges()
    # Madhopur headworks on the Ravi: the Ravi cell next to the Gurdaspur gauges whose upstream area
    # matches the Ravi catchment above the headworks (polygon of 480/484, 6 264 km2)
    lat_m, lon_m, a_ravi = 32.36, 75.60, 6264.0
    r0 = int(round((lat_m - g.lat[0]) / (g.lat[1] - g.lat[0])))
    c0 = int(round((lon_m - g.lon[0]) / (g.lon[1] - g.lon[0])))
    best = min((abs(np.log(g.up[r0 + dr, c0 + dc] / a_ravi)), r0 + dr, c0 + dc) for dr in range(-3, 4)
               for dc in range(-3, 4) if g.mask[r0 + dr, c0 + dc])
    _, rs, cs = best
    out = []
    for row in rows:
        gid = int(row['clean_id'])
        if gid in (478, 480, 482, 484):
            rg = int(round((float(row['lat']) - g.lat[0]) / (g.lat[1] - g.lat[0])))
            cg = int(round((float(row['lng']) - g.lon[0]) / (g.lon[1] - g.lon[0])))
            out.append(dict(gauge=gid, name=row['name'], lat=row['lat'], lng=row['lng'],
                            gauge_cell_row=rg, gauge_cell_col=cg, gauge_cell_up_km2=round(g.up[rg, cg]),
                            poly_km2=round(gr.polygon_area_km2(gr.read_polygon(shp[gid])), 1),
                            source_river='Ravi at Madhopur headworks (UBDC offtake)',
                            source_cell_row=rs, source_cell_col=cs,
                            source_cell_lat=round(g.lat[rs], 5), source_cell_lon=round(g.lon[cs], 5),
                            source_cell_up_km2=round(g.up[rs, cs]),
                            treatment='canal: needs diversion model (future water-use work); not a natural catchment'))
    write_csv(a.out, out)
    for o in out:
        print(o['gauge'], o['name'], 'gauge-cell up %s km2, poly %s km2 -> source cell (%d,%d) up %s km2' % (
            o['gauge_cell_up_km2'], o['poly_km2'], rs, cs, o['source_cell_up_km2']))


def cmd_final(a):
    """Per-gauge table for the final map.  Treatment per gauge: natural (routed along the D8 network),
    subcell (polygon < gauge_review.SUBCELL_CELLS cells or table treatment 'subcell': area-weighted
    cell runoff), canal (regulated diversion, not routed), or not_routable (no source resolves it)."""
    g = gr.Grid(a.fdir)
    rows, shp = gr.load_gauges()
    res = [r for r in gr.snap_all(g, rows, shp) if r['ok']]
    cor = gr.read_corrections()
    out = []
    for r in res:
        err = (r['grid_km2'] - r['poly_km2']) / r['poly_km2'] * 100
        c = cor.get(r['id'], {})
        cells = r['poly_km2'] / g.carea[r['cell_r'], r['cell_c']]
        bad = (not r['small']) and (abs(err) > 20 or r['iou'] < 0.5)
        if c.get('treatment') == 'canal':
            treat, why = 'canal', c['evidence']
        elif c.get('treatment') == 'subcell' or cells < gr.SUBCELL_CELLS:
            treat, why = 'subcell', 'polygon %.1f cells: area-weighted polygon-cell runoff' % cells + (
                ('; ' + c['evidence']) if c else '')
        elif bad:
            ex, fin, n = diagnose_short(g, r, shp)
            treat, why = 'not_routable', 'no source matches; ' + ex
        else:
            treat, why = 'natural', c.get('evidence', '')
        out.append(dict(id=r['id'], name=r['name'], poly_km2=round(r['poly_km2']), grid_km2=round(r['grid_km2']),
                        err_pct=round(err, 1), iou=round(r['iou'], 3) if r['iou'] == r['iou'] else '',
                        poly_cells=round(cells, 2), treatment=treat, reason=why))
    write_csv(a.out, out)
    n = {k: sum(o['treatment'] == k for o in out) for k in ('natural', 'subcell', 'canal', 'not_routable')}
    print('gauges %d: %s' % (len(out), n))
    nat = [o for o in out if o['treatment'] == 'natural']
    ae = np.abs([o['err_pct'] for o in nat])
    iou = np.array([o['iou'] for o in nat], float)
    print('natural catchments: median|err| %.2f%% p90 %.2f%% within10 %d within20 %d meanIoU %.3f' % (
        np.median(ae), np.percentile(ae, 90), (ae <= 10).sum(), (ae <= 20).sum(), np.nanmean(iou)))
    for o in out:
        if o['treatment'] == 'not_routable':
            print(o['id'], o['name'], o['poly_km2'], o['grid_km2'], o['err_pct'], o['iou'], '|', o['reason'][:170])


def diagnose_short(g, rec, shp):
    ex, fin, n = gr.diagnose(g, rec, shp, top=2)
    return 'exits %s; foreign inflows %s' % (ex, fin), fin, n


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('score'); s.add_argument('--maps', required=True); s.add_argument('--out-dir', required=True)
    c = sp.add_parser('combine'); c.add_argument('--maps', required=True); c.add_argument('--base', required=True)
    c.add_argument('--out', required=True); c.add_argument('--margin', type=float, default=0.1)
    c.add_argument('--tol', type=float, default=0.02)
    e = sp.add_parser('extend'); e.add_argument('--fdir', required=True); e.add_argument('--out-dir', required=True)
    f = sp.add_parser('final'); f.add_argument('--fdir', required=True); f.add_argument('--out', required=True)
    o = sp.add_parser('optimize'); o.add_argument('--fdir', required=True); o.add_argument('--ids', required=True); o.add_argument('--out', required=True)
    ap_ = sp.add_parser('apply'); ap_.add_argument('--fdir', required=True); ap_.add_argument('--out', required=True); ap_.add_argument('--overrides', default=OVERRIDES)
    k = sp.add_parser('canals'); k.add_argument('--fdir', required=True); k.add_argument('--out', required=True)
    a = ap.parse_args()
    {'score': cmd_score, 'combine': cmd_combine, 'extend': cmd_extend, 'canals': cmd_canals, 'final': cmd_final, 'optimize': cmd_optimize, 'apply': cmd_apply}[a.cmd](a)


if __name__ == '__main__':
    main()
