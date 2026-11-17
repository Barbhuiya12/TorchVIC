#!/usr/bin/env python3
"""ISC 1/16-degree D8 flow directions from GRIT v0.6 segments (Wortmann et al.).

  build_isc_fdir_grit.py --gpkg GRITv06_segments_AS_EPSG4326.gpkg \
      --domain domain.nc --out isc_fdir_grit.nc [--gauges basin_index.csv
      --val-csv validation.csv --fig network.png]

Method (IHU/COTAT-style outlet rule on the vector network):
 1. Every GRIT segment in the domain is densified (~1/4 cell) and carries a
    drainage area interpolated between drainage_area_in and _out (km2).
 2. Each cell takes the densified point with the largest area (the main
    reach in the cell) and follows the network downstream (at a bifurcation
    the branch with the larger drainage_area_in) until the path leaves the
    cell; the D8 code points to the cell it enters.  A path that ends inside
    a cell goes to the closest off-mask neighbour (mouth) or, when there is
    none, the cell is a sink (code 0).
 3. Loops in the cell graph are broken at their weakest cell; cells without a
    GRIT reach (and the broken ones) get a direction by priority flood of the
    elevation, flooding outward from the cells already drained by GRIT.
 4. Upstream area is accumulated on the final grid.
Data: GRIT is CC BY-NC 4.0.  The gpkg is read as untrusted data (sqlite
read-only, plain WKB parsing, nothing executed).
"""
import argparse
import heapq
import sqlite3
import struct

import numpy as np

ESRI = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
CODE = {v: k for k, v in ESRI.items()}
SINK = 0
END_TYPE = {'coastal_outlet': 1, 'outlet': 1, 'interbasin': 1, 'sink_outlet': 2}


# ----------------------------------------------------------------- reading
def _wkb_line(b):
    """Coordinates [n,2] of a GeoPackage LINESTRING blob (little endian WKB)."""
    flags = b[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    o = 8 + env
    assert b[o] == 1, 'big-endian WKB not expected'
    gt = struct.unpack_from('<I', b, o + 1)[0]
    assert gt % 1000 == 2, 'not a LineString'
    n = struct.unpack_from('<I', b, o + 5)[0]
    dim = 2 + (gt // 1000 in (1, 3)) + (gt // 1000 in (2, 3))
    a = np.frombuffer(b, dtype='<f8', count=n * dim, offset=o + 9)
    return a.reshape(n, dim)[:, :2]


def read_segments(gpkg, bbox):
    """List of dicts for lines whose envelope meets bbox=(w, s, e, n)."""
    w, s, e, n = bbox
    c = sqlite3.connect('file:%s?mode=ro' % gpkg, uri=True)
    q = ('select l.global_id, l.geom, l.downstream_line_ids, l.drainage_area_in,'
         ' l.drainage_area_out, n.node_type from lines l join rtree_lines_geom r on r.id=l.fid'
         ' left join nodes n on n.global_id=l.downstream_node_id'
         ' where r.maxx>=? and r.minx<=? and r.maxy>=? and r.miny<=?')
    out = []
    for gid, g, dn, ain, aout, nt in c.execute(q, (w, e, s, n)):
        xy = _wkb_line(g)
        dl = [int(x) for x in dn.split(',')] if dn else []
        out.append(dict(gid=gid, xy=xy, down=dl,
                        ain=ain if ain is not None else 0.0,
                        aout=aout if aout is not None else (ain or 0.0),
                        end=END_TYPE.get(nt, 0)))
    return out


# --------------------------------------------------------------- the method
def densify(segs, step):
    """Point arrays: x, y, area (km2), seg index, and `nxt` (next point along
    the chosen downstream path, -1 at a network end)."""
    xs, ys, ar, sid, first, count, endt = [], [], [], [], [], [], []
    pos = 0
    for k, sg in enumerate(segs):
        p = sg['xy']
        d = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(d)])
        L = cum[-1]
        m = max(int(np.ceil(L / step)), 1) + 1
        t = np.linspace(0.0, L, m)
        x = np.interp(t, cum, p[:, 0])
        y = np.interp(t, cum, p[:, 1])
        f = t / L if L > 0 else np.zeros(m)
        xs.append(x); ys.append(y)
        ar.append(sg['ain'] + f * (sg['aout'] - sg['ain']))
        sid.append(np.full(m, k))
        et = np.zeros(m, dtype=np.int8); et[-1] = sg['end']; endt.append(et)
        first.append(pos); count.append(m); pos += m
    x = np.concatenate(xs); y = np.concatenate(ys)
    a = np.concatenate(ar); sid = np.concatenate(sid)
    first = np.array(first); count = np.array(count)
    gid2k = {sg['gid']: k for k, sg in enumerate(segs)}
    nxt = np.arange(1, pos + 1)
    last = first + count - 1
    for k, sg in enumerate(segs):
        cand = [gid2k[g] for g in sg['down'] if g in gid2k]
        # main branch: larger drainage area at its upstream end
        nxt[last[k]] = (first[max(cand, key=lambda j: (segs[j]['ain'], segs[j]['aout']))]
                        if cand else -1)
    return x, y, a, sid, nxt, np.concatenate(endt)


def cell_index(x, y, lat, lon):
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    r = np.rint((y - lat[0]) / dlat).astype(np.int64)
    c = np.rint((x - lon[0]) / dlon).astype(np.int64)
    return r, c


def grit_directions(x, y, a, nxt, endt, lat, lon, mask, max_walk=4000):
    """down[r,c] = D8 code (>0), SINK (0) or -1 (no GRIT reach in the cell);
    amax[r,c] = largest GRIT area in the cell (km2)."""
    ny, nx = mask.shape
    r, c = cell_index(x, y, lat, lon)
    inside = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
    cid = np.where(inside, r * nx + c, -1)
    # only points inside the grid count as a cell's main reach; mask cells only
    flat_mask = mask.ravel()
    ok = inside & flat_mask[np.where(inside, cid, 0)]
    order = np.lexsort((-a[ok], cid[ok]))          # by cell, largest area first
    idx_ok = np.nonzero(ok)[0][order]
    cells, first = np.unique(cid[idx_ok], return_index=True)
    p0 = idx_ok[first]
    amax = np.zeros(ny * nx)
    amax[cells] = a[p0]
    down = -np.ones(ny * nx, dtype=np.int64)
    # walk downstream until the cell changes
    p = p0.copy()
    cur = cells.copy()
    res_r = np.full(len(cells), 9)                 # 9 = unresolved
    res_c = np.zeros(len(cells), dtype=np.int64)
    ended_at = np.zeros(len(cells), dtype=np.int64)
    act = np.arange(len(cells))
    for _ in range(max_walk):
        if len(act) == 0:
            break
        q = nxt[p[act]]
        end = q < 0
        if end.any():
            ended_at[act[end]] = p[act[end]]
            res_r[act[end]] = 8                    # path ended inside the cell
        go = ~end
        act, q = act[go], q[go]
        newcell = cid[q] != cur[act]
        mv = act[newcell]
        qm = q[newcell]
        nr, nc = r[qm], c[qm]
        res_r[mv] = np.sign(nr - cells[mv] // nx)
        res_c[mv] = np.sign(nc - cells[mv] % nx)
        p[act] = q
        act = act[~newcell]
    # a path that never left (cycle / cap) is treated like an ended path
    ended_at[act] = p[act]
    res_r[act] = 8
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) == (0, 0):
                continue
            m = (res_r == dr) & (res_c == dc)
            down[cells[m]] = CODE[(dr, dc)]
    # paths ended inside the cell: a coastal/outlet node -> the nearest
    # off-mask neighbour (else sink); a GRIT sink node -> sink; an open end -> left to the DEM
    for k in np.nonzero(res_r == 8)[0]:
        cc = cells[k]
        r0, c0 = divmod(int(cc), nx)
        e = ended_at[k]
        et = endt[e]
        if et == 2:
            down[cc] = SINK
        elif et == 1:
            ex, ey = x[e], y[e]
            rr, c2 = np.broadcast_arrays(r0 + np.arange(-1, 2)[:, None],
                                         c0 + np.arange(-1, 2)[None, :])
            inb = (rr >= 0) & (rr < ny) & (c2 >= 0) & (c2 < nx)
            offm = ~inb
            offm[inb] = ~mask[rr[inb], c2[inb]]
            if not offm.any():
                down[cc] = SINK              # coastal end with no ocean neighbour
                continue
            ty = lat[r0] + (rr - r0) * (lat[1] - lat[0])
            tx = lon[c0] + (c2 - c0) * (lon[1] - lon[0])
            dd = np.where(offm, np.hypot(tx - ex, ty - ey), np.inf)
            i = np.unravel_index(np.argmin(dd), dd.shape)
            dr, dc = int(np.sign(i[0] - 1)), int(np.sign(i[1] - 1))
            if (dr, dc) != (0, 0):
                down[cc] = CODE[(dr, dc)]
    return down.reshape(ny, nx), amax.reshape(ny, nx)


def _target(fd, mask):
    """Flat target index per cell: -1 mouth (off-mask / off-grid), -2 sink or
    unassigned, else the downstream cell."""
    ny, nx = mask.shape
    tgt = np.full(ny * nx, -2, dtype=np.int64)
    for code, (dr, dc) in ESRI.items():
        rr, cc = np.nonzero((fd == code) & mask)
        r2, c2 = rr + dr, cc + dc
        inb = (r2 >= 0) & (r2 < ny) & (c2 >= 0) & (c2 < nx)
        t = np.full(len(rr), -1, dtype=np.int64)
        t[inb] = np.where(mask[r2[inb], c2[inb]], r2[inb] * nx + c2[inb], -1)
        tgt[rr * nx + cc] = t
    return tgt


def _jump(fd, mask):
    """Pointer-jump the cell graph: returns (j, term, n) where j[i] is the
    node reached after >= n steps (terminal cells absorb, node n = dead end)."""
    n = mask.size
    tgt = _target(fd, mask)
    term = ((tgt == -1) | ((fd.ravel() == SINK))) & mask.ravel()
    step = np.where(tgt >= 0, tgt, n)
    step = np.where(term, np.arange(n), step)
    step = np.concatenate([step, [n]])
    j = step.copy()
    for _ in range(int(np.ceil(np.log2(n))) + 2):
        j = j[j]
    return j[:n], np.concatenate([term, [False]]), n, tgt


def drains_to_terminal(fd, mask):
    """bool [ny,nx]: the cell's path reaches a mouth or a sink (no loop, no
    unassigned cell on the way)."""
    j, term, n, _ = _jump(fd, mask)
    return (term[j] & mask.ravel()).reshape(mask.shape)


def break_cycles(fd, mask, weight):
    """Unassign (fd=-1) the weakest cell (smallest `weight`) of every loop;
    returns the number of loops broken."""
    j, term, n, tgt = _jump(fd, mask)
    members = np.unique(j[(j < n) & ~term[np.minimum(j, n)]])
    seen = set()
    broken = 0
    for s in members:
        if s in seen:
            continue
        cyc, t = [], s
        while t not in seen and t >= 0:
            seen.add(t); cyc.append(t); t = tgt[t]
        w = np.array([weight.ravel()[k] for k in cyc])
        fd.ravel()[cyc[int(np.argmin(w))]] = -1
        broken += 1
    return broken


def priority_flood_fill(fd, mask, elev, seed_ok):
    """Give a direction to every mask cell with fd==-1: flood outward from the
    cells already draining to a terminal (`seed_ok`), lowest elevation first
    (Barnes et al. 2014); each flooded cell drains to the cell that reached it.
    Unassigned coastal/edge cells drain to the outside and seed the flood."""
    ny, nx = mask.shape
    done = seed_ok.copy()
    heap = []
    for r, c in zip(*np.nonzero(seed_ok)):
        heapq.heappush(heap, (elev[r, c], int(r), int(c)))
    for r, c in zip(*np.nonzero(mask & (fd == -1))):
        for code in (1, 4, 16, 64, 2, 8, 32, 128):
            dr, dc = ESRI[code]
            r2, c2 = r + dr, c + dc
            if not (0 <= r2 < ny and 0 <= c2 < nx) or not mask[r2, c2]:
                fd[r, c] = code
                done[r, c] = True
                heapq.heappush(heap, (elev[r, c], int(r), int(c)))
                break
    while heap:
        lev, r, c = heapq.heappop(heap)
        for code, (dr, dc) in ESRI.items():
            r2, c2 = r - dr, c - dc               # neighbour that drains to (r,c)
            if not (0 <= r2 < ny and 0 <= c2 < nx) or not mask[r2, c2] or done[r2, c2]:
                continue
            if fd[r2, c2] != -1:
                continue                          # keep the GRIT direction
            fd[r2, c2] = code
            done[r2, c2] = True
            heapq.heappush(heap, (max(lev, elev[r2, c2]), r2, c2))
    return fd


def accumulate(fd, mask, cellarea_km2):
    """Upstream area (km2, cell itself included) by topological order."""
    ny, nx = mask.shape
    tgt = _target(fd, mask)
    n = ny * nx
    area = np.where(mask.ravel(), cellarea_km2.ravel(), 0.0)
    indeg = np.zeros(n, dtype=np.int64)
    m = tgt >= 0
    np.add.at(indeg, tgt[m], 1)
    stack = list(np.nonzero(mask.ravel() & (indeg == 0))[0])
    done = 0
    while stack:
        i = stack.pop()
        done += 1
        t = tgt[i]
        if t >= 0:
            area[t] += area[i]
            indeg[t] -= 1
            if indeg[t] == 0:
                stack.append(t)
    if done != int(mask.sum()):
        raise RuntimeError('flow graph still has loops (%d of %d cells ordered)'
                           % (done, mask.sum()))
    return area.reshape(ny, nx)


def read_overrides(path):
    """Override table (csv with columns row,col,to_code,...): explicit, documented
    D8 directions for single cells where the GRIT walk contradicts the catchment
    polygons (see <data root>/static/isc_fdir_overrides.csv).  Returns [(r,c,code)]."""
    import csv
    out = []
    for r in csv.DictReader(open(path)):
        code = int(r['to_code'])
        assert code in ESRI, 'bad D8 code in overrides: %r' % r
        out.append((int(r['row']), int(r['col']), code))
    return out


def build(segs, lat, lon, mask, elev, cellarea_m2, step_deg=None, overrides=()):
    """Returns fdir (uint8), source (1 GRIT, 2 DEM), upstream_area km2, info.
    `overrides` [(r,c,code)] replace the direction of those cells in the
    finished map."""
    step = step_deg or abs(lon[1] - lon[0]) / 4
    x, y, a, sid, nxt, endt = densify(segs, step)
    fd, amax = grit_directions(x, y, a, nxt, endt, lat, lon, mask)
    fd = np.where(mask, fd, -1)
    src = np.where(fd >= 0, 1, 2).astype(np.uint8)
    info = {'grit_cells': int(((fd >= 0) & mask).sum())}
    n_broken = 0
    for it in range(100):
        nb = break_cycles(fd, mask, amax)
        n_broken += nb
        good = drains_to_terminal(fd, mask)
        before = int((fd[mask] == -1).sum())
        fd = priority_flood_fill(fd, mask, elev, good)
        if nb == 0 and (before == 0 or int((fd[mask] == -1).sum()) == before):
            break
    left = mask & (fd == -1)
    fd[left] = SINK
    # explicit single-cell overrides last, on the finished map (a direction that
    # closes a loop or leaves the mask is an error in the table)
    n_ovr = 0
    for r, c, code in overrides:
        assert mask[r, c], 'override cell (%d,%d) is outside the mask' % (r, c)
        n_ovr += int(fd[r, c] != code)
        fd[r, c] = code
    if len(overrides):
        bad = mask & ~drains_to_terminal(np.where(mask, fd, SINK), mask)
        if bad.any():
            raise SystemExit('overrides create %d cells that do not drain (loop)' % bad.sum())
    info['overrides_applied'] = n_ovr
    info['cycles_broken'] = n_broken
    info['unreached_to_sink'] = int(left.sum())
    up = accumulate(np.where(mask, fd, SINK), mask, cellarea_m2 / 1e6)
    info['sinks'] = int(((fd == SINK) & mask).sum())
    return np.where(mask, fd, 255).astype(np.uint8), src, up, amax, info


# ---------------------------------------------------------------- validation
def validate(fd, up, amax, mask, lat, lon, gauges_csv, grit_pts=None):
    import csv
    rows = list(csv.DictReader(open(gauges_csv)))
    ny, nx = mask.shape
    out = []
    for g in rows:
        la, lo = float(g['lat']), float(g['lng'])
        ref = None
        for k in ('area_poly_km2', 'poly_area_km2', 'area'):
            try:
                ref = float(g[k])
                if ref > 0:
                    break
            except (KeyError, ValueError):
                pass
        r0 = int(round((la - lat[0]) / (lat[1] - lat[0])))
        c0 = int(round((lo - lon[0]) / (lon[1] - lon[0])))
        if not (0 <= r0 < ny and 0 <= c0 < nx):
            continue
        best, bv = None, np.inf
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < ny and 0 <= c < nx and mask[r, c]:
                    v = abs(np.log(up[r, c] / ref)) + 1e-6 * np.hypot(dr, dc)
                    if v < bv:
                        best, bv = (r, c), v
        if best is None:
            continue
        near = (r0, c0) if mask[r0, c0] else best
        gm = amax[best]
        out.append(dict(id=g['clean_id'], name=g['name'], lat=la, lon=lo, ref_km2=ref,
                        shp_km2=float(g.get('area_poly_km2') or 'nan'),
                        stated_km2=float(g['area']),
                        cell_r=best[0], cell_c=best[1],
                        grid_km2=up[best], grit_cell_km2=gm,
                        nearest_km2=up[near]))
    return out


def stats(vals, ref, label):
    e = (vals - ref) / ref * 100
    ae = np.abs(e)
    return ('%s: n=%d median|err|=%.1f%% p90|err|=%.1f%% within10=%d within20=%d'
            % (label, len(e), np.median(ae), np.percentile(ae, 90),
               int((ae <= 10).sum()), int((ae <= 20).sum())))


# -------------------------------------------------------------------- main
def main():
    import csv
    import netCDF4
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpkg', required=True)
    ap.add_argument('--domain', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gauges')
    ap.add_argument('--val-csv')
    ap.add_argument('--fig')
    ap.add_argument('--overrides', help='csv of single-cell D8 overrides (row,col,to_code,...)')
    ap.add_argument('--snapped-gauges', help='write clean_id,name,lat,lng at the snapped cell centres (for route_isc.py)')
    a = ap.parse_args()
    d = netCDF4.Dataset(a.domain)
    lat, lon = d['lat'][:].data, d['lon'][:].data
    mask = np.asarray(d['mask'][:]) > 0
    elev = np.asarray(d['elev'][:], dtype=np.float64)
    carea = np.asarray(d['area'][:], dtype=np.float64)
    bb = (lon[0] - 0.5, lat[0] - 0.5, lon[-1] + 0.5, lat[-1] + 0.5)
    segs = read_segments(a.gpkg, bb)
    print('segments read', len(segs), flush=True)
    ovr = read_overrides(a.overrides) if a.overrides else ()
    fd, src, up, amax, info = build(segs, lat, lon, mask, elev, carea, overrides=ovr)
    for r, c, code in ovr:
        if fd[r, c] != code:
            raise SystemExit('override (%d,%d) was not kept (loop?)' % (r, c))
    print(info, flush=True)
    o = netCDF4.Dataset(a.out, 'w')
    o.title = 'ISC 1/16 deg D8 flow directions from GRIT v0.6 (CC BY-NC 4.0)'
    o.source = ('GRIT v0.6 segments AS, Wortmann et al., doi:10.5281/zenodo.11219313; '
                'tools/build_isc_fdir_grit.py' + (' + %d cell overrides (%s)' % (len(ovr), a.overrides) if ovr else ''))
    o.comment = 'fdir: ESRI D8 (1=E,2=SE,4=S,8=SW,16=W,32=NW,64=N,128=NE); 0=sink; 255=outside mask'
    o.createDimension('lat', len(lat)); o.createDimension('lon', len(lon))
    for nme, v in (('lat', lat), ('lon', lon)):
        x = o.createVariable(nme, 'f8', (nme,)); x[:] = v
        x.units = 'degrees_north' if nme == 'lat' else 'degrees_east'
    v = o.createVariable('fdir', 'u1', ('lat', 'lon'), zlib=True); v[:] = fd
    v = o.createVariable('upstream_area', 'f8', ('lat', 'lon'), zlib=True); v[:] = np.where(mask, up, np.nan)
    v.units = 'km2'
    v = o.createVariable('source', 'u1', ('lat', 'lon'), zlib=True); v[:] = np.where(mask, src, 0)
    v.flag_values = np.array([0, 1, 2], dtype='u1'); v.flag_meanings = 'outside GRIT DEM'
    v = o.createVariable('grit_area_max', 'f8', ('lat', 'lon'), zlib=True); v[:] = amax
    v.units = 'km2'; v.long_name = 'largest GRIT drainage area of a reach in the cell'
    o.close()
    if a.gauges:
        res = validate(fd, up, amax, mask, lat, lon, a.gauges)
        ref = np.array([r['ref_km2'] for r in res])
        grid = np.array([r['grid_km2'] for r in res])
        print(stats(grid, ref, 'grid vs catchment polygon'))
        gm = np.array([r['grit_cell_km2'] for r in res])
        has = gm > 0
        print(stats(gm[has], ref[has], 'GRIT cell max area vs polygon (cells with GRIT)'))
        nr = np.array([r['nearest_km2'] for r in res])
        print(stats(nr, ref, 'unsnapped nearest cell vs polygon'))
        e = np.abs(grid - ref) / ref
        for i in np.argsort(-e)[:10]:
            r = res[i]
            print('worst %-6s %-22s ref %9.0f grid %9.0f GRITcell %9.0f' %
                  (r['id'], r['name'][:22], r['ref_km2'], r['grid_km2'], r['grit_cell_km2']))
        if a.val_csv:
            with open(a.val_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(res[0]))
                w.writeheader(); w.writerows(res)
    if a.gauges and a.snapped_gauges:
        with open(a.snapped_gauges, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['clean_id', 'name', 'lat', 'lng'])
            for r in res:
                w.writerow([r['id'], r['name'], lat[r['cell_r']], lon[r['cell_c']]])
    if a.fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('#dfe8f0')
        ax.imshow(np.where(mask, 1.0, np.nan), extent=[lon[0], lon[-1], lat[0], lat[-1]],
                  origin='lower', cmap='Greys', vmin=0, vmax=3, zorder=0)
        tgt = _target(np.where(mask, fd, SINK), mask)
        ny, nx = mask.shape
        ii = np.nonzero((tgt >= 0) & (up.ravel() > 5000))[0]
        r1, c1 = divmod(ii, nx); r2, c2 = divmod(tgt[ii], nx)
        from matplotlib.collections import LineCollection
        segsl = np.stack([np.stack([lon[c1], lat[r1]], 1), np.stack([lon[c2], lat[r2]], 1)], 1)
        lw = 0.3 + 1.2 * np.log10(up.ravel()[ii] / 5000.0)
        ax.add_collection(LineCollection(segsl, linewidths=lw, colors='#1f5fa8'))
        ax.set_xlim(lon[0], lon[-1]); ax.set_ylim(lat[0], lat[-1])
        ax.set_title('ISC D8 network from GRIT (cells with upstream area > 5000 km2)')
        ax.set_xlabel('lon'); ax.set_ylabel('lat')
        fig.savefig(a.fig, dpi=130, bbox_inches='tight')


if __name__ == '__main__':
    main()
