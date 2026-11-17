"""DEM conditioning for the flood engine.

A one-cell pit tens of metres deep (a radar artefact of the 30 m DEMs; the
Guwahati GLO-30 box has one of 70 m) fills with water, and the CFL step
dt = cfl dx / sqrt(g h) of the whole grid then follows that pit. These
tools remove such pits without touching the rest of the terrain:

    fill_depressions(z)     priority-flood fill (Barnes et al. 2014), every
                            cell drains to an open edge; optional epsilon
                            gradient on the filled flats
    breach_depressions(z)   carving: the priority-flood path from each pit
                            to its outlet is lowered instead (Lindsay 2016,
                            complete breaching)
    condition(z, ...)       fill or carve only the depressions deeper than
                            `min_depth` (real floodplain hollows are kept)
    burn_rivers(z, ...)     lowers the DEM along the river lines of a D8
                            flow map (our <data root>/static/isc_fdir.nc):
                            each coarse river cell's lowest DEM cell is
                            joined to its downstream one by a least-cost
                            path through the DEM, and the path is made
                            non-increasing downstream

All functions take and return numpy float64 arrays [ny, nx], row 0 north.
"""
import heapq

import numpy as np

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _outlets(shape, edges):
    ny, nx = shape
    m = np.zeros(shape, bool)
    for e in edges:
        if e == 'N':
            m[0] = True
        elif e == 'S':
            m[-1] = True
        elif e == 'W':
            m[:, 0] = True
        elif e == 'E':
            m[:, -1] = True
        else:
            raise ValueError('edge %r not in N/E/S/W' % (e,))
    return m


def _priority_flood(z, edges='NESW', valid=None):
    """Fill level and the parent (toward the outlet) of every cell.

    The flood starts from the open edges (and from cells next to invalid
    cells); a cell's level is max(own elevation, level of the cell it was
    reached from). Returns (level, parent flat index, processing order)."""
    z = np.asarray(z, dtype=np.float64)
    ny, nx = z.shape
    valid = np.ones(z.shape, bool) if valid is None else np.asarray(valid)
    seed = _outlets(z.shape, edges) & valid
    if not seed.any():
        raise ValueError('no outlet cell: give at least one open edge')
    level = np.where(valid, np.inf, np.nan)
    parent = np.full(ny * nx, -1, dtype=np.int64)
    done = ~valid.copy()
    heap = []
    n = 0                       # FIFO among equal levels: breadth-first
    for r, c in zip(*np.nonzero(seed)):   # paths across flats
        level[r, c] = z[r, c]
        heap.append((z[r, c], n, r * nx + c))
        n += 1
        done[r, c] = True
    heapq.heapify(heap)
    order = []
    zf = z.ravel()
    lv = level.ravel()
    dn = done.ravel()
    while heap:
        l0, _, k = heapq.heappop(heap)
        order.append(k)
        r, c = divmod(k, nx)
        for dr, dc in _N8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx:
                kk = rr * nx + cc
                if not dn[kk]:
                    dn[kk] = True
                    v = zf[kk] if zf[kk] > l0 else l0
                    lv[kk] = v
                    parent[kk] = k
                    heapq.heappush(heap, (v, n, kk))
                    n += 1
    return lv.reshape(ny, nx), parent, np.array(order, dtype=np.int64)


def fill_depressions(z, edges='NESW', epsilon=0.0, valid=None):
    """Priority-flood fill. epsilon > 0 adds that rise per cell along the
    flood paths across filled flats, so they drain (m per cell)."""
    z = np.asarray(z, dtype=np.float64)
    level, parent, order = _priority_flood(z, edges, valid)
    if epsilon <= 0:
        return level
    out = level.ravel().copy()
    zf = z.ravel()
    for k in order:
        p = parent[k]
        if p >= 0 and out[k] <= out[p] + epsilon and zf[k] <= out[p]:
            out[k] = out[p] + epsilon
    return out.reshape(z.shape)


def breach_depressions(z, edges='NESW', epsilon=0.0, valid=None):
    """Complete breaching: every cell's path to the outlet (the priority-
    flood tree) is lowered so that it never rises toward the outlet; a cell
    on the path is lowered to the lowest cell upstream of it on that path
    (minus epsilon per cell)."""
    z = np.asarray(z, dtype=np.float64)
    level, parent, order = _priority_flood(z, edges, valid)
    out = z.ravel().copy()
    # children before parents: walk the processing order backwards
    for k in order[::-1]:
        p = parent[k]
        if p >= 0 and out[p] > out[k] - epsilon:
            out[p] = out[k] - epsilon
    return out.reshape(z.shape)


def fill_pits(z):
    """Raise single-cell pits to their lowest 8-neighbour."""
    z = np.asarray(z, dtype=np.float64)
    p = np.pad(z, 1, mode='edge')
    ny, nx = z.shape
    nb = np.stack([p[1 + dr:1 + dr + ny, 1 + dc:1 + dc + nx]
                   for dr, dc in _N8])
    lo = nb.min(axis=0)
    return np.where(z < lo, lo, z)


def condition(z, method='fill', min_depth=0.0, edges='NESW', epsilon=0.0,
              valid=None):
    """Fill ('fill') or carve ('breach') only the depressions whose deepest
    point lies more than `min_depth` below its spill level; shallower
    depressions (floodplain hollows, oxbows) are kept. Returns (dem,
    info) with the number of depressions changed and the largest change."""
    from scipy import ndimage
    z = np.asarray(z, dtype=np.float64)
    full = fill_depressions(z, edges, 0.0, valid)
    depth = np.nan_to_num(full - z)
    lab, n = ndimage.label(depth > 0, structure=np.ones((3, 3)))
    if n == 0:
        return z.copy(), dict(depressions=0, changed=0, max_change=0.0)
    dmax = ndimage.maximum(depth, lab, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], dmax > min_depth])
    sel = keep[lab]
    if method == 'fill':
        new = fill_depressions(z, edges, epsilon, valid)
    elif method == 'breach':
        new = breach_depressions(z, edges, epsilon, valid)
    else:
        raise ValueError('method %r: fill or breach' % (method,))
    if method == 'breach':
        # carving acts on the paths out of the selected depressions, which
        # leave the depression: keep the carve where it serves a selected
        # depression (cells below their original level on a path that
        # starts in one)
        out = _breach_selected(z, sel, edges, epsilon, valid)
    else:
        out = np.where(sel, new, z)
    ch = np.abs(out - z)
    return out, dict(depressions=int(keep.sum()), changed=int((ch > 0).sum()),
                     max_change=float(ch.max()))


def _breach_selected(z, sel, edges, epsilon, valid):
    """Carve only the paths that drain the selected depression cells."""
    level, parent, order = _priority_flood(z, edges, valid)
    out = z.ravel().copy()
    s = sel.ravel()
    carry = np.where(s, out, np.inf)          # lowest selected level carried
    for k in order[::-1]:
        p = parent[k]
        if p < 0:
            continue
        v = carry[k]
        if np.isfinite(v):
            if out[p] > v - epsilon:
                out[p] = v - epsilon
            carry[p] = min(carry[p], out[p])
    return out.reshape(z.shape)


# ---------------------------------------------------------------------------
# river burning from a coarse D8 map
# ---------------------------------------------------------------------------
#: ESRI D8 code -> (d row, d col) in the array of isc_fdir.nc (row index
#: increases with latitude; tools/build_isc_fdir_grit.py)
ESRI = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1),
        32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


def river_cells(fdir_path, box, min_area_km2):
    """Coarse river cells inside box=(lon0, lon1, lat0, lat1): list of
    (lon, lat, downstream lon, downstream lat, upstream area km2) for the
    cells with upstream area >= min_area_km2."""
    import netCDF4
    with netCDF4.Dataset(str(fdir_path)) as d:
        lat = np.asarray(d['lat'][:], float)
        lon = np.asarray(d['lon'][:], float)
        fd = np.asarray(d['fdir'][:])
        ua = np.asarray(d['upstream_area'][:], float)
    lon0, lon1, lat0, lat1 = box
    out = []
    ii = np.nonzero((lat > lat0) & (lat < lat1))[0]
    jj = np.nonzero((lon > lon0) & (lon < lon1))[0]
    for i in ii:
        for j in jj:
            if not (ua[i, j] >= min_area_km2):
                continue
            code = int(fd[i, j])
            if code not in ESRI:
                continue
            di, dj = ESRI[code]
            i2, j2 = i + di, j + dj
            if not (0 <= i2 < len(lat) and 0 <= j2 < len(lon)):
                continue
            out.append((lon[j], lat[i], lon[j2], lat[i2], ua[i, j]))
    return out


def _least_cost_path(z, a, b, pad):
    """Cells of the least-cost 8-connected path a -> b (cost: elevation
    above the lower end + a small length term) inside the bounding box of
    a and b grown by `pad` cells."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    ny, nx = z.shape
    r0, r1 = max(min(a[0], b[0]) - pad, 0), min(max(a[0], b[0]) + pad, ny - 1)
    c0, c1 = max(min(a[1], b[1]) - pad, 0), min(max(a[1], b[1]) + pad, nx - 1)
    sub = z[r0:r1 + 1, c0:c1 + 1]
    h, w = sub.shape
    base = min(z[a], z[b])
    cost = np.maximum(sub - base, 0.0) + 1e-3
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, vals = [], [], []
    for dr, dc in _N8:
        s0 = idx[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)]
        s1 = idx[max(0, dr):h + min(0, dr), max(0, dc):w + min(0, dc)]
        L = np.hypot(dr, dc)
        rows.append(s0.ravel())
        cols.append(s1.ravel())
        vals.append(0.5 * (cost.ravel()[s0.ravel()] + cost.ravel()[s1.ravel()])
                    * L)
    g = coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                           np.concatenate(cols))),
                   shape=(h * w, h * w)).tocsr()
    ia = (a[0] - r0) * w + (a[1] - c0)
    ib = (b[0] - r0) * w + (b[1] - c0)
    _, pred = dijkstra(g, indices=ia, return_predecessors=True)
    path = [ib]
    while path[-1] != ia:
        p = pred[path[-1]]
        if p < 0:
            raise RuntimeError('no path')
        path.append(p)
    path = path[::-1]
    return [(r0 + k // w, c0 + k % w) for k in path]


def monotone_profile(v):
    """Least-squares non-increasing fit of a profile (pool adjacent
    violators): a deep one-cell pit on a river is mostly raised, a bump is
    mostly cut, never one carried along the whole reach."""
    v = np.asarray(v, dtype=np.float64)
    vals, wts, lens = [], [], []
    for x in v:
        vals.append(x)
        wts.append(1.0)
        lens.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            w = wts[-2] + wts[-1]
            m = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            n = lens[-2] + lens[-1]
            vals[-2:] = [m]
            wts[-2:] = [w]
            lens[-2:] = [n]
    return np.repeat(vals, lens)


def burn_rivers(z, lonlat_to_rc, rivers, dvic=0.0625, depth=0.0, pad=10):
    """Lower the DEM along river paths.

    z            DEM [ny, nx]
    lonlat_to_rc function (lon, lat) -> (row, col) floats on this grid
    rivers       output of river_cells
    dvic         coarse cell size (degrees)
    depth        extra lowering of the burnt cells (m)

    For each coarse river cell, the node is the lowest DEM cell inside its
    footprint (for a downstream cell outside the grid: the lowest cell of
    the grid edge it lies beyond, within its footprint); consecutive nodes
    along the D8 links are joined by a least-cost path; along each path the
    elevation is replaced by its non-increasing least-squares fit
    (monotone_profile, minus `depth`). Returns (dem, mask of burnt
    cells)."""
    z = np.asarray(z, dtype=np.float64)
    ny, nx = z.shape
    out = z.copy()
    burnt = np.zeros(z.shape, bool)

    def node(lon, lat, clip=False):
        ra, ca = lonlat_to_rc(lon - dvic / 2, lat + dvic / 2)
        rb, cb = lonlat_to_rc(lon + dvic / 2, lat - dvic / 2)
        r0, r1 = int(np.floor(min(ra, rb))), int(np.ceil(max(ra, rb)))
        c0, c1 = int(np.floor(min(ca, cb))), int(np.ceil(max(ca, cb)))
        if clip:                      # squeeze onto the nearest grid edge
            r0, r1 = min(max(r0, 0), ny - 1), max(min(r1, ny), 1)
            c0, c1 = min(max(c0, 0), nx - 1), max(min(c1, nx), 1)
        r0, r1 = max(r0, 0), min(r1, ny)
        c0, c1 = max(c0, 0), min(c1, nx)
        if r1 <= r0 or c1 <= c0:
            return None
        sub = z[r0:r1, c0:c1]
        k = int(np.argmin(sub))
        return (r0 + k // sub.shape[1], c0 + k % sub.shape[1])

    for lon, lat, lon2, lat2, _ in sorted(rivers, key=lambda x: x[4]):
        a, b = node(lon, lat), node(lon2, lat2)
        if a is None:
            continue
        if b is None:
            b = node(lon2, lat2, clip=True)
        path = _least_cost_path(z, a, b, pad)
        zz = monotone_profile([out[p] for p in path]) - depth
        for p, v in zip(path, zz):
            out[p] = v
            burnt[p] = True
    return out, burnt
