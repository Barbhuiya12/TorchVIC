"""River network routing to many gauges, on the GPU.

A grid of D8 flow directions gives every cell one downstream neighbour. A
gauge is a cell; the cells draining through it are its basin, and a gauge
downstream of another contains it (nested basins). Following RVIC's source
list, every (cell, gauge) pair gets a unit hydrograph from the flow
distance along the channel between them (Lohmann et al. 1996, as
`routing.lohmann_unit_hydrographs`), and

    Q_g(t) = sum_{pairs (c, g)} A_c/(1000 dt) sum_k UH[k, c, g] R_c(t - k)

with R in mm per step, A in m2, Q in m3/s. The network (topological order,
distance to mouth, the pair list) is built once on the host with numpy; the
time convolution runs on the device in chunks of pairs, batched over time,
with a fixed-order segmented sum into gauges so the CPU and the GPU agree to
rounding.

Flow directions are NOT derived here from data the caller has not given:
`d8_downstream` takes a direction grid. `derive_d8_from_elevation` is a
stand-in (priority flood on the cells' own elevations) for timing or for
basins whose direction grid is not available; it is not a validated river
network.
"""
import heapq
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import routing

#: ESRI D8 codes -> (drow, dcol), row index increasing with the lat axis.
ESRI_D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
           16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
          (1, 1)]
EARTH_R = 6371008.8


def _great_circle(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


class RiverNetwork:
    """Cells of a masked grid joined by downstream links.

    down [n]   index of the downstream cell, -1 at a mouth
    length [n] channel length of the link cell -> down, m (0 at a mouth)
    """

    def __init__(self, down, length):
        self.down = np.asarray(down, dtype=np.int64)
        self.length = np.asarray(length, dtype=np.float64)
        self.n = len(self.down)
        self._order()

    def _order(self):
        """Topological order (upstream first) and distance to the mouth.

        Kahn's algorithm by levels; a cycle (a bad direction grid) leaves
        cells unprocessed and raises."""
        n, down = self.n, self.down
        indeg = np.bincount(down[down >= 0], minlength=n)
        level = np.nonzero(indeg == 0)[0]
        order = []
        while level.size:
            order.append(level)
            d = down[level]
            d = d[d >= 0]
            np.subtract.at(indeg, d, 1)
            level = np.unique(d[indeg[d] == 0])
        self.order = np.concatenate(order) if order else np.zeros(0, np.int64)
        if len(self.order) != n:
            raise ValueError('flow directions contain a cycle (%d cells)'
                             % (n - len(self.order)))
        dm = np.zeros(n)
        for c in self.order[::-1]:      # mouth first
            if down[c] >= 0:
                dm[c] = dm[down[c]] + self.length[c]
        self.dist_to_mouth = dm


def d8_downstream(fdir, mask, lat, lon, codes=ESRI_D8):
    """RiverNetwork from a D8 direction grid [ny, nx] (ESRI codes).

    A cell whose code is unknown, or points off the mask, is a mouth."""
    ny, nx = mask.shape
    idx = -np.ones((ny, nx), dtype=np.int64)
    cells = np.argwhere(mask)
    idx[mask] = np.arange(len(cells))
    r, c = cells[:, 0], cells[:, 1]
    down = -np.ones(len(cells), dtype=np.int64)
    for code, (dr, dc) in codes.items():
        m = fdir[r, c] == code
        rr, cc = r[m] + dr, c[m] + dc
        ok = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
        t = -np.ones(m.sum(), dtype=np.int64)
        t[ok] = idx[rr[ok], cc[ok]]
        down[np.nonzero(m)[0]] = t
    return _with_lengths(down, cells, lat, lon), cells


def _with_lengths(down, cells, lat, lon):
    r, c = cells[:, 0], cells[:, 1]
    length = np.zeros(len(down))
    ok = down >= 0
    d = down[ok]
    length[ok] = _great_circle(lat[r[ok]], lon[c[ok]], lat[r[d]], lon[c[d]])
    return RiverNetwork(down, length)


def derive_d8_from_elevation(elev, mask, lat, lon):
    """Priority-flood D8 from the cells' own elevations (STAND-IN).

    Seeds are the mask cells touching the grid edge or a non-mask cell; the
    flood spreads inland lowest-first and each cell drains to the neighbour
    that reached it. Pits are filled by construction, so closed basins drain
    to the coast along their lowest rim. Reproducible, but a derived
    network, not an observed one."""
    ny, nx = mask.shape
    cells = np.argwhere(mask)
    idx = -np.ones((ny, nx), dtype=np.int64)
    idx[mask] = np.arange(len(cells))
    down = -np.ones(len(cells), dtype=np.int64)
    seen = np.zeros(len(cells), bool)
    heap = []
    for i, (r, c) in enumerate(cells):
        for dr, dc in _NEIGH:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < ny and 0 <= cc < nx) or not mask[rr, cc]:
                seen[i] = True
                heapq.heappush(heap, (float(elev[r, c]), i))
                break
    while heap:
        z, i = heapq.heappop(heap)
        r, c = cells[i]
        for dr, dc in _NEIGH:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx and mask[rr, cc]:
                j = idx[rr, cc]
                if not seen[j]:
                    seen[j] = True
                    down[j] = i
                    heapq.heappush(heap, (max(z, float(elev[rr, cc])), j))
    return _with_lengths(down, cells, lat, lon), cells


class PairTable:
    """(cell, gauge) pairs sorted by gauge, with the channel distance of
    each. cell/gauge/distance are numpy; `.to(device)` gives tensors."""

    def __init__(self, cell, gauge, distance, n_gauge, weight=None):
        o = np.lexsort((cell, gauge))
        self.cell, self.gauge = cell[o], gauge[o]
        self.distance = distance[o]
        # share of the cell's runoff that reaches the gauge (1 for a channel pair)
        self.weight = (np.ones(len(o)) if weight is None
                       else np.asarray(weight, dtype=np.float64)[o])
        self.n_gauge = int(n_gauge)
        self.n_pair = len(self.cell)


def build_pairs(net, gauge_cells):
    """Every cell paired with every gauge on its downstream path (itself
    included when it is a gauge), with the distance between them.

    gauge_cells [n_gauge] distinct cell indices. Walks all cells downstream
    together, one link per iteration, so the cost is (longest path) passes
    over a shrinking active set."""
    gauge_cells = np.asarray(gauge_cells, dtype=np.int64)
    if len(np.unique(gauge_cells)) != len(gauge_cells):
        raise ValueError('two gauges share a cell')
    gid = -np.ones(net.n, dtype=np.int64)
    gid[gauge_cells] = np.arange(len(gauge_cells))
    cur = np.arange(net.n)
    src = cur.copy()
    cells, gauges, dist = [], [], []
    base = net.dist_to_mouth
    while cur.size:
        g = gid[cur]
        h = g >= 0
        if h.any():
            cells.append(src[h])
            gauges.append(g[h])
            dist.append(base[src[h]] - base[cur[h]])
        nxt = net.down[cur]
        keep = nxt >= 0
        cur, src = nxt[keep], src[keep]
    return PairTable(np.concatenate(cells), np.concatenate(gauges),
                     np.concatenate(dist), len(gauge_cells))


def add_subcell_gauges(pairs, subcell):
    """Append gauges whose catchment is smaller than a few cells.

    subcell: list over the new gauges of (cells, fractions): the network cells
    the catchment polygon covers and the fraction of each cell's area inside the
    polygon. Discharge of such a gauge is

        Q(t) = sum_c f_c A_c/(1000 dt) sum_k UH[k] R_c(t - k)

    i.e. the area-weighted sum of the covered cells' runoff + baseflow, routed
    with the cell unit hydrograph only: every pair has channel distance 0, no
    channel routing inside one cell. New gauge k has index pairs.n_gauge + k.
    The sum of f_c A_c is the polygon's area, so a uniform runoff of R mm per
    step gives exactly polygon area x R."""
    cell, gauge, dist, w = [pairs.cell], [pairs.gauge], [pairs.distance], [pairs.weight]
    for k, (c, f) in enumerate(subcell):
        c = np.asarray(c, dtype=np.int64)
        cell.append(c)
        gauge.append(np.full(len(c), pairs.n_gauge + k, dtype=np.int64))
        dist.append(np.zeros(len(c)))
        w.append(np.asarray(f, dtype=np.float64))
    return PairTable(np.concatenate(cell), np.concatenate(gauge),
                     np.concatenate(dist), pairs.n_gauge + len(subcell),
                     np.concatenate(w))


def river_steps(distance, velocity, dt):
    """Length (steps) of the channel response for the longest path."""
    xmax = float(torch.as_tensor(distance).max())
    vnow = max(float(torch.as_tensor(velocity).min()), routing._V_FLOOR)
    return int(min(max(3.0 * xmax / vnow / dt, 1.0) + 48, 4000))


def pair_unit_hydrographs(distance, velocity, diffusion, cell_uh, dt,
                          chunk=20000):
    """[n_uh, pair] unit hydrographs (columns sum to 1), in pair chunks so
    the [fine step, pair] intermediate stays bounded. All chunks share one
    length (set from the longest path), so they concatenate."""
    d = torch.as_tensor(distance)
    n_r = river_steps(d, velocity, dt)
    nuh = cell_uh.shape[0]
    out = []
    for a in range(0, d.shape[0], chunk):
        di = d[a:a + chunk]
        river = routing.lohmann_river_irf(di, velocity, diffusion, n_r, dt)
        x = river.transpose(0, 1).unsqueeze(1)
        w = torch.flip(cell_uh, dims=[0]).reshape(1, 1, nuh)
        y = F.conv1d(F.pad(x, (nuh - 1, nuh - 1)), w)
        out.append(y.squeeze(1).transpose(0, 1))
    return torch.cat(out, 1)


def route_network(runoff, pairs, uh, area, dt, chunk=4096):
    """Gauge discharge, m3/s. runoff [time, cell] mm per step; uh
    [n_uh, pair]; area [cell] m2. Returns [time, gauge].

    Pairs are processed `chunk` at a time (sorted by gauge): gather the
    cells' runoff, depthwise causal conv1d with each pair's hydrograph,
    weight by cell area, and sum into gauges with segment_reduce, which adds
    in pair order on every device."""
    dev, dt_ = runoff.device, runoff.dtype
    nt = runoff.shape[0]
    nuh = uh.shape[0]
    cell = torch.as_tensor(pairs.cell, device=dev)
    gauge = torch.as_tensor(pairs.gauge, device=dev)
    w_area = (area.to(dev)[cell] * torch.as_tensor(pairs.weight, device=dev,
                                                    dtype=area.dtype)
              / (routing.MM_PER_M * dt))
    out = torch.zeros(pairs.n_gauge, nt, dtype=dt_, device=dev)
    for a in range(0, pairs.n_pair, chunk):
        sl = slice(a, a + chunk)
        x = runoff.index_select(1, cell[sl]).transpose(0, 1).unsqueeze(0)
        k = torch.flip(uh[:, sl], dims=[0]).transpose(0, 1).unsqueeze(1)
        y = F.conv1d(F.pad(x, (nuh - 1, 0)), k, groups=k.shape[0])
        y = y.squeeze(0) * w_area[sl].unsqueeze(1)           # [pair, time]
        g, counts = torch.unique_consecutive(gauge[sl], return_counts=True)
        s = torch.segment_reduce(y, 'sum', lengths=counts, axis=0)
        out[g] += s
    return out.transpose(0, 1)


def route_network_reference(runoff, pairs, uh, area, dt):
    """Loop-over-pairs transcription (host, float64), for the tests."""
    nt = runoff.shape[0]
    out = np.zeros((nt, pairs.n_gauge))
    r = runoff.cpu().numpy()
    u = uh.cpu().numpy()
    a = area.cpu().numpy()
    for p in range(pairs.n_pair):
        c, g = pairs.cell[p], pairs.gauge[p]
        q = np.convolve(r[:, c], u[:, p])[:nt]
        out[:, g] += q * a[c] * pairs.weight[p] / (routing.MM_PER_M * dt)
    return out


def site_levels(net, site_cells):
    """Topological level of each site: the longest chain of other sites
    upstream of it (0 = no site above it)."""
    carry = np.zeros(net.n, np.int64)
    lev_cell = -np.ones(net.n, np.int64)
    is_site = np.zeros(net.n, bool)
    is_site[np.asarray(site_cells, np.int64)] = True
    for c in net.order:                       # upstream first
        out = carry[c]
        if is_site[c]:
            lev_cell[c] = carry[c]
            out = carry[c] + 1
        d = net.down[c]
        if d >= 0 and out > carry[d]:
            carry[d] = out
    return lev_cell[np.asarray(site_cells, np.int64)]


#: node kinds of a StepRouter column
KIND_CHANNEL, KIND_SUBCELL, KIND_CANAL = 0, 1, 2


def build_step_router(net, gauge_cells, area, dt, velocity=1.5,
                      diffusion=800.0, cell_uh=None, subcell=(), n_canal=0,
                      site_cells=(), device='cpu', dtype=torch.float64,
                      chunk=20000):
    """A `routing.StepRouter` whose discharge equals `route_network`.

    Nodes, in order: the channel gauges (`gauge_cells`), extra channel nodes
    for `site_cells` not already gauges, the sub-cell gauges (`subcell`, as
    `add_subcell_gauges`), then `n_canal` canal gauges (kind KIND_CANAL, fed
    only by `node_inject`).  Returns (router, info); info has `site_node`
    (node of each site) and the node counts.  Sites are the cells where
    withdrawals/injections/reservoirs act."""
    gauge_cells = np.asarray(gauge_cells, dtype=np.int64)
    site_cells = np.asarray(site_cells, dtype=np.int64)
    extra = [c for c in dict.fromkeys(site_cells.tolist())
             if c not in set(gauge_cells.tolist())]
    all_cells = np.concatenate([gauge_cells, np.asarray(extra, np.int64)])
    pairs = build_pairs(net, all_cells)
    n_ch = pairs.n_gauge
    if len(subcell):
        pairs = add_subcell_gauges(pairs, subcell)
    n_node = pairs.n_gauge + int(n_canal)
    dev = torch.device(device)
    if cell_uh is None:
        cell_uh = torch.ones(1, dtype=dtype, device=dev)
    with torch.device(dev):          # a different process default device must not leak in
        uh = pair_unit_hydrographs(
            torch.as_tensor(pairs.distance, device=dev, dtype=dtype),
            velocity, diffusion, cell_uh.to(dev), dt, chunk=chunk)
    cell_area = torch.as_tensor(area, device=dev, dtype=dtype)
    cell = torch.as_tensor(pairs.cell, device=dev)
    w = (cell_area[cell] * torch.as_tensor(pairs.weight, device=dev,
                                           dtype=dtype)
         / (routing.MM_PER_M * dt))
    kind = np.zeros(n_node, np.int8)
    kind[n_ch:pairs.n_gauge] = KIND_SUBCELL
    kind[pairs.n_gauge:] = KIND_CANAL
    node_of = {int(c): i for i, c in enumerate(all_cells)}
    site_node = np.array([node_of[int(c)] for c in site_cells], np.int64)
    inj = None
    if len(site_cells):
        slot = {int(c): i for i, c in enumerate(site_cells)}
        sel = np.nonzero(np.isin(pairs.cell, site_cells)
                         & (pairs.gauge < n_ch))[0]
        # one pair per (site cell, downstream node); sites are distinct cells
        sel = sel[np.argsort(pairs.gauge[sel], kind='stable')]
        d = torch.as_tensor(pairs.distance[sel], device=dev, dtype=dtype)
        n_r = river_steps(torch.as_tensor(pairs.distance), velocity, dt)
        with torch.device(dev):
            irf = routing.lohmann_river_irf(d, velocity, diffusion, n_r, dt)
        kern = torch.zeros(len(sel), uh.shape[0], dtype=dtype, device=dev)
        kern[:, :n_r] = irf.t()
        inj = ([slot[int(c)] for c in pairs.cell[sel]], pairs.gauge[sel],
               kern)
    site_level = site_levels(net, site_cells) if len(site_cells) else None
    r = routing.StepRouter(pairs.cell, pairs.gauge, uh, w, n_node, dt,
                           kind=kind, site_cell=site_cells,
                           site_level=site_level,
                           site_node=site_node if len(site_cells) else None,
                           inj=inj)
    info = dict(site_node=site_node, n_channel=len(gauge_cells),
                n_site_extra=len(extra), n_subcell=len(subcell),
                n_canal=int(n_canal), n_node=n_node, pairs=pairs)
    return r, info
