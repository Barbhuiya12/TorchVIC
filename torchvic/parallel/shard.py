"""Cell shards: which active cells a process runs, and putting the shards'
outputs back together.

VIC's cells are independent -- the land surface of one never reads another's
(routing is a separate step) -- so a run over N processes is the single run
with its active cell list cut into N pieces. Every operation the model
applies is per cell, so a shard's values are the single run's values at the
same cells, and the merge is a concatenation along the cell axis.

Cell j of the domain's active order (row-major) goes to shard j mod N. A
cell's cost differs a lot with place (a snow cell costs about twice a snow
free one, and the cold cells lie together), so pieces cut as contiguous runs
of the grid were measured at 75 s against 128 s on the ISC domain; cycling
through the cells gives every shard the same mix. `shard_ranges` is the
contiguous cut by tile count, kept for comparison.
"""
import contextlib

import numpy as np


def cell_weights(params_path, dom):
    """Work per active cell: its covered tiles (+1), from the parameter
    file's Cv; 1 for every cell when the file has none."""
    n = int(dom.n_active)
    try:
        import xarray as xr
        from ..io.ncshare import read
        with read(params_path) as ds:
            if 'Cv' not in ds:
                return np.ones(n)
            cv = np.asarray(dom.take(ds['Cv']))
    except Exception:                                    # noqa: BLE001
        return np.ones(n)
    return 1.0 + (cv > 0).reshape(-1, n).sum(0)


def shard_ranges(weights, n_shards):
    """[(start, stop)] * n_shards: contiguous pieces of about equal weight,
    every piece non-empty."""
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)
    k = int(n_shards)
    if k < 1:
        raise ValueError('need at least one shard')
    if k > n:
        raise ValueError('%d shards for %d cells' % (k, n))
    cum = np.cumsum(w)
    cuts = [0]
    for i in range(1, k):
        c = int(np.searchsorted(cum, cum[-1] * i / k))
        c = max(c, cuts[-1] + 1)
        c = min(c, n - (k - i))
        cuts.append(c)
    cuts.append(n)
    return list(zip(cuts[:-1], cuts[1:]))


def shard_index(n_cells, n_shards, k):
    """The active-cell indices of shard `k`: k, k+N, k+2N, ..."""
    if n_shards < 1 or not 0 <= k < n_shards:
        raise ValueError('shard %d of %d' % (k, n_shards))
    if n_shards > n_cells:
        raise ValueError('%d shards for %d cells' % (n_shards, n_cells))
    return np.arange(k, n_cells, n_shards)


def restrict_domain(dom, index):
    """Cut an ImageDomain to its active cells `index`, in place. The
    grid, the mask and the dataset stay whole: files written from the
    domain (history, state) keep the full grid, with fill where the shard
    does not run."""
    sl = np.asarray(index)
    full = np.zeros(dom.mask.shape)
    full[dom.rows[sl], dom.cols[sl]] = 1.0
    for a in ('rows', 'cols', 'lat', 'lon', 'area', 'frac',
              'flow_distance'):
        v = getattr(dom, a, None)
        if v is not None:
            setattr(dom, a, v[sl])
    dom.n_active = int(len(sl))
    # ImageParams compares a file's run_cell with the domain's; the shard's
    # own cells are the set it runs
    dom.run_cell = full
    return dom


@contextlib.contextmanager
def sharded(index, n_shards):
    """Inside the block every ImageDomain built is cut to shard `index` of
    `n_shards`. Yields a dict that records the shard's cell indices, rows and
    columns and the full active-cell count once a domain was built."""
    from ..io import domain as D
    orig = D.ImageDomain.__init__
    info = {}

    def init(self, *a, **k):
        orig(self, *a, **k)
        if 'range' in info:
            raise RuntimeError('two domains built in one shard')
        idx = shard_index(int(self.n_active), n_shards, index)
        info['index'] = idx
        info['n_full'] = int(self.n_active)
        info['rows'] = np.array(self.rows[idx])
        info['cols'] = np.array(self.cols[idx])
        restrict_domain(self, idx)

    D.ImageDomain.__init__ = init
    try:
        yield info
    finally:
        D.ImageDomain.__init__ = orig


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------
def merge_data(parts, index):
    """Per-key tensors of the shards, [rec, cell_i, ...] each, as one
    [rec, cell, ...]: shard i's cells go to the active-cell positions
    index[i]."""
    import torch
    keys = list(parts[0])
    for p in parts[1:]:
        if list(p) != keys:
            raise ValueError('shards hold different variables')
    n = sum(len(ix) for ix in index)
    out = {}
    for k in keys:
        x0 = parts[0][k]
        o = torch.empty((x0.shape[0], n) + tuple(x0.shape[2:]),
                        dtype=x0.dtype)
        for p, ix in zip(parts, index):
            o[:, torch.as_tensor(np.asarray(ix))] = p[k]
        out[k] = o
    return out


def merge_state_files(paths, cells, out_path, grid):
    """One VIC state file from the shards'. Each shard file is on the whole
    grid with fill off its own cells; `cells` is [(rows, cols)] per shard.
    Every variable whose last two axes are the grid takes each shard's
    values at its cells; the rest (coordinates, scalars) come from the
    first file, which all shards write alike. `grid` is the (y, x) dimension
    names."""
    import netCDF4
    srcs = [netCDF4.Dataset(p) for p in paths]
    try:
        s0 = srcs[0]
        dst = netCDF4.Dataset(out_path, 'w', format=s0.data_model)
        try:
            for a in s0.ncattrs():
                dst.setncattr(a, s0.getncattr(a))
            for name, d in s0.dimensions.items():
                dst.createDimension(name, None if d.isunlimited()
                                    else len(d))
            for name, v in s0.variables.items():
                fill = getattr(v, '_FillValue', None)
                kw = {}
                if fill is not None:
                    kw['fill_value'] = fill
                f = v.filters() or {}
                if f.get('zlib'):
                    kw.update(zlib=True, complevel=f.get('complevel', 4),
                              shuffle=f.get('shuffle', True))
                o = dst.createVariable(name, v.dtype, v.dimensions, **kw)
                for a in v.ncattrs():
                    if a != '_FillValue':
                        o.setncattr(a, v.getncattr(a))
                o.set_auto_maskandscale(False)
                v.set_auto_maskandscale(False)
                x = np.array(v[:])
                if tuple(v.dimensions[-2:]) == tuple(grid):
                    for src, (r, c) in zip(srcs[1:], cells[1:]):
                        sv = src.variables[name]
                        sv.set_auto_maskandscale(False)
                        x[..., r, c] = np.array(sv[:])[..., r, c]
                o[:] = x
        finally:
            dst.close()
    finally:
        for s in srcs:
            s.close()
    return out_path
