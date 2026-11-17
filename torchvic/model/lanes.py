"""The covered tiles of a mosaic as one flat axis of lanes.

basin.simulate keeps every per-tile quantity on a padded [cell, tile] grid:
every cell has the mosaic's full tile count, and a tile a cell does not
carry has zero cover and contributes nothing. On the ISC domain that is
18 tile slots per cell of which 2.4 are covered on average -- 251,579 of
1,892,034 -- so seven eighths of every per-tile operation computes tiles
whose results are multiplied by a zero cover and summed away.

`Lanes` runs the same loop on the covered tiles only. Each covered
(cell, tile) pair becomes a lane, and the loop sees the lanes as cells with
one tile each: the cell's forcing, soil and parameters gathered to every
lane of the cell, the tile's vegetation to the lane. Every per-lane value is
then the value the padded grid held at that (cell, tile), bit for bit,
because each operation is elementwise.

The one place lanes and cells meet is a sum over a cell's tiles. On the
padded grid that is a reduction over the tile axis of a [cell, tile, ...]
tensor whose uncovered entries hold zero; `tsum` rebuilds exactly that
tensor (the lanes' values at their slots, zero elsewhere), reduces it with
the same operation, so the sum is the grid's to the last bit, and hands
every lane its cell's total. A cell-level quantity is therefore held
identically by each of its lanes, and `cell` takes one lane per cell back
to the cell axis.

Tiles of zero cover are not computed at all, so a per-tile state or output
on an uncovered tile has no value here; `scatter` puts the covered lanes on
the padded grid with zeros elsewhere.
"""
import copy

import torch

from ..core import fusion, xsum


def _to_cell_many(xs, rep):
    return [x.index_select(0, rep) for x in xs]


#: Lanes.to_cell of a list of tensors, one fused region on the graph path
to_cell_many = fusion.region(_to_cell_many)


class _LaneSeries(object):
    __slots__ = ('_owner', '_key')

    def __init__(self, owner, key):
        self._owner, self._key = owner, key

    def __getitem__(self, t):
        return self._owner.at(self._key, t)

    @property
    def shape(self):
        v = self._owner.atm[self._key]
        return (v.shape[0], self._owner.lanes.n_lanes) + tuple(v.shape[2:])


class LaneAtmosphere(object):
    """`atm` ([time, cell] fields) read at the lanes, one step at a time:
    `a[k][t]` is `atm[k][t]` gathered to the lanes, made once per step.

    Only the per-step reads simulate's loop makes are supported (`k in a`,
    `a[k][t]`); the whole series stays on the cell axis."""

    def __init__(self, atm, lanes):
        self.atm, self.lanes = atm, lanes
        self._t, self._cache = None, {}
        # a field's snow-step override (simulate's daily step with snow
        # sub-steps): the lanes' own buffer, read at every t
        self._over = {}

    def has_series(self, k):
        """True when `k` is a [time, cell] field (not an override)."""
        return k in self.atm

    def __setitem__(self, k, v):
        # simulate swaps a field for a snow step's lane values and puts the
        # series back after it; the series is the one __getitem__ gave
        if isinstance(v, _LaneSeries):
            self._over.pop(k, None)
        else:
            self._over[k] = v

    def __contains__(self, k):
        return k in self.atm

    def keys(self):
        return self.atm.keys()

    def __getitem__(self, k):
        if k not in self.atm:
            raise KeyError(k)
        if k in self._over:
            return self._over[k]
        return _LaneSeries(self, k)

    def at(self, k, t):
        if t != self._t:
            self._t, self._cache = t, {}
        v = self._cache.get(k)
        if v is None:
            v = self.lanes.gather_cell(self.atm[k][t])
            self._cache[k] = v
        return v


class Lanes(object):
    """The covered (cell, tile) pairs of a [cell, tile] cover, cell-major."""

    def __init__(self, cv):
        nc, ntl = int(cv.shape[0]), int(cv.shape[1])
        rc = (cv > 0.0).nonzero()
        self.n_cells, self.n_tiles = nc, ntl
        self.cell = None
        self.lane_cell = rc[:, 0].contiguous()
        self.lane_tile = rc[:, 1].contiguous()
        self.n_lanes = int(rc.shape[0])
        # one lane per cell: the first (lanes are cell-major)
        first = torch.ones(self.n_lanes, dtype=torch.bool, device=cv.device)
        first[1:] = self.lane_cell[1:] != self.lane_cell[:-1]
        self.rep = first.nonzero().flatten()
        self.complete = int(self.rep.shape[0]) == nc
        self._pad = {}
        self.device = cv.device
        # for the fused tsum: each lane's cell's first lane and the cell's
        # covered tiles as bits
        cstart = torch.zeros(nc, dtype=torch.long, device=cv.device)
        cstart[self.lane_cell[self.rep]] = self.rep
        self.lane_start = cstart.index_select(0, self.lane_cell)
        bits = torch.zeros(nc, dtype=torch.long, device=cv.device)
        if ntl <= 62:
            bits.index_add_(0, self.lane_cell,
                            torch.ones_like(self.lane_tile) << self.lane_tile)
        self.lane_mask = bits.index_select(0, self.lane_cell)
        # per cell, for the tsum kernel (model.tsum_kernel): the cell's sum
        # once, then at its lanes
        self.cell_start, self.cell_mask = cstart, bits
        if self.n_lanes < 2 ** 31 and ntl <= 30:
            # 32-bit for the tsum kernel
            self.lane_start = self.lane_start.to(torch.int32)
            self.lane_mask = self.lane_mask.to(torch.int32)
            self.cell_start = self.cell_start.to(torch.int32)
            self.cell_mask = self.cell_mask.to(torch.int32)

    # ---- cell axis -> lanes -------------------------------------------
    def gather_cell(self, x):
        """A [cell, ...] tensor at the lanes: [lane, ...]."""
        return x.index_select(0, self.lane_cell)

    def gather_tile(self, x):
        """A [cell, tile, ...] tensor at the lanes: [lane, 1, ...]."""
        return x[self.lane_cell, self.lane_tile].unsqueeze(1)

    def is_cell(self, x):
        return (torch.is_tensor(x) and x.dim() >= 1
                and int(x.shape[0]) == self.n_cells)

    def is_tile(self, x):
        return (torch.is_tensor(x) and x.dim() >= 2
                and int(x.shape[0]) == self.n_cells
                and int(x.shape[1]) == self.n_tiles)

    def mosaic(self, mosaic):
        """The mosaic at the lanes: every [cell, tile, ...] field becomes
        [lane, 1, ...]. None if a field has another shape."""
        out = copy.copy(mosaic)
        for k, v in vars(mosaic).items():
            if torch.is_tensor(v):
                if not self.is_tile(v):
                    return None
                setattr(out, k, self.gather_tile(v))
            elif isinstance(v, (dict, list, tuple)):
                if v:
                    return None
        return out

    def soil(self, sp):
        """A SoilParams (slots) at the lanes: every per-cell tensor
        gathered, dicts of tensors entry by entry. None if a tensor is not
        per cell."""
        out = object.__new__(type(sp))
        for name in type(sp).__slots__:
            if not hasattr(sp, name):
                continue
            v = getattr(sp, name)
            if torch.is_tensor(v):
                if v.dim() == 0:
                    pass
                elif not self.is_cell(v):
                    return None
                else:
                    v = self.gather_cell(v)
            elif isinstance(v, dict):
                w = {}
                for k, x in v.items():
                    if torch.is_tensor(x) and x.dim() >= 1:
                        if not self.is_cell(x):
                            return None
                        x = self.gather_cell(x)
                    w[k] = x
                v = w
            setattr(out, name, v)
        return out

    def params(self, p):
        """Run parameters: per-cell tensors gathered, scalars kept. None if
        a tensor is neither."""
        out = {}
        for k, v in p.items():
            if torch.is_tensor(v) and v.dim() >= 1:
                if v.dim() != 1 or not self.is_cell(v):
                    return None
                v = self.gather_cell(v)
            out[k] = v
        return out

    def batch(self, batch):
        """A shallow copy of `batch` with its per-cell tensors at the
        lanes (the routing reads the original's)."""
        out = copy.copy(batch)
        for k, v in vars(batch).items():
            if self.is_cell(v):
                setattr(out, k, self.gather_cell(v))
        if hasattr(out, 'n_cells'):
            out.n_cells = self.n_lanes
        return out

    def atmosphere(self, atm):
        return LaneAtmosphere(atm, self)

    # ---- lanes -> cell axis -------------------------------------------
    def tsum(self, x):
        """Sum over the tile axis (dim 1) of a [lane, 1, ...] tensor, as the
        padded grid sums it, handed back to every lane: [lane, ...].

        The padded tensor is one buffer per shape, zero once: every call
        writes the same (covered) slots and the rest stay zero, so it holds
        what a fresh zero tensor with the lanes written in would."""
        if ((torch.compiler.is_compiling() or fusion.wants((x,)))
                and min(xsum.MAX_R, 62) >= self.n_tiles
                and x.dtype == torch.float64 and x.is_contiguous()
                and x.dim() >= 2 and int(x.shape[1]) == 1):
            L = 1
            for s_ in x.shape[2:]:
                L *= int(s_)
            if L <= 4 and fusion.XSUM:
                from .tsum_kernel import tsum_gather
                return tsum_gather(x, self.cell_start, self.cell_mask,
                                   self.n_tiles).index_select(
                                       0, self.lane_cell)
        shape = (self.n_cells, self.n_tiles) + tuple(x.shape[2:])
        key = (shape, x.dtype, x.device)
        full = self._pad.get(key)
        if full is None:
            full = torch.zeros(shape, dtype=x.dtype, device=x.device)
            self._pad[key] = full
        full[self.lane_cell, self.lane_tile] = x[:, 0]
        return full.sum(1).index_select(0, self.lane_cell)

    def to_cell(self, x):
        """A cell-level [lane, ...] tensor (each lane holding its cell's
        value) on the cell axis: [cell, ...]."""
        return x.index_select(0, self.rep)

    def scatter(self, x):
        """A per-tile [lane, 1, ...] tensor on the padded [cell, tile, ...]
        grid, zero on the tiles not covered."""
        full = torch.zeros((self.n_cells, self.n_tiles) + tuple(x.shape[2:]),
                           dtype=x.dtype, device=x.device)
        full[self.lane_cell, self.lane_tile] = x[:, 0]
        return full

    def state_to_lanes(self, x):
        """A carried state from a padded-grid run, at the lanes."""
        if isinstance(x, dict):
            return {k: self.state_to_lanes(v) for k, v in x.items()}
        if self.is_tile(x):
            return self.gather_tile(x)
        return x
