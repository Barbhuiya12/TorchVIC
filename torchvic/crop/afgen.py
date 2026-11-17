# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
#
# The AFGEN table function follows the one in PCSE (pcse/util.py,
# Copyright (c) 2004-2024 Wageningen Environmental Research, Wageningen-UR,
# Allard de Wit; EUPL-1.2), see torchvic/crop/NOTICE_PCSE.txt.
"""AFGEN piecewise-linear tables, batched over cells.

PCSE evaluates `Afgen` with `bisect_left`: x <= first x gives the first y,
x >= last x gives the last y, otherwise
`y[i] + slope[i] * (x - x[i])` with `i = bisect_left(xs, x) - 1`. The same
arithmetic is done here for a whole tensor of x at once. Tables of different
length are padded with ascending x and a repeated last y (zero slope), which
does not change any value.
"""
import torch


def check_ascending(flat):
    """PCSE's rule (Afgen._check_x_ascending): x strictly ascending, or one
    break after which the trailing padding (e.g. zeros from a database) is
    dropped. Returns (xs, ys)."""
    xl = [float(v) for v in flat[0::2]]
    yl = [float(v) for v in flat[1::2]]
    asc = [xl[i] > xl[i - 1] for i in range(1, len(xl))]
    breaks = sum(1 for a, b in zip(asc, asc[1:]) if a ^ b)
    if breaks == 0:
        return xl, yl
    if breaks == 1:
        xs, ys = [xl[0]], [yl[0]]
        for i, up in zip(range(1, len(xl)), asc):
            if up:
                xs.append(xl[i])
                ys.append(yl[i])
        return xs, ys
    raise ValueError('AFGEN x values not strictly ascending: %r' % (xl,))


def afgen_py(flat, x):
    """Plain-Python AFGEN of a flat [x0, y0, x1, y1, ...] list (load time)."""
    xs, ys = check_ascending(flat)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    import bisect
    i = bisect.bisect_left(xs, x) - 1
    slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
    return ys[i] + slope * (x - xs[i])


class Afgen:
    """Batched AFGEN. `x`, `y` are [M, K] (M = 1 shared or M = cells)."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.slope = (y[:, 1:] - y[:, :-1]) / (x[:, 1:] - x[:, :-1])
        self.k = x.shape[1]
        self.x0 = x[:, 0]
        self.xl = x[:, -1]
        self.y0 = y[:, 0]
        self.yl = y[:, -1]

    @classmethod
    def from_lists(cls, tables, device, dtype=torch.float64):
        """`tables`: list of flat [x0, y0, x1, y1, ...] lists, one per set."""
        k = max(len(t) // 2 for t in tables)
        xs, ys = [], []
        for t in tables:
            x, y = check_ascending(t)
            while len(x) < k:
                x.append(x[-1] + 1.0)
                y.append(y[-1])
            xs.append(x)
            ys.append(y)
        return cls(torch.tensor(xs, dtype=dtype, device=device),
                   torch.tensor(ys, dtype=dtype, device=device))

    def expand(self, index):
        """Rows per cell: `index` [N] picks the table of each cell."""
        return Afgen(self.x[index], self.y[index])

    def _take(self, tbl, i):
        if tbl.shape[0] == 1:
            return tbl[0][i]
        return tbl.gather(1, i[:, None])[:, 0]

    def __call__(self, v):
        xs = self.x
        i = ((xs < v[:, None]).sum(1) - 1).clamp_(0, self.k - 2)
        out = self._take(self.y, i) + self._take(self.slope, i) * (v - self._take(xs, i))
        out = torch.where(v >= self.xl, self.yl.expand_as(v), out)
        out = torch.where(v <= self.x0, self.y0.expand_as(v), out)
        return out
