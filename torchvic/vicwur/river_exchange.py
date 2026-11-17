# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""River-aquifer exchange of mode fixed, solved in the routing sweep.

VIC-WUR 3.0 hands MODFLOW the discharge of the period, MODFLOW's RIV
package turns the stage of that discharge into a flux, and the flux enters
the routing as forced inflow of the *next* period. The stage and the
exchange then sit one period apart: a flood wave of the day gives a large
loss the day after, when the water is gone, the discharge swings, and the
next stage swings with it (a two-day limit cycle; ISC 2005: negative
discharge on 0.4 % of the days, and 0.2 of relaxation on the stage only
damped it).

Here the exchange is part of the day, as an implicit river reach. After a
cell has routed its inflow (slot `river_exchange`, between `route` and
`wu_local`), its discharge Q is known and the aquifer head h of the start of
the day is held. The loss L (m3 s-1, positive = river to aquifer) of the
RIV record solves

    L = Cr * (stage(Q - L) - max(h, rbot)),       0 <= Q - L

with the conductance Cr (m2 s-1) and the stage law of `groundwater.stage`
(Manning, bankfull cap): the stage the river has *after* the exchange. The
left side minus the right side increases in L, so the root is unique and
bisection finds it (`iterations`, to 2**-iterations of the bracket). The
loss is taken from the discharge bins of the ring (the water that leaves the
cell today), in proportion to them, so it cannot exceed the water in the
reach: when the root is more than Q the river runs dry, T = Q and the
shortfall is unmet demand, kept as a debt (`debt`, m3) the next days'
exchange has to pay. A gain (h above the stage) is added to the bins.
MODFLOW then solves the day with the stage the slot ended on, so its RIV
flux is the one the river saw up to the head change over the day; the small
difference (and any unmet loss) is the debt.

Only the head enters from MODFLOW before the day, and a head moves slowly,
so there is no lag loop to relax.
"""
import torch

from .groundwater import MANNING

#: bisection steps of the loss (the bracket is Q - lo, 2**-iterations of it)
ITERATIONS = 50

#: a discharge under this (m3 s-1) is none: the loss is taken in proportion
#: to the bins, and dividing by a denormal overflows
POS = 1e-12


def stage(q, bot, den, cap):
    """The river stage of discharge q >= 0 (`groundwater.stage`, fixed mode,
    on the cells that have a river): bot + min(d, cap), d from Manning's
    law, 0 below 1 cm. `den` is width * sqrt(slope) (inf: no slope)."""
    d1 = (MANNING * q / den) ** 0.6
    d = torch.where(d1 < 0.01, torch.zeros_like(d1), d1)
    return bot + torch.minimum(d, cap)


def exchange_torch(Q, C, bot, den, cap, rbot, h, debt, nit=ITERATIONS,
                   step_s=86400.0):
    """The slot's arithmetic on tensors (the reference of the Triton
    kernel). Q [n] discharge, C [n] conductance (m2 s-1), bot, den, cap,
    rbot [n] river statics, h [n] head, debt [n] m3 owed to the aquifer.
    Returns (T, stage, unmet) in m3 s-1, m, m3 s-1; T > 0 is a loss."""
    Q = Q.clamp(min=0.0)
    hh = torch.maximum(h, rbot)
    lo = C * (bot - hh)
    dry = lo >= Q
    a, b = lo.clone(), Q.clone()
    for _ in range(nit):
        mid = 0.5 * (a + b)
        g = mid - C * (stage((Q - mid).clamp(min=0.0), bot, den, cap) - hh)
        up = g > 0
        b = torch.where(up, mid, b)
        a = torch.where(up, a, mid)
    root = torch.where(dry, lo, 0.5 * (a + b))
    want = root + debt / step_s
    T = torch.minimum(want, Q)
    unmet = (want - Q).clamp(min=0.0)
    return T, stage((Q - T).clamp(min=0.0), bot, den, cap), unmet


class RiverExchange(object):
    """The slot on a router: per-cell statics and buffers in sweep order,
    fixed pointers (the sweep may replay as a CUDA graph)."""

    def __init__(self, router, statics, iterations=ITERATIONS):
        """`statics`: dict of [N] tensors in sweep order: C (m2 s-1), bot,
        den, cap, rbot (zero conductance: no river on the cell)."""
        r = self.r = router
        self.nit = int(iterations)
        for k in ('C', 'bot', 'den', 'cap', 'rbot'):
            setattr(self, k, statics[k].contiguous())
        z = lambda: torch.zeros(r.n, dtype=torch.float64, device=r.device)
        self.h, self.debt = z(), z()          # set by the connector
        self.T, self.stg, self.unmet = z(), z(), z()
        self.stg.copy_(self.bot)
        self._cells = {}
        self.river = self.C > 0
        self._cst = torch.tensor([MANNING, 0.6, 0.01, 86400.0],
                                 dtype=torch.float64, device=r.device)

    def begin(self):
        self.T.zero_()
        self.unmet.zero_()

    def _sel(self, lv, cells):
        s = self._cells.get(lv)
        if s is None:
            s = cells[self.river.index_select(0, cells)]
            self._cells[lv] = s
        return s

    def level(self, lv, cells):
        r = self.r
        sel = self._sel(lv, cells)
        if sel.numel() == 0:
            return
        if r.device.type == 'cuda':
            R, N = r.ring.shape
            from .river_exchange_kernels import kernel
            kernel[(-(-sel.numel() // 128),)](
                sel, r.ring, r.head, r.discharge, self.C, self.bot,
                self.den, self.cap, self.rbot, self.h, self.debt, self.T,
                self.stg, self.unmet, sel.numel(), N, self._cst,
                R=R, NSUB=r.nsub, NIT=self.nit, BLOCK=128)
            return
        g = lambda x: x.index_select(0, sel)
        phys = r.logical(torch.arange(r.nsub, device=r.device))
        Q = r.ring.index_select(0, phys).index_select(1, sel).sum(0)
        T, stg, un = exchange_torch(Q, g(self.C), g(self.bot), g(self.den),
                                    g(self.cap), g(self.rbot), g(self.h),
                                    g(self.debt), self.nit)
        Qc = Q.clamp(min=0.0)
        pos = Qc > POS
        for k in range(r.nsub):
            p = r.logical(k)
            q = r.ring[p].index_select(0, sel)
            q = torch.where(pos, q * (1.0 - T / torch.where(
                pos, Qc, torch.ones_like(Qc))),
                (q - T / r.nsub).clamp(min=0.0))
            r.ring[p, sel] = q
        r.discharge[sel] = Qc - T
        self.T[sel], self.stg[sel], self.unmet[sel] = T, stg, un

    def graph_ptrs(self):
        return [self.C, self.bot, self.den, self.cap, self.rbot, self.h,
                self.debt, self.T, self.stg, self.unmet]
