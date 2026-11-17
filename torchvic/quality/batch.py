"""All pure-lag channels of a router in one pass.

For a constituent with one decay rate k (or heat with one relaxation rate)
the channel's unit hydrograph is the water's times a function of the lag,
UH_k[p, l] = UH[p, l] d[l].  The tree sum is linear, so the channel's node
sums are the water kernel's sums of the channel's own inputs, scaled by d[l]
afterwards:  t_c[node, l] = d[l] * sum_p x_c[p] UH[p, l].  The inputs of every
channel are therefore multiplied against the water kernel in one batched
product (the kernel is read once for all of them, no transformed copy is
stored), the level sums are done together, and one kernel applies the lag
factors, adds to the rings and credits the ledgers.  The decayed amount per
node is  sum_p x_c[p] - sum_l d[l] t[node, l]  (the ones column of the water
tree holds sum_p x_c[p]).
"""
import torch

from torchvic.model import routing_fast as rf
from torchvic.model import routing_kernels as K


class Holder:
    """One channel's views into the stacked state (ledger access)."""

    def __init__(self, bq, i):
        self.i, self.dt = i, bq.dt
        self.ring, self.in_runoff, self.in_hooks = (
            bq.ring_all[i], bq.in_runoff_all[i], bq.hooks_all[i])
        self.out, self.lost = bq.out_all[i], bq.lost_all[i]
        self.withdrawn, self.injected = bq.wd_all[i], bq.inj_all[i]

    def balance(self):
        return self.in_runoff + self.in_hooks - self.out - self.ring.sum(0) * self.dt


class UniformChannels:
    def __init__(self, router, w, d):
        """w [C, n_pair] input weights; d [C, L] lag factors."""
        self.r = router
        dev, dt = router.dev, router.dt
        self.dt, self.dtype, self.dev = dt, router.dtype, dev
        self.C, self.L, self.N = w.shape[0], router.L, router.n_node
        ns = max(router.n_site, 1)
        z = lambda *sh: torch.zeros(*sh, dtype=self.dtype, device=dev)
        C, L, N = self.C, self.L, self.N
        self.ring_all = z(C, L, N)
        self.in_runoff_all, self.lost_all = z(C, N), z(C, N)
        self.hooks_all, self.out_all = z(C, N), z(C, N)
        self.wd_all, self.inj_all = z(C, ns), z(C, ns)
        self.w, self.d = w.contiguous(), d.contiguous()
        tree = router.tree
        self.block = tree.idx.shape[1]
        self.xpad = z(tree.idx.numel() * C)
        self.holders = [Holder(self, i) for i in range(C)]
        # level reductions for C stacked channels: rows [C, rows] over s [C, nprev]
        self.levels = []
        pad = tree.idx.shape[0]
        nprev = pad
        off = torch.arange(C, device=dev)
        for ix in tree.levels:
            cnt = (ix != pad).sum(1)
            ix_all = (ix[None] + off[:, None, None] * nprev).reshape(-1, ix.shape[1])
            self.levels.append((ix_all.contiguous(), cnt.repeat(C).contiguous(),
                                ix.shape[0]))
            pad = nprev = ix.shape[0]

    def zero(self):
        for x in (self.ring_all, self.in_runoff_all, self.lost_all, self.hooks_all,
                  self.out_all, self.wd_all, self.inj_all):
            x.zero_()

    def begin(self, flux):
        """flux [C, n_cell]: per-cell input of every channel."""
        r, tree = self.r, self.r.tree
        K.begin_xc(flux.contiguous(), r.cell, self.w, tree.idx_flat, self.xpad,
                   r.n_pair, self.block)
        nb = tree.idx.shape[0]
        s = torch.bmm(self.xpad.view(nb, self.C, self.block), tree.U)   # [nb, C, F]
        s = s.permute(1, 0, 2).contiguous()                              # [C, nb, F]
        F = s.shape[2]
        for ix_all, cnt_all, rows in self.levels:
            s = K.treered(s.view(-1, F), ix_all, cnt_all).view(self.C, rows, F)
        K.decay_update(self.ring_all, r.head, self.L, tree.ids, s, self.d,
                       self.in_runoff_all, r._dtt, lost=self.lost_all)

    def put(self, i, li, sid, mass):
        """Add mass [n] of channel i at level li's sites."""
        r = self.r
        lv, k = r.levels[li], r.lk[li]
        K.xfill(mass.contiguous(), k, r._dtt)
        t = rf.blocked(lv['tree'], k['xpad'])
        K.decay_update(self.ring_all[i:i + 1], r.head, self.L, lv['tree'].ids,
                       t.view(1, *t.shape), self.d[i:i + 1],
                       self.hooks_all[i:i + 1], r._dtt)
        self.inj_all[i].index_add_(0, sid, mass)

    def end(self):
        """[C, node] flux now; call before the water router's end()."""
        return K.end_batch(self.ring_all, self.r.head, self.out_all, self.r._dtt)
