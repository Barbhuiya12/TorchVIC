"""FastStepRouter: `routing.StepRouter` with fused CUDA (Triton) hot paths.

Same network objects, same state, same results to the last bit (every
kernel performs the same IEEE operations in the same order as the torch ops
it replaces; the tree-sum matrix products stay cuBLAS).  The difference is
the launch count: per hook level the cap, take/put and the tree-sum input
fill are one kernel and the ring update with its ledger credit is one more.
`StepRouter` stays the default; build this one with `build_fast_step_router`
or `FastStepRouter.from_router(router)`.  CPU routers are returned as is.
"""
import torch

from torchvic.model import river_network as rn
from torchvic.model import routing
from torchvic.model import routing_kernels as K


def _level_data(lv):
    """Static per-level arrays: pairs sorted by site (CSR), the kernel rows in
    that order, and each pair's slot in the tree's blocked input."""
    dev = lv['pair_site'].device
    tree = lv['tree']
    order = torch.argsort(lv['pair_site'], stable=True)
    ps = lv['pair_site'][order]
    n = lv['n']
    pstart = torch.zeros(n + 1, dtype=torch.long, device=dev)
    pstart[1:] = torch.cumsum(torch.bincount(ps, minlength=n), 0)
    npair = lv['pair_site'].numel()
    flat = tree.idx_flat
    valid = flat < npair
    pos_pair = torch.empty(npair, dtype=torch.long, device=dev)
    pos_pair[flat[valid]] = torch.arange(flat.numel(), device=dev)[valid]
    return dict(n=n, pstart=pstart, ps=ps, pnode=lv['pair_node'][order].contiguous(),
                kern_s=lv['kern'][order].contiguous(), pos=pos_pair[order].contiguous(),
                sites=lv['sites'].contiguous(),
                xpad=torch.zeros(flat.numel(), dtype=lv['kern'].dtype, device=dev))


def blocked(tree, xb):
    """_TreeSum.apply from an already gathered, zero padded [block count, block]:
    the batched product (cuBLAS) and torch's own sum(1) of each level (its
    addition order depends on the shape), with the gather of the level's
    rows as one kernel into a preallocated zero padded buffer."""
    s = torch.bmm(xb.view(tree.idx.shape).unsqueeze(1), tree.U).squeeze(1)
    if not hasattr(tree, '_fast_levels'):
        lv = []
        pad = tree.idx.shape[0]
        for ix in tree.levels:
            valid = (ix != pad).reshape(-1)
            flat = torch.arange(ix.numel(), device=ix.device)[valid]
            lv.append([ix.shape, ix.reshape(-1)[valid].contiguous(), flat.contiguous(), None])
            pad = ix.shape[0]
        tree._fast_levels = lv
    for ent in tree._fast_levels:
        shape, src, dst, buf = ent
        if buf is None or buf.shape[2] != s.shape[1]:
            buf = ent[3] = torch.zeros(shape + (s.shape[1],), dtype=s.dtype,
                                       device=s.device)
        K.gather(s.contiguous(), src, dst, buf)
        s = buf.sum(1)
    return s


class FastStepRouter(routing.StepRouter):
    @classmethod
    def from_router(cls, r):
        if r.dev.type != 'cuda':
            return r
        o = object.__new__(cls)
        o.__dict__.update(r.__dict__)
        o.lk = [_level_data(lv) for lv in o.levels]
        o._dtt = torch.tensor([o.dt, 1.0 / o.dt], dtype=torch.float64, device=o.dev)
        o._lk = {id(lv): k for lv, k in zip(o.levels, o.lk)}
        o._xb = torch.zeros(o.tree.idx_flat.numel(), dtype=o.dtype, device=o.dev)
        return o

    def begin(self, runoff):
        K.begin_x(runoff.contiguous(), self.cell, self.w, self.tree.idx_flat,
                  self._xb, self.n_pair)
        t = blocked(self.tree, self._xb)
        K.update(self.ring, self.head, self.L, self.tree.ids, t,
                 self.in_runoff, self._dtt)

    def _apply_level(self, lv, vol):
        k = self._lk[id(lv)]
        K.xfill(vol.contiguous(), k, self._dtt)
        t = blocked(lv['tree'], k['xpad'])
        K.update(self.ring, self.head, self.L, lv['tree'].ids, t,
                 self.in_hooks, self._dtt)

    def _cap(self, lv):
        k = self._lk[id(lv)]
        return K.cap(self.ring, self.head, self.L, k, self._dtt)

    def hooks(self, withdraw=None, inject=None):
        got = torch.zeros(self.n_site, dtype=self.dtype, device=self.dev)
        wd = None if withdraw is None else withdraw.contiguous()
        ij = None if inject is None else inject.contiguous()
        for lv, k in zip(self.levels, self.lk):
            K.level(self.ring, self.head, self.L, k, wd, ij, got,
                    self.withdrawn, self.injected, self._dtt)
            t = blocked(lv['tree'], k['xpad'])
            K.update(self.ring, self.head, self.L, lv['tree'].ids, t,
                     self.in_hooks, self._dtt)
        return got

    def end(self):
        return K.end(self.ring, self.head, self.out, self.L, self._dtt)


def build_fast_step_router(*args, **kw):
    """`river_network.build_step_router`, returning a FastStepRouter on CUDA."""
    r, info = rn.build_step_router(*args, **kw)
    return FastStepRouter.from_router(r), info
