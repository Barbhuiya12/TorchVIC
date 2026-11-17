# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""index_add in the CPU's order on every device.

The CPU's index_add adds the rows of `src` into `out` in index order: row i
before row j > i into the same slot, starting from what `out` holds. On a
GPU index_add adds with atomics, in whatever order the threads arrive, so
wherever two rows meet one slot the same run differs from run to run in
the last bits (on ISC: the band columns' runoff summed onto the
cells, the routing's basin areas, MODFLOW's river and drain flows -- and
the whole coupled run with them).

Here, on a GPU, pass k adds the k-th row of every slot (one row a slot a
pass, so no two adds meet): the CPU's additions, bit for bit, every run.
The passes of an index tensor are planned once (one host read) and kept
while the tensor lives; a plan made before a CUDA graph's capture is
replayed inside it without a host read."""
import weakref

import torch

#: id(index) -> (weak reference, plan). A plan is ('pass', [(rows,
#: slots), ...]) -- pass k adds the k-th row of every slot -- or, where
#: one slot takes many rows (a basin's single outlet: 291,273 of them on
#: ISC), ('seg', touched slots, gather index, offsets): every touched
#: slot's current value then its rows in order, one sequential segment sum
#: (model.seqsum, segment_reduce's own order) a slot
_PLANS = {}
#: the most rows into one slot that the passes take
MAX_PASSES = 8


def _plan(idx):
    hit = _PLANS.get(id(idx))
    if hit is not None and hit[0]() is idx:
        return hit[1]
    order = torch.sort(idx, stable=True).indices
    sidx = idx.index_select(0, order)
    first = torch.ones_like(sidx, dtype=torch.bool)
    first[1:] = sidx[1:] != sidx[:-1]
    pos = torch.arange(sidx.numel(), device=idx.device)
    start = torch.cummax(torch.where(first, pos, torch.zeros_like(pos)),
                         0).values
    rank = pos - start
    K = int(rank.max()) + 1 if rank.numel() else 0
    if K <= MAX_PASSES:
        passes = []
        for k in range(K):
            at = (rank == k).nonzero().flatten()
            passes.append((order.index_select(0, at),
                           sidx.index_select(0, at)))
        plan = ('pass', passes)
    else:
        u = sidx.index_select(0, first.nonzero().flatten())     # [m]
        m, n = int(u.numel()), int(sidx.numel())
        seg = torch.cumsum(first.long(), 0) - 1                 # [n]
        # segment j: its slot's value (row j of the touched values), then
        # its rows (m + their position in src)
        dst = pos + seg + 1
        gidx = torch.empty(m + n, dtype=torch.long, device=idx.device)
        gidx[dst] = m + order
        head = start.index_select(0, first.nonzero().flatten()) \
            + torch.arange(m, device=idx.device)
        gidx[head] = torch.arange(m, device=idx.device)
        offs = torch.cat([head, head.new_full((1,), m + n)])
        plan = ('seg', u, gidx, offs)
    if len(_PLANS) > 4096:
        for key in [k for k, v in _PLANS.items() if v[0]() is None]:
            del _PLANS[key]
    try:
        _PLANS[id(idx)] = (weakref.ref(idx), plan)
    except TypeError:
        pass
    return plan


def index_add_(out, dim, idx, src):
    """out.index_add_(dim, idx, src), the rows added in index order."""
    if out.device.type == 'cpu' or idx.numel() == 0:
        return out.index_add_(dim, idx, src)
    plan = _plan(idx)
    if plan[0] == 'pass':
        for rows, slots in plan[1]:
            out.index_copy_(dim, slots, out.index_select(dim, slots)
                            + src.index_select(dim, rows))
        return out
    _, u, gidx, offs = plan
    o = out.movedim(dim, 0)
    x = torch.cat([o.index_select(0, u), src.movedim(dim, 0)
                   .to(out.dtype)], 0).index_select(0, gidx)
    x2 = x.reshape(x.shape[0], -1)
    from ..model import seqsum
    if seqsum.available(x2):
        sums = seqsum.segment_sum(x2, offs)
    else:
        sums = torch.segment_reduce(x2, 'sum', offsets=offs, axis=0,
                                    unsafe=True)
    o.index_copy_(0, u, sums.reshape((-1,) + tuple(o.shape[1:])))
    return out

def index_add(out, dim, idx, src):
    """out.index_add(dim, idx, src) (a new tensor), in index order."""
    if out.device.type == 'cpu' or idx.numel() == 0:
        return out.index_add(dim, idx, src)
    return index_add_(out.clone(), dim, idx, src)
