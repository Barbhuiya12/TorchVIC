"""The in-order canopy removal of the clipping biomass fate (clipping._walk_torch) as one per-tile
Triton kernel (CUDA): one lane walks its tile's portions newest first or oldest first, as
Canopy_accumulation::remove_biomass pops them (canopy_growth_portioned.cpp:103-160). Only + - * /,
a float32 rounding and comparisons, contraction off: the same arithmetic as the torch walk.
A lane without a removal loads nothing. TORCHVIC_CROPSYST_KERNELS=0 keeps the torch form.
"""
from typing import List

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                               # pragma: no cover
    triton = None

from .om_kernels import KERNELS


def enabled() -> bool:
    return KERNELS and triton is not None


if triton is not None:
    @triton.jit
    def _walk_kernel(pg_ptr, pd_ptr, pb_ptr, pgb_ptr, cnt_ptr, amt_ptr, ret_ptr, av_ptr, new_ptr, m_ptr,
                     act_ptr, mg_ptr, N, R: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = r < N
        amount = tl.load(amt_ptr + r, mask=ok, other=0.0)
        d = 0.0 - amount
        d = tl.where(d < 0.0, -d, d)
        m = (tl.load(m_ptr + r, mask=ok, other=0) != 0) & ok & (d > 0.0000001)
        alive = m & (amount > 0.0)
        act = tl.zeros([BLOCK], dtype=tl.float64)
        margin = tl.full([BLOCK], float("inf"), tl.float64)
        if tl.max(alive.to(tl.int32), axis=0) > 0:
            cnt = tl.load(cnt_ptr + r, mask=m, other=0)
            retain = tl.load(ret_ptr + r, mask=m, other=0.0)
            avail = tl.load(av_ptr + r, mask=m, other=0.0)
            flags = tl.load(new_ptr + r, mask=m, other=0)
            newest = (flags & 1) != 0
            lose = (flags & 2) != 0
            more = amount
            for k in range(0, R):
                idx = tl.where(newest, cnt - 1 - k, k)
                valid = alive & (idx >= 0) & (idx < cnt)
                off = r * R + idx
                b = tl.load(pb_ptr + off, mask=valid, other=0.0)
                L = tl.load(pg_ptr + off, mask=valid, other=0.0)
                keep = (avail - L) < retain
                dm = (avail - L) - retain
                dm = tl.where(dm < 0.0, -dm, dm)
                margin = tl.where(valid & (dm < margin), dm, margin)
                take = valid & (keep == 0)
                drop = valid & keep
                whole = take & (more > b)
                part = take & (more <= b)
                lt = (L * (more / tl.where(part, b, 1.0))).to(tl.float32).to(tl.float64)
                dec = tl.where(b > more, more, b)
                avail = tl.where(whole, avail - L, tl.where(part, avail - lt, avail))
                act = tl.where(whole, act + b, tl.where(part, act + dec, act))
                more = tl.where(whole, more - b, tl.where(part, more - dec, more))
                gone = whole | (drop & lose)
                tl.store(pg_ptr + off, tl.where(gone, 0.0, L - lt), mask=gone | part)
                tl.store(pb_ptr + off, tl.where(gone, 0.0, b - dec), mask=gone | part)
                tl.store(pd_ptr + off, 0.0, mask=gone)
                tl.store(pgb_ptr + off, 0.0, mask=gone)
                alive = valid & (more > 0.0)
        tl.store(act_ptr + r, act, mask=ok)
        tl.store(mg_ptr + r, margin, mask=ok)


@torch.library.custom_op("torchvic::cs_clip_walk", mutates_args=())
def _walk_op(pg: torch.Tensor, pd: torch.Tensor, pb: torch.Tensor, pgb: torch.Tensor, cnt: torch.Tensor,
             amount: torch.Tensor, retain: torch.Tensor, avail: torch.Tensor, newest: torch.Tensor,
             mask: torch.Tensor) -> List[torch.Tensor]:
    pg, pd, pb, pgb = pg.clone(), pd.clone(), pb.clone(), pgb.clone()
    T, R = pb.shape
    act = torch.zeros((T,), dtype=pb.dtype, device=pb.device)
    mg = torch.full((T,), float("inf"), dtype=pb.dtype, device=pb.device)
    if T:
        BLOCK = 128
        _walk_kernel[((T + BLOCK - 1) // BLOCK,)](pg, pd, pb, pgb, cnt.contiguous(), amount.contiguous(),
                                                  retain.contiguous(), avail.contiguous(),
                                                  newest.to(torch.int8).contiguous(), mask.to(torch.int8).contiguous(),
                                                  act, mg, T, R=R, BLOCK=BLOCK, num_warps=4, enable_fp_fusion=False)
    return [pg, pd, pb, pgb, act, mg]


@_walk_op.register_fake
def _(pg, pd, pb, pgb, cnt, amount, retain, avail, newest, mask):
    return [torch.empty_like(pg), torch.empty_like(pd), torch.empty_like(pb), torch.empty_like(pgb),
            pb.new_empty(pb.shape[:1]), pb.new_empty(pb.shape[:1])]


def walk(pg, pd, pb, pgb, cnt, amount, retain, avail, newest, mask, lose=True):
    T = pb.shape[0]
    bc = lambda x: torch.broadcast_to(x, (T,)).contiguous()
    out = _walk_op(pg.contiguous(), pd.contiguous(), pb.contiguous(), pgb.contiguous(), bc(cnt), bc(amount),
                   bc(retain), bc(avail), bc(newest).to(torch.int8) + 2 * int(bool(lose)), bc(mask))
    return tuple(out)
