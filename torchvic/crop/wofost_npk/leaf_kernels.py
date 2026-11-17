# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The leaf-class sums of WOFOST-NPK (Lai.c, Leaves.c) on a ring of leaf
classes, in eager CUDA's own addition order.

The default (fast) kernel sums the leaf classes of a unit as float64
reductions over its row of K classes, the dead ones masked to 0.0: a
torch `sum(-1)` of a [rows, K] float64 tensor (LAI, death by age) and a
`cumsum(-1)` (death by stress). Their rounding follows eager's kernels:

* sum (Reduce.cuh, K > 128, K % 4 == 0, rows >= 16): 32 threads share a
  row; thread x reads the vectors of 4 elements x, x + 32, x + 64, ...
  (elements 4v .. 4v + 3), element i of each vector into accumulator i
  (each starting from 0.0, in turn); the accumulators are added
  ((a0 + a1) + a2) + a3, and the 32 threads' values by the warp shuffle
  tree (offsets 1, 2, 4, ...): pairwise, neighbours first.
* cumsum (ScanUtils.cuh): the row in chunks of CH elements (CH from the
  row count and length, core.xscan.log_threads_x); the first element of
  a chunk gets the running total of the chunks before (+ 0.0 for the
  first), then a Sklansky scan: element i is x_i plus the pairwise totals
  of the aligned blocks of the set bits of its index in the chunk,
  smallest first (core.xscan).

Here the same trees run on the live classes only: every other element is
0.0, and a sum that starts from +0.0 is never -0.0, so adding a known 0.0
changes nothing (as in core.xscan.segment_sum). That lets the classes live
in a ring of R <= K slots (class k in slot k mod R) and a kernel read only
the live ones. Checked against torch.sum / torch.cumsum (tests/
test_wofost_leaf_kernels.py) and against the [rows, K] kernel on the ISC.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:          # pragma: no cover
    triton = None

from ...core import xscan

#: units a program of the sum kernel (a [BU, 32, 4] tile per 128 classes)
BU_SUM = 16
#: units a program of the scan kernel (one unit a lane)
BU_SCAN = 128


def supported(K, rows):
    """Whether eager's sum over [rows, K] float64 is the tree above."""
    return triton is not None and K > 128 and K % 4 == 0 and rows >= 16


if triton is not None:
    @triton.jit
    def _ring_sum_kernel(a_ptr, b_ptr, lo_ptr, hi_ptr, out_ptr, U, R,
                         K: tl.constexpr, PROD: tl.constexpr,
                         BU: tl.constexpr):
        u = tl.program_id(0).to(tl.int64) * BU + tl.arange(0, BU)
        ok = u < U
        lo = tl.load(lo_ptr + u, mask=ok, other=0)[:, None, None]
        hi = tl.load(hi_ptr + u, mask=ok, other=0)[:, None, None]
        base = (u * R)[:, None, None]
        okk = ok[:, None, None]
        p0 = (tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :])
        p0 = p0[None, :, :]
        acc = tl.zeros([BU, 32, 4], dtype=tl.float64)
        # the 128-class blocks some unit of the program has live classes in
        has = ok & (tl.reshape(hi, (BU,)) > tl.reshape(lo, (BU,)))
        j0 = tl.min(tl.where(has, tl.reshape(lo, (BU,)), K), 0) // 128
        j1 = (tl.max(tl.where(has, tl.reshape(hi, (BU,)), 0), 0) + 127) // 128
        for j in range(j0, j1):
            p = p0 + j * 128
            m = okk & (p >= lo) & (p < hi) & (p < K)
            slot = p % R
            a = tl.load(a_ptr + base + slot, mask=m, other=0.0)
            if PROD:
                b = tl.load(b_ptr + base + slot, mask=m, other=0.0)
                v = (a * b).to(tl.float64)
            else:
                v = a.to(tl.float64)
            acc = acc + tl.where(m, v, 0.0)
        e, o = tl.split(tl.reshape(acc, (BU, 32, 2, 2)))
        a0, a2 = tl.split(e)
        a1, a3 = tl.split(o)
        val = ((a0 + a1) + a2) + a3                      # [BU, 32]
        val = tl.reshape(val, (BU, 16, 2))
        x0, x1 = tl.split(val)
        val = tl.reshape(x0 + x1, (BU, 8, 2))
        x0, x1 = tl.split(val)
        val = tl.reshape(x0 + x1, (BU, 4, 2))
        x0, x1 = tl.split(val)
        val = tl.reshape(x0 + x1, (BU, 2, 2))
        x0, x1 = tl.split(val)
        val = tl.reshape(x0 + x1, (BU, 2))
        x0, x1 = tl.split(val)
        tl.store(out_ptr + u, x0 + x1, mask=ok)

    @triton.jit
    def _scan_step(i, h, t, lim, w_ptr, base, R, carry, s0, s1, s2, s3, s4,
                   s5, s6, s7, s8, s9, ch):
        """Element i of the lanes' scans: (cs, x, active), cs_i as
        eager's Sklansky tree from the block totals s<m> of the chunk."""
        act = (i >= h) & (i < lim)
        x = tl.load(w_ptr + base + i % R, mask=act, other=0.0)
        x = x.to(tl.float64)
        l = i % ch
        x = tl.where(l == 0, x + carry, x)
        cs = x
        cs = tl.where((l & 1) != 0, cs + s0, cs)
        cs = tl.where((l & 2) != 0, cs + s1, cs)
        cs = tl.where((l & 4) != 0, cs + s2, cs)
        cs = tl.where((l & 8) != 0, cs + s3, cs)
        cs = tl.where((l & 16) != 0, cs + s4, cs)
        cs = tl.where((l & 32) != 0, cs + s5, cs)
        cs = tl.where((l & 64) != 0, cs + s6, cs)
        cs = tl.where((l & 128) != 0, cs + s7, cs)
        cs = tl.where((l & 256) != 0, cs + s8, cs)
        cs = tl.where((l & 512) != 0, cs + s9, cs)
        return cs, x, act

    @triton.jit
    def _push(l, x, act, s, m: tl.constexpr, on):
        """One bit of the binary counter: merge (newer + older) while bit
        m of l is set, store at the first clear bit."""
        b = ((l >> m) & 1) != 0
        take = on & b
        put = on & (~b)
        merged = tl.where(take, x + s, x)
        s = tl.where(put, x, s)
        return merged, s, take

    @triton.jit
    def _ring_scan_kernel(w_ptr, h_ptr, t_ptr, d_ptr, ch_ptr, m_ptr, c_ptr,
                          U, R, BU: tl.constexpr):
        """m = #{k in [h, t): d > cs_k}, csm = cs_{h + m - 1} (0 if m = 0)
        of cs = cumsum of the row (eager's order), the row being w on
        [h, t) and 0.0 elsewhere."""
        u = tl.program_id(0).to(tl.int64) * BU + tl.arange(0, BU)
        ok = u < U
        h = tl.load(h_ptr + u, mask=ok, other=0)
        t = tl.load(t_ptr + u, mask=ok, other=0)
        t = tl.where(ok, t, h)
        d = tl.load(d_ptr + u, mask=ok, other=0.0)
        ch = tl.load(ch_ptr + u, mask=ok, other=1).to(tl.int64)
        base = u * R
        lo = tl.min(tl.where(ok & (t > h), h, 2147483647), 0)
        hi = tl.max(tl.where(ok, t, 0), 0)
        z = tl.zeros([BU], dtype=tl.float64)
        cnt = tl.zeros([BU], dtype=tl.int64)
        csm = z
        for ps in tl.static_range(2):
            if ps == 0:
                lim = t
            else:
                lim = h + cnt
            s0 = z
            s1 = z
            s2 = z
            s3 = z
            s4 = z
            s5 = z
            s6 = z
            s7 = z
            s8 = z
            s9 = z
            carry = z
            for i in range(lo, hi):
                l = i % ch
                fresh = l == 0
                s0 = tl.where(fresh, 0.0, s0)
                s1 = tl.where(fresh, 0.0, s1)
                s2 = tl.where(fresh, 0.0, s2)
                s3 = tl.where(fresh, 0.0, s3)
                s4 = tl.where(fresh, 0.0, s4)
                s5 = tl.where(fresh, 0.0, s5)
                s6 = tl.where(fresh, 0.0, s6)
                s7 = tl.where(fresh, 0.0, s7)
                s8 = tl.where(fresh, 0.0, s8)
                s9 = tl.where(fresh, 0.0, s9)
                cs, x, act = _scan_step(i, h, t, lim, w_ptr, base, R, carry,
                                        s0, s1, s2, s3, s4, s5, s6, s7, s8,
                                        s9, ch)
                if ps == 0:
                    cnt = cnt + (act & (d > cs)).to(tl.int64)
                else:
                    csm = tl.where(act & (i == lim - 1), cs, csm)
                carry = tl.where(act, cs, carry)
                on = act
                x, s0, on = _push(l, x, act, s0, 0, on)
                x, s1, on = _push(l, x, act, s1, 1, on)
                x, s2, on = _push(l, x, act, s2, 2, on)
                x, s3, on = _push(l, x, act, s3, 3, on)
                x, s4, on = _push(l, x, act, s4, 4, on)
                x, s5, on = _push(l, x, act, s5, 5, on)
                x, s6, on = _push(l, x, act, s6, 6, on)
                x, s7, on = _push(l, x, act, s7, 7, on)
                x, s8, on = _push(l, x, act, s8, 8, on)
                x, s9, on = _push(l, x, act, s9, 9, on)
        tl.store(m_ptr + u, cnt, mask=ok)
        tl.store(c_ptr + u, csm, mask=ok)


@torch.library.custom_op("torchvic::wofost_ring_sum", mutates_args=())
def _ring_sum_op(a: torch.Tensor, b: torch.Tensor, lo: torch.Tensor,
                 hi: torch.Tensor, K: int, prod: bool) -> torch.Tensor:
    U, R = a.shape
    out = torch.empty(U, dtype=torch.float64, device=a.device)
    if U:
        _ring_sum_kernel[((U + BU_SUM - 1) // BU_SUM,)](
            a.contiguous(), b.contiguous(), lo.contiguous(), hi.contiguous(),
            out, U, R, K=K, PROD=prod, BU=BU_SUM, num_warps=4,
            enable_fp_fusion=False)
    return out


@_ring_sum_op.register_fake
def _(a, b, lo, hi, K, prod):
    return a.new_empty(a.shape[:1], dtype=torch.float64)


@torch.library.custom_op("torchvic::wofost_ring_scan", mutates_args=())
def _ring_scan_op(w: torch.Tensor, h: torch.Tensor, t: torch.Tensor,
                  d: torch.Tensor, ch: torch.Tensor) -> tuple[torch.Tensor,
                                                             torch.Tensor]:
    U, R = w.shape
    m = torch.empty(U, dtype=torch.int64, device=w.device)
    c = torch.empty(U, dtype=torch.float64, device=w.device)
    if U:
        _ring_scan_kernel[((U + BU_SCAN - 1) // BU_SCAN,)](
            w.contiguous(), h.contiguous(), t.contiguous(), d.contiguous(),
            ch.contiguous(), m, c, U, R, BU=BU_SCAN, num_warps=4,
            enable_fp_fusion=False)
    return m, c


@_ring_scan_op.register_fake
def _(w, h, t, d, ch):
    return (w.new_empty(w.shape[:1], dtype=torch.int64),
            w.new_empty(w.shape[:1], dtype=torch.float64))


def ring_sum(a, lo, hi, K, b=None):
    """[U] float64: eager's `where(live, f64(a [* b]), 0).sum(-1)` over the
    [U, K] row of classes, live = [lo, hi), a (b) [U, R] float32 rings
    (the product a * b in float32 first, as Lai.c)."""
    return _ring_sum_op(a, a if b is None else b, lo.to(torch.int64),
                        hi.to(torch.int64), int(K), b is not None)


def scan_chunk(K, rows):
    """The chunk of eager's cumsum over a [rows, K] tensor."""
    return xscan._chunk(int(rows), int(K))


def ring_scan(w, h, t, d, ch):
    """(m, csm): with cs = eager's cumsum(-1) of where(live, f64(w), 0)
    over the [rows, K] tensor (live = [h, t)), m = #{live k: d > cs_k}
    and csm = cs[h + m - 1] (0.0 where m = 0); d [U] float64; ch [U] the
    scan chunk of each unit's tensor (scan_chunk(K, rows), <= 1024)."""
    if not torch.is_tensor(ch):
        ch = torch.full(w.shape[:1], int(ch), dtype=torch.int32,
                        device=w.device)
    return _ring_scan_op(w, h.to(torch.int64), t.to(torch.int64),
                         d.to(torch.float64).contiguous(), ch)


# ------------------------------------------------------------ ring updates
# The leaf classes of the lanes' ring updated in place, masked by the units
# whose day keeps them (the emerging ones for the initial class, the growing
# ones for the growth day): the same values as the [U, R] selections of
# the row layout, without passing over every slot. Run eagerly (a fused
# region calls them between its parts).

if triton is not None:
    @triton.jit
    def _ring_young_kernel(a_ptr, h_ptr, t_ptr, s_ptr, o_ptr, U, R,
                           BU: tl.constexpr):
        """o = the first k in [h, t) with not (a[k] > s), else t."""
        u = tl.program_id(0).to(tl.int64) * BU + tl.arange(0, BU)
        ok = u < U
        h = tl.load(h_ptr + u, mask=ok, other=0)
        t = tl.load(t_ptr + u, mask=ok, other=0)
        t = tl.where(ok, t, h)
        sp = tl.load(s_ptr + u, mask=ok, other=0.0)
        lo = tl.min(tl.where(ok & (t > h), h, 2147483647), 0)
        hi = tl.max(tl.where(ok, t, 0), 0)
        fb = t
        found = h >= t
        for i in range(lo, hi):
            act = (i >= h) & (i < t) & (~found)
            a = tl.load(a_ptr + u * R + i % R, mask=act, other=0.0)
            young = act & (~(a > sp))
            fb = tl.where(young, i, fb)
            found = found | young
        tl.store(o_ptr + u, fb, mask=ok)

    @triton.jit
    def _ring_age_kernel(a_ptr, h_ptr, t_ptr, p_ptr, m_ptr, U, R,
                         BU: tl.constexpr):
        """a[k] = a[k] + p for k in [h, t - 1) where m (in place)."""
        u = tl.program_id(0).to(tl.int64) * BU + tl.arange(0, BU)
        ok = u < U
        m = tl.load(m_ptr + u, mask=ok, other=0) != 0
        h = tl.load(h_ptr + u, mask=ok, other=0)
        t = tl.load(t_ptr + u, mask=ok, other=0) - 1
        on = ok & m & (t > h)
        t = tl.where(on, t, h)
        p = tl.load(p_ptr + u, mask=ok, other=0.0)
        lo = tl.min(tl.where(on, h, 2147483647), 0)
        hi = tl.max(tl.where(on, t, 0), 0)
        for i in range(lo, hi):
            act = on & (i >= h) & (i < t)
            ptr = a_ptr + u * R + i % R
            a = tl.load(ptr, mask=act, other=0.0)
            tl.store(ptr, a + p, mask=act)


@torch._dynamo.disable
def ring_young(lage, h, t, span):
    """[U] long: the first class k in [h, t) of each unit's ring whose age
    is not above span ([U], the class's LifeSpan), else t."""
    U, R = lage.shape
    out = torch.empty(U, dtype=torch.int64, device=lage.device)
    if U:
        _ring_young_kernel[((U + BU_SCAN - 1) // BU_SCAN,)](
            lage, h.to(torch.int64).contiguous(),
            t.to(torch.int64).contiguous(),
            span.to(lage.dtype).contiguous(), out, U, R, BU=BU_SCAN,
            num_warps=4, enable_fp_fusion=False)
    return out


@torch._dynamo.disable
def ring_age_(lage, h, t, pa, mask):
    """lage[k] += pa for the classes k in [h, t - 1) of the units in mask
    (float32 additions, as `lage + pa` of the row layout), in place."""
    U, R = lage.shape
    if U:
        _ring_age_kernel[((U + BU_SCAN - 1) // BU_SCAN,)](
            lage, h.to(torch.int64).contiguous(),
            t.to(torch.int64).contiguous(),
            pa.to(lage.dtype).contiguous(), mask.to(torch.int8).contiguous(),
            U, R, BU=BU_SCAN, num_warps=4, enable_fp_fusion=False)
    return lage


@torch._dynamo.disable
def ring_put_(rings, slot, mask, values):
    """ring[u, slot[u]] = value[u] for the units in mask, each ring of
    `rings` with its value of `values` (a tensor [U] or a number), in
    place. A value given as ('sub', d) is ring[u, slot[u]] - d[u]."""
    U, R = rings[0].shape
    idx = torch.arange(U, device=slot.device) * R + slot.to(torch.int64)
    for r, v in zip(rings, values):
        flat = r.view(-1)
        old = flat.index_select(0, idx)
        if isinstance(v, tuple):
            new = old - v[1].to(r.dtype)
        elif torch.is_tensor(v):
            new = v.to(r.dtype)
        else:
            new = torch.full_like(old, v)
        flat.index_copy_(0, idx, torch.where(mask, new, old))
    return rings
