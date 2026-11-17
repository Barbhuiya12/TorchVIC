# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Triton kernel of the river exchange slot (`river_exchange.py`): one launch
per routing level, the bisection of every river cell of the level in one
thread, the ring's discharge bins edited in place."""
try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:                       # CPU-only torch has no Triton
    from ..core.triton_stub import triton, tl, libdevice


@triton.jit
def _stage(q, bot, den, cap, man, ex, dmin):
    d1 = libdevice.pow(man * q / den, ex)
    d = tl.where(d1 < dmin, 0.0, d1)
    return bot + tl.minimum(d, cap)


@triton.jit
def kernel(cells_ptr, ring_ptr, head_ptr, dis_ptr, c_ptr, bot_ptr, den_ptr,
           cap_ptr, rbot_ptr, h_ptr, debt_ptr, t_ptr, stg_ptr, unmet_ptr,
           nc, ncell, cst_ptr,
           R: tl.constexpr, NSUB: tl.constexpr, NIT: tl.constexpr,
           BLOCK: tl.constexpr):
    # the constants are read as float64 (a Python float in a kernel is
    # float32): Manning's n, the 3/5 power, the 1 cm floor, the step (s)
    man = tl.load(cst_ptr)
    ex = tl.load(cst_ptr + 1)
    dmin = tl.load(cst_ptr + 2)
    step_s = tl.load(cst_ptr + 3)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < nc
    c = tl.load(cells_ptr + i, mask=m, other=0)
    hd = tl.load(head_ptr)
    Q = tl.zeros([BLOCK], tl.float64)
    for k in tl.static_range(NSUB):
        p = hd + k
        p = tl.where(p >= R, p - R, p)
        Q += tl.load(ring_ptr + p * ncell + c, mask=m, other=0.0)
    Q = tl.maximum(Q, 0.0)
    C = tl.load(c_ptr + c, mask=m, other=0.0)
    bot = tl.load(bot_ptr + c, mask=m, other=0.0)
    den = tl.load(den_ptr + c, mask=m, other=1.0)
    cap = tl.load(cap_ptr + c, mask=m, other=0.0)
    hh = tl.maximum(tl.load(h_ptr + c, mask=m, other=0.0),
                    tl.load(rbot_ptr + c, mask=m, other=0.0))
    debt = tl.load(debt_ptr + c, mask=m, other=0.0)
    lo = C * (bot - hh)
    dry = lo >= Q
    a = lo
    b = Q
    for _ in range(NIT):
        mid = 0.5 * (a + b)
        g = mid - C * (_stage(tl.maximum(Q - mid, 0.0), bot, den, cap, man, ex, dmin)
                       - hh)
        up = g > 0
        b = tl.where(up, mid, b)
        a = tl.where(up, a, mid)
    root = tl.where(dry, lo, 0.5 * (a + b))
    want = root + debt / step_s
    T = tl.minimum(want, Q)
    unmet = tl.maximum(want - Q, 0.0)
    qf = tl.maximum(Q - T, 0.0)
    stg = _stage(qf, bot, den, cap, man, ex, dmin)
    for k in tl.static_range(NSUB):
        p = hd + k
        p = tl.where(p >= R, p - R, p)
        q = tl.load(ring_ptr + p * ncell + c, mask=m, other=0.0)
        # a discharge under 1e-12 m3 s-1 (bins can hold denormals) is none:
        # dividing by it overflows
        pos = Q > 1e-12
        q = tl.where(pos, q * (1.0 - T / tl.where(pos, Q, 1.0)),
                     tl.maximum(q - T / NSUB, 0.0))
        tl.store(ring_ptr + p * ncell + c, q, mask=m)
    tl.store(dis_ptr + c, Q - T, mask=m)
    tl.store(t_ptr + c, T, mask=m)
    tl.store(stg_ptr + c, stg, mask=m)
    tl.store(unmet_ptr + c, unmet, mask=m)
