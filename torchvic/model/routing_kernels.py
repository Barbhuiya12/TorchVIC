"""Triton kernels for the stepping router (few launches per hook level).

Each kernel does the same IEEE operations, in the same order, as the torch
ops it replaces in `StepRouter` (min, clamp, divide, add, one add per ring
entry), so results are bit-identical; only the launch count changes.  The
matrix products of the tree sums stay cuBLAS (bmm).  All kernels have
device-resident scalars only (no host sync) and are CUDA-graph capturable.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:                       # CPU-only torch has no Triton
    from ..core.triton_stub import triton, tl

@triton.jit
def gather_kernel(s_ptr, F, src_ptr, dst_ptr, buf_ptr, BF: tl.constexpr):
    """buf[dst[v], :] = s[src[v], :] for the valid entries of the level index
    (the padding of buf stays zero), so torch's sum(1) sees the same tensor
    `cat([s, 0])[ix]` gave."""
    v = tl.program_id(0)
    cols = tl.program_id(1) * BF + tl.arange(0, BF)
    m = cols < F
    row = tl.load(src_ptr + v)
    d = tl.load(dst_ptr + v)
    tl.store(buf_ptr + d * F + cols, tl.load(s_ptr + row * F + cols, mask=m, other=0.0), mask=m)


@triton.jit
def treered_kernel(s_ptr, F, ix_ptr, cnt_ptr, out_ptr, BF: tl.constexpr):
    """out[r] = sum of s[ix[r, k]] for the first cnt[r] entries k, in four
    interleaved accumulators (the order is not eager's: for channels that
    need not match the water to the last bit; only valid entries are read)."""
    r = tl.program_id(0)
    cols = tl.program_id(1) * BF + tl.arange(0, BF)
    cm = cols < F
    n = tl.load(cnt_ptr + r)
    a0 = tl.full([BF], 0.0, tl.float64)
    a1 = tl.full([BF], 0.0, tl.float64)
    a2 = tl.full([BF], 0.0, tl.float64)
    a3 = tl.full([BF], 0.0, tl.float64)
    for k in range(0, n, 4):
        i0 = tl.load(ix_ptr + r * 64 + k)
        a0 = a0 + tl.load(s_ptr + i0 * F + cols, mask=cm, other=0.0)
        if k + 1 < n:
            i1 = tl.load(ix_ptr + r * 64 + k + 1)
            a1 = a1 + tl.load(s_ptr + i1 * F + cols, mask=cm, other=0.0)
        if k + 2 < n:
            i2 = tl.load(ix_ptr + r * 64 + k + 2)
            a2 = a2 + tl.load(s_ptr + i2 * F + cols, mask=cm, other=0.0)
        if k + 3 < n:
            i3 = tl.load(ix_ptr + r * 64 + k + 3)
            a3 = a3 + tl.load(s_ptr + i3 * F + cols, mask=cm, other=0.0)
    tl.store(out_ptr + r * F + cols, ((a0 + a1) + a2) + a3, mask=cm)


@triton.jit
def _mul(a, b):
    """a * b rounded on its own (mul.rn): stops the compiler fusing it into
    an FMA with a following add, which torch's separate ops never do."""
    return tl.inline_asm_elementwise("mul.rn.f64 $0, $1, $2;", "=d,d,d", [a, b],
                                     dtype=tl.float64, is_pure=True, pack=1)


@triton.jit
def _div(a, b):
    """Correctly rounded a / b (div.rn), as torch's fp64 divide."""
    return tl.inline_asm_elementwise("div.rn.f64 $0, $1, $2;", "=d,d,d", [a, b],
                                     dtype=tl.float64, is_pure=True, pack=1)


BL = 128                    # lag chunk of the cap loop


@triton.jit
def _site_cap(ring_ptr, rs, head_ptr, L, pstart_ptr, pnode_ptr, kern_ptr, s,
              dt, BL: tl.constexpr):
    """Most volume removable at local site s with no negative ring slot."""
    head = tl.load(head_ptr)
    lo = tl.load(pstart_ptr + s)
    hi = tl.load(pstart_ptr + s + 1)
    mv = tl.full([BL], float('inf'), tl.float64)
    offs = tl.arange(0, BL)
    for q in range(lo, hi):
        node = tl.load(pnode_ptr + q)
        for l0 in range(0, L, BL):
            l = l0 + offs
            m = l < L
            slot = (head + l) % L
            r = tl.load(ring_ptr + slot * rs + node, mask=m, other=0.0)
            r = tl.maximum(r, 0.0)
            k = tl.load(kern_ptr + q * L + l, mask=m, other=0.0)
            ks = tl.maximum(k, tl.full([BL], 1e-300, tl.float64))
            ratio = tl.where(m & (k > 0), _div(r, ks), float('inf'))
            mv = tl.minimum(mv, ratio)
    return tl.min(mv, 0) * dt


@triton.jit
def cap_kernel(ring_ptr, rs, head_ptr, L, pstart_ptr, pnode_ptr, kern_ptr,
               cap_ptr, dt_ptr, BL: tl.constexpr):
    dt = tl.load(dt_ptr)
    s = tl.program_id(0)
    tl.store(cap_ptr + s, _site_cap(ring_ptr, rs, head_ptr, L, pstart_ptr,
                                    pnode_ptr, kern_ptr, s, dt, BL))


@triton.jit
def level_kernel(ring_ptr, rs, head_ptr, L, pstart_ptr, pnode_ptr, kern_ptr,
                 pos_ptr, sites_ptr, wd_ptr, inj_ptr, got_ptr, wcount_ptr,
                 icount_ptr, xpad_ptr, dt_ptr, snode_ptr, ra_ptr, rsa, csa, wda_ptr,
                 nsa, C: tl.constexpr,
                 HAS_W: tl.constexpr, HAS_I: tl.constexpr, BL: tl.constexpr):
    dt = tl.load(dt_ptr)
    """Per site of a level: take = min(request, cap), put = inject, write
    (put - take)/dt into the blocked input of the tree sum, update counters."""
    s = tl.program_id(0)
    sid = tl.load(sites_ptr + s)
    g = tl.full((), 0.0, tl.float64)
    if HAS_W:
        cap = _site_cap(ring_ptr, rs, head_ptr, L, pstart_ptr, pnode_ptr,
                        kern_ptr, s, dt, BL)
        req = tl.maximum(tl.load(wd_ptr + sid), 0.0)
        g = tl.minimum(req, cap)
        tl.store(got_ptr + sid, g)
        tl.store(wcount_ptr + sid, tl.load(wcount_ptr + sid) + g)
        if C > 0:        # load/heat passing the site times the delivered fraction
            head = tl.load(head_ptr)
            node = tl.load(snode_ptr + sid)
            avail = tl.maximum(tl.load(ring_ptr + head * rs + node), 0.0) * dt
            frac = tl.where(avail > 0, g / tl.maximum(avail, 1e-300), 0.0)
            for c in tl.static_range(C):
                pas = tl.maximum(tl.load(ra_ptr + c * csa + head * rsa + node), 0.0)
                tl.atomic_add(wda_ptr + c * nsa + sid, frac * pas * dt)
    v = -g
    if HAS_I:
        put = tl.maximum(tl.load(inj_ptr + sid), 0.0)
        v = v + put
        tl.store(icount_ptr + sid, tl.load(icount_ptr + sid) + put)
    x = _mul(v, tl.load(dt_ptr + 1))   # torch divides by a scalar as v * (1/dt)
    lo = tl.load(pstart_ptr + s)
    hi = tl.load(pstart_ptr + s + 1)
    for q in range(lo, hi):
        tl.store(xpad_ptr + tl.load(pos_ptr + q), x)


@triton.jit
def prop_kernel(rw_ptr, rsw, ra_ptr, rsa, csa, head_ptr, L, ids_ptr, t_ptr, F,
                ha_ptr, nha, dt_ptr, C: tl.constexpr, BL: tl.constexpr):
    """A withdrawal removes from every channel ring slot the fraction of its
    content that the water loses in that slot (t[:, l] / ring_water <= 0)."""
    dt = tl.load(dt_ptr)
    j = tl.program_id(0)
    node = tl.load(ids_ptr + j)
    head = tl.load(head_ptr)
    offs = tl.arange(0, BL)
    for c in tl.static_range(C):
        acc = tl.full([BL], 0.0, tl.float64)
        for l0 in range(0, L, BL):
            l = l0 + offs
            m = l < L
            slot = (head + l) % L
            qw = tl.load(rw_ptr + slot * rsw + node, mask=m, other=0.0)
            a = tl.load(t_ptr + j * F + l, mask=m, other=0.0)
            fr = tl.where(qw > 0, a / tl.maximum(qw, 1e-300), 0.0)
            fr = tl.minimum(tl.maximum(fr, -1.0), 0.0)
            cur = tl.load(ra_ptr + c * csa + slot * rsa + node, mask=m, other=0.0)
            d = cur * fr
            tl.store(ra_ptr + c * csa + slot * rsa + node, cur + d, mask=m)
            acc += d
        tl.store(ha_ptr + c * nha + node, tl.load(ha_ptr + c * nha + node)
                 + tl.sum(acc, 0) * dt)


@triton.jit
def xfill_kernel(vol_ptr, ps_ptr, pos_ptr, xpad_ptr, dt_ptr, P, B: tl.constexpr):
    dt = tl.load(dt_ptr)
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < P
    v = tl.load(vol_ptr + tl.load(ps_ptr + i, mask=m, other=0), mask=m, other=0.0)
    tl.store(xpad_ptr + tl.load(pos_ptr + i, mask=m, other=0), _mul(v, tl.load(dt_ptr + 1)), mask=m)


@triton.jit
def begin_x_kernel(run_ptr, cell_ptr, w_ptr, idx_ptr, xpad_ptr, n_pair, N,
                   B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < N
    p = tl.load(idx_ptr + i, mask=m, other=n_pair)
    ok = m & (p < n_pair)
    c = tl.load(cell_ptr + p, mask=ok, other=0)
    r = tl.load(run_ptr + c, mask=ok, other=0.0)
    w = tl.load(w_ptr + p, mask=ok, other=0.0)
    tl.store(xpad_ptr + i, tl.where(ok, r * w, 0.0), mask=m)


@triton.jit
def update_kernel(ring_ptr, rs, head_ptr, L, ids_ptr, t_ptr, F, cred_ptr,
                  dt_ptr, lost_ptr, HAS_LOST: tl.constexpr, BL: tl.constexpr):
    dt = tl.load(dt_ptr)
    """ring[(head + l) % L, ids[j]] += t[j, l];  cred[ids[j]] += t[j, L] dt;
    optionally lost[ids[j]] += t[j, L + 1] dt."""
    j = tl.program_id(0)
    node = tl.load(ids_ptr + j)
    head = tl.load(head_ptr)
    offs = tl.arange(0, BL)
    for l0 in range(0, L, BL):
        l = l0 + offs
        m = l < L
        slot = (head + l) % L
        a = tl.load(ring_ptr + slot * rs + node, mask=m, other=0.0)
        b = tl.load(t_ptr + j * F + l, mask=m, other=0.0)
        tl.store(ring_ptr + slot * rs + node, a + b, mask=m)
    tl.store(cred_ptr + node, tl.load(cred_ptr + node)
             + _mul(tl.load(t_ptr + j * F + L), dt))
    if HAS_LOST:
        tl.store(lost_ptr + node, tl.load(lost_ptr + node)
                 + _mul(tl.load(t_ptr + j * F + L + 1), dt))


@triton.jit
def end_kernel(ring_ptr, rs, head_ptr, out_ptr, q_ptr, n_node, dt_ptr,
               B: tl.constexpr):
    dt = tl.load(dt_ptr)
    j = tl.program_id(0) * B + tl.arange(0, B)
    m = j < n_node
    head = tl.load(head_ptr)
    q = tl.load(ring_ptr + head * rs + j, mask=m, other=0.0)
    tl.store(q_ptr + j, q, mask=m)
    tl.store(out_ptr + j, tl.load(out_ptr + j, mask=m, other=0.0) + _mul(q, dt), mask=m)
    tl.store(ring_ptr + head * rs + j, tl.full([B], 0.0, tl.float64), mask=m)


@triton.jit
def head_kernel(head_ptr, L):
    tl.store(head_ptr, (tl.load(head_ptr) + 1) % L)


def cap(ring, head, L, lvk, dt):
    out = torch.empty(lvk['n'], dtype=ring.dtype, device=ring.device)
    cap_kernel[(lvk['n'],)](ring, ring.stride(0), head, L, lvk['pstart'],
                            lvk['pnode'], lvk['kern_s'], out, dt, BL=BL)
    return out


def level(ring, head, L, lvk, wd, inj, got, wcount, icount, dt, chan=None):
    """chan = (site_node, ring_all [C, L, N], wd_all [C, n_site]) adds the
    per-channel passing-load counters of a withdrawal."""
    n = lvk['n']
    z = lvk['pstart']
    if chan is None:
        sn, ra, wda, C = z, ring, ring, 0
    else:
        sn, ra, wda = chan
        C = ra.shape[0]
    level_kernel[(n,)](ring, ring.stride(0), head, L, lvk['pstart'],
                       lvk['pnode'], lvk['kern_s'], lvk['pos'], lvk['sites'],
                       wd if wd is not None else z, inj if inj is not None else z,
                       got, wcount, icount, lvk['xpad'], dt, sn, ra,
                       ra.stride(1), ra.stride(0), wda, wda.stride(0), C=C,
                       HAS_W=wd is not None, HAS_I=inj is not None, BL=BL)


def prop(ring_w, ring_all, head, L, ids, t, hooks_all, dt):
    C = ring_all.shape[0]
    prop_kernel[(ids.numel(),)](ring_w, ring_w.stride(0), ring_all,
                                ring_all.stride(1), ring_all.stride(0), head, L,
                                ids, t, t.shape[1], hooks_all,
                                hooks_all.stride(0), dt, C=C, BL=BL)


def xfill(vol, lvk, dt):
    P = lvk['ps'].numel()
    xfill_kernel[(triton.cdiv(P, 256),)](vol, lvk['ps'], lvk['pos'], lvk['xpad'],
                                         dt, P, B=256)


def begin_x(run, cell, w, idx_flat, xpad, n_pair):
    N = idx_flat.numel()
    begin_x_kernel[(triton.cdiv(N, 1024),)](run, cell, w, idx_flat, xpad,
                                            n_pair, N, B=1024)


def update(ring, head, L, ids, t, cred, dt, lost=None):
    update_kernel[(ids.numel(),)](ring, ring.stride(0), head, L, ids, t,
                                  t.shape[1], cred, dt,
                                  lost if lost is not None else cred,
                                  HAS_LOST=lost is not None, BL=BL)


def end(ring, head, out, L, dt):
    q = torch.empty(ring.shape[1], dtype=ring.dtype, device=ring.device)
    end_kernel[(triton.cdiv(q.numel(), 128),)](ring, ring.stride(0), head, out,
                                               q, q.numel(), dt, B=128)
    head_kernel[(1,)](head, L)
    return q


def gather(s, src, dst, buf):
    gather_kernel[(src.numel(), triton.cdiv(s.shape[1], 128))](
        s, s.shape[1], src, dst, buf, BF=128)


@triton.jit
def begin_xc_kernel(run_ptr, n_cell, cell_ptr, w_ptr, n_pair, idx_ptr, xpad_ptr,
                    N, BLK: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    """xpad[blk, c, j] = flux[c, cell[p]] * w[c, p] for p = idx[blk*BLK + j]."""
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < N
    p = tl.load(idx_ptr + i, mask=m, other=n_pair)
    ok = m & (p < n_pair)
    cell = tl.load(cell_ptr + p, mask=ok, other=0)
    blk = i // BLK
    j = i % BLK
    for c in tl.static_range(C):
        r = tl.load(run_ptr + c * n_cell + cell, mask=ok, other=0.0)
        w = tl.load(w_ptr + c * n_pair + p, mask=ok, other=0.0)
        tl.store(xpad_ptr + (blk * C + c) * BLK + j, tl.where(ok, r * w, 0.0), mask=m)


@triton.jit
def decay_update_kernel(ring_ptr, rs, cs_ring, head_ptr, L, ids_ptr, t_ptr, nid, F,
                        d_ptr, cred_ptr, ncr, lost_ptr, dt_ptr,
                        HAS_LOST: tl.constexpr, BL: tl.constexpr):
    """Channel c: dec[l] = t[c, j, l] d[c, l];  ring[c, (head + l) % L, node] +=
    dec;  cred[c, node] += sum(dec) dt;  lost[c, node] += (t[c, j, L] - sum(dec)) dt."""
    dt = tl.load(dt_ptr)
    j = tl.program_id(0)
    c = tl.program_id(1)
    node = tl.load(ids_ptr + j)
    head = tl.load(head_ptr)
    offs = tl.arange(0, BL)
    acc = tl.full([BL], 0.0, tl.float64)
    for l0 in range(0, L, BL):
        l = l0 + offs
        m = l < L
        slot = (head + l) % L
        tv = tl.load(t_ptr + (c * nid + j) * F + l, mask=m, other=0.0)
        dec = _mul(tv, tl.load(d_ptr + c * L + l, mask=m, other=0.0))
        a = tl.load(ring_ptr + c * cs_ring + slot * rs + node, mask=m, other=0.0)
        tl.store(ring_ptr + c * cs_ring + slot * rs + node, a + dec, mask=m)
        acc += dec
    cs = tl.sum(acc, 0)
    tl.store(cred_ptr + c * ncr + node, tl.load(cred_ptr + c * ncr + node) + _mul(cs, dt))
    if HAS_LOST:
        ones = tl.load(t_ptr + (c * nid + j) * F + L)
        tl.store(lost_ptr + c * ncr + node,
                 tl.load(lost_ptr + c * ncr + node) + _mul(ones - cs, dt))


@triton.jit
def end_batch_kernel(ring_ptr, rs, cs_ring, head_ptr, out_ptr, q_ptr, N, dt_ptr,
                     B: tl.constexpr):
    dt = tl.load(dt_ptr)
    j = tl.program_id(0) * B + tl.arange(0, B)
    c = tl.program_id(1)
    m = j < N
    head = tl.load(head_ptr)
    q = tl.load(ring_ptr + c * cs_ring + head * rs + j, mask=m, other=0.0)
    tl.store(q_ptr + c * N + j, q, mask=m)
    tl.store(out_ptr + c * N + j, tl.load(out_ptr + c * N + j, mask=m, other=0.0) + _mul(q, dt), mask=m)
    tl.store(ring_ptr + c * cs_ring + head * rs + j, tl.full([B], 0.0, tl.float64), mask=m)


def begin_xc(flux, cell, w, idx_flat, xpad, n_pair, block):
    C, n_cell = flux.shape
    N = idx_flat.numel()
    begin_xc_kernel[(triton.cdiv(N, 1024),)](flux, n_cell, cell, w, n_pair, idx_flat,
                                             xpad, N, BLK=block, C=C, B=1024)


def decay_update(ring_all, head, L, ids, t, d, cred, dt, lost=None):
    C = ring_all.shape[0]
    decay_update_kernel[(ids.numel(), C)](ring_all, ring_all.stride(1), ring_all.stride(0),
                                          head, L, ids, t, ids.numel(), t.shape[2], d,
                                          cred, cred.stride(0),
                                          lost if lost is not None else cred, dt,
                                          HAS_LOST=lost is not None, BL=BL)


def end_batch(ring_all, head, out_all, dt):
    C, L, N = ring_all.shape
    q = torch.empty(C, N, dtype=ring_all.dtype, device=ring_all.device)
    end_batch_kernel[(triton.cdiv(N, 128), C)](ring_all, ring_all.stride(1),
                                               ring_all.stride(0), head, out_all, q,
                                               N, dt, B=128)
    return q


def treered(s, ix, cnt):
    out = torch.empty(ix.shape[0], s.shape[1], dtype=s.dtype, device=s.device)
    treered_kernel[(ix.shape[0], triton.cdiv(s.shape[1], 128))](
        s, s.shape[1], ix, cnt, out, BF=128)
    return out
