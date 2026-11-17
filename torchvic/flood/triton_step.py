"""Triton kernels for the 'torchvic' scheme (theta = 1), CUDA only.

The default float64 step is TritonStepper's compacted path (see its
docstring): k_faces_mark, k_faces_wet, k_limiter2, k_update2, k_clock2. It
evaluates the expressions of the reference step below (fused=False) on the
same numbers; k_update2 is compiled with FMA contraction off and writes
out what the compiler does to k_update: the contractions as fma calls when
Triton contracts (TRITON_DEFAULT_FP_FUSION=1, its default), the plain
products and sums when it does not (torchvic.core.fusion sets 0 at import),
so the two paths give bit-identical states and ledgers either way.

Reference step (fused=False): four launches and the edge faces, all scalars
on the device:

  A  faces     discharge of every face from h and the previous momentum
               (each cell owns its east and south faces; the west/north
               edge cells also own the boundary faces)
  B  limiter   donor factor w = min(1, h A / outgoing volume) per cell
  C  update    h += src dt + dt (Qw - Qe + Qn - Qs) / A with the faces
               scaled by their donors' factors; new momentum q = Q / dx;
               interval maximum depth; block max of h (atomic, for the next
               CFL step) and the clamp volume (atomic, normally 0)
  D  clock     (one program) t += dt, ledger (sources, boundary outflow
               summed in a fixed order), next dt from the max depth

A step whose dt is 0 (after the interval end) writes nothing. The kernels
evaluate the same expressions in the same order as engine.FloodEngine's
torch code; the compiler may contract a*b+c into one FMA, so the two agree
to round-off, not bit for bit.
"""
import os

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:                       # CPU-only torch has no Triton
    from ..core.triton_stub import triton, tl, libdevice

#: ledger slots (as engine._RAIN ...)
_RAIN, _POINT, _OUT, _CLAMP, _NSTEP, _DTMIN, _DTMAX, _CFLMIN = range(8)


@triton.jit
def _face(h0, h1, z0, z1, q0, qo, fn, gdt, dx, th, max_hflow, e103):
    qvect = tl.sqrt(q0 * q0 + qo * qo)
    y0 = z0 + h0
    y1 = z1 + h1
    dh = y0 - y1
    Sf = -dh / dx
    hflow = tl.maximum(y0, y1) - tl.maximum(z0, z1)
    hflow = tl.minimum(tl.maximum(hflow, 0.0), max_hflow)
    a = gdt * hflow
    p = libdevice.pow(tl.maximum(hflow, th), e103)
    den = 1.0 + a * fn * fn * tl.abs(qvect) / p
    Q = (q0 - a * Sf) / den * dx
    wet = ((h0 > th) | (h1 > th)) & (hflow > th)
    return tl.where(wet, Q, 0.0)


@triton.jit
def _free(h0, h1, z0, z1, q0, fn, slope, gdt, dx, th, sign, e103):
    hflow = h0
    dh = ((z0 + h0) - z1) - h1
    Sf = tl.where(slope < -0.999, -dh / dx, slope)
    a = gdt * hflow
    p = libdevice.pow(tl.maximum(hflow, th), e103)
    q = sign * (tl.abs(q0) + tl.abs(a * Sf)) / (
        1.0 + a * fn * fn * tl.abs(q0) / p)
    Q = q * dx
    on = (slope == slope) & (h0 > th)
    return tl.where(on, Q, 0.0)


@triton.jit
def k_faces(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
            sw_ptr, se_ptr, sn_ptr, ss_ptr, dt_ptr, par_ptr,
            ny, nx, DOX: tl.constexpr, DOY: tl.constexpr,
            BLOCK: tl.constexpr):
    # Python floats would reach the kernel as float32: the parameters come
    # from a float64 tensor [dx, g, th, max_hflow, A, cfl, max_dt]
    T_ = h_ptr.dtype.element_ty
    dx = tl.load(par_ptr + 0).to(T_)
    g = tl.load(par_ptr + 1).to(T_)
    th = tl.load(par_ptr + 2).to(T_)
    max_hflow = tl.load(par_ptr + 3).to(T_)
    e103 = tl.load(par_ptr + 7).to(T_)
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    m = k < ny * nx
    dt = tl.load(dt_ptr).to(h_ptr.dtype.element_ty)
    m = m & (dt > 0)
    j = k // nx
    i = k - j * nx
    sx = nx + 1                              # row stride of qx / Qx
    gdt = g * dt
    hc = tl.load(h_ptr + k, mask=m, other=0.0)
    zc = tl.load(z_ptr + k, mask=m, other=0.0)
    nc = tl.load(n_ptr + k, mask=m, other=0.0)
    if DOX:
        # east face (j, i+1)
        me = m & (i < nx - 1)
        he = tl.load(h_ptr + k + 1, mask=me, other=0.0)
        ze = tl.load(z_ptr + k + 1, mask=me, other=0.0)
        ne = tl.load(n_ptr + k + 1, mask=me, other=0.0)
        q0 = tl.load(qx_ptr + j * sx + i + 1, mask=m, other=0.0)
        a1 = tl.load(qy_ptr + j * nx + i, mask=me, other=0.0)
        a2 = tl.load(qy_ptr + j * nx + i + 1, mask=me, other=0.0)
        a3 = tl.load(qy_ptr + (j + 1) * nx + i, mask=me, other=0.0)
        a4 = tl.load(qy_ptr + (j + 1) * nx + i + 1, mask=me, other=0.0)
        qyb = (((a1 + a2) + a3) + a4) / 4.0
        Qe = _face(hc, he, zc, ze, q0, qyb, 0.5 * (nc + ne), gdt, dx, th,
                   max_hflow, e103)
        me_ = m & (i < nx - 1)
        tl.store(Qx_ptr + j * sx + i + 1, Qe, mask=me_)
    if DOY:
        # south face (j+1, i)
        ms = m & (j < ny - 1)
        hs = tl.load(h_ptr + k + nx, mask=ms, other=0.0)
        zs = tl.load(z_ptr + k + nx, mask=ms, other=0.0)
        ns = tl.load(n_ptr + k + nx, mask=ms, other=0.0)
        q0s = tl.load(qy_ptr + (j + 1) * nx + i, mask=m, other=0.0)
        b1 = tl.load(qx_ptr + j * sx + i, mask=ms, other=0.0)
        b2 = tl.load(qx_ptr + j * sx + i + 1, mask=ms, other=0.0)
        b3 = tl.load(qx_ptr + (j + 1) * sx + i, mask=ms, other=0.0)
        b4 = tl.load(qx_ptr + (j + 1) * sx + i + 1, mask=ms, other=0.0)
        qxb = (((b1 + b2) + b3) + b4) / 4.0
        Qs = _face(hc, hs, zc, zs, q0s, qxb, 0.5 * (nc + ns), gdt, dx, th,
                   max_hflow, e103)
        tl.store(Qy_ptr + (j + 1) * nx + i, Qs, mask=ms)


@triton.jit
def k_bound(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
            sw_ptr, se_ptr, sn_ptr, ss_ptr, dt_ptr, par_ptr, ny, nx,
            BLOCK: tl.constexpr):
    """The 2 (ny + nx) edge faces: FREE outflow or closed (0)."""
    T_ = h_ptr.dtype.element_ty
    dx = tl.load(par_ptr + 0).to(T_)
    g = tl.load(par_ptr + 1).to(T_)
    th = tl.load(par_ptr + 2).to(T_)
    e103 = tl.load(par_ptr + 7).to(T_)
    dt = tl.load(dt_ptr).to(T_)
    gdt = g * dt
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = (b < 2 * (ny + nx)) & (dt > 0)
    sx = nx + 1
    isx = b < 2 * ny                    # W (b < ny) and E rows
    east = (b >= ny) & isx
    south = b >= 2 * ny + nx
    r = tl.where(isx, tl.where(east, b - ny, b), 0)
    c = tl.where(isx, 0, tl.where(south, b - 2 * ny - nx, b - 2 * ny))
    # edge cell p0 and its inner neighbour p1
    j0 = tl.where(isx, r, tl.where(south, ny - 1, 0))
    i0 = tl.where(isx, tl.where(east, nx - 1, 0), c)
    j1 = tl.where(isx, j0, tl.where(south, ny - 2, 1))
    i1 = tl.where(isx, tl.where(east, nx - 2, 1), i0)
    k0 = j0 * nx + i0
    k1 = j1 * nx + i1
    h0 = tl.load(h_ptr + k0, mask=m, other=0.0)
    h1 = tl.load(h_ptr + k1, mask=m, other=0.0)
    z0 = tl.load(z_ptr + k0, mask=m, other=0.0)
    z1 = tl.load(z_ptr + k1, mask=m, other=0.0)
    fn = tl.load(n_ptr + k0, mask=m, other=0.0)
    fx = tl.where(east, j0 * sx + nx, j0 * sx)          # x faces
    fy = tl.where(south, ny * nx + i0, i0)              # y faces
    q0 = tl.where(isx, tl.load(qx_ptr + fx, mask=m & isx, other=0.0),
                  tl.load(qy_ptr + fy, mask=m & (~isx), other=0.0))
    sl = tl.where(isx, tl.where(east,
                                tl.load(se_ptr + r, mask=m & east,
                                        other=float('nan')),
                                tl.load(sw_ptr + r, mask=m & isx & (~east),
                                        other=float('nan'))),
                  tl.where(south,
                           tl.load(ss_ptr + c, mask=m & south,
                                   other=float('nan')),
                           tl.load(sn_ptr + c, mask=m & (~isx) & (~south),
                                   other=float('nan'))))
    sign = tl.where(east | south, 1.0, -1.0).to(T_)
    Q = _free(h0, h1, z0, z1, q0, fn, sl.to(T_), gdt, dx, th, sign, e103)
    tl.store(Qx_ptr + fx, Q, mask=m & isx)
    tl.store(Qy_ptr + fy, Q, mask=m & (~isx))


@triton.jit
def k_limiter(h_ptr, Qx_ptr, Qy_ptr, w_ptr, dt_ptr, par_ptr, ny, nx,
              BLOCK: tl.constexpr):
    A = tl.load(par_ptr + 4).to(h_ptr.dtype.element_ty)
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    m = k < ny * nx
    dt = tl.load(dt_ptr).to(h_ptr.dtype.element_ty)
    m = m & (dt > 0)
    j = k // nx
    i = k - j * nx
    sx = nx + 1
    qw = tl.load(Qx_ptr + j * sx + i, mask=m, other=0.0)
    qe = tl.load(Qx_ptr + j * sx + i + 1, mask=m, other=0.0)
    qn = tl.load(Qy_ptr + j * nx + i, mask=m, other=0.0)
    qs = tl.load(Qy_ptr + (j + 1) * nx + i, mask=m, other=0.0)
    out = (((tl.maximum(-qw, 0.0) + tl.maximum(qe, 0.0))
            + tl.maximum(-qn, 0.0)) + tl.maximum(qs, 0.0)) * dt
    cv = tl.load(h_ptr + k, mask=m, other=0.0) * A
    lim = out > cv
    w = tl.where(lim, cv / tl.where(lim, out, 1.0), 1.0)
    tl.store(w_ptr + k, w, mask=m)


@triton.jit
def k_update(h_ptr, src_ptr, Qx_ptr, Qy_ptr, w_ptr, qx_ptr, qy_ptr,
             hmax_ptr, bw_ptr, be_ptr, bn_ptr, bs_ptr, hm_ptr, acc_ptr,
             dt_ptr, par_ptr, ny, nx, LIMIT: tl.constexpr,
             BLOCK: tl.constexpr):
    T_ = h_ptr.dtype.element_ty
    dx = tl.load(par_ptr + 0).to(T_)
    th = tl.load(par_ptr + 2).to(T_)
    A = tl.load(par_ptr + 4).to(T_)
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    m0 = k < ny * nx
    dt = tl.load(dt_ptr).to(h_ptr.dtype.element_ty)
    m = m0 & (dt > 0)
    j = k // nx
    i = k - j * nx
    sx = nx + 1
    qw = tl.load(Qx_ptr + j * sx + i, mask=m, other=0.0)
    qe = tl.load(Qx_ptr + j * sx + i + 1, mask=m, other=0.0)
    qn = tl.load(Qy_ptr + j * nx + i, mask=m, other=0.0)
    qs = tl.load(Qy_ptr + (j + 1) * nx + i, mask=m, other=0.0)
    if LIMIT:
        wc = tl.load(w_ptr + k, mask=m, other=1.0)
        ww = tl.load(w_ptr + k - 1, mask=m & (i > 0), other=1.0)
        we = tl.load(w_ptr + k + 1, mask=m & (i < nx - 1), other=1.0)
        wn = tl.load(w_ptr + k - nx, mask=m & (j > 0), other=1.0)
        wsv = tl.load(w_ptr + k + nx, mask=m & (j < ny - 1), other=1.0)
        # face factor = factor of the cell the water leaves; at an edge the
        # outside 'cell' is the edge cell itself
        ww = tl.where(i > 0, ww, wc)
        we = tl.where(i < nx - 1, we, wc)
        wn = tl.where(j > 0, wn, wc)
        wsv = tl.where(j < ny - 1, wsv, wc)
        qw = tl.where(qw > 0, qw * ww, qw * wc)
        qe = tl.where(qe > 0, qe * wc, qe * we)
        qn = tl.where(qn > 0, qn * wn, qn * wc)
        qs = tl.where(qs > 0, qs * wc, qs * wsv)
    h = tl.load(h_ptr + k, mask=m, other=0.0)
    src = tl.load(src_ptr + k, mask=m, other=0.0)
    hs = h + src * dt
    hn = hs + dt * (((qw - qe) + qn) - qs) / A
    neg = tl.maximum(-hn, 0.0)
    hn = hn + neg
    tl.store(h_ptr + k, hn, mask=m)
    inv = 1.0 / dx
    tl.store(qx_ptr + j * sx + i + 1, qe * inv, mask=m)
    tl.store(qy_ptr + (j + 1) * nx + i, qs * inv, mask=m)
    tl.store(qx_ptr + j * sx + i, qw * inv, mask=m & (i == 0))
    tl.store(qy_ptr + i, qn * inv, mask=m & (j == 0))
    tl.store(bw_ptr + j, qw, mask=m & (i == 0))
    tl.store(be_ptr + j, qe, mask=m & (i == nx - 1))
    tl.store(bn_ptr + i, qn, mask=m & (j == 0))
    tl.store(bs_ptr + i, qs, mask=m & (j == ny - 1))
    hx = tl.load(hmax_ptr + k, mask=m, other=0.0)
    hx = tl.where(hn > th, tl.maximum(hx, hn), hx)
    tl.store(hmax_ptr + k, hx, mask=m)
    # block maximum of the new depth (h >= 0: the bit pattern orders like
    # the value) and the clamp volume
    hb = tl.max(tl.where(m, hn, 0.0), axis=0)
    if h_ptr.dtype.element_ty == tl.float64:
        tl.atomic_max(hm_ptr, hb.to(tl.int64, bitcast=True))
    else:
        tl.atomic_max(hm_ptr, hb.to(tl.int32, bitcast=True).to(tl.int64))
    cl = tl.sum(tl.where(m, neg, 0.0).to(tl.float64), axis=0) * A.to(
        tl.float64)
    if cl > 0:
        tl.atomic_add(acc_ptr + 3, cl)


@triton.jit
def k_clock(bw_ptr, be_ptr, bn_ptr, bs_ptr, hm_ptr, acc_ptr, dt_ptr, t_ptr,
            T_ptr, RP_ptr, par_ptr, eacc_ptr, ny, nx,
            F64: tl.constexpr, EDGE: tl.constexpr, BLOCK: tl.constexpr):
    dx = tl.load(par_ptr + 0)
    g = tl.load(par_ptr + 1)
    th = tl.load(par_ptr + 2)
    cfl = tl.load(par_ptr + 5)
    max_dt = tl.load(par_ptr + 6)
    dt = tl.load(dt_ptr)                                    # float64
    o = tl.arange(0, BLOCK)
    sx = tl.zeros((BLOCK,), dtype=tl.float64)
    for s in range(0, ny, BLOCK):
        mm = s + o < ny
        sx += (tl.load(be_ptr + s + o, mask=mm, other=0.0).to(tl.float64)
               - tl.load(bw_ptr + s + o, mask=mm, other=0.0).to(tl.float64))
        if EDGE:
            # outflow per face over the interval (W rows, E rows); dt is 0
            # after the interval end
            ew = eacc_ptr + s + o
            tl.store(ew, tl.load(ew, mask=mm, other=0.0) - tl.load(
                bw_ptr + s + o, mask=mm, other=0.0).to(tl.float64) * dt,
                mask=mm)
            ee = eacc_ptr + ny + s + o
            tl.store(ee, tl.load(ee, mask=mm, other=0.0) + tl.load(
                be_ptr + s + o, mask=mm, other=0.0).to(tl.float64) * dt,
                mask=mm)
    sy = tl.zeros((BLOCK,), dtype=tl.float64)
    for s in range(0, nx, BLOCK):
        mm = s + o < nx
        sy += (tl.load(bs_ptr + s + o, mask=mm, other=0.0).to(tl.float64)
               - tl.load(bn_ptr + s + o, mask=mm, other=0.0).to(tl.float64))
        if EDGE:
            en = eacc_ptr + 2 * ny + s + o
            tl.store(en, tl.load(en, mask=mm, other=0.0) - tl.load(
                bn_ptr + s + o, mask=mm, other=0.0).to(tl.float64) * dt,
                mask=mm)
            es = eacc_ptr + 2 * ny + nx + s + o
            tl.store(es, tl.load(es, mask=mm, other=0.0) + tl.load(
                bs_ptr + s + o, mask=mm, other=0.0).to(tl.float64) * dt,
                mask=mm)
    live = dt > 0
    out = tl.where(live, (tl.sum(sx, axis=0) + tl.sum(sy, axis=0)) * dt, 0.0)
    R = tl.load(RP_ptr)
    P = tl.load(RP_ptr + 1)
    tl.store(acc_ptr + 0, tl.load(acc_ptr + 0) + R * dt)
    tl.store(acc_ptr + 1, tl.load(acc_ptr + 1) + P * dt)
    tl.store(acc_ptr + 2, tl.load(acc_ptr + 2) + out)
    tl.store(acc_ptr + 4, tl.load(acc_ptr + 4) + tl.where(live, 1.0, 0.0))
    t = tl.load(t_ptr) + dt
    tl.store(t_ptr, t)
    # next step from the new maximum depth (LISFLOOD-FP CPU rule)
    bits = tl.load(hm_ptr)
    if F64:
        hm = bits.to(tl.float64, bitcast=True)
    else:
        hm = bits.to(tl.int32).to(tl.float32, bitcast=True).to(tl.float64)
    tl.store(hm_ptr, bits * 0)
    loc = cfl * dx / tl.sqrt(g * tl.maximum(hm, th))
    d = tl.where(hm > th, tl.minimum(loc, max_dt), max_dt)
    rem = tl.load(T_ptr) - t
    dn = tl.where(rem > 0, tl.minimum(d, rem), 0.0)
    tl.store(dt_ptr, dn)
    on = dn > 0
    tl.store(acc_ptr + 5, tl.where(on, tl.minimum(tl.load(acc_ptr + 5), dn),
                                   tl.load(acc_ptr + 5)))
    tl.store(acc_ptr + 6, tl.where(on, tl.maximum(tl.load(acc_ptr + 6), dn),
                                   tl.load(acc_ptr + 6)))
    tl.store(acc_ptr + 7, tl.where(on, tl.minimum(tl.load(acc_ptr + 7), d),
                                   tl.load(acc_ptr + 7)))


@triton.jit
def _faces_row(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr, pid,
               dt, gdt, dx, th, max_hflow, e103, ny, nx, ncb,
               NSCAL: tl.constexpr, SKIP: tl.constexpr, BLOCK: tl.constexpr):
    T_ = h_ptr.dtype.element_ty
    sx = nx + 1
    j = pid // ncb
    i = (pid - j * ncb) * BLOCK + tl.arange(0, BLOCK)
    k = j * nx + i
    m = (i < nx) & (dt > 0)
    me = m & (i < nx - 1)
    ms = m & (j < ny - 1)
    hc = tl.load(h_ptr + k, mask=m, other=0.0)
    he = tl.load(h_ptr + k + 1, mask=me, other=0.0)
    hs = tl.load(h_ptr + k + nx, mask=ms, other=0.0)
    if SKIP:
        act = tl.max(((hc > th) | (he > th) | (hs > th)).to(tl.int32),
                     axis=0)
    else:
        act = 1
    if act > 0:
        zc = tl.load(z_ptr + k, mask=m, other=0.0)
        ze = tl.load(z_ptr + k + 1, mask=me, other=0.0)
        zs = tl.load(z_ptr + k + nx, mask=ms, other=0.0)
        if NSCAL:
            nc = tl.load(n_ptr)
            ne = nc
            ns = nc
        else:
            nc = tl.load(n_ptr + k, mask=m, other=0.0)
            ne = tl.load(n_ptr + k + 1, mask=me, other=0.0)
            ns = tl.load(n_ptr + k + nx, mask=ms, other=0.0)
        # east face (j, i+1)
        q0 = tl.load(qx_ptr + j * sx + i + 1, mask=m, other=0.0)
        a1 = tl.load(qy_ptr + j * nx + i, mask=me, other=0.0)
        a2 = tl.load(qy_ptr + j * nx + i + 1, mask=me, other=0.0)
        a3 = tl.load(qy_ptr + (j + 1) * nx + i, mask=me, other=0.0)
        a4 = tl.load(qy_ptr + (j + 1) * nx + i + 1, mask=me, other=0.0)
        qyb = (((a1 + a2) + a3) + a4) / 4.0
        Qe = _face(hc, he, zc, ze, q0, qyb, 0.5 * (nc + ne), gdt, dx, th,
                   max_hflow, e103)
        tl.store(Qx_ptr + j * sx + i + 1, Qe, mask=me)
        # south face (j+1, i)
        q0s = tl.load(qy_ptr + (j + 1) * nx + i, mask=m, other=0.0)
        b1 = tl.load(qx_ptr + j * sx + i, mask=ms, other=0.0)
        b2 = tl.load(qx_ptr + j * sx + i + 1, mask=ms, other=0.0)
        b3 = tl.load(qx_ptr + (j + 1) * sx + i, mask=ms, other=0.0)
        b4 = tl.load(qx_ptr + (j + 1) * sx + i + 1, mask=ms, other=0.0)
        qxb = (((b1 + b2) + b3) + b4) / 4.0
        Qs = _face(hc, hs, zc, zs, q0s, qxb, 0.5 * (nc + ns), gdt, dx,
                   th, max_hflow, e103)
        tl.store(Qy_ptr + (j + 1) * nx + i, Qs, mask=ms)
    else:
        zero = tl.zeros((BLOCK,), dtype=T_)
        tl.store(Qx_ptr + j * sx + i + 1, zero, mask=me)
        tl.store(Qy_ptr + (j + 1) * nx + i, zero, mask=ms)


@triton.jit
def _faces_edge(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
                sw_ptr, se_ptr, sn_ptr, ss_ptr, pb, dt, gdt, dx, th, e103,
                ny, nx, NSCAL: tl.constexpr, BBLOCK: tl.constexpr):
    T_ = h_ptr.dtype.element_ty
    sx = nx + 1
    b = pb * BBLOCK + tl.arange(0, BBLOCK)
    m = (b < 2 * (ny + nx)) & (dt > 0)
    isx = b < 2 * ny
    east = (b >= ny) & isx
    south = b >= 2 * ny + nx
    r = tl.where(isx, tl.where(east, b - ny, b), 0)
    c = tl.where(isx, 0, tl.where(south, b - 2 * ny - nx, b - 2 * ny))
    j0 = tl.where(isx, r, tl.where(south, ny - 1, 0))
    i0 = tl.where(isx, tl.where(east, nx - 1, 0), c)
    j1 = tl.where(isx, j0, tl.where(south, ny - 2, 1))
    i1 = tl.where(isx, tl.where(east, nx - 2, 1), i0)
    k0 = j0 * nx + i0
    k1 = j1 * nx + i1
    h0 = tl.load(h_ptr + k0, mask=m, other=0.0)
    h1 = tl.load(h_ptr + k1, mask=m, other=0.0)
    z0 = tl.load(z_ptr + k0, mask=m, other=0.0)
    z1 = tl.load(z_ptr + k1, mask=m, other=0.0)
    if NSCAL:
        fn = tl.load(n_ptr) + tl.zeros((BBLOCK,), dtype=T_)
    else:
        fn = tl.load(n_ptr + k0, mask=m, other=0.0)
    fx = tl.where(east, j0 * sx + nx, j0 * sx)
    fy = tl.where(south, ny * nx + i0, i0)
    q0 = tl.where(isx, tl.load(qx_ptr + fx, mask=m & isx, other=0.0),
                  tl.load(qy_ptr + fy, mask=m & (~isx), other=0.0))
    sl = tl.where(isx, tl.where(east,
                                tl.load(se_ptr + r, mask=m & east,
                                        other=float('nan')),
                                tl.load(sw_ptr + r, mask=m & isx & (~east),
                                        other=float('nan'))),
                  tl.where(south,
                           tl.load(ss_ptr + c, mask=m & south,
                                   other=float('nan')),
                           tl.load(sn_ptr + c, mask=m & (~isx) & (~south),
                                   other=float('nan'))))
    sign = tl.where(east | south, 1.0, -1.0).to(T_)
    Q = _free(h0, h1, z0, z1, q0, fn, sl.to(T_), gdt, dx, th, sign, e103)
    tl.store(Qx_ptr + fx, Q, mask=m & isx)
    tl.store(Qy_ptr + fy, Q, mask=m & (~isx))


@triton.jit
def k_faces2(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
             sw_ptr, se_ptr, sn_ptr, ss_ptr, dt_ptr, par_ptr,
             ny, nx, ncb, NSCAL: tl.constexpr, SKIP: tl.constexpr,
             BLOCK: tl.constexpr, BBLOCK: tl.constexpr):
    """k_faces and k_bound in one launch: programs 0 .. ny*ncb-1 own a
    row segment of BLOCK cells (east and south faces), the rest the edge
    faces. Same expressions as k_faces / k_bound. SKIP: a segment whose
    cells and east/south neighbours are all at or below the depth
    threshold has only dry faces (Q = 0 exactly) and skips the arithmetic.
    NSCAL: one Manning n for the grid (n_ptr[0])."""
    T_ = h_ptr.dtype.element_ty
    dx = tl.load(par_ptr + 0).to(T_)
    g = tl.load(par_ptr + 1).to(T_)
    th = tl.load(par_ptr + 2).to(T_)
    max_hflow = tl.load(par_ptr + 3).to(T_)
    e103 = tl.load(par_ptr + 7).to(T_)
    dt = tl.load(dt_ptr).to(T_)
    gdt = g * dt
    pid = tl.program_id(0)
    # the edge programs first: their float64 power then overlaps the
    # row programs instead of trailing them
    nbe = (2 * (ny + nx) + BBLOCK - 1) // BBLOCK
    if pid >= nbe:
        pr = pid - nbe
        _faces_row(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr, pr,
                   dt, gdt, dx, th, max_hflow, e103, ny, nx, ncb,
                   NSCAL, SKIP, BLOCK)
    else:
        _faces_edge(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
                    sw_ptr, se_ptr, sn_ptr, ss_ptr, pid, dt, gdt,
                    dx, th, e103, ny, nx, NSCAL, BBLOCK)


@triton.jit
def _mark_row(h_ptr, z_ptr, Qx_ptr, Qy_ptr, list_ptr, cnt_ptr, pid, dt, th,
              max_hflow, ny, nx, ncb, BLOCK: tl.constexpr, fx_ptr, fy_ptr,
              QF: tl.constexpr):
    T_ = h_ptr.dtype.element_ty
    sx = nx + 1
    j = pid // ncb
    i = (pid - j * ncb) * BLOCK + tl.arange(0, BLOCK)
    k = j * nx + i
    m = (i < nx) & (dt > 0)
    me = m & (i < nx - 1)
    ms = m & (j < ny - 1)
    hc = tl.load(h_ptr + k, mask=m, other=0.0)
    he = tl.load(h_ptr + k + 1, mask=me, other=0.0)
    hs = tl.load(h_ptr + k + nx, mask=ms, other=0.0)
    zc = tl.load(z_ptr + k, mask=m, other=0.0)
    ze = tl.load(z_ptr + k + 1, mask=me, other=0.0)
    zs = tl.load(z_ptr + k + nx, mask=ms, other=0.0)
    # the wet test of _face, same expressions (no products: nothing for
    # the compiler to contract)
    ye = tl.maximum(zc + hc, ze + he) - tl.maximum(zc, ze)
    ye = tl.minimum(tl.maximum(ye, 0.0), max_hflow)
    # (the compares on bit patterns: depths >= -0, ye and ys >= -0)
    tb = _bits(th)
    wc_ = _bits(hc) > tb
    we = me & (wc_ | (_bits(he) > tb)) & (_bits(ye) > tb)
    ys = tl.maximum(zc + hc, zs + hs) - tl.maximum(zc, zs)
    ys = tl.minimum(tl.maximum(ys, 0.0), max_hflow)
    ws = ms & (wc_ | (_bits(hs) > tb)) & (_bits(ys) > tb)
    if QF:
        # wet flags instead of zero discharges: the readers mask their
        # loads of dry faces (a dry Q is +0 either way)
        tl.store(fx_ptr + j * sx + i + 1, we.to(tl.int8), mask=me)
        tl.store(fy_ptr + (j + 1) * nx + i, ws.to(tl.int8), mask=ms)
    else:
        zero = tl.zeros((BLOCK,), dtype=T_)
        tl.store(Qx_ptr + j * sx + i + 1, zero, mask=me & (~we))
        tl.store(Qy_ptr + (j + 1) * nx + i, zero, mask=ms & (~ws))
    ie = we.to(tl.int32)
    is_ = ws.to(tl.int32)
    ne = tl.sum(ie, axis=0)
    ns = tl.sum(is_, axis=0)
    if ne + ns > 0:
        base = tl.atomic_add(cnt_ptr, ne + ns, sem='relaxed')
        tl.store(list_ptr + base + tl.cumsum(ie, axis=0) - 1, 2 * k,
                 mask=we)
        tl.store(list_ptr + base + ne + tl.cumsum(is_, axis=0) - 1,
                 2 * k + 1, mask=ws)


@triton.jit
def k_faces_mark(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
                 sw_ptr, se_ptr, sn_ptr, ss_ptr, dt_ptr, par_ptr, list_ptr,
                 cnt_ptr, ny, nx, ncb, NSCAL: tl.constexpr,
                 BLOCK: tl.constexpr, BBLOCK: tl.constexpr, fx_ptr=None,
                 fy_ptr=None, QF: tl.constexpr = False):
    """Compacted faces, pass 1: dry interior faces get Q = 0 (what _face
    returns for them), wet ones are listed (code 2k east, 2k + 1 south of
    cell k; any order) for k_faces_wet; the edge faces as in k_faces2."""
    T_ = h_ptr.dtype.element_ty
    dx = tl.load(par_ptr + 0).to(T_)
    g = tl.load(par_ptr + 1).to(T_)
    th = tl.load(par_ptr + 2).to(T_)
    max_hflow = tl.load(par_ptr + 3).to(T_)
    e103 = tl.load(par_ptr + 7).to(T_)
    dt = tl.load(dt_ptr).to(T_)
    gdt = g * dt
    pid = tl.program_id(0)
    # the edge programs first: their float64 power then overlaps the
    # row programs instead of trailing them
    nbe = (2 * (ny + nx) + BBLOCK - 1) // BBLOCK
    if pid >= nbe:
        pr = pid - nbe
        _mark_row(h_ptr, z_ptr, Qx_ptr, Qy_ptr, list_ptr, cnt_ptr, pr, dt,
                  th, max_hflow, ny, nx, ncb, BLOCK, fx_ptr, fy_ptr, QF)
    else:
        _faces_edge(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
                    sw_ptr, se_ptr, sn_ptr, ss_ptr, pid, dt, gdt,
                    dx, th, e103, ny, nx, NSCAL, BBLOCK)


@triton.jit
def k_faces_wet(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr, dt_ptr,
                par_ptr, list_ptr, cnt_ptr, ny, nx, NSCAL: tl.constexpr,
                BLOCK: tl.constexpr):
    """Compacted faces, pass 2 (persistent programs): the listed wet faces
    through _face with the inputs k_faces2 gives it, so every Q is the same
    number."""
    T_ = h_ptr.dtype.element_ty
    dx = tl.load(par_ptr + 0).to(T_)
    g = tl.load(par_ptr + 1).to(T_)
    th = tl.load(par_ptr + 2).to(T_)
    max_hflow = tl.load(par_ptr + 3).to(T_)
    e103 = tl.load(par_ptr + 7).to(T_)
    dt = tl.load(dt_ptr).to(T_)
    gdt = g * dt
    sx = nx + 1
    cnt = tl.load(cnt_ptr)
    pid = tl.program_id(0)
    npr = tl.num_programs(0)
    for st in range(pid * BLOCK, cnt, npr * BLOCK):
        f = st + tl.arange(0, BLOCK)
        m = f < cnt
        code = tl.load(list_ptr + f, mask=m, other=0)
        k = code >> 1
        so = (code & 1) == 1
        ea = m & (~so)
        sm = m & so
        j = k // nx
        i = k - j * nx
        k1 = tl.where(so, k + nx, k + 1)
        h0 = tl.load(h_ptr + k, mask=m, other=0.0)
        h1 = tl.load(h_ptr + k1, mask=m, other=0.0)
        z0 = tl.load(z_ptr + k, mask=m, other=0.0)
        z1 = tl.load(z_ptr + k1, mask=m, other=0.0)
        if NSCAL:
            n0 = tl.load(n_ptr)
            n1 = n0
        else:
            n0 = tl.load(n_ptr + k, mask=m, other=0.0)
            n1 = tl.load(n_ptr + k1, mask=m, other=0.0)
        q0 = tl.where(so, tl.load(qy_ptr + (j + 1) * nx + i, mask=sm,
                                  other=0.0),
                      tl.load(qx_ptr + j * sx + i + 1, mask=ea, other=0.0))
        a1 = tl.where(so, tl.load(qx_ptr + j * sx + i, mask=sm, other=0.0),
                      tl.load(qy_ptr + j * nx + i, mask=ea, other=0.0))
        a2 = tl.where(so, tl.load(qx_ptr + j * sx + i + 1, mask=sm,
                                  other=0.0),
                      tl.load(qy_ptr + j * nx + i + 1, mask=ea, other=0.0))
        a3 = tl.where(so, tl.load(qx_ptr + (j + 1) * sx + i, mask=sm,
                                  other=0.0),
                      tl.load(qy_ptr + (j + 1) * nx + i, mask=ea, other=0.0))
        a4 = tl.where(so, tl.load(qx_ptr + (j + 1) * sx + i + 1, mask=sm,
                                  other=0.0),
                      tl.load(qy_ptr + (j + 1) * nx + i + 1, mask=ea,
                              other=0.0))
        qo = (((a1 + a2) + a3) + a4) / 4.0
        Q = _face(h0, h1, z0, z1, q0, qo, 0.5 * (n0 + n1), gdt, dx, th,
                  max_hflow, e103)
        tl.store(Qx_ptr + j * sx + i + 1, Q, mask=ea)
        tl.store(Qy_ptr + (j + 1) * nx + i, Q, mask=sm)


@triton.jit
def _quiet_lanes(dry_ptr, j, i, m, ny, uncb, UB: tl.constexpr):
    """Per lane: its update tile (row segment of UB cells) and the tile's
    four neighbours all dry (k_update2's QUIET flags)."""
    c = i // UB
    t = j * uncb + c
    q = tl.load(dry_ptr + t, mask=m, other=0)
    q = q & tl.load(dry_ptr + tl.where(c > 0, t - 1, t), mask=m, other=0)
    q = q & tl.load(dry_ptr + tl.where(c < uncb - 1, t + 1, t), mask=m,
                    other=0)
    q = q & tl.load(dry_ptr + tl.where(j > 0, t - uncb, t), mask=m, other=0)
    q = q & tl.load(dry_ptr + tl.where(j < ny - 1, t + uncb, t), mask=m,
                    other=0)
    return q, t


@triton.jit
def _seg_row(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr, scr_ptr,
             pid, dt, gdt, dx, th, max_hflow, e103, ny, nx, ncb,
             NSCAL: tl.constexpr, BLOCK: tl.constexpr, CH: tl.constexpr,
             dry_ptr, qst_ptr, uncb, UB: tl.constexpr, QUIET: tl.constexpr):
    T_ = h_ptr.dtype.element_ty
    sx = nx + 1
    j = pid // ncb
    i0 = (pid - j * ncb) * BLOCK
    lane = tl.arange(0, BLOCK)
    i = i0 + lane
    k = j * nx + i
    m = (i < nx) & (dt > 0)
    if QUIET:
        # settled tiles (quiet now and at the previous step): their faces
        # are 0 and were 0, nothing to compute or store
        q, t = _quiet_lanes(dry_ptr, j, i, m, ny, uncb, UB)
        q = q & tl.load(qst_ptr + t, mask=m, other=0)
        m = m & (q == 0)
    me = m & (i < nx - 1)
    ms = m & (j < ny - 1)
    hc = tl.load(h_ptr + k, mask=m, other=0.0)
    he = tl.load(h_ptr + k + 1, mask=me, other=0.0)
    hs = tl.load(h_ptr + k + nx, mask=ms, other=0.0)
    zc = tl.load(z_ptr + k, mask=m, other=0.0)
    ze = tl.load(z_ptr + k + 1, mask=me, other=0.0)
    zs = tl.load(z_ptr + k + nx, mask=ms, other=0.0)
    # the wet test of _face (k_faces_mark's expressions)
    ye = tl.maximum(zc + hc, ze + he) - tl.maximum(zc, ze)
    ye = tl.minimum(tl.maximum(ye, 0.0), max_hflow)
    we = me & ((hc > th) | (he > th)) & (ye > th)
    ys = tl.maximum(zc + hc, zs + hs) - tl.maximum(zc, zs)
    ys = tl.minimum(tl.maximum(ys, 0.0), max_hflow)
    ws = ms & ((hc > th) | (hs > th)) & (ys > th)
    zero = tl.zeros((BLOCK,), dtype=T_)
    tl.store(Qx_ptr + j * sx + i + 1, zero, mask=me & (~we))
    tl.store(Qy_ptr + (j + 1) * nx + i, zero, mask=ms & (~ws))
    ie = we.to(tl.int32)
    is_ = ws.to(tl.int32)
    ne = tl.sum(ie, axis=0)
    nw = ne + tl.sum(is_, axis=0)
    if nw > 0:
        # the wet faces packed in this program's scratch (lane * 2 + south),
        # then evaluated CH at a time with the inputs k_faces2 gives _face
        base = scr_ptr + pid.to(tl.int64) * (2 * BLOCK)
        tl.store(base + tl.cumsum(ie, axis=0) - 1, lane * 2, mask=we)
        tl.store(base + ne + tl.cumsum(is_, axis=0) - 1, lane * 2 + 1,
                 mask=ws)
        tl.debug_barrier()
        for st in range(0, nw, CH):
            f = st + tl.arange(0, CH)
            mf = f < nw
            code = tl.load(base + f, mask=mf, other=0)
            ii = i0 + (code >> 1)
            so = (code & 1) == 1
            ea = mf & (~so)
            sm = mf & so
            kk = j * nx + ii
            k1 = tl.where(so, kk + nx, kk + 1)
            h0 = tl.load(h_ptr + kk, mask=mf, other=0.0)
            h1 = tl.load(h_ptr + k1, mask=mf, other=0.0)
            z0 = tl.load(z_ptr + kk, mask=mf, other=0.0)
            z1 = tl.load(z_ptr + k1, mask=mf, other=0.0)
            if NSCAL:
                n0 = tl.load(n_ptr)
                n1 = n0
            else:
                n0 = tl.load(n_ptr + kk, mask=mf, other=0.0)
                n1 = tl.load(n_ptr + k1, mask=mf, other=0.0)
            q0 = tl.where(so, tl.load(qy_ptr + (j + 1) * nx + ii, mask=sm,
                                      other=0.0),
                          tl.load(qx_ptr + j * sx + ii + 1, mask=ea,
                                  other=0.0))
            a1 = tl.where(so, tl.load(qx_ptr + j * sx + ii, mask=sm,
                                      other=0.0),
                          tl.load(qy_ptr + j * nx + ii, mask=ea, other=0.0))
            a2 = tl.where(so, tl.load(qx_ptr + j * sx + ii + 1, mask=sm,
                                      other=0.0),
                          tl.load(qy_ptr + j * nx + ii + 1, mask=ea,
                                  other=0.0))
            a3 = tl.where(so, tl.load(qx_ptr + (j + 1) * sx + ii, mask=sm,
                                      other=0.0),
                          tl.load(qy_ptr + (j + 1) * nx + ii, mask=ea,
                                  other=0.0))
            a4 = tl.where(so, tl.load(qx_ptr + (j + 1) * sx + ii + 1,
                                      mask=sm, other=0.0),
                          tl.load(qy_ptr + (j + 1) * nx + ii + 1, mask=ea,
                                  other=0.0))
            qo = (((a1 + a2) + a3) + a4) / 4.0
            Q = _face(h0, h1, z0, z1, q0, qo, 0.5 * (n0 + n1), gdt, dx, th,
                      max_hflow, e103)
            tl.store(Qx_ptr + j * sx + ii + 1, Q, mask=ea)
            tl.store(Qy_ptr + (j + 1) * nx + ii, Q, mask=sm)


@triton.jit
def k_faces_seg(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
                sw_ptr, se_ptr, sn_ptr, ss_ptr, dt_ptr, par_ptr, scr_ptr,
                ny, nx, ncb, NSCAL: tl.constexpr, BLOCK: tl.constexpr,
                CH: tl.constexpr, BBLOCK: tl.constexpr, dry_ptr=None,
                qst_ptr=None, uncb=1, UB: tl.constexpr = 1,
                QUIET: tl.constexpr = False):
    """k_faces_mark and k_faces_wet in one launch: each row-segment program
    tests its east and south faces, zeroes the dry ones and packs the wet
    ones in its own scratch slot (no global counter, no second launch),
    then evaluates them through _face with the inputs of k_faces_wet; the
    edge faces as in k_faces2. Same Q bit for bit."""
    T_ = h_ptr.dtype.element_ty
    dx = tl.load(par_ptr + 0).to(T_)
    g = tl.load(par_ptr + 1).to(T_)
    th = tl.load(par_ptr + 2).to(T_)
    max_hflow = tl.load(par_ptr + 3).to(T_)
    e103 = tl.load(par_ptr + 7).to(T_)
    dt = tl.load(dt_ptr).to(T_)
    gdt = g * dt
    pid = tl.program_id(0)
    # the edge programs first: their float64 power then overlaps the
    # row programs instead of trailing them
    nbe = (2 * (ny + nx) + BBLOCK - 1) // BBLOCK
    if pid >= nbe:
        pr = pid - nbe
        _seg_row(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
                 scr_ptr, pr, dt, gdt, dx, th, max_hflow, e103, ny, nx, ncb,
                 NSCAL, BLOCK, CH, dry_ptr, qst_ptr, uncb, UB, QUIET)
    else:
        _faces_edge(h_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
                    sw_ptr, se_ptr, sn_ptr, ss_ptr, pid, dt, gdt,
                    dx, th, e103, ny, nx, NSCAL, BBLOCK)


@triton.jit
def k_limiter2(h_ptr, Qx_ptr, Qy_ptr, w_ptr, dt_ptr, par_ptr, ny, nx,
               BLOCK: tl.constexpr, dry_ptr=None, uncb=1,
               UB: tl.constexpr = 1, QUIET: tl.constexpr = False,
               wf_ptr=None, WFLAG: tl.constexpr = False, fx_ptr=None,
               fy_ptr=None, QF: tl.constexpr = False):
    """k_limiter; a block with no limited cell stores w = 1 without the
    divisions (the same numbers: w is 1.0 there either way). WFLAG: such a
    block stores nothing and wf_ptr[block] = 0 tells k_update2 to use 1.0
    (cells are rarely limited: the factors are then neither written nor
    read)."""
    A = tl.load(par_ptr + 4).to(h_ptr.dtype.element_ty)
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    m = k < ny * nx
    dt = tl.load(dt_ptr).to(h_ptr.dtype.element_ty)
    m = m & (dt > 0)
    j = k // nx
    i = k - j * nx
    if QUIET:
        # quiet tiles: every face 0, the factor is never used
        q, t = _quiet_lanes(dry_ptr, j, i, m, ny, uncb, UB)
        m = m & (q == 0)
    sx = nx + 1
    mqw, mqe, mqn, mqs = _qmasks(fx_ptr, fy_ptr, j, i, m, m, nx, QF)
    qw = tl.load(Qx_ptr + j * sx + i, mask=mqw, other=0.0)
    qe = tl.load(Qx_ptr + j * sx + i + 1, mask=mqe, other=0.0)
    qn = tl.load(Qy_ptr + j * nx + i, mask=mqn, other=0.0)
    qs = tl.load(Qy_ptr + (j + 1) * nx + i, mask=mqs, other=0.0)
    # k_limiter's out and cv; max(-q, 0) and max(q, 0) by the sign bit
    # (a zero's sign cannot matter: an all-zero out is never limited)
    out = (((tl.where(_bits(qw) < 0, -qw, 0.0)
             + tl.where(_bits(qe) > 0, qe, 0.0))
            + tl.where(_bits(qn) < 0, -qn, 0.0))
           + tl.where(_bits(qs) > 0, qs, 0.0)) * dt
    cv = tl.load(h_ptr + k, mask=m, other=0.0) * A
    # out > cv: out >= +0; cv = -0 (a -0 depth) compares as +0
    lim = _bits(out) > tl.maximum(_bits(cv), 0)
    anyl = tl.max(lim.to(tl.int32), axis=0)
    if anyl > 0:
        w = tl.where(lim, cv / tl.where(lim, out, 1.0), 1.0)
        tl.store(w_ptr + k, w, mask=m)
    elif not WFLAG:
        w = tl.full((BLOCK,), 1.0, dtype=h_ptr.dtype.element_ty)
        tl.store(w_ptr + k, w, mask=m)
    if WFLAG:
        tl.store(wf_ptr + pid, anyl)


@triton.jit
def _qmasks(fx_ptr, fy_ptr, j, i, mx, my, nx, QF: tl.constexpr):
    """Load masks of a cell's four discharges (W, E, N, S): with QF only
    the faces k_faces_mark flagged wet (and the domain-edge faces, whose
    flags stay 1)."""
    if QF:
        sx = nx + 1
        mw = mx & (tl.load(fx_ptr + j * sx + i, mask=mx, other=0) != 0)
        me = mx & (tl.load(fx_ptr + j * sx + i + 1, mask=mx, other=0) != 0)
        mn = my & (tl.load(fy_ptr + j * nx + i, mask=my, other=0) != 0)
        ms = my & (tl.load(fy_ptr + (j + 1) * nx + i, mask=my,
                           other=0) != 0)
        return mw, me, mn, ms
    else:
        return mx, mx, my, my


@triton.jit
def _bits(x):
    """The bit pattern of a float as a signed integer of its width: for
    non-negative values (and any value against 0) the integer order is the
    floating-point order, so these compares run on the integer units (the
    float64 compares cost FP64 throughput, 1/64 rate on RTX cards)."""
    if x.dtype == tl.float64:
        return x.to(tl.int64, bitcast=True)
    else:
        return x.to(tl.int32, bitcast=True)


@triton.jit
def _wfac(qw, qe, qn, qs, h, dt, A):
    """Donor factor of k_limiter (same expression)."""
    out = (((tl.maximum(-qw, 0.0) + tl.maximum(qe, 0.0))
            + tl.maximum(-qn, 0.0)) + tl.maximum(qs, 0.0)) * dt
    cv = h * A
    lim = out > cv
    return tl.where(lim, cv / tl.where(lim, out, 1.0), 1.0)


@triton.jit
def _wout(qw, qe, qn, qs, h, dt, A):
    """Outgoing volume and stored volume of _wfac."""
    out = (((tl.maximum(-qw, 0.0) + tl.maximum(qe, 0.0))
            + tl.maximum(-qn, 0.0)) + tl.maximum(qs, 0.0)) * dt
    return out, h * A


@triton.jit
def _wdiv(out, cv):
    """The factor of _wfac from _wout's volumes."""
    lim = out > cv
    return tl.where(lim, cv / tl.where(lim, out, 1.0), 1.0)


@triton.jit
def _edge_acc_plain(eacc_ptr, dt, qw, qe, qn, qs, jb, i, m, ny, nx):
    D_ = tl.float64
    mw = m & (i == 0)
    p = eacc_ptr + jb
    tl.store(p, tl.load(p, mask=mw, other=0.0) - qw.to(D_) * dt, mask=mw)
    mE = m & (i == nx - 1)
    p = eacc_ptr + ny + jb
    tl.store(p, tl.load(p, mask=mE, other=0.0) + qe.to(D_) * dt, mask=mE)
    mn = m & (jb == 0)
    p = eacc_ptr + 2 * ny + i
    tl.store(p, tl.load(p, mask=mn, other=0.0) - qn.to(D_) * dt, mask=mn)
    mS = m & (jb == ny - 1)
    p = eacc_ptr + 2 * ny + nx + i
    tl.store(p, tl.load(p, mask=mS, other=0.0) + qs.to(D_) * dt, mask=mS)


@triton.jit
def _edge_acc(eacc_ptr, dt_ptr, qw, qe, qn, qs, jb, i, m, ny, nx,
              FMA: tl.constexpr):
    """eacc[face] += outflow x dt on the boundary faces, the expression of
    k_clock's EDGE loop as compiled: with contraction (FMA) fma(-dt, Q, e)
    on W and N faces, fma(dt, Q, e) on E and S faces; without it e - Q dt
    and e + Q dt (float64)."""
    D_ = tl.float64
    dt = tl.load(dt_ptr)
    if not FMA:
        _edge_acc_plain(eacc_ptr, dt, qw, qe, qn, qs, jb, i, m, ny, nx)
        return
    mw = m & (i == 0)
    p = eacc_ptr + jb
    tl.store(p, tl.fma(-dt, qw.to(D_), tl.load(p, mask=mw, other=0.0)),
             mask=mw)
    mE = m & (i == nx - 1)
    p = eacc_ptr + ny + jb
    tl.store(p, tl.fma(dt, qe.to(D_), tl.load(p, mask=mE, other=0.0)),
             mask=mE)
    mn = m & (jb == 0)
    p = eacc_ptr + 2 * ny + i
    tl.store(p, tl.fma(-dt, qn.to(D_), tl.load(p, mask=mn, other=0.0)),
             mask=mn)
    mS = m & (jb == ny - 1)
    p = eacc_ptr + 2 * ny + nx + i
    tl.store(p, tl.fma(dt, qs.to(D_), tl.load(p, mask=mS, other=0.0)),
             mask=mS)


@triton.jit
def k_update2(h_ptr, ho_ptr, src_ptr, Qx_ptr, Qy_ptr, qx_ptr, qy_ptr,
              hmax_ptr, bw_ptr, be_ptr, bn_ptr, bs_ptr, pm_ptr, acc_ptr,
              dt_ptr, par_ptr, w_ptr, eacc_ptr, ny, nx, ncb,
              LIMIT: tl.constexpr, WLOAD: tl.constexpr, BLOCK: tl.constexpr,
              NPM: tl.constexpr, EDGE: tl.constexpr,
              FMA: tl.constexpr, dry_in=None, dry_out=None, qst_ptr=None,
              stat_ptr=None, QUIET: tl.constexpr = False, t_ptr=None,
              T_ptr=None, RP_ptr=None, arr_ptr=None, fcnt_ptr=None,
              CNT: tl.constexpr = False, FCLOCK: tl.constexpr = False,
              wf_ptr=None, LB: tl.constexpr = 1,
              WFLAG: tl.constexpr = False, PMS: tl.constexpr = False,
              fx_ptr=None, fy_ptr=None, QF: tl.constexpr = False,
              r0=0, r1=0, ROWS: tl.constexpr = False):
    """k_update on row segments. WLOAD: the donor factors come from
    k_limiter (w_ptr) and the depth is updated in place (ho_ptr = h_ptr).
    Otherwise k_limiter is fused: the factors of the cell and its four
    neighbours are recomputed from the unscaled faces and the step-start
    depth h_ptr (same expression as k_limiter) and the new depth goes to
    ho_ptr (double-buffered so neighbours read the old depth); a step with
    dt = 0 then copies h. Block maxima of the new depth go to
    pm_ptr[program] (bit pattern), reduced by k_clock2. Compiled with FMA
    contraction off. FMA: whether Triton contracts a*b+c in this process
    (TRITON_DEFAULT_FP_FUSION, which torchvic.core.fusion sets to 0 at
    import): then the fma calls reproduce k_update's contraction, else the
    plain products and sums are k_update's uncontracted arithmetic.

    QUIET (dry-tile skipping; the segment is the tile): dry_in[tile] = 1
    when every depth of the segment is at or below the threshold at the
    step start. A quiet tile (it and its four neighbour tiles dry) has
    only dry faces, all Q = 0; a settled tile (quiet now and at the
    previous step, qst_ptr) also had them at the previous step, so its
    faces, momentum and edge values are 0 already: the kernel skips their
    loads and stores and evaluates the same expressions on the same zeros.
    A settled tile with no source and no -0 depth (stat_ptr) keeps its
    depth exactly and is skipped whole (its depths are at or below the
    threshold, so they cannot raise the interval maximum or the CFL depth
    that k_clock2 uses). dry_out gets the flags of the new depth.

    ROWS (multi-GPU strips, flood/multigpu.py): only rows r0 <= j < r1
    (the strip's own rows, not its halo) enter the maximum depth and the
    clamp volume; every cell is still updated. Off, the code is unchanged.

    FCLOCK: the program that arrives last (arrival counter arr_ptr, after
    its own stores and atomics) runs k_clock2's work: one launch less per
    step, the same expressions on the same values."""
    T_ = h_ptr.dtype.element_ty
    dx = tl.load(par_ptr + 0).to(T_)
    th = tl.load(par_ptr + 2).to(T_)
    A = tl.load(par_ptr + 4).to(T_)
    pid = tl.program_id(0)
    j = pid // ncb
    i = (pid - j * ncb) * BLOCK + tl.arange(0, BLOCK)
    k = j * nx + i
    m0 = i < nx
    dt = tl.load(dt_ptr).to(T_)
    m = m0 & (dt > 0)
    sx = nx + 1
    if QUIET:
        c = pid - j * ncb
        d0 = tl.load(dry_in + pid)
        dw = tl.load(dry_in + tl.where(c > 0, pid - 1, pid))
        de = tl.load(dry_in + tl.where(c < ncb - 1, pid + 1, pid))
        dn = tl.load(dry_in + tl.where(j > 0, pid - ncb, pid))
        ds = tl.load(dry_in + tl.where(j < ny - 1, pid + ncb, pid))
        quiet = (((d0 & dw) & (de & dn)) & ds)
        settled = quiet & tl.load(qst_ptr + pid)
        sskip = settled & tl.load(stat_ptr + pid)
        tl.debug_barrier()
        tl.store(qst_ptr + pid, quiet)
        mf = m & (settled == 0)
        mh0 = m0 & (sskip == 0)
    else:
        mf = m
        mh0 = m0
    mqw, mqe, mqn, mqs = _qmasks(fx_ptr, fy_ptr, j, i, mf, mf, nx, QF)
    qw = tl.load(Qx_ptr + j * sx + i, mask=mqw, other=0.0)
    qe = tl.load(Qx_ptr + j * sx + i + 1, mask=mqe, other=0.0)
    qn = tl.load(Qy_ptr + j * nx + i, mask=mqn, other=0.0)
    qs = tl.load(Qy_ptr + (j + 1) * nx + i, mask=mqs, other=0.0)
    h = tl.load(h_ptr + k, mask=mh0, other=0.0)
    if LIMIT and WLOAD:
        mc = mf
        mw_ = mf & (i > 0)
        me_ = mf & (i < nx - 1)
        mn_ = mf & (j > 0)
        ms_ = mf & (j < ny - 1)
        if WFLAG:
            # factors only from k_limiter2 blocks that limited a cell (the
            # others are 1.0 and were not stored)
            mc = mc & (tl.load(wf_ptr + k // LB, mask=mc, other=0) != 0)
            mw_ = mw_ & (tl.load(wf_ptr + (k - 1) // LB, mask=mw_,
                                 other=0) != 0)
            me_ = me_ & (tl.load(wf_ptr + (k + 1) // LB, mask=me_,
                                 other=0) != 0)
            mn_ = mn_ & (tl.load(wf_ptr + (k - nx) // LB, mask=mn_,
                                 other=0) != 0)
            ms_ = ms_ & (tl.load(wf_ptr + (k + nx) // LB, mask=ms_,
                                 other=0) != 0)
        wc = tl.load(w_ptr + k, mask=mc, other=1.0)
        ww = tl.load(w_ptr + k - 1, mask=mw_, other=1.0)
        we = tl.load(w_ptr + k + 1, mask=me_, other=1.0)
        wn = tl.load(w_ptr + k - nx, mask=mn_, other=1.0)
        wsv = tl.load(w_ptr + k + nx, mask=ms_, other=1.0)
    elif LIMIT:
        # the factors of the cell and its four neighbours (k_limiter's
        # expression); the divisions only in a block with a limited cell
        oc, cc = _wout(qw, qe, qn, qs, h, dt, A)
        mw = mf & (i > 0)
        ow, cw = _wout(tl.load(Qx_ptr + j * sx + i - 1, mask=mw, other=0.0),
                       qw,
                       tl.load(Qy_ptr + j * nx + i - 1, mask=mw, other=0.0),
                       tl.load(Qy_ptr + (j + 1) * nx + i - 1, mask=mw,
                               other=0.0),
                       tl.load(h_ptr + k - 1, mask=mw, other=0.0), dt, A)
        mE = mf & (i < nx - 1)
        oe, ce = _wout(qe,
                       tl.load(Qx_ptr + j * sx + i + 2, mask=mE, other=0.0),
                       tl.load(Qy_ptr + j * nx + i + 1, mask=mE, other=0.0),
                       tl.load(Qy_ptr + (j + 1) * nx + i + 1, mask=mE,
                               other=0.0),
                       tl.load(h_ptr + k + 1, mask=mE, other=0.0), dt, A)
        mn = mf & (j > 0)
        on_, cn = _wout(tl.load(Qx_ptr + (j - 1) * sx + i, mask=mn,
                                other=0.0),
                        tl.load(Qx_ptr + (j - 1) * sx + i + 1, mask=mn,
                                other=0.0),
                        tl.load(Qy_ptr + (j - 1) * nx + i, mask=mn,
                                other=0.0), qn,
                        tl.load(h_ptr + k - nx, mask=mn, other=0.0), dt, A)
        mS = mf & (j < ny - 1)
        os_, cs = _wout(tl.load(Qx_ptr + (j + 1) * sx + i, mask=mS,
                                other=0.0),
                        tl.load(Qx_ptr + (j + 1) * sx + i + 1, mask=mS,
                                other=0.0), qs,
                        tl.load(Qy_ptr + (j + 2) * nx + i, mask=mS,
                                other=0.0),
                        tl.load(h_ptr + k + nx, mask=mS, other=0.0), dt, A)
        anyl = (((oc > cc) | (ow > cw)) | ((oe > ce) | (on_ > cn))) | (
            os_ > cs)
        if tl.max(anyl.to(tl.int32), axis=0) > 0:
            wc = _wdiv(oc, cc)
            ww = _wdiv(ow, cw)
            we = _wdiv(oe, ce)
            wn = _wdiv(on_, cn)
            wsv = _wdiv(os_, cs)
        else:
            one = tl.full((BLOCK,), 1.0, dtype=T_)
            wc = one
            ww = one
            we = one
            wn = one
            wsv = one
    if LIMIT:
        ww = tl.where(i > 0, ww, wc)
        we = tl.where(i < nx - 1, we, wc)
        wn = tl.where(j > 0, wn, wc)
        wsv = tl.where(j < ny - 1, wsv, wc)
        # the face factors and the rounding of k_update as compiled (FMA
        # contraction on): the scaled west face is rounded, the other three
        # products enter the sum unrounded (this kernel is compiled with
        # contraction off, so the fma calls below are the only ones)
        fw = tl.where(_bits(qw) > 0, ww, wc)        # qw > 0
        fe = tl.where(_bits(qe) > 0, wc, we)
        fn = tl.where(_bits(qn) > 0, wn, wc)
        fs = tl.where(_bits(qs) > 0, wc, wsv)
        sw = qw * fw
        if FMA:
            dv = tl.fma(-fe, qe, sw)
            dv = tl.fma(fn, qn, dv)
            dv = tl.fma(-fs, qs, dv)
        else:
            dv = ((sw - qe * fe) + qn * fn) - qs * fs
        qw = sw
        qe = qe * fe
        qn = qn * fn
        qs = qs * fs
    else:
        dv = ((qw - qe) + qn) - qs
    src = tl.load(src_ptr + k, mask=m & mh0, other=0.0)
    if FMA:
        hs = tl.fma(dt, src, h)
    else:
        hs = h + src * dt
    hn = hs + dt * dv / A
    neg = tl.where(_bits(hn) < 0, -hn, 0.0)       # max(-hn, 0)
    hn = hn + neg
    hst = tl.where(m, hn, h)
    tl.store(ho_ptr + k, hst, mask=mh0)
    if QUIET:
        dnew = (tl.max(tl.where(m0, hst, 0.0), axis=0) <= th).to(tl.int32)
        tl.store(dry_out + pid, tl.where(sskip != 0, 1, dnew))
    inv = 1.0 / dx
    tl.store(qx_ptr + j * sx + i + 1, qe * inv, mask=mf)
    tl.store(qy_ptr + (j + 1) * nx + i, qs * inv, mask=mf)
    tl.store(qx_ptr + j * sx + i, qw * inv, mask=mf & (i == 0))
    tl.store(qy_ptr + i, qn * inv, mask=mf & (j == 0))
    jb = j + 0 * i                      # row index as a block
    tl.store(bw_ptr + jb, qw, mask=mf & (i == 0))
    tl.store(be_ptr + jb, qe, mask=mf & (i == nx - 1))
    tl.store(bn_ptr + i, qn, mask=mf & (j == 0))
    tl.store(bs_ptr + i, qs, mask=mf & (j == ny - 1))
    if EDGE:
        # outflow per boundary face (k_clock's accumulation, the same fma);
        # only programs that hold a boundary cell
        ce = pid - j * ncb
        if ((ce == 0) | (ce == ncb - 1)) | ((j == 0) | (j == ny - 1)):
            _edge_acc(eacc_ptr, dt_ptr, qw, qe, qn, qs, jb, i, mf, ny, nx,
                      FMA)
    if ROWS:
        mr = m & ((j >= r0) & (j < r1))
    else:
        mr = m
    hnb = _bits(hn)                                # hn >= 0 here
    wet = m & (hnb > _bits(th))
    hx = tl.load(hmax_ptr + k, mask=wet, other=0.0)
    tl.store(hmax_ptr + k, hn, mask=wet & (hnb > _bits(hx)))
    # (h >= 0: the bit pattern orders like the value, so the block maximum
    # is taken on the bit patterns, without floating-point compares; NPM
    # slots keep the atomics apart, k_clock2 takes their maximum and clears
    # them)
    if h_ptr.dtype.element_ty == tl.float64:
        hbb = tl.max(tl.where(mr, hnb, 0), axis=0)
        if PMS:
            tl.store(pm_ptr + pid, hbb)
        else:
            tl.atomic_max(pm_ptr + pid % NPM, hbb, sem='relaxed')
    else:
        hb = tl.max(tl.where(mr, hn, 0.0), axis=0)
        tl.atomic_max(pm_ptr + pid % NPM,
                      hb.to(tl.int32, bitcast=True).to(tl.int64),
                      sem='relaxed')
    if tl.max((mr & (_bits(neg) > 0)).to(tl.int32), axis=0) > 0:
        cl = tl.sum(tl.where(mr, neg, 0.0).to(tl.float64), axis=0) * A.to(
            tl.float64)
        tl.atomic_add(acc_ptr + 3, cl, sem='relaxed')
    if FCLOCK:
        tl.debug_barrier()
        arr = tl.atomic_add(arr_ptr, 1, sem='acq_rel')
        if arr == tl.num_programs(0) - 1:
            _clock_body(bw_ptr, be_ptr, bn_ptr, bs_ptr, pm_ptr, acc_ptr,
                        dt_ptr, t_ptr, T_ptr, RP_ptr, par_ptr, eacc_ptr, ny,
                        nx, h_ptr.dtype.element_ty == tl.float64, False,
                        1024, NPM, fcnt_ptr, CNT, FMA, tl.num_programs(0),
                        PMS)
            tl.store(arr_ptr, 0)


@triton.jit
def _ldcg(p, mm):
    return tl.load(p, mask=mm, other=0.0, cache_modifier='.cg')


@triton.jit
def _clock_body(bw_ptr, be_ptr, bn_ptr, bs_ptr, pm_ptr, acc_ptr, dt_ptr,
                t_ptr, T_ptr, RP_ptr, par_ptr, eacc_ptr, ny, nx,
                F64: tl.constexpr, EDGE: tl.constexpr, BLOCK: tl.constexpr,
                NPM: tl.constexpr, cnt_ptr, CNT: tl.constexpr,
                FMA: tl.constexpr, npm=0, PMS: tl.constexpr = False):
    """k_clock2's work (loads through L2: k_update2's last program runs it
    on values the other programs wrote in the same launch). FMA: the two
    ledger sums a + R dt and a + P dt as Triton contracts them (fma) or
    not, so the work gives the same bits inside k_update2 (compiled with
    contraction off) as in k_clock2."""
    dx = tl.load(par_ptr + 0, cache_modifier='.cg')
    g = tl.load(par_ptr + 1, cache_modifier='.cg')
    th = tl.load(par_ptr + 2, cache_modifier='.cg')
    cfl = tl.load(par_ptr + 5, cache_modifier='.cg')
    max_dt = tl.load(par_ptr + 6, cache_modifier='.cg')
    dt = tl.load(dt_ptr, cache_modifier='.cg')          # float64
    o = tl.arange(0, BLOCK)
    if PMS:
        # one maximum per k_update2 program (each program overwrites its
        # own every step: nothing to clear)
        bits = tl.max(tl.load(pm_ptr + o, mask=o < npm, other=0,
                              cache_modifier='.cg'), axis=0)
        for s in range(BLOCK, npm, BLOCK):
            bits = tl.maximum(bits, tl.max(tl.load(
                pm_ptr + s + o, mask=s + o < npm, other=0,
                cache_modifier='.cg'), axis=0))
    else:
        op = tl.arange(0, NPM)
        bits = tl.max(tl.load(pm_ptr + op, cache_modifier='.cg'), axis=0)
        tl.store(pm_ptr + op, tl.zeros((NPM,), dtype=tl.int64))
    if CNT:
        tl.store(cnt_ptr, 0)            # the wet-face list of k_faces_mark
    a0 = tl.load(acc_ptr + 0, cache_modifier='.cg')
    a1 = tl.load(acc_ptr + 1, cache_modifier='.cg')
    a2 = tl.load(acc_ptr + 2, cache_modifier='.cg')
    a4 = tl.load(acc_ptr + 4, cache_modifier='.cg')
    a5 = tl.load(acc_ptr + 5, cache_modifier='.cg')
    a6 = tl.load(acc_ptr + 6, cache_modifier='.cg')
    a7 = tl.load(acc_ptr + 7, cache_modifier='.cg')
    R = tl.load(RP_ptr, cache_modifier='.cg')
    P = tl.load(RP_ptr + 1, cache_modifier='.cg')
    t0 = tl.load(t_ptr, cache_modifier='.cg')
    Tend = tl.load(T_ptr, cache_modifier='.cg')
    # (k_clock2 is launched with EDGE=False: k_update2 accumulates the
    # per-face boundary outflow)
    tl.static_assert(not EDGE)
    sx = tl.zeros((BLOCK,), dtype=tl.float64)
    for s in range(0, ny, BLOCK):
        mm = s + o < ny
        sx += (_ldcg(be_ptr + s + o, mm).to(tl.float64)
               - _ldcg(bw_ptr + s + o, mm).to(tl.float64))
    sy = tl.zeros((BLOCK,), dtype=tl.float64)
    for s in range(0, nx, BLOCK):
        mm = s + o < nx
        sy += (_ldcg(bs_ptr + s + o, mm).to(tl.float64)
               - _ldcg(bn_ptr + s + o, mm).to(tl.float64))
    live = dt > 0
    out = tl.where(live, (tl.sum(sx, axis=0) + tl.sum(sy, axis=0)) * dt,
                   0.0)
    if FMA:
        tl.store(acc_ptr + 0, tl.fma(R, dt, a0))
        tl.store(acc_ptr + 1, tl.fma(P, dt, a1))
    else:
        tl.store(acc_ptr + 0, a0 + R * dt)
        tl.store(acc_ptr + 1, a1 + P * dt)
    tl.store(acc_ptr + 2, a2 + out)
    tl.store(acc_ptr + 4, a4 + tl.where(live, 1.0, 0.0))
    t = t0 + dt
    tl.store(t_ptr, t)
    if F64:
        hm = bits.to(tl.float64, bitcast=True)
    else:
        hm = bits.to(tl.int32).to(tl.float32, bitcast=True).to(tl.float64)
    loc = cfl * dx / tl.sqrt(g * tl.maximum(hm, th))
    d = tl.where(hm > th, tl.minimum(loc, max_dt), max_dt)
    rem = Tend - t
    dn = tl.where(rem > 0, tl.minimum(d, rem), 0.0)
    tl.store(dt_ptr, dn)
    on = dn > 0
    tl.store(acc_ptr + 5, tl.where(on, tl.minimum(a5, dn), a5))
    tl.store(acc_ptr + 6, tl.where(on, tl.maximum(a6, dn), a6))
    tl.store(acc_ptr + 7, tl.where(on, tl.minimum(a7, d), a7))


@triton.jit
def k_clock2(bw_ptr, be_ptr, bn_ptr, bs_ptr, pm_ptr, npm, acc_ptr, dt_ptr,
             t_ptr, T_ptr, RP_ptr, par_ptr, eacc_ptr, ny, nx,
             F64: tl.constexpr, EDGE: tl.constexpr, BLOCK: tl.constexpr,
             NPM: tl.constexpr, cnt_ptr=None, CNT: tl.constexpr = False,
             FMA: tl.constexpr = False, PMS: tl.constexpr = False):
    """k_clock with the maximum depth from the NPM slots k_update2 wrote
    (max is exact in any order); the edge sums are k_clock's."""
    _clock_body(bw_ptr, be_ptr, bn_ptr, bs_ptr, pm_ptr, acc_ptr, dt_ptr,
                t_ptr, T_ptr, RP_ptr, par_ptr, eacc_ptr, ny, nx, F64, EDGE,
                BLOCK, NPM, cnt_ptr, CNT, FMA, npm, PMS)


# ---------------------------------------------------------------------------
# mixed precision (option `mixed=True` of FloodEngine; not the default)
#
# Depth, sources, maximum depth, the clock and the ledger stay float64; the
# face discharges, momentum and donor factors are float32. Every face value
# is computed once and used, in float64, by both of its cells, and the
# continuity sum is float64, so the water balance still closes to float64
# round-off; the depths differ from the float64 engine by the float32
# rounding of the fluxes (the bound is in docs/connectors/flood.md).
# The face uses the bed step dz = z0 - z1 (float32, from the float64 DEM):
#   hflow = max(h0 + min(dz, 0), h1 - max(dz, 0)),  y0 - y1 = dz + (h0 - h1)
# (exactly max(y0, y1) - max(z0, z1) and y0 - y1 of the float64 face), so a
# float32 depth is never added to a float32 elevation.
# ---------------------------------------------------------------------------
@triton.jit
def _face_mx(h0, l0, h1, l1, dz, dzl, q0, qo, fn, gdt, dx, th, max_hflow,
             e103):
    qvect = tl.sqrt(q0 * q0 + qo * qo)
    # the differences of depths and beds from float32 pairs (h0 + l0, ...),
    # so a deep, nearly flat water surface keeps its slope
    s, e = _two_sum(h0, -h1)
    e = e + (l0 - l1)
    s, e2 = _two_sum(s, dz)
    dh = s + (e2 + (e + dzl))
    Sf = -dh / dx
    neg = dz < 0.0
    s, e = _two_sum(h0, tl.minimum(dz, 0.0))
    y0 = s + (e + (l0 + tl.where(neg, dzl, 0.0)))
    s, e = _two_sum(h1, -tl.maximum(dz, 0.0))
    y1 = s + (e + (l1 - tl.where(neg, 0.0, dzl)))
    hflow = tl.maximum(y0, y1)
    hflow = tl.minimum(tl.maximum(hflow, 0.0), max_hflow)
    a = gdt * hflow
    x = tl.maximum(hflow, th)
    p = x * x * x * libdevice.cbrt(x)          # x^(10/3), a few ulp
    den = 1.0 + a * fn * fn * tl.abs(qvect) / p
    Q = (q0 - a * Sf) / den * dx
    wet = ((h0 > th) | (h1 > th)) & (hflow > th)
    return tl.where(wet, Q, 0.0)


@triton.jit
def _faces_row_mx(h_ptr, hl_ptr, dzx_ptr, dzy_ptr, n_ptr, qx_ptr, qy_ptr,
                  Qx_ptr,
                  Qy_ptr, pid, dt, gdt, dx, th, max_hflow, e103, ny, nx, ncb,
                  NSCAL: tl.constexpr, SKIP: tl.constexpr,
                  BLOCK: tl.constexpr):
    F_ = tl.float32
    sx = nx + 1
    j = pid // ncb
    i = (pid - j * ncb) * BLOCK + tl.arange(0, BLOCK)
    k = j * nx + i
    m = (i < nx) & (dt > 0)
    me = m & (i < nx - 1)
    ms = m & (j < ny - 1)
    hc = tl.load(h_ptr + k, mask=m, other=0.0)
    he = tl.load(h_ptr + k + 1, mask=me, other=0.0)
    hs = tl.load(h_ptr + k + nx, mask=ms, other=0.0)
    if SKIP:
        act = tl.max(((hc > th) | (he > th) | (hs > th)).to(tl.int32),
                     axis=0)
    else:
        act = 1
    if act > 0:
        if NSCAL:
            nc = tl.load(n_ptr).to(F_)
            ne = nc
            ns = nc
        else:
            nc = tl.load(n_ptr + k, mask=m, other=0.0).to(F_)
            ne = tl.load(n_ptr + k + 1, mask=me, other=0.0).to(F_)
            ns = tl.load(n_ptr + k + nx, mask=ms, other=0.0).to(F_)
        q0 = tl.load(qx_ptr + j * sx + i + 1, mask=m, other=0.0)
        a1 = tl.load(qy_ptr + j * nx + i, mask=me, other=0.0)
        a2 = tl.load(qy_ptr + j * nx + i + 1, mask=me, other=0.0)
        a3 = tl.load(qy_ptr + (j + 1) * nx + i, mask=me, other=0.0)
        a4 = tl.load(qy_ptr + (j + 1) * nx + i + 1, mask=me, other=0.0)
        qyb = (((a1 + a2) + a3) + a4) * 0.25
        lc = tl.load(hl_ptr + k, mask=m, other=0.0)
        le = tl.load(hl_ptr + k + 1, mask=me, other=0.0)
        dze = tl.load(dzx_ptr + k, mask=me, other=0.0)
        dzel = tl.load(dzx_ptr + ny * nx + k, mask=me, other=0.0)
        Qe = _face_mx(hc, lc, he, le, dze, dzel, q0, qyb, 0.5 * (nc + ne),
                      gdt, dx, th,
                      max_hflow, e103)
        tl.store(Qx_ptr + j * sx + i + 1, Qe, mask=me)
        q0s = tl.load(qy_ptr + (j + 1) * nx + i, mask=m, other=0.0)
        b1 = tl.load(qx_ptr + j * sx + i, mask=ms, other=0.0)
        b2 = tl.load(qx_ptr + j * sx + i + 1, mask=ms, other=0.0)
        b3 = tl.load(qx_ptr + (j + 1) * sx + i, mask=ms, other=0.0)
        b4 = tl.load(qx_ptr + (j + 1) * sx + i + 1, mask=ms, other=0.0)
        qxb = (((b1 + b2) + b3) + b4) * 0.25
        ls = tl.load(hl_ptr + k + nx, mask=ms, other=0.0)
        dzs = tl.load(dzy_ptr + k, mask=ms, other=0.0)
        dzsl = tl.load(dzy_ptr + ny * nx + k, mask=ms, other=0.0)
        Qs = _face_mx(hc, lc, hs, ls, dzs, dzsl, q0s, qxb, 0.5 * (nc + ns),
                      gdt, dx, th,
                      max_hflow, e103)
        tl.store(Qy_ptr + (j + 1) * nx + i, Qs, mask=ms)
    else:
        zero = tl.zeros((BLOCK,), dtype=F_)
        tl.store(Qx_ptr + j * sx + i + 1, zero, mask=me)
        tl.store(Qy_ptr + (j + 1) * nx + i, zero, mask=ms)


@triton.jit
def _faces_edge_mx(h_ptr, hl_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr, Qy_ptr,
                   sw_ptr, se_ptr, sn_ptr, ss_ptr, pb, dt, gdt, dx, th, e103,
                   ny, nx, NSCAL: tl.constexpr, BBLOCK: tl.constexpr):
    F_ = tl.float32
    sx = nx + 1
    b = pb * BBLOCK + tl.arange(0, BBLOCK)
    m = (b < 2 * (ny + nx)) & (dt > 0)
    isx = b < 2 * ny
    east = (b >= ny) & isx
    south = b >= 2 * ny + nx
    r = tl.where(isx, tl.where(east, b - ny, b), 0)
    c = tl.where(isx, 0, tl.where(south, b - 2 * ny - nx, b - 2 * ny))
    j0 = tl.where(isx, r, tl.where(south, ny - 1, 0))
    i0 = tl.where(isx, tl.where(east, nx - 1, 0), c)
    j1 = tl.where(isx, j0, tl.where(south, ny - 2, 1))
    i1 = tl.where(isx, tl.where(east, nx - 2, 1), i0)
    k0 = j0 * nx + i0
    k1 = j1 * nx + i1
    D_ = tl.float64
    h0 = (tl.load(h_ptr + k0, mask=m, other=0.0).to(D_)
          + tl.load(hl_ptr + k0, mask=m, other=0.0).to(D_))
    h1 = (tl.load(h_ptr + k1, mask=m, other=0.0).to(D_)
          + tl.load(hl_ptr + k1, mask=m, other=0.0).to(D_))
    z0 = tl.load(z_ptr + k0, mask=m, other=0.0)
    z1 = tl.load(z_ptr + k1, mask=m, other=0.0)
    # float64 water-surface difference, then float32
    dh = (((z0 + h0) - z1) - h1).to(F_)
    h0 = h0.to(F_)
    if NSCAL:
        fn = tl.load(n_ptr).to(F_) + tl.zeros((BBLOCK,), dtype=F_)
    else:
        fn = tl.load(n_ptr + k0, mask=m, other=0.0).to(F_)
    fx = tl.where(east, j0 * sx + nx, j0 * sx)
    fy = tl.where(south, ny * nx + i0, i0)
    q0 = tl.where(isx, tl.load(qx_ptr + fx, mask=m & isx, other=0.0),
                  tl.load(qy_ptr + fy, mask=m & (~isx), other=0.0))
    sl = tl.where(isx, tl.where(east,
                                tl.load(se_ptr + r, mask=m & east,
                                        other=float('nan')),
                                tl.load(sw_ptr + r, mask=m & isx & (~east),
                                        other=float('nan'))),
                  tl.where(south,
                           tl.load(ss_ptr + c, mask=m & south,
                                   other=float('nan')),
                           tl.load(sn_ptr + c, mask=m & (~isx) & (~south),
                                   other=float('nan')))).to(F_)
    sign = tl.where(east | south, 1.0, -1.0).to(F_)
    hflow = h0
    Sf = tl.where(sl < -0.999, -dh / dx, sl)
    a = gdt * hflow
    x = tl.maximum(hflow, th)
    p = x * x * x * libdevice.cbrt(x)          # x^(10/3), a few ulp
    q = sign * (tl.abs(q0) + tl.abs(a * Sf)) / (
        1.0 + a * fn * fn * tl.abs(q0) / p)
    Q = tl.where((sl == sl) & (h0 > th), q * dx, 0.0)
    tl.store(Qx_ptr + fx, Q, mask=m & isx)
    tl.store(Qy_ptr + fy, Q, mask=m & (~isx))


@triton.jit
def k_faces_mx(h_ptr, hl_ptr, z_ptr, dzx_ptr, dzy_ptr, n_ptr, qx_ptr, qy_ptr,
               Qx_ptr,
               Qy_ptr, sw_ptr, se_ptr, sn_ptr, ss_ptr, dt_ptr, par_ptr,
               ny, nx, ncb, NSCAL: tl.constexpr, SKIP: tl.constexpr,
               BLOCK: tl.constexpr, BBLOCK: tl.constexpr):
    F_ = tl.float32
    dx = tl.load(par_ptr + 0).to(F_)
    g = tl.load(par_ptr + 1).to(F_)
    th = tl.load(par_ptr + 2).to(F_)
    max_hflow = tl.load(par_ptr + 3).to(F_)
    e103 = tl.load(par_ptr + 7).to(F_)
    dt = tl.load(dt_ptr).to(F_)
    gdt = g * dt
    pid = tl.program_id(0)
    # the edge programs first: their float64 power then overlaps the
    # row programs instead of trailing them
    nbe = (2 * (ny + nx) + BBLOCK - 1) // BBLOCK
    if pid >= nbe:
        pr = pid - nbe
        _faces_row_mx(h_ptr, hl_ptr, dzx_ptr, dzy_ptr, n_ptr, qx_ptr, qy_ptr,
                      Qx_ptr,
                      Qy_ptr, pr, dt, gdt, dx, th, max_hflow, e103, ny, nx,
                      ncb, NSCAL, SKIP, BLOCK)
    else:
        _faces_edge_mx(h_ptr, hl_ptr, z_ptr, n_ptr, qx_ptr, qy_ptr, Qx_ptr,
                       Qy_ptr,
                       sw_ptr, se_ptr, sn_ptr, ss_ptr, pid, dt,
                       gdt, dx, th, e103, ny, nx, NSCAL, BBLOCK)


@triton.jit
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@triton.jit
def _df_add(ah, al, bh, bl):
    s, e = _two_sum(ah, bh)
    e = e + (al + bl)
    h = s + e
    return h, e - (h - s)


@triton.jit
def _df_mul(ah, al, bh, bl):
    p = ah * bh
    e = tl.fma(ah, bh, -p)
    e = e + (ah * bl + al * bh)
    h = p + e
    return h, e - (h - p)


@triton.jit
def k_update_mx(h_ptr, hl_ptr, ho_ptr, hlo_ptr, shi_ptr, slo_ptr, Qx_ptr,
                Qy_ptr, qx_ptr, qy_ptr, hmax_ptr, bw_ptr, be_ptr, bn_ptr,
                bs_ptr, pm_ptr, acc_ptr, dt_ptr, par_ptr, eacc_ptr, ny, nx,
                ncb, LIMIT: tl.constexpr, BLOCK: tl.constexpr,
                NPM: tl.constexpr, EDGE: tl.constexpr,
                FMA: tl.constexpr, WSKIP: tl.constexpr = False):
    """k_update2 (fused limiter) with float32 faces and donor factors.
    WSKIP: the five factor divisions only in a block with a limited cell
    (the same numbers). The
    continuity runs in float32 pairs (double-float, about 48 bits; no
    float64 arithmetic, which is slow on RTX cards): depth (h_ptr + hl_ptr)
    + sources (shi + slo) x dt + the four faces x dt / A, stored as a pair;
    the interval maximum is float32. Compiled with FMA contraction off (the
    error-free transforms need it)."""
    F_ = tl.float32
    D_ = tl.float64
    dx = tl.load(par_ptr + 0).to(F_)
    th = tl.load(par_ptr + 2).to(F_)
    A64 = tl.load(par_ptr + 4)
    A = A64.to(F_)
    pid = tl.program_id(0)
    j = pid // ncb
    i = (pid - j * ncb) * BLOCK + tl.arange(0, BLOCK)
    k = j * nx + i
    m0 = i < nx
    dt64 = tl.load(dt_ptr)
    dt = dt64.to(F_)
    dtl = (dt64 - dt.to(D_)).to(F_)
    dtA64 = dt64 / A64
    dA = dtA64.to(F_)
    dAl = (dtA64 - dA.to(D_)).to(F_)
    m = m0 & (dt64 > 0)
    sx = nx + 1
    qw = tl.load(Qx_ptr + j * sx + i, mask=m, other=0.0)
    qe = tl.load(Qx_ptr + j * sx + i + 1, mask=m, other=0.0)
    qn = tl.load(Qy_ptr + j * nx + i, mask=m, other=0.0)
    qs = tl.load(Qy_ptr + (j + 1) * nx + i, mask=m, other=0.0)
    hh = tl.load(h_ptr + k, mask=m0, other=0.0)
    hl = tl.load(hl_ptr + k, mask=m0, other=0.0)
    if LIMIT:
        oc, cc = _wout(qw, qe, qn, qs, hh, dt, A)
        mw = m & (i > 0)
        ow, cw = _wout(
            tl.load(Qx_ptr + j * sx + i - 1, mask=mw, other=0.0), qw,
            tl.load(Qy_ptr + j * nx + i - 1, mask=mw, other=0.0),
            tl.load(Qy_ptr + (j + 1) * nx + i - 1, mask=mw, other=0.0),
            tl.load(h_ptr + k - 1, mask=mw, other=0.0), dt, A)
        mE = m & (i < nx - 1)
        oe, ce = _wout(
            qe, tl.load(Qx_ptr + j * sx + i + 2, mask=mE, other=0.0),
            tl.load(Qy_ptr + j * nx + i + 1, mask=mE, other=0.0),
            tl.load(Qy_ptr + (j + 1) * nx + i + 1, mask=mE, other=0.0),
            tl.load(h_ptr + k + 1, mask=mE, other=0.0), dt, A)
        mn = m & (j > 0)
        on_, cn = _wout(
            tl.load(Qx_ptr + (j - 1) * sx + i, mask=mn, other=0.0),
            tl.load(Qx_ptr + (j - 1) * sx + i + 1, mask=mn, other=0.0),
            tl.load(Qy_ptr + (j - 1) * nx + i, mask=mn, other=0.0), qn,
            tl.load(h_ptr + k - nx, mask=mn, other=0.0), dt, A)
        mS = m & (j < ny - 1)
        os_, cs = _wout(
            tl.load(Qx_ptr + (j + 1) * sx + i, mask=mS, other=0.0),
            tl.load(Qx_ptr + (j + 1) * sx + i + 1, mask=mS, other=0.0), qs,
            tl.load(Qy_ptr + (j + 2) * nx + i, mask=mS, other=0.0),
            tl.load(h_ptr + k + nx, mask=mS, other=0.0), dt, A)
        if WSKIP:
            anyl = tl.max(((((oc > cc) | (ow > cw)) | ((oe > ce)
                                                       | (on_ > cn)))
                           | (os_ > cs)).to(tl.int32), axis=0)
        else:
            anyl = 1
        if anyl > 0:
            wc = _wdiv(oc, cc)
            ww = _wdiv(ow, cw)
            we = _wdiv(oe, ce)
            wn = _wdiv(on_, cn)
            wsv = _wdiv(os_, cs)
        else:
            one = tl.full((BLOCK,), 1.0, dtype=F_)
            wc = one
            ww = one
            we = one
            wn = one
            wsv = one
        ww = tl.where(i > 0, ww, wc)
        we = tl.where(i < nx - 1, we, wc)
        wn = tl.where(j > 0, wn, wc)
        wsv = tl.where(j < ny - 1, wsv, wc)
        qw = tl.where(qw > 0, qw * ww, qw * wc)
        qe = tl.where(qe > 0, qe * wc, qe * we)
        qn = tl.where(qn > 0, qn * wn, qn * wc)
        qs = tl.where(qs > 0, qs * wc, qs * wsv)
    # double-float continuity
    shi = tl.load(shi_ptr + k, mask=m, other=0.0)
    slo = tl.load(slo_ptr + k, mask=m, other=0.0)
    ph, pl = _df_mul(shi, slo, dt, dtl)
    ah, al = _df_add(hh, hl, ph, pl)
    vh, vl = _two_sum(qw, -qe)
    vh, vl = _df_add(vh, vl, qn, 0.0)
    vh, vl = _df_add(vh, vl, -qs, 0.0)
    vh, vl = _df_mul(vh, vl, dA, dAl)
    nh, nl = _df_add(ah, al, vh, vl)
    isneg = m & (nh < 0.0)            # a normalised pair has the sign of nh
    if tl.max(isneg.to(tl.int32), axis=0) > 0:
        cl = tl.sum(tl.where(isneg, -(nh.to(D_) + nl.to(D_)), 0.0),
                    axis=0) * A64
        tl.atomic_add(acc_ptr + 3, cl, sem='relaxed')
    nh = tl.where(isneg, 0.0, nh)
    nl = tl.where(isneg, 0.0, nl)
    tl.store(ho_ptr + k, tl.where(m, nh, hh), mask=m0)
    tl.store(hlo_ptr + k, tl.where(m, nl, hl), mask=m0)
    inv = 1.0 / dx
    tl.store(qx_ptr + j * sx + i + 1, qe * inv, mask=m)
    tl.store(qy_ptr + (j + 1) * nx + i, qs * inv, mask=m)
    tl.store(qx_ptr + j * sx + i, qw * inv, mask=m & (i == 0))
    tl.store(qy_ptr + i, qn * inv, mask=m & (j == 0))
    jb = j + 0 * i
    tl.store(bw_ptr + jb, qw, mask=m & (i == 0))
    tl.store(be_ptr + jb, qe, mask=m & (i == nx - 1))
    tl.store(bn_ptr + i, qn, mask=m & (j == 0))
    tl.store(bs_ptr + i, qs, mask=m & (j == ny - 1))
    if EDGE:
        ce = pid - j * ncb
        if ((ce == 0) | (ce == ncb - 1)) | ((j == 0) | (j == ny - 1)):
            _edge_acc(eacc_ptr, dt_ptr, qw, qe, qn, qs, jb, i, m, ny, nx,
                      FMA)
    hn32 = nh
    wet = m & (hn32 > th)
    hx = tl.load(hmax_ptr + k, mask=wet, other=0.0)
    tl.store(hmax_ptr + k, hn32, mask=wet & (hn32 > hx))
    # float32 maximum for the next step (k_clock2 with F64=False)
    hb = tl.max(tl.where(m, hn32, 0.0), axis=0)
    tl.atomic_max(pm_ptr + pid % NPM,
                  hb.to(tl.int32, bitcast=True).to(tl.int64),
                  sem='relaxed')


def _fma():
    """Triton's own default for FMA contraction (TRITON_DEFAULT_FP_FUSION;
    torchvic.core.fusion sets 0 at import). Every flood launch passes it as
    enable_fp_fusion (Triton's in-process kernel cache does not see a
    change of the variable once a kernel is compiled), and k_update2's
    explicit arithmetic follows it, so the default and reference paths
    compile under the same rule and stay bit-identical."""
    return os.getenv('TRITON_DEFAULT_FP_FUSION', '1') == '1'


class TritonStepper(object):
    """Buffers and launches for FloodEngine (scheme 'torchvic', CUDA).

    Default float64 step (fused=True, compact=True), five launches:
    k_faces_mark (wet test of every interior face, Q = 0 on the dry ones,
    a list of the wet ones; the edge faces), k_faces_wet (the listed faces
    through _face), k_limiter2, k_update2 (in place, row segments), k_clock2.
    Every face and cell gets the numbers of the reference path (fused=False)
    bit for bit (tests/test_flood_gpu.py); the gain is that the float64
    power and divisions run only on wet faces. compact=False: one face
    kernel (k_faces2, dry row segments skipped). wfuse=True: the limiter
    inside k_update2 (depth double-buffered). With the engine's mixed=True:
    k_faces_mx, k_update_mx, k_clock2 (not bitwise; see the mixed-precision
    section above)."""

    def __init__(self, eng, block=None, num_warps=None, split=False,
                 fused=True, skip_dry=True, ublock=None, unum_warps=None,
                 wfuse=None, npm=None, compact=True, cblock=128, cprog=None,
                 seg=False, sblock=256, sch=256, snw=8, quiet=False,
                 fclock=False, wflag=False, mskip=False, pms=False,
                 qflag=False, mblock=512, mnum_warps=8):
        mixed = bool(getattr(eng, 'mixed', False))
        # measured on H200 and RTX 6000 Ada (1.1 M cells, Guwahati): 128-cell
        # blocks for the float64 face kernel of the reference path (no
        # register spills), 256 cells / 8 warps for the compacted path
        if block is None:
            block = 256 if fused and compact and not mixed else 128
            if num_warps is None:
                num_warps = 8 if block == 256 else 4
        if num_warps is None:
            num_warps = 4
        self.split = split
        # (r0, r1): the own rows of a multi-GPU strip (k_update2's ROWS)
        self.rows = None
        self.fused = bool(fused)
        self.skip = bool(skip_dry)
        self.e = eng
        dev, dt_ = eng.device, eng.dtype
        ny, nx = eng.ny, eng.nx
        self.BLOCK, self.nw = block, num_warps
        self.grid = (triton.cdiv(ny * nx, block),)
        self.bgrid = (triton.cdiv(2 * (ny + nx), 256),)
        n = eng.n_cell
        self.nscal = n.dim() == 0
        self.n = (n.expand(ny, nx).contiguous() if n.dim() == 0
                  else n.contiguous())
        if self.fused and self.nscal:
            self.n = n.reshape(1).contiguous()
        nan = float('nan')
        sl = {}
        for e, m in (('W', ny), ('E', ny), ('N', nx), ('S', nx)):
            s = torch.full((m,), nan, dtype=dt_, device=dev)
            if e in eng.edges:
                mask, slope = eng.edges[e]
                s = torch.where(mask, slope, s)
            sl[e] = s.contiguous()
        self.sl = sl
        self.w = torch.ones(ny, nx, dtype=dt_, device=dev)
        self.b = {e: torch.zeros(ny if e in 'WE' else nx, dtype=dt_,
                                 device=dev) for e in 'WENS'}
        self.hm = torch.zeros(1, dtype=torch.int64, device=dev)
        self.dt = torch.zeros(1, dtype=torch.float64, device=dev)
        self.RP = torch.zeros(2, dtype=torch.float64, device=dev)
        self.par = torch.tensor([eng.dx, eng.g, eng.th, eng.max_hflow,
                                 eng.area, eng.cfl, eng.max_dt, 10.0 / 3.0],
                                dtype=torch.float64, device=dev)
        # fused path: row-segment programs, second depth buffer, per-program
        # maxima of the new depth
        self.ncb = triton.cdiv(nx, block)
        self.nbp = triton.cdiv(2 * (ny + nx), 256)
        self.fgrid = (ny * self.ncb + self.nbp,)
        # update row segments: 128 cells (float64; H200 31.7 vs 33.7 us per
        # step on Guwahati), 256 for mixed precision
        self.UB = int(ublock or (256 if mixed else 128))
        self.unw = int(unum_warps or 4)
        self.uncb = triton.cdiv(nx, self.UB)
        self.ugrid = (ny * self.uncb,)
        self.NPM = int(npm or 32)
        # compacted faces (float64 path): k_faces_mark lists the wet faces,
        # k_faces_wet evaluates only those
        self.compact = bool(compact) and self.fused and not mixed
        self.cblock = int(cblock)
        if self.compact:
            self.flist = torch.zeros(2 * ny * nx, dtype=torch.int32,
                                     device=dev)
            self.fcnt = torch.zeros(1, dtype=torch.int32, device=dev)
            nsm = torch.cuda.get_device_properties(dev).multi_processor_count
            # (H200, combo-16 monsoon state: 16 programs per SM 15.8 us,
            # 8 per SM 16.9 us)
            self.cgrid = (int(cprog or 16 * nsm),)
            # k_faces_mark row segments: 512 cells / 8 warps (fewer list
            # atomics; H200 20.9 vs 21.8 us at 256)
            self.MB, self.mnw = int(mblock), int(mnum_warps)
            self.mncb = triton.cdiv(nx, self.MB)
            self.mgrid = (ny * self.mncb + self.nbp,)
        # seg: the compacted faces in one launch (k_faces_seg)
        self.seg = bool(seg) and self.compact
        if self.seg:
            self.SB, self.SCH, self.snw = int(sblock), int(sch), int(snw)
            self.sncb = triton.cdiv(nx, self.SB)
            self.sgrid = (ny * self.sncb + self.nbp,)
            self.scr = torch.zeros(ny * self.sncb * 2 * self.SB,
                                   dtype=torch.int32, device=dev)
        # pms: k_update2 stores one maximum per program (no atomics), the
        # clock reduces them (float64 path)
        self.pms = bool(pms) and self.fused and not mixed
        self.pm = torch.zeros(self.ugrid[0] if self.pms else self.NPM,
                              dtype=torch.int64, device=dev)
        # quiet: dry-tile skipping (k_update2's QUIET; the tiles are its row
        # segments); dry flags double-buffered by step parity
        self.quiet = bool(quiet) and self.fused and not mixed and self.seg
        if bool(quiet) and not self.quiet:
            raise ValueError('quiet needs the float64 fused path with seg')
        self.qflip = 0
        # fclock: k_clock2's work in k_update2's last program
        self.fclock = bool(fclock) and self.fused and not mixed
        self.arr = torch.zeros(1, dtype=torch.int32, device=dev)
        # wflag: k_limiter2 stores the factors only in blocks that limit
        self.wflag = bool(wflag) and self.fused and not mixed and not (
            bool(wfuse))
        self.wf = torch.zeros(self.grid[0], dtype=torch.int32, device=dev)
        # qflag: k_faces_mark flags the wet faces instead of zeroing the dry
        # ones; k_limiter2/k_update2 load only flagged discharges; finish()
        # zeroes the unflagged ones (the state then equals the default's)
        self.qflag = bool(qflag) and self.compact and not self.seg and (
            not mixed) and not bool(wfuse)
        if self.qflag:
            self.fx = torch.ones(ny, nx + 1, dtype=torch.int8, device=dev)
            self.fy = torch.ones(ny + 1, nx, dtype=torch.int8, device=dev)
        if self.quiet:
            nt = ny * self.uncb
            self.dry = [torch.zeros(nt, dtype=torch.int32, device=dev)
                        for _ in range(2)]
            self.qst = torch.zeros(nt, dtype=torch.int32, device=dev)
            self.stat = torch.zeros(nt, dtype=torch.int32, device=dev)
        # wfuse: k_limiter inside k_update2 (double-buffered depth); off by
        # default for float64 (five float64 divisions per cell cost more
        # than the separate k_limiter), always on for mixed precision
        self.mixed = bool(getattr(eng, 'mixed', False))
        # mskip (mixed): skip the factor divisions in blocks without a
        # limited cell (same numbers; measured no faster on H200 or RTX)
        self.mskip = bool(mskip)
        self.wfuse = self.mixed if wfuse is None else bool(wfuse)
        self.h2 = (torch.empty_like(eng.h) if self.fused and self.wfuse
                   and not self.mixed else None)
        self.flip = 0
        if self.mixed:
            if not self.fused:
                raise ValueError('mixed precision needs the fused kernels')
            # bed steps z0 - z1 of the x and y faces as float32 pairs
            # ([0] high part, [1] low part)
            z = eng.z.to(torch.float64)
            f32 = torch.float32
            dzx = torch.zeros(ny, nx, dtype=torch.float64, device=dev)
            dzx[:, :-1] = z[:, :-1] - z[:, 1:]
            dzy = torch.zeros(ny, nx, dtype=torch.float64, device=dev)
            dzy[:-1] = z[:-1] - z[1:]
            self.dzx = torch.stack([dzx.to(f32), (dzx - dzx.to(f32).to(
                torch.float64)).to(f32)]).contiguous()
            self.dzy = torch.stack([dzy.to(f32), (dzy - dzy.to(f32).to(
                torch.float64)).to(f32)]).contiguous()
            self.b = {e: torch.zeros(ny if e in 'WE' else nx, dtype=f32,
                                     device=dev) for e in 'WENS'}
            self.shi = torch.zeros(ny, nx, dtype=f32, device=dev)
            self.slo = torch.zeros(ny, nx, dtype=f32, device=dev)
            # depth as float32 pairs (A: current at block boundaries, B: the
            # second buffer) and the interval maximum in float32
            self.dA = [torch.zeros(ny, nx, dtype=f32, device=dev)
                       for _ in range(2)]
            self.dB = [torch.zeros(ny, nx, dtype=f32, device=dev)
                       for _ in range(2)]
            self.hm32 = torch.zeros(ny, nx, dtype=f32, device=dev)

    def begin(self):
        """Set the first dt of an interval from the current depth."""
        e = self.e
        self.RP[0] = e.R
        self.RP[1] = e.P
        hm = torch.amax(e.h).to(torch.float64)
        loc = e.cfl * e.dx / torch.sqrt(e.g * torch.clamp(hm, min=e.th))
        d = torch.where(hm > e.th, torch.clamp(loc, max=e.max_dt),
                        torch.full_like(hm, e.max_dt))
        if e.dtype == torch.float32:
            d = d.to(torch.float32).to(torch.float64)
        self.dt.copy_(torch.minimum(d, e.T - e.t_rel).reshape(1))
        e.acc[_CFLMIN] = torch.minimum(e.acc[_CFLMIN], d)
        e.acc[_DTMIN] = torch.minimum(e.acc[_DTMIN], self.dt[0])
        e.acc[_DTMAX] = torch.maximum(e.acc[_DTMAX], self.dt[0])
        self.hm.zero_()
        if self.quiet:
            self._quiet_begin()
        if self.mixed:
            # the interval's sources and the depth as float32 pairs (x = hi
            # + lo to about 48 bits), the maximum in float32
            f32, f64 = torch.float32, torch.float64
            self.shi.copy_(e.src.to(f32))
            self.slo.copy_((e.src - self.shi.to(f64)).to(f32))
            self.dA[0].copy_(e.h.to(f32))
            self.dA[1].copy_((e.h - self.dA[0].to(f64)).to(f32))
            self.hm32.copy_(e.hmax.to(f32))
            self.flip = 0

    def _tiles(self, x):
        e = self.e
        pad = self.uncb * self.UB - e.nx
        return torch.nn.functional.pad(x, (0, pad)).reshape(
            e.ny, self.uncb, self.UB)

    def _quiet_begin(self):
        """Dry flags of the depth, no tile settled (the first step of the
        interval evaluates and stores everything), static tiles: no source
        and no -0 depth (their depth stays exactly what it is while they
        are settled)."""
        e = self.e
        h = e.h
        self.dry[0].copy_((~(self._tiles(h) > e.th).any(-1)).reshape(-1))
        self.qst.zero_()
        st = (self._tiles(e.src) == 0).all(-1) & ~self._tiles(
            torch.signbit(h)).any(-1)
        self.stat.copy_(st.reshape(-1))
        self.qflip = 0

    def finish(self):
        """After the interval: the depth and maximum back to the engine's
        float64 tensors (mixed precision); with qflag the discharges of the
        faces that were dry at the last step back to 0."""
        if self.qflag:
            e = self.e
            e.Qx.masked_fill_(self.fx == 0, 0.0)
            e.Qy.masked_fill_(self.fy == 0, 0.0)
        if self.mixed:
            e = self.e
            e.h.copy_(self.dA[0].to(torch.float64)
                      + self.dA[1].to(torch.float64))
            e.hmax.copy_(torch.maximum(e.hmax,
                                       self.hm32.to(torch.float64)))

    def sync(self):
        """Bring the depth back to the first buffer after an odd number of
        fused steps (the second buffer holds it)."""
        if self.flip:
            if self.mixed:
                self.dA[0].copy_(self.dB[0])
                self.dA[1].copy_(self.dB[1])
            else:
                self.e.h.copy_(self.h2)
            self.flip = 0
        if self.quiet and self.qflip:
            self.dry[0].copy_(self.dry[1])
            self.qflip = 0

    def mutable(self):
        if self.mixed:
            return ([self.dt, self.pm, self.hm32] + self.dA + self.dB
                    + [self.b[e] for e in 'WENS'])
        if self.fused:
            return ([self.w, self.dt, self.pm] + [self.b[e] for e in 'WENS']
                    + ([self.fcnt] if self.compact else [])
                    + (self.dry + [self.qst] if self.quiet else [])
                    + ([self.arr] if self.fclock else [])
                    + ([self.wf] if self.wflag else [])
                    + ([self.fx, self.fy] if self.qflag else []))
        return [self.w, self.hm, self.dt] + [self.b[e] for e in 'WENS']

    def step(self):
        if self.fused:
            return self._step_fused()
        e = self.e
        B, nw, grid = self.BLOCK, self.nw, self.grid
        ny, nx = e.ny, e.nx
        args = (e.h, e.z, self.n, e.qx, e.qy, e.Qx, e.Qy, self.sl['W'],
                self.sl['E'], self.sl['N'], self.sl['S'], self.dt, self.par,
                ny, nx)
        if self.split:
            k_faces[grid](*args, DOX=True, DOY=False, BLOCK=B, num_warps=nw,
            enable_fp_fusion=_fma())
            k_faces[grid](*args, DOX=False, DOY=True, BLOCK=B, num_warps=nw,
            enable_fp_fusion=_fma())
        else:
            k_faces[grid](*args, DOX=True, DOY=True, BLOCK=B, num_warps=nw,
            enable_fp_fusion=_fma())
        k_bound[self.bgrid](e.h, e.z, self.n, e.qx, e.qy, e.Qx, e.Qy,
                            self.sl['W'], self.sl['E'], self.sl['N'],
                            self.sl['S'], self.dt, self.par, ny, nx,
                            BLOCK=256, num_warps=4,
            enable_fp_fusion=_fma())
        if e.drycheck:
            k_limiter[grid](e.h, e.Qx, e.Qy, self.w, self.dt, self.par, ny,
                            nx, BLOCK=B, num_warps=nw,
            enable_fp_fusion=_fma())
        k_update[grid](e.h, e.src, e.Qx, e.Qy, self.w, e.qx, e.qy, e.hmax,
                       self.b['W'], self.b['E'], self.b['N'], self.b['S'],
                       self.hm, e.acc, self.dt, self.par, ny, nx,
                       LIMIT=e.drycheck, BLOCK=B, num_warps=nw,
            enable_fp_fusion=_fma())
        k_clock[(1,)](self.b['W'], self.b['E'], self.b['N'], self.b['S'],
                      self.hm, e.acc, self.dt, e.t_rel, e.T, self.RP, self.par,
                      e.eacc, ny, nx, F64=e.dtype == torch.float64,
                      EDGE=e.track_edges, BLOCK=1024, num_warps=4,
            enable_fp_fusion=_fma())

    def _step_fused(self):
        e = self.e
        ny, nx = e.ny, e.nx
        if self.wfuse:
            hi, ho = (e.h, self.h2) if self.flip == 0 else (self.h2, e.h)
        else:
            hi = ho = e.h
        if self.mixed:
            hi, ho = (self.dA, self.dB) if self.flip == 0 else (self.dB,
                                                                self.dA)
            k_faces_mx[self.fgrid](hi[0], hi[1], e.z, self.dzx, self.dzy,
                                   self.n, e.qx,
                                   e.qy, e.Qx, e.Qy, self.sl['W'],
                                   self.sl['E'], self.sl['N'], self.sl['S'],
                                   self.dt, self.par, ny, nx, self.ncb,
                                   NSCAL=self.nscal, SKIP=self.skip,
                                   BLOCK=self.BLOCK, BBLOCK=256,
                                   num_warps=self.nw,
            enable_fp_fusion=_fma())
            k_update_mx[self.ugrid](hi[0], hi[1], ho[0], ho[1], self.shi,
                                    self.slo, e.Qx, e.Qy,
                                    e.qx, e.qy, self.hm32, self.b['W'],
                                    self.b['E'], self.b['N'], self.b['S'],
                                    self.pm, e.acc, self.dt, self.par,
                                    e.eacc, ny, nx, self.uncb,
                                    LIMIT=e.drycheck, BLOCK=self.UB,
                                    num_warps=self.unw, NPM=self.NPM,
                                    EDGE=e.track_edges, FMA=_fma(),
                                    WSKIP=self.mskip,
                                    enable_fp_fusion=False)
        else:
            self._launch_exact(hi, ho)
        if self.fclock:
            if self.wfuse:
                self.flip ^= 1
            return
        k_clock2[(1,)](self.b['W'], self.b['E'], self.b['N'], self.b['S'],
                       self.pm, self.pm.numel(), e.acc, self.dt, e.t_rel,
                       e.T, self.RP, self.par, e.eacc, ny, nx,
                       F64=e.dtype == torch.float64 and not self.mixed,
                       EDGE=False, BLOCK=1024, NPM=self.NPM,
                       cnt_ptr=self.fcnt if self.compact else self.pm,
                       CNT=self.compact, FMA=_fma(), PMS=self.pms,
                       num_warps=4,
            enable_fp_fusion=_fma())
        if self.wfuse:
            self.flip ^= 1

    def _launch_exact(self, hi, ho):
        e = self.e
        ny, nx = e.ny, e.nx
        qa = qa_l = qa_u = fc = {}
        wl = wu = {}
        if self.wflag:
            wl = dict(wf_ptr=self.wf, WFLAG=True)
            wu = dict(wf_ptr=self.wf, LB=self.BLOCK, WFLAG=True)
        qf = dict(fx_ptr=self.fx, fy_ptr=self.fy, QF=True) if (
            self.qflag) else {}
        if self.fclock:
            fc = dict(t_ptr=e.t_rel, T_ptr=e.T, RP_ptr=self.RP,
                      arr_ptr=self.arr,
                      fcnt_ptr=self.fcnt if self.compact else self.pm,
                      CNT=self.compact, FCLOCK=True)
        if self.quiet:
            din, dout = self.dry[self.qflip], self.dry[1 - self.qflip]
            qa = dict(dry_ptr=din, qst_ptr=self.qst, uncb=self.uncb,
                      UB=self.UB, QUIET=True)
            qa_l = dict(dry_ptr=din, uncb=self.uncb, UB=self.UB, QUIET=True)
            qa_u = dict(dry_in=din, dry_out=dout, qst_ptr=self.qst,
                        stat_ptr=self.stat, QUIET=True)
            self.qflip ^= 1
        if self.seg:
            k_faces_seg[self.sgrid](hi, e.z, self.n, e.qx, e.qy, e.Qx, e.Qy,
                                    self.sl['W'], self.sl['E'], self.sl['N'],
                                    self.sl['S'], self.dt, self.par,
                                    self.scr, ny, nx, self.sncb,
                                    NSCAL=self.nscal, BLOCK=self.SB,
                                    CH=self.SCH, BBLOCK=256,
                                    num_warps=self.snw, **qa,
                                    enable_fp_fusion=_fma())
        elif self.compact:
            k_faces_mark[self.mgrid](hi, e.z, self.n, e.qx, e.qy, e.Qx, e.Qy,
                                     self.sl['W'], self.sl['E'], self.sl['N'],
                                     self.sl['S'], self.dt, self.par,
                                     self.flist, self.fcnt, ny, nx, self.mncb,
                                     NSCAL=self.nscal, BLOCK=self.MB,
                                     BBLOCK=256, num_warps=self.mnw, **qf,
                                     enable_fp_fusion=_fma())
            k_faces_wet[self.cgrid](hi, e.z, self.n, e.qx, e.qy, e.Qx, e.Qy,
                                    self.dt, self.par, self.flist, self.fcnt,
                                    ny, nx, NSCAL=self.nscal,
                                    BLOCK=self.cblock, num_warps=4,
            enable_fp_fusion=_fma())
        else:
            k_faces2[self.fgrid](hi, e.z, self.n, e.qx, e.qy, e.Qx, e.Qy,
                                 self.sl['W'], self.sl['E'], self.sl['N'],
                                 self.sl['S'], self.dt, self.par, ny, nx,
                                 self.ncb, NSCAL=self.nscal, SKIP=self.skip,
                                 BLOCK=self.BLOCK, BBLOCK=256,
                                 num_warps=self.nw,
            enable_fp_fusion=_fma())
        if e.drycheck and not self.wfuse:
            k_limiter2[self.grid](hi, e.Qx, e.Qy, self.w, self.dt, self.par,
                                  ny, nx, BLOCK=self.BLOCK, num_warps=self.nw,
                                  **qa_l, **wl, **qf,
                                  enable_fp_fusion=_fma())
        k_update2[self.ugrid](hi, ho, e.src, e.Qx, e.Qy, e.qx, e.qy, e.hmax,
                              self.b['W'], self.b['E'], self.b['N'],
                              self.b['S'], self.pm, e.acc, self.dt, self.par,
                              self.w, e.eacc, ny, nx, self.uncb,
                              LIMIT=e.drycheck, WLOAD=not self.wfuse,
                              BLOCK=self.UB, num_warps=self.unw,
                              NPM=self.NPM, EDGE=e.track_edges,
                              FMA=_fma(), enable_fp_fusion=False, **qa_u,
                              **fc, **wu, **qf, PMS=self.pms,
                              **(dict(r0=self.rows[0], r1=self.rows[1],
                                      ROWS=True) if self.rows else {}))
