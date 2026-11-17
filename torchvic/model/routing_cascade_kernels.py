"""Triton kernels of the VIC-WUR routing cascade (`routing_cascade.py`).

A program works on a tile [BC cells, RB bins] (RB = R rounded up to a power
of two): the ring bins of a cell are loaded, convolved and stored all at
once, so a level costs a few memory latencies instead of one per bin. The
arithmetic is the C arithmetic of rout_basin_run.c:66-123 and convolve
(general/src/convolute.c) in the C order:

* the convolution adds k = 0..n-1 in ascending order from 0, each product
  rounded on its own (`_mul`, no FMA);
* dt_inflow sums the upstream cells in rout_init.c:285-308 order;
* discharge and stream add the bins in bin order. No cell's routing reads
  them (only upstream ring bins), so the plain sweep leaves them to one
  all-cell `sums_kernel` afterwards; the slot sweep, whose slots read them,
  sums in the level kernel after a block barrier.

So the result is bitwise the C result. The ring is [R, C] (bin-major, cells
in sweep order) behind one rotating head; the cells of a level are
contiguous, so a tile load is coalesced along cells.

`level_kernel` runs launch groups g0..g1-1 (a level or sub-level each). With
one program it walks all of them with a block barrier between groups, so the
narrow tail of a network is one launch; a wide group gets many programs.
Device scalars only: no host sync, CUDA-graph capturable.
"""
try:
    import triton
    import triton.language as tl
except ImportError:                       # CPU-only torch has no Triton
    from ..core.triton_stub import triton, tl

from .routing_kernels import _div, _mul


ROW_MAX = 32                 # rings up to this many bins: ROW tiles


def tile(R):
    """(RB, BC, ROW, warps) of a level tile. Short rings (RB <= ROW_MAX,
    e.g. daily routing n = 1) use ROW tiles, one cell per thread, with the
    next group's indices, ring tile and UH tile prefetched before the group
    barrier; longer rings spread a cell's bins over threads (2048-element
    tiles)."""
    rb = triton.next_power_of_2(R)
    if rb <= ROW_MAX:
        return rb, 128, True, 4
    return rb, max(16, 2048 // rb), False, 4


@triton.jit
def _bins(h, R: tl.constexpr, RB: tl.constexpr):
    b = tl.arange(0, RB)
    p = h + b
    return b, tl.where(p >= R, p - R, p)


@triton.jit
def begin_kernel(ring_ptr, head_ptr, run_ptr, uhr_ptr, C,
                 R: tl.constexpr, L: tl.constexpr, NSUB: tl.constexpr,
                 RB: tl.constexpr, BC: tl.constexpr):
    """cshift with zero fill (rout_basin_run.c:66-71) as a head move of NSUB
    bins plus zeroing the bins that wrap, then the runoff convolution
    (:73-84). The caller advances the head afterwards."""
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    cm = c < C
    h = tl.load(head_ptr) + NSUB
    h = tl.where(h >= R, h - R, h)
    b, pb = _bins(h, R, RB)
    m2 = cm[:, None] & (b < R)[None, :]
    q = tl.load(ring_ptr + pb[None, :] * C + c[:, None], mask=m2, other=0.0)
    q = tl.where((b >= R - NSUB)[None, :], 0.0, q)
    nf = tl.full([BC], NSUB, tl.float64)
    sig = _div(tl.load(run_ptr + c, mask=cm, other=0.0), nf)
    sig2 = tl.broadcast_to(sig[:, None], (BC, RB))
    acc = tl.zeros([BC, RB], tl.float64)
    for k in tl.static_range(NSUB):
        jb = b - k
        okb = (jb >= 0) & (jb < L)
        u = tl.load(uhr_ptr + jb[None, :] * C + c[:, None],
                    mask=cm[:, None] & okb[None, :], other=0.0)
        acc = tl.where(okb[None, :], acc + _mul(sig2, u), acc)
    tl.store(ring_ptr + pb[None, :] * C + c[:, None], q + acc, mask=m2)


@triton.jit
def _index(upt_ptr, nup_ptr, c, cm, C):
    """Upstream count and [BC, 8] upstream sweep positions; the table is
    [8, C], so the tile is contiguous along cells like the ring tiles and
    in a ROW tile a thread holds its own cell's row."""
    m8 = tl.arange(0, 8)
    nup = tl.load(nup_ptr + c, mask=cm, other=0)
    upt = tl.load(upt_ptr + m8[None, :] * C + c[:, None],
                  mask=cm[:, None] & (m8[None, :] < nup[:, None]), other=0)
    return nup, upt


@triton.jit
def _own(ring_ptr, uhi_ptr, h, c, cm, C, R: tl.constexpr, L: tl.constexpr,
         RB: tl.constexpr):
    """A ROW tile's own ring bins and its inflow UH (k = 0 term)."""
    b = tl.arange(0, RB)
    pb = h + b
    pb = tl.where(pb >= R, pb - R, pb)
    q = tl.load(ring_ptr + pb[None, :] * C + c[:, None],
                mask=cm[:, None] & (b < R)[None, :], other=0.0)
    w = tl.load(uhi_ptr + b[None, :] * C + c[:, None],
                mask=cm[:, None] & (b < L)[None, :], other=0.0)
    return q, w


@triton.jit
def _force(fq_ptr, c, cm, NSUB: tl.constexpr, FORCE: tl.constexpr,
           BC: tl.constexpr):
    """dt_inflow starts at Fq / n (rout_basin_run.c:88-96)."""
    if FORCE:
        nf = tl.full([BC], NSUB, tl.float64)
        f = _div(tl.load(fq_ptr + c, mask=cm, other=0.0), nf)
    else:
        f = tl.zeros([BC], tl.float64)
    return f


@triton.jit
def _route_row(ring_ptr, h, nup, upt, q, w, uhi_ptr, fq_ptr, inflow_ptr, c,
               cm, C, R: tl.constexpr, L: tl.constexpr, NSUB: tl.constexpr,
               FORCE: tl.constexpr, RB: tl.constexpr, BC: tl.constexpr):
    """ROW tile; q and w (k = 0) come in, loaded ahead."""
    b = tl.arange(0, RB)
    pb = h + b
    pb = tl.where(pb >= R, pb - R, pb)
    m2 = cm[:, None] & (b < R)[None, :]
    m8 = tl.arange(0, 8)
    um2 = cm[:, None] & (m8[None, :] < nup[:, None])
    f = _force(fq_ptr, c, cm, NSUB, FORCE, BC)
    acc = tl.zeros([BC, RB], tl.float64)
    infl = tl.zeros([BC], tl.float64)
    for k in tl.static_range(NSUB):
        # dt_inflow[k] (rout_basin_run.c:93-102): upstream cells in
        # rout_init.c:285-308 order, each taken out of the row exactly
        pk = h + k
        pk = tl.where(pk >= R, pk - R, pk)
        v2 = tl.load(ring_ptr + pk * C + upt, mask=um2, other=0.0,
                     cache_modifier='.cg')
        d = f
        for m in tl.static_range(8):
            v = tl.sum(tl.where((m8 == m)[None, :], v2, 0.0), axis=1)
            d = tl.where(cm & (m < nup), d + v, d)
        infl = infl + d
        # its term of the inflow convolution (:104-110)
        jb = b - k
        okb = (jb >= 0) & (jb < L)
        if k == 0:
            wk = w
        else:
            wk = tl.load(uhi_ptr + jb[None, :] * C + c[:, None],
                         mask=cm[:, None] & okb[None, :], other=0.0)
        acc = tl.where(okb[None, :],
                       acc + _mul(tl.broadcast_to(d[:, None], (BC, RB)), wk),
                       acc)
    tl.store(ring_ptr + pb[None, :] * C + c[:, None], q + acc, mask=m2)
    tl.store(inflow_ptr + c, infl, mask=cm)


@triton.jit
def _route_spread(ring_ptr, h, up_ptr, nup_ptr, uhi_ptr, fq_ptr, inflow_ptr,
                  c, cm, C, R: tl.constexpr, L: tl.constexpr,
                  NSUB: tl.constexpr, FORCE: tl.constexpr, RB: tl.constexpr,
                  BC: tl.constexpr):
    """Long-ring tile: a cell's bins spread over threads."""
    b = tl.arange(0, RB)
    pb = h + b
    pb = tl.where(pb >= R, pb - R, pb)
    m2 = cm[:, None] & (b < R)[None, :]
    q = tl.load(ring_ptr + pb[None, :] * C + c[:, None], mask=m2, other=0.0)
    nup = tl.load(nup_ptr + c, mask=cm, other=0)
    f = _force(fq_ptr, c, cm, NSUB, FORCE, BC)
    acc = tl.zeros([BC, RB], tl.float64)
    infl = tl.zeros([BC], tl.float64)
    for k in tl.static_range(NSUB):
        pk = h + k
        pk = tl.where(pk >= R, pk - R, pk)
        d = f
        for m in tl.static_range(8):          # rout_init.c:285-308 order
            um = cm & (m < nup)
            u = tl.load(up_ptr + m * C + c, mask=um, other=0)
            v = tl.load(ring_ptr + pk * C + u, mask=um, other=0.0,
                        cache_modifier='.cg')
            d = tl.where(um, d + v, d)
        infl = infl + d
        jb = b - k
        okb = (jb >= 0) & (jb < L)
        wk = tl.load(uhi_ptr + jb[None, :] * C + c[:, None],
                     mask=cm[:, None] & okb[None, :], other=0.0)
        acc = tl.where(okb[None, :],
                       acc + _mul(tl.broadcast_to(d[:, None], (BC, RB)), wk),
                       acc)
    tl.store(ring_ptr + pb[None, :] * C + c[:, None], q + acc, mask=m2)
    tl.store(inflow_ptr + c, infl, mask=cm)


@triton.jit
def _sums(ring_ptr, h, dis_ptr, stream_ptr, c, cm, C, R: tl.constexpr,
          NSUB: tl.constexpr):
    """discharge = bins < n, stream = bins >= n, added in bin order
    (rout_basin_run.c:112-123); all loads independent, one latency."""
    dis = tl.zeros(c.shape, tl.float64)
    st = tl.zeros(c.shape, tl.float64)
    for i in tl.static_range(R):
        pi = h + i
        pi = tl.where(pi >= R, pi - R, pi)
        col = tl.load(ring_ptr + pi * C + c, mask=cm, other=0.0,
                      cache_modifier='.cg')
        if i < NSUB:
            dis = dis + col
        else:
            st = st + col
    tl.store(dis_ptr + c, dis, mask=cm)
    tl.store(stream_ptr + c, st, mask=cm)


@triton.jit
def sums_kernel(ring_ptr, head_ptr, dis_ptr, stream_ptr, C,
                R: tl.constexpr, NSUB: tl.constexpr, BLOCK: tl.constexpr):
    """_sums for all cells after the sweep: no cell's routing reads its
    upstream discharge or stream, only their ring bins."""
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _sums(ring_ptr, tl.load(head_ptr), dis_ptr, stream_ptr, c, c < C, C, R,
          NSUB)


@triton.jit
def level_kernel(ring_ptr, head_ptr, up_ptr, nup_ptr, uhi_ptr, fq_ptr,
                 inflow_ptr, dis_ptr, stream_ptr, gptr, g0, g1, C,
                 R: tl.constexpr, L: tl.constexpr, NSUB: tl.constexpr,
                 FORCE: tl.constexpr, RB: tl.constexpr, BC: tl.constexpr,
                 ROW: tl.constexpr, SUMS: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    h = tl.load(head_ptr)
    s = tl.load(gptr + g0)
    e = tl.load(gptr + g0 + 1)
    if ROW:
        # first tile of each group: everything it needs except the
        # upstream bins is loaded before the previous group's barrier
        cc = s + pid * BC + tl.arange(0, BC)
        nup, upt = _index(up_ptr, nup_ptr, cc, cc < e, C)
        q, w = _own(ring_ptr, uhi_ptr, h, cc, cc < e, C, R, L, RB)
        for g in range(g0, g1):
            e2 = tl.load(gptr + tl.minimum(g + 2, g1))
            _route_row(ring_ptr, h, nup, upt, q, w, uhi_ptr, fq_ptr,
                       inflow_ptr, cc, cc < e, C, R, L, NSUB, FORCE, RB, BC)
            if SUMS:
                tl.debug_barrier()
                _sums(ring_ptr, h, dis_ptr, stream_ptr, cc, cc < e, C, R,
                      NSUB)
            for c0 in range(s + (pid + npg) * BC, e, npg * BC):
                c1 = c0 + tl.arange(0, BC)
                nu1, up1 = _index(up_ptr, nup_ptr, c1, c1 < e, C)
                q1, w1 = _own(ring_ptr, uhi_ptr, h, c1, c1 < e, C, R, L, RB)
                _route_row(ring_ptr, h, nu1, up1, q1, w1, uhi_ptr, fq_ptr,
                           inflow_ptr, c1, c1 < e, C, R, L, NSUB, FORCE, RB,
                           BC)
                if SUMS:
                    tl.debug_barrier()
                    _sums(ring_ptr, h, dis_ptr, stream_ptr, c1, c1 < e, C,
                          R, NSUB)
            cc = e + pid * BC + tl.arange(0, BC)
            nup, upt = _index(up_ptr, nup_ptr, cc, cc < e2, C)
            q, w = _own(ring_ptr, uhi_ptr, h, cc, cc < e2, C, R, L, RB)
            tl.debug_barrier()
            s = e
            e = e2
    else:
        for g in range(g0, g1):
            e2 = tl.load(gptr + tl.minimum(g + 2, g1))
            for c0 in range(s + pid * BC, e, npg * BC):
                c1 = c0 + tl.arange(0, BC)
                _route_spread(ring_ptr, h, up_ptr, nup_ptr, uhi_ptr, fq_ptr,
                              inflow_ptr, c1, c1 < e, C, R, L, NSUB, FORCE,
                              RB, BC)
                if SUMS:
                    tl.debug_barrier()
                    _sums(ring_ptr, h, dis_ptr, stream_ptr, c1, c1 < e, C,
                          R, NSUB)
            tl.debug_barrier()
            s = e
            e = e2


@triton.jit
def rout_run_kernel(rs_ptr, rb_ptr, perm_ptr, area_ptr, def_ptr, run_ptr,
                    dt1000, N, GWM: tl.constexpr, NRR: tl.constexpr,
                    BLOCK: tl.constexpr):
    """rout_run (rout_run.c:48-81) for all cells, sweep order out, cell
    order in: non-renewable payback from baseflow (and runoff), then
    (Rs + Rb) * area / (dt * MM_PER_M)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    ci = tl.load(perm_ptr + i, mask=m, other=0)
    rs = tl.load(rs_ptr + ci, mask=m, other=0.0)
    if GWM:
        rb = tl.zeros([BLOCK], tl.float64)
    else:
        rb = tl.load(rb_ptr + ci, mask=m, other=0.0)
        d = tl.load(def_ptr + i, mask=m, other=0.0)
        on = d > 0
        gt = rb > d
        d1 = tl.where(on, tl.where(gt, 0.0, d - rb), d)
        rb = tl.where(on, tl.where(gt, rb - d, 0.0), rb)
        if NRR:
            gt2 = rs > d1
            d2 = tl.where(on, tl.where(gt2, 0.0, d1 - rs), d1)
            rs = tl.where(on, tl.where(gt2, rs - d1, 0.0), rs)
            d1 = d2
        tl.store(def_ptr + i, d1, mask=m)
    x = _mul(rs + rb, tl.load(area_ptr + i, mask=m, other=1.0))
    tl.store(run_ptr + i, _div(x, tl.full([BLOCK], dt1000, tl.float64)),
             mask=m)


def rout_run(r, rs, rb, block=512):
    rout_run_kernel[(triton.cdiv(r.n, block),)](
        rs, rs if rb is None else rb, r.perm, r.area, r.deficit, r.runoff,
        r.dt * 1000, r.n, GWM=r.gwm, NRR=r.nonrenew_runoff, BLOCK=block)


def begin(ring, head, runoff, uhr, nsub):
    R, C = ring.shape
    rb = triton.next_power_of_2(R)
    bc = max(16, min(128, 2048 // rb))
    begin_kernel[(triton.cdiv(C, bc),)](
        ring, head, runoff, uhr, C, R=R, L=uhr.shape[0], NSUB=nsub, RB=rb,
        BC=bc)


def level(r, g0, g1, nprog, sums=True):
    from .fastlaunch import launch
    R, C = r.ring.shape
    rb, bc, row, warps = tile(R)
    launch(level_kernel, (nprog,),
           (r.ring, r.head, r.up_t, r.nup, r.uh_inflow, r.force, r.inflow,
            r.discharge, r.stream, r.gptr, g0, g1, C),
           dict(R=R, L=r.uh_inflow.shape[0], NSUB=r.nsub, FORCE=r.use_force,
                RB=rb, BC=bc, ROW=row, SUMS=sums, num_warps=warps))


def sums(r, block=256):
    R, C = r.ring.shape
    sums_kernel[(triton.cdiv(C, block),)](
        r.ring, r.head, r.discharge, r.stream, C, R=R, NSUB=r.nsub,
        BLOCK=block)
