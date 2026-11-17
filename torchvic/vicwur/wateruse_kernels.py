"""Triton kernels of the VIC-WUR local water use slot (CUDA path).

`wu_local` runs, for the cells of one routing level, everything the torch
path of `vicwur_wateruse_local.sweep_slot` does after the river availability
sum: wateruse.wu_local_step (demand split, cell totals, EFR share of the
river, local-dam availability, the pro-rata division, withdrawals,
consumption, returns, deficit payback, local-dam withdrawal) and the ring
edit of wu_run_local.c:469-540 (CascadeRouter.scale_prop in exact mode, then
recompute, kept only where the cell withdraws or returns), in one launch.
It also writes the cell terms of the connector's ledger rows into per-cell
buffers that the connector sums once per step.

Bitwise. Every operation is the torch operation it replaces, in the same
order: products are rounded on their own (`_mul`, no FMA), divisions by a
tensor are correctly rounded (`_div`), and a torch division by a Python
scalar on CUDA, which is a product with the host reciprocal
(BinaryDivTrueKernel.cu), is that product here. The sums over sectors and
ring bins are the sequential sums of the torch loops. The one torch
reduction of the slot, the river availability over the ring bins
(`wateruse.availability_surf`'s `.sum(1)`), stays a torch reduction:
`surf_terms` lays its terms out as the torch expression does and the caller
sums them with torch.

Not here (torch, per level): the groundwater availability and the pumping
through the step view (GWM off), which touch the step view's soil tensors.
"""
try:
    import triton
    import triton.language as tl
except ImportError:                       # CPU-only torch has no Triton
    from ..core.triton_stub import triton, tl

from ..model.fastlaunch import launch
from ..model.routing_kernels import _div, _mul

S = tl.constexpr(5)
# The constants are exact in float32 (Triton's type of a Python float);
# the reciprocals torch multiplies by on CUDA (a / 1000. is a * (1. /
# 1000.), the double reciprocal) are formed in the kernels as correctly
# rounded float64 divisions, which is what Python's 1.0 / 1000.0 is.


@triton.jit
def _consts(like):
    z = tl.zeros(like.shape, tl.float64)
    mm = z + 1000.0
    m3 = z + 1000000.0
    one = z + 1.0
    return mm, m3, _div(one, mm), _div(one, m3)


@triton.jit(do_not_specialize=['s0', 'C'])
def surf_terms_kernel(ring_ptr, head_ptr, area_ptr, out_ptr, s0, C, NTOT, dt,
                      R: tl.constexpr, BLOCK: tl.constexpr):
    """out[i, c] = ring[logical bin i, s0 + c] * dt / area * MM_PER_M
    (wateruse.availability_surf before its sum), out [R, C]."""
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = c < C
    p = s0 + c
    h = tl.load(head_ptr)
    a = tl.load(area_ptr + p, mask=m, other=1.0)
    for i in tl.static_range(R):
        pi = h + i
        pi = tl.where(pi >= R, pi - R, pi)
        q = tl.load(ring_ptr + pi * NTOT + p, mask=m, other=0.0)
        x = _mul(_div(_mul(q, tl.zeros([BLOCK], tl.float64) + dt), a),
                 tl.zeros([BLOCK], tl.float64) + 1000.0)
        tl.store(out_ptr + i * C + c, x, mask=m)


@triton.jit
def _surf_sum(ring_ptr, head_ptr, area, dtv, MM, p, m, C, NTOT,
              R: tl.constexpr, BW: tl.constexpr, BLOCK: tl.constexpr):
    """availability_surf's sum over the ring bins of q dt / area 1000, in
    the order torch's CUDA reduction adds them (ATen Reduce.cuh) for the
    [C, R] terms with strides (1, C) that the torch path builds: C >= 2,
    the bins are not the fastest dimension and one thread sums a cell with
    vt0 = 4 accumulators (thread_reduce_impl, then the accumulators
    combined in order); C == 1, the bins are contiguous and BW = last
    power of two <= R threads take bins t, t + BW, then a warp-shuffle tree
    (block_x_reduce). Valid while no thread block splits a cell's bins
    over warps (R <= 32; the caller checks)."""
    zero = tl.zeros([BLOCK], tl.float64)
    h = tl.load(head_ptr)
    a0 = zero
    a1 = zero
    a2 = zero
    a3 = zero
    for i in tl.static_range(R):
        pi = h + i
        pi = tl.where(pi >= R, pi - R, pi)
        q = tl.load(ring_ptr + pi * NTOT + p, mask=m, other=0.0)
        x = _mul(_div(_mul(q, dtv), area), MM)
        if i % 4 == 0:
            a0 = a0 + x
        elif i % 4 == 1:
            a1 = a1 + x
        elif i % 4 == 2:
            a2 = a2 + x
        else:
            a3 = a3 + x
    multi = ((a0 + a1) + a2) + a3
    # C == 1
    b = tl.arange(0, BW)
    z2 = tl.zeros([BLOCK, BW], tl.float64)
    p0 = h + b
    p0 = tl.where(p0 >= R, p0 - R, p0)
    q0 = tl.load(ring_ptr + p0[None, :] * NTOT + p[:, None],
                 mask=m[:, None], other=0.0)
    ar = area[:, None] + z2
    lo = _mul(_div(_mul(q0, dtv[:, None] + z2), ar), MM[:, None] + z2)
    bh = b + BW
    okh = bh < R
    p1 = h + bh
    p1 = tl.where(p1 >= R, p1 - R, p1)
    q1 = tl.load(ring_ptr + p1[None, :] * NTOT + p[:, None],
                 mask=m[:, None] & okh[None, :], other=0.0)
    hi = _mul(_div(_mul(q1, dtv[:, None] + z2), ar), MM[:, None] + z2)
    hi = tl.where(okh[None, :], z2 + hi, z2)
    v = (((z2 + lo) + hi) + z2) + z2
    # lane 0 of block_x_reduce's shuffle tree (offsets 1, 2, 4, ...)
    if BW >= 2:
        l, r = tl.split(tl.reshape(v, (BLOCK, BW // 2, 2)))
        v = l + r
    if BW >= 4:
        l, r = tl.split(tl.reshape(v, (BLOCK, BW // 4, 2)))
        v = l + r
    if BW >= 8:
        l, r = tl.split(tl.reshape(v, (BLOCK, BW // 8, 2)))
        v = l + r
    if BW >= 16:
        l, r = tl.split(tl.reshape(v, (BLOCK, BW // 16, 2)))
        v = l + r
    if BW >= 32:
        l, r = tl.split(tl.reshape(v, (BLOCK, BW // 32, 2)))
        v = l + r
    one = tl.reshape(v, (BLOCK,))
    return tl.where(C == 1, one, multi)


@triton.jit(do_not_specialize=['s0', 'C'])
def wu_local_kernel(
        s0, C, NTOT, dt,
        perm_ptr, area_ptr, def_ptr, dis_ptr, str_ptr, ring_ptr, head_ptr,
        asum_ptr, dem_ptr, gfr_ptr, cf_ptr, act_ptr, agw_ptr, efrq_ptr,
        dst_ptr, dact_ptr, dslot_ptr, dhas_ptr,
        dgw_ptr, dsf_ptr, wgw_ptr, wsf_ptr, wdm_ptr, ret_ptr, con_ptr,
        cgw_ptr, acc_ptr,
        R: tl.constexpr, L: tl.constexpr, NSUB: tl.constexpr,
        GWM: tl.constexpr, EFR: tl.constexpr, DAM: tl.constexpr,
        REPSUM: tl.constexpr, BW: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = c < C
    p = s0 + c
    zero = tl.zeros([BLOCK], tl.float64)
    one = zero + 1.0
    MM, M3, INV_MM, INV_M3 = _consts(zero)
    dtv = zero + dt
    area = tl.load(area_ptr + p, mask=m, other=1.0)
    d_in = tl.load(def_ptr + p, mask=m, other=0.0)
    # -- demand (wu_run.c:94-109) and cell totals (wu_run_local.c:45-53)
    dgw0 = zero
    dsf0 = zero
    act0 = tl.zeros([BLOCK], tl.int1)
    dgw1 = zero
    dsf1 = zero
    act1 = act0
    dgw2 = zero
    dsf2 = zero
    act2 = act0
    dgw3 = zero
    dsf3 = zero
    act3 = act0
    dgw4 = zero
    dsf4 = zero
    act4 = act0
    cf0 = zero
    cf1 = zero
    cf2 = zero
    cf3 = zero
    cf4 = zero
    tg = zero
    ts = zero
    for s in tl.static_range(S):
        dem = tl.load(dem_ptr + p * S + s, mask=m, other=0.0)
        act = tl.load(act_ptr + p * S + s, mask=m, other=0) != 0
        cf = tl.load(cf_ptr + p * S + s, mask=m, other=0.0)
        if GWM:
            g = zero
            f = dem
        else:
            gf = tl.load(gfr_ptr + p * S + s, mask=m, other=0.0)
            g = _mul(dem, gf)
            f = _mul(dem, 1.0 - gf)
        g = tl.where(act, g, zero)
        f = tl.where(act, f, zero)
        tg = tg + tl.where(act, g, zero)
        ts = ts + tl.where(act, f, zero)
        if s == 0:
            dgw0, dsf0, act0, cf0 = g, f, act, cf
        elif s == 1:
            dgw1, dsf1, act1, cf1 = g, f, act, cf
        elif s == 2:
            dgw2, dsf2, act2, cf2 = g, f, act, cf
        elif s == 3:
            dgw3, dsf3, act3, cf3 = g, f, act, cf
        else:
            dgw4, dsf4, act4, cf4 = g, f, act, cf
    # -- availabilities (wu_run_local.c:95-170)
    if GWM:
        a_gw = zero
    else:
        a_gw = tl.load(agw_ptr + c, mask=m, other=0.0)
    if REPSUM:
        a_sf = _surf_sum(ring_ptr, head_ptr, area, dtv, MM, p, m, C, NTOT,
                         R, BW, BLOCK)
    else:
        a_sf = tl.load(asum_ptr + c, mask=m, other=0.0)
    if EFR:
        ci = tl.load(perm_ptr + p, mask=m, other=0)
        eq = tl.load(efrq_ptr + ci, mask=m, other=0.0)
        dis0 = tl.load(dis_ptr + p, mask=m, other=1.0)
        rr_ = _div(eq, dis0)
        frac = tl.where(rr_ < 1, rr_, one)
        a_sf = _mul(a_sf, 1.0 - frac)
    a_sf = tl.where(a_sf < 0, zero, a_sf)
    if DAM:
        has = tl.load(dhas_ptr + c, mask=m, other=0) != 0
        slot = tl.load(dslot_ptr + c, mask=m, other=0)
        st0 = tl.load(dst_ptr + slot, mask=m, other=0.0)
        dact = (tl.load(dact_ptr + slot, mask=m, other=0) != 0) & has
        av_dam = _mul(_div(_mul(st0, M3), area), MM)
        av_dam = tl.where(dact & has, av_dam, zero)
        a_dam = zero + av_dam
        a_dam = tl.where(a_dam < 0, zero, a_dam)
    else:
        a_dam = zero
    # -- division (wu_run_local.c:196-248), use (:271-345), cell sums
    gw_t = a_gw
    sf_t = a_sf
    dm_t = a_dam
    gate_g = tg > 0
    gate_s = ts > 0
    cg = zero
    cs = zero
    cd = zero
    cr = zero
    csum = zero
    ret0 = zero
    ret1 = zero
    ret2 = zero
    ret3 = zero
    ret4 = zero
    for s in tl.static_range(S):
        if s == 0:
            d_g, d_s, ok, cf = dgw0, dsf0, act0, cf0
        elif s == 1:
            d_g, d_s, ok, cf = dgw1, dsf1, act1, cf1
        elif s == 2:
            d_g, d_s, ok, cf = dgw2, dsf2, act2, cf2
        elif s == 3:
            d_g, d_s, ok, cf = dgw3, dsf3, act3, cf3
        else:
            d_g, d_s, ok, cf = dgw4, dsf4, act4, cf4
        take = d_g >= gw_t
        v = tl.where(take, gw_t, d_g)
        mg = ok & gate_g
        ag = tl.where(mg, v, zero)
        gw_t = tl.where(mg, tl.where(take, zero, gw_t - d_g), gw_t)
        tot = sf_t + dm_t
        full = d_s >= tot
        ps = _mul(d_s, _div(sf_t, tot))
        pd = _mul(d_s, _div(dm_t, tot))
        ms = ok & gate_s
        asf = tl.where(ms, tl.where(full, sf_t, ps), zero)
        adm = tl.where(ms, tl.where(full, dm_t, pd), zero)
        sf_t = tl.where(ms, tl.where(full, zero, sf_t - ps), sf_t)
        dm_t = tl.where(ms, tl.where(full, zero, dm_t - pd), dm_t)
        # use
        pos = ag > 0
        f = _div(d_g, ag)
        f = tl.where(f < 1, f, one)
        wgw = tl.where(pos, _mul(ag, f), zero)
        omc = 1.0 - cf
        ret = zero + _mul(wgw, omc)
        con = zero + _mul(wgw, cf)
        tot2 = asf + adm
        pos2 = tot2 > 0
        f = _div(_mul(d_s, _div(asf, tot2)), asf)
        f = tl.where(f < 1, f, one)
        wsf = tl.where(pos2, _mul(asf, f), zero)
        f = _div(_mul(d_s, _div(adm, tot2)), adm)
        f = tl.where(f < 1, f, one)
        wdm = tl.where(pos2, _mul(adm, f), zero)
        ret = ret + _mul(wsf, omc)
        ret = ret + _mul(wdm, omc)
        con = con + _mul(wsf, cf)
        con = con + _mul(wdm, cf)
        wgw = tl.where(ok, wgw, zero)
        wsf = tl.where(ok, wsf, zero)
        wdm = tl.where(ok, wdm, zero)
        ret = tl.where(ok, ret, zero)
        con = tl.where(ok, con, zero)
        cg = cg + tl.where(ok, wgw, zero)
        cs = cs + tl.where(ok, wsf, zero)
        cd = cd + tl.where(ok, wdm, zero)
        cr = cr + tl.where(ok, ret, zero)
        csum = csum + con
        o = p * S + s
        tl.store(dgw_ptr + o, d_g, mask=m)
        tl.store(dsf_ptr + o, d_s, mask=m)
        tl.store(wgw_ptr + o, wgw, mask=m)
        tl.store(wsf_ptr + o, wsf, mask=m)
        tl.store(wdm_ptr + o, wdm, mask=m)
        tl.store(ret_ptr + o, ret, mask=m)
        tl.store(con_ptr + o, con, mask=m)
        if s == 0:
            ret0 = ret
        elif s == 1:
            ret1 = ret
        elif s == 2:
            ret2 = ret
        elif s == 3:
            ret3 = ret
        else:
            ret4 = ret
    if not GWM:
        tl.store(cgw_ptr + c, cg, mask=m)
    # -- deficit payback from the returns (wu_run_local.c:439-469)
    d = d_in
    rr = cr
    if not GWM:
        gate = cr > 0
        for s in tl.static_range(S):
            if s == 0:
                rs, ok = ret0, act0
            elif s == 1:
                rs, ok = ret1, act1
            elif s == 2:
                rs, ok = ret2, act2
            elif s == 3:
                rs, ok = ret3, act3
            else:
                rs, ok = ret4, act4
            mm = gate & ok
            x = tl.where(d < rs, d, rs)
            nd = tl.where(d > 0, d - x, d)
            nd = tl.where(nd < 0, zero, nd)
            nr = rr - x
            nr = tl.where(nr < 0, zero, nr)
            d = tl.where(mm, nd, d)
            rr = tl.where(mm, nr, rr)
    tl.store(def_ptr + p, d, mask=m)
    # -- local dam withdrawal (wu_run_local.c:544-561)
    idam = zero
    if DAM:
        valid = dact & has & (cd > 0)
        safe = tl.where(a_dam == 0, one, a_dam)
        dd = _mul(_mul(_mul(_mul(cd, _div(av_dam, safe)), INV_MM), area),
                  INV_M3)
        new = st0 - dd
        new = tl.where(new < 0, zero, new)
        new = tl.where(valid, new, st0)
        tl.store(dst_ptr + slot, new, mask=m & has)
        idam = tl.where(has, _mul(st0 - new, M3), zero)
    # -- the ring edit (wu_run_local.c:469-540): scale_prop(lo=0, exact)
    #    and recompute, where the cell withdraws or returns
    q_w = _div(_mul(_div(cs, MM), area), dtv)
    q_r = _div(_mul(_div(rr, MM), area), dtv)
    wpos = q_w > 0
    rpos = q_r > 0
    gate_r = wpos | rpos
    h = tl.load(head_ptr)
    tot_r = zero
    for i in tl.static_range(R):
        pi = h + i
        pi = tl.where(pi >= R, pi - R, pi)
        tot_r = tot_r + tl.load(ring_ptr + pi * NTOT + p, mask=m, other=0.0)
    posr = tot_r > 0
    dis = zero
    stv = zero
    for i in tl.static_range(R):
        pi = h + i
        pi = tl.where(pi >= R, pi - R, pi)
        q0 = tl.load(ring_ptr + pi * NTOT + p, mask=m, other=0.0)
        q = q0
        q = tl.where(wpos & posr, q - _mul(q_w, _div(q0, tot_r)), q)
        q = tl.where(rpos & posr, q + _mul(q_r, _div(q0, tot_r)), q)
        q = tl.where(rpos & (~posr), q + _div(q_r, zero + (L - 1)), q)
        q = tl.where(gate_r & (q < 0), zero, q)
        tl.store(ring_ptr + pi * NTOT + p, q, mask=m & gate_r)
        if i < NSUB:
            dis = dis + q
        else:
            stv = stv + q
    tl.store(dis_ptr + p, dis, mask=m & gate_r)
    tl.store(str_ptr + p, stv, mask=m & gate_r)
    # -- the cell terms of the ledger row (m3), summed once per step
    a_m3 = _mul(area, INV_MM)
    tl.store(acc_ptr + 0 * NTOT + p, _mul(d_in - d, a_m3), mask=m)
    tl.store(acc_ptr + 1 * NTOT + p,
             _mul(tl.abs(d_in) + tl.abs(d), a_m3), mask=m)
    tl.store(acc_ptr + 2 * NTOT + p, _mul(q_w, dtv), mask=m)
    tl.store(acc_ptr + 3 * NTOT + p, _mul(q_r, dtv), mask=m)
    tl.store(acc_ptr + 4 * NTOT + p, _mul(csum, a_m3), mask=m)
    tl.store(acc_ptr + 5 * NTOT + p, idam, mask=m)
    # ring change: the edited bins' sum less the old one (0 where no edit)
    tl.store(acc_ptr + 6 * NTOT + p,
             tl.where(gate_r, (dis + stv) - tot_r, zero), mask=m)
    tl.store(acc_ptr + 7 * NTOT + p,
             tl.where(wpos & (~posr), one, zero), mask=m)


#: rows of the per-cell ledger buffer
ACC_ROWS = ('pb', 'dmag', 'i_ring', 'q_ret', 'e_cons', 'i_dam', 'ring',
            'bad')


def _f32_exact(x):
    import numpy as np
    x = float(x)
    if float(np.float32(x)) != x:
        raise ValueError('a kernel float argument must be exact in float32')
    return x


def surf_terms(r, s0, C, dt, block=128):
    import torch
    dt = _f32_exact(dt)
    out = torch.empty((r.R, C), dtype=torch.float64, device=r.device)
    launch(surf_terms_kernel, (triton.cdiv(C, block),),
           (r.ring, r.head, r.area, out, s0, C, r.n, float(dt)),
           dict(R=r.R, BLOCK=block))
    return out


#: the river availability sum is the kernel's (torch's order, `_surf_sum`)
#: for rings up to this many bins
REPSUM_MAX = 32


def last_pow2(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def wu_local(r, s0, C, dt, asum, dem, gfr, cf, act, acc, outs, agw=None,
             efr_q=None, dam=None, cgw=None, gwm=False, block=64):
    """One level [s0, s0 + C) of the local water use. outs: (dgw, dsf, wgw,
    wsf, wdm, returned, consumed) [N, 5] written in place; dam: (storage,
    active, slot [C], has [C]) or None; acc [8, N] ledger buffer."""
    dgw, dsf, wgw, wsf, wdm, ret, con = outs
    dt = _f32_exact(dt)
    z = asum if asum is not None else acc
    if dam is not None:
        dst, dact, dslot, dhas = dam
    else:
        dst, dact, dslot, dhas = z, z, z, z
    launch(wu_local_kernel, (triton.cdiv(C, block),),
           (s0, C, r.n, float(dt),
            r.perm, r.area, r.deficit, r.discharge, r.stream, r.ring, r.head,
            z, dem, gfr, cf, act, z if agw is None else agw,
            z if efr_q is None else efr_q,
            dst, dact, dslot, dhas,
            dgw, dsf, wgw, wsf, wdm, ret, con,
            z if cgw is None else cgw, acc),
           dict(R=r.R, L=r.L, NSUB=r.nsub, GWM=bool(gwm),
                EFR=efr_q is not None, DAM=dam is not None,
                REPSUM=asum is None, BW=last_pow2(r.R), BLOCK=block))


@triton.jit(do_not_specialize=['D', 'M'])
def service_demand_kernel(dsf_ptr, act_ptr, svc_ptr, frac_ptr, valid_ptr,
                          area_ptr, out_ptr, D, M, AREA2D: tl.constexpr,
                          BLOCK: tl.constexpr):
    """dams.service_demand: terms ds * area * frac / 1000 / 1e6 (the two
    scalar divisions as torch does them on CUDA), summed over service cells
    then sectors in C order."""
    d = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    md = d < D
    dem = tl.zeros([BLOCK], tl.float64)
    MM, M3, INV_MM, INV_M3 = _consts(dem)
    if not AREA2D:
        a1 = tl.load(area_ptr + d, mask=md, other=0.0)
    for m in range(M):
        o = d * M + m
        cell = tl.load(svc_ptr + o, mask=md, other=0)
        fr = tl.load(frac_ptr + o, mask=md, other=0.0)
        va = tl.load(valid_ptr + o, mask=md, other=0) != 0
        if AREA2D:
            ar = tl.load(area_ptr + o, mask=md, other=0.0)
        else:
            ar = a1
        for s in tl.static_range(S):
            ds = tl.load(dsf_ptr + cell * S + s, mask=md, other=0.0)
            ok = va & (tl.load(act_ptr + cell * S + s, mask=md, other=0) != 0)
            t = _mul(_mul(_mul(_mul(ds, ar), fr), INV_MM), INV_M3)
            dem = dem + tl.where(ok, t, 0.0)
    tl.store(out_ptr + d, dem, mask=md)


def service_demand(demand_surf, act_u8, svc, svc_frac, svc_valid_u8, area,
                   block=64):
    """CUDA dams.service_demand; act_u8, svc_valid_u8 uint8 views of the
    bool masks; area [D] or [D, M]."""
    import torch
    D, M = svc.shape
    out = torch.empty(D, dtype=torch.float64, device=svc.device)
    launch(service_demand_kernel, (triton.cdiv(D, block),),
           (demand_surf, act_u8, svc, svc_frac, svc_valid_u8, area, out, D,
            M), dict(AREA2D=area.dim() == 2, BLOCK=block))
    return out
