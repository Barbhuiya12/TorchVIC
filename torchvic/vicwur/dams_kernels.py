"""Triton kernels of a set of dams on a day without a month change (CUDA
path of `dams.GlobalDams.run` and `dams.LocalDams.run` there): the daily
part of global_dam_register, dam_operate and the write-back into the
routing ring (dam_register.c:299-340, dam_run.c:33-127, 163-187), and
local_dam_register with dam_operate (dam_register.c:263-297), one launch
per set.

Bitwise the torch path: the same operations in the same order, products
rounded on their own (`_mul`), divisions by a tensor correctly rounded
(`_div`), a torch division by a Python scalar on CUDA (a product with the
double reciprocal, BinaryDivTrueKernel.cu) as that product, the release
correction's power by libdevice's pow (torch's CUDA pow for an exponent
other than 0.5, 2, 3, -0.5, -1, -2; the caller keeps the torch path for
those), and every constant read as float64 from memory. The set's state is
updated in place.
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:                       # CPU-only torch has no Triton
    from ..core.triton_stub import triton, tl, libdevice

from ..model.fastlaunch import launch
from ..model.routing_kernels import _div, _mul

#: torch.pow(x, e) takes another path for these exponents (PowKernel.cu)
POW_SPECIAL = (0.5, 2.0, 3.0, -0.5, -1.0, -2.0)


@triton.jit
def _operate(st0, inflow, op0, cap, alpha, beta, inv_1ma, inv_gs):
    """dams.operate: fill, corrected release, empty, full -> (s, r)."""
    zero = st0 * 0.0
    one = zero + 1.0
    s = st0 + inflow
    hi = _mul(cap, alpha)
    lo = cap - _mul(cap, alpha)
    diff = tl.where(s > hi, s - hi, tl.where(s < lo, s - lo, zero))
    a = 1.0 - _div(s, cap)
    b = _div(s, cap)
    f = tl.where(a > b, a, b)
    f = libdevice.pow(_mul(f - alpha, inv_1ma), beta)
    f = tl.where(diff != 0, f, one)
    corr = op0 + _mul(_mul(diff, f), inv_gs)
    r = tl.where(0 > corr, zero, corr)
    s = s - r
    neg = s < 0
    r = tl.where(neg, r + s, r)
    s = tl.where(neg, zero, s)
    over = s > cap
    r = tl.where(over, r + (s - cap), r)
    s = tl.where(over, cap, s)
    return s, r


@triton.jit(do_not_specialize=['D'])
def local_dam_kernel(D, k_ptr, cid_ptr, ro_ptr, pos_ptr, area_ptr,
                     inflow_ptr, demand_ptr, efr_ptr, st_ptr, rel_ptr,
                     oprel_ptr, cap_ptr, frac_ptr, act_ptr,
                     BLOCK: tl.constexpr):
    """LocalDams.run on a day without a month change: local_dam_register
    (inflow = runoff area frac / 1e6 / 1000, dam_register.c:263-297) and
    dam_operate; the write-back's runoff is not used by the coupling."""
    d = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = d < D
    z = d * 0
    zero = tl.zeros([BLOCK], tl.float64)
    inv_m3 = tl.load(k_ptr + 1 + z)
    alpha = tl.load(k_ptr + 2 + z)
    beta = tl.load(k_ptr + 3 + z)
    inv_1ma = tl.load(k_ptr + 4 + z)
    inv_gs = tl.load(k_ptr + 5 + z)
    inv_mm = tl.load(k_ptr + 9 + z)
    ci = tl.load(cid_ptr + d, mask=m, other=0)
    ro = tl.load(ro_ptr + ci, mask=m, other=0.0)
    ar = tl.load(area_ptr + tl.load(pos_ptr + d, mask=m, other=0), mask=m,
                 other=0.0)
    frac = tl.load(frac_ptr + d, mask=m, other=0.0)
    inflow = _mul(_mul(_mul(_mul(ro, ar), frac), inv_m3), inv_mm)
    st0 = tl.load(st_ptr + d, mask=m, other=0.0)
    op0 = tl.load(oprel_ptr + d * 12, mask=m, other=0.0)
    s, r = _operate(st0, inflow, op0, tl.load(cap_ptr + d, mask=m,
                                               other=1.0),
                    alpha, beta, inv_1ma, inv_gs)
    act = tl.load(act_ptr + d, mask=m, other=0) != 0
    tl.store(st_ptr + d, tl.where(act, s, st0), mask=m)
    tl.store(rel_ptr + d, tl.where(act, r, tl.load(rel_ptr + d, mask=m,
                                                    other=0.0)), mask=m)
    tl.store(inflow_ptr + d, tl.where(act, inflow, zero), mask=m)
    tl.store(demand_ptr + d, tl.where(act, tl.load(demand_ptr + d, mask=m,
                                                    other=0.0), zero),
             mask=m)
    tl.store(efr_ptr + d, tl.where(act, tl.load(efr_ptr + d, mask=m,
                                                 other=0.0), zero), mask=m)


def _consts(dams, dt, n, device):
    key = (float(dt), int(n), dams.alpha, dams.beta, dams.gamma, dams.spd)
    k = getattr(dams, '_kconst', None)
    if k is not None and k[0] == key:
        return k[1]
    t = torch.tensor([float(dt), 1.0 / 1e6, dams.alpha, dams.beta,
                         1.0 / (1 - dams.alpha),
                         1.0 / (dams.gamma * dams.spd), 1e6, 1.0 / float(dt),
                         1.0 / n, 1.0 / 1000.0],
                        dtype=torch.float64, device=device)
    dams._kconst = (key, t)
    return t


def local_dam(dams, runoff, cid, pos, area, block=128):
    """LocalDams.run on a day without a month change, CUDA; runoff [N]
    (cell order), area [N] (sweep order); the set's state in place."""
    k = _consts(dams, 86400.0, 1, runoff.device)
    launch(local_dam_kernel, (triton.cdiv(dams.D, block),),
           (dams.D, k, cid, runoff, pos, area, dams.inflow, dams.demand,
            dams.efr, dams.storage, dams.release, dams.op_release,
            dams.capacity, dams.inflow_frac, dams.active.view(torch.uint8)),
           dict(BLOCK=block))


@triton.jit(do_not_specialize=['D'])
def global_dam_kernel(
        D, NTOT, k_ptr, pos_ptr, cid_ptr, ring_ptr, head_ptr, dis_ptr,
        dem_ptr, efrq_ptr,
        inflow_ptr, demand_ptr, efr_ptr, tin_ptr, tdem_ptr, tefr_ptr,
        nreg_ptr, st_ptr, rel_ptr, oprel_ptr, cap_ptr, frac_ptr, act_ptr,
        R: tl.constexpr, NSUB: tl.constexpr, HAS_DEM: tl.constexpr,
        HAS_EFR: tl.constexpr, BLOCK: tl.constexpr):
    d = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = d < D
    z = d * 0
    zero = tl.zeros([BLOCK], tl.float64)
    one = zero + 1.0
    dt = tl.load(k_ptr + 0 + z)
    inv_m3 = tl.load(k_ptr + 1 + z)
    alpha = tl.load(k_ptr + 2 + z)
    beta = tl.load(k_ptr + 3 + z)
    inv_1ma = tl.load(k_ptr + 4 + z)
    inv_gs = tl.load(k_ptr + 5 + z)
    m3 = tl.load(k_ptr + 6 + z)
    inv_dt = tl.load(k_ptr + 7 + z)
    inv_n = tl.load(k_ptr + 8 + z)
    pos = tl.load(pos_ptr + d, mask=m, other=0)
    dis0 = tl.load(dis_ptr + pos, mask=m, other=0.0)
    frac = tl.load(frac_ptr + d, mask=m, other=0.0)
    # global_dam_register, daily part (dam_register.c:327-339)
    inflow = _mul(_mul(_mul(dis0, dt), frac), inv_m3)
    if HAS_DEM:
        demand = tl.load(dem_ptr + d, mask=m, other=0.0)
    else:
        demand = tl.load(demand_ptr + d, mask=m, other=0.0)
    if HAS_EFR:
        ci = tl.load(cid_ptr + d, mask=m, other=0)
        eq = tl.load(efrq_ptr + ci, mask=m, other=0.0)
        efr = _mul(_mul(_mul(eq, dt), frac), inv_m3)
    else:
        efr = tl.load(efr_ptr + d, mask=m, other=0.0)
    tl.store(tin_ptr + d, tl.load(tin_ptr + d, mask=m, other=0.0) + inflow,
             mask=m)
    tl.store(tdem_ptr + d,
             tl.load(tdem_ptr + d, mask=m, other=0.0) + demand, mask=m)
    tl.store(tefr_ptr + d, tl.load(tefr_ptr + d, mask=m, other=0.0) + efr,
             mask=m)
    tl.store(nreg_ptr + d, tl.load(nreg_ptr + d, mask=m, other=0) + 1,
             mask=m)
    # dam_operate (dam_run.c:33-62) with dam_corr_release (dam_operation.c
    # :230-265)
    st0 = tl.load(st_ptr + d, mask=m, other=0.0)
    op0 = tl.load(oprel_ptr + d * 12, mask=m, other=0.0)
    s, r = _operate(st0, inflow, op0, tl.load(cap_ptr + d, mask=m,
                                               other=1.0),
                    alpha, beta, inv_1ma, inv_gs)
    act = tl.load(act_ptr + d, mask=m, other=0) != 0
    rel = tl.where(act, r, tl.load(rel_ptr + d, mask=m, other=0.0))
    tl.store(st_ptr + d, tl.where(act, s, st0), mask=m)
    tl.store(rel_ptr + d, rel, mask=m)
    tl.store(inflow_ptr + d, tl.where(act, inflow, zero), mask=m)
    tl.store(demand_ptr + d, tl.where(act, demand, zero), mask=m)
    tl.store(efr_ptr + d, tl.where(act, efr, zero), mask=m)
    # write-back (dam_run.c:97-127)
    add = _mul(_mul(_mul(rel, m3), inv_dt), inv_n)
    keep = 1.0 - frac
    h = tl.load(head_ptr)
    dsum = zero
    for i in tl.static_range(NSUB):
        pi = h + i
        pi = tl.where(pi >= R, pi - R, pi)
        q0 = tl.load(ring_ptr + pi * NTOT + pos, mask=m, other=0.0)
        q = _mul(q0, keep) + add
        q = tl.where(q < 0, zero, q)
        dsum = dsum + q
        tl.store(ring_ptr + pi * NTOT + pos, q, mask=m & act)
    tl.store(dis_ptr + pos, dsum, mask=m & act)


def global_dam(r, dams, pos, cid, dt, demand=None, efr_q=None, block=128):
    """GlobalDams.run on a day without a month change, CUDA; r the
    CascadeRouter (ring and discharge updated in place)."""
    D = dams.D
    n = r.nsub
    k = _consts(dams, dt, n, r.device)
    z = dams.inflow
    launch(global_dam_kernel, (triton.cdiv(D, block),),
           (D, r.n, k, pos, cid, r.ring, r.head, r.discharge,
            z if demand is None else demand, z if efr_q is None else efr_q,
            dams.inflow, dams.demand, dams.efr, dams.total_inflow,
            dams.total_demand, dams.total_efr, dams.register_steps,
            dams.storage, dams.release, dams.op_release, dams.capacity,
            dams.inflow_frac, dams.active.view(torch.uint8)),
           dict(R=r.R, NSUB=n, HAS_DEM=demand is not None,
                HAS_EFR=efr_q is not None, BLOCK=block))


def own(dams):
    """Make the set's state tensors contiguous and distinct, so the kernel
    can write them in place."""
    seen = set()
    for kname in ('inflow', 'demand', 'efr', 'total_inflow', 'total_demand',
                  'total_efr', 'register_steps', 'storage', 'release',
                  'op_release', 'capacity', 'inflow_frac', 'active'):
        x = getattr(dams, kname)
        p = x.untyped_storage().data_ptr()
        if not x.is_contiguous() or p in seen or \
                x.untyped_storage().nbytes() != x.numel() * x.element_size():
            x = x.clone(memory_format=torch.contiguous_format)
            setattr(dams, kname, x)
            p = x.untyped_storage().data_ptr()
        seen.add(p)
