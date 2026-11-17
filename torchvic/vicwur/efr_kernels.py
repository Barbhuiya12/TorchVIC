"""Triton kernel of the EFR search (CUDA path of `efr.efr_step`).

`search` evaluates the trial baseflow of every grid fraction and every cell
(efr.trial_baseflow: the sum over tiles then bands, efr_force.c:72-103) and
picks the fraction (efr.select_frac, efr_force.c:106-109) in one launch, the
[C, G] trial table never stored. Bitwise: the operations of the torch path
in its order, products rounded on their own (`_mul`), divisions correctly
rounded (`_div`), the power by libdevice's pow (what torch's CUDA pow is),
and the non-representable constants (the 0.01 step, the loop's fall-through
value) read as float64 from memory.
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:                       # CPU-only torch has no Triton
    from ..core.triton_stub import triton, tl, libdevice

from ..model.routing_kernels import _div, _mul


@triton.jit(do_not_specialize=['C', 'G', 'V'])
def search_kernel(liq_ptr, grid_ptr, k_ptr, res_ptr, mm_ptr, dm_ptr,
                  ds_ptr, ws_ptr, c_ptr, cv_ptr, af_ptr, valid_ptr, req_ptr,
                  out_ptr, C, G, V, B: tl.constexpr, BUG: tl.constexpr,
                  VALID: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < C
    zero = tl.zeros([BLOCK], tl.float64)
    one = zero + 1.0
    res = tl.load(res_ptr + i, mask=m, other=0.0)
    mm = tl.load(mm_ptr + i, mask=m, other=1.0)
    dm = tl.load(dm_ptr + i, mask=m, other=0.0)
    ds = tl.load(ds_ptr + i, mask=m, other=0.0)
    ws = tl.load(ws_ptr + i, mask=m, other=1.0)
    if BUG:
        j0 = i * 0
        ds0 = tl.load(ds_ptr + j0, mask=m, other=0.0)
        ws0 = tl.load(ws_ptr + j0, mask=m, other=1.0)
        c0 = tl.load(c_ptr + j0, mask=m, other=1.0)
    else:
        ds0 = ds
        ws0 = ws
        c0 = tl.load(c_ptr + i, mask=m, other=1.0)
    req = tl.load(req_ptr + i, mask=m, other=0.0)
    mres = mm - res
    a1 = _mul(dm, 1.0 - _div(ds0, ws0))
    omw = 1.0 - ws0
    # the grid is searched until every cell of the block has its first
    # violation (later fractions do not change the answer)
    found = ~m
    first = tl.zeros([BLOCK], tl.int32)
    g = 0
    go = tl.min(found.to(tl.int32), axis=0) == 0
    while go:
        gv = tl.load(grid_ptr + g + i * 0)
        tot = zero
        for j in range(V):
            cvj = tl.load(cv_ptr + i * V + j, mask=m, other=0.0)
            if VALID:
                okj = tl.load(valid_ptr + i * V + j, mask=m, other=0) != 0
            for k in tl.static_range(B):
                lq = tl.load(liq_ptr + (i * V + j) * B + k, mask=m,
                             other=0.0)
                rel = _div(_mul(lq, gv) - res, mres)
                bf = _div(_mul(_mul(rel, dm), ds), ws)
                nl = _mul(a1, libdevice.pow(_div(rel - ws0, omw), c0))
                bf = bf + tl.where(rel > ws0, nl, zero)
                bf = tl.where(bf < 0, zero, bf)
                afk = tl.load(af_ptr + i * B + k, mask=m, other=0.0)
                term = _mul(_mul(bf, cvj), afk)
                if VALID:
                    term = tl.where(okj, term, zero)
                tot = tot + term
        viol = tot < req
        first = tl.where((~found) & viol, g, first)
        found = found | viol
        g += 1
        go = (g < G) & (tl.min(found.to(tl.int32), axis=0) == 0)
    found = found & m
    f = tl.load(grid_ptr + first, mask=m, other=0.0)
    f = f + tl.load(k_ptr + i * 0)
    chosen = tl.where(one < f, one, f)
    last = tl.load(k_ptr + 1 + i * 0)
    tl.store(out_ptr + i, tl.where(found, chosen, last), mask=m)


def search(liq, grid, step_last, res, max_moist, dsmax, ds, ws, c, cv,
           area_fract, req, valid=None, bug=False, block=128):
    """frac [C] of efr.select_frac(trial_baseflow(...)); step_last [2]
    float64 = (FRAC_STEP, the loop's fall-through value)."""
    nc, nv, nb = liq.shape
    f = lambda x: x.contiguous()
    out = torch.empty(nc, dtype=torch.float64, device=liq.device)
    v8 = None if valid is None else f(valid).view(torch.uint8)
    search_kernel[(triton.cdiv(nc, block),)](
        f(liq), f(grid), f(step_last), f(res), f(max_moist), f(dsmax),
        f(ds), f(ws), f(c), f(cv), f(area_fract),
        liq if v8 is None else v8, f(req), out, nc, grid.shape[0], nv,
        B=nb, BUG=bool(bug), VALID=valid is not None, BLOCK=block)
    return out
