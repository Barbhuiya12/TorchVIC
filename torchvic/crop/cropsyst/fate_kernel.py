"""The portion removal of the harvest biomass fate (n_engine.harvest_fate) as one per-tile
Triton kernel (CUDA): only harvest tiles do work.

harvest_fate computes, for every tile and every day, two newest-first removals over the 401
canopy portions (flip, a cumsum, flip; twice) and the sum of what is left, though only harvest
tiles use it: four [T, 401] scans and their element-wise passes, about 15 ms a day at 338,475
tiles. Here one lane walks its tile's portions once, newest first: both removals advance together
(each one's cumsum as eager CUDA computes it, core.xscan's chunked Sklansky order with the chunk
of the [T, 401] scan), the result goes to a scratch row, and a second walk in list order gives
cumsum(portions)[..., -1]. Only + - and comparisons; a lane without a harvest loads nothing.

TORCHVIC_CROPSYST_KERNELS=0 (om_kernels.KERNELS) keeps the torch form.
"""
import hashlib
import importlib.util
import os
import sys

import torch

from ...core import xscan

_KERNELS = {}


def _source(P, CH):
    R = P + 1                                 # the portions and the untouchable one at the end
    K = max(1, (CH - 1).bit_length())
    L = ["import triton", "import triton.language as tl", "", "",
         "@triton.jit",
         "def kern(pbm_ptr, u_ptr, a1_ptr, a2_ptr, m_ptr, scr_ptr, out_ptr, N, BLOCK: tl.constexpr):",
         "    r = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
         "    ok = r < N",
         "    m = (tl.load(m_ptr + r, mask=ok, other=0) != 0) & ok",
         "    if tl.max(m.to(tl.int32), axis=0) > 0:",
         "        u = tl.load(u_ptr + r, mask=m, other=0.0)",
         "        a1 = tl.load(a1_ptr + r, mask=m, other=0.0)",
         "        a2 = tl.load(a2_ptr + r, mask=m, other=0.0)",
         "        z = tl.zeros([BLOCK], dtype=tl.float64)",
         "        ca = z",
         "        cb = z"]
    L += ["        sa%d = z" % k for k in range(K)] + ["        sb%d = z" % k for k in range(K)]

    def fold(ind, acc, s):
        out = []
        for k in range(K):
            out.append(ind + "if ((li >> %d) & 1) != 0:" % k)
            out.append(ind + "    %s = %s + %s%d" % (acc, acc, s, k))
        return out

    def push(ind, x, s):
        out = [ind + "carry = %s" % x, ind + "go = li >= 0"]
        for k in range(K):
            out += [ind + "if go:",
                    ind + "    if ((li >> %d) & 1) != 0:" % k,
                    ind + "        carry = carry + %s%d" % (s, k),
                    ind + "    else:",
                    ind + "        %s%d = carry" % (s, k),
                    ind + "        go = li < 0"]
        return out

    i = "            "
    # walk 1: newest first (k = 0 is the untouchable portion), both removals
    L.append("        for k in range(0, %d):" % R)
    L.append(i + "li = k & %d" % (CH - 1))
    L.append(i + "x = tl.where(k == 0, u, tl.load(pbm_ptr + r * %d + (%d - k), mask=m & (k > 0), other=0.0))" % (P, P))
    for (x, c, s, a, y) in (("x", "ca", "sa", "a1", "r1"), ("r1", "cb", "sb", "a2", "r2")):
        L.append(i + "xe = tl.where(li == 0, %s + %s, %s)" % (x, c, x))
        L.append(i + "acc = xe")
        L += fold(i, "acc", s)
        L.append(i + "%s = tl.where(li == %d, acc, %s)" % (c, CH - 1, c))
        L += push(i, "xe", s)
        # _remove_newest_first: before = cumsum - x, left = amount - before
        L.append(i + "lft = %s - (acc - %s)" % (a, x))
        L.append(i + "whole = lft > %s" % x)
        L.append(i + "part = (~whole) & (lft > 0.0)")
        L.append(i + "%s = tl.where(whole, 0.0, tl.where(part, %s - lft, %s))" % (y, x, x))
    L.append(i + "tl.store(scr_ptr + r * %d + k, r2, mask=m)" % R)
    # walk 2: list order, cumsum(...)[..., -1]
    L.append("        cc = z")
    L.append("        res = z")
    L.append("        for j in range(0, %d):" % R)
    L.append(i + "li = j & %d" % (CH - 1))
    L.append(i + "x = tl.load(scr_ptr + r * %d + (%d - j), mask=m, other=0.0)" % (R, R - 1))
    L.append(i + "xe = tl.where(li == 0, x + cc, x)")
    L.append(i + "if (li == %d) | (j == %d):" % (CH - 1, R - 1))
    L.append(i + "    acc = xe")
    L += fold(i + "    ", "acc", "sa")
    L.append(i + "    cc = acc")
    L.append(i + "    res = acc")
    L += push(i, "xe", "sa")
    L.append("        tl.store(out_ptr + r, res, mask=m)")
    return "\n".join(L) + "\n"


def _kernel(P, CH):
    k = _KERNELS.get((P, CH))
    if k is not None:
        return k
    src = _source(P, CH)
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    d = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    d = os.path.join(d, "torchvic_fate")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "fate_%s.py" % h)
    if not os.path.exists(path):
        tmp = "%s.%d" % (path, os.getpid())
        with open(tmp, "w") as fh:
            fh.write(src)
        os.replace(tmp, path)
    name = "torchvic_fate_%s" % h
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    k = _KERNELS[(P, CH)] = mod.kern
    return k


@torch.library.custom_op("torchvic::cs_harvest_left", mutates_args=())
def _left_op(p_bm: torch.Tensor, u: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
    T, P = p_bm.shape
    out = torch.zeros((T,), dtype=p_bm.dtype, device=p_bm.device)
    if T:
        scr = torch.empty((T, P + 1), dtype=p_bm.dtype, device=p_bm.device)
        CH = xscan._chunk(T, P + 1)
        BLOCK = 128
        _kernel(P, CH)[((T + BLOCK - 1) // BLOCK,)](p_bm.contiguous(), u.contiguous(), a1.contiguous(),
                                                    a2.contiguous(), mask.contiguous(), scr, out, T,
                                                    BLOCK=BLOCK, num_warps=4, enable_fp_fusion=False)
    return out


@_left_op.register_fake
def _(p_bm, u, a1, a2, mask):
    return p_bm.new_empty(p_bm.shape[:1])


def harvest_left(p_bm, u, a1, a2, mask):
    """cn._seqsum of _remove_newest_first(_remove_newest_first(cat(p_bm, u), a1), a2), for the
    tiles of mask (0 elsewhere)."""
    return _left_op(p_bm, u, a1, a2, mask)
