"""Eager CUDA's float64 sum over a short dimension, as an explicit tree of
additions that a fused region can compute bit for bit (ROADMAP W2 P4).

A sum over dim 1 of a contiguous [N, R] or [N, R, L] tensor is, in eager
PyTorch on a CUDA device, ATen's reduce_kernel (Reduce.cuh) with a launch
configuration fixed by R and L alone for the sizes here (R, L <= 128):

* R contiguous (no trailing dims, or L == 1): `B = min(last_pow2(R), 32)`
  threads share an output. Thread t reads elements t, t+B, t+2B, ... into
  four accumulators in turn (the k-th it reads into accumulator k mod 4,
  each starting from 0.0), adds the four in order, and the B threads'
  values are combined by the warp shuffle tree (offsets 1, 2, 4, ...).
* L > 1 (R strided): one thread per output reads all R elements in order
  into the four accumulators as above and adds them in order.

Inductor's own sum is a different tree, which is why core.fusion leaves
reductions to eager's kernel; `sum1` writes eager's tree out as additions,
which a region fuses with what produces and consumes the sum. 0.0 + e is
e except for e = -0.0, where it is +0.0: the first addition into an
accumulator is written as that `where`, and every later one is a plain
addition (a sum that began from +0.0 cannot be -0.0 again, so adding a
known zero is exact and is left out).

Checked against eager for R = 1..40 and 64, L = 1..25 (R = 100 and 128
follow another order), N = 1..112,680
with values over 60 binades and random zeros (benchmarks/p4/reduce_order.py,
tests/test_xsum.py), on the H200 and the RTX 6000 Ada.
"""
import torch

#: the largest R and L the tree is written for (beyond, eager's kernel
#: vectorises loads or stores and the order changes)
MAX_R = 64
MAX_L = 25


def last_pow2(n):
    n = int(n)
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return max(1, n - (n >> 1))


def _first(e):
    # 0.0 + e
    return torch.where(e == 0.0, torch.zeros_like(e), e)


def thread_sum(elems):
    """One thread's value: elements in the order it reads them, into four
    accumulators in turn, then accumulators 0+1, +2, +3."""
    acc = [None] * 4
    for k, e in enumerate(elems):
        i = k % 4
        acc[i] = _first(e) if acc[i] is None else acc[i] + e
    out = acc[0]
    for a in acc[1:]:
        if a is not None:
            out = out + a
    return out


def tree(elems):
    """Eager's sum of `elems` (a list of R same-shaped tensors, the
    reduced dimension's entries in order) for the contiguous case."""
    R = len(elems)
    B = min(last_pow2(R), 32)
    v = [thread_sum(elems[t::B]) for t in range(B)]
    off = 1
    while off < B:
        for t in range(0, B, 2 * off):
            v[t] = v[t] + v[t + off]
        off *= 2
    return v[0]


def supported(x, dim=1):
    if not (torch.is_tensor(x) and x.is_cuda and x.dtype == torch.float64
            and x.dim() >= 2 and dim == 1 and x.is_contiguous()):
        return False
    R = int(x.shape[1])
    L = 1
    for s in x.shape[2:]:
        L *= int(s)
    return 1 <= R <= MAX_R and L <= MAX_L and x.numel() > 0


def sum1(x):
    """x.sum(1) for a contiguous float64 CUDA tensor of 2 or more dims, bit
    for bit; x.sum(1) itself where the tree is not known."""
    if not supported(x):
        return x.sum(1)
    R = int(x.shape[1])
    L = 1
    for s in x.shape[2:]:
        L *= int(s)
    elems = [x[:, j] for j in range(R)]
    if L == 1:
        return tree(elems)
    return thread_sum(elems)


def sum_decomp(x, dim=None, keepdim=False, dtype=None):
    """aten.sum.dim_IntList as eager's tree where `sum1`'s cases cover it
    (one reduced dimension of a contiguous float64 CUDA tensor; the dims
    before it are outputs like N, those after it L), else NotImplemented:
    the decomposition core.fusion registers for its regions."""
    if dtype is not None or not torch.is_tensor(x) or x.dim() == 0:
        return NotImplemented
    if dim is None or len(dim) != 1:
        return NotImplemented
    d = int(dim[0]) % x.dim()
    if not (x.is_cuda and x.dtype == torch.float64 and x.is_contiguous()):
        return NotImplemented
    if x.numel() == 0:
        return NotImplemented
    shape = tuple(int(s) for s in x.shape)
    R = shape[d]
    N = 1
    for s in shape[:d]:
        N *= s
    L = 1
    for s in shape[d + 1:]:
        L *= s
    if not (1 <= R <= MAX_R and L <= MAX_L):
        return NotImplemented
    x3 = x.reshape(N, R, L)
    if L == 1:
        out = tree([x3[:, j, 0] for j in range(R)])
    else:
        out = thread_sum([x3[:, j, :] for j in range(R)])
    oshape = shape[:d] + ((1,) if keepdim else ()) + shape[d + 1:]
    return out.reshape(oshape)
