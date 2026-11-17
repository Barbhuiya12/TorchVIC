"""Lanes.tsum as one hand-written Triton kernel (ROADMAP W2 P4).

Each lane reads its cell's row of the padded [cell, tile] grid from the
lanes themselves (the cell's lanes are consecutive from `start`, one per
covered tile in tile order; `mask` has bit j set where tile j is covered),
zero where a tile is not covered, and sums the row with eager CUDA's own
addition tree (core.xsum, Reduce.cuh's order) -- the padded grid's sum to
the last bit, in one kernel and without the padded buffer.

The kernel's source is written out for the tile count and the trailing
size (the tree is unrolled), so it is generated, kept in the Triton cache
directory, and imported; it is registered as the custom op
torchvic::tsum_gather, which a fused region (core.fusion) calls as one
opaque operation: dynamo traces one call instead of the tree's two hundred
operations, which is where a region's compile time went.

Only additions are performed, each IEEE in Triton; 0.0 + e is written as
`where(e == 0, +0.0, e)` as in core.xsum.
"""
import hashlib
import importlib.util
import os
import sys

import torch

from ..core import xsum

_KERNELS = {}


def _cache_dir():
    d = os.environ.get('TRITON_CACHE_DIR') or os.path.join(
        os.path.expanduser('~'), '.triton', 'cache')
    d = os.path.join(d, 'torchvic_tsum')
    os.makedirs(d, exist_ok=True)
    return d


def _thread_code(names, out, lines, ind):
    """core.xsum.thread_sum written out: `names` the elements in order."""
    acc = [None] * 4
    for k, e in enumerate(names):
        i = k % 4
        v = '%s_a%d_%d' % (out, i, k)
        if acc[i] is None:
            lines.append('%s%s = tl.where(%s == 0.0, zero, %s)' % (ind, v, e, e))
        else:
            lines.append('%s%s = %s + %s' % (ind, v, acc[i], e))
        acc[i] = v
    res = acc[0]
    for j, a in enumerate(acc[1:]):
        if a is not None:
            v = '%s_c%d' % (out, j)
            lines.append('%s%s = %s + %s' % (ind, v, res, a))
            res = v
    return res


def _tree_code(names, out, lines, ind):
    """core.xsum.tree written out."""
    R = len(names)
    B = min(xsum.last_pow2(R), 32)
    v = [_thread_code(names[t::B], '%s_t%d' % (out, t), lines, ind)
         for t in range(B)]
    off = 1
    while off < B:
        for t in range(0, B, 2 * off):
            nv = '%s_s%d_%d' % (out, off, t)
            lines.append('%s%s = %s + %s' % (ind, nv, v[t], v[t + off]))
            v[t] = nv
        off *= 2
    return v[0]


def _source(R, L):
    ind = '    '
    lines = [
        'import triton',
        'import triton.language as tl',
        '',
        '',
        '@triton.jit',
        'def tsum_kernel(x_ptr, st_ptr, mk_ptr, out_ptr, n, BLOCK: tl.constexpr):',
        ind + 'pid = tl.program_id(0)',
        ind + 'offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)',
        ind + 'ok = offs < n',
        ind + 'st = tl.load(st_ptr + offs, mask=ok, other=0)',
        ind + 'mk = tl.load(mk_ptr + offs, mask=ok, other=0)',
        ind + 'zero = tl.zeros([BLOCK], dtype=tl.float64)',
        ind + 'rank = st * 0',
    ]
    # 32-bit index arithmetic; an uncovered tile is not read at all (its
    # masked load gives +0.0, the padded grid's zero)
    for j in range(R):
        lines.append(ind + 'c%d = (((mk >> %d) & 1) != 0) & ok' % (j, j))
        lines.append(ind + 'i%d = st + rank' % j)
        lines.append(ind + 'rank = rank + c%d.to(tl.int32)' % j)
    for col in range(L):
        names = []
        for j in range(R):
            e = 'e%d_%d' % (col, j)
            lines.append(ind + '%s = tl.load(x_ptr + i%d * %d + %d, '
                         'mask=c%d, other=0.0)' % (e, j, L, col, j))
            names.append(e)
        if L == 1:
            r = _tree_code(names, 'o%d' % col, lines, ind)
        else:
            r = _thread_code(names, 'o%d' % col, lines, ind)
        lines.append(ind + 'tl.store(out_ptr + offs * %d + %d, %s, mask=ok)'
                     % (L, col, r))
    return '\n'.join(lines) + '\n'


def _kernel(R, L):
    key = (R, L)
    k = _KERNELS.get(key)
    if k is not None:
        return k
    src = _source(R, L)
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = os.path.join(_cache_dir(), 'tsum_%d_%d_%s.py' % (R, L, h))
    if not os.path.exists(path):
        tmp = '%s.%d' % (path, os.getpid())
        with open(tmp, 'w') as fh:
            fh.write(src)
        os.replace(tmp, path)
    name = 'torchvic_tsum_%d_%d_%s' % (R, L, h)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    k = _KERNELS[key] = mod.tsum_kernel
    return k


@torch.library.custom_op('torchvic::tsum_gather', mutates_args=())
def tsum_gather(x: torch.Tensor, start: torch.Tensor, mask: torch.Tensor,
                n_tiles: int) -> torch.Tensor:
    """x: [lane, 1, ...] float64 CUDA; `start`, `mask` per output (a cell,
    or a lane: the lane's cell's); returns [len(start), ...]: the padded
    grid's sum over the tiles of each output's cell (Lanes.tsum)."""
    xs = x[:, 0].contiguous()
    n = int(start.shape[0])
    tail = tuple(xs.shape[1:])
    L = 1
    for s in tail:
        L *= int(s)
    out = torch.empty((n,) + tail, dtype=x.dtype, device=x.device)
    if start.dtype != torch.int32:
        start, mask = start.to(torch.int32), mask.to(torch.int32)
    if n:
        BLOCK = 128
        grid = ((n + BLOCK - 1) // BLOCK,)
        _kernel(int(n_tiles), L)[grid](xs, start, mask, out, n, BLOCK=BLOCK)
    return out


@tsum_gather.register_fake
def _(x, start, mask, n_tiles):
    return x.new_empty((start.shape[0],) + tuple(x.shape[2:]))
