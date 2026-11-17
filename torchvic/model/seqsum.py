"""Segmented sums in sequence order on the GPU, as one Triton kernel.

`segment_sum(x, offsets)` is torch.segment_reduce(x, 'sum', lengths=...,
axis=0) to the last bit: each segment's rows are added from 0 one after
the other, in row order, for every column -- segment_reduce's own order
(one thread per segment and column, `acc = acc + x[r]`). It is that kernel
with its memory latency hidden: a program reads U rows ahead (independent
loads) and then adds them in order, so it is bound by the additions rather
than by one load at a time (on the ISC domain the routing sum of 112,825
sources into one outlet took 10.5 ms a call with segment_reduce), and it
needs no host copy of the lengths.

Rows past a segment's end are read as +0.0 and added: exact, since the
running sum starts at +0.0 and an IEEE sum is -0.0 only when both terms
are, so it is never -0.0 and acc + 0.0 is acc (NaN and inf stay).
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                     # pragma: no cover
    triton = None

#: rows a program reads at once (the kernel is written for 32)
_R = 32

if triton is not None:
    @triton.jit
    def _segsum_kernel(x_ptr, off_ptr, out_ptr, F, BF: tl.constexpr):
        s = tl.program_id(0)
        cols = tl.program_id(1) * BF + tl.arange(0, BF)
        cm = cols < F
        a = tl.load(off_ptr + s)
        b = tl.load(off_ptr + s + 1)
        rr = tl.arange(0, 32)
        acc = tl.zeros([BF], dtype=tl.float64)
        # tiles of 32 rows, the next one read while this one is added
        r1 = a + rr
        t_next = tl.load(x_ptr + r1[None, :] * F + cols[:, None],
                         mask=cm[:, None] & (r1[None, :] < b), other=0.0)
        for r0 in range(a, b, 32):
            t = t_next
            r1 = r0 + 32 + rr
            t_next = tl.load(x_ptr + r1[None, :] * F + cols[:, None],
                             mask=cm[:, None] & (r1[None, :] < b),
                             other=0.0)
            # the tile's rows one by one (splits only move values), added
            # in row order
            t_s = tl.reshape(t, (BF, 16, 2))
            t0, t1 = tl.split(t_s)
            t0_s = tl.reshape(t0, (BF, 8, 2))
            t00, t01 = tl.split(t0_s)
            t00_s = tl.reshape(t00, (BF, 4, 2))
            t000, t001 = tl.split(t00_s)
            t000_s = tl.reshape(t000, (BF, 2, 2))
            t0000, t0001 = tl.split(t000_s)
            t0000_s = tl.reshape(t0000, (BF, 1, 2))
            t00000, t00001 = tl.split(t0000_s)
            t0001_s = tl.reshape(t0001, (BF, 1, 2))
            t00010, t00011 = tl.split(t0001_s)
            t001_s = tl.reshape(t001, (BF, 2, 2))
            t0010, t0011 = tl.split(t001_s)
            t0010_s = tl.reshape(t0010, (BF, 1, 2))
            t00100, t00101 = tl.split(t0010_s)
            t0011_s = tl.reshape(t0011, (BF, 1, 2))
            t00110, t00111 = tl.split(t0011_s)
            t01_s = tl.reshape(t01, (BF, 4, 2))
            t010, t011 = tl.split(t01_s)
            t010_s = tl.reshape(t010, (BF, 2, 2))
            t0100, t0101 = tl.split(t010_s)
            t0100_s = tl.reshape(t0100, (BF, 1, 2))
            t01000, t01001 = tl.split(t0100_s)
            t0101_s = tl.reshape(t0101, (BF, 1, 2))
            t01010, t01011 = tl.split(t0101_s)
            t011_s = tl.reshape(t011, (BF, 2, 2))
            t0110, t0111 = tl.split(t011_s)
            t0110_s = tl.reshape(t0110, (BF, 1, 2))
            t01100, t01101 = tl.split(t0110_s)
            t0111_s = tl.reshape(t0111, (BF, 1, 2))
            t01110, t01111 = tl.split(t0111_s)
            t1_s = tl.reshape(t1, (BF, 8, 2))
            t10, t11 = tl.split(t1_s)
            t10_s = tl.reshape(t10, (BF, 4, 2))
            t100, t101 = tl.split(t10_s)
            t100_s = tl.reshape(t100, (BF, 2, 2))
            t1000, t1001 = tl.split(t100_s)
            t1000_s = tl.reshape(t1000, (BF, 1, 2))
            t10000, t10001 = tl.split(t1000_s)
            t1001_s = tl.reshape(t1001, (BF, 1, 2))
            t10010, t10011 = tl.split(t1001_s)
            t101_s = tl.reshape(t101, (BF, 2, 2))
            t1010, t1011 = tl.split(t101_s)
            t1010_s = tl.reshape(t1010, (BF, 1, 2))
            t10100, t10101 = tl.split(t1010_s)
            t1011_s = tl.reshape(t1011, (BF, 1, 2))
            t10110, t10111 = tl.split(t1011_s)
            t11_s = tl.reshape(t11, (BF, 4, 2))
            t110, t111 = tl.split(t11_s)
            t110_s = tl.reshape(t110, (BF, 2, 2))
            t1100, t1101 = tl.split(t110_s)
            t1100_s = tl.reshape(t1100, (BF, 1, 2))
            t11000, t11001 = tl.split(t1100_s)
            t1101_s = tl.reshape(t1101, (BF, 1, 2))
            t11010, t11011 = tl.split(t1101_s)
            t111_s = tl.reshape(t111, (BF, 2, 2))
            t1110, t1111 = tl.split(t111_s)
            t1110_s = tl.reshape(t1110, (BF, 1, 2))
            t11100, t11101 = tl.split(t1110_s)
            t1111_s = tl.reshape(t1111, (BF, 1, 2))
            t11110, t11111 = tl.split(t1111_s)
            acc = acc + tl.reshape(t00000, (BF,))
            acc = acc + tl.reshape(t10000, (BF,))
            acc = acc + tl.reshape(t01000, (BF,))
            acc = acc + tl.reshape(t11000, (BF,))
            acc = acc + tl.reshape(t00100, (BF,))
            acc = acc + tl.reshape(t10100, (BF,))
            acc = acc + tl.reshape(t01100, (BF,))
            acc = acc + tl.reshape(t11100, (BF,))
            acc = acc + tl.reshape(t00010, (BF,))
            acc = acc + tl.reshape(t10010, (BF,))
            acc = acc + tl.reshape(t01010, (BF,))
            acc = acc + tl.reshape(t11010, (BF,))
            acc = acc + tl.reshape(t00110, (BF,))
            acc = acc + tl.reshape(t10110, (BF,))
            acc = acc + tl.reshape(t01110, (BF,))
            acc = acc + tl.reshape(t11110, (BF,))
            acc = acc + tl.reshape(t00001, (BF,))
            acc = acc + tl.reshape(t10001, (BF,))
            acc = acc + tl.reshape(t01001, (BF,))
            acc = acc + tl.reshape(t11001, (BF,))
            acc = acc + tl.reshape(t00101, (BF,))
            acc = acc + tl.reshape(t10101, (BF,))
            acc = acc + tl.reshape(t01101, (BF,))
            acc = acc + tl.reshape(t11101, (BF,))
            acc = acc + tl.reshape(t00011, (BF,))
            acc = acc + tl.reshape(t10011, (BF,))
            acc = acc + tl.reshape(t01011, (BF,))
            acc = acc + tl.reshape(t11011, (BF,))
            acc = acc + tl.reshape(t00111, (BF,))
            acc = acc + tl.reshape(t10111, (BF,))
            acc = acc + tl.reshape(t01111, (BF,))
            acc = acc + tl.reshape(t11111, (BF,))
        tl.store(out_ptr + s * F + cols, acc, mask=cm)


def available(x):
    return (triton is not None and x.is_cuda and x.dtype == torch.float64
            and x.dim() == 2)


def segment_sum(x, offsets):
    """x [rows, F] float64 on a GPU (rows grouped by segment), offsets
    [n_seg + 1] int64 on its device (segment s is rows offsets[s] ..
    offsets[s+1]) -> [n_seg, F], segment_reduce's sums bit for bit."""
    x = x.contiguous()
    n_seg = int(offsets.shape[0]) - 1
    F = int(x.shape[1])
    out = torch.empty((n_seg, F), dtype=x.dtype, device=x.device)
    if n_seg == 0 or F == 0:
        return out
    BF = 8 if F <= 8 else (32 if F <= 32 else 64)
    _segsum_kernel[(n_seg, triton.cdiv(F, BF))](
        x, offsets, out, F, BF=BF, num_warps=1 if BF <= 32 else 2)
    return out
