# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The WOFOST-NPK fused regions (torch.compile with core.fusion's
eager-identical setup) and what they need beyond it.

core.fusion makes inductor's float64 arithmetic eager's. The crop runs in
float32 with the C's promotion, and Triton divides float32 by float32 with
an approximate division (div.full.f32) where eager divides exactly (IEEE
round to nearest). Inside the WOFOST regions a float32 / float32 division
is computed as float64 division rounded to float32: the correctly rounded
float32 quotient (a double holds 2 * 24 + 2 bits or more, so the double
rounding is exact, Figueroa 1995), eager's bit for bit. The lowering is
swapped only while a WOFOST region compiles or runs (`compile_region`).
And inductor's pattern rewrites are off in these regions: one of them
(pointless_convert) removes a float64 -> float32 -> float64 round trip,
the C's float rounding."""
import contextlib

import torch

try:
    from torch._inductor.custom_graph_pass import CustomGraphPass as _Pass
except ImportError:     # pragma: no cover
    _Pass = object


class _Salt(_Pass):
    """A no-op pre-grad pass whose uuid enters inductor's cache keys: the
    WOFOST regions (lowered with the float32 division below) are never
    taken from or for a graph lowered without it."""

    def __call__(self, graph):
        return None

    def uuid(self):
        return 'torchvic-wofost-ieee-f32-div-casts-2'


#: inductor options of the WOFOST regions: one block shape a pointwise
#: kernel (no benchmarking of four at the first call; a block shape
#: changes which thread computes an element, never how); no pattern
#: rewrites (joint_graph.pointless_convert turns float64 -> float32 ->
#: float64 into no conversion, dropping the C's float rounding that
#: every S()/D() pair of the kernel is); the cache salt
OPTIONS = {'triton.autotune_pointwise': False,
           'pattern_matcher': False,
           'post_grad_custom_pre_pass': _Salt()}


@contextlib.contextmanager
def ieee_f32_div():
    import torch._inductor.lowering as L
    aten = torch.ops.aten
    ops = [op for op in (aten.div.Tensor, aten.true_divide.Tensor)
           if op in L.lowerings]
    saved = {op: L.lowerings[op] for op in ops}
    f32, f64 = torch.float32, torch.float64

    def make(prev):
        def div(a, b):
            if (isinstance(a, L.TensorBox) and isinstance(b, L.TensorBox)
                    and a.get_dtype() == f32 and b.get_dtype() == f32):
                q = prev(L.to_dtype(a, f64), L.to_dtype(b, f64))
                return L.to_dtype(q, f32)
            return prev(a, b)
        return div
    for op in ops:
        L.lowerings[op] = make(saved[op])
    try:
        yield
    finally:
        for op in ops:
            L.lowerings[op] = saved[op]


def compile_region(fn):
    """fn compiled as one fused region, called in the f32-division context;
    inside a CUDA graph capture nothing compiles (a call the warm-up did
    not compile for runs eagerly: the same numbers)."""
    from ...core import fusion
    fusion.setup()
    c = torch.compile(fn, dynamic=False, fullgraph=False, options=OPTIONS)

    def call(*a):
        with ieee_f32_div():
            if torch.cuda.is_current_stream_capturing():
                with torch.compiler.set_stance(fusion._CAPTURE_STANCE):
                    return c(*a)
            return c(*a)
    call.__wrapped__ = fn
    return call
