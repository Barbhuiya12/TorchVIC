"""Eager aten ops of the graph-path warm-up step OUTSIDE compiled regions
(compiled code runs with the counting mode switched off), by torchvic
line, per step. FUSION on: what the CUDA graph gets as eager kernels."""
import os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
_compile = torch.compile
def compile_(fn, *a, **k):
    c = _compile(fn, *a, **k)
    def call(*x, **y):
        with _disable_current_modes():
            return c(*x, **y)
    return call
torch.compile = compile_
SKIP = {'detach', 'view', '_unsafe_view', 'as_strided', 'expand', 'slice',
        'select', 'unsqueeze', 'squeeze', 'permute', 't', 'transpose',
        'reshape', 'unbind', 'split', 'alias', 'empty', 'empty_like',
        'empty_strided', 'lift_fresh', 'is_nonzero', '_local_scalar_dense',
        'expand_as', 'broadcast_to', 'view_as', 'split_with_sizes'}
class M(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.c = collections.Counter(); self.op = collections.Counter(); self.on = False; self.tag = None
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if not self.on:
            return out
        name = func.overloadpacket.__name__
        if name in SKIP:
            return out
        f = sys._getframe(1); site = '?'
        while f is not None:
            fn = f.f_code.co_filename
            if '/torchvic/' in fn:
                site = '%s:%d %s' % (fn.split('/torchvic/')[-1], f.f_lineno, f.f_code.co_name); break
            f = f.f_back
        if self.tag:
            site = '[%s] %s' % (self.tag, site)
        self.c[site] += 1; self.op[name] += 1
        return out
mode = M()
from torchvic.core import fusion as _F
_op = _F.opaque
def opaque_(fn):
    def inner(*a, **k):
        # the opaque solver's own eager ops count (as "opaque")
        if mode.on:
            with mode:
                mode.tag = 'opaque'
                try:
                    return fn(*a, **k)
                finally:
                    mode.tag = None
        return fn(*a, **k)
    inner.__name__ = getattr(fn, '__name__', 'opaque')
    return _op(inner)
_F.opaque = opaque_
import profile_isc_torchvic as P
from torchvic.model import basin
from torchvic.core import graphsafe

_step_calls = {'n': 0}
_sim = basin.simulate
def sim(*a, **k):
    _step_calls['n'] += 1
    if _step_calls['n'] != 2:
        return _sim(*a, **k)
    _seg_begin = graphsafe.Segmenter.begin
    def begin(self):
        if mode.on:
            mode.on = False
            report()
        return _seg_begin(self)
    graphsafe.Segmenter.begin = begin
    import torchvic.model.basin as B
    # count from the first step of this call (the warm-up step) to its capture
    orig_set = B.graphsafe
    mode.on = True
    with mode:
        return _sim(*a, **k)
def report():
    n = sum(mode.c.values())
    print('== eager body ops in the warm-up step (+ setup of this call): %d' % n)
    for k_, v_ in mode.op.most_common(25):
        print('  op %5d  %s' % (v_, k_))
    for k_, v_ in mode.c.most_common(150):
        print('  %5d  %s' % (v_, k_))
    sys.stdout.flush()
basin.simulate = sim
P.main()
