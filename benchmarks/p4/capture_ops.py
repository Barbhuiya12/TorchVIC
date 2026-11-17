"""Eager (not compiled) aten calls recorded into the CUDA graph of one
step, by torchvic line, during the first capture; and the calls made
around each replay (host side) in the next steps."""
import os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import profile_isc_torchvic as P
from torchvic.core import graphsafe
SKIP = {'detach', 'view', '_unsafe_view', 'as_strided', 'expand', 'slice',
        'select', 'unsqueeze', 'squeeze', 'permute', 't', 'transpose',
        'reshape', 'unbind', 'split', 'alias', 'empty', 'empty_like',
        'empty_strided', 'lift_fresh', 'is_nonzero', '_local_scalar_dense'}
class M(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.c = collections.Counter(); self.n = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        if name in SKIP:
            return out
        f = sys._getframe(1); site = '?'
        while f is not None:
            fn = f.f_code.co_filename
            if '/torchvic/' in fn:
                site = '%s:%d %s' % (fn.split('/torchvic/')[-1], f.f_lineno, f.f_code.co_name); break
            f = f.f_back
        self.c[(name, site)] += 1; self.n += 1
        return out
mode = M()
state = {'cap': 0}
_begin = graphsafe.Segmenter.begin
_finish = graphsafe.Segmenter.finish
def begin(self):
    _begin(self)
    if state['cap'] == 0 and not getattr(self, '_m', False):
        self._m = True
        mode.__enter__()
def finish(self):
    _finish(self)
    if getattr(self, '_m', False) and state['cap'] == 0:
        mode.__exit__(None, None, None)
        state['cap'] = 1
        print('== eager aten calls in the first capture: %d' % mode.n)
        for (n, s), v in mode.c.most_common(70):
            print('  %5d  %-22s %s' % (v, n, s))
graphsafe.Segmenter.begin = begin
graphsafe.Segmenter.finish = finish
P.main()
