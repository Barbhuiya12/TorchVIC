"""Kernel-launching aten calls of the graph-safe step, by torchvic line,
split into calls inside a fused region (core.fusion.region: compiled on
the graph path) and calls in the eager step body. Runs the profile
driver with TORCHVIC_FUSION=0 so every op is visible to a dispatch mode:

  TORCHVIC_FUSION=0 TORCHVIC_GRAPHS_NO_CAPTURE=1 python sites.py GLOBAL \
      --cache C --cells 0 --days 1 --start-day 1 [--top 80]
"""
import collections
import os
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import profile_isc_torchvic as P  # noqa: E402
from torchvic.core import fusion  # noqa: E402
from torchvic.model import basin  # noqa: E402

TOP = 80
if '--top' in sys.argv:
    i = sys.argv.index('--top')
    TOP = int(sys.argv[i + 1])
    del sys.argv[i:i + 2]

NOLAUNCH = {'empty', 'empty_like', 'empty_strided', 'new_empty',
            'new_empty_strided', 'detach', 'alias', 'lift_fresh',
            '_local_scalar_dense', 'resize_', 'set_', 'view', '_unsafe_view',
            'as_strided', 'expand', 'slice', 'select', 'unsqueeze',
            'squeeze', 'permute', 't', 'transpose', 'reshape', 'unbind',
            'split', 'chunk', 'narrow', 'diagonal', 'view_as', 'expand_as',
            'split_with_sizes', 'unfold', 'movedim', 'is_nonzero'}

_RCODE = (fusion.region(lambda: 0).__code__, fusion.run.__code__)


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.on = False
        self.body = collections.Counter()
        self.reg = collections.Counter()
        self.byop = collections.Counter()
        self.regop = collections.Counter()
        self.sums = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if not self.on:
            return out
        name = func.overloadpacket.__name__
        if name in NOLAUNCH:
            return out
        dev = None
        for a in list(args) + list((kwargs or {}).values()):
            if torch.is_tensor(a):
                dev = a.device
                break
        if dev is None or dev.type != 'cuda':
            if not (torch.is_tensor(out) and out.is_cuda):
                return out
        f = sys._getframe(1)
        site = '?'
        g, inreg = f, False
        while g is not None:
            if g.f_code in _RCODE:
                inreg = True
                break
            g = g.f_back
        while f is not None:
            fn = f.f_code.co_filename
            if '/torchvic/' in fn and 'python_dispatch' not in fn:
                site = '%s:%d %s' % (fn.split('/torchvic/')[-1], f.f_lineno,
                                     f.f_code.co_name)
                break
            f = f.f_back
        if name in ('sum', 'mean', 'prod', 'cumsum', 'amax', 'any', 'all'):
            x0 = args[0] if args else None
            if torch.is_tensor(x0):
                self.sums[(name, inreg, tuple(x0.shape),
                           x0.is_contiguous(), str(x0.dtype),
                           str(args[1:]))] += 1
        if inreg:
            self.reg[site] += 1
            self.regop[name] += 1
        else:
            self.body[site] += 1
            self.byop[name] += 1
        return out


MODE = Count()
_sim = basin.simulate
_calls = [0]
NSTEPS = [0]


def simulate(*a, **k):
    _calls[0] += 1
    if _calls[0] == 2:
        NSTEPS[0] = len(k['batch'].dates) if hasattr(k['batch'], 'dates') \
            else 8
        MODE.on = True
        try:
            return _sim(*a, **k)
        finally:
            MODE.on = False
    return _sim(*a, **k)


basin.simulate = simulate

if __name__ == '__main__':
    with MODE:
        P.main()
    n = max(NSTEPS[0], 1)
    tb, tr = sum(MODE.body.values()), sum(MODE.reg.values())
    print('steps %d: launching aten calls/step body %.0f, in regions %.0f'
          % (n, tb / n, tr / n))
    print('== body by op')
    for k_, v_ in MODE.byop.most_common(30):
        print('  %7.1f  %s' % (v_ / n, k_))
    print('== body by line')
    for k_, v_ in MODE.body.most_common(TOP):
        print('  %7.1f  %s' % (v_ / n, k_))
    print('== regions by op')
    for k_, v_ in MODE.regop.most_common(30):
        print('  %7.1f  %s' % (v_ / n, k_))
    print('== reductions (op, in region, shape, contiguous, dtype, args)')
    for k_, v_ in MODE.sums.most_common(60):
        print('  %7.1f  %s' % (v_ / n, k_))
    print('== regions by line')
    for k_, v_ in MODE.reg.most_common(30):
        print('  %7.1f  %s' % (v_ / n, k_))
