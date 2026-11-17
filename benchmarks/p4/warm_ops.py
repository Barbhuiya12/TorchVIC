"""Kernel-launching aten ops of one graph-path warm-up step (compiled
regions live), by innermost torchvic frame, split into ops inside a
compiled region (inductor fallbacks) and ops of the eager body."""
import os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import profile_isc_torchvic as P
from torchvic.model import basin
from torchvic.core import graphsafe
from torch.profiler import profile, ProfilerActivity
_seg = graphsafe.Segmenter.begin
st = {'n': 0, 'prof': None}
_sim = basin.simulate
def sim(*a, **k):
    st['n'] += 1
    if st['n'] != 2:
        return _sim(*a, **k)
    # profile the second simulate call up to its first capture
    prof = profile(activities=[ProfilerActivity.CPU], with_stack=True,
                   experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True))
    prof.__enter__()
    st['prof'] = prof
    def begin(self):
        if st['prof'] is not None:
            torch.cuda.synchronize()
            st['prof'].__exit__(None, None, None)
            report(st['prof'])
            st['prof'] = None
        return _seg(self)
    graphsafe.Segmenter.begin = begin
    return _sim(*a, **k)
basin.simulate = sim
LAUNCH = ('cudaLaunchKernel', 'cuLaunchKernel', 'cudaMemcpyAsync', 'cudaMemsetAsync')
def report(prof):
    evs = prof.events()
    body = collections.Counter(); reg = collections.Counter(); regk = collections.Counter()
    for e in evs:
        if e.name not in LAUNCH:
            continue
        p = e.cpu_parent; inreg = False; frames = None; op = None
        while p is not None:
            if p.name.startswith('Torch-Compiled Region') and not inreg:
                inreg = p.name.replace('Torch-Compiled Region: ', 'R')
            if op is None and p.name.startswith('aten::'):
                op = p.name
            if frames is None and p.stack:
                fr = [f for f in p.stack if '/torchvic/' in f]
                if fr:
                    frames = fr[0].split('/torchvic/')[-1]
            p = p.cpu_parent
        key = '%-10s %s' % (inreg or '', (op or e.name)[:40])
        (reg if inreg else body)[key] += 1
    print('== warm-up step: launches in compiled regions %d, eager body %d'
          % (sum(reg.values()), sum(body.values())))
    print('-- eager body by line/op')
    for k, v in body.most_common(80):
        print('  %5d  %s' % (v, k))
    print('-- in compiled regions (fallbacks) by line/op')
    for k, v in reg.most_common(40):
        print('  %5d  %s' % (v, k))
    sys.stdout.flush()
P.main()
