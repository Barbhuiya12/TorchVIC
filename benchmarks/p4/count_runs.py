"""How often fusion.run and Segmenter.replay are called per simulate call."""
import os, sys, collections, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import profile_isc_torchvic as P
from torchvic.core import fusion, graphsafe
from torchvic.model import basin
C = collections.Counter()
T = collections.Counter()
_run = fusion.run
def run(fn, *a):
    C['run:' + fn.__name__] += 1
    t = time.time()
    try:
        return _run(fn, *a)
    finally:
        T['run:' + fn.__name__] += time.time() - t
fusion.run = run
_rep = graphsafe.Segmenter.replay
def replay(self):
    C['replay'] += 1
    t = time.time()
    try:
        return _rep(self)
    finally:
        T['replay'] += time.time() - t
graphsafe.Segmenter.replay = replay
_sim = basin.simulate
def sim(*a, **k):
    C.clear(); T.clear()
    t = time.time()
    out = _sim(*a, **k)
    print('simulate %.2f s: %s' % (time.time() - t, {k: (v, round(T[k], 3)) for k, v in C.items()}), flush=True)
    return out
basin.simulate = sim
P.main()
