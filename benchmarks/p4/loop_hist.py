"""Iterations each captured loop takes per replay (graphsafe.Segmenter)."""
import os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import profile_isc_torchvic as P
from torchvic.core import graphsafe
H = collections.Counter()
NL = collections.Counter()
def replay(self):
    n_iter = 0
    li = 0
    for it in self.items:
        if it[0] == 'graph':
            it[1].replay(); continue
        _, g, stop, cap, pin, evs = it
        n = 0
        while n < cap and not bool(stop):
            g.replay(); n += 1
        H[(li, n)] += 1
        NL[li] += 1
        li += 1
        n_iter += n
    return n_iter
graphsafe.Segmenter.replay = replay
P.main()
print('loops per step: %d' % len(NL))
for li in sorted(NL):
    d = {n: c for (l, n), c in H.items() if l == li}
    tot = sum(d.values())
    print('loop %d: replays %d, mean iters %.2f, hist %s' % (li, tot, sum(n * c for n, c in d.items()) / tot, dict(sorted(d.items()))))
