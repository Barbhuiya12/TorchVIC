"""run_isc_torchvic with each simulate call (one window) timed, and the
time to its first replay (the eager warm-up step and the capture)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torchvic.model import basin
from torchvic.core import graphsafe
import run_isc_torchvic as R
_sim = basin.simulate
_rep = graphsafe.Segmenter.replay
st = {}
def replay(self):
    if st.get('first') is None:
        torch.cuda.synchronize()
        st['first'] = time.time()
    return _rep(self)
graphsafe.Segmenter.replay = replay
def sim(*a, **k):
    torch.cuda.synchronize()
    st['first'] = None
    t0 = time.time()
    out = _sim(*a, **k)
    torch.cuda.synchronize()
    t1 = time.time()
    n = len(k['batch'].dates) if hasattr(k.get('batch'), 'dates') else -1
    pre = (st['first'] - t0) if st.get('first') else float('nan')
    print('window %4d steps: %.2f s, to first replay %.2f s, after %.2f s (%.2f ms/step)'
          % (n, t1 - t0, pre, t1 - (st['first'] or t0), 1e3 * (t1 - (st['first'] or t0)) / max(n - 1, 1)), flush=True)
    return out
basin.simulate = sim
sys.exit(R.main())
