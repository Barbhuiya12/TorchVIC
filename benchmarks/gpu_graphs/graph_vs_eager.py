#!/usr/bin/env python3
"""Graph path vs eager path, bit for bit, on verification-matrix cases.

    graph_vs_eager.py CASE[,CASE...] [--mode WB] [--extra 'KEY VAL;...']
        [--core 5.1.0] [--device cuda]

Each case (verification/matrix.py) runs three times on the covered tiles
(basin.LANE_COMPACT): eagerly, as CUDA graphs (basin.GRAPHS), and under
graphsafe.SyncCheck (basin.SYNC_CHECK). Prints, per case, the graph
statistics, the outputs that differ between the first two, and every place
in the step that would wait for the device inside a capture. Exit 1 on
any difference or sync.
"""
import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
sys.path.insert(0, os.path.join(ROOT, 'verification'))

import torch                                               # noqa: E402

from torchvic import case as C                             # noqa: E402
from torchvic.core import version                          # noqa: E402
from torchvic.model import basin                           # noqa: E402
import matrix                                              # noqa: E402


def _run(c, graphs, check):
    basin.LANE_COMPACT, basin.GRAPHS, basin.SYNC_CHECK = True, graphs, check
    basin.SYNC_FOUND.clear()
    with torch.no_grad():
        o = basin.simulate(**c.kwargs())
    return o, dict(basin.GRAPH_STATS), dict(basin.SYNC_FOUND)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cases')
    ap.add_argument('--mode', default='WB')
    ap.add_argument('--extra', default='')
    ap.add_argument('--core', default='5.1.0')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    version.set_core(a.core)
    dev = torch.device(a.device)
    extra = [x for x in a.extra.split(';') if x]
    bad_any = False
    for name in a.cases.split(','):
        tmp = tempfile.mkdtemp()
        r = matrix.row(name, a.mode)
        txt = matrix.global_text(r, tmp, tmp, ['OUT_PREC'],
                                 extra_lines=extra)
        g = os.path.join(tmp, 'g.txt')
        open(g, 'w').write(txt)
        c = C.load(g, device=dev, log=lambda *x: None)
        e, _, _ = _run(c, False, False)
        q, st, _ = _run(c, True, False)
        _, st_c, found = _run(c, True, True)
        diff = []
        for k, v in e.items():
            if not torch.is_tensor(v) or k == 'discharge_mm':
                continue
            if not torch.equal(torch.nan_to_num(v, nan=1e300),
                               torch.nan_to_num(q[k], nan=1e300)):
                diff.append(k)
        ran = (st.get('graphs', 0) >= 1 if dev.type == 'cuda'
               else st_c.get('eager_steps', 0) >= 1)
        ok = not diff and not found and ran
        bad_any |= not ok
        print('%s %s core %s %s: %s graphs %s eager %s redo %s replays %s'
              % ('OK  ' if ok else 'FAIL', name, a.core, extra or '',
                 'identical' if not diff else 'DIFF ' + ','.join(diff),
                 st.get('graphs'), st.get('eager_steps'),
                 st.get('redo_bits'), st.get('loop_replays')), flush=True)
        for k, v in sorted(found.items()):
            print('     SYNC %s %s' % (k, v), flush=True)
    return 1 if bad_any else 0


if __name__ == '__main__':
    sys.exit(main())
