#!/usr/bin/env python3
"""Verification-matrix cases as CUDA graphs, saved or compared bit for bit.

    cases_vs_base.py CASE[,CASE...] --save DIR | --compare DIR
        [--mode WB] [--core 5.1.0] [--extra 'KEY VAL;...'] [--eager]

Runs each case (verification/matrix.py) through basin.simulate on the
covered tiles with CUDA graphs, with the torchvic first on PYTHONPATH, in
two windows (the state carried across) -- or eagerly (--eager) -- and
writes every output and state
tensor to DIR/CASE.pt -- or compares them with what DIR holds: run once
with the base tree (save) and once with the tree under test (compare).
Prints, per case, IDENTICAL or each differing variable with its largest
absolute difference. Exit 1 on any difference (discharge_mm excepted:
index_add's order on a GPU is not fixed).
"""
import argparse
import os
import sys
import tempfile

import torch


def flat(o, pre=''):
    out = {}
    for k, v in o.items():
        if torch.is_tensor(v):
            out[pre + k] = v.detach().cpu()
        elif isinstance(v, dict):
            out.update(flat(v, pre + k + '.'))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cases')
    ap.add_argument('--save')
    ap.add_argument('--compare')
    ap.add_argument('--mode', default='WB')
    ap.add_argument('--core', default='5.1.0')
    ap.add_argument('--extra', default='')
    ap.add_argument('--eager', action='store_true',
                    help='the eager step (no CUDA graphs) instead')
    a = ap.parse_args(argv)
    import torchvic
    root = os.path.dirname(os.path.dirname(torchvic.__file__))
    sys.path.insert(0, os.path.join(root, 'verification'))
    import matrix
    from torchvic import case as C
    from torchvic.core import version
    from torchvic.model import basin, state as model_state
    version.set_core(a.core)
    print('torchvic', root, flush=True)
    extra = [x for x in a.extra.split(';') if x]
    bad = False
    for name in a.cases.split(','):
        tmp = tempfile.mkdtemp()
        r = matrix.row(name, a.mode)
        g = os.path.join(tmp, 'g.txt')
        open(g, 'w').write(matrix.global_text(r, tmp, tmp, ['OUT_PREC'],
                                              extra_lines=extra))
        c = C.load(g, device=torch.device('cuda'), log=lambda *x: None)
        basin.LANE_COMPACT, basin.GRAPHS = True, not a.eager
        kw = c.kwargs()
        nt = int(kw['atm']['tair'].shape[0])
        cut = (nt // 2) // 8 * 8
        res, st = {}, None
        with torch.no_grad():
            for i, (s0, s1) in enumerate(((0, cut), (cut, nt))):
                o = basin.simulate(**model_state.time_window(kw, s0, s1),
                                   initial_state=st, return_state=True)
                st = o['state']
                res.update(flat(o, 'w%d.' % i))
        stats = dict(basin.GRAPH_STATS)
        tag = '%s_%s%s' % (name, a.core, '_x' if extra else '')
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            torch.save(res, os.path.join(a.save, tag + '.pt'))
            print('saved %s: %d tensors, %d steps, graphs %s' % (
                tag, len(res), nt, stats.get('graphs')), flush=True)
            continue
        base = torch.load(os.path.join(a.compare, tag + '.pt'))
        rows = []
        for k in sorted(set(base) | set(res)):
            if k not in base or k not in res:
                rows.append('%s missing in %s' % (k, 'base' if k in res
                                                  else 'new'))
                continue
            x, y = base[k], res[k]
            if x.shape != y.shape:
                rows.append('%s shape %s vs %s' % (k, x.shape, y.shape))
                continue
            if torch.equal(torch.nan_to_num(x, nan=1e300),
                           torch.nan_to_num(y, nan=1e300)):
                continue
            d = (x.double() - y.double()).abs().nan_to_num().max().item() \
                if x.is_floating_point() else float((x != y).sum())
            rows.append('%s max|d| %.3e' % (k, d))
        real = [r_ for r_ in rows if 'discharge_mm' not in r_]
        bad |= bool(real)
        print('%s %s: %d steps, %d tensors, graphs %s replays %s redo %s'
              % ('IDENTICAL' if not real else 'DIFFERENT', tag, nt, len(res),
                 stats.get('graphs'), stats.get('loop_replays'),
                 stats.get('redo_bits')), flush=True)
        for r_ in rows:
            print('    ' + r_, flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
