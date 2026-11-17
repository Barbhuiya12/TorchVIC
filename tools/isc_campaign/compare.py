"""Compare two runs' series (aligned at the end): per variable the number of
differing values, max abs / relative difference, bitwise yes/no.
  python compare.py RUN_A RUN_B [--tail N]  -> table on stdout, JSON in RUN_B/cmp_<A>.json
"""
import json
import os
import sys

import numpy as np


def compare(a, b, tail=None):
    za, zb = np.load(a + '/series.npz'), np.load(b + '/series.npz')
    res = {}
    for k in sorted(set(za.files) & set(zb.files)):
        x, y = za[k], zb[k]
        n = min(x.shape[0], y.shape[0], tail or 10 ** 9)
        x, y = x[-n:], y[-n:]
        d = np.abs(x.astype(np.float64) - y)
        ok = np.isfinite(d)
        scale = np.maximum(np.abs(x).max(), 1e-30)
        res[k] = dict(rows=int(n), differ=int((x != y).sum() - (np.isnan(x) & np.isnan(y)).sum()),
                      max_abs=float(d[ok].max()) if ok.any() else 0.0,
                      max_rel_to_max=float(d[ok].max() / scale) if ok.any() else 0.0,
                      bitwise=bool(np.array_equal(x, y, equal_nan=True)))
    return res


if __name__ == '__main__':
    a, b = sys.argv[1], sys.argv[2]
    tail = int(sys.argv[4]) if len(sys.argv) > 4 else None
    r = compare(a, b, tail)
    for k, v in r.items():
        print('%-34s rows %4d differ %9d max_abs %.3g rel %.2g %s' % (
            k, v['rows'], v['differ'], v['max_abs'], v['max_rel_to_max'],
            'BITWISE' if v['bitwise'] else 'differs'))
    json.dump(r, open('%s/cmp_%s.json' % (b, os.path.basename(a.rstrip('/'))), 'w'),
              indent=1)
    print('all bitwise:', all(v['bitwise'] for v in r.values()))
