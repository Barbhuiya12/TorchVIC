#!/usr/bin/env python3
"""Two runs' history files, variable by variable: bit-identical or not.

    compare_runs.py DIR_A DIR_B [--json FILE]

Every NetCDF file in DIR_A is matched by name in DIR_B; for each data
variable the number of differing values, the largest absolute difference and
that difference relative to the variable's largest magnitude are reported,
and whether the NaN (missing) pattern is the same. Read one variable at a
time, so a 105,113-cell year fits in memory.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import xarray as xr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    rows, identical, worst_rel = [], True, 0.0
    files = sorted(glob.glob(os.path.join(a.a, '*.nc')))
    if not files:
        print('no NetCDF files in %s' % a.a)
        return 2
    for fa in files:
        fb = os.path.join(a.b, os.path.basename(fa))
        if not os.path.exists(fb):
            print('%s: missing in %s' % (os.path.basename(fa), a.b))
            identical = False
            continue
        da, db = xr.open_dataset(fa), xr.open_dataset(fb)
        for v in da.data_vars:
            if v not in db or da[v].dtype.kind not in 'fiu':
                continue
            x = np.asarray(da[v].values, dtype=np.float64)
            y = np.asarray(db[v].values, dtype=np.float64)
            if x.shape != y.shape:
                rows.append(dict(file=os.path.basename(fa), var=v,
                                 note='shape %s vs %s' % (x.shape, y.shape)))
                identical = False
                continue
            nx, ny = np.isnan(x), np.isnan(y)
            same_nan = bool((nx == ny).all())
            ok = ~(nx | ny)
            d = np.abs(x[ok] - y[ok])
            n_diff = int((d != 0).sum()) + int((nx != ny).sum())
            dmax = float(d.max()) if d.size else 0.0
            scale = float(np.abs(y[ok]).max()) if d.size else 0.0
            rel = dmax / scale if scale > 0 else dmax
            worst_rel = max(worst_rel, rel)
            identical &= (n_diff == 0)
            rows.append(dict(file=os.path.basename(fa), var=v, n=int(x.size),
                             n_diff=n_diff, max_abs=dmax, max_rel=rel,
                             same_nan=same_nan))
            print('%-28s %-16s differ %9d / %10d  max|d| %.3e  rel %.3e  %s'
                  % (os.path.basename(fa), v, n_diff, x.size, dmax, rel,
                     'NaN pattern same' if same_nan else 'NaN PATTERN DIFFERS'),
                  flush=True)
        da.close(); db.close()
    print('%s (largest relative difference %.3e)'
          % ('BIT-IDENTICAL' if identical else 'DIFFERENT', worst_rel))
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(dict(a=a.a, b=a.b, identical=identical,
                           max_rel=worst_rel, rows=rows), fh, indent=1)
    return 0 if identical else 1


if __name__ == '__main__':
    sys.exit(main())
