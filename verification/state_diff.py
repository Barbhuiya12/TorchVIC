"""Diff two VIC image-driver state files, variable by variable.

    python state_diff.py A.nc B.nc [--tol 1e-12]

Dimensions, coordinate variables, fill (absent tile) positions and every
STATE_* variable, dz_node, node_depth and time are compared. A value passes
when |a - b| <= tol * max(1, |a|, |b|). Exit status 1 on any failure.
"""
import argparse
import sys

import numpy as np
import netCDF4 as nc4


def diff(pa, pb, tol=1e-12, out=print):
    a, b = nc4.Dataset(pa), nc4.Dataset(pb)
    bad = 0
    da = {k: len(v) for k, v in a.dimensions.items()
          if k not in ('time', 'tv_route_step')}
    db = {k: len(v) for k, v in b.dimensions.items()
          if k not in ('time', 'tv_route_step')}
    if da != db:
        out('DIMS differ: %s vs %s' % (da, db))
        bad += 1
    # TorchVIC-only extras (TV_*) are not in VIC's file
    names = sorted(k for k in set(a.variables) | set(b.variables)
                   if not k.startswith('TV_') and k != 'tv_route_step')
    for k in names:
        if k not in a.variables or k not in b.variables:
            out('%-30s only in %s' % (k, 'A' if k in a.variables else 'B'))
            bad += 1
            continue
        x = np.ma.filled(a[k][:].astype(float), np.nan)
        y = np.ma.filled(b[k][:].astype(float), np.nan)
        if x.shape != y.shape:
            out('%-30s shape %s vs %s' % (k, x.shape, y.shape))
            bad += 1
            continue
        fx, fy = np.isnan(x), np.isnan(y)
        if (fx != fy).any():
            out('%-30s absent-tile pattern differs at %d points'
                % (k, int((fx != fy).sum())))
            bad += 1
            continue
        m = ~fx
        err = np.abs(x - y)[m] / np.maximum(1.0, np.maximum(np.abs(x[m]),
                                                            np.abs(y[m])))
        worst = float(err.max()) if err.size else 0.0
        ok = worst <= tol
        bad += (not ok)
        out('%-30s %-5s n=%-6d max rel/abs %.3e' % (k, 'ok' if ok else 'FAIL',
                                                   int(m.sum()), worst))
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--tol', type=float, default=1e-12)
    r = ap.parse_args()
    sys.exit(1 if diff(r.a, r.b, r.tol) else 0)
