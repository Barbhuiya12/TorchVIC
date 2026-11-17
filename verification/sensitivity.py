#!/usr/bin/env python3
"""How much each option moves VIC, and whether it moves TorchVIC.

    sensitivity.py      -> results/sensitivity.csv

For every matrix row that changes options of a base row (same case, mode,
core and step, no options), the area-weighted period totals of a few key
variables are compared three ways: VIC(option) vs VIC(base) -- the
option's effect in VIC; TorchVIC(option) vs TorchVIC(base) -- its effect in
TorchVIC (exactly 0 means TorchVIC ignored it); and TorchVIC(option) vs
VIC(option) -- the agreement on the row itself.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix as M                                          # noqa: E402
from compare import _open, _cells, _agg_sum                 # noqa: E402

VARS = ('OUT_EVAP', 'OUT_RUNOFF', 'OUT_BASEFLOW', 'OUT_SWE', 'OUT_SOIL_MOIST',
        'OUT_LATENT', 'OUT_SENSIBLE')


def totals(d, mask=None):
    ds = _open(d)
    if mask is None:
        mask = np.isfinite(ds['OUT_PREC'].values[0])
    out = {}
    for v in VARS:
        if v not in ds:
            continue
        a = np.nansum(_cells(ds, v, mask), -1)
        f = np.sum if _agg_sum(v) else np.mean
        out[v] = float(np.mean(f(a, 0)))
    return out, mask


def main():
    rows = {r['id']: r for r in M.MATRIX}
    res = []
    for r in M.MATRIX:
        if not r['opts']:
            continue
        base = [b for b in M.MATRIX if not b['opts'] and b['case'] == r['case']
                and b['mode'] == r['mode'] and b['core'] == r['core']
                and b['daily'] == r['daily'] and not b.get('params')
                and 'noflag' not in b['id']]
        if not base:
            continue
        b = base[0]
        R = lambda x, w: os.path.join(M.WORK, 'runs', x, w)  # noqa: E731
        try:
            vb, m = totals(R(b['id'], 'vic'))
            tb, _ = totals(R(b['id'], 'tv'), m)
            vo, _ = totals(R(r['id'], 'vic'), m)
            to, _ = totals(R(r['id'], 'tv'), m)
        except Exception as e:                              # noqa: BLE001
            print(r['id'], 'skipped:', e)
            continue
        for v in VARS:
            if v not in vb or v not in vo or v not in to:
                continue
            den = max(abs(vb[v]), 1e-9)
            res.append(dict(row=r['id'], base=b['id'], var=v,
                            vic_base=vb[v], vic_opt=vo[v], tv_base=tb[v],
                            tv_opt=to[v],
                            vic_effect=(vo[v] - vb[v]) / den,
                            tv_effect=(to[v] - tb[v]) / den,
                            tv_vs_vic=(to[v] - vo[v]) / max(abs(vo[v]), 1e-9),
                            tv_ignores=bool(to[v] == tb[v]
                                            and vo[v] != vb[v])))
        print(r['id'], 'done', flush=True)
    out = os.path.join(HERE, 'results', 'sensitivity.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    print('wrote', out)


if __name__ == '__main__':
    main()
