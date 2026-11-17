#!/usr/bin/env python3
"""TorchVIC on one GPU against TorchVIC on the CPU, same global file, same
core, float64: every output variable at every step.

    gpu_check.py [--compare-only] [ROW_ID ...]   (default: GPU_ROWS)

Needs the CPU runs of run_matrix.py (WORK/runs/<id>/tv); writes
WORK/runs/<id>/tv_gpu and results/gpu_check.json.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix as M                                          # noqa: E402
from run_matrix import _env, PY, period_days                # noqa: E402

GPU_ROWS = ['igp_EB_510_3h', 'himal_FS_510_3h', 'stehe_EB_510_1h',
            'wg_WB_501_3h']


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    only_compare = '--compare-only' in argv
    ids = [a for a in argv if not a.startswith('--')] or GPU_ROWS
    from compare import _open
    rows = {r['id']: r for r in M.MATRIX}
    res = {}
    for rid in ids:
        r = rows[rid]
        d = os.path.join(M.WORK, 'runs', rid)
        g = os.path.join(d, 'tv_gpu')
        os.makedirs(g, exist_ok=True)
        if only_compare:
            res[rid] = _diff(_open(os.path.join(d, 'tv')), _open(g))
            print(rid, res[rid], flush=True)
            continue
        for f in glob.glob(os.path.join(g, '*.nc')):
            os.remove(f)
        steps = period_days(r) * M.CASES[r['case']]['forcing_spd']
        cmd = [PY, '-m', 'torchvic', 'run', os.path.join(d, 'global.tv.txt'),
               '--core', r['core'], '--device', 'cuda', '--out', g,
               '--steps', str(steps), '--quiet']
        if r['daily']:
            cmd.append('--daily')
        p = subprocess.run(cmd, capture_output=True, text=True, env=_env(4),
                           cwd=d)
        print(rid, p.returncode, p.stdout.strip()[-200:], p.stderr[-500:],
              flush=True)
        if p.returncode:
            res[rid] = dict(error=p.stderr[-500:])
            continue
        res[rid] = _diff(_open(os.path.join(d, 'tv')), _open(g))
        print(rid, res[rid], flush=True)
    with open(os.path.join(HERE, 'results', 'gpu_check.json'), 'w') as f:
        json.dump(res, f, indent=1)


def _diff(a, b):
    if True:
        n = same = 0
        mabs = mrel = 0.0
        worst = None
        for v in a.data_vars:
            if not v.startswith('OUT_'):
                continue
            x, y = a[v].values.astype('f8'), b[v].values.astype('f8')
            ok = np.isfinite(x) | np.isfinite(y)
            n += 1
            if np.array_equal(x[ok], y[ok]):
                same += 1
                continue
            d_ = np.abs(np.nan_to_num(x[ok]) - np.nan_to_num(y[ok]))
            ma = float(d_.max())
            # relative to the variable's mean size, residual diagnostics
            # (OUT_*_ERROR, ~1e-13) excepted from the relative figure
            mr = (0.0 if v.endswith('_ERROR') else
                  ma / (float(np.nanmean(np.abs(x[ok]))) + 1e-300))
            if mr > mrel:
                worst = v
            mabs, mrel = max(mabs, ma), max(mrel, mr)
        return dict(n_vars=n, n_identical=same, max_abs=mabs, max_rel=mrel,
                    worst=worst)


if __name__ == '__main__':
    main()
