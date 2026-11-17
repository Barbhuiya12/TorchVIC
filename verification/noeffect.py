#!/usr/bin/env python3
"""VIC 5.1.0 itself, with and without each option TorchVIC reads to no
effect (torchvic/io/options.py) and a few it refuses: are VIC's outputs
bit-identical? Walnut Gulch (31 days) and Stehekin (10 days), energy
balance, every output variable at every step in double precision.

    noeffect.py            -> results/noeffect.txt
"""
import os, sys, subprocess, glob, datetime as dt
import numpy as np, xarray as xr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix as M
from run_matrix import _env
ROOT = os.path.join(M.WORK, 'noeffect')
OUT = []
CASES = {'wg': 31, 'stehe': 10}
OPTS = [('NOFLUX', 'TRUE'), ('JULY_TAVG_SUPPLIED', 'TRUE'),
        ('LAKE_PROFILE', 'TRUE'), ('LAKE_NODES', '5'), ('NODES', '5'),
        ('STATENAME', '@D@/state'), ('ROUT_PARAM', '/nonexistent/rout.nc'),
        ('QUICK_SOLVE', 'TRUE'), ('TFALLBACK', 'FALSE'),
        ('AERO_RESIST_CANSNOW', 'AR_410'), ('AERO_RESIST_CANSNOW', 'AR_406'),
        ('GRND_FLUX_TYPE', 'GF_406'), ('CLOSE_ENERGY', 'TRUE')]
ov = M.outvars()
def run(case, tag, extra):
    d = os.path.join(ROOT, case, tag)
    os.makedirs(os.path.join(d, 'out'), exist_ok=True)
    for f in glob.glob(os.path.join(d, 'out', '*')): os.remove(f)
    c = dict(M.CASES[case]); e = dt.date(*c['start']) + dt.timedelta(days=CASES[case] - 1)
    c['end'] = (e.year, e.month, e.day); M.CASES['_ne'] = c
    r = M.row(case, 'EB'); r['case'] = '_ne'
    txt = M.global_text(r, os.path.join(d, 'out'), d, ov,
                        extra_lines=[x.replace('@D@', d) for x in extra])
    if 'STATENAME' in ' '.join(extra):
        txt += 'STATEYEAR %d\nSTATEMONTH %d\nSTATEDAY %d\nSTATESEC 0\n' % (e.year, e.month, e.day)
    g = os.path.join(d, 'global.txt'); open(g, 'w').write(txt)
    p = subprocess.run([M.VIC['5.1.0'], '-g', g], capture_output=True, text=True, env=_env(1), cwd=d)
    if p.returncode: return None, (p.stdout + p.stderr).strip().splitlines()[-1][:160]
    f = sorted(glob.glob(os.path.join(d, 'out', '*.nc')))[0]
    return xr.open_dataset(f), ''
for case in CASES:
    base, _ = run(case, 'base', [])
    for k, v in OPTS:
        ds, err = run(case, '%s_%s' % (k, v.strip('/').replace('/', '_')), ['%s %s' % (k, v)])
        if ds is None:
            OUT.append('%-6s %-30s VIC rejects: %s' % (case, k + ' ' + v, err)); print(OUT[-1]); continue
        diff = []
        for n in base.data_vars:
            a, b = np.asarray(base[n].values, float), np.asarray(ds[n].values, float)
            if a.shape != b.shape or not np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9)):
                m = np.nanmax(np.abs(a - b)) if a.shape == b.shape else np.inf
                diff.append('%s %.3g' % (n, m))
        OUT.append('%-6s %-30s %s' % (case, k + ' ' + v, 'bit-identical (%d vars)' % len(base.data_vars) if not diff else 'DIFFERS: ' + ', '.join(diff[:6]) + (' ... (%d vars)' % len(diff)))); print(OUT[-1], flush=True)
with open(os.path.join(HERE, 'results', 'noeffect.txt'), 'w') as f:
    f.write('VIC 5.1.0 with each option against without it (noeffect.py)\n'
            + '\n'.join(OUT) + '\n')
