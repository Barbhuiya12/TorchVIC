"""S10: VIC with the energy balance instead of the water balance, on ISC,
through benchmarks/run_isc_torchvic.py with the benchmark global files
(3-hourly forcing 2004-2005, 2004 = spin-up; one snow band per cell).
  run_eb.py {wb,eb} [--device cuda]
writes isc_campaign/out/S10_<wb|eb>/series.npz (2005, daily, cell level: the
variables the global file outputs) so that sanity.py and compare.py apply."""
import glob
import json
import os
import subprocess
import sys
import time

import netCDF4 as nc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
R = os.environ.get('TORCHVIC_DATA_ROOT', 'data').rstrip('/') + '/'
OUT = R + 'verification/isc_campaign/out/'
GF = {'wb': 'isc_vic5_3h.txt', 'eb': 'isc_vic5_3h_eb.txt'}


def main():
    kind = sys.argv[1]
    dev = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv \
        else 'cuda'
    d = OUT + 'S10_' + kind + '/'
    os.makedirs(d + 'out', exist_ok=True)
    os.makedirs(d + 'log', exist_ok=True)
    g = open(REPO + '/benchmarks/isc/' + GF[kind]).read() \
        .replace('@DATA_ROOT@', R.rstrip('/')).replace('@RESULT_DIR@', d + 'out/').replace('@LOG_DIR@', d + 'log/')
    open(d + 'global.txt', 'w').write(g)
    t0 = time.time()
    r = subprocess.run([sys.executable, REPO + '/benchmarks/run_isc_torchvic.py',
                        d + 'global.txt', '--out', d + 'out', '--device', dev,
                        '--timing', d + 'timing.json'],
                       capture_output=True, text=True)
    open(d + 'run.log', 'w').write(r.stdout + r.stderr)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    S = {}
    names = dict(OUT_PREC='vic.rainf', OUT_EVAP='vic.et',
                 OUT_RUNOFF='vic.runoff', OUT_BASEFLOW='vic.baseflow',
                 OUT_SWE='vic.swe', OUT_SOIL_MOIST='vic.soil_moist')
    acc = {}
    for f in sorted(glob.glob(d + 'out/*.nc')):
        with nc.Dataset(f) as ds:
            tv = ds['time']
            dates = nc.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))
            sel = np.array([x.year == 2005 for x in dates])
            if not sel.any():
                continue
            dom = nc.Dataset(R + 'vic_params_isc/domain.nc')
            m = np.ma.filled(dom['mask'][:], 0) > 0
            for v, k in names.items():
                if v not in ds.variables:
                    continue
                a = np.ma.filled(ds[v][:], np.nan)[sel]
                if a.ndim == 4:                       # layers
                    a = a.sum(1)
                acc.setdefault(k, []).append(a[:, m].astype(np.float32))
    for k, v in acc.items():
        S[k] = np.concatenate(v, 0)
    np.savez(d + 'series.npz', **S)
    json.dump(dict(step='S10_' + kind, mode='fixed', days=365, device=dev,
                   wall_s=time.time() - t0, ledger=dict(all_ok=True)),
              open(d + 'meta.json', 'w'))
    print({k: v.shape for k, v in S.items()})


if __name__ == '__main__':
    main()
