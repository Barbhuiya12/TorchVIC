"""Timing of ensemble scaling and DA cost.

    python -m torchvic.ensemble.bench GLOBAL --steps 64 --members 1 2 4 8 16 32

One warm-up run per size (graph capture, compilation), then the timed run
with the device synchronised. Also checks that member 0 with zero
perturbation equals the deterministic run bitwise, and times one LETKF
analysis on the largest ensemble.
"""
import argparse
import json
import time

import torch

from .. import case
from .assim import Assimilator
from .members import Ensemble
from .perturb import ForcingPerturbation, sample_parameters


def _sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('global_file')
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--members', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out')
    ap.add_argument('--da', action='store_true')
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    c = case.load(a.global_file, device=dev, steps=a.steps)
    print('case', c, flush=True)
    res = dict(cells=c.n_cells, steps=c.n_steps, device=str(dev), runs=[])
    # deterministic reference and bitwise check
    d = c.run()
    _sync(dev)
    o = Ensemble(c, 2).run()
    ok = all(torch.equal(o[k][m], d[k].T) for k in ('et', 'runoff', 'baseflow')
             for m in range(2))
    res['zero_perturbation_bitwise'] = bool(ok)
    print('zero perturbation == deterministic, bitwise:', ok, flush=True)
    del d, o
    last = None
    for n in a.members:
        fp = ForcingPerturbation(prcp_sigma=0.3, tair_sigma=0.5, phi=0.8,
                                 control=True, seed=1)
        prm = sample_parameters({'rmin_scale': (0.7, 1.4)}, n, seed=2,
                                control={'rmin_scale': 1.0}, dtype=c.dtype)
        e = Ensemble(c, n, forcing=fp, params=prm)
        e.run(keep=('runoff',))                       # warm-up
        _sync(dev)
        t = time.time()
        e.run(keep=('runoff',))
        _sync(dev)
        dt = time.time() - t
        row = dict(members=n, seconds=dt, ms_per_step=1e3 * dt / c.n_steps,
                   s_per_member=dt / n,
                   peak_gb=(torch.cuda.max_memory_allocated() / 1e9
                            if dev.type == 'cuda' else 0.0))
        res['runs'].append(row)
        print(json.dumps(row), flush=True)
        last = e
        del e
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    if a.da and last is not None:
        e = last
        o = e.run(0, c.n_steps // 2, return_state=True, keep=('runoff',))
        A = Assimilator(e, layers=(0, 1), method='letkf', radius_km=100.0,
                        inflation=1.05, max_local_obs=32, keep_cells=False)
        x = A.matrix(o['state'])
        cells = torch.arange(0, e.nc, 10)
        obs = A.obs_soil_moisture(cells, x[:, cells, 0].mean(0).cpu(), 2.0,
                                  layers=[0])
        A.analysis(o['state'], obs)
        _sync(dev)
        t = time.time()
        A.analysis(o['state'], obs)
        _sync(dev)
        res['letkf_seconds'] = time.time() - t
        res['letkf_members'] = e.n
        res['letkf_obs'] = int(cells.numel())
        print('letkf', res['letkf_seconds'], flush=True)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
