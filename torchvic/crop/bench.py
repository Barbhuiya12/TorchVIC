# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Time the batched WOFOST engine: cells x seasons per second.

    python -m torchvic.crop.bench REF.npz [--n 112225] [--device cuda] [--mode pp|cwb]

REF.npz is one PCSE reference case (tests/data/crop_pcse); its weather and
parameters are replicated over N cells with a small per-cell perturbation of
radiation and temperature so the cells are not identical.
"""
import argparse
import json
import time

import numpy as np
import torch

from .params import CropParams
from .run import simulate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('--n', type=int, default=112225)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--mode', default='pp')
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args(argv)
    z = np.load(a.ref)
    meta = json.loads(str(z['meta']))
    dev, dt = a.device, torch.float64
    n, T = a.n, meta['ndays']
    g = torch.Generator().manual_seed(1)
    jit = 1.0 + 0.1 * (torch.rand(n, generator=g, dtype=dt) - 0.5)
    shift = 2.0 * (torch.rand(n, generator=g, dtype=dt) - 0.5)
    f = {}
    for k, col in (('tmin', 'TMIN'), ('tmax', 'TMAX'), ('irrad', 'IRRAD'), ('rain', 'RAIN'),
                   ('e0', 'E0'), ('es0', 'ES0'), ('et0', 'ET0')):
        base = torch.tensor(z['wx_' + col], dtype=dt)[:, None]
        if k in ('tmin', 'tmax'):
            f[k] = (base + shift[None]).to(dev)
        elif k == 'irrad':
            f[k] = (base * jit[None]).to(dev)
        else:
            f[k] = base.expand(T, n).contiguous().to(dev)
    f['doy'] = torch.tensor(z['doy'], dtype=dt)[:, None].to(dev)
    lat = torch.full((n,), float(z['lat']), dtype=dt, device=dev)
    cp = CropParams.from_sets([meta['params']], n=n, device=dev)
    mode = a.mode
    hi = None if meta['harvest_idx'] is None else torch.full((n,), meta['harvest_idx'])
    md = torch.full((n,), meta['max_duration'])
    ts = []
    for r in range(a.reps + 1):
        if dev.startswith('cuda'):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, w = simulate(cp, f, lat, mode=mode, outputs=('tagp',), harvest_idx=hi, max_duration=md)
        if dev.startswith('cuda'):
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    best = min(ts[1:])
    name = torch.cuda.get_device_name() if dev.startswith('cuda') else 'cpu x%d' % torch.get_num_threads()
    print('%s mode=%s case=%s: %d cells x %d days  best %.3fs  (first %.3fs)  %.0f cells(seasons)/s  '
          '%.2f us per cell-day' % (name, mode, meta['name'], n, T, best, ts[0], n / best,
                                    1e6 * best / (n * T)))
    print('TAGP mean %.1f  min %.1f max %.1f' % (out['tagp'][-1].mean(), out['tagp'][-1].min(), out['tagp'][-1].max()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
