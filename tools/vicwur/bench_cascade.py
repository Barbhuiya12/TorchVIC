#!/usr/bin/env python
"""Benchmark: CascadeRouter ms per model step on the ISC network
(<data root>/static/isc_fdir.nc, 1/16 deg D8) on one GPU.

    bench_cascade.py [steps]

For n=1/L=16 and n=4/L=48 (random UHs), times rout_run + begin + sweep with
(a) one launch per level, (b) the fused narrow tail, each eager and as a
CUDA graph replay; checks the GPU against the CPU path on 2 steps; and
prints a torch.profiler table of GPU time per step by kernel (default plan).
"""
import os
import sys
import time

import netCDF4
import numpy as np
import torch

from torchvic.model.river_network import d8_downstream
from torchvic.model.routing_cascade import CascadeRouter

FDIR = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'static/isc_fdir.nc')


def network():
    with netCDF4.Dataset(FDIR) as d:
        f = np.array(d['fdir'][:])
        lat, lon = np.array(d['lat'][:]), np.array(d['lon'][:])
    net, cells = d8_downstream(f, f != 255, lat, lon)
    down = net.down.copy()
    down[down < 0] = np.nonzero(down < 0)[0]          # outlet = own index
    return down


def uh(rng, L, n):
    u = rng.random((L, n)) + 1e-3
    return u / u.sum(0)


def timeit(fn, steps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t) / steps


def profile(r, rs, rb, steps=20):
    """GPU time per step by kernel, eager default plan."""
    from torch.profiler import ProfilerActivity, profile as prof
    for _ in range(3):
        r.step(rs, rb)
    torch.cuda.synchronize()
    with prof(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(steps):
            r.step(rs, rb)
        torch.cuda.synchronize()
    rows = {}
    for e in p.key_averages():
        if e.device_type.name != 'CUDA' or e.self_device_time_total <= 0:
            continue
        k = e.key
        k = ('begin_kernel' if 'begin_kernel' in k else
             'level_kernel' if 'level_kernel' in k else
             'sums_kernel' if 'sums_kernel' in k else
             'rout_run_kernel' if 'rout_run_kernel' in k else 'torch ops')
        t, n = rows.get(k, (0.0, 0))
        rows[k] = (t + e.self_device_time_total, n + e.count)
    tot = sum(t for t, _ in rows.values())
    for k, (t, n) in sorted(rows.items(), key=lambda x: -x[1][0]):
        print('    %-22s %8.1f us/step  %5.1f launches/step'
              % (k, t / steps, n / steps))
    print('    %-22s %8.1f us/step' % ('GPU total', tot / steps))


def main(steps=200):
    down = network()
    n = len(down)
    rng = np.random.default_rng(0)
    area = rng.uniform(3e7, 5e7, n)
    rs = torch.rand(n, dtype=torch.float64, device='cuda') * 5
    rb = torch.rand(n, dtype=torch.float64, device='cuda')
    print('ISC cells %d, GPU %s' % (n, torch.cuda.get_device_name()))
    for nsub, L in ((1, 16), (4, 48)):
        a = (down, uh(rng, L, n), uh(rng, L, n), area, 86400.0, nsub)
        r = CascadeRouter(*a, device='cuda')
        sch = r.schedule
        print('n=%d L=%d: %d levels, %d groups, plan %d launches'
              % (nsub, L, sch.n_levels, len(sch.gptr) - 1, len(r._plan)))
        # correctness: GPU vs CPU path, 2 steps
        c = CascadeRouter(*a, device='cpu')
        for _ in range(2):
            x, y = torch.rand(n, dtype=torch.float64), torch.rand(
                n, dtype=torch.float64)
            r.step(x.cuda(), y.cuda())
            c.step(x, y)
        same = torch.equal(r.ring_logical().cpu(), c.ring_logical()) and \
            torch.equal(r.discharge.cpu(), c.discharge)
        print('  GPU == CPU bitwise after 2 steps: %s' % same)
        bc = CascadeRouter(*a, device='cuda').block
        for name, tail in (('per-level', 0), ('tail 1 tile (default)', None),
                           ('tail 2 tiles', 2 * bc), ('tail 8 tiles', 8 * bc)):
            r = CascadeRouter(*a, device='cuda', tail=tail)
            ms = timeit(lambda: r.step(rs, rb), steps)
            r.capture()

            def g():
                r.rout_run(rs, rb)
                r.replay()
            mg = timeit(g, steps)
            print('  %-22s launches %4d  eager %.3f ms/step  graph %.3f '
                  'ms/step' % (name, len(r._plan), ms, mg))
        profile(CascadeRouter(*a, device='cuda'), rs, rb)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
