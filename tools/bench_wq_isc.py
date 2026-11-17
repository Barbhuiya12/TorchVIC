"""Time the water-quality engine on the ISC network (daily steps).

Gauges: the real ISC gauge list snapped to the nearest cell; sites are random cells; runoff and
forcing are random.  Usage:
  bench_wq_isc.py --domain domain.nc --fdir isc_fdir.nc:fdir --gauges basin_index.csv [--sites 2000]
"""
import argparse
import time

import netCDF4
import numpy as np
import torch

from torchvic.model import river_network as rn
from torchvic.model import routing_fast
from torchvic.quality import WaterQuality


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--fdir', required=True)
    ap.add_argument('--gauges', required=True, help='basin_index.csv (real ISC gauges, nearest-cell snap)')
    ap.add_argument('--sites', type=int, default=0)
    ap.add_argument('--std', action='store_true', help='StepRouter instead of FastStepRouter')
    ap.add_argument('--point', action='store_true')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--steps', type=int, default=200)
    a = ap.parse_args()
    dev, dt64 = 'cuda', torch.float64
    d = netCDF4.Dataset(a.domain)
    lat, lon = d['lat'][:].data, d['lon'][:].data
    mask = np.asarray(d['mask'][:]) > 0
    area_g = np.asarray(d['area'][:], dtype=np.float64)
    f, v = a.fdir.split(':')
    net, cells = rn.d8_downstream(np.asarray(netCDF4.Dataset(f)[v][:]), mask, lat, lon)
    area = area_g[cells[:, 0], cells[:, 1]]
    rng = np.random.default_rng(0)
    import csv
    cl, cc = lat[cells[:, 0]], lon[cells[:, 1]]
    gl = []
    for row in csv.DictReader(open(a.gauges)):
        la, lo = float(row['lat']), float(row['lng'])
        gl.append(int(np.argmin((cl - la) ** 2 + ((cc - lo) * np.cos(np.radians(la))) ** 2)))
    gc = np.array(sorted(set(gl)))
    sites = rng.choice(net.n, a.sites, replace=False) if a.sites else []
    t0 = time.time()
    build = (rn.build_step_router if a.std else routing_fast.build_fast_step_router)
    r, info = build(net, gc, area, 86400.0, site_cells=sites, device=dev)
    sync()
    print('cells %d, router built %.1fs: nodes %d, pairs %d, L %d'
          % (net.n, time.time() - t0, info['n_node'], r.n_pair, r.L), flush=True)
    A = torch.as_tensor(area, device=dev)
    t0 = time.time()
    wq = WaterQuality(r, A, {'nitrate': 1e-6, 'phosphorus': 0.0},
                      temperature=dict(depth=1.0), site_cells=sites)
    sync()
    print('quality engine built %.1fs' % (time.time() - t0), flush=True)
    n = net.n
    g = torch.Generator(device=dev).manual_seed(0)
    ro = torch.rand(n, generator=g, device=dev, dtype=dt64) * 5
    ld = ro * 1e-3 * A
    met = dict(t_air=torch.full((n,), 15., device=dev, dtype=dt64),
               vp=torch.full((n,), 1.2, device=dev, dtype=dt64),
               wind=torch.full((n,), 2., device=dev, dtype=dt64),
               sw=torch.full((n,), 200., device=dev, dtype=dt64))
    kw = dict(loads={'nitrate': ld, 'phosphorus': 0.01 * ld}, met=met)
    if len(sites):
        kw['withdraw'] = torch.full((len(sites),), 2e5, device=dev, dtype=dt64)
        if a.point:
            kw['point'] = {'nitrate': torch.full((len(sites),), 1e3, device=dev, dtype=dt64)}
    for _ in range(5):
        wq.step(ro, **kw)
    sync(); t0 = time.time()
    for _ in range(a.steps):
        wq.step(ro, **kw)
    sync()
    print('eager: %.3f ms/step (water + 2 constituents + heat, %d sites)'
          % ((time.time() - t0) / a.steps * 1e3, len(sites)), flush=True)
    print('hook levels: %d' % len(r.levels))
    if len(sites):
        wr = torch.full((len(sites),), 2e5, device=dev, dtype=dt64)
        r.reset()
        for _ in range(5):
            r.step(ro, withdraw=wr)
        sync(); t0 = time.time()
        for _ in range(a.steps):
            r.step(ro, withdraw=wr)
        sync()
        print('water router alone (eager, withdrawals): %.3f ms/step'
              % ((time.time() - t0) / a.steps * 1e3))
        r.reset()
    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(10):
                wq.step(ro, **kw)
            sync()
        print(pr.key_averages().table(sort_by='cuda_time_total', row_limit=12, max_name_column_width=60))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        wq.step(ro, **kw)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = wq.step(ro, **kw)
    for _ in range(5):
        gr.replay()
    sync(); t0 = time.time()
    for _ in range(a.steps):
        gr.replay()
    sync()
    print('cuda graph: %.3f ms/step' % ((time.time() - t0) / a.steps * 1e3))
    wq.zero_state()
    for _ in range(a.steps):
        gr.replay()
    led = wq.ledger()
    for k, x in led.items():
        ref = (wq.r.in_runoff if k == 'water' else
               wq.temp[0].in_runoff if k == 'heat' else wq.ch[k].in_runoff)
        print('ledger %-10s max |residual|/max in = %.2e' % (k, (x.abs().max() / ref.max()).item()))
    for qm in (1e-6, 1.0):
        m = out['q'] > qm
        print('q > %g m3/s (%d nodes): nitrate max %.3e, T range %.2f..%.2f'
              % (qm, int(m.sum()), out['conc']['nitrate'][m].max(),
                 out['t'][m].min(), out['t'][m].max()))


if __name__ == '__main__':
    main()
