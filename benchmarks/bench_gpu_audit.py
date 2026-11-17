# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Host-sync audit of the coupled-system connectors on the GPU.

Runs each connector case with sync-free stand-ins for VIC (no float()/.item()
in the stand-in itself), `torch.cuda.set_sync_debug_mode('warn')` around the
per-day loop, and counts every host sync by the innermost torchvic frame that
caused it, plus the time per coupled day.

    python benchmarks/bench_gpu_audit.py --case river --cells 112825 --days 30
    cases: river (routing+reservoir+quality), irrigation, crop, both,
           groundwater, all (irrigation+crop+groundwater+river on one grid)
"""
import argparse
import collections
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'tests'))

DAY = 86400.0


class SyncCounter(object):
    def __init__(self, depth_filter='torchvic'):
        self.hits = collections.Counter()
        self.filter = depth_filter

    def __enter__(self):
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter('always')
        self._old = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            if 'synchroniz' not in str(message):
                return
            where = '?'
            for fr in reversed(traceback.extract_stack()):
                fn = fr.filename
                if self.filter in fn and 'bench_gpu_audit' not in fn:
                    where = '%s:%d' % (fn.split('torchvic/')[-1], fr.lineno)
                    break
            self.hits[where] += 1
        warnings.showwarning = show
        torch.cuda.set_sync_debug_mode('warn')
        return self

    def __exit__(self, *a):
        torch.cuda.set_sync_debug_mode(0)
        warnings.showwarning = self._old
        self._ctx.__exit__(*a)


def measure(m, days, dev, warm=2):
    """Per-day host syncs (by site) and ms/day of m.run()."""
    with torch.inference_mode():
        m.initialize()
        m.run(until=warm * DAY)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        n_timed = days - warm
        t0 = time.time()
        m.run(until=(warm + 2) * DAY)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t_one = (time.time() - t0) / 2
        sc = SyncCounter() if dev.type == 'cuda' else None
        if sc:
            with sc:
                m.run(until=(warm + 2 + 3) * DAY)
        else:
            m.run(until=(warm + 2 + 3) * DAY)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        m.run()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        wall = time.time() - t0
        left = days - (warm + 2 + 3)
    out = dict(ms_per_day=1e3 * wall / max(left, 1))
    if sc:
        out['syncs_per_day'] = sum(sc.hits.values()) / 3.0
        out['sites'] = {k: v / 3.0 for k, v in sc.hits.most_common()}
    out['ledger_ok'] = all(r['ok'] for r in m.ledger.balances)
    return out


# --------------------------------------------------------------------------
def case_river(n, days, dev, quality=True, reservoir=True):
    from torchvic.coupling import (TorchVIC, Connector, Proposal, Balance,
                                   VarSpec, GridSpec)
    from torchvic.model import river_network as rn
    import test_coupling_routing as T
    ny = int(round(np.sqrt(n / 1.2)))
    nx = int(round(n / ny))
    elev_net = _big_net(ny, nx)
    net = elev_net
    gen = torch.Generator().manual_seed(0)
    area = (torch.rand(net.n, generator=gen, dtype=torch.float64) * 1e7
            + 2e7).numpy()
    table = (torch.rand(days + 1, net.n, generator=gen,
                        dtype=torch.float64) * 5).to(dev)
    gauges = np.unique([int(net.order[-1]), int(net.order[-40]),
                        int(net.order[len(net.order) // 2])])

    class Source(Connector):
        name = 'src'
        backends = ('cpu', 'cuda')
        interval_s = DAY
        exported_fluxes = ('total_runoff',)
        outputs = (VarSpec('total_runoff', 'mm', ('cell',), kind='flux',
                           grid=T.GRID),)

        def __init__(self):
            super().__init__()
            self.k = 0
            self.a = torch.as_tensor(area, dtype=torch.float64, device=dev)

        def grids(self):
            return [GridSpec(T.GRID, net.n, area=torch.as_tensor(area))]

        def initialize(self, context):
            self.dev = context.device

        def time_span(self):
            return (0.0, (days + 1) * DAY)

        def compute(self, inputs, context, dt):
            k0 = int(round(context.t / DAY))
            x = table[k0]
            v = (x * self.a).sum() * 1e-3
            return Proposal(self.name, context.t, context.t + dt,
                            state=k0 + 1, exports=dict(total_runoff=x),
                            balance=Balance(P=v, Q_out=v, n_terms=x.numel(),
                                            scale=2 * v.abs()))

        def commit(self, proposal, context):
            self.k = proposal.state

    rk = dict(net=net, gauge_cells=gauges, area=area, cell_uh=[0.3, 0.5, 0.2])
    mods = dict(src=dict(connector='two_store'))
    rt = dict(rk)
    if reservoir:
        dams, rkk, cells = T.reservoir_config(net)
        rkk = dict(rkk, request_m3=2.0e5)
        rt['reservoirs'] = 'reservoir'
    if quality:
        rt['quality'] = 'quality'
    mods['routing'] = dict(connector='routing', **rt)
    if reservoir:
        mods['reservoir'] = dict(connector='reservoir', dams=dams, **rkk)
    if quality:
        mods['quality'] = dict(connector='quality', **dict(
            T.QUAL, loads={'n': 3.0, 'tracer': 1.5}))
    m = TorchVIC.from_dict(
        dict(run=dict(device=str(dev), log=False), modules=mods,
             connections=[{'from': 'src.total_runoff',
                           'to': 'routing.inflow'}]),
        connectors=dict(src=Source()))
    return m, net.n


def _big_net(ny, nx, seed=3):
    from torchvic.model import river_network as rn
    g = np.random.default_rng(seed)
    elev = g.random((ny, nx)) * 100 + np.linspace(0, 400, nx)[None, :]
    mask = np.ones((ny, nx), bool)
    lat = 20 + np.arange(ny) * 0.0625
    lon = 70 + np.arange(nx) * 0.0625
    net, cells = rn.derive_d8_from_elevation(elev, mask, lat, lon)
    return net


def case_agri(which, n, days, dev):
    import bench_conn_agri as B
    drv = B.build(n, days, dev)
    params = json.load(open(os.path.join(HERE, '..', 'configs',
                                         'crop_params_sbarley.json')))
    return B.case_model(which, drv, n, days, dev, params), n


def case_gw(n, days, dev, extra=None):
    from torchvic.coupling import (TorchVIC, Connector, Proposal, Balance,
                                   VarSpec, GridSpec)
    ny = int(round(np.sqrt(n)))
    nx = n // ny
    n = ny * nx
    AREA = 25.0e6
    lat, lon = np.meshgrid(10.0 + 0.01 * np.arange(ny),
                           -100.0 + 0.01 * np.arange(nx), indexing='ij')
    dom = dict(lat=lat.ravel(), lon=lon.ravel(), area=np.full(n, AREA),
               frac=np.ones(n))
    GRID = 'vic_cells'

    class Src(Connector):
        name = 'src'
        backends = ('cpu', 'cuda')
        interval_s = DAY
        exported_fluxes = ('baseflow', 'runoff')
        owned_state = ('src_nothing',)
        outputs = (VarSpec('baseflow', 'mm', ('cell',), kind='flux', grid=GRID),
                   VarSpec('runoff', 'mm', ('cell',), kind='flux', grid=GRID),
                   VarSpec('request', 'm3', ('cell',), kind='state',
                           grid=GRID))

        def __init__(self):
            super().__init__()
            self.i = 0

        def grids(self):
            return [GridSpec(GRID, n, area=torch.full(
                (n,), AREA, dtype=torch.float64))]

        def initialize(self, ctx):
            self.dev = ctx.device
            self.bf = torch.full((n,), 2.0, dtype=torch.float64,
                                 device=self.dev)
            self.ro = torch.full((n,), 1.0, dtype=torch.float64,
                                 device=self.dev)
            self.rq = torch.full((n,), 2.0e4, dtype=torch.float64,
                                 device=self.dev)
            self.v = (self.bf + self.ro).sum() * (AREA / 1000.0)

        def compute(self, inputs, ctx, dt):
            return Proposal(self.name, ctx.t, ctx.t + dt, state=self.i + 1,
                            exports=dict(baseflow=self.bf, runoff=self.ro,
                                         request=self.rq),
                            balance=Balance(P=self.v, Q_out=self.v,
                                            n_terms=2 * n),
                            backend=str(self.dev))

        def commit(self, p, ctx):
            self.i = p.state

    m = TorchVIC.from_dict(
        dict(run=dict(device=str(dev), log=False, duration_days=days),
             modules=dict(src=dict(connector='two_store'),
                          groundwater=dict(connector='groundwater',
                                           domain=dom, **(extra or {}))),
             connections=[{'from': 'src.baseflow',
                           'to': 'groundwater.recharge'},
                          {'from': 'src.request',
                           'to': 'groundwater.pumping_request'}]),
        connectors=dict(src=Src()))
    return m, n


def run_flood(n, days, dev):
    """FloodConnector (engine backend) driven directly: compute + validate +
    commit per day on a ny x nx basin, runoff and inflow on the device."""
    from torchvic.coupling.connectors import flood as F
    from torchvic.coupling.connector import Context, flags_of
    from torchvic.coupling.hostfetch import Fetch
    ny = nx = int(round(np.sqrt(n)))
    yy, xx = np.mgrid[0:ny, 0:nx]
    dem = 20.0 + 0.002 * ((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) ** 0.5
    fl = F.FloodConnector(dem=dem, dx=100.0, interval_s=DAY, backend='engine',
                          inflow_points=[(ny // 2, nx // 2)])
    ctx = Context(t=0.0, device=str(dev), dtype=torch.float64)
    fl.initialize(ctx)
    V = torch.full((ny * nx,), 50.0, dtype=torch.float64, device=dev)
    Vp = torch.full((1,), 1.0e4, dtype=torch.float64, device=dev)

    def day(k):
        ctx.t = k * DAY
        p = fl.compute(dict(runoff=V, inflow=Vp), ctx, DAY)
        res = fl.validate(p, ctx)
        fh = Fetch()
        hs = [(f, fh.add(f.vector())) for f in flags_of(res)]
        fh.run()
        assert not [f for f, h in hs if f.text(h.value())]
        fl.commit(p, ctx)
    with torch.inference_mode():
        for k in range(2):
            day(k)
        torch.cuda.synchronize()
        sc = SyncCounter()
        with sc:
            for k in range(2, 5):
                day(k)
        torch.cuda.synchronize()
        t0 = time.time()
        for k in range(5, days):
            day(k)
        torch.cuda.synchronize()
        wall = time.time() - t0
    return dict(ms_per_day=1e3 * wall / (days - 5),
                syncs_per_day=sum(sc.hits.values()) / 3.0,
                sites={k: v / 3.0 for k, v in sc.hits.most_common()},
                case='flood', cells=ny * nx, device=str(dev))


def run_config(cfg, warm):
    """A coupled YAML through the runtime (StreamDriver for a large VIC
    forcing): the first `warm` days untimed, then every host sync and the
    wall time of the rest, per simulated day."""
    from torchvic.coupling import TorchVIC
    m = TorchVIC.from_config(cfg, log=lambda *x: None)
    m.cfg['output']['dir'] = os.environ.get('AUDIT_OUT', 'audit_out')
    with torch.inference_mode():
        m.initialize()
        m.run(until=warm * DAY)
        torch.cuda.synchronize()
        t_a = m.runtime.time.t
        t0 = time.time()
        with SyncCounter() as sc:
            m.run()
        torch.cuda.synchronize()
        wall = time.time() - t0
        days = (m.runtime.time.t - t_a) / DAY
    return dict(case='config', config=os.path.basename(cfg), days=days,
                ms_per_day=1e3 * wall / max(days, 1),
                syncs_per_day=sum(sc.hits.values()) / max(days, 1),
                sites={k: v / max(days, 1) for k, v in sc.hits.most_common(12)},
                ledger_ok=all(r['ok'] for r in m.ledger.balances)
                and all(r['ok'] for r in m.ledger.coupled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='river')
    ap.add_argument('--cells', type=int, default=112825)
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None)
    ap.add_argument('--config', default=None,
                    help='case config: a coupled YAML (e.g. the streamed '
                    'ISC VIC); syncs counted after the first --warm days')
    ap.add_argument('--warm', type=int, default=31)
    a = ap.parse_args()
    dev = torch.device(a.device)
    c = a.case
    if c == 'config':
        r = run_config(a.config, a.warm)
        print('RESULT ' + json.dumps(r), flush=True)
        if a.out:
            json.dump(r, open(a.out, 'w'), indent=1)
        return
    if c == 'river':
        m, n = case_river(a.cells, a.days, dev)
    elif c == 'routing':
        m, n = case_river(a.cells, a.days, dev, quality=False,
                          reservoir=False)
    elif c in ('irrigation', 'crop', 'both'):
        m, n = case_agri(c, a.cells, a.days, dev)
    elif c == 'flood':
        r = run_flood(a.cells, a.days, dev)
        print('RESULT ' + json.dumps(r), flush=True)
        if a.out:
            json.dump(r, open(a.out, 'w'), indent=1)
        return
    elif c == 'groundwater':
        m, n = case_gw(a.cells, a.days, dev)
    else:
        raise SystemExit(c)
    r = measure(m, a.days, dev)
    r.update(case=c, cells=n, device=str(dev),
             timing=m.report()['timing'])
    print('RESULT ' + json.dumps(r), flush=True)
    if a.out:
        json.dump(r, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
