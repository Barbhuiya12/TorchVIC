# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Cost of the irrigation and crop connectors per exchange day at a given
number of cells, with a stand-in for VIC (same exports, no physics), so the
connectors' own time is measured apart from VIC's.

    python benchmarks/bench_conn_agri.py --cells 112825 --days 31 --device cuda

Prints, per case (irrigation / crop / both): wall time per day of the whole
runtime step (compute + validate + ledger + commit) and the runtime's own
phase timings, to compare with VIC's cost per day (ISC 112,825 cells: the
direct run is 0.076 s per day on an H200, 55.7 s for two years).
"""
import argparse
import datetime
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from torchvic.coupling import (TorchVIC, Connector, Proposal, VarSpec,
                               GridSpec)
from torchvic.coupling.connectors.crop import CropConnector
from torchvic.coupling.connectors.wateruse import WaterUseConnector
from torchvic.coupling.vic import KwargsDriver, GRID

DAY = 86400.0


class MockVic(Connector):
    name = 'vic'
    interval_s = DAY
    backends = ('cpu', 'cuda')

    def __init__(self, drv, n, dev):
        super().__init__()
        self.n, self.dev, self.drv = n, dev, drv
        g = torch.Generator(device='cpu').manual_seed(1)
        mx = drv.soil().max_moist
        self.moist = (mx * (0.45 + 0.3 * torch.rand(
            mx.shape, generator=g, dtype=torch.float64).to(dev))).contiguous()
        self.transp = torch.rand(n, generator=g, dtype=torch.float64).to(dev)
        self.pet = 1.0 + 4.0 * torch.rand(n, generator=g,
                                          dtype=torch.float64).to(dev)
        self.add = None
        fl = lambda nm: VarSpec(nm, 'mm', ('cell',), kind='flux', grid=GRID)
        self.outputs = (
            VarSpec('soil_moisture', 'mm', ('cell', 'layer'), kind='state',
                    grid=GRID), fl('transp'), fl('pet'), fl('soil_add_unmet'))
        self.inputs = (fl('soil_add'),)
        self.imported_fluxes = ('soil_add',)

    def grids(self):
        d = self.drv
        return [GridSpec(GRID, self.n, area=torch.as_tensor(
            d.area * d.frac, dtype=torch.float64), kind='cells')]

    def time_span(self):
        return (0.0, self.drv.n_steps * DAY)

    def compute(self, inputs, ctx, dt):
        add = inputs.get('soil_add')
        m = self.moist
        if add is not None:
            m = m.clone()
            m[:, 0] = m[:, 0] + add * 0.0   # delivery is not simulated here
        m = m * 0.97                         # dries 3 % a day
        ex = dict(soil_moisture=m, transp=self.transp, pet=self.pet,
                  soil_add_unmet=torch.zeros_like(self.pet))
        return Proposal(self.name, ctx.t, ctx.t + dt, state=m, exports=ex,
                        backend=str(self.dev))

    def commit(self, p, ctx):
        self.moist = p.state


def build(n, days, dev):
    g = torch.Generator(device='cpu').manual_seed(0)
    L = 3
    mx = (100.0 + 100.0 * torch.rand(n, L, generator=g,
                                     dtype=torch.float64)).to(dev)
    soil = SimpleNamespace(max_moist=mx, depth=torch.tensor(
        [0.1, 0.3, 1.0], dtype=torch.float64, device=dev).expand(n, L),
        w_crit_mm=None)
    cv = torch.zeros(n, 4, dtype=torch.float64)
    cv[:, 0], cv[:, 1], cv[:, 2], cv[:, 3] = 0.3, 0.3, 0.3, 0.1
    rf = torch.zeros(n, 4, L, dtype=torch.float64)
    rf[:, :3, 0], rf[:, :3, 1], rf[:, 3, 0] = 0.4, 0.6, 1.0
    mos = SimpleNamespace(cv=cv.to(dev), root_frac=rf.to(dev))
    dates = [datetime.datetime(2004, 4, 1) + datetime.timedelta(days=i)
             for i in range(days)]
    t = torch.arange(days, dtype=torch.float64).unsqueeze(1)
    atm = dict(tair=(10 + 8 * torch.sin(t / 20) + torch.zeros(days, n)).to(dev),
               shortwave=(200 + 100 * torch.cos(t / 30) + torch.zeros(days, n)).to(dev),
               prcp=(torch.rand(days, n, generator=g, dtype=torch.float64) * 4).to(dev))
    area = np.full(n, 1.0e8)
    frac = np.ones(n)
    drv = KwargsDriver(dict(soil_params=soil, mosaic=mos),
                       lambda s0, s1: dict(atm={k: v[s0:s1]
                                                for k, v in atm.items()}),
                       dates, 1, area=area, frac=frac, n_cells=n)
    return drv


def case_model(name, drv, n, days, dev, params):
    mock = MockVic(drv, n, dev)
    conns, mods, links = dict(vic=mock), dict(vic=dict(connector='two_store')), []
    if name in ('irrigation', 'both'):
        conns['irrigation'] = WaterUseConnector(
            driver=drv, wcr_fract=0.9, wfc_fract=1.0, efficiency=0.7,
            conveyance_loss=0.1, irrigated_fraction=0.6)
        mods['irrigation'] = dict(connector='irrigation')
        links += [{'from': 'vic.soil_moisture', 'to': 'irrigation.soil_moisture'},
                  {'from': 'vic.soil_add_unmet', 'to': 'irrigation.soil_unmet'},
                  {'from': 'irrigation.soil_add', 'to': 'vic.soil_add', 'lag': 1}]
    if name in ('crop', 'both'):
        conns['crop'] = CropConnector(driver=drv, params=params,
                                      lat=45.0, max_days=days)
        mods['crop'] = dict(connector='crop')
        links += [{'from': 'vic.transp', 'to': 'crop.transp'},
                  {'from': 'vic.pet', 'to': 'crop.pet'}]
        if name == 'both':
            links += [{'from': 'crop.crop_active', 'to': 'irrigation.season'}]
    m = TorchVIC.from_dict(
        dict(run=dict(device=str(dev), log=False), modules=mods,
             connections=links), connectors=conns)
    return m


def case(name, drv, n, days, dev, params):
    m = case_model(name, drv, n, days, dev, params)
    with torch.inference_mode():
        m.initialize()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        # steady state: the first day warms allocator and kernels
        m.run(until=DAY)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        m.run()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        wall = time.time() - t0
    rep = m.report()
    return dict(case=name, cells=n, days=days - 1, device=str(dev),
                s_per_day=wall / (days - 1), runtime_timing=rep['timing'],
                ledger_ok=all(r['ok'] for r in m.ledger.balances))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', type=int, default=112825)
    ap.add_argument('--days', type=int, default=31)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--params', default='configs/crop_params_sbarley.json')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    params = json.load(open(a.params))
    drv = build(a.cells, a.days, dev)
    res = []
    for name in ('irrigation', 'crop', 'both'):
        r = case(name, drv, a.cells, a.days, dev, params)
        print(json.dumps(r), flush=True)
        res.append(r)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
