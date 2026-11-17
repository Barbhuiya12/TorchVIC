"""One-month flood event on the Brahmaputra/Guwahati test basin through the
coupled runtime: stored TorchVIC runoff (ISC daily run, July 2004) replayed
into the flood connector (LISFLOOD-FP ACC, GPU), plus a stand-in for routed
upstream inflow at the eastern boundary. Prints one JSON line.

    python benchmarks/bench_flood_connector.py --days 31 --solver acc_gpu --out DIR
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchvic.coupling import Connector, GridSpec, TorchVIC, VarSpec, registry  # noqa: E402
from torchvic.coupling.connector import Balance, Proposal                      # noqa: E402
from torchvic.coupling.connectors import flood as F                            # noqa: E402

ROOT = Path(os.environ.get('TORCHVIC_DATA_ROOT', 'data'))
DAY = 86400.0


class Replay(Connector):
    """Stored VIC runoff (+ baseflow) as the 'vic' module of the event."""
    name = 'vic'
    exported_fluxes = ('runoff',)
    backends = ('cpu', 'cuda')

    def __init__(self, mm, area, **c):
        super().__init__(**c)
        self.mm, self.area, self.interval_s, self.k = mm, area, DAY, 0
        self.outputs = (VarSpec('runoff', 'mm', ('cell',), kind='flux', grid='vic_cells'),)

    def grids(self):
        return [GridSpec('vic_cells', self.mm.shape[1], area=torch.as_tensor(self.area))]

    def compute(self, inputs, context, dt):
        x = torch.as_tensor(self.mm[int(round(context.t / DAY))], dtype=torch.float64)
        v = float((x * 1e-3 * torch.as_tensor(self.area)).sum())
        return Proposal('vic', context.t, context.t + dt, exports=dict(runoff=x),
                        balance=Balance(P=v, Q_out=v, n_terms=len(x), scale=2 * v), backend='cpu')

    def commit(self, proposal, context):
        pass


class Inflow(Connector):
    """Stand-in for the routing connector's discharge at the east boundary."""
    name = 'bc'
    exported_fluxes = ('inflow',)
    backends = ('cpu', 'cuda')

    def __init__(self, q, n, **c):
        super().__init__(**c)
        self.q, self.n, self.interval_s = q, n, DAY
        self.outputs = (VarSpec('inflow', 'm3', ('point',), kind='flux', grid=F.INFLOW_GRID),)

    def grids(self):
        return [GridSpec(F.INFLOW_GRID, self.n)]

    def compute(self, inputs, context, dt):
        v = self.q[int(round(context.t / DAY))] * dt
        x = torch.full((self.n,), v / self.n, dtype=torch.float64)
        return Proposal('bc', context.t, context.t + dt, exports=dict(inflow=x),
                        balance=Balance(P=v, Q_out=v, scale=2 * v), backend='cpu')

    def commit(self, proposal, context):
        pass


def main():
    import netCDF4
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=31)
    ap.add_argument('--solver', default='acc_gpu')
    ap.add_argument('--basin', default=str(ROOT / 'static/flood_basins/brahmaputra_guwahati_100m.npz'))
    ap.add_argument('--pairs', default=str(ROOT / 'static/flood_basins/brahmaputra_guwahati_100m_vic_pairs.csv'))
    ap.add_argument('--start-day', type=int, default=182, help='day of year index (0-based); 182 = 1 Jul 2004')
    ap.add_argument('--q0', type=float, default=10000.0)
    ap.add_argument('--q1', type=float, default=28000.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b = F.load_basin(a.basin)
    ny, nx = b['dem'].shape
    # VIC runoff + baseflow of the box's 256 cells, daily, mm
    dom = netCDF4.Dataset(str(ROOT / 'vic_params_isc/domain.nc'))
    lat, lon = dom['lat'][:], dom['lon'][:]
    i0 = int(np.argmin(abs(lat - (b['lat0'] + 0.03125))))
    j0 = int(np.argmin(abs(lon - (b['lon0'] + 0.03125))))
    area = np.array(dom['area'][i0:i0 + 16, j0:j0 + 16] * dom['frac'][i0:i0 + 16, j0:j0 + 16]).reshape(-1)
    r = netCDF4.Dataset(str(ROOT / 'benchmark_isc/runs/b112825_h200daily_r1/out/fluxes.2004-01-01.nc'))
    sl = slice(a.start_day, a.start_day + a.days)
    mm = np.array(r['OUT_RUNOFF'][sl, i0:i0 + 16, j0:j0 + 16] + r['OUT_BASEFLOW'][sl, i0:i0 + 16, j0:j0 + 16])
    mm = np.nan_to_num(mm.reshape(a.days, -1), nan=0.0)
    # inflow cells: the low channel cells of the column 4 from the east edge, around the lowest point
    dem = b['dem']
    r_in, c_in = [int(v) for v in b['inflow_rc'][0]]
    col = dem[:, c_in]
    win = col[max(r_in - 80, 0):r_in + 80]
    lo = float(np.percentile(win, 15))          # the river level (single-cell pits excluded)
    rows = [k for k in range(max(r_in - 80, 0), min(r_in + 80, ny)) if abs(col[k] - lo) <= 0.7]
    pts = [(k, c_in) for k in rows]
    q = np.linspace(a.q0, a.q1, a.days)
    registry.register('replay', lambda **k: Replay(mm, area, **k), 'test', replace=True)
    registry.register('inflow', lambda **k: Inflow(q, len(pts), **k), 'test', replace=True)
    cfg = dict(run=dict(device='cuda' if a.solver == 'acc_gpu' else 'cpu', log=False, duration_days=a.days),
               modules=dict(vic=dict(connector='replay'), bc=dict(connector='inflow'),
                            flood=dict(connector='flood', basin=a.basin, solver=a.solver,
                                       interval_s=DAY, inflow_points=pts, free_edges=[['W', 1e-4]],
                                       workdir=str(out / 'work'))),
               connections=[{'from': 'vic.runoff', 'to': 'flood.runoff', 'mapping': 'vic_to_flood'},
                            {'from': 'bc.inflow', 'to': 'flood.inflow'}],
               mappings=dict(vic_to_flood=dict(source='vic_cells', target='flood_cells', file=a.pairs)),
               ledger=dict(enabled=True, on_violation='reject'))
    t0 = time.time()
    m = TorchVIC.from_dict(cfg).initialize()
    t_init = time.time() - t0
    fl = m.connector('flood')
    walls = []
    t1 = time.time()
    while True:
        t = time.time()
        if not m.step():
            break
        walls.append(time.time() - t)
        rec = fl.volume_record()
        print(json.dumps(dict(day=len(walls), wall=round(walls[-1], 1), 
                              handed=rec['handed'], outflow=rec['outflow'], stored=fl.totals['stored'],
                              residual=rec['residual'], flooded_km2=float((fl.depth > 0.01).sum()) * 1e-2)), flush=True)
    total = time.time() - t1
    rec = fl.volume_record()
    res = dict(solver=a.solver, days=a.days, cells=ny * nx, vic_cells=len(area), inflow_cells=len(pts),
               init_s=round(t_init, 1), run_s=round(total, 1), 
               process_s=round(fl.timing['run_s'], 1), per_day_s=round(float(np.mean(walls)), 1),
               ledger_rows=len(m.ledger.balances), ledger_ok=all(r['ok'] for r in m.ledger.balances),
               coupled_ok=all(r['ok'] for r in m.ledger.coupled), **rec,
               vic_runoff_m3=float((mm * 1e-3 * area[None, :]).sum()), inflow_m3=float(q.sum() * DAY),
               mean_depth_m=float(fl.depth.mean()), max_depth_m=float(fl.depth.max()),
               flooded_km2=float((fl.depth > 0.01).sum()) * 1e-2)
    m.finalize(write=False)
    (out / 'result.json').write_text(json.dumps(res, indent=1))
    np.save(out / 'depth_end.npy', fl.depth.numpy().astype(np.float32))
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
