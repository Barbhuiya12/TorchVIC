"""Brahmaputra/Guwahati 100 m basin (1.10 M cells), daily intervals: the
TorchVIC flood engine with the inputs of the LISFLOOD-FP connector
benchmark (<data root>/verification/conn_flood/bench_gpu_31d_river: stored
TorchVIC runoff of July 2004 mapped by the conservative VIC->grid pairs,
77 river inflow cells with 10000 -> 28000 m3/s, free west edge slope 1e-4,
n = 0.03). Momentum is kept across days. Prints one JSON line per day and a
RESULT line.

    python tools/flood/bench_engine.py --days 31 --scheme torchvic \
        --max-dt 10 [--condition fill:5] [--dtype float32] --out DIR
"""
import os
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torchvic.flood import FloodEngine, dem as D  # noqa: E402

ROOT = Path(os.environ.get('TORCHVIC_DATA_ROOT', 'data'))
DAY = 86400.0


def basin_inputs(days, start_day=182, q0=10000.0, q1=28000.0):
    import netCDF4
    b = np.load(ROOT / 'static/flood_basins/brahmaputra_guwahati_100m.npz')
    dem = b['dem']
    ny, nx = dem.shape
    dom = netCDF4.Dataset(str(ROOT / 'vic_params_isc/domain.nc'))
    lat, lon = dom['lat'][:], dom['lon'][:]
    i0 = int(np.argmin(abs(lat - (b['lat0'] + 0.03125))))
    j0 = int(np.argmin(abs(lon - (b['lon0'] + 0.03125))))
    area = np.array(dom['area'][i0:i0 + 16, j0:j0 + 16]
                    * dom['frac'][i0:i0 + 16, j0:j0 + 16]).reshape(-1)
    r = netCDF4.Dataset(str(ROOT / 'benchmark_isc/runs/b112825_h200daily_r1/'
                            'out/fluxes.2004-01-01.nc'))
    sl = slice(start_day, start_day + days)
    mm = np.array(r['OUT_RUNOFF'][sl, i0:i0 + 16, j0:j0 + 16]
                  + r['OUT_BASEFLOW'][sl, i0:i0 + 16, j0:j0 + 16])
    mm = np.nan_to_num(mm.reshape(days, -1), nan=0.0)
    pairs = np.loadtxt(ROOT / 'static/flood_basins/'
                       'brahmaputra_guwahati_100m_vic_pairs.csv',
                       delimiter=',', skiprows=1)
    src, tgt, fr = pairs[:, 0].astype(int), pairs[:, 1].astype(int), pairs[:, 2]
    vic_m3 = mm * 1e-3 * area[None, :]                       # [day, vic]
    cell_m3 = np.stack([np.bincount(tgt, weights=v[src] * fr,
                                    minlength=ny * nx) for v in vic_m3])
    r_in, c_in = [int(v) for v in b['inflow_rc'][0]]
    col = dem[:, c_in]
    win = col[max(r_in - 80, 0):r_in + 80]
    lo = float(np.percentile(win, 15))
    rows = [k for k in range(max(r_in - 80, 0), min(r_in + 80, ny))
            if abs(col[k] - lo) <= 0.7]
    pts = [(k, c_in) for k in rows]
    q = np.linspace(q0, q1, days)
    return dict(b=b, dem=dem, cell_m3=cell_m3, vic_m3=vic_m3.sum(1), pts=pts,
                q=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=31)
    ap.add_argument('--scheme', default='torchvic')
    ap.add_argument('--max-dt', type=float, default=10.0)
    ap.add_argument('--dtype', default='float64')
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--condition', default='',
                    help='fill:MIN_DEPTH or breach:MIN_DEPTH (m)')
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--no-graphs', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--lisflood-day1', type=float, default=None,
                    metavar='INITIAL_TSTEP',
                    help='also run LISFLOOD-FP acc_gpu for day 1 with this '
                    'initial_tstep and compare the day-1 depth')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    inp = basin_inputs(a.days)
    dem = inp['dem']
    cond = None
    if a.condition:
        meth, md = a.condition.split(':')
        tc = time.time()
        dem, info = D.condition(dem, meth, float(md), edges='NESW')
        cond = dict(method=meth, min_depth=float(md), seconds=round(time.time() - tc, 1), **info)
        print('CONDITION ' + json.dumps(cond), flush=True)
    dtype = getattr(torch, a.dtype)
    e = FloodEngine(dem, float(inp['b']['dx']), 0.03, free_edges={'W': 1e-4},
                    inflow_points=inp['pts'], scheme=a.scheme,
                    max_dt=a.max_dt, device=a.device, dtype=dtype,
                    block=a.block, compile=not a.no_compile,
                    graphs=not a.no_graphs)
    t_init = time.time() - t0
    walls, rows = [], []
    npt = len(inp['pts'])
    t1 = time.time()
    for d in range(a.days):
        tw = time.time()
        o = e.run(DAY, runoff=inp['cell_m3'][d],
                  inflow=np.full(npt, inp['q'][d] * DAY / npt))
        if a.device != 'cpu':
            torch.cuda.synchronize()
        walls.append(time.time() - tw)
        L = o['ledger']
        h = e.h
        row = dict(day=d + 1, wall=round(walls[-1], 2), steps=L['steps'],
                   dt_cfl_min=L['dt_cfl_min'], dt_max=L['dt_max'],
                   rain=L['rain'], inflow=L['inflow'], outflow=L['outflow'],
                   stored=L['stored1'], clamp=L['clamp'],
                   rel_residual=L['rel_residual'],
                   max_depth=float(h.max()),
                   flooded_km2=float((h > 0.01).sum()) * 1e-2)
        rows.append(row)
        if d == 0:
            np.save(out / 'depth_day1.npy', e.h.double().cpu().numpy())
        print(json.dumps(row), flush=True)
    total = time.time() - t1
    lis = None
    if a.lisflood_day1 is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import validate_engine as V
        dx = float(inp['b']['dx'])
        case = dict(z=dem, dx=dx, T=DAY, n=0.03, max_dt=a.lisflood_day1,
                    points=inp['pts'], Q=[inp['q'][0] / npt] * npt,
                    free={'W': 1e-4},
                    rain_mm=inp['cell_m3'][0].reshape(dem.shape)
                    / (dx * dx) * 1000.0)
        r = V.run_lisflood(out / 'lisflood_day1', case, 'acc_gpu',
                           verbose=False)
        h1 = np.load(out / 'depth_day1.npy')
        dh = np.abs(r['h'] - h1)
        lis = dict(initial_tstep=a.lisflood_day1, wall_s=round(r['wall'], 1),
                   loop_s=r['loop_s'], max_abs_h=float(dh.max()),
                   rms_h=float(np.sqrt((dh ** 2).mean())),
                   vol_lisflood=float(r['h'].sum()) * dx * dx,
                   vol_engine=float(h1.sum()) * dx * dx,
                   wet_mismatch=int(((r['h'] > 1e-3) != (h1 > 1e-3)).sum()))
        print('LISFLOOD ' + json.dumps(lis), flush=True)
    T = e.totals
    handed = T['rain'] + T['inflow']
    dS = e.storage() - T['stored_start']
    res = dict(scheme=a.scheme, dtype=a.dtype, max_dt=a.max_dt,
               device=torch.cuda.get_device_name() if a.device != 'cpu' else 'cpu',
               days=a.days, cells=int(dem.size), inflow_cells=npt,
               condition=cond, lisflood_day1=lis, init_s=round(t_init, 1), run_s=round(total, 1),
               first_day_s=round(walls[0], 1),
               per_day_s=round(float(np.mean(walls[1:] if len(walls) > 1 else walls)), 2),
               steps=T['steps'], us_per_step=1e6 * total / max(T['steps'], 1),
               handed=handed, vic_runoff_m3=float(inp['vic_m3'].sum()),
               inflow_m3=float(inp['q'].sum() * DAY), outflow=T['outflow'],
               stored_change=dS, clamp=T['clamp'],
               residual=dS - (handed - T['outflow'] + T['clamp']),
               rel_residual=(dS - (handed - T['outflow'] + T['clamp'])) / handed,
               max_rel_residual_day=max(abs(r['rel_residual']) for r in rows),
               max_depth_m=float(e.h.max()),
               flooded_km2=float((e.h > 0.01).sum()) * 1e-2)
    (out / 'result.json').write_text(json.dumps(res, indent=1))
    np.save(out / 'depth_end.npy', e.h.double().cpu().numpy().astype(np.float32))
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
