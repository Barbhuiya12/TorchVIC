"""TorchVIC flood engine against LISFLOOD-FP ACC (the external binary).

Cases: flat (closed, point source + rain), slope (plane, free east edge,
rain), basin (closed parabolic bowl, rain), guwahati (a sub-area of the
Brahmaputra/Guwahati 100 m basin with river point inflows, rain and a free
west edge). Each case runs LISFLOOD-FP acc_gpu (and acc_cpu with --cpu) and
the engine in its three schemes; prints one JSON line per comparison and
writes <out>/validate.json.

    python tools/flood/validate_engine.py --out DIR [--cases flat,slope] [--cpu]
"""
import argparse
import json
import os
import shutil
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torchvic.flood import FloodEngine  # noqa: E402

DATA = Path(os.environ.get('TORCHVIC_DATA_ROOT', 'data'))
BIN = os.environ.get('TORCHVIC_LISFLOOD_BIN') or os.environ.get('LISFLOOD_BIN') or str(
    DATA / 'static/upstream/lisflood-fp-8.2-zenodo/build_cuda/lisflood')
NCLIB = os.environ.get('LISFLOOD_NETCDF_LIB', '')


def write_asc(path, a, dx):
    a = np.asarray(a, float)
    ny, nx = a.shape
    with open(path, 'w') as f:
        f.write('ncols %d\nnrows %d\nxllcorner 0.0\nyllcorner 0.0\n'
                'cellsize %.17g\nNODATA_value -9999\n' % (nx, ny, dx))
        for row in a:
            f.write(' '.join('%.17g' % v for v in row) + '\n')


def read_asc(path):
    with open(path) as f:
        for _ in range(6):
            f.readline()
        a = np.loadtxt(f)
    return a


def write_rain(path, mm, T, dx):
    import netCDF4
    ny, nx = mm.shape
    d = netCDF4.Dataset(str(path), 'w')
    d.createDimension('time', 2)
    d.createDimension('x', nx)
    d.createDimension('y', ny)
    t = d.createVariable('time', 'f8', ('time',))
    t.units, t.axis = 'hour', 'T'
    x = d.createVariable('x', 'f8', ('x',))
    x.units, x.axis = 'm', 'X'
    y = d.createVariable('y', 'f8', ('y',))
    y.units, y.axis = 'm', 'Y'
    r = d.createVariable('rainfall_depth', 'f8', ('time', 'y', 'x'))
    r.units = 'mm'
    t[:] = [0.0, T / 3600.0]
    x[:] = (np.arange(nx) + 0.5) * dx
    y[:] = ((np.arange(ny) + 0.5) * dx)[::-1]
    r[0] = mm
    r[1] = 0.0
    d.close()


def lisflood_rate(mm, T):
    """The rate (m/s) LISFLOOD-FP derives from one rain level (rain.tpp:
    depth / (level spacing in h x 3600) / 1000)."""
    dt = (T / 3600.0 - 0.0) * 3600.0
    return (np.asarray(mm, float) / dt) / 1000.0


def run_lisflood(wd, case, solver, verbose=True):
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    z, dx, T = case['z'], case['dx'], case['T']
    write_asc(wd / 'dem.asc', z, dx)
    par = ['DEMfile dem.asc', 'resroot case', 'dirroot out',
           'sim_time %.17g' % T, 'initial_tstep %.17g' % case['max_dt'],
           'massint %.17g' % (T / 10), 'saveint %.17g' % T,
           'fpfric %.17g' % case['n'], 'elevoff', 'output_precision 15',
           'acceleration', 'voutput']
    if solver == 'acc_gpu':
        par.append('cuda')
    ny = z.shape[0]
    bci = []
    for (r, c), Q in zip(case['points'], case['Q']):
        bci.append('P %.17g %.17g QFIX %.17g' % ((c + 0.5) * dx,
                                                  (ny - 1 - r + 0.5) * dx,
                                                  Q / dx))
    for e, s in case['free'].items():
        L = (z.shape[0] if e in 'EW' else z.shape[1]) * dx
        bci.append('%s 0 %.17g FREE %.17g' % (e, L, s))
    if bci:
        (wd / 'case.bci').write_text('\n'.join(bci) + '\n')
        par.append('bcifile case.bci')
    if case.get('rain_mm') is not None:
        write_rain(wd / 'rain.nc', case['rain_mm'], T, dx)
        par.append('dynamicrainfile rain.nc')
    (wd / 'case.par').write_text('\n'.join(par) + '\n')
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = NCLIB + ':' + env.get('LD_LIBRARY_PATH', '')
    # The LISFLOOD-FP acc_gpu binary segfaults (rc=-11) in ~5% of runs, late in
    # the run, independent of the engine (flaky_flood log). A signal death is
    # retried (fresh output dir); a normal non-zero exit is a real failure.
    for attempt in range(6):
        shutil.rmtree(wd / 'out', ignore_errors=True)
        t0 = time.time()
        p = subprocess.run([BIN] + (['-v'] if verbose else []) + ['case.par'],
                           cwd=wd, env=env,
                           capture_output=True, text=True)
        wall = time.time() - t0
        if p.returncode >= 0:
            break
    if p.returncode != 0:
        raise RuntimeError('lisflood %s failed rc=%s: %s' % (solver, p.returncode, p.stdout[-800:]
                                                      + p.stderr[-800:]))
    out = wd / 'out'
    wds = sorted(out.glob('case-*.wd'),
                 key=lambda q: int(re.findall(r'-(\d+)\.wd$', q.name)[0]))
    h = read_asc(wds[-1])
    mx = read_asc(out / 'case.max')
    steps = [m for m in re.findall(r't=([\d.eE+-]+)\s+dt=([\d.eE+-]+)', p.stdout)]
    mass = np.loadtxt(out / 'case.mass', skiprows=1, ndmin=2)
    m = re.search(r'loop time\s*([\d.]+)', p.stdout, re.I)
    return dict(h=h, max=mx, wall=wall, nsteps=len(steps),
                loop_s=float(m.group(1)) if m else None,
                t_end=float(steps[-1][0]) + float(steps[-1][1]) if steps else None,
                mass_vol=float(mass[-1, 5]), mass_qin=float(mass[-1, 6]),
                mass_qout=float(mass[-1, 8]))


def run_engine(case, scheme, device, dtype=torch.float64, end='clip',
               backend='auto'):
    z = case['z']
    e = FloodEngine(z, case['dx'], case['n'], free_edges=case['free'],
                    inflow_points=case['points'], scheme=scheme,
                    max_dt=case['max_dt'], device=device, dtype=dtype,
                    block=64, end=end, backend=backend)
    rate = None
    if case.get('rain_mm') is not None:
        rate = lisflood_rate(case['rain_mm'], case['T'])
    t0 = time.time()
    o = e.run(case['T'], rate=rate, discharge=case['Q'] if case['points']
              else None)
    if device != 'cpu':
        torch.cuda.synchronize()
    wall = time.time() - t0
    return dict(h=e.h.double().cpu().numpy(),
                max=e.hmax.double().cpu().numpy(), wall=wall,
                ledger=o['ledger'], nsteps=o['ledger']['steps'])


# ---------------------------------------------------------------------------
def cases(names):
    out = {}
    if 'flat' in names:
        ny = nx = 64
        out['flat'] = dict(z=np.zeros((ny, nx)), dx=10.0, T=900.0, n=0.03,
                           max_dt=10.0, points=[(32, 32)], Q=[5.0], free={},
                           rain_mm=np.full((ny, nx), 10.0))
    if 'slope' in names:
        ny, nx = 40, 120
        x = np.arange(nx)[None, :] * np.ones((ny, 1))
        y = np.arange(ny)[:, None] * np.ones((1, nx))
        z = 5.0 - 0.002 * x * 10.0 + 0.05 * np.cos(y / 4.0)
        out['slope'] = dict(z=z, dx=10.0, T=1800.0, n=0.035, max_dt=10.0,
                            points=[], Q=[], free={'E': 0.002},
                            rain_mm=np.full((ny, nx), 25.0))
    if 'basin' in names:
        ny = nx = 80
        y, x = np.mgrid[0:ny, 0:nx]
        r2 = ((x - 39.5) ** 2 + (y - 41.2) ** 2) * 400.0
        z = 1e-5 * r2 + 0.1 * np.sin(x / 5.0) * np.cos(y / 7.0)
        out['basin'] = dict(z=z, dx=20.0, T=3600.0, n=0.03, max_dt=10.0,
                            points=[], Q=[], free={},
                            rain_mm=np.full((ny, nx), 60.0))
    if 'bowl_steep' in names:
        # slopes up to 0.16: beyond the local-inertial range (Froude ~ 1);
        # a stress test of the wet/dry limiter and of mass conservation
        ny = nx = 80
        y, x = np.mgrid[0:ny, 0:nx]
        r2 = ((x - 39.5) ** 2 + (y - 41.2) ** 2) * 400.0
        z = 1e-4 * r2 + 0.3 * np.sin(x / 5.0) * np.cos(y / 7.0)
        out['bowl_steep'] = dict(z=z, dx=20.0, T=3600.0, n=0.03,
                                 max_dt=10.0, points=[], Q=[], free={},
                                 rain_mm=np.full((ny, nx), 60.0))
    if 'guwahati' in names:
        b = np.load(DATA / 'static/flood_basins/'
                    'brahmaputra_guwahati_100m.npz')
        dem = b['dem']
        r_in, c_in = [int(v) for v in b['inflow_rc'][0]]
        ny, nx = dem.shape
        r0, r1 = max(r_in - 128, 0), min(r_in + 128, ny)
        c0, c1 = nx - 256, nx
        z = dem[r0:r1, c0:c1].copy()
        col = dem[:, c_in]
        win = col[max(r_in - 80, 0):r_in + 80]
        lo = float(np.percentile(win, 15))
        rows = [k for k in range(max(r_in - 80, 0), min(r_in + 80, ny))
                if abs(col[k] - lo) <= 0.7 and r0 <= k < r1]
        pts = [(k - r0, c_in - c0) for k in rows]
        Q = [2000.0 / len(pts)] * len(pts)
        out['guwahati'] = dict(z=z, dx=float(b['dx']), T=6 * 3600.0, n=0.03,
                               max_dt=10.0, points=pts, Q=Q,
                               free={'W': 1e-4},
                               rain_mm=np.full(z.shape, 20.0))
    return out


def compare(a, b, case):
    dh = np.abs(a['h'] - b['h'])
    dm = np.abs(a['max'] - b['max'])
    A = case['dx'] ** 2
    va, vb = float(a['h'].sum()) * A, float(b['h'].sum()) * A
    return dict(max_abs_h=float(dh.max()), rms_h=float(np.sqrt((dh ** 2).mean())),
                max_abs_max=float(dm.max()), h_peak=float(np.max(b['h'])),
                vol_a=va, vol_b=vb, rel_vol=(va - vb) / max(abs(vb), 1e-30),
                wet_mismatch=int(((a['h'] > 1e-3) != (b['h'] > 1e-3)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--cases', default='flat,slope,basin,bowl_steep,guwahati')
    ap.add_argument('--cpu', action='store_true', help='also LISFLOOD acc_cpu')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for name, c in cases(a.cases.split(',')).items():
        ref = {}
        sols = (['acc_gpu'] if a.device != 'cpu' else []) + (['acc_cpu'] if a.cpu else [])
        for s in sols:
            ref[s] = run_lisflood(out / name / s, c, s)
        eng = {}
        for sch in ('lisflood_gpu', 'lisflood_cpu', 'torchvic'):
            eng[sch] = run_engine(c, sch, a.device)
        eng['torchvic_f32'] = run_engine(c, 'torchvic', a.device, torch.float32)
        eng['torchvic_os'] = run_engine(c, 'torchvic', a.device, end='overshoot')
        if a.device != 'cpu':
            # round-off sensitivity of the case: the same scheme on the CPU
            eng['lisflood_gpu@cpu'] = run_engine(c, 'lisflood_gpu', 'cpu')
            eng['torchvic@cpu'] = run_engine(c, 'torchvic', 'cpu')
            eng['torchvic_torch'] = run_engine(c, 'torchvic', a.device,
                                               backend='torch')
        pairs = [('lisflood_gpu', 'acc_gpu'), ('lisflood_cpu', 'acc_cpu'),
                 ('torchvic', 'acc_gpu'), ('torchvic', 'acc_cpu'),
                 ('torchvic_os', 'acc_cpu'), ('torchvic_os', 'acc_gpu'),
                 ('torchvic_f32', 'torchvic'),
                 ('lisflood_gpu@cpu', 'lisflood_gpu'),
                 ('torchvic@cpu', 'torchvic'),
                 ('torchvic_torch', 'torchvic')]
        for x, y in pairs:
            A = eng.get(x)
            B = ref.get(y) if y in ref else eng.get(y)
            if A is None or B is None:
                continue
            r = dict(case=name, engine=x, reference=y, cells=int(c['z'].size),
                     steps_engine=A['nsteps'], steps_ref=B['nsteps'],
                     wall_engine=round(A['wall'], 3), wall_ref=round(B['wall'], 3),
                     **compare(A, B, c))
            if 'ledger' in A:
                r['ledger_rel_residual'] = A['ledger']['rel_residual']
                r['ledger_clamp'] = A['ledger']['clamp']
            if y in ref:
                r['ref_t_end'] = B['t_end']
                r['ref_mass_vol'] = B['mass_vol']
                r['ref_mass_qout'] = B['mass_qout']
            if 'ledger' in A:
                r['engine_t_end'] = A['ledger']['t_end']
                r['engine_outflow'] = A['ledger']['outflow']
            print(json.dumps(r), flush=True)
            res.append(r)
    (out / 'validate.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
