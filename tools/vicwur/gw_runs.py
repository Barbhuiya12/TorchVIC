#!/usr/bin/env python
"""Reference runs of the VIC-WUR 3.0 groundwater coupling on R1.

  gw_runs.py static          write <data root>/vicwur/R1/modflow.nc (the
                             VIC-WUR 3.0 MODFLOW static grids of R1, synthetic)
  gw_runs.py gen             write the run directories and sbatch files
  gw_runs.py ref G11         MF6 reference of G11 from its C outputs
  gw_runs.py loop            the VIC-WUR 3.0 coupled loop of G11c (inside sbatch)
  gw_runs.py oracle RUN      fixed-mode daily oracle (the TorchVIC replay's
                             package inputs through MF6), first 365 days

Runs (<data root>/verification/vicwur_4/gruns/R1/<run>), all by the dump build
<data root>/build/vicwur3_dump (VICWUR_DUMP), OUT_RUNOFF, OUT_BASEFLOW,
OUT_GWRECHARGE, OUT_RECHARGE (and OUT_DISCHARGE) written as OUT_TYPE_DOUBLE
(output precision only):

  G11   the G-run of tools/vicwur/gruns.py: GWM on, VIC only (no routing).
        One continuous C run. Its OUT_GWRECHARGE drives the MODFLOW model
        month by month offline (`ref`): VIC-WUR 3.0's MODFLOW with nothing to
        hand the baseflow to and no state edit fed back (POC-like for VIC).
  G11c  G11 + routing (n=1) + water use (MUN, local) + non-renewable: the
        whole VIC-WUR 3.0 coupled loop
        (plugins/modflow/vic_global_online_full.py): per calendar month the
        C runs the month from the last month's state with the DISCHARGE
        forcing of the MODFLOW baseflow (ROUTING_FORCE), then VIC-WUR 3.0's
        own mf_run/config_module (tools/vicwur/mf_w3.py, MF6 6.7.0) runs the
        month on its OUT_GWRECHARGE and mean OUT_DISCHARGE, writes
        -(RIV + DRN) into the next month's forcing record and adds the EVT
        capillary rise to the state file (FOC, vic_runner.py:267-321). The
        driver bugs (the 1/days factor, the int truncation, Cv^2, band 0
        only) are fixed here as in TorchVIC; each month's dump is appended
        to dump.bin.

Static grids of R1 (synthetic: R1 has no MODFLOW data; values chosen to
reach every branch of config_module.py): a small river on c0 and c4
(W <= 30 m: the drain-like RIV at the DEM top), wide rivers on c1-c3 (Manning
stage), a wide flat river on c5 (no record), CHD (ibound 2) on c5, an
aquifer depth NaN (-> 200 m) on c5, confining-layer K on c2/c3, a KQ3 below
the floor and one above 1, Sy below 0.11, a negative Z0 and min_dem, EVT on
c1-c4 (c0 NaN, c5 rate 0).
"""
import argparse
import calendar
import datetime
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gruns as GR                                          # noqa: E402

ROOT = GR.ROOT
OUT = ROOT + '/verification/vicwur_4'
RUNS = OUT + '/gruns/R1'
STATIC = GR.R1 + '/modflow.nc'
DUMP_EXE = ROOT + '/build/vicwur3_dump/vic_image.exe'
VENV_MF = ROOT + '/verification/design_gwcrop/venv/bin/python'
PY = sys.executable
LOG = OUT + '/run.log'
DOUBLE = ('OUT_RUNOFF', 'OUT_BASEFLOW', 'OUT_GWRECHARGE', 'OUT_RECHARGE',
          'OUT_DISCHARGE')
G11C = GR.runs()['G11c']


def log(msg):
    with open(LOG, 'a') as f:
        f.write('%s %s\n' % (datetime.datetime.now().isoformat(
            timespec='seconds'), msg))


# ------------------------------------------------------------------ static
_STATIC = {}


def static_values():
    """Per-cell values c0..c5 (row-major on the 2 x 3 grid); a copy of the
    values read once."""
    if not _STATIC:
        _STATIC.update(_static_values())
    return {k: v.copy() for k, v in _STATIC.items()}


def _static_values():
    import netCDF4 as nc
    with nc.Dataset(GR.R1 + '/domain.nc') as d:
        area = np.asarray(d['area'][:], float).ravel()
    with nc.Dataset(GR.R1 + '/params.nc') as p:
        depth = np.asarray(p['depth'][:], float).reshape(3, -1)
        bulk = np.asarray(p['bulk_density'][:], float).reshape(3, -1)
        dens = np.asarray(p['soil_density'][:], float).reshape(3, -1)
    nan = np.nan
    dem = np.array([60., 45., 30., 15., 40., 8.])
    v = dict(
        dem=dem,
        aqdepth=np.array([80., 100., 120., 150., 60., nan]),
        ksat_log=np.array([-11.5, -11.0, -10.8, -10.5, -12.0, -10.5]),
        ksat_l1_conf_log=np.array([nan, nan, -14.0, -13.0, nan, nan]),
        ksat_l2_conf_log=np.array([nan, nan, nan, -10.0, nan, nan]),
        cellarea=area,
        qbank=np.array([10., 60., 150., 300., 25., 100.]),
        riv_slope=np.array([3e-3, 1.5e-3, 8e-4, 3e-4, 2e-3, 2e-5]),
        z0=np.array([57., 42., 27., 12., 37., -1.]),
        min_dem=np.array([54., 39., 24., 9., 34., -2.]),
        kq3=np.array([0.05, 0.02, 0.01, 5e-5, 0.03, 2.0]),
        sy=np.array([0.05, 0.15, 0.2, 0.25, 0.08, 0.3]),
        landmask=np.ones(6),
        ibound=np.array([1, 1, 1, 1, 1, 2], float),
        ldd=np.array([1, 1, 1, 1, 1, 2], float),
        head0=np.stack([dem - 2.0, dem - 2.3]),
        cpsurface=dem - 1.0,
        cprate=np.array([0.002, 0.002, 0.003, 0.002, 0.0025, 0.0]),
        cpdistinct=np.array([3.0, 3.0, 3.5, 3.0, 2.5, 3.0]),
        max_moist=depth * (1 - bulk / dens) * 1000.)
    v['cprate'][0] = nan
    v['head0'][:, 5] = 0.0
    return v


def static():
    import netCDF4 as nc
    v = static_values()
    with nc.Dataset(GR.R1 + '/domain.nc') as d:
        lat, lon = np.asarray(d['lat'][:]), np.asarray(d['lon'][:])
    with nc.Dataset(STATIC, 'w') as ds:
        ds.title = ('W3 MODFLOW static grids of R1 (synthetic), '
                    'tools/vicwur/gw_runs.py static')
        ds.createDimension('layer', 2)
        ds.createDimension('nlayer', 3)
        ds.createDimension('lat', len(lat))
        ds.createDimension('lon', len(lon))
        ds.createVariable('lat', 'f8', ('lat',))[:] = lat
        ds.createVariable('lon', 'f8', ('lon',))[:] = lon
        for k, a in v.items():
            if a.ndim == 1:
                x = ds.createVariable(k, 'f8', ('lat', 'lon'))
                x[:] = a.reshape(len(lat), len(lon))
            else:
                dim = 'layer' if a.shape[0] == 2 else 'nlayer'
                x = ds.createVariable(k, 'f8', (dim, 'lat', 'lon'))
                x[:] = a.reshape(a.shape[0], len(lat), len(lon))
    log('static: wrote %s' % STATIC)


def static_grid(pad=True):
    """The static grids (R, C) for mf_w3.py; `pad` adds one ocean column
    (land mask NaN, ibound 0, ldd NaN) as VIC-WUR 3.0's global grid has, so
    that np.unique(ldd)[:-1] (config_module.py:383) drops the no-data id."""
    v = static_values()
    out = {}
    for k, a in v.items():
        g = a.reshape(a.shape[:-1] + (2, 3))
        if pad:
            fill = {'ibound': 0.0, 'cellarea': float(v['cellarea'][0])}.get(
                k, np.nan)
            g = np.concatenate([g, np.full(g.shape[:-1] + (1,), fill)], -1)
        out[k] = g
    out['ibound'] = out['ibound'].astype(np.int64)
    return out


# ------------------------------------------------------------------ global
def global_text(run, c, rd, start=GR.START, end=GR.END, init_state=None,
                force_prefix=None):
    txt, cons = GR.global_text(run, c, 'R1', 'W3', rd + '/W3', start=start,
                               end=end, init_state=init_state)
    for v in DOUBLE:
        if 'OUTVAR %s\n' % v in txt:
            txt = txt.replace('OUTVAR %s\n' % v,
                              'OUTVAR %s * OUT_TYPE_DOUBLE * *\n' % v)
        elif v in ('OUT_GWRECHARGE', 'OUT_RECHARGE'):
            txt += 'OUTVAR %s * OUT_TYPE_DOUBLE * *\n' % v
    if force_prefix:
        txt = txt.replace('DECOMPOSITION BASIN\n', (
            'DECOMPOSITION BASIN\nROUTING_FORCE TRUE\n'
            'PLUGIN_FORCE_TYPE DISCHARGE discharge_mf MONTH %s\n')
            % force_prefix)
    return txt, cons


SB = """#!/bin/bash
#SBATCH -J w4_{run}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 04:00:00
#SBATCH -o {rd}/slurm-%j.out
export LD_LIBRARY_PATH={envnc}/lib
export HDF5_USE_FILE_LOCKING=FALSE
cd {rd}
{body}
"""


def gen():
    R = GR.runs()
    c = dict(GR.BASE, **R['G11'])
    rd = RUNS + '/G11'
    for sub in ('out', 'state', 'log'):
        os.makedirs('%s/W3/%s' % (rd, sub), exist_ok=True)
    txt, _ = global_text('G11', c, rd)
    open(rd + '/global_W3.txt', 'w').write(txt)
    body = ('date -Is > {rd}/W3/time.start\n'
            'VICWUR_DUMP={rd}/dump.bin /usr/bin/env mpirun --oversubscribe '
            '-np 1 {exe} -g {rd}/global_W3.txt > {rd}/W3/log/stdout.log 2>&1\n'
            'echo $? > {rd}/W3/rc\n'
            'date -Is > {rd}/W3/time.end\n'
            '{py} {me} ref G11 > {rd}/ref.log 2>&1\n'
            'echo $? > {rd}/ref.rc\n').format(rd=rd, exe=DUMP_EXE, py=PY,
                                              me=os.path.abspath(__file__))
    open(rd + '/run.sbatch', 'w').write(SB.format(
        run='G11', rd=rd, envnc=GR.ENVNC, body=body))
    rd = RUNS + '/G11c'
    for sub in ('out', 'state', 'log', 'force', 'months', 'mf'):
        os.makedirs('%s/W3/%s' % (rd, sub), exist_ok=True)
    c = dict(GR.BASE, **G11C)
    txt, _ = global_text('G11c', c, rd,
                         force_prefix=rd + '/W3/force/discharge_mf_')
    open(rd + '/global_W3.txt', 'w').write(txt)
    body = ('{py} {me} loop > {rd}/loop.log 2>&1\n'
            'echo $? > {rd}/W3/rc\n').format(py=PY, rd=rd,
                                             me=os.path.abspath(__file__))
    open(rd + '/run.sbatch', 'w').write(SB.format(
        run='G11c', rd=rd, envnc=GR.ENVNC, body=body))
    log('gen: G11, G11c under %s' % RUNS)


# ------------------------------------------------------------------ helpers
def months(start=GR.START, end=GR.END):
    y, m = start[0], start[1]
    out = []
    while (y, m) <= (end[0], end[1]):
        out.append((y, m, calendar.monthrange(y, m)[1]))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def read_daily(files, var):
    """[T, 2, 3] float64 of a history variable over sorted files."""
    import netCDF4 as nc
    parts = []
    for p in files:
        with nc.Dataset(p) as ds:
            ds.set_auto_mask(False)
            parts.append(np.asarray(ds[var][:], np.float64))
    return np.concatenate(parts)


def mf_call(inp, out):
    r = subprocess.run([VENV_MF, os.path.join(HERE, 'mf_w3.py'), 'run', inp,
                        out], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError('mf_w3 failed:\n' + r.stdout[-3000:]
                           + r.stderr[-3000:])


def mf_inputs(ws, periods, gw, q, mode='exact', head0=None, wells=None,
              foc=True):
    st = static_grid()
    pad = lambda a: np.concatenate(
        [a, np.zeros(a.shape[:-1] + (1,))], -1)
    d = dict(st, periods=np.asarray(periods), gwrecharge_mm=pad(gw),
             discharge=pad(q), mode=mode, foc=foc, well_layer=1,
             xres=0.0625, yres=0.0625, ws=ws)
    if head0 is not None:
        d['head0_run'] = head0
    if wells is not None:
        d['wells'] = pad(wells)
    return d


# ------------------------------------------------------------------ G11 ref
def ref(run):
    rd = RUNS + '/' + run
    files = sorted(glob.glob(rd + '/W3/out/fluxes.*.nc'))
    gw = read_daily(files, 'OUT_GWRECHARGE')
    ms = months()
    T = sum(n for _, _, n in ms)
    assert gw.shape[0] == T, (gw.shape, T)
    per, g, q = [], [], []
    t = 0
    for y, m, n in ms:
        per.append((y, m, 1, n))
        g.append(gw[t:t + n].sum(0))
        q.append(np.zeros((2, 3)))
        t += n
    # FOC (capillary rise on) and POC (off: nothing goes back to VIC, so a
    # coupled run with VIC in the loop meets the continuous C run)
    for foc, name in ((True, 'mf_ref'), (False, 'mf_ref_poc')):
        inp = rd + '/%s_in.npz' % name
        np.savez(inp, **mf_inputs(rd + '/mf_ws', per, np.stack(g),
                                  np.stack(q), foc=foc))
        mf_call(inp, rd + '/%s.npz' % name)
        log('ref %s: %d MODFLOW months -> %s/%s.npz' % (run, len(per), rd,
                                                        name))


# ------------------------------------------------------------------ G11c
def w3_state_edit(sm, cv, live, cpr, max_moist, layer=2):
    """vic_runner.update_statefile (:267-321) on sm [veg, band, layer, cell]
    with the fixes (cpr mm added to each live tile, not cpr * Cv; every
    band, not band 0; no int truncation): numpy, in VIC-WUR 3.0's loop
    order. live [veg, band, cell]. Returns the new sm."""
    new = sm.copy()
    new[:, :, layer] = sm[:, :, layer] + cpr[None, None] * live
    for ly in range(layer, -1, -1):
        tot = np.nansum(new[:, :, ly] * cv[:, None], axis=0)   # [band, cell]
        sat = tot > max_moist[ly][None]
        if sat.any():
            exc = tot - max_moist[ly][None]
            exc[exc < 0] = 0
            for i in range(cv.shape[0]):
                on = sat & live[i]
                new[i, :, ly][on] = np.broadcast_to(max_moist[ly][None],
                                                    sat.shape)[on]
                if ly > 0:
                    new[i, :, ly - 1][live[i] & sat] += exc[live[i] & sat]
        else:
            break
    return new


def write_force(rd, year, records):
    """The DISCHARGE plugin forcing of `year` (12 monthly records, m3 s-1,
    vic_runner.create_extra_forcing_file)."""
    import netCDF4 as nc
    with nc.Dataset(GR.R1 + '/domain.nc') as d:
        lat, lon = np.asarray(d['lat'][:]), np.asarray(d['lon'][:])
    p = rd + '/W3/force/discharge_mf_%4d.nc' % year
    with nc.Dataset(p, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', 12)
        ds.createDimension('lat', len(lat))
        ds.createDimension('lon', len(lon))
        t = ds.createVariable('time', 'f8', ('time',))
        # VIC-WUR 3.0 writes 'days since 1979-01-01', standard; the R1 run's
        # clock is proleptic_gregorian and the C refuses another calendar
        t[:] = [(datetime.datetime(year, m, 1)
                 - datetime.datetime(year, 1, 1)).days for m in range(1, 13)]
        t.units = 'days since %d-01-01' % year
        t.calendar = 'proleptic_gregorian'
        ds.createVariable('lat', 'f8', ('lat',))[:] = lat
        ds.createVariable('lon', 'f8', ('lon',))[:] = lon
        v = ds.createVariable('discharge_mf', 'f8', ('time', 'lat', 'lon'),
                              fill_value=-9999.0)
        v[:] = records


def loop():
    """The VIC-WUR 3.0 coupled loop on G11c (vic_global_online_full.py:41-84)."""
    import netCDF4 as nc
    rd = RUNS + '/G11c'
    c = dict(GR.BASE, **G11C)
    prefix = rd + '/W3/force/discharge_mf_'
    ms = months()
    force = {}                                    # year -> [12, 2, 3]
    force[ms[0][0]] = np.zeros((12, 2, 3))
    write_force(rd, ms[0][0], force[ms[0][0]])
    with nc.Dataset(GR.R1 + '/params.nc') as p:
        cv = np.asarray(p['Cv'][:], float).reshape(-1, 6)
        af = np.asarray(p['AreaFract'][:], float).reshape(-1, 6)
    live = (cv > 0)[:, None, :] & (af > 0)[None, :, :]
    mx = static_values()['max_moist']
    head = None
    state = None
    open(rd + '/dump.bin', 'wb').close()
    rec = dict(cpr=[], gw=[], q=[], force=[])
    env = dict(os.environ, LD_LIBRARY_PATH=GR.ENVNC + '/lib',
               HDF5_USE_FILE_LOCKING='FALSE')
    for k, (y, m, n) in enumerate(ms):
        g = '%s/W3/months/global_%04d%02d.txt' % (rd, y, m)
        txt, _ = global_text('G11c', c, rd, start=(y, m, 1), end=(y, m, n),
                             init_state=state, force_prefix=prefix)
        # a yearly history file cannot start on 1 March after Feb 29
        # (vic_time.c:874); VIC-WUR 3.0's runs are monthly anyway
        txt = txt.replace('HISTFREQ NYEARS 1', 'HISTFREQ NMONTHS 1')
        open(g, 'w').write(txt)
        dump = '%s/W3/months/dump_%04d%02d.bin' % (rd, y, m)
        env['VICWUR_DUMP'] = dump
        r = subprocess.run(['/usr/bin/env', 'mpirun', '--oversubscribe',
                            '-np', '1', GR.BUILDS['W3'].replace(
                                'vicwur3/', 'vicwur3_dump/'), '-g', g],
                           env=env, capture_output=True, text=True)
        open('%s/W3/log/stdout_%04d%02d.log' % (rd, y, m), 'w').write(
            r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError('C run %04d-%02d failed' % (y, m))
        with open(rd + '/dump.bin', 'ab') as f, open(dump, 'rb') as s:
            f.write(s.read())
        os.remove(dump)
        out = glob.glob('%s/W3/out/fluxes.%04d-%02d*.nc' % (rd, y, m))[0]
        gw = read_daily([out], 'OUT_GWRECHARGE')
        q = read_daily([out], 'OUT_DISCHARGE')
        assert gw.shape[0] == n
        inp = '%s/W3/mf/in_%04d%02d.npz' % (rd, y, m)
        oo = '%s/W3/mf/out_%04d%02d.npz' % (rd, y, m)
        np.savez(inp, **mf_inputs(rd + '/W3/mf/ws', [(y, m, 1, n)],
                                  gw.sum(0)[None], q.mean(0)[None],
                                  head0=head))
        mf_call(inp, oo)
        o = np.load(oo)
        head = o['head'][0]
        # mf_run.get_baseflow_array without the 1/days factor (:319)
        qrd = (o['q_RIV'][0, 0] + o['q_DRN'][0, 0])[:, :3]
        frc = qrd / 86400 * -1
        frc = np.where(np.abs(frc) > 1e8, 0, frc)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        if ny not in force:
            force[ny] = np.zeros((12, 2, 3))
        force[ny][nm - 1] = frc
        write_force(rd, ny, force[ny])
        # FOC: the EVT outflow volume over the month, mm, into the state
        cpr = (-o['q_EVT'][0, 0][:, :3] * n / static_grid(False)['cellarea']
               * 1000).ravel()
        state = '%s/W3/state/states.%04d%02d01_00000.nc' % (rd, ny, nm)
        if k + 1 < len(ms):
            with nc.Dataset(state, 'a') as ds:
                v = ds['STATE_SOIL_MOISTURE']       # veg, band, layer, y, x
                sm = np.asarray(v[:], float)
                shp = sm.shape
                sm = sm.reshape(shp[:3] + (6,))
                new = w3_state_edit(sm, cv, live, cpr, mx)
                v[:] = new.reshape(shp)
        rec['cpr'].append(cpr)
        rec['gw'].append(gw.sum(0).ravel())
        rec['q'].append(q.mean(0).ravel())
        rec['force'].append(frc.ravel())
        log('loop G11c %04d-%02d: C + MF6 done, head %.4f..%.4f m' % (
            y, m, head[:, :, :3].min(), head[:, :, :3].max()))
    np.savez(rd + '/loop.npz', **{k: np.stack(v) for k, v in rec.items()})
    log('loop G11c: %d months done' % len(ms))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['static', 'gen', 'ref', 'loop'])
    ap.add_argument('run', nargs='?')
    a = ap.parse_args()
    if a.cmd == 'static':
        static()
    elif a.cmd == 'gen':
        gen()
    elif a.cmd == 'ref':
        ref(a.run)
    else:
        loop()
