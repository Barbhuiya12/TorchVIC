"""Compare one matrix row's VIC and TorchVIC per-step histories.

Metrics, per output variable (per layer for layered ones), over the active
cells of the domain (VIC's own mask):

  total      the period total of a summed flux (VIC's default AGG_TYPE SUM)
             or the period mean of anything else, per cell; `rel_total` is
             the area-weighted domain value's relative difference, and
             `rel_cell` the largest per-cell one. The denominator is the
             larger of |VIC's value| and the variable's mean absolute size
             (so a flux that changes sign, such as the ground heat flux,
             is not divided by its near-zero mean); temperatures are
             compared in kelvin.
  max_abs    the largest |TorchVIC - VIC| at any cell and step
  rmse       over all cells and steps
  first      the first step (and its date and cell) at which
             |d| > 1e-6 (|VIC| + mean|VIC|)

Grade of a variable: pass if max(rel_total, rel_cell) <= 1e-4 (0.01 %),
warn if <= 1e-2, fail otherwise. OUT_WATER_ERROR and OUT_ENERGY_ERROR are
closure diagnostics and are not graded as variables (TorchVIC's
OUT_ENERGY_ERROR is by default the surface-balance residual at the root,
VIC's is put_data.c's sum of fluxes; they are different quantities. In
CLOSE_ENERGY, SPATIAL_SNOW and daily-step runs with snow sub-steps TorchVIC
itself writes put_data's expression of its own rows there).

Closure (TorchVIC; VIC's for reference), per cell per step:
  water   P - E - R - B - d(sum soil liquid+ice + SWE + canopy snow + Wdew)
          from the written states (steps 1..), and the model's own
          OUT_WATER_ERROR (all steps)
  energy  put_data.c 5.1.0:540-552's expression on each model's own
          fluxes (FULL_ENERGY rows), and TorchVIC's own OUT_ENERGY_ERROR
          (the root residual, except in the forced cases named above, where
          it is put_data's expression of TorchVIC's rows). VIC's put_data
          expression does not close even for VIC (its terms belong to
          different control volumes), so it is reported side by side, not
          graded.
Grade of closure: TorchVIC's water residual max |.| <= 1e-10 mm pass,
<= 1e-6 warn, else fail.
"""
import glob
import os

import numpy as np

PASS, WARN = 1e-4, 1e-2
CLOSE_PASS, CLOSE_WARN = 1e-10, 1e-6
NOT_GRADED = {'OUT_WATER_ERROR', 'OUT_ENERGY_ERROR', 'OUT_SURF_COND',
              'OUT_TIME_VICRUN_WALL', 'OUT_TIME_VICRUN_CPU'}
#: Not graded on LAKES TRUE rows only. VIC 5.1.0 never sets the lake's
#: bare-soil evaporation: put_data.c:599 sums layer[].esoil of the lake's
#: soil (lake_var.soil, malloc'd in vic_alloc.c:61), which initialize_lake.c
#: and the lake branch of vic_run.c (5.1.0:368-466) never write, so VIC
#: reports uninitialised memory (0 in one run of the Findley row, 8.8e177 in
#: another). There is no VIC value to compare with.
NOT_GRADED_LAKES = {'OUT_EVAP_BARE'}
WATER = {'OUT_PREC', 'OUT_RAINF', 'OUT_SNOWF', 'OUT_EVAP', 'OUT_TRANSP_VEG',
         'OUT_EVAP_BARE', 'OUT_EVAP_CANOP', 'OUT_SUB_SNOW', 'OUT_SUB_CANOP',
         'OUT_SUB_BLOWING', 'OUT_SUB_SURFACE', 'OUT_RUNOFF', 'OUT_BASEFLOW',
         'OUT_INFLOW', 'OUT_SNOW_MELT', 'OUT_REFREEZE', 'OUT_WDEW',
         'OUT_SWE', 'OUT_SNOW_DEPTH', 'OUT_SNOW_CANOPY', 'OUT_SNOW_COVER',
         'OUT_SOIL_MOIST', 'OUT_SOIL_LIQ', 'OUT_SOIL_ICE', 'OUT_ROOTMOIST',
         'OUT_SOIL_WET', 'OUT_ASAT', 'OUT_DELSOILMOIST', 'OUT_DELSWE',
         'OUT_DELINTERCEPT'}
ENERGY = {'OUT_SWNET', 'OUT_LWNET', 'OUT_R_NET', 'OUT_LATENT',
          'OUT_LATENT_SUB', 'OUT_SENSIBLE', 'OUT_GRND_FLUX', 'OUT_DELTAH',
          'OUT_FUSION', 'OUT_ADVECTION', 'OUT_DELTACC', 'OUT_SNOW_FLUX',
          'OUT_RFRZ_ENERGY', 'OUT_MELT_ENERGY', 'OUT_SURF_TEMP',
          'OUT_SOIL_TEMP', 'OUT_BARESOILT', 'OUT_VEGT', 'OUT_RAD_TEMP',
          'OUT_SNOW_SURF_TEMP', 'OUT_SNOW_PACK_TEMP', 'OUT_SALBEDO'}
# everything else (aerodynamic resistance, PET, OUT_ALBEDO, the forcing
# echoes, LAI, fcanopy) is 'diag': reported and graded, but outside the
# row's grade, which is the water and energy groups and the closure.


def group(name):
    b = name.split('[')[0]
    return 'water' if b in WATER else 'energy' if b in ENERGY else 'diag'


TEMPS = {'OUT_SURF_TEMP', 'OUT_SOIL_TEMP', 'OUT_BARESOILT', 'OUT_VEGT',
         'OUT_RAD_TEMP', 'OUT_SNOW_SURF_TEMP', 'OUT_SNOW_PACK_TEMP',
         'OUT_AIR_TEMP', 'OUT_SOIL_TNODE'}


def _open(d):
    import xarray as xr
    files = sorted(glob.glob(os.path.join(d, '*.nc')))
    if not files:
        raise IOError('no history in %s' % d)
    parts = [xr.open_dataset(f, mask_and_scale=True).load() for f in files]
    return xr.concat(parts, dim='time') if len(parts) > 1 else parts[0]


def _cells(ds, v, mask):
    """[time, cell, k] float64 of variable v over the masked cells."""
    a = ds[v].values.astype('f8')
    if a.ndim == 3:
        a = a[:, None]
    a = np.moveaxis(a, 1, -1)              # time, y, x, k
    return a[:, mask, :]


def _agg_sum(name):
    from torchvic.io.history import default_agg
    return default_agg(name) == 'SUM'


def grade(x):
    if x is None or not np.isfinite(x):
        return 'fail'
    return 'pass' if x <= PASS else ('warn' if x <= WARN else 'fail')


def compare_dirs(vic_dir, tv_dir, r, status=None):
    vic, tv = _open(vic_dir), _open(tv_dir)
    out = dict(n_steps_vic=int(vic.sizes['time']),
               n_steps_tv=int(tv.sizes['time']))
    if out['n_steps_vic'] != out['n_steps_tv']:
        out['grade'] = 'fail'
        out['error'] = 'step counts differ: VIC %d, TorchVIC %d' % (
            out['n_steps_vic'], out['n_steps_tv'])
        return out
    mask = np.isfinite(vic['OUT_PREC'].values[0])
    sm = vic['OUT_SOIL_MOIST'].values[:, 0][:, mask]
    out['vic_nan_fraction'] = float(np.mean(~np.isfinite(sm)))
    if out['vic_nan_fraction'] > 0:
        # VIC's own run went to NaN (not a comparison): say so, and grade
        # the row by that rather than by the variables
        out['grade'] = 'VIC NaN'
        out['first_vic_nan'] = str(vic['time'].values[
            int(np.argmax((~np.isfinite(sm)).any(1)))])[:19]
        return out
    from matrix import CASES
    import xarray as xr
    with xr.open_dataset(CASES[r['case']]['domain']) as dom:
        area = dom['area'].values[mask].astype('f8')
    w = area / area.sum()
    times = [str(t)[:19] for t in vic['time'].values]
    out['n_cells'] = int(mask.sum())
    rows, missing = [], []
    for v in sorted(vic.data_vars):
        if not v.startswith('OUT_'):
            continue
        a = _cells(vic, v, mask)
        if v not in tv.data_vars:
            if np.nanmax(np.abs(a)) > 0:
                missing.append(v)
            continue
        b = _cells(tv, v, mask)
        if a.shape != b.shape:
            rows.append(dict(var=v, error='shape %s vs %s' % (a.shape, b.shape),
                             grade='fail'))
            continue
        for k in range(a.shape[-1]):
            name = v if a.shape[-1] == 1 else '%s[%d]' % (v, k)
            rows.append(_metrics(name, v, a[..., k], b[..., k], w, times))
    out['vars'] = rows
    out['not_written_by_torchvic'] = missing
    order = ['pass', 'warn', 'fail']
    for x in rows:
        x['group'] = group(x['var'])
    skip = NOT_GRADED | (NOT_GRADED_LAKES
                         if str(r.get('opts', {}).get('LAKES', '')).upper()
                         == 'TRUE' else set())
    graded = [x for x in rows if x['var'].split('[')[0] not in skip]
    for g in ('water', 'energy', 'diag'):
        gr = [x for x in graded if x['group'] == g]
        worst = max(gr, key=lambda x: x.get('score', np.inf)) if gr else None
        out['worst_' + g] = worst['var'] if worst else None
        out['score_' + g] = worst.get('score') if worst else None
        out['grade_' + g] = grade(worst.get('score')) if worst else 'n/a'
    out['closure'] = closure(vic, tv, mask, r)
    cw = out['closure'].get('tv_water_resid_max')
    vw = out['closure'].get('vic_water_resid_max') or 0.0
    # pass: closes to 1e-10 mm; warn: to 1e-6 mm, or no worse than VIC's
    # own residual on the same row (a leak inherited from VIC's equations)
    out['closure_inherited'] = bool(cw is not None and cw > CLOSE_PASS
                                    and cw <= 1.01 * vw + CLOSE_PASS)
    out['grade_closure'] = ('pass' if cw is not None and cw <= CLOSE_PASS else
                            'warn' if cw is not None and (
                                cw <= CLOSE_WARN or out['closure_inherited'])
                            else 'fail')
    keys = ['grade_water', 'grade_energy', 'grade_closure']
    out['grade'] = order[max(order.index(out[k]) for k in keys
                             if out[k] in order)]
    return out


# Absolute floor of the relative-difference denominators. For variables in mm
# (water states and fluxes) it is 1e-8 mm, not 1e-12: VIC's own Q12
# cancellation noise (runoff.c:478-481) leaves ~1e-12 mm in near-empty
# layers (igp NIJSSEN2001 OUT_SOIL_MOIST[2]: VIC total 1.42e-12 mm, max_abs
# 3.2e-12 mm, score 0.295 -> ~4e-5), which is noise, not a physics
# difference. Scores can only go down; only differences below ~1e-12 mm
# absolute are hidden. Other units keep 1e-12.
FLOOR_MM, FLOOR_DEFAULT = 1e-8, 1e-12


def abs_floor(base):
    try:
        from torchvic.io.history import outvar_metadata
        u = outvar_metadata().get(base, {}).get('units')
    except Exception:                                       # noqa: BLE001
        u = None
    return FLOOR_MM if u == 'mm' else FLOOR_DEFAULT


def _metrics(name, base, a, b, w, times):
    floor = abs_floor(base)
    off = 273.15 if base in TEMPS else 0.0
    ok = np.isfinite(a) & np.isfinite(b)
    nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
    a0 = np.where(ok, a, 0.0) + off
    b0 = np.where(ok, b, 0.0) + off
    f = np.sum if _agg_sum(base) else np.mean
    A, B = f(a0, 0), f(b0, 0)                      # per cell
    scale_c = f(np.abs(a0), 0)                     # per-cell size
    DA, DB = float((A * w).sum()), float((B * w).sum())
    scale = float((scale_c * w).sum())
    den = max(abs(DA), scale, floor)
    rel_total = abs(DB - DA) / den
    den_c = np.maximum(np.maximum(np.abs(A), scale_c), max(scale, floor))
    rel_cell = float(np.max(np.abs(B - A) / den_c)) if A.size else 0.0
    d = np.where(ok, b - a, 0.0)
    max_abs = float(np.abs(d).max()) if d.size else 0.0
    rmse = float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0
    thr = 1e-6 * (np.abs(a0) + np.mean(np.abs(a0)))
    hit = np.argwhere(np.abs(d) > thr)
    first = None
    if len(hit):
        t, c = hit[np.argmin(hit[:, 0] * 10 ** 6 + hit[:, 1])]
        first = dict(step=int(t), date=times[int(t)], cell=int(c),
                     vic=float(a[t, c]), tv=float(b[t, c]))
    trivial = max_abs == 0.0
    score = 0.0 if trivial else max(rel_total, rel_cell)
    if nan_mismatch:
        score = np.inf
    return dict(var=name, vic_total=DA - (off * (len(a) if f is np.sum else 1)
                                          if off else 0.0),
                tv_total=DB - (off * (len(a) if f is np.sum else 1)
                               if off else 0.0),
                rel_total=rel_total, rel_cell=rel_cell, max_abs=max_abs,
                rmse=rmse, first=first, nan_mismatch=nan_mismatch,
                score=score, grade=grade(score))


def _storage(ds, mask):
    s = 0.0
    for v in ('OUT_SOIL_LIQ', 'OUT_SOIL_ICE'):
        if v in ds:
            s = s + np.nansum(_cells(ds, v, mask), -1)
    for v in ('OUT_SWE', 'OUT_SNOW_CANOPY', 'OUT_WDEW', 'OUT_SURFSTOR'):
        if v in ds:
            s = s + _cells(ds, v, mask)[..., 0]
    return s


def _water_resid(ds, mask):
    g = lambda v: (_cells(ds, v, mask)[..., 0] if v in ds else 0.0)  # noqa
    flow = g('OUT_PREC') - g('OUT_EVAP') - g('OUT_RUNOFF') - g('OUT_BASEFLOW')
    s = _storage(ds, mask)
    return flow[1:] - (s[1:] - s[:-1])


def _energy_putdata(ds, mask):
    g = lambda v: (_cells(ds, v, mask)[..., 0] if v in ds else 0.0)  # noqa
    return (g('OUT_SWNET') + g('OUT_LWNET')
            - (g('OUT_LATENT') + g('OUT_LATENT_SUB'))
            - (g('OUT_SENSIBLE') + g('OUT_ADV_SENS'))
            - (g('OUT_GRND_FLUX') + g('OUT_DELTAH') + g('OUT_FUSION'))
            + (g('OUT_ADVECTION') - g('OUT_DELTACC') + g('OUT_SNOW_FLUX')
               + g('OUT_RFRZ_ENERGY')))


def _mx(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.abs(x).max()) if x.size else None


def closure(vic, tv, mask, r):
    c = {}
    c['tv_water_resid_max'] = _mx(_water_resid(tv, mask))
    c['vic_water_resid_max'] = _mx(_water_resid(vic, mask))
    if 'OUT_WATER_ERROR' in tv:
        c['tv_OUT_WATER_ERROR_max'] = _mx(_cells(tv, 'OUT_WATER_ERROR', mask))
    if 'OUT_WATER_ERROR' in vic:
        c['vic_OUT_WATER_ERROR_max'] = _mx(_cells(vic, 'OUT_WATER_ERROR',
                                                  mask))
    if r['mode'] in ('EB', 'FS'):
        et, ev = _energy_putdata(tv, mask), _energy_putdata(vic, mask)
        c['tv_putdata_energy_max'] = _mx(et)
        c['tv_putdata_energy_mean'] = float(np.nanmean(et))
        c['vic_putdata_energy_max'] = _mx(ev)
        c['vic_putdata_energy_mean'] = float(np.nanmean(ev))
        if 'OUT_ENERGY_ERROR' in tv:
            c['tv_root_resid_max'] = _mx(_cells(tv, 'OUT_ENERGY_ERROR', mask))
        if 'OUT_ENERGY_ERROR' in vic:
            c['vic_OUT_ENERGY_ERROR_max'] = _mx(_cells(vic, 'OUT_ENERGY_ERROR',
                                                       mask))
    return c
