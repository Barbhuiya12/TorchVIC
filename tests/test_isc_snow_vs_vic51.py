"""ISC snow cells against a pristine VIC 5.1.0 image driver.

A 4 x 6 window of the Indian-subcontinent (ISC) benchmark domain on the
Tibetan plateau by Nam Co (30.4-30.6 N, 90.3-90.6 E, 4,700-5,000 m):
grassland, barren and open-water tiles, all without an overstory, under a
seasonal pack, 1 January - 31 March 2004, 3-hourly water balance,
SNOW_BAND FALSE. Both models read the same domain, parameters and
3-hourly forcing, cut from the ISC files into the test's directory.

In the full ISC run these were the worst cells: TorchVIC's ET was 100-180
mm/yr above VIC's and its peak SWE a third of VIC's. Three causes, all
pinned here:

* the open-field pack's resistance (CalcAerodynamic.c 5.1.0:97-104 with
  the cell's snow_rough and the tile's wind_h reference height, vic_run.c
  5.1.0:168-201) -- it used 0.03 m and the global WIND_H, 3x too little
  resistance at wind_h 3 m, snow_rough 0.0005 m;
* snow->coldcontent (snow_melt.c 5.1.0:478, :433-456), which picks the
  albedo curve and MELTING -- it was read as surf_temp >= 0;
* a thin pack's OUT_SNOW_MELT, calc_surf_energy_bal's melt
  (surface_fluxes.c 5.1.0:610, :951-952).

Data: <data root>/benchmark_isc/snow_box (benchmarks/isc_snow_box.py builds
it from the ISC files); moved with TORCHVIC_ISC_SNOW_BOX.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import _paths                                           # noqa: E402

BOX = _paths.env_path(('TORCHVIC_ISC_SNOW_BOX',), 'benchmark_isc/snow_box')
VIC51 = _paths.VIC51_IMAGE
ROWS, COLS = slice(0, 4), slice(0, 6)
OUTVARS = ('OUT_PREC', 'OUT_EVAP', 'OUT_SUB_SNOW', 'OUT_SNOW_MELT',
           'OUT_SWE', 'OUT_RUNOFF', 'OUT_BASEFLOW', 'OUT_SOIL_MOIST',
           'OUT_SALBEDO')

pytestmark = pytest.mark.skipif(
    not (os.path.exists(VIC51)
         and os.path.exists(os.path.join(BOX, 'params.nc'))),
    reason='pristine VIC 5.1.0 or the ISC snow box is not present')


def _case(tmp):
    xr = pytest.importorskip('xarray')
    sl = dict(lat=ROWS, lon=COLS)
    for f in ('domain', 'params'):
        with xr.open_dataset(os.path.join(BOX, f + '.nc'),
                             mask_and_scale=False, decode_times=False) as d:
            d.isel(**sl).load().to_netcdf(os.path.join(tmp, f + '.nc'))
    with xr.open_dataset(os.path.join(BOX, 'isc_3h.2004.nc'),
                         mask_and_scale=False, decode_times=False) as d:
        d.isel(time=slice(0, 8 * 91), **sl).load().to_netcdf(
            os.path.join(tmp, 'isc_3h.2004.nc'))
    lines = ['MODEL_STEPS_PER_DAY 8', 'SNOW_STEPS_PER_DAY 8',
             'RUNOFF_STEPS_PER_DAY 8', 'STARTYEAR 2004', 'STARTMONTH 1',
             'STARTDAY 1', 'ENDYEAR 2004', 'ENDMONTH 3', 'ENDDAY 31',
             'CALENDAR PROLEPTIC_GREGORIAN', 'FULL_ENERGY FALSE',
             'FROZEN_SOIL FALSE', 'NODES 3',
             'DOMAIN %s/domain.nc' % tmp]
    lines += ['DOMAIN_TYPE %s %s' % kv for kv in (
        ('LAT', 'lat'), ('LON', 'lon'), ('MASK', 'mask'), ('AREA', 'area'),
        ('FRAC', 'frac'), ('YDIM', 'lat'), ('XDIM', 'lon'))]
    lines += ['FORCING1 %s/isc_3h.' % tmp]
    lines += ['FORCE_TYPE %s %s' % kv for kv in (
        ('AIR_TEMP', 'air_temp'), ('PREC', 'prec'), ('PRESSURE', 'pressure'),
        ('SWDOWN', 'swdown'), ('LWDOWN', 'lwdown'), ('VP', 'vp'),
        ('WIND', 'wind'))]
    lines += ['WIND_H 10.0', 'PARAMETERS %s/params.nc' % tmp,
              'SNOW_BAND FALSE', 'BASEFLOW ARNO', 'JULY_TAVG_SUPPLIED FALSE',
              'ORGANIC_FRACT FALSE', 'LAI_SRC FROM_VEGLIB',
              'FCAN_SRC FROM_VEGLIB', 'ALB_SRC FROM_VEGLIB',
              'RESULT_DIR @OUT@/', 'LOG_DIR @OUT@/', 'OUTFILE fluxes',
              'AGGFREQ NDAYS 1', 'HISTFREQ NYEARS 1', 'OUT_FORMAT NETCDF4']
    lines += ['OUTVAR %s' % v for v in OUTVARS]
    text = '\n'.join(lines) + '\n'
    out = {}
    for who in ('vic', 'tv'):
        d = os.path.join(tmp, who)
        os.makedirs(d)
        g = os.path.join(d, 'global.txt')
        open(g, 'w').write(text.replace('@OUT@', d))
        out[who] = (g, d)
    return out


def test_isc_snow_cells_are_vic51s(tmp_path):
    xr = pytest.importorskip('xarray')
    case = _case(str(tmp_path))
    g, d = case['vic']
    env = dict(os.environ, OMPI_MCA_btl_vader_single_copy_mechanism='none')
    r = subprocess.run([VIC51, '-g', g], env=env, capture_output=True,
                       text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    g, d = case['tv']
    r = subprocess.run([sys.executable, '-m', 'torchvic', 'run', g,
                        '--core', '5.1.0', '--device', 'cpu', '--out', d,
                        '--quiet'], capture_output=True, text=True,
                       timeout=1800, cwd=_paths.REPO)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    vic = xr.open_dataset(os.path.join(case['vic'][1],
                                       'fluxes.2004-01-01.nc'))
    tv = xr.open_dataset(os.path.join(d, 'fluxes.2004-01-01.nc'))
    assert vic.sizes['time'] == tv.sizes['time'] == 91
    mask = np.isfinite(vic['OUT_SWE'].values[0])
    assert mask.sum() == 24
    # every cell carries a pack, and it is not a trace
    assert (vic['OUT_SWE'].values.max(0)[mask] > 3.0).all()
    bad = []
    for v, tol in (('OUT_PREC', 1e-6), ('OUT_EVAP', 0.02),
                   ('OUT_SUB_SNOW', 0.02), ('OUT_SNOW_MELT', 0.02),
                   ('OUT_SWE', 0.02), ('OUT_RUNOFF', 0.02),
                   ('OUT_BASEFLOW', 0.02), ('OUT_SOIL_MOIST', 0.02)):
        a = vic[v].values.astype('f8')
        b = tv[v].values.astype('f8')
        if a.ndim == 4:
            a, b = a.sum(1), b.sum(1)
        # fluxes: run totals; states: means over the days, mm per cell
        f = np.mean if v in ('OUT_SWE', 'OUT_SOIL_MOIST') else np.sum
        A, B = f(a, 0)[mask], f(b, 0)[mask]
        dmax = float(np.abs(B - A).max())
        print('%-15s VIC %10.4f  TorchVIC %10.4f  max cell |d| %.2e mm'
              % (v, A.mean(), B.mean(), dmax))
        if dmax > tol:
            bad.append(v)
    a = vic['OUT_SALBEDO'].values[:, mask]
    b = tv['OUT_SALBEDO'].values[:, mask]
    ok = np.isfinite(a) & np.isfinite(b)
    print('OUT_SALBEDO max daily |d| %.2e' % np.abs(a - b)[ok].max())
    assert np.abs(a - b)[ok].max() < 1e-3
    assert not bad, '%s off VIC 5.1.0 by more than the tolerance' % bad
