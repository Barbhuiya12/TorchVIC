"""ISC forest cells under snow against a pristine VIC 5.1.0 image driver.

A 4 x 6 window of the Indian-subcontinent (ISC) benchmark domain in the
eastern Himalaya (28.97-29.16 N, 94.78-95.09 E): evergreen and mixed
forest tiles with an overstory (cover 0.94-1.0), a seasonal pack of
14-330 mm and up to 8 mm of snow held in the canopy, 1 January - 31 March
2004, 3-hourly water balance, SNOW_BAND FALSE. Both models read the same
domain, parameters and 3-hourly forcing, cut from the ISC files into the
test's directory.

In the full ISC run these cells carried most of the remaining snow-cell
ET difference (overstory tiles: 23,652 of 25,400 mm total |dET|). The
causes, each pinned here:

* the opening state: Tfoliage = T[0] (0 C in water balance) and the
  ground's LongUnderOut = EMISS_SNOW sigma (T[0] + 273.15)^4
  (generate_default_state.c 5.1.0:94-107), not the first step's air;
* the ground's LongUnderOut each step, LongUnderIn - NetLongUnder
  (calc_surf_energy_bal.c 5.1.0:612): a solved pack emits at its own
  surface temperature (SnowPackEnergyBalance.c 5.1.0:176-177), not Tair;
* RefreezeEnergy of the canopy from the liquid store the step opened with,
  IntRainOrg (snow_intercept.c 5.1.0:129, :289, :330;
  func_canopy_energy_bal.c 5.1.0:257);
* VIC's vp = svp(Tair) where the forcing is supersaturated, and the forcing
  scaled in double (vic_force.c 5.1.0:151, :397-405);
* the thin pack under an overstory evaluated at Tsurf = Tair like any
  other (snow_melt.c 5.1.0:302-304), with its density left uncompacted
  (solve_snow.c 5.1.0:343-353);
* the albedo age from the sky's snowfall, not the canopy's throughfall
  (solve_snow.c 5.1.0:127, :296), and snow->swq with its liquid.

Canopy air temperature is not one of them: the image driver never reads
CLOSE_ENERGY (initialize_options.c 5.1.0:31), so calc_atmos_energy_bal
returns Tcanopy = Tair (calc_atmos_energy_bal.c 5.1.0:73-103) and
vic_init.c 5.1.0:1558 sets MAX_ITER_GRND_CANOPY to 0.

Data: <data root>/benchmark_isc/forest_box (benchmarks/isc_snow_box.py
--rows 333:343 --cols 458:468 builds it); moved with
TORCHVIC_ISC_FOREST_BOX.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import _paths                                           # noqa: E402

BOX = _paths.env_path(('TORCHVIC_ISC_FOREST_BOX',), 'benchmark_isc/forest_box')
VIC51 = _paths.VIC51_IMAGE
ROWS, COLS = slice(2, 6), slice(0, 6)
OUTVARS = ('OUT_PREC', 'OUT_EVAP', 'OUT_TRANSP_VEG', 'OUT_EVAP_CANOP',
           'OUT_EVAP_BARE', 'OUT_SUB_SNOW', 'OUT_SUB_CANOP',
           'OUT_SNOW_CANOPY', 'OUT_SNOW_MELT', 'OUT_SWE', 'OUT_RUNOFF',
           'OUT_BASEFLOW', 'OUT_SOIL_MOIST', 'OUT_SALBEDO')

pytestmark = pytest.mark.skipif(
    not (os.path.exists(VIC51)
         and os.path.exists(os.path.join(BOX, 'params.nc'))),
    reason='pristine VIC 5.1.0 or the ISC forest box is not present')


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


def test_isc_forest_snow_cells_are_vic51s(tmp_path):
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
    # a real pack under every canopy, and snow held in the canopies
    assert (vic['OUT_SWE'].values.max(0)[mask] > 10.0).all()
    assert (vic['OUT_SNOW_CANOPY'].values.max(0)[mask] > 0.1).all()
    bad = []
    # mm per cell over the 91 days (states: means over the days). The
    # tolerances are what is left of VIC's own floating-point residues: a
    # canopy store VIC leaves at ~1e-16 mm instead of 0 runs its snow
    # branch on the next step (solve_snow.c 5.1.0:164-168).
    # Measured: ET 0.076, its parts <= 0.04, sublimation <= 1e-3, melt
    # 0.59 (the rain a residue-pack carries), SWE 0.013 mm per cell.
    for v, tol in (('OUT_PREC', 1e-6), ('OUT_EVAP', 0.1),
                   ('OUT_TRANSP_VEG', 0.06), ('OUT_EVAP_CANOP', 0.06),
                   ('OUT_EVAP_BARE', 0.06), ('OUT_SUB_SNOW', 0.002),
                   ('OUT_SUB_CANOP', 0.001), ('OUT_SNOW_CANOPY', 1e-4),
                   ('OUT_SNOW_MELT', 1.0), ('OUT_SWE', 0.03),
                   ('OUT_RUNOFF', 0.01), ('OUT_BASEFLOW', 1e-6),
                   ('OUT_SOIL_MOIST', 0.05)):
        a = vic[v].values.astype('f8')
        b = tv[v].values.astype('f8')
        if a.ndim == 4:
            a, b = a.sum(1), b.sum(1)
        f = (np.mean if v in ('OUT_SWE', 'OUT_SOIL_MOIST', 'OUT_SNOW_CANOPY')
             else np.sum)
        A, B = f(a, 0)[mask], f(b, 0)[mask]
        dmax = float(np.abs(B - A).max())
        print('%-16s VIC %10.4f  TorchVIC %10.4f  max cell |d| %.2e mm'
              % (v, A.mean(), B.mean(), dmax))
        if dmax > tol:
            bad.append(v)
    a = vic['OUT_SALBEDO'].values[:, mask]
    b = tv['OUT_SALBEDO'].values[:, mask]
    ok = np.isfinite(a) & np.isfinite(b)
    print('OUT_SALBEDO max daily |d| %.2e' % np.abs(a - b)[ok].max())
    assert np.abs(a - b)[ok].max() < 1e-2
    assert not bad, '%s off VIC 5.1.0 by more than the tolerance' % bad
