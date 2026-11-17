#!/usr/bin/env python
"""MODFLOW static fields from GLHYMPS 2.0 (C14/C15): permeability and porosity.

Polygons are clipped to the ISC box, reprojected to EPSG:4326 and rasterised
(GDAL) at 1/128 deg (8 x 8 sub-cells per ISC cell), then averaged
conservatively onto the ISC grid.  Fields (modflow_static.nc):
  log10_k    log10 permeability, m2 (area mean of logK_Ferr/100)
  ksat       hydraulic conductivity m/s = 10^(log10_k) * 1e7 (geometric mean)
  porosity   - (Porosity_x / 100, area mean)
  Cells with no GLHYMPS value (water, ice, gaps) are filled from the nearest
  valid cell and listed in `nofill` (1 = filled).
Needs GDAL command line tools (conda env geospatial).
"""
import os
import subprocess
import sys
import numpy as np
import netCDF4 as nc
import rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa
from regrid import Regridder, fill_nearest   # noqa

GDAL = '/apps/compilers/miniforge3/envs/geospatial'
SUB = 8
ATTRS = {'logk': 'logK_Ferr_', 'por': 'Porosity_x'}
NODATA = -9999.0


def sh(cmd):
    env = dict(os.environ, PATH=GDAL + '/bin:' + os.environ['PATH'],
               LD_LIBRARY_PATH=GDAL + '/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
               GDAL_DATA=GDAL + '/share/gdal', PROJ_DATA=GDAL + '/share/proj', PROJ_LIB=GDAL + '/share/proj')
    log('RUN ' + ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    g = ISC()
    wd = WORK + '/glhymps'
    shp = wd + '/GLHYMPS.shp'
    n0, n1 = g.lat[0] - 1 / 32, g.lat[-1] + 1 / 32
    w0, w1 = g.lon[0] - 1 / 32, g.lon[-1] + 1 / 32
    clip = wd + '/glhymps_isc.gpkg'
    if not os.path.exists(clip):
        sh(['ogr2ogr', '-f', 'GPKG', clip, shp, '-t_srs', 'EPSG:4326',
            '-spat', str(w0), str(n0), str(w1), str(n1),
            '-spat_srs', 'EPSG:4326', '-nlt', 'PROMOTE_TO_MULTI',
            '-select', ','.join(ATTRS.values()), '-nln', 'g',
            '--config', 'OGR_ENABLE_PARTIAL_REPROJECTION', 'YES'])
    res = 1.0 / (16 * SUB)
    ras = {}
    for k, a in ATTRS.items():
        out = '%s/%s_1_128.tif' % (wd, k)
        if not os.path.exists(out):
            sh(['gdal_rasterize', '-l', 'g', '-a', a, '-te', str(w0), str(n0),
                str(w1), str(n1), '-tr', str(res), str(res), '-a_nodata',
                str(NODATA), '-init', str(NODATA), '-ot', 'Float32',
                '-co', 'COMPRESS=DEFLATE', clip, out])
        with rasterio.open(out) as ds:
            arr = ds.read(1).astype(float)
        arr = arr[::-1]
        arr[arr == NODATA] = np.nan
        ras[k] = arr
    ny, nx = ras['logk'].shape
    assert (ny, nx) == (g.ny * SUB, g.nx * SUB), (ny, nx)
    lat = n0 + (np.arange(ny) + 0.5) * res
    lon = w0 + (np.arange(nx) + 0.5) * res
    rg = Regridder(lat, lon, g.lat, g.lon)
    logk = ras['logk'] / 100.0
    logk[(logk > -5) | (logk < -30)] = np.nan          # no-data / gaps
    por = ras['por'] / 100.0
    por[(por <= 0) | (por > 1)] = np.nan
    rows = []
    out = {}
    # GLHYMPS gives porosity 0 (missing) for many GUM-derived sediment
    # polygons (e.g. Indo-Gangetic plain) that do have a permeability: fill
    # them with the median porosity of valid cells of the same logK bin.
    miss = np.isfinite(logk) & ~np.isfinite(por)
    ok = np.isfinite(logk) & np.isfinite(por)
    bins = np.round(logk * 4).astype(float)
    nb = int(miss.sum())
    if nb and ok.any():
        ub = np.unique(bins[ok])
        med = np.array([np.median(por[ok & (bins == b)]) for b in ub])
        ix = np.clip(np.searchsorted(ub, bins[miss]), 0, len(ub) - 1)
        por[miss] = med[ix]
    rows.append('porosity missing where logK valid (fine cells): %d (%.2f %%),'
                ' filled by logK-bin median' % (nb, 100.0 * nb / max(
                    np.isfinite(logk).sum(), 1)))
    for nm, a in (('log10_k', logk), ('porosity', por)):
        m, cov = rg.mean(a)
        st, dt, pe = rg.mass_check(a, m)
        rows.append('%s area integral: src %.6e  dst %.6e  err %.3e %%'
                    % (nm, st, dt, pe))
        out[nm] = (m, cov)
    nofill = ~np.isfinite(out['log10_k'][0]) | ~np.isfinite(out['porosity'][0])
    lk = fill_nearest(out['log10_k'][0])
    po = fill_nearest(out['porosity'][0])
    ks = 10.0 ** lk * 1e7
    rows.append('cells (active) without GLHYMPS value, nearest-filled: %d'
                % int((nofill & g.active).sum()))
    os.makedirs(OUT + '/params', exist_ok=True)
    with nc.Dataset(OUT + '/params/modflow_static.nc', 'w') as ds:
        coords(ds, g)
        for n_, a, un, ln in (
                ('log10_k', lk, 'log10(m2)', 'log10 permeability (GLHYMPS 2.0 logK_Ferr)'),
                ('ksat', ks, 'm s-1', 'hydraulic conductivity 10^logk*1e7'),
                ('porosity', po, '-', 'GLHYMPS 2.0 porosity'),
                ('nofill', nofill.astype('i4'), '1', '1 = nearest-filled')):
            v = ds.createVariable(n_, a.dtype, ('lat', 'lon'), zlib=True)
            v.units, v.long_name = un, ln
            v[:] = a
        for n_, why in (('aqdepth', 'Pelletier depth to bedrock (ORNL 1304, Earthdata login)'),
                        ('confining_K', 'needs Pelletier / literature'),
                        ('Sy', 'CGWB specific yield not downloaded'),
                        ('initial_head', 'steady-state run (later wave)')):
            v = ds.createVariable(n_, 'f8', ('lat', 'lon'), fill_value=FILL)
            v.PLACEHOLDER = 'all fill: ' + why
            v[:] = FILL
        ds.title = 'MODFLOW static fields, ISC: GLHYMPS K and porosity'
    for nm, a, un in (('log10_k', lk, 'log10 m2'), ('ksat', ks, 'm/s'),
                      ('porosity', po, '-')):
        rows.append(stats_row(nm, a, g.active, un))
    for r in rows:
        log('CHECK ' + r)
    log('modflow done')


if __name__ == '__main__':
    main()
