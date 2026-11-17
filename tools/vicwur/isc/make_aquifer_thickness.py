#!/usr/bin/env python
"""Aquifer thickness for the ISC MODFLOW set-up: de Graaf for sediments, Pelletier elsewhere.

Replaces the BDTICM-based aqdepth of make_aquifer.py (BDTICM saturates in deep
alluvium: IGP median 27 m).  Sources (<data root>/raw/vicwur/):
  degraaf_aquifer/shapefiles.zip  de Graaf et al. 2015 (HESS 19, 823) aquifer thickness at
        0.5 deg (field MEAN_thk_m), as digitised by Niazi et al. 2024 (Zenodo 10631421, CC BY 4.0)
  pelletier/average_soil_and_sedimentary-deposit_thickness.tif  Pelletier et al. 2016
        (ORNL DAAC 1304), 1 km, m, capped at 50 m by design, nodata -1
  glhymps2/GLHYMPS.shp  GLHYMPS 2.0 (Huscroft et al. 2018); GUM_K == 2 marks polygons taken
        from the Global Unconsolidated sediments Map (GUM), i.e. unconsolidated sediments
Rule, applied on a 1/128 deg sub-grid (8 x 8 per ISC cell) before conservative averaging:
  GLHYMPS GUM_K == 2 and de Graaf valid  -> de Graaf thickness
  otherwise                               -> Pelletier thickness (de Graaf if Pelletier is no-data)
aqdepth = clip(result, AQ_MIN, AQ_MAX); AQ_MAX = 1000 m is the GLOBGM maximum total
groundwater thickness; AQ_MIN = 20 m is the hard-rock weathered zone (CGWB 10-30 m; Pelletier is soil only,
1-2 m).  The VIC-WUR 3.0 default of 200 m cuts the IGP.
Writes (in place, backup modflow_static.nc.bdticm_bak kept): aqdepth, layer1_bottom,
layer2_bottom, plus aq_degraaf, aq_pelletier, aq_unconsol_frac.
"""
import os
import sys
import shutil
import subprocess
import zipfile
import numpy as np
import netCDF4 as nc
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                    # noqa
from common import *                             # noqa
from regrid import Regridder, fill_nearest       # noqa

common.LOG = ROOT + '/verification/aquifer/run.log'
GDAL = '/apps/compilers/miniforge3/envs/geospatial'
SUB = 8
AQ_MIN, AQ_MAX = 20.0, 1000.0   # 20 m: hard-rock weathered zone, CGWB 10-30 m
AW = WORK + '/aquifer'
PEL = RAW + '/pelletier/average_soil_and_sedimentary-deposit_thickness.tif'
DG_ZIP = RAW + '/degraaf_aquifer/shapefiles.zip'
GLH = WORK + '/glhymps/GLHYMPS.shp'


def sh(cmd):
    env = dict(os.environ, PATH=GDAL + '/bin:' + os.environ['PATH'],
               LD_LIBRARY_PATH=GDAL + '/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
               GDAL_DATA=GDAL + '/share/gdal', PROJ_DATA=GDAL + '/share/proj', PROJ_LIB=GDAL + '/share/proj')
    log('RUN ' + ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def raster(src, layer, attr, out, te, res, srcsrs=None):
    if os.path.exists(out):
        return
    sh(['gdal_rasterize', '-l', layer, '-a', attr, '-te'] + [str(x) for x in te] +
       ['-tr', str(res), str(res), '-a_nodata', '-9999', '-init', '-9999', '-ot', 'Float32',
        '-co', 'COMPRESS=DEFLATE', src, out])


def load(path, flip=True):
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
    a = a[::-1]
    a[a == -9999] = np.nan
    return a


def main():
    g = ISC()
    os.makedirs(AW, exist_ok=True)
    n0, n1 = g.lat[0] - 1 / 32, g.lat[-1] + 1 / 32
    w0, w1 = g.lon[0] - 1 / 32, g.lon[-1] + 1 / 32
    res = 1.0 / (16 * SUB)
    te = (w0, n0, w1, n1)
    bb = [str(x) for x in (w0 - 1, n0 - 1, w1 + 1, n1 + 1)]
    # ---- de Graaf polygons -> 1/128 raster
    if not os.path.exists(AW + '/degraaf_isc.gpkg'):
        with zipfile.ZipFile(DG_ZIP) as z:
            z.extractall(AW + '/dg', [n for n in z.namelist() if '/inputs.' in n])
        sh(['ogr2ogr', '-f', 'GPKG', AW + '/degraaf_isc.gpkg', AW + '/dg/shapefiles/inputs.shp',
            '-t_srs', 'EPSG:4326', '-nlt', 'PROMOTE_TO_MULTI', '-nln', 'g', '-spat'] + bb +
           ['-spat_srs', 'EPSG:4326', '--config', 'OGR_ENABLE_PARTIAL_REPROJECTION', 'YES'])
    raster(AW + '/degraaf_isc.gpkg', 'g', 'Aqfr_th', AW + '/dg_128.tif', te, res)
    # ---- GLHYMPS GUM_K
    if not os.path.exists(AW + '/gum_isc.gpkg'):
        sh(['ogr2ogr', '-f', 'GPKG', AW + '/gum_isc.gpkg', GLH, '-t_srs', 'EPSG:4326', '-nlt',
            'PROMOTE_TO_MULTI', '-select', 'GUM_K', '-nln', 'g', '-spat'] + bb +
           ['-spat_srs', 'EPSG:4326', '--config', 'OGR_ENABLE_PARTIAL_REPROJECTION', 'YES'])
    raster(AW + '/gum_isc.gpkg', 'g', 'GUM_K', AW + '/gum_128.tif', te, res)
    dg = load(AW + '/dg_128.tif')
    gum = load(AW + '/gum_128.tif')
    ny, nx = dg.shape
    assert (ny, nx) == (g.ny * SUB, g.nx * SUB), (ny, nx)
    lat = n0 + (np.arange(ny) + 0.5) * res
    lon = w0 + (np.arange(nx) + 0.5) * res
    # ---- Pelletier 1 km, window then nearest-sample onto the sub-grid
    with rasterio.open(PEL) as ds:
        win = from_bounds(w0, n0, w1, n1, ds.transform).round_offsets().round_lengths()
        pa = ds.read(1, window=win).astype(np.float64)
        t = ds.window_transform(win)
    pa[pa < 0] = np.nan
    pr = (t.f + t.e * (np.arange(pa.shape[0]) + 0.5))
    pc = t.c + t.a * (np.arange(pa.shape[1]) + 0.5)
    ri = np.clip(np.round((lat - t.f) / t.e - 0.5).astype(int), 0, pa.shape[0] - 1)
    ci = np.clip(np.round((lon - t.c) / t.a - 0.5).astype(int), 0, pa.shape[1] - 1)
    pel = pa[np.ix_(ri, ci)]
    unc = (gum == 2) & np.isfinite(dg)
    thk = np.where(unc, dg, np.where(np.isfinite(pel), pel, dg))
    log('sub-grid cells %d x %d; unconsolidated (GUM_K=2 & de Graaf) %.3f; Pelletier valid %.3f; de Graaf valid %.3f'
        % (ny, nx, unc.mean(), np.isfinite(pel).mean(), np.isfinite(dg).mean()))
    rg = Regridder(lat, lon, g.lat, g.lon)
    out = {}
    for k, a in (('aq_raw', thk), ('aq_degraaf', dg), ('aq_pelletier', pel), ('aq_unconsol_frac', unc.astype(float))):
        m, cov = rg.mean(a)
        out[k] = m
        if k == 'aq_raw':
            st, dt, pe = rg.mass_check(a, m)
            log('CHECK thickness area integral err %.3e %%' % pe)
            nofill = ~np.isfinite(m) | (cov < 0.5)
            log('CHECK active cells with coverage < 0.5 (nearest-filled): %d' % int((nofill & g.active).sum()))
            out[k] = fill_nearest(np.where(cov < 0.5, np.nan, m))
    raw = out['aq_raw']
    aq = np.clip(raw, AQ_MIN, AQ_MAX)
    with nc.Dataset(DOMAIN) as d:
        elev = np.array(d['elev'][:], float)
    b1, b2 = elev - 0.1 * aq, elev - aq
    p = OUT + '/params/modflow_static.nc'
    bak = p + '.bdticm_bak'
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
    with nc.Dataset(p, 'a') as ds:
        def put(name, arr, units, ln):
            v = ds[name] if name in ds.variables else ds.createVariable(
                name, 'f8', ('lat', 'lon'), fill_value=FILL, zlib=True)
            v.units, v.long_name = units, ln
            v[:] = np.where(np.isfinite(arr), arr, FILL)
        put('aqdepth', aq, 'm', 'aquifer thickness = clip(de Graaf on GUM sediments else Pelletier, %g, %g)' % (AQ_MIN, AQ_MAX))
        put('layer1_bottom', b1, 'm', 'MODFLOW layer 1 bottom = elev - 0.1 aqdepth')
        put('layer2_bottom', b2, 'm', 'MODFLOW layer 2 bottom = elev - aqdepth')
        put('aq_degraaf', out['aq_degraaf'], 'm', 'de Graaf 2015 aquifer thickness, conservative mean (NaN-aware)')
        put('aq_pelletier', out['aq_pelletier'], 'm', 'Pelletier 2016 soil+sediment thickness, conservative mean (NaN-aware)')
        put('aq_unconsol_frac', out['aq_unconsol_frac'], '-', 'area fraction taking de Graaf (GLHYMPS GUM_K=2)')
        bd = np.array(ds['bdticm_m'][:], float)
    for nm, arr in (('aqdepth', aq), ('layer2_bottom', b2)):
        log('CHECK ' + stats_row(nm, arr, g.active, 'm'))
    boxes = (('Indo-Gangetic plain', (25, 31, 75, 87)), ('Peninsular (Deccan)', (16, 20, 74, 79)),
             ('Himalaya', (28, 33, 79, 85)), ('Thar desert', (26, 29, 70, 73)))
    for nm, (la0, la1, lo0, lo1) in boxes:
        box = np.zeros_like(g.active)
        box[np.ix_((g.lat >= la0) & (g.lat <= la1), (g.lon >= lo0) & (g.lon <= lo1))] = True
        box &= g.active
        f = lambda a: np.nanmedian(np.where(a[box] > -1e29, a[box], np.nan))
        log('TABLE %-20s cells %5d  aqdepth median %7.1f (p10 %6.1f p90 %7.1f)  old bdticm %6.1f  degraaf %7.1f  pelletier %5.1f  unconsol frac %.2f'
            % (nm, box.sum(), np.median(aq[box]), np.percentile(aq[box], 10), np.percentile(aq[box], 90),
               np.median(bd[box]), f(out['aq_degraaf']), f(out['aq_pelletier']), out['aq_unconsol_frac'][box].mean()))
    log('aquifer thickness done')


if __name__ == '__main__':
    main()
