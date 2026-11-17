#!/usr/bin/env python
"""MODFLOW aquifer thickness, layer bottoms and specific yield for the ISC grid.

Aquifer thickness: absolute depth to bedrock BDTICM (Shangguan et al. 2017,
SoilGrids 250 m, cm, open from ISRIC), the ISC subset in
<data root>/raw/vicwur/bdticm_isc/.  Open replacement for Pelletier et al. 2016.
Conservative area-weighted regrid to the ISC grid (NaN-aware: no-data cells
are left out of the mean).  Fields written into
<data root>/vicwur/isc/params/modflow_static.nc (in place, existing GLHYMPS
fields kept):
  bdticm_m          mean depth to bedrock, m, no cap and no floor
  aqdepth           aquifer thickness, m = clip(bdticm_m, AQ_MIN, AQ_MAX)
  layer1_bottom     m a.s.l. = elev - 0.1 * aqdepth   (VIC-WUR 3.0 cal_botlayer_elevation)
  layer2_bottom     m a.s.l. = elev - aqdepth
  Sy                specific yield, see below
AQ_MAX = 200 m is the VIC-WUR 3.0 default aquifer depth; AQ_MIN = 5 m avoids zero-thickness
layers on outcropping bedrock (layer 1 is 10 % of aqdepth in the MODFLOW set-up).

Specific yield: GLHYMPS 2.0 porosity (already regridded in this file) times a
drainable fraction that rises with permeability,
   Sy = porosity * f,   f = 0.05 + 0.75 * clip((log10_k + 15) / 4, 0, 1)
(f 0.05 for clay-like logK <= -15, 0.8 for sand/gravel logK >= -11), then clipped
to [0.005, 0.30] (Johnson 1967 range of specific yield for clay to gravel).
This is a documented heuristic, not a measured Sy: porosity alone overstates the
drainable water of fine sediments.
"""
import os
import sys
import numpy as np
import netCDF4 as nc
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa
from regrid import Regridder, fill_nearest   # noqa

AQ_MIN, AQ_MAX = 5.0, 200.0
SY_MIN, SY_MAX = 0.005, 0.30
TIF = RAW + '/bdticm_isc/BDTICM_M_250m_ISC.tif'


def main():
    g = ISC()
    with rasterio.open(TIF) as ds:
        a = ds.read(1).astype(np.float32)
        t = ds.transform
        nod = ds.nodata
    ny, nx = a.shape
    lat = (t.f + t.e * (np.arange(ny) + 0.5))[::-1]
    lon = t.c + t.a * (np.arange(nx) + 0.5)
    a = a[::-1]
    a[a == nod] = np.nan
    a[a < 0] = np.nan
    a = (a / 100.0).astype(np.float64)               # m
    rows = ['source cells %d x %d, valid fraction %.4f, range %.1f..%.1f m'
            % (ny, nx, np.isfinite(a).mean(), np.nanmin(a), np.nanmax(a))]
    rg = Regridder(lat, lon, g.lat, g.lon)
    m, cov = rg.mean(a)
    st, dt, pe = rg.mass_check(a, m)
    rows.append('BDTICM area integral (m x m2): src %.6e dst %.6e err %.3e %%' % (st, dt, pe))
    nofill = ~np.isfinite(m) | (cov < 0.5)
    rows.append('active cells with coverage < 0.5 or no data (nearest-filled): %d'
                % int((nofill & g.active).sum()))
    bd = fill_nearest(np.where(cov < 0.5, np.nan, m))
    aq = np.clip(bd, AQ_MIN, AQ_MAX)
    with nc.Dataset(DOMAIN) as d:
        elev = np.array(d['elev'][:], float)
    b1 = elev - 0.1 * aq
    b2 = elev - aq
    p = OUT + '/params/modflow_static.nc'
    with nc.Dataset(p, 'a') as ds:
        lk = np.array(ds['log10_k'][:], float)
        por = np.array(ds['porosity'][:], float)
        f = 0.05 + 0.75 * np.clip((lk + 15.0) / 4.0, 0.0, 1.0)
        sy = np.clip(por * f, SY_MIN, SY_MAX)

        def put(name, arr, units, long_name, new=True):
            if name in ds.variables:
                v = ds[name]
                for at in ('PLACEHOLDER',):
                    if at in v.ncattrs():
                        v.delncattr(at)
            else:
                v = ds.createVariable(name, 'f8', ('lat', 'lon'), fill_value=FILL, zlib=True)
            v.units, v.long_name = units, long_name
            v[:] = arr
        put('bdticm_m', bd, 'm', 'depth to bedrock, Shangguan 2017 BDTICM 250 m, conservative mean')
        put('aqdepth', aq, 'm', 'aquifer thickness = clip(bdticm_m, %g, %g)' % (AQ_MIN, AQ_MAX))
        put('layer1_bottom', b1, 'm', 'MODFLOW layer 1 bottom = elev - 0.1 aqdepth')
        put('layer2_bottom', b2, 'm', 'MODFLOW layer 2 bottom = elev - aqdepth')
        put('Sy', sy, '-', 'specific yield = GLHYMPS porosity x drainable fraction f(logK), see make_aquifer.py')
    for nm, arr, u in (('bdticm_m', bd, 'm'), ('aqdepth', aq, 'm'), ('layer1_bottom', b1, 'm'),
                       ('layer2_bottom', b2, 'm'), ('Sy', sy, '-'), ('porosity', por, '-'),
                       ('log10_k', lk, 'log10 m2')):
        rows.append(stats_row(nm, arr, g.active, u))
    # regional medians by hand-picked boxes (sanity: IGP deep, Deccan trap shallow, Himalaya shallow)
    for nm, (la0, la1, lo0, lo1) in (('Indo-Gangetic plain', (25, 31, 75, 87)),
                                     ('Deccan plateau', (16, 20, 74, 79)),
                                     ('Thar desert', (26, 29, 70, 73)),
                                     ('Himalaya', (28, 33, 79, 85))):
        la = (g.lat >= la0) & (g.lat <= la1)
        lo = (g.lon >= lo0) & (g.lon <= lo1)
        box = np.zeros_like(g.active)
        box[np.ix_(la, lo)] = True
        box &= g.active
        rows.append('region %-20s cells %5d  bdticm median %6.1f m  aqdepth median %6.1f  Sy median %.3f' % (
            nm, box.sum(), np.median(bd[box]), np.median(aq[box]), np.median(sy[box])))
    for r in rows:
        log('CHECK ' + r)
    log('aquifer done')


if __name__ == '__main__':
    main()
