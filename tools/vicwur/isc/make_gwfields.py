#!/usr/bin/env python
"""Fill placeholders from open data (run after make_landuse / make_modflow).


   equipped for irrigation (ha) and % of it irrigated with groundwater, 5
   arcmin.  gw area and AEI are regridded as extensive fields; fraction =
   gw area / AEI.  Cells with no AEI take the nearest cell with AEI.
2. modflow_static.nc wtd (m) and initial_head (m a.s.l. = domain elev - wtd):
   Fan / Miguez-Macho annual-mean water table depth (30 arcsec, EURASIA
   file, GFNL-USC THREDDS, EartH2Observe).  Mean regridded; nearest fill.
"""
import os
import sys
import numpy as np
import netCDF4 as nc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa
from regrid import Regridder, fill_nearest   # noqa


def asc(path):
    a = np.loadtxt(path, skiprows=6)
    return a[::-1]            # south-up; 5 arcmin, lon -180..180


def main():
    g = ISC()
    rows = []
    # ---- GMIA
    d = RAW + '/gmia_v5'
    ha = asc(d + '/gmia_v5_aei_ha.asc')
    pct = asc(d + '/gmia_v5_aeigw_pct_aei.asc')
    lat = -90 + (np.arange(2160) + 0.5) / 12.0
    lon = -180 + (np.arange(4320) + 0.5) / 12.0
    ha = np.where(ha < 0, 0.0, ha)
    gw = ha * np.where(pct < 0, 0.0, pct) / 100.0
    la = np.where((lat > g.lat[0] - 0.3) & (lat < g.lat[-1] + 0.3))[0]
    lo = np.where((lon > g.lon[0] - 0.3) & (lon < g.lon[-1] + 0.3))[0]
    s = (slice(la.min(), la.max() + 1), slice(lo.min(), lo.max() + 1))
    rg = Regridder(lat[s[0]], lon[s[1]], g.lat, g.lon)
    ar = rg.src_area() / 1e4                       # ha
    res = {}
    for nm, a in (('aei', ha[s]), ('aei_gw', gw[s])):
        dens = a / ar
        m, _ = rg.mean(dens)
        st, dt, pe = rg.mass_check(dens, m)
        rows.append('GMIA %s area: src %.6e ha  dst %.6e ha  err %.3e %%'
                    % (nm, st / 1e4, dt / 1e4, pe))
        res[nm] = np.nan_to_num(m)
    with np.errstate(invalid='ignore', divide='ignore'):
        fr = np.where(res['aei'] > 0, res['aei_gw'] / res['aei'], np.nan)
    nfill = int((~np.isfinite(fr) & g.active).sum())
    fr = np.clip(fill_nearest(fr), 0, 1)
    rows.append('groundwater_fraction: %d active cells without AEI nearest-filled' % nfill)
    rows.append(stats_row('groundwater_fraction', fr, g.active, '-'))
    with nc.Dataset(OUT + '/params/irr.nc', 'r+') as ds:
        v = ds['groundwater_fraction']
        if 'PLACEHOLDER' in v.ncattrs():
            v.delncattr('PLACEHOLDER')
        v.source = ('GMIA v5 (Siebert et al. 2010, FAO): AEI groundwater '
                    'share, area-weighted')
        v[:] = fr
    # ---- water table
    with nc.Dataset(RAW + '/fan2013_wtd/EURASIA_WTD_annualmean.nc') as f:
        la_ = np.array(f['lat'][:], float)
        lo_ = np.array(f['lon'][:], float)
        i = np.where((la_ > g.lat[0] - 0.1) & (la_ < g.lat[-1] + 0.1))[0]
        j = np.where((lo_ > g.lon[0] - 0.1) & (lo_ < g.lon[-1] + 0.1))[0]
        w = np.ma.filled(f['WTD'][0, i.min():i.max() + 1, j.min():j.max() + 1]
                         .astype(float), np.nan)
        mk = np.array(f['mask'][i.min():i.max() + 1, j.min():j.max() + 1])
    w[mk == 0] = np.nan
    w = -w                      # file stores height of the table (negative below surface)
    w[w < 0] = 0.0
    rg2 = Regridder(la_[i], lo_[j], g.lat, g.lon)
    m, cov = rg2.mean(w)
    st, dt, pe = rg2.mass_check(w, m)
    rows.append('WTD area integral: src %.6e  dst %.6e  err %.3e %%'
                % (st, dt, pe))
    nf = int((~np.isfinite(m) & g.active).sum())
    wtd = fill_nearest(m)
    with nc.Dataset(DOMAIN) as dd:
        elev = np.array(dd['elev'][:], float)
    head = elev - wtd
    rows.append('wtd: %d active cells nearest-filled' % nf)
    rows.append(stats_row('wtd', wtd, g.active, 'm'))
    rows.append(stats_row('initial_head', head, g.active, 'm asl'))
    with nc.Dataset(OUT + '/params/modflow_static.nc', 'r+') as ds:
        if 'wtd' not in ds.variables:
            v = ds.createVariable('wtd', 'f8', ('lat', 'lon'), zlib=True)
        v = ds['wtd']
        v.units, v.long_name = 'm', ('equilibrium/annual-mean water table '
                                      'depth, Fan/Miguez-Macho (GFNL-USC)')
        v[:] = wtd
        h = ds['initial_head']
        if 'PLACEHOLDER' in h.ncattrs():
            h.delncattr('PLACEHOLDER')
        h.units, h.long_name = 'm', 'domain elev - wtd (m above sea level)'
        h.source = 'elevation of domain.nc minus Fan/Miguez-Macho WTD'
        h[:] = head
    for r in rows:
        log('CHECK ' + r)
    log('gwfields done')


if __name__ == '__main__':
    main()
