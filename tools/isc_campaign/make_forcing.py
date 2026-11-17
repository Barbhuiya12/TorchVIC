"""Spin-up forcing for the ISC campaign: met.2004.nc (as tools/vicwur/isc/smoke_met.py)
and efr_q/efr_b.2004.nc (the 2005 placeholder series, time axis re-based), in
<data root>/verification/isc_campaign/forcing, next to links to the 2005/2006 files."""
import os
import shutil
import sys
import netCDF4 as nc
import numpy as np
R = os.environ.get('TORCHVIC_DATA_ROOT', 'data').rstrip('/') + '/'
SM = R + 'vicwur/isc/smoke/forcing/'
OUT = R + 'verification/isc_campaign/forcing/'
os.makedirs(OUT, exist_ok=True)
for f in ('met.2005.nc', 'met.2006.nc', 'efr_q.2005.nc', 'efr_b.2005.nc'):
    if not os.path.lexists(OUT + f):
        os.symlink(SM + f, OUT + f)
for k in ('efr_q', 'efr_b'):
    dst = OUT + k + '.2004.nc'
    shutil.copy(SM + k + '.2005.nc', dst)
    with nc.Dataset(dst, 'a') as d:
        d['time'].units = 'days since 2004-01-01'
        t = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
        d['time'][:] = t
src = R + 'forcing/vic5_daily/isc_daily.2004.nc'
dst = OUT + 'met.2004.nc'
with nc.Dataset(src) as s, nc.Dataset(dst, 'w') as d:
    for dim in ('time', 'lat', 'lon'):
        d.createDimension(dim, len(s.dimensions[dim]))
        v = d.createVariable(dim, s[dim].dtype, (dim,))
        v.setncatts({k: s[dim].getncattr(k) for k in s[dim].ncattrs()})
        v[:] = s[dim][:]
    for k in ('tair', 'prcp', 'vp', 'sp', 'wind', 'swdown', 'lwdown'):
        if k == 'tair':
            a = 0.5 * (s['tasmax'][:] + s['tasmin'][:])
            units = 'degC'
        else:
            a = s[k][:]
            units = s[k].getncattr('units') if 'units' in s[k].ncattrs() else ''
        o = d.createVariable(k, 'f4', ('time', 'lat', 'lon'),
                             fill_value=np.float32(1e20), zlib=True)
        o.units = units
        o[:] = np.ma.asarray(a).astype(np.float32)
        print(k, 'ok', flush=True)
