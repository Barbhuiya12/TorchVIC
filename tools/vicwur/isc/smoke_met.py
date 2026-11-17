"""ISC smoke: the VIC-WUR image driver has no TMAX/TMIN (as R1,
tools/vicwur/ref_domains.py): met.YYYY.nc with tair = (tasmax + tasmin)/2
and the other daily fields of <data root>/forcing/vic5_daily/isc_daily.YYYY."""
import os
import sys
import netCDF4 as nc
import numpy as np
src = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'forcing/vic5_daily/isc_daily.%d.nc')
dst = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'vicwur/isc/smoke/forcing/met.%d.nc')
for y in map(int, sys.argv[1:]):
    with nc.Dataset(src % y) as s, nc.Dataset(dst % y, 'w') as d:
        for dim in ('time', 'lat', 'lon'):
            d.createDimension(dim, len(s.dimensions[dim]))
            v = d.createVariable(dim, s[dim].dtype, (dim,))
            v.setncatts({k: s[dim].getncattr(k) for k in s[dim].ncattrs()})
            v[:] = s[dim][:]
        for k in ('tair', 'prcp', 'vp', 'sp', 'wind', 'swdown', 'lwdown'):
            if k == 'tair':
                a = 0.5 * (s['tasmax'][:] + s['tasmin'][:])
                units = s['tasmax'].getncattr('units') if 'units' in s['tasmax'].ncattrs() else 'degC'
            else:
                a = s[k][:]
                units = s[k].getncattr('units') if 'units' in s[k].ncattrs() else ''
            o = d.createVariable(k, 'f4', ('time', 'lat', 'lon'), fill_value=np.float32(1e20), zlib=True)
            o.units = units
            o[:] = np.ma.asarray(a).astype(np.float32)
        print(y, 'ok')
