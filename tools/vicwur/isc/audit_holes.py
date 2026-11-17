"""Hole audit of ISC outputs: NaN / fill / all-zero fields in active cells."""
import os
import glob, sys
import numpy as np, netCDF4 as nc
A = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
with nc.Dataset(A + '/vic_params_isc/domain.nc') as d:
    act = np.ma.filled(d['mask'][:], 0) > 0
print('active', int(act.sum()))
files = sorted(glob.glob(A + '/vicwur/isc/params/*.nc') + glob.glob(A + '/vicwur/isc/dams/*.nc')
               + glob.glob(A + '/vicwur/isc/wofost/*.nc') + glob.glob(A + '/vicwur/isc/forcing/*.2005.nc')
               + glob.glob(A + '/vicwur/isc/forcing/*.2000.nc') + glob.glob(A + '/vicwur/isc/forcing/*.2010.nc'))
for f in files:
    with nc.Dataset(f) as d:
        for n, v in d.variables.items():
            if v.ndim < 2 or v.dimensions[-2:] != ('lat', 'lon'):
                continue
            a = v[:]
            fv = getattr(v, '_FillValue', None)
            m = np.ma.getmaskarray(a)
            x = np.ma.filled(a, 0).astype(float)
            a2 = x.reshape(-1, *act.shape)
            m2 = m.reshape(-1, *act.shape)
            nan = (np.isnan(a2) | m2)[:, act].sum()
            big = (np.abs(a2[:, act]) > 1e30).sum()
            zero = (a2[:, act] == 0).all(0).sum() if a2.shape[0] > 1 else (a2[0][act] == 0).sum()
            zero_cells = int(zero) if a2.shape[0] > 1 else int(zero)
            if nan or big or zero_cells > 0.3 * act.sum():
                print('%s %s nan/mask=%d big=%d allzero_cells=%d' % (f.split('isc/')[1], n, nan, big, zero_cells))
