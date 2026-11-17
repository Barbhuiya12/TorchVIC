"""Shared paths, ISC grid, logging and the per-variable sanity table."""
import os
import sys
import time
import numpy as np
import netCDF4 as nc

VT = os.environ.get('TORCHVIC_WORK_ROOT', '.')
DATA_ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
ROOT = DATA_ROOT
RAW = ROOT + '/raw/vicwur'
WORK = ROOT + '/vicwur/isc_work'
OUT = ROOT + '/vicwur/isc'
DOMAIN = ROOT + '/vic_params_isc/domain.nc'
PARAMS = ROOT + '/vic_params_isc/params.nc'
FDIR = ROOT + '/static/isc_fdir.nc'
LOG = os.environ.get('VICWUR_LOG', ROOT + '/verification/vicwur_52b/run.log')
FILL = -1.0e30
YEARS = list(range(2000, 2011))


def log(msg):
    line = '%s %s' % (time.strftime('%Y-%m-%dT%H:%M:%S'), msg)
    print(line, flush=True)
    try:
        with open(LOG, 'a') as f:
            f.write(line + '\n')
    except OSError:
        pass


class ISC:
    def __init__(self):
        with nc.Dataset(DOMAIN) as d:
            self.lat = np.array(d['lat'][:])
            self.lon = np.array(d['lon'][:])
            self.mask = np.array(d['mask'][:]).astype(bool)
            self.area = np.array(d['area'][:], float)
        with nc.Dataset(PARAMS) as p:
            self.run = np.array(p['run_cell'][:]) > 0
        self.active = self.mask & self.run
        self.ny, self.nx = self.mask.shape


def stats_row(name, a, mask, unit=''):
    v = np.asarray(a, float)
    v = v[..., mask] if v.shape[-2:] == mask.shape else v.ravel()
    nan = int(np.isnan(v).sum())
    f = v[np.isfinite(v)]
    return '%-34s %-9s min %-12.5g mean %-12.5g max %-12.5g NaN %d n %d' % (
        name, unit, f.min(), f.mean(), f.max(), nan, v.size)


def coords(ds, g):
    ds.createDimension('lat', g.ny)
    ds.createDimension('lon', g.nx)
    v = ds.createVariable('lat', 'f8', ('lat',))
    v.units, v.long_name = 'degrees_north', 'latitude of grid cell center'
    v[:] = g.lat
    v = ds.createVariable('lon', 'f8', ('lon',))
    v.units, v.long_name = 'degrees_east', 'longitude of grid cell center'
    v[:] = g.lon


def month_starts(y):
    dm = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
          31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return np.concatenate([[0], np.cumsum(dm)[:-1]]), np.array(dm)


def yearly_file(path, y, g, vname, data, units, times, extra=None,
                attrs=None):
    """time x [extra] x lat x lon f8 file, time 'days since y-01-01'."""
    with nc.Dataset(path, 'w') as ds:
        coords(ds, g)
        ds.createDimension('time', len(times))
        v = ds.createVariable('time', 'i4', ('time',))
        v.units = 'days since %d-01-01' % y
        v.calendar = 'proleptic_gregorian'
        v[:] = times
        dims = ('time',) + ((extra[0],) if extra else ()) + ('lat', 'lon')
        if extra:
            ds.createDimension(extra[0], extra[1])
        o = ds.createVariable(vname, 'f8', dims, fill_value=FILL,
                              zlib=True, complevel=2)
        o.units = units
        for k, val in (attrs or {}).items():
            o.setncattr(k, val)
        o[:] = data
