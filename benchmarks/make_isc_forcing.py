#!/usr/bin/env python3
"""The ISC benchmark forcing in VIC 5 image-driver files, from the daily Zarr.

    make_isc_forcing.py daily --years 2003 2004 2005 2006
    make_isc_forcing.py subdaily --first 2004-01-01 --last 2005-12-31 \\
        [--device cuda] [--cell-chunk 16384] [--no-write] [--timing FILE]

`daily` writes one NetCDF file a year, <data root>/forcing/vic5_daily/
isc_daily.YYYY.nc: the Zarr's eight daily fields as they are (tasmax,
tasmin degC; prcp mm; vp, sp kPa; wind m s-1; swdown, lwdown W m-2) on the
full 474 x 507 grid. TorchVIC reads these with FORCE_TYPE TMAX/TMIN/... and
runs MTCLIM on the device.

`subdaily` writes the 3-hourly files pristine VIC 5 reads,
<data root>/forcing/vic5_3h/isc_3h.YYYY.nc, by TorchVIC's own MTCLIM chain
(torchvic.forcing.daily.expand_forcing) over the daily files, with the same
90-day margin on each side that `torchvic.case.load` reads around a window
(case.MTCLIM_MARGIN_DAYS), cut back to --first..--last. The cells are
independent, so the chain runs over blocks of --cell-chunk cells.
Variables (VIC 5 image names and units): air_temp C, prec mm/step,
pressure kPa, swdown W/m2, lwdown W/m2, vp kPa, wind m/s, float32,
contiguous (one read per time slice), fill outside the domain mask.

--no-write times the MTCLIM step only (used for the CPU timing).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
ZARR = ROOT + '/forcing/era5_isc_daily_1940_2026.zarr'
DOMAIN = ROOT + '/vic_params_isc/domain.nc'
PARAMS = ROOT + '/vic_params_isc/params.nc'
DAILY = ROOT + '/forcing/vic5_daily/isc_daily.'
SUBDAILY = ROOT + '/forcing/vic5_3h/isc_3h.'
DAILY_VARS = ('tasmax', 'tasmin', 'prcp', 'vp', 'sp', 'wind', 'swdown',
              'lwdown')
# VIC FORCE_TYPE of each daily variable (the benchmark's global files say
# the same)
DAILY_TYPES = {'TMAX': 'tasmax', 'TMIN': 'tasmin', 'PREC': 'prcp',
               'VP': 'vp', 'PRESSURE': 'sp', 'WIND': 'wind',
               'SWDOWN': 'swdown', 'LWDOWN': 'lwdown'}
# 3-hourly output: name, TorchVIC field, scale from TorchVIC's unit, units
SUB_VARS = (('air_temp', 'tair', 1.0, 'C', 'AIR_TEMP'),
            ('prec', 'prcp', 1.0, 'mm/step', 'PREC'),
            ('pressure', 'press', 1e-3, 'kPa', 'PRESSURE'),
            ('swdown', 'shortwave', 1.0, 'W/m2', 'SWDOWN'),
            ('lwdown', 'longwave', 1.0, 'W/m2', 'LWDOWN'),
            ('vp', 'vp', 1e-3, 'kPa', 'VP'),
            ('wind', 'wind', 1.0, 'm/s', 'WIND'))
FILL = np.float32(9.96921e36)


def _coords(nc, lat, lon):
    nc.createDimension('lat', lat.size)
    nc.createDimension('lon', lon.size)
    v = nc.createVariable('lat', 'f8', ('lat',))
    v.units, v.standard_name, v.axis = 'degrees_north', 'latitude', 'Y'
    v[:] = lat
    v = nc.createVariable('lon', 'f8', ('lon',))
    v.units, v.standard_name, v.axis = 'degrees_east', 'longitude', 'X'
    v[:] = lon


def make_daily(years):
    import netCDF4
    import xarray as xr
    z = xr.open_zarr(ZARR)
    dom = xr.open_dataset(DOMAIN)
    mask = dom['mask'].values > 0
    lat, lon = z['lat'].values, z['lon'].values
    assert np.allclose(lat, dom['lat'].values) and \
        np.allclose(lon, dom['lon'].values), 'zarr and domain grids differ'
    os.makedirs(os.path.dirname(DAILY), exist_ok=True)
    for y in years:
        t0 = time.time()
        sel = z.sel(time=slice('%d-01-01' % y, '%d-12-31' % y))
        nt = sel.sizes['time']
        path = '%s%d.nc' % (DAILY, y)
        nc = netCDF4.Dataset(path + '.tmp', 'w', format='NETCDF4')
        nc.title = 'ISC daily forcing %d, VIC 5 image layout' % y
        nc.source = ('%s (ERA5, 1/16 deg), written by TorchVIC '
                     'benchmarks/make_isc_forcing.py' % os.path.basename(ZARR))
        nc.createDimension('time', nt)
        tv = nc.createVariable('time', 'i4', ('time',))
        tv.units = 'days since %d-01-01' % y
        tv.calendar = 'proleptic_gregorian'
        tv[:] = np.arange(nt, dtype=np.int32)
        _coords(nc, lat, lon)
        for k in DAILY_VARS:
            a = sel[k].values.astype(np.float32)
            bad = int((~np.isfinite(a[:, mask])).sum())
            if bad:
                raise ValueError('%s %d: %d non-finite values in the domain'
                                 % (k, y, bad))
            a = np.where(np.isfinite(a), a, FILL)
            v = nc.createVariable(k, 'f4', ('time', 'lat', 'lon'),
                                  fill_value=FILL, contiguous=True)
            for att in ('units', 'long_name'):
                v.setncattr(att, z[k].attrs[att])
            v.set_auto_maskandscale(False)
            v[:] = a
        nc.close()
        os.replace(path + '.tmp', path)
        print('wrote %s (%d days, %.0f s)' % (path, nt, time.time() - t0),
              flush=True)


def make_subdaily(first, last, device, cell_chunk, write, timing, margin):
    import torch
    import xarray as xr
    from torchvic.io.domain import ImageDomain
    from torchvic.io.forcing import ImageForcing
    from torchvic.forcing import daily as daily_forcing
    rec = dict(device=str(device), cell_chunk=cell_chunk, margin_days=margin)
    T = time.time()
    dom = ImageDomain(DOMAIN)
    with xr.open_dataset(PARAMS) as p:
        elev = dom.take(p['elev']).astype(np.float64)
    y0, y1 = int(first[:4]), int(last[:4])
    years = list(range(y0 - (1 if margin else 0), y1 + (1 if margin else 0)
                       + 1))
    fc = ImageForcing(DAILY, dom, force_type=DAILY_TYPES, years=years)
    t = time.time()
    fields, times = fc.read(dtype=torch.float64, device='cpu')
    rec['read_daily_s'] = time.time() - t
    a = int(np.searchsorted(times, np.datetime64(first)))
    b = int(np.searchsorted(times, np.datetime64(last))) + 1
    a0, b0 = max(a - margin, 0), min(b + margin, len(times))
    lead, n_win = a - a0, b - a
    spd = 8
    print('%d cells; days %s..%s, MTCLIM over %s..%s (%d days)'
          % (dom.n_active, times[a], times[b - 1], times[a0], times[b0 - 1],
             b0 - a0), flush=True)
    nc_ = dom.n_active
    nt = n_win * spd
    out = {f: np.empty((nt, nc_), np.float32) for _, f, _, _, _ in SUB_VARS}
    win = {k: v[a0:b0] for k, v in fields.items()}
    stamps = times[a0:b0]
    torch.cuda.synchronize() if str(device).startswith('cuda') else None
    t = time.time()
    t_dev = 0.0
    new_stamps = None
    for c0 in range(0, nc_, cell_chunk):
        c1 = min(c0 + cell_chunk, nc_)
        f = {k: v[:, c0:c1].to(device) for k, v in win.items()}
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        td = time.time()
        sub, st = daily_forcing.expand_forcing(
            f, stamps, spd, lat=np.asarray(dom.lat[c0:c1]),
            lng=np.asarray(dom.lon[c0:c1]), elev=elev[c0:c1],
            log=lambda *x: None)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        t_dev += time.time() - td
        s0, s1 = lead * spd, (lead + n_win) * spd
        new_stamps = st[s0:s1]
        for _, k, _, _, _ in SUB_VARS:
            out[k][:, c0:c1] = sub[k][s0:s1].to('cpu', torch.float32).numpy()
        del f, sub
        print('  cells %d..%d done (%.1f s)' % (c0, c1, time.time() - t),
              flush=True)
    rec['mtclim_s'] = time.time() - t
    rec['mtclim_compute_s'] = t_dev
    if str(device).startswith('cuda'):
        rec['mtclim_peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
        rec['gpu'] = torch.cuda.get_device_name(0)
    else:
        rec['torch_threads'] = torch.get_num_threads()
    rec['n_cells'], rec['n_days'], rec['n_days_mtclim'] = nc_, n_win, b0 - a0
    if write:
        t = time.time()
        write_subdaily(out, new_stamps, dom)
        rec['write_s'] = time.time() - t
    rec['total_s'] = time.time() - T
    print(json.dumps(rec, indent=1), flush=True)
    if timing:
        with open(timing, 'w') as fh:
            json.dump(rec, fh, indent=1)


def write_subdaily(out, stamps, dom):
    import netCDF4
    os.makedirs(os.path.dirname(SUBDAILY), exist_ok=True)
    lat = np.asarray(dom.ds['lat'].values)
    lon = np.asarray(dom.ds['lon'].values)
    years = stamps.astype('datetime64[Y]').astype(int) + 1970
    for y in np.unique(years):
        idx = np.nonzero(years == y)[0]
        assert str(stamps[idx[0]])[5:19] == '01-01T00:00:00', stamps[idx[0]]
        path = '%s%d.nc' % (SUBDAILY, y)
        t0 = time.time()
        nc = netCDF4.Dataset(path + '.tmp', 'w', format='NETCDF4')
        nc.title = ('ISC 3-hourly forcing %d, VIC 5 image layout' % y)
        nc.source = ('isc_daily.*.nc disaggregated by TorchVIC\'s MTCLIM chain '
                     '(torchvic.forcing.daily.expand_forcing, 8 steps a day, '
                     '90-day margins), benchmarks/make_isc_forcing.py')
        nc.createDimension('time', idx.size)
        tv = nc.createVariable('time', 'i4', ('time',))
        tv.units = 'hours since %d-01-01' % y
        tv.calendar = 'proleptic_gregorian'
        tv[:] = (3 * np.arange(idx.size)).astype(np.int32)
        _coords(nc, lat, lon)
        for name, k, scale, units, ftype in SUB_VARS:
            grid = np.full((idx.size, dom.ny, dom.nx), FILL, np.float32)
            vals = out[k][idx]
            if scale != 1.0:
                vals = (vals.astype(np.float64) * scale).astype(np.float32)
            grid[:, dom.rows, dom.cols] = vals
            v = nc.createVariable(name, 'f4', ('time', 'lat', 'lon'),
                                  fill_value=FILL, contiguous=True)
            v.units, v.long_name = units, ftype
            v.set_auto_maskandscale(False)
            v[:] = grid
            del grid
        nc.close()
        os.replace(path + '.tmp', path)
        print('wrote %s (%d steps, %.0f s)' % (path, idx.size,
                                              time.time() - t0), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('daily')
    d.add_argument('--years', type=int, nargs='+', required=True)
    s = sub.add_parser('subdaily')
    s.add_argument('--first', default='2004-01-01')
    s.add_argument('--last', default='2005-12-31')
    s.add_argument('--device', default='cuda')
    s.add_argument('--cell-chunk', type=int, default=16384)
    s.add_argument('--margin-days', type=int, default=90)
    s.add_argument('--no-write', action='store_true')
    s.add_argument('--timing', default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'daily':
        make_daily(a.years)
    else:
        make_subdaily(a.first, a.last, a.device, a.cell_chunk,
                      not a.no_write, a.timing, a.margin_days)


if __name__ == '__main__':
    sys.exit(main())
