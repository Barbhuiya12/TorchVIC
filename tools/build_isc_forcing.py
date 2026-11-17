#!/usr/bin/env python3
"""Assemble the daily ERA5 partials into the ISC daily forcing Zarr.

Input: the per-chunk partial daily reductions written by edh_era5_daily.py
(OUT/main/<field>/<field>_NNNN.npz, 0.25 deg ISC box). Days split across two
time chunks are completed here (sums add, maxima take the max, minima the
min); only days with all 24 hours are kept.

Output: one Zarr store with the variable names and units of the original
ISC store (tasmax, tasmin degC; prcp mm; vp, sp kPa; wind m s-1;
swdown, lwdown W m-2) on the 1/16 deg grid extended south to 8.03125 N
(474 x 507), bilinear from 0.25 deg as the original store was made.

    python build_isc_forcing.py --parts DIR --out STORE.zarr \\
        --first 1940-01-01 --last 2026-05-30
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import xarray as xr
import zarr

FIELDS = {  # name in store: (partial dir, finish, units, long name, combine)
    'tasmax': ('tmax', lambda v: v - 273.15, 'degC',
               'daily maximum of hourly 2 m temperature (UTC day)', np.fmax),
    'tasmin': ('tmin', lambda v: v - 273.15, 'degC',
               'daily minimum of hourly 2 m temperature (UTC day)', np.fmin),
    'prcp': ('prcp', lambda v: v * 1000.0, 'mm',
             'daily total precipitation (UTC day)', np.add),
    'vp': ('vp', lambda v: v / 24.0, 'kPa',
           'daily mean of hourly vapour pressure from 2 m dewpoint (Tetens)', np.add),
    'sp': ('sp', lambda v: v / 24.0 / 1000.0, 'kPa',
           'daily mean surface pressure', np.add),
    'wind': ('wind', lambda v: v / 24.0, 'm s-1',
             'daily mean of hourly 10 m wind speed', np.add),
    'swdown': ('sw', lambda v: v / 86400.0, 'W m-2',
               'daily mean surface downward shortwave radiation', np.add),
    'lwdown': ('lw', lambda v: v / 86400.0, 'W m-2',
               'daily mean surface downward longwave radiation', np.add),
}
CHUNKS = (366, 79, 169)
LAT = np.round(8.03125 + 0.0625 * np.arange(474), 5)
LON = np.round(66.15625 + 0.0625 * np.arange(507), 5)

ATTRS = {
    'title': 'ISC daily forcing from ERA5, 1/16 deg',
    'source': 'ERA5 hourly single levels (Hersbach et al. 2020) via DestinE '
              'Earth Data Hub, reanalysis-era5-single-levels-v0.zarr',
    'method': 'UTC-day statistics of hourly ERA5 at 0.25 deg, bilinear to the '
              '1/16 deg cell centres (tasmax/tasmin: max/min of hourly t2m)',
    'grid': '474 x 507 cells, 8.03125-37.59375 N, 66.15625-97.78125 E '
            '(VICGlobal 1/16 deg grid, zero offset)',
    'replaces': 'forcings.zarr, whose vp and sp are scrambled from 1990-91, '
                'whose swdown/lwdown are not in W m-2 in any year, and which '
                'stops at 8.906 N',
}


def daily_series(parts, sub, combine, first, last):
    days = np.arange(np.datetime64(first), np.datetime64(last) + 1)
    nd = len(days)
    acc = cnt = lat = lon = None
    for f in sorted(glob.glob(os.path.join(parts, sub, '*.npz'))):
        z = np.load(f)
        d = z['days'].astype('datetime64[D]')
        k = ((d - days[0]) / np.timedelta64(1, 'D')).astype(int)
        m = (k >= 0) & (k < nd)
        if not m.any():
            continue
        if acc is None:
            lat, lon = z['lat'], z['lon']
            fill = -np.inf if combine is np.fmax else np.inf if combine is np.fmin else 0.0
            acc = np.full((nd, len(lat), len(lon)), fill, np.float64)
            cnt = np.zeros(nd, np.int16)
        acc[k[m]] = combine(acc[k[m]], z['part'][m])
        cnt[k[m]] += z['count'][m]
    bad = cnt != 24
    return days, acc, bad, lat, lon


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--parts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--first', default='1940-01-01')
    ap.add_argument('--last', default='2026-05-30')
    ap.add_argument('--block', type=int, default=366, help='days per write')
    args = ap.parse_args()
    parts = os.path.join(args.parts, 'main')
    for i, (name, (sub, finish, units, long, combine)) in enumerate(FIELDS.items()):
        t0 = time.time()
        days, acc, bad, lat, lon = daily_series(parts, sub, combine,
                                                args.first, args.last)
        if bad.any():
            raise SystemExit('%s: %d incomplete days, first %s' % (
                name, bad.sum(), days[bad][0]))
        coarse = xr.DataArray(finish(acc).astype(np.float32),
                              coords={'time': days.astype('datetime64[ns]'),
                                      'lat': lat, 'lon': lon},
                              dims=('time', 'lat', 'lon'))
        coarse = coarse.sortby('lat')
        del acc
        for b0 in range(0, len(days), args.block):
            blk = coarse.isel(time=slice(b0, b0 + args.block)).interp(
                lat=LAT, lon=LON, method='linear').astype(np.float32)
            if i == 0:
                # the first field sets up the store: coordinates and time axis
                ds = blk.to_dataset(name=name)
                ds[name].attrs = {'units': units, 'long_name': long}
                if b0 == 0:
                    ds.attrs = ATTRS
                    ds.to_zarr(args.out, mode='w', zarr_format=2,
                               encoding={name: {'chunks': CHUNKS}})
                else:
                    ds.to_zarr(args.out, append_dim='time', zarr_format=2)
            else:
                # later fields: a full-size array, filled block by block
                if b0 == 0:
                    g = zarr.open_group(args.out, mode='a', zarr_format=2)
                    arr = g.create_array(name, shape=(len(days), len(LAT), len(LON)),
                                         chunks=CHUNKS, dtype='float32',
                                         fill_value=np.nan, overwrite=True)
                    arr.attrs.update({'_ARRAY_DIMENSIONS': ['time', 'lat', 'lon'],
                                      'units': units, 'long_name': long})
                arr[b0:b0 + blk.shape[0]] = blk.values
        print('%s %s: %d days written in %.0f s' % (
            time.strftime('%H:%M:%S'), name, len(days), time.time() - t0), flush=True)
    zarr.consolidate_metadata(args.out, zarr_format=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
