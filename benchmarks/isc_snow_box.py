#!/usr/bin/env python3
"""Cut a small window of the ISC benchmark (domain, parameters, 3-hourly
forcing) for VIC-vs-TorchVIC snow diagnosis and tests/test_isc_snow_vs_vic51.

    isc_snow_box.py [--rows 358:368] [--cols 385:395] [--out DIR]

The default window is 10 x 10 cells on the Tibetan plateau by Nam Co
(30.4-31.0 N, 90.2-90.8 E), grassland, barren and open water under a
seasonal pack: the cells where TorchVIC disagreed with VIC 5.1.0 most in
the full ISC run. Output: DIR/domain.nc, params.nc, isc_3h.2004.nc,
isc_3h.2005.nc (default <data root>/benchmark_isc/snow_box).
"""
import argparse
import os

import xarray as xr

ALL = os.environ.get('TORCHVIC_DATA_ROOT', 'data')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', default='358:368')
    ap.add_argument('--cols', default='385:395')
    ap.add_argument('--out', default=os.path.join(ALL, 'benchmark_isc',
                                                  'snow_box'))
    a = ap.parse_args(argv)
    r0, r1 = map(int, a.rows.split(':'))
    c0, c1 = map(int, a.cols.split(':'))
    sl = dict(lat=slice(r0, r1), lon=slice(c0, c1))
    os.makedirs(a.out, exist_ok=True)
    kw = dict(mask_and_scale=False, decode_times=False)
    for f in ('domain', 'params'):
        with xr.open_dataset(os.path.join(ALL, 'vic_params_isc', f + '.nc'),
                             **kw) as d:
            d.isel(**sl).load().to_netcdf(os.path.join(a.out, f + '.nc'))
    for y in (2004, 2005):
        with xr.open_dataset(os.path.join(ALL, 'forcing', 'vic5_3h',
                                          'isc_3h.%d.nc' % y), **kw) as d:
            d.isel(**sl).load().to_netcdf(
                os.path.join(a.out, 'isc_3h.%d.nc' % y))
        print('wrote', os.path.join(a.out, 'isc_3h.%d.nc' % y))


if __name__ == '__main__':
    main()
