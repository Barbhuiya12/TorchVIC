#!/usr/bin/env python3
"""Cut the five ISC verification windows (domain, parameters, 3-hourly
forcing for 2004) out of the full Indian-subcontinent inputs.

    make_isc_windows.py [--out DIR] [--years 2004]

Each window is 6 x 6 cells of the 1/16 deg ISC grid (rows/cols are indices
into <data root>/vic_params_isc/domain.nc):

  igp     Indo-Gangetic plain, 26.0 N 82.0 E: cropland, monsoon, 92-102 m
  thar    Thar desert, 27.0 N 71.0 E: open shrub / barren, 187 mm/yr
  himal   Gangotri, 30.9 N 79.0 E: grass / barren / snow-ice, 3,450-5,600 m,
          4-5 elevation bands per cell
  forest  Anamalai, Western Ghats, 10.5 N 76.9 E: ~50 % forest, overstory
  tntip   Tamil Nadu / Kerala tip, 8.3 N 77.4 E: the domain's southern edge

The forcing files are contiguous netCDF of ~20 GB a year; run this on a
compute node (verification/run_all.sh does), not on the login node.
Output: DIR/<name>/{domain,params}.nc and isc_3h.<year>.nc
(default <data root>/verification/windows).
"""
import argparse
import os

import xarray as xr

ALL = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
WINDOWS = {
    'igp': (284, 250),
    'thar': (300, 74),
    'himal': (363, 202),
    'forest': (36, 169),
    'tntip': (1, 177),
}
SIZE = 6


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(ALL, 'verification',
                                                  'windows'))
    ap.add_argument('--years', default='2004')
    a = ap.parse_args(argv)
    kw = dict(mask_and_scale=False, decode_times=False)
    sls = {k: dict(lat=slice(r, r + SIZE), lon=slice(c, c + SIZE))
           for k, (r, c) in WINDOWS.items()}
    for name in WINDOWS:
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
    for f in ('domain', 'params'):
        with xr.open_dataset(os.path.join(ALL, 'vic_params_isc', f + '.nc'),
                             **kw) as d:
            for name, sl in sls.items():
                d.isel(**sl).load().to_netcdf(
                    os.path.join(a.out, name, f + '.nc'))
    for y in [int(x) for x in a.years.split(',')]:
        src = os.path.join(ALL, 'forcing', 'vic5_3h', 'isc_3h.%d.nc' % y)
        with xr.open_dataset(src, **kw) as d:
            for name, sl in sls.items():
                dst = os.path.join(a.out, name, 'isc_3h.%d.nc' % y)
                if os.path.exists(dst):
                    continue
                d.isel(**sl).load().to_netcdf(dst + '.part')
                os.replace(dst + '.part', dst)
                print('wrote', dst, flush=True)


if __name__ == '__main__':
    main()
