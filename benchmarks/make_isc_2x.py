#!/usr/bin/env python3
"""A domain twice the ISC size, for the multi-GPU scaling test: the ISC grid
tiled twice along x (474 x 1014, 225,360 active cells), its parameters and
the first DAYS days of the 2004 3-hourly forcing tiled the same way, so that
every cell of the right half is a copy of the cell 507 columns to its left
(the same physics, a different place). Writes domain.nc, params.nc and
isc2x_3h.2004.nc in OUT; benchmarks/isc/isc_vic5_3h.txt with these paths and
the first DAYS days is the matching global file.

    make_isc_2x.py OUT [--days 62]
"""
import argparse
import os

import numpy as np
import xarray as xr

ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')


def tile_x(ds):
    """Every variable with a lon dimension, repeated along it; lon continues
    with the same spacing."""
    lon = ds['lon'].values
    dx = float(lon[1] - lon[0])
    lon2 = np.concatenate([lon, lon + dx * len(lon)])
    out = {}
    for k, v in ds.data_vars.items():
        if 'lon' in v.dims:
            ax = v.dims.index('lon')
            out[k] = (v.dims, np.concatenate([v.values, v.values], axis=ax),
                      v.attrs)
        else:
            out[k] = v
    coords = {c: ds[c] for c in ds.coords if 'lon' not in ds[c].dims}
    coords['lon'] = ('lon', lon2, ds['lon'].attrs)
    return xr.Dataset(out, coords=coords, attrs=ds.attrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--days', type=int, default=62)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    enc = dict(zlib=True, complevel=1, shuffle=True)
    for name in ('domain.nc', 'params.nc'):
        ds = xr.open_dataset(os.path.join(ROOT, 'vic_params_isc', name))
        d2 = tile_x(ds)
        e = {k: dict(v.encoding, **enc) if v.ndim >= 2 else {}
             for k, v in d2.data_vars.items()}
        for k in e:
            e[k].pop('chunksizes', None)
            e[k].pop('original_shape', None)
            e[k].pop('source', None)
        d2.to_netcdf(os.path.join(a.out, name), encoding=e)
        print('wrote', name, flush=True)
    ds = xr.open_dataset(os.path.join(ROOT, 'forcing/vic5_3h/isc_3h.2004.nc'))
    ds = ds.isel(time=slice(0, a.days * 8))
    d2 = tile_x(ds)
    e = {k: {'dtype': 'float32'} for k in d2.data_vars}
    d2.to_netcdf(os.path.join(a.out, 'isc2x_3h.2004.nc'), encoding=e)
    print('wrote forcing', flush=True)


if __name__ == '__main__':
    main()
