#!/usr/bin/env python
"""Per-year crop nc for ISC: Cc and the crop calendar follow the yearly Cv.

A crop unit exists, and so sows, in a year only if the Cv of its veg tile is
> 0 that year.  VIC-WUR: crop_init.c:76-100 reads Cc and builds the crop tile
map from Cc (a crop with Cc sum 0 over the months has no tile and never runs);
crop_run.c:296-323 stops the run when a crop grows or sows with Cc <= 0 or
veg_con Cv <= 0; lu_force.c:83-88 takes Cv per year from the forcing.
Cc(y) = share of the crop in its veg tile (Cc / sum Cc of the tile in the base
nc, MIRCA-OS 2000 harvested-area shares) x Cv(y) of the tile, zero where
Cv(y) = 0 (cells whose base Cc is ~0 take the shares of the nearest cell with crops; unit absent: plant_day/harvest_day of an absent unit are never
read).  Ncrop(y) is recounted.  Base file untouched.
Usage: make_crop_yearly.py [year ...]"""
import os
import sys
import numpy as np
import netCDF4 as nc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa
from regrid import fill_nearest   # noqa

W = OUT + '/wofost'
BASE = W + '/crop_params_VICWOFOST_isc.nc'


def main(years):
    with nc.Dataset(BASE) as b:
        cc = np.ma.filled(b['Cc'][0], 0.0).astype(float)          # [NC,y,x]
        vc = np.ma.filled(b['crop_veg_class'][:], 0).astype(int)
    vegs = vc.reshape(vc.shape[0], -1).max(1)
    shares = np.zeros_like(cc)
    for v in (13, 18, 19):
        idx = np.where(vegs == v)[0]
        t = cc[idx].sum(0)
        sh = np.where(t > 0, cc[idx] / np.where(t > 0, t, 1), np.nan)
        # tile exists in a later year but the base Cc is ~0 there (params Cv is
        # rounding dust): shares of the nearest cell with crops, as make_wofost
        shares[idx] = np.stack([fill_nearest(sh[k], valid=(t > 0)) for k in range(len(idx))])
    for y in years:
        with nc.Dataset('%s/forcing/cv.%d.nc' % (OUT, y)) as d:
            cv = np.ma.filled(d['cv'][0], 0.0).astype(float)
        c = np.zeros_like(cc)
        for v in (13, 18, 19):
            idx = np.where(vegs == v)[0]
            c[idx] = shares[idx] * np.where(cv[v - 1] > 0, cv[v - 1], 0.0)[None]
        c[c < 1e-12] = 0.0
        f = '%s/crop_params_VICWOFOST_isc.%d.nc' % (W, y)
        with nc.Dataset(BASE) as b, nc.Dataset(f, 'w') as o:
            for n_, L in b.dimensions.items():
                o.createDimension(n_, len(L))
            for n_, v in b.variables.items():
                fill = getattr(v, '_FillValue', None)
                x = o.createVariable(n_, v.dtype, v.dimensions, fill_value=fill,
                                     zlib=v.ndim >= 3, complevel=2)
                for a in v.ncattrs():
                    if a != '_FillValue':
                        x.setncattr(a, v.getncattr(a))
                if n_ == 'Cc':
                    x[:] = np.broadcast_to(c, (12,) + c.shape)
                    x.comment = ('MIRCA-OS 2000 shares x Cv(%d) of the veg tile; '
                                 '0 where Cv = 0 (no crop unit); constant over months' % y)
                elif n_ == 'Ncrop':
                    x[:] = (c > 0).sum(0).astype('i4')
                else:
                    x[:] = b[n_][:]
            o.title = b.title + ', year %d' % y
        bad = sum(int(((c[vegs == v] > 0).any(0) & (cv[v - 1] <= 0)).sum())
                  for v in (13, 18, 19))
        log('CHECK crop yearly %d: units with Cc>0 on Cv=0 tiles: %d; '
            'sum|Cc-Cv| max (tiles with a crop) %.2e' % (
                y, bad, max(np.abs(c[vegs == v].sum(0) - cv[v - 1]).max()
                            for v in (13, 18, 19))))


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]] or YEARS)
