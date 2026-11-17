"""Campaign test copies of the yearly Cv forcing with the -1e-16 rounding
negatives set to 0 (and every cell renormalised to 1)."""
import os
import shutil
import sys
import netCDF4 as nc
import numpy as np
R = os.environ.get('TORCHVIC_DATA_ROOT', 'data').rstrip('/') + '/'
for y in map(int, sys.argv[1:]):
    dst = R + 'verification/isc_campaign/forcing/cv.%d.nc' % y
    shutil.copy(R + 'vicwur/isc/forcing/cv.%d.nc' % y, dst)
    with nc.Dataset(dst, 'a') as d:
        c = np.ma.filled(d['cv'][:], 0.0)
        c = np.where(c < 0, 0.0, c)
        s = c.sum(1, keepdims=True)
        c = np.where(s > 0, c / np.where(s > 0, s, 1), c)
        d['cv'][:] = c
    print(y, 'ok')
