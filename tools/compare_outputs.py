#!/usr/bin/env python3
"""compare_outputs.py A B : every variable of every netCDF file in two
directories (or two files) array_equal (NaN equal), masks included."""
import glob
import os
import sys

import numpy as np
import netCDF4


def files(p):
    return [p] if os.path.isfile(p) else sorted(glob.glob(os.path.join(p, '*.nc')))


def main(a, b):
    fa, fb = files(a), files(b)
    if [os.path.basename(x) for x in fa] != [os.path.basename(x) for x in fb]:
        print('FILE LISTS DIFFER', fa, fb)
        return 1
    bad = n = 0
    for x, y in zip(fa, fb):
        with netCDF4.Dataset(x) as da, netCDF4.Dataset(y) as db:
            if list(da.variables) != list(db.variables):
                print('VARIABLES DIFFER in', x)
                bad += 1
                continue
            for k in da.variables:
                va, vb = da[k], db[k]
                va.set_auto_maskandscale(False)
                vb.set_auto_maskandscale(False)
                same = np.array_equal(np.asarray(va[:]), np.asarray(vb[:]),
                                      equal_nan=np.asarray(va[:]).dtype.kind == 'f')
                n += 1
                if not same:
                    bad += 1
                    print('DIFF', os.path.basename(x), k)
    print('%d variables compared, %d differ' % (n, bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:3]))
