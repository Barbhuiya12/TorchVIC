#!/usr/bin/env python
"""CO2 forcing (C10): NOAA GML Mauna Loa annual mean, ppm, uniform over the
ISC grid, one record per year: co2.YYYY.nc, var `co2`
(`PLUGIN_FORCE_TYPE CO2 co2 YEAR <dir>/co2.`)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *   # noqa


def main():
    g = ISC()
    os.makedirs(OUT + '/forcing', exist_ok=True)
    tab = {}
    for ln in open(RAW + '/co2_noaa_mlo/co2_annmean_mlo.txt'):
        if ln.startswith('#') or not ln.strip():
            continue
        w = ln.split()
        tab[int(w[0])] = float(w[1])
    vals = []
    for y in YEARS:
        a = np.full((1, g.ny, g.nx), tab[y])
        yearly_file('%s/forcing/co2.%d.nc' % (OUT, y), y, g, 'co2', a, 'ppm',
                    [0], attrs={'source': 'NOAA GML Mauna Loa annual mean'})
        vals.append(tab[y])
    v = np.array(vals)
    log('CHECK co2 ppm min %.2f mean %.2f max %.2f NaN 0 (uniform field)' % (v.min(), v.mean(), v.max()))
    log('co2 done')


if __name__ == '__main__':
    main()
