"""Apply make_landuse.fill_roots to an existing params_wur.nc (same result as
regenerating, without redoing the regrids).  Usage: fix_roots_inplace.py FILE"""
import sys
import numpy as np
import netCDF4 as nc
import make_landuse as m
with nc.Dataset(sys.argv[1], 'a') as d:
    cv = np.ma.filled(d['Cv'][:], 0.0)
    m.fill_roots(d, cv)
    d.setncattr('roots_fixed', 'fill_roots (tools/vicwur/isc/make_landuse.py)')
