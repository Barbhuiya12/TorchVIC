"""ISC smoke: params_wur.nc has vegetated tiles (Cv > 0) without root
zones (e.g. class 14 where ISC params had none), which VIC-WUR 3.0's
calc_root_fraction refuses (calc_root_fraction.c:173-178). Copy the file to
<data root>/vicwur/isc/smoke/params_wur_roots.nc and give such tiles the most
common root profile of their class (else of the cell's dominant vegetated
class, else 0.1/0.6/0.8 m, 0.33/0.55/0.12). A smoke-test input only."""
import os
import shutil
import collections
import netCDF4 as nc
import numpy as np
SRC = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'vicwur/isc/params/params_wur.nc')
DST = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'vicwur/isc/smoke/params_wur_roots.nc')
shutil.copyfile(SRC, DST)
with nc.Dataset(DST, 'a') as d:
    cv = np.ma.filled(d['Cv'][:], 0.0)
    rd = np.ma.filled(d['root_depth'][:], 0.0)
    rf = np.ma.filled(d['root_fract'][:], 0.0)
    nveg = cv.shape[0]
    bare = nveg - 1                      # PBARE = NVEGTYPES (README)
    default = (np.array([0.1, 0.6, 0.8]), np.array([0.33, 0.55, 0.12]))
    n = 0
    for v in range(nveg):
        if v == bare:
            continue
        has = rf[v].sum(0) > 0
        need = (cv[v] > 0) & ~has
        if not need.any():
            continue
        if has.any():
            prof = collections.Counter(
                tuple(np.r_[rd[v][:, has][:, k], rf[v][:, has][:, k]])
                for k in range(0, int(has.sum()), max(1, int(has.sum()) // 5000)))
            p = np.array(prof.most_common(1)[0][0])
            z, f = p[:3], p[3:]
        else:
            z, f = default
        rd[v][:, need] = z[:, None]
        rf[v][:, need] = f[:, None]
        n += int(need.sum())
        print('class', v + 1, 'tiles', int(need.sum()), z, f)
    d['root_depth'][:] = rd
    d['root_fract'][:] = rf
    d.setncattr('smoke_roots', 'vegetated tiles without root zones given '
                'their class profile (tools/vicwur/isc/smoke_params.py)')
print('filled', n)
