"""A ~500-cell sub-domain of ISC (a box on the Ganga near Allahabad) for the
device comparison: domain.nc with the mask cut to the box, rout_n4.nc with the
cells whose downstream cell is outside the box turned into outlets, and the
groundwater statics of the box's cells. Files in isc_campaign/subset500/."""
import os
import shutil
import sys

import netCDF4 as nc
import numpy as np

R = os.environ.get('TORCHVIC_DATA_ROOT', 'data').rstrip('/') + '/'
OUT = R + 'verification/isc_campaign/subset500/'
BOX = dict(lat=(25.0, 26.4), lon=(81.0, 82.4))


def build(box=BOX, out=OUT, params=R + 'vicwur/isc/smoke/params_wur_roots.nc'):
    """The box's sub-domain in `out` (default: the Ganga box, subset500/);
    `params` is the full VIC parameter file whose run_cell is cut."""
    BOX, OUT = box, out
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(R + 'vic_params_isc/domain.nc', OUT + 'domain.nc')
    with nc.Dataset(OUT + 'domain.nc', 'a') as d:
        lat, lon = d['lat'][:], d['lon'][:]
        m = np.ma.filled(d['mask'][:], 0) > 0
        la = (lat >= BOX['lat'][0]) & (lat <= BOX['lat'][1])
        lo = (lon >= BOX['lon'][0]) & (lon <= BOX['lon'][1])
        box = la[:, None] & lo[None, :]
        sub = m & box
        d['mask'][:] = sub.astype(np.int32)
    shutil.copy(R + 'vicwur/isc/smoke/rout_n4.nc', OUT + 'rout_n4.nc')
    with nc.Dataset(OUT + 'rout_n4.nc', 'a') as r:
        did = np.ma.filled(r['downstream_id'][:], -1)
        dwn = np.ma.filled(r['downstream'][:], -1)
        ids = set(did[sub].tolist())
        out = sub & ~np.isin(dwn, list(ids))
        dwn2 = dwn.copy()
        dwn2[out] = did[out]
        r['downstream'][:] = dwn2
    shutil.copy(params, OUT + 'params.nc')
    with nc.Dataset(OUT + 'params.nc', 'a') as p:
        rc = p['run_cell']
        rc[:] = sub.astype(rc.dtype)
    sel = sub[m]
    st = dict(np.load(R + 'vicwur/isc/smoke/gw_static.npz'))
    n = int(m.sum())
    st2 = {}
    for k, v in st.items():
        v = np.asarray(v)
        ax = [i for i, s in enumerate(v.shape) if s == n]
        st2[k] = v.take(np.flatnonzero(sel), axis=ax[0]) if ax else v
    np.savez(OUT + 'gw_static_sub.npz', **st2)
    print('cells', int(sub.sum()), 'outlets', int(out.sum()))


if __name__ == '__main__':
    # subset.py [LAT0 LAT1 LON0 LON1 OUTDIR PARAMS]
    if len(sys.argv) > 1:
        a = sys.argv[1:]
        build(dict(lat=(float(a[0]), float(a[1])),
                   lon=(float(a[2]), float(a[3]))), a[4].rstrip('/') + '/',
              a[5])
    else:
        build()
