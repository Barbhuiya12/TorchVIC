"""Load the 52b files (WOFOST text + crop nc, dams, MODFLOW statics) with the
torchvic readers and print sanity tables.  Run:  python -I validate_52b.py
(torch environment: <data root>/../env/venv/bin/python)."""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(HERE + '/../../..'))      # repository root
from torchvic.vicwur import params as P                                  # noqa
from torchvic.crop.wofost_npk.params import CropTemplates, setup_from_vicwur  # noqa
from torchvic.vicwur import groundwater as W                             # noqa
import netCDF4 as nc                                                     # noqa

A = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
O = A + '/vicwur/isc'
LOG = A + '/verification/vicwur_52b/run.log'


def log(m):
    line = '%s VALIDATE %s' % (time.strftime('%Y-%m-%dT%H:%M:%S'), m)
    print(line, flush=True)
    open(LOG, 'a').write(line + '\n')


def main():
    g = P.Grid(A + '/vic_params_isc/domain.nc', O + '/params/params_wur.nc')
    log('grid active cells %d' % g.n)
    # ---- WOFOST text
    t0 = time.time()
    wt = P.read_wofost_text(O + '/wofost/wofost_params_VICWOFOST_isc.txt', 23)
    tm = CropTemplates.from_wofost_text(wt)
    log('wofost text: %d crop classes, 77 scalars, 22 tables read (%.1f s); max_optimum_temp %s'
        % (wt.n_crops, time.time() - t0, np.round(tm.max_optimum_temp[:4], 1).tolist()))
    # ---- crop nc, every flag
    t0 = time.time()
    cr = P.read_crop(O + '/wofost/crop_params_VICWOFOST_isc.nc', g, 20, dist_season=True,
                     dist_tsum=True, dist_fert=True, dist_min=True, calc_min=True)
    log('crop nc: read_crop all flags OK (%.1f s): classes %d, Cc %s, fields %s, ncrop max %d'
        % (time.time() - t0, cr.n_crop_types, tuple(cr.Cc.shape), sorted(cr.fields), int(cr.ncrop.max())))
    cv = P.read_cv(O + '/params/params_wur.nc', g)
    vc = cr.crop_veg_class.numpy()
    cc = cr.Cc.numpy()
    for v in (13, 18, 19):
        s = np.zeros(g.n)
        for k in range(cc.shape[1]):
            m = vc[k] == v - 1
            s[m] += cc[:, k, :].mean(0)[m]
        log('veg %d: max |sum Cc - Cv| = %.2e' % (v, np.abs(s - cv[v - 1]).max()))
    lat = np.repeat(g.lat[:, None], g.nx, 1)[g.active]
    af = np.zeros((g.n, 1))
    af[:, 0] = 1.0
    cvk = np.zeros((g.n, wt.n_crops))
    for k in range(wt.n_crops):
        m = vc[k] == vc[k]
        cvk[:, k] = cc[:, k, :].mean(0)
    us = setup_from_vicwur(tm, cr, lat, af, np.zeros((g.n, 1)), cvk, dist_season=True,
                           dist_tsum=True, dist_fert=True, dist_min=True, calc_min=True)
    act = us.active
    cyc = us.cycle[act]
    log('setup_from_vicwur OK: active (cell,crop) units %d, cycle length days min %d median %d max %d'
        % (int(act.sum()), cyc.min(), int(np.median(cyc)), cyc.max()))
    # rice rows
    for k in range(3):
        a = us.active[:, k]
        log('class %d: units %d, plant doy median %d, harvest doy median %d, tsum1 %.0f tsum2 %.0f'
            % (k, a.sum(), np.median(us.start_doy[a, k]), np.median(us.end_doy[a, k]),
               us.tsum1[a, k].mean(), us.tsum2[a, k].mean()))
    # ---- dams (slow reader loops: time it)
    t0 = time.time()
    d = P.read_dams(O + '/dams/dams.nc', g)
    log('dams: read_dams OK (%.1f s): %d dams, %d global, capacity total %.0f hm3, warnings %d, '
        'service cells total %d' % (time.time() - t0, d.n_dams, int(d.is_global.sum()),
                                    float(d.capacity.sum()), len(d.warnings), int(d.nservice.sum())))
    # ---- MODFLOW statics
    with nc.Dataset(O + '/params/modflow_static.nc') as ds:
        v = {k: np.array(ds[k][:], float) for k in ('aqdepth', 'layer1_bottom', 'layer2_bottom', 'Sy',
                                                    'bdticm_m', 'porosity', 'log10_k')}
    with nc.Dataset(A + '/vic_params_isc/domain.nc') as ds:
        elev = np.array(ds['elev'][:], float)
    lm = np.ones_like(elev)
    aq = W.aqdepth(lm, v['aqdepth'])
    b1, b2 = W.botm(W.top(lm, elev), aq)
    log('groundwater.botm vs file layer bottoms: max |diff| %.3e / %.3e m; Sy in [%.3f, %.3f]; '
        'aqdepth reader fill-free: %s' % (np.abs(b1 - v['layer1_bottom']).max(),
                                         np.abs(b2 - v['layer2_bottom']).max(), v['Sy'].min(),
                                         v['Sy'].max(), bool((v['aqdepth'] > 0).all())))
    log('52b readers OK')


if __name__ == '__main__':
    main()
