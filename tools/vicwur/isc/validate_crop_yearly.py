"""Check the per-year ISC crop nc: the torchvic readers load them (all flags) and no
crop unit (Cc > 0, so it sows) sits on a tile with Cv(y) <= 0.  Run with -I."""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)
from common import *            # noqa
from torchvic.vicwur import params as P, forcing as F

A = ROOT
g = P.Grid(A + '/vic_params_isc/domain.nc', OUT + '/params/params_wur.nc')
tot = 0
for y in YEARS:
    cr = P.read_crop('%s/wofost/crop_params_VICWOFOST_isc.%d.nc' % (OUT, y), g, 20,
                     dist_season=True, dist_tsum=True, dist_fert=True, dist_min=True,
                     calc_min=True)
    cv = np.asarray(F.read_year('%s/forcing/cv.%d.nc' % (OUT, y), 'cv'))   # [12?,20,N] or [20,N]
    cv = cv.reshape(-1, 20, g.ny * g.nx)[..., g.active.ravel()]
    cc = cr.Cc.numpy()                    # [12, NC, N]
    vc = cr.crop_veg_class.numpy()        # [NC, N] 0-based
    bad = 0
    for k in range(cc.shape[1]):
        v = vc[k]
        okv = v >= 0
        for m in range(12):
            cvm = cv[min(m, cv.shape[0] - 1)]
            c = np.take_along_axis(cvm, np.clip(v, 0, 19)[None], 0)[0]
            bad += int(((cc[m, k] > 0) & okv & (c <= 0)).sum())
    tot += bad
    log('CHECK readers %d: loaded, Cc>0 units on Cv<=0 tiles: %d, units with Cc>0: %d' % (
        y, bad, int((cc[0] > 0).sum())))
log('CHECK crop-yearly total sowings on Cv=0 tiles: %d' % tot)
