#!/usr/bin/env python
"""Check that the torchvic groundwater readers accept the rebuilt aqdepth / layer bottoms."""
import os
import sys
import numpy as np
import netCDF4 as nc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(HERE + '/../../..'))
from torchvic.vicwur import groundwater as W     # noqa

A = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
LOG = A + '/verification/aquifer/run.log'


def log(m):
    print(m, flush=True)
    with open(LOG, 'a') as f:
        f.write(m + '\n')


with nc.Dataset(A + '/vicwur/isc/params/modflow_static.nc') as ds:
    v = {k: np.array(ds[k][:], float) for k in ('aqdepth', 'layer1_bottom', 'layer2_bottom', 'Sy')}
with nc.Dataset(A + '/vic_params_isc/domain.nc') as ds:
    elev = np.array(ds['elev'][:], float)
lm = np.ones_like(elev)
aq = W.aqdepth(lm, v['aqdepth'])
b1, b2 = W.botm(W.top(lm, elev), aq)
log('VALIDATE groundwater readers: aqdepth %.1f..%.1f m (reader fill-free: %s, unchanged by reader: %s); '
    'botm vs file bottoms max |diff| %.2e / %.2e m'
    % (aq.min(), aq.max(), bool((v['aqdepth'] > 0).all()), bool(np.array_equal(aq, v['aqdepth'])),
       np.abs(b1 - v['layer1_bottom']).max(), np.abs(b2 - v['layer2_bottom']).max()))
