#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""ISC restart check: the full stack with groundwater,
continuous vs restarted from a checkpoint at day CUT, must agree bitwise.

  isc_restart_check.py [--days 8] [--cut 4]       (GPU, ~25 min)
Exit code 0 if every compared series is identical on the days after the cut.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isc_run as R                                                  # noqa

ap = argparse.ArgumentParser()
ap.add_argument('--days', type=int, default=8)
ap.add_argument('--cut', type=int, default=4)
ap.add_argument('--variant', default='nowofost_fixroots_')
ap.add_argument('--ck', default='/tmp/isc_restart_ck.pt')
a = ap.parse_args()
R.KEEP = 'host'
gp = '%s/global_isc_%s2005.txt' % (R.SMOKE, a.variant)
st = lambda: dict(np.load(R.SMOKE + '/gw_static.npz'))
MODS = ('groundwater', 'vicwur_routing', 'vicwur_nonrenew',
        'vicwur_irrigation', 'vicwur_wateruse_local', 'vicwur_dams',
        'vicwur_landuse', 'vicwur_efr')


def flat(x, key, out):
    if torch.is_tensor(x):
        out[key] = x.detach().cpu().double().numpy()
    elif isinstance(x, dict):
        for k, v in x.items():
            flat(v, '%s.%s' % (key, k), out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            flat(v, '%s[%d]' % (key, i), out)
    elif isinstance(x, (int, float, np.ndarray)):
        out[key] = np.asarray(x, np.float64)


def grab(m):
    """Final state of the connectors (their checkpoint blobs)."""
    out = {}
    for n in MODS:
        flat(m.runtime.connectors[n].save_state(), n, out)
    return out


m, c = R.build(gp, a.days, 'cuda', 'fixed', 31, st())
del c
with torch.inference_mode():
    m.initialize()
    m.run(until=a.cut * 86400.0)
    m.save_checkpoint(a.ck)
    m.run()
    m.finalize(write=False)
full = grab(m)
del m
from torchvic.coupling.connectors import vicwur_common as VC          # noqa
VC._GROUPS.clear()
m2, c = R.build(gp, a.days, 'cuda', 'fixed', 31, st(), restart=a.ck)
del c
with torch.inference_mode():
    m2.initialize()
    m2.run()
    m2.finalize(write=False)
re = grab(m2)
bad = 0
for k, v in re.items():
    d = float(np.abs(full[k] - v).max()) if full[k].shape == v.shape else float('inf')
    if d != 0.0:
        R.log('%-48s max|diff| %.3e' % (k, d))
        bad += 1
R.log('compared %d state arrays' % len(re))
R.log('RESTART %s' % ('BITWISE' if not bad else 'DIFFERS (%d arrays)' % bad))
sys.exit(1 if bad else 0)
