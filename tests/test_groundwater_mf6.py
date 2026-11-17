"""Groundwater engine against MODFLOW 6 6.7.0 reference runs.

References are produced by <data root>/verification/gw_engine/mf6_ref.py
(MODFLOW 6 examples ex-gwf-twri01 and ex-gwf-fhb as published, and generic
cases driven through libmf6) and read from TORCHVIC_GW_REF (default
<data root>/verification/gw_engine/ref). Tolerances follow the closure set in
both solvers (IMS dvclose 1e-9 / 1e-10): heads 1e-7 (model length unit),
package budget terms 1e-9 of the step's total inflow. Measured (2026-10-14):
twri 4.2e-8 ft, fhb 2.7e-11 m, mixed 2.7e-10 m, latlon 1.5e-11 m.
"""
import os
import _paths  # noqa: E402

import numpy as np
import pytest
import torch

import groundwater_cases as gc

REF = _paths.env_path(('TORCHVIC_GW_REF',), 'verification/gw_engine/ref')


def _ref(name):
    p = os.path.join(REF, name + '.npz')
    if not os.path.exists(p):
        pytest.skip(f'MODFLOW 6 reference {p} absent (run mf6_ref.py)')
    return dict(np.load(p))


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('name', ['twri', 'fhb', 'mixed', 'latlon'])
@pytest.mark.parametrize('precond', ['jacobi', 'rbgs'])
@pytest.mark.parametrize('device', DEVICES)
def test_against_mf6(name, precond, device):
    ref = _ref(name)
    c = getattr(gc, name)()
    m = gc.build_engine(c, device=device, precond=precond)
    heads, bud, res = gc.run_engine(c, m)
    assert heads.shape == ref['heads'].shape
    dh, db = gc.compare(heads, bud, ref, official=c['kind'] == 'official')
    assert dh <= 1e-7, dh
    for k, v in db.items():
        assert v <= 1e-9, (k, v)
    # water balance closure per step (percent discrepancy)
    for r in res:
        assert abs(r.discrepancy()[1]) < 1e-6
