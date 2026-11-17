# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""isc-crop-fail: cell moisture of the spin-up (tools/isc_campaign/spin_init.py)."""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'isc_campaign'))
import spin_init as SI                                                 # noqa: E402


def test_cell_mean_cover_weighted_over_tiles_and_bands():
    # 2 cells; cell 0 has columns 0, 1 (two bands), cell 1 column 2; 2 tiles
    cv = torch.tensor([[0.75, 0.25], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    lane_cell = torch.tensor([0, 0, 1, 2])
    lane_tile = torch.tensor([0, 1, 0, 1])
    m = torch.tensor([[100., 10.], [200., 20.], [300., 30.], [400., 40.]], dtype=torch.float64)
    out = SI.cell_mean(m, lane_cell, lane_tile, cv, torch.tensor([0, 0, 1]), 2)
    col0 = (0.75 * m[0] + 0.25 * m[1])
    assert torch.allclose(out[0], (col0 + m[2]) / 2)
    assert torch.allclose(out[1], m[3])


def test_write_init_sets_active_cells(tmp_path):
    nc = pytest.importorskip('netCDF4')
    src, dst = str(tmp_path / 'a.nc'), str(tmp_path / 'b.nc')
    d = nc.Dataset(src, 'w')
    d.createDimension('l', 2); d.createDimension('y', 2); d.createDimension('x', 2)
    mk = d.createVariable('mask', 'i4', ('y', 'x')); mk[:] = [[1, 0], [1, 1]]
    im = d.createVariable('init_moist', 'f8', ('l', 'y', 'x')); im[:] = 5.0
    d.close()
    SI.write_init(src, dst, np.array([[1., 2.], [3., 4.], [5., 6.]]))
    r = nc.Dataset(dst)
    a = np.asarray(r['init_moist'][:])
    r.close()
    assert a[0, 0, 0] == 1 and a[1, 1, 0] == 4 and a[0, 1, 1] == 5 and a[0, 0, 1] == 5.0
    with pytest.raises(ValueError):
        SI.write_init(src, dst, np.zeros((2, 2)))
