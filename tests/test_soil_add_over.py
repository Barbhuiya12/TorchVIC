# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR 3.0's capillary-rise edit checks the Cv-weighted cell total only,
so a tile may end above max_moist (runoff.c:242-268 resolves it in the next
vic_run). soil_add_mode 'over' takes the addition whole; 'refuse' still
refuses it and 'cap' still stops at max_moist."""
import types

import pytest

torch = pytest.importorskip('torch')
from torchvic.coupling.vic import VicConnector            # noqa: E402
from torchvic.coupling.connector import ComputeError      # noqa: E402


def _v(mode):
    v = VicConnector(soil_add_mode=mode)
    soil = types.SimpleNamespace(
        max_moist=torch.tensor([[10., 20., 30.]], dtype=torch.float64),
        resid_moist=torch.zeros(1, 3, dtype=torch.float64))
    v.driver = types.SimpleNamespace(
        n_cells=1, dtype=torch.float64, device=torch.device('cpu'),
        soil=lambda: soil,
        mosaic=lambda: types.SimpleNamespace(
            cv=torch.ones(1, 1, dtype=torch.float64)))
    return v


MOIST = torch.tensor([[[5., 10., 25.]]], dtype=torch.float64)   # [cell, tile, layer]
ADD = torch.tensor([[0., 0., 9.]], dtype=torch.float64)         # layer 2: 34 > 30


def test_over_takes_whole():
    new, applied, unmet = _v('over')._add_soil_cols(MOIST, ADD, None)
    assert new[0, 0, 2].item() == 34.0
    assert applied.item() == 9.0 and unmet.item() == 0.0


def test_refuse_and_cap_unchanged():
    with pytest.raises(ComputeError):
        _v('refuse')._add_soil_cols(MOIST, ADD, None)
    new, applied, unmet = _v('cap')._add_soil_cols(MOIST, ADD, None)
    assert new[0, 0, 2].item() == 30.0 and unmet.item() == 4.0


def test_bad_mode():
    with pytest.raises(Exception):
        VicConnector(soil_add_mode='nope')
