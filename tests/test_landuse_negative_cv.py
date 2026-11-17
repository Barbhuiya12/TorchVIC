"""A forced Cv of -1e-16 (the rounding residue of the ISC land-use maps) is
taken as 0 by map_cv_forcing and gives a finite, non-negative Cv update."""
import torch

from torchvic.vicwur import landuse as LU


def test_negative_rounding_residue_is_zero():
    forced = torch.tensor([[0.6, -1.1e-16, 0.4 + 1.1e-16, 0.0]],
                          dtype=torch.float64)
    tile_class = torch.tensor([[0, 1, 2, -1]])
    cv, fatal = LU.map_cv_forcing(forced, tile_class)
    assert float(cv.min()) >= 0.0 and not bool(fatal.any())
    active = torch.tensor([[True, True, True, False]])
    cv_old = torch.tensor([[0.5, 1e-6, 0.5 - 1e-6, 0.0]],
                          dtype=torch.float64)
    cvn, dcv, s, fat, warn = LU.cv_update(cv, cv_old, active)
    assert bool(torch.isfinite(cvn).all()) and float(cvn.min()) >= 0.0
    assert abs(float(cvn.sum()) - 1.0) < 1e-12
