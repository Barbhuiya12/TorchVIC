"""CLOSE_ENERGY TRUE: the canopy-air temperature is solved and the
ground/canopy balances iterate to closure, as VIC runs them.

calc_atmos_energy_bal.c:73-102 (5.0.1 identical): with CLOSE_ENERGY the
canopy air temperature is root_brent's root of the atmospheric energy
balance over Tair +- CANOPY_DT, not Tair, and the reported sensible heat
is re-evaluated at it. vic_init.c:1556 (5.0.1: surface_fluxes.c:228-232)
leaves MAX_ITER_GRND_CANOPY at 10, so the ground/canopy iteration in
surface_fluxes.c:443-721 runs instead of the single pass.
"""
import inspect

import pytest

torch = pytest.importorskip('torch')

from torchvic.core import energy as E  # noqa: E402
from torchvic.model import basin  # noqa: E402

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')

CPU = torch.device('cpu')
D = torch.float64
CP = E.CONST_CPMAIR


def _closed(tair, ins, ra, dens):
    return tair - ins * ra / (CP * dens)


def test_solve_canopy_air_temp_matches_closed_form():
    g = torch.Generator(device=CPU).manual_seed(11)
    n = 64
    tair = (torch.rand(n, generator=g, dtype=D, device=CPU) - 0.5) * 60.0
    ins = (torch.rand(n, generator=g, dtype=D, device=CPU) - 0.5) * 40.0
    ra = 20.0 + 200.0 * torch.rand(n, generator=g, dtype=D, device=CPU)
    dens = 0.9 + 0.5 * torch.rand(n, generator=g, dtype=D, device=CPU)
    tc, sh, fb = E.solve_canopy_air_temp(ins, ra, tair, dens)
    # root_brent stops the temperature within ROOT_BRENT_T of the root
    assert torch.allclose(tc, _closed(tair, ins, ra, dens), atol=2e-7,
                          rtol=0.0)
    # and the reported sensible heat is the balance at the solved
    # temperature, exactly (func_atmos_energy_bal.c:34-40)
    assert torch.equal(sh, CP * dens * (tair - tc) / ra)
    assert not bool(fb.any())


def test_solve_canopy_air_temp_widens_bracket():
    # root_brent.c:168-179 widens by 10 C a side where Tair +- 1 does not
    # bracket the root; the linear root is far outside here.
    tair = torch.full((8,), -3.0, dtype=D, device=CPU)
    ins = torch.full((8,), 400.0, dtype=D, device=CPU)
    ra = torch.full((8,), 60.0, dtype=D, device=CPU)
    dens = torch.full((8,), 1.1, dtype=D, device=CPU)
    tc, sh, fb = E.solve_canopy_air_temp(ins, ra, tair, dens)
    assert torch.allclose(tc, _closed(tair, ins, ra, dens), atol=2e-7,
                          rtol=0.0)
    assert torch.equal(sh, CP * dens * (tair - tc) / ra)
    assert not bool(fb.any())


def test_close_energy_is_off_by_default():
    sig = inspect.signature(basin.simulate)
    assert sig.parameters['close_energy'].default is False


def test_close_energy_without_full_energy_is_refused(tmp_path):
    import os
    import sys

    import _paths  # noqa: E402

    sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
    matrix = pytest.importorskip('matrix')
    from torchvic import case as C  # noqa: E402
    from torchvic.io.global_param import GlobalParam  # noqa: E402

    r = matrix.row('wg', 'WB')
    txt = matrix.global_text(r, str(tmp_path), str(tmp_path), ['OUT_PREC'],
                            extra_lines=['CLOSE_ENERGY TRUE'])
    g = tmp_path / 'g.txt'
    g.write_text(txt)
    # VIC stops: CLOSE_ENERGY TRUE needs FULL_ENERGY TRUE
    # (get_global_param.c 5.1.0:961-964)
    with pytest.raises(ValueError, match='CLOSE_ENERGY TRUE with FULL_ENERGY '
                                         'FALSE'):
        C.image_options(GlobalParam(str(g)))
