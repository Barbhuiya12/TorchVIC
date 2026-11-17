"""SNOW_BAND with more than one band per cell, wired into the step loop.

VIC runs one column per (cell, band) with area, shifting only the air
temperature and the precipitation per band
(`surface_fluxes.c:357-359`) and weighting every output by
`Cv * AreaFract[band]` (`vic_run.c:325-330`, `put_data.c`). TorchVIC
expands the cells to pseudo-cells in `case.load` and contracts the
outputs back in `Case.run` (`torchvic/core/snowbands.py`), so the
physics loop itself is unchanged and a single-band domain expands to
itself.
"""
import os
import sys

import pytest

import _paths                                           # noqa: E402

sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
matrix = pytest.importorskip('matrix')

from torchvic import case as C                          # noqa: E402
from torchvic.io.global_param import GlobalParam        # noqa: E402


def _params(case):
    c = matrix.CASES[case]
    if not os.path.exists(c['params']):
        pytest.skip('%s is absent' % c['params'])
    from torchvic.io.domain import ImageDomain
    from torchvic.io.params import ImageParams
    dom = ImageDomain(c['domain'])
    return ImageParams(c['params'], dom)


def _global(tmp_path, case='stehe', mode='WB', edit=None, **kw):
    r = matrix.row(case, mode, **kw)
    txt = matrix.global_text(r, str(tmp_path), str(tmp_path), ['OUT_PREC'])
    for a, b in (edit or {}).items():
        assert a in txt, a
        txt = txt.replace(a, b)
    g = tmp_path / 'g.txt'
    g.write_text(txt)
    return str(g)


def test_multi_band_inputs_are_accepted(tmp_path):
    par = _params('stehe')                      # 3-5 bands with area a cell
    g = GlobalParam(_global(tmp_path, case='stehe',
                            edit={'SNOW_BAND FALSE': 'SNOW_BAND TRUE'}))
    assert C.check_inputs(g, par) == []


def test_single_band_pfactor_is_vics_normalised_one():
    # vic_init.c:1042-1059 normalises one band's fraction to 1 and then
    # divides by its area of 1, so a single band's Pfactor is 1 whatever
    # the file holds; refusing it would refuse what VIC runs.
    import numpy as np
    from types import SimpleNamespace
    gp = SimpleNamespace(get_bool=lambda k: k == 'SNOW_BAND',
                         get=lambda k: None)
    par = SimpleNamespace(
        bands={'AreaFract': np.array([[1.0, 1.0]]),
               'Pfactor': np.array([[0.3, 2.5]])},
        veg={})
    assert C.check_inputs(gp, par) == []


def test_multiband_run_contracts_to_cells(tmp_path):
    c = matrix.CASES['stehe']
    if not os.path.exists(c['params']):
        pytest.skip('%s is absent' % c['params'])
    from torchvic.core import version as core_version
    g = _global(tmp_path, case='stehe',
                edit={'SNOW_BAND FALSE': 'SNOW_BAND TRUE'})
    prev = core_version.CORE
    try:
        case = C.load(g, device='cpu', core='5.1.0', steps=24)
    finally:
        core_version.set_core(prev)
    ex = case.snowbands
    assert ex is not None and len(ex['cell']) == 64
    assert case.n_cells == 16
    # The foliage temperature starts lapsed per band
    # (generate_default_state.c:106), so the run is given the Tfactor.
    import numpy as np
    assert np.array_equal(case.kwargs()['band_tfactor'].cpu().numpy(),
                          ex['tfactor'])
    prev = core_version.CORE
    try:
        out = case.run()
    finally:
        core_version.set_core(prev)
    assert tuple(out['runoff'].shape) == (24, 16)
    assert tuple(out['baseflow'].shape) == (24, 16)
    assert tuple(out['swe'].shape) == (24, 16)
    # The contracted history is the area-weighted pseudo-cell history.
    from torchvic.core import snowbands
    from torchvic.model import basin
    prev = core_version.CORE
    core_version.set_core('5.1.0')
    try:
        raw = basin.simulate(**case.kwargs())
    finally:
        core_version.set_core(prev)
    assert tuple(raw['runoff'].shape) == (24, 64)
    import torch
    want = snowbands.contract(raw['runoff'], ex, axis=1)
    assert torch.equal(out['runoff'], want)


def _banded_case(n_steps=2):
    # A Case far enough to contract: two bands (0.9/0.1) over one cell.
    import numpy as np
    from torchvic.core import snowbands
    c = C.Case.__new__(C.Case)
    c.snowbands = snowbands.expand(np.array([[0.9], [0.1]]),
                                   np.array([[1000.0], [2000.0]]),
                                   np.array([[0.9], [0.1]]))
    c.dates = [0] * n_steps
    c._cell_echo = {}
    return c


def test_snow_means_contract_over_the_snowy_fraction():
    # put_data.c:433-437 averages the pack albedo and temperatures over
    # the snow-covered fraction: a band without snow must not dilute
    # the snowy bands' mean.
    import numpy as np
    import torch
    c = _banded_case(n_steps=1)
    out = {'salbedo': torch.tensor([[0.85, 0.0]], dtype=torch.float64),
           'snow_cover': torch.tensor([[1.0, 0.0]], dtype=torch.float64)}
    got = c.contract_bands(out)['salbedo']
    assert np.allclose(got.numpy(), [[0.85]], rtol=0, atol=1e-15)


def test_aero_resist_contracts_as_one_over_mean_conductance():
    # surface_fluxes.c:1023-1025: the resistance is the inverse of the
    # mean conductance, not the mean of the inverses.
    import numpy as np
    import torch
    c = _banded_case(n_steps=1)
    out = {'eb_aero_cond': torch.tensor([[0.02, 0.03]], dtype=torch.float64),
           'eb_aero_resist': torch.tensor([[50.0, 100.0 / 3.0]], dtype=torch.float64)}
    got = c.contract_bands(out)['eb_aero_resist']
    assert np.allclose(got.numpy(), [[1.0 / 0.021]], rtol=0, atol=1e-12)


def test_rad_temp_contracts_as_root_mean_fourth_power():
    # put_data.c:456: the fourth root of the mean fourth power.
    import numpy as np
    import torch
    c = _banded_case(n_steps=1)
    out = {'eb_rad_temp': torch.tensor([[300.0, 200.0]], dtype=torch.float64)}
    got = c.contract_bands(out)['eb_rad_temp']
    want = (0.9 * 300.0 ** 4 + 0.1 * 200.0 ** 4) ** 0.25
    assert np.allclose(got.numpy(), [[want]], rtol=0, atol=1e-9)


def test_forcing_echoes_are_the_cells_forcing():
    # put_data.c:115-132 reports force->x, not a band mean.
    import torch
    c = _banded_case()
    c._cell_echo = {'tair': torch.tensor([[3.0], [4.0]], dtype=torch.float64)}
    out = {'air_temp': torch.tensor([[10.0, -2.0], [11.0, -1.0]], dtype=torch.float64)}
    got = c.contract_bands(out)['air_temp']
    assert torch.equal(got, torch.tensor([[3.0], [4.0]], dtype=torch.float64))


def test_single_band_run_skips_expansion(tmp_path):
    par = _params('wg')                         # one band with area a cell
    g = _global(tmp_path, case='wg',
                edit={'SNOW_BAND FALSE': 'SNOW_BAND TRUE'})
    assert C.check_inputs(
        GlobalParam(g), par) == []
    from torchvic.core import version as core_version
    prev = core_version.CORE
    try:
        case = C.load(g, device='cpu', core='5.1.0', steps=8)
    finally:
        core_version.set_core(prev)
    # No cell has more than one band, so there is nothing to expand and
    # the run is the single-column run, bit for bit.
    assert getattr(case, 'snowbands', None) is None
    assert 'band_tfactor' not in case.kwargs()


def test_snow_expansion_helper_is_the_one_case_load_uses(tmp_path):
    # run_isc_torchvic.py expands bands through C.snow_expansion and
    # C.expand_bands_*; case.load must hold the same expansion.
    import numpy as np
    c = matrix.CASES['stehe']
    if not os.path.exists(c['params']):
        pytest.skip('%s is absent' % c['params'])
    from torchvic.core import version as core_version
    g = _global(tmp_path, case='stehe',
                edit={'SNOW_BAND FALSE': 'SNOW_BAND TRUE'})
    prev = core_version.CORE
    try:
        case = C.load(g, device='cpu', core='5.1.0', steps=8)
    finally:
        core_version.set_core(prev)
    ex = C.snow_expansion(case.par)
    assert ex is not None
    for k in ('cell', 'band', 'tfactor', 'pfactor', 'weight'):
        assert np.array_equal(np.asarray(ex[k]), np.asarray(case.snowbands[k]))
    soil, mosaic, init = C.expand_bands_static(
        case.soil, case.mosaic, case.init_moist, ex, 'cpu')
    assert init.shape[0] == len(ex['cell']) == case.init_moist.shape[0]
