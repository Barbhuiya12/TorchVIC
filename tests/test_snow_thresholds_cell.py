"""Per-cell snow thresholds (found by the calibration work).

t_max_snow / t_min_rain as [cell] tensors as well as scalars, a parameter
extension (VIC 5's MAX_SNOW_TEMP / MIN_RAIN_TEMP are global scalars): a
[cell] tensor holding the scalar is the scalar run bit for bit, and a mixed
tensor gives each cell the run with its own scalar (batched vs single), on
the padded loop, the lanes and the CUDA graphs. Case: the ISC forest box
with the edited forcing of test_graph_snow_adapt.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_graph_snow_adapt import (  # noqa: E402,F401
    _case, _edges, _run, _same, _state_same, pytestmark)
from torchvic.model import basin                          # noqa: E402


def _p_cell(c, **kv):
    p = dict(c.params())
    n = int(c.mosaic.cv.shape[0])
    for k, v in kv.items():
        p[k] = v if torch.is_tensor(v) else torch.full(
            (n,), float(v), dtype=torch.float64, device='cuda')
    return p


@pytest.mark.parametrize('layout', ['pad', 'lanes', 'graph'])
def test_cell_thresholds_equal_scalar(tmp_path, layout):
    lanes, graphs = dict(pad=(False, False), lanes=(True, False),
                         graph=(True, True))[layout]
    c = _edges(_case(tmp_path, 40))
    p0 = c.params()
    ref, st0, _ = _run(c, lanes, graphs)
    pc = _p_cell(c, t_max_snow=float(p0['t_max_snow']),
                 t_min_rain=float(p0['t_min_rain']))
    got, st1, _ = _run(c, lanes, graphs, p=pc)
    cv = c.mosaic.cv
    for a, b in zip(ref, got):
        bad = _same(a, b, cv)
        assert not bad, bad
    if lanes:
        assert not _state_same(st0, st1)


@pytest.mark.parametrize('layout', ['pad', 'lanes', 'graph'])
def test_cell_thresholds_batched_vs_single(tmp_path, layout):
    lanes, graphs = dict(pad=(False, False), lanes=(True, False),
                         graph=(True, True))[layout]
    c = _case(tmp_path, 40)
    n = int(c.mosaic.cv.shape[0])
    sets = ((1.5, -0.5), (0.5, -1.5))           # (t_max_snow, t_min_rain)
    pick = torch.arange(n, device='cuda') % 2 == 0
    runs = [_run(c, lanes, graphs, p=_p_cell(c, t_max_snow=a,
                                             t_min_rain=b))[0]
            for a, b in sets]
    tmx = torch.where(pick, sets[0][0], sets[1][0]).double()
    tmn = torch.where(pick, sets[0][1], sets[1][1]).double()
    mix, _, _ = _run(c, lanes, graphs, p=_p_cell(c, t_max_snow=tmx,
                                                 t_min_rain=tmn))
    cv = c.mosaic.cv
    for i, cells in enumerate((pick.nonzero()[:, 0],
                               (~pick).nonzero()[:, 0])):
        for a, b in zip(runs[i], mix):
            bad = _same(a, b, cv, cells)
            assert not bad, (i, bad)
    # the two settings differ (the thresholds reach the run)
    assert any(_same(a, b, cv) for a, b in zip(runs[0], runs[1]))


def test_check_snow_range_per_cell():
    ok = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    basin.check_snow_range(ok, torch.tensor(-0.5, dtype=torch.float64))
    basin.check_snow_range(0.5, -0.5)
    with pytest.raises(ValueError):
        basin.check_snow_range(ok, torch.tensor([-0.5, 1.0, 0.0]))
    with pytest.raises(ValueError):
        basin.check_snow_range(-0.5, 0.5)
