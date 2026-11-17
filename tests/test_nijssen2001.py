"""BASEFLOW NIJSSEN2001: the parameter file's d1..d4 become ARNO's
Ds, Dsmax, Ws and c before the column ever sees them.

VIC reads the four baseflow columns into Ds, Dsmax, Ws, c and then, when
BASEFLOW is NIJSSEN2001, converts them in place (image driver
vic/drivers/shared_image/src/vic_init.c:815-829, classic driver
vic/drivers/classic/src/read_soilparam.c:693-707, identical text):

    Dsmax' = Dsmax * (1 / (max_moist - Ws))^-c + Ds * max_moist
    Ds'    = Ds * Ws / Dsmax'
    Ws'    = Ws / max_moist

with max_moist the bottom layer's, in mm. The conversion is in
`torchvic.core.soil.nijssen2001_to_arno` and `build_soil` applies it for
both drivers, because `ClassicParams` inherits `ImageParams` and both
reach the same builder.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

from torchvic.core import soil as S                   # noqa: E402
from torchvic.io import build as image_run            # noqa: E402
from torchvic.io.global_param import GlobalParam      # noqa: E402

import _paths                                           # noqa: E402

sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
matrix = pytest.importorskip('matrix')

DT = torch.float64

# d1, d2, d3, d4 and the bottom layer's max_moist (mm) for two cells,
# with the conversion's results evaluated in the C's own operation order
# (math.pow, left to right), written out so the test pins the order.
_CELLS = ((0.3, 0.5, 200.0, 2.0, 1000.0),
          (0.1, 0.05, 50.0, 3.0, 800.0))
_WANT = ((0.00018732438339057135, 320300.0, 0.2, 2.0),
         (2.3703613805553564e-07, 21093830.000000004, 0.0625, 3.0))


def test_nijssen2001_to_arno_matches_the_c_operation_order():
    d1 = torch.tensor([c[0] for c in _CELLS], dtype=DT)
    d2 = torch.tensor([c[1] for c in _CELLS], dtype=DT)
    d3 = torch.tensor([c[2] for c in _CELLS], dtype=DT)
    d4 = torch.tensor([c[3] for c in _CELLS], dtype=DT)
    mm = torch.tensor([c[4] for c in _CELLS], dtype=DT)
    got = S.nijssen2001_to_arno(d1, d2, d3, d4, mm)
    for g, w in zip(got, zip(*_WANT)):
        assert g.tolist() == list(w)


class _FakeParams(object):
    """The least `build_soil` reads: the file's own values and layouts."""

    def __init__(self):
        import numpy as np
        nlayer, ncell = 3, 2
        z = np.zeros((nlayer, ncell))
        self.soil = {
            'infilt': np.full(ncell, 0.2),
            'Ds': np.array([c[0] for c in _CELLS]),
            'Dsmax': np.array([c[1] for c in _CELLS]),
            'Ws': np.array([c[2] for c in _CELLS]),
            'c': np.array([c[3] for c in _CELLS]),
            'expt': z + 3.5, 'Ksat': z + 100.0,
            'resid_moist': z + 0.05, 'depth': z + 1.0,
            'avg_T': np.full(ncell, 10.0), 'dp': np.full(ncell, 4.0),
            'bulk_density': z + 1500.0, 'soil_density': z + 2600.0,
            'quartz': z + 0.3, 'rough': np.full(ncell, 0.01),
        }
        self._mm = z + 500.0
        self._mm[-1] = [c[4] for c in _CELLS]
        self.ds = None
        self.extra = {}

    def max_moist(self):
        return self._mm

    def critical_moisture(self):
        return self._mm * 0.5, self._mm * 0.3


def test_build_soil_converts_under_nijssen2001_and_not_under_arno():
    kw = dict(dtype=DT, device=torch.device('cpu'),
              runoff_steps_per_day=8, model_steps_per_day=8)
    file_vals = ([c[0] for c in _CELLS], [c[1] for c in _CELLS],
                 [c[2] for c in _CELLS], [c[3] for c in _CELLS])
    for baseflow, want in (('ARNO', file_vals),
                           ('NIJSSEN2001', list(zip(*_WANT)))):
        sp = image_run.build_soil(_FakeParams(), **kw, baseflow=baseflow)
        got = [sp.Ds.tolist(), sp.Dsmax.tolist(), sp.Ws.tolist(),
               sp.c.tolist()]
        assert got == [list(w) for w in want], baseflow
    # The default is VIC's default: the file's values untouched.
    sp = image_run.build_soil(_FakeParams(), **kw)
    assert sp.Ds.tolist() == file_vals[0]
    assert sp.Dsmax.tolist() == file_vals[1]


def test_image_options_accepts_nijssen2001(tmp_path):
    from torchvic import case as C
    r = matrix.row('igp', 'WB', BASEFLOW='NIJSSEN2001')
    g = tmp_path / 'g_nijssen.txt'
    g.write_text(matrix.global_text(r, str(tmp_path), str(tmp_path),
                                    ['OUT_BASEFLOW']))
    C.image_options(GlobalParam(str(g)))


def test_classic_options_accepts_nijssen2001(tmp_path):
    from torchvic.io.classic import ClassicOptions
    glob = _paths.STEHE_CLASSIC_GLOBAL
    if not os.path.exists(glob):
        pytest.skip('samples/data/classic/Stehekin is not present')
    text = open(glob).read()
    assert 'BASEFLOW            ARNO' in text
    g = tmp_path / 'global_nijssen.txt'
    g.write_text(text.replace('BASEFLOW            ARNO',
                              'BASEFLOW NIJSSEN2001'))
    old = os.environ.get('test_data_dir')
    os.environ['test_data_dir'] = _paths.SAMPLES
    try:
        opts = ClassicOptions(GlobalParam(str(g)))
    finally:
        if old is None:
            del os.environ['test_data_dir']
        else:
            os.environ['test_data_dir'] = old
    assert [r for r in opts.refused if r.startswith('BASEFLOW')] == []
    assert opts.baseflow == 'NIJSSEN2001'
