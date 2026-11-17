"""A daily model step with snow sub-stepping runs on the lanes, and is the
padded loop's answer.

MODEL_STEPS_PER_DAY 1 with SNOW_STEPS_PER_DAY 8 keeps the 3-hourly records
beside the daily means (case.subdaily, [day, snow step, cell]); a day with a
pack, canopy snow or snowflag[NR] is sub-stepped (surface_fluxes.c:258-269).
The lanes carry one lane per covered tile, so the records are read at the
lanes like the rest of the forcing. Case: the ISC forest box of
test_lanes_graphs (Himalayan cells, seasonal pack, canopy snow),
January-February 2004.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_isc_forest_snow_vs_vic51 as forest            # noqa: E402
from torchvic.model import basin                           # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(forest.BOX, 'params.nc')),
    reason='the ISC forest box is not present')

NDAYS = 59


def _case(tmp, device):
    from torchvic import case
    g, _ = forest._case(str(tmp))['tv']
    text = open(g).read()
    for key in ('MODEL_STEPS_PER_DAY', 'RUNOFF_STEPS_PER_DAY'):
        text = text.replace('%s 8' % key, '%s 1' % key)
    text = text.replace('ENDMONTH 3\nENDDAY 31', 'ENDMONTH 2\nENDDAY 29')
    open(g, 'w').write(text)
    return case.load(g, device=device, daily=True,
                     log=lambda *a: None)


def _run(c, lanes):
    old = basin.LANE_COMPACT
    basin.LANE_COMPACT = lanes
    try:
        with torch.no_grad():
            o = basin.simulate(**c.kwargs(), return_state=True)
        return o
    finally:
        basin.LANE_COMPACT = old


def _maxdiff(a, b, cv):
    lc = cv > 0.0
    worst, where = 0.0, None
    for k, v in a.items():
        if not torch.is_tensor(v) or k not in b or not torch.is_tensor(b[k]):
            continue
        w = b[k]
        if k == 'soil_moisture_tile':
            v, w = v[lc], w[lc]
        if k == 'discharge_mm' and v.is_cuda:
            continue            # index_add order on a GPU
        d = (torch.nan_to_num(v.double(), nan=0.0)
             - torch.nan_to_num(w.double(), nan=0.0)).abs()
        if d.numel() and float(d.max()) > worst:
            worst, where = float(d.max()), k
    return worst, where


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
def test_daily_snow_step_on_the_lanes_is_the_padded_loop(tmp_path):
    c = _case(tmp_path, torch.device('cuda'))
    assert c.subdaily is not None and c.snow_steps == 8
    pad = _run(c, False)
    lan = _run(c, True)
    assert lan['state'].get('lanes') is not None        # the lanes did run
    assert float(pad['swe'].max()) > 10.0               # snow days
    worst, where = _maxdiff(pad, lan, c.mosaic.cv)
    assert worst <= 1e-9, (worst, where)
