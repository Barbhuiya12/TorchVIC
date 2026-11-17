"""The water-balance loop on the covered tiles only (basin.LANE_COMPACT) and
replayed as CUDA graphs (basin.GRAPHS) is the padded loop bit for bit.

Both are speed devices and nothing else: every output must be the padded
run's to the last bit. The case is the ISC forest box of
test_isc_forest_snow_vs_vic51 (24 Himalayan cells with forest tiles, a
seasonal pack and snow in the canopy -- every stage the graph-safe paths
change), January-February 2004, 3-hourly water balance, cut into two
windows so the carried state crosses a month and a window.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_isc_forest_snow_vs_vic51 as forest            # noqa: E402
from torchvic.model import basin, state as model_state    # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(forest.BOX, 'params.nc')),
    reason='the ISC forest box is not present')

# per-tile outputs: the padded run computes the uncovered tiles too
PER_TILE = ('soil_moisture_tile',)


def _run(c, lanes, graphs, cut):
    """The case in two windows, the second from the first's state."""
    old = basin.LANE_COMPACT, basin.GRAPHS
    basin.LANE_COMPACT, basin.GRAPHS = lanes, graphs
    try:
        kw = c.kwargs()
        outs, st = [], None
        nt = int(kw['atm']['tair'].shape[0])
        with torch.no_grad():
            for a, b in ((0, cut), (cut, nt)):
                o = basin.simulate(**model_state.time_window(kw, a, b),
                                   initial_state=st, return_state=True)
                st = o.pop('state')
                outs.append(o)
        return outs, st, dict(basin.GRAPH_STATS)
    finally:
        basin.LANE_COMPACT, basin.GRAPHS = old


def _same(a, b, cv):
    bad = []
    lc = (cv > 0.0)
    for k, v in a.items():
        if not torch.is_tensor(v):
            continue
        w = b[k]
        if k in PER_TILE:
            v, w = v[lc], w[lc]
        if k == 'discharge_mm':
            # the outlet weights sum the cell areas with index_add, whose
            # order on a GPU is not fixed: two padded runs differ there too
            if v.is_cuda:
                continue
        if not torch.equal(torch.nan_to_num(v, nan=1e300),
                           torch.nan_to_num(w, nan=1e300)):
            bad.append(k)
    return bad


def _case(tmp, device):
    from torchvic import case
    g, _ = forest._case(str(tmp))['tv']
    return case.load(g, device=device, steps=8 * 59, log=lambda *a: None)


def test_no_lanes_on_the_cpu(tmp_path):
    # the CPU's vectorised transcendentals round an element by where it
    # sits in the tensor, so the padded loop stays there
    c = _case(tmp_path, torch.device('cpu'))
    c.atm = {k: (v[:16] if torch.is_tensor(v) and v.dim() == 2 else v)
             for k, v in c.atm.items()}
    _, st, _ = _run(c, True, True, 8)
    assert st.get('lanes') is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
def test_lanes_are_the_padded_loop(tmp_path):
    c = _case(tmp_path, torch.device('cuda'))
    pad, _, _ = _run(c, False, False, 8 * 31)
    lan, st, _ = _run(c, True, False, 8 * 31)
    assert st.get('lanes') is not None          # the lanes did run
    cv = c.mosaic.cv
    for p, q in zip(pad, lan):
        bad = _same(p, q, cv)
        assert not bad, bad
    assert float(pad[0]['swe'].max()) > 10.0
    assert float(pad[0]['snow_canopy'].max()) > 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
def test_graphs_are_the_padded_loop(tmp_path):
    c = _case(tmp_path, torch.device('cuda'))
    pad, _, _ = _run(c, False, False, 8 * 31)
    gra, st, stats = _run(c, True, True, 8 * 31)
    assert stats.get('graphs', 0) >= 1           # a graph was replayed
    cv = c.mosaic.cv
    for p, q in zip(pad, gra):
        bad = _same(p, q, cv)
        assert not bad, bad


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
def test_band_tfactor_on_the_lanes_is_the_padded_loop(tmp_path):
    # A19: a snow-band case's Tfactor (generate_default_state.c:106) lapses
    # the initial foliage temperature per column; on the lanes each covered
    # tile takes its cell's value
    c = _case(tmp_path, torch.device('cuda'))
    base, _, _ = _run(c, False, False, 8 * 31)
    n = int(c.mosaic.cv.shape[0])
    c.options = dict(c.options, band_tfactor=torch.linspace(
        -3.0, 2.0, n, dtype=torch.float64, device='cuda'))
    pad, _, _ = _run(c, False, False, 8 * 31)
    lan, st, _ = _run(c, True, False, 8 * 31)
    gra, _, _ = _run(c, True, True, 8 * 31)
    assert st.get('lanes') is not None          # the lanes did run
    cv = c.mosaic.cv
    # the factor reaches the run
    assert any(_same(p, q, cv) for p, q in zip(base, pad))
    for p, q, r in zip(pad, lan, gra):
        bad = _same(p, q, cv) + _same(p, r, cv)
        assert not bad, bad
