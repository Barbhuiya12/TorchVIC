"""basin.GRAPH_SNOW_ADAPT (found by the calibration work).

On the CUDA graph path a step with snow on no tile, run
through a graph captured for snow (snow_step on an all-false mask, every
output zeroed by _mask_tree), must be the snow-free branch (the cached
zeros) bit for bit, so the day need not run again eagerly. Checked on the
ISC forest box (24 Himalayan cells, forest tiles, a seasonal pack), with
the forcing edited per random cell so that snow appears and vanishes
within a day, a snowfall of a few 1e-12 m leaves a pack near the 1e-11
threshold, and steps sit on the rain/snow thresholds: every per-step
output and the final state, against the padded eager loop and the lanes
eagerly, in the water balance, the energy balance and with frozen soil.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_isc_forest_snow_vs_vic51 as forest            # noqa: E402
from torchvic.model import basin                          # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not os.path.exists(os.path.join(forest.BOX,
                                                       'params.nc')),
                       reason='the ISC forest box is not present'),
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason='no CUDA device')]

PER_TILE = ('soil_moisture_tile',)
SPD = 8


def _case(tmp, days, **opt):
    from torchvic import case
    g, _ = forest._case(str(tmp))['tv']
    c = case.load(g, device=torch.device('cuda'), steps=SPD * days,
                  log=lambda *a: None)
    if opt.pop('full_energy', False):
        c.full_energy = True
    c.options = dict(c.options, **opt)
    return c


def _edges(c, seed=0):
    """Forcing edited per random cell: snow on single steps of warm days
    (appearing and vanishing within a day), trace snowfalls near the
    1e-11 m snow-mask threshold, steps exactly on the thresholds."""
    g = torch.Generator().manual_seed(seed)
    ta, pr = c.atm['tair'], c.atm['prcp']
    nt, nc = ta.shape
    p = c.params()
    tmx, tmn = float(p['t_max_snow']), float(p['t_min_rain'])
    ta, pr = ta.clone(), pr.clone()
    cells = torch.randperm(nc, generator=g)
    warm, trace, edge = cells[:nc // 2], cells[nc // 2:3 * nc // 4], \
        cells[3 * nc // 4:]
    # summer-warm everywhere: the packs melt away in the first weeks, and
    # then the domain has snow on the steps below only (the graph's
    # snow assumption is domain-wide)
    ta += 12.0
    # one cold wet step on random days (snow on some steps of a day only)
    for d in torch.randperm(nt // SPD, generator=g)[:nt // SPD // 3]:
        s = int(d) * SPD + int(torch.randint(SPD, (1,), generator=g))
        k = warm[torch.randperm(warm.numel(), generator=g)[:3]]
        ta[s, k] = tmn - 1.0
        pr[s, k] = torch.rand(k.numel(), generator=g,
                              dtype=pr.dtype).to(pr.device) * 2.0 + 1e-3
    # trace snowfalls around the snow-mask threshold (prcp is mm per step)
    for i, s in enumerate(torch.randperm(nt, generator=g)[:nt // 4]):
        k = trace[i % trace.numel()]
        ta[s, k] = tmn - 0.5
        pr[s, k] = (1e-9, 5e-9, 1e-8, 2e-8, 1e-6)[i % 5]
    # steps exactly on (and next to) the thresholds
    vals = (tmx, tmn, tmx - 1e-12, tmn + 1e-12, 0.5 * (tmx + tmn))
    for i, s in enumerate(torch.randperm(nt, generator=g)[:nt // 4]):
        k = edge[i % edge.numel()]
        ta[s, k] = vals[i % len(vals)]
        pr[s, k] = 0.3
    c.atm = dict(c.atm, tair=ta, prcp=pr)
    return c


def _run(c, lanes, graphs, adapt=True, cut=None, p=None):
    old = (basin.LANE_COMPACT, basin.GRAPHS, basin.GRAPH_SNOW_ADAPT)
    basin.LANE_COMPACT, basin.GRAPHS = lanes, graphs
    basin.GRAPH_SNOW_ADAPT = adapt
    try:
        from torchvic.model import state as model_state
        kw = c.kwargs(p=p)
        nt = int(kw['atm']['tair'].shape[0])
        cut = cut or nt
        outs, st, redo = [], None, 0
        with torch.no_grad():
            for a, b in ((0, cut), (cut, nt)):
                if a == b:
                    continue
                o = basin.simulate(**model_state.time_window(kw, a, b),
                                   initial_state=st, return_state=True)
                st = o.pop('state')
                outs.append(o)
                redo += int(basin.GRAPH_STATS.get('redone_days', 0)
                            if graphs else 0)
        return outs, st, redo
    finally:
        basin.LANE_COMPACT, basin.GRAPHS, basin.GRAPH_SNOW_ADAPT = old


def _eq(v, w):
    return torch.equal(torch.nan_to_num(v, nan=1e300),
                       torch.nan_to_num(w, nan=1e300))


def _same(a, b, cv, cells=None):
    bad = []
    lc = cv > 0.0
    for k, v in a.items():
        if not torch.is_tensor(v) or k == 'discharge_mm':
            continue            # discharge: index_add order on a GPU
        w = b[k]
        if k in PER_TILE:
            # [cell, tile, ...]: the padded run computes uncovered tiles too
            if cells is not None:
                v, w, m = v[cells], w[cells], lc[cells]
                v, w = v[m], w[m]
            else:
                v, w = v[lc], w[lc]
        elif cells is not None:
            ax = [i for i, n in enumerate(v.shape) if n == cv.shape[0]]
            if ax:
                v, w = v.index_select(ax[0], cells), \
                    w.index_select(ax[0], cells)
        if v.shape != w.shape or not _eq(v, w):
            bad.append(k)
    return bad


def _state_same(s, t, keys=False):
    """Final states: every tensor both hold (`keys`: and the same keys)."""
    bad = []
    if keys and set(s) != set(t):
        bad.append('keys %s' % sorted(set(s) ^ set(t)))
    for k in set(s) & set(t):
        v, w = s[k], t[k]
        if isinstance(v, dict):
            bad += ['%s.%s' % (k, b) for b in _state_same(v, w, keys)]
        elif torch.is_tensor(v) and not _eq(v, w):
            bad.append(k)
    return bad


@pytest.mark.parametrize('mode', ['wb', 'eb', 'eb_frozen'])
def test_graph_snow_adapt_is_bitwise(tmp_path, mode):
    opt = dict(wb={}, eb=dict(full_energy=True),
               eb_frozen=dict(full_energy=True, frozen_soil=True))[mode]
    days = 60 if mode == 'wb' else 30
    c = _edges(_case(tmp_path, days, **opt))
    cut = SPD * (days // 2) + 3             # a window ends mid-day
    pad, _, _ = _run(c, False, False, cut=cut)
    lan, st_l, _ = _run(c, True, False, cut=cut)
    on, st_on, redo_on = _run(c, True, True, True, cut=cut)
    off, st_off, redo_off = _run(c, True, True, False, cut=cut)
    assert st_l.get('lanes') is not None
    cv = c.mosaic.cv
    for a, b, d, e in zip(pad, lan, on, off):
        bad = (_same(a, b, cv) + _same(a, d, cv) + _same(a, e, cv))
        assert not bad, bad
    assert not _state_same(st_l, st_on)
    assert not _state_same(st_off, st_on, keys=True)
    # snow came and went within days: the redo was there to be saved
    assert float(pad[0]['swe'].max()) > 1.0
    assert redo_on < redo_off, (redo_on, redo_off)
    assert basin.GRAPH_STATS.get('graphs', 0) >= 1
