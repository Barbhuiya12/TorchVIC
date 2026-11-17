"""The step the CUDA-graph path captures asks the device nothing.

A CUDA graph capture refuses any operation that waits for the device
(`bool(t.any())`, `.item()`, `t.nonzero()`, a boolean-mask index, a copy
from the host): "operation not permitted when stream is capturing". An
option that adds such a place to the step breaks every GPU run at its
first capture (main 0a90ee7a: `bool(aero_410.any())` in
canopy.canopy_residual, AR_410). Two guards, with every option family the
graph path runs switched on (AERO_RESIST_CANSNOW AR_410, SNOW_DENSITY
DENS_SNTHRM, CORRPREC, SHARE_LAYER_MOIST FALSE, BASEFLOW NIJSSEN2001, a
vegetation history on the step axis) and on the default options, under
both VIC cores:

* on any device, the graph path with every step run under
  graphsafe.SyncCheck (basin.SYNC_CHECK) in place of its capture finds no
  such place -- this one runs on a CPU;
* on a CUDA device, the captured and replayed run is the eager one bit
  for bit (marker gpu).

The case is the ISC forest box of test_isc_forest_snow_vs_vic51: forest
tiles, a seasonal pack and snow in the canopy.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_isc_forest_snow_vs_vic51 as forest            # noqa: E402
from torchvic.core import version, xmath                  # noqa: E402
from torchvic.model import basin, state as model_state    # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(forest.BOX, 'params.nc')),
    reason='the ISC forest box is not present')

#: every option family the graph path runs, off its default
ALL_ON = ('AERO_RESIST_CANSNOW AR_410', 'SNOW_DENSITY DENS_SNTHRM',
          'CORRPREC TRUE', 'SHARE_LAYER_MOIST FALSE',
          'BASEFLOW NIJSSEN2001')
OPTION_SETS = {'default': (), 'all_on': ALL_ON}


def _case(tmp, device, lines, days, veghist):
    from torchvic import case
    g, _ = forest._case(str(tmp))['tv']
    if lines:
        with open(g, 'a') as f:
            f.write('\n'.join(lines) + '\n')
    c = case.load(g, device=device, steps=8 * days, log=lambda *a: None)
    if veghist:
        # a vegetation history (torchvic.io.veghist): the monthly fields
        # on the step axis, as FORCING2 leaves them
        m = c.mosaic
        mo = torch.as_tensor([d.month - 1 for d in c.batch.dates],
                             device=m.lai.device)
        for nm in ('lai', 'fcanopy', 'albedo', 'displacement', 'veg_rough'):
            x = getattr(m, nm, None)
            if x is not None:
                setattr(m, nm, x.index_select(2, mo))
        m.step_axis = True
    return c


@pytest.fixture
def _modes():
    old = (basin.LANE_COMPACT, basin.GRAPHS, basin.SYNC_CHECK, xmath.EXACT,
           version.CORE)
    yield
    (basin.LANE_COMPACT, basin.GRAPHS, basin.SYNC_CHECK, xmath.EXACT,
     core) = old
    version.set_core(core)


@pytest.mark.parametrize('core', ['5.0.1', '5.1.0'])
@pytest.mark.parametrize('opts', sorted(OPTION_SETS))
def test_the_captured_step_asks_the_device_nothing(tmp_path, _modes, core,
                                                   opts):
    version.set_core(core)
    # glibc's exact math runs on the host, on a CPU only (it is off on a
    # CUDA device, where the graphs run)
    xmath.EXACT = False
    basin.LANE_COMPACT, basin.GRAPHS, basin.SYNC_CHECK = True, True, True
    c = _case(tmp_path, torch.device('cpu'), OPTION_SETS[opts], 3,
              veghist=opts == 'all_on')
    basin.SYNC_FOUND.clear()
    with torch.no_grad():
        basin.simulate(**c.kwargs())
    st = dict(basin.GRAPH_STATS)
    assert st.get('steps') == 24 and st.get('eager_steps') == 24, st
    assert not basin.SYNC_FOUND, basin.SYNC_FOUND


def test_the_check_finds_a_sync():
    from torchvic.core import graphsafe
    x = torch.ones(3)
    chk = graphsafe.SyncCheck()
    with chk:
        bool(x.any())
        x[x > 0.0] = 2.0
        torch.as_tensor([1.0, 2.0])
        y = x * 2.0 + torch.where(x > 1.0, x, -x)      # no sync
    assert len(chk.found) == 3, chk.found
    assert float(y[0]) == 6.0


def _run(c, graphs, cut, lanes=True):
    basin.LANE_COMPACT, basin.GRAPHS = lanes, graphs
    kw = c.kwargs()
    outs, st = [], None
    nt = int(kw['atm']['tair'].shape[0])
    with torch.no_grad():
        for a, b in ((0, cut), (cut, nt)):
            o = basin.simulate(**model_state.time_window(kw, a, b),
                               initial_state=st, return_state=True)
            st = o.pop('state')
            outs.append(o)
    return outs, dict(basin.GRAPH_STATS)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
@pytest.mark.parametrize('core', ['5.0.1', '5.1.0'])
@pytest.mark.parametrize('opts', sorted(OPTION_SETS))
def test_graphs_are_the_eager_step(tmp_path, _modes, core, opts):
    version.set_core(core)
    c = _case(tmp_path, torch.device('cuda'), OPTION_SETS[opts], 40,
              veghist=opts == 'all_on')
    eag, _ = _run(c, False, 8 * 31)
    gra, stats = _run(c, True, 8 * 31)
    assert stats.get('graphs', 0) >= 1, stats      # a graph was replayed
    lc = c.mosaic.cv > 0.0
    for p, q in zip(eag, gra):
        bad = []
        for k, v in p.items():
            if not torch.is_tensor(v) or k == 'discharge_mm':
                # discharge: index_add's order on a GPU is not fixed
                continue
            w = q[k]
            if k == 'soil_moisture_tile':
                v, w = v[lc], w[lc]
            if not torch.equal(torch.nan_to_num(v, nan=1e300),
                               torch.nan_to_num(w, nan=1e300)):
                bad.append(k)
        assert not bad, bad
    # and under the check, on the device itself
    basin.SYNC_CHECK = True
    basin.SYNC_FOUND.clear()
    _run(c, True, 8 * 31)
    assert not basin.SYNC_FOUND, basin.SYNC_FOUND


# ---- the energy balance (FULL_ENERGY TRUE) on the graph path ----------
#: global-file lines of each energy mode, in place of the case's
#: FULL_ENERGY FALSE / FROZEN_SOIL FALSE / NODES 3 (VIC's FROZEN_SOIL
#: TRUE turns QUICK_FLUX off, get_global_param.c 5.1.0:101-116, so the
#: frozen-soil row says QUICK_FLUX TRUE after it)
ENERGY = {
    'eb': ('FULL_ENERGY TRUE', 'FROZEN_SOIL FALSE', 'NODES 3'),
    'fs': ('FULL_ENERGY TRUE', 'FROZEN_SOIL TRUE', 'QUICK_FLUX TRUE',
           'IMPLICIT FALSE', 'EXP_TRANS FALSE', 'NODES 3'),
    'fd': ('FULL_ENERGY TRUE', 'FROZEN_SOIL FALSE', 'QUICK_FLUX FALSE',
           'IMPLICIT FALSE', 'EXP_TRANS FALSE', 'NODES 6'),
    'implicit': ('FULL_ENERGY TRUE', 'FROZEN_SOIL FALSE',
                 'QUICK_FLUX FALSE', 'IMPLICIT TRUE', 'EXP_TRANS TRUE',
                 'NOFLUX TRUE', 'NODES 10'),
}


def _energy_case(tmp, device, mode, days):
    from torchvic import case
    g, _ = forest._case(str(tmp))['tv']
    text = open(g).read().replace(
        'FULL_ENERGY FALSE\nFROZEN_SOIL FALSE\nNODES 3\n',
        '\n'.join(ENERGY[mode]) + '\n')
    assert 'FULL_ENERGY TRUE' in text
    open(g, 'w').write(text)
    return case.load(g, device=device, steps=8 * days, log=lambda *a: None)


@pytest.fixture
def _eb_modes(_modes):
    old = basin.EB_LANES_FD
    basin.EB_LANES_FD = True
    yield
    basin.EB_LANES_FD = old


@pytest.mark.parametrize('core', ['5.0.1', '5.1.0'])
@pytest.mark.parametrize('mode', sorted(ENERGY))
def test_the_captured_energy_step_asks_the_device_nothing(
        tmp_path, _eb_modes, core, mode):
    version.set_core(core)
    xmath.EXACT = False
    basin.LANE_COMPACT, basin.GRAPHS, basin.SYNC_CHECK = True, True, True
    c = _energy_case(tmp_path, torch.device('cpu'), mode, 2)
    basin.SYNC_FOUND.clear()
    with torch.no_grad():
        o = basin.simulate(**c.kwargs(), return_state=True)
    assert o['state'].get('lanes') is not None       # on the covered tiles
    st = dict(basin.GRAPH_STATS)
    assert st.get('steps') == 16 and st.get('eager_steps') == 16, st
    # the CPU's implicit profile solve is the C library's, not the CUDA
    # kernel the graph path runs: its host reads are not the graph's
    found = {k: v for k, v in basin.SYNC_FOUND.items()
             if not (mode == 'implicit'
                     and k.startswith(('core/newtc.py',
                                       'core/soiltemp.py')))}
    assert not found, basin.SYNC_FOUND


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
@pytest.mark.parametrize('core', ['5.0.1', '5.1.0'])
@pytest.mark.parametrize('mode', sorted(ENERGY))
def test_energy_graphs_are_the_padded_eager_step(tmp_path, _eb_modes, core,
                                                mode):
    # main's energy balance on a GPU: the padded tile grid, eagerly
    version.set_core(core)
    c = _energy_case(tmp_path, torch.device('cuda'), mode, 20)
    pad, _ = _run(c, False, 8 * 10, lanes=False)
    gra, stats = _run(c, True, 8 * 10)
    assert stats.get('graphs', 0) >= 1, stats
    lc = c.mosaic.cv > 0.0
    for p, q in zip(pad, gra):
        bad = []
        for k, v in p.items():
            if not torch.is_tensor(v) or k == 'discharge_mm':
                continue
            w = q[k]
            if k == 'soil_moisture_tile':
                v, w = v[lc], w[lc]
            if not torch.equal(torch.nan_to_num(v, nan=1e300),
                               torch.nan_to_num(w, nan=1e300)):
                bad.append(k)
        assert not bad, bad
