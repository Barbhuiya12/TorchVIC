"""VIC-WUR through the real driver (VIC's Stehekin sample): the
day-coupled window with snow bands, VIC-WUR's phase group split at the day
boundaries (P1-O of day d, then F, U of day d+1), and the 'liquid' pumping
floor with FROZEN_SOIL on the lanes layout. Each multi-day window must be
the run cut into one-day windows, bit for bit, and the ledger must close."""
import atexit
import os
import shutil
import sys
import tempfile

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                                            # noqa: E402
from torchvic import case as _case                       # noqa: E402
from torchvic.coupling import TorchVIC                   # noqa: E402
from torchvic.vicwur.order import Options, PhaseScheduler  # noqa: E402
from test_coupling_daily import Irrigator, _equal        # noqa: E402

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def _global(tag, **opts):
    """The relocated Stehekin global with `opts`, in a scratch directory of
    its own. relocated_global writes one file per sample directory under
    the session's scratch, which other test modules share (they read it at
    collection): writing a variant there would hand them this module's
    options (FROZEN_SOIL), so _paths' scratch is swapped for the call."""
    saved = list(_paths._TMP)
    d = tempfile.mkdtemp(prefix='torchvic_vicwur_%s_' % tag)
    atexit.register(shutil.rmtree, d, True)
    _paths._TMP[:] = [d]
    try:
        return _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                       set_options=opts)
    finally:
        _paths._TMP[:] = saved


BANDED = _global('banded')
FLAT = _global('flat', SNOW_BAND='FALSE')
FROZEN = _global('frozen', SNOW_BAND='FALSE',
                 FROZEN_SOIL='TRUE\nQUICK_FLUX TRUE\nIMPLICIT FALSE\n'
                             'EXP_TRANS FALSE')


@pytest.fixture(autouse=True)
def _stehe_env(monkeypatch):
    monkeypatch.setenv('test_data_dir', _paths.SAMPLES)
    if not os.path.exists(BANDED) or not os.path.isdir(
            os.path.join(_paths.SAMPLES, 'image', 'Stehekin')):
        pytest.skip('VIC\'s Stehekin sample is not present')


def _model(glob, device, vic, irr=None):
    c = _case.load(glob, device=device, log=lambda *a: None)
    mods = dict(vic=dict(dict(global_param=glob), **vic))
    conns, given = [], {}
    if irr is not None:
        area = torch.as_tensor(c.dom.area * c.dom.frac, dtype=torch.float64)
        given['irr'] = Irrigator(area)
        mods['irr'] = dict(connector='two_store')
        conns = [{'from': 'vic.soil_moisture', 'to': 'irr.soil_moisture',
                  'lag': 1},
                 {'from': 'irr.water', 'to': 'vic.%s' % irr}]
    m = TorchVIC.from_dict(dict(run=dict(device=device, log=False),
                                modules=mods, connections=conns),
                           connectors=given)
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    return m


def _closes(m):
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)


# ------------------------------------------------------------ snow bands
@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
def test_banded_day_coupled_window_is_the_daily_exchange(device,
                                                         monkeypatch):
    if device == 'cuda':
        # the banded case is compared on the padded tile grid on the GPU
        # too (band_tfactor on the lanes: test_lanes_graphs, A19)
        from torchvic.model import basin
        monkeypatch.setattr(basin, 'LANE_COMPACT', False)
    # the banded energy-balance state carries a soil_ice store (zero
    # without FROZEN_SOIL), which the default refuses
    kw = dict(soil_add_bands=True, soil_add_frozen=True)
    daily = _model(BANDED, device, dict(exchange='day', **kw), 'soil_add')
    win = _model(BANDED, device, dict(window_days=4, day_coupled=['irr'],
                                      **kw), 'soil_add')
    assert win.connector('vic').driver.banded
    # the banded case's forcing series come back whole per window (a
    # reporting quirk independent of the coupling; VIC-only runs show it too)
    forcing = ('air_temp', 'swdown', 'lwdown', 'pressure', 'vp', 'vpd',
               'rel_humid', 'density', 'wind')
    a, b = ({k: v for k, v in m.outputs('vic').items() if k not in forcing}
            for m in (daily, win))
    if device == 'cpu':
        bad = _equal(a, b)
    else:
        # the GPU's padded banded grid is not bit-invariant to the window
        # cut even for VIC alone (1e-14 in soil_moisture_tile, no coupling
        # at all); equal to rounding there
        bad = [k for k, v in a.items() if torch.is_tensor(v) and not (
            k in b and v.shape == b[k].shape
            and torch.allclose(v, b[k], rtol=1e-12, atol=1e-12))]
    assert not bad, bad[:10]
    assert torch.equal(daily.outputs('irr')['water'],
                       win.outputs('irr')['water'])
    _closes(daily)
    _closes(win)
    assert daily.ledger.totals('vic')['I_ext'] > 0


# ----------------------------------------------------------- phase group
class PumpIrr(object):
    """A VIC-WUR stand-in: in P3 irrigates `irr` mm into layer 0 and pumps
    `pump` mm from the bottom layer of every column; records its clock."""

    def __init__(self, irr=1.0, pump=0.4, floor='zero'):
        self.irr, self.pump, self.floor = irr, pump, floor
        self.times, self.view, self.low, self.ice = [], None, None, None

    def set_time(self, t):
        self.times.append(t)

    def set_view(self, v):
        self.view = v

    def pass3(self):
        v = self.view
        nl = v.moist.shape[-1]
        v.add(0, self.irr)
        v.withdraw(nl - 1, self.pump, floor=self.floor)
        if v.ice is not None:
            gap = (v.moist - v.ice)[..., nl - 1].min()
            top = v.ice.max()
            self.ice = top if self.ice is None else torch.maximum(
                self.ice, top)
            self.low = gap if self.low is None else torch.minimum(
                self.low, gap)

    def save_state(self):
        return list(self.times)

    def load_state(self, b):
        self.times = list(b)


def _group(**kw):
    g = PhaseScheduler(Options(irrigation=True))
    comp = PumpIrr(**kw)
    g.register('irrigation', comp)
    return g, comp


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
def test_phase_group_window_is_the_daily_exchange(device):
    g1, c1 = _group()
    g4, c4 = _group()
    kw = dict(soil_add_floor='zero', soil_add_mode='cap')
    daily = _model(FLAT, device, dict(exchange='day', phase_group=g1, **kw))
    win = _model(FLAT, device, dict(window_days=4, phase_group=g4, **kw))
    bad = _equal(daily.outputs('vic'), win.outputs('vic'))
    assert not bad, bad[:10]
    _closes(daily)
    _closes(win)
    assert win.ledger.totals('vic')['I_ext'] > 0
    # clock: F, U of day 0 at 0; then at each boundary P1-O of day d
    # (d * 86400) and F, U of day d+1; the last day's P1-O at the end
    days = 10
    want = [0.0]
    for d_ in range(days - 1):
        want += [d_ * 86400.0, (d_ + 1) * 86400.0]
    want.append((days - 1) * 86400.0)
    assert c1.times == want and c4.times == want


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('floor', ['liquid', 'zero'])
def test_liquid_floor_with_frozen_soil_on_the_lanes(device, floor,
                                                    monkeypatch):
    from torchvic.model import basin
    # the CPU takes the lanes only under the host-read check (basin
    # SYNC_CHECK), which also lists any host read in the step loop
    monkeypatch.setattr(basin, 'SYNC_CHECK', True)
    basin.SYNC_FOUND.clear()
    # Stehekin's parameters switch frozen soil off per cell (fs_active 0):
    # on everywhere, so the January soil holds ice
    load = _case.load

    def _load(*a, **k):
        c = load(*a, **k)
        c.soil.fs_active = torch.ones_like(c.soil.fs_active)
        return c
    monkeypatch.setattr(_case, 'load', _load)
    g, comp = _group(irr=0.0, pump=500.0, floor=floor)
    m = _model(FROZEN, device, dict(window_days=5, phase_group=g,
                                    soil_add_floor=floor,
                                    soil_add_frozen=True,
                                    soil_add_mode='cap'))
    st = m.connector('vic').state
    assert st.get('lanes') is not None            # the compact layout
    assert st.get('soil_ice') is not None
    assert comp.low is not None and float(comp.ice) > 0.0   # it froze
    if floor == 'liquid':
        # the pump never took the layer below its ice
        assert float(comp.low) >= -1e-9
    else:
        # VIC-WUR's clip at 0 (wu_run_local.c:417-433): below the ice
        assert float(comp.low) < 0.0
    _closes(m)
    assert m.ledger.totals('vic')['I_ext'] < 0     # water was taken
    # no host read from the soil edits in the step loop; a physics line
    # can be listed when another device ran first in the same process
    # (its lazily made constants are made again here) -- a fresh process
    # lists none (checked in a verification run)
    ours = {k: v for k, v in basin.SYNC_FOUND.items()
            if k.startswith(('coupling/', 'vicwur/'))}
    assert not ours, ours
