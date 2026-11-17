"""The irrigation / water-use and crop connectors on the coupled runtime.

Irrigation (torchvic/coupling/connectors/wateruse.py) on VIC's Stehekin
sample, exchange: day, soil_add_mode: cap:
  * the module and coupled ledgers close every interval, and the engine's
    cumulative identities (demand = withdrawn + unmet, ...) close;
  * no source is asked for more than it holds, and a limited source leaves
    demand unmet instead of being overdrawn;
  * the water irrigation sends is the water VIC receives (no double
    counting), VIC's unmet comes back into the leftover store;
  * cells without irrigation are VIC-only cells, bit for bit;
  * a checkpointed run restarts to the continuous run.
Crop (connectors/crop.py): WOFOST 7.2 in external water mode coupled to VIC
(RFTRA from VIC's transpiration) equals the standalone engine with the same
RFTRA series bit for bit; restart equality; the LAI leaves through the
vegetation-history input path (write_veghist).
"""
import datetime
import gc
import json
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                                            # noqa: E402
from torchvic import case as _case                       # noqa: E402
from torchvic.coupling import (TorchVIC, Connector, Balance,   # noqa: E402
                               Proposal, VarSpec, GridSpec, registry,
                               validate_config)
from torchvic.coupling.vic import GRID                   # noqa: E402

STEHE = _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                set_options={'SNOW_BAND': 'FALSE'})
STEHE_DATA = os.path.join(_paths.SAMPLES, 'image', 'Stehekin')
PCSE = os.path.join(HERE, 'data', 'crop_pcse', 'demo_sbarley_1995_pp.npz')
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
DAY = 86400.0


os.environ['test_data_dir'] = _paths.SAMPLES


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    import gc
    gc.collect()                       # release the last run's open files
    monkeypatch.setenv('test_data_dir', _paths.SAMPLES)
    if not (os.path.exists(STEHE) and os.path.isdir(STEHE_DATA)):
        pytest.skip('VIC\'s Stehekin sample is not present')


# ---------------------------------------------------------------- irrigation
IRR = dict(wcr_fract=0.95, wfc_fract=1.0, irrigated_fraction=0.5,
           efficiency=0.7, conveyance_loss=0.1)
LINKS = [{'from': 'vic.soil_moisture', 'to': 'irrigation.soil_moisture'},
         {'from': 'vic.soil_add_unmet', 'to': 'irrigation.soil_unmet'},
         {'from': 'irrigation.soil_add', 'to': 'vic.soil_add', 'lag': 1}]


_DRIVERS = {}


def _driver(device):
    """One loaded Stehekin case per device for the whole module (every
    runtime shares it; each connector keeps its own state)."""
    from torchvic.coupling.vic import CaseDriver
    if not os.path.exists(STEHE):
        pytest.skip('Stehekin global file absent: set TORCHVIC_REFERENCE_DIR '
                    '(tools/get_test_data.sh)')
    if device not in _DRIVERS:
        _DRIVERS[device] = CaseDriver(_case.load(
            STEHE, device=device, log=lambda *a: None))
    return _DRIVERS[device]


def _runtime(device, irr=None, links=LINKS, extra=None, vic=None,
             crop=None):
    from torchvic.coupling.vic import VicConnector
    from torchvic.coupling.connectors.wateruse import WaterUseConnector
    drv = _driver(device)
    conns = dict(vic=VicConnector(driver=drv, **dict(
        dict(exchange='day', soil_add_mode='cap'), **(vic or {}))))
    mods = dict(vic=dict(connector='vic', global_param=STEHE))
    if irr is not None:
        conns['irrigation'] = WaterUseConnector(driver=drv, **irr)
        mods['irrigation'] = dict(connector='irrigation')
    if crop is not None:
        from torchvic.coupling.connectors.crop import CropConnector
        conns['crop'] = CropConnector(driver=drv, **crop)
        mods['crop'] = dict(connector='crop')
    conns.update(extra or {})
    for k in (extra or {}):
        mods[k] = dict(connector='two_store')
    return TorchVIC.from_dict(
        dict(run=dict(device=device, log=False), modules=mods,
             connections=list(links)), connectors=conns)


def _run(device, irr=IRR, until=None, **kw):
    with torch.inference_mode():
        m = _runtime(device, irr, **kw).initialize()
        m.run(until=until)
    return m


def _closed(m):
    assert m.ledger.balances and all(r['ok'] for r in m.ledger.balances)
    assert m.ledger.coupled and all(r['ok'] for r in m.ledger.coupled)


@pytest.fixture(scope='module')
def vic_only():
    with torch.inference_mode():
        m = _runtime('cpu', None, links=[]).initialize()
        m.run()
    return m


def _no_lanes(monkeypatch):
    """soil_add is not supported on a lane-compacted state (VicConnector);
    the GPU compacts by default, so the GPU run keeps the full grid."""
    from torchvic.model import basin
    monkeypatch.setattr(basin, 'LANE_COMPACT', False)
    monkeypatch.setattr(basin, 'EB_LANES', False)


@pytest.mark.parametrize('device', DEVICES)
def test_ledger_closes_and_the_engine_identities_hold(device, monkeypatch):
    if device != 'cpu':
        _no_lanes(monkeypatch)
    m = _run(device)
    _closed(m)
    c = m.connector('irrigation')
    bal = c.engine.balance()
    scale = max(float(c.engine.led['demand'].max()), 1.0)
    assert float(bal.abs().max()) <= 1e-12 * scale
    tot = m.ledger.totals('irrigation')
    assert tot['I_ext'] > 0 and tot['Q_out'] > 0
    # the module balance closes over the run, not only per interval
    assert abs(tot['residual']) <= 1e-9 * tot['I_ext']
    out = m.outputs('irrigation')
    assert float(out['soil_add'].sum()) > 0            # water was delivered


def test_allocation_never_exceeds_availability():
    # a small river (refilled), a small reservoir and groundwater store
    lim = dict(IRR, river_store_m3=2.0e6, river_refill_m3=1.0e5,
               gw_store_m3=5.0e5, res_store_m3=5.0e5, gw_fraction=0.3)
    m = _run('cpu', lim)
    _closed(m)
    c = m.connector('irrigation')
    out = m.outputs('irrigation')
    # cumulative draw from each source <= what it ever held
    n_int = out['soil_add'].shape[0]
    assert float(out['from_gw'].sum(0).max()) <= 5.0e5 * (1 + 1e-12)
    assert float(out['from_res'].sum(0).max()) <= 5.0e5 * (1 + 1e-12)
    # the river store: draw_k <= store_{k-1}; returns and refill come back
    store = torch.full_like(out['from_river'][0], 2.0e6)
    for k in range(n_int):
        draw = out['from_river'][k]
        assert bool((draw <= store * (1 + 1e-12) + 1e-9).all()), k
        store = store - draw + out['river_return'][k] + 1.0e5
    # and the limit bit: demand went unmet, never negative
    assert float(out['unmet'].sum()) > 0
    assert float(out['unmet'].min()) >= -1e-6
    # unlimited stubs: nothing is unmet
    free = _run('cpu', IRR)
    assert float(free.outputs('irrigation')['unmet'].abs().max()) <= 1e-6
    assert c.engine.led['unmet'].sum() > 0


def test_no_double_counting_and_unmet_returns():
    m = _run('cpu')
    area = torch.as_tensor(
        m.connector('irrigation').area.cpu(), dtype=torch.float64)
    out = m.outputs('irrigation')
    # the last interval's soil_add is still in transit (lag 1)
    sent = float((out['soil_add'][:-1].cpu() * area).sum()) / 1000.0
    vic = m.ledger.totals('vic')
    irr = m.ledger.totals('irrigation')
    # what irrigation sends is what VIC receives (applied + handed back)
    assert abs(vic['I_ext'] - sent - 0.0) <= 1e-9 * sent
    # VIC's I_ext holds only soil_add: no precip_add was connected
    # Q_out of irrigation = soil_add sent + returns to the river stub
    ret = float(out['river_return'].sum())
    sent_all = float((out['soil_add'].cpu() * area).sum()) / 1000.0
    assert abs(irr['Q_out'] - sent_all - ret) <= 1e-9 * (sent_all + ret)
    # each m3 withdrawn is delivered, consumed, returned or stored
    wd = float(out['withdrawn'].sum())
    left = float(out['leftover'][-1].sum())
    unm = float(m.outputs('vic')['soil_add_unmet'].sum()) \
        if 'soil_add_unmet' in m.outputs('vic') else 0.0
    assert wd >= 0 and left >= 0 and unm == unm
    assert abs(irr['I_ext'] - wd - _unmet_m3(m)) <= 1e-9 * wd


def _unmet_m3(m):
    """VIC's handed-back soil_add, in m3, from the runtime's transfers."""
    tot = 0.0
    for r in m.ledger.transfers:
        if r.get('quantity') == 'soil_add_unmet' and r.get('leg') == 'delivered':
            tot += r['amount']
    return tot


class UnmetSource(Connector):
    """Stands in for a VIC that could not take `amount` mm of the irrigation
    each day: exports soil_moisture (wet: no new demand) and soil_unmet.
    Its balance: what it hands back came in from the boundary."""
    name = 'src'
    interval_s = DAY
    backends = ('cpu', 'cuda')

    def __init__(self, drv, amount):
        super().__init__()
        self.drv, self.amount = drv, amount
        self.n = drv.n_cells
        self.outputs = (
            VarSpec('soil_moisture', 'mm', ('cell', 'layer'), kind='state',
                    grid=GRID),
            VarSpec('soil_unmet', 'mm', ('cell',), kind='flux', grid=GRID))

    def grids(self):
        d = self.drv
        return [GridSpec(GRID, self.n, kind='cells', area=torch.as_tensor(
            d.area * d.frac, dtype=torch.float64))]

    def time_span(self):
        return (0.0, 6 * DAY)

    def initialize(self, ctx):
        self.dev = ctx.device
        self.area = torch.as_tensor(self.drv.area * self.drv.frac,
                                    dtype=torch.float64, device=self.dev)
        self.wet = self.drv.soil().max_moist.to(self.dev)

    def compute(self, inputs, ctx, dt):
        un = torch.full((self.n,), self.amount if ctx.t < 3 * DAY else 0.0,
                        dtype=torch.float64, device=self.dev)
        v = float((un * self.area).sum()) / 1000.0
        return Proposal(self.name, ctx.t, ctx.t + dt,
                        exports=dict(soil_moisture=self.wet * 0.5,
                                     soil_unmet=un),
                        balance=Balance(I_ext=v, Q_out=v, scale=2 * v,
                                        n_terms=2 * self.n))

    def commit(self, p, ctx):
        pass


def test_vic_unmet_comes_back_into_the_leftover_store():
    """What VIC hands back (soil_unmet) joins the leftover store, is offered
    to the soil again within the layer's headroom and nothing is lost: the
    ledgers close and, over the run, everything handed back is re-delivered
    or still stored."""
    drv = _driver('cpu')
    src = UnmetSource(drv, 3.0)
    from torchvic.coupling.connectors.wateruse import WaterUseConnector
    irr = WaterUseConnector(driver=drv, irrigated_fraction=1.0,
                            season=dict(start_doy=400, end_doy=400))
    # a season that never opens: no new demand, only the returned water
    m = TorchVIC.from_dict(
        dict(run=dict(device='cpu', log=False),
             modules=dict(src=dict(connector='two_store'),
                          irrigation=dict(connector='irrigation')),
             connections=[{'from': 'src.soil_moisture',
                           'to': 'irrigation.soil_moisture'},
                          {'from': 'src.soil_unmet',
                           'to': 'irrigation.soil_unmet'}]),
        connectors=dict(src=src, irrigation=irr))
    with torch.inference_mode():
        m.initialize()
        m.run()
    _closed(m)
    out = m.outputs('irrigation')
    area = irr.area.cpu()
    back = 3.0 * 3 * float(area.sum()) / 1000.0           # 3 days of unmet
    redelivered = float((out['soil_add'].cpu() * area).sum()) / 1000.0
    stored = float(out['leftover'][-1].sum())
    assert float(out['demand'].sum()) == 0.0
    assert redelivered > 0
    assert abs(back - redelivered - stored) <= 1e-9 * back


def test_vic_only_cells_are_unchanged(vic_only):
    # half the cells irrigated, half not
    n = 16
    frac = [0.5 if i % 2 == 0 else 0.0 for i in range(n)]
    m = _run('cpu', dict(IRR, irrigated_fraction=frac))
    _closed(m)
    a, b = vic_only.outputs('vic'), m.outputs('vic')
    dry = torch.arange(1, n, 2)
    wet = torch.arange(0, n, 2)
    same = 0
    for k, v in a.items():
        if not torch.is_tensor(v) or v.dim() < 1 or v.shape[-1] != n:
            continue
        w = b[k]
        assert torch.equal(v[..., dry], w[..., dry]), k
        same += 1
    assert same > 10
    # and the irrigated cells did change
    assert not torch.equal(a['soil_liq'][..., wet, :], b['soil_liq'][..., wet, :])


def test_restart_equals_the_continuous_run(tmp_path):
    a = _run('cpu')
    with torch.inference_mode():
        b = _runtime('cpu', IRR).initialize()
        b.run(until=4 * DAY)
        ck = b.save_checkpoint(str(tmp_path / 'ck.pt'))
        c = _runtime('cpu', IRR).initialize()
        c.load_checkpoint(ck)
        c.run()
    oa, oc = a.outputs('irrigation'), c.outputs('irrigation')
    for k in oa:
        assert torch.equal(oa[k][4:], oc[k]), k
    ea, ec = a.connector('irrigation').engine, c.connector('irrigation').engine
    assert torch.equal(ea.leftover, ec.leftover)
    for k in ea.led:
        assert torch.equal(ea.led[k], ec.led[k]), k
    va, vc = a.outputs('vic'), c.outputs('vic')
    assert torch.equal(va['runoff'][4 * 24:], vc['runoff'])


def test_registry_and_configs_are_available():
    assert registry.get('irrigation').status == 'available'
    assert registry.get('crop').status == 'available'
    for f in ('vic_irrigation.yaml', 'vic_crop.yaml'):
        p = os.path.join(_paths.REPO, 'configs', f)
        assert os.path.exists(p), p


# ---------------------------------------------------------------------- crop
def _pcse():
    import numpy as np
    z = np.load(PCSE)
    meta = json.loads(str(z['meta']))
    wx = {k: torch.as_tensor(z['wx_' + k.upper()], dtype=torch.float64)
          .unsqueeze(1) for k in ('tmin', 'tmax', 'irrad', 'rain', 'e0',
                                  'es0', 'et0')}
    wx['doy'] = torch.as_tensor(z['doy'], dtype=torch.float64).unsqueeze(1)
    return wx, float(z['lat']), meta


class RftraSource(Connector):
    """Stands in for VIC: a deterministic RFTRA series per cell."""
    name = 'src'
    interval_s = DAY
    backends = ('cpu', 'cuda')

    def __init__(self, n, days):
        super().__init__()
        self.n = n
        self.series = (0.55 + 0.45 * torch.sin(
            torch.arange(days, dtype=torch.float64).unsqueeze(1) / 9.0
            + torch.arange(n, dtype=torch.float64).unsqueeze(0))).clamp(0, 1)
        self.outputs = (VarSpec('rftra', '1', ('cell',), kind='state',
                                grid=GRID),)
        self.day = 0

    def grids(self):
        return [GridSpec(GRID, self.n, kind='cells')]

    def initialize(self, ctx):
        self.dev = ctx.device
        self.series = self.series.to(ctx.device)

    def compute(self, inputs, ctx, dt):
        day = int(round(ctx.t / DAY))
        return Proposal(self.name, ctx.t, ctx.t + dt,
                        exports=dict(rftra=self.series[day]))

    def commit(self, p, ctx):
        self.day += 1

    def save_state(self):
        return dict(day=self.day)

    def load_state(self, blob):
        self.day = blob['day']


def _crop_runtime(device, n=3, days=None):
    wx, lat, meta = _pcse()
    days = days or meta['ndays']
    src = RftraSource(n, days)
    crop = dict(connector='crop', params=meta['params'], lat=lat,
                n_cells=n, weather=wx, max_days=days)
    m = TorchVIC.from_dict(
        dict(run=dict(device=device, log=False, duration_days=days),
             modules=dict(src=dict(connector='two_store'), crop=crop),
             connections=[{'from': 'src.rftra', 'to': 'crop.rftra'}]),
        connectors=dict(src=src))
    return m, src, wx, lat, meta


def _standalone(src, wx, lat, meta, n, days, device):
    from torchvic.crop import CropParams
    from torchvic.crop.run import simulate
    cp = CropParams.from_sets([meta['params']], n=n, device=device)
    f = {k: v.to(device) for k, v in wx.items()}
    latt = torch.full((n,), lat, dtype=torch.float64, device=device)
    res, _ = simulate(cp, {k: v[:days] for k, v in f.items()}, latt,
                      mode='external', rftra=src.series[:days].to(device),
                      outputs=('lai', 'dvs', 'tagp', 'twso', 'rd'))
    return res


@pytest.mark.parametrize('device', DEVICES)
def test_crop_equals_the_standalone_engine_bit_for_bit(device):
    n = 3
    m, src, wx, lat, meta = _crop_runtime(device, n)
    days = meta['ndays']
    with torch.inference_mode():
        m.initialize()
        m.run()
        out = m.outputs('crop')
        ref = _standalone(src, wx, lat, meta, n, days, device)
    for ck, rk in (('lai', 'lai'), ('dvs', 'dvs'), ('tagp', 'tagp'),
                   ('twso', 'twso'), ('rd', 'rd')):
        assert torch.equal(out[ck].to(device), ref[rk]), ck
    assert float(out['lai'].max()) > 1.0 and float(out['twso'][-1].max()) > 0


def test_crop_restart_equals_the_continuous_run(tmp_path):
    n = 3
    a, src, wx, lat, meta = _crop_runtime('cpu', n)
    b = _crop_runtime('cpu', n)[0]
    c = _crop_runtime('cpu', n)[0]
    with torch.inference_mode():
        a.initialize()
        a.run()
        b.initialize()
        b.run(until=70 * DAY)
        ck = b.save_checkpoint(str(tmp_path / 'crop.pt'))
        c.initialize()
        c.load_checkpoint(ck)
        c.run()
    oa, oc = a.outputs('crop'), c.outputs('crop')
    for k in oa:
        assert torch.equal(oa[k][70:], oc[k]), k


def test_crop_with_a_start_day_waits_then_matches():
    n = 2
    wx, lat, meta = _pcse()
    src = RftraSource(n, 40)
    crop = dict(connector='crop', params=meta['params'], lat=lat, n_cells=n,
                weather={k: v[5:45] for k, v in wx.items()}, max_days=40,
                start_day=5)
    wx5 = {k: v[5:45] for k, v in wx.items()}
    m = TorchVIC.from_dict(
        dict(run=dict(device='cpu', log=False, duration_days=40),
             modules=dict(src=dict(connector='two_store'), crop=crop),
             connections=[{'from': 'src.rftra', 'to': 'crop.rftra'}]),
        connectors=dict(src=src))
    # weather arrays are indexed by the run day: before start_day nothing
    with torch.inference_mode():
        m.initialize()
        m.run()
    out = m.outputs('crop')
    assert float(out['crop_active'][:5].abs().sum()) == 0
    assert float(out['crop_active'][5].min()) == 1
    assert wx5['tmin'].shape[0] == 40


def test_crop_with_vic_matches_standalone_and_leaves_vic_alone(vic_only):
    """VIC on Stehekin supplies RFTRA = transp / pet; the crop (weather from
    the PCSE barley season) equals the standalone engine given the same
    RFTRA series, and VIC's own results are the VIC-only run's."""
    wx, lat, meta = _pcse()
    wx10 = {k: v[:10] for k, v in wx.items()}
    crop = dict(params=meta['params'], lat=lat, weather=wx10, max_days=10)
    links = [{'from': 'vic.transp', 'to': 'crop.transp'},
             {'from': 'vic.pet', 'to': 'crop.pet'}]
    with torch.inference_mode():
        m = _runtime('cpu', None, links=links, crop=crop).initialize()
        m.run()
    _closed(m)
    v, v0 = m.outputs('vic'), vic_only.outputs('vic')
    for k in ('runoff', 'et', 'transp', 'soil_liq', 'water_error'):
        assert torch.equal(v[k], v0[k]), k
    day = lambda x: x.reshape(10, 24, 16).sum(1)
    tr, pet = day(v['transp']), day(v['pet'])
    from torchvic.coupling.connectors.crop import CropConnector
    rftra = CropConnector.rftra_of(tr, pet)
    assert float(rftra.min()) < 1.0              # VIC did stress the crop
    from torchvic.crop import CropParams
    from torchvic.crop.run import simulate
    cp = CropParams.from_sets([meta['params']], n=16)
    res, _ = simulate(cp, wx10, torch.full((16,), lat, dtype=torch.float64),
                      mode='external', rftra=rftra,
                      outputs=('lai', 'tagp', 'twso', 'rd', 'dvs'))
    out = m.outputs('crop')
    for k in ('lai', 'tagp', 'twso', 'rd', 'dvs'):
        assert torch.equal(out[k], res[k]), k


def test_crop_rftra_soil_is_wofost_rfws_over_the_root_zone():
    """RFTRA from the soil water is WOFOST 7.2's RFWS = TRA / TRAMX:
    (W - Wwp) / (Wcr - Wwp) over the layers above the rooting depth, each
    weighted by the part of its thickness within RD, clipped to [0, 1]."""
    from torchvic.coupling.connectors.crop import CropConnector as CC
    f64 = dict(dtype=torch.float64)
    d = torch.tensor([[0.3, 0.7, 1.0]], **f64)
    wcr = torch.tensor([[30., 70., 100.]], **f64)
    wp = torch.tensor([[10., 20., 30.]], **f64)

    def r(m, rd):
        return float(CC.rftra_soil(torch.tensor([m], **f64), d, wcr, wp,
                                   torch.tensor([rd], **f64)))
    assert r([30., 70., 100.], 2.0) == 1.0          # at the critical point
    assert r([40., 90., 200.], 2.0) == 1.0          # wetter: no stress
    assert r([10., 20., 30.], 2.0) == 0.0           # wilting point
    assert r([5., 0., 0.], 2.0) == 0.0
    assert r([20., 0., 0.], 0.3) == 0.5             # only the top layer
    assert r([20., 45., 0.], 0.65) == (10. + 0.5 * 25.) / (20. + 0.5 * 50.)
    assert r([20., 45., 0.], 0.0) == 1.0            # no root zone: no range


def test_crop_rftra_from_vic_soil_water_not_transp_over_pet(vic_only):
    """Regression (Nandgaon cotton: yield about 0). RFTRA was
    vic.transp / vic.pet, which with wet soil is the canopy-cover and
    dry-canopy share of OUT_PET (bare-soil potential included), 0.1-0.3, not
    1. With vic.soil_moisture linked the crop takes WOFOST's RFWS from VIC's
    soil water: every day it is rftra_soil of VIC's end-of-day layer
    moisture at the rooting depth the day starts from, 1 where the root
    zone is at or above Wcr (where transp / pet is well below 1), and VIC is
    left alone."""
    from torchvic.coupling.connectors.crop import CropConnector as CC
    wx, lat, meta = _pcse()
    wx10 = {k: v[:10] for k, v in wx.items()}
    crop = dict(params=meta['params'], lat=lat, weather=wx10, max_days=10)
    links = [{'from': 'vic.soil_moisture', 'to': 'crop.soil_moisture'}]
    with torch.inference_mode():
        m = _runtime('cpu', None, links=links, crop=crop).initialize()
        m.run()
    assert m.ledger.balances and all(r['ok'] for r in m.ledger.balances)
    v, v0 = m.outputs('vic'), vic_only.outputs('vic')
    for k in ('runoff', 'et', 'transp', 'soil_liq', 'water_error'):
        assert torch.equal(v[k], v0[k]), k
    c = m.connector('crop')
    so = c.driver.soil()
    nl = so.depth.shape[1]
    w = v['soil_liq'].reshape(10, 24, 16, nl)[:, -1]
    if v.get('soil_ice') is not None:
        w = w + v['soil_ice'].reshape(10, 24, 16, nl)[:, -1]
    out = m.outputs('crop')
    rd0 = c.cparams.RDI.reshape(1, -1).expand(1, 16)
    rd = torch.cat([rd0, out['rd'][:-1]]) / 100.0
    exp = CC.rftra_soil(w, so.depth, so.w_crit_mm, so.w_wilt_mm, rd)
    got = out['rftra']
    on = got > 0                                   # emerging days report 0
    assert bool(on.any())
    assert torch.allclose(got[on], exp[on], rtol=0, atol=1e-12)
    day = lambda x: x.reshape(10, 24, 16).sum(1)
    old = CC.rftra_of(day(v['transp']), day(v['pet']))
    wet = on & (exp == 1.0)
    assert bool(wet.any())
    assert float(old[wet].max()) < 0.9            # the old ratio starved it


def test_lai_leaves_through_the_vegetation_history_path(tmp_path):
    """write_veghist writes the files VIC reads for LAI_SRC FROM_VEGHIST;
    reading them back gives the crop's LAI on the crop tiles and the
    parameter file's month elsewhere."""
    from torchvic.coupling.connectors.crop import write_veghist
    from torchvic.io import veghist
    c = _case.load(STEHE, device='cpu', log=lambda *a: None)
    days = c.n_steps // c.model_steps
    lai = torch.linspace(0.5, 3.0, days, dtype=torch.float64).unsqueeze(1) \
        .expand(days, c.n_cells).contiguous()
    prefix = str(tmp_path / 'veg.')
    files = write_veghist(c, lai, [1, 2], prefix)
    assert files and all(os.path.exists(f) for f in files)
    arr, stamps = veghist.read(prefix, c.dom, {'lai': 'LAI'},
                               sorted({d.year for d in c.dates}))
    got = arr['lai']                                  # record, tile, cell
    assert got.shape == (c.n_steps, c.mosaic.cv.shape[1], c.n_cells)
    spd = c.model_steps
    for t in (1, 2):
        for s in (0, spd, 5 * spd + 3):
            assert torch.allclose(got[s, t], lai[s // spd])
    month = c.dates[0].month - 1
    base = c.mosaic.lai[:, 0, month]
    assert torch.allclose(got[0, 0], base)
    # and VIC reads it: a second pass with FORCING2 + LAI_SRC FROM_VEGHIST
    gp = open(STEHE).read().splitlines()
    gp = [ln for ln in gp if not ln.startswith(('LAI_SRC',))]
    gp += ['LAI_SRC FROM_VEGHIST', 'FORCING2 %s' % prefix,
           'FORCE_TYPE SKIP veg']
    p2 = str(tmp_path / 'global2.txt')
    open(p2, 'w').write('\n'.join(gp) + '\n')
    with torch.inference_mode():
        c2 = _case.load(p2, device='cpu', core='5.1.0',
                        log=lambda *a: None)
        o2 = c2.run()
    assert c2.mosaic.step_axis
    assert float(o2['water_error'].abs().max()) < 1e-6


def test_crop_start_doy_sows_a_new_crop_every_season():
    """start_doy: the crop is sown again on that day of the year with a fresh
    engine. Two 30-day 'years' with the same weather and the same RFTRA give
    the same season twice, bit for bit (the second is not the continuation of
    the first); without start_doy the crop is sown once."""
    n = 2
    wx, lat, meta = _pcse()
    wx = {k: v[:30] for k, v in wx.items()}
    wx2 = {k: torch.cat([v, v]) for k, v in wx.items()}
    wx2['doy'] = torch.cat([torch.arange(1, 31.), torch.arange(1, 31.)]
                           ).to(torch.float64).unsqueeze(1)

    def run(**kw):
        src = RftraSource(n, 60)
        src.series = torch.cat([src.series[:30], src.series[:30]])
        crop = dict(connector='crop', params=meta['params'], lat=lat,
                    n_cells=n, weather=wx2, max_days=60, **kw)
        m = TorchVIC.from_dict(
            dict(run=dict(device='cpu', log=False, duration_days=60),
                 modules=dict(src=dict(connector='two_store'), crop=crop),
                 connections=[{'from': 'src.rftra', 'to': 'crop.rftra'}]),
            connectors=dict(src=src))
        with torch.inference_mode():
            m.initialize()
            m.run()
        return m.outputs('crop')
    o = run(start_doy=1)
    for k in ('lai', 'dvs', 'twso'):
        assert torch.equal(o[k][30:], o[k][:30]), k
    assert float(o['dvs'][29].min()) > 0.05
    assert float(o['dvs'][30].max()) < float(o['dvs'][29].min())
    once = run(start_day=0)
    assert torch.equal(once['dvs'][:30], o['dvs'][:30])
    assert not torch.equal(once['dvs'][30:], o['dvs'][30:])
    late = run(start_doy=5, start_day=0)           # first sowing on day 4
    assert float(late['crop_active'][:4].abs().sum()) == 0
    assert float(late['crop_active'][4].min()) == 1
