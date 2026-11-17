"""Irrigation drawing on the groundwater module (VIC + irrigation +
groundwater, the pumping loop) on VIC's Stehekin sample.

  * module and coupled ledgers close every interval;
  * the water the wells deliver is what irrigation asked of them one day
    earlier (irrigation's payable = asked - delivered, never negative);
  * a checkpointed run restarts to the continuous run bit for bit.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                                            # noqa: E402
from torchvic import case as _case                       # noqa: E402
from torchvic.coupling import TorchVIC                   # noqa: E402

STEHE = _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                set_options={'SNOW_BAND': 'FALSE'})
STEHE_DATA = os.path.join(_paths.SAMPLES, 'image', 'Stehekin')
os.environ['test_data_dir'] = _paths.SAMPLES

LINKS = [
    {'from': 'vic.soil_moisture', 'to': 'irrigation.soil_moisture'},
    {'from': 'vic.soil_add_unmet', 'to': 'irrigation.soil_unmet'},
    {'from': 'irrigation.soil_add', 'to': 'vic.soil_add', 'lag': 1},
    {'from': 'vic.baseflow', 'to': 'groundwater.recharge'},
    {'from': 'irrigation.gw_request', 'to': 'groundwater.pumping_request'},
    {'from': 'groundwater.pumping', 'to': 'irrigation.gw_pumped', 'lag': 1},
]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    import gc
    gc.collect()
    monkeypatch.setenv('test_data_dir', _paths.SAMPLES)
    if not (os.path.exists(STEHE) and os.path.isdir(STEHE_DATA)):
        pytest.skip("VIC's Stehekin sample is not present")


def _model(gw_unmet=False):
    from torchvic.coupling.vic import CaseDriver, VicConnector
    from torchvic.coupling.connectors.wateruse import WaterUseConnector
    c = _case.load(STEHE, device='cpu', log=lambda *a: None)
    drv = CaseDriver(c)
    aq = dict(k=2.0, sy=0.1, ss=0.0, thickness=60.0, depth_to_water=4.0)
    if gw_unmet:
        aq.update(thickness=1.0, depth_to_water=0.5)     # almost no water
    conns = dict(
        vic=VicConnector(driver=drv, exchange='day', soil_add_mode='cap'),
        irrigation=WaterUseConnector(
            driver=drv, wcr_fract=0.95, wfc_fract=1.0, irrigated_fraction=0.5,
            gw_fraction=1.0, groundwater=True))
    mods = dict(vic=dict(connector='vic', global_param=STEHE),
                irrigation=dict(connector='irrigation'),
                groundwater=dict(connector='groundwater', domain=c.dom,
                                 interval_days=1.0, aquifer=aq,
                                 drain=dict(cond=5.0e4)))
    return TorchVIC.from_dict(
        dict(run=dict(device='cpu', log=False), modules=mods,
             connections=list(LINKS)), connectors=conns)


def _closed(m):
    assert m.ledger.balances and all(r['ok'] for r in m.ledger.balances)
    assert m.ledger.coupled and all(r['ok'] for r in m.ledger.coupled)


def test_pumping_loop_closes_and_is_paid():
    with torch.inference_mode():
        m = _model().initialize()
        m.run()
    _closed(m)
    irr = m.connector('irrigation')
    gw = m.ledger.totals('groundwater')
    ir = m.ledger.totals('irrigation')
    assert gw['Q_out'] > 0 and ir['Q_out'] > 0
    pay = irr.payable
    assert float(pay.min()) >= -1e-6
    sent = sum(x['amount'] for x in m.ledger.transfers
               if x['source'] == 'groundwater' and x['leg'] == 'sent'
               and x['quantity'] == 'pumping')
    got = sum(x['amount'] for x in m.ledger.transfers
              if x['source'] == 'groundwater' and x['leg'] == 'delivered'
              and x['quantity'] == 'pumping')
    assert sent >= got > 0
    # owed = pumping still in transit + what the wells could not deliver
    unfunded = float(pay.sum()) - (sent - got)
    assert unfunded >= -1e-6
    print('sent', sent, 'got', got, 'payable', float(pay.sum()))


def test_restart_equals_continuous(tmp_path):
    with torch.inference_mode():
        a = _model().initialize()
        a.run()
        b = _model().initialize()
        b.run(until=5 * 86400)
        p = b.save_checkpoint(str(tmp_path / 'ck.pt'))
        c = _model().initialize()
        c.load_checkpoint(p)
        c.run()
    for n in ('irrigation', 'groundwater'):
        sa, sc = a.connector(n).save_state(), c.connector(n).save_state()
        for k, v in sa.items():
            if torch.is_tensor(v):
                assert torch.equal(v, sc[k]), (n, k)
    _closed(c)


def test_split_is_a_partition():
    from torchvic.coupling import Context, registry
    from torchvic.coupling.connectors.split import SplitConnector
    assert registry.get('split').status == 'available'
    area = [1e6, 2e6, 3e6, 4e6, 5e6]
    s = SplitConnector(cells=[3, 1], area=area)
    s.initialize(Context(device=torch.device('cpu')))
    x = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
    p = s.compute({'runoff': x}, Context(device=torch.device('cpu')), 86400.0)
    r, b = p.exports['routed'], p.exports['box']
    assert torch.equal(r, torch.tensor([1., 0., 3., 0., 5.],
                                       dtype=torch.float64))
    assert torch.equal(b, torch.tensor([4., 2.], dtype=torch.float64))
    assert abs(p.balance.residual()) <= 1e-9
    assert s.grids()[0].name == 'box_cells'
