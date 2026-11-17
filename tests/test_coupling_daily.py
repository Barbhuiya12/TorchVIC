"""Daily coupling inside VIC's windows: a daily module (here an irrigation
stand-in) run between the days of a multi-day VIC window (`day_coupled`,
basin.simulate's day_hook) gives, bit for bit, the run in which VIC is cut
into one-day windows and the module exchanges with it at every boundary --
while VIC keeps one simulate call (and on a GPU one set of captured
graphs) per window. The ledger closes both ways."""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                                            # noqa: E402
from torchvic import case as _case                       # noqa: E402
from torchvic.coupling import (TorchVIC, Connector, Balance,  # noqa: E402
                               Proposal, VarSpec, CouplingError)
from torchvic.coupling.vic import GRID                   # noqa: E402

STEHE = _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                set_options={'SNOW_BAND': 'FALSE'})
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _stehe_env(monkeypatch):
    monkeypatch.setenv('test_data_dir', _paths.SAMPLES)
    if not os.path.exists(STEHE) or not os.path.isdir(
            os.path.join(_paths.SAMPLES, 'image', 'Stehekin')):
        pytest.skip('VIC\'s Stehekin sample is not present')


class Irrigator(Connector):
    """Each day: water to bring the top layer up to `target` mm (at most
    `cap` mm a day), from outside the domain, as soil_add or as
    precip_add; reads VIC's soil moisture at the day's start."""
    name = 'irr'
    interval_s = 86400.0
    backends = ('cpu', 'cuda')

    def __init__(self, area, target=40.0, cap=3.0):
        super().__init__()
        self.area, self.target, self.cap = area, target, cap
        self.inputs = (VarSpec('soil_moisture', 'mm', ('cell', 'layer'),
                               kind='state', grid=GRID),)
        self.outputs = (VarSpec('water', 'mm', ('cell',), kind='flux',
                                grid=GRID),)
        self.days = 0

    def initialize(self, ctx):
        self.device = ctx.device

    def compute(self, inputs, ctx, dt):
        sm = inputs.get('soil_moisture')
        n = int(self.area.shape[0])
        if sm is None:
            w = torch.zeros((n,), dtype=torch.float64,
                           device=self.device)
        else:
            w = (self.target - sm[:, 0]).clamp(0.0, self.cap)
        v = (w * self.area.to(w.device)).sum() / 1000.0
        return Proposal(self.name, ctx.t, ctx.t + dt,
                        state=self.days + 1, exports=dict(water=w),
                        balance=Balance(I_ext=v, Q_out=v, scale=2 * v,
                                        n_terms=2 * n),
                        series=dict(water=w.unsqueeze(0)))

    def commit(self, p, ctx):
        self.days = p.state

    def save_state(self):
        return dict(days=self.days)

    def load_state(self, blob):
        self.days = blob['days']


def _run(device, target, vic):
    c = _case.load(STEHE, device=device, log=lambda *a: None)
    area = torch.as_tensor(c.dom.area * c.dom.frac, dtype=torch.float64)
    irr = Irrigator(area)
    m = TorchVIC.from_dict(
        dict(run=dict(device=device, log=False),
             modules=dict(vic=dict(dict(global_param=STEHE), **vic),
                          irr=dict(connector='two_store')),
             connections=[{'from': 'vic.soil_moisture',
                           'to': 'irr.soil_moisture', 'lag': 1},
                          {'from': 'irr.water', 'to': 'vic.%s' % target}]),
        connectors=dict(irr=irr))
    with torch.inference_mode():
        m.initialize().run()
    return m


def _equal(a, b, pre=''):
    bad = []
    for k, v in a.items():
        if isinstance(v, dict):
            bad += _equal(v, b.get(k, {}), pre + k + '.')
        elif torch.is_tensor(v) and not (k in b and torch.equal(v, b[k])):
            bad.append(pre + k)
    return bad


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('target', ['soil_add', 'precip_add'])
def test_day_coupled_window_is_the_daily_exchange(device, target):
    daily = _run(device, target, dict(exchange='day'))
    win = _run(device, target, dict(window_days=4, day_coupled=['irr']))
    a, b = daily.outputs('vic'), win.outputs('vic')
    bad = _equal(a, b)
    assert not bad, bad[:10]
    assert torch.equal(daily.outputs('irr')['water'],
                       win.outputs('irr')['water'])
    for m in (daily, win):
        assert all(r['ok'] for r in m.ledger.balances)
        assert all(r['ok'] for r in m.ledger.coupled)
    assert daily.ledger.totals('vic')['I_ext'] > 0
    assert abs(daily.ledger.totals('vic')['I_ext']
               - win.ledger.totals('vic')['I_ext']) <= 1e-9 * abs(
        daily.ledger.totals('vic')['I_ext'])
    # one VIC call per window, not per day
    assert sum(win.report()['backends']['vic']['ran'].values()) == 3
    assert win.connector('irr').days == 10


@pytest.mark.reference
def test_day_coupling_rolls_back_and_checks_its_connections():
    c = _case.load(STEHE, device='cpu', log=lambda *a: None)
    area = torch.as_tensor(c.dom.area * c.dom.frac, dtype=torch.float64)

    class Failing(Irrigator):
        def compute(self, inputs, ctx, dt):
            if ctx.t >= 6 * 86400:
                raise RuntimeError('irrigation failed')
            return super().compute(inputs, ctx, dt)
    irr = Failing(area)
    m = TorchVIC.from_dict(
        dict(run=dict(device='cpu', log=False),
             modules=dict(vic=dict(global_param=STEHE, window_days=4,
                                   day_coupled=['irr']),
                          irr=dict(connector='two_store')),
             connections=[{'from': 'vic.soil_moisture',
                           'to': 'irr.soil_moisture', 'lag': 1},
                          {'from': 'irr.water', 'to': 'vic.soil_add'}]),
        connectors=dict(irr=irr))
    from torchvic.coupling import StepRejected
    with torch.inference_mode():
        m.initialize()
        m.step()
        assert irr.days == 4
        while m.runtime.time.t < 4 * 86400:
            m.step()
        with pytest.raises(StepRejected, match='irrigation failed'):
            m.step()
    # the failed window's first two days were undone
    assert irr.days == 4
    assert m.connector('vic').t_step == 4 * 24
    # a day-coupled module reads soil_moisture at lag 1 only
    with pytest.raises(CouplingError, match='lag'):
        TorchVIC.from_dict(
            dict(run=dict(device='cpu', log=False),
                 modules=dict(vic=dict(global_param=STEHE, window_days=4,
                                       day_coupled=['irr']),
                              irr=dict(connector='two_store')),
                 connections=[{'from': 'vic.soil_moisture',
                               'to': 'irr.soil_moisture', 'lag': 0},
                              {'from': 'irr.water', 'to': 'vic.soil_add'}]),
            connectors=dict(irr=Irrigator(area))).initialize()
