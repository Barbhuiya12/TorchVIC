"""Day sessions (coupling/vic_session.py): VIC's day exchange of a streamed
case on a GPU, run as one simulate call per block of days and handed out
a day at a time, is the call-a-day run bit for bit -- every day's series,
the ledger rows, the history files and the end state -- with water added
every day (precip_add, soil_add), across a checkpoint restart inside a
block and after a rejected step; and it makes one call a block."""
import math
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                                            # noqa: E402
from torchvic import case as _case                       # noqa: E402
from torchvic.coupling import (TorchVIC, Connector, Balance,  # noqa: E402
                               Proposal, StepRejected, VarSpec)
from torchvic.coupling.vic import GRID                   # noqa: E402
from torchvic.model import basin                         # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.reference,
              pytest.mark.skipif(not torch.cuda.is_available(),
                                 reason='needs a GPU')]

STEHE = _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                set_options={'SNOW_BAND': 'FALSE'})
STEHE_DATA = os.path.join(_paths.SAMPLES, 'image', 'Stehekin')
DAY = 86400.0


@pytest.fixture(autouse=True)
def _stehe_env(monkeypatch):
    monkeypatch.setenv('test_data_dir', _paths.SAMPLES)
    if not (os.path.exists(STEHE) and os.path.isdir(STEHE_DATA)):
        pytest.skip('VIC\'s Stehekin sample is not present')


class Water(Connector):
    """`amount` mm a day on every cell from outside the domain (from
    `start` s on); fails once at `fail_at` s (a rejected step)."""
    name = 'water'
    interval_s = DAY
    backends = ('cpu', 'cuda')

    def __init__(self, area, amount, start=0.0, fail_at=None):
        super().__init__()
        self.area, self.amount, self.start = area, amount, start
        self.fail_at = fail_at
        self.n = int(area.shape[0])
        self.inputs = (VarSpec('runoff', 'mm', ('cell',), kind='flux',
                               grid=GRID),)
        self.outputs = (VarSpec('water', 'mm', ('cell',), kind='flux',
                                grid=GRID),)

    def initialize(self, ctx):
        self.device = ctx.device

    def compute(self, inputs, ctx, dt):
        if self.fail_at is not None and ctx.t == self.fail_at:
            self.fail_at = None
            raise RuntimeError('water failed once')
        a = self.amount if ctx.t >= self.start else 0.0
        # a little more where VIC ran off more the day before: the water
        # depends on VIC's own outputs
        r = inputs.get('runoff')
        x = torch.full((self.n,), a, dtype=torch.float64, device=self.device)
        if r is not None and a:
            x = x + 0.01 * r.clamp(min=0.0)
        v = (x * self.area.to(x.device)).sum() / 1000.0
        return Proposal(self.name, ctx.t, ctx.t + dt, exports=dict(water=x),
                        balance=Balance(I_ext=v, Q_out=v, scale=2 * v,
                                        n_terms=2 * self.n))

    def commit(self, p, ctx):
        pass


def _model(session, out=None, target=None, amount=0.0, fail_at=None,
           session_days=4):
    vic = dict(global_param=STEHE, core='5.1.0', stream=True,
               exchange='day', forcing_feed=False, day_session=session,
               session_days=session_days, soil_add_mode='cap')
    mods = dict(vic=vic)
    conns, given = [], None
    if target is not None:
        c = _case.load(STEHE, device='cpu', log=lambda *a: None)
        area = torch.as_tensor(c.dom.area * c.dom.frac, dtype=torch.float64)
        given = dict(water=Water(area, amount, fail_at=fail_at,
                                 start=DAY if target == 'soil_add' else 0.0))
        mods['water'] = dict(connector='two_store')
        # soil_add arrives a day later: the water can read the day's own
        # runoff (VIC runs first); precip_add the day's: the day before's
        conns = [{'from': 'vic.runoff', 'to': 'water.runoff',
                  'lag': 0 if target == 'soil_add' else 1},
                 {'from': 'water.water', 'to': 'vic.%s' % target,
                  'lag': 1 if target == 'soil_add' else 0}]
    cfg = dict(run=dict(device='cuda', log=False), modules=mods,
               connections=conns)
    if out is not None:
        cfg['output'] = dict(dir=str(out))
    m = TorchVIC.from_dict(cfg, connectors=given)
    days = []

    def sink(name, prop):
        if name == 'vic':
            days.append({k: v.clone() for k, v in prop.series.items()
                         if torch.is_tensor(v)})
    m.runtime.outputs_mgr.add_sink(sink)
    return m, days


def _calls(monkeypatch):
    n = [0]
    real = basin.simulate

    def counted(*a, **k):
        n[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(basin, 'simulate', counted)
    return n


def _same_rows(a, b):
    def norm(r):
        return {k: v for k, v in r.items()
                if isinstance(v, (int, float, str, bool))}
    ra, rb = [norm(r) for r in a], [norm(r) for r in b]
    if len(ra) != len(rb):
        return False
    for x, y in zip(ra, rb):
        if set(x) != set(y):
            return False
        for k in x:
            u, w = x[k], y[k]
            if isinstance(u, float) and math.isnan(u):
                if not (isinstance(w, float) and math.isnan(w)):
                    return False
            elif u != w:
                return False
    return True


def _diff_state(a, b, pre=''):
    bad = []
    for k, v in a.items():
        if isinstance(v, dict):
            bad += _diff_state(v, b.get(k) or {}, pre + k + '.')
        elif torch.is_tensor(v):
            w = b.get(k)
            if w is None or not torch.equal(v, w):
                bad.append(pre + k)
    return bad


def _history(d):
    import glob
    import numpy as np
    import xarray as xr
    out = {}
    for f in sorted(glob.glob(os.path.join(str(d), '*.nc'))):
        with xr.open_dataset(f) as x:
            for v in x.data_vars:
                out[(os.path.basename(f), v)] = np.asarray(x[v].values)
    return out


def _check(a, b):
    (ma, da), (mb, db) = a, b
    assert len(da) == len(db) == 10
    for i, (x, y) in enumerate(zip(da, db)):
        assert set(x) == set(y), i
        bad = [k for k in x if not torch.equal(x[k], y[k])]
        assert not bad, (i, bad[:8])
    assert _same_rows(ma.ledger.balances, mb.ledger.balances)
    assert _same_rows(ma.ledger.coupled, mb.ledger.coupled)
    assert not _diff_state(ma.connector('vic').state,
                           mb.connector('vic').state)


@pytest.mark.parametrize('target', [None, 'precip_add', 'soil_add'])
def test_a_day_session_is_the_call_a_day_run(monkeypatch, tmp_path, target):
    import numpy as np
    runs = {}
    for session in (False, True):
        n = _calls(monkeypatch)
        m, days = _model(session, out=tmp_path / str(session),
                         target=target, amount=2.0)
        with torch.inference_mode():
            m.initialize().run()
            m.finalize()
        runs[session] = (m, days)
        # one call a day, or one a block of 4 days (4 + 4 + 2)
        assert n[0] == (3 if session else 10), (session, n[0])
        # VIC alone: its sessions run ahead of the days handed out
        assert m.connector('vic').alone is (target is None)
        assert all(r['ok'] for r in m.ledger.balances)
        if target is not None:
            assert m.ledger.totals('vic')['I_ext'] > 0
    _check(runs[False], runs[True])
    ha, hb = _history(tmp_path / 'False'), _history(tmp_path / 'True')
    assert ha and sorted(ha) == sorted(hb)
    for k in ha:
        assert np.array_equal(ha[k], hb[k], equal_nan=True), k


def test_a_session_restarts_where_the_run_left_it(monkeypatch, tmp_path):
    """A rejected step inside a block and a checkpoint taken inside one:
    the next day is computed from the committed state (a new call), and
    the run is the uninterrupted one."""
    with torch.inference_mode():
        ref = _model(False, target='soil_add', amount=1.0)
        ref[0].initialize().run()
        m, days = _model(True, target='soil_add', amount=1.0,
                         fail_at=6 * DAY)
        m.initialize()
        with pytest.raises(StepRejected, match='failed once'):
            m.run()
        assert m.runtime.time.t == 6 * DAY
        m.run(until=7 * DAY)
        ck = m.save_checkpoint(str(tmp_path / 'ck.pt'))
        m.run()
        _check(ref, (m, days))
        r, rdays = _model(True, target='soil_add', amount=1.0)
        r.initialize()
        r.load_checkpoint(ck)
        r.run()
    assert len(rdays) == 3
    assert not _diff_state(ref[0].connector('vic').state,
                           r.connector('vic').state)
    for x, y in zip(ref[1][7:], rdays):
        assert all(torch.equal(x[k], y[k]) for k in x)


def test_a_session_alone_runs_ahead_and_restarts(tmp_path):
    """VIC alone (its session runs ahead): a checkpoint inside a block and
    the restart from it give the uninterrupted run."""
    with torch.inference_mode():
        ref, rdays = _model(False)
        ref.initialize().run()
        m, days = _model(True)
        m.initialize()
        m.run(until=6 * DAY)
        ck = m.save_checkpoint(str(tmp_path / 'ck.pt'))
        m.run()
        _check((ref, rdays), (m, days))
        r, sdays = _model(True)
        r.initialize()
        r.load_checkpoint(ck)
        r.run()
    assert not _diff_state(ref.connector('vic').state,
                           r.connector('vic').state)
    for x, y in zip(rdays[6:], sdays):
        assert all(torch.equal(x[k], y[k]) for k in x)
