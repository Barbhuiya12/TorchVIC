"""VIC through the coupled runtime is VIC: every output and the end state
are the direct run's bit for bit (torch.equal), whatever the exchange
interval, on the CPU and on a GPU; the ledger closes VIC's water balance to
VIC's own OUT_WATER_ERROR; and the injection points add water without
touching VIC's equations (precip_add is the same as more forcing
precipitation; soil_add is ledgered and capped)."""
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                                            # noqa: E402
from torchvic import case as _case                       # noqa: E402
from torchvic.coupling import (TorchVIC, Connector, Balance,  # noqa: E402
                               Proposal, StepRejected, VarSpec)
from torchvic.coupling.managers import concat_series     # noqa: E402
from torchvic.coupling.vic import FINAL_KEYS, GRID       # noqa: E402
from torchvic.model import state as model_state          # noqa: E402

STEHE = _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                set_options={'SNOW_BAND': 'FALSE'})
STEHE_DATA = os.path.join(_paths.SAMPLES, 'image', 'Stehekin')
WG = os.path.join(_paths.SAMPLE_DATA, 'image', 'WalnutGulch', 'parameters',
                  'global_param.WalnutGulch.L2015.txt')
CONFIG = os.path.join(_paths.REPO, 'configs', 'vic_only.yaml')

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _stehe_env(monkeypatch):
    monkeypatch.setenv('test_data_dir', _paths.SAMPLES)


def _need_stehe():
    if not (os.path.exists(STEHE) and os.path.isdir(STEHE_DATA)):
        pytest.skip('VIC\'s Stehekin sample is not present')


def _differences(a, b, pre=''):
    bad = []
    for k, v in a.items():
        if isinstance(v, dict):
            if not isinstance(b.get(k), dict):
                bad.append(pre + k + ' missing')
            else:
                bad += _differences(v, b[k], pre + k + '.')
        elif torch.is_tensor(v):
            w = b.get(k)
            if w is None or not torch.equal(v, w):
                bad.append(pre + k)
    for k in b:
        if k not in a:
            bad.append(pre + k + ' extra')
    return bad


def _runtime(gp, device, log=False, connectors=None, connections=(),
             **vic):
    mods = dict(vic=dict(dict(global_param=gp), **vic))
    for k in (connectors or {}):
        mods[k] = dict(connector='two_store')
    return TorchVIC.from_dict(
        dict(run=dict(device=device, log=log), modules=mods,
             connections=list(connections)), connectors=connectors)


def _direct_windows(c, steps):
    """The case run window by window by hand (Case.run on time_window
    slices, the state carried), without the connector."""
    parts, state = [], None
    nt = c.n_steps
    for t0 in range(0, nt, steps):
        w = model_state.time_window(dict(batch=c.batch, atm=c.atm,
                                         subdaily=c.subdaily), t0,
                                    min(t0 + steps, nt))
        extra = dict(batch=w['batch'], atm=w['atm'], return_state=True)
        if state is not None:
            extra['initial_state'] = state
        out = c.run(**extra)
        state = out.pop('state')
        parts.append(out)
    out = concat_series(parts, FINAL_KEYS)
    out['state'] = state
    return out


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('mode', [dict(window_days=0), dict(window_days=3),
                                  dict(exchange='day')])
def test_vic_only_through_the_runtime_is_the_direct_run(device, mode):
    _need_stehe()
    with torch.inference_mode():
        c = _case.load(STEHE, device=device, log=lambda *a: None)
        whole = c.run(return_state=True)
        m = _runtime(STEHE, device, **mode).initialize()
        m.run()
        out = m.outputs('vic')
        steps = m.connector('vic').window_steps()
        direct = whole if steps == c.n_steps else _direct_windows(c, steps)
    bad = _differences(direct, out)
    assert not bad, '%s on %s differs from the direct run in %s' % (
        mode, device, bad[:10])
    if device == 'cpu':
        # on the CPU a split run is the single run (test_state_carry.py)
        assert not _differences(whole, out)
    # the ledger closed VIC's balance on every interval, to VIC's own error
    rows = m.ledger.balances
    assert rows and all(r['ok'] for r in rows)
    n_win = -(-c.n_steps // steps)
    assert len(rows) == n_win
    area = torch.as_tensor(c.dom.area * c.dom.frac, dtype=torch.float64)
    own = float((whole['water_error'].cpu().sum(0) * area / 1000.0).sum())
    closure = sum(r['closure'] for r in rows)
    assert abs(closure - own) <= 1e-9 * max(abs(own), 1.0)
    rep = m.report()['backends']['vic']
    assert rep['device'].startswith(device)
    assert sum(rep['ran'].values()) == n_win


class Sprinkler(Connector):
    """Water from outside the domain onto VIC's cells: `amount` mm per
    interval on every cell (from `start` s on), as precip_add or soil_add.
    Its balance: what it takes in from the boundary it sends on."""
    name = 'sprinkler'
    interval_s = 86400.0
    backends = ('cpu', 'cuda')

    def __init__(self, area, amount, start=0.0):
        super().__init__()
        self.area, self.amount, self.start = area, amount, start
        self.n = int(area.shape[0])
        self.outputs = (VarSpec('water', 'mm', ('cell',), kind='flux',
                                grid=GRID),)

    def initialize(self, ctx):
        self.device = ctx.device

    def compute(self, inputs, ctx, dt):
        a = self.amount if ctx.t >= self.start else 0.0
        x = torch.full((self.n,), a, dtype=torch.float64,
                       device=self.device)
        v = float((x.cpu() * self.area).sum()) / 1000.0
        return Proposal(self.name, ctx.t, ctx.t + dt, exports=dict(water=x),
                        balance=Balance(I_ext=v, Q_out=v, scale=2 * v,
                                        n_terms=2 * self.n))

    def commit(self, p, ctx):
        pass


def _sprinkled(device, amount, target, **vic):
    with torch.inference_mode():
        c = _case.load(STEHE, device=device, log=lambda *a: None)
        area = torch.as_tensor(c.dom.area * c.dom.frac,
                               dtype=torch.float64)
        sp = Sprinkler(area, amount,
                       start=86400.0 if target == 'soil_add' else 0.0)
        m = TorchVIC.from_dict(
            dict(run=dict(device=device, log=False),
                 modules=dict(vic=dict(dict(global_param=STEHE), **vic),
                              sprinkler=dict(connector='two_store')),
                 connections=[{'from': 'sprinkler.water',
                               'to': 'vic.%s' % target}]),
            connectors=dict(sprinkler=sp))
        m.initialize()
        return c, m


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
def test_precip_add_is_more_forcing_precipitation(device):
    """2 mm a day on every cell through precip_add is, bit for bit, the
    run whose forcing precipitation is 2/24 mm larger every hour; the VIC
    and the coupled ledgers close."""
    _need_stehe()
    c, m = _sprinkled(device, 2.0, 'precip_add', exchange='day')
    with torch.inference_mode():
        m.run()
        out = m.outputs('vic')
        spd = c.model_steps
        atm = dict(c.atm)
        atm['prcp'] = atm['prcp'] + (torch.full(
            (spd, c.n_cells), 2.0, dtype=c.dtype, device=c.device) / spd
        ).repeat(c.n_steps // spd, 1)
        direct = _direct_windows_atm(c, spd, atm)
    assert not _differences(direct, out)
    assert all(r['ok'] for r in m.ledger.balances)
    assert m.ledger.coupled and all(r['ok'] for r in m.ledger.coupled)
    vic = m.ledger.totals('vic')
    assert vic['I_ext'] > 0


def _direct_windows_atm(c, steps, atm):
    parts, state = [], None
    for t0 in range(0, c.n_steps, steps):
        w = model_state.time_window(dict(batch=c.batch, atm=atm,
                                         subdaily=c.subdaily), t0,
                                    min(t0 + steps, c.n_steps))
        # the connector adds a [cell] interval total as total / n per step
        extra = dict(batch=w['batch'], atm=w['atm'], return_state=True)
        if state is not None:
            extra['initial_state'] = state
        out = c.run(**extra)
        state = out.pop('state')
        parts.append(out)
    out = concat_series(parts, FINAL_KEYS)
    out['state'] = state
    return out


@pytest.mark.reference
def test_soil_add_is_ledgered_and_capped():
    _need_stehe()
    # refused above capacity by default
    c, m = _sprinkled('cpu', 5000.0, 'soil_add', exchange='day')
    with torch.inference_mode():
        m.step()
        with pytest.raises(StepRejected, match='capacity'):
            m.step()
    assert m.runtime.tick == 1
    # capped: what fits goes in, the rest is handed back, and every
    # balance closes
    c, m = _sprinkled('cpu', 5000.0, 'soil_add', exchange='day',
                      soil_add_mode='cap', exports=['soil_add_unmet'])
    with torch.inference_mode():
        m.run()
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    vic = m.ledger.totals('vic')
    assert 0 < vic['I_ext'] and vic['Q_out'] > 0
    # a small addition fits: VIC's top layer holds it after the boundary
    c, m = _sprinkled('cpu', 1.0, 'soil_add', exchange='day')
    with torch.inference_mode():
        m.run()
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
def test_checkpoint_restart_of_vic_is_the_continuous_run(device, tmp_path):
    _need_stehe()
    with torch.inference_mode():
        a = _runtime(STEHE, device, window_days=2).initialize()
        a.run()
        b = _runtime(STEHE, device, window_days=2).initialize()
        b.run(until=4 * 86400)
        ck = b.save_checkpoint(str(tmp_path / 'vic.pt'))
        r = _runtime(STEHE, device, window_days=2).initialize()
        r.load_checkpoint(ck)
        r.run()
    assert not _differences(a.connector('vic').state,
                            r.connector('vic').state)
    oa, orr = a.outputs('vic'), r.outputs('vic')
    n = orr['runoff'].shape[0]
    for k in ('runoff', 'baseflow', 'swe', 'et', 'discharge_mm'):
        assert torch.equal(oa[k][-n:], orr[k]), k
    assert len(r.ledger.balances) == len(a.ledger.balances)


@pytest.mark.reference
def test_vic_only_yaml_writes_what_torchvic_run_writes(tmp_path):
    if not os.path.exists(WG):
        pytest.skip('the Walnut Gulch sample data is not present')
    env = dict(os.environ, TORCHVIC_SAMPLE_DATA=_paths.SAMPLE_DATA,
               TORCHVIC_GLOBAL=WG,
               PYTHONPATH=_paths.REPO)
    py = [sys.executable, '-m', 'torchvic']
    r = subprocess.run(py + ['validate', '--config', CONFIG], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    a, b = tmp_path / 'direct', tmp_path / 'coupled'
    r = subprocess.run(py + ['run', WG, '--device', 'cpu', '--out', str(a),
                             '--core', '5.1.0', '--quiet'], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run(py + ['run', '--config', CONFIG, '--device', 'cpu',
                             '--out', str(b), '--quiet'], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (b / 'ledger.csv').exists()
    import xarray as xr
    files = sorted(p.name for p in a.iterdir() if p.suffix == '.nc')
    assert files
    for f in files:
        with xr.open_dataset(a / f) as x, xr.open_dataset(b / f) as y:
            assert set(x.data_vars) == set(y.data_vars)
            for v in x.data_vars:
                assert x[v].equals(y[v]), (f, v)


# ---------------------------------------------------------------------------
# Vegetation history (FORCING2) through windows, and the lai input
# ---------------------------------------------------------------------------
def _igp_veghist(tmp_path):
    sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
    try:
        import matrix
    except Exception as e:                               # noqa: BLE001
        pytest.skip('verification/matrix.py: %s' % e)
    rows = [r for r in matrix.MATRIX if r['case'] == 'igp'
            and r['id'].endswith('veghist') and r['mode'] == 'WB']
    if not rows or not os.path.isdir(os.path.join(matrix.VARIANTS,
                                                  'igp_veghist')):
        pytest.skip('the igp vegetation-history variant is not present')
    g = tmp_path / 'global.txt'
    g.write_text(matrix.global_text(rows[0], str(tmp_path), str(tmp_path),
                                    ['OUT_RUNOFF'],
                                    extra_lines=rows[0]['extra']))
    return str(g)


class LaiSource(Connector):
    """Exports, for each VIC interval, LAI per tile and step: the case's own
    history times `factor` (a crop model's place)."""
    name = 'crop'
    backends = ('cpu', 'cuda')

    def __init__(self, lai, spd, factor=1.0, days=31):
        super().__init__()
        self.lai, self.spd, self.factor = lai, spd, factor
        self.interval_s = days * 86400.0
        self.outputs = (VarSpec('lai', '1', ('cell', 'tile'), kind='state',
                                grid=GRID),)

    def compute(self, inputs, ctx, dt):
        s0 = int(round(ctx.t / 86400.0 * self.spd))
        s1 = s0 + int(round(dt / 86400.0 * self.spd))
        return Proposal(self.name, ctx.t, ctx.t + dt, exports=dict(
            lai=self.lai[:, :, s0:s1] * self.factor))

    def commit(self, p, ctx):
        pass


@pytest.mark.reference
def test_vegetation_history_windows_and_the_lai_input(tmp_path):
    g = _igp_veghist(tmp_path)
    steps = 8 * 62
    with torch.inference_mode():
        c = _case.load(g, device='cpu', core='5.1.0', steps=steps,
                       log=lambda *a: None)
        assert getattr(c.mosaic, 'step_axis', False)
        whole = c.run(return_state=True)
        vic = dict(global_param=g, core='5.1.0', steps=steps,
                   window_days=31)
        m = _runtime(g, 'cpu', **{k: v for k, v in vic.items()
                                  if k != 'global_param'})
        m.initialize().run()
        # the windows take their own steps of the per-step vegetation
        assert not _differences(whole, m.outputs('vic'))

        def coupled(factor):
            src = LaiSource(c.mosaic.lai, c.model_steps, factor)
            t = TorchVIC.from_dict(
                dict(run=dict(device='cpu', log=False),
                     modules=dict(vic=dict(vic),
                                  crop=dict(connector='two_store')),
                     connections=[{'from': 'crop.lai', 'to': 'vic.lai'}]),
                connectors=dict(crop=src))
            t.initialize().run()
            return t
        same = coupled(1.0)
        # the history's own LAI fed back through the input: the same run
        assert not _differences(whole, same.outputs('vic'))
        half = coupled(0.5)
    o = half.outputs('vic')
    assert not torch.equal(o['transp'], whole['transp'])
    assert all(r['ok'] for r in half.ledger.balances)
    # a case without a vegetation history refuses the input, saying why
    _need_stehe()
    with torch.inference_mode():
        cs = _case.load(STEHE, device='cpu', log=lambda *a: None)
        src = LaiSource(torch.ones(cs.n_cells, cs.mosaic.n_tiles,
                                   cs.n_steps, dtype=torch.float64),
                        cs.model_steps, days=10)
        t = TorchVIC.from_dict(
            dict(run=dict(device='cpu', log=False),
                 modules=dict(vic=dict(global_param=STEHE, window_days=10),
                              crop=dict(connector='two_store')),
                 connections=[{'from': 'crop.lai', 'to': 'vic.lai'}]),
            connectors=dict(crop=src))
        t.initialize()
        with pytest.raises(StepRejected, match='FROM_VEGHIST'):
            t.step()


@pytest.mark.gpu
@pytest.mark.reference
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
@pytest.mark.parametrize('mode', ['refuse', 'cap'])
def test_soil_add_on_the_lane_layout_is_the_padded_grid(monkeypatch, mode):
    """On a GPU the loop runs on the covered tiles only (LANE_COMPACT);
    soil_add there gives the padded grid's run bit for bit."""
    _need_stehe()
    from torchvic.model import basin
    amount = 1.0 if mode == 'refuse' else 5000.0
    runs = {}
    for compact in (True, False):
        monkeypatch.setattr(basin, 'LANE_COMPACT', compact)
        c, m = _sprinkled('cuda', amount, 'soil_add', exchange='day',
                          soil_add_mode=mode)
        with torch.inference_mode():
            m.run()
        st = m.connector('vic').state
        assert ('lanes' in st) == compact
        assert all(r['ok'] for r in m.ledger.balances)
        assert all(r['ok'] for r in m.ledger.coupled)
        runs[compact] = m.outputs('vic')
    a, b = runs[True], runs[False]
    for k in ('runoff', 'baseflow', 'et', 'swe', 'soil_liq',
              'water_error', 'discharge_mm'):
        assert torch.equal(a[k], b[k]), k


def _history_arrays(d):
    import glob
    import numpy as np
    import xarray as xr
    out = {}
    for f in sorted(glob.glob(os.path.join(str(d), '*.nc'))):
        with xr.open_dataset(f) as x:
            for v in x.data_vars:
                out[(os.path.basename(f), v)] = np.asarray(x[v].values)
    return out


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('mode', [dict(window_days=3, forcing_feed=False),
                                  dict(window_days=3, forcing_feed=True),
                                  dict(exchange='day', forcing_feed=True)])
def test_streamed_forcing_is_the_loaded_case(device, mode, tmp_path):
    """stream: true (coupling/vic_stream.py: the forcing read a window at a
    time by a prefetch thread or the reader processes, the history
    aggregated per window) writes the history and ends on the state of the
    case.load path, bit for bit; a restart in the middle reopens the
    stream where the checkpoint is."""
    import numpy as np
    _need_stehe()
    runs = {}
    for stream in (False, True):
        d = tmp_path / str(stream)
        m = TorchVIC.from_dict(dict(
            run=dict(device=device, log=False),
            modules=dict(vic=dict(global_param=STEHE, core='5.1.0',
                                  stream=stream, **mode)),
            output=dict(dir=str(d))))
        with torch.inference_mode():
            m.initialize()
            m.run()
            m.finalize()
        drv = type(m.connector('vic').driver).__name__
        assert drv == ('StreamDriver' if stream else 'CaseDriver')
        assert all(r['ok'] for r in m.ledger.balances)
        runs[stream] = (_history_arrays(d), m.connector('vic').state)
    (ha, sa), (hb, sb) = runs[False], runs[True]
    assert ha and sorted(ha) == sorted(hb)
    for k in ha:
        assert np.array_equal(ha[k], hb[k], equal_nan=True), k
    assert not _differences(sa, sb)
    # restart from a checkpoint after 4 days: the stream reopens there
    vic = dict(global_param=STEHE, core='5.1.0', stream=True, **mode)
    with torch.inference_mode():
        b = TorchVIC.from_dict(dict(run=dict(device=device, log=False),
                                    modules=dict(vic=vic))).initialize()
        b.run(until=4 * 86400)
        ck = b.save_checkpoint(str(tmp_path / 'vic.pt'))
        b.finalize(write=False)
        r = TorchVIC.from_dict(dict(run=dict(device=device, log=False),
                                    modules=dict(vic=vic))).initialize()
        r.load_checkpoint(ck)
        r.run()
        r.finalize(write=False)
    assert not _differences(sa, r.connector('vic').state)
