"""The coupled runtime's framework: registration, configuration, the
dependency graph, units and grids, the transaction (a failed compute or a
rejected validation commits nothing), checkpoints, and conservation through
the ledger -- all with the conservative two-store test connector, on the
CPU (and on a GPU when there is one)."""
import os

import pytest

torch = pytest.importorskip('torch')

from torchvic.coupling import (TorchVIC, Connector, ConfigError,  # noqa: E402
                               CouplingError, StepRejected, VarSpec,
                               registry, validate_config)
from torchvic.coupling.toy import TwoStoreConnector              # noqa: E402
from torchvic.coupling.managers import Mapping                  # noqa: E402

DAY = 86400.0


def _two(days=6, conn=None, up=None, down=None, mappings=None, ledger=None,
         device='cpu', **run):
    """Upstream two-store -> downstream two-store over `days`."""
    mods = dict(up=dict(dict(connector='two_store', n=4, grid='g_up'),
                        **(up or {})),
                down=dict(dict(connector='two_store', n=4, grid='g_up',
                               precip=0.5), **(down or {})))
    cfg = dict(run=dict(device=device, log=False, duration_days=days, **run),
               modules=mods,
               connections=conn if conn is not None else [
                   {'from': 'up.outflow', 'to': 'down.inflow'}])
    if mappings:
        cfg['mappings'] = mappings
    if ledger:
        cfg['ledger'] = ledger
    return TorchVIC.from_dict(cfg)


# ---------------------------------------------------------------------------
# registration and configuration
# ---------------------------------------------------------------------------
def test_registration_and_planned_connectors_are_explained():
    class Mine(TwoStoreConnector):
        pass
    registry.register('mine_test', lambda **k: Mine(**k), 'test')
    try:
        assert 'mine_test' in registry.available()
        with pytest.raises(CouplingError, match='registered already'):
            registry.register('mine_test', Mine)
        m = TorchVIC.from_dict(dict(run=dict(device='cpu', log=False,
                                             duration_days=2),
                                    modules=dict(a=dict(connector='mine_test'))))
        m.initialize().run()
        assert isinstance(m.connector('a'), Mine)
        assert m.connector('a').name == 'a'
    finally:
        registry.unregister('mine_test')
    # a planned connector enabled: refused with its reason (every connector
    # the registry shipped as planned is implemented by now, so this one is
    # registered here)
    registry.register('plan_test', None, 'planned', 'planned for phase 5')
    try:
        with pytest.raises(ConfigError, match='planned for phase 5'):
            TorchVIC.from_dict(dict(modules=dict(plan_test=dict())))
        # disabled: listed, explained, not run
        m = TorchVIC.from_dict(dict(run=dict(device='cpu', log=False,
                                             duration_days=1),
                                    modules=dict(a=dict(connector='two_store'),
                                                 plan_test=dict(enabled=False))))
        assert 'plan_test' in m.runtime.disabled
        assert 'plan_test' not in m.runtime.connectors
    finally:
        registry.unregister('plan_test')


def test_initialize_and_finalize_are_called_once():
    calls = []

    class Spy(TwoStoreConnector):
        def initialize(self, ctx):
            calls.append('init')
            super().initialize(ctx)

        def finalize(self, ctx):
            calls.append('fin')
    m = TorchVIC.from_dict(dict(run=dict(device='cpu', log=False,
                                         duration_days=3),
                                modules=dict(a=dict(connector='two_store'))),
                           connectors=dict(a=Spy()))
    m.initialize()
    m.initialize()
    m.run()
    m.finalize(write=False)
    m.finalize(write=False)
    assert calls == ['init', 'fin']
    assert m.runtime.tick == 3 and m.runtime.time.done
    assert not m.step()


def test_invalid_config_lists_every_problem(tmp_path):
    p = tmp_path / 'bad.yaml'
    p.write_text('version: 2\nrun: {device: tpu, dtype: half}\n'
                 'modules:\n  vic: {window_days: -1}\n'
                 'connections:\n  - {from: vic, to: nowhere.x, lag: -1}\n'
                 'ledger: {on_violation: ignore}\nbogus: 1\n')
    problems = validate_config(str(p))
    text = '\n'.join(problems)
    for want in ('version', 'run.device', 'run.dtype', 'global_param',
                 'window_days', 'module.variable', "no module 'nowhere'",
                 'lag', 'on_violation', "unknown key 'bogus'"):
        assert want in text, (want, text)
    assert validate_config(str(tmp_path / 'absent.yaml'))
    q = tmp_path / 'env.yaml'
    q.write_text('modules:\n  vic: {global_param: ${NO_SUCH_VAR_X}/g.txt}\n')
    assert any('NO_SUCH_VAR_X' in x for x in validate_config(str(q)))


def test_missing_dependency_and_cycles():
    with pytest.raises(CouplingError, match='missing dependency'):
        TorchVIC.from_dict(dict(run=dict(device='cpu', log=False,
                                         duration_days=1),
                                modules=dict(a=dict(connector='two_store'))),
                           connectors=dict(a=TwoStoreConnector(
                               dependencies=('groundwater',))))
    # a lag-0 cycle without iterate is refused; with a lag it orders
    cyc = [{'from': 'up.outflow', 'to': 'down.inflow'},
           {'from': 'down.outflow', 'to': 'up.inflow'}]
    with pytest.raises(CouplingError, match='cycle without lag'):
        _two(conn=cyc)
    cyc[1]['lag'] = 1
    m = _two(conn=cyc).initialize()
    assert m.runtime.graph.order == ['up', 'down']
    m.run()
    _closes(m)


def test_unit_and_grid_mismatch_are_refused_at_initialisation():
    class Energy(TwoStoreConnector):
        def __init__(self, **k):
            super().__init__(**k)
            self.inputs = (VarSpec('inflow', 'W m-2', ('cell',),
                                   kind='flux', grid=self.grid),)
    with pytest.raises(CouplingError, match='unit mismatch'):
        TorchVIC.from_dict(
            dict(run=dict(device='cpu', log=False, duration_days=1),
                 modules=dict(up=dict(connector='two_store'),
                              down=dict(connector='two_store')),
                 connections=[{'from': 'up.outflow', 'to': 'down.inflow'}]),
            connectors=dict(down=Energy()))
    with pytest.raises(CouplingError, match='grid mismatch'):
        _two(down=dict(grid='g_down'))
    # the same with a conservative mapping runs
    pairs = [[0, 0, 1.0], [1, 0, 1.0], [2, 1, 0.25], [2, 2, 0.75],
             [3, 3, 1.0]]
    m = _two(down=dict(grid='g_down', area_m2=2.0e6),
             conn=[{'from': 'up.outflow', 'to': 'down.inflow',
                    'mapping': 'm'}],
             mappings=dict(m=dict(source='g_up', target='g_down',
                                  pairs=pairs)))
    m.initialize().run()
    _closes(m)
    # a mapping whose fractions do not sum to 1 is not conservative
    with pytest.raises(CouplingError, match='not conservative'):
        Mapping('x', 'a', 'b', [[0, 0, 0.5], [1, 0, 1.0]], 2, 1)
    # two connectors may not own the same store
    class Soil(TwoStoreConnector):
        owned_state = ('soil_moisture',)
    with pytest.raises(CouplingError, match='one writer'):
        TorchVIC.from_dict(dict(run=dict(device='cpu', log=False,
                                         duration_days=1),
                                modules=dict(a=dict(connector='two_store'),
                                             b=dict(connector='two_store'))),
                           connectors=dict(a=Soil(), b=Soil()))


def _snapshot(m):
    return {k: (c.a.clone(), c.b.clone())
            for k, c in m.runtime.connectors.items()}


@pytest.mark.parametrize('lever', ['fail_at', 'reject_at', 'leak'])
def test_a_failed_step_commits_nothing(lever):
    down = {lever: 2 * DAY} if lever != 'leak' else {}
    m = _two(down=down).initialize()
    m.step()
    m.step()
    if lever == 'leak':
        m.connector('down').leak = 0.01
    before = _snapshot(m)
    t, tick = m.runtime.time.t, m.runtime.tick
    rows = len(m.ledger.balances)
    bufs = {k: [list(e) for e in b.entries]
            for k, b in m.runtime.buffers.items()}
    with pytest.raises(StepRejected) as e:
        m.step()
    assert {'fail_at': 'told to fail', 'reject_at': 'told to reject',
            'leak': 'does not close'}[lever] in str(e.value)
    after = _snapshot(m)
    for k in before:     # the upstream computed fine and was not committed
        assert torch.equal(before[k][0], after[k][0])
        assert torch.equal(before[k][1], after[k][1])
    assert (m.runtime.time.t, m.runtime.tick) == (t, tick)
    assert len(m.ledger.balances) == rows
    assert {k: [list(e) for e in b.entries]
            for k, b in m.runtime.buffers.items()} == bufs


def test_ledger_warn_mode_commits_and_records():
    m = _two(down=dict(leak=0.01),
             ledger=dict(on_violation='warn')).initialize()
    m.run()
    assert m.ledger.warnings and m.runtime.time.done


def test_checkpoint_restart_equals_the_continuous_run(tmp_path):
    a = _two(days=8).initialize()
    a.run()
    b = _two(days=8).initialize()
    b.run(until=3 * DAY)
    ck = b.save_checkpoint(str(tmp_path / 'ck.pt'))
    c = _two(days=8).initialize().load_checkpoint(ck)
    assert c.runtime.time.t == 3 * DAY
    c.run()
    for k in ('up', 'down'):
        assert torch.equal(a.connector(k).a, c.connector(k).a)
        assert torch.equal(a.connector(k).b, c.connector(k).b)
    assert len(c.ledger.balances) == len(a.ledger.balances)
    # a checkpoint of another configuration is refused
    d = _two(days=8, down=dict(k_out=0.5)).initialize()
    with pytest.raises(CouplingError, match='another configuration'):
        d.load_checkpoint(ck)


def _closes(m):
    """Every module row and every coupled row closed, and over the whole
    run the water is accounted: inputs - outputs = storage change (the
    stores and the buffers in transit)."""
    assert m.ledger.balances and all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    tot = {k: sum(r[k] for r in m.ledger.coupled)
           for k in ('P', 'ET', 'dS', 'Q_boundary', 'I_boundary',
                     'residual')}
    scale = sum(abs(v) for v in tot.values())
    assert abs(tot['P'] + tot['I_boundary'] - tot['ET'] - tot['Q_boundary']
               - tot['dS']) <= 1e-12 * scale


def test_two_store_transfer_conserves_water_across_rates():
    # same rate, lag 0
    m = _two().initialize()
    m.run()
    _closes(m)
    sent = sum(t['amount'] for t in m.ledger.transfers
               if t.get('leg') == 'sent')
    got = sum(t['amount'] for t in m.ledger.transfers
              if t.get('leg') == 'delivered')
    assert sent > 0 and abs(sent - got) <= 1e-12 * sent
    # a 2-day producer feeding a daily consumer: prorated, in transit
    m = _two(up=dict(interval_s=2 * DAY)).initialize()
    m.run()
    _closes(m)
    assert m.runtime.time.base == DAY
    # a daily producer feeding a 3-day consumer needs a lag
    with pytest.raises(CouplingError, match='lag: 1'):
        _two(days=9, down=dict(interval_s=3 * DAY)).initialize()
    m = _two(days=9, down=dict(interval_s=3 * DAY),
             conn=[{'from': 'up.outflow', 'to': 'down.inflow',
                    'lag': 1}]).initialize()
    m.run()
    _closes(m)
    left = m.runtime._buffer_volume(m.runtime.buffers)
    assert left > 0          # the last 3 days are still in transit


def test_iterated_cycle_converges_and_closes():
    it = {'max_iter': 200, 'tol': 0.0}
    cyc = [{'from': 'up.outflow', 'to': 'down.inflow', 'iterate': it},
           {'from': 'down.outflow', 'to': 'up.inflow', 'iterate': it}]
    m = _two(conn=cyc).initialize()
    assert list(m.runtime.graph.iterative) == [('up', 'down')]
    m.run()
    _closes(m)
    # a state cannot feed a flux input
    with pytest.raises(CouplingError, match='a state cannot feed a flux'):
        _two(conn=[{'from': 'down.store_b', 'to': 'up.inflow'}])


def test_deferred_ledger_checks_detect_after_commit():
    m = _two(ledger=dict(check_every=3)).initialize()
    m.run()
    _closes(m)
    m = _two(down=dict(leak=0.01), ledger=dict(check_every=3)).initialize()
    with pytest.raises(CouplingError, match='deferred ledger check'):
        m.run()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_two_store_on_the_gpu_matches_the_cpu():
    a = _two(device='cpu').initialize()
    a.run()
    b = _two(device='cuda').initialize()
    b.run()
    assert b.connector('down').a.is_cuda
    assert torch.equal(a.connector('down').b, b.connector('down').b.cpu())
    _closes(b)
    assert b.report()['backends']['down']['device'].startswith('cuda')


def test_connector_base_declares_its_info():
    class Bare(Connector):
        name = 'bare'
    i = Bare().info()
    assert i.name == 'bare' and i.backends == ('cpu',)
    with pytest.raises(NotImplementedError):
        Bare().compute({}, None, 1.0)
    assert os.path.basename(__file__).startswith('test_coupling')


# ---------------------------------------------------------------------------
# speculative windows (ledger.speculate): the checked run's results
# ---------------------------------------------------------------------------
def _rows(m):
    keep = ('module', 't0', 't1', 'P', 'I_ext', 'ET', 'Q_out', 'dS',
            'residual', 'tolerance', 'ok')
    return ([tuple(r.get(k) for k in keep) for r in m.ledger.balances],
            [tuple(sorted((k, v) for k, v in r.items()))
             for r in m.ledger.coupled])


def _same_run(a, b):
    assert _rows(a) == _rows(b)
    assert a.runtime.time.t == b.runtime.time.t
    assert a.runtime.tick == b.runtime.tick
    for k in ('up', 'down'):
        ca, cb = a.connector(k), b.connector(k)
        assert torch.equal(ca.a, cb.a) and torch.equal(ca.b, cb.b)
        sa, sb = a.outputs(k), b.outputs(k)
        assert sorted(sa) == sorted(sb)
        assert all(torch.equal(sa[n], sb[n]) for n in sa)
    assert a.report()['backends'] == b.report()['backends']


class _Optimist(TwoStoreConnector):
    """Its first try at day 3 is unusable and says so with a device flag
    that asks for a redo (the runtime's optimistic-path protocol)."""
    tries = 0

    def compute(self, inputs, context, dt):
        from torchvic.coupling.connector import DeviceFlag
        p = super().compute(inputs, context, dt)
        if context.t == 3 * DAY and not context.redo:
            type(self).tries += 1
            p.flags.append(DeviceFlag('optimist: try again',
                                      torch.ones((), dtype=torch.bool),
                                      retry=lambda: None))
        return p


def test_speculative_windows_are_the_checked_run():
    """Steps checked together in one host copy (ledger.speculate: N) commit
    what checking every step commits; a window that meets a failure goes
    back to its start and runs again checked, so the failure -- rejected
    or redone -- is met at the same step with the same committed state;
    a step that raises is met where it raises."""
    ref = _two(days=9).initialize()
    ref.run()
    m = _two(days=9, ledger=dict(speculate=4)).initialize()
    assert m.runtime._spec_ticks() == 4
    seen = []
    m.runtime.outputs_mgr.add_sink(lambda n, p: seen.append((n, p.t0)))
    m.run()
    _same_run(ref, m)
    assert m.runtime.spec_replays == 0
    assert len(seen) == 18 and seen == sorted(seen, key=lambda x: x[1])
    for kw in (dict(down=dict(reject_at=5 * DAY)),
               dict(down=dict(fail_at=5 * DAY)),
               dict(down=dict(leak=0.01))):
        a = _two(days=9, **kw).initialize()
        with pytest.raises(StepRejected) as ea:
            a.run()
        b = _two(days=9, ledger=dict(speculate=4), **kw).initialize()
        with pytest.raises(StepRejected) as eb:
            b.run()
        assert ea.value.t == eb.value.t
        assert ea.value.problems == eb.value.problems
        _same_run(a, b)
        # a step that raises is met where it raises (the steps before it
        # checked then); a check that fails sends the window back
        assert b.runtime.spec_replays == (1 if 'leak' in kw['down'] else 0)
    # a redo asked for by a device flag: redone, and the run is the same
    registry.register('optimist_test', lambda **k: _Optimist(**k), 'test')
    try:
        runs = []
        for sp in (0, 4):
            m = _two(days=9, up=dict(connector='optimist_test'),
                     ledger=dict(speculate=sp)).initialize()
            m.run()
            runs.append(m)
        _same_run(runs[0], runs[1])
        assert runs[1].runtime.spec_replays == 1
    finally:
        registry.unregister('optimist_test')
    # speed-only settings do not change the configuration's digest
    assert _two(days=9, ledger=dict(speculate=4)).runtime.checkpoints.digest \
        == _two(days=9).runtime.checkpoints.digest
