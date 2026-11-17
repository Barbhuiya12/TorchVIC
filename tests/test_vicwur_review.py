"""VIC-WUR review fixes (docs/connectors/vicwur.md): the probes of an early
review of the phase scheduler and the step view, and the day-coupled phase
split. CPU only."""
import pytest
import torch

from torchvic.core import frozen
from torchvic.vicwur.order import Options, PhaseScheduler
from torchvic.vicwur.view import SoilView, StepView
from test_vicwur_order import _stub, _conn, _view, _sched, Vic


def test_withdraw_negative_amount_must_not_add_water():
    v = _view()
    before = v.moist.clone()
    v.withdraw(2, torch.tensor([[[-5.0]]]))
    assert float((v.moist - before).max()) <= 0.0


def test_view_default_frost_weights_are_vic_frost_fract():
    ice = torch.zeros(1, 1, 1, 3, 3, dtype=torch.float64)
    ice[..., 2] = 8.0
    v = StepView(torch.full((1, 1, 1, 3), 10.0, dtype=torch.float64),
                 torch.full((1, 3), 30.0, dtype=torch.float64),
                 torch.ones(1, 1, dtype=torch.float64),
                 torch.ones(1, 1, dtype=torch.float64), ice=ice)
    w = frozen.frost_fractions(3)              # [0.25, 0.5, 0.25]
    assert abs(float(v.ice[0, 0, 0, 0]) - 8.0 * w[2]) < 1e-12


def test_banded_hook_applied_is_per_column():
    # basin's day_hook injection is on the run's cells (band columns)
    d, m = _stub(bands=True)
    c = _conn(d, soil_add_bands=True, soil_add_mode='cap')
    moist = torch.full((m, 2, 3), 10.0, dtype=torch.float64)
    new, applied, unmet = c._add_soil_cols(moist, torch.tensor(
        [1.0, 2.0], dtype=torch.float64), None)
    assert applied.shape[0] == new.shape[0]
    # the ledger's total is on cells and weighs the bands by AreaFract
    cells = c._cols_to_cells(applied)
    assert cells.shape[0] == 2
    assert abs(float(cells[0]) - 1.0) < 1e-12
    assert abs(float(cells[1]) - 2.0) < 1e-12


def test_rollback_restores_vic_when_a_later_phase_fails():
    s, _ = _sched(Options(irrigation=True))

    def boom():
        raise RuntimeError('boom')
    s.kinds['irrigation'].pass3 = boom
    with pytest.raises(RuntimeError):
        s.step()                                  # vic.run by default
    assert float(s.kinds['vic'].m) == 0.0


def test_step_needs_a_vic_component():
    s = PhaseScheduler(Options())
    with pytest.raises(ValueError, match="'vic'"):
        s.step(lambda: None)


def test_between_runs_post_of_day_d_then_pre_of_day_d1():
    s, log = _sched(Options(irrigation=True, wateruse=True, wofost=True))
    s.trace = []
    v = _view()
    s.between(v, 100.0, 186500.0)
    ph = [e[0] for e in s.trace]
    t = [e for e in s.trace if e[0] == 'T']
    assert t == [('T', 100.0), ('T', 186500.0)]
    i = ph.index('T', 1)
    assert set(ph[1:i]) <= {'P1', 'P2', 'P3', 'O'} and 'P1' in ph[1:i]
    assert set(ph[i + 1:]) == {'F', 'U'}
    assert 'V' not in ph and log.count('VIC') == 0
    # first step of a run: no plugin_run before the first vic_run
    s.trace = []
    s.between(None, None, 0.0, post=False)
    assert {e[0] for e in s.trace} == {'T', 'F', 'U'}


def test_between_rolls_back_and_returns_delta():
    s, _ = _sched(Options(irrigation=True))
    irr = s.kinds['irrigation']
    v = _view()

    def p3():
        irr.n += 1
        v.add(0, torch.full((1, 1, 1), 2.0))
    irr.pass3 = p3
    d = s.between(v, 0.0, 86400.0)
    assert float(d[0, 0, 0, 0]) == 2.0 and irr.n == 1

    def bad():
        irr.n += 1
        v.add(0, torch.full((1, 1, 1), 2.0))
        raise RuntimeError('x')
    irr.pass3 = bad
    before = v.moist.clone()
    with pytest.raises(RuntimeError):
        s.between(v, 0.0, 86400.0)
    assert irr.n == 1 and torch.equal(v.moist, before)


def test_soil_view_is_a_copy_per_tile_and_band():
    d, m = _stub(bands=True)
    c = _conn(d, soil_add_bands=True, soil_add_floor='zero')
    moist = torch.arange(m * 2 * 3, dtype=torch.float64).reshape(m, 2, 3)
    sv = c._soil_view(moist, None, None)
    x = sv.moist
    assert x.shape == (2, 2, 2, 3)
    assert torch.equal(x[0, :, 1], moist[1]) and torch.equal(
        x[1, :, 0], moist[2])
    x.add_(5.0)                                   # read-only: a copy
    assert float(moist[0, 0, 0]) == 0.0
    assert torch.allclose(sv.area_fract, torch.tensor(
        [[0.25, 0.75], [1.0, 0.0]], dtype=torch.float64))
    # the cached band index is reused
    assert c._band_idx(moist.device)[0] is c._band_idx(moist.device)[0]


def test_phase_group_needs_a_pumping_floor():
    d, _ = _stub()
    with pytest.raises(Exception, match='phase_group'):
        _conn(d, phase_group=object())
