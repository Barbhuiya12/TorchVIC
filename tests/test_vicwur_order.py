"""VIC-WUR's phase scheduler, the step-local view and VicConnector's soil_add
with bands, negative values and frozen soil (docs/connectors/vicwur.md).
CPU only; the VIC-only equality tests prove the off state."""
import types

import numpy as np
import pytest
import torch

from torchvic.coupling.vic import VicConnector
from torchvic.vicwur.order import Options, PhaseScheduler, plan
from torchvic.vicwur.view import StepView

W3 = dict(force_co2=True, routing=True, force_routing=True,
          force_landuse=True, efr=True, wateruse=True, wofost=True,
          irrigation=True, dams=True, local_with=True, remote_with=True,
          nonrenew_with=True)


def _order(**kw):
    return [(c.phase, c.kind, c.hook) for c in plan(Options(**kw))]


def test_full_order_is_plugin_simulation_c():
    got = _order(**W3)
    ph = [g[0] for g in got]
    # F U V P1 P2 P3 O, never out of order
    assert ph == sorted(ph, key=['F', 'U', 'V', 'P1', 'P2', 'P3', 'O'].index)
    assert [g[1] for g in got if g[0] == 'F'] == [
        'co2', 'routing', 'landuse', 'efr', 'wateruse', 'wofost']
    assert [g[1] for g in got if g[0] == 'U'] == [
        'co2', 'wofost', 'irrigation']
    assert [(g[1], g[2]) for g in got if g[0] == 'P1'] == [
        ('irrigation', 'pass1'), ('wofost', 'pass1'),
        ('irrigation', 'pass1_demand'), ('wateruse', 'pass1'),
        ('routing', 'pass1')]
    assert [g[2] for g in got if g[0] == 'P2'] == [
        'sweep:local_dam', 'sweep:route', 'sweep:wu_local',
        'sweep:global_dam', 'sweep:wu_remote']
    assert [(g[1], g[2]) for g in got if g[0] == 'P3'] == [
        ('wateruse', 'pass3_nonrenew'), ('irrigation', 'pass3_withdrawn'),
        ('irrigation', 'pass3'), ('wofost', 'pass3')]
    assert [g[1] for g in got if g[0] == 'O'] == [
        'routing', 'landuse', 'efr', 'dams', 'wateruse', 'irrigation',
        'wofost', 'balance']
    assert all(c.cite for c in plan(Options(**W3)))


def test_gates():
    # nothing on: VIC and the store_error call only
    assert _order() == [('V', 'vic', 'run'), ('O', 'balance', 'put_data')]
    # FORCE_ROUTING needs ROUTING (:42)
    assert ('F', 'routing', 'pre_force') not in _order(force_routing=True)
    # irr_get_withdrawn only with POTENTIAL, WATERUSE or WOFOST (:154-158)
    p3 = [g[2] for g in _order(irrigation=True) if g[0] == 'P3']
    assert p3 == ['pass3']
    p3 = [g[2] for g in _order(irrigation=True, potential_irrigation=True)
          if g[0] == 'P3']
    assert p3 == ['pass3_withdrawn', 'pass3']
    # rout_run runs for non-renewable water use without ROUTING (:115-118)
    assert ('P1', 'routing', 'pass1') in _order(wateruse=True,
                                                nonrenew_with=True)
    # the sweep: slots follow their own flags (:127-139)
    p2 = [g[2] for g in _order(routing=True, wateruse=True, local_with=True)
          if g[0] == 'P2']
    assert p2 == ['sweep:route', 'sweep:wu_local']
    # RANDOM: rout_random_run only, no dams or water-use slots (:142-144)
    p2 = [g[2] for g in _order(routing=True, dams=True, wateruse=True,
                               local_with=True, remote_with=True,
                               decomposition='random') if g[0] == 'P2']
    assert p2 == ['random_run']
    with pytest.raises(ValueError):
        plan(Options(routing=True, decomposition='x'))


class Rec(object):
    def __init__(self, kind, log):
        self.kind, self.log, self.n = kind, log, 0
        for h in ('pre_force', 'pre_vic', 'pass1', 'pass1_demand', 'pass3',
                  'pass3_withdrawn', 'pass3_nonrenew', 'put_data'):
            setattr(self, h, self._mk(h))

    def _mk(self, h):
        return lambda: self.log.append((self.kind, h))

    def sweep_slot(self, lv, slot, cells):
        self.log.append((self.kind, 'sweep', slot, lv))

    def save_state(self):
        return self.n

    def load_state(self, b):
        self.n = b


class Vic(object):
    """VIC as a component: one counter it advances in place."""

    def __init__(self):
        self.m = torch.zeros(1)

    def run(self):
        self.m.add_(1.0)

    def save_state(self):
        return self.m.clone()

    def load_state(self, b):
        self.m.copy_(b)


def _sched(opts, levels=((0, 1), (2,))):
    log = []
    s = PhaseScheduler(opts, [torch.tensor(l) for l in levels])
    s.register('vic', Vic())
    for k in ('co2', 'routing', 'landuse', 'efr', 'wateruse', 'wofost',
              'irrigation', 'dams', 'balance'):
        s.register(k, Rec(k, log))
    return s, log


def test_scheduler_runs_the_plan_with_levels():
    s, log = _sched(Options(**W3))
    s.trace = []
    s.run(lambda: log.append('VIC'))
    # VIC is called between U and P1
    i = log.index('VIC')
    assert log[i - 1] == ('irrigation', 'pre_vic')
    assert log[i + 1] == ('irrigation', 'pass1')
    sweep = [(e[3], e[0], e[2]) for e in log
             if isinstance(e, tuple) and e[1] == 'sweep']
    # one level does all its slots before the next level starts
    assert sweep == [(0, 'dams', 'local_dam'), (0, 'routing', 'route'),
                     (0, 'wateruse', 'wu_local'), (0, 'dams', 'global_dam'),
                     (0, 'wateruse', 'wu_remote'),
                     (1, 'dams', 'local_dam'), (1, 'routing', 'route'),
                     (1, 'wateruse', 'wu_local'), (1, 'dams', 'global_dam'),
                     (1, 'wateruse', 'wu_remote')]
    # the trace matches the log's phase sequence
    assert [t[0] for t in s.trace][0] == 'F' and s.trace[-1][0] == 'O'
    assert log[-1] == ('balance', 'put_data')


def test_scheduler_skips_off_kinds():
    s, log = _sched(Options(irrigation=True))
    s.run(lambda: log.append('VIC'))
    assert {e[0] for e in log if e != 'VIC'} <= {'irrigation', 'balance'}


def test_rollback_restores_components_and_view():
    opts = Options(irrigation=True)
    s, log = _sched(opts)
    irr = s.kinds['irrigation']
    view = _view()

    def pass3():
        irr.n += 5
        view.add(0, torch.full((1, 1, 1), 3.0))
        raise RuntimeError('boom')
    irr.pass3 = pass3
    before = view.moist.clone()
    with pytest.raises(RuntimeError):
        s.step(lambda: None, view)
    assert irr.n == 0
    assert torch.equal(view.moist, before)
    assert float(view.delta.abs().sum()) == 0.0
    # the same step without the failure commits: delta moves into base
    irr.pass3 = lambda: view.add(0, torch.full((1, 1, 1), 3.0))
    got = []
    s.step(lambda: None, view, commit=lambda v: got.append(v.commit()))
    assert float(got[0][0, 0, 0, 0]) == 3.0
    assert float(view.moist[0, 0, 0, 0]) == float(before[0, 0, 0, 0]) + 3.0


def test_rollback_when_commit_fails():
    s, _ = _sched(Options())
    view = _view()
    before = view.moist.clone()

    def bad(v):
        v.add(0, torch.full((1, 1, 1), 1.0))
        v.commit()
        raise RuntimeError('commit')
    with pytest.raises(RuntimeError):
        s.step(lambda: None, view, commit=bad)
    assert torch.equal(view.moist, before)


# ---------------------------------------------------------------- the view
def _view(ice=None, nb=1, af=None, cv=None, moist=None):
    m = moist if moist is not None else torch.full((1, 1, nb, 3), 10.0)
    cv = cv if cv is not None else torch.ones(1, 1)
    af = af if af is not None else torch.full((1, nb), 1.0 / nb)
    mx = torch.tensor([[20.0, 30.0, 40.0]])
    wfc = torch.tensor([[14.0, 20.0, 25.0]])
    return StepView(m, mx, cv, af, ice=ice, wfc=wfc)


def test_view_add_caps_and_leftover():
    v = _view()
    a, lo = v.add(0, torch.tensor([[[4.0]]]))
    assert float(a) == 4.0 and float(lo) == 0.0
    a, lo = v.add(0, torch.tensor([[[20.0]]]))       # room 6
    assert float(a) == 6.0 and float(lo) == 14.0
    assert float(v.moist[0, 0, 0, 0]) == 20.0
    a, lo = v.add(0, torch.tensor([[[-3.0]]]))       # received <= 0: no-op
    assert float(a) == 0.0 and float(lo) == 0.0
    # efficient irrigation fills to Wfc (irr_use.c:249-251); a layer over
    # its Wfc has no room (:255)
    v = _view()
    a, lo = v.add(0, torch.tensor([[[10.0]]]), efficient=torch.tensor([[True]]))
    assert float(a) == 4.0 and float(lo) == 6.0
    v.base[...] = 15.0
    a, _ = v.add(0, torch.tensor([[[1.0]]]), efficient=torch.tensor([[True]]))
    assert float(a) == 0.0


def test_view_reads_edits_so_far_for_shortage():
    v = _view()
    v.withdraw(2, torch.tensor([[[4.0]]]))           # C5 pumps the bottom
    v.add(0, torch.tensor([[[2.0]]]))                # C8 irrigates layer 0
    assert float(v.moist[0, 0, 0, 2]) == 6.0         # C8 reads the pump
    assert float(v.moist[0, 0, 0, 0]) == 12.0
    assert float(v.base[0, 0, 0, 2]) == 10.0         # VIC state untouched


def test_view_withdraw_clips_at_zero_and_skips_dead_columns():
    cv = torch.tensor([[0.5, 0.0]])
    af = torch.tensor([[0.6, 0.4, 0.0]])
    m = torch.full((1, 2, 3, 3), 5.0)
    mx = torch.tensor([[20.0, 30.0, 40.0]])
    v = StepView(m, mx, cv, af)
    taken, unmet = v.withdraw(2, torch.full((1, 2, 3), 8.0))
    assert taken[0, 0, 0] == 5.0 and unmet[0, 0, 0] == 3.0
    assert float(taken[0, 1].abs().sum()) == 0.0     # Cv = 0 tile
    assert float(taken[0, 0, 2]) == 0.0              # AreaFract = 0 band
    assert float(v.moist[0, 0, 0, 2]) == 0.0
    # the cell total: Cv * AreaFract weights
    assert abs(float(v.cell_total()) - (-5.0 * 0.5 * (0.6 + 0.4))) < 1e-6


def test_view_frozen_total_moisture_ice_untouched():
    ice = torch.full((1, 1, 1, 3), 4.0)
    v = _view(ice=ice)
    v.add(0, torch.tensor([[[3.0]]]))
    assert float(v.ice[0, 0, 0, 0]) == 4.0           # irr_use.c:265 total
    assert float(v.liquid[0, 0, 0, 0]) == 9.0
    t, u = v.withdraw(1, torch.tensor([[[9.0]]]), floor='zero')  # C clip: to 0
    assert float(t) == 9.0 and float(v.moist[0, 0, 0, 1]) == 1.0
    t, u = v.withdraw(2, torch.tensor([[[9.0]]]), floor='liquid')
    assert float(t) == 6.0 and float(u) == 3.0       # stops at the ice
    assert float(v.moist[0, 0, 0, 2]) == 4.0
    assert float(v.ice[0, 0, 0, 2]) == 4.0


def test_view_frost_areas_weighted():
    ice = torch.zeros(1, 1, 1, 3, 2)
    ice[..., 0] = 2.0
    ice[..., 1] = 4.0
    v = StepView(torch.full((1, 1, 1, 3), 10.0), torch.full((1, 3), 30.0),
                 torch.ones(1, 1), torch.ones(1, 1), ice=ice,
                 frost_fract=[0.5, 0.5])
    assert float(v.ice[0, 0, 0, 0]) == 3.0


# ------------------------------------------ VicConnector soil_add (stub)
def _stub(bands=False, nl=3, ice=None):
    # 2 cells, 2 tiles; cell 0 has bands (0.25, 0.75), cell 1 one band
    if bands:
        ex = dict(cell=np.array([0, 0, 1]), band=np.array([0, 1, 0]),
                  weight=np.array([0.25, 0.75, 1.0]), n_cell=2)
        m = 3
    else:
        ex, m = None, 2
    cv = torch.tensor([[0.6, 0.4]] * m, dtype=torch.float64)
    soil = types.SimpleNamespace(
        max_moist=torch.tensor([[20., 30., 40.]] * m, dtype=torch.float64),
        resid_moist=torch.tensor([[2., 3., 4.]] * m, dtype=torch.float64))
    d = types.SimpleNamespace(
        dtype=torch.float64, device=torch.device('cpu'), banded=bands,
        n_cells=2, soil=lambda: soil,
        mosaic=lambda: types.SimpleNamespace(cv=cv),
        case=types.SimpleNamespace(snowbands=ex))
    return d, m


def _conn(d, **kw):
    return VicConnector(driver=d, **kw)


def test_defaults_refuse_bands_and_frozen():
    d, m = _stub(bands=True)
    c = _conn(d)
    with pytest.raises(Exception, match='snow bands'):
        c._soil_add_ok({'moist': 0})
    d, m = _stub()
    c = _conn(d)
    with pytest.raises(Exception, match='FROZEN_SOIL'):
        c._soil_add_ok({'soil_ice': torch.zeros(1)})
    c = _conn(d, soil_add_frozen=True)
    c._soil_add_ok({'soil_ice': torch.zeros(1)})
    with pytest.raises(Exception):
        _conn(d, soil_add_floor='x')


def test_soil_add_bands_per_band_and_cell_totals():
    d, m = _stub(bands=True)
    c = _conn(d, soil_add_bands=True, soil_add_mode='cap')
    moist = torch.full((m, 2, 3), 10.0, dtype=torch.float64)
    add = torch.zeros(2, 2, 2, 3, dtype=torch.float64)   # cell,tile,band,layer
    add[0, :, 0, 0] = 4.0                                # band 0 only
    add[0, :, 1, 0] = 20.0                               # band 1: room 10
    add[1, :, 0, 1] = 1.0
    new, applied, unmet = c._add_soil(moist, add, None)
    assert float(new[0, 0, 0]) == 14.0 and float(new[1, 0, 0]) == 20.0
    assert float(new[2, 0, 1]) == 11.0
    # cell total: sum over tiles (cv 1.0 total) and bands (area weight)
    assert abs(float(applied[0]) - (0.25 * 4 + 0.75 * 10)) < 1e-12
    assert abs(float(unmet[0]) - 0.75 * 10) < 1e-12
    assert abs(float(applied[1]) - 1.0) < 1e-12
    # a per-cell [cell] addition is the same mm on every band
    new, applied, _ = c._add_soil(moist, torch.tensor([1.0, 2.0],
                                  dtype=torch.float64), None)
    assert float(new[1, 0, 0]) == 11.0 and float(new[2, 0, 0]) == 12.0
    assert abs(float(applied[0]) - 1.0) < 1e-12


def test_soil_add_bands_off_is_the_cell_path():
    d, m = _stub()
    c = _conn(d, soil_add_mode='cap')
    moist = torch.full((m, 2, 3), 10.0, dtype=torch.float64)
    new, applied, unmet = c._add_soil(moist, torch.tensor(
        [3.0, 30.0], dtype=torch.float64), None)
    assert float(new[0, 0, 0]) == 13.0 and float(new[1, 0, 0]) == 20.0
    assert float(applied[1]) == 10.0 and float(unmet[1]) == 20.0


def test_negative_soil_add_floors():
    d, m = _stub()
    moist = torch.tensor([[[5., 5., 5.]] * 2] * 2, dtype=torch.float64)
    ice = torch.full((2, 2, 3), 3.0, dtype=torch.float64)
    neg = torch.tensor([-8.0, -1.0], dtype=torch.float64)
    # default: below the residual moisture (2) is refused
    with pytest.raises(Exception, match='residual'):
        _conn(d)._add_soil(moist, neg, None)
    # VIC-WUR's clip at 0; the part not taken is unmet and negative
    c = _conn(d, soil_add_floor='zero')
    new, applied, unmet = c._add_soil(moist, neg, None)
    assert float(new[0, 0, 0]) == 0.0 and float(new[1, 0, 0]) == 4.0
    assert float(applied[0]) == -5.0 and float(unmet[0]) == -3.0
    # liquid only: stops at the ice (3)
    c = _conn(d, soil_add_floor='liquid')
    new, applied, unmet = c._add_soil(moist, neg, None, ice=ice)
    assert float(new[0, 0, 0]) == 3.0 and float(applied[0]) == -2.0
    assert float(unmet[0]) == -6.0
    # positive parts are untouched by the floor
    pos = torch.tensor([2.0, 3.0], dtype=torch.float64)
    n1, a1, _ = _conn(d)._add_soil(moist, pos, None)
    n2, a2, _ = _conn(d, soil_add_floor='zero')._add_soil(moist, pos, None)
    assert torch.equal(n1, n2) and torch.equal(a1, a2)


def test_negative_soil_add_with_frost_areas():
    d, m = _stub()
    moist = torch.full((2, 2, 3), 5.0, dtype=torch.float64)
    ice = torch.zeros(2, 2, 3, 2, dtype=torch.float64)
    ice[..., 0], ice[..., 1] = 2.0, 4.0                 # weighted 3
    c = _conn(d, soil_add_floor='liquid', soil_add_frozen=True)
    new, applied, _ = c._add_soil(moist, torch.tensor(
        [-9.0, 0.0], dtype=torch.float64), None, ice=ice)
    assert float(new[0, 0, 0]) == 3.0
    assert float(new[1, 0, 0]) == 5.0


def test_soil_add_is_a_signed_sum_on_the_gpu_layout_free_path():
    # rollback of the connector's addition: _add_soil is pure (no state
    # written), so a rejected step leaves moist as it was
    d, m = _stub()
    c = _conn(d, soil_add_mode='cap')
    moist = torch.full((m, 2, 3), 10.0, dtype=torch.float64)
    keep = moist.clone()
    c._add_soil(moist, torch.tensor([5.0, -1.0], dtype=torch.float64), None)
    assert torch.equal(moist, keep)
