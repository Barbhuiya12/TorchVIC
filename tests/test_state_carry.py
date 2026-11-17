"""Carrying basin.simulate's state from one run into the next.

A calibration spins the model up and then scores a window. The window is
only the model if it starts where a single long run would have been: every
store the time loop carries -- soil, pack, canopy, thermal nodes, frozen
soil, carbon, the snow-free fast path's cache, the DEL* baselines and the
runoff still in the routing -- has to come across. One missed and the split
run is a different model that still closes its water balance, so the test
is bit-identity of every output over B, not a tolerance.

All on the CPU, where the arithmetic is deterministic; a split that is
bit-identical there is the same computation on any device.
"""
import datetime as dtm
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                                           # noqa: E402
REPO = _paths.REPO
from torchvic.model import basin  # noqa: E402
from torchvic.forcing import daily as daily_forcing
from torchvic.model import state as model_state
import test_spatial_snow as tss           # noqa: E402

CPU = torch.device('cpu')
KW = dict(dtype=torch.float64, device=CPU)
B37 = _paths.B37_GLOBAL
WG = os.path.join(_paths.SAMPLE_DATA, 'image', 'WalnutGulch',
                  'parameters', 'global_param.WalnutGulch.L2015.txt')

# Outputs that are the state at the end of the run rather than a series.
FINAL = ('canopy_storage', 'soil_moisture', 'soil_moisture_tile')


def _differences(full, part, t0, name=''):
    """Every tensor in `full` from step t0 against `part`, recursively;
    returns the names that are not bit-identical (or missing)."""
    bad, n = [], 0
    for k, v in full.items():
        key = name + k
        if isinstance(v, dict):
            if not isinstance(part.get(k), dict):
                bad.append(key + ' missing')
                continue
            b, m = _differences(v, part[k], t0, key + '.')
            bad += b
            n += m
            continue
        if not torch.is_tensor(v):
            continue
        n += 1
        ref = v if (k in FINAL or v.dim() == 0) else v[t0:]
        if k not in part:
            bad.append(key + ' missing')
        elif not torch.equal(ref, part[k]):
            d = (ref - part[k]).abs()
            d = d[torch.isfinite(d)]
            bad.append('%s %.3e' % (key, float(d.max()) if d.numel() else
                                    float('nan')))
    return bad, n


def _split_equals_single(kw, cuts, **extra):
    """Run `kw` once, and again cut at `cuts` with each piece started from
    the previous piece's state; every piece's outputs must be the single
    run's. Returns the single run and the last piece's state."""
    full = basin.simulate(**kw, **extra)
    nt = int(kw['atm']['tair'].shape[0])
    edges = [0] + list(cuts) + [nt]
    state = None
    for a, b in zip(edges[:-1], edges[1:]):
        out = basin.simulate(**model_state.time_window(kw, a, b),
                             initial_state=state, return_state=True, **extra)
        sub = {k: (v[:b] if torch.is_tensor(v) and k not in FINAL
                   and v.dim() and v.shape[0] == nt else v)
               for k, v in full.items()}
        if b < nt:
            sub = {k: v for k, v in sub.items() if k not in FINAL}
        bad, n = _differences(sub, out, a)
        assert n > 40, n
        assert not bad, 'steps %d..%d differ from the single run: %s' % (
            a, b, bad[:10])
        state = out['state']
    return full, state


def _synthetic(start, overstory=True, options=True, n_steps=96, spd=8):
    """The spatial-snow test's one cell with a snowy first third, dated
    `start`; with `options`, SPATIAL_SNOW, FROZEN_SOIL, BLOWING and CARBON
    all on, so every optional store is carried."""
    b, atm, p, sp, m, _ = tss._case(n_steps=n_steps, steps_per_day=spd)
    b.dates = [start + dtm.timedelta(hours=24 // spd * i)
               for i in range(n_steps)]
    if overstory:
        m.overstory = torch.tensor([[1.0, 0.0]], **KW)
    extra = {}
    if options:
        sp.max_snow_distrib_slope = torch.tensor([0.6], **KW)
        sp.bubble = torch.full((1, 3), 30.0, **KW)
        sp.depth = sp.thermal['depth'].clone()
        sp.fs_active = torch.ones(1, dtype=torch.bool, device=CPU)
        sp.thermal['avg_temp'] = torch.full((1,), -2.0, **KW)
        like = torch.ones_like(m.cv)
        m.sigma_slope, m.lag_one, m.fetch = like * 0.005, like * 0.95, \
            like * 2000.0
        m.carbon = {k: torch.full_like(m.cv, v) for k, v in dict(
            Ctype=0.0, MaxCarboxRate=3.0e-5, MaxETransport=6.0e-5,
            CO2Specificity=0.0, LightUseEff=0.5, NscaleFlag=1.0,
            Wnpp_inhib=0.5, NPPfactor_sat=0.5).items()}
        atm['par'] = daily_forcing.SW2PAR * atm['shortwave']
        atm['Catm'] = torch.full_like(atm['shortwave'],
                                      daily_forcing.CATM_CURRENT_PPM * 1e-6)
        atm['fdir'] = torch.full_like(atm['shortwave'], 0.6)
        atm['coszen'] = torch.full_like(atm['shortwave'], 0.5)
        atm['wind'] = torch.full_like(atm['shortwave'], 9.0)
        extra = dict(spatial_snow=True, frozen_soil=True, blowing=True,
                     carbon=True, carbon_options=dict(
                         initial_pools=(500.0, 2000.0, 8000.0),
                         annual_npp_prev=120.0))
    return dict(batch=b, atm=atm, p=p, soil_params=sp, mosaic=m,
                steps_per_day=spd), extra


def _wg_cold(steps_per_day_days=10, first_day=20, **load):
    """Walnut Gulch 3-hourly with the energy balance, made to snow. Its own
    2008 forcing leaves no pack at all (the whole year scanned: the
    coldest precipitation, days 46 and 360-361, melts in the step it
    falls; SWE 0.0000 mm at every step), so the forcing is cooled by 12 C
    and its precipitation multiplied by 8: over 21-30 January the pack
    builds to 39.7 mm, which is the state being carried."""
    if not os.path.exists(WG):
        pytest.skip('the Walnut Gulch sample data is not present')
    from torchvic import case
    from torchvic.core import snow
    # the sample global file runs 1-10 January; this one runs the year
    gf = _paths.relocated_global(WG, set_options={'ENDMONTH': '12',
                                                  'ENDDAY': '31'})
    c = case.load(gf, device=CPU, steps=8 * steps_per_day_days,
                  first_step=8 * first_day, log=lambda *a: None, **load)
    a = c.atm
    a['tair'] = a['tair'] - 12.0
    a['prcp'] = a['prcp'] * 8.0
    sat = snow.svp(a['tair'])
    a['eact_air'] = 0.6 * sat
    a['vpd'] = sat - a['eact_air']
    return c


# --------------------------------------------------------------------------
# Synthetic: every optional store on, one cell, a few seconds a run.


def test_every_option_splits_bit_for_bit_across_new_year():
    """SPATIAL_SNOW, FROZEN_SOIL, BLOWING, CARBON and a covered overstory,
    cut exactly at 1 January (vic_run.c:301-306 rolls the annual NPP over
    between the two runs) and again mid-melt: 93 outputs, all equal."""
    kw, extra = _synthetic(dtm.datetime(2007, 12, 28))
    _, st = _split_equals_single(kw, [32, 50], full_energy=True, **extra)
    for k in ('spatial_snow', 'soil_ice', 'carbon', 'canopy',
              'ground_long_out', 't_node'):
        assert st[k] is not None, k


@pytest.mark.parametrize('full_energy', [True, False])
def test_snow_and_canopy_split_at_several_points(full_energy):
    """The pack building, holding and melting, with the canopy's snow and
    (water balance) the quick-flux T0/T1: three cuts in one chain."""
    kw, _ = _synthetic(dtm.datetime(2008, 1, 5), options=False)
    _, st = _split_equals_single(kw, [20, 40, 70], full_energy=full_energy)
    assert (st['wb_t'] is None) == full_energy


def test_the_state_is_needed():
    """Starting B cold instead of from A's state has to move B's outputs --
    otherwise the tests above prove nothing about the carry."""
    kw, _ = _synthetic(dtm.datetime(2008, 1, 5), options=False)
    full = basin.simulate(**kw, full_energy=True)
    cold = basin.simulate(**model_state.time_window(kw, 20), full_energy=True)
    assert not torch.equal(full['swe'][20:], cold['swe'])
    assert not torch.equal(full['discharge_mm'][20:], cold['discharge_mm'])


def test_discharge_needs_the_routing_tail():
    """The routing tail alone: B from A's state with the tail emptied loses
    the runoff A's steps were still delivering, and nothing else."""
    kw, _ = _synthetic(dtm.datetime(2008, 1, 5), options=False)
    full = basin.simulate(**kw, full_energy=False)
    a = basin.simulate(**model_state.time_window(kw, 0, 40), full_energy=False,
                       return_state=True)
    st = dict(a['state'])
    assert st['routing_prefix'].shape[0] > 0
    b_kw = model_state.time_window(kw, 40)
    no_tail = basin.simulate(**b_kw, full_energy=False, initial_state=st,
                             routing_prefix=st['routing_prefix'][:0])
    assert torch.equal(no_tail['runoff'], full['runoff'][40:])
    assert not torch.equal(no_tail['discharge_mm'], full['discharge_mm'][40:])
    # An explicit prefix is the same as the state's own.
    with_tail = basin.simulate(**b_kw, full_energy=False, initial_state=st,
                               routing_prefix=st['routing_prefix'])
    assert torch.equal(with_tail['discharge_mm'], full['discharge_mm'][40:])


def test_the_callers_state_is_not_written_into():
    """B run twice from one state gives the same B: the loop copies the
    state's dicts and never writes into its tensors."""
    kw, extra = _synthetic(dtm.datetime(2007, 12, 28))
    a = basin.simulate(**model_state.time_window(kw, 0, 30), full_energy=True,
                       return_state=True, **extra)
    snap = model_state.map_state(a['state'], lambda t: t.clone())
    b_kw = model_state.time_window(kw, 30)
    o1 = basin.simulate(**b_kw, full_energy=True, initial_state=a['state'],
                        **extra)
    o2 = basin.simulate(**b_kw, full_energy=True, initial_state=a['state'],
                        **extra)
    assert not _differences(o1, o2, 0)[0]

    def same(x, y):
        if torch.is_tensor(x):
            return torch.equal(x, y)
        if isinstance(x, dict):
            return set(x) == set(y) and all(same(x[k], y[k]) for k in x)
        return x == y
    assert same(snap, a['state'])


def test_a_state_from_another_configuration_is_refused():
    kw, extra = _synthetic(dtm.datetime(2008, 1, 5), options=False)
    a = basin.simulate(**model_state.time_window(kw, 0, 16), full_energy=True,
                       return_state=True)
    b_kw = model_state.time_window(kw, 16)
    with pytest.raises(ValueError, match='t_node|wb_t'):
        basin.simulate(**b_kw, full_energy=False, initial_state=a['state'])
    with pytest.raises(ValueError, match='steps a day'):
        basin.simulate(**dict(b_kw, steps_per_day=4), full_energy=True,
                       initial_state=a['state'])
    with pytest.raises(ValueError, match='version'):
        basin.simulate(**b_kw, full_energy=True,
                       initial_state=dict(a['state'], version=0))


# --------------------------------------------------------------------------
# Gradients: through the state, and cut at it.


NAMES = ('pcf', 'rmin_scale', 'lai_scale', 'new_snow_albedo', 'uh_scale')


def _leaves(p):
    p = dict(p)
    for k in NAMES:
        p[k] = p[k].clone().requires_grad_(True)
    return p


def _loss(o):
    return (o['et'].sum() + o['discharge_mm'].sum() + o['swe'].sum()
            + o['soil_liq'].sum())


def test_gradient_through_an_attached_state_is_the_single_runs():
    """B from A's state, not detached: B's loss has the same gradient, to
    the bit, as the single run's loss restricted to B's steps."""
    kw, _ = _synthetic(dtm.datetime(2008, 1, 5), options=False)
    p1 = _leaves(kw['p'])
    full = basin.simulate(**dict(kw, p=p1), full_energy=True)
    sl = {k: v[40:] for k, v in full.items() if torch.is_tensor(v)
          and v.dim() and k not in FINAL}
    g1 = torch.autograd.grad(_loss(sl), [p1[k] for k in NAMES])
    p2 = _leaves(kw['p'])
    kw2 = dict(kw, p=p2)
    a = basin.simulate(**model_state.time_window(kw2, 0, 40), full_energy=True,
                       return_state=True)
    b = basin.simulate(**model_state.time_window(kw2, 40), full_energy=True,
                       initial_state=a['state'])
    g2 = torch.autograd.grad(_loss(b), [p2[k] for k in NAMES])
    for k, x, y in zip(NAMES, g1, g2):
        assert bool(torch.isfinite(y).all()), k
        assert torch.equal(x, y), (k, x, y)
    assert any(float(x.abs().max()) > 0 for x in g1)


def test_warm_up_gives_the_window_and_cuts_the_graph():
    """warm_up spins up without a graph and hands the window its state.
    With `exact` the warm-up runs in grad mode and the window is the single
    run bit for bit. Without it the warm-up is under no_grad, where
    implicit.root_with_grad skips its Newton correction (implicit.py:318),
    so the energy-balance window is the single run to solver tolerance:
    measured 9.6e-13 mm at most here. Either way the state holds no graph
    and the window's gradient is finite."""
    kw, _ = _synthetic(dtm.datetime(2008, 1, 5), options=False)
    full = basin.simulate(**kw, full_energy=True)
    # The exact window runs in grad mode (leaf parameters), where
    # root_with_grad adds its Newton correction; the single run it is
    # compared with bit for bit must be in grad mode too (a no-grad run
    # differs from it by solver tolerance, 1.8e-15 W/m2 in eb_sensible).
    full_g = basin.simulate(**dict(kw, p=_leaves(kw['p'])), full_energy=True)
    for exact in (True, False):
        p = _leaves(kw['p'])
        st, win = model_state.warm_up(dict(kw, p=p, full_energy=True), n_steps=40,
                                exact=exact)
        assert not any(t.requires_grad for t in _tensors(st))
        out = basin.simulate(**win)
        if exact:
            assert not _differences(full_g, out, 40)[0]
        else:
            for k in ('swe', 'runoff', 'baseflow', 'et', 'discharge_mm'):
                assert float((full[k][40:] - out[k]).abs().max()) < 1e-10, k
        g = torch.autograd.grad(_loss(out), [p[k] for k in NAMES],
                                allow_unused=True)
        for k, x in zip(NAMES, g):
            assert x is None or bool(torch.isfinite(x).all()), k
    with pytest.raises(ValueError):
        model_state.warm_up(kw, ndays=1, n_steps=8)


def _tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _tensors(v)


# --------------------------------------------------------------------------
# Real domains.


@pytest.mark.skipif(not os.path.exists(B37),
                    reason='India_basin37_Nandgaon not built')
def test_basin37_daily_water_balance_splits_bit_for_bit():
    """Basin 37, 100 cells, the daily water balance through the monsoon
    onset (days 150-209 of 2000), cut at day 30 and 45."""
    from torchvic import case
    c = case.load(B37, device=CPU, steps=60, first_step=150, daily=True,
                  no_energy=True, log=lambda *a: None)
    full, st = _split_equals_single(c.kwargs(), [30, 45])
    assert float(full['runoff'].sum()) > 0.0
    assert st['routing_prefix'].shape[1] == c.n_cells


@pytest.mark.skipif(not os.path.exists(B37),
                    reason='India_basin37_Nandgaon not built')
def test_basin37_hourly_energy_balance_splits_bit_for_bit():
    """Basin 37 hourly with the full energy balance, two days of July,
    cut at hour 29 (a night-time step)."""
    from torchvic import case
    c = case.load(B37, device=CPU, steps=2, first_step=190,
                  log=lambda *a: None)
    assert c.model_steps == 24 and c.full_energy
    _split_equals_single(c.kwargs(), [29])


def test_walnut_gulch_snowpack_splits_bit_for_bit():
    """Walnut Gulch 3-hourly energy balance with a pack (see _wg_cold), cut
    while it is 15 mm and growing and again at 38 mm."""
    c = _wg_cold()
    full, _ = _split_equals_single(c.kwargs(), [56, 72])
    assert float(full['swe'][55].max()) > 10.0


def test_an_ensemble_splits_bit_for_bit():
    """Three members of Walnut Gulch with different parameters, as a
    calibration's population runs them, through the same split."""
    c = _wg_cold(steps_per_day_days=6, first_day=24)
    e = c.ensemble(3)
    p = e.member_params({
        'rmin_scale': torch.tensor([0.7, 1.0, 1.4], **KW),
        'pcf': torch.tensor([0.8, 1.0, 1.2], **KW)})
    full, st = _split_equals_single(e.kwargs(p=p), [25])
    assert st['moist'].shape[0] == 3 * c.n_cells
    q = full['discharge_mm']
    assert q.shape[1] == 3 and not torch.equal(q[:, 0], q[:, 2])


def test_a_state_refuses_a_window_that_does_not_follow_it():
    """The state records the last date it saw; a window starting anywhere
    but the next step is refused unless allow_gap says the reuse is
    deliberate (review of the state-carry track)."""
    kw, extra = _synthetic(dtm.datetime(2001, 3, 1), options=False,
                           n_steps=48)
    sa = basin.simulate(**model_state.time_window(kw, 0, 24), return_state=True,
                        **extra)['state']
    gap = model_state.time_window(kw, 25, 48)          # skips one step
    with pytest.raises(ValueError, match='allow_gap'):
        basin.simulate(**gap, initial_state=sa, **extra)
    basin.simulate(**gap, initial_state=sa, allow_gap=True, **extra)
    # the contiguous window is accepted without it
    basin.simulate(**model_state.time_window(kw, 24, 48), initial_state=sa,
                   **extra)
