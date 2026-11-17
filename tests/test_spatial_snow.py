"""SPATIAL_SNOW: sub-grid snow cover fraction and the heat the bare part
advects onto the pack.

Both functions are checked the way every other transcription in this suite
is: the C is rewritten as a scalar Python loop, straight from the source with
its branches intact, and the tensor version is asked to reproduce it on
randomly drawn states. A formula that agrees on one state and not on the arm
next to it is exactly the failure this catches.

References are VIC 5.0.1, `vic/vic_run/src/calc_snow_coverage.c` and
`vic/vic_run/src/advected_sensible_heat.c`.
"""
import math
import random

import pytest
import torch

from torchvic.core import snow

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- the C ----

def c_calc_snow_coverage(store_snow, max_slope, old_coverage, swq, old_swq,
                         depth, old_depth, melt, max_snow_depth, snowfall,
                         store_swq, slope, store_coverage):
    """calc_snow_coverage.c:38-160, line for line."""
    if snowfall > 0:
        coverage = 1.0
        if store_snow:
            if store_swq == 0 and old_coverage < 1:
                store_coverage = old_coverage
            elif store_swq == 0:
                store_coverage = 1.0
            store_swq += swq - old_swq
            if depth >= max_slope / 2.:
                store_snow = False
                store_swq = 0.0
                slope = 0.0
                store_coverage = 1.0
        elif old_coverage < 1:
            store_snow = True
            store_swq = swq - old_swq
    elif melt > 0:
        if store_swq > 0 and swq < old_swq:
            store_swq += swq - old_swq
            if store_swq <= 0:
                store_swq = 0.0
                old_coverage = store_coverage
                store_coverage = 1.0
        if store_swq == 0:
            if slope == 0:
                if old_depth > max_slope / 2.:
                    slope = -max_slope
                else:
                    slope = -2. * old_depth
                max_snow_depth = -slope
                store_snow = True
            old_max_snow_depth = max_snow_depth
            max_snow_depth = 2. * depth
            if max_snow_depth < old_max_snow_depth or old_max_snow_depth == 0:
                coverage = -max_snow_depth / slope
                coverage = 1.0 if coverage > 1 else coverage
            else:
                coverage = old_coverage
        else:
            coverage = old_coverage
    else:
        coverage = old_coverage

    return dict(coverage=coverage, store_snow=store_snow,
                store_swq=store_swq, snow_distrib_slope=slope,
                max_snow_depth=max_snow_depth, store_coverage=store_coverage)


def c_advected_sensible_heat(cover, air_dens, t_air, t_grnd, ra):
    """advected_sensible_heat.c:38-70, line for line."""
    q_bare = 1013.0 * air_dens * (t_air - t_grnd) / ra
    if cover > 0.6:
        fs = 1.0
    elif cover > 0.5:
        fs = math.pow(10, 3. * cover - 1.8)
    elif cover > 0.2:
        fs = math.pow(10, 5.6667 * cover - 3.1333)
    else:
        fs = 0.01
    return (q_bare * (1.0 - cover)) / cover * fs


# ------------------------------------------------------------- the states ---

def draw_states(n, seed=20260921):
    """States that reach every arm of the C, not only the common one."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        max_slope = rng.choice([0.1, 0.3, 0.6, 1.0])
        old_depth = rng.choice([0.0, 0.02, 0.1, 0.4, 1.2])
        depth = max(0.0, old_depth + rng.uniform(-0.3, 0.3))
        old_swq = old_depth * rng.uniform(0.1, 0.4)
        swq = max(0.0, depth * rng.uniform(0.1, 0.4))
        snowfall = rng.choice([0.0, 0.0, 0.0, 0.001, 0.02])
        melt = rng.choice([0.0, 0.0, 1e-5, 1e-3, 5e-3])
        slope = rng.choice([0.0, 0.0, -0.1, -0.5])
        state = dict(
            store_snow=rng.random() < 0.5, max_slope=max_slope,
            old_coverage=rng.choice([0.0, 0.25, 0.6, 1.0]),
            swq=swq, old_swq=old_swq, depth=depth, old_depth=old_depth,
            melt=melt, max_snow_depth=rng.choice([0.0, 0.2, 0.9]),
            snowfall=snowfall, store_swq=rng.choice([0.0, 0.0, 0.004]),
            slope=slope, store_coverage=rng.choice([0.3, 1.0]))
        # The C divides by the slope at :133. A state that melts with no
        # depth at all and no slope yet leaves that division at 0/0; the
        # tensor version documents that it holds the old fraction instead,
        # so it is not a state the two are asked to agree on.
        if (snowfall <= 0 and melt > 0 and slope == 0 and old_depth == 0
                and state['store_swq'] == 0):
            continue
        out.append(state)
    return out


STATES = draw_states(600)


def _tensors(states):
    def col(key, dtype=torch.float64):
        return torch.tensor([s[key] for s in states], dtype=dtype)
    return dict(
        store_snow=torch.tensor([s['store_snow'] for s in states]),
        max_slope=col('max_slope'), old_coverage=col('old_coverage'),
        swq=col('swq'), old_swq=col('old_swq'), depth=col('depth'),
        old_depth=col('old_depth'), melt=col('melt'),
        max_snow_depth=col('max_snow_depth'), snowfall=col('snowfall'),
        store_swq=col('store_swq'), snow_distrib_slope=col('slope'),
        store_coverage=col('store_coverage'))


def test_states_reach_every_arm_of_the_c():
    """A transcription test is worth only the arms its states visit."""
    arms = {'accumulate': 0, 'bury': 0, 'open': 0, 'shed': 0, 'expose': 0,
            'shrink': 0, 'hold': 0}
    for s in STATES:
        if s['snowfall'] > 0:
            arms['accumulate'] += 1
            if s['store_snow'] and s['depth'] >= s['max_slope'] / 2:
                arms['bury'] += 1
            if not s['store_snow'] and s['old_coverage'] < 1:
                arms['open'] += 1
        elif s['melt'] > 0:
            if s['store_swq'] > 0 and s['swq'] < s['old_swq']:
                arms['shed'] += 1
            if s['store_swq'] == 0:
                arms['expose'] += 1
                if s['slope'] == 0:
                    arms['shrink'] += 1
        else:
            arms['hold'] += 1
    for name, hits in arms.items():
        assert hits >= 5, (name, arms)


def test_snow_coverage_matches_the_c_on_every_state():
    got = snow.snow_coverage(**_tensors(STATES))
    for i, s in enumerate(STATES):
        want = c_calc_snow_coverage(
            s['store_snow'], s['max_slope'], s['old_coverage'], s['swq'],
            s['old_swq'], s['depth'], s['old_depth'], s['melt'],
            s['max_snow_depth'], s['snowfall'], s['store_swq'], s['slope'],
            s['store_coverage'])
        for key, wanted in want.items():
            mine = got[key][i].item()
            if key == 'store_snow':
                assert bool(mine) == bool(wanted), (i, key, s)
            else:
                assert abs(mine - wanted) < 1e-12, (i, key, mine, wanted, s)


def test_coverage_stays_between_zero_and_one():
    got = snow.snow_coverage(**_tensors(STATES))
    c = got['coverage']
    assert torch.isfinite(c).all()
    assert float(c.min()) >= 0.0, float(c.min())
    assert float(c.max()) <= 1.0, float(c.max())


def test_advected_sensible_heat_matches_the_c():
    rng = random.Random(7)
    cover, dens, tair, tgrnd, ra = [], [], [], [], []
    for _ in range(400):
        cover.append(rng.uniform(0.01, 1.0))
        dens.append(rng.uniform(1.0, 1.4))
        tair.append(rng.uniform(-20, 15))
        tgrnd.append(rng.uniform(-20, 15))
        ra.append(rng.uniform(5, 200))
    t = [torch.tensor(x, dtype=torch.float64)
         for x in (cover, dens, tair, tgrnd, ra)]
    got = snow.advected_sensible_heat(*t)
    for i in range(len(cover)):
        want = c_advected_sensible_heat(cover[i], dens[i], tair[i],
                                        tgrnd[i], ra[i])
        assert abs(got[i].item() - want) < 1e-10, (i, got[i].item(), want)


def test_advected_sensible_heat_is_zero_at_zero_cover():
    """SnowPackEnergyBalance.c:207-211 never calls it at zero cover, and the
    tensor version has to be finite there anyway -- every arm is evaluated."""
    z = torch.zeros(4, dtype=torch.float64)
    got = snow.advected_sensible_heat(
        z, torch.full((4,), 1.2, dtype=torch.float64),
        torch.full((4,), 5.0, dtype=torch.float64), z,
        torch.full((4,), 50.0, dtype=torch.float64))
    assert torch.equal(got, z)


def test_coverage_is_differentiable_in_the_soil_parameter():
    """max_snow_distrib_slope is a calibration knob, so the fraction has to
    carry a gradient back to it, not just a value."""
    # Both entries have to reach :123, `slope = -max_slope`, for the soil
    # parameter to be in the answer at all: old_depth above max_slope / 2,
    # and a shallower depth now so that melt has reduced the cover.
    max_slope = torch.tensor([0.3, 0.5], dtype=torch.float64,
                             requires_grad=True)
    n = 2
    f = torch.full((n,), 0.0, dtype=torch.float64)
    got = snow.snow_coverage(
        store_snow=torch.zeros(n, dtype=torch.bool), max_slope=max_slope,
        old_coverage=torch.ones(n, dtype=torch.float64),
        swq=torch.full((n,), 0.05, dtype=torch.float64),
        old_swq=torch.full((n,), 0.08, dtype=torch.float64),
        depth=torch.full((n,), 0.10, dtype=torch.float64),
        old_depth=torch.full((n,), 0.40, dtype=torch.float64),
        melt=torch.full((n,), 1e-3, dtype=torch.float64),
        max_snow_depth=f, snowfall=f, store_swq=f,
        snow_distrib_slope=f, store_coverage=torch.ones(n,
                                                        dtype=torch.float64))
    got['coverage'].sum().backward()
    assert max_slope.grad is not None
    assert torch.isfinite(max_slope.grad).all()
    assert float(max_slope.grad.abs().max()) > 0.0, max_slope.grad


def test_the_snow_residual_picks_up_advected_sensible_heat():
    """The term has to reach the residual SnowPackEnergyBalance.c:262-264
    forms, not merely exist as a function."""
    n = 3
    one = torch.ones(n, dtype=torch.float64)
    st = dict(
        z=2.0 * one, t_air=2.0 * one, wind=2.0 * one, z0=0.01 * one,
        ra=40.0 * one, long_snow_in=250.0 * one, net_short=30.0 * one,
        air_dens=1.25 * one, eact_air=500.0 * one, lv=2.5e6 * one,
        press=95000.0 * one, vpd=200.0 * one, rain=torch.zeros(n,
                                                               dtype=torch.float64),
        dt=3600.0 * one, swe_surface=0.05 * one,
        old_t_surf=-2.0 * one, surface_liquid_water=torch.zeros(
            n, dtype=torch.float64), t_grnd=1.0 * one,
        snow_depth=0.2 * one, snow_density=250.0 * one,
        blowing_mass_flux=torch.zeros(n, dtype=torch.float64))
    t_surf = -1.0 * one
    without = snow.snow_pack_energy_balance(t_surf, st)
    st_cov = dict(st)
    st_cov['snow_cover_fract'] = torch.tensor([0.3, 0.7, 1.0],
                                              dtype=torch.float64)
    with_cov = snow.snow_pack_energy_balance(t_surf, st_cov)
    # Full cover advects nothing: (1 - f) is zero.
    assert abs(float(with_cov[2] - without[2])) < 1e-12
    # Partial cover does, and the warmer bare ground adds energy.
    assert float(with_cov[0] - without[0]) > 1.0
    assert float(with_cov[1] - without[1]) > 1.0
    # More of it at 0.7 than at 0.3, which is not the naive expectation:
    # the bare fraction is smaller, but Fs -- the share of the bare patch's
    # sensible heat that actually reaches the snow -- is 1 above 0.6 cover
    # and 0.037 at 0.3 (advected_sensible_heat.c:55-65), and Fs wins.
    assert float(with_cov[1]) > float(with_cov[0])


# ---------------------------------------------------------------------- run --
# The option inside the step loop. A synthetic single cell with two vegetation
# tiles, forced cold and snowy for ten days and then warm, so the pack builds
# and melts out -- which is the only regime where a cover fraction between 0
# and 1 can appear at all.
import datetime as dtm  # noqa: E402

from torchvic.model import basin  # noqa: E402
from torchvic.core import column, vegetation

DEVICE = torch.device('cpu')
KW = dict(dtype=torch.float64, device=DEVICE)


def _case(n_steps=192, steps_per_day=8):
    n, nt = 1, 2
    one = torch.ones(n, **KW)
    m = vegetation.VegMosaic(
        cv=torch.tensor([[0.6, 0.4]], **KW),
        lai=torch.full((1, 2, 12), 2.0, **KW),
        albedo=torch.full((1, 2, 12), 0.15, **KW),
        veg_rough=torch.full((1, 2, 12), 0.1, **KW),
        displacement=torch.full((1, 2, 12), 0.6, **KW),
        rmin=torch.full((1, 2), 100.0, **KW),
        rgl=torch.full((1, 2), 100.0, **KW),
        rarc=torch.full((1, 2), 50.0, **KW),
        root_frac=torch.tensor([[[0.3, 0.3, 0.4]] * 2], **KW),
        dtype=torch.float64, device=DEVICE,
        wind_h=torch.full((1, 2), 10.0, **KW),
        rad_atten=torch.full((1, 2), 0.5, **KW),
        wind_atten=torch.full((1, 2), 0.5, **KW),
        trunk_ratio=torch.full((1, 2), 0.2, **KW),
        overstory=torch.tensor([[0.0, 0.0]], **KW))
    # Cold and snowing for the first third, then well above freezing.
    cold = int(n_steps / 3)
    tair = torch.cat([torch.full((cold, n), -8.0, **KW),
                      torch.full((n_steps - cold, n), 9.0, **KW)])
    prcp = torch.cat([torch.full((cold, n), 1.2, **KW),
                      torch.zeros(n_steps - cold, n, **KW)])
    atm = dict(tair=tair, tmax=tair + 2.0, tmin=tair - 2.0, prcp=prcp,
               press=torch.full((n_steps, n), 95000.0, **KW),
               vp=torch.full((n_steps, n), 300.0, **KW),
               eact_air=torch.full((n_steps, n), 300.0, **KW),
               vpd=torch.full((n_steps, n), 200.0, **KW),
               shortwave=torch.full((n_steps, n), 150.0, **KW),
               longwave=torch.full((n_steps, n), 260.0, **KW),
               wind=torch.full((n_steps, n), 2.0, **KW),
               air_dens=torch.full((n_steps, n), 1.25, **KW),
               lv=torch.full((n_steps, n), 2.5e6, **KW))

    class B:
        dates = [dtm.datetime(2008, 1, 1) + dtm.timedelta(
            hours=24 // steps_per_day * i) for i in range(n_steps)]
        elevation = torch.tensor([1000.0], **KW)
        cell_area = torch.tensor([1.0e6], **KW)
        source2outlet = torch.zeros(1, dtype=torch.long, device=DEVICE)
        n_outlets = 1

    sp = column.SoilParams(
        b_infilt=0.2 * one, Ds=0.02 * one, Dsmax=10 * one, Ws=0.8 * one,
        c=2 * one, expt=torch.stack([4 * one] * 3, -1),
        Ksat=torch.stack([250 * one] * 3, -1),
        max_moist=torch.stack([100 * one, 300 * one, 600 * one], -1),
        resid_moist=torch.stack([10 * one] * 3, -1),
        runoff_steps_per_day=24, model_steps_per_day=steps_per_day)
    sp.thermal = dict(avg_temp=4 * one, dp=4 * one,
                      depth=torch.tensor([[0.1, 0.3, 1.0]], **KW),
                      bulk_density=torch.full((1, 3), 1500.0, **KW),
                      soil_density=torch.full((1, 3), 2650.0, **KW),
                      quartz=torch.full((1, 3), 0.4, **KW), rough=0.01 * one,
                      snow_rough=0.001 * one)
    p = {k: torch.tensor(v, **KW) for k, v in dict(
        pcf=1., t_max_snow=.5, t_min_rain=-.5, rs=120., et_beta=1.,
        new_snow_albedo=.85, alb_accum_a=.94, alb_accum_b=.58,
        alb_thaw_a=.82, alb_thaw_b=.46, uh_shape=2.5, uh_scale=2.,
        lai_scale=1., rmin_scale=1., w_crit=.7, w_wilt=.3).items()}
    return B(), atm, p, sp, m, steps_per_day


_RUNS = {}


def _run(spatial, slope=0.6, **extra):
    # The energy balance over 24 days is a few seconds; five tests asking for
    # the same two runs is not, so each pair of arguments is run once.
    key = (spatial, slope, tuple(sorted(extra.items())))
    if key in _RUNS:
        return _RUNS[key]
    b, atm, p, sp, m, spd = _case()
    if slope is not None:
        sp.max_snow_distrib_slope = torch.tensor([slope], **KW)
    out = basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=spd,
                         full_energy=True, spatial_snow=spatial, **extra)
    _RUNS[key] = out
    return out


def test_spatial_snow_off_changes_nothing():
    """The option has to be inert when it is off: every series the loop
    returns identical, not merely close."""
    a = _run(False)
    b = _run(False, slope=None)
    for k, v in a.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        assert torch.equal(v, b[k]), k


def test_the_soil_column_is_required():
    with pytest.raises(ValueError, match='max_snow_distrib_slope'):
        _run(True, slope=None)


def test_the_cover_fraction_is_a_fraction_and_actually_goes_partial():
    out = _run(True)
    c = out['snow_cover']
    assert torch.isfinite(c).all()
    assert float(c.min()) >= -1e-12 and float(c.max()) <= 1.0 + 1e-12
    # The point of the option: somewhere in the melt the cell is neither
    # fully covered nor bare. Without this the test would pass on a model
    # that ignored the option entirely.
    partial = ((c > 1e-6) & (c < 1.0 - 1e-6)).sum()
    assert int(partial) > 0, float(c.max())


def test_the_water_balance_still_closes_with_a_partial_cover():
    """Rain that falls on the bare part of a covered tile bypasses the pack
    (solve_snow.c:304). Losing it is a hole exactly the size of that rain."""
    out = _run(True)
    err = out['water_error'].abs().max()
    assert float(err) < 1e-9, float(err)


def test_the_option_actually_moves_the_answer():
    on, off = _run(True), _run(False)
    assert not torch.equal(on['swe'], off['swe'])
    assert not torch.equal(on['eb_sensible'], off['eb_sensible'])


# ------------------------------------------------------------ water balance --
# FULL_ENERGY FALSE: a pack under SNOW_MIN_SWQ_EB_THRES is folded into the
# surface balance at Tsurf = Tair (_wb_thin_pack), which with SPATIAL_SNOW
# weights the pack's deltaCC, refreeze energy and latent heats by the cover
# (func_surf_energy_bal.c 5.1.0:617-628, :779-781) and recomputes the cover
# (calc_surf_energy_bal.c 5.1.0:680-689).
_WB = {}


def _run_wb(spatial, slope=0.6, trace=False):
    key = (spatial, slope, trace)
    if key in _WB:
        return _WB[key]
    b, atm, p, sp, m, spd = _case()
    if trace:
        # 0.3 mm of snow on each of the first three steps: a pack under
        # SNOW_MIN_SWQ_EB_THRES (1 mm), the thin-pack branch's
        atm['prcp'] = torch.zeros_like(atm['prcp'])
        atm['prcp'][:3] = 0.3
    if slope is not None:
        sp.max_snow_distrib_slope = torch.tensor([slope], **KW)
    out = basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=spd,
                         full_energy=False, spatial_snow=spatial)
    _WB[key] = out
    return out


def test_water_balance_mode_closes_and_goes_partial():
    out = _run_wb(True)
    assert float(out['water_error'].abs().max()) < 1e-9
    c = out['snow_cover']
    assert float(c.min()) >= -1e-12 and float(c.max()) <= 1.0 + 1e-12
    assert int(((c > 1e-6) & (c < 1.0 - 1e-6)).sum()) > 0


def test_water_balance_mode_with_a_cover_that_never_shrinks():
    """A distribution slope far below the pack depth keeps the cover at 1
    until the pack is gone: every cover weight is then exactly 1 or 0, and
    the run is the option-off one (the thin-pack branch included)."""
    for trace in (False, True):
        on = _run_wb(True, slope=1e-9, trace=trace)
        off = _run_wb(False, trace=trace)
        for k in ('swe', 'runoff', 'baseflow', 'evap'):
            if k in off:
                assert torch.allclose(on[k], off[k], rtol=0, atol=1e-12), k
    assert not torch.equal(_run_wb(True)['swe'], _run_wb(False)['swe'])
