"""FROZEN_SOIL through the step loop, the QUICK_FLUX path.

With SPATIAL_FROST off -- VIC's default -- Nfrost is 1, frost_fract[0] is 1
and frost_slope is 0, so soil_conduction.c:659-688 collapses to one line per
layer: the ice is whatever the unfrozen-water curve will not hold at that
layer's temperature, clipped into [0, moist]. That single line, the ice
reducing the capacity and the floor the liquid moves between (runoff.c:
155-169), the latent heat of fusion in the surface balance
(func_surf_energy_bal.c:561-574) and the ice in the thermal properties
(soil_conduction.c:711-716) are the whole of the option on this path.

These tests pin the properties that have to hold whatever the numbers are:
the option is inert when off, the water balance still closes when it is on,
the ice is between zero and the moisture, and it actually forms.
"""
import pytest
import torch

from torchvic.model import basin
from torchvic.core import column, frozen

from test_spatial_snow import KW, _case  # noqa: E402

torch.set_default_dtype(torch.float64)


def _frozen_case(bubble=30.0):
    b, atm, p, sp, m, spd = _case()
    # The unfrozen-water curve needs the bubbling pressure; the synthetic
    # SoilParams in _case has none.
    sp.bubble = torch.full((1, 3), bubble, **KW)
    # device=, not the ambient default: another test module in the same
    # session may have set the default device to CUDA, and a bool flag
    # there against a float64 CPU case is a device mismatch inside the
    # residual rather than a wrong answer.
    sp.fs_active = torch.ones(1, dtype=torch.bool,
                              device=KW['device'])
    # Cold enough to freeze: _case's deep temperature is +4 C.
    sp.thermal['avg_temp'] = torch.full((1,), -6.0, **KW)
    return b, atm, p, sp, m, spd


_RUNS = {}


def _run(frozen_soil, **extra):
    key = (frozen_soil, tuple(sorted(extra.items())))
    if key in _RUNS:
        return _RUNS[key]
    b, atm, p, sp, m, spd = _frozen_case(**extra)
    out = basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=spd,
                         full_energy=True, frozen_soil=frozen_soil)
    _RUNS[key] = out
    return out


def test_the_column_is_unchanged_when_nothing_is_frozen():
    """column.step with an all-zero ice array must give bit-identical
    results to no ice at all, or every existing number moves."""
    n = 4
    one = torch.ones(n, **KW)
    sp = column.SoilParams(
        b_infilt=0.2 * one, Ds=0.02 * one, Dsmax=10 * one, Ws=0.8 * one,
        c=2 * one, expt=torch.stack([4 * one] * 3, -1),
        Ksat=torch.stack([250 * one] * 3, -1),
        max_moist=torch.stack([100 * one, 300 * one, 600 * one], -1),
        resid_moist=torch.stack([10 * one] * 3, -1),
        runoff_steps_per_day=24, model_steps_per_day=8)
    state = torch.stack([60 * one, 180 * one, 400 * one], -1)
    ppt = 12.0 * one
    evap = torch.stack([1.0 * one, 0.6 * one, 0.0 * one], -1)
    a = column.step(state, ppt, evap, sp)
    bmz = column.step(state, ppt, evap, sp, ice=torch.zeros_like(state))
    for x, y in zip(a, bmz):
        assert torch.equal(x, y)


def test_ice_caps_the_liquid_a_layer_can_hold():
    """What the option actually does to the column. runoff.c:165-169 forms
    the surface runoff from liquid PLUS ice, so a single step's runoff does
    not change; what changes is that the liquid is held between
    `resid - ice` and `max_moist - ice`, so the layer saturates on less
    water and sends the rest downward.
    """
    n = 4
    one = torch.ones(n, **KW)
    sp = column.SoilParams(
        b_infilt=0.2 * one, Ds=0.02 * one, Dsmax=10 * one, Ws=0.8 * one,
        c=2 * one, expt=torch.stack([4 * one] * 3, -1),
        Ksat=torch.stack([250 * one] * 3, -1),
        max_moist=torch.stack([100 * one, 300 * one, 600 * one], -1),
        resid_moist=torch.stack([10 * one] * 3, -1),
        runoff_steps_per_day=24, model_steps_per_day=8)
    state = torch.stack([60 * one, 180 * one, 400 * one], -1)
    ppt = 40.0 * one
    evap = torch.zeros_like(state)
    ns_dry, ro_dry, _ = column.step(state, ppt, evap, sp)
    last = None
    for ice_v in (0.0, 45.0, 80.0):
        ice = torch.stack([ice_v * one, 0 * one, 0 * one], -1)
        ns, ro, _ = column.step(state, ppt, evap, sp, ice=ice)
        # The surface runoff is formed from the total, so it is unchanged.
        assert torch.allclose(ro, ro_dry), ice_v
        # The liquid stays inside the window the ice leaves it.
        liq0 = ns[:, 0] - ice_v
        assert float(liq0.max()) <= 100.0 - ice_v + 1e-9, (ice_v,
                                                           float(liq0.max()))
        # More ice, less drainage out of the layer, so more water stays.
        if last is not None:
            assert float(ns[:, 0].min()) >= last - 1e-9, (ice_v, float(ns[0, 0]))
        last = float(ns[:, 0].max())


def test_the_option_is_inert_when_off():
    a = _run(False)
    b = _run(False)
    for k, v in a.items():
        if torch.is_tensor(v) and v.is_floating_point():
            assert torch.equal(v, b[k]), k


def test_it_still_needs_a_bubbling_pressure():
    for full_energy in (True, False):
        b, atm, p, sp, m, spd = _frozen_case()
        sp.bubble = None
        with pytest.raises(ValueError, match='bubble'):
            basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=spd,
                           full_energy=full_energy, frozen_soil=True)


def test_water_balance_mode_needs_the_thermal_profile():
    # The ice is estimated against the QUICK_FLUX layer temperatures, so
    # without the thermal properties there is nothing to freeze against.
    b, atm, p, sp, m, spd = _frozen_case()
    sp.thermal = None
    with pytest.raises(ValueError, match='thermal'):
        basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=spd,
                       full_energy=False, frozen_soil=True)


def test_it_runs_in_water_balance_mode():
    """FROZEN_SOIL TRUE with FULL_ENERGY FALSE: VIC runs the QUICK_FLUX
    frozen soil without the surface solve (calc_surf_energy_bal.c:522-525),
    and so does this. The ice forms, stays within the moisture, and the
    water balance closes up to the evaporation VIC withholds."""
    b, atm, p, sp, m, spd = _frozen_case()
    out = basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=spd,
                         full_energy=False, frozen_soil=True)
    ice, liq = out['soil_ice'], out['soil_liq']
    assert torch.isfinite(ice).all()
    assert float(ice.min()) >= -1e-12, float(ice.min())
    assert float(liq.min()) >= -1e-9, float(liq.min())
    assert float(ice.max()) > 1.0, float(ice.max())
    err = out['water_error'].sum(0)
    withheld = out['evap_withheld'].sum(0)
    assert float((err + withheld).abs().max()) < 1e-9, (
        float(err.abs().max()), float(withheld.abs().max()))
    # The rows VIC reports in this mode: the fusion term that joins
    # NetBareRad (put_data.c:928) and the layer temperatures (:693).
    for k in ('eb_fusion', 'eb_layer_temp'):
        assert k in out, k
        assert torch.isfinite(out[k]).all(), k
    assert float(out['eb_fusion'].abs().max()) > 0.0
    # And without frozen soil those rows are absent, as before.
    b, atm, p, sp, m, spd = _frozen_case()
    off = basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=spd,
                         full_energy=False, frozen_soil=False)
    assert 'eb_fusion' not in off and 'eb_layer_temp' not in off
    assert float(off['soil_ice'].abs().max()) == 0.0


def test_water_balance_without_it_is_bit_identical():
    """Every new branch on the water-balance path is behind frozen_soil,
    so two runs without it are the same run to the digit."""
    outs = []
    for _ in range(2):
        b, atm, p, sp, m, spd = _frozen_case()
        outs.append(basin.simulate(b, atm, p, sp, mosaic=m,
                                   steps_per_day=spd, full_energy=False,
                                   frozen_soil=False))
    for k, v in outs[0].items():
        if torch.is_tensor(v) and v.is_floating_point():
            assert torch.equal(v, outs[1][k]), k


def test_ice_forms_and_stays_between_zero_and_the_moisture():
    out = _run(True)
    ice, liq = out['soil_ice'], out['soil_liq']
    assert torch.isfinite(ice).all()
    assert float(ice.min()) >= -1e-12, float(ice.min())
    assert float(liq.min()) >= -1e-9, float(liq.min())
    # It has to actually freeze something, or the test says nothing.
    assert float(ice.max()) > 1.0, float(ice.max())


def test_the_water_balance_closes_up_to_the_evaporation_vic_withholds():
    """runoff.c:108-133 withholds a layer's evaporative demand when it
    has no liquid above residual and does not write that back to
    layer[lindex].evap, so VIC reports the demand and removes less. The
    balance is therefore short by exactly that, in VIC and here. Ice
    makes it common; without ice a column rarely dries to residual,
    which is why the unfrozen runs still close to 1e-14.

    The test is that the shortfall is accounted for to the last digit,
    not that it is absent.
    """
    out = _run(True)
    err = out['water_error'].sum(0)
    withheld = out['evap_withheld'].sum(0)
    assert float((err + withheld).abs().max()) < 1e-9, (
        float(err.abs().max()), float(withheld.abs().max()))
    assert float(withheld.abs().max()) > 0.0


def test_nothing_is_withheld_when_nothing_is_frozen():
    out = _run(False)
    # Exactly zero up to the rounding of `demand - applied`.
    assert float(out['evap_withheld'].abs().max()) < 1e-12
    assert float(out['water_error'].abs().max()) < 1e-9


def test_the_option_moves_the_answer():
    on, off = _run(True), _run(False)
    assert not torch.equal(on['runoff'], off['runoff'])
    assert not torch.equal(on['soil_liq'], off['soil_liq'])
    assert float(off['soil_ice'].abs().max()) == 0.0


def test_frost_fractions_are_vics():
    # vic_init.c:711-725.
    assert frozen.frost_fractions(1) == [1.0]
    assert frozen.frost_fractions(2) == [0.5, 0.5]
    assert frozen.frost_fractions(3) == [0.25, 0.5, 0.25]
    f5 = frozen.frost_fractions(5)
    assert f5 == [0.125, 0.25, 0.25, 0.25, 0.125]
    assert abs(sum(f5) - 1.0) < 1e-15


def test_layer_ice_with_one_area_is_the_one_line_per_layer():
    # With Nfrost 1 the helper is the formula the QUICK_FLUX path is
    # otherwise written as, to the value.
    t = torch.tensor([-6.0, -0.5, 0.0, 2.0], **KW)
    moist = torch.full((4,), 200.0, **KW)
    mx = torch.full((4,), 400.0, **KW)
    bb = torch.full((4,), 30.0, **KW)
    ex = torch.full((4,), 6.0, **KW)
    got = frozen.layer_ice_quick_flux(moist, mx, bb, ex, t, None,
                                      n_frost=1)
    assert got.shape == (4, 1)
    want = torch.clamp(
        moist - frozen.maximum_unfrozen_water(t, mx, bb, ex), min=0.0)
    want = torch.minimum(want, moist)
    assert torch.equal(got[..., 0], want)


def test_layer_ice_spreads_over_the_slope():
    # A layer at -1 C under a 4 C slope: the cold end freezes more than
    # the warm end, every area within [0, moist], and the mid area reads
    # the layer temperature itself.
    moist = torch.full((1,), 200.0, **KW)
    mx = torch.full((1,), 400.0, **KW)
    bb = torch.full((1,), 30.0, **KW)
    ex = torch.full((1,), 6.0, **KW)
    t = torch.full((1,), -1.0, **KW)
    ice = frozen.layer_ice_quick_flux(moist, mx, bb, ex, t, 4.0, n_frost=3)
    assert ice.shape == (1, 3)
    assert bool((ice >= 0.0).all()) and bool((ice <= 200.0).all())
    # The per-area ice carries its area's fraction, so the cold-to-warm
    # order reads on the unweighted values.
    fr = frozen.frost_fractions(3)
    u = [float(ice[0, f]) / fr[f] for f in range(3)]
    assert u[0] > u[1] > u[2]
    mid = torch.clamp(
        moist - frozen.maximum_unfrozen_water(t, mx, bb, ex), min=0.0)
    assert torch.equal(ice[0, 1], 0.5 * mid[0])
    # Frozen soil off somewhere freezes nothing there.
    off = frozen.layer_ice_quick_flux(moist, mx, bb, ex, t, 4.0, n_frost=3,
                                      fs_active=torch.zeros(1, dtype=torch.bool))
    assert float(off.abs().max()) == 0.0


def test_frost_areas_need_a_count_a_slope_and_a_nonnegative_slope():
    for bad in (0, 11, '3', None):
        b, atm, p, sp, m, spd = _frozen_case()
        with pytest.raises(ValueError, match='n_frost'):
            basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=spd,
                           full_energy=False, frozen_soil=True, n_frost=bad)
    b, atm, p, sp, m, spd = _frozen_case()
    with pytest.raises(ValueError, match='frost_slope'):
        basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=spd,
                       full_energy=False, frozen_soil=True, n_frost=3)
    b, atm, p, sp, m, spd = _frozen_case()
    sp.frost_slope = torch.full((1,), -1.0, **KW)
    with pytest.raises(ValueError, match='frost_slope'):
        basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=spd,
                       full_energy=False, frozen_soil=True, n_frost=3)


def _frost_case(full_energy, n_frost=3, slope=4.0):
    b, atm, p, sp, m, spd = _frozen_case()
    sp.frost_slope = torch.full((1,), slope, **KW)
    return (dict(batch=b, atm=atm, p=p, soil_params=sp, mosaic=m,
                 steps_per_day=spd),
            dict(full_energy=full_energy, frozen_soil=True, n_frost=n_frost))


@pytest.mark.parametrize('full_energy', [False, True])
def test_frost_runs_close_and_carry_their_ice(full_energy):
    """SPATIAL_FROST in either mode: every area's ice stays within its
    moisture, the reported ice is the area-weighted ice, the water
    closes up to what VIC withholds, and a split run started from the
    carried state is the single run bit for bit."""
    import test_state_carry as tsc
    kw, extra = _frost_case(full_energy)
    full = basin.simulate(**kw, **extra)
    ice = full['soil_ice']
    assert torch.isfinite(ice).all()
    assert float(ice.min()) >= -1e-12, float(ice.min())
    assert float(ice.max()) > 1.0, float(ice.max())
    err = full['water_error'].sum(0)
    withheld = full['evap_withheld'].sum(0)
    assert float((err + withheld).abs().max()) < 1e-9, (
        float(err.abs().max()), float(withheld.abs().max()))
    # The distribution moves the answer: the same run as one area is a
    # different run once anything freezes.
    kw1, extra1 = _frost_case(full_energy)
    extra1['n_frost'] = 1
    one = basin.simulate(**kw1, **extra1)
    assert not torch.equal(full['soil_ice'], one['soil_ice'])
    assert not torch.equal(full['soil_moisture'], one['soil_moisture'])
    # And the carried per-area ice restarts the run exactly.
    tsc._split_equals_single(kw, [96], **extra)


def test_evap_is_split_by_liquid_and_kept_whole_when_condensing():
    one = torch.ones(2, **KW)
    sp = column.SoilParams(
        b_infilt=0.2 * one, Ds=0.02 * one, Dsmax=10 * one, Ws=0.8 * one,
        c=2 * one, expt=torch.stack([4 * one] * 3, -1),
        Ksat=torch.stack([250 * one] * 3, -1),
        max_moist=torch.stack([100 * one, 300 * one, 600 * one], -1),
        resid_moist=torch.stack([10 * one] * 3, -1),
        runoff_steps_per_day=24, model_steps_per_day=8)
    state = torch.stack([60 * one, 180 * one, 400 * one], -1)
    ice = torch.stack([torch.stack([50 * one, 10 * one], -1),
                       torch.zeros(2, 2, **KW), torch.zeros(2, 2, **KW)],
                      -2)
    # Layer 0: avail 0 and 40; a demand of 8 goes 0 and 16 -- per-area
    # millimetres, unweighted (the weighting is on the columns'
    # outcomes, runoff.c:361-368), so 16 over half the cell is the 8.
    ev = torch.stack([8 * one, torch.zeros_like(one),
                      torch.zeros_like(one)], -1)
    got = column.distribute_evap_frost(ev, state, ice, sp, [0.5, 0.5])
    assert got.shape == (2, 3, 2)
    assert torch.equal(got[:, 0, 0], torch.zeros(2, **KW))
    assert torch.allclose(got[:, 0, 1], 16 * one)
    # Nothing liquid anywhere: zero everywhere, not the demand.
    ice_dry = torch.stack([torch.stack([55 * one, 55 * one], -1),
                           torch.zeros(2, 2, **KW),
                           torch.zeros(2, 2, **KW)], -2)
    got = column.distribute_evap_frost(ev, state, ice_dry, sp, [0.5, 0.5])
    assert float(got.abs().max()) == 0.0
    # Condensation stays whole in every area (runoff.c:136-140).
    ev_neg = torch.stack([-3 * one, torch.zeros_like(one),
                          torch.zeros_like(one)], -1)
    got = column.distribute_evap_frost(ev_neg, state, ice, sp, [0.5, 0.5])
    assert torch.equal(got[:, 0, 0], -3 * one)
    assert torch.equal(got[:, 0, 1], -3 * one)


def test_the_unfrozen_water_curve_carries_a_gradient_to_the_soil():
    """bubble and expt are calibration knobs, so the ice they set has to
    carry a gradient back to them rather than only a value."""
    t = torch.full((6,), -3.0, **KW)
    mx = torch.full((6,), 300.0, **KW)
    bb = torch.full((6,), 30.0, **KW).requires_grad_(True)
    ex = torch.full((6,), 6.0, **KW).requires_grad_(True)
    u = frozen.maximum_unfrozen_water(t, mx, bb, ex)
    u.sum().backward()
    for g in (bb.grad, ex.grad):
        assert g is not None and torch.isfinite(g).all()
        assert float(g.abs().max()) > 0.0
