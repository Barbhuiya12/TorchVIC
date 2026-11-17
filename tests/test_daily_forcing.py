"""torchvic.forcing.daily: a daily record assembled into VIC 5's forcing set.

The wrapper around MTCLIM (torchvic.forcing.mtclim, tests/test_mtclim.py)
that a forcing set needs, built as MetSim's metsim/disaggregate.py builds
it: precipitation spread over the day, wind held flat, pressure from
elevation and temperature, vapour pressure interpolated between the times
of the daily minimum, and longwave from cloud fraction and humidity.

calc_longwave() is checked against a numpy port of MetSim's formulas and,
as a measured difference, against VIC 4.2.d's compiled function (the two
share every coefficient and differ only in the Stefan-Boltzmann constant).
The rest is checked by the properties its arithmetic forces: what is
conserved, what is held constant, and where the interpolation passes exactly
through its knots.
"""
import datetime
import math
import pathlib
import subprocess

import numpy as np
import pytest
import torch

from torchvic.forcing import daily as df
from torchvic.forcing import mtclim

import _paths                                           # noqa: E402
REF = pathlib.Path(_paths.oracle('ref_mtclim'))
DT = torch.float64
#: Ten standalone scripts in this directory call torch.set_default_device at
#: module scope and pytest imports every test_*.py when it collects, so by
#: the time these run the process default may already be CUDA. Take the live
#: default rather than fight it -- what matters is that the reference
#: tensors and the ones the model builds land in the same place, not which
#: place that is.
DEV = torch.empty(0).device
START = datetime.date(1970, 3, 1)
NDAYS = 60


def _days(n=NDAYS, seed=0):
    """A seasonal temperature cycle with dry spells in the precipitation,
    so the 90-day window and the rain-day transmittance scalar both bite."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    i = torch.arange(n, dtype=DT, device=DEV)
    base = 8.0 - 14.0 * torch.cos(2.0 * torch.pi * i / 365.0)
    wet = torch.rand(n, generator=g, dtype=DT, device=DEV) < 0.35
    return dict(
        tmax=(base + 4.0 + 6.0 * torch.rand(n, generator=g, dtype=DT, device=DEV))[None],
        tmin=(base - 4.0 - 6.0 * torch.rand(n, generator=g, dtype=DT, device=DEV))[None],
        prcp=(wet * 25.0 * torch.rand(n, generator=g, dtype=DT, device=DEV))[None].to(DT),
        wind=(1.0 + 4.0 * torch.rand(n, generator=g, dtype=DT, device=DEV))[None],
    )


def _run(days=None, **kw):
    kw.setdefault('steps_per_day', 24)
    kw.setdefault('device', DEV)
    return df.daily_to_subdaily(days if days is not None else _days(), START,
                                47.8, -121.5, 1300.0, **kw)


# ---------------------------------------------------------------------------
# calc_longwave, against the compiled function
# ---------------------------------------------------------------------------
def _metsim_longwave(lw, cloud, air_c, vp_pa, tskc):
    """metsim.disaggregate.longwave, one value at a time."""
    t = air_c + 273.15
    vp = vp_pa / 100.0
    clear = {
        'TVA': lambda: 0.74 + 0.0049 * vp,
        'ANDERSON': lambda: 0.68 + 0.036 * math.sqrt(vp),
        'BRUTSAERT': lambda: 1.24 * (vp / t) ** 0.14285714,
        'SATTERLUND': lambda: 1.08 * (1 - math.exp(-(vp ** (t / 2016)))),
        'IDSO': lambda: 0.7 + 5.95e-5 * vp * math.exp(1500 / t),
        'PRATA': lambda: (1 - (1 + 46.5 * vp / t) * math.exp(
            -math.sqrt(1.2 + 3.0 * (46.5 * vp / t)))),
    }[lw]()
    eps = (tskc + (1 - tskc) * clear if cloud
           else (1.0 + 0.17 * tskc ** 2) * clear)
    return eps * 5.669e-8 * t ** 4


def test_longwave_matches_metsims_formulas():
    """All six clear-sky emissivities and both cloud methods, over the range
    MTCLIM produces."""
    worst = 0.0
    for lw in df.LW_TYPES:
        for cloud in (True, False):
            for at in (-30.0, -5.0, 10.0, 25.0, 40.0):
                for vp in (50.0, 400.0, 1500.0, 4000.0):
                    for tskc in (0.0, 0.3, 0.7, 1.0):
                        got = float(df.calc_longwave(
                            torch.tensor(tskc, dtype=DT, device=DEV),
                            torch.tensor(at, dtype=DT, device=DEV),
                            torch.tensor(vp, dtype=DT, device=DEV),
                            lw_type=lw, lw_cloud_deardorff=cloud))
                        ref = _metsim_longwave(lw, cloud, at, vp, tskc)
                        worst = max(worst, abs(got - ref) / abs(ref))
    assert worst < 1e-12, worst


def test_longwave_differs_from_vic42d_only_by_the_stefan_boltzmann_constant():
    """VIC 4.2.d's calc_longwave.c has the same six emissivities and the same
    cloud methods; MetSim's constant is 5.669e-8 where VIC 4's is 5.6696e-8.
    Measured against the compiled function: every value is lower by the
    same relative 1.06e-4 (0.03 W/m2 at 300 W/m2) and by nothing else."""
    if not REF.exists():
        pytest.skip('ref_mtclim missing; run '
                    'tests/oracle/build_ref_mtclim.sh')
    out = subprocess.run([str(REF), 'longwave', '0'], capture_output=True,
                         text=True, check=True).stdout
    rows = [ln.replace('|', ',').split(',') for ln in out.splitlines()
            if not ln.startswith('#')]
    assert len(rows) == 6 * 2 * 9 * 7 * 11
    want = 5.669e-8 / 5.6696e-8 - 1.0
    worst = 0.0
    for r in rows:
        lw, cloud = int(r[0]), int(r[1])
        at, vp, tskc, ref = (float(x) for x in r[2:6])
        got = float(df.calc_longwave(
            torch.tensor(tskc, dtype=DT, device=DEV),
            torch.tensor(at, dtype=DT, device=DEV),
            torch.tensor(vp, dtype=DT, device=DEV), lw_type=df.LW_TYPES[lw],
            lw_cloud_deardorff=bool(cloud)))
        worst = max(worst, abs((got - ref) / ref - want))
    assert worst < 1e-11, worst


def test_unknown_longwave_type_is_refused():
    with pytest.raises(ValueError):
        df.calc_longwave(torch.zeros(1, dtype=DT, device=DEV), torch.zeros(1, dtype=DT, device=DEV),
                         torch.zeros(1, dtype=DT, device=DEV), lw_type='BRAS')


# ---------------------------------------------------------------------------
# The time-zone split
# ---------------------------------------------------------------------------
def test_local_time_offset_rounds_away_from_zero():
    """The whole-hour part rounds half away from zero. Python's round()
    would send a half to even and put a cell exactly 7.5 hours from its
    meridian on the wrong day."""
    assert df.local_time_offset(-121.5, -120.0)[1] == 0
    assert df.local_time_offset(-121.5, 0.0)[1] == 8
    assert df.local_time_offset(121.5, 0.0)[1] == -8
    # 7.5 hours exactly: away from zero, both signs.
    assert df.local_time_offset(-112.5, 0.0)[1] == 8
    assert df.local_time_offset(112.5, 0.0)[1] == -8
    # 121.5 W sits 1.5 degrees west of its meridian, so local solar time
    # runs 6 minutes behind the clock: (theta_l - theta_s) * 24 / 360.
    frac, whole = df.local_time_offset(-121.5, -120.0)
    assert abs(frac - 0.1) < 1e-15 and whole == 0


def test_a_utc_record_puts_noon_eight_hours_later():
    """The whole-hour offset shifts the whole record. A cell at 121.5 W read
    in UTC has its solar noon at UTC hour 20, and the same daily numbers read
    at the local meridian put it at hour 12."""
    days = _days()
    local = _run(days, time_zone_lng=-120.0)
    utc = _run(days, time_zone_lng=0.0)
    assert int(local['shortwave'][0, :24].argmax()) == 12
    assert int(utc['shortwave'][0, :24].argmax()) == 20


def test_cells_in_different_time_zones_are_refused():
    """The offset decides how many local days the record has, so one call
    covers one offset. Silently using the first cell's would shift the
    others' radiation by hours."""
    days = _days()
    days = {k: v.expand(2, -1).contiguous() for k, v in days.items()}
    with pytest.raises(ValueError, match='whole-hour'):
        df.daily_to_subdaily(days, START, [47.8, 47.8], [-121.5, -60.0],
                             [1300.0, 1300.0], time_zone_lng=[-120.0, -120.0])


# ---------------------------------------------------------------------------
# What the assembly must conserve or hold constant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('spd', [1, 2, 4, 8, 24])
def test_daily_precipitation_is_conserved_at_every_step_length(spd):
    """The daily total is divided evenly over the day's model steps (MetSim's
    UNIFORM precipitation). Every step length must put the same water on the
    ground."""
    days = _days()
    out = _run(days, steps_per_day=spd)
    per_day = out['prcp'].reshape(1, NDAYS, spd)
    assert float((per_day.sum(-1) - days['prcp']).abs().max()) < 1e-12
    # Evenly, not weighted by anything.
    assert float((per_day - per_day[..., :1]).abs().max()) < 1e-15


def test_wind_is_held_constant_through_the_day():
    """metsim.disaggregate.wind: constant over the day."""
    days = _days()
    out = _run(days, steps_per_day=24)
    w = out['wind'].reshape(1, NDAYS, 24)
    assert float((w - w[..., :1]).abs().max()) == 0.0
    assert float((w[..., 0] - days['wind']).abs().max()) < 1e-15


def test_missing_wind_falls_back_to_vics_default():
    """DEFAULT_WIND_SPEED, 1.5 m/s."""
    days = _days()
    days.pop('wind')
    out = _run(days)
    assert float((out['wind'] - df.DEFAULT_WIND_SPEED).abs().max()) == 0.0


def test_minimum_wind_speed_applies_at_every_step_length():
    """The floor is applied whatever the model step. (VIC 4.2.d floored only
    the daily mean of a whole-day step; MetSim has no floor at all, so this
    is a choice made here and a documented difference.)"""
    days = _days()
    days['wind'] = torch.full_like(days['wind'], 0.02)
    for spd in (24, 4, 1):
        assert float(_run(days, steps_per_day=spd)['wind'].min()) == \
            df.MIN_WIND_SPEED


def test_pressure_follows_metsims_column_formula():
    """metsim.disaggregate.pressure: standard pressure through the
    temperature of the column, with MetSim's constants (P_STD 101325 Pa,
    G 9.80665, R 287, lapse 0.0065 K/m)."""
    out = _run(steps_per_day=24)
    elev = 1300.0
    want = mtclim.P_STD * torch.exp(
        -(elev * mtclim.G_STD)
        / (mtclim.R_DRY * (mtclim.KELVIN + out['tair']
                           - 0.5 * elev * mtclim.LAPSE_RATE)))
    assert float((out['press'] - want).abs().max()) < 1e-9
    assert not torch.allclose(out['press'], out['press'][0, 0])   # it varies


def test_pressure_from_elevation_alone_when_plapse_is_off():
    out = _run(plapse=False)
    want = float(mtclim.atm_pres(torch.tensor(1300.0, dtype=DT)))
    assert float((out['press'] - want).abs().max()) < 1e-9


def test_vic42d_terrain_and_snow_options_are_refused():
    """MetSim's MTCLIM is flat-surface only; asking for a slope, a horizon,
    the snow correction or an iteration mode must fail loudly rather than be
    ignored."""
    for kw in (dict(slope=10.0), dict(aspect=90.0), dict(ehoriz=5.0),
               dict(whoriz=5.0), dict(swe_corr=True),
               dict(vp_iter='CONVERGE')):
        with pytest.raises(ValueError, match='flat surface'):
            _run(**kw)


def test_cloud_fraction_is_held_constant_over_the_day():
    """MTCLIM gives one cloud fraction per day (metsim.disaggregate.tskc)."""
    out = _run(steps_per_day=24)
    c = out['tskc'].reshape(1, NDAYS, 24)
    assert float((c - c[..., :1]).abs().max()) == 0.0


def test_shortwave_keeps_mtclims_daily_total():
    """Aggregating the hourly shortwave to a coarser step must not move the
    day's energy."""
    days = _days()
    fine = _run(days, steps_per_day=24)['shortwave'].reshape(1, NDAYS, 24)
    for spd in (1, 2, 4, 8):
        coarse = _run(days, steps_per_day=spd)['shortwave'].reshape(
            1, NDAYS, spd)
        assert float((coarse.mean(-1) - fine.mean(-1)).abs().max()) < 1e-10


# ---------------------------------------------------------------------------
# Vapour pressure
# ---------------------------------------------------------------------------
def test_vp_interpolation_passes_through_the_daily_value_at_the_cold_time():
    """metsim.disaggregate.vapor_pressure: the daily value sits at the time
    of the daily minimum temperature, where the air is nearest saturation,
    and a straight line runs between consecutive minima. At each such time
    the line is the knot.

    Checked on the interpolation itself rather than on the assembled field,
    because the saturation clamp comes afterwards and can pull a knot down;
    that clamp has its own test.
    """
    out = _run(steps_per_day=24, vp_interp=True)
    mt = out['mtclim']
    daily = mt['s_vp'].reshape(1, NDAYS)
    # With the minimum at exactly 06:00 every day the knots fall on hours and
    # the interpolant equals the daily value there, to rounding.
    six = torch.full((1, NDAYS), 360.0, dtype=DT, device=DEV)
    vp_h = df._vp_to_hourly(daily, six, True).reshape(NDAYS, 24)
    assert float((vp_h[:, 6] - daily[0]).abs().max()) < 1e-9
    # Between two knots it is a straight line: no overshoot of its ends.
    assert float(vp_h[1:-1].max()) <= float(daily.max()) + 1e-9
    assert float(vp_h[1:-1].min()) >= float(daily.min()) - 1e-9
    # and the real knots (sunrise-dependent) give a continuous series whose
    # second difference vanishes away from them
    real = df._vp_to_hourly(daily, mt['t_tmin'].reshape(1, NDAYS),
                            True).reshape(-1)
    knots = (mt['t_tmin'].reshape(-1)
             + 1440.0 * torch.arange(NDAYS, dtype=DT, device=DEV)) / 60.0
    d2 = (real[:-2] - 2 * real[1:-1] + real[2:]).abs()
    near = torch.zeros_like(d2, dtype=torch.bool)
    for k in knots.tolist():
        h = int(math.floor(k))
        near[max(h - 1, 0):h + 2] = True
    assert float(d2[~near].max()) < 1e-9


def test_vp_is_flat_when_interpolation_is_off():
    """Held constant through the day. Flat before the saturation clamp, and
    the assembled field is flat wherever the clamp does not bite."""
    out = _run(steps_per_day=24, vp_interp=False)
    mt = out['mtclim']
    daily = mt['s_vp'].reshape(1, NDAYS)
    v = df._vp_to_hourly(daily, mt['t_tmin'].reshape(1, NDAYS), False)
    v = v.reshape(1, NDAYS, 24)
    assert float((v - v[..., :1]).abs().max()) == 0.0
    assembled = out['vp'].reshape(1, NDAYS, 24)
    unclamped = assembled >= v - 1e-12
    assert float((assembled - v)[unclamped].abs().max()) < 1e-12


def test_vp_never_exceeds_saturation_and_the_longwave_sees_the_clamp():
    """A supersaturated vapour pressure is pulled back to saturation and
    *written back*, so the longwave is computed from the clamped value. Feeding a vapour pressure above saturation and getting
    the unclamped longwave back would be the bug this pins."""
    days = _days()
    out = _run(days, steps_per_day=24, vp_interp=False)
    sat = mtclim.svp(out['tair'])
    assert float((out['vp'] - sat).max()) < 1e-9
    assert float(out['vpd'].min()) >= 0.0
    want = df.calc_longwave(out['tskc'], out['tair'], out['vp'])
    assert float((out['longwave'] - want).abs().max()) == 0.0


def test_supplied_daily_vapour_pressure_is_used_as_given():
    """An observed vapour pressure replaces the humidity estimate, and
    MTCLIM then estimates only the radiation from it."""
    days = _days()
    obs = torch.full_like(days['prcp'], 400.0)
    free = _run(days, vp_interp=False)
    given = _run(dict(days, vp=obs), vp_interp=False)
    assert float((given['mtclim']['s_vp'] - obs).abs().max()) < 1e-9
    assert float((free['mtclim']['s_vp'] - obs).abs().max()) > 1.0


# ---------------------------------------------------------------------------
# What this is for
# ---------------------------------------------------------------------------
def test_the_forcing_set_is_the_one_vic5_requires():
    out = _run()
    for name in ('tair', 'prcp', 'press', 'shortwave', 'longwave', 'vp',
                 'wind'):
        assert name in out, name
        assert torch.isfinite(out[name]).all(), name


def test_every_field_is_physically_ranged():
    """A silent unit error anywhere in the chain shows up here: pressure in
    kPa, vapour pressure in kPa, or a radiation sign flip."""
    out = _run(steps_per_day=24)
    assert 200.0 < float(out['shortwave'].max()) < 1400.0
    assert float(out['shortwave'].min()) >= 0.0
    assert 100.0 < float(out['longwave'].min()) and \
        float(out['longwave'].max()) < 500.0
    assert 50000.0 < float(out['press'].min()) < 110000.0
    assert 0.0 < float(out['vp'].min()) and float(out['vp'].max()) < 8000.0
    assert -60.0 < float(out['tair'].min()) and float(out['tair'].max()) < 60.0


def test_the_whole_chain_carries_a_gradient():
    """The point of the rewrite. A daily forcing record is now a leaf that
    the loss can reach: calibrating against streamflow can move the inputs
    the disaggregation builds, not only the soil parameters."""
    days = _days()
    days['tmax'] = days['tmax'].clone().requires_grad_(True)
    out = _run(days, steps_per_day=24)
    (out['shortwave'].sum() + out['longwave'].sum()
     + out['vp'].sum() + out['tair'].sum()).backward()
    g = days['tmax'].grad
    assert torch.isfinite(g).all()
    assert float(g.abs().min()) > 0.0        # every day matters


# ---------------------------------------------------------------------------
# The reader-level wiring
# ---------------------------------------------------------------------------
def _reader_shaped(n=45, ncell=3, seed=1):
    """What ClassicForcing.read / ImageForcing.read hand back: [time, cell]
    tensors keyed by canonical name, plus a stamps array."""
    import numpy as np
    g = torch.Generator(device=DEV).manual_seed(seed)
    i = torch.arange(n, dtype=DT, device=DEV)
    base = (8.0 - 14.0 * torch.cos(2.0 * torch.pi * i / 365.0)
            ).unsqueeze(1).expand(n, ncell)
    fields = dict(
        tmax=base + 6.0, tmin=base - 6.0,
        prcp=10.0 * torch.rand(n, ncell, generator=g, dtype=DT, device=DEV),
        wind=1.0 + 3.0 * torch.rand(n, ncell, generator=g, dtype=DT, device=DEV),
        albedo=torch.full((n, ncell), 0.2, dtype=DT, device=DEV))
    stamps = (np.datetime64('1985-06-01T00:00:00', 'ns')
              + np.arange(n) * np.timedelta64(86400, 's')
              ).astype('datetime64[ns]')
    return fields, stamps, n, ncell


def test_expand_forcing_returns_what_a_subdaily_file_would_have():
    """The caller must not be able to tell the forcing was daily: same
    layout, same names, a stamps array on the model's own step."""
    fields, stamps, n, ncell = _reader_shaped()
    sub, st = df.expand_forcing(fields, stamps, 24, [45.0, 46.0, 47.0],
                                [-120.0, -120.2, -120.4],
                                [500.0, 900.0, 1400.0],
                                time_zone_lng=-120.0, log=lambda m: None)
    for name in ('tair', 'prcp', 'press', 'shortwave', 'longwave', 'vp',
                 'wind'):
        assert sub[name].shape == (n * 24, ncell), name
        assert torch.isfinite(sub[name]).all(), name
    assert len(st) == n * 24
    assert (st[1] - st[0]) == np.timedelta64(3600, 's')
    assert st[0] == stamps[0]
    # Per cell, and different cells really do differ (elevation and latitude
    # both enter): a broadcast bug would make them identical.
    assert float((sub['press'][:, 0] - sub['press'][:, 2]).abs().max()) > 1000.0
    assert float((sub['shortwave'][:, 0]
                  - sub['shortwave'][:, 2]).abs().max()) > 1.0


def test_expand_forcing_conserves_precipitation_and_carries_other_fields():
    fields, stamps, n, ncell = _reader_shaped()
    sub, _st = df.expand_forcing(fields, stamps, 24, 45.0, -120.0, 500.0,
                                 log=lambda m: None)
    assert float((sub['prcp'].reshape(n, 24, ncell).sum(1)
                  - fields['prcp']).abs().max()) < 1e-12
    # Any other daily field is held over its day.
    assert sub['albedo'].shape == (n * 24, ncell)
    assert bool((sub['albedo'] == 0.2).all())
    # TMAX and TMIN are consumed, not passed on: nothing downstream reads them.
    assert 'tmax' not in sub and 'tmin' not in sub


def test_expand_forcing_says_what_is_estimated_rather_than_read():
    """Shortwave, longwave and vapour pressure in the returned set were
    never in the file. A run that silently presented them as observations
    would be the worst kind of quiet wrong answer."""
    fields, stamps, _n, _c = _reader_shaped()
    said = []
    df.expand_forcing(fields, stamps, 24, 45.0, -120.0, 500.0,
                      log=said.append)
    # The property, not a wording: some line must say which fields MTCLIM
    # estimated, and name each one this file did not supply. The message
    # used to be one fixed sentence claiming shortwave, longwave and vapour
    # pressure were ALL estimated -- true here, false for a file that
    # supplies them, which is the case the next test holds.
    est = [l for l in said if 'estimated' in l]
    assert est, said
    for name in ('shortwave', 'longwave'):
        if name not in fields:
            assert name in est[0], (name, est[0])


def test_measured_longwave_and_pressure_are_kept_not_overwritten():
    """An observation beats a parameterisation.

    expand_forcing copied MTCLIM's output over every name it produces, so a
    file supplying LWDOWN and PRESSURE had both silently replaced by
    estimates. Found on the Indian data, where the log claimed the file's
    radiation had been read and the run was using MTCLIM's instead. Now the
    measured daily value is held across the day, like any other daily
    field, and the log does not
    list it as estimated.
    """
    fields, stamps, n, c = _reader_shaped()
    lw = torch.full_like(fields['tmax'], 321.0)        # distinctive values
    ps = torch.full_like(fields['tmax'], 91234.0)
    fields['longwave'] = lw
    fields['press'] = ps
    said = []
    sub, _st = df.expand_forcing(fields, stamps, 24, 45.0, -120.0, 500.0,
                                 log=said.append)
    assert torch.equal(sub['longwave'], lw.repeat_interleave(24, dim=0))
    assert torch.equal(sub['press'], ps.repeat_interleave(24, dim=0))
    est = ' '.join(l for l in said if 'estimated' in l)
    assert 'longwave' not in est and 'press' not in est, said
    assert any('held at the measured' in l for l in said), said


def test_expand_forcing_refuses_a_file_without_the_daily_extremes():
    """MTCLIM cannot start without them."""
    fields, stamps, _n, _c = _reader_shaped()
    fields.pop('tmin')
    with pytest.raises(ValueError, match='TMIN'):
        df.expand_forcing(fields, stamps, 24, 45.0, -120.0, 500.0,
                          log=lambda m: None)


# ---------------------------------------------------------------------------
# The four fields a CARBON run needs and no public dataset supplies
# ---------------------------------------------------------------------------
def test_carbon_forcings_are_derived_not_demanded():
    """VIC 5 reads CATM, FDIR and PAR straight out of the forcing file
    (vic_force.c:165-178) and stops without them, which is why a CARBON run
    needs a dataset almost nobody has. MetSim has none of them, so they are
    derived here (fdir from the Erbs diffuse fraction)."""
    out = _run(steps_per_day=24, carbon=True)
    for name in ('fdir', 'par', 'Catm', 'coszen'):
        assert name in out, name
        assert torch.isfinite(out[name]).all(), name
    # PAR is a fixed fraction of the shortwave.
    assert float((out['par'] - df.SW2PAR * out['shortwave']).abs().max()) == 0.0
    # One global concentration, ppm converted to a mixing ratio.
    assert float((out['Catm'] - df.CATM_CURRENT_PPM * 1e-6).abs().max()) == 0.0
    # The direct fraction is held over the day.
    f = out['fdir'].reshape(1, NDAYS, 24)
    assert float((f - f[..., :1]).abs().max()) == 0.0
    assert 0.0 <= float(out['fdir'].min()) and float(out['fdir'].max()) <= 1.0


def test_coszen_peaks_at_local_noon_and_follows_the_season():
    """compute_coszen.c. Checked against where the sun actually is rather
    than against a stored number: at 47.8 N the solar zenith at noon is the
    latitude minus the declination, and it must be highest near midsummer."""
    out = _run(steps_per_day=24, carbon=True, time_zone_lng=-120.0)
    cz = out['coszen'].reshape(NDAYS, 24)
    # March 1 onwards; noon is hour 12 at the meridian, give or take the
    # 6-minute longitude offset.
    assert int(cz[0].argmax()) == 12
    noon = cz[:, 12]
    assert float(noon.min()) > 0.0
    assert bool((noon[1:] >= noon[:-1] - 1e-3).all())   # rising into spring
    # Day 0 is March 1, day-of-year 60: the declination is -7.9 degrees, so
    # the noon zenith angle is the latitude minus the declination.
    decl = -0.4092797 * math.cos((60.0 + 11.25) * 0.017214)
    assert abs(float(noon[0]) - math.cos(math.radians(47.8) - decl)) < 2e-3
    assert float(cz.min()) < 0.0        # VIC does not clamp the night side


def test_carbon_forcings_reach_the_reader_level_wiring():
    fields, stamps, n, ncell = _reader_shaped()
    sub, _st = df.expand_forcing(fields, stamps, 24, 45.0, -120.0, 500.0,
                                 carbon=True, log=lambda m: None)
    for name in ('fdir', 'par', 'Catm', 'coszen'):
        assert sub[name].shape == (n * 24, ncell), name
    # And they are absent when nobody asked for them.
    plain, _ = df.expand_forcing(fields, stamps, 24, 45.0, -120.0, 500.0,
                                 log=lambda m: None)
    assert 'Catm' not in plain
