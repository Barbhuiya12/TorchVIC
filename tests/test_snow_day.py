"""A daily model step sub-steps a snow day at SNOW_STEPS_PER_DAY.

surface_fluxes.c:258-269: a pack, canopy snow, or snowflag[NR] runs hidx
0..NF-1 at snow_dt; otherwise one step at the daily mean (index NR).
runoff() is once, on the summed throughfall (surface_fluxes.c:954, :1080).
A day that does not snow is the daily-mean step, bit for bit.
"""
import datetime as dtm

import numpy as np
import pytest
import torch

from torchvic.core import column, vegetation
from torchvic.model import basin
from torchvic import case

torch.set_default_dtype(torch.float64)
DEVICE = torch.device('cpu')
KW = dict(dtype=torch.float64, device=DEVICE)


def _times(n, hours=3):
    t0 = np.datetime64('2004-01-01T00:00:00')
    return t0 + np.arange(n) * np.timedelta64(hours, 'h')


def test_subdaily_pack_is_the_days_records_in_order():
    """hidx 0 is the day's first record, and the daily precip is their sum
    (vic_force.c:446-447, average * NF)."""
    nrec, per, nc = 16, 8, 2
    times = _times(nrec)
    pr = torch.arange(nrec * nc, dtype=torch.float64).reshape(nrec, nc) + 1.0
    ta = torch.full((nrec, nc), 3.0)
    fields = dict(prcp=pr, tair=ta,
                  shortwave=torch.full((nrec, nc), 100.0),
                  longwave=torch.full((nrec, nc), 200.0),
                  vp=torch.full((nrec, nc), 500.0))
    off = np.zeros(nc)
    daily, dates, sub = case.daily_by_off_gmt(
        fields, times, off, with_subdaily=True)
    assert len(dates) == 2
    assert tuple(sub['prcp'].shape) == (2, per, nc)
    assert torch.equal(sub['tair'][0, 0], ta[0])
    assert torch.equal(sub['prcp'][0].sum(0), daily['prcp'][0])
    assert torch.equal(sub['prcp'][1, -1], pr[15])


def _mosaic(n):
    return vegetation.VegMosaic(
        cv=torch.tensor([[0.6, 0.4]] * n, **KW),
        lai=torch.full((n, 2, 12), 2.0, **KW),
        albedo=torch.full((n, 2, 12), 0.15, **KW),
        veg_rough=torch.full((n, 2, 12), 0.1, **KW),
        displacement=torch.full((n, 2, 12), 0.6, **KW),
        rmin=torch.full((n, 2), 100.0, **KW),
        rgl=torch.full((n, 2), 100.0, **KW),
        rarc=torch.full((n, 2), 50.0, **KW),
        root_frac=torch.tensor([0.3, 0.3, 0.4], **KW).view(1, 1, 3)
        .expand(n, 2, 3).clone(),
        dtype=torch.float64, device=DEVICE,
        wind_h=torch.full((n, 2), 10.0, **KW),
        rad_atten=torch.full((n, 2), 0.5, **KW),
        wind_atten=torch.full((n, 2), 0.5, **KW),
        trunk_ratio=torch.full((n, 2), 0.2, **KW),
        overstory=torch.zeros(n, 2, **KW))


def _soil(n, steps_per_day):
    one = torch.ones(n, **KW)
    sp = column.SoilParams(
        b_infilt=0.2 * one, Ds=0.02 * one, Dsmax=10 * one, Ws=0.8 * one,
        c=2 * one, expt=torch.stack([4 * one] * 3, -1),
        Ksat=torch.stack([250 * one] * 3, -1),
        max_moist=torch.stack([100 * one, 300 * one, 600 * one], -1),
        resid_moist=torch.stack([10 * one] * 3, -1),
        runoff_steps_per_day=24, model_steps_per_day=steps_per_day)
    sp.thermal = dict(
        avg_temp=4 * one, dp=4 * one,
        depth=torch.tensor([[0.1, 0.3, 1.0]], **KW).expand(n, 3).clone(),
        bulk_density=torch.full((n, 3), 1500.0, **KW),
        soil_density=torch.full((n, 3), 2650.0, **KW),
        quartz=torch.full((n, 3), 0.4, **KW), rough=0.01 * one,
        snow_rough=0.001 * one)
    return sp


def _params():
    return {k: torch.tensor(v, **KW) for k, v in dict(
        pcf=1., t_max_snow=.5, t_min_rain=-.5, rs=120., et_beta=1.,
        new_snow_albedo=.85, alb_accum_a=.94, alb_accum_b=.58,
        alb_thaw_a=.82, alb_thaw_b=.46, uh_shape=2.5, uh_scale=2.,
        lai_scale=1., rmin_scale=1., w_crit=.7, w_wilt=.3).items()}


class _Batch(object):
    def __init__(self, n, n_days):
        self.dates = [dtm.datetime(2004, 1, 1) + dtm.timedelta(days=i)
                      for i in range(n_days)]
        self.elevation = torch.full((n,), 1000.0, **KW)
        self.cell_area = torch.full((n,), 1.0e6, **KW)
        self.source2outlet = torch.zeros(n, dtype=torch.long)
        self.n_outlets = 1


def _atm_day(tair, prcp):
    """Daily means, and 8 sub-steps that average back to them."""
    tair = torch.as_tensor(tair, dtype=torch.float64)
    prcp = torch.as_tensor(prcp, dtype=torch.float64)
    if tair.dim() == 1:
        tair = tair.unsqueeze(0)
        prcp = prcp.unsqueeze(0)
    nd, nc = tair.shape
    nf = 8

    def series(daily, vary=None):
        # vary is [nf] added to every cell, with mean 0, so the day's
        # mean is unchanged. Precipitation is split into equal pieces.
        out = daily.unsqueeze(1).expand(nd, nf, nc).clone()
        return out

    sub_t = series(tair)
    sub_p = series(prcp) / nf
    atm = _const_atm(tair, prcp)
    sub = _const_atm(sub_t.reshape(nd * nf, nc),
                     sub_p.reshape(nd * nf, nc))
    sub = {k: v.reshape(nd, nf, nc) for k, v in sub.items()}
    return atm, sub


def _const_atm(tair, prcp):
    nt, nc = tair.shape
    ones = torch.ones(nt, nc, **KW)
    return dict(tair=tair, tmax=tair + 2.0, tmin=tair - 2.0, prcp=prcp,
                press=95000.0 * ones, vp=300.0 * ones, eact_air=300.0 * ones,
                vpd=200.0 * ones, shortwave=150.0 * ones,
                longwave=260.0 * ones, wind=2.0 * ones,
                air_dens=1.25 * ones, lv=2.5e6 * ones)


def _run(atm, n, full_energy, sub=None):
    nd = atm['tair'].shape[0]
    return basin.simulate(
        _Batch(n, nd), atm, _params(), _soil(n, 1), mosaic=_mosaic(n),
        steps_per_day=1, full_energy=full_energy, subdaily=sub,
        snow_steps=None if sub is None else 8)


def _equal(a, b):
    bad = []
    for k, v in a.items():
        if not torch.is_tensor(v) or k not in b or not torch.is_tensor(b[k]):
            continue
        if v.shape != b[k].shape:
            bad.append('%s shape %s %s' % (k, tuple(v.shape), tuple(b[k].shape)))
            continue
        if not torch.equal(v, b[k]):
            d = (v - b[k]).abs()
            d = d[torch.isfinite(d)]
            bad.append('%s %.3e' % (k, float(d.max()) if d.numel() else
                                    float('nan')))
    return bad


def test_a_warm_day_with_subdaily_records_matches_the_daily_mean():
    atm, sub = _atm_day([[15.0], [16.0]], [[8.0], [4.0]])
    # Give the sub-steps a diurnal swing of mean zero. A snow day would
    # use it; a warm day must not.
    swing = torch.tensor([-2., -1., 0., 1., 2., 1., 0., -1.])
    sub['tair'] = sub['tair'] + swing.view(1, 8, 1)
    bare = _run(atm, 1, False, sub=None)
    with_sub = _run(atm, 1, False, sub=sub)
    bad = _equal(bare, with_sub)
    assert not bad, bad[:8]


def test_a_cold_substep_snows_when_the_daily_mean_is_rain():
    """Four steps at -8 C with rain, four at 12 C. The daily mean is 2 C,
    all rain (t_max_snow 0.5). will_it_snow is true (forcing_utils.c:94-96),
    so the day is sub-stepped and the cold steps fall as snow."""
    daily_t = torch.tensor([[2.0]])
    daily_p = torch.tensor([[4.0]])
    atm = _const_atm(daily_t, daily_p)
    sub_t = torch.tensor([-8., -8., -8., -8., 12., 12., 12., 12.])
    sub_p = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.])
    sub = _const_atm(sub_t.view(8, 1), sub_p.view(8, 1))
    sub = {k: v.reshape(1, 8, 1) for k, v in sub.items()}
    mean_only = _run(atm, 1, False, sub=None)
    substepped = _run(atm, 1, False, sub=sub)
    assert float(mean_only['swe'][0, 0]) == 0.0
    assert float(substepped['swe'][0, 0]) > 0.0
    assert not torch.equal(mean_only['swe'], substepped['swe'])
    # The energy path uses the same gate and the same once-a-day column.
    eb = _run(atm, 1, True, sub=sub)
    assert float(eb['swe'][0, 0]) > 0.0
    assert eb['swe'].shape[0] == 1 and eb['eb_latent'].shape[0] == 1
    assert eb['eb_aero_cond'].shape[0] == 1


def test_a_warm_cell_beside_a_snow_cell_stays_on_the_daily_path():
    """The gate is per cell (surface_fluxes.c:258). The warm cell's day is
    the daily mean even though its neighbour sub-steps."""
    # cell 0: the cold pattern above. cell 1: 20 C and rain all day.
    daily_t = torch.tensor([[2.0, 20.0], [15.0, 15.0]])
    daily_p = torch.tensor([[4.0, 8.0], [0.0, 0.0]])
    atm = _const_atm(daily_t, daily_p)
    nf = 8
    sub_t = daily_t.unsqueeze(1).expand(2, nf, 2).clone()
    sub_p = daily_p.unsqueeze(1).expand(2, nf, 2).clone() / nf
    sub_t[0, :, 0] = torch.tensor([-8., -8., -8., -8., 12., 12., 12., 12.])
    sub_p[0, :, 0] = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.])
    sub = _const_atm(sub_t.reshape(16, 2), sub_p.reshape(16, 2))
    sub = {k: v.reshape(2, nf, 2) for k, v in sub.items()}
    mean_only = _run(atm, 2, False, sub=None)
    mixed = _run(atm, 2, False, sub=sub)
    assert float(mixed['swe'][0, 0]) > 0.0
    n_days, n_cells = 2, 2
    for k in ('swe', 'runoff', 'baseflow', 'et', 'transp', 'evap_canop',
              'sub_snow', 'soil_moisture'):
        v, r = mixed[k], mean_only[k]
        if v.shape[0] == n_days and v.dim() > 1 and v.shape[1] == n_cells:
            assert torch.equal(v[:, 1], r[:, 1]), k
        elif v.shape[0] == n_cells:
            assert torch.equal(v[1], r[1]), k
        else:
            raise AssertionError('%s %s' % (k, tuple(v.shape)))
