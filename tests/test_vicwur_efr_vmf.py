# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""tools/vicwur_inputs/efr_vmf.py against Droppers et al. 2020 (GMD 13:5029)
Appendix B, Eqs. B1-B4 (VMF, Pastor et al. 2014).  The upstream VIC-WUR tree
has no support tool for the EFR forcing (tools/ holds only code_format and
ide), so the paper formula is the reference; it is re-implemented here with
plain Python loops and if-chains, independent of the numpy code.
"""
import calendar
import datetime
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'efr_vmf', os.path.join(HERE, '..', 'tools', 'vicwur_inputs',
                            'efr_vmf.py'))
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)


def series(years, f):
    """Daily series: f(month, year) per day, plus the month numbers."""
    q, mo = [], []
    d = datetime.date(years[0], 1, 1)
    while d.year <= years[1]:
        q.append(f(d.month, d.year))
        mo.append(d.month)
        d += datetime.timedelta(days=1)
    return np.array(q, dtype=float), np.array(mo)


def paper_b1_b3(nf_d, nf_y):
    """Eqs. B1-B3 as written in the paper."""
    if nf_d <= 0.4 * nf_y:
        return 0.6 * nf_d
    elif 0.4 * nf_y < nf_d <= 0.8 * nf_y:
        return 0.45 * nf_d
    else:
        return 0.3 * nf_d


def test_eqs_b1_b4_class_boundaries():
    ny = 10.0
    cases = [(0.0, 0.0), (3.0, 1.8), (4.0, 2.4), (4.0000001, 0.45 * 4.0000001),
             (6.0, 2.7), (8.0, 3.6), (8.0000001, 0.3 * 8.0000001),
             (12.0, 3.6), (50.0, 15.0)]
    for nf, want in cases:
        assert V.efr_streamflow(nf, ny) == pytest.approx(want, rel=1e-14)
        assert V.efr_streamflow(nf, ny) == pytest.approx(
            paper_b1_b3(nf, ny), rel=1e-14)
    assert V.efr_baseflow(2.0) == pytest.approx(1.8)            # B4


def test_monthly_vmf_from_a_discharge_series():
    """Monthly VMF (Pastor 2014) from a 31-year daily discharge series:
    month means, MAF = mean of the series, class factors per month."""
    mm = [2.0, 3.0, 5.0, 9.0, 14.0, 22.0, 30.0, 25.0, 12.0, 6.0, 4.0, 2.5]
    q, mo = series((1980, 2010), lambda m, y: mm[m - 1] * (1 + 0.1 * np.sin(y)))
    # an independent loop
    tot = {m: [] for m in range(1, 13)}
    for v, m in zip(q, mo):
        tot[m].append(v)
    mmf = [sum(tot[m]) / len(tot[m]) for m in range(1, 13)]
    maf = sum(q) / len(q)
    want = [paper_b1_b3(x, maf) for x in mmf]
    got = V.efr_from_series(q, mo, daily=False)
    np.testing.assert_allclose(got, want, rtol=1e-13)
    # all three classes occur (the check is not vacuous)
    cl = {0.6 if x <= 0.4 * maf else 0.45 if x <= 0.8 * maf else 0.3
          for x in mmf}
    assert cl == {0.6, 0.45, 0.3}
    # per cell: the same on a [T, cells] array
    qq = np.stack([q, 2 * q], 1)
    g2 = V.efr_from_series(qq, mo, daily=False)
    np.testing.assert_allclose(g2[:, 0], want, rtol=1e-13)
    np.testing.assert_allclose(g2[:, 1], [paper_b1_b3(2 * x, 2 * maf)
                                          for x in mmf], rtol=1e-13)


def test_daily_vmf_interpolates_the_monthly_mean():
    mm = [2.0, 3.0, 5.0, 9.0, 14.0, 22.0, 30.0, 25.0, 12.0, 6.0, 4.0, 2.5]
    q, mo = series((1980, 2010), lambda m, y: mm[m - 1])
    daily = V.efr_from_series(q, mo, daily=True)
    assert daily.shape == (365,)
    maf = q.mean()
    # independent interpolation by hand: positions of mid-month, wrap
    nd = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mid, s = [], 0
    for n in nd:
        mid.append(s + n / 2.0)
        s += n
    for day in (0, 7, 15, 100, 200, 300, 364):
        x = day + 0.5
        pts = [(mid[-1] - 365, mm[-1])] + list(zip(mid, mm)) + \
              [(mid[0] + 365, mm[0])]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if x0 <= x <= x1:
                nf = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                break
        assert daily[day] == pytest.approx(paper_b1_b3(nf, maf), rel=1e-12)
    # the interpolated flow in the middle of a 31-day month is the monthly mean
    assert daily[15] == pytest.approx(paper_b1_b3(mm[0], maf), rel=1e-12)
    # a constant series sits at NFs,y > 0.8 NFs,y on every day: class B3
    c, mc = series((1980, 1985), lambda m, y: 7.0)
    d = V.efr_from_series(c, mc, daily=True)
    np.testing.assert_allclose(d, 0.3 * 7.0, rtol=1e-14)
    assert V.efr_from_series(c, mc, daily=True, leap=True).shape == (366,)


def test_baseflow_efr_is_90_percent():
    mm = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    b, mo = series((1990, 1999), lambda m, y: mm[m - 1])
    np.testing.assert_allclose(V.baseflow_from_series(b, mo, daily=False),
                               0.9 * np.array(mm), rtol=1e-14)


def test_cli_writes_monthly_forcing_files(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    from torchvic.io import ncshare
    mm = [2.0, 3.0, 5.0, 9.0, 14.0, 22.0, 30.0, 25.0, 12.0, 6.0, 4.0, 2.5]
    q, mo = series((1980, 1984), lambda m, y: mm[m - 1])
    days = (datetime.date(1984, 12, 31) - datetime.date(1980, 1, 1)).days + 1
    assert len(q) == days
    for nm, scale in (('q', 1.0), ('b', 0.5)):
        with netCDF4.Dataset(str(tmp_path / (nm + '.nc')), 'w') as ds:
            ds.createDimension('time', days)
            ds.createDimension('lat', 1)
            ds.createDimension('lon', 2)
            for n in ('lat', 'lon'):
                v = ds.createVariable(n, 'f8', (n,))
                v[:] = np.arange(len(ds.dimensions[n]), dtype=float)
            t = ds.createVariable('time', 'i4', ('time',))
            t.units = 'days since 1980-01-01'
            t.calendar = 'proleptic_gregorian'
            t[:] = np.arange(days)
            v = ds.createVariable('x', 'f8', ('time', 'lat', 'lon'))
            v[:] = (scale * q)[:, None, None] * np.array([1.0, 2.0])
    out = str(tmp_path / 'o')
    V.main(['--discharge', str(tmp_path / 'q.nc'), '--qvar', 'x',
            '--baseflow', str(tmp_path / 'b.nc'), '--bvar', 'x',
            '--start', '1980', '--end', '1984', '--out', out,
            '--years', '2003', '2004', '--monthly-vmf'])
    ds = ncshare.read(os.path.join(out, 'efr_q.2003.nc'), decode_times=False)
    maf = q.mean()
    want = np.array([paper_b1_b3(x, maf) for x in mm])
    np.testing.assert_allclose(ds['efr_q'].values[:, 0, 0], want, rtol=1e-6)
    np.testing.assert_allclose(ds['efr_q'].values[:, 0, 1],
                               [paper_b1_b3(2 * x, 2 * maf) for x in mm],
                               rtol=1e-6)
    assert list(ds['time'].values) == [0, 31, 59, 90, 120, 151, 181, 212,
                                       243, 273, 304, 334]
    ds = ncshare.read(os.path.join(out, 'efr_b.2004.nc'), decode_times=False)
    np.testing.assert_allclose(ds['efr_b'].values[:, 0, 0],
                               0.9 * 0.5 * np.array(mm), rtol=1e-6)
    assert list(ds['time'].values)[2] == 60          # 2004 is a leap year
    assert calendar.isleap(2004)
