#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Environmental flow requirements by the variable monthly flow (VMF) method
for the VIC-WUR EFR plugin.

Formula source.  There is no VIC-WUR support tool for this: the upstream
tree (<data root>/static/upstream/vicwur, git bb12fb3d, and the 3.0.0 Zenodo
copy) has only tools/code_format and tools/ide, and the model overview
(VIC-WUR's docs/Overview/ModelOverview_vicwur.md:53-54) says the forcings come from the
paper.  The reference is therefore Droppers et al. 2020 (GMD 13:5029),
Appendix B, Eqs. B1-B4 (VMF of Pastor et al. 2014 for streamflow, the 90 %
presumptive standard of Gleeson and Richter 2018 for baseflow):

    EFRs,d = 0.60 NFs,d   where NFs,d <= 0.4 NFs,y                  (B1)
    EFRs,d = 0.45 NFs,d   where 0.4 NFs,y < NFs,d <= 0.8 NFs,y      (B2)
    EFRs,d = 0.30 NFs,d   where NFs,d > 0.8 NFs,y                   (B3)
    EFRb,d = 0.90 NFb,d                                             (B4)

NFs,d is the multi-year monthly mean natural streamflow interpolated to the
day (paper: "interpolated multiyear monthly average flow"; naturalised run,
1980-2010), NFs,y the mean natural yearly streamflow, NFb,d the natural
baseflow treated like NFs,d.  `daily=False` gives the original monthly VMF
(Pastor et al. 2014: NFs,d := the monthly mean).

Interpolation (the paper does not give it): linear in the day of the year
between the monthly means placed at the middle of each month, periodic over
the year.  NFs,y is the mean of the input series.  Units are the input's: the
streamflow EFR is in the unit of the discharge (m3/s), and the baseflow EFR
is in the unit of the baseflow series; the VIC-WUR forcing wants baseflow in
mm per MODEL STEP (efr_force.c units), so feed mm/step.

Usage:
  efr_vmf.py --discharge q.nc --qvar discharge --baseflow b.nc --bvar baseflow
             --start 1980 --end 2010 --out DIR --years 2003 2010
             [--daily-records] [--monthly-vmf]
reads daily series [time, lat, lon] (time axis with calendar), writes
DIR/efr_q.YYYY.nc and DIR/efr_b.YYYY.nc, 12 monthly records per year (or
365/366 daily records with --daily-records), 'days since YYYY-01-01'.
"""
import argparse
import calendar
import os

import numpy as np

LOW, MID = 0.4, 0.8                  # B1-B3 thresholds (x NFs,y)
F_LOW, F_MID, F_HIGH = 0.60, 0.45, 0.30
F_BASE = 0.90                        # B4
_MDAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def monthly_mean(q, months):
    """Multi-year monthly mean [12, ...] of a series [T, ...] with month
    numbers [T] (1..12); NaNs are ignored."""
    q = np.asarray(q, dtype=np.float64)
    months = np.asarray(months)
    out = np.empty((12,) + q.shape[1:])
    for m in range(12):
        out[m] = np.nanmean(q[months == m + 1], axis=0)
    return out


def daily_from_monthly(mmf, leap=False):
    """Interpolate monthly values [12, ...] to the days of one year
    ([365 or 366, ...]): linear between mid-month points, periodic."""
    mmf = np.asarray(mmf, dtype=np.float64)
    nd = _MDAYS.copy()
    if leap:
        nd[1] = 29
    start = np.concatenate([[0], np.cumsum(nd)[:-1]])
    mid = start + nd / 2.0                          # day-count positions
    n = int(nd.sum())
    x = np.arange(n) + 0.5
    xp = np.concatenate([[mid[-1] - n], mid, [mid[0] + n]])
    fp = np.concatenate([mmf[-1:], mmf, mmf[:1]], axis=0)
    out = np.empty((n,) + mmf.shape[1:])
    flat = fp.reshape(len(xp), -1)
    res = np.empty((n, flat.shape[1]))
    for i in range(flat.shape[1]):
        res[:, i] = np.interp(x, xp, flat[:, i])
    return res.reshape(out.shape)


def vmf_factor(nf, nf_year):
    """B1-B3: 0.6 / 0.45 / 0.3 by the class of the flow nf against the
    yearly mean (thresholds 0.4 and 0.8, inclusive upper bounds)."""
    nf = np.asarray(nf, dtype=np.float64)
    return np.where(nf <= LOW * nf_year, F_LOW,
                    np.where(nf <= MID * nf_year, F_MID, F_HIGH))


def efr_streamflow(nf, nf_year):
    """EFRs for streamflow, B1-B3."""
    return vmf_factor(nf, nf_year) * np.asarray(nf, dtype=np.float64)


def efr_baseflow(nfb):
    """EFRb, B4: 90 % of the natural baseflow."""
    return F_BASE * np.asarray(nfb, dtype=np.float64)


def efr_from_series(q, months, daily=True, leap=False):
    """VMF streamflow EFR climatology from a natural flow series.

    q [T, ...] natural flow, months [T].  daily=True gives the paper's daily
    EFR [365|366, ...] (interpolated monthly means); daily=False the monthly
    VMF [12, ...].  NFs,y = mean of the series.
    """
    q = np.asarray(q, dtype=np.float64)
    nf_year = np.nanmean(q, axis=0)
    mmf = monthly_mean(q, months)
    nf = daily_from_monthly(mmf, leap) if daily else mmf
    return efr_streamflow(nf, nf_year)


def baseflow_from_series(b, months, daily=True, leap=False):
    """Baseflow EFR climatology, B4 on the (interpolated) monthly means."""
    mmb = monthly_mean(b, months)
    return efr_baseflow(daily_from_monthly(mmb, leap) if daily else mmb)


def _read(path, var, start, end):
    from torchvic.io import ncshare
    ds = ncshare.read(path)                           # no handle left open
    da = ds[var].sel(time=slice(str(start), str(end)))
    t = da['time'].dt
    return (np.asarray(da.values, dtype=np.float64), np.asarray(t.month),
            da['lat'].values, da['lon'].values)


def _write_year(path, year, vals, lat, lon, vname, units, times):
    import netCDF4
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('lat', len(lat))
        ds.createDimension('lon', len(lon))
        ds.createDimension('time', len(times))
        for n, a in (('lat', lat), ('lon', lon)):
            v = ds.createVariable(n, 'f8', (n,))
            v[:] = a
        v = ds.createVariable('time', 'i4', ('time',))
        v.units = 'days since %d-01-01' % year
        v.calendar = 'proleptic_gregorian'
        v[:] = times
        v = ds.createVariable(vname, 'f4', ('time', 'lat', 'lon'))
        v.units = units
        v[:] = vals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--discharge', required=True)
    ap.add_argument('--qvar', default='discharge')
    ap.add_argument('--baseflow', required=True)
    ap.add_argument('--bvar', default='baseflow')
    ap.add_argument('--start', type=int, default=1980)
    ap.add_argument('--end', type=int, default=2010)
    ap.add_argument('--out', required=True)
    ap.add_argument('--years', type=int, nargs=2, required=True)
    ap.add_argument('--daily-records', action='store_true',
                    help='write 365/366 daily records, default 12 monthly')
    ap.add_argument('--monthly-vmf', action='store_true',
                    help='classify the monthly means (Pastor 2014) instead '
                         'of the interpolated daily flow (paper)')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    q, mq, lat, lon = _read(a.discharge, a.qvar, a.start, a.end)
    b, mb, _, _ = _read(a.baseflow, a.bvar, a.start, a.end)
    shp = q.shape[1:]
    q, b = q.reshape(len(q), -1), b.reshape(len(b), -1)
    for y in range(a.years[0], a.years[1] + 1):
        leap = calendar.isleap(y)
        daily = not a.monthly_vmf
        eq = efr_from_series(q, mq, daily, leap)
        eb = baseflow_from_series(b, mb, daily, leap)
        nd = np.where(leap, _MDAYS + (np.arange(12) == 1), _MDAYS)
        ms = np.concatenate([[0], np.cumsum(nd)[:-1]])
        if a.daily_records:
            if not daily:
                raise SystemExit('--daily-records needs the daily VMF')
            times = np.arange(len(eq))
        else:
            times = ms
            if daily:                      # month mean of the daily EFR
                eq = np.stack([eq[s:s + n].mean(0) for s, n in zip(ms, nd)])
                eb = np.stack([eb[s:s + n].mean(0) for s, n in zip(ms, nd)])
        for nm, v, un in (('efr_q', eq, 'm3/s'), ('efr_b', eb, 'mm')):
            _write_year(os.path.join(a.out, '%s.%d.nc' % (nm, y)), y,
                        v.reshape((len(v),) + shp), lat, lon, nm, un, times)


if __name__ == '__main__':
    main()
