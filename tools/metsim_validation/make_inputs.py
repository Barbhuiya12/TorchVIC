"""Daily inputs for the MetSim comparison (run with the metsim venv python).

Ten sites: the daily series of tests/oracle/ref_mtclim for seeds SEEDS
(120 days, 44-76 N, 300-1500 m) plus Walnut Gulch 2008 (Livneh 3-hourly
forcing reduced to daily Tmax, Tmin, precipitation sum and mean wind).
MetSim needs 90 days of state before the first day, so each record gets 90
leading days (the first 90 days of the series reversed) and both models run
over leading + record; only the record is compared.

usage: make_inputs.py ORACLE_BIN WALNUT_FORCING WALNUT_PARAMS OUT.npz
"""
import datetime
import subprocess
import sys

import numpy as np

SEEDS = [0, 1, 2, 3, 4, 5, 7, 11, 13, 19]
LEAD = 90


def main(oracle, wg_force, wg_params, out):
    rec = {}
    rng = np.random.default_rng(42)
    for s in SEEDS:
        txt = subprocess.run([oracle, 'daily', str(s)], capture_output=True,
                             text=True, check=True).stdout.splitlines()
        rows = [l for l in txt if not l.startswith('#')]
        lat, lng, elev, _ap, yday0, _sd = [float(x) for x in rows[0].split(',')]
        d = np.array([[float(x) for x in r.replace('|', ',').split(',')]
                      for r in rows[1:]])
        # 2002 is not a leap year; the 90 leading days stay clear of 2000
        start = datetime.date(2002, 1, 1) + datetime.timedelta(
            days=int(yday0) - 1)
        rec['seed%d' % s] = dict(lat=lat, lng=lng, elev=elev, start=start,
                                 prec=d[:, 2], tmax=d[:, 3], tmin=d[:, 4],
                                 wind=1.0 + 4.0 * rng.random(len(d)))
    # MetSim indexes sunrise times by day of year over only the days that
    # have a sunrise, so any site with polar day or night in the year (the
    # 68 and 76 N series) is misaligned or fails. The same four weather
    # series are also run at 62 N so that all ten are compared.
    for s in (3, 4, 13, 19):
        r = dict(rec['seed%d' % s])
        r['lat'] = 62.0
        rec['seed%d_lat62' % s] = r
    import xarray as xr
    f = xr.open_dataset(wg_force)
    p = xr.open_dataset(wg_params)
    la, lo = 1, 2
    t = f['temp'].values[:, la, lo]
    pr = f['prec'].values[:, la, lo]
    w = f['wind'].values[:, la, lo]
    nd = len(t) // 8
    rec['walnut_gulch_2008'] = dict(
        lat=float(f.lat.values[la]), lng=float(f.lon.values[lo]),
        elev=float(p['elev'].values[la, lo]), start=datetime.date(2008, 1, 1),
        prec=pr[:nd * 8].reshape(nd, 8).sum(1),
        tmax=t[:nd * 8].reshape(nd, 8).max(1),
        tmin=t[:nd * 8].reshape(nd, 8).min(1),
        wind=w[:nd * 8].reshape(nd, 8).mean(1))
    arrs = {}
    for k, r in rec.items():
        n = len(r['tmax'])
        # leading days: the series' first LEAD days reversed; the record
        # starts at r['start'], so the leading block is the LEAD days before
        for v in ('tmax', 'tmin', 'prec', 'wind'):
            arrs['%s/%s' % (k, v)] = np.concatenate((r[v][:LEAD][::-1], r[v]))
        arrs['%s/meta' % k] = np.array(
            [r['lat'], r['lng'], r['elev'], r['start'].toordinal()])
    arrs['sites'] = np.array(list(rec))
    arrs['lead'] = np.array(LEAD)
    np.savez(out, **arrs)
    print('wrote', out, {k: len(r['tmax']) for k, r in rec.items()})


if __name__ == '__main__':
    main(*sys.argv[1:5])
