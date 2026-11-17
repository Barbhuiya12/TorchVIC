#!/usr/bin/env python3
"""Daily ERA5 fields over the ISC box from the Earth Data Hub hourly store.

Reads the DestinE Earth Data Hub ERA5 single-levels hourly Zarr
(reanalysis-era5-single-levels-v0.zarr, chunks 4320 h x 64 x 64, 0.25 deg;
credentials in ~/.netrc, machine data.earthdatahub.destine.eu) chunk by chunk
over HTTP, keeps only the box, and reduces each 180-day time chunk to UTC
daily values:

  d2m   -> vp_kPa      daily mean of hourly Tetens vapour pressure
  sp    -> sp_kPa      daily mean surface pressure
  ssrd  -> swdown_Wm2  daily total / 86400  (hourly accumulations)
  strd  -> lwdown_Wm2  daily total / 86400
  t2m   -> tasmax_C, tasmin_C   daily max and min of hourly t2m (field 't2m'
                                 writes both; as the original ISC store)
  tp    -> prcp_mm     daily total x 1000
  u10, v10 -> wind_ms  daily mean of hourly speed

Accumulated and since-previous-post-processing fields at valid time t cover
the hour ending at t, so they are assigned to the UTC day of t - 1 h.

One .npz per (field, time chunk) under OUT/<field>/; resumable.

    python edh_era5_daily.py --out DIR --fields vp sp sw lw [--box main|tip]
           [--first 1940-01-01 --last 2024-12-31] [--workers 48]
"""
import argparse
import base64
import netrc
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numcodecs
import numpy as np

STORE = 'https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr'
HOST = 'data.earthdatahub.destine.eu'
NT, NY, NX = 4320, 64, 64
T0 = np.datetime64('1940-01-01T00')          # checked against the store's time
BOXES = {'main': (37.75, 66.0, 8.0, 98.0),    # N, W, S, E
         'tip': (9.25, 76.0, 8.0, 78.75)}
FIELDS = {'vp': ['d2m'], 'sp': ['sp'], 'sw': ['ssrd'], 'lw': ['strd'],
          'tmax': ['t2m'], 'tmin': ['t2m'], 't2m': ['t2m'], 'prcp': ['tp'],
          'wind': ['u10', 'v10']}
ACCUM = {'ssrd', 'strd', 'tp'}


def auth_header():
    user, _, pw = netrc.netrc().authenticators(HOST)
    return 'Basic ' + base64.b64encode(('%s:%s' % (user, pw)).encode()).decode()


AUTH = auth_header()
BLOSC = numcodecs.Blosc()


def get(url, tries=60):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Authorization': AUTH})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            print('retry %s: %s' % (url.rsplit('/', 2)[-2:], e), flush=True)
            time.sleep(min(10 * (i + 1), 120))


def chunk(var, it, iy, ix):
    raw = BLOSC.decode(get('%s/%s/%d.%d.%d' % (STORE, var, it, iy, ix)))
    return np.frombuffer(raw, '<f4').reshape(NT, NY, NX)


def box_index(box):
    n, w, s, e = box
    lat = 90.0 - 0.25 * np.arange(721)
    lon = 0.25 * np.arange(1440)
    iy = np.where((lat <= n + 1e-9) & (lat >= s - 1e-9))[0]
    ix = np.where((lon >= w - 1e-9) & (lon <= e + 1e-9))[0]
    return iy, ix, lat[iy], lon[ix]


def read_box(var, it, iy, ix, pool):
    cy = sorted(set(iy // NY))
    cx = sorted(set(ix // NX))
    parts = {(a, b): pool.submit(chunk, var, it, a, b) for a in cy for b in cx}
    big = np.concatenate([np.concatenate([parts[(a, b)].result() for b in cx], 2)
                          for a in cy], 1)
    return big[:, iy - cy[0] * NY][:, :, ix - cx[0] * NX]


def tetens_kpa(td_k):
    t = td_k - 273.15
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def reduce_daily(field, arrs, hours):
    """Partial daily reductions of one time chunk.

    hours: datetime64[h] of each step. Returns (days, value[day, y, x],
    count[day]): a sum for means and totals, a max or min otherwise. Days
    cut by a chunk edge are partial here and are completed when the chunks
    are assembled (sums add, maxima take the max, counts add to 24)."""
    var = FIELDS[field][0]
    shift = np.timedelta64(1, 'h') if var in ACCUM else np.timedelta64(0, 'h')
    day = (hours - shift).astype('datetime64[D]')
    days = np.unique(day)
    a = arrs[0]
    if field == 'vp':
        a = tetens_kpa(a)
    elif field == 'wind':
        a = np.hypot(arrs[0], arrs[1])
    out = np.empty((len(days),) + a.shape[1:], np.float64)
    cnt = np.zeros(len(days), np.int16)
    for k, d in enumerate(days):
        m = day == d
        cnt[k] = int(m.sum())
        x = a[m]
        out[k] = x.max(0) if field == 'tmax' else x.min(0) if field == 'tmin' \
            else x.sum(0, dtype=np.float64)
    return days, out, cnt


# how a finished day is formed from the combined partials (count == 24)
FINISH = {'vp': lambda v: v / 24, 'sp': lambda v: v / 24 / 1000.0,
          'wind': lambda v: v / 24, 'sw': lambda v: v / 86400.0,
          'lw': lambda v: v / 86400.0, 'prcp': lambda v: v * 1000.0,
          'tmax': lambda v: v - 273.15, 'tmin': lambda v: v - 273.15}
COMBINE = {'tmax': np.fmax, 'tmin': np.fmin}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fields', nargs='+', required=True, choices=list(FIELDS))
    ap.add_argument('--box', default='main', choices=list(BOXES))
    ap.add_argument('--first', default='1940-01-01')
    ap.add_argument('--last', default='2024-12-31')
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()
    iy, ix, lat, lon = box_index(BOXES[args.box])
    t_first = np.datetime64(args.first + 'T00')
    t_last = np.datetime64(args.last + 'T23')
    i0 = int((t_first - T0) / np.timedelta64(1, 'h')) // NT
    i1 = int((t_last + np.timedelta64(1, 'h') - T0) / np.timedelta64(1, 'h')) // NT
    with ThreadPoolExecutor(args.workers) as pool:
        for field in args.fields:
            d = os.path.join(args.out, args.box, field)
            os.makedirs(d, exist_ok=True)
            # one time chunk at a time; its spatial chunks (and the next
            # chunk's first hour for accumulations) are fetched in parallel
            tasks = []
            for it in range(i0, i1 + 1):
                path = os.path.join(d, '%s_%04d.npz' % (field, it))
                done = [path] if field != 't2m' else [
                    path.replace('/t2m/', '/tmax/').replace('t2m_', 'tmax_'),
                    path.replace('/t2m/', '/tmin/').replace('t2m_', 'tmin_')]
                if not all(os.path.exists(x) for x in done):
                    tasks.append((it, path))

            def one(task):
                it, path = task
                t0 = time.time()
                hrs = T0 + np.arange(it * NT, (it + 1) * NT).astype('timedelta64[h]')
                arrs = [read_box(v, it, iy, ix, pool) for v in FIELDS[field]]
                for f, pth in ((('tmax', path.replace('/t2m/', '/tmax/').replace('t2m_', 'tmax_')),
                                ('tmin', path.replace('/t2m/', '/tmin/').replace('t2m_', 'tmin_')))
                               if field == 't2m' else ((field, path),)):
                    os.makedirs(os.path.dirname(pth), exist_ok=True)
                    days, part, cnt = reduce_daily(f, arrs, hrs)
                    np.savez(pth + '.tmp.npz', days=days.astype(str), count=cnt,
                             lat=lat, lon=lon, part=part)
                    os.replace(pth + '.tmp.npz', pth)
                return it, time.time() - t0
            with ThreadPoolExecutor(max(1, args.workers // 9)) as outer:
                for k, (it, dt) in enumerate(outer.map(one, tasks), 1):
                    print('%s %s %s chunk %d (%d/%d) %.0f s' % (
                        time.strftime('%H:%M:%S'), args.box, field, it, k,
                        len(tasks), dt), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
