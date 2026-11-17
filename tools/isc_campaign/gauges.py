"""Gauge scores of a run: KGE, bias and peak timing at the ISC gauges that
have 2005 observations (<data root>/discharge/final_basins; >= 180 days).

Model cell: within 0.35 deg of the gauge, the one whose accumulated upstream
area is closest to the gauge's catchment area (as plots_isc / fix_isc_gw).
  python gauges.py RUN_DIR      -> RUN_DIR/gauges.csv and a summary line
"""
import os
import sys

import numpy as np
import pandas as pd

R = os.environ.get('TORCHVIC_DATA_ROOT', 'data').rstrip('/') + '/'
CACHE = R + 'verification/isc_campaign/gauge_cells.csv'


def _network():
    import netCDF4 as nc
    dom = nc.Dataset(R + 'vic_params_isc/domain.nc')
    lat, lon = dom['lat'][:], dom['lon'][:]
    mask = np.ma.filled(dom['mask'][:], 0) > 0
    area = np.ma.filled(dom['area'][:], 0)[mask].astype(float)
    N = int(mask.sum())
    ds = nc.Dataset(R + 'vicwur/isc/smoke/rout_n4.nc')
    cid = np.ma.filled(ds['downstream_id'][:], -1)[mask]
    dwn = np.ma.filled(ds['downstream'][:], -1)[mask]
    pos = {int(c): i for i, c in enumerate(cid)}
    dsn = np.array([pos.get(int(d), i) for i, d in enumerate(dwn)])
    up = area / 1e6
    indeg = np.bincount(dsn[dsn != np.arange(N)], minlength=N)
    st = [i for i in range(N) if indeg[i] == 0]
    acc = up.copy()
    while st:
        i = st.pop()
        j = dsn[i]
        if j == i:
            continue
        acc[j] += acc[i]
        indeg[j] -= 1
        if indeg[j] == 0:
            st.append(j)
    idx = np.argwhere(mask)
    return lat, lon, idx, acc


def gauge_table():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    lat, lon, idx, acc = _network()
    b = pd.read_csv(R + 'discharge/final_basins/basin_index.csv')
    rows = []
    for _, g in b.iterrows():
        f = R + 'discharge/final_basins/discharge/%d.csv' % g.clean_id
        if not os.path.exists(f):
            continue
        o = pd.read_csv(f, parse_dates=['Date']).set_index('Date')['Q']
        o = o[o.index.year == 2005]
        if o.notna().sum() < 180:
            continue
        la, lo = lat[idx[:, 0]], lon[idx[:, 1]]
        i0 = np.argmin((la - g.lat) ** 2 + (lo - g.lng) ** 2)
        d = np.hypot(la - la[i0], lo - lo[i0])
        cand = np.where(d < 0.35)[0]
        sc = np.abs(np.log(acc[cand] / g.area)) + 2 * d[cand]
        c = cand[np.argmin(sc)]
        rows.append(dict(id=int(g.clean_id), name=g['name'], river=g.river,
                         area=g.area, cell=int(c), cell_area=acc[c],
                         ratio=acc[c] / g.area, n_obs=int(o.notna().sum())))
    t = pd.DataFrame(rows)
    t.to_csv(CACHE, index=False)
    return t


def kge(s, o):
    if s.std() == 0 or o.std() == 0:
        return float('nan')
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2
                       + (s.mean() / o.mean() - 1) ** 2)


def score(q, tab=None, last_date=pd.Timestamp('2005-12-31')):
    """q [T, N] m3/s, ending on 31 Dec 2005; DataFrame of scores."""
    tab = gauge_table() if tab is None else tab
    T = q.shape[0]
    dates = pd.date_range(end=last_date, periods=T)
    rows = []
    for _, g in tab.iterrows():
        o = pd.read_csv(R + 'discharge/final_basins/discharge/%d.csv' % g.id,
                        parse_dates=['Date']).set_index('Date')['Q'] \
            .reindex(dates)
        s = q[:, int(g.cell)].astype(float)
        ok = o.notna().values & np.isfinite(s)
        if ok.sum() < 100:
            continue
        so, oo = s[ok], o.values[ok]
        po = dates[ok][np.argmax(oo)]
        ps = dates[ok][np.argmax(so)]
        rows.append(dict(id=g.id, name=g['name'], river=g.river,
                         area_ratio=g.ratio, KGE=kge(so, oo),
                         bias_pct=100 * (so.mean() / oo.mean() - 1),
                         peak_lag_d=(ps - po).days,
                         sim_mean=so.mean(), obs_mean=oo.mean(),
                         neg_frac=float((s < 0).mean())))
    return pd.DataFrame(rows)


def summary(df):
    big = df[(df.area_ratio > 0.5) & (df.area_ratio < 2)]
    return dict(n=len(df), n_matched=len(big),
                kge_median=float(big.KGE.median()),
                kge_pos=float((big.KGE > 0).mean()),
                bias_median=float(big.bias_pct.median()),
                peak_lag_abs_median=float(big.peak_lag_d.abs().median()),
                neg_gauges=int((df.neg_frac > 0).sum()))


if __name__ == '__main__':
    rd = sys.argv[1]
    z = np.load(rd + '/series.npz')
    df = score(z['vicwur_routing.discharge'])
    df.to_csv(rd + '/gauges.csv', index=False)
    print(summary(df))
