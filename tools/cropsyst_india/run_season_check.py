"""One season per crop through the cropsyst backend (torchvic.crop.cropsyst.engine) at a representative
Indian cell, well-watered (layer moisture held at 85 % of the saturated storage: this is an irrigated /
potential-water run, no water stress), ISC daily met 2005-2007.  Writes season_check.csv.  python -I."""
import os, sys, math
import numpy as np, netCDF4 as nc, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torchvic.crop.cropsyst import engine as E

A = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
D = A + '/cropsyst_india'
CELLS = {  # crop: (lat, lon, place) representative cell of the main growing region
    'rice_kharif': (22.5, 88.0, 'West Bengal'), 'rice_boro': (23.0, 88.0, 'West Bengal'),
    'wheat': (30.9, 75.9, 'Punjab'), 'maize': (25.5, 85.5, 'Bihar'), 'sorghum': (18.5, 76.0, 'Maharashtra'),
    'pearl_millet': (26.5, 73.0, 'Rajasthan'), 'chickpea_pulses': (23.5, 77.5, 'Madhya Pradesh'),
    'sugarcane': (27.5, 79.0, 'Uttar Pradesh'), 'cotton': (22.0, 72.0, 'Gujarat'),
    'groundnut': (21.5, 70.5, 'Gujarat'), 'mustard_rapeseed': (27.0, 76.5, 'Rajasthan'),
    'potato': (27.0, 79.0, 'Uttar Pradesh'), 'soybean': (23.0, 76.5, 'Madhya Pradesh')}


# marketable-yield equivalent of the dry yield (moisture / dry-matter content; FLAGGED typical values)
DM = {'rice_kharif': 0.86, 'rice_boro': 0.86, 'wheat': 0.88, 'maize': 0.88, 'sorghum': 0.88, 'pearl_millet': 0.88,
      'chickpea_pulses': 0.88, 'cotton': 0.90, 'groundnut': 0.90, 'mustard_rapeseed': 0.92, 'soybean': 0.88,
      'potato': 0.20, 'sugarcane': 0.30 / 0.70}
REF = pd.read_csv(A + '/raw/yield_ref/ref_cells.csv').set_index('crop')   # ref_yields.py: GYGA Yp (Zenodo 13137398), FAOSTAT India


def e0(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def eto(tx, tn, vp, sp, wind, sw, lw):
    t = (tx + tn) / 2
    es = (e0(tx) + e0(tn)) / 2
    d = 4098 * e0(t) / (t + 237.3) ** 2
    g = 0.000665 * sp
    rs = sw * 0.0864
    tk = t + 273.16
    rnl = (0.97 * 5.67e-8 * tk ** 4 - lw) * 0.0864
    rn = 0.77 * rs - rnl
    u2 = wind * 0.748
    return np.maximum((0.408 * d * rn + g * 900 / tk * u2 * (es - vp)) / (d + g * (1 + 0.34 * u2)), 0.0)


def weather():
    W = {k: [] for k in ('tmax', 'tmin', 'srad', 'rhmin', 'et_ref')}
    names = list(CELLS)
    for y in (2005, 2006, 2007):
        d = nc.Dataset('%s/forcing/vic5_daily/isc_daily.%d.nc' % (A, y))
        lat, lon = np.array(d['lat'][:]), np.array(d['lon'][:])
        v = {k: [] for k in ('tasmax', 'tasmin', 'vp', 'sp', 'wind', 'swdown', 'lwdown')}
        for n in names:
            la, lo = CELLS[n][:2]
            i, j = int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin())
            for k in v:
                v[k].append(np.ma.filled(d[k][:, i, j], np.nan).astype(float))
        v = {k: np.stack(x, 1) for k, x in v.items()}
        W['tmax'].append(v['tasmax']); W['tmin'].append(v['tasmin']); W['srad'].append(v['swdown'])
        W['rhmin'].append(np.clip(100 * v['vp'] / e0(v['tasmax']), 5, 100))
        W['et_ref'].append(eto(v['tasmax'], v['tasmin'], v['vp'], v['sp'], v['wind'], v['swdown'], v['lwdown']))
    return {k: torch.tensor(np.concatenate(x, 0), dtype=torch.float64) for k, x in W.items()}


def main(D=D, write=True):
    cal = pd.read_csv(D + '/calendars_india.csv')
    names = list(CELLS)
    codes = [int(cal[(cal.crop == n) & (cal.system == 'ir')].code.iloc[0]) for n in names]
    w = weather()
    nd = w['tmax'].shape[0]
    year = np.concatenate([np.full(365, 2005), np.full(365, 2006), np.full(nd - 730, 2007)])
    doy = np.concatenate([np.arange(1, 366), np.arange(1, 366), np.arange(1, nd - 730 + 1)])
    w['co2'] = 380.0
    L = 5
    thick = [0.1, 0.3, 0.4, 0.5, 0.7]
    soil = dict(thickness=thick, quartz=[0.4] * L, clay=[0.25] * L)
    cfg = dict(db_dir=D + '/Database', rotations=codes, soil=soil, first_year=2005)
    run = E.build_run(cfg, w, [CELLS[n][0] for n in names], [CELLS[n][1] for n in names], year, doy,
                      len(names), 'cpu')
    mm = run.S.max_moist * 0.85
    st = run.state
    T = len(names)
    first = {}
    bm_at = np.zeros(len(names))
    lai_max = np.zeros(T)
    sow_day = -np.ones(T, int)
    gstress = np.zeros(T); gn = np.zeros(T)
    for d in range(nd):
        st, o = run.step(st, moist_mm=mm)
        lai_max = np.maximum(lai_max, o.lai.numpy())
        for t in range(T):
            if bool(o.sown[t]) and sow_day[t] < 0:
                sow_day[t] = d
            if sow_day[t] >= 0 and d == sow_day[t] + 364:
                bm_at[t] = float(o.biomass[t])
            if sow_day[t] >= 0 and t not in first and bool(o.present[t]):
                gstress[t] += float(o.water_stress[t]); gn[t] += 1
            if bool(o.harvest[t]) and t not in first and sow_day[t] >= 0:
                first[t] = (d, float(o.yield_[t]) * 10.0, float(o.hi[t]))
    rows = []
    for t, n in enumerate(names):
        h = first.get(t)
        cut = False
        if h is None and sow_day[t] >= 0:                    # no maturity in 12 months: cut at the next sowing
            h = (int(sow_day[t]) + 364, float(bm_at[t]) * 10.0, float('nan')); cut = True
        rows.append(dict(crop=n, place=CELLS[n][2], lat=CELLS[n][0], lon=CELLS[n][1], sow_day_index=int(sow_day[t]),
                         harvest_day_index=h[0] if h else -1, season_days=h[0] - sow_day[t] if h else -1,
                         yield_t_ha_dry=round(h[1], 2) if h else float('nan'), hi=round(h[2], 3) if h else float('nan'),
                         lai_max=round(lai_max[t], 2), mean_ws=round(gstress[t] / max(gn[t], 1), 3),
                         cut_at_resowing=cut))
        y = rows[-1]['yield_t_ha_dry'] / DM[n]
        yp, fa = REF.loc[n, 'gyga_yp_irrigated'], REF.loc[n, 'fao_india_2005_10_t_ha']
        rows[-1].update(yield_marketable=round(y, 2), gyga_yp_irrigated=yp, ratio_to_yp=round(y / yp, 2) if yp == yp else float('nan'),
                        fao_india_actual=fa, ratio_to_fao=round(y / fa, 2))
    df = pd.DataFrame(rows)
    if write:
        df.to_csv(D + '/season_check.csv', index=False)
    print(df.to_string())
    return df


if __name__ == '__main__':
    main()
