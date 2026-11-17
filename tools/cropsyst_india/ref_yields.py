"""Reference yields for the India season check -> <data root>/raw/yield_ref/ref_cells.csv
 GYGA Yp (Aramburu-Merlos et al. 2024, Zenodo 13137398) at each check cell: nearest valid pixel within 0.6 deg
 (irrigated and rainfed rice, wheat, maize only: the only open machine-readable GYGA Yp); FAOSTAT India national
 mean yield 2005-2010 (QCL, element Yield) for all 13 crops.  rasterio via `uv run --offline --with rasterio --with pandas`."""
import os
import numpy as np, pandas as pd, rasterio
R = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'raw/yield_ref')
CELLS = {'rice_kharif': (22.5, 88.0, 'rice'), 'rice_boro': (23.0, 88.0, 'rice'), 'wheat': (30.9, 75.9, 'wheat'),
         'maize': (25.5, 85.5, 'maize')}
FAO = {'rice_kharif': 'Rice', 'rice_boro': 'Rice', 'wheat': 'Wheat', 'maize': 'Maize (corn)', 'sorghum': 'Sorghum',
       'pearl_millet': 'Millet', 'chickpea_pulses': 'Chick peas, dry', 'sugarcane': 'Sugar cane',
       'cotton': 'Seed cotton, unginned', 'groundnut': 'Groundnuts, excluding shelled', 'mustard_rapeseed': 'Rape or colza seed',
       'potato': 'Potatoes', 'soybean': 'Soya beans'}
d = pd.read_csv(R + '/faostat/Production_Crops_Livestock_E_All_Data_(Normalized).csv', encoding='latin1', low_memory=False)
d = d[(d['Area'] == 'India') & (d['Element'] == 'Yield') & d['Year'].between(2005, 2010)]
print(d.Unit.unique(), [i for i in d.Item.unique() if i.startswith(('Rape','Mustard'))])
d = d.assign(v=np.where(d.Unit.str.contains('kg'), d.Value / 1e3, d.Value / 1e4))
rows = []
for k, item in FAO.items():
    s = d[d.Item == item]
    if s.empty:
        s = d[d.Item.str.startswith(item.split(',')[0])]
    rows.append(dict(crop=k, fao_item=s.Item.iloc[0] if len(s) else '', fao_india_2005_10_t_ha=round(s.v.mean(), 2) if len(s) else np.nan))
out = pd.DataFrame(rows)
for sysn in ('Irrigated', 'Rainfed'):
    out['gyga_yp_' + sysn.lower()] = np.nan
for k, (la, lo, c) in CELLS.items():
    for sysn in ('Irrigated', 'Rainfed'):
        with rasterio.open('%s/gyga/%s_%s.tif' % (R, sysn, c)) as t:
            r, cc = t.index(lo, la); n = 72
            w = t.read(1, window=((r - n, r + n + 1), (cc - n, cc + n + 1))).astype(float)
            yy, xx = np.nonzero(np.isfinite(w) & (w > 0))
            j = np.argmin((yy - n) ** 2 + (xx - n) ** 2)
            out.loc[out.crop == k, 'gyga_yp_' + sysn.lower()] = round(w[yy[j], xx[j]], 2)
            out.loc[out.crop == k, 'gyga_dist_deg_' + sysn.lower()] = round(np.hypot(yy[j] - n, xx[j] - n) * t.res[0], 2)
out.to_csv(R + '/ref_cells.csv', index=False)
print(out.to_string())
