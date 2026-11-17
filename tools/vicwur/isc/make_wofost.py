#!/usr/bin/env python
"""VIC-WUR WOFOST inputs for the ISC domain, built from open data.

Replaces the two files of VIC-WUR that are not public
(`wofost_params_VICWOFOST_global.txt`, `crop_params_VICWOFOST_global.nc`):

  <data root>/vicwur/isc/wofost/wofost_params_VICWOFOST_isc.txt   one line per crop class
  <data root>/vicwur/isc/wofost/crops/crop_<class>.txt            77 scalars + 22 tables
  <data root>/vicwur/isc/wofost/crops/management_default.txt
  <data root>/vicwur/isc/wofost/crop_params_VICWOFOST_isc.nc      crop nc (calendar, cover, TSUM, fert)
  <data root>/vicwur/isc/wofost/params_provenance.csv             source of every value

Format: torchvic/vicwur/params.py (read_wofost_text, read_crop) and
torchvic/crop/wofost_npk/params.py; guide tools/vicwur/ref_domains.py.

Crop parameters: ajwdewit/WOFOST_crop_parameters, branch wofost81 (EUPL),
core and nitrogen values; branch wofost72 (EUPL) is the fallback for a name
the 8.1 file lacks.  P, K and the VIC feedback values are not in the library
and are flagged defaults (see README).  Calendars and areas: MIRCA-OS
(year 2000, same year as Cv in params_wur.nc); statistical unit polygons:
Natural Earth admin-1 (public domain).  Soil carbon and pH: SoilGrids 2.0.

Crop classes (veg class of params_wur.nc): 13 rainfed crop, 18 irrigated
non-paddy, 19 paddy.  Rice irrigated is split into a kharif class and a
rabi/boro class (both veg 19); rainfed rice is veg 13.
"""
import difflib
import glob
import os
import re
import sys
import numpy as np
import netCDF4 as nc
import pandas as pd
import yaml
import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa
from regrid import Regridder, fill_nearest   # noqa

LIB81 = RAW + '/wofost_crop_parameters_wofost81/repo/'
LIB72 = ROOT + '/static/upstream/wofost_crop_parameters/wofost72/'
OUTW = OUT + '/wofost'
MIRCA = glob.glob(WORK + '/mirca/bag/*/data/contents')[0]
YEAR = 2000
V_RF, V_IRR, V_PAD = 13, 18, 19
NVEG = 20

# --------------------------------------------------------------- crop list
# (class name, library file, variety, MIRCA grid crop, MIRCA calendar crops,
#  system, veg class, season, start at emergence, max height m)
RICE_KH, RICE_BO = 'rice_kharif_ir', 'rice_boro_ir'
CLASSES = [
    (RICE_KH, 'rice', 'Rice_IR72_WS', 'Rice', ('Rice1', 'Rice2', 'Rice3'), 'ir', V_PAD, 'kharif', 1, 1.0),
    (RICE_BO, 'rice', 'Rice_IR72_DS', 'Rice', ('Rice1', 'Rice2', 'Rice3'), 'ir', V_PAD, 'rabi', 1, 1.0),
    ('rice_rf', 'rice', 'Rice_IR72_WS', 'Rice', ('Rice1', 'Rice2', 'Rice3'), 'rf', V_RF, 'any', 1, 1.0),
]
OTHERS = [
    # name, file, variety, MIRCA grid name, MIRCA calendar names, height
    ('wheat', 'wheat', 'Winter_wheat_106', 'Wheat', ('Wheat1', 'Wheat2'), 0.9),
    ('maize', 'maize', 'Maize_VanHeemst_1988', 'Maize', ('Maize',), 2.5),
    ('sorghum', 'sorghum', 'Sorghum_VanHeemst_1988', 'Sorghum', ('Sorghum',), 2.0),
    ('millet', 'millet', 'Millet_VanHeemst_1988', 'Millet', ('Millet',), 1.8),
    ('pulses', 'chickpea', 'Chickpea_VanHeemst_1988', 'Pulses', ('Pulses',), 0.6),
    ('sugarcane', 'sugarcane', 'Sugarcane_VanHeemst_1988', 'Sugar_cane', ('Sugar cane',), 3.0),
    ('cotton', 'cotton', 'Cotton_VanHeemst_1988', 'Cotton', ('Cotton',), 1.2),
    ('groundnut', 'groundnut', 'Groundnut_VanHeemst_1988', 'Groundnuts', ('Groundnuts',), 0.4),
    ('potato', 'potato', 'Potato_704', 'Potatoes', ('Potatoes',), 0.6),
    ('rapeseed', 'rapeseed', 'Oilseed_rape_1004', 'Rapeseed', ('Rapeseed',), 1.2),
]
for nm, f, v, g, cal, h in OTHERS:
    CLASSES.append((nm + '_rf', f, v, g, cal, 'rf', V_RF, 'any', 0, h))
    CLASSES.append((nm + '_ir', f, v, g, cal, 'ir', V_IRR, 'any', 0, h))
NC = len(CLASSES)

# Indicative fertiliser (kg/ha elemental N, P, K per season, irrigated; rainfed
# x0.5).  FLAGGED DEFAULTS typical of Indian practice, not a dataset.
FERT = {'rice': (100, 18, 33), 'wheat': (120, 26, 33), 'maize': (120, 26, 33),
        'sorghum': (60, 13, 8), 'millet': (40, 9, 8), 'pulses': (20, 20, 17),
        'sugarcane': (150, 26, 50), 'cotton': (80, 17, 33),
        'groundnut': (20, 17, 25), 'potato': (120, 35, 83),
        'rapeseed': (80, 17, 17)}
FERT_DVS = (0.2, 0.6)         # basal (50 % N, all P and K), top dress (50 % N)

# ------------------------------------------------------------ text format
SCALARS = ('TBASEM TEFFMX TSUMEM IDSL DLO DLC VERNSAT VERNBASE TSUM1 TSUM2 '
           'DVSI DVSEND TDWI RGRLAI SPA SPAN TBASE CVL CVO CVR CVS Q10 RML '
           'RMO RMR RMS PERDL CFET DEPNR IAIRDU RDI RRI RDMCR RDRLV_NPK '
           'DVS_NPK_STOP DVS_NPK_TRANSL NPK_TRANSLRT_FR NCRIT_FR PCRIT_FR '
           'KCRIT_FR NMAXRT_FR NMAXST_FR PMAXRT_FR PMAXST_FR KMAXRT_FR '
           'KMAXST_FR NLAI_NPK NLUE_NPK NMAXSO PMAXSO KMAXSO NPART NSLA_NPK '
           'NRESIDLV NRESIDST NRESIDRT PRESIDLV PRESIDST PRESIDRT KRESIDLV '
           'KRESIDST KRESIDRT TCNT TCPT TCKT NFIX_FR MaxHeight RGL RadAtten '
           'WindAtten TrunkRatio Albedo MinStomResist MaxArchResist Fcanopy '
           'FRTRL CRITLAI').split()
TABLES = ('VERNRTB DTSMTB SLATB SSATB KDIFTB EFFTB AMAXTB TMPFTB TMNFTB '
          'CO2AMAXTB CO2EFFTB CO2TRATB RFSETB FRTB FLTB FSTB FOTB RDRRTB '
          'RDRSTB NMAXLV_TB PMAXLV_TB KMAXLV_TB').split()
assert len(SCALARS) == 77 and len(TABLES) == 22
YAML_ALIAS = {'DVS_NPK_TRANSL': 'DVS_N_TRANSL'}      # name in the 8.1 file

# Flagged defaults for what neither library branch has.  The N:P:K ratios of
# the maximum concentrations follow the R1 reference set (tools/vicwur/
# ref_domains.py: P about N/5.3 in organs, N/7 in leaves; K = N in leaves).
DEFAULTS = dict(
    RDRLV_NPK=0.05, DVS_NPK_STOP=1.3, NPK_TRANSLRT_FR=0.15, PCRIT_FR=1.0,
    KCRIT_FR=1.0, PMAXRT_FR=0.5, PMAXST_FR=0.5, KMAXRT_FR=0.5, KMAXST_FR=0.5,
    NLAI_NPK=1.0, NLUE_NPK=1.1, NPART=1.0, NSLA_NPK=0.5,
    PRESIDLV=0.0005, PRESIDST=0.0003, PRESIDRT=0.0003,
    KRESIDLV=0.0025, KRESIDST=0.0050, KRESIDRT=0.0040, TCPT=10.0, TCKT=10.0,
    RGL=100.0, RadAtten=0.5, WindAtten=0.5, TrunkRatio=0.2, Albedo=0.2,
    MinStomResist=100.0, MaxArchResist=25.0, Fcanopy=0.9, FRTRL=0.2,
    CRITLAI=0.05, DVS_NPK_TRANSL=0.8, NCRIT_FR=1.0, NMAXRT_FR=0.5,
    NMAXST_FR=0.5, NRESIDLV=0.004, NRESIDST=0.0015, NRESIDRT=0.0015,
    TCNT=10.0, NFIX_FR=0.0, NMAXSO=0.031, VERNSAT=-99.0, VERNBASE=-99.0,
    TEFFMX=30.0, TSUMEM=0.0)
PMAXSO_RATIO, KMAXSO_RATIO = 0.0059 / 0.0310, 0.0045 / 0.0310
P_LV_RATIO, K_LV_RATIO = 1.0 / 7.0, 1.0
DEF_VERN = [-8.0, 0.0, -4.0, 0.4, 3.0, 1.0, 10.0, 0.0, 17.0, 0.0]


def fmt(x):
    return '%.8g' % x


def load_variety(libfile, variety):
    out = {}
    for tag, base in (('wofost81', LIB81), ('wofost72', LIB72)):
        p = base + libfile + '.yaml'
        if not os.path.exists(p):
            continue
        d = yaml.safe_load(open(p))['CropParameters']['Varieties']
        if variety in d:
            for k, v in d[variety].items():
                if k != 'Metadata' and k not in out:
                    out[k] = (v[0], tag)
    return out


def build_crop(cls):
    """-> (scalars, tables, provenance rows) of one crop class."""
    name, lib, var_, _, _, sysm, veg, seas, emerg, hgt = cls
    y = load_variety(lib, var_)
    sc, tb, prov = {}, {}, []
    for k in SCALARS:
        ky = YAML_ALIAS.get(k, k)
        if k == 'MaxHeight':
            v, src = hgt, 'DEFAULT (FAO-56 table 12 typical crop height, m)'
        elif ky in y:
            v, src = y[ky]
            v = float(v)
            src = ('library %s' % src)
        elif k == 'PMAXSO':
            v, src = sc['NMAXSO'] * PMAXSO_RATIO, 'DEFAULT (NMAXSO x R1 P:N ratio)'
        elif k == 'KMAXSO':
            v, src = sc['NMAXSO'] * KMAXSO_RATIO, 'DEFAULT (NMAXSO x R1 K:N ratio)'
        elif k in DEFAULTS:
            v, src = DEFAULTS[k], 'DEFAULT'
        else:
            raise KeyError('%s: no value for %s' % (name, k))
        if k in ('VERNSAT', 'VERNBASE') and int(sc.get('IDSL', y['IDSL'][0])) < 2:
            v, src = -99.0, 'set by reader (IDSL < 2)'
        sc[k] = float(v)
        prov.append((name, k, fmt(sc[k]), src))
    for t in TABLES:
        if t in y:
            v = [float(a) for a in y[t][0]]
            src = 'library %s' % y[t][1]
        elif t in ('PMAXLV_TB', 'KMAXLV_TB'):
            n = [float(a) for a in y['NMAXLV_TB'][0]]
            r = P_LV_RATIO if t[0] == 'P' else K_LV_RATIO
            v = [a if i % 2 == 0 else a * r for i, a in enumerate(n)]
            src = 'DEFAULT (NMAXLV_TB x %.4g)' % r
        elif t == 'VERNRTB':
            if sc['IDSL'] >= 2:
                v, src = DEF_VERN, 'DEFAULT'
            else:
                continue
        else:
            raise KeyError('%s: no table %s' % (name, t))
        tb[t] = v
        prov.append((name, t, ' '.join(fmt(a) for a in v), src))
    return sc, tb, prov


def write_crop_file(path, name, lib, var_, sc, tb):
    L = ['* WOFOST crop file for VIC-WUR ISC: class %s, library file %s, '
         'variety %s' % (name, lib, var_),
         '* built by tools/vicwur/isc/make_wofost.py; sources in '
         'params_provenance.csv', '']
    for k in SCALARS:
        v = sc[k]
        L.append('%s = %s' % (k, fmt(v)))
    L.append('')
    for t in TABLES:
        if t not in tb:
            continue
        v = tb[t]
        pts = list(zip(v[0::2], v[1::2]))
        L.append('%s = %s , %s' % (t, fmt(pts[0][0]), fmt(pts[0][1])))
        for a, b in pts[1:]:
            L.append('  %s , %s' % (fmt(a), fmt(b)))
        L.append('')
    open(path, 'w').write('\n'.join(L) + '\n')


def write_mgmt(path):
    L = ['* WOFOST management file (default); fertiliser comes from the crop nc',
         '', 'NRFTAB = 0.7', 'PRFTAB = 0.7', 'KRFTAB = 0.7', 'NMINS = 0.0',
         'RTNMINS = 0.0', 'PMINS = 0.0', 'RTPMINS = 0.0', 'KMINS = 0.0',
         'RTKMINS = 0.0', '', 'FERNTAB = 01-01 0.0', '',
         'FERPTAB = 01-01 0.0', '', 'FERKTAB = 01-01 0.0', '',
         'IRRTAB = 01-01 0.0', '']
    open(path, 'w').write('\n'.join(L))


# ------------------------------------------------------- MIRCA calendars
ISO = {'IND': 356, 'PAK': 586, 'NPL': 524, 'BGD': 50, 'BTN': 64, 'AFG': 4,
       'CHN': 156, 'MMR': 104, 'LKA': 144, 'IRN': 364, 'TJK': 762,
       'UZB': 860, 'TKM': 795, 'KGZ': 417, 'KAZ': 398, 'OMN': 512,
       'ARE': 784, 'THA': 764, 'LAO': 418, 'MNG': 496, 'RUS': 643,
       'TUR': 792, 'IRQ': 368}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower().replace('&', 'and'))


def read_calendar(system):
    p = '%s/Crop Calendar/MIRCA-OS_%d_%s.csv' % (MIRCA, YEAR, system)
    d = pd.read_csv(p, encoding='latin1')
    d = d.rename(columns={'State': 'Province'})
    d['cc'] = d.unit_code // 10000
    d['prov'] = d.Province.fillna('').map(norm)
    return d


def unit_table(cal, ccode, prov):
    """rows of the unit (province, else country, else all)."""
    if ccode is not None:
        s = cal[cal.cc == ccode]
        if prov is not None and (s.prov == prov).any():
            return s[s.prov == prov]
        if len(s):
            return s
    return cal


#: planting months a crop's library variety can use (MIRCA-OS lists a spurious
#: Jun-Oct wheat sub-crop with the same area as the real one in many units)
WINDOW = {'wheat': (10, 11, 12, 1, 2), 'rapeseed': (9, 10, 11, 12, 1)}


def pick(sub, names, season, window=None):
    """Dominant (planting, maturity, weight) entry for crop names / season."""
    s = sub[sub.Crop.isin(names)]
    if window is not None and s.Planting_Month.isin(window).any():
        s = s[s.Planting_Month.isin(window)]
    if season == 'kharif':
        s2 = s[s.Planting_Month.between(4, 9)]
    elif season == 'rabi':
        s2 = s[~s.Planting_Month.between(4, 9)]
    else:
        s2 = s
    if len(s2) == 0:
        s2 = s
    g = s2.groupby(['Planting_Month', 'Maturity_Month']).Growing_area.sum()
    if g.sum() > 0:
        (pm, mm), w = g.idxmax(), g.max()
    else:
        (pm, mm), w = g.index[0], 0.0
    return int(pm), int(mm), float(w)


def kharif_share(sub, names):
    s = sub[sub.Crop.isin(names)]
    a = s.Growing_area.sum()
    if a <= 0:
        return None
    return s[s.Planting_Month.between(4, 9)].Growing_area.sum() / a


def doy(month, day=15):
    dm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return int(sum(dm[:month - 1]) + day)


# ----------------------------------------------------------------- grids
def read_tif(path, g):
    with rasterio.open(path) as ds:
        w = from_bounds(g.lon[0] - 0.5, g.lat[0] - 0.5, g.lon[-1] + 0.5,
                        g.lat[-1] + 0.5, ds.transform)
        w = w.round_offsets().round_lengths()
        a = ds.read(1, window=w, boundless=True, fill_value=0).astype(float)
        t = ds.window_transform(w)
        nod = ds.nodata
    if nod is not None:
        a[a == nod] = 0.0
    a[~np.isfinite(a)] = 0.0
    a[a < 0] = 0.0
    ny, nx = a.shape
    lat = t.f + t.e * (np.arange(ny) + 0.5)
    lon = t.c + t.a * (np.arange(nx) + 0.5)
    return lat[::-1], lon, a[::-1]


def mirca_grid(crop, system):
    pat = '%s/Annual Harvested Area Grids/**/%d/5-arcminute/MIRCA-OS_%s_%d_%s.tif' % (
        MIRCA, YEAR, crop, YEAR, system)
    hit = glob.glob(pat, recursive=True)
    if not hit:
        raise FileNotFoundError(pat)
    return hit[0]


def soilgrids(g, name, scale):
    p = RAW + '/soilgrids_isc/%s_ISC_5km.tif' % name
    with rasterio.open(p) as ds:
        a = ds.read(1).astype(float)
        t = ds.transform
        nod = ds.nodata
    if nod is not None:
        a[a == nod] = np.nan
    a[a <= 0] = np.nan                 # 0 = no data over water and gaps
    ny, nx = a.shape
    lat = t.f + t.e * (np.arange(ny) + 0.5)
    lon = t.c + t.a * (np.arange(nx) + 0.5)
    rg = Regridder(lat[::-1], lon, g.lat, g.lon)
    m, cov = rg.mean(a[::-1] * scale)
    return fill_nearest(m)


def nc_var(ds, name, dt, dims, data, units=None, fill=None, **at):
    v = ds.createVariable(name, dt, dims, fill_value=fill, zlib=True,
                          complevel=2)
    if units:
        v.units = units
    for k, a in at.items():
        v.setncattr(k, a)
    v[:] = data
    return v


def main():
    g = ISC()
    rows = []
    os.makedirs(OUTW + '/crops', exist_ok=True)
    # ---- crop files, provenance
    prov = []
    scal_all = []
    for cls in CLASSES:
        sc, tb, pv = build_crop(cls)
        prov += pv
        scal_all.append(sc)
        write_crop_file('%s/crops/crop_%s.txt' % (OUTW, cls[0]), cls[0],
                        cls[1], cls[2], sc, tb)
    write_mgmt(OUTW + '/crops/management_default.txt')
    pd.DataFrame(prov, columns=['class', 'parameter', 'value', 'source']
                 ).to_csv(OUTW + '/params_provenance.csv', index=False)
    L = ['** WOFOST text parameter file for VIC-WUR ISC (one line per crop '
         'class, same order as crop_class of the crop nc)', '']
    for cls in CLASSES:
        L.append('%s/crops/ crop_%s.txt management_default.txt 01-01 %d'
                 % (OUTW, cls[0], cls[8]))
    open(OUTW + '/wofost_params_VICWOFOST_isc.txt', 'w').write('\n'.join(L) + '\n')
    log('wofost: wrote %d crop files' % NC)

    # ---- unit of each cell (Natural Earth admin-1)
    ne = gpd.read_file(WORK + '/ne_admin1/ne_10m_admin_1_states_provinces.shp')
    ne = ne[['adm0_a3', 'name', 'geometry']].copy()
    LAT, LON = np.meshgrid(g.lat, g.lon, indexing='ij')
    pts = gpd.GeoDataFrame({'k': np.arange(g.ny * g.nx)},
                           geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()),
                           crs='EPSG:4326')
    j = gpd.sjoin(pts, ne, how='left', predicate='within')
    j = j.drop_duplicates('k').set_index('k').sort_index()
    miss = j['adm0_a3'].isna().values & g.active.ravel()
    if miss.any():
        m = gpd.sjoin_nearest(pts[miss], ne.to_crs('EPSG:4326'), how='left')
        m = m.drop_duplicates('k').set_index('k')
        j.loc[m.index, 'adm0_a3'] = m['adm0_a3']
        j.loc[m.index, 'name'] = m['name']
    log('wofost: cells outside every admin-1 polygon (nearest used): %d' % int(miss.sum()))
    a3 = j['adm0_a3'].values
    nm = j['name'].values
    cal = {s: read_calendar(s) for s in ('ir', 'rf')}
    # province name match for each (country, NE name)
    prov_names = {}
    for s in cal:
        for cc, grp in cal[s].groupby('cc'):
            prov_names[(s, cc)] = sorted(set(grp.prov) - {''})
    unit_key = {}
    matched = {}
    for c3, nme in sorted(set(zip(a3[g.active.ravel()], nm[g.active.ravel()]))):
        cc = ISO.get(c3)
        key = None
        if cc is not None:
            names = prov_names.get(('ir', cc), [])
            hit = difflib.get_close_matches(norm(nme), names, 1, 0.72)
            key = (cc, hit[0] if hit else None)
        unit_key[(c3, nme)] = key
        matched[(c3, nme)] = bool(key and key[1])
    ctry = {}
    for (c3, nme), k in unit_key.items():
        ctry.setdefault(c3, [0, 0])
        ctry[c3][0] += 1
        ctry[c3][1] += matched[(c3, nme)]
    for c3, (n, mt) in sorted(ctry.items()):
        rows.append('admin-1 units %s: %d, matched to a MIRCA province: %d' % (c3, n, mt))
    ukeys = sorted(set(unit_key.values()), key=lambda x: (x is None, str(x)))
    uidx = {k: i for i, k in enumerate(ukeys)}
    cell_unit = np.array([uidx[unit_key[(a, b)]] if (a, b) in unit_key else -1
                          for a, b in zip(a3, nm)])
    # ---- per class and unit calendars
    nU = len(ukeys)
    plant = np.zeros((NC, nU), int)
    harv = np.zeros((NC, nU), int)
    kshare = np.ones(nU)
    for ui, key in enumerate(ukeys):
        cc, prov_ = key if key else (None, None)
        sub_by = {s: unit_table(cal[s], cc, prov_) for s in cal}
        for ci, cls in enumerate(CLASSES):
            sysm, names, seas = cls[5], cls[4], cls[7]
            sub = sub_by[sysm]
            pm, mm, w = pick(sub, names, seas, WINDOW.get(cls[0].rsplit('_', 1)[0]))
            plant[ci, ui], harv[ci, ui] = doy(pm), doy(mm)
        ks = kharif_share(sub_by['ir'], ('Rice1', 'Rice2', 'Rice3'))
        if ks is None:
            ks = kharif_share(cal['ir'][cal['ir'].cc.isin([ISO['IND']])],
                              ('Rice1', 'Rice2', 'Rice3'))
        kshare[ui] = ks
    # ---- crop areas on the ISC grid (fraction of cell) per class
    frac = np.zeros((NC, g.ny, g.nx))
    for ci, cls in enumerate(CLASSES):
        lat, lon, ha = read_tif(mirca_grid(cls[3], cls[5]), g)
        rg = Regridder(lat, lon, g.lat, g.lon)
        dens = ha * 1e4 / rg.src_area()
        m, cov = rg.mean(dens)
        m = np.nan_to_num(m)
        st, dt, pe = rg.mass_check(dens, m)
        rows.append('MIRCA %-12s %s %d harvested area: src %.6e ha dst %.6e ha err %.2e %%'
                    % (cls[3], cls[5], YEAR, st / 1e4, dt / 1e4, pe))
        frac[ci] = m
    ks_cell = np.where(cell_unit >= 0, kshare[np.maximum(cell_unit, 0)], 1.0
                       ).reshape(g.ny, g.nx)
    ik, ib = 0, 1
    rice_ir = frac[ik].copy()
    frac[ik] = rice_ir * ks_cell
    frac[ib] = rice_ir * (1 - ks_cell)
    # ---- cover Cc: shares within each veg tile scaled to Cv(2000)
    with nc.Dataset(OUT + '/params/params_wur.nc') as p:
        cv = np.array(p['Cv'][:], float)
    vegs = np.array([c[6] for c in CLASSES])
    cc_ = np.zeros_like(frac)
    nfilled = {}
    for v in (V_RF, V_IRR, V_PAD):
        idx = np.where(vegs == v)[0]
        tot = frac[idx].sum(0)
        share = np.where(tot > 0, frac[idx] / np.where(tot > 0, tot, 1), np.nan)
        need = (tot <= 0) & (cv[v - 1] > 0) & g.active
        nfilled[v] = int(need.sum())
        for k in range(len(idx)):
            share[k] = np.where(np.isfinite(share[k]), share[k], np.nan)
        # no MIRCA crop where the tile exists: nearest cell's shares
        sh = np.stack([fill_nearest(share[k], valid=(tot > 0)) for k in range(len(idx))])
        cc_[idx] = sh * cv[v - 1][None]
    cc_[:, ~g.active] = 0.0
    cc_[cc_ < 1e-12] = 0.0
    ncrop = (cc_ > 0).sum(0).astype('i4')
    rows.append('cells where a veg tile has Cv > 0 but no MIRCA crop (nearest shares): %s' % nfilled)
    for v in (V_RF, V_IRR, V_PAD):
        idx = np.where(vegs == v)[0]
        d = (cc_[idx].sum(0) - cv[v - 1])[g.active]
        rows.append('veg %d: sum(Cc) - Cv(%d): max abs %.3e, cells with Cc sum < Cv %d' % (
            v, YEAR, np.abs(d).max(), int((d < -1e-9).sum())))
    # ---- cell fields
    pl = np.where(cell_unit[None] >= 0, plant[:, np.maximum(cell_unit, 0)],
                  plant[:, :1]).reshape(NC, g.ny, g.nx)
    hv = np.where(cell_unit[None] >= 0, harv[:, np.maximum(cell_unit, 0)],
                  harv[:, :1]).reshape(NC, g.ny, g.nx)
    ts1 = np.array([s['TSUM1'] for s in scal_all])[:, None, None] * np.ones((1, g.ny, g.nx))
    ts2 = np.array([s['TSUM2'] for s in scal_all])[:, None, None] * np.ones((1, g.ny, g.nx))
    carbon = 0.5 * (soilgrids(g, 'soc_0-5cm_mean', 0.1) + soilgrids(g, 'soc_5-15cm_mean', 0.1))
    ph = 0.5 * (soilgrids(g, 'phh2o_0-5cm_mean', 0.1) + soilgrids(g, 'phh2o_5-15cm_mean', 0.1))
    nf = len(FERT_DVS)
    fert = np.zeros((nf, NC, g.ny, g.nx))
    fp = np.zeros_like(fert)
    fk = np.zeros_like(fert)
    for ci, cls in enumerate(CLASSES):
        n, pp, k = FERT['pulses' if cls[1] == 'chickpea' else cls[1]]
        f = 1.0 if cls[5] == 'ir' else 0.5
        fert[0, ci], fert[1, ci] = 0.5 * n * f, 0.5 * n * f
        fp[0, ci], fk[0, ci] = pp * f, k * f
    os.makedirs(OUTW, exist_ok=True)
    with nc.Dataset(OUTW + '/crop_params_VICWOFOST_isc.nc', 'w') as ds:
        coords(ds, g)
        ds.createDimension('crop_class', NC)
        ds.createDimension('month', 12)
        ds.createDimension('fertilizer_times', nf)
        nc_var(ds, 'crop_class', 'i4', ('crop_class',), np.arange(NC))
        nc_var(ds, 'month', 'i4', ('month',), np.arange(1, 13))
        nc_var(ds, 'fertilizer_times', 'f8', ('fertilizer_times',), np.arange(nf))
        nc_var(ds, 'Ncrop', 'i4', ('lat', 'lon'), ncrop)
        nc_var(ds, 'Cc', 'f8', ('month', 'crop_class', 'lat', 'lon'),
               np.broadcast_to(cc_, (12,) + cc_.shape), 'fraction',
               comment='MIRCA-OS 2000 harvested-area shares scaled to Cv(2000) of the veg tile; constant over months')
        cvc = np.array(vegs, 'i4')[:, None, None] * np.ones((NC, g.ny, g.nx), 'i4')
        nc_var(ds, 'crop_veg_class', 'i4', ('crop_class', 'lat', 'lon'), cvc,
               comment='1-based veg class: 13 rainfed, 18 irrigated, 19 paddy')
        nc_var(ds, 'plant_day', 'f8', ('crop_class', 'lat', 'lon'), pl.astype(float), 'doy',
               comment='15th of the MIRCA-OS planting month (unit calendar)')
        nc_var(ds, 'harvest_day', 'f8', ('crop_class', 'lat', 'lon'), hv.astype(float), 'doy',
               comment='15th of the MIRCA-OS maturity month')
        nc_var(ds, 'TSUM1', 'f8', ('crop_class', 'lat', 'lon'), ts1, 'K d')
        nc_var(ds, 'TSUM2', 'f8', ('crop_class', 'lat', 'lon'), ts2, 'K d')
        nc_var(ds, 'Tfactor', 'f8', ('lat', 'lon'), np.zeros((g.ny, g.nx)), 'K')
        for nm_, a, why in (('DVS_point', np.broadcast_to(np.array(FERT_DVS)[:, None, None, None], fert.shape), ''),
                            ('N_amount', fert, 'kg ha-1'), ('P_amount', fp, 'kg ha-1'),
                            ('K_amount', fk, 'kg ha-1')):
            nc_var(ds, nm_, 'f8', ('fertilizer_times', 'crop_class', 'lat', 'lon'), a,
                   why or None, PLACEHOLDER='flagged default (indicative Indian fertiliser rates)')
        shp = (NC, g.ny, g.nx)
        for nm_, val in (('N_mins', 10.0), ('P_mins', 2.0), ('K_mins', 5.0),
                         ('N_recovery', 0.5), ('P_recovery', 0.5), ('K_recovery', 0.5),
                         ('mineralization_period', 60.0)):
            nc_var(ds, nm_, 'f8', ('crop_class', 'lat', 'lon'), np.full(shp, val),
                   PLACEHOLDER='flagged default (R1 values)')
        nc_var(ds, 'carbon', 'f8', ('lat', 'lon'), carbon, 'g kg-1',
               comment='SoilGrids 2.0 soc 0-15 cm mean (dg/kg / 10), 5 km, conservative regrid')
        nc_var(ds, 'pH', 'f8', ('lat', 'lon'), ph, '-',
               comment='SoilGrids 2.0 phh2o 0-15 cm mean (pH x10 / 10)')
        ds.title = 'VIC-WUR crop nc for ISC (open-data equivalent of crop_params_VICWOFOST_global.nc)'
    rows.append(stats_row('plant_day (active crop cells)', pl[:, g.active], np.ones(1, bool)))
    rows.append(stats_row('carbon g/kg', carbon, g.active))
    rows.append(stats_row('pH', ph, g.active))
    rows.append('Ncrop: min %d max %d mean %.2f (active)' % (
        ncrop[g.active].min(), ncrop[g.active].max(), ncrop[g.active].mean()))
    for ci, cls in enumerate(CLASSES):
        a = cc_[ci] * g.area
        rows.append('class %-16s veg %d  cells %6d  area %10.1f km2  plant doy mode %d harvest doy mode %d' % (
            cls[0], cls[6], int((cc_[ci] > 0).sum()), a.sum() / 1e6,
            np.bincount(pl[ci][cc_[ci] > 0].astype(int)).argmax() if (cc_[ci] > 0).any() else -1,
            np.bincount(hv[ci][cc_[ci] > 0].astype(int)).argmax() if (cc_[ci] > 0).any() else -1))
    for r in rows:
        log('CHECK ' + r)
    log('wofost done')


if __name__ == '__main__':
    main()
