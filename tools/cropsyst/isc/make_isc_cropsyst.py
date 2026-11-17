#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Gridded CropSyst inputs for the ISC domain (VIC-WUR crop tiles 13, 18, 19).

  make_isc_cropsyst.py [--years 2004 2005] [--cv-min 1e-4] [--out DIR]

Reads (<data root>):
  vicwur/isc/wofost/crop_params_VICWOFOST_isc.<year>.nc   MIRCA-OS 2000 crop shares per cell (Cc, harvested area scaled to
                                                           the tile's Cv), the veg tile of each crop class, the sowing day
                                                           of each class and cell (MIRCA-OS unit calendars, admin-1 mapped)
  vicwur/isc/forcing/cv.<year>.nc                          Cv per class and year (which tiles exist)
  vicwur/isc/params/params_wur.nc                          soil layers of the VIC params
  cropsyst_india/Database                                  the India crop set (13 crops, .crp/.mgt)
Writes (cropsyst_india/isc):
  isc_cropsyst.nc   per crop tile (veg 13 rainfed, 18 irrigated, 19 paddy) and cell: rot_code (0: no CropSyst crop),
                    crop (index into the attribute `crops`), wofost_class, sow_doy, share (area share of the chosen
                    crop within the tile); per cell and layer: thickness (m), max_moist (mm), air_entry (J kg-1),
                    campbell_b (the VIC soil's own retention, not Saxton)
  Database/         Crop and Management are links to the India set; Rotation/Name one 1-year rotation per
                    (crop, system, sowing day) used, Rotation/Code/<code>.CS_rotation links
  rotations_isc.csv code, crop, system, sow_doy, tile classes, cells
  README.md
"""
import argparse
import csv
import os
import sys

import numpy as np

A = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
ISC = A + '/vicwur/isc'
INDIA = A + '/cropsyst_india'

#: VIC-WUR veg tiles (1-based classes) of the crop tiles
CROP_TILES = (13, 18, 19)
#: WOFOST crop class (order of crop_class in the crop nc) -> (CropSyst crop file of the India set, system)
#: rice_rf (class 2) is rainfed kharif rice; the dry-season rice (boro) exists only irrigated (paddy)
CLASS_MAP = (
    ('rice_kharif', 'ir'), ('rice_boro', 'ir'), ('rice_kharif', 'rf'),
    ('wheat', 'rf'), ('wheat', 'ir'), ('maize', 'rf'), ('maize', 'ir'),
    ('sorghum', 'rf'), ('sorghum', 'ir'), ('pearl_millet', 'rf'), ('pearl_millet', 'ir'),
    ('chickpea_pulses', 'rf'), ('chickpea_pulses', 'ir'), ('sugarcane', 'rf'), ('sugarcane', 'ir'),
    ('cotton', 'rf'), ('cotton', 'ir'), ('groundnut', 'rf'), ('groundnut', 'ir'),
    ('potato', 'rf'), ('potato', 'ir'), ('mustard_rapeseed', 'rf'), ('mustard_rapeseed', 'ir'))
CROPS = tuple(sorted({c for c, _ in CLASS_MAP}))
#: first code of the per-(crop, system, sowing day) rotations (< 10000: codes above are non-irrigated, see
#: calendars.decompose_code; the India set uses 7001-7026)
CODE0 = 8000
#: irrigation technology (irrigation_lib.h codes): paddy flooded, other irrigated crops furrow (surface
#: irrigation dominates India: FAO AQUASTAT; a flagged typical choice, not a dataset)
IRR_TECH = {19: 71, 18: 73}
#: rabi sowing window (day of year, 15 Oct - 31 Dec) and the cold limit of the sowing season: a rabi crop (wheat,
#: chickpea, lentil, mustard/rapeseed, potato sown in the window) is not placed in a cell whose Nov-Dec mean daily
#: minimum temperature (ISC forcing, mean over the generator years) is below COLD_TMIN (Tibet/Himalaya: the crop was
#: sown into frozen ground, emerged in June and failed, isc_crop_fail/run.log 2026-10-26). No crop is substituted and no
#: area is moved: the tile keeps VIC's own vegetation and has no CropSyst crop member. The input has one sowing day per
#: crop class and cell, so there is no spring-sown variant to use instead (none is invented).
RABI_SOW = (288, 366)
COLD_TMIN = 0.0
COLD_MONTHS = (11, 12)


def cold_mask(years, forcing=A + '/forcing/vic5_daily/isc_daily.%d.nc', limit=COLD_TMIN):
    """[lat, lon] bool: Nov-Dec mean tasmin (degC) of the years below `limit` (NaN cells: False)."""
    import netCDF4 as nc
    tot = cnt = 0.0
    for y in years:
        d = nc.Dataset(forcing % y)
        t = nc.num2date(d['time'][:], d['time'].units, only_use_cftime_datetimes=False)
        m = np.array([x.month in COLD_MONTHS for x in t])
        x = np.ma.filled(d['tasmin'][np.where(m)[0]], np.nan).astype(np.float64)
        tot = tot + np.nansum(x, 0)
        cnt = cnt + np.isfinite(x).sum(0)
        d.close()
    with np.errstate(invalid='ignore', divide='ignore'):
        return (tot / cnt) < limit


#: thermal feasibility (isc-chickpea): the 1-year rotation must be able to bring the crop to physiological maturity.
#: Growing degree days (crop file [phenology] base_temp / cutoff_temp, daily mean of tasmax and tasmin, mean of the
#: generator years, SEASON_DAYS from the sowing day, wrapping the year) below the crop's [phenology] maturity: no crop.
#: S9_nosub: maize sown 15 Jun, GDD to 31 Dec < 1600: 100 % of 8200 tiles failed (never matured), > 1800: 0 %.
#: Rabi sowings (is_rabi) are not tested: they overwinter and mature in spring (the cold rule covers their frozen
#: cells; wheat/potato/mustard tiles below the GDD limit all yielded). Sugarcane (12-month crop) is not tested.
NO_THERMAL_TEST = ('sugarcane',)
#: length of the growing window after sowing; the diagnosis window of S9_nosub (sowing 15 Jun to 31 Dec = 200 d)
SEASON_DAYS = 200


def crop_thermal(crop, db=INDIA + '/Database/Crop/Name'):
    """(base_temp, cutoff_temp, maturity GDD) of the crop file's [phenology] block."""
    sec, v = None, {}
    for ln in open('%s/%s.crp' % (db, crop)):
        ln = ln.strip()
        if ln.startswith('['):
            sec = ln
        elif sec == '[phenology]' and '=' in ln:
            k, x = ln.split('=', 1)
            if k in ('base_temp', 'cutoff_temp', 'maturity'):
                v[k] = float(x.split()[0])
    return v['base_temp'], v['cutoff_temp'], v['maturity']


def tmean_climatology(years, forcing=A + '/forcing/vic5_daily/isc_daily.%d.nc'):
    """[365, lat, lon] daily mean of (tasmax + tasmin) / 2 (degC) over the years (day 366 dropped)."""
    import netCDF4 as nc
    tot = 0.0
    for y in years:
        d = nc.Dataset(forcing % y)
        hi = np.ma.filled(d['tasmax'][:365], np.nan).astype(np.float32)
        lo = np.ma.filled(d['tasmin'][:365], np.nan).astype(np.float32)
        d.close()
        tot = tot + 0.5 * (hi + lo)
    return tot / len(years)


def thermal_infeasible(tm, base, cutoff, maturity, sow):
    """[lat, lon] bool: GDD over the SEASON_DAYS from sowing day `sow` (1-based, wraps) are below `maturity` (NaN: False)."""
    t = np.roll(tm, -(int(sow) - 1), axis=0)[:SEASON_DAYS]
    g = np.clip(np.minimum(t, cutoff) - base, 0.0, None).sum(0)
    with np.errstate(invalid='ignore'):
        return g < maturity


#: MIRCA-OS lumps kharif and rabi pulses in one class. Chickpea (the only pulse of the 13-crop India set; no pigeonpea,
#: mungbean or urad) is a rabi crop of South Asia: a pulses sowing outside the rabi window (15 Jun in the unit calendars)
#: is a kharif pulse, not chickpea, and gets no CropSyst crop (no substitution).
RABI_ONLY = ('chickpea_pulses',)


def is_rabi(sow):
    """Sowing day in the rabi window (NaN: False)."""
    return np.isfinite(sow) & (sow >= RABI_SOW[0]) & (sow <= RABI_SOW[1])


def choose(cc, veg, plant, cv_ok, cold=None, infeasible=None, kharif_pulse=True):
    """Dominant crop of every tile class per cell.

    cc [n_class, ...] crop share (Cc), veg [n_class, ...] the 1-based veg tile of each class, plant [n_class, ...] sowing
    day, cv_ok {tile: bool [...]} the tile exists in the run. Returns dict tile -> (wofost_class, sow_doy, share),
    wofost_class -1 where the tile has no CropSyst crop (tile missing or no modelled crop). cold [lat, lon] bool: in
    a tile whose dominant crop is rabi (is_rabi(sow), see RABI_SOW) gets no CropSyst crop in these cells.
    infeasible(class, sow_day) -> [lat, lon] bool (thermal_infeasible): the crop cannot mature there, no CropSyst crop."""
    out = {}
    for t in CROP_TILES:
        sel = np.where((veg == t) & np.isfinite(cc) & (cc > 0), cc, 0.0)
        tot = sel.sum(0)
        k = sel.argmax(0)
        ok = (tot > 0) & cv_ok[t]
        top = np.take_along_axis(sel, k[None], 0)[0]
        sow = np.take_along_axis(plant, k[None], 0)[0]
        crop_of = np.array([m[0] for m in CLASS_MAP])[np.clip(k, 0, len(CLASS_MAP) - 1)]
        if kharif_pulse:
            ok &= ~(np.isin(crop_of, RABI_ONLY) & ~is_rabi(sow))  # kharif pulses: not chickpea, no crop
        if cold is not None:
            ok &= ~(is_rabi(sow) & cold)      # no substitution: the tile keeps VIC's vegetation, no crop member
        if infeasible is not None:
            for kk, ss in set(zip(k[ok].tolist(), np.rint(sow[ok]).astype(int).tolist())):
                if is_rabi(ss):
                    continue
                hit = (k == kk) & (np.rint(np.nan_to_num(sow)) == ss)
                ok &= ~(hit & infeasible(kk, ss))
        out[t] = (np.where(ok, k, -1).astype(np.int16),
                  np.where(ok & np.isfinite(sow), np.rint(sow), 0).astype(np.int16),
                  np.where(ok, top / np.where(tot > 0, tot, 1.0), 0.0).astype(np.float32))
    return out


def removed_area(cc, veg, plant, cv_ok, cold, infeasible, lat, lon, out):
    """Write removed_area_ha.csv: per crop and rule (kharif pulse, rabi cold, thermal; applied in this order) the cells and
    the harvested area (Cc of the dominant crop x cell area, ha) of the tiles that lose their CropSyst crop."""
    dl = np.deg2rad(abs(float(lat[1] - lat[0])))
    dn = np.deg2rad(abs(float(lon[1] - lon[0])))
    ha = (6371000.0 ** 2 * dl * dn * np.cos(np.deg2rad(lat)))[:, None] / 1e4 * np.ones((1, len(lon)))
    steps = [('none', dict(cold=None, infeasible=None, kharif_pulse=False)),
             ('kharif_pulse', dict(cold=None, infeasible=None)),
             ('rabi_cold', dict(cold=cold, infeasible=None)),
             ('thermal', dict(cold=cold, infeasible=infeasible))]
    sels = [choose(cc, veg, plant, cv_ok, **kw) for _, kw in steps]
    rows = []
    for i in range(1, len(steps)):
        for t in CROP_TILES:
            k0, k1 = sels[i - 1][t][0], sels[i][t][0]
            lost = (k0 >= 0) & (k1 < 0)
            for kk in sorted(set(k0[lost].tolist())):
                m = lost & (k0 == kk)
                area = np.where(m, np.take(cc, kk, 0) * ha, 0.0).sum()
                rows.append((CLASS_MAP[kk][0], CLASS_MAP[kk][1], steps[i][0], t, int(m.sum()), round(float(area), 1)))
    os.makedirs(out, exist_ok=True)
    with open(out + '/removed_area_ha.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['crop', 'system', 'rule', 'tile_class', 'cells', 'area_ha'])
        w.writerows(sorted(rows))
    return rows


def rotation_table(sel):
    """sel: {tile: (cls, sow, share)}. Returns (codes {(crop, sys, sow): code}, rot_code {tile: int32 array})."""
    keys = set()
    for t, (k, s, _) in sel.items():
        for kk, ss in set(zip(k[k >= 0].tolist(), s[k >= 0].tolist())):
            keys.add((CLASS_MAP[kk][0], CLASS_MAP[kk][1], int(ss)))
    codes = {key: CODE0 + i for i, key in enumerate(sorted(keys))}
    if CODE0 + len(codes) >= 10000:
        raise ValueError('more than %d rotations' % (10000 - CODE0))
    rc = {}
    for t, (k, s, _) in sel.items():
        r = np.zeros(k.shape, np.int32)
        for (c, y, d), code in codes.items():
            for kk in [i for i, m in enumerate(CLASS_MAP) if m == (c, y)]:
                r[(k == kk) & (s == d)] = code
        rc[t] = r
    return codes, rc


def soil(thickness, bulk, sdens, wcr_frac, wpwp_frac):
    """The VIC soil's own wilting point and field capacity as the engine's retention curve.

    thickness (m), max_moist (mm, as vicwur_setup.soil_statics: depth (1 - bulk/soil density) 1000). VIC's Wpwp_FRACT
    (the wilting point) and Wcr_FRACT (0.7 x field capacity above the wilting point: soil_statics' wfc = wpwp +
    (wcr - wpwp)/0.7) are fractions of max_moist. The engine's matric potential is psi = AE s^-b (s = moisture / max_moist,
    AE negative J/kg); b and AE are fitted so that psi(s_wp) = -1500 and psi(s_fc) = -33 J/kg (the potentials the engine's
    uptake and VIC-CropSyst take for the wilting point and field capacity), i.e. the crop's water stress sets in at
    VIC's own Wpwp and Wfc. (Reading b from VIC's expt = 3 + 2 b and AE from the bubble pressure gave b up to 13.5 and a
    wilting point at 57 % of saturation: nearly every crop died of drought.) b is limited to 2.5-12."""
    mm = thickness * (1.0 - bulk / sdens) * 1000.0
    wp = np.clip(wpwp_frac, 1e-3, 0.95)
    fc = np.clip(wp + (wcr_frac - wp) / 0.7, wp * 1.05, 0.98)
    b = np.clip(np.log(1500.0 / 33.0) / np.log(fc / wp), 2.5, 12.0)
    return dict(thickness=thickness, max_moist=mm, air_entry=-33.0 * fc ** b, campbell_b=b)


def rot_text(crop, system, sow, code):
    return ('[rotation]\ndescription=%s %s, India, sowing day %d (isc-cropsyst, MIRCA-OS 2000 calendar)\nyears=1\n'
            'sowing:count=1\n[version]\nmajor=4\n\n[sowing:1]\nID=%s_%s_%d\nevent_synchronization=relative_date\n'
            'event_date=%04d (0/0/%d (relative date))\nenabled=true\ndescription=%s_%s\n'
            'crop=../../../../Database/Crop/Name/%s.crp\nmanagement=../../../../Database/Management/Name/%s_%s.mgt\n'
            % (crop, system, sow, crop, system, sow, sow, sow, crop, system, crop, crop, system))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2004, 2005, 2006])
    ap.add_argument('--cv-min', type=float, default=1e-4)
    ap.add_argument('--out', default=INDIA + '/isc')
    a = ap.parse_args()
    import netCDF4 as nc
    year = a.years[-1]
    d = nc.Dataset('%s/wofost/crop_params_VICWOFOST_isc.%d.nc' % (ISC, year))
    f = lambda v: np.ma.filled(v[:], np.nan).astype(np.float64)
    cc = f(d['Cc'])[0]                                   # [class, lat, lon], constant over the months
    veg = np.ma.filled(d['crop_veg_class'][:], -1)
    plant = f(d['plant_day'])
    lat, lon = d['lat'][:], d['lon'][:]
    p = nc.Dataset(ISC + '/params/params_wur.nc')
    act = np.ma.filled(p['mask'][:], 0) > 0
    cv_ok = {t: np.zeros(act.shape, bool) for t in CROP_TILES}
    for y in a.years:
        c = nc.Dataset('%s/forcing/cv.%d.nc' % (ISC, y))['cv']
        for t in CROP_TILES:
            cv_ok[t] |= np.ma.filled(c[0, t - 1], 0.0) >= a.cv_min
    for t in CROP_TILES:
        cv_ok[t] &= act
    cold = cold_mask(a.years)
    tm = tmean_climatology(a.years)
    cache = {}

    def infeasible(kk, ss):
        crop = CLASS_MAP[kk][0]
        if crop in NO_THERMAL_TEST:
            return np.zeros(tm.shape[1:], bool)
        if (crop, ss) not in cache:
            cache[(crop, ss)] = thermal_infeasible(tm, *crop_thermal(crop), ss)
        return cache[(crop, ss)]
    sel = choose(cc, veg, plant, cv_ok, cold, infeasible)
    removed_area(cc, veg, plant, cv_ok, cold, infeasible, lat, lon, a.out)
    codes, rc = rotation_table(sel)
    g = lambda k: np.ma.filled(p[k][:], np.nan).astype(np.float64)
    s = soil(g('depth'), g('bulk_density'), g('soil_density'), g('Wcr_FRACT'), g('Wpwp_FRACT'))
    os.makedirs(a.out, exist_ok=True)
    o = nc.Dataset(a.out + '/isc_cropsyst.nc', 'w')
    o.createDimension('tile', len(CROP_TILES))
    o.createDimension('layer', 3)
    o.createDimension('lat', len(lat))
    o.createDimension('lon', len(lon))
    o.setncattr('title', 'CropSyst crop tiles of the ISC VIC-WUR domain (isc-cropsyst)')
    o.setncattr('crops', ' '.join(CROPS))
    o.setncattr('veg_class_of_tile', 'tile 0..2 = VIC-WUR classes 13 (rainfed), 18 (irrigated), 19 (paddy), 1-based')
    o.setncattr('irrigation_technology_code', 'class 18: %d (furrow), class 19: %d (flood), irrigation_lib.h' %
                (IRR_TECH[18], IRR_TECH[19]))
    o.setncattr('rabi_cold_rule', 'tiles whose dominant crop is rabi (sowing day %d-%d) get no crop in cells with Nov-Dec mean tmin < %g degC (%d cells)'
                % (RABI_SOW + (COLD_TMIN, int(np.nansum(cold)))))
    o.setncattr('kharif_pulse_rule', 'pulses class sown outside the rabi window (kharif pulses, not chickpea): no crop')
    o.setncattr('thermal_rule', 'no crop where growing degree days (crop file base/cutoff, %d d from sowing, mean of the years) < crop maturity; sugarcane not tested' % SEASON_DAYS)
    o.setncattr('sources', 'MIRCA-OS 2000 (Cc, plant_day via crop_params_VICWOFOST_isc.%d.nc), cv.%s.nc, params_wur.nc'
                % (year, ','.join(map(str, a.years))))
    v = o.createVariable('lat', 'f8', ('lat',)); v[:] = lat
    v = o.createVariable('lon', 'f8', ('lon',)); v[:] = lon
    sh = ('tile', 'lat', 'lon')
    def var(n, t, dims, fill, desc):
        x = o.createVariable(n, t, dims, zlib=True, fill_value=fill)
        x.setncattr('long_name', desc)
        return x
    rot = var('rot_code', 'i4', sh, 0, 'CropSyst rotation code (Rotation/Code/<code>.CS_rotation); 0: no CropSyst crop')
    wc = var('wofost_class', 'i2', sh, -1, 'crop class of the VIC-WUR crop nc that was chosen; -1 none')
    cr = var('crop', 'i2', sh, -1, 'index into the attribute crops; -1 none')
    sw = var('sow_doy', 'i2', sh, 0, 'sowing day of the year')
    sr = var('share', 'f4', sh, 0.0, 'area share of the chosen crop within the tile (the other crops of the tile are not modelled)')
    for i, t in enumerate(CROP_TILES):
        k, sd, sh_ = sel[t]
        rot[i] = rc[t]
        wc[i] = k
        cr[i] = np.array([-1] + [CROPS.index(c) for c, _ in CLASS_MAP], np.int16)[k + 1]
        sw[i] = sd
        sr[i] = sh_
    for n, desc in (('thickness', 'm'), ('max_moist', 'mm'), ('air_entry', 'J kg-1'), ('campbell_b', '1')):
        x = o.createVariable(n, 'f8', ('layer', 'lat', 'lon'), zlib=True, fill_value=np.nan)
        x.setncattr('units', desc)
        x[:] = np.where(act[None], s[n], np.nan)
    o.close()
    # rotation database
    rdir = a.out + '/Database/Rotation'
    os.makedirs(rdir + '/Name', exist_ok=True)
    os.makedirs(rdir + '/Code', exist_ok=True)
    for sub in ('Crop', 'Management'):
        ln = a.out + '/Database/' + sub
        if not os.path.islink(ln):
            os.symlink('../../Database/%s' % sub, ln)
    rows = []
    for (c, y, d_), code in sorted(codes.items(), key=lambda kv: kv[1]):
        nm = '%s_%s_%d' % (c, y, d_)
        open('%s/Name/%s.rot' % (rdir, nm), 'w').write(rot_text(c, y, d_, code))
        ln = '%s/Code/%d.CS_rotation' % (rdir, code)
        if os.path.lexists(ln):
            os.remove(ln)
        os.symlink('../Name/%s.rot' % nm, ln)
        ncell = {t: int((rc[t] == code).sum()) for t in CROP_TILES}
        rows.append([code, c, y, d_] + [ncell[t] for t in CROP_TILES])
    with open(a.out + '/rotations_isc.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['code', 'crop', 'system', 'sow_doy', 'cells_tile13', 'cells_tile18', 'cells_tile19'])
        w.writerows(rows)
    tot = {t: int((rc[t] > 0).sum()) for t in CROP_TILES}
    print('rotations %d; cells with a CropSyst crop per tile %s' % (len(codes), tot))


if __name__ == '__main__':
    sys.exit(main())
