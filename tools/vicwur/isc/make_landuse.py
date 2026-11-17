#!/usr/bin/env python
"""Irrigated / paddy / rainfed tile split, irrigation parameters, extended
params (20 veg classes) and the yearly Cv forcing from LUH2 + MIRCA-OS.

Veg classes of the output (1-indexed): 1-17 as the ISC params, 13 = rainfed
cropland, 14 = urban, 15 = crop/natural mosaic (emptied), 18 = irrigated
non-paddy crop (new), 19 = paddy (new), 20 = bare soil (was 18; the last
class, so Pbare = NVEGTYPES = 20 as the VIC-WUR code requires).

Cv(y) construction (see README.md):
  fixed   open water(1), snow/ice(16), barren(17), bare soil(20): ISC params
  urban   LUH2 urban(y)
  crop    LUH2 c3ann+c4ann+c3per+c4per+c3nfx (y), split into rainfed /
          irrigated non-paddy / paddy by MIRCA-OS maximum monthly growing
          areas (2000, 2005, 2010, linear in time)
  natural ISC classes 2-12 rescaled to the remaining fraction (grassland if
          the cell has none); old classes 13 and 15 are dropped.
"""
import glob
import os
import sys
import numpy as np
import netCDF4 as nc
import rasterio
from rasterio.windows import from_bounds
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa
from regrid import Regridder, fill_nearest, R_EARTH   # noqa

NVEG = 20
V_RF, V_URB, V_MOS, V_IRR, V_PAD, V_BARE = 13, 14, 15, 18, 19, 20
SNAP = (2000, 2005, 2010)
GW_PLACEHOLDER = 0.5
EFF_PLACEHOLDER = 0.5
OFFSET_PLACEHOLDER = 0
LUH_CROP = ('c3ann', 'c4ann', 'c3per', 'c4per', 'c3nfx')
EPS = 1e-6


def mirca_find(pat):
    """MIRCA-OS 5-arcminute tif by file name (the rar folder names in the
    HydroShare bag are swapped, so search the whole bag)."""
    hit = [p for p in glob.glob('%s/mirca/bag/*/data/contents/**/%s' % (WORK, pat),
                                recursive=True) if '30arc' not in p
           and '/5-arcminute/' in p]
    if not hit:
        raise FileNotFoundError(pat)
    return sorted(hit)[0]


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


def mirca_fields(g, log_rows):
    """ir, rf, rice_ir cell fractions on the ISC grid for each snapshot."""
    res = {}
    for y in SNAP:
        f = {}
        for nm, pat in (('ir', 'MIRCA-OS_%d_ir*.tif' % y),
                        ('rf', 'MIRCA-OS_%d_rf*.tif' % y),
                        ('rice', 'MIRCA-OS_Rice_%d_ir.tif' % y)):
            hit = [mirca_find(pat)]
            lat, lon, ha = read_tif(hit[0], g)
            rg = Regridder(lat, lon, g.lat, g.lon)
            ar = rg.src_area()
            dens = ha * 1e4 / ar                    # fraction of source cell
            m, cov = rg.mean(dens)
            m = np.nan_to_num(m)
            st, dt, pe = rg.mass_check(dens, m)
            log_rows.append('%s %d area: src %.6e ha  dst %.6e ha  err %.3e %%'
                            % (nm, y, st / 1e4, dt / 1e4, pe))
            f[nm] = m
        res[y] = f
    return res


def luh2_fields(g, years, log_rows):
    p = RAW + '/luh2/states4.nc'
    out = {}
    with nc.Dataset(p) as d:
        lat = np.array(d['lat'][:])
        lon = np.array(d['lon'][:])
        la = np.where((lat > g.lat[0] - 0.6) & (lat < g.lat[-1] + 0.6))[0]
        lo = np.where((lon > g.lon[0] - 0.6) & (lon < g.lon[-1] + 0.6))[0]
        sl = (slice(la.min(), la.max() + 1), slice(lo.min(), lo.max() + 1))
        slat, slon = lat[sl[0]][::-1], lon[sl[1]]
        t0 = int(round(float(d['time'][0])))
        rg = Regridder(slat, slon, g.lat, g.lon)
        for y in years:
            ti = y - 850 - t0
            crop = np.zeros((len(slat), len(slon)))
            ok = None
            for v in LUH_CROP + ('urban',):
                a = np.ma.filled(d[v][ti, sl[0], sl[1]].astype(float), np.nan)
                a[a > 1e10] = np.nan
                a = a[::-1]
                if v == 'urban':
                    urb = a
                else:
                    crop = crop + a
            res = {}
            for nm, a in (('crop', crop), ('urban', urb)):
                m, cov = rg.mean(a)
                if y in (years[0], years[-1]):
                    st, dt, pe = rg.mass_check(a, m)
                    log_rows.append('LUH2 %s %d area: src %.6e km2  dst %.6e '
                                    'km2  err %.3e %%' % (nm, y, st / 1e6,
                                                          dt / 1e6, pe))
                res[nm] = fill_nearest(m)
            out[y] = res
    return out


def split_weights(mir, y):
    """Interpolate MIRCA snapshots in time; return partition p_rf,p_irr,p_pad."""
    ys = np.array(SNAP, float)
    yy = min(max(y, SNAP[0]), SNAP[-1])
    k = int(np.clip(np.searchsorted(ys, yy, 'right') - 1, 0, len(ys) - 2))
    w = (yy - ys[k]) / (ys[k + 1] - ys[k])
    f = {n: (1 - w) * mir[SNAP[k]][n] + w * mir[SNAP[k + 1]][n]
         for n in ('ir', 'rf', 'rice')}
    pad = np.minimum(f['rice'], f['ir'])
    irr = np.maximum(f['ir'] - pad, 0)
    rf = f['rf']
    tot = irr + pad + rf
    with np.errstate(invalid='ignore', divide='ignore'):
        pr = np.where(tot > 0, rf / tot, 1.0)
        pi = np.where(tot > 0, irr / tot, 0.0)
        pp = np.where(tot > 0, pad / tot, 0.0)
    return pr, pi, pp


def build_cv(base, crop, urban, p):
    """base [18,ny,nx] ISC Cv; returns [20,ny,nx]."""
    pr, pi, pp = p
    fixed = base[0] + base[15] + base[16] + base[17]
    avail = np.clip(1.0 - fixed, 0, None)
    crop = np.clip(crop, 0, 1)
    urban = np.clip(urban, 0, 1)
    s = crop + urban
    sc = np.where(s > avail, avail / np.where(s > 0, s, 1), 1.0)
    crop, urban = crop * sc, urban * sc
    R = avail - crop - urban
    nat = base[1:12]                                  # classes 2..12
    n = nat.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(n > 0, nat / np.where(n > 0, n, 1), 0.0)
    share[9, :, :] = np.where(n > 0, share[9], 1.0)   # class 11 grassland
    cv = np.zeros((NVEG,) + base.shape[1:])
    cv[0], cv[15], cv[16] = base[0], base[15], base[16]
    cv[V_BARE - 1] = base[17]
    cv[1:12] = share * R
    cv[V_RF - 1] = crop * pr
    cv[V_IRR - 1] = crop * pi
    cv[V_PAD - 1] = crop * pp
    cv[V_URB - 1] = urban
    return cv


def fill_roots(dst, cv0):
    """Every vegetated tile (Cv > 0, not bare) gets root depths/fractions.
    The ISC params carry roots only where a class existed in the source
    cell; LUH2/MIRCA re-partitioning creates tiles of classes 11, 13, 14,
    18, 19 elsewhere.  Rule: own roots if present; else for classes 18/19
    the cell's rainfed crop (13) roots; else the domain default of the
    class = most common root profile of that class in the ISC params (the
    VIC veg library profile the ISC params were built from; VIC-WUR
    VegParam.md / Params.md root_depth, root_fract); class 18/19 default
    = class 13 default.  Fractions are normalised to sum to 1."""
    import collections
    rd = np.ma.filled(dst['root_depth'][:], 0.0)
    rf = np.ma.filled(dst['root_fract'][:], 0.0)

    def have(c):
        return rf[c - 1].sum(0) > 0

    def modal(c):
        h = have(c)
        if not h.any():
            return None
        idx = np.flatnonzero(h.ravel())
        idx = idx[::max(1, len(idx) // 20000)]
        z = rd[c - 1].reshape(3, -1)[:, idx].T
        f = rf[c - 1].reshape(3, -1)[:, idx].T
        cnt = collections.Counter(tuple(np.round(np.r_[a, b], 6))
                                  for a, b in zip(z, f))
        p = np.array(cnt.most_common(1)[0][0])
        return p[:3], p[3:]
    dflt = {}
    for c in range(1, V_BARE):
        dflt[c] = modal(c)
    for c in (V_IRR, V_PAD):
        dflt[c] = dflt[V_RF]
    for c in tuple(range(2, 15)) + (V_IRR, V_PAD):
        need = (cv0[c - 1] > 0) & ~have(c)
        if c in (V_IRR, V_PAD):
            need = cv0[c - 1] > 0              # always from the rainfed crop
            src = have(V_RF) & need
            rd[c - 1][:, src] = rd[V_RF - 1][:, src]
            rf[c - 1][:, src] = rf[V_RF - 1][:, src]
            need &= ~src
        if need.any():
            z, f = dflt[c] if dflt[c] is not None else dflt[11]
            rd[c - 1][:, need] = z[:, None]
            rf[c - 1][:, need] = f[:, None]
    for c in range(1, V_BARE):
        t = rf[c - 1].sum(0)
        ok = t > 0
        rf[c - 1] = np.where(ok, rf[c - 1] / np.where(ok, t, 1), rf[c - 1])
    dst['root_depth'][:] = rd
    dst['root_fract'][:] = rf


def write_params(g, base_cv, cv_first, cv_ever, path):
    """Copy ISC params with the veg_class dim extended to 20."""
    mp = np.array(list(range(17)) + [12, 12, 17])      # source class index
    with nc.Dataset(PARAMS) as src, nc.Dataset(path, 'w') as dst:
        for n_, L in src.dimensions.items():
            dst.createDimension(n_, NVEG if n_ == 'veg_class' else len(L))
        for n_, v in src.variables.items():
            if n_ == 'veg_descr':
                o = dst.createVariable(n_, str, ('veg_class',))
                names = [str(x) for x in v[:]]
                new = [names[i] for i in mp]
                new[V_IRR - 1], new[V_PAD - 1] = 'IrrigatedCrop', 'Paddy'
                new[V_RF - 1] = 'RainfedCropland'
                for k, s_ in enumerate(new):
                    o[k] = s_
                continue
            fill = getattr(v, '_FillValue', None)
            o = dst.createVariable(n_, v.dtype, v.dimensions,
                                   fill_value=fill, zlib=v.ndim >= 3)
            for a in v.ncattrs():
                if a != '_FillValue':
                    o.setncattr(a, v.getncattr(a))
            src.set_auto_maskandscale(False)
            a = v[:]
            if v.dimensions and v.dimensions[0] == 'veg_class':
                a = a[mp]
                if n_ == 'veg_class':
                    a = np.arange(1, NVEG + 1).astype(a.dtype)
            o[...] = a
        # Cv at the first year; classes that appear later get EPS so the
        # tile exists (the Cv forcing is normalised by VIC-WUR).
        cv0 = cv_first.copy()
        need = (cv_ever > 0) & (cv0 <= 0)
        cv0[need] = EPS
        cv0[:, ~g.active] = 0.0
        sm = cv0.sum(0)
        cv0 = np.where(sm > 0, cv0 / np.where(sm > 0, sm, 1), 0.0)
        nveg = (cv0[:V_BARE - 1] > 0).sum(0).astype('i4')
        dst['Cv'][:] = cv0
        dst['Nveg'][:] = nveg
        fill_roots(dst, cv0)
        b = np.full(NVEG, 0.2)
        b[[V_IRR - 1, V_PAD - 1]] = 0.1
        b[V_BARE - 1] = 0.0
        o = dst.createVariable('b_co2', 'f8', ('veg_class', 'lat', 'lon'))
        o.units = '-'
        o.PLACEHOLDER = ('R1 synthetic values (0.2 natural, 0.1 crop, 0 bare);'
                         ' the Li et al. 2019 table is not available')
        o[:] = np.broadcast_to(b[:, None, None], (NVEG, g.ny, g.nx))
        dst.title = 'ISC params extended to 20 veg classes for VIC-WUR'
        dst.veg_classes_added = ('18 IrrigatedCrop, 19 Paddy (roots of the cell rainfed '
                                 'crop, else class default), 20 Bare Soil (moved from 18)')
    return cv0


def write_irr(g, path):
    with nc.Dataset(path, 'w') as ds:
        coords(ds, g)
        ds.createDimension('irr_class', 2)
        for n_, dt, val in (('irr_class', 'i4', [0, 1]),
                            ('veg_class', 'i4', [V_IRR, V_PAD]),
                            ('paddy', 'i4', [0, 1])):
            v = ds.createVariable(n_, dt, ('irr_class',))
            v[:] = val
        for n_, val, dt, un, why in (
                ('groundwater_fraction', GW_PLACEHOLDER, 'f8', '-',
                 'GMIA v5 / CGWB groundwater share not downloaded'),
                ('irrigation_efficiency', EFF_PLACEHOLDER, 'f8', '-',
                 'AQUASTAT mirror (2016) holds no efficiency; FAO current '
                 'table needed'),
                ('irrigation_offset', OFFSET_PLACEHOLDER, 'i4', 'days',
                 'MIRCA-OS calendars are by statistical unit with no '
                 'geometry; needs the unit map')):
            v = ds.createVariable(n_, dt, ('lat', 'lon'))
            v.units = un
            v.PLACEHOLDER = 'constant %s: %s' % (val, why)
            v[:] = np.full((g.ny, g.nx), val)
        ds.title = 'VIC-WUR irrigation parameters, ISC (placeholders marked)'


def main():
    g = ISC()
    rows = []
    for d in ('forcing', 'params'):
        os.makedirs(OUT + '/' + d, exist_ok=True)
    mir = mirca_fields(g, rows)
    luh = luh2_fields(g, YEARS, rows)
    log('mirca + luh2 regridded')
    with nc.Dataset(PARAMS) as p:
        base = np.ma.filled(p['Cv'][:], 0.0).astype(float)
    base = np.where(g.active[None], base, 0.0)
    cvs = {}
    for y in YEARS:
        cvs[y] = build_cv(base, luh[y]['crop'], luh[y]['urban'],
                          split_weights(mir, y))
    ever = np.max([cvs[y] for y in YEARS], axis=0)
    cv0 = write_params(g, base, cvs[YEARS[0]], ever, OUT + '/params/params_wur.nc')
    for y in YEARS:
        cv = cvs[y]
        cv[:, ~g.active] = cv0[:, ~g.active]
        # regridding leaves -1e-16 residues: a negative Cv made the land-use
        # tile moisture NaN (ISC campaign); clip and renormalise
        cv = np.where(cv < 0, 0.0, cv)
        cv[:, g.active] /= cv[:, g.active].sum(0)
        cvs[y] = cv
        sm = cv.sum(0)
        bad = g.active & (np.abs(sm - 1) > 1e-9)
        rows.append('Cv %d: max |sum-1| over active cells %.2e' %
                    (y, np.abs(sm - 1)[g.active].max()))
        yearly_file('%s/forcing/cv.%d.nc' % (OUT, y), y, g, 'cv',
                    cv[None], 'fraction', [0], extra=('veg_class', NVEG),
                    attrs={'long_name': 'vegetation tile fraction'})
    write_irr(g, OUT + '/params/irr.nc')
    # area totals (km2) of the classes, first and last year
    for y in (YEARS[0], YEARS[-1]):
        for c, nm in ((V_RF, 'rainfed'), (V_IRR, 'irrigated'),
                      (V_PAD, 'paddy'), (V_URB, 'urban')):
            a = (cvs[y][c - 1] * g.area)[g.active].sum() / 1e6
            rows.append('class area %d %-9s %.0f km2' % (y, nm, a))
    for c, nm in ((V_RF, 'rainfed'), (V_IRR, 'irrigated'), (V_PAD, 'paddy')):
        rows.append(stats_row('Cv %s %d' % (nm, YEARS[-1]), cvs[YEARS[-1]][c - 1],
                              g.active, 'frac'))
    rows.append('Cv sum over classes %d min/max %.12f %.12f' % (
        YEARS[0], cvs[YEARS[0]].sum(0)[g.active].min(),
        cvs[YEARS[0]].sum(0)[g.active].max()))
    rows.append('active cells whose tile set grew after year 0 (EPS floor): %d'
                % int((((ever > 0) & (cvs[YEARS[0]] <= 0))[:, g.active]).any(0).sum()))
    for r in rows:
        log('CHECK ' + r)
    log('landuse done')


if __name__ == '__main__':
    main()
