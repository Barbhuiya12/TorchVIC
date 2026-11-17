#!/usr/bin/env python
"""ISC irrigation season (crop calendar) as monthly Cv, and irrigation efficiency.

VIC-WUR has no calendar input of its own for irrigation.  The irrigation
plugin only irrigates a tile with Cv > 0 (VIC-WUR 3.0
plugins/irrigation/src/irr_run.c, `if (veg_fract > 0)`), and it starts a new
season when the tile's Cv changes (irr_run.c:125-138, offset counter).
The input format for a time-varying season is the monthly Cv forcing
(CV[time(month), veg_class, lat, lon], "MIRCA monthly"), the same Cv file
as make_landuse.py writes with time = 1 (YEAR); here time = 12 (MONTH).

Season per cell and tile (MIRCA-OS v0.1 Monthly Growing Area Grids, 5 arcmin,
irrigated, 2000/2005/2010, linear in time), two options (--season):

  ggcmi (default)  every MIRCA-OS crop file keeps its irrigated annual area
           A_c = max_m a_c(m) and its split between sub-crops, but is grown in
           the months of the GGCMI Phase 3 irrigated calendar (Jaegermeyr et
           al. 2021, Nature Food 2:873, doi 10.1038/s43016-021-00400-y; data
           Zenodo 10.5281/zenodo.5062513, 0.5 deg, planting_day -> maturity_day,
           <data root>/raw/ggcmi_calendar): a_c(m) = A_c x (share of month m inside
           the season).  MIRCA crop -> GGCMI crop is CROP_MAP below.  The GGCMI
           value is taken by nearest 0.5 deg cell for each 5 arcmin cell, then
           everything is regridded conservatively to ISC as before.  Where GGCMI
           has no value for the crop in the cell (or the crop has no GGCMI
           calendar: perennials, fodder, others), the MIRCA-OS months stay.
  mirca    the MIRCA-OS growing months as they are (previous behaviour).

  G(m)     sum over all irrigated crops of the monthly growing area
  paddy    Rice_1 + Rice_2 + Rice_3 irrigated (capped by G)
  non-paddy G - paddy
  The irrigated land L = Cv_irr + Cv_paddy of the year grows the share
  G(m)/max_m G(m) of itself, split between the two tiles by their current
  growing area (tile_months); the area not growing goes to the bare soil tile
  (class 20, fallow).  Cells with no irrigated crop keep f = 1.
Irrigation efficiency (--eff): Jaegermeyr et al. 2015 (HESS 19, 3073) Table 5,
South Asia surface field application efficiency 0.48 (Rohwer et al. 2007:
0.48), times the open-canal conveyance efficiency 0.75 (Table 1, loam) for the
surface-water share only; the groundwater share (GMIA v5) is pumped at the
field (conveyance 1).
"""
import glob
import os
import shutil
import sys
import numpy as np
import netCDF4 as nc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa
from regrid import Regridder     # noqa

SNAP = (2000, 2005, 2010)
V_IRR, V_PAD, V_BARE = 18, 19, 20
EPS = 1e-6
EF_SURF_SA = 0.48        # Jaegermeyr 2015 Table 5, South Asia, surface
EC_CANAL = 0.75          # Jaegermeyr 2015 Table 1, open canal, loam
CALDIR = OUT + '/forcing_cal'
GGCMI = ROOT + '/raw/ggcmi_calendar'
GGCMI_FILE = '%s/%s_ir_ggcmi_crop_calendar_phase3_v1.01.nc4'
SEASONS = ('ggcmi', 'mirca')

# MIRCA-OS crop (file name token) -> GGCMI Phase 3 irrigated crops, first one
# with a value in the cell is used; () = no GGCMI calendar, MIRCA months stay.
#   Rice_1/2/3: MIRCA's equal-thirds split (Punjab, Haryana, UP: Apr-Jul, Aug-Nov,
#   Dec-Mar) is an artefact, so rice_seasonal() sums the three sub-crops into one
#   irrigated rice area per cell and gives it all to ri1 (main season); a share goes
#   to ri2 (rabi/boro) only where GGCMI has a valid ri2 AND the cell lies in a state of
#   RICE2_STATES (documented second-season rice); that share is the part of the MIRCA
#   rice area in sub-crops whose MIRCA months overlap ri2 more than ri1.  No cap from
#   Indian DES/ICRISAT state statistics (not available offline): the MIRCA share is used.
#   Wheat: Wheat_1 and Wheat_2 both go entirely to one season, swh (wwh where no swh).
#   swh spring wheat (the Indian wheat), wwh winter wheat where swh is missing
#   pea pulses (GGCMI pea; bea = beans, not used)
CROP_MAP = {
    'Barley': ('bar',), 'Cassava': ('cas',), 'Cocoa': (), 'Coffee': (),
    'Cotton': ('cot',), 'Fodder': (), 'Groundnuts': ('nut',),
    'Maize': ('mai',), 'Millet': ('mil',), 'Oil_palm': (),
    'Others_annual_1': (), 'Others_annual_2': (), 'Others_annual_3': (),
    'Others_annual_4': (), 'Others_perennial': (),
    'Potatoes': ('pot',), 'Pulses': ('pea',), 'Rapeseed': ('rap',),
    'Rice_1': ('ri1', 'ri2'), 'Rice_2': ('ri1', 'ri2'), 'Rice_3': ('ri1', 'ri2'),
    'Rye': ('rye',), 'Sorghum': ('sor',), 'Soybeans': ('soy',),
    'Sugar_beet': ('sgb',), 'Sugar_cane': ('sgc',), 'Sunflower': ('sun',),
    'Wheat_1': ('swh', 'wwh'), 'Wheat_2': ('swh', 'wwh'),
}
RICE2_STATES = ('West Bengal', 'Odisha', 'Andhra Pradesh', 'Telangana', 'Tamil Nadu',
                'Assam', 'Karnataka', 'Kerala', 'Bihar')      # Natural Earth admin-1, India
_DM = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def mdir(y):
    h = glob.glob('%s/mirca/bag/*/data/contents/*/x/Monthly Growing Area Grids/%d'
                  % (WORK, y))
    return h[0]


def read_month(path, g):
    with nc.Dataset(path) as d:
        lat = np.array(d['latitude'][:])
        lon = np.array(d['longitude'][:])
        la = np.where((lat > g.lat[0] - 0.5) & (lat < g.lat[-1] + 0.5))[0]
        lo = np.where((lon > g.lon[0] - 0.5) & (lon < g.lon[-1] + 0.5))[0]
        a = np.ma.filled(d['harvested_area'][:, la.min():la.max() + 1,
                                              lo.min():lo.max() + 1], 0.0)
        a = np.nan_to_num(a.astype(float))
        mon = np.array(d['month'][:]).astype(int)
    # the month coordinate is unordered (1,10,11,12,2,...) and some files
    # hold only the months with area: place each record at its month
    full = np.zeros((12,) + a.shape[1:])
    full[mon - 1] = a
    a = full
    a[a < 0] = 0
    slat, slon = lat[la], lon[lo]
    if slat[0] > slat[-1]:
        a, slat = a[:, ::-1], slat[::-1]
    return slat, slon, a


def season_months(plant, mature):
    """Share of each month (12, ...) inside planting -> maturity (days of the
    year, 365-day calendar, season wraps the year end, both days included).
    NaN, fill and out-of-range days give NaN (no calendar)."""
    plant = np.asarray(plant, float)
    mature = np.asarray(mature, float)
    ok = (np.isfinite(plant) & np.isfinite(mature) & (plant >= 1) & (plant <= 365)
          & (mature >= 1) & (mature <= 365))
    p = np.where(ok, plant, 1.0)
    ln = np.mod(np.where(ok, mature, 1.0) - p, 365.0)
    ln = np.where(ln == 0, 364.0, ln)                # same day: a full year
    out = np.empty((12,) + p.shape)
    d0 = 0
    for m in range(12):
        days = np.arange(d0 + 1, d0 + _DM[m] + 1, dtype=float)
        d0 += _DM[m]
        inside = np.mod(days.reshape((-1,) + (1,) * p.ndim) - p, 365.0) <= ln
        out[m] = inside.mean(0)
    return np.where(ok, out, np.nan)


def ggcmi_months(code, lat, lon):
    """GGCMI irrigated season shares (12, len(lat), len(lon)) of crop `code` on
    the (ascending) 5 arcmin centres lat/lon, nearest 0.5 deg cell; NaN = none."""
    with nc.Dataset(GGCMI_FILE % (GGCMI, code)) as d:
        glat = np.array(d['lat'][:], float)
        glon = np.array(d['lon'][:], float)
        p = np.ma.filled(d['planting_day'][:].astype(float), np.nan)
        m = np.ma.filled(d['maturity_day'][:].astype(float), np.nan)
    sh = season_months(p, m)
    i = np.abs(glat[None, :] - np.asarray(lat)[:, None]).argmin(1)
    j = np.abs(glon[None, :] - np.asarray(lon)[:, None]).argmin(1)
    return sh[:, i][:, :, j]


def crop_name(path):
    return os.path.basename(path).split('_', 1)[1].rsplit('_', 2)[0]


def seasonal_area(a, shares):
    """MIRCA monthly growing area a (12, ...) -> annual area x GGCMI month
    shares; where the first share is NaN the next one is tried, then MIRCA.
    Annual area = max over the months (a sub-crop file holds one season)."""
    area = a.max(0)
    new = a.copy()
    todo = np.ones(a.shape[1:], bool)
    for sh in shares:
        use = todo & np.isfinite(sh[0])
        new = np.where(use[None], area[None] * np.nan_to_num(sh), new)
        todo &= ~use
    return new


def rice2_mask(lat, lon):
    """Cells (ascending lat, lon centres) inside a RICE2_STATES state of India."""
    from matplotlib.path import Path
    from report_calendar import read_shp
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    m = np.zeros(len(pts), bool)
    for nm, a3, rings in read_shp(WORK + '/ne_admin1/ne_10m_admin_1_states_provinces'):
        if a3 == 'IND' and nm in RICE2_STATES:
            ins = np.zeros(len(pts), bool)
            for ring in rings:
                ins ^= Path(ring).contains_points(pts)
            m |= ins
    return m.reshape(LAT.shape)


def rice_seasonal(subs, ri1, ri2, second):
    """All MIRCA rice sub-crop monthly areas (list of (12, ...)) -> one rice area
    in ri1, plus the off-season share in ri2 where ri2 is valid and `second`."""
    areas = [a.max(0) for a in subs]
    tot = sum(areas)
    v1, v2 = np.isfinite(ri1[0]), np.isfinite(ri2[0]) & second
    s1, s2 = np.nan_to_num(ri1), np.nan_to_num(ri2)
    off = np.zeros_like(tot)
    for a, ar in zip(subs, areas):
        pat = a / np.where(ar > 0, ar, 1.0)
        o2, o1 = (pat * s2).sum(0), (pat * s1).sum(0)
        off += np.where(o2 > o1, ar, 0.0)
    share = np.where(v1 & v2, off / np.where(tot > 0, tot, 1.0), 0.0)
    main = tot[None] * ((1 - share)[None] * s1 + share[None] * s2)
    only2 = ~v1 & np.isfinite(ri2[0])
    out = np.where(v1[None], main, np.where(only2[None], tot[None] * s2, sum(subs)))
    return out


def snapshot(y, g, season='ggcmi'):
    files = sorted(glob.glob(mdir(y) + '/MIRCA-OS_*_%d_ir.nc' % y))
    tot = rice = None
    rice_raw = []
    for p in files:
        lat, lon, a = read_month(p, g)
        if season == 'ggcmi':
            if crop_name(p).startswith('Rice_'):
                rice_raw.append(a)
                continue
            codes = CROP_MAP[crop_name(p)]
            a = seasonal_area(a, [ggcmi_months(c, lat, lon) for c in codes])
        tot = a if tot is None else tot + a
        if os.path.basename(p).startswith('MIRCA-OS_Rice_'):
            rice = a if rice is None else rice + a
    if rice_raw:
        a = rice_seasonal(rice_raw, ggcmi_months('ri1', lat, lon),
                          ggcmi_months('ri2', lat, lon), rice2_mask(lat, lon))
        tot = a if tot is None else tot + a
        rice = a
    rg = Regridder(lat, lon, g.lat, g.lon)
    ar = rg.src_area()
    pad = np.minimum(rice, tot)
    npd = np.maximum(tot - pad, 0)
    out = {}
    for nm, a in (('np', npd), ('pad', pad)):
        out[nm] = np.stack([np.nan_to_num(rg.mean(a[k] * 1e4 / ar)[0])
                            for k in range(12)])
    tot_m = out['np'] + out['pad']
    log('calendar snapshot %d (%s): %d irrigated crop files; mean growing share by month %s'
        % (y, season, len(files), np.round((tot_m / np.maximum(tot_m.max(0), 1e-30))[:, g.active].mean(1), 2).tolist()))
    return out


def interp(snaps, y):
    ys = np.array(SNAP, float)
    yy = min(max(y, SNAP[0]), SNAP[-1])
    k = int(np.clip(np.searchsorted(ys, yy, 'right') - 1, 0, 1))
    w = (yy - ys[k]) / (ys[k + 1] - ys[k])
    return {n: (1 - w) * snaps[SNAP[k]][n] + w * snaps[SNAP[k + 1]][n]
            for n in ('np', 'pad')}


def tile_months(cv_irr, cv_pad, a_np, a_pad):
    """Monthly irrigated / paddy tile fractions of one year.

    cv_irr, cv_pad: yearly tile fractions (lat, lon); a_np, a_pad: monthly
    growing area as cell fraction (12, lat, lon).  Returns (irr, pad, freed).
    The land L = cv_irr + cv_pad grows the share g(m) of the peak total
    growing area; the tiles split it by their current growing area, a tile
    with no yearly Cv (no tile in the params) gives its share to the other.
    """
    land = cv_irr + cv_pad
    tot = a_np + a_pad
    mx = tot.max(0)
    grow = np.where(mx > 0, tot / np.where(mx > 0, mx, 1.0), 1.0)
    wp = np.where(tot > 0, a_pad / np.where(tot > 0, tot, 1.0), 0.0)
    yp = np.where(land > 0, cv_pad / np.where(land > 0, land, 1.0), 0.0)
    wp = np.where(mx > 0, wp, yp[None])             # no irrigated crop: yearly split
    wp = np.where(cv_pad[None] > 0, wp, 0.0)         # tile must exist
    wp = np.where(cv_irr[None] > 0, wp, 1.0 * (cv_pad[None] > 0))
    irr = land[None] * grow * (1 - wp)
    pad = land[None] * grow * wp
    freed = land[None] - irr - pad
    return irr, pad, freed


def calendar(g, season='ggcmi'):
    os.makedirs(CALDIR, exist_ok=True)
    snaps = {y: snapshot(y, g, season) for y in SNAP}
    ever = np.zeros(g.mask.shape, bool)
    for y in YEARS:
        f = interp(snaps, y)
        with nc.Dataset('%s/forcing/cv.%d.nc' % (OUT, y)) as d:
            cv = np.ma.filled(d['cv'][0], 0.0)
        m = np.broadcast_to(cv[None], (12,) + cv.shape).copy()
        irr, pad, freed = tile_months(cv[V_IRR - 1], cv[V_PAD - 1],
                                      f['np'], f['pad'])
        m[:, V_IRR - 1] = irr
        m[:, V_PAD - 1] = pad
        m[:, V_BARE - 1] += freed
        ever |= (freed > 0).any(0)
        s = m.sum(1)
        err = np.abs(s - 1)[:, g.active].max()
        log('CHECK calendar cv %d: max |sum-1| %.2e; irrigated tile growing share by month %s'
            % (y, err, np.round(((m[:, V_IRR - 1] + m[:, V_PAD - 1]).sum((1, 2)) /
                                 max((cv[V_IRR - 1] + cv[V_PAD - 1]).sum(), 1e-30)), 2).tolist()))
        log('CHECK calendar cv %d: paddy share of the growing area by month %s'
            % (y, np.round(m[:, V_PAD - 1].sum((1, 2)) /
                           np.maximum((m[:, V_IRR - 1] + m[:, V_PAD - 1]).sum((1, 2)), 1e-30), 2).tolist()))
        t, _ = month_starts(y)
        yearly_file('%s/cv.%d.nc' % (CALDIR, y), y, g, 'cv', m, 'fraction', t,
                    extra=('veg_class', 20),
                    attrs={'long_name': 'vegetation tile fraction, monthly irrigation season'})
    return ever


def params_bare(g, ever):
    """The bare tile must exist wherever it receives area (forced Cv > 0 on a
    class without a tile is fatal, lu_force.c:70-91)."""
    p = OUT + '/params/params_wur.nc'
    bak = p + '.nocal_bak'
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
    with nc.Dataset(p, 'a') as d:
        cv = np.ma.filled(d['Cv'][:], 0.0)
        need = ever & g.active & (cv[V_BARE - 1] <= 0)
        cv[V_BARE - 1][need] = EPS
        sm = cv.sum(0)
        cv = np.where(sm > 0, cv / np.where(sm > 0, sm, 1), 0.0)
        d['Cv'][:] = cv
    log('CHECK params Cv: bare tile created (Cv %g) in %d cells' % (EPS, int(need.sum())))


def efficiency(g):
    p = OUT + '/params/irr.nc'
    bak = p + '.placeholder_bak'
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
    with nc.Dataset(p, 'a') as d:
        gwf = np.ma.filled(d['groundwater_fraction'][:], 0.0)
        eff = EF_SURF_SA * (gwf + (1 - gwf) * EC_CANAL)
        v = d['irrigation_efficiency']
        v[:] = eff
        for a in list(v.ncattrs()):
            if a == 'PLACEHOLDER':
                v.delncattr(a)
        v.source = ('Jaegermeyr et al. 2015 HESS 19:3073 Table 5 (South Asia surface '
                    'field application efficiency 0.48; Rohwer et al. 2007: 0.48) x '
                    'conveyance (Table 1 open canal, loam 0.75) on the surface-water '
                    'share, 1 on the groundwater share (GMIA v5)')
    log('CHECK efficiency: min %.3f mean %.3f max %.3f' % (
        eff[g.active].min(), eff[g.active].mean(), eff[g.active].max()))


if __name__ == '__main__':
    g = ISC()
    season = (sys.argv[sys.argv.index('--season') + 1] if '--season' in sys.argv
              else 'ggcmi')
    assert season in SEASONS, season
    if '--eff' in sys.argv:
        efficiency(g)
    else:
        log('calendar season option: %s' % season)
        params_bare(g, calendar(g, season))
