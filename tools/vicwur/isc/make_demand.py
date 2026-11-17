#!/usr/bin/env python
"""Sector demand forcing from Huang et al. 2018 (monthly, 0.5 deg, mm/month).

Writes <data root>/vicwur/isc/forcing/{mun,man,ene,liv}_{demand,gw,cons}.YYYY.nc
(12 monthly records per year, `PLUGIN_FORCE_TYPE <S>_DEMAND <s>_demand MONTH`).
  demand  mm per day (value is held over the month: mm per model step for the
          daily step), area-weighted conservative regrid of mm/month / days.
  cons    consumption fraction = cons / withdrawal (regridded totals).
  gw      PLACEHOLDER constant (no groundwater-share data downloaded).
"""
import os
import sys
import numpy as np
import netCDF4 as nc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa
from regrid import Regridder, fill_nearest   # noqa

SECT = {'mun': ('domestic', 'dom'), 'man': ('manufacturing', 'mfg'),
        'ene': ('electricity', 'elec'), 'liv': ('livestock', 'liv')}
GW_PLACEHOLDER = 0.5
M0 = 1971


def load(kind, folder, tag):
    with nc.Dataset('%s/huang/%s/%s_%s.nc' % (WORK, folder, kind, tag)) as d:
        lat, lon = np.array(d['lat'][:]), np.array(d['lon'][:])
        a = np.array(d[kind + '_' + tag][(YEARS[0] - M0) * 12:
                                         (YEARS[-1] - M0 + 1) * 12, :], float)
    return lat, lon, a


def to_grid(lat, lon, a):
    iy = np.rint((lat + 89.75) / 0.5).astype(int)
    ix = np.rint((lon + 179.75) / 0.5).astype(int)
    g = np.full((a.shape[0], 360, 720), np.nan)
    g[:, iy, ix] = a
    g[g > 1e10] = np.nan
    return g


def main():
    g = ISC()
    os.makedirs(OUT + '/forcing', exist_ok=True)
    rg = Regridder(np.arange(360) * 0.5 - 89.75, np.arange(720) * 0.5 - 179.75,
                   g.lat, g.lon)
    rows, chk = [], []
    nmon = len(YEARS) * 12
    days = np.concatenate([month_starts(y)[1] for y in YEARS]).astype(float)
    for s, (folder, tag) in SECT.items():
        lat, lon, w = load('withd', folder, tag)
        _, _, c = load('cons', folder, tag)
        W = to_grid(lat, lon, w)
        C = to_grid(lat, lon, c)
        dem = np.empty((nmon, g.ny, g.nx))
        cons = np.empty_like(dem)
        nfill = 0
        for t in range(nmon):
            mw, cov = rg.mean(W[t])
            mc, _ = rg.mean(C[t])
            if t % 60 == 0:
                st, dt_, pe = rg.mass_check(W[t], mw)
                chk.append('%s withdrawal month %d: src %.6e m3  dst %.6e m3'
                           '  err %.3e %%' % (s, t, st / 1e3, dt_ / 1e3, pe))
            bad = ~np.isfinite(mw)
            nfill = max(nfill, int((bad & g.active).sum()))
            mw = fill_nearest(mw)
            mc = fill_nearest(mc)
            dem[t] = mw / days[t]
            with np.errstate(invalid='ignore', divide='ignore'):
                fr = np.where(mw > 0, mc / mw, np.nan)
            cons[t] = np.clip(fr, 0, 1)
        # cells with no withdrawal: sector mean consumption fraction
        ok = np.isfinite(cons)
        mean_fr = np.nanmean(np.where(ok, cons, np.nan)) if ok.any() else 0.
        cons = np.where(ok, cons, mean_fr)
        gw = np.full_like(dem[:12], GW_PLACEHOLDER)
        for i, y in enumerate(YEARS):
            sl = slice(i * 12, i * 12 + 12)
            ms, _ = month_starts(y)
            yearly_file('%s/forcing/%s_demand.%d.nc' % (OUT, s, y), y, g,
                        s + '_demand', dem[sl], 'mm/day', ms, attrs={
                            'source': 'Huang et al. 2018 withd_%s (Zenodo 1209296)' % tag})
            yearly_file('%s/forcing/%s_cons.%d.nc' % (OUT, s, y), y, g,
                        s + '_cons', cons[sl], '-', ms)
            yearly_file('%s/forcing/%s_gw.%d.nc' % (OUT, s, y), y, g,
                        s + '_gw', gw, '-', ms, attrs={
                            'PLACEHOLDER': 'constant %.2f, no groundwater-share '
                            'data (CGWB/GMIA) downloaded' % GW_PLACEHOLDER})
        rows.append(stats_row(s + '_demand', dem, g.active, 'mm/day'))
        rows.append(stats_row(s + '_cons', cons, g.active, '-'))
        # volume totals over the ISC active cells, km3/yr mean 2000-2010
        vol = (dem * days[:, None, None] * 1e-3 * g.area[None]).sum((1, 2))
        rows.append('%s mean annual volume, all ISC cells (domain area): '
                    '%.3f km3/yr' % (s, vol.sum() / len(YEARS) / 1e9))
        log('demand %s done, uncovered ISC cells filled by nearest: %d'
            % (s, nfill))
    for r in chk + rows:
        log('CHECK ' + r)


if __name__ == '__main__':
    main()
