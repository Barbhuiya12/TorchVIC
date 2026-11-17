# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The `vicwur_cropsyst` module block and the CropSyst irrigation block of an ISC run, from the gridded inputs of
make_isc_cropsyst.py (<data root>/cropsyst_india/isc).

  tile_table(rot_code, cell_of_col, band_of_col, tile_class, cv_col)  one CropSyst tile per (cell, band) column
  module_block(case, global_file, n_days, start)  -> dict for the run config (connector 'vicwur_cropsyst')
  irrigation_block(inputs, n_cells)                -> `cropsyst_irrigation=` of vicwur_setup.modules
"""
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_isc_cropsyst as G                                      # noqa: E402

A = G.A
INPUTS = G.INDIA + '/isc'
#: irrigation classes of params/irr.nc (1-based veg classes) in file order
IRR_CLASSES = (18, 19)


def read_inputs(path=INPUTS + '/isc_cropsyst.nc'):
    import netCDF4 as nc
    d = nc.Dataset(path)
    f = lambda k: np.ma.filled(d[k][:], np.nan if d[k].dtype.kind == 'f' else 0)
    return dict(rot_code=f('rot_code'), sow_doy=f('sow_doy'), share=f('share'), crop=f('crop'),
                soil={k: f(k) for k in ('thickness', 'max_moist', 'air_entry', 'campbell_b')},
                lat=f('lat'), lon=f('lon'), crops=d.getncattr('crops').split())


def active_mask(global_file):
    """The domain mask (lat, lon) of the run's global file (row-major order = the run's cell order)."""
    import netCDF4 as nc
    from torchvic.vicwur import params as P
    g = P.parse_global(global_file)
    dom = g['DOMAIN'][-1][0]
    with nc.Dataset(dom) as d:
        return np.ma.filled(d['mask'][:], 0) > 0


def tile_table(rot_code, cell_of_col, band_of_col, cv_col, classes=G.CROP_TILES):
    """CropSyst tiles of the case's columns.

    rot_code [n_tile_class, n_cell] rotation code (0: none), cell_of_col / band_of_col [n_col] the cell and band of
    each VIC column, cv_col [n_col, n_vic_tile] the tile's Cv at the start. A tile exists on every column of a cell
    that has a rotation for the tile class and where the VIC tile exists (Cv > 0). Returns dict(code, cell, tile
    (0-based VIC tile), band, column), sorted by tile class, then column."""
    out = {k: [] for k in ('code', 'cell', 'tile', 'band', 'column')}
    for i, t in enumerate(classes):
        code = rot_code[i][cell_of_col]
        ok = (code > 0) & (cv_col[:, t - 1] > 0)
        col = np.flatnonzero(ok)
        out['code'].append(code[col].astype(np.int64))
        out['cell'].append(cell_of_col[col].astype(np.int64))
        out['tile'].append(np.full(len(col), t - 1, np.int64))
        out['band'].append(band_of_col[col].astype(np.int64))
        out['column'].append(col.astype(np.int64))
    return {k: np.concatenate(v) for k, v in out.items()}


def weather(start, n_days, active, tmax_var='tasmax'):
    """Daily weather of the run's cells from the ISC daily forcing (<data root>/forcing/vic5_daily): tmax, tmin (deg C),
    srad (W m-2), rhmin (% = vapour pressure / saturation at tmax) [n_day, n_cell], year, doy [n_day]."""
    import netCDF4 as nc
    from torchvic.crop.cropsyst import growth as gr
    import torch
    days = [start + datetime.timedelta(days=k) for k in range(n_days)]
    out = {k: [] for k in ('tmax', 'tmin', 'srad', 'vp')}
    for y in sorted({d.year for d in days}):
        ds = nc.Dataset('%s/forcing/vic5_daily/isc_daily.%d.nc' % (A, y))
        idx = [(d - datetime.date(y, 1, 1)).days for d in days if d.year == y]
        lo, hi = idx[0], idx[-1] + 1
        for k, v in (('tmax', tmax_var), ('tmin', 'tasmin'), ('srad', 'swdown'), ('vp', 'vp')):
            out[k].append(np.ma.filled(ds[v][lo:hi], np.nan)[:, active])
        ds.close()
    w = {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}
    for k, v in w.items():
        if not np.isfinite(v).all():
            raise ValueError('ISC daily forcing %s has non-finite values on the run cells' % k)
    es = gr.saturation_vp_kpa(torch.as_tensor(w['tmax'])).numpy()
    rh = np.clip(100.0 * w['vp'] / es, 0.0, 100.0)
    return dict(tmax=w['tmax'], tmin=w['tmin'], srad=w['srad'], rhmin=rh,
                year=[d.year for d in days], doy=[d.timetuple().tm_yday for d in days])


def module_block(case, global_file, n_days, start, hooks=None, inputs=None, group='isc', n_portions=None):
    """The `vicwur_cropsyst` module block (without 'connector') for the loaded `case` (load_case)."""
    inp = inputs or read_inputs()
    act2 = active_mask(global_file)
    n = int(act2.sum())
    ex = case.snowbands
    if ex is None:
        cell_of_col = np.arange(case.mosaic.cv.shape[0])
        band_of_col = np.zeros_like(cell_of_col)
    else:
        cell_of_col, band_of_col = np.asarray(ex['cell']), np.asarray(ex['band'])
    cv = case.mosaic.cv.detach().cpu().numpy()
    rot = np.stack([inp['rot_code'][i][act2] for i in range(len(G.CROP_TILES))])
    tt = tile_table(rot, cell_of_col, band_of_col, cv)
    soil = {k: v[:, act2].T for k, v in inp['soil'].items()}
    lat = np.broadcast_to(inp['lat'][:, None], act2.shape)[act2]
    lon = np.broadcast_to(inp['lon'][None, :], act2.shape)[act2]
    cs = dict(db_dir=INPUTS + '/Database', tiles=tt['code'].tolist(), tile_cell=tt['cell'], soil=soil,
              first_year=start.year)
    if n_portions:
        cs['n_portions'] = int(n_portions)
    irr = np.isin(tt['tile'] + 1, IRR_CLASSES)
    h = dict(transp=True, lai=True)
    h.update(hooks or {})
    return dict(cropsyst=cs, tile=tt['tile'], band=tt['band'], weather=weather(start, n_days, act2),
                lat=lat, lon=lon, n_cells=n, hooks=h, irrigated=irr, group=group), tt


def irrigation_block(n_cells, mad=0.5, root_fraction=0.6, window=(1, 12)):
    """`cropsyst_irrigation=` of vicwur_setup.modules: the live crop of the group drives irrigation on classes 18, 19.
    Technology: flood on paddy, furrow on the other irrigated crops (IRR_TECH); depletion over `root_fraction` of the
    root depth and allowed depletion `mad`, as the India management files; the season window is the whole year
    (rabi irrigation is November-March)."""
    code = np.tile(np.array([G.IRR_TECH[c] for c in IRR_CLASSES], np.int64), (n_cells, 1))
    return dict(crop_fn='group', code=code, window=tuple(window), mad=mad,
                observe=dict(mode=1, root_fraction=root_fraction))
