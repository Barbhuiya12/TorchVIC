#!/usr/bin/env python
"""Generator for the VIC-WUR reference domains R1 (wur_chain) and R1b.

R1  : 6 cells (2 x 3 window of the ISC grid, Cauvery), synthetic hydrography
      c0->c1->c2->c3 (outlet), tributary c4->c2, delta receiver c5 of source
      c3 (c5 is its own outlet).  Real daily ERA5 forcing (the source of
      <data root>/forcing/vic5_daily) and ISC soil/veg, plus an irrigated
      non-paddy class (13), a paddy class (15) and the Pbare bare class (18).
      Starts 1 July 2003, runs 7 years.  Every VIC-WUR plugin input is made:
      routing (n=1 and 4 per day), dams (2 global + 1 local), water use
      (MUN LIV ENE MAN), irrigation, remote receiving cells, EFR, monthly Cv,
      CO2, WOFOST crop nc + text, b_co2.
R1b : the same, with unequal cell areas (remote-area quirk).  Only domain.nc
      is stored; every other file is R1's.

Usage: ref_domains.py [--out <data root>/vicwur] [--years 2003 2010]
Run with the env python (netCDF4, numpy, xarray, zarr, yaml).
"""
import argparse
import os
import sys

import numpy as np
import netCDF4 as nc

VT = os.environ.get('TORCHVIC_WORK_ROOT', '.')
DATA_ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
ROOT = DATA_ROOT
ISC_PARAMS = ROOT + '/vic_params_isc/params.nc'
ISC_DOMAIN = ROOT + '/vic_params_isc/domain.nc'
ZARR = ROOT + '/forcing/era5_isc_daily_1940_2026.zarr'
WOFOST_YAML = ROOT + '/static/upstream/wofost_crop_parameters/wofost72/'
DAILY_VARS = ('tasmax', 'tasmin', 'prcp', 'vp', 'sp', 'wind', 'swdown',
              'lwdown')
FILL = np.float32(9.96921e36)

NCELL = 6
NLAT, NLON = 2, 3
CHAIN = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
# downstream cell index of each cell (own index = outlet)
DOWN = [1, 2, 3, 3, 2, 5]
# veg classes (1-indexed) in the ISC params
V_IRR, V_PADDY, V_BARE = 13, 15, 18
# crop fractions per cell
CV_IRR = np.array([0.25, 0.30, 0.20, 0.15, 0.30, 0.10])
CV_PADDY = np.array([0.10, 0.15, 0.20, 0.10, 0.05, 0.30])
CV_BARE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
AREA_R1B_FACTOR = np.array([1.0, 0.5, 2.0, 1.5, 0.7, 3.0])


def find_window():
    """2 x 3 window of active ISC cells nearest to (11.9N, 77.2E)."""
    with nc.Dataset(ISC_DOMAIN) as d:
        m = np.ma.filled(d['mask'][:], 0)
        lat, lon = d['lat'][:], d['lon'][:]
    best = None
    for i in range(len(lat) - NLAT + 1):
        for j in range(len(lon) - NLON + 1):
            if m[i:i + NLAT, j:j + NLON].all():
                dist = (lat[i] - 11.9) ** 2 + (lon[j] - 77.2) ** 2
                if best is None or dist < best[0]:
                    best = (dist, i, j)
    return best[1], best[2]


def sl(i, j):
    return (slice(i, i + NLAT), slice(j, j + NLON))


def flat(a):
    """(..., 2, 3) -> (..., 6) cell order c0..c5 (row-major)."""
    return a.reshape(a.shape[:-2] + (NCELL,))


def unflat(a):
    return a.reshape(a.shape[:-1] + (NLAT, NLON))


def coords(ds, lat, lon):
    ds.createDimension('lat', NLAT)
    ds.createDimension('lon', NLON)
    v = ds.createVariable('lat', 'f8', ('lat',))
    v.units, v.long_name = 'degrees_north', 'latitude of grid cell center'
    v[:] = lat
    v = ds.createVariable('lon', 'f8', ('lon',))
    v.units, v.long_name = 'degrees_east', 'longitude of grid cell center'
    v[:] = lon


def var(ds, name, dtype, dims, data, units=None, fill=None):
    kw = {'fill_value': fill} if fill is not None else {}
    v = ds.createVariable(name, dtype, dims, **kw)
    if units:
        v.units = units
    v[:] = data
    return v


# ------------------------------------------------------------------ domain
def make_domain(path, lat, lon, area, mask, elev):
    with nc.Dataset(path, 'w') as ds:
        coords(ds, lat, lon)
        var(ds, 'mask', 'i4', ('lat', 'lon'), mask)
        var(ds, 'area', 'f8', ('lat', 'lon'), area, 'm2')
        var(ds, 'frac', 'f8', ('lat', 'lon'), np.ones((NLAT, NLON)))
        var(ds, 'elev', 'f8', ('lat', 'lon'), elev, 'm')
        ds.title = 'VIC-WUR reference domain'


# ------------------------------------------------------------------ params
def make_params(path, i, j):
    """ISC params of the window, with synthetic Cv (irrigated, paddy, bare)."""
    nat_cv = None
    with nc.Dataset(ISC_PARAMS) as src, nc.Dataset(path, 'w') as dst:
        s = sl(i, j)
        for n, L in src.dimensions.items():
            if n == 'lat':
                dst.createDimension(n, NLAT)
            elif n == 'lon':
                dst.createDimension(n, NLON)
            else:
                dst.createDimension(n, len(L))
        for n, v in src.variables.items():
            if n in ('veg_descr',):
                o = dst.createVariable(n, str, v.dimensions)
                for k in range(len(v)):
                    o[k] = str(v[k])
                continue
            fill = getattr(v, '_FillValue', None)
            o = dst.createVariable(n, v.dtype, v.dimensions,
                                   fill_value=fill)
            for a in v.ncattrs():
                if a != '_FillValue':
                    o.setncattr(a, v.getncattr(a))
            if n == 'lat':
                o[:] = v[i:i + NLAT]
            elif n == 'lon':
                o[:] = v[j:j + NLON]
            elif v.dimensions[-2:] == ('lat', 'lon'):
                o[...] = v[(Ellipsis,) + s]
            else:
                o[:] = v[:]
        # synthetic land cover: native classes rescaled, crops added
        cv = np.ma.filled(src['Cv'][(slice(None),) + s], 0.0).astype(float)
        cv = flat(cv)                                   # (18, 6)
        nat = cv.copy()
        for c in (V_IRR, V_PADDY, V_BARE):
            nat[c - 1] = 0.0
        for k in range(NCELL):
            if nat[:, k].sum() <= 0:
                nat[8, k] = 1.0                          # savanna fallback
            nat[:, k] /= nat[:, k].sum()
        new = nat * (1.0 - CV_IRR - CV_PADDY - CV_BARE)
        new[V_IRR - 1] = CV_IRR
        new[V_PADDY - 1] = CV_PADDY
        new[V_BARE - 1] = CV_BARE
        assert np.allclose(new.sum(0), 1.0)
        dst['Cv'][:] = unflat(new)
        nveg = (new[:V_BARE - 1] > 0).sum(0)
        dst['Nveg'][:] = unflat(nveg).astype('i4')
        # root zones of crop classes: crops root in the top two layers
        rd = dst['root_depth']
        rf = dst['root_fract']
        for c in (V_IRR, V_PADDY):
            rd[c - 1, :, :, :] = np.array([0.10, 0.40, 0.50])[:, None, None]
            rf[c - 1, :, :, :] = np.array([0.50, 0.40, 0.10])[:, None, None]
        dst['gridcell'][:] = np.arange(1, NCELL + 1).reshape(NLAT, NLON)
        dst['run_cell'][:] = 1
        # b_co2 [veg_class, lat, lon] (Li et al. 2019 style, synthetic)
        b = np.zeros(18)
        b[:] = 0.2
        b[V_IRR - 1] = 0.1
        b[V_PADDY - 1] = 0.1
        b[V_BARE - 1] = 0.0
        o = dst.createVariable('b_co2', 'f8', ('veg_class', 'lat', 'lon'))
        o.units = '-'
        o[:] = np.broadcast_to(b[:, None, None], (18, NLAT, NLON))
        dst.title = 'R1 params: ISC soil/veg window, synthetic Cv'
    return new


# ----------------------------------------------------------------- routing
def gamma_uh(n, dt_s, mean_s, shape):
    """Discretised gamma unit hydrograph over n bins, sum exactly 1."""
    t = (np.arange(n) + 0.5) * dt_s
    theta = mean_s / shape
    pdf = t ** (shape - 1) * np.exp(-t / theta)
    pdf /= pdf.sum()
    return pdf


def make_routing(path, lat, lon, ids, down_ids, steps_per_day):
    n = 6 * steps_per_day
    dt = 86400 // steps_per_day
    with nc.Dataset(path, 'w') as ds:
        coords(ds, lat, lon)
        ds.createDimension('time', n)
        var(ds, 'time', 'i4', ('time',), np.arange(n) * dt, 'seconds')
        var(ds, 'downstream_id', 'i4', ('lat', 'lon'), ids.reshape(NLAT, NLON))
        var(ds, 'downstream', 'i4', ('lat', 'lon'),
            down_ids.reshape(NLAT, NLON))
        uh_i = np.zeros((n, NCELL))
        uh_r = np.zeros((n, NCELL))
        for k in range(NCELL):
            uh_r[:, k] = gamma_uh(n, dt, (0.8 + 0.1 * k) * 86400, 2.0)
            uh_i[:, k] = gamma_uh(n, dt, (1.2 + 0.2 * k) * 86400, 3.0)
        var(ds, 'uh_inflow', 'f8', ('time', 'lat', 'lon'),
            uh_i.reshape(n, NLAT, NLON), '-')
        var(ds, 'uh_runoff', 'f8', ('time', 'lat', 'lon'),
            uh_r.reshape(n, NLAT, NLON), '-')


# -------------------------------------------------------------------- dams
def make_dams(path, lat, lon):
    nd = 3
    idmap = -np.ones(NCELL, 'i4')
    idmap[1], idmap[2], idmap[4] = 1, 2, 3          # dams in c1, c2, c4
    svc = -np.ones((nd, NCELL), 'i4')
    frac = -np.ones((nd, NCELL))
    # dam 1 (global, c1) serves c1, c2 ; dam 2 (global, c2) serves c2, c3
    svc[0, 1], frac[0, 1] = 1, 0.5
    svc[0, 2], frac[0, 2] = 1, 0.5
    svc[1, 2], frac[1, 2] = 2, 0.5
    svc[1, 3], frac[1, 3] = 2, 1.0
    # dam 3 (local, c4) serves c4
    svc[2, 4], frac[2, 4] = 3, 1.0
    with nc.Dataset(path, 'w') as ds:
        coords(ds, lat, lon)
        ds.createDimension('dam_class', nd)
        var(ds, 'dam_class', 'i4', ('dam_class',), np.arange(nd))
        var(ds, 'id_map', 'i4', ('lat', 'lon'), idmap.reshape(NLAT, NLON),
            fill=-1)
        var(ds, 'id', 'i4', ('dam_class',), [1, 2, 3])
        var(ds, 'type', 'i4', ('dam_class',), [1, 1, 0], '1 global, 0 local')
        var(ds, 'year', 'i4', ('dam_class',), [2003, 2005, 2004], 'year AD')
        var(ds, 'capacity', 'f8', ('dam_class',), [30.0, 15.0, 3.0], 'hm3')
        var(ds, 'inflow_fraction', 'f8', ('dam_class',), [1.0, 1.0, 0.8])
        var(ds, 'Nservice', 'i4', ('dam_class',), [2, 2, 1])
        var(ds, 'service', 'i4', ('dam_class', 'lat', 'lon'),
            svc.reshape(nd, NLAT, NLON), fill=-1)
        var(ds, 'service_fraction', 'f8', ('dam_class', 'lat', 'lon'),
            frac.reshape(nd, NLAT, NLON), fill=-1.0)


# --------------------------------------------------------------- water use
def make_wateruse(path, lat, lon, ids):
    """Remote receiving: source c3 -> receiver c5; source c2 -> receiver c4.
    The value at a receiving cell is its source's receiving_id."""
    nrec = np.zeros(NCELL, 'i4')
    nrec[3], nrec[2] = 1, 1
    rec = -np.ones((1, NCELL), 'i4')
    rec[0, 5] = ids[3]
    rec[0, 4] = ids[2]
    with nc.Dataset(path, 'w') as ds:
        coords(ds, lat, lon)
        ds.createDimension('wu_receiving', 1)
        var(ds, 'wu_receiving', 'i4', ('wu_receiving',), [0])
        var(ds, 'receiving_id', 'i4', ('lat', 'lon'),
            ids.reshape(NLAT, NLON), fill=-1)
        var(ds, 'Nreceiving', 'i4', ('lat', 'lon'), nrec.reshape(NLAT, NLON))
        var(ds, 'receiving', 'i4', ('wu_receiving', 'lat', 'lon'),
            rec.reshape(1, NLAT, NLON), fill=-1)


def make_irrigation(path, lat, lon, paddy_only=False):
    vc = [V_PADDY] if paddy_only else [V_IRR, V_PADDY]
    pd = [1] if paddy_only else [0, 1]
    n = len(vc)
    with nc.Dataset(path, 'w') as ds:
        coords(ds, lat, lon)
        ds.createDimension('irr_class', n)
        var(ds, 'irr_class', 'i4', ('irr_class',), np.arange(n))
        var(ds, 'veg_class', 'i4', ('irr_class',), vc)
        var(ds, 'paddy', 'i4', ('irr_class',), pd)
        var(ds, 'groundwater_fraction', 'f8', ('lat', 'lon'),
            np.array([0.2, 0.3, 0.4, 0.5, 0.25, 0.35]).reshape(NLAT, NLON))
        var(ds, 'irrigation_efficiency', 'f8', ('lat', 'lon'),
            np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.45]).reshape(NLAT, NLON))
        var(ds, 'irrigation_offset', 'i4', ('lat', 'lon'),
            np.array([0, 5, 10, 15, 20, 3], 'i4').reshape(NLAT, NLON))


# ------------------------------------------------------------ yearly files
def days_in(y):
    return 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365


def month_starts(y):
    dm = [31, 29 if days_in(y) == 366 else 28, 31, 30, 31, 30, 31, 31, 30,
          31, 30, 31]
    return np.concatenate([[0], np.cumsum(dm)[:-1]])


def yearly_file(path, y, lat, lon, vname, data, units, times, extra_dim=None):
    """time x [extra] x lat x lon float32 file with 'days since y-01-01'."""
    with nc.Dataset(path, 'w') as ds:
        coords(ds, lat, lon)
        ds.createDimension('time', len(times))
        v = ds.createVariable('time', 'i4', ('time',))
        v.units = 'days since %d-01-01' % y
        v.calendar = 'proleptic_gregorian'
        v[:] = times
        dims = ('time',) + ((extra_dim[0],) if extra_dim else ()) + \
            ('lat', 'lon')
        if extra_dim:
            ds.createDimension(extra_dim[0], extra_dim[1])
        o = ds.createVariable(vname, 'f8', dims, fill_value=-1.0e30)
        o.units = units
        o[:] = data


def season(doy, phase):
    return 0.5 * (1 + np.sin(2 * np.pi * (doy - phase) / 365.0))


def make_forcing_files(fdir, y0, y1, lat, lon, cv_cells, ids):
    os.makedirs(fdir, exist_ok=True)
    rng = np.random.RandomState(7)
    base_scale = np.linspace(0.8, 1.2, NCELL)
    gwf = np.array([0.2, 0.3, 0.4, 0.5, 0.25, 0.35])
    for y in range(y0, y1 + 1):
        n = days_in(y)
        doy = np.arange(n)
        # sector demand (mm per day), groundwater fraction, consumption fraction
        sect = {'mun': (0.10, 100), 'liv': (0.03, 150), 'ene': (0.04, 200),
                'man': (0.06, 50)}
        for s, (amp, ph) in sect.items():
            dem = amp * (0.7 + 0.6 * season(doy, ph))[:, None] * \
                base_scale[None, :] * (1 + 0.02 * (y - y0))
            gw = np.broadcast_to((gwf * 0.8)[None, :], (n, NCELL)).copy()
            cons = np.broadcast_to(
                np.linspace(0.15, 0.45, NCELL)[None, :], (n, NCELL)).copy()
            for nm, arr, un in ((s + '_demand', dem, 'mm/day'),
                                (s + '_gw', gw, '-'), (s + '_cons', cons, '-')):
                yearly_file('%s/%s.%d.nc' % (fdir, nm, y), y, lat, lon, nm,
                            arr.reshape(n, NLAT, NLON), un, doy)
        # EFR (monthly): discharge m3/s, baseflow mm/day
        ms = month_starts(y)
        mdoy = ms + 15
        efq = (0.3 + 0.8 * season(mdoy, 220))[:, None] * \
            (0.5 + 0.5 * np.arange(1, NCELL + 1) / NCELL)[None, :]
        efb = (0.05 + 0.10 * season(mdoy, 200))[:, None] * \
            np.ones((1, NCELL))
        yearly_file('%s/efr_q.%d.nc' % (fdir, y), y, lat, lon, 'efr_q',
                    efq.reshape(12, NLAT, NLON), 'm3/s', ms)
        yearly_file('%s/efr_b.%d.nc' % (fdir, y), y, lat, lon, 'efr_b',
                    efb.reshape(12, NLAT, NLON), 'mm/day', ms)
        # monthly Cv: paddy area grows 25 % over the run and is larger in the
        # kharif season; the difference is taken from the irrigated class
        cv = np.broadcast_to(cv_cells[None, :, :], (12, 18, NCELL)).copy()
        grow = 1.0 + 0.25 * (y - y0) / max(y1 - y0, 1)
        sea = 0.7 + 0.6 * season(mdoy, 150)
        for m in range(12):
            p = cv_cells[V_PADDY - 1] * grow * sea[m]
            p = np.minimum(p, cv_cells[V_PADDY - 1] + cv_cells[V_IRR - 1]
                           - 0.02)
            d = p - cv_cells[V_PADDY - 1]
            cv[m, V_PADDY - 1] = p
            cv[m, V_IRR - 1] = cv_cells[V_IRR - 1] - d
        assert np.allclose(cv.sum(1), 1.0) and (cv >= 0).all()
        yearly_file('%s/cv.%d.nc' % (fdir, y), y, lat, lon, 'cv',
                    cv.reshape(12, 18, NLAT, NLON), 'fraction', ms,
                    extra_dim=('veg_class', 18))
        # CO2 (yearly, ppm)
        co2 = np.full((1, NLAT, NLON), 375.0 + 2.0 * (y - 2003))
        yearly_file('%s/co2.%d.nc' % (fdir, y), y, lat, lon, 'co2', co2, 'ppm',
                    [0])


def make_met_forcing(fdir, y0, y1, ii, jj):
    import xarray as xr
    os.makedirs(fdir, exist_ok=True)
    z = xr.open_zarr(ZARR)
    lat = z['lat'].values[ii:ii + NLAT]
    lon = z['lon'].values[jj:jj + NLON]
    for y in range(y0, y1 + 1):
        sel = z.sel(time=slice('%d-01-01' % y, '%d-12-31' % y)).isel(
            lat=slice(ii, ii + NLAT), lon=slice(jj, jj + NLON))
        n = sel.sizes['time']
        assert n == days_in(y), (y, n)
        with nc.Dataset('%s/met.%d.nc' % (fdir, y), 'w') as ds:
            coords(ds, lat, lon)
            ds.createDimension('time', n)
            v = ds.createVariable('time', 'i4', ('time',))
            v.units = 'days since %d-01-01' % y
            v.calendar = 'proleptic_gregorian'
            v[:] = np.arange(n)
            for k in ('tair', 'prcp', 'vp', 'sp', 'wind', 'swdown', 'lwdown'):
                if k == 'tair':     # the VIC-WUR image driver has no TMAX/TMIN
                    a = (0.5 * (sel['tasmax'].values + sel['tasmin'].values)
                         ).astype(np.float32)
                else:
                    a = sel[k].values.astype(np.float32)
                assert np.isfinite(a).all(), (k, y)
                o = ds.createVariable(k, 'f4', ('time', 'lat', 'lon'),
                                      fill_value=FILL)
                o.units = 'degC' if k == 'tair' else z[k].attrs.get('units', '')
                o[:] = a


# ------------------------------------------------------------------ WOFOST
SCALARS = ['TBASEM', 'TEFFMX', 'TSUMEM', 'IDSL', 'DLO', 'DLC', 'VERNSAT',
           'VERNBASE', 'TSUM1', 'TSUM2', 'DVSI', 'DVSEND', 'TDWI', 'RGRLAI',
           'SPA', 'SPAN', 'TBASE', 'CVL', 'CVO', 'CVR', 'CVS', 'Q10', 'RML',
           'RMO', 'RMR', 'RMS', 'PERDL', 'CFET', 'DEPNR', 'IAIRDU', 'RDI',
           'RRI', 'RDMCR', 'RDRLV_NPK', 'DVS_NPK_STOP', 'DVS_NPK_TRANSL',
           'NPK_TRANSLRT_FR', 'NCRIT_FR', 'PCRIT_FR', 'KCRIT_FR',
           'NMAXRT_FR', 'NMAXST_FR', 'PMAXRT_FR', 'PMAXST_FR', 'KMAXRT_FR',
           'KMAXST_FR', 'NLAI_NPK', 'NLUE_NPK', 'NMAXSO', 'PMAXSO', 'KMAXSO',
           'NPART', 'NSLA_NPK', 'NRESIDLV', 'NRESIDST', 'NRESIDRT',
           'PRESIDLV', 'PRESIDST', 'PRESIDRT', 'KRESIDLV', 'KRESIDST',
           'KRESIDRT', 'TCNT', 'TCPT', 'TCKT', 'NFIX_FR', 'MaxHeight', 'RGL',
           'RadAtten', 'WindAtten', 'TrunkRatio', 'Albedo', 'MinStomResist',
           'MaxArchResist', 'Fcanopy', 'FRTRL', 'CRITLAI']
TABLES = ['VERNRTB', 'DTSMTB', 'SLATB', 'SSATB', 'KDIFTB', 'EFFTB', 'AMAXTB',
          'TMPFTB', 'TMNFTB', 'CO2AMAXTB', 'CO2EFFTB', 'CO2TRATB', 'RFSETB',
          'FRTB', 'FLTB', 'FSTB', 'FOTB', 'RDRRTB', 'RDRSTB', 'NMAXLV_TB',
          'PMAXLV_TB', 'KMAXLV_TB']
# synthetic placeholders for what the WOFOST 7.2 yaml does not carry (NPK
# module of WOFOST-NPK, and the VIC feedback parameters); values follow the
# VIC-WUR WOFOST 8.1-style defaults in kind, not tuned to a crop
SYNTH = dict(RDRLV_NPK=0.05, DVS_NPK_STOP=1.3, DVS_NPK_TRANSL=0.8,
             NPK_TRANSLRT_FR=0.15, NCRIT_FR=1.0, PCRIT_FR=1.0, KCRIT_FR=1.0,
             NMAXRT_FR=0.5, NMAXST_FR=0.5, PMAXRT_FR=0.5, PMAXST_FR=0.5,
             KMAXRT_FR=0.5, KMAXST_FR=0.5, NLAI_NPK=1.0, NLUE_NPK=1.1,
             NMAXSO=0.0310, PMAXSO=0.0059, KMAXSO=0.0045, NPART=1.0,
             NSLA_NPK=0.5, NRESIDLV=0.004, NRESIDST=0.0015, NRESIDRT=0.0015,
             PRESIDLV=0.0005, PRESIDST=0.0003, PRESIDRT=0.0003,
             KRESIDLV=0.0025, KRESIDST=0.0050, KRESIDRT=0.0040, TCNT=10.0,
             TCPT=10.0, TCKT=10.0, NFIX_FR=0.0, MaxHeight=1.0, RGL=100.0,
             RadAtten=0.5, WindAtten=0.5, TrunkRatio=0.2, Albedo=0.2,
             MinStomResist=100.0, MaxArchResist=25.0, Fcanopy=0.9, FRTRL=0.2,
             CRITLAI=0.05, VERNSAT=0.0, VERNBASE=0.0)
SYNTH_TABLES = dict(
    NMAXLV_TB=[0.0, 0.05, 0.4, 0.05, 0.7, 0.03, 1.0, 0.025, 2.0, 0.02],
    PMAXLV_TB=[0.0, 0.007, 0.4, 0.007, 0.7, 0.004, 1.0, 0.003, 2.0, 0.002],
    KMAXLV_TB=[0.0, 0.05, 0.4, 0.05, 0.7, 0.03, 1.0, 0.025, 2.0, 0.02],
    VERNRTB=[-8.0, 0.0, -4.0, 0.4, 3.0, 1.0, 10.0, 0.0, 17.0, 0.0],
    DTSMTB=[0.0, 0.0, 30.0, 30.0, 45.0, 30.0])


def _fmt(x):
    return '%.6g' % x


def write_crop_file(path, y, comment):
    L = ['* WOFOST crop file generated by tools/vicwur/ref_domains.py (%s)'
         % comment, '* NPK and VIC feedback values are synthetic placeholders',
         '']
    for k in SCALARS:
        v = y[k] if k in y else SYNTH[k]
        if isinstance(v, (list, tuple)):
            v = v[0]
        L.append('%s = %s' % (k, _fmt(float(v))))
    L.append('')
    for t in TABLES:
        v = y.get(t, SYNTH_TABLES.get(t))
        v = [float(a) for a in v]
        pts = list(zip(v[0::2], v[1::2]))
        L.append('%s = %s , %s' % (t, _fmt(pts[0][0]), _fmt(pts[0][1])))
        for a, b in pts[1:]:
            L.append('  %s , %s' % (_fmt(a), _fmt(b)))
        L.append('')
    open(path, 'w').write('\n'.join(L) + '\n')


def write_mgmt_file(path):
    L = ['* WOFOST management file (synthetic)', '',
         'NRFTAB = 0.7', 'PRFTAB = 0.7', 'KRFTAB = 0.7',
         'NMINS = 0.0', 'RTNMINS = 0.0', 'PMINS = 0.0', 'RTPMINS = 0.0',
         'KMINS = 0.0', 'RTKMINS = 0.0', '',
         'FERNTAB = 01-01 0.0', '', 'FERPTAB = 01-01 0.0', '',
         'FERKTAB = 01-01 0.0', '', 'IRRTAB = 01-01 0.0', '']
    open(path, 'w').write('\n'.join(L))


def make_wofost(outdir, lat, lon):
    import yaml
    wdir = outdir + '/wofost'
    os.makedirs(wdir, exist_ok=True)
    crops = [('maize', 'Maize_VanHeemst_1988', V_IRR),
             ('rice', 'Rice_IR72', V_PADDY)]
    for name, var_, _ in crops:
        d = yaml.safe_load(open(WOFOST_YAML + name + '.yaml'))
        p = d['CropParameters']['Varieties'][var_]
        flat_ = {k: v[0] for k, v in p.items() if k != 'Metadata'}
        write_crop_file('%s/crop_%s.txt' % (wdir, name), flat_,
                        '%s %s' % (name, var_))
    write_mgmt_file(wdir + '/management_default.txt')
    # one text line per crop class (crop 0 maize on class 13, crop 1 rice on 15)
    lines = ['** WOFOST text parameter file for R1', '']
    for name, _, _ in crops:
        lines.append('%s/ crop_%s.txt management_default.txt 01-01 0'
                     % (wdir, name))
    open(outdir + '/wofost_text.txt', 'w').write('\n'.join(lines) + '\n')
    # crop nc
    nc_crop = 2
    nfert = 1
    with nc.Dataset(outdir + '/crop.nc', 'w') as ds:
        coords(ds, lat, lon)
        ds.createDimension('crop_class', nc_crop)
        ds.createDimension('month', 12)
        ds.createDimension('fertilizer_times', nfert)
        var(ds, 'crop_class', 'i4', ('crop_class',), np.arange(nc_crop))
        var(ds, 'month', 'i4', ('month',), np.arange(1, 13))
        var(ds, 'fertilizer_times', 'f8', ('fertilizer_times',), [0])
        var(ds, 'Ncrop', 'i4', ('lat', 'lon'),
            np.full((NLAT, NLON), nc_crop, 'i4'))
        cc = np.zeros((12, nc_crop, NCELL))
        cc[:, 0, :] = CV_IRR[None, :]
        cc[:, 1, :] = CV_PADDY[None, :]
        var(ds, 'Cc', 'f8', ('month', 'crop_class', 'lat', 'lon'),
            cc.reshape(12, nc_crop, NLAT, NLON), 'fraction')
        cvc = np.zeros((nc_crop, NCELL), 'i4')
        cvc[0], cvc[1] = V_IRR, V_PADDY
        var(ds, 'crop_veg_class', 'i4', ('crop_class', 'lat', 'lon'),
            cvc.reshape(nc_crop, NLAT, NLON))
        # plant/harvest day (maize kharif Jun 15 .. Oct 15 ; rice Jul 1..Nov 10)
        def cls(a):
            return np.array(a, 'f8')[:, None] * np.ones((1, NLAT * NLON))
        var(ds, 'plant_day', 'f8', ('crop_class', 'lat', 'lon'),
            cls([166, 182]).reshape(nc_crop, NLAT, NLON), 'doy')
        var(ds, 'harvest_day', 'f8', ('crop_class', 'lat', 'lon'),
            cls([288, 314]).reshape(nc_crop, NLAT, NLON), 'doy')
        var(ds, 'TSUM1', 'f8', ('crop_class', 'lat', 'lon'),
            cls([900, 800]).reshape(nc_crop, NLAT, NLON), 'K')
        var(ds, 'TSUM2', 'f8', ('crop_class', 'lat', 'lon'),
            cls([900, 700]).reshape(nc_crop, NLAT, NLON), 'K')
        var(ds, 'Tfactor', 'f8', ('lat', 'lon'), np.zeros((NLAT, NLON)), 'K')
        shp = (nfert, nc_crop, NLAT, NLON)
        for nm, val in (('DVS_point', 0.5), ('N_amount', 60.0),
                        ('P_amount', 20.0), ('K_amount', 20.0)):
            var(ds, nm, 'f8', ('fertilizer_times', 'crop_class', 'lat', 'lon'),
                np.full(shp, val))
        shp = (nc_crop, NLAT, NLON)
        for nm, val in (('N_mins', 10.0), ('P_mins', 2.0), ('K_mins', 5.0),
                        ('N_recovery', 0.5), ('P_recovery', 0.5),
                        ('K_recovery', 0.5), ('mineralization_period', 60.0)):
            var(ds, nm, 'f8', ('crop_class', 'lat', 'lon'), np.full(shp, val))
        var(ds, 'carbon', 'f8', ('lat', 'lon'), np.full((NLAT, NLON), 15.0))
        var(ds, 'pH', 'f8', ('lat', 'lon'), np.full((NLAT, NLON), 6.5))


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=ROOT + '/vicwur')
    ap.add_argument('--years', nargs=2, type=int, default=[2003, 2010])
    a = ap.parse_args(argv)
    y0, y1 = a.years
    out = a.out
    r1, r1b = out + '/R1', out + '/R1b'
    os.makedirs(r1, exist_ok=True)
    os.makedirs(r1b, exist_ok=True)
    ii, jj = find_window()
    print('window', ii, jj)
    with nc.Dataset(ISC_DOMAIN) as d:
        lat = d['lat'][ii:ii + NLAT]
        lon = d['lon'][jj:jj + NLON]
        area_isc = np.ma.filled(d['area'][ii:ii + NLAT, jj:jj + NLON], 0.0)
        elev = np.ma.filled(d['elev'][ii:ii + NLAT, jj:jj + NLON], 0.0)
    area = np.full((NLAT, NLON), area_isc.mean())
    mask = np.ones((NLAT, NLON), 'i4')
    make_domain(r1 + '/domain.nc', lat, lon, area, mask, elev)
    make_domain(r1b + '/domain.nc', lat, lon,
                (area.ravel() * AREA_R1B_FACTOR).reshape(NLAT, NLON), mask,
                elev)
    cv = make_params(r1 + '/params.nc', ii, jj)
    ids = np.arange(1, NCELL + 1, dtype='i4')
    down_ids = ids[np.array(DOWN)]
    make_routing(r1 + '/rout_n1.nc', lat, lon, ids, down_ids, 1)
    make_routing(r1 + '/rout_n4.nc', lat, lon, ids, down_ids, 4)
    make_dams(r1 + '/dams.nc', lat, lon)
    make_wateruse(r1 + '/wateruse.nc', lat, lon, ids)
    make_irrigation(r1 + '/irr.nc', lat, lon)
    make_irrigation(r1 + '/irr_paddy.nc', lat, lon, paddy_only=True)
    make_wofost(r1, lat, lon)
    make_forcing_files(r1 + '/forcing', y0, y1, lat, lon, cv, ids)
    make_met_forcing(r1 + '/forcing', y0, y1, ii, jj)
    print('R1 written to', r1)


if __name__ == '__main__':
    sys.exit(main())
