#!/usr/bin/env python3
"""Input variants for matrix rows whose option needs data the sample and
ISC files do not carry (written to WORK/variants, small, login node OK):

  himal_spatial_snow/params.nc   the himal window's parameters plus
                                 max_snow_distrib_slope = 0.5 m, the soil
                                 variable VIC 5's image driver reads under
                                 SPATIAL_SNOW (vic_init.c 5.1.0:692-699)
  himal_frost/params.nc          the himal window's parameters plus
                                 frost_slope = 4 C, the soil temperature
                                 spread VIC 5's image driver reads under
                                 SPATIAL_FROST (vic_init.c 5.1.0:703-710)
  stehe_frost/params.nc          the Stehekin frozen-soil sample parameters
                                 plus frost_slope = 4 C, as above
  {himal,stehe}_blowing/params.nc
                                 the parameters plus the three per-tile
                                 fields BLOWING reads (vic_init.c
                                 5.1.0:1283-1330): sigma_slope 0.08,
                                 lag_one 0.8, fetch 2000 m
  wg_carbon/params.nc            the Walnut Gulch parameters plus the carbon
  wg_carbon/forc.2008.nc         library VIC's image driver reads under
                                 CARBON (vic_init.c 5.1.0:285-375: Ctype,
                                 MaxCarboxRate, MaxiE_or_CO2Spec, LUE,
                                 Nscale, Wnpp_inhib, NPPfactor_sat), and the
                                 livneh forcing plus the three columns
                                 CARBON demands (vic_force.c 5.1.0:182-232):
                                 CATM 383 ppm as a mixing ratio, PAR = 0.45
                                 x shortwave, FDIR from the shortwave
  stehe_organic/params.nc        the Stehekin sample parameters plus the
                                 ORGANIC_FRACT
                                 fields (vic_init.c 5.1.0:572-605): organic
                                 0.30/0.20/0.10 by layer, bulk_density_org
                                 300 kg/m3, soil_density_org 1300 kg/m3, and
                                 bulk_density_comb (BULK_DENSITY_COMB,
                                 :607-617) = 0.95 x the mixed bulk density
  {stehe,wg}_organic             as above; wg from the Walnut Gulch file
  stehe_maxalb/params.nc         the Stehekin parameters plus
                                 max_snow_albedo per class, 0.72 to 0.90
                                 (vic_init.c 5.1.0:186-196)
  igp_veghist/veghist.2004.nc    a FORCING2 vegetation history for the igp
                                 window: LAI, fcanopy and albedo (time,
                                 veg_class, lat, lon) at the forcing's
                                 3-hourly steps, the parameter file's monthly
                                 value times a 17- and 23-day wave and a
                                 diurnal cycle (vic_force.c 5.1.0:264-359)
  igp_veghist0/veghist.2004.nc   the same with LAI 0 on days 10 and 11 of
                                 every month, which exercises the leafless
                                 rule (:419-439); VIC loses the canopy water
                                 of those steps (OUT_WATER_ERROR 0.18 mm), so
                                 its row carries VIC's own closure warn
"""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matrix as M                                          # noqa: E402


def veghist(out, zero=False):
    """igp_veghist{,0}/veghist.2004.nc, see the module docstring."""
    d = os.path.join(out, 'igp_veghist0' if zero else 'igp_veghist')
    os.makedirs(d, exist_ok=True)
    c = M.CASES['igp']
    with xr.open_dataset(c['params'], decode_times=False) as p, \
            xr.open_dataset(c['forcing'] + '2004.nc', decode_times=False) as f:
        t = xr.decode_cf(f[['time']])['time'].to_index()
        month = np.asarray(t.month) - 1
        days = (np.asarray(t - t[0]) / np.timedelta64(1, 'D')).astype('f8')
        hour = np.asarray(t.hour, dtype='f8')
        dom = np.asarray(t.day)

        def monthly(name):
            a = np.nan_to_num(p[name].values)      # class, month, y, x
            return np.moveaxis(a[:, month], 1, 0)  # time, class, y, x

        wave17 = 0.7 + 0.5 * np.sin(2 * np.pi * days / 17.0)
        wave23 = 0.85 + 0.15 * np.sin(2 * np.pi * days / 23.0)
        diurnal = 1.0 + 0.1 * np.cos(2 * np.pi * hour / 24.0)
        lai = monthly('LAI') * (wave17 * diurnal)[:, None, None, None]
        if zero:
            lai[(dom == 10) | (dom == 11)] = 0.0
        fcan = np.clip(monthly('fcanopy') * wave23[:, None, None, None],
                       0.0, 1.0)
        alb = (monthly('albedo')
               + (0.03 * np.sin(2 * np.pi * days / 31.0)
                  + 0.01 * np.cos(2 * np.pi * hour / 24.0))[
                      :, None, None, None])
        dims = ('time', 'veg_class', 'lat', 'lon')
        ds = xr.Dataset(
            {'LAI': (dims, lai.astype('f8')),
             'fcanopy': (dims, fcan.astype('f8')),
             'albedo': (dims, alb.astype('f8'))},
            coords=dict(time=f['time'], lat=f['lat'], lon=f['lon'],
                        veg_class=p['veg_class']))
        ds.to_netcdf(os.path.join(d, 'veghist.2004.nc'))


def main():
    out = os.path.join(M.WORK, 'variants')
    d = os.path.join(out, 'himal_spatial_snow')
    os.makedirs(d, exist_ok=True)
    with xr.open_dataset(M.CASES['himal']['params'], mask_and_scale=False,
                         decode_times=False) as p:
        p = p.load()
    v = xr.full_like(p['elev'], 0.5)
    v.attrs = dict(units='m', long_name='max_snow_distrib_slope')
    p['max_snow_distrib_slope'] = v
    p.to_netcdf(os.path.join(d, 'params.nc'))
    d = os.path.join(out, 'himal_frost')
    os.makedirs(d, exist_ok=True)
    with xr.open_dataset(M.CASES['himal']['params'], mask_and_scale=False,
                         decode_times=False) as p:
        p = p.load()
    v = xr.full_like(p['elev'], 4.0)
    v.attrs = dict(units='C', long_name='frost_slope')
    p['frost_slope'] = v
    p.to_netcdf(os.path.join(d, 'params.nc'))
    d = os.path.join(out, 'stehe_frost')
    os.makedirs(d, exist_ok=True)
    with xr.open_dataset(M.FS_STEHE, mask_and_scale=False,
                         decode_times=False) as p:
        p = p.load()
    v = xr.full_like(p['elev'], 4.0)
    v.attrs = dict(units='C', long_name='frost_slope')
    p['frost_slope'] = v
    p.to_netcdf(os.path.join(d, 'params.nc'))
    for case, src in (('himal', M.CASES['himal']['params']),
                      ('stehe', M.CASES['stehe']['params'])):
        d = os.path.join(out, case + '_blowing')
        os.makedirs(d, exist_ok=True)
        with xr.open_dataset(src, mask_and_scale=False,
                             decode_times=False) as p:
            p = p.load()
        for name, val, units in (('sigma_slope', 0.08, '-'),
                                 ('lag_one', 0.8, '-'),
                                 ('fetch', 2000.0, 'm')):
            v = xr.full_like(p['rarc'], val)
            v.attrs = dict(units=units, long_name=name)
            p[name] = v
        p.to_netcdf(os.path.join(d, 'params.nc'))
    for case, src in (('stehe', M.CASES['stehe']['params']),
                      ('wg', M.CASES['wg']['params'])):
        d = os.path.join(out, case + '_organic')
        os.makedirs(d, exist_ok=True)
        with xr.open_dataset(src, mask_and_scale=False,
                             decode_times=False) as p:
            p = p.load()
        bd, sd = p['bulk_density'], p['soil_density']
        frac = np.array([0.30, 0.20, 0.10])[:bd.sizes['nlayer']]
        org = xr.full_like(bd, 0.0)
        for j, f in enumerate(frac):
            org[j] = f
        mix = (1 - org) * bd + org * 300.0
        for name, val, units in (
                ('organic', org, '-'),
                ('bulk_density_org', xr.full_like(bd, 300.0), 'kg/m3'),
                ('soil_density_org', xr.full_like(bd, 1300.0), 'kg/m3'),
                ('bulk_density_comb', 0.95 * mix, 'kg/m3')):
            v = val.copy()
            v.attrs = dict(units=units, long_name=name)
            p[name] = v
        p.to_netcdf(os.path.join(d, 'params.nc'))
    d = os.path.join(out, 'stehe_maxalb')
    os.makedirs(d, exist_ok=True)
    with xr.open_dataset(M.CASES['stehe']['params'], mask_and_scale=False,
                         decode_times=False) as p:
        p = p.load()
    v = xr.full_like(p['Cv'], 0.0)
    alb = np.linspace(0.72, 0.90, v.sizes['veg_class'])
    for j, a in enumerate(alb):
        v[j] = a
    v.attrs = dict(units='-', long_name='max_snow_albedo')
    p['max_snow_albedo'] = v
    p.to_netcdf(os.path.join(d, 'params.nc'))
    d = os.path.join(out, 'wg_carbon')
    os.makedirs(d, exist_ok=True)
    c = M.CASES['wg']
    with xr.open_dataset(c['params'], mask_and_scale=False,
                         decode_times=False) as p:
        p = p.load()
    nv, ny, nx = p['Cv'].shape
    cls = np.arange(nv).reshape(nv, 1, 1) * np.ones((nv, ny, nx))
    # classes 7 and 8 are C3 with nitrogen scaling, 9 (grass) is C4 without
    c4 = (cls == 9)
    lib = dict(
        Ctype=(c4.astype('i4'), 'N/A'),
        MaxCarboxRate=(np.full(cls.shape, 6.0e-5), 'mol CO2/m2s'),
        MaxiE_or_CO2Spec=(np.where(c4, 1.0e-5, 1.2e-4), 'mol CO2/m2s'),
        LUE=(np.where(c4, 0.04, 0.08), 'mol CO2/mol photons'),
        Nscale=((~c4).astype('i4'), 'N/A'),
        Wnpp_inhib=(np.full(cls.shape, 0.5), 'fraction'),
        NPPfactor_sat=(np.full(cls.shape, 0.5), 'fraction'))
    for k, (v, u) in lib.items():
        x = xr.DataArray(v, dims=p['Cv'].dims, coords=p['Cv'].coords)
        x.attrs = dict(units=u, long_name=k)
        p[k] = x
    p.to_netcdf(os.path.join(d, 'params.nc'))
    with xr.open_dataset(c['forcing'] + '2008.nc') as f:
        f = f.load()
    sw = f['shortwave']
    f['CATM'] = xr.full_like(sw, 383.0e-6)
    f['CATM'].attrs = dict(units='mixing ratio')
    f['PAR'] = 0.45 * sw
    f['PAR'].attrs = dict(units='W m-2')
    f['FDIR'] = 0.3 + 0.4 * (sw / float(sw.max()))
    f['FDIR'].attrs = dict(units='fraction')
    f.to_netcdf(os.path.join(d, 'forc.2008.nc'))
    veghist(out)
    veghist(out, zero=True)
    print('wrote', out)


if __name__ == '__main__':
    main()
