"""Run MetSim 2.0.1 (installed in the metsim venv) on the make_inputs.py
records, hourly and 3-hourly. Writes OUTDIR/<site>_<step>.npz.

MetSim options (everything else is the package default):
    method mtclim, prec_type uniform, utc_offset False (local solar time),
    lw_type prata, lw_cloud cloud_deardorff, tdew_tol 1e-6,
    tmax_daylength_fraction 0.67, rain_scalar 0.75, tday_coef 0.45,
    lapse_rate 0.0065, sw_prec_thresh 0, calendar standard,
    out_precision f8 (the default f4 would hide round-off), scheduler
    synchronous. Record = 90 state days + the compared days.

usage: run_metsim.py INPUTS.npz OUTDIR
"""
import datetime
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')
import metsim                                            # noqa: E402
from metsim import MetSim                                # noqa: E402

DVARS = ['t_min', 't_max', 'prec', 'vapor_pressure', 'shortwave', 'tskc',
         'pet']
VARS = ['temp', 'prec', 'shortwave', 'longwave', 'vapor_pressure',
        'air_pressure', 'spec_humid', 'rel_humid', 'wind', 'tskc']


def main(inp, outdir, tol=1e-6, only=None):
    z = np.load(inp, allow_pickle=False)
    lead = int(z['lead'])
    os.makedirs(outdir, exist_ok=True)
    for site in z['sites']:
        site = str(site)
        if only and site not in only.split(','):
            continue
        lat, lng, elev, o = z[site + '/meta']
        start = datetime.date.fromordinal(int(o))
        n = len(z[site + '/tmax']) - lead
        dates = pd.date_range(pd.Timestamp(start), periods=n, freq='D')
        sdates = pd.date_range(pd.Timestamp(start) - pd.Timedelta('90D'),
                               periods=90, freq='D')
        co = dict(lat=[lat], lon=[lng])
        def rec(v, sl):
            a = z['%s/%s' % (site, v)][sl]
            return (('time', 'lat', 'lon'), a[:, None, None])
        ds = xr.Dataset(dict(t_max=rec('tmax', slice(lead, None)),
                             t_min=rec('tmin', slice(lead, None)),
                             prec=rec('prec', slice(lead, None)),
                             wind=rec('wind', slice(lead, None))),
                        coords=dict(time=dates, **co))
        st = xr.Dataset(dict(t_max=rec('tmax', slice(0, lead)),
                             t_min=rec('tmin', slice(0, lead)),
                             prec=rec('prec', slice(0, lead))),
                        coords=dict(time=sdates, **co))
        dom = xr.Dataset(dict(elev=(('lat', 'lon'), [[elev]]),
                              mask=(('lat', 'lon'), np.ones((1, 1), 'i4'))),
                         coords=co)
        w = os.path.join(outdir, 'work_' + site)
        os.makedirs(w, exist_ok=True)
        ds.to_netcdf(w + '/forcing.nc')
        st.to_netcdf(w + '/state.nc')
        dom.to_netcdf(w + '/domain.nc')
        for step in (60, 180, 1440):
            od = os.path.join(w, 'out%d' % step)
            os.makedirs(od, exist_ok=True)
            params = dict(
                time_step=step, start=str(dates[0].date()).replace('-', '/'),
                stop=str(dates[-1].date()).replace('-', '/'),
                forcing=w + '/forcing.nc', domain=w + '/domain.nc',
                state=w + '/state.nc', forcing_fmt='netcdf', out_dir=od,
                out_vars=VARS if step < 1440 else DVARS, out_precision='f8', scheduler='synchronous',
                prec_type='uniform', tdew_tol=tol, utc_offset=False, verbose=0,
                forcing_vars=dict(t_max='t_max', t_min='t_min', prec='prec',
                                  wind='wind'),
                state_vars=dict(t_max='t_max', t_min='t_min', prec='prec'),
                domain_vars=dict(elev='elev', mask='mask', lat='lat',
                                 lon='lon'))
            try:
                MetSim(params).run()
            except Exception as e:
                print(site, step, 'METSIM FAILED:', type(e).__name__, e, flush=True)
                continue
            f = [x for x in os.listdir(od) if x.endswith('.nc')][0]
            o = xr.open_dataset(os.path.join(od, f))
            res = {v: o[v].values[:, 0, 0] for v in (VARS if step < 1440 else DVARS)}
            np.savez(os.path.join(outdir, '%s_%d.npz' % (site, step)),
                     **res)
            print(site, step, len(next(iter(res.values()))), flush=True)
    print('metsim', metsim.__version__)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], *[float(sys.argv[3])] * (len(sys.argv) > 3),
         *sys.argv[4:5])
