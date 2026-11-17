"""Pack a small regression fixture from the MetSim 2.0.1 runs (numpy only).

usage: make_fixture.py INPUTS.npz METSIM_OUTDIR OUT.npz
Keeps two sites, the last 60 days of each run (the trailing 30- and 90-day
windows are full there), hourly and 3-hourly, float32.
"""
import json
import sys

import numpy as np

SITES = ('seed0', 'seed2')
VARS = ('temp', 'shortwave', 'longwave', 'vapor_pressure', 'air_pressure',
        'wind')
KEEP = 60
OPTIONS = dict(
    package='metsim==2.0.1 (PyPI), python 3.8, numpy 1.19, pandas 0.25.3, '
            'xarray 0.10.9, numba 0.53',
    method='mtclim', prec_type='uniform', utc_offset=False, lw_type='prata',
    lw_cloud='cloud_deardorff', tdew_tol=1e-6,
    tmax_daylength_fraction=0.67, rain_scalar=0.75, tday_coef=0.45,
    lapse_rate=0.0065, sw_prec_thresh=0.0, calendar='standard',
    out_precision='f8 (stored here as f4)', time_steps_min=[60, 180],
    state='90 leading days (inputs/<site>_* first 90 entries)',
    units='temp C, shortwave and longwave W/m2, vapor_pressure and '
          'air_pressure kPa, wind m/s')


def main(inp, mdir, out):
    z = np.load(inp)
    d = dict(metsim_version='2.0.1', options=json.dumps(OPTIONS),
             sites=np.array(SITES), lead=z['lead'], keep=np.array(KEEP))
    for s in SITES:
        d[s + '/meta'] = z[s + '/meta']
        for v in ('tmax', 'tmin', 'prec', 'wind'):
            d['%s/%s' % (s, v)] = z['%s/%s' % (s, v)]
        for step, spd in ((60, 24), (180, 8)):
            m = np.load('%s/%s_%d.npz' % (mdir, s, step))
            for v in VARS:
                d['%s/%d/%s' % (s, step, v)] = m[v][-KEEP * spd:].astype(
                    np.float32)
    np.savez_compressed(out, **d)


if __name__ == '__main__':
    main(*sys.argv[1:4])
