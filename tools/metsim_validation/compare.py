"""TorchVIC's daily-forcing path against real MetSim 2.0.1 output.

usage: compare.py INPUTS.npz METSIM_OUTDIR   (torchvic venv, CPU)
The first day of each MetSim run is dropped by default: MetSim has no
vapour pressure before that day's minimum and extrapolates, where TorchVIC,
given the 90 leading days, interpolates from the day before.
Prints max abs and rmse per variable and step, pooled over the sites whose
MetSim run succeeded, plus the worst site/day when requested.
"""
import datetime
import os
import sys

import numpy as np
import torch

torch.set_default_device('cpu')
from torchvic.forcing import daily                       # noqa: E402

# (name, TorchVIC key, MetSim key, MetSim-to-TorchVIC scale)
#: MetSim 2.0.1 misindexes sunrise times at sites that have polar day or night
#: in the year (the 68 and 76 N series), so seed3 is not comparable.
EXCLUDE = {'seed3'}

PAIRS = [('SWdown W/m2', 'shortwave', 'shortwave', 1.0),
         ('LWdown W/m2', 'longwave', 'longwave', 1.0),
         ('Tair C', 'tair', 'temp', 1.0),
         ('vp Pa', 'vp', 'vapor_pressure', 1000.0),
         ('pressure Pa', 'press', 'air_pressure', 1000.0),
         ('wind m/s', 'wind', 'wind', 1.0),
         ('prec mm/step', 'prcp', 'prec', 1.0),
         ('tskc', 'tskc', 'tskc', 1.0)]


def run_site(z, site, spd):
    lat, lng, elev, o = z[site + '/meta']
    lead = int(z['lead'])
    start = datetime.date.fromordinal(int(o)) - datetime.timedelta(days=90)
    days = {k: torch.as_tensor(z['%s/%s' % (site, v)],
                               dtype=torch.float64)[None]
            for k, v in (('tmax', 'tmax'), ('tmin', 'tmin'),
                         ('prcp', 'prec'), ('wind', 'wind'))}
    out = daily.daily_to_subdaily(days, start, lat, lng, elev,
                                  steps_per_day=spd)
    return {k: v[0].numpy()[90 * spd:] for k, v in out.items()
            if torch.is_tensor(v)}


def main(inp, mdir, verbose=False, skip_day0=True):
    z = np.load(inp)
    res = {}
    for spd, step in ((24, 60), (8, 180)):
        pool = {n: [] for n, *_ in PAIRS}
        used = []
        for site in z['sites']:
            site = str(site)
            if site in EXCLUDE:
                continue
            f = os.path.join(mdir, '%s_%d.npz' % (site, step))
            if not os.path.exists(f):
                continue
            ms = np.load(f)
            tv = run_site(z, site, spd)
            used.append(site)
            for n, kt, km, sc in PAIRS:
                d = tv[kt] - ms[km] * sc
                if skip_day0:
                    d = d[spd:]
                pool[n].append(d)
                if verbose:
                    print('  %-20s %-14s %-12s max %.3e' % (
                        site, step, n, np.abs(d).max()))
        print('\nstep %d min (%d steps/day), sites %d: %s' % (
            step, spd, len(used), ' '.join(used)))
        print('  %-14s %12s %12s' % ('variable', 'max abs', 'rmse'))
        for n, *_ in PAIRS:
            d = np.concatenate(pool[n])
            print('  %-14s %12.4e %12.4e' % (
                n, np.abs(d).max(), np.sqrt((d ** 2).mean())))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == 'v',
         not (len(sys.argv) > 3 and sys.argv[3] == 'all'))
