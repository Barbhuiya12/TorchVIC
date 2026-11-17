"""Restart-file verification (STATENAME / INIT_STATE) of one matrix row.

    python state_matrix.py stehe:EB --days 10 --split 5 [--only write,read,round]
                           [--opt KEY=VALUE ...] [--out DIR]

Per row, in DIR (default <data root>/verification/item16_state/<row>):

  vic_cont/   pristine VIC over the whole period, STATENAME at day --split
  vic_half2/  pristine VIC from that state (INIT_STATE) to the end
  tv_cont/    TorchVIC over the whole period, STATENAME at day --split
  tv_half2/   TorchVIC from VIC's state file (INIT_STATE vic_cont/...)
  tv_rt/      TorchVIC from its own state file (tv_cont/...)

Stage `write`:  tv_cont's state file against vic_cont's, every variable
                (state_diff.py, |d| <= 1e-12 max(1,|v|)).
Stage `read`:   tv_half2 against vic_half2, and against vic_cont's second
                half, with compare.py's grade (pass <= 1e-4).
Stage `round`:  tv_rt against tv_cont's second half, bit for bit.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix as M                                           # noqa: E402
import run_matrix as R                                       # noqa: E402
import state_diff                                            # noqa: E402

OUTROOT = os.path.join(M.WORK, 'item16_state')


def _case_with(r, start, end):
    c = dict(M.CASES[r['case']])
    c['start'], c['end'] = start, end
    name = '_s16_%s_%s_%s' % (r['case'], start, end)
    M.CASES[name] = c
    rr = dict(r)
    rr['case'] = name
    return rr


def write_global(r, d, start, end, state=None, init=None):
    rr = _case_with(r, start, end)
    extra = []
    if init:
        extra.append('INIT_STATE %s' % init)
    if state:
        name, when = state
        extra += ['STATENAME %s' % name, 'STATEYEAR %d' % when.year,
                  'STATEMONTH %d' % when.month, 'STATEDAY %d' % when.day,
                  'STATESEC 0']
    os.makedirs(os.path.join(d, 'out'), exist_ok=True)
    txt = M.global_text(rr, os.path.join(d, 'out'), d, M.outvars(),
                        extra_lines=extra)
    g = os.path.join(d, 'global.txt')
    open(g, 'w').write(txt)
    return g


def run(who, r, g, d):
    t0 = time.time()
    env = R._env(1 if who == 'vic' else 4)
    if who == 'vic':
        cmd = [M.VIC[r['core']], '-g', g]
    else:
        cmd = [R.PY, '-m', 'torchvic', 'run', g, '--core', r['core'],
               '--out', os.path.join(d, 'out')]
        if r['daily']:
            cmd.append('--daily')
        env['TORCHVIC_NUM_THREADS'] = '4'
    with open(os.path.join(d, 'run.log'), 'w') as log:
        log.write(' '.join(cmd) + '\n')
        log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           env=env, cwd=d)
    if p.returncode != 0:
        txt = open(os.path.join(d, 'run.log'), errors='replace').read()
        i = txt.rfind('Traceback')
        print('RUN FAILED (%s in %s): %s' % (who, d, (txt[i:] if i >= 0
              else txt[-600:])[-900:]), file=sys.stderr)
    return p.returncode, time.time() - t0


def state_of(d):
    f = sorted(glob.glob(os.path.join(d, 'st*.nc')))
    return f[-1] if f else None


def hist_arrays(d):
    import xarray as xr
    f = sorted(glob.glob(os.path.join(d, 'out', '*.nc')))
    ds = xr.open_dataset(f[0], mask_and_scale=True).load()
    return ds


def second_half(ds, n_first):
    return ds.isel(time=slice(n_first, None))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('row')
    ap.add_argument('--days', type=int, default=10)
    ap.add_argument('--split', type=int, default=5)
    ap.add_argument('--only', default='write,read,round')
    ap.add_argument('--opt', action='append', default=[])
    ap.add_argument('--out', default='')
    ap.add_argument('--core', default='5.1.0')
    ap.add_argument('--daily', action='store_true',
                    help='MODEL_STEPS_PER_DAY 1 (matrix daily rows)')
    ap.add_argument('--reuse', action='store_true',
                    help='keep the pristine VIC continuous run already in '
                    'the directory (TorchVIC\'s is always rerun: a reused '
                    'tv_cont from older code is not the run tv_rt is '
                    'compared with)')
    ap.add_argument('--extra', action='append', default=[],
                    help='a global-file line, e.g. "SPATIAL_FROST TRUE 3"')
    ap.add_argument('--params', default='', help='matrix.FS_STEHE etc.: '
                    'a parameter file other than the case\'s')
    a = ap.parse_args(argv)
    case, mode = a.row.split(':')[:2]
    opts = dict(kv.split('=', 1) for kv in a.opt)
    r = M.row(case, mode, core=a.core, daily=a.daily,
              params=((a.params if os.path.exists(a.params)
                       else getattr(M, a.params)) if a.params else None),
              extra=a.extra, **opts)
    c = M.CASES[case]
    tag = '%s_%s%s%s%s' % (case, mode, '_daily' if a.daily else '', ''.join('_%s=%s' % kv for kv in opts.items()),
                         ('_' + os.path.basename(os.path.dirname(a.params))) if a.params and os.path.exists(a.params) else '')
    tag += ''.join('_' + x.replace(' ', '') for x in a.extra)
    if a.params:
        tag += '_' + os.path.basename(a.params if os.path.exists(a.params)
                                      else getattr(M, a.params, a.params))[:24]
    root = a.out or os.path.join(OUTROOT, tag)
    start = c['start']
    s0 = dt.date(*start)
    end = s0 + dt.timedelta(days=a.days - 1)
    mid = dt.datetime.combine(s0 + dt.timedelta(days=a.split), dt.time())
    end_t = (end.year, end.month, end.day)
    spd = 1 if a.daily else c['forcing_spd']
    stages = a.only.split(',')
    res = dict(row=tag, days=a.days, split=a.split)
    dirs = {k: os.path.join(root, k) for k in
            ('vic_cont', 'vic_half2', 'tv_cont', 'tv_half2', 'tv_rt')}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    def clean(d):
        for f in glob.glob(os.path.join(d, 'out', '*')) + \
                glob.glob(os.path.join(d, 'st*.nc')):
            os.remove(f)

    # --- continuous runs, state at the split
    for who in ('vic', 'tv'):
        d = dirs[who + '_cont']
        if who == 'vic' and a.reuse and state_of(d) and \
                glob.glob(os.path.join(d, 'out', '*')):
            res[who + '_cont_rc'] = 'reused'
            continue
        clean(d)
        g = write_global(r, d, start, end_t, state=(os.path.join(d, 'st'), mid))
        res[who + '_cont_rc'] = run(who, r, g, d)
    sv, st = state_of(dirs['vic_cont']), state_of(dirs['tv_cont'])
    res['vic_state'], res['tv_state'] = sv, st
    if 'write' in stages and sv and st:
        lines = []
        bad = state_diff.diff(sv, st, 1e-12, out=lines.append)
        open(os.path.join(root, 'write_diff.txt'), 'w').write('\n'.join(lines))
        res['write_bad'] = bad
        res['write_worst'] = max(float(l.split()[-1]) for l in lines
                                 if 'max rel/abs' in l)
    half_start = (mid.year, mid.month, mid.day)
    cmp = None
    if 'read' in stages or 'round' in stages:
        from compare import compare_dirs
        for who, init, key in (('vic', sv, 'vic_half2'), ('tv', sv, 'tv_half2'),
                               ('tv', st, 'tv_rt')):
            if key == 'tv_rt' and 'round' not in stages:
                continue
            if key != 'vic_half2' and 'read' not in stages and key == 'tv_half2':
                continue
            d = dirs[key]
            clean(d)
            g = write_global(r, d, half_start, end_t, init=init)
            res[key + '_rc'] = run(who, r, g, d)
        n1 = a.split * spd
        for key, ref in (('tv_half2', 'vic_half2'), ('tv_half2', 'vic_cont')):
            if key in dirs and os.path.isdir(os.path.join(dirs[key], 'out')):
                try:
                    # vic_cont's second half is cut into a directory view
                    refd = dirs[ref]
                    if ref == 'vic_cont':
                        import xarray as xr
                        cutd = os.path.join(root, 'vic_cont_half2', 'out')
                        os.makedirs(cutd, exist_ok=True)
                        ds = second_half(hist_arrays(refd), n1)
                        ds.to_netcdf(os.path.join(cutd, 'h.nc'))
                        refd = os.path.join(root, 'vic_cont_half2')
                    cr = compare_dirs(os.path.join(refd, 'out'),
                                      os.path.join(dirs[key], 'out'), r)
                    res['read_vs_' + ref] = {k: cr.get(k) for k in (
                        'grade', 'grade_water', 'grade_energy', 'worst_water',
                        'score_water', 'worst_energy', 'score_energy',
                        'error')}
                except Exception as e:                      # noqa: BLE001
                    res['read_vs_' + ref] = 'error %s: %s' % (type(e).__name__, e)
        # the continuous runs' own second halves: what the restart is
        # measured against when the row itself is not a pass
        try:
            import xarray as xr
            cuts = {}
            for who in ('vic', 'tv'):
                cd = os.path.join(root, who + '_cont_half2')
                os.makedirs(os.path.join(cd, 'out'), exist_ok=True)
                second_half(hist_arrays(dirs[who + '_cont']), n1).to_netcdf(
                    os.path.join(cd, 'out', 'h.nc'))
                cuts[who] = os.path.join(cd, 'out')
            cr = compare_dirs(cuts['vic'], cuts['tv'], r)
            res['base_second_half_tv_cont_vs_vic_cont'] = {k: cr.get(k) for k in (
                'grade', 'grade_water', 'grade_energy', 'worst_water',
                'score_water', 'worst_energy', 'score_energy')}
        except Exception as e:                              # noqa: BLE001
            res['base_second_half_tv_cont_vs_vic_cont'] = 'error %s' % e
        if 'round' in stages:
            h1 = second_half(hist_arrays(dirs['tv_cont']), n1)
            h2 = hist_arrays(dirs['tv_rt'])
            worst, nbad = 0.0, []
            for v in h1.data_vars:
                if not v.startswith('OUT_') or v not in h2.data_vars:
                    continue
                x, y = h1[v].values, h2[v].values
                if x.shape != y.shape:
                    nbad.append((v, 'shape'))
                    continue
                if not np.array_equal(x, y, equal_nan=True):
                    d_ = np.nanmax(np.abs(x - y))
                    worst = max(worst, float(d_))
                    nbad.append((v, float(d_)))
            res['round_bitwise'] = not nbad
            res['round_worst_abs'] = worst
            res['round_differs'] = nbad[:8]
    print(json.dumps(res, indent=1, default=str))
    json.dump(res, open(os.path.join(root, 'result.json'), 'w'), indent=1,
              default=str)
    return res


if __name__ == '__main__':
    main()
