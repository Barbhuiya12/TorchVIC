#!/usr/bin/env python3
"""Run the verification matrix (verification/matrix.py): every row through
pristine VIC (the row's release) and TorchVIC (the same core), then compare.

    run_matrix.py [--only SUBSTR[,SUBSTR]] [--jobs N] [--threads T]
                  [--device cpu] [--days N] [--rerun] [--compare-only]

For each row, in WORK/runs/<id>/:
  global.vic.txt, global.tv.txt   the two global files (identical but for
                                  RESULT_DIR / LOG_DIR)
  vic/, tv/                       their per-step double-precision history
  vic.log, tv.log                 stdout+stderr, return codes in status.json
  compare.json                    verification/compare.py's metrics
Then verification/results/rows/<id>.json gets the comparison (small, kept
in the repository) and scoreboard.py builds the scoreboard from those.

`--days N` cuts every row to its first N days (a smoke test). Runs on a
Slurm CPU allocation (verification/run_all.sh).
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix as M                                          # noqa: E402
import compare as C                                         # noqa: E402

PY = sys.executable
NETCDF_LIB = os.path.join(os.environ.get('NETCDF_HOME', '/usr'), 'lib')
RESULTS = os.path.join(HERE, 'results')


def _env(threads):
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = NETCDF_LIB + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMPI_MCA_btl_vader_single_copy_mechanism'] = 'none'
    env['PYTHONNOUSERSITE'] = '1'
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[k] = str(threads)
    # official rows run with VIC's glibc pow/exp/sqrt (torchvic/core/xmath.py,
    # CPU only); TORCHVIC_EXACT_MATH=0 in the caller's environment runs the
    # default kernels instead
    env.setdefault('TORCHVIC_EXACT_MATH', '1')
    if env['TORCHVIC_EXACT_MATH'] == '1':
        env['ATEN_CPU_CAPABILITY'] = 'default'
    return env


def period_days(r, days=0):
    c = M.CASES[r['case']]
    a, b = dt.date(*c['start']), dt.date(*c['end'])
    n = (b - a).days + 1
    return min(n, days) if days else n


def prepare(r, days=0, root=None):
    """Write the row's two global files; returns its run directory."""
    root = root or os.path.join(M.WORK, 'runs')
    d = os.path.join(root, r['id'])
    for who in ('vic', 'tv'):
        os.makedirs(os.path.join(d, who), exist_ok=True)
        os.makedirs(os.path.join(d, who + '_logs'), exist_ok=True)
    rr = dict(r)
    if days:
        c = dict(M.CASES[r['case']])
        e = dt.date(*c['start']) + dt.timedelta(days=days - 1)
        c['end'] = (e.year, e.month, e.day)
        M.CASES['_tmp_' + r['case']] = c
        rr['case'] = '_tmp_' + r['case']
    ov = M.outvars(r['core'], full=True)
    if r['core'] != '5.1.0':
        # an older core stops on a name its output table does not have
        import re
        hdr = os.path.join(os.path.dirname(M.VIC[r['core']]), '..', 'shared_all',
                           'include', 'vic_driver_shared_all.h')
        if os.path.exists(hdr):
            known = set(re.findall(r'\bOUT_[A-Z0-9_]+\b', open(hdr).read()))
            ov = [n for n in ov if n in known]
    for who in ('vic', 'tv'):
        txt = M.global_text(rr, os.path.join(d, who),
                            os.path.join(d, who + '_logs'), ov)
        with open(os.path.join(d, 'global.%s.txt' % who), 'w') as f:
            f.write(txt)
    return d


def run_vic(r, d, threads=1):
    t0 = time.time()
    with open(os.path.join(d, 'vic.log'), 'w') as log:
        p = subprocess.run([M.VIC[r['core']], '-g',
                            os.path.join(d, 'global.vic.txt')],
                           stdout=log, stderr=subprocess.STDOUT,
                           env=_env(threads), cwd=d)
    return p.returncode, time.time() - t0


def run_tv(r, d, device='cpu', threads=4, days=0, extra=()):
    c = M.CASES[r['case']]
    steps = period_days(r, days) * c['forcing_spd']
    cmd = [PY, '-m', 'torchvic', 'run', os.path.join(d, 'global.tv.txt'),
           '--core', r['core'], '--device', device, '--out',
           os.path.join(d, 'tv'), '--steps', str(steps)]
    if r['daily'] and 'noflag' not in r['id']:
        cmd.append('--daily')
    cmd.extend(extra)
    t0 = time.time()
    env = _env(threads)
    env['TORCHVIC_NUM_THREADS'] = str(threads)
    with open(os.path.join(d, 'tv.log'), 'w') as log:
        log.write(' '.join(cmd) + '\n')
        log.flush()
        # cwd is the run directory, not the repository: `python -m` puts
        # the cwd first on sys.path, which would shadow the committed
        # snapshot on PYTHONPATH (verification/snapshot.sh) with the
        # working tree, which may hold uncommitted edits
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           env=env, cwd=d)
    return p.returncode, time.time() - t0


def one(r, args):
    d = prepare(r, args.days)
    st_path = os.path.join(d, 'status.json')
    st = {}
    if os.path.exists(st_path) and not args.rerun:
        st = json.load(open(st_path))
    if not args.compare_only:
        if args.rerun or st.get('vic_rc') != 0:
            for f in os.listdir(os.path.join(d, 'vic')):
                os.remove(os.path.join(d, 'vic', f))
            st['vic_rc'], st['vic_s'] = run_vic(r, d)
        if args.rerun or st.get('tv_rc') != 0:
            for f in os.listdir(os.path.join(d, 'tv')):
                os.remove(os.path.join(d, 'tv', f))
            st['tv_rc'], st['tv_s'] = run_tv(r, d, args.device, args.threads,
                                             args.days)
        json.dump(st, open(st_path, 'w'), indent=1)
    res = dict(row=r, status=st)
    try:
        res.update(C.compare_dirs(os.path.join(d, 'vic'),
                                  os.path.join(d, 'tv'), r, st))
    except Exception as e:                                  # noqa: BLE001
        res['compare_error'] = '%s: %s' % (type(e).__name__, e)
    res['vic_log_tail'] = _tail(os.path.join(d, 'vic.log'))
    res['tv_log_tail'] = _tail(os.path.join(d, 'tv.log'))
    os.makedirs(os.path.join(RESULTS, 'rows'), exist_ok=True)
    with open(os.path.join(RESULTS, 'rows', r['id'] + '.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    return r['id'], res.get('grade', 'error'), st


def _tail(path, n=12):
    if not os.path.exists(path):
        return ''
    lines = open(path, errors='replace').read().splitlines()
    keep = [ln for ln in lines if any(k in ln for k in (
        'ERROR', 'Error', 'error', 'refus', 'not implemented', 'Traceback',
        'WARN', 'ran ', 'wind height', 'not run'))]
    return '\n'.join((keep or lines)[-n:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', default='')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--days', type=int, default=0)
    ap.add_argument('--rerun', action='store_true')
    ap.add_argument('--compare-only', action='store_true')
    a = ap.parse_args(argv)
    rows = M.MATRIX
    if a.only:
        keys = a.only.split(',')
        rows = [r for r in rows if any(k in r['id'] for k in keys)]
    print('%d rows, %d at a time' % (len(rows), a.jobs), flush=True)
    code = subprocess.run([PY, '-c', 'import torchvic; print(torchvic.__file__)'],
                          capture_output=True, text=True, env=_env(1),
                          cwd='/').stdout.strip()
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'code.json'), 'w') as f:
        json.dump(dict(commit=os.environ.get('VERIFY_SHA', 'working tree'),
                       torchvic=code), f)
    print('TorchVIC from', code, 'commit', os.environ.get('VERIFY_SHA'),
          flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(one, r, a): r['id'] for r in rows}
        for f in as_completed(futs):
            try:
                rid, grade, st = f.result()
                print('%-60s %-6s %s' % (rid, grade, st), flush=True)
            except Exception as e:                          # noqa: BLE001
                print('%-60s CRASH %s' % (futs[f], e), flush=True)


if __name__ == '__main__':
    main()
