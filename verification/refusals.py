#!/usr/bin/env python3
"""Probe every VIC 5 option TorchVIC does not run as VIC does: is it
refused by name, announced and run anyway, or run silently?

    refusals.py [--out verification/results/refusals.csv]

Each probe is one change to a base matrix row, run by pristine VIC 5.1.0
for one day and by `torchvic run` for one day (no history). Outcome:

  VIC      accepted | rejected: <VIC's log_err message>
  TorchVIC refused-by-name   exit != 0 and the message names the option
           error              exit != 0 for another reason (quoted)
           announced          ran; the log says the option is not
                              implemented as VIC implements it
           no-effect          ran; the log names it as read to no effect
                              (torchvic/io/options.py says why)
           silent             ran; nothing in the log names the option
"""
import argparse
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix as M                                          # noqa: E402
from run_matrix import _env, PY                             # noqa: E402

FE = (('FULL_ENERGY', 'TRUE'),)


def P(key, value, base='wg', mode='EB', extra=(), daily=False, why=''):
    return dict(key=key, value=value, base=base, mode=mode, extra=extra,
                daily=daily, why=why)


PROBES = [
    P('LAKES', 'TRUE', base='findley', extra=(('LAKE_PROFILE', 'FALSE'),),
      why='lake/wetland tile'),
    P('LAKE_PROFILE', 'TRUE', base='findley', extra=(('LAKES', 'TRUE'),)),
    P('SPATIAL_FROST', 'TRUE 0.5', why='sub-grid frost'),
    P('CORRPREC', 'TRUE', why='gauge undercatch'),
    P('ORGANIC_FRACT', 'TRUE'),
    P('BULK_DENSITY_COMB', 'TRUE'),
    P('CARBON', 'TRUE', extra=(('VEGLIB_PHOTO', 'TRUE'),)),
    P('FROZEN_SOIL', 'TRUE', mode='EB',
      why='VIC turns QUICK_FLUX off: finite-difference profile'),
    P('QUICK_FLUX', 'FALSE', mode='EB',
      why='finite-difference soil temperature without frozen soil'),
    P('QUICK_SOLVE', 'TRUE', extra=(('QUICK_FLUX', 'FALSE'),)),
    P('IMPLICIT', 'FALSE', extra=(('QUICK_FLUX', 'FALSE'),)),
    P('EXP_TRANS', 'FALSE', extra=(('QUICK_FLUX', 'FALSE'),)),
    P('NOFLUX', 'TRUE', extra=(('QUICK_FLUX', 'FALSE'),)),
    P('TFALLBACK', 'FALSE'),
    P('CLOSE_ENERGY', 'TRUE'),
    P('MAX_SNOW_ALBEDO', 'TRUE'),
    P('FCAN_SRC', 'FROM_VEGPARAM', why='WG params carry no fcanopy'),
    P('JULY_TAVG_SUPPLIED', 'TRUE'),
    P('COMPUTE_TREELINE', 'TRUE', why='both drivers need JULY_TAVG_SUPPLIED '
      'too: stays refused. With it VIC\'s image driver still stops; '
      'TorchVIC runs it as the classic driver does '
      '(verification/classic_rows.py)'),
    # VIC stops on each of these; TorchVIC runs them
    P('ARNO_PARAMS', 'TRUE', why='4.2.d spelling: read as BASEFLOW NIJSSEN2001'),
    P('NIJSSEN2001_BASEFLOW', 'TRUE',
      why='4.2.d spelling: read as BASEFLOW NIJSSEN2001'),
    P('ATMOS_STEPS_PER_DAY', '24', why='classic key: validated, no effect'),
    P('OUTPUT_FORCE', 'TRUE', why='4.2.d: the forcing files, not the model'),
    P('INIT_STATE', '/nonexistent/state.nc'),
    P('STATENAME', '/tmp/vm_state', extra=(('STATEYEAR', '2008'),
                                          ('STATEMONTH', '1'),
                                          ('STATEDAY', '2'),
                                          ('STATESEC', '0'))),
    P('CONSTANTS', '@CONSTANTS@'),
    P('MODEL_STEPS_PER_DAY', '1', mode='WB',
      why='daily step given only in the global file, no --daily'),
    P('BLOWING', 'TRUE', mode='WB'),
]


def _last_wins(txt, keys):
    """Keep only the last line of each key in `keys`, in its place."""
    lines = txt.splitlines()
    out = []
    for i, ln in enumerate(lines):
        k = ln.split()[0] if ln.split() else ''
        if k in keys and any((x.split() or [''])[0] == k
                             for x in lines[i + 1:]):
            continue
        out.append(ln)
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'results',
                                                  'refusals.csv'))
    a = ap.parse_args(argv)
    root = os.path.join(M.WORK, 'probes')
    os.makedirs(root, exist_ok=True)
    consts = os.path.join(root, 'constants.txt')
    with open(consts, 'w') as f:
        f.write('SNOW_MAX_SURFACE_SWE 0.2\n')
    rows = []
    for i, p in enumerate(PROBES):
        pid = '%02d_%s_%s' % (i, p['key'], re.sub(r'\W', '', p['value'])[:20])
        d = os.path.join(root, pid)
        for s in ('vic', 'tv', 'logs'):
            os.makedirs(os.path.join(d, s), exist_ok=True)
            for f in os.listdir(os.path.join(d, s)):
                os.remove(os.path.join(d, s, f))
        c = dict(M.CASES[p['base']])
        c['end'] = c['start']
        M.CASES['_probe'] = c
        value = p['value'].replace('@WG_FORCING@', c['forcing']).replace(
            '@CONSTANTS@', consts)
        r = M.row(p['base'], p['mode'])
        r['case'] = '_probe'
        extra = ['%s %s' % kv for kv in p['extra']]
        r['opts'] = {}
        out = {}
        for who in ('vic', 'tv'):
            txt = M.global_text(r, os.path.join(d, who),
                                os.path.join(d, 'logs'),
                                ['OUT_PREC', 'OUT_EVAP', 'OUT_RUNOFF'],
                                extra_lines=extra + ['%s %s' % (p['key'],
                                                                value)])
            txt = _last_wins(txt, [p['key']] + [k for k, _ in p['extra']])
            g = os.path.join(d, 'global.%s.txt' % who)
            open(g, 'w').write(txt)
            if who == 'vic':
                pr = subprocess.run([M.VIC['5.1.0'], '-g', g],
                                    capture_output=True, text=True,
                                    env=_env(1), cwd=d)
                text = pr.stdout + pr.stderr
                for lf in sorted(os.listdir(os.path.join(d, 'logs'))):
                    text += open(os.path.join(d, 'logs', lf),
                                 errors='replace').read()
                msg = [ln.split('ERROR', 1)[-1].strip(' :]')
                       for ln in text.splitlines() if 'ERROR' in ln]
                msg = [m for m in msg if m and not m.startswith('in ')] or msg
                out['vic'] = ('accepted' if pr.returncode == 0 else
                              'rejected: ' + (' '.join(msg)[:200] or
                                              'exit %d' % pr.returncode))
            else:
                pr = subprocess.run([PY, '-m', 'torchvic', 'run', g, '--core',
                                     '5.1.0', '--device', 'cpu', '--steps',
                                     str(c['forcing_spd']), '--no-history'],
                                    capture_output=True, text=True,
                                    env=_env(2), cwd=d)
                text = pr.stdout + pr.stderr
                # the run directory's name carries the key, so a line
                # naming only a path (LOG_DIR ...) does not name the option
                named = [ln.strip() for ln in text.splitlines()
                         if p['key'] in ln.replace(d, '')]
                if pr.returncode != 0:
                    last = [ln for ln in text.splitlines() if ln.strip()]
                    err = ' '.join(named[-2:]) or (last[-1] if last else '')
                    out['tv'] = ('refused-by-name' if named else 'error')
                    out['tv_msg'] = err[:300]
                else:
                    ann = [ln for ln in named if 'not implemented' in ln
                           or 'not run' in ln or 'refus' in ln
                           or 'no effect' in ln]
                    out['tv'] = ('no-effect (announced)' if any(
                        'no effect' in ln for ln in ann) else
                        'announced' if ann else 'silent')
                    out['tv_msg'] = (ann or named or [''])[0][:300]
        rows.append(dict(probe=pid, option='%s %s' % (p['key'], p['value']),
                         base='%s %s' % (p['base'], p['mode']),
                         vic_5_1_0=out['vic'], torchvic=out['tv'],
                         torchvic_message=out.get('tv_msg', ''),
                         why=p['why']))
        print('%-36s VIC %-10s TorchVIC %-16s %s' % (
            rows[-1]['option'], out['vic'][:10], out['tv'],
            out.get('tv_msg', '')[:110]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
