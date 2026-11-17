"""Analyse the build-up matrix: sanity findings and gauge scores per step,
each step diffed against the previous one.

  python matrix.py [STEP ...]    reads out/<step>/series.npz, writes
  out/<step>/findings.{json,md}, gauges.csv and MATRIX.md in isc_campaign/.

A finding is keyed (module, variable, check). Against the previous step it is
  new        not present before: introduced by the module the step adds;
  worse      present, cells x 1.5 or more (or severity up);
  same / better otherwise.
The 'introduced_by' of a finding is the first step it appears in along the
chain (S6x and S6f both follow S5; S7 follows S6f).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gauges as G                                           # noqa
import sanity as SA                                          # noqa
import steps as S                                            # noqa


def key(f):
    return (f['module'], f['variable'], f['check'])


def analyse(step, force=False):
    d = S.OUT + 'out/%s/' % step
    fj = d + 'findings.json'
    if force or not os.path.exists(fj):
        F = SA.run_dir_check(d)
    else:
        F = json.load(open(fj))
    gj = d + 'gauges.csv'
    if force or not os.path.exists(gj):
        z = np.load(d + 'series.npz')
        if 'vicwur_routing.discharge' in z.files:
            G.score(z['vicwur_routing.discharge']).to_csv(gj, index=False)
    return F


def main(steps):
    allF, chain = {}, {}
    sev = {'high': 3, 'medium': 2, 'low': 1}
    out = ['# ISC build-up matrix (2005 after a 92-day spin-up)\n']
    for st in steps:
        if not os.path.exists(S.OUT + 'out/%s/series.npz' % st):
            out.append('## %s: not run\n' % st)
            continue
        F = analyse(st)
        allF[st] = F
        prev = S.PREV.get(st)
        pk = {key(f): f for f in allF.get(prev, [])}
        rows = []
        for f in F:
            p = pk.get(key(f))
            if p is None:
                tag = 'new'
            elif f['cells'] >= 1.5 * max(p['cells'], 1) or \
                    sev[f['severity']] > sev[p['severity']]:
                tag = 'worse'
            else:
                tag = 'same'
            first = chain.setdefault(key(f), st) if tag == 'new' else \
                chain.get(key(f), st)
            if tag == 'new':
                chain[key(f)] = st
            f['vs_prev'] = tag
            f['introduced_by'] = chain[key(f)]
            rows.append(f)
        gone = [k for k in pk if k not in {key(f) for f in F}]
        meta = json.load(open(S.OUT + 'out/%s/meta.json' % st))
        out.append('## %s  (+%s; mode %s; wall %.0f s; ledger all_ok=%s)\n' % (
            st, S.ADDS.get(st, ''), meta['mode'], meta['wall_s'],
            meta['ledger']['all_ok']))
        gp = S.OUT + 'out/%s/gauges.csv' % st
        if os.path.exists(gp):
            sm = G.summary(pd.read_csv(gp))
            out.append('gauges: %d scored, %d with catchment ratio 0.5-2: '
                       'median KGE %.2f, KGE>0 %.0f%%, median bias %.0f%%, '
                       'median |peak lag| %.0f d, gauges with negative Q %d\n'
                       % (sm['n'], sm['n_matched'], sm['kge_median'],
                          100 * sm['kge_pos'], sm['bias_median'],
                          sm['peak_lag_abs_median'], sm['neg_gauges']))
        cols = ['severity', 'module', 'variable', 'check', 'cells',
                'first_day', 'vs_prev', 'introduced_by', 'detail']
        out.append('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols))
        for f in rows:
            out.append('| ' + ' | '.join(str(f[c]) for c in cols) + ' |')
        if gone:
            out.append('\nresolved vs %s: %s' % (prev, ', '.join(
                '/'.join(k) for k in gone)))
        out.append('')
    open(S.OUT + 'MATRIX.md', 'w').write('\n'.join(out))
    json.dump(allF, open(S.OUT + 'matrix_findings.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:] or list(S.STEPS))
