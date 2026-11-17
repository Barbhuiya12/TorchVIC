#!/usr/bin/env python3
"""Build the verification scoreboard from results/rows/*.json,
results/refusals.csv and results/gpu_check.json:

  results/scoreboard.csv   one line per matrix row
  results/variables.csv    one line per row and output variable
  results/SCOREBOARD.md    the tables, thresholds stated
  results/fig_scores.png   log10 score of the key variables, every row
  results/fig_closure.png  water-balance residual, TorchVIC vs VIC
  results/fig_series.png   daily domain series, a snow-free and a snow row
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix as M                                          # noqa: E402
from compare import PASS, WARN, CLOSE_PASS, CLOSE_WARN      # noqa: E402

RES = os.path.join(HERE, 'results')
KEY = ['OUT_EVAP', 'OUT_TRANSP_VEG', 'OUT_EVAP_BARE', 'OUT_EVAP_CANOP',
       'OUT_SUB_SNOW', 'OUT_RUNOFF', 'OUT_BASEFLOW', 'OUT_SWE',
       'OUT_SNOW_DEPTH', 'OUT_SNOW_MELT', 'OUT_SOIL_MOIST[0]',
       'OUT_SOIL_MOIST[1]', 'OUT_SOIL_MOIST[2]', 'OUT_SOIL_ICE[0]',
       'OUT_R_NET', 'OUT_LATENT', 'OUT_SENSIBLE', 'OUT_GRND_FLUX',
       'OUT_SURF_TEMP', 'OUT_SOIL_TEMP[0]']


def _f(x, fmt='%.1e'):
    return '' if x is None else (fmt % x if isinstance(x, (int, float))
                                 else str(x))


def load_rows():
    rows = []
    order = {r['id']: i for i, r in enumerate(M.MATRIX)}
    for f in glob.glob(os.path.join(RES, 'rows', '*.json')):
        j = json.load(open(f))
        if j['row']['id'] not in order:
            continue
        st = j.get('status', {})
        if st.get('vic_rc') not in (0, None) and st.get('tv_rc') == 0:
            j['grade'] = 'VIC rejects'
        elif st.get('vic_rc') not in (0, None):
            j['grade'] = 'both reject'
        elif st.get('tv_rc') not in (0, None):
            j['grade'] = 'TorchVIC error'
            log = os.path.join(M.WORK, 'runs', j['row']['id'], 'tv.log')
            if os.path.exists(log):
                for ln in open(log, errors='replace'):
                    if ln.startswith('ValueError'):
                        j['grade'] = 'TorchVIC refuses'
                        j['tv_error'] = ln.strip()[:160]
                        break
        elif j.get('grade') == 'VIC NaN':
            pass
        elif 'vars' not in j:
            j['grade'] = 'fail'
            j['tv_error'] = j.get('error') or j.get('compare_error', '')
        rows.append(j)
    rows.sort(key=lambda j: order[j['row']['id']])
    return rows


def _first(j, var):
    for v in j.get('vars', []):
        if v['var'] == var and v.get('first'):
            return '%s (cell %d)' % (v['first']['date'], v['first']['cell'])
    return ''


def _earliest(j):
    """The water/energy variable that leaves VIC first (1e-6 relative),
    among those that do not pass: where a diagnosis starts."""
    c = [v for v in j.get('vars', []) if v.get('first') and v['grade'] != 'pass'
         and v.get('group') in ('water', 'energy')]
    if not c:
        return ''
    v = min(c, key=lambda v: (v['first']['step'], -v['score']))
    f = v['first']
    return '%s %s cell %d (VIC %.4g, TV %.4g)' % (
        v['var'].replace('OUT_', ''), f['date'], f['cell'], f['vic'], f['tv'])


def _vic_err(j):
    t = j.get('vic_log_tail', '')
    for ln in t.splitlines():
        if 'ERROR' in ln:
            return ln.split('errno: None:')[-1].split('errno:')[-1].strip()[:110]
    d = os.path.join(M.WORK, 'runs', j['row']['id'], 'vic_logs')
    if os.path.isdir(d):
        for f in os.listdir(d):
            for ln in open(os.path.join(d, f), errors='replace'):
                if ln.startswith('[ERROR]'):
                    return ln.split(':', 3)[-1].strip()[:110]
    return ''


def main():
    rows = load_rows()
    sb = []
    for j in rows:
        r = j['row']
        c = j.get('closure', {})
        sb.append(dict(
            id=r['id'], case=r['case'], mode=r['mode'], core=r['core'],
            step='24h' if r['daily'] else '%dh' % (24 // M.CASES[r['case']]
                                                   ['forcing_spd']),
            options=' '.join('%s=%s' % kv for kv in sorted(r['opts'].items())),
            expect=r['expect'], grade=j.get('grade'),
            grade_water=j.get('grade_water'), worst_water=j.get('worst_water'),
            score_water=j.get('score_water'),
            first_water=_first(j, j.get('worst_water')),
            earliest=_earliest(j),
            grade_energy=j.get('grade_energy'),
            worst_energy=j.get('worst_energy'),
            score_energy=j.get('score_energy'),
            first_energy=_first(j, j.get('worst_energy')),
            grade_diag=j.get('grade_diag'), worst_diag=j.get('worst_diag'),
            score_diag=j.get('score_diag'),
            grade_closure=j.get('grade_closure'),
            tv_water_resid=c.get('tv_water_resid_max'),
            vic_water_resid=c.get('vic_water_resid_max'),
            closure_inherited=j.get('closure_inherited'),
            tv_root_resid=c.get('tv_root_resid_max'),
            tv_putdata_energy_max=c.get('tv_putdata_energy_max'),
            vic_putdata_energy_max=c.get('vic_putdata_energy_max'),
            not_written_by_torchvic=' '.join(j.get('not_written_by_torchvic',
                                                   [])),
            vic_s=j.get('status', {}).get('vic_s'),
            tv_s=j.get('status', {}).get('tv_s'),
            note=r.get('note', ''), vic_error=(
                _vic_err(j) if j.get('grade') in ('VIC rejects', 'both reject')
                else j.get('tv_error', '') if j.get('grade') ==
                'TorchVIC refuses' or 'vars' not in j and j.get('grade') ==
                'fail' else 'VIC 5 run is NaN from %s'
                % j.get('first_vic_nan') if j.get('grade') == 'VIC NaN'
                else '')))
    with open(os.path.join(RES, 'scoreboard.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sb[0]))
        w.writeheader()
        w.writerows(sb)
    with open(os.path.join(RES, 'variables.csv'), 'w', newline='') as f:
        fields = ['id', 'var', 'group', 'grade', 'vic_total', 'tv_total',
                  'rel_total', 'rel_cell', 'max_abs', 'rmse', 'first_step',
                  'first_date', 'first_cell', 'nan_mismatch']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for j in rows:
            for v in j.get('vars', []):
                fi = v.get('first') or {}
                w.writerow(dict(id=j['row']['id'], var=v['var'],
                                group=v.get('group'), grade=v['grade'],
                                vic_total=v.get('vic_total'),
                                tv_total=v.get('tv_total'),
                                rel_total=v.get('rel_total'),
                                rel_cell=v.get('rel_cell'),
                                max_abs=v.get('max_abs'), rmse=v.get('rmse'),
                                first_step=fi.get('step'),
                                first_date=fi.get('date'),
                                first_cell=fi.get('cell'),
                                nan_mismatch=v.get('nan_mismatch')))
    figures(rows)
    markdown(sb, rows)


def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ok = [j for j in rows if 'vars' in j]
    S = np.full((len(ok), len(KEY)), np.nan)
    for i, j in enumerate(ok):
        d = {v['var']: v['score'] for v in j['vars']}
        for k, name in enumerate(KEY):
            if name in d:
                S[i, k] = np.log10(max(d[name], 1e-16))
    fig, ax = plt.subplots(figsize=(11, 0.17 * len(ok) + 2))
    im = ax.imshow(S, aspect='auto', cmap='RdYlGn_r', vmin=-12, vmax=0,
                   interpolation='nearest')
    ax.set_xticks(range(len(KEY)))
    ax.set_xticklabels([k.replace('OUT_', '') for k in KEY], rotation=70,
                       fontsize=7)
    ax.set_yticks(range(len(ok)))
    ax.set_yticklabels([j['row']['id'] for j in ok], fontsize=5.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label('log10 relative difference of period totals\n'
                 '(max of domain and per-cell); pass <= -4, warn <= -2')
    ax.set_title('TorchVIC vs pristine VIC: every matrix row, key variables')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_scores.png'), dpi=150)
    plt.close(fig)
    series(plt)
    tv = [j['closure'].get('tv_water_resid_max') or 1e-17 for j in ok]
    vc = [j['closure'].get('vic_water_resid_max') or 1e-17 for j in ok]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(vc, tv, 'o', ms=4)
    lim = [1e-15, 1e1]
    ax.plot(lim, lim, 'k:', lw=0.8)
    ax.axhline(CLOSE_PASS, color='g', lw=0.8, ls='--')
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel('VIC: max |water-balance residual| per cell-step (mm)')
    ax.set_ylabel('TorchVIC: same (mm)')
    ax.set_title('Water-balance closure, every matrix row')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_closure.png'), dpi=150)
    plt.close(fig)


def classic_section():
    """Rows whose reference is VIC's classic driver (verification/
    classic_rows.py, results/classic_rows/*.json)."""
    fs = sorted(glob.glob(os.path.join(RES, 'classic_rows', '*.json')))
    if not fs:
        return []
    L = ['', '## Classic-driver rows', '',
         'Pristine `vic_classic.exe` 5.1.0 against `torchvic run` on the '
         'same classic global file (verification/classic_rows.py): '
         'COMPUTE_TREELINE, JULY_TAVG_SUPPLIED and ATMOS_STEPS_PER_DAY, '
         'which the image driver stops on.', '',
         '| row | grade | water (worst, score) | energy (worst, score) | '
         'diag |', '|---|---|---|---|---|']
    for f in fs:
        j = json.load(open(f))
        if 'grade_water' not in j:
            L.append('| %s | **%s** | %s |  |  |' % (
                j['row']['id'], j.get('grade', 'error'),
                str(j.get('error') or j.get('compare_error', ''))[:120]))
            continue
        L.append('| %s | **%s** | %s %s %s | %s %s %s | %s %s |' % (
            j['row']['id'], j['grade'], j['grade_water'],
            (j['worst_water'] or '').replace('OUT_', ''),
            _f(j['score_water']), j['grade_energy'],
            (j['worst_energy'] or '').replace('OUT_', ''),
            _f(j['score_energy']), j['grade_diag'],
            (j['worst_diag'] or '').replace('OUT_', '')))
    return L


def _commit():
    try:
        return json.load(open(os.path.join(RES, 'code.json')))['commit']
    except (IOError, ValueError, KeyError):
        return 'unknown'


def series(plt, rows=(('igp_EB_510_3h', 'OUT_EVAP'),
                       ('himal_WB_510_3h', 'OUT_SWE'),
                       ('himal_EB_510_3h', 'OUT_EVAP'))):
    from compare import _open, _cells
    fig, axs = plt.subplots(len(rows), 1, figsize=(8, 2.4 * len(rows)))
    for ax, (rid, v) in zip(axs, rows):
        d = os.path.join(M.WORK, 'runs', rid)
        try:
            a, b = _open(os.path.join(d, 'vic')), _open(os.path.join(d, 'tv'))
        except IOError:
            continue
        m = np.isfinite(a['OUT_PREC'].values[0])
        spd = 24 // int(round((a['time'].values[1] - a['time'].values[0])
                              / np.timedelta64(1, 'h')))
        for ds, lab, st in ((a, 'VIC', '-'), (b, 'TorchVIC', '--')):
            x = _cells(ds, v, m)[..., 0].mean(1)
            x = x[:len(x) // spd * spd].reshape(-1, spd)
            x = x.sum(1) if v != 'OUT_SWE' else x.mean(1)
            ax.plot(x, st, lw=1, label=lab)
        ax.set_title('%s, %s (domain mean, daily)' % (rid, v), fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_series.png'), dpi=130)
    plt.close(fig)


def markdown(sb, rows):
    L = ['# TorchVIC verification scoreboard', '',
         'TorchVIC against pristine VIC 5.1.0 and 5.0.1 image drivers '
         '(`reference/`), every output variable at every model step, in '
         'double precision. TorchVIC commit: **%s** (a snapshot of the '
         'commit, not the working tree). Regenerate with `verification/run_all.sh` '
         '(CPU) and `verification/gpu_check.py` (one GPU).'
         % _commit(), '',
         'Official rows run with VIC\'s glibc transcendentals '
         '(TORCHVIC_EXACT_MATH=1, CPU: torchvic/core/xmath.py); the default '
         'and GPU paths may differ in the last bit.', '',
         '## Thresholds', '',
         '* A variable\'s **score** is the larger of the relative difference '
         'of the area-weighted domain period total (summed fluxes) or mean '
         '(states, temperatures in K) and the largest per-cell relative '
         'difference; the denominator is max(|VIC|, mean |VIC|), floored at '
         '1e-8 mm for variables in mm and 1e-12 otherwise '
         '(verification/compare.py abs_floor).',
         '* LAKES TRUE rows do not grade OUT_EVAP_BARE: VIC never writes the '
         'lake soil\'s esoil and reports uninitialised memory there '
         '(compare.py NOT_GRADED_LAKES).',
         '* The QUICK_FLUX FALSE rows with 12 nodes run 14 days; the other '
         'rows run their whole period.',
         '* **pass** score <= %g (0.01 %%); **warn** <= %g (1 %%); **fail** '
         'otherwise.' % (PASS, WARN),
         '* **Closure**: TorchVIC\'s water-balance residual P - E - R - B - '
         'dS per cell per step: pass <= %g mm; warn <= %g mm, or no larger '
         'than VIC\'s own residual on the same row (inherited); fail '
         'otherwise.' % (CLOSE_PASS, CLOSE_WARN),
         '* A row\'s grade is the worst of its **water** variables, its '
         '**energy** variables and its closure. **diag** variables '
         '(aerodynamic resistance/conductance, PET, OUT_ALBEDO, forcing '
         'echoes, LAI, fcanopy) are graded separately and do not set it.',
         '* `expect`: `supported` is a configuration TorchVIC claims to run '
         'as VIC does; `gap` is a documented or suspected departure, run to '
         'measure it.', '']
    from collections import Counter
    cnt = Counter((s['expect'], s['grade']) for s in sb)
    L += ['## Summary', '', '| expect | pass | warn | fail | other |',
          '|---|---|---|---|---|']
    for e in ('supported', 'gap'):
        oth = sum(v for (x, g), v in cnt.items()
                  if x == e and g not in ('pass', 'warn', 'fail'))
        L.append('| %s | %d | %d | %d | %d |' % (
            e, cnt[(e, 'pass')], cnt[(e, 'warn')], cnt[(e, 'fail')], oth))
    L += ['', '![scores](fig_scores.png)', '',
          '## Matrix', '',
          '| row | expect | grade | water (worst, score) | energy (worst, '
          'score) | diag | closure TV / VIC (mm) | earliest diverging '
          'water/energy variable |',
          '|---|---|---|---|---|---|---|---|']
    for s in sb:
        if s['grade'] in ('VIC rejects', 'both reject', 'TorchVIC error',
                          'TorchVIC refuses', 'VIC NaN') or (
                              s['grade_water'] is None):
            L.append('| %s | %s | **%s** | %s |  |  |  |  |' % (
                s['id'], s['expect'], s['grade'], s['vic_error']))
            continue
        L.append('| %s | %s | **%s** | %s %s %s | %s %s %s | %s %s | %s / %s'
                 '%s | %s |' % (
                     s['id'], s['expect'], s['grade'], s['grade_water'],
                     (s['worst_water'] or '').replace('OUT_', ''),
                     _f(s['score_water']), s['grade_energy'],
                     (s['worst_energy'] or '').replace('OUT_', ''),
                     _f(s['score_energy']), s['grade_diag'],
                     (s['worst_diag'] or '').replace('OUT_', ''),
                     _f(s['tv_water_resid']), _f(s['vic_water_resid']),
                     ' (inherited)' if s['closure_inherited'] else '',
                     s['earliest']))
    L += classic_section()
    L += ['', '![series](fig_series.png)', '',
          '## Energy-balance closure (FULL_ENERGY rows)', '',
          'TorchVIC\'s `OUT_ENERGY_ERROR` is by default (`energy_error="solver"`) '
          'the residual of the surface energy balance at its root. VIC\'s '
          '`OUT_ENERGY_ERROR` is put_data.c\'s sum of fluxes, which mixes '
          'control volumes (snow pack, canopy, ground); the same expression '
          'is evaluated on both models\' outputs for reference. Where the '
          'root\'s residual is not the balance the run closes (CLOSE_ENERGY, '
          'SPATIAL_SNOW, a daily model step with snow sub-steps) TorchVIC '
          'itself writes put_data\'s expression of its own rows as '
          '`OUT_ENERGY_ERROR`: its first column then equals the second.', '',
          '| row | TorchVIC OUT_ENERGY_ERROR max (W m-2) | put_data expr, '
          'TorchVIC max | put_data expr, VIC max |', '|---|---|---|---|']
    for s in sb:
        if s['tv_putdata_energy_max'] is not None:
            L.append('| %s | %s | %s | %s |' % (
                s['id'], _f(s['tv_root_resid']),
                _f(s['tv_putdata_energy_max']),
                _f(s['vic_putdata_energy_max'])))
    L += ['', '![closure](fig_closure.png)', '']
    sen = os.path.join(RES, 'sensitivity.csv')
    if os.path.exists(sen):
        L += ['## Option sensitivity: does the option move VIC, and TorchVIC?',
              '', 'Relative change of the domain period total (`sensitivity.py`)'
              ': VIC(option) vs VIC(base), TorchVIC(option) vs TorchVIC(base), '
              'and TorchVIC vs VIC on the option row. `ignored` = the option '
              'moves VIC and leaves TorchVIC bit-identical.', '',
              '| row | var | VIC effect | TorchVIC effect | TorchVIC vs VIC | '
              'ignored |', '|---|---|---|---|---|---|']
        for p in csv.DictReader(open(sen)):
            if p['var'] not in ('OUT_EVAP', 'OUT_RUNOFF', 'OUT_BASEFLOW',
                                'OUT_SWE', 'OUT_SOIL_MOIST'):
                continue
            L.append('| %s | %s | %+.2e | %+.2e | %+.2e | %s |' % (
                p['row'], p['var'].replace('OUT_', ''), float(p['vic_effect']),
                float(p['tv_effect']), float(p['tv_vs_vic']),
                '**yes**' if p['tv_ignores'] == 'True' else ''))
        L.append('')
    ref = os.path.join(RES, 'refusals.csv')
    if os.path.exists(ref):
        L += ['## Options TorchVIC does not run as VIC does', '',
              'Each probe changes one option of a base row and runs both '
              'models for one day (`verification/refusals.py`). TorchVIC '
              'outcomes: refused-by-name (exit, message names the option), '
              'announced (runs, logs "not implemented as VIC implements '
              'it"), silent (runs, nothing names the option).', '',
              '| option | base | VIC 5.1.0 | TorchVIC | TorchVIC message |',
              '|---|---|---|---|---|']
        for p in csv.DictReader(open(ref)):
            L.append('| %s | %s | %s | **%s** | %s |' % (
                p['option'].replace(M.WORK, '...'), p['base'],
                p['vic_5_1_0'].replace('|', '/')[:90], p['torchvic'],
                p['torchvic_message'].replace('|', '/').replace(
                    M.WORK, '...')[:120]))
        L.append('')
    gpu = os.path.join(RES, 'gpu_check.json')
    if os.path.exists(gpu):
        g = json.load(open(gpu))
        L += ['## TorchVIC CPU vs GPU (one RTX 6000 Ada)', '',
              'Same commit, float64, every variable at every step. max rel = '
              'max |GPU - CPU| / mean |CPU| of the variable (residual '
              'diagnostics excepted).', '',
              '| row | variables | bit-identical | max abs diff | max rel diff '
              '| worst |', '|---|---|---|---|---|---|']
        for k, v in g.items():
            L.append('| %s | %d | %d | %s | %s | %s |' % (
                k, v['n_vars'], v['n_identical'], _f(v['max_abs']),
                _f(v['max_rel']), v.get('worst') or ''))
        L.append('')
    with open(os.path.join(RES, 'SCOREBOARD.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', os.path.join(RES, 'SCOREBOARD.md'))


if __name__ == '__main__':
    main()
