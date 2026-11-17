#!/usr/bin/env python3
"""Matrix rows whose reference is VIC's *classic* driver.

    classic_rows.py [--only SUBSTR[,SUBSTR]] [--days N] [--list]

The image driver stops on COMPUTE_TREELINE (get_global_param.c:968-970), so
the main matrix (verification/matrix.py, vic_image.exe) cannot hold it. This
holds the rows that need `vic_classic.exe`: the same inputs go to the
pristine classic driver and to `torchvic run` (which reads a classic global
file through torchvic.io.classic), and the per-step histories are compared
with the main matrix's metrics and grades (verification/compare.py).

The classic executable is built from the pristine tree, in a scratch copy
(only the Makefile's CFLAGS gain -fcommon, which gcc >= 10 needs):

    cp -r reference/VIC-5.1.0/vic $WORK/item16b_refusedzero/build/vic
    cd .../drivers/classic && make          # CFLAGS += -fcommon

`$TORCHVIC_VIC_CLASSIC` names another build.

Cases (classic files, the two UW-Hydro samples):
  stehe     Stehekin, 16 cells, five snow bands, 10 January days, hourly.
            The soil file's last column (54) is the cell's July air
            temperature, 9.3..15.3 C, so JULY_TAVG_SUPPLIED TRUE reads it.
  findley   Findley Lake, one cell, one band, hourly (LAKES FALSE).
            Its soil file has no July column; one is appended.
Each row gives a mode (WB or EB), global-file lines and, where the input
must change, a soil or vegetation-parameter edit applied to both runs.

The rows' histories are written to WORK/classic_rows/<id>/ and the
comparison to verification/results/classic_rows/<id>.json.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix as M                                          # noqa: E402
import compare as C                                         # noqa: E402

REPO = os.path.dirname(HERE)
WORK = os.path.join(M.WORK, 'item16b_refusedzero')
OUT = os.path.join(M.WORK, 'classic_rows')
RESULTS = os.path.join(HERE, 'results', 'classic_rows')
VIC_CLASSIC = os.environ.get(
    'TORCHVIC_VIC_CLASSIC',
    os.path.join(WORK, 'build', 'vic', 'drivers', 'classic',
                 'vic_classic.exe'))
SAMPLES = os.path.join(REPO, 'data', 'samples', 'classic', 'Stehekin')
STEHE_P = os.path.join(SAMPLES, 'parameters')
FINDLEY = os.path.join(M.REPO, 'data', 'VIC_sample_data', 'classic',
                       'FindleyLake')
FIND_P = os.path.join(FINDLEY, 'parameters')

CASES = {
    'stehe': dict(
        soil=os.path.join(STEHE_P, 'Stehekin_soil.txt'),
        veglib=os.path.join(STEHE_P, 'Stehekin_veglib.txt'),
        vegparam=os.path.join(STEHE_P, 'Stehekin_vegparam.txt'),
        snowband=os.path.join(STEHE_P, 'Stehekin_snowbands.txt'),
        bands=5, forcing=os.path.join(SAMPLES, 'forcings', 'full_data_'),
        force_types=('PREC', 'AIR_TEMP', 'SWDOWN', 'LWDOWN', 'SKIP',
                     'PRESSURE', 'VP', 'WIND'),
        spd=24, force_start=(1949, 1, 1), start=(1949, 1, 1),
        end=(1949, 1, 10), july_col=True,
        extra=['ROOT_ZONES 3', 'VEGPARAM_LAI TRUE', 'LAI_SRC FROM_VEGPARAM',
               'GRID_DECIMAL 4', 'WIND_H 10.0']),
    'findley': dict(
        soil=os.path.join(FIND_P, 'soil_param.FindleyLake'),
        veglib=os.path.join(FIND_P, 'veg_lib_IGBP'),
        vegparam=os.path.join(FIND_P, 'veg_param.FindleyLake'),
        snowband=None, bands=1,
        forcing=os.path.join(FINDLEY, 'forcings',
                             'livneh_NAmerExt_15Oct2014.FindleyLake_'),
        force_types=('PREC', 'AIR_TEMP', 'SWDOWN', 'LWDOWN', 'PRESSURE', 'VP',
                     'WIND'),
        spd=24, force_start=(1970, 1, 1), start=(1970, 1, 1),
        end=(1970, 3, 31), july_col=False,
        extra=['ROOT_ZONES 3', 'VEGLIB_FCAN FALSE', 'VEGPARAM_LAI TRUE',
               'VEGPARAM_FCAN TRUE', 'VEGPARAM_ALB TRUE',
               'LAI_SRC FROM_VEGPARAM', 'FCAN_SRC FROM_VEGPARAM',
               'ALB_SRC FROM_VEGPARAM', 'EQUAL_AREA TRUE', 'RESOLUTION 1.29',
               'GRID_DECIMAL 4', 'WIND_H 10.0']),
}


# -- input edits ------------------------------------------------------------
def soil_july(value):
    """Append (or, when the file has it, replace) the July air temperature
    column of every soil row: value is a number or a callable(row)->number."""
    def edit(text, case):
        out = []
        for ln in text.splitlines():
            f = ln.split()
            if not f:
                continue
            if case['july_col']:
                f = f[:53]
            v = value(f) if callable(value) else value
            out.append(' '.join(f + ['%g' % v]))
        return '\n'.join(out) + '\n'
    return edit


def veg_overstory_only(cells):
    """Vegetation file with the non-overstory tiles of the named grid cells
    (classes 7-11 on Stehekin) dropped and the rest renormalised to cover 1,
    so the treeline's own tile (read_vegparam.c:371-433) is made for them."""
    drop = {7, 8, 9, 10, 11}

    def edit(text, case):
        lines = text.splitlines()
        out, i = [], 0
        while i < len(lines):
            f = lines[i].split()
            gid, n = int(f[0]), int(f[1])
            tiles = [(lines[i + 1 + 2 * k], lines[i + 2 + 2 * k])
                     for k in range(n)]
            if gid in cells:
                tiles = [t for t in tiles if int(t[0].split()[0]) not in drop]
                tot = sum(float(t[0].split()[1]) for t in tiles)
                new = []
                for a, b in tiles:
                    g = a.split()
                    g[1] = '%.9f' % (float(g[1]) / tot)
                    new.append((' '.join(g), b))
                tiles = new
            out.append('%d %d' % (gid, len(tiles)))
            for a, b in tiles:
                out += [a, b]
            i += 1 + 2 * n
        return '\n'.join(out) + '\n'
    return edit


def row(rid, case, mode, tl, soil=None, veg=None, note='', extra=()):
    return dict(id=rid, case=case, mode=mode, treeline=tl, soil=soil,
                veg=veg, note=note, extra=list(extra))


MODES = {'WB': M.WB, 'EB': M.EB}

CLASSIC_ROWS = [
    # The base rows: no treeline. The treeline rows are graded beside these.
    row('stehe_classic_WB_base', 'stehe', 'WB', None,
        note='classic driver, Stehekin, water balance'),
    row('stehe_classic_EB_base', 'stehe', 'EB', None),
    row('findley_classic_WB_base', 'findley', 'WB', None),
    row('findley_classic_EB_base', 'findley', 'EB', None),
    # ATMOS_STEPS_PER_DAY (classic get_global_param.c:68-70, :811-856) only
    # sets atmos_dt, which the log prints: results are the base row's.
    row('stehe_classic_WB_atmos48', 'stehe', 'WB', None,
        extra=['ATMOS_STEPS_PER_DAY 48'],
        note='ATMOS_STEPS_PER_DAY 48: atmos_dt 1800 s, no effect'),
    row('stehe_classic_EB_atmos24', 'stehe', 'EB', None,
        extra=['ATMOS_STEPS_PER_DAY 24'], note='the classic default, spelt'),
    row('findley_classic_EB_atmos96', 'findley', 'EB', None,
        extra=['ATMOS_STEPS_PER_DAY 96'], note='15-minute atmos_dt, no effect'),
    # Stehekin: the soil file's own July_Tavg (9.3..15.3 C) lapsed by each
    # band's Tfactor (-5.1..+5.1 C): bands from the top of the range down
    # are above the treeline, in every cell.
    row('stehe_classic_WB_treeline_bare', 'stehe', 'WB', '-1',
        note='COMPUTE_TREELINE -1: no default tile above the treeline'),
    row('stehe_classic_EB_treeline_bare', 'stehe', 'EB', '-1'),
    # All-overstory cells (classes 1-6 only) in the file: the tile VIC makes
    # (bare soil, then class 7 = a non-overstory class).
    row('stehe_classic_WB_treeline_overstory_bare', 'stehe', 'WB', '-1',
        veg=veg_overstory_only({86340, 86804, 85875, 86339}),
        note='four cells with overstory only: 0.001 to bare soil'),
    row('stehe_classic_EB_treeline_overstory_class7', 'stehe', 'EB', '7',
        veg=veg_overstory_only({86340, 86804, 85875, 86339}),
        note='four cells with overstory only: 0.001 to class 7'),
    # One band at the cell's elevation (Tfactor 0): the July temperature is
    # the cell's. Findley is 82% overstory; July_Tavg 8 C puts it above the
    # treeline, 14 C below.
    row('findley_classic_WB_treeline_above', 'findley', 'WB', '-1',
        soil=soil_july(8.0), note='July_Tavg 8 C: above the treeline'),
    row('findley_classic_EB_treeline_above', 'findley', 'EB', '-1',
        soil=soil_july(8.0)),
    row('findley_classic_EB_treeline_below', 'findley', 'EB', '-1',
        soil=soil_july(14.0), note='July_Tavg 14 C: below, nothing changes'),
    row('findley_classic_EB_treeline_off999', 'findley', 'EB', '-1',
        soil=soil_july(-999.0),
        note='July_Tavg -999: VIC skips the cell (vic_force.c:292-296)'),
]


# -- running ----------------------------------------------------------------
def _sub(text, old, new):
    return text.replace(old, new)


def prepare(r, days=0):
    c = CASES[r['case']]
    d = os.path.join(OUT, r['id'])
    for s in ('vic', 'tv', 'in'):
        os.makedirs(os.path.join(d, s), exist_ok=True)

    def edit(path, fn, name):
        text = open(path).read()
        if fn is not None:
            text = fn(text, c)
        p = os.path.join(d, 'in', name)
        open(p, 'w').write(text)
        return p
    soil = edit(c['soil'], r['soil'], 'soil')
    veg = edit(c['vegparam'], r['veg'], 'vegparam')
    start, end = c['start'], c['end']
    if days:
        e = dt.date(*start) + dt.timedelta(days=days - 1)
        end = (e.year, e.month, e.day)
    L = ['MODEL_STEPS_PER_DAY %d' % c['spd'], 'SNOW_STEPS_PER_DAY %d' % c['spd'],
         'RUNOFF_STEPS_PER_DAY %d' % c['spd'],
         'STARTYEAR %d' % start[0], 'STARTMONTH %d' % start[1],
         'STARTDAY %d' % start[2], 'ENDYEAR %d' % end[0],
         'ENDMONTH %d' % end[1], 'ENDDAY %d' % end[2],
         'CALENDAR PROLEPTIC_GREGORIAN', 'NODES 3']
    L += ['%s %s' % kv for kv in MODES[r['mode']]]
    L += ['FORCING1 %s' % c['forcing'], 'FORCE_FORMAT ASCII']
    L += ['FORCE_TYPE %s' % t for t in c['force_types']]
    L += ['FORCE_STEPS_PER_DAY %d' % c['spd'],
          'FORCEYEAR %d' % c['force_start'][0],
          'FORCEMONTH %d' % c['force_start'][1],
          'FORCEDAY %d' % c['force_start'][2],
          'SOIL %s' % soil, 'VEGLIB %s' % c['veglib'],
          'VEGPARAM %s' % veg, 'BASEFLOW ARNO']
    if c['bands'] > 1:
        L.append('SNOW_BAND %d %s' % (c['bands'], c['snowband']))
    else:
        L.append('SNOW_BAND 1')
    L += c['extra']
    if r['treeline'] is not None:
        L += ['COMPUTE_TREELINE %s' % r['treeline'], 'JULY_TAVG_SUPPLIED TRUE']
    L += r['extra']
    names = M.outvars()
    return d, L, names


def vic_global(d, L, names):
    X = list(L) + ['RESULT_DIR %s/' % os.path.join(d, 'vic'),
                   'LOG_DIR %s/' % os.path.join(d, 'vic'),
                   'OUTFILE fluxes', 'AGGFREQ NSTEPS 1']
    X += ['OUTVAR %s %%.17g OUT_TYPE_DOUBLE 1' % v for v in names]
    p = os.path.join(d, 'global.vic.txt')
    open(p, 'w').write('\n'.join(X) + '\n')
    return p


def run_vic(d, g):
    with open(os.path.join(d, 'vic.log'), 'w') as log:
        p = subprocess.run([VIC_CLASSIC, '-g', g], stdout=log,
                           stderr=subprocess.STDOUT, cwd=d)
    return p.returncode


def read_vic(d, lat, lon, decimals=4):
    """{name: array[time, cell, k]} from the classic ASCII files."""
    fmt = '%%.%df' % decimals
    cols, per_cell = None, []
    for la, lo in zip(lat, lon):
        f = os.path.join(d, 'vic', 'fluxes_%s_%s.txt' % (fmt % la, fmt % lo))
        with open(f) as fh:
            hdr = None
            for ln in fh:
                if ln.startswith('#'):
                    continue
                hdr = ln.split()
                break
        a = np.loadtxt(f, comments='#', skiprows=3, ndmin=2)
        if cols is None:
            cols = hdr
        per_cell.append(a)
    data = np.stack(per_cell, axis=1)               # time, cell, col
    out = {}
    for j, h in enumerate(cols):
        m = re.match(r'(OUT_\w+?)(?:_(\d+))?$', h)
        if not m or h in ('YEAR', 'MONTH', 'DAY', 'SEC'):
            continue
        base = h
        if m.group(2) is not None and h.rsplit('_', 1)[0] in (
                'OUT_SOIL_MOIST', 'OUT_SOIL_LIQ', 'OUT_SOIL_ICE',
                'OUT_SOIL_TEMP', 'OUT_SOIL_TNODE', 'OUT_ROOTMOIST',
                'OUT_SMLIQFRAC', 'OUT_SMFROZFRAC', 'OUT_SOIL_WET'):
            base = h.rsplit('_', 1)[0]
        out.setdefault(base, []).append(data[:, :, j])
    return {k: np.stack(v, axis=-1) for k, v in out.items()}


def run_tv(d, L, names, steps):
    g = os.path.join(d, 'global.tv.txt')
    open(g, 'w').write('\n'.join(L + ['RESULT_DIR %s/' % os.path.join(d, 'tv'),
                                      'OUTFILE fluxes', 'AGGFREQ NSTEPS 1']) +
                       '\n')
    import torch
    from torchvic import case as _case
    from torchvic.io import history
    with torch.inference_mode():
        cs = _case.load(g, device=torch.device('cpu'), dtype=torch.float64,
                        steps=steps, core='5.1.0', log=lambda *a: None)
        out = cs.run()
    series = {}
    for v in names:
        a, why = history.outvar_series(out, v)
        if a is None:
            continue
        a = a.detach().cpu().numpy().astype('f8')
        series[v] = a if a.ndim == 3 else a[..., None]
    return series, cs


def compare(vic, tv, cs, r):
    n = next(iter(vic.values())).shape[0]
    w = np.full(next(iter(vic.values())).shape[1],
                1.0 / next(iter(vic.values())).shape[1])
    times = ['step %d' % i for i in range(n)]
    rows, missing = [], []
    for v in sorted(vic):
        a = vic[v]
        if v not in tv:
            if np.nanmax(np.abs(a)) > 0:
                missing.append(v)
            continue
        b = tv[v]
        if a.shape != b.shape:
            rows.append(dict(var=v, error='shape %s vs %s' % (a.shape,
                                                              b.shape),
                             grade='fail', score=np.inf))
            continue
        for k in range(a.shape[-1]):
            name = v if a.shape[-1] == 1 else '%s[%d]' % (v, k)
            rows.append(C._metrics(name, v, a[..., k], b[..., k], w, times))
    for x in rows:
        x['group'] = C.group(x['var'])
    graded = [x for x in rows if x['var'].split('[')[0] not in C.NOT_GRADED]
    out = dict(n_steps=n, n_cells=int(w.size), vars=rows,
               not_written_by_torchvic=missing)
    order = ['pass', 'warn', 'fail']
    for g in ('water', 'energy', 'diag'):
        gr = [x for x in graded if x['group'] == g]
        worst = max(gr, key=lambda x: x.get('score', np.inf)) if gr else None
        out['worst_' + g] = worst['var'] if worst else None
        out['score_' + g] = worst.get('score') if worst else None
        out['grade_' + g] = C.grade(worst.get('score')) if worst else 'n/a'
    out['grade'] = order[max(order.index(out['grade_' + k])
                             for k in ('water', 'energy')
                             if out['grade_' + k] in order)]
    return out


def one(r, days=0):
    c = CASES[r['case']]
    d, L, names = prepare(r, days)
    g = vic_global(d, L, names)
    rc = run_vic(d, g)
    if rc != 0:
        return dict(row=r['id'], grade='VIC rc=%d' % rc,
                    vic_log=open(os.path.join(d, 'vic.log')).read()[-600:])
    start, end = c['start'], c['end']
    nd = (dt.date(*end) - dt.date(*start)).days + 1
    if days:
        nd = min(nd, days)
    import torch                                            # noqa: F401
    from torchvic.io import classic
    from torchvic.io.global_param import GlobalParam
    tv, cs = run_tv(d, L, names, nd * c['spd'])
    lat, lon = np.asarray(cs.dom.lat), np.asarray(cs.dom.lon)
    vic = read_vic(d, lat, lon)
    res = compare(vic, tv, cs, r)
    res['row'] = {k: r[k] for k in ('id', 'case', 'mode', 'treeline', 'note')}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--only', default='')
    ap.add_argument('--days', type=int, default=0)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    # official rows run with VIC's glibc transcendentals (torchvic/core/
    # xmath.py, CPU); TORCHVIC_EXACT_MATH=0 in the environment runs the
    # default kernels. Set here, not at import, so importing this module
    # (the tests do) leaves the process's mode alone.
    if os.environ.setdefault('TORCHVIC_EXACT_MATH', '1') == '1':
        os.environ['ATEN_CPU_CAPABILITY'] = 'default'
        from torchvic.core import xmath
        xmath.enable()
    rows = CLASSIC_ROWS
    if a.only:
        ks = a.only.split(',')
        rows = [r for r in rows if any(k in r['id'] for k in ks)]
    if a.list:
        for r in rows:
            print(r['id'])
        return 0
    os.makedirs(RESULTS, exist_ok=True)
    for r in rows:
        res = one(r, a.days)
        with open(os.path.join(RESULTS, r['id'] + '.json'), 'w') as f:
            json.dump(res, f, indent=1, default=float)
        print('%-52s %-5s water %.2e (%s)  energy %.2e (%s)  not written: %s'
              % (r['id'], res.get('grade'), res.get('score_water') or 0,
                 res.get('worst_water'), res.get('score_energy') or 0,
                 res.get('worst_energy'),
                 ','.join(res.get('not_written_by_torchvic', [])) or '-'),
              flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
