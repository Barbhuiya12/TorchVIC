#!/usr/bin/env python
"""G-run configs, sbatch scripts and run checker for VIC-WUR on R1 / R1b.

  gruns.py gen   [--domain R1|R1b] [--outdir DIR]   write global files + sbatch
  gruns.py check [--domain R1|R1b] [--outdir DIR] [RUN ...]
                                    summarise finished runs (rc, log, outputs)

Run matrix (see `runs()` below): every run is daily
(MODEL_STEPS_PER_DAY 1), BASIN decomposition, mpi 1, all plugin OUTVARs of
the switched-on plugins, state at the end.  Runs are cumulative (G2 = G1 +
DAMS, ...).  Each run is executed once with both builds: VIC-WUR 3.0 (tag W3,
<data root>/build/vicwur3) and W (git master, <data root>/build/vicwur_git).
G11 (GWM) needs VIC-WUR 3.0 only; W has no GWM.

Layout: <outdir>/gruns/<domain>/<run>/{global_<W3|W>.txt, run.sbatch,
<W3|W>/{out,state,log}}.  Files of R1 are used by R1b except domain.nc.
"""
import argparse
import glob
import os
import re
import sys

VT = os.environ.get('TORCHVIC_WORK_ROOT', '.')
DATA_ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
ROOT = DATA_ROOT
DATA = ROOT + '/vicwur'
ENVNC = os.environ.get('NETCDF_HOME', '/usr')
BUILDS = {'W3': ROOT + '/build/vicwur3/vic_image.exe',
          'W': ROOT + '/build/vicwur_git/vic_image.exe'}
START = (2003, 7, 1)
END = (2010, 6, 30)
SPLIT = (2007, 1, 1)          # G12 restart date (a month boundary)
PBARE = 18
R1 = DATA + '/R1'
F = R1 + '/forcing'

MET = [('AIR_TEMP', 'tair'), ('PREC', 'prcp'),
       ('PRESSURE', 'sp'), ('SWDOWN', 'swdown'), ('LWDOWN', 'lwdown'),
       ('VP', 'vp'), ('WIND', 'wind')]
CORE_OUT = ['OUT_PREC', 'OUT_EVAP', 'OUT_RUNOFF', 'OUT_BASEFLOW', 'OUT_SWE',
            'OUT_SOIL_MOIST', 'OUT_TRANSP_VEG', 'OUT_EVAP_CANOP',
            'OUT_EVAP_BARE', 'OUT_AIR_TEMP']
OUT_ROUT = ['OUT_DISCHARGE', 'OUT_STREAM_INFLOW', 'OUT_STREAM_RUNOFF',
            'OUT_STREAM_MOIST', 'OUT_ROUTING_ERROR']
OUT_NONREN = ['OUT_NONREN_DEFICIT']
OUT_DAM = ['OUT_%sDAM_%s' % (p, v) for p in ('G', 'L') for v in
           ('INFLOW', 'DEMAND', 'EFR', 'STORAGE', 'RELEASE', 'HIST_INFLOW',
            'HIST_DEMAND', 'HIST_EFR', 'OP_STORAGE', 'OP_RELEASE')]
OUT_WU = ['OUT_DE_GW_SECT', 'OUT_DE_SURF_SECT', 'OUT_DE_NREN_SECT',
          'OUT_DE_REM_SECT', 'OUT_DE_TREM_SECT', 'OUT_WI_GW_SECT',
          'OUT_WI_SURF_SECT', 'OUT_WI_DAM_SECT', 'OUT_WI_REM_SECT',
          'OUT_WI_TREM_SECT', 'OUT_WI_NREN_SECT', 'OUT_WITHDRAWN',
          'OUT_DEMAND', 'OUT_RETURNED', 'OUT_CONSUMED']
OUT_IRR = ['OUT_SHORTAGE', 'OUT_REQUIREMENT', 'OUT_RECEIVED', 'OUT_LEFTOVER',
           'OUT_APPLIED']
OUT_EFR = ['OUT_EFR_DISCHARGE', 'OUT_EFR_BASEFLOW', 'OUT_EFR_MOIST']
OUT_LU = ['OUT_CV']
OUT_CROP = ['OUT_CROP_' + s for s in (
    'CULTIVATE GROW EVAP EVAP_BARE EVAP_CANOP TRANSP_VEG DVS WLV WST WSO WRT '
    'WDLV WDST WDRT NLV NST NSO NRT LAI WSTRESS NSTRESS TMAXSTRESS '
    'TMINSTRESS NPKI NNI NSOIL NUPT NDEM PNI PSOIL PUPT PDEM KNI KSOIL KUPT '
    'KDEM ROOTDEPTH').split()]

# cumulative feature switches in matrix order
BASE = dict(share=True, rout=0, dams=False, wu=False, wu_sectors=('mun',),
            irr=False, irr_file='irr.nc', irr_opts=(), remote=False,
            comp=False, nonren=False, nonren_runoff=False, nren_lim=None,
            efr=False, landuse=False, wofost=False, wofost_opts=(),
            co2=False, gwm=False, local=True)


def runs():
    """Ordered dict run -> overrides to BASE (cumulative where noted)."""
    R = {}
    R['G0a'] = dict(share=True)
    R['G0b'] = dict(share=False)
    g1 = dict(rout=1)
    R['G1a'] = dict(g1, rout=1)
    R['G1b'] = dict(g1, rout=4)
    g2 = dict(g1, rout=4, dams=True)
    R['G2'] = g2
    g3 = dict(g2, wu=True)
    R['G3'] = g3
    g4 = dict(g3, irr=True)
    R['G4'] = g4
    R['G4e'] = dict(g4, irr_opts=('EFFICIENT_IRRIGATION',))
    R['G4o'] = dict(g4, irr_opts=('OFFSET_IRRIGATION',))
    R['G4p'] = dict(g4, irr_opts=('POTENTIAL_IRRIGATION',))
    R['G4d'] = dict(g4, irr_file='irr_paddy.nc')
    g5 = dict(g4, remote=True)
    R['G5'] = g5
    g6 = dict(g5, nonren=True)
    R['G6a'] = g6
    R['G6b'] = dict(g6, nonren_runoff=True)
    R['G6c'] = dict(g6, nonren_runoff=True, nren_lim=20.0)
    g7 = dict(g6, efr=True)
    R['G7'] = g7
    g8 = dict(g7, landuse=True)
    R['G8'] = g8
    g9 = dict(g8, wofost=True, co2=True)
    R['G9'] = g9
    R['G9b'] = dict(g9, wofost_opts=('WOFOST_CALC_MINERALIZATION',))
    g10 = dict(g9, wu_sectors=('mun', 'liv', 'ene', 'man'), comp=True,
               irr_opts=('EFFICIENT_IRRIGATION', 'OFFSET_IRRIGATION'))
    R['G10'] = g10
    R['G11'] = dict(gwm=True)
    # G11 + routing, water use, non-renewable: the run of the VIC-WUR 3.0
    # coupled MODFLOW loop (tools/vicwur/gw_runs.py adds the MODFLOW forcing)
    R['G11c'] = dict(gwm=True, rout=1, wu=True, nonren=True)
    R['G12'] = dict(g10)                   # split run, see sbatch
    return R


def outvars(c):
    v = list(CORE_OUT)
    if c['rout']:
        v += OUT_ROUT
    if c['nonren']:
        v += OUT_NONREN
    if c['dams']:
        v += OUT_DAM
    if c['wu']:
        v += OUT_WU
    if c['irr']:
        v += OUT_IRR
    if c['efr']:
        v += OUT_EFR
    if c['landuse']:
        v += OUT_LU
    if c['wofost']:
        v += OUT_CROP
    return v


def global_text(run, c, domain, build, outdir, start=START, end=END,
                init_state=None, state=None):
    rd = outdir
    dom = (DATA + '/' + domain + '/domain.nc')
    st = state or (end[0], end[1], end[2])
    # VIC writes the state at the start of the day after the last day
    import datetime
    d = datetime.date(*st) + datetime.timedelta(days=1)
    L = []
    a = L.append
    a('# VIC-WUR reference run %s on %s, build %s (tools/vicwur/gruns.py)'
      % (run, domain, build))
    a('MODEL_STEPS_PER_DAY 1')
    a('SNOW_STEPS_PER_DAY 1')
    a('RUNOFF_STEPS_PER_DAY 1')
    a('STARTYEAR %d\nSTARTMONTH %d\nSTARTDAY %d' % start)
    a('ENDYEAR %d\nENDMONTH %d\nENDDAY %d' % end)
    a('CALENDAR PROLEPTIC_GREGORIAN')
    a('FULL_ENERGY FALSE\nFROZEN_SOIL FALSE')
    a('STATENAME %s/state/states' % rd)
    a('STATEYEAR %d\nSTATEMONTH %d\nSTATEDAY %d\nSTATESEC 0'
      % (d.year, d.month, d.day))
    if init_state:
        a('INIT_STATE %s' % init_state)
    else:
        a('# INIT_STATE')
    a('DOMAIN %s' % dom)
    for k, n in (('LAT', 'lat'), ('LON', 'lon'), ('MASK', 'mask'),
                 ('AREA', 'area'), ('FRAC', 'frac'), ('YDIM', 'lat'),
                 ('XDIM', 'lon')):
        a('DOMAIN_TYPE %s %s' % (k, n))
    for t, v in MET:
        a('FORCE_TYPE %s %s %s/met.' % (t, v, F))
    a('WIND_H 10.0')
    a('PARAMETERS %s/params.nc' % R1)
    a('SNOW_BAND TRUE\nBASEFLOW ARNO\nJULY_TAVG_SUPPLIED FALSE\n'
      'ORGANIC_FRACT FALSE')
    a('LAI_SRC FROM_VEGLIB\nFCAN_SRC FROM_VEGLIB\nALB_SRC FROM_VEGLIB')
    a('SHARE_LAYER_MOIST %s' % ('TRUE' if c['share'] else 'FALSE'))
    if c['wofost']:
        a('BCO2_SRC FROM_VEGPARAM')
    if c['gwm']:
        a('GWM TRUE')
    # ---- plugins
    consts = []
    if c['nren_lim'] is not None:
        consts.append('NONRENEWABLE_LIMIT %g' % c['nren_lim'])
    if consts:
        path = '%s/constants.txt' % rd
        a('CONSTANTS %s' % path)
    if c['rout']:
        a('ROUTING TRUE\nROUT_STEPS_PER_DAY %d' % c['rout'])
        a('ROUTING_PARAMETERS %s/rout_n%d.nc' % (R1, c['rout']))
        a('DECOMPOSITION BASIN')
    if c['dams']:
        a('DAMS TRUE\nDAMS_PARAMETERS %s/dams.nc' % R1)
    if c['wu']:
        a('WATERUSE TRUE')
        a('WATERUSE_PARAMETERS %s/wateruse.nc' % R1)
        a('LOCAL_WITHDRAWAL %s' % ('TRUE' if c['local'] else 'FALSE'))
        a('REMOTE_WITHDRAWAL %s' % ('TRUE' if c['remote'] else 'FALSE'))
        a('COMPENSATION_WITHDRAWAL %s' % ('TRUE' if c['comp'] else 'FALSE'))
        a('NONRENEWABLE_WITHDRAWAL %s' % ('TRUE' if c['nonren'] else 'FALSE'))
        a('NONRENEWABLE_RUNOFF %s'
          % ('TRUE' if c['nonren_runoff'] else 'FALSE'))
        for s in c['wu_sectors']:
            S = s.upper()
            a('PLUGIN_FORCE_TYPE %s_DEMAND %s_demand DAY %s/%s_demand.'
              % (S, s, F, s))
            a('PLUGIN_FORCE_TYPE %s_GROUNDWATER %s_gw DAY %s/%s_gw.'
              % (S, s, F, s))
            a('PLUGIN_FORCE_TYPE %s_CONSUMPTION %s_cons DAY %s/%s_cons.'
              % (S, s, F, s))
    if c['irr']:
        a('IRRIGATION TRUE')
        a('IRRIGATION_PARAMETERS %s/%s' % (R1, c['irr_file']))
        for o in c['irr_opts']:
            a('%s TRUE' % o)
        a('PBARE %d' % PBARE)
    if c['efr']:
        a('EFR TRUE')
        a('PLUGIN_FORCE_TYPE EFR_DISCHARGE efr_q MONTH %s/efr_q.' % F)
        a('PLUGIN_FORCE_TYPE EFR_BASEFLOW efr_b MONTH %s/efr_b.' % F)
    if c['landuse']:
        a('LANDUSE_FORCE TRUE')
        a('PLUGIN_FORCE_TYPE CV cv MONTH %s/cv.' % F)
        if not c['irr']:
            a('PBARE %d' % PBARE)
    if c['co2']:
        a('PLUGIN_FORCE_TYPE CO2 co2 YEAR %s/co2.' % F)
    if c['wofost']:
        a('WOFOST TRUE\nWOFOST_STEPS_PER_DAY 1')
        a('CROP_PARAMETERS %s/crop.nc' % R1)
        a('WOFOST_TEXT_PARAMETERS %s/wofost_text.txt' % R1)
        a('WOFOST_DIST_SEASON TRUE\nWOFOST_DIST_TSUM TRUE\n'
          'WOFOST_DIST_FERTILIZER TRUE\nWOFOST_DIST_MINERALIZATION TRUE')
        for o in c['wofost_opts']:
            a('%s TRUE' % o)
    a('RESULT_DIR %s/out/' % rd)
    a('LOG_DIR %s/log/' % rd)
    a('OUTFILE fluxes\nAGGFREQ NDAYS 1\nHISTFREQ NYEARS 1\nOUT_FORMAT NETCDF4')
    for v in outvars(c):
        a('OUTVAR %s' % v)
    return '\n'.join(L) + '\n', consts


SBATCH = """#!/bin/bash
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 04:00:00
#SBATCH -o {rd}/slurm-%j.out
# {run} on {domain}: {builds}
export LD_LIBRARY_PATH={envnc}/lib
cd {rd}
{body}
"""


def run_cmd(build, rd, glob_file, tag):
    return ('mkdir -p {rd}/{b}/out {rd}/{b}/state {rd}/{b}/log\n'
            'date -Is > {rd}/{b}/time{t}.start\n'
            '/usr/bin/env mpirun --oversubscribe -np 1 {exe} -g {g} '
            '> {rd}/{b}/log/stdout{t}.log 2>&1\n'
            'echo $? > {rd}/{b}/rc{t}\n'
            'date -Is > {rd}/{b}/time{t}.end\n').format(
                rd=rd, b=build, exe=BUILDS[build], g=glob_file, t=tag)


def gen(domain, outdir):
    R = runs()
    base_dir = '%s/gruns/%s' % (outdir, domain)
    names = []
    for run, over in R.items():
        c = dict(BASE, **over)
        rd = '%s/%s' % (base_dir, run)
        os.makedirs(rd, exist_ok=True)
        builds = ['W3'] if c['gwm'] else ['W3', 'W']
        body = ''
        for b in builds:
            bd = '%s/%s' % (rd, b)
            for sub in ('out', 'state', 'log'):
                os.makedirs('%s/%s' % (bd, sub), exist_ok=True)
            if run == 'G12':
                t1, cons = global_text(run, c, domain, b, bd, start=START,
                                       end=(2006, 12, 31))
                g1 = '%s/global_%s_part1.txt' % (rd, b)
                g2 = '%s/global_%s_part2.txt' % (rd, b)
                open(g1, 'w').write(t1)
                t2, _ = global_text(run, c, domain, b, bd + '/p2',
                                    start=SPLIT, end=END)
                open(g2, 'w').write(t2)
                os.makedirs(bd + '/p2/out', exist_ok=True)
                os.makedirs(bd + '/p2/state', exist_ok=True)
                os.makedirs(bd + '/p2/log', exist_ok=True)
                body += run_cmd(b, rd, g1, '1')
                body += ('ST=$(ls {bd}/state/states.*.nc | head -1)\n'
                         'sed "s|^# INIT_STATE|INIT_STATE $ST|" {g2} > {g2}.run\n'
                         .format(bd=bd, g2=g2))
                body += ('mkdir -p {bd}/p2/out {bd}/p2/state {bd}/p2/log\n'
                         '/usr/bin/env mpirun --oversubscribe -np 1 {exe} -g '
                         '{g2}.run > {bd}/log/stdout2.log 2>&1\n'
                         'echo $? > {bd}/rc2\n'.format(
                             bd=bd, exe=BUILDS[b], g2=g2))
                if cons:
                    open('%s/constants.txt' % bd, 'w').write(
                        '\n'.join(cons) + '\n')
                continue
            txt, cons = global_text(run, c, domain, b, bd)
            g = '%s/global_%s.txt' % (rd, b)
            open(g, 'w').write(txt)
            if cons:
                open('%s/constants.txt' % bd, 'w').write(
                    '\n'.join(cons) + '\n')
            body += run_cmd(b, rd, g, '')
        open('%s/run.sbatch' % rd, 'w').write(SBATCH.format(
            name='%s_%s' % (run, domain), rd=rd, run=run, domain=domain,
            builds=' + '.join(builds), envnc=ENVNC, body=body))
        names.append(run)
    sub = '%s/submit_all.sh' % base_dir
    with open(sub, 'w') as fh:
        fh.write('#!/bin/bash\n# submit every G-run of %s (cpu partition)\n'
                 % domain)
        for r in names:
            fh.write('sbatch %s/%s/run.sbatch\n' % (base_dir, r))
    os.chmod(sub, 0o755)
    print('wrote %d runs under %s' % (len(names), base_dir))
    return names


# ------------------------------------------------------------------ check
def xbuild(base_dir, run):
    """Largest |W - VIC-WUR 3.0| over the common output variables of one run."""
    import numpy as np
    import netCDF4 as nc
    worst, wv, n = 0.0, '-', 0
    fa = sorted(glob.glob('%s/%s/W3/out/fluxes*.nc' % (base_dir, run)))
    for f in fa:
        g = f.replace('/W3/', '/W/')
        if not os.path.exists(g):
            continue
        with nc.Dataset(f) as a, nc.Dataset(g) as b:
            for v in a.variables:
                if v.startswith('OUT_') and v in b.variables:
                    x = np.ma.filled(a[v][:].astype(float), np.nan)
                    y = np.ma.filled(b[v][:].astype(float), np.nan)
                    d = float(np.nanmax(np.abs(x - y))) if x.size else 0.0
                    n += 1
                    if d > worst:
                        worst, wv = d, v
    return n, worst, wv


def check(domain, outdir, sel):
    import numpy as np
    import netCDF4 as nc
    R = runs()
    base_dir = '%s/gruns/%s' % (outdir, domain)
    rows = []
    for run, over in R.items():
        if sel and run not in sel:
            continue
        c = dict(BASE, **over)
        for b in (['W3'] if c['gwm'] else ['W3', 'W']):
            bd = '%s/%s/%s' % (base_dir, run, b)
            row = dict(run=run, build=b)
            rcs = []
            for t in (('1', '2') if run == 'G12' else ('',)):
                p = '%s/rc%s' % (bd, t)
                rcs.append(open(p).read().strip() if os.path.exists(p)
                           else 'NA')
            row['rc'] = ','.join(rcs)
            logs = sorted(glob.glob(bd + '/log/stdout*.log'))
            txt = ''.join(open(f, errors='replace').read() for f in logs)
            row['ERR'] = len(re.findall(r'\bERROR\b', txt))
            row['WARN'] = len(re.findall(r'\bWARN', txt))
            row['sim_ok'] = ('Completed running VIC' in txt)
            files = sorted(glob.glob(bd + '/out/fluxes*.nc') +
                           glob.glob(bd + '/p2/out/fluxes*.nc'))
            want = [v for v in outvars(c)]
            missing, nan, absent = [], 0, []
            nz = {}
            if files:
                nan_vars = set()
                seen = set()
                for f in files:
                    with nc.Dataset(f) as ds:
                        for v in want:
                            if v not in ds.variables:
                                missing.append(v)
                                continue
                            a = np.ma.filled(ds[v][:].astype(float), np.nan)
                            seen.add(v)
                            if not np.isfinite(a).all():
                                nan_vars.add(v)
                            nz[v] = max(nz.get(v, 0.0), float(np.nanmax(
                                np.abs(a))) if a.size else 0.0)
                nan = len(nan_vars)
                missing = sorted(set(missing) - seen)
                row['zero_vars'] = sorted(v for v in seen if nz.get(v, 0) == 0)
            else:
                row['zero_vars'] = []
            row['files'] = len(files)
            row['missing'] = missing
            row['nan_vars'] = nan
            rows.append(row)
    for r in rows:
        if r['build'] == 'W':
            r['xdiff'] = xbuild(base_dir, r['run'])
    for r in rows:
        print('%-4s %-3s rc=%-4s ok=%s ERR=%d WARN=%d files=%d missing=%d '
              'nanvars=%d allzero=%d' % (
                  r['run'], r['build'], r['rc'], r['sim_ok'], r['ERR'],
                  r['WARN'], r['files'], len(r['missing']), r['nan_vars'],
                  len(r['zero_vars'])))
        if 'xdiff' in r:
            print('     W vs W3 max|diff| over %d vars: %.3g (worst %s)'
                  % r['xdiff'])
        if r['missing']:
            print('     missing:', ' '.join(r['missing']))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['gen', 'check'])
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--domain', default='R1')
    ap.add_argument('--outdir', default=DATA)
    a = ap.parse_args(argv)
    if a.cmd == 'gen':
        gen(a.domain, a.outdir)
    else:
        check(a.domain, a.outdir, a.runs)


if __name__ == '__main__':
    sys.exit(main())
