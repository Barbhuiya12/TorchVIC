#!/usr/bin/env python
"""Set up dump-build runs of the R1 VIC-WUR 3.0 G-runs.

For every R1 G-run, rewrite global_W3*.txt and run.sbatch so that they use
<data root>/build/vicwur3_dump/vic_image.exe and write below
<data root>/verification/vicwur_03/gruns/R1/<run>[d]/.  Runs with suffix '_dump' set
VICWUR_DUMP (dump on); the others leave it unset (dump off).  The W block of
each sbatch is dropped (the dump patch is on VIC-WUR 3.0 only).
"""
import glob
import os
import re
import sys

VT = os.environ.get('TORCHVIC_WORK_ROOT', '.')
DATA_ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
SRC = DATA_ROOT + '/vicwur/gruns/R1'
DST = DATA_ROOT + '/verification/vicwur_03/gruns/R1'
OLDEXE = DATA_ROOT + '/build/vicwur3/vic_image.exe'
NEWEXE = DATA_ROOT + '/build/vicwur3_dump/vic_image.exe'
DUMP_RUNS = ['G2', 'G4', 'G10']


def conv(txt, run, new):
    txt = txt.replace(SRC + '/' + run, DST + '/' + new)
    return txt.replace(OLDEXE, NEWEXE)


def make(run, new, dump):
    os.makedirs(DST + '/' + new, exist_ok=True)
    for f in glob.glob('%s/%s/global_W3*.txt' % (SRC, run)):
        if f.endswith('.run'):
            continue
        t = conv(open(f).read(), run, new)
        open('%s/%s/%s' % (DST, new, os.path.basename(f)), 'w').write(t)
    for x in glob.glob('%s/%s/W3/constants.txt' % (SRC, run)):
        os.makedirs('%s/%s/W3' % (DST, new), exist_ok=True)
        open('%s/%s/W3/constants.txt' % (DST, new), 'w').write(open(x).read())
    keep = []
    for ln in open('%s/%s/run.sbatch' % (SRC, run)):
        if re.search(r'/W/|vicwur_git|global_W_|global_W\.txt', ln):
            continue
        ln = conv(ln, run, new)
        if dump and 'mpirun' in ln:
            ln = 'VICWUR_DUMP=%s/%s/dump.bin %s' % (DST, new, ln)
        keep.append(ln)
    s = ''.join(keep).replace('#SBATCH -J %s_R1' % run,
                              '#SBATCH -J D%s' % new)
    s = re.sub(r'-o .*slurm-%j.out', '-o %s/%s/slurm-%%j.out' % (DST, new), s)
    open('%s/%s/run.sbatch' % (DST, new), 'w').write(s)
    return '%s/%s/run.sbatch' % (DST, new)


def main():
    runs = sorted(os.path.basename(d) for d in glob.glob(SRC + '/G*'))
    out = []
    for r in runs:
        out.append(make(r, r, False))
        if r in DUMP_RUNS:
            out.append(make(r, r + '_dump', True))
    open(DST + '/submit_all.sh', 'w').write(
        '#!/bin/bash\n' + ''.join('sbatch %s\n' % o for o in out))
    print(len(out), 'runs')


if __name__ == '__main__':
    sys.exit(main())
