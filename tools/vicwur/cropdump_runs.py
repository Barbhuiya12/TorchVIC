#!/usr/bin/env python
"""R1 G-runs with WOFOST for the replay of the WOFOST-NPK kernel:
G9, G9b (WOFOST_CALC_MINERALIZATION) and G10 (everything on), each
with the crop-dump build <data root>/build/vicwur3_cropdump (VIC-WUR 3.0 + the
dump patch + tools/vicwur/vicwur_cropdump.patch) and VICWUR_DUMP set.

  cropdump_runs.py gen [--outdir DIR]    global files + run.sbatch per run

Layout: <outdir>/<run>_cdump/{global_W3.txt, run.sbatch, dump.bin,
W3/{out,state,log}}.  The configs are tools/vicwur/gruns.py's.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gruns  # noqa: E402

EXE = gruns.ROOT + '/build/vicwur3_cropdump/vic_image.exe'
OUT = gruns.ROOT + '/verification/vicwur_111/gruns/R1'
RUNS = ('G9', 'G9b', 'G10')


def gen(outdir=OUT, runs=RUNS):
    R = gruns.runs()
    for run in runs:
        c = dict(gruns.BASE, **R[run])
        rd = '%s/%s_cdump' % (outdir, run)
        bd = rd + '/W3'
        for sub in ('out', 'state', 'log'):
            os.makedirs('%s/%s' % (bd, sub), exist_ok=True)
        txt, cons = gruns.global_text(run, c, 'R1', 'W3', bd)
        g = rd + '/global_W3.txt'
        open(g, 'w').write(txt)
        if cons:
            open(bd + '/constants.txt', 'w').write('\n'.join(cons) + '\n')
        body = ('date -Is > {bd}/time.start\n'
                'VICWUR_DUMP={rd}/dump.bin /usr/bin/env mpirun '
                '--oversubscribe -np 1 {exe} -g {g} > {bd}/log/stdout.log '
                '2>&1\necho $? > {bd}/rc\ndate -Is > {bd}/time.end\n'
                ).format(bd=bd, rd=rd, exe=EXE, g=g)
        open(rd + '/run.sbatch', 'w').write(gruns.SBATCH.format(
            name='C%s' % run, rd=rd, run=run + ' (crop dump)', domain='R1',
            builds='W3 cropdump', envnc=gruns.ENVNC, body=body))
        print(rd + '/run.sbatch')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['gen'])
    ap.add_argument('--outdir', default=OUT)
    a = ap.parse_args(argv)
    gen(a.outdir)


if __name__ == '__main__':
    main()
