#!/usr/bin/env python
"""Dump runs of the connector replay tests.

  replay_runs.py gen [--outdir DIR]     write the global files and sbatch files
                                        of G1a G1b G2 G3 G5 G6a G6b G6c G7
                                        and G4 G8 (kernel level)

Every run is the G-run of tools/vicwur/gruns.py (same global text, R1), run by
the dump build <data root>/build/vicwur3_dump with VICWUR_DUMP set, so each run
leaves `dump.bin` (tools/vicwur/vicwur_dump.patch) beside the usual `W3/out`
history files.  The one change to the global text: OUT_RUNOFF and
OUT_BASEFLOW are written as OUT_TYPE_DOUBLE (an output precision only; the
dump and every other output are bit for bit those of the float32 run), because
the replay feeds VIC's runoff and baseflow to the connectors and the
non-renewable deficit is paid back from them.  Layout: <outdir>/gruns/R1/<run>/{global_W3.txt,
run.sbatch, dump.bin, W3/{out,state,log}}.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gruns as G                                          # noqa: E402

OUT = G.ROOT + '/verification/vicwur_23'
RUNS = ['G1a', 'G1b', 'G2', 'G3', 'G4', 'G5', 'G6a', 'G6b', 'G6c', 'G7', 'G8']
DUMP_EXE = G.ROOT + '/build/vicwur3_dump/vic_image.exe'

SB = """#!/bin/bash
#SBATCH -J w23_{run}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 02:00:00
#SBATCH -o {rd}/slurm-%j.out
export LD_LIBRARY_PATH={envnc}/lib
cd {rd}
mkdir -p {rd}/W3/out {rd}/W3/state {rd}/W3/log
date -Is > {rd}/W3/time.start
VICWUR_DUMP={rd}/dump.bin /usr/bin/env mpirun --oversubscribe -np 1 {exe} -g {rd}/global_W3.txt > {rd}/W3/log/stdout.log 2>&1
echo $? > {rd}/W3/rc
date -Is > {rd}/W3/time.end
"""


def gen(outdir):
    R = G.runs()
    for run in RUNS:
        c = dict(G.BASE, **R[run])
        rd = '%s/gruns/R1/%s' % (outdir, run)
        for sub in ('out', 'state', 'log'):
            os.makedirs('%s/W3/%s' % (rd, sub), exist_ok=True)
        txt, cons = G.global_text(run, c, 'R1', 'W3', rd + '/W3')
        for v in ('OUT_RUNOFF', 'OUT_BASEFLOW'):
            assert 'OUTVAR %s\n' % v in txt
            txt = txt.replace('OUTVAR %s\n' % v,
                              'OUTVAR %s * OUT_TYPE_DOUBLE * *\n' % v)
        open(rd + '/global_W3.txt', 'w').write(txt)
        if cons:
            open(rd + '/W3/constants.txt', 'w').write('\n'.join(cons) + '\n')
        open(rd + '/run.sbatch', 'w').write(SB.format(
            run=run, rd=rd, envnc=G.ENVNC, exe=DUMP_EXE))
    with open(outdir + '/gruns/R1/submit_all.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        for run in RUNS:
            f.write('sbatch %s/gruns/R1/%s/run.sbatch\n' % (outdir, run))
    os.chmod(outdir + '/gruns/R1/submit_all.sh', 0o755)
    print('wrote', len(RUNS), 'runs under', outdir + '/gruns/R1')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['gen'])
    ap.add_argument('--outdir', default=OUT)
    a = ap.parse_args()
    gen(a.outdir)
