#!/usr/bin/env python
"""Dump-build check: outputs vs the unpatched VIC-WUR 3.0 build, bitwise, and
a reader/consistency check of the dump files.

  dump_check.py            compare every run, print one line per run
"""
import glob
import os
import struct
import sys

import numpy as np

import netCDF4

VT = os.environ.get('TORCHVIC_WORK_ROOT', '.')
DATA_ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
SRC = DATA_ROOT + '/vicwur/gruns/R1'
DST = DATA_ROOT + '/verification/vicwur_03/gruns/R1'


def same_nc(a, b):
    """True when every variable of both files is bitwise identical."""
    with netCDF4.Dataset(a) as x, netCDF4.Dataset(b) as y:
        for d in (x, y):
            d.set_auto_maskandscale(False)
        if sorted(x.variables) != sorted(y.variables):
            return False, 'variables differ'
        for k in x.variables:
            u, v = np.asarray(x[k][:]), np.asarray(y[k][:])
            if u.shape != v.shape or u.dtype != v.dtype or \
                    u.tobytes() != v.tobytes():
                return False, k
    return True, ''


def files(root):
    out = {}
    for sub in ('out', 'state', 'p2/out', 'p2/state'):
        for f in sorted(glob.glob('%s/W3/%s/*.nc' % (root, sub))):
            out[sub + '/' + os.path.basename(f)] = f
    return out


def read_dump(path):
    """-> list of (tag, array) records."""
    recs = []
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if not h:
                break
            tag, n = struct.unpack('<iq', h)
            recs.append((tag, np.frombuffer(f.read(8 * n), '<f8')))
    return recs


def main():
    bad = 0
    for run in sorted(os.path.basename(d) for d in glob.glob(DST + '/G*')):
        base = run.replace('_dump', '')
        a, b = files(SRC + '/' + base), files(DST + '/' + run)
        if not a or sorted(a) != sorted(b):
            print(run, 'FILE SET DIFFERS', len(a), len(b))
            bad += 1
            continue
        nd = 0
        for k in a:
            ok, why = same_nc(a[k], b[k])
            if not ok:
                nd += 1
                print('  ', run, k, 'DIFFERS', why)
        msg = '%-9s %2d files, %s' % (run, len(a),
                                      'bitwise identical' if not nd else
                                      '%d DIFFER' % nd)
        bad += nd
        dp = DST + '/' + run + '/dump.bin'
        if os.path.exists(dp):
            r = read_dump(dp)
            steps = sum(1 for t, _ in r if t == 1)
            msg += ', dump: %d records, %d steps, %.1f MB' % (
                len(r), steps, os.path.getsize(dp) / 1e6)
        elif run.endswith('_dump'):
            msg += ', NO DUMP FILE'
            bad += 1
        print(msg)
    print('RESULT', 'PASS' if not bad else 'FAIL (%d)' % bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
