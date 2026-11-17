#!/usr/bin/env python
"""Extract the routing records of a VICWUR_DUMP file (dump patch,
plugin_dump.c) into a small .npz for the routing replay.

    dump_rout.py dump.bin rout.npz

rout.npz holds
  step [nt, 4]        current, year, month, day      (tag 1)
  rout [nt, nc, 5+R]  runoff inflow stream nonrenew_deficit discharge
                      dt_discharge[R]                 (tag 4)
  dims [15]           the tag 0 header
The soil, wu, irr and dam records are skipped without being read.
"""
import struct
import sys

import numpy as np


def read_tags(path, keep):
    """-> {tag: [arrays]} for the tags in `keep`, other records skipped."""
    out = {t: [] for t in keep}
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            tag, n = struct.unpack('<iq', h)
            if tag in out:
                out[tag].append(np.frombuffer(f.read(8 * n), '<f8'))
            else:
                f.seek(8 * n, 1)
    return out


def extract(path):
    r = read_tags(path, (0, 1, 4))
    dims = r[0][0]
    nc, ring = int(dims[0]), int(dims[5])
    rout = np.stack(r[4]).reshape(len(r[4]), nc, 5 + ring)
    return {'dims': dims, 'step': np.stack(r[1]), 'rout': rout}


if __name__ == '__main__':
    np.savez_compressed(sys.argv[2], **extract(sys.argv[1]))
