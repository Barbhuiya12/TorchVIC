#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Soil-moisture spin-up of the ISC domain by cycling the forcing.

The VIC-WUR ISC soil has two 6 m layers (about 2600 mm each, wilting point 26 % of saturation) and an init_moist of
about 220 mm (8 %): a 92-day spin-up leaves the root layer below the wilting point for most of the run and every rainfed
crop starves. This tool runs VIC alone (step S0) over a window, takes the cover-weighted layer moisture of each cell at
the end and writes it as init_moist of a copy of the parameter file; run it in a loop (the next run reads the copy).

  spin_init.py cycle PARAMS_IN CELL_NPY S0-ARGS...   run S0 (run_step arguments) with PARAMS_IN, save [n_cell, L] moisture
  spin_init.py write PARAMS_IN PARAMS_OUT CELL_NPY   copy PARAMS_IN with init_moist = CELL_NPY on the active cells
"""
import os
import runpy
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def cell_mean(moist, lane_cell, lane_tile, cv, col_cell, n_cell):
    """Cover-weighted layer moisture per cell. moist [lane, L] (one row per covered tile of a column), lane_cell /
    lane_tile [lane] the column and tile of a lane, cv [column, tile] the tile covers, col_cell [column] the cell of
    each column (snow bands: several columns per cell, equal weight). Returns [n_cell, L]."""
    import torch
    c = cv.to(moist.dtype)
    pad = moist.new_zeros(cv.shape[0], cv.shape[1], moist.shape[-1])
    pad[lane_cell, lane_tile] = moist * c[lane_cell, lane_tile].unsqueeze(-1)
    col = pad.sum(1) / c.sum(1).clamp(min=1e-12).unsqueeze(-1)
    cm = moist.new_zeros(n_cell, moist.shape[-1])
    cnt = moist.new_zeros(n_cell)
    cm.index_add_(0, col_cell, col)
    cnt.index_add_(0, col_cell, torch.ones(col.shape[0], dtype=moist.dtype, device=moist.device))
    return cm / cnt.clamp(min=1).unsqueeze(-1)


def write_init(src, dst, cell_moist):
    """Copy the parameter file `src` to `dst` with init_moist [L, lat, lon] set on the active cells (row-major order)."""
    import netCDF4 as nc
    shutil.copyfile(src, dst)
    d = nc.Dataset(dst, 'a')
    act = np.ma.filled(d['mask'][:], 0) > 0
    x = np.asarray(cell_moist)
    if x.shape[0] != int(act.sum()):
        raise ValueError('%d cells given, the mask has %d' % (x.shape[0], int(act.sum())))
    v = d['init_moist']
    a = np.ma.filled(v[:], 0.0)
    for k in range(a.shape[0]):
        a[k][act] = x[:, k]
    v[:] = a
    d.close()


def cycle(params_in, out_npy, run_args):
    """One S0 run over the window of `run_args` with `params_in`, the end moisture to `out_npy`."""
    import torch
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, '..', 'vicwur', 'isc'))
    import isc_run as R
    from torchvic.model import lanes as LM
    ob, GL = R.build, {}

    def build(*a, **k):
        GL['m'], GL['c'] = ob(*a, **k)
        return GL['m'], GL['c']

    R.build = build
    sys.argv = ['run_step.py'] + run_args + ['--params', params_in]
    try:
        runpy.run_path(os.path.join(HERE, 'run_step.py'), run_name='__main__')
    except SystemExit:
        pass
    st = GL['m'].runtime.connectors['vic'].state
    cv, ex = GL['c'].mosaic.cv, GL['c'].snowbands
    ln = LM.Lanes(cv)
    mo = st['moist'].reshape(st['moist'].shape[0], -1)
    cell = torch.as_tensor(np.asarray(ex['cell'])).to(mo.device)
    np.save(out_npy, cell_mean(mo, ln.lane_cell, ln.lane_tile, cv, cell, int(ex['n_cell'])).cpu().numpy())


if __name__ == '__main__':
    if sys.argv[1] == 'cycle':
        cycle(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        write_init(sys.argv[2], sys.argv[3], np.load(sys.argv[4]))
