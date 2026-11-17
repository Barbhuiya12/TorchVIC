"""Unit tests for tools/fdir_sources.py and tools/fdir_multi.py helpers (synthetic grids, numpy only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
import build_isc_fdir_grit as bg  # noqa: E402
import fdir_sources as fs  # noqa: E402


def test_pyflwdir_codes_map_to_project_convention():
    # pyflwdir/ESRI north-up: 64 = north; project rows increase with latitude: north = (+1, 0)
    assert bg.ESRI[int(fs.LUT[64])] == (1, 0)
    assert bg.ESRI[int(fs.LUT[4])] == (-1, 0)          # south
    assert bg.ESRI[int(fs.LUT[1])] == (0, 1)           # east
    assert bg.ESRI[int(fs.LUT[128])] == (1, 1)         # north-east
    assert fs.LUT[0] == 0 and fs.LUT[247] == 255 and fs.LUT[255] == 255


def test_finish_breaks_loops_and_fills():
    mask = np.ones((4, 4), bool)
    fd = np.full((4, 4), 255, np.uint8)
    fd[1, 1], fd[1, 2] = 1, 16                          # two cells pointing at each other
    fd[0, :] = 1                                        # bottom row drains east, off the grid
    elev = np.arange(16, dtype=float).reshape(4, 4)
    out, flag, up, info = fs.finish(fd, mask, elev, np.ones((4, 4)))
    assert info['loops_broken'] == 1
    assert bg.drains_to_terminal(np.where(mask, out, 0), mask).all()
    assert up.sum() >= 16 and (out != 255).all()
