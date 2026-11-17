"""Synthetic checks of tools/build_isc_fdir_grit.py (no GRIT data needed)."""
import importlib.util
import pathlib

import numpy as np

_p = pathlib.Path(__file__).resolve().parents[1] / 'tools' / 'build_isc_fdir_grit.py'
_s = importlib.util.spec_from_file_location('bfg', _p)
bfg = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bfg)


def _grid(ny=5, nx=8):
    lat = 10 + 0.0625 * np.arange(ny)
    lon = 70 + 0.0625 * np.arange(nx)
    mask = np.ones((ny, nx), bool)
    return lat, lon, mask


def _seg(gid, pts, ain, aout, down, end=0):
    return dict(gid=gid, xy=np.array(pts, float), ain=ain, aout=aout, down=down, end=end)


def test_river_and_dem_cells():
    lat, lon, mask = _grid()
    y = lat[2]
    # two segments west -> east along row 2, ending at a coastal outlet past the grid
    segs = [_seg(1, [(lon[0], y), (lon[3], y)], 10, 100, [2]),
            _seg(2, [(lon[3], y), (lon[7] + 0.03, y)], 100, 200, [], end=1)]
    elev = np.add.outer(np.arange(5)[::-1] * 0.0, np.arange(8)[::-1] * 1.0) + 5
    elev[2, :] = 1
    area = np.full(mask.shape, 4e7)
    fd, src, up, amax, info = bfg.build(segs, lat, lon, mask, elev, area)
    assert (fd[2, :7] == 1).all()                      # east along the river
    assert (src[2] == 1).all() and (src[0] == 2).all()
    assert info['unreached_to_sink'] == 0
    # every path ends at an edge/mouth or a sink: accumulation succeeded (no loops)
    assert up[2, 7] >= up[2, 6] >= up[2, 0]


def test_cycle_is_broken():
    lat, lon, mask = _grid(3, 3)
    fd = -np.ones((3, 3), dtype=np.int64)
    fd[1, 1] = 1; fd[1, 2] = 16                        # E and W: 2-cycle
    w = np.zeros((3, 3)); w[1, 1] = 5; w[1, 2] = 1
    assert bfg.break_cycles(fd, mask, w) == 1
    assert fd[1, 2] == -1 and fd[1, 1] == 1
