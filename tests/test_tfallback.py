"""TFALLBACK FALSE: TorchVIC stops where VIC 5.1.0 stops.

VIC (calc_surf_energy_bal.c:311, snow_melt.c:267, snow_intercept.c:342,
calc_atmos_energy_bal.c:87) takes the error_* path of a temperature solve
whose root_brent bracket holds no root when TFALLBACK is FALSE, and ends the
run at that cell and step (the image driver dies in the error print). With
TFALLBACK TRUE (initialize_options.c:51) the solve keeps its previous
temperature and counts a fallback.

The unit tests drive core/tfallback.py; the end-to-end one runs Stehekin
with one forcing record made unsolvable (shortwave 1e7 W m-2 at step 100,
cell 5, which pristine VIC 5.1.0 fails on).
"""
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')

from torchvic.core import tfallback as T                 # noqa: E402


def _mask(n, *hits):
    m = torch.zeros(n, 3, dtype=torch.bool)
    for c, t in hits:
        m[c, t] = True
    return m


def test_true_counts_every_solve_and_never_raises():
    T.start(True)
    T.set_step(4)
    T.note('surface', _mask(4, (1, 0), (2, 2)))
    T.note('snow pack', _mask(4, (3, 1)))
    T.end_step(4)
    T.note('surface', _mask(4, (0, 0)))
    T.end_step(5)
    assert T.counts() == {'surface': 3, 'snow pack': 1}


def test_false_raises_for_the_lowest_cell_naming_step_tile_and_solves():
    T.start(False)
    T.set_step(7)
    T.note('canopy air', _mask(5, (3, 2)))
    T.note('surface', _mask(5, (3, 1), (4, 0)))
    T.note('canopy foliage', _mask(5, (3, 0)))
    with pytest.raises(T.TemperatureSolveError) as e:
        T.end_step(7)
    assert (e.value.step, e.value.cell) == (7, 3)
    assert e.value.kinds == ['canopy foliage', 'surface', 'canopy air']
    assert 'step 7' in str(e.value) and 'cell 3' in str(e.value)
    assert 'canopy foliage, surface, canopy air' in str(e.value)


def test_false_with_nothing_failed_is_silent_and_pending_is_cleared():
    T.start(False)
    T.note('surface', _mask(3))
    T.end_step(0)
    assert T.S.pending == []


def test_restrict_keeps_only_the_selected_cells():
    T.start(False)
    m0 = T.mark()
    T.note('snow pack', _mask(4, (0, 0), (2, 1)))
    T.restrict(m0, torch.tensor([False, True, True, False]))
    with pytest.raises(T.TemperatureSolveError) as e:
        T.end_step(1)
    assert e.value.cell == 2


def test_graph_safe_path_records_nothing():
    from torchvic.core import graphsafe
    T.start(False)
    graphsafe.ACTIVE = True
    try:
        T.note('surface', _mask(2, (0, 0)))
    finally:
        graphsafe.ACTIVE = False
    T.end_step(0)


def test_global_file_value_reaches_simulate(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                    'verification'))
    matrix = pytest.importorskip('matrix')
    from torchvic import case as C
    from torchvic.io.global_param import GlobalParam
    for val, want in (('FALSE', False), ('TRUE', True), (None, True)):
        r = matrix.row('wg', 'EB')
        extra = [] if val is None else ['TFALLBACK ' + val]
        txt = matrix.global_text(r, str(tmp_path), str(tmp_path),
                                 ['OUT_PREC'], extra_lines=extra)
        g = tmp_path / 'g.txt'
        g.write_text(txt)
        assert C.image_options(GlobalParam(str(g)))['tfallback'] is want


def test_the_option_table_no_longer_refuses_false():
    from torchvic.io import options as O
    assert O.disposition('TFALLBACK', 'FALSE')[0] == O.IMPLEMENTED
    assert O.disposition('TFALLBACK', 'TRUE')[0] == O.IMPLEMENTED


def _stehekin_unsolvable(tmp_path):
    xr = pytest.importorskip('xarray')
    pytest.importorskip('netCDF4')
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                    'verification'))
    matrix = pytest.importorskip('matrix')
    c = dict(matrix.CASES['stehe'])
    src = c['forcing'] + '1949.nc'
    if not os.path.exists(src):
        pytest.skip('Stehekin sample data absent (%s)' % src)
    d = xr.open_dataset(src).load()
    d['dswrf'][100, 1, 2] = 1e7
    fdir = tmp_path / 'f'
    fdir.mkdir()
    d.to_netcdf(str(fdir / 'forc.1949.nc'))
    c['forcing'] = str(fdir / 'forc.')
    matrix.CASES['_unsolvable'] = c
    return matrix


def _run(matrix, tmp_path, fb):
    r = matrix.row('stehe', 'EB')
    r['case'] = '_unsolvable'
    out = tmp_path / ('out' + fb)
    out.mkdir()
    txt = matrix.global_text(r, str(out), str(out), ['OUT_PREC'],
                             extra_lines=['TFALLBACK ' + fb])
    g = tmp_path / ('g' + fb + '.txt')
    g.write_text(txt)
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return subprocess.run(
        [sys.executable, '-m', 'torchvic', 'run', str(g), '--core', '5.1.0',
         '--device', 'cpu', '--out', str(out), '--no-history'],
        capture_output=True, text=True, env=env, cwd=str(tmp_path))


def test_unsolvable_step_stops_at_vics_step_and_cell_and_true_counts_it(
        tmp_path):
    matrix = _stehekin_unsolvable(tmp_path)
    f = _run(matrix, tmp_path, 'FALSE')
    assert f.returncode != 0
    assert 'TemperatureSolveError' in f.stderr
    assert 'surface' in f.stderr and 'step 100,' in f.stderr
    assert 'cell 5' in f.stderr          # lat 48.3125 lon -120.8125
    t = _run(matrix, tmp_path, 'TRUE')
    assert t.returncode == 0, t.stderr[-500:]
    assert 'TFALLBACK TRUE' in t.stdout and 'surface' in t.stdout
