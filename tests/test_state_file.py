"""VIC image-driver state files: STATENAME / STATEYEAR.. write one, INIT_STATE
starts from one (torchvic/io/state_file.py; vic_store.c, vic_restore.c).

TorchVIC alone: a Stehekin run split in two through a state file gives the
single run's second half bit for bit; the file has VIC's layout. The
comparison with pristine VIC's own files is verification/state_matrix.py
(it runs vic_image.exe); here a stored copy of VIC's file is used when the
verification tree has one.
"""
import glob
import os
import sys

import numpy as np
import pytest
import torch

import _paths                                               # noqa: E402

sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
matrix = pytest.importorskip('matrix')
from torchvic import case as C                              # noqa: E402
from torchvic.io import options as O, state_file as SF      # noqa: E402

if not os.path.exists(matrix.CASES['stehe']['domain']):
    pytest.skip('Stehekin sample data is absent', allow_module_level=True)

import state_diff                                           # noqa: E402

SPD = 24


def _global(tmp, name, start, end, mode='EB', extra=(), case='stehe'):
    r = matrix.row(case, mode)
    c = dict(matrix.CASES[case])
    c['start'], c['end'] = start, end
    matrix.CASES['_t_' + name] = c
    r = dict(r, case='_t_' + name)
    res = os.path.join(str(tmp), name)
    os.makedirs(res, exist_ok=True)
    txt = matrix.global_text(r, res, res, ['OUT_PREC'],
                             extra_lines=list(extra))
    g = os.path.join(res, 'global.txt')
    open(g, 'w').write(txt)
    return g


def _split_run(tmp, mode='EB', extra=(), **opts):
    """A Stehekin run of 10 days with a state at day 6, then days 6-10
    from that state."""
    st = os.path.join(str(tmp), 'st')
    kw = dict(mode=mode, extra=list(extra))
    r = matrix.row('stehe', mode, **opts)
    c = dict(matrix.CASES['stehe'])
    matrix.CASES['_t_split_' + mode] = c
    g1 = _global(tmp, 'full', (1949, 1, 1), (1949, 1, 10), mode=mode,
                 extra=['STATENAME %s' % st, 'STATEYEAR 1949',
                        'STATEMONTH 1', 'STATEDAY 6', 'STATESEC 0']
                 + list(extra))
    with torch.inference_mode():
        c1 = C.load(g1, core='5.1.0')
        plan = SF.save_plan(c1)
        out1 = c1.run()
        path = SF.save_state(c1, plan, log=lambda *a: None)
        g2 = _global(tmp, 'half2', (1949, 1, 6), (1949, 1, 10), mode=mode,
                     extra=['INIT_STATE %s' % path] + list(extra))
        c2 = C.load(g2, core='5.1.0')
        out2 = c2.run()
    del c1, c2           # a Case keeps its parameter files open
    return dict(path=path, plan=plan, out1=out1, out2=out2, tmp=str(tmp))


@pytest.fixture(scope='module')
def split(tmp_path_factory):
    return _split_run(tmp_path_factory.mktemp('state'))


def _same_second_half(o1, o2, n_total=10 * SPD, n_half=5 * SPD):
    n = n_total - n_half
    checked = 0
    for k, v in o1.items():
        w = o2.get(k)
        if (not torch.is_tensor(v) or w is None or v.dim() < 1
                or v.shape[0] != n_total or w.shape[0] != n_half):
            continue
        assert torch.equal(v[n:], w), k
        checked += 1
    return checked


def test_the_state_file_is_named_and_dated_as_vic_names_it(split):
    n, when, path = split['plan']
    assert n == 5 * SPD
    assert os.path.basename(split['path']).endswith('.19490106_00000.nc')
    assert os.path.exists(split['path'])


def test_the_file_has_vic_5_1_0_s_variables_and_dimensions(split):
    import netCDF4 as nc4
    ds = nc4.Dataset(split['path'])
    assert [v for v in ds.variables if v.startswith('STATE_')] == list(
        SF.VARIABLES)
    assert {k: len(v) for k, v in ds.dimensions.items()
            if k not in ('time', 'tv_route_step')} == {
        'lon': 5, 'lat': 4, 'veg_class': 12, 'snow_band': 1, 'nlayer': 3,
        'frost_area': 1, 'soil_node': 3}
    v = ds['STATE_SOIL_MOISTURE']
    assert v.dimensions == ('veg_class', 'snow_band', 'nlayer', 'lat', 'lon')
    assert v.units == 'mm' and v._FillValue == SF.FILL_D
    assert ds['STATE_SOIL_ICE'].dimensions == (
        'veg_class', 'snow_band', 'nlayer', 'frost_area', 'lat', 'lon')
    assert ds['STATE_SNOW_AGE'].dtype == np.int32
    assert ds['time'][0] == 711497.0           # 1949-01-06, days since 1-1-1
    ds.close()


def test_second_half_from_the_state_file_is_the_single_run_bit_for_bit(split):
    assert _same_second_half(split['out1'], split['out2']) > 40


def test_the_water_balance_mode_round_trips_bit_for_bit(tmp_path_factory):
    r = _split_run(tmp_path_factory.mktemp('stwb'), mode='WB')
    assert _same_second_half(r['out1'], r['out2']) > 30


def test_a_state_off_the_step_grid_writes_nothing(tmp_path):
    g = _global(tmp_path, 'off', (1949, 1, 1), (1949, 1, 2),
                extra=['STATENAME %s/st' % tmp_path, 'STATEYEAR 1949',
                       'STATEMONTH 1', 'STATEDAY 1', 'STATESEC 1800'])
    c = C.load(g, core='5.1.0', steps=SPD)
    n, when, path = SF.save_plan(c)
    assert n is None
    assert SF.save_state(c, (n, when, path), log=lambda *a: None) is None
    del c


def test_an_incomplete_state_date_is_refused_as_vic_refuses_it(tmp_path):
    g = _global(tmp_path, 'inc', (1949, 1, 1), (1949, 1, 2),
                extra=['STATENAME %s/st' % tmp_path, 'STATEYEAR 1949'])
    c = C.load(g, core='5.1.0', steps=SPD)
    with pytest.raises(ValueError, match='STATEYEAR, STATEMONTH and STATEDAY'):
        SF.save_plan(c)
    del c


def test_init_state_checks_the_files_dimensions(split, tmp_path):
    import shutil
    import netCDF4 as nc4
    bad = os.path.join(str(tmp_path), 'bad.nc')
    shutil.copy(split['path'], bad)
    ds = nc4.Dataset(bad, 'a')
    ds['lon'][0] = ds['lon'][0] + 1.0
    ds.close()
    g = _global(tmp_path, 'dims', (1949, 1, 6), (1949, 1, 7))
    c = C.load(g, core='5.1.0', steps=SPD)
    with pytest.raises(ValueError, match='Longitudes'):
        SF.read_state_file(bad, c)
    SF.read_state_file(split['path'], c)


def test_init_state_and_statename_are_implemented_in_the_options_table():
    assert O.disposition('INIT_STATE', '/some/state.nc')[0] == O.IMPLEMENTED
    assert O.disposition('INIT_STATE', 'FALSE')[0] == O.NO_EFFECT
    for k in ('STATENAME', 'STATEYEAR', 'STATEMONTH', 'STATEDAY', 'STATESEC'):
        assert O.disposition(k, '1')[0] == O.IMPLEMENTED


def test_a_missing_init_state_file_is_refused_by_name(tmp_path):
    missing = os.path.join(str(tmp_path), 'nonexistent', 'state.nc')
    g = _global(tmp_path, 'miss', (1949, 1, 6), (1949, 1, 7),
                extra=['INIT_STATE %s' % missing])
    c = C.load(g, core='5.1.0', steps=SPD)
    with pytest.raises(ValueError, match='INIT_STATE .*does not exist'):
        c.kwargs()
    del c


def test_the_frozen_soil_switch_is_read_from_the_image_parameters():
    # vic_init.c 5.1.0:684-689: Stehekin's default file has fs_active 0,
    # its FROZEN_SOIL file 1
    from torchvic.io.params import ImageParams
    assert 'fs_active' in ImageParams.SOIL_2D


@pytest.mark.skipif(not os.path.exists(matrix.CASES['wg']['domain']),
                    reason='Walnut Gulch sample data is absent')
def test_a_snow_free_restart_carries_the_stand_ins_energy_terms(tmp_path):
    # the snow-free fast path's zero stand-in for snow_step holds a nested
    # dict of energy terms (read by basin.py's `se = new_state['energy']`);
    # it went into the file flattened without them
    import netCDF4 as nc4
    st = os.path.join(str(tmp_path), 'st')
    g1 = _global(tmp_path, 'wgfull', (2008, 1, 11), (2008, 1, 16),
                 case='wg', extra=['STATENAME %s' % st, 'STATEYEAR 2008',
                                   'STATEMONTH 1', 'STATEDAY 14',
                                   'STATESEC 0'])
    with torch.inference_mode():
        c1 = C.load(g1, core='5.1.0')
        plan = SF.save_plan(c1)
        out1 = c1.run()
        path = SF.save_state(c1, plan, log=lambda *a: None)
        ds = nc4.Dataset(path)
        assert any(v.startswith('TV_SNOWZERO_STATE_ENERGY_')
                   for v in ds.variables)
        ds.close()
        g2 = _global(tmp_path, 'wghalf', (2008, 1, 14), (2008, 1, 16),
                     case='wg', extra=['INIT_STATE %s' % path])
        c2 = C.load(g2, core='5.1.0')
        out2 = c2.run()
    del c1, c2
    assert _same_second_half(out1, out2, n_total=6 * 8, n_half=3 * 8) > 30


VIC_STATE = os.path.join(matrix.WORK, 'item16_state', 'stehe_EB', 'vic_cont')


@pytest.mark.skipif(not glob.glob(os.path.join(VIC_STATE, 'st.*.nc')),
                    reason='no stored pristine VIC state file '
                           '(verification/state_matrix.py makes it)')
def test_the_file_is_pristine_vics_to_1e_12(split):
    ref = sorted(glob.glob(os.path.join(VIC_STATE, 'st.19490106*.nc')))[0]
    lines = []
    bad = state_diff.diff(ref, split['path'], 1e-12, out=lines.append)
    assert bad == 0, '\n'.join(lines)
