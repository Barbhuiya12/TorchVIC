"""VIC-WUR plugin parameter readers on the R1 reference domain.

R1 is made by tools/vicwur/ref_domains.py into <data root>/vicwur/R1 (set
TORCHVIC_VICWUR_DIR to move it); the tests skip when it is absent.  The
expected values are the ones the generator writes and the C readers derive.
"""
import os
import _paths  # noqa: E402
import shutil

import numpy as np
import pytest

netCDF4 = pytest.importorskip('netCDF4')
torch = pytest.importorskip('torch')

from torchvic.vicwur import params as P                    # noqa: E402

DATA = _paths.env_path(('TORCHVIC_VICWUR_DIR',), 'vicwur')
R1 = DATA + '/R1'
GLOBAL = DATA + '/gruns/R1/G10/global_W3.txt'

pytestmark = pytest.mark.skipif(not os.path.isdir(R1) or
                                not os.path.exists(GLOBAL),
                                reason='R1 reference domain not generated')


@pytest.fixture(scope='module')
def grid():
    return P.Grid(R1 + '/domain.nc', R1 + '/params.nc')


@pytest.fixture(scope='module')
def inp():
    return P.load_vicwur_inputs(GLOBAL)


def _copy(tmp_path, name):
    dst = str(tmp_path / name)
    shutil.copy(R1 + '/' + name, dst)
    return dst


def test_grid(grid):
    assert grid.n == 6 and grid.active.shape == (grid.ny, grid.nx)


def test_all_files_load_on_cpu(inp):
    assert inp.warnings == []
    for comp in (inp.routing, inp.dams, inp.remote, inp.irrigation,
                 inp.crop, inp.wofost):
        for t in comp.tensors().values():
            assert t.device.type == 'cpu'
    assert inp.b_co2.shape == (18, 6)


def test_routing_chain_and_uh(inp):
    r = inp.routing
    # c0->c1->c2->c3 (outlet), c4->c2, c5 is its own outlet
    assert r.downstream.tolist() == [1, 2, 3, 3, 2, 5]
    assert r.outlet.tolist() == [False, False, False, True, False, True]
    assert r.n_upstream.tolist() == [0, 1, 2, 1, 0, 0]
    assert r.upstream[2].tolist()[:2] == [1, 4]
    assert r.uh_length == 24 and r.rout_dt == 21600
    assert r.uh_inflow.shape == (24, 6)
    # sequential-sum renormalisation as in rout_init.c
    for u in (r.uh_inflow, r.uh_runoff):
        s = torch.zeros(6, dtype=torch.float64)
        for j in range(u.shape[0]):
            s = s + u[j]
        assert torch.allclose(s, torch.ones(6, dtype=torch.float64),
                              atol=1e-14)


def test_routing_n1_time_check(grid):
    r = P.read_routing(R1 + '/rout_n1.nc', grid, 1)
    assert r.uh_length == 6 and r.rout_dt == 86400
    with pytest.raises(P.VicWurInputError, match='do not match'):
        P.read_routing(R1 + '/rout_n1.nc', grid, 4)


def test_dams(inp):
    d = inp.dams
    assert d.n_dams == 3
    assert d.dam_cell.tolist() == [1, 2, 4]
    assert d.dam_class.tolist() == [0, 1, 2]
    assert d.is_global.tolist() == [True, True, False]     # type 1 = global
    assert d.year.tolist() == [2003, 2005, 2004]
    np.testing.assert_array_equal(d.capacity.numpy(), [30.0, 15.0, 3.0])
    np.testing.assert_array_equal(d.inflow_fraction.numpy(), [1, 1, 0.8])
    assert d.nservice.tolist() == [2, 2, 1]
    assert d.service.tolist() == [[1, 2], [2, 3], [4, -1]]
    np.testing.assert_array_equal(
        d.service_frac.numpy(), [[0.5, 0.5], [0.5, 1.0], [1.0, -1.0]])


def test_remote_value_at_receiver_is_source_id(inp):
    r = inp.remote
    assert r.nreceiving.tolist() == [0, 0, 1, 1, 0, 0]
    assert r.receiving[2, 0].item() == 4 and r.receiving[3, 0].item() == 5
    assert (r.receiving[[0, 1, 4, 5]] == -1).all()


def test_irrigation(inp):
    i = inp.irrigation
    assert i.veg_class.tolist() == [13, 15]
    assert i.paddy.tolist() == [False, True]
    assert i.active.all()
    assert i.pbare == 18
    # paddy gw fraction forced to 0, non-paddy keeps the cell value
    np.testing.assert_allclose(i.gw_fraction[:, 0].numpy(),
                               [0.2, 0.3, 0.4, 0.5, 0.25, 0.35])
    assert (i.gw_fraction[:, 1] == 0).all()
    # one efficiency per cell copied to every class
    np.testing.assert_allclose(i.efficiency[:, 0].numpy(),
                               [0.5, 0.55, 0.6, 0.65, 0.7, 0.45])
    assert torch.equal(i.efficiency[:, 0], i.efficiency[:, 1])
    assert i.offset[:, 0].tolist() == [0, 5, 10, 15, 20, 3]


def test_irrigation_offset_only_with_option(grid):
    cv = P.read_cv(R1 + '/params.nc', grid)
    i = P.read_irrigation(R1 + '/irr.nc', grid, cv, offset_irrigation=False)
    assert (i.offset == 0).all() and i.pbare is None
    assert P.read_irrigation(R1 + '/irr_paddy.nc', grid, cv).n_classes == 1


def test_irrigation_pbare_range(grid):
    cv = P.read_cv(R1 + '/params.nc', grid)
    with pytest.raises(P.VicWurInputError, match='bare'):
        P.read_irrigation(R1 + '/irr.nc', grid, cv, pbare=5)
    i = P.read_irrigation(R1 + '/irr.nc', grid, cv, pbare=P.PADDY_BARE_UNSET)
    assert i.pbare == 18


def test_crop_nc(inp):
    c = inp.crop
    assert c.n_crop_types == 2 and c.ncrop.tolist() == [2] * 6
    assert c.cidx.tolist() == [[0, 1]] * 6
    assert c.crop_veg_class[:, 0].tolist() == [12, 14]       # 0-based
    assert c.Cc.shape == (12, 2, 6)
    assert set(c.fields) >= {'plant_day', 'harvest_day', 'TSUM1', 'TSUM2',
                             'DVS_point', 'N_amount', 'N_mins'}
    assert c.fields['plant_day'][:, 0].tolist() == [166.0, 182.0]
    assert c.fields['DVS_point'].shape == (1, 2, 6)


def test_crop_nc_flags_select_variables(grid):
    c = P.read_crop(R1 + '/crop.nc', grid, 18)
    assert c.fields == {}
    c = P.read_crop(R1 + '/crop.nc', grid, 18, dist_tsum=True)
    assert set(c.fields) == {'TSUM1', 'TSUM2'}


def test_wofost_text(inp):
    w = inp.wofost
    assert w.n_crops == 2 and w.scalars.shape == (2, 77)
    names = list(w.scalar_names)
    assert len(w.tables) == 22
    # values are float32 as in C: 1/3 is not exactly representable
    tb = w.scalars[:, names.index('TSUM1')]
    assert (tb.numpy() == tb.numpy().astype(np.float32)).all()
    assert (w.scalars[:, names.index('MaxHeight')] == 1.0).all()
    # maize/rice are the two crops (IDSL < 2, so no vernalisation)
    assert (w.scalars[:, names.index('IDSL')] < 2).all()
    assert (w.scalars[:, names.index('VERNSAT')] == -99.0).all()
    x, y, n = w.tables['SLATB']
    assert n.tolist() == [len(x[0][~torch.isnan(x[0])]),
                          len(x[1][~torch.isnan(x[1])])]
    assert w.start_month.tolist() == [1, 1] and w.emergence.tolist() == [0, 0]
    m, d, a, n = w.mng_tables['FERNTAB']
    assert m.tolist() == [[1], [1]] and a.tolist() == [[0.0], [0.0]]
    assert w.mng_scalars[0].tolist() == pytest.approx(
        [0.7, 0.7, 0.7, 0, 0, 0, 0, 0, 0], abs=1e-6)


def test_wofost_text_crop_count_errors(tmp_path):
    p = str(tmp_path / 'w.txt')
    line = open(R1 + '/wofost_text.txt').read().splitlines()
    open(p, 'w').write('\n'.join(line[:3]) + '\n')       # one crop only
    assert P.read_wofost_text(p, 2).n_crops == 1          # k + 1 < NC is ok
    with pytest.raises(P.VicWurInputError, match='more'):
        P.read_wofost_text(R1 + '/wofost_text.txt', 1)
    with pytest.raises(P.VicWurInputError, match='less'):
        P.read_wofost_text(p, 4)


def test_wofost_pH_not_Ph(tmp_path):
    with netCDF4.Dataset(_copy(tmp_path, 'crop.nc'), 'a') as ds:
        ds.renameVariable('pH', 'Ph')
    with pytest.raises(P.UnknownVariableError, match='Ph'):
        P.read_crop(str(tmp_path / 'crop.nc'), P.Grid(
            R1 + '/domain.nc', R1 + '/params.nc'), 18, dist_min=True,
            calc_min=True)


# ---------------------------------------------------------- negative tests
def test_unknown_variable_is_error(tmp_path, grid):
    for name, fn, args in (
            ('rout_n4.nc', P.read_routing, (4,)),
            ('dams.nc', P.read_dams, ()),
            ('wateruse.nc', P.read_remote, ()),):
        path = _copy(tmp_path, name)
        with netCDF4.Dataset(path, 'a') as ds:
            ds.createVariable('bogus', 'i4', ('lat', 'lon'))
        with pytest.raises(P.UnknownVariableError, match='bogus'):
            fn(path, grid, *args)


def test_missing_variable_is_error(tmp_path, grid):
    path = _copy(tmp_path, 'wateruse.nc')
    with netCDF4.Dataset(path, 'a') as ds:
        ds.renameVariable('Nreceiving', 'N_receiving')
    with pytest.raises(P.UnknownVariableError):
        P.read_remote(path, grid)


def test_wrong_grid_is_error(tmp_path, grid):
    path = _copy(tmp_path, 'dams.nc')
    with netCDF4.Dataset(path, 'a') as ds:
        ds['lat'][:] = ds['lat'][:] + 1.0
    with pytest.raises(P.VicWurInputError, match='lat'):
        P.read_dams(path, grid)


def test_dam_service_overflow_is_error(tmp_path, grid):
    path = _copy(tmp_path, 'dams.nc')
    with netCDF4.Dataset(path, 'a') as ds:
        ds['Nservice'][:] = [1, 2, 1]
    with pytest.raises(P.VicWurInputError, match='servicing'):
        P.read_dams(path, grid)


def test_dam_service_shortfall_warns_and_shrinks(tmp_path, grid):
    path = _copy(tmp_path, 'dams.nc')
    with netCDF4.Dataset(path, 'a') as ds:
        ds['Nservice'][:] = [3, 2, 1]
    with pytest.warns(UserWarning, match='servicing'):
        d = P.read_dams(path, grid)
    assert d.nservice.tolist() == [2, 2, 1]


def test_dam_bad_type_is_error(tmp_path, grid):
    path = _copy(tmp_path, 'dams.nc')
    with netCDF4.Dataset(path, 'a') as ds:
        ds['type'][:] = [1, 2, 0]
    with pytest.raises(P.VicWurInputError, match='dam type'):
        P.read_dams(path, grid)


def test_remote_overflow_and_shortfall(tmp_path, grid):
    path = _copy(tmp_path, 'wateruse.nc')
    with netCDF4.Dataset(path, 'a') as ds:
        ds['Nreceiving'][:] = 0
    with pytest.raises(P.VicWurInputError, match='receiving'):
        P.read_remote(path, grid)
    with netCDF4.Dataset(path, 'a') as ds:
        ds['Nreceiving'][:] = 3
    with pytest.warns(UserWarning, match='receiving'):
        r = P.read_remote(path, grid)
    assert r.nreceiving.sum().item() == 2


def test_no_downstream_cell_is_outlet_with_warning(tmp_path, grid):
    path = _copy(tmp_path, 'rout_n4.nc')
    with netCDF4.Dataset(path, 'a') as ds:
        d = ds['downstream'][:]
        d[0, 0] = 9999
        ds['downstream'][:] = d
    with pytest.warns(UserWarning, match='outflow'):
        r = P.read_routing(path, grid, 4)
    assert r.downstream[0].item() == 0 and r.outlet[0].item()


def test_uh_sum_warns_and_renormalises(tmp_path, grid):
    path = _copy(tmp_path, 'rout_n4.nc')
    with netCDF4.Dataset(path, 'a') as ds:
        ds['uh_runoff'][:] = ds['uh_runoff'][:] * 1.1
    with pytest.warns(UserWarning, match='uh_runoff'):
        r = P.read_routing(path, grid, 4)
    assert torch.allclose(r.uh_runoff.sum(0),
                          torch.ones(6, dtype=torch.float64))


def test_crop_ncrop_mismatch_is_error(tmp_path, grid):
    path = _copy(tmp_path, 'crop.nc')
    with netCDF4.Dataset(path, 'a') as ds:
        ds['Ncrop'][:] = 1
    with pytest.raises(P.VicWurInputError, match='ncrop'):
        P.read_crop(path, grid, 18)


def test_crop_text_missing_variable_is_error(tmp_path):
    src = open(R1 + '/wofost/crop_maize.txt').read()
    bad = tmp_path / 'c.txt'
    bad.write_text(src.replace('TCNT =', 'TCNX ='))
    with pytest.raises(P.VicWurInputError, match='TCNT'):
        P.parse_crop_file(str(bad))
    sc, tb = P.parse_crop_file(R1 + '/wofost/crop_maize.txt')
    assert len(sc) == 77 and len(tb) == 21 + (1 if 'VERNRTB' in tb else 0)


def test_global_file_parser():
    g = P.parse_global(GLOBAL)
    assert g['ROUTING'] == [['TRUE']]
    assert len(g['PLUGIN_FORCE_TYPE']) == 16


def test_missing_rout_steps_per_day_is_error(tmp_path):
    # rout_get_global_param.c:94-96: C stops when ROUT_STEPS_PER_DAY is absent
    txt = ''.join(l for l in open(GLOBAL)
                  if not l.upper().startswith('ROUT_STEPS_PER_DAY'))
    gp = tmp_path / 'global_no_rout.txt'
    gp.write_text(txt)
    with pytest.raises(P.VicWurInputError, match='ROUT_STEPS_PER_DAY'):
        P.load_vicwur_inputs(str(gp))
