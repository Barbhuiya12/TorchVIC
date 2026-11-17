# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""isc-cropsyst: the gridded CropSyst inputs of the ISC domain (tools/cropsyst/isc) and the banded member.

Without data: crop choice per tile, rotation codes, tile table over bands, soil retention, the member's band
option and its irrigation signals on bands. With the generated files (<data root>/cropsyst_india/isc): every rotation
loads through the cropsyst readers.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'cropsyst', 'isc'))
import _paths  # noqa: E402
import make_isc_cropsyst as G                                          # noqa: E402
import isc_inputs as I                                                 # noqa: E402


def _toy():
    nc_ = len(G.CLASS_MAP)
    cc = np.zeros((nc_, 2, 3))
    veg = np.full((nc_, 2, 3), -1)
    plant = np.full((nc_, 2, 3), np.nan)
    for k in range(nc_):
        veg[k] = 19 if k in (0, 1) else (13 if G.CLASS_MAP[k][1] == 'rf' else 18)
    cc[3, 0, 0], cc[2, 0, 0] = 0.2, 0.5          # rainfed: rice_rf beats wheat_rf
    plant[3, 0, 0], plant[2, 0, 0] = 319, 166
    cc[4, 0, 1], plant[4, 0, 1] = 0.4, 319       # irrigated wheat
    cc[0, 1, 2], plant[0, 1, 2] = 0.3, 196       # paddy
    ok = {t: np.ones((2, 3), bool) for t in G.CROP_TILES}
    ok[13][0, 1] = False
    return cc, veg, plant, ok


def test_choose_takes_the_dominant_crop_of_each_tile_and_its_sowing_day():
    cc, veg, plant, ok = _toy()
    sel = G.choose(cc, veg, plant, ok)
    k, sow, share = sel[13]
    assert k[0, 0] == 2 and sow[0, 0] == 166 and abs(share[0, 0] - 0.5 / 0.7) < 1e-6
    assert k[0, 1] == -1 and sow[0, 1] == 0            # no crop in the tile
    assert sel[18][0][0, 1] == 4 and sel[18][1][0, 1] == 319
    assert sel[19][0][1, 2] == 0 and sel[19][1][1, 2] == 196
    ok[13][0, 0] = False                                # the tile is not in the run
    assert G.choose(cc, veg, plant, ok)[13][0][0, 0] == -1


def test_rotation_codes_are_unique_per_crop_system_and_sowing_day_and_below_10000():
    cc, veg, plant, ok = _toy()
    codes, rc = G.rotation_table(G.choose(cc, veg, plant, ok))
    assert len(set(codes.values())) == len(codes) == 3
    assert all(G.CODE0 <= c < 10000 for c in codes.values())
    assert rc[13][0, 0] == codes[('rice_kharif', 'rf', 166)]
    assert rc[18][0, 1] == codes[('wheat', 'ir', 319)]
    assert rc[13][0, 1] == 0


def test_soil_retention_puts_the_wilting_point_and_field_capacity_at_vics_own():
    s = G.soil(np.array([0.3]), np.array([1300.0]), np.array([2600.0]), np.array([0.5]), np.array([0.3]))
    assert abs(s['max_moist'][0] - 150.0) < 1e-9          # 0.3 m x porosity 0.5
    fc = 0.3 + (0.5 - 0.3) / 0.7                          # soil_statics: wfc = wpwp + (wcr - wpwp)/0.7
    psi = lambda x: s['air_entry'][0] * x ** (-s['campbell_b'][0])
    assert s['air_entry'][0] < 0                          # a positive AE gave a positive potential and 1e31 mm uptake
    assert abs(psi(0.3) + 1500.0) < 1e-6 and abs(psi(fc) + 33.0) < 1e-6
    lim = G.soil(np.array([0.3]), np.array([1300.0]), np.array([2600.0]), np.array([0.95]), np.array([0.02]))
    assert 2.5 <= lim['campbell_b'][0] <= 12.0


def test_tile_table_has_one_tile_per_band_column_and_skips_missing_tiles():
    rot = np.array([[8001, 0], [0, 8002], [0, 0]])        # classes 13, 18, 19 x 2 cells
    cell = np.array([0, 0, 0, 1, 1])
    band = np.array([0, 1, 2, 0, 1])
    cv = np.zeros((5, 20))
    cv[:, 12] = 0.3
    cv[2, 12] = 0.0                                       # no VIC tile on column 2
    cv[:, 17] = 0.2
    tt = I.tile_table(rot, cell, band, cv)
    assert tt['cell'].tolist() == [0, 0, 1, 1] and tt['band'].tolist() == [0, 1, 0, 1]
    assert tt['tile'].tolist() == [12, 12, 17, 17] and tt['code'].tolist() == [8001, 8001, 8002, 8002]
    assert len({(c, t, b) for c, t, b in zip(tt['cell'], tt['tile'], tt['band'])}) == 4


def test_irrigation_block_codes_flood_on_paddy_furrow_on_other_irrigated():
    b = I.irrigation_block(4)
    assert b['code'].shape == (4, 2) and b['code'][0].tolist() == [73, 71]
    assert b['window'] == (1, 12)


def test_member_takes_bands_and_gives_band_resolved_irrigation_signals():
    torch = pytest.importorskip('torch')
    from torchvic.coupling import registry
    from torchvic.coupling.connectors import vicwur_common as VC
    from torchvic.coupling.spec import CouplingError
    mk = registry.get('vicwur_cropsyst').make
    cs = dict(tiles=[8000] * 3, tile_cell=[0, 0, 1], db_dir='x', soil={})
    with pytest.raises(CouplingError, match='band'):
        mk(cropsyst=cs, tile=[0, 0, 0], band=[0, 0], group=VC.WurGroup('b1'))
    with pytest.raises(CouplingError, match='one VIC tile and band'):
        mk(cropsyst=cs, tile=[0, 0, 0], band=[0, 0, 0], group=VC.WurGroup('b2'))
    g = VC.WurGroup('b3')
    m = mk(cropsyst=cs, tile=[5, 5, 5], band=[0, 1, 0], group=g)
    g.layout = lambda: dict(area_fract=torch.ones(2, 2))
    D = torch.float64
    m.dev = torch.device('cpu')
    m.fb = dict(present=torch.tensor([1.0, 1.0, 0.0], dtype=D), water_stress=torch.tensor([0.9, 0.5, 1.0], dtype=D),
                root_depth=torch.tensor([0.4, 0.6, 0.0], dtype=D))
    s = m.signals_for([5], 2)
    assert s['root_depth_mm'].shape == (2, 1, 2)
    assert s['root_depth_mm'][0, 0].tolist() == pytest.approx([400.0, 600.0])
    assert s['wsi'][0, 0].tolist() == pytest.approx([0.1, 0.5])
    assert s['active'].tolist() == [[True], [False]]
    # one band per tile (the CRB cases): the signals keep their [cell, class] shape
    g2 = VC.WurGroup('b4')
    m2 = mk(cropsyst=dict(tiles=[8000, 8000], tile_cell=[0, 1], db_dir='x', soil={}), tile=[5, 5], group=g2)
    g2.layout = lambda: dict(area_fract=torch.ones(2, 1))
    m2.dev = torch.device('cpu')
    m2.fb = dict(present=torch.ones(2, dtype=D), water_stress=torch.ones(2, dtype=D),
                 root_depth=torch.full((2,), 0.5, dtype=D))
    assert m2.signals_for([5], 2)['wsi'].shape == (2, 1)


GEN = os.path.join(I.INPUTS, 'isc_cropsyst.nc')


@pytest.mark.skipif(not os.path.exists(GEN), reason='generated ISC CropSyst inputs not present')
def test_generated_rotations_load_through_the_readers_and_cover_every_code():
    pytest.importorskip('torch')
    from torchvic.crop.cropsyst import calendars as cal
    inp = I.read_inputs()
    codes = sorted({int(c) for c in np.unique(inp['rot_code']) if c > 0})
    rs = cal.load_rotations(I.INPUTS + '/Database', codes)
    assert len(rs.codes) == len(codes) > 20
    assert (inp['share'][inp['rot_code'] > 0] > 0).all()
    assert np.isfinite(inp['soil']['max_moist'][:, np.isfinite(inp['soil']['max_moist'][0])]).all()


def test_engine_refuses_a_positive_air_entry_potential():
    torch = pytest.importorskip('torch')
    from torchvic.crop.cropsyst import engine as cse
    from torchvic.crop.cropsyst.readers import InputError
    cfg = dict(db_dir=I.INPUTS + '/Database', tiles=[8000], tile_cell=[0],
               soil=dict(thickness=[0.3, 1.0], max_moist=[100.0, 300.0], air_entry=[0.8, 0.8], campbell_b=[6.0, 6.0]))
    if not os.path.exists(I.INPUTS + '/Database/Rotation/Code/8000.CS_rotation'):
        pytest.skip('generated rotations not present')
    w = dict(tmax=[20.0] * 3, tmin=[10.0] * 3, srad=[200.0] * 3, rhmin=[50.0] * 3)
    with pytest.raises(InputError, match='negative'):
        cse.build_run(cfg, w, [25.0], [80.0], [2005] * 3, [1, 2, 3], 1, 'cpu')


def test_member_gives_land_use_the_crop_tiles_lai_by_column():
    """Regression: the land-use member asks the group's crop role for lai_prev_columns (WOFOST has it); the
    CropSyst member had none and a run with land use stopped on the first window."""
    torch = pytest.importorskip('torch')
    from torchvic.coupling import registry
    from torchvic.coupling.connectors import vicwur_common as VC
    D = torch.float64
    g = VC.WurGroup('lp')
    cs = dict(tiles=[8000] * 3, tile_cell=[0, 0, 1], db_dir='x', soil={})
    m = registry.get('vicwur_cropsyst').make(cropsyst=cs, tile=[2, 2, 2], band=[0, 1, 1], group=g)
    m.dev = torch.device('cpu')
    m.cell_t, m.vt_t, m.vb_t = (torch.as_tensor(x) for x in (m.tcell, m.vt, m.vb))
    cols = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    g.layout = lambda: dict(n_cell=2, n_tile=4, area_fract=torch.ones(2, 2))
    g.to_columns = lambda x: x[cols[:, 0], :, cols[:, 1]]
    assert m.lai_prev_columns() is None                      # before the first day
    m.fb = dict(lai=torch.tensor([1.5, 2.5, 3.5], dtype=D), uptake=torch.zeros(3, 3, dtype=D),
                canopy_take=torch.zeros(3, dtype=D), green_cover=torch.zeros(3, dtype=D),
                cover_total=torch.zeros(3, dtype=D), present=torch.ones(3, dtype=D))
    lai, mask = m.lai_prev_columns()
    assert lai.shape == (4, 4) and mask.shape == (4, 4)
    # columns (cell, band): (0,0) (0,1) (1,0) (1,1); the tiles sit on VIC tile 2
    assert lai[:, 2].tolist() == [1.5, 2.5, 0.0, 3.5]
    assert mask[:, 2].tolist() == [True, True, False, True] and not mask[:, 0].any()


def test_rabi_dominant_tile_in_a_cold_cell_gets_no_crop_and_nothing_is_substituted():
    cc, veg, plant, ok = _toy()
    cold = np.zeros((2, 3), bool)
    cold[0, 0] = True
    cold[0, 1] = True
    cc[21, 0, 0], plant[21, 0, 0] = 0.9, 319          # rainfed mustard: biggest, rabi
    assert G.choose(cc, veg, plant, ok)[13][0][0, 0] == 21          # no mask: mustard
    k, sow, share = G.choose(cc, veg, plant, ok, cold)[13]
    assert k[0, 0] == -1 and sow[0, 0] == 0 and share[0, 0] == 0    # no rice_rf substitute
    cc[21, 0, 0] = 0.0                                               # dominant is kharif rice: kept in the cold cell
    assert G.choose(cc, veg, plant, ok, cold)[13][0][0, 0] == 2
    assert G.choose(cc, veg, plant, ok, cold)[18][0][0, 1] == -1     # cold, irrigated wheat 319: none
    assert G.choose(cc, veg, plant, ok, np.zeros((2, 3), bool))[18][0][0, 1] == 4   # warm: untouched


def test_cold_mask_is_nov_dec_mean_tmin_below_zero(tmp_path):
    import netCDF4 as nc
    f = str(tmp_path / 'd.%d.nc')
    d = nc.Dataset(f % 2004, 'w')
    for n, m in (('time', 366), ('lat', 1), ('lon', 2)):
        d.createDimension(n, m)
    t = d.createVariable('time', 'f8', ('time',)); t.units = 'days since 2004-01-01'; t[:] = np.arange(366)
    v = d.createVariable('tasmin', 'f4', ('time', 'lat', 'lon'))
    x = np.full((366, 1, 2), 10.0); x[305:, 0, 0] = -5.0; x[:305, 0, 1] = -30.0   # Jan-Oct cold in cell 1 only
    v[:] = x
    d.close()
    assert G.cold_mask([2004], f).tolist() == [[True, False]]
    assert G.is_rabi(np.array([319., 166., np.nan])).tolist() == [True, False, False]


@pytest.mark.skipif(
    not os.path.exists(_paths.data('cropsyst_india/Database/Crop/Name/maize.crp')),
    reason='CropSyst India crop database absent: set TORCHVIC_DATA_ROOT')
def test_thermal_rule_drops_a_crop_that_cannot_mature_in_the_one_year_rotation():
    tm = np.zeros((365, 1, 3))
    tm[:, 0, 0] = 25.0                                    # warm: 17 x 200 = 3400 GDD (base 8)
    tm[:, 0, 1] = 12.0                                    # cold: 4 x 200 = 800 < maize maturity 1679
    tm[:, 0, 2] = np.nan
    b, c, m = G.crop_thermal('maize')
    assert (b, c, m) == (8.0, 30.0, 1679.0)
    r = G.thermal_infeasible(tm, b, c, m, 166)
    assert r.tolist() == [[False, True, False]]
    tm[:, 0, 0] = np.where(np.arange(365) < 200, 25.0, -20.0)    # warm days 1-200 only: sowing day wraps the year
    assert not G.thermal_infeasible(tm, b, c, m, 1)[0, 0] and G.thermal_infeasible(tm, b, c, m, 201)[0, 0]
    cc, veg, plant, ok = _toy()
    sel = G.choose(cc, veg, plant, ok, infeasible=lambda k, s: np.array([[True, False, False], [False, False, False]]))
    assert sel[13][0][0, 0] == -1 and sel[18][0][0, 1] == 4
    assert sel[19][0][1, 2] == 0
    sel = G.choose(cc, veg, plant, ok, infeasible=lambda k, s: np.ones((2, 3), bool))
    assert sel[18][0][0, 1] == 4                           # wheat sown 319 (rabi) is exempt
    assert sel[13][0][0, 0] == -1 and sel[19][0][1, 2] == -1      # kharif sowings dropped


def test_kharif_pulse_sowing_gets_no_chickpea_and_rabi_pulse_keeps_it():
    cc, veg, plant, ok = _toy()
    k = [i for i, m in enumerate(G.CLASS_MAP) if m == ('chickpea_pulses', 'rf')][0]
    cc[k, 0, 0], plant[k, 0, 0] = 0.9, 166                # kharif pulses beat rice_rf in the tile
    cc[k, 1, 0], plant[k, 1, 0] = 0.9, 288                # rabi chickpea
    veg[k] = 13
    sel = G.choose(cc, veg, plant, ok)
    assert sel[13][0][0, 0] == -1 and sel[13][1][0, 0] == 0
    assert sel[13][0][1, 0] == k and sel[13][1][1, 0] == 288
