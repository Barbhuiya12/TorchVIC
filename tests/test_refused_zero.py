"""Nothing stays refused. The options VIC's image driver (or a
release of VIC) stops on, run here as the driver that does run them runs
them:

* ARNO_PARAMS and NIJSSEN2001_BASEFLOW (VIC 4.2.d's spellings; every
  release stops on them and names BASEFLOW): read as that BASEFLOW line,
  so the run is the BASEFLOW run, bit for bit.
* ATMOS_STEPS_PER_DAY (the classic driver's): validated as the classic
  driver validates it, no effect on results.
* COMPUTE_TREELINE and JULY_TAVG_SUPPLIED (the classic driver's): against
  the pristine `vic_classic.exe` (verification/classic_rows.py).
* OUTPUT_FORCE (VIC 4.2.d's): the forcing files, against `vicNl`.

"COMPUTE_TREELINE without JULY_TAVG_SUPPLIED" and a lake with a treeline
stay refused: VIC itself stops on them in every driver.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import _paths                                           # noqa: E402

sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
matrix = pytest.importorskip('matrix')

from torchvic import case as C                          # noqa: E402
from torchvic.core import snowbands, treeline           # noqa: E402
from torchvic.io import history as H                    # noqa: E402
from torchvic.io import options as O                    # noqa: E402
from torchvic.io.global_param import GlobalParam        # noqa: E402


def _global(tmp_path, extra=(), case='wg', mode='WB', edit=None, name='g',
            **kw):
    r = matrix.row(case, mode, **kw)
    txt = matrix.global_text(r, str(tmp_path), str(tmp_path), ['OUT_PREC'],
                             extra_lines=list(extra))
    for a, b in (edit or {}).items():
        assert a in txt, a
        txt = txt.replace(a, b)
    g = tmp_path / ('%s.txt' % name)
    g.write_text(txt)
    return str(g)


# -- ARNO_PARAMS, NIJSSEN2001_BASEFLOW ---------------------------------------
@pytest.mark.parametrize('old,new', [
    ('ARNO_PARAMS TRUE', 'NIJSSEN2001'), ('ARNO_PARAMS FALSE', 'ARNO'),
    ('NIJSSEN2001_BASEFLOW TRUE', 'NIJSSEN2001'),
    ('NIJSSEN2001_BASEFLOW FALSE', 'ARNO'),
    ('arno_params true', 'NIJSSEN2001')])
def test_4_2_d_baseflow_spellings_are_the_baseflow_line(tmp_path, old, new):
    g = tmp_path / 'g.txt'
    g.write_text('FULL_ENERGY FALSE\n%s\nNODES 3\n' % old)
    gp = GlobalParam(str(g))
    assert gp.get('BASEFLOW') == new
    assert ('BASEFLOW', [new]) in gp.lines
    assert gp.translated and 'BASEFLOW %s' % new in gp.translated[0]
    assert old.split()[0].upper() not in gp.raw      # no second reading
    refused, announced = O.check(gp)
    assert not [r for r in refused if 'ARNO' in r or 'NIJSSEN' in r]
    assert any('BASEFLOW' in a and 'spelling' in a for a in announced)


def test_the_old_spelling_in_the_options_table_is_not_refused():
    for key in ('ARNO_PARAMS', 'NIJSSEN2001_BASEFLOW'):
        for v in ('TRUE', 'FALSE'):
            assert O.disposition(key, v)[0] == O.IMPLEMENTED


@pytest.mark.parametrize('old,new', [
    ('ARNO_PARAMS TRUE', 'BASEFLOW NIJSSEN2001'),
    ('NIJSSEN2001_BASEFLOW TRUE', 'BASEFLOW NIJSSEN2001'),
    ('ARNO_PARAMS FALSE', 'BASEFLOW ARNO'),
    ('NIJSSEN2001_BASEFLOW FALSE', 'BASEFLOW ARNO')])
def test_the_old_spelling_runs_bit_identically_to_baseflow(tmp_path, old, new):
    """Walnut Gulch, water balance, two days: the output of the run is the
    BASEFLOW run's, every element, not to a tolerance."""
    if not os.path.exists(matrix.CASES['wg']['domain']):
        pytest.skip('Walnut Gulch sample data not present')
    ga = _global(tmp_path, extra=[new], name='a')
    gb = _global(tmp_path, extra=[old], name='b')
    dev = torch.device('cpu')
    with torch.inference_mode():
        oa = C.load(ga, device=dev, steps=16).run()
        ob = C.load(gb, device=dev, steps=16).run()
    n = 0
    for k, v in oa.items():
        if torch.is_tensor(v):
            assert torch.equal(v, ob[k]), k
            n += 1
    assert n > 10


# -- ATMOS_STEPS_PER_DAY -----------------------------------------------------
def _gp(tmp_path, *lines):
    g = tmp_path / 'a.txt'
    g.write_text('\n'.join(lines) + '\n')
    return GlobalParam(str(g))


@pytest.mark.parametrize('atmos,ok', [
    ('24', True), ('48', True), ('72', True), ('0', True), ('x', True),
    ('3', False), ('30', False), ('25', False), ('1500', False),
    ('1440', True)])
def test_atmos_steps_per_day_is_validated_as_the_classic_driver_does(
        tmp_path, atmos, ok):
    # get_global_param.c:811-856: 0 -> 24; >= 4; > 24 a multiple of 24;
    # <= 1440; a multiple of the model and snow steps
    gp = _gp(tmp_path, 'MODEL_STEPS_PER_DAY 24', 'SNOW_STEPS_PER_DAY 24',
             'ATMOS_STEPS_PER_DAY %s' % atmos)
    errs = O.atmos_steps_errors(gp)
    assert (not errs) == ok, errs
    refused, _ = O.check(gp)
    assert (not [r for r in refused if 'ATMOS' in r]) == ok


def test_atmos_steps_must_divide_by_the_model_step(tmp_path):
    gp = _gp(tmp_path, 'MODEL_STEPS_PER_DAY 8', 'SNOW_STEPS_PER_DAY 8',
             'ATMOS_STEPS_PER_DAY 12')
    errs = O.atmos_steps_errors(gp)
    assert errs and 'MODEL_STEPS_PER_DAY' in errs[0]


def test_atmos_steps_per_day_changes_no_result(tmp_path):
    if not os.path.exists(matrix.CASES['wg']['domain']):
        pytest.skip('Walnut Gulch sample data not present')
    ga = _global(tmp_path, name='a')
    gb = _global(tmp_path, extra=['ATMOS_STEPS_PER_DAY 48'], name='b')
    said = []
    with torch.inference_mode():
        oa = C.load(ga, device=torch.device('cpu'), steps=16).run()
        ob = C.load(gb, device=torch.device('cpu'), steps=16,
                    log=said.append).run()
    assert any('ATMOS_STEPS_PER_DAY' in s and 'no effect' in s for s in said)
    for k, v in oa.items():
        if torch.is_tensor(v):
            assert torch.equal(v, ob[k]), k


# -- COMPUTE_TREELINE --------------------------------------------------------
def test_compute_treeline_value_is_read_as_the_classic_driver_reads_it(
        tmp_path):
    # get_global_param.c:189-197: FALSE off; anything else on, atoi = class
    for text, want in (('FALSE', (False, -1)), ('-1', (True, -1)),
                       ('7', (True, 7)), ('TRUE', (True, 0)),
                       ('12abc', (True, 12))):
        assert O.treeline(_gp(tmp_path, 'COMPUTE_TREELINE %s' % text)) == want
    assert O.treeline(_gp(tmp_path, 'NODES 3')) == (False, -1)
    assert O.treeline(_gp(tmp_path, 'COMPUTE_TREELINE 5',
                          'COMPUTE_TREELINE FALSE')) == (False, 5)   # the C keeps the class


def test_treeline_needs_july_tavg_supplied_in_both_drivers(tmp_path):
    # get_global_param.c:971-973 (image), :1299-1303 (classic): VIC stops
    off = 'FROZEN_SOIL FALSE'            # resets IMPLICIT and EXP_TRANS
    gp = _gp(tmp_path, off, 'COMPUTE_TREELINE -1')
    refused, _ = O.check(gp)
    assert any('JULY_TAVG_SUPPLIED FALSE' in r for r in refused)
    gp = _gp(tmp_path, off, 'COMPUTE_TREELINE -1', 'JULY_TAVG_SUPPLIED TRUE')
    assert not O.check(gp)[0]
    gp = _gp(tmp_path, off, 'COMPUTE_TREELINE FALSE')
    assert not O.check(gp)[0]
    assert O.disposition('COMPUTE_TREELINE', '12')[0] == O.IMPLEMENTED
    assert O.disposition('JULY_TAVG_SUPPLIED', 'TRUE')[0] == O.IMPLEMENTED


def test_above_treeline_is_the_lapsed_july_temperature_at_most_10c():
    # compute_treeline.c:76-85; July_Tavg -999 is skipped (vic_force.c:292)
    july = np.array([12.0, 12.0, 12.0, 8.0, -999.0, 10.0])
    tf = np.array([0.0, -2.0, -2.0001, 0.0, -5.0, 0.0])
    assert treeline.above_treeline(july, tf).tolist() == [
        False, True, True, True, False, True]


def test_overstory_is_dropped_and_the_rest_reweighted_above_the_treeline():
    # put_data.c:76-103, :180-190: Cv 0.5 overstory, 0.3 and 0.2 not
    cv = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [1.0, 0.0, 0.0]])
    ov = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
    out = treeline.adjust_cover(cv, ov, np.array([True, False, True]))
    np.testing.assert_allclose(out[0], [0.0, 0.6, 0.4], atol=1e-15)
    np.testing.assert_array_equal(out[1], cv[1])
    assert out[2].tolist() == [0.0, 0.0, 0.0]        # nothing is left
    t = treeline.adjust_cover(torch.tensor(cv), torch.tensor(ov),
                              np.array([True, False, True]))
    np.testing.assert_allclose(t.numpy(), out, atol=1e-15)


class _Par(object):
    def __init__(self, cv, ov, nz=2):
        n, c = cv.shape[1], cv.shape[0]
        self.veg = dict(Cv=cv.astype(float).copy(),
                        overstory=np.repeat(np.asarray(ov, float)[:, None],
                                            n, 1),
                        LAI=np.ones((c, 12, n)), albedo=np.ones((c, 12, n)),
                        veg_rough=np.ones((c, 12, n)),
                        displacement=np.ones((c, 12, n)),
                        root_depth=np.arange(c * nz * n, dtype=float)
                        .reshape(c, nz, n) + 1.0,
                        root_fract=np.ones((c, nz, n)) * 0.5)
        self.veg_class_ids = list(range(10, 10 + c))


def test_the_tile_vic_adds_for_overstory_only_cells():
    # read_vegparam.c:371-433: classes 10, 11 overstory, 12 not, 13 = bare
    cv = np.array([[0.6, 0.5, 0.2], [0.4, 0.5, 0.2], [0.0, 0.0, 0.6],
                   [0.0, 0.0, 0.0]])
    # cell 0: overstory only, sums to 1 -> acts; cell 1: same; cell 2: has 12
    p = _Par(cv, [1, 1, 0, 0])
    assert treeline.add_above_treeline_tile(p, -1) == 2
    np.testing.assert_allclose(p.veg['Cv'][:, 0], [0.5995, 0.3995, 0, 0.001])
    np.testing.assert_allclose(p.veg['Cv'][:, 1], [0.4995, 0.4995, 0, 0.001])
    np.testing.assert_array_equal(p.veg['Cv'][:, 2], cv[:, 2])
    p = _Par(cv, [1, 1, 0, 0])
    assert treeline.add_above_treeline_tile(p, 12) == 2
    np.testing.assert_allclose(p.veg['Cv'][:, 0], [0.5995, 0.3995, 0.001, 0])
    # the new tile has the last tile's roots and nothing else (:398-409);
    # its monthly fields are the zeros VIC's calloc leaves
    np.testing.assert_array_equal(p.veg['root_depth'][2, :, 0],
                                  p.veg['root_depth'][1, :, 0])
    for k in ('LAI', 'albedo', 'veg_rough', 'displacement'):
        assert p.veg[k][2, :, 0].tolist() == [0.0] * 12
        assert p.veg[k][2, :, 2].tolist() == [1.0] * 12      # other cell
    assert p.fcanopy_from_default


def test_the_default_tile_class_must_exist_and_have_no_overstory():
    cv = np.array([[1.0], [0.0], [0.0]])
    with pytest.raises(ValueError, match='not defined in the vegetation'):
        treeline.add_above_treeline_tile(_Par(cv, [1, 1, 0]), 99)
    with pytest.raises(ValueError, match='to have overstory'):
        treeline.add_above_treeline_tile(_Par(cv, [1, 1, 0]), 11)


def test_classic_band_mean_replaces_the_cell_elevation_only_beyond_1m():
    # read_snowband.c:79-85 (the image driver's test always replaces it)
    area = np.array([[0.5, 0.5], [0.5, 0.5]])
    elev = np.array([[1000.0, 1000.0], [1002.0, 1004.0]])      # means 1001, 1002
    pf = np.array([[0.5, 0.5], [0.5, 0.5]])
    cell = np.array([1000.5, 1000.5])
    img = snowbands.band_factors(area, elev, pf)[3]
    cls = snowbands.band_factors(area, elev, pf, cell_elev=cell)[3]
    np.testing.assert_allclose(img, [1001.0, 1002.0])
    np.testing.assert_allclose(cls, [1000.5, 1002.0])


# -- COMPUTE_TREELINE against the pristine classic driver --------------------
CLASSIC_EXE = os.environ.get('TORCHVIC_VIC_CLASSIC') or os.path.join(
    matrix.WORK, 'item16b_refusedzero', 'build', 'vic', 'drivers', 'classic',
    'vic_classic.exe')


def _classic_row(rid, days):
    import classic_rows as CR
    if not os.path.exists(CLASSIC_EXE):
        pytest.skip('%s is not built (see verification/classic_rows.py)'
                    % CLASSIC_EXE)
    r = [x for x in CR.CLASSIC_ROWS if x['id'] == rid][0]
    c = CR.CASES[r['case']]
    if not os.path.exists(c['forcing'] + '48.1875_-120.6875') and \
            r['case'] == 'stehe':
        pytest.skip('Stehekin classic sample data not present')
    CR.VIC_CLASSIC = CLASSIC_EXE
    return CR.one(r, days)


@pytest.mark.parametrize('rid', [
    'stehe_classic_EB_treeline_bare',
    'stehe_classic_EB_treeline_overstory_class7',
    'stehe_classic_WB_treeline_overstory_bare'])
def test_stehekin_treeline_agrees_with_vic_classic(rid):
    res = _classic_row(rid, 3)
    assert res['grade'] == 'pass', res.get('grade')
    assert res['score_water'] < 1e-8 and res['score_energy'] < 1e-8


@pytest.mark.parametrize('rid', ['findley_classic_EB_treeline_above',
                                 'findley_classic_EB_treeline_off999'])
def test_findley_treeline_agrees_with_vic_classic(rid):
    res = _classic_row(rid, 6)
    assert res['grade'] == 'pass'
    assert res['score_water'] < 1e-6 and res['score_energy'] < 1e-6


def test_the_treeline_changes_what_vic_classic_writes():
    """The rows above would prove nothing if the option were a no-op in VIC."""
    import classic_rows as CR
    a = _classic_row('findley_classic_EB_base', 6)
    b = _classic_row('findley_classic_EB_treeline_above', 6)
    ev = lambda r: [v for v in r['vars'] if v['var'] == 'OUT_EVAP'][0]  # noqa
    assert a['grade'] == b['grade'] == 'pass'
    va = CR.read_vic(os.path.join(CR.OUT, 'findley_classic_EB_base'),
                     [47.3188], [-121.5853])
    vb = CR.read_vic(os.path.join(CR.OUT, 'findley_classic_EB_treeline_above'),
                     [47.3188], [-121.5853])
    assert np.abs(va['OUT_EVAP'] - vb['OUT_EVAP']).max() > 1e-3
    assert np.abs(va['OUT_SWE'] - vb['OUT_SWE']).max() > 0.1


def test_treeline_off_for_a_cell_with_july_tavg_minus_999_is_the_base_run():
    a = _classic_row('findley_classic_EB_base', 4)
    b = _classic_row('findley_classic_EB_treeline_off999', 4)
    assert a['grade'] == b['grade'] == 'pass'


# -- OUTPUT_FORCE ------------------------------------------------------------
def test_force_data_partitions_rain_and_snow_and_clamps_vp():
    # write_forcing_file.c:60-71; initialize_atmos.c:1340-1344
    f = {'tair': torch.tensor([[-5.0], [0.5], [0.0], [-0.5], [10.0]]),
         'prcp': torch.full((5, 1), 2.0),
         'shortwave': torch.full((5, 1), 100.0),
         'longwave': torch.full((5, 1), 250.0),
         'press': torch.full((5, 1), 90000.0),
         'vp': torch.tensor([[100.0], [600.0], [700.0], [800.0], [5000.0]]),
         'wind': torch.tensor([[0.0], [3.0], [3.0], [3.0], [3.0]])}
    d = H.force_data_42d(f)
    assert d['OUT_RAINF'].ravel().tolist() == [0.0, 2.0, 1.0, 0.0, 2.0]
    assert d['OUT_SNOWF'].ravel().tolist() == [2.0, 0.0, 1.0, 2.0, 0.0]
    assert d['OUT_WIND'].ravel().tolist() == [0.1, 3.0, 3.0, 3.0, 3.0]
    # 10 C: svp 1.2279 kPa < vp 5 kPa -> vp := svp, vpd 0
    assert d['OUT_VPD'][4, 0] == 0.0 and d['OUT_VP'][4, 0] < 5.0
    assert d['OUT_PRESSURE'][0, 0] == 90.0
    assert d['OUT_CATM'][0, 0] == 0.0               # not supplied here
    a = H.force_data_42d(f, alma=True, dt_hours=3)
    assert a['OUT_PREC'][0, 0] == 2.0 / 10800
    assert a['OUT_AIR_TEMP'][0, 0] == -5.0 + 273.15
    assert a['OUT_PRESSURE'][0, 0] == 90.0 * 1000


def test_force_data_estimates_what_4_2_d_estimates():
    f = {'tair': torch.tensor([[10.0]]), 'prcp': torch.zeros(1, 1),
         'shortwave': torch.zeros(1, 1), 'longwave': torch.zeros(1, 1),
         'vp': torch.tensor([[800.0]])}
    d = H.force_data_42d(f, elevation=[1000.0])
    import math
    p = 101300. * math.exp(-1000.0 * 9.81 / (287 * (273.15 + 10.0
                                                    + 0.5 * 1000.0 * -0.0065)))
    assert abs(d['OUT_PRESSURE'][0, 0] - p / 1000.) < 1e-12
    assert d['OUT_DENSITY'][0, 0] == p / (287.0 * (273.15 + 10.0))
    with pytest.raises(ValueError, match='no pressure'):
        H.force_data_42d(f)


def test_coszen_is_vicnls_to_the_last_digit():
    # OUT_COSZEN of vicNl 4.2.d, Stehekin cell 48.1875_-120.6875, forcing
    # step 100 (day 5, hour 100 counted from the run's start): PI is
    # vicNl_def.h's 3.1415927 and the hour is not reduced to the day
    got = H.coszen_42d(48.1875, -120.6875, -120.0, 5, 100)
    assert got == -0.5871545784331333
    assert H.coszen_42d(48.1875, -120.6875, -120.0, 1, 0) == \
        -0.904359229732 or abs(
            H.coszen_42d(48.1875, -120.6875, -120.0, 1, 0)
            + 0.904359229732) < 1e-11


def test_the_default_output_forcing_file_is_set_output_defaults(tmp_path):
    g = tmp_path / 'g.txt'
    g.write_text('TIME_STEP 1\nOUTPUT_FORCE TRUE\nGRID_DECIMAL 4\n')
    s = H.parse_output_42d(str(g))
    assert s.output_force
    assert [f.prefix for f in s.files] == ['full_data']
    assert s.files[0].names == ['OUT_PREC', 'OUT_AIR_TEMP', 'OUT_SHORTWAVE',
                                'OUT_LONGWAVE', 'OUT_DENSITY', 'OUT_PRESSURE',
                                'OUT_VP', 'OUT_WIND']
    assert (s.type_of('OUT_PREC'), s.mult_of('OUT_PREC')) == (
        'OUT_TYPE_USINT', 40.0)
    assert (s.type_of('OUT_AIR_TEMP'), s.mult_of('OUT_AIR_TEMP')) == (
        'OUT_TYPE_SINT', 100.0)
    assert s.format_of('OUT_VP') == '%.4f'


STEHE = os.path.join(_paths.SAMPLES, 'classic', 'Stehekin')
VICNL = os.environ.get('VIC42D_EXE') or os.path.join(_paths.VIC42D_SRC,
                                                     'vicNl')


def _stehe_globals(tmp, name, extra='', daily=False):
    """A VIC 4.2.d global file and the same inputs as a classic global file
    TorchVIC reads, both under OUTPUT_FORCE TRUE."""
    d = os.path.join(str(tmp), name)
    for sub in ('vic', 'tv'):
        os.makedirs(os.path.join(d, sub))
    if daily:
        fdir = os.path.join(d, 'daily')
        os.makedirs(fdir)
        for fn in os.listdir(os.path.join(STEHE, 'forcings')):
            if not fn.startswith('full_data_'):
                continue
            a = np.loadtxt(os.path.join(STEHE, 'forcings', fn))
            x = a[:240].reshape(10, 24, -1)
            out = np.stack([x[:, :, 0].sum(1), x[:, :, 1].max(1),
                            x[:, :, 1].min(1), x[:, :, 7].mean(1)], 1)
            np.savetxt(os.path.join(fdir, fn), out, fmt='%.4f',
                       delimiter='\t')
        forcing = ('FORCING1 %s/full_data_\nFORCE_FORMAT ASCII\nN_TYPES 4\n'
                   'FORCE_TYPE PREC\nFORCE_TYPE TMAX\nFORCE_TYPE TMIN\n'
                   'FORCE_TYPE WIND\nFORCE_DT 24\n' % fdir)
        startday = '02'
    else:
        forcing = ('FORCING1 %s/forcings/full_data_\nFORCE_FORMAT ASCII\n'
                   'N_TYPES 8\nFORCE_TYPE PREC\nFORCE_TYPE AIR_TEMP\n'
                   'FORCE_TYPE SHORTWAVE\nFORCE_TYPE LONGWAVE\n'
                   'FORCE_TYPE DENSITY\nFORCE_TYPE PRESSURE\nFORCE_TYPE VP\n'
                   'FORCE_TYPE WIND\nFORCE_DT 1\n' % STEHE)
        startday = '01'
    body = ('NLAYER 3\nNODES 3\nTIME_STEP 1\nSNOW_STEP 1\nSTARTYEAR 1949\n'
            'STARTMONTH 01\nSTARTDAY %s\nSTARTHOUR 00\nENDYEAR 1949\n'
            'ENDMONTH 01\nENDDAY 10\nFULL_ENERGY FALSE\nFROZEN_SOIL FALSE\n'
            '%sFORCEYEAR 1949\nFORCEMONTH 01\nFORCEDAY 01\nFORCEHOUR 00\n'
            'GRID_DECIMAL 4\nWIND_H 10.0\nSOIL %s/parameters/Stehekin_soil.txt'
            '\nBASEFLOW ARNO\nJULY_TAVG_SUPPLIED FALSE\nORGANIC_FRACT FALSE\n'
            'VEGLIB %s/parameters/Stehekin_veglib.txt\nVEGPARAM '
            '%s/parameters/Stehekin_vegparam.txt\nROOT_ZONES 3\n'
            'VEGPARAM_LAI TRUE\nLAI_SRC FROM_VEGPARAM\nSNOW_BAND 5 '
            '%s/parameters/Stehekin_snowbands.txt\nOUTPUT_FORCE TRUE\n'
            % (startday, forcing, STEHE, STEHE, STEHE, STEHE))
    g42 = os.path.join(d, 'g42.txt')
    open(g42, 'w').write(body + 'RESULT_DIR %s/vic\n%s' % (d, extra))
    tv = (body.replace('TIME_STEP 1', 'MODEL_STEPS_PER_DAY 24')
          .replace('SNOW_STEP 1', 'SNOW_STEPS_PER_DAY 24')
          .replace('NLAYER 3\n', '').replace('STARTHOUR 00\n', '')
          .replace('N_TYPES 8\n', '').replace('N_TYPES 4\n', '')
          .replace('FORCE_DT 1\n', '').replace('FORCE_DT 24\n', '')
          .replace('FORCEHOUR 00\n', '')
          .replace('FORCEDAY 01\n', 'FORCEDAY 01\nFORCE_STEPS_PER_DAY %d\n'
                   % (1 if daily else 24)))
    gtv = os.path.join(d, 'gtv.txt')
    open(gtv, 'w').write(tv + 'RESULT_DIR %s/tv\n%s' % (d, extra))
    return d, g42, gtv


def _output_force_matches_vicnl(tmp, name, extra='', daily=False,
                                bounds=None):
    """TorchVIC's OUTPUT_FORCE files against vicNl's, 16 cells.

    Byte for byte, except the columns named in `bounds` ({column index:
    largest allowed |difference|}): those hold quantities the MTCLIM
    disaggregation produces, which since the module was rebuilt on MetSim
    (licence; CHANGELOG.md, 0.4.7) no longer reproduce VIC 4.2.d's. Every
    other column must still be identical to the byte, and the named ones
    must stay within the bound set just above their measured distance. The
    files must be ASCII for the numeric comparison."""
    if not os.path.exists(VICNL):
        pytest.skip('%s is not built' % VICNL)
    if not os.path.isdir(STEHE):
        pytest.skip('Stehekin classic sample data not present')
    d, g42, gtv = _stehe_globals(tmp, name, extra, daily)
    r = subprocess.run([VICNL, '-g', g42], capture_output=True, text=True,
                       cwd=d, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    case = C.load(gtv, device=torch.device('cpu'), log=lambda *a: None)
    assert isinstance(case, C.ForceCase)
    paths = case.write_forcing(log=lambda *a: None)
    theirs = sorted(os.listdir(os.path.join(d, 'vic')))
    assert sorted(os.path.basename(p) for p in paths) == theirs
    assert len(theirs) == 16
    bounds = bounds or {}
    bad = []
    for f in theirs:
        a = open(os.path.join(d, 'vic', f), 'rb').read()
        b = open(os.path.join(d, 'tv', f), 'rb').read()
        if a == b:
            continue
        if not bounds:
            bad.append(f)
            continue
        x = np.loadtxt(os.path.join(d, 'vic', f))
        y = np.loadtxt(os.path.join(d, 'tv', f))
        assert x.shape == y.shape, f
        for j in range(x.shape[1]):
            if j in bounds:
                if np.abs(y[:, j] - x[:, j]).max() > bounds[j]:
                    bad.append((f, j, float(np.abs(y[:, j] - x[:, j]).max())))
            elif not np.array_equal(x[:, j], y[:, j]):
                bad.append((f, j))
    assert not bad, bad[:3]
    return d


def test_output_force_is_vicnls_to_the_byte(tmp_path):
    _output_force_matches_vicnl(tmp_path, 'plain')


def test_output_force_binary_with_header_is_vicnls_to_the_byte(tmp_path):
    _output_force_matches_vicnl(tmp_path, 'bin',
                                'BINARY_OUTPUT TRUE\nPRT_HEADER TRUE\n')


def test_output_force_alma_with_header_is_vicnls_to_the_byte(tmp_path):
    _output_force_matches_vicnl(tmp_path, 'alma',
                                'ALMA_OUTPUT TRUE\nPRT_HEADER TRUE\n')


def test_output_force_daily_forcing_through_mtclim_is_near_vicnls(
        tmp_path):
    """TMAX/TMIN/PREC/WIND daily, disaggregated to the hour: the soil file's
    time zone, floats for lat/lng/elevation, and no file margin (vicNl sees
    only its own window), 16 cells, 10 days.

    Precipitation and wind are still VIC 4.2.d's to the byte. The other
    columns are no longer, because MTCLIM is MetSim's now. Measured
    (TorchVIC minus vicNl, over 3456 hourly records, mean/max |difference|):

        AIR_TEMP   rmse 0.39 C       max 1.85 C
        SHORTWAVE  rmse 2.4 W/m2     max 17.2 W/m2
        LONGWAVE   rmse 1.9 W/m2     max 6.3 W/m2
        DENSITY    rmse 0.0014       max 0.0063 kg/m3
        PRESSURE   mean +0.026       max 0.122 kPa (101325 Pa and 9.80665 in
                                     place of 101300 and 9.81)
        VP         rmse 0.005        max 0.020 kPa

    The bounds below are those maxima plus about 10 %.
    """
    _output_force_matches_vicnl(
        tmp_path, 'daily', daily=True,
        bounds={1: 2.04, 2: 19.0, 3: 6.95, 4: 0.0070, 5: 0.135, 6: 0.0218})


def test_output_force_every_variable_is_vicnls(tmp_path):
    """Hourly forcing: every variable is VIC 4.2.d's to the byte except FDIR,
    the direct fraction of the shortwave, which VIC derives from MTCLIM's
    transmittance and which is now one minus the Erbs et al. daily diffuse
    fraction (MetSim has none). Measured: mean +0.044, rmse 0.076, max
    0.156 (the bound is 0.172)."""
    names = ('PREC RAINF SNOWF AIR_TEMP SHORTWAVE LONGWAVE DENSITY PRESSURE '
             'VP VPD QAIR REL_HUMID WIND CATM COSZEN FDIR PAR TSKC').split()
    extra = 'N_OUTFILES 1\nOUTFILE forc %d\n' % len(names) + ''.join(
        'OUTVAR OUT_%s %%.4f\n' % n for n in names)
    _output_force_matches_vicnl(tmp_path, 'allvars', extra,
                                bounds={names.index('FDIR'): 0.172})


def test_output_force_false_is_a_normal_model_run(tmp_path):
    if not os.path.exists(matrix.CASES['wg']['domain']):
        pytest.skip('Walnut Gulch sample data not present')
    g = _global(tmp_path, extra=['OUTPUT_FORCE FALSE'])
    c = C.load(g, device=torch.device('cpu'), steps=4)
    assert not isinstance(c, C.ForceCase)


def test_a_force_case_has_nothing_to_simulate(tmp_path):
    if not os.path.exists(matrix.CASES['wg']['domain']):
        pytest.skip('Walnut Gulch sample data not present')
    g = _global(tmp_path, extra=['OUTPUT_FORCE TRUE'])
    c = C.load(g, device=torch.device('cpu'), steps=4)
    assert isinstance(c, C.ForceCase)
    with pytest.raises(ValueError, match='skips the model'):
        c.run()


# -- COMPUTE_TREELINE on image-driver inputs ---------------------------------
def test_treeline_runs_on_image_inputs_and_needs_july_tavg(tmp_path):
    """The parameter file's `July_Tavg` (the image counterpart of the soil
    file's last column) drives the same treeline: Stehekin, five bands."""
    xr = pytest.importorskip('xarray')
    c = matrix.CASES['stehe']
    if not os.path.exists(c['params']):
        pytest.skip('Stehekin image sample data not present')
    extra = ['COMPUTE_TREELINE -1', 'JULY_TAVG_SUPPLIED TRUE']
    # no July_Tavg: refused by name (before and after the options check)
    g = _global(tmp_path, case='stehe', mode='WB', extra=extra,
                SNOW_BAND='TRUE', name='nojuly')
    with pytest.raises(ValueError, match='July_Tavg'):
        C.load(g, device=torch.device('cpu'), steps=24)
    with xr.open_dataset(c['params']) as ds:
        ds = ds.load()
    ds['July_Tavg'] = (ds['elev'] * 0.0 + 11.0).astype('f8')
    p = str(tmp_path / 'params_july.nc')
    ds.to_netcdf(p)
    kw = dict(case='stehe', mode='WB', SNOW_BAND='TRUE', params=p)
    with torch.inference_mode():
        on = C.load(_global(tmp_path, extra=extra, name='on', **kw),
                    device=torch.device('cpu'), steps=48)
        assert on.treeline
        a = on.run()
        b = C.load(_global(tmp_path, name='off', **kw),
                   device=torch.device('cpu'), steps=48).run()
    assert torch.isfinite(a['et']).all() and torch.isfinite(a['swe']).all()
    assert (a['et'] - b['et']).abs().max() > 1e-6     # it changed
    # July_Tavg 100 C: no band above the treeline, the run is the base run
    ds['July_Tavg'] = ds['July_Tavg'] * 0 + 100.0
    p2 = str(tmp_path / 'params_hot.nc')
    ds.to_netcdf(p2)
    kw['params'] = p2
    with torch.inference_mode():
        hot = C.load(_global(tmp_path, extra=extra, name='hot', **kw),
                     device=torch.device('cpu'), steps=48).run()
    for k in ('et', 'runoff', 'swe'):
        assert torch.equal(hot[k], b[k]), k
