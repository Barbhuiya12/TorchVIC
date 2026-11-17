"""The verification matrix (verification/) is well formed, and every option
TorchVIC refuses is refused by name, not silently ignored.

The matrix itself runs VIC and TorchVIC for hours (verification/run_all.sh);
this checks only what can go wrong before it starts: every row's global
file is read by TorchVIC's own parser with no unrecognised key, every
(option, value) torchvic.io.options refuses stops image_options with a
message naming it, the refusals that need the parameter or forcing file
stop case.load, the model step comes from the global file, and
docs/OPTIONS.md is the table the code holds.
"""
import os
import sys

import pytest

import _paths                                           # noqa: E402

sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
matrix = pytest.importorskip('matrix')

from torchvic import case as C                          # noqa: E402
from torchvic.io import options as O                    # noqa: E402
from torchvic.io.global_param import GlobalParam        # noqa: E402


def test_row_ids_are_unique_and_rows_name_known_cases_and_modes():
    ids = [r['id'] for r in matrix.MATRIX]
    assert len(ids) == len(set(ids))
    for r in matrix.MATRIX:
        assert r['case'] in matrix.CASES and r['mode'] in matrix.MODES
        assert r['core'] in matrix.VIC
        assert r['expect'] in ('supported', 'gap')


def test_every_row_writes_a_global_file_torchvic_reads(tmp_path):
    ov = matrix.outvars()
    for r in matrix.MATRIX:
        g = tmp_path / ('%s.txt' % abs(hash(r['id'])))
        g.write_text(matrix.global_text(r, str(tmp_path), str(tmp_path), ov))
        gp = GlobalParam(str(g))
        unknown = [u for u in getattr(gp, 'unknown', [])]
        assert not unknown, (r['id'], unknown)
        c = matrix.CASES[r['case']]
        spd = 1 if r['daily'] else c['forcing_spd']
        assert gp.get_int('MODEL_STEPS_PER_DAY') == spd
        assert gp.get_int('SNOW_STEPS_PER_DAY') == c['forcing_spd']


def _global(tmp_path, extra=(), case='wg', mode='EB', edit=None, **kw):
    r = matrix.row(case, mode, **kw)
    txt = matrix.global_text(r, str(tmp_path), str(tmp_path), ['OUT_PREC'],
                             extra_lines=list(extra))
    for a, b in (edit or {}).items():
        assert a in txt, a
        txt = txt.replace(a, b)
    g = tmp_path / 'g.txt'
    g.write_text(txt)
    return str(g)


def test_the_base_rows_are_not_refused(tmp_path):
    for mode in ('WB', 'EB', 'FS', 'FSWB'):
        C.image_options(GlobalParam(_global(tmp_path, mode=mode)))


@pytest.mark.parametrize('key,value', O.refused_values())
def test_refused_options_are_refused_by_name(tmp_path, key, value):
    g = _global(tmp_path, extra=['%s %s' % (key, value)])
    with pytest.raises(ValueError, match=key):
        C.image_options(GlobalParam(g))


def test_an_unknown_value_is_refused_by_name(tmp_path):
    g = _global(tmp_path, extra=['SNOW_DENSITY DENS_FOO'])
    with pytest.raises(ValueError, match='SNOW_DENSITY DENS_FOO'):
        C.image_options(GlobalParam(g))


def test_no_effect_options_are_announced(tmp_path):
    said = []
    g = _global(tmp_path, extra=['NOFLUX TRUE', 'ATMOS_STEPS_PER_DAY 24',
                                 'STATE_FORMAT NETCDF4', 'NOT_A_VIC_KEY 1'])
    C.image_options(GlobalParam(g), log=said.append)
    text = '\n'.join(said)
    for key in ('NOFLUX TRUE', 'ATMOS_STEPS_PER_DAY 24', 'STATE_FORMAT',
                'NOT_A_VIC_KEY'):
        assert key in text, key


def test_frozen_soil_off_the_quick_flux_path_runs_the_profile(tmp_path):
    """FROZEN_SOIL TRUE with QUICK_FLUX FALSE left as VIC leaves it (off)
    runs the finite-difference profile with freezing: FROZEN_SOIL TRUE
    turns QUICK_FLUX off (get_global_param.c:104-107), and that path is
    implemented. (It used to be refused by name, when TorchVIC's frozen
    soil was the QUICK_FLUX form only.) IMPLICIT FALSE: with NODES 3
    VIC's own default IMPLICIT TRUE reads past its single unknown
    (frozen_soil.c:793-796), which is refused below."""
    g = _global(tmp_path, edit={'FROZEN_SOIL FALSE': 'FROZEN_SOIL TRUE'},
                extra=['IMPLICIT FALSE'])
    opts = C.image_options(GlobalParam(g))
    assert opts['quick_flux'] is False
    assert opts['frozen_soil'] is True
    assert opts['implicit'] is False


def test_implicit_on_three_nodes_is_refused_as_uninitialized(tmp_path):
    """IMPLICIT TRUE with NOFLUX FALSE on a 3-node column is refused by
    name: VIC reads T_2[1] past its single unknown there
    (frozen_soil.c:793-796) -- uninitialized stack, not physics -- so
    there is nothing reproducible to run."""
    g = _global(tmp_path, edit={'FROZEN_SOIL FALSE': 'FROZEN_SOIL TRUE'})
    with pytest.raises(ValueError, match='uninitialized memory'):
        C.image_options(GlobalParam(g))


def test_spatial_frost_counts_are_parsed_and_bounded(tmp_path):
    g = _global(tmp_path, mode='FS', extra=('SPATIAL_FROST TRUE 3',))
    assert O.spatial_frost_n(GlobalParam(g)) == 3
    C.image_options(GlobalParam(g))
    g = _global(tmp_path, mode='FS',
                extra=('SPATIAL_FROST TRUE 3', 'SPATIAL_FROST FALSE'))
    assert O.spatial_frost_n(GlobalParam(g)) == 1
    for bad in ('SPATIAL_FROST TRUE 11', 'SPATIAL_FROST TRUE 0',
                'SPATIAL_FROST TRUE', 'SPATIAL_FROST TRUE x'):
        g = _global(tmp_path, mode='FS', extra=(bad,))
        with pytest.raises(ValueError, match='SPATIAL_FROST TRUE'):
            C.image_options(GlobalParam(g))


def test_spatial_frost_without_frozen_soil_is_announced(tmp_path):
    g = _global(tmp_path, mode='WB', extra=('SPATIAL_FROST TRUE 3',))
    refused, announced = O.check(GlobalParam(g))
    assert not refused
    assert any(s.startswith('SPATIAL_FROST TRUE: read, no effect')
               for s in announced)


def test_frozen_soil_without_the_energy_balance_is_accepted(tmp_path):
    # FROZEN_SOIL TRUE with FULL_ENERGY FALSE runs frozen soil in
    # water-balance mode, as VIC does (calc_surf_energy_bal.c:522-525);
    # --no-energy on a frozen-soil row runs it the same way.
    C.image_options(GlobalParam(_global(tmp_path, mode='FSWB')))
    C.image_options(GlobalParam(_global(tmp_path, mode='FS')),
                    no_energy=True)


def test_quick_flux_off_without_energy_or_frozen_is_refused(tmp_path):
    """QUICK_FLUX FALSE in the water balance without frozen soil is
    refused as VIC refuses it (get_global_param.c:939-945)."""
    g = _global(tmp_path, mode='WB', extra=['QUICK_FLUX FALSE'])
    with pytest.raises(ValueError, match='QUICK_FLUX FALSE'):
        C.image_options(GlobalParam(g))


def test_quick_solve_with_exp_trans_is_refused_as_vic_refuses(tmp_path):
    """QUICK_SOLVE TRUE with QUICK_FLUX FALSE needs NOFLUX FALSE and
    EXP_TRANS FALSE (get_global_param.c:943-949)."""
    g = _global(tmp_path, mode='EB',
                extra=['QUICK_FLUX FALSE', 'QUICK_SOLVE TRUE',
                       'EXP_TRANS TRUE', 'NODES 12'])
    with pytest.raises(ValueError, match='QUICK_SOLVE TRUE'):
        C.image_options(GlobalParam(g))


def test_nodes_outside_3_to_50_are_refused_with_quick_flux_off(tmp_path):
    g = _global(tmp_path, mode='EB', extra=['QUICK_FLUX FALSE', 'NODES 60'])
    with pytest.raises(ValueError, match='MAX_NODES'):
        C.image_options(GlobalParam(g))
    g = _global(tmp_path, mode='EB', extra=['QUICK_FLUX FALSE', 'NODES 2'])
    with pytest.raises(ValueError, match='fewer than 3 nodes'):
        C.image_options(GlobalParam(g))


def test_no_flux_alias_is_honoured(tmp_path):
    """NO_FLUX TRUE sets NOFLUX (get_global_param.c:125-130), and a
    FALSE line of either spelling leaves a TRUE standing."""
    g = _global(tmp_path, mode='EB',
                extra=['QUICK_FLUX FALSE', 'NO_FLUX TRUE'])
    assert C.image_options(GlobalParam(g))['noflux'] is True
    g = _global(tmp_path, mode='EB',
                extra=['QUICK_FLUX FALSE', 'NOFLUX TRUE', 'NOFLUX FALSE'])
    assert C.image_options(GlobalParam(g))['noflux'] is True


def test_thermal_switches_are_quiet_where_they_select(tmp_path):
    """With QUICK_FLUX FALSE the thermal switches select the profile's
    solver, grid and boundaries, so their lines are read in silence;
    with QUICK_FLUX TRUE they change nothing and are announced."""
    g = _global(tmp_path, mode='EB', extra=['QUICK_FLUX FALSE'])
    _, announced = O.check(GlobalParam(g))
    assert not any('NODES' in a for a in announced), announced
    g = _global(tmp_path, mode='EB')
    _, announced = O.check(GlobalParam(g))
    assert any(a.startswith('NODES 3') for a in announced), announced


def test_quick_flux_off_in_the_frozen_water_balance_is_refused(tmp_path):
    """FROZEN_SOIL with QUICK_FLUX FALSE runs the profile in the water
    balance in VIC; TorchVIC's water-balance frozen soil is the QUICK_FLUX
    path only, and the profile's per-frost-area ice is not implemented."""
    g = _global(tmp_path, mode='FSWB', extra=['QUICK_FLUX FALSE'])
    with pytest.raises(ValueError, match='QUICK_FLUX FALSE in the water'):
        C.image_options(GlobalParam(g))
    g = _global(tmp_path, mode='FS', extra=['QUICK_FLUX FALSE'])
    with pytest.raises(ValueError, match='--no-energy'):
        C.image_options(GlobalParam(g), no_energy=True)
    g = _global(tmp_path, mode='FS',
                extra=['QUICK_FLUX FALSE', 'SPATIAL_FROST TRUE 3'])
    with pytest.raises(ValueError, match='per-frost-area'):
        C.image_options(GlobalParam(g))
    g = _global(tmp_path, mode='EB',
                extra=['QUICK_FLUX FALSE', 'TFALLBACK FALSE'])
    with pytest.raises(ValueError, match='TFALLBACK FALSE with QUICK_FLUX'):
        C.image_options(GlobalParam(g))


def test_implicit_left_at_vics_default_is_refused_as_vic_refuses(tmp_path):
    # No FROZEN_SOIL line: VIC's IMPLICIT and EXP_TRANS stay TRUE and VIC
    # stops with QUICK_FLUX TRUE (get_global_param.c:933-936).
    g = _global(tmp_path, edit={'FROZEN_SOIL FALSE\n': ''})
    with pytest.raises(ValueError, match='IMPLICIT TRUE with QUICK_FLUX'):
        C.image_options(GlobalParam(g))


def test_aero_resist_cansnow_reaches_simulate(tmp_path):
    g = _global(tmp_path, extra=['AERO_RESIST_CANSNOW AR_406'])
    assert C.image_options(GlobalParam(g))['aero_resist_cansnow'] == 'AR_406'
    g = _global(tmp_path, extra=['AERO_RESIST_CANSNOW AR_410'])
    assert C.image_options(GlobalParam(g))['aero_resist_cansnow'] == 'AR_410'
    g = _global(tmp_path, extra=['AERO_RESIST_CANSNOW AR_406_LS'])
    assert C.image_options(GlobalParam(g))['aero_resist_cansnow'] == 'AR_406_LS'
    assert C.image_options(GlobalParam(_global(tmp_path)))[
        'aero_resist_cansnow'] == 'AR_406_FULL'


def test_grnd_flux_type_reaches_simulate(tmp_path):
    g = _global(tmp_path, extra=['GRND_FLUX_TYPE GF_406'])
    assert C.image_options(GlobalParam(g))['grnd_flux_type'] == 'GF_406'
    g = _global(tmp_path, extra=['GRND_FLUX_TYPE GF_410'])
    assert C.image_options(GlobalParam(g))['grnd_flux_type'] == 'GF_410'
    assert C.image_options(GlobalParam(_global(tmp_path)))[
        'grnd_flux_type'] == 'GF_410'


def test_close_energy_reaches_simulate(tmp_path):
    g = _global(tmp_path, extra=['CLOSE_ENERGY TRUE'])
    assert C.image_options(GlobalParam(g))['close_energy'] is True
    assert C.image_options(GlobalParam(_global(tmp_path)))[
        'close_energy'] is False


def test_snow_density_reaches_simulate(tmp_path):
    g = _global(tmp_path, extra=['SNOW_DENSITY DENS_SNTHRM'])
    assert C.image_options(GlobalParam(g))['snow_density'] == 'DENS_SNTHRM'
    assert C.image_options(GlobalParam(_global(tmp_path)))[
        'snow_density'] == 'DENS_BRAS'


def test_docs_options_table_is_the_codes():
    path = os.path.join(_paths.REPO, 'docs', 'OPTIONS.md')
    assert open(path).read() == O.markdown(), (
        'docs/OPTIONS.md is stale: python -m torchvic.io.options > '
        'docs/OPTIONS.md')


def test_readme_options_summary_is_the_codes():
    text = open(os.path.join(_paths.REPO, 'README.md')).read()
    if O.README_BEGIN not in text:
        # the short README links to docs/OPTIONS.md, which the test above
        # checks against the code
        assert 'docs/OPTIONS.md' in text
        pytest.skip('README.md carries no options block')
    assert text == O.readme_block(text), (
        'README.md options block is stale: python -m torchvic.io.options '
        'README.md')


def test_every_image_driver_key_vic_has_is_classified():
    from torchvic.io import global_param as G
    if G._INVENTORY is None:
        pytest.skip('no option inventory')
    image = {k for k, e in G._INVENTORY['options'].items()
             if 'image' in e.get('drivers', [])}
    assert image == set(O.IMAGE_OPTIONS), sorted(image ^ set(O.IMAGE_OPTIONS))


# -- refusals that need the parameter or forcing file ----------------------
def _params(case):
    c = matrix.CASES[case]
    if not os.path.exists(c['params']):
        pytest.skip('%s is absent' % c['params'])
    from torchvic.io.domain import ImageDomain
    from torchvic.io.params import ImageParams
    dom = ImageDomain(c['domain'])
    return ImageParams(c['params'], dom)


def test_snow_bands_with_area_are_run(tmp_path):
    par = _params('stehe')                      # 3-5 bands with area a cell
    g = GlobalParam(_global(tmp_path, case='stehe',
                            edit={'SNOW_BAND FALSE': 'SNOW_BAND TRUE'}))
    assert C.check_inputs(g, par) == []
    g = GlobalParam(_global(tmp_path, case='stehe'))
    assert C.check_inputs(g, par) == []


def test_one_band_with_area_is_run(tmp_path):
    par = _params('wg')                         # one band with area a cell
    g = GlobalParam(_global(tmp_path, case='wg',
                            edit={'SNOW_BAND FALSE': 'SNOW_BAND TRUE'}))
    assert C.check_inputs(g, par) == []


def test_fcanopy_from_a_file_without_it_is_refused_as_vic_refuses(tmp_path):
    par = _params('wg')                         # no fcanopy in the file
    g = GlobalParam(_global(tmp_path, case='wg', FCAN_SRC='FROM_VEGPARAM'))
    assert any(s.startswith('FCAN_SRC FROM_VEGPARAM')
               for s in C.check_inputs(g, par))


def test_blowing_without_its_fields_is_refused_by_name(tmp_path):
    par = _params('wg')
    g = GlobalParam(_global(tmp_path, case='wg', BLOWING='TRUE'))
    assert any('sigma_slope' in s for s in C.check_inputs(g, par))


def test_spatial_frost_needs_its_slope_and_keeps_it_nonnegative(tmp_path):
    g = GlobalParam(_global(tmp_path, case='wg', mode='FSWB',
                            extra=('SPATIAL_FROST TRUE 3',)))
    par = _params('wg')                         # no frost_slope in the file
    assert any('frost_slope' in s for s in C.check_inputs(g, par))
    par.soil['frost_slope'] = [4.0] * len(par.soil['infilt'])
    assert not any('frost_slope' in s for s in C.check_inputs(g, par))
    par.soil['frost_slope'][0] = -1.0
    assert any('frost_slope' in s for s in C.check_inputs(g, par))


def test_blowing_fields_are_read_from_the_parameter_file(tmp_path):
    p = os.path.join(matrix.WORK, 'variants', 'stehe_blowing', 'params.nc')
    if not os.path.exists(p):
        pytest.skip('verification/make_variants.py has not been run')
    from torchvic.io.domain import ImageDomain
    from torchvic.io.params import ImageParams
    par = ImageParams(p, ImageDomain(matrix.CASES['stehe']['domain']))
    for k in ('sigma_slope', 'lag_one', 'fetch'):
        assert k in par.veg
    g = GlobalParam(_global(tmp_path, case='stehe', BLOWING='TRUE'))
    assert C.check_inputs(g, par) == []


@pytest.mark.parametrize('case,variant', (('himal', 'himal_frost'),
                                         ('stehe', 'stehe_frost')))
def test_frost_slope_is_read_from_the_parameter_file(tmp_path, case,
                                                     variant):
    p = os.path.join(matrix.WORK, 'variants', variant, 'params.nc')
    if not os.path.exists(p):
        pytest.skip('verification/make_variants.py has not been run')
    from torchvic.io.domain import ImageDomain
    from torchvic.io.params import ImageParams
    par = ImageParams(p, ImageDomain(matrix.CASES[case]['domain']))
    assert 'frost_slope' in par.soil
    g = GlobalParam(_global(tmp_path, case=case, mode='FSWB',
                            extra=('SPATIAL_FROST TRUE 3',)))
    assert C.check_inputs(g, par) == []


def test_spatial_snow_reaches_simulate(tmp_path):
    g = _global(tmp_path, extra=['SPATIAL_SNOW TRUE'])
    assert C.image_options(GlobalParam(g))['spatial_snow'] is True
    assert C.image_options(GlobalParam(_global(tmp_path)))[
        'spatial_snow'] is False


def test_spatial_snow_without_its_field_is_refused_by_name(tmp_path):
    par = _params('wg')                       # no max_snow_distrib_slope
    g = GlobalParam(_global(tmp_path, case='wg', SPATIAL_SNOW='TRUE'))
    assert any('max_snow_distrib_slope' in s for s in C.check_inputs(g, par))


def test_spatial_snow_field_is_read_from_the_parameter_file(tmp_path):
    p = os.path.join(matrix.WORK, 'variants', 'himal_spatial_snow',
                     'params.nc')
    if not os.path.exists(p):
        pytest.skip('verification/make_variants.py has not been run')
    from torchvic.io.domain import ImageDomain
    from torchvic.io.params import ImageParams
    par = ImageParams(p, ImageDomain(matrix.CASES['himal']['domain']))
    assert 'max_snow_distrib_slope' in par.soil
    g = GlobalParam(_global(tmp_path, case='himal', SPATIAL_SNOW='TRUE'))
    assert C.check_inputs(g, par) == []


def _wg_forcing():
    if not os.path.exists(matrix.CASES['wg']['params']):
        pytest.skip('Walnut Gulch sample data is absent')


def test_the_model_step_comes_from_the_global_file(tmp_path):
    # MODEL_STEPS_PER_DAY 1 and no `daily`: VIC's daily step, three June
    # days of a 3-hourly forcing (no snow).
    _wg_forcing()
    g = _global(tmp_path, mode='WB', daily=True,
                edit={'STARTMONTH 1': 'STARTMONTH 6',
                      'ENDMONTH 12': 'ENDMONTH 6', 'ENDDAY 31': 'ENDDAY 3'})
    c = C.load(g, device='cpu')
    assert c.model_steps == 1 and c.n_steps == 3
    assert [d.day for d in c.dates] == [1, 2, 3]
    assert c.subdaily is None
    # and the sub-daily step of the same file runs the window, 8 a day
    g = _global(tmp_path, mode='WB',
                edit={'STARTMONTH 1': 'STARTMONTH 6',
                      'ENDMONTH 12': 'ENDMONTH 6', 'ENDDAY 31': 'ENDDAY 3'})
    c = C.load(g, device='cpu')
    assert c.model_steps == 8 and c.n_steps == 24


def test_a_daily_step_that_can_snow_keeps_the_subdaily_records(tmp_path):
    # surface_fluxes.c:258-269. The year's forcing can snow, so the daily
    # step keeps the snow-step records beside the daily means.
    _wg_forcing()
    g = _global(tmp_path, mode='WB', daily=True)
    c = C.load(g, device='cpu')
    assert c.model_steps == 1 and c.snow_steps == 8
    assert c.subdaily['tair'].shape == (c.n_steps, 8, c.n_cells)
    assert c.subdaily['prcp'].shape == c.subdaily['tair'].shape


def test_an_aggregation_other_than_daily_is_refused(tmp_path):
    _wg_forcing()
    g = _global(tmp_path, mode='WB',
                edit={'MODEL_STEPS_PER_DAY 8': 'MODEL_STEPS_PER_DAY 4',
                      'SNOW_STEPS_PER_DAY 8': 'SNOW_STEPS_PER_DAY 4'})
    with pytest.raises(ValueError, match='MODEL_STEPS_PER_DAY 4'):
        C.load(g, device='cpu')


def test_a_window_the_forcing_does_not_cover_is_refused(tmp_path):
    _wg_forcing()
    g = _global(tmp_path, mode='WB', edit={'ENDYEAR 2008': 'ENDYEAR 2030'})
    with pytest.raises(ValueError, match='not covered by the forcing'):
        C.load(g, device='cpu')


def test_grade_denominator_floor_is_1e8_for_mm_and_1e12_otherwise():
    import numpy as np
    import compare as K
    assert K.abs_floor('OUT_SOIL_MOIST') == 1e-8
    assert K.abs_floor('OUT_EVAP') == 1e-8
    assert K.abs_floor('OUT_LATENT') == 1e-12
    assert K.abs_floor('OUT_SURF_TEMP') == 1e-12
    w = np.ones(1)
    a = np.full((4, 1), 1.42e-12)
    b = a + np.array([[0.0], [4e-13], [0.0], [0.0]])
    mm = K._metrics('OUT_SOIL_MOIST[2]', 'OUT_SOIL_MOIST', a, b, w, ['t'] * 4)
    assert mm['score'] < 1e-4 and mm['grade'] == 'pass'
    wm = K._metrics('OUT_LATENT', 'OUT_LATENT', a, b, w, ['t'] * 4)
    assert wm['grade'] == 'fail'               # W m-2 floor unchanged
