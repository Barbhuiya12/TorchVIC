"""Full energy (FULL_ENERGY TRUE) on snow-free ISC windows against the
pristine VIC 5.0.1 and 5.1.0 image drivers, both cores.

The verification windows (verification/make_isc_windows.py: 6 x 6 cells,
3-hourly, ISC fcanopy, LAI and albedo from the veglib) run through both
models with verification/matrix.py's global files. Each case pins a fix:

* forest, 2 days, 5.0.1: the fractional canopy on the energy path
  (vic_run.c 5.0.1:132-141; func_surf_energy_bal.c 5.0.1:703-805): LAI
  and Wdew per plant, the plants' fluxes times fcanopy and (1 - fcanopy)
  ARNO bare soil at Ra_used[0]. Was EVAP_BARE 0 against VIC's 0.066 mm.
* forest, 2 days, both cores: OUT_RAD_TEMP is Tfoliage only on an
  overstory tile *with* snow (put_data.c 5.1.0:820-825), and an overstory
  tile reports aero_resist[1] when no canopy ran (vic_run.c 5.1.0:249-250).
* igp, to 13 March, both cores: canopy_evap.c 5.1.0:478-508's wilting
  clamp on the energy path, and root_brent.c's own iteration for the
  surface temperature. Was TRANSP_VEG 0.0251 mm against VIC's 0.0106 at
  2004-03-12 12:00, cell 6.
"""
import os
import sys

import numpy as np
import pytest

import _paths                                           # noqa: E402

sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
M = pytest.importorskip('matrix')
R = pytest.importorskip('run_matrix')
C = pytest.importorskip('compare')

WIN = os.path.join(M.WINDOWS, '%s', 'params.nc')


def _run(tmp, rid, days):
    r = [x for x in M.MATRIX if x['id'] == rid][0]
    if not (os.path.exists(M.VIC[r['core']])
            and os.path.exists(WIN % r['case'])):
        pytest.skip('pristine VIC %s or the %s window is not present'
                    % (r['core'], r['case']))
    os.environ['PYTHONPATH'] = _paths.REPO + os.pathsep + os.environ.get(
        'PYTHONPATH', '')
    d = R.prepare(r, days, root=str(tmp))
    st = {}
    st['vic_rc'], _ = R.run_vic(r, d)
    st['tv_rc'], _ = R.run_tv(r, d, 'cpu', 4, days)
    assert st['vic_rc'] == 0 and st['tv_rc'] == 0, open(
        os.path.join(d, 'tv.log')).read()[-3000:]
    res = C.compare_dirs(os.path.join(d, 'vic'), os.path.join(d, 'tv'), r, st)
    return {v['var']: v for v in res['vars'] if 'score' in v}, res


def _check(v, names, tol):
    bad = {n: (v[n]['score'], v[n]['first']) for n in names
           if v[n]['score'] > tol}
    assert not bad, bad


@pytest.mark.parametrize('core', ['501', '510'])
def test_forest_snow_free_energy_balance_is_vics(tmp_path, core):
    v, res = _run(tmp_path, 'forest_EB_%s_3h' % core, 2)
    assert res['grade'] == 'pass', res['grade']
    _check(v, ('OUT_EVAP_BARE', 'OUT_TRANSP_VEG', 'OUT_EVAP_CANOP',
               'OUT_LATENT', 'OUT_SENSIBLE', 'OUT_RAD_TEMP', 'OUT_SURF_TEMP',
               'OUT_AERO_COND', 'OUT_AERO_RESIST', 'OUT_PET'), 1e-8)
    assert v['OUT_EVAP_BARE']['vic_total'] > 0.1


@pytest.mark.parametrize('core', ['501', '510'])
def test_igp_wilting_clamp_on_the_energy_path(tmp_path, core):
    v, res = _run(tmp_path, 'igp_EB_%s_3h' % core, 73)
    assert res['grade'] == 'pass', res['grade']
    _check(v, ('OUT_TRANSP_VEG', 'OUT_EVAP_BARE', 'OUT_EVAP', 'OUT_LATENT',
               'OUT_SURF_TEMP'), 1e-8)
    for n in ('OUT_TRANSP_VEG', 'OUT_SURF_TEMP'):
        assert v[n]['first'] is None, (n, v[n]['first'])
