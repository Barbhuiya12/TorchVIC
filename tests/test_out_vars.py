"""The VIC output variables TorchVIC gained from out-vars.

VIC 5.1.0 writes 88 variables in a water-balance run on the ISC domain and
TorchVIC wrote 53; the 35 it left out are formed here (torchvic.model.outvars
and the rows `basin.simulate` collects for them), with VIC's arithmetic, from
put_data.c and the routines that fill its structures.

(a) Unit tests: the C routines written out literally -- find_0_degree_fronts
    (soil_conduction.c:722-766), calc_energy_balance_error (put_data.c:1086-
    1099), the band placement of the snow-band contraction -- against the
    tensor versions.
(b) The pristine VIC image drivers (5.1.0 and 5.0.1, reference/): the
    verification-matrix rows run through both models with every OUT_*
    variable at every step in double precision, and each new variable is
    compared with VIC's own: |TorchVIC - VIC| <= 1e-12 (absolute, or relative
    to the variable's VIC maximum where that is above 1). Rows: the
    snow-covered, overstory himal window (energy balance, 5.0.1 and 5.1.0;
    SNOW_BAND TRUE; frozen soil; water balance with frozen soil) and
    Stehekin (hourly snow).
(c) That no variable that existed changed: the default OUT_ENERGY_ERROR is
    still the solver's residual, and the VIC form is the option.
"""
import glob
import math
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import _paths                                                # noqa: E402
from torchvic.core import snowbands                          # noqa: E402
from torchvic.io import history as H                         # noqa: E402
from torchvic.model import outvars as O                      # noqa: E402

sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))

D = torch.float64
MISSING = -99999.0


# ---------------------------------------------------------------------------
# (a) the C routines, literally
# ---------------------------------------------------------------------------
def _linear_interp(x, lx, ux, ly, uy):
    # vic_run/src/interpoloation.c:19
    return (x - lx) / (ux - lx) * (uy - ly) + ly


def _fronts_c(T, Z, max_fronts=3):
    # soil_conduction.c 5.1.0:722-766
    n = len(T)
    nthaw = nfrost = 0
    fd = [MISSING] * max_fronts
    td = [MISSING] * max_fronts
    for k in range(n - 2, -1, -1):
        if T[k] > 0 and T[k + 1] <= 0 and nthaw < max_fronts:
            td[nthaw] = _linear_interp(0, T[k], T[k + 1], Z[k], Z[k + 1])
            nthaw += 1
        elif T[k] < 0 and T[k + 1] >= 0 and nfrost < max_fronts:
            fd[nfrost] = _linear_interp(0, T[k], T[k + 1], Z[k], Z[k + 1])
            nfrost += 1
    return fd, td


@pytest.mark.parametrize('nodes', [3, 6, 12])
def test_zero_degree_fronts_are_the_c_loop(nodes):
    g = torch.Generator().manual_seed(7 + nodes)
    T = (torch.randn(200, nodes, generator=g, dtype=D) * 6.0)
    # a few exact zeros, which the C's >, <=, <, >= treat differently
    T[::13, 1] = 0.0
    T[5::17, 0] = 0.0
    Z = torch.cumsum(torch.rand(200, nodes, generator=g, dtype=D) + 0.1, 1)
    fd, td = O.find_0_degree_fronts(T, Z)
    for i in range(T.shape[0]):
        f_c, t_c = _fronts_c(T[i].tolist(), Z[i].tolist())
        assert fd[i].tolist() == f_c, (i, T[i].tolist())
        assert td[i].tolist() == t_c, (i, T[i].tolist())


def test_front_rows_are_put_data_788_798():
    fd = torch.tensor([[[1.5, MISSING, MISSING], [0.25, 0.5, MISSING]]],
                      dtype=D)
    area = torch.tensor([[0.3, 0.7]], dtype=D)
    got = O.front_rows(fd, area)
    # energy.fdepth * AreaFactor * CM_PER_M where it is not MISSING
    want = torch.tensor([[(1.5 * 0.3) * 100.0 + (0.25 * 0.7) * 100.0,
                          (0.5 * 0.7) * 100.0, 0.0]], dtype=D)
    assert torch.equal(got.sum(1), want)


def test_energy_error_is_calc_energy_balance_error():
    g = torch.Generator().manual_seed(3)
    rows = {k: torch.randn(5, 4, generator=g, dtype=D) * 50.0
            for k in O.ENERGY_ERROR_ROWS + ('eb_latent_sub',
                                            'eb_advected_sensible')}
    got = O.energy_error(rows)
    for t in range(5):
        for c in range(4):
            r = {k: float(v[t, c]) for k, v in rows.items()}
            # put_data.c:539-551's arguments, calc_energy_balance_error's body
            net_rad = r['eb_net_short'] + r['eb_net_long']
            latent = r['eb_latent'] + r['eb_latent_sub']
            sensible = r['eb_sensible'] + r['eb_advected_sensible']
            grnd = r['eb_grnd_flux'] + r['eb_delta_h'] + r['eb_fusion']
            snow = (r['eb_advection'] - r['eb_deltacc'] + r['eb_snow_flux']
                    + r['eb_rfrz_energy'])
            want = net_rad - latent - sensible - grnd + snow
            assert float(got[t, c]) == want


def test_bands_are_placed_not_weighted():
    ex = dict(cell=np.array([0, 0, 1]), band=np.array([0, 2, 1]),
              weight=np.array([0.5, 0.5, 1.0]), n_cell=2, n_band=3)
    v = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=D)
    got = snowbands.to_bands(v, ex)
    assert got.shape == (2, 2, 3)
    assert got[0].tolist() == [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]
    assert got[1].tolist() == [[4.0, 0.0, 5.0], [0.0, 6.0, 0.0]]


def test_derive_forms_put_data_derived_variables():
    nt, nc = 4, 3
    g = torch.Generator().manual_seed(11)
    liq = torch.rand(nt, nc, 3, generator=g, dtype=D) * 100.0
    ice = torch.rand(nt, nc, 3, generator=g, dtype=D) * 30.0
    liq[0, 0, 0] = 0.0
    ice[0, 0, 0] = 0.0
    depth = torch.tensor([[0.1, 0.3, 0.6]] * nc, dtype=D)
    atm = dict(eact_air=torch.rand(nt, nc, generator=g, dtype=D) * 2000.0,
               press=90000.0 + torch.rand(nt, nc, generator=g, dtype=D))
    out = dict(swe=torch.zeros(nt, nc, dtype=D), soil_liq=liq, soil_ice=ice,
               sub_snow=torch.rand(nt, nc, generator=g, dtype=D))
    O.derive(out, atm, depth=depth)
    from torchvic.core import snow
    assert torch.equal(out['qair'],
                       (snow.CONST_EPS * atm['eact_air']) / atm['press'])
    # a layer holding nothing is 0/0, as in VIC
    assert math.isnan(float(out['smliqfrac'][0, 0, 0]))
    assert math.isnan(float(out['smfrozfrac'][0, 0, 0]))
    ok = torch.isfinite(out['smliqfrac'])
    assert torch.equal(out['smliqfrac'][ok], (liq / (liq + ice))[ok])
    assert torch.equal(out['smfrozfrac'][ok], (1.0 - liq / (liq + ice))[ok])
    assert torch.equal(out['soil_liq_frac'], liq / (depth * 1000.0))
    assert torch.equal(out['soil_ice_frac'], ice / (depth * 1000.0))
    # no CARBON: the forcing echoes are MISSING and the pools zero
    assert float(out['catm'].min()) == MISSING == float(out['par'].max())
    assert float(out['gpp'].abs().max()) == 0.0
    # no blowing: surface flux is the negative of the reported sublimation
    assert torch.equal(out['sub_surface'], -out['sub_snow'])
    assert float(out['sub_blowing'].abs().max()) == 0.0
    # a water-balance run has no energy error: VIC's MISSING
    assert float(out['energy_error_missing'].max()) == MISSING


def test_every_new_name_is_a_vic_output_with_a_source():
    meta = H.outvar_metadata()
    for name, src in H.OUTVAR_SOURCES_OUTVARS.items():
        assert name in meta, name
        assert H.OUTVAR_SOURCES[name] == src
    # the alternates of OUT_ENERGY_ERROR are read in order
    out = {'energy_error_missing': torch.ones(2, 2)}
    x, k = H.outvar_series(out, 'OUT_ENERGY_ERROR')
    assert k == 'energy_error_missing'
    out['eb_energy_error'] = torch.zeros(2, 2)
    x, k = H.outvar_series(out, 'OUT_ENERGY_ERROR')
    assert k == 'eb_energy_error'
    assert H.key_agg('eb_energy_error')[1] == 'OUT_ENERGY_ERROR'
    assert H.key_agg('energy_error_missing')[1] == 'OUT_ENERGY_ERROR'


# ---------------------------------------------------------------------------
# (b) pristine VIC, matrix rows
# ---------------------------------------------------------------------------

M = pytest.importorskip('matrix')
R = pytest.importorskip('run_matrix')
C = pytest.importorskip('compare')


def _row_run(tmp_path_factory, rid, days, extra=()):
    pytest.importorskip('dask', reason='the comparison opens the history files with xarray.open_mfdataset')
    r = [x for x in M.MATRIX if x['id'] == rid][0]
    win = os.path.join(M.WINDOWS, r['case'], 'params.nc')
    if not (os.path.exists(M.VIC[r['core']])
            and (r['case'] in ('stehe', 'wg', 'findley')
                 or os.path.exists(win))):
        pytest.skip('pristine VIC %s or the %s window is not present'
                    % (r['core'], r['case']))
    os.environ['PYTHONPATH'] = _paths.REPO + os.pathsep + os.environ.get(
        'PYTHONPATH', '')
    root = str(tmp_path_factory.mktemp('outvars'))
    d = R.prepare(r, days, root=root)
    vic_rc, _ = R.run_vic(r, d)
    tv_rc, _ = R.run_tv(r, d, 'cpu', 4, days, extra=extra)
    assert vic_rc == 0 and tv_rc == 0, open(
        os.path.join(d, 'tv.log')).read()[-3000:]
    return d


def _open(dd):
    import xarray as xr
    return xr.open_mfdataset(sorted(glob.glob(os.path.join(dd, '*.nc'))),
                             combine='by_coords')


def _compare(d, names, tol=1e-12):
    """{variable: max |TorchVIC - VIC|} for `names`; every one must be
    written by TorchVIC, be within `tol` (scaled up where VIC's maximum is
    above 1) and carry VIC's NaN pattern."""
    vic, tv = _open(os.path.join(d, 'vic')), _open(os.path.join(d, 'tv'))
    mask = np.isfinite(vic['OUT_PREC'].values[0])
    worst = {}
    for n in names:
        assert n in tv.data_vars, '%s was not written by TorchVIC' % n
        a = vic[n].values[..., mask]
        b = tv[n].values[..., mask]
        assert a.shape == b.shape, (n, a.shape, b.shape)
        assert (np.isnan(a) == np.isnan(b)).all(), n
        with np.errstate(all='ignore'):
            dif = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        scale = max(1.0, float(np.nanmax(np.abs(a)))) if a.size else 1.0
        assert dif <= tol * scale, (n, dif, scale)
        worst[n] = dif
    vic.close()
    tv.close()
    return worst


def _nonzero(d, names):
    vic = _open(os.path.join(d, 'vic'))
    out = [n for n in names if float(np.nanmax(np.abs(vic[n].values))) > 0.0]
    vic.close()
    return out


#: the variables out-vars formed that every energy-balance row has
EB_COMMON = ('OUT_QAIR', 'OUT_SMLIQFRAC', 'OUT_SMFROZFRAC',
             'OUT_SOIL_LIQ_FRAC', 'OUT_SOIL_ICE_FRAC', 'OUT_ZWT',
             'OUT_ZWT_LUMPED', 'OUT_AERO_COND1', 'OUT_AERO_COND2',
             'OUT_AERO_RESIST1', 'OUT_AERO_RESIST2', 'OUT_SOIL_TNODE',
             'OUT_SOIL_TNODE_WL', 'OUT_SUB_SURFACE', 'OUT_SUB_BLOWING',
             'OUT_SURFSTOR', 'OUT_DELSURFSTOR', 'OUT_ADV_SENS', 'OUT_CATM',
             'OUT_FDIR', 'OUT_PAR', 'OUT_SURF_FROST_FRAC', 'OUT_DISCHARGE',
             'OUT_SURFT_FBFLAG', 'OUT_SNOWT_FBFLAG', 'OUT_TFOL_FBFLAG',
             'OUT_TCAN_FBFLAG', 'OUT_SOILT_FBFLAG', 'OUT_GPP', 'OUT_NPP',
             'OUT_RAUT', 'OUT_RHET', 'OUT_NEE', 'OUT_CLITTER', 'OUT_CINTER',
             'OUT_CSLOW', 'OUT_LITTERFALL', 'OUT_APAR')
BANDS = ('OUT_SWE_BAND', 'OUT_SNOW_DEPTH_BAND', 'OUT_SNOW_CANOPY_BAND',
         'OUT_SNOW_MELT_BAND', 'OUT_SNOW_COVER_BAND', 'OUT_DELTACC_BAND',
         'OUT_ADVECTION_BAND', 'OUT_SNOW_FLUX_BAND', 'OUT_RFRZ_ENERGY_BAND',
         'OUT_MELT_ENERGY_BAND', 'OUT_SNOW_SURFT_BAND', 'OUT_SNOW_PACKT_BAND',
         'OUT_LATENT_SUB_BAND', 'OUT_SWNET_BAND', 'OUT_LWNET_BAND',
         'OUT_ALBEDO_BAND', 'OUT_LATENT_BAND', 'OUT_SENSIBLE_BAND',
         'OUT_GRND_FLUX_BAND', 'OUT_IN_LONG_BAND')


@pytest.fixture(scope='module')
def himal_eb(tmp_path_factory):
    return _row_run(tmp_path_factory, 'himal_EB_510_3h', 3,
                    extra=('--energy-error', 'vic'))


@pytest.mark.reference
@pytest.mark.slow
def test_himal_energy_balance_new_variables_are_vics(himal_eb):
    worst = _compare(himal_eb, EB_COMMON)
    # the nonzero ones really were exercised
    for n in ('OUT_QAIR', 'OUT_ZWT', 'OUT_AERO_COND1', 'OUT_AERO_COND2',
              'OUT_SOIL_TNODE', 'OUT_SUB_SURFACE', 'OUT_SOIL_LIQ_FRAC'):
        assert n in _nonzero(himal_eb, [n])
    assert max(worst.values()) <= 1e-12 * 2000.0


@pytest.mark.reference
@pytest.mark.slow
def test_himal_energy_error_is_vics_with_the_option(himal_eb):
    # VIC reports up to 9.9 W/m2 on these three days (put_data.c:538-552)
    w = _compare(himal_eb, ['OUT_ENERGY_ERROR'])
    assert w['OUT_ENERGY_ERROR'] <= 1e-12 * 10.0
    assert 'OUT_ENERGY_ERROR' in _nonzero(himal_eb, ['OUT_ENERGY_ERROR'])


@pytest.mark.reference
@pytest.mark.slow
def test_himal_energy_error_default_stays_the_solvers(tmp_path_factory):
    d = _row_run(tmp_path_factory, 'himal_EB_510_3h', 3)
    vic, tv = _open(os.path.join(d, 'vic')), _open(os.path.join(d, 'tv'))
    mask = np.isfinite(vic['OUT_PREC'].values[0])
    a = vic['OUT_ENERGY_ERROR'].values[..., mask]
    b = tv['OUT_ENERGY_ERROR'].values[..., mask]
    # the solver's residual closes to ~1e-6; VIC's diagnostic does not
    assert float(np.abs(b).max()) < 1e-3
    assert float(np.abs(a).max()) > 1.0


@pytest.mark.reference
@pytest.mark.slow
def test_himal_energy_balance_501(tmp_path_factory):
    d = _row_run(tmp_path_factory, 'himal_EB_501_3h', 3,
                 extra=('--energy-error', 'vic'))
    _compare(d, [n for n in EB_COMMON if n != 'OUT_DISCHARGE']
             + ['OUT_ENERGY_ERROR'])


@pytest.mark.reference
@pytest.mark.slow
def test_himal_snow_bands(tmp_path_factory):
    d = _row_run(tmp_path_factory, 'himal_EB_510_3h_SNOW_BAND=TRUE', 10)
    _compare(d, BANDS)
    assert len(_nonzero(d, BANDS)) >= 15


@pytest.mark.reference
@pytest.mark.slow
def test_himal_single_band_axis(himal_eb):
    # SNOW_BAND FALSE: one band, [step, 1, cell] in the file
    _compare(himal_eb, BANDS)
    tv = _open(os.path.join(himal_eb, 'tv'))
    assert tv['OUT_SWE_BAND'].dims[1] == 'snow_band'
    assert tv['OUT_SWE_BAND'].shape[1] == 1
    tv.close()


@pytest.mark.reference
@pytest.mark.slow
def test_himal_frozen_soil(tmp_path_factory):
    d = _row_run(tmp_path_factory, 'himal_FS_510_3h', 10,
                 extra=('--energy-error', 'vic'))
    names = ('OUT_FDEPTH', 'OUT_TDEPTH', 'OUT_SURF_FROST_FRAC',
             'OUT_SMFROZFRAC', 'OUT_SOIL_ICE_FRAC', 'OUT_SOIL_TNODE',
             'OUT_SUB_SURFACE', 'OUT_ENERGY_ERROR')
    _compare(d, names)
    assert len(_nonzero(d, names)) == len(names)


@pytest.mark.reference
@pytest.mark.slow
def test_himal_frozen_soil_water_balance(tmp_path_factory):
    d = _row_run(tmp_path_factory, 'himal_FSWB_510_3h', 10)
    names = ('OUT_FDEPTH', 'OUT_TDEPTH', 'OUT_SURF_FROST_FRAC',
             'OUT_SMFROZFRAC', 'OUT_SOIL_ICE_FRAC', 'OUT_SUB_SURFACE',
             'OUT_ZWT', 'OUT_ZWT_LUMPED', 'OUT_ENERGY_ERROR', 'OUT_QAIR')
    _compare(d, names)


@pytest.mark.reference
@pytest.mark.slow
def test_stehekin_snow(tmp_path_factory):
    d = _row_run(tmp_path_factory, 'stehe_EB_510_1h', 5,
                 extra=('--energy-error', 'vic'))
    _compare(d, [n for n in EB_COMMON] + ['OUT_ENERGY_ERROR'] + list(BANDS))


@pytest.mark.reference
@pytest.mark.slow
def test_himal_blowing_surface_flux(tmp_path_factory):
    # BLOWING TRUE: OUT_SUB_SURFACE = -OUT_SUB_SNOW - OUT_SUB_BLOWING
    d = _row_run(tmp_path_factory, 'himal_EB_510_3h_BLOWING=TRUE', 10)
    _compare(d, ['OUT_SUB_SURFACE', 'OUT_SUB_BLOWING', 'OUT_SUB_SNOW'])


@pytest.mark.reference
@pytest.mark.slow
def test_himal_water_balance_energy_rows(tmp_path_factory):
    # FULL_ENERGY FALSE: VIC's Tsurf = Tair evaluation (wbenergy.py)
    d = _row_run(tmp_path_factory, 'himal_WB_510_3h', 10)
    names = ('OUT_SWNET', 'OUT_LWNET', 'OUT_R_NET', 'OUT_LATENT',
             'OUT_SENSIBLE', 'OUT_SURF_TEMP', 'OUT_RAD_TEMP', 'OUT_VEGT',
             'OUT_SNOW_FLUX', 'OUT_PET', 'OUT_SOIL_TEMP', 'OUT_SOIL_TNODE',
             'OUT_AERO_COND', 'OUT_AERO_COND1', 'OUT_AERO_COND2',
             'OUT_AERO_RESIST', 'OUT_AERO_RESIST1', 'OUT_AERO_RESIST2',
             'OUT_SWNET_BAND', 'OUT_LWNET_BAND', 'OUT_LATENT_BAND',
             'OUT_SENSIBLE_BAND', 'OUT_SNOW_FLUX_BAND', 'OUT_ENERGY_ERROR',
             'OUT_SUB_SURFACE', 'OUT_ZWT')
    _compare(d, names, tol=1e-11)
