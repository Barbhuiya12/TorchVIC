"""The carbon cycle against VIC 5.0.1's own compiled functions.

`torchvic.core.carbon` is compared with faparl(), calc_Nscale_factors(),
canopy_assimilation() and compute_soil_resp() as upstream ships them,
called through tests/oracle/ref_snow.c (modes `carbon` and `soilresp`)
and linked against the pristine tree -- not the copy in this repository,
for the reason that file's header gives.

Nothing here iterates, so unlike the snow and blowing-snow comparisons
there is no solver tolerance to allow for: every number is a closed-form
expression. The floor is the oracle's own printing, %.12g, which is twelve
significant digits -- so a comparison of a single printed number cannot be
asked for better than about 1e-12 relative, and a ratio of two of them not
better than 1e-11. The thresholds below are set at 1e-10, an order clear of
that, and the measured agreement is printed so a regression shows the
number rather than only a failure.
"""
import os
import subprocess

import pytest
import torch

from torchvic.core import carbon

torch.set_default_dtype(torch.float64)
KW = dict(dtype=torch.float64)

import _paths                                           # noqa: E402
REF = _paths.oracle('ref_snow')

IN_COLS = ('LAI', 'CosZen', 'AlbSoilPAR', 'Fdir', 'Tfoliage', 'SWdown',
           'elevation', 'Catm', 'Ctype', 'MaxCarboxRate', 'MaxETransport',
           'CO2Specificity')
OUT_COLS = ('faPAR0', 'faPAR1', 'faPAR2', 'Nscale0', 'Nscale1', 'Nscale2',
            'rc', 'Ci', 'GPP', 'Rdark', 'Rphoto', 'Rmaint', 'Rgrowth',
            'Raut', 'NPP', 'rs0', 'rs1', 'rs2')
SR_IN = ('dt', 'T0', 'T1', 'T2', 'w0', 'w1', 'w2', 'CLitter', 'CInter',
         'CSlow')
SR_OUT = ('RhLitter', 'RhInter', 'RhSlow')


def _oracle(mode):
    if not os.path.exists(REF):
        pytest.skip('ref_snow not built; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([REF, mode], capture_output=True, text=True,
                         check=True).stdout
    rows = []
    for line in out.splitlines():
        if line.startswith('#') or '|' not in line:
            continue
        lhs, rhs = line.split('|')
        rows.append(([float(x) for x in lhs.split(',')],
                     [float(x) for x in rhs.split(',')]))
    assert rows, 'oracle produced no rows for mode %s' % mode
    return rows


@pytest.fixture(scope='module')
def cases():
    return _oracle('carbon')


@pytest.fixture(scope='module')
def sr_cases():
    return _oracle('soilresp')


def _cols(rows, names, side):
    return {n: torch.tensor([r[side][i] for r in rows], **KW)
            for i, n in enumerate(names)}


#: The canopy layer boundaries the oracle uses, VIC's default Ncanopy = 3.
BND = torch.tensor([[1.0 / 3.0], [2.0 / 3.0], [1.0]], **KW)


def _run(cin):
    bnd = BND.expand(3, cin['LAI'].numel())
    apar = carbon.fapar(cin['LAI'], bnd, cin['CosZen'], cin['AlbSoilPAR'],
                        cin['Fdir'])
    nscale = carbon.nscale_factors(
        torch.ones_like(cin['LAI'], dtype=torch.bool).expand(3, -1),
        bnd, cin['LAI'], cin['CosZen'])
    res = carbon.canopy_assimilation(
        cin['Ctype'], cin['MaxCarboxRate'], cin['MaxETransport'],
        cin['CO2Specificity'], nscale, cin['Tfoliage'], cin['SWdown'], apar,
        cin['elevation'], cin['Catm'], bnd, cin['LAI'], 'ci')
    return apar, nscale, res


def test_the_oracle_covers_both_plant_types_and_both_light_regimes(cases):
    cin = _cols(cases, IN_COLS, 0)
    assert int((cin['Ctype'] == 0).sum()) > 20
    assert int((cin['Ctype'] == 1).sum()) > 20
    # Sun above and below PHOTO_ZENITHMINPAR is not what varies here -- the
    # oracle's lowest zenith is 0.05 -- but leaf area above and below
    # PHOTO_LAILIMIT is, and that is what switches the nitrogen scaling on.
    assert int((cin['LAI'] > carbon.PHOTO_LAILIMIT).sum()) > 20
    assert int((cin['LAI'] <= carbon.PHOTO_LAILIMIT).sum()) > 20


def test_absorbed_par_matches_vic(cases):
    cin = _cols(cases, IN_COLS, 0)
    want = _cols(cases, OUT_COLS, 1)
    apar, _, _ = _run(cin)
    for layer in range(3):
        got = apar[layer]
        w = want['faPAR%d' % layer]
        err = (got - w).abs().max()
        assert float(err / w.abs().max()) < 1e-10, (layer, float(err))


def test_nitrogen_scaling_matches_vic(cases):
    cin = _cols(cases, IN_COLS, 0)
    want = _cols(cases, OUT_COLS, 1)
    _, nscale, _ = _run(cin)
    for layer in range(3):
        assert torch.allclose(nscale[layer], want['Nscale%d' % layer],
                              rtol=1e-10, atol=0.0), layer


@pytest.mark.parametrize('name', ('GPP', 'Rdark', 'Rphoto', 'Rmaint',
                                  'Rgrowth', 'Raut', 'NPP', 'Ci'))
def test_the_canopy_fluxes_match_vic(cases, name):
    cin = _cols(cases, IN_COLS, 0)
    want = _cols(cases, OUT_COLS, 1)
    _, _, res = _run(cin)
    got = res[name]
    w = want[name]
    scale = w.abs().max()
    err = (got - w).abs().max()
    print('%-8s max |error| %.3e against a range of %.3e, relative %.3e'
          % (name, err, scale, err / scale))
    assert float(err / scale) < 1e-10, (name, float(err), float(scale))


def test_the_stomatal_resistance_matches_vic(cases):
    """Including the elements VIC caps at HUGE_RESIST, which is where the
    assimilation does not exceed the dark respiration.

    Compared as a conductance, not a resistance. photosynth.c:504-506 forms
    rs from 1 / (Agross - Rdark), and where the nitrogen scaling is at its
    1e-10 floor those two are nearly equal tiny numbers: one element of the
    288 here has a net assimilation that has cancelled to five significant
    digits, and its resistance differs by 4e-05 relative on a value of
    2.1e+09 s/m. That is the same closed stoma either way, and a conductance
    says so where a resistance does not.
    """
    cin = _cols(cases, IN_COLS, 0)
    want = _cols(cases, OUT_COLS, 1)
    _, _, res = _run(cin)
    capped = 0
    worst_r, worst_g = 0.0, 0.0
    for layer in range(3):
        got = res['rsLayer'][layer]
        w = want['rs%d' % layer]
        at_cap = w >= carbon.HUGE_RESIST
        capped += int(at_cap.sum())
        assert torch.equal(got[at_cap], w[at_cap]), layer
        free = ~at_cap
        if not int(free.sum()):
            continue
        rel = ((got[free] - w[free]).abs() / w[free].abs()).max()
        worst_r = max(worst_r, float(rel))
        # The conductance, against the largest conductance in the set.
        cg, cw = 1.0 / got[free], 1.0 / w[free]
        gerr = (cg - cw).abs().max() / cw.abs().max()
        worst_g = max(worst_g, float(gerr))
    print('stomatal resistance: worst %.3e relative; as a conductance %.3e'
          % (worst_r, worst_g))
    assert worst_g < 1e-10, worst_g
    assert capped > 10, capped


def test_the_canopy_resistance_matches_vic(cases):
    cin = _cols(cases, IN_COLS, 0)
    want = _cols(cases, OUT_COLS, 1)
    _, _, res = _run(cin)
    got, w = res['rc'], want['rc']
    rel = ((got - w).abs() / w.abs()).max()
    print('rc: max relative error %.3e over a range of %.3e to %.3e s/m'
          % (rel, w.min(), w.max()))
    assert float(rel) < 1e-10, float(rel)


def test_soil_respiration_matches_vic(sr_cases):
    cin = _cols(sr_cases, SR_IN, 0)
    want = _cols(sr_cases, SR_OUT, 1)
    n = cin['dt'].numel()
    dz = torch.tensor([[100.0], [300.0], [1000.0]], **KW).expand(3, n)
    t_node = torch.stack([cin['T0'], cin['T1'], cin['T2']], 0)
    w_node = torch.stack([cin['w0'], cin['w1'], cin['w2']], 0)
    rl, ri, rs = carbon.soil_resp(dz, 1400.0, cin['dt'], t_node, w_node,
                                 cin['CLitter'], cin['CInter'], cin['CSlow'])
    for got, w, name in ((rl, want['RhLitter'], 'RhLitter'),
                         (ri, want['RhInter'], 'RhInter'),
                         (rs, want['RhSlow'], 'RhSlow')):
        rel = ((got - w).abs() / w.abs().clamp(min=1e-300)).max()
        print('%-9s max relative error %.3e' % (name, rel))
        assert float(rel) < 1e-10, (name, float(rel))


def test_the_pools_close_over_a_step():
    """soil_carbon_balance.c:176-186 moves carbon between three pools and to
    the air. What leaves the litter must arrive somewhere."""
    n = 5
    one = torch.ones(n, **KW)
    dz = torch.tensor([[100.0], [300.0], [1000.0]], **KW).expand(3, n)
    t_node = torch.stack([5.0 * one, 6.0 * one, 7.0 * one], 0)
    w_node = torch.stack([0.4 * one, 0.5 * one, 0.6 * one], 0)
    c_l, c_i, c_s = 600.0 * one, 3500.0 * one, 10000.0 * one
    out = carbon.soil_carbon_balance(dz, None, 1400.0, 3600.0, t_node,
                                     w_node, c_l, c_i, c_s, 400.0 * one)
    before = c_l + c_i + c_s
    after = out['CLitter'] + out['CInter'] + out['CSlow']
    to_air = out['RhLitter2Atm'] + out['RhInter'] + out['RhSlow']
    err = (after - (before + out['Litterfall'] - to_air)).abs().max()
    assert float(err) < 1e-9, float(err)
    assert float(out['RhTot'].min()) > 0.0


def test_the_fluxes_carry_gradients_to_the_plant_parameters():
    """MaxCarboxRate and MaxETransport are the two knobs a carbon
    calibration would turn, so net production has to be differentiable in
    them rather than merely a function of them.

    Both, which needs both regimes in the batch: gross assimilation is the
    smaller of the carboxylation- and the transport-limited rate
    (photosynth.c:404-412), so in any one light level only one of the two
    parameters reaches it. The light is swept here for that reason.
    """
    n = 8
    one = torch.ones(n, **KW)
    vcmax = (33.0e-6 * one).requires_grad_(True)
    jmax = (75.0e-6 * one).requires_grad_(True)
    bnd = BND.expand(3, n)
    lai = 4.0 * one
    coszen = 0.1 + 0.1 * torch.arange(n, **KW)
    swdown = 20.0 + 120.0 * torch.arange(n, **KW)
    # Per leaf area and in moles of photons, which is what VIC hands
    # canopy_assimilation (surface_fluxes.c:419-434) and what puts the
    # light-limited regime within reach.
    frac = carbon.fapar(lai, bnd, coszen, 0.08 * one, 0.7 * one)
    apar = carbon.apar_per_leaf(0.45 * swdown, frac,
                                carbon.canopy_layer_lai(lai, bnd))
    nscale = carbon.nscale_factors(
        torch.ones(n, dtype=torch.bool).expand(3, -1), bnd, lai, coszen)
    res = carbon.canopy_assimilation(
        torch.zeros(n, dtype=torch.long), vcmax, jmax, torch.zeros(n, **KW),
        nscale, 20.0 * one, swdown, apar, 800.0 * one, 3.7e-4 * one,
        bnd, lai, 'ci')
    res['NPP'].sum().backward()
    for g, name in ((vcmax.grad, 'MaxCarboxRate'),
                    (jmax.grad, 'MaxETransport')):
        assert g is not None and torch.isfinite(g).all(), name
        assert float(g.abs().max()) > 0.0, name


def test_the_soil_pools_carry_a_gradient_to_their_turnover():
    """The three pools are a store a calibration would fit, and their
    respiration has to be differentiable in the carbon they hold."""
    n = 5
    one = torch.ones(n, **KW)
    dz = torch.tensor([[100.0], [300.0], [1000.0]], **KW).expand(3, n)
    t_node = torch.stack([5.0 * one, 6.0 * one, 7.0 * one], 0)
    w_node = torch.stack([0.4 * one, 0.5 * one, 0.6 * one], 0)
    c_l = (600.0 * one).requires_grad_(True)
    c_i = (3500.0 * one).requires_grad_(True)
    c_s = (10000.0 * one).requires_grad_(True)
    out = carbon.soil_carbon_balance(dz, None, 1400.0, 3600.0, t_node,
                                     w_node, c_l, c_i, c_s, 400.0 * one)
    out['RhTot'].sum().backward()
    for g, name in ((c_l.grad, 'CLitter'), (c_i.grad, 'CInter'),
                    (c_s.grad, 'CSlow')):
        assert g is not None and torch.isfinite(g).all(), name
        assert float(g.abs().max()) > 0.0, name


# ---------------------------------------------------------------------------
# RC_MODE = RC_PHOTO: the canopy resistance photosynthesis implies, which is
# the path on which turning CARBON on changes transpiration rather than only
# adding diagnostics.
# ---------------------------------------------------------------------------
RCPS_IN = ('LAI', 'CosZen', 'Tair', 'SWdown', 'PAR', 'vpd', 'gsm_inv',
           'Catm', 'Ctype', 'MaxCarboxRate', 'MaxETransport',
           'CO2Specificity')
RCPS_OUT = ('rc', 'rs0', 'rs1', 'rs2', 'aPAR0', 'aPAR1', 'aPAR2')


@pytest.fixture(scope='module')
def rcps_cases():
    return _oracle('rcps')


def _rcps_inputs(cin):
    n = cin['LAI'].numel()
    bnd = BND.expand(3, n)
    frac = carbon.fapar(cin['LAI'], bnd, cin['CosZen'],
                        torch.full_like(cin['LAI'], 0.08),
                        torch.full_like(cin['LAI'], 0.7))
    lai_layer = carbon.canopy_layer_lai(cin['LAI'], bnd)
    apar = carbon.apar_per_leaf(cin['PAR'], frac, lai_layer)
    nscale = carbon.nscale_factors(
        torch.ones(n, dtype=torch.bool).expand(3, -1), bnd, cin['LAI'],
        cin['CosZen'])
    return bnd, apar, nscale


def test_absorbed_par_per_leaf_matches_vic(rcps_cases):
    """The conversion from the absorbed fraction to moles of photons per
    leaf area, surface_fluxes.c:419-434. Three orders of magnitude separate
    the two, so getting it wrong is not a rounding error."""
    cin = _cols(rcps_cases, RCPS_IN, 0)
    want = _cols(rcps_cases, RCPS_OUT, 1)
    _, apar, _ = _rcps_inputs(cin)
    for layer in range(3):
        w = want['aPAR%d' % layer]
        rel = ((apar[layer] - w).abs() / w.abs()).max()
        assert float(rel) < 1e-10, (layer, float(rel))


def test_the_photosynthetic_canopy_resistance_matches_vic(rcps_cases):
    cin = _cols(rcps_cases, RCPS_IN, 0)
    want = _cols(rcps_cases, RCPS_OUT, 1)
    bnd, apar, nscale = _rcps_inputs(cin)
    rc, rs_layer = carbon.calc_rc_ps(
        cin['Ctype'], cin['MaxCarboxRate'], cin['MaxETransport'],
        cin['CO2Specificity'], nscale, cin['Tair'], cin['SWdown'], apar,
        torch.full_like(cin['LAI'], 800.0), cin['Catm'], bnd, cin['LAI'],
        cin['gsm_inv'], cin['vpd'])
    rel = ((rc - want['rc']).abs() / want['rc'].abs()).max()
    print('rc under RC_PHOTO: max relative error %.3e over %.3f to %.3f s/m'
          % (rel, want['rc'].min(), want['rc'].max()))
    assert float(rel) < 1e-10, float(rel)
    capped = 0
    for layer in range(3):
        w = want['rs%d' % layer]
        at_cap = w >= carbon.CANOPY_RSMAX
        capped += int(at_cap.sum())
        assert torch.equal(rs_layer[layer][at_cap], w[at_cap]), layer
        free = ~at_cap
        if int(free.sum()):
            r = ((rs_layer[layer][free] - w[free]).abs() / w[free].abs()).max()
            assert float(r) < 1e-10, (layer, float(r))
    # The cap has to be reached somewhere, or it is untested.
    assert capped > 5, capped


def test_the_photosynthetic_resistance_responds_to_the_stress_it_is_given():
    """rc = rc0 / (gsm_inv * vpdfactor), so drying the soil or opening the
    vapour pressure deficit must close the canopy, up to the cap."""
    n = 6
    one = torch.ones(n, **KW)
    bnd = BND.expand(3, n)
    lai = 3.0 * one
    frac = carbon.fapar(lai, bnd, 0.5 * one, 0.08 * one, 0.7 * one)
    apar = carbon.apar_per_leaf(150.0 * one, frac,
                               carbon.canopy_layer_lai(lai, bnd))
    nscale = carbon.nscale_factors(
        torch.ones(n, dtype=torch.bool).expand(3, -1), bnd, lai, 0.5 * one)

    def rc_at(gsm_inv, vpd):
        rc, _ = carbon.calc_rc_ps(
            torch.zeros(n, dtype=torch.long), 33.0e-6 * one, 75.0e-6 * one,
            torch.zeros(n, **KW), nscale, 18.0 * one, 300.0 * one, apar,
            800.0 * one, 3.7e-4 * one, bnd, lai, gsm_inv * one, vpd * one)
        return float(rc.max())

    assert rc_at(1.0, 200.0) < rc_at(0.4, 200.0)
    assert rc_at(1.0, 200.0) < rc_at(1.0, 3000.0)
    # Monotone in both, and never past the cap. It does not reach the cap
    # here: an unstressed rc0 of about 0.07 s/m survives a stress of
    # 1/(0.05 * 0.125) and still comes out near 11 s/m, which is worth
    # knowing -- CANOPY_RSMAX binds on a sparse or cold canopy, not on
    # a dry one.
    assert rc_at(0.05, 3500.0) > rc_at(0.4, 3000.0)
    assert rc_at(0.05, 3500.0) <= carbon.CANOPY_RSMAX


def test_the_saturation_inhibition_matches_its_endpoints():
    """canopy_evap.c:263-275: no inhibition below the threshold, and exactly
    NPPfactor_sat at saturation."""
    n = 5
    one = torch.ones(n, **KW)
    wmax = 100.0 * one
    f = carbon.npp_factor(50.0 * one, wmax, 0.8 * one, 0.5 * one)
    assert torch.allclose(f, one)
    f = carbon.npp_factor(wmax, wmax, 0.8 * one, 0.5 * one)
    assert torch.allclose(f, 0.5 * one)
    f = carbon.npp_factor(90.0 * one, wmax, 0.8 * one, 0.5 * one)
    assert torch.allclose(f, 0.75 * one)


def test_the_solar_zenith_cosine_matches_vic():
    """compute_coszen.c:31-79, which is what decides whether faparl runs at
    all and how the nitrogen scales down the canopy."""
    from torchvic.core import atmos
    rows = _oracle('coszen')
    cin = _cols(rows, ('lat', 'lng', 'tzlng', 'doy', 'sec'), 0)
    want = torch.tensor([r[1][0] for r in rows], **KW)
    got = atmos.coszen(cin['lat'], cin['lng'], cin['tzlng'], cin['doy'],
                       cin['sec'])
    err = (got - want).abs().max()
    print('coszen: max |error| %.3e over %d states spanning %.3f to %.3f'
          % (err, want.numel(), want.min(), want.max()))
    assert float(err) < 1e-12, float(err)
    # Night and day both present, or the comparison says little.
    assert int((want < 0).sum()) > 20 and int((want > 0.5).sum()) > 5
