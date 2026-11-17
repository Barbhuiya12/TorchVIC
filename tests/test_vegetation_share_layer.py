"""canopy_evap.c transpiration(), both cases, against a scalar transcription
of the C loop. The reference below is the C line by line, one tile, one
layer at a time, so that the vectorised function's `where` selections and
its two deliberate quirks (spare_evap assigned, not accumulated; rootless
upper layers counted as dry) are checked on inputs that reach every branch.
"""

import os

import pytest
import torch

from torchvic.core import vegetation as V  # noqa: E402

D = torch.float64
CPU = torch.device('cpu')


def c_transpiration(avail, wcr, wpwp, root, lai, rmin, rgl, net_short, tair,
                    vpd, ra, rarc, rad, elev, share):
    """canopy_evap.c:74-215 for one tile, Python floats, no vectorisation."""
    n = len(root)
    t = lambda x: torch.tensor(float(x), dtype=D, device=CPU)
    avail_moist = [0.0] * n
    moist1 = 0.0
    wcr1 = 0.0
    for i in range(n - 1):
        if root[i] > 0.0:
            avail_moist[i] = avail[i]
            moist1 += avail_moist[i]
            wcr1 += wcr[i]
        else:
            avail_moist[i] = 0.0
    moist2 = avail[n - 1]
    avail_moist[n - 1] = moist2
    layerevap = [0.0] * n
    if share and ((moist1 >= wcr1 and moist2 >= wcr[n - 1] and wcr1 > 0.0)
                  or (moist1 >= wcr1 and (1 - root[n - 1]) >= 0.5)
                  or (moist2 >= wcr[n - 1] and root[n - 1] >= 0.5)):
        rc = float(V.calc_rc(t(rmin), t(net_short), t(rgl), t(tair), t(vpd),
                             t(lai), t(1.0)))
        evap = float(V.penman(t(tair), t(elev), t(rad), t(vpd), t(ra), t(rc),
                              t(rarc)))
        root_sum = 1.0
        spare_evap = 0.0
        for i in range(n):
            if avail_moist[i] >= wcr[i]:
                layerevap[i] = evap * root[i]
            else:
                if avail_moist[i] >= wpwp[i]:
                    gsm_inv = (avail_moist[i] - wpwp[i]) / (wcr[i] - wpwp[i])
                else:
                    gsm_inv = 0.0
                layerevap[i] = evap * gsm_inv * root[i]
                root_sum -= root[i]
                spare_evap = evap * root[i] * (1.0 - gsm_inv)
        if spare_evap > 0.0:
            for i in range(n):
                if avail_moist[i] >= wcr[i]:
                    layerevap[i] += root[i] * spare_evap / root_sum
    else:
        for i in range(n):
            if avail_moist[i] >= wcr[i]:
                gsm_inv = 1.0
            elif avail_moist[i] >= wpwp[i]:
                gsm_inv = (avail_moist[i] - wpwp[i]) / (wcr[i] - wpwp[i])
            else:
                gsm_inv = 0.0
            if gsm_inv > 0.0:
                rc = float(V.calc_rc(t(rmin), t(net_short), t(rgl), t(tair),
                                     t(vpd), t(lai), t(gsm_inv)))
                layerevap[i] = float(V.penman(t(tair), t(elev), t(rad),
                                              t(vpd), t(ra), t(rc), t(rarc))
                                     ) * root[i]
            else:
                layerevap[i] = 0.0
    return layerevap


def test_both_cases_match_the_c_loop_on_random_soils():
    # CPU-pinned: other tests in this directory flip the process default to CUDA
    g = torch.Generator(device=CPU).manual_seed(7)
    n_cases, nl = 400, 3
    wcr = torch.tensor([[30.0, 90.0, 300.0]], dtype=D, device=CPU).expand(n_cases, nl)
    wpwp = torch.tensor([[12.0, 40.0, 130.0]], dtype=D, device=CPU).expand(n_cases, nl)
    # moisture anywhere from below wilting to above critical, per layer
    avail = wpwp * 0.5 + (wcr * 1.4 - wpwp * 0.5) * torch.rand(n_cases, nl, generator=g, dtype=D, device=CPU)
    # roots: some tiles with a rootless upper layer, some bottom-heavy
    root = torch.rand(n_cases, nl, generator=g, dtype=D, device=CPU)
    root[::5, 1] = 0.0
    root[1::5, 2] = 0.8
    root = root / root.sum(-1, keepdim=True)
    lai = 0.2 + 2.0 * torch.rand(n_cases, generator=g, dtype=D, device=CPU)
    rmin = torch.full((n_cases,), 100.0, dtype=D, device=CPU)
    rgl = torch.full((n_cases,), 75.0, dtype=D, device=CPU)
    ns = 400.0 * torch.rand(n_cases, generator=g, dtype=D, device=CPU)
    tair = 5.0 + 25.0 * torch.rand(n_cases, generator=g, dtype=D, device=CPU)
    vpd = 200.0 + 3000.0 * torch.rand(n_cases, generator=g, dtype=D, device=CPU)
    ra = 20.0 + 100.0 * torch.rand(n_cases, generator=g, dtype=D, device=CPU)
    rarc = torch.full((n_cases,), 40.0, dtype=D, device=CPU)
    rad = ns - 60.0
    elev = torch.full((n_cases,), 1300.0, dtype=D, device=CPU)
    for share in (True, False):
        got = V.transpiration(avail, wcr, wpwp, root, lai, rmin, rgl, ns,
                              tair, vpd, ra, rarc, rad, elev,
                              share_layer_moist=share)
        worst = 0.0
        n_case1 = 0
        for k in range(n_cases):
            ref = c_transpiration(avail[k].tolist(), wcr[k].tolist(),
                                  wpwp[k].tolist(), root[k].tolist(),
                                  float(lai[k]), float(rmin[k]), float(rgl[k]),
                                  float(ns[k]), float(tair[k]), float(vpd[k]),
                                  float(ra[k]), float(rarc[k]), float(rad[k]),
                                  float(elev[k]), share)
            worst = max(worst, max(abs(float(got[k, i]) - ref[i])
                                   for i in range(nl)))
        assert worst < 1e-12, ('share=%s: vectorised transpiration differs '
                               'from the C loop by %.3e mm/day' % (share, worst))
    # the random soils must actually reach CASE 1 and CASE 2 both
    share_on = V.transpiration(avail, wcr, wpwp, root, lai, rmin, rgl, ns, tair,
                               vpd, ra, rarc, rad, elev, share_layer_moist=True)
    share_off = V.transpiration(avail, wcr, wpwp, root, lai, rmin, rgl, ns, tair,
                                vpd, ra, rarc, rad, elev, share_layer_moist=False)
    differs = (share_on - share_off).abs().sum(-1) > 1e-9
    assert 50 < int(differs.sum()) < n_cases - 50, (
        'only %d of %d cases distinguish CASE 1 from CASE 2' % (int(differs.sum()), n_cases))


def test_a_dry_top_over_a_wet_rooted_bottom_transpires_little_from_the_top():
    """The Walnut Gulch dry season: upper layers just above wilting point,
    the bottom layer -- most of the roots -- above Wcr. CASE 1: the top
    layers give evap * gsm_inv * root with a small gsm_inv, and the bottom
    takes its own share plus the last dry layer's shortfall. CASE 2 alone
    let the top layers transpire through the CANOPY_RSMAX cap, which is
    not proportional to gsm_inv and is much larger here. (Below wilting
    point both cases give exactly nothing, by VIC's `if (gsm_inv > 0.0)`.)
    """
    t = lambda *v: torch.tensor([list(v)], dtype=D, device=CPU)
    s = lambda v: torch.tensor([v], dtype=D, device=CPU)
    avail, wcr, wpwp = t(13.0, 42.0, 250.0), t(30.0, 90.0, 200.0), t(12.0, 40.0, 130.0)
    root = t(0.2, 0.2, 0.6)
    args = (s(0.5), s(50.0), s(100.0), s(600.0), s(27.0), s(3000.0), s(40.0),
            s(25.0), s(540.0), s(1300.0))
    on = V.transpiration(avail, wcr, wpwp, root, *args, share_layer_moist=True)[0]
    off = V.transpiration(avail, wcr, wpwp, root, *args, share_layer_moist=False)[0]
    assert float(off[0]) > 0.0 and float(off[1]) > 0.0
    # measured 0.0756 and 0.0544 against 0.1770 and 0.1770: CASE 1 scales
    # the unstressed rate by gsm_inv (0.06 and 0.04 here) and then adds
    # nothing to the dry layers, CASE 2 goes through the cap
    assert float(on[0]) < float(off[0]) and float(on[1]) < float(off[1]), (
        'CASE 1 top layers %.4f %.4f vs CASE 2 %.4f %.4f'
        % (float(on[0]), float(on[1]), float(off[0]), float(off[1])))
    # The shortfall goes to the wet bottom layer, and with it CASE 1's total
    # is the *larger*: measured 5.52 mm/day against CASE 2's 4.44
    # (bottom 5.39 vs 4.08). So whether CASE 1 lowers or raises a basin's
    # transpiration depends on which case its soils sit in; the year decides.
    assert float(on[2]) > float(off[2])
    assert float(on.sum()) > float(off.sum())


def test_the_flag_reaches_the_physics_from_a_global_file(tmp_path):
    """`options.SHARE_LAYER_MOIST` was implemented and unreachable: both
    cases of canopy_evap.c were transcribed and tested, but no reader
    consumed the flag, so a global file asking for CASE 2 alone got CASE 1
    anyway. That is the silent-wrong-answer shape this project exists to
    avoid, so the path from the file to the physics is tested, not just the
    physics.

    VIC's default is TRUE (initialize_options.c:67), so an absent key means
    TRUE and only an explicit FALSE turns the sharing off.
    """
    from torchvic.io.classic import ClassicOptions, GlobalParam
    from test_classic_equivalence import CLASSIC_GLOBAL, sample_env
    if not os.path.exists(CLASSIC_GLOBAL):
        pytest.skip('VIC 5.1.0 system-test global file absent: %s'
                    % CLASSIC_GLOBAL)

    def variant(line):
        g = tmp_path / ('global_%s.txt' % (line or 'absent').replace(' ', '_'))
        text = open(CLASSIC_GLOBAL).read()
        if line:
            text += '\n%s\n' % line
        g.write_text(text)
        with sample_env():
            return ClassicOptions(GlobalParam(str(g)))

    assert variant(None).share_layer_moist is True
    assert variant('SHARE_LAYER_MOIST TRUE').share_layer_moist is True
    assert variant('SHARE_LAYER_MOIST FALSE').share_layer_moist is False
    # And it is not refused: the option is honoured, not announced.
    assert not [r for r in variant('SHARE_LAYER_MOIST FALSE').refused
                if 'SHARE_LAYER_MOIST' in r]


def test_basin_simulate_passes_the_flag_down_to_the_canopy():
    """`basin.simulate(share_layer_moist=...)` has to reach
    `vegetation.transpiration`, through two `VegMosaic` methods. Checked by
    calling the mosaic the way simulate calls it and asserting the two
    answers differ on a soil where the cases differ -- a dry rooted top
    layer over a wet bottom, which is CASE 1's whole point."""
    from torchvic.core.vegetation import VegMosaic

    n, nt, nl = 1, 1, 3
    T = lambda v, *shape: torch.full(shape, float(v), dtype=D, device=CPU)
    mos = VegMosaic(cv=T(1.0, n, nt), lai=T(3.0, n, nt, 12),
                    rmin=T(100.0, n, nt), rgl=T(100.0, n, nt),
                    rarc=T(60.0, n, nt),
                    root_frac=torch.tensor([[[0.2, 0.2, 0.6]]], dtype=D,
                                           device=CPU),
                    dtype=D, device=CPU)
    # The top two layers are well below Wcr and the bottom, which holds
    # most of the roots, is well above -- canopy_evap.c:290-292's third
    # clause, `moist2 >= Wcr2 && root2 >= 0.5`, so CASE 1 applies and hands
    # the dry layers' shortfall to the wet one.
    avail = torch.tensor([[20.0, 30.0, 200.0]], dtype=D, device=CPU)
    wcr = torch.tensor([[60.0, 90.0, 120.0]], dtype=D, device=CPU)
    wpwp = torch.tensor([[10.0, 15.0, 20.0]], dtype=D, device=CPU)
    args = (0, T(0.0, n, nt), T(0.0, n), avail, wcr, wpwp,
            T(250.0, n), T(20.0, n), T(1200.0, n), T(60.0, n),
            T(200.0, n), T(500.0, n))
    on = mos.evap_per_tile(*args, share_layer_moist=True)[0]
    off = mos.evap_per_tile(*args, share_layer_moist=False)[0]
    assert float((on - off).abs().max()) > 1e-6
    # The default is VIC's.
    dflt = mos.evap_per_tile(*args)[0]
    assert float((dflt - on).abs().max()) == 0.0
    # CASE 1 puts the shortfall on the wet bottom layer; CASE 2 lets the dry
    # layers transpire on their own reduced resistance instead.
    assert float(on[..., -1].sum()) > float(off[..., -1].sum())
