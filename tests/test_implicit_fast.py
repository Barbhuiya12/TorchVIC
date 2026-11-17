"""soiltemp._newt_raph_fast against the per-node transcription, bit for bit.

The batched implicit solver evaluates every node and every Jacobian
column of newt_raph (frozen_soil.c:624, newt_raph_func_fast.c) at once,
unrolling fda_heat_eqn's statics into three per-node arrays. Its answer
must be the per-node `_newt_raph`'s to the bit -- which is itself held
to VIC's own solve_T_profile_implicit by test_implicit_profile.py -- on
batches wide enough that ATen's vectorized pow and its scalar tail both
run, frozen and thawed, with and without NOFLUX and EXP_TRANS, on
profiles that converge and profiles that exhaust NEWT_RAPH_MAXTRIAL.
"""
import pytest
import torch

import _paths                                           # noqa: F401,E402
from torchvic.core import frozen, soiltemp

DT = torch.float64


def _case(ncol, nn, noflux, exp_trans, seed):
    """A batch of `ncol` soil columns of `nn` nodes, and the solver's
    equation object built from it as solve_profile_implicit builds it."""
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*s, generator=g,
                                                      dtype=DT)
    if exp_trans:
        dp = torch.full((ncol,), 4.0, dtype=DT)
        bexp = torch.log(dp[0] + 1.0) / (nn - 1)
        zsum = (torch.exp(bexp * torch.arange(nn, dtype=DT)) - 1.0)
    else:
        dp = torch.full((ncol,), 4.0, dtype=DT)
        zsum = torch.cumsum(torch.cat([torch.zeros(1, dtype=DT),
                                       u(0.1, 0.8, nn - 1)]), 0)
    zsum = zsum.expand(ncol, nn).clone()
    # nn - 1 spacings (the NOFLUX bottom row reads the last), from the
    # grid continued one node past its base.
    zx = torch.cat([zsum, 2 * zsum[:, -1:] - zsum[:, -2:-1]], 1)
    alpha = zx[:, 2:] - zx[:, :-2]
    beta = zx[:, 1:-1] - zx[:, :-2]
    gamma = zx[:, 2:] - zx[:, 1:-1]
    depth = torch.tensor([0.1, 0.4, 2.0], dtype=DT).expand(ncol, 3).clone()
    # Profiles either side of freezing, some harsh enough to fail.
    t0 = u(-12.0, 6.0, ncol, nn)
    t0[:, 0] = u(-25.0, 15.0, ncol)
    moist = u(0.08, 0.42, ncol, nn)
    max_moist = torch.full((ncol, nn), 0.45, dtype=DT)
    bubble = u(15.0, 45.0, ncol, nn)
    expt = u(7.0, 13.0, ncol, nn)
    bdmin = torch.full((ncol, 3), 1450.0, dtype=DT)
    sdmin = torch.full((ncol, 3), 2685.0, dtype=DT)
    quartz = u(0.05, 0.7, ncol, 3)
    bd = u(1300.0, 1600.0, ncol, 3)
    sd = torch.full((ncol, 3), 2685.0, dtype=DT)
    organic = u(0.0, 0.3, ncol, 3)
    ice = moist - frozen.maximum_unfrozen_water(t0, max_moist, bubble,
                                                expt)
    ice = torch.where((ice > 0) & (t0 < 0), ice, torch.zeros_like(ice))
    # Half the columns open with no ice: their trial ice then differs
    # from the step's, so the stale-static paths are exercised.
    ice[ncol // 2:] = 0.0
    lay = lambda t: t[:, 1:2].expand(ncol, nn)
    kappa = frozen.soil_conductivity(moist, moist - ice, lay(sdmin),
                                     lay(bdmin), lay(quartz), lay(sd),
                                     lay(bd), lay(organic))
    cs = frozen.volumetric_heat_capacity(lay(bd) / lay(sd), moist - ice,
                                         ice, lay(organic))
    n = nn - 1 if noflux else nn - 2
    eqn = soiltemp._FdaHeatEqn(
        t0, moist, ice, kappa, cs, max_moist, bubble, expt, alpha, beta,
        gamma, zsum, 10800.0, dp, bdmin, sdmin, quartz, bd, sd, organic,
        depth, 3, n, noflux, exp_trans)
    return eqn, t0[:, 1:n + 1].clone(), n


@pytest.mark.parametrize('noflux', [False, True])
@pytest.mark.parametrize('exp_trans', [False, True])
@pytest.mark.parametrize('ncol,nn', [(37, 10), (5, 4), (70, 7)])
def test_fast_newton_is_the_per_node_newton(noflux, exp_trans, ncol, nn):
    eqn, x, n = _case(ncol, nn, noflux, exp_trans, seed=ncol * nn)
    want, werr = soiltemp._newt_raph(eqn, x.clone(), n)
    eqn2, x2, _ = _case(ncol, nn, noflux, exp_trans, seed=ncol * nn)
    got, gerr = soiltemp._newt_raph_fast(eqn2, x2, n)
    assert torch.equal(gerr, werr)
    # Every column, converged or not: the iterates themselves agree.
    assert torch.equal(got, want)


def test_the_cases_cover_both_outcomes():
    """The synthetic batches converge in places and fail in others, so
    the per-column stopping and the NEWT_RAPH_MAXTRIAL exit both run."""
    eqn, x, n = _case(37, 10, False, False, seed=370)
    _, err = soiltemp._newt_raph_fast(eqn, x, n)
    assert bool(err.any()) and bool((~err).any())


@pytest.mark.parametrize('c', [1, 7, 16, 17, 31, 648, 4001])
def test_rowcall_rounds_as_per_row_calls(c):
    """ATen's CPU pow rounds a call's vectorized body and its scalar
    tail differently; _rowcall must reproduce k separate [C] calls,
    including at sizes where the stacked call would be threaded."""
    g = torch.Generator().manual_seed(c)
    k = 11
    a = torch.rand(k, c, generator=g, dtype=DT) * 3 + 1e-3
    b = -torch.rand(k, c, generator=g, dtype=DT) * 6
    for fn, args in ((frozen._pow, (a, b)), (frozen._pow, (7.7, a)),
                     (frozen._pow, (2.2, b)), (torch.log10, (a,))):
        want = torch.stack([fn(*[t[i].clone() if torch.is_tensor(t)
                                 else t for t in args])
                            for i in range(k)])
        assert torch.equal(soiltemp._rowcall(fn, *args), want)


def test_soil_conductivity_split_is_the_whole():
    """soil_conductivity_fixed + soil_conductivity_wu is the one
    expression soil_conductivity evaluated before the split."""
    g = torch.Generator().manual_seed(3)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(50, generator=g,
                                                   dtype=DT)
    moist, wu = u(0.0, 0.4), u(0.0, 0.4)
    wu[:10] = moist[:10]
    args = (u(2600, 2700), u(1300, 1600), u(0.0, 0.8), u(2600, 2700),
            u(1300, 1600), u(0.0, 0.3))
    k = frozen.soil_conductivity(moist, wu, *args)
    sdm, bdm, q, sd, bd, org = args
    ki, kw = 2.2, 0.57
    kdry_min = (0.135 * bdm + 64.7) / (sdm - 0.947 * bdm)
    kdry = (1 - org) * kdry_min + org * 0.05
    por = 1.0 - bd / sd
    sr = moist / torch.where(por != 0.0, por, torch.ones_like(por))
    ks_min = torch.where(q < 0.2, 7.7 ** q * 3.0 ** (1.0 - q),
                         7.7 ** q * 2.2 ** (1.0 - q))
    ks = (1 - org) * ks_min + org * 0.25
    safe_sr = torch.clamp(sr, min=torch.finfo(DT).tiny)
    ksat = torch.where(wu == moist, ks ** (1.0 - por) * kw ** por,
                       ks ** (1.0 - por) * ki ** (por - wu) * kw ** wu)
    ke = torch.where(wu == moist, 0.7 * torch.log10(safe_sr) + 1.0, sr)
    want = torch.maximum((ksat - kdry) * ke + kdry, kdry)
    want = torch.where(moist > 0.0, want, kdry)
    assert torch.equal(k, want)
