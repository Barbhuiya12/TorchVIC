"""The compiled implicit profile solve (torchvic.core.newtc) against the
torch one, bit for bit, and the column masks that let it skip work.

newtc runs newt_raph (newt_raph_func_fast.c:18-169, fda_heat_eqn's
statics as soiltemp._newt_raph_fast unrolls them) as one C loop per
column; soiltemp._solve_implicit_c adds the boundaries and the cold
nose of solve_T_profile_implicit (frozen_soil.c:235-322). Both must be
the torch solver's answer to the bit -- which test_implicit_fast.py
holds to the per-node transcription -- on batches whose width puts
columns in ATen's scalar pow tail, frozen and thawed, with and without
NOFLUX and EXP_TRANS, converging and exhausting NEWT_RAPH_MAXTRIAL.
"""
import contextlib

import pytest
import torch

import _paths                                           # noqa: F401,E402
from torchvic.core import implicit, newtc, soiltemp
from test_implicit_fast import _case

DT = torch.float64


@contextlib.contextmanager
def _torch_only():
    old = newtc.ENABLED
    newtc.ENABLED = False
    try:
        yield
    finally:
        newtc.ENABLED = old


def test_the_compiled_loop_is_there():
    """On a CPU with a C compiler the fast path must not fall away
    silently (the run would be right, and 15x slower)."""
    import shutil
    if shutil.which('gcc') is None:
        pytest.skip('no C compiler')
    assert newtc.available(torch.zeros(1, dtype=DT))


@pytest.mark.parametrize('noflux', [False, True])
@pytest.mark.parametrize('exp_trans', [False, True])
@pytest.mark.parametrize('ncol,nn', [(37, 10), (17, 12), (650, 12)])
def test_compiled_newton_is_the_torch_newton(noflux, exp_trans, ncol, nn):
    if not newtc.available(torch.zeros(1, dtype=DT)):
        pytest.skip('no compiled loop')
    eqn, x, n = _case(ncol, nn, noflux, exp_trans, seed=ncol + nn)
    got, gerr = soiltemp._newt_raph_fast(eqn, x.clone(), n)
    with _torch_only():
        want, werr = soiltemp._newt_raph_fast(eqn, x.clone(), n)
    assert bool(werr.any()) and bool((~werr).any())
    assert torch.equal(gerr, werr)
    assert torch.equal(got, want)


@pytest.mark.parametrize('compiled', [True, False])
def test_unwanted_columns_change_nothing_else(compiled):
    """`cols`: the wanted columns come out as from the full solve, the
    others unchanged and failed."""
    if compiled and not newtc.available(torch.zeros(1, dtype=DT)):
        pytest.skip('no compiled loop')
    eqn, x, n = _case(70, 9, False, False, seed=11)
    cols = torch.rand(70, generator=torch.Generator().manual_seed(2)) < 0.4
    ctx = contextlib.nullcontext() if compiled else _torch_only()
    with ctx:
        full, ferr = soiltemp._newt_raph_fast(eqn, x.clone(), n)
        got, gerr = soiltemp._newt_raph_fast(eqn, x.clone(), n, cols)
    assert torch.equal(got[cols], full[cols])
    assert torch.equal(gerr[cols], ferr[cols])
    assert bool(gerr[~cols].all())
    assert torch.equal(got[~cols], x[~cols])


def _profile_inputs(ncol, nn, noflux, exp_trans, seed):
    eqn, _, _ = _case(ncol, nn, noflux, exp_trans, seed)
    g = torch.Generator().manual_seed(seed + 1)
    ts = eqn.t0[:, 0] + 6.0 * (torch.rand(ncol, generator=g, dtype=DT)
                               - 0.5)
    args = (eqn.t0, eqn.moist, eqn.ice, eqn.kappa, eqn.cs, eqn.max_moist,
            eqn.bubble, eqn.expt, eqn.alpha, eqn.beta, eqn.gamma, eqn.zsum,
            10800.0, eqn.dp, eqn.bulk_dens_min, eqn.soil_dens_min,
            eqn.quartz, eqn.bulk_density, eqn.soil_density, eqn.organic,
            eqn.depth, 3, nn, True, noflux, exp_trans)
    return ts, args


@pytest.mark.parametrize('frozen_soil', [False, True])
@pytest.mark.parametrize('noflux', [False, True])
@pytest.mark.parametrize('exp_trans', [False, True])
def test_compiled_profile_is_the_torch_profile(frozen_soil, noflux,
                                               exp_trans):
    """The whole trial solve -- Newton, boundaries, cold nose, the
    explicit sweep for the failed columns -- with and without the
    compiled path and its per-step cache, all columns and a subset."""
    if not newtc.available(torch.zeros(1, dtype=DT)):
        pytest.skip('no compiled loop')
    ts, args = _profile_inputs(53, 10, noflux, exp_trans, seed=5)
    tail = (torch.ones(53, dtype=torch.bool), frozen_soil, True)
    with _torch_only():
        want = soiltemp._solve_profile_n(ts, *args, *tail)
    cache = {}
    got = soiltemp._solve_profile_n(ts, *args, *tail, cache=cache)
    assert cache, 'the compiled path did not run'
    assert torch.equal(got, want)
    # a second trial on the same step's cache
    ts2 = ts + 0.37
    with _torch_only():
        want2 = soiltemp._solve_profile_n(ts2, *args, *tail)
    assert torch.equal(soiltemp._solve_profile_n(ts2, *args, *tail,
                                                 cache=cache), want2)
    cols = torch.arange(53) % 3 != 1
    got3 = soiltemp._solve_profile_n(ts2, *args, *tail, cols=cols,
                                     cache=cache)
    assert torch.equal(got3[cols], want2[cols])


def test_compacted_sweep_is_the_full_sweep(monkeypatch):
    """The explicit fallback on the failed columns only (where every
    operation rounds an element alike wherever it sits) is the full
    batch's sweep on those columns. Without EXP_TRANS (its float logf
    is batched) and frozen soil (pow) every operation is IEEE, so the
    compaction can be forced on any CPU."""
    ts, args = _profile_inputs(61, 9, False, False, seed=8)
    tail = (torch.ones(61, dtype=torch.bool), False, True)
    with _torch_only():
        want = soiltemp._solve_profile_n(ts, *args, *tail)
        monkeypatch.setattr(newtc, 'position_free', lambda d: True)
        got = soiltemp._solve_profile_n(ts, *args, *tail,
                                        cols=torch.ones(61, dtype=torch.bool))
    assert torch.equal(got, want)


def test_root_brent_live_ctx_keeps_the_root():
    """root_brent's `live_ctx` sees the elements still iterating and the
    root is the same: a residual that is wrong outside the mask is
    never read there."""
    g = torch.Generator().manual_seed(4)
    c = torch.rand(200, generator=g, dtype=DT) * 50 - 25
    state = {'mask': None}

    def f(x):
        r = x * x * x - c
        if state['mask'] is not None:
            r = torch.where(state['mask'], r, torch.full_like(r, float('nan')))
        return r

    @contextlib.contextmanager
    def ctx(mask):
        state['mask'] = mask
        try:
            yield
        finally:
            state['mask'] = None

    lo, hi = torch.full_like(c, -4.0), torch.full_like(c, 4.0)
    want = implicit.root_brent(f, lo, hi, f(lo), f(hi))
    got = implicit.root_brent(f, lo, hi, f(lo), f(hi), live_ctx=ctx)
    assert torch.equal(got, want)


def test_only_nests_and_restores():
    a = torch.tensor([True, True, False])
    b = torch.tensor([True, False, True])
    assert soiltemp.ACTIVE is None
    with soiltemp.only(a):
        with soiltemp.only(b):
            assert torch.equal(soiltemp.ACTIVE, a & b)
        assert torch.equal(soiltemp.ACTIVE, a)
    assert soiltemp.ACTIVE is None
