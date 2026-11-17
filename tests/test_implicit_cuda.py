"""The CUDA implicit profile solve (newtc.profile_cuda) against the torch
GPU solver, bit for bit.

The kernel runs solve_T_profile_implicit (frozen_soil.c:235-322) with
newt_raph (newt_raph_func_fast.c:18-169) one thread per column; its
arithmetic is the torch CUDA kernels' (CUDA pow, division by a CPU
scalar as a multiplication by the reciprocal, no FMA). Random columns
either side of freezing, converging and exhausting NEWT_RAPH_MAXTRIAL,
with and without NOFLUX, EXP_TRANS and the cold nose; a column subset;
a capture in a CUDA graph.
"""
import contextlib

import pytest
import torch

import _paths                                           # noqa: F401,E402
from torchvic.core import newtc, soiltemp
from test_implicit_c import _profile_inputs

pytestmark = pytest.mark.gpu

DT = torch.float64


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    if not newtc.cuda_available(torch.zeros(1, dtype=DT, device='cuda')):
        pytest.skip('no NVRTC')


@contextlib.contextmanager
def _torch_only():
    old = newtc.CUDA_ENABLED
    newtc.CUDA_ENABLED = False
    try:
        yield
    finally:
        newtc.CUDA_ENABLED = old


def _to(args, dev):
    return tuple(a.to(dev) if torch.is_tensor(a) else a for a in args)


@pytest.mark.parametrize('tfallback', [False, True])
@pytest.mark.parametrize('noflux', [False, True])
@pytest.mark.parametrize('exp_trans', [False, True])
@pytest.mark.parametrize('ncol,nn', [(37, 10), (650, 12)])
def test_kernel_is_the_torch_gpu_solve(tfallback, noflux, exp_trans, ncol,
                                       nn):
    _cuda()
    ts, args = _profile_inputs(ncol, nn, noflux, exp_trans, seed=ncol + nn)
    ts, args = ts.cuda(), _to(args, 'cuda')
    tail = (torch.ones(ncol, dtype=torch.bool, device='cuda'), True,
            tfallback)
    for k in range(2):
        tsk = ts + 0.37 * k
        with _torch_only():
            want = soiltemp._solve_profile_n(tsk, *args, *tail)
        cache = {}
        got = soiltemp._solve_profile_n(tsk, *args, *tail, cache=cache)
        assert cache, 'the kernel path did not run'
        assert torch.equal(got, want)
        # the raw solve: same failures, same profile where it converged
        a = _solve_args(tsk, args)
        t0 = args[0].clone()
        t0[:, 0] = tsk
        with _torch_only():
            w, we = soiltemp.solve_profile_implicit(
                tsk, t0, *a, tfallback=tfallback)
        g, ge = soiltemp._solve_implicit_c(
            tsk, args[0], *a, tfallback, None, {})
        assert bool(we.any()) or n > 0
        assert torch.equal(ge, we)
        assert torch.equal(g, w)


def _solve_args(ts, args):
    (t0, moist, ice, kappa, cs, maxm, bub, expt, al, be, ga, zsum, dt, dp,
     bdm, sdm, qz, bd, sd, org, depth, nl, nn, impl, noflux, exp_trans) = args
    n1 = nn - 1
    return (moist, ice, kappa, cs, maxm, bub, expt, al[:, :n1], be[:, :n1],
            ga[:, :n1], zsum, dt, dp, bdm, sdm, qz, bd, sd, org, depth, nl,
            nn, noflux, exp_trans)


def test_unwanted_columns():
    _cuda()
    ts, args = _profile_inputs(300, 12, False, False, seed=3)
    ts, args = ts.cuda(), _to(args, 'cuda')
    tail = (torch.ones(300, dtype=torch.bool, device='cuda'), True, True)
    cols = torch.arange(300, device='cuda') % 3 != 1
    with _torch_only():
        want = soiltemp._solve_profile_n(ts, *args, *tail, cols=cols)
    got = soiltemp._solve_profile_n(ts, *args, *tail, cols=cols, cache={})
    assert torch.equal(got[cols], want[cols])


def test_kernel_captures_in_a_graph():
    _cuda()
    ts, args = _profile_inputs(200, 12, True, False, seed=9)
    ts, args = ts.cuda(), _to(args, 'cuda')
    a = _solve_args(ts, args)
    cache = {}
    want, werr = soiltemp._solve_implicit_c(ts, args[0], *a, True,
                                            None, cache)
    tbuf = ts.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        soiltemp._solve_implicit_c(tbuf, args[0], *a, True, None,
                                   cache)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            out, oerr = soiltemp._solve_implicit_c(tbuf, args[0], *a,
                                                   True, None, cache)
    tbuf.copy_(ts + 1.0)
    g.replay()
    torch.cuda.synchronize()
    w1, e1 = soiltemp._solve_implicit_c(ts + 1.0, args[0], *a, True,
                                        None, cache)
    assert torch.equal(out, w1) and torch.equal(oerr, e1)
    assert not torch.equal(out, want)
