"""Exact-math mode (torchvic/core/xmath.py): each routed libm site returns
glibc's bits (math.*); with the mode off the helpers are the torch ops."""
import math

import pytest
import torch

from torchvic.core import column, frozen, xmath

N = 20003


@pytest.fixture
def exact(monkeypatch):
    monkeypatch.setattr(xmath, 'EXACT', True)


def _r(lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(N, dtype=torch.float64, generator=g) * (hi - lo) + lo


def _same(t, ref):
    assert [float(v) for v in t.tolist()] == ref


def test_helpers_match_math(exact):
    x, e, y = _r(.05, 5, 1), _r(-3, 3, 2), _r(-30, 30, 3)
    _same(xmath.pow(x, e), [math.pow(a, b) for a, b in zip(x.tolist(), e.tolist())])
    _same(xmath.pow(x, 0.5), [math.pow(a, .5) for a in x.tolist()])
    _same(xmath.pow(x, 3.0), [math.pow(a, 3.) for a in x.tolist()])
    _same(xmath.pow(10.0, x), [math.pow(10., a) for a in x.tolist()])
    _same(xmath.exp(y), [math.exp(a) for a in y.tolist()])
    _same(xmath.sqrt(x), [math.sqrt(a) for a in x.tolist()])


def test_off_is_torch():
    assert not xmath.EXACT
    x, e = _r(.05, 5, 4), _r(-3, 3, 5)
    assert torch.equal(xmath.pow(x, e), torch.pow(x, e))
    assert torch.equal(xmath.exp(x), torch.exp(x))
    assert torch.equal(xmath.sqrt(x), torch.sqrt(x))


def test_q12_site_matches_c_order(exact):
    ks, liq, res, mx, ex = (_r(1, 9, 6), _r(10, 40, 7), _r(1, 5, 8),
                            _r(45, 60, 9), _r(4, 12, 10))
    got = column._calc_q12_c(ks, liq, res, mx, ex)
    ref = [l - math.pow(math.pow(l - r, 1.0 - e) - k / math.pow(m - r, e)
                        * (1.0 - e), 1.0 / (1.0 - e)) - r
           for k, l, r, m, e in zip(*(t.tolist() for t in (ks, liq, res, mx, ex)))]
    _same(got, ref)


def test_forward_mode_dual_falls_through(exact):
    """implicit._local_dfdt puts a forward-mode tangent on T; exact mode must
    hand such a tensor to torch so the tangent survives."""
    import torch.autograd.forward_ad as fwAD
    x = _r(.05, 5, 11)
    with fwAD.dual_level():
        y = xmath.exp(fwAD.make_dual(x, torch.ones_like(x)))
        t = fwAD.unpack_dual(y).tangent
    assert t is not None and torch.equal(t, torch.exp(x))
