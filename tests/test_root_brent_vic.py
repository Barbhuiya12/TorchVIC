"""implicit.root_brent against a line-for-line scalar transcription of VIC's
root_brent.c 5.1.0:227-294 (identical in 5.0.1): the same trial points and
stopping rule, so the same root to the last bit on smooth residuals."""
import math
import sys

import torch

from torchvic.core.implicit import root_brent

EPS = sys.float_info.epsilon


def c_root_brent(f, a, b, T=1e-7, maxiter=1000):
    fa, fb = f(a), f(b)
    assert fa * fb < 0
    fc = fb
    c = d = e = 0.0
    for _ in range(maxiter):
        if fb * fc > 0:
            c, fc = a, fa
            d = b - a
            e = d
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2 * EPS * abs(b) + T
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0:
            return b
        if abs(e) < tol or abs(fa) <= abs(fb):
            d = m
            e = d
        else:
            s = fb / fa
            if a == c:
                p = 2 * m * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            else:
                p = -p
            s = e
            e = d
            if 2 * p < 3 * m * q - abs(tol * q) and p < abs(0.5 * s * q):
                d = p / q
            else:
                d = m
                e = d
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = f(b)
    raise RuntimeError('too many iterations')


def test_root_brent_matches_c_transcription():
    g = torch.Generator().manual_seed(3)
    n = 400
    k = torch.rand(n, generator=g, dtype=torch.float64) * 40 + 1
    r = torch.rand(n, generator=g, dtype=torch.float64) * 30 - 15
    w = torch.rand(n, generator=g, dtype=torch.float64) * 1e-3

    def ft(t, k_, r_, w_):
        # a surface-balance-like residual: radiative T^4, linear, cubic
        return (-w_ * (t + 273.15) ** 4 + w_ * (r_ + 273.15) ** 4
                - k_ * (t - r_) - 0.01 * (t - r_) ** 3)

    lo = r - 1.0 - torch.rand(n, generator=g, dtype=torch.float64) * 20
    hi = r + 1.0 + torch.rand(n, generator=g, dtype=torch.float64) * 20
    got = root_brent(ft, lo, hi, ft(lo, k, r, w), ft(hi, k, r, w),
                     args=(k, r, w))
    for i in range(n):
        ki, ri, wi = float(k[i]), float(r[i]), float(w[i])

        def fs(t):
            return (-wi * (t + 273.15) ** 4 + wi * (ri + 273.15) ** 4
                    - ki * (t - ri) - 0.01 * (t - ri) ** 3)
        want = c_root_brent(fs, float(lo[i]), float(hi[i]))
        assert math.isclose(float(got[i]), want, rel_tol=0, abs_tol=1e-12), i


def test_root_brent_stops_at_vic_tolerance_not_tighter():
    # VIC stops once the bracket is within 2 eps |b| + 1e-7: the returned
    # root is within that of the true one, and generally not the exact root.
    f = lambda t: t ** 3 - 2.0                                  # noqa: E731
    lo = torch.tensor([0.0], dtype=torch.float64)
    hi = torch.tensor([5.0], dtype=torch.float64)
    got = float(root_brent(f, lo, hi, f(lo), f(hi))[0])
    assert abs(got - 2.0 ** (1 / 3)) < 2e-7
    assert got == c_root_brent(lambda t: t ** 3 - 2.0, 0.0, 5.0)
