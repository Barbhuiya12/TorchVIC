#!/usr/bin/env python3
"""Check torchvic.core.implicit against roots whose derivatives are known exactly.

The snow surface temperature is found by root-finding, so its gradient comes
from the implicit function theorem rather than from transcription. That
machinery has to be correct before any physics is built on it, and it can be
checked without physics at all: pick equations whose roots and derivatives
have closed forms, and compare.

Run: tests/test_implicit.py
"""
import os
import sys


import torch                                     # noqa: E402
from torchvic.core import implicit  # noqa: E402

# This project is GPU-native and the differentiable half is no exception.
# Override with TORCHVIC_DEVICE=cpu.
DEVICE = os.environ.get('TORCHVIC_DEVICE') or (
    'cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)


RTOL = 1e-9


def report(name, got, want, failures, rtol=RTOL):
    got = torch.as_tensor(got, dtype=torch.float64)
    want = torch.as_tensor(want, dtype=torch.float64)
    err = (got - want).abs()
    rel = err / want.abs().clamp(min=1e-30)
    ok = bool((rel < rtol).all())
    print('  %-42s max|rel| %9.3g  %s'
          % (name, rel.max().item(), 'PASS' if ok else 'FAIL'))
    if not ok:
        print('      got  %s' % got.tolist())
        print('      want %s' % want.tolist())
        failures.append(name)


def main():
    failures = []
    print('torchvic.core.implicit against analytic roots')

    # 1. f(T, a) = T^2 - a.  Root sqrt(a); dT/da = 1/(2 sqrt(a)).
    a = torch.tensor([2.0, 9.0, 100.0], dtype=torch.float64,
                     requires_grad=True)
    T = implicit.solve(lambda t, a_: t * t - a_,
                       torch.zeros_like(a.detach()),
                       torch.full_like(a.detach(), 50.0), args=(a,))
    report('root of T^2 - a', T.detach(), a.detach().sqrt(), failures)
    T.sum().backward()
    report('d/da of sqrt(a)', a.grad, 1.0 / (2.0 * a.detach().sqrt()),
           failures)

    # 2. f(T, k, c) = exp(kT) - c.  Root ln(c)/k.
    #    dT/dk = -ln(c)/k^2 ; dT/dc = 1/(k c)
    k = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64, requires_grad=True)
    # `method='bisect'` deliberately. exp(kT) - c over [0, 20] spans the
    # bracket by seventeen orders of magnitude -- f(20) is 2.4e17 against
    # f(0) of -6 -- and false position parks its trial point on the lower
    # endpoint there. The default Illinois solver, at its twelve iterations,
    # returns 3.0 where the root is 0.973; it needs about sixty. That is a
    # real property of the solver and is stated in its docstring rather than
    # hidden here: twelve is calibrated to this model's own energy balances,
    # which are smooth over their brackets, and is not a general guarantee.
    # An analytic check of the implicit-function-theorem gradient wants the
    # solver that always converges, so it asks for it.
    T = implicit.solve(lambda t, k_, c_: torch.exp(k_ * t) - c_,
                       torch.zeros(3, dtype=torch.float64),
                       torch.full((3,), 20.0, dtype=torch.float64),
                       args=(k, c), method='bisect')
    report('root of exp(kT) - c', T.detach(),
           c.detach().log() / k.detach(), failures)
    T.sum().backward()
    report('d/dk of ln(c)/k', k.grad,
           -c.detach().log() / k.detach() ** 2, failures)
    report('d/dc of ln(c)/k', c.grad,
           1.0 / (k.detach() * c.detach()), failures)

    # 3. A shape closer to the snow surface energy balance: a quartic
    #    radiative term against a linear turbulent term.
    #    f(T, R, h) = R - sigma*(T+273.15)^4 - h*T = 0, solved for T.
    #    No closed form, so compare the gradient against a central finite
    #    difference of the root itself. This is the check that matters --
    #    it is the same form the physics will take.
    sigma = 5.67e-8
    R = torch.tensor([300.0, 500.0], dtype=torch.float64, requires_grad=True)
    h = torch.tensor([10.0, 25.0], dtype=torch.float64, requires_grad=True)

    def energy(t, R_, h_):
        return R_ - sigma * (t + 273.15) ** 4 - h_ * t

    lo = torch.full((2,), -60.0, dtype=torch.float64)
    hi = torch.full((2,), 60.0, dtype=torch.float64)
    T = implicit.solve(energy, lo, hi, args=(R, h))
    resid = energy(T.detach(), R.detach(), h.detach()).abs().max()
    print('  %-42s %9.3g' % ('residual at the root', resid.item()))
    T.sum().backward()

    eps = 1e-6
    fd = []
    for i in range(2):
        d = []
        for var, base in ((0, R), (1, h)):
            up = [R.detach().clone(), h.detach().clone()]
            dn = [R.detach().clone(), h.detach().clone()]
            up[var][i] += eps
            dn[var][i] -= eps
            tu = implicit.bisect(energy, lo, hi, args=tuple(up))
            td = implicit.bisect(energy, lo, hi, args=tuple(dn))
            d.append(((tu[i] - td[i]) / (2 * eps)).item())
        fd.append(d)

    report('dT/dR vs finite difference',
           R.grad, torch.tensor([fd[0][0], fd[1][0]], dtype=torch.float64),
           failures, rtol=1e-5)
    report('dT/dh vs finite difference',
           h.grad, torch.tensor([fd[0][1], fd[1][1]], dtype=torch.float64),
           failures, rtol=1e-5)

    print()
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
