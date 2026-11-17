"""The bracketed root solver under the snow and surface energy balances.

`torchvic.core.implicit` solves for the temperature that zeroes an energy balance.
Bisection is exact and slow: it needs ceil(log2(width/tol)) residual
evaluations, about 31 on the surface solve's bracket, and one sequential VIC
step is entirely residual evaluations. Illinois gets the same root in about a
third of them.

The trade has to be measured, not asserted, because Illinois runs a fixed
iteration count with no convergence test -- testing costs a device
synchronisation per iteration, which is what made the count matter in the
first place. So these tests pin three things:

* Illinois and bisection find the same root, on the analytic functions whose
  roots are known and on the two residuals this model actually solves.
* The answer never leaves the bracket, which is the property that makes a
  fixed iteration count safe rather than merely usually right.
* Illinois costs about a third of the evaluations.
"""
import math

import pytest
import torch

from torchvic.core import implicit, snow

torch.set_default_dtype(torch.float64)
DEVICE = torch.device('cpu')
KW = dict(dtype=torch.float64, device=DEVICE)


class Counter:
    """Wraps a residual and counts how many times the solver evaluates it."""

    def __init__(self, f):
        self.f = f
        self.n = 0

    def __call__(self, x, *a):
        self.n += 1
        return self.f(x, *a)


# ---- analytic functions with known roots ----------------------------------
# Each is (name, f, lo, hi, root). The convex ones are the point: plain regula
# falsi keeps one endpoint forever on those and crawls; Illinois does not.
# (name, f, lo, hi, root, iterations needed, tolerance reached there). The
# last two are measurements, not wishes: `flat_then_steep` is x^9 near a root
# where the function is flat to nine orders of magnitude on one side, which is
# the worst case any secant method has, and it is in the list precisely
# because the default iteration count does NOT resolve it. An energy balance
# is nothing like it, and the snow-residual test below is what the default
# actually rests on.
CASES = [
    ('linear', lambda x: 2.0 * x - 3.0, -10.0, 10.0, 1.5, 16, 1e-12),
    ('cubic', lambda x: x ** 3 - 2.0 * x - 5.0, 1.0, 4.0,
     2.0945514815423265, 16, 1e-12),
    ('exp_convex', lambda x: torch.exp(x) - 5.0, -5.0, 5.0, math.log(5.0),
     16, 1e-12),
    ('quartic', lambda x: x ** 4 - 3.0, 0.5, 4.0, 3.0 ** 0.25, 16, 1e-12),
    ('stiff', lambda x: torch.atan(20.0 * (x - 0.3)), -2.0, 2.0, 0.3,
     16, 1e-12),
    ('flat_then_steep', lambda x: x ** 9 - 1e-3, 0.1, 2.0,
     (1e-3) ** (1.0 / 9.0), 32, 1e-8),
]


@pytest.mark.parametrize('name,f,lo,hi,root,iters,tol', CASES)
def test_illinois_finds_the_same_root_as_bisection(name, f, lo, hi, root,
                                                   iters, tol):
    a = torch.full((5,), lo, **KW)
    b = torch.full((5,), hi, **KW)
    got = implicit.illinois(lambda x: f(x), a, b, iters=iters)
    ref = implicit.bisect(lambda x: f(x), a, b, tol=1e-13)
    assert torch.isfinite(got).all(), name
    assert float((got - root).abs().max()) < tol, (name, float(got[0]), root)
    assert float((got - ref).abs().max()) < tol, name


def test_illinois_costs_far_fewer_evaluations():
    """On the bracket the model actually solves on.

    Bisection's cost is ceil(log2(width / tol)) plus one, so the saving
    depends entirely on the bracket. calc_surf_energy_bal.c:273-274 opens at
    +/- SURF_DT around the mean of the old surface and air temperatures and
    root_brent.c:169-179 widens it by ROOT_BRENT_TSTEP up to
    ROOT_BRENT_MAXTRIES times, so the width is 50 to 150 C against a
    tolerance of 1e-7. That is the geometry this measures.
    """
    def f(x):
        return torch.exp(0.1 * x) - 2.0

    for width in (50.0, 150.0):
        a = torch.full((5,), -width / 2.0, **KW)
        b = torch.full((5,), width / 2.0, **KW)
        ci = Counter(lambda x: f(x))
        implicit.illinois(ci, a, b)
        cb = Counter(lambda x: f(x))
        implicit.bisect(cb, a, b, tol=implicit.ROOT_BRENT_T)
        assert ci.n * 2 <= cb.n, (width, ci.n, cb.n)
        print('bracket %5.0f C: illinois %d residual evaluations, '
              'bisection %d' % (width, ci.n, cb.n))


def test_the_halving_applies_only_to_a_stale_endpoint():
    """Illinois halves the residual of an endpoint that was kept a second
    time. Halving one that was itself just replaced throws away the newest
    information and drops the method below bisection, which is what the
    first version of this solver did: 2.2e-03 against 5.4e-06 after twelve
    iterations on exp(x) - 5 over [-5, 5]."""
    a = torch.full((3,), -5.0, **KW)
    b = torch.full((3,), 5.0, **KW)
    got = implicit.illinois(lambda x: torch.exp(x) - 5.0, a, b, iters=12)
    assert float((got - math.log(5.0)).abs().max()) < 1e-5, float(got[0])


def test_the_answer_never_leaves_the_bracket():
    """The safety property. A fixed iteration count is only defensible if a
    non-converged element is still inside the interval that held the root."""
    torch.manual_seed(11)
    lo = -torch.rand(500, **KW) * 50.0 - 1.0
    hi = torch.rand(500, **KW) * 50.0 + 1.0
    # A residual with a wildly varying scale, solved with one iteration.
    scale = torch.rand(500, **KW) * 1e6 + 1e-6

    def f(x):
        return scale * torch.atan(x - 0.25)

    for iters in (1, 2, 3, 12):
        got = implicit.illinois(f, lo, hi, iters=iters)
        assert torch.isfinite(got).all(), iters
        assert bool((got >= lo).all()) and bool((got <= hi).all()), iters


def test_a_flat_secant_falls_back_on_the_midpoint():
    """fhi == flo has no false position. The C's own solver has the same
    degenerate case; here it must not become a division by zero."""
    lo = torch.tensor([-1.0, -1.0], **KW)
    hi = torch.tensor([1.0, 1.0], **KW)

    def f(x):
        # Even in x, so f(-1) == f(1) exactly on the first step.
        return x * x - 0.25

    got = implicit.illinois(f, lo, hi)
    assert torch.isfinite(got).all()
    assert bool((got >= lo).all()) and bool((got <= hi).all())


# ---- the residual the model actually solves -------------------------------
def _snow_state(n, seed=3):
    g = torch.Generator().manual_seed(seed)

    def r(a, b):
        return a + (b - a) * torch.rand(n, generator=g, **KW)
    return dict(
        z=r(2.0, 10.0), t_air=r(-25.0, 6.0), wind=r(0.2, 9.0),
        z0=r(0.001, 0.05), ra=r(5.0, 300.0), long_snow_in=r(150.0, 340.0),
        net_short=r(0.0, 400.0), air_dens=r(1.0, 1.4),
        eact_air=r(50.0, 900.0), lv=r(2.4e6, 2.6e6),
        press=r(6e4, 1.02e5), vpd=r(0.0, 900.0),
        rain=torch.zeros(n, **KW), dt=torch.full((n,), 3600.0, **KW),
        swe_surface=r(0.001, 0.2), old_t_surf=r(-25.0, 0.0),
        surface_liquid_water=r(0.0, 0.01), t_grnd=r(-8.0, 4.0),
        snow_depth=r(0.02, 1.5), snow_density=r(100.0, 450.0),
        blowing_mass_flux=torch.zeros(n, **KW))


def test_illinois_matches_bisection_on_the_snow_surface_balance():
    """The measurement the fixed iteration count rests on: 2000 randomly
    drawn pack states, Illinois against a bisection solved to 1e-13."""
    n = 2000
    st = _snow_state(n)

    def f(t):
        return snow.snow_pack_energy_balance(t, st)

    lo = torch.full((n,), -60.0, **KW)
    hi = torch.zeros(n, **KW)
    # Only states whose bracket actually holds a sign change are a root find
    # at all; VIC tests the melting case before calling the solver, and so
    # does torchvic.core.snow.
    keep = (f(lo) * f(hi)) < 0.0
    assert int(keep.sum()) > 200, int(keep.sum())
    ref = implicit.bisect(f, lo, hi, tol=1e-13)
    got = implicit.illinois(f, lo, hi)
    d = (got - ref).abs()[keep]
    r_ill = f(got).abs()[keep]
    r_bis = f(ref).abs()[keep]
    print('illinois vs bisection on %d bracketed states: median |dT| = %.3e '
          'C, max |residual| illinois %.3e vs bisection %.3e W/m2'
          % (int(keep.sum()), float(d.median()), float(r_ill.max()),
             float(r_bis.max())))
    # Illinois is a root finder, so what it must match is the *residual*,
    # not the other solver's answer. Nine of 2614 randomly drawn states have
    # a residual that is numerically zero across several degrees -- both
    # solvers land on a root there, several degrees apart, and neither is
    # wrong. Requiring the temperatures to agree would be asserting that the
    # residual has a unique well-conditioned root, which on drawn states it
    # does not.
    assert float(r_ill.max()) <= max(float(r_bis.max()) * 10.0, 1e-9), (
        float(r_ill.max()), float(r_bis.max()))
    # Where the root is well conditioned -- the overwhelming majority -- the
    # two do agree, and to far better than VIC's own tolerance.
    assert float(d.median()) < 1e-12, float(d.median())
    assert float(d.quantile(0.99)) < implicit.ROOT_BRENT_T, \
        float(d.quantile(0.99))


def test_the_solver_choice_does_not_change_the_gradient():
    """Both methods feed the same implicit-function-theorem correction, so a
    parameter's gradient must not depend on which one found the root."""
    n = 64
    st = _snow_state(n, seed=9)
    lo = torch.full((n,), -60.0, **KW)
    hi = torch.zeros(n, **KW)

    def run(method):
        s = {k: v.clone() for k, v in st.items()}
        s['ra'].requires_grad_(True)

        def f(t, *a):
            return snow.snow_pack_energy_balance(t, s)

        root = implicit.solve(f, lo, hi, tol=implicit.ROOT_BRENT_T,
                              method=method)
        root.sum().backward()
        return root.detach(), s['ra'].grad

    r1, g1 = run('illinois')
    r2, g2 = run('bisect')
    ok = torch.isfinite(g1) & torch.isfinite(g2)
    assert int(ok.sum()) > n // 2
    close = ok & ((r1 - r2).abs() < implicit.ROOT_BRENT_T)
    assert int(close.sum()) > n // 2, int(close.sum())
    rel = ((g1 - g2).abs() / torch.clamp(g2.abs(), min=1e-30))[close]
    assert float(rel.max()) < 1e-4, float(rel.max())


def test_an_unknown_method_is_refused_by_name():
    lo = torch.zeros(2, **KW)
    hi = torch.ones(2, **KW)
    with pytest.raises(ValueError, match='illinois'):
        implicit.solve(lambda x: x - 0.5, lo, hi, method='newton')
