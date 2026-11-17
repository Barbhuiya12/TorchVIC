"""The implicit derivative costs one local evaluation, not the whole history.

`implicit.root_with_grad` needs df/dT at each solved root: the surface
energy balance, the canopy, the pack, the soil nodes, once per solve, every
time step. It used to take it with `torch.autograd.grad`, and reverse mode
walks everything reachable from its output before it runs -- which, in a
calibration, is every earlier time step, reached through `args` and through
the tensors the residuals' closures capture. The forward pass got slower
every day it ran: 222 ms per step on day 1, 501 on day 10 (100 cells,
cuda:0), against a flat 181 without gradients.

It is now forward mode: a tangent on T through one evaluation, whatever the
history. Measured after: 274 ms per step on day 1, 275 on day 10, against
208 without gradients in the same run -- flat. A full 20-day epoch on those
100 cells went from 725 s to 144 s (forward 129 s, backward 14 s, peak
4.23 GiB).

Three things are held here:

* no reverse-mode call happens inside the forward loop at all -- the
  structural statement, which a timing test could only approximate;
* the forward-mode derivative is the reverse-mode one: outputs and the
  parameter gradient of a gradient-carrying run agree between the two;
* the model's own solves never fall back to the slow path.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_progress import GLOBAL, _case, _params  # noqa: E402

from torchvic.model import basin  # noqa: E402
from torchvic.core import implicit

needs_data = pytest.mark.skipif(not os.path.exists(GLOBAL),
                                reason='VIC sample data not present')


#: Parameters whose gradient runs through the surface energy solve and its
#: evaporation closure -- the path the fix changed. (d/d pcf is zero on
#: these rainless Walnut Gulch days and would compare 0 with 0.)
KNOBS = ('rmin_scale', 'lai_scale')


def _run(kw):
    p = _params()
    for k in KNOBS:
        p[k] = p[k].clone().requires_grad_(True)
    out = basin.simulate(p=p, **kw)
    g = torch.autograd.grad(out['et'].sum() + out['runoff'].sum(),
                            [p[k] for k in KNOBS], allow_unused=True)
    g = torch.stack([torch.zeros((), dtype=torch.float64) if x is None
                     else x for x in g])
    return out, g


@needs_data
def test_no_reverse_mode_call_inside_the_forward_loop(monkeypatch):
    """The quadratic cost was a torch.autograd.grad per solve per step."""
    kw = _case()
    calls = []
    real = torch.autograd.grad

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(torch.autograd, 'grad', spy)
    before = implicit.REVERSE_FALLBACKS[0]
    p = _params()
    p['rmin_scale'] = p['rmin_scale'].clone().requires_grad_(True)
    out = basin.simulate(p=p, **kw)
    assert not calls, ('%d reverse-mode calls inside the forward loop; each '
                       'walks the whole history' % len(calls))
    assert implicit.REVERSE_FALLBACKS[0] == before
    # And the run still carries a gradient.
    assert out['runoff'].requires_grad


@needs_data
def test_forward_mode_derivative_is_the_reverse_mode_one(monkeypatch):
    """Same partial derivative by either mode, so the same model."""
    kw = _case()
    fwd_out, fwd_g = _run(kw)

    def reverse_only(f, root, args):
        t = root.detach().requires_grad_(True)
        with torch.enable_grad():
            ft = f(t, *implicit._detach_tree(args))
            d, = torch.autograd.grad(ft.sum(), t)
        return d
    monkeypatch.setattr(implicit, '_local_dfdt', reverse_only)
    rev_out, rev_g = _run(kw)

    for k in ('runoff', 'baseflow', 'et', 'soil_liq', 'swe'):
        if k not in fwd_out:
            continue
        x, y = fwd_out[k].detach(), rev_out[k].detach()
        scale = max(float(y.abs().max()), 1e-30)
        assert float((x - y).abs().max()) <= 1e-12 * scale, k
    assert torch.isfinite(fwd_g).all()
    assert (rev_g != 0).all(), 'a zero gradient compares nothing: %r' % rev_g
    rel = ((fwd_g - rev_g).abs() / rev_g.abs()).max()
    assert float(rel) <= 1e-12, (
        'forward %r vs reverse %r' % (fwd_g.tolist(), rev_g.tolist()))


def test_the_derivative_ignores_history_held_in_a_closure():
    """A residual whose closure holds a long live chain -- the evaporation
    callbacks' shape -- still gets the right df/dT, from forward mode."""
    a = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    h = a
    for _ in range(500):          # a long history, live
        h = h * 1.0 + 0.0
    f = lambda t, c: h * t ** 3 - c      # noqa: E731  closure over h
    c = torch.tensor([16.0, 81.0], dtype=torch.float64)
    root = torch.tensor([2.0, 3.0], dtype=torch.float64)
    before = implicit.REVERSE_FALLBACKS[0]
    d = implicit._local_dfdt(f, root, (c,))
    assert implicit.REVERSE_FALLBACKS[0] == before
    assert torch.allclose(d, 3 * a.detach() * root ** 2, rtol=0, atol=0)
