"""Compiling the solvers' residuals leaves the gradient unchanged.

`energy.enable_compile()` runs the surface and snow-pack residuals through
torch.compile wherever no derivative passes through them -- the bracket
search and the Illinois iterations, which run under no_grad -- and keeps
the evaluation that carries the parameter gradient, and the forward-mode
df/dT, eager. Compiling the whole residual instead gave gradients 15-53%
wrong (the dual tensor's tangent was lost) and a slower run; this holds
the design to its claim:

* parameter gradients match the eager run to 1e-12 relative,
* the outputs match to 1e-12 relative (compiled arithmetic is not bit for
  bit, which is why it is off by default),
* no implicit derivative falls back to reverse mode.
"""
import os

import pytest
import torch

from torchvic.model import basin
from torchvic.core import energy, implicit, snow

import _paths  # noqa: E402
GLOBAL = _paths.B37_GLOBAL

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.exists(GLOBAL)),
    reason='needs CUDA and India_basin37_Nandgaon')

KNOBS = ('pcf', 'rmin_scale', 'lai_scale')


def _run(c):
    p = c.params()
    for k in KNOBS:
        p[k] = p[k].clone().requires_grad_(True)
    out = basin.simulate(**c.kwargs(p=p))
    g = torch.autograd.grad(out['discharge_mm'].sum() + out['et'].sum(),
                            [p[k] for k in KNOBS])
    return out, torch.stack(g)


def test_compiled_solvers_keep_the_gradient():
    from torchvic import case
    c = case.load(GLOBAL, device='cuda', steps=1, first_step=851,
                  log=lambda *a: None)
    ref_out, ref_g = _run(c)
    before = implicit.REVERSE_FALLBACKS[0]
    try:
        energy.enable_compile(True)
        out, g = _run(c)
    finally:
        energy.enable_compile(False)
    assert not energy.COMPILE and not snow.COMPILE
    assert implicit.REVERSE_FALLBACKS[0] == before
    assert (ref_g != 0).all()
    rel = ((g - ref_g).abs() / ref_g.abs()).max()
    assert float(rel) < 1e-12, (g.tolist(), ref_g.tolist())
    for k in ('discharge_mm', 'et', 'soil_liq', 'runoff'):
        d = (out[k] - ref_out[k]).abs().max()
        # Against the physical scale, not the output's own maximum: runoff
        # is a difference of soil stores of ~100 mm, so its rounding floor
        # is ~1e-14 mm however small the runoff. A dry hour's runoff peaks
        # at 3.5e-5 mm, and 1.3e-14 of compiled rounding read against that
        # looked like 4e-10.
        scale = max(float(ref_out[k].abs().max()), 1.0)
        assert float(d) <= 1e-12 * scale, (k, float(d))


def test_compiled_column_keeps_the_daily_water_balance_gradient():
    """With compilation on, the column step is compiled for runs without a
    gradient and eager with one (energy.COLUMN_COMPILE_GRAD: the compiled
    backward returned NaN on some delta-VIC batches). Here the gradient
    equal to eager to 1e-12 and the column restored after."""
    from torchvic import case
    from torchvic.core import column
    c = case.load(GLOBAL, device='cuda', steps=20, first_step=851,
                  daily=True, no_energy=True, log=lambda *a: None)
    eager_step = column.step
    ref_out, ref_g = _run(c)
    try:
        energy.enable_compile(True)
        assert column.step is not eager_step
        out, g = _run(c)
    finally:
        energy.enable_compile(False)
    assert column.step is eager_step
    rel = ((g - ref_g).abs() / ref_g.abs()).max()
    assert float(rel) < 1e-12, (g.tolist(), ref_g.tolist())
    for k in ('et', 'soil_liq', 'runoff', 'baseflow'):
        d = (out[k] - ref_out[k]).abs().max()
        scale = max(float(ref_out[k].abs().max()), 1.0)     # see above
        assert float(d) <= 1e-12 * scale, (k, float(d))
