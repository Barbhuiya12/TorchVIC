"""The canopy's interception capacity is a run parameter, lai_water_factor.

VIC reads it from its parameter file as VEG_LAI_WATER_FACTOR (0.1 mm of
storage per unit LAI by default). It became calibratable because on India
basin 37 the model's canopy evaporated 0.97 mm/day of intercepted monsoon
rain against GLEAM's 0.14, and no other knob could lower it without also
lowering transpiration, which already agreed. Held here:

* the default reproduces VIC's constant exactly;
* a smaller capacity holds less, lets more through, and evaporates less --
  on a sub-daily step, where only water already held can evaporate;
* the derivative with respect to it is finite and agrees with a finite
  difference.
"""
import pytest
import torch

from torchvic.core import vegetation as veg

DT = torch.float64
HOUR = 3600.0


@pytest.fixture(autouse=True)
def _on_cpu():
    was = torch.empty(0).device
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(was)


def _inputs(n=4):
    t = lambda v: torch.full((n,), float(v), dtype=DT)
    return dict(wdew=torch.linspace(0.0, 0.4, n, dtype=DT), ppt=t(2.0),
                lai=t(3.0), rgl=t(100.0), net_short=t(300.0), tair=t(25.0),
                vpd=t(800.0), ra=t(40.0), rarc=t(2.0), rad=t(350.0),
                elevation=t(300.0))


def _run(factor, x=None):
    x = x or _inputs()
    kw = dict(dt=HOUR)
    if factor is not None:
        kw['lai_water_factor'] = factor
    return veg.canopy_interception(
        x['wdew'], x['ppt'], x['lai'], x['rgl'], x['net_short'], x['tair'],
        x['vpd'], x['ra'], x['rarc'], x['rad'], x['elevation'], **kw)


def test_the_default_is_vics_constant():
    assert veg.VEG_LAI_WATER_FACTOR == 0.1
    a, b = _run(None), _run(veg._lwf(None))
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_smaller_capacity_holds_less_and_evaporates_less():
    big = _run(0.1)
    small = _run(0.02)
    ce_b, wd_b, tf_b, _ = big
    ce_s, wd_s, tf_s, _ = small
    lai = 3.0
    assert float(wd_s.max()) <= 0.02 * lai + 1e-12
    assert float(wd_b.max()) <= 0.1 * lai + 1e-12
    assert (tf_s >= tf_b).all() and float((tf_s - tf_b).sum()) > 0
    assert (ce_s <= ce_b + 1e-15).all() and float((ce_b - ce_s).sum()) > 0
    # water is conserved either way: held + fallen = evaporated + held + through
    x = _inputs()
    for ce, wd, tf, _ in (big, small):
        assert torch.allclose(x['wdew'] + x['ppt'], ce + wd + tf, atol=1e-12)


def test_the_gradient_is_finite_and_matches_a_finite_difference():
    f = torch.tensor(0.05, dtype=DT, requires_grad=True)
    ce, wd, tf, dry = _run(f)
    loss = ce.sum() + 0.3 * wd.sum()
    (g,) = torch.autograd.grad(loss, [f])
    h = 1e-6
    with torch.no_grad():
        up = _run(torch.tensor(0.05 + h, dtype=DT))
        dn = _run(torch.tensor(0.05 - h, dtype=DT))
        fd = ((up[0].sum() + 0.3 * up[1].sum())
              - (dn[0].sum() + 0.3 * dn[1].sum())) / (2 * h)
    assert torch.isfinite(g) and float(g) != 0.0
    assert abs(float(g - fd)) < 1e-6 * max(1.0, abs(float(fd))), (g, fd)
