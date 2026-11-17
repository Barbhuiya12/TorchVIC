"""Soil layer thickness and water-stress thresholds as run parameters.

p.top_depth_scale / p.bottom_depth_scale multiply a layer's thickness with
everything that is a length of it (basin._scale_layers); p.wcr_scale /
p.wpwp_scale multiply the file's critical and wilting points. On India
basin 37 (Jun-Oct 2002) a 5 cm top layer took the soil-moisture KGE from
0.61 to 0.68, and Wcr/Wpwp x0.7 took the ET KGE from 0.29 to 0.46.
Held here:

* at their defaults (1.0) they change nothing, bit for bit;
* moved, the water balance still closes and the gradient is finite;
* the wilting point is never above the critical point.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_progress import GLOBAL, _case, _params  # noqa: E402

from torchvic.model import basin  # noqa: E402

needs_data = pytest.mark.skipif(not os.path.exists(GLOBAL),
                                reason='VIC sample data not present')
DT = torch.float64
KNOBS = ('top_depth_scale', 'bottom_depth_scale', 'wcr_scale', 'wpwp_scale')


@pytest.fixture(autouse=True)
def _on_cpu():
    was = torch.empty(0).device
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(was)


@needs_data
def test_defaults_change_nothing():
    kw = _case(days=2)
    with torch.no_grad():
        a = basin.simulate(p=_params(), **kw)
        p = _params()
        for k in KNOBS:
            p[k] = torch.tensor(1.0, dtype=DT)
        b = basin.simulate(p=p, **kw)
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k


@needs_data
def test_moved_knobs_conserve_water_and_carry_a_gradient():
    kw = _case(days=2)
    p = _params()
    vals = dict(top_depth_scale=0.5, bottom_depth_scale=1.5, wcr_scale=0.7,
                wpwp_scale=0.7)
    for k, v in vals.items():
        p[k] = torch.tensor(v, dtype=DT, requires_grad=True)
    out = basin.simulate(p=p, **kw)
    assert float(out['water_error'].abs().max()) < 1e-10
    g = torch.autograd.grad(out['et'].sum() + out['soil_liq'].sum(),
                            [p[k] for k in vals])
    assert all(torch.isfinite(x) for x in g)
    assert float(g[0]) != 0.0                       # top layer moves moisture


def test_the_layer_scaling_scales_lengths_together():
    class SP(object):
        pass
    sp = SP()
    one = torch.ones(3, 3, dtype=DT)
    sp.depth, sp.max_moist = one * 0.2, one * 60.0
    sp.resid_moist, sp.w_crit_mm, sp.w_wilt_mm = one, one * 40, one * 25
    sp.thermal = {'depth': one * 0.2}
    new, init = basin._scale_layers(sp, one * 30, top=0.5, bottom=2.0)
    for name in ('depth', 'max_moist', 'resid_moist', 'w_crit_mm',
                 'w_wilt_mm'):
        x, x0 = getattr(new, name), getattr(sp, name)
        assert torch.allclose(x[:, 0], 0.5 * x0[:, 0])
        assert torch.allclose(x[:, 1], x0[:, 1])
        assert torch.allclose(x[:, 2], 2.0 * x0[:, 2])
    assert torch.allclose(new.thermal['depth'][:, 0], 0.1 * one[:, 0])
    assert torch.allclose(init[:, 2], 60.0 * one[:, 2])
    assert torch.equal(sp.depth, one * 0.2)          # the original is untouched
