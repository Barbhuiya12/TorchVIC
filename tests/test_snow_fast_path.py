"""Skipping snow_step on snow-free steps changes nothing.

basin.simulate skips the snow routine on a step where no tile holds ice and
no snow falls, and stands a zero-valued copy of its outputs in: everything
it returns is masked away then (state zeroed where ~has_snow, the rain goes
to the soil, sublimation and melt dropped, energy terms times a cover of
0). On India basin 37 in daily water-balance mode that step was 52 of 82 ms.
The claim held here is bit-identity, outputs and gradient, with the skip
on and off -- in a tropical case, where it engages, and in a Walnut Gulch
winter, where snow is present and it must not.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_progress import GLOBAL as WG_GLOBAL, _case, _params  # noqa: E402

from torchvic.model import basin  # noqa: E402

import _paths  # noqa: E402
B37 = _paths.B37_GLOBAL


@pytest.fixture(autouse=True)
def _on_cpu_and_restore():
    was = torch.empty(0).device
    torch.set_default_device('cpu')
    flag = basin.SNOW_FREE_FAST_PATH
    yield
    basin.SNOW_FREE_FAST_PATH = flag
    torch.set_default_device(was)


def _both(run):
    basin.SNOW_FREE_FAST_PATH = True
    a, ga = run()
    basin.SNOW_FREE_FAST_PATH = False
    b, gb = run()
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k
    for x, y in zip(ga, gb):
        assert torch.equal(x, y)


@pytest.mark.skipif(not os.path.exists(B37),
                    reason='India_basin37_Nandgaon not built')
def test_tropical_daily_water_balance_is_bit_identical():
    from torchvic import case
    c = case.load(B37, device='cpu', steps=40, first_step=851, daily=True,
                  no_energy=True, log=lambda *a: None)

    def run():
        p = c.params()
        p['pcf'] = p['pcf'].clone().requires_grad_(True)
        o = basin.simulate(**c.kwargs(p=p))
        return o, torch.autograd.grad(o['et'].sum() + o['runoff'].sum(),
                                      [p['pcf']])
    _both(run)


@pytest.mark.skipif(not os.path.exists(WG_GLOBAL),
                    reason='VIC sample data not present')
def test_a_snowy_winter_is_bit_identical():
    kw = _case(days=3)

    def run():
        p = _params(pcf=1.1, grad=True)
        o = basin.simulate(p=p, **kw)
        return o, torch.autograd.grad(o['runoff'].sum() + o['et'].sum()
                                      + o['swe'].sum(), [p['pcf']])
    _both(run)
