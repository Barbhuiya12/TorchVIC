"""Lohmann routing along each cell's flow length to the gauge.

`routing.lohmann_unit_hydrographs` turns each cell's flow length into its
own unit hydrograph to the gauge (VIC's routing scheme, in torch). Held
here:

* each unit hydrograph sums to 1, so routing conserves water;
* the channel response has the physics it claims: mean travel time x / v
  (the inverse-Gaussian mean), later and flatter for farther cells, all in
  the first step at the outlet itself;
* the routing parameters carry finite gradients, and the velocity gradient
  agrees with a finite difference.
"""
import os

import pytest
import torch

import _paths
from torchvic.model import basin, routing

DT = torch.float64
DEV = torch.device('cpu')
HOUR = 3600.0


@pytest.fixture(autouse=True)
def _on_cpu():
    """Other modules in this directory set the process default device to
    CUDA at import; these tests build small tensors without naming one."""
    was = torch.empty(0).device
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(was)


def test_unit_hydrographs_sum_to_one_and_the_near_cell_is_faster():
    d = torch.tensor([0.0, 5e3, 40e3, 90e3], dtype=DT)
    cell = basin.gamma_unit_hydrograph(60, 2.5, 2.0, DT, DEV)
    uh = routing.lohmann_unit_hydrographs(d, 1.5, 800.0, cell, dt=HOUR)
    assert torch.allclose(uh.sum(0), torch.ones(4, dtype=DT), atol=1e-12)
    t = torch.arange(uh.shape[0], dtype=DT)
    mean = (uh * t[:, None]).sum(0)
    assert (mean[1:] > mean[:-1]).all()
    peak = uh.max(0).values
    assert (peak[1:] < peak[:-1]).all()                  # farther is flatter


def test_channel_travel_time_is_x_over_v():
    """The inverse-Gaussian mean is x / v. Step k covers [k, k+1) dt, so
    the mean step index is x / v / dt - 1/2."""
    for x, v in ((50e3, 1.5), (90e3, 0.8), (20e3, 3.0)):
        h = routing.lohmann_river_irf(torch.tensor([x], dtype=DT), v, 800.0,
                                      400, HOUR)
        t = torch.arange(400, dtype=DT)
        got = float((h[:, 0] * t).sum())
        want = x / v / HOUR - 0.5
        assert abs(got - want) < 0.05 * want + 0.1, (x, v, got, want)


def test_the_outlet_cell_has_no_channel_delay_and_a_finite_gradient():
    d = torch.tensor([0.0, 30e3], dtype=DT)
    v = torch.tensor(1.5, dtype=DT, requires_grad=True)
    D = torch.tensor(800.0, dtype=DT, requires_grad=True)
    h = routing.lohmann_river_irf(d, v, D, 100, HOUR)
    assert float(h[0, 0]) == 1.0 and float(h[1:, 0].abs().sum()) == 0.0
    gv, gD = torch.autograd.grad((h * torch.arange(100.0, dtype=DT)[:, None]
                                  ).sum(), [v, D])
    assert torch.isfinite(gv) and torch.isfinite(gD) and float(gv) < 0


def test_routing_with_per_cell_hydrographs_conserves_water():
    d = torch.tensor([0.0, 25e3, 60e3], dtype=DT)
    cell = basin.gamma_unit_hydrograph(60, 2.5, 2.0, DT, DEV)
    uh = routing.lohmann_unit_hydrographs(d, 1.2, 800.0, cell, dt=HOUR)
    nt = 48 + uh.shape[0]
    ro = torch.zeros(nt, 3, dtype=DT)
    ro[:48] = torch.rand(48, 3, dtype=DT, device=DEV)
    q = routing.route(ro, uh, torch.zeros(3, dtype=torch.long), 1)
    assert abs(float(q.sum() - ro.sum())) < 1e-10


def test_velocity_gradient_matches_a_finite_difference():
    d = torch.tensor([10e3, 45e3, 80e3], dtype=DT)
    cell = basin.gamma_unit_hydrograph(60, 2.5, 2.0, DT, DEV)
    ro = torch.rand(72, 3, dtype=DT, device=DEV)

    def peak_loss(v):
        uh = routing.lohmann_unit_hydrographs(d, v, 800.0, cell, dt=HOUR)
        q = routing.route(ro, uh[:200], torch.zeros(3, dtype=torch.long), 1)
        return (q ** 2).sum()
    v = torch.tensor(1.5, dtype=DT, requires_grad=True)
    (g,) = torch.autograd.grad(peak_loss(v), [v])
    h = 1e-5
    with torch.no_grad():
        fd = (peak_loss(torch.tensor(1.5 + h, dtype=DT))
              - peak_loss(torch.tensor(1.5 - h, dtype=DT))) / (2 * h)
    assert abs(float(g - fd)) < 1e-6 * max(1.0, abs(float(fd))), (g, fd)


GLOBAL = _paths.B37_GLOBAL


@pytest.mark.skipif(not os.path.exists(GLOBAL),
                    reason='India_basin37_Nandgaon not built')
def test_basin37_loads_its_network_and_routes_to_the_gauge():
    from torchvic import case
    c = case.load(GLOBAL, device=DEV, steps=3, first_step=180,
                  log=lambda *a: None)
    b = c.kwargs()['batch']
    assert b.flow_distance is not None and int((b.flow_distance == 0).sum()) == 1
    p = c.params()
    p['route_velocity'] = p['route_velocity'].clone().requires_grad_(True)
    out = basin.simulate(**c.kwargs(p=p))
    q = out['discharge_mm']
    assert q.shape[1] == 1 and bool(torch.isfinite(q).all())
    (g,) = torch.autograd.grad(q.sum(), [p['route_velocity']])
    assert torch.isfinite(g) and float(g) != 0.0
