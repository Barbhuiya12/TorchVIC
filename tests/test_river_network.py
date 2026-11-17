"""Network routing to nested gauges: GPU == CPU == a loop over pairs."""
import numpy as np
import pytest
import torch

from torchvic.model import river_network as rn

DT = torch.float64
STEP = 86400.0


def make_net(ny=14, nx=17, seed=3):
    g = np.random.default_rng(seed)
    elev = g.random((ny, nx)) * 100 + np.linspace(0, 400, nx)[None, :]
    mask = np.ones((ny, nx), bool)
    mask[0, :4] = False
    mask[-1, -3:] = False
    lat = 20 + np.arange(ny) * 0.0625
    lon = 70 + np.arange(nx) * 0.0625
    net, cells = rn.derive_d8_from_elevation(elev, mask, lat, lon)
    return net, cells


def setup(device, nt=40):
    net, cells = make_net()
    gauges = np.array([int(net.order[-1]), int(net.order[-40]),
                       int(net.order[len(net.order) // 2])])
    gauges = np.unique(gauges)
    pairs = rn.build_pairs(net, gauges)
    gen = torch.Generator(device='cpu').manual_seed(0)
    with torch.device('cpu'):
        runoff = (torch.rand(nt, net.n, generator=gen, dtype=DT) * 5).to(device)
        area = (torch.rand(net.n, generator=gen, dtype=DT) * 1e7
                + 2e7).to(device)
    cell_uh = torch.tensor([0.3, 0.5, 0.2], dtype=DT, device=device)
    with torch.device(device):
        uh = rn.pair_unit_hydrographs(
            torch.as_tensor(pairs.distance, device=device), 1.5, 800.0,
            cell_uh, STEP)
    return net, pairs, runoff, area, uh


def test_network_is_a_forest_and_gauge_pairs_nest():
    net, _ = make_net()
    assert len(net.order) == net.n
    gauges = np.array([int(net.order[-1]), int(net.order[-30])])
    p = rn.build_pairs(net, np.unique(gauges))
    # a gauge's own cell is a pair at distance 0
    for k, gc in enumerate(np.unique(gauges)):
        m = (p.gauge == k) & (p.cell == gc)
        assert m.sum() == 1 and p.distance[m][0] == 0.0


def test_unit_hydrographs_sum_to_one():
    _, _, _, _, uh = setup('cpu')
    assert torch.allclose(uh.sum(0), torch.ones(uh.shape[1], dtype=DT, device=uh.device),
                          atol=1e-12)


def test_matches_loop_over_pairs():
    net, pairs, runoff, area, uh = setup('cpu')
    with torch.device('cpu'):
        q = rn.route_network(runoff, pairs, uh, area, STEP, chunk=7)
    ref = rn.route_network_reference(runoff, pairs, uh, area, STEP)
    err = np.abs(q.numpy() - ref).max() / np.abs(ref).max()
    assert err < 1e-12, err


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')
def test_gpu_equals_cpu():
    net, pairs, runoff, area, uh = setup('cpu')
    with torch.device('cpu'):
        qc = rn.route_network(runoff, pairs, uh, area, STEP, chunk=11)
    _, _, r2, a2, u2 = setup('cuda')
    qg = rn.route_network(r2, pairs, u2, a2, STEP, chunk=11).cpu()
    err = (qc - qg).abs().max() / qc.abs().max()
    assert float(err) < 1e-12, float(err)


def test_mass_is_conserved_for_a_pulse():
    net, pairs, runoff, area, uh = setup('cpu', nt=400)
    runoff[:] = 0
    runoff[0] = 3.0
    with torch.device('cpu'):
        q = rn.route_network(runoff, pairs, uh, area, STEP)
    got = (q * STEP).sum(0).numpy()
    want = np.zeros(pairs.n_gauge)
    for c, g in zip(pairs.cell, pairs.gauge):
        want[g] += 3.0 * float(area[c]) / 1000.0
    assert np.allclose(got, want, rtol=1e-9)


def _subcell_case(device='cpu', nt=30):
    net, pairs, runoff, area, uh = setup(device, nt)
    # two sub-cell gauges: one over 3 cells (fractions .25 .6 1.0), one over a single partial cell
    sub = [(np.array([5, 17, 40]), np.array([0.25, 0.6, 1.0])),
           (np.array([60]), np.array([0.4]))]
    p2 = rn.add_subcell_gauges(pairs, sub)
    cell_uh = torch.tensor([0.3, 0.5, 0.2], dtype=DT, device=device)
    uh2 = rn.pair_unit_hydrographs(torch.as_tensor(p2.distance, device=device),
                                   1.5, 800.0, cell_uh, STEP)
    return pairs, p2, runoff, area, uh2, sub, cell_uh


def test_subcell_gauge_is_the_area_weighted_sum_of_cell_runoff():
    pairs, p2, runoff, area, uh2, sub, cell_uh = _subcell_case()
    assert p2.n_gauge == pairs.n_gauge + 2
    q = rn.route_network(runoff, p2, uh2, area, STEP).numpy()
    r, a, u = runoff.numpy(), area.numpy(), cell_uh.numpy()
    for k, (c, f) in enumerate(sub):
        want = 0.0
        for ci, fi in zip(c, f):             # area x fraction x runoff, cell unit hydrograph only
            want = want + np.convolve(r[:, ci], u)[:r.shape[0]] * a[ci] * fi / (1000.0 * STEP)
        assert np.abs(q[:, pairs.n_gauge + k] - want).max() <= 1e-12 * np.abs(want).max()
    # channel gauges are untouched
    q0 = rn.route_network(runoff, pairs, rn.pair_unit_hydrographs(
        torch.as_tensor(pairs.distance), 1.5, 800.0, cell_uh, STEP), area, STEP).numpy()
    assert np.abs(q[:, :pairs.n_gauge] - q0).max() <= 1e-12 * np.abs(q0).max()


def test_subcell_uniform_runoff_gives_polygon_area_times_runoff():
    pairs, p2, runoff, area, uh2, sub, cell_uh = _subcell_case()
    runoff = torch.full_like(runoff, 2.0)            # mm per step everywhere
    q = rn.route_network(runoff, p2, uh2, area, STEP).numpy()
    for k, (c, f) in enumerate(sub):
        poly_area = float((area.numpy()[c] * f).sum())
        assert abs(q[-1, pairs.n_gauge + k] - poly_area * 2.0 / (1000.0 * STEP)) < 1e-12 * poly_area


def test_subcell_reference_matches_fast_path():
    pairs, p2, runoff, area, uh2, sub, cell_uh = _subcell_case()
    a = rn.route_network(runoff, p2, uh2, area, STEP).numpy()
    b = rn.route_network_reference(runoff, p2, uh2, area, STEP)
    assert np.abs(a - b).max() <= 1e-12 * np.abs(b).max()
