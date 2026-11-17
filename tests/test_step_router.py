"""Stepping router == whole-series routing; ledger; withdrawals/injections."""
import numpy as np
import pytest
import torch

from torchvic.model import river_network as rn
from test_river_network import make_net, STEP, DT

DEVS = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
CELL_UH = [0.3, 0.5, 0.2]


def build(device, sites=(), subcell=(), n_canal=0, nt=40, seed=0):
    net, cells = make_net()
    gauges = np.unique([int(net.order[-1]), int(net.order[-40]),
                        int(net.order[len(net.order) // 2])])
    gen = torch.Generator().manual_seed(seed)
    runoff = (torch.rand(nt, net.n, generator=gen, dtype=DT) * 5).to(device)
    area = (torch.rand(net.n, generator=gen, dtype=DT) * 1e7 + 2e7).to(device)
    cuh = torch.tensor(CELL_UH, dtype=DT, device=device)
    r, info = rn.build_step_router(net, gauges, area, STEP, cell_uh=cuh,
                                   subcell=subcell, n_canal=n_canal,
                                   site_cells=sites, device=device)
    return net, gauges, runoff, area, cuh, r, info


@pytest.mark.parametrize('device', DEVS)
def test_step_equals_whole_series(device):
    sub = [(np.array([3, 4, 5]), np.array([1.0, 0.5, 0.25]))]
    net, gauges, runoff, area, cuh, r, info = build(
        device, sites=[int(net_c) for net_c in (7, 60)], subcell=sub,
        n_canal=1)
    pairs = info['pairs']
    with torch.device(device):
        uh = rn.pair_unit_hydrographs(
            torch.as_tensor(pairs.distance, device=device), 1.5, 800.0, cuh,
            STEP)
    want = rn.route_network(runoff, pairs, uh, area, STEP)
    got = torch.stack([r.step(runoff[t])[0] for t in range(runoff.shape[0])])
    n = pairs.n_gauge
    scale = want.abs().max(0).values.clamp(min=1e-300)
    assert ((got[:, :n] - want).abs().max(0).values / scale).max() < 1e-12
    assert (got[:, n:] == 0).all()
    assert r.kind[-1] == rn.KIND_CANAL and r.kind[n - 1] == rn.KIND_SUBCELL


@pytest.mark.parametrize('device', DEVS)
def test_ledger_closes_and_drains(device):
    net, gauges, runoff, area, cuh, r, info = build(device, nt=30)
    for t in range(30):
        r.step(runoff[t])
    assert r.balance().abs().max() < 1e-9 * r.in_runoff.max()
    for t in range(r.L + 2):                      # drain: no more runoff
        r.step(torch.zeros(net.n, dtype=DT, device=device))
    assert r.ring.abs().max() < 1e-12 * r.in_runoff.max()
    assert ((r.out - r.in_runoff).abs() / r.in_runoff).max() < 1e-12


@pytest.mark.parametrize('device', DEVS)
def test_withdraw_inject(device):
    net, gauges, runoff, area, cuh, r, info = build(device, sites=[7, 60],
                                                    nt=30)
    ref = build(device, sites=[7, 60], nt=30)[5]
    sn = r.site_node
    big = torch.full((2,), 1e30, dtype=DT, device=device)
    want_del = []
    for t in range(30):
        ref.begin(runoff[t])
        av = ref.available()
        ref.end()
        q, got = r.step(runoff[t], withdraw=big * (t % 3 == 0))
        if t % 3 == 0:
            torch.testing.assert_close(got, av, rtol=0, atol=1e-6 * float(av.max()))
        want_del.append(got)
    # delivered is bounded by availability, nothing negative; ledger closes
    assert r.withdrawn.min() >= 0
    assert r.balance().abs().max() < 1e-9 * r.in_runoff.max()
    # inject the withdrawn volume back: total out matches the undisturbed run
    r.reset()
    ref.reset()
    for t in range(30):
        r.step(runoff[t], inject=torch.full((2,), 1e5, dtype=DT, device=device))
        ref.step(runoff[t])
    assert r.balance().abs().max() < 1e-9 * r.in_runoff.max()
    drain = torch.zeros(net.n, dtype=DT, device=device)
    for t in range(r.L + 2):
        r.step(drain)
        ref.step(drain)
    extra = (r.out - ref.out)
    # the injection at site 60's cell reaches the node of that cell fully
    assert extra[sn[1]] >= 30 * 1e5 * (1 - 1e-9)
    assert abs(float(r.injected.sum()) - 2 * 30 * 1e5) < 1e-3


def test_withdraw_limited_and_downstream_reduced():
    net, gauges, runoff, area, cuh, r, info = build('cpu', sites=[7], nt=20)
    ref = build('cpu', sites=[7], nt=20)[5]
    for t in range(20):
        _, got = r.step(runoff[t], withdraw=torch.full((1,), 1e30, dtype=DT))
        ref.step(runoff[t])
    # everything passing the site was taken: its own node reads zero
    assert r.out[info['site_node'][0]] < 1e-12 * ref.out[info['site_node'][0]] + 1e-6
    # nodes downstream of it lose exactly the withdrawn volume (after draining)
    z = torch.zeros(net.n, dtype=DT)
    for t in range(r.L + 2):
        r.step(z); ref.step(z)
    d = ref.out - r.out
    assert d.min() >= -1e-6
    assert float(d.max()) <= float(r.withdrawn.sum()) * (1 + 1e-9)
    assert abs(float(d[info['site_node'][0]]) - float(r.withdrawn[0])) < 1e-3 * 1e-3 * float(r.withdrawn[0])


def test_save_load_and_canal_inject():
    net, gauges, runoff, area, cuh, r, info = build('cpu', n_canal=1, nt=20)
    r2 = build('cpu', n_canal=1, nt=20)[5]
    n = r.n_node
    ni = torch.zeros(n, dtype=DT)
    ni[-1] = 5e6
    for t in range(10):
        r.step(runoff[t], node_inject=ni)
    r2.load_state_dict(r.state_dict())
    a = [r.step(runoff[t])[0] for t in range(10, 20)]
    b = [r2.step(runoff[t])[0] for t in range(10, 20)]
    assert all((x == y).all() for x, y in zip(a, b))
    assert abs(float(r.out[-1]) - 5e6 * 10) < 1e-6 * 5e7


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_graph_capture():
    net, gauges, runoff, area, cuh, r, info = build('cuda', sites=[7], nt=12)
    ref = build('cpu', sites=[7], nt=12)[5]
    x = runoff[0].clone()
    w = torch.zeros(1, dtype=DT, device='cuda')
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        r.step(x, withdraw=w)
    torch.cuda.current_stream().wait_stream(s)
    r.reset()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        q, _ = r.step(x, withdraw=w)
    out = []
    for t in range(12):
        x.copy_(runoff[t])
        g.replay()
        out.append(q.cpu().clone())
    for t in range(12):
        want, _ = ref.step(runoff[t].cpu(), withdraw=torch.zeros(1, dtype=DT))
        torch.testing.assert_close(out[t], want, rtol=1e-12, atol=1e-9)


def _chain(net, n):
    """n cells along one river, upstream to downstream."""
    c = int(np.argmax(net.dist_to_mouth))
    path = []
    while c >= 0:
        path.append(c)
        c = int(net.down[c])
    step = max(len(path) // (n + 1), 1)
    return path[step::step][:n]


def _min_slot(r):
    return float(r.ring.min())


@pytest.mark.parametrize('device', DEVS)
def test_chained_withdrawals_no_negative_and_levels(device):
    net, _ = make_net()
    chain = _chain(net, 5)
    assert list(rn.site_levels(net, chain)) == [0, 1, 2, 3, 4]
    net, gauges, runoff, area, cuh, r, info = build(device, sites=chain,
                                                    nt=40)
    big = torch.full((len(chain),), 1e30, dtype=DT, device=device)
    tot = 0.0
    for t in range(40):
        q, got = r.step(runoff[t], withdraw=big)
        assert got.min() >= 0
        assert q.min() >= -1e-9 * float(q.abs().max())
        assert _min_slot(r) >= -1e-9 * float(r.ring.abs().max())
        tot += float(got.sum())
    assert tot > 0
    assert tot <= float(r.in_runoff.max()) * (1 + 1e-9) * len(chain)
    assert r.balance().abs().max() < 1e-9 * r.in_runoff.max()
    for t in range(r.L + 2):
        q, _ = r.step(torch.zeros(net.n, dtype=DT, device=device))
        assert q.min() >= -1e-9 * float(r.out.max()) / r.L


@pytest.mark.parametrize('device', DEVS)
def test_injection_then_withdrawal_downstream(device):
    net, _ = make_net()
    chain = _chain(net, 3)
    net, gauges, runoff, area, cuh, r, info = build(device, sites=chain)
    z = torch.zeros(net.n, dtype=DT, device=device)
    V = 5e6
    inj = torch.tensor([V, 0.0, 0.0], dtype=DT, device=device)
    big = torch.full((3,), 1e30, dtype=DT, device=device)
    delivered = torch.zeros(3, dtype=DT, device=device)
    for t in range(r.L + 3):
        _, got = r.step(z, withdraw=big, inject=inj if t == 0 else None)
        delivered += got
        assert _min_slot(r) >= -1e-9 * V / STEP
    # the upstream site's own step: its release is withdrawn again at once
    # (level 0 withdraws before it injects); downstream sites then catch it
    assert float(delivered[1] + delivered[2]) > 0.2 * V
    assert float(delivered.sum()) <= V * (1 + 1e-9)
    assert r.balance().abs().max() < 1e-9 * V


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_graph_capture_with_levels():
    net, _ = make_net()
    chain = _chain(net, 4)
    net, gauges, runoff, area, cuh, r, info = build('cuda', sites=chain,
                                                    nt=10)
    ref = build('cpu', sites=chain, nt=10)[5]
    x = runoff[0].clone()
    w = torch.full((4,), 3e5, dtype=DT, device='cuda')
    j = torch.full((4,), 1e5, dtype=DT, device='cuda')
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        r.step(x, withdraw=w, inject=j)
    torch.cuda.current_stream().wait_stream(s)
    r.reset()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        q, got = r.step(x, withdraw=w, inject=j)
    for t in range(10):
        x.copy_(runoff[t])
        g.replay()
        want, wg = ref.step(runoff[t].cpu(), withdraw=w.cpu(), inject=j.cpu())
        torch.testing.assert_close(q.cpu(), want, rtol=1e-10, atol=1e-8)
        torch.testing.assert_close(got.cpu(), wg, rtol=1e-10, atol=1e-6)
