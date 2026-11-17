"""Water quality / temperature engine on the stepping router."""
import numpy as np
import pytest
import torch

from torchvic.model import river_network as rn
from torchvic.model import routing
from torchvic.quality import WaterQuality, heat
from test_river_network import make_net, STEP, DT

BUILD = rn.build_step_router
if torch.cuda.is_available():
    from torchvic.model import routing_fast
    BUILD = routing_fast.build_fast_step_router      # CPU routers come back unchanged
DEVS = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _on_cpu():
    was = torch.empty(0).device
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(was)


def reach(dev, lag=3, L=8, sites=True):
    """One cell, one node, delta unit hydrograph at `lag`; area makes the
    load fraction 1."""
    uh = torch.zeros(L, 1, dtype=DT, device=dev)
    uh[lag, 0] = 1.0
    kern = torch.zeros(1, L, dtype=DT, device=dev)
    kern[0, 0] = 1.0
    kw = {}
    if sites:
        kw = dict(site_node=[0], inj=([0], np.array([0]), kern),
                  site_level=[0])
    r = routing.StepRouter(np.array([0]), np.array([0]), uh,
                           np.array([1.0]), 1, STEP, **kw)
    return r, np.array([1000.0 * STEP])


def run(wq, nt, **kw):
    return [wq.step(**{k: (v(t) if callable(v) else v) for k, v in kw.items()})
            for t in range(nt)]


@pytest.mark.parametrize('dev', DEVS)
def test_decay_single_reach_analytic(dev):
    k = 2.0e-6
    r, area = reach(dev, lag=3)
    wq = WaterQuality(r, area, {'n': k}, site_cells=[0])
    M = 5.0e5
    z1 = torch.zeros(1, dtype=DT, device=dev)
    outs = []
    for t in range(8):
        ld = {'n': torch.full((1,), M, dtype=DT, device=dev)} if t == 0 else None
        outs.append(wq.step(torch.full((1,), 2.0 if t == 0 else 0.0, dtype=DT, device=dev),
                            loads=ld))
    flux = torch.stack([o['conc']['n'] * o['q'] for o in outs]).flatten()
    want = M / STEP * np.exp(-k * 3 * STEP)
    assert abs(flux[3].item() - want) < 1e-12 * want
    assert flux[[0, 1, 2, 4, 5, 6, 7]].abs().max() == 0
    lost = wq.decayed('n').item()
    assert abs(lost - M * (1 - np.exp(-k * 3 * STEP))) < 1e-9 * M
    # decayed + delivered = loaded
    assert abs(lost + flux.sum().item() * STEP - M) < 1e-9 * M
    assert wq.ledger()['n'].abs().max() < 1e-9 * M


def make(dev, nt=60, sites=(), seed=0, cuh=(0.3, 0.5, 0.2)):
    net, cells = make_net()
    gauges = np.unique([int(net.order[-1]), int(net.order[-40]),
                        int(net.order[len(net.order) // 2])])
    gen = torch.Generator().manual_seed(seed)
    area = (torch.rand(net.n, generator=gen, dtype=DT) * 1e7 + 2e7).to(dev)
    r, info = BUILD(net, gauges, area, STEP,
                    cell_uh=torch.tensor(cuh, dtype=DT, device=dev),
                    site_cells=list(sites), device=dev)
    return net, area, r, gen


@pytest.mark.parametrize('dev', DEVS)
def test_conservative_tracer_balance_and_concentration(dev):
    net, area, r, gen = make(dev)
    wq = WaterQuality(r, area, {'tr': 0.0})
    c = 3.7                                          # g m-3 in all runoff
    nt = 50
    ro = (torch.rand(nt, net.n, generator=gen, dtype=DT) * 5 + 1).to(dev)
    last = None
    for t in range(nt):
        ld = {'tr': c * ro[t] * 1e-3 * area}         # g per step per cell
        last = wq.step(ro[t], loads=ld)
        if t > 0:
            assert (last['conc']['tr'][last['q'] > 0] - c).abs().max() < 1e-9
    z = torch.zeros(net.n, dtype=DT, device=dev)
    for t in range(r.L + 2):
        wq.step(z)
    ch = wq.ch['tr']
    assert ch.balance().abs().max() < 1e-12 * ch.in_runoff.max()
    assert ((ch.out - ch.in_runoff).abs() / ch.in_runoff).max() < 1e-12
    assert ch.lost.abs().max() < 1e-9 * ch.in_runoff.max()
    # tracer mass is c times the water volume
    assert ((ch.out - c * r.out).abs() / ch.out).max() < 1e-12


@pytest.mark.parametrize('dev', DEVS)
def test_decay_network_and_per_cell_rate(dev):
    net, area, r, gen = make(dev)
    k = (torch.rand(net.n, generator=gen, dtype=DT) * 3e-6).to(dev)
    wq = WaterQuality(r, area, {'a': k, 'b': 1e-6})
    nt = 40
    ro = (torch.rand(nt, net.n, generator=gen, dtype=DT) * 5 + 1).to(dev)
    for t in range(nt):
        ld = (ro[t] * 1e-3 * area)
        wq.step(ro[t], loads={'a': ld, 'b': ld})
    z = torch.zeros(net.n, dtype=DT, device=dev)
    for t in range(r.L + 2):
        wq.step(z)
    for n in 'ab':
        ch = wq.ch[n]
        tot = ch.in_runoff + ch.lost            # undecayed load reaching the node
        assert ch.balance().abs().max() < 1e-12 * tot.max()
        assert ((ch.out - ch.in_runoff).abs() / ch.in_runoff).max() < 1e-12
        assert (ch.lost > 0).all() and (ch.out < tot).all()


@pytest.mark.parametrize('dev', DEVS)
def test_withdrawal_removes_load_in_proportion(dev):
    r, area = reach(dev, lag=0)
    wq = WaterQuality(r, area, {'n': 1e-6}, site_cells=[0])
    M = 1e6
    q0 = 4.0                                   # m3/s -> V = q0 * STEP
    V = q0 * STEP
    take = 0.25 * V
    o = wq.step(torch.full((1,), q0, dtype=DT, device=dev),
                loads={'n': torch.full((1,), M, dtype=DT, device=dev)},
                withdraw=torch.full((1,), take, dtype=DT, device=dev))
    assert abs(o['got'][0].item() - take) < 1e-6
    assert abs(o['q'][0].item() - 0.75 * q0) < 1e-12
    assert abs(o['conc']['n'][0].item() - M / V) < 1e-12 * M / V     # same concentration
    removed = wq.ch['n'].withdrawn[0].item()
    assert abs(removed - 0.25 * M) < 1e-9 * M
    assert wq.ledger()['n'].abs().max() < 1e-9 * M
    assert wq.ledger()['water'].abs().max() < 1e-6


@pytest.mark.parametrize('dev', DEVS)
def test_network_withdraw_inject_point_ledgers(dev):
    net, area, r, gen = make(dev, sites=[7, 60])
    wq = WaterQuality(r, area, {'n': 5e-7}, site_cells=[7, 60])
    nt = 40
    ro = (torch.rand(nt, net.n, generator=gen, dtype=DT) * 5 + 1).to(dev)
    c = 2.0
    for t in range(nt):
        o = wq.step(ro[t], loads={'n': c * ro[t] * 1e-3 * area},
                    point={'n': torch.tensor([5e4, 0.0], dtype=DT, device=dev)},
                    withdraw=torch.tensor([1e5, 3e5], dtype=DT, device=dev) * (t % 2),
                    inject=torch.tensor([0.0, 2e5], dtype=DT, device=dev),
                    inject_conc={'n': torch.tensor([0.0, 9.0], dtype=DT, device=dev)})
    led = wq.ledger()
    ch = wq.ch['n']
    tot = ch.in_runoff.max()
    assert led['n'].abs().max() < 1e-12 * tot
    assert led['water'].abs().max() < 1e-9 * r.in_runoff.max()
    assert ch.withdrawn.sum() > 0 and ch.injected.sum() > 0


def test_equilibrium_temperature_analytic():
    # saturated air, no sun, no wind: Te = Ta = Td
    ta = torch.tensor([12.0], dtype=DT)
    es = 0.6108 * torch.exp(17.27 * ta / (ta + 237.3))
    te, K = heat.equilibrium_temperature(ta, es, torch.zeros(1, dtype=DT),
                                         torch.zeros(1, dtype=DT))
    assert abs(te.item() - 12.0) < 1e-10
    te2, K2 = heat.equilibrium_temperature(ta, es, torch.zeros(1, dtype=DT),
                                           torch.tensor([300.0], dtype=DT))
    assert abs(te2.item() - (12.0 + 300 * 0.94 / K2.item())) < 1e-10


MET = lambda dev, ta, sw: dict(
    t_air=torch.full((1,), ta, dtype=DT, device=dev),
    vp=0.6108 * torch.exp(17.27 * torch.full((1,), ta, dtype=DT, device=dev)
                          / (torch.full((1,), ta, dtype=DT, device=dev) + 237.3)),
    wind=torch.full((1,), 2.0, dtype=DT, device=dev),
    sw=torch.full((1,), sw, dtype=DT, device=dev))


@pytest.mark.parametrize('dev', DEVS)
def test_temperature_relaxes_to_equilibrium_single_reach(dev):
    r, area = reach(dev, lag=3)
    h = 1.5
    wq = WaterQuality(r, area, temperature=dict(depth=h, k_ref=40.0), site_cells=[0])
    met = MET(dev, 15.0, 200.0)
    te = heat.equilibrium_temperature(met['t_air'], met['vp'], met['wind'],
                                      met['sw'], 0.06, 0.0)[0].item()
    T0 = 4.0
    k = heat.relaxation_rate(40.0, h)
    outs = [wq.step(torch.full((1,), 3.0 if t == 0 else 0.0, dtype=DT, device=dev),
                    met=met, t_runoff=torch.full((1,), T0, dtype=DT, device=dev))
            for t in range(6)]
    got = outs[3]['t'][0].item()
    want = te + (T0 - te) * np.exp(-k * 3 * STEP)
    assert abs(got - want) < 1e-10
    # runoff already at equilibrium stays there
    wq2 = WaterQuality(reach(dev, lag=3)[0], area, temperature=dict(depth=h), site_cells=[0])
    o = [wq2.step(torch.full((1,), 3.0, dtype=DT, device=dev), met=met,
                  t_runoff=torch.full((1,), te, dtype=DT, device=dev)) for t in range(6)]
    assert abs(o[5]['t'][0].item() - te) < 1e-10


@pytest.mark.parametrize('dev', DEVS)
def test_network_steady_equilibrium_and_heat_ledger(dev):
    net, area, r, gen = make(dev, sites=[7])
    wq = WaterQuality(r, area, temperature=dict(depth=1.0), site_cells=[7])
    n = net.n
    ta = 10.0
    met = {k: v.expand(n).contiguous() for k, v in MET(dev, ta, 150.0).items()}
    te = heat.equilibrium_temperature(met['t_air'], met['vp'], met['wind'],
                                      met['sw'])[0][0].item()
    ro = torch.full((n,), 2.0, dtype=DT, device=dev)
    for t in range(60):
        o = wq.step(ro, met=met, t_runoff=torch.full((n,), te, dtype=DT, device=dev),
                    withdraw=torch.tensor([1e5], dtype=DT, device=dev))
    ok = o['q'] > 1e-6
    assert (o['t'][ok] - te).abs().max() < 1e-10
    # colder runoff warms downstream, never beyond Te
    wq = WaterQuality(make(dev)[2], area, temperature=dict(depth=1.0))
    for t in range(60):
        o = wq.step(ro, met=met, t_runoff=torch.full((n,), 2.0, dtype=DT, device=dev))
    ok = o['q'] > 1e-6
    assert (o['t'][ok] > 2.0).all() and (o['t'][ok] <= te + 1e-9).all()
    hl = wq.ledger()['heat']
    assert hl.abs().max() < 1e-12 * wq.temp[0].in_runoff.max()
    assert (wq.atmosphere_heat() > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_graph_capture():
    dev = 'cuda'
    net, area, r, gen = make(dev, sites=[7, 60])
    wq = WaterQuality(r, area, {'n': 1e-6}, temperature=dict(depth=1.0),
                      site_cells=[7, 60])
    n = net.n
    ro = torch.full((n,), 2.0, dtype=DT, device=dev)
    ld = ro * 1e-3 * area
    met = {k: v.expand(n).contiguous() for k, v in MET(dev, 10.0, 150.0).items()}
    wd = torch.tensor([1e5, 5e4], dtype=DT, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                          # warm up
        wq.step(ro, loads={'n': ld}, met=met, withdraw=wd)
    torch.cuda.current_stream().wait_stream(s)
    wq.zero_state()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = wq.step(ro, loads={'n': ld}, met=met, withdraw=wd)
    wq.zero_state()
    eager = WaterQuality(make(dev, sites=[7, 60])[2], area, {'n': 1e-6},
                         temperature=dict(depth=1.0), site_cells=[7, 60])
    for t in range(5):
        g.replay()
        e = eager.step(ro, loads={'n': ld}, met=met, withdraw=wd)
        assert torch.allclose(out['conc']['n'], e['conc']['n'], rtol=1e-12, atol=0)
        assert torch.allclose(out['t'], e['t'], rtol=1e-12, atol=0)
        assert torch.allclose(out['q'], e['q'], rtol=1e-12, atol=0)
    assert (out['t'] > 0).any()
    assert wq.ledger()['n'].abs().max() < 1e-9 * wq.ch['n'].in_runoff.max()


@pytest.mark.parametrize('dev', DEVS)
def test_drained_reach_never_overshoots(dev):
    # single reach drained to ~0: remaining flow keeps the concentration/temperature
    r, area = reach(dev, lag=0)
    wq = WaterQuality(r, area, {'n': 0.0}, temperature=dict(depth=1.0), site_cells=[0])
    met = MET(dev, 15.0, 200.0)
    q0 = 4.0
    V = q0 * STEP
    o = wq.step(torch.full((1,), q0, dtype=DT, device=dev),
                loads={'n': torch.full((1,), 8.0 * V * 1e-3 * 1e3, dtype=DT, device=dev)},
                met=met, t_runoff=torch.full((1,), 3.0, dtype=DT, device=dev),
                withdraw=torch.full((1,), V * (1 - 1e-7), dtype=DT, device=dev))
    assert o['q'][0] < 1e-6 * q0 * 10
    r2, _ = reach(dev, lag=0)
    ref = WaterQuality(r2, area, {'n': 0.0}, temperature=dict(depth=1.0), site_cells=[0])
    o2 = ref.step(torch.full((1,), q0, dtype=DT, device=dev),
                  loads={'n': torch.full((1,), 8.0 * V, dtype=DT, device=dev)},
                  met=met, t_runoff=torch.full((1,), 3.0, dtype=DT, device=dev))
    # undrained values (valid mask is bypassed by reading the rings directly)
    assert abs(wq.ch['n'].out[0].item() / max(wq.r.out[0].item(), 1e-300)
               - 8.0) < 1e-6 * 8.0
    assert wq.ledger()['n'].abs().max() < 1e-9 * 8.0 * V


@pytest.mark.parametrize('dev', DEVS)
def test_network_drained_sites_bounded(dev):
    net, area, r, gen = make(dev, sites=[7, 60, 100, 5])
    wq = WaterQuality(r, area, {'n': 0.0}, temperature=dict(depth=1.0),
                      site_cells=[7, 60, 100, 5])
    n = net.n
    met = {k: v.expand(n).contiguous() for k, v in MET(dev, 10.0, 150.0).items()}
    te = heat.equilibrium_temperature(met['t_air'], met['vp'], met['wind'],
                                      met['sw'])[0][0].item()
    ro = torch.full((n,), 2.0, dtype=DT, device=dev)
    c = 3.0
    big = torch.full((4,), 1e30, dtype=DT, device=dev)
    for t in range(50):
        o = wq.step(ro, loads={'n': c * ro * 1e-3 * area}, met=met,
                    t_runoff=torch.full((n,), 2.0, dtype=DT, device=dev),
                    withdraw=big * (t % 3 != 0))
        ok = o['q'] > 1e-6
        if t > 5 and ok.any():
            assert o['conc']['n'][ok].max() <= c * (1 + 1e-9)
            assert o['t'][ok].max() <= te + 1e-9
            assert o['t'][ok].min() >= 2.0 - 1e-9
    assert wq.ledger()['n'].abs().max() < 1e-12 * wq.ch['n'].in_runoff.max()
