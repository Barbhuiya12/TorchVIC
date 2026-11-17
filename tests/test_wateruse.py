"""Irrigation / human water-use engine: allocation, accounting, router."""
import time
import pytest
import torch

from torchvic.wateruse import (WaterUseEngine, WaterUseConfig, StubSource,
                               irrigation_requirement)

DT = torch.float64
DEVS = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def mk_soil(N, K=2, dev='cpu', moist=40.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g, dtype=DT).to(dev)
    wfc = torch.full((N, K, 3), 50.0, dtype=DT, device=dev)
    return dict(
        moist=moist * torch.ones(N, K, 3, dtype=DT, device=dev) * (0.5 + r(N, K, 3)),
        ice=torch.zeros(N, K, 3, dtype=DT, device=dev),
        wfc=wfc, wcr=0.7 * wfc, root=torch.ones(N, K, 3, dtype=DT, device=dev),
        max_moist0=torch.full((N, K), 80.0, dtype=DT, device=dev),
        frac=torch.full((N, K), 0.3, dtype=DT, device=dev),
        paddy=torch.zeros(N, K, dtype=torch.bool, device=dev),
        season_open=torch.ones(N, K, dtype=torch.bool, device=dev),
        moist0=torch.full((N, K), 30.0, dtype=DT, device=dev))


def eng(N=5, dev='cpu', cfg=None, **kw):
    area = torch.full((N,), 1e8, dtype=DT, device=dev)
    return WaterUseEngine(area, n_other=2, cfg=cfg or WaterUseConfig(), **kw)


def close(e):
    b = e.balance()
    scale = max(float(e.led["demand"].max()), 1.0)
    assert b.abs().max() < 1e-12 * scale, b.abs().max()


def test_requirement_flags():
    z = torch.zeros(1, 1, 3, dtype=DT)
    wfc, wcr = torch.full((1, 1, 3), 50., dtype=DT), torch.full((1, 1, 3), 35., dtype=DT)
    root = torch.tensor([[[1., 1., 0.]]], dtype=DT)
    t = torch.ones(1, 1, dtype=torch.bool)
    f = torch.zeros(1, 1, dtype=torch.bool)
    lo = torch.tensor([[[20., 20., 5.]]], dtype=DT)   # 40 < Wcr 70
    req, fl = irrigation_requirement(lo, z, None, wfc, wcr, torch.tensor([[80.]], dtype=DT),
                                     root, torch.zeros(1, 1, dtype=DT), f, t)
    assert fl.item() and req.item() == pytest.approx(100 - 40)   # irr_run.c:155-157
    hi = torch.tensor([[[40., 40., 5.]]], dtype=DT)   # 80 >= Wfc 100? no, between Wcr and Wfc
    req, fl = irrigation_requirement(hi, z, None, wfc, wcr, torch.tensor([[80.]], dtype=DT),
                                     root, torch.zeros(1, 1, dtype=DT), f, t)
    assert not fl.item() and req.item() == 0                     # above critical point
    req, fl = irrigation_requirement(lo, z, None, wfc, wcr, torch.tensor([[80.]], dtype=DT),
                                     root, torch.zeros(1, 1, dtype=DT), f, f)
    assert not fl.item()                                         # offset not passed
    req, fl = irrigation_requirement(lo, z, None, wfc, wcr, torch.tensor([[80.]], dtype=DT),
                                     root, torch.zeros(1, 1, dtype=DT), t, t)
    assert fl.item() and req.item() == pytest.approx(80 - 20)    # paddy, irr_run.c:148-150


@pytest.mark.parametrize('dev', DEVS)
def test_source_limited_never_exceeds_availability(dev):
    N = 6
    e = eng(N, dev, irr_gw_frac=torch.full((N,), 0.5, dtype=DT, device=dev),
            gw_frac=torch.full((N, 2), 0.5, dtype=DT, device=dev))
    g = torch.Generator().manual_seed(1)
    gw = StubSource(torch.rand(N, generator=g, dtype=DT).to(dev) * 3e5)
    res = StubSource(torch.rand(N, generator=g, dtype=DT).to(dev) * 3e5)
    riv = torch.rand(N, generator=g, dtype=DT).to(dev) * 3e5
    gw0, res0 = gw.store.clone(), res.store.clone()
    dem = torch.rand(N, 2, generator=g, dtype=DT).to(dev) * 4e5
    o = e.step(dem, mk_soil(N, dev=dev), gw=gw, res=res, river_avail=riv)
    assert (o['gw'].sum(-1) <= gw0 + 1e-9).all()
    assert (o['res'].sum(-1) <= res0 + 1e-9).all()
    assert (o['river'].sum(-1) <= riv + 1e-9).all()
    assert (o['unmet'] >= -1e-9).all()
    assert torch.allclose(o['withdrawn'] + o['unmet'], o['demand'])
    assert (gw.store >= 0).all()
    close(e)


def test_hand_case_10_3_2():
    # demand 10 (m3), river 3, groundwater 2 -> withdrawn 5, unmet 5; no irrigation
    e = WaterUseEngine(torch.full((1,), 1e8, dtype=DT), n_other=1, irrigation=False,
                       gw_frac=torch.tensor([[0.5]], dtype=DT))
    o = e.step(torch.tensor([[10.]], dtype=DT), gw=StubSource(torch.tensor([2.], dtype=DT)),
               river_avail=torch.tensor([3.], dtype=DT))
    assert o['withdrawn'].item() == pytest.approx(5.0) and o['unmet'].item() == pytest.approx(5.0)
    # G=1 with fallback off: only groundwater
    e = WaterUseEngine(torch.full((1,), 1e8, dtype=DT), n_other=1, irrigation=False,
                       gw_frac=torch.tensor([[1.0]], dtype=DT),
                       cfg=WaterUseConfig(fallback=False))
    o = e.step(torch.tensor([[10.]], dtype=DT), gw=StubSource(torch.tensor([2.], dtype=DT)),
               river_avail=torch.tensor([3.], dtype=DT))
    assert o['river'].item() == 0 and o['gw'].item() == pytest.approx(2.0)
    # zero availability: nothing
    o = e.step(torch.tensor([[10.]], dtype=DT))
    assert o['withdrawn'].item() == 0


def test_priority_vs_proportional():
    d = torch.tensor([[4., 6., 10.]], dtype=DT)
    kw = dict(n_other=3, irrigation=False)
    area = torch.ones(1, dtype=DT) * 1e8
    pr = WaterUseEngine(area, cfg=WaterUseConfig(allocation="priority"), **kw)
    pp = WaterUseEngine(area, cfg=WaterUseConfig(allocation="proportional"), **kw)
    a = torch.tensor([8.], dtype=DT)
    w1 = pr.step(d, river_avail=a)['withdrawn']
    w2 = pp.step(d, river_avail=a)['withdrawn']
    # priority (wu_run_local.c:194-250): first sectors served in full, last gets the rest
    assert w1.tolist() == [[4., 4., 0.]]
    assert torch.allclose(w2, d * 0.4)
    assert w1.sum() == pytest.approx(8.) and w2.sum() == pytest.approx(8.)


def test_no_double_counting_between_sources():
    N = 4
    e = eng(N, irr_gw_frac=torch.full((N,), 0.3, dtype=DT),
            gw_frac=torch.full((N, 2), 0.3, dtype=DT))
    gw = StubSource(torch.full((N,), 1e5, dtype=DT))
    res = StubSource(torch.full((N,), 1e5, dtype=DT))
    soil = mk_soil(N)
    for _ in range(5):
        o = e.step(torch.full((N, 2), 3e4, dtype=DT), soil, gw=gw, res=res,
                   river_avail=torch.full((N,), 1e5, dtype=DT))
    L = e.led
    assert torch.allclose(gw.drawn, L['wd_gw']) and torch.allclose(1e5 - gw.store, L['wd_gw'])
    assert torch.allclose(res.drawn, L['wd_res'])
    assert torch.allclose(L['demand'], L['wd_gw'] + L['wd_res'] + L['wd_riv'] + L['unmet'])
    close(e)


@pytest.mark.parametrize('lr', ['river', 'soil'])
@pytest.mark.parametrize('alloc', ['priority', 'proportional'])
def test_conservation_and_cap(lr, alloc):
    N = 7
    cfg = WaterUseConfig(loss_return=lr, allocation=alloc)
    e = eng(N, cfg=cfg, cons_frac=torch.full((N, 2), 0.2, dtype=DT),
            conv_loss=torch.full((N, 2), 0.1, dtype=DT),
            irr_eff=torch.full((N,), 0.6, dtype=DT),
            irr_conv_loss=torch.full((N,), 0.15, dtype=DT))
    soil = mk_soil(N)
    soil['moist0'] = torch.full((N, 2), 78.0, dtype=DT)   # tiny headroom: leftover builds
    for k in range(6):
        o = e.step(torch.full((N, 2), 2e4, dtype=DT), soil,
                   river_avail=torch.full((N,), 6e4 * (k % 3 + 1), dtype=DT))
        assert (soil['moist0'] + o['soil_add'] <= soil['max_moist0'] + 1e-9).all()
        assert (o['overflow'] >= 0).all()
    assert e.leftover_m3.sum() > 0
    close(e)
    if lr == 'soil':
        assert e.led['irr_loss_river'].sum() == 0 and e.led['irr_loss_soil'].sum() > 0
    else:
        assert e.led['irr_loss_soil'].sum() == 0 and e.led['irr_loss_river'].sum() > 0


def test_checkpoint_restart():
    N = 4
    mk = lambda: eng(N, irr_eff=torch.full((N,), 0.7, dtype=DT))
    soil = mk_soil(N)
    soil['moist0'] = torch.full((N, 2), 79.0, dtype=DT)
    riv = torch.full((N,), 5e4, dtype=DT)
    a = mk()
    for _ in range(4):
        a.step(torch.full((N, 2), 1e3, dtype=DT), soil, river_avail=riv)
    sd = a.state_dict()
    b = mk()
    b.load_state_dict(sd)
    oa = a.step(torch.full((N, 2), 1e3, dtype=DT), soil, river_avail=riv)
    ob = b.step(torch.full((N, 2), 1e3, dtype=DT), soil, river_avail=riv)
    assert torch.equal(oa['soil_add'], ob['soil_add']) and torch.equal(a.leftover, b.leftover)


def test_with_step_router():
    from test_step_router import build, DT as RDT
    sites = [7, 60]
    net, gauges, runoff, area, cuh, r, info = build('cpu', sites=sites, nt=40)
    a = area[torch.as_tensor(sites)]
    e = WaterUseEngine(a, site_of_cell=torch.tensor([0, 1]), n_other=1, irrigation=False,
                       cons_frac=torch.full((2, 1), 0.3, dtype=DT),
                       conv_loss=torch.full((2, 1), 0.1, dtype=DT))
    tot_w = torch.zeros(2, dtype=DT)
    for t in range(40):
        r.begin(runoff[t])
        o = e.step(torch.full((2, 1), 4e5, dtype=DT), router=r)
        tot_w += o['river'].sum(-1)
        r.end()
    assert r.balance().abs().max() < 1e-9 * r.in_runoff.max()
    assert torch.allclose(r.withdrawn[:2], e.led['wd_riv'])
    assert torch.allclose(r.injected[:2], e.led['ret_river_other'])
    assert (e.led['wd_riv'] > 0).all() and (e.led['unmet'] >= -1e-9).all()
    close(e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_gpu_step_time():
    # the 2 ms bound is the H200's; a slower or shared card (RTX) measures
    # 2.4 ms on the same code
    if 'H200' not in torch.cuda.get_device_name():
        pytest.skip('the step-time bound is an H200 figure')
    N, K = 112825, 4
    dev = 'cuda'
    e = eng(N, dev, irr_gw_frac=torch.full((N,), 0.4, dtype=DT, device=dev),
            gw_frac=torch.full((N, 2), 0.3, dtype=DT, device=dev))
    soil = mk_soil(N, K, dev)
    dem = torch.full((N, 2), 1e4, dtype=DT, device=dev)
    riv = torch.full((N,), 1e5, dtype=DT, device=dev)
    gw = StubSource(torch.full((N,), 1e9, dtype=DT, device=dev))
    res = StubSource(torch.full((N,), 1e9, dtype=DT, device=dev))
    f = lambda: e.step(dem, soil, gw=gw, res=res, river_avail=riv)
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(50):
        f()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / 50 * 1e3
    print(f'wateruse step {ms:.3f} ms for {N} cells')
    assert ms < 2.0
