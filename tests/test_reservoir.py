"""Reservoir engine: storage transitions, limits, conservation with the
stepping router, restart equality, timing."""
import time
import numpy as np
import pytest
import torch

from torchvic.model import river_network as rn
from torchvic.reservoir import (Reservoirs, ReservoirParams, RULE_HANASAKI,
                                RULE_TARGET, RULE_PASS)

DT = 86400.0
F64 = torch.float64


@pytest.fixture(autouse=True)
def _on_cpu():
    was = torch.empty(0).device
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(was)


def tree_net(n, length=50000.0):
    """Heap-shaped binary tree draining to cell 0."""
    down = np.array([-1] + [(i - 1) // 2 for i in range(1, n)])
    ln = np.where(down >= 0, length, 0.0)
    return rn.RiverNetwork(down, ln)


def make(sites, n=15, device='cpu', **kw):
    net = tree_net(n)
    area = np.full(n, 4e7)
    r, info = rn.build_step_router(net, [0], area, DT, site_cells=np.array(sites),
                                   device=device)
    return net, r


def mk(sites, cap, device='cpu', n=15, **kw):
    net, r = make(sites, n, device)
    p = ReservoirParams(site=np.arange(len(sites)), capacity=np.asarray(cap, float), **kw)
    return r, Reservoirs(p, r)


# -- arithmetic of one operation (no routing) ---------------------------------
def op(res, I, request=0.0, evap=0.0):
    R = res.R
    ix = torch.arange(R)
    f = lambda v: torch.as_tensor(np.broadcast_to(np.asarray(v, float), (R,)).copy())
    return res.operate_all(f(I), f(request), f(evap))


def test_fill_then_spill_is_conservative():
    r, res = mk([1], [100.0], rule=RULE_TARGET, target_frac=np.ones(12),
                relax_steps=1e300, init_frac=0.0)
    S0, tot_in, tot_out = 0.0, 0.0, 0.0
    for _ in range(12):
        d = op(res, 30.0)
        tot_in += 30.0
        tot_out += float(d)
        S = float(res.storage)
        assert S <= 100.0 + 1e-12
    assert S == 100.0                       # full
    assert tot_in - tot_out == pytest.approx(S, abs=1e-9)    # nothing vanished
    d = op(res, 30.0)                       # full: all spills
    assert float(d) == pytest.approx(30.0, rel=1e-12) and float(res.last['spill']) == pytest.approx(30.0, rel=1e-12)


def test_empty_to_dead_storage_and_not_below():
    r, res = mk([1], [100.0], dead=[20.0], rule=RULE_TARGET,
                target_frac=np.zeros(12), relax_steps=1.0, init_frac=0.6)
    d = op(res, 0.0)
    assert float(res.storage) == pytest.approx(20.0)
    assert float(d) == pytest.approx(40.0)
    d = op(res, 0.0)
    assert float(res.storage) == 20.0 and float(d) == 0.0
    d = op(res, 5.0)                        # inflow passes through the outlet
    assert float(res.storage) == pytest.approx(20.0) and float(d) == pytest.approx(5.0)


def test_environmental_flow_and_its_limit():
    Q = 2.0 / DT                              # 2 m3 per step
    r, res = mk([1], [100.0], dead=[10.0], q_env=[Q], rule=RULE_TARGET,
                target_frac=np.ones(12), relax_steps=1e300, init_frac=0.5)
    d = op(res, 0.0)
    assert float(res.last['env']) == pytest.approx(2.0) and float(d) == pytest.approx(2.0)
    res.storage.fill_(10.5)                   # only 0.5 above dead storage
    d = op(res, 0.0)
    assert float(res.last['env']) == pytest.approx(0.5)
    assert float(res.storage) == pytest.approx(10.0)


def test_withdrawal_limited_by_storage_above_dead():
    r, res = mk([1], [100.0], dead=[20.0], rule=RULE_TARGET,
                target_frac=np.ones(12), relax_steps=1e300, init_frac=0.5)
    op(res, 0.0, request=10.0)
    assert float(res.last['withdrawn']) == 10.0 and float(res.storage) == 40.0
    op(res, 5.0, request=1000.0)
    assert float(res.last['withdrawn']) == pytest.approx(25.0)   # 40 + 5 - 20
    assert float(res.storage) == pytest.approx(20.0)
    assert float(res.last['release']) == 0.0


def test_evaporation_from_area_storage_relation():
    r, res = mk([1], [100.0], area_max=[10.0], area_exp=[1.0], rule=RULE_TARGET,
                target_frac=np.ones(12), relax_steps=1e300, init_frac=0.5)
    op(res, 0.0, evap=0.2)                    # A = 10 * 0.5 = 5 m2, 0.2 m
    assert float(res.last['evap']) == pytest.approx(1.0)
    assert float(res.storage) == pytest.approx(49.0)


def test_rule_curve_release():
    tg = np.full(12, 0.5)
    r, res = mk([1], [100.0], rule=RULE_TARGET, target_frac=tg, relax_steps=4.0,
                init_frac=0.9)
    op(res, 0.0)
    assert float(res.last['release']) == pytest.approx((90 - 50) / 4)


def test_hanasaki_schedule_hand_cases():
    I = np.array([[10, 12, 14, 16, 18, 20, 18, 16, 14, 12, 10, 8.0]])
    Dm = np.array([[0, 0, 5, 10, 20, 30, 20, 10, 5, 0, 0, 0.0]])
    Iy, Dy = I.mean(), Dm.mean()
    cap = 0.85 * 0 + 0.85 * 1.0 * Iy * 365.25 * 86400 * 2.0     # c = 1.7
    for irr, ex in ((False, np.full(12, Iy)),
                    (True, Iy - Dy + Dm[0])):
        r, res = mk([1], [cap], irrigation=[irr], inflow_clim=I, demand_clim=Dm,
                    init_frac=0.85 / 0.85 * 0.85)
        k = float(res.storage) / (0.85 * cap)
        np.testing.assert_allclose(res.sched[0].numpy(), k * ex, rtol=1e-12)
    # D_y > I_y branch (A.2) and small reservoir c < 0.5
    D2 = Dm * 8
    cap2 = 0.2 * Iy * 365.25 * 86400
    r, res = mk([1], [cap2], irrigation=[True], inflow_clim=I, demand_clim=D2,
                init_frac=0.5)
    D2y = D2.mean()
    Rp = Iy * D2[0] / D2y
    k = 0.5 / 0.85
    c = 0.2
    cc = (c / 0.5) ** 2
    np.testing.assert_allclose(res.sched[0].numpy(),
                               cc * k * Rp + (1 - cc) * I[0], rtol=1e-12)


def test_hanasaki_learns_climatology_online_then_operates():
    r, res = mk([1], [1e9], rule=RULE_HANASAKI, init_frac=0.5)
    assert not bool((res.n_years > 0).all())
    for m in range(13):
        res.new_month(m % 12, 2000)
        for _ in range(3):
            op(res, 10.0 * DT)
    assert bool((res.n_years > 0).all())
    # passthrough before the year is learned, scheduled release after
    assert float(res.sched.abs().sum()) > 0


# -- coupling with the router --------------------------------------------------
def drive(r, res, nt=60, seed=0, request=True, evap=True, month_len=10):
    g = np.random.default_rng(seed)
    runoff = torch.as_tensor(g.random((nt, 15)) * 40 + 5)
    q = []
    for t in range(nt):
        if t % month_len == 0:
            res.new_month((t // month_len) % 12, 2000 + t // (12 * month_len))
        req = torch.as_tensor(g.random(res.R) * 3e6) if request else None
        ev = torch.as_tensor(np.full(res.R, 0.004)) if evap else None
        qq, _ = res.step(runoff[t], request=req, evap=ev)
        q.append(qq.clone())
    return torch.stack(q)


def _params(**kw):
    return dict(site=np.arange(3), capacity=[4e8, 6e8, 3e8], dead=[2e7, 0, 1e7],
                q_env=[1.0, 0.5, 0.0], irrigation=[True, False, True],
                rule=[RULE_HANASAKI, RULE_TARGET, RULE_HANASAKI],
                target_frac=np.full((3, 12), 0.4), relax_steps=5,
                area_max=[3e7, 5e7, 2e7], init_frac=0.3,
                inflow_clim=np.full((3, 12), 20.0), demand_clim=np.full((3, 12), 10.0),
                **kw)


def build3():
    net, r = make([1, 3, 2])           # 3 is below 1: dams in series
    return r, Reservoirs(ReservoirParams(**_params()), r)


def test_conservation_with_router_to_float64():
    r, res = build3()
    drive(r, res, nt=60)
    bal = r.balance()
    scale = float(r.in_runoff.max())
    assert float(bal.abs().max()) / scale < 1e-13
    # whole system at the outlet node 0: runoff in = out + ring + dS + evap + use
    c = res.cum
    dS = float((res.storage - res.init_frac * res.cap).sum())
    lhs = float(r.in_runoff[0])
    rhs = (float(r.out[0]) + float(r.ring[:, 0].sum() * DT) + dS
           + float(c['evap'].sum()) + float(c['withdrawn'].sum()))
    assert abs(lhs - rhs) / lhs < 1e-13
    assert float(res.volume_balance().abs().max()) / 1e8 < 1e-13
    assert float(c['residual'].abs().max()) / 1e8 < 1e-13
    assert float(c['withdrawn'].sum()) > 0 and float(c['evap'].sum()) > 0
    assert float((res.storage > res.cap + 1e-6).sum()) == 0
    assert bool((res.storage >= res.dead - 1e-6).all())


def test_spill_reaches_downstream():
    net, r = make([1])
    res = Reservoirs(ReservoirParams(site=[0], capacity=[1e6], rule=RULE_TARGET,
                                     target_frac=np.ones(12), relax_steps=1e300,
                                     init_frac=1.0), r)
    drive(r, res, nt=20, request=False, evap=False)
    assert float(res.cum['spill'].sum()) > 0
    assert float(res.cum['release'].sum()) < 1e-200
    assert float(res.storage) == 1e6
    assert abs(float(r.balance().abs().max())) / float(r.in_runoff.max()) < 1e-13
    # storage full and no release: everything arriving at the site continues
    assert float(res.cum['spill']) == pytest.approx(float(res.cum['inflow']), rel=1e-13)


def test_inactive_reservoir_is_transparent():
    net, r = make([1, 3])
    p = ReservoirParams(site=[0, 1], capacity=[1e8, 1e8], build_year=[3000, 3000])
    res = Reservoirs(p, r)
    q = drive(r, res, nt=30, request=False, evap=False)
    s0 = res.storage.clone()
    net, r2 = make([1, 3])
    outs = []
    g = np.random.default_rng(0)
    runoff = torch.as_tensor(g.random((30, 15)) * 40 + 5)
    for t in range(30):
        outs.append(r2.step(runoff[t])[0])
    ref = torch.stack(outs)
    assert float((q - ref).abs().max()) / float(ref.max()) < 1e-13
    assert torch.equal(res.storage, s0)


def test_restart_equality(tmp_path):
    r, res = build3()
    full = drive(r, res, nt=60, seed=4)
    # same run in two halves, checkpoint mid-month at step 25
    r2, res2 = build3()
    g = np.random.default_rng(4)
    # reproduce drive() exactly: same draw order (runoff first, then per-step req)
    runoff = torch.as_tensor(g.random((60, 15)) * 40 + 5)
    reqs = [torch.as_tensor(g.random(3) * 3e6) for _ in range(60)]
    ev = torch.as_tensor(np.full(3, 0.004))

    def run(r_, s_, a, b):
        out = []
        for t in range(a, b):
            if t % 10 == 0:
                s_.new_month((t // 10) % 12, 2000)
            out.append(s_.step(runoff[t], request=reqs[t], evap=ev)[0].clone())
        return out

    first = run(r2, res2, 0, 25)
    r2.save(tmp_path / 'r.pt'); res2.save(tmp_path / 's.pt')
    r3, res3 = build3()
    r3.load(tmp_path / 'r.pt'); res3.load(tmp_path / 's.pt')
    rest = run(r3, res3, 25, 60)
    assert torch.equal(torch.stack(first + rest), full)
    for k in ('storage', 'sched', 'clim_in', 'k', 'month'):
        assert torch.equal(getattr(res3, k), getattr(res, k))
    assert torch.equal(r3.ring, r.ring)


def test_batched_equals_scalar_loop():
    """[R] batch equals one-reservoir-at-a-time application of the same arithmetic."""
    g = np.random.default_rng(1)
    R = 40
    net, r = make(list(range(1, 15)) , 15)
    sites = np.arange(14)
    kw = dict(site=sites, capacity=g.random(14) * 100 + 50, dead=g.random(14) * 10,
              q_env=g.random(14) * 1e-4, rule=RULE_TARGET,
              target_frac=g.random((14, 12)), relax_steps=3.0,
              init_frac=g.random(14), area_max=g.random(14) * 5, area_exp=1.3)
    res = Reservoirs(ReservoirParams(**kw), r)
    ref = []
    for i in range(14):
        n2, r2 = make([1 + i], 15)
        one = {k: (v[i:i + 1] if isinstance(v, np.ndarray) and k != 'site' else v)
               for k, v in kw.items()}
        one['site'] = np.array([0])
        ref.append(Reservoirs(ReservoirParams(**one), r2))
    for _ in range(6):
        I = torch.as_tensor(g.random(14) * 30)
        rq = torch.as_tensor(g.random(14) * 10)
        ev = torch.as_tensor(g.random(14) * 0.1)
        d = res.operate_all(I, rq, ev)
        for i in range(14):
            di = ref[i].operate_all(I[i:i + 1], rq[i:i + 1], ev[i:i + 1])
            assert float(di) == float(d[i])
            assert float(ref[i].storage) == float(res.storage[i])


def test_timing_1000_reservoirs():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = 4000
    net = tree_net(n, 20000.0)
    sites = np.arange(1, 1001) * 3                  # 1000 distinct cells
    r, info = rn.build_step_router(net, [0], np.full(n, 4e7), DT,
                                   site_cells=sites, device=dev)
    R = len(sites)
    res = Reservoirs(ReservoirParams(site=np.arange(R), capacity=np.full(R, 1e9),
                                     rule=RULE_TARGET, target_frac=np.full((R, 12), 0.5),
                                     relax_steps=10.0, area_max=np.full(R, 1e7)), r)
    runoff = torch.rand(n, dtype=F64, device=dev) * 20
    req = torch.rand(R, dtype=F64, device=dev) * 1e5
    ev = torch.full((R,), 0.003, dtype=F64, device=dev)
    if dev == 'cuda':
        # graph replay equals eager stepping on the same state
        r.reset(); res.reset()
        for _ in range(3):
            res.step(runoff, request=req, evap=ev)
        sd, rsd = res.state_dict(), r.state_dict()
        eager = res.step(runoff, request=req, evap=ev)[0].clone()
        res.load_state_dict(sd); r.load_state_dict(rsd)
        replay = res.capture(runoff, req, ev, warmup=1)
        res.load_state_dict(sd); r.load_state_dict(rsd)
        assert torch.equal(replay(), eager)
        f = replay
    else:
        f = lambda: res.step(runoff, request=req, evap=ev)
    for _ in range(3):
        f()
    if dev == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        f()
    if dev == 'cuda':
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 50 * 1e3
    print(f'1000 reservoirs, {len(r.levels)} router levels, {dev}: {ms:.3f} ms/step')
    assert float(r.balance().abs().max()) / float(r.in_runoff.max()) < 1e-12
    if dev == 'cuda':
        assert ms < (1.0 if 'H200' in torch.cuda.get_device_name() else 3.0)
