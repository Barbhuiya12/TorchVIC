"""River-aquifer exchange of mode fixed in the routing sweep
(torchvic/vicwur/river_exchange.py), and the router's limit on withdrawals.

* the root of the implicit loss satisfies its equation, the loss never
  exceeds the water in the reach, a dry reach books the shortfall as unmet,
  a gain adds water, and the result does not depend on the bisection steps;
* the CUDA kernel equals the torch reference;
* on a router the slot moves exactly T of water out of the ring, the ring
  stays non-negative however large the demand, and VIC-WUR 3.0's forced
  inflow (mode exact) does drive the discharge negative: that is
  VIC-WUR 3.0's behaviour, kept in mode exact (docs/connectors/vicwur.md).
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

import _paths                                               # noqa: E402,F401
from torchvic.model.routing_cascade import CascadeRouter    # noqa: E402
from torchvic.vicwur import river_exchange as RX            # noqa: E402

F64 = torch.float64
CUDA = torch.cuda.is_available()
DEVICES = ['cpu'] + (['cuda'] if CUDA else [])


def _cells(n, seed=0):
    g = np.random.default_rng(seed)
    f = lambda a: torch.as_tensor(a, dtype=F64)
    Q = f(10 ** g.uniform(-2, 3.5, n))
    width = g.uniform(30, 300, n)
    slope = g.uniform(1e-4, 1e-2, n)
    return dict(
        Q=Q, C=f(g.uniform(0.01, 5.0, n)),
        bot=f(g.uniform(0, 300, n)), den=f(width * slope ** 0.5),
        cap=f(g.uniform(0.5, 6.0, n)), rbot=f(g.uniform(-5, 5, n)),
        h=f(g.uniform(-10, 330, n)), debt=f(g.uniform(-1e5, 1e6, n)))


def _stage(x, q):
    return RX.stage(q, x['bot'], x['den'], x['cap'])


def test_root_equation_and_bounds():
    x = _cells(4000)
    T, stg, un = RX.exchange_torch(**x)
    Q = x['Q']
    assert (T <= Q + 1e-12).all()                       # not more than Q
    assert (un >= 0).all()
    hh = torch.maximum(x['h'], x['rbot'])
    free = (T - x['debt'] / 86400.0 < Q - 1e-9) & (un == 0)
    # where the water sufficed, T minus the debt term is the root
    L = T - x['debt'] / 86400.0
    res = L - x['C'] * (_stage(x, (Q - L).clamp(min=0)) - hh)
    # the stage law jumps by 1 cm at d = 0.01 m, so the root of the monotone
    # function is within the jump of the residual
    assert (res[free].abs() <= x['C'][free] * 0.011 + 1e-9).all()
    # the stage handed on is the stage after the exchange
    assert torch.allclose(stg, _stage(x, (Q - T).clamp(min=0)),
                          rtol=0, atol=0)
    # dry: loss = Q, the shortfall is the unmet demand
    dry = un > 0
    assert dry.any() and (T[dry] == Q[dry]).all()
    assert (T[~dry] < Q[~dry] + 1e-12).all()


def test_unmet_equals_shortfall_and_gain_adds():
    x = _cells(10)
    x['debt'].zero_()
    x['Q'] = torch.full((10,), 5.0, dtype=F64)
    x['h'] = torch.full((10,), 100.0, dtype=F64)
    x['rbot'] = torch.zeros(10, dtype=F64)
    # stage far above the head: loses all of Q; far below: gains
    x['bot'] = torch.tensor([200.0] * 5 + [0.0] * 5, dtype=F64)
    x['C'] = torch.full((10,), 50.0, dtype=F64)
    T, stg, un = RX.exchange_torch(**x)
    assert (T[:5] == 5.0).all() and (un[:5] > 0).all()
    assert (T[5:] < 0).all()                  # gain: water into the river
    assert (un[5:] == 0).all()


@pytest.mark.parametrize('nit', [30, 40, 70, 100])
def test_insensitive_to_iterations(nit):
    x = _cells(4000, seed=3)
    ref = RX.exchange_torch(**x, nit=60)
    out = RX.exchange_torch(**x, nit=nit)
    span = (x['Q'] - x['C'] * (x['bot'] - torch.maximum(x['h'], x['rbot']))
            ).abs().max().item()
    tol = 2.0 ** -(min(nit, 60) - 2) * max(span, 1.0) \
        + (2.0 ** -(nit - 2) * max(span, 1.0))
    for a, b in zip(ref[:1], out[:1]):
        assert (a - b).abs().max().item() <= tol + 1e-9


@pytest.mark.skipif(not CUDA, reason='needs a GPU')
def test_kernel_equals_torch_reference():
    n = 3000
    x = _cells(n, seed=5)
    r = _router(n, 'cuda')
    st = {k: x[k].cuda() for k in ('C', 'bot', 'den', 'cap', 'rbot')}
    rx = RX.RiverExchange(r, st, 60)
    rx.h.copy_(x['h'].cuda())
    rx.debt.copy_(x['debt'].cuda())
    q0 = x['Q'].cuda()
    r.ring.zero_()
    r.ring[:r.nsub] = q0 / r.nsub
    r.head.zero_()
    cells = torch.arange(n, device='cuda')
    rx.begin()
    rx.level(0, cells)
    T, stg, un = RX.exchange_torch(
        x['Q'], x['C'], x['bot'], x['den'], x['cap'], x['rbot'], x['h'],
        x['debt'], 60)
    # libdevice's pow and torch's differ in the last bits, and the stage
    # law jumps by 1 cm at d = 0.01 m: a bisection step can fall the other
    # way on a rare cell; every other cell agrees to rounding
    for a, b in ((rx.T, T), (rx.stg, stg), (rx.unmet, un)):
        bad = ((a.cpu() - b).abs() > 1e-9 * (1.0 + b.abs())).sum().item()
        assert bad <= 2, bad
    d = (r.discharge.cpu() - (x['Q'] - T)).abs()
    assert (d > 1e-9 * (1.0 + x['Q'])).sum().item() <= 2


def _router(n, device='cpu', nsub=2, L=4, force=False):
    """A chain of n cells (cell i drains to i + 1), daily UH of L bins."""
    down = np.minimum(np.arange(n) + 1, n - 1)
    uh = np.zeros((L, n))
    uh[:3] = np.array([0.5, 0.3, 0.2])[:, None]
    uh[3:] = 0.0
    return CascadeRouter(down, uh, uh, np.full(n, 2e8), 86400.0, nsub,
                         force=force, device=device)


@pytest.mark.parametrize('device', DEVICES)
def test_router_slot_moves_exactly_T_and_stays_non_negative(device):
    n = 12
    r = _router(n, device)
    g = np.random.default_rng(1)
    f = lambda a: torch.as_tensor(a, dtype=F64, device=device)
    st = dict(C=f(np.full(n, 40.0)), bot=f(np.full(n, 50.0)),
              den=f(np.full(n, 3000.0)), cap=f(np.full(n, 3.0)),
              rbot=f(np.zeros(n)))
    rx = RX.RiverExchange(r, st, 50)
    rx.h.copy_(f(np.full(n, 20.0)))            # head under the stage: loses
    ro = f(g.uniform(0, 200, n))
    worst, moved = 0.0, 0.0
    for d in range(30):
        r.rout_run(ro * (d % 3 == 0), torch.zeros_like(ro))
        r.begin()
        rx.begin()
        for gi in range(len(r._gptr) - 1):
            cells = r.group_cells(gi)
            r.route_group(gi, cells)
            m0 = r.ring.sum()
            rx.level(gi, cells)
            moved += (m0 - r.ring.sum()).item() - rx.T[cells].sum().item()
        worst = min(worst, r.ring.min().item(), r.discharge.min().item())
    assert worst >= -1e-12
    assert abs(moved) <= 1e-9
    assert rx.unmet.sum().item() > 0           # the demand outran the water


def test_w3_forced_inflow_can_make_the_discharge_negative():
    """Mode exact keeps VIC-WUR 3.0: a negative forced inflow (a losing
    river, -(RIV + DRN) < 0) is added to the cell's inflow whatever the ring
    holds, so the discharge goes negative. This is VIC-WUR 3.0's behaviour
    (the C comments of rout_basin_run.c and wu_run_local.c, 'a bin made
    negative ... stays as it is'), kept in exact mode; only mode fixed limits
    the loss."""
    n = 6
    r = _router(n, force=True)
    r.set_force(torch.full((n,), -50.0, dtype=F64))
    ro = torch.full((n,), 5.0, dtype=F64)
    lo = 0.0
    for d in range(10):
        r.rout_run(ro, torch.zeros_like(ro))
        r.begin()
        r.sweep()
        lo = min(lo, r.discharge.min().item())
    assert lo < -1.0


@pytest.mark.parametrize('device', DEVICES)
def test_denormal_discharge_stays_finite(device):
    """A cell whose bins hold denormals (5e-310) and a gaining river: the
    loss is shared in proportion to the bins, which must not divide by a
    denormal (ISC 2004, day 24: inf and NaN in the ring)."""
    n = 4
    r = _router(n, device)
    f = lambda a: torch.as_tensor(a, dtype=F64, device=device)
    st = dict(C=f(np.full(n, 3.8)), bot=f(np.full(n, 100.0)),
              den=f(np.full(n, 0.37)), cap=f(np.full(n, 3.3)),
              rbot=f(np.full(n, 100.0)))
    rx = RX.RiverExchange(r, st, 50)
    rx.h.copy_(f(np.full(n, 100.5)))           # head above the stage: gains
    r.ring.zero_()
    r.head.zero_()
    r.ring[0, 1], r.ring[1, 1], r.ring[2, 1] = 5.5e-310, 6.4e-315, 7e-320
    rx.begin()
    rx.level(0, torch.arange(n, device=device))
    assert torch.isfinite(r.ring).all()
    assert (r.ring >= 0).all()
    assert (rx.T[1] < 0).item()                # the gain is added
    assert abs(r.ring[:r.nsub, 1].sum().item() + rx.T[1].item()) < 1e-9
