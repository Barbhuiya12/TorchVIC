"""Multi-GPU flood engine (torchvic/flood/multigpu.py): row strips with a
halo exchange equal the single-GPU float64 engine bit for bit.

Strips may share a device (devices=['cuda:0', 'cuda:0']): the same code
path (streams, peer copies, the clocks) on one GPU, so the gpu suite checks
it with one GPU; the two-GPU case runs where two GPUs are visible."""
import numpy as np
import pytest
import torch

from torchvic.flood import FloodEngine

CUDA = torch.cuda.is_available()
TWO = CUDA and torch.cuda.device_count() >= 2
STATE = ('h', 'qx', 'qy', 'Qx', 'Qy', 'hmax', 'hmax_run')

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not CUDA, reason='needs CUDA')]


def basin(ny=61, nx=47, dx=25.0):
    y, x = np.mgrid[0:ny, 0:nx]
    z = (0.004 * x * dx + 0.3 * np.cos(y / 5.0) * np.sin(x / 7.0)
         + 2e-5 * ((y - ny / 2) * dx) ** 2)
    return z, dx


def make(devices, **kw):
    z, dx = basin()
    ny, nx = z.shape
    n = 0.03 + 0.01 * np.sin(np.arange(ny * nx).reshape(ny, nx) / 13.0)
    sl = np.full(ny, 1e-3)
    sl[:7] = np.nan                     # part of the W edge closed
    k = dict(free_edges={'W': sl, 'N': -1.0, 'S': 2e-3},
             inflow_points=[(30, 40), (31, 40), (59, 3), (1, 20)],
             max_dt=10.0, track_edges=True, block=16)
    k.update(kw)
    if len(devices) > 1:
        k['devices'] = devices
    return FloodEngine(z, dx, n, device=devices[0], **k)


def drive(e, days=4):
    z, _ = basin()
    out = []
    for d in range(days):
        r = np.full(z.shape, (2.0 + d) * 1e-3)
        r[:, :10] *= 3.0
        o = e.run(1800.0, runoff=r * 625.0,
                  inflow=[4e4 * (1 + d), 2e4, 3e3 * d, 500.0])
        out.append(o)
    return out


def assert_same(e1, eK, o1, oK):
    for n in STATE:
        assert torch.equal(getattr(e1, n).cpu(), getattr(eK, n).cpu()), n
    for a, b in zip(o1, oK):
        assert torch.equal(a['edge_out'].cpu(), b['edge_out'].cpu())
        for x in ('depth', 'max_depth', 'speed'):
            assert torch.equal(a[x].cpu(), b[x].cpu()), x
        la = {k: v for k, v in a['ledger'].items() if k != 'blocks'}
        lb = {k: v for k, v in b['ledger'].items() if k != 'blocks'}
        assert la == lb
    assert e1.totals == eK.totals and e1.t == eK.t and e1.steps == eK.steps


@pytest.mark.parametrize('k', [2, 3])
def test_strips_equal_single_gpu(k):
    e1 = make(['cuda:0'])
    eK = make(['cuda:0'] * k)
    o1, oK = drive(e1), drive(eK)
    assert e1.storage() > 0 and o1[-1]['ledger']['outflow'] > 0
    assert abs(o1[-1]['ledger']['rel_residual']) < 1e-13
    assert_same(e1, eK, o1, oK)


def test_strips_restart_equals_continuous():
    eA = make(['cuda:0'] * 2)
    oA = drive(eA, 4)
    eB = make(['cuda:0'] * 2)
    drive(eB, 2)
    sd = eB.state_dict()
    eC = make(['cuda:0'] * 2)
    eC.load_state_dict(sd)
    z, _ = basin()
    oC = []
    for d in (2, 3):
        r = np.full(z.shape, (2.0 + d) * 1e-3)
        r[:, :10] *= 3.0
        oC.append(eC.run(1800.0, runoff=r * 625.0,
                         inflow=[4e4 * (1 + d), 2e4, 3e3 * d, 500.0]))
    assert_same(eA, eC, oA[2:], oC)


@pytest.mark.parametrize('k', [2, 3])
def test_strips_host_staged_copies(k):
    e1 = make(['cuda:0'])
    eK = make(['cuda:0'] * k, triton_cfg=dict(copy='host'))
    assert_same(e1, eK, drive(e1, 2), drive(eK, 2))
    assert eK._stepper.copy == 'host'


def test_flag_barriers_need_distinct_devices():
    e = make(['cuda:0'] * 2, triton_cfg=dict(sync='flag'))
    with pytest.raises(ValueError):
        drive(e, 1)


def test_strips_without_graphs():
    e1 = make(['cuda:0'])
    eK = make(['cuda:0'] * 2, graphs=False)
    assert_same(e1, eK, drive(e1, 2), drive(eK, 2))


@pytest.mark.skipif(not TWO, reason='needs two GPUs')
@pytest.mark.parametrize('cfg', [{}, dict(sync='event'), dict(copy='host')])
def test_two_gpus_equal_single_gpu(cfg):
    e1 = make(['cuda:0'])
    eK = make(['cuda:0', 'cuda:1'], triton_cfg=cfg)
    assert_same(e1, eK, drive(e1), drive(eK))


def test_multigpu_refuses_other_paths():
    with pytest.raises(ValueError):
        make(['cuda:0'] * 2, dtype=torch.float32)
    with pytest.raises(ValueError):
        make(['cuda:0'] * 2, mixed=True)
    with pytest.raises(ValueError):
        make(['cuda:0'] * 2, backend='torch')
    e = make(['cuda:0'] * 2, triton_cfg=dict(qflag=True))
    with pytest.raises(ValueError):
        drive(e, 1)
    e = FloodEngine(np.zeros((6, 8)), 10.0, devices=['cuda:0'] * 2)
    with pytest.raises(ValueError):               # 3 rows per strip
        e.run(10.0)


def test_two_way_coupled_system_unchanged():
    """The two-way coupled system (routing hand-over, return outlet) with
    the flood connector's `devices` equals the single-GPU run: flood state,
    every ledger row, the routing state."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_coupling_flood_2way import build, runoff_table
    res = []
    for kw in (None, dict(devices=['cuda:0', 'cuda:0'])):
        m = build('cuda', table=runoff_table(nt=6, wet=3), flood_kw=kw)
        m.initialize().run()
        res.append(m)
    a, b = res
    assert b.connector('flood').engine.devices == ['cuda:0', 'cuda:0']
    for k in STATE:
        assert torch.equal(getattr(a.connector('flood').engine, k),
                           getattr(b.connector('flood').engine, k)), k
    for rows in ('balances', 'coupled'):
        ra, rb = getattr(a.ledger, rows), getattr(b.ledger, rows)
        assert len(ra) == len(rb) > 0
        for x, y in zip(ra, rb):
            kx = {k: v for k, v in x.items() if isinstance(v, float)}
            ky = {k: v for k, v in y.items() if isinstance(v, float)}
            assert kx == ky
    rta, rtb = a.connector('routing').router, b.connector('routing').router
    for k in rta._STATE:
        assert torch.equal(getattr(rta, k), getattr(rtb, k)), k
    assert a.connector('flood').totals == b.connector('flood').totals
    from torchvic.coupling.connectors import flood as F
    with pytest.raises(Exception):
        F.FloodConnector(dem=basin()[0], dx=25.0, backend='engine',
                         dtype='mixed', devices=['cuda:0', 'cuda:0']
                         ).initialize(type('C', (), dict(
                             device='cuda', t=0.0, say=print))())
