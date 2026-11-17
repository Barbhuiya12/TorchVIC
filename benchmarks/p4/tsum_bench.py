"""Lanes.tsum fused against the padded-grid reference: bit for bit and
time, on a synthetic ISC-like cover (112,680 cells, 18 tiles, ~2.4
covered)."""
import time
import torch
from torchvic.core import fusion, graphsafe
from torchvic.model.lanes import Lanes
from torchvic.model.tsum_kernel import tsum_gather

torch.manual_seed(0)
torch.set_grad_enabled(False)
dev = 'cuda'
nc, ntl = 112680, 18
cv = torch.rand(nc, ntl, dtype=torch.float64, device=dev)
cv = torch.where(torch.rand(nc, ntl, device=dev) < 0.14, cv, torch.zeros_like(cv))
cv[:, 17] = torch.where(cv.sum(1) == 0, torch.ones_like(cv[:, 17]), cv[:, 17])
ln = Lanes(cv)
print('lanes', ln.n_lanes)


def t_(f, x, n=50):
    for _ in range(3):
        f(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        f(x)
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e6


for T in ((), (3,), (2,), (1,), (10,), (3, 1), (1, 3)):
    x = torch.randn((ln.n_lanes, 1) + T, dtype=torch.float64, device=dev) * \
        torch.exp2(torch.randint(-20, 20, (ln.n_lanes, 1) + T, device=dev).double())
    graphsafe.ACTIVE = False
    ref = ln.tsum(x)
    tr = t_(ln.tsum, x)
    graphsafe.ACTIVE = True
    got = ln.tsum(x)
    tf = t_(ln.tsum, x)
    graphsafe.ACTIVE = False
    from torch.profiler import profile, ProfilerActivity
    graphsafe.ACTIVE = True
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(10):
            ln.tsum(x)
        torch.cuda.synchronize()
    graphsafe.ACTIVE = False
    kf = sum(e.self_device_time_total for e in pr.key_averages()) / 10
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(10):
            ln.tsum(x)
        torch.cuda.synchronize()
    ke = sum(e.self_device_time_total for e in pr.key_averages()) / 10
    print('   device us: eager %.1f fused %.1f' % (ke, kf))
    got2 = tsum_gather(x, ln.cell_start, ln.cell_mask, ln.n_tiles).index_select(0, ln.lane_cell)
    print('   kernel op identical %s' % torch.equal(ref.view(torch.int64), got2.view(torch.int64)))
    print('T=%s identical %s  eager %.1f us  fused %.1f us' % (
        T, torch.equal(ref.view(torch.int64), got.view(torch.int64)), tr, tf))
