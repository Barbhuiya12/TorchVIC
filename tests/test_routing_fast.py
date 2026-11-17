"""FastStepRouter == StepRouter, bit for bit, on CUDA."""
import numpy as np
import pytest
import torch

from torchvic.model import river_network as rn
from torchvic.model import routing_fast
from test_river_network import make_net, STEP, DT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def pair(sites, subcell=(), n_canal=0):
    net, cells = make_net()
    gauges = np.unique([int(net.order[-1]), int(net.order[-40]),
                        int(net.order[len(net.order) // 2])])
    g = torch.Generator().manual_seed(1)
    area = (torch.rand(net.n, generator=g, dtype=DT) * 1e7 + 2e7).cuda()
    cuh = torch.tensor([0.3, 0.5, 0.2], dtype=DT, device='cuda')
    mk = lambda: rn.build_step_router(net, gauges, area, STEP, cell_uh=cuh,
                                      subcell=subcell, n_canal=n_canal,
                                      site_cells=sites, device='cuda')[0]
    a = mk()
    return net, a, routing_fast.FastStepRouter.from_router(mk()), g


def same(a, b):
    for k in a._STATE:
        assert torch.equal(getattr(a, k), getattr(b, k)), k


def test_bit_identical_all_hooks():
    net, a, b, g = pair([7, 60, 100, 5])
    assert type(b).__name__ == 'FastStepRouter' and type(a).__name__ == 'StepRouter'
    for t in range(45):
        ro = (torch.rand(net.n, generator=g, dtype=DT) * 5).cuda()
        wd = (torch.rand(4, generator=g, dtype=DT) * 4e5).cuda() * (t % 4 != 0)
        ij = (torch.rand(4, generator=g, dtype=DT) * 2e5).cuda()
        for kw in (dict(withdraw=wd, inject=ij), dict(withdraw=wd), dict(inject=ij),
                   dict()):
            qa, ga = a.step(ro, **kw)
            qb, gb = b.step(ro, **kw)
            assert torch.equal(qa, qb)
            assert (ga is None) == (gb is None) and (ga is None or torch.equal(ga, gb))
        same(a, b)
    big = torch.full((4,), 1e30, dtype=DT, device='cuda')
    assert torch.equal(a.intercept(big > 0), b.intercept(big > 0))
    same(a, b)


def test_sweep_matches():
    net, a, b, g = pair([7, 60, 100, 5])

    def visit(li, sid, cap, avail):
        return 0.5 * avail, 0.1 * avail
    for t in range(30):
        ro = (torch.rand(net.n, generator=g, dtype=DT) * 5).cuda()
        for r in (a, b):
            r.begin(ro)
            r.sweep(visit)
            r.end()
        same(a, b)
