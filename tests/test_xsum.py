"""Eager CUDA's float64 sum over a short dimension, written out as its own
addition tree (core.xsum), and the covered-tile sum built on it
(model.tsum_kernel, Lanes.tsum on the graph path), are eager's bit for
bit (ROADMAP W2 P4). GPU only: the tree is CUDA's Reduce.cuh order."""
import pytest

torch = pytest.importorskip('torch')

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='no CUDA device')


def _spread(shape, pz, dev):
    """Values over many binades, with exact zeros and -0.0."""
    m = torch.randn(shape, dtype=torch.float64, device=dev)
    x = m * torch.exp2(torch.randint(-30, 30, shape, device=dev).double())
    x = torch.where(torch.rand(shape, device=dev) < pz, torch.zeros_like(x), x)
    return torch.where(torch.rand(shape, device=dev) < 0.02,
                       torch.full_like(x, -0.0), x)


def _same(a, b):
    return torch.equal(a.reshape(-1).view(torch.int64),
                       b.reshape(-1).view(torch.int64))


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize('R', [1, 2, 3, 5, 8, 17, 18, 19, 33, 64])
@pytest.mark.parametrize('L', [None, 2, 3, 7])
def test_tree_is_eager_sum(R, L):
    from torchvic.core import xsum
    torch.manual_seed(R * 100 + (L or 0))
    for n in (1, 1000, 50000):
        x = _spread((n, R) if L is None else (n, R, L), 0.5, 'cuda')
        assert _same(x.sum(1), xsum.sum1(x))


@pytest.mark.gpu
@needs_cuda
def test_tree_compiled_is_eager_sum():
    from torchvic.core import fusion, xsum
    fusion.setup()
    torch.manual_seed(0)
    for R, L in ((18, None), (18, 3), (5, None), (33, 2)):
        x = _spread((20000, R) if L is None else (20000, R, L), 0.6, 'cuda')
        f = torch.compile(lambda a: a.sum(1), dynamic=False)
        assert _same(x.sum(1), f(x))
        torch._dynamo.reset()


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize('tail', [(), (1,), (3,), (2,), (1, 3)])
def test_tsum_kernel_is_padded_sum(tail):
    from torchvic.core import graphsafe
    from torchvic.model.lanes import Lanes
    from torchvic.model.tsum_kernel import tsum_gather
    torch.manual_seed(len(tail))
    nc, ntl = 20000, 18
    cv = torch.rand(nc, ntl, dtype=torch.float64, device='cuda')
    cv = torch.where(torch.rand(nc, ntl, device='cuda') < 0.3, cv,
                     torch.zeros_like(cv))
    cv[:, 17] = torch.where(cv.sum(1) == 0, torch.ones_like(cv[:, 17]),
                            cv[:, 17])
    ln = Lanes(cv)
    x = _spread((ln.n_lanes, 1) + tail, 0.1, 'cuda')
    old = graphsafe.ACTIVE
    try:
        graphsafe.ACTIVE = False
        ref = ln.tsum(x)                  # the padded grid's sum
    finally:
        graphsafe.ACTIVE = old
    got = tsum_gather(x, ln.cell_start, ln.cell_mask, ln.n_tiles)
    assert _same(ref, got.index_select(0, ln.lane_cell))
