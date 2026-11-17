"""Eager CUDA's float64 cumsum over the last dimension (ScanUtils.cuh: chunks of
2 * num_threads_x with a carry, a Sklansky scan in each, num_threads_x from the row count and
length), written out by core.xscan: cumsum and cumsum_last are eager's bit for bit for short and
long rows, one chunk or several, few rows or many. GPU only."""
import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")]


def _spread(shape):
    """Values over many binades, with exact zeros, -0.0, a NaN and an infinity."""
    x = torch.randn(shape, dtype=torch.float64, device="cuda") * \
        torch.exp2(torch.randint(-30, 30, shape, device="cuda").double())
    x = torch.where(torch.rand(shape, device="cuda") < 0.2, torch.zeros_like(x), x)
    x = torch.where(torch.rand(shape, device="cuda") < 0.02, torch.full_like(x, -0.0), x)
    if shape[-1] > 3:
        x[0, 2] = float("nan")
        x[-1, 1] = float("inf")
    return x


def _same(a, b):
    return torch.equal(a.reshape(-1).view(torch.int64), b.reshape(-1).view(torch.int64))


@pytest.mark.parametrize("R", [1, 2, 3, 5, 8, 9, 33, 100, 400, 401, 1500])
@pytest.mark.parametrize("rows", [1, 3, 200, 50000])
def test_cumsum_and_last_are_eager_cumsum(R, rows):
    from torchvic.core import xscan
    torch.manual_seed(R * 7 + rows)
    x = _spread((rows, R))
    ref = torch.cumsum(x, -1)
    assert _same(ref, xscan.cumsum(x))
    assert _same(ref[..., -1], xscan.cumsum_last(x))


def test_chunk_rule_follows_the_row_count():
    """With fewer rows than columns the chunk is short and a row is scanned in several chunks."""
    from torchvic.core import xscan
    assert xscan.log_threads_x(338475, 400) == 9 and xscan.log_threads_x(3, 400) == 8
    assert xscan.log_threads_x(200, 400) == 5            # 64-element chunks: 7 for a row of 400


@pytest.mark.parametrize("N,L", [(1, 3), (96, 3), (192, 5)])
def test_segment_sum_is_slot_order_and_reproducible(N, L):
    """segment_sum adds in slot order (the CPU scatter_add's order), the same bits on every run."""
    from torchvic.core import xscan
    torch.manual_seed(N + L)
    v = _spread((5000, N))
    idx = torch.randint(0, L, (5000, N), device="cuda")
    ref = torch.zeros(5000, L, dtype=torch.float64, device="cuda")
    for j in range(N):                                   # one slot at a time: slot order
        ref = ref.scatter_add(-1, idx[:, j:j + 1], v[:, j:j + 1])
    a, b = xscan.segment_sum(v, idx, L), xscan.segment_sum(v, idx, L)
    assert _same(ref, a) and _same(a, b)
