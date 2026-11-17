"""model/seqsum.py: the GPU's segmented sum in row order is
torch.segment_reduce's, bit for bit (empty segments, one long segment, a
single column, many columns, NaN, inf, -0.0), and the router's discharge
with it is the one segment_reduce gave."""
import pytest

torch = pytest.importorskip('torch')

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(),
                                 reason='needs a GPU')]


def _ref(x, lengths):
    return torch.segment_reduce(x, 'sum', lengths=lengths, axis=0)


@pytest.mark.parametrize('lengths,F', [([112825], 8), ([112825], 1),
                                       ([5, 0, 17, 1, 0, 33], 3),
                                       ([300, 1000, 2, 15], 248),
                                       ([0], 4), ([1], 70)])
def test_segment_sum_is_segment_reduce(lengths, F):
    from torchvic.model import seqsum
    g = torch.Generator(device='cuda').manual_seed(7)
    n = sum(lengths)
    x = torch.randn(n, F, dtype=torch.float64, device='cuda',
                    generator=g) * torch.logspace(-8, 8, F, device='cuda',
                                                   dtype=torch.float64)
    if n > 40:
        x[3, 0] = -0.0
        x[11, -1] = float('nan')
        x[29, 0] = float('inf')
        x[30:40, :] = -0.0
    L = torch.tensor(lengths, device='cuda')
    off = torch.cat([L.new_zeros(1), torch.cumsum(L, 0)])
    got = seqsum.segment_sum(x, off)
    want = _ref(x, L)
    assert got.shape == want.shape
    assert torch.equal(got.isnan(), want.isnan())
    ok = ~want.isnan()
    assert torch.equal(got[ok], want[ok])
    # the sign of zero too
    assert torch.equal(torch.signbit(got[ok]), torch.signbit(want[ok]))


def test_route_is_the_segment_reduce_route(monkeypatch):
    from torchvic.model import routing, seqsum
    g = torch.Generator(device='cuda').manual_seed(3)
    nt, ns, nuh = 67, 5000, 60
    q = torch.rand(nt, ns, dtype=torch.float64, device='cuda', generator=g)
    uh = torch.rand(nuh, dtype=torch.float64, device='cuda', generator=g)
    uh = (uh / uh.sum()).unsqueeze(1).expand(-1, ns)
    for n_out in (1, 7):
        s2o = torch.randint(0, n_out, (ns,), device='cuda', generator=g)
        a = routing.route(q, uh, s2o, n_out)[59:]
        b = routing.route(q, uh, s2o, n_out, skip=59)
        monkeypatch.setattr(seqsum, 'available', lambda x: False)
        c = routing.route(q, uh, s2o, n_out)[59:]
        monkeypatch.undo()
        assert torch.equal(a, b) and torch.equal(a, c)
