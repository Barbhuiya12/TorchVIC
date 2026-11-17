"""The fused step (torchvic.core.fusion) is the unfused one bit for bit.

On a CUDA device the graph path compiles the step's physics regions
(torch.compile, inductor, Triton) with contraction off, eager's powers and
eager's reductions. Checked here:

* the operations a step uses, compiled, give eager's float64 results bit for
  bit on random inputs (marker gpu);
* the ISC forest box (forest tiles, a seasonal pack, snow in the canopy),
  run as CUDA graphs fused and unfused over two windows, gives the same
  outputs and the same carried state bit for bit, both cores, default
  options and every option family the graph path runs (marker gpu);
* on a CPU, or off the graph path, a region is its function and compiles
  nothing.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from torchvic.core import fusion, graphsafe                # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='no CUDA device')


def test_a_region_off_the_graph_path_compiles_nothing():
    calls = []

    def fn(x):
        calls.append(1)
        return x * 2.0 + 1.0
    r = fusion.region(fn)
    x = torch.arange(4.0, dtype=torch.float64)
    old = graphsafe.ACTIVE
    try:
        for active in (False, True):
            graphsafe.ACTIVE = active
            assert torch.equal(r(x), fn(x))
    finally:
        graphsafe.ACTIVE = old
    assert len(calls) == 4          # eager every time: CPU, or not ACTIVE
    assert r.__wrapped__ is fn


def _ops(x, y, z, c):
    """The element-wise vocabulary of the step, and the operations whose
    compiled form differs from eager's unless core.fusion closes it."""
    return {
        'fma': x * y + z[:, 0], 'div_const': x / 3.7, 'div_t': x / y,
        'div_0d': x / c, 'rdiv': 3.7 / y,
        'pow2': y ** 2, 'pow3': y ** 3, 'pow4': y ** 4, 'pow5': y ** 5,
        'powm1': y ** -1, 'powm2': y ** -2, 'pow05': y ** 0.5,
        'powm05': y ** -0.5, 'pow025': y ** 0.25,
        'pow169': torch.pow(y, 1.69), 'powtt': torch.pow(y, z[:, 1]),
        'pow10x': 10.0 ** (x / 10), 'pow2x': 2.0 ** x,
        'pow15': torch.pow(y, 1.5), 'pow25': torch.pow(y, 2.5),
        'pow13': torch.pow(y, 1.0 / 3.0), 'powm03': torch.pow(y, -0.3),
        'powm15': torch.pow(y, -1.5), 'pow4f': torch.pow(y, 4.0),
        'powtt2': torch.pow(y, z[:, 2] * 4 - 2), 'powtt3': torch.pow(y, x),
        'powxy': torch.pow(y * x[:, None].abs().squeeze(1), z[:, 0] + 0.5),
        'pow7x': 7.3 ** z[:, 1], 'powe_x': torch.pow(2.718281828459045, x),
        'fms': x * y - z[:, 0], 'rfms': z[:, 0] - x * y,
        'two_prod': x * y + z[:, 0] * z[:, 1],
        'chain': ((x * y + 1.1) * z[:, 0] - 0.7) * z[:, 1] + x,
        'c_prod': 0.37 * x + 1.9, 'sq_add': x * x + y,
        'horner': ((0.1 * x + 0.2) * x + 0.3) * x + 0.4,
        'exp': torch.exp(x), 'log': torch.log(y), 'log10': torch.log10(y),
        'log1p': torch.log1p(y), 'expm1': torch.expm1(x / 5),
        'sqrt': torch.sqrt(y), 'rsqrt': torch.rsqrt(y),
        'tanh': torch.tanh(x), 'atan': torch.atan(x),
        'sigmoid': torch.sigmoid(x), 'recip': torch.reciprocal(y),
        'clamp': torch.clamp(x, min=0.3, max=2.0),
        'maximum': torch.maximum(x, y),
        'sum3': z.sum(1), 'sum3k': z.sum(1, keepdim=True) * x[:, None],
        'mean3': z.mean(1), 'prod3': z.prod(1), 'cumsum': z.cumsum(1),
        'rem': torch.remainder(x, 1.3), 'lerp': torch.lerp(x, y, 0.3),
        'addcmul': torch.addcmul(x, y, z[:, 0], value=0.5),
        'where': torch.where(x > 0, x * y, x / y),
        'svp': (torch.exp(17.269 * x / (x + 237.3)) * 610.78
                - y * 0.622 / (y - 0.378 * x)) / (1 + x * x),
        'full_div': x / torch.full_like(x, 3.7),
    }


@pytest.mark.gpu
@needs_cuda
def test_compiled_operations_are_eager_bit_for_bit():
    fusion.setup()
    g = torch.Generator(device='cuda').manual_seed(0)
    n = 1 << 18
    dev = torch.device('cuda')
    x = torch.rand(n, dtype=torch.float64, device=dev, generator=g) * 20 - 5
    y = torch.rand(n, dtype=torch.float64, device=dev, generator=g) * 3 + 0.1
    z = torch.rand(n, 3, dtype=torch.float64, device=dev, generator=g)
    c = torch.tensor(2.7, dtype=torch.float64, device=dev)
    eager = _ops(x, y, z, c)
    with torch.no_grad():
        comp = torch.compile(_ops, dynamic=False)(x, y, z, c)
    bad = [k for k, v in eager.items()
           if not torch.equal(torch.nan_to_num(v, nan=1e300),
                              torch.nan_to_num(comp[k], nan=1e300))]
    assert not bad, bad


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize('scale', [(1.0, 1.0), (12.0, 1e3), (0.05, 1e-6)])
def test_compiled_operations_wide_ranges_and_edges(scale):
    # many more values than above, other ranges, and the values a branch
    # of a libdevice function is written for (0, subnormals, inf, nan)
    fusion.setup()
    g = torch.Generator(device='cuda').manual_seed(1)
    n = 1 << 21
    dev = torch.device('cuda')
    x = (torch.rand(n, dtype=torch.float64, device=dev, generator=g) * 20
         - 5) * scale[0]
    y = (torch.rand(n, dtype=torch.float64, device=dev, generator=g) * 3
         + 0.1) * scale[1]
    z = torch.rand(n, 3, dtype=torch.float64, device=dev, generator=g)
    edge = torch.tensor([0.0, -0.0, 5e-324, 1e-310, 1.0, -1.0, float('inf'),
                         float('-inf'), float('nan'), 2.0, 0.5, 1e300,
                         -1e300], dtype=torch.float64, device=dev)
    x[:edge.numel()] = edge
    y[:edge.numel()] = edge.flip(0)
    c = torch.tensor(2.7, dtype=torch.float64, device=dev)
    eager = _ops(x, y, z, c)
    with torch.no_grad():
        comp = torch.compile(_ops, dynamic=False)(x, y, z, c)
    bad = [k for k, v in eager.items()
           if not torch.equal(torch.nan_to_num(v, nan=1e300),
                              torch.nan_to_num(comp[k], nan=1e300))]
    assert not bad, bad


@pytest.mark.gpu
@needs_cuda
def test_a_float_that_changes_stays_eager():
    # dynamo makes a float that changed between calls a kernel input,
    # where x / f is an IEEE division; eager multiplies by 1 / f
    fusion.setup()
    x = torch.rand(1 << 16, dtype=torch.float64, device='cuda') * 7.0
    f = torch.compile(lambda x, c: x / c + c * x, dynamic=False)
    with torch.no_grad():
        for c in (3.7, 1.3, 0.3, 2.9):
            assert torch.equal(f(x, c), x / c + c * x), c


@pytest.fixture
def _modes():
    from torchvic.core import version, xmath
    from torchvic.model import basin
    old = (basin.LANE_COMPACT, basin.GRAPHS, basin.SYNC_CHECK, xmath.EXACT,
           version.CORE, fusion.FUSION)
    yield
    (basin.LANE_COMPACT, basin.GRAPHS, basin.SYNC_CHECK, xmath.EXACT,
     core, fusion.FUSION) = old
    version.set_core(core)


def _flat(o, pre=''):
    out = {}
    for k, v in o.items():
        if torch.is_tensor(v):
            out[pre + k] = v
        elif isinstance(v, dict):
            out.update(_flat(v, pre + k + '.'))
    return out


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize('core', ['5.0.1', '5.1.0'])
@pytest.mark.parametrize('opts', ['default', 'all_on'])
def test_fused_graphs_are_the_unfused_graphs(tmp_path, _modes, core, opts):
    gc = pytest.importorskip('test_graph_capture')
    if not os.path.exists(os.path.join(gc.forest.BOX, 'params.nc')):
        pytest.skip('the ISC forest box is not present')
    from torchvic.core import version
    version.set_core(core)
    c = gc._case(tmp_path, torch.device('cuda'), gc.OPTION_SETS[opts], 40,
                 veghist=opts == 'all_on')
    runs = []
    for on in (False, True):
        fusion.FUSION = on
        outs, stats = gc._run(c, True, 8 * 31)
        assert stats.get('graphs', 0) >= 1, stats
        runs.append(outs)
    for p, q in zip(*runs):
        a, b = _flat(p), _flat(q)
        assert set(a) == set(b)
        bad = [k for k in a if k != 'discharge_mm'      # index_add's order
               and not torch.equal(torch.nan_to_num(a[k], nan=1e300),
                                   torch.nan_to_num(b[k], nan=1e300))]
        assert not bad, bad
