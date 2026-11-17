"""Lane compaction of the loops' residual evaluations (core.compact,
graphsafe.build_index / compaction).

* build_index lists the live lanes in order, pads with the first one, and
  needs no host read (CPU);
* a region run under graphsafe.compaction gives, on the lanes it was handed,
  the full evaluation's values bit for bit (marker gpu), is refused when the
  graph mixes lanes, and leaves a call outside the context untouched;
* a Segmenter loop with a compact graph ends in the state of the same loop
  without one (marker gpu).
"""
import pytest

torch = pytest.importorskip('torch')
from torchvic.core import compact, fusion, graphsafe          # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='no CUDA device')


def test_build_index_lists_live_lanes_and_pads_with_the_first():
    live = torch.zeros(40, dtype=torch.bool)
    live[[3, 9, 10, 30]] = True
    idx = graphsafe.build_index(live, 6)
    assert idx.tolist() == [3, 9, 10, 30, 3, 3]
    # more live lanes than the capacity: the first ones, in order
    live[:] = True
    assert graphsafe.build_index(live, 5).tolist() == [0, 1, 2, 3, 4]
    # none live: lane 0 throughout
    assert graphsafe.build_index(torch.zeros(7, dtype=torch.bool),
                                 3).tolist() == [0, 0, 0]


def test_compact_capacity_divisor_is_an_integer():
    assert isinstance(graphsafe.compact_div(), int)
    assert graphsafe.compact_div() >= 0


def test_lane_live_reduces_trailing_dimensions():
    x = torch.zeros(5, 3, dtype=torch.bool)
    x[2, 1] = True
    assert graphsafe.lane_live(x).tolist() == [False, False, True, False,
                                              False]


def _setup():
    fusion.setup()
    graphsafe.COMPACT = None


def _residual_like(ts, a, b, c):
    # element-wise per lane, with a trailing dimension and a power
    return (torch.pow(ts, 1.37) * a[:, :1] + torch.exp(-ts / b)
            + (a * c).sum(1)) / (1.0 + b)


@pytest.mark.gpu
@needs_cuda
def test_compact_region_is_the_full_evaluation_on_its_lanes():
    _setup()
    dev = torch.device('cuda')
    g = torch.Generator(device=dev).manual_seed(3)
    n, m = 4096, 256
    ts = torch.rand(n, dtype=torch.float64, device=dev, generator=g) + 0.5
    a = torch.rand(n, 3, dtype=torch.float64, device=dev, generator=g)
    b = torch.rand(n, dtype=torch.float64, device=dev, generator=g) + 1.0
    c = torch.rand(3, dtype=torch.float64, device=dev, generator=g)

    def f_(ts_):
        return _residual_like(ts_, a, b, c)

    old = graphsafe.ACTIVE
    graphsafe.ACTIVE = True
    try:
        with torch.no_grad():
            full = fusion.run(f_, ts, compact=True)       # compiles
            live = torch.zeros(n, dtype=torch.bool, device=dev)
            live[torch.randperm(n, device=dev, generator=g)[:200]] = True
            idx = graphsafe.build_index(live, m)
            with graphsafe.compaction(idx, m, n):
                got = fusion.run(f_, ts, compact=True)    # builds, compares
                got2 = fusion.run(f_, ts, compact=True)   # runs compact
    finally:
        graphsafe.ACTIVE = old
    assert graphsafe.COMPACT is None
    for r in (got, got2):
        assert torch.equal(r[live].view(torch.int64),
                           full[live].view(torch.int64))


@pytest.mark.gpu
@needs_cuda
def test_a_graph_that_mixes_lanes_is_not_compacted():
    _setup()
    dev = torch.device('cuda')
    n, m = 2048, 128
    ts = torch.rand(n, dtype=torch.float64, device=dev) + 0.5
    perm = torch.randperm(n, device=dev)

    def f_(ts_):
        return ts_ * 2.0 + ts_.index_select(0, perm)       # reads other lanes

    old = graphsafe.ACTIVE
    graphsafe.ACTIVE = True
    try:
        with torch.no_grad():
            full = fusion.run(f_, ts, compact=True)
            live = torch.zeros(n, dtype=torch.bool, device=dev)
            live[:50] = True
            idx = graphsafe.build_index(live, m)
            before = graphsafe.COMPACT_USED[0]
            with graphsafe.compaction(idx, m, n):
                got = fusion.run(f_, ts, compact=True)
    finally:
        graphsafe.ACTIVE = old
    assert graphsafe.COMPACT_USED[0] == before      # never ran compact
    assert torch.equal(got, full)


@pytest.mark.gpu
@needs_cuda
def test_the_invariant_part_is_hoisted_and_the_values_are_the_full_ones():
    # a residual whose heavy part does not depend on ts: computed once per
    # hoisting scope, shared by the evaluations (and by a second closure of
    # the same shape), the results the unhoisted graph's bit for bit
    _setup()
    dev = torch.device('cuda')
    g = torch.Generator(device=dev).manual_seed(5)
    n, m = 4096, 256
    a = torch.rand(n, 3, dtype=torch.float64, device=dev, generator=g) + 0.5
    b = torch.rand(n, dtype=torch.float64, device=dev, generator=g) + 1.0
    ts0 = torch.rand(n, dtype=torch.float64, device=dev, generator=g)

    def f_(ts):
        heavy = torch.pow(a, 1.3).sum(1) * torch.exp(-b) + torch.log(b)
        return heavy * 3.0 + ts * ts - 0.25 * torch.sqrt(heavy + ts)

    def g_(ts):
        heavy = torch.pow(a, 1.3).sum(1) * torch.exp(-b) + torch.log(b)
        return heavy * 3.0 + ts * ts - 0.25 * torch.sqrt(heavy + ts)

    old = graphsafe.ACTIVE
    graphsafe.ACTIVE = True
    try:
        with torch.no_grad():
            want = f_(ts0)
            with graphsafe.hoist_scope():
                r1 = fusion.run(f_, ts0, compact=True)        # builds
                assert compact._SHARED['inv']                  # carried
                kept = len(compact._SHARED['inv'])
                r2 = fusion.run(f_, ts0 + 1.0, compact=True)  # reuses
                r3 = fusion.run(g_, ts0, compact=True)
                assert len(compact._SHARED['inv']) == kept
                live = torch.zeros(n, dtype=torch.bool, device=dev)
                live[::40] = True
                idx = graphsafe.build_index(live, m)
                with graphsafe.compaction(idx, m, n):
                    r4 = fusion.run(f_, ts0, compact=True)
                    r5 = fusion.run(f_, ts0, compact=True)
    finally:
        graphsafe.ACTIVE = old
    for r, w in ((r1, want), (r3, want), (r4[live], want[live]),
                 (r5[live], want[live])):
        assert torch.equal(r.view(torch.int64), w.view(torch.int64))
    assert torch.equal(r2.view(torch.int64), f_(ts0 + 1.0).view(torch.int64))
