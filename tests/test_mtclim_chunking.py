"""`solar_geometry`'s day chunking changes memory and nothing else.

The routine's working arrays are [cell, day, 2881] -- the hour-angle axis
at the radiation routine's own 30-second resolution -- and at 365 days that
was 148 MiB per cell with about sixteen of them live at once. It put 200
cells of the Indian 6 km domain past a 24 GB card, on a two-day run,
because the day axis is 365 whatever the run covers.

Days are independent here: the only reduction is over the hour-angle axis
*within* a day, and the scatter writes into that day's own slots. So
processing them in groups is the same operands in the same order, and the
claim this file exists to hold is the strong one -- **bit-identical**, not
close -- against `day_chunk=None`, which is the unchunked path the
oracle-checked tests were written against.

A tolerance here would defeat the purpose. If chunking changed a value at
all it would mean the days are not independent after all, and the size of
the change would not tell you which day was wrong.
"""
import numpy as np
import pytest
import torch

from torchvic.forcing import mtclim

#: **The CPU, explicitly, and not the ambient default.**
#:
#: Bit-exactness is a CPU statement in this project
#: (docs/differences.md) and this file is where that bites
#: hardest. Chunking changes the *shape* of the tensor whose last axis is
#: reduced by `.sum(-1)`, and CUDA picks its reduction split from the
#: shape; float addition is not associative, so the same 2881 operands
#: summed in a different order differ by an ulp. Measured on an A5000:
#: flat_potrad 4.55e-13 on values of 887.7, ttmax0 4.44e-16 on 0.837 --
#: one to two ulp, and exactly zero at day_chunk=365, which is a single
#: pass and so reduces the same shape as the unchunked path.
#:
#: The CPU reduces in an order that does not depend on the outer shape, so
#: there the claim is the strong one and is asserted as such below. The
#: CUDA behaviour is asserted too, with its own bound, rather than being
#: the reason to weaken the CPU test.
#:
#: Ten standalone scripts in this directory call torch.set_default_device
#: at module scope and pytest imports every test_*.py when it collects, so
#: by the time these run the process default may already be CUDA. Pinning
#: the *inputs* is not enough -- `solar_geometry` builds its own tensors
#: with `device=device`, and None there means the ambient default, so CPU
#: inputs met CUDA internals and twelve of these failed in the full suite
#: while passing alone. Pass the device explicitly, every time.
DEV = torch.device('cpu')

#: Latitudes chosen for the cases where the hour-angle sweep behaves
#: differently: the tropics, the two Indian cells the energy solver
#: struggles on, a polar day and a polar night, and the equator exactly.
LATS = [0.0, 10.28125, 27.53125, -23.5, 45.0, 66.6, 78.0, 89.9, -89.9]
ELEVS = [0.0, 1200.0, 300.0, 50.0, 800.0, 1500.0, 2500.0, 100.0, 2800.0]


@pytest.fixture(scope='module')
def unchunked():
    return mtclim.solar_geometry(
        torch.tensor(LATS, dtype=torch.float64, device=DEV),
        torch.tensor(ELEVS, dtype=torch.float64, device=DEV),
        device=DEV, day_chunk=None)


@pytest.mark.parametrize('chunk', [1, 2, 7, 13, 16, 32, 64, 128, 364, 365,
                                   366, 400])
def test_every_chunk_size_is_bit_identical(unchunked, chunk):
    got = mtclim.solar_geometry(
        torch.tensor(LATS, dtype=torch.float64, device=DEV),
        torch.tensor(ELEVS, dtype=torch.float64, device=DEV),
        device=DEV, day_chunk=chunk)
    assert set(got) == set(unchunked)
    for k in unchunked:
        a, b = unchunked[k], got[k]
        assert a.shape == b.shape, k
        assert torch.equal(a, b), (
            '%s differs at day_chunk=%d; worst %.3e'
            % (k, chunk, float((a - b).abs().max())))


def test_chunk_1_and_365_bracket_the_loop(unchunked):
    """The two ends: one day at a time, and every day at once.

    A chunk of 1 exercises the loop 365 times with a length-1 day axis,
    which is where a shape assumption would break; 365 is one pass and must
    match the unchunked path trivially.
    """
    for chunk in (1, 365):
        got = mtclim.solar_geometry(
            torch.tensor(LATS, dtype=torch.float64, device=DEV),
            torch.tensor(ELEVS, dtype=torch.float64, device=DEV),
            device=DEV, day_chunk=chunk)
        for k in unchunked:
            assert torch.equal(unchunked[k], got[k]), '%s at chunk %d' % (
                k, chunk)


def test_the_default_is_what_the_module_says(unchunked):
    """The default path, not just the explicitly-passed one."""
    got = mtclim.solar_geometry(
        torch.tensor(LATS, dtype=torch.float64, device=DEV),
        torch.tensor(ELEVS, dtype=torch.float64, device=DEV),
        device=DEV)
    for k in unchunked:
        assert torch.equal(unchunked[k], got[k]), k
    assert 1 <= mtclim.DAY_CHUNK <= 365


def test_day_366_repeats_day_365(unchunked):
    """MetSim's solar_geom copies day 365 into day 366, here written into a pre-allocated slot.

    The unchunked path built 365 days and concatenated the last one on;
    this writes into a pre-allocated 366th. Same value, and the shapes the
    rest of MTCLIM indexes with.
    """
    for k in ('ttmax0', 'flat_potrad', 'daylength'):
        v = unchunked[k]
        assert v.shape[-1] == 366, k
        assert torch.equal(v[..., -1], v[..., -2]), k
    t = unchunked['tiny_radfract']
    assert t.shape[-2:] == (366, mtclim.TINYSTEPS_PER_DAY)
    assert torch.equal(t[..., -1, :], t[..., -2, :])


def test_radiation_fractions_sum_to_one_except_where_metsim_drops_slots():
    """A property of the output, independent of how it was computed.

    `tiny_radfract` is normalised by the day's total potential radiation,
    so an ordinary day with daylight sums to one across its 2880 slots.
    Checked because the normalisation moved inside the chunk loop, and a
    chunk that normalised by the wrong day's total would still be finite
    and still be the right shape.

    Polar days are the documented exception and are asserted as such rather
    than tolerated into the general case. Where the hour-angle sweep covers
    the whole 24 hours, two of the 2881 steps land in the same 30-second
    slot; MetSim assigns inside an ascending loop so the later one wins and
    the earlier one's radiation is dropped, and this reproduces that
    deliberately (see the scatter_reduce_ note in solar_geometry). The sum
    then falls short. Measured: 4.44e-16 on the 2307 ordinary lit days,
    up to 3.46e-04 on the 493 polar ones -- and identical in the unchunked
    path, so it is MetSim's behaviour and not the chunking's.
    """
    g = mtclim.solar_geometry(
        torch.tensor(LATS, dtype=torch.float64, device=DEV),
        torch.tensor(ELEVS, dtype=torch.float64, device=DEV),
        device=DEV)
    tot = g['tiny_radfract'].sum(-1)
    dl = g['daylength']
    lit = dl > 0.0
    polar = dl >= 86400.0
    normal = lit & ~polar
    assert normal.any() and polar.any(), 'the fixture must cover both'

    assert torch.allclose(tot[normal], torch.ones_like(tot[normal]),
                          rtol=0, atol=1e-12), (
        'ordinary lit days: worst %.3e'
        % float((tot[normal] - 1.0).abs().max()))

    # Short, never over: dropping a duplicate slot can only remove
    # radiation. The bound is what MetSim's own assign-semantics cost.
    short = 1.0 - tot[polar]
    assert (short >= -1e-12).all(), 'a polar day gained radiation'
    assert float(short.max()) < 1.0e-3, (
        'polar days lose more than the duplicate-slot drop explains: %.3e'
        % float(short.max()))

    # And a polar night contributes nothing rather than NaN.
    dark = ~lit
    if dark.any():
        assert torch.equal(tot[dark], torch.zeros_like(tot[dark]))


def test_the_polar_day_shortfall_is_not_the_chunkings():
    """The same shortfall, to the bit, with no chunking at all."""
    args = (torch.tensor(LATS, dtype=torch.float64, device=DEV),
            torch.tensor(ELEVS, dtype=torch.float64, device=DEV))
    a = mtclim.solar_geometry(*args, device=DEV)['tiny_radfract'].sum(-1)
    b = mtclim.solar_geometry(*args, device=DEV,
                              day_chunk=None)['tiny_radfract'].sum(-1)
    assert torch.equal(a, b)


# --- and what the same chunking does on the GPU ---------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA here')
def test_on_cuda_the_reduction_reorders_by_an_ulp():
    """The device statement, measured rather than assumed.

    Chunking changes the shape of the tensor `.sum(-1)` reduces, CUDA
    chooses its split from the shape, and float addition is not
    associative. So on CUDA the chunked and unchunked paths agree to an ulp
    and not to the bit -- which is a property of the backend's reduction,
    not of this loop, and the proof is that day_chunk=365 is a single pass
    over the same shape and comes out exactly equal.

    Asserted with a bound so that a real chunking error, which would be
    orders of magnitude larger, still fails here.
    """
    dev = torch.device('cuda')
    lat = torch.tensor(LATS, dtype=torch.float64, device=dev)
    elev = torch.tensor(ELEVS, dtype=torch.float64, device=dev)
    ref = mtclim.solar_geometry(lat, elev, device=dev, day_chunk=None)

    one_pass = mtclim.solar_geometry(lat, elev, device=dev, day_chunk=365)
    for k in ref:
        assert torch.equal(ref[k], one_pass[k]), (
            '%s differs at day_chunk=365, which reduces the same shape as '
            'the unchunked path -- that is a chunking error, not the '
            'backend' % k)

    for chunk in (1, 16, 128):
        got = mtclim.solar_geometry(lat, elev, device=dev, day_chunk=chunk)
        for k in ref:
            a, b = ref[k], got[k]
            scale = a.abs().max().clamp(min=1.0)
            rel = float((a - b).abs().max() / scale)
            assert rel < 1e-14, (
                '%s at day_chunk=%d differs by %.3e relative, far more than '
                'a reordered reduction explains' % (k, chunk, rel))
