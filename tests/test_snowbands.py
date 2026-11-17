"""VIC's elevation bands, against the arithmetic in vic_init.c.

The model runs one column per cell. VIC runs one per (cell, band) with
non-zero area, shifting only the air temperature and the precipitation
(`surface_fluxes.c:385-386`) and weighting every output by
`Cv * AreaFract[band]` (`vic_run.c:349-422`). `torchvic.core.snowbands` closes
that gap by expanding the cell list rather than by adding physics, so the
whole of what has to be right is the two factors and the weighting.

Both factors have a wrinkle that is silent when wrong, and each has a test
of its own here:

  * Tfactor is lapsed from the **area-weighted mean band elevation**, not
    from the file's `elev`, because vic_init.c:1005-1010 overwrites the
    cell's elevation with that mean. Using `elev` shifts every band.

  * Pfactor is a precipitation *fraction* that vic_init.c:1042-1059
    normalises to sum one and then divides by AreaFract. Skipping the
    division leaves the precipitation wrong by the area of the band.

And the property that makes the feature safe to turn on: a domain whose
cells each have a single band must expand to itself, with tfactor 0 and
pfactor 1, so no single-band answer can move.
"""
import numpy as np
import pytest

from torchvic.core import snowbands


def test_single_band_is_the_identity():
    """One band per cell: same cells, no shift, no scaling."""
    n = 6
    area = np.ones((1, n))
    elev = np.linspace(100.0, 3000.0, n).reshape(1, n)
    pf = np.ones((1, n))
    ex = snowbands.expand(area, elev, pf)
    assert len(ex['cell']) == n
    assert np.array_equal(ex['cell'], np.arange(n))
    assert np.array_equal(ex['band'], np.zeros(n, dtype=ex['band'].dtype))
    assert np.allclose(ex['tfactor'], 0.0, rtol=0, atol=0)
    assert np.allclose(ex['pfactor'], 1.0, rtol=0, atol=1e-15)
    assert np.allclose(ex['weight'], 1.0)
    # And the cell's elevation is its own.
    assert np.allclose(ex['mean_elev'], elev[0])


def test_tfactor_lapses_from_the_band_mean_not_the_files_elevation():
    """vic_init.c:1005-1016, and the substitution it makes first."""
    area = np.array([[0.25], [0.75]])
    elev = np.array([[1000.0], [2000.0]])
    pf = np.array([[0.25], [0.75]])
    ex = snowbands.expand(area, elev, pf)

    mean = 0.25 * 1000.0 + 0.75 * 2000.0        # 1750, not the file's elev
    assert np.allclose(ex['mean_elev'], mean)
    want = (np.array([1000.0, 2000.0]) - mean) * snowbands.LAPSE_RATE
    assert np.allclose(ex['tfactor'], want, rtol=0, atol=1e-12)
    # Signs: the high band is colder, the low band warmer.
    assert ex['tfactor'][0] > 0 and ex['tfactor'][1] < 0
    # And the area-weighted mean shift is zero -- the bands redistribute
    # temperature, they do not add any.
    assert abs(float((ex['tfactor'] * ex['weight']).sum())) < 1e-12


def test_pfactor_is_a_fraction_divided_by_area():
    """vic_init.c:1042-1059. A band with a tenth of the area and a fifth
    of the precipitation gets a multiplier of two, not of a fifth."""
    area = np.array([[0.1], [0.9]])
    elev = np.array([[1000.0], [1200.0]])
    pf = np.array([[0.2], [0.8]])               # fractions, already summing 1
    ex = snowbands.expand(area, elev, pf)
    assert np.allclose(ex['pfactor'], [0.2 / 0.1, 0.8 / 0.9])
    # The area-weighted multiplier is one: bands redistribute the cell's
    # precipitation, they do not create any.
    assert abs(float((ex['pfactor'] * ex['weight']).sum()) - 1.0) < 1e-12


def test_precipitation_fractions_are_normalised_first():
    """:1042-1050 divides by the sum when it is not one."""
    area = np.array([[0.5], [0.5]])
    elev = np.array([[1000.0], [1100.0]])
    pf = np.array([[1.0], [3.0]])               # sums to 4, not 1
    ex = snowbands.expand(area, elev, pf)
    assert np.allclose(ex['pfactor'], [0.25 / 0.5, 0.75 / 0.5])
    assert abs(float((ex['pfactor'] * ex['weight']).sum()) - 1.0) < 1e-12


def test_zero_area_bands_are_not_run():
    """vic_run.c:187 -- `if (soil_con->AreaFract[band] > 0)`.

    WalnutGulch's parameter file carries five bands of which four have zero
    area; those must not become columns.
    """
    area = np.array([[1.0, 0.4], [0.0, 0.6], [0.0, 0.0]])
    elev = np.array([[1200.0, 900.0], [1400.0, 1500.0], [1600.0, 2000.0]])
    pf = np.array([[1.0, 0.3], [0.0, 0.7], [0.0, 0.0]])
    ex = snowbands.expand(area, elev, pf)
    assert len(ex['cell']) == 3, 'one column for cell 0, two for cell 1'
    assert list(ex['cell']) == [0, 1, 1]
    assert list(ex['band']) == [0, 0, 1]


def test_nan_bands_are_absent_bands():
    """The Indian file writes NaN for bands a cell does not have, the same
    convention its vegetation cover uses."""
    area = np.array([[1.0], [np.nan]])
    elev = np.array([[800.0], [np.nan]])
    pf = np.array([[1.0], [np.nan]])
    ex = snowbands.expand(area, elev, pf)
    assert len(ex['cell']) == 1
    assert np.isfinite(ex['tfactor']).all() and np.isfinite(ex['pfactor']).all()


# --- weighting back ---------------------------------------------------------

def test_contract_is_the_area_weighted_mean():
    area = np.array([[0.25], [0.75]])
    elev = np.array([[1000.0], [2000.0]])
    pf = np.array([[0.25], [0.75]])
    ex = snowbands.expand(area, elev, pf)
    per_band = np.array([4.0, 8.0])
    got = snowbands.contract(per_band, ex)
    assert got.shape == (1,)
    assert np.allclose(got, 0.25 * 4.0 + 0.75 * 8.0)


def test_contract_preserves_a_uniform_field():
    """If every band reports the same value, the cell reports it too --
    because the areas sum to one."""
    rng = np.random.default_rng(0)
    n = 5
    raw = rng.random((4, n))
    area = raw / raw.sum(axis=0, keepdims=True)
    elev = rng.uniform(500.0, 5000.0, (4, n))
    ex = snowbands.expand(area, elev, area.copy())
    ones = np.ones(len(ex['cell']))
    assert np.allclose(snowbands.contract(ones, ex), 1.0, rtol=0, atol=1e-12)


def test_contract_handles_a_leading_time_axis():
    """The model's outputs are [step, cell]; the cell axis is the last."""
    area = np.array([[0.5, 0.2], [0.5, 0.8]])
    elev = np.array([[1000.0, 500.0], [2000.0, 900.0]])
    ex = snowbands.expand(area, elev, area.copy())
    # Cell-major: cell 0's two bands, then cell 1's two.
    assert list(ex['cell']) == [0, 0, 1, 1]
    series = np.array([[1.0, 3.0, 10.0, 20.0],
                       [2.0, 4.0, 30.0, 40.0]])      # [step, pseudo-cell]
    got = snowbands.contract(series, ex)
    assert got.shape == (2, 2)
    assert np.allclose(got[:, 0], [0.5 * 1 + 0.5 * 3, 0.5 * 2 + 0.5 * 4])
    assert np.allclose(got[:, 1], [0.2 * 10 + 0.8 * 20, 0.2 * 30 + 0.8 * 40])


def test_contract_matches_between_numpy_and_torch():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(7)
    raw = rng.random((3, 4))
    area = raw / raw.sum(axis=0, keepdims=True)
    elev = rng.uniform(0.0, 4000.0, (3, 4))
    ex = snowbands.expand(area, elev, area.copy())
    v = rng.random((6, len(ex['cell'])))
    a = snowbands.contract(v, ex)
    # Explicitly on the CPU. Ten scripts in this directory call
    # torch.set_default_device at import and pytest imports them all when
    # it collects, so by the time this runs the default may be CUDA --
    # and .numpy() on a CUDA tensor raises. This passed alone and failed
    # in the suite, exactly as test_mtclim_chunking.py once did.
    t = torch.tensor(v, dtype=torch.float64, device='cpu')
    b = snowbands.contract(t, ex).cpu().numpy()
    assert np.allclose(a, b, rtol=0, atol=1e-13)


def test_area_fractions_are_renormalised_when_they_do_not_sum_to_one():
    """vic_init.c:988-996, and the reason it is not optional here.

    165 of the Indian domain's 104,841 cells have band areas summing to
    less than one, the worst to 0.506. VIC divides each fraction by the sum
    and warns. Without it the "area-weighted mean" is not a mean at all: a
    field that is the same on every band contracts to 0.506 of itself, and
    every flux out of those cells is short by the same factor.
    """
    area = np.array([[0.2], [0.3]])             # sums to 0.5
    elev = np.array([[1000.0], [2000.0]])
    pf = np.array([[0.2], [0.3]])
    ex = snowbands.expand(area, elev, pf)
    assert np.allclose(ex['weight'], [0.4, 0.6])
    assert abs(float(ex['weight'].sum()) - 1.0) < 1e-15

    # The mean elevation uses the renormalised areas too.
    assert np.allclose(ex['mean_elev'], 0.4 * 1000.0 + 0.6 * 2000.0)

    # A uniform field survives.
    assert np.allclose(snowbands.contract(np.ones(2), ex), 1.0,
                       rtol=0, atol=1e-15)


def test_a_negative_area_fraction_is_refused():
    """vic_init.c:980-984 makes this log_err, which halts VIC."""
    with pytest.raises(ValueError):
        snowbands.expand(np.array([[1.2], [-0.2]]),
                         np.array([[1000.0], [2000.0]]),
                         np.array([[1.0], [0.0]]))


def test_vics_other_band_fatals_are_refused():
    """vic_init.c:1016 (no area at all), :1049 (a negative precipitation
    fraction) and :1058-1059 (precipitation on a band with no area) are
    all log_err, which halts VIC."""
    with pytest.raises(ValueError, match='no snow band area'):
        snowbands.expand(np.array([[0.0], [0.0]]),
                         np.array([[1000.0], [2000.0]]),
                         np.array([[0.0], [0.0]]))
    with pytest.raises(ValueError, match='negative snow band '
                                         'precipitation fraction'):
        snowbands.expand(np.array([[0.5], [0.5]]),
                         np.array([[1000.0], [2000.0]]),
                         np.array([[1.2], [-0.2]]))
    with pytest.raises(ValueError, match='no area carries precipitation'):
        snowbands.expand(np.array([[1.0], [0.0]]),
                         np.array([[1000.0], [2000.0]]),
                         np.array([[0.9], [0.1]]))
