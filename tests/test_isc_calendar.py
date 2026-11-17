"""ISC calendar builder: monthly tile areas keep the MIRCA-OS area-months."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'vicwur', 'isc'))
import make_calendar as mc  # noqa: E402


def _punjab():
    """Rice in thirds (Apr-Jul, Aug-Nov, Dec-Mar), wheat in halves."""
    a_pad = np.zeros((12, 1, 1))
    a_np = np.zeros((12, 1, 1))
    rice = 0.30 / 3.0
    for ms in ([4, 5, 6, 7], [8, 9, 10, 11], [12, 1, 2, 3]):
        for m in ms:
            a_pad[m - 1] += rice
    for ms in ([6, 7, 8, 9, 10], [11, 12, 1, 2, 3]):
        for m in ms:
            a_np[m - 1] += 0.40 / 2.0
    return a_np, a_pad


def test_tile_months_area_months_follow_source():
    a_np, a_pad = _punjab()
    cv_pad = np.array([[0.30]])          # annual shares (sum of sub-crops)
    cv_irr = np.array([[0.40]])
    irr, pad, freed = mc.tile_months(cv_irr, cv_pad, a_np, a_pad)
    land = 0.70
    peak = (a_np + a_pad).sum(1).max()
    # the land grows the same share of its peak as the source does
    np.testing.assert_allclose((irr + pad)[:, 0, 0],
                               land * (a_np + a_pad)[:, 0, 0] / peak, rtol=1e-12)
    # paddy keeps its source share of the growing area every month (not 100 %
    # of its annual Cv in all 12 months as with per-tile maxima)
    np.testing.assert_allclose(pad[:, 0, 0] / (irr + pad)[:, 0, 0],
                               a_pad[:, 0, 0] / (a_np + a_pad)[:, 0, 0], rtol=1e-12)
    assert pad[:, 0, 0].max() < 0.30 - 0.05   # old: 0.30 in all 12 months
    np.testing.assert_allclose(irr + pad + freed, land, rtol=1e-12)
    assert (freed >= -1e-15).all()


def test_tile_months_tile_must_exist_and_no_crop_keeps_year():
    a_np, a_pad = _punjab()
    # no paddy tile in the params: the share goes to the irrigated tile
    irr, pad, freed = mc.tile_months(np.array([[0.4]]), np.array([[0.0]]), a_np, a_pad)
    assert (pad == 0).all() and (irr > 0).any()
    # no irrigated crop in the cell: yearly tiles unchanged, nothing freed
    z = np.zeros((12, 1, 1))
    irr, pad, freed = mc.tile_months(np.array([[0.2]]), np.array([[0.1]]), z, z)
    np.testing.assert_allclose(irr[:, 0, 0], 0.2)
    np.testing.assert_allclose(pad[:, 0, 0], 0.1)
    assert np.abs(freed).max() < 1e-15


def test_season_months_wrap_and_missing():
    # Nov 16 (day 320) -> Mar 15 (day 74): wraps the year end
    sh = mc.season_months(np.array([320.0, np.nan, 0.0]), np.array([74.0, 100.0, 50.0]))
    assert sh.shape == (12, 3)
    assert np.isnan(sh[:, 1:]).all()               # NaN and day 0 = no calendar
    days = (sh[:, 0] * mc._DM).sum()
    assert abs(days - ((74 - 320) % 365 + 1)) < 1e-9   # both end days included
    assert sh[0, 0] == 1 and sh[1, 0] == 1 and sh[6, 0] == 0   # Jan, Feb in; Jul out
    assert 0 < sh[10, 0] < 1 and 0 < sh[2, 0] < 1               # partial Nov, Mar


def test_seasonal_area_keeps_annual_area_and_falls_back():
    a = np.zeros((12, 1, 3))
    a[3:7] = 2.0                                    # MIRCA: 2 units, Apr-Jul
    ggcmi = mc.season_months(np.array([[182.0, np.nan, 182.0]]),
                             np.array([[304.0, np.nan, 304.0]]))   # Jul-Oct
    new = mc.seasonal_area(a, [ggcmi])
    np.testing.assert_allclose(new.max(0), 2.0)     # annual area unchanged
    np.testing.assert_allclose(new[:, 0, 1], a[:, 0, 1])   # no GGCMI: MIRCA months
    assert new[3, 0, 0] == 0 and new[8, 0, 0] == 2.0       # months moved to Jul-Oct
    # second choice used where the first has no value
    alt = mc.season_months(np.array([[1.0, 1.0, 1.0]]), np.array([[60.0, 60.0, 60.0]]))
    new2 = mc.seasonal_area(a, [ggcmi, alt])
    assert new2[0, 0, 1] > 0 and new2[8, 0, 1] == 0
    # crop with no GGCMI calendar: unchanged
    np.testing.assert_array_equal(mc.seasonal_area(a, []), a)


@pytest.mark.skipif(not os.path.isdir(mc.GGCMI),
                    reason='GGCMI calendar data absent: set TORCHVIC_DATA_ROOT')
def test_crop_map_covers_every_mirca_crop():
    names = [mc.crop_name('/x/MIRCA-OS_%s_2005_ir.nc' % n) for n in
             ('Rice_1', 'Others_annual_4', 'Sugar_cane', 'Oil_palm', 'Wheat_2')]
    assert names == ['Rice_1', 'Others_annual_4', 'Sugar_cane', 'Oil_palm', 'Wheat_2']
    assert set(mc.CROP_MAP) == {
        'Barley', 'Cassava', 'Cocoa', 'Coffee', 'Cotton', 'Fodder', 'Groundnuts',
        'Maize', 'Millet', 'Oil_palm', 'Others_annual_1', 'Others_annual_2',
        'Others_annual_3', 'Others_annual_4', 'Others_perennial', 'Potatoes',
        'Pulses', 'Rapeseed', 'Rice_1', 'Rice_2', 'Rice_3', 'Rye', 'Sorghum',
        'Soybeans', 'Sugar_beet', 'Sugar_cane', 'Sunflower', 'Wheat_1', 'Wheat_2'}
    for codes in mc.CROP_MAP.values():
        for c in codes:
            assert os.path.exists(mc.GGCMI_FILE % (mc.GGCMI, c)), c


@pytest.mark.skipif(not os.path.isdir(mc.GGCMI),
                    reason='GGCMI calendar data absent: set TORCHVIC_DATA_ROOT')
def test_ggcmi_punjab_rice_is_kharif():
    lat, lon = np.array([31.0]), np.array([75.5])       # Punjab
    sh = mc.ggcmi_months('ri1', lat, lon)[:, 0, 0]
    assert sh[6:10].min() == 1 and sh[0:4].max() == 0   # Jul-Oct in, Jan-Apr out


def _rice_inputs():
    ri1 = mc.season_months(np.array([[180.0, 180.0]]), np.array([[300.0, 300.0]]))   # Jul-Oct
    ri2 = mc.season_months(np.array([[320.0, 320.0]]), np.array([[74.0, 74.0]]))     # Nov-Mar
    third = []
    for ms in ([4, 5, 6, 7], [8, 9, 10, 11], [12, 1, 2, 3]):     # MIRCA equal thirds
        a = np.zeros((12, 1, 2))
        a[[m - 1 for m in ms]] = 1.0
        third.append(a)
    return third, ri1, ri2


def test_rice_one_season_outside_second_season_states():
    subs, ri1, ri2 = _rice_inputs()
    out = mc.rice_seasonal(subs, ri1, ri2, np.array([[False, False]]))
    np.testing.assert_allclose(out.max(0), 3.0 * ri1.max(0))        # all rice area in ri1
    np.testing.assert_allclose(out, 3.0 * ri1)                       # no winter rice


def test_rice_second_season_share_in_listed_state_only():
    subs, ri1, ri2 = _rice_inputs()
    out = mc.rice_seasonal(subs, ri1, ri2, np.array([[True, False]]))
    assert out[0, 0, 0] > 0 and out[0, 0, 1] == 0                    # Jan rice only where listed
    # the Dec-Mar sub-crop (1/3 of the area) overlaps ri2 more than ri1
    np.testing.assert_allclose(out[0, 0, 0], 3.0 * (1 / 3))
    # no ri2 value: ri1 only even in a listed state
    ri2n = np.full_like(ri2, np.nan)
    out = mc.rice_seasonal(subs, ri1, ri2n, np.array([[True, True]]))
    np.testing.assert_allclose(out, 3.0 * ri1)
    # no GGCMI rice at all: MIRCA months stay
    out = mc.rice_seasonal(subs, np.full_like(ri1, np.nan), ri2n, np.array([[True, True]]))
    np.testing.assert_allclose(out, sum(subs))


@pytest.mark.skipif(not os.path.isdir(mc.GGCMI),
                    reason='GGCMI calendar data absent: set TORCHVIC_DATA_ROOT')
def test_rice2_mask_states():
    pytest.importorskip('matplotlib', reason='rice2_mask uses matplotlib.path.Path')
    m = mc.rice2_mask(np.array([31.0, 22.8, 10.8]), np.array([75.5, 88.0, 79.0]))
    assert not m[0, 0] and m[1, 1] and m[2, 2]       # Punjab no; West Bengal, Tamil Nadu yes
