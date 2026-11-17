"""GDW/GRanD reader and placement on our flow map."""
import os
import _paths  # noqa: E402
import numpy as np
import pytest

from torchvic.reservoir import read_gdw_dams, place_dams, merge_by_cell

ROOT = _paths.DATA_ROOT
DBF = ROOT + '/water_use/dams/GDW_barriers_v1_delta.dbf'
FDIR = ROOT + '/static/isc_fdir.nc'


def test_place_by_upstream_area_synthetic():
    lat, lon = np.arange(10) * 0.0625, np.arange(10) * 0.0625
    A = np.ones((10, 10))
    A[5, 5], A[5, 6], A[6, 5] = 1000.0, 50.0, 400.0     # main stem and tributaries
    dams = dict(lat=np.array([lat[5], lat[5], lat[2]]), lon=np.array([lon[6], lon[5], lon[2]]),
                catch_km2=np.array([900.0, 1100.0, 1e6]), cap_mcm=np.array([10., 20., 5.]),
                year=np.array([1980., np.nan, 2000.]), area_skm=np.array([1., 2., 1.]),
                irrigation=np.array([True, False, False]))
    iy, ix, ok = place_dams(dams, lat, lon, A, radius=2)
    assert list(ok) == [True, True, False]               # third has no matching cell
    assert (iy[0], ix[0]) == (5, 5) and (iy[1], ix[1]) == (5, 5)
    y, x, cap, yr, irr, area, n = merge_by_cell(dams, iy, ix, ok, year_default=1900)
    assert len(y) == 1 and n[0] == 2 and cap[0] == 30e6 and yr[0] == 1900 and irr[0]


@pytest.mark.skipif(not (os.path.exists(DBF) and os.path.exists(FDIR)), reason='data')
def test_gdw_india_on_isc_flow_map():
    xr = pytest.importorskip('xarray')
    d = read_gdw_dams(DBF, min_capacity_mcm=10)
    assert len(d['lat']) > 1000 and np.all(d['cap_mcm'] >= 10)
    g = xr.open_dataset(FDIR)
    iy, ix, ok = place_dams(d, g.lat.values, g.lon.values, g.upstream_area.values)
    assert ok.mean() > 0.7
    # capacity-weighted placement success is higher (large dams have large catchments)
    assert d['cap_mcm'][ok].sum() > 0.9 * d['cap_mcm'].sum()
    assert np.all(np.isfinite(g.upstream_area.values[iy[ok], ix[ok]]))
