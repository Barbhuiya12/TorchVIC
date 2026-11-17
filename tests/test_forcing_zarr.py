"""Zarr forcing: a store gives ImageForcing's NetCDF values bit for bit.

Same variables, units and conversions as the NetCDF path (one code path
after the open); one store or per-year stores; `plan`/`read_steps` and the
prefetch thread read time blocks from it; a store on another grid is refused.
"""
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

pytest.importorskip('zarr')
from torchvic.io.forcing import ImageForcing, WindowPrefetch   # noqa: E402


class _Dom(object):
    y_dim, x_dim = 'lat', 'lon'

    def __init__(self, mask):
        self.ny, self.nx = mask.shape
        self.rows, self.cols = np.nonzero(mask)
        self.n_active = int(self.rows.size)
        self.lat = 30.0 + self.rows * 0.0625
        self.lon = 70.0 + self.cols * 0.0625

    def check_dims(self, da, what=None):
        assert tuple(da.dims[-2:]) == ('lat', 'lon'), what


FT = {'TMAX': 'tasmax', 'TMIN': 'tasmin', 'PREC': 'prcp', 'VP': 'vp',
      'PRESSURE': 'sp'}
UNITS = {'tasmax': 'degC', 'tasmin': 'degC', 'prcp': 'mm', 'vp': 'kPa',
         'sp': 'kPa'}


def _ds(year, n, ny=4, nx=5, seed=0):
    rng = np.random.default_rng(seed + year)
    t = pd.date_range('%d-01-01' % year, periods=n, freq='D')
    ds = xr.Dataset(coords=dict(time=t, lat=30.0 + np.arange(ny) * 0.0625,
                                lon=70.0 + np.arange(nx) * 0.0625))
    for v, u in UNITS.items():
        a = rng.normal(20, 8, (n, ny, nx)).astype(np.float32)
        a[:, 0, 0] = np.nan
        ds[v] = (('time', 'lat', 'lon'), a, {'units': u})
    return ds


def _make(tmp_path):
    mask = np.ones((4, 5), bool)
    mask[0, 0] = mask[2, 3] = False
    d3, d4 = _ds(2003, 20), _ds(2004, 30)
    for y, d in ((2003, d3), (2004, d4)):
        d.to_netcdf(tmp_path / ('f.%d.nc' % y))
        d.to_zarr(tmp_path / ('z.%d.zarr' % y))
    xr.concat([d3, d4], 'time').to_zarr(tmp_path / 'all.zarr')
    return _Dom(mask)


@pytest.mark.parametrize('zpath', ['z.', 'all.zarr'])
def test_zarr_read_equals_netcdf(tmp_path, zpath):
    dom = _make(tmp_path)
    ref, tref = ImageForcing(str(tmp_path / 'f.'), dom, force_type=FT).read()
    got, tgot = ImageForcing(str(tmp_path / zpath), dom, force_type=FT).read()
    assert np.array_equal(tref, tgot)
    assert sorted(ref) == sorted(got)
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    assert torch.allclose(got['press'], torch.as_tensor(got['press']))
    assert float(got['press'].max()) > 1e4          # kPa -> Pa applied


def test_zarr_years_and_blocks(tmp_path):
    dom = _make(tmp_path)
    fc = ImageForcing(str(tmp_path / 'all.zarr'), dom, force_type=FT,
                      years=[2004])
    whole, t = fc.read()
    assert len(t) == 30
    ref, _ = ImageForcing(str(tmp_path / 'f.'), dom, force_type=FT,
                          years=[2004]).read()
    for k in ref:
        assert torch.equal(ref[k], whole[k])
    fc = ImageForcing(str(tmp_path / 'all.zarr'), dom, force_type=FT)
    full, _ = fc.read()
    fc.plan()
    for a, b in ((0, 7), (7, 33), (33, 50)):
        blk = fc.read_steps(a, b)
        for k in full:
            assert np.array_equal(blk[k], full[k][a:b].numpy())
    pf = WindowPrefetch(fc, [(0, 20), (20, 50)], list(full),
                        torch.device('cpu'), torch.float64)
    got = [w for w in pf]
    pf.close()
    assert len(got) == 2
    fc.close()


def test_zarr_other_grid_refused(tmp_path):
    dom = _make(tmp_path)
    d = _ds(2003, 5)
    d = d.assign_coords(lat=d.lat + 1.0)
    d.to_zarr(tmp_path / 'bad.zarr')
    with pytest.raises(ValueError, match='another grid'):
        ImageForcing(str(tmp_path / 'bad.zarr'), dom, force_type=FT).read()


def test_zarr_missing_units_refused_like_netcdf(tmp_path):
    dom = _make(tmp_path)
    d = _ds(2003, 5)
    d['sp'].attrs.pop('units')
    d.to_zarr(tmp_path / 'nu.zarr')
    with pytest.raises(ValueError, match='no units attribute'):
        ImageForcing(str(tmp_path / 'nu.zarr'), dom, force_type=FT).read()


def test_era5_daily_default_mapping(tmp_path):
    dom = _make(tmp_path)
    d = _ds(2003, 5)
    for v in ('wind', 'swdown', 'lwdown'):
        d[v] = (('time', 'lat', 'lon'),
                np.ones((5, 4, 5), np.float32), {'units': 'W m-2' if v != 'wind'
                                                  else 'm s-1'})
    d.to_zarr(tmp_path / 'era.zarr')
    got, _ = ImageForcing(str(tmp_path / 'era.zarr'), dom).read()
    assert {'tmax', 'tmin', 'prcp', 'vp', 'press', 'wind', 'shortwave',
            'longwave'} <= set(got)
