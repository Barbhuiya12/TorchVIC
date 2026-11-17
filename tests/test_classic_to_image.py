"""The classic-to-image converter against VIC's own Stehekin sample.

VIC ships Stehekin twice: as classic ASCII files and as the image driver's
`Stehekin_test_params_20160327.nc`. Converting the first onto the grid of
`domain.stehekin.20151028.nc` must give the second, variable by variable,
with the names, dimensions, dtypes and fill values VIC reads. Three
differences are expected and each is pinned to its size:

* the bare-soil class: the classic driver derives roughness = rough,
  displacement = rough * 0.667 / 0.123 and wind_h = WIND_H for it
  (read_soilparam.c:489-494, read_veglib.c:225); the sample stores 0, 0 and
  2 m. With `bare_soil=BARE_SOIL_SAMPLE` these three are exact.
* Cv, <= 1e-6: three cells' tile covers sum to 0.999999 in the ASCII file.
  The classic driver rescales them to 1 (read_vegparam.c:628-645); the
  sample keeps them and gives bare soil the 1e-6 remainder.
* Pfactor, <= 3e-17: three cells' band fractions do not sum to 1; the
  classic driver divides by the sum (read_snowband.c), the sample does not.

With VIC 5.1.0's image driver present, the converted file (sample bare-soil
convention) is run against the official one: the fluxes and states agree to
the 1e-6 the Cv difference leaves.
"""
import glob
import os
import subprocess

import numpy as np
import pytest

netCDF4 = pytest.importorskip('netCDF4')
pytest.importorskip('xarray')

from torchvic.io.classic_to_image import BARE_SOIL_SAMPLE, cell_area, convert
from torchvic.io.classic import cell_area_vic
from torchvic.io.domain import ImageDomain
from torchvic.io.params import ImageParams

import _paths                                           # noqa: E402

CLASSIC = os.path.join(_paths.SAMPLES, 'classic', 'Stehekin', 'parameters')
IMAGE = os.path.join(_paths.SAMPLES, 'image', 'Stehekin', 'parameters')
OFFICIAL = os.path.join(IMAGE, 'Stehekin_test_params_20160327.nc')
DOMAIN = os.path.join(IMAGE, 'domain.stehekin.20151028.nc')

pytestmark = pytest.mark.skipif(
    not (os.path.exists(OFFICIAL) and os.path.isdir(CLASSIC)),
    reason='VIC Stehekin samples not present (tools/get_test_data.sh)')

#: The values the sample stores differently from the classic driver, and by
#: how much at most (see the module docstring).
BARE_ONLY = ('veg_rough', 'displacement', 'wind_h')
TOLERANCE = {'Cv': 1.0000001e-6, 'Pfactor': 3e-17}


def _convert(tmp, bare_soil=None, **kw):
    f = os.path.join(CLASSIC, 'Stehekin_%s.txt')
    out = dict(out_domain=os.path.join(tmp, 'domain.nc'),
               out_params=os.path.join(tmp, 'params.nc'))
    cv = convert(f % 'soil', f % 'vegparam', f % 'veglib', f % 'snowbands',
                 grid=DOMAIN, snow_bands=5, root_zones=3, vegparam_lai=True,
                 lai_src='FROM_VEGPARAM', wind_h=10.0, bare_soil=bare_soil,
                 log=lambda *a: None, **dict(out, **kw))
    return cv, out


@pytest.fixture(scope='module')
def converted(tmp_path_factory):
    return _convert(str(tmp_path_factory.mktemp('c2i')))


@pytest.fixture(scope='module')
def converted_sample(tmp_path_factory):
    return _convert(str(tmp_path_factory.mktemp('c2i_sample')),
                    bare_soil=BARE_SOIL_SAMPLE)


def _raw(v):
    v.set_auto_mask(False)
    return v[:]


def _diffs(path):
    a, b = netCDF4.Dataset(OFFICIAL), netCDF4.Dataset(path)
    out = {}
    try:
        missing = [v for v in a.variables if v not in b.variables]
        assert not missing, 'not written: %s' % missing
        for name, x in a.variables.items():
            y = b[name]
            assert y.dimensions == x.dimensions, name
            assert y.dtype == x.dtype, name
            if name in ('mask',):          # sample's mask has no _FillValue
                continue
            assert getattr(y, '_FillValue', None) == \
                getattr(x, '_FillValue', None), name
            if x.dtype == str:
                assert list(x[:]) == list(y[:]), name
                continue
            X, Y = _raw(x).astype(float), _raw(y).astype(float)
            out[name] = float(np.abs(X - Y).max())
    finally:
        a.close(), b.close()
    return out


def test_every_variable_matches_the_official_file(converted):
    d = _diffs(converted[1]['out_params'])
    for name, mx in d.items():
        if name in BARE_ONLY:
            continue
        assert mx <= TOLERANCE.get(name, 0.0), (name, mx)
    assert d['wind_h'] == 8.0                       # 10 m against 2 m
    assert 0 < d['veg_rough'] <= 0.01 + 1e-12       # rough, at most 0.01 m
    assert 0 < d['displacement'] <= 0.01 * 0.667 / 0.123 + 1e-12


def test_bare_soil_as_the_sample_stores_it(converted_sample):
    d = _diffs(converted_sample[1]['out_params'])
    bad = {k: v for k, v in d.items() if v > TOLERANCE.get(k, 0.0)}
    assert not bad, bad


def test_differences_are_the_classic_drivers_normalisation(converted):
    """Cv: the official file keeps the raw covers and gives bare soil the
    remainder; the converted file rescales as the classic driver does."""
    with netCDF4.Dataset(OFFICIAL) as a, \
            netCDF4.Dataset(converted[1]['out_params']) as b:
        A, B = a['Cv'][:].filled(np.nan), b['Cv'][:].filled(np.nan)
    np.testing.assert_allclose(np.nansum(B, 0)[~np.isnan(B[0])], 1.0,
                               atol=1e-15)
    cells = np.argwhere(np.abs(np.nan_to_num(A - B)).max(0) > 1e-9)
    assert len(cells) == 2                 # 85875 and 86337, raw sum 0.999999
    for y, x in cells:
        assert abs(A[-1, y, x] - 1e-6) < 1e-12 and B[-1, y, x] == 0.0
        np.testing.assert_allclose(B[:-1, y, x], A[:-1, y, x] / (1 - 1e-6),
                                   rtol=1e-12)


def test_nveg_satisfies_vics_tile_count(converted):
    """vic_init.c:1155-1166: classes with cover > 0, bare soil included,
    number Nveg or Nveg + 1."""
    with netCDF4.Dataset(converted[1]['out_params']) as b:
        cv = b['Cv'][:].filled(0)
        nveg = b['Nveg'][:]
    k = (cv > 0).sum(0)
    on = ~np.ma.getmaskarray(nveg)
    assert np.all((k[on] >= nveg[on]) & (k[on] <= nveg[on] + 1))


def test_domain(converted):
    cv, out = converted
    with netCDF4.Dataset(DOMAIN) as a, netCDF4.Dataset(out['out_domain']) as b:
        assert b['mask'].dtype == np.int32          # get_global_domain.c:46
        np.testing.assert_array_equal(b['mask'][:], a['mask'][:].filled(0))
        np.testing.assert_array_equal(b['lat'][:], a['lat'][:])
        np.testing.assert_array_equal(b['lon'][:], a['lon'][:])
        area, official = b['area'][:], a['area'][:]
        np.testing.assert_array_equal(b['frac'][:], b['mask'][:])
        lat2, lon2 = np.meshgrid(b['lat'][:], b['lon'][:], indexing='ij')
    np.testing.assert_array_equal(area, cell_area_vic(lat2, lon2, 0.125))
    # the sample's areas are cdo gridarea's (R = 6371 km)
    assert np.abs(area / official - 1).max() < 2e-4
    sph = cell_area(lat2, lon2, 0.125, 'sphere')
    assert np.abs(sph / official - 1).max() < 7e-5


def test_image_readers_take_the_converted_files(converted):
    """The repository's own image readers on the written files give the
    classic reader's arrays back."""
    cv, out = converted
    dom = ImageDomain(out['out_domain'], params=out['out_params'])
    par = ImageParams(out['out_params'], dom)
    assert dom.n_active == cv.n_cells == 16
    order = np.lexsort((cv.ix, cv.iy))       # ImageDomain is row-major
    for k, v in cv.params.soil.items():
        np.testing.assert_array_equal(par.soil[k], np.asarray(v)[..., order],
                                      err_msg=k)
    for k in ('Cv', 'LAI', 'albedo', 'root_depth', 'root_fract', 'rmin'):
        np.testing.assert_array_equal(par.veg[k],
                                      cv.params.veg[k][..., order], err_msg=k)


def test_cells_beyond_the_grid(tmp_path):
    """A run cell beyond the grid stops the conversion unless asked to be
    dropped, and the dropped cells are named."""
    import xarray as xr
    with xr.open_dataset(DOMAIN) as d:
        lat, lon = d['lat'].values[:3], d['lon'].values
    f = os.path.join(CLASSIC, 'Stehekin_%s.txt')
    kw = dict(snow_bands=5, root_zones=3, vegparam_lai=True,
              lai_src='FROM_VEGPARAM', log=lambda *a: None)
    with pytest.raises(ValueError, match='outside the grid'):
        convert(f % 'soil', f % 'vegparam', f % 'veglib', f % 'snowbands',
                grid=(lat, lon), **kw)
    cv = convert(f % 'soil', f % 'vegparam', f % 'veglib', f % 'snowbands',
                 grid=(lat, lon), outside='drop',
                 out_params=str(tmp_path / 'p.nc'), **kw)
    assert sorted(cv.dropped) == [87265, 87266, 87267]
    assert cv.n_cells == 13
    with netCDF4.Dataset(str(tmp_path / 'p.nc')) as b:
        assert int((b['run_cell'][:] == 1).sum()) == 13
        assert 87266 not in b['gridcell'][:].compressed()


def test_global_file_gives_the_same_files(converted, tmp_path):
    src = os.path.join(_paths.VIC_SYSTEM_TESTS, 'global.classic.STEHE.txt')
    if not os.path.exists(src):
        pytest.skip('VIC 5.1.0 system-test global file not present')
    g = tmp_path / 'global.txt'
    g.write_text(open(src).read().replace('$test_data_dir', _paths.SAMPLES))
    out = str(tmp_path / 'p.nc')
    convert(global_param=str(g), grid=DOMAIN, out_params=out,
            log=lambda *a: None)
    d = _diffs(out)
    ref = _diffs(converted[1]['out_params'])
    assert d == ref


@pytest.mark.skipif(not os.path.exists(_paths.VIC51_IMAGE),
                    reason='pristine VIC 5.1.0 image driver not built')
def test_vic_runs_the_converted_file_as_the_official_one(converted_sample,
                                                         tmp_path):
    src = os.path.join(_paths.VIC_SYSTEM_TESTS, 'global.image.STEHE.txt')
    if not os.path.exists(src):
        pytest.skip('VIC 5.1.0 system-test global file not present')

    def run(name, params):
        d = tmp_path / name
        d.mkdir()
        lines = []
        for ln in open(src).read().replace('$test_data_dir',
                                           _paths.SAMPLES).splitlines():
            key = ln.split()[0] if ln.split() else ''
            if key.startswith('STATE') or key == 'ROUT_PARAM':
                continue
            if key == 'PARAMETERS':
                ln = 'PARAMETERS %s' % params
            elif key == 'RESULT_DIR':
                ln = 'RESULT_DIR %s/' % d
            lines.append(ln)
        lines.append('LOG_DIR %s/' % d)
        (d / 'global.txt').write_text('\n'.join(lines) + '\n')
        env = dict(os.environ,
                   OMPI_MCA_btl_vader_single_copy_mechanism='none')
        r = subprocess.run([_paths.VIC51_IMAGE, '-g', str(d / 'global.txt')],
                           env=env, capture_output=True, text=True,
                           timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        return netCDF4.Dataset(glob.glob(str(d / 'fluxes*.nc'))[0])

    a = run('official', OFFICIAL)
    b = run('converted', converted_sample[1]['out_params'])
    n = 0
    for name, x in a.variables.items():
        if not name.startswith('OUT_'):
            continue
        X, Y = x[:].filled(np.nan), b[name][:].filled(np.nan)
        # the 1e-6 of cover on two cells is all that differs
        assert np.nanmax(np.abs(X - Y)) <= 1e-5, name
        n += 1
    assert n >= 15
