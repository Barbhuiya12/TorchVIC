"""One NetCDF handle per file (torchvic/io/ncshare.py).

netCDF-C closes with H5F_CLOSE_STRONG, so closing one of two datasets open on
the same file corrupts the other and the next open segfaults inside libhdf5.
The check runs in a child process because that is how the fault shows.
"""
import os
import subprocess
import sys

import pytest

netCDF4 = pytest.importorskip('netCDF4')
sys.path.insert(0, os.path.dirname(__file__))
import _paths                                            # noqa: E402

_CHILD = r'''
import gc, sys
import numpy as np
import xarray as xr
from torchvic.io.domain import _open
p, mode = sys.argv[1], sys.argv[2]
mk = {'plain': lambda: xr.open_dataset(p), 'shared': lambda: _open(p)}[mode]
name = [k for k in mk().data_vars if mk()[k].ndim >= 1][0]
ref = xr.open_dataset(p)[name].values.copy()
a = mk()
for i in range(6):
    b = mk()
    b.close(); del b; gc.collect()
    assert np.array_equal(a[name].values, ref, equal_nan=True), 'corrupted'
    with mk() as c:
        assert np.array_equal(c[name].values, ref, equal_nan=True)
print('ok')
'''


@pytest.fixture
def ncfile(tmp_path):
    import numpy as np
    p = str(tmp_path / 'f.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('x', 40)
        v = d.createVariable('v', 'f8', ('x',))
        v[:] = np.random.RandomState(0).rand(40) + 1.0
    return p


SAMPLE = os.path.join(_paths.SAMPLES, 'image', 'Stehekin', 'parameters',
                      'Stehekin_test_params_20160327.nc')


def _child(p, mode):
    return subprocess.run([sys.executable, '-c', _CHILD, p, mode],
                          capture_output=True, text=True)


@pytest.mark.parametrize('which', ['synthetic', 'stehekin'])
def test_shared_open_survives_close_of_a_second_open(ncfile, which):
    if which == 'stehekin':
        if not os.path.exists(SAMPLE):
            pytest.skip('VIC\'s Stehekin sample is not present')
        ncfile = SAMPLE
    r = _child(ncfile, 'shared')
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr[-800:]


def test_shared_handle_closes_with_its_last_copy(ncfile):
    import gc
    from torchvic.io import ncshare
    a, b = ncshare.open_shared(ncfile), ncshare.open_shared(ncfile)
    b.close()
    assert float(a['v'].values.min()) >= 1.0 and len(ncshare._OPEN) == 1
    del b
    a.close()
    gc.collect()
    assert not ncshare._OPEN


def test_read_leaves_no_handle_open(ncfile):
    from torchvic.io import ncshare
    ds = ncshare.read(ncfile)
    assert float(ds['v'].min()) >= 1.0 and not ncshare._OPEN


def _same(a, b):
    import xarray as xr
    xr.testing.assert_identical(a, b)
    assert list(a.data_vars) == list(b.data_vars)
    assert a.encoding == b.encoding
    for k in a.variables:
        assert a[k].dtype == b[k].dtype, k
        assert a[k].encoding == b[k].encoding, k


def test_preread_equals_read_and_leaves_no_handle_open():
    """parallel.preread inflates the parameter file in a forked pool into
    shared buffers; the dataset is the one io.ncshare.read returns."""
    if not os.path.exists(SAMPLE):
        pytest.skip('VIC\'s Stehekin sample is not present')
    from torchvic.io import ncshare
    from torchvic.parallel import preread
    ref = ncshare.read(SAMPLE)
    full = preread.read_full(SAMPLE, procs=3)
    assert len(preread.wanted_names(ref)) > 10
    _same(full, ref)
    assert not ncshare._OPEN and not preread._BUF
    with preread.preloaded(SAMPLE, procs=3) as ds:
        _same(ds, ref)
        got = ncshare.read(SAMPLE)
        _same(got, ref)
        assert got['LAI'].values is not None
    assert not ncshare._MEM and not ncshare._OPEN


def test_preread_params_equal_serial_read():
    if not os.path.exists(SAMPLE):
        pytest.skip('VIC\'s Stehekin sample is not present')
    import numpy as np
    from torchvic.io.domain import ImageDomain
    from torchvic.io.params import ImageParams
    from torchvic.parallel import preread
    dom = ImageDomain(SAMPLE, params=SAMPLE)
    ref = ImageParams(SAMPLE, dom)
    with preread.preloaded(SAMPLE, dom, procs=4):
        par = ImageParams(SAMPLE, dom)
    for grp in ('soil', 'veg', 'bands'):
        a, b = getattr(ref, grp), getattr(par, grp)
        assert sorted(a) == sorted(b)
        for k in a:
            assert np.asarray(a[k]).dtype == np.asarray(b[k]).dtype, k
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]),
                                  equal_nan=True), k
