"""Reading the image forcing a window at a time gives `read`'s values, and
the GPU's deterministic sums give the CPU's.

`ImageForcing.plan` + `read_steps` (and `WindowPrefetch`, which runs them on
a background thread) let run_isc_torchvic.py read the next window while the
current one computes; they must hand the model exactly what `read` does,
across a file (year) boundary, with the unit scales applied in the file's
dtype. `history.aggregate` and `routing.route` sum with segment_reduce on
the GPU, which adds in the CPU index_add's order, so they repeat bit for
bit (and the aggregation equals the CPU's).
"""
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from torchvic.io import history as H
from torchvic.io.forcing import ImageForcing, WindowPrefetch
from torchvic.model import routing


class _Dom(object):
    y_dim, x_dim = 'lat', 'lon'

    def __init__(self, mask):
        self.rows, self.cols = np.nonzero(mask)
        self.n_active = int(self.rows.size)

    def check_dims(self, da, what=None):
        assert tuple(da.dims[-2:]) == ('lat', 'lon'), what


FT = {'AIR_TEMP': 'air_temp', 'PREC': 'prec', 'PRESSURE': 'pressure',
      'VP': 'vp'}
UNITS = {'air_temp': 'C', 'prec': 'mm/step', 'pressure': 'kPa', 'vp': 'kPa'}


def _files(tmp_path, ny=4, nx=5):
    rng = np.random.default_rng(3)
    for year, n in ((2003, 20), (2004, 28)):
        t = pd.date_range('%d-01-01' % year, periods=n, freq='3h')
        ds = xr.Dataset(coords=dict(time=t, lat=np.arange(ny) * 1.0,
                                    lon=np.arange(nx) * 1.0))
        for v, u in UNITS.items():
            a = rng.normal(50, 30, (n, ny, nx)).astype(np.float32)
            a[:, 0, 0] = np.nan                       # an inactive fill
            ds[v] = (('time', 'lat', 'lon'), a, {'units': u})
            ds[v].encoding['_FillValue'] = np.float32(9.96921e36)
        ds.to_netcdf(tmp_path / ('f.%d.nc' % year))
    mask = np.ones((ny, nx), bool)
    mask[0, 0] = mask[2, 3] = False
    return str(tmp_path / 'f.'), _Dom(mask)


def test_read_steps_and_prefetch_give_reads_values(tmp_path):
    prefix, dom = _files(tmp_path)
    whole, stamps = ImageForcing(prefix, dom, force_type=FT).read(
        dtype=torch.float64)
    fc = ImageForcing(prefix, dom, force_type=FT)
    t2 = fc.plan()
    assert np.array_equal(stamps, t2)
    spans = [(0, 7), (7, 19), (19, 33), (33, 48)]     # (19, 33) spans files
    for a, b in spans:
        blk = fc.read_steps(a, b)
        for k, v in whole.items():
            got = torch.from_numpy(blk[k]).to(torch.float64)
            assert torch.equal(got, v[a:b]), (k, a, b)
    got = list(WindowPrefetch(fc, spans, sorted(whole), 'cpu',
                              torch.float64))
    for (a, b), f in zip(spans, got):
        for k, v in whole.items():
            assert torch.equal(f[k], v[a:b]), (k, a, b)
    fc.close()


def _sum_cases():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 7, 3, dtype=torch.float64, generator=g) * 1e3
    regular = [(i, i + 7) for i in range(0, 40, 8)]
    ragged = [(0, 2), (3, 12), (13, 13), (14, 39)]
    return x, (regular, ragged)


def test_segment_sums_are_index_adds_order():
    # what the GPU branch relies on, checked on the CPU: segment_reduce
    # adds each segment from 0 in order, as index_add does
    x, cases = _sum_cases()
    for recs in cases:
        length = torch.tensor([l - f + 1 for f, l in recs])
        steps = torch.cat([torch.arange(f, l + 1) for f, l in recs])
        wid = torch.repeat_interleave(torch.arange(len(recs)), length)
        ref = torch.zeros((len(recs),) + x.shape[1:],
                          dtype=x.dtype).index_add(0, wid, x[steps])
        seg = torch.segment_reduce(x[steps], 'sum', lengths=length, axis=0)
        assert torch.equal(seg, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')
def test_gpu_sums_equal_the_cpus_and_repeat():
    x, cases = _sum_cases()
    for recs in cases:
        for agg in ('SUM', 'AVG'):
            cpu = H.aggregate(x, recs, agg)
            for _ in range(3):
                assert torch.equal(H.aggregate(x.cuda(), recs, agg).cpu(), cpu)
    g = torch.Generator().manual_seed(1)
    ro = torch.rand(30, 500, dtype=torch.float64, generator=g)
    uh = torch.rand(6, 500, dtype=torch.float64, generator=g)
    for s2o, n in ((torch.zeros(500, dtype=torch.long), 1),
                   (torch.randint(0, 7, (500,), generator=g), 9)):
        # the convolution itself is cuDNN's on the GPU, so GPU and CPU
        # agree to rounding; the GPU's outlet sums repeat bit for bit
        cpu = routing.route(ro, uh, s2o, n)
        gpu = routing.route(ro.cuda(), uh.cuda(), s2o.cuda(), n)
        assert torch.allclose(gpu.cpu(), cpu, rtol=1e-13, atol=0)
        for _ in range(3):
            again = routing.route(ro.cuda(), uh.cuda(), s2o.cuda(), n)
            assert torch.equal(again, gpu)
