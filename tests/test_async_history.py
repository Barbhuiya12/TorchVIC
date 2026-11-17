"""The history writer that runs beside the model (io.async_history) writes
the files history.write_record writes, value for value, whatever the way
the records are cut into windows; and the reduced-parameter cache (io.params)
gives the parameters the file gives."""
import datetime as dt
import glob
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')
xr = pytest.importorskip('xarray')

import _paths                                             # noqa: E402,F401
from torchvic.io import history as H                      # noqa: E402
from torchvic.io.async_history import AsyncStreamWriter   # noqa: E402


class _GP(object):
    path = 'synthetic'

    def __init__(self, lines):
        self.lines = lines

    def get(self, k):
        return None


class _Dom(object):
    names = dict(y_dim='lat', x_dim='lon', lat_var='lat', lon_var='lon')
    ny, nx = 3, 4

    def __init__(self):
        self.rows = np.array([0, 0, 1, 2, 2])
        self.cols = np.array([0, 3, 1, 0, 2])
        self.ds = xr.Dataset(coords=dict(lat=('lat', [10.0, 11.0, 12.0]),
                                         lon=('lon', [70.0, 71, 72, 73])))


def _setup(hist):
    lines = [('OUTFILE', ['fluxes']), ('AGGFREQ', ['NDAYS', '1']),
             ('HISTFREQ', hist), ('OUTVAR', ['OUT_PREC']),
             ('OUTVAR', ['OUT_SOIL_MOIST']),
             ('OUTFILE', ['mon']), ('AGGFREQ', ['NDAYS', '2']),
             ('OUTVAR', ['OUT_EVAP'])]
    gp = _GP(lines)
    return gp, H.parse_streams(gp)


@pytest.mark.parametrize('hist,cuts', [
    (['NMONTHS', '1'], [31, 31, 28]),
    (['NDAYS', '30'], [10, 40, 40]),
    (['NYEARS', '1'], [90]),
])
def test_async_equals_write_record(tmp_path, hist, cuts):
    gp, streams = _setup(hist)
    dom = _Dom()
    n_days, spd = sum(cuts), 8
    t0 = dt.datetime(2004, 1, 1)
    dates = [t0 + dt.timedelta(hours=3 * i) for i in range(n_days * spd)]
    g = torch.Generator().manual_seed(1)
    nc = len(dom.rows)
    full = {'fluxes': {'OUT_PREC': torch.rand(n_days, nc, generator=g,
                                              dtype=torch.float64),
                       'OUT_SOIL_MOIST': torch.rand(n_days, nc, 3,
                                                    generator=g,
                                                    dtype=torch.float64)},
            'mon': {'OUT_EVAP': torch.rand(n_days // 2, nc, generator=g,
                                           dtype=torch.float64)}}
    # the reference: one Record, one call
    ref_dir = str(tmp_path / 'ref')
    for s in streams:
        recs, tail = H.windows(dates, 10800, s.agg)
        data = full[s.prefix]
        rec = H.Record(s, s.agg, recs, tail, H.as_datetimes(dates), 10800,
                       data, {k: 'x' for k in data}, {}, [])
        H.write_record(rec, dom, ref_dir, gp=gp, log=lambda *a: None)
    # the same records, handed over window by window
    out = str(tmp_path / 'async')
    w = AsyncStreamWriter(streams, dates, spd, dom, gp, out, source='x')
    r0 = 0
    for c in cuts:
        w.add('fluxes', {k: v[r0:r0 + c] for k, v in full['fluxes'].items()})
        w.add('mon', {k: v[r0 // 2:(r0 + c) // 2]
                      for k, v in full['mon'].items()})
        r0 += c
    paths = w.close()
    ref = sorted(glob.glob(os.path.join(ref_dir, '*.nc')))
    got = sorted(glob.glob(os.path.join(out, '*.nc')))
    assert [os.path.basename(p) for p in ref] == \
        [os.path.basename(p) for p in got]
    assert sorted(paths) == got and got
    for a, b in zip(ref, got):
        x = xr.open_dataset(a, decode_times=False, mask_and_scale=False)
        y = xr.open_dataset(b, decode_times=False, mask_and_scale=False)
        assert list(x.variables) == list(y.variables)
        for k in x.variables:
            assert x[k].dtype == y[k].dtype and x[k].dims == y[k].dims, k
            assert np.array_equal(x[k].values, y[k].values), k
            ea = {kk: vv for kk, vv in x[k].attrs.items()}
            eb = {kk: vv for kk, vv in y[k].attrs.items()}
            assert ea == eb, k
        x.close()
        y.close()


def test_async_writer_error_is_raised(tmp_path):
    gp, streams = _setup(['NMONTHS', '1'])
    dom = _Dom()
    dates = [dt.datetime(2004, 1, 1) + dt.timedelta(hours=3 * i)
             for i in range(10 * 8)]
    w = AsyncStreamWriter(streams, dates, 8, dom, gp, str(tmp_path / 'o'),
                          source='x')
    bad = {'OUT_PREC': torch.zeros(10, 7, dtype=torch.float64),
           'OUT_SOIL_MOIST': torch.zeros(10, 7, 3, dtype=torch.float64)}
    w.add('fluxes', bad)               # 7 cells for a 5-cell domain
    with pytest.raises(RuntimeError, match='history writer failed'):
        w.close()
