# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.forcing against the netCDF values of reference domain R1
(tools/vicwur/ref_domains.py; data in <data root>/vicwur/R1/forcing).

Expected values are read straight from the files with the cadence rule
written out from the date alone: DAY -> day of year, MONTH -> month, YEAR ->
record 0, held between reads (plugin_forcing_utils.c:168-257)."""
import datetime
import os
import _paths  # noqa: E402

import numpy as np
import pytest
import torch

from torchvic.io import ncshare  # noqa: E402
from torchvic.vicwur import forcing as F  # noqa: E402

FDIR = _paths.env_path(('TORCHVIC_VICWUR_R1_FORCING', 'VICWUR_R1_FORCING'),
                      'vicwur/R1/forcing')
pytestmark = pytest.mark.skipif(not os.path.isdir(FDIR),
                                reason='R1 forcing data absent')

START = (2003, 7, 1, 0)
NDAYS = 3 * 365 + 366 + 40          # to Aug 2007: crosses 2004 leap year


_OPEN = []


@pytest.fixture(autouse=True)
def _close_forcings():
    """Join every prefetch thread before the next test: a thread still
    reading a year file while another test opens a NetCDF file corrupts the
    shared HDF5 state (segfault / HDF error in later tests of one process)."""
    yield
    while _OPEN:
        _OPEN.pop().close()


def PF(prefix, varname, freq, start, nsteps, spd=1, **kw):
    """PluginForcing with met_start = the file's own first time stamp (the
    met forcing lined up with the plugin file); calendar from the file."""
    if 'met_start' not in kw:
        kw['met_start'] = F.file_info('%s%4d.nc' % (prefix, start[0]),
                                      varname)['start']
    f = F.PluginForcing(prefix, varname, freq, start, nsteps, spd, **kw)
    _OPEN.append(f)
    return f


def raw(name, var, year, rec):
    d = ncshare.read('%s/%s.%d.nc' % (FDIR, name, year),
                     decode_times=False, mask_and_scale=False)
    a = np.asarray(d[var].values[rec])
    return a.reshape(a.shape[:-2] + (-1,))


def day0(i, spd=1):
    d = datetime.date(2003, 7, 1) + datetime.timedelta(days=i // spd)
    return d


def sample_steps(n, spd):
    s = {0, 1, 2, spd, n - 1}
    for k in range(0, n, 37):
        s.add(k)
    for d in (datetime.date(2003, 12, 31), datetime.date(2004, 1, 1),
              datetime.date(2004, 2, 29), datetime.date(2004, 3, 1),
              datetime.date(2004, 12, 31), datetime.date(2005, 1, 1),
              datetime.date(2003, 7, 31), datetime.date(2003, 8, 1)):
        i = (d - datetime.date(2003, 7, 1)).days * spd
        s.update({i, i + spd - 1})
    return sorted(k for k in s if 0 <= k < n)


@pytest.mark.parametrize('spd', [1, 4])
def test_daily_sector_demand_held_by_day(spd):
    n = NDAYS * spd
    f = PF(FDIR + '/mun_demand.', 'mun_demand', 'DAY', START, n,
                        spd)
    for k in sample_steps(n, spd):
        d = day0(k, spd)
        want = raw('mun_demand', 'mun_demand', d.year, d.timetuple().tm_yday - 1)
        got = f.get(k).numpy()
        assert np.array_equal(got, want), (k, d)
        assert f.reads(k) == (k == 0 or k % spd == 0)
    assert f.n_loads == 5            # 2003..2007, each file read once
    assert f.skip0 == 181            # round((Jul 1 - Jan 1) * 365/365)


def test_monthly_cv_and_efr_held_by_month():
    n = NDAYS
    for name, var in (('cv', 'cv'), ('efr_q', 'efr_q'), ('efr_b', 'efr_b')):
        f = PF(FDIR + '/%s.' % name, var, 'MONTH', START, n, 1)
        assert f.skip0 == 6          # start_month - file_month
        for k in sample_steps(n, 1):
            d = day0(k)
            want = raw(name, var, d.year, d.month - 1)
            assert np.array_equal(f.get(k).numpy(), want), (name, k, d)
            assert f.reads(k) == (k == 0 or d.day == 1)


def test_yearly_co2_record0_and_year_change_read():
    n = NDAYS
    f = PF(FDIR + '/co2.', 'co2', 'YEAR', START, n, 1)
    assert f.skip0 == 0
    for k in sample_steps(n, 1):
        d = day0(k)
        assert np.array_equal(f.get(k).numpy(),
                              raw('co2', 'co2', d.year, 0)), (k, d)
        assert f.reads(k) == (k == 0 or (d.month, d.day) == (1, 1))
    assert (f.get(0) == 375.0).all() and (f.get(n - 1) == 383.0).all()


def test_start_on_a_leap_day_and_active_cell_subset():
    st = (2004, 3, 1, 0)
    f = PF(FDIR + '/mun_demand.', 'mun_demand', 'DAY', st, 400, 1,
                        cells=[1, 4])
    assert f.skip0 == 60             # Jan 31 + Feb 29
    d = datetime.date(2004, 3, 1) + datetime.timedelta(days=300)
    want = raw('mun_demand', 'mun_demand', d.year, d.timetuple().tm_yday - 1)
    assert np.array_equal(f.get(300).numpy(), want[[1, 4]])


def test_no_interpolation_inside_a_month():
    f = PF(FDIR + '/cv.', 'cv', 'MONTH', START, 62, 1)
    a, b = f.get(10), f.get(25)      # both July 2003
    assert torch.equal(a, b)
    assert not torch.equal(f.get(30), f.get(31))   # Jul 31 vs Aug 1


def test_negative_cases():
    with pytest.raises(ValueError, match='time'):       # misaligned met start
        PF(FDIR + '/cv.', 'cv', 'MONTH', START, 10, 1,
                        met_start=(2003, 7, 1, 0))
    with pytest.raises(ValueError, match='met_start is required'):
        F.PluginForcing(FDIR + '/cv.', 'cv', 'MONTH', START, 10, 1,
                        met_start=None)
    with pytest.raises(TypeError):                      # no met_start at all
        F.PluginForcing(FDIR + '/cv.', 'cv', 'MONTH', START, 10, 1)
    with pytest.raises(ValueError, match='Calendar in plugin forcing'):
        PF(FDIR + '/cv.', 'cv', 'MONTH', START, 10, 1, calendar='noleap')
    with pytest.raises(ValueError, match='timestep'):   # 12 records, DAY
        PF(FDIR + '/cv.', 'cv', 'DAY', START, 10, 1)
    with pytest.raises(Exception):                      # missing year file
        f = PF(FDIR + '/cv.', 'cv', 'MONTH', START, 365 * 10, 1)
        f.get(365 * 8 + 10)


def test_lookup_is_a_device_gather():
    f = PF(FDIR + '/mun_demand.', 'mun_demand', 'DAY', START, 400, 1)
    assert f._rec_dev.device == f.get(5).device
    assert f.get(5).dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')
def test_cuda_matches_cpu():
    a = PF(FDIR + '/cv.', 'cv', 'MONTH', START, 800, 1)
    b = PF(FDIR + '/cv.', 'cv', 'MONTH', START, 800, 1, device='cuda')
    for k in (0, 200, 400, 799):
        assert torch.equal(a.get(k), b.get(k).cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')
def test_cuda_no_host_sync_and_prefetch_overlap():
    n = 365 * 4
    f = PF(FDIR + '/mun_demand.', 'mun_demand', 'DAY', START, n,
                        1, device='cuda')
    f.get(1)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        for k in range(1, 150):          # within 2003: no window swap
            f.get(k)
    finally:
        torch.cuda.set_sync_debug_mode('default')
    # overlap: a long kernel chain on the compute stream while the copy of the
    # next year runs on the side stream; the copy event completes before the
    # compute chain does
    x = torch.randn(4096, 4096, device='cuda')
    torch.cuda.synchronize()
    f2 = PF(FDIR + '/mun_demand.', 'mun_demand', 'DAY', START, n,
                         1, device='cuda')
    f2._pending[2004].result() if 2004 in f2._pending else None
    s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s0.record()
    for _ in range(40):
        x = x @ x / 64.0
    e0.record()
    d, ev = f2._buf.get(2004) or f2._pending[2004].result()
    ev.synchronize()
    assert ev.query()
    torch.cuda.synchronize()
    assert f2._side is not torch.cuda.current_stream()
    assert s0.elapsed_time(e0) > 0
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as pr:
        g = PF(FDIR + '/cv.', 'cv', 'MONTH', START, 800, 1,
                            device='cuda')
        for _ in range(10):
            x = x @ x / 64.0
        g.get(400)
        torch.cuda.synchronize()
    names = {e.name for e in pr.events()}
    assert any('Memcpy' in nm for nm in names)
