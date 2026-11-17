"""VIC 5's output aggregation in TorchVIC: torchvic.io.history.

A global file that says

    OUTFILE  fluxes
    AGGFREQ  NDAYS 1
    OUTVAR   OUT_EVAP
    OUTVAR   OUT_SOIL_MOIST

gets, from VIC 5, one record a day: the day's evaporation summed, the soil
moisture at the end of the day. torchvic.io.history does the same from the same
lines. The rules are VIC's C (agg_data.c, alarms.c, vic_history.c,
vic_write.c); the module docstring states them.

(a) Synthetic series: every AGG_TYPE, and the NDAYS / NHOURS / NMONTHS /
    NYEARS / NSTEPS / DATE / END / NEVER periods, against values worked out
    by hand.
(b) The compiled VIC 5.1.0 image driver (reference/VIC-5.1.0; its
    aggregation code is the same as 5.0.1's):
    * Walnut Gulch, for every AGG_TYPE under seven AGGFREQs: VIC's own
      per-step history, aggregated by torchvic.io.history, is VIC's aggregated
      history to the bit; the records' times, bounds and file names are
      VIC's; and the file torchvic.io.history writes from those steps is VIC's
      file, float32 values included.
    * India basin 37, 2001, 3-hourly water balance: TorchVIC's daily history
      against VIC's daily history, beside the same comparison at the model
      step. The aggregation adds nothing: each daily difference is the sum
      (or end, or mean) of the step differences it covers.
(c) VIC 4.2.d's convention (convention='vic4.2d'), against VIC 4.2.d's C
    (tools/extract_vic42d_outvars.py re-reads the defaults, the three
    aggregation branches, the OUT_AERO_RESIST overrides and the ALMA block)
    and its compiled vicNl (reference/VIC-VIC.4.2.d/src/vicNl, or
    $VIC42D_EXE): VIC 4.2.d's own per-step output, at %.17g, aggregated and
    written by torchvic.io.history under a 4.2.d global file, is the file vicNl
    writes, byte for byte -- daily, OUT_STEP 6 and 7, from STARTHOUR 12,
    %.4f with a header, BINARY with types and mults, ALMA_OUTPUT, the
    default files, SKIPYEAR (FindleyLake) -- and MOISTFRACT to 1e-15. Then
    TorchVIC's own run of Stehekin, written by the same 4.2.d global file.
"""
import datetime as dt
import glob
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import _paths                                             # noqa: E402
from torchvic.io import history as H  # noqa: E402

REPO = _paths.REPO
D = torch.float64
CPU = torch.device('cpu')


def _up(*rel):
    """A path under one of the data roots of _paths, named as it was
    relative to the original checkout."""
    roots = {('samples', 'data'): _paths.SAMPLES,
             ('data', 'VIC_sample_data'): _paths.SAMPLE_DATA,
             ('..', 'reference'): _paths.REFERENCE}
    root = roots.get(tuple(rel[:2]))
    if root is None:
        raise ValueError('no data root for %s' % (rel,))
    return os.path.join(root, *rel[2:])


REF = _paths.REFERENCE
VIC51_TREE = _paths.VIC51_TREE
VIC501_TREE = _paths.VIC501_TREE
VIC51 = _paths.VIC51_IMAGE
WG_FULL = _paths.WG_FULL
B37 = _paths.B37_GLOBAL


def _script(name):
    """One of the tools/ scripts, imported by path."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_paths.TOOLS, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _steps(start, n, hours=3):
    return [start + dt.timedelta(hours=hours * i) for i in range(n)]


# ---------------------------------------------------------------------------
# (a) the rules, on synthetic series
# ---------------------------------------------------------------------------
def test_every_agg_type_on_three_days():
    """agg_data.c on a 3-hourly ramp that crosses zero: SUM from 0 in step
    order, AVG = SUM / count, END/BEG the last/first step, MAX/MIN started
    from 0 (VIC) or from the period's first value ('true')."""
    t = _steps(dt.datetime(2001, 1, 1), 24)
    x = (torch.arange(24, dtype=D) - 10.0).reshape(24, 1)
    recs, tail = H.windows(t, 10800, H.Freq('NDAYS', 1))
    assert recs == [(0, 7), (8, 15), (16, 23)] and tail is None
    day = x.reshape(3, 8, 1)
    want = {'SUM': day.sum(1), 'AVG': day.sum(1) / 8, 'END': day[:, -1],
            'BEG': day[:, 0]}
    for agg, w in want.items():
        assert torch.equal(H.aggregate(x, recs, agg), w), agg
    # day 1 is -10..-3: VIC's MAX is 0, the day's own is -3; day 2 is
    # -2..5: VIC's MIN is -2 (below 0), day 3 is 6..13: VIC's MIN is 0
    assert H.aggregate(x, recs, 'MAX')[:, 0].tolist() == [0.0, 5.0, 13.0]
    assert H.aggregate(x, recs, 'MAX', 'true')[:, 0].tolist() == \
        [-3.0, 5.0, 13.0]
    assert H.aggregate(x, recs, 'MIN')[:, 0].tolist() == [-10.0, -2.0, 0.0]
    assert H.aggregate(x, recs, 'MIN', 'true')[:, 0].tolist() == \
        [-10.0, -2.0, 6.0]
    # a layered output keeps its trailing axes
    y = torch.randn(24, 5, 3, dtype=D)
    assert H.aggregate(y, recs, 'END').shape == (3, 5, 3)
    assert torch.allclose(H.aggregate(y, recs, 'AVG'),
                          y.reshape(3, 8, 5, 3).mean(1), atol=1e-15)


def test_sum_is_accumulated_in_vics_order():
    """((0 + x1) + x2) + ...: the order agg_stream_data adds in, so a SUM
    is VIC's to the bit and not merely to rounding."""
    g = torch.Generator().manual_seed(3)
    x = torch.rand(80, 7, dtype=D, generator=g) * 10.0 ** torch.randint(
        -8, 3, (80, 7), generator=g).to(D)
    recs, _ = H.windows(_steps(dt.datetime(2001, 1, 1), 80), 10800,
                        H.Freq('NDAYS', 1))
    got = H.aggregate(x, recs, 'SUM').numpy()
    a = x.numpy()
    for r, (f, l) in enumerate(recs):
        acc = np.zeros(7)
        for i in range(f, l + 1):
            acc = acc + a[i]
        assert np.array_equal(got[r], acc)


def test_ndays_is_anchored_at_the_first_step():
    """alarms.c: the first period ends n days after the first step, not at
    midnight. A run starting 03:00 aggregates 03:00-03:00 (measured with VIC
    5.1.0, STARTSEC 10800); a partial last period is dropped."""
    t = _steps(dt.datetime(2008, 1, 2, 3), 8 * 7)
    recs, tail = H.windows(t, 10800, H.Freq('NDAYS', 2))
    assert recs == [(0, 15), (16, 31), (32, 47)]
    assert tail == (48, 55)
    assert t[recs[1][0]] == dt.datetime(2008, 1, 4, 3)


def test_nhours_and_nsteps():
    t = _steps(dt.datetime(2008, 1, 1), 16)
    assert H.windows(t, 10800, H.Freq('NHOURS', 6))[0] == \
        [(i, i + 1) for i in range(0, 16, 2)]
    assert H.windows(t, 10800, H.Freq('NHOURS', 24))[0] == \
        H.windows(t, 10800, H.Freq('NDAYS', 1))[0]
    assert H.windows(t, 10800, H.Freq('NSTEPS', 5)) == \
        ([(0, 4), (5, 9), (10, 14)], (15, 15))
    assert H.windows(t, 10800, H.Freq('NSTEPS', 1))[0] == \
        [(i, i) for i in range(16)]
    # a period VIC never closes: it would write nothing at all
    with pytest.raises(ValueError, match='whole number'):
        H.windows(t, 10800, H.Freq('NHOURS', 2))


def test_nmonths_follow_the_start_day():
    """reset_alarm shifts forward a step, adds the months, shifts back:
    from 1 January the calendar months (February 2008 with its 29 days),
    from 15 January 15th to 14th; a day the month lacks is refused."""
    t = _steps(dt.datetime(2008, 1, 1), 8 * (31 + 29 + 15))
    recs, tail = H.windows(t, 10800, H.Freq('NMONTHS', 1))
    assert [l - f + 1 for f, l in recs] == [8 * 31, 8 * 29]
    assert tail == (8 * 60, 8 * 75 - 1)
    t = _steps(dt.datetime(2008, 1, 15), 8 * 97)
    recs, _ = H.windows(t, 10800, H.Freq('NMONTHS', 1))
    assert [t[f] for f, _ in recs] == [dt.datetime(2008, m, 15)
                                      for m in (1, 2, 3)]
    assert t[recs[0][1]] == dt.datetime(2008, 2, 14, 21)
    recs2, _ = H.windows(_steps(dt.datetime(2008, 1, 1), 8 * 200), 10800,
                         H.Freq('NMONTHS', 2))
    assert [l - f + 1 for f, l in recs2][:3] == [8 * 60, 8 * 61, 8 * 61]
    with pytest.raises(ValueError, match='Invalid date'):
        H.windows(_steps(dt.datetime(2008, 1, 31), 8 * 40), 10800,
                  H.Freq('NMONTHS', 1))


def test_nyears_date_end_never():
    t = [dt.datetime(2007, 1, 1) + dt.timedelta(days=i) for i in range(800)]
    recs, tail = H.windows(t, 86400, H.Freq('NYEARS', 1))
    assert [l - f + 1 for f, l in recs] == [365, 366]
    with pytest.raises(ValueError, match='Feb 29'):
        H.windows([dt.datetime(2008, 3, 1) + dt.timedelta(hours=3 * i)
                   for i in range(8)], 10800, H.Freq('NYEARS', 1))
    t = _steps(dt.datetime(2008, 1, 1), 80)
    # through the step starting at 00:00 of the date, then nothing more
    # (measured: 2008-01-01 00:00 .. 2008-01-04 03:00)
    assert H.windows(t, 10800, H.Freq.parse(['DATE', '2008-01-04'])) == \
        ([(0, 24)], (25, 79))
    assert H.windows(t, 10800, H.Freq('END')) == ([(0, 79)], None)
    assert H.windows(t, 10800, H.Freq('NEVER')) == ([], (0, 79))


def test_a_daily_model_step():
    """case.dates of a daily model are dates: NDAYS 1 is every step."""
    t = [dt.date(2001, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    assert H.windows(t, 86400, H.Freq('NDAYS', 1))[0] == \
        [(i, i) for i in range(10)]
    assert H.windows(t, 86400, H.Freq('NDAYS', 7)) == ([(0, 6)], (7, 9))


def test_history_files_split_and_are_named_as_vics():
    """vic_write.c: the write alarm counts records and is checked on each
    record's closing step; initialize_history_file names by AGGFREQ."""
    t = _steps(dt.datetime(2008, 1, 1), 80)
    recs, _ = H.windows(t, 10800, H.Freq('NDAYS', 1))
    g = H.file_groups(recs, t, 10800, H.Freq('NDAYS', 3))
    assert g == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert H.file_groups(recs, t, 10800, H.Freq('END')) == [list(range(10))]
    recs6, _ = H.windows(t[:24], 10800, H.Freq('NHOURS', 6))
    assert [len(x) for x in H.file_groups(recs6, t[:24], 10800,
                                          H.Freq('NSTEPS', 7))] == [7, 5]
    s = dt.datetime(2008, 1, 2, 18)
    assert H.file_name('h', H.Freq('NHOURS', 6), s) == 'h.2008-01-02-64800.nc'
    assert H.file_name('h', H.Freq('NDAYS', 1), s) == 'h.2008-01-02.nc'
    assert H.file_name('h', H.Freq('NMONTHS', 1), s) == 'h.2008-01.nc'
    assert H.file_name('h', H.Freq('NYEARS', 1), s) == 'h.2008.nc'


def _gp(tmp_path, text):
    from torchvic.io.global_param import GlobalParam
    p = os.path.join(str(tmp_path), 'g.txt')
    open(p, 'w').write(text)
    return GlobalParam(p)


def test_the_output_section_is_parsed_as_vic_parses_it(tmp_path):
    """parse_output_info.c: streams in file order; OUTVAR's tokens are
    name, format, type, mult, aggtype; '*' is the default."""
    gp = _gp(tmp_path, """
OUTFILE fluxes
AGGFREQ NDAYS
COMPRESS TRUE
OUT_FORMAT NETCDF4
OUTVAR OUT_EVAP
OUTVAR OUT_SOIL_MOIST * OUT_TYPE_DOUBLE
OUTVAR OUT_AIR_TEMP * * * AGG_TYPE_MAX
OUTFILE snow
AGGFREQ NMONTHS 2
HISTFREQ NYEARS 1
OUTVAR OUT_SWE
""")
    s = H.parse_streams(gp)
    assert [x.prefix for x in s] == ['fluxes', 'snow']
    f, sn = s
    assert f.agg == H.Freq('NDAYS', 1) and f.compress == 5
    assert f.out_format == 'NETCDF4' and f.hist == H.Freq('END')
    assert [(v.name, v.agg, v.agg_given, v.out_type) for v in f.vars] == [
        ('OUT_EVAP', 'SUM', False, 'OUT_TYPE_FLOAT'),
        ('OUT_SOIL_MOIST', 'END', False, 'OUT_TYPE_DOUBLE'),
        ('OUT_AIR_TEMP', 'MAX', True, 'OUT_TYPE_FLOAT')]
    assert sn.agg == H.Freq('NMONTHS', 2) and sn.hist == H.Freq('NYEARS', 1)
    assert sn.out_format == 'NETCDF4_CLASSIC'
    # no OUTFILE: nothing (VIC would write its defaults)
    assert H.parse_streams(_gp(tmp_path, 'MODEL_STEPS_PER_DAY 8\n')) == []


@pytest.mark.parametrize('text, match', [
    ('AGGFREQ NDAYS 1\nOUTFILE f\n', 'OUTFILE" must be'),
    ('OUTFILE f\nOUTVAR OUT_NOT_A_VARIABLE\n', 'not found'),
    ('OUTFILE f\nOUTVAR OUT_EVAP\nOUTVAR OUT_EVAP\n', 'twice'),
    ('OUTFILE f\nOUTVAR OUT_EVAP * OUT_TYPE_SINT\n', 'cannot write'),
    ('OUTFILE f\nOUTVAR OUT_EVAP * * * AGG_TYPE_MEDIAN\n', 'Unknown agg'),
    ('OUTFILE f\nAGGFREQ WEEKLY 1\n', 'unknown frequency'),
    ('OUTFILE f\nAGGFREQ NDAYS 0\n', 'invalid n'),
    ('OUTFILE f\nAGGFREQ DATE\n', 'no date string'),
    ('OUTFILE f\nOUT_FORMAT HDF5\n', 'valid NETCDF'),
    ('OUTFILE f\nAGGFREQ NDAYS 1\nOUTFILE g\nOUTVAR OUT_EVAP\n',
     'less than 1'),
])
def test_what_vic_refuses_is_refused(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        H.parse_streams(_gp(tmp_path, text))


def test_what_vic_ignores_silently_is_said(tmp_path):
    """An aggregation type in the format position is VIC's ASCII format and
    changes nothing; a mult is never applied. Both are said."""
    said = []
    s = H.parse_streams(_gp(tmp_path, 'OUTFILE f\nOUTVAR OUT_EVAP '
                                      'AGG_TYPE_AVG\nOUTVAR OUT_RUNOFF * * '
                                      '1000\n'), log=said.append)
    assert s[0].vars[0].agg == 'SUM' and not s[0].vars[0].agg_given
    assert any('fifth token' in m for m in said)
    assert any('never applied' in m for m in said)


def test_the_result_directory(tmp_path, monkeypatch):
    """RESULT_DIR, or the caller's directory; an unset variable in it --
    UW-Hydro's ${VIC_SAMPLE_RESULTS} -- is refused, not created."""
    monkeypatch.delenv('VIC_SAMPLE_RESULTS', raising=False)
    gp = _gp(tmp_path, 'RESULT_DIR ${VIC_SAMPLE_RESULTS}/sample_image\n')
    with pytest.raises(ValueError, match='not set'):
        H.resolve_result_dir(gp)
    assert H.resolve_result_dir(gp, '/elsewhere') == '/elsewhere'
    monkeypatch.setenv('VIC_SAMPLE_RESULTS', str(tmp_path))
    assert H.resolve_result_dir(gp) == os.path.join(str(tmp_path),
                                                    'sample_image')
    said = []
    assert H.resolve_result_dir(_gp(tmp_path, 'NLAYER 3\n'),
                                log=said.append) == os.getcwd()
    assert said


def test_default_aggregation_is_vics_switch():
    """DEFAULT_END / DEFAULT_SUM are get_default_outvar_aggtype's switch,
    re-read from the C of both releases."""
    ex = _script('extract_vic_outvars')
    trees = [t for t in (VIC51_TREE, VIC501_TREE) if os.path.isdir(t)]
    if not trees:
        pytest.skip('no pristine VIC source tree')
    for tree in trees:
        g = ex.default_aggtypes(tree)
        assert set(g['AGG_TYPE_END']) == H.DEFAULT_END, tree
        assert set(g['AGG_TYPE_SUM']) == H.DEFAULT_SUM, tree
        assert g['AGG_TYPE_AVG'] == ['default'], tree


def test_outvar_inventory_is_current():
    """torchvic/data/vic5_outvars.json is what extract_vic_outvars.py reads from
    VIC 5.1.0 now; every default in it is default_agg's."""
    import json
    if not os.path.isdir(VIC51_TREE):
        pytest.skip('no pristine VIC 5.1.0 tree')
    ex = _script('extract_vic_outvars')
    now = ex.extract(VIC51_TREE)['variables']
    shipped = json.load(open(H.OUTVAR_INVENTORY))['variables']
    assert now == shipped
    for name, v in shipped.items():
        assert v['default_agg'] == H.default_agg(name), name


def test_aggregate_output_and_gradients():
    """Every per-step output by its OUT_* default; a gradient flows back to
    the steps a record covers and not to a dropped tail."""
    nt, nc = 8 * 3 + 5, 4
    g = torch.Generator().manual_seed(0)
    out = {k: torch.rand(nt, nc, dtype=D, generator=g).requires_grad_(True)
           for k in ('et', 'swe', 'air_temp', 'runoff')}
    out['soil_liq'] = torch.rand(nt, nc, 3, dtype=D, generator=g)
    out['discharge_mm'] = torch.rand(nt, 1, dtype=D, generator=g)
    out['soil_moisture'] = torch.rand(nc, 3, dtype=D)        # no time axis
    t = _steps(dt.datetime(2001, 6, 1), nt)
    said = []
    day, starts, tail = H.aggregate_output(out, t, 8, log=said.append)
    assert starts == [dt.datetime(2001, 6, d) for d in (1, 2, 3)]
    assert tail == (24, 28) and any('partial' in m for m in said)
    assert 'soil_moisture' not in day
    w = {k: v[:24].reshape(3, 8, *v.shape[1:]) for k, v in out.items()
         if k != 'soil_moisture'}
    assert torch.allclose(day['et'], w['et'].sum(1), atol=1e-15)
    assert torch.equal(day['swe'], w['swe'][:, -1])
    assert torch.allclose(day['air_temp'], w['air_temp'].mean(1), atol=1e-15)
    assert torch.equal(day['soil_liq'], w['soil_liq'][:, -1])
    assert torch.allclose(day['discharge_mm'], w['discharge_mm'].sum(1),
                          atol=1e-15)
    loss = sum(day[k].sum() for k in ('et', 'swe', 'air_temp', 'runoff'))
    loss.backward()
    assert torch.equal(out['et'].grad[:24], torch.ones(24, nc, dtype=D))
    assert float(out['et'].grad[24:].abs().sum()) == 0.0
    assert torch.equal(out['swe'].grad[7::8][:3], torch.ones(3, nc, dtype=D))
    assert float(out['swe'].grad[:7].abs().sum()) == 0.0
    x = torch.randn(24, 2, dtype=D, requires_grad=True)
    recs = [(0, 7), (8, 15), (16, 23)]
    for agg in H.AGG_TYPES:
        x.grad = None
        H.aggregate(x, recs, agg, 'true').sum().backward()
        assert torch.isfinite(x.grad).all(), agg


def test_a_daily_observation_is_aggregated_by_vics_rule():
    """config.Observation with freq daily: et summed, swe at the end of the
    day, air_temp averaged; `agg` overrides. It summed all of them."""
    from torchvic import config

    class Ctx(object):
        model_steps = 8                 # the case's steps a day

    nt = 16
    x = torch.arange(nt, dtype=D).reshape(nt, 1).repeat(1, 3)
    out = {'et': x, 'swe': x, 'air_temp': x}
    s = {k: config.Observation(k, None, freq='daily').simulate(out, Ctx())
         for k in out}
    assert s['et'][:, 0].tolist() == [28.0, 92.0]
    assert s['swe'][:, 0].tolist() == [7.0, 15.0]
    assert s['air_temp'][:, 0].tolist() == [3.5, 11.5]
    o = config.Observation('swe', None, freq='daily', agg='mean')
    assert o.simulate(out, Ctx())[:, 0].tolist() == [3.5, 11.5]
    with pytest.raises(ValueError):
        config.Observation('swe', None, freq='daily', agg='median')


# ---------------------------------------------------------------------------
# (b) against the compiled VIC 5.1.0 image driver
# ---------------------------------------------------------------------------
DROP = ('OUTFILE', 'OUTVAR', 'AGGFREQ', 'OUT_FORMAT', 'HISTFREQ', 'COMPRESS',
        'RESULT_DIR', 'LOG_DIR')


def _vic_global(src, tmp, subs, output):
    """`src` with `subs`, its output section replaced by `output`, writing
    to tmp/results. Returns (global file, results directory)."""
    res, logs = os.path.join(tmp, 'results'), os.path.join(tmp, 'logs')
    os.makedirs(res, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    subs, lines, seen = dict(subs), [], set()
    for ln in open(src).read().splitlines():
        key = ln.split()[0].upper() if ln.split() else ''
        if key in DROP:
            continue
        if key in subs:
            ln = '%s %s' % (key, subs[key])
            seen.add(key)
        lines.append(ln)
    lines += ['%s %s' % (k, v) for k, v in subs.items() if k not in seen]
    lines += ['RESULT_DIR %s' % res, 'LOG_DIR %s/' % logs] + list(output)
    gf = os.path.join(tmp, 'global.txt')
    open(gf, 'w').write('\n'.join(lines) + '\n')
    return gf, res


def _run_vic(gf):
    env = dict(os.environ, OMPI_MCA_btl_vader_single_copy_mechanism='none')
    r = subprocess.run([VIC51, '-g', gf], env=env, capture_output=True,
                       text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def _open(res, prefix):
    """A stream's history, its files concatenated in time (not decoded)."""
    import xarray as xr
    files = sorted(glob.glob(os.path.join(res, prefix + '.*.nc')))
    assert files, 'no %s.* history in %s' % (prefix, res)
    parts = [xr.open_dataset(f, decode_times=False).load() for f in files]
    return (xr.concat(parts, dim='time') if len(parts) > 1 else parts[0],
            [os.path.basename(f) for f in files])


WG_VARS = ('OUT_PREC', 'OUT_RAINF', 'OUT_SNOWF', 'OUT_EVAP', 'OUT_RUNOFF',
           'OUT_BASEFLOW', 'OUT_SOIL_MOIST', 'OUT_SWE', 'OUT_AIR_TEMP')
#: basin.simulate's key for each, as history.OUTVAR_SOURCES reads it back
WG_KEYS = {'OUT_RAINF': 'rainf', 'OUT_SNOWF': 'snowf', 'OUT_EVAP': 'et',
           'OUT_RUNOFF': 'runoff', 'OUT_BASEFLOW': 'baseflow',
           'OUT_SOIL_MOIST': 'soil_liq', 'OUT_SWE': 'swe',
           'OUT_AIR_TEMP': 'air_temp'}


def _stream_lines(prefix, freq, agg='*', typ='OUT_TYPE_DOUBLE', hist=None,
                  names=WG_VARS):
    out = ['OUTFILE %s' % prefix, 'AGGFREQ %s' % freq]
    if hist:
        out.append('HISTFREQ %s' % hist)
    return out + ['OUTVAR %s * %s * %s' % (v, typ, agg) for v in names]


WG_CASES = {
    # AGGFREQ, run dates, HISTFREQ
    'NDAYS 1': ({'ENDMONTH': '01', 'ENDDAY': '10'}, 'NDAYS 3'),
    'NDAYS 7': ({'ENDMONTH': '01', 'ENDDAY': '10'}, None),
    'NHOURS 6': ({'ENDMONTH': '01', 'ENDDAY': '03'}, 'NSTEPS 7'),
    'NSTEPS 5': ({'ENDMONTH': '01', 'ENDDAY': '10'}, None),
    'DATE 2008-01-04': ({'ENDMONTH': '01', 'ENDDAY': '10'}, None),
    'NMONTHS 1': ({'STARTDAY': '15', 'ENDMONTH': '04', 'ENDDAY': '20'},
                  'NMONTHS 2'),
    'NDAYS 2 from 03:00': ({'STARTDAY': '02', 'STARTSEC': '10800',
                            'ENDMONTH': '01', 'ENDDAY': '09'}, None),
}


def _need_vic(*paths):
    for p in (VIC51,) + paths:
        if not os.path.exists(p):
            pytest.skip('%s is not present' % p)
    pytest.importorskip('xarray')
    pytest.importorskip('netCDF4')


@pytest.mark.parametrize('case', sorted(WG_CASES))
def test_walnut_gulch_aggregation_is_vics_to_the_bit(tmp_path, case):
    """VIC 5.1.0 on Walnut Gulch (water balance) writes, in one run, its
    per-step history (NSTEPS 1, double) and the same variables under
    `case` for every AGG_TYPE, plus one stream as VIC writes it by default
    (float, default aggregations). torchvic.io.history, fed VIC's per-step
    values, must give:

    * VIC's records, value for value: max |difference| 0 (double);
    * VIC's time and time_bnds, and VIC's file names under HISTFREQ;
    * and, written by write_record, VIC's default-type file: same
      variables, dimensions, attributes and float32 values.
    """
    _need_vic(WG_FULL)
    import xarray as xr
    from torchvic.io.global_param import GlobalParam
    from torchvic.io.domain import ImageDomain
    subs, hist = WG_CASES[case]
    subs = dict(subs, FULL_ENERGY='FALSE')
    freq = case.split(' from ')[0]
    out_lines = _stream_lines('st', 'NSTEPS 1')
    for agg in H.AGG_TYPES:
        out_lines += _stream_lines('a' + agg.lower(), freq,
                                   agg='AGG_TYPE_' + agg)
    out_lines += _stream_lines('dflt', freq, typ='*', hist=hist)
    gf, res = _vic_global(WG_FULL, str(tmp_path), subs, out_lines)
    _run_vic(gf)
    st, _ = _open(res, 'st')
    gp = GlobalParam(gf)
    dom = ImageDomain(gp.path_of('DOMAIN'), names=gp.domain_vars,
                      params=gp.path_of('PARAMETERS'))
    # the steps' own start times, as VIC stamped them
    import cftime
    times = list(cftime.num2date(st['time'].values, st['time'].units,
                                 st['time'].calendar,
                                 only_use_cftime_datetimes=False,
                                 only_use_python_datetimes=True))
    f = H.Freq.parse(freq.split())
    recs, _ = H.windows(times, 10800, f)
    worst = 0.0
    for agg in H.AGG_TYPES:
        vic, _ = _open(res, 'a' + agg.lower())
        assert vic.sizes['time'] == len(recs), (agg, vic.sizes['time'])
        assert np.array_equal(vic['time'].values,
                              st['time'].values[[a for a, _ in recs]])
        assert np.array_equal(vic['time_bnds'].values[:, 0],
                              vic['time'].values)
        assert np.allclose(vic['time_bnds'].values[:, 1],
                           st['time'].values[[b for _, b in recs]] + 0.125,
                           rtol=0, atol=1e-9)
        for v in WG_VARS:
            x = torch.as_tensor(st[v].values, dtype=D)
            got = H.aggregate(x, recs, agg).numpy()
            want = vic[v].values
            m = np.isfinite(want)
            d = float(np.max(np.abs(got[m] - want[m]))) if m.any() else 0.0
            worst = max(worst, d)
            assert d == 0.0, (case, agg, v, d)
    # the file itself, from VIC's per-step values through write_record
    rows, cols = dom.rows, dom.cols
    out = {}
    for v, k in WG_KEYS.items():
        a = st[v].values
        a = a[:, :, rows, cols].transpose(0, 2, 1) if a.ndim == 4 \
            else a[:, rows, cols]
        out[k] = torch.as_tensor(a, dtype=D)
    out['soil_ice'] = torch.zeros_like(out['soil_liq'])
    stream = H.parse_streams(gp)[-1]
    assert stream.prefix == 'dflt'
    rec = H.stream_records(out, stream, times, 8, log=lambda *a: None)
    mine = os.path.join(str(tmp_path), 'mine')
    paths = H.write_record(rec, dom, mine, gp=gp, log=lambda *a: None)
    # its own files may be rewritten; VIC's, in VIC's RESULT_DIR, may not
    assert H.write_record(rec, dom, mine, gp=gp,
                          log=lambda *a: None) == paths
    with pytest.raises(FileExistsError, match='not written by TorchVIC'):
        H.write_record(rec, dom, res, gp=gp, log=lambda *a: None)
    theirs = sorted(os.path.basename(p) for p in
                    glob.glob(os.path.join(res, 'dflt.*.nc')))
    assert sorted(os.path.basename(p) for p in paths) == theirs
    for name in theirs:
        a = xr.open_dataset(os.path.join(res, name), decode_times=False,
                            mask_and_scale=False)
        b = xr.open_dataset(os.path.join(mine, name), decode_times=False,
                            mask_and_scale=False)
        assert np.array_equal(a['time'].values, b['time'].values)
        assert np.array_equal(a['time_bnds'].values, b['time_bnds'].values)
        for k in ('units', 'calendar'):
            assert a['time'].attrs[k] == b['time'].attrs[k]
        for v in WG_VARS:
            assert a[v].dims == b[v].dims, v
            assert a[v].dtype == b[v].dtype == np.float32, v
            for k in ('long_name', 'standard_name', 'units', 'description',
                      'cell_methods', '_FillValue'):
                assert a[v].attrs.get(k) == b[v].attrs.get(k) or \
                    np.array_equal(a[v].attrs.get(k), b[v].attrs.get(k)), \
                    (v, k)
            assert np.array_equal(a[v].values, b[v].values), (name, v)
    # and the comparison can tell VIC's zero-started MIN from the period's
    # own: soil moisture never reaches 0, VIC's MIN of it is 0 throughout
    vmin, _ = _open(res, 'amin')
    x = torch.as_tensor(st['OUT_SOIL_MOIST'].values, dtype=D)
    true = H.aggregate(x, recs, 'MIN', 'true').numpy()
    m = np.isfinite(true)
    assert (vmin['OUT_SOIL_MOIST'].values[m] == 0.0).all()
    assert (true[m] > 0.0).all()
    print('%s: %d records, %d file(s), max |VIC - torchvic.io.history| over '
          'every AGG_TYPE and variable %.1e' % (case, len(recs), len(theirs),
                                                worst))


def _weighted(dom):
    w = np.asarray(dom.area, np.float64)
    return w / w.sum()


def _cells(a, dom):
    """[time, (layer,) y, x] -> [time, cell(, layer)]."""
    if a.ndim == 4:
        return a[:, :, dom.rows, dom.cols].transpose(0, 2, 1)
    return a[:, dom.rows, dom.cols]


B37_VARS = ('OUT_PREC', 'OUT_EVAP', 'OUT_RUNOFF', 'OUT_BASEFLOW',
            'OUT_SOIL_MOIST', 'OUT_SWE', 'OUT_AIR_TEMP', 'OUT_SURF_TEMP',
            'OUT_LATENT', 'OUT_SENSIBLE')
#: mode -> (FULL_ENERGY, the step-level tolerance TorchVIC meets on this
#: case -- measured ET +0.0011% / baseflow -0.0039% in the
#: water balance and ET +0.0117% / baseflow -0.0172% in the energy balance
#: -- and what TorchVIC does not produce in that mode)
B37_MODES = {
    'water_balance': ('FALSE', 1e-4, set()),
    'energy_balance': ('TRUE', 5e-4, set()),
}


def _write_subdaily_forcing(gf, out, start, ndays, device):
    """TorchVIC's forcing for the run, written as VIC 5 image forcing files,
    and the global file `gf` re-pointed at them (rewritten in place)."""
    import xarray as xr
    from torchvic import case
    from torchvic.io.global_param import GlobalParam
    gp = GlobalParam(gf)
    spd = int(gp.get_int('MODEL_STEPS_PER_DAY'))
    f0 = dt.date(int(gp.get('FORCEYEAR')), int(gp.get('FORCEMONTH', 1)),
                 int(gp.get('FORCEDAY', 1)))
    fspd = int(gp.get_int('FORCE_STEPS_PER_DAY') or 1)   # case.load
    # the global file's STARTYEAR.. is `start`: case.window starts there
    assert (start - f0).days >= 0
    c = case.load(gf, device=device, steps=ndays * fspd,  # counts records
                  log=lambda *a: None)
    a = c.atm
    dom = c.dom
    fdir = os.path.join(out, 'forcing')
    os.makedirs(fdir, exist_ok=True)
    y, x = dom.y_dim, dom.x_dim
    with xr.open_dataset(gp.get('DOMAIN')) as d:
        lat = d[gp.domain_vars['lat_var']].values
        lon = d[gp.domain_vars['lon_var']].values
    fields = (('air_temp', 'tair', 1.0, 'C'), ('prec', 'prcp', 1.0, 'mm'),
              ('pressure', 'press', 1e-3, 'kPa'),
              ('swdown', 'shortwave', 1.0, 'W m-2'),
              ('lwdown', 'longwave', 1.0, 'W m-2'),
              ('vp', 'eact_air', 1e-3, 'kPa'), ('wind', 'wind', 1.0, 'm/s'))
    t0 = dt.datetime.combine(start, dt.time())
    times = [t0 + dt.timedelta(seconds=86400 // spd * i)
             for i in range(ndays * spd)]
    years = sorted(set(t.year for t in times))
    for yr in years:
        idx = [i for i, t in enumerate(times) if t.year == yr]
        ds = xr.Dataset(coords={'time': [times[i] for i in idx], y: lat,
                                x: lon})
        for name, key, k, units in fields:
            v = a[key].detach().cpu().double().numpy()[idx] * k
            g = np.full((len(idx), len(lat), len(lon)), np.nan)
            g[:, dom.rows, dom.cols] = v
            ds[name] = (('time', y, x), g, dict(units=units))
        ds.time.encoding.update(units='hours since %s' % t0.isoformat(' '),
                                calendar='proleptic_gregorian')
        ds.to_netcdf(os.path.join(fdir, 'forcing.%d.nc' % yr))
    lines = []
    for ln in open(gf).read().splitlines():
        key = ln.split()[0].upper() if ln.split() else ''
        if key in ('FORCING1', 'FORCE_TYPE', 'FORCE_STEPS_PER_DAY',
                   'FORCEYEAR', 'FORCEMONTH', 'FORCEDAY'):
            continue
        lines.append(ln)
    lines += ['FORCING1 %s/forcing.' % fdir]
    lines += ['FORCE_TYPE %-9s %s' % (vk, n) for vk, n in
              (('AIR_TEMP', 'air_temp'), ('PREC', 'prec'),
               ('PRESSURE', 'pressure'), ('SWDOWN', 'swdown'),
               ('LWDOWN', 'lwdown'), ('VP', 'vp'), ('WIND', 'wind'))]
    # VIC 5's image driver takes the step and the start from the files'
    # time axis and rejects FORCE_STEPS_PER_DAY / FORCEYEAR as unknown
    open(gf, 'w').write('\n'.join(lines) + '\n')
    return gf


@pytest.mark.parametrize('mode', sorted(B37_MODES))
def test_basin37_daily_history_is_vics(tmp_path, mode):
    """India basin 37, 2001, 100 cells, 3-hourly, TorchVIC's 5.1.0 core
    against VIC 5.1.0, both reading TorchVIC's disaggregated forcing
    (_write_subdaily_forcing), in the water balance and in
    the energy balance. Both write the same three streams from one global
    file: per-step (NSTEPS 1, double), daily (NDAYS 1, double) and daily as
    VIC writes it by default (float).

    Asserted: each model's daily history is its own per-step history
    aggregated by torchvic.io.history, exactly; TorchVIC's per-step totals agree
    with VIC's within the tolerance this case is known to meet, and its
    daily ones within the same -- a summed or averaged variable's daily
    difference IS its per-step difference (to 1e-9), an end-of-period
    state's is bounded by the same tolerance. A variable VIC writes and
    TorchVIC does not compute in the mode (the surface energy terms of the
    water balance) is named as missing, not written as zeros."""
    _need_vic(B37)
    from torchvic import case
    from torchvic.core import version as core_version
    full_energy, tol, absent = B37_MODES[mode]
    start, end = dt.date(2001, 1, 1), dt.date(2001, 12, 31)
    ndays = (end - start).days + 1
    subs = dict(MODEL_STEPS_PER_DAY=8, SNOW_STEPS_PER_DAY=8,
                RUNOFF_STEPS_PER_DAY=24, FULL_ENERGY=full_energy,
                # SNOW_BAND FALSE: this comparison runs at the file's elev
                QUICK_FLUX='TRUE', SNOW_BAND='FALSE', WIND_H='10.0',
                STARTYEAR=start.year, STARTMONTH='%02d' % start.month,
                STARTDAY='%02d' % start.day, ENDYEAR=end.year,
                ENDMONTH='%02d' % end.month, ENDDAY='%02d' % end.day)
    out_lines = (_stream_lines('steps', 'NSTEPS 1', names=B37_VARS)
                 + _stream_lines('dailyd', 'NDAYS 1', names=B37_VARS)
                 + _stream_lines('daily', 'NDAYS 1', typ='*',
                                 names=B37_VARS))
    tmp = str(tmp_path)
    gf, res = _vic_global(B37, tmp, subs, out_lines)
    old, threads = core_version.CORE, torch.get_num_threads()
    core_version.set_core('5.1.0')
    # 100 cells are small tensors: a few threads run a water-balance year
    # in about a minute and a hundred contending ones took over twenty
    torch.set_num_threads(min(threads, 4))
    try:
        _write_subdaily_forcing(gf, tmp, start, ndays, 'cpu')
        _run_vic(gf)
        c = case.load(gf, device=CPU, steps=ndays * 8, log=lambda *a: None)
        assert c.model_steps == 8
        assert c.full_energy == (full_energy == 'TRUE')
        with torch.no_grad():
            out = c.run(p=c.params())
    finally:
        core_version.set_core(old)
        torch.set_num_threads(threads)
    said = []
    mine = os.path.join(tmp, 'mine')
    paths = c.write_history(out, result_dir=mine, log=said.append)
    assert sorted(os.path.basename(p) for p in paths) == \
        ['daily.2001-01-01.nc', 'dailyd.2001-01-01.nc',
         'steps.2001-01-01-00000.nc']
    for v in absent:
        assert any(v in m and 'not written' in m for m in said), v
    dom, w = c.dom, _weighted(c.dom)
    vs, _ = _open(res, 'steps')
    vd, _ = _open(res, 'dailyd')
    vf, _ = _open(res, 'daily')
    gs, _ = _open(mine, 'steps')
    gd, _ = _open(mine, 'dailyd')
    gf32, _ = _open(mine, 'daily')
    assert set(B37_VARS) - set(gd.data_vars) == absent
    assert np.array_equal(vd['time'].values, gd['time'].values)
    assert np.array_equal(vd['time_bnds'].values, gd['time_bnds'].values)
    assert np.array_equal(vs['time'].values, gs['time'].values)
    recs = [(8 * d, 8 * d + 7) for d in range(ndays)]
    lines = ['India basin 37, %s .. %s, 100 cells, 3-hourly %s, TorchVIC '
             '5.1.0 core vs VIC 5.1.0 (area-weighted basin totals for SUM, '
             'means otherwise; |d| per cell, layers summed)'
             % (start, end, mode.replace('_', ' ')),
             '  %-15s %-4s | %-37s | %-37s | %s'
             % ('variable', 'agg', 'per step: TorchVIC vs VIC',
                'daily: TorchVIC vs VIC', 'daily float files')]
    for v in B37_VARS:
        agg = H.default_agg(v)
        # the aggregation is each model's own, exactly
        for steps, daily in ((vs, vd), (gs, gd)):
            if v not in daily:
                continue
            x = torch.as_tensor(steps[v].values, dtype=D)
            assert np.array_equal(H.aggregate(x, recs, agg).numpy(),
                                  daily[v].values, equal_nan=True), v
        if v in absent:
            lines.append('  %-15s %-4s | not computed by TorchVIC in this '
                         'mode; VIC writes a basin mean of %.4g'
                         % (v, agg, float(np.nanmean(vd[v].values))))
            continue
        row = []
        for a_, b_ in ((vs, gs), (vd, gd), (vf, gf32)):
            a = _cells(a_[v].values.astype(np.float64), dom)
            b = _cells(b_[v].values.astype(np.float64), dom)
            if a.ndim == 3:
                a, b = a.sum(-1), b.sum(-1)
            A, B = (a * w).sum(-1), (b * w).sum(-1)
            TA, TB = ((A.sum(), B.sum()) if agg == 'SUM'
                      else (A.mean(), B.mean()))
            rel = (TB - TA) / abs(TA) if abs(TA) > 1e-12 else 0.0
            d = np.abs(a - b)
            row.append((rel, float(np.percentile(d, 99)), float(d.max())))
        assert vf[v].dtype == gf32[v].dtype == np.float32, v
        (srel, sp99, smax), (drel, dp99, dmax), (_, _, fmax) = row
        lines.append('  %-15s %-4s | %+.4e%% p99 %.1e max %.1e | %+.4e%% '
                     'p99 %.1e max %.1e | max %.1e'
                     % (v, agg, 100 * srel, sp99, smax, 100 * drel, dp99,
                        dmax, fmax))
        assert abs(srel) < tol, (v, srel)
        assert abs(drel) < tol, (v, drel)
        if agg in ('SUM', 'AVG'):
            # the daily difference is the per-step difference, aggregated
            assert abs(drel - srel) < 1e-9, (v, drel, srel)
    print('\n'.join(lines))



# ---------------------------------------------------------------------------
# (c) VIC 4.2.d's convention, against VIC 4.2.d's own C and its vicNl
# ---------------------------------------------------------------------------
VIC42D_SRC = _up('..', 'reference', 'VIC-VIC.4.2.d', 'src')
VIC42D = os.environ.get('VIC42D_EXE') or os.path.join(VIC42D_SRC, 'vicNl')
STEHE_CLASSIC = _up('samples', 'data', 'classic', 'Stehekin')
#: VIC's Stehekin image global file with SNOW_BAND FALSE: TorchVIC refuses
#: its 3-5 bands with area (torchvic.io.options), and these tests are about
#: the output conventions, not the bands.
STEHE_IMAGE_GLOBAL = _paths.relocated_global(
    _paths.STEHE_IMAGE_GLOBAL, set_options={'SNOW_BAND': 'FALSE'})
FINDLEY_CLASSIC = _up('data', 'VIC_sample_data', 'classic', 'FindleyLake')


def test_42d_defaults_are_create_output_list():
    """DEFAULT_END_42D / DEFAULT_SUM_42D, the AVG/END/SUM branches, the
    OUT_AERO_RESIST overrides and ALMA_42D are VIC 4.2.d's C, re-read by
    tools/extract_vic42d_outvars.py; the shipped inventory is current;
    and every name VIC 4.2.d and VIC 5 share has the same default."""
    import json
    if not os.path.exists(os.path.join(VIC42D_SRC, 'put_data.c')):
        pytest.skip('no pristine VIC 4.2.d source')
    ex = _script('extract_vic42d_outvars')
    v5 = json.load(open(H.OUTVAR_INVENTORY))
    inv = ex.extract(VIC42D_SRC, v5)
    var = inv['variables']
    assert len(var) == 167 and inv['n_outvar_types'] == 180
    assert {n for n, v in var.items() if v['default_agg'] == 'END'} == \
        H.DEFAULT_END_42D
    assert {n for n, v in var.items() if v['default_agg'] == 'SUM'} == \
        H.DEFAULT_SUM_42D
    assert inv['default_agg_fallback'] == H.DEFAULT_OTHER_42D == 'AVG'
    # put_data.c:669-684 -- three branches; AVG divides each step first
    assert inv['aggregation_branches'] == {
        'END': 'aggdata = out_data[v].data[i]',
        'SUM': 'aggdata += out_data[v].data[i]',
        'AVG': 'aggdata += out_data[v].data[i]/out_step_ratio'}
    assert sorted(set(inv['aggregation_branches'])) == \
        sorted(H.AGG_TYPES_42D)
    assert inv['aggregation_overrides'] == [
        'out_data[OUT_AERO_RESIST%s].aggdata[0] = 1/out_data[OUT_AERO_COND'
        '%s].aggdata[0];' % (k, k) for k in ('', '1', '2')]
    assert [tuple(x) for x in inv['alma_output']] == list(H.ALMA_42D)
    shipped = json.load(open(H.OUTVAR_INVENTORY_42D))
    for k in ('variables', 'aggregation_branches', 'aggregation_overrides',
              'alma_output', 'vs_vic5'):
        assert shipped[k] == inv[k], k
    d = inv['vs_vic5']
    assert set(d['only_in_vic5']) == H.VIC5_ONLY
    assert d['default_agg_differs'] == {}
    for n in set(var) & set(v5['variables']):
        assert H.default_agg(n, 'vic4.2d') == H.default_agg(n), n
    # the renamed ones keep their default across the rename too
    for old, key in H.OUTVAR_SOURCES_42D.items():
        new = [n for n, s in H.OUTVAR_SOURCES.items() if s == key]
        assert new and H.default_agg(new[0]) == H.default_agg_42d(old)
    print('VIC 4.2.d: %d OUT_* variables (%d shared with VIC 5, whose '
          'default AGG_TYPE is the same for all of them); only in 4.2.d: '
          '%s; only in VIC 5: %s'
          % (len(var), len(set(var) & set(v5['variables'])),
             ', '.join(d['only_in_vic42d']), ', '.join(d['only_in_vic5'])))


def test_42d_periods_stamps_and_skipyear_by_hand():
    """put_data.c on synthetic series: periods of OUT_STEP/TIME_STEP steps
    from the first step, AVG = sum of x/n (not VIC 5's sum/n), a partial
    last period dropped, the carry across cells, SKIPYEAR by make_dmy."""
    recs, tails = H.windows_42d(10, 4)
    assert recs == [(0, 3), (4, 7)] and tails == [(8, 9)]
    recs, tails = H.windows_42d(10, 4, n_cells=2)
    assert recs == [(0, 3), (4, 7), (10, 13), (14, 17)]
    assert tails == [(8, 9), (18, 19)]
    # carried: cell 1's last 2 steps open the record cell 2 closes at 11
    recs, tails = H.windows_42d(10, 4, n_cells=2, carry=True)
    assert recs == [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19)]
    assert tails == []
    g = torch.Generator().manual_seed(7)
    x = torch.rand(24 * 50, 3, dtype=D, generator=g) * 100
    recs, _ = H.windows_42d(x.shape[0], 24)
    avg = H.aggregate_42d(x, recs, 'AVG', 24).numpy()
    a = x.numpy()
    for r, (f, l) in enumerate(recs):
        acc = np.zeros(3)
        for i in range(f, l + 1):
            acc = acc + a[i] / 24.0
        assert np.array_equal(avg[r], acc)
    vic5 = H.aggregate(x, recs, 'AVG').numpy()
    # the same mean to rounding, not the same bits
    assert np.allclose(avg, vic5, rtol=1e-14, atol=0)
    assert (avg != vic5).any()
    assert torch.equal(H.aggregate_42d(x, recs, 'END', 24),
                       x[23::24])
    for agg in ('MAX', 'MIN', 'BEG'):
        with pytest.raises(ValueError, match='only AVG, END and SUM'):
            H.aggregate_42d(x, recs, agg, 24)
    # OUT_STEP: hours or an AGGFREQ of at most a day, a multiple of the step
    assert H.out_step_42d(0, 3) == 3 and H.out_step_42d(None, 3) == 3
    assert H.out_step_42d('NDAYS 1', 3) == 24
    assert H.out_step_42d('NSTEPS 2', 3) == 6
    assert H.out_step_42d('NHOURS 12', 1) == 12
    for bad in ('NDAYS 2', 'NMONTHS 1', 'NHOURS 4', 'END', 2):
        with pytest.raises(ValueError):
            H.out_step_42d(bad, 3)
    with pytest.raises(ValueError, match='whole number of hours'):
        H.step_hours_42d(48)
    # make_dmy.c:163-169: the start year's length decides, not Feb 29's
    # place in the skipped span
    assert H.skip_records_42d(dt.datetime(1971, 7, 1), 1, 1) == 365 * 24
    assert H.skip_records_42d(dt.datetime(1972, 1, 1), 3, 1) == 366 * 8
    assert H.skip_records_42d(dt.date(1970, 1, 1), 24, 3) == 365 * 2 + 366
    # aggregate_output: 4.2.d's stamps are the periods' last steps
    t = _steps(dt.datetime(2001, 6, 1), 8 * 3 + 5)
    out = {'et': torch.rand(29, 2, dtype=D, generator=g),
           'swe': torch.rand(29, 2, dtype=D, generator=g),
           'eb_aero_cond': torch.rand(29, 2, dtype=D, generator=g) + 0.01,
           'eb_aero_resist': torch.rand(29, 2, dtype=D, generator=g)}
    said = []
    day, stamps, tail = H.aggregate_output(out, t, 8, log=said.append,
                                           convention='vic4.2d')
    assert stamps == [dt.datetime(2001, 6, d, 21) for d in (1, 2, 3)]
    assert tail == (24, 28) and any('partial' in m for m in said)
    assert torch.equal(day['swe'], out['swe'][7:24:8])
    assert torch.equal(day['eb_aero_resist'],
                       1.0 / H.aggregate_42d(out['eb_aero_cond'],
                                             [(0, 7), (8, 15), (16, 23)],
                                             'AVG', 8))


def _gp_text(tmp_path, text, name='g42.txt'):
    from torchvic.io.global_param import GlobalParam
    p = os.path.join(str(tmp_path), name)
    open(p, 'w').write(text)
    return GlobalParam(p)


def test_42d_output_section_is_read_as_vicNl_reads_it(tmp_path):
    """get_global_param.c / parse_output_info.c: detection, OUT_STEP,
    N_OUTFILES/OUTFILE/OUTVAR, format/type/mult per VARIABLE (the last
    setting holds in every file), the defaults without N_OUTFILES, and
    what 4.2.d would get wrong refused by name."""
    said = []
    gp = _gp_text(tmp_path, """TIME_STEP 3
OUT_STEP 24
SKIPYEAR 2
BINARY_OUTPUT TRUE
PRT_HEADER TRUE
GRID_DECIMAL 5
N_OUTFILES 2
OUTFILE fluxes 3
OUTVAR OUT_PREC %.3f OUT_TYPE_USINT 40
OUTVAR OUT_EVAP
OUTVAR OUT_SOIL_MOIST * OUT_TYPE_INT * AGG_TYPE_MAX
OUTFILE snow 2
OUTVAR OUT_SWE %.2f OUT_TYPE_SINT 10
OUTVAR OUT_PREC %.5e
""")
    assert H.is_vic42d_global(gp) and H.convention_of(gp) == 'vic4.2d'
    s = H.parse_output_42d(gp, log=said.append)
    assert (s.dt, s.out_dt, s.skipyear, s.binary, s.prt_header,
            s.grid_decimal) == (3, 24, 2, True, True, 5)
    assert [(f.prefix, f.names) for f in s.files] == [
        ('fluxes', ['OUT_PREC', 'OUT_EVAP', 'OUT_SOIL_MOIST']),
        ('snow', ['OUT_SWE', 'OUT_PREC'])]
    # OUT_PREC's second line sets a format only: its type and mult stay
    assert s.format_of('OUT_PREC') == '%.5e'
    assert s.type_of('OUT_PREC') == 'OUT_TYPE_USINT'
    assert s.mult_of('OUT_PREC') == 40.0
    assert s.format_of('OUT_EVAP') == '%.4f'
    assert s.type_of('OUT_SOIL_MOIST') == 'OUT_TYPE_FLOAT'
    assert any('OUT_TYPE_INT' in m for m in said)
    assert any('AGG_TYPE_MAX' in m and 'fourth token' in m for m in said)
    assert s.files[0].agg_of('OUT_SOIL_MOIST') == 'END'
    # a VIC 5 file is not one
    assert H.convention_of(_gp_text(tmp_path, 'MODEL_STEPS_PER_DAY 8\n'
                                    'OUTFILE f\nOUTVAR OUT_EVAP\n',
                                    'g5.txt')) == 'vic5'
    # no N_OUTFILES: set_output_defaults.c's two files, 20 + 4 variables
    s = H.parse_output_42d(_gp_text(tmp_path, 'TIME_STEP 24\n'),
                           log=said.append)
    assert [(f.prefix, len(f.names)) for f in s.files] == [('fluxes', 20),
                                                          ('snow', 4)]
    assert s.out_dt == 24
    s = H.parse_output_42d(_gp_text(tmp_path, 'TIME_STEP 1\nFULL_ENERGY '
                                    'TRUE\nBLOWING TRUE\n'), log=said.append)
    assert [(f.prefix, len(f.names)) for f in s.files] == [('fluxes', 26),
                                                          ('snow', 17)]
    for text, match in (
            ('TIME_STEP 3\nOUT_STEP 4\n', 'Invalid output step'),
            ('TIME_STEP 3\nOUT_STEP 48\n', 'Invalid output step'),
            ('TIME_STEP 1\nOUTFILE f 1\n', 'N_OUTFILES'),
            ('TIME_STEP 1\nN_OUTFILES 1\nOUTFILE f 2\nOUTVAR OUT_PREC\n',
             'OUT_ASAT'),
            ('TIME_STEP 1\nN_OUTFILES 1\nOUTFILE f 1\nOUTVAR OUT_PREC\n'
             'OUTVAR OUT_EVAP\n', 'past the end'),
            ('TIME_STEP 1\nN_OUTFILES 2\nOUTFILE f 1\nOUTVAR OUT_PREC\n',
             'empty prefix'),
            ('TIME_STEP 1\nN_OUTFILES 1\nOUTFILE f 1\nOUTVAR OUT_SWDOWN\n',
             'not found'),
            ('TIME_STEP 1\nN_OUTFILES 1\nOUTFILE f 1\nOUTVAR OUT_PREC %d\n',
             'floating-point')):
        with pytest.raises(ValueError, match=match):
            H.parse_output_42d(_gp_text(tmp_path, text), log=said.append)


# The per-step run every Stehekin comparison aggregates: all variables at
# %.17g (which round-trips a double), with a header naming the columns.
STEP_VARS_42D = (
    'OUT_PREC', 'OUT_EVAP', 'OUT_RUNOFF', 'OUT_BASEFLOW', 'OUT_SOIL_MOIST',
    'OUT_SWE', 'OUT_AIR_TEMP', 'OUT_LATENT', 'OUT_AERO_COND',
    'OUT_AERO_RESIST', 'OUT_REL_HUMID', 'OUT_SUB_SNOW', 'OUT_SUB_CANOP',
    'OUT_SOIL_TEMP', 'OUT_PRESSURE', 'OUT_VP', 'OUT_SOIL_LIQ', 'OUT_WDEW',
    'OUT_NET_SHORT', 'OUT_R_NET', 'OUT_EVAP_CANOP', 'OUT_TRANSP_VEG',
    'OUT_EVAP_BARE', 'OUT_SURF_TEMP', 'OUT_ALBEDO', 'OUT_IN_LONG',
    'OUT_WIND', 'OUT_SNOW_DEPTH', 'OUT_SNOW_CANOPY', 'OUT_SNOW_COVER')
FLUX_VARS_42D = ('OUT_PREC', 'OUT_EVAP', 'OUT_RUNOFF', 'OUT_BASEFLOW',
                 'OUT_SOIL_MOIST', 'OUT_SWE', 'OUT_AIR_TEMP', 'OUT_LATENT')
LAYERED_42D = ('OUT_SOIL_MOIST', 'OUT_SOIL_TEMP', 'OUT_SOIL_LIQ')


def _outfiles_42d(files, fmt='%.17g', extra=None):
    """N_OUTFILES/OUTFILE/OUTVAR lines: files is [(prefix, [names])]."""
    lines = ['N_OUTFILES %d' % len(files)]
    for prefix, names in files:
        lines.append('OUTFILE %s %d' % (prefix, len(names)))
        for n in names:
            e = (extra or {}).get(n)
            lines.append('OUTVAR %s%s' % (n, e if e is not None else
                                          (' ' + fmt if fmt else '')))
    return '\n'.join(lines) + '\n'


def _stehekin_global_42d(tmp, name, output, starthour=0):
    s = STEHE_CLASSIC
    res = os.path.join(tmp, name)
    os.makedirs(res, exist_ok=True)
    text = """NLAYER 3
NODES 3
TIME_STEP 1
SNOW_STEP 1
STARTYEAR 1949
STARTMONTH 01
STARTDAY 01
STARTHOUR %02d
ENDYEAR 1949
ENDMONTH 01
ENDDAY 10
FULL_ENERGY FALSE
FROZEN_SOIL FALSE
FORCING1 %s/forcings/full_data_
FORCE_FORMAT ASCII
N_TYPES 8
FORCE_TYPE PREC
FORCE_TYPE AIR_TEMP
FORCE_TYPE SHORTWAVE
FORCE_TYPE LONGWAVE
FORCE_TYPE DENSITY
FORCE_TYPE PRESSURE
FORCE_TYPE VP
FORCE_TYPE WIND
FORCE_DT 1
FORCEYEAR 1949
FORCEMONTH 01
FORCEDAY 01
FORCEHOUR 00
GRID_DECIMAL 4
WIND_H 10.0
SOIL %s/parameters/Stehekin_soil.txt
BASEFLOW ARNO
JULY_TAVG_SUPPLIED FALSE
ORGANIC_FRACT FALSE
VEGLIB %s/parameters/Stehekin_veglib.txt
VEGPARAM %s/parameters/Stehekin_vegparam.txt
ROOT_ZONES 3
VEGPARAM_LAI TRUE
LAI_SRC FROM_VEGPARAM
SNOW_BAND 5 %s/parameters/Stehekin_snowbands.txt
RESULT_DIR %s
""" % (starthour, s, s, s, s, s, res)
    gf = os.path.join(tmp, 'global_%s.txt' % name)
    open(gf, 'w').write(text + output)
    return gf, res


def _run_42d(gf):
    r = subprocess.run([VIC42D, '-g', gf], capture_output=True, text=True,
                       timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def _cells_42d(soil):
    """(lat, lon, layer depths [layer, cell]) in the soil file's order --
    the order vicNl runs the cells in."""
    rows = [ln.split() for ln in open(soil) if ln.strip()
            and not ln.startswith('#') and ln.split()[0] == '1']
    lat = np.array([float(r[2]) for r in rows])
    lon = np.array([float(r[3]) for r in rows])
    depth = np.array([[float(r[22 + k]) for r in rows] for k in range(3)])
    return lat, lon, depth


def _init_moist_42d(soil):
    """The soil file's init_moist [layer, cell] (mm), in the file's order."""
    rows = [ln.split() for ln in open(soil) if ln.strip()
            and not ln.startswith('#') and ln.split()[0] == '1']
    return np.array([[float(r[18 + k]) for r in rows] for k in range(3)])


def _steps_42d(res, prefix, lat, lon, dec=4):
    return H.outputs_from_42d([os.path.join(res, '%s_%.*f_%.*f'
                                            % (prefix, dec, a, dec, b))
                               for a, b in zip(lat, lon)])


@pytest.fixture(scope='module')
def stehekin_steps_42d(tmp_path_factory):
    """VIC 4.2.d on Stehekin, 16 cells, 1-hour water balance, 1949-01-01 ..
    01-10: its per-step output (OUT_STEP 0) from 00:00 and from 12:00."""
    if not os.path.exists(VIC42D):
        pytest.skip('%s is not built' % VIC42D)
    if not os.path.isdir(STEHE_CLASSIC):
        pytest.skip('samples/data/classic/Stehekin is not present')
    tmp = str(tmp_path_factory.mktemp('stehekin42d'))
    lat, lon, depth = _cells_42d(os.path.join(STEHE_CLASSIC, 'parameters',
                                              'Stehekin_soil.txt'))
    steps = {}
    for h in (0, 12):
        gf, res = _stehekin_global_42d(
            tmp, 'steps%02d' % h, 'OUT_STEP 0\nPRT_HEADER TRUE\n'
            + _outfiles_42d([('all', STEP_VARS_42D)]), starthour=h)
        _run_42d(gf)
        steps[h] = _steps_42d(res, 'all', lat, lon)
    return dict(tmp=tmp, lat=lat, lon=lon, depth=depth, steps=steps)


MISC_VARS_42D = ('OUT_AERO_COND', 'OUT_AERO_RESIST', 'OUT_REL_HUMID')
#: case -> (STARTHOUR, output section, carry)
STEHE_CASES_42D = {
    'daily': (0, 'OUT_STEP 24\n' + _outfiles_42d(
        [('fluxes', FLUX_VARS_42D), ('misc', MISC_VARS_42D)]), False),
    'daily as written (%.4f, header)': (0, 'OUT_STEP 24\nPRT_HEADER TRUE\n'
                                        + _outfiles_42d(
        [('fluxes', FLUX_VARS_42D), ('misc', MISC_VARS_42D)], fmt=''),
        False),
    'OUT_STEP 6': (0, 'OUT_STEP 6\n' + _outfiles_42d(
        [('fluxes', FLUX_VARS_42D), ('misc', MISC_VARS_42D)]), False),
    'OUT_STEP 7, carried': (0, 'OUT_STEP 7\n' + _outfiles_42d(
        [('fluxes', FLUX_VARS_42D)]), True),
    'daily from 12:00, carried': (12, 'OUT_STEP 24\n' + _outfiles_42d(
        [('fluxes', FLUX_VARS_42D)]), True),
    'BINARY, header, types and mults': (
        0, 'OUT_STEP 24\nBINARY_OUTPUT TRUE\nPRT_HEADER TRUE\n'
        + _outfiles_42d([('fluxes', ('OUT_PREC', 'OUT_AIR_TEMP',
                                     'OUT_SOIL_MOIST', 'OUT_EVAP', 'OUT_SWE',
                                     'OUT_AERO_RESIST'))], extra={
            'OUT_PREC': ' %.4f OUT_TYPE_USINT 40',
            'OUT_AIR_TEMP': ' %.4f OUT_TYPE_SINT 100',
            'OUT_SOIL_MOIST': ' %.4f OUT_TYPE_DOUBLE',
            'OUT_EVAP': ' %.4f OUT_TYPE_FLOAT 2.5',
            'OUT_SWE': '', 'OUT_AERO_RESIST': ''}), False),
    'ALMA_OUTPUT': (0, 'OUT_STEP 24\nALMA_OUTPUT TRUE\n' + _outfiles_42d(
        [('alma', ('OUT_PREC', 'OUT_EVAP', 'OUT_RUNOFF', 'OUT_SUB_SNOW',
                   'OUT_SUB_CANOP', 'OUT_SOIL_TEMP', 'OUT_AIR_TEMP',
                   'OUT_PRESSURE', 'OUT_VP', 'OUT_SWE',
                   'OUT_SOIL_MOIST'))]), False),
    'no N_OUTFILES (the default files)': (0, 'OUT_STEP 24\n', False),
}


@pytest.mark.parametrize('case', list(STEHE_CASES_42D))
def test_stehekin_42d_files_are_vicNls_to_the_byte(stehekin_steps_42d,
                                                   case):
    """VIC 4.2.d's own per-step output (%.17g, so the doubles themselves),
    aggregated and written by torchvic.io.history under VIC 4.2.d's global
    file for `case`, is the file vicNl writes for that case: every cell's
    file identical byte for byte -- ASCII at %.17g and at the default %.4f,
    the dates, the header, BINARY with its casts and mults, ALMA_OUTPUT,
    the default files, SKIP-less sub-daily stamps, and the partial period
    vicNl carries from one cell into the next (carry=True; without it the
    cells after the first differ, and by how much is printed)."""
    import filecmp
    st = stehekin_steps_42d
    start, output, carry = STEHE_CASES_42D[case]
    name = 'c%d' % list(STEHE_CASES_42D).index(case)
    gf, res = _stehekin_global_42d(st['tmp'], name, output, starthour=start)
    _run_42d(gf)
    out, times = st['steps'][start]
    spec = H.parse_output_42d(gf, log=lambda *a: None)
    assert H.convention_of(H._as_gp(gf)) == 'vic4.2d'
    mine = os.path.join(st['tmp'], name + '_mine')
    paths = H.write_42d(out, times, st['lat'], st['lon'], spec, mine,
                        carry=carry, log=lambda *a: None)
    theirs = sorted(f for f in os.listdir(res))
    assert sorted(os.path.basename(p) for p in paths) == theirs
    same = [filecmp.cmp(p, os.path.join(res, os.path.basename(p)),
                        shallow=False) for p in paths]
    assert all(same), [os.path.basename(p) for p, s in zip(paths, same)
                       if not s][:4]
    worst = 0.0
    if not spec.binary:
        for p in paths:
            cols = None if spec.prt_header else [
                c for f in spec.files if os.path.basename(p).startswith(
                    f.prefix + '_') for nm in f.names
                for c in ([nm] if nm not in LAYERED_42D
                          else ['%s_%d' % (nm, k) for k in range(3)])]
            _, a = H.read_ascii_42d(p, cols)
            _, b = H.read_ascii_42d(os.path.join(res, os.path.basename(p)),
                                    cols)
            for k in a:
                worst = max(worst, float(np.nanmax(np.abs(a[k] - b[k]))))
    msg = ('%s: %d files, all identical to vicNl\'s; max |value diff| %.1e'
           % (case, len(paths), worst))
    if carry:
        clean = os.path.join(st['tmp'], name + '_clean')
        cp = H.write_42d(out, times, st['lat'], st['lon'], spec, clean,
                         carry=False, log=lambda *a: None)
        differ = [os.path.basename(p) for p in cp if not filecmp.cmp(
            p, os.path.join(res, os.path.basename(p)), shallow=False)]
        first = os.path.basename(cp[0])
        assert first not in differ and differ     # cell 1 carries nothing
        d = 0.0
        cols = [c for f in spec.files for nm in f.names
                for c in ([nm] if nm != 'OUT_SOIL_MOIST'
                          else ['%s_%d' % (nm, k) for k in range(3)])]
        for b in differ:
            _, x = H.read_ascii_42d(os.path.join(clean, b), cols)
            _, y = H.read_ascii_42d(os.path.join(res, b), cols)
            d = max(d, float(np.abs(x['OUT_PREC'][0] - y['OUT_PREC'][0])))
        msg += ('; each cell started clean instead, %d of %d files differ '
                '(their first records\' OUT_PREC by up to %.4g mm: vicNl\'s '
                'holds the previous cell\'s last hours)'
                % (len(differ), len(cp), d))
    print(msg)


def test_stehekin_42d_moistfract(stehekin_steps_42d):
    """MOISTFRACT TRUE: soil moisture / (depth * 1000) at each step, then
    aggregated. vicNl divides each tile before summing the tiles and this
    divides the cell, so the two agree to rounding (measured), not bits."""
    st = stehekin_steps_42d
    gf, res = _stehekin_global_42d(
        st['tmp'], 'moist', 'OUT_STEP 24\nMOISTFRACT TRUE\n' + _outfiles_42d(
            [('moist', ('OUT_SOIL_MOIST', 'OUT_SOIL_LIQ',
                        'OUT_DELSOILMOIST'))]))
    _run_42d(gf)
    out, times = st['steps'][0]
    spec = H.parse_output_42d(gf, log=lambda *a: None)
    mine = os.path.join(st['tmp'], 'moist_mine')
    init = _init_moist_42d(os.path.join(STEHE_CLASSIC, 'parameters',
                                        'Stehekin_soil.txt'))
    H.write_42d(out, times, st['lat'], st['lon'], spec, mine,
                depth=st['depth'], log=lambda *a: None, init_moist=init)
    cols = ['OUT_SOIL_MOIST_%d' % k for k in range(3)] + \
        ['OUT_SOIL_LIQ_%d' % k for k in range(3)] + ['OUT_DELSOILMOIST']
    worst = worst_del = big_del = 0.0
    for f in os.listdir(res):
        _, a = H.read_ascii_42d(os.path.join(mine, f), cols)
        _, b = H.read_ascii_42d(os.path.join(res, f), cols)
        for k in cols[:-1]:
            assert 0.05 < float(b[k].min()) and float(b[k].max()) < 0.6, k
            worst = max(worst, float(np.max(np.abs(a[k] / b[k] - 1))))
        # a change in summed fractions (put_data.c:586-607), not mm
        k = 'OUT_DELSOILMOIST'
        worst_del = max(worst_del, float(np.max(np.abs(a[k] - b[k]))))
        big_del = max(big_del, float(np.max(np.abs(b[k]))))
    # read_soilparam.c:342 rounds the depth to mm in single precision
    assert H.depth_42d([0.3, 1.5]).tolist() == [float(np.float32(0.3)), 1.5]
    assert worst < 1e-14
    # the change is a difference of two sums of fractions of order 1, so
    # its rounding is theirs (mm, the defect, would be ~1000x bigger)
    assert big_del < 1.0 and worst_del < 1e-14
    # without the initial state it is not guessed: written as nan
    nan_dir = os.path.join(st['tmp'], 'moist_noinit')
    first = H.write_42d(out, times, st['lat'], st['lon'], spec, nan_dir,
                        depth=st['depth'], log=lambda *a: None)[0]
    _, c = H.read_ascii_42d(first, cols)
    assert np.isnan(c['OUT_DELSOILMOIST']).all()
    print('MOISTFRACT: %d files, max relative |torchvic.io.history - vicNl| '
          '%.1e (volumetric fractions); OUT_DELSOILMOIST max |diff| %.1e '
          'of max |value| %.3g' % (len(os.listdir(res)), worst, worst_del,
                                    big_del))


def test_findleylake_42d_skipyear(tmp_path):
    """SKIPYEAR 1 from 1971-07-01: vicNl skips 365 days (1971's length, the
    year the skipped span starts in, though it holds 29 Feb 1972) and
    writes from 1972-06-30. torchvic.io.history, fed vicNl's per-step output,
    writes the same file byte for byte, and without SKIPYEAR the whole
    run."""
    import filecmp
    if not os.path.exists(VIC42D):
        pytest.skip('%s is not built' % VIC42D)
    if not os.path.isdir(FINDLEY_CLASSIC):
        pytest.skip('data/VIC_sample_data/classic/FindleyLake is not present')
    F = FINDLEY_CLASSIC
    base = """NLAYER 3
NODES 3
TIME_STEP 1
SNOW_STEP 1
STARTYEAR 1971
STARTMONTH 07
STARTDAY 01
STARTHOUR 00
ENDYEAR 1972
ENDMONTH 12
ENDDAY 31
FULL_ENERGY FALSE
FROZEN_SOIL FALSE
FORCING1 {F}/forcings/livneh_NAmerExt_15Oct2014.FindleyLake_
FORCE_FORMAT ASCII
N_TYPES 7
FORCE_TYPE PREC
FORCE_TYPE AIR_TEMP
FORCE_TYPE SHORTWAVE
FORCE_TYPE LONGWAVE
FORCE_TYPE PRESSURE
FORCE_TYPE VP
FORCE_TYPE WIND
FORCE_DT 1
FORCEYEAR 1970
FORCEMONTH 01
FORCEDAY 01
FORCEHOUR 00
GRID_DECIMAL 4
WIND_H 10.0
SOIL {F}/parameters/soil_param.FindleyLake
BASEFLOW ARNO
VEGLIB {F}/parameters/veg_lib_IGBP
VEGPARAM {F}/parameters/veg_param.FindleyLake
ROOT_ZONES 3
VEGPARAM_LAI TRUE
VEGPARAM_VEGCOVER TRUE
VEGPARAM_ALB TRUE
LAI_SRC FROM_VEGPARAM
VEGCOVER_SRC FROM_VEGPARAM
ALB_SRC FROM_VEGPARAM
SNOW_BAND 1
""".format(F=F)
    names = ('OUT_PREC', 'OUT_EVAP', 'OUT_SWE', 'OUT_SOIL_MOIST',
             'OUT_AIR_TEMP')
    runs = {}
    for name, sec in (('steps', 'OUT_STEP 0\nPRT_HEADER TRUE\n'),
                      ('skip1', 'OUT_STEP 24\nSKIPYEAR 1\n'),
                      ('skip0', 'OUT_STEP 24\n')):
        res = os.path.join(str(tmp_path), name)
        os.makedirs(res)
        gf = os.path.join(str(tmp_path), name + '.txt')
        open(gf, 'w').write(base + 'RESULT_DIR %s\n' % res + sec
                            + _outfiles_42d([('fluxes', names)]))
        _run_42d(gf)
        runs[name] = (gf, res)
    lat, lon, _ = _cells_42d(os.path.join(F, 'parameters',
                                          'soil_param.FindleyLake'))
    out, times = _steps_42d(runs['steps'][1], 'fluxes', lat, lon)
    assert len(times) == 550 * 24
    for name in ('skip1', 'skip0'):
        gf, res = runs[name]
        spec = H.parse_output_42d(gf, log=lambda *a: None)
        mine = os.path.join(str(tmp_path), name + '_mine')
        paths = H.write_42d(out, times, lat, lon, spec, mine,
                            log=lambda *a: None)
        for p in paths:
            assert filecmp.cmp(p, os.path.join(res, os.path.basename(p)),
                               shallow=False), (name, p)
    first = open(paths[0]).readline().split('\t')[:3]
    skipped = open(os.path.join(runs['skip1'][1],
                                os.path.basename(paths[0]))).readlines()
    assert open(os.path.join(runs['skip1'][1], os.path.basename(
        paths[0]))).readline().split('\t')[:3] == ['1972', '06', '30']
    assert first == ['1971', '07', '01'] and len(skipped) == 550 - 365
    print('FindleyLake SKIPYEAR 1 from 1971-07-01: vicNl writes %d of 550 '
          'days, from 1972-06-30; torchvic.io.history\'s file is identical'
          % len(skipped))


def test_vic_gpu_stehekin_in_the_42d_convention(stehekin_steps_42d):
    """TorchVIC on Stehekin (the image files, the water balance, 1-hour
    steps) written by VIC 4.2.d's global file: the convention is picked
    from that file by itself, and the files are vicNl's -- same names,
    same lines, same date fields -- with TorchVIC's values. Those values
    are TorchVIC's per-step output aggregated as 4.2.d aggregates
    (aggregate_output, convention='vic4.2d'), exactly; how far they are
    from VIC 4.2.d's is printed, day and step, as the physics of two model
    versions and not as a property of the aggregation."""
    st = stehekin_steps_42d
    if not os.path.isdir(_up('samples', 'data', 'image', 'Stehekin')):
        pytest.skip('samples/data/image/Stehekin is not present')
    from torchvic import case
    old = os.environ.get('test_data_dir')
    os.environ['test_data_dir'] = os.path.dirname(os.path.dirname(
        _up('samples', 'data', 'image', 'Stehekin')))
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 8))
    try:
        c = case.load(STEHE_IMAGE_GLOBAL, device=CPU, no_energy=True,
                      log=lambda *a: None)
        with torch.no_grad():
            out = c.run(p=c.params())
    finally:
        torch.set_num_threads(threads)
        if old is None:
            os.environ.pop('test_data_dir', None)
        else:
            os.environ['test_data_dir'] = old
    assert c.model_steps == 24 and not c.full_energy
    lines = []
    for label, key in (('daily', 'daily'),
                       ('daily as written',
                        'daily as written (%.4f, header)')):
        start, output, _ = STEHE_CASES_42D[key]
        name = 'c%d' % list(STEHE_CASES_42D).index(key)
        gf, res = _stehekin_global_42d(st['tmp'], name + 'g', output)
        _run_42d(gf)
        said = []
        mine = os.path.join(st['tmp'], name + '_vicgpu')
        paths = c.write_history(out, result_dir=mine, output_global=gf,
                                log=said.append)
        assert sorted(os.path.basename(p) for p in paths) == \
            sorted(os.listdir(res))
        # (a 4.2.d global file asks for none of the rows out-vars forms, so
        # the water-balance run does not form OUT_LATENT: out_vars)
        assert any('OUT_LATENT' in m and 'nan' in m for m in said)
        for p in paths:
            a = open(p).read().splitlines()
            b = open(os.path.join(res, os.path.basename(p))).read() \
                .splitlines()
            assert len(a) == len(b)
            dates = [ln.split('\t')[:3] for ln in a if not ln.startswith('#')]
            assert dates == [ln.split('\t')[:3] for ln in b
                             if not ln.startswith('#')]
            assert [ln for ln in a if ln.startswith('#')] == \
                [ln for ln in b if ln.startswith('#')]
        lines.append('%s: %d files named, dated and headed as vicNl\'s'
                     % (label, len(paths)))
    # the values: TorchVIC's steps, aggregated by 4.2.d's arithmetic
    day, stamps, tail = c.aggregate(out, 'NDAYS 1', convention='vic4.2d',
                                    log=lambda *a: None)
    assert tail is None
    assert stamps == [dt.datetime(1949, 1, d, 23) for d in range(1, 11)]
    recs = [(24 * d, 24 * d + 23) for d in range(10)]
    lat, lon = np.asarray(c.dom.lat), np.asarray(c.dom.lon)
    key = {'OUT_EVAP': 'et', 'OUT_RUNOFF': 'runoff',
           'OUT_BASEFLOW': 'baseflow', 'OUT_SWE': 'swe',
           'OUT_AIR_TEMP': 'air_temp'}
    name = 'c%d' % list(STEHE_CASES_42D).index('daily')
    mine = os.path.join(st['tmp'], name + '_vicgpu')
    res = os.path.join(st['tmp'], name + 'g')
    cols = [cc for nm in FLUX_VARS_42D for cc in (
        [nm] if nm != 'OUT_SOIL_MOIST' else
        ['OUT_SOIL_MOIST_%d' % k for k in range(3)])]
    vic_steps, _ = st['steps'][0]
    lines.append('  %-15s %-4s | daily TorchVIC vs VIC 4.2.d (basin, max '
                 '|d| per cell) | per step' % ('variable', 'agg'))
    for nm in FLUX_VARS_42D:
        if nm == 'OUT_LATENT':
            continue
        agg = H.default_agg(nm, 'vic4.2d')
        mine_v, vic_v = [], []
        for a_, b_ in zip(lat, lon):
            f = 'fluxes_%.4f_%.4f' % (a_, b_)
            x = H.read_ascii_42d(os.path.join(mine, f), cols)[1]
            y = H.read_ascii_42d(os.path.join(res, f), cols)[1]
            ks = [k for k in cols if k == nm or k.startswith(nm + '_')]
            mine_v.append(np.stack([x[k] for k in ks], -1))
            vic_v.append(np.stack([y[k] for k in ks], -1))
        mv = np.stack(mine_v, 1)
        vv = np.stack(vic_v, 1).sum(-1)
        src_t = H.outvar_series(out, nm)[0]
        want = H.aggregate_42d(src_t, recs, agg, 24)
        if nm in key:                       # the API is the same numbers
            assert torch.equal(day[key[nm]], want), nm
        want = want.numpy().reshape(mv.shape)
        # the file holds TorchVIC's steps aggregated by 4.2.d's arithmetic
        assert np.array_equal(mv, want), nm
        mv = mv.sum(-1)
        # the file's cells are the domain's; vicNl's the soil file's order
        # -- matched by name above
        TA = vv.sum() if agg == 'SUM' else vv.mean()
        TB = mv.sum() if agg == 'SUM' else mv.mean()
        rel = (TB - TA) / abs(TA) if abs(TA) > 1e-12 else 0.0
        order = [int(np.flatnonzero((np.abs(st['lat'] - a) < 1e-9)
                                    & (np.abs(st['lon'] - b) < 1e-9))[0])
                 for a, b in zip(lat, lon)]
        vs = vic_steps[nm].numpy()[:, order]
        vs = vs.sum(-1) if vs.ndim == 3 else vs
        src = src_t.numpy()
        src = src.sum(-1) if src.ndim == 3 else src
        SA = vs.sum() if agg == 'SUM' else vs.mean()
        SB = src.sum() if agg == 'SUM' else src.mean()
        srel = (SB - SA) / abs(SA) if abs(SA) > 1e-12 else 0.0
        lines.append('  %-15s %-4s | %+.3f%% max %.3g | %+.3f%% max %.3g'
                     % (nm, agg, 100 * rel, float(np.abs(mv - vv).max()),
                        100 * srel, float(np.abs(src - vs).max())))
        assert np.isfinite(mv).all()
    print('TorchVIC (image files, 5.x core, water balance) on Stehekin '
          '1949-01-01..10, written by VIC 4.2.d\'s global file:\n'
          + '\n'.join(lines))


def test_42d_against_vic5_rules_on_the_same_steps(stehekin_steps_42d):
    """The same per-step values (VIC 4.2.d's, Stehekin), aggregated by each
    version's rules: what the conventions differ by, measured. SUM and END
    are the same numbers; AVG differs in the last bits (sum of x/n against
    sum/n); OUT_AERO_RESIST is a different quantity (1 / mean conductance
    against the mean resistance); and a sub-daily record is stamped at its
    last step instead of its first."""
    out, times = stehekin_steps_42d['steps'][0]
    recs5, _ = H.windows(times, 3600, H.Freq('NDAYS', 1))
    recs4, _ = H.windows_42d(len(times), 24)
    assert recs5 == recs4
    lines = []
    for nm in ('OUT_PREC', 'OUT_EVAP', 'OUT_SOIL_MOIST', 'OUT_SWE',
               'OUT_AIR_TEMP', 'OUT_LATENT', 'OUT_REL_HUMID'):
        x = out[nm]
        a5 = H.aggregate(x, recs5, H.default_agg(nm)).numpy()
        a4 = H.aggregate_42d(x, recs4, H.default_agg(nm, 'vic4.2d'),
                             24).numpy()
        if H.default_agg(nm) in ('SUM', 'END'):
            assert np.array_equal(a4, a5), nm
            continue
        rel = float(np.max(np.abs(a4 - a5) / np.maximum(np.abs(a5), 1e-300)))
        assert rel < 1e-14, nm
        lines.append('%s AVG: %d of %d daily values differ, by at most %.1e '
                     'relative' % (nm, int((a4 != a5).sum()), a4.size, rel))
    cond = out['OUT_AERO_COND']
    r4 = (1.0 / H.aggregate_42d(cond, recs4, 'AVG', 24)).numpy()
    r5 = H.aggregate(out['OUT_AERO_RESIST'], recs5, 'AVG').numpy()
    ratio = r5 / r4
    assert (ratio >= 1 - 1e-12).all()       # the mean of 1/c >= 1/mean(c)
    lines.append('OUT_AERO_RESIST: VIC 5\'s mean resistance over VIC '
                 '4.2.d\'s 1/(mean conductance): median %.3f, max %.3f'
                 % (float(np.median(ratio)), float(ratio.max())))
    r6, _ = H.windows(times, 3600, H.Freq('NHOURS', 6))
    assert [times[a].hour for a, _ in r6[:4]] == [0, 6, 12, 18]
    assert [times[b].hour for _, b in H.windows_42d(len(times), 6)[0][:4]] \
        == [5, 11, 17, 23]
    lines.append('6-hourly stamps: VIC 5 00 06 12 18 (period start), VIC '
                 '4.2.d 05 11 17 23 (last step)')
    print('Stehekin, VIC 4.2.d\'s per-step values, daily by each version\'s '
          'rules:\n  ' + '\n  '.join(lines))


def test_cli_run_writes_the_42d_files(stehekin_steps_42d):
    """torchvic run: --output-global <a VIC 4.2.d file> writes
    that file's OUTFILEs as vicNl names and dates them; --agg-convention
    vic4.2d writes the global file's own VIC 5 stream (AGGFREQ NHOURS 1:
    OUT_STEP 1, so an hour column) as 4.2.d's per-cell ASCII files."""
    st = stehekin_steps_42d
    if not os.path.isdir(_up('samples', 'data', 'image', 'Stehekin')):
        pytest.skip('samples/data/image/Stehekin is not present')
    tmp = os.path.join(st['tmp'], 'cli')
    os.makedirs(tmp)
    key = 'daily as written (%.4f, header)'
    start, output, _ = STEHE_CASES_42D[key]
    gf, res = _stehekin_global_42d(tmp, 'vic', output)
    _run_42d(gf)
    study = os.path.join(tmp, 'study.yaml')
    open(study, 'w').write('case:\n  global_param: %s\n  device: cpu\n'
                           '  no_energy: true\n' % STEHE_IMAGE_GLOBAL)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
               test_data_dir=os.path.dirname(os.path.dirname(
                   _up('samples', 'data', 'image', 'Stehekin'))))
    cli = [sys.executable, '-m', 'torchvic', 'run', study]
    a = os.path.join(tmp, 'a')
    r = subprocess.run(cli + ['--output-global', gf, '--result-dir', a],
                       env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "VIC 4.2.d's convention" in r.stdout
    names = sorted(f for f in os.listdir(a) if not f.startswith('.'))
    assert names == sorted(os.listdir(res))
    for f in names:
        x = open(os.path.join(a, f)).read().splitlines()
        y = open(os.path.join(res, f)).read().splitlines()
        assert [ln.split('\t')[:3] for ln in x] == \
            [ln.split('\t')[:3] for ln in y]
    b = os.path.join(tmp, 'b')
    r = subprocess.run(cli + ['--agg-convention', 'vic4.2d', '--result-dir',
                              b], env=env, capture_output=True, text=True,
                       timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    files = sorted(f for f in os.listdir(b) if not f.startswith('.'))
    assert len(files) == 16 and files[0].startswith('fluxes_48.19_-120.')
    rows = open(os.path.join(b, files[0])).read().splitlines()
    assert len(rows) == 240
    assert rows[0].split('\t')[:4] == ['1949', '01', '01', '00']
    assert rows[-1].split('\t')[:4] == ['1949', '01', '10', '23']
    print('CLI: --output-global -> %d files as vicNl names and dates them; '
          '--agg-convention vic4.2d -> %d hourly files' % (len(names),
                                                            len(files)))
