"""A study file: one YAML (or JSON) file that names the data and, optionally,
observations to report the run against.

The data are a *native VIC 5 image-driver global parameter file*, so the
domain, the parameters and the forcing are the ones VIC itself reads and
nothing is duplicated here.

    case:
      global_param: forcing/global_param.WalnutGulch.txt
      device: auto             # auto | cpu | cuda | cuda:0
      dtype: float64           # float64 | float32
      core: 5.1.0              # 5.1.0 | 5.0.1 (default: 5.1.0, the current core)
      exact_math: false        # true: glibc pow/exp/sqrt on the CPU (xmath.py)
      steps: 2928              # 0 or absent = the whole forcing
      first_step: 0
      no_energy: false
      no_veg: false
      daily: false             # a daily model step (MODEL_STEPS_PER_DAY 1)
      daily_day: forcing       # forcing | off_gmt
      mtclim_margin: 90        # days of daily forcing read around a window

    observations:              # optional
      runoff:
        file: obs/runoff_daily.txt
        freq: daily            # daily | step  (daily: VIC's NDAYS 1 rule --
                               # fluxes summed, states end of day, rates
                               # averaged; `agg: sum|avg|end|...` overrides)
        cells: mean            # mean | a cell index | a list of them

    report:                    # optional
      metrics: [kge, nse]      # any of torchvic.metrics.METRICS (default kge)

Everything has a default except `case.global_param`, so the smallest usable
file is two lines. `load()` returns a `Study`, whose `case` is ready to run
and whose `report(out)` scores a run against the observations.
"""
import csv
import datetime as _dt
import json
import os

import numpy as np
import torch

from . import case as _case, metrics
from .io import history

DTYPES = {'float64': torch.float64, 'float32': torch.float32,
          'double': torch.float64, 'float': torch.float32}


def _read(path):
    with open(path) as fh:
        text = fh.read()
    if path.endswith(('.json',)):
        return json.loads(text)
    try:
        import yaml
    except ImportError:                                  # pragma: no cover
        raise ImportError('reading %s needs PyYAML (pip install pyyaml), or '
                          'write the file as .json' % path)
    return yaml.safe_load(text)


def _resolve(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)


# ---------------------------------------------------------------------------
# Dated series. Every observation is placed by its DATE LABEL, never by its
# position in a file: a gauge record that writes a gap as a missing row would
# otherwise move every later value one day earlier.
# ---------------------------------------------------------------------------
def to_days(times):
    """Anything time-like -> datetime64[D] (the calendar day it falls on)."""
    t = np.asarray(times)
    if t.dtype.kind == 'M':
        return t.astype('datetime64[D]')
    if t.dtype.kind == 'O' and t.size and hasattr(t.flat[0], 'year') \
            and not isinstance(t.flat[0], (_dt.date, str)):
        # cftime under a standard calendar
        return np.array(['%04d-%02d-%02d' % (x.year, x.month, x.day)
                         for x in t.flat], dtype='datetime64[D]')
    out = []
    for x in t.flat:
        if isinstance(x, _dt.datetime):
            x = x.date()
        out.append(np.datetime64(x if isinstance(x, _dt.date) else
                                 str(x)[:10], 'D'))
    return np.array(out, dtype='datetime64[D]')


def match_days(src_days, model_days, what, allow_unsorted=False):
    """(src_idx, dst_idx): the entries of `src_days` that are model days.

    Label matching, never slicing by count. `src_days` must be strictly
    increasing -- a repeated date has two values and no rule picks one, and
    an unsorted axis is usually two records spliced together -- unless
    `allow_unsorted` (the order is then taken from the labels).
    """
    src = np.asarray(src_days, dtype='datetime64[D]')
    dst = np.asarray(model_days, dtype='datetime64[D]')
    if src.size > 1:
        d = np.diff(src).astype(np.int64)
        if (d == 0).any():
            dup = src[1:][d == 0]
            raise ValueError('%s: %d repeated date(s), first %s; refusing to '
                             'pick one' % (what, int(dup.size), dup[0]))
        if (d < 0).any() and not allow_unsorted:
            raise ValueError('%s: dates are not in order (%d step(s) '
                             'backwards, first at %s); sort the file or set '
                             'allow_unsorted' % (what, int((d < 0).sum()),
                                                 src[1:][d < 0][0]))
    order = np.argsort(src, kind='stable')
    s_sorted = src[order]
    if s_sorted.size > 1 and (np.diff(s_sorted).astype(np.int64) == 0).any():
        raise ValueError('%s: repeated dates' % what)
    pos = np.searchsorted(s_sorted, dst)
    pos_c = np.clip(pos, 0, max(s_sorted.size - 1, 0))
    hit = (s_sorted.size > 0) & (pos < s_sorted.size) & \
        (s_sorted[pos_c] == dst) if s_sorted.size else \
        np.zeros(dst.shape, dtype=bool)
    dst_idx = np.nonzero(hit)[0]
    src_idx = order[pos_c[hit]]
    return src_idx, dst_idx


def _parse_date(s, fmt=None):
    """ISO 8601 (YYYY-MM-DD, optionally with a time), or `fmt`.

    Nothing else is guessed: 03-12-1985 is 3 December in one convention and
    12 March in the other, and guessing wrong moves a record by months while
    every date still parses. A file in another format must say which with
    `date_format` (strptime syntax, e.g. %d-%m-%Y).
    """
    s = str(s).strip()
    if fmt:
        return _dt.datetime.strptime(s, fmt).date()
    try:
        return _dt.date.fromisoformat(s[:10])
    except ValueError:
        raise ValueError('date %r is not ISO 8601 (YYYY-MM-DD); declare the '
                         'file\'s format with date_format, e.g. %%d-%%m-%%Y'
                         % s)


def read_csv_dates(path, value_column, date_column=None, date_format=None,
                   fill_values=(), what=None):
    """(days datetime64[D], values float64) from a date,value CSV.

    The date column is required: `date_column`, or the first column called
    date/Date/DATE/time. A value that is blank, `nan` or one of
    `fill_values` is NaN. Duplicate and unsorted dates are left for
    `match_days` to refuse.
    """
    what = what or os.path.basename(path)
    with open(path, newline='') as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames or []
        dc = date_column
        if dc is None:
            for c in cols:
                if c.strip().lower() in ('date', 'time', 'datetime', 'day'):
                    dc = c
                    break
        if dc is None or dc not in cols:
            raise ValueError('%s has no date column (columns %s). Reading a '
                             'series by row position is exactly the defect '
                             'this reader exists to refuse: a missing row '
                             'moves every later value earlier'
                             % (what, cols))
        if value_column not in cols:
            raise ValueError('%s has no column %r (columns %s)'
                             % (what, value_column, cols))
        fills = set(float(x) for x in (fill_values or ()))
        days, vals = [], []
        for row in rd:
            d = _parse_date(row[dc], date_format)
            s = (row[value_column] or '').strip()
            try:
                v = float(s) if s else np.nan
            except ValueError:
                v = np.nan
            if v in fills:
                v = np.nan
            days.append(np.datetime64(d, 'D'))
            vals.append(v)
    return np.array(days, dtype='datetime64[D]'), np.array(vals, np.float64)


def read_series_csv(path, model_dates, value_column, date_column=None,
                    date_format=None, fill_values=(), allow_unsorted=False,
                    what=None):
    """A date,value CSV on the model's days: ([n] values, NaN where the file
    has no row, report dict)."""
    what = what or os.path.basename(path)
    days, vals = read_csv_dates(path, value_column, date_column, date_format,
                                fill_values, what)
    md = np.asarray(model_dates, dtype='datetime64[D]')
    si, di = match_days(days, md, what, allow_unsorted=allow_unsorted)
    out = np.full(md.shape, np.nan)
    out[di] = vals[si]
    rep = dict(file=path, rows=int(days.size), rows_in_window=int(si.size),
               finite_in_window=int(np.isfinite(out).sum()),
               window_days=int(md.size),
               first=str(days.min()) if days.size else None,
               last=str(days.max()) if days.size else None)
    return out, rep


def _series_days(dates, n, freq, model_steps):
    """The day label of each of the `n` values a series is compared with:
    one per model step (a daily model), or one per `model_steps` steps from
    the first (freq daily, which Observation.simulate aggregates that way)."""
    if dates is None:
        raise ValueError('a dated series needs the run\'s dates')
    if freq == 'daily':
        steps = max(int(model_steps or 1), 1)
        pick = [dates[k * steps] for k in range(n) if k * steps < len(dates)]
    else:
        if int(model_steps or 1) != 1:
            raise ValueError('a dated series gives one value per day; the '
                             'model has %d steps a day, so set `freq: daily`'
                             % int(model_steps))
        pick = list(dates[:n])
    days = to_days(pick)
    if days.size < n:
        days = np.concatenate([days, days[-1] + 1 + np.arange(n - days.size)])
    return days


def read_series(path, n, dtype, device, dates=None, freq='step',
                model_steps=1, column=None, date_column=None,
                date_format=None):
    """Observations for a run of `n` values (steps, or days with freq daily).

    Three forms:

    * a CSV with a header and a DATE column (`date,value`): placed by date
      on the run's own days, a missing row being a missing day. This is the
      form for any record with gaps. `column` names the value column when
      there is more than one; dates are ISO 8601 unless `date_format` says
      otherwise (read_series_csv below, which refuses repeated or
      unordered dates).
    * one value per line, by POSITION from the run's first step (or day).
      A blank line or `nan` is a missing value. np.loadtxt, used here
      before, skipped blank lines, so a blank moved every later value one
      step earlier.
    * two columns `index value`, the index counted in steps (or days) from
      the run's first; absent rows are missing.

    Everything outside the window is dropped. Returns [n] with nan where
    unobserved.
    """
    with open(path) as fh:
        lines = [ln.rstrip('\r\n') for ln in fh]
    # comment lines go (np.loadtxt's '#'), and so do trailing comments
    body = [ln.split('#', 1)[0] for ln in lines
            if not ln.lstrip().startswith('#')]
    first = next((ln for ln in body if ln.strip()), '')
    toks = first.replace(',', ' ').split()
    try:
        [float(t) for t in toks]
        header = False
    except ValueError:
        header = True
    v = np.full(n, np.nan)
    if header:
        import csv as _csv
        cols = next(_csv.reader([first]))
        dcol = date_column or next(
            (c for c in cols if c.strip().lower() in
             ('date', 'time', 'datetime', 'day')), None)
        if column is None:
            rest = [c for c in cols if c != dcol]
            if len(rest) != 1:
                raise ValueError('%s has columns %s; name the value column '
                                 'with `column:`' % (path, cols))
            column = rest[0]
        days = _series_days(dates, n, freq, model_steps)
        v, _ = read_series_csv(path, days, column, date_column=dcol,
                                    date_format=date_format)
        return torch.as_tensor(v, dtype=dtype, device=device)
    rows = [ln.replace(',', ' ').split() for ln in body]
    ncol = len(toks)
    if ncol <= 1:
        vals = [float(r[0]) if r else np.nan for r in rows]
        # a file's final newline is not a missing value
        while vals and not rows[len(vals) - 1]:
            vals.pop()
        take = min(n, len(vals))
        v[:take] = vals[:take]
    else:
        a = np.array([[float(x) for x in r[:2]] for r in rows if r])
        idx = a[:, 0].astype(int)
        keep = (idx >= 0) & (idx < n)
        v[idx[keep]] = a[keep, 1]
    return torch.as_tensor(v, dtype=dtype, device=device)


def _agg_name(agg):
    """An observation's `agg:` as one of VIC's AGG_TYPEs, or None."""
    return None if agg is None else history.normalize_agg(agg)


class Observation(object):
    """One observed variable: the series, and how to compare it.

    With `freq: daily` the model's steps are aggregated to the day the way
    VIC 5's AGGFREQ NDAYS 1 aggregates the OUT_* variable the output carries
    (torchvic.io.history.key_agg): a flux such as `et`, `runoff` or
    `discharge_mm` is summed, a store such as `swe` or `soil_liq` is its
    end-of-day value, and a rate or temperature is averaged. `agg` overrides
    that. This used to sum every variable, which made a daily snow or soil
    moisture observation meet eight times the state on a three-hourly model.
    """

    def __init__(self, name, values, freq='step', cells='mean', agg=None):
        self.name, self.values, self.freq, self.cells = name, values, freq, cells
        self.agg = _agg_name(agg)

    @property
    def daily_agg(self):
        """How `freq: daily` reduces a day of steps: `agg`, or VIC's
        default for this output."""
        return self.agg or history.key_agg(self.name)[0]

    def simulate(self, out, case):
        """The model's counterpart of this observation, [time, 1]."""
        if self.name not in out:
            raise KeyError('the model has no output %r; available: %s'
                           % (self.name, ', '.join(sorted(out))))
        x = out[self.name]
        if x.dim() == 3:                       # [time, cell, layer]
            x = x.sum(-1)
        if self.name == 'discharge_mm':
            sim = x[:, :1]                     # the first outlet
        elif self.cells == 'mean':
            sim = x.mean(-1, keepdim=True)
        else:
            idx = torch.as_tensor(
                [self.cells] if isinstance(self.cells, int) else self.cells,
                dtype=torch.long, device=x.device)
            sim = x.index_select(-1, idx).mean(-1, keepdim=True)
        if self.freq == 'daily':
            spd = case.model_steps
            nd = sim.shape[0] // spd
            w = sim[:nd * spd].reshape(nd, spd, -1)
            agg = self.daily_agg
            # the period's own extremes: a comparison with observations has
            # no use for VIC's zero-started MAX/MIN (torchvic.io.history)
            sim = (w.sum(1) if agg == 'SUM' else w.mean(1) if agg == 'AVG'
                   else w[:, -1] if agg == 'END' else w[:, 0] if agg == 'BEG'
                   else w.amax(1) if agg == 'MAX' else w.amin(1))
        return sim

    def target(self, sim):
        """The observations, shaped to broadcast against `sim`."""
        v = self.values[:sim.shape[0]]
        return v.reshape(-1, 1).expand_as(sim)


class Study(object):
    """A loaded study file: the case, and the observations to report on."""

    def __init__(self, cfg, path, log=None, device=None, dtype=None,
                 core=None):
        self.cfg, self.path = cfg, path
        base = os.path.dirname(os.path.abspath(path))
        say = log or (lambda *a: None)
        c = dict(cfg.get('case') or {})
        if 'global_param' not in c:
            raise KeyError('%s: case.global_param is required -- it is the '
                           'native VIC global parameter file' % path)
        if c.get('exact_math'):
            from .core import xmath
            xmath.enable()
        self.device = resolve_device(device or c.get('device', 'auto'))
        dt = dtype or c.get('dtype', 'float64')
        if isinstance(dt, str):
            if dt not in DTYPES:
                raise ValueError('dtype is one of %s, not %r'
                                 % (', '.join(sorted(DTYPES)), dt))
            dt = DTYPES[dt]
        self.dtype = dt
        core = core or c.get('core')
        # Days of a daily forcing read on each side of the window for
        # MTCLIM (case.load's margin_days). It changes the window's forcing
        # -- up to 4.05 W/m2 of shortwave on a 100-day window between 0 and
        # 90 -- so a study can pin it.
        mm = c.get('mtclim_margin')
        self.mtclim_margin = (_case.MTCLIM_MARGIN_DAYS if mm is None
                              else int(mm))
        self.case = _case.load(_resolve(c['global_param'], base),
                               device=self.device, dtype=self.dtype,
                               steps=int(c.get('steps', 0) or 0),
                               first_step=int(c.get('first_step', 0) or 0),
                               no_energy=bool(c.get('no_energy', False)),
                               no_veg=bool(c.get('no_veg', False)),
                               daily=bool(c.get('daily', False)), log=say,
                               margin_days=self.mtclim_margin,
                               daily_day=str(c.get('daily_day', 'forcing')),
                               core=None if core is None else str(core))

        self.observations = {}
        for name, spec in (cfg.get('observations') or {}).items():
            spec = {'file': spec} if isinstance(spec, str) else dict(spec)
            n = self.case.n_steps
            if spec.get('freq', 'step') == 'daily':
                n = self.case.n_steps // self.case.model_steps
            self.observations[name] = Observation(
                name, read_series(_resolve(spec['file'], base), n,
                                  self.case.dtype, self.device,
                                  dates=self.case.dates,
                                  freq=spec.get('freq', 'step'),
                                  model_steps=self.case.model_steps,
                                  column=spec.get('column'),
                                  date_column=spec.get('date_column'),
                                  date_format=spec.get('date_format')),
                freq=spec.get('freq', 'step'), cells=spec.get('cells', 'mean'),
                agg=spec.get('agg'))
        rep = dict(cfg.get('report') or {})
        names = rep.get('metrics', ['kge'])
        self.metrics = [names] if isinstance(names, str) else list(names)
        for m in self.metrics:
            metrics.metric(m)                   # fail now if it is misspelt

    def report(self, out):
        """{variable: {metric: value}} of the run `out` against each
        observation."""
        rows = {}
        for name, ob in self.observations.items():
            sim = ob.simulate(out, self.case)
            for m in self.metrics:
                v = metrics.score(m, sim, ob.target(sim), dim=0)
                rows.setdefault(name, {})[m] = float(v.reshape(-1)[0])
        return rows

    def __repr__(self):
        return ('<Study %s | %s | %d observation(s)>'
                % (os.path.basename(self.path), self.case,
                   len(self.observations)))


def load(path, log=None, device=None, dtype=None, core=None):
    """Read a study file and everything it points at. `device`, `dtype` and
    `core` override the file's own."""
    return Study(_read(path), path, log=log, device=device, dtype=dtype,
                 core=core)


def resolve_device(device):
    """'auto' is the CUDA device with the most free memory, or the CPU when
    there is none; anything else is passed through."""
    device = str(device)
    if device == 'auto':
        if not torch.cuda.is_available():
            return 'cpu'
        best, free_best = 0, -1
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > free_best:
                best, free_best = i, free
        return 'cuda:%d' % best
    return device
