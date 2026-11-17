# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR plugin forcing reader (connector C0, forcing half).

Reads `PLUGIN_FORCE_TYPE` files (sector demand, groundwater and consumption
fractions, EFR, Cv, CO2, DISCHARGE, ...) the way VIC-WUR does, but keeps
every value on the device and fetches the next file ahead of time.

VIC-WUR 3.0 cadence rule (<data root>/build/vicwur3/src_tree/vic/plugins)
  * file `<prefix><yyyy>.nc`, no dot added (general/src/plugin_forcing_utils.c:
    70,134,177,204); one file per model year.
  * cursor (plugin_start_forcing, plugin_forcing_utils.c:168-257):
      - step 0: forcerun = true, offset = 0 (:174-186).
      - new year (:189-213): offset = skip = 0, file reopened.
      - else offset += 1 on a step (STEP), day (DAY), month (MONTH) or year
        (YEAR) change (:214-234).
      - forcerun = true on those same changes, including the year change
        (:236-255).
  * record read = forceskip + forceoffset (e.g. wu_force.c:96-99,
    co2_forcing.c:31-34, lu_force.c:66-68, efr_force.c:158-159), only when
    forcerun is set; otherwise the previously read record is HELD. There is no
    interpolation and no division of the value (a MONTH demand is mm per model
    step as stored).
  * first-year skip (general/src/plugin_get_forcing_file.c:207-251, the
    arithmetic at :232-249): STEP and DAY round((start - file_start) *
    file_steps_per_year / days_per_year_of_file_year); MONTH start_month -
    file_month; YEAR 0.
  * file checks (:92-116, :148-197): the file's first time stamp must equal
    the met forcing's first time stamp; the records per year must match the
    frequency (STEP model_steps_per_day*days, DAY days, MONTH 12, YEAR 1).
  * The values are doubles (get_scatter_nc_field_double); fill values are not
    masked, exactly as C reads them.

GPU design
  * Each year file is read once on a worker thread into pinned memory and
    copied to a device buffer on a side stream; while year y is in use the
    reader prefetches y+1. Switching windows is a device-side event wait, not
    a host sync of the compute stream.
  * `rec_dev[step]` (per-step record within its year window) is a device int
    tensor built once from the C state machine; the lookup is
    `buf.index_select(0, rec_dev[step:step+1])`, a device gather with no
    host synchronisation. `reads(step)` returns the host flag forcerun.
"""
import datetime
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

FORCE_STEP, FORCE_DAY, FORCE_MONTH, FORCE_YEAR = 'STEP', 'DAY', 'MONTH', 'YEAR'
_UNITS = {'days': 86400.0, 'day': 86400.0, 'hours': 3600.0, 'hour': 3600.0,
          'minutes': 60.0, 'minute': 60.0, 'seconds': 1.0, 'second': 1.0}
_DIM = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _leap(y, calendar):
    if calendar in ('noleap', '365_day', 'no_leap'):
        return False
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


def days_in_year(y, calendar='proleptic_gregorian'):
    return 365 + int(_leap(y, calendar))


def _ymd_add(y, m, d, ndays, calendar):
    """Calendar date `ndays` after (y, m, d)."""
    if calendar in ('noleap', '365_day', 'no_leap'):
        doy = sum(_DIM[:m - 1]) + d - 1 + ndays
        y += doy // 365
        doy %= 365
        for mm in range(12):
            if doy < _DIM[mm]:
                return y, mm + 1, doy + 1
            doy -= _DIM[mm]
    dt = datetime.date(y, m, d) + datetime.timedelta(days=int(ndays))
    return dt.year, dt.month, dt.day


def _days_between(a, b, calendar):
    """Days from date a=(y,m,d) to date b."""
    if calendar in ('noleap', '365_day', 'no_leap'):
        f = lambda t: t[0] * 365 + sum(_DIM[:t[1] - 1]) + t[2] - 1
        return f(b) - f(a)
    return (datetime.date(*b) - datetime.date(*a)).days


def step_dates(start, nsteps, steps_per_day, calendar='proleptic_gregorian'):
    """(year, month, day) arrays of the start of every model step
    (C `dmy[current]`). start = (y, m, d, sec)."""
    y0, m0, d0, s0 = start
    dt = 86400 // steps_per_day
    if dt * steps_per_day != 86400:
        raise ValueError('steps_per_day must divide 86400')
    sec = s0 + np.arange(nsteps, dtype=np.int64) * dt
    day = sec // 86400
    uniq = np.unique(day)
    tab = {int(u): _ymd_add(y0, m0, d0, int(u), calendar) for u in uniq}
    ymd = np.array([tab[int(u)] for u in day], dtype=np.int64).reshape(-1, 3)
    return ymd[:, 0], ymd[:, 1], ymd[:, 2]


def parse_time_units(units):
    m = re.match(r'\s*(\w+)\s+since\s+(\d+)-(\d+)-(\d+)'
                 r'(?:[ T](\d+):(\d+):(\d+(?:\.\d*)?))?', units)
    if not m:
        raise ValueError('cannot parse time units %r' % units)
    unit = m.group(1).lower()
    if unit not in _UNITS:
        raise ValueError('time unit %r not supported' % unit)
    hms = [float(m.group(i) or 0) for i in (5, 6, 7)]
    return (_UNITS[unit], (int(m.group(2)), int(m.group(3)), int(m.group(4))),
            int(hms[0] * 3600 + hms[1] * 60 + hms[2]))


_CAL_NAMES = {'standard': 'standard', 'gregorian': 'gregorian',
              'proleptic_gregorian': 'proleptic_gregorian',
              'noleap': 'noleap', 'no_leap': 'noleap', '365_day': '365_day',
              '360_day': '360_day'}


def _CAL_ENUM(name):
    """str_to_calendar (input_tools.c:257-270): case-insensitive, NOLEAP and
    NO_LEAP are one calendar, STANDARD / GREGORIAN / PROLEPTIC_GREGORIAN and
    365_DAY stay distinct."""
    try:
        return _CAL_NAMES[str(name).lower()]
    except KeyError:
        raise ValueError('unknown calendar %r' % (name,))


def file_info(path, varname, calendar='proleptic_gregorian'):
    """First time stamp and record spacing of a plugin forcing file
    (plugin_get_forcing_file.c:60-146). Returns a dict."""
    from torchvic.io import ncshare
    ds = ncshare.read(path, decode_times=False, mask_and_scale=False)
    tv = ds['time'].values
    n = len(tv)
    t0 = float(tv[0])
    t1 = float(tv[1]) if n > 1 else None
    units = ds['time'].attrs['units']
    shape = ds[varname].shape
    if 'calendar' not in ds['time'].attrs:
        raise ValueError('plugin forcing file %s has no time calendar '
                         'attribute (plugin_get_forcing_file.c:76-77)' % path)
    cal = ds['time'].attrs['calendar']
    if _CAL_ENUM(cal) != _CAL_ENUM(calendar):
        # plugin_get_forcing_file.c:193-197 (calendar != global_param.calendar)
        raise ValueError('Calendar in plugin forcing file (%s) does not match '
                         'the calendar of VIC\'s clock (%s) '
                         '(plugin_get_forcing_file.c:193-197)' % (cal, calendar))
    unit_s, origin, osec = parse_time_units(units)
    secs = osec + t0 * unit_s
    ymd = _ymd_add(origin[0], origin[1], origin[2], int(secs // 86400), cal)
    dsec = int(round(secs % 86400))
    dpy = days_in_year(ymd[0], cal)
    if n == 1:
        spy = 1
    else:
        spy = int(np.rint(dpy * 86400.0 / ((t1 - t0) * unit_s)))
    return dict(nrec=n, start=(ymd[0], ymd[1], ymd[2], dsec), steps_per_year=spy,
                calendar=cal, shape=shape)


def read_year(path, varname, cells=None):
    """All records of a year file as float64 [nrec, ..., ncell]; raw fill
    values are kept (C reads doubles without masking)."""
    from torchvic.io import ncshare
    ds = ncshare.read(path, decode_times=False, mask_and_scale=False)
    a = np.asarray(ds[varname].values, dtype=np.float64)
    a = a.reshape(a.shape[:-2] + (-1,))
    if cells is not None:
        a = a[..., np.asarray(cells, dtype=np.int64)]
    return np.ascontiguousarray(a)


def cadence(freq, start, nsteps, steps_per_day, skip0, calendar):
    """The C state machine (plugin_forcing_utils.c:168-257) run over all
    steps. Returns (year[step], record_in_file[step], forcerun[step]);
    record = forceskip + forceoffset of the step, held when forcerun is
    false (the read of the last forcerun step)."""
    y, m, d = step_dates(start, nsteps, steps_per_day, calendar)
    rec = np.zeros(nsteps, dtype=np.int64)
    run = np.zeros(nsteps, dtype=bool)
    off, skip = 0, skip0
    held = skip + off
    for c in range(nsteps):
        if c == 0:
            run[c] = True
        else:
            newyear = y[c] != y[c - 1]
            if newyear:
                off, skip = 0, 0
            else:
                if freq == FORCE_STEP:
                    off += 1
                elif freq == FORCE_DAY and d[c] != d[c - 1]:
                    off += 1
                elif freq == FORCE_MONTH and m[c] != m[c - 1]:
                    off += 1
            run[c] = (freq == FORCE_STEP
                      or (freq == FORCE_DAY and d[c] != d[c - 1])
                      or (freq == FORCE_MONTH and m[c] != m[c - 1])
                      or (freq == FORCE_YEAR and newyear))
        if run[c]:
            held = skip + off
        rec[c] = held
    return y, rec, run


def forceskip(freq, start, file_start, file_spy, calendar):
    """plugin_get_forcing_file.c:232-249 (start, file_start = (y,m,d,sec))."""
    if freq in (FORCE_STEP, FORCE_DAY):
        dd = _days_between(file_start[:3], start[:3], calendar)
        days = dd + (start[3] - file_start[3]) / 86400.0
        x = days * file_spy / float(days_in_year(file_start[0], calendar))
        return int(np.floor(x + 0.5))      # C round(): half away from zero
    if freq == FORCE_MONTH:
        return start[1] - file_start[1]
    if freq == FORCE_YEAR:
        return 0
    raise ValueError('forcing frequency %r not implemented' % freq)


class PluginForcing(object):
    """One PLUGIN_FORCE_TYPE stream on the device.

        f = PluginForcing(prefix, varname, 'DAY', start=(2003, 7, 1, 0),
                          nsteps=n, steps_per_day=1, device='cuda')
        x = f.get(step)       # [..., ncell] device view of the held record
        f.reads(step)         # forcerun of that step (host bool)

    prefix is the path prefix as written in the global file ("cv." gives
    cv.2003.nc). cells = flat (lat*nlon + lon) indices of the active cells,
    default every grid cell. met_start (required, keyword) = the met forcing's
    first time stamp (y, m, d, sec); it must equal the plugin file's. The file's
    time calendar attribute must equal `calendar` (the run's calendar).
    """

    def __init__(self, prefix, varname, freq, start, nsteps, steps_per_day=1,
                 device='cpu', cells=None, calendar='proleptic_gregorian',
                 *, met_start, prefetch=True):
        self.prefix, self.varname, self.freq = prefix, varname, freq.upper()
        self.cells = cells
        self.device = torch.device(device)
        self.calendar = calendar
        self.nsteps = nsteps
        first = start[0]
        info = file_info(self._path(first), varname, calendar)
        self.info = info
        if met_start is None:
            raise ValueError('met_start is required: VIC-WUR compares the plugin '
                             'forcing time with the met forcing time '
                             '(plugin_get_forcing_file.c:99-116)')
        if tuple(met_start) != tuple(info['start']):
            raise ValueError('plugin forcing file time %s must match the met '
                             'forcing file time %s (plugin_get_forcing_file.c:'
                             '99-116)' % (info['start'], tuple(met_start)))
        self._check_records(info, steps_per_day, first)
        self.skip0 = forceskip(self.freq, start, info['start'],
                               info['steps_per_year'], calendar)
        y, rec, run = cadence(self.freq, start, nsteps, steps_per_day,
                              self.skip0, calendar)
        self.year, self.rec, self.run = y, rec, run
        self._years = [int(v) for v in np.unique(y)]
        self._rec_dev = torch.as_tensor(rec).to(self.device)
        self._buf = {}                    # year -> (device tensor, event)
        self._pending = {}                # year -> Future
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(1) if prefetch else None
        self._side = (torch.cuda.Stream(self.device)
                      if self.device.type == 'cuda' else None)
        self.n_loads = 0                  # file reads (each year once)
        self._ensure(int(y[0]))

    def _path(self, year):
        return '%s%4d.nc' % (self.prefix, year)

    def _check_records(self, info, spd, year):
        n, f = info['steps_per_year'], self.freq
        dpy = days_in_year(year, self.calendar)
        want = {FORCE_STEP: spd * dpy, FORCE_DAY: dpy, FORCE_MONTH: 12,
                FORCE_YEAR: 1}[f]
        if n != want:
            raise ValueError('plugin forcing file timestep must match the '
                             'force model timestep: %d records/year, %s '
                             'needs %d (plugin_get_forcing_file.c:148-191)'
                             % (n, f, want))

    # -- window management (host side, no device synchronisation) ----------
    def _load(self, year):
        a = torch.from_numpy(read_year(self._path(year), self.varname,
                                       self.cells))
        if self.device.type == 'cuda':
            a = a.pin_memory()
            with torch.cuda.stream(self._side):
                d = a.to(self.device, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._side)
            d.record_stream(torch.cuda.current_stream(self.device))
            return d, ev
        return a.to(self.device), None

    def _submit(self, year):
        if year in self._buf or year in self._pending or \
                year not in self._years:
            return
        self.n_loads += 1
        if self._pool is not None:
            self._pending[year] = self._pool.submit(self._load, year)
        else:
            self._buf[year] = self._load(year)

    def _ensure(self, year):
        with self._lock:
            self._submit(year)
            if year in self._pending:
                self._buf[year] = self._pending.pop(year).result()
            d, ev = self._buf[year]
            if ev is not None:
                torch.cuda.current_stream(self.device).wait_event(ev)
                self._buf[year] = (d, None)
            for old in [k for k in self._buf if k < year]:
                del self._buf[old]
            self._submit(year + 1)         # prefetch the next window
            return d

    def get(self, step):
        """Held record at model step `step` (host int): [..., ncell]."""
        y = int(self.year[step])
        buf = self._buf[y][0] if y in self._buf else self._ensure(y)
        return buf.index_select(0, self._rec_dev[step:step + 1]).squeeze(0)

    def reads(self, step):
        """forcerun at `step` (a new record is read), host bool."""
        return bool(self.run[step])

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
