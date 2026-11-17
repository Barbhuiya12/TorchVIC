"""StreamDriver: VIC from a global file with the forcing read a window at a
time, for domains whose forcing does not fit in memory.

`case.load` reads the whole forcing (and, for a daily one, expands all of it
through MTCLIM) onto the device before the first step. On the ISC domain
(112,825 cells, 2 years at 3 h) that is ~37 GB of forcing plus every
per-step output; the VIC connector built that way could not run it. This
driver is the fast production path of benchmarks/run_isc_torchvic.py behind
the connector:

* a sub-daily forcing is read one exchange window at a time
  (ImageForcing.plan / read_steps), ahead of the compute: by one reader
  process per field into shared memory and copied to the device on a side
  stream (torchvic.parallel.feed, `forcing_feed: true`, the default on a
  GPU), or by a background thread (io.forcing.WindowPrefetch). At most
  `prefetch` windows wait on the device;
* a DAILY forcing is expanded by MTCLIM over blocks of cells on the device
  into host memory (the benchmark's sequence, the same 90-day margins) and
  each window is copied to the device when it runs;
* the static set-up is case.load's sequence (options, domain, parameters
  inflated by a process pool, band-mean elevation, soil, mosaic, wind
  height, initial moisture, CONSTANTS); each window is basin.simulate with
  VIC's own CUDA graphs and fused regions, the state carried;
* the history streams are aggregated per window (history.stream_records)
  and kept on the host; the per-step series is not kept.

Nothing of VIC's physics is changed: the window's forcing values are
read_steps' (= ImageForcing.read's), and a run cut into windows is the
single run (tests/test_state_carry.py).

A case uses this driver when `stream: true`, or with `stream: auto` (the
default) when it qualifies and its forcing is larger than AUTO_BYTES. What
the stream path does not build (snow-band expansion, LAKES, CARBON, a
vegetation history, COMPUTE_TREELINE, INIT_STATE, a classic-driver file, a
daily model step, `steps` / `first_step`) runs through case.load.
Cells with more than one snow band are not expanded here: with
`snow_bands: mean` they run as one column at the band-mean elevation, as
benchmarks/run_isc_torchvic.py runs them; without it such a case goes to
case.load, or is refused when its forcing is too large for that.
"""
import copy
import os
import shutil
import tempfile
import time

import numpy as np
import torch

from .spec import CouplingError
from .vic import KwargsDriver

#: the sub-daily fields basin.simulate's atmosphere is built from
SUB = ('tair', 'prcp', 'press', 'shortwave', 'longwave', 'vp', 'wind')

#: committed history records are kept on the device up to this many bytes
#: and then brought to the host in one copy
PEND_BYTES = 256 * 2 ** 20

#: `stream: auto` streams a forcing larger than this (fields x steps x
#: cells x 8 bytes); smaller cases keep case.load
AUTO_BYTES = 2 * 2 ** 30


def _say(log):
    return log if log is not None else (lambda *a: None)


def unsupported(gp, options, daily=False, steps=0, first_step=0):
    """[reasons] this global file needs case.load (empty: it can stream)."""
    from ..io import options as _opt, veghist
    why = []
    if gp.get('DOMAIN') is None or gp.get('SOIL') is not None:
        why.append('a classic-driver global file')
    of = gp.get('OUTPUT_FORCE')
    if of is not None and str(of).upper() == 'TRUE':
        why.append('OUTPUT_FORCE')
    st = gp.get('INIT_STATE')
    if st and str(st).upper() != 'FALSE':
        why.append('INIT_STATE')
    if gp.get_bool('LAKES'):
        why.append('LAKES')
    if options.get('carbon'):
        why.append('CARBON')
    if veghist.wanted(gp):
        why.append('a vegetation history (FROM_VEGHIST)')
    if _opt.treeline(gp)[0]:
        why.append('COMPUTE_TREELINE')
    if daily or (gp.get_int('MODEL_STEPS_PER_DAY') or 0) <= 1:
        why.append('a daily model step')
    if steps or first_step:
        why.append('steps / first_step')
    return why


class StreamDriver(KwargsDriver):
    """KwargsDriver whose windows are read from the forcing files as the
    run reaches them (see the module notes). Built by `open_stream`."""

    def __init__(self, static, dates, steps_per_day, area, frac, n_cells,
                 dom, gp, global_file, elev, forcing=None, a0=0, host=None,
                 feed=True, prefetch=2, core=None, write_procs=4,
                 par=None, log=None):
        super().__init__(static, self._window, dates, steps_per_day,
                         area=area, frac=frac, n_cells=n_cells)
        self.dom, self.gp, self.global_file = dom, gp, global_file
        self.elev, self.fc, self.a0, self.host = elev, forcing, a0, host
        self.feed_on = bool(feed)
        self.prefetch = max(int(prefetch), 1)
        self.core = core
        self.write_procs = int(write_procs)
        self.log = _say(log)
        self.case = None
        #: the parameter file (ImageParams), as a loaded case's `par`
        self.par = par
        self._pre = self._feed = self._it = self._dir = None
        self._batch0 = None
        self._readers = []
        self._spans, self._k = [], 0
        self.wait_s = 0.0
        self.windows_read = 0
        from ..io import history
        self.streams = history.parse_streams(gp, log=lambda *a: None)
        self.parts = {s.prefix: {} for s in self.streams}
        #: committed records still on the device (see _flush_parts)
        self._pend, self._pend_bytes = [], 0
        #: the (stream, variable) records each committed window added
        self._hist_windows = []

    # -- forcing windows ----------------------------------------------------
    def _open(self, t0, length):
        """Read the windows [t0, t0 + length), ... of the run ahead."""
        self._close_stream()
        nt = self.n_steps
        spans = [(s, min(s + length, nt)) for s in range(t0, nt, length)]
        fspans = [(self.a0 + a, self.a0 + b) for a, b in spans]
        dev, dt = self.device, self.dtype
        if self.feed_on:
            import multiprocessing as mp
            from ..parallel import feed
            ctx = mp.get_context('fork')
            shm = '/dev/shm' if os.path.isdir('/dev/shm') else None
            self._dir = tempfile.mkdtemp(prefix='torchvic_feed_', dir=shm)
            fd = feed.Feed(ctx, self.global_file, fspans, SUB,
                           [np.asarray(self.dom.rows)],
                           [np.asarray(self.dom.cols)], self._dir)
            # no forcing handle of this process is open across the fork
            # (open_stream closed its plan; io.ncshare)
            self._readers = [ctx.Process(target=feed.reader, args=(fd, k),
                                         daemon=True)
                             for k in range(len(SUB))]
            for r in self._readers:
                r.start()
            self._feed = fd
            self._pre = feed.SharedWindows(fd, 0, dev, dt,
                                           depth=self.prefetch)
        else:
            from ..io import forcing as forcing_io
            self.fc.plan()          # through io.ncshare's shared handle
            self._pre = forcing_io.WindowPrefetch(self.fc, fspans, SUB, dev,
                                                  dt, depth=self.prefetch)
        self._it = iter(self._pre)
        self._spans, self._k = spans, 0

    def _close_stream(self):
        if self._pre is None:
            return
        if self._feed is not None:
            self._feed.abort.set()
        self.wait_s += self._pre.wait_s
        self._pre.close()
        for r in self._readers:
            r.join(5)
            if r.is_alive():
                r.terminate()
        if self._feed is None and self.fc is not None:
            self.fc.close()
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._pre = self._feed = self._it = self._dir = None
        self._readers = []

    def _forcing(self, t0, t1):
        if self.host is not None:
            return {k: v[t0:t1].to(self.device, non_blocking=True)
                    for k, v in self.host.items()}
        if self._pre is None or self._k >= len(self._spans) or \
                self._spans[self._k] != (t0, t1):
            # the first window, a different window length, or a restart
            self._open(t0, t1 - t0)
            if self._spans[0] != (t0, t1):
                raise CouplingError('vic: window %d..%d is not on the stream '
                                    '(%s)' % (t0, t1, self._spans[0]))
        f = next(self._it)
        self._k += 1
        self.windows_read += 1
        if self._k == len(self._spans):
            self._close_stream()
        return f

    def _window(self, t0, t1):
        from ..io import build as image_run
        f = self._forcing(t0, t1)
        dates = self.dates[t0:t1]
        if self._batch0 is None:
            self._batch0 = image_run.ImageBatch(
                self.dom, dates, dtype=self.dtype, device=self.device,
                elevation=self.elev)
        # the domain's tensors (latitude, area, ...) are the same for every
        # window: copied to the device once, not once a window (a host
        # sync each); only the dates are the window's
        batch = copy.copy(self._batch0)
        batch.dates = dates
        batch.n_days = len(dates)
        batch.t0, batch.t1 = 0, batch.n_days
        atm = image_run.build_atmosphere(f, batch)
        atm.pop('_sources', None)
        return dict(batch=batch, atm=atm)

    def run(self, t0, t1, state, precip_add=None, lai=None, day_hook=None,
            vicwur_core=None, **sim):
        from ..core import version as core_version
        if self.core is not None:
            core_version.set_core(self.core)
        kw = dict(sim)
        if vicwur_core is not None:
            kw['vicwur_core'] = vicwur_core
        with torch.no_grad():
            return super().run(t0, t1, state, precip_add=precip_add,
                               lai=lai, day_hook=day_hook, **kw)

    def close(self):
        self._close_stream()

    # -- history ------------------------------------------------------------
    def aggregate(self, series, t0, t1):
        """A committed window's per-step output into the history streams'
        records (on the host)."""
        from ..io import history
        keys = []
        for s in self.streams:
            rec = history.stream_records(series, s, self.dates[t0:t1],
                                         self.steps_per_day,
                                         log=lambda *a: None)
            if rec.tail is not None:
                raise CouplingError(
                    'vic: the window %s .. %s does not end on an AGGFREQ '
                    'boundary of OUTFILE %s; the streamed history needs '
                    'windows of whole output periods'
                    % (self.dates[t0], self.dates[t1 - 1], s.prefix))
            for k, v in rec.data.items():
                self._pend.append((s.prefix, k, v))
                self._pend_bytes += v.numel() * v.element_size()
                keys.append((s.prefix, k))
        self._hist_windows.append(keys)
        if self._pend_bytes >= PEND_BYTES:
            self._flush_parts()

    def hist_mark(self):
        """The history's length in committed windows (hist_reset)."""
        return len(self._hist_windows)

    def hist_reset(self, mark):
        """The history as it was at hist_mark() == mark: the records of
        the windows aggregated since then dropped (a speculative window
        rolled back, runtime._window)."""
        while len(self._hist_windows) > mark:
            for pre, k in reversed(self._hist_windows.pop()):
                if self._pend and self._pend[-1][:2] == (pre, k):
                    v = self._pend.pop()[2]
                    self._pend_bytes -= v.numel() * v.element_size()
                else:
                    self.parts[pre][k].pop()

    def _flush_parts(self):
        """The records aggregated on the device since the last flush into
        `parts` on the host, in one copy per dtype (a window's .cpu() per
        variable was a host sync each)."""
        if not self._pend:
            return
        by = {}
        for i, (_, _, v) in enumerate(self._pend):
            by.setdefault(v.dtype, []).append(i)
        host = [None] * len(self._pend)
        for dt, ix in by.items():
            vs = [self._pend[i][2] for i in ix]
            if vs[0].device.type == 'cpu':
                got = vs
            else:
                flat = torch.cat([v.reshape(-1) for v in vs]).cpu()
                got = list(flat.split([v.numel() for v in vs]))
                got = [g.reshape(v.shape) for g, v in zip(got, vs)]
            for i, g in zip(ix, got):
                host[i] = g
        for (pre, k, _), h in zip(self._pend, host):
            self.parts[pre].setdefault(k, []).append(h)
        self._pend, self._pend_bytes = [], 0

    def write_history(self, result_dir, log=None):
        """The OUTFILE streams as torchvic.io.history writes them; the
        files of a stream are written by `write_procs` forked processes."""
        from ..io import history
        say = _say(log)
        self._flush_parts()
        paths = []
        dt = 86400 // self.steps_per_day
        for s in self.streams:
            if not self.parts[s.prefix]:
                continue
            recs, tail = history.windows(self.dates, dt, s.agg)
            data = {k: torch.cat(v, 0)
                    for k, v in self.parts[s.prefix].items()}
            rec = history.Record(s, s.agg, recs, tail,
                                 history.as_datetimes(self.dates), dt, data,
                                 {k: 'torchvic' for k in data}, {}, [])
            groups = history.file_groups(rec.records, rec.times, rec.dt,
                                         s.hist)
            if self.write_procs <= 1 or len(groups) <= 1:
                paths += history.write_record(rec, self.dom, result_dir,
                                              gp=self.gp, log=say)
                continue
            import concurrent.futures as cf
            import multiprocessing as mp
            global _WRITE
            _WRITE = (rec, self.dom, result_dir, self.gp)
            with cf.ProcessPoolExecutor(
                    min(self.write_procs, len(groups)),
                    mp_context=mp.get_context('fork')) as ex:
                for got in ex.map(_write_group, range(len(groups))):
                    paths += got
                    for p in got:
                        say('  wrote %s' % p)
            _WRITE = None
        return paths


_WRITE = None


def _write_group(gn):
    from ..io import history
    rec, dom, out, gp = _WRITE
    return history.write_record(rec, dom, out, gp=gp, log=lambda *x: None,
                                only={gn})


def _min_tair(fc, a, b, step):
    """The run's lowest air temperature, read window by window (what
    resolve_wind_h needs without WIND_H)."""
    lo = np.inf
    for s in range(a, b, step):
        blk = fc.read_steps(s, min(s + step, b))
        lo = min(lo, float(np.min(blk['tair'])))
    return lo


def open_stream(global_file, device, dtype, core=None, no_energy=False,
                sample_data=None, stream='auto', daily=False, steps=0,
                first_step=0, window_steps=None, feed=None, prefetch=2,
                write_procs=4, snow_bands='expand', log=None):
    """A StreamDriver for `global_file`, or None where case.load is the
    path (stream: false; a case the stream does not build; with
    stream: auto, a forcing below AUTO_BYTES)."""
    from .. import case as C
    from ..core import version as core_version
    from ..io import build as image_run
    from ..io.domain import ImageDomain
    from ..io.forcing import ImageForcing
    from ..io.global_param import GlobalParam
    from ..io.params import ImageParams
    from ..parallel import preread
    say = _say(log)
    if stream is False or str(stream).lower() in ('false', 'no', 'off'):
        return None
    force = stream is True or str(stream).lower() in ('true', 'yes', 'on')
    if core is not None:
        core_version.set_core(core)
    gp = GlobalParam(global_file, sample_data=C.sample_data_dir()
                     if sample_data is None else sample_data)
    options = C.image_options(gp, log=lambda *a: None, no_energy=no_energy)
    why = unsupported(gp, options, daily, steps, first_step)
    if why:
        if force:
            say('vic: stream: true, but %s needs case.load (%s)'
                % (global_file, ', '.join(why)))
        return None
    dom = ImageDomain(C._fix_path(gp.domain_file, say), names=gp.domain_vars)
    fc = ImageForcing(gp.forcing_prefix, dom, force_type=gp.force_type)
    times = fc.plan()
    found = {can for _, _, take in fc._plan for _, can, _ in take}
    is_daily = 'tair' not in found and {'tmax', 'tmin'} <= found
    spd = gp.get_int('MODEL_STEPS_PER_DAY')
    nc = int(dom.n_active)
    if not is_daily and not set(SUB) <= found:
        fc.close()
        if force:
            say('vic: stream: true, but the forcing lacks %s'
                % sorted(set(SUB) - found))
        return None
    refused = []
    spd_f = (None if is_daily else max(int(round(
        86400.0 / (ImageForcing.step_seconds(times) or 86400.0))), 1))
    snow_spd = gp.get_int('SNOW_STEPS_PER_DAY')
    if spd_f is not None:
        if spd != spd_f:
            refused.append('MODEL_STEPS_PER_DAY %d with a forcing of %d '
                           'records a day' % (spd, spd_f))
        if snow_spd and snow_spd != spd_f:
            refused.append('SNOW_STEPS_PER_DAY %d with a forcing of %d '
                           'records a day: VIC stops on this' % (snow_spd,
                                                                 spd_f))
    cal = gp.get('CALENDAR')
    if cal is not None:
        fcal = C._forcing_calendar(times)
        if C._CAL_FAMILY.get(str(cal).upper(), cal) != \
                C._CAL_FAMILY.get(fcal, fcal):
            refused.append('CALENDAR %s with a forcing on the %s calendar'
                           % (cal, fcal.lower()))
    try:
        if is_daily:
            w0, w1 = C.window(gp, times, 1)
        else:
            w0, w1 = C.window(gp, times, spd)
    except ValueError as e:
        refused.append(str(e))
        w0 = w1 = 0
    if refused:
        fc.close()
        raise ValueError('%s selects what TorchVIC does not run as VIC does '
                         '(docs/OPTIONS.md):\n  %s'
                         % (gp.path, '\n  '.join(refused)))
    n_steps = (w1 - w0) * (spd if is_daily else 1)
    size = len(SUB) * n_steps * nc * 8
    small = not force and size < AUTO_BYTES
    if small and snow_bands != 'mean':
        fc.close()
        return None
    say('vic: streaming the forcing (%.1f GB over the run): %s'
        % (size / 1e9, 'daily, MTCLIM into host memory' if is_daily else
           'read a window at a time'))
    t = time.time()
    with preread.preloaded(gp.params_file, dom):
        par = ImageParams(C._fix_path(gp.params_file, say), dom)
    say(str(dom)); say(str(par))
    par.apply_density_options(bool(gp.get_bool('ORGANIC_FRACT')),
                              bool(gp.get_bool('BULK_DENSITY_COMB')))
    refused = C.check_inputs(gp, par)
    if refused:
        fc.close()
        raise ValueError('%s selects what TorchVIC does not run as VIC does '
                         '(docs/OPTIONS.md):\n  %s'
                         % (gp.path, '\n  '.join(refused)))
    if gp.get_bool('SNOW_BAND') and all(
            k in par.bands for k in ('AreaFract', 'elevation', 'Pfactor')):
        live = (np.asarray(par.bands['AreaFract']) > 0).sum(0)
        multi = bool((live > 1).any())
        if multi:
            if snow_bands != 'mean':
                fc.close()
                if not force and size >= AUTO_BYTES:
                    raise CouplingError(
                        'vic: %s has %d cells with more than one snow band '
                        'and a %.0f GB forcing: case.load (bands expanded '
                        'to columns, the whole forcing in memory) cannot '
                        'hold it. snow_bands: mean streams it as '
                        'benchmarks/run_isc_torchvic.py runs it -- one '
                        'column per cell at the band-mean elevation, no '
                        'band expansion (not VIC\'s SNOW_BAND TRUE '
                        'columns)' % (global_file, int((live > 1).sum()),
                                      size / 1e9))
                if force:
                    say('vic: stream: true, but the snow bands need '
                        'case.load (or snow_bands: mean)')
                return None
            say('vic: snow_bands: mean -- %d cells with more than one band '
                'run as one column at the band-mean elevation (the '
                'production benchmark\'s set-up), not expanded to bands'
                % int((live > 1).sum()))
    else:
        multi = False
    if small and not multi:
        # snow_bands: mean asks for nothing case.load would not do here
        fc.close()
        return None
    dev = torch.device(device) if device is not None else torch.device('cpu')
    if dev.type == 'cuda':
        if dev.index is not None:
            torch.cuda.set_device(dev)
        torch.zeros(1, device=dev)
    import pandas as pd
    host = None
    if is_daily:
        from ..forcing import daily as daily_forcing
        fields, times = fc.read(dtype=dtype, device='cpu', log=say)
        day = times.astype('datetime64[D]')
        first = np.datetime64(gp.start_date)
        last = np.datetime64(gp.end_date)
        a0_ = int(np.searchsorted(day, first))
        b0_ = int(np.searchsorted(day, last)) + 1
        m = C.MTCLIM_MARGIN_DAYS
        lo, hi = max(a0_ - m, 0), min(b0_ + m, len(times))
        lead, n_win = a0_ - lo, b0_ - a0_
        chunk = daily_forcing.auto_cell_chunk(hi - lo, dev, dtype)
        say('daily forcing: MTCLIM on %s over %s..%s (%d days, %d-day '
            'margins), %d cells in blocks of %d'
            % (dev, day[lo], day[hi - 1], hi - lo, m, nc, chunk))
        win = {k: v[lo:hi] for k, v in fields.items()}
        stamps = times[lo:hi]
        elev_soil = np.asarray(par.soil['elev'])
        host = {k: torch.empty((n_win * spd, nc), dtype=dtype,
                               pin_memory=dev.type == 'cuda') for k in SUB}
        s0, s1 = lead * spd, (lead + n_win) * spd
        for c0 in range(0, nc, chunk):
            c1 = min(c0 + chunk, nc)
            f = {k: v[:, c0:c1].to(dev) for k, v in win.items()}
            sub, st = daily_forcing.expand_forcing(
                f, stamps, spd, lat=np.asarray(dom.lat[c0:c1]),
                lng=np.asarray(dom.lon[c0:c1]), elev=elev_soil[c0:c1],
                log=(say if c0 == 0 else (lambda *x: None)))
            for k in SUB:
                host[k][:, c0:c1].copy_(sub[k][s0:s1])
            del f, sub
        stamps_sub = st[s0:s1]
        del win, fields
        fc.close()
        fc, a0 = None, 0
    else:
        stamps_sub = times[w0:w1]
        a0 = w0
    dates = [pd.Timestamp(x).to_pydatetime() for x in stamps_sub]
    say('%d steps at %d/day (%s .. %s), %d cells'
        % (len(dates), spd, dates[0], dates[-1], nc))
    runoff_steps = gp.get_int('RUNOFF_STEPS_PER_DAY')
    elev = image_run.band_mean_elevation(par,
                                         snow_band=gp.get_bool('SNOW_BAND'))
    soil = image_run.build_soil(par, dtype, dev,
                                runoff_steps_per_day=runoff_steps,
                                model_steps_per_day=spd,
                                baseflow=(gp.get('BASEFLOW') or 'ARNO'))
    fcan_src = (gp.get('FCAN_SRC') or 'FROM_DEFAULT').upper()
    mosaic = image_run.build_mosaic(
        par, dtype, dev, log=say, fcan_src=fcan_src,
        max_snow_albedo=bool(gp.get_bool('MAX_SNOW_ALBEDO')))
    min_t = None
    if host is not None:
        min_t = float(host['tair'].min())
    elif gp.get('WIND_H') is None:
        min_t = _min_tair(fc, w0, w1, 31 * spd)
    wind_h, wind_src, _ = C.resolve_wind_h(
        gp, C.params_reference_heights(par), min_t, mosaic=True, log=say)
    say('wind height %g m (%s)' % (wind_h, wind_src))
    init_moist = torch.as_tensor(par.soil['init_moist'], dtype=dtype,
                                 device=dev).T.contiguous()
    full_energy = bool(gp.get_bool('FULL_ENERGY') and not no_energy)
    p = C.default_params(dtype, dev, **(gp.constants()
                                        if hasattr(gp, 'constants') else {}))
    static = dict(p=p, soil_params=soil, mosaic=mosaic, wind_h=wind_h,
                  init_moist=init_moist, steps_per_day=spd,
                  full_energy=full_energy, log_every=0, **options)
    if 'out_vars' not in static:
        # the rows the OUTFILE streams ask for, as case.load's kwargs
        from ..case import requested_outvars
        static['out_vars'] = requested_outvars(gp)
    area = np.asarray(dom.area, dtype=np.float64)
    frac = np.asarray(dom.frac, dtype=np.float64)
    ok = area.shape == (nc,) and frac.shape == (nc,)
    if feed is None:
        feed = dev.type == 'cuda'
    if host is not None:
        feed = False
    if fc is not None:
        # nothing of the forcing stays open between runs: the windows
        # reopen it (one shared handle, io.ncshare) or the feed's reader
        # processes open their own after the fork
        fc.close()
    say('vic: set-up %.1f s; forcing %s, %d window(s) ahead'
        % (time.time() - t, 'host (MTCLIM)' if host is not None else
           ('reader processes (feed)' if feed else 'prefetch thread'),
           prefetch))
    return StreamDriver(static, dates, spd, area if ok else None,
                        frac if ok else None, nc, dom, gp, global_file, elev,
                        forcing=fc, a0=a0, host=host, feed=feed,
                        prefetch=prefetch, core=core,
                        write_procs=write_procs, par=par, log=log)
