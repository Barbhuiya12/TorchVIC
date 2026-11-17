#!/usr/bin/env python3
"""TorchVIC over the full ISC domain, end to end, in time chunks.

    run_isc_torchvic.py GLOBAL --out DIR [--device cuda|cpu] [--core 5.1.0]
        [--chunk-days 31] [--mtclim-cell-chunk 0] [--timing FILE]

`torchvic run` holds the whole forcing and every per-step output of the run
on the device at once. For 105,113 cells and 5,848 three-hourly steps that
is about 50 GB of forcing and ~200 GB of per-step outputs, more than any
of the GPUs here, so this driver runs the same model in pieces:

* a sub-daily forcing is read a window at a time (ImageForcing.plan and
  read_steps, the values ImageForcing.read gives) on a background thread
  that copies it to the device on a side stream while the previous window
  computes (WindowPrefetch; --prefetch 0 reads it all first, as before);
  a daily forcing is read once into host memory;
* a DAILY forcing is disaggregated by the same MTCLIM chain case.load uses
  (expand_forcing, the same 90-day margins), over blocks of cells on the
  device -- MTCLIM's 30-second solar geometry is 8.4 MB a cell, so a block
  is a few thousand cells;
* the time loop runs over windows of --chunk-days days with all cells at
  once; each window's state is carried into the next (basin.simulate
  return_state / initial_state, which tests/test_state_carry.py shows is the
  single run bit for bit on the CPU), and each window's output is
  aggregated to the global file's OUTFILE streams before the next starts;
* the history is written once, as torchvic.io.history writes it.

The model set-up -- options, domain, parameters, band-mean elevation, soil,
mosaic, wind height, initial moisture, physics switches, parameters and
CONSTANTS -- is case.load's sequence, step for step. The run's window is the
global file's STARTYEAR..ENDYEAR (torchvic run takes the whole forcing).

--gpus N runs the same thing over N processes, one per GPU (cuda:0..N-1 of
the job's allocation): each process takes every N-th active cell
(torchvic.parallel.shard), runs the pipeline above
on it and saves its aggregated streams; the parent concatenates the pieces
along the cell axis and writes the files one process writes. Cells are
independent, so the values are the single run's, bit for bit.

The forcing is read once, whatever N: one reader process per field reads
each window, cuts it into the shards' cells and leaves it in tmpfs slots
(torchvic.parallel.feed; --feed 0 gives every process its own reads, as
before), so a shard never waits on I/O. The parameter file is inflated by a
pool of processes (torchvic.parallel.preread), the shards are forked from
the planned parent, and a stream's history files are written by a process
pool (--write-procs). Every process records a timeline (timing JSON,
"events"). On the ISC domain the physics of a step is launch-bound: N GPUs
take about 23 s + 0.078 s per 1000 cells per process, so they pay off only
for domains of several hundred thousand cells (benchmarks/make_isc_2x.py).

Timing (wall clock, device synchronised at each boundary) is written as
JSON: read, mtclim, setup, physics (incl. host-to-device copies and the
per-window aggregation), write, total, and the peak device memory;
read_wait_s is the time the time loop waited for a prefetched window.
"""
import time as _time
_T_START = _time.time()          # before the heavy imports (timeline)
import argparse
import datetime as _dt
import json
import os
import sys
import time

import numpy as np
import torch

from torchvic import case as C
from torchvic.core import version as core_version
from torchvic.forcing import daily as daily_forcing
from torchvic.io import build as image_run
from torchvic.io import forcing as forcing_io
from torchvic.io import history
from torchvic.io.domain import ImageDomain
from torchvic.io.forcing import ImageForcing
from torchvic.io.global_param import GlobalParam
from torchvic.io.params import ImageParams
from torchvic.model import basin
from torchvic.parallel import shard, preread, feed

SUB = ('tair', 'prcp', 'press', 'shortwave', 'longwave', 'vp', 'wind')


#: set in a shard process spawned by run_sharded: the forcing feed
FEED = None
#: timeline of this process: [(event, seconds since the process started)]
EVENTS = []


def ev(name):
    EVENTS.append((name, round(time.time() - _T_START, 3)))


def _daily_only(fc, gp):
    """True if the forcing declares TMAX/TMIN and no AIR_TEMP (MTCLIM)."""
    ft = set(gp.force_type or {})
    return 'AIR_TEMP' not in ft and {'TMAX', 'TMIN'} <= ft


def sync(dev):
    if str(dev).startswith('cuda'):
        torch.cuda.synchronize()


def write_streams(streams, dates, spd, parts, dom, gp, out, say, procs=0):
    """The OUTFILE streams of `parts` ({prefix: {var: [rec, cell, ...]
    tensors or lists of window tensors}}), as torchvic.io.history writes
    them. The files of a stream are independent (one per HISTFREQ period):
    with `procs` > 1 they are written by forked processes at once."""
    paths = []
    dt = 86400 // spd
    for s in streams:
        recs, tail = history.windows(dates, dt, s.agg)
        data = {k: (v if torch.is_tensor(v) else torch.cat(v, 0))
                for k, v in parts[s.prefix].items()}
        rec = history.Record(s, s.agg, recs, tail,
                             history.as_datetimes(dates), dt, data,
                             {k: 'run_isc_torchvic' for k in data}, {}, [])
        groups = history.file_groups(rec.records, rec.times, rec.dt, s.hist)
        if procs <= 1 or len(groups) <= 1:
            paths += history.write_record(rec, dom, out, gp=gp, log=say)
            continue
        import concurrent.futures as cf
        import multiprocessing as mp
        global _WRITE
        _WRITE = (rec, dom, out, gp)
        say_ = lambda *x: None
        with cf.ProcessPoolExecutor(min(procs, len(groups)),
                                    mp_context=mp.get_context('fork')) as ex:
            for got in ex.map(_write_group, range(len(groups))):
                paths += got
                for pth in got:
                    say('  wrote %s' % pth)
    return paths


_WRITE = None


def _write_group(gn):
    rec, dom, out, gp = _WRITE
    return history.write_record(rec, dom, out, gp=gp,
                                log=lambda *x: None, only={gn})


def _plan_windows(a, gp, dom):
    """The forcing windows main() runs, [(first step, last step + 1)] of the
    forcing's time axis, and the fields they carry -- or None where the
    forcing is not read window by window (daily forcing, no WIND_H,
    --prefetch 0)."""
    spd = gp.get_int('MODEL_STEPS_PER_DAY')
    fc = ImageForcing(gp.forcing_prefix, dom, force_type=gp.force_type)
    if not (a.prefetch and gp.get('WIND_H') is not None
            and not _daily_only(fc, gp)):
        return None
    try:
        times = fc.plan()
        found = {can for _, _, take in fc._plan for _, can, _ in take}
        if not set(SUB) <= found:
            return None
    finally:
        fc.close()
    first = np.datetime64(gp.start_date)
    last = np.datetime64(gp.end_date)
    a0 = int(np.searchsorted(times, first))
    b0 = int(np.searchsorted(times, last + np.timedelta64(1, 'D')))
    if a.days:
        b0 = min(b0, a0 + a.days * spd)
    nt = b0 - a0
    step = a.chunk_days * spd
    return [(a0 + w0, a0 + min(w0 + step, nt)) for w0 in range(0, nt, step)]


def _shard_entry(argv, feed_, log):
    """Process target of one shard (spawned by run_sharded)."""
    global FEED, _T_START
    FEED = feed_
    _T_START = time.time()
    EVENTS.clear()
    sys.stdout = sys.stderr = open(log, 'w', buffering=1)
    try:
        rc = main(argv)
    except BaseException:                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        feed_.abort.set()
        raise
    sys.exit(rc)


def run_sharded(a, argv):
    """--gpus N (or the forcing feed on one GPU): N shard processes of this
    script, one shard and one GPU each, fed by one reader process per
    forcing field, then the merge and the write."""
    import pickle
    import shutil
    import tempfile
    import torch.multiprocessing as mp
    say = (lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True))
    T0 = time.time()
    n = a.gpus
    os.makedirs(a.out, exist_ok=True)
    shm = '/dev/shm' if os.path.isdir('/dev/shm') else a.out
    parts = tempfile.mkdtemp(prefix='torchvic_parts_', dir=shm)
    drop = {'--gpus', '--device', '--timing'}
    base, skip = [], False
    for x in argv:
        if skip:
            skip = False
        elif x in drop:
            skip = True
        elif x.split('=')[0] in drop:
            pass
        else:
            base.append(x)
    # fork, not spawn: the parent has imported torch and the model once (no
    # CUDA, no open file below), and each child starts with them in hand
    ctx = mp.get_context('fork')
    gp = GlobalParam(a.global_file)
    dom = ImageDomain(gp.domain_file, names=gp.domain_vars)
    spans = _plan_windows(a, gp, dom) if a.feed else None
    dom.ds.close()       # no HDF5 handle open across the fork; reopens on use
    ev('parent_planned')
    feed_ = None
    readers = []
    if spans:
        idx = [shard.shard_index(int(dom.n_active), n, i) for i in range(n)]
        feed_ = feed.Feed(ctx, a.global_file, spans, SUB,
                          [np.asarray(dom.rows)[ix] for ix in idx],
                          [np.asarray(dom.cols)[ix] for ix in idx], parts)
        readers = [ctx.Process(target=feed.reader, args=(feed_, k),
                               daemon=True) for k in range(len(SUB))]
        for r in readers:
            r.start()
    procs = []
    for i in range(n):
        dev = 'cpu' if a.device == 'cpu' else 'cuda:%d' % i
        cmd = base + ['--device', dev, '--shard', '%d/%d' % (i, n),
                      '--parts', parts, '--timing',
                      os.path.join(parts, 'timing_%d.json' % i)]
        if i:
            cmd.append('--quiet')
        p = ctx.Process(target=_shard_entry,
                        args=(cmd, feed_,
                              os.path.join(parts, 'shard_%d.log' % i)))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    rc = [p.exitcode for p in procs]
    if feed_ is not None:
        feed_.abort.set()
    for r in readers:
        r.join(5)
        if r.is_alive():
            r.terminate()
    if any(rc):
        for i in range(n):
            print('--- shard %d (exit %s) ---' % (i, rc[i]))
            print(open(os.path.join(parts, 'shard_%d.log' % i)).read()[-3000:])
        shutil.rmtree(parts, ignore_errors=True)
        return 1
    ev('shards_done')
    say(open(os.path.join(parts, 'shard_0.log')).read())
    t = time.time()
    tms = [json.load(open(os.path.join(parts, 'timing_%d.json' % i)))
           for i in range(n)]
    paths = []
    if n > 1:
        streams = history.parse_streams(gp, log=say)
        with open(os.path.join(parts, 'meta.pkl'), 'rb') as fh:
            meta = pickle.load(fh)
        shards = [torch.load(os.path.join(parts, 'data_%d.pt' % i))
                  for i in range(n)]
        index = [np.load(os.path.join(parts, 'index_%d.npy' % i))
                 for i in range(n)]
        ev('merge_loaded')
        merged = {s.prefix: shard.merge_data([d[s.prefix] for d in shards],
                                             index) for s in streams}
        del shards
        ev('merged')
        paths = write_streams(streams, meta['dates'], meta['spd'], merged,
                              dom, gp, a.out, say, procs=a.write_procs)
        ev('written')
    tm = dict(gpus=n, total_s=time.time() - T0, merge_write_s=time.time() - t,
              shards=tms, n_cells=sum(x['n_cells'] for x in tms),
              n_steps=tms[0]['n_steps'], fed=feed_ is not None,
              slowest_shard_s=max(x['total_s'] for x in tms),
              slowest_physics_s=max(x['physics_s'] for x in tms),
              gpu=tms[0].get('gpu'), events=EVENTS, start_epoch=_T_START)
    shutil.rmtree(parts, ignore_errors=True)
    print(json.dumps({k: v for k, v in tm.items() if k != 'shards'}, indent=1),
          flush=True)
    for pth in paths:
        print('wrote %s' % pth)
    if a.timing:
        with open(a.timing, 'w') as fh:
            json.dump(tm, fh, indent=1)
    return 0


def _Irrigator(area, dev):
    from torchvic.coupling import Balance, Connector, Proposal, VarSpec
    from torchvic.coupling.vic import GRID

    class Irrigator(Connector):
        name = 'irr'
        interval_s = 86400.0
        backends = ('cpu', 'cuda')
        inputs = (VarSpec('soil_moisture', 'mm', ('cell', 'layer'),
                          kind='state', grid=GRID),)
        outputs = (VarSpec('water', 'mm', ('cell',), kind='flux',
                           grid=GRID),)

        def __init__(self):
            super().__init__()
            self.area = area.to(dev)

        def compute(self, inputs, ctx, dt):
            sm = inputs.get('soil_moisture')
            w = (torch.zeros_like(self.area) if sm is None else
                 (20.0 - sm[:, 0]).clamp(0.0, 3.0))
            v = (w * self.area).sum() / 1000.0
            return Proposal(self.name, ctx.t, ctx.t + dt,
                            exports=dict(water=w),
                            balance=Balance(I_ext=v, Q_out=v, scale=2 * v,
                                            n_terms=2 * w.shape[0]))

        def commit(self, p, ctx):
            pass
    return Irrigator()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('global_file')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--core', default=None,
                    help='VIC core (default: the library\'s default core, 5.1.0)')
    ap.add_argument('--dtype', default='float64',
                    choices=['float64', 'float32'])
    ap.add_argument('--chunk-days', type=int, default=31)
    ap.add_argument('--mtclim-cell-chunk', type=int, default=0,
                    help='cells per MTCLIM block (0: the largest power of '
                    'two up to 4096 that fits the free device memory)')
    ap.add_argument('--days', type=int, default=0,
                    help='run only this many days from the start (0: all)')
    ap.add_argument('--timing', default=None)
    ap.add_argument('--prefetch', type=int, default=2,
                    help='read a sub-daily forcing this many windows ahead '
                    'on a background thread (0: read it all first)')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--runtime', action='store_true',
                    help='run the windows through the coupled runtime '
                    '(torchvic.coupling, VIC alone) instead of calling '
                    'basin.simulate directly')
    ap.add_argument('--irrigate', default='none',
                    choices=['none', 'day', 'window'],
                    help='with --runtime: a daily irrigation stand-in '
                    'coupled to VIC every day, through VIC windows of '
                    '--chunk-days (day: exchanged at window boundaries, use '
                    '--chunk-days 1; window: run between the days of each '
                    'window, day_coupled)')
    ap.add_argument('--routing', default=None, metavar='FDIR:VAR,DOMAIN,GAUGES',
                    help='with --runtime: couple the river router '
                    '(connectors/routing.py) to VIC: flow directions '
                    '(file:variable), the domain file (lat, lon, mask, '
                    'area, frac) and the gauge csv (lat, lng); routing '
                    'step one day; the timing of its compute goes to '
                    'the timing JSON')
    ap.add_argument('--dump-state', default=None,
                    help='torch.save the end state here (for comparisons)')
    ap.add_argument('--init-state', default=None,
                    help='start from a state written by --dump-state (one '
                    'process; the warm-up of the benchmark)')
    ap.add_argument('--gpus', type=int, default=1,
                    help='split the cells over this many processes, one per '
                    'GPU (cuda:0..N-1); N CPU processes with --device cpu')
    ap.add_argument('--feed', type=int, default=1,
                    help='1: the forcing is read once by reader processes '
                    'into shared memory and each shard takes its cells '
                    '(also for one GPU); 0: every process reads its own')
    ap.add_argument('--write-procs', type=int, default=4,
                    help='processes writing the independent history files '
                    'of a stream at once (1: one after the other)')
    ap.add_argument('--async-write', type=int, default=1,
                    help='1: the history files are written by a forked '
                    'process while the windows compute (records appended '
                    'window by window); 0: written after the last window')
    ap.add_argument('--shard', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--parts', default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.shard is None and (a.gpus > 1 or a.feed):
        return run_sharded(a, argv if argv is not None else sys.argv[1:])
    shard_i, shard_n = ((None, None) if a.shard is None
                        else map(int, a.shard.split('/')))
    feed_i = shard_i
    if shard_n == 1:
        shard_i = shard_n = None     # one shard is the whole run
    ev('args')
    say = (lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True))
    dev = torch.device(a.device)
    ev('imports')
    dtype = getattr(torch, a.dtype)
    tm = {}
    T0 = time.time()
    torch.set_grad_enabled(False)

    with torch.inference_mode():
        # ---- read: global file, domain, parameters, forcing -------------
        t = time.time()
        a.core = a.core or core_version.CORE
        core_version.set_core(a.core)
        gp = GlobalParam(a.global_file)
        spd = gp.get_int('MODEL_STEPS_PER_DAY')
        runoff_steps = gp.get_int('RUNOFF_STEPS_PER_DAY')
        options = C.image_options(gp, log=say)
        if shard_i is None:
            dom = ImageDomain(gp.domain_file, names=gp.domain_vars)
        else:
            with shard.sharded(shard_i, shard_n) as info:
                dom = ImageDomain(gp.domain_file, names=gp.domain_vars)
        ev('domain')
        with preread.preloaded(gp.params_file, dom):
            par = ImageParams(gp.params_file, dom)
        say(str(dom)); say(str(par))
        ev('params')
        if dev.type == 'cuda':
            if dev.index is not None:
                torch.cuda.set_device(dev)  # Triton launches on the current device
            torch.zeros(1, device=dev)
            ev('cuda')
        fc = ImageForcing(gp.forcing_prefix, dom, force_type=gp.force_type)
        # A sub-daily forcing is read a window at a time on a background
        # thread while the previous window computes (WindowPrefetch); only
        # the time axis is read here. MTCLIM needs the daily forcing whole,
        # and without WIND_H resolve_wind_h needs the run's lowest tair.
        stream = (a.prefetch and gp.get('WIND_H') is not None
                  and not _daily_only(fc, gp))
        if stream:
            times = fc.plan(log=say)
            fields = None
            ev('plan')
        else:
            fields, times = fc.read(dtype=dtype, device='cpu', log=say)
        tm['read_s'] = time.time() - t
        tm['prefetch'] = bool(stream)

        first = np.datetime64(gp.start_date)
        last = np.datetime64(gp.end_date)
        is_daily = (not stream and 'tair' not in fields
                    and 'tmax' in fields and 'tmin' in fields)
        nc = dom.n_active

        # ---- MTCLIM (daily forcing only) --------------------------------
        t = time.time()
        if is_daily:
            day = times.astype('datetime64[D]')
            a0_ = int(np.searchsorted(day, first))
            b0_ = int(np.searchsorted(day, last)) + 1
            m = C.MTCLIM_MARGIN_DAYS
            lo, hi = max(a0_ - m, 0), min(b0_ + m, len(times))
            lead, n_win = a0_ - lo, b0_ - a0_
            if a.days:
                n_win = min(n_win, a.days)
            if a.mtclim_cell_chunk <= 0:
                a.mtclim_cell_chunk = daily_forcing.auto_cell_chunk(
                    hi - lo, dev, dtype)
            say('daily forcing: MTCLIM on %s over %s..%s (%d days, %d-day '
                'margins), %d cells in blocks of %d'
                % (dev, day[lo], day[hi - 1], hi - lo, m, nc,
                   a.mtclim_cell_chunk))
            win = {k: v[lo:hi] for k, v in fields.items()}
            stamps = times[lo:hi]
            elev_soil = np.asarray(par.soil['elev'])
            host = {k: torch.empty((n_win * spd, nc), dtype=dtype,
                                   pin_memory=dev.type == 'cuda')
                    for k in SUB}
            s0, s1 = lead * spd, (lead + n_win) * spd
            for c0 in range(0, nc, a.mtclim_cell_chunk):
                c1 = min(c0 + a.mtclim_cell_chunk, nc)
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
            sync(dev)
        else:
            a0_ = int(np.searchsorted(times, first))
            b0_ = int(np.searchsorted(times, last + np.timedelta64(1, 'D')))
            if a.days:
                b0_ = min(b0_, a0_ + a.days * spd)
            if stream:
                host = None
            else:
                host = {k: fields[k][a0_:b0_].contiguous() for k in SUB}
                if dev.type == 'cuda':
                    host = {k: v.pin_memory() for k, v in host.items()}
            stamps_sub = times[a0_:b0_]
            del fields
        tm['mtclim_s'] = time.time() - t if is_daily else 0.0
        if dev.type == 'cuda':
            tm['mtclim_peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        import pandas as pd
        dates = [pd.Timestamp(x).to_pydatetime() for x in stamps_sub]
        nt = len(dates)
        say('%d steps at %d/day (%s .. %s), %d cells'
            % (nt, spd, dates[0], dates[-1], nc))

        # ---- set-up: case.load's static sequence ------------------------
        t = time.time()
        elev = image_run.band_mean_elevation(
            par, snow_band=gp.get_bool('SNOW_BAND'))
        soil = image_run.build_soil(par, dtype, dev,
                                    runoff_steps_per_day=runoff_steps,
                                    model_steps_per_day=spd)
        fcan_src = (gp.get('FCAN_SRC') or 'FROM_DEFAULT').upper()
        mosaic = image_run.build_mosaic(par, dtype, dev, log=say,
                                        fcan_src=fcan_src)
        wind_h, wind_src, _ = C.resolve_wind_h(
            gp, C.params_reference_heights(par),
            None if host is None else float(host['tair'].min()),
            mosaic=True, log=say)
        init_moist = torch.as_tensor(par.soil['init_moist'], dtype=dtype,
                                     device=dev).T.contiguous()
        full_energy = bool(gp.get_bool('FULL_ENERGY'))
        p = C.default_params(dtype, dev, **(gp.constants()
                                            if hasattr(gp, 'constants')
                                            else {}))
        streams = history.parse_streams(gp, log=say)
        # the VIC variables the streams ask for are the only extra rows
        # basin.simulate forms (out_vars)
        out_names = frozenset(v.name for s_ in streams for v in s_.vars) or None
        # SNOW_BAND TRUE: the same expansion case.load makes (one pseudo-
        # cell per live (cell, band)); soil, mosaic and initial moisture
        # are expanded once, the forcing window by window, and every
        # window's output is contracted back on cells by Case.contract_bands.
        # SNOW_BAND FALSE, or one band a cell: nothing here runs.
        snow_ex, band_kw, shim = None, {}, None
        if gp.get_bool('SNOW_BAND') and all(
                k in par.bands for k in ('AreaFract', 'elevation',
                                         'Pfactor')):
            snow_ex = C.snow_expansion(par)
        if snow_ex is not None:
            from torchvic.io import options as _opt
            if a.runtime:
                raise SystemExit('SNOW_BAND TRUE with more than one band is '
                                 'not run through --runtime')
            if _opt.treeline(gp)[0]:
                raise SystemExit('COMPUTE_TREELINE with snow bands is not '
                                 'wired in this driver (use torchvic run)')
            say(C.snowbands.summary(snow_ex))
            soil, mosaic, init_moist = C.expand_bands_static(
                soil, mosaic, init_moist, snow_ex, dev)
            band_kw['band_tfactor'] = torch.as_tensor(
                np.asarray(snow_ex['tfactor']), dtype=dtype, device=dev)
            shim = object.__new__(C.Case)       # contract_bands' state only
            shim.snowbands, shim.treeline, shim.mosaic = snow_ex, False, mosaic
        sync(dev)
        tm['setup_s'] = time.time() - t

        # ---- physics: windows of chunk-days, state carried --------------
        t = time.time()
        aw = None
        if a.async_write and not a.runtime and not (
                shard_i is not None and shard_n > 1):
            # forks: before the window prefetch makes its thread
            from torchvic.io.async_history import AsyncStreamWriter
            os.makedirs(a.out, exist_ok=True)
            aw = AsyncStreamWriter(streams, dates, spd, dom, gp, a.out,
                                   source='run_isc_torchvic')
        step = a.chunk_days * spd
        state = None
        if a.init_state:
            from torchvic.model import state as _ms0
            state = _ms0.map_state(torch.load(a.init_state, weights_only=False),
                                   lambda x: x.to(dev))
        parts = {s.prefix: {} for s in streams}
        t_sim = 0.0
        spans = [(w0, min(w0 + step, nt)) for w0 in range(0, nt, step)]
        if stream and FEED is not None:
            want = [(a0_ + w0, a0_ + w1) for w0, w1 in spans]
            if list(FEED.spans) != want:
                raise RuntimeError('the feed was cut for other windows')
            pre = feed.SharedWindows(FEED, feed_i, dev, dtype,
                                     depth=a.prefetch)
            chunks = iter(pre)
        elif stream:
            pre = forcing_io.WindowPrefetch(
                fc, [(a0_ + w0, a0_ + w1) for w0, w1 in spans], SUB, dev,
                dtype, depth=a.prefetch)
            chunks = iter(pre)
        else:
            chunks = ({k: v[w0:w1].to(dev, non_blocking=True)
                       for k, v in host.items()} for w0, w1 in spans)
        t_pre = t_agg = 0.0
        if a.runtime:
            # the same windows through the coupled runtime (VIC alone):
            # VicConnector with this pipeline's forcing windows, the
            # ledger on, the aggregation as an output sink
            from torchvic.coupling import TorchVIC
            from torchvic.coupling.vic import KwargsDriver, VicConnector

            def window(s0, s1):
                fchunk = next(chunks)
                batch = image_run.ImageBatch(dom, dates[s0:s1], dtype=dtype,
                                             device=dev, elevation=elev)
                atm = image_run.build_atmosphere(fchunk, batch)
                atm.pop('_sources', None)
                return dict(batch=batch, atm=atm)

            daily_ = []

            def sink(name, prop):
                if name != 'vic':
                    return
                if a.routing:       # VIC's own daily series, for the check
                    for i_ in range(0, prop.series['runoff'].shape[0], spd):
                        daily_.append(prop.series['runoff'][i_:i_ + spd]
                                      .sum(0) + prop.series['baseflow']
                                      [i_:i_ + spd].sum(0))
                w0, w1 = prop.info['steps']
                for s in streams:
                    rec = history.stream_records(prop.series, s,
                                                 dates[w0:w1], spd,
                                                 log=lambda *x: None)
                    if rec.tail is not None:
                        raise ValueError('window does not end on an '
                                         'AGGFREQ boundary')
                    for k, v in rec.data.items():
                        parts[s.prefix].setdefault(k, []).append(v.cpu())
                say('  %s .. %s  %.1f s' % (dates[w0].date(),
                                            dates[w1 - 1].date(),
                                            time.time() - t))

            area = np.asarray(dom.area, dtype=np.float64)
            frac = np.asarray(dom.frac, dtype=np.float64)
            ok = area.shape == (nc,) and frac.shape == (nc,)
            static = dict(p=p, soil_params=soil, mosaic=mosaic,
                          wind_h=wind_h, init_moist=init_moist,
                          steps_per_day=spd, full_energy=full_energy,
                          log_every=0, **options)
            vic_opts = dict(soil_add_mode='cap')
            if a.irrigate == 'window':
                vic_opts['day_coupled'] = ['irr']
            vc = VicConnector(
                driver=KwargsDriver(static, window, dates, spd,
                                    area=area if ok else None,
                                    frac=frac if ok else None, n_cells=nc),
                window_days=a.chunk_days, **vic_opts)
            mods = dict(vic=dict(global_param=a.global_file,
                                 window_days=a.chunk_days, **vic_opts))
            conns, given = [], dict(vic=vc)
            if a.irrigate != 'none':
                # a daily irrigation stand-in: tops the top layer up towards
                # 20 mm, at most 3 mm a day, from outside the domain
                given['irr'] = _Irrigator(torch.as_tensor(
                    area * frac, dtype=torch.float64), dev)
                mods['irr'] = dict(connector='two_store')
                conns = [{'from': 'vic.soil_moisture',
                          'to': 'irr.soil_moisture', 'lag': 1},
                         {'from': 'irr.water', 'to': 'vic.soil_add'}]
            if a.routing:
                if shard_n:
                    raise SystemExit('--routing runs on one GPU')
                fd, dm, gg = a.routing.split(',')
                mods['routing'] = dict(connector='routing', fdir=fd,
                                       domain=dm, gauges=gg, step_s=86400.0)
                conns.append({'from': 'vic.total_runoff_daily',
                              'to': 'routing.inflow'})
            cm = TorchVIC.from_dict(
                dict(run=dict(device=str(dev), dtype=a.dtype, log=False),
                     modules=mods, connections=conns),
                connectors=given)
            cm.initialize()
            if a.routing:
                rcn = cm.connector('routing')
                tm['routing_cells'] = rcn.n_cell
                tm['routing_nodes'] = rcn.n_node
                tm['routing_pairs'] = int(rcn.router.n_pair)
                orig = rcn.compute

                def timed(inputs, context, dt, _o=orig):
                    sync(dev)
                    t_ = time.time()
                    r_ = _o(inputs, context, dt)
                    sync(dev)
                    tm['routing_compute_s'] += time.time() - t_
                    return r_
                rcn.compute = timed
                tm['routing_compute_s'] = 0.0
            cm.runtime.outputs_mgr.keep('vic', 'none')
            if a.irrigate != 'none':
                cm.runtime.outputs_mgr.keep('irr', 'none')
            cm.runtime.outputs_mgr.add_sink(sink)
            cm.run()
            cm.finalize(write=False)
            if a.routing:
                # the router on its own, fed VIC's daily series
                from torchvic.coupling import Context
                from torchvic.coupling.connectors import RoutingConnector
                ref = RoutingConnector(fdir=fd, domain=dm, gauges=gg)
                ref.initialize(Context(device=dev))
                qs = torch.stack([ref.router.step(x.to(torch.float64))[0]
                                  for x in daily_])
                got = cm.outputs('routing')['discharge']
                tm['routing_check'] = dict(
                    days=int(qs.shape[0]),
                    bitwise_equal=bool(torch.equal(got, qs)),
                    max_abs_diff=float((got - qs).abs().max()),
                    peak_m3s=float(got.max()), peak_standalone=float(qs.max()))
                say('routing check: %s' % json.dumps(tm['routing_check']))
            state = vc.state
            rep = cm.report()
            tm['runtime'] = dict(timing=rep['timing'],
                                 backends=rep['backends'],
                                 ledger=rep['ledger']['vic'],
                                 ledger_rows_ok=all(
                                     r['ok'] for r in cm.ledger.balances))
            say('runtime: %s' % json.dumps(tm['runtime']))
        for w0, w1 in (spans if not a.runtime else []):
            tp = time.time()
            fchunk = next(chunks)
            if w0 == 0:
                ev('first_window')
            batch = image_run.ImageBatch(dom, dates[w0:w1], dtype=dtype,
                                         device=dev, elevation=elev)
            atm = image_run.build_atmosphere(fchunk, batch)
            atm.pop('_sources', None)
            if snow_ex is not None:
                batch, atm, _, shim._cell_echo = C.expand_bands_forcing(
                    batch, atm, None, snow_ex, dtype, dev)
                shim.dates = dates[w0:w1]
            sync(dev)
            ts = time.time()
            t_pre += ts - tp
            out = basin.simulate(
                batch=batch, atm=atm, p=p, soil_params=soil, mosaic=mosaic,
                wind_h=wind_h, init_moist=init_moist, steps_per_day=spd,
                full_energy=full_energy, log_every=0,
                initial_state=state, return_state=True, out_vars=out_names,
                **band_kw, **options)
            sync(dev)
            t_sim += time.time() - ts
            tm.setdefault('sim_windows_s', []).append(round(time.time() - ts, 2))
            for k_, v_ in getattr(basin, 'GRAPH_STATS', {}).items():
                gs = tm.setdefault('graph_stats', {})
                if isinstance(v_, (int, float)):
                    gs[k_] = gs.get(k_, 0) + v_
                elif v_ not in ('{}', ''):
                    gs[k_] = gs.get(k_, '') + v_
            state = out.pop('state')
            if snow_ex is not None:
                out = shim.contract_bands(out)
            ta = time.time()
            for s in streams:
                rec = history.stream_records(out, s, dates[w0:w1], spd,
                                             log=lambda *x: None)
                if rec.tail is not None:
                    raise ValueError('window %s..%s does not end on an '
                                     'AGGFREQ boundary' % (dates[w0],
                                                          dates[w1 - 1]))
                if aw is not None:
                    aw.add(s.prefix, {k: v.cpu()
                                      for k, v in rec.data.items()})
                    continue
                for k, v in rec.data.items():
                    parts[s.prefix].setdefault(k, []).append(v.cpu())
            t_agg += time.time() - ta
            del out, atm, fchunk, batch
            say('  %s .. %s  %.1f s' % (dates[w0].date(),
                                        dates[w1 - 1].date(),
                                        time.time() - t))
        sync(dev)
        if a.dump_state and shard_i is None:
            from torchvic.model import state as _ms
            torch.save(_ms.map_state(state, lambda x: x.cpu()),
                       a.dump_state)
        ev('physics_done')
        tm['physics_s'] = time.time() - t
        tm['simulate_s'] = t_sim
        tm['window_input_s'] = t_pre
        tm['aggregate_s'] = t_agg
        if stream:
            tm['read_wait_s'] = pre.wait_s
            pre.close()
            fc.close()
        if dev.type == 'cuda':
            tm['physics_peak_gb'] = torch.cuda.max_memory_allocated() / 1e9

        # ---- write: the OUTFILE streams ---------------------------------
        t = time.time()
        os.makedirs(a.out, exist_ok=True)
        if aw is not None:
            paths = aw.close()
        elif shard_i is not None and shard_n > 1:
            # a shard hands its piece of every stream to the parent
            import pickle
            torch.save({s.prefix: {k: torch.cat(v, 0)
                                   for k, v in parts[s.prefix].items()}
                        for s in streams},
                       os.path.join(a.parts, 'data_%d.pt' % shard_i))
            np.save(os.path.join(a.parts, 'index_%d.npy' % shard_i),
                    info['index'])
            if shard_i == 0:
                with open(os.path.join(a.parts, 'meta.pkl'), 'wb') as fh:
                    pickle.dump(dict(dates=dates, spd=spd), fh)
            paths = []
        else:
            paths = write_streams(streams, dates, spd, parts, dom, gp,
                                  a.out, say, procs=a.write_procs)
        tm['write_s'] = time.time() - t
        ev('write_done')
    tm['total_s'] = time.time() - T0
    tm['events'] = EVENTS
    tm['start_epoch'] = _T_START
    tm.update(device=str(dev), core=a.core, dtype=a.dtype,
              chunk_days=a.chunk_days, n_cells=nc, n_steps=nt,
              steps_per_day=spd, daily_forcing=is_daily,
              torch=torch.__version__, threads=torch.get_num_threads())
    if dev.type == 'cuda':
        tm['gpu'] = torch.cuda.get_device_name(dev)
    if shard_i is not None:
        tm.update(shard=shard_i, shards=shard_n)
    print(json.dumps(tm, indent=1), flush=True)
    for pth in paths:
        print('wrote %s' % pth)
    if a.timing:
        with open(a.timing, 'w') as fh:
            json.dump(tm, fh, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
