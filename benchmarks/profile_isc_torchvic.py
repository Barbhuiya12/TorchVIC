#!/usr/bin/env python3
"""Profile TorchVIC's time loop on the ISC domain (3-hourly forcing).

    profile_isc_torchvic.py GLOBAL [--cells N] [--days D] [--start-day S]
        [--profile K] [--save FILE] [--compare FILE] [--json FILE]

The set-up is run_isc_torchvic.py's (case.load's sequence) on the 2004
forcing file. `--cells N` takes every k-th active cell so that N cells
remain (the whole domain when 0), `--days` runs that many days from
`--start-day` (days after 1 January 2004) in one basin.simulate call.

Timing: one warm-up simulate of 1 day, then the timed window (device
synchronised), reported as wall per step and peak device memory.
`--profile K` runs K steps more under torch.profiler and prints the kernel
count, kernel time against wall time, the top ops and the host syncs.
`--save` writes every output tensor of the timed window (and the carried
state) to FILE; `--compare` loads such a file and reports the largest
absolute and relative difference per output (bit-identical or not).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from torchvic import case as C
from torchvic.core import version as core_version
from torchvic.io import build as image_run
from torchvic.io.domain import ImageDomain
from torchvic.io.forcing import ImageForcing
from torchvic.io.global_param import GlobalParam
from torchvic.io.params import ImageParams
from torchvic.model import basin

CACHE_DAYS = 32
SUB = ('tair', 'prcp', 'press', 'shortwave', 'longwave', 'vp', 'wind')


def subset_domain(dom, n):
    """Every k-th active cell, n of them (spread over the whole domain)."""
    idx = np.linspace(0, dom.n_active - 1, n).round().astype(int)
    idx = np.unique(idx)
    dom.rows, dom.cols = dom.rows[idx], dom.cols[idx]
    dom.lat, dom.lon = dom.lat[idx], dom.lon[idx]
    for k in ('area', 'frac', 'flow_distance'):
        v = getattr(dom, k, None)
        if v is not None:
            setattr(dom, k, v[idx])
    dom.n_active = int(idx.size)
    if dom.run_cell is None:      # the subset is deliberate
        dom.run_cell = np.ones((dom.ny, dom.nx), np.int32)
    return idx


def flat(out, prefix=''):
    """Every tensor of a nested output dict, keyed by path."""
    res = {}
    for k, v in out.items():
        key = prefix + str(k)
        if torch.is_tensor(v):
            res[key] = v.detach().cpu()
        elif isinstance(v, dict):
            res.update(flat(v, key + '.'))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if torch.is_tensor(x):
                    res['%s.%d' % (key, i)] = x.detach().cpu()
                elif isinstance(x, dict):
                    res.update(flat(x, '%s.%d.' % (key, i)))
    return res


def compare(a, b):
    worst_abs = worst_rel = 0.0
    identical = True
    rows = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            rows.append((k, 'missing in %s' % ('new' if k in a else 'base')))
            identical = False
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            rows.append((k, 'shape/dtype %s %s vs %s %s'
                         % (x.shape, x.dtype, y.shape, y.dtype)))
            identical = False
            continue
        if not x.is_floating_point():
            same = bool((x == y).all())
            identical &= same
            if not same:
                rows.append((k, 'integer/bool differs'))
            continue
        nan_same = bool((torch.isnan(x) == torch.isnan(y)).all())
        xx = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        yy = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        d = (xx - yy).abs()
        da = float(d.max()) if d.numel() else 0.0
        scale = float(yy.abs().max()) if d.numel() else 0.0
        dr = da / scale if scale > 0 else da
        bit = bool(torch.equal(torch.nan_to_num(x, nan=1e300),
                               torch.nan_to_num(y, nan=1e300)))
        identical &= bit and nan_same
        worst_abs, worst_rel = max(worst_abs, da), max(worst_rel, dr)
        if not bit or not nan_same:
            rows.append((k, 'max|d| %.3e  rel %.3e  nan-pattern %s'
                         % (da, dr, 'same' if nan_same else 'DIFFERS')))
    return identical, worst_abs, worst_rel, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('global_file')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--core', default='5.1.0')
    ap.add_argument('--cells', type=int, default=0)
    ap.add_argument('--days', type=int, default=10)
    ap.add_argument('--start-day', type=int, default=0)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--profile', type=int, default=0,
                    help='profile this many steps after the timed window')
    ap.add_argument('--sync-debug', type=int, default=0,
                    help='count host syncs over this many steps, by site')
    ap.add_argument('--breakdown', action='store_true',
                    help='with --profile: time per model stage')
    ap.add_argument('--sites', type=int, default=0,
                    help='with --profile: the N torchvic lines that launch '
                    'the most kernels per step (eager steps only: a graph '
                    'replay has no Python stack)')
    ap.add_argument('--trace', default=None, help='chrome trace file')
    ap.add_argument('--save', default=None)
    ap.add_argument('--compare', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--cache', default=None,
                    help='forcing of the first %d days, all cells (.pt); '
                    'written on the first use' % 32)
    ap.add_argument('--compile', action='store_true',
                    help='energy.enable_compile() (column step compiled)')
    ap.add_argument('--kernels', default=None,
                    help='with --profile: TSV of device kernels by name '
                    '(count, total us), longest first')
    ap.add_argument('--timeline', default=None,
                    help='with --profile: every device kernel in order '
                    '(start us, duration us, name) to this TSV')
    ap.add_argument('--op-sites', default=None,
                    help='aten ops the warm-up day dispatches (outside the '
                    'compiled regions), by torchvic line: TSV to this file')
    ap.add_argument('--cprofile', type=int, default=0,
                    help='cProfile the last timed repeat, top N by tottime')
    ap.add_argument('--no-inference-mode', action='store_true')
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    dtype = torch.float64
    torch.set_grad_enabled(False)
    say = lambda *x: print(*x, flush=True)
    res = dict(cells_requested=a.cells, days=a.days, start_day=a.start_day)
    import os
    import subprocess
    import torchvic
    here = os.path.dirname(os.path.abspath(torchvic.__file__))
    try:
        rev = subprocess.run(['git', '-C', here, 'describe', '--always',
                              '--dirty'], capture_output=True,
                             text=True).stdout.strip()
    except Exception:
        rev = '?'
    res.update(torchvic=here, commit=rev, device_name=(
        torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'))
    say('torchvic %s (%s)' % (here, rev))

    if a.compile:
        from torchvic.core import energy
        energy.enable_compile(True)
    ctx = (torch.no_grad() if a.no_inference_mode else torch.inference_mode())
    with ctx:
        t = time.time()
        core_version.set_core(a.core)
        gp = GlobalParam(a.global_file)
        spd = gp.get_int('MODEL_STEPS_PER_DAY')
        runoff_steps = gp.get_int('RUNOFF_STEPS_PER_DAY')
        options = C.image_options(gp, log=lambda *x: None)
        dom = ImageDomain(gp.domain_file, names=gp.domain_vars)
        import os
        ck = None
        if a.cache and os.path.exists(a.cache):
            ck = torch.load(a.cache, weights_only=False)
            if int(ck['fields']['tair'].shape[1]) != dom.n_active:
                # a cache of another domain (the 105,113-cell mask before
                # the 110,353-cell one): its cells are not this domain's
                say('cache %s has %d cells, the domain %d: rebuilt'
                    % (a.cache, int(ck['fields']['tair'].shape[1]),
                       dom.n_active))
                ck = None
        if ck is not None:
            fields, times = ck['fields'], ck['times']
        else:
            fc = ImageForcing(gp.forcing_prefix, dom,
                              force_type=gp.force_type, years=['2004'])
            fields, times = fc.read(dtype=dtype, device='cpu')
            if a.cache:
                n = CACHE_DAYS * 8
                fields = {k: fields[k][:n].clone() for k in SUB}
                times = times[:n]
                torch.save(dict(fields=fields, times=times), a.cache)
        if a.cells:
            idx = subset_domain(dom, a.cells)
            fields = {k: v[:, torch.as_tensor(idx)] for k, v in
                      fields.items()}
        par = ImageParams(gp.params_file, dom)
        nc = dom.n_active
        s0 = a.start_day * spd
        s1 = s0 + a.days * spd
        host = {k: fields[k][s0 - spd if s0 >= spd else 0:s1].contiguous()
                for k in SUB}
        stamps = times[s0 - spd if s0 >= spd else 0:s1]
        del fields
        import pandas as pd
        dates = [pd.Timestamp(x).to_pydatetime() for x in stamps]
        say('read %.1f s; %d cells, %d steps' % (time.time() - t, nc,
                                                len(dates)))
        res['n_cells'] = nc

        elev = image_run.band_mean_elevation(
            par, snow_band=gp.get_bool('SNOW_BAND'))
        soil = image_run.build_soil(par, dtype, dev,
                                    runoff_steps_per_day=runoff_steps,
                                    model_steps_per_day=spd)
        fcan_src = (gp.get('FCAN_SRC') or 'FROM_DEFAULT').upper()
        mosaic = image_run.build_mosaic(par, dtype, dev, log=lambda *x: None,
                                        fcan_src=fcan_src)
        wind_h, _, _ = C.resolve_wind_h(
            gp, C.params_reference_heights(par), float(host['tair'].min()),
            mosaic=True, log=lambda *x: None)
        init_moist = torch.as_tensor(par.soil['init_moist'], dtype=dtype,
                                     device=dev).T.contiguous()
        full_energy = bool(gp.get_bool('FULL_ENERGY'))
        p = C.default_params(dtype, dev, **(gp.constants()
                                            if hasattr(gp, 'constants')
                                            else {}))
        say('n_tiles %d' % mosaic.n_tiles)
        res['n_tiles'] = int(mosaic.n_tiles)

        def run(w0, w1, state):
            f = {k: v[w0:w1].to(dev) for k, v in host.items()}
            batch = image_run.ImageBatch(dom, dates[w0:w1], dtype=dtype,
                                         device=dev, elevation=elev)
            atm = image_run.build_atmosphere(f, batch)
            atm.pop('_sources', None)
            return basin.simulate(
                batch=batch, atm=atm, p=p, soil_params=soil, mosaic=mosaic,
                wind_h=wind_h, init_moist=init_moist, steps_per_day=spd,
                full_energy=full_energy, log_every=0, initial_state=state,
                return_state=True, allow_gap=True, **options)

        def sync():
            if dev.type == 'cuda':
                torch.cuda.synchronize()

        # Warm-up: the first day (or the day before the window), which also
        # gives the window its starting state.
        t = time.time()
        nw = spd if s0 >= spd else 0
        state = None
        if os.environ.get('PROF_WARM'):
            import cProfile
            import pstats
            pr = cProfile.Profile()
            pr.enable()
        if nw:
            w = run(0, nw, None)
            state = w.pop('state')
            del w
        else:
            w = run(0, spd, None)
            del w
        if os.environ.get('PROF_WARM'):
            sync()
            pr.disable()
            for key in ('tottime', 'cumtime'):
                pstats.Stats(pr).sort_stats(key).print_stats(35)
        sync()
        if os.environ.get('PROF_COUNTERS'):
            from torch._dynamo.utils import counters
            for k_ in ('inductor', 'aot_autograd'):
                print('counters', k_, dict(counters[k_]), flush=True)
            from torch._dynamo.utils import compile_times
            print(compile_times(repr='str', aggregate=True), flush=True)
        res['warmup_s'] = time.time() - t
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        walls = []
        for r in range(a.repeat):
            sync()
            t = time.time()
            if a.cprofile and r + 1 == a.repeat:
                import cProfile
                import pstats
                pr = cProfile.Profile()
                pr.enable()
                out = run(nw, len(dates), state)
                sync()
                pr.disable()
                pstats.Stats(pr).sort_stats('tottime').print_stats(a.cprofile)
            else:
                out = run(nw, len(dates), state)
            sync()
            walls.append(time.time() - t)
            if r + 1 < a.repeat:
                del out
        nt = len(dates) - nw
        res['n_steps'] = nt
        res['wall_s'] = walls
        res['ms_per_step'] = 1000 * min(walls) / nt
        res['ns_per_cell_step'] = 1e9 * min(walls) / nt / nc
        if dev.type == 'cuda':
            res['peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
            res['out_gb'] = sum(v.numel() * v.element_size()
                                for v in flat_dev(out).values()) / 1e9
        from torchvic.core import canopy as _cn
        if _cn.WIDEN_TRIES:
            say('canopy widenings (tries: calls): %s'
                % dict(sorted(_cn.WIDEN_TRIES.items())))
        from torchvic.core import implicit as _im
        if _im.BRENT_ITERS:
            say('root_brent iterations (last index: calls): %s'
                % dict(sorted(_im.BRENT_ITERS.items())))
        from torchvic.core import graphsafe as _gs
        if _gs.SEG_PROF is not None:
            _gs.seg_summary()
        if basin.GRAPH_STATS:
            res['graph_stats'] = dict(basin.GRAPH_STATS)
            say('graphs: %s' % basin.GRAPH_STATS)
        say('timed: %d steps, %s s -> %.2f ms/step, %.1f ns/cell-step, '
            'peak %.2f GB' % (nt, ['%.2f' % x for x in walls],
                              res['ms_per_step'], res['ns_per_cell_step'],
                              res.get('peak_gb', 0)))

        if a.op_sites and a.repeat >= 1:
            import collections
            import traceback
            from torch.utils._python_dispatch import TorchDispatchMode
            opcount = collections.Counter()

            class OpSites(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    fr = [f for f in traceback.extract_stack()
                          if '/torchvic/' in f.filename]
                    key = ' <- '.join('%s:%d' % (f.filename.split('/torchvic/')[-1], f.lineno) for f in fr[-1:-3:-1])
                    opcount[(key, str(func).replace('aten.', ''))] += 1
                    return func(*args, **(kwargs or {}))
            with OpSites():
                o_ = run(nw, nw + 8, state)
                sync()
            del o_
            by = collections.Counter()
            for (k_, o2), n_ in opcount.items():
                by[k_] += n_
            with open(a.op_sites, 'w') as fh:
                for k_, n_ in by.most_common():
                    ops = collections.Counter({o2: n2 / 8 for (kk, o2), n2 in opcount.items() if kk == k_})
                    fh.write('%.1f\t%s\t%s\n' % (n_ / 8, k_, dict(ops.most_common(4))))
        if a.save or a.compare:
            cur = flat(out)
            if a.save:
                torch.save(cur, a.save)
                say('saved %d tensors to %s' % (len(cur), a.save))
            if a.compare:
                base = torch.load(a.compare)
                # A run on the covered tiles only (basin.LANE_COMPACT) has
                # no value on an uncovered tile: per-tile rows and the carried
                # state are compared on the covered tiles.
                from torchvic.model.lanes import Lanes
                ln = Lanes(mosaic.cv.cpu())
                lc, lt = ln.lane_cell, ln.lane_tile
                pad = (ln.n_cells, ln.n_tiles)
                for k_ in list(base):
                    b_, c_ = base[k_], cur.get(k_)
                    if c_ is None or b_.dim() < 2 or tuple(b_.shape[:2]) != pad:
                        continue
                    if (tuple(c_.shape) != tuple(b_.shape)
                            and c_.shape[0] == ln.n_lanes):
                        base[k_] = b_[lc, lt].unsqueeze(1)
                    elif (tuple(c_.shape) == tuple(b_.shape)
                          and k_ == 'soil_moisture_tile'):
                        base[k_], cur[k_] = b_[lc, lt], c_[lc, lt]
                ident, wa, wr, rows = compare(cur, base)
                res.update(identical=ident, max_abs=wa, max_rel=wr)
                say('compare vs %s: %s, max|d| %.3e, max rel %.3e'
                    % (a.compare, 'BIT-IDENTICAL' if ident else 'DIFFERENT',
                       wa, wr))
                for r_ in rows[:40]:
                    say('   %s: %s' % r_)
        del out

        if a.sync_debug and dev.type == 'cuda':
            # Every device-to-host synchronisation in K steps, by the
            # torchvic line that asked for it.
            import collections
            import traceback
            import warnings
            sites = collections.Counter()

            def show(message, category, filename, lineno, file=None,
                     line=None):
                fr = [f for f in traceback.extract_stack()
                      if '/torchvic/' in f.filename]
                key = (' <- '.join('%s:%d' % (f.filename.split('/torchvic/')[-1],
                                              f.lineno)
                                   for f in fr[-1:-4:-1]) or
                       '%s:%d' % (filename, lineno))
                sites[key] += 1
            old_show = warnings.showwarning
            warnings.showwarning = show
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.showwarning = show
                torch.cuda.set_sync_debug_mode(1)
                try:
                    o = run(nw, nw + a.sync_debug, state)
                    sync()
                finally:
                    torch.cuda.set_sync_debug_mode(0)
            warnings.showwarning = old_show
            del o
            tot = sum(sites.values())
            res['syncs'] = dict(steps=a.sync_debug, total=tot,
                                per_step=tot / a.sync_debug,
                                sites=dict(sites.most_common()))
            say('host syncs in %d steps: %d (%.1f/step)'
                % (a.sync_debug, tot, tot / a.sync_debug))
            for k_, v_ in sites.most_common(40):
                say('  %6d  %s' % (v_, k_))

        if a.profile and a.breakdown:
            # Label the time loop's main stages, so the profile's table
            # attributes host and device time to them.
            from torch.profiler import record_function
            from torchvic.core import (canopy, column, energy, implicit,
                                       snow, vegetation)
            from torchvic.model import lanes as lanes_mod

            def label(owner, name, tag):
                fn = getattr(owner, name)

                def wrapped(*aa, **kk):
                    with record_function('STAGE ' + tag):
                        return fn(*aa, **kk)
                setattr(owner, name, wrapped)
            for owner, name, tag in (
                    (column, 'step', 'column.step'),
                    (snow, 'snow_step', 'snow.snow_step'),
                    (canopy, 'canopy_step', 'canopy.canopy_step'),
                    (energy, 'water_balance_ground_flux',
                     'energy.water_balance_ground_flux'),
                    (vegetation.VegMosaic, 'step', 'VegMosaic.step'),
                    (basin, '_wb_thin_pack', 'basin._wb_thin_pack'),
                    (basin, 'export_state', 'basin.export_state'),
                    (canopy, 'canopy_residual', 'canopy.canopy_residual'),
                    (canopy, '_widen_and_solve', 'canopy._widen_and_solve'),
                    (canopy, 'melt_and_release', 'canopy.melt_and_release'),
                    (canopy, 'intercept', 'canopy.intercept'),
                    (vegetation, 'canopy_interception',
                     'vegetation.canopy_interception'),
                    (snow, 'snow_pack_energy_balance',
                     'snow.snow_pack_energy_balance'),
                    (implicit, 'illinois', 'implicit.illinois'),
                    (basin, '_mask_tree', 'basin._mask_tree'),
                    (energy, 'thermal_properties', 'energy.thermal_properties'),
                    (vegetation, 'aerodynamic_resistance_overstory',
                     'vegetation.aerodynamic_resistance_overstory'),
                    (lanes_mod.Lanes, 'tsum', 'Lanes.tsum'),
                    (lanes_mod.Lanes, 'to_cell', 'Lanes.to_cell')):
                if hasattr(owner, name):
                    label(owner, name, tag)

        if a.profile:
            from torch.profiler import profile, ProfilerActivity
            k = a.profile
            acts = [ProfilerActivity.CPU]
            if dev.type == 'cuda':
                acts.append(ProfilerActivity.CUDA)
            sync()
            with profile(activities=acts, record_shapes=False,
                         with_stack=bool(a.sites)) as prof:
                t = time.time()
                o = run(nw, nw + k, state)
                sync()
                wall = time.time() - t
            del o
            ev = prof.key_averages()
            if a.sites:
                import collections
                by = collections.Counter()
                for e in prof.events():
                    if e.name != 'cudaLaunchKernel':
                        continue
                    p_, fr = e, []
                    while p_ is not None and not fr:
                        fr = [f for f in (p_.stack or [])
                              if '/torchvic/' in f]
                        p_ = p_.cpu_parent
                    by[fr[0].split('/torchvic/')[-1] if fr else '?'] += 1
                say('kernel launches per step by torchvic line:')
                for k_, v_ in by.most_common(a.sites):
                    say('  %7.1f  %s' % (v_ / k, k_))
            if a.breakdown:
                # kernel launches inside each labelled stage (nested stages
                # count in their parents too), per step
                evs = prof.events()
                stages = [e for e in evs if e.name.startswith('STAGE ')
                          and e.device_type != torch.autograd.DeviceType.CUDA]
                launches = sorted(e.time_range.start for e in evs
                                  if e.name == 'cudaLaunchKernel')
                import bisect
                cnt = {}
                for e in stages:
                    n_ = (bisect.bisect_right(launches, e.time_range.end)
                          - bisect.bisect_left(launches, e.time_range.start))
                    cnt[e.name] = cnt.get(e.name, 0) + n_
                say('kernel launches per step by stage (of %.0f):'
                    % (len(launches) / k))
                for nm, n_ in sorted(cnt.items(), key=lambda x: -x[1]):
                    say('  %7.0f  %s' % (n_ / k, nm[6:]))
            CUDA = torch.autograd.DeviceType.CUDA
            dev_ev = [e for e in ev if e.device_type == CUDA
                      and not e.key.startswith('STAGE ')]
            n_k = sum(e.count for e in dev_ev)
            kt = sum(getattr(e, 'self_device_time_total',
                             getattr(e, 'self_cuda_time_total', 0))
                     for e in dev_ev) / 1e6
            if not dev_ev:
                kt = sum(getattr(e, 'self_device_time_total',
                                 getattr(e, 'self_cuda_time_total', 0))
                         for e in ev) / 1e6
                n_k = sum(e.count for e in ev if e.key == 'cudaLaunchKernel')
            res['profile'] = dict(steps=k, wall_s=wall, kernels=n_k,
                                  kernels_per_step=n_k / k,
                                  kernel_time_s=kt,
                                  gpu_busy_frac=kt / wall if wall else 0)
            names = {}
            for e in ev:
                if e.key in ('cudaStreamSynchronize', 'cudaDeviceSynchronize',
                             'cudaMemcpyAsync', 'cudaLaunchKernel',
                             'aten::item', 'aten::_local_scalar_dense',
                             'aten::nonzero', 'aten::is_nonzero',
                             'cudaMemcpy', 'cudaEventSynchronize',
                             'aten::copy_', 'aten::to', 'aten::masked_select',
                             'aten::unique', 'aten::index'):
                    names[e.key] = e.count
            if a.timeline:
                kv = sorted((e.time_range.start, e.time_range.elapsed_us(),
                             e.name) for e in prof.events()
                            if e.device_type == CUDA
                            and not e.name.startswith('STAGE '))
                t0_ = kv[0][0] if kv else 0
                with open(a.timeline, 'w') as fh:
                    for st_, du_, nm_ in kv:
                        fh.write('%.1f\t%.1f\t%s\n' % (st_ - t0_, du_, nm_))
            if a.kernels:
                rows = sorted(((getattr(e, 'self_device_time_total',
                                        getattr(e, 'self_cuda_time_total', 0)),
                                e.count, e.key) for e in dev_ev),
                              reverse=True)
                with open(a.kernels, 'w') as fh:
                    for t_, c_, n_ in rows:
                        fh.write('%.0f\t%d\t%s\n' % (t_ / k, c_ // k, n_))
            res['profile']['calls'] = names
            say('profile %d steps: wall %.3f s, %d kernels (%.0f/step), '
                'kernel time %.3f s, GPU busy %.1f%%'
                % (k, wall, n_k, n_k / k, kt, 100 * kt / wall))
            say('calls: %s' % json.dumps(names))
            sort = ('device_time_total' if dev.type == 'cuda'
                    else 'cpu_time_total')
            try:
                say(ev.table(sort_by=sort, row_limit=40))
            except Exception:
                say(ev.table(sort_by='cuda_time_total', row_limit=40))
            say(ev.table(sort_by='count', row_limit=25))
            if os.environ.get('P4_NAMES'):
                # the device kernels by total time, full names, per step
                dk = sorted(dev_ev, key=lambda e: -getattr(
                    e, 'self_device_time_total',
                    getattr(e, 'self_cuda_time_total', 0)))
                say('kernels by device time (us/step, calls/step, name):')
                for e in dk[:int(os.environ['P4_NAMES'])]:
                    tt = getattr(e, 'self_device_time_total',
                                 getattr(e, 'self_cuda_time_total', 0))
                    say('  K %8.1f %7.1f  %s' % (tt / k, e.count / k,
                                                 e.key[:160]))
            stages = {}
            for e in ev:
                if e.key.startswith('STAGE ') or e.key == 'run':
                    stages[e.key[6:]] = dict(
                        calls=e.count, cpu_s=e.cpu_time_total / 1e6,
                        device_s=getattr(e, 'device_time_total',
                                         getattr(e, 'cuda_time_total', 0))
                        / 1e6)
            if stages:
                res['profile']['stages'] = stages
                for k_, v_ in sorted(stages.items(),
                                     key=lambda kv: -kv[1]['cpu_s']):
                    say('  stage %-36s %5d calls  host %.3f s  device %.3f s'
                        % (k_, v_['calls'], v_['cpu_s'], v_['device_s']))
            if a.trace:
                prof.export_chrome_trace(a.trace)
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return 0


def flat_dev(out):
    res = {}
    for k, v in out.items():
        if torch.is_tensor(v):
            res[k] = v
    return res


if __name__ == '__main__':
    sys.exit(main())
