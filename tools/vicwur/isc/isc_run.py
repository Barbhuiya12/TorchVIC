#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The full stack of VIC-WUR on ISC (112,825 cells).

  isc_run.py [--days 365] [--profile] [--out JSON] [--mode fixed]

Builds the run from the smoke global file (tools/vicwur/isc/smoke_inputs.py)
through torchvic.coupling.connectors.vicwur_setup (load_case, modules with
the groundwater member) and runs it through the coupling runtime on the GPU.
Times the set-up and the run loop apart, reports peak GPU memory and host
RSS, checks that every ledger row closes. With --profile, the phase hooks of
the VIC-WUR group (and the VIC step, the sweep and every connector's compute)
are wrapped in cuda-synchronised timers: the result is a per-(phase, kind)
table; its total is larger than the plain run's (synchronisation), the shares
are what counts.
"""
import argparse
import collections
import datetime
import json
import os
import resource
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from torchvic.coupling import TorchVIC                               # noqa
from torchvic.coupling.connectors import vicwur_setup as WS          # noqa
from torchvic.coupling.connectors import vicwur_common as VC         # noqa
from torchvic.vicwur import order as O                               # noqa

ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
SMOKE = ROOT + '/vicwur/isc/smoke'
SCRATCH = ROOT + '/verification/vicwur_3/isc_scratch'
#: the kept output series (run.keep_outputs_on): 'none' (the timing and
#: memory of the model; nothing reads them here), 'host', or 'device' (as
#: measured: a year's history fills the GPU)
KEEP = 'none'


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sync():
    torch.cuda.synchronize()


def log(m):
    print('%s %s' % (time.strftime('%H:%M:%S'), m), flush=True)


#: the VIC module's forcing ('host': a window at a time on the device)
FORCING = 'host'


#: MODFLOW solver settings per mode. Exact (monthly, 31-day steps) with the
#: ISC statics stalled in the outer iteration (step 7, day 181: 100 outer
#: iterations; RIV/DRN/EVT switch on and off around their thresholds). The
#: VIC-WUR 3.0 settings (outer_dvclose 1e-9) are kept for the R1 replays; on
#: ISC the head closure is 1e-6 m (MF6 practice for metre heads) with 500
#: outer iterations. Fixed (daily) converges with the VIC-WUR 3.0 settings.
SOLVER = {'exact': dict(outer_dvclose=1e-6, outer_maximum=500)}


#: the CropSyst tile table of the last build (code, cell, tile, band, column)
CS_TILES = {}


def build(gp, days, device, mode, window_days, static, restart=None, cropsyst=None):
    os.makedirs(SCRATCH, exist_ok=True)
    c = WS.load_case(gp, SCRATCH, steps=days, device=device)
    log('load_case done')
    cs_irr = None
    if cropsyst is not None:
        # CropSyst on the crop tiles (tools/cropsyst/isc): the member, and the irrigation scheme on the live crop
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                                        'tools', 'cropsyst', 'isc'))
        import isc_inputs as CSI
        from torchvic.vicwur import params as P
        act = CSI.active_mask(gp)
        cs_irr = CSI.irrigation_block(int(act.sum()))
    member = None
    if cropsyst is not None:
        def member(case, start, nsteps):
            blk, tt = CSI.module_block(case, gp, nsteps, start, hooks=cropsyst.get('hooks'),
                                       n_portions=cropsyst.get('n_portions'))
            blk['weather'] = {k: (torch.as_tensor(v, device=device) if k not in ('year', 'doy') else v)
                              for k, v in blk['weather'].items()}
            CS_TILES.update(tt)
            log('cropsyst: %d tiles on %d cells' % (len(tt['code']), blk['n_cells']))
            return blk
    mods = WS.modules(gp, c, group='isc', device=device,
                      window_days=window_days, mode=mode,
                      groundwater=dict(static=static, solver=SOLVER.get(mode)),
                      cropsyst_irrigation=cs_irr, cropsyst_member=member)
    mods['vic']['forcing_on'] = FORCING
    log('modules done: %s' % ', '.join(sorted(mods)))
    cfg = dict(run=dict(device=device, log=False,
                        keep_outputs_on=KEEP), modules=mods,
               connections=[])
    if restart:
        cfg['checkpoint'] = dict(restart_from=restart)
    VC._GROUPS.clear()
    return TorchVIC.from_dict(cfg), c


class Prof(object):
    """cuda-synchronised timers on the phase scheduler."""

    def __init__(self):
        self.t = collections.defaultdict(float)
        self.n = collections.defaultdict(int)

    def patch(self):
        P = self
        call0, sweep0, run0 = (O.PhaseScheduler._call,
                               O.PhaseScheduler._sweep, O.PhaseScheduler.run)

        def _call(s, call, *a):
            sync()
            t0 = time.perf_counter()
            r = call0(s, call, *a)
            sync()
            P.add('%s %s %s' % (call.phase, call.kind, call.hook),
                  time.perf_counter() - t0)
            return r

        def _sweep(s, calls):
            sync()
            t0 = time.perf_counter()
            sweep0(s, calls)
            sync()
            P.add('P2 sweep (all slots)', time.perf_counter() - t0)

        def run(s, vic_run, view=None):
            def timed():
                sync()
                t0 = time.perf_counter()
                vic_run()
                sync()
                P.add('V vic step', time.perf_counter() - t0)
            sync()
            t0 = time.perf_counter()
            run0(s, timed, view)
            sync()
            P.add('(group.run total)', time.perf_counter() - t0)
        O.PhaseScheduler._call = _call
        O.PhaseScheduler._sweep = _sweep
        O.PhaseScheduler.run = run

    def add(self, k, dt):
        self.t[k] += dt
        self.n[k] += 1

    def table(self, days):
        tot = self.t.get('(group.run total)', 1e-9)
        rows = sorted(((k, v) for k, v in self.t.items()
                       if k != '(group.run total)'),
                      key=lambda kv: -kv[1])
        return [dict(item=k, ms_per_day=1e3 * v / days,
                     share=v / tot, calls=self.n[k]) for k, v in rows], \
            1e3 * tot / days


CROP_ERR = []


def _crop_err_off():
    from torchvic.coupling.connectors import vicwur_crop as VCR
    p0 = VCR.VicWurCropConnector.pass3

    def pass3(self):
        f0 = self.flag
        self.flag = lambda cond: None
        try:
            p0(self)
        finally:
            del self.flag
        CROP_ERR.append((self.s['err'] != 0).sum())
    VCR.VicWurCropConnector.pass3 = pass3


def ledger_status(m):
    bal = m.ledger.balances
    cpl = m.ledger.coupled
    bad = [r for r in bal if not r['ok']]
    badc = [r for r in cpl if not r['ok']]
    mods = collections.Counter(r['module'] for r in bal)
    return dict(rows=len(bal), coupled_rows=len(cpl), all_ok=not bad,
                coupled_ok=not badc, bad=[dict(module=r['module'],
                                               residual=r.get('residual'))
                                          for r in bad[:10]],
                bad_coupled=[str(r)[:200] for r in badc[:5]],
                modules=dict(mods))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2005)
    ap.add_argument('--days', type=int, default=365)
    ap.add_argument('--mode', default='fixed')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--window', type=int, default=31)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--torch-profiler', action='store_true')
    ap.add_argument('--out', default=None)
    # 'nowofost_': every plugin but WOFOST-NPK (its padded leaf classes,
    # [cell, band, crop, longest cycle + 2] x 3 fields, are 56 GB per copy
    # on ISC and the transaction keeps copies: it does not fit an H200)
    ap.add_argument('--variant', default='')
    # the first `--warm` days (compilation, graph capture, MODFLOW's cold
    # first step) run before the clock starts; ms_per_day is over the rest
    ap.add_argument('--warm', type=int, default=0)
    # cap the process's device memory (GB), as on a smaller card
    ap.add_argument('--mem-limit-gb', type=float, default=0)
    ap.add_argument('--forcing', default='host', choices=('host', 'device'))
    ap.add_argument('--keep', default='none',
                    choices=('none', 'host', 'device'))
    # the smoke crop inputs sow crops on tiles whose Cv is 0 in the land-use
    # record (WOFOST err bit 2, where the C would stop): 'ignore' measures
    # the stack with that check off (the count is reported)
    ap.add_argument('--crop-err', default='reject',
                    choices=('reject', 'ignore'))
    a = ap.parse_args()
    if a.crop_err == 'ignore':
        _crop_err_off()
    if a.mem_limit_gb:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, a.mem_limit_gb * 2**30 / tot))
    global KEEP, FORCING
    KEEP = a.keep
    FORCING = a.forcing
    gp = '%s/global_isc_%s%d.txt' % (SMOKE, a.variant, a.year)
    static = dict(np.load(SMOKE + '/gw_static.npz'))
    res = dict(days=a.days, mode=a.mode, device=a.device,
               window_days=a.window, profile=a.profile, keep=a.keep,
               variant=a.variant, mem_limit_gb=a.mem_limit_gb,
               forcing=a.forcing)
    prof = Prof() if a.profile else None
    if prof:
        prof.patch()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    m, case = build(gp, a.days, a.device, a.mode, a.window, static)
    # the set-up's case (its forcing on the device for the whole run) is
    # not the run's: the VIC module loads its own; keeping it would hold
    # a second copy of the year's forcing (15 GB at 365 days)
    del case
    t1 = time.perf_counter()
    with torch.inference_mode():
        m.initialize()
        sync()
        t2 = time.perf_counter()
        if a.warm:
            m.run(until=a.warm * 86400.0)
            sync()
            res['warm_gpu_max_allocated_gb'] = \
                torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            if prof:
                prof.t.clear()
                prof.n.clear()
            t2 = time.perf_counter()
        log('initialized; running %d days' % a.days)
        if a.torch_profiler:
            from torch.profiler import ProfilerActivity, profile
            pr = profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA])
            pr.__enter__()
        m.run()
        sync()
        t3 = time.perf_counter()
        if a.torch_profiler:
            pr.__exit__(None, None, None)
            nd = a.days - a.warm
            ka = pr.key_averages()
            res['torch_profiler_cuda_total_ms_per_day'] = sum(
                e.self_device_time_total for e in ka) / 1e3 / nd
            res['torch_profiler_cuda'] = [
                dict(name=e.key[:90], cuda_ms_per_day=e.self_device_time_total
                     / 1e3 / nd, calls=e.count)
                for e in sorted(ka,
                                key=lambda e: -e.self_device_time_total)[:40]]
        m.finalize(write=False)
    res['setup_build_s'] = t1 - t0
    res['setup_initialize_s'] = t2 - t1
    res['setup_s'] = t2 - t0
    res['run_s'] = t3 - t2
    res['warm_days'] = a.warm
    res['ms_per_day'] = 1e3 * (t3 - t2) / (a.days - a.warm)
    res['gpu_max_allocated_gb'] = torch.cuda.max_memory_allocated() / 2**30
    res['gpu_max_reserved_gb'] = torch.cuda.max_memory_reserved() / 2**30
    res['host_peak_rss_gb'] = rss_mb() / 1024
    res['ledger'] = ledger_status(m)
    if CROP_ERR:
        res['crop_err_units_max'] = int(torch.stack(CROP_ERR).max())
    try:
        res['runtime_timing'] = m.runtime.report()['timing']
    except Exception as e:                      # noqa
        res['runtime_timing'] = repr(e)
    if prof:
        res['phase_table'], res['group_run_ms_per_day_synced'] = \
            prof.table(a.days - a.warm)
    s = json.dumps(res, indent=1, default=float)
    log(s)
    if a.out:
        open(a.out, 'w').write(s)


if __name__ == '__main__':
    main()
