#!/usr/bin/env python3
"""Per-connector share and torch.profiler of one combination on a 31-day window.

    profile_coupled.py N --out DIR [--warm-day 180]

Warms to --warm-day (monsoon), times a 31-day window with every connector's
compute/validate/commit wrapped in a CUDA-synchronised timer (exclusive time:
a connector run inside another, e.g. the day-coupled modules inside VIC's
window, is subtracted from the outer), then profiles the next 31 days with
torch.profiler. Measurement only; nothing is changed in the model.
"""
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_combos as rc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('n', type=int)
    ap.add_argument('--out', required=True)
    ap.add_argument('--warm-day', type=int, default=180)
    ap.add_argument('--flood-dtype', default=None,
                    help="the flood connector's dtype (float64 default, "
                    "mixed: float32 faces, float64 depth)")
    ap.add_argument('--flood-devices', default=None,
                    help='multi-GPU flood engine, e.g. cuda:0,cuda:1')
    a = ap.parse_args()
    import torch
    from torchvic.coupling import TorchVIC
    key, name, in_repo, mode = rc.COMBOS[a.n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gdir = Path(os.environ.get('COMBO_RUNS', str(rc.COMBO / 'runs'))) / (
        'glob_prof_c%d' % a.n)
    rc.set_env(rc.write_global(gdir, eb=a.n in rc.EB))
    cfgp = (rc.REPO / 'configs' / (name + '.yaml')) if in_repo else \
        rc.COMBO / 'configs' / (name + '.yaml')
    m = TorchVIC.from_config(str(cfgp), log=rc.say)
    m.cfg['output']['dir'] = str(out / 'results')
    m.cfg['output']['history'] = False
    if a.flood_dtype:
        # the connectors exist after from_config; the engine is built at
        # initialize from this attribute
        m.runtime.connectors['flood'].dtype_name = a.flood_dtype
    if a.flood_devices:
        m.runtime.connectors['flood'].devices = a.flood_devices.split(',')

    acc = collections.defaultdict(float)
    stack = []
    timing_on = [False]

    def wrap(cn, c, meth):
        f = getattr(c, meth)

        def g(*x, **k):
            if not timing_on[0]:
                return f(*x, **k)
            torch.cuda.synchronize()
            t = time.perf_counter()
            stack.append(0.0)
            r = f(*x, **k)
            torch.cuda.synchronize()
            d = time.perf_counter() - t
            inner = stack.pop()
            acc['%s.%s' % (cn, meth)] += d - inner
            if stack:
                stack[-1] += d
            return r
        setattr(c, meth, g)

    res = dict(n=a.n, key=key, warm_day=a.warm_day,
               flood_dtype=a.flood_dtype or 'float64')
    with torch.inference_mode():
        m.initialize()
        for cn, c in m.runtime.connectors.items():
            for meth in ('compute', 'validate', 'commit'):
                wrap(cn, c, meth)
        t0 = a.warm_day * 86400.0
        m.run(until=t0)
        torch.cuda.synchronize()
        timing_on[0] = True
        t = time.perf_counter()
        m.run(until=t0 + 31 * 86400.0)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        timing_on[0] = False
        res['window_days'] = 31
        res['window_ms_per_day_synced'] = round(1e3 * wall / 31, 1)
        per = collections.defaultdict(float)
        for k, v in acc.items():
            per[k.split('.')[0]] += v
        res['per_connector_ms_per_day'] = {k: round(1e3 * v / 31, 1)
                                           for k, v in per.items()}
        res['per_connector_phase_ms_per_day'] = {
            k: round(1e3 * v / 31, 1) for k, v in acc.items()}
        res['other_ms_per_day'] = round(
            1e3 * (wall - sum(per.values())) / 31, 1)
        from torch.profiler import profile, ProfilerActivity
        t1 = t0 + 31 * 86400.0
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            m.run(until=t1 + 31 * 86400.0)
            torch.cuda.synchronize()
        ev = prof.key_averages()
        rows = []
        for e in ev:
            cu = getattr(e, 'device_time_total',
                         getattr(e, 'cuda_time_total', 0.0))
            cu_self = getattr(e, 'self_device_time_total',
                              getattr(e, 'self_cuda_time_total', 0.0))
            rows.append((cu_self, e.key, e.count, cu,
                         e.self_cpu_time_total))
        rows.sort(reverse=True)
        res['profiler_top_cuda_self'] = [
            dict(op=r[1][:90], count=r[2], cuda_self_ms_per_day=round(
                r[0] / 31e3, 2), cpu_self_ms_per_day=round(r[4] / 31e3, 2))
            for r in rows[:25]]
        res['profiler_total_cuda_self_ms_per_day'] = round(
            sum(r[0] for r in rows) / 31e3, 1)
        res['profiler_launch_counts'] = {
            r[1][:60]: r[2] for r in sorted(rows, key=lambda r: -r[2])[:8]}
        res['backends'] = m.report()['backends']
        res['gpu'] = torch.cuda.get_device_name(0)
        m.finalize()
    (out / 'profile.json').write_text(json.dumps(res, indent=1, default=str))
    rc.say('PROFILE', json.dumps({k: res[k] for k in (
        'n', 'window_ms_per_day_synced', 'per_connector_ms_per_day',
        'other_ms_per_day')}))


if __name__ == '__main__':
    main()
