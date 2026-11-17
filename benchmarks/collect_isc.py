#!/usr/bin/env python3
"""Collect the ISC benchmark runs into results.csv, a summary and a figure.

    collect_isc.py   (reads benchmarks/results_isc/logs, writes
                      results_isc/results.csv, summary.csv, wall_per_year.png)

One row per run in results.csv: wall clock (run_isc.sh, end to end), peak
host memory (/usr/bin/time -v: the largest single process), the phase split
(VIC's own timing table; TorchVIC's JSON), peak GPU memory. Runs over
January 2004 only (NAME *_jan2004) are extrapolated to the 2-year run and
marked so (and their phase times scaled the same way): VIC as init + run time x 731/31 days; TorchVIC as read + setup +
MTCLIM x 911/211 MTCLIM days + (physics + write) x 731/31.
summary.csv: per configuration, median and range of the wall clock, simulated
years per day (SYPD), cell-steps per second, speed-up against one core
(the single-rank VIC run; the extrapolated one if the full run is absent) and against the full node (VIC, 384
MPI ranks).
"""
import csv
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results_isc')
LOGS = os.path.join(RES, 'logs')
N_CELLS, N_STEPS, N_DAYS = 110353, 5848, 731
DOMAIN_LABEL, COMMIT = 'ISC 110,353 cells', '86ce3f56'
SIM_YEARS = N_DAYS / 365.25

CONFIGS = [  # (name prefix, label, hardware)
    ('vic_n384_r', 'VIC 5.1.0 -O3, 384 MPI ranks', 'CPU node, 2x EPYC 9965 (384 cores)'),
    ('vic_n192_r', 'VIC 5.1.0 -O3, 192 MPI ranks', 'CPU node, 2x EPYC 9965 (192 of 384 cores)'),
    ('vic_omp384_r', 'VIC 5.1.0 -O3, 1 rank x 384 OpenMP threads', 'CPU node, 2x EPYC 9965 (384 cores)'),
    ('vic_n1_r', 'VIC 5.1.0 -O3, 1 rank', '1 core of EPYC 9965'),
    ('vic_n1_jan2004', 'VIC 5.1.0 -O3, 1 rank (extrapolated from Jan 2004)', '1 core of EPYC 9965'),
    ('vic_nobands_n384_r', 'VIC 5.1.0 -O3, 384 ranks, SNOW_BAND FALSE', 'CPU node, 2x EPYC 9965 (384 cores)'),
    ('tv_h200_3h_r', 'TorchVIC, 3-hourly forcing', '1x H200 NVL (141 GB)'),
    ('tv_h200_daily_r', 'TorchVIC, daily forcing + MTCLIM on GPU', '1x H200 NVL (141 GB)'),
    ('tv_h200_3h_basefix_r', 'TorchVIC 152fe8a (before the speed work), 3-hourly forcing', '1x H200 NVL (141 GB)'),
    ('tv_h200_3h_optimised_r', 'TorchVIC c224741 (perf branch) OPTIMISED (canopy/snow on active tiles, zwt curves once, no constant syncs), 3-hourly forcing', '1x H200 NVL (141 GB)'),
    ('tv_h200_3h_perf2_r', 'TorchVIC 5df54ba (perf2: deterministic segment_reduce sums, forcing prefetch 2), 3-hourly forcing', '1x H200 NVL (141 GB)'),
    ('tv_rtx6k_3h_perf2_r', 'TorchVIC 5df54ba (perf2: deterministic segment_reduce sums, forcing prefetch 2), 3-hourly forcing', '1x RTX 6000 Ada (48 GB)'),
    ('tv_h200_3h_maina77a_r', 'TorchVIC a77afb3 (main before the graphs branch), 3-hourly forcing', '1x H200 NVL (141 GB)'),
    ('tv_h200_3h_graphs_r', 'TorchVIC 7d78e3b (graphs: water balance on the covered tiles only, one CUDA graph per window), 3-hourly forcing', '1x H200 NVL (141 GB)'),
    ('tv_rtx6k_3h_graphs_r', 'TorchVIC 7d78e3b (graphs: water balance on the covered tiles only, one CUDA graph per window), 3-hourly forcing', '1x RTX 6000 Ada (48 GB)'),
    ('tv_h200_3h_mainc376_r', 'TorchVIC c376b90 (main before the graphs merge: root_brent snow and canopy), 3-hourly forcing', '1x H200 NVL (141 GB)'),
    ('tv_h200_3h_graphs2_r', 'TorchVIC 5581c49 (graphs on c376b90: covered tiles only, CUDA graphs with root_brent loops), 3-hourly forcing', '1x H200 NVL (141 GB)'),
    ('tv_rtx6k_3h_graphs2_r', 'TorchVIC 5581c49 (graphs on c376b90: covered tiles only, CUDA graphs with root_brent loops), 3-hourly forcing', '1x RTX 6000 Ada (48 GB)'),
    ('tv_rtx6k_3h_perf2_pf0', 'TorchVIC 5df54ba (perf2), 3-hourly forcing, prefetch 0 (forcing read before the time loop)', '1x RTX 6000 Ada (48 GB)'),
    ('tv_rtx6k_3h_r', 'TorchVIC, 3-hourly forcing', '1x RTX 6000 Ada (48 GB)'),
    ('tv_rtx6k_daily_r', 'TorchVIC, daily forcing + MTCLIM on GPU', '1x RTX 6000 Ada (48 GB)'),
    ('tv_cpu_3h_jan2004', 'TorchVIC, 3-hourly forcing (extrapolated from Jan 2004)', 'CPU node, 384 torch threads'),
    ('tv_cpu_daily_jan2004', 'TorchVIC, daily forcing + MTCLIM (extrapolated from Jan 2004)', 'CPU node, 384 torch threads'),
]


def time_file(name):
    out = {}
    p = os.path.join(LOGS, name + '.time')
    for line in open(p):
        line = line.strip()
        if line.startswith('wall_s'):
            out['wall_s'] = float(line.split()[1])
        elif line.startswith('Maximum resident set size'):
            out['peak_host_gb'] = int(line.split(':')[1]) / 1e6
        elif line.startswith('Exit status'):
            out['exit'] = int(line.split(':')[1])
    return out


def vic_table(name):
    t = {}
    for line in open(os.path.join(LOGS, name + '.log')):
        m = re.match(r'\|\s*(Init|Run|Final|Total|Force|Write) Time\s*\|\s*([0-9.]+)', line)
        if m:
            t[m.group(1).lower()] = float(m.group(2))
    return t


def collect():
    rows = []
    for tf in sorted(glob.glob(os.path.join(LOGS, '*.time'))):
        name = os.path.basename(tf)[:-5]
        cfg = next((c for c in CONFIGS if name.startswith(c[0])), None)
        if cfg is None:
            continue
        r = dict(run=name, domain=DOMAIN_LABEL, commit=COMMIT, config=cfg[1], hardware=cfg[2])
        r.update(time_file(name))
        if 'wall_s' not in r or r.get('exit', 0) != 0:
            continue
        jan = name.endswith('jan2004')
        r['measured_days'] = 31 if jan else N_DAYS
        r['measured_wall_s'] = r['wall_s']
        if name.startswith('vic'):
            t = vic_table(name)
            r.update(read_init_s=t.get('init'), forcing_read_s=t.get('force'),
                     write_s=t.get('write'),
                     physics_s=(t['run'] - t['force'] - t['write']) if t else None)
            if jan:
                r['wall_s'] = (r['wall_s'] - t['run']) + t['run'] * N_DAYS / 31
                for k in ('forcing_read_s', 'write_s', 'physics_s'):
                    r[k] *= N_DAYS / 31
        else:
            j = json.load(open(os.path.join(LOGS, name + '.json')))
            r.update(read_init_s=j['read_s'] + j['setup_s'],
                     mtclim_s=j['mtclim_s'], physics_s=j['physics_s'],
                     simulate_s=j['simulate_s'], write_s=j['write_s'],
                     peak_gpu_gb=max(j.get('mtclim_peak_gb', 0),
                                     j.get('physics_peak_gb', 0)) or None,
                     chunk_days=j['chunk_days'])
            if jan:
                r['wall_s'] = (r['wall_s'] - j['mtclim_s'] - j['physics_s']
                               - j['write_s']
                               + j['mtclim_s'] * 911 / 211
                               + (j['physics_s'] + j['write_s']) * N_DAYS / 31)
                r['mtclim_s'] *= 911 / 211
                for k in ('physics_s', 'simulate_s', 'write_s'):
                    r[k] *= N_DAYS / 31
        r['extrapolated'] = jan
        rows.append(r)
    return rows


def summarise(rows):
    full = [r['wall_s'] for r in rows if r['run'].startswith('vic_n384_r')]
    one = ([r['wall_s'] for r in rows if r['run'].startswith('vic_n1_r')]
           or [r['wall_s'] for r in rows if r['run'].startswith('vic_n1_jan2004')])
    nobl = [r['wall_s'] for r in rows if r['run'].startswith('vic_nobands_n384_r')]
    nob = st.median(nobl) if nobl else None
    node = st.median(full) if full else None
    core = st.median(one) if one else None
    out = []
    for prefix, label, hw in CONFIGS:
        rs = [r for r in rows if r['run'].startswith(prefix)]
        if not rs:
            continue
        w = [r['wall_s'] for r in rs]
        med = st.median(w)
        def mean_of(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return st.median(v) if v else None
        out.append(dict(
            domain=DOMAIN_LABEL, commit=COMMIT,
            config=label, hardware=hw, n_runs=len(rs),
            extrapolated=rs[0]['extrapolated'],
            wall_median_s=med, wall_min_s=min(w), wall_max_s=max(w),
            wall_per_sim_year_s=med / SIM_YEARS,
            sypd=86400.0 * SIM_YEARS / med,
            cell_steps_per_s=N_CELLS * N_STEPS / med,
            speedup_vs_1core=(core / med) if core else None,
            speedup_vs_full_node=(node / med) if node else None,
            speedup_vs_vic_nobands=(nob / med) if nob else None,
            read_init_s=mean_of('read_init_s'), forcing_read_s=mean_of('forcing_read_s'),
            mtclim_s=mean_of('mtclim_s'), physics_s=mean_of('physics_s'),
            write_s=mean_of('write_s'),
            peak_host_gb=max(r.get('peak_host_gb') or 0 for r in rs),
            peak_gpu_gb=mean_of('peak_gpu_gb')))
    return out


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})


def figure(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = [x for x in summary]
    lab = ['%s\n%s' % (x['config'].replace(' (extrapolated from Jan 2004)', '*'),
                       x['hardware']) for x in s]
    y = [x['wall_per_sim_year_s'] for x in s]
    lo = [x['wall_per_sim_year_s'] - x['wall_min_s'] / SIM_YEARS for x in s]
    hi = [x['wall_max_s'] / SIM_YEARS - x['wall_per_sim_year_s'] for x in s]
    col = ['#4a6fa5' if x['config'].startswith('VIC') else '#c8553d' for x in s]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(s) + 1.5))
    pos = range(len(s))[::-1]
    ax.barh(list(pos), y, xerr=[lo, hi], color=col, capsize=3)
    for p_, v in zip(pos, y):
        ax.text(v * 1.08, p_, '%.0f s' % v, va='center', fontsize=8)
    ax.set_yticks(list(pos))
    ax.set_yticklabels(lab, fontsize=7.5)
    ax.set_xscale('log')
    ax.set_xlabel('wall clock per simulated year, s (log scale; end to end; '
                  'bars: median, whiskers: range)')
    ax.set_title('ISC, 110,353 cells, 3-hourly water balance, 2004-2005 '
                 '(* extrapolated from January 2004)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    rows = collect()
    write_csv(os.path.join(RES, 'results.csv'), rows)
    summary = summarise(rows)
    write_csv(os.path.join(RES, 'summary.csv'), summary)
    figure(summary, os.path.join(RES, 'wall_per_year.png'))
    print('| configuration | hardware | runs | wall s (median, range) | s per sim. year | SYPD | cell-steps/s | vs 1 core | vs full node | peak host GB | peak GPU GB |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    f = lambda v, p='%.1f': '-' if v is None else p % v
    for x in summary:
        print('| %s | %s | %d | %.0f (%.0f-%.0f) | %.0f | %.0f | %.3g | %s | %s | %s | %s |'
              % (x['config'], x['hardware'], x['n_runs'], x['wall_median_s'],
                 x['wall_min_s'], x['wall_max_s'], x['wall_per_sim_year_s'],
                 x['sypd'], x['cell_steps_per_s'],
                 f(x['speedup_vs_1core']), f(x['speedup_vs_full_node'], '%.2f'),
                 f(x['peak_host_gb']), f(x['peak_gpu_gb'])))
    print('\n| configuration | hardware | read+init s | forcing read s | MTCLIM s | physics s | write s |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for x in summary:
        print('| %s | %s | %s | %s | %s | %s | %s |'
              % (x['config'], x['hardware'], f(x['read_init_s'], '%.0f'),
                 f(x['forcing_read_s'], '%.0f'), f(x['mtclim_s'], '%.0f'),
                 f(x['physics_s'], '%.0f'), f(x['write_s'], '%.0f')))


if __name__ == '__main__':
    main()
