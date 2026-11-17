"""GPU idle gaps in a chrome trace (profile_isc_torchvic --trace): per
step, by the host call running while the device waited."""
import json
import sys
import collections

tr = json.load(open(sys.argv[1]))
nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 16
ev = tr['traceEvents'] if isinstance(tr, dict) else tr
gpu = sorted([e for e in ev if e.get('ph') == 'X' and e.get('cat') in
              ('kernel', 'gpu_memcpy', 'gpu_memset')], key=lambda e: e['ts'])
cpu = [e for e in ev if e.get('ph') == 'X' and e.get('cat') in
       ('cuda_runtime', 'cpu_op', 'python_function', 'user_annotation')]
cpu.sort(key=lambda e: e['ts'])
t0, t1 = gpu[0]['ts'], gpu[-1]['ts'] + gpu[-1]['dur']
# steady state: the last FRAC of the span (the window's first step is
# eager, before its graph is captured)
FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
cut = t1 - FRAC * (t1 - t0)
gpu = [e for e in gpu if e['ts'] >= cut]
t0 = gpu[0]['ts']
busy = 0.0
gaps = []
end = gpu[0]['ts']
for e in gpu:
    if e['ts'] > end:
        gaps.append((end, e['ts'] - end))
    end = max(end, e['ts'] + e['dur'])
    busy += e['dur']
print('steady part: span %.1f ms, kernel sum %.1f ms, gaps %d total %.1f ms'
      % ((t1 - t0) / 1e3, busy / 1e3, len(gaps), sum(g for _, g in gaps) / 1e3))
nk = sum(1 for e in gpu if e.get('cat') == 'kernel')
nm = sum(1 for e in gpu if e.get('cat') != 'kernel')
print('steady: %.0f kernels/step, %.0f memcpy+memset/step, span %.2f ms/step'
      % (nk / nsteps, nm / nsteps, (t1 - t0) / 1e3 / nsteps))
cat = collections.Counter()
catt = collections.Counter()
for e in gpu:
    n = e['name']
    if e.get('cat') != 'kernel':
        k = n.split(' (')[0]
    elif n.startswith('triton_'):
        k = 'triton ' + n.split('_')[1]
    elif 'reduce_kernel' in n:
        k = 'aten reduce'
    elif 'index' in n.lower() or 'gather' in n.lower() or 'scatter' in n.lower():
        k = 'aten index'
    elif 'elementwise' in n:
        k = 'aten elementwise'
    else:
        k = 'aten other: ' + n[:50]
    cat[k] += 1
    catt[k] += e['dur']
for k, v in cat.most_common(14):
    print('  %-44s %7.1f/step %8.1f us/step' % (k[:44], v / nsteps, catt[k] / nsteps))
import re as _re
an = collections.Counter()
for e in gpu:
    n = e['name']
    if e.get('cat') == 'kernel' and not n.startswith('triton_'):
        m = _re.search(r'native::(?:\(anonymous namespace\)::)?([A-Za-z_]+)(?:<[^<>]*(?:<[^<>]*>[^<>]*)*>)?', n)
        fun = _re.findall(r'(?:Functor|_kernel|kernel_impl|_kernel_cuda)[A-Za-z_]*', n)
        key = n[:30] + ' | ' + ' '.join(_re.findall(r'([A-Za-z]+Functor[A-Za-z_]*|[a-z_]+_kernel[a-z_]*)', n)[:2])
        an[key] += 1
print('aten kernels by name (per step):')
for k, v in an.most_common(30):
    print('  %6.1f  %s' % (v / nsteps, k[:150]))
hist = collections.Counter()
for _, g in gaps:
    b = '<2us' if g < 2 else '<5us' if g < 5 else '<20us' if g < 20 else \
        '<100us' if g < 100 else '<1ms' if g < 1000 else '>=1ms'
    hist[b] += g
print('gap time by size (ms):', {k: round(v / 1e3, 2) for k, v in hist.items()})
# what the host was doing at the middle of each gap >= 5 us
import bisect
starts = [e['ts'] for e in cpu]
by = collections.Counter()
for s, g in gaps:
    if g < 5:
        continue
    m = s + g / 2
    i = bisect.bisect_right(starts, m)
    names = []
    for e in cpu[max(0, i - 400):i]:
        if e['ts'] <= m <= e['ts'] + e['dur']:
            names.append(e['name'][:60])
    key = ' > '.join(names[-3:]) if names else '(host idle/python)'
    by[key] += g
for k, v in by.most_common(25):
    print('  %8.2f ms/step  %s' % (v / 1e3 / nsteps, k))
# host-side duration of graph launches and syncs in the steady part
for name in ('cudaGraphLaunch', 'cudaEventSynchronize', 'cudaStreamSynchronize',
             'cudaMemcpyAsync', 'cudaLaunchKernel'):
    ds = sorted(e['dur'] for e in cpu if e['name'] == name and e['ts'] >= cut)
    if ds:
        print('%-22s n/step %6.1f  total %.2f ms/step  median %.1f us  max %.1f us'
              % (name, len(ds) / nsteps, sum(ds) / 1e3 / nsteps,
                 ds[len(ds) // 2], ds[-1]))
