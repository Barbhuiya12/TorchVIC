"""Where a process's compile time goes: the profile driver, then dynamo's
per-phase totals. AUTOGRAD_CACHE=1 turns on AOTAutograd's cache."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
if os.environ.get('AUTOGRAD_CACHE') == '1':
    import torch._functorch.config as fc
    fc.enable_autograd_cache = True
import profile_isc_torchvic as P  # noqa: E402

t0 = time.time()
P.main()
import torch._dynamo.utils as du  # noqa: E402
print('== compile times (total wall %.1f s)' % (time.time() - t0))
print(du.compile_times(repr='str', aggregate=True))
try:
    ms = du.get_compilation_metrics()
    rows = []
    for m in ms:
        rows.append((getattr(m, 'entire_frame_compile_time_s', None) or 0.0,
                     '%s [%s]' % (getattr(m, 'co_name', '?'), getattr(m, 'compile_id', '?')), getattr(m, 'co_filename', '?').split('/')[-1],
                     getattr(m, 'backend_compile_time_s', None) or 0.0,
                     getattr(m, 'graph_op_count', None)))
    rows.sort(reverse=True)
    print('per-frame compile (s): total %.1f over %d frames' % (sum(r[0] for r in rows), len(rows)))
    for r in rows[:30]:
        print('  %6.2f  backend %5.2f  ops %5s  %s (%s)' % (r[0], r[3], r[4], r[1], r[2]))
except Exception as e:
    print('metrics:', e)
try:
    from torch._dynamo.utils import counters
    for k in ('inductor', 'aot_autograd', 'stats'):
        print(k, dict(counters[k]))
except Exception as e:
    print(e)
