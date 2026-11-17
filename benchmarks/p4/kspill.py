"""Registers and spills of every Triton kernel of an inductor output
module, and the module's benchmark (random inputs)."""
import importlib.util
import sys
import os
os.environ['TRITON_DEFAULT_FP_FUSION'] = '0'
import torch
path = sys.argv[1]
spec = importlib.util.spec_from_file_location('m', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
ks = {k: v for k, v in vars(m).items() if isinstance(v, CachingAutotuner)}
m.benchmark_compiled_module(times=20, repeat=3) if hasattr(m, 'benchmark_compiled_module') else None
for k, v in ks.items():
    for l in getattr(v, 'launchers', []):
        print('%-100s regs %s spills %s cfg %s' % (k[:100], l.n_regs, l.n_spills, l.config))
