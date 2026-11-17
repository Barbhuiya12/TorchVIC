# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The flood engine and fusion.setup().

`fusion.setup()` rewires inductor for the whole process (no FMA contraction,
eager-identical reductions). Two facts about the flood engine follow:

* The coupled flood connector runs the engine's default scheme ('torchvic'),
  which on a GPU is the hand-written Triton stepper: nothing is compiled by
  inductor, so its depths are the same whether setup() ran before it or not.
* Scheme 'lisflood_gpu' (the comparison with the LISFLOOD-FP CUDA binary, which
  nvcc compiles with FMA) is compiled by inductor, and agrees with the binary
  only when setup() has not run in the process. That is a property of the
  comparison, not of the engine, and the engine says so with a RuntimeWarning
  (tests/test_flood_gpu.py runs it in a fresh interpreter)."""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUDA = torch.cuda.is_available()

PROG = r'''
import hashlib, sys
sys.path.insert(0, %r)
import torch
if sys.argv[1] == 'early':
    from torchvic.core import fusion
    fusion.setup()
compiled = []
_c = torch.compile
torch.compile = lambda f, *a, **k: (compiled.append(1), _c(f, *a, **k))[1]
import validate_engine as V
c = V.cases(['basin'])['basin']
h = V.run_engine(c, 'torchvic', 'cuda')['h']
print('COMPILED', len(compiled))
print('DIGEST', hashlib.sha256(h.tobytes()).hexdigest())
''' % os.path.join(ROOT, 'tools', 'flood')


def _run(order):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''))
    out = subprocess.run([sys.executable, '-c', PROG, order], env=env, cwd=ROOT,
                         capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0, out.stderr[-2000:]
    return dict(x.split() for x in out.stdout.splitlines() if x.startswith(('DIGEST', 'COMPILED')))


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
def test_coupled_flood_scheme_does_not_depend_on_setup_order():
    early, late = _run('early'), _run('late')
    assert early['COMPILED'] == late['COMPILED'] == '0', 'the default scheme went through inductor'
    assert early['DIGEST'] == late['DIGEST']


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
def test_lisflood_gpu_scheme_warns_after_setup(monkeypatch):
    from torchvic.core import fusion
    from torchvic.flood.engine import FloodEngine
    monkeypatch.setattr(fusion, '_ready', True)
    with pytest.warns(RuntimeWarning, match='setup'):
        FloodEngine(np.zeros((8, 8)), 10.0, 0.03, scheme='lisflood_gpu', device='cuda')
    monkeypatch.setattr(fusion, '_ready', False)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        FloodEngine(np.zeros((8, 8)), 10.0, 0.03, scheme='lisflood_gpu', device='cuda')
