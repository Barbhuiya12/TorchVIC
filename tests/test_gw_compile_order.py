# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The groundwater solver's compiled kernels do not depend on when the fused
regions were set up.

`core.fusion.setup()` rewires inductor for the whole process (reductions,
divisions and powers as eager's). It used to run lazily at the first fused
region, so a function compiled before it and one compiled after it rounded
differently. The groundwater solver was compiled at its first step, which
came before the first fused region in a continuous run and after it in a run
restarted from a checkpoint: the two differed in the last bits from the first
solve on (Nandgaon, all_wur_cropsyst, 2002-01-01 checkpoint). The solver now
runs `fusion.ensure_setup()` before it compiles."""
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PROG = r'''
import hashlib, sys
import torch
from torchvic.core import fusion
if sys.argv[1] == 'early':
    fusion.setup()
from torchvic.groundwater import Grid, GroundwaterModel
g = Grid.dis(2, 12, 15, 250.0, 200.0, 40.0, [10.0, -30.0])
m = GroundwaterModel(g, k11=[8.0, 3.0], k33=[0.8, 0.3], icelltype=[1, 0],
                     ss=1e-5, sy=0.15, strt=25.0, variablecv=True,
                     outer_dvclose=1e-10, inner_dvclose=1e-11, inner_rclose=1e-8,
                     device='cuda', cuda_graph=True)
m.set_recharge(2e-3)
m.set_wells(([1, 0], [5, 8], [6, 3]), [-4e3, -1e3], auto_flow_reduce=0.1)
m.set_rivers(([0] * 3, [2, 3, 4], [10, 10, 10]), 24.0, 2e3, 20.0)
for _ in range(6):
    r = m.step(1.0)
print('SETUP_RAN', int(fusion._ready))
print('DIGEST', hashlib.sha256(m.h.cpu().numpy().tobytes()).hexdigest())
'''


def _run(order):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''))
    out = subprocess.run([sys.executable, '-c', PROG, order], env=env, cwd=ROOT,
                         capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0, out.stderr[-2000:]
    d = dict(x.split() for x in out.stdout.splitlines() if x.startswith(('DIGEST', 'SETUP_RAN')))
    return d


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_solver_heads_do_not_depend_on_when_fusion_was_set_up():
    early, late = _run('early'), _run('late')
    assert early['SETUP_RAN'] == '1'
    assert late['SETUP_RAN'] == '1', 'the solver compiled without fusion.setup()'
    assert early['DIGEST'] == late['DIGEST']


def test_ensure_setup_calls_setup_once_it_applies(monkeypatch):
    from torchvic.core import fusion
    calls = []
    monkeypatch.setattr(fusion, 'setup', lambda: calls.append(1))
    monkeypatch.setattr(fusion, 'FUSION', True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    fusion.ensure_setup()
    assert calls == [1]
    monkeypatch.setattr(fusion, 'FUSION', False)
    fusion.ensure_setup()                                   # fused regions off: nothing
    assert calls == [1]
    monkeypatch.setattr(fusion, 'FUSION', True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    fusion.ensure_setup()                                   # no GPU: nothing
    assert calls == [1]


def test_groundwater_compile_goes_through_ensure_setup(monkeypatch):
    """The solver's compile sites ask for the setup first (checked on any
    device: compile is forced, torch.compile replaced)."""
    from torchvic.core import fusion
    from torchvic.groundwater import Grid, GroundwaterModel
    order = []
    monkeypatch.setattr(fusion, 'ensure_setup', lambda: order.append('setup'))
    monkeypatch.setattr(torch, 'compile', lambda f, **k: (order.append('compile'), f)[1])
    g = Grid.dis(1, 1, 5, 100.0, 50.0, 10.0, [-90.0])
    m = GroundwaterModel(g, k11=5.0, strt=0.0, compile=True, cuda_graph=False)
    m._functions()
    assert order[:2] == ['setup', 'compile']
    assert order.count('setup') == 1 and order.count('compile') == 4
