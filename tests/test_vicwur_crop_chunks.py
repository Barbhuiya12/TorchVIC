# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""WOFOST-NPK in the coupled run: the crop state written in place a chunk of
cells at a time (WofostNPK.crop_run_inplace, compact leaf classes) equals
the run on all cells at once, and on CUDA the chunks replayed from CUDA
graphs (WofostNPK.day_graphed) equal the eager run, bit for bit (R1 G10,
every plugin)."""
import os
import sys

import pytest

torch = pytest.importorskip('torch')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools', 'vicwur'))

import l3_runs as L                                                  # noqa
from torchvic.coupling.connectors import vicwur_setup as WS          # noqa


def _run(tmp, monkeypatch, chunk, days=80, device='cpu', graph=False):
    real = WS.modules

    def modules(*a, **k):
        m = real(*a, **k)
        m['vicwur_crop']['chunk_lanes'] = chunk
        m['vicwur_crop']['graph'] = graph
        return m
    monkeypatch.setattr(WS, 'modules', modules)
    monkeypatch.setattr(L, 'NDAYS', days)
    m, _ = L.build('G10', 'fixed', tmp, device=device,
                   group='ck_%d_%s_%d' % (chunk, device, graph))
    L.run_once(m)
    return m


def test_chunked_inplace_equals_whole(tmp_path, monkeypatch):
    if not os.path.exists(L.global_path('G10')):
        pytest.skip('VIC-WUR reference run absent: G10')
    a = _run(str(tmp_path / 'a'), monkeypatch, 0)
    c = a.runtime.connectors['vicwur_crop']
    B, NC = c.m.shape[1:]
    b = _run(str(tmp_path / 'b'), monkeypatch, B * NC * 2)
    assert len(b.runtime.connectors['vicwur_crop'].m.chunks(
        b.runtime.connectors['vicwur_crop']._cells())) > 1
    assert L._diff(L.states(a), L.states(b)) is None
    assert L.outputs_equal(a, b, 0) == []
    assert L.ledger_rel(L.ledger_rows(a), L.ledger_rows(b)) == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_graphed_day_equals_eager(tmp_path, monkeypatch):
    if not os.path.exists(L.global_path('G10')):
        pytest.skip('VIC-WUR reference run absent: G10')
    a = _run(str(tmp_path / 'a'), monkeypatch, 0, 100, 'cuda', False)
    c = a.runtime.connectors['vicwur_crop']
    B, NC = c.m.shape[1:]
    b = _run(str(tmp_path / 'b'), monkeypatch, B * NC * 3, 100, 'cuda',
             True)
    assert b.runtime.connectors['vicwur_crop'].m._graphs is not None
    assert L._diff(L.states(a), L.states(b)) is None
    assert L.outputs_equal(a, b, 0) == []
