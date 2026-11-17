# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The R1 G-runs coupled, and their restarts.

Every G-run is built from its VIC-WUR 3.0 global file by vicwur_setup and run
through the coupling runtime with VIC (the vicwur3 core) in the loop
(tools/vicwur/l3_runs.py; the 7-year runs of every G-run in both modes are
in <data root>/verification/vicwur_3/l3). Here, shorter: exact mode within the
coupled-run tolerance (1e-6 of max(1, |x|) of the float32 VIC-WUR 3.0 files)
and restarts that equal the continuous run bit for bit in both modes, at
checkpoints inside a crop season, a monthly land-use record and a MODFLOW
month.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools', 'vicwur'))

import l3_runs as L                                                  # noqa


def _need(run):
    if not os.path.exists(L.global_path(run)) or not L.w3_files(run):
        pytest.skip('VIC-WUR reference run absent: ' + run)
    pytest.importorskip('xarray')
    pytest.importorskip('netCDF4')


def _short(monkeypatch, days, restart_days=None):
    monkeypatch.setattr(L, 'NDAYS', days)
    if restart_days:
        monkeypatch.setattr(L, 'RESTART_DAYS', restart_days)


@pytest.mark.parametrize('run', ['G7', 'G10', 'G11c'])
def test_l3_exact_within_the_gate(tmp_path, monkeypatch, run):
    """A year of the run in exact mode: every compared output within the
    coupled-run tolerance of VIC-WUR 3.0, the ledger closes."""
    _need(run)
    _short(monkeypatch, 365)
    m, case = L.build(run, 'exact', str(tmp_path))
    L.run_once(m)
    st = L.compare(L.collect(m, case), L.w3(run), run)
    worst = {k: v['gate'] for k, v in st.items()}
    assert max(worst.values()) < 1e-6, worst
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)


@pytest.mark.parametrize('mode', ['exact', 'fixed'])
@pytest.mark.parametrize('run', ['G10', 'G11c'])
def test_restart_equals_continuous(tmp_path, monkeypatch, run, mode):
    """G10 (every plugin, crops) and G11c (GWM + MODFLOW) over 434 days,
    restarted on 2004-05-06 and 2004-08-07 (31-day windows: mid-month, so
    inside a monthly land-use record and a MODFLOW month; August inside
    the kharif crop season): states, outputs and ledger rows bit for bit,
    and the stopping run equals the continuous run."""
    _need(run)
    _short(monkeypatch, 434, (310, 403))
    m, _ = L.build(run, mode, str(tmp_path))
    L.run_once(m)
    rep = L.restart_check(run, mode, str(tmp_path), m)
    assert rep['stopping_run_states'] == 'bitwise', rep
    assert rep['stopping_run_ledger'] == 'bitwise', rep
    assert rep['stopping_run_outputs'] == 'bitwise', rep
    for k, v in rep.items():
        if k.startswith('restart_'):
            assert v['states'] == 'bitwise', (k, v)
            assert v['outputs'] == 'bitwise', (k, v)
            assert v['ledger'] == 'bitwise', (k, v)
            assert v['ledger_ok'], (k, v)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_g10_restart_equals_continuous(tmp_path, monkeypatch):
    """The same on CUDA, G10 fixed, 124 days restarted at day 62: states
    and outputs bit for bit, ledger rows to rounding."""
    _need('G10')
    _short(monkeypatch, 124, (62,))
    monkeypatch.setattr(L, 'build', _cuda(L.build))
    m, _ = L.build('G10', 'fixed', str(tmp_path))
    L.run_once(m)
    rep = L.restart_check('G10', 'fixed', str(tmp_path), m)
    r = rep['restart_62']
    assert rep['stopping_run_states'] == 'bitwise', rep
    assert r['states'] == 'bitwise' and r['outputs'] == 'bitwise', r
    # the ledger's sums are device reductions (atomics on CUDA): the
    # rows of the restarted run equal the continuous ones to rounding
    assert r['ledger_rel'] <= 1e-12 and r['ledger_ok'], (
        r['ledger_rel'], L.LEDGER_DIFF[:3])


def _cuda(f):
    def g(*a, **k):
        k['device'] = 'cuda'
        return f(*a, **k)
    return g
