# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The restart flag `checkpoint.w3_restart`.

VIC-WUR 3.0.0 restarts from a state file that holds VIC's state, the
routing ring, the non-renewable deficit and the irrigation leftover,
received, offset and prev_Cv; the new process starts every other plugin
afresh (dams empty and inactive, WOFOST units new, Cv from the parameter
file with lu_apply at its first step, VIC's stored water stress 0). R1 G12
is G10 split at 2007-01-01 (day 1280); TorchVIC's checkpoint holds
everything, so its plain restart equals the continuous run bit for bit,
and with `w3_restart` it follows VIC-WUR 3.0's own split run (exact mode,
the coupled-run tolerance 1e-6 of max(1, |x|) of the float32 VIC-WUR 3.0
files). Runs tools/vicwur/l3_runs.py's G12 set-up, shortened.
"""
import os
import sys

import pytest

torch = pytest.importorskip('torch')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools', 'vicwur'))

import l3_runs as L                                                  # noqa
from torchvic.coupling.schema import validate                       # noqa


def _need():
    if not os.path.exists(L.global_path('G12')) or len(L.w3_files('G12')) \
            < 8:
        pytest.skip('VIC-WUR reference run absent: G12')
    pytest.importorskip('xarray')
    pytest.importorskip('netCDF4')


def _split(tmp, w3_restart, days, device='cpu'):
    """G12 exact: run to the split, checkpoint, restart; returns (stopping
    run, restarted run, case)."""
    m1, case = L.build('G12', 'exact', tmp, days=days, group='w3t',
                       window_days=32, device=device)
    ck = os.path.join(tmp, 'ck.pt')
    with torch.inference_mode():
        m1.initialize().run(until=L.SPLIT * L.DAY)
        m1.save_checkpoint(ck)
    m2, _ = L.build('G12', 'exact', tmp, days=days, group='w3t',
                    restart=ck, window_days=32, w3_restart=w3_restart,
                    device=device)
    L.run_once(m2)
    return m1, m2, case


def _joined(m1, m2, case):
    r1, r2 = L.collect(m1, case), L.collect(m2, case)
    import numpy as np
    return {k: np.concatenate([r1[k][:L.SPLIT], r2[k][L.SPLIT:]
                               if len(r2[k]) > L.SPLIT else r2[k]])
            for k in r1 if k in r2}


def test_schema_takes_the_flag():
    def ck(**c):
        p = validate(dict(run={}, modules={}, connections=[], checkpoint=c))
        return [x for x in p if x.startswith('checkpoint')]
    assert ck(restart_from=__file__, w3_restart=True) == []
    assert ck(restart_from=__file__, w3_restart=False) == []
    assert any('needs checkpoint.restart_from' in x
               for x in ck(w3_restart=True))
    assert any('true or false' in x
               for x in ck(restart_from=__file__, w3_restart='yes'))


def test_flag_off_restart_equals_continuous(tmp_path, monkeypatch):
    """Without the flag the split run (G10 stopped at 2007-01-01 and
    restarted) is the continuous G10 bit for bit (states, every output row
    after the split, ledger rows)."""
    _need()
    days = L.SPLIT + 64
    monkeypatch.setattr(L, 'NDAYS', days)
    _, m2, _ = _split(str(tmp_path), False, days)
    full, _ = L.build('G12', 'exact', str(tmp_path), days=days, group='w3f',
                      window_days=32)
    L.run_once(full)
    assert L._diff(L.states(full), L.states(m2)) is None
    assert not L.outputs_equal(full, m2, L.SPLIT)
    lf, l2 = L.ledger_rows(full), L.ledger_rows(m2)
    assert lf[len(lf) - len(l2):] == l2 or lf == l2


def test_g12_w3_restart_matches_w3_split_run(tmp_path, monkeypatch):
    """With the flag, TorchVIC's split run against VIC-WUR 3.0's own split
    run (W3/out + W3/p2/out) through 2008-05 (the first season of the new
    crop units and the first year of the restarted dams, whose op_month is
    set on 2008-01-01): every compared output within the coupled-run
    tolerance; the ledger closes."""
    _need()
    days = 1770
    monkeypatch.setattr(L, 'NDAYS', days)
    m1, m2, case = _split(str(tmp_path), True, days)
    st = L.compare(_joined(m1, m2, case), L.w3('G12'), 'G10')
    worst = {k: v['gate'] for k, v in st.items()}
    assert max(worst.values()) < 1e-6, worst
    assert all(r['ok'] for r in m2.ledger.balances)
