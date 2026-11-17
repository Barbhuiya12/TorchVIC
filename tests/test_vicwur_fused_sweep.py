# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The CUDA kernels of VIC-WUR (vicwur.wateruse_kernels: local water use
with its ring edit, the dams' service demand; vicwur.dams_kernels: a set of
dams on a day without a month change; vicwur.efr_kernels: the EFR search)
equal the torch path they replace bit for bit: R1 G-runs on CUDA with the
kernels on and off over 100 days (month changes included), states and
outputs compared, ledger rows to rounding (local water use sums its cell
terms once per step instead of per level)."""
import os
import sys

import pytest

torch = pytest.importorskip('torch')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools', 'vicwur'))

import l3_runs as L                                                  # noqa
from torchvic.vicwur import fused as FZ                               # noqa

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available(), reason='needs CUDA')]

try:
    import triton
    import triton.language as tl
    from torchvic.vicwur import wateruse_kernels as WK

    @triton.jit
    def _surf_k(ring_ptr, head_ptr, area_ptr, out_ptr, s0, C, NTOT, dt,
                R: tl.constexpr, BW: tl.constexpr, BLOCK: tl.constexpr):
        c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = c < C
        p = s0 + c
        z = tl.zeros([BLOCK], tl.float64)
        area = tl.load(area_ptr + p, mask=m, other=1.0)
        a = WK._surf_sum(ring_ptr, head_ptr, area, z + dt, z + 1000.0, p, m,
                         C, NTOT, R, BW, BLOCK)
        tl.store(out_ptr + c, a, mask=m)
except ImportError:                                  # no Triton
    _surf_k = None


def _need(run):
    if not os.path.exists(L.global_path(run)):
        pytest.skip('VIC-WUR reference run absent: ' + run)
    pytest.importorskip('netCDF4')


def _run(run, mode, scratch, fused, monkeypatch, days):
    monkeypatch.setattr(FZ, 'ENABLED', fused)
    monkeypatch.setattr(L, 'NDAYS', days)
    m, _ = L.build(run, mode, scratch, device='cuda',
                   group='fz_%s_%s_%d' % (run, mode, fused))
    L.run_once(m)
    return m


@pytest.mark.parametrize('run,mode', [('G10', 'fixed'), ('G10', 'exact'),
                                      ('G2', 'fixed'), ('G3', 'fixed'),
                                      ('G7', 'exact'), ('G11c', 'fixed')])
def test_fused_sweep_bitwise(tmp_path, monkeypatch, run, mode):
    _need(run)
    a = _run(run, mode, str(tmp_path / 'a'), False, monkeypatch, 100)
    b = _run(run, mode, str(tmp_path / 'b'), True, monkeypatch, 100)
    if 'vicwur_wateruse_local' in b.runtime.connectors:
        assert b.runtime.connectors['vicwur_wateruse_local']._fused
        assert not a.runtime.connectors['vicwur_wateruse_local']._fused
    assert L._diff(L.states(a), L.states(b)) is None
    assert L.outputs_equal(a, b, 0) == []
    ra, rb = L.ledger_rows(a), L.ledger_rows(b)
    assert L.ledger_rel(ra, rb) <= 1e-12, L.LEDGER_DIFF[:3]
    assert all(r['ok'] for r in b.ledger.balances)


@pytest.mark.parametrize('bug', [False, True])
def test_efr_search_kernel_bitwise(monkeypatch, bug):
    """efr.efr_step with the search kernel against the torch path on random
    states (every grid point a candidate, NaN-free and NaN cells)."""
    from torchvic.vicwur import efr as E
    g = torch.Generator(device='cuda').manual_seed(7)
    C, V, B = 3000, 9, 4
    r = lambda *s: torch.rand(*s, generator=g, device='cuda',
                              dtype=torch.float64)
    mm = 100 + 400 * r(C)
    moist = r(C, V, B) * mm[:, None, None]
    ice = r(C, V, B, 1) * 5
    args = (moist, ice, torch.ones(C, 1, device='cuda', dtype=torch.float64),
            0.05 * r(C), 0.5 + r(C), mm, 5 + 20 * r(C), 0.2 * r(C),
            0.5 + 0.4 * r(C), 1 + 2 * r(C), r(C, V), r(C, B),
            0.5 * r(C), r(C))
    valid = r(C, V) > 0.2
    out = {}
    for on in (False, True):
        monkeypatch.setattr(FZ, 'ENABLED', on)
        out[on] = E.efr_step(*args, valid=valid, bug=bug)
    for k, a, b in zip(E.EfrState._fields, out[False], out[True]):
        eq = (a == b) | (torch.isnan(a) & torch.isnan(b))
        assert bool(eq.all()), k
    # the search picked a fraction inside the grid for most cells
    assert float((out[True].frac < 1).double().mean()) > 0.1


def test_surf_sum_is_torch_order():
    """The water-use kernel's river availability sum (_surf_sum) equals
    torch's CUDA reduction of the same [C, R] terms bit for bit, for one
    cell (contiguous bins: a warp tree) and many (one thread per cell)."""
    k = _surf_k
    g = torch.Generator(device='cuda').manual_seed(3)
    for R in (27, 24, 16, 5):
        N = 5000
        ring = torch.randn(R, N, generator=g, device='cuda',
                           dtype=torch.float64) * torch.exp(
            4 * torch.randn(R, N, generator=g, device='cuda',
                            dtype=torch.float64))
        area = 1e6 + 1e8 * torch.rand(N, generator=g, device='cuda',
                                      dtype=torch.float64)
        for head in (0, min(7, R - 1)):
            hd = torch.tensor(head, device='cuda')
            phys = (hd + torch.arange(R, device='cuda')) % R
            for s0, C in ((0, 1), (11, 1), (3, 2), (100, 3), (7, 64),
                          (300, 65), (1000, 1234), (0, N)):
                cells = torch.arange(s0, s0 + C, device='cuda')
                dd = ring.index_select(1, cells).index_select(0, phys).T
                ref = (dd * 86400.0 / area.index_select(0, cells)[:, None]
                       * 1000.0).sum(1)
                out = torch.empty(C, dtype=torch.float64, device='cuda')
                k[(triton.cdiv(C, 64),)](ring, hd, area, out, s0, C, N,
                                         86400.0, R=R,
                                         BW=WK.last_pow2(R), BLOCK=64)
                assert torch.equal(out, ref), (R, head, s0, C)


def test_host_forcing_is_device_forcing(tmp_path, monkeypatch):
    """vic `forcing_on: host` (the run's forcing in pinned host memory, a
    window at a time on the device) gives the same run."""
    _need('G10')
    from torchvic.coupling.connectors import vicwur_setup as WS
    real = WS.modules
    runs = []
    for where in ('device', 'host'):
        def modules(*a, **k):
            m = real(*a, **k)
            m['vic']['forcing_on'] = where
            return m
        monkeypatch.setattr(WS, 'modules', modules)
        runs.append(_run('G10', 'fixed', str(tmp_path / where), True,
                         monkeypatch, 70))
    a, b = runs
    assert b.runtime.connectors['vic'].driver.host_forcing
    assert L._diff(L.states(a), L.states(b)) is None
    assert L.outputs_equal(a, b, 0) == []
