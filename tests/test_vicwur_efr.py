# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.efr (EFR kernel).

Oracle: tests/oracle/ref_vicwur_efr.c, a single-cell copy of
efr_derived_forcing and efr_put_data (VIC-WUR 3.0 plugins/efr/src), against
the batched kernel in float64 to 1e-12 on random and edge inputs, with and
without the cell-0 Ws/Ds/c bug of the non-linear term.  The frac of the grid
search must match exactly.
Replay of the R1 G7 dump (the run with EFR): tests/test_vicwur_efr_l2.py.
"""
import os
import struct
import subprocess

import numpy as np
import pytest
import torch

from torchvic.vicwur import efr as E
from torchvic.vicwur import wateruse as W

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'oracle', 'ref_vicwur_efr.c')
NV, NB, NF, NG = 3, 3, 2, 100


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    exe = str(tmp_path_factory.mktemp('efr') / 'ref_efr')
    r = subprocess.run(['cc', '-O0', '-ffp-contract=off', '-o', exe, SRC,
                        '-lm'], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('cannot build oracle: ' + r.stderr[:200])
    return exe


def make_cases(n, seed, bug):
    g = np.random.default_rng(seed)
    u = lambda *s, lo=0.0, hi=1.0: g.uniform(lo, hi, s)
    zero = lambda a, p: np.where(g.random(a.shape) < p, 0.0, a)
    d = {'bug': np.full(n, float(bug))}
    d['nv'] = g.integers(1, NV + 1, n).astype(float)
    d['spd'] = g.choice([1.0, 2.0, 4.0], n)
    d['resid'] = u(n, lo=0.0, hi=0.12)
    d['depth'] = u(n, lo=0.3, hi=2.0)
    d['maxm'] = d['depth'] * 1000 * u(n, lo=0.3, hi=0.5)
    d['Dsmax'] = u(n, lo=0.5, hi=40.0)
    d['Ds'] = u(n, lo=0.001, hi=0.9)
    d['Ws'] = u(n, lo=0.05, hi=0.97)
    d['c'] = u(n, lo=1.0, hi=6.0)
    d['Ds0'] = np.full(n, d['Ds'][0])          # cell 0 of the domain
    d['Ws0'] = np.full(n, d['Ws'][0])
    d['c0'] = np.full(n, d['c'][0])
    d['disch'] = zero(u(n, lo=0, hi=400), 0.2)
    ff = u(n, NF)
    ff[g.random(n) < 0.3] = [1.0, 0.0]
    d['ff'] = ff / ff.sum(1, keepdims=True)
    d['AF'] = zero(u(n, NB), 0.2)
    d['Cv'] = zero(u(n, NV, lo=0.01, hi=0.6), 0.25)
    res = d['resid'] * d['depth'] * 1000
    # moisture from below the residual to above saturation, so liq*frac
    # crosses the residual and the Ws kink at different frac
    d['moist'] = res[:, None, None] + u(n, NV, NB, lo=-0.1, hi=1.15) * \
        (d['maxm'] - res)[:, None, None]
    d['ice'] = zero(u(n, NV, NB, NF, lo=0, hi=30), 0.5)
    # required baseflow: spans the whole grid, plus 0 and very large
    req = 10.0 ** g.uniform(-4, 1.7, n)
    req[g.random(n) < 0.08] = 0.0
    req[g.random(n) < 0.05] = 1e9
    d['req'] = req
    # edge cells
    d['nv'][0] = NV
    d['ice'][1] = 0.0
    d['moist'][2] = res[2]                       # liquid at the residual
    d['AF'][3] = 0.0                             # no band at all
    d['Cv'][4] = 0.0                             # Cv = 0 everywhere
    return d


ORDER = ['bug', 'nv', 'spd', 'resid', 'depth', 'maxm', 'Dsmax', 'Ds', 'Ws',
         'c', 'Ds0', 'Ws0', 'c0', 'req', 'disch', 'ff', 'AF', 'Cv', 'moist',
         'ice']


def run_oracle(exe, d, tmp_path):
    n = len(d['req'])
    rec = np.concatenate([np.asarray(d[k], dtype=np.float64).reshape(n, -1)
                          for k in ORDER], axis=1)
    fi, fo = str(tmp_path / 'in.bin'), str(tmp_path / 'out.bin')
    with open(fi, 'wb') as f:
        f.write(struct.pack('<d', float(n)))
        f.write(np.ascontiguousarray(rec, dtype='<f8').tobytes())
    subprocess.run([exe, fi, fo], check=True)
    o = np.fromfile(fo, dtype='<f8').reshape(n, 3 + NV * NB + NG)
    return dict(frac=o[:, 0], cell=o[:, 1], disch=o[:, 2],
                moist=o[:, 3:3 + NV * NB].reshape(n, NV, NB),
                calc=o[:, 3 + NV * NB:])


def run_torch(d, bug, device='cpu', gchunk=None):
    n = len(d['req'])
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64,
                                  device=device)
    nv = torch.as_tensor(d['nv'], device=device)
    valid = torch.arange(NV, device=device)[None, :] < nv[:, None]
    spd = d['spd']
    out = []
    # steps_per_day is a scalar of the global file: run each value's subset
    res = None
    for s in np.unique(spd):
        ix = np.nonzero(spd == s)[0]
        sel = lambda k: t(np.asarray(d[k])[ix])
        r = E.efr_step(sel('moist'), sel('ice'), sel('ff'), sel('resid'),
                       sel('depth'), sel('maxm'), sel('Dsmax'), sel('Ds'),
                       sel('Ws'), sel('c'), sel('Cv'), sel('AF'), sel('req'),
                       sel('disch'), steps_per_day=float(s),
                       valid=valid[torch.as_tensor(ix, device=device)],
                       bug=bug, gchunk=gchunk)
        out.append((ix, r))
    # the bug reads cell 0 of the batch: only valid when one subset; the
    # tests keep spd constant per call when bug is on (see test_l1)
    return out


def compare(o, ix, r, d, tol=1e-12):
    cpu = lambda x: x.detach().cpu().numpy()
    np.testing.assert_array_equal(cpu(r.frac), o['frac'][ix], 'frac')
    np.testing.assert_allclose(cpu(r.efr_moist), o['moist'][ix],
                               rtol=tol, atol=tol, err_msg='efr_moist')
    np.testing.assert_allclose(cpu(r.cell_moist), o['cell'][ix],
                               rtol=tol, atol=tol, err_msg='OUT_EFR_MOIST')
    np.testing.assert_allclose(cpu(r.discharge), o['disch'][ix])


@pytest.mark.parametrize('bug', [True, False])
@pytest.mark.parametrize('seed', [1, 2])
def test_l1_random_edge(oracle, tmp_path, bug, seed):
    d = make_cases(4000, seed, bug)
    if bug:                          # one steps_per_day so cell 0 is shared
        d['spd'][:] = 4.0
    o = run_oracle(oracle, d, tmp_path)
    for ix, r in run_torch(d, bug):
        # efr_moist of padded tiles (>= nv) is 0 in C and unused: mask
        keep = (np.arange(NV)[None, :] < d['nv'][ix][:, None])
        mm = np.where(keep[:, :, None], r.efr_moist.numpy(), 0.0)
        r = r._replace(efr_moist=torch.as_tensor(mm))
        compare(o, ix, r, d)
    # coverage actually reached
    f = o['frac']
    assert (f == 1.0).sum() > 200
    assert (np.abs(f) < 1e-12).sum() > 100          # no violation: ~ -7.5e-16
    assert ((f > 0.05) & (f < 0.95)).sum() > 500    # interior grid points
    assert (d['req'] == 0).any() and (d['req'] > 1e8).any()


@pytest.mark.parametrize('bug', [True, False])
def test_l1_trial_baseflow_grid(oracle, tmp_path, bug):
    """The full 100-point scan against the C calculated_baseflow."""
    d = make_cases(3000, 5, bug)
    d['spd'][:] = 1.0
    o = run_oracle(oracle, d, tmp_path)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
    ice = E.ice_sum(t(d['ice']), t(d['ff']))
    liq = t(d['moist']) - ice
    grid, last = E.search_grid()
    valid = torch.arange(NV)[None, :] < torch.as_tensor(d['nv'])[:, None]
    calc = E.trial_baseflow(liq, grid, t(d['resid']) * t(d['depth']) * 1000.,
                            t(d['maxm']), t(d['Dsmax']) / 1.0, t(d['Ds']),
                            t(d['Ws']), t(d['c']), t(d['Cv']), t(d['AF']),
                            valid, bug)
    np.testing.assert_allclose(calc.numpy(), o['calc'], rtol=1e-12,
                               atol=1e-12)


def test_cell0_bug_matters():
    """exact mode reads Ds, Ws, c of cell 0 in the non-linear term; the
    fixed mode reads the cell's own, so they differ when the cells differ."""
    d = make_cases(500, 7, True)
    d['spd'][:] = 1.0
    a = run_torch(d, True)[0][1]
    b = run_torch(d, False)[0][1]
    assert (a.frac != b.frac).any()
    # cell 0 itself is the same in both modes
    assert a.frac[0] == b.frac[0]


def test_grid_constants():
    grid, last = E.search_grid()
    assert grid.shape == (100,) and grid[0].item() == 1.0
    f = 1.0
    for v in grid.tolist():
        assert v == f
        f -= 0.01
    assert last.item() == f and abs(f + 7.53e-16) < 1e-17


def test_chunking_does_not_change_result():
    d = make_cases(300, 9, False)
    d['spd'][:] = 1.0
    a = run_torch(d, False)[0][1]
    for ch in (1, 7, 33):
        b = run_torch(d, False, gchunk=ch)[0][1]
        assert torch.equal(a.frac, b.frac)
        assert torch.equal(a.efr_moist, b.efr_moist)


def _cell(moist=120., ice=0., req=0.0, nb=1):
    t = lambda *a: torch.tensor([list(a)], dtype=torch.float64)
    z = lambda v: torch.tensor([v], dtype=torch.float64)
    kw = dict(moist=torch.full((1, 1, nb), moist, dtype=torch.float64),
              ice=torch.full((1, 1, nb, 1), ice, dtype=torch.float64),
              frost_fract=torch.ones(1, 1, dtype=torch.float64),
              resid=z(0.05), depth=z(1.0), max_moist=z(400.),
              dsmax_year=z(10.), ds=z(0.3), ws=z(0.6), c=z(2.0),
              cv=torch.ones(1, 1, dtype=torch.float64),
              area_fract=torch.full((1, nb), 1.0 / nb, dtype=torch.float64),
              req_baseflow=z(req), req_discharge=z(1.0))
    return kw


def test_hand_req_zero_gives_ice_and_agw_drops_by_ice():
    kw = _cell(moist=120., ice=15., req=0.0)
    r = E.efr_step(**kw)
    # frac falls through to -7.5e-16: efr_moist = liq*frac + ice ~ ice
    assert abs(r.efr_moist.item() - 15.0) < 1e-12
    resid_mm = torch.tensor([50.], dtype=torch.float64)
    a0, A0 = W.availability_gw(kw['moist'], kw['ice'], kw['frost_fract'],
                               resid_mm, kw['area_fract'], kw['cv'])
    a1, A1 = W.availability_gw(kw['moist'], kw['ice'], kw['frost_fract'],
                               resid_mm, kw['area_fract'], kw['cv'],
                               efr_moist=r.efr_moist)
    # ice is subtracted twice: A_gw drops by the ice amount
    assert abs((A0 - A1).item() - 15.0) < 1e-12


def test_hand_req_above_full_baseflow_gives_frac_one():
    kw = _cell(moist=300., ice=5., req=1e6)
    r = E.efr_step(**kw)
    assert r.frac.item() == 1.0
    assert r.efr_moist.item() == 300.0          # liq*1 + ice = moist


def test_hand_baseflow_requirement_is_met_by_the_grid_point():
    """frac is the first grid point whose trial baseflow is below req, then
    one step up: the trial baseflow at the returned frac is >= req unless
    frac is 1 (the step up lands on a grid point that was not below)."""
    kw = _cell(moist=300., ice=0., req=0.5)
    r = E.efr_step(**kw)
    grid, _ = E.search_grid()
    liq = kw['moist']
    calc = E.trial_baseflow(liq, grid, torch.tensor([50.], dtype=torch.float64),
                            kw['max_moist'], kw['dsmax_year'], kw['ds'],
                            kw['ws'], kw['c'], kw['cv'], kw['area_fract'])
    k = int(torch.argmax((calc < 0.5).to(torch.int8)))
    assert calc[0, k] < 0.5 and (k == 0 or calc[0, k - 1] >= 0.5)
    assert abs(r.frac.item() - min(1.0, grid[k].item() + 0.01)) < 1e-15


def test_hand_discharge_zero_gives_surface_zero():
    c = lambda v: torch.tensor([v], dtype=torch.float64)
    s = E.surface_scale(c(5.), c(0.))
    assert s.item() == 0.0
    a = W.availability_surf(torch.ones(1, 4, dtype=torch.float64), 86400.,
                            c(1e8), c(5.), c(0.))
    assert a.item() == 0.0
    # efr_q 0 with discharge 0: NaN in C, min gives 1 -> scale 0 as well
    assert E.surface_scale(c(0.), c(0.)).item() == 0.0
    # half the discharge reserved
    assert E.surface_scale(c(2.), c(4.)).item() == 0.5
    # more than the discharge reserved: all of it
    assert E.surface_scale(c(9.), c(4.)).item() == 0.0
    # dam EFR volume in hm3 (dam_register.c:241)
    assert abs(E.dam_efr(c(2.), 86400., c(0.5)).item() - 0.0864) < 1e-15


def test_tile_band_output_shapes_and_cell_sum():
    d = make_cases(50, 11, False)
    d['spd'][:] = 1.0
    r = run_torch(d, False)[0][1]
    assert r.efr_moist.shape == (50, NV, NB)
    ref = np.zeros(50)
    for j in range(NV):
        for k in range(NB):
            ref += r.efr_moist[:, j, k].numpy() * d['Cv'][:, j] * d['AF'][:, k] \
                * (j < d['nv'])
    np.testing.assert_allclose(r.cell_moist.numpy(), ref, rtol=1e-12)


@pytest.mark.gpu
def test_gpu_matches_oracle_without_host_sync(oracle, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    d = make_cases(3000, 3, True)
    d['spd'][:] = 2.0
    o = run_oracle(oracle, d, tmp_path)
    t = lambda k: torch.as_tensor(np.asarray(d[k]), dtype=torch.float64,
                                  device='cuda')
    valid = (torch.arange(NV, device='cuda')[None, :] <
             torch.as_tensor(d['nv'], device='cuda')[:, None])
    args = [t(k) for k in ('moist', 'ice', 'ff', 'resid', 'depth', 'maxm',
                           'Dsmax', 'Ds', 'Ws', 'c', 'Cv', 'AF', 'req',
                           'disch')]
    E.efr_step(*args, steps_per_day=2.0, valid=valid)     # warm up
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode('error')
    try:
        r = E.efr_step(*args, steps_per_day=2.0, valid=valid, bug=True)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    keep = (np.arange(NV)[None, :] < d['nv'][:, None])
    mm = np.where(keep[:, :, None], r.efr_moist.cpu().numpy(), 0.0)
    r = r._replace(efr_moist=torch.as_tensor(mm), frac=r.frac.cpu(),
                   cell_moist=r.cell_moist.cpu(), discharge=r.discharge.cpu())
    compare(o, np.arange(3000), r, d)
