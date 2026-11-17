# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.irrigation (irrigation per-tile kernels).

Oracle: tests/oracle/ref_vicwur_irr.c, a single-cell copy of the VIC-WUR 3.0
C arithmetic (irr_run.c, irr_use.c), against the batched kernels in float64
to 1e-12 on random and edge inputs.  Replay: irr_var of the R1 G4 runs
(VICWUR_DUMP build), see tests/test_vicwur_irrigation_l2.py.
"""
import os
import struct
import subprocess

import numpy as np
import pytest
import torch

from torchvic.vicwur import irrigation as IR

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'oracle', 'ref_vicwur_irr.c')
NI, NB, NL, NF = 3, 3, 3, 2


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    exe = str(tmp_path_factory.mktemp('irr') / 'ref_irr')
    r = subprocess.run(['cc', '-O0', '-ffp-contract=off', '-o', exe, SRC,
                        '-lm'], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('cannot build oracle: ' + r.stderr[:200])
    return exe


def make_cases(n, seed):
    g = np.random.default_rng(seed)
    u = lambda *s, lo=0.0, hi=1.0: g.uniform(lo, hi, s)
    zero = lambda a, p: np.where(g.random(a.shape) < p, 0.0, a)
    d = {}
    for k in ('efficient', 'potential', 'share', 'wateruse', 'wofost',
              'new_day'):
        d[k] = (g.random(n) < 0.5).astype(float)
    d['potential'] = (g.random(n) < 0.25).astype(float)
    d['wateruse'] = (g.random(n) < 0.8).astype(float)
    d['wofost'] = (g.random(n) < 0.25).astype(float)
    d['ksat0'] = u(n, lo=1, hi=900)
    d['expt'] = np.full(n, 0.33)
    d['paddy'] = (g.random((n, NI)) < 0.3).astype(float)
    d['cv'] = zero(u(n, NI, lo=0.02, hi=0.5), 0.2)
    d['offmax'] = g.integers(0, 6, (n, NI)).astype(float)
    d['offset'] = g.integers(0, 7, (n, NI, NB)).astype(float)
    same = g.random((n, NI, NB)) < 0.6
    d['prev_cv'] = np.where(same, d['cv'][:, :, None],
                            zero(u(n, NI, NB, lo=0, hi=.5), 0.3))
    d['gw'] = u(n, NI)
    d['eff'] = u(n, NI, lo=0.3, hi=1.0)
    d['af'] = zero(u(n, NB), 0.2)
    wpwp = u(n, NL, lo=10, hi=40)
    wcr = wpwp + u(n, NL, lo=5, hi=60)
    wfc = wpwp + (wcr - wpwp) / 0.7
    d['wpwp'], d['wfc'] = wpwp, wfc
    d['maxm0'] = wfc[:, 0] + u(n, lo=1, hi=40)
    d['wcr'] = (wcr[:, None, None, :] * u(n, NI, NB, NL, lo=0.9, hi=1.0)
                * (g.random((n, NI, NB, NL)) < 0.7)
                + wcr[:, None, None, :] * (g.random((n, NI, NB, NL)) >= 0.7))
    d['wcr'] = np.where(g.random((n, NI, NB, 1)) < 0.05, 0.0, d['wcr'])
    ff = u(n, NF)
    d['frost'] = ff / ff.sum(1, keepdims=True)
    mx = d['maxm0'][:, None, None, None] * u(n, NI, NB, NL, lo=0.1, hi=1.4)
    d['moist'] = mx
    d['ice'] = zero(u(n, NI, NB, NL, NF, lo=0, hi=20), 0.6)
    r = u(n, NI, NL)
    r = zero(r, 0.3)
    r[..., 0] = np.where(r.sum(-1) == 0, 1.0, r[..., 0])
    r = r / r.sum(-1, keepdims=True)
    d['root'] = np.repeat(r[:, :, None, :], NB, 2)
    d['leftover'] = zero(u(n, NI, NB, lo=0, hi=15), 0.6)
    d['wofost_irr'] = zero(u(n, NI, NB, lo=0, hi=20), 0.6)
    d['ws_in'] = u(n, NI, NB, NL)
    d['available'] = zero(u(n, lo=0, hi=1.2), 0.2) * u(n, lo=0, hi=15)
    d['available'][:3] = -1.0
    # hand edges
    d['cv'][3] = 0.2; d['af'][3] = 0.4; d['offmax'][3] = 0
    d['moist'][4] = 0.0; d['cv'][4] = 0.3; d['af'][4] = 0.5   # dry tile
    d['offmax'][4] = 0; d['offset'][4] = 0; d['leftover'][4] = 0
    d['ice'][5] = 30.0
    d['maxm0'][6] = d['wfc'][6, 0]                            # cap 0 efficient
    d['moist'][7, :, :, 0] = d['maxm0'][7] * 1.2              # over-full
    return d


def pack(d):
    n = len(d['ksat0'])
    f = lambda k: np.asarray(d[k], float).reshape(n, -1)
    cols = [f('efficient'), f('potential'), f('share'), f('wateruse'),
            f('wofost'), f('new_day'), f('ksat0'), f('expt'), f('paddy'),
            f('cv'), f('offmax'), f('prev_cv'), f('offset'), f('gw'),
            f('eff'), f('af'), f('maxm0'), f('frost'), f('moist'),
            f('root'), f('wcr'), f('wfc'), f('wpwp'), f('ice'),
            f('leftover'), f('wofost_irr'), f('ws_in'), f('available')]
    return np.concatenate(cols, 1)


def run_oracle(exe, d, tmp_path):
    rec = pack(d)
    fi, fo = str(tmp_path / 'in.bin'), str(tmp_path / 'out.bin')
    with open(fi, 'wb') as f:
        f.write(struct.pack('<d', float(len(rec))))
        f.write(rec.astype('<f8').tobytes())
    subprocess.run([exe, fi, fo], check=True)
    o = np.fromfile(fo, '<f8').reshape(len(rec), -1)
    names = [('ksat_pre', NI * NB), ('ksat_req', NI * NB),
             ('offset', NI * NB), ('prev_cv', NI * NB), ('req', NI * NB),
             ('D', 1), ('G', 1), ('E', 1), ('D0', 1), ('flag', NI * NB),
             ('moist0', NI * NB), ('leftover', NI * NB),
             ('received', NI * NB), ('applied', NI * NB),
             ('shortage', NI * NB), ('ws', NI * NB * NL), ('cs', NI * NB),
             ('prev_left', NI * NB)]
    out, p = {}, 0
    for k, w in names:
        out[k] = o[:, p:p + w]
        p += w
    assert p == o.shape[1]
    return out


def run_torch(d, device='cpu'):
    t = lambda k: torch.as_tensor(np.asarray(d[k]), dtype=torch.float64,
                                  device=device)
    b = lambda k: t(k) > 0
    n = len(d['ksat0'])
    ice, moist = t('ice'), t('moist')
    cv, af, paddy = t('cv'), t('af'), b('paddy')
    act = IR.active_mask(cv, af)
    offset, prev = IR.offset_step(t('offset'), t('prev_cv'), cv,
                                  t('offmax'), b('new_day')[:, None, None])
    ksat_pre = IR.paddy_ksat(t('ksat0'), paddy, NB)
    ksat_req = IR.paddy_ksat(t('ksat0'), paddy, NB, active=act)
    req, flag = IR.requirement(moist, ice, t('frost'), t('root'), t('wcr'),
                               t('wfc'), t('maxm0'), t('leftover'), offset,
                               t('offmax'), paddy, cv, af)
    D, G, E, D0 = IR.demand(req, flag, cv, af, t('gw'), t('eff'))
    # per-case modes are masks: run each mode on all cells and select
    kw = dict(wfc0=t('wfc')[:, 0], max_moist0=t('maxm0'), paddy=paddy,
              cv=cv, area_fract=af)
    res = {}
    for ef in (False, True):
        for pot in (False, True):
            for wu in (False, True):
                for wo in (False, True):
                    res[(ef, pot, wu, wo)] = IR.get_withdrawn(
                        moist[..., 0], t('leftover'), req, flag, act,
                        efficient=ef, potential=pot,
                        available=t('available') if wu else None,
                        wofost=t('wofost_irr') if wo else None, **kw)
    ef_ = b('efficient'); po_ = b('potential'); wu_ = b('wateruse')
    wo_ = b('wofost')
    out = [torch.zeros_like(moist[..., 0]) for _ in range(4)]
    for key, r in res.items():
        if key[1]:
            m = (ef_ == key[0]) & po_ & (not key[2]) & (not key[3])
        else:
            m = (ef_ == key[0]) & ~po_ & (wu_ == key[2]) & (wo_ == key[3])
        for j in range(4):
            out[j] = torch.where(m[:, None, None], r[j], out[j])
    m0, left, rec, app = out
    moist2 = torch.cat([m0.unsqueeze(-1), moist[..., 1:]], -1)
    sh, ws, cs = IR.shortage_stress(
        moist2, ice, t('frost'), t('root'), t('wcr'), t('wpwp'), cv, af,
        flag, True, water_stress_in=t('ws_in'))
    sh2, ws2, cs2 = IR.shortage_stress(
        moist2, ice, t('frost'), t('root'), t('wcr'), t('wpwp'), cv, af,
        flag, False, water_stress_in=t('ws_in'))
    share = b('share')
    sh = torch.where(share[:, None, None], sh, sh2)
    ws = torch.where(share[:, None, None, None], ws, ws2)
    cs = torch.where(share[:, None, None], cs, cs2)
    r = dict(ksat_pre=ksat_pre, ksat_req=ksat_req, offset=offset,
             prev_cv=prev, req=req, D=D, G=G, E=E, D0=D0,
             flag=flag.to(torch.float64), moist0=m0, leftover=left,
             received=rec, applied=app, shortage=sh, ws=ws, cs=cs)
    r = {k: v.detach().cpu().numpy().reshape(n, -1) for k, v in r.items()}
    r['bal'] = IR.balance_residual(rec, t('leftover'), app,
                                   left).cpu().numpy().reshape(n, -1)
    return r


def compare(o, r, n):
    for k in ('ksat_pre', 'ksat_req', 'offset', 'prev_cv', 'req', 'D', 'G',
              'E', 'D0', 'flag', 'moist0', 'leftover', 'received', 'applied',
              'shortage', 'ws', 'cs'):
        np.testing.assert_allclose(r[k], o[k], rtol=1e-12, atol=1e-12,
                                   err_msg=k)


def test_l1_random_edge(oracle, tmp_path):
    d = make_cases(6000, 1)
    o = run_oracle(oracle, d, tmp_path)
    r = run_torch(d)
    compare(o, r, 6000)
    # coverage of the branches
    assert (o['flag'] > 0).sum() > 3000 and (o['flag'] == 0).sum() > 3000
    assert (o['leftover'] > 0).any() and (o['applied'] > 0).any()
    assert (o['shortage'] > 0).any() and (o['D'] > 0).any()
    pad = np.repeat(d['paddy'][:, :, None], NB, 2).reshape(6000, -1) > 0
    assert (o['flag'][pad] > 0).any() and (o['flag'][~pad] > 0).any()
    # offset gate and Cv-change reset reached
    assert (o['offset'] == 0).any() and (o['offset'] > 3).any()
    # water invariant per tile and band, on the C result
    act = (np.repeat(d['cv'][:, :, None], NB, 2) > 0) & \
          (d['af'][:, None, :] > 0)
    bal = o['received'] + o['prev_left'] - o['applied'] - o['leftover']
    assert np.abs(bal.reshape(-1, NI, NB)[act]).max() < 1e-10
    # frozen tiles keep their leftover
    assert np.all(r['bal'].reshape(-1, NI, NB)[~act] == 0.0)


def test_l1_other_seed(oracle, tmp_path):
    d = make_cases(3000, 2)
    compare(run_oracle(oracle, d, tmp_path), run_torch(d), 3000)


def test_offset_counter_across_cv_change():
    """Off-by-one of the season offset: on the day Cv changes the counter
    goes 0 -> 1, so irrigation opens offset-1 days later."""
    f = lambda *a: torch.tensor(a, dtype=torch.float64)
    off = torch.zeros(1, 1, 1, dtype=torch.float64)
    prev = torch.zeros(1, 1, 1, dtype=torch.float64)
    om = torch.tensor([[3.0]], dtype=torch.float64)
    day = torch.tensor(True)
    seq = []
    cv = torch.tensor([[0.3]], dtype=torch.float64)
    for _ in range(5):
        off, prev = IR.offset_step(off, prev, cv, om, day)
        seq.append(int(off.item()))
    assert seq == [1, 2, 3, 3, 3]           # prev_Cv starts 0: day 1 changes
    cv = torch.tensor([[0.4]], dtype=torch.float64)
    off, prev = IR.offset_step(off, prev, cv, om, day)
    assert int(off.item()) == 1             # change resets, then +1
    off, prev = IR.offset_step(off, prev, cv, om, torch.tensor(False))
    assert int(off.item()) == 1             # not a new day: unchanged


def test_apply_cap_hand():
    t = lambda *a: torch.tensor([[list(a)]], dtype=torch.float64)
    c = lambda v: torch.tensor([v], dtype=torch.float64)
    pad = torch.tensor([[False]])
    # EFFICIENT: cap = Wfc0 - moist0 = 6; received 10 -> applied 6, left 4
    m, a, l = IR.apply(t(24.), t(10.), c(30.), c(50.), pad, True)
    assert (m.item(), a.item(), l.item()) == (30.0, 6.0, 4.0)
    # not EFFICIENT: cap = max_moist0 - moist0 = 26
    m, a, l = IR.apply(t(24.), t(10.), c(30.), c(50.), pad, False)
    assert (a.item(), l.item()) == (10.0, 0.0)
    # paddy ignores EFFICIENT; over-full tile: cap clipped at 0
    m, a, l = IR.apply(t(55.), t(2.), c(30.), c(50.),
                       torch.tensor([[True]]), True)
    assert (a.item(), l.item()) == (0.0, 2.0)
    # received <= 0 is a no-op
    m, a, l = IR.apply(t(24.), t(-1.), c(30.), c(50.), pad, True)
    assert (m.item(), a.item(), l.item()) == (24.0, 0.0, 0.0)


@pytest.mark.gpu
def test_gpu(oracle, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    d = make_cases(4000, 3)
    o = run_oracle(oracle, d, tmp_path)
    compare(o, run_torch(d, 'cuda'), 4000)
