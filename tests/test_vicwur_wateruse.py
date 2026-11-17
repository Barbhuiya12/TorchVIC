# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.wateruse (water-use core and local water use).

Oracle: tests/oracle/ref_vicwur_wu_local.c, a single-cell copy of the
VIC-WUR 3.0 C arithmetic, against the batched kernels in float64 to 1e-12,
on random and edge inputs.  Replay spot check: wu_var of the dumped R1 G10
run (VICWUR_DUMP build).
"""
import os
import _paths  # noqa: E402
import struct
import subprocess

import numpy as np
import pytest
import torch

from torchvic.vicwur import wateruse as W

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'oracle', 'ref_vicwur_wu_local.c')
NS, NV, NB, NF, NK, ND = 5, 3, 3, 2, 8, 3
DUMP = _paths.env_path(('TORCHVIC_VICWUR_R1_DUMP', 'VICWUR_R1_DUMP'), 'verification/vicwur_03/gruns/R1/G10_dump/dump.bin')


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    exe = str(tmp_path_factory.mktemp('wu') / 'ref_wu')
    r = subprocess.run(['cc', '-O0', '-ffp-contract=off', '-o', exe, SRC,
                        '-lm'], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('cannot build oracle: ' + r.stderr[:200])
    return exe


def make_cases(n, seed):
    """Random cells with edge values mixed in (probabilities per field)."""
    g = np.random.default_rng(seed)

    def u(*shape, lo=0.0, hi=1.0):
        return g.uniform(lo, hi, shape)

    def zero(a, p):
        return np.where(g.random(a.shape) < p, 0.0, a)

    d = {}
    d['efr'] = (g.random(n) < 0.5).astype(float)
    d['gwm'] = (g.random(n) < 0.1).astype(float)
    d['act'] = (g.random((n, NS)) < 0.8).astype(float)
    d['dem'] = zero(u(n, NS, lo=-0.3, hi=8), 0.2)
    d['gfr'] = np.where(g.random((n, NS)) < 0.3,
                        g.integers(0, 2, (n, NS)), u(n, NS))
    d['cf'] = u(n, NS)
    keep = g.random(n) < 0.5
    d['ret0'] = np.where(keep[:, None], zero(u(n, NS, hi=2), .5), 0.0)
    d['con0'] = np.where(keep[:, None], zero(u(n, NS, hi=2), .5), 0.0)
    d['resid'] = u(n, lo=0, hi=60)
    d['moist'] = u(n, NV, NB, lo=0, hi=150)
    d['efrm'] = u(n, NV, NB, lo=0, hi=60)
    d['ice'] = zero(u(n, NV, NB, NF, lo=0, hi=40), 0.5)
    ff = u(n, NF)
    d['ff'] = ff / ff.sum(1, keepdims=True)
    d['AF'] = zero(u(n, NB), 0.2)
    d['Cv'] = zero(u(n, NV), 0.2)
    q = zero(u(n, NK, lo=0, hi=300), 0.3)
    q[g.random(n) < 0.15] = 0.0            # empty ring: zero availability
    d['dtq'] = q
    d['dt'] = np.full(n, 86400.0)
    d['area'] = u(n, lo=1e8, hi=1e9)
    d['disch'] = zero(u(n, lo=0, hi=300), 0.2)
    d['efr_q'] = zero(u(n, lo=0, hi=400), 0.2)
    d['stor'] = zero(u(n, ND, lo=0, hi=40), 0.3)
    d['dact'] = (g.random((n, ND)) < 0.7).astype(float)
    d['dloc'] = (g.random((n, ND)) < 0.7).astype(float)
    d['deficit'] = zero(u(n, lo=0, hi=3), 0.4)
    # scale so demand sometimes exceeds, sometimes not, availability
    sc = 10.0 ** g.uniform(-2, 1.2, n)
    d['dem'] = d['dem'] * sc[:, None]
    # hand-made edge cells
    e = 0
    d['moist'][e] = d['resid'][e]; d['Cv'][e] = 1; d['AF'][e] = 0.5   # A_gw=0
    d['dtq'][e + 1] = 0; d['stor'][e + 1] = 0                       # no water
    d['dem'][e + 1] = 3.0; d['act'][e + 1] = 1; d['gfr'][e + 1] = 0.5
    d['disch'][e + 2] = 0; d['efr_q'][e + 2] = 0; d['efr'][e + 2] = 1  # 0/0
    d['dem'][e + 3] = -1.0; d['act'][e + 3] = 1                     # negative
    d['gwm'][e + 3] = 0
    d['moist'][e + 4] = 0; d['ice'][e + 4] = 50                     # negative av
    d['dtq'][e + 5] = 0; d['stor'][e + 5] = 5; d['dact'][e + 5] = 1
    d['dloc'][e + 5] = 1; d['dem'][e + 5] = 2.0; d['act'][e + 5] = 1  # dam only
    d['gwm'][e + 5] = 0
    return d


def pack(d):
    n = len(d['efr'])
    cols = [d['efr'][:, None], d['gwm'][:, None], d['act'], d['dem'],
            d['gfr'], d['cf'], d['ret0'], d['con0'],
            d['moist'].reshape(n, -1), d['efrm'].reshape(n, -1),
            d['ice'].reshape(n, -1), d['ff'], d['resid'][:, None], d['AF'],
            d['Cv'], d['dtq'], d['dt'][:, None], d['area'][:, None],
            d['efr_q'][:, None], d['disch'][:, None], d['stor'], d['dact'],
            d['dloc'], d['deficit'][:, None]]
    return np.concatenate([np.asarray(c, float) for c in cols], 1)


def run_oracle(exe, d, tmp_path):
    rec = pack(d)
    fi, fo = str(tmp_path / 'in.bin'), str(tmp_path / 'out.bin')
    with open(fi, 'wb') as f:
        f.write(struct.pack('<d', float(len(rec))))
        f.write(rec.astype('<f8').tobytes())
    subprocess.run([exe, fi, fo], check=True)
    o = np.fromfile(fo, '<f8').reshape(len(rec), -1)
    names = [('dgw', NS), ('dsf', NS), ('A_gw', 1), ('A_sf', 1), ('A_dam', 1),
             ('Dgw', 1), ('Dsf', 1), ('_', 1), ('avgw', NV * NB),
             ('avdam', ND), ('availgw', NS), ('availsf', NS), ('availdm', NS),
             ('wgw', NS), ('wsf', NS), ('wdm', NS), ('ret', NS), ('con', NS),
             ('Wgw', 1), ('Wsf', 1), ('Wdm', 1), ('R', 1), ('moist', NV * NB),
             ('stor', ND), ('deficit', 1), ('_2', 1), ('unmet', NS)]
    out, k = {}, 0
    for nm, w in names:
        out[nm] = o[:, k:k + w]
        k += w
    assert k == o.shape[1]
    return out


def run_torch(d, device='cpu'):
    n = len(d['efr'])
    t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt,
                                                    device=device)
    b = lambda a: t(a, torch.bool)
    efr = d['efr'] > 0
    gwm = d['gwm'] > 0
    # the kernels take one global efr/gwm switch: run the two switch
    # settings over their subsets and merge
    res = None
    for e in (False, True):
        for gm in (False, True):
            sel = (efr == e) & (gwm == gm)
            if not sel.any():
                continue
            ix = np.nonzero(sel)[0]
            s = lambda a: np.asarray(a)[ix]
            r = W.wu_local_step(
                t(s(d['dem'])), t(s(d['gfr'])), t(s(d['cf'])),
                t(s(d['moist'])), t(s(d['ice'])), t(s(d['ff'])),
                t(s(d['resid'])), t(s(d['AF'])), t(s(d['Cv'])),
                t(s(d['dtq'])), 86400.0, t(s(d['area'])),
                dam_storage=t(s(d['stor'])), dam_active=b(s(d['dact']) > 0),
                dam_local=b(s(d['dloc']) > 0),
                efr_moist=t(s(d['efrm'])) if e else None,
                efr_q=t(s(d['efr_q'])) if e else None,
                discharge=t(s(d['disch'])) if e else None,
                active=b(s(d['act']) > 0), deficit=t(s(d['deficit'])),
                returned_in=t(s(d['ret0'])), consumed_in=t(s(d['con0'])),
                gwm=gm)
            r = {k: (v.detach().cpu().numpy() if v is not None else None)
                 for k, v in r.items()}
            if res is None:
                res = {k: np.zeros((n,) + v.shape[1:]) for k, v in r.items()
                       if v is not None}
            for k, v in r.items():
                if v is not None:
                    res[k][ix] = v
    return res


def compare(o, r, d):
    tol = dict(rtol=1e-12, atol=1e-12)
    eq = lambda a, b, what: np.testing.assert_allclose(
        a, b, err_msg=what, **tol)
    eq(r['demand_gw'], o['dgw'], 'demand_gw')
    eq(r['demand_surf'], o['dsf'], 'demand_surf')
    eq(r['A_gw'], o['A_gw'][:, 0], 'A_gw')
    eq(r['A_surf'], o['A_sf'][:, 0], 'A_surf')
    eq(r['A_dam'], o['A_dam'][:, 0], 'A_dam')
    eq(r['av_gw'].reshape(len(o['dgw']), -1), o['avgw'], 'av_gw')
    eq(r['available_gw'], o['availgw'], 'available_gw')
    eq(r['available_surf'], o['availsf'], 'available_surf')
    eq(r['available_dam'], o['availdm'], 'available_dam')
    eq(r['withdrawn_gw'], o['wgw'], 'withdrawn_gw')
    eq(r['withdrawn_surf'], o['wsf'], 'withdrawn_surf')
    eq(r['withdrawn_dam'], o['wdm'], 'withdrawn_dam')
    eq(r['returned'], o['ret'], 'returned')
    eq(r['consumed'], o['con'], 'consumed')
    eq(r['cell_gw'], o['Wgw'][:, 0], 'cell_gw')
    eq(r['cell_surf'], o['Wsf'][:, 0], 'cell_surf')
    eq(r['cell_dam'], o['Wdm'][:, 0], 'cell_dam')
    eq(r['returned_to_river'], o['R'][:, 0], 'returned after payback')
    eq(r['unmet'], o['unmet'], 'unmet')
    eq(r['deficit'], o['deficit'][:, 0], 'deficit')
    eq(r['moist'].reshape(len(o['dgw']), -1), o['moist'], 'moist')
    eq(r['dam_storage'], o['stor'], 'dam storage')


def test_l1_random_edge(oracle, tmp_path):
    d = make_cases(6000, 1)
    o = run_oracle(oracle, d, tmp_path)
    compare(o, run_torch(d), d)
    # edge coverage actually reached
    assert (o['A_gw'] == 0).any() and (o['A_sf'] == 0).any()
    assert ((o['Dsf'] > o['A_sf'] + o['A_dam']) & (o['A_dam'] > 0)).any()
    assert (o['wdm'] > 0).any() and (o['wgw'] > 0).any()


def test_l1_other_seed(oracle, tmp_path):
    d = make_cases(3000, 2)
    compare(run_oracle(oracle, d, tmp_path), run_torch(d), d)


def test_hand_cases():
    t = lambda *a: torch.tensor([list(a)], dtype=torch.float64)
    z = torch.zeros(1, dtype=torch.float64)
    c = lambda v: torch.tensor([v], dtype=torch.float64)
    # gw priority: demands 3, 2 with A_gw=4 -> 3, 1
    ag, _, _ = W.division(t(3, 2, 0, 0, 0), t(0., 0, 0, 0, 0), c(4.), z, z,
                          c(5.), z)
    assert ag[0, :2].tolist() == [3.0, 1.0]
    # surface split: demand 6, A_sf=3, A_dam=1 -> 3/1 ; demand 2 -> 1.5/.5
    _, s, m = W.division(t(0., 0, 0, 0, 0), t(6, 0, 0, 0, 0), z, c(3.),
                         c(1.), z, c(6.))
    assert (s[0, 0].item(), m[0, 0].item()) == (3.0, 1.0)
    _, s, m = W.division(t(0., 0, 0, 0, 0), t(2, 0, 0, 0, 0), z, c(3.),
                         c(1.), z, c(2.))
    assert (s[0, 0].item(), m[0, 0].item()) == (1.5, 0.5)
    # EFR with discharge 0 gives A_sf = 0 (macro)
    a = W.availability_surf(torch.ones(1, 4, dtype=torch.float64), 86400.,
                            c(1e8), c(5.), c(0.))
    assert a.item() == 0.0
    # payback: deficit 1 with return 3 leaves 2 for the river
    d, r = W.payback(c(1.), t(3, 0, 0, 0, 0), c(3.))
    assert (d.item(), r.item()) == (0.0, 2.0)


def _read_wu(path, maxrec=None):
    out = []
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            tag, n = struct.unpack('<iq', h)
            if tag == 5:
                out.append(np.frombuffer(f.read(8 * n), '<f8').copy())
                if maxrec and len(out) >= maxrec:
                    break
            else:
                f.seek(8 * n, 1)
    return out


@pytest.mark.skipif(not os.path.exists(DUMP), reason='R1 G10 dump absent')
def test_l2_g10_wu_var():
    """wu_var of the dumped G10 run (6 cells, 5 sectors): the use rule
    reproduces withdrawn_*; the division reproduces available_* when the
    pools are exhausted (pool = sum of the sector shares) or the demand is
    met (share = demand)."""
    recs = _read_wu(DUMP)
    assert len(recs) > 1000
    x = torch.as_tensor(np.stack(recs), dtype=torch.float64)    # [T,6*5*22]
    x = x.reshape(len(recs) * 6, NS, 22)
    act = ~torch.isnan(x[..., 0])
    x = torch.nan_to_num(x)
    dgw, dsf = x[..., 6], x[..., 7]
    ag, asf, adm = x[..., 0], x[..., 1], x[..., 2]
    u = W.use(dgw, dsf, ag, asf, adm, torch.full_like(ag, 0.5), active=act)
    for k, j in (('withdrawn_gw', 11), ('withdrawn_surf', 12),
                 ('withdrawn_dam', 13)):
        torch.testing.assert_close(u[k], x[..., j] * act, rtol=1e-12,
                                   atol=1e-12)
    assert (x[..., 11] > 0).any() and (x[..., 12] > 0).any()
    # division
    tg, ts = W.cell_demand(dgw, dsf, act)
    sg = (ag * act).sum(-1)
    s_s = (asf * act).sum(-1)
    s_d = (adm * act).sum(-1)
    dg2, ds2 = (dgw * act).sum(-1), (dsf * act).sum(-1)
    # a pool is exhausted when its shares sum to less than the demand
    gw_short = sg < dg2 * (1 - 1e-12)
    sf_short = (s_s + s_d) < ds2 * (1 - 1e-12)
    ng, nsf, nd = W.division(dgw, dsf, sg, s_s, s_d, tg, ts, act)
    sel = gw_short
    torch.testing.assert_close(ng[sel], (ag * act)[sel], rtol=1e-12,
                               atol=1e-12)
    sel = sf_short
    torch.testing.assert_close(nsf[sel], (asf * act)[sel], rtol=1e-12,
                               atol=1e-12)
    torch.testing.assert_close(nd[sel], (adm * act)[sel], rtol=1e-12,
                               atol=1e-12)
    # demand met: shares = demand
    met = ~sf_short
    torch.testing.assert_close(nsf[met] + nd[met], (dsf * act)[met],
                               rtol=1e-12, atol=1e-12)
    print('G10 cell-steps', x.shape[0], 'gw short', int(gw_short.sum()),
          'sf short', int(sf_short.sum()))


@pytest.mark.gpu
def test_gpu(oracle, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    d = make_cases(4000, 3)
    o = run_oracle(oracle, d, tmp_path)
    compare(o, run_torch(d, 'cuda'), d)
