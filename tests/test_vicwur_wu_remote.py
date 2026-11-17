# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.wateruse_remote (remote water use).

Oracle: tests/oracle/ref_vicwur_wu_remote.c compiles the unchanged VIC-WUR 3.0
wu_remote.c and calls wu_remote(i) for the cells of a routing order on random
graphs; the batched float64 kernels (edge list, sector priority, availability
sharing, deficit payback, ring edit) must match every wu_var field, the
deficit and the ring to 1e-12.  Edge cases: no receivers, one source shared
by many receivers, sectors competing for a scarce ring, a receiver with two
sources, cells that are source and receiver, empty rings, EFR with 0/0,
GWM, unequal areas.
Replay of the R1 dump runs G6a, G6b, G6c, G9 and G10 (remote water use on
cells 2->4 and 3->5, built with VICWUR_DUMP): demand, division, use and the
receiver bookkeeping reproduce wu_var of every step to 1e-9, and the deficit
chain of EVERY cell (incl. the receivers and sources) reproduces
the dumped deficit from the previous deficit, VIC baseflow and runoff and the
remote payback (float32 history output: 1e-8 mm without, 1e-6 mm with NONRENEWABLE_RUNOFF).
"""
import glob
import os
import _paths  # noqa: E402
import struct
import subprocess

import numpy as np
import pytest
import torch

from torchvic.io import ncshare
from torchvic.vicwur import nonrenew as NR
from torchvic.vicwur import wateruse as W
from torchvic.vicwur import wateruse_remote as R

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'oracle', 'ref_vicwur_wu_remote.c')
SHIMS = [os.path.join(HERE, 'oracle', 'vicwur_shim_rm'),
         os.path.join(HERE, 'oracle', 'vicwur_shim')]
W3 = _paths.env_path(('TORCHVIC_VICWUR_SRC', 'VICWUR_SRC'), 'build/vicwur3/src_tree/vic')
VER = _paths.env_path(('TORCHVIC_VICWUR_VERIFY',), 'verification')
R1 = _paths.env_path(('TORCHVIC_VICWUR_DIR',), 'vicwur') + '/R1'
NS = 5
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
# wu_var fields written by the oracle (wateruse.h:67-96, 24 per sector)
FIELDS = ['available_gw', 'available_surf', 'available_dam',
          'available_tremote', 'available_remote', 'available_nonrenew',
          'demand_gw', 'demand_surf', 'demand_tremote', 'demand_remote',
          'demand_nonrenew', 'withdrawn_gw', 'withdrawn_surf',
          'withdrawn_dam', 'withdrawn_tremote', 'withdrawn_remote',
          'withdrawn_nonrenew', 'returned', 'consumed',
          'available_tmp', 'demand_tmp', 'withdrawn_tmp', 'returned_tmp',
          'consumed_tmp']
STATE = [f for f in FIELDS if f in R.RemoteState.TOTALS + R.RemoteState.TMPS]
HAND_OK = 1e-12


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    src = os.path.join(W3, 'plugins', 'wateruse', 'src', 'wu_remote.c')
    if not os.path.exists(src):
        pytest.skip('W3 source tree absent: ' + src)
    p = os.path.join(W3, 'plugins')
    exe = str(tmp_path_factory.mktemp('wr') / 'ref_wu_remote')
    cmd = ['cc', '-o', exe, SRC] + ['-I' + d for d in SHIMS] + [
        '-I' + p + '/routing/include', '-I' + p + '/wateruse/include',
        '-I' + p + '/efr/include', '-I' + p + '/wateruse/src', '-O0',
        '-std=c99', '-ffp-contract=off', '-D_DEFAULT_SOURCE',
        '-Wno-absolute-value', '-lm']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('cannot build oracle: ' + r.stderr[:300])
    return exe


# ---------------------------------------------------------------- graphs
def make_graph(g, n, kind='random', nsub=1, uh=5, efr=False, gwm=False):
    """One random network and state.  `order` is a routing order in which
    every receiver comes before its sources (wu_set_routing_order)."""
    d = dict(n=n, nsub=nsub, uh=uh, efr=efr, gwm=gwm, dt=86400.0)
    order = g.permutation(n)
    pos = np.empty(n, int)
    pos[order] = np.arange(n)
    edges = []                      # (src, dst), dst earlier in the order
    if kind == 'none':
        pass
    elif kind == 'star':            # one source, many receivers
        s = order[-1]
        edges = [(s, int(c)) for c in order[:-1]]
    elif kind == 'two_sources':     # receivers served by two sources
        r = order[:3]
        for s in order[-2:]:
            edges += [(int(s), int(c)) for c in r]
    elif kind == 'chain':           # source of one is receiver of the next
        for a, b in zip(order[:-1], order[1:]):
            edges.append((int(b), int(a)))
    else:
        for s in order[1:]:
            k = g.integers(0, min(pos[s], 5) + 1)
            if g.random() < 0.3:
                k = 0
            for c in g.choice(order[:pos[s]], size=k, replace=False):
                edges.append((int(s), int(c)))
    nk = max([sum(1 for e in edges if e[0] == c) for c in range(n)] + [1])
    rec = -np.ones((n, nk), int)
    for c in range(n):
        r = [e[1] for e in edges if e[0] == c]
        g.shuffle(r)
        rec[c, :len(r)] = r
    d['rec'], d['order'], d['nk'] = rec, order, nk
    d['edges'] = [(c, int(x)) for c in range(n) for x in rec[c] if x >= 0]

    def u(*shape, lo=0.0, hi=1.0):
        return g.uniform(lo, hi, shape)

    def zero(a, p):
        return np.where(g.random(a.shape) < p, 0.0, a)

    full = g.random() < 0.5         # all sectors everywhere, as in VIC-WUR
    d['act'] = (np.ones((n, NS)) if full else
                (g.random((n, NS)) < 0.75).astype(float))
    d['dgw'] = zero(u(n, NS, hi=6), 0.3) * d['act']
    d['dsf'] = zero(u(n, NS, hi=12), 0.2) * d['act']
    for k in ('wgw', 'wsf', 'wdm'):       # local use took part of the demand
        d[k] = u(n, NS) * (d['dgw'] if k == 'wgw' else d['dsf']) / 2 \
            * (g.random((n, NS)) < 0.6)
    d['wtr0'] = np.zeros((n, NS))
    d['cf'] = np.where(g.random((n, NS)) < 0.15,
                       g.integers(0, 2, (n, NS)).astype(float), u(n, NS))
    d['ret0'] = zero(u(n, NS, hi=2), 0.4)
    d['con0'] = zero(u(n, NS, hi=2), 0.4)
    d['area'] = u(n, lo=5e7, hi=5e8)
    if g.random() < 0.3:
        d['area'][:] = 2e8                              # equal areas
    d['deficit'] = zero(u(n, lo=0, hi=15), 0.4)
    R_ = uh + nsub - 1
    ring = zero(u(n, R_, lo=0, hi=400), 0.3)
    ring *= 10.0 ** g.uniform(-2.5, 1.0, n)[:, None]    # scarce to plentiful
    ring[g.random(n) < 0.12] = 0.0                      # empty ring
    d['ring'] = ring
    d['discharge'] = u(n, lo=0, hi=300)
    d['efr_q'] = zero(u(n, lo=0, hi=300), 0.2)
    d['discharge'][g.random(n) < 0.1] = 0.0
    return d


def pack(d):
    n = d['n']
    head = [n, d['nk'], d['uh'], d['nsub'], float(d['efr']), float(d['gwm']),
            d['dt'], len(d['order'])]
    rows = []
    for c in range(n):
        nrec = int((d['rec'][c] >= 0).sum())
        rows.append(np.concatenate([
            [d['area'][c], d['deficit'][c], d['efr_q'][c], d['discharge'][c],
             nrec], np.maximum(d['rec'][c], 0), d['act'][c], d['dgw'][c],
            d['dsf'][c], d['wgw'][c], d['wsf'][c], d['wdm'][c], d['wtr0'][c],
            d['cf'][c], d['ret0'][c], d['con0'][c], d['ring'][c]]))
    return np.concatenate([np.asarray(head, float), d['order'].astype(float),
                           np.concatenate(rows)])


def run_oracle(exe, d, tmp_path):
    fi, fo = str(tmp_path / 'in.bin'), str(tmp_path / 'out.bin')
    pack(d).astype('<f8').tofile(fi)
    r = subprocess.run([exe, fi, fo], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    o = np.fromfile(fo, '<f8')
    k = d['uh'] + d['nsub'] - 1
    o = o.reshape(d['n'], NS * 24 + 3 + k)
    out = {f: o[:, :NS * 24].reshape(d['n'], NS, 24)[:, :, i]
           for i, f in enumerate(FIELDS)}
    out['deficit'] = o[:, NS * 24]
    out['discharge'] = o[:, NS * 24 + 1]
    out['stream'] = o[:, NS * 24 + 2]
    out['ring'] = o[:, NS * 24 + 3:]
    return out


def run_torch(d, device='cpu', order=True):
    n = d['n']
    t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt,
                                                    device=device)
    act = t(d['act'] > 0, torch.bool)
    loc = R.LocalUse(t(d['dgw']), t(d['dsf']), t(d['wgw']), t(d['wsf']),
                     t(d['wdm']), t(d['cf']), act)
    st = R.RemoteState(n, device)
    st.returned, st.consumed = t(d['ret0']).clone(), t(d['con0']).clone()
    st.withdrawn_tremote = t(d['wtr0']).clone()
    deficit, ring = t(d['deficit']).clone(), t(d['ring']).clone()
    disch, stream = t(d['discharge']).clone(), torch.zeros(n, dtype=torch.float64,
                                                           device=device)
    area, efr_q = t(d['area']), t(d['efr_q'])
    src = [e[0] for e in d['edges']]
    dst = [e[1] for e in d['edges']]
    plan = R.RemotePlan(src, dst, n, order=d['order'] if order else None,
                        device=device)
    flags = []
    for b in plan.batches:
        o = R.remote_batch(b, st, loc, deficit, ring[b.src], area, d['dt'],
                           d['nsub'], d['uh'], disch[b.src], stream[b.src],
                           efr_q[b.src] if d['efr'] else None, d['gwm'])
        deficit = o['deficit']
        ring[b.src] = o['ring']
        disch[b.src] = o['discharge']
        stream[b.src] = o['stream']
        flags.append(o['violation'])
    res = {f: getattr(st, f) for f in STATE}
    res.update(deficit=deficit, ring=ring, discharge=disch, stream=stream)
    # fields the remote pass does not touch keep their local values
    res['demand_gw'], res['demand_surf'] = loc.dgw, loc.dsf
    res['withdrawn_gw'], res['withdrawn_surf'] = loc.wgw, loc.wsf
    res['withdrawn_dam'] = loc.wdm
    for f in ('available_gw', 'available_surf', 'available_dam',
              'available_nonrenew', 'demand_nonrenew', 'withdrawn_nonrenew'):
        res[f] = torch.zeros_like(loc.dgw)
    return res, plan, flags


def compare(o, r, d, tol=1e-12):
    act = d['act'] > 0
    for f in FIELDS:
        a = np.asarray(r[f].cpu())
        b = o[f]
        np.testing.assert_allclose(np.where(act, a, 0), np.where(act, b, 0),
                                   rtol=tol, atol=tol, err_msg=f)
    for f in ('deficit', 'discharge', 'stream', 'ring'):
        np.testing.assert_allclose(np.asarray(r[f].cpu()), o[f], rtol=tol,
                                   atol=tol, err_msg=f)


CONFIGS = [dict(efr=False, gwm=False, nsub=1, uh=5),
           dict(efr=True, gwm=False, nsub=2, uh=6),
           dict(efr=False, gwm=True, nsub=1, uh=4),
           dict(efr=True, gwm=True, nsub=4, uh=7)]
KINDS = ['random', 'random', 'random', 'star', 'two_sources', 'chain',
         'none']


def cases(seed, per=6, n=14):
    g = np.random.default_rng(seed)
    for cfg in CONFIGS:
        for i in range(per):
            kind = KINDS[i % len(KINDS)]
            yield make_graph(g, n + (i % 3) * 7, kind, **cfg)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_l1_random_graphs(oracle, tmp_path, seed):
    used = 0
    for d in cases(seed, per=7):
        o = run_oracle(oracle, d, tmp_path)
        r, plan, flags = run_torch(d)
        compare(o, r, d)
        used += plan.n_edges
        # without a balance error C would have stopped: no violation flag
        assert not any(bool(f.any()) for f in flags)
    assert used > 200


def test_l1_work_was_done(oracle, tmp_path):
    """The comparison is not vacuous: water moves, payback and the ring edit
    happen, the pool is exhausted in some graphs and not in others."""
    g = np.random.default_rng(11)
    tot = dict(w=0, paid=0, short=0, ample=0, ring_changed=0)
    for _ in range(30):
        d = make_graph(g, 20, 'random', nsub=2, uh=5)
        r, plan, _ = run_torch(d)
        w = r['withdrawn_remote'].sum().item()
        tot['w'] += w > 0
        tot['paid'] += (r['deficit'] < torch.as_tensor(d['deficit'])
                        - 1e-9).any().item()
        a = r['available_remote'].sum(-1)
        dm = r['demand_remote'].sum(-1)
        tot['short'] += ((a < dm - 1e-9) & (dm > 0)).any().item()
        tot['ample'] += ((a >= dm - 1e-9) & (dm > 0)).any().item()
        tot['ring_changed'] += (r['ring'] - torch.as_tensor(d['ring'])
                                ).abs().max().item() > 1e-9
    assert all(v >= 10 for v in tot.values()), tot


def test_no_receivers(oracle, tmp_path):
    d = make_graph(np.random.default_rng(5), 8, 'none')
    r, plan, flags = run_torch(d)
    assert plan.batches == [] and plan.n_edges == 0
    compare(run_oracle(oracle, d, tmp_path), r, d)
    assert r['ring'].numpy().tolist() == d['ring'].tolist()


def test_one_source_many_receivers_scarce(oracle, tmp_path):
    """All receivers draw on one ring; sector 0 takes the pool first, the
    later sectors get nothing (priority across receivers and sectors)."""
    n = 13
    g = np.random.default_rng(9)
    d = make_graph(g, n, 'star', nsub=1, uh=6)
    s = int(d['order'][-1])
    d['act'][:] = 1
    d['dgw'][:] = 0
    d['dsf'][:] = g.uniform(1, 3, (n, NS))
    d['wgw'][:] = d['wsf'][:] = d['wdm'][:] = 0
    d['cf'][:] = 0.4
    d['area'][:] = 8.64e7                  # 1 m3/s over dt = 1 mm
    d['ring'][:] = 0
    d['ring'][s] = [2.0, 1, 0, 0, 0, 0]    # pool 3 mm
    d['efr'] = False
    o = run_oracle(oracle, d, tmp_path)
    r, plan, flags = run_torch(d)
    compare(o, r, d)
    rc = [c for c in range(n) if c != s]
    wr = np.asarray(r['withdrawn_tremote'])
    assert wr[rc, 0].sum() == pytest.approx(3.0, abs=1e-12)   # MUN takes all
    assert np.abs(wr[rc, 1:]).max() == 0.0
    # pro rata inside the sector: share = demand * 3 / sum(demand)
    dem = d['dsf'][rc, 0]
    np.testing.assert_allclose(wr[rc, 0], dem * 3.0 / dem.sum(), rtol=1e-12)
    assert len(plan.batches) == 1 and plan.batches[0].n_src == 1


def test_sectors_compete_two_receivers(oracle, tmp_path):
    """Hand case: ring 4 mm; MUN demands 3 + 2 take 4 (frac .8), ENE gets
    nothing; with ring 8 mm MUN is fully served (5) and ENE shares 3 of 6."""
    n = 3
    d = make_graph(np.random.default_rng(2), n, 'none', nsub=1, uh=3)
    d['rec'][:] = -1
    d['rec'][2, :1] = 0
    d['nk'] = 2
    d['rec'] = np.full((n, 2), -1)
    d['rec'][2] = [0, 1]
    d['order'] = np.array([0, 1, 2])
    d['edges'] = [(2, 0), (2, 1)]
    for k in ('dgw', 'wgw', 'wsf', 'wdm', 'ret0', 'con0', 'deficit'):
        d[k] = np.zeros_like(d[k])
    d['act'][:] = 1
    d['dsf'][:] = 0
    d['dsf'][0, :2] = [3, 2]
    d['dsf'][1, :2] = [2, 4]
    d['cf'][:] = 0.5
    d['area'][:] = 8.64e7
    d['efr'] = False
    for ring, want in (([4.0, 0, 0], [[2.4, 0], [1.6, 0]]),
                       ([6.0, 2, 0], [[3, 1], [2, 2]])):
        d['ring'] = np.zeros((n, 3))
        d['ring'][2] = ring
        r, plan, _ = run_torch(d)
        compare(run_oracle(oracle, d, tmp_path), r, d)
        wt = np.asarray(r['withdrawn_tremote'])
        np.testing.assert_allclose(wt[:2, :2], want, atol=1e-12)
        # source ring: lost the withdrawal, regained the return (cf .5)
        tot = sum(ring)
        got = r['ring'][2].sum().item()
        taken = np.sum(want)
        assert got == pytest.approx(tot - taken + 0.5 * taken, abs=1e-12)


def test_receiver_with_two_sources(oracle, tmp_path):
    """A receiver served by two sources: the second sees the demand left by
    the first (withdrawn_tremote enters the unmet demand), demand_tremote
    is the maximum, available and withdrawn add up; they run in two
    batches in routing order."""
    n = 3
    d = make_graph(np.random.default_rng(4), n, 'none', nsub=1, uh=3)
    d['rec'] = np.array([[-1], [0], [0]])
    d['nk'] = 1
    d['order'] = np.array([0, 1, 2])
    d['edges'] = [(1, 0), (2, 0)]
    for k in ('dgw', 'wgw', 'wsf', 'wdm', 'ret0', 'con0', 'deficit'):
        d[k] = np.zeros_like(d[k])
    d['act'][:] = 1
    d['dsf'][:] = 0
    d['dsf'][0, 0] = 4.0
    d['cf'][:] = 0.25
    d['area'][:] = 8.64e7
    d['efr'] = False
    d['ring'] = np.zeros((n, 3))
    d['ring'][1] = [1.5, 0, 0]
    d['ring'][2] = [5.0, 0, 0]
    r, plan, _ = run_torch(d)
    compare(run_oracle(oracle, d, tmp_path), r, d)
    assert len(plan.batches) == 2
    assert r['demand_tmp'][0, 0].item() == pytest.approx(4 - 1.5)  # last one
    assert r['withdrawn_tremote'][0, 0].item() == pytest.approx(4.0)
    assert r['available_tremote'][0, 0].item() == pytest.approx(4.0)
    assert r['demand_tremote'][0, 0].item() == pytest.approx(4.0)
    assert r['consumed'][0, 0].item() == pytest.approx(1.0)
    assert r['withdrawn_remote'][1, 0].item() == pytest.approx(1.5)
    assert r['withdrawn_remote'][2, 0].item() == pytest.approx(2.5)


def test_unequal_areas_use_the_source_area():
    """The receiver demand is mm over the receiver area, the ring edit
    converts it with the SOURCE area: a source of area 2e8 and a receiver of
    area 5e7 move W mm * 2e8 / dt, whatever the receiver area."""
    n = 2
    d = make_graph(np.random.default_rng(1), n, 'none', nsub=1, uh=3)
    d['rec'] = np.array([[-1], [0]])
    d['nk'] = 1
    d['order'] = np.array([0, 1])
    d['edges'] = [(1, 0)]
    for k in ('dgw', 'wgw', 'wsf', 'wdm', 'ret0', 'con0', 'deficit'):
        d[k] = np.zeros_like(d[k])
    d['act'][:] = 1
    d['dsf'][:] = 0
    d['dsf'][0, 0] = 2.0
    d['cf'][:] = 1.0                                   # nothing returned
    d['efr'] = False
    d['ring'] = np.zeros((n, 3))
    d['ring'][1] = [100.0, 50.0, 50.0]                 # m3/s
    for a0 in (5e7, 4e8):
        d['area'] = np.array([a0, 2e8])
        r, plan, _ = run_torch(d)
        # ring availability = 200 * 86400 / 2e8 * 1000 = 86.4 mm >> 2 mm
        assert r['withdrawn_remote'][1, 0].item() == pytest.approx(2.0)
        q = 2.0 / 1000 * 2e8 / 86400
        assert r['ring'][1].sum().item() == pytest.approx(200 - q, rel=1e-12)


def test_deficit_payback_moves_returned():
    """The receiver's deficit is repaid first from the remote return; the
    `returned` that did so moves from the source to the receiver."""
    n = 2
    d = make_graph(np.random.default_rng(1), n, 'none', nsub=1, uh=3)
    d['rec'] = np.array([[-1], [0]])
    d['nk'] = 1
    d['order'] = np.array([0, 1])
    d['edges'] = [(1, 0)]
    for k in ('dgw', 'wgw', 'wsf', 'wdm', 'ret0', 'con0'):
        d[k] = np.zeros_like(d[k])
    d['act'][:] = 1
    d['dsf'][:] = 0
    d['dsf'][0, 0] = 4.0
    d['cf'][:] = 0.5
    d['area'][:] = 8.64e7
    d['efr'] = False
    d['ring'] = np.zeros((n, 3))
    d['ring'][1] = [10.0, 0, 0]
    d['deficit'] = np.array([1.5, 0.0])
    r, plan, _ = run_torch(d)
    # W = 4, returned = 2; the deficit 1.5 is repaid
    assert r['deficit'][0].item() == pytest.approx(0.0, abs=1e-15)
    assert r['returned'][0, 0].item() == pytest.approx(1.5)
    assert r['returned'][1, 0].item() == pytest.approx(0.5)    # 2 - 1.5
    # only 0.5 mm of return reaches the source ring: 10 - 4 + 0.5
    assert r['ring'][1].sum().item() == pytest.approx(6.5, abs=1e-12)


# ---------------------------------------------------------------- plan
def test_plan_batches_serial_equivalence():
    """Batches never share a cell and keep the serial order of conflicting
    sources; invalid plans are refused."""
    g = np.random.default_rng(3)
    for _ in range(20):
        d = make_graph(g, 25, 'random')
        src = [e[0] for e in d['edges']]
        dst = [e[1] for e in d['edges']]
        plan = R.RemotePlan(src, dst, 25, order=d['order'])
        rank = np.empty(25, int)
        rank[d['order']] = np.arange(25)
        seen = set()
        for b in plan.batches:
            cells = set()
            s_ = b.src.tolist()
            for s in s_:
                t = {s} | set(plan.dst[plan.src == s].tolist())
                assert not (cells & t)
                cells |= t
        # conflicting sources keep the serial order
        for i, b1 in enumerate(plan.batches):
            for b2 in plan.batches[i + 1:]:
                for s1 in b1.src.tolist():
                    for s2 in b2.src.tolist():
                        t1 = {s1} | set(plan.dst[plan.src == s1].tolist())
                        t2 = {s2} | set(plan.dst[plan.src == s2].tolist())
                        if t1 & t2:
                            assert rank[s1] < rank[s2]
    with pytest.raises(ValueError):
        R.RemotePlan([0], [0], 3)                         # self edge
    with pytest.raises(ValueError):
        R.RemotePlan([0, 0], [1, 1], 3)                   # duplicate
    with pytest.raises(ValueError):
        R.RemotePlan([0, 1], [2, 2], 3, batch_of=[0, 0, 0])  # shared receiver
    src, dst = R.edges_from_receiving(np.array([[-1, -1], [2, 0], [-1, -1]]))
    assert src.tolist() == [1, 1] and dst.tolist() == [2, 0]


# -------------------------------------------------------------- devices
@pytest.mark.parametrize('device', DEVICES)
def test_devices_agree(oracle, tmp_path, device):
    for d in cases(21, per=4):
        compare(run_oracle(oracle, d, tmp_path), run_torch(d, device)[0], d)


@pytest.mark.gpu
def test_gpu(oracle, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    for d in cases(33, per=7, n=30):
        compare(run_oracle(oracle, d, tmp_path), run_torch(d, 'cuda')[0], d)


@pytest.mark.gpu
def test_gpu_no_host_sync_and_deterministic():
    """The batches run with torch.cuda.set_sync_debug_mode('error') and under
    torch.use_deterministic_algorithms; two runs agree bit for bit."""
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    g = np.random.default_rng(8)
    d = make_graph(g, 200, 'random', nsub=2, uh=6, efr=True)
    run_torch(d, 'cuda')                      # warm up (kernel loading)
    torch.cuda.synchronize()
    outs = []
    torch.use_deterministic_algorithms(True)
    try:
        for _ in range(2):
            n = d['n']
            t = lambda a, dt=torch.float64: torch.as_tensor(
                np.asarray(a), dtype=dt, device='cuda')
            loc = R.LocalUse(t(d['dgw']), t(d['dsf']), t(d['wgw']),
                             t(d['wsf']), t(d['wdm']), t(d['cf']),
                             t(d['act'] > 0, torch.bool))
            st = R.RemoteState(n, 'cuda')
            st.returned, st.consumed = t(d['ret0']), t(d['con0'])
            deficit, ring = t(d['deficit']), t(d['ring'])
            disch = t(d['discharge'])
            stream = torch.zeros(n, dtype=torch.float64, device='cuda')
            area, efr_q = t(d['area']), t(d['efr_q'])
            plan = R.RemotePlan([e[0] for e in d['edges']],
                                [e[1] for e in d['edges']], n,
                                order=d['order'], device='cuda')
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode('error')
            try:
                for b in plan.batches:
                    o = R.remote_batch(b, st, loc, deficit, ring[b.src], area,
                                       d['dt'], d['nsub'], d['uh'],
                                       disch[b.src], stream[b.src],
                                       efr_q[b.src], False)
                    deficit = o['deficit']
                    ring[b.src] = o['ring']
                    disch[b.src] = o['discharge']
                    stream[b.src] = o['stream']
            finally:
                torch.cuda.set_sync_debug_mode('default')
            torch.cuda.synchronize()
            outs.append((deficit.cpu(), ring.cpu(), st.returned.cpu(),
                         st.withdrawn_tremote.cpu()))
    finally:
        torch.use_deterministic_algorithms(False)
    assert len(plan.batches) > 1
    for a, b in zip(*outs):
        assert a.equal(b)
    assert torch.isfinite(outs[0][1]).all()


# -------------------------------------------------------- router adapter
def _router(n_cells, receiving, device='cpu'):
    from torchvic.model.routing_cascade import CascadeRouter
    L = 5
    down = np.arange(n_cells)
    down[:-1] = np.arange(1, n_cells)
    uh = np.zeros((L, n_cells))
    uh[0] = 1.0
    return CascadeRouter(down, uh, uh, np.full(n_cells, 2e8), 86400.0, 2,
                         receiving=receiving, device=device)


@pytest.mark.parametrize('device', DEVICES)
def test_router_slot_matches_oracle(oracle, tmp_path, device):
    """RemoteSlot on a CascadeRouter (sweep order, groups from its
    schedule, ring with head) equals the C order on the same state."""
    n = 9
    g = np.random.default_rng(17)
    d = make_graph(g, n, 'random', nsub=2, uh=5, efr=True)
    # the routing order the router builds must put receivers first: use the
    # router's own order for the oracle
    receiving = [[int(x) for x in d['rec'][c] if x >= 0] for c in range(n)]
    # chain network 0 -> 1 -> ... -> 8: every cell is upstream of the next,
    # so receivers (earlier cells) must have smaller index
    edges = [(s, r) for s in range(n) for r in receiving[s] if r < s]
    receiving = [[r for (s, r) in edges if s == c] for c in range(n)]
    assert len(edges) >= 5
    rt = _router(n, receiving, device)
    d['edges'] = edges
    nk = max([len(r) for r in receiving] + [1])
    d['rec'] = -np.ones((n, nk), int)
    for c, r in enumerate(receiving):
        d['rec'][c, :len(r)] = r
    d['nk'] = nk
    d['order'] = rt.schedule.order
    d['area'][:] = 2e8
    d['gwm'] = False
    d['deficit'] = g.uniform(1, 15, n)
    t = lambda a, dt=torch.float64: torch.as_tensor(
        np.asarray(a), dtype=dt, device=device)
    rt.load_state(t(d['ring']), t(d['deficit']))
    rt.discharge.copy_(rt.to_sweep(t(d['ring'][:, :2].sum(1))))
    d['discharge'] = d['ring'][:, :2].sum(1)
    d['efr_q'] = d['discharge'] * g.uniform(0, 0.6, n)     # EFR takes part
    rt.stream.zero_()
    sw = lambda a: rt.to_sweep(t(a))
    perm = rt.perm
    loc = R.LocalUse(*[t(d[k])[perm] for k in ('dgw', 'dsf', 'wgw', 'wsf',
                                                'wdm', 'cf')],
                     t(d['act'] > 0, torch.bool)[perm])
    st = R.RemoteState(n, device)
    st.returned, st.consumed = t(d['ret0'])[perm], t(d['con0'])[perm]
    efr = sw(d['efr_q'])
    plan = R.plan_for_router(rt, [e[0] for e in edges], [e[1] for e in edges])
    slot = R.RemoteSlot(plan, rt, st, loc, efr_q=efr)
    slot.begin()
    for gi in range(len(rt._gptr) - 1):
        slot(rt, rt.group_cells(gi))
    o = run_oracle(oracle, d, tmp_path)
    assert np.abs(o['ring'] - d['ring']).max() > 1e-6          # water moved
    assert (o['deficit'] < d['deficit'] - 1e-9).any()          # and was repaid
    ring = rt.ring_logical().T                         # [N sweep, R]
    np.testing.assert_allclose(rt.to_cells(ring.T).T.cpu().numpy(), o['ring'],
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rt.to_cells(rt.deficit).cpu().numpy(),
                               o['deficit'], rtol=1e-12, atol=1e-12)
    for f in ('withdrawn_tremote', 'available_tremote', 'returned',
              'withdrawn_remote'):
        np.testing.assert_allclose(
            rt.to_cells(getattr(st, f).T).T.cpu().numpy() * d['act'],
            o[f] * d['act'], rtol=1e-12, atol=1e-12, err_msg=f)


# -------------------------------------------------------------- replay
def read_dump(path):
    out, dims = {4: [], 5: [], 7: []}, None
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            tag, n = struct.unpack('<iq', h)
            a = np.frombuffer(f.read(8 * n), '<f8')
            if tag == 0:
                dims = a.astype(int)
            elif tag in out:
                out[tag].append(a.copy())
    return dims, {k: np.stack(v) for k, v in out.items() if v}


def read_flows(rundir, nt):
    """OUT_BASEFLOW and OUT_RUNOFF [T, N] (float32 history files)."""
    bf, ro = [], []
    for fn in sorted(glob.glob(os.path.join(rundir, 'W3', 'out',
                                            'fluxes.*.nc'))):
        ds = ncshare.read(fn, decode_times=False, mask_and_scale=False)
        for v, acc in (('OUT_BASEFLOW', bf), ('OUT_RUNOFF', ro)):
            a = np.asarray(ds[v].values, dtype=np.float64)
            acc.append(a.reshape(a.shape[0], -1))
        ds.close()
    bf, ro = np.concatenate(bf), np.concatenate(ro)
    assert len(bf) == nt
    return bf, ro


def read_edges():
    from torchvic.vicwur import params as P
    grid = P.Grid(R1 + '/domain.nc', R1 + '/params.nc')
    rp = P.read_remote(R1 + '/wateruse.nc', grid)
    src, dst = R.edges_from_receiving(rp.receiving)
    dp = P.read_dams(R1 + '/dams.nc', grid)
    return grid, src, dst, dp


RUNS = {'G6a': (os.path.join(VER, 'vicwur_14', 'gruns', 'R1', 'G6a'),
                False, np.inf),
        'G6b': (os.path.join(VER, 'vicwur_14', 'gruns', 'R1', 'G6b'),
                True, np.inf),
        'G6c': (os.path.join(VER, 'vicwur_14', 'gruns', 'R1', 'G6c'),
                True, 20.0),
        'G9': (os.path.join(VER, 'vicwur_14', 'gruns', 'R1', 'G9'),
               False, np.inf),
        'G10': (os.path.join(VER, 'vicwur_03', 'gruns', 'R1', 'G10_dump'),
                False, np.inf)}


@pytest.mark.parametrize('run', sorted(RUNS))
def test_l2_replay(run):
    rundir, nrr, lim = RUNS[run]
    dump = os.path.join(rundir, 'dump.bin')
    if not os.path.exists(dump) or not os.path.isdir(R1):
        pytest.skip('dump or R1 domain absent: ' + dump)
    grid, esrc, edst, dams = read_edges()
    dims, rec = read_dump(dump)
    N, ring_len = int(dims[0]), int(dims[5])
    T = rec[4].shape[0]
    assert N == grid.n
    x = rec[5].reshape(T, N, NS, 22)
    act = torch.as_tensor(~np.isnan(x[..., 0]))
    x = torch.as_tensor(np.nan_to_num(x), dtype=torch.float64)
    deficit_end = torch.as_tensor(rec[4].reshape(T, N, 5 + ring_len)[..., 3],
                                  dtype=torch.float64)
    g = lambda j: x[..., j]
    dgw, dsf, wgw, wsf, wdm = g(6), g(7), g(11), g(12), g(13)
    a_trem, d_trem, w_trem = g(3), g(8), g(14)
    a_rem, d_rem, w_rem = g(4), g(9), g(15)
    w_nr, ret, con = g(16), g(17), g(18)
    a_tmp, d_tmp, w_tmp = g(19), g(20), g(21)
    plan = R.RemotePlan(esrc, edst, N)
    assert plan.n_edges == 2 and len(plan.batches) == 1
    b = plan.batches[0]
    # every receiver has one source here: wu_var temporaries are its own
    assert len(set(b.edst.tolist())) == b.n_edges

    # per-sector consumption fraction: consumed = cf * (all withdrawals)
    wall = wgw + wsf + wdm + w_trem + w_nr
    cf = torch.where(wall > 0, con / torch.where(wall > 0, wall,
                                                 torch.ones_like(wall)),
                     torch.full_like(wall, .5))
    assert (cf[act] >= -1e-12).all() and (cf[act] <= 1 + 1e-12).all()

    # (1) demand (:172-192): receivers; withdrawn_tremote enters it, which
    # is 0 before the (single) source of a receiver
    zero = torch.zeros_like(w_trem)
    dem = R.demand_tmp(dgw, dsf, wgw, wsf, wdm, zero)
    ed, es = b.edst, b.esrc
    dem_e = dem[:, ed] * act[:, ed]                     # [T, E, S]
    torch.testing.assert_close(dem_e, d_tmp[:, ed] * act[:, ed],
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dem_e, d_trem[:, ed] * act[:, ed],
                               rtol=1e-12, atol=1e-12)
    # source total demand_remote = sum over its receivers
    for e in range(b.n_edges):
        torch.testing.assert_close(d_rem[:, es[e]] * act[:, es[e]],
                                   dem_e[:, e] * act[:, es[e]],
                                   rtol=1e-12, atol=1e-12)

    # (2) division (:261-296) and use (:337-353).  The dump has the shares
    # but not the pool; a pool that was exhausted equals their sum, one
    # that was not served every demand in full.
    ar_src = a_rem[:, es] * act[:, es]                  # [T, E, S]
    short = ar_src.sum(-1) < (dem_e * act[:, es]).sum(-1) * (1 - 1e-12)
    pool = torch.where(short, ar_src.sum(-1),
                       torch.full_like(short, 1e30, dtype=torch.float64))
    av_all = torch.stack([
        R._unpad(R.division(R._pad(dem_e[t], b), pool[t], act[t][es])[0], b)
        for t in range(T)])                             # [T, E, S]
    torch.testing.assert_close(av_all, a_tmp[:, ed] * act[:, ed],
                               rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(av_all, a_trem[:, ed] * act[:, ed],
                               rtol=1e-9, atol=1e-9)
    w_all, ret_all, con_all = R.use(av_all, dem_e, cf[:, ed])
    torch.testing.assert_close(w_all, w_tmp[:, ed] * act[:, ed],
                               rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(w_all, w_trem[:, ed] * act[:, ed],
                               rtol=1e-9, atol=1e-9)
    for e in range(b.n_edges):
        torch.testing.assert_close(w_rem[:, es[e]] * act[:, es[e]],
                                   w_all[:, e] * act[:, es[e]],
                                   rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(a_rem[:, es[e]] * act[:, es[e]],
                                   av_all[:, e] * act[:, es[e]],
                                   rtol=1e-9, atol=1e-9)
    assert (w_all > 0).any() and short.any() and (~short).any()
    assert ((w_all > 0).sum() > 500)

    # (3) deficit chain of every cell.  Per step: P1 baseflow (and runoff)
    # payback, local payback from the local returns, the remote paybacks of
    # the sources that serve the cell, then the non-renewable booking.
    bf, ro = read_flows(rundir, T)
    bf = torch.as_tensor(bf, dtype=torch.float64)
    ro = torch.as_tensor(ro, dtype=torch.float64)
    ro_file = ro.clone()
    # a local dam rewrites OUT_RUNOFF after rout_run has used it
    # (dam_run.c:67-76): runoff_file = runoff * (1 - inflow_frac) +
    # release * M3_PER_HM3 * MM_PER_M / area, so rout_run saw the inverse
    if 7 in rec:
        ndmax = int(dims[7])
        assert ndmax == 1
        dv = torch.as_tensor(rec[7].reshape(T, N, -1), dtype=torch.float64)
        rel, on = dv[..., 4], dv[..., 0] > 0           # release, active
        area = torch.as_tensor(grid.area_active, dtype=torch.float64)
        for c, gl, f in zip(dams.dam_cell.tolist(), dams.is_global.tolist(),
                            dams.inflow_fraction.tolist()):
            if not gl:
                seen = (ro[:, c] - rel[:, c] * 1e6 * 1000 / area[c]) / (1 - f)
                ro[:, c] = torch.where(on[:, c], seen, ro[:, c])
    loc_ret = (wgw * (1 - cf)) + (wsf * (1 - cf)) + (wdm * (1 - cf))
    loc_ret = loc_ret * act
    def chain(remote):
        errs = torch.zeros(T, N, dtype=torch.float64)
        for t in range(T):
            d_ = deficit_end[t - 1] if t else torch.zeros(N, dtype=torch.float64)
            # P1 (rout_run.c:49-76), torchvic.vicwur.nonrenew
            d1, _, _ = NR.payback_flow(d_, bf[t], ro[t], nrr)
            # local payback (wu_run_local.c:439-469)
            d2, _ = W.payback(d1, loc_ret[t], loc_ret[t].sum(-1), act[t])
            # remote payback as a receiver of the cell's source (:405-449)
            r_src = R._seq_sum(R._pad(ret_all[t], b).sum(1))      # [B]
            d3, _, _ = R.payback(d2, ret_all[t], ed, (r_src > 0)[b.eb],
                                 act[t][ed] & act[t][es])
            if not remote:
                d3 = d2
            # non-renewable booking and own payback (wu_nonrenewable.c:210-252)
            wn = w_nr[t] * act[t]
            r_nr = (wn * (1 - cf[t])).sum(-1)
            d4, _ = NR.book_and_repay(d3, wn.sum(-1), wn, cf[t], r_nr, act[t])
            errs[t] = (d4 - deficit_end[t]).abs()
        return errs

    errs = chain(True)
    # the remote payback is part of the chain: without it the receivers fail
    no_rem = chain(False).max(0).values
    assert (no_rem[ed] > 1e-3).all() and (no_rem[es] < 1e-6).all()
    # The residual is the float32 rounding of the history fields the chain
    # reads (OUT_BASEFLOW, OUT_RUNOFF; the dump itself is float64): a step
    # can be off by at most half an ulp of each, i.e. 2**-24 * |value|,
    # times 1 / (1 - inflow_frac) = 5 for the runoff of the local-dam cell
    # (the dam rewrite is inverted above).  Every step of every cell is
    # inside that bound, and the bound is within 1.5x of the residual.
    amp = torch.ones(N, dtype=torch.float64)
    if 7 in rec:
        for c, gl, f in zip(dams.dam_cell.tolist(), dams.is_global.tolist(),
                            dams.inflow_fraction.tolist()):
            if not gl:
                amp[c] = 1 / (1 - f)
    bound = 2.0 ** -24 * (bf.abs() + (ro_file.abs() * amp if nrr else 0)) \
        + 1e-12
    assert (errs <= bound).all(), (errs - bound).max()
    assert errs.max() > 0.2 * bound.max() or not nrr
    worst = errs.max(0).values
    print(run, 'max |deficit err| per cell (mm)', worst.tolist(),
          'rounding bound', bound.max(0).values.tolist(),
          'max deficit', deficit_end.max().item())
    assert worst.max() < (1e-6 if nrr else 1e-8)
    # the remote cells are in it: the receivers are paid back from remote
    rem = torch.zeros(N, dtype=torch.bool)
    rem[ed], rem[es] = True, True
    assert rem.sum() == 4
    moved = ((ret_all.sum(-1) > 0) & (deficit_end[:, ed] > 0)).sum()
    assert moved > 100
