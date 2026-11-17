# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.nonrenew (non-renewable groundwater).

Oracle: tests/oracle/ref_vicwur_nonrenew.c compiles the unchanged VIC-WUR 3.0
wu_nonrenewable.c and rout_run.c; the batched float64 kernels must match it
to 1e-12 on random and edge inputs, in every option setting (GWM,
NONRENEWABLE_RUNOFF, finite and infinite NONRENEWABLE_LIMIT, ROUT_STEPS 1/2/4).
Replay of the R1 dump runs G6a (NONREN), G6b (+NONRENEWABLE_RUNOFF), G6c
(+NONRENEWABLE_LIMIT 20) and G10 (all plugins), built with VICWUR_DUMP:
demand, division and use reproduce wu_var of every cell and step, and the
deficit of the cells with no remote water use reproduces the dumped deficit
from the previous step's deficit and the step's VIC baseflow and runoff
(float32 history output; the chain is checked to 1e-5 mm, it agrees to 5e-7).
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

HERE = os.path.dirname(os.path.abspath(__file__))
SHIMS = [os.path.join(HERE, 'oracle', 'vicwur_shim_nr'),
         os.path.join(HERE, 'oracle', 'vicwur_shim')]
SRC = os.path.join(HERE, 'oracle', 'ref_vicwur_nonrenew.c')
W3 = _paths.env_path(('TORCHVIC_VICWUR_SRC', 'VICWUR_SRC'), 'build/vicwur3/src_tree/vic')
VER = _paths.env_path(('TORCHVIC_VICWUR_VERIFY',), 'verification')
NS = 5
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    P = os.path.join(W3, 'plugins')
    if not os.path.exists(os.path.join(P, 'wateruse', 'src',
                                       'wu_nonrenewable.c')):
        pytest.skip('W3 source tree absent: ' + W3)
    exe = str(tmp_path_factory.mktemp('nr') / 'ref_nr')
    inc = ['-I' + d for d in SHIMS + [
        os.path.join(P, 'routing', 'include'),
        os.path.join(P, 'wateruse', 'include'),
        os.path.join(P, 'wateruse', 'src'), os.path.join(P, 'routing', 'src')]]
    r = subprocess.run(['cc', '-o', exe, SRC] + inc + [
        '-O0', '-std=c99', '-ffp-contract=off', '-D_DEFAULT_SOURCE',
        '-Wno-unused-variable', '-Wno-unused-but-set-variable',
        '-Wno-absolute-value', '-lm'], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('cannot build oracle: ' + r.stderr[-300:])
    return exe


# (gwm, nonrenew_runoff, uh_length, n_sub, nren_lim)
CONFIGS = [(0, 0, 5, 1, np.inf), (0, 1, 5, 1, np.inf), (0, 0, 7, 4, 20.0),
           (0, 1, 6, 4, 3.0), (1, 0, 5, 1, np.inf), (1, 1, 8, 2, 5.0),
           (0, 1, 24, 4, 0.0), (0, 0, 3, 2, 1e-3)]


def make_cases(n, seed, cfg):
    """Random cells with edge values mixed in."""
    gwm, nrr, uh, nsub, lim = cfg
    g = np.random.default_rng(seed)
    k = uh + nsub - 1

    def u(*shape, lo=0.0, hi=1.0):
        return g.uniform(lo, hi, shape)

    def zero(a, p):
        return np.where(g.random(a.shape) < p, 0.0, a)

    d = dict(cfg=cfg, n=n)
    d['area'] = u(n, lo=1e8, hi=1e9)
    d['deficit'] = zero(u(n, lo=0, hi=60) * (10.0 ** g.uniform(-3, 1, n)), .3)
    if np.isfinite(lim):
        d['deficit'] = np.minimum(d['deficit'], lim * g.uniform(0, 1.3, n))
    d['bf'] = zero(u(n, lo=-0.2, hi=8) * (10.0 ** g.uniform(-3, 0.5, n)), .2)
    d['ro'] = zero(u(n, lo=-0.1, hi=8) * (10.0 ** g.uniform(-3, 0.5, n)), .2)
    d['act'] = (g.random((n, NS)) < 0.85).astype(float)
    d['dgw'] = zero(u(n, NS, lo=0, hi=6), .3)
    d['dsf'] = zero(u(n, NS, lo=0, hi=6), .3)
    tot = d['dgw'] + d['dsf']
    # served part: sometimes all, sometimes none, sometimes more than asked
    frac = np.where(g.random((n, NS)) < .25, 1.0, u(n, NS, lo=0, hi=1.1))
    frac = np.where(g.random((n, NS)) < .1, 0.0, frac)
    tw = tot * frac
    sp = u(n, NS, 3)
    sp = sp / sp.sum(-1, keepdims=True)
    d['wgw'], d['wsf'], d['wdm'] = [tw * sp[..., i] for i in range(3)]
    d['wtr'] = zero(u(n, NS, lo=0, hi=0.5), .8)
    d['cf'] = np.where(g.random((n, NS)) < .15,
                       g.integers(0, 2, (n, NS)).astype(float), u(n, NS))
    keep = g.random(n) < 0.5
    d['ret0'] = np.where(keep[:, None], zero(u(n, NS, hi=2), .5), 0.0)
    d['con0'] = np.where(keep[:, None], zero(u(n, NS, hi=2), .5), 0.0)
    q = zero(u(n, k, lo=0, hi=300), .3)
    q[g.random(n) < .2] = 0.0                       # empty ring
    d['ring'] = q
    d['disch'] = u(n, lo=0, hi=400)
    d['stream'] = u(n, lo=0, hi=400)
    # more unmet demand in half the cells
    big = g.random(n) < .5
    d['dgw'][big] *= 4
    d['dsf'][big] *= 4
    # hand-made edge cells
    d['act'][0] = 1; d['dsf'][0] = [3, 5, 4, 0, 0]; d['dgw'][0] = 0
    d['wgw'][0] = d['wsf'][0] = d['wdm'][0] = d['wtr'][0] = 0   # all unmet
    d['deficit'][0] = 0
    d['dgw'][1] = 0; d['dsf'][1] = 0                       # no demand at all
    d['dsf'][2] = [0, 7, 0, 0, 0]; d['dgw'][2] = 0           # ENERGY only
    d['wgw'][2] = d['wsf'][2] = d['wdm'][2] = 0; d['act'][2] = 1
    d['ring'][3] = 0; d['dsf'][3] = [2, 0, 3, 0, 0]; d['wsf'][3] = 0
    d['wgw'][3] = d['wdm'][3] = 0; d['act'][3] = 1; d['cf'][3] = .3
    d['bf'][4] = 0.0; d['ro'][4] = 0.0
    d['deficit'][5] = 5.0; d['bf'][5] = 5.0                 # bf == deficit
    d['ro'][5] = 0.0
    d['deficit'][6] = 2.0; d['bf'][6] = 1.0; d['ro'][6] = 2.0
    return d


def pack(d):
    n = d['n']
    cols = [d['area'][:, None], d['deficit'][:, None], d['bf'][:, None],
            d['ro'][:, None], d['act'], d['dgw'], d['dsf'], d['wgw'],
            d['wsf'], d['wdm'], d['wtr'], d['cf'], d['ret0'], d['con0'],
            d['ring'], d['disch'][:, None], d['stream'][:, None]]
    return np.concatenate([np.asarray(c, float) for c in cols], 1)


def run_oracle(exe, d, tmp_path):
    gwm, nrr, uh, nsub, lim = d['cfg']
    rec = pack(d)
    fi, fo = str(tmp_path / 'in.bin'), str(tmp_path / 'out.bin')
    with open(fi, 'wb') as f:
        f.write(np.array([d['n'], gwm, nrr, uh, nsub, lim, 86400.0],
                         '<f8').tobytes())
        f.write(rec.astype('<f8').tobytes())
    subprocess.run([exe, fi, fo], check=True)
    o = np.fromfile(fo, '<f8').reshape(d['n'], -1)
    names = [('deficit_r', 1), ('runoff', 1), ('dnr', NS), ('avnr', NS),
             ('wnr', NS), ('ret', NS), ('con', NS), ('dcell', 1),
             ('acell', 1), ('wcell', 1), ('rcell', 1), ('deficit', 1),
             ('disch', 1), ('stream', 1), ('ring', uh + nsub - 1)]
    out, k = {}, 0
    for nm, w in names:
        out[nm] = o[:, k:k + w]
        k += w
    assert k == o.shape[1]
    return out


def run_torch(d, device='cpu'):
    gwm, nrr, uh, nsub, lim = d['cfg']
    t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt,
                                                    device=device)
    area, dt = t(d['area']), 86400.0
    act = t(d['act'] > 0, torch.bool)
    # P1: rout_run.c
    dfc, bf, ro = NR.payback_flow(t(d['deficit']), t(d['bf']), t(d['ro']),
                                  bool(nrr), bool(gwm))
    runoff = NR.runoff_to_discharge(ro, bf, area, dt)
    # P3: wu_nonrenew
    r = NR.nonrenew_step(
        t(d['dgw']), t(d['dsf']), t(d['wgw']), t(d['wsf']), t(d['wdm']),
        t(d['cf']), dfc, lim, act, wtremote=t(d['wtr']),
        returned_in=t(d['ret0']), consumed_in=t(d['con0']), gwm=bool(gwm),
        dt_discharge=t(d['ring']), area=area, dt=dt, n_sub=nsub,
        uh_length=uh, discharge=t(d['disch']), stream=t(d['stream']))
    r['deficit_r'], r['runoff'] = dfc, runoff
    return {k: v.detach().cpu().numpy() for k, v in r.items()}


def compare(o, r, n):
    eq = lambda a, b, what: np.testing.assert_allclose(
        a, b, rtol=1e-12, atol=1e-12, err_msg=what)
    eq(r['deficit_r'], o['deficit_r'][:, 0], 'deficit after rout_run')
    eq(r['runoff'], o['runoff'][:, 0], 'rout_var.runoff')
    eq(r['demand_nonrenew'], o['dnr'], 'demand_nonrenew')
    eq(r['available_nonrenew'], o['avnr'], 'available_nonrenew')
    eq(r['withdrawn_nonrenew'], o['wnr'], 'withdrawn_nonrenew')
    eq(r['returned'], o['ret'], 'returned')
    eq(r['consumed'], o['con'], 'consumed')
    eq(r['cell_demand'], o['dcell'][:, 0], 'cell demand')
    eq(r['cell_available'], o['acell'][:, 0], 'cell availability')
    eq(r['cell_withdrawn'], o['wcell'][:, 0], 'cell withdrawn')
    eq(r['deficit'], o['deficit'][:, 0], 'deficit at end')
    eq(r['dt_discharge'], o['ring'], 'ring')
    eq(r['discharge'], o['disch'][:, 0], 'discharge')
    eq(r['stream'], o['stream'][:, 0], 'stream')
    assert not r['violation'].any()


@pytest.mark.parametrize('ic', range(len(CONFIGS)))
def test_l1_random_edge(oracle, tmp_path, ic):
    d = make_cases(3000, 10 + ic, CONFIGS[ic])
    o = run_oracle(oracle, d, tmp_path)
    r = run_torch(d)
    compare(o, r, d['n'])
    # the interesting branches were reached
    lim = CONFIGS[ic][4]
    if lim > 0:
        assert (o['wnr'] > 0).any()
    else:
        assert (o['wnr'] == 0).all()                   # limit 0 serves nothing
    if np.isfinite(lim) and lim > 0:
        assert (o['dnr'] > o['wnr'] + 1e-9).any()      # stock cap binds
    if not CONFIGS[ic][0] and lim >= 1:
        assert (o['ring'] != d['ring']).any()          # ring edit happened
        assert (o['deficit'][:, 0] < o['deficit_r'][:, 0] +
                o['wcell'][:, 0] - 1e-12).any()        # immediate payback
    if CONFIGS[ic][1] and not CONFIGS[ic][0]:
        assert (o['runoff'][:, 0] < (d['bf'] + d['ro']) * d['area'] /
                86.4e6 - 1e-12).any()                  # runoff payback


def test_l1_energy_never_served(oracle, tmp_path):
    d = make_cases(500, 99, CONFIGS[0])
    o = run_oracle(oracle, d, tmp_path)
    assert (o['dnr'][:, 1] == 0).all() and (o['wnr'][:, 1] == 0).all()
    assert run_torch(d)['demand_nonrenew'][:, 1].max() == 0


def _c(*a):
    return torch.tensor([list(a)], dtype=torch.float64)


def _s(v):
    return torch.tensor([v], dtype=torch.float64)


def test_hand_cases():
    """The cases of docs/connectors/vicwur.md."""
    z = torch.zeros(1, 5, dtype=torch.float64)
    mm = torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.bool)
    # (a) D_nr = [3, 4] (MUN, MAN), NREN_LIM 5, cf 0.5 -> W = [3, 2],
    # deficit 5 - 1.5 - 1 = 2.5, nothing to the river
    r = NR.nonrenew_step(z, _c(3, 0, 4, 0, 0), z, z, z,
                         torch.full((1, 5), .5, dtype=torch.float64),
                         _s(0.0), 5.0, mm)
    assert r['withdrawn_nonrenew'][0].tolist() == [3, 0, 2, 0, 0]
    assert r['deficit'].item() == 2.5 and r['returned_to_river'].item() == 0
    # (b) next step baseflow 1 -> deficit 1.5; with NONRENEWABLE_RUNOFF and
    # runoff 2 -> deficit 0 and runoff 0.5
    d, bf, ro = NR.payback_flow(_s(2.5), _s(1.0), _s(2.0), False)
    assert (d.item(), bf.item(), ro.item()) == (1.5, 0.0, 2.0)
    d, bf, ro = NR.payback_flow(_s(2.5), _s(1.0), _s(2.0), True)
    assert (d.item(), bf.item(), ro.item()) == (0.0, 0.0, 0.5)
    # baseflow larger than the deficit: the rest flows on
    d, bf, ro = NR.payback_flow(_s(0.5), _s(1.0), _s(2.0), True)
    assert (d.item(), bf.item(), ro.item()) == (0.0, 0.5, 2.0)
    # GWM: baseflow dropped, deficit untouched
    d, bf, ro = NR.payback_flow(_s(0.5), _s(1.0), _s(2.0), True, gwm=True)
    assert (d.item(), bf.item(), ro.item()) == (0.5, 0.0, 2.0)
    # (c) infinite limit: everything unmet is served
    r = NR.nonrenew_step(z, _c(30, 0, 40, 0, 0), z, z, z,
                         torch.full((1, 5), .5, dtype=torch.float64),
                         _s(100.0), None, mm)
    assert r['withdrawn_nonrenew'][0].tolist() == [30, 0, 40, 0, 0]
    # (d) energy never served
    r = NR.nonrenew_step(z, _c(0, 9, 0, 0, 0), z, z, z,
                         torch.full((1, 5), .5, dtype=torch.float64),
                         _s(0.0), None, torch.ones(1, 5, dtype=torch.bool))
    assert r['demand_nonrenew'].abs().sum().item() == 0
    assert r['cell_withdrawn'].item() == 0 and r['deficit'].item() == 0
    # the limit caps the stock: deficit 4 of limit 5 leaves 1
    r = NR.nonrenew_step(z, _c(3, 0, 4, 0, 0), z, z, z,
                         torch.zeros(1, 5, dtype=torch.float64),
                         _s(4.0), 5.0, mm)
    assert r['withdrawn_nonrenew'][0].tolist() == [1, 0, 0, 0, 0]
    # conservation: dD = W - paybacks; here cf = 0 returns all, no deficit
    assert r['deficit'].item() == 4.0
    # balance check flags withdrawn > available
    assert NR.balance_violation(_c(1., 0, 0, 0, 0), _c(.5, 0, 0, 0, 0),
                                _c(1., 0, 0, 0, 0), _s(1.), _s(.5),
                                _s(1.)).item()


def test_ring_return_hand():
    ring = torch.tensor([[1., 1., 2., 0., 2.]], dtype=torch.float64)
    # n_sub 2, UH_LENGTH 4: bins 2..4 hold 2, 0, 2; return 86.4 mm over
    # area 1e6 m2 and dt 86400 s = 1 m3/s, split 1/2, 0, 1/2
    out, dis, st = NR.ring_return(ring, _s(86.4), _s(1e6), 86400.0, 2, 4)
    assert out[0].tolist() == pytest.approx([1, 1, 2.5, 0, 2.5], abs=1e-12)
    assert dis.item() == pytest.approx(2.0) and st.item() == pytest.approx(5.0)
    # all future bins empty: equal shares over UH_LENGTH - 1 bins
    ring = torch.tensor([[1., 1., 0., 0., 0.]], dtype=torch.float64)
    out, _, _ = NR.ring_return(ring, _s(86.4), _s(1e6), 86400.0, 2, 4)
    assert out[0, 2:].tolist() == pytest.approx([1 / 3.] * 3)
    # nothing returned: untouched, the given discharge and stream pass
    out, dis, st = NR.ring_return(ring, _s(0.0), _s(1e6), 86400.0, 2, 4,
                                  _s(7.0), _s(9.0))
    assert out.equal(ring) and dis.item() == 7.0 and st.item() == 9.0


# -------------------------------------------------------------- replay
def read_dump(path):
    out, dims = {4: [], 5: []}, None
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
    return dims, {k: np.stack(v) for k, v in out.items()}


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


RUNS = {'G6a': (os.path.join(VER, 'vicwur_14', 'gruns', 'R1', 'G6a'),
                False, np.inf),
        'G6b': (os.path.join(VER, 'vicwur_14', 'gruns', 'R1', 'G6b'),
                True, np.inf),
        'G6c': (os.path.join(VER, 'vicwur_14', 'gruns', 'R1', 'G6c'),
                True, 20.0),
        'G10': (os.path.join(VER, 'vicwur_03', 'gruns', 'R1', 'G10_dump'),
                False, np.inf)}


@pytest.mark.parametrize('run', sorted(RUNS))
def test_l2_replay(run):
    rundir, nrr, lim = RUNS[run]
    dump = os.path.join(rundir, 'dump.bin')
    if not os.path.exists(dump):
        pytest.skip('dump absent: ' + dump)
    dims, rec = read_dump(dump)
    N, ring = int(dims[0]), int(dims[5])
    T = rec[4].shape[0]
    ro_ = rec[4].reshape(T, N, 5 + ring)
    x = rec[5].reshape(T, N, NS, 22)
    act = torch.as_tensor(~np.isnan(x[..., 0]))
    x = torch.as_tensor(np.nan_to_num(x), dtype=torch.float64)
    deficit_end = torch.as_tensor(ro_[..., 3], dtype=torch.float64)   # [T,N]
    g = lambda j: x[..., j]
    dgw, dsf, wgw, wsf, wdm, wtr = g(6), g(7), g(11), g(12), g(13), g(14)
    a_nr, d_nr, w_nr, ret, con = g(5), g(10), g(16), g(17), g(18)

    # (1) demand: exact
    d, dcell = NR.demand(dgw, dsf, wgw, wsf, wdm, wtr, act)
    torch.testing.assert_close(d, d_nr * act, rtol=1e-12, atol=1e-12)
    # (2) use: given the dumped shares, withdrawn_nonrenew
    u = NR.use(d_nr, a_nr, torch.full_like(a_nr, .5), active=act)
    torch.testing.assert_close(u['withdrawn'], w_nr * act, rtol=1e-12,
                               atol=1e-12)
    # (3) division: a pool that is exhausted gives the shares; one that is
    # not gives the demand
    s_a = (a_nr * act).sum(-1)
    s_d = (d_nr * act).sum(-1)
    short = s_a < s_d * (1 - 1e-12)
    av_cell = torch.where(short, s_a, torch.full_like(s_a, float('inf')))
    dv = NR.division(d_nr, av_cell, s_d, act)
    torch.testing.assert_close(dv, a_nr * act, rtol=1e-10, atol=1e-10)
    assert (w_nr > 0).any() and short.any() == (lim < np.inf)

    # (4) deficit chain, cells with no remote water use at all
    remote = (g(8).abs() + g(9).abs() + g(3).abs() + g(4).abs() + g(14).abs()
              + g(15).abs()).sum((0, 2)) > 0
    clean = (~remote).nonzero().squeeze(1)
    assert len(clean) >= 2
    bf, ro = read_flows(rundir, T)
    bf = torch.as_tensor(bf, dtype=torch.float64)[:, clean]
    ro = torch.as_tensor(ro, dtype=torch.float64)[:, clean]
    tot = ret + con
    cf = torch.where(tot > 0, con / torch.where(tot > 0, tot, tot + 1),
                     torch.full_like(tot, .5))
    w_cell = u['cell_withdrawn']
    ret_p2 = ret - w_nr * (1 - cf)                 # P2 returns only
    ret_p2 = torch.where(ret_p2.abs() < 1e-12, torch.zeros_like(ret_p2),
                         ret_p2)
    prev = torch.cat([torch.zeros(1, N, dtype=torch.float64),
                      deficit_end[:-1]])
    errs = []
    for ci, c in enumerate(clean.tolist()):
        d1, _, _ = NR.payback_flow(prev[:, c], bf[:, ci], ro[:, ci], nrr)
        rp = ret_p2[:, c]
        d2, _ = W.payback(d1, rp, rp.sum(-1), act[:, c])
        r_nr = u['cell_returned'][:, c]
        pred, _ = NR.book_and_repay(d2, w_cell[:, c], w_nr[:, c], cf[:, c],
                                    r_nr, act[:, c])
        errs.append((pred - deficit_end[:, c]).abs().max().item())
    print(run, 'cells', clean.tolist(), 'max |deficit err| mm', errs,
          'max deficit', deficit_end.max().item(),
          'steps with W>0', int((w_cell > 0).sum()))
    assert max(errs) < 1e-5


# --------------------------------------------------------------- devices
@pytest.mark.parametrize('device', DEVICES)
def test_devices_agree(oracle, tmp_path, device):
    d = make_cases(1500, 5, CONFIGS[1])
    compare(run_oracle(oracle, d, tmp_path), run_torch(d, device), d['n'])


@pytest.mark.gpu
def test_gpu(oracle, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    for ic in (0, 3, 5):
        d = make_cases(4000, 30 + ic, CONFIGS[ic])
        compare(run_oracle(oracle, d, tmp_path), run_torch(d, 'cuda'),
                d['n'])


@pytest.mark.gpu
def test_gpu_no_host_sync():
    """The step does not synchronise: it runs under torch.cuda.set_sync_debug_mode('error')."""
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    d = make_cases(2000, 77, CONFIGS[1])
    gwm, nrr, uh, nsub, lim = d['cfg']
    t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt,
                                                    device='cuda')
    args = [t(d[k]) for k in ('dgw', 'dsf', 'wgw', 'wsf', 'wdm', 'cf',
                              'deficit')]
    act = t(d['act'] > 0, torch.bool)
    ring, area = t(d['ring']), t(d['area'])
    bf, ro = t(d['bf']), t(d['ro'])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        dfc, b, r = NR.payback_flow(args[6], bf, ro, True)
        out = NR.nonrenew_step(*args[:6], dfc, lim, act, dt_discharge=ring,
                               area=area, dt=86400.0, n_sub=nsub,
                               uh_length=uh)
    finally:
        torch.cuda.set_sync_debug_mode('default')
    torch.cuda.synchronize()
    assert torch.isfinite(out['deficit']).all()
