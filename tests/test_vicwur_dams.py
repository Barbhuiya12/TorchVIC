# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.dams (global and local dams).

Oracle: tests/oracle/ref_vicwur_dam.c #includes the unmodified VIC-WUR 3.0
dam sources (dam_operation.c, dam_register.c, dam_run.c, ...) and drives
them; the batched kernels must agree in float64 to 1e-12 on
  * the release correction dam_corr_release (empty, full, in the band, at
    the band edges, several alpha, beta, gamma and steps per day),
  * dam_calc_opt_release / dam_corr_opt_release / dam_calc_opt_storage on
    random monthly arrays (efr above inflow, zero demand, demand above
    inflow, small and large relative capacity, c = inf),
  * the plan at a month change (op_month, op_release, op_storage, k, c and
    the climatologies) on random 60-month histories, with and without
    history (first year), at every month boundary and calendar,
  * whole runs of mixed global and local dams, 2-3 per cell, with service
    demand, EFR, build years before, inside and after the run, daily and
    sub-daily steps, leap years: every dam variable, the history, the ring
    write-back and the local runoff write-back, every step.
Also: the refactored reservoir engine plan is bitwise what it was.
Replay of the R1 G-run dumps: tests/test_vicwur_dams_l2.py.
"""
import os
import subprocess

import numpy as np
import pytest
import torch

from torchvic.reservoir import system as RS
from torchvic.vicwur import dams as DM

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, 'oracle', 'build_ref_vicwur_dam.sh')
RT = dict(rtol=1e-12, atol=1e-13)
f64 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    d = tmp_path_factory.mktemp('dam')
    env = dict(os.environ, TORCHVIC_ORACLE_DIR=str(d))
    r = subprocess.run(['bash', BUILD], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        pytest.skip('cannot build oracle: ' + (r.stderr + r.stdout)[-300:])
    return str(d / 'ref_vicwur_dam'), d


def run(oracle, arr):
    exe, d = oracle
    fi, fo = str(d / 'in.bin'), str(d / 'out.bin')
    np.ascontiguousarray(arr, dtype='<f8').tofile(fi)
    r = subprocess.run([exe, fi, fo], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return np.fromfile(fo, dtype='<f8')


# ------------------------------------------------------------- reservoir plan
def _legacy_plan(I, D, Iy, Dy, E, irr, k, c):
    """The body of Reservoirs._plan before it was refactored (verbatim)."""
    Ib = (Iy - E).clamp(min=0)
    Dm, Dyc = D, Dy[:, None]
    r = torch.where(Dyc <= 0, Ib[:, None].expand_as(D),
        torch.where(Dyc > Ib[:, None], Ib[:, None] * Dm / Dyc.clamp(min=1e-300),
                    Ib[:, None] - Dyc + Dm))
    r = torch.where(irr[:, None], r, Ib[:, None].expand_as(D)) + E[:, None]
    cc = ((c / 0.5) ** 2)[:, None]
    return torch.where((c >= 0.5)[:, None], k[:, None] * r,
                       cc * k[:, None] * r + (1 - cc) * I)


def test_reservoir_plan_is_bitwise_unchanged():
    g = torch.Generator().manual_seed(3)
    n = 4000
    I = torch.rand(n, 12, generator=g, dtype=torch.float64) * 100
    D = torch.rand(n, 12, generator=g, dtype=torch.float64) * 80 \
        * (torch.rand(n, 1, generator=g, dtype=torch.float64) > 0.2)
    E = torch.rand(n, generator=g, dtype=torch.float64) * 30
    irr = torch.rand(n, generator=g) > 0.4
    k = torch.rand(n, generator=g, dtype=torch.float64) * 1.2
    c = torch.rand(n, generator=g, dtype=torch.float64) * 2
    Iy, Dy = I.mean(1), D.mean(1)
    old = _legacy_plan(I, D, Iy, Dy, E, irr, k, c)
    Ib = (Iy - E).clamp(min=0)
    new = RS.hanasaki_target(I, D, Ib[:, None], Dy[:, None], E[:, None], k, c,
                             irr=irr)
    assert torch.equal(old, new)
    # the C order of the ratio differs from the module's by an ulp at most
    alt = RS.hanasaki_target(I, D, Ib[:, None], Dy[:, None], E[:, None], k, c,
                             irr=irr, ratio_first=True)
    torch.testing.assert_close(alt, new, rtol=1e-14, atol=0)


# ----------------------------------------------------------------- corr_release
def test_corr_release_l1(oracle):
    g = np.random.default_rng(1)
    n = 6000
    alpha = g.choice([0.85, 0.7, 0.5, 0.3, 0.95], n)
    beta = g.choice([0.6, 0.3, 0.0, 1.0, 2.5], n)
    gamma = g.choice([5.0, 2.0, 0.5, 30.0], n)
    spd = g.choice([1.0, 2.0, 4.0], n)
    cap = 10 ** g.uniform(-1, 3, n)
    sto = cap * g.uniform(0, 1.0, n)
    rel = 10 ** g.uniform(-3, 2, n) * g.integers(0, 2, n)
    # edge storages: empty, full, exactly on both band edges, inside, above
    edge = np.array([0.0, 1.0, 0.85, 0.15, 0.5, 0.999999, 1e-9])
    for i in range(len(edge)):
        sto[i] = cap[i] * edge[i]
        alpha[i] = 0.85
    sto[len(edge):len(edge) + 3] = cap[len(edge):len(edge) + 3] * 1.0000001
    rec = np.stack([alpha, beta, gamma, spd, rel, sto, cap], 1)
    got = run(oracle, np.concatenate([[2, n], rec.ravel()]))
    exp = np.empty(n)
    for s in np.unique(spd):
        m = spd == s
        exp_t = DM.corr_release(f64(rel[m]), f64(sto[m]), f64(cap[m]),
                                f64(alpha[m]), f64(beta[m]), f64(gamma[m]),
                                float(s))
        exp[m] = exp_t.numpy()
    np.testing.assert_allclose(exp, got, **RT)
    assert (got == 0).any() and (got > rel).any() and (got < rel).any()


# -------------------------------------------------------------- opt release/storage
def test_opt_release_l1(oracle):
    g = np.random.default_rng(2)
    n = 3000
    u = lambda *s, lo=0.0, hi=1.0: g.uniform(lo, hi, s)
    zero = lambda a, p: np.where(g.random(a.shape) < p, 0.0, a)
    I = 10 ** u(n, 12, lo=-1, hi=2.5)
    D = zero(10 ** u(n, 12, lo=-1, hi=2.5), 0.3) * 10 ** u(n, 1, lo=-1.5, hi=1)
    D[g.random(n) < 0.15] = 0.0                       # no demand at all
    E = zero(I * u(n, 12, lo=0, hi=1.4), 0.2)         # efr above inflow too
    steps = g.choice([28., 29., 30., 31., 62., 124.], (n, 12))
    k = u(n, lo=0, hi=1.4)
    k[:20] = 0.0                                      # empty reservoir
    c = 10 ** u(n, lo=-2.5, hi=0.7)
    c[20:40] = np.inf                                 # no history: c = inf
    c[40:60] = 0.5                                    # the 0.5 boundary
    cur = u(n, lo=0, hi=300)
    rec = np.concatenate([I, D, E, steps, k[:, None], c[:, None],
                          cur[:, None]], 1)
    out = run(oracle, np.concatenate([[3, n], rec.ravel()])).reshape(n, 36)
    r0 = DM.opt_release(f64(I), f64(D), f64(E))
    r1 = DM.corr_opt_release(f64(I), r0, f64(k), f64(c))
    s = DM.opt_storage(f64(I), r1, f64(steps), f64(cur))
    np.testing.assert_allclose(r0.numpy(), out[:, :12], **RT)
    np.testing.assert_allclose(r1.numpy(), out[:, 12:24], **RT)
    np.testing.assert_allclose(s.numpy(), out[:, 24:], **RT)
    assert (c < 0.5).any() and (c >= 0.5).any() and (D.sum(1) == 0).any()
    assert ((E.mean(1) > I.mean(1))).any()


# ---------------------------------------------------------------------- plan
def plan_cases(n, seed):
    g = np.random.default_rng(seed)
    u = lambda *s, lo=0.0, hi=1.0: g.uniform(lo, hi, s)
    c = dict(cal=g.choice([0, 2, 3], n).astype(float))
    c['year'] = g.choice([1999, 2000, 2003, 2004, 2008, 2100, 2400], n).astype(float)
    c['month'] = g.integers(1, 13, n).astype(float)
    c['spd'] = g.choice([1., 2., 4.], n)
    c['alpha'] = g.choice([0.85, 0.7, 0.4], n)
    c['beta'] = g.choice([0.6, 0.3], n)
    c['gamma'] = g.choice([5.0, 2.0], n)
    c['cap'] = 10 ** u(n, lo=0, hi=3.7)
    c['sto'] = c['cap'] * u(n)
    c['sto'][:10] = 0.0
    c['sto'][10:20] = c['cap'][10:20]
    c['mr'] = g.integers(0, 75, n).astype(float)
    c['mr'][:30] = np.arange(30)                      # every first-year age
    base = 10 ** u(n, 1, lo=0, hi=2.2)
    seas = 1 + 0.9 * np.sin(2 * np.pi * (np.arange(60)[None, :] / 12.
                                         + u(n, 1)))
    c['hi'] = base * seas * u(n, 60, lo=0.5, hi=1.5)
    c['hd'] = np.where(g.random((n, 1)) < 0.2, 0.0,
                       base * u(n, 1, lo=0, hi=1.5) * u(n, 60, lo=0.2, hi=1.8))
    c['he'] = np.where(g.random((n, 1)) < 0.3, 0.0,
                       base * u(n, 1, lo=0, hi=0.7) * u(n, 60))
    return c


def test_plan_l1(oracle):
    n = 2500
    c = plan_cases(n, 4)
    rec = np.concatenate([np.stack([c['cal'], c['year'], c['month'], c['spd'],
                                    c['alpha'], c['beta'], c['gamma'],
                                    c['cap'], c['sto'], c['mr']], 1),
                          c['hi'], c['hd'], c['he']], 1)
    out = run(oracle, np.concatenate([[1, n], rec.ravel()])).reshape(n, 87)
    exp_om = np.zeros(n); exp_rel = np.zeros((n, 12)); exp_sto = np.zeros((n, 12))
    exp_k = np.zeros(n); exp_c = np.zeros(n)
    exp_ci = np.zeros((n, 12))
    # the plan is vectorised over dams: one batch per distinct scalar
    # parameter set and (year, month)
    keys = np.stack([c['cal'], c['spd'], c['alpha'], c['beta'], c['gamma']], 1)
    for key in np.unique(keys, axis=0):
        m = (keys == key).all(1)
        idx = np.where(m)[0]
        cal = {0: 'standard', 2: 'proleptic_gregorian', 3: 'noleap'}[int(key[0])]
        for yr, mo in sorted({(int(c['year'][i]), int(c['month'][i]))
                              for i in idx}):
            sel = idx[(c['year'][idx] == yr) & (c['month'][idx] == mo)]
            cap = f64(c['cap'][sel])
            ds = DM.GlobalDams(cap, torch.ones_like(cap), torch.full_like(cap, 0).long(),
                               alpha=key[2], beta=key[3], gamma=key[4],
                               steps_per_day=int(key[1]), exact=True)
            ds.storage = f64(c['sto'][sel])
            ds.history_inflow = f64(c['hi'][sel])
            ds.history_demand = f64(c['hd'][sel])
            ds.history_efr = f64(c['he'][sel])
            ds.months_running = torch.as_tensor(c['mr'][sel]).long()
            cal_t = DM.make_calendar([yr, yr], [1, mo], cal, int(key[1]))
            st = cal_t.at(1)
            om = DM.operation_start(ds.history_inflow, ds.months_running,
                                    st.month)
            ds._plan(torch.ones(len(sel), dtype=torch.bool), st)
            mr = ds.months_running
            ci = DM.climatology(ds.history_inflow, mr)
            exp_om[sel] = om.numpy()
            exp_rel[sel] = ds.op_release.numpy()
            exp_sto[sel] = ds.op_storage.numpy()
            exp_k[sel] = DM.k_factor(ds.capacity, ds.storage, ds.alpha).numpy()
            exp_c[sel] = DM.c_factor(ci, ds.capacity, st.steps).numpy()
            exp_ci[sel] = ci.numpy()
    np.testing.assert_array_equal(exp_om, out[:, 0])
    np.testing.assert_allclose(exp_rel, out[:, 1:13], **RT)
    np.testing.assert_allclose(exp_sto, out[:, 13:25], rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(exp_k, out[:, 25], **RT)
    np.testing.assert_allclose(exp_c, out[:, 26], rtol=1e-12)
    np.testing.assert_allclose(exp_ci, out[:, 27:39], **RT)
    # coverage: no-history plans, every op_month, both c regimes
    assert (c['mr'] < 12).sum() > 100
    assert set(np.unique(exp_om)) >= set(range(1, 13))
    assert (exp_c < 0.5).any() and (exp_c >= 0.5).any()


def test_fixed_mode_aligns_the_steps_per_month():
    """exact=False: the steps factor starts at the current month, as the
    climatology does; exact=True keeps the C misalignment."""
    cal = DM.make_calendar([2001, 2001], [1, 5], 'standard', 1)
    st = cal.at(1)
    d = DM.GlobalDams([10.0], [1.0], [1990], exact=True)
    de = DM.GlobalDams([10.0], [1.0], [1990], exact=False)
    assert torch.equal(d._steps(st), st.steps)
    al = de._steps(st)
    assert al.tolist() == [31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30]   # May first
    assert al.shape == (12,)


# ---------------------------------------------------------------- calendar
def test_calendar_matches_c():
    assert DM.days_per_month(2, 2000, 'standard') == 29
    assert DM.days_per_month(2, 1900, 'standard') == 28
    assert DM.days_per_month(2, 1900, 'proleptic_gregorian') == 28
    assert DM.days_per_month(2, 2004, 'noleap') == 28
    assert DM.days_per_month(2, 2004, 'julian') == 29
    assert DM.days_per_month(2, 1900, 'julian') == 29
    cal = DM.make_calendar([2003, 2003, 2004, 2004], [12, 12, 1, 1], 'standard')
    assert cal.new_month.tolist() == [False, False, True, False]
    assert cal.new_year.tolist() == [False, False, True, False]
    assert cal.new_month_host == [False, False, True, False]
    assert cal.steps[2, 1] == 29 and cal.steps[1, 1] == 28


# --------------------------------------------------------------------- run
def scenario(seed, nc=6, nt=1500, spd=1, rspd=4, cal='standard', y0=2003,
             m0=7, d0=1, alpha=0.85, beta=0.6, gamma=5.0, wu=True, efr=True,
             ns=3, maxs=3, dams_per_cell=3):
    g = np.random.default_rng(seed)
    u = lambda *s, lo=0.0, hi=1.0: g.uniform(lo, hi, s)
    sc = dict(nc=nc, nt=nt, spd=spd, rspd=rspd, lr=rspd // spd + 3, cal=cal,
              y0=y0, m0=m0, d0=d0, alpha=alpha, beta=beta, gamma=gamma,
              wu=wu, efr=efr, ns=ns, maxs=maxs)
    sc['area'] = u(nc, lo=5e7, hi=1.2e9)
    sc['ns_active'] = g.integers(0, ns + 1, nc)
    dams = []
    for cell in range(nc):
        for j in range(g.integers(0, dams_per_cell + 1)):
            ty = int(g.integers(0, 2))
            nsv = int(g.integers(0, maxs + 1))
            svc = list(g.integers(0, nc, nsv)) + [0] * (maxs - nsv)
            fr = list(u(nsv, lo=0.1, hi=1)) + [0.0] * (maxs - nsv)
            yr = int(g.choice([y0 - 10, y0, y0 + 1, y0 + 2, y0 + 3, y0 + 9]))
            cap = float(10 ** g.uniform(0.3, 3.4))
            fra = float(g.choice([1.0, 0.8, 0.5, 1.0]))
            dams.append([cell, ty, yr, cap, fra, nsv] + svc + fr)
    if not any(d[1] == 1 for d in dams):
        dams.append([0, 1, y0, 100.0, 1.0, 1] + [1, 0, 0] + [0.5, 0, 0])
    if not any(d[1] == 0 for d in dams):
        dams.append([1, 0, y0, 5.0, 0.8, 0] + [0, 0, 0] + [0, 0, 0])
    dams.sort(key=lambda d: d[0])
    sc['dams'] = np.array(dams, float)
    t = np.arange(nt)
    seas = 1 + 0.8 * np.sin(2 * np.pi * t[:, None] / (365.25 * spd)
                            + u(nc, lo=0, hi=2 * np.pi)[None])
    base = 10 ** u(nc, lo=0.5, hi=2.2)
    sc['runoff'] = np.maximum(base[None] * 0.02 * seas
                              * u(nt, nc, lo=0, hi=2), 0)
    sc['runoff'][g.random((nt, nc)) < 0.05] = 0.0
    n = rspd // spd
    ring = base[None, :, None] * seas[:, :, None] \
        * u(nt, nc, sc['lr'], lo=0.2, hi=1.8) / 4
    ring[g.random((nt, nc)) < 0.04] = 0.0
    sc['ring'] = ring
    # the C ring discharge is the sum of the first n bins (set by routing)
    sc['disch'] = ring[:, :, :n].sum(2)
    sc['efr_q'] = np.where(g.random((nt, nc)) < 0.2, 0.0,
                           base[None] * u(nt, nc, lo=0, hi=0.5)) * efr
    sc['ds'] = np.where(g.random((nt, nc, ns)) < 0.3, 0.0,
                        u(nt, nc, ns, lo=0, hi=4)) * wu
    return sc


def pack_scenario(sc):
    dm = sc['dams']
    head = [4, len(dm), sc['nc'], sc['nt'], sc['spd'], sc['rspd'], sc['lr'],
            {'standard': 0, 'proleptic_gregorian': 2, 'noleap': 3}[sc['cal']],
            sc['y0'], sc['m0'], sc['d0'], sc['alpha'], sc['beta'], sc['gamma'],
            float(sc['wu']), float(sc['efr']), sc['ns'], sc['maxs']]
    parts = [np.array(head, float),
             np.stack([sc['area'], sc['ns_active']], 1).ravel(), dm.ravel()]
    step = np.concatenate([sc['runoff'][:, :, None], sc['disch'][:, :, None],
                           sc['ring'], sc['efr_q'][:, :, None], sc['ds']], 2)
    parts.append(step.ravel())
    return np.concatenate(parts)


def dates(sc):
    """(year, month) of every step, as the harness builds dmy."""
    y, m, d = sc['y0'], sc['m0'], sc['d0']
    out = []
    for t in range(sc['nt']):
        if t > 0 and t % sc['spd'] == 0:
            d += 1
            if d > DM.days_per_month(m, y, sc['cal']):
                d, m = 1, m + 1
                if m > 12:
                    m, y = 1, y + 1
        out.append((y, m))
    return out


def run_python(sc, exact=True, device='cpu'):
    """The sets of rank 0, 1, ... for local then global dams, one step at a
    time, returning the same arrays as the harness."""
    dm = sc['dams']
    nd, nc, nt, lr = len(dm), sc['nc'], sc['nt'], sc['lr']
    n = sc['rspd'] // sc['spd']
    dt = 86400.0 / sc['spd']
    ys, ms = zip(*dates(sc))
    cal = DM.make_calendar(ys, ms, sc['cal'], sc['spd'], device=device)
    cell = dm[:, 0].astype(int)
    sets = []
    rank = DM.rank_in_cell(cell)         # class order over local + global
    for ty, cls in ((0, DM.LocalDams), (1, DM.GlobalDams)):
        sel_t = np.where(dm[:, 1] == ty)[0]
        rk = DM.rank_in_cell(cell[sel_t])
        for r in range(rk.max() + 1 if len(sel_t) else 0):
            ix = sel_t[rk == r]
            d = cls(dm[ix, 3], dm[ix, 4], dm[ix, 2].astype(int),
                    alpha=sc['alpha'], beta=sc['beta'], gamma=sc['gamma'],
                    steps_per_day=sc['spd'], exact=exact, device=device)
            svc = torch.as_tensor(dm[ix, 6:6 + sc['maxs']].astype(np.int64))
            frac = f64(dm[ix, 6 + sc['maxs']:6 + 2 * sc['maxs']])
            nsv = torch.as_tensor(dm[ix, 5].astype(np.int64))
            valid = torch.arange(sc['maxs'])[None, :] < nsv[:, None]
            sets.append((ty, ix, d, torch.as_tensor(cell[ix]), svc.to(device),
                         frac.to(device), valid.to(device)))
    area = f64(sc['area']).to(device)
    act = (torch.arange(sc['ns'])[None, :]
           < torch.as_tensor(sc['ns_active'])[:, None]).to(device)
    DW = 12 + 180 + 24
    out = np.zeros((nt, nd * DW + nc * (2 + lr)))
    for t in range(nt):
        st = cal.at(t)
        runoff = f64(sc['runoff'][t]).to(device)
        disch = f64(sc['disch'][t]).to(device)
        ring = f64(sc['ring'][t]).to(device).clone()
        efr_q = f64(sc['efr_q'][t]).to(device)
        ds = f64(sc['ds'][t]).to(device)
        for ty, ix, d, cl, svc, frac, valid in sets:
            if ty == 0:
                runoff = runoff.index_put((cl,), d.run(st, runoff[cl], area[cl]))
        for ty, ix, d, cl, svc, frac, valid in sets:
            if ty == 1:
                dem = None
                if sc['wu']:
                    ar = area[cl] if exact else area[svc]
                    dem = DM.service_demand(ds, act, svc, frac, valid, ar)
                q = efr_q[cl] if sc['efr'] else None
                rn, dn = d.run(st, ring[cl][:, :n], disch[cl], dt, dem, q)
                ring = ring.index_put((cl,), torch.cat([rn, ring[cl][:, n:]], 1))
                disch = disch.index_put((cl,), dn)
        for ty, ix, d, cl, svc, frac, valid in sets:
            rows = torch.cat([
                d.active.to(torch.float64)[:, None], d.inflow[:, None],
                d.demand[:, None], d.efr[:, None], d.release[:, None],
                d.storage[:, None], d.op_month.to(torch.float64)[:, None],
                d.months_running.to(torch.float64)[:, None],
                d.total_inflow[:, None], d.total_demand[:, None],
                d.total_efr[:, None],
                d.register_steps.to(torch.float64)[:, None],
                d.history_inflow, d.history_demand, d.history_efr,
                d.op_release, d.op_storage], 1).cpu().numpy()
            for a, i in enumerate(ix):
                out[t, i * DW:(i + 1) * DW] = rows[a]
        base = nd * DW
        cellrow = torch.cat([runoff[:, None], disch[:, None], ring], 1).cpu().numpy()
        out[t, base:] = cellrow.ravel()
    return out


def compare_run(oracle, sc, exact_dev=('cpu',)):
    dm = sc['dams']
    nd, nc, nt, lr = len(dm), sc['nc'], sc['nt'], sc['lr']
    ref = run(oracle, pack_scenario(sc)).reshape(nt, -1)
    DW = 12 + 180 + 24
    assert ref.shape[1] == nd * DW + nc * (2 + lr)
    for dev in exact_dev:
        got = run_python(sc, True, dev)
        dd = got[:, :nd * DW].reshape(nt, nd, DW)
        rr = ref[:, :nd * DW].reshape(nt, nd, DW)
        # discrete flags first: exact
        for col, name in ((0, 'active'), (6, 'op_month'), (7, 'months_running'),
                          (11, 'register_steps')):
            np.testing.assert_array_equal(dd[:, :, col], rr[:, :, col], name)
        np.testing.assert_allclose(dd, rr, **RT)
        np.testing.assert_allclose(got[:, nd * DW:], ref[:, nd * DW:], **RT)
    return ref, nd, DW


def test_run_l1_mixed_dams(oracle):
    sc = scenario(11)
    ref, nd, DW = compare_run(oracle, sc)
    r = ref[:, :nd * DW].reshape(len(ref), nd, DW)
    # coverage: dams of both types, activation inside the run, a plan after
    # a full year of history, op_month found, spill and a hit of both bounds
    assert set(sc['dams'][:, 1]) == {0., 1.}
    act = r[:, :, 0]
    assert (act[0] == 0).any() and (act[-1] == 1).any()
    gl = sc['dams'][:, 1] == 1
    assert (r[-1, gl, 7] > 12).all()
    assert ((r[-1, gl, 6] >= 1) & (r[-1, gl, 6] <= 12)).any()
    assert (r[:, gl, 4] > 0).any()
    assert (r[-1, gl, 12 + 180:12 + 192] != 0).any()      # a plan was made
    assert (r[:, ~gl, 4] > 0).any()                        # local release


@pytest.mark.parametrize('seed,kw', [
    (12, dict(spd=2, rspd=4, nt=2 * 800)),                 # sub-daily steps
    (13, dict(spd=1, rspd=1, nt=1100, m0=1, y0=2003)),     # one ring bin, Jan start
    (14, dict(spd=1, rspd=8, nt=900, y0=2007, m0=12, d0=20)),   # leap year, Dec start
    (15, dict(cal='noleap', nt=1100, m0=2, d0=27)),
    (16, dict(cal='proleptic_gregorian', y0=1899, m0=1, d0=1, nt=900)),
    (17, dict(alpha=0.4, beta=0.3, gamma=2.0, nt=1200)),   # alpha < 0.5
    (18, dict(alpha=0.7, beta=1.5, gamma=0.5, wu=False, nt=1200)),
    (19, dict(efr=False, nt=1200, dams_per_cell=1)),
    (20, dict(nc=9, nt=1500, dams_per_cell=3, m0=3)),
])
def test_run_l1_variants(oracle, seed, kw):
    compare_run(oracle, scenario(seed, **kw))


def test_run_l1_first_year_has_no_release(oracle):
    """Start in July: the first plan (January, no full year of history) has
    an empty climatology, c = inf and a zero release for a year."""
    sc = scenario(21, m0=7, y0=2003, nt=800, dams_per_cell=2)
    sc['dams'][:, 2] = 1990                       # all built
    ref, nd, DW = compare_run(oracle, sc)
    r = ref[:, :nd * DW].reshape(len(ref), nd, DW)
    gl = np.where(sc['dams'][:, 1] == 1)[0]
    assert len(gl)
    # op_release is 0 in the first January, non-zero a year later
    jan1 = 31 + 31 + 30 + 31 + 30 + 31 + 31 - 1 + 1     # first step of 2004-01-01
    assert (r[jan1 + 5, gl, 12 + 180:12 + 180 + 12] == 0).all()
    assert (r[-1, gl, 12 + 180:12 + 180 + 12] != 0).any()


# -------------------------------------------------------------- state and misc
def test_state_roundtrip_and_no_host_sync():
    sc = scenario(22, nt=120)
    sc['dams'][:, 2] = 1990
    cal = DM.make_calendar(*zip(*dates(sc)), 'standard', 1)
    d = DM.GlobalDams([50.0, 10.0], [1.0, 0.8], [1990, 1990], exact=True)
    ring = torch.ones(2, 4, dtype=torch.float64) * 5
    disch = ring.sum(1)
    sd0 = d.state_dict()
    for t in range(40):
        ring, disch = d.run(cal.at(t), ring, disch, 86400.0)
    sd1 = d.state_dict()
    d2 = DM.GlobalDams([50.0, 10.0], [1.0, 0.8], [1990, 1990], exact=True)
    d2.load_state_dict(sd0)
    r2 = torch.ones(2, 4, dtype=torch.float64) * 5
    q2 = r2.sum(1)
    for t in range(40):
        r2, q2 = d2.run(cal.at(t), r2, q2, 86400.0)
    for k in DM.STATE:
        assert torch.equal(d2.state_dict()[k], sd1[k]), k
    assert torch.equal(r2, ring)


def test_masked_month_block_equals_host_gated():
    """A step given as a tensor index (the CUDA-graph path, every month
    block runs and is masked) equals the host-gated path bit for bit."""
    sc = scenario(23, nt=900)
    ys, ms = zip(*dates(sc))
    cal = DM.make_calendar(ys, ms, 'standard', 1)

    def go(tensor_step):
        g = DM.GlobalDams([50.0, 10.0, 400.0], [1.0, 0.8, 1.0],
                          [1990, 2004, 2003], exact=True)
        l = DM.LocalDams([5.0, 30.0], [0.8, 1.0], [2003, 2005], exact=True)
        ring = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4)
        disch = ring.sum(1)
        ro = torch.tensor([2.0, 7.0], dtype=torch.float64)
        for t in range(900):
            st = cal.at(torch.tensor(t) if tensor_step else t)
            ring, disch = g.run(st, ring * (1 + 0.3 * np.sin(t / 40.)), disch, 86400.0,
                                demand=torch.full((3,), 0.4, dtype=torch.float64))
            ro = l.run(st, ro * 1.001, torch.full((2,), 3e8, dtype=torch.float64))
        return g.state_dict(), l.state_dict(), ring, ro

    a, b = go(False), go(True)
    for k in DM.STATE:
        assert torch.equal(a[0][k], b[0][k]), k
        assert torch.equal(a[1][k], b[1][k]), k
    assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
    assert (a[0]['months_running'] > 12).all()


def test_mass_identity():
    """storage change = inflow - release for every active step, and the dam
    never goes below 0 or above capacity."""
    d = DM.GlobalDams(f64(np.array([5.0, 200.0, 40.0])), f64(np.ones(3)),
                      [1990] * 3, exact=True)
    g = torch.Generator().manual_seed(5)
    ys, ms = zip(*[(2000 + (t // 365), 1 + (t % 365) // 31 % 12) for t in range(1500)])
    cal = DM.make_calendar(ys, ms, 'standard', 1)
    ring = torch.rand(3, 4, generator=g, dtype=torch.float64) * 30
    disch = ring.sum(1)
    for t in range(1500):
        s0 = d.storage.clone()
        ring = (torch.rand(3, 4, generator=g, dtype=torch.float64) * 60)
        disch = ring.sum(1)
        ring, disch = d.run(cal.at(t), ring, disch, 86400.0)
        assert (d.storage >= 0).all() and (d.storage <= d.capacity).all()
        a = d.active
        resid = d.storage - s0 - d.inflow + d.release
        assert (resid[a].abs() <= 1e-12).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_run_matches_cpu_without_host_sync(oracle):
    """The same run on CUDA agrees with the C reference to 1e-12 and a
    step does no host read (sync debug mode raises on any)."""
    sc = scenario(31, nt=900)
    dm = sc['dams']
    ref = run(oracle, pack_scenario(sc)).reshape(sc['nt'], -1)
    got = []
    for tensor_step in (False, True):        # host-gated and masked month block
        got.append(run_python_nosync(sc, tensor_step))
    assert np.array_equal(got[0], got[1])
    got = got[0]
    nd, nc, nt, lr = len(dm), sc['nc'], sc['nt'], sc['lr']
    DW = 12 + 180 + 24
    np.testing.assert_allclose(got[:, :nd * DW], ref[:, :nd * DW], **RT)
    np.testing.assert_allclose(got[:, nd * DW:], ref[:, nd * DW:], **RT)


def run_python_nosync(sc, tensor_step=False):
    """run_python on CUDA, with the per-step reads moved after the loop: the
    step itself must not synchronise."""
    import types
    dev = 'cuda'
    dm = sc['dams']
    nd, nc, nt, lr = len(dm), sc['nc'], sc['nt'], sc['lr']
    n = sc['rspd'] // sc['spd']
    dt = 86400.0 / sc['spd']
    ys, ms = zip(*dates(sc))
    cal = DM.make_calendar(ys, ms, sc['cal'], sc['spd'], device=dev)
    cell = dm[:, 0].astype(int)
    sets = []
    for ty, cls in ((0, DM.LocalDams), (1, DM.GlobalDams)):
        sel_t = np.where(dm[:, 1] == ty)[0]
        rk = DM.rank_in_cell(cell[sel_t])
        for r in range(rk.max() + 1 if len(sel_t) else 0):
            ix = sel_t[rk == r]
            d = cls(dm[ix, 3], dm[ix, 4], dm[ix, 2].astype(int),
                    alpha=sc['alpha'], beta=sc['beta'], gamma=sc['gamma'],
                    steps_per_day=sc['spd'], exact=True, device=dev)
            svc = torch.as_tensor(dm[ix, 6:6 + sc['maxs']].astype(np.int64)).to(dev)
            frac = f64(dm[ix, 6 + sc['maxs']:6 + 2 * sc['maxs']]).to(dev)
            nsv = torch.as_tensor(dm[ix, 5].astype(np.int64))
            valid = (torch.arange(sc['maxs'])[None, :] < nsv[:, None]).to(dev)
            sets.append((ty, ix, d, torch.as_tensor(cell[ix]).to(dev), svc, frac,
                         valid))
    area = f64(sc['area']).to(dev)
    act = (torch.arange(sc['ns'])[None, :]
           < torch.as_tensor(sc['ns_active'])[:, None]).to(dev)
    R, Dc, Rg, Q, DS = (torch.as_tensor(sc[k]).to(dev, torch.float64)
                        for k in ('runoff', 'disch', 'ring', 'efr_q', 'ds'))
    hist = []
    steps = torch.arange(nt, device=dev)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')      # the step loop only
    try:
        hist = _step_loop(sc, sets, cal, steps, area, act, R, Dc, Rg, Q, DS, n,
                          dt, tensor_step)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    DW = 12 + 180 + 24
    out = np.zeros((nt, nd * DW + nc * (2 + lr)))
    for t, (rows, cellrow) in enumerate(hist):
        for (ty, ix, d, cl, svc, frac, valid), rw in zip(sets, rows):
            rw = rw.cpu().numpy()
            for a, i in enumerate(ix):
                out[t, i * DW:(i + 1) * DW] = rw[a]
        out[t, nd * DW:] = cellrow.cpu().numpy().ravel()
    return out


def _step_loop(sc, sets, cal, steps, area, act, R, Dc, Rg, Q, DS, n, dt,
               tensor_step):
    hist = []
    nt = sc['nt']
    for t in range(nt):
        st = cal.at(steps[t:t + 1] if tensor_step else t)
        runoff, disch, ring = R[t], Dc[t], Rg[t].clone()
        for ty, ix, d, cl, svc, frac, valid in sets:
            if ty == 0:
                runoff = runoff.index_put((cl,), d.run(st, runoff[cl], area[cl]))
        for ty, ix, d, cl, svc, frac, valid in sets:
            if ty == 1:
                dem = DM.service_demand(DS[t], act, svc, frac, valid, area[cl]) \
                    if sc['wu'] else None
                q = Q[t][cl] if sc['efr'] else None
                rn, dn = d.run(st, ring[cl][:, :n], disch[cl], dt, dem, q)
                ring = ring.index_put((cl,), torch.cat([rn, ring[cl][:, n:]], 1))
                disch = disch.index_put((cl,), dn)
        rows = []
        for ty, ix, d, cl, svc, frac, valid in sets:
            rows.append(torch.cat([
                d.active.to(torch.float64)[:, None], d.inflow[:, None],
                d.demand[:, None], d.efr[:, None], d.release[:, None],
                d.storage[:, None], d.op_month.to(torch.float64)[:, None],
                d.months_running.to(torch.float64)[:, None],
                d.total_inflow[:, None], d.total_demand[:, None],
                d.total_efr[:, None], d.register_steps.to(torch.float64)[:, None],
                d.history_inflow, d.history_demand, d.history_efr,
                d.op_release, d.op_storage], 1))
        hist.append((rows, torch.cat([runoff[:, None], disch[:, None], ring], 1)))
    return hist
