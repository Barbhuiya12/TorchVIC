# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.dams replayed on the R1 G-runs with dams (VICWUR_DUMP).

R1 has 3 dams: two global (cells 1 and 2 of the 2 x 3 grid, built 2003 and
2005) and one local (built 2004).  The C run (dump build, <data root>/build/
vicwur3_dump, cpu partition) writes dam_var every step: active, inflow,
demand, efr, release, storage, op_month, months_running, the monthly totals,
register_steps, the three 60-month histories, op_release and op_storage.

The replay feeds the kernels only what the C had as input, and compares every
one of those fields, every dam and step, to 1e-9 (relative and absolute):
  * the date of the step (the dump's own year / month);
  * the inflow: the C reads it from the routing ring or OUT_RUNOFF, which the
    dump does not hold before the dam.  An active dam's inflow is dumped; an
    inactive dam zeroes it after registering, but its monthly total shows it
    (total_inflow of this step minus the last, or the total at a month
    change).  That volume gives the dam cell's pre-dam discharge (the
    first dam of a cell) and its runoff (a local dam); a second dam of the
    cell gets its inflow from the first one's write-back (compounding), so
    its inflow is a check, not an input;
  * the dam demand: service_demand of the dumped wu_var demand_surf of the
    service cells (G4, G7, G10);  it must equal the demand the C registered;
  * the EFR: efr_q of the EFR forcing files (G7, G10) through dam_efr;
  * the water the sectors draw from a local dam after its operation
    (withdrawn_dam of the dump, through wateruse.dam_withdraw).
Runs without water use or EFR also check the write-back: the cell discharge
the C leaves after the dam (the dump's rout_var.discharge) is
q (1 - frac) + R 1e6 / dt, and a local dam's OUT_RUNOFF (netCDF, float32)
runoff (1 - frac) + R 1e6 1000 / area.  Runs: G2d (the default alpha 0.85,
beta 0.6, gamma 5), G2a (alpha 0.7, beta 0.3, gamma 2, a constants file), and
the dams inside G4 (water use and irrigation), G7 (+ remote, non-renewable
and EFR) and G10 (everything).
"""
import glob
import os
import _paths  # noqa: E402
import struct

import netCDF4
import numpy as np
import pytest
import torch

from torchvic.vicwur import dams as DM
from torchvic.vicwur import efr as E
from torchvic.vicwur import forcing as F
from torchvic.vicwur import params as P
from torchvic.vicwur import wateruse as WU

V = _paths.data('verification')
RUNS = {
    'G2d': (os.environ.get('VICWUR_17_GRUNS', V + '/vicwur_17/gruns') + '/G2d'),
    'G2a': (os.environ.get('VICWUR_17_GRUNS', V + '/vicwur_17/gruns') + '/G2a'),
    'G4': V + '/vicwur_12/gruns/G4_dump',
    'G7': V + '/vicwur_16/gruns/G7_dump',
    'G10': V + '/vicwur_03/gruns/R1/G10_dump',
}
START = (2003, 7, 1, 0)
TOL = dict(rtol=1e-9, atol=1e-9)
f64 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)


def read_dump(path):
    out = {1: [], 4: [], 5: [], 7: []}
    dims = None
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            tag, n = struct.unpack('<iq', h)
            if tag in out or tag == 0:
                a = np.frombuffer(f.read(8 * n), '<f8')
                if tag == 0:
                    dims = a.astype(int)
                else:
                    out[tag].append(a.copy())
            else:
                f.seek(8 * n, 1)               # the soil, Cv and irr records
    return dims, {k: np.stack(v) for k, v in out.items() if v}


def read_constants(g):
    c = dict(DAM_ALPHA=0.85, DAM_BETA=0.6, DAM_GAMMA=5.0)
    if 'CONSTANTS' in g:
        with open(g['CONSTANTS'][0][0]) as fh:
            for ln in fh:
                p = ln.split()
                if len(p) >= 2 and p[0] in c:
                    c[p[0]] = float(p[1])
    return c


def read_out(rd, var, grid):
    parts = []
    for p in sorted(glob.glob(rd + '/W3/out/fluxes.*.nc')):
        with netCDF4.Dataset(p) as ds:                  # closed at once
            parts.append(np.asarray(ds[var][:], dtype=np.float64))
    return grid.take(np.concatenate(parts))


def efr_series(g, T):
    """efr_q [T,N] m3/s as the C reads it (MONTH forcing, held)."""
    fq = [x for x in g['PLUGIN_FORCE_TYPE'] if x[0] == 'EFR_DISCHARGE'][0]
    ms = F.file_info(fq[3] + '%4d.nc' % START[0], fq[1])['start']
    fq_ = F.PluginForcing(fq[3], fq[1], 'MONTH', START, T, 1, met_start=ms)
    try:
        return np.stack([fq_.get(i).numpy() for i in range(T)])
    finally:
        fq_.close()


def replay(run):
    rd = RUNS[run]
    dump = os.path.join(rd, 'dump.bin')
    if not os.path.exists(dump):
        pytest.skip('dump absent: ' + dump)
    g = P.parse_global(os.path.join(rd, 'global_W3.txt'))
    grid = P.Grid(g['DOMAIN'][0][0], g['PARAMETERS'][0][0])
    dp = P.read_dams(g['DAMS_PARAMETERS'][0][0], grid)
    cst = read_constants(g)
    dims, rec = read_dump(dump)
    N, ring_len, ndmax, hist, nw = (int(dims[0]), int(dims[5]), int(dims[7]),
                                    int(dims[8]), int(dims[14]))
    wu_on, efr_on = bool(dims[10]), 'EFR_DISCHARGE' in {
        x[0] for x in g.get('PLUGIN_FORCE_TYPE', [])}
    T = rec[1].shape[0]
    W = 11 + 3 * hist + 24                   # no register_steps in the dump
    dv = rec[7].reshape(T, N, ndmax, W)
    ys, ms = rec[1][:, 1].astype(int), rec[1][:, 2].astype(int)
    spd = 1
    dt = 86400.0 / spd
    n = 4 // spd                         # ROUT_STEPS_PER_DAY 4 in every run
    cal = DM.make_calendar(ys, ms, 'proleptic_gregorian', spd)
    area = f64(grid.area_active)
    cell = dp.dam_cell.numpy()
    gl = dp.is_global.numpy()
    nd = dp.n_dams
    # the position of every dam inside its cell (the dump's j index)
    pos = DM.rank_in_cell(cell)
    sets = []
    for ty, cls in ((0, DM.LocalDams), (1, DM.GlobalDams)):
        sel_t = np.where(gl == bool(ty))[0]
        rk = DM.rank_in_cell(cell[sel_t])
        for r in range(rk.max() + 1 if len(sel_t) else 0):
            ix = sel_t[rk == r]
            d = cls(dp.capacity[ix], dp.inflow_fraction[ix], dp.year[ix],
                    alpha=cst['DAM_ALPHA'], beta=cst['DAM_BETA'],
                    gamma=cst['DAM_GAMMA'], steps_per_day=spd, exact=True)
            sets.append((ty, r, ix, d))
    frac = dp.inflow_fraction.numpy()
    svc = dp.service.clamp(min=0)
    valid = dp.service >= 0
    sfrac = dp.service_frac.clamp(min=0)
    act_sector = None
    wu = None
    if wu_on:
        wu = rec[5].reshape(T, N, nw, 22)
    q_efr = efr_series(g, T) if efr_on else None
    # C-side per-dam rows [T, nd, W]
    rows = np.stack([dv[:, cell[i], pos[i]] for i in range(nd)], 1)
    act, infl, tot_in = rows[:, :, 0], rows[:, :, 1], rows[:, :, 8]
    newm = np.zeros(T, bool)
    newm[1:] = ms[1:] != ms[:-1]
    # inflow of each step: dumped when active, else the monthly-total step
    prev = np.zeros_like(tot_in)
    prev[1:] = tot_in[:-1]
    prev[newm] = 0.0
    inflow = np.where(act > 0, infl, tot_in - prev)          # global dams
    # pre-dam discharge of a cell (m3/s) and local runoff (mm) from rank 0
    q_pre = np.zeros((T, N))
    ro_pre = np.zeros((T, N))
    for i in range(nd):
        if not _first_of_type(i, cell, gl):
            continue
        c = cell[i]
        if gl[i]:
            q_pre[:, c] = inflow[:, i] * 1e6 / (dt * frac[i])
        else:
            ro_pre[:, c] = infl[:, i] * 1e6 * 1000.0 / (grid.area_active[c]
                                                         * frac[i])
    # ----- replay
    out = np.zeros((T, nd, W))
    comp = np.zeros((T, N))                   # post-dam discharge
    ro_post = np.zeros((T, N))
    for t in range(T):
        st = cal.at(t)
        ring = (f64(q_pre[t]) / n)[:, None].expand(N, n).clone()
        disch = f64(q_pre[t]).clone()
        runoff = f64(ro_pre[t]).clone()
        dem_all = None
        if wu_on:
            ds = np.nan_to_num(wu[t][:, :, 7])
            sa = ~np.isnan(wu[t][:, :, 7])
            dem_t = (f64(ds), torch.as_tensor(sa))
        for ty, r, ix, d in sets:
            ixt = torch.as_tensor(ix)
            cl = dp.dam_cell[ixt]
            if ty == 0:
                new = d.run(st, runoff[cl], area[cl])
                runoff = runoff.index_put((cl,), new)
                if wu_on:
                    # wu_run_local then draws the sectors' dam water from
                    # the storage (wateruse.py), after this
                    # step's dam operation
                    wd = f64(np.nansum(wu[t][:, :, 13], 1))[cl]
                    on = d.active[:, None]
                    loc = torch.ones_like(on)
                    av, a = WU.availability_dam(d.storage[:, None], on, loc,
                                               area[cl])
                    d.storage = WU.dam_withdraw(d.storage[:, None], av, a, wd,
                                               on, loc, area[cl])[:, 0]
            else:
                dem = None
                if wu_on:
                    dem = DM.service_demand(dem_t[0], dem_t[1], svc[ixt],
                                            sfrac[ixt], valid[ixt], area[cl])
                q = f64(q_efr[t])[cl] if efr_on else None
                rn, dn = d.run(st, ring[cl], disch[cl], dt, dem, q)
                ring = ring.index_put((cl,), rn)
                disch = disch.index_put((cl,), dn)
            rw = torch.cat([
                d.active.to(torch.float64)[:, None], d.inflow[:, None],
                d.demand[:, None], d.efr[:, None], d.release[:, None],
                d.storage[:, None], d.op_month.to(torch.float64)[:, None],
                d.months_running.to(torch.float64)[:, None],
                d.total_inflow[:, None], d.total_demand[:, None],
                d.total_efr[:, None],
                d.history_inflow, d.history_demand, d.history_efr,
                d.op_release, d.op_storage], 1).numpy()
            out[t, ix] = rw
        comp[t] = disch.numpy()
        ro_post[t] = runoff.numpy()
    return dict(rd=rd, out=out, rows=rows, rec=rec, comp=comp, ro_post=ro_post,
                grid=grid, dp=dp, cell=cell, gl=gl, wu_on=wu_on, efr_on=efr_on,
                cst=cst, T=T, N=N, ring_len=ring_len, frac=frac,
                q_pre=q_pre, ro_pre=ro_pre, dt=dt, n=n, pos=pos)


def _first_of_type(i, cell, gl):
    """True when dam i is the first dam of its cell among dams of its type."""
    return not any(cell[j] == cell[i] and gl[j] == gl[i] for j in range(i))


def check(run):
    r = replay(run)
    out, rows = r['out'], r['rows']
    nd = out.shape[1]
    # discrete state is exact
    for col, nm in ((0, 'active'), (6, 'op_month'), (7, 'months_running')):
        np.testing.assert_array_equal(out[:, :, col], rows[:, :, col], nm)
    names = ['active', 'inflow', 'demand', 'efr', 'release', 'storage',
             'op_month', 'months_running', 'total_inflow', 'total_demand',
             'total_efr']
    for c, nm in enumerate(names):
        np.testing.assert_allclose(out[:, :, c], rows[:, :, c], err_msg=nm,
                                   **TOL)
    for a, b, nm in ((11, 71, 'history_inflow'), (71, 131, 'history_demand'),
                     (131, 191, 'history_efr'), (191, 203, 'op_release'),
                     (203, 215, 'op_storage')):
        np.testing.assert_allclose(out[:, :, a:b], rows[:, :, a:b],
                                   err_msg=nm, **TOL)
    # write-back
    rec, T, N, rl = r['rec'], r['T'], r['N'], r['ring_len']
    if not r['wu_on'] and not r['efr_on']:
        q_c = rec[4].reshape(T, N, 5 + rl)[:, :, 4]
        gcells = np.unique(r['cell'][r['gl']])
        np.testing.assert_allclose(r['comp'][:, gcells], q_c[:, gcells],
                                   err_msg='cell discharge after the dams',
                                   **TOL)
        ro = read_out(r['rd'], 'OUT_RUNOFF', r['grid']).reshape(T, N)
        for i in np.where(~r['gl'])[0]:
            c = r['cell'][i]
            on = rows[:, i, 0] > 0      # inactive: the runoff passes through
            assert on.sum() > 100
            np.testing.assert_allclose(
                r['ro_post'][on, c], ro[on, c], rtol=2e-6, atol=1e-5,
                err_msg='OUT_RUNOFF after the local dam')
    err = float(np.max(np.abs(out - rows)))
    st = dict(run=run, steps=T, dams=nd, max_abs_err=err, global_dams=int(r['gl'].sum()),
              active_final=int(rows[-1, :, 0].sum()),
              planned=int((rows[-1, :, 191:203] != 0).any(1).sum()),
              spill_or_empty_steps=int(((rows[:, :, 5] <= 1e-12)
                                        | (rows[:, :, 5] >= r['dp'].capacity.numpy()
                                           - 1e-12)).sum()),
              op_months=sorted(set(rows[-1, :, 6].astype(int).tolist())),
              demand_steps=int((rows[:, :, 2] > 0).sum()),
              efr_steps=int((rows[:, :, 3] > 0).sum()))
    print(st)
    return r, st


@pytest.mark.parametrize('run', sorted(RUNS))
def test_l2_replay(run):
    r, st = check(run)
    assert st['dams'] == 3 and st['global_dams'] == 2
    assert st['active_final'] == 3 and st['planned'] >= 1
    if run in ('G4', 'G7', 'G10'):
        assert st['demand_steps'] > 0
    if run in ('G7', 'G10'):
        assert st['efr_steps'] > 0
