# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Replay of the R1 G-runs G1-G8 through the VIC-WUR connectors
(docs/connectors/vicwur.md).

Each run (tools/vicwur/gruns.py configuration) was made by the dump build
<data root>/build/vicwur3_dump (VICWUR_DUMP; tools/vicwur/replay_runs.py) and its
outputs equal the vicwur3 runs bit for bit.  tests/vicwur_l2_replay.py runs
the VIC-WUR connector phase group of the run (routing, dams, local and remote
water use, non-renewable groundwater, EFR: those the run switched on) through
the coupling runtime on R1, driven day by day with the dumped soil state, tile
Cv, VIC runoff and baseflow and the plugin forcing (what the C had).  Every
step it compares

  * routing: runoff, inflow, stream, deficit, discharge and the whole ring
    (dump tag 4), and OUT_DISCHARGE, OUT_STREAM_*, OUT_NONREN_DEFICIT of the
    history files;
  * dams: every field of every dam (tag 7: the discrete state exactly) and
    OUT_GDAM_* / OUT_LDAM_*;
  * water use: all 22 wu_var fields of every sector (tag 5), and the OUT_DE_*,
    OUT_WI_*, OUT_DEMAND, OUT_WITHDRAWN, OUT_RETURNED, OUT_CONSUMED files;
  * the soil the plugins left: bottom-layer moisture of every tile and band
    after the connectors pumped it, against the dump;
  * EFR: OUT_EFR_MOIST, OUT_EFR_BASEFLOW, OUT_EFR_DISCHARGE;
  * the end state against the VIC-WUR 3.0 state file (STATE_DISCHARGE_DT,
    STATE_NONRENEW_DEFICIT).

Tolerances are those of the data.  The dump is float64: routing, dams, water
use and soil agree to rtol 1e-12 (observed <= 1e-13); the routing of G1a, G1b
and G2 is bitwise on the CPU (the CUDA kernels differ by rounding).  The
history files are float32: compared after rounding to float32 (one ulp).
VIC's runoff and baseflow (not in the dump) come from the run's own
OUT_RUNOFF and OUT_BASEFLOW, written as float64 for this purpose
(tools/vicwur/replay_runs.py): in float32 their rounding is paid back from the
non-renewable deficit, accumulates, and flips a dam's plan on a 1e-36 inflow.

G4 (irrigation) and G8 (land use): here they are covered by their
kernel-level replay (tests/test_vicwur_irrigation_l2.py,
tests/test_vicwur_landuse_l2.py, run below); the connector-level replay of
the two is not implemented and its two tests are skipped.  G5-G7 carry
irrigation in their configuration: the IRR sector's demand, groundwater and
consumption fractions are the irrigation kernel's irr_set_demand on the dumped
requirement (checked against the dumped demand to 1e-9).
"""
import os

import netCDF4
import numpy as np
import pytest

torch = pytest.importorskip('torch')

import vicwur_l2_replay as RP                                    # noqa: E402

RUNS = ['G1a', 'G1b', 'G2', 'G3', 'G5', 'G6a', 'G6b', 'G6c', 'G7']
#: runs whose routing is bitwise on CPU
BITWISE_ROUTING = ('G1a', 'G1b', 'G2')
CUDA = torch.cuda.is_available()
DEVICES = ['cpu', pytest.param('cuda', marks=[
    pytest.mark.gpu, pytest.mark.skipif(not CUDA, reason='no CUDA')])]
TOL = dict(rtol=1e-12, atol=1e-12)
F32 = 2.0 ** -23                       # one float32 ulp, relative
DAY = RP.DAY


def _need(run):
    if not RP.available(run):
        pytest.skip('dump run absent or failed: ' + RP.run_dir(run))


def _close(a, b, what, tol=TOL, stats=None):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    np.testing.assert_allclose(a, b, err_msg=what, **tol)
    if stats is not None:
        stats[what] = float(np.max(np.abs(a - b))) if a.size else 0.0


def _same(a, b, what, stats=None):
    assert np.array_equal(a, b), '%s: not bitwise (max %.3g)' % (
        what, np.max(np.abs(a - b)))
    if stats is not None:
        stats[what] = 0.0


def _f32(a, b, what, stats=None):
    """History file b (float32) against the float64 value a: one ulp (and the
    1e-12 of TOL: residues below it are not data)."""
    a32 = np.asarray(a, np.float64).astype(np.float32).astype(np.float64)
    b = np.asarray(b, np.float64)
    np.testing.assert_allclose(a32, b, rtol=F32, atol=1e-12, err_msg=what)
    if stats is not None:
        stats[what] = float(np.max(np.abs(a32 - b))) if b.size else 0.0


def _stack(rec, key, T):
    return np.stack([rec[d][key] for d in range(T)])


def compare(case, rec, device, stats):
    c, T, N = case.c, case.T, case.N
    grid, rd = case.grid, case.rd
    cpu = device == 'cpu'
    bit = cpu and case.run in BITWISE_ROUTING
    assert sorted(rec) == list(range(T))
    # ---- routing (tag 4) ---------------------------------------------
    rout = _stack(rec, 'rout', T)
    ref = case.rout
    for k, nm in enumerate(['runoff', 'inflow', 'stream', 'deficit',
                            'discharge']):
        if nm == 'runoff':
            _same(rout[:, :, k], ref[:, :, k], nm, stats)   # the C's
        elif bit:
            _same(rout[:, :, k], ref[:, :, k], nm, stats)
        else:
            _close(rout[:, :, k], ref[:, :, k], nm, TOL, stats)
    if bit:
        _same(rout[:, :, 5:], ref[:, :, 5:], 'ring', stats)
    else:
        _close(rout[:, :, 5:], ref[:, :, 5:], 'ring', TOL, stats)
    # history files
    sm = rout[:, :, 2] * DAY / grid.area_active * 1000.0
    for var, val in (('OUT_DISCHARGE', rout[:, :, 4]),
                     ('OUT_STREAM_INFLOW', rout[:, :, 1]),
                     ('OUT_STREAM_RUNOFF', rout[:, :, 0]),
                     ('OUT_STREAM_MOIST', sm),
                     ('OUT_NONREN_DEFICIT', rout[:, :, 3])):
        if var == 'OUT_NONREN_DEFICIT' and not c['nonren']:
            continue
        _f32(val, RP.read_out(rd, var, grid), var, stats)
    # ---- soil left by the plugins ---------------------------------------
    mo = _stack(rec, 'moist', T)
    _close(mo, case.moist[..., -1], 'bottom-layer moisture', TOL, stats)
    # ---- dams (tag 7) -----------------------------------------------------
    if c['dams']:
        compare_dams(case, rec, T, stats)
    # ---- water use (tag 5) --------------------------------------------------
    if c['wu']:
        compare_wu(case, rec, T, stats)
    # ---- EFR ----------------------------------------------------------------
    if c['efr']:
        e = _stack(rec, 'efr', T)                     # [T, N, 3]
        o_m = RP.read_out(rd, 'OUT_EFR_MOIST', grid)
        o_b = RP.read_out(rd, 'OUT_EFR_BASEFLOW', grid)
        o_q = RP.read_out(rd, 'OUT_EFR_DISCHARGE', grid)
        _f32(e[:, :, 0], o_m, 'OUT_EFR_MOIST', stats)
        _f32(e[:, :, 1], o_b, 'OUT_EFR_BASEFLOW', stats)
        _f32(e[:, :, 2], o_q, 'OUT_EFR_DISCHARGE', stats)
    return stats


def compare_dams(case, rec, T, stats):
    dp = case.inp.dams
    cell, glb = dp.dam_cell.numpy(), dp.is_global.numpy()
    nd = len(cell)
    from torchvic.vicwur import dams as DM
    pos = DM.rank_in_cell(cell)
    out = _stack(rec, 'dam', T)                         # [T, nd, W]
    rows = np.stack([case.dam[:, cell[i], pos[i]] for i in range(nd)], 1)
    for col, nm in ((0, 'active'), (7, 'months_running')):
        _same(out[:, :, col], rows[:, :, col], 'dam ' + nm, stats)
    # the operational month: exact wherever the inflow it is made of is not a
    # rounding residue (DumpVic.set_op_month)
    yr = np.minimum(rows[:, :, 7].astype(int) // 12, 5)
    hist = rows[:, :, 11:71]
    used = np.arange(60)[None, None, :] < (12 * yr)[:, :, None]
    resid = (np.where(used, np.abs(hist), 0.0).max(2) < 1e-25)
    flip = out[:, :, 6] != rows[:, :, 6]
    assert not (flip & ~resid).any(), 'dam op_month differs on a real inflow'
    stats['op_month_residue_flips'] = int(flip.sum())
    names = ['active', 'inflow', 'demand', 'efr', 'release', 'storage',
             'op_month', 'months_running', 'total_inflow', 'total_demand',
             'total_efr']
    for k, nm in enumerate(names):
        if nm != 'op_month':
            _close(out[:, :, k], rows[:, :, k], 'dam ' + nm, TOL, stats)
    for a, b, nm in ((11, 71, 'history_inflow'), (71, 131, 'history_demand'),
                     (131, 191, 'history_efr'), (191, 203, 'op_release'),
                     (203, 215, 'op_storage')):
        _close(out[:, :, a:b], rows[:, :, a:b], 'dam ' + nm, TOL, stats)
    # history files, OUT_GDAM_* / OUT_LDAM_*
    hist = {n: np.stack([[rec[d]['dam_' + n][i] for i in range(nd)]
                         for d in range(T)])
            for n in ('hist_inflow', 'hist_demand', 'hist_efr')}
    cols = dict(INFLOW=1, DEMAND=2, EFR=3, STORAGE=5, RELEASE=4,
                OP_RELEASE=191, OP_STORAGE=203)
    for i in range(nd):
        P_ = 'G' if glb[i] else 'L'
        for nm, k in cols.items():
            f = RP.read_out(case.rd, 'OUT_%sDAM_%s' % (P_, nm), case.grid)
            _f32(out[:, i, k], _dam_col(f, pos[i], cell[i]),
                 'OUT_%sDAM_%s' % (P_, nm))
        for nm in ('INFLOW', 'DEMAND', 'EFR'):
            f = RP.read_out(case.rd, 'OUT_%sDAM_HIST_%s' % (P_, nm),
                            case.grid)
            _f32(hist['hist_' + nm.lower()][:, i],
                 _dam_col(f, pos[i], cell[i]),
                 'OUT_%sDAM_HIST_%s' % (P_, nm))


def _dam_col(f, j, c):
    """One dam's column of an OUT_xDAM_* file [T, (class,) cell]."""
    return f[:, c] if f.ndim == 2 else f[:, j, c]


WU_OUT = {'OUT_DE_GW_SECT': 6, 'OUT_DE_SURF_SECT': 7, 'OUT_DE_NREN_SECT': 10,
          'OUT_DE_REM_SECT': 9, 'OUT_DE_TREM_SECT': 8, 'OUT_WI_GW_SECT': 11,
          'OUT_WI_SURF_SECT': 12, 'OUT_WI_DAM_SECT': 13,
          'OUT_WI_REM_SECT': 15, 'OUT_WI_TREM_SECT': 14,
          'OUT_WI_NREN_SECT': 16}


def compare_wu(case, rec, T, stats):
    wu = _stack(rec, 'wu', T)                          # [T, N, 5, 22]
    ref = case.wu
    act = ~np.isnan(ref[0, :, :, 0])
    a = np.where(act[None, :, :, None], wu, 0.0)
    b = np.nan_to_num(ref)
    for j, nm in enumerate(RP.WUF):
        _close(a[..., j], b[..., j], 'wu_var ' + nm, TOL, stats)
    # history files, in the C's order of summation
    g, rd = case.grid, case.rd
    tot = dict(OUT_DEMAND=np.zeros((T, case.N)),
               OUT_WITHDRAWN=np.zeros((T, case.N)),
               OUT_RETURNED=np.zeros((T, case.N)),
               OUT_CONSUMED=np.zeros((T, case.N)))
    for s in range(5):
        if not case.active[s]:
            continue
        x = a[:, :, s]
        tot['OUT_DEMAND'] += x[..., 6] + x[..., 7]
        tot['OUT_WITHDRAWN'] += (x[..., 11] + x[..., 12] + x[..., 13]
                                 + x[..., 14] + x[..., 16])
        tot['OUT_RETURNED'] += x[..., 17]
        tot['OUT_CONSUMED'] += x[..., 18]
    for var, j in WU_OUT.items():
        f = RP.read_out(rd, var, g)                     # [T, 5, N]
        for s in np.where(case.active)[0]:
            _f32(a[:, :, s, j], f[:, s], var)
    for var, x in tot.items():
        _f32(x, RP.read_out(rd, var, g), var)


def final_state(case, m, stats):
    """The end state of the connectors against the VIC-WUR 3.0 state file."""
    import glob
    r = m.runtime.connectors['vicwur_routing'].router
    st = r.state()
    f = sorted(glob.glob(case.rd + '/W3/state/states.*.nc'))[0]
    with netCDF4.Dataset(f) as ds:
        ds.set_auto_mask(False)
        ring = case.grid.take(np.asarray(ds['STATE_DISCHARGE_DT'][:]))
        dfc = case.grid.take(np.asarray(ds['STATE_NONRENEW_DEFICIT'][:]))
    ring = ring.reshape(ring.shape[0], -1).T                  # [N, R]
    _close(st['discharge_dt'].cpu().numpy(), ring, 'STATE_DISCHARGE_DT',
           TOL, stats)
    _close(st['nonrenew_deficit'].cpu().numpy(), dfc,
           'STATE_NONRENEW_DEFICIT', TOL, stats)


def coverage(case, stats):
    """The run exercised what it is meant to (not a replay of zeros)."""
    c = case.c
    if c['dams']:
        rows = case.dam
        assert (rows[..., 0] > 0).any() and (rows[..., 4] > 0).any()
    if c['wu']:
        w = case.wu
        assert (np.nan_to_num(w[..., 11]) > 0).sum() > 100       # pumping
        assert (np.nan_to_num(w[..., 12]) > 0).sum() > 100       # river
        stats['pumped_cell_steps'] = int((np.nansum(w[..., 11], 2) > 0).sum())
    if c['remote']:
        assert (np.nan_to_num(w[..., 15]) > 0).sum() > 100
    if c['nonren']:
        assert (np.nan_to_num(w[..., 16]) > 0).sum() > 100
        assert case.rout[:, :, 3].max() > 0
    if c['efr']:
        assert case.rout.shape[0] == case.T


def replay_run(run, device):
    _need(run)
    case = RP.Case(run)
    rec, m = RP.replay(case, device)
    stats = {}
    try:
        coverage(case, stats)
        compare(case, rec, device, stats)
        final_state(case, m, stats)
    finally:
        print(run, device, {k: float('%.3g' % v) if isinstance(v, float)
                            else v for k, v in sorted(stats.items())})
    return stats


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('run', RUNS)
def test_l2_replay(run, device):
    replay_run(run, device)


# --------------------------------------------------- G4 and G8: kernel level
def test_g4_irrigation_kernel_level():
    """G4 (irrigation): the irrigation kernels' replay."""
    import test_vicwur_irrigation_l2 as IL
    IL.test_l2_replay('G4_dump')


def test_g8_landuse_kernel_level():
    """G8 (land use): the land-use kernel's replay."""
    import test_vicwur_landuse_l2 as LL
    LL.test_l2_replay('G8')


@pytest.mark.skip(reason='connector-level replay of the irrigation '
                         'connector (G4) is not implemented')
def test_g4_connector_level():
    pass


@pytest.mark.skip(reason='connector-level replay of the land-use '
                         'connector (G8) is not implemented')
def test_g8_connector_level():
    pass
