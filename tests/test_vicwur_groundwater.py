# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR 3.0's groundwater (MODFLOW) coupling through the groundwater
connector's vicwur3 profile (docs/connectors/vicwur.md).

  * the package set (torchvic.vicwur.groundwater) equals what VIC-WUR 3.0's
    own config_module.py builds, bit for bit, in both modes; the
    capillary-rise cascade equals VIC-WUR 3.0's vic_runner.update_statefile,
    bit for bit;
  * synthetic runs through the runtime with routing, local water use and
    non-renewable (all GWM): the ledger of every module closes, the
    exchanges cancel (baseflow = the routing's forced inflow; mode fixed:
    wells = the non-renewable withdrawal), restart equals continuous
    bitwise (mid-month and at a month end), the GPU equals the CPU and does
    no host sync in the phases;
  * replay on R1 (tools/vicwur/gw_runs.py): G11, the C's OUT_GWRECHARGE month
    by month against MF6 6.7.0 run by VIC-WUR 3.0's own mf_run.py; G11c, the
    whole coupled loop (C restarted monthly with the MODFLOW baseflow as
    DISCHARGE forcing and the capillary rise in its state): heads, RIV, DRN,
    EVT, RCH, the forced inflow, the soil edit, the routing and water use of
    every day; restart of the replay; mode fixed (daily, wells) against MF6
    on the replay's own package inputs.
"""
import datetime
import os
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'vicwur'))

from torchvic.coupling import TorchVIC, registry                     # noqa
from torchvic.coupling.connectors import vicwur_common as VC         # noqa
from torchvic.coupling.connectors.groundwater import (               # noqa
    VicWur3GroundwaterConnector)
from torchvic.vicwur import groundwater as W                         # noqa
from torchvic.vicwur.order import Options, plan                      # noqa
import test_vicwur_connectors as TC                                  # noqa
import gw_runs as GW                                                 # noqa
import mf_w3                                                         # noqa

DAY = 86400.0
HAVE_W3 = os.path.isdir(mf_w3.W3MF)
HAVE_R1 = os.path.exists(GW.GR.R1 + '/params.nc')
HAVE_MF6 = os.path.exists(GW.VENV_MF) and os.path.exists(mf_w3.MF6)
if HAVE_R1:
    GW.static_values()      # read before the VIC-WUR 3.0 fixture stubs netCDF4


# ----------------------------------------------------- VIC-WUR 3.0's own code
@pytest.fixture
def w3():
    """VIC-WUR 3.0's config_module and mf_run with their I/O stubbed
    (mf_w3.py), sys.modules and sys.path restored afterwards."""
    if not (HAVE_W3 and HAVE_R1):
        pytest.skip('W3 source or R1 data absent')
    path = list(sys.path)
    with mock.patch.dict(sys.modules):
        for k in ('config_module', 'mf_run', 'vic_runner',
                  'support_function'):
            sys.modules.pop(k, None)
        cm, mr = mf_w3._stubs()
        yield cm, mr
    sys.path[:] = path


def _w3_config(cm, mode='exact'):
    import tempfile
    if mode == 'fixed':
        mf_w3._fix(cm, mode)
    z = dict(GW.static_grid(), foc=True, xres=0.0625, yres=0.0625)
    return cm, mf_w3._config(cm, z, tempfile.mkdtemp())


def _ours(pad=True):
    s = GW.static_grid(pad)
    lm = s['landmask']
    aq = W.aqdepth(lm, s['aqdepth'])
    s['aqdepth_c'] = aq
    tp = W.top(lm, s['dem'])
    return s, aq, tp


def _eq(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    assert ((a == b) | (np.isnan(a) & np.isnan(b))).all(), (a, b)


@pytest.mark.parametrize('mode', ['exact', 'fixed'])
def test_package_set_is_w3s_config_module(w3, mode):
    cm, c = _w3_config(w3[0], mode)
    s, aq, tp = _ours()
    d = 0.0625 * W.DEG_M
    _eq(c.cal_toplayer_elevation(), tp)
    for a, b in zip(c.cal_botlayer_elevation(), W.botm(tp, aq)):
        _eq(a, b)
    kh, kv, st = c.get_npf_param()
    okh, okv, ost = W.npf(s, aq, d, d)
    for x, y in zip(kh + kv + st, okh + okv + ost):
        _eq(x, y)
    # RIV: records (cell, stage, cond, rbot) for a range of discharges,
    # negative ones included
    r = W.riv_static(s, tp)
    on = W.riv_records(r)
    for q in (0.0, 0.5, 7.0, 120.0, -3.0):
        qq = np.where(np.isnan(s['landmask']), 0.0, q)
        qq[0, 1] = 400.0
        c.set_ts_discharge(qq)
        recs = c.get_riv_param()
        st = W.stage_np(r, qq, mode == 'exact')
        mine = [[0, i, j, st[i, j], r['cond'][i, j], r['rbot'][i, j]]
                for i, j in zip(*np.nonzero(on))]
        _eq(recs, mine)
        # the tensor kernel equals the numpy one
        R = {k: torch.as_tensor(r[k][on]) for k in (
            'width', 'slope_used', 'bot_bkfl', 'head_ini', 'rbot')}
        R['riv'] = torch.as_tensor(r['riv'][on])
        _eq(W.stage(R, torch.as_tensor(qq[on]), mode == 'exact').numpy(),
            st[on])
    # DRN, CHD, EVT
    el, cd = W.drn(s, ost[0])
    drn = [[0, i, j, el[i, j], cd[i, j]] for i, j in zip(*np.nonzero(cd))]
    _eq(c.get_drn_param(), drn)
    chd = [[k, i, j, 0] for k in (0, 1)
           for i, j in zip(*np.nonzero(W.chd_cells(s)))]
    _eq(c.get_chd_input(), chd)
    eon, sf, rt, dp = W.evt(s)
    cpr = c.get_cpr_param()
    assert [x[0] for x in cpr] == [(0, i, j) for i, j in zip(
        *np.nonzero(eon))]
    _eq([x[1:] for x in cpr], [[sf[i, j], rt[i, j], dp[i, j]]
                               for i, j in zip(*np.nonzero(eon))])
    # RCH from a month's OUT_GWRECHARGE sum (negative values included)
    # (the ocean column holds no recharge: VIC has no cell there)
    gw = np.array([[3.1, -0.4, 0.0, np.nan], [1e-9, 2.5, 7.0, np.nan]])
    c.set_ts_gwrecharge(gw / 31 / 1000)
    recs = c.get_rch_param(datetime.datetime(2004, 1, 1))
    rate = W.recharge_rate(torch.as_tensor(gw), 31.0,
                           torch.as_tensor(s['cellarea']), d, d,
                           mode == 'exact').numpy()
    rate = np.where(np.isnan(rate), 0.0, rate)
    mine = [[0, i, j, rate[i, j]] for i, j in zip(*np.nonzero(rate))]
    _eq(recs, mine)
    _eq(c.get_initial_head(), W.initial_head(s))


def test_capillary_cascade_is_w3s_update_statefile(w3):
    """vic_runner.update_statefile (with its cpr * Cv, band 0) on a state
    with saturated layers equals capillary_rise(exact) given cpr * Cv per
    column, bit for bit."""
    cm, mr = w3
    import vic_runner as vr
    g = np.random.default_rng(5)
    NV, NL, R, C = 4, 3, 3, 4
    cv = g.dirichlet(np.ones(NV), (R, C)).transpose(2, 0, 1)
    mx = np.stack([np.full((R, C), 30.0), np.full((R, C), 300.0),
                   np.full((R, C), 400.0)])
    sm = np.stack([g.uniform(20, 29.9, (NV, R, C)), g.uniform(285, 299,
                                                              (NV, R, C)),
                   g.uniform(330, 399, (NV, R, C))], 1)[:, None]
    cpr = g.uniform(0, 400, (R, C))
    store = {'STATE_SOIL_MOISTURE': sm.copy()}

    class Var(object):
        def __init__(self, a):
            self.a = a

        def __getitem__(self, k):
            return self.a[k]

        def __setitem__(self, k, v):
            self.a[k] = v

    class DS(object):
        def __init__(self, *a, **k):
            self.variables = {k: Var(v) for k, v in store.items()}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    cfg = mock.MagicMock()
    cfg.paths.vic_parameter = {'Cv': cv}
    cfg.paths.capillary = {'max_moist': mx}
    cfg.paths.statefile_dir = '/nonexistent'
    with mock.patch.object(vr.nc, 'Dataset', DS):
        vr.update_statefile(datetime.datetime(2004, 1, 1), cfg, cpr)
    ref = store['STATE_SOIL_MOISTURE'][:, 0]          # [veg, layer, y, x]
    m = torch.as_tensor(sm[:, 0].reshape(NV, NL, R * C).transpose(2, 0, 1)
                        [:, :, None, :].copy())       # [cell, tile, 1, L]
    cvt = torch.as_tensor(cv.reshape(NV, -1).T.copy())
    live = torch.ones(R * C, NV, 1, dtype=torch.bool)
    col = (torch.as_tensor(cpr.reshape(-1))[:, None] * cvt)[:, :, None]
    new, lost = W.capillary_rise(m, cvt, live, col,
                                 torch.as_tensor(mx.reshape(NL, -1).T.copy()),
                                 2, exact=True)
    got = new[:, :, 0].numpy().transpose(1, 2, 0).reshape(NV, NL, R, C)
    assert (got == ref).all()
    assert (lost > 0).any()                    # the top layer overflowed


def test_cascade_fixed_conserves_per_column():
    g = torch.Generator().manual_seed(2)
    m = 50 + 300 * torch.rand(5, 3, 2, 3, generator=g, dtype=torch.float64)
    cv = torch.tensor([[0.5, 0.3, 0.2]] * 5, dtype=torch.float64)
    live = torch.ones(5, 3, 2, dtype=torch.bool)
    mx = torch.tensor([[60.0, 320.0, 340.0]] * 5, dtype=torch.float64)
    cpr = torch.tensor([0.0, 5.0, 80.0, 400.0, 1e3], dtype=torch.float64)
    new, lost = W.capillary_rise(m, cv, live, cpr, mx, 2, exact=False)
    tot = lambda x: (x * cv[:, :, None, None]).sum((1, 3))      # [cell, band]
    assert torch.allclose(tot(new) + lost, tot(m) + cpr[:, None],
                          rtol=0, atol=1e-10)


# ------------------------------------------------------------- synthetic
class GwSim(TC.Sim):
    """TC.Sim's soil, runoff and baseflow; OUT_GWRECHARGE is a function of
    the day's baseflow (so it survives a restart), negative on some days."""

    def view(self, moist, runoff, baseflow):
        v = super().view(moist, runoff, baseflow)
        v.gwrecharge = 0.4 * baseflow - 0.15
        return v


SYN = ('vicwur_routing', 'vicwur_wateruse_local', 'vicwur_nonrenew',
       'groundwater')


def syn_build(device, mode='exact', group='tv4syn', restart=None,
              ndays=TC.NDAYS):
    if not HAVE_R1:
        pytest.skip('R1 data absent')
    win = TC.Window(device, SYN, group, ndays=ndays)
    win.sim = GwSim(device)
    registry.register('tv4_syn_window', lambda **c: win, 'test',
                      replace=True)
    cfg = TC.module_cfgs(SYN[:3], group)
    for k in SYN[:3]:
        cfg[k]['gwm'] = True
    sv = GW.static_values()
    lat, lon = np.meshgrid([11.90625, 11.96875],
                           [77.21875, 77.28125, 77.34375], indexing='ij')
    cfg['groundwater'] = dict(
        profile='vicwur3', domain=dict(lat=lat.ravel(), lon=lon.ravel()),
        static=sv, start_date=TC.START, mode=mode, group=group)
    mods = {'vic': dict(connector='tv4_syn_window')}
    for k, c in cfg.items():
        mods[k] = dict(connector=k, **c)
    c = dict(run=dict(device=device, log=False), modules=mods,
             connections=[])
    if restart:
        c['checkpoint'] = dict(restart_from=restart)
    return TorchVIC.from_dict(c), win


def syn_run(device, mode='exact', **kw):
    m, win = syn_build(device, mode, **kw)
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    return m, win


def rows(m, mod):
    return [r for r in m.ledger.balances if r['module'] == mod]


def test_registration_and_plan():
    if HAVE_R1:
        c = registry.get('groundwater').make(
            profile='vicwur3', domain=dict(lat=[0.0, 0.0625], lon=[0.0, 0.0]),
            static={}, start_date='2003-07-01', group='tv4reg')
        assert isinstance(c, VicWur3GroundwaterConnector)
        assert c.period == 'month' and c.exact
    o = Options(routing=True, force_routing=True, wateruse=True,
                local_with=True, nonrenew_with=True, groundwater=True)
    calls = plan(o)
    last = [c for c in calls if c.phase == 'O']
    assert last[-1].kind == 'groundwater'           # after store_error
    assert ('F', 'routing', 'pre_force') in [(c.phase, c.kind, c.hook)
                                             for c in calls]


@pytest.mark.parametrize('mode', ['exact', 'fixed'])
def test_ledger_closes_and_exchanges_cancel(mode):
    m, win = syn_build('cpu', mode)
    if mode == 'fixed':          # opt-in: the unused pumped water recharges
        m.runtime.connectors['vicwur_nonrenew'].return_to = 'aquifer'
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    gw, rt = rows(m, 'groundwater'), rows(m, 'vicwur_routing')
    nr = rows(m, 'vicwur_nonrenew')
    assert len(gw) == TC.NDAYS
    g = lambda r, k: r.get('detail_' + k, 0.0)
    for a, b in zip(gw, rt):
        # the baseflow the aquifer gives = the routing's forced inflow
        k = 'river_forced' if 'detail_river_forced' in a else 'river'
        assert abs(g(a, k) - g(b, 'force')) <= 1e-9 * (
            abs(g(b, 'force')) + 1.0)
    pumped = sum(g(r, 'pumping') for r in gw)
    src = sum(g(r, 'gwm_source') for r in nr)
    assert src > 0
    if mode == 'fixed':
        for a, b in zip(gw, nr):
            assert abs(g(a, 'pumping') - g(b, 'gwm_source')) <= 1e-9 * (
                abs(g(b, 'gwm_source')) + 1.0)
        # the unused part of the pumped water goes back to the aquifer: what
        # the non-renewable module returns is what the aquifer takes as
        # recharge (and none of it enters the river ring)
        back = sum(g(r, 'nonrenew_return') for r in gw)
        assert back > 0
        assert abs(back - sum(g(r, 'ring_return') for r in nr)) <= 1e-9 * back
    else:
        assert pumped == 0.0
        # VIC-WUR 3.0 returns the unused pumped water to the river
        assert sum(g(r, 'nonrenew_return') for r in gw) == 0.0
        # month ends 2003-12-31 and 2004-01-31: two MODFLOW steps
        assert win.phase_group.roles['groundwater'].nstep == 2
    # MODFLOW puts no recharge on the CHD cell (c5): named, not lost
    assert sum(g(r, 'recharge_not_applied') for r in gw) > 0
    assert sum(g(r, 'capillary_rise') for r in gw) > 0
    assert sum(g(r, 'force') for r in rt) > 0


def _states(m):
    return {k: c.save_state() for k, c in m.runtime.connectors.items()}


@pytest.mark.parametrize('mode,cut', [('exact', 30), ('exact', 17),
                                      ('fixed', 30)])
def test_restart_equals_continuous(tmp_path, mode, cut):
    full, wf = syn_run('cpu', mode, group='tv4r')
    ck = str(tmp_path / 'ck.pt')
    m1, _ = syn_build('cpu', mode, group='tv4r')
    with torch.inference_mode():
        m1.initialize().run(until=cut * DAY)
        m1.save_checkpoint(ck)
    VC._GROUPS.clear()
    m2, w2 = syn_build('cpu', mode, group='tv4r', restart=ck)
    with torch.inference_mode():
        m2.initialize().run()
        m2.finalize(write=False)
    assert not TC.same(_states(full), _states(m2))
    a = [r['residual'] for r in rows(full, 'groundwater')]
    b = [r['residual'] for r in rows(m2, 'groundwater')]
    assert a[-len(b):] == b


def test_refuses_a_group_without_gwm():
    m, win = syn_build('cpu')
    m.runtime.connectors['vicwur_wateruse_local'].gwm = False
    with pytest.raises(Exception, match='gwm=True'):
        m.initialize()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('mode', ['exact', 'fixed'])
def test_gpu_equals_cpu_and_ledger_closes(mode):
    mc, _ = syn_run('cpu', mode, group='tv4g1')
    mg, _ = syn_run('cuda', mode, group='tv4g2')
    assert all(r['ok'] for r in mg.ledger.balances)
    a, b = _states(mc)['groundwater'], _states(mg)['groundwater']
    for k in ('h', 'force', 'cpr_mm', 'acc_pump'):
        x, y = a[k].cpu().double(), b[k].cpu().double()
        assert torch.allclose(x, y, rtol=1e-9, atol=1e-8), k


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_phases_do_no_host_sync():
    """Mode fixed steps MODFLOW every day: after the synchronous warm-up
    steps, a whole between-days (P1-O with a MODFLOW step, then F, U) runs
    under CUDA sync-debug 'error'."""
    m, win = syn_build('cuda', 'fixed', group='tv4s')
    m.initialize()
    with torch.inference_mode():
        m.run(until=10 * DAY)
    grp, sim = win.phase_group, win.sim
    gw = grp.roles['groundwater']
    assert gw.nstep > gw.sync_steps
    moist, ro, bf = sim.day(100, win.moist)
    v = sim.view(moist, ro, bf)
    n0 = gw.nstep
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        with torch.inference_mode():
            grp._set_view(v)
            grp._set_time(10 * DAY)
            grp._phases(('P1', 'P2', 'P3', 'O'))
            grp._set_time(11 * DAY)
            grp._phases(('F', 'U'))
    finally:
        torch.cuda.set_sync_debug_mode('default')
    assert gw.nstep == n0 + 1


# -------------------------------------------------------------- replay (R1)
import vicwur_gw_replay as RP                                       # noqa


def need(run):
    return pytest.mark.skipif(not RP.available(run),
                              reason='%s reference run absent' % run)


def _cells(a):
    """[..., 2, R, C+pad] MF grid -> [..., 6] cells (row-major, no pad)."""
    return np.asarray(a)[..., :3].reshape(np.asarray(a).shape[:-2] + (6,))


def _mf_check(rec, heads, q, tol_h=1e-9, tol_q=1e-9):
    """rec: the replay's records at the MODFLOW steps; heads [P, 2, R, C'],
    q {pkg: [P, 2, R, C']} of the reference."""
    days = sorted(rec)
    assert len(days) == heads.shape[0]
    worst = {}
    for i, d in enumerate(days):
        g = rec[d]
        dh = np.abs(g['head'] - _cells(heads[i])).max()
        assert dh <= tol_h, (d, dh)
        worst['head'] = max(worst.get('head', 0), dh)
        for k in ('RIV', 'DRN', 'EVT', 'RCH', 'WEL'):
            if 'q_' + k not in q:
                continue
            ref = _cells(q['q_' + k][i])
            ref = ref[1] if k == 'WEL' else ref[0]
            sc = np.abs(ref).max() + 1.0
            e = np.abs(g[k] - ref).max()
            assert e <= tol_q * sc, (d, k, e, sc)
            worst[k] = max(worst.get(k, 0), e / sc)
    return worst


def _cascade_check(case, rec, cv_of_day):
    """The soil edit of every MODFLOW step equals gw_runs.w3_state_edit
    (the VIC-WUR 3.0 code's numpy port with the documented fixes) on the
    soil the plugins left on that day."""
    for d, g in rec.items():
        base = g['base']                         # [cell, tile, band, layer]
        cv = cv_of_day(d)                         # [cell, tile]
        live = (cv > 0)[:, :, None] & (case.S['af'] > 0)[:, None, :]
        sm = base.transpose(1, 2, 3, 0)           # [tile, band, layer, cell]
        new = GW.w3_state_edit(sm, cv.T, live.transpose(1, 2, 0), g['cpr'],
                               GW.static_values()['max_moist'])
        ref = new.transpose(3, 0, 1, 2) - base
        assert np.abs(g['delta'] - ref).max() <= 1e-9, d


@need('G11')
def test_l2_g11_monthly_against_mf6():
    case = RP.load('G11')
    win, m = RP.replay(case, 'cpu')
    ref = np.load(RP.run_dir('G11') + '/mf_ref.npz')
    w = _mf_check(win.gwrec, ref['head'], ref)
    # the forced inflow handed on = -(RIV + DRN) / 86400 (VIC-WUR 3.0's
    # mf_run without its 1/days), and VIC-WUR 3.0's own number is ours / days
    days = sorted(win.gwrec)
    for i, d in enumerate(days):
        g = win.gwrec[d]
        f = -(_cells(ref['q_RIV'][i])[0] + _cells(ref['q_DRN'][i])[0]) / 86400
        assert np.abs(g['force'] - f).max() <= 1e-12 * (np.abs(f).max() + 1)
        w3 = ref['w3_baseflow'][i][:, :3].ravel()
        nd = (datetime.date(2003, 7, 1) + datetime.timedelta(days=d)).day
        assert np.abs(w3 * nd - g['force']).max() <= 1e-11 * (
            np.abs(f).max() + 1)
    _cascade_check(case, win.gwrec, lambda d: case.cv[d])
    assert all(r['ok'] for r in m.ledger.balances)
    with open(GW.LOG, 'a') as f:
        f.write('%s test_l2_g11: 84 months, max |dh| %.2e m, flows/scale %s\n'
                % (datetime.datetime.now().isoformat(timespec='seconds'),
                   w['head'], {k: '%.1e' % v for k, v in w.items()
                               if k != 'head'}))


@need('G11c')
def test_l2_g11c_coupled_loop():
    case = RP.load('G11c')
    win, m = RP.replay(case, 'cpu')
    rd = RP.run_dir('G11c')
    lp = np.load(rd + '/loop.npz')
    import glob
    outs = sorted(glob.glob(rd + '/W3/mf/out_*.npz'))
    heads = np.concatenate([np.load(p)['head'] for p in outs])
    q = {k: np.concatenate([np.load(p)[k] for p in outs])
         for k in ('q_RIV', 'q_DRN', 'q_EVT', 'q_RCH')}
    w = _mf_check(win.gwrec, heads, q)
    days = sorted(win.gwrec)
    f = np.stack([win.gwrec[d]['force'] for d in days])
    assert np.abs(f - lp['force']).max() <= 1e-12 * (np.abs(f).max() + 1)
    cpr = np.stack([win.gwrec[d]['cpr'] for d in days])
    assert np.abs(cpr - lp['cpr']).max() <= 1e-12 * (np.abs(cpr).max() + 1)
    _cascade_check(case, win.gwrec, lambda d: case.cv[d])
    # routing every day: discharge (with the MODFLOW forced inflow), the
    # whole ring and the non-renewable withdrawal against the dump
    T = case.T
    rout = np.stack([win.rec[d]['rout'] for d in range(T)])
    C = case.rout[:T]
    sc = np.abs(C).max(0) + 1e-9
    e = np.abs(rout - C)
    assert (e <= 1e-10 * sc + 1e-12).all(), np.unravel_index(e.argmax(),
                                                              e.shape)
    wu = np.stack([win.rec[d]['wu'] for d in range(T)])
    W_ = np.nan_to_num(case.wu[:T])
    act = case.active
    e = np.abs(wu[:, :, act] - W_[:, :, act])
    assert e.max() <= 1e-10 * (np.abs(W_).max() + 1), e.max()
    assert all(r['ok'] for r in m.ledger.balances)
    with open(GW.LOG, 'a') as f_:
        f_.write('%s test_l2_g11c: %d months, max |dh| %.2e m, flows/scale '
                 '%s, forced inflow, cpr, cascade, %d days of routing and '
                 'water use within 1e-10\n' % (
                     datetime.datetime.now().isoformat(timespec='seconds'),
                     len(days), w['head'], {k: '%.1e' % v for k, v in
                                            w.items() if k != 'head'}, T))


@need('G11c')
@pytest.mark.parametrize('cut', [62, 75])
def test_l2_g11c_restart_equals_continuous(tmp_path, cut):
    case = RP.load('G11c', nmax=120)
    full, mf = RP.replay(case, 'cpu', tag='f')
    ck = str(tmp_path / 'ck.pt')
    RP.replay(case, 'cpu', until=cut * DAY, checkpoint=ck, tag='r')
    VC._GROUPS.clear()
    w2, m2 = RP.replay(case, 'cpu', restart=ck, tag='r')
    assert not TC.same(_states(mf), _states(m2))
    for d in w2.gwrec:
        for k, v in w2.gwrec[d].items():
            assert np.array_equal(v, full.gwrec[d][k]), (d, k)


@need('G11c')
@pytest.mark.skipif(not HAVE_MF6, reason='validation venv / mf6 absent')
def test_l2_g11c_fixed_daily_against_mf6(tmp_path):
    """Mode fixed (daily steps, wells for the non-renewable withdrawal,
    signed recharge, Manning as commented) on 92 days of G11c: the replay's
    own package inputs (the day's OUT_GWRECHARGE plus the excess carried,
    discharge, wells) through VIC-WUR 3.0's code with mode fixed and MF6."""
    nd = 92
    case = RP.load('G11c', nmax=nd)
    # the reference is VIC-WUR 3.0's code with daily steps: the stability
    # numerics of mode fixed (FIXED_STABILITY) are off here, tested on
    # their own
    from torchvic.coupling.connectors import groundwater as GC
    with mock.patch.dict(GC.FIXED_STABILITY, river_exchange=False,
                         stage_cap=False):
        win, m = RP.replay(case, 'cpu', mode='fixed')
    assert all(r['ok'] for r in m.ledger.balances)
    rec = win.gwrec
    days = sorted(rec)
    assert days == list(range(nd))
    flow = np.stack([rec[d]['RCH'] for d in days])     # m3/d
    ca = GW.static_values()['cellarea']
    gwmm = flow / ca * 1000.0                           # mm over the day
    q = np.stack([win.rec[d]['rout'][:, 4] for d in days])
    wells = np.stack([rec[d]['WEL'] for d in days])
    per = []
    for d in days:
        t = datetime.date(2003, 7, 1) + datetime.timedelta(days=d)
        per.append((t.year, t.month, t.day, 1))
    sh = lambda a: a.reshape(-1, 2, 3)
    inp = GW.mf_inputs(str(tmp_path / 'ws'), per, sh(gwmm), sh(q), 'fixed',
                       wells=sh(wells))
    pi, po = str(tmp_path / 'in.npz'), str(tmp_path / 'out.npz')
    np.savez(pi, **inp)
    GW.mf_call(pi, po)
    o = np.load(po)
    w = _mf_check(rec, o['head'], o, tol_h=1e-9, tol_q=1e-9)
    assert np.abs(wells).max() > 0
    with open(GW.LOG, 'a') as f:
        f.write('%s test_l2_g11c_fixed_daily: %d days, max |dh| %.2e m, '
                'flows/scale %s\n' % (
                    datetime.datetime.now().isoformat(timespec='seconds'),
                    nd, w['head'], {k: '%.1e' % v for k, v in w.items()
                                    if k != 'head'}))


# ---------------------------------------------------- coupled run (R1 G11)
@need('G11')
def test_l3_g11_vic_in_the_loop(tmp_path):
    """VIC itself in the loop: VicConnector (the vicwur3 core with
    VIC-WUR 3.0's GWM soil, asked for by the groundwater member) hands the
    day's OUT_GWRECHARGE to the phase group (gwrecharge_view) and the
    groundwater member (POC: nothing goes back to VIC, as in the continuous
    C G11 run) steps MODFLOW month by month. The view's recharge equals the
    C's OUT_GWRECHARGE (double) every day, the heads and flows equal MF6 on
    the C's recharge (VIC-WUR 3.0's own mf_run, mf_ref_poc.npz), the ledger
    closes."""
    from torchvic.coupling.connectors import vicwur_setup as WS
    rd = RP.run_dir('G11')
    if not os.path.exists(rd + '/mf_ref_poc.npz'):
        pytest.skip('mf_ref_poc.npz absent (gw_runs.py ref G11)')
    pytest.importorskip('xarray')
    gp = rd + '/global_W3.txt'
    c = WS.load_case(gp, str(tmp_path))
    group = 'tv4_l3_%d' % os.getpid()
    case = RP.load('G11')
    dom, sv = RP.domain(case)
    # vicwur_setup adds the groundwater member (GWM TRUE) and VIC's
    # gwrecharge view; POC: nothing back to VIC
    mods = WS.modules(gp, c, group=group, mode='exact',
                      groundwater=dict(static=sv, foc=False))
    assert mods['vic']['day_coupled'] == ['groundwater']
    assert mods['vic']['gwrecharge_view']
    m = TorchVIC.from_dict(dict(run=dict(device='cpu', log=False),
                                modules=mods, connections=[]))
    gw = m.runtime.connectors['groundwater']
    seen, rec = [], {}
    put, mf = gw.put_data, gw._mf_step

    def put_data():
        seen.append(gw.view.gwrecharge.clone())
        put()

    def mf_step(step, T):
        mf(step, T)
        rec[step] = {k: (v.numpy().copy() if torch.is_tensor(v) else v)
                     for k, v in gw.last.items()}
    gw.put_data, gw._mf_step = put_data, mf_step
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    T = case.T
    g = torch.stack(seen).numpy()
    assert g.shape[0] in (T - 1, T)
    assert np.abs(g - case.gw[:g.shape[0]]).max() <= 1e-12
    ref = np.load(rd + '/mf_ref_poc.npz')
    # the recharge is TorchVIC's VIC's (3e-14 mm off the C's): flows within
    # 1e-7 m3/d of MF6 on the C's
    w = _mf_check(rec, ref['head'][:len(rec)], ref, tol_q=1e-7)
    assert all(r['ok'] for r in m.ledger.balances)
    with open(GW.LOG, 'a') as f:
        f.write('%s test_l3_g11: VIC (GWM) in the loop, %d days of '
                'OUT_GWRECHARGE on the view within 1e-12 mm, %d months, max '
                '|dh| %.2e m\n' % (datetime.datetime.now().isoformat(
                    timespec='seconds'), g.shape[0], len(rec), w['head']))


# ---------------------------------------------------- losing-river regression
def _lossy_run(xch):
    """Fixed mode with the river stage forced 500 m up: the river loses far
    more than it carries. `xch`: the river exchange of the sweep, else
    VIC-WUR 3.0's formulas with daily steps. Returns (lowest discharge of
    any cell at the end of a day, the model, the number of river cells)."""
    from torchvic.coupling.connectors import groundwater as GC
    from torchvic.vicwur import river_exchange as RX
    with mock.patch.dict(GC.FIXED_STABILITY, river_exchange=xch):
        m, win = syn_build('cpu', 'fixed', group='tv4loss%d' % int(xch))
    gw = m.runtime.connectors['groundwater']
    seen = []
    orig = VicWur3GroundwaterConnector.put_data

    def put_data(self):
        seen.append(self.router.discharge.min().item())
        orig(self)
    up = (mock.patch.object(RX, 'stage', lambda q, bot, den, cap: bot + 500.0)
          if xch else mock.patch.object(
              W, 'stage', lambda r, q, exact=True:
              torch.full_like(q, 500.0) + r['rbot']))
    with mock.patch.object(VicWur3GroundwaterConnector, 'put_data',
                           put_data), up:
        with torch.inference_mode():
            m.initialize().run()
            m.finalize(write=False)
    assert gw.xch == xch
    return min(seen), m, (int(gw.rx.river.sum()) if xch else 0)


def test_losing_river_cannot_take_more_than_it_carries():
    """ISC 2005: an unbounded river loss made the routed
    discharge negative and oscillating. With the river exchange in the
    sweep the loss is taken from the water in the reach: the discharge never
    goes below zero, the shortfall is a debt to the aquifer, and the ledger
    closes. VIC-WUR 3.0's formulas (river_exchange off) do overdraw."""
    if not HAVE_R1:
        pytest.skip('R1 data absent')
    lo, m, nriv = _lossy_run(True)
    assert nriv > 0
    assert lo >= -1e-12
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    gw = m.runtime.connectors['groundwater']
    assert gw.debt.sum().item() > 0          # unmet loss, owed to the aquifer
    unc, _, _ = _lossy_run(False)
    assert unc < -1e-3                  # without the bound it does overdraw
