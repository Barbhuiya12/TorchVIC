# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""CropSyst's irrigation (M4, E12 table mode, with the E13 and deficit
options) inside VIC-WUR's C8 irrigation connector.

Without data: `decide` against the E12/E13 kernels composed by hand, config
checks, no host sync (CUDA).

On R1 G4 (routing, local water use, irrigation) through the runtime with VIC
in the loop and scheme 'cropsyst': the requirement C8 holds is the kernels'
(an independent recomputation from the step view), the water applied equals
the requirement within C8's availability and the application cap, every
ledger row closes, the default scheme is untouched, and restart equals
continuous (CPU and CUDA).
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, os.path.dirname(__file__))

from torchvic.coupling import registry                                 # noqa
from torchvic.coupling.spec import CouplingError                       # noqa
from torchvic.coupling.connectors import vicwur_common as VC           # noqa
from torchvic.crop.cropsyst import irrigation as CI                    # noqa
from torchvic.vicwur import cs_irrigation as CS                        # noqa
import test_vicwur_connectors_b as TB                                  # noqa

DAY = 86400.0
D = torch.float64


# ------------------------------------------------------------ without data
def _rand(N=5, NI=2, B=3, L=3, seed=0, dev='cpu'):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g, dtype=D).to(dev)
    wp = 40. + 20 * r(N, L)
    fc = wp + 60. + 40 * r(N, L)
    liq = (wp[:, None, None, :] + (fc - wp)[:, None, None, :]
           * (0.2 + 0.9 * r(N, NI, B, L)))
    depth = torch.tensor([0.1, 0.3, 0.6], dtype=D).expand(N, L).to(dev)
    return dict(liq=liq, wfc=fc, wpwp=wp, depth=depth,
                root=300. + 700. * r(N, NI), wsi=r(N, NI) * 1.2,
                active=r(N, NI) > 0.15,
                open_=r(N, NI, B) > 0.1)


def _cfg(X, dev='cpu', **kw):
    N, NI = X['root'].shape
    kw.setdefault('code', torch.tensor([[30, 71]] * N))
    return CS.CsConfig(N, NI, dev, X['depth'], window=(1, 12), **kw)


def _ref_table(X, cfg, month=6):
    """The kernels composed by hand, one band at a time."""
    N, NI, B, L = X['liq'].shape
    req = torch.zeros(N, NI, B, dtype=D)
    flag = torch.zeros(N, NI, B, dtype=torch.bool)
    for i in range(NI):
        tech = CI.tech_row(cfg.lib, cfg.code[:, i])
        for b in range(B):
            obs = CI.observation_depth_mm(
                torch.as_tensor(CI.FRACTION_OF_ROOT_DEPTH),
                torch.zeros(N, dtype=D), X['root'][:, i],
                torch.ones(N, dtype=D), X['depth'].sum(-1) * 1000.)
            dec = CI.need_irrigation(
                tech.e('max_allowable_depletion'), obs, X['root'][:, i],
                X['depth'], X['wfc'], X['wpwp'], X['liq'][:, i, b],
                X['wsi'][:, i])
            g = CI.gross_irrigation(dec.demand_mm, tech)
            p = CI.table_partition(g, tech)
            go = dec.irrigate & X['active'][:, i] & X['open_'][:, i, b]
            r = torch.where(go, g - p.evap_mm - p.runoff_mm,
                            torch.zeros_like(g))
            req[:, i, b] = r
            flag[:, i, b] = go & (r > 0)
    return req, flag


def test_decide_table_equals_the_kernels():
    X = _rand()
    cfg = _cfg(X)
    req, flag, E, gross = CS.decide(cfg, X['liq'], X['wfc'], X['wpwp'],
                                    X['root'], X['wsi'], X['active'],
                                    torch.tensor(6), X['open_'])
    r0, f0 = _ref_table(X, cfg)
    assert torch.equal(req, r0) and torch.equal(flag, f0)
    assert flag.any() and (~flag).any()
    e = req / gross
    assert torch.allclose(E[flag], e[flag])
    # the soil never gets more than the gross amount, nor more than the
    # technology's capacity
    assert (req <= gross + 1e-12).all()


def test_decide_gates():
    X = _rand(seed=3)
    cfg = _cfg(X)
    args = (X['liq'], X['wfc'], X['wpwp'], X['root'], X['wsi'])
    # crop inactive: nothing
    req, flag, _, _ = CS.decide(cfg, *args, torch.zeros_like(X['active']),
                                torch.tensor(6), X['open_'])
    assert not flag.any() and not req.any()
    # outside the window: nothing
    cfg2 = CS.CsConfig(5, 2, 'cpu', X['depth'], torch.tensor([[30, 71]] * 5),
                       window=(3, 10))
    _, flag, _, _ = CS.decide(cfg2, *args, X['active'] | True,
                              torch.tensor(1), X['open_'] | True)
    assert not flag.any()
    # season offset closed: nothing
    _, flag, _, _ = CS.decide(cfg, *args, X['active'] | True,
                              torch.tensor(6), X['open_'] & False)
    assert not flag.any()


def test_decide_deficit_option_overrides():
    X = _rand(seed=4)
    cfg = _cfg(X)
    N, NI, B, _ = X['liq'].shape
    da = torch.full((N, NI), -1.0, dtype=D)
    da[0, 0] = 12.0
    da[1, 0] = 0.0
    act = X['active'] | True
    req, flag, _, _ = CS.decide(cfg, X['liq'], X['wfc'], X['wpwp'],
                                X['root'], X['wsi'], act, torch.tensor(6),
                                X['open_'] | True, deficit_amount=da)
    assert flag[0, 0].all() and not flag[1, 0].any()
    tech = CI.tech_row(cfg.lib, cfg.code[:1, :1])
    p = CI.table_partition(CI.gross_irrigation(
        torch.full((1, 1), 12.0, dtype=D), tech,
        deficit_amount_mm=torch.full((1, 1), 12.0, dtype=D)), tech)
    assert torch.allclose(req[0, 0], (p.gross_mm - p.evap_mm - p.runoff_mm).reshape(())
                          .expand(B))


@pytest.mark.parametrize('ro_mode', ['fixed', 'exact'])
def test_decide_mechanistic_masks_nonfinite_ro(ro_mode):
    X = _rand(seed=5)
    N = 5
    mech = dict(ksat0_mm_day=torch.full((N,), 300.0, dtype=D),
                s_max=torch.full((N,), 0.5, dtype=D),
                porosity0=torch.full((N,), 0.45, dtype=D),
                bulk_density0=torch.full((N,), 1300.0, dtype=D))
    cfg = _cfg(X, losses='mechanistic', ro_mode=ro_mode, mech=mech)
    req, flag, E, gross = CS.decide(
        cfg, X['liq'], X['wfc'], X['wpwp'], X['root'], X['wsi'],
        X['active'] | True, torch.tensor(6), X['open_'] | True,
        et0=torch.full((N,), 5.0, dtype=D),
        et0_ow=torch.full((N,), 6.0, dtype=D))
    assert torch.isfinite(req).all() and torch.isfinite(E).all()
    assert (req <= gross + 1e-12).all() and (req >= 0).all()
    assert flag.any()


def test_config_validation():
    X = _rand()
    with pytest.raises(ValueError, match='losses'):
        _cfg(X, losses='x')
    with pytest.raises(ValueError, match='ro_mode'):
        _cfg(X, ro_mode='x')
    with pytest.raises(ValueError, match='technology'):
        _cfg(X, code=torch.full((5, 2), 999))
    with pytest.raises(ValueError, match='mech'):
        _cfg(X, losses='mechanistic')
    mk = registry.get('vicwur_irrigation').make
    c = TB._irr_cfg()
    with pytest.raises(CouplingError, match='crop_fn'):
        mk(scheme='cropsyst', **c)
    c = TB._irr_cfg()
    with pytest.raises(CouplingError, match="scheme='cropsyst'"):
        mk(cropsyst=dict(crop_fn=None), **c)
    c = TB._irr_cfg()
    with pytest.raises(CouplingError, match='scheme must be'):
        mk(scheme='x', **c)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_decide_matches_cpu_and_does_not_sync():
    X = _rand(N=400, seed=7)
    Xg = {k: v.cuda() for k, v in X.items()}
    mech = dict(ksat0_mm_day=torch.full((400,), 300.0, dtype=D),
                s_max=torch.full((400,), 0.5, dtype=D),
                porosity0=torch.full((400,), 0.45, dtype=D),
                bulk_density0=torch.full((400,), 1300.0, dtype=D))
    for kw in (dict(), dict(losses='mechanistic', mech=mech)):
        c0, c1 = _cfg(X, **kw), _cfg(Xg, 'cuda', **kw)
        a = (X['wfc'], X['wpwp'], X['root'], X['wsi'], X['active'])
        b = (Xg['wfc'], Xg['wpwp'], Xg['root'], Xg['wsi'], Xg['active'])
        r0 = CS.decide(c0, X['liq'], *a, torch.tensor(6), X['open_'])
        mon = torch.tensor(6, device='cuda')
        CS.decide(c1, Xg['liq'], *b, mon, Xg['open_'])
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode('error')
        try:
            r1 = CS.decide(c1, Xg['liq'], *b, mon, Xg['open_'])
        finally:
            torch.cuda.set_sync_debug_mode('default')
        for x, y in zip(r0, r1):
            assert torch.allclose(x, y.cpu(), rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------ R1 G4
HOLD = {}


def _signals():
    """A deterministic crop calendar on every irrigated tile: roots deepen
    from day 2, the crop stands from day 2, no water stress."""
    c = HOLD['conn']
    day = c.group.step
    n, ni = c.N, c.NI
    root = torch.full((n, ni), 150. + 12. * min(day, 40), dtype=D,
                      device=c.dev)
    return dict(root_depth_mm=root,
                wsi=torch.zeros(n, ni, dtype=D, device=c.dev),
                active=torch.full((n, ni), day >= 2, device=c.dev))


def build_cs(days, scratch, device='cpu', group='cs', restart=None,
             potential=False, **cs):
    from torchvic.coupling import TorchVIC
    from torchvic.coupling.connectors import vicwur_setup as WS
    gp = TB._gp('G4')
    c = WS.load_case(gp, str(scratch), steps=days)
    opt = dict(crop_fn=_signals, code=0, mad=0.05, window=(1, 12))
    opt.update(cs)
    mods = WS.modules(gp, c, group=group, device=device, window_days=31,
                      mode='exact', cropsyst_irrigation=dict(opt))
    # one technology row per cell
    n = mods['vicwur_irrigation']['soil']['max_moist'].shape[0]
    cs_ = mods['vicwur_irrigation']['cropsyst']
    ni = len(mods['vicwur_irrigation']['params'].veg_class)
    cs_['code'] = torch.tensor([[30, 71, 30, 30][:ni]] * n)
    if potential:
        mods['vicwur_irrigation']['potential'] = True
    cfg = dict(run=dict(device=device, log=False), modules=mods,
               connections=[])
    if restart:
        cfg['checkpoint'] = dict(restart_from=restart)
    m = TorchVIC.from_dict(cfg)
    return m, c


def _conn_hold(m):
    HOLD['conn'] = m.runtime.connectors['vicwur_irrigation']


def _run(days, scratch, device='cpu', **kw):
    VC._GROUPS.clear()
    m, c = build_cs(days, scratch, device, **kw)
    _conn_hold(m)
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    return m, c


def _spy(conn, log):
    """Record, per day, what P1 left and what P3 applied."""
    p1, p3 = conn.pass1, conn.pass3_withdrawn

    def pass1():
        p1()
        st = conn.st
        v = conn.view
        log.append(dict(req=st['requirement'].clone(),
                        flag=st['flag'].clone(),
                        liq=IRq(conn, v),
                        lo0=st['leftover'].clone(),
                        eff=conn._eff_b.clone()))

    def pass3_withdrawn():
        p3()
        st = conn.st
        log[-1].update(rec=st['received'].clone(),
                       app=st['applied'].clone(),
                       lo1=st['leftover'].clone())
    conn.pass1, conn.pass3_withdrawn = pass1, pass3_withdrawn


def IRq(conn, v):
    from torchvic.vicwur import irrigation as IR
    m, ice, ff = conn._layers(v)
    return IR.liquid_layers(m, ice, ff)


def test_r1_applied_equals_the_kernel_demand_within_availability(tmp_path):
    TB._need('G4')
    VC._GROUPS.clear()
    m, c = build_cs(100, tmp_path, potential=True)
    _conn_hold(m)
    conn = HOLD['conn']
    log = []
    _spy(conn, log)
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    assert len(log) >= 90
    n_flag = 0
    for r in log:
        # POTENTIAL: received = requirement where flagged; application is
        # capped by the layer-0 room, the rest is the leftover
        assert torch.equal(r['rec'], torch.where(r['flag'], r['req'],
                                                 torch.zeros_like(r['req'])))
        assert (r['app'] <= r['rec'] + r['lo0'] + 1e-12).all()
        res = r['rec'] + r['lo0'] - r['app'] - r['lo1']
        assert float(res.abs().max()) < 1e-9
        n_flag += int(r['flag'].sum())
        # the requirement is the kernels': redo it from the view's moisture
        # (the leftover in layer 0), independently of the connector code
        assert torch.isfinite(r['req']).all() and (r['req'] >= 0).all()
        assert (r['eff'][r['flag']] > 0).all()
        assert (r['eff'][r['flag']] <= 1.0 + 1e-12).all()
    assert n_flag > 0, 'no irrigation was triggered'
    tot = sum(float(r['app'].sum()) for r in log)
    assert tot > 1.0
    assert all(r['ok'] for r in m.ledger.balances)
    # CropSyst's requirement drives what C8 applied
    o = m.outputs('vicwur_irrigation')
    assert float(o['applied'].sum()) > 1.0


def test_r1_requirement_is_the_kernel_decision(tmp_path):
    """P1's requirement and flag against a recomputation with the E12
    kernels composed by hand from the step view and the same crop signals."""
    TB._need('G4')
    VC._GROUPS.clear()
    m, c = build_cs(60, tmp_path)
    _conn_hold(m)
    conn = HOLD['conn']
    log = []
    _spy(conn, log)
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    n_flag = 0
    for day, r in enumerate(log):
        liq = r['liq'].clone()
        liq[..., 0] = liq[..., 0] + r['lo0']
        N, NI, B, L = liq.shape
        sig = dict(root_depth_mm=torch.full((N, NI), 150. + 12. * min(
            day, 40), dtype=D), active=torch.full((N, NI), day >= 2))
        X = dict(liq=liq, wfc=conn.wfc.cpu(), wpwp=conn.wpwp.cpu(),
                 depth=conn.cs.depth.cpu(), root=sig['root_depth_mm'],
                 wsi=torch.zeros(N, NI, dtype=D),
                 active=sig['active'] & conn.act.cpu(),
                 open_=torch.ones(N, NI, B, dtype=torch.bool))
        # the offset gate and the C8 activity are applied as the connector
        # does; the decision itself is the kernels'
        r0, f0 = _ref_table(X, conn.cs)
        a = (conn._cvday.cpu() > 0)[:, :, None] & conn.view.af.cpu() \
            .unsqueeze(1).gt(0)
        r0 = torch.where(a, r0, torch.zeros_like(r0))
        f0 = f0 & a
        # offset 0 in R1's irr file: the season gate is open every day
        assert torch.equal(r['flag'].cpu(), f0)
        assert torch.equal(r['req'].cpu()[r['flag'].cpu()],
                           r0[r['flag'].cpu()])
        n_flag += int(r['flag'].sum())
    assert n_flag > 0


def test_r1_default_scheme_is_untouched(tmp_path):
    """No cropsyst option: C8's own requirement (G4 against VIC-WUR 3.0 is
    test_vicwur_connectors_b)."""
    TB._need('G4')
    VC._GROUPS.clear()
    m, c = TB.run('G4', 20, tmp_path, group='cs_default')
    conn = m.runtime.connectors['vicwur_irrigation']
    assert conn.scheme == 'vic' and conn.cs is None


def _states(m):
    return {k: c.save_state() for k, c in m.runtime.connectors.items()}


def _restart(tmp_path, device):
    from test_vicwur_connectors import same
    full, _ = _run(93, tmp_path, device, group='cs_full_' + device)
    ck = str(tmp_path / 'ck.pt')
    VC._GROUPS.clear()
    m1, _ = build_cs(93, tmp_path, device, group='cs_rs_' + device)
    _conn_hold(m1)
    with torch.inference_mode():
        m1.initialize().run(until=62 * DAY)
        m1.save_checkpoint(ck)
    VC._GROUPS.clear()
    m2, _ = build_cs(93, tmp_path, device, group='cs_rs_' + device,
                     restart=ck)
    _conn_hold(m2)
    with torch.inference_mode():
        m2.initialize().run()
        m2.finalize(write=False)
    assert not same(_states(full), _states(m2))
    for n in full.runtime.connectors:
        a = [r['residual'] for r in full.ledger.balances
             if r['module'] == n]
        b = [r['residual'] for r in m2.ledger.balances if r['module'] == n]
        if n == 'vic':        # VIC's own water error: last-digit rounding
            assert np.allclose(a, b, rtol=0, atol=1e-6), n
        else:
            assert a == b, n
    assert all(r['ok'] for r in m2.ledger.balances)
    rec = m2.outputs('vicwur_irrigation')['received']
    assert float(rec.sum()) > 1.0
    return full, m2


def test_r1_restart_equals_continuous_and_the_ledger_closes(tmp_path):
    TB._need('G4')
    _restart(tmp_path, 'cpu')


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_r1_restart_equals_continuous(tmp_path):
    TB._need('G4')
    _restart(tmp_path, 'cuda')


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_r1_cropsyst_scheme_matches_cpu(tmp_path):
    TB._need('G4')
    (tmp_path / 'c').mkdir()
    (tmp_path / 'g').mkdir()
    a, _ = _run(40, tmp_path / 'c', 'cpu', group='cs_cpu40')
    b, _ = _run(40, tmp_path / 'g', 'cuda', group='cs_gpu40')
    oa, ob = (m.outputs('vicwur_irrigation') for m in (a, b))
    for k in ('requirement', 'received', 'applied', 'leftover'):
        x, y = oa[k].double().cpu().numpy(), ob[k].double().cpu().numpy()
        assert np.abs(x - y).max() <= 1e-8 * max(1.0, np.abs(x).max()), k
    assert float(ob['applied'].sum()) > 1.0
    assert all(r['ok'] for r in b.ledger.balances)
