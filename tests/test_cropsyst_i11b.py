# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The VIC-CropSyst options beyond the core hooks (tests/test_cropsyst_core_hooks.py), all opt-in
(docs/connectors/cropsyst.md):

  vcs_drain    VicwurCore(vcs_drain=): the drainage law of VIC-CropSyst's runoff.c:575-621
  canopy_crop  VicwurCore(canopy_crop=): CROPSYST_HANDLE_CANOPY_EVAP (canopy_evap.c:609-611,
               VIC_land_unit_simulation.cpp:353-360, :435-446)
  17 layers    the CRB soil through the lanes and the CUDA graphs (the column's layer count is
               the soil parameters'; see tests/test_cropsyst_i12.py)

C = <data root>/build/vic_cropsyst/VIC-CropSyst-Package/VIC_CropSyst.
"""
import os
import sys

import numpy as np
import pytest
import torch

import _paths  # noqa: F401
from torchvic.core import column as COL
from torchvic.vicwur import core as VC
from torchvic.vicwur import cs_core as CS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'cropsyst_i12'))
D = torch.float64


@pytest.fixture(autouse=True)
def _cpu():
    yield
    torch.set_default_device('cpu')


# ----------------------------------------------------------------------- the law, by hand

def c_runoff_vcs(liq, p, row, fc, irrigated, nsub):
    """runoff.c:528-780 for one column without ice, evaporation, precipitation and surface
    runoff: `nsub` sub-steps (dt), Ksat / 24 in C, here Ksat / nsub as the repository's
    runoff_steps_per_day. Returns (liq, baseflow)."""
    n = len(liq)
    mx, rs = p['max_moist'][row], p['resid_moist'][row]
    ksat, ex = p['Ksat'][row] / nsub, p['expt'][row]
    wp = p['wp'][row]
    min_liq = [rs[0]] + [wp[l] for l in range(1, n)]                     # :301-304
    liq = list(liq)
    base = 0.0
    dsmax = p['Dsmax'] / nsub
    for _ in range(nsub):
        Q12 = [0.0] * n
        for l in range(n - 1):                                            # :552-625
            tmp = max(liq[l], min_liq[l])                                 # :555-559
            if liq[l] > min_liq[l]:                                       # :564
                q = ksat[l] * ((tmp - rs[l]) / (mx[l] - rs[l])) ** ex[l]  # :580-583
                max_q = max(tmp - fc[l], 0.0)                             # :601
                if irrigated and q < max_q:                               # :604-606
                    q = max_q
                if q > max_q and max_q > 0.0:                             # :608-610
                    q = max_q
                if tmp < fc[l]:                                           # :619-621
                    q = 0.0
                Q12[l] = q
        inflow = 0.0
        for l in range(n - 1):                                            # :629-716
            liq[l] = liq[l] + inflow - Q12[l]
            assert liq[l] <= mx[l] + 1e-9
            if liq[l] < min_liq[l]:                                       # :706-710
                Q12[l] += liq[l] - min_liq[l]
                liq[l] = min_liq[l]
            inflow = Q12[l]
        lb = n - 1                                                        # :725-770
        rel = (liq[lb] - rs[lb]) / (mx[lb] - rs[lb])
        frac = dsmax * p['Ds'] / p['Ws']
        bf = frac * rel
        if rel > p['Ws']:
            frac = (rel - p['Ws']) / (1.0 - p['Ws'])
            bf += dsmax * (1.0 - p['Ds'] / p['Ws']) * frac ** p['c']
        bf = max(bf, 0.0)
        liq[lb] += Q12[lb - 1] - bf
        if liq[lb] < min_liq[lb]:                                         # :761-764
            bf += liq[lb] - min_liq[lb]
            liq[lb] = min_liq[lb]
        assert liq[lb] <= mx[lb] + 1e-9
        base += bf
    return liq, base


def _params(n, L, seed):
    g = np.random.default_rng(seed)
    mx = g.uniform(60, 90, (n, L))
    rs = g.uniform(0.0, 3.0, (n, L))
    wp = mx * g.uniform(0.2, 0.3, (n, L))
    fcm = mx * g.uniform(0.4, 0.6, (n, L))
    return dict(max_moist=mx, resid_moist=rs, Ksat=g.uniform(60, 300, (n, L)),
                expt=g.uniform(6, 14, (n, L)), wp=wp, fc=fcm, Ds=0.2, Dsmax=6.0, Ws=0.7, c=2.0)


def _sp(p, nsub, fc=True, irr=None):
    n, L = p['max_moist'].shape
    t = lambda x: torch.as_tensor(x, dtype=D)
    sp = COL.SoilParams(b_infilt=torch.full((n,), 0.2, dtype=D), Ds=torch.full((n,), p['Ds'], dtype=D),
                        Dsmax=torch.full((n,), p['Dsmax'], dtype=D), Ws=torch.full((n,), p['Ws'], dtype=D),
                        c=torch.full((n,), p['c'], dtype=D), expt=t(p['expt']), Ksat=t(p['Ksat']),
                        max_moist=t(p['max_moist']), resid_moist=t(p['resid_moist']),
                        runoff_steps_per_day=nsub, model_steps_per_day=1)
    sp.w_wilt_mm = t(p['wp'])
    rs = sp.resid_moist
    sp.min_liq = torch.cat([rs[:, :1], sp.w_wilt_mm[:, 1:]], -1)             # the Wpwp floor
    if fc:
        sp.vcs_fc = t(p['fc'])
        sp.vcs_irrig = None if irr is None else torch.as_tensor(irr, dtype=torch.bool).reshape(-1, 1)
    return sp


@pytest.mark.parametrize('L', (3, 17))
def test_vcs_drain_law_is_runoff_c_by_hand(L):
    """The law (cap above Field_Capacity, none below it, irrigated rows redistribute to
    Field_Capacity, gate liq > min_liq) against a line-by-line port of runoff.c:528-780, 24
    sub-steps, on rows of both kinds."""
    n, nsub = 8, 24
    p = _params(n, L, 100 + L)
    g = np.random.default_rng(L)
    irr = np.arange(n) % 2 == 1
    st = np.clip(p['wp'] * 1.1 + g.uniform(0.0, 1.0, (n, L)) * (p['max_moist'] * 0.55 - p['wp'] * 1.1),
                 None, p['max_moist'] * 0.6)
    sp = _sp(p, nsub, irr=irr)
    s1, ro, bf = COL.step(torch.as_tensor(st, dtype=D), torch.zeros(n, dtype=D),
                          torch.zeros(n, L, dtype=D), sp)
    assert float(ro.abs().max()) == 0.0
    for r in range(n):
        liq, base = c_runoff_vcs(st[r], p, r, p['fc'][r], bool(irr[r]), nsub)
        assert np.allclose(s1[r].numpy(), liq, rtol=0, atol=1e-9), (r, s1[r].numpy() - liq)
        assert abs(float(bf[r]) - base) < 1e-9
    # irrigated and rainfed rows differ (the redistribution did something)
    s_ni, _, _ = COL.step(torch.as_tensor(st, dtype=D), torch.zeros(n, dtype=D), torch.zeros(n, L, dtype=D),
                          _sp(p, nsub, irr=np.zeros(n, bool)))
    assert (s_ni - s1).abs().max() > 1e-6
    # the conservation of the column's water
    assert float((torch.as_tensor(st, dtype=D).sum(-1) - s1.sum(-1) - bf).abs().max()) < 1e-9


def test_vcs_fc_unset_is_vic5_and_the_stored_field_capacity_is_inert():
    """`vcs_fc_mm` (the soil's data) does not switch the law on; with `vcs_fc` None the step is
    VIC 5's, bit for bit."""
    n, L = 5, 6
    p = _params(n, L, 1)
    st = torch.as_tensor(0.5 * p['max_moist'], dtype=D)
    a = _sp(p, 24, fc=False)
    b = _sp(p, 24, fc=False)
    b.vcs_fc_mm = torch.as_tensor(p['fc'], dtype=D)
    args = (st, torch.full((n,), 3.0, dtype=D), torch.zeros(n, L, dtype=D))
    ra, rb = COL.step(*args, a), COL.step(*args, b)
    assert all(torch.equal(x, y) for x, y in zip(ra, rb))
    c = _sp(p, 24, fc=True)
    rc = COL.step(*args, c)
    assert not torch.equal(ra[0], rc[0])


def test_saxton_soil_params_carry_the_field_capacity_with_vcs_drainage():
    """read_soilparam.c:784, :858-859: FC from the Saxton function, 0.5 max_moist where it exceeds
    max_moist; present only with vcs_drainage."""
    from torchvic.crop.cropsyst import ptf
    L = 4
    mx = torch.tensor([[10.0, 60.0, 60.0, 400.0]], dtype=D)
    sp = COL.SoilParams(b_infilt=torch.full((1,), 0.2, dtype=D), Ds=torch.full((1,), 0.1, dtype=D),
                        Dsmax=torch.full((1,), 5.0, dtype=D), Ws=torch.full((1,), 0.8, dtype=D),
                        c=torch.full((1,), 2.0, dtype=D), expt=torch.full((1, L), 11.0, dtype=D),
                        Ksat=torch.full((1, L), 100.0, dtype=D), max_moist=mx, resid_moist=0.0 * mx)
    clay = torch.tensor([[0.2, 0.25, 0.3, 0.35]], dtype=D)
    qz = torch.tensor([[0.4, 0.35, 0.3, 0.25]], dtype=D)
    dm = torch.full((1, L), 0.1, dtype=D)
    assert getattr(CS.saxton_soil_params(sp, clay, qz, dm), 'vcs_fc_mm', None) is None
    on = CS.saxton_soil_params(sp, clay, qz, dm, vcs_drainage=True)
    fc = ptf.saxton_ptf(clay, qz, dm).field_capacity_mm
    want = torch.where(fc > mx, 0.5 * mx, fc)
    assert torch.equal(on.vcs_fc_mm, want)
    assert float(fc[0, 0]) > 10.0 and float(on.vcs_fc_mm[0, 0]) == 5.0     # the clamp is exercised


def test_core_options_validate():
    with pytest.raises(ValueError):
        VC.VicwurCore(vcs_drain=True, gwm='fixed')
    with pytest.raises(ValueError):
        VC.VicwurCore(vcs_drain=dict(nonsense=1))
    with pytest.raises(ValueError):
        VC.VicwurCore(canopy_crop=dict(take=1.0))
    off = VC.VicwurCore(vcs_drain=False, canopy_crop=None)
    assert off.vcs_drain is None and off.canopy_crop is None and not off.active()
    assert VC.VicwurCore(vcs_drain=True).active()


# ----------------------------------------------------------------------- the crop's side of P7

def test_engine_icpt_serves_the_potential_first_and_none_is_unchanged():
    """transpiration.cpp:65-78, :105-106: the stored water evaporates first, up to the potential
    transpiration; the uptake is asked for the rest; act = uptake + used. icpt_m None is the zero
    store of before."""
    from torchvic.crop.cropsyst import uptake as up
    L = 5
    tp = up.TranspirationParams(*[torch.full((3,), v, dtype=D) for v in (12.0, -1000.0, -1500.0, 5.0)])
    pot = torch.tensor([0.004, 0.004, 0.0], dtype=D)
    icpt = torch.tensor([0.0015, 0.006, 0.001], dtype=D)
    psi = torch.full((3, L), -100.0, dtype=D)
    fr = torch.full((3, L), 1.0 / L, dtype=D)
    args = dict(fract_root_length=fr, soil_water_pot=psi, water_pot_at_fc=torch.full_like(psi, -33.0),
                params=tp, leaf_water_pot_prev=torch.zeros(3, dtype=D))
    r = up.root_uptake(pot, torch.ones(3, dtype=D), icpt, **args)
    r0 = up.root_uptake(pot, torch.ones(3, dtype=D), torch.zeros(3, dtype=D), **args)
    used = icpt - r.interception_m
    assert torch.allclose(used, torch.minimum(icpt, pot), atol=1e-15)
    assert torch.allclose(r.act_transp_m, r.uptake_act_m + used, atol=1e-15)
    assert float(r.uptake_act_m[0]) < float(r0.uptake_act_m[0])           # the uptake is asked for less
    assert float(r.interception_m[1]) > 0.0 and float(r.interception_m[0]) == 0.0


# ----------------------------------------------------------------------- in the model

def _crb(case_id, tmp_path, days, device='cpu', **kw):
    import cropsyst_i12_l3 as L3
    try:
        L3.run_dir(case_id)
    except FileNotFoundError:
        pytest.skip('the reference run is absent')
    return L3.build(case_id, str(tmp_path), device, days, **kw)


def _go(m):
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    return m


def test_canopy_crop_on_a_crb_run_conserves_water_and_moves_the_canopy_evaporation(tmp_path):
    """P7 in the coupled CRB run: the ledger closes; VIC evaporates nothing from the crop tiles'
    canopy itself (their canopy evaporation is what the crop took, a day later); the total is
    lower than VIC's own (the crop takes at most its potential transpiration)."""
    import cropsyst_i12_l3 as L3
    a, _ = _crb(259094, tmp_path / 'a', 240, canopy_evap=False)
    _go(a)
    b, _ = _crb(259094, tmp_path / 'b', 240, canopy_evap=True)
    _go(b)
    assert all(r['ok'] for r in b.ledger.balances)
    ea = L3.series(a, 'vic')['evap_canop'][:240, 0]
    eb = L3.series(b, 'vic')['evap_canop'][:240, 0]
    take = L3.series(b, 'vicwur_cropsyst')['canopy_take'][:240]            # [day, tile] mm, the crop's day
    # the step of day d+1 books what the crop took on day d (cell value: tile mean by Cv)
    cv = np.full(take.shape[1], 1.0 / take.shape[1])
    assert abs(eb[1:].sum() - (take[:-1] * cv).sum()) < 1e-9
    assert eb[0] == 0.0
    assert eb.sum() < ea.sum()
    assert ea.sum() > 1.0


def test_crop_soil_evap_on_a_crb_run(tmp_path):
    """The crop's soil evaporation (canopy_evap.c:760-831): the ledger closes, the crop tiles'
    bare-soil evaporation is larger than ARNO's (1 - fcanopy) share (zero under a full canopy) and
    never above the reference ET of the day; it goes with canopy_evap and the core validates."""
    import cropsyst_i12_l3 as L3
    from torchvic.coupling.spec import CouplingError
    a, _ = _crb(330492, tmp_path / 'a', 240, canopy_evap=True)
    _go(a)
    ba = L3.series(a, 'vic')['evap_bare'][:240, 0]
    b, _ = _crb(330492, tmp_path / 'b', 240, canopy_evap=True, soil_evap=True)
    _go(b)
    assert all(r['ok'] for r in b.ledger.balances)
    bb = L3.series(b, 'vic')['evap_bare'][:240, 0]
    assert bb.sum() > ba.sum() + 1.0 and (bb >= 0).all()
    with pytest.raises(CouplingError):
        m, _ = _crb(330492, tmp_path / 'c', 5, soil_evap=True)
        _go(m)
    with pytest.raises(ValueError):
        VC.VicwurCore(crop_soil_evap=dict(mask=1, cover_green=0, cover_total=0))
    with pytest.raises(ValueError):
        VC.VicwurCore(crop_soil_evap=dict(mask=1))
    assert VC.VicwurCore(crop_soil_evap=None).crop_soil_evap is None


def test_irrigation_system_evaporation_leaves_as_et(tmp_path):
    """scheme='cropsyst': Ed of the E13 table partition leaves as evaporation (canopy_evap.c:560-565,
    booked into ET at :1013), exported as `evap` and in the irrigation ledger row as ET; with the
    water-use connector it is consumption, not return flow (the consumption fraction is E + Ed/gross)."""
    import cropsyst_i12_l3 as L3
    m, _ = _crb(259094, tmp_path, 240)
    _go(m)
    ir = L3.series(m, 'vicwur_irrigation')
    assert ir['evap'][:240].sum() > 1.0 and (ir['evap'] >= 0).all()
    assert ir['evap'][:240].sum() < ir['received'][:240].sum()            # the table's Ed is a share of the gross
    assert all(r['ok'] for r in m.ledger.balances)


def test_irrigation_demand_consumption_includes_the_evaporated_share():
    """pass1_demand: D = D0 / E unchanged, cons_frac = E + Ed/gross, so consumed = D0 + D (E2 - E)
    and the crop is handed the consumed water less what evaporated."""
    from torchvic.vicwur import irrigation as IR
    req = torch.tensor([[[10.0]], [[0.0]]], dtype=D)
    flag = torch.tensor([[[True]], [[False]]])
    cv = torch.ones(2, 1, dtype=D)
    af = torch.ones(2, 1, dtype=D)
    gw = torch.zeros(2, 1, dtype=D)
    E = torch.full((2, 1, 1), 0.6, dtype=D)
    edf = torch.full((2, 1, 1), 0.15, dtype=D)
    d, _, e, d0 = IR.demand(req, flag, cv, af, gw, E)
    _, _, e2, _ = IR.demand(req, flag, cv, af, gw, E + edf)
    assert abs(float(d[0]) - 10.0 / 0.6) < 1e-12 and float(d[1]) == 0.0
    consumed = d * e2
    assert abs(float(consumed[0]) - (10.0 + float(d[0]) * 0.15)) < 1e-12
    assert abs(float(d[0] * (e2 - e)[0]) - float(d[0]) * 0.15) < 1e-12


def test_canopy_crop_off_is_the_default(tmp_path):
    import cropsyst_i12_l3 as L3
    a, _ = _crb(259094, tmp_path / 'a', 30)
    _go(a)
    sa = L3.series(a, 'vic')
    b, _ = _crb(259094, tmp_path / 'b', 30, canopy_evap=False, vcs_drain=True)
    _go(b)
    sb = L3.series(b, 'vic')
    assert all(np.array_equal(sa[k], sb[k]) for k in ('et', 'baseflow', 'runoff', 'soil_liq'))


def test_vcs_drain_on_a_crb_run_lowers_the_drainage_and_conserves(tmp_path):
    import cropsyst_i12_l3 as L3
    a, _ = _crb(259094, tmp_path / 'a', 200, vcs_drain=False)
    _go(a)
    bfa = L3.series(a, 'vic')['baseflow'].sum()
    b, _ = _crb(259094, tmp_path / 'b', 200, vcs_drain=True)
    _go(b)
    assert all(r['ok'] for r in b.ledger.balances)
    assert L3.series(b, 'vic')['baseflow'].sum() < bfa


# ----------------------------------------------------------------------- lanes and graphs (GPU)

_skip = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def gpu(f):
    return pytest.mark.gpu(_skip(f))


def _gpu_run(monkeypatch, tmp_path, mode, days=40, **kw):
    import cropsyst_i12_l3 as L3
    from torchvic.model import basin
    monkeypatch.setattr(basin, 'VW_LANES', mode != 'padded')
    monkeypatch.setattr(basin, 'GRAPHS', mode == 'graphs')
    m, info = _crb(259094, tmp_path / mode, days, 'cuda', **kw)
    _go(m)
    torch.cuda.synchronize()
    return L3.series(m, 'vic'), L3.series(m, 'vicwur_cropsyst')


@gpu
def test_17_layers_all_options_padded_lanes_graphs_bitwise(tmp_path, monkeypatch):
    """The CRB soil (17 layers) with the Wpwp floor, the drainage law, the canopy hand-off and the
    CropSyst uptake: the padded run, the lanes and the CUDA graphs agree bit for bit."""
    runs = {m: _gpu_run(monkeypatch, tmp_path, m, days=200, canopy_evap=True, soil_evap=True)
            for m in ('padded', 'lanes', 'graphs')}
    v0, c0 = runs['padded']
    assert v0['soil_liq'].shape[-1] == 17
    for m in ('lanes', 'graphs'):
        v1, c1 = runs[m]
        for k in ('et', 'baseflow', 'runoff', 'soil_liq', 'soil_ice', 'evap_canop', 'transp', 'wdew'):
            assert np.array_equal(v0[k], v1[k]), (m, k)
        for k in ('lai', 'uptake', 'canopy_take', 'water_stress'):
            assert np.array_equal(c0[k], c1[k]), (m, k)
    assert c0['canopy_take'].max() > 0.0


@gpu
def test_member_day_with_the_options_does_not_sync(tmp_path, monkeypatch):
    """The member's P3 day with the canopy hand-off (the store read, the engine's icpt, the
    feedback) and the VIC step with the drainage law do not synchronise with the host (the
    engine's tables are built once, on the first day)."""
    from torchvic.coupling.connectors import vicwur_common as VCOM
    from torchvic.coupling.connectors.vicwur_cropsyst import VicWurCropSystConnector as C
    orig = C.pass3
    calls = []

    def spy(self):
        if self.run_ is None:
            self._build()
        if not calls:
            calls.append(1)
            return orig(self)
        torch.cuda.set_sync_debug_mode('error')
        try:
            return orig(self)
        finally:
            torch.cuda.set_sync_debug_mode(0)
    C.pass3 = spy
    try:
        VCOM._GROUPS.clear()
        v, c = _gpu_run(monkeypatch, tmp_path, 'graphs', days=20, canopy_evap=True, soil_evap=True)
    finally:
        C.pass3 = orig
    assert len(calls) == 1 and c['canopy_take'].shape[0] >= 19
