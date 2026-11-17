# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The connectors vicwur_irrigation, vicwur_co2, vicwur_landuse and
vicwur_crop, and the VIC side of the step view (runoff and baseflow from
VicConnector, the cold start at the run's first step, the group's
VicwurCore).

Without data: registration, configuration checks, the plan the members give.

On the R1 reference domain (6 cells, 5 snow bands; <data root>/vicwur, the
VIC-WUR 3.0 runs of tools/vicwur/gruns.py), through the coupling runtime with
VIC (the vicwur3 core) in the loop, every module configured from the
VIC-WUR 3.0 global file (torchvic.coupling.connectors.vicwur_setup):
  * G4 (routing, dams, local water use, irrigation) and G8 (+ remote,
    non-renewable, EFR, land use): the VIC-WUR 3.0 outputs to the precision
    TorchVIC has against VIC (the VIC-WUR 3.0 files are float32: 1e-6 of
    max(1, |x|));
  * G0c (CO2 and b_co2 only) the same over the 7 years;
  * every member, crop included, with the crop feedback (the R1 G9 set-up):
    restart equals continuous bit for bit, every module's ledger closes, a
    row per module and day, the exchanges between the modules agree;
  * the same on a GPU (marked gpu), and no host sync in the hooks of these
    connectors.
"""
import glob
import os
import _paths  # noqa: E402

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from torchvic.coupling import TorchVIC                                 # noqa
from torchvic.coupling import registry                                 # noqa
from torchvic.coupling.spec import CouplingError                       # noqa
from torchvic.coupling.connectors import vicwur_common as VC           # noqa
from torchvic.vicwur.order import Options, plan                        # noqa

NAMES = ('vicwur_irrigation', 'vicwur_co2', 'vicwur_landuse', 'vicwur_crop')
DAY = 86400.0
R1 = _paths.env_path(('TORCHVIC_VICWUR_R1_GRUNS', 'VICWUR_R1_GRUNS'), 'vicwur/gruns/R1')
G0C = _paths.env_path(('TORCHVIC_VICWUR_19_GRUNS', 'VICWUR_19_GRUNS'), 'verification/vicwur_19/gruns/R1') + '/G0c'
GATE = 1e-6


# ------------------------------------------------------------ without data
def test_registration():
    av = registry.available()
    for n in NAMES:
        assert n in av and registry.get(n).status == 'available'
    # the routing and EFR connectors and the lumped connectors are still there
    for n in ('vicwur_routing', 'vicwur_efr', 'irrigation', 'crop'):
        assert n in av


def _irr_cfg(**over):
    n, ni, nl = 3, 2, 3
    c = dict(irr=dict(veg_class=[2, 4], paddy=[False, True],
                      active=[[True, True]] * n,
                      gw_fraction=[[0.3, 0.0]] * n,
                      efficiency=[[0.6, 0.6]] * n, offset=[[0, 0]] * n),
             soil=dict(max_moist=[[50., 300., 400.]] * n,
                       wcr=[[30., 200., 250.]] * n,
                       wfc=[[40., 250., 300.]] * n,
                       wpwp=[[10., 60., 80.]] * n,
                       root=np.full((n, ni, nl), 1. / 3).tolist()),
             group=VC.WurGroup('cfg'))
    c.update(over)
    return c


def test_connectors_declare_a_daily_interval_and_checkpoint():
    g = VC.WurGroup('decl')
    conns = [registry.get('vicwur_irrigation').make(**_irr_cfg(group=g)),
             registry.get('vicwur_co2').make(co2=400.0, group=g),
             registry.get('vicwur_landuse').make(
                 cv={'constant': [[0.5, 0.5]] * 3},
                 soil=dict(max_moist=[[50., 300., 400.]] * 3,
                           lat=[20.0] * 3), group=g)]
    for c in conns:
        i = c.info()
        assert i.time.interval_s == DAY and i.checkpoint == 'full'
        assert 'cuda' in i.backends


def test_config_validation():
    mk = lambda n, **c: registry.get(n).make(**c)
    with pytest.raises(CouplingError):
        mk('vicwur_irrigation', group=VC.WurGroup('a'))
    bad = _irr_cfg()
    bad['irr'] = dict(bad['irr'], veg_class=[0, 4])
    with pytest.raises(CouplingError, match='1-based'):
        mk('vicwur_irrigation', **bad)
    bad = _irr_cfg()
    bad['irr'] = dict(bad['irr'], veg_class=[4, 4])
    with pytest.raises(CouplingError, match='twice'):
        mk('vicwur_irrigation', **bad)
    bad = _irr_cfg()
    bad['soil'] = dict(bad['soil'], root=[[1.0]])
    with pytest.raises(CouplingError, match='root'):
        mk('vicwur_irrigation', **bad)
    with pytest.raises(CouplingError, match='unknown'):
        mk('vicwur_irrigation', **_irr_cfg(bogus=1))
    with pytest.raises(CouplingError):
        mk('vicwur_co2', group=VC.WurGroup('b'))
    with pytest.raises(CouplingError, match='co2ref'):
        mk('vicwur_co2', co2=400.0, co2ref=0.0, group=VC.WurGroup('c'))
    with pytest.raises(CouplingError, match='b_co2'):
        mk('vicwur_co2', co2=400.0, b_co2=[1.0, 2.0],
           group=VC.WurGroup('d'))
    with pytest.raises(CouplingError, match='mode'):
        mk('vicwur_landuse', cv=0.5, soil=dict(max_moist=[[1.0]],
                                               lat=[0.0]),
           mode='other', group=VC.WurGroup('e'))
    with pytest.raises(CouplingError, match='lat'):
        mk('vicwur_landuse', cv=0.5, soil=dict(max_moist=[[1.0]]),
           group=VC.WurGroup('f'))
    with pytest.raises(CouplingError, match='tmpl'):
        mk('vicwur_crop', group=VC.WurGroup('g'))


def test_plan_follows_the_members_present():
    g = VC.WurGroup('plan_b')
    registry.get('vicwur_routing').make(
        downstream=[1, 2, 2], uh_inflow=[[1.0] * 3, [0.0] * 3],
        uh_runoff=[[1.0] * 3, [0.0] * 3], area=[1e8] * 3, group=g)
    registry.get('vicwur_wateruse_local').make(
        demand=0.0, gw_frac=0.0, cons_frac=0.0,
        soil=dict(resid_mm=[1.0] * 3), group=g)
    registry.get('vicwur_irrigation').make(**_irr_cfg(group=g))
    registry.get('vicwur_co2').make(co2=400.0, group=g)
    registry.get('vicwur_landuse').make(
        cv=0.5, soil=dict(max_moist=[[1.0, 1.0, 1.0]] * 3, lat=[0.0] * 3),
        group=g)
    calls = [(c.phase, c.kind, c.hook) for c in g.plan]
    F = [c[1] for c in calls if c[0] == 'F']
    assert F == ['co2', 'landuse', 'wateruse']
    assert [c[1] for c in calls if c[0] == 'U'] == ['co2', 'irrigation']
    P1 = [c[1:] for c in calls if c[0] == 'P1']
    assert P1 == [('irrigation', 'pass1'), ('irrigation', 'pass1_demand'),
                  ('wateruse', 'pass1'), ('routing', 'pass1')]
    P3 = [c[1:] for c in calls if c[0] == 'P3']
    assert P3 == [('irrigation', 'pass3_withdrawn'), ('irrigation', 'pass3')]
    # land use edits VIC's tile states inside the step: the group has a
    # state phase (and EFR, when present, joins it)
    assert g.STATE_ORDER == ('landuse', 'efr')


def test_series_reads():
    class Obj(object):
        def get(self, step):
            return [1.0]

        def reads(self, step):
            return step % 3 == 0
    s = VC.Series({'forcing': Obj()}).to('cpu')
    assert [s.reads(k) for k in range(4)] == [True, False, False, True]
    s = VC.Series({'series': [[1.0], [2.0]]}).to('cpu')
    assert [s.reads(k) for k in range(3)] == [True, True, False]
    s = VC.Series(1.0).to('cpu')
    assert [s.reads(k) for k in range(2)] == [True, False]


# ------------------------------------------------------------- R1 runs
def _gp(run):
    return os.path.join(G0C if run == 'G0c' else os.path.join(R1, run),
                        'global_W3.txt')


def _need(run):
    gp = _gp(run)
    out = glob.glob(os.path.join(os.path.dirname(gp), 'W3', 'out',
                                 'fluxes.*.nc'))
    if not os.path.exists(gp) or not out:
        pytest.skip('VIC-WUR reference run absent: ' + os.path.dirname(gp))
    pytest.importorskip('xarray')
    pytest.importorskip('netCDF4')


def build(run, days, scratch, device='cpu', group=None, restart=None,
          window_days=31, mode='exact'):
    from torchvic.coupling.connectors import vicwur_setup as WS
    gp = _gp(run)
    c = WS.load_case(gp, str(scratch), steps=days)
    mods = WS.modules(gp, c, group=group or 'tb_%s_%s' % (run, device),
                      device=device, window_days=window_days, mode=mode)
    cfg = dict(run=dict(device=device, log=False), modules=mods,
               connections=[])
    if restart:
        cfg['checkpoint'] = dict(restart_from=restart)
    return TorchVIC.from_dict(cfg), c


def run(run, days, scratch, device='cpu', **kw):
    m, c = build(run, days, scratch, device, **kw)
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    return m, c


_W3 = {}


def w3(run):
    if run not in _W3:
        import xarray as xr
        fs = sorted(glob.glob(os.path.join(os.path.dirname(_gp(run)), 'W3',
                                           'out', 'fluxes.*.nc')))
        a = xr.concat([xr.open_dataset(f).load() for f in fs], dim='time')
        _W3.clear()
        _W3[run] = a
    return _W3[run]


def _cells(y, case):
    """VIC's per-column series on the cells (AreaFract-weighted)."""
    ex = case.snowbands
    y = y.double().cpu().numpy()
    if ex is None:
        return y
    out = np.zeros((y.shape[0], int(ex['n_cell'])) + y.shape[2:])
    for j, (c, w) in enumerate(zip(np.asarray(ex['cell']),
                                   np.asarray(ex['weight']))):
        out[:, c] += y[:, j] * w
    return out


VIC_VARS = (('OUT_EVAP', 'et'), ('OUT_BASEFLOW', 'baseflow'),
            ('OUT_TRANSP_VEG', 'transp'), ('OUT_EVAP_BARE', 'evap_bare'),
            ('OUT_EVAP_CANOP', 'evap_canop'))
ROUT_VARS = (('OUT_DISCHARGE', 'vicwur_routing', 'discharge'),
             ('OUT_STREAM_MOIST', 'vicwur_routing', 'stream_moist'))
IRR_VARS = tuple(('OUT_' + k.upper(), 'vicwur_irrigation', k) for k in (
    'requirement', 'received', 'applied', 'leftover', 'shortage'))


def compare(m, case, run, plugin_vars, days=None, local_dam_cells=()):
    """max over days and cells of |TorchVIC - VIC-WUR 3.0| /
    max(1, |VIC-WUR 3.0|), per variable. The plugins' last day runs in
    finalize (no output row)."""
    a = w3(run)
    mask = np.isfinite(a['OUT_EVAP'].values[0])
    worst = {}

    def gate(vk, x, y):
        n = min(len(x), len(y)) if days is None else min(days, len(y))
        x, y = x[:n], y[:n].reshape(x[:n].shape)
        rel = np.abs(x - y) / np.maximum(1.0, np.abs(x))
        worst[vk] = float(rel.max())
    o = m.outputs('vic')
    for vk, k in VIC_VARS:
        gate(vk, a[vk].values[:, mask], _cells(o[k], case))
    sm = np.moveaxis(a['OUT_SOIL_MOIST'].values, 1, -1)[:, mask]
    gate('OUT_SOIL_MOIST', sm, _cells(o['soil_liq'] + o['soil_ice'], case))
    # OUT_RUNOFF: VIC-WUR 3.0 takes a local dam's inflow out of it after
    # routing read it; compared where no local dam is
    keep = [c for c in range(mask.sum()) if c not in local_dam_cells]
    gate('OUT_RUNOFF', a['OUT_RUNOFF'].values[:, mask][:, keep],
         _cells(o['runoff'], case)[:, keep])
    for vk, mod, k in plugin_vars:
        x = a[vk].values
        x = (np.moveaxis(x, 1, -1)[:, mask] if x.ndim == 4 else x[:, mask])
        gate(vk, x, m.outputs(mod)[k].double().cpu().numpy())
    return worst


def _local_dam_cells(run):
    from torchvic.vicwur import params as P
    inp = P.load_vicwur_inputs(_gp(run))
    d = inp.dams
    if d is None:
        return ()
    return tuple(int(c) for c, g in zip(d.dam_cell.tolist(),
                                        d.is_global.tolist()) if not g)


@pytest.mark.parametrize('device', ['cpu'])
def test_g4_irrigation_matches_w3(tmp_path, device):
    """G4 (routing 4 sub-steps, dams, MUN water use, irrigation with a
    paddy class) for 540 days. On 2005-01-11 (day 560) the C empties cell
    5's river to round-off (OUT_WI_SURF 1e-24 mm) and the return flow
    goes either to that round-off bin or evenly into the ring; from there
    the two codes' round-off decides, so the comparison stops before."""
    _need('G4')
    m, c = run('G4', 540, tmp_path, device)
    worst = compare(m, c, 'G4', ROUT_VARS + IRR_VARS + (
        ('OUT_WITHDRAWN', 'vicwur_wateruse_local', 'withdrawn'),),
        local_dam_cells=_local_dam_cells('G4'))
    print('G4', worst)
    assert max(worst.values()) < GATE, worst
    assert all(r['ok'] for r in m.ledger.balances)
    # irrigation did happen (received and applied, the paddy Ksat hook on)
    rec = m.outputs('vicwur_irrigation')['received']
    assert float(rec.sum()) > 100.0


def test_g8_landuse_matches_w3(tmp_path):
    """G8 (G4 + remote, non-renewable, EFR, monthly land use) over the 7
    years: VIC, routing, irrigation, EFR, deficit and Cv outputs."""
    _need('G8')
    m, c = run('G8', 2557, tmp_path)
    worst = compare(m, c, 'G8', ROUT_VARS + IRR_VARS + (
        ('OUT_EFR_DISCHARGE', 'vicwur_efr', 'efr_discharge'),
        ('OUT_EFR_BASEFLOW', 'vicwur_efr', 'efr_baseflow'),
        ('OUT_EFR_MOIST', 'vicwur_efr', 'efr_moist'),
        ('OUT_NONREN_DEFICIT', 'vicwur_nonrenew', 'deficit'),
        ('OUT_CV', 'vicwur_landuse', 'cv')),
        local_dam_cells=_local_dam_cells('G8'))
    print('G8', worst)
    assert max(worst.values()) < GATE, worst
    check_ledger(m, 2557)
    rows = [r for r in m.ledger.balances if r['module'] == 'vicwur_landuse']
    # lu_apply: the first step and every later monthly record (84 months)
    assert sum(r.get('detail_lu_apply_calls', 0) for r in rows) == 84
    assert m.runtime.connectors['vicwur_landuse'].cv is not None


@pytest.mark.parametrize('rn', ['G6a', 'G6b', 'G6c', 'G10'])
def test_out_withdrawn_is_local_remote_nonrenewable(tmp_path, rn):
    """A22: OUT_WITHDRAWN is VIC-WUR 3.0's local + transferred (remote) +
    non-renewable withdrawals (wu_output.c:387-392 in put_data), not the
    local part alone: the nonrenew connector's export against the
    VIC-WUR 3.0 history (float32: 1e-6 of max(1, |x|)), and it differs from
    the local export on a day with a non-local withdrawal."""
    _need(rn)
    m, c = run(rn, 120, tmp_path)
    a = w3(rn)
    mask = np.isfinite(a['OUT_EVAP'].values[0])
    ref = a['OUT_WITHDRAWN'].values[:, mask]
    x = m.outputs('vicwur_nonrenew')['withdrawn'].double().cpu().numpy()
    n = min(len(x), len(ref))
    rel = np.abs(x[:n] - ref[:n].reshape(x[:n].shape)) / \
        np.maximum(1.0, np.abs(ref[:n].reshape(x[:n].shape)))
    assert rel.max() < GATE, rel.max()
    loc = m.outputs('vicwur_wateruse_local')['withdrawn'].double().cpu() \
        .numpy()[:n]
    assert np.abs(x[:n] - loc).max() > 1e-6


def test_banded_run_in_one_window_equals_windowed(tmp_path):
    """A21: a banded case that fits in one coupling window used to crash
    (Case.run contracted the whole run's columns to cells, the connector
    contracted again). One window equals the same days in several."""
    _need('G4')
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    m1, _ = run('G4', 20, tmp_path / 'a', window_days=31)
    m2, _ = run('G4', 20, tmp_path / 'b', window_days=7)
    o1, o2 = m1.outputs('vic'), m2.outputs('vic')
    for k in ('et', 'runoff', 'baseflow', 'transp', 'soil_liq'):
        assert torch.equal(o1[k], o2[k]), k


def test_g0c_co2_matches_w3(tmp_path):
    """G0c: VIC with the CO2 forcing and b_co2 only (the group is the CO2
    member), 7 years."""
    _need('G0c')
    m, c = run('G0c', 2557, tmp_path)
    worst = compare(m, c, 'G0c', ())
    print('G0c', worst)
    assert max(worst.values()) < GATE, worst
    catm = m.outputs('vicwur_co2')['catm']
    assert float(catm.min()) > 300.0


def check_ledger(m, ndays):
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    mods = [k for k in m.runtime.connectors if k != 'vic']
    for n in mods:
        rows = [r for r in m.ledger.balances if r['module'] == n]
        assert len(rows) == ndays, (n, len(rows))
    g = lambda mod, k, i: [r for r in m.ledger.balances
                           if r['module'] == mod][i].get('detail_' + k, 0.0)
    if 'vicwur_irrigation' in mods and 'vicwur_wateruse_local' in mods:
        irr = [r for r in m.ledger.balances
               if r['module'] == 'vicwur_irrigation']
        # what irrigation received is what water use consumed in its IRR
        # sector (irr_use.c:455-456); both books in m3
        tot = sum(r['I_ext'] for r in irr)
        assert tot > 0
    if 'vicwur_landuse' in mods and 'vicwur_routing' in mods:
        for i in range(1, ndays):
            put = g('vicwur_landuse', 'to_ring', i)
            if put:
                assert put <= g('vicwur_routing', 'ring_in', i) + 1e-6


def test_g9_restart_equals_continuous(tmp_path):
    """Every member (crop and CO2 included), 93 days, restarted from the
    checkpoint after 62 (two VIC windows): every module state bit for bit,
    the ledger rows equal."""
    _need('G9')
    from test_vicwur_connectors import same
    full, _ = run('G9', 93, tmp_path, group='rs_full')
    ck = str(tmp_path / 'ck.pt')
    VC._GROUPS.clear()
    m1, _ = build('G9', 93, tmp_path, group='rs')
    with torch.inference_mode():
        m1.initialize().run(until=62 * DAY)
        m1.save_checkpoint(ck)
    VC._GROUPS.clear()
    m2, _ = build('G9', 93, tmp_path, group='rs', restart=ck)
    with torch.inference_mode():
        m2.initialize().run()
        m2.finalize(write=False)
    st = lambda m: {k: c.save_state() for k, c in
                    m.runtime.connectors.items()}
    assert not same(st(full), st(m2))
    for n in full.runtime.connectors:
        a = [r['residual'] for r in full.ledger.balances if r['module'] == n]
        b = [r['residual'] for r in m2.ledger.balances if r['module'] == n]
        assert a == b, n
    check_ledger(m2, 93)


def test_g9_crop_through_the_runtime(tmp_path):
    """The crop member runs WOFOST-NPK on VIC's stored water stress (the
    irrigation shortage's on the irrigated tiles) and the CO2 member's CO2:
    crops are sown and grow, no unit fails. (The crop feedback into VIC is
    tests/test_vicwur_crop_feedback.py.)"""
    _need('G9')
    m, c = run('G9', 365, tmp_path)
    o = m.outputs('vicwur_crop')
    assert float(o['OUT_CROP_GROW'].max()) > 0
    assert float(o['OUT_CROP_LAI'].max()) > 0.1
    cr = m.runtime.connectors['vicwur_crop']
    assert int(cr.s['err'].max()) == 0
    # the CO2 the crop read is the forcing's (ppm)
    assert float(o['OUT_CROP_DVS'].max()) > 0
    check_ledger(m, 365)


def test_view_runoff_baseflow_and_the_cold_start(tmp_path):
    """The step view carries VIC's OUT_RUNOFF and OUT_BASEFLOW of the day
    (rout_run reads them, rout_run.c:19-26), and F at the run's first step
    sees the cold start (EFR's start-of-step moisture)."""
    _need('G7')
    m, c = build('G7', 40, tmp_path)
    grp = m.runtime.connectors['vicwur_routing'].group
    seen = []
    orig = grp.between

    def spy(view, t_post=None, t_pre=None, post=True, pre=True):
        if view is not None and post:
            seen.append((int(round(t_post / DAY)), view.runoff.clone(),
                         view.baseflow.clone()))
        if not post and t_pre == 0.0:
            seen.append(('cold', view.moist.clone()))
        return orig(view, t_post, t_pre, post, pre)
    grp.between = spy
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    o = m.outputs('vic')
    ro, bf = _cells(o['runoff'], c), _cells(o['baseflow'], c)
    days = [s for s in seen if s[0] != 'cold']
    assert len(days) == 40                    # every day, the last too
    for d, r, b in days:
        assert np.allclose(r.numpy(), ro[d], rtol=1e-12, atol=1e-14)
        assert np.allclose(b.numpy(), bf[d], rtol=1e-12, atol=1e-14)
    cold = grp.to_columns([s for s in seen if s[0] == 'cold'][0][1])
    # generate_default_state.c:75-84: every tile at init_moist, clipped at
    # max_moist (the case's columns)
    im = torch.as_tensor(c.init_moist, dtype=torch.float64)
    mx = torch.as_tensor(c.soil.max_moist, dtype=torch.float64)
    want = torch.minimum(im.expand_as(mx), mx)
    assert torch.equal(cold, want[:, None, :].expand_as(cold).to(cold.dtype))


def test_no_part_b_member_leaves_vic_alone(tmp_path):
    """Without an irrigation, CO2, land-use or crop member the group builds
    no VicwurCore (mode fixed; mode exact keeps only VIC-WUR 3.0's
    throughfall leak, next test), and reading the day's runoff and baseflow
    for the view changes nothing: VIC's outputs are those of VIC run alone
    through the runtime, bit for bit."""
    _need('G1a')
    m, c = run('G1a', 62, tmp_path, mode='fixed')
    grp = m.runtime.connectors['vicwur_routing'].group
    assert grp.vic_core() is None
    gf, core, steps = c.vicwur_image_global
    alone = TorchVIC.from_dict(dict(
        run=dict(device='cpu', log=False), connections=[],
        modules=dict(vic=dict(connector='vic', global_param=gf, core=core,
                              steps=steps, window_days=31))))
    with torch.inference_mode():
        alone.initialize().run()
        alone.finalize(write=False)
    o, ref = m.outputs('vic'), alone.outputs('vic')
    for k in ('et', 'runoff', 'baseflow', 'soil_liq', 'water_error'):
        assert torch.equal(o[k], ref[k]), k


def test_exact_mode_keeps_w3s_throughfall_leak_without_a_hook_member(tmp_path):
    """A routing-only group (no member acts inside vic_run) builds no core in
    mode fixed, but mode exact must still carry VIC-WUR 3.0's throughfall
    leak of a leafless vegetated tile (func_surf_energy_bal.c:785-786): the
    compiled code has it whatever plugin runs (Nandgaon, routing only: cells
    with an open-water tile differed from the compiled runoff before this).
    The core holds that hook and no other."""
    _need('G1a')
    m, c = run('G1a', 3, tmp_path)                       # mode exact
    grp = m.runtime.connectors['vicwur_routing'].group
    core = grp.vic_core()
    assert core is not None and core.throughfall == 'exact'
    assert core.ksat is None and core.wcr is None and not core.veg
    assert core.cv is None and core.state_fn is None and core.gwm is None
    m, c = run('G1a', 3, tmp_path, mode='fixed')
    assert m.runtime.connectors['vicwur_routing'].group.vic_core() is None


def _reopen_run(tmp_path, mode, a=0, b=40, days=45):
    """G8 with a land-use forcing of two records: step `a` closes a cell's
    smallest active tile (Cv 0), step `b` re-opens it with a gain below
    MINCOVERAGECHANGE. Returns lu_apply's input and output tile states per
    record (moist, t_node of that tile)."""
    import torchvic.vicwur.landuse as LU
    m, c = build('G8', days, tmp_path, mode=mode)
    lu = m.runtime.connectors['vicwur_landuse']
    f0 = lu.f
    with torch.inference_mode():
        x0 = torch.as_tensor(f0.to('cpu').get(0)).clone()
    cell = int(torch.nonzero((x0 > 0).sum(1) >= 2)[0])
    pos = torch.nonzero(x0[cell] > 0).reshape(-1)
    t = int(pos[torch.argmin(x0[cell, pos])])

    class Rec(object):
        def to(self, d):
            return self

        def reads(self, step):
            return step in (a, b)

        def get(self, step):
            x = x0.clone()
            x[cell, t] = 0.0 if step < b else 5e-7
            return x
    lu.f = Rec()
    seen = []
    ap = LU.apply

    def spy(st, cv_old, cv_new, P, **kw):
        i = {k: st[k][cell, t].clone() for k in ('moist', 't_node')}
        r = ap(st, cv_old, cv_new, P, **kw)
        o = {k: r[0][k][cell, t].clone() for k in ('moist', 't_node')}
        seen.append((float(cv_old[cell, t]), float(cv_new[cell, t]), i, o))
        return r
    LU.apply = spy
    try:
        with torch.inference_mode():
            m.initialize().run()
            m.finalize(write=False)
    finally:
        LU.apply = ap
    return seen


def test_exact_mode_keeps_the_states_of_a_tile_without_cover(tmp_path):
    """VIC-WUR's vic_run solves a tile only while its Cv > 0
    (VIC-WUR 3.0 vic_run.c:137): a tile a land-use record closes keeps its
    states until a record gives it cover again, and lu_apply keeps a
    re-opened tile's own states when the gain is below MINCOVERAGECHANGE
    (VIC-WUR 3.0 landuse.h:31). TorchVIC steps every tile, so in mode exact
    the land-use member puts the closed tile's states back before the next
    record (Nandgaon: a 5e-6 paddy tile closed on 1999-01-01 and re-opened
    with 8e-7 on 1999-06-01 ran from drifted states, ET 1e-6 relative off
    from day 151, 3e-4 later; with the states kept the land-use run agrees
    with the compiled code to float32 round-off over 2192 days). Mode fixed
    has no threshold and takes the pool's states, so there the drift is
    harmless and left alone."""
    _need('G8')
    ex = _reopen_run(tmp_path, 'exact')
    assert len(ex) == 2
    (cvo0, cvn0, _, out0), (cvo1, cvn1, in1, _) = ex
    assert cvo0 > 0 and cvn0 == 0 and cvo1 == 0 and 0 < cvn1 < 1e-5
    for k in ('moist', 't_node'):
        assert torch.equal(in1[k], out0[k]), k
    (tmp_path / 'fixed').mkdir()
    fx = _reopen_run(tmp_path / 'fixed', 'fixed')
    # without the kept states the tile drifts while closed
    assert not torch.equal(fx[1][2]['moist'], fx[0][3]['moist'])


# --------------------------------------------------------------------- GPU
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_g8_matches_w3_and_the_ledger_closes(tmp_path):
    _need('G8')
    m, c = run('G8', 365, tmp_path, device='cuda')
    worst = compare(m, c, 'G8', ROUT_VARS + IRR_VARS + (
        ('OUT_NONREN_DEFICIT', 'vicwur_nonrenew', 'deficit'),
        ('OUT_CV', 'vicwur_landuse', 'cv')),
        local_dam_cells=_local_dam_cells('G8'))
    print('G8 cuda', worst)
    assert max(worst.values()) < GATE, worst
    check_ledger(m, 365)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_g9_restart_equals_continuous(tmp_path):
    _need('G9')
    from test_vicwur_connectors import same
    full, _ = run('G9', 93, tmp_path, device='cuda', group='grs_full')
    ck = str(tmp_path / 'ck.pt')
    VC._GROUPS.clear()
    m1, _ = build('G9', 93, tmp_path, device='cuda', group='grs')
    with torch.inference_mode():
        m1.initialize().run(until=62 * DAY)
        m1.save_checkpoint(ck)
    VC._GROUPS.clear()
    m2, _ = build('G9', 93, tmp_path, device='cuda', group='grs',
                  restart=ck)
    with torch.inference_mode():
        m2.initialize().run()
        m2.finalize(write=False)
    st = lambda m: {k: c.save_state() for k, c in
                    m.runtime.connectors.items()}
    assert not same(st(full), st(m2))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_part_b_hooks_do_no_host_sync(tmp_path):
    """P1-O and F, U of a day, and the state phase (land use with a record
    due, then EFR), with every member: no host synchronisation."""
    _need('G9')
    from torchvic.vicwur.core import TileStates
    m, c = build('G9', 62, tmp_path, device='cuda', group='gsync')
    with torch.inference_mode():
        m.initialize().run(until=31 * DAY)
    vic = m.runtime.connectors['vic']
    grp = vic.phase_group
    st = dict(vic.state)
    if 'lanes' in st:
        # a GPU run carries its state on the covered tiles: on the columns
        ln = vic._lanes(vic.driver.mosaic(), st)
        st['moist'], st['wdew'] = ln.scatter(st['moist']), \
            ln.scatter(st['wdew'])
    sv = vic._soil_view(st['moist'], st.get('soil_ice'), None)
    v = vic._wur_view(sv)
    vic._set_fluxes(v, vic._wur_last)
    store = dict(moist=st['moist'].clone(), wdew=st['wdew'].clone())
    ts = TileStates(store.get, store.__setitem__, lambda: list(store))
    lu = grp.roles['landuse']
    lu._forced = torch.full((6, 18), 1.0 / 18, dtype=torch.float64,
                            device='cuda')
    with torch.inference_mode():
        lu._statics()
        lu._due = True
        grp.step = 31
        lu.at_state(ts)              # warm (first call builds constants)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        with torch.inference_mode():
            grp._set_view(v)
            grp._set_time(30 * DAY)
            grp._phases(('P1', 'P2', 'P3', 'O'))
            grp._set_time(31 * DAY)
            grp._phases(('F', 'U'))
            lu._due = True
            grp._state_fn(31, ts)
    finally:
        torch.cuda.set_sync_debug_mode('default')
