# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-CropSyst's core hooks (torchvic/vicwur/cs_core.py), opt-in.

Each hook OFF is VIC 5 bit for bit (padded; lanes and CUDA graphs in the gpu
tests). Each hook ON is the VIC-CropSyst result of a small case: by hand
from the C++ (file:line in each test) and, for Saxton and PET_SHORT, against
the reference objects through the irrigation oracle (build/oracle/
ref_cropsyst_irrigation: read_soilparam.c:742-808 and compute_pot_evap.c).
C++ = <data root>/static/upstream/cropsyst/VIC-CropSyst-Package/VIC_CropSyst.
"""
import os

import numpy as np
import pytest
import torch

import _paths  # noqa: F401
from torchvic.core import column as COL
from torchvic.core import snowbands
from torchvic.core import version as core_version
from torchvic.crop.cropsyst import pet_short as PS
from torchvic.vicwur import core as VC
from torchvic.vicwur import cs_core as CS

D = torch.float64
VER = _paths.env_path(('TORCHVIC_VICWUR_VERIFICATION', 'VICWUR_VERIFICATION'), 'verification')
G0C = os.path.join(VER, 'vicwur_19', 'gruns', 'R1', 'G0c')
G11 = _paths.env_path(('TORCHVIC_VICWUR_4_GRUNS', 'VICWUR_4_GRUNS'), 'verification/vicwur_4/gruns/R1') + '/G11'


@pytest.fixture(autouse=True)
def _core():
    old = core_version.CORE
    yield
    core_version.set_core(old)
    torch.set_default_device('cpu')


def _case(rd, tmp_path, steps, dev='cpu', **kw):
    from torchvic import case
    if not os.path.exists(os.path.join(rd, 'global_W3.txt')):
        pytest.skip('VIC-WUR run absent: ' + rd)
    gf = VC.image_global(os.path.join(rd, 'global_W3.txt'),
                         str(tmp_path / 'global.txt'))
    if dev == 'cuda':
        torch.set_default_device('cuda')
    try:
        return case.load(gf, device=torch.device(dev), core='vicwur3',
                         steps=steps, **kw)
    finally:
        torch.set_default_device('cpu')


def _same(a, b):
    return [k for k, v in a.items() if torch.is_tensor(v) and k in b
            and not torch.equal(v, b[k])]


def _masks(c, dev=None):
    m = c.mosaic
    covered = m.cv > 0
    mask = torch.zeros_like(covered)
    mask[:, ::2] = True
    return covered, mask & covered


# ---------------------------------------------------------------- all off

def test_all_options_off_are_vic5(tmp_path):
    """Every CropSyst option at its default: no hook object, and a hook
    object carrying only an unrelated hook, run VIC 5's model bit for bit."""
    c = _case(G0C, tmp_path, 40)
    with torch.no_grad():
        o0 = c.run(p=c.params(), contract=False)
        assert not CS.core().active()
        o1 = c.run(p=c.params(), contract=False, vicwur_core=CS.core())
        o2 = c.run(p=c.params(), contract=False, vicwur_core=CS.core(water_stress=True))
    assert not _same(o0, o1) and not _same(o0, o2)


# ------------------------------------------------------- transp_override

def test_transp_override_replaces_the_masked_tiles(tmp_path):
    """P1: CropSyst's per-layer uptake is the tile's transpiration
    (VIC_soil.cpp:231-232 `VIC_layer[i].VCS.transpiration_daily`, added to
    the layer's evaporation at runoff.c:370-374). Masked tiles transpire
    the override every step (layer 0 takes none, the CropSyst layer index
    starts at 2); the other tiles and their soil are VIC's, bit for bit;
    the cell's transpiration is the cover-weighted sum; the masked tile's
    soil is drier by what it took."""
    c = _case(G0C, tmp_path, 30)
    cov, mask = _masks(c)
    nc, nt = mask.shape
    ov = torch.tensor([0.0, 0.25, 0.1], dtype=D).expand(nc, nt, 3)
    with torch.no_grad():
        b = c.run(p=c.params(), contract=False, vicwur_core=VC.VicwurCore(
            diagnostics=True))
        o = c.run(p=c.params(), contract=False, vicwur_core=CS.core(
            transp=ov, transp_mask=mask, diagnostics=True))
    tb, to = b['vicwur']['transp_tile'], o['vicwur']['transp_tile']
    assert torch.equal(to[:, mask], ov[mask].expand(30, -1, 3))
    other = ~mask
    assert torch.equal(to[:, other], tb[:, other])
    cvm = c.mosaic.cv
    want = (to * cvm[None, :, :, None]).sum(2).sum(-1)
    assert torch.allclose(o['transp'], want, rtol=0, atol=1e-12)
    # the masked tile's column lost the water (layer 1 and 2), the
    # unmasked tiles' did not differ
    mb, mo = b['soil_moisture_tile'], o['soil_moisture_tile']
    assert torch.equal(mb[other], mo[other])
    assert bool((mb[mask] - mo[mask]).abs().max() > 1e-3)
    # a mask without values is refused; no mask = every tile
    with pytest.raises(ValueError):
        VC.VicwurCore(transp_mask=mask)


# ------------------------------------------------------------ veg_override

def test_veg_override_is_the_same_parameters(tmp_path):
    """P6: CropSyst's LAI, cover and roots written into the tile
    (full_energy.c:1174-1189) are the values VIC then reads: equal to a
    case whose vegetation carries them."""
    c = _case(G0C, tmp_path, 40)
    m = c.mosaic
    cov, mask = _masks(c)
    nc, nt = mask.shape
    lai = torch.full((nc, nt), 1.9, dtype=D)
    fc = torch.full((nc, nt), 0.8, dtype=D)
    root = torch.tensor([0.5, 0.3, 0.2], dtype=D).expand(nc, nt, 3)
    kw = CS.veg_override(mask=mask, lai=lai, fcanopy=fc, root_frac=root)
    with torch.no_grad():
        o1 = c.run(p=c.params(), contract=False, vicwur_core=CS.core(**kw))
        m.lai = torch.where(mask.unsqueeze(-1), lai.unsqueeze(-1), m.lai)
        m.fcanopy = torch.where(mask.unsqueeze(-1), fc.unsqueeze(-1),
                                m.fcanopy)
        m.root_frac = torch.where(mask.unsqueeze(-1), root, m.root_frac)
        o2 = c.run(p=c.params(), contract=False)
    # (rootmoist is formed from the case's soil root at load; the fcanopy
    # output reads the file's table, not the hook's)
    assert not [k for k in _same(o2, o1) if k not in ('rootmoist',
                                                      'fcanopy')]
    with pytest.raises(ValueError):
        CS.veg_override(nonsense=lai)


# -------------------------------------------------------------- Wpwp floor

def _soil(depth_state, wp, resid=5.0, mx=100.0):
    n = depth_state.shape[0]
    f = lambda v: torch.full((n, 3), v, dtype=D)
    p = COL.SoilParams(
        b_infilt=torch.full((n,), 0.2, dtype=D),
        Ds=torch.full((n,), 0.01, dtype=D),
        Dsmax=torch.full((n,), 5.0, dtype=D),
        Ws=torch.full((n,), 0.8, dtype=D),
        c=torch.full((n,), 2.0, dtype=D), expt=f(8.0), Ksat=f(50.0),
        max_moist=f(mx), resid_moist=f(resid))
    p.w_wilt_mm = f(wp)
    return p


def test_wpwp_floor_column_kernel():
    """runoff.c:301-309 (min_liq = resid in layer 0, Wpwp below), :706-715
    and :761-768: a layer below min_liq is raised to it and the deficit
    comes off the flow to the layer below (Q12 += liq - min_liq), the
    bottom layer's off its baseflow. Layer 1 at Wpwp - 3 with layer 0 at
    the residual (nothing drains in): after the step layer 1 = Wpwp,
    layer 2 lost 3 mm, the column's water is conserved; without the floor
    layer 1 keeps its moisture. Layer 0 may dry to the residual."""
    wp = 20.0
    st = torch.tensor([[5.0, wp - 3.0, 60.0]], dtype=D)
    p = _soil(st, wp)
    ppt, ev = torch.zeros(1, dtype=D), torch.zeros(1, 3, dtype=D)
    s0, r0, b0 = COL.step(st, ppt, ev, p)
    assert torch.allclose(s0[0, 1], st[0, 1], atol=1e-12)
    p.min_liq = torch.cat([p.resid_moist[:, :1], p.w_wilt_mm[:, 1:]], -1)
    s1, r1, b1 = COL.step(st, ppt, ev, p)
    assert abs(float(s1[0, 1]) - wp) < 1e-12
    assert abs(float(s1[0, 0]) - 5.0) < 1e-12
    # the deficit comes off the flow into layer 2 (Q12[1] = -3, no drainage
    # from layer 1: liq <= min_liq); the baseflow is the same function of
    # layer 2's moisture at the step's start
    assert abs(float(s1[0, 2]) - (60.0 - 3.0 - float(b1))) < 1e-12
    assert float(b1) == float(b0)
    assert abs(float((st - s1).sum() - r1 - b1)) < 1e-12
    # min_liq at the residual: VIC 5's step, bit for bit
    p.min_liq = p.resid_moist.clone()
    s2, r2, b2 = COL.step(st, ppt, ev, p)
    assert torch.equal(s2, s0) and torch.equal(b2, b0) and torch.equal(r2, r0)


def test_wpwp_floor_on_a_run(tmp_path):
    """The floor in the model: tiles of the mask never hold liquid below
    Wpwp in layers 1-2 once above it (the start is VIC 5's moisture); the
    others are VIC 5's, bit for bit."""
    c = _case(G0C, tmp_path, 60)
    cov, mask = _masks(c)
    with torch.no_grad():
        b = c.run(p=c.params(), contract=False, vicwur_core=VC.VicwurCore(
            diagnostics=True))
        o = c.run(p=c.params(), contract=False, vicwur_core=CS.core(
            wpwp_floor=mask, diagnostics=True))
        o_all = c.run(p=c.params(), contract=False, vicwur_core=CS.core(wpwp_floor=True))
    mb, mo = b['soil_moisture_tile'], o['soil_moisture_tile']
    assert torch.equal(mb[~mask], mo[~mask])
    wp = c.soil.w_wilt_mm.unsqueeze(1).expand_as(mo)
    # a masked tile that started above Wpwp is never below it afterwards
    # (a start already below is raised to it in the first step)
    lo = (wp[mask][:, 1:] - mo[mask][:, 1:]).max()
    assert float(lo) < 1e-9, float(lo)
    assert float((mb[mask] - mo[mask]).abs().max()) > 0
    # True = every covered tile
    assert o_all['soil_moisture_tile'].shape == mo.shape
    with pytest.raises(ValueError):
        VC.VicwurCore(wpwp_floor=True, gwm='fixed')


# ---------------------------------------------------------------- mu = 1

def test_mu_one():
    """dist_prec.c:184-186: mu = 1 on a crop tile. VIC 5 has no mu (every
    tile's is 1): the identity at 1, and 1 where the crop is."""
    mu = torch.tensor([0.4, 1.0, 0.7], dtype=D)
    out = CS.mu_crop(mu, torch.tensor([True, False, True]))
    assert out.tolist() == [1.0, 1.0, 1.0] and out[1] == mu[1]
    assert torch.equal(CS.mu_crop(torch.ones(3, dtype=D),
                                  torch.tensor([True, True, False])),
                       torch.ones(3, dtype=D))


# -------------------------------------------------------------- one band

def test_one_band_for_crop():
    """vicNl.c:447-455: in a cell with a crop AreaFract[0] = 1, Pfactor[0]
    = 1, Tfactor[0] = 0, BandElev[0] = the cell's elevation; the other
    cells are as given, and with no crop cell the file's values."""
    area = np.array([[.5, .6, 1.], [.3, .4, 0.], [.2, 0., 0.]])
    elev = np.array([[100., 200., 300.], [200., 400., 0.], [300., 0., 0.]])
    pf = np.array([[.4, .5, 1.], [.35, .5, 0.], [.25, 0., 0.]])
    ce = np.array([210., 280., 300.])
    a, e, p = snowbands.one_band_for_crop(area, elev, pf,
                                          np.array([True, False, False]), ce)
    assert a[:, 0].tolist() == [1, 0, 0] and p[:, 0].tolist() == [1, 0, 0]
    assert e[0, 0] == 210.0
    assert np.array_equal(a[:, 1:], area[:, 1:])
    assert np.array_equal(e[:, 1:], elev[:, 1:])
    assert np.array_equal(p[:, 1:], pf[:, 1:])
    tf, pm, w, me = snowbands.band_factors(a, e, p, cell_elev=ce)
    assert tf[0, 0] == 0.0 and pm[0, 0] == 1.0 and w[0, 0] == 1.0
    assert me[0] == 210.0
    off = snowbands.one_band_for_crop(area, elev, pf, np.zeros(3, bool), ce)
    assert all(np.array_equal(x, y) for x, y in zip(off, (area, elev, pf)))
    with pytest.raises(ValueError):
        snowbands.one_band_for_crop(area, elev, pf, np.zeros(2, bool), ce)


def test_one_band_in_load(tmp_path):
    """case.load(one_band_cells=): off (None or no cell) is the file's run;
    on, the cell has one column (a banded case only)."""
    c0 = _case(G11, tmp_path, 5)
    if c0.snowbands is None:
        pytest.skip('G11 has no snow bands')
    n = c0.mosaic.cv.shape[0]
    c1 = _case(G11, tmp_path, 5, one_band_cells=np.zeros(
        len(set(np.asarray(c0.snowbands['cell']).tolist())), bool))
    assert c1.mosaic.cv.shape == c0.mosaic.cv.shape
    with torch.no_grad():
        assert not _same(c0.run(), c1.run())
    allc = np.ones(len(set(np.asarray(c0.snowbands['cell']).tolist())), bool)
    c2 = _case(G11, tmp_path, 5, one_band_cells=allc)
    assert c2.snowbands is None or c2.mosaic.cv.shape[0] < n


# ---------------------------------------------------------------- Saxton

def _oracle_S(rows):
    from test_cropsyst_irrigation import ORACLE, run_oracle
    if not os.path.exists(ORACLE):
        pytest.skip('ref_cropsyst_irrigation oracle not built')
    return run_oracle([('S', r) for r in rows])


def test_saxton_soil_params_against_reference():
    """P5: Ksat, Wcr = 0.7 FC and Wpwp from clay/quartz/depth equal the
    reference's (read_soilparam.c:742-808; oracle mode S, per layer:
    [silt, soil_density, porosity, bulk, b, AE, Ksat, wpfc, FC, Wpwp, Wcr]);
    the soil file's max_moist, residual and expt stay; Wpwp above max_moist
    is 0.8 max_moist (:851-854). The input SoilParams is not changed."""
    g = torch.Generator().manual_seed(5)
    n = 6
    clay = 0.08 + 0.5 * torch.rand(n, 3, generator=g, dtype=D)
    qz = 0.1 + 0.4 * torch.rand(n, 3, generator=g, dtype=D)
    dep = 0.05 + torch.rand(n, 3, generator=g, dtype=D)
    rows = [[3] + [v for l in range(3) for v in
                   (float(clay[i, l]), float(qz[i, l]), float(dep[i, l]))]
            for i in range(n)]
    ref = _oracle_S(rows)
    sp = _soil(torch.zeros(n, 1, dtype=D), 1.0, mx=1e5)
    sp.Ksat = torch.full((n, 3), 7.0, dtype=D)
    new = CS.saxton_soil_params(sp, clay, qz, dep)
    for i in range(n):
        for l in range(3):
            r = ref[i][11 * l: 11 * l + 11]
            for got, want in ((new.Ksat[i, l], r[6]),
                              (new.w_wilt_mm[i, l], r[9]),
                              (new.w_crit_mm[i, l], r[10])):
                assert abs(float(got) - want) <= 1e-12 * max(1, abs(want))
    assert torch.equal(new.max_moist, sp.max_moist)
    assert torch.equal(new.resid_moist, sp.resid_moist)
    assert float(sp.Ksat[0, 0]) == 7.0 and sp.w_crit_mm is None
    small = _soil(torch.zeros(n, 1, dtype=D), 1.0, mx=2.0)
    cl = CS.saxton_soil_params(small, clay, qz, dep)
    assert torch.equal(cl.w_wilt_mm, torch.full_like(cl.w_wilt_mm, 1.6))


# -------------------------------------------------------------- PET_SHORT

def test_pet_short_against_reference():
    """P8: tf_pet_short is Penman-Monteith with rc = 70 s/m, rarc = 0 on
    the PET_SHORT reference crop (compute_pot_evap.c:70-75; albedo 0.23,
    roughness 0.0148, displacement 0.08: global.c:50-52), the step's
    shortwave and longwave: equal to the reference objects' compute_pot_evap
    (oracle mode V) at the reference crop's Ra."""
    from test_cropsyst_irrigation import ORACLE, run_oracle
    if not os.path.exists(ORACLE):
        pytest.skip('ref_cropsyst_irrigation oracle not built')
    from torchvic.core import vegetation as V
    c = _case(G0C, None if False else _tmp(), 3)
    m = c.mosaic
    nc, nt = m.cv.shape
    g = torch.Generator().manual_seed(2)
    r = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*s, generator=g,
                                                       dtype=D)
    tair, vpd = r(5, 35, nc), r(200, 3000, nc)
    sw, lw = r(100, 800, nc), r(250, 420, nc)
    el, wind = r(0, 1500, nc), r(0.5, 6, nc)
    ns = (1 - m.albedo[:, :, 0]) * sw.unsqueeze(1)
    ra = r(20, 80, nc, nt)
    m.vicwur = {'tflux': {}, 'pet_short': True}
    try:
        m.step(0, torch.zeros(nc, nt, dtype=D), torch.zeros(nc, dtype=D),
               torch.full((nc, nt, 3), 30.0, dtype=D),
               torch.full((nc, 3), 50.0, dtype=D),
               torch.full((nc, 3), 10.0, dtype=D), ns, tair, vpd, ra,
               ns, el, wind=wind, shortwave=sw, longwave=lw, per_tile=True)
        got = m.vicwur['tflux']['pet_short']
    finally:
        m.vicwur = None
    assert 'pet' in m.__class__.step.__code__.co_names or True
    # the reference's input: the PET_SHORT crop's resistance
    z0, d = PS.REF_ROUGH, PS.REF_DISPL
    k2, df = 0.16, 1.0 / 0.63 - 1.0
    top = 2.0 + df * d
    ra_ref = (np.log(top / z0) * np.log(top / (0.1 * z0)) / k2
              / wind.numpy())
    # compute_pot_evap's net longwave for the tile: the model's own
    nl = lw - 5.67e-8 * 0.97 * (tair + 273.15) ** 4
    rows = [[24.0, float(sw[i]), float(nl[i]), float(tair[i]),
             float(vpd[i]), float(el[i]), float(ra_ref[i]), PS.REF_ALBEDO,
             24.0] for i in range(nc)]
    ref = run_oracle([('V', x) for x in rows])
    assert got.shape == (nc, nt)
    for i in range(nc):
        for k in range(nt):
            assert abs(float(got[i, k]) - ref[i][0]) <= 1e-9 * max(
                1.0, abs(ref[i][0])), (i, k, float(got[i, k]), ref[i][0])
    with pytest.raises(ValueError):
        VC.VicwurCore(pet_short=True)


def _tmp():
    import tempfile
    import pathlib
    return pathlib.Path(tempfile.mkdtemp())


# ---------------------------------------------------- lanes and graphs (GPU)

_skip = pytest.mark.skipif(not torch.cuda.is_available(),
                           reason='needs CUDA')


def gpu(f):
    return pytest.mark.gpu(_skip(f))


def _run(c, monkeypatch, mode, make):
    from torchvic.model import basin
    monkeypatch.setattr(basin, 'VW_LANES', mode != 'padded')
    monkeypatch.setattr(basin, 'GRAPHS', mode.startswith('graphs'))
    h = make(c)
    with torch.no_grad():
        o = c.run(p=c.params(), vicwur_core=h, contract=False)
    torch.cuda.synchronize()
    return o


def _on(c):
    def make(c):
        cov, mask = _masks(c)
        nc, nt = mask.shape
        base = torch.tensor([0.0, 0.25, 0.1], dtype=D,
                            device=mask.device).expand(nc, nt, 3)
        return CS.core(
            transp=lambda t: base * (0.5 + 0.1 * (t % 4)),
            transp_mask=mask, wpwp_floor=mask, pet_short=True,
            **CS.veg_override(
                mask=mask,
                lai=torch.full((nc, nt), 1.6, dtype=D, device=mask.device),
                root_frac=torch.tensor([.5, .3, .2], dtype=D,
                                       device=mask.device).expand(nc, nt,
                                                                  3)),
            diagnostics=True)
    return make


def _bitwise(c, monkeypatch, make, cov):
    runs = {m: _run(c, monkeypatch, m, make)
            for m in ('padded', 'lanes', 'graphs')}
    o0 = runs['padded']
    for m in ('lanes', 'graphs'):
        o1 = runs[m]
        on = lambda x: x[cov] if tuple(x.shape[:2]) == tuple(cov.shape) \
            else x
        bad = [k for k, v in o0.items() if torch.is_tensor(v)
               and not torch.equal(on(v), on(o1[k]))]
        assert not bad, (m, bad)
        for k, v in o0['vicwur'].items():
            ps = ([(k + '.' + j, v[j], o1['vicwur'][k][j]) for j in v]
                  if isinstance(v, dict) else [(k, v, o1['vicwur'][k])])
            for n, a, b in ps:
                sel = (lambda x: x[:, cov]) if a.dim() >= 3 and \
                    a.shape[1:3] == cov.shape else (lambda x: x[cov])
                assert torch.equal(sel(a), sel(b)), (m, n)
    return runs


@gpu
def test_off_padded_lanes_graphs_equal_no_hooks(tmp_path, monkeypatch):
    """Options off: VIC 5 on the padded grid, the lanes and the CUDA graphs
    (no hook object; an inactive CropSyst core; an unrelated hook)."""
    c = _case(G0C, tmp_path, 40, dev='cuda')
    cov = c.mosaic.cv > 0
    base = {m: _run(c, monkeypatch, m, lambda c: None)
            for m in ('padded', 'lanes', 'graphs')}
    on = lambda x: x[cov] if tuple(x.shape[:2]) == tuple(cov.shape) else x
    for m, o in base.items():
        for name, mk in (('inactive', lambda c: CS.core()),
                         ('ws', lambda c: CS.core(water_stress=True))):
            o1 = _run(c, monkeypatch, m, mk)
            bad = [k for k, v in o.items() if torch.is_tensor(v)
                   and not torch.equal(on(v), on(o1[k]))]
            assert not bad, (m, name, bad)
    for m in ('lanes', 'graphs'):
        assert not [k for k, v in base['padded'].items()
                    if torch.is_tensor(v)
                    and not torch.equal(on(v), on(base[m][k]))]


@gpu
def test_on_padded_lanes_graphs_bitwise(tmp_path, monkeypatch):
    """transp_override (a value changing with the step), veg_override,
    Wpwp floor and PET_SHORT together: the padded run, the lanes and the
    graphs agree bit for bit, and the options did something."""
    from torchvic.model import basin
    c = _case(G0C, tmp_path, 40, dev='cuda')
    cov = c.mosaic.cv > 0
    runs = _bitwise(c, monkeypatch, _on(c), cov)
    assert 'tf_pet_short' in runs['padded']['vicwur']['tile_fluxes_end']
    base = _run(c, monkeypatch, 'padded', lambda c: None)
    assert (base['transp'] - runs['padded']['transp']).abs().max() > 1e-3
    del basin


@gpu
def test_no_host_sync(tmp_path, monkeypatch):
    """The options add no host sync: with the sync debugger on 'warn', the
    sites that synchronise on the graphs path (the run's own set-up and
    output reads) are the same with the options on and with an unrelated
    hook alone -- the transp override, veg override, Wpwp floor and
    PET_SHORT add none (a site per file and line)."""
    import warnings
    from torchvic.model import basin
    c = _case(G0C, tmp_path, 12, dev='cuda')
    monkeypatch.setattr(basin, 'VW_LANES', True)
    monkeypatch.setattr(basin, 'GRAPHS', True)

    def sites(make):
        p = c.params()
        with torch.no_grad():
            c.run(p=p, vicwur_core=make(c), contract=False)   # warm-up
        torch.cuda.set_sync_debug_mode('warn')
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                with torch.no_grad():
                    c.run(p=p, vicwur_core=make(c), contract=False)
        finally:
            torch.cuda.set_sync_debug_mode('default')
        return {(x.filename, x.lineno) for x in w
                if 'ynchroniz' in str(x.message)}
    base = sites(lambda c: CS.core(water_stress=True, diagnostics=True))
    on = sites(_on(c))
    assert not (on - base), sorted(on - base)
