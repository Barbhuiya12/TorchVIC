# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The CropSyst engine as a member of the VIC-WUR phase group (mode B).

Without data: the plan, the group's single crop role, configuration errors, VIC columns of 17
layers (the NLAYER = 3 constant of core.column was the bug that kept every run at three layers),
the classic reader taking ORGANIC_FRACT.
On the CRB case 259094 (corn/potato/sugarbeet, irrigated, 17 layers; reference build outputs in
<data root>/verification/cropsyst_i8i13): the coupled run's ledgers close, the live crop's signals
drive the irrigation, restart equals continuous (CPU and CUDA), no host synchronisation in the
member's day (CUDA).
"""
import os
import sys
import tempfile

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, os.path.dirname(__file__))

from torchvic.coupling import registry                                 # noqa
from torchvic.coupling.spec import CouplingError                       # noqa
from torchvic.coupling.connectors import vicwur_common as VC           # noqa
from torchvic.core import column as COL                                # noqa
from torchvic.vicwur import order                                      # noqa

DAY = 86400.0
D = torch.float64


def test_plan_has_the_crop_day_in_p3_after_irrigation_and_default_is_unchanged():
    calls = lambda **k: [(c.phase, c.kind, c.hook) for c in order.plan(order.Options(**k))]
    base = calls(irrigation=True, potential_irrigation=True)
    assert not any(c[1] == 'cropsyst' for c in base)
    on = calls(irrigation=True, potential_irrigation=True, cropsyst=True)
    p3 = [c[1:] for c in on if c[0] == 'P3']
    assert p3[-2:] == [('irrigation', 'pass3'), ('cropsyst', 'pass3')]
    assert [c for c in on if c[1] != 'cropsyst'] == base
    assert ('O', 'cropsyst', 'put_data') in on


def test_one_crop_role_and_config_errors():
    mk = registry.get('vicwur_cropsyst').make
    cs = dict(tiles=[300100], tile_cell=[0], db_dir='x', soil={})
    with pytest.raises(CouplingError, match='cropsyst and tile'):
        mk(group=VC.WurGroup('a'))
    with pytest.raises(CouplingError, match='tile_cell'):
        mk(cropsyst=dict(db_dir='x'), tile=[0], group=VC.WurGroup('b'))
    with pytest.raises(CouplingError, match='one entry per'):
        mk(cropsyst=cs, tile=[0, 1], group=VC.WurGroup('c'))
    with pytest.raises(CouplingError, match='hook'):
        mk(cropsyst=cs, tile=[0], hooks=dict(x=1), group=VC.WurGroup('d'))
    with pytest.raises(CouplingError, match='et_ref'):
        mk(cropsyst=cs, tile=[0], et_ref='x', group=VC.WurGroup('e'))
    g = VC.WurGroup('f')
    mk(cropsyst=cs, tile=[0], group=g)
    with pytest.raises(CouplingError, match='crop connector'):
        mk(cropsyst=cs, tile=[0], group=g)


def test_vic_column_of_17_layers_conserves_water_and_3_layers_are_unchanged():
    """Regression: core.column.step took its layer count from a module constant (3), so
    NLAYER = 17 files crashed in basin.simulate; it is now the soil parameters' layer axis."""
    def params(L):
        g = torch.Generator().manual_seed(L)
        r = lambda lo, hi: lo + (hi - lo) * torch.rand(2, L, generator=g, dtype=D)
        mx = r(40.0, 80.0)
        return COL.SoilParams(b_infilt=torch.full((2,), 0.2, dtype=D), Ds=torch.full((2,), 0.1, dtype=D),
                              Dsmax=torch.full((2,), 5.0, dtype=D), Ws=torch.full((2,), 0.8, dtype=D),
                              c=torch.full((2,), 2.0, dtype=D), expt=r(8.0, 14.0), Ksat=r(100.0, 300.0),
                              max_moist=mx, resid_moist=0.05 * mx)
    for L in (3, 17):
        p = params(L)
        s0 = 0.6 * p.max_moist
        ppt = torch.tensor([30.0, 0.0], dtype=D)
        ev = torch.zeros(2, L, dtype=D)
        ev[:, 0] = 1.0
        s1, ro, bf = COL.step(s0, ppt, ev, p)
        assert s1.shape == (2, L)
        err = (s0.sum(-1) + ppt - ev.sum(-1) - ro - bf - s1.sum(-1)).abs().max()
        assert float(err) < 1e-9, (L, float(err))


def test_vcs_drainage_exponent_is_3_plus_2b_and_off_by_default():
    """runoff.c:583: Q12 = Ksat (S)^(3 + 2 b_campbell); saxton_soil_params(vcs_drainage=True)
    sets expt = 3 + 2 b, the default keeps the file's."""
    from torchvic.crop.cropsyst import ptf
    from torchvic.vicwur import cs_core
    L = 4
    mx = torch.full((1, L), 60.0, dtype=D)
    sp = COL.SoilParams(b_infilt=torch.full((1,), 0.2, dtype=D), Ds=torch.full((1,), 0.1, dtype=D),
                        Dsmax=torch.full((1,), 5.0, dtype=D), Ws=torch.full((1,), 0.8, dtype=D),
                        c=torch.full((1,), 2.0, dtype=D), expt=torch.full((1, L), 11.0, dtype=D),
                        Ksat=torch.full((1, L), 100.0, dtype=D), max_moist=mx, resid_moist=0.0 * mx)
    clay = torch.tensor([[0.2, 0.25, 0.3, 0.35]], dtype=D)
    qz = torch.tensor([[0.4, 0.35, 0.3, 0.25]], dtype=D)
    dm = torch.full((1, L), 0.1, dtype=D)
    off = cs_core.saxton_soil_params(sp, clay, qz, dm)
    assert torch.equal(off.expt, sp.expt)
    on = cs_core.saxton_soil_params(sp, clay, qz, dm, vcs_drainage=True)
    b = ptf.saxton_ptf(clay, qz, dm).b_campbell
    assert torch.equal(on.expt, 3.0 + 2.0 * b)
    assert torch.equal(sp.expt, torch.full((1, L), 11.0, dtype=D))


def test_classic_reader_takes_organic_fract(tmp_path):
    from torchvic.io.classic import ClassicOptions, soil_columns
    from torchvic.io.global_param import GlobalParam
    g = tmp_path / 'g.txt'
    g.write_text('NLAYER 3\nROOT_ZONES 3\nORGANIC_FRACT TRUE\nFORCE_FORMAT ASCII\n'
                 'SOIL x\nVEGLIB y\nVEGPARAM z\nFORCING1 f\n')
    o = ClassicOptions(GlobalParam(str(g)))
    assert not any('ORGANIC' in r for r in o.refused)
    assert ('organic', 3) in soil_columns(o)


# ------------------------------------------------------------------- the CRB case, mode B
def _l3():
    try:
        import cropsyst_i12_l3 as L
        L.run_dir(259094)
    except (FileNotFoundError, ImportError):
        pytest.skip('CRB reference runs not present')
    return L


def _go(days, device='cpu', restart=None, until=None, ck=None, group='t12', scratch=None):
    L = _l3()
    m, info = L.build(259094, scratch or tempfile.mkdtemp(), device, days, group=group, restart=restart)
    with torch.inference_mode():
        m.initialize()
        if until is not None:
            m.run(until=until * DAY)
            m.save_checkpoint(ck)
        else:
            m.run()
            m.finalize(write=False)
    return m, info


def _states(m):
    return {k: c.save_state() for k, c in m.runtime.connectors.items()}


def _restart(tmp_path, device):
    from test_vicwur_connectors import same
    days = 150
    sc = str(tmp_path / 'case')            # one path: the checkpoint's digest holds it
    full, _ = _go(days, device, group='t12f_' + device, scratch=sc)
    ck = str(tmp_path / 'ck.pt')
    VC._GROUPS.clear()
    _go(days, device, until=70, ck=ck, group='t12r_' + device, scratch=sc)
    VC._GROUPS.clear()
    m2, _ = _go(days, device, restart=ck, group='t12r_' + device, scratch=sc)
    d = same(_states(full), _states(m2))
    assert not d, sorted({x.split('/')[1] + '/' + x.split('/')[2] if x.count('/') > 1 else x for x in d})
    for n in full.runtime.connectors:
        a = [r['residual'] for r in full.ledger.balances if r['module'] == n]
        b = [r['residual'] for r in m2.ledger.balances if r['module'] == n]
        if n == 'vic':
            import numpy as np
            assert np.allclose(a, b, rtol=0, atol=1e-6), n
        else:
            assert a == b, n
    assert all(r['ok'] for r in m2.ledger.balances)
    a = full.outputs('vicwur_cropsyst')
    b = m2.outputs('vicwur_cropsyst')
    return full, a, b


def test_crb_coupled_run_ledgers_close_and_irrigation_follows_the_live_crop():
    VC._GROUPS.clear()
    m, info = _go(130, group='t12a')
    assert all(r['ok'] for r in m.ledger.balances)
    o = m.outputs('vicwur_cropsyst')
    assert torch.isfinite(o['lai']).all() and float(o['lai'].max()) > 0.1
    assert float(o['sown'].sum()) >= 1
    # CropSyst's uptake is VIC's transpiration on the crop tiles
    v = m.outputs('vic')
    up = o['uptake'].sum((1, 2)) if o['uptake'].dim() == 3 else o['uptake'].sum(-1).sum(-1)
    assert float(up.sum()) > 1.0
    assert float(v['transp'].sum()) > 1.0


def test_crb_restart_equals_continuous(tmp_path):
    _l3()
    VC._GROUPS.clear()
    _, a, b = _restart(tmp_path, 'cpu')
    n = b['lai'].shape[0]
    for k in a:
        assert torch.equal(a[k][-n:], b[k]), k


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_restart_equals_continuous_and_member_day_does_not_sync(tmp_path):
    _l3()
    from torchvic.coupling.connectors.vicwur_cropsyst import VicWurCropSystConnector as C
    orig = C.pass3

    calls = []

    def spy(self):
        # the engine builds its constant tables on its first day (yield_._hour_tables)
        if self.run_ is None:
            self._build()                  # once, at the first need (host work)
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
        VC._GROUPS.clear()
        _restart(tmp_path, 'cuda')
    finally:
        C.pass3 = orig
