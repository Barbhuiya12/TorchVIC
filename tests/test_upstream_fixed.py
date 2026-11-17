# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Upstream bugs fixed by default: each fixed mode keeps
the physical property the C code breaks. The exact modes are replayed
bitwise against the references in their own tests."""
import inspect

import torch

from torchvic.vicwur import dams as DM
from torchvic.vicwur import efr as E
from torchvic.vicwur.view import StepView
from torchvic.coupling.connectors import vicwur_setup as WS
from torchvic.coupling.connectors.vicwur_efr import VicWurEfrConnector
from torchvic.coupling.connectors.vicwur_dams import VicWurDamsConnector

F64 = torch.float64


def test_defaults_are_fixed():
    assert inspect.signature(WS.modules).parameters['mode'].default == 'fixed'
    assert inspect.signature(
        VicWurEfrConnector.__init__).parameters['exact'].default is False
    assert inspect.signature(
        VicWurDamsConnector.__init__).parameters['exact'].default is False
    assert inspect.signature(
        DM.GlobalDams.__init__).parameters['exact'].default is False
    assert inspect.signature(
        StepView.withdraw).parameters['floor'].default == 'liquid'


def test_pumping_never_takes_moisture_below_ice():
    """Upstream bug 3 (wu_run_local.c:421-433): fixed floor is the ice."""
    ice = torch.full((1, 1, 1, 3), 4.0, dtype=F64)
    v = StepView(torch.full((1, 1, 1, 3), 10.0, dtype=F64),
                 torch.tensor([[20.0, 30.0, 40.0]], dtype=F64),
                 torch.ones(1, 1, dtype=F64), torch.ones(1, 1, dtype=F64),
                 ice=ice)
    t, u = v.withdraw(2, torch.tensor([[[50.0]]], dtype=F64))
    assert float(v.moist[0, 0, 0, 2]) >= float(ice[0, 0, 0, 2])
    assert float(t) == 6.0 and float(u) == 44.0
    t2 = v.withdraw_cells(1, torch.tensor([0]), torch.full((1, 1, 1), 50.0,
                                                          dtype=F64))
    assert float(v.moist[0, 0, 0, 1]) >= 4.0 - 1e-15 and float(t2) == 6.0


def test_efr_fixed_uses_each_cells_own_parameters():
    """Upstream bug 5 (efr_force.c:94-98): the nonlinear baseflow term of
    cell i must not depend on cell 0's Ds, Ws, c."""
    n, g = 3, torch.Generator().manual_seed(2)
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g, dtype=F64)
    kw = dict(moist=(350. + 45 * torch.rand(n, 1, 1, generator=g,
                                             dtype=F64)),
              ice=torch.zeros(n, 1, 1, 1, dtype=F64),
              frost_fract=torch.ones(n, 1, dtype=F64),
              resid=r(.05, .1), depth=torch.ones(n, dtype=F64),
              max_moist=torch.full((n,), 400., dtype=F64),
              dsmax_year=r(5., 15.), ds=r(.1, .5), ws=r(.4, .8),
              c=r(1.5, 3.), cv=torch.ones(n, 1, dtype=F64),
              area_fract=torch.ones(n, 1, dtype=F64),
              req_baseflow=r(.05, 2.), req_discharge=torch.ones(n, dtype=F64))
    both = E.efr_step(**kw)
    ex = E.efr_step(**kw, bug=True)
    for i in range(n):
        one = {k: v[i:i + 1] for k, v in kw.items()}
        alone = E.efr_step(**one)
        assert torch.equal(both.frac[i], alone.frac[0])
        assert torch.equal(both.efr_moist[i], alone.efr_moist[0])
    tb = E.trial_baseflow(kw['moist'] - 0., E.search_grid()[0], kw['resid'] * kw['depth'] * 1000.,
                          kw['max_moist'], kw['dsmax_year'], kw['ds'], kw['ws'], kw['c'], kw['cv'],
                          kw['area_fract'])
    tx = E.trial_baseflow(kw['moist'] - 0., E.search_grid()[0], kw['resid'] * kw['depth'] * 1000.,
                          kw['max_moist'], kw['dsmax_year'], kw['ds'], kw['ws'], kw['c'], kw['cv'],
                          kw['area_fract'], bug=True)
    assert not torch.equal(tb[1:], tx[1:])           # the bug is visible


def test_dams_fixed_use_service_cell_area_and_conserve_mass():
    """Upstream bug 5 (dam_register.c:129-133, :200-228): fixed mode still
    closes storage change = inflow - release and stays in [0, capacity]."""
    d = DM.GlobalDams(torch.tensor([5.0, 200.0, 40.0], dtype=F64),
                      torch.ones(3, dtype=F64), [1990] * 3)
    assert d.exact is False
    g = torch.Generator().manual_seed(5)
    ys, ms = zip(*[(2000 + (t // 365), 1 + (t % 365) // 31 % 12)
                   for t in range(800)])
    cal = DM.make_calendar(ys, ms, 'standard', 1)
    for t in range(800):
        s0 = d.storage.clone()
        ring = torch.rand(3, 4, generator=g, dtype=F64) * 60
        ring, disch = d.run(cal.at(t), ring, ring.sum(1), 86400.0)
        assert (d.storage >= 0).all() and (d.storage <= d.capacity).all()
        resid = d.storage - s0 - d.inflow + d.release
        assert (resid[d.active].abs() <= 1e-12).all()


def test_throughfall_fixed_conserves_water(tmp_path):
    """Upstream bug 2: the 'fixed' throughfall loses no rain (water error
    equals the run without the flag); 'exact' leaks the cover share."""
    import test_vicwur_crop_feedback as T
    from torchvic.vicwur import core as VCORE
    c, _ = T._g0(tmp_path, 40)
    m = c.mosaic
    nc, nt = m.cv.shape
    k = int(torch.argmax(m.cv[0]))
    mask = torch.zeros(nc, nt, dtype=torch.bool)
    mask[:, k] = True
    veg = {'lai': torch.zeros(nc, nt, dtype=F64),
           'fcanopy': torch.full((nc, nt), 1e-4, dtype=F64)}
    run = lambda tf: c.run(p=c.params(), vicwur_core=VCORE.VicwurCore(
        veg=veg, veg_mask=mask, throughfall=tf))['water_error']
    with torch.no_grad():
        off, fx, ex = run(False), run('fixed'), run('exact')
    assert torch.equal(off, fx)
    assert float((off[0] - ex[0]).abs().max()) > 1e-5
