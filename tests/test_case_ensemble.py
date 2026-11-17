"""The case loader and its ensemble axis.

An ensemble member must be the same model as a standalone run -- if it is
not, every calibration built on it optimises a different model from the one
that will be reported.
"""
import os
import sys

import pytest
import torch

from torchvic import case  # noqa: E402

import _paths                                           # noqa: E402
WG = _paths.WG_FULL
DEVICE = torch.empty(0).device


def _case(**kw):
    if not os.path.exists(WG):
        pytest.skip('the Walnut Gulch global file is not present')
    return case.load(WG, device=DEVICE, steps=kw.pop('steps', 32), **kw)


def test_a_member_is_the_same_model_as_a_standalone_run():
    c = _case()
    n = 4
    e = c.ensemble(n)
    assert e.n_cells == n * c.n_cells
    scales = torch.tensor([0.7, 1.0, 1.3, 1.6], dtype=c.dtype, device=DEVICE)
    p = e.member_params({'rmin_scale': scales})
    out = e.run(p=p)
    et = e.split_members(out['et'], n)                  # [time, member, cell]
    assert et.shape == (c.n_steps, n, c.n_cells)
    for m in range(n):
        one = c.run(p=c.params(rmin_scale=float(scales[m])))
        for key in ('et', 'runoff', 'baseflow', 'swe'):
            a = e.split_members(out[key], n)[:, m]
            # Not bit-exact by right: the same arithmetic over a longer
            # cell axis reduces in a different order, and on CPU that shows
            # as ~1e-13 mm on runoff (it was exactly 0 on CUDA). The bound
            # is what floating point allows, not a fitted tolerance.
            worst = float((a - one[key]).abs().max())
            assert worst < 1e-9, \
                'member %d differs from a standalone run in %s by %.3e' \
                % (m, key, worst)
    # and the members are not all the same model
    spread = et.sum(0).mean(-1)
    assert float(spread.max() - spread.min()) > 1e-6


def test_each_member_routes_to_its_own_outlet():
    c = _case()
    n = 3
    e = c.ensemble(n)
    out = e.run(p=e.member_params(
        {'rmin_scale': torch.tensor([0.6, 1.0, 1.8], dtype=c.dtype, device=DEVICE)}))
    q = out['discharge_mm']
    assert q.shape[1] == n
    assert float((q[:, 0] - q[:, 2]).abs().max()) > 0.0


def test_gradients_reach_p_parameters_and_soil_parameters():
    c = _case(steps=24)
    p = c.params()
    p['rmin_scale'] = p['rmin_scale'].clone().requires_grad_(True)
    c.run(p=p)['et'].sum().backward()
    assert p['rmin_scale'].grad is not None and float(p['rmin_scale'].grad) != 0.0
    for name in ('b_infilt', 'Ds', 'Dsmax', 'Ws', 'expt'):
        orig = getattr(c.soil, name)
        leaf = orig.detach().clone().requires_grad_(True)
        setattr(c.soil, name, leaf)
        try:
            o = c.run()
            (o['runoff'].sum() + o['baseflow'].sum()).backward()
            assert leaf.grad is not None, name
            assert float(leaf.grad.abs().max()) > 0.0, \
                'no gradient reaches the soil parameter %s' % name
        finally:
            setattr(c.soil, name, orig)


def test_soil_parameters_vary_by_member_and_match_a_standalone_run():
    c = _case(steps=32)
    n = 3
    e = c.ensemble(n)
    b = torch.tensor([0.05, 0.2, 0.6], dtype=c.dtype, device=DEVICE)
    dsm = torch.tensor([2.0, 10.0, 30.0], dtype=c.dtype, device=DEVICE)
    p = e.member_params({'b_infilt': b, 'Dsmax': dsm})
    assert not torch.equal(e.soil.b_infilt, c.ensemble(n).soil.b_infilt)
    assert torch.equal(c.soil.b_infilt, c.soil.b_infilt.clone())   # base untouched
    out = e.run(p=p)
    q = e.split_members(out['runoff'], n).sum(0).mean(-1)
    assert float(q.max() - q.min()) > 1e-6, 'soil parameters changed nothing'
    for m in range(n):
        old = (c.soil.b_infilt, c.soil.Dsmax)
        c.soil.b_infilt = torch.full_like(old[0], float(b[m]))
        c.soil.Dsmax = torch.full_like(old[1], float(dsm[m]))
        try:
            one = c.run()
        finally:
            c.soil.b_infilt, c.soil.Dsmax = old
        a = e.split_members(out['runoff'], n)[:, m]
        assert float((a - one['runoff']).abs().max()) < 1e-9


def test_member_params_rejects_unknown_names():
    c = _case(steps=8)
    e = c.ensemble(2)
    with pytest.raises(ValueError, match='unknown parameter'):
        e.member_params({'infilt': torch.tensor([0.1, 0.2])})
    with pytest.raises(ValueError, match='unknown parameter'):
        e.member_params({'rmin_scale': [1.0, 1.1], 'nonsense': [1, 2]})
