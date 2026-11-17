"""Ensembles: perturbation generators, members as extra cells."""
import os
import sys

import pytest
import torch

import _paths                                           # noqa: E402
from torchvic import case                               # noqa: E402
from torchvic.ensemble import (Ensemble, ForcingPerturbation,  # noqa: E402
                               latin_hypercube, sample_parameters)

WG = _paths.WG_FULL
CPU = torch.device('cpu')
CUDA = torch.cuda.is_available()


def _case(device=CPU, steps=24, first=0):
    if not os.path.exists(WG):
        pytest.skip('the Walnut Gulch global file is not present')
    return case.load(WG, device=device, steps=steps, first_step=first)


def test_latin_hypercube_has_one_point_per_stratum():
    u = latin_hypercube(16, 3, seed=4)
    assert u.shape == (16, 3) and float(u.min()) >= 0 and float(u.max()) < 1
    for j in range(3):
        assert sorted((u[:, j] * 16).floor().long().tolist()) == list(range(16))


def test_sampling_is_seeded_and_reproducible():
    r = {'a': (0.5, 2.0), 'b': (1.0, 100.0)}
    p1 = sample_parameters(r, 8, seed=3, log=('b',))
    p2 = sample_parameters(r, 8, seed=3, log=('b',))
    p3 = sample_parameters(r, 8, seed=4, log=('b',))
    assert torch.equal(p1['a'], p2['a']) and torch.equal(p1['b'], p2['b'])
    assert not torch.equal(p1['a'], p3['a'])
    assert float(p1['b'].min()) >= 1.0 and float(p1['b'].max()) <= 100.0
    c = sample_parameters(r, 8, seed=3, control={'a': 1.0, 'b': 10.0})
    assert float(c['a'][0]) == 1.0 and float(c['b'][0]) == 10.0


def test_forcing_perturbation_properties():
    c = _case()
    ens = c.ensemble(5)
    fp = ForcingPerturbation(prcp_sigma=0.5, tair_sigma=1.0, phi=0.8,
                             spatial='member', control=True, seed=2)
    a = fp.apply(ens.atm, 5, c.n_cells)
    b = ForcingPerturbation(prcp_sigma=0.5, tair_sigma=1.0, phi=0.8,
                            spatial='member', control=True,
                            seed=2).apply(ens.atm, 5, c.n_cells)
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k            # seeded
    assert a is not ens.atm and ens.atm['prcp'] is not a['prcp']
    nc = c.n_cells
    # control member untouched, others changed; precipitation stays >= 0
    assert torch.equal(a['prcp'][:, :nc], ens.atm['prcp'][:, :nc])
    assert torch.equal(a['tair'][:, :nc], ens.atm['tair'][:, :nc])
    assert not torch.equal(a['tair'][:, nc:2 * nc], ens.atm['tair'][:, nc:2 * nc])
    assert float(a['prcp'].min()) >= 0
    # the temperature shift is shared by the member's cells ('member')
    d = a['tair'] - ens.atm['tair']
    assert torch.allclose(d[:, nc:2 * nc], d[:, nc:nc + 1].expand(-1, nc))
    # tmax follows tair
    assert torch.allclose(a['tmax'] - ens.atm['tmax'], d)


def test_zero_spread_returns_the_inputs_untouched():
    c = _case()
    ens = c.ensemble(3)
    out = ForcingPerturbation().apply(ens.atm, 3, c.n_cells)
    for k in ens.atm:
        if torch.is_tensor(ens.atm[k]):
            assert out[k] is ens.atm[k]


def test_precipitation_factor_has_unit_mean():
    f, _ = ForcingPerturbation(prcp_sigma=0.4, seed=1).factors(
        2000, 50, 1)
    assert abs(float(f.mean()) - 1.0) < 0.02


def test_single_member_zero_perturbation_is_the_deterministic_run_bitwise():
    c = _case(steps=16)
    d = c.run()
    o = Ensemble(c, 1).run()
    for k in ('et', 'runoff', 'baseflow', 'swe'):
        assert torch.equal(o[k][0], d[k].T), k
    assert torch.equal(o['soil_moisture'][0], d['soil_moisture'])
    assert o['et'].shape == (1, c.n_cells, 16)


@pytest.mark.slow
def test_members_equal_the_deterministic_run_on_cpu_to_rounding():
    # more cells in one reduction reorder CPU sums (~1e-13); CUDA is bitwise
    # (test below)
    c = _case(steps=16)
    d = c.run()
    o = Ensemble(c, 3).run()
    for k in ('et', 'runoff', 'baseflow'):
        for m in range(3):
            assert float((o[k][m] - d[k].T).abs().max()) < 1e-9


def test_perturbed_members_differ_and_come_back_as_member_cell_time():
    c = _case(steps=24, first=196 * 8)
    fp = ForcingPerturbation(prcp_sigma=0.5, seed=3, control=True)
    e = Ensemble(c, 4, forcing=fp, params={'rmin_scale': [1.0, 0.8, 1.2, 1.5]})
    o = e.run()
    assert o['et'].shape == (4, c.n_cells, 24)
    assert o['discharge_mm'].shape[:1] == (4,)
    assert o['soil_moisture'].shape == (4, c.n_cells, 3)
    d = c.run()
    assert float((o['et'][0] - d['et'].T).abs().max()) < 1e-9
    assert float((o['et'][1] - o['et'][2]).abs().max()) > 0


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
def test_gpu_members_equal_the_deterministic_run_bitwise():
    dev = torch.device('cuda')
    c = _case(device=dev, steps=24)
    d = c.run()
    o = Ensemble(c, 6).run()
    for k in ('et', 'runoff', 'baseflow', 'swe'):
        for m in range(6):
            assert torch.equal(o[k][m], d[k].T), (k, m)
    for m in range(6):
        assert torch.equal(o['soil_moisture'][m], d['soil_moisture'])


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
def test_gpu_windowed_run_carries_state_in_memory():
    dev = torch.device('cuda')
    c = _case(device=dev, steps=32)
    e = Ensemble(c, 3, forcing=ForcingPerturbation(prcp_sigma=0.3, seed=1))
    whole = e.run()
    a = e.run(0, 16, return_state=True)
    b = e.run(16, 32, state=a['state'])
    assert torch.equal(torch.cat([a['et'], b['et']], 2), whole['et'])
