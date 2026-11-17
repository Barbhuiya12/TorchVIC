"""QUICK_FLUX FALSE: the finite-difference soil temperature profile.

With QUICK_FLUX FALSE VIC solves the soil temperature profile by finite
differences (Cherkauer and Lettenmaier, 1999) instead of the Liang et al.
(1999) two-node flux: func_surf_energy_bal.c:404-445 solves the profile at
every trial surface temperature (solve_T_profile_implicit with IMPLICIT,
solve_T_profile without), and calc_surf_energy_bal.c carries the solved
nodes into the layer temperatures. torchvic.core.soiltemp is that path.

The row test below runs himal_EB_510_3h_QUICK_FLUX=FALSE (FROZEN_SOIL
FALSE, so IMPLICIT and EXP_TRANS are FALSE too, NODES 3) for three days
through pristine VIC 5.1.0 and through TorchVIC and grades the pair with
verification/compare.py, the same comparison the scoreboard uses.
"""
import os
import sys

import pytest

import _paths                                           # noqa: E402

sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
matrix = pytest.importorskip('matrix')
run_matrix = pytest.importorskip('run_matrix')
compare = pytest.importorskip('compare')

RID = 'himal_EB_510_3h_QUICK_FLUX=FALSE'
NEED = (os.path.join(matrix.WORK, 'windows', 'himal', 'params.nc'),
        matrix.VIC['5.1.0'])

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in NEED),
    reason='the himal window or the pristine VIC 5.1.0 image driver is '
           'not present')


def test_quick_flux_false_matches_vic(tmp_path, monkeypatch):
    """Three days of the QUICK_FLUX=FALSE row, VIC against TorchVIC."""
    r = next(x for x in matrix.MATRIX if x['id'] == RID)
    d = run_matrix.prepare(r, days=3, root=str(tmp_path))
    rc, _ = run_matrix.run_vic(r, d)
    assert rc == 0, open(os.path.join(d, 'vic.log')).read()[-2000:]
    # run_tv inherits this process's environment: point the subprocess at
    # this working tree, or it imports whatever torchvic is installed.
    monkeypatch.setenv('PYTHONPATH', _paths.REPO)
    rc, _ = run_matrix.run_tv(r, d, 'cpu', 4, 3)
    assert rc == 0, open(os.path.join(d, 'tv.log')).read()[-2000:]
    res = compare.compare_dirs(os.path.join(d, 'vic'), os.path.join(d, 'tv'),
                               r, {})
    assert res['grade'] == 'pass', (
        res['grade'], res.get('worst_water'), res.get('score_water'),
        res.get('worst_energy'), res.get('score_energy'),
        res['closure'].get('tv_water_resid_max'))


def test_node_grid_without_exp_trans():
    """vic_init.c:846-879: surface, first layer base, twice that, dp."""
    import torch
    from torchvic.core import soiltemp
    depth = torch.tensor([[0.25, 6.0, 6.0]], dtype=torch.float64)
    dp = torch.tensor([4.0], dtype=torch.float64)
    zsum, dz = soiltemp.node_grid(depth, dp, 3, False)
    assert zsum.tolist() == [[0.0, 0.25, 4.0]]
    assert dz.tolist() == [[0.25, 0.25, (4.0 - 0.5 - 0.125) * 2.0]]
    # The first layer's depth is cell 0's for every cell (:860-863).
    depth2 = torch.tensor([[0.25, 6.0, 6.0], [0.5, 6.0, 6.0]],
                          dtype=torch.float64)
    zsum2, _ = soiltemp.node_grid(depth2, dp.expand(2), 3, False)
    assert zsum2[1, 1].item() == 0.25


def test_node_grid_with_exp_trans_is_float():
    """vic_init.c:887-914: expf(Bexp j) - 1, in single precision."""
    import torch
    from torchvic.core import soiltemp
    depth = torch.tensor([[0.1, 0.9, 2.0]], dtype=torch.float64)
    dp = torch.tensor([4.0], dtype=torch.float64)
    zsum, _ = soiltemp.node_grid(depth, dp, 12, True)
    be = float(torch.log(torch.tensor(5.0, dtype=torch.float32))) / 11
    want = [float(torch.exp(torch.tensor(be * j, dtype=torch.float32)))
            - 1.0 for j in range(12)]
    assert zsum[0].tolist() == want
    # Too few nodes for the damping depth stops the run (:895-911).
    with pytest.raises(ValueError, match='too small for EXP_TRANS'):
        soiltemp.node_grid(depth, dp, 5, True)


def test_tridiag_matches_the_dense_solve():
    """newt_raph_func_fast.c:132-169 against torch.linalg.solve."""
    import torch
    from torchvic.core import soiltemp
    g = torch.Generator().manual_seed(7)
    for n in (1, 2, 5, 11):
        a = torch.randn(4, n, dtype=torch.float64, generator=g)
        b = torch.randn(4, n, dtype=torch.float64, generator=g) * 4
        c = torch.randn(4, n, dtype=torch.float64, generator=g)
        r = torch.randn(4, n, dtype=torch.float64, generator=g)
        got = soiltemp.tridiag(a, b, c, r, n)
        m = torch.diag_embed(b, 0) + torch.diag_embed(a[:, 1:], -1) \
            + torch.diag_embed(c[:, :-1], 1)
        want = torch.linalg.solve(m, r.unsqueeze(-1)).squeeze(-1)
        assert float((got - want).abs().max()) < 1e-12


def test_quick_solve_nodes():
    """calc_surf_energy_bal.c:257-279: down past the thaw front."""
    import torch
    from torchvic.core import soiltemp
    # A crossing between nodes 4 and 5 of 12: (4 + 1) + 4.
    t = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0,
                       -5.0, -6.0, -7.0]])
    assert soiltemp.quick_solve_nodes(t, 12).tolist() == [9]
    # No crossing, surface frozen and node 1 thawed: the whole column.
    t = torch.tensor([[-1.0, 2.0, 3.0]])
    assert soiltemp.quick_solve_nodes(t, 3).tolist() == [3]
    # No crossing and not that either: 3.
    t = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.5]])
    assert soiltemp.quick_solve_nodes(t, 12).tolist() == [3]


def test_gf406_on_the_profile_is_the_surface_term_alone():
    """func_surf_energy_bal.c:466-486: without EXP_TRANS, GF_406 drops
    the deep term and the halving; GF_410 keeps both."""
    import torch
    from torchvic.core import soiltemp
    ts = torch.tensor([1.5, -2.0], dtype=torch.float64)
    t_new = torch.tensor([[1.5, 0.7, 0.2], [-2.0, -0.5, 0.1]],
                         dtype=torch.float64)
    one = torch.ones(2, dtype=torch.float64)
    zsum = torch.tensor([[0.0, 0.1, 2.0]] * 2, dtype=torch.float64)
    common = (ts, t_new, t_new, 0.5 * one, ts, 0 * one, 0.3 * one,
              0.45 * one, 20.0 * one, 10.0 * one, 1.2 * one, 0.9 * one,
              2.0e6 * one, 0.1 * one, 10800.0, zsum, 0.8 * one, False,
              False, False)
    _, g410, _, _ = soiltemp.profile_ground_terms(*common)
    _, g406, _, _ = soiltemp.profile_ground_terms(
        *common, grnd_flux_type='GF_406')
    k1, k2, d1 = 1.2, 0.9, 0.1
    t1, t11 = t_new[:, 1], t_new[:, 2]
    assert torch.equal(g406, 0.8 * one * (k1 * one / d1 * (t1 - ts)))
    assert torch.equal(g410, 0.8 * one * (k1 * one / d1 * (t1 - ts)
                                          + k2 * one / d1 * (t11 - t1)) / 2.0)
