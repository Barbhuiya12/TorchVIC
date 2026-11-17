"""torchvic.core.soiltemp.solve_profile_implicit against VIC's own.

frozen_soil.c:235-322 drives newt_raph over fda_heat_eqn at :624-1010.
`ref_snow tprof_implicit` calls **VIC's own solve_T_profile_implicit**,
not a transcription of it. The soil arithmetic is identical in VIC 5.0.1
and 5.1.0 (only the OpenMP threadprivate pragmas differ), so the 5.0.1
oracle checks the 5.1.0 core as well.

The oracle's Tfbflag column is not compared: in calc_surf_energy_bal.c
the flags are scratch the iteration writes and never reads back, and
bit-exact T already proves the cold-nose fallback did the same thing.
"""
import os
import subprocess

import pytest
import torch

import _paths                                           # noqa: E402
REF = _paths.oracle('ref_snow')
DT = torch.float64
#: As in test_t_profile.py: exact agreement is a CPU statement, since
#: CUDA's `pow` differs from glibc's by a ulp on the unfrozen-water curve.
DEV = torch.device('cpu')
MAX_NODES = 50          # vic_def.h:64
NLAYER = 3


@pytest.fixture(scope='module')
def tprofi():
    if not os.path.exists(REF):
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    from torchvic.core import soiltemp
    # One process per case: VIC keeps the trial conductivities in
    # statics that mixed node counts would poison (see the mode's
    # comment); every real run holds its count fixed.
    rows = []
    for c in range(48):
        out = subprocess.run([REF, 'tprof_implicit', str(c)],
                             capture_output=True, text=True,
                             check=True).stdout
        rows.extend(l.split('|') for l in out.splitlines()
                    if not l.startswith('#'))
    cases = []
    for ra, rb in rows:
        a = [float(x) for x in ra.strip().split(',')]
        b = [float(x) for x in rb.strip().split(',')]
        n = int(a[1])
        blk = lambda i: torch.tensor(a[6 + i * MAX_NODES:
                                       6 + (i + 1) * MAX_NODES],
                                     dtype=DT, device=DEV).unsqueeze(0)
        (t0, zsum, kappa, cs, moist, max_moist, bubble, expt, ice,
         alpha, beta, gamma) = [blk(i) for i in range(12)]
        lay = lambda i: torch.tensor(
            a[6 + 12 * MAX_NODES + i * NLAYER:
              6 + 12 * MAX_NODES + (i + 1) * NLAYER],
            dtype=DT, device=DEV).unsqueeze(0)
        depth, bdmin, sdmin, qz, bd, sd, org = [lay(i) for i in range(7)]
        noflux, exp_trans = bool(int(a[2])), bool(int(a[3]))
        got, error = soiltemp.solve_profile_implicit(
            t0[:, :1], t0[:, :n], moist[:, :n], ice[:, :n], kappa[:, :n],
            cs[:, :n], max_moist[:, :n], bubble[:, :n], expt[:, :n],
            alpha[:, :n - 1], beta[:, :n - 1], gamma[:, :n - 1],
            zsum[:, :n], a[4], torch.tensor(a[5], dtype=DT, device=DEV),
            bdmin, sdmin, qz, bd, sd, org, depth, NLAYER, n, noflux,
            exp_trans)
        ref = torch.tensor(b[1:1 + MAX_NODES], dtype=DT, device=DEV)[:n]
        cases.append(dict(case=int(a[0]), n=n, noflux=noflux,
                          exp_trans=exp_trans, status=b[0],
                          t0=t0[0, :n], got=got[0], ref=ref,
                          converged=not bool(error[0])))
    return cases


def test_the_sample_reaches_every_branch(tprofi):
    """Both grid branches, with and without the no-flux bottom, over
    node counts 3 through 8 -- and cases on each side of convergence,
    since a Newton transcription must fail exactly where VIC's fails."""
    assert len(tprofi) == 48
    assert len({c['n'] for c in tprofi}) == 6
    conv = [c for c in tprofi if c['status'] == 0.0]
    fail = [c for c in tprofi if c['status'] != 0.0]
    assert len(conv) >= 20, len(conv)
    assert len(fail) >= 8, len(fail)
    assert len({(c['noflux'], c['exp_trans']) for c in conv}) == 4
    assert len({(c['noflux'], c['exp_trans']) for c in fail}) >= 2


def test_convergence_matches_vic(tprofi):
    """Newton fails on harsh profiles -- 24 of these 48 -- and the port
    must fail on exactly the same ones, since the caller falls back to
    the explicit sweep on failure."""
    for c in tprofi:
        assert c['converged'] == (c['status'] == 0.0), c['case']


def test_the_profile_matches_vic(tprofi):
    """Every node of every converged case, bit for bit: the residual,
    the tridiagonal Jacobian steps, the line search, the boundaries,
    and the cold-nose fix."""
    conv = [c for c in tprofi if c['status'] == 0.0]
    worst = max(float((c['got'] - c['ref']).abs().max()) for c in conv)
    assert worst == 0.0, worst


def test_the_three_node_grid_without_noflux_is_refused():
    """With a single unknown VIC reads T_2[1] past it
    (frozen_soil.c:793-796) -- uninitialized stack, which -O2 and -O3
    resolve differently (16.99 C on OUT_SURF_TEMP over 14 days). The
    port refuses the grid instead of reproducing the read."""
    from torchvic.core import soiltemp
    kw = dict(ts=torch.zeros(1, 1, dtype=DT, device=DEV),
              t0=torch.zeros(1, 1, 3, dtype=DT, device=DEV),
              moist=torch.full((1, 1, 3), 0.2, dtype=DT, device=DEV),
              ice=torch.zeros(1, 1, 3, dtype=DT, device=DEV),
              kappa=torch.ones(1, 1, 3, dtype=DT, device=DEV),
              cs=torch.full((1, 1, 3), 2.0e6, dtype=DT, device=DEV),
              porosity_node=torch.full((1, 1, 3), 0.4, dtype=DT,
                                       device=DEV),
              bubble_node=torch.full((1, 1, 3), 20.0, dtype=DT,
                                     device=DEV),
              expt_node=torch.full((1, 1, 3), 10.0, dtype=DT, device=DEV),
              alpha=torch.ones(1, 1, 2, dtype=DT, device=DEV),
              beta=torch.ones(1, 1, 2, dtype=DT, device=DEV),
              gamma=torch.ones(1, 1, 2, dtype=DT, device=DEV),
              zsum=torch.tensor([[[0.0, 2.0, 4.0]]], dtype=DT,
                                device=DEV),
              deltat=10800.0, dp=torch.tensor(4.0, dtype=DT, device=DEV),
              bulk_dens_min=torch.full((1, 1, 3), 1300.0, dtype=DT,
                                       device=DEV),
              soil_dens_min=torch.full((1, 1, 3), 2650.0, dtype=DT,
                                       device=DEV),
              quartz=torch.full((1, 1, 3), 0.5, dtype=DT, device=DEV),
              bulk_density=torch.full((1, 1, 3), 1300.0, dtype=DT,
                                      device=DEV),
              soil_density=torch.full((1, 1, 3), 2650.0, dtype=DT,
                                      device=DEV),
              organic=torch.zeros(1, 1, 3, dtype=DT, device=DEV),
              depth=torch.tensor([[[0.5, 1.5, 4.0]]], dtype=DT,
                                 device=DEV),
              nlayers=3, nnodes=3, noflux=False, exp_trans=False)
    with pytest.raises(ValueError, match='uninitialized'):
        soiltemp.solve_profile_implicit(**kw)
    # ...while the same grid with the no-flux bottom solves (n = 2).
    got, error = soiltemp.solve_profile_implicit(
        **dict(kw, noflux=True))
    assert got.shape[-1] == 3
    assert error.shape == (1, 1)
