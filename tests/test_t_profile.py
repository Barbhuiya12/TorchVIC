"""torchvic.core.frozen.solve_T_profile against VIC 5.0.1's own.

frozen_soil.c:125-246 sets the coefficients of the soil heat equation and
:346-560 sweeps the profile until it settles. `ref_snow tprof` calls
**VIC's own solve_T_profile**, not a transcription of it -- which matters
here more than usual: the coefficient block in `tests/oracle/ref_soil.c`
IS a transcription, and it was missing the no-flux bottom node of the
EXP_TRANS branch (frozen_soil.c:218-226). The Python it was compared
against was missing it too, so the pair agreed on zero where VIC has a
coefficient, and the test passed. A hand transcription standing in for the
function it is meant to check can only find the mistakes it did not make.
"""
import os
import subprocess

import pytest
import torch

import _paths                                           # noqa: E402
REF = _paths.oracle('ref_snow')
DT = torch.float64
#: The oracle comparison asserts exact agreement on the closed-form nodes,
#: which is a CPU statement -- CUDA's `pow` differs from glibc's by a ulp
#: and the frozen nodes' unfrozen-water curve is a `pow`. See
#: docs/differences.md, and tests/test_snow_formulas.py, which pins its own
#: comparisons the same way.
DEV = torch.device('cpu')
MAX_NODES = 50          # vic_def.h:64


@pytest.fixture(scope='module')
def tprof():
    if not os.path.exists(REF):
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    from torchvic.core import frozen
    out = subprocess.run([REF, 'tprof'], capture_output=True, text=True,
                         check=True).stdout
    rows = [l.split('|') for l in out.splitlines() if not l.startswith('#')]
    cases = []
    for ra, rb in rows:
        a = [float(x) for x in ra.strip().split(',')]
        b = [float(x) for x in rb.strip().split(',')]
        n = int(a[1])
        blk = lambda i: torch.tensor(a[7 + i * MAX_NODES:
                                       7 + (i + 1) * MAX_NODES],
                                     dtype=DT, device=DEV).unsqueeze(0)
        (t0, zsum, kappa, cs, moist, max_moist, bubble, expt, ice,
         alpha, beta, gamma) = [blk(i) for i in range(12)]
        got, fb, ok = frozen.solve_T_profile(
            t0[:, :n], zsum[:, :n], kappa[:, :n], cs[:, :n], moist[:, :n],
            max_moist[:, :n], bubble[:, :n], expt[:, :n], ice[:, :n],
            alpha[:, :n - 1], beta[:, :n - 1], gamma[:, :n - 1],
            a[5], a[6], n, fs_active=bool(int(a[4])),
            noflux=bool(int(a[2])), exp_trans=bool(int(a[3])))
        ref = torch.tensor(b[1:1 + MAX_NODES], dtype=DT, device=DEV)[:n]
        ref_fb = torch.tensor(b[1 + MAX_NODES:1 + 2 * MAX_NODES], dtype=DT,
                              device=DEV)[:n]
        cases.append(dict(case=int(a[0]), n=n, noflux=bool(int(a[2])),
                          exp_trans=bool(int(a[3])),
                          fs_active=bool(int(a[4])), status=b[0],
                          t0=t0[0, :n], got=got[0], ref=ref,
                          fb=fb[0], ref_fb=ref_fb, converged=bool(ok[0])))
    return cases


def test_the_sample_reaches_every_branch(tprof):
    """Four switches decide which equation each node takes -- the grid
    transform, the no-flux bottom, whether frozen soil is active, and
    whether the node is below zero -- and a comparison is only worth what
    it reached."""
    seen = set()
    frozen_nodes = 0
    for c in tprof:
        seen.add((c['noflux'], c['exp_trans'], c['fs_active']))
        frozen_nodes += int((c['t0'] < 0.0).sum())
    assert len(seen) == 8, sorted(seen)
    assert frozen_nodes > 40
    assert len({c['n'] for c in tprof}) >= 5
    assert all(c['status'] == 0.0 for c in tprof)


def test_the_profile_matches_vic(tprof):
    """Every node of every case, bit for bit. The unfrozen nodes are a
    closed form, and the frozen ones take VIC's own root_brent sequence
    (implicit.root_brent) after VIC's own widening, so the whole sweep
    -- coefficients, roots, cold-nose fix -- rounds as VIC's does."""
    worst = max(float((c['got'] - c['ref']).abs().max()) for c in tprof)
    assert worst == 0.0, worst
    # The sampling still covers every branch: all four grid-and-boundary
    # combinations in closed form, and frozen nodes besides.
    closed = [c for c in tprof
              if not c['fs_active'] or bool((c['t0'] >= 0.0).all())]
    assert len(closed) >= 30
    assert len({(c['noflux'], c['exp_trans']) for c in closed}) == 4
    rest = [c for c in tprof if c not in closed]
    assert len(rest) >= 8


def test_the_no_flux_bottom_node_has_coefficients_under_exp_trans(tprof):
    """frozen_soil.c fills the no-flux bottom node's coefficients in BOTH
    grid branches (:199-207 and :218-226). This port filled it only in the
    uniform branch, and so did the hand transcription in
    `tests/oracle/ref_soil.c` that checked it -- the pair agreed on zero
    where VIC has a number, and the test passed.

    Measured on the twelve EXP_TRANS-with-NOFLUX cases of this oracle,
    which calls VIC's own solve_T_profile: with the block, bit-exact;
    without it, 6.74 C, because the bottom node is left at the temperature
    it started the step with while VIC moves it.
    """
    from torchvic.core import frozen
    both = [c for c in tprof if c['noflux'] and c['exp_trans']]
    assert len(both) >= 8
    n = both[0]['n']
    one = lambda v, m=None: torch.full((1, m or n), float(v), dtype=DT,
                                       device=DEV)
    a, b, c, d, e = frozen.thermal_coefficients(
        one(2.0e6), one(1.3), one(0.5), one(0.3, n - 1), one(0.2, n - 1),
        one(0.2, n - 1), 3600.0, 4.0, n, exp_trans=True, noflux=True)
    for name, arr in (('A', a), ('C', c), ('E', e)):
        assert float(arr[0, -1]) != 0.0, name
    # ...and the surface node stays zero, because it is a boundary
    # condition rather than a solved node.
    assert float(a[0, 0]) == 0.0


def test_the_fallback_flags_match_vic(tprof):
    """`Tfbflag` marks a node the solver could not settle: a failed root
    find, the cold-nose correction, or a whole profile that never
    converged. They are integers, so a disagreement is a disagreement."""
    for c in tprof:
        assert bool((c['fb'].to(DT) == c['ref_fb']).all()), c['case']


def test_the_profile_carries_a_gradient(tprof):
    """Differentiated through the sweep and through the frozen nodes' root
    find. The sweep is a fixed number of Gauss-Seidel passes, so the
    gradient is an unrolled one; the root find inside it is implicit."""
    from torchvic.core import frozen
    c = max(tprof, key=lambda x: int((x['t0'] < 0.0).sum()))
    n = c['n']
    one = lambda v: torch.full((1, n), float(v), dtype=DT, device=DEV)
    t0 = c['t0'].unsqueeze(0).clone().requires_grad_(True)
    zsum = torch.linspace(0.0, 4.0, n, dtype=DT, device=DEV).unsqueeze(0)
    got, _fb, _ok = frozen.solve_T_profile(
        t0, zsum, one(1.3), one(2.0e6), one(150.0), one(300.0), one(25.0),
        one(10.0), one(20.0), one(0.6)[:, :n - 1], one(0.3)[:, :n - 1],
        one(0.3)[:, :n - 1], 3600.0, 4.0, n)
    got.sum().backward()
    assert torch.isfinite(t0.grad).all()
    assert float(t0.grad.abs().sum()) > 0.0
