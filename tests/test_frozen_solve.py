#!/usr/bin/env python3
"""Check the differentiable frozen-node solve.

soil_thermal_eqn is bit-exact against VIC (tests/test_vic_frozen.py),
so a root of it is a root of VIC's own residual. This checks that the solve
finds one and that gradients survive it, the same way the snow surface
temperature was checked.

Run: tests/test_frozen_solve.py
"""
import os
import sys


import torch                                    # noqa: E402
from torchvic.core import frozen, implicit  # noqa: E402

# This project is GPU-native and the differentiable half is no exception.
# Override with TORCHVIC_DEVICE=cpu.
DEVICE = os.environ.get('TORCHVIC_DEVICE') or (
    'cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)



def main():
    failures = []
    # VIC's own half-width, param.SOIL_DT = 0.25 (initialize_parameters.c
    # :203). The wider bracket this used to open with is not merely slower:
    # `soil_thermal_eqn` JUMPS at T = 0, where the ice term switches on --
    # measured at 6e+05 to 7e+05 in the residual across the crossing, since
    # E is 2.7e7 and the ice it multiplies goes from ice0 to zero. A bracket
    # that straddles zero therefore brackets the jump, not a root: a
    # bisection converges to the discontinuity and the residual there is not
    # small. VIC has the same property and root_brent the same behaviour;
    # what keeps it out of VIC's way is that the frozen branch is only
    # entered when T < 0 and the bracket is only half a degree wide.
    n = 6
    c = lambda v: torch.full((n,), v, dtype=torch.float64)
    t0 = torch.tensor([-8.0, -5.0, -3.0, -2.0, -1.0, -0.5],
                      dtype=torch.float64)
    soil_dt = 0.25

    # The coefficients come from `thermal_coefficients` on a real node grid
    # and the starting ice from the unfrozen-water curve at T0, rather than
    # from round numbers. That is not tidiness: with invented coefficients
    # the residual has no root within VIC's own bracket at all -- it runs
    # from +5e6 to +4e6 across [T0 - 3, T0 + 3] and never crosses -- and the
    # only way to make the check "pass" was to open a bracket twelve times
    # wider than VIC's, which then straddles T = 0, where the residual JUMPS
    # by 6e+05 as the ice term switches on. A bisection converges to that
    # jump and the residual there is not small. The bracket is now VIC's
    # own, param.SOIL_DT = 0.25.
    nn, dp, dt = 5, 4.0, 3600.0
    grid = lambda v, m=nn: torch.full((1, m), float(v), dtype=torch.float64)
    zsum = torch.linspace(0.0, dp, nn,
                          dtype=torch.float64).unsqueeze(0)
    A, B, C, D, E = frozen.thermal_coefficients(
        grid(2.0e6), grid(1.3), zsum, grid(0.6, nn - 1), grid(0.3, nn - 1),
        grid(0.3, nn - 1), dt, dp, nn, exp_trans=True, noflux=False)
    j = 1
    moist, max_moist, bubble, expt = c(0.18), c(0.40), c(14.66), c(12.676)
    ice0 = torch.clamp(moist - frozen.maximum_unfrozen_water(
        t0, max_moist, bubble, expt), min=0.0)
    st = dict(tl=t0 - 0.4, tu=t0 + 0.4, t0=t0, moist=moist,
              max_moist=max_moist, bubble=bubble, expt=expt, ice0=ice0,
              a=c(float(A[0, j])), b=c(float(B[0, j])), c=c(float(C[0, j])),
              d=c(float(D[0, j])), e=c(float(E[0, j])), node=c(3.0),
              exp_trans=True)

    print('frozen-node solve')
    t, bracketed = frozen.solve_frozen_node(st, t0, soil_dt=soil_dt)
    print('  bracketed  %d of %d' % (int(bracketed.sum()), n))
    resid = frozen.soil_thermal_eqn(t, st).abs()
    print('  max |residual| at the root   %.3g' % resid[bracketed].max())
    # Not every node brackets in [T0 +/- SOIL_DT]. VIC's root_brent widens
    # the interval when that happens -- which `calc_soil_thermal_fluxes`
    # reproduces with implicit.widen_bracket -- while this entry point tells
    # the caller instead, so a non-root is never differentiated. The count is
    # reported rather than asserted, and everything below uses only the
    # bracketed cases.
    if not bool(bracketed.any()):
        failures.append('nothing bracketed')
    if not bool(resid[bracketed].max() < 1e-6):
        failures.append('residual at root')

    # Gradient w.r.t. a calibratable soil parameter, against a central
    # difference of the root itself.
    print('\ngradient of the root')
    k = bracketed.nonzero().flatten()
    sub = {kk: (v[k] if torch.is_tensor(v) else v) for kk, v in st.items()}
    t0k = t0[k]
    mm = sub['max_moist'].clone().requires_grad_(True)
    stg = dict(sub)
    stg['max_moist'] = mm
    tg = implicit.solve(frozen.soil_thermal_eqn, t0k - soil_dt,
                        t0k + soil_dt, args=(stg,))
    tg.sum().backward()

    eps = 1e-6
    up, dn = dict(sub), dict(sub)
    up['max_moist'] = sub['max_moist'] + eps
    dn['max_moist'] = sub['max_moist'] - eps
    fd = ((implicit.bisect(frozen.soil_thermal_eqn, t0k - soil_dt,
                           t0k + soil_dt, args=(up,)) -
           implicit.bisect(frozen.soil_thermal_eqn, t0k - soil_dt,
                           t0k + soil_dt, args=(dn,))) / (2 * eps))
    rel = ((mm.grad - fd).abs() / fd.abs().clamp(min=1e-30)).max()
    print('  dT/d(max_moist) vs central difference  max|rel| %.3g' % rel)
    print('  finite: %s' % bool(torch.isfinite(mm.grad).all()))
    if not bool(torch.isfinite(mm.grad).all()):
        failures.append('gradient finite')
    if not bool(rel < 1e-4):
        failures.append('gradient vs finite difference')

    print()
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
