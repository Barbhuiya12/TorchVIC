#!/usr/bin/env python3
"""Check torchvic.core.snow.snow_step as a whole: conservation, behaviour, gradients.

The individual functions are checked against VIC's own code by
tests/test_vic_snow.py. This file checks the assembled step, where
the obligations are different: mass must be conserved across the chain, the
pack must actually melt under forcing that should melt it, and gradients must
survive the root find in the middle.

Run: tests/test_snow_step.py
"""
import os
import sys


import torch                                     # noqa: E402
from torchvic.core import snow  # noqa: E402

# This project is GPU-native and the differentiable half is no exception.
# Override with TORCHVIC_DEVICE=cpu.
DEVICE = os.environ.get('TORCHVIC_DEVICE') or (
    'cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)



def const(n, v):
    return torch.full((n,), v, dtype=torch.float64)


def base_forcing(n, air_temp, net_short=40.0, long_in=250.0):
    return dict(t_grnd=const(n, 0.0), wind=const(n, 2.0),
                net_short=const(n, net_short),
                long_snow_in=const(n, long_in), air_dens=const(n, 1.2),
                eact_air=const(n, 400.0), lv=const(n, 2.5e6),
                press=const(n, 101300.0), vpd=const(n, 200.0),
                rain=const(n, 0.0), dt=const(n, 3600.0), z=const(n, 2.0),
                z0=const(n, 0.01), ra=const(n, 60.0),
                snow_depth=const(n, 0.4), snow_density=const(n, 250.0),
                air_temp=const(n, air_temp), snowfall=const(n, 0.0))


def run(state, steps, air_temp, snowfall=0.0, **kw):
    n = state['ice'].shape[0]
    total_melt = torch.zeros(n, dtype=torch.float64)
    total_snow = torch.zeros(n, dtype=torch.float64)
    total_sub = torch.zeros(n, dtype=torch.float64)
    for _ in range(steps):
        f = base_forcing(n, air_temp, **kw)
        f['snowfall'] = const(n, snowfall)
        state, melt = snow.snow_step(state, f)
        total_melt = total_melt + melt
        total_snow = total_snow + f['snowfall']
        # The pack also loses mass to the air. snow_step returns it as
        # `sublimation` (snow.py, out['sublimation'] = -vapor_flux, the
        # sign put_data.c:627 reports), and a balance that counts only
        # snowfall and melt is short by exactly that -- which is what this
        # script reported as a mass conservation failure for as long as
        # nothing ran it.
        total_sub = total_sub + state['sublimation']
    return state, total_melt, total_snow, total_sub


def main():
    failures = []
    n = 3
    print('torchvic.core.snow.snow_step')

    # 1. Mass conservation. Water in minus water out must equal the change in
    #    storage, to rounding. This is the property the chain can most easily
    #    break, because five functions hand state to each other.
    st0 = dict(ice=const(n, 0.10), surf_water=const(n, 0.0),
               pack_water=const(n, 0.0), surf_temp=const(n, -5.0),
               pack_temp=const(n, -3.0))
    store0 = st0['ice'] + st0['surf_water'] + st0['pack_water']
    st, melt, snowfall, sub = run(dict(st0), 48, -6.0, snowfall=0.001)
    store1 = st['ice'] + st['surf_water'] + st['pack_water']
    resid = (store1 - store0) - (snowfall - melt - sub)
    print('  cold run, 48 steps   snow %.6f melt %.6f sublim %.6f'
          % (snowfall.max(), melt.max(), sub.max()))
    print('                       mass residual  %.3g' % resid.abs().max())
    if not bool(resid.abs().max() < 1e-12):
        failures.append('mass conservation, cold')

    # 2. The pack must melt under forcing that should melt it, and must not
    #    go negative doing so.
    st, melt, snowfall, sub = run(dict(st0), 200, 8.0, net_short=250.0,
                                  long_in=330.0)
    store1 = st['ice'] + st['surf_water'] + st['pack_water']
    resid = (store1 - store0) - (snowfall - melt - sub)
    print('  warm run, 200 steps  total melt %.6f  final swq %.6f'
          % (melt.max(), st['swq'].max()))
    print('                       mass residual  %.3g' % resid.abs().max())
    if not bool(melt.max() > 0.0):
        failures.append('no melt under strong forcing')
    if not bool((st['ice'] >= -1e-15).all()):
        failures.append('negative ice')
    if not bool(resid.abs().max() < 1e-12):
        failures.append('mass conservation, warm')

    # 3. Gradients must survive the root find. This is what the whole
    #    implicit-differentiation machinery exists for: a parameter a
    #    calibration would touch, differentiated through a multi-step
    #    simulation containing a solve.
    lw = const(n, 250.0).clone().requires_grad_(True)
    state = dict(st0)
    total = torch.zeros(n, dtype=torch.float64)
    for _ in range(12):
        f = base_forcing(n, 2.0, net_short=200.0)
        f['long_snow_in'] = lw
        state, m = snow.snow_step(state, f)
        total = total + m
    total.sum().backward()
    ok = bool(torch.isfinite(lw.grad).all())
    print('  d(total melt)/d(longwave in)  %s  %s'
          % ([round(v, 8) for v in lw.grad.tolist()],
             'finite' if ok else 'NOT FINITE'))
    if not ok:
        failures.append('gradient through snow_step')
    if not bool((lw.grad > 0).all()):
        failures.append('melt should increase with incoming longwave')

    print()
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
