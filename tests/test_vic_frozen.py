#!/usr/bin/env python3
"""Check torchvic.core.frozen against VIC's own soil_conduction.c.

The reference is produced by `ref_soil frozen`, which calls VIC's
soil_conductivity(), volumetric_heat_capacity() and
maximum_unfrozen_water() rather than transcribing them.

Run: tests/test_vic_frozen.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths                                   # noqa: E402

import torch                                      # noqa: E402
from torchvic.core import frozen  # noqa: E402

# This project is GPU-native and the differentiable half is no exception.
# Override with TORCHVIC_DEVICE=cpu.
DEVICE = os.environ.get('TORCHVIC_DEVICE') or (
    'cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)


REF = _paths.oracle('ref_soil')
RTOL = 1e-12
ATOL = 1e-13

SOIL_DENS_MIN = 2650.0
BULK_DENS_MIN = 1569.1
SOIL_DENSITY = 2619.1
BULK_DENSITY = 1569.1


def load():
    out = subprocess.run([REF, 'frozen'], capture_output=True, text=True,
                         check=True).stdout
    rows = {}
    for line in out.splitlines():
        if line.startswith('#') or line.startswith('fn,'):
            continue
        if line.startswith('soil_thermal_eqn,'):
            ins, outs = line.split(',', 1)[1].split('|', 1)
            rows.setdefault('soil_thermal_eqn', []).append(
                [float(x) for x in ins.split(',')] + [float(outs)])
            continue
        if line.startswith('node_update,'):
            ins, outs = line.split(',', 1)[1].split('|', 1)
            rows.setdefault('node_update', []).append(
                [float(x) for x in ins.split(',')] + [float(outs)])
            continue
        if line.startswith('thermal_coeff,'):
            ins, outs = line.split(',', 1)[1].split('|', 1)
            rows.setdefault('thermal_coeff', []).append(
                [float(x) for x in ins.split(',')] +
                [float(x) for x in outs.split('|')])
            continue
        p = line.split(',')
        rows.setdefault(p[0], []).append([float(x) for x in p[1:]])
    return rows


def check(name, got, want, failures):
    got = got.to(torch.float64)
    want = torch.tensor(want, dtype=torch.float64)
    ok = torch.allclose(got, want, rtol=RTOL, atol=ATOL)
    err = (got - want).abs()
    rel = err / want.abs().clamp(min=1e-30)
    print('  %-30s %4d values  max|abs| %9.3g  max|rel| %9.3g  %s'
          % (name, len(want), err.max().item(), rel.max().item(),
             'PASS' if ok else 'FAIL'))
    if not ok:
        bad = (~torch.isclose(got, want, rtol=RTOL, atol=ATOL)).nonzero()
        for i in bad[:4].flatten().tolist():
            print('      row %d: got %.17g want %.17g' % (i, got[i], want[i]))
        failures.append(name)


def main():
    rows = load()
    failures = []
    print('torchvic.core.frozen against VIC soil_conduction.c')

    d = rows['maximum_unfrozen_water']
    col = lambda i: torch.tensor([r[i] for r in d], dtype=torch.float64)
    check('maximum_unfrozen_water',
          frozen.maximum_unfrozen_water(col(0), col(1), col(2), col(3)),
          [r[8] for r in d], failures)

    d = rows['volumetric_heat_capacity']
    col = lambda i: torch.tensor([r[i] for r in d], dtype=torch.float64)
    check('volumetric_heat_capacity',
          frozen.volumetric_heat_capacity(col(0), col(1), col(2), col(3)),
          [r[8] for r in d], failures)

    for tag in ('soil_conductivity', 'soil_conductivity_frozen'):
        d = rows[tag]
        col = lambda i: torch.tensor([r[i] for r in d], dtype=torch.float64)
        n = len(d)
        c = lambda v: torch.full((n,), v, dtype=torch.float64)
        check(tag,
              frozen.soil_conductivity(col(0), col(1), c(SOIL_DENS_MIN),
                                       c(BULK_DENS_MIN), col(2),
                                       c(SOIL_DENSITY), c(BULK_DENSITY),
                                       col(3)),
              [r[8] for r in d], failures)

    # Heat-equation coefficients. Rows are (exp_trans, noflux, node) with
    # the five coefficients appended; regroup into [config][node].
    N = 5
    tc = rows['thermal_coeff']
    cs = torch.tensor([[1.8e6 + 1.0e5 * j for j in range(N)]],
                      dtype=torch.float64)
    kap = torch.tensor([[1.2 + 0.1 * j for j in range(N)]],
                       dtype=torch.float64)
    zs = torch.tensor([[0.05 * j * j + 0.1 * j for j in range(N)]],
                      dtype=torch.float64)
    al = torch.tensor([[0.3 + 0.05 * j for j in range(N - 1)]],
                      dtype=torch.float64)
    be = torch.tensor([[0.15 + 0.02 * j for j in range(N - 1)]],
                      dtype=torch.float64)
    ga = torch.tensor([[0.12 + 0.03 * j for j in range(N - 1)]],
                      dtype=torch.float64)
    for et in (0, 1):
        for nf in (0, 1):
            want = [r for r in tc if r[0] == et and r[1] == nf]
            want.sort(key=lambda r: r[2])
            got = frozen.thermal_coefficients(cs, kap, zs, al, be, ga,
                                              3600.0, 4.0, N,
                                              exp_trans=bool(et),
                                              noflux=bool(nf))
            flat = torch.cat([g[0] for g in got])
            ref = [w[3 + k] for k in range(5) for w in want]  # A..E follow et,nf,j
            check('thermal_coeff exp_trans=%d noflux=%d' % (et, nf),
                  flat, ref, failures)

    nu = rows['node_update']
    for et in (0, 1):
        for nfn in (0, 1):
            w = [r for r in nu if r[0] == et and r[1] == nfn]
            col = lambda i: torch.tensor([r[i] for r in w],
                                         dtype=torch.float64)
            n = len(w)
            c = lambda v: torch.full((n,), v, dtype=torch.float64)
            got = frozen.node_update_unfrozen(
                col(2), col(3), col(4), col(5), col(6),
                c(1.7e5), c(7.2e2), c(2.3e4), c(1.9e4), c(2.7e7),
                exp_trans=bool(et), noflux_node=bool(nfn))
            check('node_update exp_trans=%d noflux=%d' % (et, nfn), got,
                  [r[7] for r in w], failures)

    ste = rows['soil_thermal_eqn']
    for et in (0, 1):
        w = [r for r in ste if r[0] == et]
        n = len(w)
        col = lambda i: torch.tensor([r[i] for r in w], dtype=torch.float64)
        c = lambda v: torch.full((n,), v, dtype=torch.float64)
        st = dict(tl=col(3), tu=col(4), t0=c(-1.5), moist=c(0.3),
                  max_moist=c(0.4), bubble=c(14.66), expt=c(12.676),
                  ice0=c(0.05), a=c(1.7e5), b=c(7.2e2), c=c(2.3e4),
                  d=c(1.9e4), e=c(2.7e7), node=col(1),
                  exp_trans=bool(et))
        check('soil_thermal_eqn exp_trans=%d' % et,
              frozen.soil_thermal_eqn(col(2), st), [r[5] for r in w],
              failures)

    print('\ngradient check')
    t = torch.tensor([-8.0, -2.0, -0.2], dtype=torch.float64,
                     requires_grad=True)
    mm = torch.full((3,), 0.4, dtype=torch.float64)
    frozen.maximum_unfrozen_water(t, mm, torch.full((3,), 14.66,
                                                    dtype=torch.float64),
                                  torch.full((3,), 12.676,
                                             dtype=torch.float64)
                                  ).sum().backward()
    ok = bool(torch.isfinite(t.grad).all())
    print('  d(unfrozen water)/dT   %s  %s'
          % ([round(v, 6) for v in t.grad.tolist()],
             'finite' if ok else 'NOT FINITE'))
    if not ok:
        failures.append('unfrozen water gradient')

    m = torch.tensor([0.1, 0.35], dtype=torch.float64, requires_grad=True)
    c2 = lambda v: torch.full((2,), v, dtype=torch.float64)
    frozen.soil_conductivity(m, m * 0.4, c2(SOIL_DENS_MIN), c2(BULK_DENS_MIN),
                             c2(0.693), c2(SOIL_DENSITY), c2(BULK_DENSITY),
                             c2(0.0)).sum().backward()
    ok = bool(torch.isfinite(m.grad).all())
    print('  d(conductivity)/d(moist) %s  %s'
          % ([round(v, 6) for v in m.grad.tolist()],
             'finite' if ok else 'NOT FINITE'))
    if not ok:
        failures.append('conductivity gradient')

    print()
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
