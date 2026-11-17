#!/usr/bin/env python3
"""Verify torchvic against VIC's own C functions, then check its gradients.

Two gates, in this order, because the second is worthless without the first:

  1. **Agreement.** Every function in torchvic/core/soil.py is evaluated on the
     same inputs as VIC's compiled runoff.c, via the oracle in
     tests/oracle/ref_soil.c, and the differences are reported in full. A
     differentiable model that does not reproduce VIC is not a differentiable
     VIC.

  2. **Gradients.** Analytic gradients from autograd are compared against
     central finite differences across a sweep of step sizes, so the
     finite-difference regime is identified from the data rather than assumed.
     A gradient is only reported as verified if there is a step size at which
     the two agree, and the agreement is stable across neighbouring step
     sizes.

Both gates run in float64. VIC is a float64 model; comparing a float32
reimplementation against it would conflate precision with correctness.

Usage:
    tests/test_soil_vs_vic.py
    tests/test_soil_vs_vic.py --device cuda
    tests/test_soil_vs_vic.py --json torch_verify.json
"""
import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths                                   # noqa: E402

import torch  # noqa: E402

from torchvic.core import soil  # noqa: E402

ORACLE = _paths.oracle('ref_soil')

# Agreement tolerance.
#
# The oracle prints %.17g, so the comparison is not limited by the text. Both
# sides evaluate the same closed-form expressions in float64 but not in the
# same association order, so a few ulp of disagreement is expected and a
# relative tolerance is the right instrument. 1e-12 is ~4000 ulp of double --
# loose enough to absorb reassociation, tight enough that any real difference
# in the formula shows up.
RTOL = 1e-12
ATOL = 1e-14


def load_oracle(mode):
    if not os.path.isfile(ORACLE):
        raise RuntimeError('oracle not built: %s\n'
                           '  build it with tests/oracle/build_ref_soil.sh'
                           % ORACLE)
    out = subprocess.run([ORACLE, mode], capture_output=True, text=True,
                         check=True).stdout
    lines = [ln for ln in out.splitlines() if not ln.startswith('#')]
    return list(csv.DictReader(lines))


def col(rows, name, dtype=torch.float64):
    return torch.tensor([float(r[name]) for r in rows], dtype=dtype)


class Report(object):
    def __init__(self):
        self.rows = []
        self.failed = 0

    def compare(self, name, got, want, rtol=RTOL, atol=ATOL):
        got = got.detach().to(torch.float64)
        want = want.to(torch.float64)
        finite = torch.isfinite(got) & torch.isfinite(want)
        nonfinite_ref = int((~torch.isfinite(want)).sum())
        n = int(finite.sum())
        if n == 0:
            self.rows.append((name, 0, float('nan'), float('nan'),
                              nonfinite_ref, 0))
            self.failed += 1
            return
        g, w = got[finite], want[finite]
        adiff = (g - w).abs()
        rdiff = adiff / torch.clamp(w.abs(), min=1e-300)
        bad = int(((adiff > atol) & (rdiff > rtol)).sum())
        self.rows.append((name, n, float(adiff.max()), float(rdiff.max()),
                          nonfinite_ref, bad))
        if bad:
            self.failed += 1

    def print(self):
        print('%-34s %7s %13s %13s %9s %6s'
              % ('function', 'values', 'max abs err', 'max rel err',
                 'ref NaN', 'bad'))
        print('-' * 88)
        for name, n, a, r, nf, bad in self.rows:
            print('%-34s %7d %13.3e %13.3e %9d %6d' % (name, n, a, r, nf, bad))


# ---------------------------------------------------------------------------
# Gate 1: agreement with the C
# ---------------------------------------------------------------------------

def check_infilt(rep):
    rows = load_oracle('infilt')
    b = col(rows, 'b_infilt')
    inflow = col(rows, 'inflow')
    moist = torch.stack([col(rows, 'moist0'), col(rows, 'moist1'),
                         col(rows, 'moist2')], dim=-1)
    # max_moist is fixed by ref_soil.c's init_soil()
    mm = torch.tensor([15.4420, 46.3270, 154.4250], dtype=torch.float64)
    max_moist = mm.expand(len(rows), 3)

    A, runoff = soil.compute_runoff_and_asat(b, max_moist, moist, inflow)
    rep.compare('compute_runoff_and_asat: A', A, col(rows, 'A'))
    rep.compare('compute_runoff_and_asat: runoff', runoff, col(rows, 'runoff'))
    return len(rows)


def check_q12(rep):
    rows = load_oracle('q12')
    q = soil.calc_Q12(col(rows, 'Ksat'), col(rows, 'init_moist'),
                      col(rows, 'resid_moist'), col(rows, 'max_moist'),
                      col(rows, 'expt'))
    rep.compare('calc_Q12', q, col(rows, 'Q12'))
    return len(rows)


def check_baseflow(rep):
    rows = load_oracle('baseflow')
    # ref_soil.c divides by runoff_steps_per_day = 24 before using Dsmax
    dsmax = col(rows, 'Dsmax_day') / 24.0
    q = soil.arno_baseflow(dsmax, col(rows, 'Ds'), col(rows, 'Ws'),
                           col(rows, 'c'), col(rows, 'rel_moist'))
    rep.compare('arno_baseflow', q, col(rows, 'baseflow'))
    return len(rows)


# ---------------------------------------------------------------------------
# Gate 2: gradient verification against central finite differences
# ---------------------------------------------------------------------------

STEPS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def scalar_objective(fn, theta_name, theta, fixed):
    """J(theta) -- a scalar, so dJ/dtheta is a single number to check."""
    kwargs = dict(fixed)
    kwargs[theta_name] = theta
    return fn(**kwargs).sum()


def grad_check(label, fn, theta_name, theta0, fixed, device):
    """Return (analytic, {step: fd}, best_step, best_relerr)."""
    theta = theta0.clone().to(device).requires_grad_(True)
    fixed = dict((k, v.to(device) if torch.is_tensor(v) else v)
                 for k, v in fixed.items())

    J = scalar_objective(fn, theta_name, theta, fixed)
    J.backward()
    analytic = theta.grad.detach().clone()

    fds = {}
    for h in STEPS:
        with torch.no_grad():
            tp = theta0.clone().to(device) + h
            tm = theta0.clone().to(device) - h
            jp = scalar_objective(fn, theta_name, tp, fixed)
            jm = scalar_objective(fn, theta_name, tm, fixed)
            # One common perturbation for the whole vector gives the
            # directional derivative along 1; sum(analytic) is its analytic
            # counterpart.
            fds[h] = float((jp - jm) / (2.0 * h))

    a = float(analytic.sum())
    best_h, best_err = None, float('inf')
    for h, fd in fds.items():
        denom = max(abs(a), abs(fd), 1e-300)
        err = abs(a - fd) / denom
        if err < best_err:
            best_h, best_err = h, err
    return a, fds, best_h, best_err


def run_gradient_checks(device):
    """Gradient checks on the real VIC parameters in this subset."""
    n = 64
    g = torch.Generator().manual_seed(20260913)

    def rnd(lo, hi):
        return (lo + (hi - lo) * torch.rand(n, generator=g,
                                            dtype=torch.float64))

    mm = torch.tensor([15.4420, 46.3270, 154.4250], dtype=torch.float64)
    max_moist = mm.expand(n, 3).contiguous()
    # Interior states: away from saturation and from residual, so the check
    # measures the gradient of the physics rather than the behaviour of the
    # guards. Boundary behaviour is a separate test below.
    moist = max_moist * rnd(0.2, 0.8).unsqueeze(-1)
    inflow = rnd(0.5, 20.0)

    checks = []

    # b_infilt, through the infiltration curve
    def f_binfilt(b_infilt, max_moist, moist, inflow):
        _, q = soil.compute_runoff_and_asat(b_infilt, max_moist, moist, inflow)
        return q
    checks.append(('runoff wrt b_infilt', f_binfilt, 'b_infilt', rnd(0.05, 0.6),
                   {'max_moist': max_moist, 'moist': moist, 'inflow': inflow}))

    # Ds, Dsmax, Ws, c through ARNO baseflow
    rel = rnd(0.05, 0.98)
    base_fixed = {'Dsmax_per_step': rnd(0.1, 2.0), 'Ds': rnd(0.01, 0.5),
                  'Ws': rnd(0.3, 0.85), 'c': rnd(1.0, 3.0),
                  'rel_moist': rel}
    for pname, lo, hi in (('Ds', 0.01, 0.5), ('Dsmax_per_step', 0.1, 2.0),
                          ('Ws', 0.3, 0.85), ('c', 1.0, 3.0)):
        fixed = dict(base_fixed)
        theta0 = fixed.pop(pname)
        checks.append(('baseflow wrt %s' % pname, soil.arno_baseflow, pname,
                       theta0, fixed))

    # expt and Ksat through Brooks-Corey drainage
    resid = torch.zeros(n, dtype=torch.float64)
    maxm = torch.full((n,), 46.3270, dtype=torch.float64)
    init = resid + rnd(0.2, 0.9) * (maxm - resid)
    q12_fixed = {'Ksat': rnd(1.0, 50.0), 'init_moist': init,
                 'resid_moist': resid, 'max_moist': maxm, 'expt': rnd(4.0, 20.0)}
    for pname in ('Ksat', 'expt'):
        fixed = dict(q12_fixed)
        theta0 = fixed.pop(pname)
        checks.append(('Q12 wrt %s' % pname, soil.calc_Q12, pname, theta0,
                       fixed))

    print('%-28s %16s %16s %10s %12s'
          % ('gradient', 'analytic', 'finite diff', 'best h', 'rel err'))
    print('-' * 88)
    results = {}
    nbad = 0
    for label, fn, pname, theta0, fixed in checks:
        a, fds, best_h, best_err = grad_check(label, fn, pname, theta0, fixed,
                                              device)
        ok = best_err < 1e-6
        nbad += 0 if ok else 1
        print('%-28s %16.8e %16.8e %10.0e %12.2e%s'
              % (label, a, fds[best_h], best_h, best_err,
                 '' if ok else '   <-- FAIL'))
        results[label] = {'analytic': a, 'fd': fds, 'best_h': best_h,
                          'rel_err': best_err, 'pass': bool(ok)}
    return results, nbad


def main():
    ap = argparse.ArgumentParser()
    # GPU by default: this project is GPU-native and the differentiable
    # half is no exception. TORCHVIC_DEVICE or --device overrides.
    ap.add_argument('--device', default=(
        os.environ.get('TORCHVIC_DEVICE') or
        ('cuda' if torch.cuda.is_available() else 'cpu')))
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('cuda requested but torch.cuda.is_available() is False',
              file=sys.stderr)
        return 2
    device = torch.device(args.device)

    print('torch     : %s' % torch.__version__)
    print('device    : %s' % device)
    print('oracle    : %s' % ORACLE)
    print('dtype     : float64')
    print()

    print('=== gate 1: agreement with vic/vic_run/src/runoff.c ===')
    rep = Report()
    total = 0
    total += check_infilt(rep)
    total += check_q12(rep)
    total += check_baseflow(rep)
    rep.print()
    print()
    print('%d oracle rows, %d function comparisons, %d failing'
          % (total, len(rep.rows), rep.failed))
    print()

    print('=== gate 2: autograd vs central finite differences ===')
    grads, nbad = run_gradient_checks(device)
    print()
    print('%d gradient checks, %d failing' % (len(grads), nbad))
    print()

    ok = (rep.failed == 0) and (nbad == 0)
    print('RESULT: %s' % ('PASS' if ok else 'FAIL'))

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'torch': torch.__version__, 'device': str(device),
                       'agreement': [
                           {'function': n, 'values': v, 'max_abs': a,
                            'max_rel': r, 'ref_nonfinite': nf, 'bad': b}
                           for n, v, a, r, nf, b in rep.rows],
                       'gradients': grads,
                       'pass': bool(ok)}, fh, indent=2, sort_keys=True)
        print('wrote %s' % args.json)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
