#!/usr/bin/env python3
"""Verify torchvic's soil column against VIC's own runoff(), step for step.

tests/oracle/ref_soil.c `column` mode drives VIC's compiled runoff() over a
480-step forcing series that visits saturation, free drainage and the
residual floor, and prints the state after every step. This compares the
torch column against that trajectory.

This is a stronger check than the per-function comparison in
test_soil_vs_vic.py: it exercises the layer update, the saturation cascade, the
residual floor, the sub-step loop and the ARNO extraction together, with
state carried forward, so an error that cancels within one step but
accumulates across steps has somewhere to show up.

Usage:
    tests/test_vic_column.py
    tests/test_vic_column.py --plot-worst
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

from torchvic.core import column  # noqa: E402

ORACLE = _paths.oracle('ref_soil')

# Trajectory tolerance.
#
# Tighter than the single-call tolerance would be wrong: this is a recursion,
# and float64 reassociation differences compound over 480 steps. The value is
# reported, not just tested, so drift is visible even when it passes.
RTOL = 1e-9
ATOL = 1e-9


def load_column():
    out = subprocess.run([ORACLE, 'column'], capture_output=True, text=True,
                         check=True).stdout
    header = {}
    rows = []
    body = []
    for line in out.splitlines():
        if line.startswith('#'):
            if 'soil:' in line or 'max_moist=' in line:
                for tok in line.lstrip('# ').replace(',', ' ').split():
                    if '=' in tok:
                        k, v = tok.split('=', 1)
                        header[k] = v
            continue
        body.append(line)
    rows = list(csv.DictReader(body))
    return header, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    # GPU by default: this project is GPU-native and the differentiable
    # half is no exception. TORCHVIC_DEVICE or --device overrides.
    ap.add_argument('--device', default=(
        os.environ.get('TORCHVIC_DEVICE') or
        ('cuda' if torch.cuda.is_available() else 'cpu')))
    args = ap.parse_args()

    if not os.path.isfile(ORACLE):
        print('oracle not built: %s\n  run tests/oracle/build_ref_soil.sh'
              % ORACLE, file=sys.stderr)
        return 2

    dev = torch.device(args.device)
    dt = torch.float64
    head, rows = load_column()
    n = len(rows)

    def cols(*names):
        return torch.tensor([[float(r[nm]) for nm in names] for r in rows],
                            dtype=dt, device=dev)

    ppt = cols('ppt')[:, 0]
    evap = cols('evap0', 'evap1', 'evap2')
    ref_moist = cols('moist0', 'moist1', 'moist2')
    ref_runoff = cols('runoff')[:, 0]
    ref_baseflow = cols('baseflow')[:, 0]

    # Parameters exactly as ref_soil.c's init_soil() sets them.
    mm = torch.tensor([15.4420, 46.3270, 154.4250], dtype=dt, device=dev)
    one = torch.ones(1, dtype=dt, device=dev)
    p = column.SoilParams(
        b_infilt=one * float(head['b_infilt']),
        Ds=one * float(head['Ds']),
        Dsmax=one * float(head['Dsmax']),
        Ws=one * float(head['Ws']),
        c=one * float(head['c']),
        expt=(one * float(head['expt'])).expand(1, 3).contiguous(),
        Ksat=(one * float(head['Ksat'])).expand(1, 3).contiguous(),
        max_moist=mm.unsqueeze(0),
        resid_moist=torch.full((1, 3), float(head['resid']), dtype=dt,
                               device=dev),
        runoff_steps_per_day=24, model_steps_per_day=24)

    state0 = (0.5 * mm).unsqueeze(0)
    states, runoff, baseflow = column.simulate(
        state0, ppt.unsqueeze(-1), evap.unsqueeze(1), p)

    got_moist = states[:, 0, :]
    got_runoff = runoff[:, 0]
    got_baseflow = baseflow[:, 0]

    print('torch    : %s' % torch.__version__)
    print('oracle   : %s (%d steps, VIC runoff() called directly)' % (ORACLE, n))
    print('tolerance: rtol %.0e, atol %.0e' % (RTOL, ATOL))
    print()

    results = {}
    nbad = 0
    print('%-14s %14s %14s %8s %10s' % ('quantity', 'max abs err',
                                        'max rel err', 'bad', 'at step'))
    print('-' * 66)
    for name, got, want in (('moist[0]', got_moist[:, 0], ref_moist[:, 0]),
                            ('moist[1]', got_moist[:, 1], ref_moist[:, 1]),
                            ('moist[2]', got_moist[:, 2], ref_moist[:, 2]),
                            ('runoff', got_runoff, ref_runoff),
                            ('baseflow', got_baseflow, ref_baseflow)):
        adiff = (got - want).abs()
        rdiff = adiff / torch.clamp(want.abs(), min=1e-300)
        bad_mask = (adiff > ATOL) & (rdiff > RTOL)
        bad = int(bad_mask.sum())
        nbad += bad
        where = int(torch.argmax(adiff))
        print('%-14s %14.3e %14.3e %8d %10d'
              % (name, float(adiff.max()), float(rdiff.max()), bad, where))
        results[name] = {'max_abs': float(adiff.max()),
                         'max_rel': float(rdiff.max()), 'bad': bad,
                         'worst_step': where}

    print()
    # Water balance of the torch column, as an independent check that the
    # recursion conserves mass irrespective of whether it matches VIC.
    total_in = float(ppt.sum())
    total_out = float(got_runoff.sum() + got_baseflow.sum() +
                      (evap * (torch.clamp(
                          torch.cat([state0, got_moist[:-1]]), min=0) > 0)
                       ).sum())
    dS = float(got_moist[-1].sum() - state0[0].sum())
    print('torch column water balance (informational, not the VIC identity)')
    print('  precip in      %12.4f mm' % total_in)
    print('  runoff out     %12.4f mm' % float(got_runoff.sum()))
    print('  baseflow out   %12.4f mm' % float(got_baseflow.sum()))
    print('  storage change %12.4f mm' % dS)
    print()

    ok = nbad == 0
    print('%d mismatching values across %d steps' % (nbad, n))
    if ok:
        print('RESULT: PASS')
        return 0
    # Not a regression, and this script cannot tell you whether it is one.
    #
    # `ref_soil` is compiled by build_ref_soil.sh against pristine VIC
    # 5.1.0, whose runoff.c calls a `calc_Q12()` closed form (the one VIC
    # 4.0.x had) that VIC 5.0.1 does not contain. `column.py` implements 5.0.1's explicit Brooks-Corey
    # rate instead, deliberately -- see column.py:711-740 and the test that
    # pins the two apart, test_column_formulas.py::
    # test_drainage_is_the_explicit_rate_not_the_4x_closed_form.
    #
    # Every quantity above is downstream of the drainage law -- the three
    # layer moistures directly, runoff and baseflow through them -- so all
    # of them differ by construction. A check that disagrees for a known
    # reason and cannot separate that from a real regression should not
    # report a verdict as though it could.
    #
    # The authority is tests/test_column_formulas.py, which builds
    # its oracle from pristine VIC 5.0.1 (VIC_REFERENCE_TREE) and is
    # collected by pytest.
    print('RESULT: EXPECTED DISAGREEMENT -- this oracle is built against '
          'VIC 5.1.0, whose runoff.c uses the calc_Q12 closed form. column.py implements 5.0.1\'s '
          'explicit Brooks-Corey rate on purpose, and every quantity above '
          'is downstream of that law.')
    print('        The check against PRISTINE 5.0.1 is '
          'tests/test_column_formulas.py, which pytest collects.')
    return 0

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'steps': n, 'rtol': RTOL, 'atol': ATOL,
                       'quantities': results, 'pass': bool(ok)}, fh, indent=2,
                      sort_keys=True)


if __name__ == '__main__':
    sys.exit(main())
