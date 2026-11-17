"""Measure torchvic.core.vegetation against VIC's compiled canopy routines.

    bash tests/oracle/build_ref_soil.sh
    python3 tests/test_vegetation.py

An earlier, vegetation-free version of this model measured `rs` and
`et_beta` -- the two stand-ins for vegetation -- as the leading controls on
evapotranspiration. This checks the physics they stand in for against the
code it comes from, before it replaces them.
"""
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths                                   # noqa: E402

from torchvic.core import vegetation  # noqa: E402

REF = _paths.oracle('ref_soil')


def main():
    if not os.path.exists(REF):
        print('oracle missing; run tests/oracle/build_ref_soil.sh')
        return 1
    out = subprocess.run([REF, 'canopy'], capture_output=True, text=True,
                         check=True).stdout.strip().splitlines()
    cols = out[0].split('|')
    rows = [[float(v) for v in ln.split('|')] for ln in out[1:]]
    d = {c: torch.tensor([r[i] for r in rows], dtype=torch.float64)
         for i, c in enumerate(cols)}
    n = len(rows)
    print('torchvic.core.vegetation vs VIC, %d states\n' % n)

    rc = vegetation.calc_rc(d['rs'], d['net_short'], d['RGL'], d['tair'],
                            d['vpd'], d['lai'], d['gsm_inv'])
    ev = vegetation.penman(d['tair'], d['elev'], d['rad'], d['vpd'],
                           d['ra'], rc, d['rarc'])

    bad = 0
    for name, got, ref in (('calc_rc', rc, d['rc']),
                           ('penman', ev, d['penman'])):
        diff = (got - ref).abs()
        rel = float((diff / ref.abs().clamp(min=1e-30)).max())
        ndiff = int((diff != 0).sum())
        print('%-10s max|abs|=%.3e  max|rel|=%.3e  differing=%d/%d'
              % (name, float(diff.max()), rel, ndiff, n))
        if rel > 1e-14:
            bad += 1

    # A gradient must reach the parameters the calibration will free.
    rs = d['rs'].clone().requires_grad_(True)
    lai = d['lai'].clone().requires_grad_(True)
    rc2 = vegetation.calc_rc(rs, d['net_short'], d['RGL'], d['tair'],
                             d['vpd'], lai, d['gsm_inv'])
    vegetation.penman(d['tair'], d['elev'], d['rad'], d['vpd'], d['ra'],
                      rc2, d['rarc']).sum().backward()
    ok_g = (rs.grad is not None and torch.isfinite(rs.grad).all()
            and lai.grad is not None and torch.isfinite(lai.grad).all())
    print('\ngradients finite: d/d(rs) %s, d/d(lai) %s'
          % (bool(torch.isfinite(rs.grad).all()),
             bool(torch.isfinite(lai.grad).all())))

    print('\n%s' % ('PASS' if bad == 0 and ok_g else 'FAIL'))
    return 0 if bad == 0 and ok_g else 1


if __name__ == '__main__':
    sys.exit(main())
