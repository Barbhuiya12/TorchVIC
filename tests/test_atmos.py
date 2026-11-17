"""Measure torchvic.core.atmos against VIC's own compiled routines.

    bash tests/oracle/build_ref_soil.sh
    python3 tests/test_atmos.py
"""
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths                                   # noqa: E402
from torchvic.core import atmos  # noqa: E402
from torchvic.core.snow import svp  # noqa: E402

REF = _paths.oracle('ref_soil')


def main():
    if not os.path.exists(REF):
        print('oracle missing; run tests/oracle/build_ref_soil.sh')
        return 1
    out = subprocess.run([REF, 'atmos'], capture_output=True, text=True,
                         check=True).stdout.strip().splitlines()
    cols = out[0].split('|')
    rows = [[float(v) for v in ln.split('|')] for ln in out[1:]]
    d = {c: torch.tensor([r[i] for r in rows], dtype=torch.float64)
         for i, c in enumerate(cols)}

    tair, elev = d['tair'], d['elev']
    got = {
        'scale_height': atmos.scale_height(tair, elev),
        'pressure': atmos.air_pressure(tair, elev),
        'latent_heat': atmos.latent_heat_of_vaporization(tair),
        'air_density': atmos.air_density(tair, atmos.air_pressure(tair, elev)),
        'svp': svp(tair),
        'svp_slope': atmos.svp_slope(tair),
    }
    print('torchvic.core.atmos vs VIC, %d states\n' % len(rows))
    bad = 0
    for k, v in got.items():
        ref = d[k]
        diff = (v - ref).abs()
        rel = float((diff / ref.abs().clamp(min=1e-30)).max())
        nz = int((diff != 0).sum())
        print('%-14s max|abs|=%.3e  max|rel|=%.3e  differing=%d/%d'
              % (k, float(diff.max()), rel, nz, len(rows)))
        if rel > 1e-15:
            bad += 1
    print()
    print('FAIL' if bad else 'PASS: all derivations match VIC')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
