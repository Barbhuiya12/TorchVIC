"""Measure torchvic.model.routing.route against VIC's own compiled convolution.

tests/test_routing.py already checks `route` against `route_reference`,
but both of those are Python and both were written by the same hand. This test
compiles vic/extensions/rout_rvic/src/rout_convolution.c and cshift.c and runs
them, so the comparison is against the code VIC actually ships.

Build the oracle first:  bash tests/oracle/build_ref_route.sh
"""
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths                                   # noqa: E402
from torchvic.model.routing import route, route_reference  # noqa: E402

REF = _paths.oracle('ref_route')


def gamma_uh(nuh, shape, scale, dtype=torch.float64):
    """A unit hydrograph with the shape RVIC parameter files carry: a single
    rising and falling pulse, normalised so the column sums to one."""
    k = torch.arange(1, nuh + 1, dtype=dtype)
    uh = k ** (shape - 1.0) * torch.exp(-k / scale)
    return uh / uh.sum()


def make_case(ns, no, nuh, nt, seed, offsets=None, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    runoff = torch.rand(nt, ns, generator=g, dtype=dtype) * 20.0
    uh = torch.stack([gamma_uh(nuh, 2.0 + 0.5 * (i % 4), 3.0 + (i % 5), dtype)
                      for i in range(ns)], dim=1)
    s2o = torch.arange(ns) % no
    area = torch.rand(no, generator=g, dtype=dtype) * 1e8 + 1e8
    off = (torch.zeros(ns, dtype=torch.long) if offsets is None
           else torch.as_tensor(offsets, dtype=torch.long))
    return runoff, uh, s2o, area, off


def run_c(path, runoff, uh, s2o, area, off, dt):
    nt, ns = runoff.shape
    nuh, no = uh.shape[0], area.shape[0]
    with open(path, 'w') as fh:
        fh.write('%d %d %d %d %.17g\n' % (ns, no, nuh, nt, dt))
        fh.write(' '.join(str(int(x)) for x in s2o) + '\n')
        fh.write(' '.join(str(int(x)) for x in off) + '\n')
        for r in range(nuh):
            fh.write(' '.join('%.17g' % v for v in uh[r]) + '\n')
        fh.write(' '.join('%.17g' % v for v in area) + '\n')
        for t in range(nt):
            fh.write(' '.join('%.17g' % v for v in runoff[t]) + '\n')
    out = subprocess.run([REF, path], capture_output=True, text=True,
                         check=True).stdout
    return torch.tensor([[float(v) for v in line.split()]
                         for line in out.strip().splitlines()],
                        dtype=torch.float64)


def compare(name, got, ref):
    diff = (got - ref).abs()
    denom = ref.abs().clamp(min=1e-30)
    amax = float(diff.max())
    rmax = float((diff / denom).max())
    print('%-34s n=%-7d max|abs|=%.3e  max|rel|=%.3e'
          % (name, got.numel(), amax, rmax))
    return amax, rmax


def main():
    if not os.path.exists(REF):
        print('oracle missing; run tests/oracle/build_ref_route.sh')
        return 1
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(prefix='ref_route_'),
                       'ref_route_input.txt')
    dt = 86400.0
    failures = []

    print('route() vs VIC rout_convolution.c, double precision\n')
    for ns, no, nuh, nt, seed in [(1, 1, 10, 40, 1), (8, 1, 24, 120, 2),
                                  (32, 4, 48, 200, 3), (57, 8, 96, 365, 4)]:
        runoff, uh, s2o, area, off = make_case(ns, no, nuh, nt, seed)
        ref = run_c(tmp, runoff, uh, s2o, area, off, dt)
        got = route(runoff, uh, s2o, no, area=area, dt=dt)
        a, r = compare('%2d src -> %d outlet, uh=%d, nt=%d'
                       % (ns, no, nuh, nt), got, ref)
        if r > 1e-13:
            failures.append(('route', ns, no, r))

        gotr = route_reference(runoff, uh, s2o, no)
        gotr = gotr * area / 1000.0 / dt
        a2, r2 = compare('   route_reference, same case', gotr, ref)
        if r2 > 1e-13:
            failures.append(('route_reference', ns, no, r2))

    # Mass conservation: with unit hydrographs summing to one and no offsets,
    # everything that goes in must come out, once the tail has drained.
    print()
    runoff, uh, s2o, area, off = make_case(16, 2, 32, 400, 9)
    tail = torch.zeros(200, 16, dtype=runoff.dtype)
    r_all = torch.cat([runoff, tail])
    q = route(r_all, uh, s2o, 2)
    for o in range(2):
        m = s2o == o
        print('outlet %d  in=%.12f mm  out=%.12f mm  diff=%.3e'
              % (o, float(runoff[:, m].sum()), float(q[:, o].sum()),
                 float(q[:, o].sum() - runoff[:, m].sum())))
        if abs(float(q[:, o].sum() - runoff[:, m].sum())) > 1e-9:
            failures.append(('mass', o, 0, 0))

    # Non-zero source_time_offset is a known, measured divergence, not a
    # failure. convolution() writes a source's contribution into ring rows
    # [offset, offset + n_timesteps - 1], but cshift is called with
    # nx = n_timesteps and so rotates only the first n_timesteps rows of a
    # ring that is full_time_length long. Anything written at or past row
    # n_timesteps never reaches row 0, which is where discharge is read, and
    # is silently dropped. The loss is measured below: it grows with the
    # offset and reaches everything when the offset equals the unit
    # hydrograph length. route() does not reproduce this.
    print('\nMass conservation of VIC convolution() vs source_time_offset')
    print('(unit hydrographs sum to 1, long zero tail, so nothing may be '
          'lost)')
    print('%-8s %14s %14s %9s' % ('offset', 'in (mm)', 'out (mm)', 'lost'))
    nuh_m, ns_m, nt_m = 20, 4, 120
    losses = []
    for off_val in (0, 1, 2, 5, 10, 19, 20):
        r_m, uh_m, s2o_m, area_m, _ = make_case(ns_m, 1, nuh_m, nt_m, 7)
        r_m[nt_m // 2:] = 0.0
        off_m = torch.full((ns_m,), off_val, dtype=torch.long)
        ref_m = run_c(tmp, r_m, uh_m, s2o_m, area_m, off_m, 1.0)
        out_m = float((ref_m[:, 0] * 1000.0 / float(area_m[0])).sum())
        in_m = float(r_m.sum())
        losses.append((off_val, 100.0 * (in_m - out_m) / in_m))
        print('%-8d %14.6f %14.6f %8.2f%%'
              % (off_val, in_m, out_m, losses[-1][1]))
    if abs(losses[0][1]) > 1e-9:
        failures.append(('offset-0 mass', 0, 0, losses[0][1]))
    if losses[-1][1] < 99.99:
        failures.append(('offset-drop mechanism not reproduced', 0, 0,
                         losses[-1][1]))

    print()
    runoff, uh, s2o, area, _ = make_case(6, 2, 20, 80, 11)
    off = torch.tensor([0, 1, 2, 3, 0, 5], dtype=torch.long)
    ref = run_c(tmp, runoff, uh, s2o, area, off, dt)
    got = route(runoff, uh, s2o, 2, source_time_offset=off, area=area, dt=dt)
    a, r = compare('non-zero offset (expected to differ)', got, ref)
    offset_matches = r <= 1e-13

    print()
    if failures:
        print('FAIL: %d comparison(s) outside 1e-13 relative' % len(failures))
        for f in failures:
            print('   ', f)
        return 1
    print('PASS: zero-offset routing agrees with VIC to machine precision,')
    print('      and the offset mass loss in VIC is reproduced and bounded.')
    if offset_matches:
        print('UNEXPECTED: non-zero offset now agrees with VIC. Either the')
        print('  extension was fixed upstream or route() has acquired the')
        print('  same defect. Do not accept this without reading both.')
        return 1
    print('DIVERGENCE (deliberate): non-zero source_time_offset differs by')
    print('  %.3e relative. VIC drops the part of a shifted hydrograph that'
          % r)
    print('  lands past ring row n_timesteps; route() keeps it and conserves')
    print('  mass. RVIC parameter files in normal use carry offset 0, where')
    print('  the two agree exactly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
