#!/usr/bin/env python3
"""Check the differentiable router against RVIC's own algorithm.

torchvic.model.routing.route rewrites RVIC's ring-buffer convolution as a single
depthwise conv1d. route_reference transcribes the C loop directly. If the two
agree, the fast path implements the algorithm VIC ships rather than something
that merely resembles it.

Also checks the properties a router must have irrespective of implementation:
mass conservation when the unit hydrographs sum to one, and finite gradients
to both the runoff and the hydrograph -- the latter being what calibration
needs, since a differentiable router is only useful if the routing
parameters themselves can be calibrated.

Run: tests/test_routing.py
"""
import os
import sys


import torch                                    # noqa: E402
from torchvic.model import routing  # noqa: E402

# This project is GPU-native and the differentiable half is no exception.
# Override with TORCHVIC_DEVICE=cpu.
DEVICE = os.environ.get('TORCHVIC_DEVICE') or (
    'cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)


DT = torch.float64


def make_uh(nuh, ns, generator):
    """Unit hydrographs shaped like real ones: a rising limb, a peak, a long
    recession, each column normalised to sum to 1."""
    k = torch.arange(1, nuh + 1, dtype=DT).unsqueeze(1)
    scale = 2.0 + 6.0 * torch.rand(1, ns, dtype=DT, generator=generator)
    uh = (k / scale) * torch.exp(-k / scale)
    return uh / uh.sum(dim=0, keepdim=True)


def main():
    failures = []
    g = torch.Generator(device=DEVICE).manual_seed(11)
    nt, ns, nuh, no = 60, 5, 12, 2
    runoff = torch.rand(nt, ns, dtype=DT, generator=g) * 4.0
    uh = make_uh(nuh, ns, g)
    s2o = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    offs = torch.tensor([0, 2, 0, 1, 3], dtype=torch.long)

    print('routing: conv1d against RVIC ring-buffer transcription')
    for label, off in (('no offsets', None), ('with time offsets', offs)):
        fast = routing.route(runoff, uh, s2o, no, source_time_offset=off)
        ref = routing.route_reference(runoff, uh, s2o, no,
                                      source_time_offset=off)
        err = (fast - ref).abs().max().item()
        rel = (err / ref.abs().max().clamp(min=1e-30)).item()
        print('  %-20s max|abs| %9.3g   max|rel| %9.3g   %s'
              % (label, err, rel, 'PASS' if rel < 1e-12 else 'FAIL'))
        if rel >= 1e-12:
            failures.append(label)

    # Mass conservation. With unit hydrographs summing to 1 and a long enough
    # tail, everything that goes in comes out.
    print('\nmass conservation')
    long_nt = 400
    r2 = torch.rand(long_nt, ns, dtype=DT, generator=g) * 4.0
    # Zero the last steps so the hydrograph tail can drain within the record.
    r2[long_nt - 3 * nuh:] = 0.0
    d2 = routing.route(r2, uh, s2o, no)
    print('  runoff in   %.10f' % r2.sum())
    print('  discharge out %.10f' % d2.sum())
    resid = abs(float(r2.sum() - d2.sum()))
    print('  residual    %.3g' % resid)
    if resid > 1e-9:
        failures.append('mass conservation')

    # Gradients to runoff and, critically, to the unit hydrograph itself.
    print('\ngradients')
    r3 = runoff.clone().requires_grad_(True)
    uh3 = uh.clone().requires_grad_(True)
    routing.route(r3, uh3, s2o, no).sum().backward()
    for name, t in (('d(discharge)/d(runoff)', r3),
                    ('d(discharge)/d(unit_hydrograph)', uh3)):
        ok = bool(torch.isfinite(t.grad).all()) and bool((t.grad != 0).any())
        print('  %-34s max %9.4g   %s'
              % (name, t.grad.abs().max(), 'finite and non-zero' if ok
                 else 'PROBLEM'))
        if not ok:
            failures.append(name)

    print()
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
