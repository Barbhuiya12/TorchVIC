"""VIC's CONSTANTS file is read, and calc_Q12 stays finite at large exponents.

* A global file's CONSTANTS VEG_LAI_WATER_FACTOR reaches the parameters a
  case runs with; any other constant is refused by name (VIC would apply it
  and TorchVIC would not, which is a silent difference).
* column._brooks_corey_integral is VIC 5.1.0's calc_Q12 (runoff.c:478-481)
  in its literal C form, bit-identical to the compiled pristine C (the
  earlier expm1/log1p ratio form was ~16 ulp off VIC and was the sole
  January soil seed on the himal case). Where the C form is not finite (expt from ~53 to 250; VIC
  itself gives -inf/NaN) it falls back to the ratio form, so value and
  gradient stay finite there.
"""
import itertools
import math
import os
import subprocess
import tempfile

import pytest
import torch

from torchvic.core import column, version as core_version, soil as soilmod
from torchvic.io.global_param import GlobalParam


def _global(tmp_path, constants_text):
    c = tmp_path / 'constants.txt'
    c.write_text(constants_text)
    g = tmp_path / 'global.txt'
    g.write_text('MODEL_STEPS_PER_DAY 8\nCONSTANTS %s\n' % c)
    return GlobalParam(str(g))


def test_veg_lai_water_factor_is_read(tmp_path):
    gp = _global(tmp_path, '# canopy\nVEG_LAI_WATER_FACTOR 0.05\n')
    assert gp.constants() == {'lai_water_factor': 0.05}
    assert 'CONSTANTS' not in gp.unsupported


def test_a_constant_vic_gpu_does_not_apply_is_refused(tmp_path):
    gp = _global(tmp_path, 'VEG_LAI_WATER_FACTOR 0.05\nLAPSE_RATE -0.0065\n')
    with pytest.raises(ValueError, match='LAPSE_RATE'):
        gp.constants()


def test_no_constants_file_is_no_override(tmp_path):
    g = tmp_path / 'global.txt'
    g.write_text('MODEL_STEPS_PER_DAY 8\n')
    assert GlobalParam(str(g)).constants() == {}


def test_case_params_carry_the_constant(tmp_path):
    from torchvic import case as _case
    gp = _global(tmp_path, 'VEG_LAI_WATER_FACTOR 0.037\n')
    c = _case.Case.__new__(_case.Case)
    c.constants, c.dtype, c.device = gp.constants(), torch.float64, 'cpu'
    assert float(c.params()['lai_water_factor']) == 0.037
    # an explicit override still wins
    assert float(c.params(lai_water_factor=0.2)['lai_water_factor']) == 0.2


# ---------------------------------------------------------------------------
# calc_Q12 at large expt
# ---------------------------------------------------------------------------

D = torch.float64


def _span_pow_integral(Ksat, tmp_liq, resid_moist, max_moist, expt):
    """column._brooks_corey_integral as it stood at 85a1af7e, verbatim: the
    reference the ratio form must reproduce. Copied rather than read from
    the history, so the comparison still runs in an export of the tree."""
    span = torch.clamp(max_moist - resid_moist, min=soilmod.TINY)
    a = torch.clamp(tmp_liq - resid_moist, min=0.0)
    em1 = expt - 1.0
    b = Ksat * em1 / torch.pow(span, expt)
    x = b * soilmod._pow_nonneg(a, em1)
    return -a * torch.expm1(-torch.log1p(x) / em1)


def _file_like_grid():
    """980 states: Ksat per hourly runoff sub-step (mm/day / 24,
    runoff.c:151-152) from 1 to the file's 1257 mm/day, the India file's
    expt range (3.2 ... 27.56), saturation from 1e-6 to 1, and (resid,
    max_moist) mm from a 0.1 m top layer to a 4 m bottom one."""
    rows = list(itertools.product(
        [1.0 / 24, 10.0 / 24, 60.0 / 24, 250.0 / 24, 1257.0959 / 24],
        [3.2, 4.0, 8.0, 11.7, 14.66, 25.0, 27.56],
        [1e-6, 1e-3, 0.05, 0.3, 0.7, 0.999, 1.0],
        [(2.0, 42.0), (6.3, 126.0), (20.0, 420.0), (60.0, 1500.0)]))
    K = torch.tensor([r[0] for r in rows], dtype=D)
    e = torch.tensor([r[1] for r in rows], dtype=D)
    r = torch.tensor([r[3][0] for r in rows], dtype=D)
    M = torch.tensor([r[3][1] for r in rows], dtype=D)
    f = torch.tensor([r[2] for r in rows], dtype=D)
    return K, r + f * (M - r), r, M, e


def test_ratio_fallback_is_the_span_power_form_at_file_values():
    """The large-expt fallback (`_brooks_corey_integral_ratio`) is the
    span-power form to rounding: max rel diff 3.0e-15 over 980 states."""
    K, t, r, M, e = _file_like_grid()
    old = _span_pow_integral(K, t, r, M, e)
    new = column._brooks_corey_integral_ratio(K, t, r, M, e)
    assert bool((old > 0).all()) and bool(torch.isfinite(new).all())
    rel = float(((new - old).abs() / old.abs()).max())
    print('calc_Q12 ratio vs span-power form: %d states, max rel diff '
          '%.3e, %d bit-equal' % (old.numel(), rel, int((old == new).sum())))
    assert rel <= 1e-12


def test_drainage_is_vic510s_own_calc_q12_bit_for_bit():
    """column's drainage IS VIC 5.1.0's calc_Q12, bit for bit.

    calc_Q12 is cut out of the pristine runoff.c and compiled, not
    re-typed: an oracle must call, not transcribe. TorchVIC uses the
    literal C form (runoff.c:478-481) so its
    soil water is bit-identical to VIC; the expm1/log1p ratio form is more
    accurate where Q12 << a (the C cancels there: 3.1e-10 vs 80-digit
    mpmath, the ratio form 2.0e-15) but its ~16-ulp gap to VIC was the
    sole January soil seed on himal and the NIJSSEN2001 igp noise.
    Exactness to VIC wins over accuracy.

    Pinned: all 980 states finite and bit-equal to the C element by
    element; batched within 1 ulp of tmp_liq (SLEEF pow); grad path equal
    to the no-grad path; and the ratio form is still measured off the C (so the test fails
    if column goes back to it).
    """
    import _paths
    src = os.path.join(_paths.VIC51_TREE, 'vic_run', 'src', 'runoff.c')
    src = src if os.path.exists(src) else None
    if src is None:
        pytest.skip('pristine VIC 5.1.0 tree (reference/VIC-5.1.0) not found')
    lines = open(src).read().splitlines()
    i = lines.index('calc_Q12(double Ksat,')
    j = next(k for k in range(i, len(lines)) if lines[k] == '}')
    fn = '\n'.join(lines[i - 1:j + 1])
    assert 'pow(max_moist - resid_moist, expt)' in fn, fn
    shim = ('#include <math.h>\n#include <stdio.h>\n' + fn + '\n'
            'int main(void) {\n'
            '    double K, t, r, M, e;\n'
            '    while (scanf("%lf %lf %lf %lf %lf", &K, &t, &r, &M, &e)'
            ' == 5)\n'
            '        printf("%a\\n", calc_Q12(K, t, r, M, e));\n'
            '    return 0;\n'
            '}\n')
    tmp = tempfile.mkdtemp(prefix='ref_q12_')
    c, exe = os.path.join(tmp, 'q12.c'), os.path.join(tmp, 'q12')
    with open(c, 'w') as fh:
        fh.write(shim)
    b = subprocess.run(['gcc', '-O0', '-o', exe, c, '-lm'],
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip('cannot compile the calc_Q12 shim: ' + b.stderr[-500:])
    K, t, r, M, e = _file_like_grid()
    # hex floats both ways: %.17g would measure the printf, not calc_Q12
    inp = '\n'.join(' '.join(float(v[i]).hex() for v in (K, t, r, M, e))
                    for i in range(K.numel()))
    out = subprocess.run([exe], input=inp, capture_output=True, text=True,
                         check=True).stdout.split()
    assert len(out) == K.numel()
    ref = torch.tensor([float.fromhex(x) for x in out], dtype=D)
    assert bool(torch.isfinite(ref).all())
    with torch.no_grad():
        new = column._brooks_corey_integral(K, t, r, M, e)
    Kg = K.clone().requires_grad_(True)
    newg = column._brooks_corey_integral(Kg, t, r, M, e)
    one = torch.cat([column._brooks_corey_integral(
        K[k:k + 1], t[k:k + 1], r[k:k + 1], M[k:k + 1], e[k:k + 1])
        for k in range(K.numel())])
    # Batched, torch's vectorised (SLEEF) pow can round 1 ulp off libm's;
    # the C form's cancellation turns that into <= 1 ulp of tmp_liq in Q12
    # (measured: 10 of 980 states, 0.5-1 ulp of tmp_liq, max 1.1e-13 mm).
    # Element by element it is libm's pow exactly.
    ulp_t = torch.tensor([math.ulp(float(x)) for x in t], dtype=D)
    print('calc_Q12 C-literal vs VIC 5.1.0 C: batch-1 %d of %d bit-equal; '
          'batch-%d %d bit-equal (grad path %d), max %.2f ulp of tmp_liq'
          % (int((one == ref).sum()), K.numel(), K.numel(),
             int((new == ref).sum()), int((newg == ref).sum()),
             float(((new - ref).abs() / ulp_t).max())))
    assert torch.equal(one, ref)
    assert bool(((new - ref).abs() <= ulp_t).all())
    assert torch.equal(newg.detach(), new)
    ratio = column._brooks_corey_integral_ratio(K, t, r, M, e)
    assert not torch.equal(ratio, ref)


@pytest.mark.parametrize('expt', [53.0, 150.0, 160.0, 250.0])
def test_calc_q12_value_and_gradient_are_finite_at_large_expt(expt):
    """The old ordering's ceiling is expt < 53; NaN was measured at 160 and above.
    The C-literal form is not finite here either (VIC gives -inf/NaN), so
    these states exercise the ratio-form fallback and the safe-input
    substitution that keeps torch.where from leaking a NaN gradient.

    States: a 42 mm top layer and a 1,500 mm bottom one, each at residual,
    1e-9 of the span above it, mid-range and saturated. The gradient is
    probed with respect to every input, at the limits, because a finite
    value over a NaN gradient is exactly what torch.where hides.
    """
    r = torch.tensor([2.0] * 4 + [60.0] * 4, dtype=D)
    M = torch.tensor([42.0] * 4 + [1500.0] * 4, dtype=D)
    f = torch.tensor([0.0, 1e-9, 0.5, 1.0] * 2, dtype=D)
    K = torch.full_like(r, 250.0 / 24).requires_grad_(True)
    t = (r + f * (M - r)).detach().requires_grad_(True)
    rr = r.clone().requires_grad_(True)
    MM = M.clone().requires_grad_(True)
    e = torch.full_like(r, expt).requires_grad_(True)
    q = column._brooks_corey_integral(K, t, rr, MM, e)
    assert bool(torch.isfinite(q).all()), q
    a = (t - r).detach()
    # VIC's C form cancels where Q12 << a and can return a few ulp of t
    # below zero (-2.3e-13 on the 1,500 mm layer); TorchVIC returns VIC's
    # number there, not a clamped one.
    assert bool((q >= -1e-15 * t.detach()).all()), q
    assert bool((q <= a * (1 + 1e-12)).all()), q
    q.sum().backward()
    for name, g in (('Ksat', K.grad), ('tmp_liq', t.grad),
                    ('resid', rr.grad), ('max_moist', MM.grad),
                    ('expt', e.grad)):
        assert bool(torch.isfinite(g).all()), (name, g)
    # The span-power ordering overflows once span**expt passes 1.8e308
    # (1440 mm: expt > 97.6), which is what makes this a test of the fix.
    with torch.no_grad():
        old = _span_pow_integral(K, t, r, M, e)
    assert bool(torch.isfinite(old).all()) == (expt < 97.6)


def test_a_5_1_column_step_at_expt_250_has_finite_gradients():
    """The same through column.step on the 5.1.0 core: 20 steps of a wet
    three-layer column with a 1.5 m bottom layer at expt 250, and the
    gradient of baseflow plus the final top-layer storage with respect to
    expt and the initial state. (Runoff + baseflow + all storage would be
    conserved water, whose gradient is zero whatever the drainage does.)"""
    old_core = core_version.CORE
    core_version.set_core('5.1.0')
    try:
        maxm = torch.tensor([[42.0, 126.0, 1500.0]], dtype=D)
        resid = torch.tensor([[2.0, 6.3, 60.0]], dtype=D)
        e = torch.full((1, 3), 250.0, dtype=D, requires_grad=True)
        p = column.SoilParams(
            b_infilt=torch.tensor([0.2], dtype=D),
            Ds=torch.tensor([0.001], dtype=D),
            Dsmax=torch.tensor([10.0], dtype=D),
            Ws=torch.tensor([0.9], dtype=D), c=torch.tensor([2.0], dtype=D),
            expt=e, Ksat=torch.tensor([[250.0, 150.0, 60.0]], dtype=D),
            max_moist=maxm, resid_moist=resid,
            runoff_steps_per_day=24, model_steps_per_day=8)
        s0 = (resid + 0.97 * (maxm - resid)).requires_grad_(True)
        s, total = s0, torch.zeros((), dtype=D)
        for k in range(20):
            ppt = torch.tensor([3.0 if k < 5 else 0.0], dtype=D)
            evap = torch.tensor([[0.1, 0.05, 0.0]], dtype=D)
            s, ro, bf = column.step(s, ppt, evap, p)
            total = total + bf.sum()
        assert bool(torch.isfinite(s).all()) and bool(torch.isfinite(total))
        (total + s[:, 0].sum()).backward()
        assert bool(torch.isfinite(e.grad).all()), e.grad
        assert bool(torch.isfinite(s0.grad).all()), s0.grad
        assert float(e.grad.abs().sum()) > 0.0
    finally:
        core_version.set_core(old_core)
