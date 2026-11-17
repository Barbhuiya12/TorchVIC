"""The surface energy balance against VIC's own compiled routines.

Two of its pieces are closed forms transcribed from the C, and a transcription
is checked by running the C, not by reading it: the shim below is compiled
at test time against the pristine 5.0.1 tree, so the oracle cannot be the
repo's own modified copy -- which is how two earlier "agreements" were
manufactured. The rest of the module is a root find and is checked for the
properties a root find must have: it finds a root, the residual is zero
there, the bracket widens as VIC's does, a non-finite residual falls back and
is counted instead of poisoning the sum, and the implicit-function-theorem
gradient is finite.

The one constant test exists because CONST_G was 9.81 here and 9.80616 in
VIC, and that alone put stability_correction 4.5e-4 off compiled VIC.
"""
import os
import shutil
import subprocess

import pytest

torch = pytest.importorskip('torch')

from torchvic.core import energy as E  # noqa: E402

import _paths                                                      # noqa: E402

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')
VIC = _paths.VIC501_TREE
# The two oracle tests claim bit-exactness against a C binary running on the
# CPU's libm. On CUDA, float64 log and sqrt land 1 ulp off glibc -- the
# atmos audit measured exactly this -- so the claim is only meaningful on the
# CPU, and other scripts in this directory flip the process default to CUDA
# at import. The solver tests take the live default; the oracles take CPU.
DEVICE = torch.empty(0).device
CPU = torch.device('cpu')

SHIM = r'''
#include <stdio.h>
#include <vic_run.h>
#include <vic_driver_shared_all.h>
parameters_struct param; option_struct options; global_param_struct global_param;
int main(void){
  initialize_parameters();
  double ts_v[]={-15,-5,0,5,15,30}, t1o_v[]={-5,5,15}, dp_v[]={1.0,4.0};
  printf("estimate_T1\n");
  for(int a=0;a<6;a++)for(int b=0;b<3;b++)for(int c=0;c<2;c++){
    printf("%g %g %g %.17g\n",ts_v[a],t1o_v[b],dp_v[c],
      estimate_T1(ts_v[a],t1o_v[b],18.0,0.1,0.1,0.8,0.9,2.5e6,dp_v[c],10800.0));}
  printf("StabilityCorrection\n");
  double tsu[]={-10,0,10,25}, ta[]={-5,5,20}, w[]={0.5,2.85,8.0};
  for(int a=0;a<4;a++)for(int b=0;b<3;b++)for(int c=0;c<3;c++)
    printf("%g %g %g %.17g\n",tsu[a],ta[b],w[c],
      StabilityCorrection(5.0,0.402,tsu[a],ta[b],w[c],0.0738));
  return 0;}
'''


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    """VIC's own numbers, from VIC's own code, compiled here and now."""
    if not os.path.isdir(VIC):
        pytest.skip('pristine VIC 5.0.1 not present under TORCHVIC_REFERENCE_DIR')
    gcc = shutil.which('gcc')
    if gcc is None:
        pytest.skip('no gcc on PATH')
    d = tmp_path_factory.mktemp('ref_energy')
    src = d / 'ref_energy.c'
    src.write_text(SHIM)
    exe = d / 'ref_energy'
    srcs = [os.path.join(VIC, 'vic_run', 'src', f)
            for f in ('estimate_T1.c', 'StabilityCorrection.c')]
    srcs += [os.path.join(VIC, 'drivers', 'shared_all', 'src', f)
             for f in ('initialize_parameters.c', 'vic_log.c', 'open_file.c')]
    # -fcommon: 5.0.1 predates GCC 10 making -fno-common the default.
    cmd = [gcc, '-fcommon', '-O2', '-std=c99', '-w',
           '-I' + os.path.join(VIC, 'vic_run', 'include'),
           '-I' + os.path.join(VIC, 'drivers', 'shared_all', 'include'),
           str(src)] + srcs + ['-o', str(exe), '-lm']
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    out = subprocess.run([str(exe)], capture_output=True, text=True).stdout
    lines = [ln for ln in out.split('\n') if ln.strip()]
    i, j = lines.index('estimate_T1') + 1, lines.index('StabilityCorrection')
    t1 = [tuple(map(float, ln.split())) for ln in lines[i:j]]
    sc = [tuple(map(float, ln.split())) for ln in lines[j + 1:]]
    assert len(t1) == 36 and len(sc) == 36
    return t1, sc


def _t(x):
    return torch.tensor(x, dtype=torch.float64, device=CPU)


def test_estimate_T1_is_bit_exact_against_compiled_vic(oracle):
    worst = 0.0
    for ts, t1o, dp, ref in oracle[0]:
        got = float(E.estimate_T1(_t(ts), _t(t1o), _t(18.0), _t(0.1), _t(0.1),
                                  _t(0.8), _t(0.9), _t(2.5e6), _t(dp),
                                  _t(10800.0)))
        worst = max(worst, abs(got - ref))
    assert worst == 0.0, 'estimate_T1 differs from VIC by %.3e' % worst


def test_stability_correction_is_bit_exact_against_compiled_vic(oracle):
    worst = 0.0
    for tsu, ta, w, ref in oracle[1]:
        got = float(E.stability_correction(_t(5.0), _t(0.402), _t(tsu),
                                           _t(ta), _t(w), _t(0.0738)))
        worst = max(worst, abs(got - ref))
    assert worst == 0.0, 'stability_correction differs from VIC by %.3e' % worst


def test_gravity_is_vics_not_a_round_number():
    # 9.81 put stability_correction 4.5e-4 off compiled VIC. That is the
    # third-decimal-constant class of defect, and it closes every balance.
    assert E.CONST_G == 9.80616


def _args(n, sw):
    t = lambda v: torch.full((n,), float(v), dtype=torch.float64, device=DEVICE)
    def evap_of(net_rad, ra):
        return torch.clamp(net_rad, min=0.0) * 0.0004 * 3.0
    return (t(18.0), t(0.1), t(0.1), t(0.8), t(0.9), t(2.5e6), t(2.6e6), t(4.0),
            t(10800.0), t(0.65), sw, t(320.0), t(40.0), t(5.0), t(0.402),
            t(0.0738), t(2.85), t(1.1), t(2.45e6), evap_of)


def test_the_solver_finds_a_root_and_the_residual_is_zero_there():
    n = 32
    tair = torch.linspace(-10, 35, n, dtype=torch.float64, device=DEVICE)
    t0, t1 = tair - 2.0, tair - 4.0
    sw = torch.linspace(0, 600, n, dtype=torch.float64, device=DEVICE)
    args = (t0,) + _args(n, sw)
    ts, failed = E.solve_surface_temperature(tair, t0, t1, args)
    assert bool(torch.isfinite(ts).all())
    assert int(failed.sum()) == 0
    # root_brent stops (root_brent.c 5.1.0:245-247) once the bracket is
    # within 2 eps |b| + ROOT_BRENT_T of the root: VIC's root, not the exact
    # one. The exact one, to ~1e-13 W m-2, is the Illinois solver's.
    old = E.SURFACE_SOLVER
    try:
        E.SURFACE_SOLVER = 'illinois'
        exact, _ = E.solve_surface_temperature(tair, t0, t1, args)
    finally:
        E.SURFACE_SOLVER = old
    res, _ = E._residual(exact, tair, t1, *args)
    assert float(res.abs().max()) < 1e-9
    assert float((ts - exact).abs().max()) < 2 * E.ROOT_BRENT_T


def test_the_first_bracket_is_too_narrow_and_the_widening_fires():
    # VIC's +/- SURF_DT bracket is the common failure, not the exception; the
    # reference audit measured it too narrow on 86 per cent of real steps.
    n = 32
    tair = torch.linspace(-10, 35, n, dtype=torch.float64, device=DEVICE)
    t0, t1 = tair - 2.0, tair - 4.0
    sw = torch.linspace(300, 600, n, dtype=torch.float64, device=DEVICE)
    args = (t0,) + _args(n, sw)
    mid = 0.5 * (t0 + tair)
    f = lambda x: E._residual(x, tair, t1, *args)[0]
    naive = (f(mid - E.SURF_DT) * f(mid + E.SURF_DT) < 0)
    assert int(naive.sum()) < n, 'the +/-1 bracket held everywhere; test is not exercising the widening'
    ts, failed = E.solve_surface_temperature(tair, t0, t1, args)
    assert int(failed.sum()) == 0     # widening rescued every one


def test_a_non_finite_residual_falls_back_and_is_counted():
    # NaN compares False against everything, so `(flo*fhi) >= 0` let NaN
    # through to bisection, which returned NaN, and 0 * NaN poisoned the
    # cell sum. A non-finite endpoint is a failure and takes Ts_old.
    n = 8
    tair = torch.full((n,), 20.0, dtype=torch.float64, device=DEVICE)
    t0, t1 = tair - 2.0, tair - 4.0
    sw = torch.full((n,), 400.0, dtype=torch.float64, device=DEVICE)
    args = list((t0,) + _args(n, sw))
    def poison(net_rad, ra):
        out = torch.clamp(net_rad, min=0.0) * 0.0012
        out[3] = float('nan')
        return out
    args[-1] = poison
    ts, failed = E.solve_surface_temperature(tair, t0, t1, tuple(args))
    assert bool(failed[3]) and int(failed.sum()) == 1
    assert float(ts[3]) == float(t0[3])
    assert bool(torch.isfinite(ts).all())


def test_the_gradient_through_the_implicit_solve_is_finite():
    n = 16
    tair = torch.linspace(0, 30, n, dtype=torch.float64,
                          device=DEVICE).requires_grad_(True)
    t0, t1 = tair.detach() - 2.0, tair.detach() - 4.0
    sw = torch.full((n,), 400.0, dtype=torch.float64, device=DEVICE)
    args = (t0,) + _args(n, sw)
    ts, _ = E.solve_surface_temperature(tair, t0, t1, args)
    g, = torch.autograd.grad(ts.sum(), tair)
    assert bool(torch.isfinite(g).all())
    assert 0.3 < float(g.mean()) < 1.2     # dTs/dTair is O(1) and positive


def test_energy_closes_at_the_solved_temperature():
    n = 16
    tair = torch.linspace(0, 30, n, dtype=torch.float64, device=DEVICE)
    t0, t1 = tair - 2.0, tair - 4.0
    sw = torch.full((n,), 400.0, dtype=torch.float64, device=DEVICE)
    a = _args(n, sw)
    depth = torch.tensor([0.1, 0.3, 1.0], dtype=torch.float64,
                         device=DEVICE).expand(n, 3)
    out = E.surface_step(tair, t0, t1, *a[:1], a[1], *a[3:], depth)
    closure = out['net_rad'] - (out['latent'] + out['sensible']
                                + out['grnd_flux'] + out['delta_h'])
    # The balance closes to the residual at root_brent's root, which is
    # what VIC reports as its energy error (~1e-6 W m-2 at ROOT_BRENT_T).
    assert float((closure - out['energy_error']).abs().max()) < 1e-9
    assert float(out['energy_error'].abs().max()) < 1e-4


# ---------------------------------------------------------------------------
# The gradient traps. Each of these produced a finite forward value and a NaN
# derivative, which the implicit-function step then divided by. Bisection
# could not see them; only autograd could. Found on the first warm afternoon
# with rain, three days into the run, and traced by measuring d(residual)/dTs
# at the bisected root term by term.
# ---------------------------------------------------------------------------

def test_stability_correction_gradient_is_finite_on_the_stable_side():
    # Ri > 1/16 selects the stable form; the unselected unstable form took
    # sqrt(1 - 16 Ri) of a negative number, and 0 * d(NaN) is NaN.
    z, d, z0 = _t(10.0), _t(0.0), _t(0.03)          # a zero-cover tile's geometry
    wind = _t(0.5)                                  # calm, so Ri is large
    t_air = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    t_surf = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    corr = E.stability_correction(z, d, t_surf, t_air, wind, z0)
    assert torch.isfinite(corr)
    g, = torch.autograd.grad(corr, t_surf)
    assert torch.isfinite(g), 'NaN gradient through the unselected unstable branch'


def test_transpiration_gradient_is_finite_with_a_layer_at_wilting_point():
    # gsm = 0 on a layer masks that layer's transpiration out of the forward
    # value; the masked branch still went through calc_rc's division by gsm.
    from torchvic.core.vegetation import VegMosaic
    kw = dict(dtype=torch.float64, device=CPU)
    m = VegMosaic(cv=torch.ones(1, 1, **kw),
                  lai=torch.full((1, 1, 12), 0.5, **kw),
                  rmin=torch.full((1, 1), 100.0, **kw),
                  rgl=torch.full((1, 1), 100.0, **kw),
                  rarc=torch.full((1, 1), 50.0, **kw),
                  root_frac=torch.tensor([[[0.1, 0.2, 0.7]]], **kw),
                  dtype=torch.float64, device=CPU)
    w_wilt = torch.tensor([[20.0, 60.0, 200.0]], **kw)
    w_crit = torch.tensor([[50.0, 120.0, 400.0]], **kw)
    avail = torch.tensor([[35.0, 60.0, 300.0]], **kw)   # layer 1 exactly at wilting
    rad = torch.tensor([[300.0]], **kw).requires_grad_(True)
    one = torch.ones(1, **kw)
    tr = m._transpiration(0, avail, w_crit, w_wilt,
                          torch.tensor([[250.0]], **kw), 20.0 * one,
                          1500.0 * one, torch.tensor([[40.0]], **kw),
                          rad, 1300.0 * one)
    assert bool(torch.isfinite(tr).all())
    assert float(tr[0, 0, 1]) == 0.0                     # masked, as VIC's if
    g, = torch.autograd.grad(tr.sum(), rad)
    assert bool(torch.isfinite(g).all()), 'NaN gradient through the masked gsm branch'


@pytest.mark.parametrize('solver', ['illinois', 'brent'])
def test_a_root_with_a_non_finite_gradient_falls_back_and_is_counted(
        monkeypatch, solver):
    # implicit_solve returns NaN when d(residual)/dTs is NaN at the root. That
    # is not a converged result; it must take Ts_old and be counted, or it
    # poisons the soil column on the next step.
    n = 6
    tair = torch.full((n,), 20.0, dtype=torch.float64, device=DEVICE)
    t0, t1 = tair - 2.0, tair - 4.0
    sw = torch.full((n,), 400.0, dtype=torch.float64, device=DEVICE)
    args = (t0,) + _args(n, sw)
    monkeypatch.setattr(E, 'SURFACE_SOLVER', solver)
    if solver == 'illinois':
        real = E.implicit_solve
        def poisoned(f, lo, hi, args=(), tol=1e-12, **k):
            r = real(f, lo, hi, args=args, tol=tol, **k)
            r = r.clone(); r[2] = float('nan'); return r
        monkeypatch.setattr(E, 'implicit_solve', poisoned)
    else:
        # the gradient's Newton step, NaN where df/dTs is
        real = E.root_with_grad
        def poisoned(f, root, args=()):
            r = real(f, root, args=args)
            return r + torch.where(torch.arange(r.numel()) == 2,
                                   torch.tensor(float('nan'), dtype=r.dtype),
                                   torch.zeros_like(r))
        monkeypatch.setattr(E, 'root_with_grad', poisoned)
    ts, failed = E.solve_surface_temperature(tair, t0, t1, args)
    assert bool(failed[2]) and int(failed.sum()) == 1
    assert float(ts[2]) == float(t0[2])
    assert bool(torch.isfinite(ts).all())
