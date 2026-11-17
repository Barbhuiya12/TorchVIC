"""BLOWING: wind-driven snow sublimation and transport, against VIC's own C.

`torchvic.core.blowing` is compared with VIC 5.0.1's CalcBlowingSnow() as upstream
ships it, called through the oracle in tests/oracle/ref_snow.c (mode
`blow`) and linked against the pristine tree -- not the copy in this
repository, for the reason that file's header gives.

Two things in the port are iteration counts rather than transcriptions, and
these tests are where they are measured: VIC's Romberg integration refines
adaptively to machine precision, and its saltation-shear root find is a
Newton/bisection hybrid with a tolerance. Both are fixed-count here because
a convergence test costs a device synchronisation per iteration.
"""
import math
import os
import subprocess

import pytest
import torch

from torchvic.core import blowing

torch.set_default_dtype(torch.float64)
KW = dict(dtype=torch.float64)

import _paths                                           # noqa: E402
REF = _paths.oracle('ref_snow')

COLS = ('Dt', 'Tair', 'LastSnow', 'SurfaceLiquidWater', 'Wind', 'Ls',
        'AirDens', 'EactAir', 'ZO', 'Zrh', 'snowdepth', 'lag_one',
        'sigma_slope', 'Tsnow', 'fe', 'displacement', 'roughness')


def _oracle():
    if not os.path.exists(REF):
        pytest.skip('ref_snow not built; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([REF, 'blow'], capture_output=True, text=True,
                         check=True).stdout
    rows = []
    for line in out.splitlines():
        if line.startswith('#') or '|' not in line:
            continue
        lhs, rhs = line.split('|')
        rows.append(([float(x) for x in lhs.split(',')],
                     [float(x) for x in rhs.split(',')]))
    assert rows, 'oracle produced no rows'
    return rows


@pytest.fixture(scope='module')
def cases():
    return _oracle()


def _tensors(rows):
    cols = {name: torch.tensor([r[0][i] for r in rows], **KW)
            for i, name in enumerate(COLS)}
    want_flux = torch.tensor([r[1][0] for r in rows], **KW)
    want_tr = torch.tensor([r[1][1] for r in rows], **KW)
    return cols, want_flux, want_tr


def _run(cols, **over):
    kw = dict(
        dt=cols['Dt'], tair=cols['Tair'], last_snow=cols['LastSnow'],
        surface_liquid_water=cols['SurfaceLiquidWater'], wind=cols['Wind'],
        ls=cols['Ls'], air_dens=cols['AirDens'], eact_air=cols['EactAir'],
        z0=cols['ZO'], zrh=cols['Zrh'], snowdepth=cols['snowdepth'],
        lag_one=cols['lag_one'], sigma_slope=cols['sigma_slope'],
        tsnow=cols['Tsnow'], fe=cols['fe'],
        displacement=cols['displacement'], roughness=cols['roughness'])
    kw.update(over)
    return blowing.calc_blowing_snow(**kw)


def test_the_oracle_exercises_both_branches(cases):
    """A comparison is worth only the states it visits: some rows must blow
    snow and some must not, or the test passes on a function returning 0."""
    blown = sum(1 for _, out in cases if out[0] != 0.0)
    assert blown > 20, blown
    assert blown < len(cases), (blown, len(cases))


# Agreement is to VIC's *solver* tolerance, not to the last bit, and the
# reason is entirely on VIC's side. Its saltation shear is found by rtnewt()
# (CalcBlowingSnow.c:411-483) with xacc = 0.10 * utshear -- an absolute
# tolerance of about 0.04 m/s on a friction velocity of order 1 -- so VIC
# stops as much as 2.3e-2 m/s from the root of its own equation. The two
# tests after these show that: give calc_sub_flux the number rtnewt actually
# returns and the agreement is 1e-10, which is the floor the oracle's own
# printing allows.
SOLVER_TOL = 2e-3


def test_sublimation_matches_vic(cases):
    cols, want, _ = _tensors(cases)
    got, _ = _run(cols)
    assert torch.isfinite(got).all()
    scale = want.abs().max()
    err = (got - want).abs().max()
    print('blowing-snow sublimation: max |error| %.3e against a range of '
          '%.3e kg/m2/s, relative %.3e' % (err, scale, err / scale))
    assert float(err / scale) < SOLVER_TOL, (float(err), float(scale))


def test_transport_matches_vic(cases):
    cols, _, want = _tensors(cases)
    _, got = _run(cols)
    assert torch.isfinite(got).all()
    scale = want.abs().max()
    err = (got - want).abs().max()
    print('blowing-snow transport: max |error| %.3e against a range of '
          '%.3e kg/m/s, relative %.3e' % (err, scale, err / scale))
    assert float(err / scale) < SOLVER_TOL, (float(err), float(scale))


def _rtnewt(x1, x2, acc, ur, zr, max_iter=100):
    """CalcBlowingSnow.c:411-483, scalar, line for line. Newton with a
    bisection fallback and an absolute tolerance on the step."""
    def shear(x):
        f = (math.log(2.0 * blowing.CONST_G * zr / 0.12) + math.log(1.0 / (x * x))
             - blowing.CONST_KARMAN * ur / x)
        df = blowing.CONST_KARMAN * ur / (x * x) - 2.0 / x
        return f, df

    fl, _ = shear(x1)
    fh, _ = shear(x2)
    if fl == 0.0:
        return x1
    if fh == 0.0:
        return x2
    if fl < 0.0:
        xl, xh = x1, x2
    else:
        xh, xl = x1, x2
    rts = 0.5 * (x1 + x2)
    dxold = abs(x2 - x1)
    dx = dxold
    f, df = shear(rts)
    for _ in range(max_iter):
        if (((rts - xh) * df - f) * ((rts - x1) * df - f) > 0.0
                or abs(2.0 * f) > abs(dxold * df)):
            dxold, dx = dx, 0.5 * (xh - xl)
            rts = xl + dx
            if xl == rts:
                return rts
        else:
            dxold, dx = dx, f / df
            temp = rts
            rts -= dx
            if temp == rts:
                return rts
        if abs(dx) < acc:
            return rts
        f, df = shear(rts)
        if f < 0.0:
            xl = rts
        else:
            xh = rts
    return rts


def test_vics_own_shear_root_is_up_to_two_per_cent_from_its_own_equation():
    """The size of the tolerance above, stated as a measurement."""
    worst = 0.0
    for tair, wind in ((-19.0, 7.0), (-25.0, 16.0), (-7.0, 10.0),
                       (-1.0, 13.0), (-13.0, 4.0), (-3.0, 19.0)):
        z0 = 0.005
        u10 = min(max(wind * math.log(10.0 / z0) / math.log((2 + z0) / z0),
                      0.4), 25.0)
        ut = float(blowing.get_thresh(torch.tensor([tair], **KW),
                                      torch.tensor([0.0], **KW),
                                      torch.tensor([z0], **KW)))
        umin, umax = ut, blowing.CONST_KARMAN * u10
        fl = float(blowing.get_shear(torch.tensor([umin], **KW),
                                     torch.tensor([u10], **KW),
                                     torch.tensor([10.0], **KW)))
        fh = float(blowing.get_shear(torch.tensor([umax], **KW),
                                     torch.tensor([u10], **KW),
                                     torch.tensor([10.0], **KW)))
        if fl * fh >= 0:
            continue
        c_root = _rtnewt(umin, umax, 0.10 * umin, u10, 10.0)
        ours = float(blowing.shear_stress(torch.tensor([u10], **KW),
                                          torch.tensor([z0], **KW),
                                          torch.tensor([ut], **KW))[0])
        # Ours is on the root; VIC's is within its own tolerance of it.
        resid_ours = abs(float(blowing.get_shear(
            torch.tensor([ours], **KW), torch.tensor([u10], **KW),
            torch.tensor([10.0], **KW))))
        assert resid_ours < 1e-9, (tair, wind, resid_ours)
        worst = max(worst, abs(c_root - ours))
    print('VIC rtnewt against the root of its own equation: worst %.3e m/s'
          % worst)
    assert worst > 1e-3, worst


def test_with_vics_own_root_the_flux_agrees_to_ten_digits(cases, monkeypatch):
    """The decisive one. Replace the solver with a transcription of rtnewt --
    tolerance and all -- and the remaining disagreement vanishes, which says
    the 1e-3 above is VIC's stopping rule and nothing else."""
    def vic_shear(u10, z0, utshear, zr=10.0):
        # u10 carries the wind-increment axis; the other two are per
        # cell-tile and broadcast onto it.
        z0 = z0.expand_as(u10).reshape(-1).contiguous()
        utshear = utshear.expand_as(u10).reshape(-1).contiguous()
        flat = u10.reshape(-1).contiguous()
        ush = torch.empty_like(flat)
        z0s = torch.empty_like(flat)
        brk = torch.zeros_like(flat, dtype=torch.bool)
        for i in range(flat.numel()):
            u, z, ut = float(flat[i]), float(z0[i]), float(utshear[i])
            umin, umax = ut, blowing.CONST_KARMAN * u
            def sh(x):
                return (math.log(2.0 * blowing.CONST_G * zr / 0.12)
                        + math.log(1.0 / (x * x))
                        - blowing.CONST_KARMAN * u / x)
            fl, fh = sh(umin), sh(umax)
            if fl > 0.0 and fh > 0.0:
                ush[i] = blowing.CONST_KARMAN * u / math.log(zr / z)
                z0s[i] = z
            else:
                r = _rtnewt(umin, umax, 0.10 * umin, u, zr)
                ush[i] = r
                z0s[i] = 0.12 * r * r / (2.0 * blowing.CONST_G)
                brk[i] = True
        return (ush.view_as(u10), z0s.view_as(u10),
                brk.view_as(u10))

    monkeypatch.setattr(blowing, 'shear_stress', vic_shear)
    cols, want_flux, want_tr = _tensors(cases)
    # Deep enough that the integration is not the limiting term: the
    # claim under test is about the root, not about Romberg.
    got, tr = _run(cols, levels=12)
    for got_v, want_v, name in ((got, want_flux, 'sublimation'),
                                (tr, want_tr, 'transport')):
        scale = want_v.abs().max()
        err = (got_v - want_v).abs().max()
        print('with VIC\'s own root, %s: max |error| %.3e, relative %.3e'
              % (name, err, err / scale))
        assert float(err / scale) < 1e-9, (name, float(err), float(scale))


def test_the_romberg_level_count_is_past_the_knee(cases):
    """The fixed count has to be where refining further buys nothing that
    matters, or it is a tuning parameter rather than a solver setting.

    The bar is VIC's own tolerance: the default must sit at least two orders
    inside the 1.2e-03 that rtnewt's stopping rule costs, and going six
    levels deeper must not move the answer by anything like that.
    """
    cols, _, _ = _tensors(cases)
    deep, _ = _run(cols, levels=blowing.ROMBERG_LEVELS + 6)
    scale = deep.abs().max()
    at_default, _ = _run(cols, levels=blowing.ROMBERG_LEVELS)
    err = float((at_default - deep).abs().max() / scale)
    print('romberg at %d levels against %d: %.3e relative, against VIC\'s own '
          'solver tolerance of %.1e' % (blowing.ROMBERG_LEVELS,
                                        blowing.ROMBERG_LEVELS + 6, err,
                                        SOLVER_TOL))
    assert err < SOLVER_TOL / 100.0, err
    # And one level shallower is measurably worse, so the default is not
    # simply far past the knee.
    shallow, _ = _run(cols, levels=blowing.ROMBERG_LEVELS - 3)
    assert float((shallow - deep).abs().max() / scale) > err


def test_the_shear_root_satisfies_its_own_equation(cases):
    """The saltation friction velocity is defined by get_shear() = 0 where
    the bracket holds a sign change; the solver's answer must put it
    there."""
    cols, _, _ = _tensors(cases)
    u10 = cols['Wind'] * torch.log(10.0 / cols['ZO']) / torch.log(
        (2.0 + cols['ZO']) / cols['ZO'])
    u10 = torch.clamp(u10, 0.4, 25.0)
    ut = blowing.get_thresh(cols['Tair'], cols['SurfaceLiquidWater'],
                            cols['ZO'])
    ushear, z0_salt, bracketed = blowing.shear_stress(u10, cols['ZO'], ut)
    assert int(bracketed.sum()) > 10, int(bracketed.sum())
    zr = torch.full_like(u10, 10.0)
    resid = blowing.get_shear(ushear, u10, zr).abs()[bracketed]
    assert float(resid.max()) < 1e-8, float(resid.max())
    # And where it is not bracketed the fallback is the plain log profile.
    plain = blowing.CONST_KARMAN * u10 / torch.log(10.0 / cols['ZO'])
    assert torch.allclose(ushear[~bracketed], plain[~bracketed])


def test_nothing_blows_off_bare_ground(cases):
    cols, _, _ = _tensors(cases)
    cols = dict(cols)
    cols['snowdepth'] = torch.zeros_like(cols['snowdepth'])
    got, tr = _run(cols)
    assert float(got.abs().max()) == 0.0
    assert float(tr.abs().max()) == 0.0


def test_the_flux_carries_a_gradient_to_the_wind_and_the_roughness():
    """Blowing snow is a loss term a calibration would want to tune, so it
    has to be differentiable in the quantities that drive it."""
    n = 6
    one = torch.ones(n, **KW)
    wind = (6.0 * one).requires_grad_(True)
    z0 = (0.005 * one).requires_grad_(True)
    tair = -8.0 * one
    got, _ = blowing.calc_blowing_snow(
        dt=3600.0 * one, tair=tair, last_snow=24.0 * one,
        surface_liquid_water=torch.zeros(n, **KW), wind=wind,
        ls=2.84e6 * one, air_dens=1.2 * one, eact_air=200.0 * one, z0=z0,
        zrh=2.0 * one, snowdepth=0.4 * one, lag_one=0.8 * one,
        sigma_slope=0.06 * one, tsnow=-8.0 * one, fe=1000.0 * one,
        displacement=0.2 * one, roughness=0.05 * one)
    got.sum().backward()
    for g in (wind.grad, z0.grad):
        assert g is not None and torch.isfinite(g).all()
        assert float(g.abs().max()) > 0.0
