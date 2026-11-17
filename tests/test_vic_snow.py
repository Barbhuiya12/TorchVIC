#!/usr/bin/env python3
"""Check torchvic.core.snow against VIC's own snow_utility.c.

The reference is produced by `ref_soil snow`, which calls VIC's
new_snow_density() and snow_albedo() rather than transcribing them, so a
disagreement here is a disagreement with the shipped model and not with a
second reading of the documentation.

Run: tests/test_vic_snow.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths                                   # noqa: E402

import torch                                          # noqa: E402
from torchvic.core import snow, canopy  # noqa: E402

# This project is GPU-native and the differentiable half is no exception.
# Override with TORCHVIC_DEVICE=cpu.
DEVICE = os.environ.get('TORCHVIC_DEVICE') or (
    'cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)


REF = _paths.oracle('ref_soil')
# Both functions are algebraic in double precision, so the only difference
# expected is the order of floating-point operations. 1e-12 relative is far
# tighter than any physical tolerance and still leaves room for that.
RTOL = 1e-12
ATOL = 1e-13


def load_reference():
    if not os.path.exists(REF):
        sys.exit('ref_soil missing; run tests/oracle/build_ref_soil.sh')
    out = subprocess.run([REF, 'snow'], capture_output=True, text=True,
                         check=True).stdout
    rows = {}
    for line in out.splitlines():
        if line.startswith('#') or line.startswith('fn,'):
            continue
        # pack_refreeze rows carry six outputs after a '|' and are parsed
        # separately; skip them here rather than widening this loader.
        if line.split(',', 1)[0] in ('pack_refreeze', 'rebalance_layers',
                                     'surface_melt', 'accumulate_snowfall',
                                     'canopy_intercept',
                                     'canopy_energy_bal'):
            continue
        parts = line.split(',')
        rows.setdefault(parts[0], []).append([float(x) for x in parts[1:]])
    return rows


def check(name, got, want, failures):
    got = got.to(torch.float64)
    want = torch.tensor(want, dtype=torch.float64)
    ok = torch.allclose(got, want, rtol=RTOL, atol=ATOL)
    err = (got - want).abs()
    rel = err / want.abs().clamp(min=1e-30)
    print('  %-28s %4d values  max|abs| %9.3g  max|rel| %9.3g  %s'
          % (name, len(want), err.max().item(), rel.max().item(),
             'PASS' if ok else 'FAIL'))
    if not ok:
        bad = (~torch.isclose(got, want, rtol=RTOL, atol=ATOL)).nonzero()
        for i in bad[:5].flatten().tolist():
            print('      row %d: got %.17g want %.17g' % (i, got[i], want[i]))
        failures.append(name)


def main():
    rows = load_reference()
    failures = []
    print('torchvic.core.snow against VIC snow_utility.c')

    # new_snow_density is deliberately NOT compared here. This script's
    # oracle, ref_soil, compiles pristine VIC 5.1.0, which caps
    # new_snow_density at param.SNOW_NEW_SNOW_DENS_MAX -- a VIC 5.1
    # parameter that 5.0.1 does not have. On the 5.0.1 core, comparing the two
    # here would report the release difference as a defect: 390 kg/m3 and 5.5e6 kg/m3 apart at the warm
    # end, where the cap binds and VIC 5.0.1's formula does not.
    #
    # tests/test_snow_formulas.py compares the same function
    # against ref_snow, which compiles pristine VIC 5.0.1, and states
    # the divergence and what it costs. That is the check; this would only
    # be a second, wrong one. The rest of this script -- snow_albedo, the
    # canopy interception, the surface temperature solve -- uses functions
    # the two trees agree on.

    data = rows['snow_albedo']
    new_snow = torch.tensor([r[0] for r in data], dtype=torch.float64)
    nsa = torch.tensor([r[1] for r in data], dtype=torch.float64)
    swq = torch.tensor([r[2] for r in data], dtype=torch.float64)
    cc = torch.tensor([r[3] for r in data], dtype=torch.float64)
    last = torch.tensor([r[4] for r in data], dtype=torch.float64)
    melting = torch.tensor([bool(r[5]) for r in data])
    want = [r[6] for r in data]
    check('snow_albedo', snow.snow_albedo(new_snow, nsa, swq, cc, 3600.0,
                                          last, melting), want, failures)

    # pack_refreeze returns six values per case, packed after a '|'
    rows_pr = []
    for line in subprocess.run([REF, 'snow'], capture_output=True, text=True,
                               check=True).stdout.splitlines():
        if line.startswith('pack_refreeze,'):
            ins, outs = line.split(',', 1)[1].split('|', 1)
            rows_pr.append(([float(x) for x in ins.split(',')],
                            [float(x) for x in outs.split('|')]))
    ins = lambda i: torch.tensor([r[0][i] for r in rows_pr],
                                 dtype=torch.float64)
    got = snow.pack_refreeze(ins(0), ins(1), ins(2), ins(3), ins(4), ins(5))
    names = ['surf_water', 'pack_water', 'pack_swq', 'ice_gain', 'pack_temp',
             'melt']
    for j, nm in enumerate(names):
        check('pack_refreeze.%s' % nm, got[j],
              [r[1][j] for r in rows_pr], failures)

    rows_ce = []
    for line in subprocess.run([REF, 'snow'], capture_output=True, text=True,
                               check=True).stdout.splitlines():
        if line.startswith('canopy_energy_bal,'):
            ins, outs = line.split(',', 1)[1].split('|', 1)
            rows_ce.append(([float(x) for x in ins.split(',')],
                            [float(x) for x in outs.split('|')]))
    ce = lambda i: torch.tensor([r[0][i] for r in rows_ce],
                                dtype=torch.float64)
    m = len(rows_ce)
    k = lambda v: torch.full((m,), v, dtype=torch.float64)
    st_ce = dict(long_over_in=k(260.0), long_under_out=k(300.0),
                 net_short_over=k(70.0), air_dens=k(1.2), eact_air=k(400.0),
                 press=k(101300.0), ra=ce(2), t_canopy=ce(1),
                 rainfall=ce(4), int_rain=ce(3), vpd=k(200.0), dt=k(3600.0))
    rest, refr = canopy.canopy_energy_balance(ce(0), st_ce,
                                              return_refreeze=True)
    check('canopy_energy_bal.residual', rest, [r[1][0] for r in rows_ce],
          failures)
    check('canopy_energy_bal.refreeze', refr, [r[1][1] for r in rows_ce],
          failures)

    rows_ci = []
    for line in subprocess.run([REF, 'snow'], capture_output=True, text=True,
                               check=True).stdout.splitlines():
        if line.startswith('canopy_intercept,'):
            ins, outs = line.split(',', 1)[1].split('|', 1)
            rows_ci.append(([float(x) for x in ins.split(',')],
                            [float(x) for x in outs.split('|')]))
    ci = lambda i: torch.tensor([r[0][i] for r in rows_ci], dtype=torch.float64)
    m = len(rows_ci)
    cc = lambda v: torch.full((m,), v, dtype=torch.float64)
    got = canopy.intercept(ci(0), cc(4.0), ci(1), ci(2), ci(3), ci(4), ci(5),
                           cc(0.8), cc(0.0006))
    for j, nm in enumerate(['int_snow', 'int_rain', 'snow_throughfall',
                            'rain_throughfall', 'blown_snow']):
        check('canopy_intercept.%s' % nm, got[j],
              [r[1][j] for r in rows_ci], failures)

    rows_ac = []
    for line in subprocess.run([REF, 'snow'], capture_output=True, text=True,
                               check=True).stdout.splitlines():
        if line.startswith('accumulate_snowfall,'):
            ins, outs = line.split(',', 1)[1].split('|', 1)
            rows_ac.append(([float(x) for x in ins.split(',')],
                            [float(x) for x in outs.split('|')]))
    ac = lambda i: torch.tensor([r[0][i] for r in rows_ac],
                                dtype=torch.float64)
    g = snow.accumulate_snowfall(ac(0), ac(1), ac(2), ac(3), ac(4))
    for j, nm in enumerate(['surface_swq', 'pack_swq', 'surf_temp',
                            'pack_temp']):
        check('accumulate_snowfall.%s' % nm, g[j],
              [r[1][j] for r in rows_ac], failures)

    rows_sm = []
    for line in subprocess.run([REF, 'snow'], capture_output=True, text=True,
                               check=True).stdout.splitlines():
        if line.startswith('surface_melt,'):
            ins, outs = line.split(',', 1)[1].split('|', 1)
            rows_sm.append(([float(x) for x in ins.split(',')],
                            [float(x) for x in outs.split('|')]))
    sm = lambda i: torch.tensor([r[0][i] for r in rows_sm],
                                dtype=torch.float64)
    ice_in = sm(2) + sm(3)
    dt_sm = torch.full((len(rows_sm),), 3600.0, dtype=torch.float64)
    g = snow.surface_melt(sm(0), sm(1), sm(2), sm(3), sm(4), ice_in, dt_sm)
    # returned order: melt, surf_water, surface_swq, pack_swq, pack_water,
    #                 ice, melt_energy
    for j, nm in enumerate(['melt', 'surf_water', 'surface_swq', 'pack_swq',
                            'pack_water', 'ice', 'melt_energy']):
        check('surface_melt.%s' % nm, g[j], [r[1][j] for r in rows_sm],
              failures)

    rows_rb = []
    for line in subprocess.run([REF, 'snow'], capture_output=True, text=True,
                               check=True).stdout.splitlines():
        if line.startswith('rebalance_layers,'):
            ins, outs = line.split(',', 1)[1].split('|', 1)
            rows_rb.append(([float(x) for x in ins.split(',')],
                            [float(x) for x in outs.split('|')]))
    rb = lambda i: torch.tensor([r[0][i] for r in rows_rb],
                                dtype=torch.float64)
    pw = torch.full((len(rows_rb),), 0.003, dtype=torch.float64)
    sw = torch.full((len(rows_rb),), 0.001, dtype=torch.float64)
    got = snow.rebalance_layers(rb(0), rb(1), rb(2), rb(3), rb(4), pw, sw)
    for j, nm in enumerate(['surface_swq', 'pack_swq', 'surf_temp',
                            'pack_temp', 'swq']):
        check('rebalance_layers.%s' % nm, got[j],
              [r[1][j] for r in rows_rb], failures)

    data = rows['snow_pack_energy_balance']
    n = len(data)
    ts = torch.tensor([r[0] for r in data], dtype=torch.float64)

    def col(i):
        return torch.tensor([r[i] for r in data], dtype=torch.float64)

    def const(v):
        return torch.full((n,), v, dtype=torch.float64)

    st = dict(z=const(2.0), t_air=col(1), wind=col(2), z0=const(0.01),
              ra=const(60.0), long_snow_in=const(250.0),
              net_short=const(80.0), air_dens=const(1.2),
              eact_air=const(400.0), lv=const(2.5e6), press=const(101300.0),
              vpd=const(200.0), blowing_mass_flux=const(0.0),
              rain=const(0.0), dt=const(3600.0), swe_surface=const(0.05),
              old_t_surf=col(4), snow_depth=col(3), snow_density=const(250.0),
              t_grnd=const(0.0), surface_liquid_water=col(5))
    check('snow_pack_energy_balance',
          snow.snow_pack_energy_balance(ts, st), [r[6] for r in data],
          failures)

    data = rows['svp']
    t = torch.tensor([r[0] for r in data], dtype=torch.float64)
    check('svp', snow.svp(t), [r[6] for r in data], failures)

    data = rows['latent_heat_from_snow']
    dens = torch.tensor([r[0] for r in data], dtype=torch.float64)
    eact = torch.tensor([r[1] for r in data], dtype=torch.float64)
    press = torch.tensor([r[2] for r in data], dtype=torch.float64)
    ra = torch.tensor([r[3] for r in data], dtype=torch.float64)
    tm = torch.tensor([r[4] for r in data], dtype=torch.float64)
    vpd = torch.tensor([r[5] for r in data], dtype=torch.float64)
    lh, lhs, _, _ = snow.latent_heat_from_snow(
        dens, eact, 2.5e6, press, ra, tm, vpd, torch.zeros_like(dens))
    check('latent_heat_from_snow', lh + lhs, [r[6] for r in data], failures)

    data = rows['calc_outgoing_longwave']
    tk = torch.tensor([r[0] for r in data], dtype=torch.float64)
    check('calc_outgoing_longwave',
          snow.calc_outgoing_longwave(tk, 0.97), [r[6] for r in data],
          failures)

    data = rows['calc_sensible_heat']
    dens = torch.tensor([r[0] for r in data], dtype=torch.float64)
    t1 = torch.tensor([r[1] for r in data], dtype=torch.float64)
    t0 = torch.tensor([r[2] for r in data], dtype=torch.float64)
    ra = torch.tensor([r[3] for r in data], dtype=torch.float64)
    check('calc_sensible_heat',
          snow.calc_sensible_heat(dens, t1, t0, ra), [r[6] for r in data],
          failures)

    data = rows['stability_correction']
    z = torch.tensor([r[0] for r in data], dtype=torch.float64)
    ts = torch.tensor([r[1] for r in data], dtype=torch.float64)
    ta = torch.tensor([r[2] for r in data], dtype=torch.float64)
    wd = torch.tensor([r[3] for r in data], dtype=torch.float64)
    z0 = torch.tensor([r[4] for r in data], dtype=torch.float64)
    check('stability_correction',
          snow.stability_correction(z, torch.zeros_like(z), ts, ta, wd, z0),
          [r[6] for r in data], failures)

    data = rows['snow_density_bras']
    depth0 = torch.tensor([r[0] for r in data], dtype=torch.float64)
    sswq = torch.tensor([r[1] for r in data], dtype=torch.float64)
    new_snow = torch.tensor([r[2] for r in data], dtype=torch.float64)
    stemp = torch.tensor([r[3] for r in data], dtype=torch.float64)
    air = torch.tensor([r[4] for r in data], dtype=torch.float64)
    want = [r[6] for r in data]
    check('snow_density_bras',
          snow.snow_density_bras(depth0, sswq, stemp, new_snow, sswq, air,
                                 3600.0),
          want, failures)

    data = rows['snow_density_snthrm']
    dens0 = torch.tensor([r[0] for r in data], dtype=torch.float64)
    depth0 = torch.tensor([r[1] for r in data], dtype=torch.float64)
    sswq = torch.tensor([r[2] for r in data], dtype=torch.float64)
    new_snow = torch.tensor([r[3] for r in data], dtype=torch.float64)
    stemp = torch.tensor([r[4] for r in data], dtype=torch.float64)
    air = torch.tensor([r[5] for r in data], dtype=torch.float64)
    wet = torch.tensor([r[6] for r in data], dtype=torch.float64)
    want = [r[7] for r in data]
    check('snow_density_snthrm',
          snow.snow_density_snthrm(dens0, depth0, stemp, new_snow, sswq,
                                   air, 3600.0, wet * depth0,
                                   torch.zeros_like(depth0)),
          want, failures)

    # Gradients must exist and be finite for the parameters a calibration
    # would actually touch. new_snow_albedo is the one that matters here;
    # last_snow is a discrete counter and is deliberately not differentiated.
    print('\ngradient check')
    t = torch.tensor([-10.0, -2.0, 1.0, 5.0], dtype=torch.float64,
                     requires_grad=True)
    snow.new_snow_density(t).sum().backward()
    finite = bool(torch.isfinite(t.grad).all())
    print('  d(new_snow_density)/d(air_temp)  %s  %s'
          % (t.grad.tolist(), 'finite' if finite else 'NOT FINITE'))
    if not finite:
        failures.append('new_snow_density gradient')

    a = torch.tensor([0.85], dtype=torch.float64, requires_grad=True)
    snow.snow_albedo(torch.tensor([0.0], dtype=torch.float64), a,
                     torch.tensor([1.0], dtype=torch.float64),
                     torch.tensor([-1.0], dtype=torch.float64),
                     3600.0, torch.tensor([5.0], dtype=torch.float64),
                     torch.tensor([False])).sum().backward()
    finite = bool(torch.isfinite(a.grad).all())
    print('  d(snow_albedo)/d(new_snow_albedo) %s  %s'
          % (a.grad.tolist(), 'finite' if finite else 'NOT FINITE'))
    if not finite:
        failures.append('snow_albedo gradient')

    # snow_density feeds calibration through the pack state, so its
    # gradient w.r.t. snow water equivalent is the one that matters.
    q = torch.tensor([0.05, 0.3], dtype=torch.float64, requires_grad=True)
    snow.snow_density_bras(
        torch.tensor([0.2, 0.8], dtype=torch.float64), q,
        torch.tensor([-5.0, -5.0], dtype=torch.float64),
        torch.tensor([5.0, 0.0], dtype=torch.float64), q,
        torch.tensor([-3.0, -3.0], dtype=torch.float64), 3600.0).sum().backward()
    finite = bool(torch.isfinite(q.grad).all())
    print('  d(snow_density)/d(swq)            %s  %s'
          % ([round(v, 6) for v in q.grad.tolist()],
             'finite' if finite else 'NOT FINITE'))
    if not finite:
        failures.append('snow_density gradient')

    # --- solving the balance for surface temperature --------------------
    #
    # The residual above is bit-exact against VIC, so a root of it is a root
    # of VIC's own SnowPackEnergyBalance. That is a stronger statement than
    # matching root_brent's returned value, which is only defined to the
    # solver's tolerance anyway.
    #
    # The residual is monotonic in T -- outgoing longwave, sensible heat and
    # cold-content change all increase with surface temperature -- so the root
    # is unique and which solver finds it does not matter.
    from torchvic.core import implicit  # noqa: E402

    print('\nsurface temperature solve')
    lo = torch.full_like(ts, -80.0)
    hi = torch.full_like(ts, 10.0)
    flo = snow.snow_pack_energy_balance(lo, st)
    fhi = snow.snow_pack_energy_balance(hi, st)
    bracketed = (torch.sign(flo) != torch.sign(fhi))
    print('  cases with a sign change in [-80, 10] C   %d of %d'
          % (int(bracketed.sum()), len(ts)))

    keep = bracketed.nonzero().flatten()
    sub = {k: (v[keep] if torch.is_tensor(v) else v) for k, v in st.items()}
    t_star = implicit.bisect(snow.snow_pack_energy_balance, lo[keep],
                             hi[keep], args=(sub,))
    resid = snow.snow_pack_energy_balance(t_star, sub).abs()
    print('  max |residual| at the solved temperature  %.3g W/m2'
          % resid.max().item())
    if not bool(resid.max() < 1e-8):
        failures.append('surface temperature solve residual')

    # Gradient of the solved temperature w.r.t. incoming longwave, against a
    # central difference of the root itself. This is the differentiability
    # claim for the snow surface: not that the residual differentiates, but
    # that the *solution* does.
    lw = st['long_snow_in'][keep].clone().requires_grad_(True)
    sub_g = dict(sub)
    sub_g['long_snow_in'] = lw
    t_g = implicit.solve(snow.snow_pack_energy_balance, lo[keep], hi[keep],
                         args=(sub_g,))
    t_g.sum().backward()

    eps = 1e-4
    up, dn = dict(sub), dict(sub)
    up['long_snow_in'] = sub['long_snow_in'] + eps
    dn['long_snow_in'] = sub['long_snow_in'] - eps
    fd = ((implicit.bisect(snow.snow_pack_energy_balance, lo[keep], hi[keep],
                           args=(up,)) -
           implicit.bisect(snow.snow_pack_energy_balance, lo[keep], hi[keep],
                           args=(dn,))) / (2 * eps))
    rel = ((lw.grad - fd).abs() / fd.abs().clamp(min=1e-30)).max()
    print('  dT/d(longwave in) vs central difference   max|rel| %.3g'
          % rel.item())
    if not bool(rel < 1e-4):
        failures.append('surface temperature gradient')

    print()
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
