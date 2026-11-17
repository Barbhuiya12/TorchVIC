"""Differentiable PyTorch implementation of VIC's frozen-soil thermal physics.

A reimplementation, not a wrapper, for the same reason as `torchvic.core.soil`
and `torchvic.core.snow`. Verification against the C is the central obligation:
tests/test_vic_frozen.py checks every function here against the
`frozen` mode of tests/oracle/ref_soil.c, which calls VIC's own
soil_conduction.c.

Scope. This module covers the **closed-form** thermal properties that the
frozen-soil solver is built from: soil thermal conductivity, volumetric heat
capacity, and the unfrozen-water curve. It does not cover
`calc_soil_thermal_fluxes` or `newt_raph`, which iterate.

A note on the iterative part, recorded here because it changes what is worth
building. VIC's implicit frozen-soil solver was measured
solver failing to converge on 17.58% of calls, with one cell failing 23,035
times out of 23,035, and this is a property of upstream VIC present
identically on CPU and GPU. A differentiable version of a solver that does
not converge would inherit that: the implicit function theorem gives
dT*/dp = -(df/dT)^-1 (df/dp) at a root, and there is no root at a
non-converged point. Any gradient taken there would be meaningless rather
than merely inaccurate.

The explicit solver, `solve_T_profile`, is direct and has no such problem --
and runs the same case 583x faster. It is the sensible target for a
differentiable frozen soil, and the implicit path should not be attempted
until its convergence is understood.

Shapes: batched over a leading cell axis, matching the other modules.
"""
import torch

from . import xmath

CONST_LATICE = 3.337e5
CONST_TKTRIP = 273.16
CONST_G = 9.80616
CM_PER_M = 100.0


def _pow(a, b):
    """`a ** b`, the closed forms' power (soiltemp._rowcall batches it);
    glibc pow in exact-math mode (xmath.py), torch.pow otherwise."""
    return xmath.pow(a, b)


def maximum_unfrozen_water(t, max_moist, bubble, expt, pow=None,
                           expo=None):
    """Unfrozen water content at sub-freezing temperature.
    soil_conduction.c:788.

    The Clapp-Hornberger freezing-point depression curve: below 0 C only part
    of the soil water can freeze, set by the matric potential the remaining
    film can sustain.

    Above freezing all water is unfrozen, which the C returns without
    evaluating the power. Here both sides are evaluated and selected, so the
    power's argument is guarded to stay positive -- at T = 0 it is zero, and
    the exponent is negative, so an unguarded evaluation is a division by zero
    whose NaN would propagate through the backward pass even though `where`
    discards the value.

    `expo`, if given, is the exponent -(2 / (expt - 3)) already formed
    (soiltemp's implicit solver holds it across its Newton trials).
    """
    pow = pow or _pow
    tiny = torch.finfo(t.dtype).tiny
    arg = ((-CONST_LATICE * t) / CONST_TKTRIP /
           (CONST_G * bubble / CM_PER_M))
    if expo is None:
        expo = -(2.0 / (expt - 3.0))
    unfrozen = max_moist * pow(torch.clamp(arg, min=tiny), expo)
    unfrozen = torch.clamp(unfrozen, max=max_moist)
    unfrozen = torch.clamp(unfrozen, min=torch.zeros_like(unfrozen))
    return torch.where(t < 0.0, unfrozen, max_moist)


def volumetric_heat_capacity(soil_fract, water_fract, ice_fract,
                             organic_fract, solid=None):
    """Volumetric heat capacity of the soil, J/m3/K.  soil_conduction.c:90.

    A volume-weighted sum over mineral soil, organic soil, water, ice and air.
    The constants are VIC's and are written in the same order so the
    accumulation rounds identically.

    `solid`, if given, is the first two terms' sum already formed
    (volumetric_heat_capacity_solid), which a caller evaluating many
    water/ice states of one soil can hold.
    """
    cs = (volumetric_heat_capacity_solid(soil_fract, organic_fract)
          if solid is None else solid)
    cs = cs + 4.2e6 * water_fract
    cs = cs + 1.9e6 * ice_fract
    cs = cs + 1.3e3 * (1.0 - (soil_fract + water_fract + ice_fract))
    return cs


def volumetric_heat_capacity_solid(soil_fract, organic_fract):
    """volumetric_heat_capacity's mineral and organic terms."""
    cs = 2.0e6 * soil_fract * (1 - organic_fract)
    return cs + 2.7e6 * soil_fract * organic_fract


def soil_conductivity(moist, wu, soil_dens_min, bulk_dens_min, quartz,
                      soil_density, bulk_density, organic, pow=None,
                      log10=torch.log10):
    """Soil thermal conductivity, W/m/K.  soil_conduction.c:17.

    Johansen's method: a dry conductivity and a saturated conductivity, with
    a Kersten number interpolating between them by degree of saturation.

    `wu` is the unfrozen water content. The C branches on `Wu == moist`,
    exact floating-point equality, to decide whether any ice is present; that
    comparison is reproduced exactly rather than replaced by a tolerance,
    because widening it would change which saturated conductivity is used.

    `pow` and `log10` default to `**` and torch.log10; soiltemp's batched
    implicit solver passes row-wise versions (soiltemp._rowcall) so a
    batch of nodes rounds as the per-node calls do. The work is split
    in two -- what does not depend on `wu`, then what does -- so that
    solver can hold the first part across its Newton trials; the two
    together are the one expression the C evaluates.
    """
    fixed = soil_conductivity_fixed(moist, soil_dens_min, bulk_dens_min,
                                    quartz, soil_density, bulk_density,
                                    organic, pow=pow, log10=log10)
    return soil_conductivity_wu(fixed, moist, wu, pow=pow)


def soil_conductivity_fixed(moist, soil_dens_min, bulk_dens_min, quartz,
                            soil_density, bulk_density, organic, pow=None,
                            log10=torch.log10):
    """soil_conductivity's terms that do not depend on the unfrozen
    water: (kdry, porosity, sr, ksat_u, ke_u, ks ** (1 - porosity))."""
    pow = pow or _pow
    kw = 0.57
    kdry_org = 0.05
    ks_org = 0.25

    kdry_min = ((0.135 * bulk_dens_min + 64.7) /
                (soil_dens_min - 0.947 * bulk_dens_min))
    kdry = (1 - organic) * kdry_min + organic * kdry_org

    porosity = 1.0 - bulk_density / soil_density
    safe_por = torch.where(porosity != 0.0, porosity,
                           torch.ones_like(porosity))
    sr = moist / safe_por

    ks_min = torch.where(
        quartz < 0.2,
        pow(7.7, quartz) * pow(3.0, 1.0 - quartz),
        pow(7.7, quartz) * pow(2.2, 1.0 - quartz))
    ks = (1 - organic) * ks_min + organic * ks_org

    # log10(Sr) is -inf at Sr = 0; that branch is only selected where moist
    # is positive, but the unselected value must stay finite for the gradient.
    safe_sr = torch.clamp(sr, min=torch.finfo(sr.dtype).tiny)
    ks_por = pow(ks, 1.0 - porosity)
    ksat_u = ks_por * pow(kw, porosity)
    ke_u = 0.7 * log10(safe_sr) + 1.0
    return kdry, porosity, sr, ksat_u, ke_u, ks_por


def soil_conductivity_wu(fixed, moist, wu, pow=None):
    """soil_conductivity from its fixed terms and the unfrozen water."""
    pow = pow or _pow
    ki = 2.2
    kw = 0.57
    kdry, porosity, sr, ksat_u, ke_u, ks_por = fixed
    unfrozen = wu == moist
    ksat_f = ks_por * pow(ki, porosity - wu) * pow(kw, wu)
    ke_f = sr

    ksat = torch.where(unfrozen, ksat_u, ksat_f)
    ke = torch.where(unfrozen, ke_u, ke_f)
    k = (ksat - kdry) * ke + kdry
    k = torch.maximum(k, kdry)
    return torch.where(moist > 0.0, k, kdry)

CONST_RHOICE = 0.917e3


def thermal_coefficients(cs, kappa, zsum, alpha, beta, gamma, deltat, dp,
                         nnodes, exp_trans=True, noflux=False):
    """Per-node coefficients of the soil heat equation.  frozen_soil.c:125.

    Returns (A, B, C, D, E), each shaped [cell, nnodes]. These are the terms
    `calc_soil_thermal_fluxes` iterates on, and the C computes them once per
    solve and caches them -- they were `threadprivate` statics before the GPU
    port moved them into a caller-owned context (GH #735).

    `exp_trans` selects VIC's EXP_TRANS grid transformation, which spaces the
    thermal nodes logarithmically with depth rather than uniformly. It is the
    configuration the frozen-soil tests use. With it off, the coefficients are
    formed from the node spacings alpha/beta/gamma instead.

    Interior nodes 1..nnodes-2 are always filled. Node nnodes-1 is filled
    only when `noflux` is set, in either branch, which is the C's behaviour
    (:199-207 and :218-226); node 0 is left at zero, because the surface
    temperature is a boundary condition rather than a solved node.

    Shapes: `cs` and `kappa` are [cell, nnodes]; `zsum` is [cell, nnodes];
    `alpha`, `beta`, `gamma` are [cell, nnodes-1] and are indexed j-1 as in
    the C.
    """
    shape = cs.shape
    a = torch.zeros(shape, dtype=cs.dtype, device=cs.device)
    b = torch.zeros_like(a)
    c = torch.zeros_like(a)
    d = torch.zeros_like(a)
    e = torch.zeros_like(a)

    j = slice(1, nnodes - 1)
    jm = slice(0, nnodes - 2)      # alpha[j-1] for j in 1..nnodes-2

    if exp_trans:
        # The C computes this as `logf(Dp + 1.) / (double)(Nnodes - 1)`.
        # logf is the *single-precision* natural log, in an otherwise
        # double-precision routine, and its result is only then promoted for
        # the division. Computing it in double instead disagrees with VIC by
        # 3.7e-08 relative -- small, but exactly the size that says a float
        # crept into a double calculation, and it propagates into every
        # EXP_TRANS coefficient.
        # `device=` is not decoration: torch.as_tensor without it lands on
        # the *default* device, which a caller working on the other one
        # then cannot multiply by.
        bexp = torch.log(
            torch.as_tensor(dp + 1.0, dtype=torch.float32,
                            device=cs.device)
        ).to(cs.dtype) / (nnodes - 1)
        # Per column ([...]) against per-node slices ([..., node]): the
        # trailing axis keeps the tile count from aligning with the
        # node count (with NODES 3 the slice is 1 wide and the bug is
        # silent; with 12 it errors).
        bexn = bexp[..., None]
        zp1 = zsum[..., j] + 1
        a[..., j] = 4 * bexn * bexn * cs[..., j] * zp1 * zp1
        b[..., j] = (kappa[..., 2:nnodes] - kappa[..., 0:nnodes - 2]) * deltat
        c[..., j] = 4 * deltat * kappa[..., j]
        d[..., j] = 2 * deltat * kappa[..., j] * bexn
        e[..., j] = (4 * bexn * bexn * CONST_RHOICE * CONST_LATICE *
                     zp1 * zp1)

        if noflux:
            # frozen_soil.c:218-226. The C fills the bottom node in BOTH
            # branches; this one was missing it, so with EXP_TRANS and
            # NOFLUX together the bottom node's A and C stayed zero and the
            # update below divides by A + 2C. It was unreachable only
            # because nothing called the pair.
            k = nnodes - 1
            zk = zsum[..., k] + 1
            a[..., k] = 4 * bexp * bexp * cs[..., k] * zk * zk
            b[..., k] = (kappa[..., k] - kappa[..., k - 1]) * deltat
            c[..., k] = 4 * deltat * kappa[..., k]
            d[..., k] = 2 * deltat * kappa[..., k] * bexp
            e[..., k] = (4 * bexp * bexp * CONST_RHOICE * CONST_LATICE
                         * zk * zk)
    else:
        al = alpha[..., jm]
        a[..., j] = cs[..., j] * al * al
        b[..., j] = (kappa[..., 2:nnodes] - kappa[..., 0:nnodes - 2]) * deltat
        c[..., j] = 2 * deltat * kappa[..., j] * al / gamma[..., jm]
        d[..., j] = 2 * deltat * kappa[..., j] * al / beta[..., jm]
        e[..., j] = CONST_RHOICE * CONST_LATICE * al * al

        if noflux:
            k = nnodes - 1
            alk = alpha[..., k - 1]
            a[..., k] = cs[..., k] * alk * alk
            b[..., k] = (kappa[..., k] - kappa[..., k - 1]) * deltat
            c[..., k] = 2 * deltat * kappa[..., k] * alk / gamma[..., k - 1]
            d[..., k] = 2 * deltat * kappa[..., k] * alk / beta[..., k - 1]
            e[..., k] = CONST_RHOICE * CONST_LATICE * alk * alk

    return a, b, c, d, e


def node_update_unfrozen(t_j, t_jp1, t_jm1, t0_j, ice_j, a, b, c, d, e,
                         exp_trans=True, noflux_node=False):
    """One Gauss-Seidel update of an unfrozen thermal node.
    frozen_soil.c:389 (interior) and :459 (NOFLUX bottom node).

    The C takes this branch when the node is at or above freezing, or when
    frozen soil is inactive. Frozen nodes instead solve `soil_thermal_eqn`
    with root_brent, which is a separate path.

    `noflux_node` selects the bottom-boundary form, which uses T[j] in place
    of T[j+1] because there is no node below it.

    Returns the updated temperature. The denominators differ between the two
    grid options -- (A + C + D) for the uniform grid, (A + 2C) for the
    transformed one -- and both are the C's.
    """
    upper = t_j if noflux_node else t_jp1

    if exp_trans:
        num = (a * t0_j +
               b * (upper - t_jm1) +
               c * (upper + t_jm1) -
               d * (upper - t_jm1) +
               e * (0.0 - ice_j))
        den = a + 2.0 * c
    else:
        num = (a * t0_j +
               b * (upper - t_jm1) +
               c * upper +
               d * t_jm1 +
               e * (0.0 - ice_j))
        den = a + c + d

    safe_den = torch.where(den != 0.0, den, torch.ones_like(den))
    return num / safe_den


def soil_thermal_eqn(t, st):
    """Residual of the soil heat equation at a frozen node.
    soil_thermal_eqn.c.

    This is what VIC drives to zero with root_brent when a thermal node is
    below freezing and frozen soil is active. `st` is a dict holding the
    node's neighbours and coefficients, fixed while the root is sought:
    tl, tu, t0, moist, max_moist, bubble, expt, ice0, a, b, c, d, e, node.

    The ice content is a function of T through the unfrozen-water curve, which
    is what makes this nonlinear and is why it needs a root find rather than
    the closed-form update used for unfrozen nodes.

    The "cold nose" correction is the C's, comment included in spirit: at the
    near-surface node only, when the neighbours differ by more than 5 K and
    the node is colder than both, a diffusion term that would drive it colder
    still is dropped. VIC calls it an inelegant fix; it is reproduced because
    removing it changes results, not because it is defensible.
    """
    ice = torch.clamp(
        st['moist'] - maximum_unfrozen_water(t, st['max_moist'],
                                             st['bubble'], st['expt']),
        min=0.0)
    ice = torch.minimum(ice, st['max_moist'])
    ice = torch.where(t < 0.0, ice, torch.zeros_like(ice))

    tl, tu, t0 = st['tl'], st['tu'], st['t0']
    a, b, c, d, e = st['a'], st['b'], st['c'], st['d'], st['e']
    dice = e * (ice - st['ice0'])

    if st.get('exp_trans', True):
        base = (-a * (t - t0) + b * (tl - tu) + c * (tl - 2.0 * t + tu) -
                d * (tl - tu) + dice)
        no_flux1 = (-a * (t - t0) + c * (tl - 2.0 * t + tu) -
                    d * (tl - tu) + dice)
        flux1 = b * (tl - tu)
        flux2 = c * (tl - 2.0 * t + tu) - d * (tl - tu)
    else:
        base = (-a * (t - t0) + b * (tl - tu) + c * (tl - t) -
                d * (t - tu) + dice)
        no_flux1 = -a * (t - t0) + c * (tl - t) - d * (t - tu) + dice
        flux1 = b * (tl - tu)
        flux2 = c * (tl - t) - d * (t - tu)

    cold_nose = ((st['node'] == 1) &
                 (torch.abs(tl - tu) > 5.0) & (t < tl) & (t < tu) &
                 (flux1 < 0) & (flux2 > 0) &
                 (torch.abs(flux1) > torch.abs(flux2)))
    return torch.where(cold_nose, no_flux1, base)


def solve_frozen_node(st, t0, soil_dt=0.25, tol=1e-12):
    """Solve the heat equation at a frozen node, differentiably.

    Brackets as VIC does -- `root_brent` is called with
    [T0 - SOIL_DT, T0 + SOIL_DT] at frozen_soil.c:409 -- and solves with
    `torchvic.core.implicit`, so gradients come from the implicit function
    theorem rather than from unrolling the solver.

    Where the bracket does not contain a sign change the C's root_brent
    widens it; here the caller is told instead, by returning the bracketing
    mask alongside the root, because silently returning a non-root and
    differentiating it is the failure mode this module's docstring warns
    about. A gradient at a non-converged point is meaningless, not
    approximate.

    `calc_soil_thermal_fluxes` DOES widen, because that is what VIC's own
    caller gets; it uses `implicit.widen_bracket` directly rather than this
    function, so the stricter contract here is not quietly relaxed for
    everyone who calls it. The widening is not free of consequence:
    `soil_thermal_eqn` is discontinuous where the ice term switches on at
    0 C and where the cold-nose correction changes branch, so a bracket
    widened by 10 C at a time can straddle a jump and the sign change it
    finds is the jump rather than a root.

    Returns (t, bracketed).
    """
    from . import implicit

    # param.SOIL_DT is 0.25, not 1.0 (initialize_parameters.c:203). It is the
    # half-width of the bracket root_brent is given at frozen_soil.c:420 and
    # :471, so a wider one is not merely slower: it changes which root a
    # bisection converges to when the residual has more than one crossing.
    lo = t0 - soil_dt
    hi = t0 + soil_dt
    flo = soil_thermal_eqn(lo, st)
    fhi = soil_thermal_eqn(hi, st)
    bracketed = torch.sign(flo) != torch.sign(fhi)

    root = implicit.solve(soil_thermal_eqn, lo, hi, args=(st,), tol=tol)
    return root, bracketed


#: initialize_parameters.c:203 and :212.
SOIL_DT = 0.25
FROZEN_MAXITER = 1000
#: calc_soil_thermal_fluxes' own convergence threshold, frozen_soil.c:373.
T_PROFILE_THRESHOLD = 1.0e-2
#: sweeps the eager calc_soil_thermal_fluxes took to converge every column
#: (sweep count -> calls), for choosing GRAPH_SWEEPS
SWEEP_ITERS = {}
#: sweeps it runs on the CUDA graph path (graphsafe.ACTIVE), every one: a
#: sweep after a column converged leaves it as it is, and a column still
#: going after them is recorded (graphsafe.need, bit SWEEP_BIT) and its
#: day run again eagerly. The ISC boxes converge in 2-3 sweeps.
#: TORCHVIC_GRAPH_SWEEPS sets it.
import os as _os
GRAPH_SWEEPS = int(_os.environ.get('TORCHVIC_GRAPH_SWEEPS', '6'))
#: graphsafe.need's bit for it (basin's own run to 5)
SWEEP_BIT = 6


def calc_soil_thermal_fluxes(t0, moist, max_moist, ice, bubble, expt,
                             a, b, c, d, e, nnodes, fs_active=True,
                             noflux=False, exp_trans=True, frozen_soil=True,
                             tfallback=True, max_iter=FROZEN_MAXITER,
                             soil_dt=SOIL_DT):
    """The soil temperature profile, solved node by node.
    frozen_soil.c:346-560.  Returns (T, fbflag, converged).

    A Gauss-Seidel sweep: each interior node is updated from its two
    neighbours, using the value the node ABOVE it took in this same sweep,
    and the sweep repeats until no node moves by more than 1e-2 C. The
    surface node and, unless NOFLUX, the bottom node are boundary
    conditions and are not solved.

    An unfrozen node has a closed form -- the heat equation is linear in T
    there -- and a frozen one does not, because the ice content is itself a
    function of T through the unfrozen-water curve; that node takes VIC's
    own root_brent sequence (implicit.root_brent) after VIC's own
    widening, with the implicit-function-theorem correction carrying
    the gradient. Which of the two a node takes is
    decided on its CURRENT temperature at each sweep, so a node can change
    branch between sweeps as it crosses freezing.

    Three things the C does that are reproduced rather than tidied:

    * `maxdiff` is initialised to the threshold at the top of each sweep and
      only ever raised (:375, :444), so `maxdiff <= threshold` means no node
      moved by more than the threshold -- the test is on the largest move,
      not on a norm.
    * the "runaway cold nose" fix (:512-530): a node that was colder than
      both its neighbours before the solve and has grown colder relative to
      both is replaced by their mean. The C calls it a crude fix and it is;
      it changes answers, so it is here. It runs only with TFALLBACK, VIC's
      default.
    * non-convergence is not an approximation to the answer: with TFALLBACK
      the WHOLE profile reverts to the previous step's (:534-540) and every
      node is flagged. Per column, because in a batch some converge.

    `Tfbflag` counts as the C counts -- zeroed at entry (:389-392), so it is
    per call rather than cumulative, whatever the name suggests.

    Shapes: everything is [..., nnodes] except the scalars; `a`..`e` come
    from `thermal_coefficients`. `nnodes` is a Python integer because the
    sweep is a loop over nodes.
    """
    # The sweep walks the nodes in order and each node reads the value the
    # one above it took in the SAME sweep, so the profile is held as a list
    # of per-column tensors rather than one [..., node] tensor: assigning
    # into a slice of a tensor that carries a gradient forces a copy of the
    # whole profile at every node of every sweep, and there can be a
    # thousand sweeps.
    from . import graphsafe, implicit

    batch = t0.shape[:-1]
    # the CUDA graph path: a fixed count of sweeps (GRAPH_SWEEPS), and the
    # frozen-node branch only where frozen soil can be on at all
    gs = graphsafe.ACTIVE
    if gs and not (torch.is_tensor(frozen_soil) or frozen_soil):
        frozen_off = True
    else:
        frozen_off = False
    if gs and not frozen_off:
        raise RuntimeError('the frozen-node solve of the explicit profile '
                           'sweep is not on the CUDA graph path')

    def _bb(v):
        # a Python bool as a fill (no copy from the host)
        t = (v if torch.is_tensor(v)
             else torch.full((), bool(v), dtype=torch.bool,
                             device=t0.device))
        t = t.to(torch.bool)
        while t.dim() < len(batch):
            t = t.unsqueeze(-1)
        return t.expand(batch)

    _fa = _bb(fs_active)
    _fz = _bb(frozen_soil)
    cols = [t0[..., j] for j in range(nnodes)]
    flags = [torch.zeros_like(t0[..., 0], dtype=torch.bool)
             for _ in range(nnodes)]
    active = torch.ones_like(t0[..., 0], dtype=torch.bool)
    converged = torch.zeros_like(active)

    last = nnodes - 1
    solved = list(range(1, last)) + ([last] if noflux else [])

    for _sweep in range(min(max_iter, GRAPH_SWEEPS) if gs else max_iter):
        if not gs and not bool(active.any()):
            SWEEP_ITERS[_sweep] = SWEEP_ITERS.get(_sweep, 0) + 1
            break
        moved = torch.zeros_like(active)
        for j in solved:
            # :460-468 -- with NOFLUX the bottom node has no node below it,
            # so the C uses the node itself in place of T[j+1].
            old = cols[j]
            upper = cols[j + 1] if j < last else old
            lower = cols[j - 1]
            aj, bj, cj, dj, ej = (a[..., j], b[..., j], c[..., j],
                                  d[..., j], e[..., j])
            fusion = ej * (0.0 - ice[..., j])
            if exp_trans:
                num = (aj * t0[..., j] + bj * (upper - lower)
                       + cj * (upper + lower) - dj * (upper - lower) + fusion)
                den = aj + 2.0 * cj
            else:
                num = (aj * t0[..., j] + bj * (upper - lower)
                       + cj * upper + dj * lower + fusion)
                den = aj + cj + dj
            safe_den = torch.where(den != 0.0, den, torch.ones_like(den))
            t_explicit = torch.where(den != 0.0, num / safe_den, old)

            # :401 -- the branch is decided on the node's CURRENT value, so
            # a node can change branch between sweeps as it crosses zero.
            unfrozen = (old >= 0.0) | ~_fa | ~_fz
            if not frozen_off and bool((~unfrozen).any()):
                st = dict(tl=upper, tu=lower, t0=t0[..., j],
                          moist=moist[..., j], max_moist=max_moist[..., j],
                          bubble=bubble[..., j], expt=expt[..., j],
                          ice0=ice[..., j], a=aj, b=bj, c=cj, d=dj, e=ej,
                          exp_trans=exp_trans,
                          node=torch.full_like(old, float(j)))
                # :409 opens with [T0 - SOIL_DT, T0 + SOIL_DT] and
                # root_brent widens it from there (root_brent.c:168-179),
                # so the widening is here rather than in
                # `solve_frozen_node`, whose stricter contract stands.
                # The root itself is VIC's own Brent sequence
                # (implicit.root_brent), not an Illinois solve: a root
                # found tighter than VIC's sits up to ROOT_BRENT_T away
                # from VIC's, and the sweep iterates on it. Gradients
                # come from the implicit-function-theorem correction,
                # whose value is added and taken away again.
                lo, hi, failed = implicit.widen_bracket(
                    soil_thermal_eqn, t0[..., j] - soil_dt,
                    t0[..., j] + soil_dt, args=(st,))
                flo = soil_thermal_eqn(lo, st)
                fhi = soil_thermal_eqn(hi, st)
                with torch.no_grad():
                    r0 = implicit.root_brent(soil_thermal_eqn, lo, hi,
                                             flo, fhi, args=(st,))
                if torch.is_grad_enabled():
                    rg = implicit.root_with_grad(soil_thermal_eqn, r0,
                                                 args=(st,))
                    t_root = r0 + (rg - rg.detach())
                else:
                    t_root = r0
                bracketed = ~failed
                # :422-431 -- TFALLBACK keeps the previous step's value
                # where the root find failed, and flags the node.
                t_root = torch.where(bracketed, t_root, t0[..., j])
                flags[j] = flags[j] | (active & ~bracketed & ~unfrozen)
            else:
                t_root = t_explicit
            new = torch.where(active, torch.where(unfrozen, t_explicit,
                                                  t_root), old)
            cols[j] = new
            moved = moved | (active
                             & ((old - new).abs() > T_PROFILE_THRESHOLD))

        done_now = active & ~moved
        converged = converged | done_now
        active = active & ~done_now
    if gs and max_iter > GRAPH_SWEEPS:
        # a column the graph's sweeps did not converge: the eager loop
        # would have gone on
        graphsafe.need(active, SWEEP_BIT)

    if tfallback:
        # :512-530, the cold nose. Every interior node, on the profile the
        # sweeps left, against the profile they started from.
        for j in range(1, last):
            dl0 = t0[..., j - 1] - t0[..., j]
            du0 = t0[..., j + 1] - t0[..., j]
            dl = cols[j - 1] - cols[j]
            du = cols[j + 1] - cols[j]
            nose = (((dl0 > 0) & (du0 > 0) & (dl - dl0 > 0) & (du - du0 > 0))
                    | ((dl0 < 0) & (du0 < 0) & (dl - dl0 < 0)
                       & (du - du0 < 0)))
            cols[j] = torch.where(nose, 0.5 * (cols[j - 1] + cols[j + 1]),
                                  cols[j])
            flags[j] = flags[j] | nose

    t = torch.stack(cols, -1)
    fbflag = torch.stack(flags, -1)
    if tfallback:
        # :534-540 -- a column that never converged goes back where it came
        # from, whole.
        failed = ~converged
        t = torch.where(failed.unsqueeze(-1), t0, t)
        fbflag = fbflag | failed.unsqueeze(-1)
    return t, fbflag, converged


def frost_fractions(n_frost):
    """The frost sub-area fractions.  vic_init.c:711-725.

    One area holds the whole cell; two split it evenly; more share
    1 / (Nfrost - 1) with the two end areas halved. Returns a list of
    floats in VIC's own arithmetic order.
    """
    if n_frost == 1:
        return [1.0]
    if n_frost == 2:
        return [0.5, 0.5]
    out = []
    for k in range(n_frost):
        f = 1.0 / (n_frost - 1)
        if k == 0 or k == n_frost - 1:
            f /= 2.0
        out.append(f)
    return out


def layer_ice_quick_flux(moist, max_moist, bubble, expt, t_layer,
                         frost_slope, n_frost=1, fs_active=None):
    """Layer ice per frost sub-area.  soil_conduction.c:636-688.

    The layer temperature is spread over a uniform distribution of width
    `frost_slope` (SPATIAL_FROST's soil-file input, C), each sub-area
    reads the temperature at its cumulative mid-fraction
    (linear_interp.c: `((x - lx) / (ux - lx) * (uy - ly) + ly)`), and its
    ice is that area's fraction of what the unfrozen-water curve will
    not hold there, clamped into [0, moist].

    With one frost area (or a zero slope) every area reads the layer
    temperature itself and this is the one line per layer the QUICK_FLUX
    path is otherwise written as. `fs_active`, where given, is broadcast
    against the result and zeroes the ice where frozen soil is off.

    Shapes: the moist inputs broadcast as usual; the result appends the
    frost axis, [..., n_frost]. `frost_slope` None is a zero slope: every
    area reads the layer temperature itself.
    """
    if frost_slope is None:
        frost_slope = 0.0
    fr = frost_fractions(n_frost)
    # The cumulative mid-fractions (:645-652): half the first area, then
    # half of each boundary crossed.
    mids = []
    acc = 0.0
    for f, w in enumerate(fr):
        acc = acc + (w / 2.0 if f == 0 else (fr[f - 1] / 2.0 + w / 2.0))
        mids.append(acc)
    # a Python number made a device tensor once (devconst: no host copy
    # inside a CUDA graph capture)
    from .devconst import device_constant
    slope = (device_constant(frost_slope, t_layer.dtype, t_layer.device)
             if not torch.is_tensor(frost_slope) else frost_slope.to(
                 dtype=t_layer.dtype, device=t_layer.device))
    while slope.dim() < t_layer.dim():
        slope = slope.unsqueeze(-1)
    tmin = t_layer - slope / 2.0
    out = []
    for w, mid in zip(fr, mids):
        # linear_interp(mid, 0, 1, tmin, tmin + slope), in the C's order.
        tmp_t = (mid - 0.0) / (1.0 - 0.0) * ((tmin + slope) - tmin) + tmin
        tmp_ice = moist - maximum_unfrozen_water(tmp_t, max_moist, bubble,
                                                 expt)
        ice = w * tmp_ice
        ice = torch.where(ice < 0.0, torch.zeros_like(ice), ice)
        ice = torch.where(ice > moist, moist, ice)
        out.append(ice)
    ice = torch.stack(out, dim=-1)
    if fs_active is not None:
        act = (torch.as_tensor(fs_active, dtype=torch.bool,
                               device=ice.device)
               if not torch.is_tensor(fs_active) else fs_active)
        while act.dim() < ice.dim():
            act = act.unsqueeze(-1)
        ice = torch.where(act.expand_as(ice), ice,
                          torch.zeros_like(ice))
    return ice


def solve_T_profile(t0, zsum, kappa, cs, moist, max_moist, bubble, expt, ice,
                    alpha, beta, gamma, deltat, dp, nnodes, fs_active=True,
                    noflux=False, exp_trans=True, frozen_soil=True,
                    tfallback=True):
    """The soil temperature profile for one step.  frozen_soil.c:125-246.
    Returns (T, fbflag, converged).

    `thermal_coefficients` for the coefficients, then
    `calc_soil_thermal_fluxes` for the sweep. The C computes the
    coefficients once per solve and caches them in what used to be
    thread-private statics; here they are just recomputed, because the cache
    existed to avoid recomputing them inside the iteration and the
    iteration does not recompute them either way.

    This is VIC's EXPLICIT solver, which is what `IMPLICIT FALSE` -- the
    default -- selects. `soiltemp.solve_profile_implicit` is the
    Newton-Raphson on the whole profile at once.
    """
    a, b, c, d, e = thermal_coefficients(cs, kappa, zsum, alpha, beta, gamma,
                                         deltat, dp, nnodes,
                                         exp_trans=exp_trans, noflux=noflux)
    return calc_soil_thermal_fluxes(
        t0, moist, max_moist, ice, bubble, expt, a, b, c, d, e, nnodes,
        fs_active=fs_active, noflux=noflux, exp_trans=exp_trans,
        frozen_soil=frozen_soil, tfallback=tfallback)
