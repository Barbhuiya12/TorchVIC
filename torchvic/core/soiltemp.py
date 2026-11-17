"""The finite-difference soil temperature profile (QUICK_FLUX FALSE).

With QUICK_FLUX TRUE VIC's ground heat flux is Liang et al.'s (1999)
two-node closed form. With QUICK_FLUX FALSE VIC solves the soil
temperature profile by finite differences (Cherkauer and Lettenmaier,
1999): func_surf_energy_bal.c:404-445 solves the whole profile at every
trial surface temperature -- solve_T_profile_implicit (a Newton-Raphson
on all nodes at once, frozen_soil.c:235-322) with IMPLICIT, else the
Gauss-Seidel sweep solve_T_profile (frozen_soil.c:129-228), falling back
to the sweep when the Newton iteration fails -- and
calc_layer_average_thermal_props carries the solved nodes into the
layer temperatures (soil_conduction.c).

The node grid is VIC's (vic_init.c:833-945): uniform below twice the
first layer's depth without EXP_TRANS, exponential (Zsum = exp(Bexp j)
- 1) with it. Bexp and the exponential nodes are computed in *float*
(logf/expf) in an otherwise double-precision setup, and that is
reproduced: the double-precision value disagrees by 3.7e-08 relative.

What is reused and what is transcribed. The closed forms --
soil_conductivity, volumetric_heat_capacity, maximum_unfrozen_water
(torchvic.core.frozen) -- are VIC's to the bit and are called, not
copied. The explicit sweep is torchvic.core.frozen.solve_T_profile,
whose closed-form nodes are bit-exact against VIC's own (tests/
test_t_profile.py) and whose frozen nodes use VIC's own root_brent
sequence. Transcribed here: the grid, the node soil parameters
(set_node_parameters), the per-step node moisture distribution, the
implicit Newton solver (newt_raph_func_fast.c + fda_heat_eqn), the
ground-flux/storage/fusion terms the profile feeds, and the
profile-to-layer averaging.

Shapes: batched over leading cell/tile axes, [..., node] for profiles
and [..., layer] for layers, matching torchvic.core.frozen. `nnodes`
is a Python integer; the sweeps loop over nodes.
"""
import contextlib

import torch

from . import frozen, newtc
from .devconst import device_constant


def _graph():
    """On the CUDA graph path (graphsafe.ACTIVE): the loops below run a
    fixed count with no answer from the device."""
    from . import graphsafe
    return graphsafe.ACTIVE


def _node_min(lo, zsum, nmin, nnodes):
    """soil_conduction.c's lower node search: step down while the layer's
    top lies above the node (at most nnodes - 2 steps; a step for an
    element that has stopped leaves it)."""
    g = _graph()
    for _ in range(nnodes):
        take = ((lo < zsum.gather(-1, nmin.unsqueeze(-1)).squeeze(-1))
                & (nmin > 0))
        if not g and not bool(take.any()):
            break
        nmin = torch.where(take, nmin - 1, nmin)
    return nmin


def _node_max(hi, zsum, nmax, nnodes):
    """... and the upper one: step up while the layer's base lies below
    the node (at most nnodes - 1 steps)."""
    g = _graph()
    for _ in range(nnodes + 1):
        # The C reads Zsum_node[max] before testing max < Nnodes, so the
        # out-of-range read is harmless there; the gather here clamps to
        # a valid node, which the take mask never selects.
        zm = zsum.gather(
            -1, nmax.clamp(max=nnodes - 1).unsqueeze(-1)).squeeze(-1)
        take = (hi > zm) & (nmax < nnodes)
        if not g and not bool(take.any()):
            break
        nmax = torch.where(take, nmax + 1, nmax)
    return nmax


def _first_at_or_below(zsum, d1, idx):
    """The first node at or below depth d1, from `idx` on."""
    g = _graph()
    for _ in range(zsum.shape[-1]):
        take = zsum.gather(-1, idx.unsqueeze(-1)).squeeze(-1) < d1
        if not g and not bool(take.any()):
            break
        idx = torch.where(take, idx + 1, idx)
    return idx

#: initialize_parameters.c 5.1.0: the Newton-Raphson limits for the
#: implicit profile solve (newt_raph_func_fast.c).
NEWT_RAPH_MAXTRIAL = 150
NEWT_RAPH_TOLX = 1.0e-4
NEWT_RAPH_TOLF = 1.0e-1
NEWT_RAPH_R_MAX = 2.0
NEWT_RAPH_R_MIN = -5.0
NEWT_RAPH_RELAX1 = 0.9
NEWT_RAPH_RELAX2 = 0.7
NEWT_RAPH_RELAX3 = 0.2
NEWT_RAPH_EPS2 = 1.0e-4
#: Solve the implicit profile with `_newt_raph_fast` (all nodes and
#: Jacobian columns at once) rather than the per-node `_newt_raph`
#: transcription; the two agree bit for bit and the switch is kept so
#: that can be re-checked on a run.
FAST_NEWTON = True
#: how often (in Newton trials) the batched solver checks off the CPU
#: whether every column has stopped
GPU_DONE_EVERY = 10
#: The columns whose implicit profile solve is wanted, a [...] bool (or
#: None: all), set with `only` around a residual evaluation whose value
#: is kept only in places -- the surface-temperature search's trial for
#: the elements still iterating (implicit.root_brent). The other
#: columns are not iterated (_newt_raph_fast); their profile is the
#: failed solve's, which nothing reads.
ACTIVE = None


@contextlib.contextmanager
def only(mask):
    """Solve only where `mask` (and any enclosing `only`) holds."""
    global ACTIVE
    prev = ACTIVE
    ACTIVE = (mask if prev is None or mask is None
              or prev.shape != mask.shape else prev & mask)
    try:
        yield
    finally:
        ACTIVE = prev


#: vic_def.h:64.
MAX_NODES = 50
MM_PER_M = 1000.0
#: initialize_parameters.c:40: the moisture below the hydrologic column,
#: as a fraction of the node's porosity.
SOIL_SLAB_MOIST_FRACT = 1.0


def _bc(shape, x, vec=False):
    """`x` ([cell] or [cell, k]) broadcast onto a [..., (k)] batch.

    Unsqueezes trailing batch axes, never aligning the cell axis with
    a tile axis; a tensor already on the batch passes through.

    `vec` says `x` carries a trailing node/layer axis. Without it a
    per-cell [cell, k] field whose k equals the tile count (NODES 12
    on a 12-tile cell, 3 layers on a 3-tile cell) looks like a
    per-column [cell, tile] field and would pass through unbroadcast.
    """
    if not torch.is_tensor(x) or x.dim() == 0:
        return x
    shape = tuple(shape)
    if x.dim() == len(shape) + 1 and x.shape[:-1] == shape:
        return x
    if vec:
        trail = 1
    elif x.dim() == len(shape) and x.shape == shape:
        return x
    else:
        trail = x.dim() - 1
    v = x
    while v.dim() < len(shape) + trail:
        v = v.unsqueeze(-1 - trail)
    return v.expand(shape + ((x.shape[-1],) if trail else ()))


def bexp(dp, nnodes):
    """The exponential grid parameter.  vic_init.c:892, frozen_soil.c:168.

    The C computes `logf(Dp + 1.) / (double)(Nnodes - 1)`: the log in
    single precision, promoted, then divided in double. `dp` is a
    per-cell tensor; the value is per cell.
    """
    return torch.log(
        torch.as_tensor(dp + 1.0, dtype=torch.float32, device=dp.device)
    ).to(dp.dtype) / (nnodes - 1)


def node_grid(depth, dp, nnodes, exp_trans):
    """Thermal node depths and thicknesses.  vic_init.c:833-945.

    `depth` is [cell, layer] in m, `dp` [cell] the damping depth.
    Returns (zsum, dz), each [cell, nnodes]: the node positions below
    the surface and the thickness each node stands for.

    Without EXP_TRANS the nodes sit at the surface, the first layer's
    base, twice that, and then evenly spaced out to the damping depth
    -- except that the first layer's depth is read from cell 0's soil
    column for every cell (`soil_con[0].depth[0]`, :860-863), which is
    reproduced rather than fixed. With EXP_TRANS the nodes are
    expf(Bexp j) - 1, in float.

    VIC stops (log_err) when the thicknesses do not sum to dp to the
    millimetre, and with EXP_TRANS when fewer than three nodes would
    land in the top 50 cm; both stop the run here with ValueError.
    """
    if nnodes < 3:
        raise ValueError(
            'QUICK_FLUX FALSE with NODES %d: VIC reads Tnew_node[1] and '
            'Tnew_node[2] after the profile solve '
            '(func_surf_energy_bal.c:449-450), so fewer than 3 nodes is '
            'not a runnable grid' % nnodes)
    if nnodes > MAX_NODES:
        raise ValueError(
            'NODES %d exceeds VIC\'s MAX_NODES %d (get_global_param.c:956)'
            % (nnodes, MAX_NODES))
    zsum = torch.zeros(depth.shape[0], nnodes, dtype=depth.dtype,
                       device=depth.device)
    dz = torch.zeros_like(zsum)
    d0 = depth[:, 0]
    if not exp_trans:
        ref = depth[0, 0]
        # :856-862 -- dz from the cell's own first layer, Zsum from
        # cell 0's.
        dz[:, 0] = d0
        dz[:, 1] = d0
        if nnodes > 2:
            dz[:, 2] = d0
        zsum[:, 1] = ref
        z = 2.0 * ref
        if nnodes > 2:
            zsum[:, 2] = z
        # :863-872 -- extra nodes evenly spaced between twice the
        # first layer and the damping depth.
        tmpdp = dp - ref * 2.5
        for j in range(3, nnodes - 1):
            dz[:, j] = tmpdp / (float(nnodes) - 3.5)
            z = z + (dz[:, j] + dz[:, j - 1]) / 2.0
            zsum[:, j] = z
        # :873-879 -- the bottom node takes up the slack to dp.
        k = nnodes - 1
        dz[:, k] = (dp - z - dz[:, k - 1] / 2.0) * 2.0
        z = z + (dz[:, k - 1] + dz[:, k]) / 2.0
        zsum[:, k] = z
        # :880-886 -- VIC stops unless the thicknesses sum to dp to
        # the millimetre.
        if bool((((z * MM_PER_M + 0.5).to(torch.int64))
                 != ((dp * MM_PER_M + 0.5).to(torch.int64))).any()):
            bad = torch.nonzero(
                (z * MM_PER_M + 0.5).to(torch.int64)
                != (dp * MM_PER_M + 0.5).to(torch.int64)).flatten()
            raise ValueError(
                'thermal node thicknesses do not sum to dp (vic_init.c:880; '
                'cells %s)' % bad[:5].tolist())
    else:
        be = bexp(dp, nnodes)
        # :895-911 -- at least 3 nodes in the top 50 cm.
        tiny = torch.log(torch.as_tensor(
            dp + 1.0, dtype=torch.float32, device=dp.device)).to(dp.dtype)
        if bool((nnodes < 5 * tiny + 1).any()):
            raise ValueError(
                'NODES %d is too small for EXP_TRANS with this damping '
                'depth (vic_init.c:895: 5 ln(dp+1) < Nnodes-1)' % nnodes)
        for j in range(nnodes):
            # :912-914 -- expf(Bexp * j) - 1, in float.
            zsum[:, j] = (torch.exp(torch.as_tensor(
                be * j, dtype=torch.float32, device=dp.device))
                .to(dp.dtype) - 1.0)
        # :915-923 -- as written this compares node 0 (always exactly
        # 0) against the first layer's depth, so it never fires; kept
        # so a future VIC that means node 1 is caught the same way.
        if bool((zsum[:, 0] > depth[:, 0]).any()):
            raise ValueError(
                'first thermal node below the first soil layer '
                '(vic_init.c:915)')
        # :925-942 -- node thicknesses from the positions.
        dz[:, 0] = zsum[:, 1] - zsum[:, 0]
        for j in range(1, nnodes - 1):
            dz[:, j] = ((zsum[:, j + 1] - zsum[:, j]) / 2.0
                        + (zsum[:, j] - zsum[:, j - 1]) / 2.0)
        dz[:, nnodes - 1] = zsum[:, nnodes - 1] - zsum[:, nnodes - 2]
    return zsum, dz


def node_parameters(zsum, depth, porosity, expt, bubble, nnodes, nlayers,
                    noflux):
    """Soil properties at the thermal nodes.  set_node_parameters,
    soil_conduction.c:114-177.

    `zsum` is [cell, nnodes]; the layer properties are [cell, layer].
    Returns (porosity_node, expt_node, bubble_node, alpha, beta, gamma):
    the node properties are [cell, nnodes], the spacing constants
    [cell, nnodes - 1], indexed j - 1 as in the C. A node exactly on a
    layer boundary (exact floating-point equality, as in the C) takes
    the mean of the two layers; any node past the column's base takes
    the bottom layer.

    Called once per cell (compute_derived_state_vars.c:118); the result
    never changes during the run.
    """
    dev, dt = zsum.device, zsum.dtype
    nc = zsum.shape[0]
    por_n = torch.zeros(nc, nnodes, dtype=dt, device=dev)
    ex_n = torch.zeros_like(por_n)
    bu_n = torch.zeros_like(por_n)
    # The layer index is per cell: cells cross their layer boundaries
    # at different nodes.
    lidx = torch.zeros(nc, dtype=torch.int64, device=dev)
    lsum = torch.zeros(nc, dtype=dt, device=dev)
    past = torch.zeros(nc, dtype=torch.bool, device=dev)
    last = nlayers - 1
    for n in range(nnodes):
        li = lidx.clamp(max=last)
        on_edge = ((zsum[:, n] == lsum + depth.gather(1, li.unsqueeze(1))
                    .squeeze(1)) & (n != 0)
                   & (lidx != last))
        nxt = (li + 1).clamp(max=last)
        g = lambda t, i: t.gather(1, i.unsqueeze(1)).squeeze(1)
        por_n[:, n] = torch.where(on_edge, (g(porosity, li) + g(porosity, nxt))
                                  / 2.0, g(porosity, li))
        ex_n[:, n] = torch.where(on_edge, (g(expt, li) + g(expt, nxt))
                                 / 2.0, g(expt, li))
        bu_n[:, n] = torch.where(on_edge, (g(bubble, li) + g(bubble, nxt))
                                 / 2.0, g(bubble, li))
        # :154-161 -- advance past the layer the node cleared, once.
        adv = (zsum[:, n] > lsum + g(depth, li)) & ~past
        lsum = torch.where(adv, lsum + g(depth, li), lsum)
        lidx = torch.where(adv, lidx + 1, lidx)
        now_past = adv & (lidx == nlayers)
        past = past | now_past
        lidx = torch.where(now_past, torch.full_like(lidx, last), lidx)
    alpha = torch.zeros(nc, nnodes - 1, dtype=dt, device=dev)
    beta = torch.zeros_like(alpha)
    gamma = torch.zeros_like(alpha)
    # :165-169 -- spacings between node triples.
    alpha[:, :nnodes - 2] = zsum[:, 2:] - zsum[:, :nnodes - 2]
    beta[:, :nnodes - 2] = zsum[:, 1:nnodes - 1] - zsum[:, :nnodes - 2]
    gamma[:, :nnodes - 2] = zsum[:, 2:] - zsum[:, 1:nnodes - 1]
    if noflux:
        # :170-176 -- the no-flux bottom boundary mirrors the last gap.
        alpha[:, nnodes - 2] = 2.0 * (zsum[:, nnodes - 1]
                                     - zsum[:, nnodes - 2])
        beta[:, nnodes - 2] = (zsum[:, nnodes - 1] - zsum[:, nnodes - 2])
        gamma[:, nnodes - 2] = (zsum[:, nnodes - 1] - zsum[:, nnodes - 2])
    return por_n, ex_n, bu_n, alpha, beta, gamma


def distribute_node_moisture(moist_mm, temp, zsum, porosity_node, expt_node,
                             bubble_node, depth, soil_dens_min,
                             bulk_dens_min, quartz, soil_density,
                             bulk_density, organic, nnodes, nlayers,
                             fs_active, frozen_soil):
    """Moisture, ice, conductivity and heat capacity at the nodes.
    distribute_node_moisture_properties, soil_conduction.c:186-293.

    `moist_mm` is [..., layer] in mm, `temp` [..., node] in C; the node
    soil properties are [..., node], the layer properties [..., layer].
    Returns (moist_node, ice_node, kappa_node, cs_node), each [...,
    nnodes]; the moisture is a volume fraction.

    VIC calls this once at setup (compute_derived_state_vars.c:132)
    and again every time runoff() leaves new moisture (runoff.c:381):
    the profile the surface solve iterates on is always the previous
    step's. `fs_active` is per cell (or per [...]); `frozen_soil` is
    the FROZEN_SOIL switch. Ice forms only where both hold and the
    node is below freezing.
    """
    shape = temp.shape[:-1]
    dev, dt = temp.device, temp.dtype

    bc = lambda x: _bc(shape, x, vec=True)        # all per node/layer
    zsum = bc(zsum)
    porosity_node = bc(porosity_node)
    expt_node = bc(expt_node)
    bubble_node = bc(bubble_node)
    depth = bc(depth)
    soil_dens_min = bc(soil_dens_min)
    bulk_dens_min = bc(bulk_dens_min)
    quartz = bc(quartz)
    soil_density = bc(soil_density)
    bulk_density = bc(bulk_density)
    organic = bc(organic)
    act = torch.as_tensor(fs_active, dtype=torch.bool, device=dev)
    while act.dim() < len(shape):
        act = act.unsqueeze(-1)
    act = act.expand(shape)
    last = nlayers - 1
    moist_n = torch.zeros(shape + (nnodes,), dtype=dt, device=dev)
    ice_n = torch.zeros_like(moist_n)
    kappa_n = torch.zeros_like(moist_n)
    cs_n = torch.zeros_like(moist_n)
    lidx = torch.zeros(shape, dtype=torch.int64, device=dev)
    lsum = torch.zeros(shape, dtype=dt, device=dev)
    past = torch.zeros(shape, dtype=torch.bool, device=dev)
    g = lambda t, i: t.gather(-1, i.unsqueeze(-1)).squeeze(-1)
    if torch.is_tensor(frozen_soil):
        while frozen_soil.dim() < len(shape):
            frozen_soil = frozen_soil.unsqueeze(-1)
        active = act & frozen_soil.expand(shape)
    elif frozen_soil:
        active = act
    else:
        active = torch.zeros(shape, dtype=torch.bool, device=dev)
    for n in range(nnodes):
        li = lidx.clamp(max=last)
        # :221-238 -- the node's moisture from its layer (mean of two
        # on a boundary), or the slab fraction past the column's base.
        on_edge = ((zsum[..., n] == lsum + g(depth, li)) & (n != 0)
                   & (lidx != last))
        nxt = (li + 1).clamp(max=last)
        lay = torch.where(on_edge, (g(moist_mm, li) / g(depth, li)
                                   + g(moist_mm, nxt) / g(depth, nxt))
                          / MM_PER_M / 2.0,
                          g(moist_mm, li) / g(depth, li) / MM_PER_M)
        slab = (device_constant(SOIL_SLAB_MOIST_FRACT, dt, dev)
                * porosity_node[..., n])
        m = torch.where(~past, lay, slab)
        # :242-245 -- the C's HACK: clamp to the node's porosity.
        m = torch.where(m - porosity_node[..., n] > 0,
                        porosity_node[..., n], m)
        frozen_here = (temp[..., n] < 0) & active
        # :246-265 -- frozen: ice is what the curve will not hold.
        ice_f = m - frozen.maximum_unfrozen_water(
            temp[..., n], porosity_node[..., n], bubble_node[..., n],
            expt_node[..., n])
        ice_f = torch.where(ice_f < 0, torch.zeros_like(ice_f), ice_f)
        ice = torch.where(frozen_here, ice_f, torch.zeros_like(m))
        wu = torch.where(frozen_here, m - ice_f, m)
        kappa_n[..., n] = frozen.soil_conductivity(
            m, wu, g(soil_dens_min, li), g(bulk_dens_min, li),
            g(quartz, li), g(soil_density, li), g(bulk_density, li),
            g(organic, li))
        cs_n[..., n] = frozen.volumetric_heat_capacity(
            g(bulk_density, li) / g(soil_density, li), m - ice, ice,
            g(organic, li))
        moist_n[..., n] = m
        ice_n[..., n] = ice
        adv = (zsum[..., n] > lsum + g(depth, li)) & ~past
        lsum = torch.where(adv, lsum + g(depth, li), lsum)
        lidx = torch.where(adv, lidx + 1, lidx)
        now_past = adv & (lidx == nlayers)
        past = past | now_past
        lidx = torch.where(now_past, torch.full_like(lidx, last), lidx)
    return moist_n, ice_n, kappa_n, cs_n


def linear_interp(x, lx, ux, ly, uy):
    """Linear interpolation.  interpoloation.c:14-22."""
    return (x - lx) / (ux - lx) * (uy - ly) + ly


def frost_temperature_and_depth(zsum, temp, depth, frost_fract, frost_slope,
                                nnodes, nlayers):
    """Node temperatures bracketed per layer.  soil_conduction.c:300-400.

    `zsum` is [..., node], `temp` [..., node], `depth` [..., layer].
    Returns (tmp_t, tmp_z): the profile cut to each layer's span,
    [..., layer, node, frost] and [..., layer, node], with the
    frost-area distribution spread over the last axis (frost_slope 0
    and one area here -- SPATIAL_FROST is refused -- so every area
    holds the mean).

    Warns (and clamps to the deepest node) where the nodes do not
    reach the column's base, as the C does with log_warn.
    """
    shape = temp.shape[:-1]
    dev, dt = temp.device, temp.dtype
    zsum = _bc(shape, zsum, vec=True)
    depth = _bc(shape, depth, vec=True)
    nfrost = frost_fract.shape[-1] if torch.is_tensor(frost_fract) \
        else int(frost_fract)
    if not torch.is_tensor(frost_fract):
        frost_fract = torch.full(shape + (nfrost,), 1.0 / max(nfrost - 1, 1),
                                 dtype=dt, device=dev)
    if not torch.is_tensor(frost_slope):
        frost_slope = torch.zeros(shape, dtype=dt, device=dev)
    lsum = torch.cat([torch.zeros_like(depth[..., :1]),
                      torch.cumsum(depth, -1)], -1)
    tmp_t = torch.zeros(shape + (nlayers, nnodes, nfrost), dtype=dt,
                        device=dev)
    tmp_z = torch.zeros(shape + (nlayers, nnodes), dtype=dt, device=dev)
    short = torch.zeros(shape, dtype=torch.bool, device=dev)
    for l in range(nlayers):
        lo = lsum[..., l]
        hi = lsum[..., l + 1]
        nmin = torch.full(shape, nnodes - 2, dtype=torch.int64, device=dev)
        nmin = _node_min(lo, zsum, nmin, nnodes)
        nmax = torch.ones(shape, dtype=torch.int64, device=dev)
        nmax = _node_max(hi, zsum, nmax, nnodes)
        over = nmax >= nnodes
        short = short | over
        nmax = torch.where(over, torch.full_like(nmax, nnodes - 1), nmax)
        for n in range(nnodes):
            use = (nmin <= n) & (n <= nmax)
            if not _graph() and not bool(use.any()):
                continue
            zn = zsum[..., n]
            tn = temp[..., n]
            is_min = use & (nmin == n)
            is_max = use & (nmax == n)
            # The layer's edges, interpolated where a node spans them
            # (:346-373: the lower edge first, then the upper, which
            # overwrites where the two coincide). Clamped neighbours:
            # node 0 is never an upper edge and the last node never a
            # lower one, so the clamped values are never selected.
            zp1 = zsum[..., min(n + 1, nnodes - 1)]
            tp1 = temp[..., min(n + 1, nnodes - 1)]
            zm1 = zsum[..., max(n - 1, 0)]
            tm1 = temp[..., max(n - 1, 0)]
            vm = torch.where(zn < lo,
                             linear_interp(lo, zn, zp1, tn, tp1), tn)
            pick = torch.where(is_min, vm, tn)
            vh = torch.where(zn > hi,
                             linear_interp(hi, zm1, zn, tm1, tn), tn)
            pick = torch.where(is_max, vh, pick)
            z = torch.where(is_min, lo, zn)
            z_hi = torch.where(is_max, hi, z)
            # :375-398 -- the frost-area distribution about the mean.
            tmin = pick - frost_slope / 2.0
            tmax = tmin + frost_slope
            frac = torch.zeros(shape, dtype=dt, device=dev)
            for f in range(nfrost):
                if nfrost > 1:
                    if f == 0:
                        frac = frost_fract[..., 0] / 2.0
                    else:
                        frac = frac + (frost_fract[..., f - 1] / 2.0
                                       + frost_fract[..., f] / 2.0)
                    val = linear_interp(frac, torch.zeros_like(frac),
                                        torch.ones_like(frac), tmin, tmax)
                else:
                    val = pick
                tmp_t[..., l, n, f] = torch.where(
                    use, val, tmp_t[..., l, n, f])
            tmp_z[..., l, n] = torch.where(use, z_hi, tmp_z[..., l, n])
    if not _graph() and bool(short.any()):
        import warnings
        warnings.warn('soil thermal nodes do not extend below the bottom '
                      'soil layer; using the deepest node temperature '
                      '(soil_conduction.c:442)')
    return tmp_t, tmp_z


def layer_temperature(tmp_t, tmp_z, zsum, depth, nnodes, nlayers):
    """Layer-average temperatures from the profile.  soil_conduction.c:406.

    `tmp_t`/`tmp_z` are frost_temperature_and_depth's; `depth` is
    [..., layer]. Returns [..., layer]. The mean over the frost areas
    is the last one (index Nfrost), which with one area is the only one.
    """
    shape = tmp_t.shape[:-3]
    dev, dt = tmp_t.device, tmp_t.dtype
    zsum = _bc(shape, zsum, vec=True)
    depth = _bc(shape, depth, vec=True)
    lsum = torch.cat([torch.zeros_like(depth[..., :1]),
                      torch.cumsum(depth, -1)], -1)
    out = torch.zeros(shape + (nlayers,), dtype=dt, device=dev)
    nf = tmp_t.shape[-1] - 1
    for l in range(nlayers):
        lo = lsum[..., l]
        hi = lsum[..., l + 1]
        nmin = torch.full(shape, nnodes - 2, dtype=torch.int64, device=dev)
        nmin = _node_min(lo, zsum, nmin, nnodes)
        nmax = torch.ones(shape, dtype=torch.int64, device=dev)
        nmax = _node_max(hi, zsum, nmax, nnodes)
        nmax = torch.where(nmax >= nnodes,
                           torch.full_like(nmax, nnodes - 1), nmax)
        tot = torch.zeros(shape, dtype=dt, device=dev)
        for n in range(nnodes - 1):
            use = (nmin <= n) & (n < nmax)
            seg = ((tmp_z[..., l, n + 1] - tmp_z[..., l, n])
                   * (tmp_t[..., l, n + 1, nf] + tmp_t[..., l, n, nf])
                   / 2.0)
            tot = tot + torch.where(use, seg, torch.zeros_like(seg))
        out[..., l] = tot / depth[..., l]
    return out


def layer_ice_content(tmp_t, tmp_z, zsum, depth, max_moist, expt, bubble,
                      nnodes, nlayers, fs_active, frozen_soil, moist):
    """Layer-average ice contents from the profile.  soil_conduction.c:468.

    Shapes as layer_temperature's, plus `max_moist` [..., layer] in mm,
    the curve parameters, and the layer's current moisture `moist` in
    mm -- the C subtracts the unfrozen water from layer[lidx].moist,
    not from max_moist. Returns [..., layer, frost]: the ice in mm in
    each frost area (one here). Zero everywhere unless frozen soil is
    on and active. Note the C clamps only from below here -- no upper
    clamp -- and that is reproduced.
    """
    shape = tmp_t.shape[:-3]
    dev, dt = tmp_t.device, tmp_t.dtype
    zsum = _bc(shape, zsum, vec=True)
    depth = _bc(shape, depth, vec=True)
    max_moist = _bc(shape, max_moist, vec=True)
    expt = _bc(shape, expt, vec=True)
    bubble = _bc(shape, bubble, vec=True)
    moist = _bc(shape, moist, vec=True)
    nfrost = tmp_t.shape[-1]
    lsum = torch.cat([torch.zeros_like(depth[..., :1]),
                      torch.cumsum(depth, -1)], -1)
    out = torch.zeros(shape + (nlayers, nfrost), dtype=dt, device=dev)
    act = torch.as_tensor(fs_active, dtype=torch.bool, device=dev)
    while act.dim() < len(shape):
        act = act.unsqueeze(-1)
    act = act.expand(shape)
    run = act & bool(frozen_soil) if not torch.is_tensor(frozen_soil) \
        else act & frozen_soil
    for l in range(nlayers):
        lo = lsum[..., l]
        hi = lsum[..., l + 1]
        nmin = torch.full(shape, nnodes - 2, dtype=torch.int64, device=dev)
        nmin = _node_min(lo, zsum, nmin, nnodes)
        nmax = torch.ones(shape, dtype=torch.int64, device=dev)
        nmax = _node_max(hi, zsum, nmax, nnodes)
        nmax = torch.where(nmax >= nnodes,
                           torch.full_like(nmax, nnodes - 1), nmax)
        for f in range(nfrost):
            ice = torch.zeros(shape + (nnodes,), dtype=dt, device=dev)
            for n in range(nnodes):
                use = (nmin <= n) & (n <= nmax)
                val = moist[..., l] - frozen.maximum_unfrozen_water(
                    tmp_t[..., l, n, f], max_moist[..., l], bubble[..., l],
                    expt[..., l])
                val = torch.where(val < 0, torch.zeros_like(val), val)
                val = torch.where(run, val, torch.zeros_like(val))
                ice[..., n] = torch.where(use, val, ice[..., n])
            tot = torch.zeros(shape, dtype=dt, device=dev)
            for n in range(nnodes - 1):
                use = (nmin <= n) & (n < nmax)
                seg = ((tmp_z[..., l, n + 1] - tmp_z[..., l, n])
                       * (ice[..., n + 1] + ice[..., n]) / 2.0)
                tot = tot + torch.where(use, seg, torch.zeros_like(seg))
            out[..., l, f] = tot / depth[..., l]
    return out


def profile_ground_terms(ts, t_new, t_old, t1_old, ts_old, ice0_prepare,
                         moist0, porosity0, bubble0, expt0, kappa1, kappa2,
                         cs1, d1, dt, zsum, atten, exp_trans, fs_active,
                         frozen_soil, grnd_flux_type='GF_410'):
    """T1, ground flux, storage change and fusion from the solved profile.
    func_surf_energy_bal.c:447-613. GRND_FLUX_TYPE acts only without
    EXP_TRANS (:466-487); the EXP_TRANS flux has one form (:489-497).

    `ts` is the trial surface temperature, `t_new` the solved profile,
    `t_old` the profile the step opened with, `t1_old` the previous
    temperature at the first layer's base (energy->T[1], or the old
    profile interpolated there with EXP_TRANS -- calc_surf_energy_bal.c
    :161-177, a separate line below), `ice0_prepare` prepare_full_energy's
    top-layer ice, `moist0` its volume fraction, and the `*0` curve
    parameters the top layer's. Returns (t1, grnd, delta_h, fusion) in
    VIC's internal sign, with `atten` (snow_coverage + (1 -
    snow_coverage) surf_atten) applied to the ground flux.
    """
    zsum = _bc(ts.shape, zsum, vec=True)
    act = torch.as_tensor(fs_active, dtype=torch.bool, device=ts.device)
    while act.dim() < ts.dim():
        act = act.unsqueeze(-1)
    act = act.expand_as(ts)
    if torch.is_tensor(frozen_soil):
        fz = frozen_soil
        while fz.dim() < ts.dim():
            fz = fz.unsqueeze(-1)
        run = act & fz.expand_as(ts)
    elif frozen_soil:
        run = act
    else:
        run = torch.zeros_like(ts, dtype=torch.bool)
    if not exp_trans:
        # :448-450: T1 is node 1, T11 node 2.
        t1 = t_new[..., 1]
        t11 = t_new[..., 2]
        # :478-486, the GF_410 form. D2 is depth[0] again, not
        # depth[1] (calc_surf_energy_bal.c:179).
        if grnd_flux_type == 'GF_406':
            # :467-473: the surface term alone.
            grnd = atten * (kappa1 / d1 * (t1 - ts))
        else:
            grnd = atten * (kappa1 / d1 * (t1 - ts)
                            + kappa2 / d1 * (t11 - t1)) / 2.0
        # :503-506: the storage between the surface and the layer base.
        delta_h = (cs1 * ((ts_old + t1_old) - (ts + t1)) * d1 / dt / 2.0)
        # :535-548: the fusion at the mean of surface and base. Note
        # the curve's maximum is the layer's porosity here, while
        # prepare_full_energy's ice0 used max_moist / (depth MM_PER_M):
        # the same quantity, rounded differently, and each kept as its
        # own call site computes it.
        t_mid = (ts + t1) / 2.0
        ice = moist0 - frozen.maximum_unfrozen_water(
            t_mid, porosity0, bubble0, expt0)
        ice = torch.where(ice < 0.0, torch.zeros_like(ice), ice)
        ice = torch.where(t_mid < 0.0, ice, torch.zeros_like(ice))
        fusion = (-frozen.CONST_RHOICE * frozen.CONST_LATICE
                  * (ice0_prepare - ice) * d1 / dt)
        fusion = torch.where(run, fusion, torch.zeros_like(fusion))
        return t1, grnd, delta_h, fusion
    # With EXP_TRANS the first layer's base falls between two nodes;
    # T1 is interpolated there (:452-463).
    nnodes = t_new.shape[-1]
    idx = torch.ones_like(ts, dtype=torch.int64)
    idx = _first_at_or_below(zsum, d1, idx)
    g = lambda t, i: t.gather(-1, i.unsqueeze(-1)).squeeze(-1)
    d_minus = g(zsum, idx - 1)
    d_plus = g(zsum, idx)
    t_minus = g(t_new, idx - 1)
    t_plus = g(t_new, idx)
    t1 = t_minus + (t_plus - t_minus) * (d1 - d_minus) / (d_plus - d_minus)
    # :488-497: the flux from the two gradients about the base.
    grnd = atten * (kappa1 / (d1 - d_minus) * (t1 - t_minus)
                    + kappa2 / (d_plus - d1) * (t_plus - t1)) / 2.0
    # func_surf_energy_bal.c:408 sets T_node[0] to the trial Ts before
    # solving, and Told_node is the SAME array (calc passes energy->T
    # for both), so the segment sums below read the trial Ts back as
    # Told[0]. The passed `t_old` is the step-opening profile with the
    # old surface still at node 0 -- alias it the same way. (Without
    # EXP_TRANS no Told_node is read at all, so only this branch is
    # affected; measured 74.6 W/m2 on delta_h at one cold trial.)
    t_old = t_old.clone()
    t_old[..., 0] = ts
    # :507-530: the storage summed over the segments below the base.
    delta_h = torch.zeros_like(ts)
    seg = torch.zeros_like(ts, dtype=torch.int64)
    # (the graph path: a fixed count of segments, each taken where the
    # eager loop would have run it at all -- some element had one more)
    gr = _graph()
    for _ in range(nnodes):
        take = (zsum.gather(-1, (seg + 1).unsqueeze(-1)).squeeze(-1) < d1)
        if not gr and not bool(take.any()):
            break
        add = (cs1 * ((g(t_old, seg) + g(t_old, seg + 1))
                      - (g(t_new, seg) + g(t_new, seg + 1)))
               * (g(zsum, seg + 1) - g(zsum, seg)) / dt / 2.0)
        dh_ = delta_h + torch.where(take, add, torch.zeros_like(add))
        delta_h = torch.where(take.any(), dh_, delta_h) if gr else dh_
        seg = torch.where(take, seg + 1, seg)
    delta_h = delta_h + (cs1 * ((g(t_old, seg) + t1_old)
                                - (g(t_new, seg) + t1))
                         * (d1 - g(zsum, seg)) / dt / 2.0)
    # :550-613: the fusion summed the same way, with the ice at each
    # segment's mean old and new temperature. The local ice0 shadows
    # prepare_full_energy's: it is recomputed per segment here.
    fusion = torch.zeros_like(ts)
    seg = torch.zeros_like(ts, dtype=torch.int64)
    for _ in range(nnodes):
        take = (zsum.gather(-1, (seg + 1).unsqueeze(-1)).squeeze(-1) < d1)
        if not gr and not bool(take.any()):
            break
        tm_o = (g(t_old, seg) + g(t_old, seg + 1)) / 2.0
        i0 = moist0 - frozen.maximum_unfrozen_water(
            tm_o, porosity0, bubble0, expt0)
        i0 = torch.where(i0 < 0.0, torch.zeros_like(i0), i0)
        i0 = torch.where(tm_o < 0.0, i0, torch.zeros_like(i0))
        tm_n = (g(t_new, seg) + g(t_new, seg + 1)) / 2.0
        ic = moist0 - frozen.maximum_unfrozen_water(
            tm_n, porosity0, bubble0, expt0)
        ic = torch.where(ic < 0.0, torch.zeros_like(ic), ic)
        ic = torch.where(tm_n < 0.0, ic, torch.zeros_like(ic))
        add = (-frozen.CONST_RHOICE * frozen.CONST_LATICE * (i0 - ic)
               * (g(zsum, seg + 1) - g(zsum, seg)) / dt)
        fu_ = fusion + torch.where(take, add, torch.zeros_like(add))
        fusion = torch.where(take.any(), fu_, fusion) if gr else fu_
        seg = torch.where(take, seg + 1, seg)
    tm_o = (g(t_old, seg) + t1_old) / 2.0
    i0 = moist0 - frozen.maximum_unfrozen_water(tm_o, porosity0, bubble0,
                                                expt0)
    i0 = torch.where(i0 < 0.0, torch.zeros_like(i0), i0)
    i0 = torch.where(tm_o < 0.0, i0, torch.zeros_like(i0))
    tm_n = (g(t_new, seg) + t1) / 2.0
    ic = moist0 - frozen.maximum_unfrozen_water(tm_n, porosity0, bubble0,
                                                expt0)
    ic = torch.where(ic < 0.0, torch.zeros_like(ic), ic)
    ic = torch.where(tm_n < 0.0, ic, torch.zeros_like(ic))
    fusion = fusion + (-frozen.CONST_RHOICE * frozen.CONST_LATICE
                       * (i0 - ic) * (d1 - g(zsum, seg)) / dt)
    fusion = torch.where(run, fusion, torch.zeros_like(fusion))
    return t1, grnd, delta_h, fusion


def t1_old_exp_trans(t_old, zsum, d1):
    """The previous temperature at the first layer's base, EXP_TRANS.
    calc_surf_energy_bal.c:165-177: the old profile interpolated to
    depth[0]. (Without EXP_TRANS -- and on the QUICK_FLUX path -- T1_old
    is energy->T[1], no interpolation.)"""
    zsum = _bc(t_old.shape[:-1], zsum, vec=True)
    idx = torch.ones_like(d1, dtype=torch.int64)
    idx = _first_at_or_below(zsum, d1, idx)
    g = lambda t, i: t.gather(-1, i.unsqueeze(-1)).squeeze(-1)
    return g(t_old, idx - 1) + (g(t_old, idx) - g(t_old, idx - 1)) * (
        d1 - g(zsum, idx - 1)) / (g(zsum, idx) - g(zsum, idx - 1))


def quick_solve_nodes(t_node, nnodes):
    """The reduced node count of a QUICK_SOLVE trial.  :257-279.

    calc_surf_energy_bal.c solves only down past the thaw front while
    bracketing Tsurf (`tmpNnodes`), then re-solves the full column at
    the root. `t_node` is [..., node], the profile the solve opened
    with. Returns [...] int64.
    """
    shape = t_node.shape[:-1]
    dev = t_node.device
    out = torch.zeros(shape, dtype=torch.int64, device=dev)
    for i in range(nnodes - 5, -1, -1):
        hit = (t_node[..., i] >= 0) & (t_node[..., i + 1] < 0)
        out = torch.where(hit, torch.full_like(out, i + 1), out)
    noval = out == 0
    deep = (t_node[..., 0] <= 0) & (t_node[..., 1] >= 0)
    out = torch.where(noval & deep, torch.full_like(out, nnodes), out)
    out = torch.where(noval & ~deep, torch.full_like(out, 3), out)
    out = torch.where(~noval, out + 4, out)
    return out


def tridiag(a, b, c, r, n):
    """A tridiagonal solve, in place.  newt_raph_func_fast.c:132-169.

    Each of `a`, `b`, `c`, `r` is [..., n]; returns the solution (the
    C leaves it in `r`). Cloned inputs: the C's forward/backward
    substitution overwrites its diagonals.
    """
    a, b, c, r = a.clone(), b.clone(), c.clone(), r.clone()
    # Forward substitution (:143-159). a[0] is never read and c[n-1]
    # never feeds r -- in the C both are uninitialized stack -- so
    # whatever the caller passed there is harmless.
    factor = b[..., 0].clone()
    b[..., 0] = 1.0
    c[..., 0] = c[..., 0] / factor
    r[..., 0] = r[..., 0] / factor
    for i in range(1, n):
        factor = a[..., i].clone()
        a[..., i] = a[..., i] - b[..., i - 1] * factor
        b[..., i] = b[..., i] - c[..., i - 1] * factor
        r[..., i] = r[..., i] - r[..., i - 1] * factor
        factor = b[..., i].clone()
        b[..., i] = 1.0
        c[..., i] = c[..., i] / factor
        r[..., i] = r[..., i] / factor
    # Backward substitution (:161-169).
    for j in range(n - 2, -1, -1):
        factor = c[..., j].clone()
        c[..., j] = c[..., j] - b[..., j + 1] * factor
        r[..., j] = r[..., j] - r[..., j + 1] * factor
        factor = b[..., j].clone()
        r[..., j] = r[..., j] / factor
    return r


class _FdaHeatEqn(object):
    """fda_heat_eqn's residual with its static state.  frozen_soil.c:624.

    The C keeps the trial conductivity, heat capacity, ice and
    temperature differences in threadprivate statics across the calls
    of one Newton solve: a full evaluation (focus -1) refreshes nodes
    0..n, and each Jacobian column (focus j) refreshes only the window
    about j, reading the rest stale. That staleness is load-bearing --
    without NOFLUX kappa_new[Nnodes-1] is *never* assigned, so the
    last row's Dkappa reads the static zero -- and is reproduced by
    holding the same arrays here across eval_all/eval_focus calls.

    `x` is [..., n], the unknowns T[1..n]; index shifts of one against
    the [..., nnodes] profiles are as in the C.
    """

    def __init__(self, t0, moist, ice, kappa, cs, max_moist, bubble, expt,
                 alpha, beta, gamma, zsum, deltat, dp, bulk_dens_min,
                 soil_dens_min, quartz, bulk_density, soil_density,
                 organic, depth, nlayers, n, noflux, exp_trans):
        self.t0 = t0
        self.moist = moist
        self.ice = ice
        self.kappa = kappa
        self.cs = cs
        self.max_moist = max_moist
        self.bubble = bubble
        self.expt = expt
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.zsum = zsum
        self.deltat = deltat
        self.dp = dp
        self.bulk_dens_min = bulk_dens_min
        self.soil_dens_min = soil_dens_min
        self.quartz = quartz
        self.bulk_density = bulk_density
        self.soil_density = soil_density
        self.organic = organic
        self.depth = depth
        self.nlayers = nlayers
        self.n = n
        self.noflux = noflux
        self.exp_trans = exp_trans
        shape = t0.shape[:-1]
        dev, dt = t0.device, t0.dtype
        nn = t0.shape[-1]
        # :722-728 -- the boundary temperatures of this solve.
        self.ts = t0[..., 0]
        self.tb = t0[..., n + 1] if not noflux else t0[..., n]
        if exp_trans:
            # :713-720 -- logf again, over n + 1 (n without NOFLUX).
            den = n if noflux else n + 1
            self.bexp = torch.log(torch.as_tensor(
                dp + 1.0, dtype=torch.float32,
                device=dp.device)).to(dp.dtype) / den
        else:
            self.bexp = None
        self.ice_new = torch.zeros(shape + (nn,), dtype=dt, device=dev)
        self.cs_new = torch.zeros(shape + (nn,), dtype=dt, device=dev)
        self.kappa_new = torch.zeros(shape + (nn,), dtype=dt, device=dev)
        self.dt_ = torch.zeros(shape + (n,), dtype=dt, device=dev)
        self.dt_down = torch.zeros(shape + (n,), dtype=dt, device=dev)
        self.dt_up = torch.zeros(shape + (n,), dtype=dt, device=dev)
        self.dkappa = torch.zeros(shape + (n,), dtype=dt, device=dev)

    def _refresh_props(self, x, lo, hi, reset):
        """kappa_new/cs_new/ice_new on nodes lo..hi.  :745-789, :897-927.

        A full evaluation resets the conductivity and heat capacity to
        the step's before refreshing (`reset`); a Jacobian column does
        not -- where the ice is unchanged the trial values stand as
        the last call left them.
        """
        dev = self.t0.device
        last = self.nlayers - 1
        g = lambda t, i: t.gather(-1, i.unsqueeze(-1)).squeeze(-1)
        lidx = torch.zeros(self.t0.shape[:-1], dtype=torch.int64,
                           device=dev)
        lsum = torch.zeros(self.t0.shape[:-1], dtype=self.t0.dtype,
                           device=dev)
        past = torch.zeros(self.t0.shape[:-1], dtype=torch.bool,
                           device=dev)
        for i in range(hi + 1):
            li = lidx.clamp(max=last)
            if reset and i >= lo:
                self.kappa_new[..., i] = self.kappa[..., i]
            if lo <= i and i >= 1:
                t = x[..., i - 1]
                below = t < 0.0
                # No FS_ACTIVE gate: the implicit residual freezes
                # below 0 C wherever it runs, frozen soil or not.
                ic = self.moist[..., i] - frozen.maximum_unfrozen_water(
                    t, self.max_moist[..., i], self.bubble[..., i],
                    self.expt[..., i])
                ic = torch.where(ic < 0.0, torch.zeros_like(ic), ic)
                ic = torch.where(below, ic, torch.zeros_like(ic))
                self.ice_new[..., i] = ic
                if reset:
                    self.cs_new[..., i] = self.cs[..., i]
                changed = ic != self.ice[..., i]
                kn = frozen.soil_conductivity(
                    self.moist[..., i],
                    self.moist[..., i] - ic,
                    g(self.soil_dens_min, li), g(self.bulk_dens_min, li),
                    g(self.quartz, li), g(self.soil_density, li),
                    g(self.bulk_density, li), g(self.organic, li))
                cn = frozen.volumetric_heat_capacity(
                    g(self.bulk_density, li) / g(self.soil_density, li),
                    self.moist[..., i] - ic, ic, g(self.organic, li))
                self.kappa_new[..., i] = torch.where(
                    changed, kn, self.kappa_new[..., i])
                self.cs_new[..., i] = torch.where(
                    changed, cn, self.cs_new[..., i])
            adv = ((self.zsum[..., i] > lsum + g(self.depth, li)) & ~past)
            lsum = torch.where(adv, lsum + g(self.depth, li), lsum)
            lidx = torch.where(adv, lidx + 1, lidx)
            now_past = adv & (lidx == self.nlayers)
            past = past | now_past
            lidx = torch.where(now_past, torch.full_like(lidx, last), lidx)

    def _diffs_and_dkappa(self, x, lo, hi):
        """DT/DT_up/DT_down/Dkappa on rows lo..hi.  :792-817, :930-958."""
        for i in range(lo, hi + 1):
            if i == 0:
                self.dt_[..., i] = x[..., i + 1] - self.ts
                self.dt_up[..., i] = x[..., i] - self.ts
                self.dt_down[..., i] = x[..., i + 1] - x[..., i]
            elif i == self.n - 1:
                self.dt_[..., i] = self.tb - x[..., i - 1]
                self.dt_up[..., i] = x[..., i] - x[..., i - 1]
                self.dt_down[..., i] = self.tb - x[..., i]
            else:
                self.dt_[..., i] = x[..., i + 1] - x[..., i - 1]
                self.dt_up[..., i] = x[..., i] - x[..., i - 1]
                self.dt_down[..., i] = x[..., i + 1] - x[..., i]
            if i < self.n - 1:
                self.dkappa[..., i] = (self.kappa_new[..., i + 2]
                                       - self.kappa_new[..., i])
            elif not self.noflux:
                self.dkappa[..., i] = (self.kappa_new[..., i + 2]
                                       - self.kappa_new[..., i])
            else:
                self.dkappa[..., i] = (self.kappa_new[..., i + 1]
                                       - self.kappa_new[..., i])

    def _rows(self, x, lo, hi, res):
        """Residual rows lo..hi from the refreshed state.  :819-864."""
        for i in range(lo, hi + 1):
            storage = (self.cs_new[..., i + 1] * (x[..., i]
                                                  - self.t0[..., i + 1])
                       / self.deltat + x[..., i]
                       * (self.cs_new[..., i + 1] - self.cs[..., i + 1])
                       / self.deltat)
            if not self.exp_trans:
                flux1 = (self.dkappa[..., i] / self.alpha[..., i]
                         * self.dt_[..., i] / self.alpha[..., i])
                flux2 = (self.kappa_new[..., i + 1]
                         * (self.dt_down[..., i] / self.gamma[..., i]
                            - self.dt_up[..., i] / self.beta[..., i])
                         / (0.5 * self.alpha[..., i]))
            else:
                zb = self.bexp * (self.zsum[..., i + 1] + 1.0)
                flux1 = (self.dkappa[..., i] / 2.0 * self.dt_[..., i]
                         / 2.0 / zb / zb)
                flux2 = (self.kappa_new[..., i + 1]
                         * ((self.dt_down[..., i] - self.dt_up[..., i])
                            / zb / zb - self.dt_[..., i] / 2.0
                            / (self.bexp * (self.zsum[..., i + 1] + 1.0)
                               * (self.zsum[..., i + 1] + 1.0))))
            flux = flux1 + flux2
            phase = (frozen.CONST_RHOICE * frozen.CONST_LATICE
                     * (self.ice_new[..., i + 1] - self.ice[..., i + 1])
                     / self.deltat)
            res[..., i] = flux + phase - storage
        return res

    def eval_all(self, x):
        """The full residual (focus -1). Returns [..., n]."""
        res = torch.zeros_like(x)
        self._refresh_props(x, 0, self.n, True)
        self._diffs_and_dkappa(x, 0, self.n - 1)
        return self._rows(x, 0, self.n - 1, res)

    def eval_focus(self, x, j):
        """The residual about column j (focus j). Returns [..., n],
        fresh on the window, zeros elsewhere (the C leaves the rest
        stale; only the window is ever read)."""
        res = torch.zeros_like(x)
        left = 0 if j == 0 else j - 1
        right = self.n - 1 if j == self.n - 1 else j + 1
        self._refresh_props(x, left + 1, right + 1, False)
        self._diffs_and_dkappa(x, left, right)
        return self._rows(x, left, right, res)


def _fdjac3(x, fvec, eqn, n):
    """Forward-difference Jacobian, tridiagonal part.  :90-126.

    `x` is [..., n] (unperturbed on return), `fvec` its residual.
    Returns (a, b, c), each [..., n]; a[0] and c[n-1] are zeros,
    which tridiag never reads.
    """
    dev, dt = x.device, x.dtype
    shape = x.shape[:-1]
    a = torch.zeros(shape + (n,), dtype=dt, device=dev)
    b = torch.zeros_like(a)
    c = torch.zeros_like(a)
    for j in range(n):
        temp = x[..., j]
        h = NEWT_RAPH_EPS2 * temp.abs()
        h = torch.where(h == 0.0, torch.full_like(h, NEWT_RAPH_EPS2), h)
        pert = x.clone()
        pert[..., j] = temp + h
        # :111 -- h is re-measured after the rounding.
        h = pert[..., j] - temp
        f = eqn.eval_focus(pert, j)
        b[..., j] = (f[..., j] - fvec[..., j]) / h
        if j != 0:
            c[..., j - 1] = (f[..., j - 1] - fvec[..., j - 1]) / h
        if j != n - 1:
            a[..., j + 1] = (f[..., j + 1] - fvec[..., j + 1]) / h
    return a, b, c


def _newt_raph(eqn, x, n):
    """The implicit profile solve.  newt_raph, :18-84.

    `x` is [..., n], starting at the previous step's nodes. Returns
    (x, error): the solved unknowns and, per [...], whether the
    iteration exhausted NEWT_RAPH_MAXTRIAL trials without meeting
    TOLF or TOLX (Error 1: the caller falls back to the explicit
    sweep). Columns are independent -- a converged column holds its
    values while the rest iterate -- so masking reproduces the
    scalar loop exactly.
    """
    done = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
    for k in range(NEWT_RAPH_MAXTRIAL):
        if bool(done.all()):
            break
        fvec = eqn.eval_all(x)
        # Sequential, as the C accumulates it: a tree reduction rounds
        # differently, and trials have been seen to stop one iteration
        # apart over a single ulp at TOLF.
        errf = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        for i in range(n):
            errf = errf + fvec[..., i].abs()
        done = done | (errf <= NEWT_RAPH_TOLF)
        if bool(done.all()):
            break
        a, b, c = _fdjac3(x, fvec, eqn, n)
        p = tridiag(a, b, c, -fvec, n)
        errx = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        for i in range(n):
            errx = errx + p[..., i].abs()
        in_range = (x < NEWT_RAPH_R_MAX) & (x > NEWT_RAPH_R_MIN)
        if 10 < k <= 20:
            step = torch.where(in_range,
                               torch.full_like(p, NEWT_RAPH_RELAX1),
                               torch.ones_like(p))
        elif 20 < k <= 60:
            step = torch.where(in_range,
                               torch.full_like(p, NEWT_RAPH_RELAX2),
                               torch.ones_like(p))
        elif k > 60:
            step = torch.where(in_range,
                               torch.full_like(p, NEWT_RAPH_RELAX3),
                               torch.ones_like(p))
        else:
            step = torch.ones_like(p)
        live = ~done
        x = torch.where(live.unsqueeze(-1), x + p * step, x)
        done = done | (live & (errx <= NEWT_RAPH_TOLX))
    return x, ~done


#: ATen's TensorIterator GRAIN_SIZE (Parallel.h): an elementwise call of
#: at least this many elements is split across threads.
_GRAIN = 32768


def _rowcall(fn, *args):
    """`fn(*args)` on [k, C] rows, rounded as k calls on contiguous [C].

    The per-node solver applies `pow` and `log10` to one contiguous [C]
    node slice at a time, and ATen's CPU kernels round the vectorized
    body of a call (SLEEF) and its scalar tail (libm, the last C mod
    2*lanes elements) differently: which rounding an element gets
    depends on where it sits in its call. One call over all k*C
    elements would move the tails. A view with a padded row stride
    keeps TensorIterator from coalescing the rows, so each row is an
    inner loop of exactly C elements, split as the single call was;
    at a size TensorIterator would thread, the rows are called one by
    one. Other devices round every element alike: one call.
    """
    ref = next(a for a in args if torch.is_tensor(a))
    k, c = ref.shape
    if ref.device.type != 'cpu':
        return fn(*args)
    if k * c < _GRAIN:
        pad = []
        for a in args:
            if torch.is_tensor(a):
                v = torch.empty((k, c + 1), dtype=a.dtype,
                                device=a.device)[:, :c]
                v.copy_(a)
                a = v
            pad.append(a)
        return fn(*pad)
    return torch.stack([fn(*[a.expand(k, c)[i].contiguous()
                             if torch.is_tensor(a) else a for a in args])
                        for i in range(k)])


def _rpow(a, b):
    return _rowcall(frozen._pow, a, b)


def _rlog10(a):
    return _rowcall(torch.log10, a)


def _newt_raph_fast(eqn, x, n, cols=None, statics=None):
    """`_newt_raph`, every node and every Jacobian column at once.

    `cols` ([...] bool, or None for all) are the columns whose answer
    is wanted; the others are not iterated and come back as failed
    (unchanged, error set). Columns are independent, so the wanted
    ones are unaffected. On the CPU the trials run as one compiled
    loop per column (torchvic.core.newtc), the same arithmetic.

    Bit for bit the per-node solver's answer (tests/
    test_implicit_profile.py holds the two equal): the same Newton
    trials, stopping rule, relaxation and tridiagonal sweep, the same
    expressions element by element, with the transcendental calls
    rounded as the per-node ones are (_rowcall). What changes is the
    order the elements are visited in, which elementwise arithmetic
    does not see.

    fda_heat_eqn's statics (frozen_soil.c:624, _FdaHeatEqn) carry state
    from one Jacobian column to the next: a column refreshes the
    properties of nodes j..j+2 only where their ice changed and reads
    the rest as earlier columns left them. Unrolled, node m holds, at
    column j, its full evaluation's value S0 = changed ? new : step's
    (j <= m-2), the value after its own perturbed column P = changed_p
    ? new_p : S0 (j = m-1), or after the column above it Q = changed
    ? new : P (j >= m) -- so every column's window is a selection from
    three per-node arrays, and all columns are evaluated together.
    Node 0 is never refreshed (kappa[0]); node n+1 without NOFLUX is
    never assigned (the static zero).
    """
    shape = x.shape[:-1]
    c = x[..., 0].numel()
    dev, dt = x.device, x.dtype

    def rows(t):
        """[..., k] -> [k, C]."""
        t = torch.as_tensor(t, dtype=dt, device=dev)
        t = t.expand(shape + t.shape[-1:])
        return t.reshape(c, t.shape[-1]).transpose(0, 1).contiguous()

    def flat(t):
        """[...] (or a scalar) -> [1, C] (or the scalar)."""
        if not torch.is_tensor(t) or t.dim() == 0:
            return t
        return t.expand(shape).reshape(1, c)

    # The layer of each node, as _refresh_props walks it (:745-789).
    last = eqn.nlayers - 1
    g = lambda t, i: t.gather(-1, i.unsqueeze(-1)).squeeze(-1)
    lidx = torch.zeros(shape, dtype=torch.int64, device=dev)
    lsum = torch.zeros(shape, dtype=dt, device=dev)
    past = torch.zeros(shape, dtype=torch.bool, device=dev)
    lis = []
    for i in range(n + 1):
        li = lidx.clamp(max=last)
        lis.append(li)
        adv = (eqn.zsum[..., i] > lsum + g(eqn.depth, li)) & ~past
        lsum = torch.where(adv, lsum + g(eqn.depth, li), lsum)
        lidx = torch.where(adv, lidx + 1, lidx)
        now_past = adv & (lidx == eqn.nlayers)
        past = past | now_past
        lidx = torch.where(now_past, torch.full_like(lidx, last), lidx)
    li = torch.stack(lis[1:], -1)                       # nodes 1..n

    def lay(t):
        t = t.expand(shape + t.shape[-1:])
        return rows(t.gather(-1, li))

    sdm, bdm = lay(eqn.soil_dens_min), lay(eqn.bulk_dens_min)
    qz, sd = lay(eqn.quartz), lay(eqn.soil_density)
    bd, org = lay(eqn.bulk_density), lay(eqn.organic)
    nod = slice(1, n + 1)
    moist, maxm = rows(eqn.moist)[nod], rows(eqn.max_moist)[nod]
    bub, expt = rows(eqn.bubble)[nod], rows(eqn.expt)[nod]
    ice_n, kap = rows(eqn.ice), rows(eqn.kappa)
    cs_n, t0 = rows(eqn.cs), rows(eqn.t0)
    alpha, beta = rows(eqn.alpha), rows(eqn.beta)
    gamma, zsum = rows(eqn.gamma), rows(eqn.zsum)
    ts, tb = flat(eqn.ts), flat(eqn.tb)
    deltat = flat(eqn.deltat)
    bexp = flat(eqn.bexp) if eqn.exp_trans else None
    two = lambda v: torch.cat([v, v])
    moist2, maxm2, bub2 = two(moist), two(maxm), two(bub)
    # maximum_unfrozen_water's exponent and soil_conductivity's terms
    # that do not depend on the unfrozen water: fixed for the solve.
    expo2 = two(-(2.0 / (expt - 3.0)))
    ice2 = two(ice_n[nod])
    fixed2 = [two(v) for v in frozen.soil_conductivity_fixed(
        moist, sdm, bdm, qz, sd, bd, org, pow=_rpow, log10=_rlog10)]
    soil2, org2 = two(bd / sd), two(org)
    solid2 = frozen.volumetric_heat_capacity_solid(soil2, org2)
    zeros2 = torch.zeros_like(moist2)
    if statics is not None:
        statics.append((shape, c, fixed2, soil2, solid2, deltat, bexp))
        return None
    if cols is not None:
        cols = cols.expand(shape).reshape(c)
    if newtc.available(x):
        return _newt_raph_c(eqn, x, n, shape, c, cols, fixed2, soil2,
                            solid2, deltat, bexp)

    def props(t):
        """ice/changed/kappa/cs of nodes 1..n at node temperatures
        `t` ([2n, C], unperturbed then perturbed).  :745-789."""
        ic = moist2 - frozen.maximum_unfrozen_water(
            t, maxm2, bub2, None, pow=_rpow, expo=expo2)
        ic = torch.where(ic < 0.0, zeros2, ic)
        ic = torch.where(t < 0.0, ic, zeros2)
        changed = ic != ice2
        wu = moist2 - ic
        kn = frozen.soil_conductivity_wu(fixed2, moist2, wu, pow=_rpow)
        cn = frozen.volumetric_heat_capacity(soil2, wu, ic, org2,
                                             solid=solid2)
        return ic, changed, kn, cn

    # The rows evaluated each trial: the full residual (focus -1),
    # then per column j its window rows j, j-1, j+1 (:90-126).
    # (row r, column j or -1).
    plan = ([(r, -1) for r in range(n)] + [(j, j) for j in range(n)]
            + [(j - 1, j) for j in range(1, n)]
            + [(j + 1, j) for j in range(n - 1)])
    nk = n + 2

    def ver(m, j):
        """Which of Q (0), P (1), S0 (2) node m holds at column j."""
        if j < 0:
            return 2
        return 0 if m <= j else (1 if m == j + 1 else 2)

    def xi(q, j):
        """Index of unknown q (-1: ts, n: tb) in [ts, x, x_pert, tb]."""
        if q < 0:
            return 0
        if q >= n:
            return 2 * n + 1
        return 1 + n + q if q == j else 1 + q

    up = lambda r: r + 1 if (eqn.noflux and r == n - 1) else r + 2
    idx = lambda f: torch.tensor([f(r, j) for r, j in plan],
                                 dtype=torch.int64, device=dev)
    i_xc = idx(lambda r, j: xi(r, j))
    i_xl = idx(lambda r, j: xi(r - 1, j))
    i_xr = idx(lambda r, j: xi(r + 1, j))
    i_klo = idx(lambda r, j: ver(r, j) * nk + r)
    i_kmid = idx(lambda r, j: ver(r + 1, j) * nk + r + 1)
    i_kup = idx(lambda r, j: ver(up(r), j) * nk + up(r))
    i_cs = idx(lambda r, j: ver(r + 1, j) * (n + 1) + r + 1)
    i_ice = idx(lambda r, j: (n + r) if r + 1 == j + 1 else r)
    rr = idx(lambda r, j: r)
    t0r, csr = t0.index_select(0, rr + 1), cs_n.index_select(0, rr + 1)
    icer = ice_n.index_select(0, rr + 1)
    ar, br = alpha.index_select(0, rr), beta.index_select(0, rr)
    gr, zr = gamma.index_select(0, rr), zsum.index_select(0, rr + 1)
    half_a = 0.5 * ar
    if eqn.exp_trans:
        zb = bexp * (zr + 1.0)
        zz = bexp * (zr + 1.0) * (zr + 1.0)
    k0 = kap[0:1]
    zero = torch.zeros_like(k0)
    zcs = torch.zeros_like(k0)

    def resid(xall, kall, csall, iceall):
        """The residual rows of `plan`.  :792-864."""
        xc = xall.index_select(0, i_xc)
        xl = xall.index_select(0, i_xl)
        xr = xall.index_select(0, i_xr)
        dt_ = xr - xl
        dt_up = xc - xl
        dt_down = xr - xc
        klo = kall.index_select(0, i_klo)
        kmid = kall.index_select(0, i_kmid)
        dkappa = kall.index_select(0, i_kup) - klo
        csn = csall.index_select(0, i_cs)
        storage = (csn * (xc - t0r) / deltat
                   + xc * (csn - csr) / deltat)
        if not eqn.exp_trans:
            flux1 = dkappa / ar * dt_ / ar
            flux2 = kmid * (dt_down / gr - dt_up / br) / half_a
        else:
            flux1 = dkappa / 2.0 * dt_ / 2.0 / zb / zb
            flux2 = kmid * ((dt_down - dt_up) / zb / zb
                            - dt_ / 2.0 / zz)
        flux = flux1 + flux2
        phase = (frozen.CONST_RHOICE * frozen.CONST_LATICE
                 * (iceall.index_select(0, i_ice) - icer) / deltat)
        return flux + phase - storage

    def tri(a, b, cc, r):
        """tridiag, :132-169, on [n, C] rows; the C's writes to a and
        to b (set to one, then multiplied by) change no value."""
        cc, r = list(cc.unbind(0)), list(r.unbind(0))
        a, b = a.unbind(0), b.unbind(0)
        factor = b[0]
        cc[0] = cc[0] / factor
        r[0] = r[0] / factor
        for i in range(1, n):
            factor = a[i]
            bi = b[i] - cc[i - 1] * factor
            r[i] = r[i] - r[i - 1] * factor
            cc[i] = cc[i] / bi
            r[i] = r[i] / bi
        for j in range(n - 2, -1, -1):
            r[j] = r[j] - r[j + 1] * cc[j]
        return torch.stack(r)

    xs = rows(x)
    done = (torch.zeros((c,), dtype=torch.bool, device=dev) if cols is None
            else ~cols)
    # Asking whether every column is done costs a device-to-host sync;
    # off the CPU it is asked every GPU_DONE_EVERY trials. A trial run
    # after the last column stopped changes nothing (every update is
    # masked by `done`), so the answer is the same.
    every = 1 if dev.type == 'cpu' else GPU_DONE_EVERY
    for k in range(NEWT_RAPH_MAXTRIAL):
        if k % every == 0 and bool(done.all()):
            break
        h = NEWT_RAPH_EPS2 * xs.abs()
        h = torch.where(h == 0.0, torch.full_like(h, NEWT_RAPH_EPS2), h)
        xp = xs + h
        h = xp - xs                                     # :111
        ic, ch, kn, cn = props(torch.cat([xs, xp]))
        icu, chu, knu = ic[:n], ch[:n], kn[:n]
        chp, knp, cnu, cnp = ch[n:], kn[n:], cn[:n], cn[n:]
        s0 = torch.where(chu, knu, kap[nod])
        p_ = torch.where(chp, knp, s0)
        q_ = torch.where(chu, knu, p_)
        cs0 = torch.where(chu, cnu, cs_n[nod])
        cp_ = torch.where(chp, cnp, cs0)
        cq_ = torch.where(chu, cnu, cp_)
        kall = torch.cat([k0, q_, zero, k0, p_, zero, k0, s0, zero])
        csall = torch.cat([zcs, cq_, zcs, cp_, zcs, cs0])
        xall = torch.cat([ts.expand(1, c), xs, xp, tb.expand(1, c)])
        f = resid(xall, kall, csall, ic)
        fvec = f[:n]
        # Sequential from zero, as the C accumulates (cumsum along
        # the node rows is that sum).
        errf = torch.cumsum(fvec.abs(), 0)[-1]
        done = done | (errf <= NEWT_RAPH_TOLF)
        if every == 1 and bool(done.all()):
            break
        b = (f[n:2 * n] - fvec) / h
        cc = torch.cat([(f[2 * n:3 * n - 1] - fvec[:n - 1]) / h[1:],
                        zero])
        a = torch.cat([zero, (f[3 * n - 1:] - fvec[1:]) / h[:n - 1]])
        p = tri(a, b, cc, -fvec)
        errx = torch.cumsum(p.abs(), 0)[-1]
        in_range = (xs < NEWT_RAPH_R_MAX) & (xs > NEWT_RAPH_R_MIN)
        if 10 < k <= 20:
            step = torch.where(in_range,
                               torch.full_like(p, NEWT_RAPH_RELAX1),
                               torch.ones_like(p))
        elif 20 < k <= 60:
            step = torch.where(in_range,
                               torch.full_like(p, NEWT_RAPH_RELAX2),
                               torch.ones_like(p))
        elif k > 60:
            step = torch.where(in_range,
                               torch.full_like(p, NEWT_RAPH_RELAX3),
                               torch.ones_like(p))
        else:
            step = torch.ones_like(p)
        live = ~done
        xs = torch.where(live, xs + p * step, xs)
        done = done | (live & (errx <= NEWT_RAPH_TOLX))
    out = xs.transpose(0, 1).reshape(shape + (n,))
    err = ~done if cols is None else (~done | ~cols)
    return out, err.reshape(shape)


def _c_arrays(eqn, n, shape, c, fixed2, soil2, solid2, deltat, bexp):
    """newtc's per-column arrays ([C, k] contiguous, keys newtc._ORDER)
    for the full batch: the equation's profiles and the terms fixed for
    the solve, as _newt_raph_fast forms them."""
    nn = eqn.t0.shape[-1]
    dt = torch.float64
    dev = eqn.t0.device

    def col(t, k):
        """[..., >= k] -> [C, k]."""
        t = device_constant(t, dt, dev)
        return t.expand(shape + t.shape[-1:])[..., :k].reshape(
            c, k).contiguous()

    def rowcol(t):
        """[n, C] (nodes 1..n) -> [C, n]."""
        return t[:n].transpose(0, 1).contiguous()

    def scal(t):
        """A scalar or [1, C] (_newt_raph_fast's `flat`) -> [C]."""
        t = device_constant(t, dt, dev)
        return (t.reshape(c) if t.dim() else t.expand(c)).contiguous()

    kdry, por, sr, ksu, keu, kspor = fixed2
    return dict(t0=col(eqn.t0, nn), ice=col(eqn.ice, nn),
                kap=col(eqn.kappa, nn), cs=col(eqn.cs, nn),
                moist=col(eqn.moist, nn), maxm=col(eqn.max_moist, nn),
                bub=col(eqn.bubble, nn), expt=col(eqn.expt, nn),
                zsum=col(eqn.zsum, nn), al=col(eqn.alpha, n),
                be=col(eqn.beta, n), ga=col(eqn.gamma, n),
                kdry=rowcol(kdry), por=rowcol(por), sr=rowcol(sr),
                ksu=rowcol(ksu), keu=rowcol(keu), kspor=rowcol(kspor),
                soil=rowcol(soil2), solid=rowcol(solid2),
                dt=scal(deltat),
                bexp=(scal(bexp) if bexp is not None
                      else torch.zeros(c, dtype=dt, device=dev)))


def _newt_consts():
    return (NEWT_RAPH_MAXTRIAL, NEWT_RAPH_TOLX, NEWT_RAPH_TOLF,
            NEWT_RAPH_R_MAX, NEWT_RAPH_R_MIN, NEWT_RAPH_RELAX1,
            NEWT_RAPH_RELAX2, NEWT_RAPH_RELAX3, NEWT_RAPH_EPS2,
            frozen.CONST_RHOICE * frozen.CONST_LATICE)


def _newt_raph_c(eqn, x, n, shape, c, cols, fixed2, soil2, solid2,
                 deltat, bexp):
    """_newt_raph_fast's trials by torchvic.core.newtc, on the wanted
    columns only; the terms fixed for the solve as the torch loop forms
    them (on the full batch, so their transcendentals round alike)."""
    idx = (torch.arange(c) if cols is None
           else cols.nonzero().reshape(-1))
    xo = x.expand(shape + (n,)).reshape(c, n).clone()
    err = torch.ones(c, dtype=torch.bool)
    if idx.numel():
        arrs = _c_arrays(eqn, n, shape, c, fixed2, soil2, solid2, deltat,
                         bexp)
        arrs = {k: v.index_select(0, idx).contiguous()
                for k, v in arrs.items()}
        xs, e = newtc.newt_raph(arrs, xo.index_select(0, idx), n,
                                eqn.t0.shape[-1], eqn.noflux, eqn.exp_trans,
                                idx, c, _newt_consts())
        xo[idx] = xs
        err[idx] = e
    return xo.reshape(shape + (n,)), err.reshape(shape)


def _fast_statics(eqn, n):
    """_newt_raph_fast's terms fixed for a solve, for the compiled
    profile solve: (shape, C, fixed2, soil2, solid2, deltat, bexp)."""
    x = eqn.t0[..., 1:n + 1]
    out = []
    _newt_raph_fast(eqn, x, n, statics=out)
    return out[0]


def _solve_implicit_c(ts, t_node, moist, ice, kappa, cs, porosity_node,
                      bubble_node, expt_node, alpha, beta, gamma, zsum,
                      deltat, dp, bulk_dens_min, soil_dens_min, quartz,
                      bulk_density, soil_density, organic, depth, nlayers,
                      nnodes, noflux, exp_trans, tfallback, cols, cache):
    """solve_profile_implicit by torchvic.core.newtc, the whole solve in
    one call: Newton, the boundaries and the cold nose (:235-322).

    Everything but the trial surface temperature is the step's, so the
    per-column arrays are built once per surface solve and held in
    `cache` (a dict that lives as long as those inputs); the trial
    replaces node 0. Returns (t_new, error) as solve_profile_implicit.
    """
    n = nnodes - 1 if noflux else nnodes - 2
    key = (nnodes, noflux, exp_trans)
    ent = cache.get(key)
    if ent is None or ent[0] is not t_node:
        t0 = t_node[..., :nnodes]
        eqn = _FdaHeatEqn(t0, moist, ice, kappa, cs, porosity_node,
                          bubble_node, expt_node, alpha, beta, gamma, zsum,
                          deltat, dp, bulk_dens_min, soil_dens_min, quartz,
                          bulk_density, soil_density, organic, depth,
                          nlayers, n, noflux, exp_trans)
        shape, c, fixed2, soil2, solid2, dtf, bexp = _fast_statics(eqn, n)
        arrs = _c_arrays(eqn, n, shape, c, fixed2, soil2, solid2, dtf,
                         bexp)
        keep = [arrs[k] for k in newtc._ORDER]
        # ATen's CUDA division by a CPU scalar is a multiplication by
        # its reciprocal (newtc._CU_SRC): / deltat in the torch solver
        dt_recip = (not torch.is_tensor(dtf)
                    or (dtf.dim() == 0 and dtf.device.type == 'cpu'))
        ent = (t_node, shape, c, keep, dt_recip)
        cache[key] = ent
    _, shape, c, keep, dt_recip = ent
    tsv = ts.expand(shape).reshape(c).to(torch.float64).contiguous()
    if tsv.device.type == 'cuda':
        # one kernel, the wanted columns masked in it (no host sync)
        t_new, err = newtc.profile_cuda(
            keep, tsv, None if cols is None else cols.expand(shape),
            c, n, nnodes, noflux, exp_trans, tfallback, dt_recip,
            _newt_consts())
        return t_new.reshape(shape + (nnodes,)), err.reshape(shape)
    idx = (torch.arange(c) if cols is None
           else cols.expand(shape).reshape(c).nonzero().reshape(-1))
    t_new, err = newtc.profile(keep, tsv, idx, c, n, nnodes, noflux,
                               exp_trans, tfallback, _newt_consts())
    return t_new.reshape(shape + (nnodes,)), err.reshape(shape)


def solve_profile_implicit(ts, t0, moist, ice, kappa, cs, porosity_node,
                           bubble_node, expt_node, alpha, beta, gamma, zsum,
                           deltat, dp, bulk_dens_min, soil_dens_min, quartz,
                           bulk_density, soil_density, organic, depth,
                           nlayers, nnodes, noflux, exp_trans,
                           tfallback=True, cols=None):
    """The profile by Newton-Raphson.  solve_T_profile_implicit, :235-322.

    All profiles are [..., nnodes]; layer fields [..., layer]. `t0`
    is the profile the step opened with, with node 0 already at the
    trial surface temperature. Returns (t_new, error): the solved
    [..., nnodes] profile and, per [...], whether Newton failed
    (the caller then runs the explicit sweep).

    With a single unknown (!NOFLUX, 3 nodes) VIC reads T_2[1] past
    the unknowns (:793-796) -- a value fda_heat_eqn's init never
    sets, i.e. uninitialized stack -- so that grid is refused here
    (and at the global file, before a run starts) rather than
    reproduced.
    """
    if not noflux and nnodes - 2 < 2:
        raise ValueError(
            'IMPLICIT with %d nodes and NOFLUX FALSE solves for a single '
            'node, where VIC reads T_2[1] past its unknowns '
            '(frozen_soil.c:793-796): uninitialized memory, not physics. '
            'Use NOFLUX TRUE, NODES >= 4, or IMPLICIT FALSE' % nnodes)
    n = nnodes - 1 if noflux else nnodes - 2
    eqn = _FdaHeatEqn(t0, moist, ice, kappa, cs, porosity_node,
                      bubble_node, expt_node, alpha, beta, gamma, zsum,
                      deltat, dp, bulk_dens_min, soil_dens_min, quartz,
                      bulk_density, soil_density, organic, depth, nlayers,
                      n, noflux, exp_trans)
    # :729-731 -- the unknowns start at the previous step's nodes.
    x = t0[..., 1:n + 1].clone()
    if FAST_NEWTON and n >= 2:
        x, error = _newt_raph_fast(eqn, x, n, cols)
    else:
        x, error = _newt_raph(eqn, x, n)
    t_new = t0.clone()
    t_new[..., 1:n + 1] = torch.where(error.unsqueeze(-1),
                                      t_new[..., 1:n + 1], x)
    ok = ~error
    # :296-300 -- the boundaries, on success only.
    t_new[..., 0] = torch.where(ok, t0[..., 0], t_new[..., 0])
    if not noflux:
        t_new[..., nnodes - 1] = torch.where(
            ok, t0[..., nnodes - 1], t_new[..., nnodes - 1])
    if tfallback:
        # :301-318 -- the runaway cold nose: a node colder (or warmer)
        # than both neighbours, grown more so, is replaced by their
        # mean. In place, ascending, as in the C.
        for j in range(1, nnodes - 1):
            dl0 = t0[..., j - 1] - t0[..., j]
            du0 = t0[..., j + 1] - t0[..., j]
            dl = t_new[..., j - 1] - t_new[..., j]
            du = t_new[..., j + 1] - t_new[..., j]
            nose = (((dl0 > 0) & (du0 > 0) & (dl - dl0 > 0)
                     & (du - du0 > 0))
                    | ((dl0 < 0) & (du0 < 0) & (dl - dl0 < 0)
                       & (du - du0 < 0)))
            t_new[..., j] = torch.where(
                ok & nose, 0.5 * (t_new[..., j - 1] + t_new[..., j + 1]),
                t_new[..., j])
    return t_new, error


def solve_profile(ts, t_node, moist_node, ice_node, kappa_node, cs_node,
                  porosity_node, bubble_node, expt_node, alpha, beta, gamma,
                  zsum, deltat, dp, bulk_dens_min, soil_dens_min, quartz,
                  bulk_density, soil_density, organic, depth, nlayers,
                  nnodes, implicit, noflux, exp_trans, fs_active,
                  frozen_soil, tfallback=True, tmp_nnodes=None, cols=None,
                  cache=None):
    """The soil temperature profile at one trial Ts.
    func_surf_energy_bal.c:410-446.

    `ts` is [...] (the trial surface temperature), `t_node` the [...,
    nnodes] profile the step opened with; the node properties are
    [..., nnodes] (the step's, from distribute_node_moisture), the
    layer fields [..., layer]. Returns the solved [..., nnodes]
    profile: the implicit Newton solve with IMPLICIT, else the
    explicit sweep, with the explicit sweep as the fallback when
    Newton fails (Error 1). FIRST_SOLN's coefficient caching is a
    no-op across trials -- the properties do not change within a
    surface solve -- so the coefficients are simply recomputed.

    `tmp_nnodes` (QUICK_SOLVE's reduced count, None or [...] int64)
    solves only the head of the column; the tail is the step's
    profile, which nothing reads on a trial (func_surf_energy_bal.c
    reads nodes 1 and 2 without EXP_TRANS, and VIC refuses
    QUICK_SOLVE with EXP_TRANS).

    Gradients flow through the explicit sweep (unrolled, as in
    torchvic.core.frozen) but not through the Newton trajectory,
    which runs under no_grad: at a non-converged point there is no
    root to differentiate (see torchvic.core.frozen's note on the
    implicit solver's measured non-convergence), and unrolling up to
    150 Newton trials per surface trial is not a gradient worth
    having. The values are VIC's either way.

    `cols` ([...] bool, None for all), with `only`'s mask, are the
    columns whose implicit solve is wanted: the rest (padding tiles,
    elements whose surface search has stopped) are not iterated and
    hold the profile they opened with. `cache`, a dict living as long
    as everything here but `ts` (one surface solve), lets the compiled
    implicit solve build its per-column arrays once.
    """
    if implicit:
        act = ACTIVE
        if act is not None and act.shape == ts.shape:
            cols = act if cols is None else (cols & act)
        if cols is not None:
            cols = cols.expand(ts.shape)
    else:
        cols = None
    # Per-cell fields ([cell], [cell, k]) onto [...] (usually
    # [cell, tile]); per-column fields pass through.
    bc = lambda x: _bc(ts.shape, x, vec=True)     # per node/layer
    dp = _bc(ts.shape, dp)
    fs_active = _bc(ts.shape, fs_active)
    depth = bc(depth)
    bulk_dens_min = bc(bulk_dens_min)
    soil_dens_min = bc(soil_dens_min)
    quartz = bc(quartz)
    bulk_density = bc(bulk_density)
    soil_density = bc(soil_density)
    organic = bc(organic)
    zsum = bc(zsum)
    porosity_node = bc(porosity_node)
    bubble_node = bc(bubble_node)
    expt_node = bc(expt_node)
    alpha = bc(alpha)
    beta = bc(beta)
    gamma = bc(gamma)
    if tmp_nnodes is None:
        return _solve_profile_n(
            ts, t_node, moist_node, ice_node, kappa_node, cs_node,
            porosity_node, bubble_node, expt_node, alpha, beta, gamma,
            zsum, deltat, dp, bulk_dens_min, soil_dens_min, quartz,
            bulk_density, soil_density, organic, depth, nlayers, nnodes,
            implicit, noflux, exp_trans, fs_active, frozen_soil,
            tfallback, cols, cache)
    # QUICK_SOLVE: per-column counts, solved per distinct count (the
    # columns are independent, so the grouping is exact).
    vals = sorted(set(int(v) for v in tmp_nnodes.reshape(-1).tolist()))
    t0 = t_node.clone()
    t0[..., 0] = ts
    out = t0.clone()
    for v in vals:
        m = tmp_nnodes == v
        got = _solve_profile_n(
            ts[m], t_node[m], moist_node[m], ice_node[m], kappa_node[m],
            cs_node[m], porosity_node[m], bubble_node[m], expt_node[m],
            alpha[m], beta[m], gamma[m], zsum[m], deltat, dp[m],
            bulk_dens_min[m], soil_dens_min[m], quartz[m],
            bulk_density[m], soil_density[m], organic[m], depth[m],
            nlayers, v, implicit, noflux, exp_trans, fs_active[m],
            frozen_soil, tfallback, None if cols is None else cols[m])
        put = out[m].clone()
        put[..., :v] = got
        out[m] = put
    return out


def _solve_profile_n(ts, t_node, moist_node, ice_node, kappa_node, cs_node,
                     porosity_node, bubble_node, expt_node, alpha, beta,
                     gamma, zsum, deltat, dp, bulk_dens_min, soil_dens_min,
                     quartz, bulk_density, soil_density, organic, depth,
                     nlayers, n, implicit, noflux, exp_trans, fs_active,
                     frozen_soil, tfallback, cols=None, cache=None):
    """solve_profile for one node count."""
    t0 = t_node[..., :n].clone()
    t0[..., 0] = ts
    sl = lambda t: t[..., :n]
    if implicit:
        with torch.no_grad():
            if (cache is not None and FAST_NEWTON
                    and (n - 1 if noflux else n - 2) >= 2
                    and (newtc.available(t0)
                         or newtc.cuda_available(
                             t0, n - 1 if noflux else n - 2))):
                t_new, error = _solve_implicit_c(
                    ts, t_node, sl(moist_node), sl(ice_node),
                    sl(kappa_node), sl(cs_node), sl(porosity_node),
                    sl(bubble_node), sl(expt_node), alpha[..., :n - 1],
                    beta[..., :n - 1], gamma[..., :n - 1], sl(zsum),
                    deltat, dp, bulk_dens_min, soil_dens_min, quartz,
                    bulk_density, soil_density, organic, depth, nlayers,
                    n, noflux, exp_trans, tfallback, cols, cache)
            else:
                t_new, error = solve_profile_implicit(
                    ts, t0, sl(moist_node), sl(ice_node), sl(kappa_node),
                    sl(cs_node), sl(porosity_node), sl(bubble_node),
                    sl(expt_node), alpha[..., :n - 1], beta[..., :n - 1],
                    gamma[..., :n - 1], sl(zsum), deltat, dp,
                    bulk_dens_min, soil_dens_min, quartz, bulk_density,
                    soil_density, organic, depth, nlayers, n, noflux,
                    exp_trans, tfallback, cols=cols)
        # The sweep for the failed solves (:258-262); a column not
        # wanted keeps its failed solve.
        need = error if cols is None else error & cols
        if _graph():
            # the CUDA graph path: the sweep on every column, kept where
            # it is needed (what the subset below gives those columns;
            # the rest keep the Newton solve, as when none is needed)
            exp_new, _, _ = frozen.solve_T_profile(
                t0, sl(zsum), sl(kappa_node), sl(cs_node), sl(moist_node),
                sl(porosity_node), sl(bubble_node), sl(expt_node),
                sl(ice_node), alpha[..., :n - 1], beta[..., :n - 1],
                gamma[..., :n - 1], deltat, dp, n, fs_active=fs_active,
                noflux=noflux, exp_trans=exp_trans, frozen_soil=frozen_soil,
                tfallback=tfallback)
            return torch.where(need.unsqueeze(-1), exp_new, t_new)
        if bool(need.any()):
            args = [t0, sl(zsum), sl(kappa_node), sl(cs_node),
                    sl(moist_node), sl(porosity_node), sl(bubble_node),
                    sl(expt_node), sl(ice_node), alpha[..., :n - 1],
                    beta[..., :n - 1], gamma[..., :n - 1]]
            kw = dict(fs_active=fs_active, noflux=noflux,
                      exp_trans=exp_trans, frozen_soil=frozen_soil,
                      tfallback=tfallback)
            if newtc.position_free(t0.device) and not bool(need.all()):
                # Only the failed columns: every operation of the sweep
                # rounds an element alike wherever it sits here, and
                # the columns are independent.
                shape = ts.shape
                c = ts.numel()
                idx = need.reshape(-1).nonzero().reshape(-1)
                pick = lambda t: t.expand(shape + t.shape[len(shape):]) \
                    .reshape((c,) + t.shape[len(shape):]).index_select(0, idx)
                pc = lambda v: (pick(v) if torch.is_tensor(v) and v.dim()
                                else v)
                kw['fs_active'] = pc(_bc(shape, fs_active))
                kw['frozen_soil'] = pc(_bc(shape, frozen_soil))
                exp_c, _, _ = frozen.solve_T_profile(
                    *[pick(a) for a in args], pc(_bc(shape, deltat)),
                    pc(_bc(shape, dp)), n,
                    **kw)
                t_new = t_new.clone()
                t_new.view(c, n)[idx] = exp_c.to(t_new.dtype)
            else:
                exp_new, _, _ = frozen.solve_T_profile(
                    *args, deltat, dp, n, **kw)
                t_new = torch.where(need.unsqueeze(-1), exp_new, t_new)
        return t_new
    t_new, _, _ = frozen.solve_T_profile(
        t0, sl(zsum), sl(kappa_node), sl(cs_node), sl(moist_node),
        sl(porosity_node), sl(bubble_node), sl(expt_node), sl(ice_node),
        alpha[..., :n - 1], beta[..., :n - 1], gamma[..., :n - 1],
        deltat, dp, n, fs_active=fs_active, noflux=noflux,
        exp_trans=exp_trans, frozen_soil=frozen_soil, tfallback=tfallback)
    return t_new


# The implicit solve's per-column arrays and its CUDA kernel run as eager
# kernels inside the CUDA graph, never traced into a fused region (core.
# fusion): fused, the arrays it builds were not eager's to the bit.
from . import fusion as _fusion                                # noqa: E402
_solve_implicit_c = _fusion.opaque(_solve_implicit_c)
