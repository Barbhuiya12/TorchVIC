"""Differentiable VIC soil column: one time step of vic_run/src/runoff.c.

This is the state recursion, not just the closed-form pieces in soil.py. It
reproduces runoff() for the configuration torchvic targets --
FROZEN_SOIL off, Nfrost = 1, so ice is identically zero -- and is verified
step for step against VIC's own runoff() by
tests/test_column_formulas.py, which compiles its oracle against
`reference/VIC-VIC.5.0.1/vic`.

**Read that path twice.** The repository's own `vic/` tree is not upstream
VIC 5.0.1: its runoff.c calls a `calc_Q12()` that 5.0.1 does not contain,
and `tests/oracle/build_ref_soil.sh` builds the older oracle against it. A
column verified against that oracle agrees, to machine precision, with
something VIC does not do. It cost 13.8 per cent of the baseflow and 16.4
per cent of the column storage at Walnut Gulch before it was found. Anything
this module is checked against must come from the pristine tree.

Besides runoff(), this module carries the parts of the soil water balance
that live in other VIC files but belong to the same column:

    arno_evap.c                bare soil evaporation, which the whole cell
                               needs wherever the vegetation does not tile
                               it completely
    compute_zwt.c              the water table, OUT_ZWT
    vic_run.c:356-371          soil wetness and root zone moisture,
                               OUT_SOIL_WET and OUT_ROOTMOIST

all of them reachable through `step(..., diagnostics=True)` or as functions
in their own right.

Two places where the C's structure cannot be copied literally:

**The saturation cascade.** When a layer overfills, runoff() pushes the
excess back up through the layers with

    tmplayer = lindex;
    while (tmp_inflow > 0) { tmplayer--; ... }

whose trip count depends on the state. With Nlayer = 3 the drainage loop runs
over layers 0 and 1 only, so the cascade can reach layer 0 and then the
surface, and no more: the loop is bounded at two iterations by the problem,
not by a tolerance. It is written out here as those two steps -- the same
computation with the data-dependence removed, so the branch becomes a
`where` and every cell in the batch executes both sides and selects.

**In-place layer updates.** The C mutates `liq[lindex]` repeatedly. Doing the
same to a row of a tensor -- `liq[..., l] = ...` -- writes through a view that
an earlier operation already read, which bumps its autograd version counter
and makes `backward()` fail with "one of the variables needed for gradient
computation has been modified by an inplace operation". Layers are therefore
kept as a Python list of `[cell]` tensors and rebound rather than written
into, and stacked only at the end. This is the same arithmetic; it is not an
approximation of it.

Nlayer is fixed at 3, which is what VIC allows anyway (vic_def.h:
MAX_LAYERS 3).

Shapes: [cell] for scalars, [cell, 3] for layered quantities at the
interface; internally layers are a list of [cell] tensors.
"""
import torch

from . import version as core_version, soil, xmath

NLAYER = 3


class SoilParams(object):
    """VIC soil parameters, batched over cells.

    Names and units are VIC's, taken from the soil parameter file:
        b_infilt     infiltration shape exponent                      [-]
        Ds           fraction of Dsmax where non-linear baseflow starts [-]
        Dsmax        maximum baseflow                             [mm/day]
        Ws           fraction of max_moist where non-linear baseflow
                     starts                                           [-]
        c            baseflow curve exponent                          [-]
        expt         Brooks-Corey exponent, per layer                 [-]
        Ksat         saturated hydraulic conductivity, per layer [mm/day]
        max_moist    maximum moisture, per layer                     [mm]
        resid_moist  residual moisture, per layer                     [mm]

    And three that only the diagnostics need, each with no default:
        depth        layer thickness, per layer                        [m]
        bubble       bubbling pressure, per layer                     [cm]
        root         root fraction, per layer; presence is what is read,
                     not the fraction itself (vic_run.c:361)           [-]
    """

    # w_crit_mm and w_wilt_mm are optional: when a parameter prior supplies
    # the critical and wilting fractions they are carried here as absolute
    # millimetres, so the vegetation's stress function and the column use the
    # same soil rather than two unconnected descriptions of it.
    #
    # depth and bubble are optional and carry no default: they are needed
    # only by the water-table diagnostic (`wrap_compute_zwt`), and a zwt
    # computed from an assumed bubbling pressure would be a number with no
    # provenance. When they are absent the diagnostic reports None rather
    # than a guess.
    # `thermal` carries what the surface energy balance reads from the soil
    # file and the column never touches -- avg_temp, dp, the thermal-property
    # inputs. A slot rather than a loose attribute because this class is
    # slotted, and assigning an unslotted name raises at the first call rather
    # than at import, which is how it was found.
    __slots__ = ('b_infilt', 'Ds', 'Dsmax', 'Ws', 'c', 'expt', 'Ksat',
                 'max_moist', 'resid_moist', 'runoff_steps_per_day',
                 'model_steps_per_day', 'w_crit_mm', 'w_wilt_mm',
                 'depth', 'bubble', 'root',
                 # SPATIAL_SNOW's one soil-file input (read_soilparam.c:
                 # 542-552). Optional: present only when the file was read
                 # with the option on, and read by basin.simulate alone.
                 'max_snow_distrib_slope',
                 # SPATIAL_FROST's one soil-file input, the width (C) of
                 # the uniform soil-temperature distribution the frost
                 # areas read (vic_init.c:703-710). Optional and read by
                 # basin.simulate alone, as above.
                 'frost_slope',
                 # read_soilparam.c:517-539, the soil file's FS_ACTIVE
                 # flag. FROZEN_SOIL runs only where it is set
                 # (prepare_full_energy.c:56, frozen_soil.c:60).
                 'fs_active',
                 # VIC-CropSyst's minimum liquid moisture per layer
                 # (runoff.c:301-309), [.., layer] mm, or None (VIC 5: the
                 # residual). Set by basin.simulate's cropsyst Wpwp floor.
                 'min_liq',
                 # VIC-CropSyst's drainage law (runoff.c:575-621): the
                 # Field_Capacity per layer, [.., layer] mm, and the
                 # irrigated-field flag per row ([.., 1] bool); None: VIC 5's.
                 # vcs_fc_mm is the soil's Field_Capacity (data, inert); the
                 # basin copies it to vcs_fc when the vcs_drain hook is on.
                 'vcs_fc', 'vcs_irrig', 'vcs_fc_mm',
                 'thermal')

    def __init__(self, b_infilt, Ds, Dsmax, Ws, c, expt, Ksat, max_moist,
                 resid_moist, runoff_steps_per_day=24, model_steps_per_day=24,
                 depth=None, bubble=None, root=None):
        self.b_infilt = b_infilt
        self.Ds = Ds
        self.Dsmax = Dsmax
        self.Ws = Ws
        self.c = c
        self.expt = expt
        self.Ksat = Ksat
        self.max_moist = max_moist
        self.resid_moist = resid_moist
        # get_global_param.c:809-813 refuses a RUNOFF_STEPS_PER_DAY that is
        # not a whole multiple of MODEL_STEPS_PER_DAY, because the sub-step
        # count runoff.c:97-98 forms is an integer division and a remainder
        # would silently drop part of the day. The same refusal belongs here:
        # 8 model steps against 20 runoff steps would give 2 sub-steps and
        # lose a sixth of every day's drainage without saying so.
        if runoff_steps_per_day % model_steps_per_day != 0:
            raise ValueError(
                'runoff_steps_per_day (%d) must be a whole multiple of '
                'model_steps_per_day (%d); VIC refuses this pair in '
                'get_global_param.c:809'
                % (runoff_steps_per_day, model_steps_per_day))
        self.runoff_steps_per_day = runoff_steps_per_day
        self.model_steps_per_day = model_steps_per_day
        self.depth = depth
        self.bubble = bubble
        self.root = root
        self.w_crit_mm = None
        self.min_liq = None
        self.vcs_fc = None
        self.vcs_irrig = None
        self.vcs_fc_mm = None
        self.thermal = None
        self.w_wilt_mm = None


def _spill_up(liq, cap, excess, level):
    """Push `excess` upward from `level`; return (layers, surface_excess).

    The C's `while (tmp_inflow > 0) { tmplayer--; ... }`, unrolled. `liq` and
    `cap` are lists of [cell] tensors; nothing is written in place.
    """
    out = list(liq)
    for target in range(level - 1, -1, -1):
        filled = out[target] + excess
        over = torch.clamp(filled - cap[target], min=0.0)
        out[target] = torch.minimum(filled, cap[target])
        excess = over
    return out, excess


# ---------------------------------------------------------------------------
# Diagnostics VIC reports and this column used to throw away.
#
# runoff() computes four of these itself and put_data.c writes them out:
# cell->asat (runoff.c:384-392), cell->zwt (runoff.c:398 via
# wrap_compute_zwt), cell->wetness and cell->rootmoist (vic_run.c:356-371),
# and cell->inflow (surface_fluxes.c:1087). Every one of them is already
# available here at the end of a step; returning them costs nothing and each
# is an independent row on the agreement scoreboard, because each reads a
# different part of the state.
#
# They are returned in a dict behind a `diagnostics=True` flag rather than
# appended to the tuple, so that a caller doing `moist, ro, bf = step(...)`
# keeps working unchanged.
# ---------------------------------------------------------------------------

# vic_def.h:69 -- MAX_ZWTVMOIST 11, the number of points in the water table
# versus moisture curve for each layer, including both layer boundaries.
MAX_ZWTVMOIST = 11

# The C compares a saturation deficit against DBL_EPSILON (compute_zwt.c:108).
DBL_EPSILON = 2.220446049250313e-16

# compute_zwt.c:50 -- the sentinel meaning "no water table in this layer".
ZWT_ABSENT = 999.0

CM_PER_M = 100.0
MM_PER_M = 1000.0
CONST_CDAY = 86400.0       # vic_physical_constants.h, seconds in a day


#: A floor under the base of every `pow` in the water-table curves. The
#: exponents are fractional, so the derivative of `x ** q` is infinite at
#: x = 0: clamping the base to zero leaves the forward pass finite and the
#: backward pass NaN, which is the trap a `torch.where` guard falls into for
#: the same reason. Small enough that no tabulated depth reaches it.
_POW_FLOOR = 1.0e-30


def zwt_lumped_curve(p):
    """The whole column's zwt-versus-moisture curve, the row VIC numbers
    `nlayers + 1`.  soil_moisture_from_water_table.c:155-224.

    `zwt_moisture_curve` builds one curve per layer, which is what OUT_ZWT
    reads. This builds the other one: the column filled from the bottom up,
    which is what OUT_ZWT_LUMPED reports and what the soil carbon cycle
    needs, because `soil_carbon_balance.c:93` takes the relative moisture of
    every thermal node from `cell->zwt_lumped`.

    The two are not the same shape and the difference is the point. Per
    layer, each layer's water table sits at its own bubbling pressure. Lumped,
    the layer holding the table is drawn with its own bubbling pressure and
    then every layer *above* it is drawn against an **effective** water table
    (:206-209) -- the depth that would give the same suction through that
    layer's own pore-size distribution. A column of sand over clay does not
    have one capillary fringe, it has one per horizon, and this is how VIC
    joins them.

    Returns (zwt_cm, moist_mm), each [cell, MAX_ZWTVMOIST], decreasing in the
    sample index exactly as the per-layer curves do.
    """
    if p.depth is None or p.bubble is None:
        raise ValueError(
            'zwt needs SoilParams.depth (m) and SoilParams.bubble (cm); '
            'neither has a defensible default')
    depth_cm = p.depth * CM_PER_M                        # [cell, layer]
    total_cm = depth_cm.sum(-1)                          # [cell]
    b_all = 0.5 * (p.expt - 3.0)
    bub_all = p.bubble
    resid = p.resid_moist                                # mm, already depth-weighted

    k = torch.arange(MAX_ZWTVMOIST, dtype=p.expt.dtype, device=p.expt.device)
    zwtp = total_cm.unsqueeze(-1) * (k / float(MAX_ZWTVMOIST - 1))  # [cell, n]

    # The depth of the top of each layer, in cm.
    top_cm = torch.cumsum(depth_cm, -1) - depth_cm       # [cell, layer]

    def L(x, j):
        """Layer j's value, broadcast onto the sample axis."""
        return x[..., j].unsqueeze(-1)

    # :177-183 -- the layer holding the water table is the deepest one whose
    # top is above it; below it every layer is saturated.
    nl = depth_cm.shape[-1]
    in_layer = []
    for j in range(nl):
        above = zwtp > L(top_cm, j) * 1.0
        deeper = torch.ones_like(above)
        for m in range(j + 1, nl):
            deeper = deeper & (zwtp <= L(top_cm, m))
        in_layer.append(above & deeper if j > 0 else deeper)

    moist = torch.zeros_like(zwtp)
    for j in range(nl):
        # Everything below layer j is full.
        below = sum((L(p.max_moist, m) for m in range(j + 1, nl)),
                    torch.zeros_like(zwtp))
        # :184-195 -- layer j itself, drawn against the real water table.
        d_j, b_j, bub_j = L(depth_cm, j), L(b_all, j), L(bub_all, j)
        top_j, res_j = L(top_cm, j), L(resid, j)
        w = (top_j + d_j - zwtp) / d_j
        w = w - (b_j / (b_j - 1.0)) * bub_j * (
            1.0 - torch.pow(torch.clamp((zwtp + bub_j - top_j) / bub_j,
                                        min=_POW_FLOOR),
                            (b_j - 1.0) / b_j)) / d_j
        tot = below + w * (L(p.max_moist, j) - res_j) + res_j
        # :199-221 -- and each layer above, against the effective table its
        # own bubbling pressure implies.
        b_save, bub_save, top_save = b_j, bub_j, top_j
        for m in range(j - 1, -1, -1):
            d_m, b_m, bub_m = L(depth_cm, m), L(b_all, m), L(bub_all, m)
            top_m, res_m = L(top_cm, m), L(resid, m)
            eff = top_save - bub_m + bub_m * torch.pow(
                torch.clamp((zwtp + bub_save - top_save) / bub_save,
                            min=_POW_FLOOR), b_m / b_save)
            wm = -(b_m / (b_m - 1.0)) * bub_m * (
                1.0 - torch.pow(torch.clamp((eff + bub_m - top_m) / bub_m,
                                            min=_POW_FLOOR),
                                (b_m - 1.0) / b_m)) / d_m
            tot = tot + wm * (L(p.max_moist, m) - res_m) + res_m
            b_save, bub_save, top_save = b_m, bub_m, top_m
        moist = torch.where(in_layer[j], tot, moist)

    # :168-174 -- a table at the surface means the whole column is full.
    full = p.max_moist.sum(-1).unsqueeze(-1).expand_as(moist)
    moist = torch.where(zwtp == 0.0, full, moist)
    return -zwtp, moist


def zwt_moisture_curve(p):
    """The per-layer zwt-versus-moisture table, drivers/.../soil_moisture_from_water_table.c:85-113.

    For each layer j and each of the MAX_ZWTVMOIST sample depths zwt' below
    the top of that layer,

        b     = 0.5 * (expt[j] - 3)                         (line 88)
        w_avg = (D - zwt' - (b/(b-1)) * bub
                            * (1 - ((zwt'+bub)/bub)^((b-1)/b))) / D   (99-103)
        moist = clamp(w_avg, 0, 1) * (max_moist - resid) + resid      (109-112)
        zwt   = -(depth above this layer) * 100 - zwt'                (96)

    with D the layer thickness in centimetres and every length in cm. The
    result is a table that decreases monotonically in the sample index: index
    0 is the layer saturated to its top, index MAX_ZWTVMOIST-1 is the water
    table at the layer's base.

    Only the per-layer curves are built. VIC also builds two lumped curves --
    the top N-1 layers together (line 118-152) and the whole column filled
    from the bottom (line 155-224) -- and the second of those is what
    OUT_ZWT_LUMPED reports. It is not built here; OUT_ZWT, which is the
    variable the scoreboard carries, needs only the per-layer curves.

    Returns (zwt_cm, moist_mm), each [cell, layer, MAX_ZWTVMOIST].
    """
    if p.depth is None or p.bubble is None:
        raise ValueError(
            'zwt needs SoilParams.depth (m) and SoilParams.bubble (cm); '
            'neither has a defensible default')

    depth_cm = p.depth * CM_PER_M                       # [cell, layer]
    b = 0.5 * (p.expt - 3.0)
    bub = p.bubble

    # The sample offsets zwt' are the same fractions of the layer thickness
    # for every layer: the C advances zwt_prime by D/(MAX_ZWTVMOIST-1) each
    # time (line 111), starting at 0.
    k = torch.arange(MAX_ZWTVMOIST, dtype=p.expt.dtype, device=p.expt.device)
    frac = k / float(MAX_ZWTVMOIST - 1)                 # [MAX_ZWTVMOIST]
    zwt_prime = depth_cm.unsqueeze(-1) * frac           # [cell, layer, n]

    b_ = b.unsqueeze(-1)
    bub_ = bub.unsqueeze(-1)
    D_ = depth_cm.unsqueeze(-1)

    # (b-1)/b is the exponent; b == 1 (expt == 5) makes b/(b-1) singular in
    # the C as well, so it is not smoothed over here -- only kept finite so
    # one bad layer cannot NaN a whole batch's gradient.
    bm1 = b_ - 1.0
    ratio = (zwt_prime + bub_) / torch.clamp(bub_, min=soil.TINY)
    term = ((b_ / torch.where(bm1.abs() > soil.TINY, bm1,
                              torch.full_like(bm1, soil.TINY))) * bub_ *
            (1.0 - soil._pow_nonneg(ratio, bm1 / torch.clamp(b_.abs(),
                                                            min=soil.TINY))))
    w_avg = torch.clamp((D_ - zwt_prime - term) /
                        torch.clamp(D_, min=soil.TINY), 0.0, 1.0)

    resid_ = p.resid_moist.unsqueeze(-1)
    moist_mm = w_avg * (p.max_moist.unsqueeze(-1) - resid_) + resid_

    # Elevation of each sample relative to the soil surface, negative
    # downward (line 96). The depth above layer j is the cumulative thickness
    # of layers 0..j-1.
    above = torch.cumsum(depth_cm, dim=-1) - depth_cm   # [cell, layer]
    zwt_cm = -above.unsqueeze(-1) - zwt_prime
    return zwt_cm, moist_mm


def compute_zwt(moist, zwt_cm, moist_mm):
    """compute_zwt(), vic_run/src/compute_zwt.c:41-70, batched.

    The C walks the table downward from the last index while
    `moist > zwtvmoist_moist[i]`. Because the table decreases with the index,
    that walk lands on the largest index whose tabulated moisture still
    covers `moist`, and the answer is a linear interpolation between that
    index and the next. The walk is replaced here by a count, which is the
    same index without the data-dependent loop.

    Returns zwt in cm, negative below the surface, with 999 where the layer
    holds too little water for a water table (compute_zwt.c:50).
    """
    m = moist.unsqueeze(-1)
    last = MAX_ZWTVMOIST - 1
    # The C's walk is `while (i >= 1 && moist > table[i]) i--`, which steps
    # past a *trailing run* of covered samples, not every covered sample
    # anywhere in the table. For the per-layer curves the two are the same,
    # because those curves decrease monotonically with the index. The lumped
    # column curve does not: a coarse horizon below a fine one can hold more
    # water with the table lower, so the curve turns back on itself, and a
    # total count would step past samples the C never reaches. Count the run.
    covered = (m > moist_mm)[..., 1:]           # the C never tests index 0
    # `run` is how many of the last entries of `covered` are all True --
    # where the walk from the last index stops. It used to be
    # cumprod(flip(covered)).sum(); the same integer is the position of the
    # first False counted from the end, or all of them where there is none.
    # (CUDA's cumprod along a 10-long innermost axis of int64 took 28 ms per
    # call on the ISC domain, a quarter of the step's device time.)
    stop = ~torch.flip(covered, [-1])
    run = torch.where(stop.any(-1), stop.to(torch.uint8).argmax(-1),
                      torch.full_like(stop[..., 0], stop.shape[-1],
                                      dtype=torch.long))
    idx = torch.clamp(last - run, min=0)

    lo = torch.gather(moist_mm, -1, torch.clamp(idx + 1, max=last).unsqueeze(-1)
                      ).squeeze(-1)
    hi = torch.gather(moist_mm, -1, idx.unsqueeze(-1)).squeeze(-1)
    zlo = torch.gather(zwt_cm, -1, torch.clamp(idx + 1, max=last).unsqueeze(-1)
                       ).squeeze(-1)
    zhi = torch.gather(zwt_cm, -1, idx.unsqueeze(-1)).squeeze(-1)

    span = hi - lo
    interp = zlo + (zhi - zlo) * (moist - lo) / torch.where(
        span.abs() > soil.TINY, span, torch.full_like(span, soil.TINY))

    # idx == last is the C's `if (i == MAX_ZWTVMOIST - 1)` branch: the layer
    # is drier than the driest tabulated state, so either there is no water
    # table (999) or the moisture sits exactly on the last sample.
    driest_moist = moist_mm[..., last]
    driest_zwt = zwt_cm[..., last]
    at_bottom = torch.where(moist < driest_moist,
                            torch.full_like(moist, ZWT_ABSENT), driest_zwt)
    return torch.where(idx == last, at_bottom, interp)


def zwt_curves(p):
    """Both water-table curves of `p` -- they depend on the soil
    parameters alone -- for `step(..., curves=)` to reuse across a run's
    steps instead of rebuilding them every step. None when `p` carries no
    depth or bubble, as the per-step path then skips the water table too."""
    if p.depth is None or p.bubble is None:
        return None
    return dict(layer=zwt_moisture_curve(p), lumped=zwt_lumped_curve(p))


def compute_zwt_lumped(moist, p, curve=None):
    """OUT_ZWT_LUMPED: the water table of every layer lumped together.
    compute_zwt.c:128-136.

    `moist` is [..., layer] in mm; the layers are summed and read off the
    column curve. A column too dry to have a table anywhere reports its own
    base (:134-135) rather than the 999 sentinel, so the value is always a
    depth and the soil carbon cycle downstream never has to test for it.
    """
    zwt_cm, moist_mm = zwt_lumped_curve(p) if curve is None else curve
    z = compute_zwt(moist.sum(-1), zwt_cm, moist_mm)
    base = -(p.depth * CM_PER_M).sum(-1)
    return torch.where(z == ZWT_ABSENT, base.expand_as(z), z)


def wrap_compute_zwt(moist, p, curve=None):
    """wrap_compute_zwt(), vic_run/src/compute_zwt.c:80-126, batched.

    Returns (zwt, layer_zwt): the column's water table in cm and each layer's
    own, both negative below the soil surface. `zwt` is OUT_ZWT.

    The column value is the water table of the lowest layer that is not
    completely saturated (compute_zwt.c:106-126); if every layer is
    saturated it is 0 (the table is at the surface), and if the layer that
    decides has no table of its own the value falls back to the depth of the
    top of the saturated stack.
    """
    NLAYER = p.max_moist.shape[-1]
    zwt_cm, moist_mm = zwt_moisture_curve(p) if curve is None else curve
    layer_zwt = compute_zwt(moist, zwt_cm, moist_mm)

    depth_cm = p.depth * CM_PER_M
    total_cm = depth_cm.sum(dim=-1)

    # compute_zwt.c:102-104 -- the bottom layer never reports "absent"; if it
    # has no table the column's base is used.
    bottom = layer_zwt[..., NLAYER - 1]
    bottom = torch.where(bottom == ZWT_ABSENT, -total_cm, bottom)
    layer_zwt = torch.cat([layer_zwt[..., :NLAYER - 1],
                           bottom.unsqueeze(-1)], dim=-1)

    # The C's `while (idx >= 0 && max_moist[idx] - moist[idx] <= EPS)` walks
    # up over the saturated layers. Unrolled: a layer only counts as part of
    # the saturated stack if every layer below it is saturated too.
    sat = (p.max_moist - moist) <= DBL_EPSILON
    below_all_sat = torch.ones_like(sat[..., 0], dtype=torch.bool)
    stack = []
    for l in range(NLAYER - 1, -1, -1):
        below_all_sat = below_all_sat & sat[..., l]
        stack.append(below_all_sat)
    stack = torch.stack(stack[::-1], dim=-1)            # [cell, layer]
    n_sat = stack.sum(dim=-1)                           # layers peeled off
    idx = NLAYER - 1 - n_sat                            # -1 when all saturated

    # tmp_depth after the walk: the total depth minus the saturated stack.
    tmp_depth = total_cm - (depth_cm * stack).sum(dim=-1)

    safe_idx = torch.clamp(idx, min=0)
    chosen = torch.gather(layer_zwt, -1, safe_idx.unsqueeze(-1)).squeeze(-1)
    # An interior layer reporting 999 falls back to -tmp_depth (line 121);
    # the bottom layer never reports 999 by the substitution above.
    chosen = torch.where(chosen == ZWT_ABSENT, -tmp_depth, chosen)
    zwt = torch.where(idx < 0, torch.zeros_like(chosen), chosen)
    return zwt, layer_zwt


def soil_wetness(moist, p):
    """cell->wetness, OUT_SOIL_WET.  vic_run/src/vic_run.c:365-371.

        wetness = mean over layers of
                  (moist - Wpwp) / (porosity * depth * 1000 - Wpwp)

    `porosity * depth * MM_PER_M` is read_soilparam.c:609's definition of
    max_moist, so the denominator is max_moist - Wpwp and no porosity is
    needed here. Wpwp is `w_wilt_mm`, which is Wpwp_FRACT * max_moist
    (read_soilparam.c:646) and is what SoilParams already carries for the
    canopy's moisture stress. Returns None when it was not supplied, because
    a wetness measured from an assumed wilting point is not VIC's.
    """
    wpwp = p.w_wilt_mm
    if wpwp is None:
        return None
    span = p.max_moist - wpwp
    return ((moist - wpwp) /
            torch.where(span.abs() > soil.TINY, span,
                        torch.full_like(span, soil.TINY))).mean(dim=-1)


def expand_params(sp, n):
    """A SoilParams with every per-cell tensor repeated `n` times along the
    cell axis, so a column can be stepped per vegetation tile with the
    cell's soil: row c*n + k is tile k of cell c. VIC keeps a soil column
    per tile (cell[iveg][band].layer, vic_run.c) with the cell's soil
    parameters; this is that layout, flattened for one vectorised step.
    Non-tensor fields (step counts) and None are carried unchanged; dicts
    of tensors (thermal) are expanded entry by entry.
    """
    def rep(v):
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return v.repeat_interleave(n, dim=0)
        if isinstance(v, dict):
            return {k: rep(x) for k, x in v.items()}
        return v
    out = object.__new__(type(sp))
    for name in type(sp).__slots__:
        if hasattr(sp, name):
            setattr(out, name, rep(getattr(sp, name)))
    return out


def tile_params(sp, n):
    """A SoilParams with every per-cell tensor repeated `n` times
    member-major (row m*n_cells + c is member m's cell c), for an ensemble
    of parameter sets run as one batch. `expand_params` interleaves instead,
    for the tiles of one cell; the two layouts must not be mixed.
    """
    def rep(v):
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return v.repeat((n,) + (1,) * (v.dim() - 1))
        if isinstance(v, dict):
            return {k: rep(x) for k, x in v.items()}
        return v
    out = object.__new__(type(sp))
    for name in type(sp).__slots__:
        if hasattr(sp, name):
            setattr(out, name, rep(getattr(sp, name)))
    return out


def root_zone_moisture(moist, root):
    """cell->rootmoist, OUT_ROOTMOIST.  vic_run/src/vic_run.c:358-364.

        rootmoist = sum over layers with root[l] > 0 of moist[l]

    Note what this is not: it is not root-fraction weighted. VIC sums the
    *whole* layer moisture of every layer the roots reach, so a layer with
    one per cent of the roots contributes as much as one with ninety-nine.

    `root` is per vegetation tile, which the column does not know about, so
    this is a function rather than something `step` can return on its own:
    the caller holds the tile's root profile and its area weight. When
    `SoilParams.root` is set -- shape broadcastable to `moist` -- `step`
    will apply it directly.
    """
    return (moist * (root > 0).to(moist.dtype)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Bare soil evaporation -- vic_run/src/arno_evap.c
#
# This column had none at all. Walnut Gulch hides that, because its Cv sums
# to 1 so every square metre is under a vegetation tile and VIC reports
# OUT_EVAP_BARE 0.000; but that is a property of one parameter file, not of
# the model. Any cell with a bare fraction -- and, through fcanopy < 1, any
# sparsely vegetated tile -- evaporates from the soil surface directly, and a
# column without the term simply keeps that water.
#
# What is and is not entangled with the energy balance, precisely:
#
#   *Not* entangled: everything arno_evap.c does with the soil. The ARNO
#   beta function (lines 86-163) and the availability cap (lines 171-181)
#   are closed-form functions of the top layer's moisture, its capacity, the
#   infiltration shape parameter b_infilt and the residual content. They are
#   ported in full below and verified against the compiled arno_evap().
#
#   Entangled: the two numbers `rad` and `ra` that reach Epot.
#   func_surf_energy_bal.c:769 and :799 call arno_evap with `NetBareRad` --
#   the net radiation at the *bare soil* surface evaluated at the surface
#   temperature the energy balance solved for -- and with `Ra_used[0]`, the
#   aerodynamic resistance after StabilityCorrection at that same
#   temperature (func_surf_energy_bal.c:700-744). Neither can be produced
#   inside the column, and neither is a soil quantity.
#
# The split follows that boundary: `bare_soil_evap` takes Epot already in
# millimetres for the step and owes nothing to the energy balance;
# `arno_evap` is the convenience that also builds Epot, and it takes `rad`
# and `ra` from its caller rather than inventing them. basin.py is where
# those come from, and the note in `arno_evap`'s docstring says what it must
# pass.
#
# param.SOIL_RARC = 100.0 s/m, initialize_parameters.c:58; arno_evap passes
# rc = 0 (arno_evap.c:83), so bare soil has architectural resistance but no
# stomatal resistance, which is the whole difference from the canopy call.
# ---------------------------------------------------------------------------
SOIL_RARC = 100.0           # initialize_parameters.c:58
ARNO_EVAP_NTERM = 30        # arno_evap.c:153, "the first 30 terms"


def bare_soil_evap(Epot, moist, max_moist, b_infilt, depth1,
                   resid_moist_frac, resid_mm=None):
    """ARNO bare soil evaporation, vic_run/src/arno_evap.c:86-181.

    Args:
        Epot:      [cell] potential bare soil evaporation for the step, mm
                   (arno_evap.c:82-84 -- penman(...) * delta_t / CONST_CDAY)
        moist:     [cell] liquid moisture of the *top* layer, mm. Evaporation
                   from bare soil comes from layer 0 only (arno_evap.c:10).
        max_moist: [cell] top layer capacity, mm
        b_infilt:  [cell] infiltration shape parameter
        depth1:    [cell] top layer thickness, m
        resid_moist_frac: [cell] residual content as a *fraction*, which is
                   what soil_con->resid_moist holds; arno_evap.c:172 forms
                   the millimetre value itself as resid * depth1 * 1000.
        resid_mm:  [cell] or None. VIC 5.1.0 hands arno_evap the residual
                   already in mm (vic_init.c 5.1.0:775, arno_evap.c 5.1.0
                   :151-154); given, it is used as it is and
                   `resid_moist_frac` is ignored.

    Returns:
        [cell] bare soil evaporation for the step, mm.

    The algebra, unwound. The C computes

        max_infil = (1 + b) * max_moist                              (89)
        ratio     = (1 - moist/max_moist) ^ (1/(b+1))               (94-107)
        tmp       = max_infil * (1 - ratio)                          (110)
        r0        = 1 - tmp/max_infil                            (126-127)

    and r0 is identically the `ratio` of line 107, so the intermediate
    `tmp`/`max_infil` round trip cancels exactly and is not reproduced. Then

        as    = 1 - r0^b                                         (138-143)
        dummy = 1 + sum_{k=1..30} b * r0^k / (b + k)              (152-159)
        beta  = as + (1 - as) * (1 - r0) * dummy                      (161)
        evap  = Epot * beta                                           (162)

    The `tmp >= max_infil` shortcut at line 117 (evap = Epot) is not a
    separate branch here because it is not a discontinuity: it fires exactly
    when r0 = 0, where as = 1 and beta = 1, so the general expression already
    returns Epot. Removing the branch removes a `where` without changing a
    value.

    The power series is summed by repeated multiplication rather than by
    `pow`, because that is what arno_evap.c:154-157 does and the two differ
    in the last bit.

    One branch of the C is deliberately absent: `if (b_infilt == -1.0)`
    (line 90), which sets tmp = max_infil and makes every following ratio
    degenerate. b_infilt = -1 makes max_infil zero and the infiltration curve
    meaningless; VIC guards it, nothing produces it, and reproducing it here
    would mean carrying a `where` on an exact float comparison.
    """
    span = torch.clamp(max_moist, min=soil.TINY)
    # arno_evap.c:76-78 -- moist is capped at capacity before anything else.
    m = torch.minimum(moist, max_moist)
    r0 = soil._pow_nonneg(torch.clamp(1.0 - m / span, min=0.0, max=1.0),
                          1.0 / (b_infilt + 1.0))

    # arno_evap.c 5.1.0:84-139 step by step: the saturated-area ratio goes
    # through max_infil and back, then through pow(., b) and pow(., 1/b);
    # each round trip rounds, and using r0 throughout lands ulps away.
    max_infil = (1.0 + b_infilt) * span
    tmp = max_infil * (1.0 - r0)
    full = tmp >= max_infil
    ratio = 1.0 - tmp / max_infil
    ratio = torch.where(ratio != 0.0, soil._pow_nonneg(ratio, b_infilt),
                        ratio)
    a_s = 1.0 - ratio
    ratio = soil._pow_nonneg(ratio, 1.0 / b_infilt)

    dummy = torch.ones_like(r0)
    tmpsum = torch.ones_like(r0)
    for k in range(1, ARNO_EVAP_NTERM + 1):
        tmpsum = tmpsum * ratio
        dummy = dummy + b_infilt * tmpsum / (b_infilt + k)

    evap = torch.where(full, Epot,
                       Epot * (a_s + (1.0 - a_s) * (1.0 - ratio) * dummy))

    # arno_evap.c:171-181 -- withdrawal is capped at the liquid water above
    # the residual content, and is zero when there is none. Condensation
    # (evap < 0) is deliberately not capped: "we won't put limits on
    # condensation" (arno_evap.c:170).
    if resid_mm is None:
        resid_mm = resid_moist_frac * depth1 * MM_PER_M
    avail = m - resid_mm
    capped = torch.where(avail > 0, torch.minimum(evap, avail),
                         torch.zeros_like(evap))
    return torch.where(evap > 0, capped, evap)


def arno_evap(moist, max_moist, b_infilt, depth1, resid_moist_frac,
              air_temp, elevation, rad, vpd, ra, delta_t):
    """`bare_soil_evap` with Epot built as arno_evap.c:82-84 builds it.

        Epot = penman(air_temp, elevation, rad, vpd, ra, 0.0, SOIL_RARC)
               * delta_t / CONST_CDAY

    `rad` and `ra` are the caller's responsibility and are not soil
    quantities. For agreement with VIC they must be, exactly:

        rad  NetBareRad -- net radiation at the bare soil surface at the
             surface temperature the energy balance solved for, which is
             what func_surf_energy_bal.c:799 passes. Under a canopy with
             fcanopy < 1 it is `surf_atten * NetBareRad` instead
             (func_surf_energy_bal.c:769), and the tile's bare-soil
             contribution is weighted by (1 - fcanopy).
        ra   Ra_used[0] -- the aerodynamic resistance *after*
             StabilityCorrection at that surface temperature
             (func_surf_energy_bal.c:700-744), not the neutral value.

    delta_t is the model step in seconds.

    `penman` is imported here rather than at module scope: it lives in
    vegetation.py, the column has no other reason to depend on the
    vegetation, and a lazy import keeps that dependency confined to the one
    function that genuinely needs it.
    """
    from . import vegetation
    Epot = (vegetation.penman(air_temp, elevation, rad, vpd, ra,
                              torch.zeros_like(ra), SOIL_RARC)
            * delta_t / CONST_CDAY)
    return bare_soil_evap(Epot, moist, max_moist, b_infilt, depth1,
                          resid_moist_frac)


def _brooks_corey_drainage(Ksat, tmp_liq, resid_moist, max_moist, expt):
    """Gravity drainage from one layer to the next, vic_run/src/runoff.c:203-208.

        Q12 = Ksat * ((tmp_liq - resid) / (max_moist - resid)) ** expt

    with Ksat already divided by runoff_steps_per_day (runoff.c:151-152).

    This is *not* `soil.calc_Q12`, and the difference is not cosmetic.
    `soil.calc_Q12` implements

        Q12 = init - ((init-resid)^(1-expt) - Ksat/(max-resid)^expt*(1-expt))
                     ^(1/(1-expt)) - resid

    which is VIC 4.0.x's analytic integral of the same Brooks-Corey law over
    the sub-step: it is the drainage that a layer emptying continuously would
    produce, and it saturates at the water actually present. VIC 5.0.1 does
    not do that. It evaluates the instantaneous rate at the start-of-sub-step
    moisture and lets the layer-update clamps (runoff.c:277-287) absorb any
    overdraft. The two agree only in the small-flux limit; at Ksat/24 = 10 mm
    and a wet layer the explicit form is the larger of the two, so the closed
    form was systematically under-draining the upper layers and starving the
    bottom layer -- and with it the ARNO baseflow that reads the bottom
    layer's relative moisture.

    The repository's own `vic/` tree carries the 4.0.x `calc_Q12`, which is
    why the existing oracle agreed: the oracle was compiled against the
    modified source. The authority is the pristine 5.0.1 tree, and this is
    what it computes.

    The base of the power is a saturation ratio in [0, 1] and the exponent is
    positive, so `soil._pow_nonneg` is exactly right here: it is the true
    value wherever the base is positive and the limit, zero, at the residual
    content. No floor is needed, and none is applied -- the whole
    Q12_AVAIL_FLOOR / Q12_SPAN_FLOOR / Q12_INNER_FLOOR apparatus exists only
    to keep the *negative* powers of the 4.0.x closed form finite, and the
    5.0.1 form has no negative power in it.
    """
    span = max_moist - resid_moist
    ratio = (tmp_liq - resid_moist) / torch.clamp(span, min=soil.TINY)
    return Ksat * soil._pow_nonneg(ratio, expt)


def _brooks_corey_integral_ratio(Ksat, tmp_liq, resid_moist, max_moist,
                                  expt):
    """Overflow-safe reformulation of runoff.c calc_Q12, used ONLY where
    VIC's literal C form is not finite (see `_brooks_corey_integral`).

    The water
    a layer at `tmp_liq` loses over one sub-step when it drains
    continuously along the Brooks-Corey law dθ/dt = -Ksat·((θ-r)/(M-r))^e,

        Q12 = θ0 - [ (θ0-r)^(1-e) - Ksat(1-e)/(M-r)^e ]^(1/(1-e)) - r.

    Written for a gradient. With a = θ0 - r and b = Ksat(e-1)/(M-r)^e the
    same quantity is

        Q12 = a · (1 - (1 + b·a^(e-1))^(-1/(e-1)))
            = -a · expm1(-log1p(b·a^(e-1)) / (e-1)),

    which never forms a^(1-e): with e up to 30 (delta-VIC's bound) that
    term overflows to inf a hair above residual, and inf inside the
    unselected branch of a where is a NaN gradient. Here every
    intermediate is at most b·a^(e-1), and a <= M - r keeps that moderate.
    Equal to the C's direct form to rounding; more accurate than it where
    Q12 is small (the C subtracts two nearly equal numbers).

    b·a^(e-1) is formed as a ratio, Ksat(e-1)·(a/(M-r))^(e-1)/(M-r), and
    never as Ksat(e-1)/(M-r)^e times a^(e-1). The two are the same number,
    but (M-r)^e is a mm-scale span raised to e: a 1,000 mm bottom layer at
    e = 103 is 1e309, past the double range, and the product inf·0 at a
    layer near residual is NaN in value and gradient. That NaN was
    measured at expt >= 160 on 12 Indian cells (NaN at 160 and 250
    before, finite value and gradient at both after). The saturation
    ratio a/(M-r) is in [0, 1] for tmp_liq <= max_moist, so its power
    underflows gracefully to 0 instead of overflowing. At the file's
    exponents the two orderings differ by rounding only: 3.4e-13 over 75
    days of et, runoff, baseflow and soil_liq on those 12 cells, and tests/test_vic_constants_and_limits.py pins both
    forms and VIC 5.1.0's own calc_Q12 against each other.
    """
    span = torch.clamp(max_moist - resid_moist, min=soil.TINY)
    a = torch.clamp(tmp_liq - resid_moist, min=0.0)
    em1 = expt - 1.0
    x = Ksat * em1 * soil._pow_nonneg(a / span, em1) / span
    return -a * torch.expm1(-torch.log1p(x) / em1)


def _calc_q12_c(Ksat, tmp_liq, resid_moist, max_moist, expt):
    """runoff.c:478-481 verbatim, same operations in the same order."""
    return (tmp_liq - xmath.pow(xmath.pow(tmp_liq - resid_moist, 1.0 - expt)
                                - Ksat / xmath.pow(max_moist - resid_moist,
                                                   expt) * (1.0 - expt),
                                1.0 / (1.0 - expt)) - resid_moist)


def _brooks_corey_integral(Ksat, tmp_liq, resid_moist, max_moist, expt):
    """VIC 5.1.0's layer drainage, runoff.c calc_Q12 (runoff.c:469-484),
    in VIC's literal C form:

        Q12 = tmp_liq - pow(pow(tmp_liq - resid, 1 - expt)
                            - Ksat / pow(max_moist - resid, expt) * (1 - expt),
                            1 / (1 - expt)) - resid

    Bit-identical to VIC wherever VIC is finite. The expm1/log1p ratio
    form it replaces (`_brooks_corey_integral_ratio`) is more accurate where
    Q12 << a, but it differs from VIC by ~16 ulp and that was measured as
    the sole January soil seed on himal, and as the Q12 noise behind the
    NIJSSEN2001 igp row.

    runoff.c has no floors here: its only guard is the caller's
    `tmp_liq > resid_moist` gate (runoff.c:178), kept in `step`. Two
    things are added that never change a value VIC produces:
      * lanes where the C form is not finite (expt past ~50-100, where
        pow(span, expt) or the outer pow overflows; VIC itself returns
        -inf/NaN there) take the ratio form instead;
      * with a gradient wanted, lanes whose C value is finite only through
        an overflow keep VIC's value but take the ratio form's gradient,
        and every non-C lane (incl. gated-off tmp_liq <= resid, where
        pow(0, 1 - expt) = inf) evaluates the C form on harmless inputs, so
        torch.where does not turn an inf into a NaN gradient.
    """
    with torch.no_grad():
        q0 = _calc_q12_c(Ksat, tmp_liq, resid_moist, max_moist, expt)
        gate = (tmp_liq > resid_moist) & (max_moist > resid_moist)
        use_c = torch.isfinite(q0) & gate
    q_r = _brooks_corey_integral_ratio(Ksat, tmp_liq, resid_moist,
                                       max_moist, expt)
    if not (torch.is_grad_enabled() and any(
            torch.is_tensor(v) and v.requires_grad
            for v in (Ksat, tmp_liq, resid_moist, max_moist, expt))):
        return torch.where(use_c, q0, q_r)
    # Gradient path. Where every intermediate of the C form is finite the
    # C form is differentiated directly. Where the value is finite only
    # through an overflow (pow(a, 1 - expt) or pow(span, expt) = inf), the
    # value stays VIC's and the gradient is the ratio form's.
    with torch.no_grad():
        big = 1e280
        a = tmp_liq - resid_moist
        span = max_moist - resid_moist
        em1 = expt - 1.0
        p1 = torch.pow(a, -em1)
        p2 = torch.pow(span, expt)
        inner = p1 + Ksat / p2 * em1
        # each factor of the chain rule (d pow / d base and d exponent)
        # stays well inside the double range
        smooth = use_c & (p1 * expt / a < big) & (p1 * a.log().abs() < big) \
            & (p2 < big) & (p2 > 1.0 / big) & (inner > 0) \
            & (torch.pow(inner, -1.0 / em1 - 1.0) / em1 < big)
    t_s = torch.where(smooth, tmp_liq, resid_moist + 0.5)
    m_s = torch.where(smooth, max_moist, resid_moist + 1.0)
    e_s = torch.where(smooth, expt, torch.full_like(expt, 2.0)) \
        if torch.is_tensor(expt) else expt
    q_c = _calc_q12_c(Ksat, t_s, resid_moist, m_s, e_s)
    q_st = q0 + (q_r - q_r.detach())   # value q0 exactly, gradient q_r
    return torch.where(smooth, q_c, torch.where(use_c, q_st, q_r))


def distribute_evap_frost(evap, state, ice, p, frost_fract):
    """Split a layer evaporation demand over frost sub-areas.  runoff.c:86-123.

    Args:
        evap: [...] layer demand for the model step, mm (every layer).
        state: [...] TOTAL entering moisture, mm.
        ice: [..., frost] the ice each sub-area holds, mm.
        p: SoilParams (resid_moist only).
        frost_fract: the sub-area fractions, len frost.

    A positive demand is split in proportion to each area's liquid above
    residual (floored at zero); where no area holds any, every area gets
    zero. A non-positive demand -- condensation -- is left whole in every
    area, as runoff.c:136-140 leaves it. In the C's units this is per
    runoff sub-step; the division cancels, so full-step units go in and
    full-step units come out.

    Returns [..., frost] per-area demand for the model step.
    """
    fr = torch.as_tensor(list(frost_fract), dtype=evap.dtype,
                         device=evap.device)
    while fr.dim() < ice.dim():
        fr = fr.unsqueeze(0)
    avail = torch.clamp(state.unsqueeze(-1) - ice
                        - p.resid_moist.unsqueeze(-1), min=0.0)
    total = (avail * fr).sum(-1)
    frac = torch.where(total > 0.0, evap / torch.where(
        total > 0.0, total, torch.ones_like(total)),
        torch.ones_like(total))
    split = avail * frac.unsqueeze(-1)
    whole = evap.unsqueeze(-1).expand_as(split)
    return torch.where(evap.unsqueeze(-1) > 0.0, split, whole)


def step(state, ppt, evap, p, diagnostics=False, ice=None, curves=None,
         gwm=None):
    """Advance one model time step.

    Args:
        state: [cell, 3] TOTAL moisture at the start of the step, mm --
               liquid plus ice. With FROZEN_SOIL off it is all liquid.
        ice:   [cell, 3] of that total which is ice, mm, or None for
               none. runoff.c:155-158 splits the layer the same way and
               moves only the liquid: the ice reduces both the capacity
               and the residual the liquid is held between, but the
               drainage curve at :195-205 keeps the layer's full
               max_moist and resid_moist.
        ppt:   [cell] liquid water arriving at the soil surface, mm
        evap:  [cell, 3] evapotranspiration demand per layer, mm
        p:     SoilParams
        diagnostics: when True, a fourth value is returned -- a dict of the
               quantities runoff() computes for the history file. The default
               is False so that `moist, ro, bf = step(...)` is unchanged.
        curves: `zwt_curves(p)`, made once for a run whose parameters do
               not change between steps; the water table is then read off
               them instead of rebuilding both curves -- [cell, layer, 11]
               and [cell, 11] tables of powers -- every step. Same numbers.
        gwm:   None (VIC), or 'exact' / 'fixed': VIC-WUR 3.0's runoff_GWM
               (`step_gwm`; the diagnostics then carry 'gwrecharge' and
               'recharge').

    Returns:
        new_state: [cell, 3] moisture at the end of the step, mm
        runoff:    [cell] surface runoff, mm
        baseflow:  [cell] baseflow, mm
        diag:      only when diagnostics=True; see `step`'s notes below.

    The diagnostics dict carries one key per VIC history variable:

        'asat'      OUT_ASAT      saturated area fraction, recomputed from
                                  the *final* moisture (runoff.c:380-385)
        'inflow'    OUT_INFLOW    water arriving at the surface
                                  (surface_fluxes.c:1087: cell->inflow = ppt)
        'soil_wet'  OUT_SOIL_WET  None unless SoilParams.w_wilt_mm is set
        'rootmoist' OUT_ROOTMOIST None unless SoilParams.root is set
        'zwt'       OUT_ZWT       None unless SoilParams.depth and .bubble
                                  are set
        'zwt_layer'               per-layer water table, cm, same condition
        'zwt_lumped' OUT_ZWT_LUMPED  the column lumped, cm, same condition

    A key whose inputs were not supplied is present and None. It is never a
    plausible-looking substitute: OUT_SOIL_WET computed against an assumed
    wilting point would agree with VIC only by accident, and an agreement by
    accident is worse than a gap.
    """
    if gwm is not None:
        return step_gwm(state, ppt, evap, p, diagnostics, ice, curves,
                        exact=(gwm == 'exact'))
    NLAYER = p.max_moist.shape[-1]
    resid = [p.resid_moist[..., l] for l in range(NLAYER)]
    maxm = [p.max_moist[..., l] for l in range(NLAYER)]
    if ice is None:
        # Nothing frozen: the capacity is the layer's and the floor is
        # the residual, which is what every expression below reduces to.
        ice_l = [torch.zeros_like(resid[l]) for l in range(NLAYER)]
        cap = maxm
        floor = resid
    else:
        ice_l = [ice[..., l] for l in range(NLAYER)]
        # runoff.c writes `liq = max_moist - ice` and
        # `liq = resid_moist - ice`; in liquid space those are the
        # capacity and the floor.
        cap = [maxm[l] - ice_l[l] for l in range(NLAYER)]
        floor = [resid[l] - ice_l[l] for l in range(NLAYER)]

    # VIC-CropSyst (runoff.c:301-309): min_liq, the liquid the layer cannot
    # fall below, is the residual in layer 0 and Wpwp below it. Opt-in
    # (SoilParams.min_liq); None leaves every expression as VIC 5's.
    mliq = getattr(p, 'min_liq', None)
    if mliq is not None:
        fl = [torch.maximum(resid[l], mliq[..., l]) for l in range(NLAYER)]
        floor = [fl[l] - ice_l[l] for l in range(NLAYER)]

    # VIC-CropSyst's Field_Capacity and irrigated-field flag (opt-in)
    vfc = getattr(p, 'vcs_fc', None)
    virr = getattr(p, 'vcs_irrig', None)
    if vfc is not None:
        vfc = [vfc[..., l] for l in range(NLAYER)]
        if virr is not None:
            virr = virr[..., 0]

    # C: Ksat[lindex] = soil_con->Ksat[lindex] / runoff_steps_per_day
    Ksat = [p.Ksat[..., l] / p.runoff_steps_per_day for l in range(NLAYER)]
    nsub = p.runoff_steps_per_day // p.model_steps_per_day

    # runoff.c:104-141 -- evap[l][0] = layer[l].evap / runoff_steps_per_dt,
    # then the frost redistribution. With Nfrost == 1 that loop reduces to
    #     avail_liq = max(moist - ice - resid, 0);  sum_liq = avail_liq
    #     evap_fraction = evap/sum_liq if sum_liq > 0 else 1
    #     evap[l][0] = avail_liq * evap_fraction
    # which returns evap unchanged when there is available liquid and zero
    # when there is none.
    #
    # The redistribution is guarded by `if (evap[lindex][0] > 0)`
    # (runoff.c:108). A *negative* evap -- condensation, which the surface
    # energy balance produces whenever the flux is toward the ground -- takes
    # the else branch at runoff.c:136-140 and is left alone: VIC adds that
    # water to a layer even when the layer is sitting at its residual
    # content, because the availability limit is a limit on withdrawal and
    # condensation is a deposit. Zeroing it there discarded the deposit.
    # Measured against the pristine oracle over a 15-step condensation
    # stretch onto a layer previously dried to residual: 0.75 mm of
    # layer-0 moisture, 0.086 per cent of the trajectory's column storage.
    ev = []
    for l in range(NLAYER):
        e = evap[..., l] / float(nsub)
        avail = torch.clamp(state[..., l] - ice_l[l] - resid[l], min=0.0)
        ev.append(torch.where((e > 0) & (avail <= 0), torch.zeros_like(e), e))

    # runoff.c:155 -- the column moves liquid; the ice sits out the step.
    liq = [state[..., l] - ice_l[l] for l in range(NLAYER)]

    # ---- surface runoff, from the state at the start of the step ---------
    _, runoff = soil.compute_runoff_and_asat(p.b_infilt, p.max_moist,
                                             state, ppt)
    dt_runoff = runoff / float(nsub)
    dt_inflow = ppt / float(nsub)
    Dsmax = p.Dsmax / p.runoff_steps_per_day

    baseflow = torch.zeros_like(ppt)
    zero = torch.zeros_like(ppt)

    for _ in range(nsub):
        inflow = dt_inflow

        # ---- drainage between sublayers, from the pre-update state -------
        Q12 = []
        for l in range(NLAYER - 1):
            # runoff.c:197-200 -- tmp_liq = liq - evap, floored at resid
            tmp_liq = torch.maximum(liq[l] - ev[l], resid[l])
            if mliq is not None:
                # VCS runoff.c:555-559: tmp_liq floored at min_liq
                tmp_liq = torch.maximum(tmp_liq, fl[l])
            q = _brooks_corey_drainage(Ksat[l], tmp_liq, resid[l], maxm[l],
                                       p.expt[..., l])
            # runoff.c:202 -- the gate is on `liq`, the moisture *before*
            # the evaporative withdrawal, not on the floored `tmp_liq`.
            #
            # Under evaporation the two gates give the same number: if evap
            # drives tmp_liq down to the floor then the saturation ratio is
            # zero and the rate is zero anyway, so gating on either is the
            # same value by a different route. They part company under
            # condensation, where evap < 0 lifts tmp_liq *above* residual
            # while liq is still at or below it. VIC drains nothing there,
            # because the layer it is draining has nothing above residual to
            # give; gating on tmp_liq would drain the condensate the instant
            # it landed.
            if core_version.at_least('5.1.0'):
                # VIC 5.1.0 (GH#656): the exact integral of the same law over
                # the sub-step, gated on the post-evaporation moisture
                q = _brooks_corey_integral(Ksat[l], tmp_liq, resid[l],
                                           maxm[l], p.expt[..., l])
                gate = tmp_liq > resid[l]
            else:
                gate = liq[l] > resid[l]
            if mliq is not None:
                # VCS runoff.c:564: the drainage runs where liq > min_liq
                # (a row whose min_liq is not above the residual keeps
                # VIC 5's gate: only the Wpwp rows change)
                gate = torch.where(fl[l] > resid[l], liq[l] > fl[l], gate)
            if vfc is not None:
                # VIC-CropSyst's law (runoff.c:575-621): the 4.x power form
                # with exponent 3 + 2 b_campbell (the caller's expt), capped
                # at the water above Field_Capacity, redistributed to
                # Field_Capacity on an irrigated field, none below it; the
                # gate is liq > min_liq (:564)
                q = _brooks_corey_drainage(Ksat[l], tmp_liq, resid[l],
                                           maxm[l], p.expt[..., l])
                mq = torch.clamp(tmp_liq - vfc[l], min=0.0)
                if virr is not None:
                    q = torch.where(virr & (q < mq), mq, q)
                q = torch.where((q > mq) & (mq > 0.0), mq, q)
                q = torch.where(tmp_liq < vfc[l], torch.zeros_like(q), q)
                gate = liq[l] > (fl[l] if mliq is not None else resid[l])
            Q12.append(torch.where(gate, q, torch.zeros_like(q)))

        # ---- layer update, top down --------------------------------------
        for l in range(NLAYER - 1):
            dtr = dt_runoff if l == 0 else zero

            new_l = liq[l] + (inflow - dtr) - (Q12[l] + ev[l])

            # C: if (liq + ice > max_moist) { excess out; liq = max - ice; }
            over = torch.clamp(new_l - cap[l], min=0.0)
            new_l = torch.minimum(new_l, cap[l])
            liq = liq[:l] + [new_l] + liq[l + 1:]

            if l == 0:
                # C: excess from the top layer goes into Q12, not upward
                Q12[l] = Q12[l] + over
            else:
                liq, surface = _spill_up(liq, cap, over, l)
                runoff = runoff + surface

            # C: if (liq < 0) { Q12 += liq; liq = 0; }
            neg = torch.clamp(liq[l], max=0.0)
            Q12[l] = Q12[l] + neg
            liq = liq[:l] + [torch.clamp(liq[l], min=0.0)] + liq[l + 1:]

            # C: if (liq + ice < resid) { Q12 += (liq+ice) - resid;
            #                             liq = resid - ice; }
            deficit = torch.clamp(liq[l] - floor[l], max=0.0)
            Q12[l] = Q12[l] + deficit
            liq = (liq[:l] + [torch.maximum(liq[l], floor[l])] +
                   liq[l + 1:])

            # C: inflow = Q12[lindex] + tmp_inflow, and tmp_inflow is zero in
            # both branches above by the time it is read.
            inflow = Q12[l]

        # ---- ARNO baseflow from the bottom layer -------------------------
        lb = NLAYER - 1
        rel = soil.relative_moisture(liq[lb], resid[lb], maxm[lb])
        dt_baseflow = soil.arno_baseflow(Dsmax, p.Ds, p.Ws, p.c, rel)

        new_b = liq[lb] + Q12[lb - 1] - (ev[lb] + dt_baseflow)

        # C: if below residual, take the shortfall back out of baseflow
        short = torch.clamp(new_b - floor[lb], max=0.0)
        dt_baseflow = dt_baseflow + short
        new_b = torch.maximum(new_b, floor[lb])

        # C: if above maximum, cascade the excess upward and then to runoff
        over_b = torch.clamp(new_b - cap[lb], min=0.0)
        new_b = torch.minimum(new_b, cap[lb])
        liq = liq[:lb] + [new_b]
        liq, surface = _spill_up(liq, cap, over_b, lb)
        runoff = runoff + surface

        baseflow = baseflow + dt_baseflow

    # C: if (baseflow < 0) { layer.evap += baseflow; baseflow = 0; }
    # The evap adjustment is a diagnostic on the layer, not a state change,
    # so only the clamp affects the trajectory.
    baseflow = torch.clamp(baseflow, min=0.0)

    # Back to total moisture: the ice never moved.
    new_state = torch.stack([liq[l] + ice_l[l] for l in range(NLAYER)],
                            dim=-1)
    if not diagnostics:
        return new_state, runoff, baseflow

    # runoff.c:380-385 -- asat is recomputed from the moisture the step ended
    # at, with an inflow of zero, and it is that second value that reaches
    # put_data.c:647. The saturated fraction reported for a step is therefore
    # the state it leaves behind, not the state it started from, even though
    # the surface runoff was computed from the latter.
    asat, _ = soil.compute_runoff_and_asat(p.b_infilt, p.max_moist,
                                           new_state, torch.zeros_like(ppt))

    diag = {
        # The evaporation the layers could not supply. runoff.c:108-133
        # withholds a layer's demand when it has no liquid above residual
        # -- evap_fraction is 1 and avail_liq is 0, so the applied value
        # is 0 -- and does NOT write that back to layer[lindex].evap, so
        # VIC reports the demand and removes less. That inconsistency is
        # VIC's and is reproduced, but it is not hidden: this is exactly
        # the water OUT_WATER_ERROR will be short by.
        'evap_withheld': (evap - torch.stack(ev, dim=-1)
                          * float(nsub)).sum(-1),
        'asat': asat,
        'inflow': ppt,
        'soil_wet': soil_wetness(new_state, p),
        'rootmoist': (None if p.root is None
                      else root_zone_moisture(new_state, p.root)),
        'zwt': None,
        'zwt_layer': None,
        'zwt_lumped': None,
    }
    if p.depth is not None and p.bubble is not None:
        cv_ = curves or {}
        diag['zwt'], diag['zwt_layer'] = wrap_compute_zwt(
            new_state, p, curve=cv_.get('layer'))
        diag['zwt_lumped'] = compute_zwt_lumped(new_state, p,
                                                curve=cv_.get('lumped'))
    return new_state, runoff, baseflow, diag


def step_gwm(state, ppt, evap, p, diagnostics=False, ice=None, curves=None,
             exact=True):
    """VIC-WUR 3.0's runoff_GWM (vic_run/src/runoff.c:445-773, called
    instead of runoff() when GWM is on, surface_fluxes.c:1086-1093), one
    frost area; arguments and returns as `step`.

    Differences from runoff() (docs/connectors/vicwur.md, C14):
      * the infiltration curve sums every layer (:797-807);
      * every layer, the bottom one included, drains by calc_Q12
        (:609-626) and is updated the same way (:633-676): an excess over
        max_moist and a fall below 0 or below residual are booked into the
        layer's Q12, which may go negative; no ARNO baseflow, so baseflow
        is 0 (OUT_BASEFLOW = 0);
      * after the update, from the bottom layer up to layer 1, a Q12 above
        the layer's Ksat is cut to Ksat and the cut is added to the layer
        ABOVE, spilling further up over max_moist and finally to runoff
        (:677-707). The layer below already took the uncut Q12 as inflow:
        for the bottom layer the water stays in the column, for an interior
        layer the cut is created. `exact=False` (mode fixed) takes the cut
        back from the layer below (an interior layer's) and returns it to
        the draining layer itself, spilling up from there: water conserved;
      * recharge (OUT_RECHARGE) = the cut Q12 of layer N-2, gwrecharge
        (OUT_GWRECHARGE) = Q12 of the bottom layer, summed over the runoff
        sub-steps (:708-737).
    The evaporation split is runoff()'s (:516-556 = runoff.c:86-140).
    """
    N = p.max_moist.shape[-1]
    resid = [p.resid_moist[..., l] for l in range(N)]
    maxm = [p.max_moist[..., l] for l in range(N)]
    if ice is None:
        ice_l = [torch.zeros_like(resid[l]) for l in range(N)]
    else:
        ice_l = [ice[..., l] for l in range(N)]
    Ksat = [p.Ksat[..., l] / p.runoff_steps_per_day for l in range(N)]
    nsub = p.runoff_steps_per_day // p.model_steps_per_day
    ev = []
    for l in range(N):
        e = evap[..., l] / float(nsub)
        avail = torch.clamp(state[..., l] - ice_l[l] - resid[l], min=0.0)
        ev.append(torch.where((e > 0) & (avail <= 0), torch.zeros_like(e), e))
    liq = [state[..., l] - ice_l[l] for l in range(N)]
    _, runoff = soil.compute_runoff_and_asat(p.b_infilt, p.max_moist, state,
                                             ppt, all_layers=True)
    dt_runoff = runoff / float(nsub)
    dt_inflow = ppt / float(nsub)
    zero = torch.zeros_like(ppt)
    recharge, gwrecharge = zero, zero
    for _ in range(nsub):
        inflow = dt_inflow
        Q12 = []
        for l in range(N):
            # :609-626, the gate and the 5.1.0 integral of runoff()
            tmp_liq = torch.maximum(liq[l] - ev[l], resid[l])
            q = _brooks_corey_integral(Ksat[l], tmp_liq, resid[l], maxm[l],
                                       p.expt[..., l])
            Q12.append(torch.where(tmp_liq > resid[l], q,
                                   torch.zeros_like(q)))
        for l in range(N):
            dtr = dt_runoff if l == 0 else zero
            # :638-639, in the C's order
            new = liq[l] + inflow - dtr - Q12[l] - ev[l]
            tot = new + ice_l[l]
            over = tot > maxm[l]                                   # :642
            Q12[l] = torch.where(over, Q12[l] + (tot - maxm[l]), Q12[l])
            new = torch.where(over, maxm[l] - ice_l[l], new)
            neg = new < 0                                          # :654
            Q12[l] = torch.where(neg, Q12[l] + new, Q12[l])
            new = torch.where(neg, torch.zeros_like(new), new)
            tot = new + ice_l[l]
            low = tot < resid[l]                                   # :659
            Q12[l] = torch.where(low, Q12[l] + (tot - resid[l]), Q12[l])
            new = torch.where(low, resid[l] - ice_l[l], new)
            liq[l] = new
            inflow = Q12[l]
        # :677-707, bottom layer up to layer 1
        for l in range(N - 1, 0, -1):
            cut = Q12[l] > Ksat[l]
            extra = torch.where(cut, Q12[l] - Ksat[l], zero)
            Q12[l] = torch.where(cut, Ksat[l], Q12[l])
            if exact:
                first = l - 1
            else:
                first = l
                if l < N - 1:
                    liq[l + 1] = liq[l + 1] - extra
            for tl in range(first, -1, -1):
                on = extra > 0
                liq[tl] = torch.where(on, liq[tl] + extra, liq[tl])
                tot = liq[tl] + ice_l[tl]
                spill = on & (tot > maxm[tl])
                extra = torch.where(spill, tot - maxm[tl], zero)
                liq[tl] = torch.where(spill, maxm[tl] - ice_l[tl], liq[tl])
            runoff = runoff + extra                                # :690-693
        recharge = recharge + Q12[N - 2]
        gwrecharge = gwrecharge + Q12[N - 1]
    new_state = torch.stack([liq[l] + ice_l[l] for l in range(N)], dim=-1)
    baseflow = torch.zeros_like(ppt)
    if not diagnostics:
        return new_state, runoff, baseflow
    asat, _ = soil.compute_runoff_and_asat(p.b_infilt, p.max_moist,
                                           new_state, torch.zeros_like(ppt),
                                           all_layers=True)
    diag = {
        'evap_withheld': (evap - torch.stack(ev, dim=-1)
                          * float(nsub)).sum(-1),
        'asat': asat, 'inflow': ppt,
        'soil_wet': soil_wetness(new_state, p),
        'rootmoist': (None if p.root is None
                      else root_zone_moisture(new_state, p.root)),
        'zwt': None, 'zwt_layer': None, 'zwt_lumped': None,
        'gwrecharge': gwrecharge, 'recharge': recharge,
    }
    if p.depth is not None and p.bubble is not None:
        cv_ = curves or {}
        diag['zwt'], diag['zwt_layer'] = wrap_compute_zwt(
            new_state, p, curve=cv_.get('layer'))
        diag['zwt_lumped'] = compute_zwt_lumped(new_state, p,
                                                curve=cv_.get('lumped'))
    return new_state, runoff, baseflow, diag


def simulate(state0, ppt, evap, p):
    """Run the column over a time series.

    Args:
        state0: [cell, 3] initial moisture, mm
        ppt:    [time, cell] surface water input, mm
        evap:   [time, cell, 3] ET demand, mm
        p:      SoilParams

    Returns:
        states:   [time, cell, 3] moisture after each step
        runoff:   [time, cell]
        baseflow: [time, cell]
    """
    state = state0
    states, runoffs, baseflows = [], [], []
    for t in range(ppt.shape[0]):
        state, r, b = step(state, ppt[t], evap[t], p)
        states.append(state)
        runoffs.append(r)
        baseflows.append(b)
    return (torch.stack(states), torch.stack(runoffs),
            torch.stack(baseflows))


__all__ = ['SoilParams', 'step', 'step_gwm', 'simulate', 'NLAYER', 'zwt_moisture_curve',
           'compute_zwt', 'wrap_compute_zwt', 'distribute_evap_frost',
           'soil_wetness', 'root_zone_moisture', 'MAX_ZWTVMOIST',
           'ZWT_ABSENT', 'bare_soil_evap', 'arno_evap', 'SOIL_RARC']


# a fused region on the CUDA graph path (core.fusion)
from . import fusion as _fusion                                # noqa: E402
step = _fusion.region(step)
