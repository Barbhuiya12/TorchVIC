"""Differentiable PyTorch implementation of VIC's soil water balance.

This is a *reimplementation*, not a wrapper. The C physics cannot be
differentiated in place, so the
equations are written again in tensor form. That makes verification against
the C code the central obligation of this module, not an afterthought:
tests/test_soil_vs_vic.py checks every function here against
tests/oracle/ref_soil.c, which calls VIC's own runoff.c.

Scope, deliberately narrow: everything below. No root finding, so
no implicit differentiation is needed yet; no frozen soil, so the ice terms
are absent rather than approximated.

Shapes: every function is batched over a leading cell axis. Soil layers are
the last axis where present. A cell is independent of every other cell within
a time step, which is the same property the CUDA backend exploits with one
thread per cell.

Units follow VIC: moisture in mm, fluxes in mm per time step, Ksat in mm/day.

References
    Wood, E.F., D.P. Lettenmaier, V.G. Zartarian (1992), A land-surface
        hydrology parameterization with subgrid variability for general
        circulation models, JGR 97(D3), 2717-2728.  [infiltration, eq. 1, 3]
    Franchini, M. and M. Pacciani (1991), Comparative analysis of several
        conceptual rainfall-runoff models, J. Hydrol. 122, 161-219.  [ARNO]
"""
import torch


# ---------------------------------------------------------------------------
# Numerical guards
#
# VIC's expressions contain powers whose base reaches zero at physically
# attainable states: (1 - W/Wmax) at saturation, and (moist - resid) at the
# residual moisture content. In C the result is 0, or an infinity that a later
# clip absorbs. Under autodiff the *derivative* there is infinite, and one
# infinite derivative anywhere in a batch makes the whole gradient NaN. So a
# guard is needed.
#
# The guard must not be a floor chosen by eye. An earlier revision of this
# module clamped every power base at 1e-12, on the reasoning that 1e-12 is
# "small". It is not small here: the inner term of calc_Q12 legitimately
# reaches 1.6e-40 for expt = 25, and clamping it to 1e-12 changed Q12 from
# 0.80 mm to 43.16 mm -- a 54x error that the oracle comparison caught. The
# dynamic range of an intermediate has to be measured before anything is
# clamped to it.
#
# Two guards are used instead, each matched to what the limit actually is:
#
#   _pow_nonneg   for a *positive* exponent, where base -> 0 gives 0. The
#                 value is exact wherever base > 0 and is 0 at base <= 0,
#                 which is the mathematical limit, not an approximation. The
#                 double-`where` keeps NaN out of the backward pass: without
#                 the inner substitution, pow() of a negative base produces a
#                 NaN gradient that `where` would propagate even from the
#                 branch it discards.
#
#   TINY          for a *negative* exponent, where base -> 0 diverges and
#                 there is no finite limit to substitute. The base is floored
#                 at the smallest positive value that cannot displace a real
#                 intermediate. 1e-300 is ~260 decades below the smallest
#                 intermediate observed in the oracle sweep, so it acts only
#                 on inputs where the C itself returns a non-finite value.
# ---------------------------------------------------------------------------
TINY = 1e-300

# Floors for the bases of the Brooks-Corey powers in calc_Q12. TINY is the
# right floor for a quantity that is only ever multiplied, and the wrong one
# for a base raised to a large negative power.
#
# calc_Q12 evaluates avail**(1 - expt) with expt between 3 and 12, so the
# exponent is -2 to -11. At a floor of 1e-300 that is 1e3300, which overflows
# a double to inf; the forward pass then survives, because inf - x is inf and
# inf**negative is 0, while the gradient with respect to expt is
# base**exp * ln(base), which is inf * -690 and therefore NaN. That NaN is
# what stopped training: PowBackward1 returning NaN in its gradient for the
# exponent, once per dry layer.
#
# The floors below are physical rather than numerical. A layer within a
# nanometre of its residual moisture, or with a capacity below a micrometre
# of water, is a degenerate state and not a state the model is asked to
# resolve. At 1e-6 the worst case is 1e-6**-11 = 1e66, comfortably inside a
# double.
Q12_AVAIL_FLOOR = 1e-6      # mm of water above residual
Q12_SPAN_FLOOR = 1e-3       # mm of water capacity
# 1e-30 was too high, and the oracle sweep says by how much: it clamped 60
# of 432 rows on which the C returns a perfectly finite number, every one of
# them at expt = 25, where the bracketed term falls to 1.59e-40. calc_Q12
# then returned 5.38 where runoff.c returns 2.98e-08 -- a relative error of
# 1.8e8, not a rounding difference. The docstring's claim that the floor
# acts "on inputs where the C produces NaN and only there" was false.
#
# The floor still has to exist: the limit as the term goes to zero is +inf
# and the derivative is p * inner**(p-1) with p = 1/(1 - expt) in
# [-1/2, -1/24], so a floor near double underflow overflows the *gradient*
# (1e-300**-1.5 is inf) even though the value survives. 1e-60 is 20 orders
# below the smallest term any good oracle row produces (3.7e-28) and leaves
# inner**(p-1) at most 1e90. Measured at 1e-60: all 432 rows agree to
# 9.2e-13, and every gradient is finite at expt 3, 8, 12 and 25.
Q12_INNER_FLOOR = 1e-60     # the bracketed term, raised to 1/(1 - expt)


def _pow_nonneg(base, exponent):
    """base ** exponent for exponent > 0, with 0 at base <= 0.

    Exact wherever base > 0. At base <= 0 both the value and the gradient are
    zero: the value because that is the limit, the gradient because the
    derivative does not exist there and a state on the boundary must not
    contribute one.
    """
    positive = base > 0
    safe = torch.where(positive, base, torch.ones_like(base))
    out = torch.pow(safe, exponent)
    return torch.where(positive, out, torch.zeros_like(out))


def compute_runoff_and_asat(b_infilt, max_moist, moist, inflow,
                            all_layers=False):
    """Variable infiltration curve: saturated fraction and surface runoff.

    Mirrors compute_runoff_and_asat() in vic/vic_run/src/runoff.c.

    Only the top Nlayer-1 layers enter the curve, exactly as in the C
    function, which loops `lindex < options.Nlayer - 1`; with `all_layers`
    every layer does (VIC-WUR 3.0 with GWM, runoff.c:797-807).

    Args:
        b_infilt:  [cell] infiltration shape parameter (VIC `b_infilt`)
        max_moist: [cell, layer] maximum moisture per layer, mm
        moist:     [cell, layer] current moisture (liquid + ice), mm
        inflow:    [cell] water arriving at the surface this step, mm

    Returns:
        A:      [cell] saturated area fraction
        runoff: [cell] surface runoff, mm
    """
    k = None if all_layers else -1
    top_moist = moist[..., :k].sum(dim=-1)
    top_max = max_moist[..., :k].sum(dim=-1)

    # C: if (top_moist > top_max_moist) top_moist = top_max_moist;
    top_moist = torch.minimum(top_moist, top_max)

    # A = 1 - (1 - W/Wmax)^(b/(1+b))            Wood et al. eq. (1)
    ex = b_infilt / (1.0 + b_infilt)
    rel = torch.clamp(1.0 - top_moist / torch.clamp(top_max, min=TINY),
                      min=0.0, max=1.0)
    A = 1.0 - _pow_nonneg(rel, ex)

    max_infil = (1.0 + b_infilt) * top_max
    i_0 = max_infil * (1.0 - _pow_nonneg(1.0 - A, 1.0 / b_infilt))

    # Saturation-excess branch, eq. (3a):   Q = P - Wmax + W
    runoff_sat = inflow - top_max + top_moist

    # Infiltration-excess branch, eq. (3b)
    basis = torch.clamp(1.0 - (i_0 + inflow) / torch.clamp(max_infil, min=TINY),
                        min=0.0)
    runoff_unsat = (inflow - top_max + top_moist +
                    top_max * _pow_nonneg(basis, 1.0 + b_infilt))

    # The C function's branch ladder, in order:
    #   inflow == 0            -> 0
    #   max_infil == 0         -> inflow
    #   (i_0 + inflow) > max_infil -> (3a)
    #   otherwise              -> (3b)
    runoff = torch.where((i_0 + inflow) > max_infil, runoff_sat, runoff_unsat)
    runoff = torch.where(max_infil == 0.0, inflow, runoff)
    runoff = torch.where(inflow == 0.0, torch.zeros_like(runoff), runoff)

    # C: if (*runoff < 0.) *runoff = 0.;
    runoff = torch.clamp(runoff, min=0.0)
    return A, runoff


def calc_Q12(Ksat, init_moist, resid_moist, max_moist, expt):
    """Brooks-Corey drainage from one layer to the next.

    Mirrors calc_Q12() in vic/vic_run/src/runoff.c:

        Q12 = init_moist
              - ( (init_moist - resid)^(1-expt)
                  - Ksat / (max - resid)^expt * (1 - expt) )^(1/(1-expt))
              - resid

    Every argument is [cell, layer] or broadcastable to it.

    The inner base can go negative for large Ksat and small moisture; in C
    that yields NaN, which a later clip in runoff() never repairs because NaN
    survives comparison. It is floored here. The floor is a deviation from
    the C wherever it binds, so it is set low enough to bind only where the
    C is already NaN: at Q12_INNER_FLOOR = 1e-60 all 432 rows of the oracle
    sweep agree to 9.2e-13. It used to be 1e-30, which also clamped 60 rows
    where the C returns a finite number -- see the note on the constant.

    Not on the model's path. `column._brooks_corey_drainage` is what the
    column calls; this is VIC 4.0.x's closed-form integral of the same law,
    kept because the repository's own `vic/` tree still contains it.
    """
    one_minus_expt = 1.0 - expt
    avail = torch.clamp(init_moist - resid_moist, min=Q12_AVAIL_FLOOR)
    span = torch.clamp(max_moist - resid_moist, min=Q12_SPAN_FLOOR)

    inner = (torch.pow(avail, one_minus_expt) -
             Ksat / torch.pow(span, expt) * one_minus_expt)
    # expt > 1 in every VIC soil parameter set, so 1/(1 - expt) is negative
    # and the limit as inner -> 0 is +inf, not 0. _pow_nonneg's zero fallback
    # would be wrong here; floor at TINY instead, which is far below any
    # intermediate the model produces (see the note on guards above).
    return (init_moist -
            torch.pow(torch.clamp(inner, min=Q12_INNER_FLOOR),
                      1.0 / one_minus_expt) -
            resid_moist)


def arno_baseflow(Dsmax_per_step, Ds, Ws, c, rel_moist):
    """ARNO baseflow from the bottom soil layer.

    Mirrors the block inside runoff() in vic/vic_run/src/runoff.c:

        frac = Dsmax * Ds / Ws
        Qb   = frac * rel_moist
        if rel_moist > Ws:
            frac = (rel_moist - Ws) / (1 - Ws)
            Qb  += Dsmax * (1 - Ds/Ws) * frac^c
        Qb = max(Qb, 0)

    Note Dsmax_per_step is already divided by runoff_steps_per_day, as the C
    does before the sub-step loop.

    The curve is continuous at rel_moist == Ws and its derivative is not
    (unless c == 1). That kink is kept: it is the model, and a subgradient is
    well defined. Ws == 1 makes the nonlinear branch unreachable *and*
    singular; it is guarded so the gradient stays finite there.
    """
    linear = Dsmax_per_step * Ds / torch.clamp(Ws, min=TINY) * rel_moist

    denom = torch.clamp(1.0 - Ws, min=TINY)
    frac = torch.clamp((rel_moist - Ws) / denom, min=0.0)
    nonlinear = (Dsmax_per_step * (1.0 - Ds / torch.clamp(Ws, min=TINY)) *
                 _pow_nonneg(frac, c))

    q = linear + torch.where(rel_moist > Ws, nonlinear,
                             torch.zeros_like(nonlinear))
    return torch.clamp(q, min=0.0)


def nijssen2001_to_arno(d1, d2, d3, d4, max_moist_mm):
    """NIJSSEN2001's d1..d4 as ARNO's Ds, Dsmax, Ws, c.

    VIC reads the four baseflow columns into Ds, Dsmax, Ws, c and then,
    when BASEFLOW is NIJSSEN2001, converts them in place -- the image
    driver in vic/drivers/shared_image/src/vic_init.c:815-829, the
    classic driver in vic/drivers/classic/src/read_soilparam.c:693-707,
    identical text:

        Dsmax' = Dsmax * (1 / (max_moist - Ws))^-c + Ds * max_moist
        Ds'    = Ds * Ws / Dsmax'
        Ws'    = Ws / max_moist

    with max_moist the bottom layer's, in mm. c is already d4 and is
    returned untouched. The operations below follow that order exactly,
    with no guards: a valid file has d3 < max_moist, so the base is
    positive, and this runs once at load, outside any gradient.
    """
    Dsmax = (d2 * torch.pow(1.0 / (max_moist_mm - d3), -d4) +
             d1 * max_moist_mm)
    Ds = d1 * d3 / Dsmax
    Ws = d3 / max_moist_mm
    return Ds, Dsmax, Ws, d4


def relative_moisture(liq, resid_moist, max_moist):
    """(liq - resid) / (max - resid), the argument ARNO baseflow takes."""
    return ((liq - resid_moist) /
            torch.clamp(max_moist - resid_moist, min=TINY))


__all__ = ['compute_runoff_and_asat', 'calc_Q12', 'arno_baseflow',
           'nijssen2001_to_arno', 'relative_moisture', 'TINY']
