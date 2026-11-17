"""Differentiating through a root find, for VIC's snow surface temperature.

VIC solves the snowpack surface energy balance for the temperature that zeroes
it, using `root_brent` (vic/vic_run/src/root_brent.c). Five call sites across
the snow path do this. None of them can be differentiated by transcription:
Brent's bracketing has non-differentiable branches and a trip count that
depends on the data, so unrolling it would both blow up memory and attach
gradients to the *path* the solver took rather than to the answer.

The standard remedy is to differentiate the solution instead of the solver.
If T* satisfies

    f(T*, p) = 0

then implicitly differentiating gives

    dT*/dp = -(df/dT)^-1 (df/dp)        evaluated at T*

which needs no knowledge of how T* was found. In practice that is obtained by
solving for T* without gradients and then taking a single Newton step with
gradients enabled:

    T = T* - f(T*, p) / f'(T*, p)

At convergence f(T*) is zero to solver tolerance, so the returned value is T*
to that same tolerance, while autograd sees the expression on the right and
produces exactly the implicit-function-theorem derivative.

**On bit-exactness.** Everything else in torchvic matches VIC's output bit
for bit, and that standard cannot hold here. A root is found by iteration, so
it is defined only to a tolerance, and reproducing VIC's answer bit for bit
would mean reimplementing Brent's exact sequence of trial points -- which is
the thing that cannot be differentiated. Agreement is therefore to solver
tolerance, and the tolerance is stated with any result that depends on it.
"""
import math

import os

import torch

from . import fusion, graphsafe
import torch.autograd.forward_ad as fwAD


# VIC's own convergence tolerance for the snow surface temperature root find:
# param.ROOT_BRENT_T in vic/drivers/shared_all/src/initialize_parameters.c.
# It is not the default here, because this solver is generic and a caller
# solving for something other than a temperature in degrees Celsius may need
# a tighter absolute tolerance; the snow path passes it explicitly. Solving
# tighter than the model being reproduced buys nothing, and on an 80 degree
# bracket every extra decade costs about 3.3 bisections.
ROOT_BRENT_T = 1.0e-7
#: root_brent.c:168-179's bracket search. When f(a) and f(b) share a sign the
#: C widens the interval by ROOT_BRENT_TSTEP on BOTH sides and tries again, at
#: most ROOT_BRENT_MAXTRIES times (initialize_parameters.c:226-228).
ROOT_BRENT_TSTEP = 10.0
ROOT_BRENT_MAXTRIES = 5


def widen_bracket(f, lo, hi, args=(), tstep=ROOT_BRENT_TSTEP,
                  maxtries=ROOT_BRENT_MAXTRIES):
    """root_brent.c:168-179's bracket search, vectorised.
    Returns (lo, hi, failed).

    VIC does NOT take its caller's bracket as given. Where f(lo) and f(hi)
    share a sign it widens by `tstep` on both sides and tries again, up to
    `maxtries` times, and gives up after that. Callers that hand a bracketed
    solver VIC's interval as written get a bound rather than a root whenever
    the root lies outside it -- which for VIC's own callers is common, since
    several of them open with a fixed half-width around the previous step's
    value.

    Per column this is a mask: a column already bracketed keeps its bounds
    while the rest widen, which is what the scalar loop does to each in turn.
    `failed` marks the columns still unbracketed after `maxtries`, where VIC
    returns ERROR -- the caller is told rather than handed a bound.
    """
    ok = (f(lo, *args) * f(hi, *args)) < 0.0
    for _ in range(maxtries):
        if bool(ok.all()):
            break
        widen = ~ok
        lo = torch.where(widen, lo - tstep, lo)
        hi = torch.where(widen, hi + tstep, hi)
        ok = (f(lo, *args) * f(hi, *args)) < 0.0
    return lo, hi, ~ok


def bisect(f, lo, hi, args=(), tol=1e-12, max_iter=200):
    """Bracketed bisection, no gradients. Returns the root.

    Bisection rather than Brent because the two converge to the same root and
    bisection cannot fail once the interval brackets it. Speed does not
    matter: this runs under `no_grad` and its cost is not on the backward
    pass.

    `lo` and `hi` must bracket a sign change. Where they do not, the interval
    is returned unchanged and the caller must decide what that means -- VIC
    has the same obligation and handles it by testing the melting case before
    calling the solver at all.
    """
    with torch.no_grad():
        lo = lo.clone()
        hi = hi.clone()
        flo = f(lo, *args)
        # Bisection halves the bracket every iteration, so the number of
        # iterations needed to reach `tol` follows from the initial width and
        # does not have to be discovered by testing.
        #
        # Testing it costs more than it saves on a GPU. `bool(tensor)` forces
        # a device-to-host synchronisation, which stalls the queue once per
        # iteration; at roughly 47 iterations per solve and one solve per
        # model time step that dominated the run. Measured on a 296-cell
        # forward model: 246 ms/step with the test, and the arithmetic below
        # is exact rather than approximate, so nothing is traded for it.
        width = float((hi - lo).abs().max())
        need = (int(math.ceil(math.log2(width / tol)))
                if width > tol else 0)
        for _ in range(min(max_iter, max(need, 1))):
            mid = 0.5 * (lo + hi)
            fmid = f(mid, *args)
            same = (torch.sign(fmid) == torch.sign(flo))
            lo = torch.where(same, mid, lo)
            flo = torch.where(same, fmid, flo)
            hi = torch.where(same, hi, mid)
        return 0.5 * (lo + hi)


#: Illinois iterations `solve` takes before falling back on the bracket's
#: midpoint. Chosen by measurement, not by a convergence test: testing costs a
#: device-to-host synchronisation per iteration, which is what made bisection
#: expensive in the first place. See tests/test_root_solver.py, which
#: holds the measured agreement against bisection on the two residuals this
#: model actually solves.
ILLINOIS_ITERS = 12

#: Which bracketed solver `solve` uses when a caller does not say. Module
#: level rather than a call argument so that a whole run can be switched to
#: 'bisect' and compared against itself -- which is how the default was
#: chosen, and how a doubt about it should be settled again.
DEFAULT_METHOD = 'illinois'


def illinois(f, lo, hi, args=(), iters=ILLINOIS_ITERS):
    """Bracketed false position with the Illinois modification, no gradients.

    Same guarantee as `bisect` -- the bracket always holds the sign change,
    so the answer can never leave it -- at a fraction of the evaluations.
    Regula falsi keeps one endpoint forever when the function is convex over
    the bracket, which is the usual case for a surface energy balance; the
    Illinois modification (Dowell and Jarratt 1971) halves the stale
    endpoint's residual each time it is retained, which breaks that stall and
    restores superlinear convergence.

    Fixed iteration count, deliberately. `bisect` computes its count from the
    bracket width because bisection halves exactly; Illinois has no such
    closed form, and asking whether it has converged costs a synchronisation
    per iteration -- the same cost that the comment in `bisect` records as
    dominating a 296-cell step. So the count is fixed, the result is
    guaranteed to lie in the bracket, and how close it actually gets is
    measured rather than asserted.

    `lo` and `hi` must bracket a sign change; where they do not, the result is
    inside the interval and means nothing, exactly as for `bisect`.

    **Twelve iterations is tuned to this model's residuals and is not a
    general guarantee, and a caller pointing this at something else should
    know that.** `bisect` halves the bracket every iteration whatever the
    function does; Illinois has no such floor. On a residual whose magnitude
    spans the bracket by orders of magnitude, false position parks the trial
    point on one endpoint and the Illinois halving takes one bit off the
    stale side per iteration: `exp(2T) - 7` over [0, 20], where f(hi) is
    2.4e17 and f(lo) is -6, is 1.6 out after twelve iterations and needs
    about sixty to converge, while `bisect` returns the root exactly. The
    energy balances this model actually solves are smooth and of moderate
    range over their brackets -- the oracle tests measure the snow, ice and
    soil-node roots against VIC's own at 1e-7 or better -- which is why
    twelve is enough for them. It is not enough for everything. Use
    `method='bisect'`, or raise `iters`, for anything stiff.
    """
    with torch.no_grad():
        lo = lo.clone()
        hi = hi.clone()
        flo = f(lo, *args)
        fhi = f(hi, *args)
        # Which side holds the most recent evaluation. The halving below
        # applies only to the *other* one, and getting this wrong is the
        # difference between superlinear convergence and something slower
        # than bisection: halving an endpoint that was itself just replaced
        # throws away the information in it. Measured on exp(x) - 5 over
        # [-5, 5], 12 iterations: 2.2e-03 halving unconditionally, 5.4e-06
        # halving only the stale side. The C's own a/b formulation has this
        # built in, because there `b` is always the newest point.
        newest_hi = torch.ones_like(lo, dtype=torch.bool)
        # The best point seen, not the last one. Illinois is not monotone in
        # |residual|: the Illinois halving deliberately moves the secant off
        # a stale endpoint, and the trial point that follows can be worse
        # than the one before it. Keeping the best costs one comparison an
        # iteration and can only help.
        best = 0.5 * (lo + hi)
        best_f = torch.full_like(lo, float('inf'))
        for _ in range(int(iters)):
            width = hi - lo
            den = fhi - flo
            # A flat or non-finite secant has no false position to take; the
            # midpoint always does, and it is what bisection would have used.
            ok = torch.isfinite(den) & (den.abs() > 0.0)
            safe_den = torch.where(ok, den, torch.ones_like(den))
            x = hi - fhi * width / safe_den
            x = torch.where(ok, x, 0.5 * (lo + hi))
            # Keep the trial point strictly inside the bracket. Without this
            # a rounding step can put it on an endpoint, the bracket stops
            # shrinking, and the iteration stalls at the wrong value.
            eps = 1e-12 * width.abs()
            x = torch.minimum(torch.maximum(x, lo + eps), hi - eps)
            fx = f(x, *args)
            better = fx.abs() < best_f
            best = torch.where(better, x, best)
            best_f = torch.where(better, fx.abs(), best_f)
            same = torch.sign(fx) == torch.sign(flo)
            # Illinois: an endpoint that is kept a second time has its
            # residual halved, which drags the secant off the side that has
            # stopped moving. `same` means the root is above x, so lo takes
            # it and hi is the one kept.
            halve_hi = same & ~newest_hi
            halve_lo = (~same) & newest_hi
            new_flo = torch.where(same, fx,
                                  torch.where(halve_lo, 0.5 * flo, flo))
            new_fhi = torch.where(same,
                                  torch.where(halve_hi, 0.5 * fhi, fhi), fx)
            lo = torch.where(same, x, lo)
            hi = torch.where(same, hi, x)
            flo, fhi = new_flo, new_fhi
            newest_hi = ~same
        # The best trial point by |residual|, not the bracket's midpoint:
        # Illinois converges on the iterate long before the bracket
        # collapses around it, so the midpoint is the worse answer of the
        # two. "Best" rather than "last" because the forced bisection above
        # can put a worse point after a better one.
        return best


#: root_brent.c's ROOT_BRENT_MAXITER (initialize_parameters.c 5.1.0:217).
ROOT_BRENT_MAXITER = 1000


#: debugging aid (TORCHVIC_LIVE_PROBE=1, eager path): the elements still
#: iterating at each iteration of every root_brent, by (kind, lanes)
LIVE_PROBE = os.environ.get('TORCHVIC_LIVE_PROBE', '0') != '0'
LIVE_HIST = {}
if LIVE_PROBE:
    import atexit as _atexit

    def _print_hist():
        import sys
        for k_, h_ in sorted(LIVE_HIST.items(), key=str):
            for it_, v_ in sorted(h_.items()):
                v_ = sorted(v_)
                print('live %s iter %d calls %d median %d p90 %d max %d'
                      % (k_, it_, len(v_), v_[len(v_) // 2],
                         v_[int(0.9 * (len(v_) - 1))], v_[-1]),
                      file=sys.stderr)
    _atexit.register(_print_hist)


#: iterations the eager root_brent took to stop every element (iteration
#: index -> calls), for choosing GRAPH_BRENT_ITERS
BRENT_ITERS = {}
#: iterations a root_brent in a CUDA graph runs inside the step's graph;
#: the rest are a graph of their own, replayed while any element is going
GRAPH_BRENT_ITERS = 6


def root_brent(f, a, b, fa, fb, args=(), t_tol=ROOT_BRENT_T,
               max_iter=ROOT_BRENT_MAXITER, live_ctx=None, compact=False):
    """VIC's root_brent.c search, after its bracket, vectorised; no gradients.

    `a`, `b` bracket the root with residuals `fa`, `fb` already evaluated
    (root_brent.c 5.1.0:83-226 is the bracket search, which the caller has
    done). This is :227-294 step for step -- the same trial points, the
    same stopping rule |m| <= 2 DBL_EPSILON |b| + ROOT_BRENT_T or f(b) = 0
    -- per element, each element stopping where the C would and returning
    its `b`. VIC's answer is therefore reproduced to the rounding of the
    residual, not merely to its tolerance: a root solved tighter than VIC's
    (the Illinois iteration takes the balance to ~1e-13 W m-2, VIC stops at
    ~1e-6) sits up to ROOT_BRENT_T away from VIC's, and a soil-moisture
    threshold downstream can then be crossed one step apart.

    One residual evaluation per iteration for every element; an element
    that has stopped is evaluated again at its own `b` and its state held.
    Returns the root (elements that never stop within `max_iter`, where the
    C returns ERROR, return their last `b`).

    `live_ctx`, if given, is called with the mask of the elements still
    iterating and returns a context the eager iteration evaluates `f`
    in: the residual's value is kept only there, so `f` may skip work
    elsewhere (soiltemp.only).
    """
    eps = torch.finfo(a.dtype).eps
    with torch.no_grad():
        z, stop, one = _brent_start(a, b, fa, fb, eps, t_tol)
        seg = graphsafe.SEGMENTER if graphsafe.ACTIVE else None
        for _ in range(int(max_iter)):
            if graphsafe.ACTIVE:
                if _ + 1 == GRAPH_BRENT_ITERS:
                    # in a graph: the iterations from here on are a graph
                    # of their own, replayed until every element counted
                    # has stopped (graphsafe.Segmenter); on the eager
                    # graph-safe path the same loop, run (graphsafe.run_loop)
                    into = (lambda zs_, stop_: _brent_iter_into(
                        zs_, stop_, eps, t_tol, one, f, args))
                    # `compact`: the lanes still iterating are the not-done
                    # ones (graphsafe.compact_div)
                    live_of = ((lambda z_: _brent_live(z_[8], graphsafe.MASK))
                               if compact else None)
                    if seg is not None:
                        z = seg.loop(z, lambda z_: _brent_iter(
                            z_, eps, t_tol, one, f, args),
                            lambda z_: _brent_stopped(z_),
                            max_iter - _ - 1, step_into=into,
                            live_of=live_of)
                    else:
                        z = graphsafe.run_loop(
                            z, lambda z_: _brent_stopped(z_),
                            max_iter - _ - 1, into, live_of=live_of)
                    break
                # the graph-safe path runs the first iterations as the
                # capture does, all of them (an iteration after every
                # element counted has stopped changes none of them), so
                # the regions it records are compiled here, before it
            elif bool(stop):
                BRENT_ITERS[_] = BRENT_ITERS.get(_, 0) + 1
                break
            if LIVE_PROBE and not graphsafe.ACTIVE:
                _h = LIVE_HIST.setdefault(('brent', live_ctx is not None,
                                           int(z[0].numel())), {})
                _h.setdefault(_, []).append(int((~z[8]).sum()))
            # _brent_tail, then the next iteration's _brent_head
            z, live = _brent_trial(z, eps, t_tol, one)
            if live_ctx is not None and not graphsafe.ACTIVE:
                with live_ctx(live):
                    fb_new = f(z[1], *args)
            else:
                fb_new = f(z[1], *args)
            z, stop = _brent_take_head(z, live, fb_new, eps, t_tol)
        done, out, b = z[8], z[9], z[1]
        out = torch.where(done, out, b)
        return out


def _brent_start_(a, b, fa, fb, eps, t_tol):
    """root_brent.c 5.1.0:227-247: the solver's state from the bracket,
    and the first bookkeeping and stopping test (_brent_head); one fused
    region on the graph path."""
    a, b, fa, fb = a.clone(), b.clone(), fa.clone(), fb.clone()
    c = a.clone()
    fc = fb.clone()                                             # :229
    d = b - a
    e = d.clone()
    done = torch.zeros_like(a, dtype=torch.bool)
    out = b.clone()
    one = torch.ones_like(a)
    z, stop = _brent_head_([a, b, c, fa, fb, fc, d, e, done, out], eps,
                           t_tol)
    return z, stop, one


def _brent_stopped(z):
    """Every element that counts (graphsafe.MASK) has stopped: [1] bool."""
    return _brent_stopped_mask(z[8], graphsafe.MASK)


def _brent_stopped_mask_(done, mask):
    if mask is not None:
        done = done | ~mask
    return done.all().reshape(1)


def _brent_live_(done, mask):
    """The lanes a loop iteration still has to run: not done, and counted."""
    return ~done if mask is None else (~done & mask)


def _brent_head_(z, eps, t_tol):
    """root_brent.c 5.1.0:229-247: the bracket's bookkeeping and the
    stopping test. Returns the state and whether every element stopped."""
    a, b, c, fa, fb, fc, d, e, done, out = z
    s1 = (fb * fc) > 0
    c = torch.where(s1, a, c)
    fc = torch.where(s1, fa, fc)
    d = torch.where(s1, b - a, d)
    e = torch.where(s1, d, e)
    sw = fc.abs() < fb.abs()
    a, b, c = (torch.where(sw, b, a), torch.where(sw, c, b),
               torch.where(sw, b, c))
    fa, fb, fc = (torch.where(sw, fb, fa), torch.where(sw, fc, fb),
                  torch.where(sw, fb, fc))
    tol = 2 * eps * b.abs() + t_tol                     # :245
    m = 0.5 * (c - b)                                   # :246
    conv = ((m.abs() <= tol) | (fb == 0)) & ~done       # :247
    out = torch.where(conv, b, out)
    done = done | conv
    return [a, b, c, fa, fb, fc, d, e, done, out], done.all()


def _brent_trial_(z, eps, t_tol, one):
    """root_brent.c 5.1.0:248-286: the next trial point `b` for the
    elements still going (the others keep theirs); and which are going.
    Its residual is :287, taken in by _brent_take_."""
    a, b, c, fa, fb, fc, d, e, done, out = z
    tol = 2 * eps * b.abs() + t_tol
    m = 0.5 * (c - b)
    bis = (e.abs() < tol) | (fa.abs() <= fb.abs())
    sfa = torch.where(fa != 0, fa, one)
    sfc = torch.where(fc != 0, fc, one)
    s = fb / sfa
    lin = a == c
    q_ = fa / sfc
    r_ = fb / sfc
    p = torch.where(lin, 2 * m * s,
                    s * (2 * m * q_ * (q_ - r_) - (b - a) * (r_ - 1)))
    q = torch.where(lin, 1 - s, (q_ - 1) * (r_ - 1) * (s - 1))
    pos = p > 0
    q = torch.where(pos, -q, q)
    p = torch.where(pos, p, -p)
    acc = (((2 * p) < (3 * m * q - (tol * q).abs()))
           & (p < (0.5 * e * q).abs()))
    sq = torch.where(q != 0, q, one)
    d_new = torch.where(bis | ~acc, m, p / sq)
    e_new = torch.where(bis | ~acc, m, d)
    b_new = b + torch.where(d_new.abs() > tol, d_new,
                            torch.where(m > 0, tol, -tol))
    live = ~done
    d = torch.where(live, d_new, d)
    e = torch.where(live, e_new, e)
    a = torch.where(live, b, a)
    fa = torch.where(live, fb, fa)
    b = torch.where(live, b_new, b)
    return [a, b, c, fa, fb, fc, d, e, done, out], live


def _brent_take_(z, live, fb_new):
    """root_brent.c 5.1.0:287: the trial point's residual, for the
    elements still going."""
    z = list(z)
    z[4] = torch.where(live, fb_new, z[4])
    return z


def _brent_take_head_(z, live, fb_new, eps, t_tol):
    """_brent_take, then the next iteration's _brent_head: one fused
    region on the graph path (core.fusion)."""
    return _brent_head_(_brent_take_(z, live, fb_new), eps, t_tol)


def _brent_take_head_into_(zs, stop, z, live, fb_new, eps, t_tol, mask):
    """_brent_take_head written into the loop's own buffers `zs`, and the
    stop test (`_brent_stopped`) into `stop`: the iteration of a captured
    loop (graphsafe.Segmenter) as one fused region, without a copy of the
    state per iteration. The same values as _brent_take_head and
    _brent_stopped."""
    res = _brent_head_(_brent_take_(z, live, fb_new), eps, t_tol)[0]
    for d, v in zip(zs, res):
        if d is not v:
            d.copy_(v)
    stop.copy_(_brent_stopped_mask_(res[8], mask))


def _brent_iter_into(zs, stop, eps, t_tol, one, f, args):
    """_brent_iter in place: `zs` and `stop` are the loop's buffers."""
    z, live = _brent_trial(zs, eps, t_tol, one)
    _brent_take_head_into(zs, stop, z, live, f(z[1], *args), eps, t_tol,
                          graphsafe.MASK)


def _brent_iter(z, eps, t_tol, one, f, args):
    """One whole iteration: the trial point, then the bookkeeping and the
    test (the loop above, shifted by its head)."""
    z, live = _brent_trial(z, eps, t_tol, one)
    return _brent_take_head(z, live, f(z[1], *args), eps, t_tol)[0]


# fused regions on the CUDA graph path, the functions themselves elsewhere
_brent_head = fusion.region(_brent_head_)
_brent_start = fusion.region(_brent_start_)
_brent_trial = fusion.region(_brent_trial_)
_brent_take_head = fusion.region(_brent_take_head_)
_brent_take_head_into = fusion.region(_brent_take_head_into_)
_brent_stopped_mask = fusion.region(_brent_stopped_mask_)
_brent_live = fusion.region(_brent_live_)


def _detach_tree(x):
    """`x` with every tensor in it detached, through dicts, lists, tuples.

    Returns new containers; the originals and their tensors are untouched.
    """
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, dict):
        return {k: _detach_tree(v) for k, v in x.items()}
    if isinstance(x, tuple) and hasattr(x, '_fields'):        # namedtuple
        return type(x)(*[_detach_tree(v) for v in x])
    if isinstance(x, (list, tuple)):
        return type(x)(_detach_tree(v) for v in x)
    return x


#: How often _local_dfdt fell back to reverse mode; tests hold it at zero
#: on the model's own solves, so a slow path cannot come back silently.
REVERSE_FALLBACKS = [0]


def _local_dfdt(f, root, args):
    """df/dT at `root`, elementwise, without walking the model's history.

    The residual is elementwise in T, so grad(f(T).sum(), T) is its
    diagonal derivative, and so is the forward-mode derivative of f along
    a tangent of ones. The two give the same partial derivative. They
    differ in cost.

    Reverse mode, `torch.autograd.grad`, walks everything reachable from
    its output before it runs, whichever inputs it is asked about. `args`
    are the model's state and parameters, connected to every earlier time
    step in a run with gradients, and so are the tensors the residual's CLOSURES
    capture -- the evaporation callbacks close over soil moisture, the
    canopy store, the parameters -- which no detaching of `args` can
    reach. So this one local derivative cost a traversal of the whole
    history, once per solve, every step: a forward pass with gradients
    ran 222 ms per step on day 1 and 473 on day 10, while the same run
    without gradients held flat at 181. Quadratic in the window, from one
    line.

    Forward mode carries a tangent on T through the single evaluation and
    never looks at the backward graph, so its cost is the evaluation's,
    whatever the history. It runs under no_grad, which does not affect
    forward-mode AD, so the throwaway evaluation builds no backward graph
    either. An op without a forward derivative falls back to reverse mode
    on detached `args` -- correct, and cheap wherever closures hold nothing
    live. tests/test_implicit_cost.py holds the derivative equal
    to reverse mode's and the per-step cost flat.
    """
    t = root.detach()
    try:
        with fwAD.dual_level(), torch.no_grad():
            ft = f(fwAD.make_dual(t, torch.ones_like(t)), *args)
            dfdt = fwAD.unpack_dual(ft).tangent
        if dfdt is not None:
            return dfdt
    except (NotImplementedError, RuntimeError):
        pass
    REVERSE_FALLBACKS[0] += 1
    t = t.requires_grad_(True)
    local = _detach_tree(args)
    with torch.enable_grad():
        ft = f(t, *local)
        dfdt, = torch.autograd.grad(ft.sum(), t)
    return dfdt


def root_with_grad(f, root, args=()):
    """Attach implicit-function-theorem gradients to an already-found root.

    `root` is the converged T*, found however the caller likes -- it is
    detached here regardless. `args` are the parameters gradients should flow
    to. Returns T* plus one Newton correction, which is the same number to
    solver tolerance and carries the right derivative.
    """
    # Under no_grad there is nothing to attach and torch.autograd.grad would
    # raise, so the root itself is the answer. Evaluation runs this way.
    if not torch.is_grad_enabled():
        return root.detach()

    dfdt = _local_dfdt(f, root, args)

    # df/dT is the denominator of the implicit derivative, so a root where the
    # residual is locally flat sends the gradient to infinity. That is a real
    # singularity and not a numerical accident: at such a point the root's
    # position genuinely is not a differentiable function of the parameters.
    # It is floored rather than allowed through, because one infinite entry
    # poisons the whole backward pass and, through Adam's moments, every later
    # step. The sign is preserved so the correction still points the right way.
    eps = torch.finfo(dfdt.dtype).eps ** 0.5
    sign = torch.where(dfdt >= 0, torch.ones_like(dfdt), -torch.ones_like(dfdt))
    dfdt = torch.where(dfdt.abs() < eps, sign * eps, dfdt)

    # Recompute the residual with T fixed, so the only live path to the output
    # is through the parameters. This is what makes the derivative implicit
    # rather than a derivative of the bisection.
    t0 = root.detach()
    f0 = f(t0, *args)
    return t0 - f0 / dfdt.detach()


def solve(f, lo, hi, args=(), tol=1e-12, max_iter=200, method=None):
    """Solve f(T, *args) = 0 for T, with gradients flowing to `args`.

    `method` selects the bracketed solver: 'illinois' (the default) or
    'bisect'. Both keep the bracket and converge to the same root; Illinois
    gets there in about a third of the residual evaluations, which is the
    whole cost of a sequential run, and 'bisect' is kept so the two can be
    compared on a real case rather than argued about.
    """
    method = DEFAULT_METHOD if method is None else method
    if method == 'bisect':
        root = bisect(f, lo, hi, args=args, tol=tol, max_iter=max_iter)
    elif method == 'illinois':
        root = illinois(f, lo, hi, args=args)
    else:
        raise ValueError("method must be 'illinois' or 'bisect', not %r"
                         % (method,))
    # Both solvers promise a result inside the bracket, and
    # `root_with_grad`'s single Newton correction keeps it there AT A ROOT,
    # where the residual is ~0 and the correction with it. Where the bracket
    # held no root the residual is not small, and the correction throws the
    # answer clean out of the interval: a soil node bracketed on [-11, -5]
    # with no sign change came back at +9.3 C. The caller is supposed to be
    # checking the bracketing mask, but a number that far outside the
    # interval it asked about is a trap rather than a wrong answer, so the
    # correction is clamped back into the bracket. At a root the clamp never
    # binds and the gradient passes through it untouched.
    return torch.clamp(root_with_grad(f, root, args=args),
                       min=torch.minimum(lo, hi), max=torch.maximum(lo, hi))
