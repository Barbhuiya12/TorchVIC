"""The time step's host-free code paths, for capture in a CUDA graph.

A CUDA graph replays a fixed sequence of kernels on fixed buffers: no
operation inside it may wait for the device (`bool(t.any())`,
`t.nonzero()`, `.item()`) and no shape may depend on the data. The time
step has a handful of such places, each a shortcut that skips work nobody
needs (a loop that stops once every element has converged, a stage run
only on the tiles that need it). With `ACTIVE` set, each takes a path
that needs no answer from the device:

* a loop whose further iterations are exact no-ops for every element that
  has stopped runs its iterations unconditionally;
* a loop that may need more iterations than its graph-safe count runs that
  count and records, on the device, whether any element was still going;
* a stage run on a data-dependent subset runs on a fixed superset and its
  results are masked to the subset, as the subset's scatter would have
  left them.

Every such path gives the eager path's result bit for bit while the
recorded assumptions hold. `need` records a violated assumption in a
device flag; the caller reads the flag once in a while (once a day of
steps in basin.simulate) and, where it is set, runs those steps again
eagerly. With `ACTIVE` unset nothing here does anything.
"""
import os

import torch

from .diaglog import log

#: graph-safe paths on (set only by basin.simulate around a graph's steps)
ACTIVE = False
#: [1] int64 on the device: bit b set when assumption b was violated since
#: the last reset (bits are named by the callers of `need`)
FLAG = None
#: [1] int64 on the device, or None: set by `seen` (a note for the
#: caller's choice of graph, not a violated assumption)
SEEN = None
#: optional mask of the elements whose violations count (e.g. the canopy
#: tiles that actually run their stage this step); broadcast against `cond`
MASK = None
#: which branch a graph assumes where the eager step branches on the data
#: as a whole (e.g. 'canopy_any': some tile runs its canopy this step)
MODES = {}
#: the branch the last eager step took at each such place
LAST = {}

#: the assumptions, by bit
MASS_RELEASE, SNOW_ANY, SNOW_NONE, WIDEN, BRENT = range(5)


def need(cond, bit=0):
    """Record on the device that `cond` (a bool tensor, True where the
    graph-safe path is not the eager one) holds anywhere."""
    if FLAG is None:
        return
    if MASK is not None and cond.dim() > 0:
        cond = cond & MASK
    FLAG.bitwise_or_(cond.any().reshape(1).long() * (1 << bit))



def seen(cond):
    """Note on the device that `cond` (a 0-dim bool tensor) held."""
    if SEEN is None:
        return
    SEEN.bitwise_or_(cond.reshape(1).long())


#: the Segmenter capturing the current step's graph, or None
SEGMENTER = None


#: debugging aid (TORCHVIC_SEG_PROF=1): per replay item, device time and
#: loop iterations; summary() prints by item
#: Lane compaction (core.compact): the residual evaluations of a captured
#: loop's late iterations run on the lanes still iterating only. A loop
#: with a stop flag per lane (`live_of`) gets a second graph, which gathers
#: the (at most lanes // COMPACT_DIV) live lanes' inputs of the residual,
#: evaluates it on those and scatters the results back; the replay takes it
#: once the live count is at or under that capacity. 0 turns it off.
_CDIV = [None]


def compact_div():
    """The divisor of the lane count that is the compact capacity; 0 = off.
    TORCHVIC_COMPACT_DIV sets it; by default 16 on a card whose float64
    throughput is a small fraction of its float32 (the workstation and
    consumer ones: the surface residual is float64-bound there and the
    compact graph several times faster) and off on the data-centre ones
    (compute capability 7.0, 8.0, 9.0, 10.0), launch-bound, where the
    hoisted full iteration is as fast as the compact one and its build
    (compile, capture) would only lengthen the start-up."""
    if _CDIV[0] is None:
        v = os.environ.get('TORCHVIC_COMPACT_DIV')
        if v is not None:
            _CDIV[0] = int(v)
        else:
            try:
                cc = torch.cuda.get_device_capability()
                _CDIV[0] = 0 if tuple(cc) in ((7, 0), (8, 0), (9, 0),
                                              (10, 0)) else 16
            except Exception:
                _CDIV[0] = 16
    return _CDIV[0]
#: (idx, capacity, lanes) while a compact iteration is built or dry-run
COMPACT = None
#: compact residual evaluations made (capture-time count of the last loop)
COMPACT_USED = [0]


def compaction(idx, cap, lanes, mode='gather'):
    """The lanes `idx` (cap of them out of `lanes`) are evaluated by the
    compact graphs inside it. mode 'gather': the inputs gathered at each
    evaluation; 'enter': gathered into persistent buffers (and one evaluation
    made); 'iter': the buffers read, only the temperature gathered."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        global COMPACT
        old = COMPACT
        COMPACT = (idx, cap, lanes, mode)
        try:
            yield
        finally:
            COMPACT = old
    return cm()


#: the scope of the surface solve's hoisted invariants (core.compact): an
#: object made by hoist_scope, None outside one (nothing is kept then)
HOIST = [None]


def hoist_scope():
    import contextlib

    @contextlib.contextmanager
    def cm():
        old = HOIST[0]
        HOIST[0] = object()
        try:
            yield
        finally:
            HOIST[0] = old
    return cm()


def lane_live(live):
    """[lanes] bool: a lane is live if any element of its row is."""
    return live if live.dim() == 1 else live.reshape(live.shape[0],
                                                      -1).any(1)


def build_index(live, cap):
    """int64 [cap]: the positions of the first `cap` True elements of the
    1-D bool `live`, the remainder filled with the first one (that lane
    computed twice: the same inputs, so the same outputs, written to one
    place) and 0 where none is live. No host read."""
    n = live.numel()
    pos = torch.cumsum(live.to(torch.int32), 0) - 1
    slot = torch.where(live & (pos < cap), pos, torch.full_like(pos, cap))
    idx = torch.zeros(cap + 1, dtype=torch.int64, device=live.device)
    idx.scatter_(0, slot.to(torch.int64),
                 torch.arange(n, dtype=torch.int64, device=live.device))
    idx = idx[:cap]
    cnt = live.sum()
    return torch.where(torch.arange(cap, device=live.device) < cnt, idx,
                       idx[0])


#: a loop whose iteration takes less device time than this many
#: microseconds runs one iteration ahead of the host's read of its stop flag
#: (replay); a longer one is replayed only while the flag says so. A
#: negative value (TORCHVIC_LOOP_SPEC_US=-1) is the earlier scheme: always
#: ahead
SPEC_US = float(os.environ.get('TORCHVIC_LOOP_SPEC_US', '120'))

COMPACT_LOG = os.environ.get('TORCHVIC_COMPACT_LOG', '0') != '0'

SEG_TAGS = {}
SEG_PROF = [] if os.environ.get('TORCHVIC_SEG_PROF', '0') != '0' else None


def seg_summary():
    import collections
    tot = collections.defaultdict(lambda: [0, 0.0, 0])
    for sid, k, kind, n, a, b in SEG_PROF:
        b.synchronize()
        r = tot[(sid, k, kind)]
        r[0] += 1
        r[1] += a.elapsed_time(b)
        r[2] += n
    rows = sorted(tot.items(), key=lambda kv: -kv[1][1])
    for (sid, k, kind), (c, ms, n) in rows[:40]:
        log.debug('seg %x item %d %s calls %d total %.1f ms  iters %d  %s' % (sid & 0xffff, k, kind, c, ms, n, SEG_TAGS.get((sid, k), '')))
    log.debug('seg total device-window ms %.1f' % sum(r[1] for r in tot.values()))


#: debugging aid (TORCHVIC_CAPTURE_OPS=FILE): the aten ops the captured step
#: dispatches outside the compiled regions, by torchvic line, written at exit
CAPTURE_OPS = os.environ.get('TORCHVIC_CAPTURE_OPS')


def capture_ops():
    import contextlib
    if not CAPTURE_OPS:
        return contextlib.nullcontext()
    import atexit
    import collections
    import traceback
    from torch.utils._python_dispatch import TorchDispatchMode
    if not hasattr(capture_ops, 'count'):
        capture_ops.count = collections.Counter()

        def dump():
            by = collections.Counter()
            for (k_, o_), n_ in capture_ops.count.items():
                by[k_] += n_
            with open(CAPTURE_OPS, 'w') as fh:
                for k_, n_ in by.most_common():
                    ops = collections.Counter(
                        {o_: n2 for (kk, o_), n2 in capture_ops.count.items()
                         if kk == k_})
                    fh.write('%d\t%s\t%s\n' % (n_, k_, dict(ops.most_common(4))))
        atexit.register(dump)

    class Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            st = traceback.extract_stack()
            if any(f.filename.endswith('core/fusion.py') for f in st):
                return func(*args, **(kwargs or {}))     # inside a region
            fr = [f for f in st if '/torchvic/' in f.filename
                  and not f.filename.endswith('core/graphsafe.py')]
            key = ' <- '.join('%s:%d' % (f.filename.split('/torchvic/')[-1],
                                         f.lineno) for f in fr[-1:-7:-1])
            capture_ops.count[(key, str(func).replace('aten.', ''))] += 1
            return func(*args, **(kwargs or {}))
    return Mode()


_GEN_STATE = [False]


def ensure_generator_graph_state():
    """Make the CUDA generator's graph-capture state outside inference mode.
    The first capture in a process allocates it (the philox seed and offset
    tensors a captured kernel reads), and allocated inside torch.inference_mode
    they are inference tensors: every later capture outside inference mode
    (a test, a user's torch.cuda.graph) then fails in capture_begin with
    "Inplace update to inference tensor outside InferenceMode". basin.simulate
    captures inside inference mode, so the state is made here, by a capture of
    one trivial kernel, before the first one."""
    if _GEN_STATE[0] or not torch.cuda.is_available():
        return
    _GEN_STATE[0] = True
    with torch.inference_mode(False):
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            x = torch.zeros(1, device='cuda')
            g.capture_begin()
            x.add_(1.0)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()


def _install_capture_hook():
    """Every CUDAGraph capture in the process (the connectors', a test's, a
    user's) first makes the generator's graph state outside inference mode."""
    cg = torch.cuda.graphs.CUDAGraph
    if getattr(cg.capture_begin, '_tv_hooked', False):
        return
    orig = cg.capture_begin

    def capture_begin(self, *a, **k):
        ensure_generator_graph_state()
        # begun outside inference mode wherever it is called from: the
        # generator state a capture (re)allocates is then a normal tensor,
        # not an inference tensor that fails the next capture outside it
        with torch.inference_mode(False):
            return orig(self, *a, **k)
    capture_begin._tv_hooked = True
    cg.capture_begin = capture_begin


_install_capture_hook()


class Segmenter(object):
    """A step captured as a sequence of CUDA graphs with host-decided loops
    between them.

    A loop whose trip count depends on the data -- an iterative solve that
    stops when every element has converged -- cannot live inside one
    graph without running its largest possible count every step. Here the
    capture is cut where such a loop begins: the loop's state is copied to
    fixed buffers, one iteration (state to state) is captured as a graph
    of its own, and the capture of the rest of the step resumes from the
    buffers. On replay each loop graph is replayed until its stop flag --
    computed on the device by the graph before it and by every iteration
    -- reads true: one host read per iteration, as the eager loop makes,
    and every iteration the eager loop's own, so the result is the same.
    """

    def __init__(self, pool):
        ensure_generator_graph_state()
        self.pool = pool
        self.items = []
        self.cur = None
        self._mode = {}         # loop item -> 'spec' | 'sync' (see replay)
        self._tm = {}           # loop item -> [iterations timed, ms]
        self._tc = {}           # the same for its compact iterations
        self._cm = {}           # loop item -> 'on' | 'off' (compact graph)

    def begin(self):
        self.cur = torch.cuda.CUDAGraph()
        # thread-local: a forcing reader on another thread (allocating and
        # copying on its own stream) must not invalidate the capture
        self.cur.capture_begin(pool=self.pool,
                               capture_error_mode='thread_local')

    def _cut(self):
        self.cur.capture_end()
        self.items.append(('graph', self.cur))
        self.cur = None

    def loop(self, z, step, stopped, cap, step_into=None, live_of=None,
             compact_into=None):
        """Inside a capture: `z` (a list of tensors) is the loop's state,
        `step(z)` one iteration, `stopped(z)` a [1] bool. Returns the state
        buffers the capture continues from. `step_into(zs, stop)`, if
        given, is one iteration and the stop test written into the loop's
        buffers in place (no copy of the state per iteration).
        `live_of(zs)`, if given, is the bool mask of the lanes still
        iterating: the loop then also gets a compact iteration graph
        (`compact_into(zs, stop)`, or `step_into`, run under
        graphsafe.compaction), see compact_div."""
        zs = [x.clone() for x in z]
        stop = stopped(zs).clone()
        cnt, lanes, ncap = None, 0, 0
        if live_of is not None and compact_div() > 0 and step_into is not None:
            lanes = int(lane_live(live_of(zs)).numel())
            ncap = lanes // compact_div()
            if ncap >= 64:
                cnt = lane_live(live_of(zs)).sum(
                    dtype=torch.int32).reshape(1).clone()
        self._cut()
        self.begin()
        if step_into is not None:
            step_into(zs, stop)
        else:
            z2 = step(zs)
            for x, y in zip(zs, z2):
                if x is not y:
                    x.copy_(y)
            stop.copy_(stopped(zs))
        if cnt is not None:
            cnt.copy_(lane_live(live_of(zs)).sum(
                dtype=torch.int32).reshape(1))
        g = self.cur
        g.capture_end()
        self.cur = None
        gc = ge = None
        if cnt is not None:
            COMPACT_USED[0] = 0
            idx_buf = torch.zeros(ncap, dtype=torch.int64,
                                  device=stop.device)
            # enter: the lanes still going, and their inputs into buffers
            # (one evaluation on copies of the state does it)
            self.begin()
            idx_buf.copy_(build_index(lane_live(live_of(zs)), ncap))
            zd = [x.clone() for x in zs]
            with compaction(idx_buf, ncap, lanes, 'enter'):
                (compact_into or step_into)(zd, stop.clone())
            ge = self.cur
            ge.capture_end()
            self.cur = None
            # one compact iteration: the buffers read, the state in place
            self.begin()
            with compaction(idx_buf, ncap, lanes, 'iter'):
                (compact_into or step_into)(zs, stop)
            cnt.copy_(lane_live(live_of(zs)).sum(
                dtype=torch.int32).reshape(1))
            gc = self.cur
            gc.capture_end()
            self.cur = None
            if not COMPACT_USED[0]:
                gc = ge = None          # no residual could be compacted
        pin = torch.zeros(2, dtype=torch.bool, device='cpu',
                          pin_memory=True)
        pinc = torch.zeros(1, dtype=torch.int32, device='cpu',
                           pin_memory=True)
        evs = [torch.cuda.Event(), torch.cuda.Event()]
        if SEG_PROF is not None:
            import traceback
            fr = [f for f in traceback.extract_stack()
                  if '/torchvic/' in f.filename
                  and not f.filename.endswith('core/graphsafe.py')]
            SEG_TAGS[(id(self), len(self.items))] = ' <- '.join(
                '%s:%d' % (f.filename.split('/torchvic/')[-1], f.lineno)
                for f in fr[-1:-4:-1]) + (' compact %d' % ncap if gc else '')
        self.items.append(('loop', g, stop, int(cap), pin, evs, gc, cnt,
                           ncap, pinc, ge))
        self.begin()
        return zs

    def finish(self):
        self._cut()

    def replay(self):
        """Replay the step. A loop's next iteration is launched before the
        host reads whether the previous one stopped the loop (one
        iteration ahead, the flag copied to pinned memory behind each),
        so the device never waits for the host between iterations; an
        iteration launched after the loop stopped is one more of the
        iterations the loop graph is built of, an exact no-op for every
        element that counts (see `loop`), and is not counted."""
        n_iter = 0
        prof = SEG_PROF
        if prof is not None:
            e0 = torch.cuda.Event(enable_timing=True)
            e0.record()
            ev = [e0]
        for k_, it in enumerate(self.items):
            if it[0] == 'graph':
                it[1].replay()
                if prof is not None:
                    e_ = torch.cuda.Event(enable_timing=True)
                    e_.record()
                    ev.append(e_)
                    prof.append((id(self), k_, 'g', 0, ev[-2], e_))
                continue
            _, g, stop, cap, pin, evs, gc, cnt, ncap, pinc, ge = it
            mode = self._mode.get(k_)
            n = 0
            if mode == 'spec' or SPEC_US < 0:
                pin[0].copy_(stop[0], non_blocking=True)
                evs[0].record()
                while True:
                    ahead = n < cap
                    if ahead:
                        g.replay()
                        j = (n + 1) % 2
                        pin[j].copy_(stop[0], non_blocking=True)
                        evs[j].record()
                    evs[n % 2].synchronize()
                    if not ahead or bool(pin[n % 2]):
                        break
                    n += 1
            else:
                # one iteration at a time, the flag read first: no
                # iteration runs that the loop did not need. The device
                # idles for the flag's round trip (~30 us) between
                # iterations, which an iteration of a few hundred
                # microseconds or more (the residual of a surface solve) far
                # exceeds; a cheaper one runs ahead (above)
                meas = mode is None
                entered = False
                ep = None                   # a timed compact episode
                kc = 0                      # its compact iterations
                while n < cap:
                    pin[0].copy_(stop[0], non_blocking=True)
                    if gc is not None:
                        pinc[0].copy_(cnt[0], non_blocking=True)
                    evs[0].record()
                    evs[0].synchronize()
                    if bool(pin[0]):
                        break
                    gsel = g
                    if gc is not None and not meas and int(pinc[0]) <= ncap:
                        # few enough lanes for the compact graph: used where
                        # it is the faster, the one entry (the inputs of the
                        # residual gathered) included (a launch-bound GPU
                        # runs the full iteration, its invariants hoisted,
                        # as fast as the compact one; a float64-bound one
                        # runs the compact one several times as fast)
                        cm = self._cm.get(k_)
                        if cm != 'off':
                            gsel = gc
                            if not entered:
                                if cm is None:
                                    ep = torch.cuda.Event(enable_timing=True)
                                    ep.record()
                                ge.replay()
                                entered = True
                            kc += 1
                    if meas:
                        t0, t1 = (torch.cuda.Event(enable_timing=True),
                                  torch.cuda.Event(enable_timing=True))
                        t0.record()
                    gsel.replay()
                    if meas:
                        t1.record()
                        t1.synchronize()
                        tm = self._tm.setdefault(k_, [0, 0.0])
                        tm[0] += 1
                        tm[1] += t0.elapsed_time(t1)
                    n += 1
                if ep is not None:
                    e1 = torch.cuda.Event(enable_timing=True)
                    e1.record()
                    e1.synchronize()
                    tc = self._tc.setdefault(k_, [0, 0.0, 0])
                    tc[0] += 1
                    tc[1] += ep.elapsed_time(e1)
                    tc[2] += kc
                    if tc[0] >= 3 and k_ in self._tm:
                        full = self._tm[k_][1] / self._tm[k_][0]
                        self._cm[k_] = ('on' if tc[1] < 0.9 * full * tc[2]
                                        else 'off')
                        if COMPACT_LOG:
                            import sys
                            log.debug('compact loop %d: full iteration %.3f ms, '
                                  'compact episode %.3f ms for %.1f '
                                  'iterations -> %s' % (
                                      k_, full, tc[1] / tc[0], tc[2] / tc[0],
                                      self._cm[k_]))
                if meas and k_ in self._tm and self._tm[k_][0] >= 6:
                    self._mode[k_] = ('spec' if 1000.0 * self._tm[k_][1]
                                      / self._tm[k_][0] < SPEC_US else 'sync')
            n_iter += n
            if prof is not None:
                e_ = torch.cuda.Event(enable_timing=True)
                e_.record()
                ev.append(e_)
                prof.append((id(self), k_, 'l', n, ev[-2], e_))
        return n_iter


def widen(f, lo, hi, flo, fhi, args, tstep, maxtries, compact=False):
    """root_brent.c's bracket widening (both sides by `tstep` while f(lo)
    and f(hi) share a sign, at most `maxtries` times) on the graph-safe
    path: inside a captured step its tries are a Segmenter loop, replayed
    only while an element that counts (MASK) is unbracketed; on the eager
    graph-safe path the loop stops when none is. Every try leaves a
    bracketed element as it is, so the result is the eager loop's.
    Returns lo, hi, flo, fhi."""
    def step(z):
        lo_, hi_, flo_, fhi_ = z
        bad = (flo_ * fhi_) >= 0.0
        lo_ = torch.where(bad, lo_ - tstep, lo_)
        hi_ = torch.where(bad, hi_ + tstep, hi_)
        flo_ = torch.where(bad, f(lo_, *args), flo_)
        fhi_ = torch.where(bad, f(hi_, *args), fhi_)
        return [lo_, hi_, flo_, fhi_]

    def stopped(z):
        bad = (z[2] * z[3]) >= 0.0
        if MASK is not None:
            bad = bad & MASK
        return (~bad.any()).reshape(1)

    z = [lo, hi, flo, fhi]
    into = (lambda zs, stop: _widen_into(zs, stop, f, args, tstep, MASK))
    live_of = cinto = None
    if compact:
        # the unbracketed lanes; a compact try is the same one with the two
        # residual evaluations (lanes of their own) between two regions
        live_of = lambda zs: _widen_live(zs[2], zs[3], MASK)    # noqa: E731
        cinto = (lambda zs, stop: _widen_compact(zs, stop, f, args, tstep,
                                                 MASK))
    if SEGMENTER is not None:
        z = SEGMENTER.loop(z, step, stopped, maxtries, step_into=into,
                           live_of=live_of, compact_into=cinto)
    else:
        z = run_loop(z, stopped, maxtries, into, live_of=live_of,
                     compact_into=cinto)
    return tuple(z)


def run_loop(z, stopped, cap, step_into, live_of=None, compact_into=None):
    """A Segmenter loop run eagerly on the graph-safe path, as its capture
    is built (the state cloned into the loop's buffers, `step_into`
    writing each iteration into them), so the regions the capture records
    are compiled here, before it; the eager loop's iterations and
    result."""
    zs = [x.clone() for x in z]
    stop = stopped(zs).clone()
    n = 0
    while n < cap and not bool(stop):
        step_into(zs, stop)
        n += 1
    if n == 0 and cap > 0:
        # not one iteration needed here; the capture records one all the
        # same, so it is made once on copies (compiling what it calls)
        step_into([x.clone() for x in zs], stop.clone())
    if live_of is not None and compact_div() > 0:
        # the compact iteration the capture will record, made once on the
        # loop's first state: its residuals are built (and compared with
        # the full evaluation) here, before the capture
        zd = [x.clone() for x in z]
        lv = lane_live(live_of(zd))
        ncap = int(lv.numel()) // compact_div()
        if ncap >= 64:
            idx = build_index(lv, ncap)
            with compaction(idx, ncap, int(lv.numel())):
                (compact_into or step_into)(zd, stop.clone())
    return zs


def _widen_into_(zs, stop, f, args, tstep, mask):
    """One widening try and the stop test, written into the loop's buffers
    (a captured loop's iteration as one fused region): the values of
    widen's `step` and `stopped`."""
    lo_, hi_, flo_, fhi_ = zs
    bad = (flo_ * fhi_) >= 0.0
    lo2 = torch.where(bad, lo_ - tstep, lo_)
    hi2 = torch.where(bad, hi_ + tstep, hi_)
    flo2 = torch.where(bad, f(lo2, *args), flo_)
    fhi2 = torch.where(bad, f(hi2, *args), fhi_)
    for d, v in zip(zs, (lo2, hi2, flo2, fhi2)):
        d.copy_(v)
    bad2 = (flo2 * fhi2) >= 0.0
    if mask is not None:
        bad2 = bad2 & mask
    stop.copy_((~bad2.any()).reshape(1))


def _widen_prep_(zs, tstep):
    lo_, hi_, flo_, fhi_ = zs
    bad = (flo_ * fhi_) >= 0.0
    return (bad, torch.where(bad, lo_ - tstep, lo_),
            torch.where(bad, hi_ + tstep, hi_))


def _widen_fin_(zs, stop, bad, lo2, hi2, f_lo, f_hi, mask):
    flo2 = torch.where(bad, f_lo, zs[2])
    fhi2 = torch.where(bad, f_hi, zs[3])
    for d, v in zip(zs, (lo2, hi2, flo2, fhi2)):
        d.copy_(v)
    bad2 = (flo2 * fhi2) >= 0.0
    if mask is not None:
        bad2 = bad2 & mask
    stop.copy_((~bad2.any()).reshape(1))


def _widen_live_(flo, fhi, mask):
    bad = (flo * fhi) >= 0.0
    return bad if mask is None else (bad & mask)


_widen_regions = {}


def _widen_region(name, fn):
    from . import fusion
    if name not in _widen_regions:
        _widen_regions[name] = fusion.region(fn)
    return _widen_regions[name]


def _widen_live(*a):
    return _widen_region('live', _widen_live_)(*a)


def _widen_compact(zs, stop, f, args, tstep, mask):
    """One widening try (widen's `step`, `stopped`) with its two residual
    evaluations outside the regions, which can then run on the live lanes
    only."""
    bad, lo2, hi2 = _widen_region('prep', _widen_prep_)(zs, tstep)
    f_lo = f(lo2, *args)
    f_hi = f(hi2, *args)
    _widen_region('fin', _widen_fin_)(zs, stop, bad, lo2, hi2, f_lo, f_hi,
                                      mask)


def _widen_into(*a):
    from . import fusion
    global _widen_into_region
    if _widen_into_region is None:
        _widen_into_region = fusion.region(_widen_into_)
    return _widen_into_region(*a)


_widen_into_region = None


# --- a host-sync detector, for checking the captured step on any device ---

#: the operations that wait for the device or copy from the host, which a
#: CUDA graph capture refuses ("operation not permitted when stream is
#: capturing") -- by name, as torch's function mode reports them
_SYNC_OPS = frozenset((
    '__bool__', '__float__', '__int__', '__index__', '__complex__',
    'item', 'tolist', 'numpy', 'cpu', 'nonzero', 'argwhere', 'unique',
    'unique_consecutive', 'masked_select', 'equal', 'allclose',
    'is_nonzero', 'tensor', 'nonzero_static_', 'bincount'))


def _bool_index(idx):
    if isinstance(idx, torch.Tensor):
        return idx.dtype == torch.bool
    if isinstance(idx, (tuple, list)):
        return any(_bool_index(i) for i in idx)
    return False


class SyncCheck(torch.overrides.TorchFunctionMode):
    """Inside it, every operation that would wait for the device (or copy
    from the host) during a CUDA graph capture is recorded with the
    torchvic line that made it: `found` maps 'file:line' to the operation.
    The operations still run, so the step goes on and every such place in
    it is found in one pass, on a CPU as on a GPU."""

    def __init__(self):
        super(SyncCheck, self).__init__()
        self.found = {}
        self.off = 0

    def _where(self):
        import traceback
        other = None
        for fr in reversed(traceback.extract_stack()[:-2]):
            fn = fr.filename.replace('\\', '/')
            if '/torchvic/' in fn and not fn.endswith('core/graphsafe.py'):
                return '%s:%d' % (fn.split('/torchvic/', 1)[1], fr.lineno)
            if other is None and '/torch/' not in fn \
                    and not fn.endswith('core/graphsafe.py'):
                other = '%s:%d' % (os.path.basename(fn), fr.lineno)
        return other or '?'

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not self.off:
            name = getattr(func, '__name__', '')
            bad = name in _SYNC_OPS
            if name == 'as_tensor':
                # from host data: a copy (a tensor is passed through)
                bad = not isinstance(args[0] if args else kwargs.get('data'),
                                     torch.Tensor)
            elif name == 'where' and len(args) + len(kwargs) == 1:
                bad = True
            elif name == 'repeat_interleave' and 'output_size' not in kwargs:
                bad = len(args) > 1 and isinstance(args[1], torch.Tensor)
            elif name in ('__getitem__', '__setitem__', 'index_put_',
                          'index_put'):
                bad = len(args) > 1 and _bool_index(args[1])
            if bad:
                self.found.setdefault(self._where(), name)
        return func(*args, **kwargs)


class CheckSegmenter(object):
    """The Segmenter's interface for a step run under SyncCheck instead of
    captured: its loops run as the replay would (host-decided, outside the
    check), every other operation is checked."""

    def __init__(self, check):
        self.check = check

    def loop(self, z, step, stopped, cap, step_into=None, live_of=None,
             compact_into=None):
        self.check.off += 1
        try:
            n = 0
            while n < cap and not bool(stopped(z)):
                z2 = step(z)
                z = [y for y in z2]
                n += 1
        finally:
            self.check.off -= 1
        return z
