"""Lane compaction of a compiled region (the surface residual in the late
iterations of a root find).

A region compiled by core.fusion evaluates every lane. In the late
iterations of Brent's search or of the bracket widening a few lanes in a
thousand are still going, and the residual (the bulk of the step on a GPU
with a weak float64 unit) is evaluated for all of them. Here the backend
that compiles a region marked `compact` returns an object that, inside
graphsafe.compaction(idx, capacity, lanes), gathers every input whose
leading dimension is the lane count at `idx`, runs a second compilation of
the same graph on the capacity-sized inputs, and scatters the outputs with
a lane dimension back to full size (zeros where no lane was evaluated).

The arithmetic of a lane does not depend on the other lanes (the kernels
are element-wise or reduce along trailing dimensions), so the values are
the full evaluation's bit for bit; the first compact call of a region,
made in the eager step before a capture, runs both and compares them, and
a region that differs (or has an operation that mixes lanes) stays on the
full evaluation.
"""
import copy
import os
import sys

import torch

from .diaglog import log

from . import graphsafe

CALLS = [0]
DEBUG2 = os.environ.get('TORCHVIC_COMPACT_DEBUG') == '2'
WHAT = ['?']
DEBUG = os.environ.get('TORCHVIC_COMPACT_DEBUG', '0') == '1'

#: a graph with one of these operations mixes lanes (or writes in place)
_MIXING = ('index_select', 'gather', 'scatter', 'index_put', 'cumsum',
           'cumprod', 'sort', 'topk', 'unique', 'nonzero', 'bincount',
           'tsum', 'segsum', 'masked_select', 'take', 'roll', 'flip',
           'repeat_interleave', 'copy_', 'index_copy', 'masked_scatter')


def _mixes_lanes(gm):
    for n in gm.graph.nodes:
        if n.op not in ('call_function', 'call_method'):
            continue
        name = str(getattr(n.target, '__name__', n.target))
        if any(m in name for m in _MIXING):
            return name
        if (n.op == 'call_method' and name.endswith('_')
                and not name.endswith('__')):
            return name                      # an in-place method
    return None


def _compile(inner, gm, inputs):
    """inner(gm, inputs) as it is inside dynamo: in a tracing context with
    a shape environment, without which inductor's FX graph cache is bypassed
    (every graph built outside a trace would compile again in each process)."""
    from torch._dynamo.utils import get_metrics_context
    from torch._guards import (CompileContext, CompileId, TracingContext,
                               compile_context, tracing)
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch._dynamo.utils import counters
    c0 = dict(counters['aot_autograd'])
    t0 = __import__('time').time()
    from torch._dynamo.utils import dynamo_timed
    with compile_context(CompileContext(CompileId(0, 0))), \
            get_metrics_context(), \
            dynamo_timed('OutputGraph.call_user_compiler',
                         phase_name='backend_compile',
                         log_pt2_compile_event=True,
                         dynamo_compile_column_us=(
                             'aot_autograd_cumulative_compile_time_us')):
        out = _compile_in(inner, gm, inputs, tracing, TracingContext,
                          FakeTensorMode, ShapeEnv)
    if DEBUG2:
        c1 = counters['aot_autograd']
        log.debug('compact compile %s: %.2f s, %s' % (
            WHAT[0], __import__('time').time() - t0,
            {k: c1[k] - c0.get(k, 0) for k in c1
             if c1[k] != c0.get(k, 0)}))
    return out


def _compile_in(inner, gm, inputs, tracing, TracingContext, FakeTensorMode,
                ShapeEnv):
    if DEBUG:
        import torch._functorch.config as fcfg
        old = fcfg.strict_autograd_cache
        fcfg.strict_autograd_cache = True
        try:
            with tracing(TracingContext(FakeTensorMode(
                    shape_env=ShapeEnv()))):
                return inner(copy.deepcopy(gm), inputs)
        except Exception as exc:
            log.debug('compact: autograd cache refused a graph: %s: %s'
                  % (type(exc).__name__, str(exc)[:300]))
        finally:
            fcfg.strict_autograd_cache = old
    with tracing(TracingContext(FakeTensorMode(shape_env=ShapeEnv()))):
        return inner(gm, inputs)


def _resized(gm, lanes, cap):
    """A copy of the graph with the lane count written into it (an
    `expand` to the lane count, a `view` of it, a `full` of it: dynamo
    specialised the shapes of the trace) replaced by the capacity."""
    from torch.fx.node import map_aggregate
    g2 = copy.deepcopy(gm)

    def sub(a):
        return cap if (type(a) is int and a == lanes) else a
    for n in g2.graph.nodes:
        if n.op in ('call_function', 'call_method', 'call_module'):
            n.args = map_aggregate(n.args, sub)
            n.kwargs = map_aggregate(n.kwargs, sub)
    g2.recompile()
    return g2


def _bit_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        it = {torch.float64: torch.int64, torch.float32: torch.int32,
              torch.float16: torch.int16, torch.bfloat16: torch.int16}[
                  a.dtype]
        a, b = a.contiguous().view(it), b.contiguous().view(it)
    return bool(torch.equal(a, b))


class _Lane(object):
    """A compiled graph that can run on a subset of its lanes (module
    docstring). Called as the compiled graph is, with its tensors."""

    def __init__(self, gm, full, inner, tag, ts_idx=None):
        self.ts_idx = ts_idx         # the argument that changes between calls
        self.pbuf = {}               # persistent gathered inputs (enter/iter)
        self.gm = gm                 # an untouched copy, for the compact build
        self.full = full
        self.inner = inner
        self.tag = tag
        self.compact = None
        # a graph that reads other lanes is not compacted
        self.off = _mixes_lanes(gm) is not None
        self.lane_out = None

    def _gather(self, args, idx, lanes):
        return [a.index_select(0, idx)
                if (torch.is_tensor(a) and a.dim() >= 1
                    and a.shape[0] == lanes) else a for a in args]

    def __call__(self, *args):
        cx = graphsafe.COMPACT
        if cx is None or self.off:
            return self.full(*args)
        idx, cap, lanes, mode = cx
        if not any(torch.is_tensor(a) and a.dim() >= 1
                   and a.shape[0] == lanes for a in args):
            return self.full(*args)
        capturing = torch.cuda.is_current_stream_capturing()
        if self.compact is None:
            if capturing:
                return self.full(*args)       # not built before the capture
            return self._build(args, idx, cap, lanes)
        return self._run(args, idx, cap, lanes, mode)

    def _build(self, args, idx, cap, lanes):
        ref = self.full(*args)
        ok = False
        try:
            with torch.no_grad():
                g_in = self._gather(args, idx, lanes)
                WHAT[0] = self.tag + '/compact'
                self.compact = _compile(self.inner,
                                        _resized(self.gm, lanes, cap), g_in)
                self.lane_out = [torch.is_tensor(r) and r.dim() >= 1
                                 and r.shape[0] == lanes for r in ref]
                got = self.compact(*g_in)
                ok = len(got) == len(ref)
                for i, (r, o) in enumerate(zip(ref, got)):
                    if not ok:
                        break
                    if self.lane_out[i]:
                        ok = (torch.is_tensor(o) and o.shape[0] == cap
                              and _bit_equal(r.index_select(0, idx), o))
                    elif torch.is_tensor(r):
                        ok = torch.is_tensor(o) and _bit_equal(r, o)
                    else:
                        ok = bool(r == o)
        except Exception as exc:
            ok = False
            if DEBUG:
                import sys
                import traceback
                log.debug('compact %s: build failed: %s' % (self.tag, exc))
                log.debug(traceback.format_exc())
        if not ok:
            if DEBUG:
                import sys
                log.debug('compact %s: off (no compact build, or it differs)'
                      % self.tag)
            self.off = True
            self.compact = None
            return ref
        if DEBUG:
            import sys
            log.debug('compact %s: on, %d of %d lanes, %d outputs'
                  % (self.tag, cap, lanes, len(got)))
        return self._scatter(got, idx, lanes)

    def _run(self, args, idx, cap, lanes, mode):
        if mode == 'gather' or self.ts_idx is None:
            g_in = self._gather(args, idx, lanes)
        else:
            g_in = []
            for i, a in enumerate(args):
                lane = (torch.is_tensor(a) and a.dim() >= 1
                        and a.shape[0] == lanes)
                if not lane:
                    g_in.append(a)
                elif i == self.ts_idx:
                    g_in.append(a.index_select(0, idx))
                elif mode == 'enter':
                    buf = self.pbuf.get(i)
                    if (buf is None or buf.dtype != a.dtype
                            or buf.shape[1:] != a.shape[1:]
                            or buf.shape[0] != cap):
                        buf = torch.empty((cap,) + tuple(a.shape[1:]),
                                          dtype=a.dtype, device=a.device)
                        self.pbuf[i] = buf
                    torch.index_select(a, 0, idx, out=buf)
                    g_in.append(buf)
                else:                                   # 'iter'
                    buf = self.pbuf.get(i)
                    g_in.append(a.index_select(0, idx) if buf is None
                                else buf)
        got = self.compact(*g_in)
        return self._scatter(got, idx, lanes)

    def _scatter(self, got, idx, lanes):
        graphsafe.COMPACT_USED[0] += 1
        out = []
        for i, o in enumerate(got):
            if self.lane_out[i]:
                full = torch.zeros((lanes,) + tuple(o.shape[1:]),
                                   dtype=o.dtype, device=o.device)
                full.index_copy_(0, idx, o)
                out.append(full)
            else:
                out.append(o)
        return type(got)(out) if isinstance(got, (tuple, list)) else out


#: hoisting of what a graph computes from its inputs but not from `ts`
#: (TORCHVIC_HOIST=0 turns it off)
HOIST = os.environ.get('TORCHVIC_HOIST', '1') != '0'

#: compiled invariant graphs by (structure, input layouts)
_INV_COMPILED = {}

#: the invariants of the current hoisting scope, shared by the graphs with the
#: same invariant part: {'scope', 'inv': {(signature, ids): outputs}, 'keep'}
_SHARED = {'scope': None, 'inv': {}, 'keep': []}

_INPLACE = ('setitem', 'copy_', 'index_put_', 'masked_fill_', 'add_', 'mul_',
            'sub_', 'div_', 'fill_', 'zero_', 'clamp_', 'scatter_',
            'index_copy_', 'masked_scatter_', 'set_', 'resize_')


def _mutates(gm):
    for n in gm.graph.nodes:
        if n.op in ('call_function', 'call_method'):
            name = str(getattr(n.target, '__name__', n.target))
            if name in _INPLACE or (n.op == 'call_method'
                                    and name.endswith('_')
                                    and not name.endswith('__')):
                return name
    return None


def _graph_sig(g):
    """A key of the graph's structure, its placeholders by position; None
    when it reads a constant (get_attr), which another graph may not share."""
    from torch.fx.node import Node, map_aggregate
    idx = {}
    rows = []
    for n in g.nodes:
        if n.op == 'get_attr':
            return None
        idx[n] = len(idx)

        def d(a):
            return ('N', idx[a]) if isinstance(a, Node) else a
        if n.op == 'placeholder':
            rows.append('ph')
        else:
            rows.append((n.op, str(n.target),
                         repr(map_aggregate(n.args, d)),
                         repr(map_aggregate(dict(n.kwargs), d))))
    return hash(tuple(rows))


def _partition(gm):
    """Split a graph by its dependence on the placeholder `l_ts_` (the
    surface temperature being solved for) into the invariant graph (inputs:
    the other placeholders it reads; outputs: the values the rest needs) and
    the dependent one (inputs: `ts`, those values, the placeholders it reads
    itself; outputs: the original ones). None if not possible."""
    import torch.fx as fx
    nodes = list(gm.graph.nodes)
    ts = [n for n in nodes if n.op == 'placeholder'
          and n.name.startswith('l_ts_')]
    if len(ts) != 1 or nodes[-1].op != 'output':
        return None
    ts = ts[0]
    out = nodes[-1]
    dep = {ts}
    for n in nodes:
        if n.op in ('placeholder', 'output') or n in dep:
            continue
        if any(i in dep for i in n.all_input_nodes):
            dep.add(n)
    front, ph_d = [], []
    seen = set()

    def need(x):
        if x in dep or x in seen:
            return
        seen.add(x)
        (ph_d if x.op == 'placeholder' else front).append(x)
    for n in nodes:
        if n in dep and n is not ts:
            for i in n.all_input_nodes:
                need(i)
    for i in out.all_input_nodes:
        need(i)
    for f in front:
        ev = f.meta.get('example_value')
        if not torch.is_tensor(ev):
            return None
    # ancestors of the frontier among the invariant nodes
    anc = set()
    stack = list(front)
    while stack:
        n = stack.pop()
        if n in anc:
            continue
        anc.add(n)
        stack.extend(n.all_input_nodes)
    ph_i = [n for n in nodes if n.op == 'placeholder' and n in anc]
    # the invariant graph
    gi = fx.Graph()
    env = {}
    for n in nodes:
        if n not in anc:
            continue
        if n.op == 'placeholder':
            env[n] = gi.placeholder(n.name)
            env[n].meta = dict(n.meta)      # the cache checks the values
        else:
            env[n] = gi.node_copy(n, lambda x: env[x])
    gi.output(tuple(env[f] for f in front))
    inv = fx.GraphModule(gm, gi)
    # the dependent graph
    gd = fx.Graph()
    envd = {ts: gd.placeholder(ts.name)}
    envd[ts].meta = dict(ts.meta)
    for i, f in enumerate(front):
        envd[f] = gd.placeholder('inv%d' % i)
        envd[f].meta = dict(f.meta)
    for p in ph_d:
        envd[p] = gd.placeholder(p.name)
        envd[p].meta = dict(p.meta)
    for n in nodes:
        if n in dep and n is not ts:
            envd[n] = gd.node_copy(n, lambda x: envd[x])
    gd.output(fx.node.map_arg(out.args[0], lambda x: envd[x]))
    depgm = fx.GraphModule(gm, gd)
    pos = {n: i for i, n in enumerate(
        [n for n in nodes if n.op == 'placeholder'])}
    return dict(inv=inv, dep=depgm, ts_pos=pos[ts],
                inv_pos=[pos[p] for p in ph_i],
                pd_pos=[pos[p] for p in ph_d], n_front=len(front),
                n_dep=len(dep), n_nodes=len(nodes), sig=_graph_sig(gi))


class _Residual(object):
    """A compiled residual graph: its surface-temperature-invariant part
    evaluated once per hoisting scope (graphsafe.hoist_scope: one surface
    solve) and shared by every evaluation in it, the dependent part per
    evaluation, on a subset of the lanes under graphsafe.compaction. The
    first call (in the eager step before a capture) builds both and
    compares them with the original graph; a mismatch leaves the original."""

    def __init__(self, gm, full, inner, tag):
        self.gm = gm
        self.full = full
        self.inner = inner
        self.tag = tag
        self.plain = _Lane(copy.deepcopy(gm), full, inner, tag)
        self.state = 'new' if HOIST else 'off'

    def __call__(self, *args):
        if self.state == 'off':
            return self.plain(*args)
        if self.state == 'new':
            if torch.cuda.is_current_stream_capturing():
                return self.plain(*args)       # not built before the capture
            self._build(args)
            if self.state == 'off':
                return self.plain(*args)
        inv = self._inv(args)
        dargs = ([args[self.ts_pos]] + list(inv)
                 + [args[i] for i in self.pd_pos])
        return self.lane(*dargs)

    def _inv(self, args):
        iargs = [args[i] for i in self.inv_pos]
        scope = graphsafe.HOIST[0]
        if scope is None:
            return self.inv_c(*iargs)
        S = _SHARED
        if S['scope'] is not scope:
            S['scope'], S['inv'], S['keep'] = scope, {}, []
        key = (self.sig, tuple(id(a) for a in iargs)) \
            if self.sig is not None else (id(self), tuple(id(a)
                                                          for a in iargs))
        hit = S['inv'].get(key)
        if hit is None:
            hit = self.inv_c(*iargs)
            S['inv'][key] = hit
            S['keep'].append(iargs)           # their ids stay theirs
        return hit

    def _build(self, args):
        ok = False
        try:
            bad = _mutates(self.gm)
            part = None if bad else _partition(copy.deepcopy(self.gm))
            if part is None:
                raise RuntimeError('no partition (%s)' % (bad or 'shape'))
            with torch.no_grad():
                iargs = [args[i] for i in part['inv_pos']]
                # the closures of one residual (f_, g_, at_root) have the same
                # invariant graph and the same inputs: one compilation
                ikey = (part['sig'], tuple(
                    (tuple(a.shape), tuple(a.stride()), a.dtype,
                     a.data_ptr() % 16 == 0) for a in iargs)) \
                    if part['sig'] is not None else None
                inv_c = _INV_COMPILED.get(ikey) if ikey is not None else None
                if inv_c is None:
                    WHAT[0] = self.tag + '/inv'
                    inv_c = _compile(self.inner, copy.deepcopy(part['inv']),
                                     iargs)
                    if ikey is not None:
                        _INV_COMPILED[ikey] = inv_c
                inv = inv_c(*iargs)
                dargs = ([args[part['ts_pos']]] + list(inv)
                         + [args[i] for i in part['pd_pos']])
                WHAT[0] = self.tag + '/dep'
                dep_c = _compile(self.inner, copy.deepcopy(part['dep']), dargs)
                got = dep_c(*dargs)
                ref = self.full(*args)
                ok = len(got) == len(ref) and len(inv) == part['n_front']
                for r, o in zip(ref, got):
                    if not ok:
                        break
                    if torch.is_tensor(r):
                        ok = torch.is_tensor(o) and _bit_equal(r, o)
                    else:
                        ok = bool(r == o)
        except Exception as exc:
            ok = False
            if DEBUG:
                import traceback
                log.debug('hoist %s: build failed: %s' % (self.tag, exc))
                log.debug(traceback.format_exc())
        if not ok:
            if DEBUG:
                log.debug('hoist %s: off' % self.tag)
            self.state = 'off'
            return
        self.inv_c = inv_c
        self.lane = _Lane(part['dep'], dep_c, self.inner, self.tag + '/dep',
                          ts_idx=0)
        self.ts_pos = part['ts_pos']
        self.inv_pos = part['inv_pos']
        self.pd_pos = part['pd_pos']
        self.sig = part['sig']
        self.state = 'ok'
        if DEBUG:
            log.debug('hoist %s: on, %d of %d nodes depend on ts, %d values '
                  'carried' % (self.tag, part['n_dep'], part['n_nodes'],
                               part['n_front']))


class CompactBackend(object):
    """The backend core.fusion compiles its closures with: the inductor
    backend, and for a graph compiled while `fusion._COMPACT_NEXT` is set
    (a call marked compact) a _Residual around it."""

    def __init__(self, inner, flag):
        self.inner = inner
        self.flag = flag

    def __call__(self, gm, example_inputs):
        tag = self.flag[0]
        CALLS[0] += 1
        keep = None
        if tag and (graphsafe.compact_div() > 0 or HOIST):
            keep = copy.deepcopy(gm)
        full = self.inner(gm, example_inputs)
        if keep is None:
            return full
        why = _mutates(keep)
        if why:
            if DEBUG:
                log.debug('compact %s: not compactable: %s' % (tag, why))
            return full
        return _Residual(keep, full, self.inner, tag)
