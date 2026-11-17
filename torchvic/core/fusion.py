"""Kernel fusion of the time step that keeps eager's float64 results bit
for bit (ROADMAP W2 P3).

torch.compile (inductor, Triton kernels) fuses a chain of element-wise
operations into one kernel: one launch and one pass over memory in place of
one of each per operation. Out of the box the fused arithmetic is not
eager's, in six ways, and `setup` closes each:

1. Contraction. Triton lets LLVM and ptxas turn a*b + c into one fused
   multiply-add (one rounding where eager, one kernel per operation, rounds
   twice). Every product is therefore emitted as PTX `mul.rn` (inline
   assembly, which no pass fuses into an add: _asm_mul), and Triton's own
   contraction stays on, because libdevice's functions (pow above all) are
   eager's only when compiled with it (nvcc's default, which compiled
   eager's pow): with contraction off, `pow` of 4M random doubles differed
   from eager's in 19 to 49 values, with it on in none. That lets pow, the
   one operation that used to cut every fused region in two, fuse
   (point 2). TORCHVIC_FUSION_FP=legacy is the earlier scheme
   (TRITON_DEFAULT_FP_FUSION=0, eager pow).
2. Powers. Inductor unrolls x**n for an integer n into products ((x*x)*(x*x)
   for x**4, 1/x * 1/x for x**-2), sends 2**x to exp2 and x**-0.5 to
   sqrt(1/x); eager has its own short list (x*x, x*x*x, 1/x, sqrt, rsqrt)
   and calls pow otherwise. Here inductor keeps the cases where its code is
   eager's (n = 1, 2, 3, -1, 0.5) and every other power is eager's pow,
   called from the compiled graph.
3. Reductions. A sum, mean, product or cumulative sum over a dimension adds
   in another order than eager's reduction kernel; they are eager's kernels.
4. Composites that eager computes with a fused multiply-add inside one
   kernel (lerp, addcmul, addcdiv, logaddexp, xlogy, softplus, remainder,
   fmod) are eager's kernels too.
5. Division by a tensor inductor finds constant (a device constant, a
   full_like) is reciprocal-multiplied by inductor; eager divides. It stays
   a division here.
6. A Python float that changes between calls (a second case in the same
   process) becomes a kernel input by default, and `x / f` an IEEE
   division where eager multiplies by 1/f: floats are kept constants.

Everything else a step uses was checked to be eager's bit for bit with
contraction off: + - * / (a division by a Python number is a product with
its reciprocal in both), exp, log, log10, log1p, expm1, sqrt, rsqrt, tanh,
atan, atan2, sin, cos, acos, erf, sinh, sigmoid, hypot, reciprocal, clamp,
minimum/maximum, where, abs, floor, amax/amin (tests/test_fusion.py, and
the whole model against its eager run: benchmarks/fusion).

CUDA only. On a CPU inductor's C++ divides where eager multiplies by the
reciprocal and vectorises exp and pow differently, so the CPU path (and
exact-math mode, xmath) is never compiled.

Switch: TORCHVIC_FUSION=0 in the environment (or FUSION = False before
the first step) runs every region eagerly. Compiled code is cached in
TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR (by default under
~/.cache/torchvic/compile, TORCHVIC_COMPILE_CACHE=0 leaves torch's own); the
contraction setting is part of Triton's cache key.
"""
import math
import os

import torch

from .diaglog import log

#: compile the fused regions (CUDA, no gradient) -- see the module docstring
FUSION = os.environ.get('TORCHVIC_FUSION', '1') != '0'

_ready = False


def _persistent_compile_cache():
    """Where inductor and Triton keep what they compile, when the
    environment names no place: $TORCHVIC_COMPILE_CACHE, else
    ~/.cache/torchvic/compile (XDG_CACHE_HOME), kept across runs (a second
    run finds every kernel, FX graph and AOTAutograd entry and compiles
    nothing). TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR set by the user
    win. Not made where the directory cannot be."""
    base = os.environ.get('TORCHVIC_COMPILE_CACHE')
    if base is None:
        base = os.path.join(os.environ.get(
            'XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache')),
            'torchvic', 'compile')
    if base in ('', '0'):
        return
    try:
        for var, sub in (('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                         ('TRITON_CACHE_DIR', 'triton')):
            if var not in os.environ:
                d = os.path.join(base, sub)
                os.makedirs(d, exist_ok=True)
                os.environ[var] = d
    except OSError:
        pass


if FUSION:
    _persistent_compile_cache()

#: debugging aid (TORCHVIC_FUSION_CHECK=1, with TORCHVIC_GRAPHS_NO_CAPTURE=1
#: so that every step runs in Python): each region call runs eagerly as
#: well, on copies of its containers, and the two are compared; MISMATCH
#: counts the calls that differed by region, and the first one's inputs
#: go to CHECK_DIR/<region>.pt
CHECK = os.environ.get('TORCHVIC_FUSION_CHECK', '0') != '0'
CHECK_DIR = os.environ.get('TORCHVIC_FUSION_CHECK_DIR')
MISMATCH = {}

#: debugging aid (TORCHVIC_FUSION_COUNT=1): region calls on the graph path,
#: compiled or eager, by function, printed at exit
COUNT = os.environ.get('TORCHVIC_FUSION_COUNT', '0') != '0'
import collections as _collections
COUNTS = _collections.Counter()
if COUNT:
    import atexit as _atexit

    def _print_counts():
        import sys
        for k_, v_ in sorted(COUNTS.items()):
            log.debug('fusion count %-60s %-8s %d' % (k_[0], k_[1], v_))
    _atexit.register(_print_counts)


def graphsafe_active():
    from . import graphsafe
    return graphsafe.ACTIVE


#: debugging aid (TORCHVIC_FUSION_STRICT=1): a region that would need a new
#: compile inside a CUDA graph capture raises instead of running eagerly
STRICT = os.environ.get('TORCHVIC_FUSION_STRICT', '0') != '0'
_CAPTURE_STANCE = {'0': 'eager_on_recompile', '1': 'fail_on_recompile',
                   '2': 'default'}[os.environ.get('TORCHVIC_FUSION_STRICT',
                                                  '0')]


def _own_eval_frame_callback():
    """torch's get_eval_frame_callback (torch._C._dynamo.eval_frame, torch
    2.6) returns the thread's callback -- None, False or a callable -- as a
    borrowed reference: the caller's release then drops a reference the
    function never took. torch.compiler.set_stance reads it three times
    (__init__, __enter__, __exit__, through _dynamo.eval_frame._set_stance),
    and the captured step enters a stance for each compiled region it runs
    (region / run below) while dynamo has set the callback to False: a
    no-grad LANE_COMPACT run lost about two references to False per step.
    CPython 3.11's bools are not immortal, so a few thousand runs in one
    process ended in "Fatal Python error: bool_dealloc".

    The getter is replaced, once, by one that takes the reference its
    caller will release, where a probe shows the reference is borrowed (a
    call lowers its result's count). A torch that returns a new reference,
    or a CPython whose None is immortal, is left alone. _set_stance imports
    the name from the C module at each call, so it reads the replacement."""
    try:
        import ctypes
        import sys
        from torch._C._dynamo import eval_frame as ef
        get = ef.get_eval_frame_callback
    except (ImportError, AttributeError):
        return False
    if getattr(get, '_torchvic', False):
        return True
    held = get()
    if held is None and sys.version_info >= (3, 12):
        return False                    # immortal: the probe cannot tell
    n0 = sys.getrefcount(held)
    get()
    lost = n0 - sys.getrefcount(held)
    if lost <= 0:
        return False
    incref = ctypes.pythonapi.Py_IncRef
    incref.argtypes, incref.restype = [ctypes.py_object], None
    for _ in range(2 * lost):
        incref(held)                    # the probe's two calls

    def get_eval_frame_callback():
        r = get()
        incref(r)
        return r
    get_eval_frame_callback._torchvic = True
    get_eval_frame_callback.__wrapped__ = get
    ef.get_eval_frame_callback = get_eval_frame_callback
    return True


#: True where torch's get_eval_frame_callback was found to return a
#: borrowed reference and was replaced (see _own_eval_frame_callback)
EVAL_FRAME_GETTER_FIXED = _own_eval_frame_callback()

#: 'asm' (default): products as mul.rn, Triton contraction on, libdevice pow
#: fused; 'legacy': contraction off, every pow eager's kernel
FP_MODE = os.environ.get('TORCHVIC_FUSION_FP', 'asm')
_FP_FUSION = '0' if FP_MODE == 'legacy' else '1'

if FUSION:
    # read by Triton wherever a kernel is compiled, the compile workers
    # included, which inherit the environment when they start: so set at
    # import, before any compile in this process
    os.environ['TRITON_DEFAULT_FP_FUSION'] = _FP_FUSION

#: reductions and FMA composites run as eager's own kernels (points 3, 4)
_EAGER_OPS = ('sum', 'mean', 'prod', 'cumsum', 'cumprod', 'nansum', 'var',
              'std', 'var_mean', 'std_mean', 'logsumexp', 'norm',
              'linalg_vector_norm', 'lerp', 'addcmul', 'addcdiv',
              'logaddexp', 'logaddexp2', 'xlogy', 'softplus', 'remainder',
              'fmod', 'dot', 'mv', 'mm', 'bmm', 'addmm', 'matmul',
              'trapezoid', 'trapz')

#: and, with contraction on (FP_MODE 'asm'), the one libdevice function found
#: to differ from eager's (log1p: 4 of 262,144 random values)
_EAGER_OPS_ASM = ('log1p',)

#: bumped whenever a lowering or an eager fallback set up here changes: it
#: is part of the compile cache keys (_CacheSalt), which cannot see them
_LOWERING_REV = 'asm2'

#: sums over one short dimension as eager's addition tree, fused
#: (core.xsum); TORCHVIC_XSUM=0 keeps every sum eager's kernel
XSUM = os.environ.get('TORCHVIC_XSUM', '1') != '0'

#: integer and half-integer powers inductor writes as eager does
_POW_NATIVE = (1.0, 2.0, 3.0, -1.0, 0.5)


#: private inductor names setup() rewires: (module, attribute) pairs
_PRIVATE_API = (
    ('torch._inductor.runtime.triton_heuristics',
     ('pointwise', 'triton_config', 'cached_autotune',
      'autotune_hints_to_configs', 'disable_pointwise_autotuning',
      'HeuristicType')),
    ('torch._inductor.codegen.triton', ('TritonOverrides',)),
    ('torch._inductor.lowering', ('register_lowering',
                                  'ELEMENTWISE_TYPE_PROMOTION_KIND')),
    ('torch._inductor.codecache', ('torch_key',)),
    ('torch._inductor.ir', ()),
    ('torch._inductor.virtualized', ('V',)),
)
_api_checked = None


def api_available():
    """True when the installed torch has the private inductor names that
    setup() patches. They are not a public interface and move between torch
    releases (torch 2.15 has no `disable_pointwise_autotuning`); when one is
    missing, fusion is switched off for the process with a one-time warning
    and every region runs the unfused (eager) path, which is the same model.
    """
    global _api_checked, FUSION
    if _api_checked is not None:
        return _api_checked
    import importlib
    missing = []
    for mod, names in _PRIVATE_API:
        try:
            m = importlib.import_module(mod)
        except Exception:
            missing.append(mod)
            continue
        missing += ['%s.%s' % (mod, n) for n in names if not hasattr(m, n)]
    _api_checked = not missing
    if missing:
        FUSION = False
        import warnings
        warnings.warn(
            'torchvic: torch %s lacks inductor internals that kernel fusion '
            'needs (%s); running unfused (same results, slower on GPU). '
            'Use the torch version in the install guide to get fusion.'
            % (torch.__version__, ', '.join(missing)),
            RuntimeWarning, stacklevel=2)
    return _api_checked


def setup():
    """Configure inductor for eager-identical float64 arithmetic. Idempotent;
    must run before the first compile in this process (the Triton option is
    read when a kernel is compiled, by the compile workers too). Does nothing
    (and turns fusion off) when the torch install lacks the private inductor
    names it needs, see `api_available`."""
    global _ready
    if _ready:
        return
    if not api_available():
        return
    os.environ['TRITON_DEFAULT_FP_FUSION'] = _FP_FUSION
    import torch._inductor.config as icfg
    import torch._inductor.decomposition as idec
    import torch._inductor.lowering as L
    import torch._dynamo as dynamo
    aten = torch.ops.aten
    # 3, 4: eager's kernels for reductions and FMA composites
    for name in _EAGER_OPS + (_EAGER_OPS_ASM if FP_MODE != 'legacy' else ()):
        pkt = getattr(aten, name, None)
        if pkt is None:
            continue
        for ov in pkt.overloads():
            op = getattr(pkt, ov)
            idec.decompositions.pop(op, None)
            # the dict CompositeImplicit ops are decomposed by in AOT
            L.make_fallback(op, warn=False, override_decomp=True)
    # 3 (P4): a sum over one short dimension is eager's own addition tree
    # written out (core.xsum), which fuses with its neighbours; any other
    # sum stays eager's kernel
    from . import xsum
    if XSUM:
        idec.decompositions[aten.sum.dim_IntList] = xsum.sum_decomp
    # 2: powers
    native = L.lowerings[aten.pow.Tensor_Scalar]

    def pow_exact(a, b):
        if (isinstance(b, (int, float)) and not isinstance(b, bool)
                and float(b) in _POW_NATIVE
                and isinstance(a, L.TensorBox)):
            return native(a, b)
        if FP_MODE != 'legacy':
            # libdevice's pow, contraction on, is eager's ::pow: a tensor
            # exponent, a float exponent that is not an integer (inductor
            # unrolls an integer one) nor -0.5 (eager's rsqrt), a number
            # as base other than 2 (inductor's exp2)
            if isinstance(a, L.TensorBox) and isinstance(b, L.TensorBox):
                return native(a, b)
            if (isinstance(a, L.TensorBox) and isinstance(b, float)
                    and math.isfinite(b) and b != int(b) and b != -0.5):
                return native(a, b)
            if (isinstance(b, L.TensorBox) and isinstance(a, (int, float))
                    and not isinstance(a, bool) and float(a) not in (1.0, 2.0)
                    and float(a) > 0):
                return native(a, b)
        if isinstance(a, L.TensorBox) and isinstance(b, L.TensorBox):
            return L.fallback_pow_tensor_tensor(a, b)
        if isinstance(a, L.TensorBox):
            return L.fallback_pow_tensor_scalar(a, b)
        return L.fallback_pow_scalar(a, b)
    for ov in aten.pow.overloads():
        op = getattr(aten.pow, ov)
        if op in L.lowerings:
            L.lowerings[op] = pow_exact
    if FP_MODE != 'legacy':
        _asm_products()
    for op in (aten.square.default, aten.square.out):
        idec.decompositions.pop(op, None)
    # 5: a division by a Python number is a product with its reciprocal in
    # eager CUDA (div_true_kernel_cuda) and in inductor alike; a division by
    # a tensor that inductor finds constant (a device_constant, a
    # full_like) is an IEEE division in eager and must stay one
    div_native = L.lowerings[aten.div.Tensor]

    def div_exact(a, b):
        if (isinstance(b, (int, float)) and not isinstance(b, bool)) or \
                not isinstance(a, L.TensorBox) or \
                not isinstance(b, L.TensorBox):
            return div_native(a, b)
        if not (a.get_dtype().is_floating_point
                and b.get_dtype().is_floating_point):
            return div_native(a, b)
        return L.make_pointwise(lambda x, y: L.ops.truediv(x, y))(a, b)
    L.register_lowering(
        [aten.div.Tensor, aten.true_divide.Tensor], broadcast=True,
        type_promotion_kind=L.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )(div_exact)
    # how the fused graph is cut into kernels changes no element's
    # arithmetic (contraction off, every operation eager's): only speed.
    # TORCHVIC_INDUCTOR="name=value,..." sets inductor options (tuning)
    for kv in filter(None, os.environ.get('TORCHVIC_INDUCTOR', '').split(',')):
        k_, v_ = kv.split('=')
        obj = icfg
        parts = k_.split('.')
        for p_ in parts[:-1]:
            obj = getattr(obj, p_)
        cur = getattr(obj, parts[-1])
        v_ = (v_ == 'True') if isinstance(cur, bool) else type(cur)(v_)
        setattr(obj, parts[-1], v_)
    _tune_pointwise()
    _split_flag_reductions()
    _compile_caches()
    # no autotuning or reordering that could change what a kernel computes
    icfg.max_autotune = False
    icfg.max_autotune_pointwise = False
    icfg.triton.cudagraphs = False        # the caller captures (basin)
    # indices the step gathers with are in range by construction; the
    # bounds assert per indirect load is a check, not arithmetic
    icfg.assert_indirect_indexing = False
    # 6: every Python number a region reads is a constant of its kernels,
    # as in eager. By default dynamo turns a float that changed between
    # calls (a second case in one process) into an input of the kernel,
    # where `x / f` is then an IEEE division and eager's is a product with
    # 1/f; shapes stay static likewise
    dynamo.config.specialize_float = True
    dynamo.config.automatic_dynamic_shapes = False
    # one compiled variant per call site and shape; the step has many
    dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 64)
    dynamo.config.accumulated_cache_size_limit = max(
        dynamo.config.accumulated_cache_size_limit, 1024)
    _ready = True


def ensure_setup():
    """`setup()` when fused regions are on and a GPU is present, else
    nothing. Call before a torch.compile of anything outside this module
    (the groundwater solver): setup() rewires inductor for the whole
    process (reductions, divisions, powers as eager's), so a function
    compiled before it and one compiled after it do not round alike. When
    setup() ran was an accident of the run's history (the first fused
    region of the first captured window), so a run restarted from a
    checkpoint compiled such a function under another configuration than
    the continuous run did, and the two differed in the last bits."""
    if FUSION and torch.cuda.is_available():
        setup()


def _asm_products():
    """Every float product Triton compiles is PTX `mul.rn` (inline
    assembly, opaque to LLVM and ptxas, which therefore cannot contract it
    with an add into a fused multiply-add): eager rounds each product."""
    import torch._inductor.codegen.triton as tt
    if getattr(tt.TritonOverrides, '_torchvic_mul', False):
        return
    asm = {torch.float64: ("mul.rn.f64 $0, $1, $2;", "=d,d,d", 'tl.float64'),
           torch.float32: ("mul.rn.f32 $0, $1, $2;", "=f,f,f", 'tl.float32')}

    def mul(a, b):
        da, db = getattr(a, 'dtype', None), getattr(b, 'dtype', None)
        if da is not None and da == db and da in asm:
            code, con, ty = asm[da]
            return ("tl.inline_asm_elementwise('%s', '%s', [%s, %s], "
                    "dtype=%s, is_pure=True, pack=1)" % (code, con, a, b, ty))
        return '%s * %s' % (a, b)
    tt.TritonOverrides.mul = staticmethod(mul)
    tt.TritonOverrides._torchvic_mul = True


def _tune_pointwise():
    """Inductor's 1-D pointwise kernels autotune over two block shapes, 4
    and 2 elements a thread; a long fused region spills its registers at
    either (the vegetation step: 1,656 bytes, 0.6 ms a step on the H200).
    One element a thread (256 and 128 a block) is offered as well. A
    block shape changes which thread computes an element, never how."""
    import functools
    import torch._inductor.runtime.triton_heuristics as th
    if getattr(th.pointwise, '_torchvic', False):
        return
    orig = th.pointwise

    def pointwise(size_hints, triton_meta, tile_hint=None, filename=None,
                  min_elem_per_thread=0, inductor_meta=None):
        im = {} if inductor_meta is None else inductor_meta
        if len(size_hints) != 1 or th.disable_pointwise_autotuning(im):
            return orig(size_hints, triton_meta, tile_hint=tile_hint,
                        filename=filename,
                        min_elem_per_thread=min_elem_per_thread,
                        inductor_meta=inductor_meta)
        numel = size_hints['x']
        bs = max(256, min(numel // 128, 1024))
        tc = functools.partial(th.triton_config,
                               min_elem_per_thread=min_elem_per_thread)
        configs = [tc(size_hints, bs, num_elements_per_warp=256),
                   tc(size_hints, bs // 2, num_elements_per_warp=64),
                   tc(size_hints, 256, num_elements_per_warp=32),
                   tc(size_hints, 128, num_elements_per_warp=32),
                   *th.autotune_hints_to_configs(
                       im.get('autotune_hints', set()), size_hints, bs,
                       triton_meta['device'])]
        seen, uniq = set(), []
        for c in configs:
            k = (tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages)
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        return th.cached_autotune(size_hints, uniq, triton_meta=triton_meta,
                                  inductor_meta=im,
                                  heuristic_type=th.HeuristicType.POINTWISE,
                                  filename=filename)
    pointwise._torchvic = True
    th.pointwise = pointwise


#: a whole-tensor boolean reduction (any/all, a graph-safe flag) over more
#: than this many elements is split into blocks of this many
FLAG_SPLIT = 512


try:
    from torch._inductor.custom_graph_pass import CustomGraphPass as _Pass
except ImportError:     # pragma: no cover
    _Pass = object


class _CacheSalt(_Pass):
    """A no-op post-grad pass whose uuid enters inductor's FX graph and
    AOTAutograd cache keys: graphs lowered with _split_flag_reductions are
    not mistaken for ones lowered without it."""

    def __call__(self, graph):
        return None

    def uuid(self):
        return 'torchvic-flag-split-%d-%s-%s' % (FLAG_SPLIT, FP_MODE,
                                                 _LOWERING_REV)


def _split_flag_reductions():
    """A flag a region computes from its whole tensor (`(...).any()`,
    graphsafe's stop tests and checks) is a reduction to one element, and
    inductor fuses the element-wise work feeding it into the reduction
    kernel, then splits that kernel into ceil(n / 8192) blocks (none at
    n <= 8192): the canopy's fused residual on 22,279 lanes ran on 3
    blocks, 0.84 ms a step on the RTX 6000 against 0.05 ms as a pointwise
    kernel. Such a reduction is split here into blocks of FLAG_SPLIT
    elements, as parallel as the pointwise kernel. any/all/max/min of
    booleans are exact in any order, so no value changes."""
    import math
    import torch._inductor.config as icfg
    import torch._inductor.ir as ir
    from torch._inductor.virtualized import V
    if getattr(ir.Reduction.num_splits, '_torchvic', False):
        return
    orig = ir.Reduction.num_splits

    def num_splits(device, dst_dtype, src_dtype, inner_fn, ranges,
                   reduction_ranges, reduction_type, reduction_numel,
                   input_node=None):
        hint, split = orig(device, dst_dtype, src_dtype, inner_fn, ranges,
                           reduction_ranges, reduction_type,
                           reduction_numel, input_node)
        if (dst_dtype != torch.bool or src_dtype != torch.bool
                or reduction_type not in ('any', 'max', 'min')
                or split == -1 or device.type != 'cuda'):
            return hint, split
        sv = V.graph.sizevars
        rn = sv.symbolic_hint(reduction_numel)
        nn = sv.symbolic_hint(ir.sympy_product(ranges))
        if not (isinstance(rn, int) or hasattr(rn, 'is_Integer')
                and rn.is_Integer) or int(nn) != 1:
            return hint, split
        rn = int(rn)
        if rn <= 8 * FLAG_SPLIT:
            return hint, split
        return ir.ReductionHint.INNER, max(int(split),
                                           int(math.ceil(rn / FLAG_SPLIT)))
    num_splits._torchvic = True
    ir.Reduction.num_splits = staticmethod(num_splits)
    if icfg.post_grad_custom_post_pass is None:
        icfg.post_grad_custom_post_pass = _CacheSalt()


def _compile_caches():
    """A warm process loads what an earlier one compiled: AOTAutograd's
    cache on top of inductor's FX graph cache (no re-trace of the joint
    graph), and torch's source hash kept in the cache directory (0.7 s to
    compute, the same for one installation)."""
    import json
    import torch._functorch.config as fcfg
    import torch._inductor.codecache as cc
    fcfg.enable_autograd_cache = True
    # a region returning a view of its input (a month's slice of a field)
    # has the view rebuilt by as_strided rather than by replaying the view
    # ops, which AOTAutograd's cache cannot store; the same tensor either way
    fcfg.view_replay_for_aliased_outputs = False
    if getattr(cc.torch_key, '_torchvic', False):
        return
    orig = cc.torch_key
    try:
        from torch._inductor.runtime.cache_dir_utils import cache_dir
        d = cache_dir()
    except Exception:
        return
    root = os.path.dirname(torch.__file__)
    stamp = [torch.__version__, root]
    for f in ('', '__init__.py', 'version.py', '_inductor',
              '_inductor/codecache.py', '_dynamo', '_functorch'):
        try:
            stamp.append(os.stat(os.path.join(root, f)).st_mtime_ns)
        except OSError:
            return
    path = os.path.join(d, 'torchvic_torch_key.json')

    def torch_key():
        try:
            with open(path) as fh:
                rec = json.load(fh)
            if rec.get('stamp') == stamp:
                return bytes.fromhex(rec['key'])
        except (OSError, ValueError, KeyError):
            pass
        key = orig()
        try:
            os.makedirs(d, exist_ok=True)
            tmp = '%s.%d' % (path, os.getpid())
            with open(tmp, 'w') as fh:
                json.dump(dict(stamp=stamp, key=key.hex()), fh)
            os.replace(tmp, path)
        except OSError:
            pass
        return key
    import functools
    torch_key = functools.lru_cache(None)(torch_key)
    torch_key._torchvic = True
    cc.torch_key = torch_key


def wants(args):
    """Compile this call: fusion on, the step on its CUDA graph path
    (graphsafe.ACTIVE: fixed shapes, so one compile per region), no
    gradient, a CUDA tensor among the arguments, exact-math mode off."""
    from . import graphsafe, xmath
    if (not FUSION or not graphsafe.ACTIVE or torch.is_grad_enabled()
            or xmath.EXACT):
        return False
    if not _ready and not api_available():
        return False
    for a in args:
        if torch.is_tensor(a):
            return a.is_cuda
        if isinstance(a, dict):
            a = a.values()
        elif not isinstance(a, (list, tuple)):
            continue
        for v in a:
            if torch.is_tensor(v):
                return v.is_cuda
    return False


def region(fn):
    """`fn` (a pure function of tensors: no host reads, no in-place updates
    of its arguments) compiled as one fused region where `wants` holds,
    else `fn` itself. The compile happens at the first such call -- in the
    eager step before a graph capture, never inside one."""
    compiled = [None]

    def call(*a, **k):
        if torch.compiler.is_compiling():
            # traced as part of an enclosing region: inlined there
            return fn(*a, **k)
        if not wants(a):
            if COUNT and graphsafe_active():
                COUNTS[(getattr(fn, '__qualname__', '?'), 'eager')] += 1
            return fn(*a, **k)
        if COUNT:
            COUNTS[(getattr(fn, '__qualname__', '?'), 'compiled')] += 1
        if compiled[0] is None:
            setup()
            compiled[0] = torch.compile(fn, dynamic=False, fullgraph=False)
        if CHECK and not torch.cuda.is_current_stream_capturing():
            return _checked(fn, compiled[0], a, k)
        if torch.cuda.is_current_stream_capturing():
            # inside a capture nothing may be compiled (a new kernel is
            # run to tune it): a call the warm-up step did not compile
            # for runs eagerly, which gives the same numbers
            with torch.compiler.set_stance(_CAPTURE_STANCE):
                return compiled[0](*a, **k)
        return compiled[0](*a, **k)
    call.__wrapped__ = fn
    call.__name__ = getattr(fn, '__name__', 'region')
    return call


def run(fn, *args, compact=False):
    """fn(*args) -- a closure made afresh each step, reading the step's tensors
    from its enclosing scope -- as one fused region where `wants(args)`
    holds, else called as it is. The compiled code is dynamo's, cached on
    fn's code object, so a new closure of the same code reuses it while
    its guards (the closed-over values' shapes, dtypes and constants)
    hold. fn must be pure as `region` asks: read its closure, return its
    results (a dict update of a closed-over dict is allowed)."""
    if torch.compiler.is_compiling():
        return fn(*args)
    if not wants(args if args else _closure_values(fn)):
        if COUNT and graphsafe_active():
            COUNTS[(fn.__name__, 'eager')] += 1
        return fn(*args)
    if COUNT:
        COUNTS[(fn.__name__, 'compiled')] += 1
    setup()
    comp = _compile_closure(fn)
    if compact and os.environ.get('TORCHVIC_COMPACT_DEBUG') and \
            fn.__name__ not in _SEEN_RUN:
        import sys
        _SEEN_RUN.add(fn.__name__)
        log.debug('fusion.run compact call: %s, capturing %s'
              % (fn.__name__, torch.cuda.is_current_stream_capturing()))
    if CHECK and not torch.cuda.is_current_stream_capturing():
        return _checked(fn, comp, args, {})
    if torch.cuda.is_current_stream_capturing():
        if os.environ.get('TORCHVIC_FUSION_GUARDS'):
            _report_guards(fn, args)
        with torch.compiler.set_stance(_CAPTURE_STANCE):
            return comp(*args)
    # a graph compiled by this call may run on a subset of its lanes
    # (core.compact); read by the backend while the call compiles
    _COMPACT_NEXT[0] = fn.__name__ if compact else None
    try:
        if compact and os.environ.get('TORCHVIC_COMPACT_DEBUG') and \
                fn.__name__ not in _SEEN_RUN2:
            import sys
            from . import compact as _cp
            n0 = _cp.CALLS[0]
            res = comp(*args)
            _SEEN_RUN2.add(fn.__name__)
            log.debug('fusion.run %s: first call made %d backend calls'
                  % (fn.__name__, _cp.CALLS[0] - n0))
            return res
        return comp(*args)
    finally:
        _COMPACT_NEXT[0] = None


_RUN_CTX = []

#: the name of the closure being compiled when its call is marked compact
_COMPACT_NEXT = [None]
_SEEN_RUN = set()
_SEEN_RUN2 = set()


def _compile_closure(fn):
    """torch.compile(fn, dynamic=False, fullgraph=False) for a closure made
    afresh each step, through one dynamo context made once. torch.compile
    makes a new inductor backend at every call and dynamo keeps each in
    eval_frame.cached_backends until torch._dynamo.reset(): a run grew by
    a backend (and its config dict) per compiled step, without end. The
    same backend object also keeps dynamo's backend guard satisfied, so a
    new closure of the same code reuses its compiled variant as before."""
    if not _RUN_CTX:
        import torch._dynamo
        backend = torch._TorchCompileInductorWrapper('default', None, False)
        from . import compact
        backend = compact.CompactBackend(backend, _COMPACT_NEXT)
        if os.environ.get('TORCHVIC_HOIST_PROBE'):
            backend = _ProbeBackend(backend)
        _RUN_CTX.append(torch._dynamo.optimize(
            backend=backend, nopython=False, dynamic=False, disable=False))
    return _RUN_CTX[0](fn)


def _closure_values(fn):
    """The values fn closes over (for `wants` when it takes no arguments)."""
    out = []
    for c in fn.__closure__ or ():
        try:
            v = c.cell_contents
        except ValueError:
            continue
        if torch.is_tensor(v) or isinstance(v, (dict, list, tuple)):
            out.append(v)
    return out


def _report_guards(fn, args):
    """Debugging aid: which guard of each compiled variant of `fn` fails
    for this call."""
    import sys
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    code = fn.__code__
    loc = dict(zip(code.co_varnames[:code.co_argcount], args))
    for name, cell in zip(code.co_freevars, fn.__closure__ or ()):
        try:
            loc[name] = cell.cell_contents
        except ValueError:
            pass
    for i, ce in enumerate(_debug_get_cache_entry_list(code)):
        r = ce.guard_manager.check_verbose(loc)
        log.debug('fusion guards %s[%d]: %s %s' % (
            fn.__name__, i, r.result, list(r.verbose_code_parts)[:6]))


def opaque(fn):
    """`fn` run as it is, never traced into an enclosing region: for the
    root finders, whose loops are host-decided (graphsafe.Segmenter). The
    region resumes after the call; regions inside `fn` are compiled as
    their own."""
    import torch._dynamo
    out = torch._dynamo.disable(fn, recursive=True)
    out.__wrapped__ = fn
    return out


def _copy_tree(x):
    """New containers around the same tensors (a region may update its
    dict and list arguments, never a tensor in place)."""
    if isinstance(x, dict):
        return type(x)((k, _copy_tree(v)) for k, v in x.items()) \
            if type(x) is dict else x
    if isinstance(x, list):
        return [_copy_tree(v) for v in x]
    if isinstance(x, tuple) and not hasattr(x, '_fields'):
        return tuple(_copy_tree(v) for v in x)
    return x


def _diff_tree(x, y, path, out):
    if torch.is_tensor(x) and torch.is_tensor(y):
        if x.shape != y.shape or not torch.equal(
                torch.nan_to_num(x, nan=1e300), torch.nan_to_num(y, nan=1e300)):
            d = ((x.double() - y.double()).abs().nan_to_num().max().item()
                 if x.shape == y.shape and x.is_floating_point() else -1.0)
            out.append((path, d))
    elif isinstance(x, dict) and isinstance(y, dict):
        for k in x:
            if k in y:
                _diff_tree(x[k], y[k], '%s.%s' % (path, k), out)
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        for i, (u, v) in enumerate(zip(x, y)):
            _diff_tree(u, v, '%s[%d]' % (path, i), out)


def _checked(fn, comp, a, k):
    global FUSION
    a0, k0 = _copy_tree(a), _copy_tree(k)
    a1, k1 = _copy_tree(a), _copy_tree(k)
    FUSION = False
    try:
        ref = fn(*a1, **k1)
    finally:
        FUSION = True
    out = comp(*a, **k)
    bad = []
    _diff_tree(ref, out, 'out', bad)
    _diff_tree(a1, a, 'args', bad)
    _diff_tree(k1, k, 'kw', bad)
    if bad:
        name = getattr(fn, '__qualname__', getattr(fn, '__name__', '?'))
        n = MISMATCH.get(name, 0)
        MISMATCH[name] = n + 1
        if n == 0:
            import sys
            log.debug('fusion check: %s differs: %s' % (name, bad[:8]))
            if CHECK_DIR:
                os.makedirs(CHECK_DIR, exist_ok=True)
                try:
                    torch.save(dict(args=a0, kwargs=k0),
                               os.path.join(CHECK_DIR, name + '.pt'))
                except Exception as e:          # an unpicklable argument
                    log.debug('fusion check: inputs not saved: %s' % e)
    return out


class _ProbeBackend(object):
    """Debugging aid (TORCHVIC_HOIST_PROBE=1): for each graph dynamo hands
    the backend, the share of its nodes (and of their cost: a power 20, an
    exp or log 8, a division 4, anything else 1) that do not depend on the
    placeholder `L_ts_`, the surface temperature being solved for."""

    def __init__(self, inner):
        self.inner = inner
        self.n = 0

    def __call__(self, gm, example_inputs):
        import sys
        dep = set()
        tot = {'n': 0, 'dn': 0, 'c': 0.0, 'dc': 0.0}
        heavy = {'pow': 20.0, 'exp': 8.0, 'log': 8.0, 'div': 4.0,
                 'sqrt': 4.0, 'tanh': 8.0, 'expm1': 8.0}
        ph = [n.name for n in gm.graph.nodes if n.op == 'placeholder']
        for n in gm.graph.nodes:
            if n.op == 'placeholder':
                if n.name.startswith('l_ts_') or n.name == 'ts':
                    dep.add(n)
                continue
            if n.op != 'call_function' and n.op != 'call_method':
                continue
            nm = str(getattr(n.target, '__name__', n.target))
            c = 1.0
            for k, v in heavy.items():
                if k in nm:
                    c = v
            tot['n'] += 1
            tot['c'] += c
            if any(a in dep for a in n.all_input_nodes):
                dep.add(n)
                tot['dn'] += 1
                tot['dc'] += c
        self.n += 1
        log.debug('hoist-probe graph %d: placeholders %d (ts: %s) nodes %d, '
              'dependent on ts %d (%.0f%%); cost %.0f, dependent %.0f '
              '(%.0f%%)' % (self.n, len(ph),
                            any(p.startswith('l_ts_') or p == 'ts'
                                for p in ph), tot['n'], tot['dn'],
                            100.0 * tot['dn'] / max(tot['n'], 1), tot['c'],
                            tot['dc'], 100.0 * tot['dc'] / max(tot['c'], 1)))
        return self.inner(gm, example_inputs)
