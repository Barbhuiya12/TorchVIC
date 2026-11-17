"""Exact-math mode: glibc's pow/exp/sqrt at the sites that feed thresholds.

VIC 5.1.0 is C compiled -O0 against glibc, so pow, exp and sqrt there are the
glibc results. torch's CPU kernels (SLEEF under AVX2/AVX512, and its own
vectorised sqrt) differ from glibc in the last bit for 0.4 to 2.9 percent of
inputs, enough to move a value across an `if`. Exact mode is for verification
runs on the CPU:

  1. `enable()` sets ATEN_CPU_CAPABILITY=default (plain C++ kernels, whose
     pow is glibc's) before torch's first operation, and
  2. the functions below call libm (a small C loop over the tensor's buffer,
     or math.* element-wise when no C compiler is there) at the sites whose
     VIC counterpart is a libm call that feeds a threshold.

Switch: `torchvic run --exact-math`, the environment variable
TORCHVIC_EXACT_MATH=1, or `enable()` before the first torch operation.
Off (the default) every function here is the torch operation it replaces,
so GPU and default CPU runs are bit-identical to before. Active only for CPU
float64 tensors without autograd (reverse or forward mode); anything else
falls through to torch.
"""
import ctypes
import hashlib
import math
import os
import subprocess
import sys
import tempfile

import torch
import torch.autograd.forward_ad as _fwAD

EXACT = False
_lib = None

_SRC = r"""
#include <math.h>
#include <stddef.h>
void tv_pow(const double *a, const double *b, double *o, size_t n)
{ for (size_t i = 0; i < n; i++) o[i] = pow(a[i], b[i]); }
void tv_exp(const double *a, double *o, size_t n)
{ for (size_t i = 0; i < n; i++) o[i] = exp(a[i]); }
void tv_sqrt(const double *a, double *o, size_t n)
{ for (size_t i = 0; i < n; i++) o[i] = sqrt(a[i]); }
"""


def enable():
    """Switch exact mode on. Must run before torch's first operation, or the
    CPU kernel set is already chosen; returns whether the capability took."""
    global EXACT
    os.environ['TORCHVIC_EXACT_MATH'] = '1'
    os.environ['ATEN_CPU_CAPABILITY'] = 'default'
    EXACT = True
    ok = torch.backends.cpu.get_cpu_capability() == 'NO AVX'
    if not ok:
        print('torchvic: --exact-math was set after torch had run an '
              'operation; set ATEN_CPU_CAPABILITY=default in the '
              'environment', file=sys.stderr)
    return ok


def _load():
    global _lib
    if _lib is not None:
        return _lib
    _lib = False
    try:
        tag = hashlib.md5(_SRC.encode()).hexdigest()[:10]
        d = os.path.join(tempfile.gettempdir(), 'torchvic_xmath_%d' % os.getuid())
        os.makedirs(d, exist_ok=True)
        so = os.path.join(d, 'xmath_%s.so' % tag)
        if not os.path.exists(so):
            c = so[:-3] + '.%d.c' % os.getpid()
            open(c, 'w').write(_SRC)
            subprocess.run(['gcc', '-O1', '-fPIC', '-shared', '-fno-builtin',
                            '-fno-fast-math', '-ffp-contract=off', c, '-o',
                            so + '.%d' % os.getpid(), '-lm'], check=True,
                           capture_output=True)
            os.replace(so + '.%d' % os.getpid(), so)
        lib = ctypes.CDLL(so)
        for n in ('tv_pow', 'tv_exp', 'tv_sqrt'):
            getattr(lib, n).restype = None
        _lib = lib
    except Exception:                                       # noqa: BLE001
        _lib = False
    return _lib


def _active(*ts):
    if not EXACT:
        return False
    for t in ts:
        if isinstance(t, torch.Tensor):
            if (t.device.type != 'cpu' or t.dtype != torch.float64
                    or t.requires_grad):
                return False
            # a forward-mode dual (implicit._local_dfdt's tangent on T)
            # carries its derivative only through torch's own op
            if _fwAD.unpack_dual(t).tangent is not None:
                return False
    return True


def _np(t, shape):
    return torch.broadcast_to(t, shape).contiguous().numpy()


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _shape(*ts):
    return torch.broadcast_shapes(*[t.shape for t in ts
                                    if isinstance(t, torch.Tensor)])


def pow(a, b):
    """a ** b for tensors/scalars: glibc pow in exact mode."""
    if not _active(a, b):
        return torch.pow(a, b)
    ref = a if isinstance(a, torch.Tensor) else b
    sh = _shape(a, b)
    a = torch.as_tensor(a, dtype=torch.float64) if not isinstance(
        a, torch.Tensor) else a
    b = torch.as_tensor(b, dtype=torch.float64) if not isinstance(
        b, torch.Tensor) else b
    x, y = _np(a, sh), _np(b, sh)
    import numpy as np
    out = np.empty(sh, dtype=np.float64)
    lib = _load()
    if lib:
        lib.tv_pow(_ptr(x), _ptr(y), _ptr(out), ctypes.c_size_t(out.size))
    else:
        out.ravel()[:] = [math.pow(p, q) for p, q in zip(x.ravel(), y.ravel())]
    return torch.from_numpy(out).to(ref.device)


def _unary(name, fn, torch_fn, x):
    if not _active(x):
        return torch_fn(x)
    import numpy as np
    a = x.contiguous().numpy()
    out = np.empty(a.shape, dtype=np.float64)
    lib = _load()
    if lib:
        getattr(lib, name)(_ptr(a), _ptr(out), ctypes.c_size_t(out.size))
    else:
        out.ravel()[:] = [fn(v) for v in a.ravel()]
    return torch.from_numpy(out)


def exp(x):
    return _unary('tv_exp', math.exp, torch.exp, x)


def sqrt(x):
    return _unary('tv_sqrt', math.sqrt, torch.sqrt, x)


if os.environ.get('TORCHVIC_EXACT_MATH') == '1':
    enable()
