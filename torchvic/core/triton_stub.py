# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Stand-ins for Triton, for installs that do not have it.

CPU-only torch wheels do not ship Triton (the CUDA wheels depend on it). A
number of modules define Triton kernels at import time, and the coupled
system's connectors import them, so without Triton even
`import torchvic.coupling.connectors` failed. Those modules now read

    try:
        import triton
        import triton.language as tl
    except ImportError:                  # CPU-only torch has no Triton
        from ..core.triton_stub import triton, tl

When Triton is importable the real modules are imported, exactly as before and
this file is not touched. When it is not, the modules still import: their
kernels become objects that raise one clear error if they are launched or
inspected, and so does any call into `triton` or `triton.language` other than
the few that a module makes while it is being imported (`@triton.jit`,
`tl.constexpr`). A kernel is launched only for a CUDA tensor, which needs a CUDA
build of torch and so Triton; a CPU run takes the torch path and never reaches
one. If a CPU path ever did reach a kernel it would stop with that error
rather than return a wrong result.
"""

_MESSAGE = ('%s needs Triton: this path launches a Triton kernel. CUDA builds '
            'of PyTorch include Triton (on Linux `pip install triton` adds '
            'it); it is missing or incomplete here, and CPU-only installs '
            'cannot run this path. Use the CPU device, or a CUDA build of '
            'PyTorch.')


class _Absent(object):
    """A name from Triton that is not there. Reading an attribute gives
    another such name (a module evaluates `tl.float64` or
    `libdevice.pow` in a default argument or an alias); calling one is a
    launch or a computation that needs Triton, and raises."""

    def __init__(self, name):
        self.__dict__['_name'] = name

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return _Absent('%s.%s' % (self.__dict__['_name'], attr))

    def __call__(self, *args, **kwargs):
        raise RuntimeError(_MESSAGE % self.__dict__['_name'])

    def __repr__(self):
        return '<%s: Triton is not installed>' % self.__dict__['_name']


class _Kernel(object):
    """What `@triton.jit` returns without Triton. It keeps the function but
    cannot be launched (`kernel[grid](...)`, `kernel(...)`) or inspected
    (`kernel.params`, `kernel.run`: what the launch helpers read)."""

    def __init__(self, fn):
        self.__dict__['fn'] = fn
        self.__dict__['__name__'] = getattr(fn, '__name__', 'kernel')
        self.__dict__['__doc__'] = getattr(fn, '__doc__', None)

    def _absent(self, *args, **kwargs):
        raise RuntimeError(_MESSAGE % self.__dict__['__name__'])

    __call__ = _absent

    def __getitem__(self, grid):
        return self._absent

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        self._absent()


class _Constexpr(object):
    """`tl.constexpr` as an annotation (`BF: tl.constexpr`) and as a value
    (`S = tl.constexpr(5)`)."""

    def __init__(self, value=None):
        self.value = value

    def __repr__(self):
        return 'constexpr(%r)' % (self.value,)


def _jit(fn=None, **options):
    """`@triton.jit` and `@triton.jit(do_not_specialize=[...])`."""
    if fn is None:
        return _jit
    return fn if isinstance(fn, _Kernel) else _Kernel(fn)


class _Triton(_Absent):
    """`triton`: the decorators a module applies while it is imported work;
    everything else (`triton.cdiv`, ...) is a launch-time call and raises."""
    jit = staticmethod(_jit)

    @staticmethod
    def autotune(*args, **kwargs):
        """`@triton.autotune(configs=[...], key=[...])`."""
        return _jit

    @staticmethod
    def heuristics(*args, **kwargs):
        """`@triton.heuristics({...})`."""
        return _jit

    @staticmethod
    def Config(*args, **kwargs):
        return _Absent('triton.Config')


class _Language(_Absent):
    """`triton.language`: only `constexpr` is used while a module is imported."""
    constexpr = _Constexpr


triton = _Triton('triton')
tl = _Language('triton.language')
libdevice = _Absent('triton.language.extra.libdevice')
