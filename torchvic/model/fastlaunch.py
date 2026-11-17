"""Low-overhead launches of Triton kernels that run many times per step.

`jitfn[grid](...)` re-binds and re-specialises every argument in Python on
every call (tens of microseconds; the VIC-WUR sweep makes some 2000 such
launches a day). `launch(jitfn, grid, args, constexprs)` does that once per
specialisation: the first call goes through Triton (compiling if needed)
and keeps the compiled kernel under the call's specialisation key, which
is what Triton's own key depends on -- the dtype and 16-byte alignment of
every pointer and, for an integer not listed in `do_not_specialize`,
whether it is 1 or a multiple of 16 -- plus the constexpr values. Later
calls with the same key launch the compiled kernel directly. The kernel and
its arguments are Triton's own, so the result is the same.

`args` are the kernel's non-constexpr arguments in signature order and
`constexprs` its constexpr keyword arguments (num_warps etc. included).
"""
import torch

_CACHE = {}


def _params(jitfn):
    """The non-constexpr parameters of jitfn: (name, do_not_specialize)."""
    ps = getattr(jitfn, '_fl_params', None)
    if ps is None:
        ps = [bool(p.do_not_specialize) for p in jitfn.params
              if not p.is_constexpr]
        jitfn._fl_params = ps
    return ps


def _key(jitfn, args, constexprs):
    nospec = _params(jitfn)
    if len(nospec) != len(args):
        return None
    k = [id(jitfn), tuple(sorted(constexprs.items()))]
    for a, dns in zip(args, nospec):
        if isinstance(a, torch.Tensor):
            k.append((a.dtype, a.data_ptr() % 16 == 0))
        elif isinstance(a, bool):
            k.append(('b', a))
        elif isinstance(a, int):
            big = not -2**31 <= a < 2**31
            k.append(('i', big) if dns else ('i', big, a == 1, a % 16 == 0))
        elif isinstance(a, float):
            k.append('f')
        else:
            return None
    return tuple(k)


def launch(jitfn, grid, args, constexprs):
    key = _key(jitfn, args, constexprs)
    kern = _CACHE.get(key) if key is not None else None
    if kern is None:
        kern = jitfn[grid](*args, **constexprs)
        if key is not None and kern is not None:
            _CACHE[key] = kern
        return
    g = tuple(grid) + (1, 1)
    stream = torch.cuda.current_stream().cuda_stream
    kern.run(g[0], g[1], g[2], stream, kern.function, kern.packed_metadata,
             None, None, None, *args)
