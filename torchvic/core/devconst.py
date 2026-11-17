"""Python numbers as tensors on a device, made once.

`torch.as_tensor(0.622, dtype=..., device='cuda')` copies one number from
pageable host memory to the GPU, and CUDA synchronises the stream before
such a copy: the host waits until every kernel already queued has run. In
the snow-pack residual that happened twice per evaluation, fourteen
evaluations a step -- 38 of the 49 host synchronisations of an ISC
water-balance step. The tensor is the same number every time, so it is made
once per (value, dtype, device) and reused; the arithmetic it enters is
unchanged, and so is every result, bit for bit.

The cached tensors are ordinary tensors (made outside inference mode, with
no gradient), so they can be used in inference mode and in a run with
gradients alike, and nothing writes to them.
"""
import struct

import torch

_CACHE = {}


def device_constant(value, dtype, device):
    """`torch.as_tensor(value, dtype=dtype, device=device)` for a Python
    number, cached. Anything else (a tensor, an array) is passed to
    torch.as_tensor unchanged, as before."""
    if torch.compiler.is_compiling():
        # traced into a fused region (core.fusion): the number becomes a
        # constant of the compiled kernel, a tensor passes through
        if torch.is_tensor(value):
            return value.to(dtype=dtype, device=device)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return torch.full((), value, dtype=dtype, device=device)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return torch.as_tensor(value, dtype=dtype, device=device)
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    # The bit pattern, not the value: -0.0 == 0.0 as dictionary keys.
    bits = (struct.pack('<d', float(value)) if isinstance(value, float)
            else ('i', value))
    key = (bits, dtype, device)
    t = _CACHE.get(key)
    if t is None and device.type == 'cuda' and \
            torch.cuda.is_current_stream_capturing():
        # first asked for inside a CUDA graph capture (a fused region's
        # eager fallback, core.fusion): a fill kernel of the graph, which
        # copies nothing from the host; not cached, as it holds its value
        # only once the graph has run
        return torch.full((), value, dtype=dtype, device=device)
    if t is None:
        with torch.inference_mode(False), torch.no_grad():
            t = torch.as_tensor(value, dtype=dtype, device=device)
        _CACHE[key] = t
    return t
