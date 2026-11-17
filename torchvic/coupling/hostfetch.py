# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""One device-to-host copy for everything the runtime must read on the host.

A step (or a window of steps, `ledger.check_every`) needs a few numbers on
the host: the ledger's terms, the per-element residual summaries, and the
connectors' validity flags. Reading each with `float(t)` costs one stream
synchronisation each; here every device value is queued with `add`, `run`
concatenates them per device and brings them over in one copy, and each
handle then returns a Python float (or list of floats).

    F = Fetch()
    h = F.add(x)            # x: a device tensor (any shape) or a number
    F.run()                 # the only synchronisation
    h.value()               # float, or a list for a non-scalar tensor

Values are carried as float64 (exact for float32 and float64 sources), so
the numbers equal `float(t)`.
"""
import torch


class _Const(object):
    def __init__(self, v):
        self.v = v

    def value(self):
        return self.v


class _Handle(object):
    def __init__(self, fetch, dev, off, n, scalar):
        self.f, self.dev, self.off, self.n, self.scalar = \
            fetch, dev, off, n, scalar

    def value(self):
        v = self.f._vals[self.dev][self.off:self.off + self.n]
        return v[0] if self.scalar else v


class Fetch(object):
    def __init__(self):
        self._parts = {}
        self._size = {}
        self._vals = {}

    def add(self, x):
        if not torch.is_tensor(x):
            return _Const(x)
        t = x.detach().reshape(-1)
        if t.dtype != torch.float64:
            t = t.to(torch.float64)
        dev = t.device
        off = self._size.get(dev, 0)
        self._parts.setdefault(dev, []).append(t)
        self._size[dev] = off + t.numel()
        return _Handle(self, dev, off, t.numel(), x.dim() == 0)

    def run(self):
        for dev, parts in self._parts.items():
            self._vals[dev] = (parts[0] if len(parts) == 1
                               else torch.cat(parts)).tolist()
        self._parts = {}
        return self


def host(x):
    """`x` as a Python float or list in one copy (a convenience for one
    value; use Fetch to share the copy between many)."""
    return Fetch().add(x) if not torch.is_tensor(x) else _one(x)


def _one(x):
    f = Fetch()
    h = f.add(x)
    f.run()
    return h.value()
