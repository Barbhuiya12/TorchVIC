# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Helpers the agricultural connectors share."""
import numpy as np
import torch

from ..spec import CouplingError

import weakref

_DRIVERS = weakref.WeakValueDictionary()


def shared_driver(vic_cfg, context):
    """The VIC driver (static data: soil, vegetation, cell areas, forcing
    calendar) of a `vic:` configuration block. One load per (global file,
    device, dtype) in the process while a connector holds it: the second connector reuses it
    (a weak cache: the open parameter files are released with the run)."""
    from ..vic import load_driver
    gp = (vic_cfg or {}).get('global_param')
    if not gp:
        raise CouplingError('this connector needs the VIC case it is '
                            'coupled to: give global_param (or driver=)')
    key = (str(gp), str(context.device), str(context.dtype),
           repr(sorted((k, str(v)) for k, v in vic_cfg.items())))
    d = _DRIVERS.get(key)
    if d is None:
        d = _DRIVERS[key] = load_driver(vic_cfg, context)
    return d


def clear_drivers():
    _DRIVERS.clear()


def store(value, n, dtype, device, default=1.0e30):
    """A [n] tensor from None (default, effectively unlimited), a number or
    a sequence."""
    if value is None:
        value = default
    t = torch.as_tensor(value, dtype=dtype, device=device)
    if t.dim() == 0:
        t = t.expand(n).clone()
    if t.shape != (n,):
        raise CouplingError('expected %d values, got %s' % (n,
                                                           tuple(t.shape)))
    return t


def fraction_layers(frac, max_moist, what):
    """absolute mm [cell, layer] from a fraction of max_moist (a number, or
    one per layer), or None."""
    if frac is None:
        return None
    f = torch.as_tensor(frac, dtype=max_moist.dtype, device=max_moist.device)
    if f.dim() > 1 or (f.dim() == 1 and f.shape[0] != max_moist.shape[-1]):
        raise CouplingError('%s: a number or one per layer' % what)
    return max_moist * f


def critical_from_params(driver, device, dtype, max_moist, default):
    """Wcr in mm [cell, layer] from the parameter file's Wcr_FRACT when the
    case has one, else `default`."""
    try:
        par = getattr(driver, 'par', None)          # a streamed case
        par = par if par is not None else driver.case.par
        wcr = par.critical_moisture()[0]
        wcr = torch.as_tensor(np.asarray(wcr), dtype=dtype, device=device)
        if wcr.shape == max_moist.shape:
            return wcr
        if wcr.dim() == 2 and wcr.T.shape == max_moist.shape:
            return wcr.T.contiguous()
    except Exception:                                    # noqa: BLE001
        pass
    return default
