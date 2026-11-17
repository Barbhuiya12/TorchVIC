# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Run the WOFOST engine over a whole forcing series for many cells."""
import torch

from .wofost72 import Drivers, Wofost72

FORCING = ('tmin', 'tmax', 'irrad', 'rain', 'e0', 'es0', 'et0')


def drivers_at(forcing, i, lat, doy0=None):
    """`forcing[name]` is [T, N] (or [T, 1]); lat [N]; `forcing['doy']` is
    [T, N] / [T, 1] day of year."""
    n = lat.shape[0]
    g = lambda k: forcing[k][i].expand(n)
    return Drivers(tmin=g('tmin'), tmax=g('tmax'), irrad=g('irrad'), rain=g('rain'),
                   e0=g('e0'), es0=g('es0'), et0=g('et0'), lat=lat, doy=g('doy'))


@torch.no_grad()
def simulate(params, forcing, lat, mode='pp', rftra=None, tra=None, evs=None,
             outputs=('dvs', 'lai', 'tagp', 'twso'), harvest_idx=None, max_duration=None,
             engine=None):
    """Simulate T days for all cells; returns {name: [T, N]} and the engine.

    forcing  dict of [T, N] (or [T, 1]) tensors: tmin tmax irrad rain e0 es0 et0 doy
    lat      [N] latitude (degrees)
    rftra    [T, N] transpiration reduction factor, for mode 'external'
    """
    T = forcing['tmin'].shape[0]
    w = engine or Wofost72(params, mode=mode, max_days=T, harvest_idx=harvest_idx,
                           max_duration=max_duration)
    res = {k: [] for k in outputs}

    def rec():
        o = w.output()
        for k in outputs:
            res[k].append(o[k].clone())

    def ext(i):
        return (None if rftra is None else rftra[i], None if tra is None else tra[i],
                None if evs is None else evs[i])
    w.start(drivers_at(forcing, 0, lat), *ext(0))
    rec()
    for i in range(1, T):
        w.advance(drivers_at(forcing, i, lat), *ext(i))
        rec()
    return {k: torch.stack(v) for k, v in res.items()}, w
