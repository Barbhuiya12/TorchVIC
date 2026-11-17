"""E15 of the VIC-CropSyst port (docs/connectors/cropsyst.md): the short-grass
potential ET that drives CropSyst. VIC-CropSyst overrides the canopy and architectural resistances of the
PET_SHORT reference crop with rc = 70 s/m and rarc = 0 (VIC_CropSyst/compute_pot_evap.c:70-75, ifdef
VIC_CROPSYST_VERSION), then calls the ordinary Penman-Monteith (penman.c). Everything else of
compute_pot_evap (net short wave from the PET_SHORT albedo, net radiation, scaling by dt/24 and the daily
accumulation) is kept here so the kernel replaces the call for that one PET type.

Elementwise, any batch shape, no host syncs: capturable in a CUDA graph.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from ...core import vegetation

PET_SHORT_RC = 70.0          # s/m, compute_pot_evap.c:72
PET_SHORT_RARC = 0.0         # s/m, compute_pot_evap.c:73
HOURS_PER_DAY = 24.0
REF_ALBEDO = 0.23            # global.c:50, ref_veg_albedo[PET_SHORT]
REF_ROUGH = 0.0148           # global.c:51, ref_veg_rough[PET_SHORT] [m]
REF_DISPL = 0.08             # global.c:52, ref_veg_displ[PET_SHORT] [m]


class PetShort(NamedTuple):
    pot_evap_mm: torch.Tensor          # mm per time step (pot_evap[PET_SHORT])
    pot_evap_daily_mm: torch.Tensor    # the increment added to the daily total over `run_time_hours`
    daily_rate_mm: torch.Tensor        # penman() result, mm/day


def pet_short(tair, elevation, shortwave, net_longwave, albedo, vpd, ra, dt_hours,
              run_time_hours=None, rc=PET_SHORT_RC, rarc=PET_SHORT_RARC) -> PetShort:
    """Penman-Monteith potential ET of the short reference crop with the CropSyst resistances.

    net short wave = (1 - albedo) shortwave, radiation = net short + net long wave (compute_pot_evap.c:62-68);
    daily rate from `core.vegetation.penman` (penman.c); the step value is rate * dt / 24 (:77)."""
    net_short = (1.0 - albedo) * shortwave
    rad = net_short + net_longwave
    daily = vegetation.penman(tair, elevation, rad, vpd, ra, rc, rarc)
    run = dt_hours if run_time_hours is None else run_time_hours
    return PetShort(daily * dt_hours / HOURS_PER_DAY, daily * run / HOURS_PER_DAY, daily)
