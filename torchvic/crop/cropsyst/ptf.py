"""E14 of the VIC-CropSyst port (docs/connectors/cropsyst.md): the Saxton
pedotransfer function that turns clay and quartz (sand) fractions and the layer depth into the
Campbell soil parameters, and the maximum sorptivity used by the mechanistic irrigation runoff (E13).

    VIC_CropSyst/read_soilparam.c:742-808 (clay_input = 1 branch), :811-813 (S_max)
    VIC_CropSyst/crop/VIC_soil_hydrology.c  calc_Maximum_sorptivity

Re-derived from the shipped C++ (Academic Free License; no code text is copied). Every
function is elementwise, so any batch shape works ([cell, layer], [cell, crop, layer]); there are no
host syncs and no data-dependent control flow, so the functions are CUDA-graph capturable. The layer
axis is the last one. Constants are the reference's, including its 33.33 kPa field-capacity and
1500 kPa wilting-point heads (read_soilparam.c:784-785) and the 2650 kg/m3 particle density (:743).
"""
from __future__ import annotations

from typing import NamedTuple

import torch

SOIL_DENSITY = 2650.0                    # read_soilparam.c:743 [kg/m3]
FIELD_CAPACITY_HEAD = 33.33              # read_soilparam.c:784 [J/kg]
WILT_POINT_HEAD = 1500.0                 # read_soilparam.c:785 [J/kg]
WATER_POT_AT_FC = -33.0                  # read_soilparam.c:779 [J/kg]
WCR_FRACTION_OF_FC = 0.7                 # read_soilparam.c:788 (VIC Wcr = 0.7 FC)


class SaxtonSoil(NamedTuple):
    silt: torch.Tensor                   # fraction
    soil_density: torch.Tensor           # kg/m3
    porosity: torch.Tensor               # m3/m3
    bulk_density: torch.Tensor           # kg/m3
    b_campbell: torch.Tensor
    air_entry: torch.Tensor              # J/kg, negative (soil.VCS.AE)
    ksat: torch.Tensor                   # mm/day
    water_pot_at_fc: torch.Tensor        # J/kg
    field_capacity_mm: torch.Tensor
    wpwp_mm: torch.Tensor
    wcr_mm: torch.Tensor
    s_max: torch.Tensor                  # [..., ] cm, from layer 0


def max_sorptivity(porosity: torch.Tensor, b_campbell: torch.Tensor) -> torch.Tensor:
    """calc_Maximum_sorptivity (Rawls 1992): (2b+3)/(b+3) * 100 (0.341 porosity - 0.108), floored at 0.
    The sand and clay arguments of the C function are unused. A NaN gives 0, as the C `s >= 0` test."""
    hb = 100.0 * (0.341 * porosity - 0.108)
    s = (2.0 * b_campbell + 3.0) / (b_campbell + 3.0) * hb
    return torch.where(s >= 0.0, s, torch.zeros_like(s))


def saxton_ptf(clay: torch.Tensor, quartz: torch.Tensor, depth_m: torch.Tensor) -> SaxtonSoil:
    """E14 per layer. clay, quartz [fraction] and depth_m [m] have shape [..., layer]; s_max is
    returned per profile ([...]) from layer 0, as read_soilparam.c:811 does."""
    c = clay * 100.0                     # percent
    q = quartz * 100.0
    q2 = q * q
    c2 = c * c
    silt = 1.0 - clay - quartz                                            # :742
    porosity = 0.332 - 0.0007251 * q + torch.log10(c) * 0.1276            # :744
    soil_density = torch.full_like(porosity, SOIL_DENSITY)                # :743
    bulk = soil_density * (1.0 - porosity)                                # :745
    b = -(-3.14 - 0.00222 * c2 - 0.00003484 * q2 * c)                     # :746
    a_val = 100.0 * torch.exp(-4.396 - 0.0715 * c - 0.000488 * q2 - 0.00004285 * q2 * c)   # :747
    ae = -a_val * torch.pow(porosity, -b)                                 # :748
    ksat = 10.0 * 24.0 * torch.exp(12.012 - 0.0755 * q + (-3.895 + 0.03671 * q - 0.1103 * c
                                                           + 0.00087546 * c2) / porosity)   # :752
    fc = depth_m * 1000.0 * torch.pow(FIELD_CAPACITY_HEAD / a_val, -1.0 / b)               # :784
    wp = depth_m * 1000.0 * torch.pow(WILT_POINT_HEAD / a_val, -1.0 / b)                   # :785
    wcr = fc * WCR_FRACTION_OF_FC                                          # :788
    s_max = max_sorptivity(porosity[..., 0], b[..., 0])                    # :811
    return SaxtonSoil(silt, soil_density, porosity, bulk, b, ae, ksat,
                      torch.full_like(porosity, WATER_POT_AT_FC), fc, wp, wcr, s_max)
