# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-CropSyst's changes to VIC's core (docs/connectors/cropsyst.md), every
one OPT-IN and VIC 5 bit for bit when off.

They ride on the VIC-WUR hook mechanism (`torchvic.vicwur.core.VicwurCore`,
on the padded grid, the covered-tile lanes and in the CUDA graphs); only
what CropSyst needs beyond it is added there (transp, transp_mask,
wpwp_floor, pet_short) or here:

  transp_override  VicwurCore(transp=, transp_mask=): CropSyst's per-layer
                   uptake replaces VIC's transpiration on the crop tiles
                   (P1; VIC_soil.cpp:231-232 writes it into the layer,
                   runoff.c:370-374 adds it to layer 0's evaporation)
  veg_override     `veg_override()` -> VicwurCore(veg=, veg_mask=): CropSyst's
                   LAI, cover (fcanopy), albedo, displacement, roughness,
                   roots and canopy resistances (P6; full_energy.c:1174-1189)
  wpwp_floor       VicwurCore(wpwp_floor=): minimum liquid moisture Wpwp
                   below the top layer (P3; runoff.c:301-309, :555-564,
                   :706-715, :761-768)
  mu1              `mu_crop`: VIC 5 has no distributed precipitation, every
                   tile's mu is 1, which is what dist_prec.c:184-186 forces on
                   a crop tile; the option is the identity here (P4)
  one_band         `case.load(one_band_cells=)` /
                   `snowbands.one_band_for_crop` (P4; vicNl.c:447-455)
  saxton           `saxton_soil_params` (P5; read_soilparam.c:722-815)
  vcs_drain        VicwurCore(vcs_drain=): VIC-CropSyst's drainage law, Field_
                   Capacity cap and floor, irrigated redistribution
                   (runoff.c:575-621) with `saxton_soil_params(vcs_drainage=)`
  pet_short        VicwurCore(pet_short=True, tile_fluxes=True): tf_pet_short
                   (P8; compute_pot_evap.c:70-75)

`core()` builds a VicwurCore from the options; every default is off.
"""
import torch

from .core import VicwurCore

__all__ = ['core', 'veg_override', 'mu_crop', 'saxton_soil_params',
           'MU_CROP', 'WCR_FRACTION_OF_FC', 'WPWP_CLAMP', 'FC_CLAMP']

#: dist_prec.c:184-186 `if (is_crop) prcp->mu[veg] = 1.0`
MU_CROP = 1.0
#: read_soilparam.c:788 / :853: Wcr = 0.7 FC; Wpwp above max_moist -> 0.8
#: max_moist; read_soilparam.c:858-859: FC above max_moist -> 0.5 max_moist
WCR_FRACTION_OF_FC = 0.7
WPWP_CLAMP = 0.8
FC_CLAMP = 0.5

_VEG_FIELDS = ('lai', 'fcanopy', 'albedo', 'displacement', 'veg_rough',
               'rmin', 'rgl', 'rarc', 'root_frac')


def veg_override(mask=None, **fields):
    """The `veg` and `veg_mask` arguments of VicwurCore for CropSyst's daily
    values on its tiles (full_energy.c:1174-1189: LAI, albedo, displacement,
    roughness, rmin, rarc, RGL; Wdmax follows LAI as VIC forms it; the
    cover and the roots are CropSyst's too): each field a [col, tile]
    ([col, tile, layer] for root_frac) tensor or callable(t), `mask`
    [col, tile] bool, the tiles that take them."""
    bad = [k for k in fields if k not in _VEG_FIELDS]
    if bad:
        raise ValueError('veg_override: unknown field(s) %s (known: %s)'
                         % (', '.join(bad), ', '.join(_VEG_FIELDS)))
    return dict(veg=fields, veg_mask=mask)


def core(transp=None, transp_mask=None, veg=None, veg_mask=None,
         wpwp_floor=None, pet_short=False, vcs_drain=None, **kw):
    """A VicwurCore with the CropSyst options (all off by default: with none
    given it is `VicwurCore(**kw)`, VIC 5's physics). `veg`/`veg_mask` as
    `veg_override` returns them."""
    if pet_short:
        kw.setdefault('tile_fluxes', True)
    return VicwurCore(transp=transp, transp_mask=transp_mask, veg=veg,
                      veg_mask=veg_mask, wpwp_floor=wpwp_floor,
                      pet_short=pet_short, vcs_drain=vcs_drain, **kw)


def mu_crop(mu, is_crop):
    """dist_prec.c:184-186: mu = 1 on a crop tile. VIC 5 has no mu (its
    precipitation is not distributed over a wet fraction), i.e. mu is 1 on
    every tile; returns `mu` with the crop tiles at 1."""
    return torch.where(torch.as_tensor(is_crop, dtype=torch.bool,
                                       device=mu.device),
                       torch.full_like(mu, MU_CROP), mu)


def saxton_soil_params(sp, clay, quartz, depth_m, vcs_drainage=False):
    """A SoilParams with the Saxton 1986 pedotransfer over the soil file's
    hydraulics (read_soilparam.c:722-815, CLAY_INPUT): Ksat, Wcr (0.7 FC, FC
    at -33 J/kg) and Wpwp (-1500 J/kg) per layer; max_moist, the residual
    and the rest stay the file's, and Wpwp above max_moist is set to 0.8
    max_moist (:851-854). `clay`, `quartz` (fractions) and `depth_m`:
    [cell, layer] like the SoilParams' own. `sp` is not changed.

    `vcs_drainage` (default False: the file's expt): VIC-CropSyst's drainage
    law, Q12 = Ksat ((liq - resid) / (max_moist - resid))^(3 + 2 b_campbell)
    (runoff.c:583, `b` of the Saxton function), i.e. the Brooks-Corey
    exponent expt = 3 + 2 b_campbell on every layer (VIC's own is
    expt = 3 + 2 b with b read from the file); it also sets `vcs_fc_mm`, the
    layer's Field_Capacity, which `VicwurCore(vcs_drain=)` needs for the rest
    of the law (cap above Field_Capacity, none below, irrigated
    redistribution; runoff.c:575-621)."""
    from ..crop.cropsyst import ptf
    s = ptf.saxton_ptf(clay, quartz, depth_m)
    out = object.__new__(type(sp))
    for name in type(sp).__slots__:
        if hasattr(sp, name):
            setattr(out, name, getattr(sp, name))
    over = s.wpwp_mm > sp.max_moist
    if vcs_drainage:
        out.expt = (3.0 + 2.0 * s.b_campbell).to(sp.Ksat.dtype)
        # Field_Capacity (:784-786), 0.5 max_moist where it exceeds max_moist
        # (:858-859); the law's cap and floor (runoff.c:601-621)
        out.vcs_fc_mm = torch.where(s.field_capacity_mm > sp.max_moist,
                                 FC_CLAMP * sp.max_moist,
                                 s.field_capacity_mm).to(sp.Ksat.dtype)
    out.Ksat = s.ksat.to(sp.Ksat.dtype)
    out.w_crit_mm = s.wcr_mm.to(sp.Ksat.dtype)
    out.w_wilt_mm = torch.where(over, WPWP_CLAMP * sp.max_moist,
                                s.wpwp_mm).to(sp.Ksat.dtype)
    return out
