# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR CO2 forcing (C10) and the b_co2 canopy-resistance hook (C13 part).

See docs/connectors/vicwur.md.
Pure tensor kernels, batched over cells, no host sync; the forcing reader is
the C0 `PluginForcing` (value held between records, never interpolated).

C references are VIC-WUR 3.0 (<data root>/build/vicwur3/src_tree/vic):
  plugins/co2/src/co2_forcing.c:30-41         co2_force.CO2 = record (ppm)
  plugins/co2/src/co2_update_step_vars.c:14-17  Catm[i<NF] = CO2 * 1e-6,
                                              Catm[NR] = average(Catm, NF)
  drivers/image/src/vic_force.c:261-268       default Catm = CANOPY_CO2REF*1e-6
  drivers/image/src/vic_force.c:489,500       default Catm[NR] = average
  drivers/shared_all/src/forcing_utils.c:33-51  average()
  vic_run/include/vic_physical_constants.h:59   PPM_to_MIXRATIO = 1e-6
  drivers/shared_all/src/initialize_parameters.c:74  CANOPY_CO2REF = 366
  vic_run/src/penman.c:83-87                  co2factor in calc_rc
  vic_run/src/canopy_evap.c:302,414; surface_fluxes.c:793
                                              calc_rc / compute_pot_evap get
                                              Catm / PPM_to_MIXRATIO
  drivers/shared_image/src/vic_init.c:185-199 b_co2 per veg class, default 0,
                                              read only for BCO2_SRC in
                                              FROM_VEGLIB / FROM_VEGPARAM
  vic_run/src/compute_pot_evap.c:70           calc_rc with b_co2

The round trip is exact C: the forcing ppm is multiplied by 1e-6 into Catm
and divided by 1e-6 again before it reaches calc_rc, so the ppm the canopy
sees is `(CO2 * 1e-6) / 1e-6`, which is not always CO2 to the last bit.

The VIC-side hook is `torchvic.core.vegetation.calc_rc(..., co2factor=)`
(and `transpiration(..., co2factor=)`), default None = VIC 5 bit for bit.
Wiring it into the fused basin step (compute_pot_evap at
`torchvic/model/basin.py` rc_pot, the canopy evaporation call) is the
vicwur_core option of torchvic.vicwur.core; the factor is built here once
per step and tile with `tile_factor`.
"""
import torch

PPM_TO_MIXRATIO = 1.0e-6        # vic_physical_constants.h:59
CANOPY_CO2REF = 366.0           # initialize_parameters.c:74, ppm

__all__ = ['PPM_TO_MIXRATIO', 'CANOPY_CO2REF', 'average', 'catm_from_co2',
           'catm_default', 'ppm_from_catm', 'co2factor', 'tile_factor',
           'calc_rc_co2', 'Co2Forcing']


def average(ar, n=None):
    """forcing_utils.c:33-51: n == 1 returns ar[0]; else the sequential sum
    over the first n entries divided by n.  ar [..., K]; reduces the last
    axis."""
    n = ar.shape[-1] if n is None else n
    if n <= 0:
        raise ValueError('Divide by zero or negative')
    if n == 1:
        return ar[..., 0]
    s = torch.zeros_like(ar[..., 0])
    for i in range(n):
        s = s + ar[..., i]
    return s / n


def catm_from_co2(co2_ppm, nf=1):
    """Catm [C, NF + 1] (mixing ratio) from the CO2 forcing [C] (ppm):
    Catm[i < NF] = CO2 * PPM_to_MIXRATIO, Catm[NR] = average(Catm, NF)
    (co2_update_step_vars.c:14-17; NR = NF)."""
    c = co2_ppm * PPM_TO_MIXRATIO
    cols = [c for _ in range(nf)]
    body = torch.stack(cols, -1)
    return torch.cat([body, average(body, nf).unsqueeze(-1)], -1)


def catm_default(ncell, nf=1, co2ref=CANOPY_CO2REF, dtype=torch.float64,
                 device='cpu'):
    """Catm when FORCE_CO2 is off and CARBON is off: CANOPY_CO2REF *
    PPM_to_MIXRATIO in every sub-step (vic_force.c:263-267), NR slot the
    average (:500).  [C, NF + 1]."""
    v = torch.full((ncell,), float(co2ref) * PPM_TO_MIXRATIO, dtype=dtype,
                   device=device)
    body = torch.stack([v for _ in range(nf)], -1)
    return torch.cat([body, average(body, nf).unsqueeze(-1)], -1)


def ppm_from_catm(catm):
    """Catm / PPM_to_MIXRATIO: what calc_rc and compute_pot_evap receive
    (canopy_evap.c:302, surface_fluxes.c:793) and OUT_CATM (put_data.c:157)."""
    return catm / PPM_TO_MIXRATIO


def co2factor(b_co2, co2_ppm, co2ref=CANOPY_CO2REF):
    """penman.c:84: 1 / (1 + b_co2 * (co2 / CANOPY_CO2REF - 1)).

    b_co2 is the veg-library value of the tile's class (0 by default: no
    CO2 effect); no guard, as in C.  Broadcasts."""
    return 1 / (1 + b_co2 * (co2_ppm / co2ref - 1))


def tile_factor(b_co2, veg_index, catm, step_slot=0, co2ref=CANOPY_CO2REF):
    """co2factor of every [cell, tile].

    b_co2 [nveg, C] (params.read_bco2); veg_index [C, T] long, the veg class
    of each tile (a bare tile, class -1 or beyond, gets b_co2 = 0, i.e.
    factor 1); catm [C, NF + 1]; step_slot = the sub-step index hidx the
    caller is in (0 for daily).  Returns [C, T]."""
    nveg = b_co2.shape[0]
    ok = (veg_index >= 0) & (veg_index < nveg)
    ix = torch.where(ok, veg_index, torch.zeros_like(veg_index))
    b = torch.gather(b_co2.t(), 1, ix)
    b = torch.where(ok, b, torch.zeros_like(b))
    ppm = ppm_from_catm(catm[:, step_slot]).unsqueeze(1)
    return co2factor(b, ppm, co2ref)


def calc_rc_co2(rs, net_short, rgl, tair, vpd, lai, gsm_inv, b_co2, catm,
                ref_crop=False, co2ref=CANOPY_CO2REF):
    """calc_rc with the b_co2 term (penman.c:34-92): the VIC 5 kernel with
    the factor of this step's Catm [mixing ratio, broadcastable to lai]."""
    from ..core import vegetation
    f = co2factor(b_co2, ppm_from_catm(catm), co2ref)
    return vegetation.calc_rc(rs, net_short, rgl, tair, vpd, lai, gsm_inv,
                              ref_crop, co2factor=f)


class Co2Forcing(object):
    """The CO2 PLUGIN_FORCE_TYPE stream on the device.

        f = Co2Forcing(prefix, start=(2003, 7, 1, 0), nsteps=n,
                       met_start=..., device='cuda')
        catm = f.catm(step)      # [C, NF + 1], mixing ratio, held record
        f.ppm(step)              # [C], the file value

    `prefix` is the path prefix of the global file's `PLUGIN_FORCE_TYPE CO2
    co2 YEAR <prefix>` line (file <prefix><yyyy>.nc, variable `co2`).  The
    frequency is whatever the line says (YEAR in the R1 runs); the cadence,
    first-year skip and file checks are `PluginForcing`'s.  steps_per_day is
    the model steps per day; nf the sub-steps per model step (SNOW_STEPS_PER_
    DAY / MODEL_STEPS_PER_DAY, 1 for the daily vicwur profile).
    """

    def __init__(self, prefix, start, nsteps, met_start, freq='YEAR',
                 varname='co2', steps_per_day=1, nf=1, device='cpu',
                 cells=None, calendar='proleptic_gregorian', prefetch=True):
        from .forcing import PluginForcing
        self.nf = nf
        self.pf = PluginForcing(prefix, varname, freq, start, nsteps,
                                steps_per_day, device=device, cells=cells,
                                calendar=calendar, met_start=met_start,
                                prefetch=prefetch)

    @classmethod
    def from_global(cls, g, start, nsteps, met_start, steps_per_day=1, nf=1,
                    device='cpu', cells=None, **kw):
        """From a parse_global() dict: the `PLUGIN_FORCE_TYPE CO2 varname
        FREQ prefix` line (plugin_forcing_utils.c:85-87)."""
        for w in g.get('PLUGIN_FORCE_TYPE', []):
            if w[0].upper() == 'CO2':
                return cls(w[3], start, nsteps, met_start, freq=w[2],
                           varname=w[1], steps_per_day=steps_per_day, nf=nf,
                           device=device, cells=cells, **kw)
        raise ValueError('no PLUGIN_FORCE_TYPE CO2 line in the global file')

    def ppm(self, step):
        """The held CO2 record [C] (co2_forcing.c:39); reshaped to cells."""
        a = self.pf.get(step)
        return a.reshape(-1)

    def catm(self, step):
        return catm_from_co2(self.ppm(step), self.nf)

    def reads(self, step):
        return self.pf.reads(step)

    def close(self):
        self.pf.close()
