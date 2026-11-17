# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
#
# Day length and radiation geometry follow `astro` in PCSE (pcse/util.py,
# Copyright (c) 2004-2024 Wageningen Environmental Research, Wageningen-UR,
# Allard de Wit; EUPL-1.2), after Spitters et al. (1989) and Goudriaan and
# van Laar (1994), see torchvic/crop/NOTICE_PCSE.txt.
"""Astronomical day length and diffuse radiation, batched over cells."""
import math

import torch

RAD = math.pi / 180.0          # CPython math.radians(1.)
PI = math.pi


def astro(doy, lat, avrad):
    """Return (DAYL, DAYLP, SINLD, COSLD, DIFPP, ATMTR, DSINBE, ANGOT).

    doy: day of year (float tensor [N]); lat: latitude in degrees; avrad:
    daily global radiation in J m-2 d-1.
    """
    angle = -4.
    dec = -torch.asin(math.sin(23.45 * RAD) * torch.cos(2. * PI * (doy + 10.) / 365.))
    sc = 1370. * (1. + 0.033 * torch.cos(2. * PI * doy / 365.))
    sinld = torch.sin(RAD * lat) * torch.sin(dec)
    cosld = torch.cos(RAD * lat) * torch.cos(dec)
    aob = sinld / cosld
    inside = aob.abs() <= 1.0
    aobc = torch.where(inside, aob, torch.zeros_like(aob))
    root = torch.sqrt(1. - aobc ** 2)
    dayl_in = 12.0 * (1. + 2. * torch.asin(aobc) / PI)
    dayl_out = torch.where(aob > 1.0, torch.full_like(aob, 24.0), torch.zeros_like(aob))
    dayl = torch.where(inside, dayl_in, dayl_out)
    dsinb_in = 3600. * (dayl * sinld + 24. * cosld * root / PI)
    dsinbe_in = 3600. * (dayl * (sinld + 0.4 * (sinld ** 2 + cosld ** 2 * 0.5)) +
                         12. * cosld * (2. + 3. * 0.4 * sinld) * root / PI)
    dsinb_out = 3600. * (dayl * sinld)
    dsinbe_out = 3600. * (dayl * (sinld + 0.4 * (sinld ** 2 + cosld ** 2 * 0.5)))
    dsinb = torch.where(inside, dsinb_in, dsinb_out)
    dsinbe = torch.where(inside, dsinbe_in, dsinbe_out)

    aobcorr = (-math.sin(angle * RAD) + sinld) / cosld
    inc = aobcorr.abs() <= 1.0
    acc = torch.where(inc, aobcorr, torch.zeros_like(aobcorr))
    daylp = torch.where(
        inc, 12.0 * (1. + 2. * torch.asin(acc) / PI),
        torch.where(aobcorr > 1.0, torch.full_like(aob, 24.0), torch.zeros_like(aob)))

    angot = sc * dsinb
    pos = dayl > 0.0
    atmtr = torch.where(pos, avrad / torch.where(pos, angot, torch.ones_like(angot)),
                        torch.zeros_like(angot))
    frdif = torch.where(
        atmtr > 0.75, torch.full_like(atmtr, 0.23),
        torch.where(atmtr > 0.35, 1.33 - 1.46 * atmtr,
                    torch.where(atmtr > 0.07, 1. - 2.3 * (atmtr - 0.07) ** 2,
                                torch.ones_like(atmtr))))
    difpp = frdif * atmtr * 0.5 * sc
    return dayl, daylp, sinld, cosld, difpp, atmtr, dsinbe, angot
