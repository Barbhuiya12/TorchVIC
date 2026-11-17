# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
#
# Ported from PCSE (pcse/soil/classic_waterbalance.py, Copyright (c) 2004-2024
# Wageningen Environmental Research, Wageningen-UR, Allard de Wit; EUPL-1.2),
# which follows the WOFOST 7.2 water balance (van Diepen et al., 1989;
# Supit et al., 1994). See torchvic/crop/NOTICE_PCSE.txt.
"""WOFOST's own classic water balances, for validation against PCSE only.

In a coupled run VIC owns soil water and evapotranspiration and these are not
used: the crop engine takes the transpiration reduction factor from VIC.
`PotentialWaterBalance` is PCSE's WaterbalancePP (soil kept at field capacity)
and `ClassicWaterBalance` is WaterbalanceFD (free drainage), both batched
over cells. Units are PCSE's: cm and cm/day.
"""
import math

import torch

DEFAULT_RD = 10.0


def limit(lo, hi, v):
    return torch.minimum(torch.maximum(v, lo), hi)


def _t(x, like):
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


class PotentialWaterBalance:
    """PCSE WaterbalancePP: SM = SMFCF; tracks soil evaporation only."""

    def __init__(self, p):
        self.p = p
        z = torch.zeros_like(p.SMFCF)
        self.sm = p.SMFCF.clone()
        self.dsos = torch.zeros_like(p.SMFCF, dtype=torch.long)
        self.dslr = torch.ones_like(z)
        self.rainold = z.clone()
        self.evst = z.clone()
        self.wtrat = z.clone()
        self.evs = z.clone()
        self.wtra = z.clone()

    def calc_rates(self, drv, tra_present, tra, evsmx_crop):
        self.wtra = torch.where(tra_present, tra, torch.zeros_like(tra))
        evsmx = torch.where(tra_present, evsmx_crop, drv.es0)
        wet = self.rainold >= 1
        dslr_dry = self.dslr + 1
        evsmxt = evsmx * (torch.sqrt(dslr_dry) - torch.sqrt(dslr_dry - 1))
        evs_dry = torch.minimum(evsmx, evsmxt + self.rainold)
        self.evs = torch.where(wet, evsmx, evs_dry)
        self.dslr = torch.where(wet, torch.ones_like(self.dslr), dslr_dry)
        self.rainold = drv.rain

    def integrate(self, rd, active):
        self.sm = self.p.SMFCF.clone()
        self.evst = torch.where(active, self.evst + self.evs, self.evst)
        self.wtrat = torch.where(active, self.wtrat + self.wtra, self.wtrat)


class ClassicWaterBalance:
    """PCSE WaterbalanceFD (freely draining, no groundwater, no irrigation
    unless `rirr` is passed to calc_rates)."""

    def __init__(self, p):
        self.p = p
        like = p.SMFCF
        z = torch.zeros_like(like)
        smlim = limit(p.SMW, p.SM0, p.SMLIM)
        rd = torch.full_like(like, DEFAULT_RD)
        rdm = torch.maximum(rd, p.RDMSOL)
        self.rdold = rd.clone()
        self.rdm = rdm
        self.ss = p.SSI.clone()
        self.sm = limit(p.SMW, smlim, p.SMW + p.WAV / rd)
        self.w = self.sm * rd
        self.wi = self.w.clone()
        self.wlow = limit(z, p.SM0 * (rdm - rd), (p.WAV + rdm * p.SMW - self.w))
        self.wlowi = self.wlow.clone()
        self.wwlow = self.w + self.wlow
        self.dslr = torch.where(self.sm >= (p.SMW + 0.5 * (p.SMFCF - p.SMW)),
                                torch.ones_like(like), torch.full_like(like, 5.))
        self.rinold = z.clone()
        self.dsos = torch.zeros_like(like, dtype=torch.long)
        for k in ('wtrat', 'evst', 'evwt', 'tsr', 'raint', 'wdrt', 'totinf', 'totirr',
                  'perct', 'losst'):
            setattr(self, k, z.clone())
        self.ssi = p.SSI.clone()
        self.evs = z.clone()
        self.evw = z.clone()
        self.wtra = z.clone()
        self._rates = None
        # NINFTB = Afgen([0,0, 0.5,0, 1.5,1])
        self._pending = None

    @staticmethod
    def _ninf(rain):
        # Afgen([0.0,0.0, 0.5,0.0, 1.5,1.0])
        slope = (1.0 - 0.0) / (1.5 - 0.5)
        mid = 0.0 + slope * (rain - 0.5)
        return torch.where(rain >= 1.5, torch.ones_like(rain),
                           torch.where(rain <= 0.0, torch.zeros_like(rain),
                                       torch.where(rain <= 0.5, torch.zeros_like(rain), mid)))

    def calc_rates(self, drv, tra_present, tra, evwmx_crop, evsmx_crop, rd, rirr=None):
        s, p = self, self.p
        z = torch.zeros_like(p.SMFCF)
        rirr = z if rirr is None else rirr
        wtra = torch.where(tra_present, tra, z)
        evwmx = torch.where(tra_present, evwmx_crop, drv.e0)
        evsmx = torch.where(tra_present, evsmx_crop, drv.es0)

        flooded = s.ss > 1.
        evw = torch.where(flooded, evwmx, z)
        wet = s.rinold >= 1
        dslr_new = s.dslr + 1.
        evsmxt = evsmx * (torch.sqrt(s.dslr + 1) - torch.sqrt(s.dslr))
        evs_dry = torch.minimum(evsmx, evsmxt + s.rinold)
        evs = torch.where(flooded, z, torch.where(wet, evsmx, evs_dry))
        dslr = torch.where(flooded, s.dslr,
                           torch.where(wet, torch.ones_like(s.dslr), dslr_new))

        rinpre0 = torch.where(p.IFUNRN == 0, (1. - p.NOTINF) * drv.rain,
                              (1. - p.NOTINF * self._ninf(drv.rain)) * drv.rain)
        rinpre = rinpre0 + rirr + s.ss
        avail = rinpre - evw
        rinpre = torch.where(s.ss > 0.1, torch.minimum(p.SOPE, avail), rinpre)

        we = p.SMFCF * rd
        perc1 = limit(z, p.SOPE, (s.w - we) - wtra - evs)
        welow = p.SMFCF * (s.rdm - rd)
        loss = limit(z, p.KSUB, (s.wlow - welow + perc1))
        perc2 = ((s.rdm - rd) * p.SM0 - s.wlow) + loss
        perc = torch.minimum(perc1, perc2)
        rin = torch.minimum(rinpre, (p.SM0 - s.sm) * rd + wtra + evs + perc)
        dw = rin - wtra - evs - perc
        dwlow = perc - loss
        wtmp = s.w + dw
        neg = wtmp < 0.0
        evs = torch.where(neg, evs + wtmp, evs)
        dw = torch.where(neg, -s.w, dw)
        sstmp = drv.rain + rirr - evw - rin
        dss = torch.minimum(sstmp, (p.SSMAX - s.ss))
        dtsr = sstmp - dss
        self._rates = dict(evs=evs, evw=evw, wtra=wtra, rin=rin, rirr=rirr, perc=perc,
                           loss=loss, dw=dw, dwlow=dwlow, dtsr=dtsr, dss=dss,
                           draint=drv.rain, dslr=dslr)
        self.evs, self.evw, self.wtra = evs, evw, wtra

    def integrate(self, rd, active):
        s, p, r = self, self.p, self._rates

        def acc(old, inc):
            return torch.where(active, old + inc, old)

        s.dslr = torch.where(active, r['dslr'], s.dslr)
        s.rinold = torch.where(active, r['rin'], s.rinold)
        s.wtrat = acc(s.wtrat, r['wtra'])
        s.evwt = acc(s.evwt, r['evw'])
        s.evst = acc(s.evst, r['evs'])
        s.raint = acc(s.raint, r['draint'])
        s.totinf = acc(s.totinf, r['rin'])
        s.totirr = acc(s.totirr, r['rirr'])
        ss = s.ss + r['dss']
        tsr = s.tsr + r['dtsr']
        w = s.w + r['dw']
        perct = s.perct + r['perc']
        losst = s.losst + r['loss']
        wlow = s.wlow + r['dwlow']
        # _redistribute_water
        rdchange = rd - s.rdold
        grow = rdchange > 0.001
        wdr_g = torch.minimum(wlow, wlow * rdchange / (p.RDMSOL - s.rdold))
        wdr = torch.where(grow, wdr_g, w * rdchange / s.rdold)
        nz = wdr.abs() > 0.
        wlow = torch.where(nz, wlow - wdr, wlow)
        w = torch.where(nz, w + wdr, w)
        wdrt = torch.where(nz, s.wdrt + wdr, s.wdrt)
        sm = w / rd
        dsos = torch.where(sm >= (p.SM0 - p.CRAIRC), s.dsos + 1, torch.zeros_like(s.dsos))
        s.ss = torch.where(active, ss, s.ss)
        s.tsr = torch.where(active, tsr, s.tsr)
        s.w = torch.where(active, w, s.w)
        s.perct = torch.where(active, perct, s.perct)
        s.losst = torch.where(active, losst, s.losst)
        s.wlow = torch.where(active, wlow, s.wlow)
        s.wwlow = torch.where(active, s.w + s.wlow, s.wwlow)
        s.wdrt = torch.where(active, wdrt, s.wdrt)
        s.sm = torch.where(active, sm, s.sm)
        s.dsos = torch.where(active, dsos, s.dsos)
        s.rdold = torch.where(active, rd, s.rdold)

    def balance_errors(self):
        """(root zone, whole profile) closure, PCSE's WBALRT and WBALTT."""
        s = self
        wbalrt = s.totinf + s.wi + s.wdrt - s.evst - s.wtrat - s.perct - s.w
        wbaltt = (s.ssi + s.raint + s.totirr + s.wi - s.w + s.wlowi - s.wlow - s.wtrat
                  - s.evwt - s.evst - s.tsr - s.losst - s.ss)
        return wbalrt, wbaltt
