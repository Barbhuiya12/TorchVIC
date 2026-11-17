# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
#
# PyTorch/GPU port of the WOFOST 7.2 crop model as implemented in PCSE
# (https://github.com/ajwdewit/pcse, version 6.0.13; Copyright (c) 2004-2024
# Wageningen Environmental Research, Wageningen-UR, Allard de Wit and Herman
# Berghuijs; EUPL-1.2, which lists GPL-3.0 as a compatible licence). The
# equations, constants and the order of the arithmetic follow pcse/crop/
# wofost72.py, phenology.py, partitioning.py, assimilation.py, respiration.py,
# evapotranspiration.py, leaf_dynamics.py, stem_dynamics.py, root_dynamics.py
# and storage_organ_dynamics.py. See torchvic/crop/NOTICE_PCSE.txt.
#
# References:
#   de Wit, A., Boogaard, H., Fumagalli, D., Janssen, S., Knapen, R.,
#   van Kraalingen, D., Supit, I., van der Wijngaart, R., van Diepen, K.
#   (2019) 25 years of the WOFOST cropping systems model. Agricultural
#   Systems 168, 154-167, doi:10.1016/j.agsy.2018.06.018.
#   van Diepen, C.A., Wolf, J., van Keulen, H., Rappoldt, C. (1989) WOFOST:
#   a simulation model of crop production. Soil Use and Management 5, 16-24.
#   Supit, I., Hooijer, A.A., van Diepen, C.A. (1994) System description of
#   the WOFOST 6.0 crop simulation model. CEC, Luxembourg.
#   Boogaard, H.L., et al. (2014) WOFOST Control Centre 2.1 / Technical
#   documentation. Alterra, Wageningen.
"""Batched WOFOST 7.2 crop growth on the GPU.

Every cell (field) runs the same daily recurrence; the state is one tensor
per variable over the cells. The day order is PCSE's engine order:

    start(drv)       initial states, then rates of day 0
    advance(drv)     integrate the rates of the previous day, then rates of
                     the new day (so outputs of day d are states after d
                     integrations, as PCSE's saved output)

Water. Three modes:
  'pp'        potential production: no water stress (RFTRA = 1).
  'external'  water-limited production with the transpiration reduction
              factor RFTRA supplied by the caller each day (the coupled mode:
              VIC owns soil water and ET and passes actual/potential
              transpiration; the crop never runs a soil bucket).
  'cwb'       PCSE's own classic water balance (free drainage), for
              validation against PCSE only.

Units are PCSE's: kg/ha, cm, cm/day, J/m2/day, degrees Celsius, LAI m2/m2.
"""
import math
from dataclasses import dataclass

import torch

from .astro import astro, PI
from .params import (START_SOWING, END_MATURITY, END_HARVEST, END_EARLIEST,
                     CropParams)
from .waterbalance import (ClassicWaterBalance, PotentialWaterBalance, limit,
                           DEFAULT_RD)

XGAUSS = (0.1127017, 0.5000000, 0.8872983)
WGAUSS = (0.2777778, 0.4444444, 0.2777778)

STAGE_EMERGING, STAGE_VEG, STAGE_REPRO, STAGE_MATURE = 0, 1, 2, 3
BIG = 10 ** 9


@dataclass
class Drivers:
    """Daily weather, one value per cell. PCSE units: radiation J m-2 d-1,
    rain / E0 / ES0 / ET0 in cm d-1 (E0, ES0, ET0: open water, bare soil and
    reference crop evaporation)."""
    tmin: torch.Tensor
    tmax: torch.Tensor
    irrad: torch.Tensor
    rain: torch.Tensor
    e0: torch.Tensor
    es0: torch.Tensor
    et0: torch.Tensor
    lat: torch.Tensor
    doy: torch.Tensor

    def __post_init__(self):
        self.temp = (self.tmin + self.tmax) / 2.
        self.dtemp = (self.temp + self.tmax) / 2.


def _sweaf(et0, depnr):
    sweaf = 1. / (0.76 + 1.5 * et0) - (5. - depnr) * 0.10
    safe = torch.where(depnr < 3., depnr, torch.ones_like(depnr))
    sweaf = torch.where(depnr < 3., sweaf + (et0 - 0.6) / (safe * (safe + 3.)), sweaf)
    return sweaf.clamp(0.10, 0.95)


class Wofost72:
    """Batched WOFOST 7.2. See the module docstring.

    params   CropParams (N cells)
    mode     'pp' | 'external' | 'cwb'
    max_days number of daily steps to allocate leaf-cohort storage for
    """

    def __init__(self, params: CropParams, mode='pp', max_days=400,
                 harvest_idx=None, max_duration=None):
        if mode not in ('pp', 'external', 'cwb'):
            raise ValueError(mode)
        self.p = p = params
        self.mode = mode
        self.n = n = p.n
        self.dev = p.device
        self.dt = p.dtype
        self.maxd = max_days
        self.harvest_idx = (torch.full((n,), BIG, dtype=torch.long, device=self.dev)
                            if harvest_idx is None else torch.as_tensor(harvest_idx).to(self.dev))
        self.max_duration = (torch.full((n,), BIG, dtype=torch.long, device=self.dev)
                             if max_duration is None else torch.as_tensor(max_duration).to(self.dev))
        self.soil = None
        if mode == 'cwb':
            if not p.has_soil:
                raise ValueError('mode cwb needs SOPE, KSUB, IFUNRN, SSMAX, SSI, WAV, '
                                 'NOTINF, SMLIM in the parameter sets')
            self.soil = ClassicWaterBalance(p)
        elif mode == 'pp':
            self.soil = PotentialWaterBalance(p)
        self.t = -1

    # ------------------------------------------------------------------ init
    def _z(self):
        return torch.zeros(self.n, dtype=self.dt, device=self.dev)

    def start(self, drv: Drivers, rftra=None, tra=None, evs=None):
        """Initial states (crop starts on the first day) and day-0 rates."""
        p, dt, dev, n = self.p, self.dt, self.dev, self.n
        z = self._z()
        S = self.s = {}
        sowing = p.start_type == START_SOWING
        S['stage'] = torch.where(sowing, STAGE_EMERGING, STAGE_VEG).to(torch.long)
        S['dvs'] = torch.where(sowing, torch.full_like(z, -0.1), p.DVSI)
        S['tsum'], S['tsume'] = z.clone(), z.clone()
        neg1 = torch.full((n,), -1, dtype=torch.long, device=dev)
        S['doe'] = torch.where(sowing, neg1, torch.zeros_like(neg1))
        S['doa'], S['dom'] = neg1.clone(), neg1.clone()
        S['vern'] = z.clone()
        S['isvern'] = torch.zeros(n, dtype=torch.bool, device=dev)
        S['force_vern'] = torch.zeros(n, dtype=torch.bool, device=dev)
        S['dov'] = neg1.clone()
        dvs = S['dvs']
        S['fr'], S['fl'], S['fs'], S['fo'] = (p.FRTB(dvs), p.FLTB(dvs), p.FSTB(dvs),
                                              p.FOTB(dvs))
        fr = S['fr']
        S['wrt'] = p.TDWI * fr
        S['dwrt'] = z.clone()
        S['twrt'] = S['wrt'] + S['dwrt']
        S['rd'] = p.RDI.clone()
        S['rdm'] = torch.maximum(p.RDI, torch.minimum(p.RDMCR, p.RDMSOL))
        S['wst'] = (p.TDWI * (1 - fr)) * S['fs']
        S['dwst'] = z.clone()
        S['twst'] = S['wst'] + S['dwst']
        S['sai'] = S['wst'] * p.SSATB(dvs)
        S['wso'] = (p.TDWI * (1 - fr)) * S['fo']
        S['dwso'] = z.clone()
        S['twso'] = S['wso'] + S['dwso']
        S['pai'] = S['wso'] * p.SPA
        wlv = (p.TDWI * (1 - fr)) * S['fl']
        T = self.maxd + 2
        S['lv'] = torch.zeros(n, T, dtype=dt, device=dev)
        S['sla'] = torch.zeros(n, T, dtype=dt, device=dev)
        S['lvage'] = torch.zeros(n, T, dtype=dt, device=dev)
        S['lv'][:, 0] = wlv
        S['sla'][:, 0] = p.SLATB(dvs)
        S['laiem'] = S['lv'][:, 0] * S['sla'][:, 0]
        S['lasum'] = S['laiem'].clone()
        S['laiexp'] = S['laiem'].clone()
        S['laimax'] = S['laiem'].clone()
        S['lai'] = S['lasum'] + S['sai'] + S['pai']
        S['wlv'] = wlv
        S['dwlv'] = z.clone()
        S['twlv'] = S['wlv'] + S['dwlv']
        S['tagp'] = S['twlv'] + S['twst'] + S['twso']
        S['gasst'], S['mrest'], S['ctrat'], S['cevst'] = z.clone(), z.clone(), z.clone(), z.clone()
        S['tmn'] = torch.zeros(n, 7, dtype=dt, device=dev)
        S['tmncnt'] = torch.zeros(n, dtype=torch.long, device=dev)
        self.done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.done_day = neg1.clone()
        self.t = 0
        self.r = {}
        self._rates(drv, rftra, tra, evs)

    # ----------------------------------------------------------------- rates
    def potential_transpiration(self, drv: Drivers):
        """(EVWMX, EVSMX, TRAMX) of the current canopy: PCSE Evapotranspiration
        before the soil water factor. LAI and DVS are those of the last
        integration. cm/day."""
        p, s = self.p, self.s
        kglob = 0.75 * p.KDIFTB(s['dvs'])
        et0_crop = torch.clamp(p.CFET * drv.et0, min=0.)
        ekl = torch.exp(-kglob * s['lai'])
        evwmx = drv.e0 * ekl
        evsmx = torch.clamp(drv.es0 * ekl, min=0.)
        tramx = et0_crop * (1. - ekl) * p.RFTRAMXCO2
        return evwmx, evsmx, tramx, et0_crop

    def _water_factor(self, et0_crop, sm, dsos):
        p = self.p
        swdep = _sweaf(et0_crop, p.DEPNR)
        smcr = (1. - swdep) * (p.SMFCF - p.SMW) + p.SMW
        rfws = limit(torch.zeros_like(sm), torch.ones_like(sm), (sm - p.SMW) / (smcr - p.SMW))
        rfos = torch.ones_like(sm)
        ox = (p.IAIRDU == 0) & (p.IOX == 1)
        rfosmx = limit(torch.zeros_like(sm), torch.ones_like(sm), (p.SM0 - sm) / p.CRAIRC)
        rfos_ox = rfosmx + (1. - torch.clamp(dsos, max=4).to(sm.dtype) / 4.) * (1. - rfosmx)
        rfos = torch.where(ox, rfos_ox, rfos)
        return rfws, rfos

    def _rates(self, drv, rftra=None, tra=None, evs=None):
        p, s, dt, dev = self.p, self.s, self.dt, self.dev
        r = self.r = {}
        z = self._z()
        one = torch.ones_like(z)
        stage = s['stage']
        dvs = s['dvs']
        dayl, daylp, sinld, cosld, difpp, atmtr, dsinbe, angot = astro(drv.doy, drv.lat, drv.irrad)

        # ---------------------------------------------------- phenology rates
        dvred = torch.where(p.IDSL >= 1,
                            limit(z, one, (daylp - p.DLC) / torch.where(p.IDSL >= 1, p.DLO - p.DLC, one)),
                            one)
        veg = stage == STAGE_VEG
        vern_on = (p.IDSL >= 2) & veg
        notv = ~s['isvern']
        before = dvs < p.VERNDVS
        vernr = torch.where(vern_on & notv & before, p.VERNRTB(drv.temp), z)
        rvern = (s['vern'] - p.VERNBASE) / torch.where(vern_on, p.VERNSAT - p.VERNBASE, one)
        vernfac = torch.where(vern_on & notv & before, limit(z, one, rvern), one)
        r['vernr'] = vernr
        r['force_new'] = vern_on & notv & ~before
        vernfac = torch.where(vern_on, vernfac, one)

        dtsume = torch.where(stage == STAGE_EMERGING,
                             limit(z, p.TEFFMX - p.TBASEM, drv.temp - p.TBASEM), z)
        dtsmtb = p.DTSMTB(drv.temp)
        dtsum = torch.where(stage == STAGE_VEG, dtsmtb * vernfac * dvred,
                            torch.where(stage == STAGE_REPRO, dtsmtb, z))
        dvr = torch.where(stage == STAGE_EMERGING, 0.1 * dtsume / p.TSUMEM,
                          torch.where(stage == STAGE_VEG, dtsum / p.TSUM1,
                                      torch.where(stage == STAGE_REPRO, dtsum / p.TSUM2, z)))
        r['dtsume'], r['dtsum'], r['dvr'] = dtsume, dtsum, dvr

        grow = (stage != STAGE_EMERGING) & ~self.done
        r['grow'] = grow

        # --------------------------------------------------------- assimilation
        tmn = torch.cat([drv.tmin[:, None], s['tmn'][:, :6]], 1)
        s['tmn'] = torch.where(grow[:, None], tmn, s['tmn'])
        s['tmncnt'] = torch.where(grow, torch.clamp(s['tmncnt'] + 1, max=7), s['tmncnt'])
        tot = s['tmn'][:, 0]
        for j in range(1, 7):
            tot = tot + torch.where(s['tmncnt'] > j, s['tmn'][:, j], z)
        tminra = tot / torch.clamp(s['tmncnt'], min=1).to(dt)

        lai = s['lai']
        amax = p.AMAXTB(dvs)
        amax = amax * p.TMPFTB(drv.dtemp)
        kdif = p.KDIFTB(dvs)
        eff = p.EFFTB(drv.dtemp)
        dtga = self._totass7(dayl, amax, eff, lai, kdif, drv.irrad, difpp, dsinbe, sinld, cosld)
        dtga = dtga * p.TMNFTB(tminra)
        pgass = dtga * 30. / 44.

        # ------------------------------------------------ evapotranspiration
        evwmx, evsmx, tramx, et0_crop = self.potential_transpiration(drv)
        r['evwmx'], r['evsmx'], r['tramx'] = evwmx, evsmx, tramx
        if self.mode == 'pp':
            rftra_ = one
        elif self.mode == 'external':
            if rftra is None:
                raise ValueError("mode 'external' needs rftra every day")
            rftra_ = rftra
        else:
            rfws, rfos = self._water_factor(et0_crop, self.soil.sm, self.soil.dsos)
            rftra_ = rfos * rfws
        r['rftra'] = rftra_
        tra_ = tramx * rftra_ if tra is None else tra
        r['tra'] = tra_
        r['evs_ext'] = z if evs is None else evs

        gass = pgass * rftra_
        rmres = (p.RMR * s['wrt'] + p.RML * s['wlv'] + p.RMS * s['wst'] + p.RMO * s['wso'])
        rmres = rmres * p.RFSETB(dvs)
        teff = torch.pow(p.Q10, (drv.temp - 25.) / 10.)
        pmres = rmres * teff
        mres = torch.minimum(gass, pmres)
        asrc = gass - mres
        fr, fl, fs, fo = s['fr'], s['fl'], s['fs'], s['fo']
        cvf = 1. / ((fl / p.CVL + fs / p.CVS + fo / p.CVO) * (1. - fr) + fr / p.CVR)
        dmi = cvf * asrc
        admi = (1. - fr) * dmi
        r['gass'], r['mres'], r['dmi'], r['admi'] = gass, mres, dmi, admi

        # ---------------------------------------------------------------- roots
        grrt = fr * dmi
        drrt = s['wrt'] * p.RDRRTB(dvs)
        rr = torch.minimum(s['rdm'] - s['rd'], p.RRI)
        rr = torch.where(fr == 0., z, rr)
        r['gwrt'], r['drrt'], r['rr'] = grrt - drrt, drrt, rr
        # ----------------------------------------------------------------- stem
        grst = admi * fs
        drst = p.RDRSTB(dvs) * s['wst']
        r['gwst'], r['drst'] = grst - drst - 0.0, drst
        # ------------------------------------------------------ storage organs
        grso = admi * fo
        drso = z
        r['gwso'], r['drso'] = grso - drso + 0.0, drso
        # --------------------------------------------------------------- leaves
        grlv = admi * fl
        dslv1 = s['wlv'] * (1. - rftra_) * p.PERDL
        laicr = 3.2 / kdif
        dslv2 = s['wlv'] * limit(z, 0.03 * one, 0.03 * (lai - laicr) / laicr)
        dslv = torch.maximum(torch.maximum(dslv1, dslv2), z)
        nsl = self.t + 1
        old = s['lvage'][:, :nsl] > p.SPAN[:, None]
        dalv = torch.where(old, s['lv'][:, :nsl], 0.).sum(1)
        drlv = torch.maximum(dslv, dalv)
        fysage = torch.clamp((drv.temp - p.TBASE) / (35. - p.TBASE), min=0.)
        slat = p.SLATB(dvs)
        exp_on = s['laiexp'] < 6.
        dteff = torch.clamp(drv.temp - p.TBASE, min=0.)
        glaiex = torch.where(exp_on, s['laiexp'] * p.RGRLAI * dteff, z)
        glasol = grlv * slat
        gla = torch.minimum(glaiex, glasol)
        slat = torch.where(exp_on & (grlv > 0.), gla / torch.where(grlv > 0., grlv, one), slat)
        r['grlv'], r['drlv'], r['fysage'], r['slat'], r['glaiex'] = grlv, drlv, fysage, slat, glaiex

        # ---------------------------------------------------------------- soil
        if self.mode == 'cwb':
            tra_present = (stage != STAGE_EMERGING)
            self.soil.calc_rates(drv, tra_present, tra_, evwmx, evsmx, s['rd'])
            r['evs'] = self.soil.evs
        elif self.mode == 'pp':
            tra_present = (stage != STAGE_EMERGING)
            self.soil.calc_rates(drv, tra_present, tra_, evsmx)
            r['evs'] = self.soil.evs
        else:
            r['evs'] = r['evs_ext']

        # ------------------------------------------------------ crop finishing
        t = self.t
        fin = ((stage == STAGE_MATURE) & ((p.end_type == END_MATURITY) | (p.end_type == END_EARLIEST)))
        fin = fin | (((p.end_type == END_HARVEST) | (p.end_type == END_EARLIEST)) & (self.harvest_idx == t))
        fin = fin | (self.max_duration == t)
        newly = fin & ~self.done
        self.done_day = torch.where(newly, torch.full_like(self.done_day, t), self.done_day)
        r['newly_done'] = newly
        self.done = self.done | newly

    def _totass7(self, dayl, amax, eff, lai, kdif, avrad, difpp, dsinbe, sinld, cosld):
        valid = (amax > 0.) & (lai > 0.) & (dayl > 0.)
        one = torch.ones_like(dayl)
        dtga = torch.zeros_like(dayl)
        dsinbe_s = torch.where(valid, dsinbe, one)
        amax_s = torch.where(valid, amax, one)
        lai_s = torch.where(valid, lai, one)
        for i in range(3):
            hour = 12.0 + 0.5 * dayl * XGAUSS[i]
            sinb = torch.clamp(sinld + cosld * torch.cos(2. * PI * (hour + 12.) / 24.), min=0.)
            par = 0.5 * avrad * sinb * (1. + 0.4 * sinb) / dsinbe_s
            pardif = torch.minimum(par, sinb * difpp)
            pardir = par - pardif
            sinb_s = torch.where(sinb > 0., sinb, one)
            fgros = self._assim7(amax_s, eff, lai_s, kdif, sinb_s, pardir, pardif)
            dtga = dtga + fgros * WGAUSS[i]
        dtga = dtga * dayl
        return torch.where(valid, dtga, torch.zeros_like(dtga))

    @staticmethod
    def _assim7(amax, eff, lai, kdif, sinb, pardir, pardif):
        scv = 0.2
        refh = (1. - math.sqrt(1. - scv)) / (1. + math.sqrt(1. - scv))
        refs = refh * 2. / (1. + 1.6 * sinb)
        kdirbl = (0.5 / sinb) * kdif / (0.8 * math.sqrt(1. - scv))
        kdirt = kdirbl * math.sqrt(1. - scv)
        fgros = torch.zeros_like(lai)
        amx2 = torch.clamp(amax, min=2.0)
        for i in range(3):
            laic = lai * XGAUSS[i]
            visdf = (1. - refs) * pardif * kdif * torch.exp(-kdif * laic)
            vist = (1. - refs) * pardir * kdirt * torch.exp(-kdirt * laic)
            visd = (1. - scv) * pardir * kdirbl * torch.exp(-kdirbl * laic)
            visshd = visdf + vist - visd
            fgrsh = amax * (1. - torch.exp(-visshd * eff / amx2))
            vispp = (1. - scv) * pardir / sinb
            pos = vispp > 0.
            vispp_s = torch.where(pos, vispp, torch.ones_like(vispp))
            fgrsun = amax * (1. - (amax - fgrsh) * (1. - torch.exp(-vispp_s * eff / amx2))
                             / (eff * vispp_s))
            fgrsun = torch.where(pos, fgrsun, fgrsh)
            fslla = torch.exp(-kdirbl * laic)
            fgl = fslla * fgrsun + (1. - fslla) * fgrsh
            fgros = fgros + fgl * WGAUSS[i]
        return fgros * lai

    # ------------------------------------------------------------- integrate
    def integrate(self):
        """Advance the states by one day with the rates of the last call."""
        p, s, r, dt = self.p, self.s, self.r, self.dt
        live = ~self.done
        stage0 = s['stage']
        t = self.t
        z = self._z()
        # ---- phenology (all live cells; emerging cells stop after this)
        veg = stage0 == STAGE_VEG
        vern_run = (p.IDSL >= 2) & veg & live
        vern = torch.where(vern_run, s['vern'] + r['vernr'], s['vern'])
        force = s['force_vern'] | (r['force_new'] & live)
        reached = vern >= p.VERNSAT
        isv = torch.where(reached, torch.ones_like(s['isvern']), force)
        s['isvern'] = torch.where(vern_run, isv, s['isvern'])
        dov = torch.where(vern_run & reached & (s['dov'] < 0), torch.full_like(s['dov'], t + 1), s['dov'])
        s['dov'] = dov
        s['vern'] = vern
        s['force_vern'] = force
        s['tsume'] = torch.where(live, s['tsume'] + r['dtsume'], s['tsume'])
        dvs = torch.where(live, s['dvs'] + r['dvr'], s['dvs'])
        s['tsum'] = torch.where(live, s['tsum'] + r['dtsum'], s['tsum'])
        newday = t + 1
        em2veg = live & (stage0 == STAGE_EMERGING) & (dvs >= 0.0)
        veg2rep = live & (stage0 == STAGE_VEG) & (dvs >= 1.0)
        rep2mat = live & (stage0 == STAGE_REPRO) & (dvs >= p.DVSEND)
        dvs = torch.where(em2veg, z, dvs)
        dvs = torch.where(veg2rep, torch.ones_like(dvs), dvs)
        dvs = torch.where(rep2mat, p.DVSEND, dvs)
        stage = stage0 + (em2veg | veg2rep | rep2mat).to(torch.long)
        nd = torch.full_like(s['doe'], newday)
        s['doe'] = torch.where(em2veg, nd, s['doe'])
        s['doa'] = torch.where(veg2rep, nd, s['doa'])
        s['dom'] = torch.where(rep2mat, nd, s['dom'])
        s['stage'], s['dvs'] = stage, dvs

        act = live & (stage0 != STAGE_EMERGING)

        def upd(name, new):
            s[name] = torch.where(act, new, s[name])

        # ---- partitioning (new DVS)
        upd('fr', p.FRTB(dvs)); upd('fl', p.FLTB(dvs)); upd('fs', p.FSTB(dvs)); upd('fo', p.FOTB(dvs))
        # ---- roots
        wrt = s['wrt'] + r['gwrt']
        dwrt = s['dwrt'] + r['drrt']
        upd('wrt', wrt); upd('dwrt', dwrt); upd('twrt', wrt + dwrt)
        upd('rd', s['rd'] + r['rr'])
        # ---- storage organs
        wso = s['wso'] + r['gwso']
        dwso = s['dwso'] + r['drso']
        upd('wso', wso); upd('dwso', dwso); upd('twso', wso + dwso)
        upd('pai', wso * p.SPA)
        # ---- stem
        wst = s['wst'] + r['gwst']
        dwst = s['dwst'] + r['drst']
        upd('wst', wst); upd('dwst', dwst); upd('twst', wst + dwst)
        upd('sai', wst * p.SSATB(dvs))
        # ---- leaves
        nsl = t + 1
        lv = s['lv'][:, :nsl]
        cum = torch.cumsum(lv, 1)
        lv_new = torch.minimum(lv, torch.clamp(cum - r['drlv'][:, None], min=0.))
        s['lv'][:, :nsl] = torch.where(act[:, None], lv_new, lv)
        age = s['lvage'][:, :nsl]
        s['lvage'][:, :nsl] = torch.where(act[:, None], age + r['fysage'][:, None], age)
        s['lv'][:, nsl] = torch.where(act, r['grlv'], z)
        s['sla'][:, nsl] = torch.where(act, r['slat'], z)
        s['lvage'][:, nsl] = 0.
        lvn = s['lv'][:, :nsl + 1]
        lasum = (lvn * s['sla'][:, :nsl + 1]).sum(1)
        upd('lasum', lasum)
        lai = s['lasum'] + s['sai'] + s['pai']
        upd('lai', lai)
        upd('laimax', torch.maximum(lai, s['laimax']))
        upd('laiexp', s['laiexp'] + r['glaiex'])
        wlv = lvn.sum(1)
        upd('wlv', wlv)
        dwlv = s['dwlv'] + r['drlv']
        upd('dwlv', dwlv)
        upd('twlv', wlv + dwlv)
        # ---- crop totals
        upd('tagp', s['twlv'] + s['twst'] + s['twso'])
        upd('gasst', s['gasst'] + r['gass'])
        upd('mrest', s['mrest'] + r['mres'])
        upd('ctrat', s['ctrat'] + r['tra'])
        upd('cevst', s['cevst'] + r['evs'])
        # ---- soil
        if self.soil is not None:
            self.soil.integrate(s['rd'], live)
        self.t += 1

    def advance(self, drv: Drivers, rftra=None, tra=None, evs=None):
        self.integrate()
        self._rates(drv, rftra, tra, evs)

    # --------------------------------------------------------------- outputs
    def output(self):
        """Current (end-of-integration) state of the crop, per cell."""
        s, r = self.s, self.r
        hi = torch.where(s['tagp'] > 0, s['twso'] / torch.where(s['tagp'] > 0, s['tagp'], torch.ones_like(s['tagp'])),
                         torch.full_like(s['tagp'], -1.))
        out = {k: s[k] for k in ('dvs', 'lai', 'tagp', 'twso', 'twlv', 'twst', 'twrt', 'rd',
                                 'laimax', 'ctrat', 'cevst', 'gasst', 'mrest', 'wlv', 'wst',
                                 'wso', 'wrt', 'stage')}
        out['hi'] = hi
        # rates of an emerging crop are zero in PCSE (its evapotranspiration is not run)
        em = s['stage'] == STAGE_EMERGING
        zero = torch.zeros_like(r['tra'])
        out['tra'] = torch.where(em, zero, r['tra'])
        out['rftra'] = torch.where(em, zero, r['rftra'])
        out['tramx'] = torch.where(em, zero, r['tramx'])
        if self.soil is not None:
            out['sm'] = self.soil.sm
        return out

    def vic_view(self):
        """What VIC reads from the crop each day (one-day lag): LAI, root
        depth (cm) and development stage; and the crop's potential
        transpiration from the current canopy."""
        return {'lai': self.s['lai'], 'rd': self.s['rd'], 'dvs': self.s['dvs'],
                'stage': self.s['stage'], 'done': self.done}
