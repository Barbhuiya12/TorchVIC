# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""M4: CropSyst's irrigation decision inside VIC-WUR's C8.

The decision is the E12 kernels of torchvic.crop.cropsyst.irrigation; this
module only shapes them to C8's [cell, irrigated tile, band] slots
(docs/connectors/cropsyst.md):

  season gate     crop active, and the irrigation window (a parameter)
  flag            depletion > MAD or WSI > tolerance, and a deficit > 0
  requirement     the water the soil receives: gross - Ed - Ro, with
                  gross = demand / efficiency capped at the technology's
                  capacity (Dp stays in the profile, 180524LML)
  E               requirement / gross: the demand C8 hands the water-use
                  connector is D = D0 / E (the table efficiency complement
                  returns as water-use return flow)
  losses          'table' (the default; Ed, Ro, Dp fractions of the gross
                  amount) or 'mechanistic' (E13: Ed, and Ro of Philip's
                  infiltration, `ro_mode` 'fixed' or 'exact'; a non-finite
                  Ro is masked to 0, as the kernel's docstring asks)
  deficit option  `deficit_amount` (from pattern_amount) overrides the
                  decision where it is >= 0

Pure tensor code: no host synchronisation, every branch a torch.where.
"""
import torch

from ..crop.cropsyst import irrigation as CI

LOSSES = ('table', 'mechanistic')
RO_MODES = ('fixed', 'exact')


class CsConfig(object):
    """The tensors and switches of one run (built once, on the host)."""

    def __init__(self, N, NI, device, depth, code, losses='table',
                 ro_mode='fixed', mad=None, window=None, observe=None,
                 management=1, tolerance=CI.WATER_STRESS_TOLERANCE,
                 mech=None, do_irrigate=True):
        if losses not in LOSSES:
            raise ValueError('losses must be one of %s' % (LOSSES,))
        if ro_mode not in RO_MODES:
            raise ValueError('ro_mode must be one of %s' % (RO_MODES,))
        f64 = dict(dtype=torch.float64, device=device)
        self.losses, self.ro_mode = losses, ro_mode
        self.depth = torch.as_tensor(depth, **f64)              # [N, L] m
        if self.depth.dim() != 2 or self.depth.shape[0] != N:
            raise ValueError('cropsyst.depth must be [N, L] metres')
        self.lib = CI.build_library(management, device)
        code = torch.as_tensor(code, device=device)
        if code.dim() == 0:
            code = code.expand(N, NI)
        if code.shape != (N, NI):
            raise ValueError('cropsyst.code must be a scalar or [N, NI]')
        self.code = code.to(torch.long).contiguous()
        _, found = CI.lookup(self.lib, self.code)
        if not bool(found.all()):                 # set-up only
            raise ValueError('cropsyst.code: a technology code is not in '
                             'the irrigation table')
        self.mad = None if mad is None else torch.as_tensor(mad, **f64)
        w = window or {}
        self.window = CI.SeasonWindow(*(w if isinstance(w, (tuple, list))
                                        else (w.get('first_month', 3),
                                              w.get('last_month', 10))))
        o = dict(mode=CI.FRACTION_OF_ROOT_DEPTH, const_depth_m=0.0,
                 root_fraction=1.0)
        o.update(observe or {})
        self.observe = o
        # device scalars made once: building them inside `decide` would be
        # a (pageable) host-to-device copy, a synchronisation
        self.t_mode = torch.as_tensor(o['mode'], device=device)
        self.t_const = torch.as_tensor(o['const_depth_m'], **f64)
        self.t_frac = torch.as_tensor(o['root_fraction'], **f64)
        self.t_true = torch.ones((), dtype=torch.bool, device=device)
        self.t_false = torch.zeros((), dtype=torch.bool, device=device)
        self.tol = float(tolerance)
        self.do_irrigate = bool(do_irrigate)
        self.mech = None
        if losses == 'mechanistic':
            m = dict(mech or {})
            need = ('ksat0_mm_day', 's_max', 'porosity0', 'bulk_density0')
            miss = [k for k in need if k not in m]
            if miss:
                raise ValueError('cropsyst.mech needs %s' % (miss,))
            self.mech = {k: torch.as_tensor(m[k], **f64) for k in need}
            per = m.get('perennial', False)
            self.mech['perennial'] = torch.as_tensor(per, device=device) \
                .to(torch.bool)


def _b(x, like):
    """[N] or [N, NI] -> [N, NI, B] against `like` [N, NI, B]."""
    if not torch.is_tensor(x):                 # a scalar: a fill, no copy
        return torch.full_like(like, float(x))
    x = torch.as_tensor(x, dtype=like.dtype, device=like.device)
    if x.dim() == 1:
        x = x[:, None, None]
    elif x.dim() == 2:
        x = x[:, :, None]
    return x.expand_as(like)


def decide(cfg, liq, wfc, wpwp, root_depth_mm, wsi, active, month, open_,
           et0=None, et0_ow=None, deficit_amount=None, losses=False):
    """The E12 decision on every [cell, irrigated tile, band].

    liq [N, NI, B, L] liquid layer moisture (leftover already in layer 0);
    wfc, wpwp [N, L]; root_depth_mm, wsi [N, NI(, B)]; active [N, NI] bool
    (the crop stands on the tile and an irrigation record exists); month a
    0-d tensor; open_ [N, NI, B] bool (C8's season offset); et0, et0_ow
    [N(, NI, B)] mm (mechanistic losses); deficit_amount [N, NI(, B)] mm or
    None. Returns (requirement [N, NI, B], flag [N, NI, B] bool,
    E [N, NI, B], gross [N, NI, B]); with `losses` also (Ed, Ro) [N, NI, B]
    mm (zero where the entry does not irrigate): Ed leaves as evaporation of
    the irrigation system (canopy_evap.c:536-565, booked into ET at :1013)."""
    N, NI, B, L = liq.shape
    z = torch.zeros(N, NI, B, dtype=liq.dtype, device=liq.device)
    lay = lambda x: x[:, None, None, :].expand(N, NI, B, L)
    depth, fc, wp = lay(cfg.depth), lay(wfc), lay(wpwp)
    root = _b(root_depth_mm, z)
    code = cfg.code[:, :, None].expand(N, NI, B)
    tech = CI.tech_row(cfg.lib, code)
    obs = CI.observation_depth_mm(cfg.t_mode, cfg.t_const, root, cfg.t_frac,
                                  (depth.sum(-1)) * 1000.0)
    mad = tech.e('max_allowable_depletion') if cfg.mad is None \
        else _b(cfg.mad, z)
    da = None if deficit_amount is None else _b(deficit_amount, z)
    dec = CI.need_irrigation(mad, obs, root, depth, fc, wp, liq, _b(wsi, z),
                             cfg.tol, deficit_amount_mm=da)
    season = CI.irrigation_season(
        month, active, torch.zeros(N, NI, dtype=z.dtype, device=z.device),
        cfg.t_true if cfg.do_irrigate else cfg.t_false, cfg.window)
    go = dec.irrigate & season.unsqueeze(-1).expand_as(dec.irrigate) \
        & active.unsqueeze(-1) & open_
    mech = cfg.losses == 'mechanistic'
    gross = CI.gross_irrigation(dec.demand_mm, tech, mechanistic=mech,
                                deficit_amount_mm=da)
    gross = torch.where(go, gross, z)
    if mech:
        m = cfg.mech
        mm = lambda k: _b(m[k], z)
        part = CI.mechanistic_partition(
            gross, tech, _b(0.0 if et0 is None else et0, z),
            _b(0.0 if et0_ow is None else et0_ow, z), liq[..., 0], mad,
            mm('ksat0_mm_day'), mm('s_max'), mm('porosity0'), wp[..., 0],
            fc[..., 0], mm('bulk_density0'),
            m['perennial'].expand(N, NI)[:, :, None].expand_as(z)
            if m['perennial'].dim() else m['perennial'],
            mode=cfg.ro_mode, top_thickness_mm=depth[..., 0] * 1000.0)
        ed = part.evap_mm
        ro = torch.where(torch.isfinite(part.runoff_mm), part.runoff_mm, z)
    else:
        part = CI.table_partition(gross, tech)
        ed, ro = part.evap_mm, part.runoff_mm
    req = torch.where(go, gross - ed - ro, z)
    req = torch.where(req > 0.0, req, z)
    flag = go & (req > 0.0)
    pos = gross > 0.0
    one = torch.ones_like(z)
    E = torch.where(pos, req / torch.where(pos, gross, one),
                    tech.e('table_efficiency'))
    if losses:
        zero = torch.zeros_like(z)
        return (req, flag, E, gross, torch.where(go, ed, zero),
                torch.where(go, ro, zero))
    return req, flag, E, gross
