# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_irrigation: VIC-WUR irrigation (C8) as a connector.

The kernels are torchvic.vicwur.irrigation; this connector runs them in
VIC-WUR's phase order on the group's step view (C paths are VIC-WUR 3.0,
<data root>/build/vicwur3/src_tree/vic/plugins/irrigation/src):

  U   the paddy layer-0 Ksat of VIC-WUR 3.0's irr_update_step_vars
      (irr_run.c:33-71: every band, no guard) is what vic_run's drainage
      reads: it reaches VIC as the run's vicwur_core `ksat` (`core_hooks`),
      constant in time
  P1  `pass1`: irr_run_requirement (irr_run.c:107-221): the daily offset
      counter, then requirement and flag on the END-of-step moisture
  P1  `pass1_demand`: irr_set_demand (irr_use.c:62-105): D0, G, E and
      D = D0/E into the water-use connector's IRR sector (demand,
      groundwater_frac, consumption_frac), cells with an irrigated tile
  P3  `pass3_withdrawn`: irr_get_withdrawn (irr_use.c:650-687): leftover
      re-applied first, then POTENTIAL (received = requirement) or the
      water-use share (received = requirement x consumed / D0 of the IRR
      sector); applied to layer 0 through the view (the next vic_run's
      soil_add); then irr_wofost (WOFOST's IRRTAB water, the crop member's
      `wofost_water`, irr_use.c:531-644) on every active tile and band
  P3  `pass3`: irr_run_shortage (irr_run.c:260-372) on the moisture after
      pumping and irrigation: shortage, layer stress, cell stress (the crop
      component reads the cell stress of the irrigated tiles)
  O   `put_data`: OUT_REQUIREMENT, OUT_SHORTAGE, OUT_RECEIVED, OUT_APPLIED,
      OUT_LEFTOVER (irr_output.c:120-170: sum over tiles and bands of
      x Cv AreaFract)

Tiles. VIC-WUR's irrigated tiles are the irrigation classes (irr file
order) whose vegetation class has a tile in the cell (irr_con_map); TorchVIC's
tile axis is the vegetation class, so irrigation class i is tile
veg_class[i] - 1, and an absent class is masked (Cv 0 there).

Options
    params       torchvic.vicwur.params.IrrigationParams (read_irrigation),
                 or irr = dict(veg_class [NI] 1-based, paddy [NI],
                 active [N, NI], gw_fraction [N, NI], efficiency [N, NI],
                 offset [N, NI])
    soil         dict of [N, L] mm: max_moist, wcr, wfc, wpwp (the C's
                 soil_con values, vic_init.c:764-795) and root [N, NI, L]
                 (the root fractions of each irrigated tile)
    efficient    EFFICIENT_IRRIGATION (room up to Wfc, not max_moist)
    potential    POTENTIAL_IRRIGATION
    share_layer_moist   SHARE_LAYER_MOIST (the stress override)
    flag_hold    after a day the supply was short (received < requirement
                 on a flagged tile), the flag stays on while the requirement
                 is positive (default False = VIC-WUR 3.0's exact flag,
                 which pulses on alternate days when the supply is short;
                 the group builder sets it in fixed mode, off in exact
                 mode). A tile that got its full requirement keeps
                 VIC-WUR 3.0's flag (root zone below Wcr): holding it then
                 would top the root zone up to Wfc every day, which
                 VIC-WUR 3.0 does not do (+19 % India withdrawals)
    paddy_ksat   VIC-WUR 3.0's U-phase paddy Ksat (default True); needs VIC
                 attached
    area         [N] m2 for the ledger (default: the router's)
    scheme       'vic' (default: VIC-WUR's own requirement and flag) or
                 'cropsyst' (the M4 scheme: CropSyst's E12 decision
                 and table or mechanistic losses; C8's availability, offset,
                 leftover and application are unchanged)
    cropsyst     for scheme 'cropsyst', dict(crop_fn, depth, code, ...):
                 crop_fn() -> dict(root_depth_mm, wsi, active [N, NI(, B)],
                 optional et0, et0_ow, deficit_amount): the crop backend
                 'cropsyst' signals of the day, or the string 'group': the
                 group's live vicwur_cropsyst member;
                 depth [N, L] m; code the
                 irrigation technology code; the rest is
                 torchvic.vicwur.cs_irrigation.CsConfig (losses, ro_mode,
                 mad, window, observe, management, tolerance, mech)

State (checkpoint): offset, prev_cv, leftover, received, applied,
requirement, flag, shortage, layer and cell stress (VIC-WUR 3.0 saves
leftover, received, offset and prev_Cv, irr_state.c:39-87).

Ledger (m3 per day, over the P3 phase): I_ext = water received (from the
water-use IRR sector's consumption, or from outside in POTENTIAL mode);
Q_out = water applied to VIC's layer 0 (VIC books it as soil_add); dS =
change of the leftover store. Land use moving leftover to the river is in
the land-use connector's row.
"""
import numpy as np
import torch

from ..connector import Balance, ComputeError
from ..spec import CouplingError, VarSpec
from ...vicwur import irrigation as IR
from ...vicwur import cs_irrigation as CS
from ...vicwur import wateruse as WU
from .vicwur_common import GRID, WurConnector, as_f64, as_i64

_SOIL = ('max_moist', 'wcr', 'wfc', 'wpwp', 'root')
OUTS = ('requirement', 'shortage', 'received', 'applied', 'leftover')


class VicWurIrrigationConnector(WurConnector):
    name = 'vicwur_irrigation'
    version = '1'
    kind = 'irrigation'
    role = 'irrigation'

    def __init__(self, params=None, irr=None, soil=None, efficient=False,
                 potential=False, share_layer_moist=True, paddy_ksat=True,
                 area=None, flag_hold=False, scheme='vic', cropsyst=None, group=None,
                 **config):
        self.flags = dict(irrigation=True,
                          potential_irrigation=bool(potential))
        super().__init__(group=group, **config)
        if params is not None:
            irr = dict(veg_class=params.veg_class, paddy=params.paddy,
                       active=params.active, gw_fraction=params.gw_fraction,
                       efficiency=params.efficiency, offset=params.offset)
        if irr is None or soil is None:
            raise CouplingError('%s: give params (or irr) and soil'
                                % self.name)
        miss = [k for k in ('veg_class', 'paddy', 'active', 'gw_fraction',
                            'efficiency', 'offset') if k not in irr]
        miss += ['soil.' + k for k in _SOIL if k not in soil]
        if miss:
            raise CouplingError('%s: missing %s' % (self.name, miss))
        npy = lambda x: np.asarray(x.cpu() if torch.is_tensor(x) else x)
        self.vc = npy(irr['veg_class']).astype(np.int64).reshape(-1)
        ni = self.NI = len(self.vc)
        if ni == 0 or (self.vc < 1).any():
            raise CouplingError('%s: veg_class must be 1-based, non-empty'
                                % self.name)
        if len(set(self.vc.tolist())) != ni:
            raise CouplingError('%s: an irrigation class twice' % self.name)
        self.irr_np = {k: npy(irr[k]) for k in ('paddy', 'active',
                                                  'gw_fraction', 'efficiency',
                                                  'offset')}
        act = self.irr_np['active']
        if act.ndim != 2 or act.shape[1] != ni:
            raise CouplingError('%s: active must be [N, NI]' % self.name)
        self.N = act.shape[0]
        for k in ('gw_fraction', 'efficiency', 'offset'):
            if self.irr_np[k].shape != (self.N, ni):
                raise CouplingError('%s: %s must be [N, NI]' % (self.name,
                                                                 k))
        self.soil_cfg = soil
        for k in ('max_moist', 'wcr', 'wfc', 'wpwp'):
            if npy(soil[k]).ndim != 2 or npy(soil[k]).shape[0] != self.N:
                raise CouplingError('%s: soil.%s must be [N, L]'
                                    % (self.name, k))
        if npy(soil['root']).shape[:2] != (self.N, ni):
            raise CouplingError('%s: soil.root must be [N, NI, L]'
                                % self.name)
        self.efficient, self.potential = bool(efficient), bool(potential)
        self.share = bool(share_layer_moist)
        self.paddy_ksat = bool(paddy_ksat)
        self.flag_hold = bool(flag_hold)
        self.area_cfg = area
        if scheme not in ('vic', 'cropsyst'):
            raise CouplingError("%s: scheme must be 'vic' or 'cropsyst'"
                                % self.name)
        self.scheme = scheme
        self.cs_cfg = dict(cropsyst or {})
        if scheme == 'cropsyst':
            for k in ('crop_fn', 'depth', 'code'):
                if k not in self.cs_cfg:
                    raise CouplingError('%s: scheme cropsyst needs '
                                        'cropsyst.%s' % (self.name, k))
        elif cropsyst:
            raise CouplingError("%s: cropsyst= needs scheme='cropsyst'"
                                % self.name)
        self._check_config()
        self.outputs = tuple(VarSpec(k, 'mm', ('cell',), kind='state',
                                     grid=GRID, description='OUT_' + k.upper())
                             for k in OUTS)
        if scheme == 'cropsyst':
            self.outputs += (VarSpec(
                'evap', 'mm', ('cell',), kind='state', grid=GRID,
                description='evaporation of the irrigation system (Ed): '
                            'leaves as ET (canopy_evap.c:560-565, :1013)'),)
        self.owned_state = ('%s.leftover' % self.name,)
        #: set by the crop member: the layer Wcr and the veg_var
        #: roots of the irrigated tiles [N, NI, B, L] as crop_update_step_vars
        #: leaves them (callables), and irr_wofost's water [N, NI, B]
        #: (a callable of the step, or a tensor); None: the soil's Wcr, the
        #: file's roots, no WOFOST water
        self.wcr_fn = None
        self.root_fn = None
        self.wofost_water = None

    # -- set-up --------------------------------------------------------------
    def _init(self, context):
        d = self.dev
        n, ni = self.N, self.NI
        self.tix = torch.as_tensor(self.vc - 1, device=d)
        self.act = torch.as_tensor(self.irr_np['active'].astype(bool),
                                   device=d)
        pad = self.irr_np['paddy'].astype(bool).reshape(-1)
        self.paddy = torch.as_tensor(pad, device=d).expand(n, ni) \
            .contiguous()
        self.gw = as_f64(self.irr_np['gw_fraction'], d)
        self.eff = as_f64(self.irr_np['efficiency'], d)
        self.omax = as_i64(self.irr_np['offset'], d)
        s = {k: as_f64(self.soil_cfg[k], d) for k in _SOIL}
        self.mm, self.wcr, self.wfc, self.wpwp = (s['max_moist'], s['wcr'],
                                                  s['wfc'], s['wpwp'])
        self.root = s['root']
        self.L = self.mm.shape[1]
        self.cs = None
        self._months = None
        if self.scheme == 'cropsyst':
            c = dict(self.cs_cfg)
            self.crop_fn = c.pop('crop_fn')
            if self.crop_fn == 'group':
                self.crop_fn = self._group_signals
            try:
                self.cs = CS.CsConfig(n, ni, d, **c)
            except (ValueError, TypeError) as e:
                raise CouplingError('%s: %s' % (self.name, e))
        self.B = None
        self.st = None
        self._area = None
        self._linked = False

    def _group_signals(self):
        """The crop signals of the group's `vicwur_cropsyst` member: root
        depth, water stress index and activity of the day before, on the
        irrigated tiles."""
        crop = self.group.need('crop', self.name)
        if not hasattr(crop, 'signals_for'):
            raise ComputeError("%s: crop_fn 'group' needs the vicwur_"
                               'cropsyst member as the group\'s crop'
                               % self.name)
        return crop.signals_for(self.vc - 1, self.N)

    def _state0(self, B):
        """The irr_var of every entry and band (irr_init: all zero)."""
        d, n, ni, L = self.dev, self.N, self.NI, self.L
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=d)
        self.B = B
        return dict(offset=torch.zeros(n, ni, B, dtype=torch.int64,
                                       device=d),
                    prev_cv=z(n, ni, B), leftover=z(n, ni, B),
                    received=z(n, ni, B), applied=z(n, ni, B),
                    requirement=z(n, ni, B),
                    flag=torch.zeros(n, ni, B, dtype=torch.bool, device=d),
                    shortage=z(n, ni, B), ws=z(n, ni, B, L),
                    cs=z(n, ni, B))

    def _ensure(self, B):
        if self.st is None:
            self.st = self._state0(B)
        elif self.B != B:
            raise ComputeError('%s: the view has %d bands, the state %d'
                               % (self.name, B, self.B))

    def area(self):
        if self._area is None:
            if self.area_cfg is not None:
                self._area = as_f64(self.area_cfg, self.dev)
            elif self.group.router is not None:
                r = self.group.router
                self._area = r.to_cells(r.area)
            else:
                raise ComputeError('%s: no area (give area, or run with '
                                   'vicwur_routing)' % self.name)
        return self._area

    def _link(self):
        """The water-use connector's IRR sector exists in every cell when
        IRRIGATION is on (wu_get_global_param.c:151, wu_alloc_free.c:
        50-58)."""
        if self._linked:
            return
        wl = self.group.roles.get('wateruse_local')
        if wl is not None:
            wl.act[:, WU.IRR] = True
        self._linked = True

    # -- per-entry views of the step view --------------------------------------
    def _cv(self, v):
        """Cv of each irrigated tile [N, NI] (0 where the class has no tile)."""
        c = v.cv.index_select(1, self.tix).to(torch.float64)
        return torch.where(self.act, c, torch.zeros_like(c))

    def _layers(self, v):
        """moist, ice [N, NI, B, L(, 1)], frost [N, 1] of the irrigated
        tiles (ice frost-weighted: one area of weight 1)."""
        m = v.moist.index_select(1, self.tix)
        ice = (torch.zeros_like(m) if v.ice is None
               else v.ice.index_select(1, self.tix))
        ff = torch.ones(m.shape[0], 1, dtype=m.dtype, device=m.device)
        return m, ice.unsqueeze(-1), ff

    def _wcr(self, B):
        if self.wcr_fn is not None:
            return self.wcr_fn()
        return self.wcr[:, None, None, :].expand(self.N, self.NI, B, self.L)

    def _root(self, B):
        if self.root_fn is not None:
            return self.root_fn()
        return self.root[:, :, None, :].expand(self.N, self.NI, B, self.L)

    # -- hooks ---------------------------------------------------------------
    def pass1(self):
        v = self.view
        if v is None:
            raise ComputeError('%s: no step view' % self.name)
        self._link()
        B = v.af.shape[1]
        self._ensure(B)
        st = self.st
        cv = self._cv(v)
        new_day = torch.ones((), dtype=torch.bool, device=self.dev)
        st['offset'], st['prev_cv'] = IR.offset_step(
            st['offset'], st['prev_cv'], cv, self.omax, new_day)
        m, ice, ff = self._layers(v)
        if self.cs is not None:
            req, flag, self._eff_b, self._edf = self._cs_requirement(
                v, m, ice, ff, cv, st)
        else:
            req, flag = IR.requirement(
                m, ice, ff, self._root(B), self._wcr(B), self.wfc,
                self.mm[:, 0], st['leftover'], st['offset'], self.omax,
                self.paddy, cv, v.af,
                hold=self._hold(st) if self.flag_hold else None)
        st['requirement'], st['flag'] = req, flag
        self._cvday = cv

    @staticmethod
    def _hold(st):
        """The tiles whose flag holds today: flagged yesterday and short of
        yesterday's requirement (state of the previous step: requirement of
        its P1, received of its P3)."""
        req, rec = st['requirement'], st['received']
        return st['flag'] & (rec < req - 1e-9 * (1.0 + req.abs()))

    def _month(self):
        """The month of the day as a 0-d device tensor (no host sync: the
        table is built once from the case's dates)."""
        if self._months is None:
            if self.group.vic is None:
                raise ComputeError('%s: scheme cropsyst needs VIC in the '
                                   'group (the dates)' % self.name)
            dts = self.group.vic.driver.dates
            self._months = torch.as_tensor([x.month for x in dts],
                                           device=self.dev)
        return self._months[min(self.group.step, len(self._months) - 1)]

    def _cs_requirement(self, v, m, ice, ff, cv, st):
        """The M4 scheme: CropSyst's E12 decision on C8's slots,
        on the same end-of-step liquid moisture and leftover as C8's own."""
        sig = self.crop_fn()
        liq = IR.liquid_layers(m, ice, ff)
        liq = torch.cat([(liq[..., 0] + st['leftover']).unsqueeze(-1),
                         liq[..., 1:]], -1)
        open_ = st['offset'] >= self.omax.unsqueeze(-1)
        act = torch.as_tensor(sig['active'], device=self.dev) \
            .to(torch.bool) & self.act
        req, flag, E, gross, ed, _ = CS.decide(
            self.cs, liq, self.wfc, self.wpwp, sig['root_depth_mm'],
            sig['wsi'], act, self._month(), open_, sig.get('et0'),
            sig.get('et0_ow'), sig.get('deficit_amount'), losses=True)
        a = IR.active_mask(cv, v.af) & self.act.unsqueeze(-1)
        pos = gross > 0.0
        edf = torch.where(pos, ed / torch.where(pos, gross, torch.ones_like(
            gross)), torch.zeros_like(gross))
        # the evaporated share of the gross water: the water-use consumption
        # fraction is E + Ed / gross (the water that leaves the cell: what
        # the crop gets plus what evaporates), the rest returns
        self._ed_mm = torch.where(a & flag, ed, torch.zeros_like(ed))
        return (torch.where(a, req, torch.zeros_like(req)), flag & a, E,
                torch.where(a, edf, torch.zeros_like(edf)))

    def pass1_demand(self):
        """irr_set_demand into the water-use connector (sweep order)."""
        wl = self.group.roles.get('wateruse_local')
        if wl is None:
            return
        st, v = self.st, self.view
        D, G, E, _ = IR.demand(st['requirement'], st['flag'], self._cvday,
                               v.af, self.gw,
                               self.eff if self.cs is None else self._eff_b)
        if self.cs is not None:
            # scheme cropsyst: Ed leaves as evaporation, so it is consumed
            # (booked by the water-use ledger as consumption), not returned
            _, _, E2, _ = IR.demand(st['requirement'], st['flag'],
                                    self._cvday, v.af, self.gw,
                                    self._eff_b + self._edf)
            self._ed_wu = D * (E2 - E)
            E = E2
        r = self.group.router
        has = self.act.any(1)                 # ni_active > 0
        for nm, x in (('dem', D), ('gfr', G), ('cf', E)):
            cur = getattr(wl, nm).clone()
            col = cur[:, WU.IRR]
            cur[:, WU.IRR] = torch.where(r.to_sweep(has.to(torch.float64))
                                         > 0, r.to_sweep(x), col)
            setattr(wl, nm, cur)
        wl.loc.cons_frac = wl.cf

    def _available(self):
        """wu_var[IRR].consumed per cell (irr_use.c:455-456)."""
        wl = self.group.roles.get('wateruse_local')
        if wl is None:
            return None
        c = self.group.router.to_cells(wl.st.consumed[:, WU.IRR])
        if self.cs is not None:
            # what the crop gets is the consumed water less the evaporated
            ed = getattr(self, '_ed_wu', None)
            if ed is not None:
                c = torch.clamp(c - ed, min=0.0)
        return c

    def pass3_withdrawn(self):
        v, st = self.view, self.st
        if st is None:
            raise ComputeError('%s: P3 before P1' % self.name)
        B = v.af.shape[1]
        cv = self._cvday
        act = IR.active_mask(cv, v.af) & self.act.unsqueeze(-1)
        m0 = v.moist.index_select(1, self.tix)[..., 0]
        w = self._weight(cv, v.af)
        S0 = (st['leftover'] * w).sum((1, 2))
        lo_prev = st['leftover']
        avail = None if self.potential else self._available()
        wof = None if self.potential else self.wofost_water
        if callable(wof):
            # irr_wofost (irr_use.c:531-644), after irr_wateruse
            # (irr_use.c:676-681)
            wof = wof()
        m1, lo, rec, app = IR.get_withdrawn(
            m0, st['leftover'], st['requirement'], st['flag'], act,
            self.wfc[:, 0], self.mm[:, 0], self.paddy, self.efficient, cv,
            v.af, potential=self.potential, available=avail, wofost=wof)
        v.delta[..., 0].index_add_(1, self.tix, m1 - m0)
        st['leftover'], st['received'], st['applied'] = lo, rec, app
        a = self.area() / 1000.
        S1 = (lo * w).sum((1, 2))
        self.acc('rec', ((rec * w).sum((1, 2)) * a).sum())
        self.acc('app', ((app * w).sum((1, 2)) * a).sum())
        self.acc('dS', ((S1 - S0) * a).sum())
        if self.cs is not None:
            # Ed of the day, mm per cell, as the water-use connector's
            # consumption when it is in the group, else booked here
            wl = self.group.roles.get('wateruse_local')
            ed_c = self._cell_sum(self._ed_mm, cv, v.af) if wl is None \
                else getattr(self, '_ed_wu', None)
            if ed_c is None:
                ed_c = torch.zeros_like(S0)
            self._ed_cell = ed_c
            if wl is None:
                self.acc('ed', (ed_c * a).sum())
        self.acc('lo', ((S0 + S1) * a).sum())
        # the per-entry invariant received + leftover_prev = applied +
        # leftover_new (the C checks it on irr_wateruse only, irr_use.c:
        # 512-525; 1e-10 there)
        res = IR.balance_residual(rec, lo_prev, app, lo)
        self.flag(torch.where(act, res.abs(), torch.zeros_like(res))
                  > 1e-9 * (1.0 + rec.abs() + lo_prev.abs()))
        del B

    def pass3(self):
        """irr_run_shortage on the moisture after pumping and irrigation."""
        v, st = self.view, self.st
        B = v.af.shape[1]
        m, ice, ff = self._layers(v)
        sh, ws, cs = IR.shortage_stress(
            m, ice, ff, self._root(B), self._wcr(B), self.wpwp, self._cvday,
            v.af, st['flag'], self.share, water_stress_in=st['ws'])
        st['shortage'], st['ws'], st['cs'] = sh, ws, cs

    @staticmethod
    def _weight(cv, af):
        return cv.unsqueeze(-1) * af.unsqueeze(1)

    def _cell_sum(self, x, cv, af):
        """irr_put_data's sum over entries then bands, Cv > 0 and
        AreaFract > 0 only, in the C order (irr_output.c:136-170)."""
        out = torch.zeros(self.N, dtype=torch.float64, device=self.dev)
        z = torch.zeros_like(out)
        for i in range(self.NI):
            for b in range(af.shape[1]):
                ok = self.act[:, i] & (cv[:, i] > 0) & (af[:, b] > 0)
                out = out + torch.where(ok, x[:, i, b] * cv[:, i] * af[:, b],
                                        z)
        return out

    def put_data(self):
        rec = self.pop('rec')
        if isinstance(rec, float):
            return
        app, dS, lo = self.pop('app'), self.pop('dS'), self.pop('lo')
        st, v = self.st, self.view
        cv = self._cvday
        exp = {k: self._cell_sum(st[k if k != 'requirement'
                                    else 'requirement'], cv, v.af)
               for k in OUTS}
        ed = self.pop('ed') if self.cs is not None else 0.0
        if self.cs is not None:
            exp['evap'] = self._ed_cell
        # Ed (no water-use connector): water that came with the source and
        # leaves as evaporation, in the ledger as ET
        bal = Balance(I_ext=rec + ed, ET=ed, Q_out=app, dS=dS,
                      n_terms=self.N * self.NI * st['leftover'].shape[2] * 4,
                      scale=rec + app + lo + 2 * ed,
                      detail=dict(received=rec, applied=app, evaporated=ed))
        self._pending = (bal, exp, {k: x.clone().unsqueeze(0)
                                    for k, x in exp.items()})

    # -- the VIC side --------------------------------------------------------
    def core_hooks(self, core):
        """VIC-WUR 3.0's U-phase paddy Ksat (irr_run.c:33-71): layer 0 of
        every paddy tile, every band, pow(Ksat[0], 0.33); the rest VIC's
        own."""
        if not self.paddy_ksat or not bool(np.asarray(
                self.irr_np['paddy']).astype(bool).any()):
            return
        G = self.group
        if G.vic is None:
            raise ComputeError('%s: paddy_ksat needs VIC in the group'
                               % self.name)
        sp = G.vic.driver.soil()
        ks = torch.as_tensor(sp.Ksat, dtype=torch.float64)
        ncol, nt = G.layout()['n_col'], G.layout()['n_tile']
        k3 = ks.unsqueeze(1).expand(ncol, nt, ks.shape[-1]).clone()
        pt = self.tix[torch.as_tensor(np.asarray(self.irr_np['paddy'])
                                      .astype(bool).reshape(-1),
                                      device=self.tix.device)]
        pt = pt.to(k3.device)
        k3[:, pt, 0] = IR.paddy_ksat(ks[:, 0], torch.ones(
            ncol, 1, dtype=torch.bool, device=ks.device), 1)[:, :, 0] \
            .expand(ncol, pt.numel())
        core.ksat = k3

    def cell_stress(self):
        """The stress irr_run_shortage stored on the irrigated tiles,
        [N, NI, B] (the crop component reads it), or None."""
        return None if self.st is None else self.st['cs']

    # -- w3_restart ----------------------------------------------------------
    #: what VIC-WUR 3.0's state file holds (irr_state.c:206-347,
    #: irr_restore :358-457)
    W3_KEPT = ('offset', 'prev_cv', 'leftover', 'received')

    def _w3_reset(self, step):
        """A new VIC-WUR 3.0 run (initialize_irr_var,
        irr_init_library.c:34-44, all 0) with the restored offset, prev_Cv,
        leftover and received; the stored layer and cell stress are VIC's
        cell water_stress, 0 after initialize_soil
        (initialize_soil.c:62-66)."""
        if self.st is None:
            return
        fresh = self._state0(self.B)
        for k in self.W3_KEPT:
            fresh[k] = self.st[k]
        self.st = fresh

    # -- state ---------------------------------------------------------------
    def _state(self):
        return {} if self.st is None else {k: v.clone()
                                            for k, v in self.st.items()}

    def _load(self, blob):
        if not blob:
            self.st = None
            self.B = None
            return
        self.st = {k: v.to(self.dev).clone() for k, v in blob.items()}
        self.B = self.st['leftover'].shape[2]
