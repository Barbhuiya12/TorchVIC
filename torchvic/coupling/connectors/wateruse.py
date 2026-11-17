# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""WaterUseConnector: irrigation and human water use on top of VIC.

The engine is torchvic.wateruse.WaterUseEngine (equations of VIC-WUR,
Droppers et al. 2020). The connector adds nothing to VIC's physics; water
reaches VIC only through its `soil_add` injection point (layer 0, capped,
VIC-WUR irr_use.c:238-265) and what VIC cannot take comes back through
`soil_unmet`.

Per exchange interval (one day for the daily engine):

  1. demand      from VIC's root-zone soil moisture at the interval end
                 (input `soil_moisture`, cell mean) against the critical
                 point and field capacity (irr_run.c:129-177), inside the
                 growing season (calendar, or the crop connector's
                 `season`), plus the other sectors' demands
  2. allocation  by source and sector priority: groundwater, reservoir,
                 river (wu_run_local.c:153-250), never above what each
                 source offers; an unmet part may fall back to the other
                 source
  3. delivery    the consumed irrigation water into VIC layer 0 (output
                 `soil_add`, mm over the cell, delivered to VIC at the next
                 interval, as VIC-WUR adds after vic_run); what the layer
                 cannot hold stays in a leftover store; what VIC reports it
                 could not add (`soil_unmet`) joins that store
  4. returns     conveyance and field losses (and non-irrigation return
                 flows) go back to the river; consumption of the other
                 sectors leaves as evapotranspiration

Sources are services with `available()` and `withdraw()` (and `inject()`
for the river), as in the engine's `StubSource`.

  Open item (needs an integration test): the river and the reservoir are
  the stubs below (`StubService`); they are not wired to the routing and
  reservoir connectors (docs/connectors/irrigation.md, "Sources"). To wire them,
  `river_available` / `river_withdrawal` / `river_return` (declared, unused
  here) would connect to the routing connector's `discharge`, `withdrawal_request`, `inflow`
  (docs/CONNECTOR_API.md, "A stepping connector"), and a test of VIC +
  routing + irrigation must be added next to tests/test_conn_agri.py.

Water balance (docs/WATER_ACCOUNTING.md), m3 over the domain:
    I_ext   water withdrawn from the sources + VIC's returned unmet
    Q_out   soil_add delivered + returns to the river (+ unrouted returns)
    ET      consumption of the other sectors
    dS      change of the leftover store
The engine's own cumulative identities (demand = withdrawn + unmet, ...)
are re-checked in `validate`.
"""
import numpy as np
import torch

from ..connector import (Balance, ComputeError, Connector, DeviceFlag,
                         Proposal)
from ..spec import CouplingError, GridSpec, VarSpec
from ...wateruse import WaterUseConfig, WaterUseEngine
from . import common

GRID = 'vic_cells'
BIG = 1.0e30


class StubService(object):
    """A source with a store (m3 per element) the connector owns. It does
    not mutate: `withdraw` returns what is delivered and remembers the
    request; the connector applies it at commit. `refill` is added to the
    store every interval, `inject` returns water to it."""

    def __init__(self, store):
        self.store = store
        self.taken = None
        self.injected = None

    def available(self):
        return self.store

    def withdraw(self, request):
        d = torch.minimum(request.clamp(min=0), self.store)
        self.taken = d if self.taken is None else self.taken + d
        return d

    def inject(self, volume):
        self.injected = volume if self.injected is None else \
            self.injected + volume

    def new_store(self):
        s = self.store
        if self.taken is not None:
            s = s - self.taken
        if self.injected is not None:
            s = s + self.injected
        return s


class _RiverView(object):
    """The engine's `router` interface over a StubService per site."""

    def __init__(self, svc):
        self.svc = svc

    def available(self):
        return self.svc.available()

    def withdraw(self, request):
        return self.svc.withdraw(request)

    def inject(self, volume):
        self.svc.inject(volume)


class WaterUseConnector(Connector):
    name = 'irrigation'
    version = '1'
    backends = ('cpu', 'cuda')
    graph_safe = False
    checkpoint = 'full'
    validation = ('engine: VIC-WUR equations (tests/test_wateruse.py); '
                  'connector: ledger closure, availability bounds, restart '
                  'equality (tests/test_conn_agri.py). River/reservoir/'
                  'groundwater are stubs until the routing connectors are '
                  'merged')
    imported_fluxes = ('soil_unmet',)
    exported_fluxes = ('soil_add',)
    owned_state = ('leftover', 'wu_ledger', 'river_store', 'gw_store',
                   'res_store')

    def __init__(self, driver=None, vic=None, global_param=None,
                 irrigated_tiles=None, irrigated_fraction=None,
                 paddy_tiles=None, wcr_fract=None, wfc_fract=None,
                 efficiency=0.7, conveyance_loss=0.1, gw_fraction=0.0,
                 allocation='priority', loss_return='river',
                 fallback=True, reservoir_first=True, efficient=False,
                 season=None, interval_days=1, other=None,
                 river_store_m3=None, river_refill_m3=0.0, site_of_cell=None,
                 gw_store_m3=None, res_store_m3=None, groundwater=False,
                 **config):
        super().__init__(**config)
        self.driver = driver
        self._cfg_vic = dict(vic or {})
        if global_param:
            self._cfg_vic['global_param'] = global_param
        self.opt = dict(
            irrigated_tiles=irrigated_tiles, paddy_tiles=paddy_tiles,
            irrigated_fraction=irrigated_fraction, wcr_fract=wcr_fract,
            wfc_fract=wfc_fract, efficiency=efficiency,
            conveyance_loss=conveyance_loss, gw_fraction=gw_fraction,
            allocation=allocation, loss_return=loss_return,
            fallback=bool(fallback), reservoir_first=bool(reservoir_first),
            efficient=bool(efficient), season=season,
            other=dict(other or {}), river_store_m3=river_store_m3,
            river_refill_m3=river_refill_m3, site_of_cell=site_of_cell,
            gw_store_m3=gw_store_m3, res_store_m3=res_store_m3)
        if allocation not in ('priority', 'proportional'):
            raise CouplingError('irrigation: allocation must be priority or '
                                'proportional')
        if loss_return not in ('river', 'soil'):
            raise CouplingError('irrigation: loss_return must be river or '
                                'soil')
        self.ext_gw = bool(groundwater)
        self.interval_s = 86400.0 * float(interval_days)
        self.engine = None
        self.river = self.gw = self.res = None   # stub stores [m3]
        self.t_index = 0
        self._declare()

    # -- declarations -------------------------------------------------------
    def _declare(self):
        self.inputs = (
            VarSpec('soil_moisture', 'mm', ('cell', 'layer'), kind='state',
                    grid=GRID, description='VIC layer moisture at the '
                    'interval end (the demand is computed from it)'),
            VarSpec('soil_unmet', 'mm', ('cell',), kind='flux', grid=GRID,
                    description='what VIC could not add under the layer '
                    'capacity (soil_add_mode: cap), returned to the '
                    'leftover store'),
            VarSpec('season', '1', ('cell',), kind='state', grid=GRID,
                    description='1 where the irrigation season is open '
                    '(the crop connector\'s crop_active); else the '
                    'calendar'),
            VarSpec('root_depth', 'm', ('cell',), kind='state', grid=GRID,
                    description='rooting depth (crop connector); layers '
                    'whose top lies above it are rooted'))
        self.outputs = (
            VarSpec('soil_add', 'mm', ('cell',), kind='flux', grid=GRID,
                    description='irrigation delivered to VIC layer 0, mm '
                    'over the cell'),
            VarSpec('demand', 'm3', ('cell',), kind='state', grid=GRID,
                    description='water demand of all sectors this interval '
                    '(diagnostic)'),
            VarSpec('unmet_demand', 'm3', ('cell',), kind='state',
                    grid=GRID, description='demand no source could supply '
                    '(diagnostic)'))
        if self.ext_gw:
            # groundwater is a module of the run (docs/CONNECTOR_API.md):
            # the groundwater share of the withdrawals is asked of its wells
            # (gw_request -> groundwater.pumping_request) and paid for by what
            # the wells deliver one day later (groundwater.pumping ->
            # gw_pumped, lag 1); the difference is the payable store
            self.inputs += (
                VarSpec('gw_pumped', 'm3', ('cell',), kind='flux', grid=GRID,
                        description='water the wells delivered (groundwater'
                        '.pumping, lag 1)'),)
            self.outputs += (
                VarSpec('gw_request', 'm3', ('cell',), kind='state',
                        grid=GRID, description='groundwater withdrawn this '
                        'interval, asked of the wells'),
                VarSpec('gw_payable', 'm3', ('cell',), kind='state',
                        grid=GRID, description='groundwater taken and not '
                        'yet delivered by the wells (diagnostic)'))

    def grids(self):
        if self.engine is None and self.driver is None:
            return [GridSpec(GRID, 0, kind='cells')]
        d = self.driver
        area = torch.as_tensor(d.area * (d.frac if d.frac is not None
                                         else 1.0), dtype=torch.float64)
        return [GridSpec(GRID, d.n_cells, area=area, kind='cells')]

    # -- life cycle ---------------------------------------------------------
    def initialize(self, context):
        if self.driver is None:
            self.driver = common.shared_driver(self._cfg_vic, context)
        d = self.driver
        if d.area is None:
            raise CouplingError('irrigation: the VIC domain has no cell '
                                'areas; water volumes need them')
        dev, dt = context.device, torch.float64
        self.device = dev
        n = d.n_cells
        o = self.opt
        soil, mosaic = d.soil(), d.mosaic()
        self.cv, self.root_frac, self.layer_top = None, None, None
        L = soil.max_moist.shape[-1]
        tiles = o['irrigated_tiles']
        cv = mosaic.cv.to(dev, dt)                       # [cell, tile]
        sel = torch.ones(cv.shape[1], dtype=torch.bool, device=dev) \
            if tiles is None else torch.zeros(cv.shape[1], dtype=torch.bool,
                                              device=dev)
        if tiles is not None:
            sel[torch.as_tensor(list(tiles), dtype=torch.long,
                                device=dev)] = True
        cvs = cv * sel
        frac = cvs.sum(1)                                # [cell]
        if isinstance(o['irrigated_fraction'], dict):
            o['irrigated_fraction'] = self._cell_field(
                o['irrigated_fraction'], 'irrigated_fraction', context)
        if isinstance(o['gw_fraction'], dict):
            o['gw_fraction'] = self._cell_field(
                o['gw_fraction'], 'gw_fraction', context)
        if o['irrigated_fraction'] is not None:
            f = o['irrigated_fraction']
            frac = (torch.full((n,), float(f), dtype=dt, device=dev)
                    if not torch.is_tensor(f) and torch.as_tensor(f).dim() == 0
                    else torch.as_tensor(f, dtype=dt, device=dev))
        rf = (mosaic.root_frac.to(dev, dt) * cvs.unsqueeze(-1)).sum(1)
        self.root_present = rf > 0                       # [cell, layer]
        mx = soil.max_moist.to(dev, dt)                  # [cell, layer]
        self.max_moist = mx
        wcr = common.fraction_layers(o['wcr_fract'], mx, 'wcr_fract')
        if wcr is None:
            wcr = getattr(soil, 'w_crit_mm', None)
            wcr = mx * 0.7 * 0.5 if wcr is None else wcr.to(dev, dt)
            wcr = common.critical_from_params(d, dev, dt, mx, wcr)
        wfc = common.fraction_layers(o['wfc_fract'], mx, 'wfc_fract')
        if wfc is None:
            wfc = wcr / 0.7          # VIC: Wcr = 0.7 Wfc (the image driver
            #                          reads Wcr_FRACT only)
        self.wcr, self.wfc = wcr, wfc
        depth = getattr(soil, 'depth', None)
        if depth is not None:
            dep = depth.to(dev, dt)
            self.layer_top = torch.cumsum(dep, -1) - dep      # m
        paddy = o['paddy_tiles']
        pf = torch.zeros(n, dtype=torch.bool, device=dev)
        if paddy:
            pcv = (cv[:, list(paddy)]).sum(1)
            pf = pcv > 0.5 * frac.clamp(min=1e-30)
        self.frac = frac
        self.paddy = pf
        # engine
        oth = o['other']
        n_other = len(oth.get('demand_mm_per_day', []) or [])
        self.n_other = n_other
        area = torch.as_tensor(d.area * (d.frac if d.frac is not None
                                         else 1.0), dtype=dt, device=dev)
        self.area = area
        rep = lambda v, k: torch.as_tensor(v, dtype=dt, device=dev).expand(
            n, k).clone()
        kw = {}
        if n_other:
            kw = dict(cons_frac=rep(oth.get('consumption', [0.2] * n_other),
                                    n_other),
                      conv_loss=rep(oth.get('conveyance_loss',
                                            [0.1] * n_other), n_other),
                      gw_frac=rep(oth.get('gw_fraction', [0.0] * n_other),
                                  n_other))
            self.demand_other = rep(oth['demand_mm_per_day'], n_other) \
                * area.unsqueeze(-1) / 1000.0 * (self.interval_s / 86400.0)
        else:
            self.demand_other = None
        site = o['site_of_cell']
        site = torch.arange(n, device=dev) if site is None else \
            torch.as_tensor(site, dtype=torch.long, device=dev)
        self.site = site
        self.n_site = int(site.max().item()) + 1
        cfg = WaterUseConfig(allocation=o['allocation'],
                             loss_return=o['loss_return'],
                             fallback=o['fallback'],
                             reservoir_first=o['reservoir_first'],
                             efficient=o['efficient'], dtype=dt)
        eff = torch.full((n,), float(o['efficiency']), dtype=dt, device=dev)
        self.engine = WaterUseEngine(
            area, site_of_cell=site, n_other=n_other, irrigation=True,
            irr_eff=eff,
            irr_gw_frac=(torch.as_tensor(o['gw_fraction'], dtype=dt,
                                         device=dev).expand(n).clone()),
            irr_conv_loss=torch.full((n,), float(o['conveyance_loss']),
                                     dtype=dt, device=dev),
            cfg=cfg, **kw)
        self.river = common.store(o['river_store_m3'], self.n_site, dt, dev)
        self.gw = common.store(o['gw_store_m3'], n, dt, dev)
        self.res = common.store(o['res_store_m3'], n, dt, dev)
        self.refill = common.store(o['river_refill_m3'], self.n_site, dt,
                                   dev, default=0.0)
        self.payable = torch.zeros(n, dtype=dt, device=dev)
        self._dates = d.dates
        self.spd = d.steps_per_day
        self.t_index = 0

    def _cell_field(self, spec, name, context):
        """[n] values from a {file: x.nc, var: v[, scale: k]} sampled at the
        VIC cell centres (nearest neighbour), as the groundwater connector
        reads its aquifer fields."""
        from .groundwater import _field
        d = self.driver
        dom = getattr(d, 'dom', None) or d.case.dom
        return _field(dict(spec), name, np.asarray(dom.lat, float),
                      np.asarray(dom.lon, float), None, context.base_dir)

    # -- the season ----------------------------------------------------------
    def _season(self, inputs, context, dt):
        n, dev = self.driver.n_cells, self.device
        s = inputs.get('season')
        if s is not None:
            return torch.as_tensor(s, device=dev) > 0.5
        c = self.opt['season']
        if not c:
            return torch.ones(n, dtype=torch.bool, device=dev)
        step = int(round(context.t / (86400.0 / self.spd)))
        step = min(step, len(self._dates) - 1)
        doy = self._dates[step].timetuple().tm_yday
        a, b = int(c.get('start_doy', 1)), int(c.get('end_doy', 366))
        on = (a <= doy <= b) if a <= b else (doy >= a or doy <= b)
        return torch.full((n,), bool(on), dtype=torch.bool, device=dev)

    def _soil(self, inputs, context, dt):
        m = inputs.get('soil_moisture')
        if m is None and context.t == 0:
            # the run's first day with VIC's moisture read at lag 1 (a
            # day-coupled irrigation): VIC has no state yet and refuses a
            # soil_add before its first interval (coupling/vic.py
            # _soil_add_ok), so the first day is taken at field capacity --
            # no demand
            m = self.wfc
        if m is None:
            raise ComputeError('irrigation: no soil_moisture input; connect '
                               'vic.soil_moisture -> irrigation.soil_moisture')
        m = torch.as_tensor(m, dtype=torch.float64, device=self.device)
        root = self.root_present
        rd = inputs.get('root_depth')
        if rd is not None and self.layer_top is not None:
            root = root | (self.layer_top < torch.as_tensor(
                rd, dtype=torch.float64, device=self.device).unsqueeze(-1))
        k = lambda x: x.unsqueeze(1)
        season = self._season(inputs, context, dt)
        return dict(moist=k(m), ice=torch.zeros_like(k(m)),
                    wfc=k(self.wfc), wcr=k(self.wcr),
                    root=k(root.to(torch.float64)),
                    max_moist0=self.max_moist[:, :1],
                    frac=self.frac.unsqueeze(1),
                    paddy=self.paddy.unsqueeze(1),
                    season_open=season.unsqueeze(1),
                    moist0=m[:, :1])

    # -- compute: from the accepted state, mutating nothing -------------------
    def compute(self, inputs, context, dt):
        e = self.engine
        if e is None:
            raise ComputeError('irrigation: not initialised')
        if abs(dt - self.interval_s) > 1e-6:
            raise ComputeError('irrigation: asked for %g s, the interval '
                               'is %g s' % (dt, self.interval_s))
        dev, f64 = self.device, torch.float64
        saved = e.state_dict()
        try:
            unmet_in = inputs.get('soil_unmet')
            n = e.N
            left0 = e.leftover_m3.clone()      # the accepted store
            u_m3 = torch.zeros(n, dtype=f64, device=dev)
            if unmet_in is not None:
                # mm over the cell -> m3; joins the leftover store
                u_m3 = torch.as_tensor(unmet_in, dtype=f64,
                                       device=dev) * self.area / 1000.0
                if e.leftover is None:
                    e.leftover = torch.zeros(n, 1, dtype=f64, device=dev)
                e.leftover = e.leftover + u_m3.unsqueeze(-1)
                e.leftover_m3 = e.leftover.sum(-1)
            pumped = None
            if self.ext_gw:
                pumped = inputs.get('gw_pumped')
                pumped = torch.zeros(n, dtype=f64, device=dev) \
                    if pumped is None else torch.as_tensor(
                        pumped, dtype=f64, device=dev)
            soil = self._soil(inputs, context, dt)
            river = StubService(self.river)
            gw = StubService(self.gw)
            res = StubService(self.res)
            out = e.step(demand_other=self.demand_other, soil=soil,
                         router=_RiverView(river), gw=gw, res=res)
            new = e.state_dict()
        finally:
            e.load_state_dict(saved)
        applied_m3 = out['applied_m3']
        W = out['withdrawn'].sum(1)
        consumed_o = out['consumed'][:, :-1].sum(1) if self.n_other else \
            torch.zeros_like(W)
        river_ret = out['river_return']
        has = e.has_site
        routed = torch.where(has, river_ret, torch.zeros_like(river_ret))
        unrouted = river_ret - routed
        dleft = new['leftover_m3'] - left0
        # soil_add as mm over the cell (what VIC's input is)
        add_mm = applied_m3 / self.area * 1000.0
        P = 0.0
        I_cell = W + u_m3
        Q_cell = applied_m3 + river_ret
        W_gw = out['gw'].sum(1)
        pay_new = self.payable
        dpay = torch.zeros_like(W)
        if self.ext_gw:
            # the wells' water arrives a day late: this interval's gw
            # withdrawal is a payable, the delivery of the last one pays it
            I_cell = W - W_gw + pumped + u_m3
            pay_new = self.payable + W_gw - pumped
            dpay = pay_new - self.payable
        res_cell = I_cell - consumed_o - Q_cell - (dleft - dpay)
        scale_cell = (W.abs() + u_m3.abs() + consumed_o.abs()
                      + applied_m3.abs() + river_ret.abs() + dleft.abs()
                      + left0.abs() + dpay.abs() + W_gw.abs())
        tot = lambda x: x.sum()
        bal = Balance(
            P=0.0, I_ext=tot(I_cell), ET=tot(consumed_o), Q_out=tot(Q_cell),
            dS=tot(dleft - dpay), n_terms=8 * e.N,
            scale=tot(scale_cell),
            per_cell=dict(residual=res_cell / self.area * 1000.0,
                          scale=scale_cell / self.area * 1000.0,
                          n_terms=8, eps=2.220446049250313e-16))
        # new stub stores (after commit): river gets returns and refill
        r_new = river.new_store() + self.refill
        g_new = gw.new_store()
        s_new = res.new_store()
        ex = dict(soil_add=add_mm, demand=out['demand'].sum(1),
                  unmet_demand=out['unmet'].sum(1))
        if self.ext_gw:
            ex['gw_request'] = W_gw
            ex['gw_payable'] = pay_new
        series = dict(
            soil_add=add_mm.unsqueeze(0), demand=ex['demand'].unsqueeze(0),
            unmet=ex['unmet_demand'].unsqueeze(0),
            withdrawn=W.unsqueeze(0),
            from_river=out['river'].sum(1).unsqueeze(0),
            from_gw=out['gw'].sum(1).unsqueeze(0),
            from_res=out['res'].sum(1).unsqueeze(0),
            river_return=river_ret.unsqueeze(0),
            leftover=new['leftover_m3'].unsqueeze(0),
            req_mm=out['req'].squeeze(1).unsqueeze(0))
        prop = Proposal(
            connector=self.name, t0=context.t, t1=context.t + dt,
            state=dict(engine=new, river=r_new, gw=g_new, res=s_new,
                       payable=pay_new),
            exports=ex, balance=bal, series=series, backend=str(dev),
            info=dict(
                avail_river=self.river,
                avail_gw=self.gw, avail_res=self.res, site=self.site,
                river_taken=river.taken, gw_taken=gw.taken,
                res_taken=res.taken, out=out, unrouted=unrouted))
        return prop

    def validate(self, proposal, context):
        info, p = proposal.info, []
        tol = 1e-9
        for nm, avail, taken in (('river', info['avail_river'],
                                  info['river_taken']),
                                 ('groundwater', info['avail_gw'],
                                  info['gw_taken']),
                                 ('reservoir', info['avail_res'],
                                  info['res_taken'])):
            if taken is None:
                continue
            bad = taken > avail * (1 + tol) + 1e-9
            p.append(DeviceFlag('irrigation: %s allocation exceeds what is '
                                'available at %%d element(s)' % nm,
                                bad.any(), values=(bad.sum(),)))
        out = info['out']
        p.append(DeviceFlag('irrigation: delivered more than the demand',
                            (out['unmet'] < -1e-9 * (out['demand'].abs()
                                                     + 1.0)).any()))
        p.append(DeviceFlag('irrigation: non-finite withdrawal',
                            ~torch.isfinite(out['withdrawn']).all()))
        p.append(DeviceFlag('irrigation: the groundwater payable went negative '
                            '(more delivered than asked)',
                            (proposal.state['payable'] < -1e-6).any())) if self.ext_gw else None
        return p

    def commit(self, proposal, context):
        s = proposal.state
        self.engine.load_state_dict(s['engine'])
        self.river, self.gw, self.res = s['river'], s['gw'], s['res']
        self.payable = s['payable']
        self.t_index += 1

    # -- restart --------------------------------------------------------------
    def save_state(self):
        sd = self.engine.state_dict()
        cpu = lambda x: None if x is None else x.cpu()
        return dict(
            leftover=cpu(sd['leftover']), leftover_m3=cpu(sd['leftover_m3']),
            led={k: v.cpu() for k, v in sd['led'].items()},
            led_S={k: v.cpu() for k, v in sd['led_S'].items()},
            river=self.river.cpu(), gw=self.gw.cpu(), res=self.res.cpu(),
            payable=self.payable.cpu(), t_index=self.t_index)

    def load_state(self, blob):
        dev = self.device
        to = lambda x: None if x is None else x.to(dev)
        self.engine.load_state_dict(dict(
            leftover=to(blob['leftover']),
            leftover_m3=to(blob['leftover_m3']),
            led={k: v.to(dev) for k, v in blob['led'].items()},
            led_S={k: v.to(dev) for k, v in blob['led_S'].items()}))
        self.river, self.gw, self.res = (blob['river'].to(dev),
                                         blob['gw'].to(dev),
                                         blob['res'].to(dev))
        if 'payable' in blob:
            self.payable = blob['payable'].to(dev)
        self.t_index = int(blob['t_index'])
