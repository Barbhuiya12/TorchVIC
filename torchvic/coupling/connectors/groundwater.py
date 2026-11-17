"""GroundwaterConnector: the MODFLOW 6 GWF engine (torchvic.groundwater) as a
module of the coupled TorchVIC system.

(`profile: vicwur3` selects VicWur3GroundwaterConnector at the end of this
file instead: VIC-WUR 3.0's MODFLOW coupling as a member of the VIC-WUR phase
group, see docs/connectors/vicwur.md.)

What it exchanges (VIC grid = groundwater grid: one VIC cell is one aquifer
cell, the aquifer cell's area is the VIC cell's land area, so a depth over a
VIC cell and a volume into the aquifer are the same water):

  in   recharge          flux, mm   VIC's OUT_BASEFLOW (see below)
  in   pumping_request   state, m3  volume wanted from each cell over the
                                    interval (irrigation / water use)
  in   river_stage       state, m   channel water level (routing); lagged by
                                    one interval; without it the configured
                                    constant stage holds
  out  discharge         flux, m3   water the aquifer gives to the rivers over
                                    the interval: drain outflow + river-bed
                                    gains - river-bed losses (negative where
                                    the river loses water to the aquifer)
  out  pumping           flux, m3   what the wells delivered (<= request)
  out  pumping_unmet     state, m3  request - delivered, handed back
  out  head, water_table_depth      states at the interval end

Recharge. VIC exposes no explicit recharge. Its bottom-layer outflow is
OUT_BASEFLOW (runoff.c:284-345, the ARNO baseflow dt_baseflow taken from the
last layer, summed over frost areas, cell->baseflow; put_data.c:633), and the
inter-layer drainage Q12 stays inside the step code. OUT_BASEFLOW is the one
VIC flux that is "water leaving the soil column downward to the groundwater
system" in VIC's own conceptual model, so it is taken as the recharge. It is
a flux VIC computes without any groundwater; the connector does not feed
anything back (no head feedback into VIC's soil column, which would change
VIC's physics). Water is counted once: the same baseflow must not also be
routed. In a configuration this means

    vic.baseflow -> groundwater.recharge        (this connector)
    vic.runoff   -> routing.inflow              (NOT vic.total_runoff)
    groundwater.discharge -> routing.inflow     (baseflow arrives as
                                                 groundwater discharge)

`ownership_problems(config)` reports a configuration that feeds baseflow to
the aquifer and also routes it (vic.total_runoff to a consumer).

Time. The aquifer steps with `interval_days` (default 1 day; K in m/d). VIC
hands over a window total (31 days by default); the runtime prorates it
uniformly over the aquifer's steps, so the day-to-day pattern of recharge
inside a VIC window is flat. `exchange: day` on VIC gives daily recharge.

Transaction. compute() runs the engine's step and then puts the committed
state back; commit() makes the proposal's heads the accepted ones. A step
that does not converge raises ComputeError and nothing changes.

Water balance (m3 over the interval, docs/WATER_ACCOUNTING.md):
    I_ext  = recharge + general-head inflow
    ET     = groundwater evapotranspiration (EVT)
    Q_out  = pumping delivered + discharge to rivers (net, signed) +
             general-head outflow
    dS     = change of the aquifer storage (the engine's STO budget terms)
    closure= the engine's own budget discrepancy (the solver's convergence
             error); `validate` bounds it by `max_discrepancy_pct`
The independent check of dS against the head change is in the tests.

Parameters. Aquifer fields (k, sy, ss, thickness, top, drain/river/ET
parameters) are a scalar, an array of one value per VIC cell, or
{file: path.nc, var: name[, scale: x]} sampled at the cell centres (nearest
neighbour). With none given the aquifer is SYNTHETIC (the defaults below) and
the run says so; docs/connectors/groundwater.md lists the sources for India.
"""
import os

import numpy as np
import torch

from ...core import ordered as _ordered

from ..connector import (Balance, ComputeError, Connector, DeviceFlag,
                         Proposal)
from ..spec import CouplingError, GridSpec, VarSpec
from ..vic import GRID
from .vicwur_common import GRID as GRID_W, STEP_S, WurConnector, as_f64

DAY = 86400.0

#: synthetic aquifer used when no parameters are given (documented, not data)
SYNTHETIC = dict(k=1.0, sy=0.1, ss=1e-5, thickness=50.0, top=0.0,
                 depth_to_water=5.0, drain_depth=2.0, drain_cond_factor=1.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def domain_from_global(global_param, log=None):
    """The active cells a VIC global parameter file runs (lat, lon, area,
    frac), the way case.load forms them (same cell order as VIC)."""
    from ... import case as _case
    from ...io.domain import ImageDomain
    from ...io.global_param import GlobalParam
    from ...io.params import ImageParams
    say = log or (lambda *a: None)
    gp = GlobalParam(global_param, sample_data=_case.sample_data_dir())
    dom = ImageDomain(_case._fix_path(gp.domain_file, say),
                      names=gp.domain_vars)
    ImageParams(_case._fix_path(gp.params_file, say), dom)
    return dom


def _axes(lat, lon, resolution=None):
    """Regular row/column axes covering the cell centres: (lat axis, lon
    axis, row index, column index) with the axes ascending."""
    out = []
    for v in (np.asarray(lat, float), np.asarray(lon, float)):
        u = np.unique(np.round(v, 9))
        d = np.diff(u)
        step = (float(resolution) if resolution else
                (float(d.min()) if d.size else None))
        if step is None:
            raise CouplingError('groundwater: a single row or column needs '
                                '`resolution` (degrees)')
        n = int(round((u[-1] - u[0]) / step)) + 1
        ax = u[0] + step * np.arange(n)
        idx = np.rint((v - u[0]) / step).astype(np.int64)
        if np.abs(ax[idx] - v).max() > 1e-6 * max(step, 1e-12) + 1e-9:
            raise CouplingError('groundwater: the cell centres are not on a '
                                'regular lat/lon grid')
        out += [ax, idx]
    return out[0], out[1], out[2], out[3]


def _field(spec, name, lat, lon, default=None, base_dir='.'):
    """Per-cell float64 array from a scalar, an array or a file spec."""
    n = len(lat)
    if spec is None:
        if default is None:
            return None
        spec = default
    if isinstance(spec, dict):
        import xarray as xr
        path = spec['file']
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        from ...io.ncshare import read
        ds = read(path)
        da = ds[spec.get('var', name)]
        ydim = [d for d in da.dims if 'lat' in d.lower() or d == 'y'][0]
        xdim = [d for d in da.dims if 'lon' in d.lower() or d == 'x'][0]
        yc = ds[ydim] if ydim in ds.coords else None
        v = da.sel({ydim: xr.DataArray(lat, dims='c'),
                    xdim: xr.DataArray(lon, dims='c')},
                   method='nearest').values.astype(np.float64)
        v = v * float(spec.get('scale', 1.0))
        ds.close()
        del yc
        return v
    a = np.asarray(spec, dtype=np.float64)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.shape != (n,):
        raise CouplingError('groundwater: %s has %s values for %d cells'
                            % (name, a.shape, n))
    return a.copy()


def ownership_problems(config):
    """Problems with how a configuration (the dict TorchVIC.from_dict takes)
    shares VIC's baseflow between the aquifer and the river routing: the
    baseflow must be counted once."""
    conns = config.get('connections') or []
    gw = [n for n, m in (config.get('modules') or {}).items()
          if (m or {}).get('connector', n) == 'groundwater'
          and (m or {}).get('enabled', True)]
    out = []
    for g in gw:
        fed = [c for c in conns if c['to'] == g + '.recharge']
        if not fed:
            continue
        src = fed[0]['from']
        vic, var = src.split('.')
        if var != 'baseflow':
            out.append('%s.recharge is fed by %s; VIC\'s recharge candidate '
                       'is %s.baseflow' % (g, src, vic))
        also = [c for c in conns if c['from'] == vic + '.total_runoff']
        if var == 'baseflow' and also:
            out.append('%s.baseflow feeds the aquifer and %s.total_runoff '
                       '(runoff + baseflow) feeds %s: the baseflow would be '
                       'counted twice; route %s.runoff instead'
                       % (vic, vic, also[0]['to'], vic))
    return out


# ---------------------------------------------------------------------------
class GroundwaterConnector(Connector):
    name = 'groundwater'
    version = '1'
    backends = ('cpu', 'cuda')
    graph_safe = False        # the engine captures its own CUDA graphs
    checkpoint = 'full'
    owned_state = ('groundwater_head',)
    imported_fluxes = ('recharge',)
    exported_fluxes = ('discharge', 'pumping')
    validation = ('engine: MODFLOW 6 6.7.0 (docs/connectors/groundwater.md '
                  'section 9); coupling: conservation, restart and pumping '
                  'limits in tests/test_coupling_groundwater.py')

    def __init__(self, domain=None, grid=GRID, interval_days=1.0,
                 aquifer=None, drain=None, river=None, evt=None,
                 pumping=True, afr=0.1, pump_cap=1.0, resolution=None,
                 max_discrepancy_pct=1e-4, solver=None, area=None,
                 global_param=None, async_convergence=True, **config):
        super().__init__(**config)
        self.domain = domain
        self.global_param = global_param
        self.grid = grid
        self.interval_s = float(interval_days) * DAY
        self.aq = dict(aquifer or {})
        self.drain_cfg = None if drain is False else dict(drain or {})
        self.river_cfg = dict(river) if river else None
        self.evt_cfg = dict(evt) if evt else None
        self.pumping_on = bool(pumping)
        self.afr = float(afr)
        self.pump_cap = float(pump_cap)
        self.resolution = resolution
        self.max_pct = float(max_discrepancy_pct)
        self.solver = dict(solver or {})
        self.async_conv = bool(async_convergence)
        self.area_override = area
        self.synthetic = False
        self.model = None
        self.n = 0
        self.t = 0.0
        self.inputs = (
            VarSpec('recharge', 'mm', ('cell',), kind='flux', grid=grid,
                    description='VIC baseflow (OUT_BASEFLOW) over the '
                                'interval, the recharge'),
            VarSpec('pumping_request', 'm3', ('cell',), kind='state',
                    grid=grid, description='volume wanted from the wells '
                                           'over the interval'),
            VarSpec('river_stage', 'm', ('cell',), kind='state', grid=grid,
                    description='channel water level (RIV stage)'))
        self.outputs = (
            VarSpec('discharge', 'm3', ('cell',), kind='flux', grid=grid,
                    description='aquifer to river over the interval (drain + '
                                'river-bed exchange, signed)'),
            VarSpec('pumping', 'm3', ('cell',), kind='flux', grid=grid,
                    description='water the wells delivered'),
            VarSpec('pumping_unmet', 'm3', ('cell',), kind='state',
                    grid=grid, description='request not delivered'),
            VarSpec('head', 'm', ('cell',), kind='state', grid=grid,
                    description='hydraulic head at the interval end'),
            VarSpec('water_table_depth', 'm', ('cell',), kind='state',
                    grid=grid, description='depth of the water table below '
                                           'the surface at the interval '
                                           'end'))

    # -- declarations -------------------------------------------------------
    def grids(self):
        if self.model is None:
            return [GridSpec(self.grid, self.n, kind='cells')]
        return [GridSpec(self.grid, self.n, area=self._area.cpu(),
                         kind='cells',
                         description='one aquifer cell per VIC cell; area = '
                                     'the VIC cell\'s land area (m2)')]

    # -- set-up -------------------------------------------------------------
    def _cells(self, context):
        dom = self.domain
        if dom is None and self.global_param:
            dom = domain_from_global(self.global_param, context.log)
        if dom is None:
            raise CouplingError('groundwater: give `domain` (an object with '
                                'lat, lon[, area, frac]) or `global_param`')
        if isinstance(dom, dict):
            g = lambda k: dom.get(k)
        else:
            g = lambda k: getattr(dom, k, None)
        return (np.asarray(g('lat'), float), np.asarray(g('lon'), float),
                g('area'), g('frac'))

    def initialize(self, context):
        from ...groundwater import Grid, GroundwaterModel
        dev = context.device if context.device is not None else 'cpu'
        self.device = torch.device(dev)
        self.dtype = torch.float64
        lat, lon, area, frac = self._cells(context)
        n = self.n = len(lat)
        bd = context.base_dir
        ax_lat, row, ax_lon, col = _axes(lat, lon, self.resolution)
        if np.abs(ax_lat).max() + 0.5 * np.abs(np.diff(ax_lat)).max(
                initial=0.0) >= 90.0:
            raise CouplingError('groundwater: the grid reaches the pole '
                                '(|lat| >= 90): east-west distances vanish')
        R, C = len(ax_lat), len(ax_lon)
        if len(np.unique(row * C + col)) != n:
            raise CouplingError('groundwater: two VIC cells fall on one '
                                'aquifer cell')
        self.R, self.C = R, C
        self.idx = torch.as_tensor(row * C + col, device=self.device)
        mask = np.zeros((R, C), bool)
        mask[row, col] = True

        q = self.aq
        raw = {k: q.get(k) for k in SYNTHETIC if k in q}
        self.synthetic = not any(k in q for k in ('k', 'sy', 'thickness'))
        if self.synthetic:
            context.say('groundwater: SYNTHETIC aquifer parameters (%s); '
                        'not data' % ', '.join('%s=%g' % kv for kv in
                                               SYNTHETIC.items()))
        f = lambda name, default=None, spec=None: _field(
            raw.get(name) if spec is None else spec, name, lat, lon,
            SYNTHETIC.get(name) if default is None else default, bd)
        k, sy, ss = f('k'), f('sy'), f('ss')
        thick, top = f('thickness'), f('top')
        dtw = f('depth_to_water')
        if area is not None:
            a = np.asarray(area, float) * (np.asarray(frac, float)
                                           if frac is not None else 1.0)
        elif self.area_override is not None:
            a = _field(self.area_override, 'area', lat, lon, None, bd)
        else:
            a = None
        full = lambda v, fill=0.0: self._grid_np(v, row, col, fill)
        area_g = None
        if a is not None:
            area_g = full(a, 1.0)
        grid = Grid.latlon(ax_lat, ax_lon, full(top), full(top - thick),
                           mask=mask, area=area_g)
        self.grid_obj = grid
        aarr = a if a is not None else grid.area[row, col]
        self._area = torch.as_tensor(aarr, dtype=torch.float64)
        self.top = torch.as_tensor(top, dtype=self.dtype, device=self.device)
        self.bot = torch.as_tensor(top - thick, dtype=self.dtype,
                                   device=self.device)
        self.area_d = self._area.to(self.device)
        h0 = top - dtw
        if 'head0' in q:
            h0 = _field(q['head0'], 'head0', lat, lon, None, bd)
        # head closure 1e-8 m: the storage flow is Sy A dh / dt, so a head
        # error of dvclose is a relative budget error of dvclose / dh (1e-6 m
        # on a 3 cm step is 3e-3 %); rclose 1e-2 m3/d is above the round-off
        # of row sums at ISC size. Settable through the `solver` option.
        sv = dict(outer_dvclose=1e-8, inner_dvclose=1e-9, inner_rclose=1e-2)
        sv.update(self.solver)
        self.model = GroundwaterModel(
            grid, k11=full(k, 1.0), icelltype=1, ss=full(ss), sy=full(sy),
            strt=full(h0, 0.0), device=self.device, dtype=self.dtype, **sv)
        m = self.model
        node = lambda sel: (np.zeros(len(sel), np.int64), row[sel], col[sel])
        allc = np.arange(n)
        # drains: the groundwater discharge to rivers/streams when no river
        # stage is given (or in addition to it)
        self.drn_sel = None
        if self.drain_cfg is not None:
            d = self.drain_cfg
            dd = _field(d.get('depth'), 'drain_depth', lat, lon,
                        SYNTHETIC['drain_depth'], bd)
            elev = top - dd
            if 'elev' in d:
                elev = _field(d['elev'], 'drain_elev', lat, lon, None, bd)
            cond = _field(d.get('cond'), 'drain_cond', lat, lon,
                          SYNTHETIC['drain_cond_factor'] * k
                          * np.sqrt(aarr), bd)
            on = cond > 0
            self.drn_sel = torch.as_tensor(np.flatnonzero(on),
                                           device=self.device)
            if on.any():
                m.set_drains(node(np.flatnonzero(on)), elev[on], cond[on])
        # river-aquifer exchange
        self.riv_sel = None
        if self.river_cfg is not None:
            r = self.river_cfg
            cond = _field(r.get('cond'), 'river_cond', lat, lon, None, bd)
            on = cond > 0
            st = _field(r.get('stage'), 'river_stage', lat, lon,
                        None, bd)
            if st is None:
                st = top - 1.0
            rb = _field(r.get('bottom'), 'river_bottom', lat, lon,
                        None, bd)
            if rb is None:
                rb = st - float(r.get('bed_thickness', 1.0))
            self.riv_sel = torch.as_tensor(np.flatnonzero(on),
                                           device=self.device)
            self.riv_np = np.flatnonzero(on)
            if on.any():
                m.set_rivers(node(self.riv_np), st[on], cond[on], rb[on])
        # wells (pumping service)
        if self.pumping_on:
            m.set_wells(node(allc), np.zeros(n), auto_flow_reduce=self.afr)
        # groundwater evapotranspiration
        if self.evt_cfg is not None:
            e = self.evt_cfg
            surf = _field(e.get('surface'), 'evt_surface', lat, lon,
                          None, bd)
            surf = top if surf is None else surf
            rate = _field(e.get('rate'), 'evt_rate', lat, lon, None, bd)
            ext = _field(e.get('depth'), 'evt_depth', lat, lon, 2.0, bd)
            m.set_evt(surface=full(surf), rate=full(rate),
                      depth=full(ext, 1.0))
        self._rch = torch.zeros(R * C, dtype=self.dtype, device=self.device)
        self._wel = torch.zeros(n, dtype=self.dtype, device=self.device)
        self._flat = self.idx
        self.t = 0.0

    def _grid_np(self, v, row, col, fill):
        g = np.full((self.R, self.C), float(fill))
        g[row, col] = v
        return g

    # -- the step -----------------------------------------------------------
    def _cellvec(self, x):
        """A (nrow, ncol) or (1, nrow, ncol) field as the [n] cell vector."""
        return x.reshape(-1)[self._flat] if x.dim() == 2 else \
            x.reshape(-1)[self._flat]

    def _scatter(self, flows, sel):
        out = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        if sel is not None and sel.numel() and flows is not None:
            out = out.index_put((sel,), flows)
        return out

    def compute(self, inputs, context, dt):
        m = self.model
        dev, dty = self.device, self.dtype
        dtd = dt / DAY
        n = self.n
        rec = inputs.get('recharge')
        rec = (torch.zeros(n, dtype=dty, device=dev) if rec is None else
               torch.as_tensor(rec, dtype=dty, device=dev))
        self._rch.zero_()
        self._rch[self._flat] = rec / 1000.0 / dtd           # m/d
        m.set_recharge(self._rch.view(self.R, self.C))
        h = m.h[0].reshape(-1)[self._flat]
        req = inputs.get('pumping_request')
        req_eff = None
        if self.pumping_on:
            if req is None:
                req_eff = torch.zeros(n, dtype=dty, device=dev)
                req = req_eff
            else:
                req = torch.as_tensor(req, dtype=dty, device=dev)
                # never ask a cell for more than the water it holds above
                # the level where the wells start to reduce (bottom +
                # afr * thickness): drainable storage, Sy
                sy = m.sy[0].reshape(-1)[self._flat]
                floor = self.bot + self.afr * (self.top - self.bot)
                cap = self.pump_cap * sy * self.area_d * torch.clamp(
                    h - floor, min=0.0)
                req_eff = torch.minimum(torch.clamp(req, min=0.0), cap)
            self._wel.copy_(-req_eff / dtd)
            m.set_wells(q=self._wel)
        stage = inputs.get('river_stage')
        if stage is not None and self.riv_sel is not None \
                and self.riv_sel.numel():
            m.set_rivers(stage=torch.as_tensor(stage, dtype=dty,
                                               device=dev)[self.riv_sel])
        snap = m.state_dict()
        try:
            self._nsync = getattr(self, '_nsync', 0)
            # after a few synchronous steps (they settle the inner-solve
            # length and the chunk count) the convergence flag is read with
            # the ledger instead of per step
            defer = (self.async_conv and self._nsync >= 3
                     and not getattr(context, 'redo', False)
                     and torch.device(dev).type == 'cuda')
            self._nsync += 1
            res = m.step(dtd, defer=True) if defer else m.step(dtd)
        except Exception as e:                           # noqa: BLE001
            m.load_state_dict(snap)
            raise ComputeError('groundwater: step over [%g, %g) s failed: '
                               '%s: %s' % (context.t, context.t + dt,
                                           type(e).__name__, e)) from e
        new = m.state_dict()
        m.load_state_dict(snap)
        b, fl = res.budget, res.flows
        z = torch.zeros((), dtype=dty, device=dev)
        pk = lambda name, i: b[name][i] if name in b else z
        q_rch = pk('RCH', 0) * dtd
        q_ghb_in, q_ghb_out = pk('GHB', 0) * dtd, pk('GHB', 1) * dtd
        q_et = pk('EVT', 1) * dtd
        sto = (fl['STO-SS'].sum() + fl['STO-SY'].sum()) * dtd   # into flow
        disc = torch.zeros(n, dtype=dty, device=dev)
        riv_in = riv_out = z
        if 'DRN' in fl:
            disc = disc - self._scatter(fl['DRN'] * dtd, self.drn_sel)
        if 'RIV' in fl:
            disc = disc - self._scatter(fl['RIV'] * dtd, self.riv_sel)
            riv_in, riv_out = pk('RIV', 0) * dtd, pk('RIV', 1) * dtd
        pump = torch.zeros(n, dtype=dty, device=dev)
        if 'WEL' in fl:
            pump = -fl['WEL'] * dtd
        unmet = (req - pump) if (self.pumping_on and req is not None) \
            else torch.zeros(n, dtype=dty, device=dev)
        t_in = sum(pk(k, 0) for k in b)
        t_out = sum(pk(k, 1) for k in b)
        pump_tot = pump.sum()
        disc_tot = disc.sum()
        dS = -sto
        # the engine's own discrepancy (rate in - rate out), over the step
        closure = (t_in - t_out) * dtd
        throughput = (t_in + t_out) * dtd
        vec = torch.stack([q_rch, q_ghb_in, q_ghb_out, q_et, dS, pump_tot,
                           disc_tot, closure, throughput, riv_in, riv_out,
                           req_eff.sum() if req_eff is not None else z,
                           (req.sum() if req is not None else z)])
        # the terms stay on the device: the runtime brings them to the
        # host with the rest of the step's ledger (one copy)
        (v_rch, v_gi, v_go, v_et, v_dS, v_pump, v_disc, v_clo, v_thr,
         v_ri, v_ro, v_reqeff, v_req) = vec.unbind(0)
        pct = torch.where(v_thr != 0, 100.0 * v_clo / (0.5 * v_thr),
                          torch.zeros_like(v_thr))
        bal = Balance(
            I_ext=v_rch + v_gi, ET=v_et, Q_out=v_pump + v_disc + v_go,
            dS=v_dS, closure=v_clo, n_terms=max(8 * n, 1), scale=v_thr.abs(),
            detail=dict(recharge=v_rch, pumping=v_pump, discharge=v_disc,
                        river_in=v_ri, river_out=v_ro, ghb_in=v_gi,
                        ghb_out=v_go, pumping_request=v_req,
                        pumping_capped=v_req - v_reqeff,
                        discrepancy_pct=pct))
        hn = new['h'][0].reshape(-1)[self._flat]
        wet = hn > m.hdry * 0.5
        head = torch.where(wet, hn, self.bot)
        wtd = self.top - head
        series = dict(head=head.unsqueeze(0), discharge=disc.unsqueeze(0),
                      pumping=pump.unsqueeze(0),
                      pumping_unmet=unmet.unsqueeze(0),
                      recharge=(rec / 1000.0 * self.area_d).unsqueeze(0))
        return Proposal(
            connector=self.name, t0=context.t, t1=context.t + dt,
            state=new, balance=bal, series=series,
            backend=str(self.device),
            exports=dict(discharge=disc, pumping=pump, pumping_unmet=unmet,
                         head=head, water_table_depth=wtd),
            info=dict(outer=res.outer_iterations, inner=res.inner_iterations,
                      converged=res.converged, request=v_req,
                      delivered=v_pump, discrepancy_pct=bal.detail[
                          'discrepancy_pct'], head=hn))

    def _redo(self):
        """The optimistic step did not converge: the runtime reruns the
        step with Context.redo set, so the engine iterates until it does
        (and remembers the chunk count for the steps after)."""
        self.redone = getattr(self, 'redone', 0) + 1

    def validate(self, proposal, context):
        bad = []
        i = proposal.info
        cv = i.get('converged', True)
        if torch.is_tensor(cv):
            bad.append(DeviceFlag(
                'groundwater: the engine did not converge in the chunks '
                'launched ahead; the step is redone with the convergence '
                'loop', ~cv.bool(), retry=self._redo))
        elif not cv:
            bad.append('groundwater: the engine did not converge')
        bad.append(DeviceFlag(
            'groundwater: budget discrepancy %%.3e %%%% > %.3e %%%%'
            % self.max_pct, i['discrepancy_pct'].abs() > self.max_pct,
            values=(i['discrepancy_pct'],)))
        bad.append(DeviceFlag('groundwater: non-finite heads',
                              ~torch.isfinite(i['head']).all()))
        bad.append(DeviceFlag(
            'groundwater: delivered %.6e m3 exceeds the request %.6e m3',
            i['delivered'] > i['request'] * (1 + 1e-9) + 1e-6,
            values=(i['delivered'], i['request'])))
        return bad

    def commit(self, proposal, context):
        self.model.load_state_dict(proposal.state)
        self.t = proposal.t1

    # -- restart ------------------------------------------------------------
    def save_state(self):
        st = self.model.state_dict()
        return dict(h=st['h'].cpu(), ibound=st['ibound'].cpu(),
                    time=st['time'], kstp=st['kstp'], t=self.t)

    def load_state(self, blob):
        self.model.load_state_dict(dict(
            h=blob['h'], ibound=blob['ibound'], time=blob['time'],
            kstp=blob['kstp']))
        self.t = blob['t']


# ---------------------------------------------------------------------------
# the vicwur3 profile: VIC-WUR 3.0's MODFLOW coupling (C15,
# docs/connectors/vicwur.md) as a component of the VIC-WUR phase group
# ---------------------------------------------------------------------------
def _static_grid(spec, lat, lon, R, C, row, col, base_dir='.'):
    """The VIC-WUR 3.0 static fields (torchvic.vicwur.groundwater.STATIC_KEYS)
    on the (R, C) grid of the VIC cells: a dict of per-cell arrays ([n], or
    [layer, n] for head0 and max_moist), or {file: path.nc} with one
    variable per key on (lat, lon) / (layer, lat, lon) axes, sampled at the
    cell centres. Grid cells without a VIC cell are outside the land mask
    (NaN) and inactive."""
    from ...vicwur.groundwater import STATIC_KEYS
    n = len(lat)
    if isinstance(spec, dict) and 'file' in spec:
        import xarray as xr
        path = spec['file']
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        from ...io.ncshare import read
        ds = read(path)
        raw = {}
        for k in STATIC_KEYS:
            if k not in ds:
                continue
            da = ds[k]
            ydim = [d for d in da.dims if 'lat' in d.lower()][0]
            xdim = [d for d in da.dims if 'lon' in d.lower()][0]
            v = da.sel({ydim: xr.DataArray(lat, dims='c'),
                        xdim: xr.DataArray(lon, dims='c')},
                       method='nearest').values.astype(np.float64)
            raw[k] = v
        ds.close()
    else:
        raw = {k: np.asarray(v, np.float64) for k, v in dict(spec).items()}
    out = {}
    for k in STATIC_KEYS:
        if k not in raw:
            continue
        v = raw[k]
        if v.shape[-1] != n:
            raise CouplingError('groundwater (vicwur3): %s has %s values '
                                'for %d cells' % (k, v.shape, n))
        lead = v.shape[:-1]
        g = np.full(lead + (R, C), np.nan)
        g[..., row, col] = v
        out[k] = g
    for k in ('landmask', 'dem', 'aqdepth', 'ksat_log', 'cellarea', 'qbank',
              'riv_slope', 'z0', 'min_dem', 'kq3', 'sy', 'head0', 'ibound',
              'ldd'):
        if k not in out:
            raise CouplingError('groundwater (vicwur3): static field %r '
                                'missing' % k)
    for k in ('ksat_l1_conf_log', 'ksat_l2_conf_log'):
        out.setdefault(k, np.full((R, C), np.nan))
    out['ibound'] = np.nan_to_num(out['ibound']).astype(np.int64)
    return out


#: mode fixed only (exact stays VIC-WUR 3.0's, bit for bit): the numerics
#: that keep the daily river-aquifer exchange stable on a network (on ISC
#: 2005 the exchange flip-flopped, +-1e6 m3 s-1, discharge negative).
#: river_exchange: the exchange of the river cells is solved in the routing
#: sweep with the discharge of the same day, as an implicit reach, and
#: taken from the water in the reach (torchvic.vicwur.river_exchange); off:
#: VIC-WUR 3.0's formulas with daily steps (the stage of the day's
#: discharge, the flux forced into the next day's routing, so the loop lags
#: one day).
#: exchange_iterations: bisection steps of the loss (the result does not
#: depend on it beyond 2**-n of the bracket, tests/test_river_exchange.py);
#: stage_cap: the stage stays at or under the bank (bankfull depth).
FIXED_STABILITY = dict(river_exchange=True, exchange_iterations=50,
                       stage_cap=True)


class VicWur3GroundwaterConnector(WurConnector):
    """VIC-WUR 3.0's MODFLOW coupling on the TorchVIC MODFLOW 6 engine: the
    `vicwur3` profile of the groundwater connector (module doc of
    torchvic.vicwur.groundwater for the exchange, its C/Python lines and the
    modes).

    A component of the VIC-WUR phase group (role and plan kind
    'groundwater'): its O hook (after plugin_store_error) adds up the day's
    OUT_GWRECHARGE (the StepView's `gwrecharge`), OUT_DISCHARGE (the group's
    router) and, mode fixed, the non-renewable withdrawal ('gw_pump' of
    vicwur_nonrenew); at the end of the period (calendar month, or day) it
    runs one MODFLOW time step of the period's length and

      * hands -(RIV + DRN) to the routing as forced inflow for the next
        period (vicwur_routing's pre_force asks `river_force()`);
      * adds the EVT outflow (capillary rise) to soil layer `cpr_layer` of
        the view with VIC-WUR 3.0's saturation cascade (lands in the next
        vic_run).

    Options
        domain       {lat, lon, area} of the VIC cells, cell order (area m2:
                     the routing's cell area)
        static       the VIC-WUR 3.0 static fields (see _static_grid)
        start_date   'YYYY-MM-DD' of the run's first day (the calendar)
        mode         'exact' (VIC-WUR 3.0) or 'fixed'
        period       'month' (VIC-WUR 3.0) or 'day'; default month for exact,
                     day for fixed
        foc          capillary rise on (VIC-WUR 3.0 FOC coupling; POC = off)
        well_layer   MODFLOW layer of the wells (fixed), 0 or 1
        cpr_layer    soil layer that takes the capillary rise (VIC-WUR 3.0: 2)
        resolution   grid step (degrees) when the cells have one row/column
        solver       GroundwaterModel solver settings
        sync_steps   MODFLOW steps solved with a host read of the
                     convergence flag before the deferred (sync-free) path
    State (checkpoint): heads, the period's sums, the held forced inflow,
    the pending volumes.

    Ledger (m3 per day): I_ext = OUT_GWRECHARGE volume + CHD inflow;
    Q_out = forced inflow delivered to the routing that day (without a router
    in the group: the baseflow at the step) + wells (fixed, booked the day
    the non-renewable withdrawal is made) + capillary rise put into the soil
    + CHD outflow; dS = aquifer storage change (STO) + change of the pending
    volumes (recharge not yet stepped, baseflow not yet delivered, wells not
    yet pumped, fixed: top-layer excess carried back); closure = the engine's
    budget discrepancy and VIC-WUR 3.0's losses by name (`detail`):
    recharge not applied (exact: negative values clipped; any mode: MODFLOW
    applies none on a CHD cell), mode exact: the top layer's excess, the
    held rate over a month of another length.
    """
    name = 'groundwater'
    version = '1'
    kind = 'groundwater'
    role = 'groundwater'
    flags = dict(groundwater=True, force_routing=True)   # + river_exchange
    validation = ('VIC-WUR 3.0 package set and exchange vs MODFLOW 6 6.7.0 run by '
                  'VIC-WUR 3.0\'s own mf_run/config_module on the R1 G11 runs '
                  '(tests/test_vicwur_groundwater.py)')

    def __init__(self, domain=None, static=None, start_date=None,
                 mode='exact', period=None, foc=True, well_layer=1,
                 cpr_layer=2, resolution=None, solver=None, sync_steps=3,
                 stage_cap=None, river_exchange=None,
                 exchange_iterations=None, group=None, profile='vicwur3',
                 **config):
        D = FIXED_STABILITY
        rx = D['river_exchange'] if river_exchange is None else river_exchange
        # the slot of the river exchange: mode fixed with a daily period
        # (the plan asks for it when the group has routing)
        self._rx_asked = bool(rx) and mode == 'fixed'
        self.flags = dict(groundwater=True, force_routing=True,
                          river_exchange=self._rx_asked
                          and (period or 'day') == 'day')
        super().__init__(group=group, **config)
        if profile != 'vicwur3':
            raise CouplingError('groundwater: unknown profile %r' % profile)
        if mode not in ('exact', 'fixed'):
            raise CouplingError("groundwater (vicwur3): mode is 'exact' or "
                                "'fixed'")
        self.mode = mode
        self.exact = mode == 'exact'
        self.period = period or ('month' if self.exact else 'day')
        if self.period not in ('month', 'day'):
            raise CouplingError("groundwater (vicwur3): period is 'month' or "
                                "'day'")
        if domain is None or static is None or start_date is None:
            raise CouplingError('groundwater (vicwur3): give domain, static '
                                'and start_date')
        import datetime
        self.domain, self.static = domain, static
        self.date0 = datetime.date.fromisoformat(str(start_date)[:10])
        self.foc = bool(foc)
        self.well_layer = int(well_layer)
        self.cpr_layer = int(cpr_layer)
        self.resolution = resolution
        self.solver = dict(solver or {})
        self.sync_steps = int(sync_steps)
        stage_cap = D['stage_cap'] if stage_cap is None else stage_cap
        self.stage_cap = bool(stage_cap) and not self.exact
        self.xch = self.flags['river_exchange']
        self.xch_iter = int(D['exchange_iterations']
                            if exchange_iterations is None
                            else exchange_iterations)
        self._check_config()
        g = GRID_W
        self.outputs = (
            VarSpec('head', 'm', ('cell',), kind='state', grid=g,
                    description='head of MODFLOW layer 1 at the last step'),
            VarSpec('head2', 'm', ('cell',), kind='state', grid=g,
                    description='head of MODFLOW layer 2'),
            VarSpec('baseflow', 'm3 s-1', ('cell',), kind='state', grid=g,
                    description='-(RIV + DRN) of the step: the routing\'s '
                                'forced inflow of the next period (mode '
                                'fixed with the river exchange: the net '
                                'exchange, the river cells\' RIV taken in '
                                'the sweep)'),
            VarSpec('capillary_rise', 'mm', ('cell',), kind='flux', grid=g,
                    description='EVT outflow put into the soil'))
        self.owned_state = ('groundwater_head',)
        self.model = None

    # -- set-up ------------------------------------------------------------
    def _init(self, context):
        from ...groundwater import Grid, GroundwaterModel
        from ...vicwur import groundwater as W
        G = self.group
        dev = torch.device(self.dev)
        dom = self.domain
        gd = (lambda k: dom.get(k)) if isinstance(dom, dict) else \
            (lambda k: getattr(dom, k, None))
        lat, lon = np.asarray(gd('lat'), float), np.asarray(gd('lon'), float)
        n = self.N = len(lat)
        ax_lat, row, ax_lon, col = _axes(lat, lon, self.resolution)
        R, C = len(ax_lat), len(ax_lon)
        if len(np.unique(row * C + col)) != n:
            raise CouplingError('groundwater (vicwur3): two cells fall on one '
                                'aquifer cell')
        self.R, self.C = R, C
        yres = (float(self.resolution) if self.resolution else
                float(np.diff(ax_lat).min()) if R > 1 else
                float(np.diff(ax_lon).min()))
        xres = (float(self.resolution) if self.resolution else
                float(np.diff(ax_lon).min()) if C > 1 else yres)
        # config_module.py:143-144: delrow from the x pixel size, delcol
        # from the y pixel size; DIS gets delr=delrow, delc=delcol
        self.delr, self.delc = xres * W.DEG_M, yres * W.DEG_M
        s = _static_grid(self.static, lat, lon, R, C, row, col,
                         context.base_dir)
        lm = s['landmask']
        aq = W.aqdepth(lm, s['aqdepth'])
        s['aqdepth_c'] = aq
        tp = W.top(lm, s['dem'])
        bm = W.botm(tp, aq)
        kh, kv, st = W.npf(s, aq, self.delr, self.delc)
        ib = s['ibound']
        idom = np.stack([ib, ib])
        grid = Grid.dis(2, R, C, self.delr, self.delc, tp, np.stack(bm),
                        idomain=idom)
        sv = dict(outer_dvclose=1e-9, inner_dvclose=1e-10, inner_rclose=1e-6)
        sv.update(self.solver)
        m = self.model = GroundwaterModel(
            grid, k11=np.stack(kh), k33=np.stack(kv), icelltype=0,
            ss=np.stack(st), sy=0.0, iconvert=0, storagecoefficient=True,
            strt=np.stack(W.initial_head(s)), device=dev,
            dtype=torch.float64, **sv)
        act = ib > 0
        # CHD h = 0 at ibound 2, both layers (config_module.py:407-425)
        ch = W.chd_cells(s) & act
        if ch.any():
            i, j = np.nonzero(ch)
            k = np.concatenate([np.zeros(len(i), int), np.ones(len(i), int)])
            m.set_chd((k, np.tile(i, 2), np.tile(j, 2)),
                      head=np.zeros(2 * len(i)))
        t = lambda a: torch.as_tensor(np.asarray(a, np.float64), device=dev)
        # RIV (:312-370): the records do not depend on the discharge
        r = W.riv_static(s, tp)
        on = W.riv_records(r) & act
        ri, rj = np.nonzero(on)
        self.riv_flat = torch.as_tensor(ri * C + rj, device=dev)
        self.riv = {k: t(r[k][on]) for k in ('width', 'slope_used',
                                             'bot_bkfl', 'head_ini', 'rbot')}
        self.riv_cond0 = t(r['cond'][on])
        if self.stage_cap:
            self.riv['d_bkfl'] = t(r['d_bkfl'][on])
        self.riv['riv'] = torch.as_tensor(r['riv'][on], device=dev)
        if on.any():
            m.set_rivers((np.zeros(len(ri), int), ri, rj),
                         stage=W.stage_np(r, np.zeros((R, C)),
                                          self.exact)[on],
                         cond=r['cond'][on], rbot=r['rbot'][on])
        # DRN (:372-405)
        el, cd = W.drn(s, st[0])
        on = (cd > 0) & act
        di, dj = np.nonzero(on)
        self.drn_flat = torch.as_tensor(di * C + dj, device=dev)
        if on.any():
            m.set_drains((np.zeros(len(di), int), di, dj), el[on], cd[on])
        # EVT, FOC (:427-442; mf_run.py:177-186)
        if self.foc:
            for k in ('cpsurface', 'cprate', 'cpdistinct'):
                if k not in s:
                    raise CouplingError('groundwater (vicwur3): foc needs %s'
                                        % k)
            eon, sf, rt, dp = W.evt(s)
            eon = eon & act
            m.set_evt(surface=np.where(eon, sf, 0.0),
                      rate=np.where(eon, rt, 0.0),
                      depth=np.where(eon, dp, 1.0))
        # wells (fixed): one per VIC cell, no AUTO_FLOW_REDUCE (confined)
        if not self.exact:
            m.set_wells((np.full(n, self.well_layer), row, col), np.zeros(n))
        m.set_recharge(np.zeros((R, C)))
        self.flat = torch.as_tensor(row * C + col, device=dev)
        ca = s['cellarea'][row, col]
        self.ca = t(ca)
        self.ca_grid = t(np.nan_to_num(s['cellarea']))
        self.max_moist = (t(s['max_moist'][:, row, col].T)
                          if 'max_moist' in s else None)
        z = lambda: torch.zeros(n, dtype=torch.float64, device=dev)
        self.acc_gw, self.acc_q, self.acc_pump = z(), z(), z()
        self.carry, self.force, self.debt = z(), z(), z()
        self.cpr_mm, self.bf_rate = z(), z()
        zs = lambda: torch.zeros((), dtype=torch.float64, device=dev)
        self.pend_rch, self.pend_bf, self.pend_pump = zs(), zs(), zs()
        self.pend_carry = zs()
        self.ndays = 0
        self.nstep = 0
        self._sync_next = False
        self._conv = None
        self.router = None
        for role in ('routing', 'wateruse_local', 'nonrenew'):
            c = G.roles.get(role)
            if c is not None and not getattr(c, 'gwm', False):
                raise CouplingError('groundwater (vicwur3): the %s connector '
                                    'of the group must run with gwm=True '
                                    '(VIC-WUR 3.0 GWM)' % role)
        if 'routing' in G.roles:
            self.router = G.build()
            if self.router.n != n:
                raise CouplingError('groundwater (vicwur3): %d cells, the '
                                    'routing has %d' % (n, self.router.n))
        if self._rx_asked and self.router is not None and not self.xch:
            raise CouplingError(
                "groundwater (vicwur3): mode fixed with routing takes the "
                "river exchange in the daily sweep: period 'day' (or "
                "river_exchange=False for VIC-WUR 3.0's formulas, which can make "
                "the discharge negative)")
        self.xch = self.xch and self.router is not None
        self.rx = None
        if self.xch:
            self._build_exchange()
        self._S_aq = zs()

    # -- the group's hooks ---------------------------------------------------
    def core_hooks(self, core):
        """VIC runs VIC-WUR 3.0's GWM soil (runoff_GWM, vicwur_core gwm) in
        this connector's mode: the recharge it hands over is
        OUT_GWRECHARGE."""
        core.gwm = self.mode

    def _build_exchange(self):
        """The per-cell statics and buffers of the sweep slot (river cells:
        a RIV record that is a river, on a VIC cell) and the maps between
        the RIV records and the cells."""
        from ...vicwur import river_exchange as RX
        dev, n = self.dev, self.N
        torch_ = torch
        nrec = int(self.riv_flat.numel())
        cell_of = torch_.full((self.R * self.C,), -1, dtype=torch_.int64,
                              device=dev)
        cell_of[self.flat] = torch_.arange(n, device=dev)
        rec_cell = cell_of[self.riv_flat]
        isr = self.riv['riv'].bool() & (rec_cell >= 0)
        self.riv_idx = torch_.nonzero(isr).reshape(-1)       # setup: one sync
        self.riv_cell = rec_cell[self.riv_idx]
        f = lambda b: b.to(torch_.float64)
        self.riv_isr_f, self.riv_oth_f = f(isr), f(~isr)
        # the records' stage with no discharge: records that are not rivers
        # (drains: stage at their start head) and rivers off the VIC grid
        from ...vicwur import groundwater as W
        self.rec_stage0 = W.stage(self.riv, torch_.zeros(
            nrec, dtype=torch_.float64, device=dev), False)
        r = self.router
        full = lambda v, fill: torch_.full((n,), fill, dtype=torch_.float64,
                                           device=dev).index_copy(
            0, self.riv_cell, v.index_select(0, self.riv_idx))
        den = self.riv['width'] * self.riv['slope_used'] ** 0.5
        den = torch_.where(torch_.isnan(den), torch_.full_like(den, float(
            'inf')), den)
        cap = (torch_.nan_to_num(self.riv['d_bkfl'], nan=1e30)
               if self.stage_cap else torch_.full_like(den, 1e30))
        st = dict(C=full(self.riv_cond0 / 86400.0, 0.0),
                  bot=full(self.riv['bot_bkfl'], 0.0),
                  den=full(den, 1.0), cap=full(cap, 0.0),
                  rbot=full(self.riv['rbot'], 0.0))
        self.rx = RX.RiverExchange(r, {k: r.to_sweep(v)
                                       for k, v in st.items()},
                                   self.xch_iter)
        self._T_day = self.zero()
        self._unmet_day = self.zero()
        self._rx_sync()

    def _rx_sync(self):
        """The slot's head and debt buffers from the state."""
        if self.rx is None:
            return
        r = self.router
        self.rx.h.copy_(r.to_sweep(self.model.h[0].reshape(-1)[self.flat]))
        self.rx.debt.copy_(r.to_sweep(self.debt))

    # sweep slot (mode fixed, daily; see torchvic.vicwur.river_exchange)
    def sweep_begin(self):
        if self.rx is not None:
            self.rx.begin()

    def sweep_slot(self, lv, slot, cells):
        if slot == 'river_exchange' and self.rx is not None:
            self.rx.level(lv, cells)

    def sweep_end(self):
        if self.rx is None:
            return
        G, T = self.group, self.rx.T
        z = torch.zeros_like(T)
        G.add('ring_out', torch.where(T > 0, T, z).sum() * STEP_S)
        G.add('ring_in', torch.where(T < 0, -T, z).sum() * STEP_S)
        self._T_day = T.sum() * STEP_S
        self._unmet_day = self.rx.unmet.sum() * STEP_S

    def graph_key(self):
        if self.rx is None:
            return None
        return tuple(t.data_ptr() for t in self.rx.graph_ptrs())

    def river_force(self):
        """The forced inflow (m3 s-1, cell order) the routing gets this
        step: VIC-WUR 3.0's DISCHARGE forcing of the period."""
        return self.force

    def _day(self, step):
        import datetime
        return self.date0 + datetime.timedelta(days=int(step))

    def _period_end(self, step):
        if self.period == 'day':
            return True
        return self._day(step + 1).day == 1

    def _next_len(self, step):
        """Days of the period that starts after `step`."""
        if self.period == 'day':
            return 1
        import calendar
        d = self._day(step + 1)
        return calendar.monthrange(d.year, d.month)[1]

    def put_data(self):
        from ...vicwur import groundwater as W
        G, v = self.group, self.view
        if v is None or v.gwrecharge is None:
            raise ComputeError('groundwater (vicwur3): the step view carries '
                               'no gwrecharge (OUT_GWRECHARGE)')
        z = self.zero()
        gw = as_f64(v.gwrecharge, self.dev)
        # mode fixed: the unused part of the water the wells pumped goes back
        # to the aquifer (vicwur_nonrenew return_to 'aquifer'), m3
        nr = G.take('nr_return')
        nr_in = z if isinstance(nr, float) else nr.sum()        # m3
        if not isinstance(nr, float):
            gw = gw + nr / self.ca * 1000.      # mm of the aquifer cell
        rin = (gw * self.ca).sum() / 1000.
        self.acc_gw += gw
        self.pend_rch = self.pend_rch + rin
        if self.router is not None:
            self.acc_q += self.router.to_cells(self.router.discharge)
            # the forced inflow (drains, records that are not rivers) and the
            # river exchange of the sweep (loss: water left the river)
            forced = self.force.sum() * STEP_S
            delivered = forced - (self._T_day if self.xch else 0.0)
            self.pend_bf = self.pend_bf - delivered
        else:
            delivered = forced = z
        pump = G.take('gw_pump')
        pump_out = z
        if not isinstance(pump, float):
            if not self.exact:
                self.acc_pump += pump
                pump_out = pump.sum()
                self.pend_pump = self.pend_pump + pump_out
            # exact: VIC-WUR 3.0 GWM, the withdrawal has no source (untracked)
        self.ndays += 1
        S0 = self.pend_rch - rin + self.pend_bf + delivered \
            - (self.pend_pump - pump_out) + self.pend_carry
        T = dict(I=rin, Q=delivered + pump_out, dS_aq=z, chd_in=z,
                 chd_out=z, cpr=z, clo=z, clip=z, lost=z, held=z, bf_out=z)
        if self._period_end(G.step):
            self._mf_step(G.step, T)
        S1 = self.pend_rch + self.pend_bf - self.pend_pump + self.pend_carry
        dS = T['dS_aq'] + (S1 - S0)
        clo = T['clo'] + T['clip'] + T['lost'] + T['held']
        I_ext = T['I'] + T['chd_in']
        Q_out = T['Q'] + T['cpr'] + T['chd_out'] + T['bf_out']
        scale = (I_ext.abs() + Q_out.abs() + T['dS_aq'].abs() + S0.abs()
                 + S1.abs() + clo.abs())
        bal = Balance(I_ext=I_ext, Q_out=Q_out, dS=dS, closure=clo,
                      n_terms=self.N * 64, scale=scale,
                      detail=dict(recharge=rin, nonrenew_return=nr_in,
                                  river=delivered,
                                  river_forced=forced,
                                  river_exchange=(-self._T_day if self.xch
                                                  else z),
                                  pumping=pump_out, capillary_rise=T['cpr'],
                                  chd_in=T['chd_in'], chd_out=T['chd_out'],
                                  aquifer_dS=T['dS_aq'],
                                  engine_discrepancy=T['clo'],
                                  recharge_not_applied=T['clip'],
                                  cpr_top_excess_lost=T['lost'],
                                  held_rate_mismatch=T['held'],
                                  baseflow_no_router=T['bf_out'],
                                  river_exchange_unmet=(
                                      self._unmet_day if self.xch else z)))
        h = self.model.h
        hv = lambda k: h[k].reshape(-1)[self.flat]
        exp = dict(head=hv(0), head2=hv(1),
                   baseflow=(self.bf_rate if self.xch else self.force).clone(),
                   capillary_rise=self.cpr_mm.clone())
        ser = {k: x.clone().unsqueeze(0) for k, x in exp.items()}
        self._pending = (bal, exp, ser)

    def _mf_step(self, step, T):
        """One MODFLOW time step over the period that ends with `step`."""
        from ...vicwur import groundwater as W
        m, nd = self.model, float(self.ndays)
        dev = self.dev
        R, C = self.R, self.C
        full = lambda x, f=0.0: torch.full((R * C,), f, dtype=torch.float64,
                                           device=dev).index_put(
            (self.flat,), x)
        # RCH: the period's OUT_GWRECHARGE (+ fixed: the excess carried)
        gw = self.acc_gw + self.carry
        rate = W.recharge_rate(gw, nd, self.ca, self.delr, self.delc,
                               self.exact)
        m.set_recharge(full(rate).view(R, C))
        self.pend_rch_step = self.pend_rch + self.pend_carry
        self.pend_rch = self.pend_rch * 0
        self.pend_carry = self.pend_carry * 0
        self.carry = torch.zeros_like(self.carry)
        # RIV stage: the stage the sweep ended on (river exchange), or the
        # stage of the period's mean discharge (VIC-WUR 3.0, daily steps)
        if self.riv_flat.numel():
            if self.xch:
                stg = self.rec_stage0.index_copy(
                    0, self.riv_idx, self.router.to_cells(
                        self.rx.stg).index_select(0, self.riv_cell))
            else:
                q = full(self.acc_q / nd)[self.riv_flat]
                stg = W.stage(self.riv, q, self.exact)
            m.set_rivers(stage=stg)
        if not self.exact:
            m.set_wells(q=-self.acc_pump / nd)
        res = self._solve(nd)
        fl, b = res.flows, res.budget
        z = self.zero()
        pk = lambda k, i: b[k][i] if k in b else z
        sto = (fl['STO-SS'].sum() + fl['STO-SY'].sum()) * nd \
            if 'STO-SS' in fl else z
        T['dS_aq'] = -sto
        # recharge the aquifer took: exact clips negative values (VIC-WUR 3.0),
        # and MODFLOW applies none on a constant-head (CHD) cell
        applied = (pk('RCH', 0) - pk('RCH', 1)) * nd
        T['clip'] = self.pend_rch_step - applied
        T['chd_in'], T['chd_out'] = pk('CHD', 0) * nd, pk('CHD', 1) * nd
        tin = sum(pk(k, 0) for k in b)
        tout = sum(pk(k, 1) for k in b)
        T['clo'] = (tin - tout) * nd
        # wells: the pending pumping is taken; what a well could not pump
        # (none on a CHD cell) stays requested
        wel = fl['WEL'] if 'WEL' in fl else None
        if wel is not None:
            self.pend_pump = self.pend_pump + wel.sum() * nd
        # RIV + DRN per cell (rates, m3/d, + into the aquifer); with the
        # river exchange the river cells' RIV is the sweep's business
        qrd = torch.zeros(R * C, dtype=torch.float64, device=dev)
        Fr = torch.zeros(self.N, dtype=torch.float64, device=dev)
        if 'RIV' in fl:
            if self.xch:
                qrd = _ordered.index_add(qrd, 0, self.riv_flat,
                                         fl['RIV'] * self.riv_oth_f)
                Fr = _ordered.index_add(
                    torch.zeros(R * C, dtype=torch.float64, device=dev), 0,
                    self.riv_flat, fl['RIV'] * self.riv_isr_f)[self.flat]
            else:
                qrd = _ordered.index_add(qrd, 0, self.riv_flat, fl['RIV'])
        if 'DRN' in fl:
            qrd = _ordered.index_add(qrd, 0, self.drn_flat, fl['DRN'])
        qc = qrd[self.flat]
        vol = -qc * nd                                 # m3 to the river
        nxt = self._next_len(step)
        vol_rel = vol.sum()           # the aquifer's release this step
        if self.exact:
            # mf_run.py:319-320 (without the 1/days): a rate held over the
            # next month, whatever its length
            frc = -qc / 86400
            frc = torch.where(frc.abs() > 1e8, torch.zeros_like(frc), frc)
        elif self.xch:
            # drains and records that are not rivers only gain water: the
            # forced inflow of the next day. The river cells' exchange was
            # made in the sweep (T); what the aquifer took or gave beyond it
            # is the debt the next sweep pays.
            frc = vol / (86400.0 * nxt)
            T_cell = self.router.to_cells(self.rx.T) * STEP_S
            self.debt = self.debt + Fr * nd - T_cell
            self._rx_sync()
            vol_rel = vol_rel - Fr.sum() * nd
        else:
            frc = vol / (86400.0 * nxt)          # the volume, handed on whole
        if self.router is not None:
            # held: the volume the next period will not deliver (exact: the
            # held rate over a month of another length; fixed: rounding)
            # plus what the last period left undelivered (rounding)
            nxt_vol = (frc * 86400 * nxt).sum() - (
                self.debt.sum() if self.xch else 0.0)
            T['held'] = self.pend_bf + (vol_rel - nxt_vol)
            self.pend_bf = nxt_vol
        else:
            T['bf_out'] = vol_rel
        self.force = frc
        # the step's net exchange with the rivers (m3 s-1, + to the river):
        # with the river exchange the river cells' RIV is in it too
        self.bf_rate = -(qc + (Fr if self.xch else 0.0)) / 86400
        # EVT -> capillary rise into the soil (mf_run.py:349-377,
        # vic_runner.py:267-321)
        self.cpr_mm = torch.zeros_like(self.cpr_mm)
        if self.foc and 'EVT' in fl:
            ev = fl['EVT']                     # per column, into the aquifer
            evt_nd = self.model._rch_nodes(self.model.ibound)
            evc = torch.zeros(R * C, dtype=torch.float64,
                              device=dev).index_add(
                0, evt_nd % (R * C), ev)[self.flat]
            vcpr = -evc * nd                                   # m3, >= 0
            self.cpr_mm = vcpr / self.ca * 1000.
            self._capillary(vcpr, T)
        # the step's exchange per cell (rates, m3/d, + into the aquifer),
        # for diagnostics and tests; no host read
        cell = lambda name, idx: (_ordered.index_add(torch.zeros(
            R * C, dtype=torch.float64, device=dev), 0, idx,
            fl[name])[self.flat] if name in fl
            else torch.zeros(self.N, dtype=torch.float64, device=dev))
        self.last = dict(RIV=cell('RIV', self.riv_flat),
                         DRN=cell('DRN', self.drn_flat),
                         EVT=-self.cpr_mm * self.ca / 1000. / nd,
                         RCH=(fl['RCH'][self.flat] if 'RCH' in fl
                              else torch.zeros_like(self.ca)),
                         WEL=(wel if wel is not None
                              else torch.zeros_like(self.ca)),
                         head=m.h.reshape(2, -1)[:, self.flat].clone(),
                         days=nd)
        self.acc_gw = torch.zeros_like(self.acc_gw)
        self.acc_q = torch.zeros_like(self.acc_q)
        self.acc_pump = (self.acc_pump + wel * nd if wel is not None
                         else torch.zeros_like(self.acc_pump))
        self.ndays = 0
        self.nstep += 1

    def _capillary(self, vcpr, T):
        from ...vicwur import groundwater as W
        v = self.view
        mx = self.max_moist
        if mx is None:
            mx = v.max_moist[:, 0, 0, :] if v.max_moist.dim() == 4 \
                else v.max_moist
        base = v.moist
        live = v.live[..., 0]
        new, lost = W.capillary_rise(base, v.cv, live, self.cpr_mm, mx,
                                     self.cpr_layer, self.exact)
        d = new - base
        v.delta = v.delta + d
        put = v.cell_total(d)                                  # mm
        T['cpr'] = (put * self.ca).sum() / 1000.
        lost_mm = (lost * v.af).sum(1)
        lv = (lost_mm * self.ca).sum() / 1000.
        if self.exact:
            T['lost'] = lv
        else:
            # returned to the aquifer with the next period's recharge
            self.carry = self.carry + lost_mm
            self.pend_carry = self.pend_carry + lv

    def _solve(self, nd):
        m = self.model
        if self.nstep < self.sync_steps or self._sync_next:
            try:
                res = m.step(nd)
            except Exception as e:                       # noqa: BLE001
                raise ComputeError('groundwater (vicwur3): MODFLOW step '
                                   'failed: %s: %s' % (type(e).__name__,
                                                       e)) from e
            self._sync_next = False
            self._conv = None
            return res
        res = m.step(nd, defer=True)
        c = res.converged
        self._conv = c if self._conv is None else (self._conv & c)
        return res

    def _redo(self):
        self._sync_next = True

    def compute(self, inputs, context, dt):
        p = super().compute(inputs, context, dt)
        if self._conv is not None:
            p.flags.append(DeviceFlag(
                'groundwater (vicwur3): MODFLOW did not converge in the '
                'chunks launched ahead; the step is redone synchronously',
                ~self._conv.bool(), retry=self._redo))
            self._conv = None
        if self.model is not None:
            p.flags.append(DeviceFlag('groundwater (vicwur3): non-finite '
                                      'heads',
                                      ~torch.isfinite(self.model.h).all()))
        return p

    # -- state ---------------------------------------------------------------
    _T = ('acc_gw', 'acc_q', 'acc_pump', 'carry', 'force', 'debt', 'cpr_mm',
          'bf_rate', 'pend_rch', 'pend_bf', 'pend_pump', 'pend_carry')

    def _state(self):
        st = self.model.state_dict()
        d = {k: getattr(self, k).clone() for k in self._T}
        d.update(h=st['h'].clone(), ibound=st['ibound'].clone(),
                 time=st['time'], kstp=st['kstp'], ndays=self.ndays,
                 nstep=self.nstep, solver=self.model.solver_state())
        return d

    def _load(self, blob):
        self.model.load_state_dict(dict(h=blob['h'], ibound=blob['ibound'],
                                        time=blob['time'],
                                        kstp=blob['kstp']))
        if 'solver' in blob:
            self.model.load_solver_state(blob['solver'])
        for k in self._T:
            setattr(self, k, blob[k].clone().to(self.dev))
        self.ndays, self.nstep = int(blob['ndays']), int(blob['nstep'])
        self._conv = None
        self._rx_sync()

    def finalize(self, context):
        pass
