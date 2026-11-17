"""QualityConnector: temperature and constituents riding on the router.

The engine is `torchvic.quality.WaterQuality`: every constituent is routed
with the same unit-hydrograph machinery as the water (mixing at confluences
is automatic), with first-order decay exact along the flow path; stream
temperature is advected heat with the surface exchange of
`quality/heat.py` (Edinger et al. 1968 linearisation about the equilibrium
temperature). One `WaterQuality.step` is the routing step itself (begin,
level-wise withdraw / inject, end) plus the constituents and heat, so this
connector is a sweep plugin of the routing connector (see
`connectors/routing.py`): it stages its inputs, the routing step runs the
engine, and the routing connector completes this connector's proposal.

Configuration:

    constituents   {name: decay rate in 1/s} (0 for a conservative tracer)
    temperature    {depth: m, k_ref: W m-2 K-1, albedo, shade}; omit for none
    loads          {name: value} mass added per step per cell (scalar or a
                   list per cell), a default for the `load_<name>` input
    met            {t_air: C, vp: kPa, wind: m s-1, sw: W m-2} scalars or
                   per-cell lists, defaults for the inputs of the same names
    t_runoff       temperature of the runoff, C (default: air temperature
                   floored at 0)
    mass_tol       relative tolerance of the mass and heat ledgers per step

Inputs (all `state`: a value sampled at the step; mass is per routing step):
    load_<name> [cell], t_air, vp, wind, sw [cell], t_runoff [cell]
    point_<name> [site] mass per step at sites
    inject_conc_<name> [site] mass per m3 of the injected water,
    inject_temp [site] C of the injected water (with the routing module's
    `injection`)
Outputs (state): conc_<name> [node] mass m-3 and temperature [node] C at the
gauges (0 where the flow is below `qmin`).

With a reservoir module on the same router the sweep is the combined one
(`_combined.py`): each dam is a well-mixed store of constituents and heat.

Ledgers: per node, `in_runoff + in_hooks - out - stored` of every
constituent and of heat is zero to rounding (the decayed mass is the engine's
`lost`, counted in `in_runoff`'s complement exactly as `WaterQuality.ledger`
documents). `validate` rejects the step when any of them exceeds `mass_tol`
of the node's total. The constituent mass is not in the water ledger (mass
is not water); `mass_ledger()` returns the cumulative per-node residuals,
decay losses and atmosphere heat.
"""
import numpy as np
import torch

from ..connector import ComputeError, Connector, DeviceFlag, Proposal
from ..spec import CouplingError, GridSpec, VarSpec
from ...model import routing_fast as rf
from ...quality import WaterQuality
from .routing import CELLS, NODES, SITES, SweepPlugin

_MET = (('t_air', 'C'), ('vp', 'kPa'), ('wind', 'm s-1'), ('sw', 'W m-2'))


class QualityConnector(SweepPlugin, Connector):
    name = 'quality'
    version = '1'
    backends = ('cpu', 'cuda')
    checkpoint = 'full'
    graph_safe = False
    validation = ('the WaterQuality engine (tests/test_quality.py: decay '
                  'analytic, conservative tracer, mixing, withdrawals, heat '
                  'equilibrium and ledger); no calibration against '
                  'observed water quality')

    def __init__(self, constituents=None, temperature=None, loads=None,
                 met=None, t_runoff=None, routing='routing', step_s=86400.0,
                 grid=CELLS, mass_units='kg', mass_tol=1e-9, name=None,
                 **config):
        super().__init__(**config)
        if name:
            self.name = name
        self.cons = {k: float(v) for k, v in (constituents or {}).items()}
        self.temp_cfg = dict(temperature) if temperature else None
        if not self.cons and self.temp_cfg is None:
            raise CouplingError('quality: give constituents and / or '
                                'temperature')
        self.loads_cfg = dict(loads or {})
        self.met_cfg = dict(met or {})
        self.t_runoff_cfg = t_runoff
        self.routing_name = routing
        self.interval_s = float(step_s)
        self.grid = grid
        self.mass_tol = float(mass_tol)
        self.units = mass_units
        self.wanted = set()
        self.peers = None
        self.wq = None
        self.staged = None
        self._prop = None
        ins = [VarSpec('load_' + n, mass_units, ('cell',), kind='state',
                       grid=grid, description='mass of %s added per step' % n)
               for n in self.cons]
        ins += [VarSpec('point_' + n, mass_units, ('site',), kind='state',
                        grid=SITES, description='point source of %s per step'
                        % n) for n in self.cons]
        ins += [VarSpec('inject_conc_' + n, mass_units + ' m-3', ('site',),
                        kind='state', grid=SITES,
                        description='concentration of %s in injected water'
                        % n) for n in self.cons]
        if self.temp_cfg is not None:
            ins += [VarSpec(k, u, ('cell',), kind='state', grid=grid,
                            description='forcing for the equilibrium '
                                        'temperature') for k, u in _MET]
            ins += [VarSpec('t_runoff', 'C', ('cell',), kind='state',
                            grid=grid, description='runoff temperature'),
                    VarSpec('inject_temp', 'C', ('site',), kind='state',
                            grid=SITES, description='temperature of '
                                                    'injected water')]
        self.inputs = tuple(ins)
        outs = [VarSpec('conc_' + n, mass_units + ' m-3', ('node',),
                        kind='state', grid=NODES,
                        description='concentration of %s at the gauges' % n)
                for n in self.cons]
        if self.temp_cfg is not None:
            outs.append(VarSpec('temperature', 'C', ('node',), kind='state',
                                grid=NODES,
                                description='water temperature at the '
                                            'gauges'))
        self.outputs = tuple(outs)

    @property
    def owned_state(self):
        return (self.name + '.rings',)

    def grids(self):
        return []

    def initialize(self, context):
        pass

    def site_cells(self):
        return ()

    def attach(self, routing, site_index, area):
        r = routing.router
        if abs(routing.interval_s - self.interval_s) > 1e-6:
            raise CouplingError('%s: step %g s differs from the routing '
                                'step %g s' % (self.name, self.interval_s,
                                               routing.interval_s))
        self.routing = routing
        self.dev, self.dtype = routing.dev, routing.dtype
        self.sites = routing.sites
        sc = self.sites if len(self.sites) else None
        self.combined = routing.res_conn is not None
        if self.combined:
            from ._combined import ReservoirQuality
            self.wq = ReservoirQuality(
                r, area, dict(self.cons), self.temp_cfg, sc,
                routing.res_conn.engine)
            self.wq.outlets = routing.outlet_idx
        else:
            self.wq = WaterQuality(
                r, area, dict(self.cons) or None,
                temperature=self.temp_cfg, site_cells=sc)
        w = self.wq
        if w.uniform:
            b = w.bq
            ts = [b.ring_all, b.in_runoff_all, b.lost_all, b.hooks_all,
                  b.out_all, b.wd_all, b.inj_all]
        else:
            ts = [w.ring_all, w.hooks_all, w.wd_all, w.inj_all]
            for c in w.all:
                ts += [c.head, c.in_runoff, c.out, c.lost]
        if self.combined:
            ts += w.state_tensors()
        self._tensors = ts
        self.n_cell = routing.n_cell

    def state_tensors(self):
        return self._tensors

    def after_restore(self):
        self.staged = None

    def accept(self):
        pass

    # -- inputs ------------------------------------------------------------
    def _cell_value(self, v, n):
        if v is None:
            return None
        if not torch.is_tensor(v):
            # a configured constant: built on the device once (a fresh
            # host-to-device copy per step would synchronise the stream)
            c = self.__dict__.setdefault('_const_cache', {})
            key = (float(v) if np.ndim(v) == 0 else None, n)
            if key[0] is not None and key in c:
                return c[key]
            t = torch.as_tensor(v, dtype=self.dtype, device=self.dev)
            t = t.expand(n).contiguous() if t.dim() == 0 else t
            if key[0] is not None:
                c[key] = t
            return t
        t = torch.as_tensor(v, dtype=self.dtype, device=self.dev)
        return t.expand(n).contiguous() if t.dim() == 0 else t

    def compute(self, inputs, context, dt):
        if self.wq is None:
            raise ComputeError('%s: not attached to a routing connector'
                               % self.name)
        n, ns = self.n_cell, self.routing.n_site
        loads = {}
        for k in self.cons:
            v = inputs.get('load_' + k)
            v = self.loads_cfg.get(k, 0.0) if v is None else v
            loads[k] = self._cell_value(v, n)
        point, conc = {}, {}
        for k in self.cons:
            v = inputs.get('point_' + k)
            if v is not None:
                point[k] = self._cell_value(v, ns)
            v = inputs.get('inject_conc_' + k)
            if v is not None:
                conc[k] = self._cell_value(v, ns)
        met = tr = it = None
        if self.temp_cfg is not None:
            met = {}
            for k, _u in _MET:
                v = inputs.get(k, self.met_cfg.get(k))
                if v is None:
                    raise ComputeError('%s: temperature needs %s (an input '
                                       'or met: %s: in the configuration)'
                                       % (self.name, k, k))
                met[k] = self._cell_value(v, n)
            tr = inputs.get('t_runoff', self.t_runoff_cfg)
            tr = self._cell_value(tr, n)
            it = inputs.get('inject_temp')
            it = self._cell_value(it, ns)
        self.staged = dict(t0=context.t, loads=loads, point=point or None,
                           conc=conc or None, met=met, t_runoff=tr,
                           inject_temp=it)
        self._prop = Proposal(connector=self.name, t0=context.t,
                              t1=context.t + dt, state=None, exports={},
                              balance=None, series={},
                              backend=str(self.dev))
        return self._prop

    def run(self, routing, infl, req, inj, t):
        st = self._staged_for(t)
        if self.combined:
            self.wq._ext = routing.res_conn.begin(t)
            if inj is None:         # the combined sweep always runs
                inj = torch.zeros(routing.n_site, dtype=self.dtype,
                                  device=self.dev)
        out = self.wq.step(infl, loads=st['loads'], point=st['point'],
                           met=st['met'], t_runoff=st['t_runoff'],
                           withdraw=req, inject=inj,
                           inject_conc=st['conc'],
                           inject_temp=st['inject_temp'])
        return out['q'], out['got'], out

    def _ledger_ratios(self):
        """{name: (max |residual| over nodes, scale)} device scalars."""
        w = self.wq
        led = w.ledger()
        res = {}
        for k, r in led.items():
            if k == 'water':
                continue
            if k == 'heat':
                chans = list(w.temp)
            else:
                chans = [w.ch[k]]
            sc = sum((c.in_runoff.abs() + c.in_hooks.abs() + c.out.abs()
                      + c.lost.abs()).max() for c in chans)
            res[k] = (r.abs().max(), sc)
        return res

    def complete(self, context, dt, out):
        p = self._prop
        ex = {}
        for k in self.cons:
            ex['conc_' + k] = out['conc'][k]
        if self.temp_cfg is not None:
            ex['temperature'] = out['t']
        p.exports = ex
        p.series = {k: v.unsqueeze(0) for k, v in ex.items()}
        rat = self._ledger_ratios()
        p.info = dict(ratios={k: torch.stack([a, b])
                              for k, (a, b) in rat.items()})

    def validate(self, proposal, context):
        bad = []
        for k, ab in proposal.info.get('ratios', {}).items():
            a, b = ab[0], ab[1]
            bad.append(DeviceFlag(
                '%s: %s ledger residual %%.3e exceeds %.1e of %%.3e'
                % (self.name, k, self.mass_tol),
                ~(a <= self.mass_tol * b.clamp(min=1e-300)) & (a > 0),
                values=(a, b)))
        for k, v in proposal.exports.items():
            bad.append(DeviceFlag('%s: non-finite %s' % (self.name, k),
                                  ~torch.isfinite(v).all()))
        return bad

    def commit(self, proposal, context):
        self.staged = None

    def save_state(self):
        self.routing._rollback()
        return dict(tensors=[t.cpu() for t in self._tensors])

    def load_state(self, blob):
        for t, s in zip(self._tensors, blob['tensors']):
            t.copy_(s.to(self.dev))
        self.staged = None

    # -- for users -------------------------------------------------------------
    def mass_ledger(self):
        """{name: per-node residual [node]} of water, every constituent and
        heat, plus `decayed_<name>` [node] and `atmosphere_heat` [node]."""
        self.routing._rollback()
        out = dict(self.wq.ledger())
        if self.combined:
            out['reservoir_store'] = self.wq.store_residual()
        for k in self.cons:
            out['decayed_' + k] = self.wq.decayed(k)
        if self.temp_cfg is not None:
            out['atmosphere_heat'] = self.wq.atmosphere_heat()
        return out
