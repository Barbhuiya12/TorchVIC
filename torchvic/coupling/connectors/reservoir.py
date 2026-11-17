"""ReservoirConnector: dams on the river network, operated inside the
routing step.

The engine is `torchvic.reservoir.Reservoirs` (Hanasaki-type release rules,
target-storage rule, run of river; dead storage, environmental flow,
evaporation, spill). A reservoir sits at a router site: each step it takes
the water reaching the site (`StepRouter.sweep` semantics: never more than
passes without driving a ring slot negative), updates its storage, and
injects release + spill back at the same site, level by level so a dam
downstream sees the release of the dam above it in the same step.

This connector

  * loads the dam table and registers the dam cells as router sites
    (`site_cells()`; `RoutingConnector.initialize` builds the router with
    them and calls `attach`),
  * owns the storage (`owned_state`): state, checkpoint and calendar are
    here,
  * stages its inputs and, in the routing step that runs after it, has its
    proposal completed: storage, supplied water, balance, series.

Dams (list of mappings, or a csv with the same columns):

    cell          index of the river cell (the router's cell numbering)
    capacity_m3   (or capacity_hm3) total storage
    dead_m3       dead storage (default 0)
    q_env         minimum release, m3 s-1 (default 0)
    q_max         maximum rule release, m3 s-1 (default unlimited)
    build_year    active from this year (default always)
    init_frac     initial storage / capacity (default 0.5)
    irrigation    true: the Hanasaki irrigation branch
    area_max, area_exp   surface area at full storage (m2) and exponent
                  (default 0: no evaporation)
    rule          hanasaki | target | pass (default hanasaki)
    target_frac   12 monthly target fractions (rule target)

Inputs:  request  state m3 per routing step, water-use demand on each dam
         evap     state m, potential evaporation depth per step (default: the
             configuration's `evap_m`, a scalar or one value per dam)
Outputs: storage  state m3 at the step end
         supplied flux  m3 delivered to `request` (a conservative transfer)
         (release and spill go back into the river inside the sweep; they
         are in `series`.)

Balance per step: inflow I (water taken from the river) =
evaporation + supplied + release + spill + dS; a dam not yet built passes
its inflow through unchanged (in and out). The ledger of the routing module
counts the take as water leaving to a connector and the injection as water
entering from one, so the transfers cancel in the coupled balance.

Calendar: the release rules are monthly. `start_date` (YYYY-MM-DD) is the
date of t = 0; the engine's `new_month(month, year)` runs at the first step
and whenever the step's date enters a new month.
"""
import datetime

import numpy as np
import torch

from ..connector import (Balance, ComputeError, Connector, DeviceFlag,
                         Proposal)
from ..spec import CouplingError, GridSpec, VarSpec
from ...reservoir import (Reservoirs, ReservoirParams, RULE_HANASAKI,
                          RULE_PASS, RULE_TARGET)
from .routing import SweepPlugin

GRID = 'reservoirs'
_RULES = {'hanasaki': RULE_HANASAKI, 'target': RULE_TARGET,
          'pass': RULE_PASS}


def _dams_from_csv(path):
    import csv
    out = []
    for row in csv.DictReader(open(path)):
        d = {}
        for k, v in row.items():
            if v is None or v == '':
                continue
            if k in ('rule',):
                d[k] = v
            elif k == 'irrigation':
                d[k] = v.strip().lower() in ('1', 'true', 'yes')
            elif k == 'target_frac':
                d[k] = [float(x) for x in v.split()]
            else:
                d[k] = float(v)
        out.append(d)
    return out


class ReservoirConnector(SweepPlugin, Connector):
    name = 'reservoir'
    version = '1'
    backends = ('cpu', 'cuda')
    checkpoint = 'full'
    graph_safe = False
    validation = ('the Reservoirs engine (tests/test_reservoir.py: storage '
                  'transitions, limits, conservation on the router, restart '
                  'equality); no calibration against observed storage')
    imported_fluxes = ()
    exported_fluxes = ('supplied',)

    def __init__(self, dams=None, routing='routing', start_date='2000-01-01',
                 step_s=86400.0, relax_steps=10.0, evap_m=None, request_m3=None, inflow_clim=None,
                 demand_clim=None, clim_years=1.0, name=None, **config):
        super().__init__(**config)
        if name:
            self.name = name
        self.dams_cfg = dams
        self.routing_name = routing
        self.start = datetime.datetime.strptime(str(start_date)[:10],
                                                '%Y-%m-%d')
        self.interval_s = float(step_s)
        self.relax_steps = float(relax_steps)
        self.request_m3 = request_m3  # default water-use demand, m3 / step
        self.evap_m = evap_m          # default potential evaporation, m / step
        self.clim = (inflow_clim, demand_clim, float(clim_years))
        self.wanted = set()
        self.peers = None
        self.engine = None
        self.staged = None
        self._prop = None
        self._cal = None
        self._cal_next = None
        self.rows = []
        self.R = 0
        self.inputs = (
            VarSpec('request', 'm3', ('reservoir',), kind='state',
                    grid=GRID, description='water-use demand on each '
                    'reservoir per routing step'),
            VarSpec('evap', 'm', ('reservoir',), kind='state', grid=GRID,
                    description='potential evaporation depth per step'))
        self.outputs = (
            VarSpec('storage', 'm3', ('reservoir',), kind='state',
                    grid=GRID, description='storage at the step end'),
            VarSpec('supplied', 'm3', ('reservoir',), kind='flux',
                    grid=GRID, description='water delivered to request'))

    @property
    def owned_state(self):
        return (self.name + '.storage',)

    def grids(self):
        return [GridSpec(GRID, self.R, kind='reservoirs')]

    # -- data ----------------------------------------------------------------
    def initialize(self, context):
        dams = self.dams_cfg
        if isinstance(dams, str):
            dams = _dams_from_csv(dams)
        if not dams:
            raise CouplingError('%s: no dams (give dams: a list or a csv)'
                                % self.name)
        R = self.R = len(dams)
        g = lambda k, d: np.array([float(x.get(k, d)) for x in dams])
        cap = np.array([float(x['capacity_m3']) if 'capacity_m3' in x
                        else float(x['capacity_hm3']) * 1e6 for x in dams])
        cells = np.array([int(x['cell']) for x in dams], dtype=np.int64)
        if len(np.unique(cells)) != R:
            raise CouplingError('%s: two dams in one cell (merge them: one '
                                'reservoir per site)' % self.name)
        rule = np.array([_RULES[str(x.get('rule', 'hanasaki')).lower()]
                         for x in dams])
        tf = np.ones((R, 12))
        for i, x in enumerate(dams):
            if 'target_frac' in x:
                tf[i] = np.asarray(x['target_frac'], dtype=float)
        self.cells = cells
        self.params_kw = dict(
            capacity=cap, dead=g('dead_m3', 0.0), q_env=g('q_env', 0.0),
            q_max=g('q_max', np.inf), build_year=g('build_year', -1e9),
            init_frac=g('init_frac', 0.5), area_max=g('area_max', 0.0),
            area_exp=g('area_exp', 1.5), rule=rule, target_frac=tf,
            irrigation=np.array([bool(x.get('irrigation', False))
                                 for x in dams]),
            relax_steps=self.relax_steps)
        if any(c is not None for c in self.clim[:2]):
            self.params_kw.update(inflow_clim=self.clim[0],
                                  demand_clim=self.clim[1],
                                  clim_years=self.clim[2])

    def site_cells(self):
        return self.cells

    def attach(self, routing, site_index, area):
        """Called by the routing connector once its router exists."""
        if abs(routing.interval_s - self.interval_s) > 1e-6:
            raise CouplingError('%s: step %g s differs from the routing '
                                'step %g s' % (self.name, self.interval_s,
                                               routing.interval_s))
        site = np.array([site_index[int(c)] for c in self.cells])
        p = ReservoirParams(site=site, **self.params_kw)
        self.engine = Reservoirs(p, routing.router, dtype=routing.dtype)
        self.routing = routing
        self.dev, self.dtype = routing.dev, routing.dtype
        self._tensors = [getattr(self.engine, k)
                         for k in self.engine._STATE]
        self.cap = self.engine.cap
        self.engine._ext_zero = torch.zeros(self.R, dtype=self.dtype,
                                            device=self.dev)

    # -- state shared with the routing transaction -----------------------------
    def state_tensors(self):
        return self._tensors

    def after_restore(self):
        self.engine._refresh(int(self.engine.month))
        self.staged = None

    def accept(self):
        self._cal = self._cal_next

    # -- the step --------------------------------------------------------------
    def compute(self, inputs, context, dt):
        if self.engine is None:
            raise ComputeError('%s: not attached to a routing connector'
                               % self.name)
        z = self.engine._ext_zero
        req, ev = inputs.get('request'), inputs.get('evap')
        for nm, v in (('request', req), ('evap', ev)):
            if v is not None and torch.as_tensor(v).shape[0] != self.R:
                raise ComputeError('%s: %s has %d reservoirs, there are %d'
                                   % (self.name, nm,
                                      torch.as_tensor(v).shape[0], self.R))
        self.staged = dict(
            t0=context.t,
            request=(None if req is None else torch.as_tensor(
                req, dtype=self.dtype, device=self.dev)) if req is not None
            or self.request_m3 is None else self._const('request_m3'),
            evap=(None if ev is None else torch.as_tensor(
                ev, dtype=self.dtype, device=self.dev)) if ev is not None
            or self.evap_m is None else self._const('evap_m'))
        self._prop = Proposal(connector=self.name, t0=context.t,
                              t1=context.t + dt, state=None, exports={},
                              balance=None, series={},
                              backend=str(self.dev))
        return self._prop

    def _const(self, attr):
        """A configured default (request_m3, evap_m) on the device, built
        once: a host-to-device copy per step would sync the stream."""
        c = self.__dict__.setdefault('_const_cache', {})
        v = getattr(self, attr)
        key = (attr, repr(v))
        if key not in c:
            c[key] = torch.as_tensor(
                v, dtype=self.dtype, device=self.dev).expand(
                    self.R).contiguous()
        return c[key]

    def begin(self, t):
        """Calendar, the opening storage and the step's (request, evap)
        [R, 2] for the sweep (as `Reservoirs.apply` forms it)."""
        st = self._staged_for(t)
        eng = self.engine
        d = self.start + datetime.timedelta(seconds=float(t))
        key = (d.year, d.month)
        if self._cal != key:
            eng.new_month(d.month - 1, d.year)
        self._cal_next = key
        self._S0 = eng.storage.clone()
        if st['request'] is not None:
            self._prop.flags.append(DeviceFlag(
                '%s: request must not be negative' % self.name,
                (st['request'] < 0).any()))
        z = eng._ext_zero
        return torch.stack([(z if st['request'] is None
                             else st['request']).clamp(min=0),
                            z if st['evap'] is None else st['evap']], 1)

    def run(self, routing, infl, req, inj, t):
        """The routing step with the reservoir sweep (called by the routing
        connector's compute): returns (discharge, delivered withdrawals)."""
        st = self._staged_for(t)
        self.begin(t)
        return self.engine.step(infl, request=st['request'], evap=st['evap'],
                                withdraw=req, inject=inj)

    def complete(self, context, dt):
        """Fill the staged proposal after the routing step ran."""
        eng, p = self.engine, self._prop
        L = eng.last
        raw, infl, ev = L['raw'], L['inflow'], L['evap']
        W, env, rel, spl = (L['withdrawn'], L['env'], L['release'],
                            L['spill'])
        S1 = eng.storage
        down = env + rel + spl + (raw - infl)
        dS = S1 - self._S0
        scale = (raw.sum() + ev.sum() + W.sum() + down.sum()
                 + (self._S0.abs() + S1.abs()).sum())
        p.balance = Balance(I_ext=raw.sum(), ET=ev.sum(),
                            Q_out=down.sum() + W.sum(), dS=dS.sum(),
                            n_terms=8 * self.R, scale=scale)
        p.exports = dict(storage=S1.clone(), supplied=W.clone())
        p.series = dict(storage=S1.clone().unsqueeze(0),
                        inflow=raw.clone().unsqueeze(0),
                        release=(env + rel).unsqueeze(0),
                        spill=spl.clone().unsqueeze(0),
                        evap=ev.clone().unsqueeze(0),
                        supplied=W.clone().unsqueeze(0))
        p.info = dict(residual=L['residual'].clone())

    def validate(self, proposal, context):
        S = proposal.exports.get('storage')
        if S is None:
            return ['%s: the routing step did not complete the reservoirs'
                    % self.name]
        tol = 1e-9 * self.cap
        return [DeviceFlag('%s: storage outside [0, capacity]' % self.name,
                           ((S < -tol) | (S > self.cap + tol)).any()),
                DeviceFlag('%s: the engine\'s storage identity is off'
                           % self.name,
                           (proposal.info['residual'].abs() > tol).any())]

    def commit(self, proposal, context):
        self.staged = None

    def save_state(self):
        self.routing._rollback()
        return dict(engine={k: getattr(self.engine, k).cpu()
                            for k in self.engine._STATE}, cal=self._cal)

    def load_state(self, blob):
        for k in self.engine._STATE:
            getattr(self.engine, k).copy_(blob['engine'][k].to(self.dev))
        self.engine._refresh(int(self.engine.month))
        self._cal = self._cal_next = blob['cal']
        self.staged = None

    # -- for users -------------------------------------------------------------
    def volume_balance(self):
        """The engine's cumulative storage identity [R] m3 (zero to
        rounding)."""
        self.routing._rollback()
        return self.engine.volume_balance()
