# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Shared pieces of the VIC-WUR connectors (docs/connectors/vicwur.md).

The VIC-WUR connectors (vicwur_routing, vicwur_dams, vicwur_wateruse_local,
vicwur_wateruse_remote, vicwur_nonrenew, vicwur_efr) do not run in the graph
order of the runtime: each is a component of one `WurGroup`, a PhaseScheduler
(torchvic.vicwur.order) that runs their hooks in VIC-WUR's phase order
(F, U, V, P1, P2 level sweep, P3, O). The group is VicConnector's
`phase_group` (a WurGroup, or its name); VIC runs its days and the group runs
between them. The connectors are also listed in VIC's `day_coupled`: the
runtime then calls their `compute` once a day, right after the group ran
P1-O of the day before, and that call returns the water balance of those
phases (one Balance per module and day, in m3).

What the plugins read from the step's StepView: `runoff`, `baseflow`
([cell] mm per step, VIC's OUT_RUNOFF and OUT_BASEFLOW, which VicConnector
takes from basin's day view), `water_stress` (VIC's stored stress per
column and tile, with a vicwur_core) and the soil (moist, ice, cv,
area_fract). At the run's first step VicConnector hands F, U the cold
start.

The plugins that act inside vic_run's own state work through two pieces: the
group builds the run's `VicwurCore` (`vic_core`) from its members'
`core_hooks(core)` (irrigation: paddy layer Ksat; CO2: b_co2 and Catm;
crop: the stored water stress) and, when a member edits VIC's carried tile
states (land use), a `state_fn` that runs the members' `at_state(states)`
hooks in F order at the very start of each VIC step: that is where C's
lu_apply runs (in F, after the P3 edits of the day before have reached
all_vars, before efr_forcing reads the start-of-step moisture). The tile
Cv in force (`cv_columns`) also reaches the step view and the core.

All state lives on the device in the router's sweep order; no host sync
inside a step (host ints only: the step index from the clock).
"""
import weakref

import numpy as np
import torch

from ..connector import Balance, ComputeError, Connector, DeviceFlag, Proposal
from ..spec import CouplingError, GridSpec
from ...vicwur.order import Options, PhaseScheduler, plan

STEP_S = 86400.0
GRID = 'vicwur_cells'

_GROUPS = {}


class _Fan(object):
    """Several connectors of one plan kind (the water-use plan kind is
    served by the local, remote and non-renewable connectors): a hook is
    called on every member that has it, in registration order."""

    def __init__(self, members):
        self.members = members

    def __getattr__(self, name):
        if name.startswith('__') or not any(hasattr(m, name)
                                            for m in self.members):
            raise AttributeError(name)
        fns = [getattr(m, name) for m in self.members if hasattr(m, name)]

        def call(*a):
            for f in fns:
                f(*a)
        return call


class WurGroup(PhaseScheduler):
    """The VIC-WUR phase group: PhaseScheduler whose options and components
    come from the connectors that join it."""

    def __init__(self, name='vicwur'):
        super().__init__(Options())
        self.name = name
        self.parts = {}                 # plan kind -> [connector]
        self.roles = {}                 # role -> connector
        self.flags = {}
        self.dev = self.dtype = None
        self.t_end = 0.0
        self.t = 0.0
        self.step = 0
        self.router = None
        self.edges = None               # remote (src, dst), cell order
        self.acc = {}                   # ring edits metered this day (m3)
        self.xfer = {}                  # amounts one connector hands another
        self.view = None
        self.vic = None                 # VicConnector (attach_vic)
        self.throughfall = 'fixed'      # vicwur3 core: 'fixed' or 'exact'
        self._lay = None
        self._w3rs = None               # (step, [reset fns]): w3_restart

    #: VicConnector hands the step view the day's runoff and baseflow
    wants_fluxes = True

    # -- membership ----------------------------------------------------------
    def join(self, conn, kind, role, **flags):
        if role in self.roles:
            raise CouplingError('vicwur group %r has a %s connector '
                                'already' % (self.name, role))
        self.roles[role] = conn
        self.parts.setdefault(kind, []).append(conn)
        self.flags.update({k: v or self.flags.get(k, False)
                           for k, v in flags.items()})
        self.o = Options(**self.flags)
        self.plan = plan(self.o)
        ms = self.parts[kind]
        self.kinds[kind] = ms[0] if len(ms) == 1 else _Fan(list(ms))

    def need(self, role, who):
        c = self.roles.get(role)
        if c is None:
            raise CouplingError('%s needs a %s connector in the same group '
                                '(%r)' % (who, role, self.name))
        return c

    def setup(self, context):
        if self.dev is None:
            self.dev, self.dtype = context.device, context.dtype
        elif torch.device(context.device) != torch.device(self.dev):
            raise CouplingError('the VIC-WUR connectors of one group must '
                                'run on one device')
        self.t_end = max(self.t_end, float(context.t_end))

    # -- the VIC side --------------------------------------------------------
    def attach_vic(self, vic):
        """VicConnector at its initialize: the case's layout (cells, snow
        bands, tiles, soil) for the members that work on VIC's columns."""
        self.vic = vic
        self._lay = None

    def layout(self):
        """dict(bands: the case's snowbands expansion or None, n_cell,
        n_col, n_tile, cell [col] and band [col] long tensors, area_fract
        [cell, band]) of the attached VIC (None without one)."""
        if self.vic is None:
            return None
        if self._lay is None:
            d = self.vic.driver
            mos = d.mosaic()
            ex = getattr(getattr(d, 'case', None), 'snowbands', None)
            ncol, nt = mos.cv.shape
            dev = self.dev if self.dev is not None else mos.cv.device
            if ex is None:
                cell = torch.arange(ncol, device=dev)
                band = torch.zeros(ncol, dtype=torch.long, device=dev)
                af = torch.ones(ncol, 1, dtype=torch.float64, device=dev)
                nc = ncol
            else:
                cell = torch.as_tensor(np.asarray(ex['cell']),
                                       dtype=torch.long, device=dev)
                band = torch.as_tensor(np.asarray(ex['band']),
                                       dtype=torch.long, device=dev)
                nc = int(ex['n_cell'])
                nb = int(ex['n_band'])
                af = torch.zeros(nc, nb, dtype=torch.float64, device=dev)
                af[cell, band] = torch.as_tensor(
                    np.asarray(ex['weight']), dtype=torch.float64,
                    device=dev)
            first = torch.full((nc,), ncol, dtype=torch.long, device=dev)
            first.scatter_reduce_(0, cell, torch.arange(ncol, device=dev),
                                  'amin')
            self._lay = dict(bands=ex, n_cell=nc, n_col=ncol, n_tile=nt,
                             cell=cell, band=band, area_fract=af,
                             first=first)
        return self._lay

    def _idx(self, name, device):
        """The layout's index `name` on `device` (cached; no copy in the
        step once built)."""
        L = self.layout()
        key = (name, torch.device(device))
        c = L.setdefault('_on', {})
        if key not in c:
            c[key] = L[name].to(device)
        return c[key]

    def to_columns(self, x):
        """[cell, tile, band, ...] -> the case's columns [col, tile, ...]."""
        return x[self._idx('cell', x.device), :,
                 self._idx('band', x.device)]

    def from_columns(self, x, fill=0.0):
        """[col, tile, ...] -> [cell, tile, band, ...] (`fill` where a band
        has no column)."""
        L = self.layout()
        nb = L['area_fract'].shape[1]
        out = torch.full((L['n_cell'], x.shape[1], nb) + tuple(x.shape[2:]),
                         fill, dtype=x.dtype, device=x.device)
        out[self._idx('cell', x.device), :, self._idx('band', x.device)] = x
        return out

    def cell_rows(self, x):
        """A per-column quantity equal over a cell's bands (Cv, the veg
        tables, the soil) on the cells: the first column of each cell."""
        x = torch.as_tensor(x)
        return x.index_select(0, self._idx('first', x.device))

    def met(self, name, step):
        """The cell value of a met forcing of the attached case at model
        step `step` (C's force[iCell].<name>[NR], before any band shift):
        'tair' (C) or 'shortwave' (W m-2). None when the driver has no
        whole-run forcing in memory."""
        d = getattr(self.vic, 'driver', None)
        case = getattr(d, 'case', None)
        if case is None:
            return None
        key = {'tair': 'tair', 'shortwave': 'shortwave'}[name]
        echo = getattr(case, '_cell_echo', None) or {}
        if key in echo:
            return echo[key][step]
        a = case.atm.get(key)
        if a is None:
            return None
        return self.cell_rows(a[step]) if self.layout()['bands'] \
            is not None else a[step]

    def cv_columns(self):
        """The tile Cv in force on the case's columns when a land-use
        member changes it, else None."""
        lu = self.roles.get('landuse')
        return None if lu is None else lu.cv_columns()

    def vic_core(self):
        """The run's VicwurCore from the members' hooks (None when no member
        acts inside vic_run: VIC then runs as it would alone)."""
        from ...vicwur.core import VicwurCore
        mem = [c for c in self.roles.values()]
        hooks = [c for c in mem if hasattr(c, 'core_hooks')]
        state = [c for c in mem if hasattr(c, 'at_state')
                 and c.wants_state()]
        # VIC-WUR 3.0's throughfall leak is in vic_run whatever plugin runs:
        # exact mode keeps it even when no member edits VIC (a routing-only
        # group)
        leak = getattr(self, 'throughfall', 'fixed') == 'exact' and getattr(
            getattr(getattr(self.vic, 'driver', None), 'case', None),
            'core', None) == 'vicwur3'
        if not hooks and not state and not leak:
            return None
        core = VicwurCore()
        # the coupling reads the layer stress at a window's end only
        core.stress_history = False
        for c in hooks:
            c.core_hooks(core)
        if state:
            core.state_fn = self._state_fn
        case = getattr(getattr(self.vic, 'driver', None), 'case', None)
        if getattr(case, 'core', None) == 'vicwur3':
            # VIC-WUR 3.0's veg_var throughfall of a leafless vegetated tile
            # (func_surf_energy_bal.c:785-786; vicwur.core throughfall)
            core.throughfall = getattr(self, 'throughfall', 'fixed')
        if not core.active():
            return None
        return core

    #: the members' at_state hooks run in F order (plugin_simulation.c
    #: :39-56: land use, then EFR)
    STATE_ORDER = ('landuse', 'efr')

    def _state_fn(self, t, states):
        for r in self.STATE_ORDER:
            c = self.roles.get(r)
            if c is not None and hasattr(c, 'at_state') and c.wants_state():
                c.at_state(states)

    def build(self):
        """The CascadeRouter (built once, by the routing connector)."""
        if self.router is None:
            self.need('routing', 'a VIC-WUR component')._build()
        return self.router

    # -- PhaseScheduler hooks ------------------------------------------------
    def _set_time(self, t):
        if t is not None:
            self.t = float(t)
            self.step = int(round(self.t / STEP_S))
            if self._w3rs is not None and self.step == self._w3rs[0]:
                # w3_restart: VIC-WUR 3.0's state file is written after
                # plugin_run of the last day (vic_image.c:158-166), so the
                # resets fall after P1-O of the day before the restart,
                # before its F
                fns, self._w3rs = self._w3rs[1], None
                for fn in fns:
                    fn()
        super()._set_time(t)

    def w3_pending(self, step, fn):
        """Run `fn` (a member's or VIC's w3_restart reset) when the group
        enters `step`'s F phase."""
        if self._w3rs is not None and self._w3rs[0] != step:
            raise CouplingError('w3_restart at two steps (%d, %d)'
                                % (self._w3rs[0], step))
        if self._w3rs is None:
            self._w3rs = (step, [])
        self._w3rs[1].append(fn)

    def _set_view(self, view):
        self.view = view
        super()._set_view(view)

    def save_state(self):
        d = {r: c.save_state() for r, c in self.roles.items()}
        d['_acc'] = (dict(self.acc), dict(self.xfer))
        return d

    def load_state(self, blob):
        for r, b in blob.items():
            if r == '_acc':
                self.acc, self.xfer = dict(b[0]), dict(b[1])
            else:
                self.roles[r].load_state(b)

    # -- ring metering -------------------------------------------------------
    def ring_of(self, cells):
        """[C, R] ring of `cells` in logical bin order."""
        r = self.router
        phys = r.logical(r._bins)
        return r.ring.index_select(1, cells).index_select(0, phys).T

    def ring_mass(self, cells):
        return self.router.ring.index_select(1, cells).sum()

    def book_ring(self, cells, m0):
        """Book the change of the cells' ring since `m0` (ring_mass) as water
        added to / taken from the river network, m3."""
        d = (self.ring_mass(cells) - m0) * STEP_S
        z = torch.zeros_like(d)
        self.add('ring_in', torch.where(d > 0, d, z))
        self.add('ring_out', torch.where(d < 0, -d, z))

    def add(self, key, v):
        self.acc[key] = self.acc.get(key, 0.0) + v

    def give(self, key, v):
        self.xfer[key] = self.xfer.get(key, 0.0) + v

    def take(self, key):
        return self.xfer.pop(key, 0.0)


def resolve_group(g, role):
    """The group a connector joins: an object, or the current group of that
    name (a new one when the name's group has this role already, i.e. a new
    run)."""
    if isinstance(g, WurGroup):
        return g
    name = g or 'vicwur'
    cur = _GROUPS.get(name)
    cur = cur() if cur is not None else None
    if cur is None or role in cur.roles:
        cur = WurGroup(name)
        _GROUPS[name] = weakref.ref(cur)
    return cur


def get_group(name):
    """The current group called `name` (VicConnector's phase_group by
    name)."""
    cur = _GROUPS.get(name)
    cur = cur() if cur is not None else None
    if cur is None:
        raise CouplingError('no VIC-WUR group %r (the vicwur_* connectors '
                            'create it)' % (name,))
    return cur


class Series(object):
    """A forcing quantity per model step: a constant; {'series': x} with one
    record a step on the first axis (the last record is held beyond the
    end); or {'forcing': obj}, an object with `get(step)` (PluginForcing)."""

    def __init__(self, spec, name='forcing'):
        self.name, self.kind, self.v, self.obj = name, 'const', None, None
        if isinstance(spec, dict):
            ks = set(spec)
            if len(ks) != 1 or not ks <= {'constant', 'series', 'forcing'}:
                raise CouplingError('%s: give constant, series or forcing '
                                    '(one of them)' % name)
            self.kind = ks.pop()
            val = spec[self.kind]
        else:
            val = spec
        if self.kind == 'forcing':
            if not callable(getattr(val, 'get', None)):
                raise CouplingError('%s: forcing needs an object with get('
                                    'step)' % name)
            self.obj = val
        else:
            self.v = torch.as_tensor(np.asarray(
                val.cpu() if torch.is_tensor(val) else val),
                dtype=torch.float64)
        self.dev = None

    def to(self, dev):
        self.dev = dev
        if self.v is not None:
            self.v = self.v.to(dev)
        return self

    def reads(self, step):
        """True when a new record is read at `step` (C's forcerun): the
        forcing object's own flag; every step of a series (until its last
        record is held); step 0 of a constant."""
        if self.kind == 'forcing':
            fn = getattr(self.obj, 'reads', None)
            return bool(fn(step)) if fn is not None else True
        if self.kind == 'series':
            return int(step) < self.v.shape[0]
        return int(step) == 0

    def get(self, step):
        if self.kind == 'forcing':
            return torch.as_tensor(self.obj.get(step), dtype=torch.float64,
                                   device=self.dev)
        if self.kind == 'series':
            return self.v[min(int(step), self.v.shape[0] - 1)]
        return self.v


class WurConnector(Connector):
    """Base of the VIC-WUR connectors: a daily component of a WurGroup whose
    compute returns the balance of the phases the group ran."""
    backends = ('cpu', 'cuda')
    checkpoint = 'full'
    interval_s = STEP_S
    graph_safe = False
    kind = None           # the plan kind the hooks are called as
    role = None
    flags = {}            # Options the connector switches on
    validation = ('direct replay of its kernels in the VIC-WUR order '
                  '(tests/test_vicwur_connectors.py); kernels: see '
                  'docs/connectors/vicwur.md')

    def __init__(self, group=None, throughfall=None, **config):
        super().__init__(**config)
        self.group = resolve_group(group, self.role)
        if throughfall is not None:
            self.group.throughfall = throughfall
        self.view = None
        self.N = 0
        self.dev = self.dtype = None
        self._pending = None
        self._bad = None
        self._acc = {}
        self.group.join(self, self.kind, self.role, **self.flags)

    def _check_config(self):
        if self.config:
            raise CouplingError('%s: unknown option(s) %s'
                                % (type(self).__name__,
                                   ', '.join(sorted(self.config))))

    # -- declarations --------------------------------------------------------
    def grids(self):
        return [GridSpec(GRID, self.N, kind='cells')]

    # -- group side ----------------------------------------------------------
    def set_view(self, view):
        self.view = view

    def flag(self, cond):
        """OR a device condition into the day's validity flag."""
        c = (cond if torch.is_tensor(cond) else torch.as_tensor(
            cond, device=self.dev)).any()
        self._bad = c if self._bad is None else (self._bad | c)

    def zero(self):
        return torch.zeros((), dtype=torch.float64, device=self.dev)

    def acc(self, key, v):
        self._acc[key] = self._acc.get(key, 0.0) + v

    def pop(self, key):
        return self._acc.pop(key, 0.0)

    # -- life cycle ----------------------------------------------------------
    def initialize(self, context):
        self.group.setup(context)
        self.dev, self.dtype = self.group.dev, torch.float64
        self._init(context)
        if self.group.router is not None:
            self.N = self.group.router.n

    def _init(self, context):
        pass

    def compute(self, inputs, context, dt):
        if abs(dt - STEP_S) > 1e-6:
            raise ComputeError('%s steps once a day (VIC-WUR runs daily)'
                               % self.name)
        p, self._pending = self._pending, None
        flags = []
        if self._bad is not None:
            flags.append(DeviceFlag(self._message(), self._bad))
            self._bad = None
        if p is None:
            bal, exp, ser = Balance(n_terms=1, scale=0.0), {}, None
        else:
            bal, exp, ser = p
        return Proposal(connector=self.name, t0=context.t, t1=context.t + dt,
                        state=None, exports=exp, balance=bal, series=ser,
                        backend=str(self.dev), flags=flags)

    def _message(self):
        return '%s: a VIC-WUR kernel check failed' % self.name

    def w3_restart(self, context, t):
        """checkpoint.w3_restart (runtime, after the load): reset what
        VIC-WUR 3.0's state file does not hold (`_w3_reset`) when the group
        enters the restart step, i.e. after P1-O of the day before it."""
        step = int(round(float(t) / STEP_S))
        self.group.w3_pending(step, lambda: self._w3_reset(step))

    def _w3_reset(self, step):
        """A new VIC-WUR 3.0 run's state at `step` for what the state file
        does not hold (default: the connector carries nothing across
        days)."""

    def commit(self, proposal, context):
        pass

    def save_state(self):
        return self._state()

    def load_state(self, blob):
        self._load(blob)
        self._pending = None
        self._bad = None
        self._acc = {}

    def _state(self):
        return {}

    def _load(self, blob):
        pass


def clone_tensors(d):
    return {k: v.clone() for k, v in d.items()}


def put(dst, blob, dev=None):
    """In-place copy of a saved tensor into the live one (other objects hold
    references to it)."""
    dst.copy_(blob.to(dst.device))


def _conv(x, dev, dtype):
    if torch.is_tensor(x):
        return x.to(device=dev, dtype=dtype)       # no host round trip
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)


def as_f64(x, dev):
    return _conv(x, dev, torch.float64)


def as_i64(x, dev):
    return _conv(x, dev, torch.int64)
