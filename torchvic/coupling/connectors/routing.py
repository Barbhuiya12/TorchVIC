"""RoutingConnector: the river network behind the Connector API.

The engine is the stepping router of `torchvic.model.routing`
(`StepRouter` on the CPU, the fused `routing_fast.FastStepRouter` on CUDA;
both equal the whole-series convolution of RVIC/Lohmann routing). The
connector adds no routing physics.

One routing step (`step_s`, default one day) is

    begin(runoff)  ->  level-wise withdraw / inject (and the sweeps of a
                       reservoir or water-quality connector)  ->  end()

Inputs (declared, so the runtime checks units and grids):

    inflow              flux  mm over the interval, VIC cells: runoff +
                        baseflow of the land surface (VIC's `total_runoff`).
                        A longer producer interval (VIC's 31-day window) is
                        spread uniformly in time over the routing steps by
                        the runtime's exchange buffer (conservative; the
                        within-window shape is not kept, see
                        docs/CONNECTOR_API.md).
    withdrawal_request  state m3 per routing step at each site: what other
                        connectors want taken out of the river.
    injection           flux  m3 added at each site (return flows).

Outputs:

    discharge           state m3 s-1 at every gauge node (the step's value
                        read from the ring, as RVIC's discharge).
    withdrawn           flux  m3 delivered to the requests (never more than
                        passes the site without driving a ring slot
                        negative: `StepRouter._cap`).

The services other connectors use are these two connections: a request goes
in as `withdrawal_request`, the delivered water comes back as `withdrawn`.

Water balance (m3 over the whole network, per step):

    R = (V_in + J) - (Out_outlets + W) - dStored - E_unrouted

V_in the volume handed over (cell depth x area), J injections, W
withdrawals (water leaving to other connectors, reservoirs included),
Out_outlets the discharge x dt read at the outlet gauges (gauges with no
gauge downstream), dStored the change of the water on its way (the ring),
E_unrouted the water of cells that drain to no gauge. V_in is measured from
the cell depths and areas, Out and dStored from the router's own state, so a
router built with other areas, or a unit hydrograph that loses water, shows
up as a residual.

Transactions: the engine updates its ring in place. `compute` takes an undo
snapshot first and `commit` drops it; if the step is rejected the next call
(compute, save_state, load_state) restores the snapshot first, so the
accepted state is the one before the rejected step.

Sweep plugins (both may be present: the sweep then does both, see
`_combined.py`): a reservoir or quality connector listed in this connector's
configuration (`reservoirs:`, `quality:`) is a declared dependency (it
computes first and stages its inputs); the routing step runs their sweep and
completes their proposals (`SweepPlugin`).
"""
import numpy as np
import torch

from ..connector import (Balance, ComputeError, Connector, DeviceFlag,
                         Proposal)
from ..spec import CouplingError, GridSpec, VarSpec
from ...model import river_network as rn
from ...model import routing_fast as rf

CELLS = 'vic_cells'
NODES = 'river_nodes'
SITES = 'river_sites'
#: a drained river holds ~1e-150 m3 in its ring tails; the relative
#: tolerance of the ledger needs a floor of one litre not to read that
#: noise as a residual (it changes the tolerance by < 1e-15 m3)
_SCALE_FLOOR_M3 = 1.0e-3


def _as(x, dev, dtype):
    return torch.as_tensor(x, device=dev, dtype=dtype)


class SweepPlugin(object):
    """Mixin of a connector whose computation happens inside the routing
    step (it needs the water passing a site, level by level). Its own
    `compute` stages its inputs and returns a proposal with empty exports;
    the routing connector's compute, which runs after it, completes the
    proposal (exports, balance, series) before the runtime validates,
    checks and commits. Consumers of a plugin's outputs at lag 0 must
    declare the routing module as a dependency (they then compute after it).
    """
    #: the runtime sets this to {module name: connector}
    peers = None
    routing_name = 'routing'

    def _staged_for(self, t0):
        s = getattr(self, 'staged', None)
        if s is None or abs(s['t0'] - t0) > 1e-6:
            raise ComputeError('%s: no inputs staged for t = %g s (is it '
                               'due with the routing connector?)'
                               % (self.name, t0))
        return s


class RoutingConnector(Connector):
    name = 'routing'
    version = '1'
    backends = ('cpu', 'cuda')
    checkpoint = 'full'
    graph_safe = False
    validation = ('equals the standalone StepRouter / whole-series routing '
                  'it wraps (tests/test_coupling_routing.py); the ledger '
                  'closes the network water')
    imported_fluxes = ('inflow', 'injection')
    exported_fluxes = ('withdrawn',)

    def __init__(self, net=None, gauge_cells=None, area=None, site_cells=(),
                 cell_uh=None, velocity=1.5, diffusion=800.0, step_s=86400.0,
                 grid=CELLS, fdir=None, domain=None, gauges=None,
                 reservoirs=None, quality=None, fast=True, subcell=(),
                 name=None, **config):
        super().__init__(**config)
        if name:
            self.name = name
        self.net, self.gauge_cells, self.area_in = net, gauge_cells, area
        self.fdir, self.domain, self.gauges_file = fdir, domain, gauges
        self.site_cells_cfg = [int(c) for c in (site_cells or ())]
        self.cell_uh = cell_uh
        self.velocity, self.diffusion = float(velocity), float(diffusion)
        self.interval_s = float(step_s)
        self.grid = grid
        self.subcell = tuple(subcell or ())
        self.fast = bool(fast)
        self.res_name, self.qual_name = reservoirs, quality
        self.dependencies = tuple(x for x in (reservoirs, quality) if x)
        self.wanted = set()
        self.peers = None
        self.router = None
        self.res_conn = self.qual_conn = None
        self.n_cell = self.n_node = self.n_site = 0
        self._undo = None            # snapshot taken by an uncommitted compute
        self._dirty = False
        self._live_cache = None
        self._tot = None
        self.t = 0.0
        self.inputs = (
            VarSpec('inflow', 'mm', ('cell',), kind='flux', grid=grid,
                    description='runoff + baseflow reaching the river over '
                                'the interval (VIC total_runoff)'),
            VarSpec('withdrawal_request', 'm3', ('site',), kind='state',
                    grid=SITES, description='water wanted from the river at '
                    'each site per routing step'),
            VarSpec('injection', 'm3', ('site',), kind='flux', grid=SITES,
                    description='water added to the river at each site'))
        self.outputs = (
            VarSpec('discharge', 'm3 s-1', ('node',), kind='state',
                    grid=NODES, description='discharge at each gauge node, '
                    'the routing step\'s value'),
            VarSpec('withdrawn', 'm3', ('site',), kind='flux', grid=SITES,
                    description='water delivered to withdrawal_request'))

    @property
    def owned_state(self):
        return (self.name + '.ring',)

    # -- declarations --------------------------------------------------------
    def grids(self):
        a = None if self.area_in is None else torch.as_tensor(
            np.asarray(self.area_in, dtype=np.float64))
        return [GridSpec(self.grid, self.n_cell, area=a, kind='cells',
                         description='cells whose runoff is routed'),
                GridSpec(NODES, self.n_node, kind='nodes',
                         description='gauge nodes (channel gauges, extra '
                                     'site nodes, sub-cell gauges)'),
                GridSpec(SITES, self.n_site, kind='sites',
                         description='withdrawal / injection / reservoir '
                                     'sites')]

    # -- construction --------------------------------------------------------
    def _network(self):
        """(net, gauge_cells, area) from the given objects or the files."""
        net, gc, area = self.net, self.gauge_cells, self.area_in
        if net is None:
            if not (self.fdir and self.domain):
                raise CouplingError('%s: give a network (net, gauge_cells, '
                                    'area) or the files fdir and domain'
                                    % self.name)
            import netCDF4
            d = netCDF4.Dataset(self.domain)
            lat, lon = d['lat'][:].data, d['lon'][:].data
            mask = np.asarray(d['mask'][:]) > 0
            area_g = np.asarray(d['area'][:], dtype=np.float64)
            if 'frac' in d.variables:
                area_g = area_g * np.asarray(d['frac'][:], dtype=np.float64)
            f, v = (self.fdir.rsplit(':', 1) if ':' in self.fdir
                    else (self.fdir, 'fdir'))
            net, cells = rn.d8_downstream(
                np.asarray(netCDF4.Dataset(f)[v][:]), mask, lat, lon)
            area = area_g[cells[:, 0], cells[:, 1]]
            if gc is None:
                if not self.gauges_file:
                    raise CouplingError('%s: give gauges (a csv with lat, '
                                        'lng) or gauge_cells' % self.name)
                import csv
                cl, cc = lat[cells[:, 0]], lon[cells[:, 1]]
                gl = []
                for row in csv.DictReader(open(self.gauges_file)):
                    la, lo = float(row['lat']), float(row['lng'])
                    gl.append(int(np.argmin(
                        (cl - la) ** 2
                        + ((cc - lo) * np.cos(np.radians(la))) ** 2)))
                gc = np.array(sorted(set(gl)))
        if gc is None or area is None:
            raise CouplingError('%s: gauge_cells and area are required'
                                % self.name)
        return net, np.asarray(gc, dtype=np.int64), np.asarray(
            area, dtype=np.float64)

    def initialize(self, context):
        dev = torch.device(context.device) if context.device is not None \
            else torch.device('cpu')
        net, gc, area = self._network()
        self.area_in = area
        self.n_cell = int(net.n)
        # sites: the configured ones first, then each plugin's cells
        sites = list(dict.fromkeys(self.site_cells_cfg))
        peers = self.peers or {}
        plugins = []
        for key, attr in ((self.res_name, 'res_conn'),
                          (self.qual_name, 'qual_conn')):
            if not key:
                continue
            if key not in peers:
                raise CouplingError('%s: module %r (reservoirs / quality) '
                                    'is not in the run' % (self.name, key))
            pc = peers[key]
            setattr(self, attr, pc)
            plugins.append(pc)
            for c in pc.site_cells():
                if int(c) not in sites:
                    sites.append(int(c))
        self.sites = np.asarray(sites, dtype=np.int64)
        kw = dict(velocity=self.velocity, diffusion=self.diffusion,
                  subcell=self.subcell, site_cells=self.sites, device=dev)
        if self.cell_uh is not None:
            kw['cell_uh'] = _as(self.cell_uh, dev, torch.float64)
        r, info = rn.build_step_router(net, gc, area, self.interval_s, **kw)
        if self.fast and dev.type == 'cuda':
            r = rf.FastStepRouter.from_router(r)
        self.router, self.info_ = r, info
        self.dev, self.dtype = r.dev, r.dtype
        self.n_node, self.n_site = int(r.n_node), int(r.n_site)
        self.area_t = _as(area, dev, self.dtype)
        self._outlets(net, gc, info)
        self.t = 0.0
        site_index = {int(c): i for i, c in enumerate(self.sites)}
        for pc in plugins:
            pc.attach(self, site_index, area)
        self._live_cache = None
        self._tot = self._totals()
        context.say('%s: %d cells, %d nodes (%d outlets), %d sites, ring '
                    'length %d, %s' % (self.name, self.n_cell, self.n_node,
                                       int(self.outlet_idx.numel()),
                                       self.n_site, r.L, type(r).__name__))

    def _outlets(self, net, gc, info):
        """Outlet gauges (channel nodes with no channel node downstream) and
        the cells that drain to some channel node."""
        pairs = info['pairs']
        n_ch = info['n_channel'] + info['n_site_extra']
        extra = [c for c in dict.fromkeys(self.sites.tolist())
                 if c not in set(gc.tolist())]
        all_cells = np.concatenate([gc, np.asarray(extra, np.int64)])
        node_of = -np.ones(net.n, np.int64)
        node_of[all_cells] = np.arange(len(all_cells))
        pc, pg = np.asarray(pairs.cell), np.asarray(pairs.gauge)
        ch = pg < n_ch
        m = ch & (node_of[pc] >= 0) & (node_of[pc] != pg)
        has_down = np.zeros(n_ch, bool)
        has_down[node_of[pc[m]]] = True
        outl = np.nonzero(~has_down)[0]
        routed = np.zeros(net.n, bool)
        routed[pc[ch]] = True
        self.outlet_idx = torch.as_tensor(outl, device=self.dev)
        self.routed = torch.as_tensor(routed, device=self.dev)
        self.n_unrouted = int((~routed).sum())

    # -- the state the engines update in place -------------------------------
    def _live(self):
        if self._live_cache is None:
            r = self.router
            ts = [getattr(r, k) for k in r._STATE]
            for pc in (self.res_conn, self.qual_conn):
                if pc is not None:
                    ts += pc.state_tensors()
            self._live_cache = ts
        return self._live_cache

    def _snapshot(self):
        live = self._live()
        if self._undo is None:
            self._undo = [t.clone() for t in live]
        else:
            torch._foreach_copy_(self._undo, live)
        self._dirty = True

    def _rollback(self):
        if self._dirty:
            torch._foreach_copy_(self._live(), self._undo)
            self._dirty = False
            for pc in (self.res_conn, self.qual_conn):
                if pc is not None:
                    # a plugin's inputs for the step being (re)computed were
                    # staged before this call (the redo of a step another
                    # connector rejected); restoring its state must not
                    # discard them
                    keep = getattr(pc, 'staged', None)
                    pc.after_restore()
                    pc.staged = keep

    def _totals(self):
        """[outlet discharge volume, outlet stored volume, withdrawn,
        injected] cumulative, m3, one device tensor."""
        r = self.router
        o = self.outlet_idx
        return torch.stack([
            r.out.index_select(0, o).sum(),
            r.ring.index_select(1, o).sum() * r.dt,
            r.withdrawn.sum(), r.injected.sum()])

    # -- life cycle ------------------------------------------------------------
    def compute(self, inputs, context, dt):
        if abs(dt - self.interval_s) > 1e-6:
            raise ComputeError('%s: interval %g s is not the routing step '
                               '%g s' % (self.name, dt, self.interval_s))
        # a redone step (runtime: a connector's optimistic result was
        # refused) finds the plugins' freshly staged inputs: rolling the
        # state back clears `staged`, so keep what this step staged
        keep = [(pc, pc.staged) for pc in (self.res_conn, self.qual_conn)
                if pc is not None]
        self._rollback()
        for pc, st in keep:
            if st is not None:
                pc.staged = st
        r, dev, dty = self.router, self.dev, self.dtype
        infl = inputs.get('inflow')
        infl = (torch.zeros(self.n_cell, dtype=dty, device=dev)
                if infl is None else _as(infl, dev, dty))
        if infl.shape[0] != self.n_cell:
            raise ComputeError('%s: inflow has %d cells, the network %d'
                               % (self.name, infl.shape[0], self.n_cell))
        req, inj = inputs.get('withdrawal_request'), inputs.get('injection')
        req = None if req is None else _as(req, dev, dty)
        inj = None if inj is None else _as(inj, dev, dty)
        for nm, v in (('withdrawal_request', req), ('injection', inj)):
            if v is not None and v.shape[0] != self.n_site:
                raise ComputeError('%s: %s has %d sites, the router %d'
                                   % (self.name, nm, v.shape[0],
                                      self.n_site))
        flags = []
        if req is not None:
            flags.append(DeviceFlag('%s: withdrawal_request must not be '
                                    'negative' % self.name,
                                    (req < 0).any()))
        if inj is not None:
            flags.append(DeviceFlag('%s: injection must not be negative'
                                    % self.name, (inj < 0).any()))
        self._snapshot()
        got = None
        extra = {}
        try:
            if self.res_conn is not None and self.qual_conn is not None:
                q, got, extra = self.qual_conn.run(self, infl, req, inj,
                                                   context.t)
            elif self.res_conn is not None:
                q, got = self.res_conn.run(self, infl, req, inj, context.t)
            elif self.qual_conn is not None:
                q, got, extra = self.qual_conn.run(self, infl, req, inj,
                                                   context.t)
            else:
                r.begin(infl)
                if req is not None or inj is not None:
                    got = r.hooks(req, inj)
                q = r.end()
        except ComputeError:
            raise
        except Exception as e:                           # noqa: BLE001
            raise ComputeError('%s: routing step failed: %s: %s'
                               % (self.name, type(e).__name__, e)) from e
        tot = self._totals()
        d = tot - self._tot
        # the step's own discharge volume, not a difference of the router's
        # cumulative `out` (which cannot resolve an increment below its ulp)
        d_out = q.index_select(0, self.outlet_idx).sum() * r.dt
        vol = infl * self.area_t * 1e-3
        v_all = vol.sum()
        v_routed = (vol * self.routed).sum()
        I_ext = v_all + d[3]
        Q_out = d_out + d[2]
        E_un = v_all - v_routed
        # withdrawn / injected are cumulative counters: their increments
        # carry the rounding of the totals
        scale = (v_all.abs() + d[3].abs() + d_out.abs() + d[2].abs()
                 + d[1].abs() + E_un.abs()
                 + self._tot[1].abs() + tot[1].abs()
                 + tot[2].abs() + tot[3].abs() + _SCALE_FLOOR_M3)
        bal = Balance(I_ext=I_ext, Q_out=Q_out, E_other=E_un, dS=d[1],
                      n_terms=self.n_cell + r.L + 4, scale=scale)
        ex = {'discharge': q}
        if 'withdrawn' in self.wanted:
            ex['withdrawn'] = (got[:self.n_site].clone() if got is not None
                               else torch.zeros(self.n_site, dtype=dty,
                                                device=dev))
        series = {'discharge': q.unsqueeze(0)}
        prop = Proposal(connector=self.name, t0=context.t,
                        t1=context.t + dt, state=None, exports=ex,
                        balance=bal, series=series, backend=str(dev),
                        info=dict(tot=tot), flags=flags)
        if self.res_conn is not None:
            self.res_conn.complete(context, dt)
        if self.qual_conn is not None:
            self.qual_conn.complete(context, dt, extra)
        return prop

    def validate(self, proposal, context):
        q = proposal.exports['discharge']
        qmin = q.min()
        return [DeviceFlag('%s: non-finite discharge' % self.name,
                           ~torch.isfinite(q).all()),
                DeviceFlag('%s: negative discharge %%.3e m3/s' % self.name,
                           qmin < -1e-9 * q.abs().max().clamp(min=1.0),
                           values=(qmin,))]

    def commit(self, proposal, context):
        self._dirty = False
        self._tot = proposal.info['tot']
        self.t = proposal.t1
        for pc in (self.res_conn, self.qual_conn):
            if pc is not None:
                pc.accept()

    def save_state(self):
        self._rollback()
        r = self.router
        return dict(router={k: getattr(r, k).cpu() for k in r._STATE},
                    t=self.t)

    def load_state(self, blob):
        r = self.router
        self._dirty = False
        for k in r._STATE:
            getattr(r, k).copy_(blob['router'][k].to(r.dev))
        self.t = float(blob['t'])
        self._tot = self._totals()

    # -- helpers for users and tests -------------------------------------------
    def outlet_nodes(self):
        return self.outlet_idx.cpu().numpy()

    def ledger(self):
        """The router's own per-node ledger [node] (m3)."""
        self._rollback()
        return self.router.balance()
