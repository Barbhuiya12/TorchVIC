# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_routing: VIC-WUR routing (C1, CascadeRouter) as a connector.

Plan hooks (docs/connectors/vicwur.md): F `pre_force` (DISCHARGE
forcing, FORCE_ROUTING), P1 `pass1` (rout_run: non-renewable payback from
baseflow and runoff, mm -> m3 s-1; then the ring shift and the runoff
convolution), P2 `sweep_slot(level, 'route')` (rout_basin_run of the level's
cells), O `put_data`. The slots of the other connectors (dams, local and
remote water use) edit the ring of the same router between its levels.

Options
    downstream, uh_inflow, uh_runoff, area, rout_steps_per_day
        or params (a vicwur.params.RoutingParams) with area
    force            a Series spec of DISCHARGE (m3 s-1, cell order), or None
                     (a `groundwater` connector of the vicwur3 profile in
                     the group adds its baseflow to it: VIC-WUR 3.0's MODFLOW
                     DISCHARGE forcing, plugins/modflow/mf_run.py:324-346)
    nonrenew_runoff, gwm   the rout_run options

State (checkpoint): ring, head, runoff, inflow, discharge, stream, force.
(The non-renewable deficit is the router's `deficit`; vicwur_nonrenew owns
and saves it.)

Ledger (m3 per day): I_ext = VIC runoff+baseflow + forced inflow + water the
other connectors put into the ring; Q_out = outlet discharge + the payback
to the non-renewable deficit + water they took from the ring; dS = change of
the in-stream storage (bins >= n_sub, `stream` * dt).
"""
import numpy as np
import torch

from ..connector import Balance, ComputeError
from ..spec import CouplingError, VarSpec
from ...model.routing_cascade import CascadeRouter
from .vicwur_common import (GRID, STEP_S, Series, WurConnector, as_f64, put)


class VicWurRoutingConnector(WurConnector):
    name = 'vicwur_routing'
    version = '1'
    kind = 'routing'
    role = 'routing'

    def __init__(self, downstream=None, uh_inflow=None, uh_runoff=None,
                 area=None, rout_steps_per_day=1, params=None, force=None,
                 nonrenew_runoff=False, gwm=False, group=None, **config):
        self.flags = dict(routing=True, force_routing=force is not None)
        super().__init__(group=group, **config)
        if params is not None:
            downstream, uh_inflow, uh_runoff = (params.downstream,
                                                params.uh_inflow,
                                                params.uh_runoff)
        if downstream is None or uh_inflow is None or uh_runoff is None \
                or area is None:
            raise CouplingError('%s: give downstream, uh_inflow, uh_runoff '
                                'and area (or params and area)' % self.name)
        npy = lambda x: np.asarray(x.cpu() if torch.is_tensor(x) else x)
        self.down = npy(downstream).astype(np.int64)
        self.uhi, self.uhr = npy(uh_inflow), npy(uh_runoff)
        self.area_np = npy(area).astype(np.float64)
        n = self.N = len(self.down)
        if self.area_np.shape != (n,) or not (self.area_np > 0).all():
            raise CouplingError('%s: area must be [N] and positive'
                                % self.name)
        for nm, u in (('uh_inflow', self.uhi), ('uh_runoff', self.uhr)):
            if u.ndim != 2 or u.shape[1] != n:
                raise CouplingError('%s: %s must be [UH_LENGTH, N]'
                                    % (self.name, nm))
        if self.down.min() < 0 or self.down.max() >= n:
            raise CouplingError('%s: downstream outside 0..N-1' % self.name)
        self.nsub = int(rout_steps_per_day)
        if self.nsub < 1 or 86400 % self.nsub:
            raise CouplingError('%s: rout_steps_per_day must divide 86400'
                                % self.name)
        self.force = None if force is None else Series(force, 'force')
        self.nonrenew_runoff, self.gwm = bool(nonrenew_runoff), bool(gwm)
        self._check_config()
        self.outputs = (
            VarSpec('discharge', 'm3 s-1', ('cell',), kind='state',
                    grid=GRID, description='OUT_DISCHARGE'),
            VarSpec('stream_moist', 'mm', ('cell',), kind='state', grid=GRID,
                    description='OUT_STREAM_MOIST'))
        self.owned_state = ('%s.ring' % self.name,)
        self.router = None

    # -- build ---------------------------------------------------------------
    def _build(self):
        G = self.group
        if self.router is not None:
            return
        rcv = None
        if G.edges is not None:
            rcv = [[] for _ in range(self.N)]
            for s, d in zip(*G.edges):
                rcv[int(s)].append(int(d))
        r = CascadeRouter(self.down, self.uhi, self.uhr, self.area_np,
                          STEP_S, self.nsub, receiving=rcv,
                          force=(self.force is not None
                                 or 'groundwater' in G.roles),
                          nonrenew_runoff=self.nonrenew_runoff,
                          gwm=self.gwm, device=G.dev)
        self.router = G.router = r
        G.levels = [r.group_cells(g) for g in range(len(r._gptr) - 1)]
        self.outlet = torch.as_tensor(self.down == np.arange(self.N),
                                      device=G.dev)[r.perm]
        if self.force is not None:
            self.force.to(G.dev)
        self._S_prev = (r.stream.sum() * STEP_S).clone()

    def _init(self, context):
        self._build()

    # -- hooks ---------------------------------------------------------------
    def pre_force(self):
        f = (self.force.get(self.group.step) if self.force is not None
             else None)
        gw = self.group.roles.get('groundwater')
        if gw is not None:
            b = gw.river_force()
            f = b if f is None else f + b
        if f is not None:
            self.router.set_force(f)

    def pass1(self):
        r, v = self.router, self.view
        if v is None or v.runoff is None or (not self.gwm
                                              and v.baseflow is None):
            raise ComputeError('%s: the step view carries no runoff/'
                               'baseflow' % self.name)
        rs = as_f64(v.runoff, r.device)
        rb = as_f64(v.baseflow, r.device) if v.baseflow is not None \
            else torch.zeros_like(rs)
        a = r.area
        vic_in = ((r.to_sweep(rs) + (torch.zeros_like(a) if self.gwm
                                     else r.to_sweep(rb))) * a).sum() / 1000.
        d0 = (r.deficit * a).sum() / 1000.
        r.rout_run(rs, rb)
        r.begin()
        d1 = (r.deficit * a).sum() / 1000.
        pb = d0 - d1
        self.acc('vic_in', vic_in)
        self.acc('payback', pb)
        # the payback is a difference of deficit totals: its rounding is
        # relative to them, so they enter the ledger's scale
        self.acc('dmag', d0.abs() + d1.abs())
        self.group.give('pb_flow', pb)
        self.acc('runoff_m3', r.runoff.sum() * STEP_S)

    def sweep_slot(self, lv, slot, cells):
        if slot == 'route':
            self.router.route_group(lv, cells)

    def graph_key(self):
        """The router's tensors the sweep reads and writes (they are
        fixed for the run: load_state writes into them)."""
        r = self.router
        if r is None or r.device.type != 'cuda':
            return None
        return tuple(t.data_ptr() for t in (
            r.ring, r.head, r.up_t, r.nup, r.uh_inflow, r.force, r.inflow,
            r.discharge, r.stream, r.gptr, r.area, r.perm, r.deficit))

    def put_data(self):
        G, r = self.group, self.router
        vic_in = self.pop('vic_in')
        if isinstance(vic_in, float):          # no P phases this day
            return
        pb = self.pop('payback')
        dmag = self.pop('dmag')
        self.pop('runoff_m3')
        force = (r.force.sum() * STEP_S if r.use_force
                 else self.zero())
        ring_in, ring_out = G.acc.pop('ring_in', 0.0), G.acc.pop('ring_out',
                                                                  0.0)
        outlet = (r.discharge * self.outlet).sum() * STEP_S
        S1 = r.stream.sum() * STEP_S
        dS = S1 - self._S_prev
        # every term by magnitude: a losing river (negative MODFLOW forced
        # inflow, negative bins) makes them signed
        ab = lambda x: x.abs() if torch.is_tensor(x) else abs(x)
        scale = (ab(vic_in) + ab(force) + ab(ring_in) + ab(outlet) + ab(pb)
                 + ab(ring_out) + ab(self._S_prev) + ab(S1) + dmag)
        self._S_prev = S1
        bal = Balance(I_ext=vic_in + force + ring_in,
                      Q_out=outlet + pb + ring_out, dS=dS,
                      n_terms=self.N * (r.R + r.nsub + 8), scale=scale,
                      detail=dict(ring_in=ring_in, ring_out=ring_out,
                                  payback=pb, outlet=outlet, force=force))
        out = r.outputs()
        exp = dict(discharge=out['OUT_DISCHARGE'].clone(),
                   stream_moist=out['OUT_STREAM_MOIST'].clone())
        ser = {k: v.clone().unsqueeze(0) for k, v in exp.items()}
        self._pending = (bal, exp, ser)

    # -- w3_restart ----------------------------------------------------------
    def _w3_reset(self, step):
        """VIC-WUR 3.0 restores the ring (STATE_DISCHARGE_DT) and the deficit
        (rout_restore.c:52-147) and derives discharge and stream from the
        ring (rout_compute_derived_state_vars, rout_restore.c:150-205);
        runoff, inflow and the held forced inflow are re-formed in the step
        (rout_run, rout_basin_run, the forcing read at current == 0)."""
        r = self.router
        dis = torch.zeros_like(r.discharge)
        st = torch.zeros_like(r.stream)
        for i in range(r.R):
            q = r.ring[r.logical(i)]
            if i < r.nsub:
                dis = dis + q
            else:
                st = st + q
        r.discharge.copy_(dis)
        r.stream.copy_(st)
        self._S_prev = (r.stream.sum() * STEP_S).clone()

    # -- state ---------------------------------------------------------------
    _KEYS = ('ring', 'head', 'runoff', 'inflow', 'discharge', 'stream',
             'force')

    def _state(self):
        return {k: getattr(self.router, k).clone() for k in self._KEYS}

    def _load(self, blob):
        for k in self._KEYS:
            put(getattr(self.router, k), blob[k])
        self._S_prev = (self.router.stream.sum() * STEP_S).clone()
        self.group.acc, self.group.xfer = {}, {}

    def finalize(self, context):
        pass
