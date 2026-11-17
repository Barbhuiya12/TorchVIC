# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_nonrenew: VIC-WUR non-renewable groundwater (C7) as a connector.

P3 `pass3_nonrenew` (wu_nonrenew): the demand that local and remote water
use left unmet is served from a virtual aquifer, the deficit grows by what
is taken, its own return repays the deficit at once, the rest goes to the
FUTURE ring bins (it reaches the downstream cell at t+1 at the earliest).
The payback from the next step's baseflow and runoff is done in the
router's rout_run (P1); the local and remote returns repay in P2.

The deficit [cell, mm] is the router's `deficit` tensor; this connector
owns and checkpoints it. Needs vicwur_wateruse_local and vicwur_routing.

Options
    limit      NREN_LIM, the deficit cap (default unlimited)
    gwm        VIC-WUR 3.0 GWM: no deficit is booked
    return_to  where the part of the pumped water the crops did not use goes:
               'river' (default, VIC-WUR 3.0: the future bins of the river
               ring, wu_nonrenewable.c:254-289) or 'aquifer' (opt-in: it
               recharges the aquifer of the same cell through the groundwater
               connector, 'nr_return', m3; for a group whose groundwater
               member is in mode fixed, whose wells take the water from the
               aquifer). A model-behaviour choice, not a bug fix: on ISC 2005
               the river return made 1.9-2.6 thousand m3 s-1 of the Krishna's
               dry-season flow at Vijayawada out of pumped groundwater.
"""
import torch

from ..connector import Balance, ComputeError
from ..spec import CouplingError, VarSpec
from ...vicwur import nonrenew as NR
from .vicwur_common import GRID, STEP_S, WurConnector, put


class VicWurNonrenewConnector(WurConnector):
    name = 'vicwur_nonrenew'
    version = '1'
    kind = 'wateruse'
    role = 'nonrenew'
    flags = dict(wateruse=True, nonrenew_with=True)

    def __init__(self, limit=None, gwm=False, return_to=None, group=None,
                 **config):
        super().__init__(group=group, **config)
        if return_to not in (None, 'river', 'aquifer'):
            raise CouplingError("%s: return_to is 'river' or 'aquifer'"
                                % self.name)
        self.return_to = return_to
        if limit is not None and not float(limit) >= 0:
            raise CouplingError('%s: limit must be >= 0' % self.name)
        self.limit = None if limit is None else float(limit)
        self.gwm = bool(gwm)
        self._check_config()
        self.outputs = (VarSpec('deficit', 'mm', ('cell',), kind='state',
                               grid=GRID, description='OUT_NONREN_DEFICIT'),
                        VarSpec('withdrawn', 'mm', ('cell',), kind='state',
                                grid=GRID, description=(
                                    'OUT_WITHDRAWN: local + transferred '
                                    '(remote) + non-renewable, VIC-WUR 3.0 '
                                    'wu_output.c:387-392')))
        self.owned_state = ('%s.deficit' % self.name,)

    def _init(self, context):
        G = self.group
        r = G.build()
        self.N = r.n
        self.loc_c = G.need('wateruse_local', self.name)
        self._D_prev = (r.deficit * r.area).sum() / 1000.

    def pass3_nonrenew(self):
        G, r = self.group, self.group.router
        L, st = self.loc_c.loc, self.loc_c.st
        cells = torch.arange(r.n, device=self.dev)
        m0 = G.ring_mass(cells)
        to_aq = self._to_aquifer()
        args = (L.dgw, L.dsf, L.wgw, L.wsf, L.wdm, self.loc_c.cf, r.deficit,
                self.limit, self.loc_c.act, st.withdrawn_tremote,
                st.returned, st.consumed, self.gwm)
        if to_aq:
            out = NR.nonrenew_step(*args)
            # the unused part of the pumped water recharges the aquifer
            G.give('nr_return', r.to_cells(out['returned_to_river']
                                           * r.area / 1000.))   # m3
        else:
            out = NR.nonrenew_step(*args, G.ring_of(cells), r.area, STEP_S,
                                   r.nsub, r.L, r.discharge, r.stream)
            phys = r.logical(r._bins)
            r.ring[phys] = out['dt_discharge'].T
            r.discharge.copy_(out['discharge'])
            r.stream.copy_(out['stream'])
        r.deficit.copy_(out['deficit'])
        st.returned, st.consumed = out['returned'], out['consumed']
        G.book_ring(cells, m0)
        a = r.area / 1000.
        w = out['withdrawn_nonrenew']
        cf = self.loc_c.cf
        cons = (w * cf).sum(1)
        self.acc('w_nr', (out['cell_withdrawn'] * a).sum())
        self.acc('cons', (cons * a).sum())
        self.acc('ring', (out['returned_to_river'] * a).sum())
        self.flag(out['violation'])
        self._w = w
        if self.gwm:
            wv = out['cell_withdrawn'] * a
            self.acc('gwm_src', wv.sum())
            if 'groundwater' in G.roles:
                G.give('gw_pump', r.to_cells(wv))

    def _to_aquifer(self):
        return self.return_to == 'aquifer'

    def _message(self):
        return ('%s: non-renewable withdrawal exceeds demand or availability '
                '(wu_nonrenewable.c:324-353)' % self.name)

    def put_data(self):
        w = self.pop('w_nr')
        if isinstance(w, float):
            return
        r = self.group.router
        cons, ring = self.pop('cons'), self.pop('ring')
        src = self.pop('gwm_src')
        pb = self.group.take('pb_flow') + self.group.take('pb_local') \
            + self.group.take('pb_remote')
        D1 = (r.deficit * r.area).sum() / 1000.
        dS = -(D1 - self._D_prev)
        scale = pb + cons + ring + w + self._D_prev.abs() + D1.abs()
        self._D_prev = D1
        bal = Balance(I_ext=pb + src, Q_out=cons + ring, dS=dS,
                      n_terms=self.N * 16, scale=scale,
                      detail=dict(withdrawn=w, consumed=cons,
                                  ring_return=ring, payback=pb,
                                  gwm_source=src))
        # OUT_WITHDRAWN (VIC-WUR 3.0 wu_output.c:387-392, put_data):
        # withdrawn_gw + _surf + _dam + _tremote + _nonrenew summed over
        # the sectors; the local connector exports the first three, the
        # remote connector's receiver total and this step's non-renewable
        # water are added here
        # under GWM: surface + dam + transferred only (wu_output.c:387-392
        # leaves the groundwater and non-renewable terms out)
        lw = self.loc_c.loc
        if self.gwm:
            wtot = (lw.wsf + lw.wdm + self.loc_c.st.withdrawn_tremote).sum(1)
        else:
            wtot = (lw.wsf + lw.wgw + lw.wdm
                    + self.loc_c.st.withdrawn_tremote + self._w).sum(1)
        exp = dict(deficit=r.to_cells(r.deficit),
                   withdrawn=r.to_cells(wtot))
        self._pending = (bal, exp, {k: x.clone().unsqueeze(0)
                                    for k, x in exp.items()})

    def _state(self):
        return dict(deficit=self.group.router.deficit.clone())

    def _load(self, blob):
        r = self.group.router
        put(r.deficit, blob['deficit'])
        self._D_prev = (r.deficit * r.area).sum() / 1000.
