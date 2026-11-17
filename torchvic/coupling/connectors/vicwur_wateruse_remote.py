# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_wateruse_remote: VIC-WUR remote water use (C6) as a connector.

P2 `sweep_slot(level, 'wu_remote')` (wu_remote: the unmet demand of the
receivers of a source cell is served from the source's river ring, the
source's returns repay the receivers' deficit first, the rest goes to the
source ring). The source/receiver edges also set the routing order (a
source waits for its receivers), so they are given to the router before it
is built. Needs vicwur_wateruse_local in the group (its demand and
withdrawals) and vicwur_routing.

Options
    edges     {'src': [...], 'dst': [...]} or a list of (src, dst) cell
              pairs (the order of one source's edges is the order it serves
              its receivers), or params (vicwur.params.RemoteParams)
    gwm       the VIC-WUR 3.0 GWM switch

State (checkpoint): the remote totals and temporaries (the temporaries hold
the last source's values at each receiver and carry over steps in C).

Ledger (m3 per day, SOURCE-cell area for every term, as C converts):
I_ext = water taken from the source ring; Q_out = water returned to the
ring + the part of the return that repaid receiver deficit; E_other =
consumption at the receivers. With unequal areas the receivers' deficit
changes (non-renewable ledger) differ from the source-area amount: that is
the C area quirk, not a coupling error.
"""
import numpy as np
import torch

from ..connector import Balance, ComputeError
from ..spec import CouplingError, VarSpec
from ...vicwur import wateruse_remote as WR
from .vicwur_common import GRID, STEP_S, WurConnector


class VicWurWateruseRemoteConnector(WurConnector):
    name = 'vicwur_wateruse_remote'
    version = '1'
    kind = 'wateruse'
    role = 'wateruse_remote'
    flags = dict(wateruse=True, remote_with=True)

    def __init__(self, edges=None, params=None, gwm=False, group=None,
                 **config):
        super().__init__(group=group, **config)
        if params is not None:
            src, dst = WR.edges_from_receiving(params.receiving)
        elif isinstance(edges, dict):
            src, dst = edges['src'], edges['dst']
        elif edges is not None:
            src, dst = zip(*edges) if len(edges) else ((), ())
        else:
            raise CouplingError('%s: give edges (or params)' % self.name)
        self.src = np.asarray(src, np.int64)
        self.dst = np.asarray(dst, np.int64)
        if self.src.shape != self.dst.shape or not len(self.src):
            raise CouplingError('%s: edges need equal, non-empty src and '
                                'dst' % self.name)
        if (self.src == self.dst).any():
            raise CouplingError('%s: a cell cannot receive from itself'
                                % self.name)
        if len(set(zip(self.src.tolist(), self.dst.tolist()))) \
                != len(self.src):
            raise CouplingError('%s: duplicate edge' % self.name)
        self.group.edges = (self.src, self.dst)
        self.gwm = bool(gwm)
        self._check_config()
        self.outputs = (VarSpec('withdrawn_remote', 'mm', ('cell',),
                                kind='state', grid=GRID),
                        VarSpec('withdrawn_tremote', 'mm', ('cell',),
                                kind='state', grid=GRID,
                                description='water this cell received '
                                            'from its source cells'))
        self.owned_state = ('%s.remote' % self.name,)

    def _init(self, context):
        G = self.group
        r = G.build()
        self.N = r.n
        self.loc_c = G.need('wateruse_local', self.name)
        if self.src.max() >= r.n or self.dst.max() >= r.n or \
                min(self.src.min(), self.dst.min()) < 0:
            raise CouplingError('%s: edge cell outside the router'
                                % self.name)
        self.plan = WR.plan_for_router(r, self.src, self.dst)
        self.slot = None            # built at the first sweep (needs the
        #                             local connector's wu_var)

    def sweep_slot(self, lv, slot, cells):
        if slot != 'wu_remote':
            return
        G, r = self.group, self.group.router
        if self.slot is None:
            self.slot = WR.RemoteSlot(self.plan, r, self.loc_c.st,
                                      self.loc_c.loc, gwm=self.gwm)
        sl = self.slot
        if lv == 0:
            sl.begin()
            efr = G.roles.get('efr')
            sl.efr_q = None if efr is None or efr.moist is None \
                else efr.q_sweep()
        sl.loc = self.loc_c.loc
        sl.st = self.loc_c.st
        i = sl.by_group.get(sl.g)
        if i is None:
            sl.g += 1
            return
        b = self.plan.batches[i]
        a_src = r.area.index_select(0, b.src)
        d0 = (r.deficit * r.area).sum() / 1000.
        m0 = G.ring_mass(b.src)
        o = sl(r, cells)
        G.book_ring(b.src, m0)
        d1 = (r.deficit * r.area).sum() / 1000.
        G.give('pb_remote', d0 - d1)
        # paybacks are differences of deficits (ledger scale)
        self.acc('dmag', d0.abs() + d1.abs())
        k = a_src / 1000.
        self.acc('i_ring', (o['q_w'] * STEP_S).sum())
        self.acc('q_ret', (o['q_r'] * STEP_S).sum())
        self.acc('q_pb', (o['payback_cell'] * k).sum())
        self.acc('e_cons', (o['consumed_cell'] * k).sum())
        self.flag(o['violation'])
        self.flag(o['empty_ring_withdrawal'])

    def put_data(self):
        i_ring = self.pop('i_ring')
        if isinstance(i_ring, float):
            self.pop('dmag')
            return
        q_ret, q_pb, cons = self.pop('q_ret'), self.pop('q_pb'), \
            self.pop('e_cons')
        scale = i_ring + q_ret + q_pb + cons + self.pop('dmag')
        bal = Balance(I_ext=i_ring, Q_out=q_ret + q_pb, E_other=cons,
                      n_terms=8 * len(self.src), scale=scale,
                      detail=dict(ring_withdrawn=i_ring, ring_returned=q_ret,
                                  payback=q_pb, consumed=cons))
        r = self.group.router
        exp = dict(withdrawn_remote=r.to_cells(
            self.loc_c.st.withdrawn_remote.sum(1)),
            withdrawn_tremote=r.to_cells(
                self.loc_c.st.withdrawn_tremote.sum(1)))
        self._pending = (bal, exp, {k: x.clone().unsqueeze(0)
                                    for k, x in exp.items()})

    # -- state ---------------------------------------------------------------
    _KEYS = tuple(k for k in WR.RemoteState.TOTALS + WR.RemoteState.TMPS
                  if k not in ('returned', 'consumed'))

    def _state(self):
        st = self.loc_c.st
        return {k: getattr(st, k).clone() for k in self._KEYS}

    def _load(self, blob):
        st = self.loc_c.st
        for k in self._KEYS:
            setattr(st, k, blob[k].to(self.dev).clone())
