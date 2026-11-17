# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_dams: VIC-WUR global (C2) and local (C3) dams as a connector.

P2 slots: `sweep_slot(level, 'local_dam')` (local_dam_run: inflow from the
cell's OUT_RUNOFF, storage, release; the write-back to OUT_RUNOFF happens
after rout_run read it) and `sweep_slot(level, 'global_dam')`
(global_dam_run: inflow from the cell discharge after local water use, the
Hanasaki-type plan at op_month, the ring head rewritten with the release).
O `put_data`.

The dams of one router level form one kernel set (GlobalDams/LocalDams of
torchvic.vicwur.dams); a cell holds at most one dam of each kind (several
dams of a cell compound through the ring in class order; not wired here).

Options
    dams     dict of arrays, one entry per dam: cell, capacity (hm3),
             inflow_frac, year (build year), is_global, and for global dams
             service [D, M] (cell index, -1 padded) and service_frac [D, M];
             or `params` (vicwur.params.DamParams)
    alpha, beta, gamma     the operation constants (defaults 0.85, 0.6, 5)
    start_date             'YYYY-MM-DD' of t = 0; calendar (default
                           proleptic_gregorian); nsteps (default from t_end)
    exact                  True keeps the C dam-cell-area demand quirk

State (checkpoint): every dam's state_dict (storage, active, histories,
op_release, op_month, ...).

Ledger (m3 per day): I_ext = dam inflow (global: taken from the river ring,
local: a fraction of the cell runoff); Q_out = release + what local water
use took from local-dam storage; dS = storage change. The local dams'
inflow is a second use of runoff the router already routed: it is
named in detail['local_inflow_double_count'].
"""
import datetime

import numpy as np
import torch

from ...core import ordered as _ordered

from ..connector import Balance, ComputeError
from ..spec import CouplingError, VarSpec
from ...vicwur import dams as DM
from ...vicwur import dams_kernels as DK
from ...vicwur import fused as FZ
from ...vicwur import wateruse_kernels as WK
from ...vicwur.forcing import step_dates
from .vicwur_common import (GRID, STEP_S, WurConnector, as_f64, as_i64, put)

M3 = DM.M3_PER_HM3


class _Set(object):
    def __init__(self, dams, pos, cid, svc_pos=None, svc_frac=None,
                 svc_valid=None, pg=None):
        self.dams, self.pos, self.cid = dams, pos, cid
        self.svc_pos, self.svc_frac, self.svc_valid = (svc_pos, svc_frac,
                                                       svc_valid)
        self.pg = pg


class VicWurDamsConnector(WurConnector):
    name = 'vicwur_dams'
    version = '1'
    kind = 'dams'
    role = 'dams'
    flags = dict(dams=True)

    def __init__(self, dams=None, params=None, alpha=0.85, beta=0.6,
                 gamma=5.0, start_date='2000-01-01',
                 calendar='proleptic_gregorian', nsteps=None, exact=False,
                 group=None, **config):
        super().__init__(group=group, **config)
        if params is not None:
            dams = dict(cell=params.dam_cell, capacity=params.capacity,
                        inflow_frac=params.inflow_fraction,
                        year=params.year, is_global=params.is_global,
                        service=params.service,
                        service_frac=params.service_frac)
        if not dams:
            raise CouplingError('%s: give dams (or params)' % self.name)
        miss = [k for k in ('cell', 'capacity', 'inflow_frac', 'year',
                            'is_global') if k not in dams]
        if miss:
            raise CouplingError('%s: dams lacks %s' % (self.name, miss))
        npy = lambda x: np.asarray(x.cpu() if torch.is_tensor(x) else x)
        self.d = {k: npy(v) for k, v in dams.items()}
        D = len(self.d['cell'])
        for k, v in self.d.items():
            if v.shape[0] != D:
                raise CouplingError('%s: dams.%s has %d entries, not %d'
                                    % (self.name, k, v.shape[0], D))
        if (self.d['capacity'] <= 0).any():
            raise CouplingError('%s: capacity must be positive' % self.name)
        if ((self.d['inflow_frac'] < 0) | (self.d['inflow_frac'] > 1)).any():
            raise CouplingError('%s: inflow_frac must be in [0, 1]'
                                % self.name)
        if not 0 < alpha < 1:
            raise CouplingError('%s: alpha must be in (0, 1) (the C code '
                                'divides by zero at 0 and 1)' % self.name)
        for g in (True, False):
            c = self.d['cell'][self.d['is_global'].astype(bool) == g]
            if len(np.unique(c)) != len(c):
                raise CouplingError('%s: two %s dams in one cell (one per '
                                    'cell and kind)' % (self.name,
                                                        'global' if g
                                                        else 'local'))
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        try:
            y, m, dd = [int(x) for x in str(start_date)[:10].split('-')]
            datetime.date(y, m, dd)
        except Exception:                                  # noqa: BLE001
            raise CouplingError('%s: start_date must be YYYY-MM-DD'
                                % self.name)
        self.start, self.calendar = (y, m, dd, 0), calendar
        self.nsteps, self.exact = nsteps, bool(exact)
        self._check_config()
        self.outputs = (VarSpec('storage', 'm3', ('dam',), kind='state',
                               grid=GRID),)
        self.owned_state = ('%s.storage' % self.name,)
        self.gsets, self.lsets = {}, {}

    # -- build ---------------------------------------------------------------
    def _init(self, context):
        G = self.group
        r = G.build()
        T = self.nsteps or int(np.ceil(G.t_end / STEP_S))
        if not T:
            raise CouplingError('%s: give nsteps (the run length is not '
                                'known here)' % self.name)
        T += 3
        ys, ms, _ = step_dates(self.start, T, 1, self.calendar)
        self.cal = DM.make_calendar(ys, ms, self.calendar, 1, G.dev)
        self.T = T
        pos = r.pos.cpu().numpy()
        gptr = np.asarray(r._gptr)
        d = self.d
        cell = d['cell'].astype(np.int64)
        if cell.min() < 0 or cell.max() >= self.N_cells(r):
            raise CouplingError('%s: dam cell outside the router' % self.name)
        lev = np.searchsorted(gptr, pos[cell], 'right') - 1
        for isg, store in ((True, self.gsets), (False, self.lsets)):
            sel = np.nonzero(d['is_global'].astype(bool) == isg)[0]
            for g in sorted(set(lev[sel].tolist())):
                ix = sel[lev[sel] == g]
                ix = ix[np.argsort(pos[cell[ix]])]
                kw = dict(capacity=d['capacity'][ix],
                          inflow_frac=d['inflow_frac'][ix],
                          build_year=d['year'][ix].astype(np.int64),
                          alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                          steps_per_day=1, exact=self.exact, device=G.dev)
                p = pos[cell[ix]]
                s = _Set(None, as_i64(p, G.dev), as_i64(cell[ix], G.dev),
                         pg=as_i64(p - gptr[g], G.dev))
                if isg:
                    s.dams = DM.GlobalDams(**kw)
                    if 'service' in d:
                        sv = d['service'][ix].astype(np.int64)
                        ok = sv >= 0
                        s.svc_pos = as_i64(np.where(ok, pos[np.maximum(
                            sv, 0)], 0), G.dev)
                        s.svc_valid = torch.as_tensor(ok, device=G.dev)
                        s.svc_frac = as_f64(np.where(
                            ok, d['service_frac'][ix], 0.0), G.dev)
                else:
                    s.dams = DM.LocalDams(**kw)
                store[g] = s
        self._S_prev = self._storage()

    @staticmethod
    def N_cells(r):
        return r.n

    def local_level(self, lv):
        """(set, slot [C], has [C], pg) of the local dams of router group
        lv for the local water-use kernel, or None."""
        s = self.lsets.get(lv)
        if s is None:
            return None
        r = self.group.router
        c = int(r._gptr[lv + 1] - r._gptr[lv])
        slot = torch.zeros(c, dtype=torch.int64, device=self.dev)
        has = torch.zeros(c, dtype=torch.bool, device=self.dev)
        slot[s.pg] = torch.arange(len(s.pg), device=self.dev)
        has[s.pg] = True
        return s, slot, has, s.pg

    def _storage(self):
        t = self.zero()
        for s in list(self.gsets.values()) + list(self.lsets.values()):
            t = t + s.dams.storage.sum() * M3
        return t

    # -- hooks ---------------------------------------------------------------
    def _st(self):
        return self.cal.at(min(self.group.step, self.T - 1))

    # -- P2 around the sweep (order.PhaseScheduler._sweep) -------------------
    def sweep_begin(self):
        """A day without a month change runs every set as one kernel
        (dams_kernels) and keeps its ledger terms in a device buffer, one
        row per level (summed in sweep_end): no Python bookkeeping inside
        the sweep, which can then be a CUDA graph."""
        sets = list(self.gsets.values()) + list(self.lsets.values())
        self._fz_day = bool(
            sets and self.dev.type == 'cuda' and FZ.ENABLED and FZ.DAMS
            and self._st().new_month_host is False
            and all(float(x.dams.beta) not in DK.POW_SPECIAL for x in sets))
        if not self._fz_day:
            return
        n = len(self.group.levels)
        if getattr(self, '_abuf', None) is None or \
                self._abuf.shape[0] != n:
            self._abuf = torch.zeros(n, 4, dtype=torch.float64,
                                     device=self.dev)
        else:
            self._abuf.zero_()
        self._ring_buf().zero_()
        for x in sets:
            DK.own(x.dams)
        if self.lsets:
            v = self.view
            if v is None or v.runoff is None:
                raise ComputeError('%s: the step view carries no runoff'
                                   % self.name)
            ro = as_f64(v.runoff, self.dev)
            if getattr(self, '_ro', None) is None or \
                    self._ro.shape != ro.shape:
                self._ro = torch.empty(ro.shape, dtype=torch.float64,
                                       device=self.dev)
            self._ro.copy_(ro)

    def sweep_end(self):
        if not getattr(self, '_fz_day', False):
            return
        A, G = self._abuf, self.group
        if self.lsets:
            self.acc('l_in', A[:, 0].sum())
            self.acc('l_rel', A[:, 1].sum())
        if self.gsets:
            self.acc('g_in', A[:, 2].sum())
            self.acc('g_rel', A[:, 3].sum())
            # the global dams' ring change per level (book_ring)
            d = self._ring_d
            z = torch.zeros_like(d)
            G.add('ring_in', torch.where(d > 0, d, z).sum())
            G.add('ring_out', torch.where(d < 0, -d, z).sum())

    def graph_key(self):
        if not getattr(self, '_fz_day', False):
            return None
        G = self.group
        loc, efr = G.roles.get('wateruse_local'), G.roles.get('efr')
        ts = [self._abuf, self._ring_buf()]
        if self.lsets:
            ts.append(self._ro)
        if loc is not None:
            ts += [loc.loc.dsf, loc.act]
        if efr is not None:
            ts.append(efr.q)
        for x in list(self.gsets.values()) + list(self.lsets.values()):
            D = x.dams
            ts += [D.inflow, D.demand, D.efr, D.total_inflow,
                   D.total_demand, D.total_efr, D.register_steps,
                   D.storage, D.release, D.op_release, D.capacity,
                   D.inflow_frac, D.active]
        return tuple(t.data_ptr() for t in ts)

    def _ring_buf(self):
        n = len(self.group.levels)
        if getattr(self, '_ring_d', None) is None or \
                self._ring_d.shape[0] != n:
            self._ring_d = torch.zeros(n, dtype=torch.float64,
                                       device=self.dev)
        return self._ring_d

    def sweep_slot(self, lv, slot, cells):
        G, r = self.group, self.group.router
        fz = getattr(self, '_fz_day', False)
        if slot == 'local_dam':
            s = self.lsets.get(lv)
            if s is None:
                return
            if fz:
                # a day without a month change: one kernel (dams_kernels)
                DK.local_dam(s.dams, self._ro, s.cid, s.pos, r.area)
                D = s.dams
                self._abuf[lv, 0].copy_((D.inflow * M3).sum())
                self._abuf[lv, 1].copy_(
                    (torch.where(D.active, D.release,
                                 torch.zeros_like(D.release)) * M3).sum())
                return
            v = self.view
            if v is None or v.runoff is None:
                raise ComputeError('%s: the step view carries no runoff'
                                   % self.name)
            ro = as_f64(v.runoff, self.dev).index_select(0, s.cid)
            area = r.area.index_select(0, s.pos)
            s.dams.run(self._st(), ro, area)
            a = s.dams.active
            z = torch.zeros_like(s.dams.release)
            self.acc('l_in', (s.dams.inflow * M3).sum())
            self.acc('l_rel', (torch.where(a, s.dams.release, z) * M3).sum())
        elif slot == 'global_dam':
            s = self.gsets.get(lv)
            if s is None:
                return
            loc = G.roles.get('wateruse_local')
            dem = None
            if loc is not None and s.svc_pos is not None:
                area = r.area.index_select(0, s.pos)
                if not self.exact:
                    area = r.area.index_select(0, s.svc_pos.reshape(-1)
                                               ).reshape(s.svc_pos.shape)
                if self.dev.type == 'cuda' and FZ.ENABLED and FZ.DAMS:
                    dem = WK.service_demand(
                        loc.loc.dsf, loc.act.view(torch.uint8), s.svc_pos,
                        s.svc_frac, s.svc_valid.view(torch.uint8), area)
                else:
                    dem = DM.service_demand(loc.loc.dsf, loc.act, s.svc_pos,
                                            s.svc_frac, s.svc_valid, area)
            efr = G.roles.get('efr')
            m0 = G.ring_mass(s.pos)
            if fz:
                DK.global_dam(r, s.dams, s.pos, s.cid, STEP_S, dem,
                              None if efr is None else efr.q)
                D = s.dams
                self._ring_buf()[lv].copy_((G.ring_mass(s.pos) - m0)
                                           * STEP_S)
                self._abuf[lv, 2].copy_((D.inflow * M3).sum())
                self._abuf[lv, 3].copy_(
                    (torch.where(D.active, D.release,
                                 torch.zeros_like(D.release)) * M3).sum())
                return
            phys = r.logical(torch.arange(r.nsub, device=self.dev))
            ring = r.ring[phys].index_select(1, s.pos).T.contiguous()
            dis = r.discharge.index_select(0, s.pos)
            eq = (None if efr is None
                  else efr.q.index_select(0, s.cid))
            ring2, dis2 = s.dams.run(self._st(), ring, dis, STEP_S, dem, eq)
            r.ring[phys[:, None], s.pos[None, :]] = ring2.T
            r.discharge.index_copy_(0, s.pos, dis2)
            G.book_ring(s.pos, m0)
            a = s.dams.active
            z = torch.zeros_like(s.dams.release)
            self.acc('g_in', (s.dams.inflow * M3).sum())
            self.acc('g_rel', (torch.where(a, s.dams.release, z) * M3).sum())

    def put_data(self):
        l_in = self.pop('l_in')
        g_in = self.pop('g_in')
        l_rel, g_rel = self.pop('l_rel'), self.pop('g_rel')
        wu = self.group.take('ldam_wu')
        if isinstance(l_in, float) and isinstance(g_in, float):
            self._S_prev = self._storage()
            return
        S1 = self._storage()
        dS = S1 - self._S_prev
        i_ext = l_in + g_in
        q = l_rel + g_rel + wu
        scale = abs(i_ext) + abs(q) + abs(self._S_prev) + abs(S1)
        self._S_prev = S1
        n = 12 * max(1, len(self.gsets) + len(self.lsets))
        bal = Balance(I_ext=i_ext, Q_out=q, dS=dS, n_terms=n, scale=scale,
                      detail=dict(local_inflow_double_count=l_in,
                                  global_inflow=g_in,
                                  release=l_rel + g_rel, global_release=g_rel,
                                  wateruse=wu))
        exp = dict(storage=self._cell_vector())
        self._pending = (bal, exp, {k: v.unsqueeze(0)
                                    for k, v in exp.items()})

    def _cell_vector(self):
        out = torch.zeros(self.group.router.n, dtype=torch.float64,
                          device=self.dev)
        for s in list(self.gsets.values()) + list(self.lsets.values()):
            _ordered.index_add_(out, 0, s.cid, s.dams.storage * M3)
        return out

    # -- w3_restart ----------------------------------------------------------
    def _w3_reset(self, step):
        """VIC-WUR 3.0 neither stores nor restores a dam
        (plugin_simulation.c:231-233, plugin_setup.c:259-261) and on a
        restart skips the cold start's S = alpha C (dam_generate_default_state
        is called on the cold path only, vic_populate_model_state.c:52-68):
        every dam is initialize_dam_var's (dam_init_library.c:33-62: inactive,
        storage 0, no history, op_month -1). And the new run's first step is
        its `current == 0`: no month change there (dam_register.c:287, 311),
        so no activation, history shift or op_month at the restart step."""
        for s in list(self.gsets.values()) + list(self.lsets.values()):
            D = s.dams
            sd = {k: torch.zeros_like(v) for k, v in D.state_dict().items()}
            sd['op_month'] = torch.full_like(sd['op_month'], DM.NODATA_DAM)
            D.load_state_dict(sd)
        if step < self.T:
            c = self.cal
            nm, ny = c.new_month.clone(), c.new_year.clone()
            nm[step] = False
            ny[step] = False
            host = list(c.new_month_host)
            host[step] = False
            self.cal = c._replace(new_month=nm, new_year=ny,
                                  new_month_host=host)
        self._S_prev = self._storage()

    # -- state ---------------------------------------------------------------
    def _state(self):
        return dict(g={k: s.dams.state_dict() for k, s in
                       self.gsets.items()},
                    l={k: s.dams.state_dict() for k, s in
                       self.lsets.items()})

    def _load(self, blob):
        for k, s in self.gsets.items():
            s.dams.load_state_dict(blob['g'][k])
        for k, s in self.lsets.items():
            s.dams.load_state_dict(blob['l'][k])
        self._S_prev = self._storage()
