# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_wateruse_local: VIC-WUR sector demand (C4) and local water use
(C5) as a connector.

F `pre_force` (wu_forcing: the held sector demand, groundwater fraction and
consumption fraction), P1 `pass1` (wu_run: reset and the groundwater /
surface split), P2 `sweep_slot(level, 'wu_local')` (wu_run_local: groundwater
availability from the bottom layer, whole-ring river availability with the
EFR rule, local-dam availability, pro-rata division over the sectors,
withdrawals, consumption and returns, pumping of the bottom layer through
the step view, deficit payback, the ring edit and the local-dam
withdrawal), O `put_data`.

Options (sector axis MUN, ENE, MAN, LIV, IRR; mm per model step, [cell, 5])
    demand, gw_frac, cons_frac   Series specs
    active       [5] bool, the sectors that exist (default all but IRR)
    soil         {'resid_mm': [cell]}: resid_moist * depth * 1000 of the
                 bottom layer
    gwm          the VIC-WUR 3.0 GWM switch (no groundwater, no deficit)

State (checkpoint): the held forcing, wu_var (demand, withdrawals, returned,
consumed) and the remote-use totals/temporaries that local use writes.

Ledger (m3 per day): I_ext = river water taken from the ring + local-dam
water + groundwater actually pumped from the soil; Q_out = water returned to
the ring + the deficit payback from the returns; E_other = consumption (it
leaves the model; it is not ET). A pump that the soil could not supply, or a
clipped ring, shows up as an open balance.
"""
import torch

from ..connector import Balance, ComputeError
from ..spec import CouplingError, VarSpec
from ...model.routing_cascade import _div
from ...vicwur import wateruse as WU
from ...vicwur import wateruse_remote as WR
from ...vicwur import fused as FZ
from ...vicwur import wateruse_kernels as WK
from .vicwur_common import (GRID, STEP_S, Series, WurConnector, as_f64, put)

S = WU.NSECT
M3 = WU.M3_PER_HM3


class VicWurWateruseLocalConnector(WurConnector):
    name = 'vicwur_wateruse_local'
    version = '1'
    kind = 'wateruse'
    role = 'wateruse_local'
    flags = dict(wateruse=True, local_with=True)

    def __init__(self, demand=None, gw_frac=None, cons_frac=None,
                 active=None, soil=None, gwm=False, group=None, **config):
        super().__init__(group=group, **config)
        if demand is None or gw_frac is None or cons_frac is None \
                or soil is None or 'resid_mm' not in soil:
            raise CouplingError('%s: give demand, gw_frac, cons_frac and '
                                'soil.resid_mm' % self.name)
        self.f = {k: Series(v, k) for k, v in (('demand', demand),
                                               ('gw_frac', gw_frac),
                                               ('cons_frac', cons_frac))}
        self.act_cfg = (active if active is not None
                        else [True, True, True, True, False])
        if len(self.act_cfg) != S:
            raise CouplingError('%s: active needs %d sectors' % (self.name,
                                                                 S))
        self.resid_cfg, self.gwm = soil['resid_mm'], bool(gwm)
        self._check_config()
        self.outputs = (VarSpec('withdrawn', 'mm', ('cell',), kind='state',
                                grid=GRID),)
        self.owned_state = ('%s.wu_var' % self.name,)

    def _init(self, context):
        G, d = self.group, self.dev
        r = G.build()
        G.need('routing', self.name)
        n = self.N = r.n
        self.resid = as_f64(self.resid_cfg, d)
        if self.resid.shape != (n,):
            raise CouplingError('%s: soil.resid_mm must be [N]' % self.name)
        for f in self.f.values():
            f.to(d)
        z = lambda: torch.zeros(n, S, dtype=torch.float64, device=d)
        self.dem, self.gfr, self.cf = z(), z(), z()
        self.act = torch.as_tensor(self.act_cfg, dtype=torch.bool,
                                   device=d).expand(n, S).contiguous()
        self.st = WR.RemoteState(n, d)
        self.loc = WR.LocalUse(z(), z(), z(), z(), z(), self.cf, self.act)
        self.dams = G.roles.get('dams')
        self.efr = G.roles.get('efr')
        self._lv = {}

    # -- hooks ---------------------------------------------------------------
    def _rows(self, name):
        x = self.f[name].get(self.group.step)
        if x.shape != (self.N, S):
            raise ComputeError('%s: %s must be [cell, 5]' % (self.name,
                                                             name))
        return x.index_select(0, self.group.router.perm)

    def pre_force(self):
        # into the held tensors (their memory is the CUDA sweep graph's)
        for k, nm in (('dem', 'demand'), ('gfr', 'gw_frac'),
                      ('cf', 'cons_frac')):
            _into(self, k, self._rows(nm))
        self.loc.cons_frac = self.cf

    def pass1(self):
        self.st.reset()
        _zero(self.st, 'returned', self.dem)
        _zero(self.st, 'consumed', self.dem)
        dgw, dsf = WU.demand(self.dem, self.gfr, self.act, self.gwm)
        L = self.loc
        _into(L, 'dgw', dgw)
        _into(L, 'dsf', dsf)
        for k in ('wgw', 'wsf', 'wdm'):
            _zero(L, k, dgw)
        self.acc('w_ring', self.zero())
        self.efr = self.group.roles.get('efr')
        r = self.group.router
        self._fused = FZ.ENABLED and FZ.WU and r.device.type == 'cuda'
        if self._fused:
            # the cell terms of the day's ledger row (wateruse_kernels)
            if getattr(self, '_accbuf', None) is None:
                self._accbuf = torch.zeros(len(WK.ACC_ROWS), r.n,
                                           dtype=torch.float64,
                                           device=r.device)
            else:
                self._accbuf.zero_()

    def _dam_args(self, lv, C):
        if self.dams is None:
            return None
        if lv not in self._lv:
            self._lv[lv] = self.dams.local_level(lv)
        return self._lv[lv]

    def sweep_slot(self, lv, slot, cells):
        if slot != 'wu_local':
            return
        if getattr(self, '_fused', False):
            return self._sweep_fused(lv, cells)
        G, r, v = self.group, self.group.router, self.view
        if v is None:
            raise ComputeError('%s: no step view' % self.name)
        dt = STEP_S
        ci = r.perm.index_select(0, cells)
        row = lambda x: x.index_select(0, cells)
        moist = v.layer_at(-1, ci)
        ice = v.ice_at(-1, ci).unsqueeze(-1)
        ff = torch.ones_like(moist[:, 0, 0]).unsqueeze(1)
        af, cv = v.af.index_select(0, ci), v.cv.index_select(0, ci)
        area = row(r.area)
        deficit = row(r.deficit)
        dam = self._dam_args(lv, cells.numel())
        kw = {}
        if dam is not None:
            s, slot_i, has, pg = dam
            st0 = s.dams.storage.index_select(0, slot_i)
            kw = dict(dam_storage=st0.unsqueeze(1),
                      dam_active=(s.dams.active.index_select(0, slot_i)
                                  & has).unsqueeze(1),
                      dam_local=has.unsqueeze(1))
        if self.efr is not None and self.efr.moist is not None:
            kw.update(efr_moist=self.efr.moist.index_select(0, ci),
                      efr_q=self.efr.q.index_select(0, ci),
                      discharge=row(r.discharge))
        out = WU.wu_local_step(
            row(self.dem), row(self.gfr), row(self.cf), moist, ice, ff,
            self.resid.index_select(0, ci), af, cv,
            G.ring_of(cells), dt, area, active=row(self.act),
            deficit=deficit, gwm=self.gwm, **kw)
        L, st = self.loc, self.st
        L.dgw.index_copy_(0, cells, out['demand_gw'])
        L.dsf.index_copy_(0, cells, out['demand_surf'])
        L.wgw.index_copy_(0, cells, out['withdrawn_gw'])
        L.wsf.index_copy_(0, cells, out['withdrawn_surf'])
        L.wdm.index_copy_(0, cells, out['withdrawn_dam'])
        st.returned.index_copy_(0, cells, out['returned'])
        st.consumed.index_copy_(0, cells, out['consumed'])
        r.deficit.index_copy_(0, cells, out['deficit'])
        a_m3 = area / 1000.
        pb = ((deficit - out['deficit']) * a_m3).sum()
        # a difference of deficits: rounded relative to them (ledger scale)
        self.acc('dmag', ((deficit.abs() + out['deficit'].abs())
                          * a_m3).sum())
        # groundwater pumping through the step view
        if not self.gwm:
            d, _ = WU.pump_depth(out['av_gw'], out['A_gw'], out['cell_gw'],
                                 af, cv)
            taken = v.withdraw_cells(-1, ci, d)
            w = v.weight_at(ci)
            pumped = ((taken * w).sum((1, 2)) * a_m3).sum()
            self.flag(torch.isnan(d).any())
        else:
            pumped = self.zero()
        # the river: the ring edit (wu_run_local.c:469-540)
        ws, rr = out['cell_surf'], out['returned_to_river']
        gate = (ws > 0) | (rr > 0)
        q_w = _div(_div(ws, 1000.) * area, dt)
        q_r = _div(_div(rr, 1000.) * area, dt)
        d0, s0 = row(r.discharge), row(r.stream)
        m0 = G.ring_mass(cells)
        bad = r.scale_prop(cells, withdraw=q_w, ret=q_r, lo=0)
        r.recompute(cells)
        r.discharge.index_copy_(0, cells, torch.where(gate, row(r.discharge),
                                                      d0))
        r.stream.index_copy_(0, cells, torch.where(gate, row(r.stream), s0))
        G.book_ring(cells, m0)
        self.flag(bad)
        # local dam withdrawal
        wd = self.zero()
        if dam is not None:
            new = out['dam_storage'][:, 0].index_select(0, pg)
            wd = ((s.dams.storage - new) * M3).sum()
            s.dams.storage = new
        G.give('ldam_wu', wd)
        G.give('pb_local', pb)
        self.acc('i_ring', (q_w * dt).sum())
        self.acc('i_dam', wd)
        self.acc('i_gw', pumped)
        self.acc('q_ret', (q_r * dt).sum())
        self.acc('q_pb', pb)
        self.acc('e_cons', (out['consumed'].sum(1) * a_m3).sum())

    # -- the CUDA path: one level kernel ------------------------------------
    def _fused_setup(self):
        r = self.group.router
        gp = [int(x) for x in r._gptr]
        self._spans = [(gp[g], gp[g + 1] - gp[g]) for g in range(len(gp) - 1)]
        self._lvl = torch.repeat_interleave(
            torch.arange(len(self._spans), device=r.device),
            torch.as_tensor([c for _, c in self._spans], device=r.device))
        self._act8 = self.act.view(torch.uint8)
        self._dam8 = {}

    def _sweep_fused(self, lv, cells):
        """The level's wu_run_local in one kernel (wateruse_kernels), bit
        for bit the torch path; the ledger terms are summed after the last
        level."""
        G, r, v = self.group, self.group.router, self.view
        if v is None:
            raise ComputeError('%s: no step view' % self.name)
        if getattr(self, '_spans', None) is None or \
                self._act8.data_ptr() != self.act.data_ptr():
            self._fused_setup()
        s0, C = self._spans[lv]
        dt = STEP_S
        # the ring sum of availability_surf: in the kernel (torch's order)
        # for short rings, else torch's own reduction of the same terms
        asum = (None if r.R <= WK.REPSUM_MAX and FZ.SUM
                else WK.surf_terms(r, s0, C, dt).T.sum(1))
        efr = self.efr
        efr_on = efr is not None and efr.moist is not None
        agw = cgw = None
        if not self.gwm:
            ci = r.perm.index_select(0, cells)
            moist = v.layer_at(-1, ci)
            ice = v.ice_at(-1, ci).unsqueeze(-1)
            ff = torch.ones_like(moist[:, 0, 0]).unsqueeze(1)
            af, cv = v.af.index_select(0, ci), v.cv.index_select(0, ci)
            av_gw, agw = WU.availability_gw(
                moist, ice, ff, self.resid.index_select(0, ci), af, cv,
                efr.moist.index_select(0, ci) if efr_on else None)
            cgw = torch.empty_like(agw)
        dam = self._dam_args(lv, C)
        dk = None
        if dam is not None:
            s, slot_i, has, pg = dam
            h8 = self._dam8.get(lv)
            if h8 is None:
                h8 = self._dam8[lv] = has.to(torch.uint8)
            dk = (s.dams.storage, s.dams.active.view(torch.uint8), slot_i,
                  h8)
        L, st = self.loc, self.st
        WK.wu_local(r, s0, C, dt, asum, self.dem, self.gfr, self.cf,
                    self._act8, self._accbuf,
                    (L.dgw, L.dsf, L.wgw, L.wsf, L.wdm, st.returned,
                     st.consumed), agw=agw,
                    efr_q=efr.q if efr_on else None, dam=dk, cgw=cgw,
                    gwm=self.gwm)
        if not self.gwm:
            a_m3 = r.area.index_select(0, cells) / 1000.
            d, _ = WU.pump_depth(av_gw, agw, cgw, af, cv)
            taken = v.withdraw_cells(-1, ci, d)
            w = v.weight_at(ci)
            self.acc('i_gw', ((taken * w).sum((1, 2)) * a_m3).sum())
            self.flag(torch.isnan(d).any())

    # -- P2 around the sweep (order.PhaseScheduler._sweep) -------------------
    def sweep_end(self):
        if getattr(self, '_fused', False):
            self._fused_close()

    def graph_key(self):
        """The tensors the fused GWM sweep reads and writes (None: no
        CUDA graph of the sweep, e.g. the pumping through the day's step
        view without GWM)."""
        if not getattr(self, '_fused', False) or not self.gwm or \
                getattr(self, '_spans', None) is None or \
                'wateruse_remote' in self.group.roles:
            return None
        ts = [self.dem, self.gfr, self.cf, self._act8, self.resid,
              self._accbuf, self.st.returned, self.st.consumed]
        ts += [getattr(self.loc, k) for k in self._LOC]
        efr = self.efr
        if efr is not None and efr.moist is not None:
            ts.append(efr.q)
        for lv in range(len(self._spans)):
            dam = self._dam_args(lv, 0)
            if dam is not None:
                s, slot_i, has, pg = dam
                ts += [s.dams.storage, s.dams.active, slot_i,
                       self._dam8.get(lv, has)]
        return tuple(t.data_ptr() for t in ts) + (self.group.router.R,)

    def _fused_close(self):
        """The day's ledger terms from the per-cell buffer."""
        G, A = self.group, self._accbuf
        pb, dmag, i_ring, q_ret, cons, i_dam = (A[k].sum() for k in range(6))
        dl = torch.zeros(len(self._spans), dtype=torch.float64,
                         device=A.device).index_add_(0, self._lvl, A[6]) \
            * STEP_S
        z = torch.zeros_like(dl)
        G.add('ring_in', torch.where(dl > 0, dl, z).sum())
        G.add('ring_out', torch.where(dl < 0, -dl, z).sum())
        self.flag(A[7] > 0)
        G.give('ldam_wu', i_dam)
        G.give('pb_local', pb)
        self.acc('dmag', dmag)
        self.acc('i_ring', i_ring)
        self.acc('i_dam', i_dam)
        if self.gwm:
            self.acc('i_gw', self.zero())
        self.acc('q_ret', q_ret)
        self.acc('q_pb', pb)
        self.acc('e_cons', cons)

    def put_data(self):
        i_ring = self.pop('i_ring')
        if isinstance(i_ring, float):
            self.pop('w_ring')
            self.pop('dmag')
            return
        self.pop('w_ring')
        i_dam, i_gw = self.pop('i_dam'), self.pop('i_gw')
        q_ret, q_pb, cons = self.pop('q_ret'), self.pop('q_pb'), \
            self.pop('e_cons')
        scale = i_ring + i_dam + i_gw + q_ret + q_pb + cons + \
            self.pop('dmag')
        n = self.N * 3 * S
        bal = Balance(I_ext=i_ring + i_dam + i_gw, Q_out=q_ret + q_pb,
                      E_other=cons, n_terms=n, scale=scale,
                      detail=dict(ring_withdrawn=i_ring, dam_withdrawn=i_dam,
                                  groundwater=i_gw, ring_returned=q_ret,
                                  payback=q_pb, consumed=cons))
        w = self.loc.wsf + self.loc.wgw + self.loc.wdm
        r = self.group.router
        exp = dict(withdrawn=r.to_cells(w.sum(1)))
        self._pending = (bal, exp, {k: x.clone().unsqueeze(0)
                                    for k, x in exp.items()})

    # -- state ---------------------------------------------------------------
    _LOC = ('dgw', 'dsf', 'wgw', 'wsf', 'wdm')

    def _state(self):
        d = dict(dem=self.dem.clone(), gfr=self.gfr.clone(),
                 cf=self.cf.clone(), returned=self.st.returned.clone(),
                 consumed=self.st.consumed.clone())
        d.update({k: getattr(self.loc, k).clone() for k in self._LOC})
        return d

    def _load(self, blob):
        dv = self.dev
        self.dem, self.gfr = blob['dem'].to(dv).clone(), \
            blob['gfr'].to(dv).clone()
        self.cf = blob['cf'].to(dv).clone()
        self.loc.cons_frac = self.cf
        self.st.returned = blob['returned'].to(dv).clone()
        self.st.consumed = blob['consumed'].to(dv).clone()
        for k in self._LOC:
            setattr(self.loc, k, blob[k].to(dv).clone())


def _into(obj, k, x):
    """obj.k = x, into obj.k's memory when it has x's shape and type."""
    y = getattr(obj, k, None)
    if torch.is_tensor(y) and y.shape == x.shape and y.dtype == x.dtype \
            and y.device == x.device and y.is_contiguous():
        y.copy_(x)
    else:
        setattr(obj, k, x.clone(memory_format=torch.contiguous_format))


def _zero(obj, k, like):
    y = getattr(obj, k, None)
    if torch.is_tensor(y) and y.shape == like.shape and \
            y.dtype == like.dtype and y.device == like.device:
        y.zero_()
    else:
        setattr(obj, k, torch.zeros_like(like))
