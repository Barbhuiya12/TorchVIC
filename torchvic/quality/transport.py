"""GPU river water quality and temperature carried by the stepping router.

Every constituent is routed with the same linear unit-hydrograph machinery
as the water (`routing.StepRouter`), so mixing at confluences is automatic
(mass adds, concentration = load / flow) and a step is a fixed sequence of
tensor ops with no host sync (CUDA-graph capturable).

First-order decay dC/dt = -k C along the flow path is exact in this form:
mass that takes a travel time t arrives multiplied by exp(-k t), so the unit
hydrograph of the constituent is UH[l] * exp(-k l dt) (and the same for the
channel kernels used by withdrawals/injections).  k may differ per cell
(the rate of the emitting cell applies along its whole path).

Stream temperature is advected heat Q*T (T >= 0 C, the freezing floor), with
the surface exchange of `heat.py`: T relaxes towards the equilibrium
temperature Te at rate kT, T(t) = Te + (T0 - Te) exp(-kT t), with T0 the
runoff temperature and Te the equilibrium temperature of the emitting cell
at the time of emission.  It is carried by two routers, one with kernel
UH*exp(-kT l dt) (heat of the runoff itself) and one with UH*(1-exp(-kT l dt))
(heat taken from the atmosphere), both linear.

Withdrawals take load/heat in proportion to the water taken at each site
(delivered volume / volume passing).  Injected water carries a
concentration/temperature; point sources inject mass.
"""
import copy

import torch

from torchvic.model import routing
from torchvic.model.routing import StepRouter
from torchvic.model import routing_fast as _rf
from torchvic.model import routing_kernels as K
from torchvic.quality import batch as _batch
from torchvic.quality import heat as _heat


def _blocked_transform(tree, kpair, lagt, fn, extra, chunk=2048):
    """New _TreeSum whose U[..., :L] is multiplied by fn(k, lagt) and whose
    last column(s) are rebuilt: [... , sum (cs), (1 - cs) if extra]."""
    t = copy.copy(tree)
    t.__dict__.pop('_fast_levels', None)
    U = tree.U
    L = U.shape[2] - 1
    kp = torch.cat([kpair, torch.zeros(1, dtype=kpair.dtype, device=kpair.device)])
    cols = 2 if extra else 1
    out = torch.empty(U.shape[0], U.shape[1], L + cols, dtype=U.dtype,
                      device=U.device)
    for a in range(0, U.shape[0], chunk):
        b = min(a + chunk, U.shape[0])
        k = kp[tree.idx[a:b]].unsqueeze(-1)                 # [c, block, 1]
        u = U[a:b, :, :L] * fn(k, lagt)
        out[a:b, :, :L] = u
        cs = u.sum(-1)
        out[a:b, :, L] = cs
        if extra:
            out[a:b, :, L + 1] = 1.0 - cs
    t.U = out
    return t


def _decay(k, lagt):
    return torch.exp(-k * lagt)


def _complement(k, lagt):
    return -torch.expm1(-k * lagt)


class _ChannelBase:
    """A StepRouter derived from a water router with transformed kernels.

    Same nodes, sites and levels; its own ring and ledger.  `fn` is the
    kernel factor (decay or its complement), `k_cell` the rate [1/s] of every
    cell.  Input to `begin` is a per-cell flux density in the units of
    `w` (see `derive`)."""

    @classmethod
    def derive(cls, base, k_cell, fn=_decay, weight=None, site_cells=None):
        o = object.__new__(cls)
        o.__dict__.update(base.__dict__)
        dev, dt, L = base.dev, base.dt, base.L
        lagt = (torch.arange(L, dtype=base.dtype, device=dev) * dt)
        k_cell = torch.as_tensor(k_cell, dtype=base.dtype, device=dev)
        kpair = (k_cell.expand(base.n_pair) if k_cell.dim() == 0
                 else k_cell[base.cell])
        o.tree = _blocked_transform(base.tree, kpair, lagt, fn, True)
        if weight is not None:
            o.w = weight
        o.levels = []
        for lv in base.levels:
            n = dict(lv)
            if k_cell.dim() == 0:
                kp = k_cell.expand(lv['pair_site'].shape[0])
            else:
                assert site_cells is not None, 'site_cells needed for per-cell rates'
                sc = torch.as_tensor(site_cells, device=dev)
                kp = k_cell[sc[lv['sites'][lv['pair_site']]]]
            tr = _blocked_transform(lv['tree'], kp, lagt, fn, False)
            kk = lv['kern'] * fn(kp.unsqueeze(-1), lagt)   # for the cap
            n.update(tree=tr, kern=kk, pos=kk > 0,
                     kern_safe=kk.clamp(min=1e-300))
            o.levels.append(n)
        if isinstance(base, _rf.FastStepRouter):
            o.lk = base.lk
            o._lk = {id(n): k for n, k in zip(o.levels, base.lk)}
        o.reset()
        return o

    def reset(self):
        super().reset()
        self.lost = torch.zeros(self.n_node, dtype=self.dtype, device=self.dev)

    _STATE = StepRouter._STATE + ('lost',)

    def _begin_ref(self, flux):
        """Add this step's per-cell input [cell]: flux * w enters the ring."""
        x = flux.index_select(0, self.cell) * self.w
        t = self.tree.apply(x)                          # [node, L + 2]
        acc = torch.zeros(self.n_node, self.L, dtype=self.dtype,
                          device=self.dev)
        acc.index_copy_(0, self.tree.ids, t[:, :self.L])
        self._add(acc)
        self.in_runoff.index_add_(0, self.tree.ids, t[:, self.L] * self.dt)
        self.lost.index_add_(0, self.tree.ids, t[:, self.L + 1] * self.dt)

    def put(self, li, sid, mass):
        """Add `mass` [n] at level li's sites through the channel kernel."""
        self._apply_level(self.levels[li], mass)
        self.injected.index_add_(0, sid, mass)

    def node_flux(self):
        return self.head_discharge()


class Channel(_ChannelBase, StepRouter):
    begin = _ChannelBase._begin_ref


class FastChannel(_ChannelBase, _rf.FastStepRouter):
    def begin(self, flux):
        K.begin_x(flux.contiguous(), self.cell, self.w, self.tree.idx_flat,
                  self._xb, self.n_pair)
        t = _rf.blocked(self.tree, self._xb)
        K.update(self.ring, self.head, self.L, self.tree.ids, t,
                 self.in_runoff, self._dtt, lost=self.lost)


def make_channel(base, k_cell, fn, weight=None, site_cells=None):
    cls = FastChannel if isinstance(base, _rf.FastStepRouter) else Channel
    return cls.derive(base, k_cell, fn, weight, site_cells)


class WaterQuality:
    """Constituents and temperature on a `StepRouter`.

    router        the water router (`river_network.build_step_router`)
    area          [cell] m2 (the router's weights are derived from it)
    constituents  {name: k} decay rate(s) [1/s], scalar or [cell]
    temperature   None or dict(depth=[cell] or scalar m, k_ref=W m-2 K-1)
    site_cells    cells of the router's sites (needed for per-cell rates)
    One call to `step` runs the whole per-step flow of the water router
    (begin, level-wise hooks, end) and the constituents/heat alongside it.
    """

    def __init__(self, router, area, constituents=None, temperature=None,
                 site_cells=None, qmin=1e-9):
        self.r = router
        dev, dt = router.dev, router.dt
        self.dev, self.dt, self.dtype = dev, dt, router.dtype
        area = torch.as_tensor(area, device=dev, dtype=self.dtype)
        # fraction of the cell's mass sent to each pair (the area weight)
        frac = router.w * (routing.MM_PER_M * dt) / area[router.cell]
        self.w_load = frac / dt                        # mass per step -> mass/s
        self.qmin = qmin
        self.names = list(constituents or {})
        self.temp = None
        self.n_site = router.n_site
        scalar = lambda k: torch.as_tensor(k).numel() == 1
        depth = None
        if temperature is not None:
            depth = torch.as_tensor(temperature['depth'], device=dev, dtype=self.dtype)
            self.k_ref = float(temperature.get('k_ref', 30.0))
            self.albedo = float(temperature.get('albedo', 0.06))
            self.shade = temperature.get('shade', 0.0)
        self.uniform = (isinstance(router, _rf.FastStepRouter)
                        and (bool(constituents) or temperature is not None)
                        and all(scalar(k) for k in (constituents or {}).values())
                        and (depth is None or depth.numel() == 1))
        if self.uniform:
            lagt = torch.arange(router.L, dtype=self.dtype, device=dev) * dt
            ws, ds = [], []
            for k in (constituents or {}).values():
                ws.append(self.w_load)
                ds.append(torch.exp(-float(k) * lagt))
            if depth is not None:
                kT = float(_heat.relaxation_rate(self.k_ref, depth.reshape(())))
                for fn in (lambda: torch.exp(-kT * lagt), lambda: -torch.expm1(-kT * lagt)):
                    ws.append(router.w)
                    ds.append(fn())
            self.bq = _batch.UniformChannels(router, torch.stack(ws), torch.stack(ds))
            h = self.bq.holders
            self.ch = {n: h[i] for i, n in enumerate(self.names)}
            if depth is not None:
                self.temp = (h[len(self.names)], h[len(self.names) + 1])
            self.ring_all, self.hooks_all = self.bq.ring_all, self.bq.hooks_all
            self.wd_all, self.inj_all = self.bq.wd_all, self.bq.inj_all
        else:
            self.ch = {n: make_channel(router, k, _decay, self.w_load, site_cells)
                       for n, k in (constituents or {}).items()}
            if depth is not None:
                kT = _heat.relaxation_rate(self.k_ref, depth)
                self.temp = (make_channel(router, kT, _decay, None, site_cells),
                             make_channel(router, kT, _complement, None, site_cells))
            # all channels share one stacked state so a hook level is one batched op
            self.all = list(self.ch.values()) + list(self.temp or ())
            C, L, N = len(self.all), router.L, router.n_node
            ns = max(self.n_site, 1)
            z = lambda *sh: torch.zeros(*sh, dtype=self.dtype, device=dev)
            self.ring_all, self.hooks_all = z(C, L, N), z(C, N)
            self.wd_all, self.inj_all = z(C, ns), z(C, ns)
            for i, c in enumerate(self.all):
                c.ring, c.in_hooks = self.ring_all[i], self.hooks_all[i]
                c.withdrawn, c.injected = self.wd_all[i], self.inj_all[i]
        self._z = torch.zeros(int(router.cell.max()) + 1, dtype=self.dtype,
                              device=self.dev)

    # -- one step ----------------------------------------------------------
    def step(self, runoff, loads=None, point=None, met=None, t_runoff=None,
             withdraw=None, inject=None, inject_conc=None, inject_temp=None):
        """Advance one step.

        runoff [cell] mm per step (the water router's input); loads
        {name: [cell]} mass per step; point {name: [site]} mass per step;
        met = dict(t_air C, vp kPa, wind m/s, sw W/m2) per cell; t_runoff [cell]
        C of the runoff (default: air temperature floored at 0);
        withdraw / inject [site] m3; inject_conc {name: [site]} mass/m3;
        inject_temp [site] C.
        Returns dict: q [node] m3/s, conc {name: [node]}, t [node] C (0 where
        q < qmin), plus 'got' (delivered withdrawals [site])."""
        r = self.r
        r.begin(runoff)
        te = t0 = None
        if self.temp is not None:
            te, _ = _heat.equilibrium_temperature(
                met['t_air'], met['vp'], met['wind'], met['sw'],
                self.albedo, self.shade)
            t0 = (met['t_air'].clamp(min=0.0) if t_runoff is None
                  else t_runoff.clamp(min=0.0))
        if self.uniform:
            rows = [loads[n] if loads and n in loads else self._zeros_cell()
                    for n in self.names]
            if self.temp is not None:
                rows += [runoff * t0, runoff * te]
            if rows:
                self.bq.begin(torch.stack(rows))
        else:
            for n, c in self.ch.items():
                c.begin(loads[n] if loads and n in loads else self._zeros_cell())
            if self.temp is not None:
                self.temp[0].begin(runoff * t0)
                self.temp[1].begin(runoff * te)
        got = torch.zeros(max(self.n_site, 1), dtype=self.dtype, device=self.dev)
        if self.n_site and (withdraw is not None or inject is not None
                            or point or inject_conc or inject_temp is not None):
            self._sweep(got, withdraw, inject, point, inject_conc, inject_temp)
        qa = self.bq.end() if self.uniform and self.bq.C else None
        q = r.end()
        out = dict(q=q, got=got)
        valid = q > self.qmin
        qs = torch.where(valid, q, torch.ones_like(q))
        zq = torch.zeros_like(q)
        if self.uniform:
            out['conc'] = {n: torch.where(valid, qa[i] / qs, zq)
                           for i, n in enumerate(self.names)}
            if self.temp is not None:
                h = qa[len(self.names)] + qa[len(self.names) + 1]
                out['t'] = torch.where(valid, h / qs, zq)
            return out
        out['conc'] = {n: torch.where(valid, c.end() / qs, zq)
                       for n, c in self.ch.items()}
        if self.temp is not None:
            h = self.temp[0].end() + self.temp[1].end()
            out['t'] = torch.where(valid, h / qs, zq)
        return out

    def _zeros_cell(self):
        return self._z

    def zero_state(self):
        """Zero every state tensor in place (graph replay restarts)."""
        for k in self.r._STATE:
            getattr(self.r, k).zero_()
        if self.uniform:
            self.bq.zero()
            return
        for rt in list(self.ch.values()) + list(self.temp or ()):
            for k in rt._STATE:
                getattr(rt, k).zero_()

    def _sweep(self, got, withdraw, inject, point, inject_conc, inject_temp):
        """Level-wise hooks (upstream to downstream) for water, every
        constituent and heat together.

        Withdrawal: the water router removes `take` = min(request, cap)
        through its channel kernel; every constituent/heat ring slot loses the
        same fraction of its own content as the water in that slot
        (d = acc_water / ring_water), so no concentration or temperature can
        change by a withdrawal and none can overshoot, even in a drained reach.
        Injections and point sources enter through each channel's kernel."""
        r, dt, L = self.r, self.dt, self.r.L
        if isinstance(r, _rf.FastStepRouter):
            return self._sweep_fast(got, withdraw, inject, point, inject_conc,
                                    inject_temp)
        slots = r._slots()
        for li, lv in enumerate(r.levels):
            sid, ids = lv['sites'], lv['tree'].ids
            put = None if inject is None else inject[sid].clamp(min=0)
            if withdraw is not None:
                node = r.site_node[sid]
                avail = r.head_discharge()[node].clamp(min=0) * dt
                take = torch.minimum(withdraw[sid].clamp(min=0), r._cap(lv))
                frac = torch.where(avail > 0, take / avail.clamp(min=1e-300),
                                   torch.zeros_like(take))
                passing = self.ring_all.index_select(1, r.head.view(1)) \
                    .squeeze(1)[:, node].clamp(min=0)            # [C, n]
                self.wd_all.index_add_(1, sid, frac * passing * dt)
                t = lv['tree'].apply(-take[lv['pair_site']] / dt)   # [ids, L + 1]
                si, ni = slots[:, None], ids[None, :]
                qw = r.ring[si, ni]                                  # [L, ids]
                a_ = t[:, :L].t()                                    # [L, ids] <= 0
                fr = torch.where(qw > 0, a_ / qw.clamp(min=1e-300),
                                 torch.zeros_like(qw)).clamp(-1.0, 0.0)
                cur = self.ring_all[:, si, ni]                       # [C, L, ids]
                d = cur * fr
                self.ring_all[:, si, ni] = cur + d
                self.hooks_all.index_add_(1, ids, d.sum(1) * dt)
                acc = torch.zeros(r.n_node, L, dtype=self.dtype, device=self.dev)
                acc.index_copy_(0, ids, t[:, :L])
                r._add(acc)
                r.in_hooks.index_add_(0, ids, t[:, L] * dt)
                r.withdrawn.index_add_(0, sid, take)
                got.index_add_(0, sid, take)
            if put is not None:
                r._apply_level(lv, put)
                r.injected.index_add_(0, sid, put)
            for n, c in self.ch.items():       # only channels that get mass
                m = None
                if point and n in point:
                    m = point[n][sid].clamp(min=0)
                if inject_conc and n in inject_conc and put is not None:
                    q = put * inject_conc[n][sid].clamp(min=0)
                    m = q if m is None else m + q
                if m is not None:
                    c.put(li, sid, m)
            if (self.temp is not None and inject_temp is not None
                    and put is not None):
                self.temp[0].put(li, sid, put * inject_temp[sid].clamp(min=0))

    def _sweep_fast(self, got, withdraw, inject, point, inject_conc, inject_temp):
        """Same as `_sweep` with the fused kernels: per level one kernel for
        cap/take/counters, the water tree sum, one kernel that removes the
        water's fraction from every channel slot, one ring update."""
        r = self.r
        wd = None if withdraw is None else withdraw.contiguous()
        chan = (r.site_node, self.ring_all, self.wd_all)
        for li, (lv, k) in enumerate(zip(r.levels, r.lk)):
            sid = lv['sites']
            if wd is not None:
                K.level(r.ring, r.head, r.L, k, wd, None, got, r.withdrawn,
                        r.injected, r._dtt, chan)
                t = _rf.blocked(lv['tree'], k['xpad'])
                if self.ring_all.shape[0]:
                    K.prop(r.ring, self.ring_all, r.head, r.L, lv['tree'].ids, t,
                           self.hooks_all, r._dtt)
                K.update(r.ring, r.head, r.L, lv['tree'].ids, t, r.in_hooks,
                         r._dtt)
            put = None if inject is None else inject[sid].clamp(min=0)
            if put is not None:
                r._apply_level(lv, put)
                r.injected.index_add_(0, sid, put)
            for i, n in enumerate(self.names):    # only channels that get mass
                m = None
                if point and n in point:
                    m = point[n][sid].clamp(min=0)
                if inject_conc and n in inject_conc and put is not None:
                    q = put * inject_conc[n][sid].clamp(min=0)
                    m = q if m is None else m + q
                if m is not None:
                    self._put(i, li, sid, m)
            if (self.temp is not None and inject_temp is not None
                    and put is not None):
                self._put(len(self.names), li, sid, put * inject_temp[sid].clamp(min=0))

    def _put(self, i, li, sid, m):
        if self.uniform:
            self.bq.put(i, li, sid, m)
        else:
            self.all[i].put(li, sid, m)

    # -- ledgers -----------------------------------------------------------
    def ledger(self):
        """Per-node residual (mass or m3 K) of each constituent and of heat:
        in_runoff + in_hooks - out - stored; zero to round-off.  The decayed
        mass (a sink) is reported separately per node as `decayed`."""
        res = {'water': self.r.balance()}
        for n, c in self.ch.items():
            res[n] = c.balance()
        if self.temp is not None:
            res['heat'] = self.temp[0].balance() + self.temp[1].balance()
        return res

    def decayed(self, name):
        """[node] mass removed by decay on the way into each node."""
        return self.ch[name].lost

    def atmosphere_heat(self):
        """[node] heat (m3 K) gained from the atmosphere by the flow arriving
        at each node: heat taken up on the way minus the runoff heat lost."""
        return self.temp[1].in_runoff - self.temp[0].lost
