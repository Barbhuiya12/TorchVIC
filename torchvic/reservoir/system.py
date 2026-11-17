"""Reservoir operation on a `routing.StepRouter`, vectorised over reservoirs.

A reservoir sits at a router site (a river cell).  Each step it intercepts
the routed water reaching the site (`StepRouter.intercept` semantics: the
largest volume whose removal leaves no negative ring slot), updates its
storage, and injects release + spill back at the same site, so the water
continues down the channel.  Sites are processed level by level in the
router's topological order, so a reservoir downstream of another sees the
upstream release of the same step.  Nothing loops over reservoirs; the
Python loop is over router levels that contain reservoirs.

Per step, for each reservoir (volumes in m3, flows in m3/s, float64):

    S1  = S + I                              inflow
    E   = min(A(S1) * evap, S1)              evaporation, A = Amax (S1/C)^p
    S2  = S1 - E
    av  = max(S2 - S_dead, 0)                water above the outlet
    Re  = min(Q_env dt, av)                  environmental flow (first claim)
    W   = min(request, av - Re)              water use withdrawal
    Rr  = min(max(Rt - Re, 0), av - Re - W, Qmax dt - Re)   rule release
    S3  = S2 - W - Re - Rr
    Sp  = max(S3 - C, 0)                     spill (goes downstream)
    S   = S3 - Sp
    downstream injection = Re + Rr + Sp

so  S_new - S_old = I - E - W - (Re + Rr + Sp)  holds exactly (the closing
residual is stored in the ledger).  Before the build year the reservoir is
inactive and passes its inflow through unchanged.

Release rules Rt:
  RULE_HANASAKI  Hanasaki, Kanae and Oki (2006), J. Hydrol. 327, 22-41,
      doi:10.1016/j.jhydrol.2005.11.011.  From monthly mean inflow I_m and
      demand D_m (m3/s): annual means I_y, D_y; non-irrigation
      R'_m = I_y; irrigation R'_m = I_y D_m / D_y if D_y >= I_y, else
      I_y + D_m - D_y (an environmental flow Q_env is taken off I_y first and
      added back).  Storage-dependent factor k = S_start / (alpha C) at the
      start of the operational year, c = C / (I_y year):
      R_m = k R'_m for c >= 0.5, otherwise
      R_m = (c/0.5)^2 k R'_m + (1 - (c/0.5)^2) I_m.
      Climatology is either given or learned online (running mean over the
      simulated years); until a reservoir has any, it passes inflow through.
      The operational year starts the month after the minimum of the
      cumulative inflow anomaly (start of the filling season); this is a
      simplification of the paper's rule and is our choice.
  RULE_TARGET    simple rule curve: relax storage toward S_tgt(month) C over
      `relax_steps` steps, Rt = max(S2 - S_tgt C, 0) / relax_steps.
  RULE_PASS      run of river: Rt = inflow.

The spill is added downstream (VICRes drops it; see
docs/connectors/reservoir.md). Dead storage, Q_env, Q_max and evaporation are
extensions we chose; none is taken from code with a licence we cannot use.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

RULE_HANASAKI, RULE_TARGET, RULE_PASS = 0, 1, 2
SEC_YEAR = 365.25 * 86400.0
ALPHA = 0.85                       # Hanasaki et al. (2006): desired maximum fraction


@dataclass
class ReservoirParams:
    """Static data, one entry per reservoir (numpy or tensors), SI units.

    site            index into the router's site list (unique)
    capacity        m3 (total storage)
    dead            m3 dead storage (default 0)
    q_env           m3/s minimum release (default 0)
    q_max           m3/s maximum rule release (default inf; spill is not capped)
    irrigation      bool, Hanasaki irrigation branch
    build_year      active when year >= build_year
    init_frac       initial storage / capacity (default 0.5)
    area_max, area_exp   surface area at full storage (m2) and exponent p;
                    area_max 0 (default) turns evaporation off
    rule            RULE_*
    target_frac     [R, 12] target storage fraction (RULE_TARGET)
    relax_steps     steps to relax to the target (RULE_TARGET)
    inflow_clim, demand_clim  [R, 12] m3/s optional prescribed climatology
    clim_years      weight (years) of the prescribed climatology
    """
    site: np.ndarray
    capacity: np.ndarray
    dead: Optional[np.ndarray] = None
    q_env: Optional[np.ndarray] = None
    q_max: Optional[np.ndarray] = None
    irrigation: Optional[np.ndarray] = None
    build_year: Optional[np.ndarray] = None
    init_frac: Optional[np.ndarray] = None
    area_max: Optional[np.ndarray] = None
    area_exp: Optional[np.ndarray] = None
    rule: Optional[np.ndarray] = None
    target_frac: Optional[np.ndarray] = None
    relax_steps: float = 10.0
    inflow_clim: Optional[np.ndarray] = None
    demand_clim: Optional[np.ndarray] = None
    clim_years: float = 1.0


def hanasaki_release(Ib, Dy, D, E, irr=None, ratio_first=False):
    """Hanasaki et al. (2006) release before the storage correction.

    Ib [N,1] mean inflow net of the environmental flow, floored at 0; Dy
    [N,1] mean demand; D [N,M] demand by month; E [N,1 or M] environmental
    flow added back; irr [N] bool, False gives the non-irrigation rule
    R' = Ib.  Demand regimes: Dy <= 0 gives Ib; Dy > Ib gives Ib D / Dy
    (supply only partly); else Ib - Dy + D.  `ratio_first` evaluates the
    second as Ib * (D / Dy), the order of VIC-WUR dam_operation.c:70-75;
    the default Ib * D / Dy is the order this module always used.  Shared by
    `Reservoirs._plan` and torchvic.vicwur.dams.
    """
    Dc = Dy.clamp(min=1e-300)
    ratio = Ib * (D / Dc) if ratio_first else Ib * D / Dc
    r = torch.where(Dy <= 0, Ib.expand_as(D),
        torch.where(Dy > Ib, ratio, Ib - Dy + D))
    if irr is not None:
        r = torch.where(irr[:, None], r, Ib.expand_as(D))
    return r + E


def hanasaki_capacity(r, I, k, c):
    """Capacity correction of the release, with k [N] = S / (alpha C) and
    c [N] = C / annual inflow: R = k R' for c >= 0.5, otherwise
    (c/0.5)^2 k R' + (1 - (c/0.5)^2) I (Hanasaki 2006; VIC-WUR
    dam_operation.c:93-119).  r, I [N,M]."""
    cc = ((c / 0.5) ** 2)[:, None]
    return torch.where((c >= 0.5)[:, None], k[:, None] * r,
                       cc * k[:, None] * r + (1 - cc) * I)


def hanasaki_target(I, D, Ib, Dy, E, k, c, irr=None, ratio_first=False):
    """Monthly target release [N,M] of the Hanasaki rule: `hanasaki_release`
    followed by `hanasaki_capacity`.  I, D [N,M] monthly inflow and demand;
    see the two functions for the rest.  Pure tensor ops (no host sync)."""
    return hanasaki_capacity(hanasaki_release(Ib, Dy, D, E, irr, ratio_first),
                             I, k, c)


def _core(S, I, c, x, C, dead, q_env, q_max, area_max, area_exp, hana, targ,
          dt: float, relax: float):
    """Pure elementwise operation (see module doc); returns (S_new, downstream
    volume, ledger rows [9, n])."""
    rq, ev = x[:, 0], x[:, 1]
    act = c[:, 3] > 0.5
    S1 = S + I
    A = area_max * (S1 / C).clamp(0, 1) ** area_exp
    E = torch.minimum(A * ev, S1)
    S2 = S1 - E
    av = (S2 - dead).clamp(min=0)
    Re = torch.minimum(q_env * dt, av)
    W = torch.minimum(rq, av - Re)
    rh = torch.where(c[:, 2] > 0.5, c[:, 0] * dt, I)
    rt = (S2 - c[:, 1] * C).clamp(min=0) / relax
    Rt = torch.where(hana, rh, torch.where(targ, rt, I))
    Rr = torch.minimum(torch.minimum((Rt - Re).clamp(min=0), av - Re - W),
                       (q_max * dt - Re).clamp(min=0))
    S3 = S2 - W - Re - Rr
    Sp = (S3 - C).clamp(min=0)
    Sn = S3 - Sp
    zero = torch.zeros_like(I)                  # inactive: pass through
    E, W, Re, Rr, Sp = (torch.where(act, v, zero) for v in (E, W, Re, Rr, Sp))
    Sn = torch.where(act, Sn, S)
    down = torch.where(act, Re + Rr + Sp, I)
    Iin = torch.where(act, I, zero)
    res = Sn - S - Iin + E + W + Re + Rr + Sp
    return Sn, down, torch.stack([Iin, E, rq, W, Re, Rr, Sp, res, I])


_COMPILED = {}


def _core_for(device):
    """Fused (torch.compile) `_core` on CUDA, eager elsewhere or on failure."""
    if torch.device(device).type != 'cuda':
        return _core
    if 'f' not in _COMPILED:
        try:
            import torch._dynamo as dyn
            dyn.config.cache_size_limit = 256
            from ..core import fusion
            fusion.ensure_setup()
            _COMPILED['f'] = torch.compile(_core, dynamic=False)
        except Exception:
            _COMPILED['f'] = _core
    return _COMPILED['f']


_LEDGER = ('inflow', 'evap', 'request', 'withdrawn', 'env', 'release',
           'spill', 'residual', 'raw')       # raw = routed inflow even if inactive


class Reservoirs:
    """State and step of R reservoirs on router sites.

    Everything the step touches is a preallocated device tensor updated in
    place, so `capture` can record a step in a CUDA graph.  Calendar work
    (`new_month`) is host side and runs between steps."""

    _STATE = ('storage', 'active', 'clim_in', 'clim_dem', 'n_years', 'sched',
              'k', 'op_month', 'month', 'led', 'led_last', 'snap', 'nstep',
              'snap_n')

    def __init__(self, params, router, dtype=torch.float64):
        p, dev = params, router.dev
        self.router, self.dev, self.dtype, self.dt = router, dev, dtype, router.dt
        t = lambda a, d=None: torch.as_tensor(np.asarray(a), device=dev,
                                              dtype=d or dtype)
        site = np.asarray(p.site, dtype=np.int64)
        R = self.R = len(site)
        assert len(np.unique(site)) == R, 'one reservoir per site (merge first)'
        full = lambda v, x: (torch.full((R,), v, dtype=dtype, device=dev)
                             if x is None else t(x).expand(R).clone())
        self.site = t(site, torch.long)
        self.cap = t(p.capacity)
        assert bool((self.cap > 0).all())
        self.dead = full(0.0, p.dead)
        assert bool((self.dead < self.cap).all())
        self.q_env = full(0.0, p.q_env)
        self.q_max = full(float('inf'), p.q_max)
        self.irr = (torch.zeros(R, dtype=torch.bool, device=dev)
                    if p.irrigation is None else t(p.irrigation, torch.bool))
        self.build_year = full(-1e9, p.build_year)
        self.area_max = full(0.0, p.area_max)
        self.area_exp = full(1.5, p.area_exp)
        self.rule = (torch.zeros(R, dtype=torch.long, device=dev)
                     if p.rule is None else t(p.rule, torch.long).expand(R).clone())
        self.tgt = (torch.zeros(R, 12, dtype=dtype, device=dev)
                    if p.target_frac is None else t(p.target_frac).expand(R, 12).clone())
        self.relax = float(p.relax_steps)
        self.init_frac = full(0.5, p.init_frac)
        self._prescribed = (p.inflow_clim, p.demand_clim, float(p.clim_years))
        # group reservoirs by the router level of their site; per-group static
        # parameters are gathered once here
        sl = {int(s_): i for i, s_ in enumerate(site)}
        self._is_res = torch.zeros(router.n_site, dtype=torch.bool, device=dev)
        self._is_res[self.site] = True
        self.groups = {}
        for li, lv in enumerate(router.levels):
            sites = lv['sites'].cpu().numpy()
            pos = np.nonzero(np.isin(sites, site))[0]
            if not len(pos):
                continue
            ridx = t([sl[int(sites[q])] for q in pos], torch.long)
            g = self._group(ridx)
            g.update(pos=t(pos, torch.long),
                     ident=bool(len(pos) == len(sites)
                                and (pos == np.arange(len(pos))).all()))
            self.groups[li] = g
        assert sum(len(g['ix']) for g in self.groups.values()) == R, 'site not in router'
        self._core_fn = _core_for(dev)
        self._all = self._group(torch.arange(R, device=dev))
        self.reset()

    def _group(self, ridx):
        g = dict(ix=ridx)
        for name in ('cap', 'dead', 'q_env', 'q_max', 'area_max', 'area_exp'):
            g[name] = getattr(self, name).index_select(0, ridx)
        rule = self.rule.index_select(0, ridx)
        g['hana'], g['targ'] = rule == RULE_HANASAKI, rule == RULE_TARGET
        return g

    # -- state -------------------------------------------------------------
    def reset(self):
        R, dev, dt_ = self.R, self.dev, self.dtype
        z = lambda *s: torch.zeros(*s, dtype=dt_, device=dev)
        self.storage = self.init_frac * self.cap
        self.active = torch.ones(R, dtype=torch.bool, device=dev)
        self.clim_in, self.clim_dem = z(R, 12), z(R, 12)
        self.n_years = z(R, 12)
        pi, pd, py = self._prescribed
        if pi is not None:
            self.clim_in = torch.as_tensor(np.asarray(pi), device=dev, dtype=dt_).expand(R, 12).clone()
            self.n_years += py
        if pd is not None:
            self.clim_dem = torch.as_tensor(np.asarray(pd), device=dev, dtype=dt_).expand(R, 12).clone()
        self.sched = z(R, 12)                    # target release m3/s by calendar month
        self.k = z(R)
        self.op_month = torch.zeros(R, dtype=torch.long, device=dev)
        self.month = torch.zeros((), dtype=torch.long, device=dev)
        self.led, self.led_last = z(len(_LEDGER), R), z(len(_LEDGER), R)
        self.snap = z(2, R)                      # (raw inflow, request) at month start
        self.nstep, self.snap_n = z(()), z(())
        self.ctl = z(R, 4)                       # this month's (target, tgt frac, have clim, active)
        self._views()
        self.last['storage'] = self.storage.clone()
        self._plan(torch.ones(R, dtype=torch.bool, device=dev))
        self._refresh(0)

    def _views(self):
        self.cum = {k: self.led[i] for i, k in enumerate(_LEDGER)}
        self.last = {k: self.led_last[i] for i, k in enumerate(_LEDGER)}

    def _refresh(self, month):
        """In place: the per-month columns the step reads."""
        self.ctl[:, 0] = self.sched[:, month]
        self.ctl[:, 1] = self.tgt[:, month]
        self.ctl[:, 2] = (self.n_years > 0).all(1).to(self.dtype)
        self.ctl[:, 3] = self.active.to(self.dtype)

    def state_dict(self):
        return {k: getattr(self, k).clone() for k in self._STATE}

    def load_state_dict(self, sd):
        for k in self._STATE:
            getattr(self, k).copy_(sd[k].to(self.dev))
        self._refresh(int(self.month))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.dev))

    # -- calendar (host call, once per month, off the hot path) ------------
    def new_month(self, month, year):
        """Close the previous month's accumulators into the climatology, set
        the active flags for `year` and recompute the Hanasaki schedule for
        reservoirs whose operational year starts in `month` (0-11)."""
        m_prev = int(self.month)
        n = (self.nstep - self.snap_n)
        has = (n > 0).expand(self.R)
        n = n.clamp(min=1)
        qi = (self.led[8] - self.snap[0]) / n / self.dt
        qd = (self.led[2] - self.snap[1]) / n / self.dt
        w = self.n_years[:, m_prev]
        new_in = (self.clim_in[:, m_prev] * w + qi) / (w + 1)
        new_dem = (self.clim_dem[:, m_prev] * w + qd) / (w + 1)
        self.clim_in[:, m_prev] = torch.where(has, new_in, self.clim_in[:, m_prev])
        self.clim_dem[:, m_prev] = torch.where(has, new_dem, self.clim_dem[:, m_prev])
        self.n_years[:, m_prev] = w + has.to(self.dtype)
        self.snap[0].copy_(self.led[8]); self.snap[1].copy_(self.led[2])
        self.snap_n.copy_(self.nstep)
        self.month.fill_(int(month))
        self.active.copy_(float(year) >= self.build_year)
        self._plan(self._start_mask(int(month)))
        self._refresh(int(month))

    def _start_mask(self, month):
        have = (self.n_years > 0).all(1)
        # first time a climatology exists, plan immediately
        fresh = have & (self.sched.abs().sum(1) == 0)
        return (have & (self.op_month == month)) | fresh

    def _plan(self, mask):
        """Hanasaki target schedule for reservoirs in `mask` (see module doc)."""
        I, D = self.clim_in, self.clim_dem
        Iy, Dy = I.mean(1), D.mean(1)
        anom = torch.cumsum(I - Iy[:, None], 1)
        self.op_month.copy_(torch.where(mask, (anom.argmin(1) + 1) % 12, self.op_month))
        E = self.q_env
        Ib = (Iy - E).clamp(min=0)
        k = (self.storage / (ALPHA * self.cap)).clamp(max=1.0 / ALPHA)
        c = self.cap / (Iy * SEC_YEAR).clamp(min=1e-300)
        sched = hanasaki_target(I, D, Ib[:, None], Dy[:, None], E[:, None], k, c,
                                irr=self.irr)
        self.k.copy_(torch.where(mask, k, self.k))
        self.sched.copy_(torch.where(mask[:, None], sched, self.sched))

    # -- the step ----------------------------------------------------------
    def _operate(self, g, I, ext):
        """Vectorised update of the reservoirs of one group; returns the
        downstream volume.  ext [R, 2] = (request >= 0, evaporation depth).
        The elementwise arithmetic is one fused kernel on CUDA (`_core`)."""
        ix = g['ix']
        S = self.storage.index_select(0, ix)
        c = self.ctl.index_select(0, ix)
        x = ext.index_select(0, ix)
        Sn, down, st = self._core_fn(S, I, c, x, g['cap'], g['dead'], g['q_env'],
                                     g['q_max'], g['area_max'], g['area_exp'],
                                     g['hana'], g['targ'], self.dt, self.relax)
        self.storage.index_copy_(0, ix, Sn)
        self.led_last.index_copy_(1, ix, st)
        self.led.index_add_(1, ix, st)
        return down

    def operate_all(self, inflow, request=None, evap=None):
        """Operate every reservoir on given inflow volumes [R] m3, without a
        router (unit tests, offline use).  Returns the downstream volume."""
        z = torch.zeros(self.R, dtype=self.dtype, device=self.dev)
        ext = torch.stack([(z if request is None else request).clamp(min=0),
                           z if evap is None else evap], 1)
        self.nstep += 1
        return self._operate(self._all, inflow, ext)

    def apply(self, request=None, evap=None, withdraw=None, inject=None):
        """Hooks pass for one step, after `router.begin`: intercept at every
        reservoir site, operate, inject release + spill, level by level
        (`StepRouter.sweep`).

        request [R] m3 water-use withdrawal from each reservoir; evap [R] m
        of potential evaporation this step.  withdraw/inject [n_site] m3 are
        ordinary router hooks for the other sites (reservoir entries ignored).
        Returns the delivered withdrawal [n_site] of those other sites."""
        r = self.router
        z = torch.zeros(self.R, dtype=self.dtype, device=self.dev)
        ext = torch.stack([(z if request is None else request).clamp(min=0),
                           z if evap is None else evap], 1)
        self.nstep += 1
        zs = torch.zeros(r.n_site, dtype=self.dtype, device=self.dev)
        got = zs.clone()
        hooks = withdraw is not None or inject is not None
        groups = self.groups

        def visit(li, sid, cap, avail):
            g = groups.get(li)
            if g is None and not hooks:
                return None
            take, put = zs[:len(sid)], zs[:len(sid)]
            ours = self._is_res[sid]
            if withdraw is not None:
                take = torch.where(ours, 0.0, withdraw[sid].clamp(min=0))
                take = torch.minimum(take, cap)
                got.index_copy_(0, sid, take)
            if inject is not None:
                put = torch.where(ours, 0.0, inject[sid])
            if g is not None:
                pos = g['pos']
                if g['ident']:
                    I = torch.minimum(cap, avail)
                    down = self._operate(g, I, ext)
                    take = I if withdraw is None else torch.where(ours, I, take)
                    put = down if inject is None else torch.where(ours, down, put)
                else:
                    I = torch.minimum(cap[pos], avail[pos])
                    down = self._operate(g, I, ext)
                    take = take.index_copy(0, pos, I)
                    put = put.index_copy(0, pos, down)
            return take, put

        r.sweep(visit)
        return got

    def step(self, runoff, baseflow=None, request=None, evap=None,
             withdraw=None, inject=None, node_inject=None):
        """One full router step with reservoirs; returns (discharge, delivered)."""
        r = self.router
        r.begin(runoff if baseflow is None else runoff + baseflow)
        got = self.apply(request, evap, withdraw, inject)
        if node_inject is not None:
            r.node_inject(node_inject)
        return r.end(), got

    def capture(self, runoff, request=None, evap=None, warmup=3):
        """Capture one full step (router begin, reservoir pass, end) in a CUDA
        graph over the given static buffers and return `replay()`; refill the
        buffers in place before each call.  The calendar call `new_month` is
        host side and updates state in place, so it can run between replays.
        The warm-up steps are real steps (state advances)."""
        kw = dict(request=request, evap=evap)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.step(runoff, **kw)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._q = self.step(runoff, **kw)[0]
        def replay():
            g.replay()
            return self._q
        return replay

    # -- ledger ------------------------------------------------------------
    def volume_balance(self):
        """Cumulative residual [R] m3 of storage change = inflow - losses
        (zero to rounding); built from the independent storage history."""
        c = self.cum
        init = self.init_frac * self.cap
        return (self.storage - init - c['inflow'] + c['evap'] + c['withdrawn']
                + c['env'] + c['release'] + c['spill'])
