# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR remote water use (C6): edge list, sector priority, availability
sharing.  See docs/connectors/vicwur.md.

A SOURCE cell serves RECEIVER cells (wu_con[source].receiving, built by
wu_set_receiving, wu_init.c:35-125).  After a source's own local use, its
routing ring is the pool: the receivers' unmet demand is shared out in
sector priority, withdrawn from the SOURCE ring, consumed at the receiver
and returned to the source ring (wu_remote.c:579-617 call order is
wu_remote.c:741-808).

C references are VIC-WUR 3.0 plugins/wateruse/src/wu_remote.c
(<data root>/build/vicwur3/src_tree/vic):
  :99-133   reset_wu_remote       temporary receiver fields to 0
  :139-199  calculate_demand_remote   receiver unmet demand
  :205-236  calculate_availability_remote   whole source ring, EFR
  :242-297  calculate_division_remote   sector priority, pro rata over edges
  :303-370  calculate_use_remote    withdrawal, consumption, return
  :376-528  calculate_hydrology_remote   deficit payback, ring edit
  :580-617  account_remote          receiver tremote totals
  :623-722  check_water_use_balance_remote

Data layout.  Cells on the leading axis, sectors (S = 5: MUN, ENE, MAN, LIV,
IRR = priority order) on the last.  The edge list is a COO pair (src, dst)
in C order (a source's receivers in wu_con.receiving order).  The static
plan (RemotePlan) groups the sources into BATCHES such that, inside one
batch, no cell is touched by two sources (a source, or one of its
receivers); the serial C order is then reproduced exactly by running the
batches in order, and all sources of a batch at once.  Every per-source
reduction runs over a padded [source, slot] matrix with a fixed reduction
order, every write to a receiver is to a unique row (index_put without
accumulate), so results are deterministic on the GPU (no atomics), and no
function reads a device value on the host.

The C `min` and `max` macros (vic_def.h:91-92) are `(a < b) ? a : b` and
`(a > b) ? a : b`: a NaN in the first argument gives the second.  `cmin` and
`cmax` reproduce them.

Quirks reproduced (mode 'exact' is the only mode of this module):
  * the receiver's demand is in mm over the RECEIVER area, the source ring
    availability in mm over the SOURCE area, and the ring edit converts with
    the SOURCE area (:382-388, :460-466);
  * the division has no lower clamp on the remaining pool (:274-290);
  * the deficit payback of a receiver (:406-449) moves `returned` from the
    source to the receiver, and the running returned total of the source is
    clipped at 0 after every step (kept as max(R - sum x, 0), equal for x >= 0);
  * the empty-ring branches (:482-483, :495-499) are reproduced, they are
    unreachable (W > 0 implies a positive ring sum).
"""
import numpy as np
import torch

from .wateruse import cmin, unmet, availability_surf, NSECT, MM_PER_M

WU_BALANCE_ERROR_THRESH = 1e-10        # wateruse.h:33


def cmax(a, b):
    """C `max` macro: (a > b) ? a : b (NaN in a gives b)."""
    return torch.where(a > b, a, b)


# --------------------------------------------------------------- edge list
def edges_from_receiving(receiving):
    """COO edge list (src, dst) from params.read_remote's padded table
    receiving[N, M] (-1 padded): a source's receivers in listing order."""
    r = np.asarray(receiving.cpu() if torch.is_tensor(receiving)
                   else receiving, np.int64)
    if r.ndim == 1:
        r = r[:, None]
    src, slot = np.nonzero(r >= 0)
    return src.astype(np.int64), r[src, slot]


def _as_np(x):
    return np.asarray(x.cpu() if torch.is_tensor(x) else x, np.int64)


class Batch(object):
    """Sources of one batch (static, device tensors).

    src [B]        source cells
    eb [E]         edge -> index of its source in src
    esrc, edst [E] edge source and receiver cell
    epos [E]       flat position in the padded [B, K] matrix (slot order)
    """

    def __init__(self, src, eb, esrc, edst, slot, K, device):
        t = lambda a: torch.as_tensor(np.asarray(a, np.int64), device=device)
        self.n_src, self.K, self.n_edges = len(src), int(K), len(edst)
        self.src, self.eb = t(src), t(eb)
        self.esrc, self.edst = t(esrc), t(edst)
        self.epos = t(np.asarray(eb) * int(K) + np.asarray(slot))


class RemotePlan(object):
    """Static topology: the edge list and its batches.

    src, dst   COO edge list (cells 0..n-1); the order of the edges of one
               source is the order in which C serves its receivers
    n          number of cells
    batch_of   optional [n] batch index of every source (e.g. the level and
               sub-level of CascadeRouter's Schedule); sources that share a
               batch must not touch a common cell.  Default: the minimal
               batching of the serial order `order` (default ascending cell
               index), where a source conflicts with an earlier one when
               {source} + receivers intersect.
    batches    list of Batch, in execution order; group[i] the batch_of key
    """

    def __init__(self, src, dst, n, batch_of=None, order=None, device='cpu'):
        src, dst = _as_np(src), _as_np(dst)
        if src.shape != dst.shape:
            raise ValueError('src and dst differ in length')
        self.n, self.device = int(n), torch.device(device)
        if len(src) and (src.min() < 0 or dst.min() < 0 or
                         src.max() >= n or dst.max() >= n):
            raise ValueError('edge cell out of range')
        if (src == dst).any():
            raise ValueError('a cell cannot receive from itself '
                             '(wu_set_routing_order stalls, wu_init.c:271)')
        key = src * self.n + dst
        if len(np.unique(key)) != len(key):
            raise ValueError('duplicate (source, receiver) edge: C would '
                             'serve it twice, in sequence')
        o = np.argsort(src, kind='stable')          # keep listing order
        self.src, self.dst = src[o], dst[o]
        srcs = np.unique(self.src)
        lo = np.searchsorted(self.src, srcs, 'left')
        hi = np.searchsorted(self.src, srcs, 'right')
        rcv = {int(s): self.dst[l:h] for s, l, h in zip(srcs, lo, hi)}
        touched = {s: set([s]) | set(map(int, r)) for s, r in rcv.items()}
        if batch_of is None:
            seq = np.arange(n) if order is None else _as_np(order)
            rank = np.empty(n, np.int64)
            rank[seq] = np.arange(n)
            nxt, batch_of = {}, {}
            for s in sorted(rcv, key=lambda c: rank[c]):
                g = max([nxt.get(c, 0) for c in touched[s]])
                batch_of[s] = g
                for c in touched[s]:
                    nxt[c] = g + 1
        else:
            bo = np.asarray(batch_of)
            batch_of = {s: int(bo[s]) for s in rcv}
        self.batch_of = batch_of
        self.group, self.batches = [], []
        for g in sorted(set(batch_of.values())):
            ss = [s for s in rcv if batch_of[s] == g]
            seen = set()
            for s in ss:
                if seen & touched[s]:
                    raise ValueError(
                        'batch %d: sources %s share a cell; they must run '
                        'in different batches' % (g, ss))
                seen |= touched[s]
            eb = np.concatenate([[i] * len(rcv[s]) for i, s in enumerate(ss)])
            es = np.concatenate([[s] * len(rcv[s]) for s in ss])
            ed = np.concatenate([rcv[s] for s in ss])
            sl = np.concatenate([np.arange(len(rcv[s])) for s in ss])
            self.group.append(g)
            self.batches.append(Batch(ss, eb, es, ed, sl, sl.max() + 1,
                                      self.device))

    @property
    def n_edges(self):
        return len(self.src)


def plan_for_router(router, src, dst):
    """RemotePlan in the sweep positions of a CascadeRouter, one batch per
    launch group of its schedule (so batch i is the group the router's
    wu_remote slot is called with; `group` holds the group index)."""
    pos = np.asarray(router.pos.cpu(), np.int64)
    s, d = pos[_as_np(src)], pos[_as_np(dst)]
    gptr = np.asarray(router._gptr, np.int64)
    batch_of = np.zeros(router.n, np.int64)
    batch_of[:] = np.searchsorted(gptr, np.arange(router.n), side='right') - 1
    return RemotePlan(s, d, router.n, batch_of=batch_of,
                      device=router.device)


# ------------------------------------------------------------------- state
class RemoteState(object):
    """wu_var fields that remote use reads and writes, [C, S] each.

    Totals (reset every step by wu_run.c:50-57): available_tremote,
    demand_tremote, withdrawn_tremote (receiver), available_remote,
    demand_remote, withdrawn_remote (source), returned, consumed.
    Temporaries (not reset; they hold the last source's values at each
    receiver, :127-130): available_tmp, demand_tmp, withdrawn_tmp,
    returned_tmp, consumed_tmp.
    """
    TOTALS = ('available_tremote', 'demand_tremote', 'withdrawn_tremote',
              'available_remote', 'demand_remote', 'withdrawn_remote',
              'returned', 'consumed')
    TMPS = ('available_tmp', 'demand_tmp', 'withdrawn_tmp', 'returned_tmp',
            'consumed_tmp')

    def __init__(self, n, device='cpu', dtype=torch.float64):
        for k in self.TOTALS + self.TMPS:
            setattr(self, k, torch.zeros(n, NSECT, dtype=dtype,
                                         device=device))

    def reset(self):
        """Start of a step (reset_wu, wu_run.c:50-57): the totals, not the
        temporaries.  `returned` and `consumed` are set by local use."""
        for k in self.TOTALS[:6]:
            getattr(self, k).zero_()


class LocalUse(object):
    """What the local pass leaves for remote use, [C, S] (wu_var after
    wu_run_local and the dams): demand_gw, demand_surf, withdrawn_gw,
    withdrawn_surf, withdrawn_dam, cons_frac (wu_force.consumption_frac) and
    active (sector exists, sidx != NODATA_WU)."""

    def __init__(self, dgw, dsf, wgw, wsf, wdm, cons_frac, active):
        self.dgw, self.dsf, self.wgw, self.wsf, self.wdm = (dgw, dsf, wgw, wsf,
                                                            wdm)
        self.cons_frac, self.active = cons_frac, active


# ----------------------------------------------------------------- kernels
def demand_tmp(dgw, dsf, wgw, wsf, wdm, wtremote, gwm=False):
    """Receiver unmet demand, the `demand_remote_tmp` of
    calculate_demand_remote (:172-192).  Any shape.  GWM (VIC-WUR 3.0):
    surface demand only, `(x > 0) ? x : 0`; else (demand_surf + demand_gw) -
    (withdrawn_surf + withdrawn_gw + withdrawn_dam + withdrawn_tremote),
    negative set to 0."""
    if gwm:
        x = dsf - (wsf + wdm + wtremote)
        return torch.where(x > 0, x, torch.zeros_like(x))
    return unmet(dgw, dsf, wgw, wsf, wdm, wtremote)


def _pad(x_e, b):
    """[E, S] -> [B, K, S], zeros in the unused slots; fixed positions."""
    z = x_e.new_zeros(b.n_src * b.K, x_e.shape[-1])
    z[b.epos] = x_e
    return z.view(b.n_src, b.K, -1)


def _unpad(x, b):
    """[B, K, S] -> [E, S]."""
    return x.reshape(b.n_src * b.K, -1)[b.epos]


def division(dem, a_rem, act_src):
    """Sector priority and pro-rata sharing (calculate_division_remote,
    :261-296).  dem [B, K, S] receiver unmet demand per edge slot (0 where
    unused or inactive), a_rem [B] source ring availability (mm), act_src
    [B, S].  Per sector in order: frac = min(pool / demand_total, 1) (macro),
    each edge gets demand * frac, the pool loses what was given.  There is no
    clamp at 0 (rounding can leave the pool at -1e-17 and the next frac
    tiny negative, as in C).  Returns (avail [B, K, S], avail_src [B, S],
    dtot [B, S])."""
    pool = a_rem
    one = torch.ones_like(a_rem)
    zero = torch.zeros_like(a_rem)
    dtot = dem.sum(1)                                  # [B, S], slot order
    outs, srcs = [], []
    for s in range(dem.shape[-1]):
        d = dtot[:, s]
        m = act_src[:, s] & (d > 0)
        den = torch.where(m, d, one)
        frac = cmin(pool / den, one)
        ar = dem[:, :, s] * frac[:, None]
        ar = torch.where(m[:, None], ar, torch.zeros_like(ar))
        tot = ar.sum(1)
        pool = torch.where(m, pool - tot, pool)
        outs.append(ar)
        srcs.append(torch.where(m, tot, zero))
    return torch.stack(outs, -1), torch.stack(srcs, -1), dtot


def use(avail, dem, cf):
    """Withdrawal, consumption and return per edge (calculate_use_remote,
    :337-353).  avail, dem, cf broadcast to [..., S].  W = avail * min(dem /
    avail, 1) when avail > 0, else 0."""
    pos = avail > 0
    den = torch.where(pos, avail, torch.ones_like(avail))
    frac = cmin(dem / den, torch.ones_like(avail))
    w = torch.where(pos, avail * frac, torch.zeros_like(avail))
    return w, w * (1 - cf), w * cf


def ring_edit(ring, q_w, q_r, nsub, uh_length):
    """Withdrawal and return on the source ring (:451-527).  ring [B, R]
    (logical bins, R = L + n - 1, bin 0 = this step), q_w, q_r [B] m3 s-1.
    S = sum of bins (before the edit); each bin q becomes
    q - q_w * (q / S) + q_r * (q / S), clipped at 0 (S > 0); with S = 0 the
    withdrawal does nothing and the return adds q_r / (L - 1) to every bin
    (these two branches are unreachable from remote use).  Applied where
    q_w > 0 or q_r > 0.  Returns (ring, discharge [B] = bins < n,
    stream [B] = bins >= n, flag [B] bool = withdrawal from an empty ring)."""
    s = ring.sum(1)
    pos = s > 0
    sd = torch.where(pos, s, torch.ones_like(s))
    q = ring
    w = q_w[:, None]
    r = q_r[:, None]
    sh = ring / sd[:, None]
    new = ring
    new = torch.where((w > 0) & pos[:, None], new - w * sh, new)
    new = torch.where((r > 0) & pos[:, None], new + r * sh, new)
    new = torch.where((r > 0) & ~pos[:, None],
                      new + _div(r, uh_length - 1), new)
    new = torch.where(new < 0, torch.zeros_like(new), new)
    cond = ((q_w > 0) | (q_r > 0))[:, None]
    out = torch.where(cond, new, q)
    dis = out[:, :nsub].sum(1)
    st = out[:, nsub:].sum(1)
    return out, dis, st, (q_w > 0) & ~pos


def balance_violation(b, w_src, av_src, dem_src, w_cell, a_rem, dtot_cell,
                      wtrem_e, atrem_e, dtrem_e, act_src, act_e,
                      thresh=WU_BALANCE_ERROR_THRESH):
    """check_water_use_balance_remote (:652-721) as a bool [B] flag: any
    source sector with withdrawn - available > thresh or withdrawn - demand
    > thresh, any receiver with withdrawn_tremote - available_tremote or
    - demand_tremote > thresh, or the cell with thresh * NWUTYPES.  C stops
    the run (log_err); a device flag lets the caller assert once a window."""
    v = (((w_src - av_src > thresh) | (w_src - dem_src > thresh)) &
         act_src).any(-1)
    ve = (((wtrem_e - atrem_e > thresh) | (wtrem_e - dtrem_e > thresh)) &
          act_e).any(-1)
    pad = ve.new_zeros(b.n_src * b.K)                  # per source, any edge
    pad[b.epos] = ve
    v = v | pad.view(b.n_src, b.K).any(1)
    n = w_src.shape[-1]
    return v | (w_cell - a_rem > thresh * n) | (w_cell - dtot_cell > thresh * n)


def remote_batch(b, st, loc, deficit, ring, area, dt, nsub, uh_length,
                 discharge, stream, efr_q=None, gwm=False):
    """wu_remote(iCell) for every source of batch `b` at once
    (wu_remote.c:741-808).

    b          Batch; st RemoteState (updated in place); loc LocalUse;
    deficit    [C] non-renewable deficit (mm), returned updated;
    ring       [B, R] ring of the batch's sources, logical bins (a tensor
               built by the caller, e.g. from CascadeRouter), returned edited;
    area [C] m2, dt s, nsub = rout steps per model step, uh_length = L;
    discharge, stream [B] the sources' rout_var.discharge and .stream (the
    EFR fraction uses the discharge; both are recomputed only where the
    ring was edited, :513-526); efr_q [B] EFR discharge (EFR on) or None.
    Returns a dict: deficit [C], ring [B, R], discharge [B], stream [B],
    withdrawn_remote_cell, returned_cell [B] (mm, after payback),
    available_ring [B] (mm), q_w, q_r [B] m3 s-1, empty_ring_withdrawal [B]
    bool, violation [B] bool.
    """
    S = loc.active.shape[-1]
    src, es, ed, eb = b.src, b.esrc, b.edst, b.eb
    act_src = loc.active[src]
    act_e = loc.active[es] & loc.active[ed]               # [E, S]
    zero_e = torch.zeros(b.n_edges, S, dtype=ring.dtype, device=ring.device)

    # demand (:139-199): receiver state, edge order
    dem_e = demand_tmp(loc.dgw[ed], loc.dsf[ed], loc.wgw[ed], loc.wsf[ed],
                       loc.wdm[ed], st.withdrawn_tremote[ed], gwm)
    dem_e = torch.where(act_e, dem_e, zero_e)
    dem_pad = _pad(dem_e, b)

    # availability (:205-236): the whole source ring, mm over the source area
    a_rem = availability_surf(ring, dt, area[src], efr_q,
                              discharge if efr_q is not None else None)

    # division (:242-297)
    ar_pad, av_src, dtot = division(dem_pad, a_rem, act_src)
    ar_e = _unpad(ar_pad, b)

    # use (:303-370)
    w_e, ret_e, con_e = use(ar_e, dem_e, loc.cons_frac[ed])
    w_e = torch.where(act_e, w_e, zero_e)
    ret_e = torch.where(act_e, ret_e, zero_e)
    con_e = torch.where(act_e, con_e, zero_e)
    w_src = _pad(w_e, b).sum(1)                         # [B, S]
    r_src = _pad(ret_e, b).sum(1)
    w_cell = _seq_sum(w_src)
    r_cell = _seq_sum(r_src)

    # account (:580-617), receivers (unique rows)
    dtrem0 = st.demand_tremote[ed]
    new_dtrem = torch.where(act_e, cmax(dem_e, dtrem0), dtrem0)
    st.demand_tremote = st.demand_tremote.index_put((ed,), new_dtrem)
    st.withdrawn_tremote = st.withdrawn_tremote.index_put(
        (ed,), st.withdrawn_tremote[ed] + w_e)
    st.available_tremote = st.available_tremote.index_put(
        (ed,), st.available_tremote[ed] + torch.where(act_e, ar_e, zero_e))
    st.consumed = st.consumed.index_put((ed,), st.consumed[ed] + con_e)
    # temporaries of the receivers (reset to 0 then set, :127-130, :342-353)
    for name, val in (('demand_tmp', dem_e), ('available_tmp', ar_e),
                      ('withdrawn_tmp', w_e), ('returned_tmp', ret_e),
                      ('consumed_tmp', con_e)):
        old = getattr(st, name)[ed]
        setattr(st, name, getattr(st, name).index_put(
            (ed,), torch.where(act_e, val, old)))
    # source totals (:292, :359-367)
    st.demand_remote = st.demand_remote.index_put(
        (src,), torch.where(act_src, dtot, st.demand_remote[src]))
    st.available_remote = st.available_remote.index_put(
        (src,), torch.where(act_src, av_src, st.available_remote[src]))
    st.withdrawn_remote = st.withdrawn_remote.index_put(
        (src,), torch.where(act_src, w_src, st.withdrawn_remote[src]))
    ret_src_tot = st.returned[src] + torch.where(act_src, r_src,
                                                 torch.zeros_like(r_src))

    # hydrology (:376-528): deficit payback, then the ring
    gate_e = (r_cell > 0)[eb]
    deficit, moved_e, xs = payback(deficit, ret_e, ed, gate_e, act_e)
    moved_pad = _pad(moved_e, b).sum(1)                 # [B, S]
    st.returned = st.returned.index_put(
        (src,), ret_src_tot - moved_pad)
    st.returned = st.returned.index_put(
        (ed,), st.returned[ed] + moved_e)
    x_tot = _seq_sum(_pad(xs, b).sum(1))
    r_left = r_cell - x_tot
    r_left = torch.where(r_left < 0, torch.zeros_like(r_left), r_left)
    r_left = torch.where(r_cell > 0, r_left, r_cell)    # gate

    a_src = area[src]
    q_w = _div(_div(w_cell, MM_PER_M) * a_src, dt)
    q_r = _div(_div(r_left, MM_PER_M) * a_src, dt)
    ring, dis, stm, empty = ring_edit(ring, q_w, q_r, nsub, uh_length)
    edited = (q_w > 0) | (q_r > 0)
    dis = torch.where(edited, dis, discharge)
    stm = torch.where(edited, stm, stream)
    viol = balance_violation(
        b, w_src, av_src, dtot, w_cell, a_rem, _seq_sum(dtot),
        st.withdrawn_tremote[ed], st.available_tremote[ed],
        st.demand_tremote[ed], act_src, act_e)
    return dict(deficit=deficit, ring=ring, discharge=dis, stream=stm,
                withdrawn_remote_cell=w_cell, returned_cell=r_left,
                consumed_cell=_seq_sum(_pad(con_e, b).sum(1)),
                payback_cell=x_tot, returned_total_cell=r_cell,
                available_ring=a_rem, q_w=q_w, q_r=q_r,
                empty_ring_withdrawal=empty, violation=viol)


def _div(a, c):
    """a / c correctly rounded (CUDA turns a division by a Python scalar
    into a product with the reciprocal; a division by a tensor is exact)."""
    return a / torch.full_like(a, c)


def _seq_sum(x):
    """Sum over the last axis in index order (C adds sector after sector)."""
    s = torch.zeros_like(x[..., 0])
    for i in range(x.shape[-1]):
        s = s + x[..., i]
    return s


def payback(deficit, ret_e, edst, gate_e, act_e):
    """Receiver deficit payback from the returned remote water
    (calculate_hydrology_remote, :405-449) on flat edges, gated on the
    source's remote returned total > 0.  deficit [C] (receivers are unique
    rows), ret_e [E, S] returned_remote_tmp, gate_e [E] bool, act_e [E, S].
    Sector by sector: x = min(deficit, returned_tmp) (macro); when the
    deficit is > 0 it loses x (floor 0) and x moves from the source's
    `returned` to the receiver's.  Returns (deficit, moved_e [E, S], x_e
    [E, S]); x_e holds every x of a gated edge, also with no deficit: their
    sum reduces the source's running returned total."""
    zero = torch.zeros_like(ret_e[:, 0])
    moved, xs = [], []
    for s in range(ret_e.shape[-1]):
        m = gate_e & act_e[:, s]
        d = deficit[edst]
        x = cmin(d, ret_e[:, s])
        nd = torch.where(d > 0, d - x, d)
        nd = torch.where(nd < 0, zero, nd)
        deficit = deficit.index_put((edst,), torch.where(m, nd, d))
        moved.append(torch.where(m & (d > 0), x, zero))
        xs.append(torch.where(m, x, zero))
    return deficit, torch.stack(moved, -1), torch.stack(xs, -1)


# --------------------------------------------------------- router adapter
class RemoteSlot(object):
    """The `wu_remote` slot of CascadeRouter.sweep (plugin_simulation.c:
    138): called once per launch group in order with (router, cells).  The
    plan must come from plan_for_router (batch i = a router group).  State,
    local-use results and the deficit are in the router's sweep order; the
    router's ring, discharge, stream, area and deficit are edited in place.
    The group counter is host-side (no device read); call `begin()` at the
    start of every sweep."""

    def __init__(self, plan, router, st, loc, efr_q=None, gwm=False):
        self.plan, self.router, self.st, self.loc = plan, router, st, loc
        self.efr_q, self.gwm = efr_q, gwm
        self.by_group = {g: i for i, g in enumerate(plan.group)}
        self.g = 0

    def begin(self):
        self.g = 0

    def __call__(self, router, cells):
        g, self.g = self.g, self.g + 1
        i = self.by_group.get(g)
        if i is None:
            return
        b, r = self.plan.batches[i], router
        phys = r.logical(torch.arange(r.R, device=r.device))
        ring = r.ring[phys][:, b.src].T.contiguous()
        efr_q = None if self.efr_q is None else self.efr_q[b.src]
        o = remote_batch(b, self.st, self.loc, r.deficit, ring, r.area, r.dt,
                         r.nsub, r.L, r.discharge[b.src], r.stream[b.src],
                         efr_q, self.gwm)
        r.deficit.copy_(o['deficit'])
        r.ring[phys[:, None], b.src[None, :]] = o['ring'].T
        r.discharge[b.src] = o['discharge']
        r.stream[b.src] = o['stream']
        return o
