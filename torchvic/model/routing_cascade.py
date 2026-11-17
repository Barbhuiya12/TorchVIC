"""VIC-WUR routing cascade (component C1, docs/connectors/vicwur.md).

Port of the VIC-WUR 3.0 routing plugin
(<data root>/build/vicwur3/src_tree/vic/plugins, cited as `routing/src/...`,
`general/src/...`, `wateruse/src/...`):

* every cell has two unit hydrographs of UH_LENGTH = L sub-steps, one for its
  own runoff and one for the inflow from its upstream cells, and a ring
  `dt_discharge[L + n - 1]` of discharge shares (m3 s-1), with
  n = ROUT_STEPS_PER_DAY / MODEL_STEPS_PER_DAY (rout_basin_run.c:53-54);
* per model step `rout_run` turns OUT_RUNOFF + OUT_BASEFLOW (after the
  non-renewable payback) into m3 s-1 (rout_run.c:48-81), and then, cell by
  cell in upstream-first order, `rout_basin_run` shifts the ring by n bins
  with zero fill, adds the runoff convolution, gathers the upstream bins
  0..n-1 as inflow, adds the inflow convolution and sums bins < n as
  discharge and bins >= n as stream (rout_basin_run.c:66-123).

The serial cell order becomes a level sweep. The levels are the generations
of `rout_basin_set_order` (rout_init.c:420-483), or of
`wu_set_routing_order` when remote water use is on (wu_init.c:132-290);
sources that share a receiver in one level are split into sub-levels in
index order (docs/connectors/vicwur.md). Cells are stored in sweep order, so each launch
group is a contiguous range.

Device paths: on CUDA one Triton kernel moves the ring head and adds the
runoff convolution for all cells; the sweep is a few launches of the fused
level kernel (one per wide level, one for the whole narrow tail), each
working on [cells x ring bins] tiles, and
`capture()` records the step in a CUDA graph. On the CPU the same arithmetic
runs as torch ops, one group at a time. Both paths do the C operations in
the C order, so they agree with the C code bit for bit.

The ring service (`ring_sum`, `scale_prop`, `head_replace`, `recompute`) is
what the sweep slots C2-C7 use to edit the ring (wu_run_local.c:470-540,
wu_nonrenewable.c:253-304, dam_run.c:97-127, irr_use.c:166-200,
lu_apply.c:851-877).
"""
import numpy as np
import torch

MAX_UPSTREAM = 8                 # routing/include/routing.h


def _div(a, b):
    """a / b correctly rounded. torch on CUDA turns a division by a Python
    scalar into a product with its reciprocal, which is not C's division;
    dividing by a full tensor is."""
    return a / torch.full_like(a, b)
SLOTS = ('local_dam', 'wu_local', 'global_dam', 'wu_remote')


class RoutingOrderError(ValueError):
    """A network the C code would hang on, overrun a buffer with, or reject
    (rout_init.c:287-297, 446-479; wu_init.c:274-290, 319-330)."""


def upstream_table(downstream):
    """[N, 8] upstream cells in ascending index (rout_init.c:285-308), -1
    padded, and their count. More than MAX_UPSTREAM raises: C writes past
    its buffer before its check (:288-297)."""
    down = np.asarray(downstream, np.int64)
    n = len(down)
    src = np.nonzero(down != np.arange(n))[0]          # i != j (:287)
    dst = down[src]
    order = np.lexsort((src, dst))
    src, dst = src[order], dst[order]
    nup = np.bincount(dst, minlength=n)
    if n and nup.max(initial=0) > MAX_UPSTREAM:
        c = int(np.argmax(nup))
        raise RoutingOrderError('cell %d has %d upstream cells; VIC-WUR '
                                'allows %d' % (c, nup[c], MAX_UPSTREAM))
    first = np.concatenate([[0], np.cumsum(nup)[:-1]])
    up = -np.ones((n, MAX_UPSTREAM), np.int64)
    up[dst, np.arange(len(src)) - first[dst]] = src
    return up, nup


class Schedule(object):
    """Routing order and launch groups.

    order   [N] the C routing_order (generation by generation, ascending
            index inside one, rout_init.c:446-479 / wu_init.c:172-232)
    level   [N] generation of every cell
    sub     [N] sub-level inside the generation (0 without remote edges)
    perm    [N] sweep position -> cell: sorted by (level, sub, index)
    pos     [N] cell -> sweep position
    gptr    [G+1] group g holds sweep positions gptr[g]..gptr[g+1]-1
    """


def build_schedule(downstream, receiving=None):
    """Levels of the upstream-ordered network.

    downstream  [N] downstream cell, own index at an outlet
    receiving   optional list, per cell, of the cells it supplies by remote
                water use (wu_con.receiving after wu_set_receiving)
    """
    down = np.asarray(downstream, np.int64)
    n = len(down)
    up, nup = upstream_table(down)
    rcv = None
    if receiving is not None:
        rcv = -np.ones((n, max([len(r) for r in receiving] + [1])), np.int64)
        for i, r in enumerate(receiving):
            rcv[i, :len(r)] = r
    done = np.zeros(n + 1, bool)          # done[n] = padding, always true
    done[n] = True
    upx = np.where(up < 0, n, up)
    rcx = None if rcv is None else np.where(rcv < 0, n, rcv)
    level = -np.ones(n, np.int64)
    gen = 0
    left = n
    while left:
        ready = ~done[:n] & done[upx].all(1)
        if rcx is not None:                       # wu_init.c:203-221
            ready &= done[rcx].all(1)
        k = int(ready.sum())
        if k == 0:
            raise RoutingOrderError(
                '%d cells cannot be ordered: a downstream cycle%s' %
                (left, '' if rcv is None else ' or a receiving cell '
                 'downstream of its source'))
        level[ready] = gen
        done[:n] |= ready                 # done_fin after the pass (:474-478)
        left -= k
        gen += 1
    order = np.lexsort((np.arange(n), level))
    sub = np.zeros(n, np.int64)
    if rcv is not None:
        inv = np.empty(n, np.int64)
        inv[order] = np.arange(n)
        for i in range(n):                        # wu_init.c:313-330
            for r in rcv[i][rcv[i] >= 0]:
                if inv[r] >= inv[down[i]]:
                    raise RoutingOrderError(
                        'receiving cell %d of source %d is not before the '
                        'source downstream cell %d' % (r, i, down[i]))
        last = {}                         # receiver -> (level, sub) last used
        for i in order:                   # greedy colouring, index order
            s = 0
            for r in rcv[i][rcv[i] >= 0]:
                lv, sb = last.get(int(r), (-1, -1))
                if lv == level[i]:
                    s = max(s, sb + 1)
            sub[i] = s
            for r in rcv[i][rcv[i] >= 0]:
                last[int(r)] = (int(level[i]), s)
    perm = np.lexsort((np.arange(n), sub, level))
    key = level[perm] * (sub.max(initial=0) + 1) + sub[perm]
    gptr = np.concatenate([[0], np.nonzero(np.diff(key))[0] + 1, [n]])
    s = Schedule()
    s.order, s.level, s.sub, s.perm = order, level, sub, perm
    s.pos = np.empty(n, np.int64)
    s.pos[perm] = np.arange(n)
    s.gptr = gptr.astype(np.int64) if n else np.zeros(1, np.int64)
    s.up, s.nup = up, nup
    s.n_levels = gen
    return s


class CascadeRouter(object):
    """VIC-WUR routing for N cells.

    downstream [N]       own index at an outlet (params.read_routing)
    uh_inflow, uh_runoff [L, N], already renormalised (rout_init.c:78-110)
    area [N] m2, dt model step s, nsub = rout steps per model step
    receiving            remote edges for the order (build_schedule)
    force                True when DISCHARGE forcing feeds the inflow
                         (FORCE_ROUTING, rout_basin_run.c:89-91)
    nonrenew_runoff, gwm the rout_run options (rout_run.c:49-77)
    tail                 groups with at most this many cells are fused into
                         single-program launches (CUDA; default one tile)
    """

    def __init__(self, downstream, uh_inflow, uh_runoff, area, dt, nsub,
                 receiving=None, force=False, nonrenew_runoff=False,
                 gwm=False, device='cpu', tail=None):
        self.device = torch.device(device)
        dev = self.device
        sch = build_schedule(downstream, receiving)
        self.schedule = sch
        n = len(sch.perm)
        L = int(np.asarray(uh_inflow).shape[0])
        self.n, self.L, self.nsub = n, L, int(nsub)
        self.R = L + self.nsub - 1
        self.dt = float(dt)
        self.use_force, self.nonrenew_runoff, self.gwm = (bool(force),
                                                          bool(nonrenew_runoff),
                                                          bool(gwm))
        f64 = dict(dtype=torch.float64, device=dev)
        perm = torch.as_tensor(sch.perm, device=dev)
        self.perm = perm                       # sweep position -> cell
        self.pos = torch.as_tensor(sch.pos, device=dev)
        up = np.where(sch.up[sch.perm] >= 0,
                      sch.pos[np.maximum(sch.up[sch.perm], 0)], -1)
        self.up = torch.as_tensor(up, device=dev).contiguous()
        self.up_t = self.up.T.clamp(min=0).contiguous()   # [8, N], kernels
        self.nup = torch.as_tensor(sch.nup[sch.perm], device=dev)
        self.uh_inflow = torch.as_tensor(np.asarray(uh_inflow, np.float64)
                                         [:, sch.perm], **f64).contiguous()
        self.uh_runoff = torch.as_tensor(np.asarray(uh_runoff, np.float64)
                                         [:, sch.perm], **f64).contiguous()
        self.area = torch.as_tensor(np.asarray(area, np.float64)[sch.perm],
                                    **f64)
        self.gptr = torch.as_tensor(sch.gptr, device=dev)
        self._gptr = sch.gptr
        # state, sweep order
        self.ring = torch.zeros(self.R, n, **f64)       # physical bins
        self.head = torch.zeros((), dtype=torch.int64, device=dev)
        self.deficit = torch.zeros(n, **f64)
        self.runoff = torch.zeros(n, **f64)
        self.inflow = torch.zeros(n, **f64)
        self.discharge = torch.zeros(n, **f64)
        self.stream = torch.zeros(n, **f64)
        self.force = torch.zeros(n, **f64)
        self._bins = torch.arange(self.R, device=dev)
        if dev.type == 'cuda':
            from .routing_cascade_kernels import tile
            self.block = tile(self.R)[1]        # cells per kernel tile
        else:
            self.block = 128
        self._plan = self._launch_plan(self.block if tail is None
                                       else tail)
        self._graph = None

    # ------------------------------------------------------------ helpers
    @classmethod
    def from_params(cls, rp, area, dt, rout_steps_per_day,
                    model_steps_per_day=1, **kw):
        """From torchvic.vicwur.params.read_routing output."""
        nsub = rout_steps_per_day // model_steps_per_day
        return cls(rp.downstream.cpu().numpy(), rp.uh_inflow.cpu().numpy(),
                   rp.uh_runoff.cpu().numpy(), area, dt, nsub, **kw)

    def _launch_plan(self, tail):
        """[(g0, g1, nprog)]: a wide group alone on many programs, a run of
        narrow groups on one program."""
        sizes = np.diff(self._gptr)
        plan, g = [], 0
        G = len(sizes)
        while g < G:
            if sizes[g] > tail:
                plan.append((g, g + 1, -(-int(sizes[g]) // self.block)))
                g += 1
            else:
                g1 = g
                while g1 < G and sizes[g1] <= tail:
                    g1 += 1
                plan.append((g, g1, 1))
                g = g1
        return plan

    def to_sweep(self, x):
        """[N] (cell order) -> sweep order."""
        return torch.as_tensor(x, dtype=torch.float64,
                               device=self.device)[self.perm]

    def to_cells(self, x):
        """[..., N] (sweep order) -> cell order."""
        out = torch.empty_like(x)
        out[..., self.perm] = x
        return out

    def logical(self, i):
        """Physical bin of logical bin(s) i."""
        return (self.head + i) % self.R

    def ring_logical(self):
        """[R, N] logical bins (bin 0 = this step), sweep order."""
        return self.ring[self.logical(self._bins)]

    def group_cells(self, g):
        return torch.arange(int(self._gptr[g]), int(self._gptr[g + 1]),
                            device=self.device)

    # ------------------------------------------------------------ the step
    def rout_run(self, out_runoff, out_baseflow):
        """P1 (rout_run.c:48-81), all cells. Inputs mm per step, cell order.
        Updates the non-renewable deficit by the payback."""
        if self.device.type == 'cuda':
            from . import routing_cascade_kernels as K
            f = dict(dtype=torch.float64, device=self.device)
            K.rout_run(self, torch.as_tensor(out_runoff, **f).contiguous(),
                       None if self.gwm else
                       torch.as_tensor(out_baseflow, **f).contiguous())
            return
        rs = self.to_sweep(out_runoff)
        if self.gwm:                                   # :49-51
            rb = torch.zeros_like(rs)
        else:
            rb = self.to_sweep(out_baseflow)
            d = self.deficit
            on = d > 0                                 # :56
            gt = rb > d
            d1 = torch.where(on, torch.where(gt, torch.zeros_like(d),
                                             d - rb), d)
            rb = torch.where(on, torch.where(gt, rb - d,
                                             torch.zeros_like(rb)), rb)
            if self.nonrenew_runoff:                   # :66-75
                gt = rs > d1
                d2 = torch.where(on, torch.where(gt, torch.zeros_like(d1),
                                                 d1 - rs), d1)
                rs = torch.where(on, torch.where(gt, rs - d1,
                                                 torch.zeros_like(rs)), rs)
                d1 = d2
            self.deficit.copy_(d1)
        self.runoff.copy_(_div((rs + rb) * self.area, self.dt * 1000))  # :78-81

    def set_force(self, discharge):
        """DISCHARGE forcing (m3 s-1, cell order), held by the caller."""
        self.force.copy_(self.to_sweep(discharge))

    def begin(self):
        """Shift every ring by nsub bins with zero fill and add the runoff
        convolution (rout_basin_run.c:66-84). Before the sweep, no cell's
        ring is read until its own turn, so all cells do this at once."""
        if self.device.type == 'cuda':
            from . import routing_cascade_kernels as K
            K.begin(self.ring, self.head, self.runoff, self.uh_runoff,
                    self.nsub)
            self.head.add_(self.nsub).remainder_(self.R)
            return
        self.head.add_(self.nsub).remainder_(self.R)
        sig = _div(self.runoff, self.nsub)                 # :74-76
        for i in range(self.R):
            acc = self._conv_bin(i, lambda k: sig, self.uh_runoff, None)
            p = self.logical(i)
            q = self.ring[p] if i < self.R - self.nsub else \
                torch.zeros_like(sig)                   # cshift zero fill
            self.ring[p] = q + acc

    def _conv_bin(self, i, sig, uh, cells):
        """Bin i of convolve (convolute.c): sum over k ascending of
        signal[k] * uh[i - k], from 0."""
        acc = None
        for k in range(self.nsub):
            j = i - k
            if 0 <= j < self.L:
                u = uh[j] if cells is None else uh[j, cells]
                t = sig(k) * u
                acc = t if acc is None else acc + t     # 0 + t == t
        return acc

    def route(self, cells):
        """rout_basin_run.c:86-123 for the cells (sweep positions) of one
        group, CPU path. begin() has done :66-84."""
        nc = cells.numel()
        f = (_div(self.force[cells], self.nsub) if self.use_force else
             torch.zeros(nc, dtype=torch.float64, device=self.device))
        up, nup = self.up[cells], self.nup[cells]
        infl = torch.zeros_like(f)
        dtin = []
        for i in range(self.nsub):
            p = self.logical(i)
            d = f
            for j in range(MAX_UPSTREAM):
                ok = j < nup
                if not bool(ok.any()):
                    break
                v = self.ring[p, up[:, j].clamp(min=0)]
                d = torch.where(ok, d + v, d)
            dtin.append(d)
            infl = infl + d
        self.inflow[cells] = infl
        dis = torch.zeros_like(f)
        st = torch.zeros_like(f)
        for i in range(self.R):
            acc = self._conv_bin(i, lambda k: dtin[k], self.uh_inflow, cells)
            p = self.logical(i)
            q = self.ring[p, cells] + acc
            self.ring[p, cells] = q
            if i < self.nsub:
                dis = dis + q
            else:
                st = st + q
        self.discharge[cells] = dis
        self.stream[cells] = st

    def route_group(self, g, cells=None):
        """The route slot of launch group g (rout_basin_run.c:86-123 for
        its cells), on the device path that `sweep` takes with slots.
        begin() has run."""
        cells = self.group_cells(g) if cells is None else cells
        if self.device.type == 'cuda':
            from . import routing_cascade_kernels as K
            K.level(self, g, g + 1, -(-cells.numel() // self.block))
        else:
            self.route(cells)

    def sweep(self, slots=None):
        """P2 over all levels. Without slots, CUDA runs the launch plan
        (fused level kernel); with slots, each group runs local_dam, route,
        wu_local, global_dam, wu_remote (plugin_simulation.c:125-138), each
        slot a callable (router, cells)."""
        slots = slots or {}
        cuda = self.device.type == 'cuda'
        if cuda and not slots:
            from . import routing_cascade_kernels as K
            for g0, g1, npg in self._plan:
                K.level(self, g0, g1, npg, sums=False)
            K.sums(self)
            return
        for g in range(len(self._gptr) - 1):
            cells = self.group_cells(g)
            if 'local_dam' in slots:
                slots['local_dam'](self, cells)
            if cuda:
                from . import routing_cascade_kernels as K
                K.level(self, g, g + 1, -(-cells.numel() // self.block))
            else:
                self.route(cells)
            for s in SLOTS[1:]:
                if s in slots:
                    slots[s](self, cells)

    def step(self, out_runoff, out_baseflow, slots=None):
        """One model step of routing: P1 rout_run, then P2."""
        self.rout_run(out_runoff, out_baseflow)
        self.begin()
        self.sweep(slots)

    # -------------------------------------------------- CUDA graph replay
    def capture(self):
        """Record begin + sweep (no slots) in a CUDA graph; `replay()` then
        runs a whole routing step after rout_run with one launch call."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        ring, head = self.ring.clone(), self.head.clone()
        out = (self.inflow, self.discharge, self.stream)
        keep = [t.clone() for t in out]
        with torch.cuda.stream(s):                 # compile + warm up
            self.begin()
            self.sweep()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.begin()
            self.sweep()
        self.ring.copy_(ring)
        self.head.copy_(head)
        for t, k in zip(out, keep):
            t.copy_(k)
        self._graph = g
        return g

    def replay(self):
        self._graph.replay()

    # ----------------------------------------------------------- outputs
    def outputs(self):
        """rout_put_data (rout_output.c:121-141), cell order."""
        sm = self.stream * self.dt / self.area * 1000
        return {k: self.to_cells(v) for k, v in (
            ('OUT_DISCHARGE', self.discharge),
            ('OUT_STREAM_INFLOW', self.inflow),
            ('OUT_STREAM_RUNOFF', self.runoff),
            ('OUT_STREAM_MOIST', sm),
            ('OUT_NONREN_DEFICIT', self.deficit))}

    def state(self):
        """discharge_dt [N, L+n-1] in cell order (rout_state.c:224) and the
        deficit (VIC-WUR 3.0 rout_state.c, STATE_NONRENEW_DEFICIT)."""
        return {'discharge_dt': self.to_cells(self.ring_logical()).T.clone(),
                'nonrenew_deficit': self.to_cells(self.deficit).clone()}

    def load_state(self, discharge_dt, nonrenew_deficit=None):
        """Restore; the ring length must be L+n-1 (rout_restore.c:45-55).
        discharge and stream are re-derived as in
        rout_compute_derived_state_vars."""
        dd = torch.as_tensor(discharge_dt, dtype=torch.float64,
                             device=self.device)
        if dd.shape != (self.n, self.R):
            raise ValueError('Rout delta time in state file does not match '
                             'parameter file')
        self.head.zero_()
        self.ring.copy_(dd[self.perm].T)
        if nonrenew_deficit is not None:
            self.deficit.copy_(self.to_sweep(nonrenew_deficit))
        self.recompute(torch.arange(self.n, device=self.device))

    # ------------------------------------------------------ ring service
    # (p is a 0-d device tensor, the physical bin: indexing with it would
    # read it on the host, so gather and scatter by index tensors)
    def _bin(self, p, cells):
        return self.ring.index_select(0, p.reshape(1)).squeeze(0
                                                               ).index_select(
            0, cells)

    def _set_bin(self, p, cells, q):
        self.ring.index_put_((p.reshape(1).expand(cells.shape[0]), cells), q)

    def ring_sum(self, cells, lo=0, hi=None):
        """Sequential sum of logical bins lo..hi-1 (wu_run_local.c:474-478,
        wu_nonrenewable.c:257-262)."""
        hi = self.R if hi is None else hi
        s = torch.zeros(cells.numel(), dtype=torch.float64,
                        device=self.device)
        for i in range(lo, hi):
            s = s + self._bin(self.logical(i), cells)
        return s

    def scale_prop(self, cells, withdraw=None, ret=None, lo=0, mode='exact'):
        """Withdraw and return in proportion to the bins lo..R-1
        (wu_run_local.c:470-525; lo = n for the future-bin returns of
        wu_nonrenewable.c:254-289 and irr_use.c:166-195). Volumes in
        m3 s-1 per cell. With no water in the bins a return is spread as
        ret / (L - 1) per bin in `exact` (:513-517), ret / (R - lo) in
        `fixed`; a withdrawal from empty bins sets the error flag that C
        log_errs on (:500-503). Returns that flag per cell. Call
        recompute() after."""
        S = self.ring_sum(cells, lo)
        z = torch.zeros_like(S)
        w = z if withdraw is None else withdraw
        r = z if ret is None else ret
        div = (self.L - 1) if mode == 'exact' else (self.R - lo)
        pos = S > 0
        # a cell's bins are edited (and clipped at 0) only when it withdraws
        # or returns (wu_run_local.c:472): a bin made negative by a negative
        # forced inflow (VIC-WUR 3.0 MODFLOW, a losing river) stays as it is
        gate = (w > 0) | (r > 0)
        for i in range(lo, self.R):
            p = self.logical(i)
            q0 = self._bin(p, cells)
            q = q0
            q = torch.where((w > 0) & pos, q - w * (q0 / S), q)
            q = torch.where((r > 0) & pos, q + r * (q0 / S), q)
            q = torch.where((r > 0) & ~pos, q + _div(r, div), q)
            self._set_bin(p, cells, torch.where(gate & (q < 0), z, q))
        return (w > 0) & ~pos

    def head_replace(self, cells, keep, add):
        """Bins 0..n-1: q = q * keep + add, clipped at 0; then discharge
        = sum of them (global dam, dam_run.c:110-126, with keep =
        1 - inflow_frac and add = release * M3_PER_HM3 / dt / n)."""
        z = torch.zeros(cells.numel(), dtype=torch.float64,
                        device=self.device)
        dis = z
        for i in range(self.nsub):
            p = self.logical(i)
            q = self._bin(p, cells) * keep + add
            q = torch.where(q < 0, z, q)
            self._set_bin(p, cells, q)
        for i in range(self.nsub):
            dis = dis + self._bin(self.logical(i), cells)
        self.discharge[cells] = dis

    def recompute(self, cells):
        """discharge = bins < n, stream = bins >= n, summed in bin order
        (wu_run_local.c:527-540)."""
        z = torch.zeros(cells.numel(), dtype=torch.float64,
                        device=self.device)
        dis, st = z, z
        for i in range(self.R):
            q = self._bin(self.logical(i), cells)
            if i < self.nsub:
                dis = dis + q
            else:
                st = st + q
        self.discharge[cells] = dis
        self.stream[cells] = st
