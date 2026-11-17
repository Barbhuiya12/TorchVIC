"""Differentiable, GPU-native streamflow routing.

VIC routes runoff to gauge locations with RVIC, whose convolution is in
`vic/extensions/rout_rvic/src/rout_convolution.c`. That code maintains a
circular ring buffer and, each time step, adds every source cell's runoff
into the ring weighted by that cell's unit hydrograph, then reads the head of
the ring as discharge.

Stripped of the buffer management, it is a causal convolution:

    discharge(t, outlet) = sum over sources feeding that outlet
                           sum over k  UH[k, source] * runoff[t - k, source]

written one time step at a time. The ring buffer exists because the C
processes a single step per call and must carry future contributions
forward; nothing about the mathematics requires it.

Writing it as a convolution instead has three consequences. It is
differentiable -- convolution is linear, so gradients flow to both the runoff
and the unit hydrograph. It is a single
depthwise `conv1d`, which is the operation GPUs are best at. And it is
verifiable: `route_reference` below transcribes the C's ring-buffer loop
directly, and tests/test_routing.py checks the two agree, so the fast
path is measured against the algorithm it replaces rather than against a
reading of the papers.

Conventions follow RVIC. `runoff` is [time, source] in mm; `unit_hydrograph`
is [uh_steps, source] and each column should sum to 1 for the router to
conserve mass; `source2outlet` maps each source to an outlet index. Discharge
is returned in the same units as the input unless `area` and `dt` are given,
in which case the C's conversion to volume flux is applied.
"""
import numpy as np
import torch
import torch.nn.functional as F

MM_PER_M = 1000.0


def route(runoff, unit_hydrograph, source2outlet, n_outlets,
          source_time_offset=None, area=None, dt=None, skip=0):
    """Route runoff to outlets. [time, source] -> [time, outlet].

    Implemented as one depthwise convolution over all sources at once,
    followed by a scatter-add into outlets. `skip`: the first steps' rows
    are not wanted (a prefix of earlier runoff that only feeds the
    convolution): the result is rows [skip:] of the full one, their sums
    over the sources not formed for the skipped rows (each row's sum is
    its own, so the rows kept are the same bits).
    """
    nt, ns = runoff.shape
    nuh = unit_hydrograph.shape[0]

    # Depthwise conv1d: one filter per source, applied to that source only.
    # conv1d correlates rather than convolves, so the kernel is reversed; the
    # input is left-padded so the result is causal and starts at t = 0 with no
    # contribution from before the record.
    x = runoff.transpose(0, 1).unsqueeze(0)              # [1, source, time]
    w = unit_hydrograph.transpose(0, 1).unsqueeze(1)     # [source, 1, uh]
    w = torch.flip(w, dims=[2])
    xp = F.pad(x, (nuh - 1, 0))
    y = F.conv1d(xp, w, groups=ns).squeeze(0)            # [source, time]

    if source_time_offset is not None:
        # The C shifts a source's contribution further down the ring. A
        # positive offset delays that source by that many steps.
        shifted = torch.zeros_like(y)
        for off in torch.unique(source_time_offset):
            o = int(off)
            m = source_time_offset == off
            if o == 0:
                shifted[m] = y[m]
            elif o < nt:
                shifted[m, o:] = y[m, :nt - o]
        y = shifted

    if skip:
        y = y[:, int(skip):]
    if y.is_cuda and not y.requires_grad:
        # (with a gradient to carry: the autograd index_add below; the
        # kernels here have no backward)
        # index_add on the GPU sums with atomics in no fixed order, so the
        # same run gave discharges a few ulp apart. Sort the sources by
        # outlet (stably: each outlet keeps its sources in source order)
        # and sum each outlet's run with segment_reduce, which adds them
        # from 0 in that order: the CPU index_add's additions, bit for bit.
        # (unsafe: the counts are the bincount of the map, so the checks
        # segment_reduce would make -- two host syncs -- hold; the
        # reduction is the same kernel)
        perm, counts, offs = _outlet_order(source2outlet, n_outlets)
        yp = y.index_select(0, perm)
        from . import seqsum
        if seqsum.available(yp):
            # the same additions in the same order, latency hidden
            # (model/seqsum.py)
            out = seqsum.segment_sum(yp, offs)
        else:
            out = torch.segment_reduce(yp, 'sum', lengths=counts, axis=0,
                                       unsafe=True)
    else:
        out = torch.zeros(n_outlets, int(y.shape[1]), dtype=runoff.dtype,
                          device=runoff.device)
        out = out.index_add(0, source2outlet, y)
    discharge = out.transpose(0, 1)                      # [time, outlet]

    if area is not None and dt is not None:
        discharge = discharge * area / (MM_PER_M * dt)
    return discharge


#: id(source2outlet) -> (weak reference, n_outlets, perm, counts, offsets): a
#: routing map's sort is made once, not at every call (two host syncs)
_ORDER = {}


def _outlet_order(source2outlet, n_outlets):
    import weakref
    k = id(source2outlet)
    hit = _ORDER.get(k)
    if hit is not None and hit[0]() is source2outlet and hit[1] == n_outlets:
        return hit[2:]
    perm = torch.argsort(source2outlet, stable=True)
    counts = torch.bincount(source2outlet, minlength=n_outlets)
    offs = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
    if len(_ORDER) > 32:
        _ORDER.clear()
    try:
        _ORDER[k] = (weakref.ref(source2outlet), n_outlets, perm, counts,
                     offs)
    except TypeError:
        pass
    return perm, counts, offs


def route_reference(runoff, unit_hydrograph, source2outlet, n_outlets,
                    source_time_offset=None):
    """Direct transcription of RVIC's ring-buffer loop, for verification.

    `vic/extensions/rout_rvic/src/rout_convolution.c`, including the zero-then-
    roll of the ring head and the per-source time offset. Deliberately written
    as the C writes it -- one step at a time, with an explicit buffer -- so
    that agreement with `route` above is evidence about the fast path rather
    than two spellings of the same shortcut.

    Slow, and not meant for production use.
    """
    nt, ns = runoff.shape
    nuh = unit_hydrograph.shape[0]
    if source_time_offset is None:
        source_time_offset = torch.zeros(ns, dtype=torch.long)

    ring = torch.zeros(nuh + int(source_time_offset.max()) + 1, n_outlets,
                       dtype=runoff.dtype, device=runoff.device)
    discharge = []

    for t in range(nt):
        # The C zeroes the head, then rolls it to the tail.
        ring = torch.cat([ring[1:], torch.zeros_like(ring[:1])], dim=0)
        for s in range(ns):
            o = int(source2outlet[s])
            off = int(source_time_offset[s])
            for k in range(nuh):
                ring[k + off, o] = (ring[k + off, o] +
                                    unit_hydrograph[k, s] * runoff[t, s])
        discharge.append(ring[0].clone())

    return torch.stack(discharge)


# ---------------------------------------------------------------------------
# Lohmann routing: a unit hydrograph per cell from its flow distance.
# ---------------------------------------------------------------------------

#: VIC's routing defaults (Lohmann et al. 1996; rout.f's input file and
#: RVIC's parameter generator): channel wave velocity, m/s, and
#: hydrodynamic diffusivity, m2/s.
LOHMANN_VELOCITY = 1.5
LOHMANN_DIFFUSION = 800.0

#: The slowest channel the unit hydrograph is made long enough for, m/s.
#: A slower velocity still conserves mass (each column is normalised) but
#: its tail is cut at the length this sets.
_V_FLOOR = 0.2


def lohmann_river_irf(distance, velocity, diffusion, n_steps, dt,
                      substeps=10):
    """The linearised Saint-Venant impulse response over each flow length.

    Lohmann, Nolte-Holube & Raschke (1996), eq. 6, as VIC's routing uses:

        h(x, t) = x / (2 t sqrt(pi t D)) * exp(-(v t - x)^2 / (4 D t))

    the first-passage density of a wave travelling x metres at velocity v
    with diffusion D. It is evaluated `substeps` times per model step and
    summed into steps, because next to the outlet it is narrower than one
    step, and each column is normalised to sum to 1 so the router conserves
    mass exactly whatever the discretisation.

    Computed in logarithms: for a far cell at a short time the prefactor is
    huge and the exponential zero, and their product in linear space is
    inf * 0. For the outlet cell itself (x = 0) the response is all in the
    first step. Both branches of that `where` are finite, so its gradient is
    too (a `where` with an inf or NaN in its unused branch gives a NaN gradient).

    distance [cell] m; velocity, diffusion scalar or [cell].
    Returns [n_steps, cell].
    """
    dtype, device = distance.dtype, distance.device
    x = distance.clamp(min=1.0)
    v = torch.as_tensor(velocity, dtype=dtype, device=device)
    D = torch.as_tensor(diffusion, dtype=dtype, device=device)
    k = torch.arange(n_steps * substeps, dtype=dtype, device=device)
    t = ((k + 0.5) * (dt / substeps)).unsqueeze(1)            # [sub, 1]
    logh = (torch.log(x) - torch.log(torch.as_tensor(2.0, dtype=dtype,
                                                     device=device))
            - 1.5 * torch.log(t) - 0.5 * torch.log(torch.pi * D)
            - (v * t - x) ** 2 / (4.0 * D * t))
    h = torch.exp(logh).reshape(n_steps, substeps, -1).sum(1)  # [step, cell]
    h = h / h.sum(0, keepdim=True).clamp(min=1e-300)
    first = torch.zeros_like(h)
    first[0] = 1.0
    return torch.where((distance < 1.0).unsqueeze(0), first, h)


def lohmann_unit_hydrographs(distance, velocity, diffusion, cell_uh, dt,
                             substeps=10):
    """Each cell's unit hydrograph to the outlet: its within-cell response
    `cell_uh` [n] (sums to 1) convolved with the channel response over its
    flow distance. Returns [n_total, cell], each column summing to 1.

    The channel response is made long enough for a wave at the CURRENT
    velocity (floored at _V_FLOOR m/s) to cross the longest path three times
    over, so its tail is inside the window; the length is recomputed on
    every call, so a slower channel lengthens it.
    """
    xmax = float(distance.detach().max())
    vnow = max(float(torch.as_tensor(velocity).detach().min()), _V_FLOOR)
    n_r = int(min(max(3.0 * xmax / vnow / dt, 1.0) + 48, 4000))
    river = lohmann_river_irf(distance, velocity, diffusion, n_r, dt,
                              substeps=substeps)              # [n_r, cell]
    nuh = cell_uh.shape[0]
    x = river.transpose(0, 1).unsqueeze(1)                    # [cell, 1, n_r]
    if cell_uh.dim() == 2:
        # One within-cell response per cell (an ensemble): depthwise.
        nc = cell_uh.shape[1]
        w = torch.flip(cell_uh, dims=[0]).transpose(0, 1).reshape(nc, 1, nuh)
        y = F.conv1d(F.pad(x.transpose(0, 1), (nuh - 1, nuh - 1)), w,
                     groups=nc)                               # [1, cell, T]
        return y.squeeze(0).transpose(0, 1)
    w = torch.flip(cell_uh, dims=[0]).reshape(1, 1, nuh)
    y = F.conv1d(F.pad(x, (nuh - 1, nuh - 1)), w)             # full conv
    return y.squeeze(1).transpose(0, 1)                       # [n_r+nuh-1, cell]


# ---------------------------------------------------------------------------
# Stepping form: the same routing, one time step per call.
# ---------------------------------------------------------------------------

class _TreeSum:
    """Deterministic, sync-free segmented weighted sum, fixed shapes.

    Items are sorted by segment id; item i has a feature row U[i] (F wide).
    apply(x) returns, per distinct id, sum_i x_i U[i].  Items are packed into
    blocks of `block` rows (zero padded, never straddling a segment), reduced
    with one batched matmul, and the block sums are reduced the same way
    level by level until each segment has one row.  Every index tensor is
    built once on the host, so apply() has no host sync, no data-dependent
    shape and can be captured in a CUDA graph (`torch.segment_reduce` and
    boolean indexing synchronise).
    """

    def __init__(self, seg, U, block=64):
        dev = U.device
        seg = np.asarray(seg)
        self.F = U.shape[1]
        self.n_item = len(seg)
        ids, counts = np.unique(seg, return_counts=True)
        self.ids = torch.as_tensor(ids, device=dev)
        pad = self.n_item
        nb = -(-counts // block)
        owner = np.repeat(np.arange(len(ids)), nb)
        idx = np.full((int(nb.sum()), block), pad, dtype=np.int64)
        start = np.concatenate([[0], np.cumsum(counts)])
        b = 0
        for k, c in enumerate(counts):
            flat = np.full(nb[k] * block, pad, dtype=np.int64)
            flat[:c] = np.arange(start[k], start[k] + c)
            idx[b:b + nb[k]] = flat.reshape(nb[k], block)
            b += nb[k]
        self.idx = torch.as_tensor(idx, device=dev)
        self.idx_flat = self.idx.reshape(-1)
        Ue = torch.cat([U, torch.zeros(1, U.shape[1], dtype=U.dtype,
                                       device=dev)])
        self.U = Ue[self.idx].contiguous()            # [block count, block, F]
        self.levels = []
        while len(owner) > len(ids):
            cnt = np.bincount(owner, minlength=len(ids))
            nb2 = -(-cnt // block)
            ix = np.full((int(nb2.sum()), block), len(owner), dtype=np.int64)
            st = np.concatenate([[0], np.cumsum(cnt)])
            b = 0
            for k, c in enumerate(cnt):
                flat = np.full(nb2[k] * block, len(owner), dtype=np.int64)
                flat[:c] = np.arange(st[k], st[k] + c)
                ix[b:b + nb2[k]] = flat.reshape(nb2[k], block)
                b += nb2[k]
            self.levels.append(torch.as_tensor(ix, device=dev))
            owner = np.repeat(np.arange(len(ids)), nb2)

    def apply(self, x):
        """x [item] -> [id, F]."""
        xe = torch.cat([x, torch.zeros(1, dtype=x.dtype, device=x.device)])
        s = torch.bmm(xe.index_select(0, self.idx_flat).view(self.idx.shape).unsqueeze(1),
                      self.U).squeeze(1)
        for ix in self.levels:
            z = torch.zeros(1, s.shape[1], dtype=s.dtype, device=s.device)
            s = torch.cat([s, z])[ix].sum(1)
        return s


class StepRouter:
    """Step-by-step network routing, equal to `river_network.route_network`.

    State (device tensors, static shape):
      ring [L, node]  m3/s: discharge each node will receive at lag k
                      (slot (head + k) mod L) from everything routed so far.
                      The RVIC ring buffer, one column per gauge/site node.
      head ()         int64 device scalar, the slot that is "now".
    A node is a gauge, a sub-cell gauge, a withdrawal/injection site or a
    canal gauge (a column that natural routing never feeds).  Every
    (cell, node) pair carries its full unit hydrograph (cell response
    convolved with the channel response, columns sum to 1), so adding
    runoff_t scatters  w_p R_c(t) UH[:, p]  into the ring and the head slot
    is discharge_t.  The sum is the whole-series convolution regrouped by
    emission time instead of by arrival time.

    One step = begin(runoff) -> [withdraw / inject / node_inject] -> end().
    `step(...)` runs all of it.  No host synchronisation, no data-dependent
    shapes, so a step can be captured in a CUDA graph.

    Hooks (volumes in m3, per step):
      withdraw(vol[site]) -> delivered[site].  Sites are processed in
          topological order, upstream to downstream, one batched pass per
          topological level (level = longest chain of sites above it).
          A site's delivered volume is
              min(request, cap),
          cap = the largest volume whose removal (through the channel
          response over the distance, from its own node and every node
          downstream, at every lag) leaves every ring slot >= 0.  It
          includes the water that actually reaches the site in this step
          after all upstream withdrawals and injections, so no discharge
          anywhere becomes negative.
      inject(vol[site])   adds vol at the site's cell (reservoir release,
          return flow); it enters the channel, so the site's own cell UH is
          not applied.  In `hooks`/`step`, a site's injection is applied
          after its own withdrawal and before the next level.
      node_inject(vol[node]) adds vol to a node's discharge now (canal
          gauges: delivered diversion).
      intercept(mask)     withdraw everything passing the masked sites
          (reservoir inflow) and return it.  A reservoir is: inflow =
          intercept(its site); storage += inflow - release - losses;
          inject(release at its site, or at a downstream site).

    Ledger (m3, cumulative, per node), checked by `balance()`:
      in_runoff   sum_p w_p R_c(t) dt  (independent of the unit hydrograph)
      in_hooks    injections minus withdrawals credited to the node
      out         discharge x dt read by end()
      stored      ring contents x dt
    in_runoff + in_hooks = out + stored for every node.
    """

    def __init__(self, cell, node, uh, weight_area, n_node, dt,
                 kind=None, site_cell=None, site_node=None, inj=None,
                 site_level=None,                 block=64):
        dev = uh.device
        self.dev, self.dt = dev, float(dt)
        self.L, self.n_pair = uh.shape
        self.n_node = int(n_node)
        self.dtype = uh.dtype
        self.cell = torch.as_tensor(cell, device=dev)
        self.w = torch.as_tensor(weight_area, device=dev, dtype=uh.dtype)
        ones = torch.ones(self.n_pair, 1, dtype=uh.dtype, device=dev)
        self.tree = _TreeSum(np.asarray(node),
                             torch.cat([uh.t(), ones], 1), block)
        self.kind = (torch.zeros(self.n_node, dtype=torch.int8, device=dev)
                     if kind is None else torch.as_tensor(kind, device=dev))
        self.n_site = 0 if site_node is None else len(site_node)
        self.levels = []
        if self.n_site:
            self.site_node = torch.as_tensor(site_node, device=dev)
            slot, nd, kern = (np.asarray(inj[0]), np.asarray(inj[1]),
                              inj[2].to(dev))
            lev = np.asarray(site_level)
            for l in range(int(lev.max()) + 1):
                sites = np.nonzero(lev == l)[0]
                loc = -np.ones(self.n_site, np.int64)
                loc[sites] = np.arange(len(sites))
                pm = np.nonzero(lev[slot] == l)[0]      # still sorted by node
                pm_t = torch.as_tensor(pm, device=dev)
                k = kern[pm_t]
                ones = torch.ones(len(pm), 1, dtype=uh.dtype, device=dev)
                self.levels.append(dict(
                    sites=torch.as_tensor(sites, device=dev),
                    pair_site=torch.as_tensor(loc[slot[pm]], device=dev),
                    pair_node=torch.as_tensor(nd[pm], device=dev),
                    kern=k, pos=k > 0, kern_safe=k.clamp(min=1e-300),
                    tree=_TreeSum(nd[pm], torch.cat([k, ones], 1), block),
                    n=len(sites)))
        self.ar = torch.arange(self.L, device=dev)
        self.reset()

    # -- state -----------------------------------------------------------
    def reset(self):
        z = lambda *s: torch.zeros(*s, dtype=self.dtype, device=self.dev)
        self.ring = z(self.L, self.n_node)
        self.head = torch.zeros((), dtype=torch.long, device=self.dev)
        self.in_runoff, self.in_hooks, self.out = (z(self.n_node),
                                                   z(self.n_node),
                                                   z(self.n_node))
        self.withdrawn = z(max(self.n_site, 1))
        self.injected = z(max(self.n_site, 1))

    _STATE = ('ring', 'head', 'in_runoff', 'in_hooks', 'out', 'withdrawn',
              'injected')

    def state_dict(self):
        return {k: getattr(self, k).clone() for k in self._STATE}

    def load_state_dict(self, sd):
        for k in self._STATE:
            getattr(self, k).copy_(sd[k].to(self.dev))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.dev))

    # -- one step --------------------------------------------------------
    def _slots(self):
        return (self.head + self.ar) % self.L

    def _add(self, acc):
        """ring[(head + k) % L, node] += acc[node, k]  (slots are distinct)."""
        self.ring.index_add_(0, self._slots(), acc.t())

    def begin(self, runoff):
        """Add this step's runoff [cell] (mm per step, runoff + baseflow)."""
        x = runoff.index_select(0, self.cell) * self.w
        t = self.tree.apply(x)                         # [node, L + 1]
        acc = torch.zeros(self.n_node, self.L, dtype=self.dtype,
                          device=self.dev)
        acc.index_copy_(0, self.tree.ids, t[:, :self.L])
        self._add(acc)
        self.in_runoff.index_add_(0, self.tree.ids, t[:, self.L] * self.dt)

    def head_discharge(self):
        """[node] m3/s now (before this step's hooks are applied)."""
        return self.ring.index_select(0, self.head.view(1)).squeeze(0)

    def available(self):
        """[site] m3 passing each site this step, before any hook."""
        return self.head_discharge()[self.site_node].clamp(min=0) * self.dt

    def _apply_level(self, lv, vol):
        """Add vol [site in level] (+ injects, - withdraws) through the kernels."""
        t = lv['tree'].apply(vol[lv['pair_site']] / self.dt)  # [node, L + 1]
        acc = torch.zeros(self.n_node, self.L, dtype=self.dtype,
                          device=self.dev)
        acc.index_copy_(0, lv['tree'].ids, t[:, :self.L])
        self._add(acc)
        # credit = sum of the volumes whose kernels reach the node (kernels sum to 1)
        self.in_hooks.index_add_(0, lv['tree'].ids, t[:, self.L] * self.dt)

    def _cap(self, lv):
        """[site in level] m3: most that can be removed with no negative slot."""
        ring = self.ring.index_select(0, self._slots())       # [lag, node]
        r = ring.index_select(1, lv['pair_node']).t().clamp(min=0)  # [pair, lag]
        ratio = torch.where(lv['pos'], r / lv['kern_safe'], float('inf'))
        m = ratio.amin(1) * self.dt
        cap = torch.full((lv['n'],), float('inf'), dtype=self.dtype,
                         device=self.dev)
        return cap.scatter_reduce(0, lv['pair_site'], m, 'amin')

    def hooks(self, withdraw=None, inject=None):
        """Withdrawals and injections of one step, upstream level by level.
        Returns the delivered withdrawal [site] (zeros if withdraw is None)."""
        got = torch.zeros(self.n_site, dtype=self.dtype, device=self.dev)
        for lv in self.levels:
            sid = lv['sites']
            g = torch.zeros(lv['n'], dtype=self.dtype, device=self.dev)
            if withdraw is not None:
                g = torch.minimum(withdraw[sid].clamp(min=0), self._cap(lv))
            v = -g
            if inject is not None:
                v = v + inject[sid].clamp(min=0)
            self._apply_level(lv, v)
            got.index_copy_(0, sid, g)
        if withdraw is not None:
            self.withdrawn += got
        if inject is not None:
            self.injected += inject.clamp(min=0)
        return got

    def sweep(self, visit):
        """Level-wise withdraw/inject pass with a caller-supplied rule.

        For each topological level of sites, upstream to downstream, calls
            visit(level, sites, cap, avail) -> (take, put) or None
        where `sites` [n] are the site ids of the level, `cap` [n] m3 is the
        most that can be removed there without a negative ring slot, and
        `avail` [n] m3 is the water now passing each site (head slot).
        `take` [n] m3 is withdrawn (clamped to [0, cap]) and `put` [n] m3 is
        injected at the same site (clamped >= 0, after the withdrawal, before
        the next level).  None skips the level.  Counters `withdrawn` and
        `injected` and the ledger are updated.  Fixed shapes and no host
        sync as long as `visit` has none, so it can be captured in a CUDA
        graph.  Reservoirs use it: take the inflow, store, put the release.
        """
        for li, lv in enumerate(self.levels):
            sid = lv['sites']
            cap = self._cap(lv)
            avail = self.head_discharge()[self.site_node[sid]].clamp(min=0) * self.dt
            out = visit(li, sid, cap, avail)
            if out is None:
                continue
            take = torch.minimum(out[0].clamp(min=0), cap)
            put = out[1].clamp(min=0)
            self._apply_level(lv, put - take)
            self.withdrawn.index_add_(0, sid, take)
            self.injected.index_add_(0, sid, put)

    def withdraw(self, request):
        """Take up to `request` [site] m3; returns the delivered volume."""
        return self.hooks(withdraw=request)

    def intercept(self, mask):
        """Withdraw all water that can pass the sites where mask [site] is true."""
        big = torch.full((self.n_site,), float('inf'), dtype=self.dtype,
                         device=self.dev)
        return self.hooks(withdraw=torch.where(mask, big, torch.zeros_like(big)))

    def inject(self, vol):
        self.hooks(inject=vol)
        return vol.clamp(min=0)

    def node_inject(self, vol):
        v = vol.clamp(min=0)
        self.ring.index_add_(0, self.head.view(1), (v / self.dt).unsqueeze(0))
        self.in_hooks += v
        return v

    def end(self):
        """Read this step's discharge [node] m3/s and advance the ring."""
        q = self.head_discharge().clone()
        self.out += q * self.dt
        self.ring.index_fill_(0, self.head.view(1), 0.0)
        self.head.add_(1).remainder_(self.L)
        return q

    def step(self, runoff, baseflow=None, withdraw=None, inject=None,
             node_inject=None):
        """One step; returns discharge [node] and the delivered withdrawal.

        Hooks: withdraw and inject level by level (see class doc), then
        node_inject."""
        self.begin(runoff if baseflow is None else runoff + baseflow)
        got = None
        if withdraw is not None or inject is not None:
            got = self.hooks(withdraw, inject)
        if node_inject is not None:
            self.node_inject(node_inject)
        return self.end(), got

    def balance(self):
        """Per-node ledger [node], m3: in_runoff + in_hooks - out - stored."""
        stored = self.ring.sum(0) * self.dt
        return self.in_runoff + self.in_hooks - self.out - stored
