"""The 'torchvic' flood step split over several GPUs of one node (opt-in).

FloodEngine(..., devices=['cuda:0', 'cuda:1']) keeps the engine's state,
sources, ledger and outputs on the first device (the primary) exactly as the
single-GPU engine does; only the sub-steps of an interval run on strips:

  rows      the grid is cut into K row strips [a_k, b_k) (equal rows); GPU k
            holds its strip plus a halo of H = 2 rows on each inner side
            (its frame [f0_k, f1_k)) and runs the unchanged float64 Triton
            step (TritonStepper's default compacted path) on the frame
  halo      after every sub-step each GPU sends the depth h and momentum
            qx rows and the qy faces of its own rows that its neighbours
            hold as halo. One step moves information one face, and a frame
            edge (closed: NaN slope, so Q = 0 there) spoils only the two
            rows next to it (the limiter factor of the edge row is wrong,
            so the depth of the edge row and the next one), which are halo:
            with the halo refilled every step, the own rows see exactly the
            numbers of the single-GPU step, face by face. Every face is
            computed by the GPU that owns its row(s), with the single-GPU
            expression; nothing is counted twice: the ledger reads each
            boundary face, the maximum depth and the clamp only from own
            rows (k_update2's ROWS mask).
  dt        the CFL step needs the maximum depth of the whole grid: each GPU
            reduces its own rows into its slots (k_update2's NPM atomic
            maxima, bit patterns) and copies them to every other GPU; each
            then runs k_clock2 over all K x NPM slots (a maximum is exact
            in any order), so every GPU takes the same dt, bit for bit.
  ledger    the primary's k_clock2 sums the boundary faces of the whole
            grid in the single-GPU order: the other GPUs copy their own
            rows of the W/E edge faces (and the last one its S faces) into
            the primary's full-length arrays every step. Sources, step
            count, dt extremes: the same expressions on the same numbers.
            The clamp volume (0 with the limiter) is each GPU's own atomic
            sum, added at the interval end in GPU order.
  coupling  the interval's sources are scattered from the primary to the
            frames at the interval start (one copy per array), the own
            rows of every state array, the interval maximum and the
            per-face edge outflow gathered back at its end: the connector,
            restart (state_dict) and coupling see one engine on one device.

The sub-steps of a block (FloodEngine.block steps) are one CUDA graph over
all the GPUs: per step, the kernels of each GPU on its own stream, an event
barrier, the peer copies (one k_push launch storing into the peers' memory
where peer access is on; otherwise each GPU stores into its pinned host
buffer and, after the barrier, each GPU reads its part from there), the
clocks. The halo copies and barriers are
the overhead of the split (measured in docs/connectors/flood.md, section
11).
"""
import contextlib

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                       # CPU-only torch has no Triton
    from ..core.triton_stub import triton, tl

from .triton_step import TritonStepper, k_clock2, _fma

#: halo rows on each inner side of a strip
HALO = 2
#: TritonStepper options a strip may take (the exact default path's tuning)
TUNING = ('block', 'num_warps', 'ublock', 'unum_warps', 'npm', 'cblock',
          'cprog', 'mblock', 'mnum_warps', 'copy', 'sync')

@triton.jit
def k_push(tab_ptr, BLOCK: tl.constexpr, VOL: tl.constexpr):
    """The copies of one strip in one launch: segment s of the table
    (dst address, src address, 8-byte words) copied by programs (s, c),
    BLOCK words each; plain loads and stores of the bit patterns. The
    destination is a peer's memory (peer access on) or a pinned host
    buffer; VOL: the source is a host buffer another GPU wrote (loads not
    cached)."""
    s = tl.program_id(0)
    c = tl.program_id(1)
    dst = tl.load(tab_ptr + 3 * s).to(tl.pointer_type(tl.int64))
    src = tl.load(tab_ptr + 3 * s + 1).to(tl.pointer_type(tl.int64))
    n = tl.load(tab_ptr + 3 * s + 2)
    o = c * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(dst + o, tl.load(src + o, mask=m, volatile=VOL), mask=m)


@triton.jit
def k_barrier(cnt_ptr, own_ptr, tab_ptr, rank, K: tl.constexpr,
              KP: tl.constexpr):
    """A barrier of the K GPUs inside a graph (no cross-device event):
    count this call, store the count into slot `rank` of every GPU's flag
    array (peer memory; the stores of the kernels before it are complete),
    wait until every slot of the own array holds it."""
    c = tl.load(cnt_ptr) + 1
    tl.store(cnt_ptr, c)
    for p in tl.static_range(K):
        a = tl.load(tab_ptr + p).to(tl.pointer_type(tl.int64))
        tl.store(a + rank, c)
    o = tl.arange(0, KP)
    m = o < K
    lag = 1
    while lag > 0:
        f = tl.load(own_ptr + o, mask=m, other=0, volatile=True)
        lag = tl.sum(tl.where(m & (f < c), 1, 0), axis=0)


def split_rows(ny, k):
    """Row bounds of k equal strips: [(a, b)]."""
    b = [round(i * ny / k) for i in range(k + 1)]
    return list(zip(b[:-1], b[1:]))


@contextlib.contextmanager
def _on(dev, stream):
    with torch.cuda.device(dev), torch.cuda.stream(stream):
        yield


class _Strip(object):
    """One GPU's frame: a FloodEngine (state arrays only) and its
    TritonStepper."""

    def __init__(self, eng, k, K, a, b, dev, cfg):
        from .engine import FloodEngine
        ny, nx = eng.ny, eng.nx
        self.k, self.a, self.b, self.dev = k, a, b, dev
        self.f0 = max(a - HALO, 0)
        self.f1 = min(b + HALO, ny)
        f0, f1 = self.f0, self.f1
        nan = float('nan')
        fe = {}
        for e, (mask, slope) in eng.edges.items():
            s = torch.where(mask, slope, torch.full_like(slope, nan))
            if e in 'WE':
                fe[e] = s[f0:f1].cpu()
            elif (e == 'N' and k == 0) or (e == 'S' and k == K - 1):
                fe[e] = s.cpu()
        n = eng.n_cell
        man = n[f0:f1].cpu() if n.dim() == 2 else float(n)
        self.e = FloodEngine(
            eng.z[f0:f1].cpu(), eng.dx, man, free_edges=fe, scheme='torchvic',
            cfl=eng.cfl, max_dt=eng.max_dt, depth_thresh=eng.th,
            max_hflow=eng.max_hflow, theta=eng.theta, g=eng.g,
            drycheck=eng.drycheck, end='clip' if eng.clip_end else
            'overshoot', dtype=eng.dtype, device=dev, block=eng.block,
            compile=False, graphs=False, backend='triton',
            track_edges=eng.track_edges)
        self.st = TritonStepper(self.e, **cfg)
        st = self.st
        if not (st.fused and st.compact) or st.seg or st.quiet or \
                st.fclock or st.wflag or st.pms or st.qflag or st.wfuse \
                or st.mixed:
            raise ValueError('multi-GPU flood: only the default exact '
                             'Triton path (tuning options %s)'
                             % (', '.join(TUNING),))
        st.rows = (a - f0, b - f0)
        self.L = f1 - f0
        self.stream = torch.cuda.Stream(dev)


class MultiGPU(object):
    """The stepper of a FloodEngine on several GPUs (begin / run_block /
    finish, as TritonStepper's begin / step / finish)."""

    def __init__(self, eng, devices, copy='auto', sync='auto', **cfg):
        bad = [x for x in cfg if x not in TUNING]
        if bad:
            raise ValueError('multi-GPU flood: triton_cfg %s not supported '
                             '(tuning options only: %s)'
                             % (bad, ', '.join(TUNING)))
        devs = [torch.device(d) for d in devices]
        K = len(devs)
        if K < 2 or any(d.type != 'cuda' for d in devs):
            raise ValueError('multi-GPU flood: devices must be 2 or more '
                             'CUDA devices, got %s' % (devices,))
        # (a device may repeat: several strips on one GPU, for tests)
        devs = [torch.device('cuda', torch.cuda.current_device()
                             if d.index is None else d.index) for d in devs]
        ny, nx = eng.ny, eng.nx
        rows = split_rows(ny, K)
        if min(b - a for a, b in rows) < 2 * HALO:
            raise ValueError('multi-GPU flood: %d rows over %d devices: a '
                             'strip needs at least %d rows'
                             % (ny, K, 2 * HALO))
        self.eng, self.devs, self.K = eng, devs, K
        self.strips = [_Strip(eng, k, K, a, b, devs[k], cfg)
                       for k, (a, b) in enumerate(rows)]
        s0 = self.strips[0]
        # the primary's clock reads full-length edge-face arrays (the
        # single-GPU k_clock2 inputs); strip 0 writes its rows into them,
        # its halo S faces into a scratch array
        dt_ = eng.dtype
        dev0 = devs[0]
        self.bw = torch.zeros(ny, dtype=dt_, device=dev0)
        self.be = torch.zeros(ny, dtype=dt_, device=dev0)
        self.bn = torch.zeros(nx, dtype=dt_, device=dev0)
        self.bs = torch.zeros(nx, dtype=dt_, device=dev0)
        s0.st.b = dict(W=self.bw, E=self.be, N=self.bn,
                       S=torch.zeros(nx, dtype=dt_, device=dev0))
        # maximum-depth slots: K x NPM per device (padded to a power of 2)
        self.NPM = s0.st.NPM
        if any(s.st.NPM != self.NPM for s in self.strips):
            raise ValueError('multi-GPU flood: one npm for all strips')
        kp = 1
        while kp < K:
            kp *= 2
        self.nslot = kp * self.NPM
        for s in self.strips:
            s.gm = torch.zeros(self.nslot, dtype=torch.int64, device=s.dev)
            s.st.pm = s.gm[s.k * self.NPM:(s.k + 1) * self.NPM]
        # peer access where the devices allow it (torch enables it on a
        # first copy); otherwise the copies go through pinned host memory
        self.p2p = all(torch.cuda.can_device_access_peer(p.index, q.index)
                       for p in devs for q in devs if p != q)
        for p in devs:
            for q in devs:
                if p != q:
                    torch.zeros(1, device=q).copy_(torch.zeros(1, device=p))
        if copy == 'auto':
            copy = 'kernel' if self.p2p else 'host'
        if copy not in ('kernel', 'host'):
            raise ValueError("multi-GPU flood: copy 'kernel' or 'host'")
        if copy == 'kernel' and not self.p2p:
            raise ValueError('multi-GPU flood: copy kernel needs peer '
                             'access between the devices')
        self.copy = copy
        # barriers: flags in device memory written by the peers (kernel
        # copies on distinct devices), else CUDA events
        distinct = len(set(d.index for d in devs)) == K
        if sync == 'auto':
            sync = 'flag' if copy == 'kernel' and distinct else 'event'
        if sync not in ('flag', 'event') or (sync == 'flag' and (
                copy != 'kernel' or not distinct)):
            # (on one GPU the spinning barrier may wait for a graph branch
            # that the device runs after it: flags need distinct devices)
            raise ValueError("multi-GPU flood: sync 'flag' (copy 'kernel', "
                             "distinct devices) or 'event'")
        self.sync = sync
        self._loaded = False
        self.KP = kp
        for s in self.strips:
            s.flag = torch.zeros(kp, dtype=torch.int64, device=s.dev)
            s.cnt = torch.zeros(1, dtype=torch.int64, device=s.dev)
        for s in self.strips:
            s.ftab = torch.tensor([p.flag.data_ptr() for p in self.strips],
                                  dtype=torch.int64, device=s.dev)
        self._bind()
        self._copies()
        self._graph = None

    # ------------------------------------------------------------------
    def _bind(self):
        """Strip 0 steps the primary's clock and ledger tensors."""
        e, s0 = self.eng, self.strips[0]
        s0.e.acc, s0.e.t_rel, s0.e.T = e.acc, e.t_rel, e.T
        self._bound = (e.acc, e.t_rel, e.T)

    def _copies(self):
        """Per strip: the peer copies after its step, (dst, src, bytes)."""
        e, S = self.eng, self.strips
        nx = e.nx
        it = torch.tensor([], dtype=e.dtype).element_size()
        for s in S:
            s.push = []
        for s in S:
            se = s.e

            def rows(dst, dt, sr, r0, r1, w):
                # rows r0..r1-1 (global) of arrays of width w
                if r1 > r0:
                    s.push.append((dt.data_ptr() + (r0 - dst.f0) * w * it,
                                   sr.data_ptr() + (r0 - s.f0) * w * it,
                                   (r1 - r0) * w * it, dst.k))
            for nb in ([S[s.k - 1]] if s.k > 0 else []) + (
                    [S[s.k + 1]] if s.k < self.K - 1 else []):
                # the neighbour's halo rows (own rows of s)
                if nb.k < s.k:
                    r0, r1 = s.a, nb.f1
                else:
                    r0, r1 = nb.f0, s.b
                ne = nb.e
                rows(nb, ne.h, se.h, r0, r1, nx)
                rows(nb, ne.qx, se.qx, r0, r1, nx + 1)
                rows(nb, ne.qy, se.qy, r0 + 1, r1, nx)
            # own maximum-depth slots to every other device
            sl = slice(s.k * self.NPM, (s.k + 1) * self.NPM)
            for p in S:
                if p is not s:
                    g = p.gm[sl]
                    s.push.append((g.data_ptr(), s.gm[sl].data_ptr(),
                                   g.numel() * 8, p.k))
            if s.k > 0:
                # own W/E boundary faces (and the S faces) to the primary
                o = (s.a - s.f0) * it
                n = (s.b - s.a) * it
                st = s.st
                s.push.append((self.bw.data_ptr() + s.a * it,
                               st.b['W'].data_ptr() + o, n, 0))
                s.push.append((self.be.data_ptr() + s.a * it,
                               st.b['E'].data_ptr() + o, n, 0))
                if s.k == self.K - 1:
                    s.push.append((self.bs.data_ptr(), st.b['S'].data_ptr(),
                                   nx * it, 0))
        assert all(n % 8 == 0 for s in S for _, _, n, _ in s.push)

        def table(rows, dev):
            # k_push table (8-byte words: every array is float64 or int64)
            PB = 1024
            t = torch.tensor([[d, r, n // 8] for d, r, n in rows],
                             dtype=torch.int64, device=dev)
            return t, (len(rows), max(-(-(n // 8) // PB) for _, _, n in rows))
        for s in S:
            s.pin = None
            if self.copy == 'kernel':
                s.out = table([(d, r, n) for d, r, n, _ in s.push], s.dev)
            else:
                # staged: s writes its segments into its pinned host buffer
                # (mapped: the device address is the host address), each
                # receiver reads its segments from there
                tot = sum(n for _, _, n, _ in s.push) // 8
                s.pin = torch.zeros(tot, dtype=torch.int64, pin_memory=True)
                base, o, rows = s.pin.data_ptr(), 0, []
                s.stage = []
                for d, r, n, rk in s.push:
                    rows.append((base + o, r, n))
                    s.stage.append((d, base + o, n, rk))
                    o += n
                s.out = table(rows, s.dev)
        if self.copy == 'host':
            for r in S:
                rows = [(d, h, n) for s in S for d, h, n, rk in s.stage
                        if rk == r.k]
                r.inp = table(rows, r.dev)
        self.senders = {r.k: [s for s in S if any(rk == r.k for *_, rk in
                                                  s.push)] for r in S}

    # ------------------------------------------------------------------
    def begin(self):
        """The interval's first dt (TritonStepper.begin on the primary)
        and the frames from the primary's state and sources."""
        e = self.eng
        if any(x is not y for x, y in zip(self._bound,
                                          (e.acc, e.t_rel, e.T))):
            self._bind()
            self._graph = None
        s0 = self.strips[0]
        st0 = s0.st
        st0.RP[0] = e.R
        st0.RP[1] = e.P
        hm = torch.amax(e.h).to(torch.float64)
        loc = e.cfl * e.dx / torch.sqrt(e.g * torch.clamp(hm, min=e.th))
        d = torch.where(hm > e.th, torch.clamp(loc, max=e.max_dt),
                        torch.full_like(hm, e.max_dt))
        if e.dtype == torch.float32:
            d = d.to(torch.float32).to(torch.float64)
        st0.dt.copy_(torch.minimum(d, e.T - e.t_rel).reshape(1))
        e.acc[7] = torch.minimum(e.acc[7], d)
        e.acc[5] = torch.minimum(e.acc[5], st0.dt[0])
        e.acc[6] = torch.maximum(e.acc[6], st0.dt[0])
        for s in self.strips:
            se, f0, f1 = s.e, s.f0, s.f1
            se.h.copy_(e.h[f0:f1])
            se.qx.copy_(e.qx[f0:f1])
            se.qy.copy_(e.qy[f0:f1 + 1])
            se.src.copy_(e.src[f0:f1])
            se.hmax.copy_(e.hmax[f0:f1])
            if se.track_edges:
                se.eacc.zero_()
            if s.k:
                s.st.dt.copy_(st0.dt)
                s.st.RP.copy_(st0.RP)
                se.t_rel.copy_(e.t_rel)
                se.T.copy_(e.T)
                se.acc.copy_(e.acc)
                se.acc[3] = 0.0
        for d in self.devs:
            torch.cuda.synchronize(d)

    def finish(self):
        """Own rows of every strip back to the primary; the clamp volumes
        added in strip order."""
        e = self.eng
        ny, nx = e.ny, e.nx
        for s in self.strips:
            torch.cuda.synchronize(s.dev)
        for s in self.strips:
            se, a, b, f0 = s.e, s.a, s.b, s.f0
            r = slice(a - f0, b - f0)
            e.h[a:b].copy_(se.h[r])
            e.hmax[a:b].copy_(se.hmax[r])
            e.qx[a:b].copy_(se.qx[r])
            e.Qx[a:b].copy_(se.Qx[r])
            # y faces a..b-1 (the last strip also face ny)
            fb = b + 1 if s.k == self.K - 1 else b
            e.qy[a:fb].copy_(se.qy[a - f0:fb - f0])
            e.Qy[a:fb].copy_(se.Qy[a - f0:fb - f0])
            if e.track_edges:
                L = s.L
                ea = se.eacc
                e.eacc[a:b].copy_(ea[r])                       # W
                e.eacc[ny + a:ny + b].copy_(ea[L + r.start:L + r.stop])
                if s.k == 0:
                    e.eacc[2 * ny:2 * ny + nx].copy_(ea[2 * L:2 * L + nx])
                if s.k == self.K - 1:
                    e.eacc[2 * ny + nx:].copy_(ea[2 * L + nx:])
            if s.k:
                e.acc[3] += se.acc[3].to(e.device)
        torch.cuda.synchronize(self.devs[0])

    def sync(self):
        pass

    # ------------------------------------------------------------------
    def _clock(self, s):
        st, se = s.st, s.e
        if s.k == 0:
            b = (self.bw, self.be, self.bn, self.bs)
            ny = self.eng.ny
        else:
            b = (st.b['W'], st.b['E'], st.b['N'], st.b['S'])
            ny = s.L
        k_clock2[(1,)](*b, s.gm, self.nslot, se.acc, st.dt, se.t_rel, se.T,
                       st.RP, st.par, se.eacc, ny, se.nx,
                       F64=se.dtype == torch.float64, EDGE=False, BLOCK=1024,
                       NPM=self.nslot, cnt_ptr=st.fcnt, CNT=True, FMA=_fma(),
                       PMS=False, num_warps=4, enable_fp_fusion=_fma())

    def step(self):
        """One sub-step on every GPU (streams of the strips; call inside
        the capture or eagerly). kernel copies: kernels, barrier (every
        GPU's update done), pushes into the peers, barrier, clocks. host
        copies: kernels, push into the own host buffer (after the
        receivers read the previous one), barrier, the receivers' reads,
        clocks."""
        S = self.strips
        if self.sync == 'flag' and self._loaded:
            for s in S:
                with _on(s.dev, s.stream):
                    s.st._launch_exact(s.e.h, s.e.h)
                    self._barrier(s)
                    k_push[s.out[1]](s.out[0], BLOCK=1024, VOL=False,
                                     num_warps=4)
                    self._barrier(s)
                    self._clock(s)
            return
        kern = self.copy == 'kernel'
        up = []
        for s in S:
            with _on(s.dev, s.stream):
                s.st._launch_exact(s.e.h, s.e.h)
                if kern:
                    ev = torch.cuda.Event()
                    ev.record(s.stream)
                    up.append(ev)
        pu = []
        for s in S:
            with _on(s.dev, s.stream):
                if kern:
                    for p in S:
                        if p is not s:
                            s.stream.wait_event(up[p.k])
                elif self._read is not None:
                    for r in S:
                        if r is not s:
                            s.stream.wait_event(self._read[r.k])
                k_push[s.out[1]](s.out[0], BLOCK=1024, VOL=False,
                                 num_warps=4)
                ev = torch.cuda.Event()
                ev.record(s.stream)
                pu.append(ev)
        rd = []
        for r in S:
            with _on(r.dev, r.stream):
                for p in self.senders[r.k]:
                    if p is not r:
                        r.stream.wait_event(pu[p.k])
                if not kern:
                    k_push[r.inp[1]](r.inp[0], BLOCK=1024, VOL=True,
                                     num_warps=4)
                    ev = torch.cuda.Event()
                    ev.record(r.stream)
                    rd.append(ev)
                self._clock(r)
        self._read = rd or None

    def _load(self):
        """Load every kernel on every device before a spinning barrier
        runs: a module loaded at its first launch (CUDA lazy loading) waits
        for the device to go idle, which a barrier waiting for a GPU whose
        launches have not been issued yet never does. Two steps with event
        barriers, then one k_barrier per device on its own (satisfied)
        flags."""
        if self._loaded:
            return
        dev0 = self.devs[0]
        with torch.cuda.device(dev0):
            self._fork()
            for _ in range(2):
                self.step()
            self._join()
        for s in self.strips:
            with _on(s.dev, s.stream):
                big = torch.full((self.KP,), 1 << 62, dtype=torch.int64,
                                 device=s.dev)
                cnt = torch.zeros(1, dtype=torch.int64, device=s.dev)
                tab = torch.full((self.K,), big.data_ptr(),
                                 dtype=torch.int64, device=s.dev)
                k_barrier[(1,)](cnt, big, tab, 0, K=self.K, KP=self.KP,
                                num_warps=1)
                torch.cuda.synchronize(s.dev)
        self._loaded = True

    def _barrier(self, s):
        k_barrier[(1,)](s.cnt, s.flag, s.ftab, s.k, K=self.K, KP=self.KP,
                        num_warps=1)

    def _fork(self):
        self._read = None
        cur = torch.cuda.current_stream(self.devs[0])
        ev = torch.cuda.Event()
        ev.record(cur)
        for s in self.strips:
            s.stream.wait_event(ev)

    def _join(self):
        cur = torch.cuda.current_stream(self.devs[0])
        for s in self.strips:
            ev = torch.cuda.Event()
            ev.record(s.stream)
            cur.wait_event(ev)

    def _mutable(self):
        m = []
        for s in self.strips:
            se, st = s.e, s.st
            m += [se.h, se.qx, se.qy, se.Qx, se.Qy, se.hmax, se.acc,
                  se.t_rel, se.eacc, st.w, st.dt, st.fcnt, s.gm,
                  st.b['W'], st.b['E'], st.b['N'], st.b['S'], s.flag,
                  s.cnt]
        return m + [self.bs]

    def run_block(self, n, graphs=True):
        """n sub-steps (one graph replay after the first capture)."""
        dev0 = self.devs[0]
        if not self._loaded:
            snap = [x.clone() for x in self._mutable()]
            self._load()
            for d in self.devs:
                torch.cuda.synchronize(d)
            for x, y in zip(self._mutable(), snap):
                x.copy_(y)
            for d in self.devs:
                torch.cuda.synchronize(d)
        if not graphs:
            with torch.cuda.device(dev0):
                self._fork()
                for _ in range(n):
                    self.step()
                self._join()
            return
        if self._graph is None or self._graph[1] != n:
            snap = [x.clone() for x in self._mutable()]
            with torch.cuda.device(dev0):
                self._fork()
                for _ in range(2):          # compile and load the kernels
                    self.step()
                self._join()
            for d in self.devs:
                torch.cuda.synchronize(d)
            for x, y in zip(self._mutable(), snap):
                x.copy_(y)
            for d in self.devs:
                torch.cuda.synchronize(d)
            g = torch.cuda.CUDAGraph()
            cap = torch.cuda.Stream(dev0)
            with torch.cuda.device(dev0):
                with torch.cuda.graph(g, stream=cap):
                    self._fork()
                    for _ in range(n):
                        self.step()
                    self._join()
            for d in self.devs:
                torch.cuda.synchronize(d)
            for x, y in zip(self._mutable(), snap):
                x.copy_(y)
            for d in self.devs:
                torch.cuda.synchronize(d)
            self._graph = (g, n)
        with torch.cuda.device(dev0):
            self._graph[0].replay()
