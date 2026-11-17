"""Local-inertial (ACC) flood hydraulics on a structured grid, in PyTorch.

The equations are the local-inertial (acceleration) form of the shallow
water equations of Bates, Horritt and Fewtrell (2010, J. Hydrol. 387) with
the semi-implicit Manning friction and the 2D friction vector of de Almeida
et al. (2012, WRR 48), written exactly as LISFLOOD-FP 8.x evaluates them
(`fp_acc.cpp` CalcFPQxAcc/CalcFPQyAcc, `cuda/acc/cuda_acc_solver.cu`). No
LISFLOOD-FP code is used; the formulas are re-implemented from the papers and
checked against the LISFLOOD-FP binary (tools/flood/validate_engine.py).

Grid and state (row 0 = north, as an ESRI ASCII raster):

    h   [ny, nx]       water depth (m)
    Qx  [ny, nx+1]     discharge through the west face of cell (j, i)
                       (m3/s, +east); Qx[:, 0] and Qx[:, nx] are boundaries
    Qy  [ny+1, nx]     discharge through the north face of cell (j, i)
                       (m3/s, +south); Qy[0] and Qy[ny] are boundaries
    qx, qy             the same divided by dx (m2/s): the momentum carried
                       to the next step (LISFLOOD-FP's Qxold/Qyold)

One face, cells 0 (upstream index) and 1:

    hflow = min(max(z0+h0, z1+h1) - max(z0, z1), MaxHflow),  >= 0
    Sf    = -((z0+h0) - (z1+h1)) / dx
    |q|   = sqrt(q0^2 + qbar^2)   (qbar: the four perpendicular faces)
    Q     = (q0 - g dt hflow Sf) / (1 + g dt hflow n^2 |q| / hflow^(10/3)) dx

when either cell is wet (h > DepthThresh) and hflow > DepthThresh, else 0.
Free (normal depth) boundaries use the LISFLOOD-FP form with |q0| and the
given floodplain slope (or the local slope for slope -1); closed edges carry
no flow. Continuity: h += dt (Qw - Qe + Qn - Qs) / dx^2.

Schemes (`scheme`):

  'torchvic'      the model of record. Global CFL step
                  dt = min(max_dt, cfl dx / sqrt(g hmax)) from the depth at
                  the step start (the LISFLOOD-FP CPU rule), shortened so an
                  interval ends exactly at its end time. Sources (runoff
                  rate x dt, point inflow x dt) enter before continuity.
                  Wet/dry: a cell may not give more than it holds; every
                  outgoing face is scaled by its donor's factor
                  min(1, h A / outgoing volume) (conservative, parallel; the
                  scaled flux is also the momentum kept). The mass ledger
                  (sources in, boundary out, storage) closes to round-off.
  'lisflood_gpu'  the step of LISFLOOD-FP's CUDA ACC solver, quirks included
                  (for validation): per-cell running-minimum time step
                  (a newly wet cell contributes max_dt; a wet cell's step
                  never grows until it dries), dynamic rain added at the step
                  start with the previous step's dt (none on the first step),
                  point sources before continuity, the wet/dry limiter on
                  local copies of the face fluxes (not conservative: the
                  neighbour still receives the unscaled flux), the run
                  overshooting the end time by the last step.
  'lisflood_cpu'  the CPU ACC order: CPU dt rule, DryCheck before the
                  sources (here on all cells at once instead of LISFLOOD's
                  sequential sweep), rain and point sources with the current
                  step, overshoot at the end.

LISFLOOD-FP 8.x behaviour found while matching it (build of the Zenodo 8.2
record, "8.1.0 (double)"):

  * CUDA ACC time step: a per-cell field holding the running minimum of
    cfl dx / sqrt(g h), reset to `initial_tstep` only when the cell dries;
    a cell wet for the first time contributes `initial_tstep` whatever its
    depth. The step is therefore never above `initial_tstep` and never
    grows during an event. The CPU ACC uses min(initial_tstep, CFL(max h))
    afresh every step. (With the connector's initial_tstep = 1 s the GPU
    takes 86400 steps per day whatever the depths.)
  * CUDA ACC dynamic rain is added at the top of the step with the previous
    step's dt: the first step gets none (the "first-step rain" loss) and
    each later step gets rate(t_n) x dt_(n-1).
  * CUDA ACC wet/dry limiter scales the outgoing faces on local copies, so
    the neighbour still receives the unscaled flux: water is created where
    a cell drains completely (the scheme `lisflood_gpu` reproduces it; its
    ledger residual is that water).
  * CUDA ACC `.mass` record: Qout stays 0 for FREE edges although water
    leaves; Qin is right.
  * Both run `while t < sim_time` without shortening the last step: a run
    ends up to one step after sim_time (volumes then exceed rate x
    sim_time).
  * The CPU DryCheck sweeps the cells in order and writes the scaled faces
    back, so its result depends on the sweep order (here: all cells at
    once).
  * hflow is capped at MaxHflow = 10 m in the momentum equation (not in the
    CFL step and not at FREE edges); g = 9.80655.

Time stepping on the device: the step, the CFL reduction and the clock are
tensors; the host reads the clock once per block of `block` steps (CUDA
graph of the whole block on a GPU). Steps after the end of the interval
have dt = 0 and change nothing.
"""
import math
import sys
import warnings

import torch

#: gravitational acceleration used by LISFLOOD-FP (lisflood.cpp)
G = 9.80655

SCHEMES = ('torchvic', 'lisflood_gpu', 'lisflood_cpu')
EDGES = ('N', 'E', 'S', 'W')

#: ledger slots (float64 on the device)
_RAIN, _POINT, _OUT, _CLAMP, _NSTEP, _DTMIN, _DTMAX, _CFLMIN = range(8)


class FloodEngine(object):
    """Local-inertial flood model on one structured grid.

    dem          [ny, nx] bed elevation (m), row 0 north
    dx           cell size (m), square cells
    manning      Manning n: a number or an [ny, nx] raster (faces use the
                 mean of their two cells, free boundaries the edge cell)
    free_edges   {edge: slope} for open edges ('N', 'E', 'S', 'W'); slope
                 is the floodplain slope of the normal-depth outflow or -1
                 for the local water-surface slope; other edges are closed.
                 A value may also be an array (one slope per edge cell, NaN
                 = closed there).
    inflow_points  [(row, col)] cells that receive point inflows (m3/s)
    nodata       DEM value of cells outside the domain (no rain there)
    track_edges  accumulate the outflow of every boundary face over an
                 interval (float64, m3; `run` returns it as `edge_out`):
                 the per-outlet return flow of two-way coupling. Off by
                 default; on or off, the state and ledger are the same.
    devices      opt-in multi-GPU step (flood/multigpu.py): two or more
                 CUDA devices of one node, e.g. ['cuda:0', 'cuda:1']; the
                 grid is cut into row strips with a halo exchange every
                 sub-step, the state, sources and ledger stay on the first
                 device, and every result equals the single-GPU float64
                 engine bit for bit. Default None: one device (`device`).
    """

    def __init__(self, dem, dx, manning=0.03, free_edges=None,
                 inflow_points=(), scheme='torchvic', cfl=0.7, max_dt=10.0,
                 depth_thresh=1e-3, max_hflow=10.0, theta=1.0, g=G,
                 drycheck=True, end='clip', nodata=-9999.0,
                 dtype=torch.float64,
                 device=None, block=64, compile=None, graphs=None,
                 backend='auto', triton_cfg=None, track_edges=False,
                 mixed=False, devices=None):
        if scheme not in SCHEMES:
            raise ValueError('scheme %r not in %s' % (scheme, SCHEMES))
        self.scheme = scheme
        devices = [str(torch.device(d)) for d in (devices or ())]
        if devices:
            device = devices[0]
        self.devices = devices if len(devices) > 1 else None
        self.device = torch.device(device if device is not None else
                                   ('cuda' if torch.cuda.is_available()
                                    else 'cpu'))
        self.dtype = dtype
        z = torch.as_tensor(dem, dtype=torch.float64)
        if z.dim() != 2 or min(z.shape) < 3:
            raise ValueError('dem must be 2-D with at least 3 x 3 cells')
        self.ny, self.nx = int(z.shape[0]), int(z.shape[1])
        self.dx = float(dx)
        self.area = self.dx * self.dx
        self.cfl, self.max_dt = float(cfl), float(max_dt)
        self.th, self.max_hflow = float(depth_thresh), float(max_hflow)
        self.theta, self.g, self.drycheck = float(theta), float(g), bool(drycheck)
        if end not in ('clip', 'overshoot'):
            raise ValueError("end must be 'clip' or 'overshoot'")
        # 'clip': the last step of an interval ends exactly at its end
        # (torchvic); 'overshoot': LISFLOOD-FP's `while t < T` (the compat
        # schemes always overshoot)
        self.clip_end = end == 'clip'
        self.track_edges = bool(track_edges)
        if self.track_edges and scheme != 'torchvic':
            raise ValueError("track_edges needs scheme 'torchvic'")
        dev, dt_ = self.device, dtype
        self.z = z.to(dev, dt_).contiguous()
        self.active = (torch.abs(z - nodata) >= 1e-6).to(dev)
        n = torch.as_tensor(manning, dtype=torch.float64)
        self.n_raster = n.dim() == 2
        if self.n_raster:
            if tuple(n.shape) != (self.ny, self.nx):
                raise ValueError('manning raster shape %s != dem %s'
                                 % (tuple(n.shape), (self.ny, self.nx)))
            n = n.to(dev, dt_)
            self.fnx = 0.5 * (n[:, :-1] + n[:, 1:])
            self.fny = 0.5 * (n[:-1, :] + n[1:, :])
            self.n_cell = n
        else:
            v = float(n)
            self.fnx = self.fny = self.n_cell = torch.tensor(v, dtype=dt_,
                                                             device=dev)
        self._setup_edges(free_edges or {})
        pts = [(int(r), int(c)) for r, c in inflow_points]
        for r, c in pts:
            if not (0 <= r < self.ny and 0 <= c < self.nx):
                raise ValueError('inflow point (%d, %d) outside the grid'
                                 % (r, c))
        self.points = pts
        self.pidx = torch.tensor([r * self.nx + c for r, c in pts],
                                 dtype=torch.long, device=dev)
        # Points sharing a cell are added in a fixed order (list order), one
        # collision-free index_add_ per rank: index_add_ on CUDA is an atomic
        # add, whose order (so the rounding of 3+ addends) varies run to run.
        seen, rank = {}, []
        for r, c in pts:
            k = r * self.nx + c
            rank.append(seen.get(k, 0))
            seen[k] = rank[-1] + 1
        self._prank = [torch.tensor([i for i, q in enumerate(rank) if q == m],
                                    dtype=torch.long, device=dev)
                       for m in range(max(rank, default=-1) + 1)]
        self.block = max(int(block), 1)
        cuda = dev.type == 'cuda'
        self.use_compile = cuda if compile is None else bool(compile)
        self.use_graphs = cuda if graphs is None else (bool(graphs) and cuda)
        if scheme == 'lisflood_gpu' and self.use_compile:
            fus = sys.modules.get('torchvic.core.fusion')
            if fus is not None and getattr(fus, '_ready', False):
                warnings.warn(
                    'torchvic.core.fusion.setup() already ran in this '
                    'process: inductor no longer contracts a*b+c into FMA, '
                    "so scheme 'lisflood_gpu' does not reproduce the "
                    'LISFLOOD-FP CUDA binary (depth differs ~1e-5 after an '
                    'hour); use a fresh process for that comparison',
                    RuntimeWarning, stacklevel=2)
        if backend == 'auto':
            backend = ('triton' if cuda and scheme == 'torchvic'
                       and self.theta == 1.0 else 'torch')
        if backend == 'triton' and not (cuda and scheme == 'torchvic'
                                        and self.theta == 1.0):
            raise ValueError("backend 'triton' needs CUDA, scheme "
                             "'torchvic' and theta = 1")
        if backend not in ('torch', 'triton'):
            raise ValueError('backend %r: torch or triton' % (backend,))
        self.backend = backend
        self.mixed = bool(mixed)
        if self.mixed and not (backend == 'triton'
                               and dtype == torch.float64):
            raise ValueError("mixed precision needs the triton backend "
                             "(CUDA, scheme 'torchvic', theta = 1) and "
                             "dtype float64")
        if self.devices and not (backend == 'triton' and not self.mixed
                                 and dtype == torch.float64):
            raise ValueError("devices (multi-GPU) needs the triton backend "
                             "(CUDA, scheme 'torchvic', theta = 1), dtype "
                             "float64 and mixed=False")
        self.triton_cfg = dict(triton_cfg or {})
        self._stepper = None
        self._graph = None
        self._step_fn = None
        self.reset()

    # ------------------------------------------------------------------
    # set-up
    # ------------------------------------------------------------------
    def _setup_edges(self, free_edges):
        """Per edge: a boolean mask of open faces and the slope per face."""
        dev, dt_ = self.device, self.dtype
        self.edges = {}
        for e in free_edges:
            if e not in EDGES:
                raise ValueError('edge %r not in N/E/S/W' % (e,))
        for e in EDGES:
            m = self.nx if e in 'NS' else self.ny
            s = free_edges.get(e)
            if s is None:
                continue
            s = torch.as_tensor(s, dtype=torch.float64)
            s = s.expand(m).clone() if s.dim() == 0 else s.reshape(m)
            mask = ~torch.isnan(s)
            if not bool(mask.any()):
                continue
            self.edges[e] = (mask.to(dev), torch.nan_to_num(s).to(dev, dt_))

    def reset(self, depth=None):
        """Dry (or `depth`) state, zero momentum, clock at 0."""
        dev, dt_ = self.device, self.dtype
        ny, nx = self.ny, self.nx
        h = torch.zeros(ny, nx, dtype=dt_, device=dev)
        if depth is not None:
            h.copy_(torch.as_tensor(depth, dtype=dt_).reshape(ny, nx))
        self.h = h
        # mixed precision: faces and momentum in float32 (triton_step)
        qd = torch.float32 if getattr(self, 'mixed', False) else dt_
        self.Qx = torch.zeros(ny, nx + 1, dtype=qd, device=dev)
        self.Qy = torch.zeros(ny + 1, nx, dtype=qd, device=dev)
        self.qx = torch.zeros_like(self.Qx)
        self.qy = torch.zeros_like(self.Qy)
        self.hmax = torch.zeros_like(h)          # interval maximum
        self.hmax_run = torch.zeros_like(h)      # whole-run maximum
        self.dtf = torch.zeros_like(h)           # lisflood_gpu dt field
        f64 = torch.float64
        self.dt_prev = torch.zeros((), dtype=dt_, device=dev)
        self.t_rel = torch.zeros((), dtype=f64, device=dev)
        self.T = torch.zeros((), dtype=f64, device=dev)
        self.rate = torch.zeros(ny, nx, dtype=dt_, device=dev)
        self.pq = torch.zeros(max(len(self.points), 1), dtype=dt_,
                              device=dev)
        self.R = torch.zeros((), dtype=f64, device=dev)   # sum(rate) * A
        self.P = torch.zeros((), dtype=f64, device=dev)   # sum(point Q)
        self.acc = torch.zeros(8, dtype=f64, device=dev)
        self.src = torch.zeros(ny, nx, dtype=dt_, device=dev)
        # outflow per boundary face over the interval (m3, float64): W rows,
        # E rows, N columns, S columns (only with track_edges)
        self.eacc = torch.zeros(2 * (ny + nx) if self.track_edges else 1,
                                dtype=f64, device=dev)
        self.t = 0.0
        self.steps = 0
        self.totals = dict(rain=0.0, inflow=0.0, outflow=0.0, clamp=0.0,
                           stored_start=self.storage(), steps=0)
        self._graph = None

    def storage(self):
        """Water stored on the grid (m3), summed in float64."""
        return float(self.h.to(torch.float64).sum()) * self.area

    # ------------------------------------------------------------------
    # one step (pure tensor code; compiled and graph-captured on a GPU)
    # ------------------------------------------------------------------
    def _face(self, h0, h1, z0, z1, q0, qo, fn, gdt, qup=None, qdn=None):
        """Discharge (m3/s) through faces between cells 0 and 1, the
        LISFLOOD-FP ACC expression (operation order kept)."""
        th = self.th
        qvect = torch.sqrt(q0 * q0 + qo * qo)
        y0 = z0 + h0
        y1 = z1 + h1
        dh = y0 - y1
        Sf = -dh / self.dx
        hflow = torch.maximum(y0, y1) - torch.maximum(z0, z1)
        hflow = torch.clamp(torch.clamp(hflow, min=0.0), max=self.max_hflow)
        a = gdt * hflow
        den = 1.0 + a * fn * fn * torch.abs(qvect) / torch.pow(
            hflow, 10.0 / 3.0)
        if self.theta == 1.0:
            Q = (q0 - a * Sf) / den * self.dx
        else:
            qc = self.theta * q0 + 0.5 * (1.0 - self.theta) * (qup + qdn)
            Q = (qc - a * Sf) / den * self.dx
            Qs = (q0 - a * Sf) / den * self.dx
            Q = torch.where(Q * dh < 0.0, Qs, Q)
        wet = ((h0 > th) | (h1 > th)) & (hflow > th)
        return torch.where(wet, Q, torch.zeros_like(Q))

    def _qupdn_x(self, qx):
        """Neighbour faces of the x faces for theta < 1 (LISFLOOD-FP
        indexing, including its end-of-row rule)."""
        qup = qx[:, 0:-2].clone()
        qdn = qx[:, 2:].clone()
        q0 = qx[:, 1:-1]
        qup[:, 0] = torch.where(qup[:, 0] == 0, q0[:, 0], qup[:, 0])
        qdn[:, -1] = torch.where(qdn[:, -1] == 0, q0[:, -1], qdn[:, -1])
        return qup, qdn

    def _qupdn_y(self, qy):
        qup = qy[0:-2, :].clone()
        qdn = qy[2:, :].clone()
        q0 = qy[1:-1, :]
        qup[0] = torch.where(qup[0] == 0, q0[0], qup[0])
        qdn[-1] = torch.where(qdn[-1] == 0, q0[-1], qdn[-1])
        return qup, qdn

    def _fluxes(self, h, qx, qy, dt):
        """Interior and boundary face discharges (m3/s) from h and the
        previous momentum."""
        z, gdt = self.z, self.g * dt
        qyb = (((qy[:-1, :-1] + qy[:-1, 1:]) + qy[1:, :-1]) + qy[1:, 1:]) / 4.0
        qxb = (((qx[:-1, :-1] + qx[:-1, 1:]) + qx[1:, :-1]) + qx[1:, 1:]) / 4.0
        if self.theta == 1.0:
            ux = uy = (None, None)
        else:
            ux, uy = self._qupdn_x(qx), self._qupdn_y(qy)
        Qxi = self._face(h[:, :-1], h[:, 1:], z[:, :-1], z[:, 1:],
                         qx[:, 1:-1], qyb, self.fnx, gdt, *ux)
        Qyi = self._face(h[:-1, :], h[1:, :], z[:-1, :], z[1:, :],
                         qy[1:-1, :], qxb, self.fny, gdt, *uy)
        zc = torch.zeros(self.ny, 1, dtype=h.dtype, device=h.device)
        zr = torch.zeros(1, self.nx, dtype=h.dtype, device=h.device)
        bw, be, bn, bs = zc[:, 0], zc[:, 0], zr[0], zr[0]
        if 'W' in self.edges:
            bw = self._free(self.edges['W'], h[:, 0], h[:, 1], z[:, 0],
                            z[:, 1], qx[:, 0], self._ncell(0, None), gdt, -1.0)
        if 'E' in self.edges:
            be = self._free(self.edges['E'], h[:, -1], h[:, -2], z[:, -1],
                            z[:, -2], qx[:, -1], self._ncell(-1, None), gdt,
                            1.0)
        if 'N' in self.edges:
            bn = self._free(self.edges['N'], h[0], h[1], z[0], z[1], qy[0],
                            self._ncell(None, 0), gdt, -1.0)
        if 'S' in self.edges:
            bs = self._free(self.edges['S'], h[-1], h[-2], z[-1], z[-2],
                            qy[-1], self._ncell(None, -1), gdt, 1.0)
        Qx = torch.cat([bw[:, None], Qxi, be[:, None]], dim=1)
        Qy = torch.cat([bn[None, :], Qyi, bs[None, :]], dim=0)
        return Qx, Qy

    def _ncell(self, col, row):
        if not self.n_raster:
            return self.n_cell
        return self.n_cell[:, col] if col is not None else self.n_cell[row]

    def _free(self, edge, h0, h1, z0, z1, q0, fn, gdt, sign):
        """Normal-depth outflow (m3/s) at an open edge (LISFLOOD-FP FREE)."""
        mask, slope = edge
        hflow = h0
        dh = ((z0 + h0) - z1) - h1
        Sf = torch.where(slope < -0.999, -dh / self.dx, slope)
        a = gdt * hflow
        q = sign * (torch.abs(q0) + torch.abs(a * Sf)) / (
            1.0 + a * fn * fn * torch.abs(q0) / torch.pow(hflow, 10.0 / 3.0))
        Q = q * self.dx
        on = mask & (h0 > self.th)
        return torch.where(on, Q, torch.zeros_like(Q))

    def _dt_cfl(self, h):
        """LISFLOOD-FP CPU rule: min(max_dt, cfl dx / sqrt(g max h))."""
        hm = torch.amax(h)
        loc = self.cfl * self.dx / torch.sqrt(self.g * torch.clamp(
            hm, min=self.th))
        return torch.where(hm > self.th, torch.clamp(loc, max=self.max_dt),
                           torch.full_like(hm, self.max_dt))

    def _divergence(self, Qx, Qy):
        return ((Qx[:, :-1] - Qx[:, 1:]) + Qy[:-1, :]) - Qy[1:, :]

    def _add_points(self, h, amount):
        self._scatter_points(h, amount)

    def _scatter_points(self, target, vals):
        """target[pidx] += vals, deterministic (duplicate cells in order)."""
        flat = target.view(-1)
        for ix in self._prank:
            flat.index_add_(0, self.pidx[ix], vals[ix])

    def _boundary_out(self, Qx, Qy):
        """Volume rate leaving through the edges (m3/s, >= 0 for outflow)."""
        return ((Qx[:, -1].sum() - Qx[:, 0].sum())
                + (Qy[-1].sum() - Qy[0].sum())).to(torch.float64)

    def _step(self):
        """Advance one step in place (all scalars stay on the device)."""
        if self.scheme == 'torchvic':
            self._step_torchvic()
        elif self.scheme == 'lisflood_gpu':
            self._step_lisflood_gpu()
        else:
            self._step_lisflood_cpu()

    def _clock(self, dt_raw, clip):
        """The step actually taken (float64 clock) and its working-precision
        value. clip=True ends exactly at T; False is LISFLOOD-FP's overshoot
        (`while t < T`)."""
        rem = self.T - self.t_rel
        d64 = dt_raw.to(torch.float64)
        a = self.acc
        a[_CFLMIN] = torch.where(rem > 0, torch.minimum(a[_CFLMIN], d64),
                                 a[_CFLMIN])
        if clip:
            d64 = torch.where(rem > 0, torch.minimum(d64, rem),
                              torch.zeros_like(d64))
        else:
            d64 = torch.where(rem > 0, d64, torch.zeros_like(d64))
        # the step the working precision can represent is the step taken
        d64 = d64.to(self.dtype).to(torch.float64)
        return d64, d64.to(self.dtype)

    def _ledger(self, d64, out_rate, clamp, rain=None):
        a = self.acc
        a[_RAIN] += self.R * d64 if rain is None else rain
        a[_POINT] += self.P * d64
        a[_OUT] += out_rate * d64
        a[_CLAMP] += clamp
        on = d64 > 0
        a[_NSTEP] += on.to(torch.float64)
        a[_DTMIN] = torch.where(on, torch.minimum(a[_DTMIN], d64), a[_DTMIN])
        a[_DTMAX] = torch.where(on, torch.maximum(a[_DTMAX], d64), a[_DTMAX])
        self.t_rel += d64

    def _track_max(self, h):
        wet = h > self.th
        self.hmax.copy_(torch.where(wet, torch.maximum(self.hmax, h),
                                    self.hmax))

    def _step_torchvic(self):
        h, A, dx = self.h, self.area, self.dx
        d64, dt = self._clock(self._dt_cfl(h), clip=self.clip_end)
        Qx, Qy = self._fluxes(h, self.qx, self.qy, dt)
        if self.drycheck:
            # a cell gives at most the water it holds at the step start:
            # every outgoing face is scaled by its donor's factor
            # min(1, h A / outgoing volume); conservative and order-free
            out = (((torch.clamp(-Qx[:, :-1], min=0.0)
                     + torch.clamp(Qx[:, 1:], min=0.0))
                    + torch.clamp(-Qy[:-1, :], min=0.0))
                   + torch.clamp(Qy[1:, :], min=0.0)) * dt
            cv = h * A
            lim = out > cv
            w = torch.where(lim, cv / torch.where(lim, out,
                                                  torch.ones_like(out)),
                            torch.ones_like(out))
            wl = torch.cat([w[:, :1], w], dim=1)      # cell west of face
            wr = torch.cat([w, w[:, -1:]], dim=1)     # cell east of face
            Qx = torch.where(Qx > 0, Qx * wl, Qx * wr)
            wu = torch.cat([w[:1], w], dim=0)         # cell north of face
            wd = torch.cat([w, w[-1:]], dim=0)        # cell south of face
            Qy = torch.where(Qy > 0, Qy * wu, Qy * wd)
        # sources of the step, then continuity (the LISFLOOD-FP CPU order
        # and arithmetic, so the two agree bit for bit where the limiter
        # does not act)
        # (src = rain rate + point inflow / A, m/s)
        hs = h + self.src * dt
        hn = hs + dt * self._divergence(Qx, Qy) / A
        neg = torch.clamp(-hn, min=0.0)
        clamp = neg.to(torch.float64).sum() * A
        hn = hn + neg
        live = d64 > 0                     # steps after the end: no-ops
        self.h.copy_(torch.where(live, hn, h))
        self.Qx.copy_(torch.where(live, Qx, self.Qx))
        self.Qy.copy_(torch.where(live, Qy, self.Qy))
        self.qx.copy_(torch.where(live, Qx * (1.0 / dx), self.qx))
        self.qy.copy_(torch.where(live, Qy * (1.0 / dx), self.qy))
        self._track_max(self.h)
        self._ledger(d64, self._boundary_out(Qx, Qy), clamp)
        if self.track_edges:
            f64 = torch.float64
            self.eacc += torch.cat([-Qx[:, 0], Qx[:, -1], -Qy[0],
                                    Qy[-1]]).to(f64) * d64

    def edge_index(self, edge, i0=None, i1=None):
        """Positions in `edge_out` of the faces of `edge` ('W', 'E': rows
        i0..i1-1; 'N', 'S': columns), all faces of the edge by default."""
        ny, nx = self.ny, self.nx
        off = dict(W=0, E=ny, N=2 * ny, S=2 * ny + nx)[edge]
        m = ny if edge in 'WE' else nx
        i0 = 0 if i0 is None else int(i0)
        i1 = m if i1 is None else int(i1)
        if not 0 <= i0 < i1 <= m:
            raise ValueError('edge %s faces %d..%d outside 0..%d'
                             % (edge, i0, i1, m))
        return torch.arange(off + i0, off + i1, device=self.device)

    def _step_lisflood_gpu(self):
        h, dx = self.h, self.dx
        th = self.th
        # dynamic rain of the step start, with the previous step's dt
        live = (self.T - self.t_rel) > 0
        rain_dt = torch.where(live, self.dt_prev, torch.zeros_like(
            self.dt_prev))
        h0 = h + self.rate * rain_dt
        rain_vol = self.R * rain_dt.to(torch.float64)
        # per-cell running-minimum step (update_dt_per_element)
        dloc = self.cfl * dx / torch.sqrt(self.g * h0)
        first = torch.abs(self.dtf - 0.0) < 1e-6
        mx = torch.full_like(h0, self.max_dt)
        dtf = torch.where(h0 > th, torch.where(first, mx, torch.minimum(
            dloc, self.dtf)), mx)
        d64, dt = self._clock(torch.amin(dtf), clip=False)
        Qx, Qy = self._fluxes(h0, self.qx, self.qy, dt)
        hp = h0.clone()
        # point sources: PS value = Q / dx (per unit width), H += PS dt / dx
        if len(self.points):
            self._add_points(hp, (self.pq[:len(self.points)] / dx) * dt / dx)
        qx0, qx1 = Qx[:, :-1], Qx[:, 1:]
        qy0, qy1 = Qy[:-1, :], Qy[1:, :]
        dV = dt * (((qx0 - qx1) + qy0) - qy1)
        if self.drycheck:
            cv = hp * (dx * dx)
            dry = (cv + dV) < 0.0
            W = -cv / torch.where(dry, dV, torch.ones_like(dV))
            qx0 = torch.where(dry & (qx0 < 0), qx0 * W, qx0)
            qx1 = torch.where(dry & (qx1 > 0), qx1 * W, qx1)
            qy0 = torch.where(dry & (qy0 < 0), qy0 * W, qy0)
            qy1 = torch.where(dry & (qy1 > 0), qy1 * W, qy1)
            dV = dt * (((qx0 - qx1) + qy0) - qy1)
        hn = hp + dV / (dx * dx)
        neg = torch.clamp(-hn, min=0.0)
        clamp = neg.to(torch.float64).sum() * self.area
        hn = hn + neg
        live_t = d64 > 0
        self.h.copy_(torch.where(live_t, hn, h0))
        self.Qx.copy_(torch.where(live_t, Qx, self.Qx))
        self.Qy.copy_(torch.where(live_t, Qy, self.Qy))
        self.qx.copy_(torch.where(live_t, Qx * (1.0 / dx), self.qx))
        self.qy.copy_(torch.where(live_t, Qy * (1.0 / dx), self.qy))
        self.dtf.copy_(torch.where(live_t, dtf, self.dtf))
        self.dt_prev.copy_(torch.where(live_t, dt, self.dt_prev))
        self._track_max(self.h)
        self._ledger(d64, self._boundary_out(Qx, Qy), clamp, rain=rain_vol)

    def _step_lisflood_cpu(self):
        h, dx, A = self.h, self.dx, self.area
        d64, dt = self._clock(self._dt_cfl(h), clip=False)
        Qx, Qy = self._fluxes(h, self.qx, self.qy, dt)
        if self.drycheck:
            # DryCheck (util.cpp) on cells deeper than DepthThresh; the
            # scaled faces are written back (here all cells at once)
            q1, q2 = Qx[:, :-1], Qx[:, 1:]
            q3, q4 = Qy[:-1, :], Qy[1:, :]
            dV = dt * (((q1 - q2) + q3) - q4)
            cv = h * A
            dry = (h > self.th) & ((cv + dV) < 0)
            W = -cv / torch.where(dry, dV, torch.ones_like(dV))
            one = torch.ones_like(W)
            wq1 = torch.where(dry & (q1 < 0), W, one)   # west face, out
            wq2 = torch.where(dry & (q2 > 0), W, one)   # east face, out
            wq3 = torch.where(dry & (q3 < 0), W, one)
            wq4 = torch.where(dry & (q4 > 0), W, one)
            fx = torch.cat([wq1[:, :1], wq1[:, 1:] * wq2[:, :-1],
                            wq2[:, -1:]], dim=1)
            fy = torch.cat([wq3[:1], wq3[1:] * wq4[:-1], wq4[-1:]], dim=0)
            Qx, Qy = Qx * fx, Qy * fy
        hs = h + self.rate * dt
        if len(self.points):
            hs = hs.clone()
            self._add_points(hs, (self.pq[:len(self.points)] / dx) * dx
                             * dt / A)
        hn = hs + dt * self._divergence(Qx, Qy) / A
        neg = torch.clamp(-hn, min=0.0)
        clamp = neg.to(torch.float64).sum() * A
        hn = hn + neg
        live = d64 > 0                     # steps after the end: no-ops
        self.h.copy_(torch.where(live, hn, h))
        self.Qx.copy_(torch.where(live, Qx, self.Qx))
        self.Qy.copy_(torch.where(live, Qy, self.Qy))
        self.qx.copy_(torch.where(live, Qx * (1.0 / dx), self.qx))
        self.qy.copy_(torch.where(live, Qy * (1.0 / dx), self.qy))
        self._track_max(self.h)
        self._ledger(d64, self._boundary_out(Qx, Qy), clamp)

    # ------------------------------------------------------------------
    # block execution
    # ------------------------------------------------------------------
    def _block(self):
        for _ in range(self.block):
            self._step_fn()
        if self._stepper is not None:
            self._stepper.sync()

    def _prepare(self):
        if self._step_fn is not None:
            return
        if self.devices:
            from .multigpu import MultiGPU
            self._stepper = MultiGPU(self, self.devices, **self.triton_cfg)
            self._step_fn = self._stepper.step
            return
        if self.backend == 'triton':
            from .triton_step import TritonStepper
            self._stepper = TritonStepper(self, **self.triton_cfg)
            self._step_fn = self._stepper.step
            return
        fn = self._step
        if self.use_compile:
            fn = torch.compile(fn, dynamic=False, fullgraph=False)
        self._step_fn = fn

    def _run_block(self):
        if self.devices:
            self._stepper.run_block(self.block, self.use_graphs)
            return
        if not self.use_graphs:
            self._block()
            return
        if self._graph is None:
            # warm up (compile) on a side stream, then capture one block
            # with the state restored, so capture does not advance it
            snap = [x.clone() for x in self._mutable()]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._step_fn()
            torch.cuda.current_stream().wait_stream(s)
            if self._stepper is not None:
                self._stepper.flip = 0
                self._stepper.qflip = 0
            for x, y in zip(self._mutable(), snap):
                x.copy_(y)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._block()
            for x, y in zip(self._mutable(), snap):
                x.copy_(y)
            self._graph = g
        self._graph.replay()

    def _mutable(self):
        m = [self.h, self.Qx, self.Qy, self.qx, self.qy, self.hmax,
             self.dtf, self.dt_prev, self.t_rel, self.acc]
        if self.track_edges:
            m.append(self.eacc)
        st = self._stepper
        if st is not None:
            m += st.mutable()
        return m

    # ------------------------------------------------------------------
    # one coupling interval
    # ------------------------------------------------------------------
    def run(self, T, runoff=None, inflow=None, rate=None, discharge=None,
            check=True):
        """Advance T seconds.

        runoff     m3 per cell over the interval ([ny*nx] or [ny, nx]),
                   added uniformly in time (rate = V / (A T))
        inflow     m3 per inflow point over the interval (rate = V / T)
        rate       alternative to runoff: m/s per cell
        discharge  alternative to inflow: m3/s per point

        check=False skips the finite/non-negative test of the inputs (two
        host syncs); the caller then owns that check (the coupling connector
        raises it as a device flag at validate).

        Returns a dict: depth, max_depth (interval), speed (cell centre,
        m/s, from the last face discharges), outflow (m3) and the ledger.
        """
        T = float(T)
        if not T > 0:
            raise ValueError('T must be > 0')
        dev, dt_ = self.device, self.dtype
        ny, nx, A = self.ny, self.nx, self.area
        if runoff is not None:
            v = torch.as_tensor(runoff, dtype=torch.float64).reshape(ny, nx)
            r = v / (A * T)
        elif rate is not None:
            r = torch.as_tensor(rate, dtype=torch.float64).reshape(ny, nx)
        else:
            r = torch.zeros(ny, nx, dtype=torch.float64)
        r = r.to(dev)
        if check and (not bool(torch.isfinite(r).all())
                      or bool((r < 0).any())):
            raise ValueError('runoff must be finite and >= 0')
        r = torch.where(self.active, r, torch.zeros_like(r))
        np_ = len(self.points)
        if inflow is not None:
            q = torch.as_tensor(inflow, dtype=torch.float64).reshape(np_) / T
        elif discharge is not None:
            q = torch.as_tensor(discharge, dtype=torch.float64).reshape(np_)
        else:
            q = torch.zeros(np_, dtype=torch.float64)
        q = q.to(dev)
        if check and np_ and (not bool(torch.isfinite(q).all())
                              or bool((q < 0).any())):
            raise ValueError('inflow must be finite and >= 0')
        self.rate.copy_(r.to(dt_))
        if np_:
            self.pq[:np_].copy_(q.to(dt_))
        # ledger rates from the working-precision values actually used
        self.R.copy_(self.rate.to(torch.float64).sum() * A)
        if np_:
            self.P.copy_(self.pq[:np_].to(torch.float64).sum())
        else:
            self.P.zero_()
        S0t = self.h.to(torch.float64).sum() * self.area   # read with acc
        self.T.fill_(T)
        self.t_rel.zero_()
        if getattr(self, '_acc0', None) is None:
            a0 = torch.zeros_like(self.acc)
            a0[_DTMIN] = math.inf
            a0[_CFLMIN] = math.inf
            self._acc0 = a0
        self.acc.copy_(self._acc0)          # device-to-device, no scalar H2D
        if self.track_edges:
            self.eacc.zero_()
        src = self.rate.clone()
        if np_:
            self._scatter_points(src, self.pq[:np_] / A)
        self.src.copy_(src)
        self.hmax.copy_(self.h)
        self._prepare()
        if self._stepper is not None:
            self._stepper.begin()
        # Steps after the interval end are exact no-ops (the clock clip /
        # `live` masks of every scheme), so blocks may be launched ahead of
        # the host's look at t_rel without changing any result: the first
        # look comes after as many blocks as the previous interval needed
        # less one, then one block at a time. A steady run polls ~2 times
        # per interval instead of once per block.
        nblocks = 0
        ahead = max(getattr(self, '_last_blocks', 1) - 1, 1)
        while True:
            for _ in range(ahead):
                self._run_block()
                nblocks += 1
            ahead = 1
            if float(self.t_rel) >= T:
                break
        self._last_blocks = nblocks
        if self._stepper is not None:
            self._stepper.finish()
        S1t = self.h.to(torch.float64).sum() * self.area
        # the interval's one bulk read: accumulators and both storages
        parts = [self.acc.to(torch.float64), S0t.reshape(1), S1t.reshape(1)]
        if self.track_edges:
            parts.append(self.eacc.sum().reshape(1))
        a = torch.cat(parts).tolist()
        edge_total = a.pop() if self.track_edges else None
        S0, S1 = a[-2], a[-1]
        a = a[:-2]
        rain, point, out, clamp = a[_RAIN], a[_POINT], a[_OUT], a[_CLAMP]
        resid = (S1 - S0) - (rain + point - out + clamp)
        scale = max(rain + point + abs(out) + S0 + S1, 1e-30)
        t_end = float(self.t_rel)
        self.t += t_end
        self.steps += int(a[_NSTEP])
        tt = self.totals
        tt['rain'] += rain
        tt['inflow'] += point
        tt['outflow'] += out
        tt['clamp'] += clamp
        tt['steps'] += int(a[_NSTEP])
        self.hmax_run.copy_(torch.maximum(self.hmax_run, self.hmax))
        ledger = dict(rain=rain, inflow=point, outflow=out, clamp=clamp,
                      stored0=S0, stored1=S1, residual=resid,
                      rel_residual=resid / scale, steps=int(a[_NSTEP]),
                      dt_min=a[_DTMIN], dt_max=a[_DTMAX],
                      dt_cfl_min=a[_CFLMIN], t_end=t_end,
                      blocks=nblocks)
        res = dict(depth=self.h.reshape(-1), max_depth=self.hmax.reshape(-1),
                   speed=self.speed().reshape(-1), outflow=out,
                   ledger=ledger)
        if self.track_edges:
            # per-face outflow (device, float64) and its sum (host)
            res['edge_out'] = self.eacc.clone()
            ledger['edge_outflow'] = edge_total
        return res

    # ------------------------------------------------------------------
    # outputs and state
    # ------------------------------------------------------------------
    def face_velocity(self):
        """Face velocities (m/s) q / hflow of the current state (as
        LISFLOOD-FP's .Vx/.Vy; q is the momentum kept for the next step,
        hflow from the current depth; edges use the edge cell's depth)."""
        z, h = self.z, self.h
        y = z + h
        hx = torch.clamp(torch.maximum(y[:, :-1], y[:, 1:])
                         - torch.maximum(z[:, :-1], z[:, 1:]), min=0.0,
                         max=self.max_hflow)
        hy = torch.clamp(torch.maximum(y[:-1], y[1:])
                         - torch.maximum(z[:-1], z[1:]), min=0.0,
                         max=self.max_hflow)
        hx = torch.cat([h[:, :1], hx, h[:, -1:]], dim=1)
        hy = torch.cat([h[:1], hy, h[-1:]], dim=0)
        ok_x = (self.qx != 0) & (hx > 0)
        ok_y = (self.qy != 0) & (hy > 0)
        vx = torch.where(ok_x, self.qx / torch.where(ok_x, hx, 1.0), 0.0)
        vy = torch.where(ok_y, self.qy / torch.where(ok_y, hy, 1.0), 0.0)
        return vx, vy

    def speed(self):
        """Cell-centre speed (m/s) from the face velocities."""
        vx, vy = self.face_velocity()
        u = 0.5 * (vx[:, :-1] + vx[:, 1:])
        v = 0.5 * (vy[:-1] + vy[1:])
        return torch.hypot(u, v)

    def state_dict(self):
        """Everything needed to continue the run bit-identically."""
        return dict(h=self.h.clone(), Qx=self.Qx.clone(), Qy=self.Qy.clone(),
                    qx=self.qx.clone(), qy=self.qy.clone(),
                    dtf=self.dtf.clone(), dt_prev=self.dt_prev.clone(),
                    hmax_run=self.hmax_run.clone(), t=self.t,
                    steps=self.steps, totals=dict(self.totals),
                    scheme=self.scheme)

    def load_state_dict(self, s):
        if s.get('scheme', self.scheme) != self.scheme:
            raise ValueError('state of scheme %r, engine is %r'
                             % (s['scheme'], self.scheme))
        for k in ('h', 'Qx', 'Qy', 'qx', 'qy', 'dtf', 'dt_prev', 'hmax_run'):
            x = getattr(self, k)
            x.copy_(s[k].to(self.device, x.dtype))
        self.t, self.steps = float(s['t']), int(s['steps'])
        self.totals = dict(s['totals'])
