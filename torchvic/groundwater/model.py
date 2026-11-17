"""GPU groundwater flow: the MODFLOW 6 GWF equations on a structured grid.

Formulation (one MODFLOW 6 option set, matched term by term; source
references are to MODFLOW 6.7.0, ``src/Model/...``):

* NPF standard formulation (no NEWTON), harmonic-mean conductance
  (CCOND_HMEAN), ``icelltype`` 0 (confined) or 1 (convertible). Horizontal
  conductance ``condmean`` with transmissivity k11*sat*thickness
  (GwfConductanceUtils.f90 hcond/convertible_standard/condmean; saturation
  ``sgwf_npf_thksat`` in gwf-npf.f90). Vertical conductance constant
  (default) or VARIABLECV without DEWATERED (``vcond``). PERCHED and
  rewetting are not supported; a convertible cell whose head falls to its
  bottom goes dry for the rest of the run as in ``sgwf_npf_wetdry``
  (checked at every outer iteration).
* STO: specific storage (default formulation, iorig_ss=0, iconf_ss=0, or
  STORAGECOEFFICIENT) and specific yield for convertible cells
  (GwfStorageUtils.f90 SsTerms/SyTerms, saturation with satomega 0).
* RCH (array, highest active cell, ``rch_cf``), WEL with optional
  AUTO_FLOW_REDUCE (``wel_cf``, sQSaturation), RIV (``riv_cf``), DRN without
  DDRN (``drn_cf``), GHB (``ghb_cf``), CHD (identity rows; budget as
  ``calc_chd_rate``).
* Backward Euler; Picard outer iterations with IMS-style closure
  (max head change <= outer_dvclose); inner PCG (solver.py).

All arrays are torch tensors (L, R, C) on one device, float64; the only
host reads are one per outer iteration and one per PCG chunk.

Sign convention as MODFLOW 6: every package gives ``hcof``/``rhs`` with
flow into the aquifer ``q = hcof*h - rhs``; budgets report rates (L3/T)
into (``in``) and out of (``out``) the aquifer per package.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .grid import Grid
from . import solver as sv

HDRY = -1.0e30


class GroundwaterNotConverged(RuntimeError):
    pass


@dataclass
class StepResult:
    converged: bool
    outer_iterations: int
    inner_iterations: int
    dt: float
    #: {package: (rate_in, rate_out)} as 0-d device tensors, L3/T
    budget: dict = field(default_factory=dict)
    #: per-entry flows into the aquifer for list packages, L3/T
    flows: dict = field(default_factory=dict)

    def rates(self):
        """Budget as python floats {package: (in, out)} (one host sync)."""
        keys = list(self.budget)
        if not keys:
            return {}
        v = torch.stack([torch.stack(self.budget[k]) for k in keys]).cpu()
        return {k: (float(v[i, 0]), float(v[i, 1]))
                for i, k in enumerate(keys)}

    def volumes(self):
        return {k: (a * self.dt, b * self.dt)
                for k, (a, b) in self.rates().items()}

    def discrepancy(self):
        """(total in - total out, percent discrepancy) as MODFLOW 6."""
        r = self.rates()
        tin = sum(a for a, _ in r.values())
        tout = sum(b for _, b in r.values())
        avg = 0.5 * (tin + tout)
        return tin - tout, (100.0 * (tin - tout) / avg if avg else 0.0)


class _Scatter:
    """Deterministic scatter-add of list entries into cells: entries are
    split into groups with unique cells (k-th occurrence), so no atomics."""

    def __init__(self, nodes, device):
        nodes = np.asarray(nodes, np.int64).ravel()
        self.n = nodes.size
        order = np.argsort(nodes, kind='stable')
        s = nodes[order]
        first = np.r_[True, s[1:] != s[:-1]] if s.size else np.zeros(0, bool)
        start = np.maximum.accumulate(np.where(first, np.arange(s.size), 0))
        rank = np.empty(s.size, np.int64)
        rank[order] = np.arange(s.size) - start
        self.nodes = torch.as_tensor(nodes, device=device)
        self.groups = []
        for g in range(int(rank.max()) + 1 if rank.size else 0):
            sel = np.flatnonzero(rank == g)
            self.groups.append((torch.as_tensor(sel, device=device),
                                torch.as_tensor(nodes[sel], device=device)))
        self.unique = len(self.groups) <= 1

    def add(self, dst_flat, vals):
        if self.unique:
            if self.n:
                dst_flat[self.nodes] = dst_flat[self.nodes] + vals
            return dst_flat
        for sel, nd in self.groups:
            dst_flat[nd] = dst_flat[nd] + vals[sel]
        return dst_flat


class GroundwaterModel:
    """MODFLOW 6 GWF (standard formulation) on a structured grid.

    Parameters are scalars or arrays broadcastable to (nlay, nrow, ncol);
    ``icelltype`` and ``iconvert`` may be per layer (nlay,).
    """

    def __init__(self, grid: Grid, k11, k33=None, icelltype=0, ss=0.0,
                 sy=0.0, iconvert=None, strt=None, storagecoefficient=False,
                 variablecv=False, dewatered=False, device='cpu', dtype=torch.float64,
                 outer_dvclose=1e-6, outer_maximum=100, inner_dvclose=1e-8,
                 inner_rclose=1e-3, inner_maximum=2000, precond='jacobi',
                 inner_per_outer=None, inner_per_outer_steady=50,
                 outer_per_check=None, cuda_graph=True,
                 compile=None, hdry=HDRY):
        self.grid = grid
        self.device = torch.device(device)
        self.dtype = dtype
        L, R, C = grid.shape
        self.shape = (L, R, C)
        self.N = L * R * C

        def arr(v, dt=dtype):
            a = np.asarray(v, float)
            if a.ndim == 1 and a.size == L and L > 1:
                a = a[:, None, None]
            elif a.ndim == 1 and a.size == L:
                a = a[:, None, None]
            a = np.broadcast_to(a, self.shape).copy()
            return torch.as_tensor(a, device=self.device, dtype=dt)

        tops = grid.cell_tops()
        self.top = arr(tops)
        self.bot = arr(grid.botm)
        self.thk = self.top - self.bot
        self.area = arr(np.broadcast_to(grid.area, self.shape))
        self.k11 = arr(k11)
        self.k33 = arr(self.k11.cpu().numpy() if k33 is None else k33)
        self.conv = arr(icelltype) != 0
        self.iconv = (self.conv.clone() if iconvert is None
                      else arr(iconvert) != 0)
        self.ss = arr(ss)
        self.sy = arr(sy)
        self.storagecoefficient = bool(storagecoefficient)
        self.variablecv = bool(variablecv)
        self.dewatered = bool(dewatered)
        self.hdry = float(hdry)
        t = lambda a: torch.as_tensor(np.asarray(a, float), device=self.device,
                                      dtype=dtype)
        self.clx1, self.clx2, self.wx = t(grid.clx1), t(grid.clx2), t(grid.wx)
        self.cly1, self.cly2, self.wy = t(grid.cly1), t(grid.cly2), t(grid.wy)
        self.idomain = torch.as_tensor(grid.idomain > 0, device=self.device)
        # saturated vertical conductance (calc_condsat -> vcond, sat = 1)
        if L > 1:
            b1 = (1.0 * self.thk[:-1]) * 0.5 / self.k33[:-1]
            b2 = (1.0 * self.thk[1:]) * 0.5 / self.k33[1:]
            self.condsat_z = self.area[:-1] / (b1 + b2)
        else:
            self.condsat_z = torch.zeros((0, R, C), device=self.device,
                                         dtype=dtype)
        # solver settings (IMS equivalents)
        self.outer_dvclose = float(outer_dvclose)
        self.outer_maximum = int(outer_maximum)
        self.inner_dvclose = float(inner_dvclose)
        self.inner_rclose = float(inner_rclose)
        self.inner_maximum = int(inner_maximum)
        cuda = self.device.type == 'cuda'
        # each outer iteration runs inner_per_outer PCG iterations (the
        # closure is tested on the last; frozen once closed); an outer
        # iteration whose inner solve did not close is never accepted, so
        # this only trades outer for inner iterations. Steady steps (no
        # storage, stiff) use inner_per_outer_steady. outer_per_check outer
        # iterations run per host read (masked no-ops after convergence).
        # CUDA defaults (3, 4) are tuned on the ISC daily step.
        self.k_in = int(min(self.inner_maximum, inner_per_outer or
                            (3 if cuda else 50)))
        self.k_in_steady = int(min(self.inner_maximum,
                                   inner_per_outer_steady))
        self.k_out = int(outer_per_check or (4 if cuda else 1))
        self.precond = precond
        self.use_graph = bool(cuda_graph) and cuda
        self.compile = cuda if compile is None else bool(compile)
        z = lambda *sh: torch.zeros(sh, device=self.device, dtype=dtype)
        S = {k: z(L, R, C) for k in ('h', 'hold', 'x', 'r', 'p', 'diag',
                                    'minv', 'b')}
        S['c'] = tuple(z(L, R, C) for _ in range(6))
        S['ib'] = torch.zeros((L, R, C), device=self.device, dtype=torch.int8)
        for k in ('rz', 'dh'):
            S[k] = z()
        for k in ('done', 'conv', 'anydry'):
            S[k] = torch.zeros((), device=self.device, dtype=torch.bool)
        for k in ('it', 'nout', 'nin'):
            S[k] = torch.zeros((), device=self.device, dtype=torch.int64)
        kk, ii, jj = np.meshgrid(np.arange(L), np.arange(R), np.arange(C),
                                 indexing='ij')
        S['red'] = torch.as_tensor((kk + ii + jj) % 2 == 0, device=self.device)
        self._S = S
        self._fns = None
        self._graphs = {}
        # state
        self.ibound = torch.where(self.idomain, 1, 0).to(torch.int8)
        h0 = self.top.clone() if strt is None else arr(strt)
        self.h = torch.where(self.idomain, h0, torch.full_like(h0, self.hdry))
        self.time = 0.0
        self.kstp = 0
        # packages
        self.rch = None               # (R, C) rate L/T
        self.lists = {}               # name -> dict
        self.chd = None
        self.evt = None               # EVT arrays per column

    # ---------------------------------------------------------- inputs
    def _nodes(self, cells):
        if isinstance(cells, (tuple, list)) and len(cells) == 3:
            k, i, j = (np.asarray(c, np.int64) for c in cells)
            return self.grid.node(k, i, j).astype(np.int64)
        return np.asarray(cells, np.int64).ravel()

    def _vec(self, v, n):
        if not torch.is_tensor(v):
            v = torch.as_tensor(np.asarray(v, float))
        return v.to(self.device, self.dtype).reshape(-1).expand(n)

    def _invalidate(self):
        """Drop compiled functions and captured graphs (structure changed)."""
        self._fns = None
        self._graphs = {}

    def _put(self, d, k, v, n):
        """Store values in a static buffer (in place when the shape holds,
        so a captured graph keeps reading the right memory)."""
        v = self._vec(v, n)
        if k in d and d[k].shape == v.shape:
            d[k].copy_(v)
        else:
            d[k] = v.contiguous().clone()
            self._invalidate()

    def _set_list(self, name, cells, **vals):
        pk = self.lists.get(name)
        if cells is not None:
            nodes = self._nodes(cells)
            pk = {'scatter': _Scatter(nodes, self.device),
                  'nodes': torch.as_tensor(nodes, device=self.device)}
            self.lists[name] = pk
            self._invalidate()
        if pk is None:
            raise ValueError(f'{name}: cells needed on first call')
        n = pk['nodes'].numel()
        for k, v in vals.items():
            if v is not None:
                self._put(pk, k, v, n)
        return pk

    def set_recharge(self, rate):
        """RCH array: rate (L/T) per column (nrow, ncol) or scalar; applied
        to the highest active cell (RCHA default, fixed_cell off)."""
        if rate is None:
            if self.rch is not None:
                self._invalidate()
            self.rch = None
            return
        r = rate if torch.is_tensor(rate) else torch.as_tensor(
            np.asarray(rate, float))
        r = r.to(self.device, self.dtype).expand(self.shape[1:])
        if self.rch is None:
            self.rch = r.contiguous().clone()
            self._invalidate()
        else:
            self.rch.copy_(r)

    def set_wells(self, cells=None, q=None, auto_flow_reduce=None):
        """WEL: q (L3/T, negative = extraction). ``auto_flow_reduce`` is the
        MODFLOW 6 AUTO_FLOW_REDUCE fraction (None = off)."""
        pk = self._set_list('WEL', cells, q=q)
        if cells is not None or auto_flow_reduce is not None:
            self._invalidate()
            pk['afr'] = auto_flow_reduce if auto_flow_reduce is None else (
                0.1 if auto_flow_reduce <= 0 else float(auto_flow_reduce))

    def set_rivers(self, cells=None, stage=None, cond=None, rbot=None):
        self._set_list('RIV', cells, stage=stage, cond=cond, rbot=rbot)

    def set_drains(self, cells=None, elev=None, cond=None):
        self._set_list('DRN', cells, elev=elev, cond=cond)

    def set_ghb(self, cells=None, bhead=None, cond=None):
        self._set_list('GHB', cells, bhead=bhead, cond=cond)

    def set_chd(self, cells=None, head=None):
        """CHD: replaces the constant-head set (previous CHD cells become
        variable-head again, as a new MODFLOW 6 stress period)."""
        flat_ib = self.ibound.view(-1)
        if cells is not None:
            if self.chd is not None:
                old = self.chd['nodes']
                flat_ib[old] = torch.where(flat_ib[old] < 0,
                                           torch.ones_like(flat_ib[old]),
                                           flat_ib[old])
            nodes = torch.as_tensor(self._nodes(cells), device=self.device)
            self.chd = {'nodes': nodes}
            self._invalidate()
        if self.chd is None:
            raise ValueError('CHD: cells needed on first call')
        nodes = self.chd['nodes']
        if head is not None:
            self._put(self.chd, 'head', head, nodes.numel())
        flat_ib[nodes] = -1
        self.h.view(-1)[nodes] = self.chd['head']

    def set_evt(self, surface=None, rate=None, depth=None, pxdp=None,
                petm=None, petm0=None):
        """EVT as array input (EVTA; highest active cell, fixed_cell off):
        per-column ``surface``, maximum ``rate`` (L/T) and extinction
        ``depth``; optional segments ``pxdp``/``petm`` (nseg-1, nrow, ncol)
        and ``petm0`` (SURF_RATE_SPECIFIED), as MODFLOW 6 ``evt_cf``."""
        R, C = self.shape[1:]
        if self.evt is None:
            self.evt = {}
            self._invalidate()
        for k, v in (('surface', surface), ('rate', rate), ('depth', depth),
                     ('pxdp', pxdp), ('petm', petm), ('petm0', petm0)):
            if v is None:
                continue
            t = v if torch.is_tensor(v) else torch.as_tensor(
                np.array(v, float))
            t = t.to(self.device, self.dtype)
            t = t.expand((t.shape[0], R, C) if k in ('pxdp', 'petm')
                         and t.dim() == 3 else (R, C))
            if k in self.evt and self.evt[k].shape == t.shape:
                self.evt[k].copy_(t)
            else:
                self.evt[k] = t.contiguous().clone()
                self._invalidate()

    # ------------------------------------------------------------ core
    def _sat(self, h, wet):
        s = torch.where(h >= self.top, torch.ones_like(h),
                        (h - self.bot) / self.thk)
        s = torch.where(self.conv, s, torch.ones_like(h))
        return torch.where(wet, s, torch.zeros_like(h))

    @staticmethod
    def _qsat(h, top, bot):
        """sQuadraticSaturation with eps = 0 (linear, clipped)."""
        b = top - bot
        br = torch.clamp((h - bot) / b, 0.0, 1.0)
        return torch.where(b > 0, br, (h >= bot).to(h.dtype))

    def _conductances(self, h, ib):
        wet = ib != 0
        sat = self._sat(h, wet)
        t = self.k11 * (sat * self.thk)
        t1, t2 = t[..., :-1], t[..., 1:]
        cx = torch.where(t1 * t2 > 0, self.wx * t1 * t2
                         / (t1 * self.clx2 + t2 * self.clx1),
                         torch.zeros_like(t1))
        cx = cx * (wet[..., :-1] & wet[..., 1:])
        t1, t2 = t[..., :-1, :], t[..., 1:, :]
        cy = torch.where(t1 * t2 > 0, self.wy * t1 * t2
                         / (t1 * self.cly2 + t2 * self.cly1),
                         torch.zeros_like(t1))
        cy = cy * (wet[..., :-1, :] & wet[..., 1:, :])
        cz = self.condsat_z
        if self.shape[0] > 1:
            if self.variablecv:
                # vcond, upper cell n, lower cell m (botn > botm)
                satm = torch.ones_like(sat[1:])
                if self.dewatered:
                    satm = torch.where(sat[1:] < 1.0, torch.zeros_like(satm),
                                       sat[1:])
                b1 = sat[:-1] * self.thk[:-1] * 0.5 / self.k33[:-1]
                b2 = satm * self.thk[1:] * 0.5 / self.k33[1:]
                den = b1 + b2
                cv = torch.where(den != 0, self.area[:-1] / den,
                                 torch.zeros_like(den))
                keep = ((~self.conv[:-1] & ~self.conv[1:])
                        | ((h[:-1] >= self.top[:-1]) & (h[1:] >= self.top[1:])))
                cz = torch.where(keep, self.condsat_z, cv)
            cz = cz * (wet[:-1] & wet[1:])
        return cx, cy, cz

    def _rch_nodes(self, ib):
        """Flat node of the highest active cell per column (rch_cf)."""
        L, R, C = self.shape
        nz = ib != 0
        k = torch.where(nz.any(0), nz.to(torch.int8).argmax(0),
                        torch.full((R, C), L - 1, device=self.device))
        cols = torch.arange(R * C, device=self.device).view(R, C)
        return (k.to(torch.int64) * R * C + cols).view(-1)

    def _boundaries(self, h, ib):
        """hcof, rhs per list entry and the recharge rhs (no host sync)."""
        hf, ibf = h.view(-1), ib.view(-1)
        out = {}
        for name, pk in self.lists.items():
            nd = pk['nodes']
            hn = hf[nd]
            on = ibf[nd] > 0
            zero = torch.zeros_like(hn)
            if name == 'WEL':
                q = pk['q']
                if pk.get('afr') is not None:
                    bt = self.bot.view(-1)[nd]
                    conv = self.conv.view(-1)[nd]
                    thick = self.top.view(-1)[nd] - bt
                    tp = bt + pk['afr'] * thick
                    b = tp - bt
                    w = hn - bt
                    s = w / b
                    c1 = -2.0 / b ** 3
                    c2 = 3.0 / b ** 2
                    qm = torch.where(s < 0, zero, torch.where(
                        s < 1, c1 * w ** 3 + c2 * w ** 2, torch.ones_like(hn)))
                    q = torch.where((q < 0) & conv, q * qm, q)
                hcof, rhs = zero, -q
            elif name == 'RIV':
                c, st, rb = pk['cond'], pk['stage'], pk['rbot']
                below = hn <= rb
                rhs = torch.where(below, -c * (st - rb), -c * st)
                hcof = torch.where(below, zero, -c)
            elif name == 'DRN':
                f = (hn > pk['elev']).to(hn.dtype)
                rhs = -f * pk['cond'] * pk['elev']
                hcof = -f * pk['cond']
            elif name == 'GHB':
                hcof = -pk['cond']
                rhs = -pk['cond'] * pk['bhead']
            hcof = torch.where(on, hcof, zero)
            rhs = torch.where(on, rhs, zero)
            out[name] = (hcof, rhs)
        if self.evt is not None:
            out['EVT'] = self._evt_terms(hf, ibf)
        if self.rch is not None:
            nd = self._rch_nodes(ib)
            rr = -self.rch.reshape(-1) * self.area.view(-1)[nd]
            rr = torch.where(ibf[nd] > 0, rr, torch.zeros_like(rr))
            out['RCH'] = (torch.zeros_like(rr), rr, nd)
        return out

    def _evt_terms(self, hf, ibf):
        """evt_cf: hcof/rhs per column at the highest active cell."""
        e = self.evt
        nd = self._rch_nodes(ibf.view(self.shape))
        hn = hf[nd]
        zero = torch.zeros_like(hn)
        c = e['rate'].reshape(-1) * self.area.view(-1)[nd]
        s = e['surface'].reshape(-1)
        x = e['depth'].reshape(-1)
        d = s - hn
        top = c * e['petm0'].reshape(-1) if 'petm0' in e else c
        if 'pxdp' in e:
            nseg = e['pxdp'].shape[0] + 1
            pxdp1 = zero
            petm1 = (e['petm0'].reshape(-1) if 'petm0' in e
                     else torch.ones_like(hn))
            found = torch.zeros_like(hn, dtype=torch.bool)
            P1, Q1, P2, Q2 = zero, zero, zero, zero
            for iseg in range(nseg):
                if iseg < nseg - 1:
                    pxdp2 = e['pxdp'][iseg].reshape(-1)
                    petm2 = e['petm'][iseg].reshape(-1)
                else:
                    pxdp2 = torch.ones_like(hn)
                    petm2 = zero
                hit = ~found & ((d <= pxdp2 * x) | (iseg == nseg - 1))
                P1 = torch.where(hit, pxdp1, P1)
                Q1 = torch.where(hit, petm1, Q1)
                P2 = torch.where(hit, pxdp2, P2)
                Q2 = torch.where(hit, petm2, Q2)
                found = found | hit
                pxdp1, petm1 = pxdp2, petm2
            thcof = -(Q1 - Q2) * c / ((P2 - P1) * x)
            trhs = thcof * (s - P1 * x) + Q1 * c
        else:
            trhs = c - c * s / x
            thcof = -c / x
        above = hn >= s
        mid = ~above & (d < x)
        rhs = torch.where(above, top, torch.where(mid, trhs, zero))
        hcof = torch.where(mid, thcof, zero)
        on = ibf[nd] > 0
        return (torch.where(on, hcof, zero), torch.where(on, rhs, zero), nd)

    def _storage(self, h, hold, ib, dt, final=False):
        """Ss and Sy matrix terms (aterm, rhs) and rates for active cells."""
        tled = 1.0 / dt
        conv = self.iconv
        one = torch.ones_like(h)
        snold = torch.where(conv, self._qsat(hold, self.top, self.bot), one)
        snnew = torch.where(conv, self._qsat(h, self.top, self.bot), one)
        thick = one if self.storagecoefficient else self.thk
        rho1 = (self.ss * thick * self.area) * tled
        a_ss = -rho1 * snnew
        zold = self.bot + 0.5 * self.thk * snold
        znew = self.bot + 0.5 * self.thk * snnew
        r_ss = torch.where(conv, -rho1 * snold * (hold - zold)
                           - rho1 * snnew * znew, -rho1 * snold * hold)
        rho2 = (self.sy * self.area) * tled
        tt = self.thk
        a_sy = torch.where(conv & (snnew < 1) & (snnew > 0), -rho2,
                           torch.zeros_like(h))
        r_sy = torch.where(snnew < 1, torch.where(
            snnew > 0, -rho2 * tt * snold - rho2 * self.bot,
            tt * (0.0 - rho2 * snold)), tt * (rho2 * snnew - rho2 * snold))
        r_sy = torch.where(conv, r_sy, torch.zeros_like(h))
        on = ib > 0
        z = torch.zeros_like(h)
        a_ss, r_ss = torch.where(on, a_ss, z), torch.where(on, r_ss, z)
        a_sy, r_sy = torch.where(on, a_sy, z), torch.where(on, r_sy, z)
        if not final:
            return a_ss + a_sy, r_ss + r_sy
        q_ss = a_ss * h - r_ss
        q_sy = torch.where(on & conv, rho2 * tt * snold - rho2 * tt * snnew, z)
        return q_ss, q_sy

    def _sum_terms(self, h, hold, ib, dt, steady, bnd):
        """Total hcof H and rhs Rt per cell (boundaries + storage)."""
        H = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        Rt = torch.zeros_like(H)
        for name, v in bnd.items():
            if len(v) == 3:                       # one node per column
                H[v[2]] = H[v[2]] + v[0]
                Rt[v[2]] = Rt[v[2]] + v[1]
                continue
            sc = self.lists[name]['scatter']
            sc.add(H, v[0])
            sc.add(Rt, v[1])
        H = H.view(self.shape)
        Rt = Rt.view(self.shape)
        if not steady:
            a, r = self._storage(h, hold, ib, dt)
            H = H + a
            Rt = Rt + r
        return H, Rt

    # ------------------------------------------------------------ step
    # One algorithm on every device: an outer iteration is _pre (drying,
    # assembly, PCG start), k_in PCG iterations, _post (head update,
    # convergence); k_out outer iterations form a chunk; the host reads the
    # convergence flag once per chunk. Every update is masked by the
    # device-side flags, so iterations after convergence are no-ops.
    def _pre(self, h, hold, ib, conv, anydry, nout, diag, b, c, x, minv,
             red, r, p, rz, done, it, dt, steady):
        act = ~conv & (nout < self.outer_maximum)
        dry = act & (ib > 0) & self.conv & self.idomain \
            & (torch.minimum(h, self.top) - self.bot <= 0)
        ib.copy_(torch.where(dry, torch.zeros_like(ib), ib))
        h.copy_(torch.where(dry, torch.full_like(h, self.hdry), h))
        anydry.copy_(dry.any())
        cx, cy, cz = self._conductances(h, ib)
        H, Rt = self._sum_terms(h, hold, ib, dt, steady,
                                self._boundaries(h, ib))
        free = ib > 0
        fixed = ib < 0
        cf = sv.faces_to_full(cx, cy, cz)
        csum = sv.nbsum(torch.ones_like(h), cf)
        bchd = sv.nbsum(torch.where(fixed, h, torch.zeros_like(h)), cf)
        diag.copy_(torch.where(free, csum - H, torch.ones_like(h)))
        b.copy_(torch.where(free, -Rt + bchd, h))
        # couplings between two free cells only
        sh = (lambda v: F.pad(v[..., 1:], (0, 1)),
              lambda v: F.pad(v[..., :-1], (1, 0)),
              lambda v: F.pad(v[..., 1:, :], (0, 0, 0, 1)),
              lambda v: F.pad(v[..., :-1, :], (0, 0, 1, 0)),
              lambda v: F.pad(v[1:], (0, 0, 0, 0, 0, 1)),
              lambda v: F.pad(v[:-1], (0, 0, 0, 0, 1, 0)))
        ff = free.to(h.dtype)
        for k in range(6):
            c[k].copy_(cf[k] * ff * sh[k](ff))
        x.copy_(h)
        sv.pcg_start(x, b, diag, c, minv, red, r, p, rz, done, it, ~act,
                     self.precond)

    def _iter(self, x, diag, c, minv, red, r, p, rz, done, it):
        sv.pcg_iter(x, diag, c, minv, red, r, p, rz, done, it,
                    self.inner_dvclose, self.inner_rclose, self.precond,
                    check=False)

    def _iter_check(self, x, diag, c, minv, red, r, p, rz, done, it):
        sv.pcg_iter(x, diag, c, minv, red, r, p, rz, done, it,
                    self.inner_dvclose, self.inner_rclose, self.precond,
                    check=True)

    def _post(self, h, ib, x, done, conv, anydry, nout, nin, it, dh):
        act = ~conv & (nout < self.outer_maximum)
        upd = (ib > 0) & act
        dmax = torch.where(upd, (x - h).abs(), torch.zeros_like(h)).amax()
        h.copy_(torch.where(upd, x, h))
        dh.copy_(torch.where(act, dmax, dh))
        conv.copy_(conv | (act & (dmax <= self.outer_dvclose) & done
                           & ~anydry))
        nout.copy_(nout + act.to(nout.dtype))
        nin.copy_(nin + it)

    def _functions(self):
        if self._fns is None:
            f = (self._pre, self._iter, self._iter_check, self._post)
            if self.compile:
                from ..core import fusion
                fusion.ensure_setup()
                f = tuple(torch.compile(g, dynamic=False) for g in f)
            self._fns = f
        return self._fns

    def _chunk(self, dt, steady):
        pre, it_fn, it_chk, post = self._functions()
        S = self._S
        for _ in range(self.k_out):
            pre(S['h'], S['hold'], S['ib'], S['conv'], S['anydry'],
                S['nout'], S['diag'], S['b'], S['c'], S['x'], S['minv'],
                S['red'], S['r'], S['p'], S['rz'], S['done'], S['it'], dt,
                steady)
            k_in = self.k_in_steady if steady else self.k_in
            for k in range(k_in):
                (it_chk if k == k_in - 1 else it_fn)(
                    S['x'], S['diag'], S['c'], S['minv'], S['red'], S['r'],
                    S['p'], S['rz'], S['done'], S['it'])
            post(S['h'], S['ib'], S['x'], S['done'], S['conv'],
                 S['anydry'], S['nout'], S['nin'], S['it'], S['dh'])

    def _graph(self, dt, steady):
        key = (float(dt), bool(steady), self.k_in, self.k_in_steady)
        g = self._graphs.get(key)
        if g is None:
            S = self._S
            keep = {k: S[k].clone() for k in ('h', 'hold', 'ib')}
            s = torch.cuda.Stream(self.device)
            s.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(s):
                for _ in range(2):          # warm-up (compiles)
                    self._chunk(dt, steady)
            torch.cuda.current_stream(self.device).wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._chunk(dt, steady)
            self._graphs[key] = g
            for k, v in keep.items():
                S[k].copy_(v)
        return g

    def _solve(self, dt, steady, defer=False):
        """Outer iterations of one step from the committed state; returns
        (converged, outer iterations, inner iterations). With defer the
        chunks are launched without reading the flag back: as many as the
        hardest synchronous step needed (iterations after convergence are
        masked no-ops, so extra chunks change nothing) and the three values
        come back as device tensors."""
        S = self._S
        if self.use_graph:
            g = self._graph(dt, steady)
        S['h'].copy_(self.h)
        S['hold'].copy_(self.h)
        S['ib'].copy_(self.ibound)
        for k in ('conv', 'nout', 'nin'):
            S[k].zero_()
        if defer:
            for _ in range(getattr(self, '_ahead', 1)):
                if self.use_graph:
                    g.replay()
                else:
                    self._chunk(dt, steady)
            return S['conv'].clone(), S['nout'].clone(), S['nin'].clone()
        n = 0
        while True:
            if self.use_graph:
                g.replay()
            else:
                self._chunk(dt, steady)
            n += 1
            conv, nout, nin = torch.stack(
                [S['conv'].to(torch.int64), S['nout'], S['nin']]).tolist()
            if conv or nout >= self.outer_maximum:
                if conv:
                    # the chunks a deferred step launches: what converged
                    # steps needed, plus one. A solve that did not converge
                    # (a stiff step tried with short inner solves before
                    # `step` lengthens them) is not counted: its chunks ran
                    # to outer_maximum, and launching that many on every
                    # later step would repeat ~25x the work as masked
                    # no-ops (the result is the same either way)
                    self._ahead = max(getattr(self, '_ahead', 1), n + 1)
                return bool(conv), nout, nin

    @torch.no_grad()
    def step(self, dt, steady=False, allow_nonconvergence=False,
             defer=False):
        """Advance one time step of length ``dt`` (backward Euler).

        Picard outer iterations as the MODFLOW 6 standard formulation; the
        state is committed only when the outer iteration converged (else
        GroundwaterNotConverged, unless ``allow_nonconvergence``). On CUDA a
        chunk of outer iterations is one CUDA-graph replay; the host reads
        the convergence flag once per chunk.

        defer=True: no host read at all. The chunks a synchronous step
        needed at most so far are launched, the result carries ``converged``
        and the iteration counts as device tensors, and the state is
        committed as it stands; the caller must test ``converged`` (the
        coupling connector does, as a device flag batched with its ledger).
        """
        S = self._S
        if defer:
            conv, nout, nin = self._solve(dt, steady, defer=True)
            res = self._budget(S['h'], S['hold'], S['ib'], dt, steady)
            self.h, self.ibound = S['h'].clone(), S['ib'].clone()
            self.time += dt
            self.kstp += 1
            res.converged, res.outer_iterations = conv, nout
            res.inner_iterations = nin
            return res
        while True:
            conv, nout, nin = self._solve(dt, steady)
            k = self.k_in_steady if steady else self.k_in
            if conv or k >= self.inner_maximum:
                break
            # not converged with short inner solves (stiff system): retry
            # the step with 8x longer inner solves and keep them
            k = min(8 * k, self.inner_maximum)
            if steady:
                self.k_in_steady = k
            else:
                self.k_in = k
        converged = bool(conv)
        if not converged and not allow_nonconvergence:
            raise GroundwaterNotConverged(
                f'outer iterations did not converge in {self.outer_maximum}'
                f' (time {self.time}, step {self.kstp + 1})')
        res = self._budget(S['h'], S['hold'], S['ib'], dt, steady)
        h, ib = S['h'].clone(), S['ib'].clone()
        res.converged = converged
        res.outer_iterations = nout
        res.inner_iterations = nin
        if converged or allow_nonconvergence:
            self.h = h
            self.ibound = ib
            self.time += dt
            self.kstp += 1
        return res

    def _budget_terms(self, h, hold, ib, dt, steady):
        """Package rates at the converged heads (coefficients evaluated at
        the final heads): names, flow shapes, and one packed tensor holding
        the (P, 2) in/out rates followed by every flow array."""
        cx, cy, cz = self._conductances(h, ib)
        bnd = self._boundaries(h, ib)
        names, rows, flows = [], [], {}

        def add(name, q, qin=None):
            qq = q if qin is None else qin
            names.append(name)
            rows.append(torch.stack([torch.clamp(qq, min=0).sum(),
                                     torch.clamp(-qq, min=0).sum()]))
            flows[name] = q

        if not steady:
            q_ss, q_sy = self._storage(h, hold, ib, dt, final=True)
            add('STO-SS', q_ss)
            add('STO-SY', q_sy)
        hf = h.view(-1)
        if self.chd is not None:
            # calc_chd_rate: flows into the CHD cell from its neighbours;
            # in/out counts only connections to variable-head cells
            fixed = ib < 0
            free = ib > 0
            qin = torch.zeros_like(h)
            qall = torch.zeros_like(h)
            for c, sl_n, sl_m in self._face_slices(cx, cy, cz):
                q = c * (h[sl_m] - h[sl_n])      # into n from m
                qall[sl_n] -= q * fixed[sl_n]
                qall[sl_m] += q * fixed[sl_m]
                qin[sl_n] -= q * (fixed[sl_n] & free[sl_m])
                qin[sl_m] += q * (fixed[sl_m] & free[sl_n])
            add('CHD', qall.view(-1)[self.chd['nodes']], qin)
        for name, v in bnd.items():
            nd = v[2] if len(v) == 3 else self.lists[name]['nodes']
            add(name, v[0] * hf[nd] - v[1])
        sizes = [tuple(flows[k].shape) for k in names]
        packed = torch.cat([torch.stack(rows).reshape(-1)]
                           + [flows[k].reshape(-1) for k in names])
        return names, sizes, packed

    def _budget(self, h, hold, ib, dt, steady):
        key = ('budget', float(dt), bool(steady))
        if self.use_graph:
            out = self._graphs.get(key)
            if out is None:
                if self.compile:
                    from ..core import fusion
                    fusion.ensure_setup()
                fn = (torch.compile(self._budget_terms, dynamic=False)
                      if self.compile else self._budget_terms)
                st = torch.cuda.Stream(self.device)
                st.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(st):
                    for _ in range(2):
                        fn(h, hold, ib, dt, steady)
                torch.cuda.current_stream(self.device).wait_stream(st)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    o = fn(h, hold, ib, dt, steady)
                out = self._graphs[key] = (g, o)
            g, (names, sizes, packed) = out
            g.replay()
            packed = packed.clone()            # one copy per step
        else:
            names, sizes, packed = self._budget_terms(h, hold, ib, dt,
                                                      steady)
        P = len(names)
        io = packed[:2 * P].view(P, 2)
        parts = torch.split(packed[2 * P:], [int(np.prod(z)) for z in sizes])
        flows = {k: v.view(z) for k, v, z in zip(names, parts, sizes)}
        res = StepResult(False, 0, 0, float(dt))
        res.budget = {k: (io[i, 0], io[i, 1]) for i, k in enumerate(names)}
        res.flows = flows
        return res

    @staticmethod
    def _face_slices(cx, cy, cz):
        e = slice(None)
        yield cx, (e, e, slice(None, -1)), (e, e, slice(1, None))
        yield cy, (e, slice(None, -1), e), (e, slice(1, None), e)
        yield cz, (slice(None, -1), e, e), (slice(1, None), e, e)

    @torch.no_grad()
    def face_flows(self):
        """Flows across x, y, z faces at the current heads, positive towards
        increasing column, row and layer (L3/T)."""
        h = self.h
        cx, cy, cz = self._conductances(h, self.ibound)
        return (cx * (h[..., :-1] - h[..., 1:]),
                cy * (h[..., :-1, :] - h[..., 1:, :]),
                cz * (h[:-1] - h[1:]))

    # ----------------------------------------------------------- state
    def state_dict(self):
        """Restart state: heads, wet/dry flags (ibound), clock."""
        return {'h': self.h.clone(), 'ibound': self.ibound.clone(),
                'time': self.time, 'kstp': self.kstp}

    def solver_state(self):
        """What the solver learned (the inner solve lengths kept after a
        stiff step, the chunks a deferred step launches); a restart that
        loads it solves as the continuous run does."""
        return dict(k_in=self.k_in, k_in_steady=self.k_in_steady,
                    ahead=getattr(self, '_ahead', 1))

    def load_solver_state(self, st):
        self.k_in = int(st['k_in'])
        self.k_in_steady = int(st['k_in_steady'])
        self._ahead = int(st['ahead'])

    def load_state_dict(self, st):
        self.h = st['h'].to(self.device, self.dtype).clone()
        self.ibound = st['ibound'].to(self.device).clone()
        self.time = float(st['time'])
        self.kstp = int(st['kstp'])

    def save(self, path):
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v)
                    for k, v in self.state_dict().items()}, path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
