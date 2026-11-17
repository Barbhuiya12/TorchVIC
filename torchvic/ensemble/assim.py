"""Ensemble data assimilation on the GPU: EnKF, ETKF and LETKF.

The analysis acts on the states an ensemble run leaves (`Ensemble.run(...,
return_state=True)['state']`, member-major on the cell axis) and returns a
state of the same layout that the next window starts from, so the runtime
calls `Assimilator.analysis` at an observation time between windows. States
travel in memory; `torchvic.io.state_file` is not involved.

State variables (cell level, per member)
    soil moisture in the listed layers, mm (tile-mean over the cell's cover)
    snow water equivalent, mm            (optional)
An increment is a cell-level quantity. It is spread uniformly over the
cell's covered tiles (moisture: added to each tile's layer; SWE: every
extensive store of the pack scaled by one factor so density and temperature
are kept), then the constraints are applied tile by tile:

    0 (or the residual moisture) <= moisture <= max_moist     per tile, layer
    SWE >= 0; a pack is never created where a member has none

Clipping changes the increment that is realised. The realised increment is
what entered the model, and it is recorded in the `DALedger` as an external
source (positive) or sink (negative) of water, with the clipped part
reported separately.

Methods
    'enkf'   stochastic EnKF, perturbed observations, covariance
             localisation (Schur product, Gaspari-Cohn) when `radius_km` is set
    'etkf'   deterministic ensemble-space transform, global
    'letkf'  the same transform per cell on the observations within
             `radius_km` (Gaspari-Cohn weights on R^-1), batched over cells
"""
import math

import torch

from ..model import lanes as lanes_mod

#: extensive snow-pack stores scaled together when SWE is changed
_SNOW_SCALED = ('ice', 'pack_water', 'surf_water', 'swq_c', 'depth',
                'cold_content')


# ---------------------------------------------------------------------------
# geometry and localisation
# ---------------------------------------------------------------------------
def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance, km, broadcasting [a,1] against [1,b]."""
    r = 6371.0
    p1, p2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    dp, dl = p2 - p1, torch.deg2rad(lon2 - lon1)
    a = torch.sin(dp / 2) ** 2 + torch.cos(p1) * torch.cos(p2) \
        * torch.sin(dl / 2) ** 2
    return 2 * r * torch.asin(torch.sqrt(torch.clamp(a, 0.0, 1.0)))


def gaspari_cohn(d, radius):
    """Gaspari and Cohn (1999) eq. 4.10 correlation: 1 at d = 0, exactly 0
    from d = `radius` outward (half-width radius / 2)."""
    z = 2.0 * d / radius
    z2, z3, z4, z5 = z * z, z ** 3, z ** 4, z ** 5
    near = -0.25 * z5 + 0.5 * z4 + 0.625 * z3 - 5.0 / 3.0 * z2 + 1.0
    far = (z5 / 12.0 - 0.5 * z4 + 0.625 * z3 + 5.0 / 3.0 * z2 - 5.0 * z
           + 4.0 - 2.0 / (3.0 * torch.clamp(z, min=1e-12)))
    return torch.where(z <= 1.0, near,
                       torch.where(z < 2.0, far, torch.zeros_like(z)))


# ---------------------------------------------------------------------------
# the state, seen at the cell level
# ---------------------------------------------------------------------------
class StateView(object):
    """Cell-level reads and constrained increments on a model state.

    The state holds tiles either on the padded [cell, tile] grid or, from a
    CUDA run, on the covered tiles only (lanes); both are flattened to one
    slot axis with an owner cell and a cover weight."""

    def __init__(self, ens, state):
        self.state = state
        nrow = ens.n * ens.nc
        self.nrow = nrow
        mo = state['moist']
        nl = mo.shape[-1]
        mosaic = ens.case.mosaic
        dev = mo.device
        if mosaic is None:
            self.owner = torch.arange(nrow, device=dev)
            self.w = torch.ones(nrow, dtype=mo.dtype, device=dev)
        else:
            cv = mosaic.cv.to(dev)
            if state.get('lanes') is not None:
                ln = lanes_mod.Lanes(cv)
                if ln.n_lanes != int(state['lanes']['n_lanes']):
                    raise ValueError('state lanes do not match the mosaic')
                self.owner = ln.lane_cell
                self.w = cv[ln.lane_cell, ln.lane_tile].to(mo.dtype)
            else:
                nt = cv.shape[1]
                self.owner = torch.arange(nrow, device=dev) \
                    .repeat_interleave(nt)
                self.w = cv.reshape(-1).to(mo.dtype)
        if mo.reshape(-1, nl).shape[0] != self.owner.shape[0]:
            raise ValueError('state moist %s does not fit the tile layout'
                             % (tuple(mo.shape),))
        self.nl = nl
        self.covered = self.w > 0
        self.maxm = ens.case.soil.max_moist.to(dev, mo.dtype)    # [rows, L]
        self.resid = ens.case.soil.resid_moist.to(dev, mo.dtype)

    # -- cell reductions ------------------------------------------------
    def _cell(self, x):                       # x [slots, ...] -> [rows, ...]
        w = self.w.reshape((-1,) + (1,) * (x.dim() - 1))
        out = torch.zeros((self.nrow,) + tuple(x.shape[1:]), dtype=x.dtype,
                          device=x.device)
        return out.index_add_(0, self.owner, x * w)

    def moist_cell(self):
        """[rows, layers] tile-mean soil moisture, mm."""
        return self._cell(self.state['moist'].reshape(-1, self.nl))

    def _swq_slots(self):
        s = self.state['snow']
        return (s['ice'] + s['pack_water'] + s['surf_water']).reshape(-1)

    def swe_cell(self):
        """[rows] tile-mean snow water equivalent, mm."""
        return self._cell(self._swq_slots()) * 1000.0

    # -- constrained increments ----------------------------------------
    def apply(self, d_moist=None, swe_target=None, lower='zero'):
        """A new state with the cell-level increment `d_moist` [rows, L] mm
        added (and the SWE brought to `swe_target` [rows] mm), and
        (realised d_moist [rows, L], realised d_swe [rows]) in mm. The
        input state is not modified."""
        st = dict(self.state)
        if d_moist is None:
            d_real = torch.zeros(self.nrow, self.nl, dtype=self.w.dtype,
                                 device=self.w.device)
        else:
            old = self.state['moist'].reshape(-1, self.nl)
            hi = self.maxm[self.owner]
            lo = (torch.zeros_like(hi) if lower == 'zero'
                  else self.resid[self.owner])
            # a tile already outside the bounds is brought to them, so the
            # constraint also repairs it (the correction is in the ledger)
            new = torch.minimum(torch.maximum(old + d_moist[self.owner], lo),
                                hi)
            new = torch.where(self.covered.unsqueeze(1), new, old)
            d_real = self._cell(new - old)
            st['moist'] = new.reshape(self.state['moist'].shape)
        d_swe = torch.zeros(self.nrow, dtype=self.w.dtype, device=self.w.device)
        if swe_target is not None:
            cur = self.swe_cell()
            target = torch.clamp(swe_target, min=0.0)
            f = torch.where(cur > 0, target / torch.clamp(cur, min=1e-300),
                            torch.ones_like(cur))
            fs = f[self.owner]
            snow = dict(self.state['snow'])
            for k in _SNOW_SCALED:
                if k in snow and torch.is_tensor(snow[k]):
                    snow[k] = (snow[k].reshape(-1) * fs).reshape(snow[k].shape)
            st['snow'] = snow
            d_swe = self.swe_cell_of(snow) - cur
        return st, d_real, d_swe

    def swe_cell_of(self, snow):
        return self._cell((snow['ice'] + snow['pack_water']
                           + snow['surf_water']).reshape(-1)) * 1000.0


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------
class DALedger(object):
    """Water added to (+) or removed from (-) the model states by analyses.

    Each entry holds, per member and cell, the increment the analysis asked
    for (`requested`, mm) and the one that entered the states after the
    constraints (`realised`, mm; soil layers summed plus SWE), so
    `clipped = realised - requested` is the constraint's share. For the
    domain balance an analysis is an external flux: use
    `external_m3(i)` (ensemble-mean volume, m3) as I_ext of the step the
    analysis follows. The state change of every member equals its realised
    increment exactly (tests/test_ensemble_da.py)."""

    def __init__(self, area_m2=None, keep_cells=True):
        self.area = area_m2
        self.keep_cells = keep_cells
        self.entries = []

    def record(self, time, requested, realised, d_moist, d_swe, variables):
        e = dict(time=time, variables=list(variables),
                 requested=requested.detach(), realised=realised.detach(),
                 d_moist=d_moist.detach(), d_swe=d_swe.detach())
        if not self.keep_cells:
            e['requested'] = e['requested'].sum(1)
            e['realised'] = e['realised'].sum(1)
            e['d_moist'] = e['d_moist'].sum(1)
            e['d_swe'] = e['d_swe'].sum(1)
        self.entries.append(e)
        return len(self.entries) - 1

    def clipped_mm(self, i):
        e = self.entries[i]
        return e['realised'] - e['requested']

    def member_m3(self, i):
        """[members] net water volume added to each member, m3."""
        r = self.entries[i]['realised']
        if r.dim() == 1:
            return r
        return (r / 1000.0 * self.area.to(r.device, r.dtype)).sum(1) \
            if self.area is not None else r.sum(1)

    def external_m3(self, i):
        """Ensemble-mean net volume added by analysis i, m3 (+ source)."""
        return float(self.member_m3(i).mean())

    def rows(self):
        """One dict per analysis for the runtime ledger."""
        out = []
        for i, e in enumerate(self.entries):
            m = self.member_m3(i)
            out.append(dict(
                time=e['time'], term='da_increment',
                external_m3=float(m.mean()), member_min_m3=float(m.min()),
                member_max_m3=float(m.max()),
                clipped_mm_abs=float(self.clipped_mm(i).abs().sum())))
        return out


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------
class ObsSet(object):
    """Point observations of linear functions of one cell's state.

    cell [no] index into the member's cells; coef [no, V] weights on the
    cell's V analysed variables; value [no]; sigma [no] error std."""

    def __init__(self, cell, coef, value, sigma):
        # as_tensor under a default device moves a tensor to it; keep ours
        def _t(x, **k):
            if isinstance(x, torch.Tensor):
                return x.to(**k) if k else x
            return torch.as_tensor(x, **k)
        self.cell = _t(cell, dtype=torch.long)
        self.coef = _t(coef)
        self.value = _t(value)
        s = _t(sigma)
        self.sigma = s.expand(self.value.shape).clone() if s.dim() == 0 else s
        self.n = int(self.value.shape[0])

    def to(self, device, dtype):
        return ObsSet(self.cell.to(device), self.coef.to(device, dtype),
                      self.value.to(device, dtype), self.sigma.to(device, dtype))


# ---------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------
def _etkf(A, Yb, d, rinv, rho=1.0):
    """Batched ensemble-space transform. A [B,N,V] prior anomalies, Yb
    [B,N,m] observation-space anomalies, d [B,m] innovation, rinv [B,m]
    inverse error variance (0 = ignore). Returns the analysis anomalies
    [B,N,V] about the prior mean (the mean shift is included) and the
    weight vector of the mean shift [B,N]."""
    n = A.shape[1]
    Yr = Yb * rinv.unsqueeze(1)                                  # [B,N,m]
    Pinv = Yr @ Yb.transpose(1, 2)
    Pinv = Pinv + (n - 1) / rho * torch.eye(n, dtype=A.dtype,
                                            device=A.device)
    lam, V = torch.linalg.eigh(Pinv)
    lam = torch.clamp(lam, min=1e-12)
    # Pa = V diag(1/lam) V^T ;  W = sqrt(N-1) V diag(lam^-1/2) V^T
    wbar = torch.einsum('bij,bj->bi', V,
                        torch.einsum('bkj,bk->bj', V,
                                     (Yr @ d.unsqueeze(-1)).squeeze(-1))
                        / lam)
    W = math.sqrt(n - 1) * (V * lam.rsqrt().unsqueeze(1)) \
        @ V.transpose(1, 2)
    wfull = wbar.unsqueeze(2) + W                                # [B,N,N]
    # Xa_i = xbar + sum_k wfull[k,i] A_k
    Aa = torch.einsum('bki,bkv->biv', wfull, A)
    return Aa, wbar


class Assimilator(object):
    """Analysis at observation times for an `Ensemble`.

    layers    soil layers analysed (cell-mean moisture, mm)
    swe       also analyse snow water equivalent
    method    'enkf' | 'etkf' | 'letkf'
    radius_km localisation cut-off (Gaspari-Cohn); required for 'letkf'
    inflation multiplicative prior covariance inflation (>= 1)
    max_local_obs  the most observations a LETKF cell uses (nearest)
    lower     'zero' or 'resid': lower moisture bound
    """

    def __init__(self, ens, layers=(0,), swe=False, method='enkf',
                 radius_km=None, inflation=1.0, max_local_obs=64, seed=0,
                 lower='zero', keep_cells=True):
        if method not in ('enkf', 'etkf', 'letkf'):
            raise ValueError("method is 'enkf', 'etkf' or 'letkf'")
        if method == 'letkf' and not radius_km:
            raise ValueError('letkf needs radius_km')
        if lower not in ('zero', 'resid'):
            raise ValueError("lower is 'zero' or 'resid'")
        self.ens, self.method = ens, method
        self.layers, self.swe = tuple(layers), bool(swe)
        self.radius, self.inflation = radius_km, float(inflation)
        self.max_local = int(max_local_obs)
        self.seed, self.lower = int(seed), lower
        self.nv = len(self.layers) + (1 if swe else 0)
        c = ens.case
        self.area = c.batch.cell_area[:ens.nc].detach() \
            if torch.is_tensor(getattr(c.batch, 'cell_area', None)) else None
        self.lat = c.batch.latitude[:ens.nc]
        self.lon = c.batch.longitude[:ens.nc]
        self.ledger = DALedger(self.area, keep_cells)
        self.count = 0

    # -- observation helpers -------------------------------------------
    def obs_soil_moisture(self, cells, value, sigma, layers=None, scale=1.0):
        """Observation = scale * sum of the listed layers' moisture (mm)."""
        layers = self.layers if layers is None else tuple(layers)
        cells = torch.as_tensor(cells, dtype=torch.long)
        coef = torch.zeros(len(cells), self.nv, dtype=torch.float64)
        for ly in layers:
            coef[:, self.layers.index(ly)] = scale
        return ObsSet(cells, coef, value, sigma)

    def obs_swe(self, cells, value, sigma):
        if not self.swe:
            raise ValueError('this assimilator does not analyse SWE')
        cells = torch.as_tensor(cells, dtype=torch.long)
        coef = torch.zeros(len(cells), self.nv, dtype=torch.float64)
        coef[:, -1] = 1.0
        return ObsSet(cells, coef, value, sigma)

    # -- state matrix ----------------------------------------------------
    def matrix(self, state):
        """[members, cells, V] analysed variables of `state`."""
        v = StateView(self.ens, state)
        cols = [v.moist_cell()[:, list(self.layers)]]
        if self.swe:
            cols.append(v.swe_cell().unsqueeze(1))
        x = torch.cat(cols, 1)
        return x.reshape(self.ens.n, self.ens.nc, self.nv)

    def analysis(self, state, obs, time=None):
        """Assimilate `obs` (an ObsSet or a list of them) into `state`.
        Returns (new state, info). The ledger gets one entry."""
        if not isinstance(obs, ObsSet):
            obs = ObsSet(torch.cat([o.cell for o in obs]),
                         torch.cat([o.coef for o in obs]),
                         torch.cat([o.value for o in obs]),
                         torch.cat([o.sigma for o in obs]))
        n, nc, nv = self.ens.n, self.ens.nc, self.nv
        view = StateView(self.ens, state)
        X = self.matrix(state)
        dev, dt = X.device, X.dtype
        obs = obs.to(dev, dt)
        # H X: [N, no]
        HX = torch.einsum('njv,jv->nj', X[:, obs.cell], obs.coef)
        xm, hm = X.mean(0, keepdim=True), HX.mean(0, keepdim=True)
        sq = math.sqrt(self.inflation)
        A, HA = (X - xm) * sq, (HX - hm) * sq
        xm, hm = xm[0], hm[0]
        self.count += 1
        if self.method == 'enkf':
            Xa = self._enkf(X, xm, A, HA, hm, obs)
        elif self.method == 'etkf':
            Aa, _ = _etkf(A.reshape(1, n, nc * nv), HA.unsqueeze(0),
                          (obs.value - hm).unsqueeze(0),
                          (1.0 / obs.sigma ** 2).unsqueeze(0))
            Xa = xm + Aa.reshape(n, nc, nv)
        else:
            Xa = self._letkf(xm, A, HA, hm, obs)
        return self._commit(view, state, X, Xa, time)

    # -- stochastic EnKF ------------------------------------------------
    def _enkf(self, X, xm, A, HA, hm, obs):
        n, nc, nv = self.ens.n, self.ens.nc, self.nv
        dt, dev = X.dtype, X.device
        g = torch.Generator().manual_seed(self.seed * 7919 + self.count)
        eps = torch.randn(n, obs.n, generator=g, dtype=dt).to(dev)
        eps = (eps - eps.mean(0)) * obs.sigma
        # innovation of each member against its perturbed observation
        HX = HA + hm
        D = obs.value + eps - HX
        Pyy = HA.T @ HA / (n - 1)
        Pxy = A.reshape(n, -1).T @ HA / (n - 1)               # [C*V, no]
        if self.radius:
            dd = distance_km(self.lat.unsqueeze(1), self.lon.unsqueeze(1),
                             self.lat[obs.cell].unsqueeze(0),
                             self.lon[obs.cell].unsqueeze(0))   # [C, no]
            rho_xy = gaspari_cohn(dd, self.radius)
            Pxy = (Pxy.reshape(nc, nv, obs.n) * rho_xy.unsqueeze(1)) \
                .reshape(nc * nv, obs.n)
            doo = distance_km(self.lat[obs.cell].unsqueeze(1),
                              self.lon[obs.cell].unsqueeze(1),
                              self.lat[obs.cell].unsqueeze(0),
                              self.lon[obs.cell].unsqueeze(0))
            Pyy = Pyy * gaspari_cohn(doo, self.radius)
        Pyy = Pyy + torch.diag(obs.sigma ** 2)
        L = torch.linalg.cholesky(Pyy)
        # K^T = Pyy^-1 Pxy^T
        Kt = torch.cholesky_solve(Pxy.T, L)                   # [no, C*V]
        # the prior (uninflated) ensemble plus the inflated spread change
        Xi = xm + A                                           # inflated prior
        return Xi + (D @ Kt).reshape(n, nc, nv)

    # -- LETKF ----------------------------------------------------------
    def _letkf(self, xm, A, HA, hm, obs, chunk=2048):
        n, nc, nv = self.ens.n, self.ens.nc, self.nv
        m = min(self.max_local, obs.n)
        out = torch.empty_like(A)
        olat, olon = self.lat[obs.cell], self.lon[obs.cell]
        innov = obs.value - hm
        rinv_all = 1.0 / obs.sigma ** 2
        for s in range(0, nc, chunk):
            e = min(nc, s + chunk)
            dd = distance_km(self.lat[s:e].unsqueeze(1),
                             self.lon[s:e].unsqueeze(1),
                             olat.unsqueeze(0), olon.unsqueeze(0))
            dsel, idx = torch.topk(dd, m, dim=1, largest=False)    # [B,m]
            w = gaspari_cohn(dsel, self.radius)
            rinv = rinv_all[idx] * w
            Yb = HA[:, idx].permute(1, 0, 2)                       # [B,N,m]
            d = innov[idx]
            Ac = A[:, s:e].permute(1, 0, 2)                        # [B,N,V]
            Aa, wbar = _etkf(Ac, Yb, d, rinv)
            out[:, s:e] = Aa.permute(1, 0, 2)
        return xm + out

    # -- constraints, state, ledger ---------------------------------------
    def _commit(self, view, state, X, Xa, time):
        n, nc, nv = self.ens.n, self.ens.nc, self.nv
        nl = view.nl
        req = (Xa - X).reshape(n * nc, nv)
        d_m = torch.zeros(n * nc, nl, dtype=X.dtype, device=X.device)
        d_m[:, list(self.layers)] = req[:, :len(self.layers)]
        target = None
        if self.swe:
            target = view.swe_cell() + req[:, -1]
        new, d_real, d_swe = view.apply(d_m, target, self.lower)
        req_tot = d_m.sum(1) + (req[:, -1] if self.swe else 0.0)
        real_tot = d_real.sum(1) + d_swe
        names = ['moist%d' % ly for ly in self.layers] \
            + (['swe'] if self.swe else [])
        i = self.ledger.record(
            time, req_tot.reshape(n, nc), real_tot.reshape(n, nc),
            d_real.reshape(n, nc, nl), d_swe.reshape(n, nc), names)
        info = dict(entry=i, external_m3=self.ledger.external_m3(i),
                    clipped_mm=float(self.ledger.clipped_mm(i).abs().sum()),
                    spread_prior=float(X.std(0).mean()),
                    spread_post=float(self.matrix(new).std(0).mean()))
        return new, info
