"""Seeded, reproducible perturbation generators for ensembles.

Everything is drawn on the CPU from a `torch.Generator` seeded per call, so a
seed gives the same numbers on every machine and device, and is then moved to
the ensemble's device. A zero spread draws nothing and leaves the input
tensors untouched (the same objects), which is what makes the zero-perturbation
member equal the deterministic run bit for bit.

* `latin_hypercube`      stratified uniform samples in [0, 1)^d
* `sample_parameters`    Latin-hypercube parameter sets, linear or log scale
* `ForcingPerturbation`  multiplicative precipitation, additive temperature
"""
import math

import torch


def latin_hypercube(n, d, seed=0, dtype=torch.float64):
    """[n, d] Latin hypercube: each column has exactly one point in each of
    the n equal strata of [0, 1), placed at random inside its stratum, the
    strata in an independent random order per column."""
    g = torch.Generator().manual_seed(int(seed))
    jitter = torch.rand(n, d, generator=g, dtype=dtype)
    out = torch.empty(n, d, dtype=dtype)
    for j in range(d):
        perm = torch.randperm(n, generator=g)
        out[:, j] = (perm.to(dtype) + jitter[:, j]) / n
    return out


def sample_parameters(ranges, n, seed=0, log=(), control=None,
                      dtype=torch.float64, device=None):
    """Parameter sets for `n` members: {name: tensor [n]}.

    `ranges` maps a parameter name (a key of the model's `p`, e.g.
    'rmin_scale', or a soil parameter `Case.member_params` accepts, e.g.
    'b_infilt') to (low, high). Names in `log` are sampled uniformly in
    log space (for scale factors and conductivities). `control` maps names to
    nominal values; when given, member 0 takes them (the unperturbed control
    member) and the Latin hypercube is drawn over members 1..n-1.
    """
    names = list(ranges)
    first = 0 if control is None else 1
    m = n - first
    u = (latin_hypercube(m, len(names), seed, dtype) if m > 0
         else torch.empty(0, len(names), dtype=dtype))
    out = {}
    for j, k in enumerate(names):
        lo, hi = ranges[k]
        if k in log:
            if lo <= 0 or hi <= 0:
                raise ValueError('%s: log sampling needs positive bounds' % k)
            v = torch.exp(math.log(lo) + u[:, j] * (math.log(hi) - math.log(lo)))
        else:
            v = lo + u[:, j] * (hi - lo)
        if control is not None:
            v = torch.cat([torch.tensor([float(control[k])], dtype=dtype), v])
        out[k] = v.to(device) if device is not None else v
    return out


def _ar1(n, steps, phi, gen, dtype):
    """[n, steps] AR(1) unit-variance noise, lag-1 correlation phi."""
    z = torch.randn(n, steps, generator=gen, dtype=dtype)
    if phi == 0.0:
        return z
    s = math.sqrt(1.0 - phi * phi)
    out = torch.empty_like(z)
    out[:, 0] = z[:, 0]
    for t in range(1, steps):
        out[:, t] = phi * out[:, t - 1] + s * z[:, t]
    return out


class ForcingPerturbation(object):
    """Perturbation of the atmosphere an ensemble is forced with.

    prcp_sigma   multiplicative precipitation: factor exp(s z - s^2/2) with
                 z ~ N(0, 1), so the factor has mean 1 and precipitation is
                 never negative.
    tair_sigma   additive air temperature, K (applied to tair, tmax, tmin;
                 air density, latent heat and vapour pressure deficit are
                 recomputed from the new temperature, vapour pressure kept).
    phi          lag-1 autocorrelation of the noise in time (per step).
    spatial      'member': one noise series per member, shared by all cells
                 (a regional forcing error); 'cell': independent per cell.
    control      member 0 is left unperturbed.
    A zero sigma draws nothing for that variable.
    """

    def __init__(self, prcp_sigma=0.0, tair_sigma=0.0, phi=0.0,
                 spatial='member', control=False, seed=0):
        if spatial not in ('member', 'cell'):
            raise ValueError("spatial is 'member' or 'cell'")
        if not 0.0 <= phi < 1.0:
            raise ValueError('phi must be in [0, 1)')
        self.prcp_sigma, self.tair_sigma = float(prcp_sigma), float(tair_sigma)
        self.phi, self.spatial = float(phi), spatial
        self.control, self.seed = bool(control), int(seed)

    def _noise(self, which, n, steps, ncell, dtype):
        gen = torch.Generator().manual_seed(self.seed * 1000003 + which)
        if self.spatial == 'member':
            z = _ar1(n, steps, self.phi, gen, dtype)          # [n, T]
            z = z.unsqueeze(-1)                               # [n, T, 1]
        else:
            z = _ar1(n * ncell, steps, self.phi, gen, dtype)  # [n*C, T]
            z = z.reshape(n, ncell, steps).permute(0, 2, 1)   # [n, T, C]
        if self.control:
            z = z.clone()
            z[0] = 0.0
        return z

    def factors(self, n, steps, ncell, dtype=torch.float64):
        """(precipitation factor, temperature shift K): each [n, T, 1] or
        [n, T, C], or None when its sigma is zero."""
        f = dt = None
        if self.prcp_sigma > 0.0:
            s = self.prcp_sigma
            f = torch.exp(s * self._noise(1, n, steps, ncell, dtype)
                          - 0.5 * s * s)
            if self.control:
                f[0] = 1.0
        if self.tair_sigma > 0.0:
            dt = self.tair_sigma * self._noise(2, n, steps, ncell, dtype)
        return f, dt

    def apply(self, atm, n, ncell):
        """The ensemble's atmosphere. `atm` is the member-major atmosphere of
        `Case.ensemble(n)` ([time, n * ncell] fields); a new dict is
        returned, the input is not modified."""
        from ..core import atmos
        from ..core import snow
        out = dict(atm)
        steps = int(atm['tair'].shape[0])
        dtype, dev = atm['tair'].dtype, atm['tair'].device
        f, dt = self.factors(n, steps, ncell, dtype)

        def flat(x):                 # [n, T, 1|C] -> [T, n*C]
            x = x.to(dev).expand(n, steps, ncell)
            return x.permute(1, 0, 2).reshape(steps, n * ncell)
        if f is not None:
            out['prcp'] = atm['prcp'] * flat(f)
        if dt is not None:
            d = flat(dt)
            tair = atm['tair'] + d
            out['tair'] = tair
            for k in ('tmax', 'tmin'):
                if k in atm:
                    out[k] = atm[k] + d
            out['air_dens'] = atmos.air_density(tair, atm['press'])
            out['lv'] = atmos.latent_heat_of_vaporization(tair)
            out['vpd'] = torch.clamp(snow.svp(tair) - atm['eact_air'], min=0.0)
        return out
