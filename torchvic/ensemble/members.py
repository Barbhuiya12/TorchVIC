"""N members of the existing model as extra cells of one run.

`Case.ensemble(n)` repeats the cell batch n times (member-major: row
`m * n_cells + c` is member m's cell c). The GPU path (lanes, CUDA graphs,
fusion) runs on that longer cell axis unchanged, and the launches do not
multiply with the members. This class adds what an ensemble needs on top:
seeded forcing and parameter perturbation, windowed runs that carry the
states (in memory, between assimilation times), and outputs reshaped to
[member, cell, time].
"""
import copy

import torch

from ..core import version as core_version
from ..model import basin
from ..model import state as state_mod
from .perturb import ForcingPerturbation


class Ensemble(object):
    """`n` members of `case`.

    forcing   a ForcingPerturbation (None: every member has the same forcing)
    params    {name: tensor [n] or [n, n_cells]} member parameters, e.g. from
              perturb.sample_parameters (None: the case's default parameters)
    """

    def __init__(self, case, n, forcing=None, params=None):
        self.base, self.n = case, int(n)
        self.nc = case.n_cells                 # cells per member
        ens = case.ensemble(self.n)
        ens = copy.copy(ens)                   # n == 1 returns the case itself
        if forcing is not None:
            ens.atm = forcing.apply(ens.atm, self.n, self.nc)
        self.case, self.forcing = ens, forcing
        self.p = (ens.member_params(params, self.n) if params
                  else ens.params())
        self.n_steps = case.n_steps

    # ------------------------------------------------------------------
    def kwargs(self, t0=0, t1=None, state=None, **extra):
        kw = self.case.kwargs(p=self.p, **extra)
        if state is not None:
            kw['initial_state'] = state
        if t0 != 0 or (t1 is not None and t1 != self.n_steps):
            kw = state_mod.time_window(kw, t0, t1)
        return kw

    def simulate(self, t0=0, t1=None, state=None, return_state=False, **extra):
        """The raw basin.simulate output over steps [t0, t1), member-major
        on the cell axis. `state` is an in-memory state (a previous
        window's out['state'], possibly after an analysis)."""
        c = self.case
        if c.core is not None:
            core_version.set_core(c.core)
        kw = self.kwargs(t0, t1, state, return_state=return_state, **extra)
        out = basin.simulate(**kw)
        if c.snowbands is not None:
            out = c.contract_bands(out)
        return out

    def run(self, t0=0, t1=None, state=None, return_state=False, keep=None,
            **extra):
        """Run and return the outputs as [member, cell, time, ...] (see
        `members`). `keep` names the outputs to return (default all); the
        state, if requested, is out['state'] and stays on the cell axis."""
        out = self.simulate(t0, t1, state, return_state, **extra)
        res = {}
        for k, v in out.items():
            if keep is not None and k not in keep and k != 'state':
                continue
            res[k] = v if k in ('state', 'state_aux') else self.members(v)
        return res

    # ------------------------------------------------------------------
    def members(self, x):
        """A simulate output with the member axis exposed.

        [time, n*cell, ...]   -> [n, cell, time, ...]
        [n*cell, ...]         -> [n, cell, ...]
        discharge_mm [time, n*outlets] -> [n, outlets, time]
        Anything else is returned as it is."""
        if not torch.is_tensor(x) or x.dim() == 0:
            return x
        n, nc = self.n, self.nc
        if x.dim() >= 2 and x.shape[1] == n * nc:
            x = x.reshape((x.shape[0], n, nc) + tuple(x.shape[2:]))
            return x.permute(1, 2, 0, *range(3, x.dim())).contiguous()
        if x.shape[0] == n * nc:
            return x.reshape((n, nc) + tuple(x.shape[1:]))
        no = int(getattr(self.case.batch, 'n_outlets', 1)) // max(n, 1)
        if x.dim() == 2 and no and x.shape[1] == n * no:
            return x.reshape(x.shape[0], n, no).permute(1, 2, 0).contiguous()
        return x
