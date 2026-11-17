"""A conservative two-store test connector.

Two linear stores per element, A and B (mm over the element's area):

    A' = inflow + precip_add_A - k_ab A           (per interval, explicit)
    B' = B + k_ab A - k_out B                     outflow leaves the domain

The interval's transfer A -> B is internal (cancels in the balance); what
enters is a constant `precip` (mm per interval) and the `inflow` input; what
leaves is `k_out B` (exported as `outflow`, a flux another connector can
take). Exact in exact arithmetic, so its ledger residual is rounding only:
it is the framework's own conservation test, and a pattern for a real
connector's Balance.

`fail_at` (seconds) makes compute raise at that time, `reject_at` makes
validate refuse, `leak` (mm per interval) makes the balance it reports
wrong by that much -- the failure tests' levers.
"""
import torch

from .connector import Balance, ComputeError, Connector, Proposal
from .spec import GridSpec, VarSpec


class TwoStoreConnector(Connector):
    name = 'two_store'
    version = '1'
    backends = ('cpu', 'cuda')
    validation = 'exact by construction (framework test connector)'

    @property
    def owned_state(self):
        # each instance owns its own two stores
        return (self.name + '.store_a', self.name + '.store_b')
    imported_fluxes = ('inflow',)
    exported_fluxes = ('outflow',)

    def __init__(self, n=4, interval_s=86400.0, grid='two_store_cells',
                 area_m2=1.0e6, precip=1.0, k_ab=0.3, k_out=0.2,
                 a0=10.0, b0=5.0, units='mm', fail_at=None, reject_at=None,
                 leak=0.0, name=None, dependencies=(), **config):
        super().__init__(**config)
        if name:
            self.name = name
        self.n = int(n)
        self.interval_s = float(interval_s)
        self.grid = grid
        self.area_m2 = float(area_m2)
        self.precip, self.k_ab, self.k_out = float(precip), float(k_ab), \
            float(k_out)
        self.a0, self.b0 = float(a0), float(b0)
        self.units = units
        self.fail_at, self.reject_at = fail_at, reject_at
        self.leak = float(leak)
        self.dependencies = tuple(dependencies)
        self.inputs = (VarSpec('inflow', units, ('cell',), kind='flux',
                               grid=grid, description='water added to A'),)
        self.outputs = (
            VarSpec('outflow', units, ('cell',), kind='flux', grid=grid,
                    description='k_out B, leaving the element'),
            VarSpec('store_b', units, ('cell',), kind='state', grid=grid,
                    description='store B at the interval end'))
        self.a = self.b = None
        self.t = 0.0
        self.dtype = torch.float64
        self.device = None

    def grids(self):
        return [GridSpec(self.grid, self.n,
                         area=torch.full((self.n,), self.area_m2,
                                         dtype=torch.float64))]

    def initialize(self, context):
        self.device = context.device if context.device is not None else 'cpu'
        self.a = torch.full((self.n,), self.a0, dtype=self.dtype,
                            device=self.device)
        self.b = torch.full((self.n,), self.b0, dtype=self.dtype,
                            device=self.device)

    def compute(self, inputs, context, dt):
        if self.fail_at is not None and context.t >= float(self.fail_at):
            raise ComputeError('%s: told to fail at t = %g s'
                               % (self.name, context.t))
        frac = dt / self.interval_s
        inflow = inputs.get('inflow')
        inflow = (torch.zeros_like(self.a) if inflow is None else
                  torch.as_tensor(inflow, dtype=self.dtype,
                                  device=self.a.device))
        p = torch.full_like(self.a, self.precip * frac)
        t_ab = self.k_ab * frac * self.a
        t_out = self.k_out * frac * self.b
        a1 = self.a + p + inflow - t_ab
        b1 = self.b + t_ab - t_out
        dS = (a1 + b1) - (self.a + self.b)
        area = self.area_m2 / 1000.0                 # mm -> m3
        P, I, Q, dSv = (float(x.sum()) * area for x in (p, inflow, t_out,
                                                         dS))
        leak = self.leak * self.n * area
        scale = (float((p.abs() + inflow.abs() + t_out.abs() + dS.abs()
                        + 2 * t_ab.abs()).sum()) * area)
        bal = Balance(P=P, I_ext=I, Q_out=Q, dS=dSv + leak, n_terms=8 * self.n,
                      scale=scale,
                      per_cell=dict(residual=p + inflow - t_out - dS
                                    - self.leak,
                                    scale=p.abs() + inflow.abs()
                                    + t_out.abs() + dS.abs()
                                    + 2 * t_ab.abs(),
                                    n_terms=8, eps=2.220446049250313e-16))
        return Proposal(connector=self.name, t0=context.t, t1=context.t + dt,
                        state=(a1, b1), exports=dict(outflow=t_out,
                                                     store_b=b1),
                        balance=bal, backend=str(self.a.device),
                        series=dict(a=a1.unsqueeze(0), b=b1.unsqueeze(0)))

    def validate(self, proposal, context):
        if self.reject_at is not None and context.t >= float(self.reject_at):
            return ['%s: told to reject at t = %g s' % (self.name, context.t)]
        a1, b1 = proposal.state
        if bool((a1 < 0).any()) or bool((b1 < 0).any()):
            return ['%s: a store went negative' % self.name]
        return []

    def commit(self, proposal, context):
        self.a, self.b = proposal.state
        self.t = proposal.t1

    def save_state(self):
        return dict(a=self.a.cpu(), b=self.b.cpu(), t=self.t)

    def load_state(self, blob):
        self.a = blob['a'].to(self.device or 'cpu')
        self.b = blob['b'].to(self.device or 'cpu')
        self.t = blob['t']
