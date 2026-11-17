"""The Connector API: what every module of a coupled TorchVIC run
implements, and what the runtime calls, in this order:

    initialize(context)                  once, before the first step
    prepare_step(context, dt)            start of each of its intervals
    compute(inputs, context, dt)  -> Proposal   the interval's new state and
                                         fluxes, from the ACCEPTED state;
                                         must not change the accepted state
    validate(proposal, context)   -> [problems]  physical checks of its own
    commit(proposal, context)            make the proposal the accepted state
    save_state() / load_state(blob)      checkpoint and restart
    finalize(context)                    once, after the last step

A step is a transaction (torchvic.coupling.runtime): every due connector
computes and validates, the ledger checks the water balance of each and of
the coupled domain, and only then does every connector commit. A connector
whose compute raises, whose validate returns a problem or whose balance does
not close leaves every connector where it was: compute never mutates.

A connector declares (class attributes, collected by `info()`):
    name, version, enabled (config), dependencies (other connectors it
    needs), inputs/outputs (VarSpec: names, units, dims, dtypes, kind,
    grid), grids (GridSpec, from `grids()` once initialised), time
    (TimeSpec: exchange interval), owned_state (the stores it alone writes),
    imported_fluxes / exported_fluxes (which inputs/outputs are water
    transfers in the ledger), backends ('cpu', 'cuda'), checkpoint
    ('full': save_state carries everything a restart needs), validation
    (how far its results have been checked against the model it ports).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .spec import ConnectorInfo, TimeSpec


class ComputeError(Exception):
    """A connector could not compute its interval (the step is rejected)."""


@dataclass
class Balance:
    """A connector's water balance over one interval, as volumes (m3) over
    its whole domain, and optionally per element (`per_cell`, the same terms
    as depths, mm). The ledger closes

        R = P + I_ext - ET - Q_out - E_other - dS - closure

    `closure` is the model's own reported balance error (VIC's
    OUT_WATER_ERROR summed), declared so that a model's known residual is
    not mistaken for a coupling error; `n_terms` is how many values were
    summed into the terms (the tolerance grows with it); `scale` is the sum
    of the terms' magnitudes (the tolerance is relative to it)."""
    P: float = 0.0
    I_ext: float = 0.0
    ET: float = 0.0
    Q_out: float = 0.0
    E_other: float = 0.0
    dS: float = 0.0
    closure: float = 0.0
    n_terms: int = 1
    eps: float = 2.220446049250313e-16
    scale: Optional[float] = None
    per_cell: Optional[Dict[str, Any]] = None
    detail: Dict[str, float] = field(default_factory=dict)

    def residual(self):
        return (self.P + self.I_ext - self.ET - self.Q_out - self.E_other
                - self.dS - self.closure)

    def magnitude(self):
        if self.scale is not None:
            return float(self.scale)
        return (abs(self.P) + abs(self.I_ext) + abs(self.ET)
                + abs(self.Q_out) + abs(self.E_other) + abs(self.dS)
                + abs(self.closure))


@dataclass
class Proposal:
    """What compute returns: the interval's new state (opaque to the
    runtime), its exported values ({output name: tensor}, VarSpec units on
    the output's grid), its water balance, and the per-step outputs for the
    output manager (`series`, e.g. VIC's simulate output)."""
    connector: str
    t0: float
    t1: float
    state: Any = None
    exports: Dict[str, Any] = field(default_factory=dict)
    balance: Optional[Balance] = None
    series: Any = None
    backend: str = ''
    info: Dict[str, Any] = field(default_factory=dict)
    flags: list = field(default_factory=list)       # DeviceFlag, see below


class DeviceFlag(object):
    """A validity check whose answer lives on the device: `bad` is a 0-d
    bool/numeric tensor, true when the check FAILED, `message` the problem
    text. A connector returns these from `validate` (or appends them to
    `Proposal.flags` in compute) instead of reading the tensor itself; the
    runtime brings all flags of a step to the host in ONE copy, together
    with the ledger terms (or once per window with ledger.check_every > 1)."""
    __slots__ = ('message', 'bad', 'values', 'retry')

    def __init__(self, message, bad, values=(), retry=None):
        # `values`: device scalars formatted into `message` (a %-format)
        # when the flag is raised; they travel in the same host copy
        self.message, self.bad, self.values = message, bad, tuple(values)
        # retry: a callable; when this flag alone rejects a step the runtime
        # calls it and runs the step once more (nothing was committed)
        self.retry = retry

    def vector(self):
        import torch
        parts = [torch.as_tensor(self.bad).to(torch.float64).reshape(())]
        parts += [torch.as_tensor(v).to(torch.float64).reshape(())
                  for v in self.values]
        return torch.stack(parts)

    def text(self, host_values):
        """The problem text if raised on the host values `vector` came
        back as, else None."""
        if not host_values[0]:
            return None
        return (self.message % tuple(host_values[1:])
                if self.values else self.message)


@dataclass
class Context:
    """What the runtime hands a connector: the clock, the device and dtype
    it chose, the run's configuration block for this connector, and a log."""
    t: float = 0.0
    t_end: float = 0.0
    step_index: int = 0
    device: Any = None
    dtype: Any = None
    config: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)
    log: Any = None
    base_dir: str = '.'
    # True while the runtime redoes a step a DeviceFlag(retry=...) rejected:
    # a connector with an optimistic (deferred host read) path takes its
    # synchronous one
    redo: bool = False
    # inside a day-coupled window (DayCoupler.day): the window module's
    # soil state at the day's start, read-only, per tile and band
    # (torchvic.vicwur.view.SoilView); None elsewhere
    soil: Any = None

    def say(self, *a):
        if self.log is not None:
            self.log(*a)


class Connector(object):
    """Base class. Subclasses set the declarations and implement the
    methods; the defaults are the no-op a stateless connector needs."""
    name = 'connector'
    version = '0'
    inputs = ()
    outputs = ()
    dependencies = ()
    owned_state = ()
    imported_fluxes = ()
    exported_fluxes = ()
    backends = ('cpu',)
    checkpoint = 'full'
    validation = 'unvalidated'
    #: the exchange interval (s); 0: the connector sets it in initialize
    interval_s = 0.0

    def __init__(self, **config):
        self.config = dict(config)
        self.enabled = bool(self.config.pop('enabled', True))

    # -- declarations -------------------------------------------------------
    def info(self):
        return ConnectorInfo(
            name=self.name, version=str(self.version),
            inputs=tuple(self.inputs), outputs=tuple(self.outputs),
            grids=tuple(g.name for g in self.grids()),
            time=TimeSpec(interval_s=float(self.interval_s)),
            owned_state=tuple(self.owned_state),
            imported_fluxes=tuple(self.imported_fluxes),
            exported_fluxes=tuple(self.exported_fluxes),
            backends=tuple(self.backends), checkpoint=self.checkpoint,
            validation=self.validation,
            dependencies=tuple(self.dependencies))

    def grids(self):
        """[GridSpec] the connector's variables live on (known once
        initialised)."""
        return []

    def time_span(self):
        """(start, end) in seconds of the run, when the connector defines
        the run's length (VIC's simulation window); else None."""
        return None

    # -- life cycle ---------------------------------------------------------
    def initialize(self, context):
        pass

    def prepare_step(self, context, dt):
        pass

    def compute(self, inputs, context, dt):
        raise NotImplementedError

    def validate(self, proposal, context):
        """Problems with `proposal` ([] when none)."""
        return []

    def commit(self, proposal, context):
        raise NotImplementedError

    def save_state(self):
        return None

    def load_state(self, blob):
        pass

    def finalize(self, context):
        pass

    # -- conveniences -------------------------------------------------------
    def __repr__(self):
        return '<%s %s %s>' % (type(self).__name__, self.name,
                               'enabled' if self.enabled else 'disabled')


def problems_of(result):
    """validate's return as a list of strings."""
    if not result:
        return []
    if isinstance(result, str):
        return [result]
    if isinstance(result, DeviceFlag):
        return []
    return [str(x) for x in result if not isinstance(x, DeviceFlag)]


def flags_of(result):
    """The DeviceFlags in validate's return."""
    if isinstance(result, DeviceFlag):
        return [result]
    if not result or isinstance(result, str):
        return []
    return [x for x in result if isinstance(x, DeviceFlag)]


def resolve_problems(result):
    """validate's return as strings, device flags read on the host (one
    copy). For tests and standalone drivers; the runtime batches them with
    the ledger."""
    from .hostfetch import Fetch
    F = Fetch()
    hs = [(f, F.add(f.vector())) for f in flags_of(result)]
    F.run()
    return problems_of(result) + [t for t in (f.text(h.value())
                                              for f, h in hs)
                                  if t is not None]


__all__ = ['Connector', 'Context', 'Proposal', 'Balance', 'ComputeError',
           'problems_of', 'flags_of', 'DeviceFlag', 'resolve_problems']
