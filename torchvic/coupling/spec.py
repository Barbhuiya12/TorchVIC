"""What a connector declares about itself: its variables (name, units,
dimensions, dtype), the grids it lives on, its time step, its backends and
how far it has been validated.

Nothing here computes. The runtime reads these declarations at
initialisation and refuses a configuration whose connections do not agree
(unit mismatch, grid mismatch, a missing dependency) before any model runs.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


class CouplingError(Exception):
    """A configuration or a declaration the runtime cannot run."""


# ---------------------------------------------------------------------------
# Units. Exchanged quantities are water amounts or states; a connection is
# accepted when the two sides name the same unit or units that differ by a
# constant factor listed here. Depth <-> volume needs an area and is the
# spatial mapper's job (SpatialMapper), not a unit conversion.
# ---------------------------------------------------------------------------
#: unit -> (dimension, factor to the dimension's base unit)
UNITS = {
    # water depth over a cell's land area, per exchange interval
    'mm': ('depth', 1e-3), 'm': ('depth', 1.0),
    'kg m-2': ('depth', 1e-3),            # 1 kg of water over 1 m2 = 1 mm
    # volume per exchange interval
    'm3': ('volume', 1.0), 'L': ('volume', 1e-3), 'hm3': ('volume', 1e6),
    # rates
    'mm s-1': ('depth_rate', 1e-3), 'm s-1': ('depth_rate', 1.0),
    'mm day-1': ('depth_rate', 1e-3 / 86400.0),
    'm3 s-1': ('volume_rate', 1.0),
    # states and diagnostics that are exchanged without conversion
    '1': ('fraction', 1.0), 'C': ('temperature_c', 1.0),
    'K': ('temperature_k', 1.0), 'W m-2': ('energy_flux', 1.0),
}


def unit_factor(src, dst):
    """The factor taking a value in `src` to `dst`; CouplingError when the
    two are not the same dimension (or either is unknown and they differ)."""
    if src == dst:
        return 1.0
    a, b = UNITS.get(src), UNITS.get(dst)
    if a is None or b is None or a[0] != b[0]:
        raise CouplingError(
            'unit mismatch: %r cannot be converted to %r%s'
            % (src, dst, '' if (a and b) else
               ' (known units: %s)' % ', '.join(sorted(UNITS))))
    return a[1] / b[1]


def unit_dimension(u):
    d = UNITS.get(u)
    return d[0] if d else None


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VarSpec:
    """One exchanged variable.

    kind: 'flux'  an amount over the producer's exchange interval (summed
                  when the consumer's interval is longer, prorated when
                  shorter -- conservative in total);
          'state' a value at the end of the interval (sampled, never
                  summed; not part of the water ledger).
    dims: names of the dimensions, the first being the grid's ('cell',
          'reach', ...); `grid` names the GridSpec it lives on.
    """
    name: str
    units: str
    dims: Tuple[str, ...] = ('cell',)
    dtype: str = 'float64'
    kind: str = 'flux'
    grid: str = ''
    description: str = ''

    def __post_init__(self):
        if self.kind not in ('flux', 'state'):
            raise CouplingError('%s: kind must be flux or state, not %r'
                                % (self.name, self.kind))


@dataclass
class GridSpec:
    """A set of spatial units: `n` elements with an area each (m2; the
    land area water depths are expressed over -- for VIC, the cell area
    times its land fraction)."""
    name: str
    n: int
    area: Optional[torch.Tensor] = None
    kind: str = 'cells'
    description: str = ''

    def area_tensor(self, dtype=torch.float64, device=None):
        if self.area is None:
            raise CouplingError('grid %r declares no areas, so depths on it '
                                'cannot become volumes' % self.name)
        # cached per (dtype, device): a host-to-device copy per call would
        # synchronise the stream every time an amount becomes a volume
        c = self.__dict__.setdefault('_area_cache', {})
        k = (dtype, None if device is None else torch.device(device))
        a = c.get(k)
        if a is None or a[0] is not self.area:
            a = c[k] = (self.area, torch.as_tensor(
                self.area, dtype=dtype, device=device))
        return a[1]


@dataclass
class TimeSpec:
    """The connector's exchange interval in seconds (0: set from the
    connector's own data at initialisation, e.g. VIC's windows), and
    whether it can run at a shorter interval than its natural one."""
    interval_s: float = 0.0
    model_step_s: float = 0.0
    flexible: bool = True


@dataclass
class ConnectorInfo:
    """Everything a connector declares (see Connector)."""
    name: str
    version: str
    inputs: Tuple[VarSpec, ...] = ()
    outputs: Tuple[VarSpec, ...] = ()
    grids: Tuple[str, ...] = ()
    time: TimeSpec = field(default_factory=TimeSpec)
    owned_state: Tuple[str, ...] = ()
    imported_fluxes: Tuple[str, ...] = ()
    exported_fluxes: Tuple[str, ...] = ()
    backends: Tuple[str, ...] = ('cpu',)
    checkpoint: str = 'full'
    validation: str = 'unvalidated'
    dependencies: Tuple[str, ...] = ()

    def output(self, name):
        for v in self.outputs:
            if v.name == name:
                return v
        raise CouplingError('%s has no output %r (it declares %s)'
                            % (self.name, name,
                               ', '.join(v.name for v in self.outputs)
                               or 'none'))

    def input(self, name):
        for v in self.inputs:
            if v.name == name:
                return v
        raise CouplingError('%s has no input %r (it declares %s)'
                            % (self.name, name,
                               ', '.join(v.name for v in self.inputs)
                               or 'none'))
