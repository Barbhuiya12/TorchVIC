"""The coupled TorchVIC system: a Connector API every module implements, a
runtime that advances the modules together in transactional steps, and a
water ledger that rejects a step whose water does not add up.

    from torchvic import TorchVIC
    m = TorchVIC.from_config('configs/vic_only.yaml').initialize()
    m.run()
    m.finalize()

See docs/ARCHITECTURE.md, docs/CONNECTOR_API.md, docs/CONFIGURATION.md and
docs/WATER_ACCOUNTING.md.
"""
from .spec import CouplingError, GridSpec, TimeSpec, VarSpec
from .connector import Balance, ComputeError, Connector, Context, Proposal
from .ledger import Ledger
from .runtime import Runtime, StepRejected, TorchVIC, validate_config
from .schema import ConfigError
from . import registry

__all__ = ['TorchVIC', 'Runtime', 'Connector', 'Context', 'Proposal',
           'Balance', 'VarSpec', 'GridSpec', 'TimeSpec', 'Ledger',
           'CouplingError', 'ComputeError', 'ConfigError', 'StepRejected',
           'validate_config', 'registry']
