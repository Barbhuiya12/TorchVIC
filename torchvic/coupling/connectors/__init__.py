"""Connectors of the coupled TorchVIC system built on the runtime.

River system: `routing`, `reservoir`, `quality` (docs/connectors/river.md).
Water use and land: `irrigation` (wateruse), `crop` (WOFOST 7.2), `groundwater`.
Flood: `flood` (TorchVIC GPU flood engine; LISFLOOD-FP ACC as an external process with backend: lisflood).
Each is registered in `torchvic.coupling.registry`.
"""
from .routing import RoutingConnector, SweepPlugin
from .reservoir import ReservoirConnector
from .quality import QualityConnector

__all__ = ['RoutingConnector', 'ReservoirConnector', 'QualityConnector',
           'SweepPlugin']
