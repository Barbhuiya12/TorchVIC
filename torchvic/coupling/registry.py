"""The connectors a configuration can name, and why the ones it cannot use
are unavailable.

    from torchvic.coupling import registry
    registry.register('mymodel', MyConnector)          # a new connector
    registry.available()                               # what can run

An entry is 'available' (it runs and has tests), 'test' (a connector for the
framework's own tests, conservative by construction) or 'planned' (named but
not implemented: a configuration that enables it is refused with the
reason). Nothing planned is presented as implemented.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from .spec import CouplingError


@dataclass
class Entry:
    name: str
    factory: Optional[Callable]
    status: str
    reason: str = ''

    def make(self, **config):
        if self.factory is None:
            raise CouplingError('connector %r is %s: %s'
                                % (self.name, self.status, self.reason))
        return self.factory(**config)


_REGISTRY = {}


def register(name, factory, status='available', reason='', replace=False):
    if status not in ('available', 'test', 'planned'):
        raise ValueError('status must be available, test or planned')
    if name in _REGISTRY and not replace:
        raise CouplingError('connector %r is registered already' % name)
    _REGISTRY[name] = Entry(name, factory, status, reason)
    return _REGISTRY[name]


def unregister(name):
    _REGISTRY.pop(name, None)


def get(name):
    try:
        return _REGISTRY[name]
    except KeyError:
        raise CouplingError(
            'no connector %r (registered: %s)'
            % (name, ', '.join('%s [%s]' % (k, e.status)
                               for k, e in sorted(_REGISTRY.items()))))


def entries():
    return dict(_REGISTRY)


def available():
    return sorted(k for k, e in _REGISTRY.items() if e.factory is not None)


def _vic(**cfg):
    from .vic import VicConnector
    return VicConnector(**cfg)


def _two_store(**cfg):
    from .toy import TwoStoreConnector
    return TwoStoreConnector(**cfg)


def _flood(**cfg):
    from .connectors.flood import FloodConnector
    return FloodConnector(**cfg)


def _routing(**cfg):
    from .connectors.routing import RoutingConnector
    return RoutingConnector(**cfg)


def _reservoir(**cfg):
    from .connectors.reservoir import ReservoirConnector
    return ReservoirConnector(**cfg)


def _quality(**cfg):
    from .connectors.quality import QualityConnector
    return QualityConnector(**cfg)


def _irrigation(**cfg):
    from .connectors.wateruse import WaterUseConnector
    return WaterUseConnector(**cfg)


def _crop(**cfg):
    from .connectors.crop import CropConnector
    return CropConnector(**cfg)


def _groundwater(**cfg):
    from .connectors.groundwater import (GroundwaterConnector,
                                         VicWur3GroundwaterConnector)
    if cfg.get('profile') == 'vicwur3':
        # VIC-WUR 3.0's MODFLOW coupling, a component of the VIC-WUR phase
        # group (docs/connectors/vicwur.md)
        return VicWur3GroundwaterConnector(**cfg)
    return GroundwaterConnector(**cfg)


def _split(**cfg):
    from .connectors.split import SplitConnector
    return SplitConnector(**cfg)


register('vic', _vic, 'available')
register('split', _split, 'available')
register('routing', _routing, 'available')
register('reservoir', _reservoir, 'available')
register('quality', _quality, 'available')
register('groundwater', _groundwater, 'available')
register('flood', _flood, 'available')
register('two_store', _two_store, 'test',
         'a conservative two-store test connector (torchvic.coupling.toy)')
register('irrigation', _irrigation, 'available')
register('crop', _crop, 'available')


# -- VIC-WUR connectors (docs/connectors/vicwur.md): components of one phase
# group (torchvic.coupling.connectors.vicwur_common.WurGroup), run in
# VIC-WUR's order between the days of a VIC window (irrigation, CO2, land
# use and crop also act inside vic_run through the group's VicwurCore). The
# lumped 'irrigation', 'reservoir' and 'crop' connectors above stay as they
# are.
def _vicwur(mod, cls):
    def make(**cfg):
        import importlib
        m = importlib.import_module('.connectors.' + mod, __package__)
        return getattr(m, cls)(**cfg)
    return make


for _n, _m, _c in (
        ('vicwur_routing', 'vicwur_routing', 'VicWurRoutingConnector'),
        ('vicwur_dams', 'vicwur_dams', 'VicWurDamsConnector'),
        ('vicwur_wateruse_local', 'vicwur_wateruse_local',
         'VicWurWateruseLocalConnector'),
        ('vicwur_wateruse_remote', 'vicwur_wateruse_remote',
         'VicWurWateruseRemoteConnector'),
        ('vicwur_nonrenew', 'vicwur_nonrenew', 'VicWurNonrenewConnector'),
        ('vicwur_efr', 'vicwur_efr', 'VicWurEfrConnector'),
        ('vicwur_irrigation', 'vicwur_irrigation',
         'VicWurIrrigationConnector'),
        ('vicwur_co2', 'vicwur_co2', 'VicWurCo2Connector'),
        ('vicwur_landuse', 'vicwur_landuse', 'VicWurLanduseConnector'),
        ('vicwur_crop', 'vicwur_crop', 'VicWurCropConnector'),
        ('vicwur_cropsyst', 'vicwur_cropsyst', 'VicWurCropSystConnector')):
    register(_n, _vicwur(_m, _c), 'available')
