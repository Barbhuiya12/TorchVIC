"""Groundwater engine: the MODFLOW 6 GWF equations on the GPU (default
backend ``torch``) or MODFLOW 6 itself in-process (backend ``mf6``)."""
from .grid import Grid, R_EARTH
from .model import GroundwaterModel, GroundwaterNotConverged, StepResult

__all__ = ['Grid', 'R_EARTH', 'GroundwaterModel', 'GroundwaterNotConverged',
           'StepResult']
