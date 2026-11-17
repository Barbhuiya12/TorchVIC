"""VIC 5 inputs and outputs: the global parameter file, the image driver's
domain, parameter and forcing files, the classic driver's ASCII files, the
model inputs built from them (`build`), and VIC-style history files.
"""
from .domain import ImageDomain
from .forcing import ImageForcing, aggregate_to_daily
from .global_param import GlobalParam
from .params import ImageParams

__all__ = ['GlobalParam', 'ImageDomain', 'ImageParams', 'ImageForcing',
           'aggregate_to_daily']
