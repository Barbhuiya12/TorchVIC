"""Diagnostic output of the compile and capture machinery (core.compact,
core.fusion, core.graphsafe).

The messages go to the `torchvic.core.diag` logger at DEBUG level. They are
silent unless one of the TORCHVIC_* diagnostic variables below is set, which
sends them to stderr; an application can also configure the logger itself.
"""
import logging
import os
import sys

_FLAGS = ('TORCHVIC_COMPACT_DEBUG', 'TORCHVIC_COMPACT_LOG',
          'TORCHVIC_FUSION_CHECK', 'TORCHVIC_FUSION_COUNT',
          'TORCHVIC_FUSION_GUARDS', 'TORCHVIC_HOIST_PROBE',
          'TORCHVIC_SEG_PROF')

log = logging.getLogger('torchvic.core.diag')

if any(os.environ.get(k, '0') not in ('', '0') for k in _FLAGS) \
        and not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(_h)
    log.setLevel(logging.DEBUG)
    log.propagate = False
