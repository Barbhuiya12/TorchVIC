# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR WOFOST-NPK crop kernel (C12; docs/connectors/vicwur.md): a batched
port of the C WOFOST-NPK that VIC-WUR couples to VIC.
See kernel.py for the C references, the precision modes and the list of
differences to WOFOST 7.2 (torchvic/crop/wofost72.py)."""
from .params import (CropTemplates, UnitSetup, Calendar, setup_from_vicwur,
                     dmy_doy, SCALARS, TABLES, MNG_SCALARS, MNG_TABLES)
from .kernel import WofostNPK, ERR_CC, ERR_CV, ERR_NOLEAF, ERR_RING, ERR_LAT

__all__ = ['CropTemplates', 'UnitSetup', 'Calendar', 'setup_from_vicwur',
           'dmy_doy', 'WofostNPK', 'SCALARS', 'TABLES', 'MNG_SCALARS',
           'MNG_TABLES', 'ERR_CC', 'ERR_CV', 'ERR_NOLEAF', 'ERR_RING',
           'ERR_LAT']
