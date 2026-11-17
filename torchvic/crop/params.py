# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
#
# Parameter names and meanings are those of WOFOST 7.2 as implemented in PCSE
# (Copyright (c) 2004-2024 Wageningen Environmental Research, Wageningen-UR,
# Allard de Wit; EUPL-1.2), see torchvic/crop/NOTICE_PCSE.txt.
"""WOFOST 7.2 parameters as per-cell tensors.

A parameter set is a plain dict of the names used by PCSE and the WOFOST
crop-parameter files (https://github.com/ajwdewit/WOFOST_crop_parameters,
branch `wofost72`): scalars are numbers, AFGEN tables are flat lists
`[x0, y0, x1, y1, ...]`. Soil and site parameters (SMFCF, SM0, SMW, CRAIRC,
RDMSOL, SOPE, KSUB, ...) sit in the same dict, as in PCSE's ParameterProvider.
"""
import torch

from .afgen import Afgen, afgen_py

TABLES = ('DTSMTB', 'VERNRTB', 'FRTB', 'FLTB', 'FSTB', 'FOTB', 'AMAXTB', 'EFFTB',
          'KDIFTB', 'TMPFTB', 'TMNFTB', 'RFSETB', 'RDRSTB', 'SSATB', 'RDRRTB', 'SLATB')

SCALARS = ('TSUMEM', 'TBASEM', 'TEFFMX', 'TSUM1', 'TSUM2', 'IDSL', 'DLO', 'DLC', 'DVSI',
           'DVSEND', 'VERNSAT', 'VERNBASE', 'VERNDVS', 'TDWI', 'CVL', 'CVO', 'CVR', 'CVS',
           'Q10', 'RMR', 'RML', 'RMS', 'RMO', 'RDI', 'RRI', 'RDMCR', 'RDMSOL', 'IAIRDU',
           'SPA', 'RGRLAI', 'SPAN', 'TBASE', 'PERDL', 'CFET', 'DEPNR', 'IOX', 'CRAIRC',
           'SM0', 'SMW', 'SMFCF')

#: only needed by the classic water balance (validation mode)
SOIL_SCALARS = ('SOPE', 'KSUB', 'IFUNRN', 'SSMAX', 'SSI', 'WAV', 'NOTINF', 'SMLIM')

_DEFAULTS = {
    'VERNSAT': 1.0, 'VERNBASE': 0.0, 'VERNDVS': 0.3, 'DLO': 1.0, 'DLC': 0.0, 'IDSL': 0.0,
    'VERNRTB': [0.0, 0.0, 1.0, 0.0], 'IAIRDU': 0.0, 'IOX': 0.0, 'CRAIRC': 0.05,
}

#: crop start / end types
START_SOWING, START_EMERGENCE = 0, 1
END_MATURITY, END_HARVEST, END_EARLIEST = 0, 1, 2


def _get(d, name):
    if name in d:
        return d[name]
    if name in _DEFAULTS:
        return _DEFAULTS[name]
    raise KeyError('WOFOST parameter %s missing' % name)


class CropParams:
    """Per-cell parameter tensors. Build with `from_sets`.

    `sets` is a list of parameter dicts, `index` [N] gives the set of each
    cell (default: cell i uses set i if there are N sets, else set 0).
    Each attribute is a [N] tensor (scalars) or an `Afgen` with N rows.
    """

    def __init__(self):
        self.n = 0
        self.t = {}
        self.sc = {}
        self.has_soil = False

    @classmethod
    def from_sets(cls, sets, index=None, n=None, device='cpu', dtype=torch.float64,
                  start_type='emergence', end_type='maturity'):
        self = cls()
        sets = list(sets)
        if index is None:
            n = n if n is not None else len(sets)
            index = torch.zeros(n, dtype=torch.long) if len(sets) == 1 else torch.arange(n)
        index = torch.as_tensor(index, dtype=torch.long).to(device)
        self.n = int(index.numel())
        for name in SCALARS:
            vals = [float(_get(s, name)) for s in sets]
            self.sc[name] = torch.tensor(vals, dtype=dtype, device=device)[index]
        for name in TABLES:
            tabs = [list(_get(s, name)) for s in sets]
            self.t[name] = Afgen.from_lists(tabs, device, dtype).expand(index)
        # CO2 effect on potential transpiration (EvapotranspirationCO2); 1 if absent
        rf = []
        for s in sets:
            if 'CO2' in s and 'CO2TRATB' in s:
                rf.append(afgen_py(s['CO2TRATB'], s['CO2']))
            else:
                rf.append(1.0)
        self.sc['RFTRAMXCO2'] = torch.tensor(rf, dtype=dtype, device=device)[index]
        if all(k in s for s in sets for k in SOIL_SCALARS):
            self.has_soil = True
            for name in SOIL_SCALARS:
                self.sc[name] = torch.tensor([float(s[name]) for s in sets], dtype=dtype,
                                             device=device)[index]
        self.start_type = _per_cell(start_type, ('sowing', 'emergence'), sets, 'CROP_START_TYPE',
                                    index, device)
        self.end_type = _per_cell(end_type, ('maturity', 'harvest', 'earliest'), sets,
                                  'CROP_END_TYPE', index, device)
        self.device = device
        self.dtype = dtype
        return self

    def __getattr__(self, name):
        if name in ('t', 'sc'):
            raise AttributeError(name)
        if name in self.sc:
            return self.sc[name]
        if name in self.t:
            return self.t[name]
        raise AttributeError(name)


def _per_cell(default, choices, sets, key, index, device):
    vals = []
    for s in sets:
        v = s.get(key, default)
        vals.append(choices.index(v))
    return torch.tensor(vals, dtype=torch.long, device=device)[index]


def load_wofost_crop_yaml(path, variety):
    """Read one variety from a WOFOST crop-parameter yaml file as a flat dict
    (what PCSE's YAMLCropDataProvider hands to the model). Needs PyYAML."""
    import yaml
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    var = doc['CropParameters']['Varieties'][variety]
    return {k: v[0] for k, v in var.items() if k != 'Metadata'}
