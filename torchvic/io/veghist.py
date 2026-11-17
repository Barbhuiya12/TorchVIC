"""The vegetation history forcing: FORCING2 with LAI_SRC, FCAN_SRC and
ALB_SRC FROM_VEGHIST (VIC 5.1.0 image driver).

What VIC does (vic/drivers/image/src, line numbers are VIC 5.1.0's):

* get_global_param.c:334-343 reads `FORCING2 <prefix>` and the FORCE_TYPE
  lines after it; with none, the file counts as unused (forcing_utils.c:
  108-148). :784-791 opens `<prefix><year>.nc`.
* vic_force.c:241-262 first sets every sub-step's albedo, displacement,
  fcanopy, LAI and roughness to the parameter file's value for the month of
  the model step.
* vic_force.c:264-363 then, for each source that is FROM_VEGHIST, overwrites
  the sub-step's value with the variable `LAI`, `fcanopy` or `albedo` of
  the second file, dimensions (time, veg_class, y, x), one value per
  forcing record and tile. Displacement and roughness are never read.
* vic_force.c:419-439 sets fcanopy and LAI to 0 in any sub-step with
  fcanopy < MIN_FCANOPY or LAI == 0.
* vic_force.c:466-474 averages the sub-steps into the model step's value,
  which update_step_vars.c:35-44 hands to every tile and band.

So the file replaces the parameter file's monthly curve with a value per
model step. TorchVIC indexes its vegetation fields by `months[t]`; here the
five fields become [cell, tile, step] and the step loop indexes them by step
(`VegMosaic.step_axis`). A step is one forcing record (a sub-daily model
step) or the mean of the day's records (a daily model step, as the C averages
its NF sub-steps).

FORCING2 cannot carry extra meteorological variables in the image driver:
every met read uses filenames.forcing[0] (vic_force.c:62-205). Its FORCE_TYPE
lines only make VIC open the file, so they are read here the same way, to
decide whether the file is in use, and nothing else is taken from it.
"""
import os

import torch

from . import build as _build
from ..core import version as core_version

#: The *_SRC option that makes each NetCDF variable (named literally,
#: vic_force.c:309, :328, :348) come from the file, and the mosaic field.
SOURCES = (('LAI_SRC', 'LAI', 'lai'),
           ('FCAN_SRC', 'fcanopy', 'fcanopy'),
           ('ALB_SRC', 'albedo', 'albedo'))


def wanted(gp):
    """{mosaic field: NetCDF variable} for the *_SRC set to FROM_VEGHIST."""
    out = {}
    for opt, var, field in SOURCES:
        v = gp.get(opt)
        if v is not None and str(v).upper() == 'FROM_VEGHIST':
            out[field] = var
    return out


def _key(t):
    """(y, m, d, H, M, S) of a numpy, pandas, datetime or cftime stamp."""
    if hasattr(t, 'year'):
        return (t.year, t.month, t.day, getattr(t, 'hour', 0),
                getattr(t, 'minute', 0), getattr(t, 'second', 0))
    import pandas as pd
    return _key(pd.Timestamp(t))


def read(prefix, domain, variables, years, dtype=torch.float64, device=None,
         log=None):
    """The history of `years` as ({field: [record, tile, cell]}, [stamp]).

    `variables` is {mosaic field: NetCDF variable}. One file a year,
    `<prefix><year>.nc` (get_global_param.c:785, vic_force.c:284)."""
    import numpy as np
    import xarray as xr
    out, stamps = {f: [] for f in variables}, []
    for y in years:
        path = '%s%04d.nc' % (prefix, y)
        if not os.path.exists(path):
            raise IOError('FORCING2: no vegetation history file %s (VIC '
                          'opens <prefix><year>.nc, get_global_param.c:785)'
                          % path)
        from .ncshare import read
        with read(path) as ds:
            for field, var in variables.items():
                if var not in ds:
                    raise ValueError(
                        '%s has no variable %r, which %s FROM_VEGHIST reads '
                        '(vic_force.c)' % (path, var, {
                            'lai': 'LAI_SRC', 'fcanopy': 'FCAN_SRC',
                            'albedo': 'ALB_SRC'}[field]))
                da = ds[var]
                if da.ndim != 4:
                    raise ValueError('%s: %s has dimensions %s; VIC reads '
                                     '(time, veg_class, y, x)'
                                     % (path, var, da.dims))
                domain.check_dims(da, what='%s in %s' % (var, path))
                v = np.asarray(da.values[..., domain.rows, domain.cols],
                               dtype=np.float64)       # t, class, cell
                out[field].append(v)
            stamps.extend(ds['time'].values)
        if log:
            log('  %s  %d records' % (os.path.basename(path),
                                      len(ds['time'])))
    arr = {f: torch.as_tensor(np.concatenate(v, axis=0), dtype=dtype,
                              device=device) for f, v in out.items()}
    return arr, stamps


def record_steps(stamps, dates, daily):
    """For each history record, the model step it belongs to (-1: none).

    A sub-daily step takes the record stamped at its start; a daily step the
    mean of the records of its calendar day, VIC's NF sub-steps."""
    if daily:
        step = {_key(d)[:3]: i for i, d in enumerate(dates)}
        got = [step.get(_key(s)[:3], -1) for s in stamps]
    else:
        step = {_key(d): i for i, d in enumerate(dates)}
        got = [step.get(_key(s), -1) for s in stamps]
    return got


def apply(mosaic, hist, stamps, dates, daily, dtype=None, log=None):
    """Replace the mosaic's monthly lai, fcanopy and albedo with the
    history's per-step values (and spread displacement and roughness to the
    same axis). `mosaic` must come from build_mosaic(keep_raw=True)."""
    if not core_version.at_least('5.1.0'):
        raise ValueError('the vegetation history is implemented for the VIC '
                         '5.1.0 core (vic_force.c 5.1.0:241-439); the 5.0.1 '
                         'core floors fcanopy instead (5.0.1:386-394)')
    raw = getattr(mosaic, 'raw_monthly', None)
    if raw is None:
        raise ValueError('apply needs the mosaic built with keep_raw=True')
    lai0, fc0 = raw
    del mosaic.raw_monthly
    dev = lai0.device
    dtype = lai0.dtype
    nt = len(dates)
    months = torch.as_tensor([d.month - 1 for d in dates], device=dev)
    g = torch.as_tensor(record_steps(stamps, dates, daily), device=dev)
    keep = g >= 0
    n_rec = int(keep.sum())
    counts = torch.bincount(g[keep], minlength=nt)
    if bool((counts == 0).any()):
        first = int((counts == 0).nonzero()[0])
        raise ValueError('FORCING2 has no record for model step %d (%s): '
                         'VIC reads one per forcing record from the same '
                         'start (get_global_param.c:794-800)'
                         % (first, dates[first]))
    if not bool(daily) and bool((counts != 1).any()):
        raise ValueError('FORCING2 has more than one record for a model step')
    gk = g[keep]
    mk = months[gk]
    covered = (mosaic.cv > 0.0).unsqueeze(-1)       # [cell, tile, 1]

    def rec(base, field):
        """[cell, tile, record]: the file's value, or the month's."""
        b = base.index_select(2, mk)
        h = hist.get(field)
        if h is None:
            return b
        f = h[keep].permute(2, 1, 0)                # cell, tile, record
        if f.shape[:2] != b.shape[:2]:
            raise ValueError('FORCING2 %s has %d classes over %d cells; the '
                             'parameter file has %d tiles over %d cells'
                             % (field, f.shape[1], f.shape[2],
                                b.shape[1], b.shape[0]))
        return torch.where(covered, f, b)

    lai_r = rec(lai0, 'lai')
    fc_r = rec(fc0, 'fcanopy')
    alb_r = rec(mosaic.albedo, 'albedo') if mosaic.albedo is not None \
        else None
    # vic_force.c:419-439, per record, before the average
    off = (fc_r < _build.MIN_FCANOPY) | (lai_r == 0.0)
    zero = torch.zeros_like(lai_r)
    lai_r = torch.where(off, zero, lai_r)
    fc_r = torch.where(off, zero, fc_r)

    def mean(x):
        s = torch.zeros(x.shape[:2] + (nt,), dtype=dtype, device=dev)
        s.index_add_(2, gk, x)
        return s / counts.to(dtype)

    mosaic.lai = mean(lai_r)
    mosaic.fcanopy = mean(fc_r)
    if alb_r is not None:
        mosaic.albedo = _build._bare_albedo(mean(alb_r), mosaic.bare_tile)
    for name in ('displacement', 'veg_rough'):
        x = getattr(mosaic, name)
        if x is not None:
            setattr(mosaic, name, x.index_select(2, months))
    mosaic.step_axis = True
    if log:
        log('vegetation history: %s per step from FORCING2 (%d records, %d '
            'steps); the rest follow the parameter file\'s month'
            % (', '.join(sorted(hist)), n_rec, nt))
    return mosaic
