# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.efr replayed on the R1 G-runs that use EFR (VICWUR_DUMP).

efr_derived_forcing runs in the F phase of step t on the START-of-step
moisture, which is the dump of step t-1 (the dump is written at the end of
step t-1, after irrigation in layer 0 and pumping in the last layer).  With
the efr_q / efr_b forcing held per month, the kernel must reproduce, from the
soil file and the dump alone:

  1. OUT_EFR_MOIST (the cell sum of efr_moist * Cv * AreaFract, float32 in
     the netCDF output), OUT_EFR_BASEFLOW and OUT_EFR_DISCHARGE, every cell
     and step;
  2. the groundwater pumping of step t in double precision.  Pumping takes
     W = sum_s withdrawn_gw from the tiles in proportion to their
     availability x_t = (moist - efr_moist - resid - ice) * AF * Cv
     (wu_run_local.c:109-126, 398-419), so y_t = x_t - pump_t = x_t (1-W/A)
     can be rebuilt from the dumped end-of-step moisture and the kernel's
     efr_moist: A = W + sum_t max(y_t, 0) must not be below W, must equal
     the pool where the demand exhausted it (then every pumped tile has
     y_t = 0, a per-tile check of efr_moist to 1e-9), and never exceed
     sum_s available_gw when the pool was not exhausted.

Runs: G7_dump (the shipped R1 EFR forcing, <data root>/verification/vicwur_16)
and the G7 variants with the baseflow requirement scaled (G7s*), which move
the grid search through the interior of the 100 points.  The G10 dump (land
use on) is replayed on the steps where no Cv record was read (lu_apply is
tested in tests/test_vicwur_landuse.py).

Coverage note: the bottom layer of R1 holds about 8 % of saturation, so
rel_liq stays below Ws and only the linear baseflow term of the C is reached;
the non-linear term and the cell-0 quirk are covered by the oracle test
(tests/test_vicwur_efr.py) only.
"""
import os
import _paths  # noqa: E402
import struct

import netCDF4
import numpy as np
import pytest
import torch

from torchvic.io import ncshare
from torchvic.vicwur import efr as E
from torchvic.vicwur import forcing as F
from torchvic.vicwur import params as P

ROOT = _paths.env_path(('TORCHVIC_VICWUR_16_GRUNS', 'VICWUR_16_GRUNS'), 'verification/vicwur_16/gruns')
G10 = (_paths.data('verification/vicwur_03/gruns/R1/G10_dump'))
RUNS = {'G7_dump': ROOT + '/G7_dump', 'G7s1': ROOT + '/G7s1',
        'G7s2': ROOT + '/G7s2', 'G7s3': ROOT + '/G7s3', 'G10': G10}
START = (2003, 7, 1, 0)


def read_dump(path):
    out = {2: [], 3: [], 5: []}
    dims = None
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            tag, n = struct.unpack('<iq', h)
            a = np.frombuffer(f.read(8 * n), '<f8')
            if tag == 0:
                dims = a.astype(int)
            elif tag in out:
                out[tag].append(a.copy())
    return dims, {k: np.stack(v) for k, v in out.items()}


def soil_params(g, grid, nl):
    with netCDF4.Dataset(g['PARAMETERS'][0][0]) as ds:   # closed at once
        raw = {k: np.array(ds[k][:], dtype=np.float64) for k in (
            'depth', 'bulk_density', 'soil_density', 'resid_moist', 'Dsmax',
            'Ds', 'Ws', 'c', 'AreaFract')}
    f = lambda k: grid.take(raw[k])
    depth = f('depth')[nl - 1]
    mm = depth * (1 - f('bulk_density')[nl - 1] / f('soil_density')[nl - 1]) \
        * 1000.
    return dict(depth=depth, max_moist=mm, resid=f('resid_moist')[nl - 1],
                Dsmax=f('Dsmax'), Ds=f('Ds'), Ws=f('Ws'), c=f('c'),
                af=f('AreaFract').T)


def read_out(run_dir, var):
    import glob
    parts = []
    for p in sorted(glob.glob(run_dir + '/W3/out/fluxes.*.nc')):
        ds = ncshare.read(p, decode_times=False)
        parts.append(np.asarray(ds[var].values, dtype=np.float64))
    a = np.concatenate(parts)
    return a.reshape(len(a), -1)


def replay(run, bug=True):
    """Replay of one run: the kernel against the C outputs of that run."""
    rd = RUNS[run]
    dump = os.path.join(rd, 'dump.bin')
    if not os.path.exists(dump):
        pytest.skip('dump absent: ' + dump)
    gpath = os.path.join(rd, 'global_W3.txt')
    g = P.parse_global(gpath)
    grid = P.Grid(g['DOMAIN'][0][0], g['PARAMETERS'][0][0])
    dims, rec = read_dump(dump)
    N, NT, NB, NL, NF = [int(x) for x in dims[:5]]
    T = rec[3].shape[0]
    S = soil_params(g, grid, NL)
    nsv = int(dims[13])
    soil = rec[2].reshape(T, N, NT, NB, NL, nsv)
    cvd = rec[3].reshape(T, N, NT)
    wu = rec[5].reshape(T, N, int(dims[14]), 22)
    valid = ~np.isnan(cvd)                                  # tile < nv_active
    cv = np.nan_to_num(cvd)
    moist = np.nan_to_num(soil[..., NL - 1, 0])             # [T,N,NT,NB]
    ice = np.nan_to_num(soil[..., NL - 1, 1:1 + NF])        # [T,N,NT,NB,NF]
    # forcing (MONTH, held), the same streams the C reads
    fq = [x for x in g['PLUGIN_FORCE_TYPE'] if x[0] == 'EFR_DISCHARGE'][0]
    fb = [x for x in g['PLUGIN_FORCE_TYPE'] if x[0] == 'EFR_BASEFLOW'][0]
    ms = F.file_info(fq[3] + '%4d.nc' % START[0], fq[1])['start']
    fq_ = F.PluginForcing(fq[3], fq[1], 'MONTH', START, T, 1, met_start=ms)
    fb_ = F.PluginForcing(fb[3], fb[1], 'MONTH', START, T, 1, met_start=ms)
    q = np.stack([fq_.get(i).numpy() for i in range(T)])    # [T,N]
    b = np.stack([fb_.get(i).numpy() for i in range(T)])
    fq_.close(); fb_.close()
    # EFR at step t from the state at the end of t-1
    ts = np.arange(1, T)
    f64 = lambda a: torch.as_tensor(np.ascontiguousarray(a),
                                    dtype=torch.float64)
    rep = lambda a: f64(np.broadcast_to(a[None], (len(ts),) + a.shape)
                        .reshape((-1,) + a.shape[1:]))
    flat = lambda a: f64(a.reshape((-1,) + a.shape[2:]))
    af = S['af']                                            # [N,B]
    frost = torch.ones(len(ts) * N, NF, dtype=torch.float64) * 0
    frost[:, 0] = 1.0                                       # FROZEN_SOIL FALSE
    r = E.efr_step(
        flat(moist[ts - 1]), flat(ice[ts - 1]), frost, rep(S['resid']),
        rep(S['depth']), rep(S['max_moist']), rep(S['Dsmax']), rep(S['Ds']),
        rep(S['Ws']), rep(S['c']), flat(cv[ts - 1]), rep(af),
        flat(b[ts]), flat(q[ts]), 1, valid=torch.as_tensor(
            valid[ts - 1].reshape(-1, NT)), bug=bug)
    efr = r.efr_moist.numpy().reshape(len(ts), N, NT, NB)
    # land use (G10): only steps where Cv did not change are replayable
    same = np.ones(len(ts), bool)
    if run == 'G10':
        same = (cv[ts] == cv[ts - 1]).all((1, 2))
    # --- 1. the cell outputs (float32 in the file)
    o_m = read_out(rd, 'OUT_EFR_MOIST')
    o_b = read_out(rd, 'OUT_EFR_BASEFLOW')
    o_q = read_out(rd, 'OUT_EFR_DISCHARGE')
    np.testing.assert_allclose(
        r.cell_moist.numpy().reshape(len(ts), N)[same], o_m[ts][same],
        rtol=3e-6, atol=3e-5, err_msg='OUT_EFR_MOIST')
    np.testing.assert_allclose(r.baseflow.numpy().reshape(len(ts), N),
                               o_b[ts], rtol=1e-6)
    np.testing.assert_allclose(r.discharge.numpy().reshape(len(ts), N),
                               o_q[ts], rtol=1e-6)
    # --- 2. pumping of step t from the dumped end-of-step moisture
    resid_mm = S['resid'] * S['depth'] * 1000.
    icet = (ice[ts] * 1.0).sum(-1)                          # frost_fract = 1
    x = (moist[ts] - efr - resid_mm[None, :, None, None] - icet) \
        * af[None, :, None, :] * cv[ts][..., None]
    okt = valid[ts][..., None] & (cv[ts][..., None] > 0) & \
        (af[None, :, None, :] > 0)
    y = np.where(okt, x, 0.0)
    W = np.nansum(wu[ts][..., 11], -1)                      # sum_s withdrawn_gw
    avs = np.nansum(wu[ts][..., 0], -1)                     # sum_s available_gw
    dem = np.nansum(wu[ts][..., 6], -1)                     # sum_s demand_gw
    A = W + np.maximum(y, 0).sum((2, 3))
    pumped = (W > 0) & same[:, None] & (
        np.where(okt, moist[ts] == 0, False).sum((2, 3)) == 0)
    exhausted = pumped & (dem > avs + 1e-9)
    # A_gw is never below what was pumped
    assert (A[pumped] >= W[pumped] - 1e-9).all()
    # exhausted pool: a pumped tile (x > 0) is drained to exactly
    # efr_moist + resid + ice, an unpumped one (x <= 0) stays below it.  So
    # no tile ends above it and at least one ends exactly on it.
    if exhausted.any():
        ye = np.where(okt, y, np.nan)[exhausted].reshape(-1, NT * NB)
        assert (np.nanmax(ye, 1) < 1e-9).all()
        assert (np.nanmin(np.abs(ye), 1) < 1e-9).all()
    st = dict(run=run, steps=int(same.sum()), pumped=int(pumped.sum()),
              exhausted=int(exhausted.sum()),
              frac_full=int((r.frac == 1).sum()),
              frac_interior=int(((r.frac > 0.01) & (r.frac < 1)).sum()),
              frac_zero=int((r.frac.abs() < 1e-12).sum()),
              moist_below_full=int((r.efr_moist.numpy() < moist[ts - 1]
                                    .reshape(-1, NT, NB) - 1e-9).sum()))
    print(st)
    assert st['frac_full'] + st['frac_interior'] + st['frac_zero'] > 0


@pytest.mark.parametrize('run', sorted(RUNS))
def test_l2_replay(run):
    replay(run)

