# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.balance: unit tests and tests against the R1 G-runs.

Reference data:
  * <data root>/vicwur/gruns/R1/<G>/W3/out  (the R1 reference runs: ROUTING_ERROR);
  * <data root>/verification/vicwur_22/gruns/R1/<G>_{d,a5,am}  (the same
    VIC-WUR 3.0 executable and globals plus OUT_WATER_ERROR/GWRECHARGE/
    SNOW_CANOPY/WDEW/SURFSTOR/SOIL_LIQ/SOIL_ICE; daily, AGGFREQ NDAYS 5,
    AGGFREQ NMONTHS 1).

The history files hold float32, so a quantity that is a difference of large
terms can only be reproduced to the float32 rounding of its terms: every
comparison with the R1 runs uses |diff| <= 4 * 2**-24 * (sum of the |terms|)
per cell and day.
Step 0 of the routing check starts from the plugin's initial storage (cold
dams at DAM_ALPHA * capacity, dam_populate.c:47); step 0 of the VIC
OUT_WATER_ERROR needs VIC's initial soil state and is not compared.
"""
import glob
import os
import _paths  # noqa: E402
import re

import numpy as np
import pytest
import torch

from torchvic.vicwur import balance as B

ALL = _paths.DATA_ROOT
GR = ALL + '/vicwur/gruns/R1'
GX = ALL + '/verification/vicwur_22/gruns/R1'
EPS = 2.0 ** -24
D = torch.float64


# ------------------------------------------------------------------ units
def _t(*s, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*s, generator=g, dtype=D)


def test_routing_balance_closes_by_construction():
    area = _t(7) * 1e8 + 1e7
    o = {k: _t(7, seed=i) for i, k in enumerate(
        'STREAM_RUNOFF STREAM_INFLOW DISCHARGE STREAM_MOIST NONREN_DEFICIT '
        'LDAM_INFLOW LDAM_RELEASE GDAM_STORAGE LDAM_STORAGE RECEIVED '
        'RETURNED APPLIED LEFTOVER'.split())}
    for k in ('WI_SURF_SECT', 'WI_DAM_SECT', 'WI_REM_SECT', 'WI_NREN_SECT'):
        o[k] = _t(7, 5, seed=len(k))
    kw = dict(routing=True, dams=True, irrigation=True, wateruse=True,
              nonrenew_with=True)
    prev = _t(7, seed=99)
    err, S = B.routing_balance(o, area, 86400., prev, **kw)
    i, ou, s = B.routing_terms(o, area, 86400., **kw)
    assert torch.equal(S, s)
    assert torch.allclose(err, i - ou - (s - prev), rtol=0, atol=1e-12)
    # GWM drops WI_NREN from the outflow and the deficit from the storage
    _, ou2, s2 = B.routing_terms(o, area, 86400., gwm=True, **{
        k: v for k, v in kw.items()})
    assert torch.allclose(s2 - s, o['NONREN_DEFICIT'], atol=1e-12)
    assert torch.all(ou2 <= ou)
    # options off: nothing is read
    e0, S0 = B.routing_balance({}, area, 1., prev, routing=False)
    assert torch.allclose(e0, prev)
    assert torch.all(S0 == 0)


def test_water_error_correction():
    sm, ss = _t(4), _t(4, seed=1)
    ap, gw = _t(4, seed=2), _t(4, 5, seed=3)
    a, b = B.water_error_correction(sm, ss, ap, gw, irrigation=True,
                                    wateruse=True)
    exp = sm + ap - gw.sum(-1)
    assert torch.allclose(a, exp, atol=1e-12)
    assert torch.allclose(b, ss + ap - gw.sum(-1), atol=1e-12)
    a, b = B.water_error_correction(sm, ss, ap, gw, irrigation=False,
                                    wateruse=True, gwm=True)
    assert torch.equal(a, sm) and torch.equal(b, ss)


def test_aggregation_types():
    for n in ('ROUTING_ERROR', 'APPLIED', 'WI_GW_SECT', 'LDAM_RELEASE',
              'CROP_DVS', 'EFR_BASEFLOW', 'OUT_BASEFLOW', 'PREC', 'RUNOFF', 'EVAP'):
        assert B.agg_type(n) == B.SUM, n
    for n in ('STREAM_MOIST', 'NONREN_DEFICIT', 'LEFTOVER', 'GDAM_STORAGE',
              'EFR_MOIST', 'GDAM_OP_STORAGE'):
        assert B.agg_type(n) == B.END, n
    for n in ('DISCHARGE', 'STREAM_INFLOW', 'REQUIREMENT', 'SHORTAGE', 'CV',
              'EFR_DISCHARGE', 'WATER_ERROR', 'AIR_TEMP', 'CROP_WSTRESS'):
        # CROP_WSTRESS is in the crop SUM list
        want = B.SUM if n == 'CROP_WSTRESS' else B.AVG
        assert B.agg_type(n) == want, n
    assert B.agg_type('OUT_CROP_LAI') == B.MAX


def test_aggregate_kinds_and_incomplete_tail():
    dmy = [(2003, 7, 1 + i) for i in range(31)] + \
          [(2003, 8, 1 + i) for i in range(10)]
    seg, last, count = B.agg_periods(dmy, 'months', 1)
    assert last.tolist() == [30] and count.tolist() == [31]
    assert (seg[31:] == -1).all()
    x = torch.arange(41, dtype=D).reshape(41, 1) - 5.
    assert torch.allclose(B.aggregate(x, B.SUM, seg, last, count),
                          x[:31].sum(0, keepdim=True))
    assert torch.allclose(B.aggregate(x, B.AVG, seg, last, count),
                          x[:31].mean(0, keepdim=True))
    assert B.aggregate(x, B.END, seg, last, count).item() == x[30].item()
    assert B.aggregate(x, B.BEG, seg, last, count).item() == x[0].item()
    # C starts MAX/MIN from 0
    assert B.aggregate(x, B.MAX, seg, last, count).item() == 25.
    assert B.aggregate(-x - 100, B.MAX, seg, last, count).item() == 0.
    assert B.aggregate(x, B.MIN, seg, last, count).item() == -5.
    seg, last, count = B.agg_periods(dmy, 'days', 5)
    assert last.tolist() == list(range(4, 40, 5)) and (count == 5).all()
    assert seg[40].item() == -1


# -------------------------------------------------- helpers for the R1 runs
def _have(p):
    return os.path.isdir(p)


def _read(rundir, names):
    """Concatenate the yearly history files (each opened once, closed)."""
    import netCDF4
    fs = sorted(glob.glob(rundir + '/out/fluxes.*.nc'))
    acc = {nm: [] for nm in names}
    for f in fs:
        d = netCDF4.Dataset(f)
        try:
            for nm in names:
                v = d['OUT_' + nm]
                v.set_auto_mask(False)
                acc[nm].append(np.asarray(v[:], np.float64))
        finally:
            d.close()
    out = {}
    for nm in names:
        a = np.concatenate(acc[nm])
        # [T, (class), lat, lon] -> [T, 1, cells] or [T, class, cells]
        a = a.reshape(a.shape[0], -1, a.shape[-2] * a.shape[-1]) \
            if a.ndim == 4 else a.reshape(a.shape[0], 1, -1)
        out[nm] = torch.tensor(a)
    return out


def _flags(run):
    t = open(f'{GR}/{run}/global_W3.txt').read()

    def on(k):
        return re.search(rf'^{k}\s+TRUE', t, re.M) is not None
    outv = set(re.findall(r'^OUTVAR OUT_(\w+)', t, re.M))
    return dict(routing=on('ROUTING'), dams=on('DAMS'),
                irrigation=on('IRRIGATION'), wateruse=on('WATERUSE'),
                nonrenew_with=on('NONRENEWABLE_WITHDRAWAL')), outv


def _area():
    import netCDF4
    d = netCDF4.Dataset(ALL + '/vicwur/R1/domain.nc')
    try:
        return torch.tensor(np.asarray(d['area'][:], np.float64).ravel())
    finally:
        d.close()


def _dam_init():
    """cold-start dam storage [mm] per cell (dam_populate.c:47)."""
    import netCDF4
    d = netCDF4.Dataset(ALL + '/vicwur/R1/dams.nc')
    try:
        idm = np.ma.filled(d['id_map'][:], -1).ravel()
        ids = np.asarray(d['id'][:])
        cap = np.asarray(d['capacity'][:], np.float64)
    finally:
        d.close()
    hm3 = np.zeros(idm.size)
    for k, i in enumerate(ids):
        hm3[idm == i] = cap[k] * 0.85
    return torch.tensor(hm3)


def _ldam_frac():
    """inflow fraction of the local dam (type 0) per cell, else 0."""
    import netCDF4
    d = netCDF4.Dataset(ALL + '/vicwur/R1/dams.nc')
    try:
        idm = np.ma.filled(d['id_map'][:], -1).ravel()
        ids = np.asarray(d['id'][:])
        typ = np.asarray(d['type'][:])
        fr = np.asarray(d['inflow_fraction'][:], np.float64)
    finally:
        d.close()
    f = np.zeros(idm.size)
    for k, i in enumerate(ids):
        if typ[k] == 0:
            f[idm == i] = fr[k]
    return torch.tensor(f)


ROUT_NAMES = ('STREAM_RUNOFF STREAM_INFLOW DISCHARGE STREAM_MOIST '
              'NONREN_DEFICIT LDAM_INFLOW LDAM_RELEASE GDAM_STORAGE '
              'LDAM_STORAGE RECEIVED RETURNED APPLIED LEFTOVER').split()
SECT = 'WI_SURF_SECT WI_DAM_SECT WI_REM_SECT WI_NREN_SECT'.split()


def _routing_runs():
    r = []
    for g in sorted(glob.glob(GR + '/*/W3/out')):
        run = g.split('/')[-3]
        if os.path.exists(f'{GR}/{run}/global_W3.txt'):
            fl, outv = _flags(run)
            if 'ROUTING_ERROR' in outv:
                r.append(run)
    return r


def _routing_inputs(fl, outv, rd, first):
    names = [n for n in ROUT_NAMES + SECT if n in outv]
    o = _read(rd, names + ['ROUTING_ERROR'])
    ref = o.pop('ROUTING_ERROR')[:, 0, :]
    if 'NONREN_DEFICIT' not in o and not fl['nonrenew_with']:
        # not written, and 0 in C without non-renewable withdrawal
        o['NONREN_DEFICIT'] = torch.zeros_like(ref)[:, None, :]
    t = {k: (v[:, 0, :] if k not in SECT else v.permute(0, 2, 1))
         for k, v in o.items()}
    return t, ref


@pytest.mark.skipif(not _have(GR), reason='R1 G-runs not available')
@pytest.mark.parametrize('run', _routing_runs())
def test_routing_error_l2(run):
    fl, outv = _flags(run)
    t, ref = _routing_inputs(fl, outv, f'{GR}/{run}/W3', None)
    area = _area()
    # the quantities the options read must be in the history
    inflow, outflow, S = B.routing_terms(t, area, 86400., **fl)
    prev = torch.cat([torch.zeros(1, S.shape[1], dtype=D), S[:-1]])
    if fl['dams']:
        prev[0] = _dam_init() * 1e6 / area * 1000.
    err, S2 = B.routing_balance(t, area, 86400., prev, **fl)
    # every output term is float32: sum the |terms| (deficit enters S
    # negatively, so flip it in the abs copy), each good to EPS/2
    ta = {k: v.abs() for k, v in t.items()}
    if 'NONREN_DEFICIT' in ta:
        ta['NONREN_DEFICIT'] = -ta['NONREN_DEFICIT']
    ia, oa, Sa = B.routing_terms(ta, area, 86400., **fl)
    tol = 2 * EPS * (ia + oa + Sa + prev.abs()) + 1e-12
    bad = (err - ref).abs() > tol
    ratio = ((err - ref).abs() / tol).max().item()
    print(f'{run}: max |diff| {(err - ref).abs().max():.3e}, '
          f'max |C| {ref.abs().max():.3e}, worst diff/tol {ratio:.3f}')
    assert not bad.any()


XRUNS = sorted(os.path.basename(p)[:-2] for p in glob.glob(GX + '/*_d')
               if os.path.exists(p + '/rc') and open(p + '/rc').read().strip()
               == '0')


def _xglobal(run, v):
    t = open(f'{GX}/{run}_{v}/global.txt').read()

    def on(k):
        return re.search(rf'^{k}\s+TRUE', t, re.M) is not None
    return dict(irrigation=on('IRRIGATION'), wateruse=on('WATERUSE'),
                gwm=on('GWM'))


@pytest.mark.skipif(not XRUNS, reason='vicwur_22 reruns not available')
@pytest.mark.parametrize('run', XRUNS)
def test_vic_water_error_l2(run):
    rd = f'{GX}/{run}_d'
    fl = _xglobal(run, 'd')
    names = ('WATER_ERROR PREC EVAP RUNOFF BASEFLOW GWRECHARGE SWE '
             'SNOW_CANOPY WDEW SURFSTOR APPLIED').split()
    names = [n for n in names if n != 'APPLIED' or fl['irrigation']]
    dams = re.search(r'^DAMS\s+TRUE', open(f'{GX}/{run}_d/global.txt').read(),
                     re.M) is not None
    if dams:
        names += ['LDAM_RELEASE', 'LDAM_INFLOW']
    o = _read(rd, names + ['SOIL_LIQ', 'SOIL_ICE'])
    c = {k: v[:, 0, :] for k, v in o.items() if not k.startswith('SOIL_')}
    liq = o['SOIL_LIQ'].permute(0, 2, 1)
    ice = o['SOIL_ICE'].permute(0, 2, 1)
    z = torch.zeros_like(c['PREC'])
    if dams:
        # OUT_RUNOFF of a local-dam cell is rewritten after put_data
        c['RUNOFF'] = B.local_dam_runoff_original(
            c['RUNOFF'], c['LDAM_INFLOW'], c['LDAM_RELEASE'], _ldam_frac(),
            _area())
    kw = dict(prec=c['PREC'], lake_chan_in=z, evap=c['EVAP'],
              runoff=c['RUNOFF'], gwrecharge=c['GWRECHARGE'],
              baseflow=c['BASEFLOW'], soil_liq=liq, soil_ice=ice,
              swe=c['SWE'], snow_canopy=c['SNOW_CANOPY'], wdew=c['WDEW'],
              surfstor=c['SURFSTOR'])
    _, S = B.vic_water_error(prev_storage=z, **kw)
    # VIC's stored storage as the plugins adapt it (plugin_store_error.c:132)
    ap = o['APPLIED'][:, 0, :] if fl['irrigation'] else z
    if fl['wateruse']:
        gw = _read(rd, ['WI_GW_SECT'])['WI_GW_SECT'].permute(0, 2, 1)
    else:
        gw = torch.zeros(*z.shape, 5, dtype=D)
    adj, _ = B.water_error_correction(S, z, ap, gw, **fl)
    prev = torch.cat([z[:1], adj[:-1]])
    err, _ = B.vic_water_error(prev_storage=prev, **kw)
    raw = torch.cat([z[:1], S[:-1]])
    unc, _ = B.vic_water_error(prev_storage=raw, **kw)
    sl = slice(1, None)
    scale = (c['PREC'].abs() + c['EVAP'].abs() + c['RUNOFF'].abs()
             + c['BASEFLOW'].abs() + S.abs() + prev.abs() + ap.abs()
             + gw.abs().sum(-1))
    tol = 4 * EPS * scale + 1e-12
    d = (err - c['WATER_ERROR']).abs()
    print(f'{run}: max |diff| {d[sl].max():.3e} max |C| '
          f'{c["WATER_ERROR"][sl].abs().max():.3e} worst diff/tol '
          f'{(d / tol)[sl].max():.3f}; uncorrected max |diff| '
          f'{(unc - c["WATER_ERROR"]).abs()[sl].max():.3e}')
    assert (d[sl] <= tol[sl]).all()
    if ap.abs().max() > 1.0:
        # the correction is what makes it match: without it the error is
        # off by the applied water of the previous step
        assert (unc - c['WATER_ERROR']).abs()[sl].max() > 0.5


AGG_NAMES = ('ROUTING_ERROR DISCHARGE STREAM_INFLOW STREAM_RUNOFF '
             'STREAM_MOIST NONREN_DEFICIT GDAM_INFLOW GDAM_STORAGE '
             'GDAM_RELEASE GDAM_DEMAND GDAM_EFR GDAM_OP_STORAGE '
             'LDAM_INFLOW LDAM_STORAGE LDAM_RELEASE WITHDRAWN DEMAND '
             'RETURNED CONSUMED SHORTAGE REQUIREMENT RECEIVED LEFTOVER '
             'APPLIED EFR_DISCHARGE EFR_BASEFLOW EFR_MOIST WI_GW_SECT '
             'WI_SURF_SECT DE_REM_SECT CV BASEFLOW PREC SWE GWRECHARGE AIR_TEMP EVAP RUNOFF '
             'WATER_ERROR CROP_LAI CROP_DVS CROP_WSTRESS CROP_GROW').split()


def _agg_cases():
    c = []
    for r in XRUNS:
        for v in ('a5', 'am'):
            if os.path.exists(f'{GX}/{r}_{v}/rc') and \
                    open(f'{GX}/{r}_{v}/rc').read().strip() == '0':
                c.append((r, v))
    return c


def _dmy():
    import datetime
    d0 = datetime.date(2003, 7, 1)
    return [((d0 + datetime.timedelta(i)).year, (d0 + datetime.timedelta(i)).month,
             (d0 + datetime.timedelta(i)).day) for i in range(2557)]


def _agg_check(run, v, device):
    import netCDF4
    fn = glob.glob(f'{GX}/{run}_d/out/fluxes.*.nc')[0]
    d = netCDF4.Dataset(fn)
    have = {k[4:] for k in d.variables if k.startswith('OUT_')}
    d.close()
    names = [n for n in AGG_NAMES if n in have]
    daily = _read(f'{GX}/{run}_d', names)
    agg = _read(f'{GX}/{run}_{v}', names)
    freq, n = ('days', 5) if v == 'a5' else ('months', 1)
    seg, last, count = B.agg_periods(_dmy(), freq, n, device)
    worst = 0.
    for nm in names:
        x = daily[nm].to(device)
        got = B.aggregate_named(nm, x, seg, last, count).cpu()
        kind = B.agg_type(nm)
        ref = agg[nm]
        assert got.shape == ref.shape, (nm, got.shape, ref.shape)
        if kind in (B.SUM, B.AVG):
            sc = B.aggregate(x.abs(), kind, seg, last, count).cpu()
        else:
            sc = got.abs()
        tol = 2 * EPS * sc + 1e-30
        r = ((got - ref).abs() / tol).max().item()
        worst = max(worst, r)
        assert r <= 1.0, (run, v, nm, kind, r)
    print(f'{run} {v}: {len(names)} variables, worst diff/tol {worst:.3f}')
    return names


@pytest.mark.skipif(not _agg_cases(), reason='vicwur_22 reruns not available')
@pytest.mark.parametrize('run,v', _agg_cases())
def test_aggregation_l2(run, v):
    _agg_check(run, v, 'cpu')


# --------------------------------------------------------------------- GPU
@pytest.mark.gpu
@pytest.mark.skipif(not _have(GR), reason='R1 G-runs not available')
def test_gpu_no_sync_and_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    run = 'G10'
    fl, outv = _flags(run)
    t, ref = _routing_inputs(fl, outv, f'{GR}/{run}/W3', None)
    area = _area()
    prev = torch.zeros_like(ref)
    prev[0] = _dam_init() * 1e6 / area * 1000.
    err_c, S_c = B.routing_balance(t, area, 86400., prev, **fl)
    g = {k: v.cuda() for k, v in t.items()}
    ga, gp = area.cuda(), prev.cuda()
    x = (ref.abs() + 1).cuda()
    seg, last, count = B.agg_periods(_dmy(), 'days', 5, 'cuda')
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        err_g, S_g = B.routing_balance(g, ga, 86400., gp, **fl)
        a1 = B.aggregate(x, B.SUM, seg, last, count)
        a2 = B.aggregate(x, B.AVG, seg, last, count)
        a3 = B.aggregate(x, B.END, seg, last, count)
        a4 = B.aggregate(x, B.MAX, seg, last, count)
        z = B.vic_water_error(x, x, x, x, x, x, x[:, :, None].expand(-1, -1, 3),
                              x[:, :, None].expand(-1, -1, 3), x, x, x, x, x)
    finally:
        torch.cuda.set_sync_debug_mode('default')
    torch.cuda.synchronize()
    assert torch.equal(err_g.cpu(), err_c)
    assert torch.equal(S_g.cpu(), S_c)
    xc = x.cpu()
    for a, k in ((a1, B.SUM), (a2, B.AVG), (a3, B.END), (a4, B.MAX)):
        c = B.aggregate(xc, k, *B.agg_periods(_dmy(), 'days', 5))
        assert torch.allclose(a.cpu(), c, rtol=1e-13, atol=0)
    assert z[0].is_cuda
    _agg_check_gpu()


def _agg_check_gpu():
    cases = _agg_cases()
    if cases:
        _agg_check(cases[0][0], 'am', 'cuda')
