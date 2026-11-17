# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.landuse replayed on the R1 G-runs with monthly Cv.

The C runs (G8 = G7 + LANDUSE_FORCE, with irr.nc; G8d the same with the
paddy irrigation file irr_paddy.nc) use the dump build <data root>/build/
vicwur3_ludump: vicwur3_dump plus tools/vicwur/vicwur_ludump.patch, which
writes, at the start and end of every lu_apply call (84 monthly records over
2003-07 .. 2010-06), every state lu_apply reads or writes and the statics
it uses (no effect on the physics).

The replay feeds the kernels only what the C had as input:
  * the Cv record from the R1 forcing files cv.<yyyy>.nc through
    torchvic.vicwur.forcing.PluginForcing (MONTH cadence; it must read a
    record exactly at the steps where the C called lu_apply, and give the
    lu_force.Cv the C holds), map_cv_forcing and cv_update;
  * the tile, band, carbon, energy, leftover and ring states before the call;
  * the soil statics, LAI, the day of year and the air temperature;
and compares every state after the call, every call, to 1e-10 (relative,
1e-9 absolute) and checks that the water and energy ledger of every call
closes to round-off.
"""
import datetime
import os
import _paths  # noqa: E402
import struct

import numpy as np
import pytest
import torch

from torchvic.vicwur import forcing as F
from torchvic.vicwur import landuse as LU
from torchvic.vicwur import params as PP

V = _paths.env_path(('TORCHVIC_VICWUR_110_GRUNS', 'VICWUR_110_GRUNS'), 'verification/vicwur_110/gruns')
RUNS = ('G8', 'G8d')
START = (2003, 7, 1, 0)
END = datetime.date(2010, 6, 30)
f64 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)
TB = ('wdew', 'swq', 'pack_water', 'surf_water', 'snow_canopy',
      'coldcontent', 'density', 'depth', 'last_snow', 'melting', 'surf_temp',
      'pack_temp', 'annual_npp', 'annual_npp_prev', 'clitter', 'cinter',
      'cslow', 'lai')


def read_ludump(path):
    recs = {}
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            tag, n = struct.unpack('<iq', h)
            recs.setdefault(tag, []).append(
                np.frombuffer(f.read(8 * n), '<f8').copy())
    return recs


def statics(dims, a):
    nc, nt, nb, nl, nn, nf, ni = (int(x) for x in dims[:7])
    per = 4 + 2 * nt + 2 * nb + 8 * nl + 5 * nn + nf + 1 + 2 * ni
    a = a.reshape(nc, per)
    S, p = {}, 0

    def take(k, w):
        nonlocal p
        v = a[:, p:p + k * w].reshape(nc, k, w) if w > 1 else a[:, p:p + k]
        p += k * w
        return v
    hd = take(4, 1)
    S['nva'], S['area'], S['lat'], S['fs'] = hd.T
    tl = take(nt, 2)
    S['vc'], S['over'] = tl[..., 0], tl[..., 1]
    bd = take(nb, 2)
    S['af'], S['tfac'] = bd[..., 0], bd[..., 1]
    ly = take(nl, 8)
    for j, k in enumerate(('max_moist', 'depth', 'bulk_density',
                           'soil_density', 'organic', 'quartz',
                           'bulk_dens_min', 'soil_dens_min')):
        S[k] = ly[..., j]
    nd = take(nn, 5)
    for j, k in enumerate(('dz_node', 'zsum_node', 'max_moist_node',
                           'expt_node', 'bubble_node')):
        S[k] = nd[..., j]
    S['frost_fract'] = take(nf, 1)
    S['ni_active'] = take(1, 1)[:, 0]
    ir = take(ni, 2)
    S['irr_vi'], S['irr_paddy'] = ir[..., 0], ir[..., 1]
    assert p == per
    return S


def state(dims, a):
    nc, nt, nb, nl, nn, nf, ni, R = (int(x) for x in dims[:8])
    k = 18 + nl + nl * nf + nn
    per = 2 * nt + nt * nb * k + ni * nb + R + 2
    a = a.reshape(nc, per)
    st = dict(cv=a[:, :nt], cvf=a[:, nt:2 * nt])
    tb = a[:, 2 * nt:2 * nt + nt * nb * k].reshape(nc, nt, nb, k)
    for j, n in enumerate(TB):
        st[n] = tb[..., j]
    st['moist'] = tb[..., 18:18 + nl]
    st['ice'] = tb[..., 18 + nl:18 + nl + nl * nf].reshape(nc, nt, nb, nl, nf)
    st['t_node'] = tb[..., 18 + nl + nl * nf:]
    p = 2 * nt + nt * nb * k
    st['leftover'] = a[:, p:p + ni * nb].reshape(nc, ni, nb)
    p += ni * nb
    st['ring'] = a[:, p:p + R]
    st['discharge'], st['stream'] = a[:, p + R], a[:, p + R + 1]
    return st


def gate(a, b, what):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.abs(a - b) <= 1e-9 + 1e-10 * np.abs(b)
    assert ok.all(), '%s: max |d| %.3e' % (what, np.abs(a - b).max())
    return float((np.abs(a - b) / np.maximum(1.0, np.abs(b))).max())


def replay(run):
    rd = os.path.join(V, run)
    path = os.path.join(rd, 'ludump.bin')
    if not os.path.exists(path):
        pytest.skip('dump absent: ' + path)
    R = read_ludump(path)
    dims = R[30][0]
    nc, nt, nb, nl, nn, nf, ni, Rl, rps, uh, K, pbare = \
        (int(x) for x in dims[:12])
    irrigation, routing, full = (bool(x) for x in dims[12:15])
    msw, dens_max, dt = dims[15], dims[18], dims[17]
    assert int(dims[16]) == 0                       # DENS_BRAS
    S = statics(dims, R[31][0])
    act = np.arange(nt)[None, :] < S['nva'][:, None]
    vc = np.where(act, np.nan_to_num(S['vc'], nan=-1), -1).astype(np.int64)
    tile_class = torch.as_tensor(np.where(vc < K, vc, -1))
    g = PP.parse_global(os.path.join(rd, 'global_W3.txt'))
    fcv = [x for x in g['PLUGIN_FORCE_TYPE'] if x[0] == 'CV'][0]
    nsteps = (END - datetime.date(*START[:3])).days + 1
    ms = F.file_info(fcv[3] + '%4d.nc' % START[0], fcv[1])['start']
    fr = F.PluginForcing(fcv[3], fcv[1], 'MONTH', START, nsteps, 1,
                         met_start=ms)
    try:
        calls = [int(r[0]) for r in R[32]]
        assert calls == [s for s in range(nsteps) if fr.reads(s)]
        forced = [fr.get(s).T.contiguous() for s in calls]   # [C, K]
    finally:
        fr.close()

    vi = np.nan_to_num(S['irr_vi'], nan=-1).astype(np.int64)
    vi = np.where(np.arange(ni)[None, :] < S['ni_active'][:, None], vi, -1)
    pad = (np.nan_to_num(S['irr_paddy']) != 0) & (vi >= 0)
    vit = torch.as_tensor(vi)
    paddy = torch.zeros(nc, nt, dtype=torch.bool)
    for i in range(ni):
        paddy |= torch.as_tensor(pad[:, i:i + 1]) & (
            torch.arange(nt) == vit[:, i:i + 1])
    pbt = torch.as_tensor(act & (vc == pbare - 1))
    pb_idx = torch.where(pbt.any(1), pbt.long().argmax(1),
                         torch.full((nc,), -1))
    order = torch.cat([torch.where(torch.as_tensor(pad), vit,
                                   torch.full_like(vit, -1)),
                       pb_idx[:, None]], 1)
    soil = {k: f64(S[k]) for k in ('zsum_node', 'max_moist_node',
                                  'expt_node', 'bubble_node', 'depth',
                                  'soil_dens_min', 'bulk_dens_min', 'quartz',
                                  'soil_density', 'bulk_density', 'organic',
                                  'frost_fract')}
    soil['fs_active'] = torch.as_tensor(S['fs'] != 0)
    cs_fn = LU.lu_heat_capacity_fn(soil, 'exact')
    actt = torch.as_tensor(act)
    worst, ledger_worst, eledger_worst, nchanged = 0.0, 0.0, 0.0, 0
    for k, s in enumerate(calls):
        pre, post = state(dims, R[33][k]), state(dims, R[34][k])
        cvf, fatal_map = LU.map_cv_forcing(f64(forced[k]), tile_class)
        assert not bool(fatal_map.any())
        gate(np.where(act, cvf.numpy(), 0), np.nan_to_num(pre['cvf']),
             'lu_force.Cv')
        cvo = f64(np.nan_to_num(pre['cv']))
        cvn, dcv, csum, fatal, warn = LU.cv_update(cvf, cvo, actt)
        nchanged += int(((dcv.abs() > LU.MINCOVERAGECHANGE) & actt).any(1)
                        .sum())
        st = {n: f64(np.nan_to_num(pre[n])) for n in TB + (
            'moist', 'ice', 't_node', 'leftover', 'ring')}
        st['melting'] = st['melting'] != 0
        air = f64(R[32][k][5:5 + nc])
        P = dict(area_fract=f64(S['af']), max_moist=f64(S['max_moist']),
                 overstory=torch.as_tensor(np.nan_to_num(S['over']) != 0),
                 active=actt, bare=torch.as_tensor(act & (vc == K)),
                 paddy=paddy & actt, pbare=pbt, lai=st.pop('lai'),
                 dz_node=f64(S['dz_node']), max_surface_swe=msw,
                 doy=int(R[32][k][4]), lat=f64(S['lat']),
                 new_density=LU.band_new_density(air, f64(S['tfac']),
                                                 dens_max=dens_max),
                 paddy_order=order, irr_tile=vit, area=f64(S['area']),
                 dt=dt, rps=rps, uh_length=uh)
        st1, terms = LU.apply(st, cvo, cvn, P, mode='exact',
                              paddy_exchange=irrigation and pbare != K,
                              route_leftover=irrigation and routing,
                              cs_fn=cs_fn, full_energy=full)
        assert not bool((fatal | terms['fatal']).any())
        worst = max(worst, gate(cvn.numpy()[act], post['cv'][act], 'cv'))
        m = act[:, :, None]
        for n in TB[:-1] + ('moist', 'ice', 't_node'):
            a = st1[n].double().numpy()
            mm = np.broadcast_to(m.reshape(m.shape + (1,) * (a.ndim - 3)),
                                 a.shape)
            worst = max(worst, gate(a[mm], post[n][mm], '%s call %d'
                                    % (n, k)))
        if irrigation and routing:
            ok = vi >= 0
            worst = max(worst, gate(st1['leftover'].numpy()[ok],
                                    post['leftover'][ok], 'leftover'))
            worst = max(worst, gate(st1['ring'].numpy(), post['ring'], 'ring'))
            worst = max(worst, gate(terms['discharge'].numpy(),
                                    post['discharge'], 'discharge'))
            worst = max(worst, gate(terms['stream'].numpy(), post['stream'],
                                    'stream'))
        led = LU.ledger(st, st1, cvo, cvn, terms, P['area_fract'],
                        P if irrigation and routing else None)
        ledger_worst = max(ledger_worst, float(
            (led['residual'].abs() / led['storage_before'].abs()
             .clamp(min=1.0)).max()))
        eledger_worst = max(eledger_worst, float(
            (led['energy_residual'].abs() / led['energy_before'].abs()
             .clamp(min=1.0)).max()))
    return dict(calls=len(calls), changed_cells=nchanged, worst=worst,
                ledger=ledger_worst, energy_ledger=eledger_worst)


@pytest.mark.parametrize('run', RUNS)
def test_l2_replay(run):
    r = replay(run)
    print(run, r)
    assert r['calls'] == 84 and r['changed_cells'] > 0
    assert r['ledger'] < 1e-12 and r['energy_ledger'] < 1e-12
