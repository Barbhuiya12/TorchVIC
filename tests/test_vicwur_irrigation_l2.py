# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.irrigation replayed on the R1 G4 runs (VICWUR_DUMP).

The dump (plugin_dump.c) is written at the end of step t, after
irr_get_withdrawn and irr_run_shortage.  So the pre-application layer-0
moisture is dump moisture - applied, the previous leftover is the leftover
of step t-1, and the kernels must reproduce requirement, flag, offset,
demand (wu_var demand_gw + demand_surf), received, applied, leftover,
layer-0 moisture, shortage, layer water stress and layer-0 Ksat of every
irrigated tile and band at every step, from the soil and irrigation files
alone.  Runs: G4 (base), G4d (paddy), G4e (EFFICIENT), G4o (OFFSET),
G4p (POTENTIAL); dumps from <data root>/verification/vicwur_12/gruns.
"""
import os
import _paths  # noqa: E402

import netCDF4
import numpy as np
import pytest
import torch

from torchvic.vicwur import irrigation as IR
from torchvic.vicwur import params as P

ROOT = _paths.env_path(('TORCHVIC_VICWUR_12_GRUNS', 'VICWUR_12_GRUNS'), 'verification/vicwur_12/gruns')
RUNS = {'G4_dump': {}, 'G4d': {}, 'G4e': {'efficient': True},
        'G4o': {}, 'G4p': {'potential': True}}
IRR_SECT = 4


def read_dump(path):
    import struct
    out = {2: [], 3: [], 5: [], 6: []}
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


def soil_arrays(g, grid):
    """max_moist, Wcr, Wfc, Wpwp [N,L]; root [nveg,N,L]; AreaFract [B,N]."""
    with netCDF4.Dataset(g['PARAMETERS'][0][0]) as ds:   # closed at once
        raw = {k: np.array(ds[k][:], dtype=np.float64) for k in (
            'depth', 'bulk_density', 'soil_density', 'Wcr_FRACT',
            'Wpwp_FRACT', 'Ksat', 'root_depth', 'root_fract',
            'AreaFract')}
    f = lambda k: grid.take(raw[k])
    depth, bulk, dens = f('depth').T, f('bulk_density').T, f('soil_density').T
    mm = depth * (1 - bulk / dens) * 1000.
    wcr_f, wpwp_f = f('Wcr_FRACT').T, f('Wpwp_FRACT').T
    wfc_f = wpwp_f + (wcr_f - wpwp_f) / 0.7
    ksat = f('Ksat').T
    rd, rf = f('root_depth'), f('root_fract')            # [nveg,Z,N]
    nveg, nz, n = rd.shape
    nl = depth.shape[1]
    top = np.cumsum(depth, 1)
    bot = top - depth
    root = np.zeros((nveg, n, nl))
    for v in range(nveg):
        for c in range(n):
            z0 = 0.0
            for z in range(nz):
                z1 = z0 + rd[v, z, c]
                if rd[v, z, c] > 0:
                    for l in range(nl):
                        hi = top[c, l] if l < nl - 1 else max(z1, top[c, l])
                        ov = max(0.0, min(z1, hi) - max(z0, bot[c, l]))
                        root[v, c, l] += rf[v, z, c] * ov / rd[v, z, c]
                z0 = z1
    s = root.sum(-1, keepdims=True)
    root = np.where(s > 0, root / np.where(s > 0, s, 1), 0)
    return dict(mm=mm, wcr=wcr_f * mm, wfc=wfc_f * mm, wpwp=wpwp_f * mm,
                root=root, af=f('AreaFract'), ksat=ksat[:, 0])


@pytest.mark.parametrize('run', sorted(RUNS))
def test_l2_replay(run):
    dump = os.path.join(ROOT, run, 'dump.bin')
    if not os.path.exists(dump):
        pytest.skip('dump absent: ' + dump)
    gpath = os.path.join(ROOT, run, 'global_W3.txt')
    g = P.parse_global(gpath)
    inp = P.load_vicwur_inputs(gpath)
    ir, grid = inp.irrigation, inp.grid
    dims, rec = read_dump(dump)
    N, NT, NB, NL, NF, _, NJ = dims[:7]
    T = rec[3].shape[0]
    S = soil_arrays(g, grid)
    efficient = 'EFFICIENT_IRRIGATION' in g and \
        g['EFFICIENT_IRRIGATION'][0][0] == 'TRUE'
    potential = g.get('POTENTIAL_IRRIGATION', [['FALSE']])[0][0] == 'TRUE'
    nsv = 7 + NF
    soil = rec[2].reshape(T, N, NT, NB, NL, nsv)
    cvd = rec[3].reshape(T, N, NT)
    wu = rec[5].reshape(T, N, 5, 22)
    iv = rec[6].reshape(T, N, NJ, NB, 8)
    # compact irrigated tile index per cell (class order, active only)
    vix = ir.veg_index.numpy()
    act = ir.active.numpy()
    jmap = -np.ones((N, NJ), int)
    jcls = np.zeros((N, NJ), int)
    ipad = np.zeros((N, NJ), bool)
    gw = np.zeros((N, NJ)); eff = np.zeros((N, NJ)); om = np.zeros((N, NJ))
    for c in range(N):
        j = 0
        for k in range(act.shape[1]):
            if act[c, k]:
                jmap[c, j] = vix[c, k]
                jcls[c, j] = int(ir.veg_class[k]) - 1
                ipad[c, j] = bool(ir.paddy[k])
                gw[c, j] = ir.gw_fraction[c, k]
                eff[c, j] = ir.efficiency[c, k]
                om[c, j] = ir.offset[c, k]
                j += 1
    valid = jmap >= 0                                       # [N,J]
    ci = np.arange(N)[:, None]
    tile = np.where(valid, jmap, 0)
    sl = soil[:, ci, tile]                                  # [T,N,J,B,L,nsv]
    moist_post = sl[..., 0]
    ksat_d = sl[:, :, :, :, 0, 1 + NF]
    ws_d = sl[..., 2 + NF]
    cv = np.where(valid[None], cvd[:, ci, tile], 0.0)       # [T,N,J]
    af = S['af'].T[None, :, :]                              # [1,N,B]
    root = np.stack([S['root'][jcls[c], c] for c in range(N)])   # [N,J,L]
    wcr = np.broadcast_to(S['wcr'][:, None, None, :], (N, NJ, NB, NL))
    f64 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)
    rep = lambda a: f64(np.broadcast_to(a[None], (T,) + a.shape).reshape(
        (T * N,) + a.shape[1:]))
    flat = lambda a: f64(np.nan_to_num(a).reshape((T * N,) + a.shape[2:]))
    amt = np.nan_to_num(iv[..., 2])
    left_d = np.nan_to_num(iv[..., 1])
    left_prev = np.concatenate([np.zeros_like(left_d[:1]), left_d[:-1]])
    off_d = np.nan_to_num(iv[..., 6])
    off_prev = np.concatenate([np.zeros_like(off_d[:1]), off_d[:-1]])
    pcv_d = np.nan_to_num(iv[..., 7])
    pcv_prev = np.concatenate([np.zeros_like(pcv_d[:1]), pcv_d[:-1]])
    m0_pre = np.nan_to_num(moist_post[..., 0]) - amt
    moist = np.nan_to_num(moist_post).copy()
    moist[..., 0] = m0_pre
    moist = flat(moist)[..., None] if False else flat(moist)
    ice = torch.zeros(T * N, NJ, NB, NL, 1, dtype=torch.float64)
    frost = torch.ones(T * N, 1, dtype=torch.float64)
    cvt = flat(cv); aft = f64(np.broadcast_to(af, (T, N, NB)).reshape(-1, NB))
    omax = rep(om)
    paddy = torch.as_tensor(np.broadcast_to(ipad[None], (T, N, NJ)).reshape(
        T * N, NJ))
    wfc = rep(S['wfc']); maxm = rep(S['mm'][:, 0])
    rootb = rep(np.repeat(root[:, :, None, :], NB, 2))
    wcrt = rep(np.ascontiguousarray(wcr))
    new_day = torch.tensor(True)
    # offset counter: from the dumped previous values
    off, pcv = IR.offset_step(flat(off_prev), flat(pcv_prev), cvt, omax,
                              new_day)
    m = torch.as_tensor(np.broadcast_to(valid[None], (T, N, NJ)).reshape(
        T * N, NJ))[..., None] & (aft[:, None, :] >= 0)
    chk = lambda a, b, what, mask=m, tol=1e-9: np.testing.assert_allclose(
        a.numpy()[mask.numpy()], b.numpy()[mask.numpy()], rtol=tol, atol=tol,
        err_msg=what)
    chk(off, flat(off_d), 'offset')
    chk(pcv, flat(pcv_d), 'prev_Cv')
    req, flag = IR.requirement(moist, ice, frost, rootb, wcrt, wfc, maxm,
                               flat(left_prev), flat(off_d), omax, paddy,
                               cvt, aft)
    actm = IR.active_mask(cvt, aft) & m
    chk(req, flat(iv[..., 4]), 'requirement', actm)
    assert torch.equal(flag[actm], (flat(iv[..., 5]) > 0)[actm])
    D, G, E, D0 = IR.demand(req, flag, cvt, aft, rep(gw), rep(eff))
    wuf = wu[:, :, IRR_SECT, :]
    has = torch.as_tensor(valid.any(1)[None].repeat(T, 0).reshape(-1))
    dd = f64(np.nan_to_num(wuf[..., 6] + wuf[..., 7]).reshape(-1))
    chk(D[has], dd[has], 'wu demand', torch.ones(int(has.sum()), dtype=bool))
    avail = f64(np.nan_to_num(wuf[..., 18]).reshape(-1))
    m0, left, rec_, app = IR.get_withdrawn(
        moist[..., 0], flat(left_prev), req, flag, actm, wfc[:, 0], maxm,
        paddy, efficient, cvt, aft, potential=potential,
        available=None if potential else avail)
    chk(rec_, flat(iv[..., 0]), 'received', actm)
    chk(app, flat(amt), 'applied', actm)
    chk(left, flat(left_d), 'leftover', actm)
    chk(m0, flat(np.nan_to_num(moist_post[..., 0])), 'layer-0 moisture', actm)
    bal = IR.balance_residual(rec_, flat(left_prev), app, left)
    assert bal[actm].abs().max() < 1e-10
    moist2 = torch.cat([m0.unsqueeze(-1), moist[..., 1:]], -1)
    sh, ws, cs = IR.shortage_stress(
        moist2, ice, frost, rootb, wcrt, rep(S['wpwp']), cvt, aft, flag,
        True)
    chk(sh, flat(iv[..., 3]), 'shortage', actm)
    wsm = actm.unsqueeze(-1).expand_as(ws)
    chk(ws, flat(ws_d), 'water stress', wsm)
    kreq = IR.paddy_ksat(rep(S['ksat']), paddy, NB, active=actm)
    chk(kreq, flat(ksat_d), 'Ksat0', m, tol=1e-12)
    # coverage
    st = dict(flag=int(flag[actm].sum()), applied=int((app[actm] > 0).sum()),
              leftover=int((left[actm] > 0).sum()),
              shortage=int((sh[actm] > 0).sum()),
              paddy=int(paddy.any()))
    print(run, st)
    assert st['flag'] > 0 and st['applied'] > 0
