# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.landuse (land use).

Oracle: tests/oracle/ref_vicwur_lu.c links the unmodified VIC-WUR 3.0
lu_apply.c, lu_support.c and vic_run, sets up random domains and calls
lu_apply() once (one Cv forcing record).  The batched kernels (cv_update +
apply, with the node heat capacities from the TorchVIC port of
distribute_node_moisture_properties) must agree in float64 to 1e-12 on every
state lu_apply writes: Cv, Wdew, the snow stores, snow_canopy, coldcontent,
density, depth, MELTING, surf/pack temperature, the carbon pools, moist,
ice, the node temperatures and Cs_node, the leftover and the routing ring.
Cases: random Cv changes; a tile going to 0; a new tile appearing; VIC's
bare tile; changes inside +-1e-5; normalisation (sum != 1); bands with
AreaFract 0; padding tiles; full columns (overflow lost); LAI 0 and no
overstory (canopy fixes); the paddy/Pbare exchange (Pbare < NVEGTYPES);
leftover to the ring with a negative bin; several nodes, frost areas,
routing sub-steps.
The ledger: water and energy close to round-off in both modes, every term
the C loses or creates named (exact), and nothing lost or created (fixed).
Replay of the R1 G8 run (land use): tests/test_vicwur_landuse_l2.py.
"""
import os
import subprocess

import numpy as np
import pytest
import torch

from torchvic.vicwur import landuse as LU

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, 'oracle', 'build_ref_vicwur_lu.sh')
f64 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)

TB_FIELDS = ('wdew', 'swq', 'pack_water', 'surf_water', 'snow_canopy',
             'coldcontent', 'density', 'depth', 'melting', 'surf_temp',
             'pack_temp', 'annual_npp', 'annual_npp_prev', 'clitter',
             'cinter', 'cslow')


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    d = tmp_path_factory.mktemp('lu')
    env = dict(os.environ, TORCHVIC_ORACLE_DIR=str(d))
    r = subprocess.run(['bash', BUILD], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        pytest.skip('cannot build oracle: ' + (r.stderr + r.stdout)[-300:])
    return str(d / 'ref_vicwur_lu'), d


def run_c(oracle, arr):
    exe, d = oracle
    fi, fo = str(d / 'in.bin'), str(d / 'out.bin')
    np.ascontiguousarray(arr, dtype='<f8').tofile(fi)
    r = subprocess.run([exe, fi, fo], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    return np.fromfile(fo, dtype='<f8'), r.stderr


# ----------------------------------------------------------------- scenario
def scenario(seed, nc=24, nt=6, nb=3, nl=3, nn=5, nf=1, ni=2, rps=4, uh=6,
             K=10, pbare=None, irrigation=True, routing=True, snow=True,
             full=False, subthr=True, edge=True):
    """Random domain and states (numpy).  K = NVEGTYPES; pbare 1-based
    (default K: no paddy exchange)."""
    g = np.random.default_rng(seed)
    pbare = K if pbare is None else pbare
    S = dict(nc=nc, nt=nt, nb=nb, nl=nl, nn=nn, nf=nf, ni=ni, rps=rps, uh=uh,
             K=K, pbare=pbare, irrigation=irrigation, routing=routing,
             full=full, dt=86400.0, doy=int(g.integers(1, 366)),
             msw=0.125)
    S['area'] = g.uniform(1e6, 5e8, nc)
    S['lat'] = g.choice([-1, 1], nc) * g.uniform(0, 60, nc)
    S['nva'] = g.integers(max(2, nt - 2), nt + 1, nc)
    S['air'] = g.uniform(-25, 15, nc)
    S['fs'] = g.integers(0, 2, nc)
    S['binf'] = g.uniform(0.05, 0.4, nc)
    af = g.dirichlet(np.ones(nb), nc)
    if nb > 1:
        af[g.random(nc) < 0.3, -1] = 0.0                     # AreaFract 0
        af /= af.sum(1, keepdims=True)
    S['af'] = af
    S['tfac'] = g.uniform(-3, 3, (nc, nb))
    depth = np.stack([g.uniform(0.05, 0.3, nc), g.uniform(0.2, 0.8, nc),
                      g.uniform(0.5, 2.0, nc)][:nl], 1)
    por = g.uniform(0.35, 0.55, (nc, nl))
    S['depth'], S['por'] = depth, por
    S['maxm'] = por * depth * 1000.0
    S['resid'] = g.uniform(0.01, 0.05, (nc, nl))
    S['wpwp'] = S['maxm'] * g.uniform(0.1, 0.3, (nc, nl))
    S['bdm'] = g.uniform(1300, 1600, (nc, nl))
    S['sdm'] = np.full((nc, nl), 2685.0)
    S['quartz'] = g.uniform(0.2, 0.7, (nc, nl))
    S['bd'] = S['bdm'] * g.uniform(0.95, 1.0, (nc, nl))
    S['sd'] = S['sdm'] * g.uniform(0.98, 1.0, (nc, nl))
    S['org'] = g.uniform(0, 0.2, (nc, nl)) * (g.random((nc, nl)) < 0.5)
    tot = depth.sum(1)
    # nodes: 0 at the surface, the last below the column; some exactly on
    # a layer boundary
    z = np.sort(g.uniform(0.01, 1.0, (nc, nn)), 1) * (tot * 1.3)[:, None]
    z[:, 0] = 0.0
    if nn > 2:
        z[::2, 1] = depth[::2, 0]                           # on a boundary
        z = np.sort(z, 1)
    S['zsum'] = z
    dz = np.empty_like(z)
    dz[:, 0] = (z[:, 1] - z[:, 0]) / 2
    dz[:, -1] = (z[:, -1] - z[:, -2]) / 2
    dz[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / 2
    S['dz'] = dz
    S['mmn'] = g.uniform(0.35, 0.55, (nc, nn))
    S['expt'] = g.uniform(5, 15, (nc, nn))
    S['bub'] = g.uniform(5, 40, (nc, nn))

    # tiles: distinct classes; the last active tile of some cells is VIC's
    # own bare tile (veg_class == K)
    vc = np.zeros((nc, nt), np.int64)
    for c in range(nc):
        vc[c] = g.permutation(K)[:nt]
        if edge and g.random() < 0.3:
            vc[c, S['nva'][c] - 1] = K
    if pbare < K:                       # make sure Pbare is present often
        for c in range(nc):
            if g.random() < 0.8 and (vc[c, :S['nva'][c]] != pbare - 1).all():
                j = g.integers(0, S['nva'][c])
                if vc[c, j] != K:
                    vc[c, j] = pbare - 1
    S['vc'] = vc
    S['over'] = g.integers(0, 2, (nc, nt))
    act = np.arange(nt)[None, :] < S['nva'][:, None]
    S['act'] = act
    cvo = g.dirichlet(np.ones(nt), nc) * act
    if edge:
        cvo[g.random((nc, nt)) < 0.15] = 0.0                 # absent tiles
    cvo /= np.maximum(cvo.sum(1, keepdims=True), 1e-300)
    cvo *= act
    # target Cv_new: random changes, tiles to 0, new tiles, changes inside
    # +-1e-5 and unchanged tiles; one absorber tile per cell restores the
    # sum to 1; the forcing is the target times a factor (normalised away)
    tgt = cvo * g.uniform(0.3, 1.8, (nc, nt))
    kind = g.random((nc, nt))
    if edge:
        tgt[kind < 0.15] = 0.0                               # tile to 0
        new = (cvo == 0) & (g.random((nc, nt)) < 0.5)        # tile appears
        tgt[new] = g.uniform(0.01, 0.3, new.sum())
    fixed = np.zeros((nc, nt), bool)
    if subthr:
        k = (kind >= 0.15) & (kind < 0.35) & (cvo > 1e-5)    # within 1e-5
        tgt[k] = cvo[k] + g.uniform(-9e-6, 9e-6, k.sum())
        same = (kind >= 0.35) & (kind < 0.45)
        tgt[same] = cvo[same]
        fixed = k | same
    tgt = tgt * act * (vc < K)
    for c in range(nc):
        free = [t for t in range(S['nva'][c]) if vc[c, t] < K
                and not fixed[c, t]]
        if not free:
            free = [t for t in range(S['nva'][c]) if vc[c, t] < K]
        a = free[int(np.argmax(tgt[c, free]))] if tgt[c, free].max() > 0 \
            else free[0]
        tgt[c, a] += 1.0 - tgt[c].sum()
        if tgt[c, a] < 0:
            tgt[c, a] = 0.0
            tgt[c] /= tgt[c].sum()
    sc = g.uniform(0.9, 1.1, nc)
    sc[::3] = 1.0
    cvf = tgt * sc[:, None]
    S['cvo'], S['cvf'] = cvo, cvf * (vc < K)

    sh = (nc, nt, nb)
    S['lai'] = g.uniform(0, 4, sh) * (g.random(sh) < 0.8)
    S['wdew'] = g.uniform(0, 1.5, sh) * (g.random(sh) < 0.7)
    swq = g.uniform(0, 0.4, sh) * (g.random(sh) < (0.6 if snow else 0.0))
    S['swq'] = swq
    S['surf_water'] = swq * g.uniform(0, 0.2, sh)
    S['pack_water'] = swq * g.uniform(0, 0.2, sh)
    S['snow_canopy'] = g.uniform(0, 0.02, sh) * (g.random(sh) < 0.4)
    S['coldcontent'] = g.uniform(-5e5, 1e5, sh)
    S['density'] = g.uniform(100, 400, sh) * (g.random(sh) < 0.8)
    S['depth_s'] = g.uniform(0, 2, sh)
    S['last_snow'] = g.integers(0, 4, sh).astype(float)
    S['melting'] = g.integers(0, 2, sh).astype(float)
    S['surf_temp'] = g.uniform(-10, 0, sh)
    S['pack_temp'] = g.uniform(-10, 0, sh)
    for k in ('annual_npp', 'annual_npp_prev', 'clitter', 'cinter', 'cslow'):
        S[k] = g.uniform(0, 500, sh)
    frac = g.uniform(0.2, 1.0, sh + (nl,))
    full_col = g.random(sh) < 0.15                           # saturated
    frac[full_col] = 1.0
    S['moist'] = frac * S['maxm'][:, None, None, :]
    S['ice'] = S['moist'][..., None] * g.uniform(0, 0.3, sh + (nl, nf)) \
        * (g.random(sh + (nl, nf)) < 0.3)
    S['T'] = g.uniform(-8, 20, sh + (nn,))
    # irrigation entries: distinct active non-bare tiles, some paddy
    vi = np.full((nc, max(ni, 1)), -1, np.int64)
    pad = np.zeros((nc, max(ni, 1)), bool)
    for c in range(nc):
        cand = [t for t in range(S['nva'][c]) if vc[c, t] != K
                and vc[c, t] != pbare - 1]
        g.shuffle(cand)
        for i in range(min(ni, len(cand))):
            vi[c, i] = cand[i]
            pad[c, i] = g.random() < 0.6
    S['irr_vi'], S['irr_paddy'] = vi[:, :ni], pad[:, :ni]
    S['leftover'] = g.uniform(0, 30, (nc, ni, nb)) * (g.random((nc, ni, nb))
                                                      < 0.7)
    R = uh + rps - 1
    ring = g.uniform(0, 50, (nc, R))
    ring[g.random((nc, R)) < 0.05] = -1e-3                   # clip at 0
    S['ring'] = ring
    return S


def c_input(S):
    nc, nt, nb, nl, nn, nf, ni = (S[k] for k in ('nc', 'nt', 'nb', 'nl',
                                                  'nn', 'nf', 'ni'))
    a = [nc, nt, nb, nl, nn, nf, ni, S['rps'], S['uh'], S['K'], S['pbare'],
         float(S['irrigation']), float(S['routing']), float(S['full']),
         S['dt'], S['doy'], S['msw']]
    for c in range(nc):
        a += [S['area'][c], S['lat'][c], S['nva'][c], S['air'][c],
              S['fs'][c], S['binf'][c]]
        for b in range(nb):
            a += [S['af'][c, b], S['tfac'][c, b]]
        for l in range(nl):
            a += [S['maxm'][c, l], S['depth'][c, l], S['por'][c, l],
                  S['resid'][c, l], S['wpwp'][c, l], S['bdm'][c, l],
                  S['sdm'][c, l], S['quartz'][c, l], S['bd'][c, l],
                  S['sd'][c, l], S['org'][c, l]]
        for n in range(nn):
            a += [S['dz'][c, n], S['zsum'][c, n], S['mmn'][c, n],
                  S['expt'][c, n], S['bub'][c, n]]
        for t in range(nt):
            a += [S['vc'][c, t], S['over'][c, t], S['cvo'][c, t],
                  S['cvf'][c, t]]
        for t in range(nt):
            for b in range(nb):
                a += [S['lai'][c, t, b], S['wdew'][c, t, b], S['swq'][c, t, b],
                      S['pack_water'][c, t, b], S['surf_water'][c, t, b],
                      S['snow_canopy'][c, t, b], S['coldcontent'][c, t, b],
                      S['density'][c, t, b], S['depth_s'][c, t, b],
                      S['last_snow'][c, t, b], S['melting'][c, t, b],
                      S['surf_temp'][c, t, b], S['pack_temp'][c, t, b],
                      S['annual_npp'][c, t, b], S['annual_npp_prev'][c, t, b],
                      S['clitter'][c, t, b], S['cinter'][c, t, b],
                      S['cslow'][c, t, b]]
                a += list(S['moist'][c, t, b])
                a += list(S['ice'][c, t, b].reshape(-1))
                a += list(S['T'][c, t, b])
        a += [int((S['irr_vi'][c] >= 0).sum())]
        for i in range(ni):
            a += [max(S['irr_vi'][c, i], 0), float(S['irr_paddy'][c, i])]
            a += list(S['leftover'][c, i])
        a += list(S['ring'][c])
    return np.array(a, dtype=np.float64)


def parse_c(S, out):
    nc, nt, nb, nl, nn, nf, ni = (S[k] for k in ('nc', 'nt', 'nb', 'nl',
                                                  'nn', 'nf', 'ni'))
    R = S['uh'] + S['rps'] - 1
    ntb = len(TB_FIELDS) + nl + nl * nf + 2 * nn
    per = nt + nt * nb * ntb + ni * nb + R + 3
    assert out.size == nc * per
    o = out.reshape(nc, per)
    r = dict(cv=o[:, :nt])
    tb = o[:, nt:nt + nt * nb * ntb].reshape(nc, nt, nb, ntb)
    for j, k in enumerate(TB_FIELDS):
        r[k] = tb[..., j]
    j = len(TB_FIELDS)
    r['moist'] = tb[..., j:j + nl]
    r['ice'] = tb[..., j + nl:j + nl + nl * nf].reshape(nc, nt, nb, nl, nf)
    j += nl + nl * nf
    r['t_node'] = tb[..., j:j + nn]
    r['cs_node'] = tb[..., j + nn:j + 2 * nn]
    p = nt + nt * nb * ntb
    r['leftover'] = o[:, p:p + ni * nb].reshape(nc, ni, nb)
    p += ni * nb
    r['ring'] = o[:, p:p + R]
    r['discharge'], r['stream'] = o[:, p + R], o[:, p + R + 1]
    r['fatal'] = o[:, p + R + 2] != 0
    return r


def torch_inputs(S, device='cpu', mode='exact'):
    """State dict, P and the Cv pair for the kernels."""
    d = lambda a: f64(a).to(device)
    nc, nt, K = S['nc'], S['nt'], S['K']
    st = dict(wdew=d(S['wdew']), swq=d(S['swq']),
              pack_water=d(S['pack_water']), surf_water=d(S['surf_water']),
              snow_canopy=d(S['snow_canopy']),
              coldcontent=d(S['coldcontent']), density=d(S['density']),
              depth=d(S['depth_s']), last_snow=d(S['last_snow']),
              melting=d(S['melting']) != 0, surf_temp=d(S['surf_temp']),
              pack_temp=d(S['pack_temp']), moist=d(S['moist']),
              ice=d(S['ice']), t_node=d(S['T']),
              leftover=d(S['leftover']), ring=d(S['ring']))
    for k in LU.CARBON:
        st[k] = d(S[k])
    vc = torch.as_tensor(S['vc'], device=device)
    act = torch.as_tensor(S['act'], device=device)
    paddy = torch.zeros(nc, nt, dtype=torch.bool, device=device)
    vi = torch.as_tensor(S['irr_vi'], device=device)
    pad = torch.as_tensor(S['irr_paddy'], device=device)
    for i in range(S['ni']):
        paddy |= (pad[:, i:i + 1] & (vi[:, i:i + 1] >= 0)
                  & (torch.arange(nt, device=device) == vi[:, i:i + 1]))
    pbt = act & (vc == S['pbare'] - 1)
    pb_idx = torch.where(pbt.any(1), pbt.long().argmax(1),
                         torch.full((nc,), -1, device=device))
    order = torch.cat([torch.where(pad & (vi >= 0), vi,
                                   torch.full_like(vi, -1)),
                       pb_idx[:, None]], 1)
    soil = dict(zsum_node=d(S['zsum']), max_moist_node=d(S['mmn']),
                expt_node=d(S['expt']), bubble_node=d(S['bub']),
                depth=d(S['depth']), soil_dens_min=d(S['sdm']),
                bulk_dens_min=d(S['bdm']), quartz=d(S['quartz']),
                soil_density=d(S['sd']), bulk_density=d(S['bd']),
                organic=d(S['org']),
                frost_fract=torch.full((nc, S['nf']), 1.0 / S['nf'],
                                       dtype=torch.float64, device=device),
                fs_active=torch.as_tensor(S['fs'] != 0, device=device))
    P = dict(area_fract=d(S['af']), max_moist=d(S['maxm']),
             overstory=torch.as_tensor(S['over'] != 0, device=device),
             active=act,
             bare=act & (vc == K), paddy=paddy & act, pbare=pbt,
             lai=d(S['lai']), dz_node=d(S['dz']), max_surface_swe=S['msw'],
             doy=S['doy'], lat=d(S['lat']),
             new_density=LU.band_new_density(d(S['air']), d(S['tfac'])),
             paddy_order=order, irr_tile=vi, area=d(S['area']), dt=S['dt'],
             rps=S['rps'], uh_length=S['uh'])
    P['soil'] = soil
    return st, P, LU.lu_heat_capacity_fn(soil, mode)


def run_torch(S, mode='exact', device='cpu'):
    st, P, cs_fn = torch_inputs(S, device, mode)
    cvo = f64(S['cvo']).to(device)
    cvn, dcv, s, fatal, warn = LU.cv_update(f64(S['cvf']).to(device), cvo,
                                            P['active'])
    st1, terms = LU.apply(
        st, cvo, cvn, P, mode=mode,
        paddy_exchange=S['irrigation'] and S['pbare'] != S['K'],
        route_leftover=S['irrigation'] and S['routing'], cs_fn=cs_fn,
        full_energy=S['full'])
    st1['cs_node'] = LU.node_heat_capacity(st1['moist'], st1['t_node'],
                                           P['soil'])
    return st, st1, cvo, cvn, terms, P, fatal


def close(a, b, what, rtol=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    err = np.abs(a - b) / np.maximum(1.0, np.abs(b))
    assert np.all(err <= rtol), '%s: max %.3e at %s' % (
        what, err.max(), np.unravel_index(err.argmax(), err.shape))
    return float(err.max())


def compare(S, c, st1, cvn, fatal=None, cv_fatal=None):
    """Every state lu_apply writes, on the active tiles (the cells whose Cv
    sum is <= 0 are skipped: the C goes on with inf/NaN there, the kernels
    keep the old Cv), and the fatal flags."""
    ok = np.ones(S['nc'], bool) if cv_fatal is None else \
        ~cv_fatal.cpu().numpy()
    act = S['act'] & ok[:, None]
    tb = np.broadcast_to(act[:, :, None], st1['wdew'].shape)
    worst = close(cvn.cpu().numpy()[ok], c['cv'][ok], 'cv')
    tbb = tb & (S['af'][:, None, :] > 0)
    for k in TB_FIELDS + ('moist', 'ice', 't_node', 'cs_node'):
        a = st1[k].cpu().numpy().astype(np.float64)
        m = (tbb if k == 'cs_node' else tb)
        m = m.reshape(m.shape + (1,) * (a.ndim - 3))
        m = np.broadcast_to(m, a.shape)
        worst = max(worst, close(a[m], c[k][m], k))
    if S['irrigation'] and S['routing']:
        worst = max(worst, close(st1['leftover'].cpu().numpy()[ok],
                                 c['leftover'][ok], 'leftover'))
        worst = max(worst, close(st1['ring'].cpu().numpy()[ok],
                                 c['ring'][ok], 'ring'))
    if fatal is not None:
        assert (fatal.cpu().numpy() == c['fatal']).all(), (
            np.nonzero(fatal.cpu().numpy() != c['fatal']))
    return worst


CASES = [
    dict(seed=1),
    dict(seed=2, nb=1, nn=3, rps=1, uh=4),
    dict(seed=3, nf=2, nn=10, nt=8, K=14),
    dict(seed=4, pbare=7, K=10),                       # paddy exchange
    dict(seed=5, pbare=5, K=8, ni=3, nt=7),
    dict(seed=6, irrigation=False),
    dict(seed=7, routing=False),
    dict(seed=8, snow=False, subthr=False),
    dict(seed=9, edge=False),
    dict(seed=10, nc=40, nt=4, nb=5, pbare=9, K=12, ni=2, rps=2, uh=8),
    dict(seed=11, nl=2, nn=4),
    dict(seed=12, full=True),                          # FULL_ENERGY check
    dict(seed=13, nf=3, nb=2, K=16, nt=10, ni=4),
]


@pytest.mark.parametrize('kw', CASES, ids=[str(i) for i in range(len(CASES))])
def test_lu_apply_matches_c(oracle, kw):
    S = scenario(**kw)
    out, err = run_c(oracle, c_input(S))
    c = parse_c(S, out)
    st0, st1, cvo, cvn, terms, P, fatal = run_torch(S, 'exact')
    compare(S, c, st1, cvn, fatal | terms['fatal'], fatal)


def _edge_case(kind):
    """One small cell per edge case, driven to the C."""
    S = scenario(100, nc=4, nt=4, nb=2, ni=1, edge=False, subthr=False,
                 pbare=4 if kind == 'paddy' else None, K=6)
    S['af'][:] = [0.6, 0.4]
    S['nva'][:] = 4
    S['act'][:] = True
    S['vc'][:] = [0, 1, 2, 3] if kind == 'paddy' else [0, 1, 2, 6]
    cvo = np.array([0.4, 0.3, 0.3, 0.0])
    if kind == 'to_zero':
        cvf = np.array([0.0, 0.5, 0.5, 0.0])
    elif kind == 'appear':
        cvf = np.array([0.3, 0.3, 0.2, 0.2])          # bare tile appears
    elif kind == 'bare_only':
        cvo = np.array([0.5, 0.0, 0.0, 0.5])
        cvf = np.array([0.2, 0.0, 0.0, 0.8])
    elif kind == 'paddy':
        cvo = np.array([0.4, 0.3, 0.2, 0.1])
        cvf = np.array([0.4, 0.1, 0.2, 0.3])          # paddy -> Pbare
    elif kind == 'full':
        cvf = np.array([0.1, 0.45, 0.45, 0.0])
    S['cvo'][:] = cvo
    S['cvf'][:] = cvf * (S['vc'] < S['K'])
    if kind == 'bare_only':
        S['cvo'][:, 3] = 0.5
    if kind == 'paddy':
        S['irr_vi'][:] = 1
        S['irr_paddy'][:] = True
    if kind == 'full':
        S['moist'][:] = S['maxm'][:, None, None, :]
        S['lai'][:] = 0.0
        S['wdew'][:] = 5.0
    return S


@pytest.mark.parametrize('kind', ['to_zero', 'appear', 'bare_only', 'paddy',
                                  'full'])
def test_lu_edge_cases_match_c(oracle, kind):
    S = _edge_case(kind)
    out, err = run_c(oracle, c_input(S))
    c = parse_c(S, out)
    st0, st1, cvo, cvn, terms, P, fatal = run_torch(S, 'exact')
    compare(S, c, st1, cvn, fatal | terms['fatal'], fatal)
    if kind == 'full':
        assert 'soil is full' in err                   # the C's warning
        assert float(terms['lost'].sum()) > 0


@pytest.mark.parametrize('kw', CASES, ids=[str(i) for i in range(len(CASES))])
@pytest.mark.parametrize('mode', ['exact', 'fixed'])
def test_ledger_closes(kw, mode):
    """Water (mm over the cell) and energy (the C's capacity-weighted
    temperatures) close across the Cv change; in `fixed` nothing is created
    or lost except the overflow runoff."""
    S = scenario(**kw)
    st0, st1, cvo, cvn, terms, P, fatal = run_torch(S, mode)
    led = LU.ledger(st0, st1, cvo, cvn, terms, P['area_fract'],
                    P if S['irrigation'] and S['routing'] else None)
    scale = led['storage_before'].abs().clamp(min=1.0)
    assert float((led['residual'].abs() / scale).max()) < 1e-12
    escale = led['energy_before'].abs().clamp(min=1.0)
    assert float((led['energy_residual'].abs() / escale).max()) < 1e-12
    if mode == 'fixed':
        # nothing created or lost: subthreshold is 0 by construction; an
        # orphan pool is left only by Cv round-off (|dCv| ~ 1e-16 givers
        # with no receiver anywhere)
        for k in ('subthreshold', 'energy_subthreshold'):
            assert float(terms[k].abs().max()) == 0.0, k
        for k, sc in (('orphan', scale), ('energy_orphan', escale),
                      ('energy_lost', escale)):
            v = (terms[k].abs() * P['area_fract']).sum(1)
            assert float((v / sc).max()) < 1e-12, k
        if 'leftover_routed' in terms:
            torch.testing.assert_close(terms['leftover_routed'],
                                       terms['leftover_removed'],
                                       rtol=1e-13, atol=1e-12)


def test_fixed_is_the_default_and_node0_heat_capacity_is_its_own():
    """Upstream bug fixed (default): lu_support.c:274-277 writes Cs[2] past
    the array, so exact gives node 0 layer 1's heat capacity. The default
    (fixed) uses node 0's own, and the water and energy ledger closes."""
    import inspect
    assert inspect.signature(LU.apply).parameters['mode'].default == 'fixed'
    assert inspect.signature(
        LU.lu_heat_capacity_fn).parameters['mode'].default == 'fixed'
    S = scenario(**CASES[0])
    st, P, _ = torch_inputs(S, 'cpu', 'fixed')
    own = LU.node_heat_capacity(st['moist'], st['t_node'], P['soil'])
    cs_fixed = LU.lu_heat_capacity_fn(P['soil'])(st['moist'], st['ice'],
                                                 st['t_node'])
    cs_exact = LU.lu_heat_capacity_fn(P['soil'], 'exact')(
        st['moist'], st['ice'], st['t_node'])
    torch.testing.assert_close(cs_fixed, own, rtol=0, atol=0)
    lay = LU.layer_heat_capacity(st['moist'], st['ice'], P['soil'])
    torch.testing.assert_close(cs_exact[..., 0], lay[..., 1], rtol=0, atol=0)
    assert float((cs_exact[..., 0] - own[..., 0]).abs().max()) > 0
    torch.testing.assert_close(cs_exact[..., 1:], own[..., 1:], rtol=0,
                               atol=0)
    st0, st1, cvo, cvn, terms, P, _ = run_torch(S, 'fixed')
    led = LU.ledger(st0, st1, cvo, cvn, terms, P['area_fract'],
                    P if S['irrigation'] and S['routing'] else None)
    assert float((led['energy_residual'].abs()
                  / led['energy_before'].abs().clamp(min=1.0)).max()) < 1e-12
    assert float((led['residual'].abs()
                  / led['storage_before'].abs().clamp(min=1.0)).max()) < 1e-12


def test_exact_terms_are_reached():
    """The exact-mode quirks the ledger names do occur on the cases."""
    tot = dict(sub=0.0, lost=0.0, orphan=0.0, routed_excess=0.0)
    for kw in CASES:
        S = scenario(**kw)
        _, _, _, _, t, P, _ = run_torch(S, 'exact')
        tot['sub'] += float(t['subthreshold'].abs().sum())
        tot['lost'] += float(t['lost'].sum())
        tot['orphan'] += float(t['orphan'].abs().sum())
        if 'leftover_routed' in t:
            tot['routed_excess'] += float((t['leftover_routed']
                                           - t['leftover_removed']).sum())
    S = _edge_case('full')
    tot['lost'] += float(run_torch(S, 'exact')[4]['lost'].sum())
    assert tot['sub'] > 0 and tot['lost'] > 0 and tot['routed_excess'] > 0


def test_cv_forcing_map_and_checks():
    forced = f64([[0.2, 0.0, 0.5, 0.3], [0.1, 0.4, 0.0, 0.5],
                  [0.0, 0.0, 0.0, 0.0]])
    tc = torch.tensor([[0, 2, 3, -1], [0, 1, -1, -1], [0, 1, 2, 3]])
    cv, fatal = LU.map_cv_forcing(forced, tc)
    assert cv.tolist() == [[0.2, 0.5, 0.3, 0.0], [0.1, 0.4, 0.0, 0.0],
                           [0.0] * 4]
    assert fatal.tolist() == [False, True, False]      # class 3 has no tile
    act = tc >= 0
    act[0, 3] = True                                   # VIC's bare tile
    cvn, dcv, s, fat, warn = LU.cv_update(cv, f64([[0.5, 0.2, 0.2, 0.1],
                                                   [0.5, 0.5, 0, 0],
                                                   [0.25] * 4]), act)
    assert fat.tolist() == [False, False, True]
    assert warn.tolist() == [False, True, True]
    torch.testing.assert_close(cvn[1], f64([0.2, 0.8, 0.0, 0.0]))
    torch.testing.assert_close(cvn[2], f64([0.25] * 4))  # kept on fatal


def test_no_paddy_exchange_when_pbare_is_last_class():
    """irr_start.c:43-51 allows Pbare = NVEGTYPES only with NBARE = 1; then
    lu_apply.c:308-309 skips the paddy step: the same as IRRIGATION off for
    the redistribution."""
    S = scenario(12, pbare=10, K=10)
    a = run_torch(S, 'exact')[1]
    S2 = dict(S, irrigation=False)
    b = run_torch(S2, 'exact')[1]
    for k in ('moist', 'swq', 'wdew'):
        assert torch.equal(a[k], b[k])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_matches_c_without_host_sync(oracle):
    """The kernels on CUDA agree with the C to 1e-12 and do no host read
    (sync debug mode raises on any)."""
    for kw in (CASES[0], CASES[3], CASES[9]):
        S = scenario(**kw)
        c = parse_c(S, run_c(oracle, c_input(S))[0])
        st, P, cs_fn = torch_inputs(S, 'cuda')
        cvo = f64(S['cvo']).cuda()
        cvf = f64(S['cvf']).cuda()
        kw = dict(mode='exact', paddy_exchange=S['pbare'] != S['K'],
                  route_leftover=True, cs_fn=cs_fn)
        # warm-up: the core's cached device constants (core/devconst.py) are
        # made on the first call; the steady state does no host read
        LU.apply(st, cvo, LU.cv_update(cvf, cvo, P['active'])[0], P, **kw)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode('error')
        try:
            cvn, dcv, s, fatal, warn = LU.cv_update(cvf, cvo, P['active'])
            st1, terms = LU.apply(st, cvo, cvn, P, **kw)
            st1['cs_node'] = LU.node_heat_capacity(st1['moist'],
                                                   st1['t_node'], P['soil'])
            led = LU.ledger(st, st1, cvo, cvn, terms, P['area_fract'], P)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        compare(S, c, st1, cvn)
        assert float(led['residual'].abs().max()) < 1e-9
