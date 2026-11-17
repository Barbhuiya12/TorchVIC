# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.vicwur.co2 (the CO2 forcing and the b_co2 hook).

Oracle: tests/oracle/ref_vicwur_co2.c links the unchanged
VIC-WUR 3.0 vic_run/src/penman.c (calc_rc with the co2factor) and copies
co2_update_step_vars.c's Catm arithmetic; the float64 kernels must match it
to 1e-12 on random and edge inputs (b_co2 = 0, negative and so large that
the factor is not positive; rs = 0; lai = 0; reference crop; sub-steps 1, 2,
4; default Catm).
Hook-off identity: calc_rc / transpiration with co2factor=None are the VIC 5
expressions, bit for bit, and a factor of 1 changes nothing.
Replay: the R1 G9 run (VICWUR_DUMP build, OUT_CATM added): the held yearly CO2
record read by Co2Forcing gives the OUT_CATM of every cell and day, and the
b_co2 of the parameter file.
"""
import glob
import os
import _paths  # noqa: E402
import subprocess

import numpy as np
import pytest
import torch

from torchvic.core import vegetation as V
from torchvic.io import ncshare
from torchvic.vicwur import co2 as C
from torchvic.vicwur import forcing as F
from torchvic.vicwur import params as P

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'oracle', 'ref_vicwur_co2.c')
W3 = _paths.env_path(('TORCHVIC_VICWUR_SRC', 'VICWUR_SRC'), 'build/vicwur3/src_tree/vic')
VER = _paths.env_path(('TORCHVIC_VICWUR_VERIFY',), 'verification')
G9 = os.path.join(VER, 'vicwur_14', 'gruns', 'R1', 'G9')
R1 = _paths.env_path(('TORCHVIC_VICWUR_R1', 'VICWUR_R1'), 'vicwur/R1')
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    if not os.path.exists(os.path.join(W3, 'vic_run', 'src', 'penman.c')):
        pytest.skip('W3 source tree absent: ' + W3)
    exe = str(tmp_path_factory.mktemp('co2') / 'ref_co2')
    r = subprocess.run(
        ['cc', '-o', exe, SRC, os.path.join(W3, 'vic_run', 'src', 'penman.c'),
         '-I' + os.path.join(W3, 'vic_run', 'include'),
         '-I' + os.path.join(W3, 'drivers', 'shared_all', 'include'),
         '-O0', '-std=c99', '-ffp-contract=off', '-fcommon',
         '-ffunction-sections', '-Wl,--gc-sections', '-lm'],
        capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('cannot build oracle: ' + r.stderr[-300:])
    return exe


def make_cases(n, seed):
    g = np.random.default_rng(seed)

    def z(a, p):
        return np.where(g.random(n) < p, 0.0, a)

    d = {}
    d['rs'] = z(g.uniform(20, 400, n), .1)
    d['ns'] = z(g.uniform(0, 600, n), .1)       # net_short >= 0 (see below)
    d['rgl'] = g.uniform(30, 150, n)
    d['tair'] = np.where(g.random(n) < .1, g.uniform(-25, 4, n),
                         g.uniform(4, 45, n))
    d['vpd'] = np.where(g.random(n) < .15, g.uniform(3500, 9000, n),
                        g.uniform(0, 3500, n))
    d['co2'] = np.where(g.random(n) < .1, 366.0, g.uniform(280, 900, n))
    d['lai'] = z(g.uniform(0.1, 8, n), .1)
    d['b'] = np.where(g.random(n) < .3, 0.0, g.uniform(0, 0.6, n))
    neg = g.random(n) < .05
    d['b'] = np.where(neg, g.uniform(-0.5, 0, n), d['b'])
    d['gsm'] = np.where(g.random(n) < .4, 1.0, g.uniform(0.02, 1, n))
    d['ref'] = (g.random(n) < .08).astype(float)
    d['dflt'] = (g.random(n) < .1).astype(float)
    # factor <= 0: b (1 - co2/ref) >= 1, which no veg library has, but C
    # computes it (no guard)
    bad = np.arange(8)
    d['b'][bad] = 3.0 + bad
    d['co2'][bad] = 100.0 + bad
    d['ref'][bad] = 0.0
    d['rs'][bad] = 100.0
    d['lai'][bad] = 3.0
    d['gsm'][bad] = 1.0
    d['dflt'][bad] = 0.0
    d['b'][8:12] = 0.0                           # exactly VIC 5
    return d


def run_oracle(exe, d, nf, ref, tmp_path):
    n = len(d['rs'])
    rec = np.stack([d['rs'], d['ns'], d['rgl'], d['tair'], d['vpd'],
                    d['co2'], d['lai'], d['b'], d['gsm'], d['ref'],
                    d['dflt']], 1)
    fi, fo = str(tmp_path / 'in.bin'), str(tmp_path / 'out.bin')
    with open(fi, 'wb') as f:
        f.write(np.array([n, nf, ref], '<f8').tobytes())
        f.write(rec.astype('<f8').tobytes())
    subprocess.run([exe, fi, fo], check=True)
    return np.fromfile(fo, '<f8').reshape(n, 8)


def run_torch(d, nf, ref, device='cpu'):
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64,
                                  device=device)
    n = len(d['rs'])
    args = [t(d[k]) for k in ('rs', 'ns', 'rgl', 'tair', 'vpd')]
    lai, b, gsm = t(d['lai']), t(d['b']), t(d['gsm'])
    refc = d['ref'][0] > 0
    co2 = t(d['co2'])
    dflt = t(d['dflt']) > 0
    out = {}
    # rc with the raw forcing ppm
    rc_raw, rc_cat = [], []
    for rc_ref in (False, True):
        sel = (d['ref'] > 0) == rc_ref
        if not sel.any():
            continue
        ix = torch.as_tensor(np.nonzero(sel)[0], device=device)
        a = [x[ix] for x in args]
        fac = C.co2factor(b[ix], co2[ix], ref)
        r1 = V.calc_rc(a[0], a[1], a[2], a[3], a[4], lai[ix], gsm[ix],
                       rc_ref, co2factor=fac)
        catm = C.catm_from_co2(co2[ix], nf)
        catm = torch.where(dflt[ix, None], C.catm_default(
            len(ix), nf, ref, device=device), catm)
        r2 = C.calc_rc_co2(a[0], a[1], a[2], a[3], a[4], lai[ix], gsm[ix],
                           b[ix], catm[:, 0], rc_ref, ref)
        rc_raw.append((ix, r1))
        rc_cat.append((ix, r2, catm))
    o = torch.zeros(n, 8, dtype=torch.float64, device=device)
    for ix, r1 in rc_raw:
        o[ix, 0] = r1
    for ix, r2, catm in rc_cat:
        o[ix, 1:1 + nf] = catm[:, :nf]
        o[ix, 5] = catm[:, nf]
        o[ix, 6] = C.ppm_from_catm(catm[:, 0])
        o[ix, 7] = r2
    return o.cpu().numpy()


CFG = [(1, 366.0), (2, 366.0), (4, 390.0), (1, 350.5)]


@pytest.mark.parametrize('ic', range(len(CFG)))
def test_l1_random_edge(oracle, tmp_path, ic):
    nf, ref = CFG[ic]
    d = make_cases(6000, 100 + ic)
    o = run_oracle(oracle, d, nf, ref, tmp_path)
    r = run_torch(d, nf, ref)
    np.testing.assert_allclose(r, o, rtol=1e-12, atol=1e-12)
    # coverage: unclamped resistances that the factor moved, clamped ones,
    # the non-positive factor, the default-Catm rows
    base = o[:, 0]
    assert ((base > 0) & (base < 5000)).sum() > 2000
    assert (base == 5000).any() and (base < 0).any()
    assert (np.abs(o[:, 6] - d['co2']) > 0).any()        # round trip inexact
    assert (d['dflt'] > 0).any()


def test_hooks_off_identity():
    g = torch.Generator().manual_seed(0)
    n = 4000
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, dtype=torch.float64,
                                                   generator=g)
    a = (u(20, 400), u(0, 600), u(30, 150), u(-5, 40), u(0, 6000), u(.1, 8),
         u(0.02, 1))
    r0 = V.calc_rc(*a)
    r1 = V.calc_rc(*a, co2factor=torch.ones_like(a[0]))
    r2 = V.calc_rc(*a, co2factor=None)
    assert torch.equal(r0, r1) and torch.equal(r0, r2)
    # b_co2 = 0 gives a factor of exactly 1
    f = C.co2factor(torch.zeros(n, dtype=torch.float64), u(280, 900), 366.0)
    assert torch.equal(f, torch.ones_like(f))


def _transp_inputs(n, seed, L=3):
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(
        *(s or (n,)), dtype=torch.float64, generator=g)
    wc = u(60, 120, n, 1, L)
    root = u(0.1, 1, n, 1, L)
    root = root / root.sum(-1, keepdim=True)
    return dict(
        avail_moist=wc * u(0.3, 1.6, n, 1, L), w_crit=wc, w_wilt=wc * .5,
        root_frac=root, lai=u(0.5, 6, n, 1), rmin=u(60, 250, n, 1),
        rgl=u(60, 150, n, 1), net_short=u(0, 500, n, 1), tair=u(5, 35, n, 1),
        vpd=u(200, 3500, n, 1), ra=u(20, 120, n, 1), rarc=u(0, 60, n, 1),
        rad=u(50, 600, n, 1), elevation=u(0, 2000, n, 1))


@pytest.mark.parametrize('share', [True, False])
def test_transpiration_hook(share):
    k = _transp_inputs(500, 3)
    t0 = V.transpiration(**k, share_layer_moist=share)
    ones = torch.ones_like(k['lai'])
    assert torch.equal(t0, V.transpiration(**k, share_layer_moist=share,
                                           co2factor=ones))
    assert torch.equal(t0, V.transpiration(**k, share_layer_moist=share,
                                           co2factor=None))
    # elevated CO2 with b_co2 > 0: factor < 1 raises rc and lowers
    # transpiration, never raises it
    f = C.co2factor(torch.full_like(ones, 0.2), torch.full_like(ones, 600.))
    assert (f < 1).all()
    t1 = V.transpiration(**k, share_layer_moist=share, co2factor=f)
    assert (t1 <= t0 + 1e-12).all() and (t1 < t0 - 1e-9).any()
    # the factor enters the resistance exactly as calc_rc does: with
    # return_rc the canopy resistance of the CASE 1 Penman is calc_rc(co2f)
    _, rc0 = V.transpiration(**k, share_layer_moist=share, return_rc=True)
    _, rc1 = V.transpiration(**k, share_layer_moist=share, return_rc=True,
                             co2factor=f)
    assert (rc1 >= rc0 - 1e-9).all()


def test_catm_and_helpers():
    ppm = torch.tensor([375.0, 377.0, 410.5], dtype=torch.float64)
    c = C.catm_from_co2(ppm, 2)
    assert c.shape == (3, 3)
    assert torch.equal(c[:, 0], ppm * 1e-6) and torch.equal(c[:, 1], c[:, 0])
    assert torch.equal(c[:, 2], (c[:, 0] + c[:, 1]) / 2)
    # n == 1: the first element, no division
    assert torch.equal(C.catm_from_co2(ppm, 1)[:, 1], ppm * 1e-6)
    d = C.catm_default(4, 3)
    assert d.shape == (4, 4) and torch.equal(
        d[:, 0], torch.full((4,), 366.0 * 1e-6, dtype=torch.float64))
    # tile factor: bare / out-of-range classes have b_co2 = 0 -> factor 1
    b = torch.tensor([[0.0, 0.1], [0.2, 0.0], [0.3, 0.4]], dtype=torch.float64)
    vi = torch.tensor([[0, 1, -1], [2, 5, 0]])
    f = C.tile_factor(b, vi, C.catm_from_co2(torch.tensor([732., 366.],
                                                         dtype=torch.float64)))
    ref = torch.tensor([[1 / (1 + 0.0 * 1), 1 / (1 + 0.2 * 1), 1.0],
                        [1 / (1 + 0.4 * 0.0), 1.0, 1 / (1 + 0.1 * 0.0)]],
                       dtype=torch.float64)
    torch.testing.assert_close(f, ref, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        C.average(torch.zeros(2, 0), 0)


# ------------------------------------------------------- the R1 G9 run
def _g9():
    gpath = os.path.join(G9, 'global_W3.txt')
    if not os.path.exists(gpath) or not glob.glob(
            os.path.join(G9, 'W3', 'out', 'fluxes.*.nc')):
        pytest.skip('G9 run absent: ' + G9)
    return P.parse_global(gpath)


def test_l2_g9_catm():
    g = _g9()
    start = (2003, 7, 1, 0)
    files = sorted(glob.glob(os.path.join(G9, 'W3', 'out', 'fluxes.*.nc')))
    cat = []
    for fn in files:
        ds = ncshare.read(fn, decode_times=False, mask_and_scale=False)
        a = np.asarray(ds['OUT_CATM'].values)
        cat.append(a.reshape(a.shape[0], -1))
        ds.close()
    cat = np.concatenate(cat)
    nsteps = len(cat)
    assert nsteps == 2557
    prefix = [w for w in g['PLUGIN_FORCE_TYPE'] if w[0] == 'CO2'][0][3]
    ms = F.file_info('%s%4d.nc' % (prefix, 2003), 'co2')['start']
    f = C.Co2Forcing.from_global(g, start, nsteps, ms)
    try:
        got = np.stack([C.ppm_from_catm(f.catm(t)[:, 1]).numpy()
                        for t in range(nsteps)])
        # the file value, held, on the day grid of the run
        ppm = np.stack([f.ppm(t).numpy() for t in range(nsteps)])
        assert f.reads(0) and f.reads(184)           # 2004-01-01 new record
    finally:
        f.close()
    assert got.shape == cat.shape
    # OUT_CATM is written as float32 of Catm[NR] / 1e-6
    np.testing.assert_array_equal(got.astype(np.float32), cat)
    assert len(np.unique(ppm)) >= 7                  # one value per year
    # held: constant inside a calendar year
    yr = np.array([(np.datetime64('2003-07-01') + t).astype(object).year
                   for t in range(nsteps)])
    for y in np.unique(yr):
        assert (ppm[yr == y] == ppm[yr == y][0]).all()


def test_l2_g9_bco2():
    g = _g9()
    inp = P.load_vicwur_inputs(os.path.join(G9, 'global_W3.txt'))
    b = inp.b_co2
    assert b is not None and b.shape[0] == 18
    assert set(np.unique(b.numpy())) <= {0.0, 0.1, 0.2} and (b > 0).any()
    # tile factors of cell 0 at the first record: vegetated classes move,
    # a class with b_co2 = 0 does not
    vi = torch.arange(18).unsqueeze(0).expand(b.shape[1], -1)
    cat = C.catm_from_co2(torch.full((b.shape[1],), 375.0,
                                     dtype=torch.float64))
    fac = C.tile_factor(b, vi, cat)
    assert (fac[b.t() == 0] == 1).all()
    assert (fac[b.t() > 0] < 1).all()              # 375 > 366: stress rises


# --------------------------------------------------------------- devices
@pytest.mark.parametrize('device', DEVICES)
def test_devices_agree(oracle, tmp_path, device):
    d = make_cases(3000, 7)
    o = run_oracle(oracle, d, 2, 366.0, tmp_path)
    np.testing.assert_allclose(run_torch(d, 2, 366.0, device), o,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_gpu_no_host_sync():
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    d = make_cases(2000, 8)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64,
                                  device='cuda')
    args = [t(d[k]) for k in ('rs', 'ns', 'rgl', 'tair', 'vpd', 'lai',
                              'gsm', 'b')]
    ppm = t(d['co2'])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        catm = C.catm_from_co2(ppm, 2)
        rc = C.calc_rc_co2(*args[:5], args[5], args[6], args[7],
                           catm[:, 0])
    finally:
        torch.cuda.set_sync_debug_mode('default')
    torch.cuda.synchronize()
    assert torch.isfinite(rc).all()
