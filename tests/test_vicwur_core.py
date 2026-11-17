# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR's vic_run hooks (torchvic.vicwur.core) and the vicwur3 core
(torchvic.core.version).

Oracle (tests/oracle/ref_vicwur_core.c, the unchanged VIC-WUR 3.0 vic_run):
  calc_root_fractions (VIC-WUR 3.0 `zone++`), transpiration with the layer
  Wcr, root, LAI and veg_lib of a hook, b_co2 with Catm, SHARE_LAYER_MOIST
  both ways (VIC-WUR 3.0's `spare_transp +=`) and the stored layer water
  stress; runoff() with a layer Ksat per step.
Identity: every hook at the value VIC 5 reads gives the run without hooks
  bit for bit (cores 5.1.0 and vicwur3); an override equals the same values
  written into the parameters.
Replay of the R1 runs (VIC-WUR 3.0 = <data root>/build/vicwur3, dumps of
<data root>/build/vicwur3_dump):
  G0a/G0b, plugins off: the vicwur3 core against VIC-WUR 3.0's outputs and its
  per-tile, per-layer dump (stored stress, transpiration, bare soil
  evaporation), with every hook on; G0c (G0a + G9's CO2 and b_co2): the
  b_co2 factor; G4d (paddy): the layer Ksat replayed from the dump with
  each day's post-plugin moisture; G8 (land use): the land-use kernel through
  the state and Cv hooks, bitwise against its own replay.
"""
import os
import struct
import subprocess

import numpy as np
import pytest
import torch

import _paths  # noqa: F401
from torchvic.core import column as COL
from torchvic.core import vegetation as V
from torchvic.core import version as core_version
from torchvic.vicwur import co2 as CO2
from torchvic.vicwur import core as VC

D = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'oracle', 'ref_vicwur_core.c')
W3 = _paths.env_path(('TORCHVIC_VICWUR_SRC', 'VICWUR_SRC'), 'build/vicwur3/src_tree/vic')
VER = _paths.env_path(('TORCHVIC_VICWUR_VERIFICATION', 'VICWUR_VERIFICATION'), 'verification')
G0 = os.path.join(VER, 'vicwur_19', 'gruns', 'R1')
G4D = os.path.join(VER, 'vicwur_12', 'gruns', 'G4d')
NBARE2 = os.path.join(VER, 'vicwur_19', 'nbare2')


@pytest.fixture(autouse=True)
def _cpu_and_core():
    old = core_version.CORE
    torch.set_default_device('cpu')
    yield
    core_version.set_core(old)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    if not os.path.exists(os.path.join(W3, 'vic_run', 'src',
                                       'canopy_evap.c')):
        pytest.skip('W3 source tree absent: ' + W3)
    exe = str(tmp_path_factory.mktemp('core') / 'ref_vicwur_core')
    s = os.path.join(W3, 'drivers', 'shared_all', 'src')
    vr = os.path.join(W3, 'vic_run', 'src')
    srcs = sorted(os.path.join(vr, f) for f in os.listdir(vr)
                  if f.endswith('.c'))
    srcs += [os.path.join(s, f) for f in (
        'calc_root_fraction.c', 'initialize_options.c',
        'initialize_global.c', 'initialize_parameters.c', 'vic_log.c',
        'open_file.c', 'forcing_utils.c')]
    r = subprocess.run(
        ['cc', '-o', exe, SRC] + srcs +
        ['-I' + os.path.join(W3, 'vic_run', 'include'),
         '-I' + os.path.join(W3, 'drivers', 'shared_all', 'include'),
         '-DLOG_LVL=25', '-DGIT_VERSION="ref-w3"', '-DUSERNAME="t"',
         '-DHOSTNAME="t"', '-O0', '-std=c99', '-ffp-contract=off',
         '-fcommon', '-ffunction-sections', '-Wl,--gc-sections', '-w',
         '-lm'], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('cannot build oracle: ' + r.stderr[-400:])
    return exe


def _call(exe, mode, head, rows, nout, tmp_path):
    fi, fo = str(tmp_path / (mode + '.in')), str(tmp_path / (mode + '.out'))
    with open(fi, 'wb') as f:
        f.write(np.asarray(head, '<f8').tobytes())
        f.write(np.ascontiguousarray(rows, '<f8').tobytes())
    subprocess.run([exe, mode, fi, fo], check=True)
    return np.fromfile(fo, '<f8').reshape(-1, nout)


def test_l1_root_fractions(oracle, tmp_path):
    """VIC-WUR 3.0 calc_root_fraction.c:58-183 (zone++ at :137) and
    VIC-WUR 3.0 vic_init.c:1259-1267's normalisation, bit for bit, over
    random and edge zones -- zones ending exactly on a layer boundary,
    reaching past the column, an empty last zone, fractions 1e-6 and 1e-4
    off 1."""
    g = np.random.default_rng(19)
    n = 4000
    lay = np.array([0.1, 0.252, 0.3, 0.5, 1.0, 1.5, 6.0])
    zon = np.array([0.05, 0.1, 0.15, 0.3, 0.6, 1.0, 1.3, 1.7, 2.3])
    dep = np.where(g.random((n, 3)) < .5, g.choice(lay, (n, 3)),
                   g.uniform(0.05, 3.0, (n, 3)))
    zd = np.where(g.random((n, 3)) < .5, g.choice(zon, (n, 3)),
                  g.uniform(0.02, 2.5, (n, 3)))
    # zones that end exactly on a layer bottom (the :113-135 branch)
    k = np.arange(0, n, 7)
    zd[k, 0] = dep[k, 0]
    zd[k[::2], 1] = dep[k[::2], 1]
    zf = g.dirichlet([1, 1, 1], n)
    zf[::11, 2] = 0.0
    zf[::11] /= zf[::11].sum(1, keepdims=True)
    zd[::13, 2] = 0.0
    zf[::13] = [0.6, 0.4, 0.0]
    zf[5::17] *= 1.0 + 1e-6            # within 1e-5: kept
    zf[6::17] *= 1.0 + 1e-4            # beyond: divided by the sum
    zd[0], zf[0], dep[0] = [0.1, 0.6, 2.3], [32 / 99., 44 / 99., 23 / 99.], \
        [0.252, 6.0, 6.0]
    r = V.root_fractions_w3(torch.as_tensor(zd), torch.as_tensor(zf),
                            torch.as_tensor(dep)).numpy()
    # the walks that leave no roots stop VIC-WUR 3.0
    # (calc_root_fraction.c:173-178): zeros here, exit status 1 there
    dead = ~(r.sum(1) > 0)
    assert dead.any() and not dead[0]
    rows = np.concatenate([zd, zf, dep], 1)
    with pytest.raises(subprocess.CalledProcessError):
        _call(oracle, 'root', [1, 3, 3], rows[dead][:1], 3, tmp_path)
    o = _call(oracle, 'root', [int((~dead).sum()), 3, 3], rows[~dead], 3,
              tmp_path)
    np.testing.assert_array_equal(r[~dead], o)
    # R1 class 2 (the main cause of the differences in the plugin-off G0
    # runs): VIC-WUR 3.0 drops the part of the 0.1-0.7 m zone below 0.252 m;
    # the overlap (5.0.1/5.1.0) keeps it
    np.testing.assert_allclose(o[0], [0.6676294615, 0.3323705385, 0.0],
                               atol=1e-9)
    ov = V.root_fractions(torch.as_tensor(zd[:1]), torch.as_tensor(zf[:1]),
                          torch.as_tensor(dep[:1])).numpy()[0]
    np.testing.assert_allclose(ov, [0.4358249158, 0.5641750842, 0.0],
                               atol=1e-9)


def _transp_cases(n, g, dt):
    d = {'dt': np.full(n, dt)}
    wmax = g.uniform(20, 400, (n, 3))
    wpwp = wmax * g.uniform(0.05, 0.3, (n, 3))
    wcr = wpwp + (wmax - wpwp) * g.uniform(0.2, 0.9, (n, 3))
    moist = np.where(g.random((n, 3)) < .15, wcr,       # exactly at Wcr
                     wmax * g.uniform(0.0, 1.0, (n, 3)))
    moist = np.where(g.random((n, 3)) < .08, wpwp * g.uniform(0.5, 1.0,
                                                             (n, 3)), moist)
    ice = np.where(g.random((n, 3)) < .1, moist * g.uniform(0, .9, (n, 3)),
                   0.0)
    root = g.dirichlet([1, 1, 1], n)
    root[::5, 0] = 0.0
    root[1::5, 2] = 0.0
    root[2::7, 1] = 0.0
    root[3::9] = [0.0, 0.0, 1.0]
    root[4::9] = [1.0, 0.0, 0.0]
    root /= root.sum(1, keepdims=True)
    d.update(moist=moist, ice=ice, wmax=wmax, wcr=wcr, wpwp=wpwp, root=root)
    d['lai'] = g.uniform(0.1, 8, n)
    d['rmin'] = np.where(g.random(n) < .05, 0.0, g.uniform(20, 300, n))
    d['rgl'] = g.uniform(30, 150, n)
    d['rarc'] = g.uniform(2, 60, n)
    d['b'] = np.where(g.random(n) < .3, 0.0, g.uniform(0, 0.6, n))
    d['catm'] = np.where(g.random(n) < .2, 366e-6,
                         g.uniform(280, 900, n) * 1e-6)
    d['rad'] = g.uniform(-100, 700, n)
    d['vpd'] = g.uniform(0, 5000, n)
    d['ns'] = g.uniform(0, 800, n)
    d['tair'] = g.uniform(-5, 45, n)
    d['ra'] = g.uniform(5, 200, n)
    d['dry'] = np.where(g.random(n) < .2, 1.0, g.uniform(0, 1, n))
    d['elev'] = g.uniform(0, 3000, n)
    return d


@pytest.mark.parametrize('dt', [86400.0, 3600.0])
@pytest.mark.parametrize('share', [True, False])
def test_l1_transpiration(oracle, tmp_path, share, dt):
    """VIC-WUR 3.0 canopy_evap.c transpiration(): a hook's layer Wcr
    (independent of the soil's), root, LAI, rmin/RGL/rarc and b_co2 with
    Catm; VIC-WUR 3.0's CASE 1 (`spare_transp +=`, :341-372) and CASE 2;
    the stored layer stress (:292-296, :403-405); the C's wilting clamp
    (:509-537) as VegMosaic.step applies it; veg_var->rc."""
    g = np.random.default_rng(1900 + share + int(dt))
    n = 8000
    d = _transp_cases(n, g, dt)
    rows = np.concatenate([d['moist'], d['ice'], d['wmax'], d['wcr'],
                           d['wpwp'], d['root']] +
                          [d[k][:, None] for k in (
                              'lai', 'rmin', 'rgl', 'rarc', 'b', 'catm',
                              'rad', 'vpd', 'ns', 'tair', 'ra', 'dry', 'dt',
                              'elev')], 1)
    o = _call(oracle, 'transp', [n, float(share)], rows, 7, tmp_path)
    t = lambda k: torch.as_tensor(d[k], dtype=D)
    am = t('moist') - t('ice')
    st = {}
    fac = CO2.co2factor(t('b'), CO2.ppm_from_catm(t('catm')))
    tr, rc = V.transpiration(
        am, t('wcr'), t('wpwp'), t('root'), t('lai'), t('rmin'), t('rgl'),
        t('ns'), t('tair'), t('vpd'), t('ra'), t('rarc'), t('rad'),
        t('elev'), share_layer_moist=share, return_rc=True,
        step=(dt, t('dry')), co2factor=fac, share_fix=True, stress=st)
    # VIC-WUR 3.0 canopy_evap.c:509-537, VegMosaic.step's wilting clamp
    cap = t('moist') - t('wpwp')
    ic = t('ice')
    cap = torch.where((ic > 0) & (ic >= t('wpwp')), t('moist') - ic, cap)
    tr = torch.clamp(torch.minimum(tr, cap), min=0.0)
    np.testing.assert_allclose(tr.numpy(), o[:, :3], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(st['gsm'].numpy(), o[:, 3:6], rtol=1e-13,
                               atol=0)
    # veg_var->rc (read by the carbon step only) differs from the C's in
    # two places that are the VIC 5 core's, unchanged here: an upper layer
    # without roots is counted out of CASE 2's conductance sum by the C
    # (its avail_moist is 0, :234-247), and a layer at rc 0 (rmin 0) adds
    # HUGE_RESIST to it (:451-456); the other rows are compared
    rooted = (d['root'][:, :2] > 0).all(1) & (d['rmin'] > 0)
    np.testing.assert_allclose(rc.numpy()[rooted], o[rooted, 6], rtol=1e-12,
                               atol=1e-12)
    if share:
        # VIC-WUR 3.0's fix moved something: with two or more dry rooted
        # layers VIC 5 hands on only the last one's shortfall
        tr5 = V.transpiration(
            am, t('wcr'), t('wpwp'), t('root'), t('lai'), t('rmin'),
            t('rgl'), t('ns'), t('tair'), t('vpd'), t('ra'), t('rarc'),
            t('rad'), t('elev'), share_layer_moist=True,
            step=(dt, t('dry')), co2factor=fac, share_fix=False)
        tr5 = torch.clamp(torch.minimum(tr5, cap), min=0.0)
        assert (tr5 - tr).abs().max() > 1e-3


def test_l1_column_layer_ksat(oracle, tmp_path):
    """VIC-WUR 3.0 runoff.c:156-157: the drainage takes layer[l].Ksat, here
    a new value every step (paddy pow(Ksat, 0.33) and random), against
    column.step with that Ksat; VIC-WUR 3.0's residual moisture as
    (resid * depth) * 1000 (:88-89)."""
    core_version.set_core('vicwur3')
    g = np.random.default_rng(191)
    for rs in (1, 4, 24):
        nsteps = 500
        maxm = np.array([60.0, 900.0, 1500.0])
        expt = np.array([12.0, 14.0, 16.0])
        ksat = np.array([2350.0, 1243.0, 1760.0])
        resid = np.array([0.0, 0.02, 0.01])
        depth = np.array([0.252, 3.0, 5.0])
        m0 = 0.5 * maxm
        head = np.concatenate([[nsteps, rs, 0.25, 0.02, 12.0, 0.8, 2.0],
                               maxm, expt, ksat, resid, depth, m0])
        ppt = np.where(g.random(nsteps) < .3, g.uniform(0, 60, nsteps), 0.0)
        ev = g.uniform(0, 1.5, (nsteps, 3)) * [1.0, 0.5, 0.2]
        lk = np.where(g.random((nsteps, 3)) < .5,
                      np.power(ksat, 0.33) * np.ones((nsteps, 3)),
                      g.uniform(1, 3000, (nsteps, 3)))
        rows = np.concatenate([ppt[:, None], ev, lk], 1)
        o = _call(oracle, 'column', head, rows, 5, tmp_path)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64)[None])
        st = t(m0)
        res = np.zeros((nsteps, 5))
        for i in range(nsteps):
            p = COL.SoilParams(
                b_infilt=t([0.25])[0], Ds=t([0.02])[0], Dsmax=t([12.0])[0],
                Ws=t([0.8])[0], c=t([2.0])[0], expt=t(expt), Ksat=t(lk[i]),
                max_moist=t(maxm), resid_moist=t((resid * depth) * 1000.0),
                runoff_steps_per_day=rs, model_steps_per_day=1)
            st, ro, bf = COL.step(st, t([ppt[i]])[0], t(ev[i]), p)
            res[i, :3] = st[0].numpy()
            res[i, 3], res[i, 4] = float(ro[0]), float(bf[0])
        np.testing.assert_allclose(res, o, rtol=1e-11, atol=1e-11)


def _gwm_series(g, nsteps, maxm, ksat):
    ppt = np.where(g.random(nsteps) < .35, g.uniform(0, 120, nsteps), 0.0)
    ev = g.uniform(-0.2, 1.5, (nsteps, 3)) * [1.0, 0.5, 0.2]
    lk = np.where(g.random((nsteps, 3)) < .5, ksat * np.ones((nsteps, 3)),
                  g.uniform(0.5, 20, (nsteps, 3)))
    return np.concatenate([ppt[:, None], ev, lk], 1)


def _gwm_torch(head, rows, mode):
    """column.step_gwm over the series of `_gwm_series`."""
    nsteps, rs = int(head[0]), int(head[1])
    maxm, expt = head[7:10], head[10:13]
    resid, depth, m0 = head[16:19], head[19:22], head[22:25]
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64)[None])
    st = t(m0)
    res = np.zeros((nsteps, 7))
    for i in range(nsteps):
        p = COL.SoilParams(
            b_infilt=t([head[2]])[0], Ds=t([head[3]])[0],
            Dsmax=t([head[4]])[0], Ws=t([head[5]])[0], c=t([head[6]])[0],
            expt=t(expt), Ksat=t(rows[i, 4:7]), max_moist=t(maxm),
            resid_moist=t((resid * depth) * 1000.0),
            runoff_steps_per_day=rs, model_steps_per_day=1)
        st, ro, bf, dg = COL.step(st, t([rows[i, 0]])[0], t(rows[i, 1:4]),
                                  p, diagnostics=True, gwm=mode)
        res[i, :3] = st[0].numpy()
        res[i, 3], res[i, 4] = float(ro[0]), float(bf[0])
        res[i, 5], res[i, 6] = float(dg['recharge'][0]), float(
            dg['gwrecharge'][0])
    return res


def test_l1_runoff_gwm(oracle, tmp_path):
    """VIC-WUR 3.0's runoff_GWM (runoff.c:445-773, GWM TRUE) against
    column.step(gwm='exact'): every layer drains, Q12 may go negative, the
    Ksat cut from the bottom layer up with its spill to the layer above and
    to runoff, OUT_RECHARGE and OUT_GWRECHARGE; small Ksat values make the
    cut happen on most steps."""
    core_version.set_core('vicwur3')
    g = np.random.default_rng(41)
    ncut = 0
    for rs in (1, 4, 24):
        nsteps = 600
        maxm = np.array([40.0, 120.0, 200.0])
        expt = np.array([12.0, 14.0, 16.0])
        ksat = np.array([350.0, 15.0, 8.0])
        resid = np.array([0.0, 0.02, 0.01])
        depth = np.array([0.1, 0.4, 0.7])
        m0 = np.array([0.95, 0.95, 0.95]) * maxm
        head = np.concatenate([[nsteps, rs, 0.25, 0.02, 12.0, 0.8, 2.0],
                               maxm, expt, ksat, resid, depth, m0])
        rows = _gwm_series(g, nsteps, maxm, ksat)
        o = _call(oracle, 'gwm', head, rows, 7, tmp_path)
        res = _gwm_torch(head, rows, 'exact')
        np.testing.assert_allclose(res, o, rtol=1e-11, atol=1e-11)
        assert (o[:, 4] == 0).all()                      # no baseflow
        # the Ksat cut was reached: mode fixed (which treats only the cut
        # differently) parts from the C
        ncut += int((np.abs(_gwm_torch(head, rows, 'fixed') - o)
                     > 1e-9).any(1).sum())
    assert ncut > 0


def test_runoff_gwm_fixed_conserves():
    """Mode fixed takes the interior cut back from the layer below:
    storage change = ppt - evap - runoff - GWRECHARGE, step by step; exact
    creates water there."""
    core_version.set_core('vicwur3')
    g = np.random.default_rng(7)
    maxm = np.array([40.0, 120.0, 200.0])
    ksat = np.array([350.0, 15.0, 8.0])
    head = np.concatenate([[400, 4, 0.25, 0.02, 12.0, 0.8, 2.0], maxm,
                           [12.0, 14.0, 16.0], ksat, [0.0, 0.02, 0.01],
                           [0.1, 0.4, 0.7], 0.95 * maxm])
    rows = _gwm_series(g, 400, maxm, ksat)
    err = {}
    for mode in ('exact', 'fixed'):
        r = _gwm_torch(head, rows, mode)
        s = np.concatenate([[head[22:25].sum()], r[:, :3].sum(1)])
        e = (rows[:, 0] - rows[:, 1:4].sum(1) - r[:, 3] - r[:, 6]
             - np.diff(s))
        err[mode] = np.abs(e).max()
    assert err['fixed'] < 1e-9
    assert err['exact'] > 1e-3


# ---------------------------------------------------------------------------
# R1 runs: helpers
# ---------------------------------------------------------------------------
def _need(path):
    if not os.path.exists(path):
        pytest.skip('VIC-WUR reference run absent: ' + path)
    pytest.importorskip('xarray')


def read_dump(path, tags=(2, 3, 6)):
    out = {k: [] for k in tags}
    dims = None
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            tag, n = struct.unpack('<iq', h)
            if tag == 0:
                dims = np.frombuffer(f.read(8 * n), '<f8').astype(int)
            elif tag in out:
                out[tag].append(np.frombuffer(f.read(8 * n), '<f8').copy())
            else:
                f.seek(8 * n, 1)
    return dims, {k: np.stack(v) for k, v in out.items() if v}


def _case(run_dir, tmp_path, core='vicwur3', steps=0):
    from torchvic import case
    gf = VC.image_global(os.path.join(run_dir, 'global_W3.txt'),
                         str(tmp_path / 'global.txt'))
    return case.load(gf, device=torch.device('cpu'), core=core, steps=steps)


class _Tiles(object):
    """The dump's compact tiles (VIC-WUR 3.0 vic_init.c:1145-1183: the
    covered classes in class order) against the case's columns (pseudo cells
    of a snow-band run) and class-indexed tiles."""

    def __init__(self, c):
        ex = c.snowbands
        ncol = c.mosaic.cv.shape[0]
        self.pc = (np.arange(ncol) if ex is None
                   else np.asarray(ex['cell']))
        self.pb = (np.zeros(ncol, int) if ex is None
                   else np.asarray(ex['band']))
        cv = c.mosaic.cv.cpu().numpy()
        P, CL, I, K, B = [], [], [], [], []
        for p in range(ncol):
            for k, cls in enumerate(np.where(cv[p] > 0)[0]):
                P.append(p), CL.append(cls), I.append(self.pc[p])
                K.append(k), B.append(self.pb[p])
        self.P, self.CL, self.I, self.K, self.B = map(np.asarray,
                                                      (P, CL, I, K, B))


def _cmp_outputs(run_dir, out, c):
    import xarray as xr
    import glob
    fs = sorted(glob.glob(os.path.join(run_dir, 'W3', 'out', 'fluxes.*.nc')))
    a = xr.concat([xr.open_dataset(f).load() for f in fs], dim='time')
    m = np.isfinite(a['OUT_EVAP'].values[0])
    worst = {}
    for vk, tk in (('OUT_EVAP', 'et'), ('OUT_RUNOFF', 'runoff'),
                   ('OUT_BASEFLOW', 'baseflow'),
                   ('OUT_TRANSP_VEG', 'transp'),
                   ('OUT_EVAP_CANOP', 'evap_canop'),
                   ('OUT_EVAP_BARE', 'evap_bare')):
        x = a[vk].values[:, m]
        y = out[tk].detach().numpy().reshape(x.shape)
        worst[vk] = float(np.abs(x - y).max())
    return worst


# ---------------------------------------------------------------------------
# identity: hooks at VIC 5's values, overrides as parameters
# ---------------------------------------------------------------------------
def _identity_hooks(c):
    m = c.mosaic
    nc, nt = m.cv.shape
    months = [d.month - 1 for d in c.dates]
    lai0, fc0, alb0 = m.lai.clone(), m.fcanopy.clone(), m.albedo.clone()
    rf0 = m.root_frac.clone()
    sp = c.soil
    return VC.VicwurCore(
        ksat=sp.Ksat.unsqueeze(1).expand(nc, nt, 3).clone(),
        wcr=sp.w_crit_mm.unsqueeze(1).expand(nc, nt, 3).clone(),
        veg={'lai': lambda t: lai0[:, :, months[t]],
             'fcanopy': lambda t: fc0[:, :, months[t]],
             'albedo': lambda t: alb0[:, :, months[t]],
             'rmin': m.rmin.clone(), 'rgl': m.rgl.clone(),
             'rarc': m.rarc.clone(), 'root_frac': rf0},
        b_co2=torch.zeros(nc, nt, dtype=D), water_stress=True,
        diagnostics=True)


def _same(o0, o1):
    bad = [k for k, v in o0.items() if torch.is_tensor(v) and k in o1
           and not torch.equal(v, o1[k])]
    return bad


@pytest.mark.parametrize('core', ['5.1.0', 'vicwur3'])
def test_hooks_at_vic5_values_are_bitwise(tmp_path, core):
    """Every hook given the value VIC 5 reads (layer Ksat and Wcr of the
    soil, the file's vegetation, b_co2 0) runs the model without hooks bit
    for bit, under the 5.1.0 core and the vicwur3 core."""
    _need(os.path.join(G0, 'G0a', 'global_W3.txt'))
    c = _case(os.path.join(G0, 'G0a'), tmp_path, core=core, steps=400)
    with torch.no_grad():
        o0 = c.run(p=c.params())
        o1 = c.run(p=c.params(), vicwur_core=_identity_hooks(c))
    assert not _same(o0, o1)
    assert 'vicwur' in o1 and o1['vicwur']['water_stress'].shape[0] == 400
    # the mosaic is the file's again after the run
    o2 = c.run(p=c.params())
    assert not _same(o0, o2)


def test_overrides_equal_the_same_parameters(tmp_path):
    """A hook's value is the value VIC then reads: a layer Ksat and Wcr, a
    LAI, rmin and root on half the tiles, equal a case whose parameters
    carry them (VIC-WUR 3.0 update_step_vars.c:56-64 copies veg_hist into
    veg_var; VIC-WUR 3.0 vic_init.c:1630-1642 starts layer Ksat/Wcr and
    veg_var root from the soil and vegetation files)."""
    _need(os.path.join(G0, 'G0a', 'global_W3.txt'))
    core_version.set_core('vicwur3')
    c = _case(os.path.join(G0, 'G0a'), tmp_path, steps=300)
    m, sp = c.mosaic, c.soil
    nc, nt = m.cv.shape
    mask = torch.zeros(nc, nt, dtype=torch.bool)
    mask[:, ::2] = True
    lai_new = torch.full((nc, nt), 1.7, dtype=D)
    rmin_new = m.rmin * 0.6
    root_new = torch.tensor([0.5, 0.3, 0.2], dtype=D).expand(nc, nt, 3)
    hooks = VC.VicwurCore(
        ksat=(sp.Ksat * 0.3).unsqueeze(1).expand(nc, nt, 3),
        wcr=(sp.w_crit_mm * 0.8).unsqueeze(1).expand(nc, nt, 3),
        veg={'lai': lai_new, 'rmin': rmin_new, 'root_frac': root_new},
        veg_mask=mask)
    with torch.no_grad():
        o1 = c.run(p=c.params(), vicwur_core=hooks)
        m.lai = torch.where(mask.unsqueeze(-1), lai_new.unsqueeze(-1), m.lai)
        m.rmin = torch.where(mask, rmin_new, m.rmin)
        m.root_frac = torch.where(mask.unsqueeze(-1), root_new, m.root_frac)
        sp.Ksat = sp.Ksat * 0.3
        sp.w_crit_mm = sp.w_crit_mm * 0.8
        o2 = c.run(p=c.params())
    bad = _same(o2, o1)
    # rootmoist (OUT_ROOTMOIST) is formed from the case's soil root at load
    assert not [k for k in bad if k != 'rootmoist'], bad
    # and it moved something
    o0 = _case(os.path.join(G0, 'G0a'), tmp_path, steps=300).run()
    assert (o0['transp'] - o1['transp']).abs().max() > 1e-3


def test_refused_where_not_wired(tmp_path):
    _need(os.path.join(G0, 'G0a', 'global_W3.txt'))
    c = _case(os.path.join(G0, 'G0a'), tmp_path, steps=5)
    with pytest.raises(NotImplementedError, match='FULL_ENERGY'):
        c.run(vicwur_core=VC.VicwurCore(water_stress=True),
              full_energy=True)
    with pytest.raises(ValueError, match='unknown field'):
        VC.VicwurCore(veg={'leaf': 1.0})


def test_nbare_needs_the_vicwur3_core():
    from torchvic.io import build
    core_version.set_core('5.1.0')
    with pytest.raises(ValueError, match='vicwur3'):
        build.build_mosaic(None, nbare=2)


# ---------------------------------------------------------------------------
# R1 runs: G0a / G0b, plugins off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('run', ['G0a', 'G0b'])
def test_l2_g0_against_w3(tmp_path, run):
    """VIC-WUR 3.0's G0a/G0b (SHARE_LAYER_MOIST TRUE/FALSE) through the vicwur3
    core with every hook on: the daily outputs within float32 output
    precision (TorchVIC's 5.1.0 core is 2.4e-7 mm off VIC 5.1.0 on G0b),
    and the dump's per-tile, per-layer stored water stress, transpiration
    and bare-soil evaporation over all 2557 days and 170 tile columns."""
    rd = os.path.join(G0, run)
    _need(os.path.join(rd, 'dump.bin'))
    c = _case(rd, tmp_path)
    with torch.no_grad():
        o = c.run(p=c.params(), vicwur_core=_identity_hooks(c))
    worst = _cmp_outputs(rd, o, c)
    print(run, worst)
    assert max(worst.values()) < 1e-6, worst
    dims, rec = read_dump(os.path.join(rd, 'dump.bin'), tags=(2,))
    N, NT, NB, NL, NF = dims[:5]
    nsv = dims[13]
    T = rec[2].shape[0]
    soil = rec[2].reshape(T, N, NT, NB, NL, nsv)
    tl = _Tiles(c)
    sel = soil[:, tl.I, tl.K, tl.B]                     # [T, pair, L, nsv]
    vw = o['vicwur']
    ws = vw['water_stress_layer'].numpy()[:, tl.P, tl.CL]
    tr = vw['transp_tile'].numpy()[:, tl.P, tl.CL]
    es = vw['esoil_tile'].numpy()[:, tl.P, tl.CL]
    d_ws = np.abs(sel[..., NF + 2] - ws).max()
    d_tr = np.abs(sel[..., NF + 5] - tr).max()
    d_es = np.abs(sel[:, :, 0, NF + 3] - es).max()
    print(run, 'pairs', len(tl.P), 'ws %.2e transp %.2e esoil %.2e'
          % (d_ws, d_tr, d_es))
    assert d_ws < 1e-7 and d_tr < 1e-6 and d_es < 1e-6
    # and the stress was stored: a layer below Wcr somewhere
    assert (sel[..., NF + 2] < 0.999).mean() > 0.05


G11 = _paths.env_path(('TORCHVIC_VICWUR_4_GRUNS', 'VICWUR_4_GRUNS'), 'verification/vicwur_4/gruns/R1') + '/G11'


def test_l2_g11_gwm_against_w3(tmp_path):
    """VIC-WUR 3.0's G11 (GWM TRUE, VIC only; tools/vicwur/gw_runs.py, the dump
    build, OUT_GWRECHARGE / OUT_RECHARGE / OUT_RUNOFF written as double)
    through the vicwur3 core with gwm 'exact': the daily outputs within
    float32 output precision, OUT_GWRECHARGE and OUT_RECHARGE within 1e-9
    mm, OUT_BASEFLOW 0, the layer moisture, and the water balance with
    OUT_GWRECHARGE in its outflow closed, over 2557 days."""
    import glob
    import xarray as xr
    _need(os.path.join(G11, 'global_W3.txt'))
    c = _case(G11, tmp_path)
    h = _identity_hooks(c)
    h.gwm = 'exact'
    with torch.no_grad():
        o = c.run(p=c.params(), vicwur_core=h)
    worst = _cmp_outputs(G11, o, c)
    assert max(worst.values()) < 1e-6, worst
    fs = sorted(glob.glob(os.path.join(G11, 'W3', 'out', 'fluxes.*.nc')))
    a = xr.concat([xr.open_dataset(f).load() for f in fs], dim='time')
    T = a['OUT_GWRECHARGE'].shape[0]
    for vk, tk, tol in (('OUT_GWRECHARGE', 'gwrecharge', 1e-9),
                        ('OUT_RECHARGE', 'recharge', 1e-9)):
        x = a[vk].values.reshape(T, -1)
        d = np.abs(x - o[tk].numpy().reshape(T, -1)).max()
        assert d < tol, (vk, d)
    sm = a['OUT_SOIL_MOIST'].values.reshape(T, 3, -1)
    d = np.abs(sm - o['soil_liq'].numpy().transpose(0, 2, 1)).max()
    assert d < 1e-3, d                              # float32 of ~2000 mm
    assert (o['baseflow'] == 0).all()
    assert o['water_error'].abs().max() < 1e-9
    assert (o['gwrecharge'] > 0).any()


# ---------------------------------------------------------------------------
# Replays: the day's post-plugin moisture from the dump
# ---------------------------------------------------------------------------
def _replay(c, dump, tl, hooks, applied=None, steps=None):
    """Run the case with each day's start moisture set to the dump's
    (post-plugin) moisture of the day before (simulate's day_hook): the
    plugins' edits replayed, the VIC core TorchVIC's. Returns the outputs
    and the end-of-step per-tile moisture [T, pair, L] (layer 0 less what
    irrigation applied that step, `applied` [T, pair])."""
    soil, NF = dump
    T = soil.shape[0] if steps is None else steps
    moist_d = soil[:, tl.I, tl.K, tl.B][..., 0]            # [T, pair, L]
    ends = np.zeros((T,) + moist_d.shape[1:])
    P, CL = torch.as_tensor(tl.P), torch.as_tensor(tl.CL)

    def hook(t, view):
        m = view.moist
        ends[t - 1] = m[P, CL].numpy()
        new = m.clone()
        new[P, CL] = torch.as_tensor(moist_d[t - 1])
        cvw = c.mosaic.cv.unsqueeze(-1)
        return new, ((new - m) * cvw).sum((1, 2))

    with torch.no_grad():
        o = c.run(p=c.params(), vicwur_core=hooks, day_hook=hook)
    if applied is not None:
        ends[..., 0] += applied[:T]
    return o, ends[:T - 1], moist_d[:T - 1]


def _tile_ksat(soil, NF, tl, c):
    """layer[l].Ksat of each tile column per step: the dump of step t - 1
    (VIC-WUR 3.0 irr_run.c:118,153-156; irr_update_step_vars, :56-68,
    writes the same before vic_run), the soil's where VIC has no tile."""
    nc, nt = c.mosaic.cv.shape
    base = c.soil.Ksat.unsqueeze(1).expand(nc, nt, 3).clone()
    kd = soil[:, tl.I, tl.K, tl.B][..., NF + 1]            # [T, pair, L]
    P, CL = torch.as_tensor(tl.P), torch.as_tensor(tl.CL)

    def at(t):
        k = base.clone()
        k[P, CL] = torch.as_tensor(kd[max(t - 1, 0)])
        return k
    return at


def test_l2_g4d_paddy_ksat(tmp_path):
    """G4d (paddy irrigation): vic_run reads layer-0 Ksat = pow(Ksat, 0.33)
    on the paddy tiles (VIC-WUR 3.0 irr_run.c:56-68). Replayed with the
    dump's post-plugin moisture each day and the dump's layer Ksat as the
    hook: the end-of-step layer 0 (before the application) and layer 1 of
    every tile within 1e-5 mm of VIC-WUR 3.0, of the paddy tiles within 1e-6
    (one day from the dump's state each time; the bottom layer also gives up
    the local groundwater pumping, which this does not replay), the
    transpiration within 1e-6. Without the hook the paddy tiles drain as
    VIC 5 would."""
    _need(os.path.join(G4D, 'dump.bin'))
    from torchvic.vicwur import params as P
    c = _case(G4D, tmp_path)
    dims, rec = read_dump(os.path.join(G4D, 'dump.bin'))
    N, NT, NB, NL, NF = dims[:5]
    nsv, NJ = dims[13], dims[6]
    T = rec[2].shape[0]
    soil = rec[2].reshape(T, N, NT, NB, NL, nsv)
    iv = rec[6].reshape(T, N, NJ, NB, 8)
    tl = _Tiles(c)
    inp = P.load_vicwur_inputs(os.path.join(G4D, 'global_W3.txt'))
    ir = inp.irrigation
    act, vix = ir.active.numpy(), ir.veg_index.numpy()
    applied = np.zeros((T, len(tl.P)))
    paddy_pair = np.zeros(len(tl.P), bool)
    for q in range(len(tl.P)):
        i, k, b = tl.I[q], tl.K[q], tl.B[q]
        j = 0
        for kk in range(act.shape[1]):
            if act[i, kk]:
                if vix[i, kk] == k:
                    applied[:, q] = np.nan_to_num(iv[:, i, j, b, 2])
                    paddy_pair[q] = bool(ir.paddy[kk])
                j += 1
    assert paddy_pair.any()
    hooks = VC.VicwurCore(ksat=_tile_ksat(soil, NF, tl, c), diagnostics=True)
    o, ends, md = _replay(c, (soil, NF), tl, hooks, applied=applied)
    post = soil[:, tl.I, tl.K, tl.B][..., 0]                 # dump of step t
    d01 = np.abs(ends[:, :, :2] - post[:T - 1, :, :2]).max()
    tr = o['vicwur']['transp_tile'].numpy()[:, tl.P, tl.CL]
    d_tr = np.abs(soil[:, tl.I, tl.K, tl.B][..., NF + 5] - tr).max()
    dpad = np.abs(ends[:, paddy_pair, :2]
                  - post[:T - 1][:, paddy_pair, :2]).max()
    print('G4d replay: layers 0-1 %.2e mm (paddy tiles %.2e), transp %.2e'
          % (d01, dpad, d_tr))
    assert d01 < 1e-5 and dpad < 1e-6 and d_tr < 1e-6
    # VIC 5's Ksat on the paddy tiles: the drainage is far off
    c2 = _case(G4D, tmp_path, steps=200)
    o2, e2, _ = _replay(c2, (soil, NF), tl, VC.VicwurCore(diagnostics=True),
                        applied=applied, steps=200)
    dp = np.abs(e2[:, paddy_pair, :2] - post[:199][:, paddy_pair, :2]).max()
    assert dp > 1.0, dp


def _co2_hooks(c, tl, gpath, **kw):
    """b_co2 of each tile's class (VIC-WUR 3.0 vic_init.c:185-199, BCO2_SRC
    FROM_VEGPARAM) and the yearly CO2 forcing as Catm per column
    (VIC-WUR 3.0 co2_update_step_vars.c:14-17), torchvic.vicwur.co2."""
    from torchvic.vicwur import forcing as F
    from torchvic.vicwur import params as P
    inp = P.load_vicwur_inputs(gpath)
    g = P.parse_global(gpath)
    nc, nt = c.mosaic.cv.shape
    b = torch.as_tensor(inp.b_co2.cpu().numpy()[:nt][:, tl.pc].T.copy(),
                        device='cpu')
    d0 = c.dates[0]
    prefix = [w for w in g['PLUGIN_FORCE_TYPE'] if w[0] == 'CO2'][0][3]
    ms = F.file_info('%s%4d.nc' % (prefix, d0.year), 'co2')['start']
    f = CO2.Co2Forcing.from_global(g, (d0.year, d0.month, d0.day, 0),
                                   len(c.dates), ms)
    try:
        catm = torch.stack([f.catm(t)[:, 0] for t in range(len(c.dates))])
    finally:
        f.close()
    catm = catm[:, torch.as_tensor(tl.pc, device=catm.device)]  # [T, col]
    dev = c.mosaic.cv.device
    return VC.VicwurCore(b_co2=b.to(dev), catm=catm.to(dev), **kw)


def test_l2_g0c_co2_factor(tmp_path):
    """G0c = G0a + G9's CO2 lines (PLUGIN_FORCE_TYPE CO2, BCO2_SRC
    FROM_VEGPARAM), a VIC-WUR 3.0 run made for this test: the b_co2 factor
    in calc_rc (VIC-WUR 3.0 penman.c:83-87) with the yearly CO2 is the only
    plugin. (In G9 itself WOFOST and LANDUSE rewrite the vegetation and move
    the canopy store between tiles, which the dump does not carry.) The
    vicwur3 core with every hook on (the factor; layer Ksat, Wcr and the
    vegetation at the files' values; the stored stress) against VIC-WUR 3.0's
    outputs and dump, as G0a; without the factor
    TorchVIC is G0a's answer, off VIC-WUR 3.0's G0c."""
    rd = os.path.join(G0, 'G0c')
    _need(os.path.join(rd, 'dump.bin'))
    c = _case(rd, tmp_path)
    tl = _Tiles(c)
    base = _identity_hooks(c)
    hooks = _co2_hooks(c, tl, os.path.join(rd, 'global_W3.txt'),
                       ksat=base.ksat, wcr=base.wcr, veg=base.veg,
                       water_stress=True, diagnostics=True)
    assert float(hooks.b_co2.abs().max()) > 0
    assert float(hooks.catm.max()) > 380e-6
    with torch.no_grad():
        o = c.run(p=c.params(), vicwur_core=hooks)
        o0 = c.run(p=c.params())
    worst = _cmp_outputs(rd, o, c)
    print('G0c', worst)
    assert max(worst.values()) < 1e-6, worst
    off = _cmp_outputs(rd, o0, c)
    assert off['OUT_TRANSP_VEG'] > 1e-2, off
    dims, rec = read_dump(os.path.join(rd, 'dump.bin'), tags=(2,))
    N, NT, NB, NL, NF = dims[:5]
    T = rec[2].shape[0]
    sel = rec[2].reshape(T, N, NT, NB, NL, dims[13])[:, tl.I, tl.K, tl.B]
    vw = o['vicwur']
    d_ws = np.abs(sel[..., NF + 2]
                  - vw['water_stress_layer'].numpy()[:, tl.P, tl.CL]).max()
    d_tr = np.abs(sel[..., NF + 5]
                  - vw['transp_tile'].numpy()[:, tl.P, tl.CL]).max()
    print('G0c ws %.2e transp %.2e' % (d_ws, d_tr))
    assert d_ws < 1e-7 and d_tr < 1e-6


def test_l2_nbare2(tmp_path):
    """NBARE (VIC-WUR 3.0 get_global_param.c:76-77): R1's parameters with
    Nveg one lower in every cell and NBARE 2, a VIC-WUR 3.0 run made for this
    test. VIC-WUR 3.0 makes Nveg + NBARE tiles (vic_alloc.c:128-129) and runs
    every tile at an index >= Nveg as bare soil (vic_run.c:135,250;
    calc_surf_energy_bal.c:172-182): here the last two covered classes, so a
    vegetated class (14) loses its canopy. The vicwur3 core reads NBARE and
    marks them; against VIC-WUR 3.0's outputs and dump as G0a. (NBARE 1 is
    VIC 5's iveg == Nveg, the G0 runs.)"""
    _need(os.path.join(NBARE2, 'dump.bin'))
    c = _case(NBARE2, tmp_path)
    bt = c.mosaic.bare_tile
    assert bt is not None and int(bt.sum(1).max()) == 2
    with torch.no_grad():
        o = c.run(p=c.params(), vicwur_core=_identity_hooks(c))
    worst = _cmp_outputs(NBARE2, o, c)
    print('NBARE 2', worst)
    assert max(worst.values()) < 1e-6, worst
    # it is not G0a: class 14 runs bare
    g0a = _cmp_outputs(os.path.join(G0, 'G0a'), o, c)
    assert g0a['OUT_TRANSP_VEG'] > 1e-2, g0a
    dims, rec = read_dump(os.path.join(NBARE2, 'dump.bin'), tags=(2,))
    N, NT, NB, NL, NF = dims[:5]
    T = rec[2].shape[0]
    tl = _Tiles(c)
    sel = rec[2].reshape(T, N, NT, NB, NL, dims[13])[:, tl.I, tl.K, tl.B]
    vw = o['vicwur']
    d_ws = np.abs(sel[..., NF + 2]
                  - vw['water_stress_layer'].numpy()[:, tl.P, tl.CL]).max()
    d_es = np.abs(sel[:, :, 0, NF + 3]
                  - vw['esoil_tile'].numpy()[:, tl.P, tl.CL]).max()
    print('NBARE 2 ws %.2e esoil %.2e' % (d_ws, d_es))
    assert d_ws < 1e-7 and d_es < 1e-6


@pytest.mark.gpu
def test_cuda_hooks_agree_with_cpu(tmp_path):
    """Every hook on, on a CUDA device (the covered tiles and the CUDA
    graphs; test_vicwur_core_lanes.py: bit for bit the padded path's),
    against the same run on the CPU: G0c's first 120 days, to 1e-6 (the
    device's libm; measured 3.4e-9 relative)."""
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    rd = os.path.join(G0, 'G0c')
    _need(os.path.join(rd, 'global_W3.txt'))
    from torchvic import case
    gf = VC.image_global(os.path.join(rd, 'global_W3.txt'),
                         str(tmp_path / 'global.txt'))
    res = {}
    for dev in ('cpu', 'cuda'):
        torch.set_default_device(dev)
        c = case.load(gf, device=torch.device(dev), core='vicwur3',
                      steps=120)
        tl = _Tiles(c)
        base = _identity_hooks(c)
        h = _co2_hooks(c, tl, os.path.join(rd, 'global_W3.txt'),
                       ksat=base.ksat, wcr=base.wcr, veg=base.veg,
                       water_stress=True, diagnostics=True)
        with torch.no_grad():
            o = c.run(p=c.params(), vicwur_core=h)
        res[dev] = {k: o[k].detach().cpu() for k in ('et', 'transp',
                                                       'runoff')}
        # the stress of the covered tiles (the GPU's lanes carry no
        # uncovered tile: zero there)
        cov = (c.mosaic.cv > 0).cpu()
        res[dev]['ws'] = o['vicwur']['water_stress'].cpu()[:, cov]
    torch.set_default_device('cpu')
    for k in res['cpu']:
        np.testing.assert_allclose(res['cuda'][k].numpy(),
                                   res['cpu'][k].numpy(), rtol=1e-6,
                                   atol=1e-9, err_msg=k)


def test_landuse_through_the_hooks(tmp_path):
    """The land-use kernel plugged into a TorchVIC run through the two hooks
    it needs (state_fn: the carried tile states; cv: the tile Cv from a step
    on), on R1 G8 (the vicwur3 core, the C's 84 monthly lu_apply calls, at
    the start of the step as lu_force runs, VIC-WUR 3.0 lu_force.c:94). At
    each call the dump's pre-apply soil moisture and canopy store are
    written into the run and read back bit for bit; the kernel then runs on
    the states read from the run (the rest -- nodes, carbon, snow, leftover,
    ring -- from the dump, as test_vicwur_landuse_l2) and matches the dump's
    post-apply states with the gate of that test; its moisture, canopy store
    and Cv go back into the run, which ends on the last Cv."""
    import test_vicwur_landuse_l2 as LL
    from torchvic.vicwur import forcing as F
    from torchvic.vicwur import landuse as LU
    from torchvic.vicwur import params as PP
    import datetime
    rd = os.path.join(LL.V, 'G8')
    _need(os.path.join(rd, 'ludump.bin'))
    R = LL.read_ludump(os.path.join(rd, 'ludump.bin'))
    dims = R[30][0]
    nc, nt, nb, nl, nn, nf, ni, Rl, rps, uh, K, pbare = \
        (int(x) for x in dims[:12])
    irrigation, routing, full = (bool(x) for x in dims[12:15])
    S = LL.statics(dims, R[31][0])
    act = np.arange(nt)[None, :] < S['nva'][:, None]
    vc = np.where(act, np.nan_to_num(S['vc'], nan=-1), -1).astype(np.int64)
    tile_class = torch.as_tensor(np.where(vc < K, vc, -1))
    g = PP.parse_global(os.path.join(rd, 'global_W3.txt'))
    fcv = [x for x in g['PLUGIN_FORCE_TYPE'] if x[0] == 'CV'][0]
    nsteps = (LL.END - datetime.date(*LL.START[:3])).days + 1
    ms = F.file_info(fcv[3] + '%4d.nc' % LL.START[0], fcv[1])['start']
    fr = F.PluginForcing(fcv[3], fcv[1], 'MONTH', LL.START, nsteps, 1,
                         met_start=ms)
    try:
        calls = [int(r[0]) for r in R[32]]
        forced = {s: fr.get(s).T.contiguous() for s in calls}
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
    soil = {k: LL.f64(S[k]) for k in (
        'zsum_node', 'max_moist_node', 'expt_node', 'bubble_node', 'depth',
        'soil_dens_min', 'bulk_dens_min', 'quartz', 'soil_density',
        'bulk_density', 'organic', 'frost_fract')}
    soil['fs_active'] = torch.as_tensor(S['fs'] != 0)
    cs_fn = LU.lu_heat_capacity_fn(soil, 'exact')
    actt = torch.as_tensor(act)

    c = _case(rd, tmp_path)
    ex = c.snowbands
    pc, pb = np.asarray(ex['cell']), np.asarray(ex['band'])
    # (column, class) <-> (cell, dump tile, band)
    Pp, CLc, Ic, Tt, Bb = [], [], [], [], []
    for p in range(len(pc)):
        for t in range(int(S['nva'][pc[p]])):
            Pp.append(p), CLc.append(vc[pc[p], t]), Ic.append(pc[p])
            Tt.append(t), Bb.append(pb[p])
    Pp, CLc, Ic, Tt, Bb = (torch.as_tensor(np.asarray(a)) for a in
                           (Pp, CLc, Ic, Tt, Bb))
    hooks = VC.VicwurCore(diagnostics=True)
    seen = {'calls': 0, 'worst': 0.0}
    at = {s: k for k, s in enumerate(calls)}

    def state_fn(t, states):
        if t not in at:
            return
        k = at[t]
        pre, post = LL.state(dims, R[33][k]), LL.state(dims, R[34][k])
        assert {'moist', 'wdew'} <= set(states.names())
        if k == 0:
            # the water-balance path carries no node temperatures
            with pytest.raises(KeyError):
                states.set('t_node', 0.0)
            with pytest.raises(ValueError):
                states.set('wdew', torch.zeros(3))
        # write the C's state before the call into the run, read it back
        for n in ('moist', 'wdew'):
            x = states.get(n).clone()
            x[Pp, CLc] = LL.f64(np.nan_to_num(pre[n]))[Ic, Tt, Bb]
            states.set(n, x)
        st = {n: LL.f64(np.nan_to_num(pre[n])) for n in LL.TB + (
            'moist', 'ice', 't_node', 'leftover', 'ring')}
        st['melting'] = st['melting'] != 0
        for n in ('moist', 'wdew'):
            back = states.get(n)[Pp, CLc]
            assert torch.equal(back, st[n][Ic, Tt, Bb])
            st[n][Ic, Tt, Bb] = back
        cvf, fatal_map = LU.map_cv_forcing(LL.f64(forced[t]), tile_class)
        cvo = LL.f64(np.nan_to_num(pre['cv']))
        cvn, dcv, csum, fatal, warn = LU.cv_update(cvf, cvo, actt)
        air = LL.f64(R[32][k][5:5 + nc])
        P = dict(area_fract=LL.f64(S['af']),
                 max_moist=LL.f64(S['max_moist']),
                 overstory=torch.as_tensor(np.nan_to_num(S['over']) != 0),
                 active=actt, bare=torch.as_tensor(act & (vc == K)),
                 paddy=paddy & actt, pbare=pbt, lai=st.pop('lai'),
                 dz_node=LL.f64(S['dz_node']), max_surface_swe=dims[15],
                 doy=int(R[32][k][4]), lat=LL.f64(S['lat']),
                 new_density=LU.band_new_density(air, LL.f64(S['tfac']),
                                                 dens_max=dims[18]),
                 paddy_order=order, irr_tile=vit, area=LL.f64(S['area']),
                 dt=dims[17], rps=rps, uh_length=uh)
        st1, terms = LU.apply(st, cvo, cvn, P, mode='exact',
                              paddy_exchange=irrigation and pbare != K,
                              route_leftover=irrigation and routing,
                              cs_fn=cs_fn, full_energy=full)
        assert not bool((fatal | terms['fatal']).any())
        m = act[:, :, None]
        for n in LL.TB[:-1] + ('moist', 'ice', 't_node'):
            a = st1[n].double().numpy()
            mm = np.broadcast_to(m.reshape(m.shape + (1,) * (a.ndim - 3)),
                                 a.shape)
            seen['worst'] = max(seen['worst'], LL.gate(
                a[mm], post[n][mm], '%s call %d' % (n, k)))
        # the kernel's states and Cv back into the run
        for n in ('moist', 'wdew'):
            x = states.get(n).clone()
            x[Pp, CLc] = st1[n][Ic, Tt, Bb]
            states.set(n, x)
        cv = c.mosaic.cv.clone() if hooks.cv is None else hooks.cv.clone()
        cv[Pp, CLc] = cvn[Ic, Tt]
        hooks.cv = cv
        seen['calls'] += 1

    hooks.state_fn = state_fn
    with torch.no_grad():
        o = c.run(p=c.params(), vicwur_core=hooks)
    print('G8 through the hooks', seen)
    assert seen['calls'] == 84
    assert torch.equal(o['vicwur']['cv_end'], hooks.cv)
    # the case keeps its file's Cv, and the Cv mattered
    assert not torch.equal(c.mosaic.cv, hooks.cv)
    with torch.no_grad():
        o0 = c.run(p=c.params())
    assert (o0['et'] - o['et']).abs().max() > 1e-3
