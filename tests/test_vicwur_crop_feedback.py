# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The crop feedback into VIC and irrigation.

C paths are VIC-WUR 3.0 (<data root>/build/vicwur3/src_tree/vic/plugins):
crops/src/crop_update_step_vars.c, wofost/src/Sweaf.c,
crops/src/crop_adjust_requirement.c, irrigation/src/irr_use.c (irr_wofost),
crops/src/crop_output.c (OUT_CROP_EVAP*).

Without data: the kernels of torchvic.vicwur.crop_feedback against scalar
transcriptions of the C (loop by loop, C floats as numpy float32) on random
states that reach every branch and quirk (the stale veg_var of the root
normalisation, the running root[0], the band scaling of Wcr, two crop
classes on one tile, inactive units, bands without area).

On R1 (the VIC-WUR 3.0 runs of tools/vicwur/gruns.py; the crop-dump build
<data root>/build/vicwur3_cropdump, tools/vicwur/cropdump_runs.py), through the
coupling runtime with VIC (the vicwur3 core) in the loop and all eleven
modules: G9, G9b and G10 over the 7 years against the VIC-WUR 3.0 history
files (every VIC, routing, irrigation, EFR, deficit, Cv and OUT_CROP_*
output; the files are float32: gate 1e-6 of max(1, |x|)), and G9's crop units
against the crop dump bit for bit (inputs and every SimUnit field).

The hooks off: VIC bit for bit (tile_fluxes and throughfall on a run without
a leafless vegetated tile, band_tfactor at the file's values).
"""
import os
import _paths  # noqa: E402

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from torchvic.vicwur import crop_feedback as CF                       # noqa
from torchvic.vicwur import core as VCORE                             # noqa
from torchvic.core import version as core_version                     # noqa

F32 = np.float32
D = torch.float64


@pytest.fixture(autouse=True)
def _core():
    old = core_version.CORE
    torch.set_default_device('cpu')
    yield
    core_version.set_core(old)


# ------------------------------------------------- scalar transcriptions
def ref_sweaf(cgn, pet):
    """Sweaf.c:21-32 in float."""
    cgn, pet = F32(cgn), F32(pet)
    s = F32(1. / (0.76 + 1.5 * float(pet)) - (5. - float(cgn)) * 0.10)
    if float(cgn) < 3.:
        s = F32(float(s) + (float(pet) - 0.6)
                / (float(cgn) * (float(cgn) + 3.)))
    a, b = F32(0.10), F32(0.95)              # limit(float, float, float)
    if s < a:
        return a
    if s <= b:
        return s
    return b


def ref_update(X):
    """crop_update_step_vars.c:4-256 for every cell, scalar C order."""
    N, B, NC = X['lai'].shape
    T, L = X['T'], X['L']
    P = X['prm']
    hist = ('lai', 'fcanopy', 'albedo', 'displacement', 'veg_rough')
    lib = ('trunk_ratio', 'wind_atten', 'wind_h', 'rad_atten', 'rgl', 'rarc',
           'rmin')
    out = {k: np.zeros((N, T)) for k in hist + lib}
    root = np.zeros((N, T, B, L))
    wcr = np.zeros((N, T, B, L))
    for n in range(N):
        tiles = sorted({int(X['tile'][n, c]) for c in range(NC)
                        if X['tile'][n, c] >= 0})
        H = {k: np.zeros(T) for k in hist}
        V = {k: np.zeros(T) for k in lib}
        # Clear (:42-82) is implicit (zeros); Set (:85-151)
        stale = None
        for b in range(B):
            for c in range(NC):
                if not X['active'][n, c]:
                    continue
                v = int(X['tile'][n, c])
                Cc = float(X['cc'][n, c])
                af = Cc * float(X['af'][n, b])
                af *= float(F32(P['Fcanopy'][c]))
                dev = float(F32(X['dev'][n, b, c]))
                tmin = dev if dev < 1.0 else 1.0
                mh = float(F32(P['MaxHeight'][c]))
                h = mh * tmin
                H['fcanopy'][v] += af
                H['lai'][v] += float(F32(X['lai'][n, b, c])) * af
                H['albedo'][v] += float(F32(P['Albedo'][c])) * af
                H['displacement'][v] += h * 0.67 * af
                H['veg_rough'][v] += h * 0.123 * af
                V['trunk_ratio'][v] += float(F32(P['TrunkRatio'][c])) * af
                V['wind_atten'][v] += float(F32(P['WindAttenuation'][c])) \
                    * af
                V['wind_h'][v] += float(F32(mh) + F32(1)) * af
                V['rad_atten'][v] += float(F32(
                    P['RadiationAttenuation'][c])) * af
                V['rgl'][v] += float(F32(P['RGL'][c])) * af
                V['rarc'][v] += float(F32(
                    P['MaxArchitecturalResistance'][c])) * tmin * af
                V['rmin'][v] += float(F32(P['MinStomatalResistance'][c])) \
                    * af
                rd = float(F32(X['rd'][n, b, c]))
                d0 = float(X['soil']['depth0'][n]) * 100.0
                if rd > d0:
                    root[n, v, b, 0] += (d0 / rd) * af
                    root[n, v, b, 1] += (1 - root[n, v, b, 0]) * af
                else:
                    root[n, v, b, 0] += af
                frac = float(F32(1) - ref_sweaf(P['CropGroupNumber'][c],
                                                X['pet'][n, b, c]))
                for l in range(L):
                    wp = float(X['soil']['wpwp'][n, l])
                    wf = float(X['soil']['wfc'][n, l])
                    wcr[n, v, b, l] += (wp + (wf - wp) * frac) * af
                stale = (v, b)
        # Transform (:154-255)
        for v in tiles:
            fc, lai = H['fcanopy'][v], H['lai'][v]
            if fc == 0 or lai == 0:
                V['rarc'][v] = 100.0
                V['trunk_ratio'][v] = 0.0
                V['wind_atten'][v] = 0.5
                V['wind_h'][v] = 10.0
                V['rmin'][v] = 0.0
                V['rad_atten'][v] = 0.5
                V['rgl'][v] = 100
                H['displacement'][v] = float(X['soil']['rough'][n])
                H['veg_rough'][v] = float(X['soil']['rough'][n])
                if stale is not None:
                    root[n, stale[0], stale[1], 0] = 1.0
                    root[n, stale[0], stale[1], 1] = 0.0
                for b in range(B):
                    for l in range(L):
                        wcr[n, v, b, l] = float(X['soil']['wcr'][n, l])
            else:
                for k in lib:
                    V[k][v] /= fc
                H['displacement'][v] /= fc
                H['veg_rough'][v] /= fc
                if stale is not None:
                    rs = 0.
                    for l in range(L):
                        rs += root[n, stale[0], stale[1], l]
                    for l in range(L):
                        # 0 / 0 is NaN in C (a stale band without area,
                        # which nothing reads); the kernel leaves 0
                        root[n, stale[0], stale[1], l] = (
                            root[n, stale[0], stale[1], l] / rs if rs != 0
                            else np.nan)
                for b in range(B):
                    for l in range(L):
                        wcr[n, v, b, l] /= fc
            if fc != 0 and lai / fc < 1.:
                H['fcanopy'][v] = lai
            f = H['fcanopy'][v]
            H['albedo'][v] = f * H['albedo'][v] + (1 - f) * 0.2
            if f < 0:
                H['fcanopy'][v] = 0.0
            elif f > 1:
                H['fcanopy'][v] = 1.0
            elif f < 0.0001:
                H['fcanopy'][v] = 0.0001
        for k in hist:
            out[k][n] = H[k]
        for k in lib:
            out[k][n] = V[k]
    out['root'], out['wcr'] = root, wcr
    return out


PRM = ('Fcanopy MaxHeight Albedo TrunkRatio WindAttenuation '
       'RadiationAttenuation RGL MaxArchitecturalResistance '
       'MinStomatalResistance CropGroupNumber').split()


def _random(seed, N=7, B=3, NC=3, T=9, L=3):
    """Random crop states and statics: classes 0 and 1 share tile 4, class
    2 is on tile 7; some units inactive, some bands without area, LAI zero
    on some, Development on both sides of 1, RootDepth on both sides of
    the top layer, crop group numbers below and above 3."""
    g = np.random.default_rng(seed)
    X = dict(T=T, L=L)
    X['tile'] = np.tile(np.array([4, 4, 7][:NC]), (N, 1))
    X['tile'][1] = [4, 7, 2][:NC]                # three tiles in one cell
    X['active'] = g.random((N, NC)) > 0.2
    X['active'][0] = False                        # a cell without units
    af = g.random((N, B))
    af[g.random((N, B)) < 0.25] = 0.0
    X['af'] = af / np.maximum(af.sum(1, keepdims=True), 1e-9)
    X['cc'] = g.uniform(0.0, 0.4, (N, NC))
    if N > 2:
        X['cc'][2, 0] = 0.0
    X['lai'] = np.where(g.random((N, B, NC)) < 0.3, 0.0,
                        g.uniform(0, 4, (N, B, NC))).astype(F32)
    if N > 3:
        X['lai'][3] = 0.0                         # every crop leafless
    X['dev'] = g.uniform(0, 2, (N, B, NC)).astype(F32)
    X['rd'] = g.uniform(0, 60, (N, B, NC)).astype(F32)
    X['pet'] = g.uniform(0, 0.9, (N, B, NC)).astype(F32)
    X['prm'] = {k: g.uniform(0.1, 2.0, NC).astype(F32) for k in PRM}
    X['prm']['CropGroupNumber'] = np.array([2.5, 4.5, 1.0][:NC], F32)
    X['prm']['Fcanopy'] = np.array([0.9, 0.7, 1.0][:NC], F32)
    X['soil'] = dict(depth0=g.uniform(0.1, 0.4, N),
                     wpwp=g.uniform(10, 40, (N, L)),
                     wfc=g.uniform(60, 120, (N, L)),
                     wcr=g.uniform(40, 60, (N, L)),
                     rough=g.uniform(0.001, 0.01, N))
    return X


def _kernel(X):
    N, B, NC = X['lai'].shape
    st = {'st_LAI': torch.as_tensor(X['lai']),
          'st_Development': torch.as_tensor(X['dev']),
          'st_RootDepth': torch.as_tensor(X['rd']),
          'met_PotEvaptrans': torch.as_tensor(X['pet'])}
    tile = torch.as_tensor(X['tile'], dtype=torch.long)
    ct = torch.zeros(N, X['T'], dtype=torch.bool)
    for n in range(N):
        for c in range(NC):
            ct[n, X['tile'][n, c]] = True
    lst = sorted({int(v) for v in X['tile'].reshape(-1)})
    return CF.update_step_vars(
        st, {k: torch.as_tensor(v) for k, v in X['prm'].items()},
        torch.as_tensor(X['cc'], dtype=D), torch.as_tensor(X['af'], dtype=D),
        torch.as_tensor(X['active']), tile, ct, lst,
        {k: torch.as_tensor(v, dtype=D) for k, v in X['soil'].items()},
        X['T']), ct


# ------------------------------------------------------------- the kernels
def test_sweaf_matches_the_float_c():
    g = np.random.default_rng(0)
    cgn = torch.as_tensor(np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0] * 40,
                                   F32))
    pet = torch.as_tensor(g.uniform(0, 3.0, cgn.shape[0]).astype(F32))
    pet[:3] = torch.tensor([0.0, 0.6, 1.4])
    got = CF.sweaf(cgn, pet).numpy()
    want = np.array([float(ref_sweaf(a, b)) for a, b in
                     zip(cgn.numpy(), pet.numpy())])
    assert np.array_equal(got, want)
    assert got.min() >= float(F32(0.10)) and got.max() <= float(F32(0.95))
    # every branch: below 0.10, between, above 0.95, CGN < 3
    assert (got == float(F32(0.10))).any() and (got == float(F32(0.95))).any()


@pytest.mark.parametrize('seed', range(6))
def test_update_step_vars_matches_the_c_order(seed):
    X = _random(seed)
    got, ct = _kernel(X)
    want = ref_update(X)
    ctn = ct.numpy()
    for k in CF.HIST + CF.LIB:
        g, w = got[k].numpy(), want[k]
        assert np.array_equal(g[ctn], w[ctn]), k
    for k in ('root', 'wcr'):
        g, w = got[k].numpy(), want[k]
        m = np.broadcast_to(ctn[:, :, None, None], g.shape)
        # a stale band without area: C's 0 / 0 is NaN there (never read:
        # nothing runs on a band without area); the kernel's root stays
        # finite
        nan = np.broadcast_to(np.isnan(w).any(-1, keepdims=True), w.shape)
        for n, t, b, l in zip(*np.nonzero(nan)):
            assert X['af'][n, b] == 0
        assert np.isfinite(g).all()
        assert np.array_equal(g[m & ~nan], w[m & ~nan]), k


def test_update_step_vars_quirks():
    """The stale veg_var: only the last band of the cell's last active
    class has normalised roots (crop_update_step_vars.c:183-184, 208-214);
    every other band's roots are the accumulated, unnormalised ones, with
    root[1] built on the running root[0] (:127-133). The layer Wcr of a
    band is divided by the cover of all bands (:216-223)."""
    X = _random(11, N=2, B=3, NC=2, T=6)
    X['tile'][:] = [[2, 5], [2, 5]]
    X['active'][:] = True
    X['af'][:] = [0.2, 0.5, 0.3]
    X['lai'][:] = 2.0
    X['rd'][:] = 50.0                    # deeper than the top layer
    X['soil']['depth0'][:] = 0.3
    got, _ = _kernel(X)
    r = got['root'].numpy()
    # stale: class 1 (tile 5), band 2: normalised
    assert abs(r[0, 5, 2].sum() - 1.0) < 1e-15
    # tile 2 and the other bands of tile 5: not normalised
    for t, b in ((2, 0), (2, 1), (2, 2), (5, 0), (5, 1)):
        assert r[0, t, b].sum() < 0.5
    # root[1] on the running root[0]: one unit of tile 2 in band 0
    a = X['cc'][0, 0] * 0.2 * float(F32(X['prm']['Fcanopy'][0]))
    r0 = (30.0 / 50.0) * a
    assert abs(r[0, 2, 0, 0] - r0) < 1e-15
    assert abs(r[0, 2, 0, 1] - (1 - r0) * a) < 1e-15
    # Wcr: band scaling (the sum over bands of a / the cover of all bands)
    w = got['wcr'].numpy()
    fcv = sum(X['cc'][0, 0] * f * float(F32(X['prm']['Fcanopy'][0]))
              for f in (0.2, 0.5, 0.3))
    frac = float(F32(1) - ref_sweaf(X['prm']['CropGroupNumber'][0],
                                    X['pet'][0, 0, 0]))
    wp, wf = X['soil']['wpwp'][0, 0], X['soil']['wfc'][0, 0]
    want = (wp + (wf - wp) * frac) * a / fcv
    assert abs(w[0, 2, 0, 0] - want) < 1e-12 * want
    # no crop growing (LAI 0): the soil defaults and the stale root 1, 0
    X['lai'][:] = 0.0
    got, _ = _kernel(X)
    assert float(got['rarc'][0, 2]) == 100.0
    assert float(got['fcanopy'][0, 2]) == 0.0001
    assert float(got['albedo'][0, 2]) == 0.2
    assert np.array_equal(got['root'].numpy()[0, 5, 2], [1.0, 0.0, 0.0])
    assert np.array_equal(got['wcr'].numpy()[0, 2, 1], X['soil']['wcr'][0])


def ref_adjust(flag, cv, af, act, irr_tile, omax, active, tile_of, cult,
               start_doy, doy, leap):
    flag = flag.copy()
    N, NI, B = flag.shape
    NC = active.shape[1]
    for n in range(N):
        for i in range(NI):
            if not act[n, i] or not cv[n, i] > 0:
                continue
            for b in range(B):
                if not af[n, b] > 0 or not flag[n, i, b]:
                    continue
                crop = cultd = off = False
                for c in range(NC):
                    if not active[n, c] or tile_of[n, c] != irr_tile[i]:
                        continue
                    crop = True
                    if cult[n, b, c] > 0:
                        cultd = True
                    pd = start_doy[n, c]
                    if pd < doy:
                        pd += 365 + leap
                    if pd - doy < omax[n, i]:
                        off = True
                if crop and not cultd and not off:
                    flag[n, i, b] = False
    return flag


@pytest.mark.parametrize('seed', range(4))
def test_adjust_requirement_matches_the_c(seed):
    g = np.random.default_rng(seed)
    N, NI, B, NC = 6, 3, 4, 3
    flag = g.random((N, NI, B)) < 0.7
    cv = np.where(g.random((N, NI)) < 0.2, 0.0, g.random((N, NI)))
    af = np.where(g.random((N, B)) < 0.2, 0.0, g.random((N, B)))
    act = g.random((N, NI)) < 0.9
    irr_tile = np.array([3, 6, 9])
    omax = g.integers(0, 30, (N, NI))
    active = g.random((N, NC)) < 0.85
    tile_of = g.choice([3, 6, 8], (N, NC))
    cult = (g.random((N, B, NC)) < 0.3).astype(np.int32)
    sdoy = g.integers(1, 366, (N, NC))
    for doy, leap in ((40, 0), (200, 1), (365, 0)):
        want = ref_adjust(flag, cv, af, act, irr_tile, omax, active, tile_of,
                          cult, sdoy, doy, leap)
        got = CF.adjust_requirement(
            torch.as_tensor(flag), torch.as_tensor(cv), torch.as_tensor(af),
            torch.as_tensor(act), torch.as_tensor(irr_tile),
            torch.as_tensor(omax), torch.as_tensor(active),
            torch.as_tensor(tile_of), torch.as_tensor(cult),
            torch.as_tensor(sdoy), torch.tensor(doy), torch.tensor(leap))
        assert np.array_equal(got.numpy(), want)
        assert (want != flag).any()          # the veto acted


def test_wofost_water_matches_the_c():
    """irr_use.c:568-603: per entry, the band's crops of the entry's tile,
    List(Irrigation) x Cc x MM_PER_CM summed in list order; :606-640 the
    log_err of a crop with irrigation management on a tile not irrigated."""
    g = np.random.default_rng(3)
    N, NI, B, NC = 5, 2, 3, 3
    amount = np.array([1.5, 0.0, 2.25], F32)
    cc = g.uniform(0, 0.4, (N, NC))
    active = g.random((N, NC)) < 0.8
    tile_of = np.tile(np.array([4, 4, 6]), (N, 1))
    irr_tile = np.array([4, 6])
    act = np.ones((N, NI), bool)
    act[2, 1] = False                       # tile 6 not irrigated in cell 2
    active[2, 2] = True
    af = np.full((N, B), 1.0 / B)
    w, bad = CF.wofost_water(torch.as_tensor(amount), torch.as_tensor(cc),
                             torch.as_tensor(active),
                             torch.as_tensor(tile_of),
                             torch.as_tensor(irr_tile), torch.as_tensor(act),
                             torch.as_tensor(af), B)
    for n in range(N):
        for i in range(NI):
            acc = 0.0
            for c in range(NC):
                if active[n, c] and tile_of[n, c] == irr_tile[i]:
                    acc += float(amount[c]) * cc[n, c] * 10.0
            assert np.array_equal(w[n, i].numpy(), np.full(B, acc))
    assert bool(bad[2]) and not bool(bad[[0, 1, 3, 4]].any())


def test_evap_outputs_match_the_c():
    """crop_output.c:1003-1032, EVAP adding the class's running sums after
    each unit."""
    g = np.random.default_rng(5)
    N, T, B, L, NC = 4, 8, 3, 3, 3
    es = g.uniform(0, 2, (N, T, B))
    tr = g.uniform(0, 2, (N, T, B, L))
    ca = g.uniform(0, 1, (N, T, B))
    cc = g.uniform(0, 0.4, (N, NC))
    af = g.random((N, B))
    gr = (g.random((N, B, NC)) < 0.6).astype(np.int32)
    active = g.random((N, NC)) < 0.8
    tile_of = np.tile(np.array([2, 5, 5]), (N, 1))
    got = CF.evap_outputs(*(torch.as_tensor(x) for x in (
        es, tr, ca, cc, af, gr, active, tile_of)))
    want = {k: np.zeros((N, NC)) for k in got}
    for n in range(N):
        for b in range(B):
            for c in range(NC):
                if not active[n, c]:
                    continue
                t = tile_of[n, c]
                w = lambda x: x * af[n, b] * cc[n, c] * gr[n, b, c]
                for l in range(L):
                    want['OUT_CROP_EVAP_BARE'][n, c] += w(es[n, t, b] if l == 0
                                                          else 0.0)
                    want['OUT_CROP_TRANSP_VEG'][n, c] += w(tr[n, t, b, l])
                want['OUT_CROP_EVAP_CANOP'][n, c] += w(ca[n, t, b])
                want['OUT_CROP_EVAP'][n, c] += (
                    want['OUT_CROP_EVAP_BARE'][n, c]
                    + want['OUT_CROP_TRANSP_VEG'][n, c]
                    + want['OUT_CROP_EVAP_CANOP'][n, c])
    for k in got:
        assert np.array_equal(got[k].numpy(), want[k]), k


# --------------------------------------------------- the hooks off: identity
G0 = os.path.join(_paths.env_path(('TORCHVIC_VICWUR_VERIFICATION', 'VICWUR_VERIFICATION'), 'verification'), 'vicwur_19', 'gruns', 'R1', 'G0a')


def _g0(tmp_path, steps):
    if not os.path.exists(os.path.join(G0, 'global_W3.txt')):
        pytest.skip('VIC-WUR reference run absent: ' + G0)
    from torchvic import case
    gf = VCORE.image_global(os.path.join(G0, 'global_W3.txt'),
                            str(tmp_path / 'global.txt'))
    return case.load(gf, device=torch.device('cpu'), core='vicwur3',
                     steps=steps), gf


def _same(o0, o1):
    return [k for k, v in o0.items() if torch.is_tensor(v) and k in o1
            and not torch.equal(v, o1[k])]


def test_tile_fluxes_and_throughfall_change_nothing(tmp_path):
    """tile_fluxes only reads; throughfall acts on a vegetated tile that
    is not VEG with fcanopy > 0, which R1's vegetation never has: VIC's
    outputs are those of the same hooks without them, bit for bit. The
    per-tile fluxes are VIC's: transpiration and bare-soil evaporation sum
    to OUT_TRANSP_VEG and OUT_EVAP_BARE."""
    c, _ = _g0(tmp_path, 60)
    with torch.no_grad():
        o0 = c.run(p=c.params(), vicwur_core=VCORE.VicwurCore(
            water_stress=True, diagnostics=True))
        o1 = c.run(p=c.params(), vicwur_core=VCORE.VicwurCore(
            water_stress=True, diagnostics=True, tile_fluxes=True,
            throughfall=True))
    bad = _same(o0, o1)
    # a tile without cover (Cv 0: C does not run it,
    # VIC-WUR 3.0 vic_run.c:137) and a leafless month differs in its own
    # state only; it weighs nothing
    assert bad in ([], ['soil_moisture_tile']), bad
    ex = c.snowbands
    first = [int(np.flatnonzero(np.asarray(ex['cell']) == n)[0])
             for n in range(int(ex['n_cell']))]
    live = c.mosaic.cv[first] > 0                          # [cell, tile]
    assert torch.equal(o0['soil_moisture_tile'][live],
                       o1['soil_moisture_tile'][live])
    tf = o1['vicwur']['tile_fluxes_end']
    assert set(tf) == {'tf_pet', 'tf_transp', 'tf_esoil', 'tf_canop'}
    # the last step's tile fluxes are the diagnostics' last rows
    assert torch.equal(tf['tf_transp'], o1['vicwur']['transp_tile'][-1])
    assert torch.equal(tf['tf_esoil'], o1['vicwur']['esoil_tile'][-1])
    assert float(tf['tf_pet'].max()) > 0.5


def test_throughfall_loses_the_cover_share_of_a_leafless_tile(tmp_path):
    """VIC-WUR 3.0 func_surf_energy_bal.c:785-786 with surface_fluxes.c:267:
    a vegetated tile made leafless (LAI 0, fcanopy MIN_FCANOPY) passes
    (1 - fcanopy) of the rain; the lost share is VIC's water error of the
    step."""
    c, _ = _g0(tmp_path, 40)
    m = c.mosaic
    nc, nt = m.cv.shape
    k = int(torch.argmax(m.cv[0]))
    mask = torch.zeros(nc, nt, dtype=torch.bool)
    mask[:, k] = True
    veg = {'lai': torch.zeros(nc, nt, dtype=D),
           'fcanopy': torch.full((nc, nt), 1e-4, dtype=D)}
    with torch.no_grad():
        o0 = c.run(p=c.params(), vicwur_core=VCORE.VicwurCore(
            veg=veg, veg_mask=mask))
        o1 = c.run(p=c.params(), vicwur_core=VCORE.VicwurCore(
            veg=veg, veg_mask=mask, throughfall=True))
    rain = c.atm['prcp'][0]                 # the columns' rain, step 0
    lost = (m.cv[:, k] * 1e-4 * rain).to(D)
    lost_cells = c.contract_bands({'x': lost.reshape(1, -1).expand(
        len(c.dates), -1)})['x'][0]
    d0 = (o0['water_error'][0] - o1['water_error'][0]).abs()
    assert float(lost_cells.max()) > 1e-5
    assert torch.allclose(d0, lost_cells, rtol=1e-8, atol=1e-12)


def test_band_tfactor_at_the_file_values_is_bitwise(tmp_path):
    """case.load(band_tfactor=...) (VIC-WUR 3.0 crop_init.c:157-163) at the
    file's own band 0 Tfactor loads the same case; the crop file's (0) shifts
    band 0's air temperature and leaves the cell's forcing echo."""
    from torchvic import case
    c0, gf = _g0(tmp_path, 40)
    ex = c0.snowbands
    tf0 = np.zeros(int(ex['n_cell']))
    b0 = np.asarray(ex['band']) == 0
    tf0[np.asarray(ex['cell'])[b0]] = np.asarray(ex['tfactor'])[b0]
    c1 = case.load(gf, device=torch.device('cpu'), core='vicwur3', steps=40,
                   band_tfactor={0: tf0})
    assert torch.equal(c0.atm['tair'], c1.atm['tair'])
    with torch.no_grad():
        assert not _same(c0.run(), c1.run())
    c2 = case.load(gf, device=torch.device('cpu'), core='vicwur3', steps=40,
                   band_tfactor={0: np.zeros_like(tf0)})
    cols = np.flatnonzero(b0)
    cells = np.asarray(ex['cell'])[cols]
    assert torch.equal(c2.atm['tair'][:, cols], c2._cell_echo['tair'][:,
                                                                       cells])
    assert torch.equal(c2._cell_echo['tair'], c0._cell_echo['tair'])
    other = np.flatnonzero(~b0)
    assert torch.equal(c2.atm['tair'][:, other], c0.atm['tair'][:, other])


# ------------------------------------------------- R1 against VIC-WUR 3.0
import test_vicwur_connectors_b as TB                                  # noqa

CROPV = ['OUT_CROP_' + s for s in (
    'CULTIVATE GROW EVAP EVAP_BARE EVAP_CANOP TRANSP_VEG DVS WLV WST WSO WRT '
    'WDLV WDST WDRT NLV NST NSO NRT LAI WSTRESS NSTRESS TMAXSTRESS '
    'TMINSTRESS NPKI NNI NSOIL NUPT NDEM PNI PSOIL PUPT PDEM KNI KSOIL KUPT '
    'KDEM ROOTDEPTH').split()]
PLUGIN = TB.ROUT_VARS + TB.IRR_VARS + (
    ('OUT_EFR_DISCHARGE', 'vicwur_efr', 'efr_discharge'),
    ('OUT_EFR_BASEFLOW', 'vicwur_efr', 'efr_baseflow'),
    ('OUT_EFR_MOIST', 'vicwur_efr', 'efr_moist'),
    ('OUT_NONREN_DEFICIT', 'vicwur_nonrenew', 'deficit'),
    ('OUT_CV', 'vicwur_landuse', 'cv')) + tuple(
        (k, 'vicwur_crop', k) for k in CROPV)
CDUMP = _paths.env_path(('TORCHVIC_VICWUR_111_GRUNS', 'VICWUR_111_GRUNS'), 'verification/vicwur_111/gruns/R1')


def _spy_crop(m):
    """Record the crop's inputs and unit states after every crop_run."""
    import wofost_npk_cases as WC
    cr = m.runtime.connectors['vicwur_crop']
    rec = []
    orig = cr.m.crop_run

    def spy(s, cal, t, at, sw, co2, pot, ws, cv=None):
        out = orig(s, cal, t, at, sw, co2, pot, ws, cv=cv)
        rec.append(dict(t=int(t), pot=pot.cpu().numpy(),
                        ws=ws.cpu().numpy(),
                        **{k: np.asarray(v) for k, v in
                           WC.fields(cr.m, out).items()}))
        return out
    cr.m.crop_run = spy
    return rec


def _check_crop_dump(rec, run):
    """The crop units against the crop dump (record tag 8) bit for bit."""
    import test_vicwur_wofost_l2 as WL
    import wofost_npk_cases as WC
    if not os.path.exists(os.path.join(CDUMP, WL.RUNS[run], 'dump.bin')):
        return False
    L = WL.load(run)
    Dd = L['D']
    n = 0
    for r in rec:
        t = r['t']
        ex = np.isfinite(Dd['Sowing'][t])
        # the VIC fields the crop reads: VIC's pot_evap of the tile and band
        # and the stored (irrigation's) water stress, double in C
        for nm, x in (('in_pot_evap', r['pot']),
                      ('in_water_stress', r['ws'])):
            y = Dd[nm][t][ex]
            assert np.abs(x[ex] - y).max() <= 1e-10 * max(1.0, np.abs(
                y).max()), (nm, t)
        for nm in WC.dump_field_names():
            if nm.endswith('death_ro') or nm not in r:
                continue
            y = Dd[nm][t]
            x = np.broadcast_to(r[nm], y.shape)
            a = np.broadcast_to(ex, y.shape)
            ok = (x[a] == y[a]) | (np.isnan(x[a]) & np.isnan(y[a]))
            assert ok.all(), (nm, t)
        n += 1
    assert n == Dd['Sowing'].shape[0]
    return True


@pytest.mark.parametrize('run', ['G9', 'G9b', 'G10'])
def test_l2_crop_feedback_matches_w3(tmp_path, run):
    """All eleven modules (G9: G8 + CO2 + WOFOST; G9b with
    WOFOST_CALC_MINERALIZATION; G10 with every sector, compensation,
    efficient and offset irrigation) over the 7 years: every VIC output,
    routing, irrigation, EFR, deficit, Cv and the 37 OUT_CROP_* within
    1e-6 of max(1, |x|) of the float32 VIC-WUR 3.0 files (observed 1.2e-7),
    with no divergence on any threshold; G9's and G10's crop units against
    the crop dump bit for bit; the ledger closes."""
    TB._need(run)
    m, c = TB.build(run, 2557, tmp_path, group='l2_%s' % run)
    with torch.inference_mode():
        m.initialize()
        rec = _spy_crop(m)
        m.run()
        m.finalize(write=False)
    worst = TB.compare(m, c, run, PLUGIN,
                       local_dam_cells=TB._local_dam_cells(run))
    print(run, sorted(worst.items(), key=lambda kv: -kv[1])[:6])
    assert max(worst.values()) < TB.GATE, worst
    TB.check_ledger(m, 2557)
    o = m.outputs('vicwur_crop')
    assert float(o['OUT_CROP_GROW'].max()) > 0
    assert float(o['OUT_CROP_EVAP'].max()) > 0.1
    cr = m.runtime.connectors['vicwur_crop']
    assert int(cr.s['err'].max()) == 0
    if run in ('G9', 'G10'):
        _check_crop_dump(rec, run)


def test_g9_feedback_reaches_vic_and_irrigation(tmp_path):
    """The feedback acts where C acts: VIC's crop tiles carry the crop's
    vegetation (and the soil defaults between seasons), irrigation reads
    the crop's Wcr and roots, and crop_adjust_requirement clears flags."""
    TB._need('G9')
    m, c = TB.build('G9', 200, tmp_path, group='fb_g9')
    grp = m.runtime.connectors['vicwur_routing'].group
    cr, irr = grp.roles['crop'], grp.roles['irrigation']
    cleared = []
    orig = CF.adjust_requirement

    def spy(flag, *a):
        out = orig(flag, *a)
        cleared.append(int((flag & ~out).sum()))
        return out
    CF.adjust_requirement = spy
    try:
        with torch.inference_mode():
            m.initialize().run()
            m.finalize(write=False)
    finally:
        CF.adjust_requirement = orig
    assert sum(cleared) > 0
    assert irr.wcr_fn is not None and irr.root_fn is not None
    assert irr.wofost_water is not None
    fb = cr._fb
    ct = cr._tiles()[0]
    # crop tiles: LAI from the crops; between seasons the soil defaults
    assert float(fb['lai'][ct].max()) > 0 or float(
        fb['fcanopy'][ct].min()) == 0.0001
    w = irr._wcr(irr.st['leftover'].shape[2])
    assert w.shape[:2] == (cr.N, irr.NI)


# --------------------------------------------------------------------- GPU
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_g9_crop_feedback_matches_w3(tmp_path):
    """G9 (G8 + CO2 + WOFOST) on CUDA, a year: the same gate against
    VIC-WUR 3.0, the ledger closes."""
    TB._need('G9')
    m, c = TB.run('G9', 365, tmp_path, device='cuda', group='gpu_fb_g9')
    worst = TB.compare(m, c, 'G9', PLUGIN,
                       local_dam_cells=TB._local_dam_cells('G9'))
    print('G9 cuda', sorted(worst.items(), key=lambda kv: -kv[1])[:6])
    assert max(worst.values()) < TB.GATE, worst
    TB.check_ledger(m, 365)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_feedback_does_no_host_sync(tmp_path):
    """U (crop_update_step_vars and its VIC columns), P1
    (crop_adjust_requirement), P3 (irr_wofost, crop_run on VIC's PET) and
    O (OUT_CROP_EVAP*) of a day on CUDA: no host synchronisation."""
    TB._need('G9')
    m, c = TB.build('G9', 62, tmp_path, device='cuda', group='gpu_fb_sync')
    with torch.inference_mode():
        m.initialize().run(until=31 * 86400.0)
    vic = m.runtime.connectors['vic']
    grp = vic.phase_group
    cr = grp.roles['crop']
    st = dict(vic.state)
    if 'lanes' in st:
        # a GPU run carries its state on the covered tiles: on the columns
        ln = vic._lanes(vic.driver.mosaic(), st)
        st['moist'], st['wdew'] = ln.scatter(st['moist']), \
            ln.scatter(st['wdew'])
    sv = vic._soil_view(st['moist'], st.get('soil_ice'), None)
    v = vic._wur_view(sv)
    vic._set_fluxes(v, vic._wur_last)
    assert v.tile_fluxes is not None and 'pet' in v.tile_fluxes
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        with torch.inference_mode():
            grp._set_view(v)
            grp._set_time(30 * 86400.0)
            grp._phases(('P1', 'P2', 'P3', 'O'))
            grp._set_time(31 * 86400.0)
            grp._phases(('F', 'U'))
            cols = cr._columns()
            cr.lai_prev_columns()
            cr.wofost_water()
    finally:
        torch.cuda.set_sync_debug_mode('default')
    assert cols['wcr'].is_cuda and cols['root_frac'].is_cuda
