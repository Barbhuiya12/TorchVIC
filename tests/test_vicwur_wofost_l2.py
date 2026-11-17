# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.crop.wofost_npk replayed on the R1 G-runs with WOFOST.

The runs (tools/vicwur/cropdump_runs.py: G9, G9b with
WOFOST_CALC_MINERALIZATION, G10 with every plugin) use the crop-dump build
<data root>/build/vicwur3_cropdump (VIC-WUR 3.0 + the dump patch +
tools/vicwur/vicwur_cropdump.patch), which writes, after crop_run of every
step, the VIC fields the crop reads (air_temp, shortwave, the CO2 forcing,
the tile's pot_evap and water_stress, Cv) and the whole SimUnit state of
every (cell, band, crop) unit (record tag 8).

The kernel is set up from the run's own input files only (the WOFOST text
file, the crop nc, the parameter and domain files: setup_from_vicwur), fed
the dumped VIC fields day by day, and compared with the dumped C state on
every unit and day:
  * float32, C promotion, C-order leaf sums: bit for bit with the C (the C
    model is float);
  * float64: the error of the C's float arithmetic, reported as a bound;
  * OUT_CROP_* of the netCDF history (crop_put_rate_data) from the kernel
    state, bit for bit in float32.
The R1 WOFOST NPK inputs are synthetic placeholders: this is code
equivalence, not realism.  The only C values not compared are N/P/K death_ro
before a unit's first growth day: uninitialised heap memory in the C.
"""
import datetime
import glob
import os
import _paths  # noqa: E402
import struct

import numpy as np
import pytest
import torch

import wofost_npk_cases as C
from torchvic.crop.wofost_npk import (CropTemplates, Calendar, WofostNPK,
                                      setup_from_vicwur)
from torchvic.io import ncshare
from torchvic.vicwur import params as P

ROOT = _paths.env_path(('TORCHVIC_VICWUR_111_GRUNS', 'VICWUR_111_GRUNS'), 'verification/vicwur_111/gruns/R1')
RUNS = {'G9': 'G9_cdump', 'G9b': 'G9b_cdump', 'G10': 'G10_cdump'}
INPUTS = ('in_air_temp in_shortwave in_co2 in_pot_evap in_water_stress '
          'in_Tfactor in_AreaFract in_Cv in_Cc').split()
EXTRA = ('leaf_n leaf_sumw leaf_sumwa leaf_headage con_tsum1 con_tsum2'
         ).split()


def read_crop_dump(path):
    """-> dates [(y, m, d)], crop records [T, N, B, NC, F]."""
    dates, recs = [], []
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            tag, n = struct.unpack('<iq', h)
            if tag in (1, 8):
                a = np.frombuffer(f.read(8 * n), '<f8')
                if tag == 1:
                    dates.append(tuple(int(x) for x in a[1:4]))
                else:
                    recs.append(a.copy())
            else:
                f.seek(8 * n, 1)
    nc_, nb, F = (int(x) for x in recs[0][:3])
    a = np.stack([r[3:] for r in recs])
    return dates, a.reshape(len(recs), -1, nb, nc_, F)


def band_tfactor(grid, pr):
    """AreaFract and Tfactor [N,B] as the image driver sets them
    (vic_init.c:1011-1072): fractions normalised when their sum is off by
    more than 1e-5, the cell elevation replaced by the area-weighted band
    mean when they differ by more than 1e-5, Tfactor = (BandElev - mean) x
    LAPSE_RATE (initialize_parameters.c:39) on every band, also the empty
    ones (their crop units still register the temperature)."""
    af = grid.take(np.asarray(pr['AreaFract'].values, np.float64)).T.copy()
    be = grid.take(np.asarray(pr['elevation'].values, np.float64)).T
    el = grid.take(np.asarray(pr['elev'].values, np.float64)).copy()
    N, B = af.shape
    tf = np.zeros((N, B))
    for i in range(N):
        s = 0.0
        for j in range(B):
            s += af[i, j]
        if not abs(s - 1.0) <= 1e-5:
            af[i] = af[i] / s
        mean = 0.0
        for j in range(B):
            mean += be[i, j] * af[i, j]
        if not abs(el[i] - mean) <= 1e-5:
            el[i] = mean
        for j in range(B):
            tf[i, j] = (be[i, j] - el[i]) * -0.0065
    return af, tf


def _flag(g, key):
    v = g.get(key)
    return bool(v) and v[0][0].upper() == 'TRUE'


def load(run):
    rd = os.path.join(ROOT, RUNS[run])
    dump = os.path.join(rd, 'dump.bin')
    if not os.path.exists(dump):
        pytest.skip('dump absent: ' + dump)
    g = P.parse_global(os.path.join(rd, 'global_W3.txt'))
    grid = P.Grid(g['DOMAIN'][0][0], g['PARAMETERS'][0][0])
    opts = {k: _flag(g, 'WOFOST_' + v) for k, v in (
        ('PIRR', 'POTENTIAL_IRRIGATION'), ('PFERT', 'POTENTIAL_FERTILIZER'),
        ('PTEMP', 'POTENTIAL_TEMPERATURE'), ('DIST_TSUM', 'DIST_TSUM'),
        ('DIST_FERT', 'DIST_FERTILIZER'), ('CALC_MIN', 'CALC_MINERALIZATION'),
        ('CONTINUE', 'CONTINUE'), ('FORCE_TSUM', 'FORCE_TSUM'),
        ('FORCE_FERT', 'FORCE_FERTILIZER'))}
    season = _flag(g, 'WOFOST_DIST_SEASON')
    dmin = _flag(g, 'WOFOST_DIST_MINERALIZATION')
    params = g['PARAMETERS'][0][0]
    pr = ncshare.read(params)
    nveg = int(pr.sizes['veg_class'])
    af, tf = band_tfactor(grid, pr)
    with ncshare.read(g['CROP_PARAMETERS'][0][0]) as cds:
        nct = int(cds.sizes['crop_class'])
    crop = P.read_crop(g['CROP_PARAMETERS'][0][0], grid, nveg,
                       dist_season=season, dist_tsum=opts['DIST_TSUM'],
                       dist_fert=opts['DIST_FERT'], dist_min=dmin,
                       calc_min=opts['CALC_MIN'])
    tf[:, 0] = crop.Tfactor.numpy()          # crop_init.c:157-163
    wt = P.read_wofost_text(g['WOFOST_TEXT_PARAMETERS'][0][0], nct)
    tmpl = CropTemplates.from_wofost_text(wt)
    rows = np.nonzero(grid.active)[0]
    lat = np.asarray(grid.lat, np.float64)[rows]
    dates, rec = read_crop_dump(dump)
    names = INPUTS + C.dump_field_names() + EXTRA
    assert rec.shape[-1] == len(names)
    D = {nm: rec[..., j] for j, nm in enumerate(names)}
    cvd = np.nanmax(D['in_Cv'], axis=2)                   # [T,N,NC]
    setup = setup_from_vicwur(tmpl, crop, lat, af, tf, cvd[0],
                              dist_season=season,
                              dist_tsum=opts['DIST_TSUM'],
                              dist_fert=opts['DIST_FERT'], dist_min=dmin,
                              calc_min=opts['CALC_MIN'],
                              continue_=opts['CONTINUE'])
    return dict(rd=rd, g=g, tmpl=tmpl, setup=setup, opts=opts, dates=dates,
                D=D, names=names, cv=cvd)


_CACHE = {}


def loaded(run):
    if run not in _CACHE:
        _CACHE.clear()
        _CACHE[run] = load(run)
    return _CACHE[run]


def replay(L, dtype, exact_sums, device='cpu', lanes=None, **kw):
    tmpl, su, D = L['tmpl'], L['setup'], L['D']
    m = WofostNPK(tmpl, su, L['opts'], dtype=dtype, device=device,
                  exact_sums=exact_sums, lanes=lanes, **kw)
    dates = []
    for (y, mo, d) in L['dates']:
        dates.append((y, mo, d, datetime.date(y, mo, d).timetuple().tm_yday))
    cal = Calendar(dates, tmpl, device=device, dtype=dtype)
    s = m.init_state()
    T = len(dates)
    f64 = lambda a: torch.as_tensor(np.nan_to_num(a), dtype=torch.float64,
                                    device=device)
    cellv = lambda nm, t: f64(np.nanmax(D[nm][t].reshape(D[nm].shape[1],
                                                         -1), axis=1))
    out = {nm: [] for nm in L['names'] if nm not in INPUTS}
    rates = []
    ls = m.is_lane_state(s)
    for t in range(T):
        s = m.crop_run(s, cal, t, cellv('in_air_temp', t),
                       cellv('in_shortwave', t), cellv('in_co2', t),
                       f64(D['in_pot_evap'][t]),
                       f64(D['in_water_stress'][t]), cv=f64(L['cv'][t]))
        if not ls:
            rec = C.fields(m, s)
            for nm in out:
                out[nm].append(rec[nm])
        rates.append({k: v.cpu().numpy() for k, v in
                      m.rate_outputs(s, cal, t).items()})
    return m, s, ({} if ls else {k: np.stack(v) for k, v in out.items()}), \
        rates


def _check_setup(L):
    """The set-up built from the files equals what the C used."""
    D, su = L['D'], L['setup']
    exists = ~np.isnan(D['Sowing'][0])                    # [N,B,NC]
    act = np.broadcast_to(su.active[:, None, :], exists.shape)
    assert np.array_equal(exists, act)
    af = np.broadcast_to(su.area_fract[:, :, None], exists.shape)
    tf = np.broadcast_to(su.tfactor[:, :, None], exists.shape)
    assert np.array_equal(D['in_AreaFract'][0][exists], af[exists])
    assert np.array_equal(D['in_Tfactor'][0][exists], tf[exists])
    return exists


@pytest.mark.parametrize('run', sorted(RUNS))
def test_l2_fp32_bitwise(run):
    L = loaded(run)
    exists = _check_setup(L)
    m, s, got, rates = replay(L, torch.float32, exact_sums=True)
    assert int(s['err'].max()) == 0
    D = L['D']
    # the replay covers growth and harvest of every unit in a band > 0
    gr = np.nan_to_num(D['growing'])
    assert (np.diff(gr, axis=0) < 0).sum() >= 10
    # N/P/K death_ro (rate and state) are neither initialised nor copied in
    # C (wofost_init_library.c:145-146,167-168, wofost_copy_library.c:
    # 241-242,269-270): heap garbage until a unit's first growth day sets
    # them (RatesToZero, InitializeNutrients); never read before that.
    first = np.argmax(gr > 0, axis=0)
    first = np.where(gr.max(0) > 0, first, gr.shape[0])
    seen = np.arange(gr.shape[0])[:, None, None, None] >= first[None]
    bad = {}
    for nm, g in got.items():
        r = D[nm]
        a = np.broadcast_to(exists, r.shape)
        if nm.endswith('death_ro'):
            a = a & seen
        rr, gg = r[a], g[a]
        if nm in ('leaf_sumw', 'leaf_sumwa'):
            ok = np.isclose(rr, gg, rtol=1e-13, atol=0, equal_nan=True)
        else:
            ok = (rr == gg) | (np.isnan(rr) & np.isnan(gg))
        if not ok.all():
            bad[nm] = int((~ok).sum())
    assert not bad, bad
    # OUT_CROP_* of the history files
    files = sorted(glob.glob(os.path.join(L['rd'], 'W3/out/fluxes.*.nc')))
    T = len(L['dates'])
    for var in rates[0]:
        parts = [np.asarray(ncshare.read(p, decode_times=False)[var].values)
                 for p in files]
        ref = np.concatenate(parts)                       # [T,NC,ny,nx]
        assert ref.shape[0] == T
        grid = P.Grid(L['g']['DOMAIN'][0][0], L['g']['PARAMETERS'][0][0])
        ref = ref[:, :, grid.active]                      # [T,NC,N]
        got_v = np.stack([r[var] for r in rates]).transpose(0, 2, 1)
        assert np.array_equal(ref, got_v.astype(np.float32)), var


def test_l2_fp64_bound():
    """float64 against the float C: the error of the C's float arithmetic
    (scaled by each field's range over the run)."""
    L = loaded('G9')
    exists = _check_setup(L)
    m, s, got, rates = replay(L, torch.float64, exact_sums=False)
    D = L['D']
    worst = 0.0
    ex = np.broadcast_to(exists, D['st_storage'].shape)
    for nm in ('st_storage', 'st_leaves', 'st_stems', 'st_roots', 'st_LAI',
               'st_Development', 'N_st_storage', 'NPK_Indx'):
        r = D[nm][ex]
        g = got[nm][ex]
        worst = max(worst, np.nanmax(np.abs(r - g)) / np.nanmax(np.abs(r)))
    for nm in ('Sowing', 'Emergence', 'growing', 'CultivateDay'):
        assert np.array_equal(D[nm][ex], got[nm][ex]), nm
    print('L2 G9 float64 vs float C, worst scaled error %.3e' % worst)
    assert worst < 1e-4, worst



@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('prec', ['f32', 'f32exact', 'f32plain', 'f64',
                                  'f32ring', 'f64ring', 'f32eager',
                                  'f32ringls', 'f32ringlseager'])
def test_l2_lanes_equal_units(prec):
    """On CUDA the day on the running units' lanes (the default there)
    equals the day on every (cell, band, crop) unit, state and OUT_CROP_*
    bit for bit; with the C-order sums the lanes equal the float C. Without
    gradient the lane day is one fused region (core.fusion) where
    _fusable; 'f32eager' runs it op by op."""
    import contextlib
    import unittest.mock as um
    L = loaded('G9')
    dt = torch.float64 if prec.startswith('f64') else torch.float32
    ex = prec == 'f32exact'
    ctx = contextlib.nullcontext()
    kw = {}
    if 'ring' in prec:
        # a row of 336 classes (as the ISC's) on a ring of 48 slots: the
        # leaf sums as eager's trees on the live classes (leaf_kernels)
        from torchvic.crop.wofost_npk import leaf_kernels as LK
        kw = dict(nleaf=336)
        if not LK.supported(336, int(L['setup'].active.sum())):
            pytest.skip('too few units for the ring kernels')
        kw_ring = dict(kw, ring=48, lane_state='ls' in prec)
    if prec == 'f32plain':
        real = WofostNPK.__init__

        def init(self, *a, **k):
            k['c_promote'] = False
            real(self, *a, **k)
        ctx = um.patch.object(WofostNPK, '__init__', init)
    grad = torch.enable_grad() if prec.endswith('eager') else \
        torch.no_grad()
    with ctx, grad:
        ma, sa, ga, ra = replay(L, dt, ex, device='cuda', lanes=None,
                                **(kw_ring if kw else {}))
        mb, sb, gb, rb = replay(L, dt, ex, device='cuda', lanes=False, **kw)
    if kw:
        sa = dict(sa, **ma.leaf_arrays(sa))
        live = _live_dense(mb, sb)
        sb = dict(sb, **{k: torch.where(live, x, torch.zeros_like(x))
                         for k, x in mb.leaf_arrays(sb).items()})
    assert ma.lanes and not mb.lanes
    if ma.is_lane_state(sa):
        # the running units' fields (the lanes) and the leaf classes
        for k in sb:
            if k in ('lw', 'la', 'lage') or k.startswith('con_'):
                continue
            x, y = ma.lane_values(sa, k), mb.lane_values(sb, k)
            assert x.dtype == y.dtype and x.shape == y.shape, k
            assert bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all()
                        ), k
        h = mb.lane_values(sb, 'lh').long()
        tt = mb.lane_values(sb, 'lt').long()
        kk = torch.arange(mb.K, device=h.device)
        live = (kk >= h[:, None]) & (kk < tt[:, None])
        R = sa['lw'].shape[-1]
        for k in ('lw', 'la', 'lage'):
            x = torch.where(live, sa[k][:, kk % R], torch.zeros(()).to(sa[k]))
            y = torch.where(live, sb[k], torch.zeros(()).to(sb[k]))
            assert torch.equal(x, y), k
        for x, y in zip(ra, rb):
            for k in y:
                assert np.array_equal(x[k], y[k], equal_nan=True), k
        return
    for k in sb:
        x, y = sa[k], sb[k]
        assert x.dtype == y.dtype and x.shape == y.shape, k
        assert bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all()), k
    for k in gb:
        assert np.array_equal(ga[k], gb[k], equal_nan=True), k
    for x, y in zip(ra, rb):
        for k in y:
            assert np.array_equal(x[k], y[k], equal_nan=True), k
    if ex:
        exists = _check_setup(L)
        D = L['D']
        for nm in ('st_LAI', 'st_storage', 'st_leaves', 'NPK_Indx',
                   'st_Development', 'N_st_storage'):
            a = np.broadcast_to(exists, D[nm].shape)
            r, g = D[nm][a], ga[nm][a]
            assert ((r == g) | (np.isnan(r) & np.isnan(g))).all(), nm


def _live_dense(m, s):
    k = torch.arange(m.K, device=s['lh'].device)
    return (k >= s['lh'].long()[..., None]) & (k < s['lt'].long()[..., None])
