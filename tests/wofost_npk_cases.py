# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Cases and I/O of the WOFOST-NPK oracle
(tests/oracle/ref_vicwur_wofost.c) for tests/test_vicwur_wofost.py.

The crop templates are the R1 crop and management files
(<data root>/vicwur/R1/wofost, synthetic NPK values: code equivalence only),
plus variants that switch on what R1 leaves off: photoperiod (IDSL 1),
vernalisation (IDSL 2), start at emergence, fertiliser dates.
"""
import copy
import datetime
import os
import _paths  # noqa: E402
import subprocess

import numpy as np
import torch

from torchvic.crop.wofost_npk import (CropTemplates, UnitSetup, Calendar,
                                      WofostNPK, SCALARS, TABLES,
                                      MNG_SCALARS, MNG_TABLES)

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, 'oracle')
R1 = _paths.data('vicwur/R1')
VICWUR_SRC = _paths.env_path(('TORCHVIC_VICWUR_SRC', 'VICWUR_SRC'), 'build/vicwur3/src_tree/vic')
OPTS = WofostNPK.OPTIONS


def build_oracle(outdir):
    """Compile the float and double oracles into outdir; returns
    {'f32': path, 'f64': path} or raises RuntimeError."""
    env = dict(os.environ, TORCHVIC_ORACLE_DIR=str(outdir),
               VICWUR_SRC=VICWUR_SRC)
    r = subprocess.run(['bash', os.path.join(ORACLE,
                                             'build_ref_vicwur_wofost.sh')],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return {'f32': os.path.join(str(outdir), 'ref_vicwur_wofost'),
            'f64': os.path.join(str(outdir), 'ref_vicwur_wofost_f64')}


def r1_templates():
    from torchvic.vicwur.params import read_wofost_text
    wt = read_wofost_text(os.path.join(R1, 'wofost_text.txt'), 2)
    return CropTemplates.from_wofost_text(wt)


def variant_templates():
    """4 classes: R1 maize, R1 rice, maize with vernalisation (IDSL 2) that
    starts at emergence, rice with photoperiod (IDSL 1) and fertiliser
    dates."""
    base = r1_templates()
    sc = {k: list(v) + [v[0], v[1]] for k, v in base.scalars.items()}
    tb = {k: list(v) + [copy.deepcopy(v[0]), copy.deepcopy(v[1])]
          for k, v in base.tables.items()}
    mg = {k: list(v) + [v[0], v[1]] for k, v in base.mng.items()}
    mt = {k: list(v) + [copy.deepcopy(v[0]), copy.deepcopy(v[1])]
          for k, v in base.mng_tables.items()}
    f = lambda v: float(np.float32(v))
    # class 2: vernalisation, daylength, start at emergence
    sc['IdentifyAnthesis'][2] = 2.0
    sc['SatVernRequirement'][2] = f(30.0)
    sc['BaseVernRequirement'][2] = f(8.0)
    sc['OptimumDaylength'][2] = f(16.0)
    sc['CriticalDaylength'][2] = f(8.0)
    tb['VernalizationRate'][2] = ([f(-8), f(-4), f(3), f(10), f(17),
                                   f(20)], [f(0), f(0), f(1), f(1), f(0),
                                            f(0)])
    # class 3: photoperiod, fertiliser on fixed dates, low mineralisation
    sc['IdentifyAnthesis'][3] = 1.0
    sc['OptimumDaylength'][3] = f(11.0)
    sc['CriticalDaylength'][3] = f(13.5)
    sc['N_fixation'][3] = f(0.2)
    mg['N_Mins'][3] = f(1.0)
    mg['P_Mins'][3] = f(0.2)
    mg['K_Mins'][3] = f(0.5)
    mt['N_Fert_table'][3] = ([6, 7, 7, 8], [20, 5, 5, 1],
                             [f(30), f(15), f(5), f(10)])
    mt['P_Fert_table'][3] = ([6, 8], [21, 1], [f(8), f(4)])
    mt['K_Fert_table'][3] = ([7], [5], [f(12)])
    em = list(base.emergence) + [1, 0]
    st = list(base.start) + [base.start[0], base.start[1]]
    return CropTemplates(sc, tb, mg, mt, em, st)


def make_case(seed, tmpl, ndays=800, start=(2003, 1, 1), nf=2):
    """Synthetic cells, bands, season dates and daily forcing."""
    g = np.random.default_rng(seed)
    N, B, NC = 4, 2, tmpl.nc
    lat = np.array([12.0, 25.5, -31.0, 48.0])[:N]
    af = np.array([[0.6, 0.4], [1.0, 0.0], [0.3, 0.7], [0.5, 0.5]])[:N]
    tf = g.uniform(-1.5, 1.5, (N, B))
    tf[:, 0] = g.uniform(-0.5, 0.5, N)
    active = np.ones((N, NC), bool)
    active[1, 1] = active[1, 3] = False
    active[2, 0] = False
    cv = g.uniform(0.1, 0.5, (N, NC))
    cc = g.uniform(0.05, 0.5, (N, NC, 12))
    sdoy = g.integers(20, 300, (N, NC))
    length = g.integers(100, 200, (N, NC))
    edoy = (sdoy + length - 1) % 365 + 1
    from torchvic.crop.wofost_npk import dmy_doy
    sm = np.zeros((N, NC), np.int64)
    sd = np.zeros((N, NC), np.int64)
    for i in range(N):
        for k in range(NC):
            sm[i, k], sd[i, k], _ = dmy_doy(sdoy[i, k])
    ts1 = g.uniform(600, 1100, (N, NC))
    ts2 = g.uniform(500, 1000, (N, NC))
    fert = np.zeros((N, NC, nf, 4))
    fert[..., 0] = np.sort(g.uniform(0.0, 1.4, (N, NC, nf)), -1)
    fert[..., 0][:, :, 0] = np.where(g.random((N, NC)) < 0.5, 0.0,
                                     fert[..., 0][:, :, 0])
    fert[..., 1] = g.uniform(10, 60, (N, NC, nf))
    fert[..., 2] = g.uniform(2, 20, (N, NC, nf))
    fert[..., 3] = g.uniform(2, 20, (N, NC, nf))
    mins = np.zeros((N, NC, 6))
    for j, nm in enumerate(('N_Mins', 'NRecoveryFrac', 'P_Mins',
                            'PRecoveryFrac', 'K_Mins', 'KRecoveryFrac')):
        mins[:, :, j] = np.array(tmpl.mng[nm], np.float32)[None, :]
    mins[:, :, 0] += np.float32(g.uniform(0, 30, (N, NC))).astype(np.float64)
    mins = mins.astype(np.float32).astype(np.float64)
    setup = dict(lat=lat, ph=g.uniform(4.0, 8.0, N),
                 carbon=g.uniform(5, 25, N), area_fract=af, tfactor=tf,
                 active=active, cv=cv, cc=cc, start_month=sm, start_day=sd,
                 start_doy=sdoy, end_doy=edoy, tsum1=ts1, tsum2=ts2,
                 fert=fert, mins=mins)
    d0 = datetime.date(*start)
    dates = []
    for t in range(ndays):
        d = d0 + datetime.timedelta(days=t)
        dates.append((d.year, d.month, d.day, d.timetuple().tm_yday))
    doy = np.array([x[3] for x in dates], np.float64)
    seas = np.cos(2 * np.pi * (doy - 200) / 365.)
    hemi = np.sign(lat)[None, :]
    amp = np.array([4.0, 8.0, 9.0, 12.0])[:N][None, :]
    mean = np.array([26.0, 22.0, 15.0, 8.0])[:N][None, :]
    air = mean + amp * seas[:, None] * hemi + g.normal(0, 2.5, (ndays, N))
    sw = np.clip(200 + 90 * seas[:, None] * hemi +
                 g.normal(0, 40, (ndays, N)), 5, 400)
    co2 = 370 + 0.01 * np.arange(ndays)[:, None] + g.normal(0, 3, (ndays, N))
    pet = g.uniform(0.5, 8.0, (ndays, N, B, NC))
    ws = np.clip(g.beta(5, 1.2, (ndays, N, B, NC)), 0, 1)
    ws[g.random(ws.shape) < 0.03] = 0.0
    ws[g.random(ws.shape) < 0.2] = 1.0
    f_ts = np.stack([g.uniform(600, 1100, (ndays, N, NC)),
                     g.uniform(500, 1000, (ndays, N, NC))], -1)
    f_fert = np.zeros((ndays, N, NC, nf, 4))
    f_fert[..., 0] = np.sort(g.uniform(0, 1.5, (ndays, N, NC, nf)), -1)
    f_fert[..., 1:] = g.uniform(1, 40, (ndays, N, NC, nf, 3))
    forc = dict(air=air, sw=sw, co2=co2, pet=pet, ws=ws, f_ts=f_ts,
                f_fert=f_fert)
    return setup, dates, forc


def write_season(path, tmpl, setup, opts, dates, forc, leafevery=0):
    """Input of mode 1, in the read order of season_run()."""
    S = setup
    N, B = S['area_fract'].shape
    NC = tmpl.nc
    NF = S['fert'].shape[2]
    T = len(dates)
    P = max(len(x) for nm in TABLES for x, _ in tmpl.tables[nm])
    P = max(P, 1)
    PM = max(len(r[0]) for nm in MNG_TABLES for r in tmpl.mng_tables[nm])
    PM = max(PM, 1)
    v = [1, N, B, NC, T, NF, P, PM, leafevery]
    v += [int(bool(opts.get(k, False))) for k in
          ('PIRR', 'PFERT', 'PTEMP', 'DIST_TSUM', 'DIST_FERT', 'CALC_MIN',
           'CONTINUE', 'FORCE_TSUM', 'FORCE_FERT')]
    v += [0]                                   # calendar standard
    for k in range(NC):
        v += [tmpl.scalars[nm][k] for nm in SCALARS]
        for nm in TABLES:
            x, y = tmpl.tables[nm][k]
            v += [len(x)] + list(x) + [0.0] * (P - len(x)) + \
                list(y) + [0.0] * (P - len(y))
        v += [tmpl.mng[nm][k] for nm in MNG_SCALARS]
        for nm in MNG_TABLES:
            m, d, a = tmpl.mng_tables[nm][k]
            pad = [0.0] * (PM - len(m))
            v += [len(m)] + list(m) + pad + list(d) + pad + list(a) + pad
        v += [tmpl.emergence[k]]
    for i in range(N):
        v += [S['lat'][i], S['ph'][i], S['carbon'][i]]
        v += list(S['area_fract'][i]) + list(S['tfactor'][i])
        for k in range(NC):
            v += [int(S['active'][i, k]), S['cv'][i, k]]
            v += list(S['cc'][i, k])
            v += [S['start_month'][i, k], S['start_day'][i, k],
                  S['start_doy'][i, k], S['end_doy'][i, k],
                  S['tsum1'][i, k], S['tsum2'][i, k]]
            for f in range(NF):
                v += list(S['fert'][i, k, f])
            v += list(S['mins'][i, k])
    head = np.array(v, np.float64)
    rows = []
    for t in range(T):
        r = list(dates[t])
        for i in range(N):
            r += [forc['air'][t, i], forc['sw'][t, i], forc['co2'][t, i]]
            for b in range(B):
                for k in range(NC):
                    r += [forc['pet'][t, i, b, k], forc['ws'][t, i, b, k]]
            for k in range(NC):
                r += list(forc['f_ts'][t, i, k])
                for f in range(NF):
                    r += list(forc['f_fert'][t, i, k, f])
        rows.append(np.array(r, np.float64))
    with open(path, 'wb') as fh:
        head.tofile(fh)
        np.concatenate(rows).tofile(fh)


def run_oracle(exe, inp, out):
    subprocess.run([exe, inp, out], check=True)
    names = open(out + '.names').read().split()
    return names


def read_season(out, names, T, N, B, NC):
    a = np.fromfile(out, np.float64)
    a = a.reshape(T, N, B, NC, len(names))
    return {nm: a[..., j] for j, nm in enumerate(names)}


def read_leaves(out):
    a = np.fromfile(out + '.leaves', np.float64)
    recs = {}
    p = 0
    while p < len(a):
        t, u, n = int(a[p]), int(a[p + 1]), int(a[p + 2])
        recs[(t, u)] = a[p + 3:p + 3 + 3 * n].reshape(n, 3)
        p += 3 + 3 * n
    return recs


def torch_season(tmpl, setup, opts, dates, forc, dtype, device='cpu',
                 c_promote=True, leafevery=0, exact_sums=False):
    """Run the kernel on the same case; returns (fields dict [T,N,B,NC],
    leaf lists {(t, unit): [n,3]}, model, final state)."""
    su = UnitSetup(continue_=opts.get('CONTINUE', False), **setup)
    m = WofostNPK(tmpl, su, opts, dtype=dtype, c_promote=c_promote,
                  device=device, exact_sums=exact_sums)
    cal = Calendar(dates, tmpl, device=device, dtype=dtype)
    s = m.init_state()
    T = len(dates)
    N, B, NC = m.shape
    tt = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    out = {}
    leaves = {}
    for t in range(T):
        forced = dict(tsum1=tt(forc['f_ts'][t, ..., 0]),
                      tsum2=tt(forc['f_ts'][t, ..., 1]),
                      dvs=tt(forc['f_fert'][t, ..., 0]),
                      N=tt(forc['f_fert'][t, ..., 1]),
                      P=tt(forc['f_fert'][t, ..., 2]),
                      K=tt(forc['f_fert'][t, ..., 3]))
        s = m.update_step_vars(s, cal, t, forced)
        s = m.crop_run(s, cal, t, tt(forc['air'][t]), tt(forc['sw'][t]),
                       tt(forc['co2'][t]), tt(forc['pet'][t]),
                       tt(forc['ws'][t]))
        rec = fields(m, s)
        for k, v in rec.items():
            out.setdefault(k, []).append(v)
        if leafevery and t % leafevery == leafevery - 1:
            la_ = m.leaf_arrays(s)
            lw, la, lg = (la_[k].double().cpu().numpy() for k in
                          ('lw', 'la', 'lage'))
            lh, lt = s['lh'].cpu().numpy(), s['lt'].cpu().numpy()
            for i in range(N):
                for b in range(B):
                    for k in range(NC):
                        u = (i * B + b) * NC + k
                        sl = slice(lh[i, b, k], lt[i, b, k])
                        leaves[(t, u)] = np.stack(
                            [lw[i, b, k, sl], lg[i, b, k, sl],
                             la[i, b, k, sl]], -1)
    return {k: np.stack(v) for k, v in out.items()}, leaves, m, s


def fields(m, s):
    """The oracle's dump fields from a state dict ([N,B,NC] numpy)."""
    N, B, NC = m.shape
    rec = {}
    for k, v in s.items():
        if k in ('lw', 'la', 'lage', 'lh', 'lt', 'err', 'met_Tmin') or \
                k.startswith('con_') or k.startswith('mng_') and k[4] in \
                'NPK' and k[5] != '_':
            continue
        rec[k] = v.double().cpu().numpy()
    for j in range(7):
        rec['met_Tmin%d' % j] = s['met_Tmin'][..., j].double().cpu().numpy()
    leaf = m.leaf_arrays(s)
    live = m._live(s['lh'].long(), s['lt'].long())
    lw = torch.where(live, leaf['lw'].double(), torch.zeros_like(
        leaf['lw'], dtype=torch.float64))
    rec['leaf_n'] = (s['lt'] - s['lh']).double().cpu().numpy()
    rec['leaf_sumw'] = lw.sum(-1).cpu().numpy()
    prod = torch.where(live, (leaf['lw'] * leaf['la']).double(),
                       torch.zeros_like(lw))
    rec['leaf_sumwa'] = prod.sum(-1).cpu().numpy()
    ha = leaf['lage'].gather(-1, s['lh'].long().clamp(0, m.K - 1)[..., None]
                          )[..., 0].double()
    rec['leaf_headage'] = torch.where(s['lt'] > s['lh'], ha,
                                      torch.full_like(ha, np.nan)
                                      ).cpu().numpy()
    for k in ('tsum1', 'tsum2'):
        rec['con_' + k] = s['con_' + k][:, None, :].expand(
            N, B, NC).double().cpu().numpy()
    rec['mng_N_Mins'] = s['mng_N_Mins'].double().cpu().numpy()
    rec['mng_NRecoveryFrac'] = s['mng_NRecoveryFrac'].double().cpu().numpy()
    return rec


def compare(ref, got, active, names, skip=()):
    """Per field: max |diff| / max(|ref|) over active units and days.
    Returns dict name -> (scaled error, max abs diff, argmax)."""
    res = {}
    act = np.broadcast_to(active[None, :, None, :], ref[names[0]].shape)
    for nm in names:
        if nm in skip:
            continue
        r = ref[nm][act]
        g = got[nm][act]
        both_nan = np.isnan(r) & np.isnan(g)
        d = np.where(both_nan, 0.0, np.abs(r - g))
        d = np.where(np.isnan(d), np.inf, d)
        scale = np.nanmax(np.abs(r)) if np.isfinite(r).any() else 1.0
        scale = max(scale, 1e-30)
        j = int(np.argmax(d)) if d.size else 0
        res[nm] = (float(d.max() / scale) if d.size else 0.0,
                   float(d.max()) if d.size else 0.0, j)
    return res


# ------------------------------------------------------------ mode 2
def function_template():
    """R1 maize with an unordered CO2TRATB (not read by the kernel) to
    test the C Afgen on a non-ascending table."""
    base = r1_templates()
    one = lambda d: {k: [v[0]] for k, v in d.items()}
    sc, mg = one(base.scalars), one(base.mng)
    tb = {k: [copy.deepcopy(v[0])] for k, v in base.tables.items()}
    mt = {k: [copy.deepcopy(v[0])] for k, v in base.mng_tables.items()}
    f = lambda v: float(np.float32(v))
    tb['CO2TRATB'] = [([f(0), f(1), f(3), f(2), f(5), f(0)],
                       [f(1), f(2), f(0.5), f(4), f(3), f(9)])]
    return CropTemplates(sc, tb, mg, mt, [0], [base.start[0]])


def make_function_records(seed, tmpl, n=400):
    """Random records of the five kinds (see function_run())."""
    g = np.random.default_rng(seed)
    f = lambda v: float(np.float32(v))
    recs = []
    for nm_i, nm in enumerate(TABLES):
        x, _ = tmpl.tables[nm][0]
        if not x:
            continue
        pts = list(x) + [x[0] - 1.0, x[-1] + 1.0] + \
            list(g.uniform(min(x) - 0.5, max(x) + 0.5, 6))
        for v in pts:
            recs.append((0, [nm_i, f(v)]))
    for _ in range(n):
        tmin = list(g.uniform(-5, 30, 7))
        recs.append((1, [int(g.integers(1, 366)), g.uniform(-60, 60),
                         g.uniform(1e6, 3e7), g.uniform(-5, 40),
                         g.uniform(250, 800), g.uniform(0, 2),
                         g.choice([0.0, g.uniform(0, 6)]),
                         int(g.integers(0, 12))] + tmin))
        recs.append((2, [g.choice([0.0, 1.0, g.uniform(0, 1)]),
                         g.uniform(0, 1), g.uniform(0, 2)]))
        k = int(g.integers(1, 12))
        wts = list(g.uniform(0, 3, k) * (g.random(k) < 0.8))
        ages = sorted(g.uniform(0, 60, k), reverse=True)
        areas = list(g.uniform(0.001, 0.004, k))
        tot = sum(wts)
        lv = f(tot)
        cls = []
        for j in range(k):
            cls += [wts[j], ages[j], areas[j]]
        recs.append((3, [k, lv, g.uniform(0, 8), g.uniform(0, 2),
                         g.choice([1.0, g.uniform(0, 1)]),
                         g.uniform(0, 1)] + cls))
        recs.append((4, [g.uniform(0, 2), g.uniform(0, 3000),
                         g.uniform(0, 3000)] +
                     list(g.uniform(0, 80, 6)) + list(g.uniform(0, 2, 6))))
    # DyingLeaves edge: Death exactly zero, a single class
    recs.append((3, [1, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, 10.0, 0.002]))
    return recs


def write_functions(path, tmpl, recs, ptemp=False):
    P = max(1, max(len(x) for nm in TABLES for x, _ in tmpl.tables[nm]))
    v = [2, len(recs), P]
    v += [tmpl.scalars[nm][0] for nm in SCALARS]
    for nm in TABLES:
        x, y = tmpl.tables[nm][0]
        v += [len(x)] + list(x) + [0.0] * (P - len(x)) + list(y) + \
            [0.0] * (P - len(y))
    v += [tmpl.mng[nm][0] for nm in MNG_SCALARS]
    v += [int(ptemp)]
    for kind, vals in recs:
        v += [kind] + list(vals)
    np.array(v, np.float64).tofile(path)


def read_functions(path):
    raw = open(path, 'rb').read()
    out = []
    p = 0
    while p < len(raw):
        n = int(np.frombuffer(raw[p:p + 4], np.int32)[0])
        p += 4
        out.append(np.frombuffer(raw[p:p + 8 * n], np.float64).copy())
        p += 8 * n
    return out


def torch_functions(tmpl, recs, dtype, exact_sums=False, ptemp=False):
    """The kernel's functions on the records, one record per cell."""
    out = []
    R = len(recs)
    lat = np.zeros(R)
    for j, (kind, vals) in enumerate(recs):
        if kind == 1:
            lat[j] = vals[1]
    su = UnitSetup(lat=lat, ph=np.zeros(R), carbon=np.zeros(R),
                   area_fract=np.ones((R, 1)), tfactor=np.zeros((R, 1)),
                   active=np.ones((R, 1), bool), cv=np.ones((R, 1)),
                   cc=np.ones((R, 1, 12)), start_month=np.ones((R, 1), int),
                   start_day=np.ones((R, 1), int),
                   start_doy=np.zeros((R, 1), int),
                   end_doy=np.zeros((R, 1), int), tsum1=np.ones((R, 1)),
                   tsum2=np.ones((R, 1)), fert=np.zeros((R, 1, 1, 4)),
                   mins=np.zeros((R, 1, 6)))
    m = WofostNPK(tmpl, su, {'PTEMP': ptemp}, dtype=dtype,
                  exact_sums=exact_sums,
                  nleaf=1 + max([v[0] for k, v in recs if k == 3] + [1]))
    base = m.init_state()
    T = lambda a: torch.as_tensor(np.asarray(a, np.float64)).to(dtype
                                                                ).reshape(
        R, 1, 1)
    res = [None] * R
    kinds = np.array([k for k, _ in recs])

    def col(kind, j, default=0.0):
        return [vals[j] if k == kind else default for k, vals in recs]

    # kind 0
    for j in np.nonzero(kinds == 0)[0]:
        ti, x = recs[j][1]
        xv = torch.full((R, 1, 1), x, dtype=torch.float64).to(dtype)
        res[j] = [m.afgen(TABLES[int(ti)], xv)[j, 0, 0].item()]
    # kind 1
    w = dict(base)
    w['met_MeteoDay'] = torch.as_tensor(col(1, 0, 1), dtype=torch.int32
                                        ).reshape(R, 1, 1)
    w['met_Radiation'] = T(col(1, 2))
    w['met_DayTemp'] = T(col(1, 3))
    w['met_CO2'] = T(col(1, 4))
    w['st_Development'] = T(col(1, 5))
    w['st_LAI'] = T(col(1, 6))
    w['GrowthDay'] = torch.as_tensor(col(1, 7), dtype=torch.int32
                                     ).reshape(R, 1, 1)
    tm = np.array([[vals[8 + c] if k == 1 else 0.0 for c in range(7)]
                   for k, vals in recs])
    w['met_Tmin'] = torch.as_tensor(tm).to(m.wd).reshape(R, 1, 1, 7)
    m._astro(w)
    a = m._daily_total_assimilation(w)
    cor = m._correct(w, a)
    for j in np.nonzero(kinds == 1)[0]:
        res[j] = [w[k][j, 0, 0].item() for k in (
            'met_Daylength', 'met_PARDaylength', 'met_SinLD', 'met_CosLD',
            'met_DSinBE', 'met_AngotRadiation', 'met_AtmosphTransm',
            'met_DiffRadPP')] + [a[j, 0, 0].item(),
                                 w['met_TempStress'][j, 0, 0].item(),
                                 cor[j, 0, 0].item(),
                                 w['met_TminStress'][j, 0, 0].item()]
    # kind 2
    w = dict(base)
    w['soil_WaterStress'] = T(col(2, 0))
    w['N_st_Indx'] = T(col(2, 1))
    w['st_Development'] = T(col(2, 2))
    m._partioning(w)
    for j in np.nonzero(kinds == 2)[0]:
        res[j] = [w[k][j, 0, 0].item() for k in ('fac_ro', 'fac_lv',
                                                 'fac_st', 'fac_so')]
    # kind 3
    w = dict(base)
    K = m.K
    lw = np.zeros((R, K))
    lg = np.zeros((R, K))
    la = np.zeros((R, K))
    nn = np.zeros(R, np.int64)
    for j, (k, vals) in enumerate(recs):
        if k != 3:
            continue
        n = int(vals[0])
        nn[j] = n
        c = np.array(vals[6:6 + 3 * n]).reshape(n, 3)
        lw[j, :n], lg[j, :n], la[j, :n] = c[:, 0], c[:, 1], c[:, 2]
    w['lw'] = torch.as_tensor(lw).to(dtype).reshape(R, 1, 1, K)
    w['lage'] = torch.as_tensor(lg).to(dtype).reshape(R, 1, 1, K)
    w['la'] = torch.as_tensor(la).to(dtype).reshape(R, 1, 1, K)
    w['lh'] = torch.zeros(R, 1, 1, dtype=torch.int32)
    w['lt'] = torch.as_tensor(nn, dtype=torch.int32).reshape(R, 1, 1)
    w['st_leaves'] = T(col(3, 1))
    w['st_LAI'] = T(col(3, 2))
    w['st_Development'] = T(col(3, 3))
    w['soil_WaterStress'] = T(col(3, 4))
    w['NPK_Indx'] = T(col(3, 5))
    d = m._dying_leaves(w)
    hw = w['lw'].gather(-1, w['lh'].long().clamp(0, K - 1)[..., None]
                        )[..., 0]
    for j in np.nonzero(kinds == 3)[0]:
        cnt = int(w['lt'][j, 0, 0] - w['lh'][j, 0, 0])
        res[j] = [d[j, 0, 0].item(), cnt,
                  hw[j, 0, 0].item() if cnt > 0 else np.nan]
    # kind 4
    w = dict(base)
    w['st_Development'] = T(col(4, 0))
    w['st_leaves'] = T(col(4, 1))
    w['st_stems'] = T(col(4, 2))
    for i, X in enumerate('NPK'):
        w['%s_st_leaves' % X] = T(col(4, 3 + 2 * i))
        w['%s_st_stems' % X] = T(col(4, 4 + 2 * i))
        w['%s_rt_Uptake_lv' % X] = T(col(4, 9 + 2 * i))
        w['%s_rt_Uptake_st' % X] = T(col(4, 10 + 2 * i))
    m._nutrient_max(w)
    m._nutrient_optimum(w)
    m._nutrition_indx(w)
    for j in np.nonzero(kinds == 4)[0]:
        res[j] = [w[k][j, 0, 0].item() for k in (
            'N_st_Indx', 'P_st_Indx', 'K_st_Indx', 'NPK_Indx',
            'NutrientStress')]
    return [np.array(r, np.float64) for r in res]


def dump_field_names():
    """Field names of the oracle's FIELDS macro (also the crop record of
    the crop-dump build, crops/src/crop_dump.c), in order."""
    import re
    src = open(os.path.join(ORACLE, 'ref_vicwur_wofost.c')).read()
    st = re.findall(r'X\(P ## _st_(\w+),',
                    src.split('#define NUTRIENT_ST')[1].split('#define')[0])
    rt = re.findall(r'X\(P ## _rt_(\w+),',
                    src.split('#define NUTRIENT_RT')[1].split('#define')[0])
    body = src.split('#define FIELDS(X)')[1].split('#define X_COUNT')[0]
    names = []
    for m in re.finditer(r'NUTRIENT_(ST|RT)\(X, (\w), |X\((\w+),', body):
        if m.group(3):
            names.append(m.group(3))
        else:
            lst = st if m.group(1) == 'ST' else rt
            tag = '_st_' if m.group(1) == 'ST' else '_rt_'
            names += [m.group(2) + tag + f for f in lst]
    return names
