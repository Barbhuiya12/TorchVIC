"""CropSyst E4 (potential transpiration, CO2), E5 (RUE/TUE growth), E7 (canopy), E8 (roots),
E9 (yield), driven with the E6 phenology over whole seasons.

Oracle comparison. tests/oracle/ref_cropsyst_growth.cpp drives the compiled VIC-CropSyst
reference objects (thermal time, Phenology_2013, Crop_CO2_response, Biomass_growth_RUE_TUE with
the V4 transpiration-dependent growth, Canopy_growth_leaf_area_index_based_V5, Crop_root_vital_V5
on a VIC::Soil_layers profile, Yield_grain with its temperature-extreme response and the hourly
stress-adjusted temperature estimator) day by day in the Crop_complete order (transcribed, see
the oracle header). E1 is replaced on both sides by act = uptake_frac * limited potential
transpiration. Every day's outputs are compared with torchvic.crop.cropsyst.daily, all cases
batched in one tensor run. Bar 1e-12 (relative, floor 1e-9), stages and flags exact.
tests/data/cropsyst_growth_l1.txt.gz is a frozen subset (`--write-golden`).

The oracle also prints the reference parser's view of each CRB crop file ("CRP" mode);
growth_params() must agree with it.
"""
import gzip
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import _paths  # noqa: E402

from torchvic.crop.cropsyst import daily as dl  # noqa: E402
from torchvic.crop.cropsyst import phenology as ph  # noqa: E402
from torchvic.crop.cropsyst import canopy as cn  # noqa: E402

ORACLE = _paths.oracle("ref_cropsyst_growth")
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "cropsyst_growth_l1.txt.gz")
TOL = 1e-12
FLOOR = 1e-9
NPHEN, NGROW, NROOT, NYLD, NCO2 = 20, 18, 7, 4, 6
NDAY = 12
SIM = _paths.env_path(('TORCHVIC_CROPSYST_SIM',), 'build/vic_cropsyst/VIC-CropSyst_v3_CRB/Simulation')
STAGES = (ph.NGS_EMERGENCE, ph.NGS_END_CANOPY_GROWTH, ph.NGS_ANTHESIS, ph.NGS_FILLING)


def _f32(x):
    return float(np.float32(x))


# ----------------------------------------------------------------------------- cases

def make_cases(n, seed=0):
    """Random and edge seasons; returns (head, layers, gsp, days) dicts per case."""
    rng = np.random.default_rng(seed)
    cases = []
    for k in range(n):
        edge = k % 10
        L = int(rng.choice([3, 5, 17]))
        N = int(rng.integers(180, 320))
        sc = rng.uniform(0.5, 0.9)
        base = _f32(rng.choice([0.0, 3.0, rng.uniform(-1, 5)]))
        cutoff, opt = _f32(base + rng.uniform(18, 26)), _f32(base + rng.uniform(20, 26))
        tmx = _f32(opt + rng.uniform(8, 16))
        emerg = _f32(rng.uniform(50, 120) * sc)
        peak = _f32(emerg + rng.uniform(500, 1100) * sc)
        flow = _f32(peak + rng.uniform(-80, 100) * sc)
        fill = _f32(flow + rng.uniform(50, 250) * sc)
        mat = _f32(fill + rng.uniform(400, 800) * sc)
        dur_acc = _f32(peak - emerg)
        dur_culm = _f32(max(fill - peak, 10.0))
        dur_sen = _f32(mat - fill)
        dur_root = _f32(rng.choice([1040.0, rng.uniform(300, 1300)]))
        v_en = int(rng.uniform() < 0.3)
        phen = [int(edge == 3), base, cutoff, opt, tmx, emerg, flow, fill, mat, dur_acc, dur_culm,
                dur_sen, dur_root, v_en, _f32(rng.uniform(0, 3)), _f32(rng.uniform(5, 9)), 0.0,
                _f32(rng.uniform(20, 50)), _f32(rng.choice([0.0, 0.5])), _f32(rng.choice([0.5, 1.0]))]
        init_gai = _f32(rng.uniform(0.01, 0.05))
        regrowth = _f32(init_gai + rng.uniform(0.005, 0.03)) if edge in (1, 6) else _f32(rng.uniform(0.005, init_gai))
        red = _f32(rng.uniform(-1300, -700))
        grow = [_f32(rng.uniform(3.5, 6)), _f32(rng.uniform(0.4, 0.6)), _f32(rng.choice([4.0, rng.uniform(1, 4)])),
                3.0, 10.0, 25.0, 45.0, _f32(rng.choice([1.0, 1.05, 1.1, 0.9, 0.0])), _f32(rng.uniform(0.45, 0.65)),
                red, _f32(red - rng.uniform(100, 400)), _f32(rng.uniform(0.5, 2)), init_gai, regrowth,
                _f32(rng.uniform(4, 7)), _f32(rng.uniform(15, 30)), _f32(rng.uniform(0.6, 0.9)),
                _f32(rng.uniform(2, 5))]
        if edge == 7:                                         # strong temperature limitation
            grow[3:7] = [5.0, 15.0, 18.0, 30.0]
        root = [_f32(rng.uniform(0.8, 1.8)), _f32(rng.uniform(2, 4)), _f32(rng.choice([0.01, 0.1, 1.0])),
                _f32(rng.uniform(0, 1)), _f32(rng.uniform(1, 3)), 2.0, 0.1]
        yl = [_f32(rng.uniform(0.3, 0.55)), int(edge != 4), int(edge != 5), _f32(rng.uniform(0.25, 0.45))]
        if edge == 8:
            yl[3] = _f32(yl[0] + 0.05)                             # sink HI below translocation
        co2 = [360.0, 360.0, 0.0032724901, 0.941574, 1.27713382,
               float(rng.choice([370.0, 450.0, 1200.0]))]
        gsps = []
        for stage in rng.choice(STAGES, size=int(rng.choice([0, 1, 2, 2, 3])), replace=False):
            hc = int(rng.choice([0, 23, 30])) if edge == 9 else 0
            hh = int(rng.choice([99, 24, 0, 6, 20]))
            dmg_c = _f32(rng.choice([5.0, 0.0])) if edge == 9 else 5.0
            gsps.append([int(stage), int(rng.integers(0, 3)),
                         5.0, -2.0, _f32(rng.choice([0.0, 1.0, 0.5])), hc, dmg_c,
                         _f32(rng.uniform(22, 30)), 40.0, 1.0, hh, _f32(rng.choice([6.0, 24.0, 4.0, 30.0]))])
        thick = [_f32(rng.uniform(0.05, 0.4)) for _ in range(L)]
        t0, amp, ph0 = rng.uniform(4, 14), rng.uniform(8, 14), rng.uniform(-1.0, 0.5)
        days = []
        for d in range(N):
            tx = t0 + amp * math.sin(ph0 + 2 * math.pi * d / 365.0) + rng.normal(0, 3) + 10
            tn = tx - rng.uniform(4, 16)
            u = rng.uniform()
            uf = 1.0 if u < 0.6 else (rng.uniform(0.2, 1.0) if u < 0.93 else (0.0 if u < 0.96 else 1.05))
            days.append([tx, tn, tn + rng.normal(0, 2), rng.uniform(2, 30), rng.uniform(0.05, 4),
                         rng.uniform(0.0005, 0.008), rng.uniform(0.05, 0.3), rng.uniform(0.06, 0.068),
                         rng.uniform(0.0005, 0.005), uf, rng.uniform(-1700, -300), int(rng.integers(4, 9))])
        cases.append(dict(N=N, L=L, co2_on=int(edge != 2), phen=phen, grow=grow, root=root, yl=yl,
                          co2=co2, gsp=gsps, thick=thick, days=days))
    return cases


def _fmt(c):
    tok = [str(c["N"]), str(c["L"]), str(c["co2_on"])]
    num = lambda v: v.hex() if isinstance(v, float) else str(int(v))
    tok += [num(float(v)) if i not in (0, 13) else str(int(v)) for i, v in enumerate(c["phen"])]
    tok += [num(float(v)) for v in c["grow"]]
    tok += [num(float(v)) for v in c["root"]]
    tok += [num(float(c["yl"][0])), str(c["yl"][1]), str(c["yl"][2]), num(float(c["yl"][3]))]
    tok += [num(float(v)) for v in c["co2"]]
    tok.append(str(len(c["gsp"])))
    for g in c["gsp"]:
        tok += [str(g[0]), str(g[1])] + [str(int(v)) if i in (3, 8) else num(float(v)) for i, v in enumerate(g[2:])]
    tok += [num(float(v)) for v in c["thick"]]
    for d in c["days"]:
        tok += [num(float(v)) for v in d[:11]] + [str(int(d[11]))]
    return " ".join(tok)


def _parse(line):
    return [float(int(x)) if x.lstrip("-").isdigit() else float.fromhex(x) for x in line.split()]


def run_oracle(cases):
    inp = "\n".join(_fmt(c) for c in cases) + "\n"
    r = subprocess.run([ORACLE], input=inp, capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = r.stdout.splitlines()
    out, i = [], 0
    for c in cases:
        days = [_parse(l) for l in lines[i:i + c["N"]]]
        assert lines[i + c["N"]].startswith("Y ")
        out.append((days, _parse(lines[i + c["N"]][2:])))
        i += c["N"] + 1
    assert i == len(lines)
    return out


# ----------------------------------------------------------------------------- torch side

def params_from_cases(cases, device="cpu"):
    T = lambda xs, dt=torch.float64: torch.tensor(xs, dtype=dt, device=device)
    n = len(cases)
    col = lambda key, j: _f32t(T([c[key][j] for c in cases]))   # float32 structs
    PP = ph.PhenologyParams(
        temps=torch.stack([col("phen", j) for j in (1, 2, 3, 4)], -1),
        nonlinear=T([c["phen"][0] for c in cases], torch.long) != 0,
        emergence=col("phen", 5), tuberization=torch.full((n,), 99999.0, dtype=torch.float64, device=device),
        flowering=col("phen", 6), filling=col("phen", 7), maturity=col("phen", 8),
        dur_accrescence=col("phen", 9), dur_culminescence=col("phen", 10), dur_senescence=col("phen", 11),
        dur_root_elongation=col("phen", 12),
        maturity_significant=torch.ones(n, dtype=torch.bool, device=device),
        has_leaf_senescence=torch.ones(n, dtype=torch.bool, device=device),
        is_tuber=torch.zeros(n, dtype=torch.bool, device=device),
        fruit_harvest=torch.zeros(n, dtype=torch.bool, device=device),
        culminate_senescence=torch.zeros(n, dtype=torch.bool, device=device),
        vern_enabled=T([c["phen"][13] for c in cases], torch.long) != 0,
        vern_low=col("phen", 14), vern_high=col("phen", 15), vern_start=col("phen", 16),
        vern_end=col("phen", 17), vern_min=col("phen", 18),
        photo_crop=torch.zeros(n, dtype=torch.bool, device=device),
        photo_phen=torch.zeros(n, dtype=torch.bool, device=device),
        photo_stop=torch.zeros(n, dtype=torch.float64, device=device),
        photo_unconstrained=torch.zeros(n, dtype=torch.float64, device=device),
        sens_water_stress=col("phen", 19),
        adj_stage=torch.full((n, 1), -1, dtype=torch.long, device=device),
        adj_offset=torch.zeros((n, 1), dtype=torch.long, device=device),
        adj_param=torch.full((n, 1), -1, dtype=torch.long, device=device),
        adj_value=torch.zeros((n, 1), dtype=torch.float64, device=device),
        is_fruit=torch.zeros(n, dtype=torch.bool, device=device),
        rapid_fruit=torch.zeros(n, dtype=torch.float64, device=device),
        grape=torch.zeros(n, dtype=torch.bool, device=device))
    G = max(1, max(len(c["gsp"]) for c in cases))
    st = torch.full((n, G), -1, dtype=torch.long)
    off = torch.zeros((n, G), dtype=torch.long)
    gsp = torch.tensor(dl.GSP_DEFAULTS, dtype=torch.float64).repeat(n, G, 1)
    for i, c in enumerate(cases):
        for j, g in enumerate(c["gsp"]):
            st[i, j], off[i, j] = g[0], g[1]
            gsp[i, j] = _f32t(torch.tensor(g[2:], dtype=torch.float64))
    g_ = lambda j: col("grow", j)
    r_ = lambda j: col("root", j)
    GP = dl.GrowthParams(
        tue_1kpa=g_(0), tue_scaling=g_(1), rue=g_(2), tl_min=g_(3), tl_low=g_(4), tl_high=g_(5), tl_max=g_(6),
        et_coef=g_(7), kc=g_(8), reduce_lwp=g_(9), stop_lwp=g_(10), max_height=g_(11), initial_gai=g_(12),
        regrowth_gai=g_(13), max_lai=g_(14), sla=g_(15), fract_lai_mature=g_(16), stem_leaf_partition=g_(17),
        root_max_depth=r_(0), root_surface_density=r_(1), root_curvature=r_(2), root_sensitivity=r_(3),
        root_length_emergence_cm=r_(4), root_shoot_emergence=r_(5), root_shoot_full=r_(6),
        hi_unstressed=col("yl", 0), hi_adjust=T([c["yl"][1] for c in cases], torch.long) != 0,
        hi_extreme=T([c["yl"][2] for c in cases], torch.long) != 0, translocation_max=col("yl", 3),
        hi_kind=torch.zeros(len(cases), dtype=torch.long, device=device),
        co2_tue_conc=col("co2", 0), co2_rue_conc=col("co2", 1), co2_alpha=_f32t(col("co2", 2)),
        co2_theta=_f32t(col("co2", 3)), co2_gra=_f32t(col("co2", 4)),
        gsp_stage=st.to(device), gsp_offset=off.to(device), gsp=gsp.to(device))
    return PP, GP


def _f32t(x):
    return x.to(torch.float32).to(torch.float64)


def torch_run(cases, device="cpu", P=400):
    """All cases batched (padded to a common L with zero-thickness layers is NOT valid, so cases
    are grouped by L); returns per case (days, Y) laid out like the oracle."""
    out = [None] * len(cases)
    for L in sorted({c["L"] for c in cases}):
        idx = [i for i, c in enumerate(cases) if c["L"] == L]
        sub = [cases[i] for i in idx]
        res = _torch_run_group(sub, device, P)
        for i, r in zip(idx, res):
            out[i] = r
    return out


def _torch_run_group(cases, device, P):
    PP, GP = params_from_cases(cases, device)
    n, L = len(cases), cases[0]["L"]
    G = GP.gsp_stage.shape[-1]
    Nmax = max(c["N"] for c in cases)
    W = torch.zeros((Nmax, n, NDAY), dtype=torch.float64)
    for i, c in enumerate(cases):
        W[:c["N"], i] = torch.tensor(c["days"], dtype=torch.float64)
    W = W.to(device)
    thick = torch.tensor([c["thick"] for c in cases], dtype=torch.float64, device=device)
    co2 = torch.tensor([c["co2"][5] for c in cases], dtype=torch.float64, device=device)
    co2_on = torch.tensor([bool(c["co2_on"]) for c in cases], device=device)
    s = ph.initial_state(PP)
    g = dl.initial_state((n,), L, G, P, device=device)
    ones = torch.ones(n, dtype=torch.bool, device=device)
    s = ph.sow(s, PP, ones, 0)
    g = dl.sow(g, ones, W[0, :, 0], W[0, :, 2])
    recs, harv = [], []
    for d in range(Nmax):
        w = W[d]
        s, g = dl.potential(s, g, PP, GP, d, w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4], w[:, 5], co2,
                            w[:, 6], w[:, 7], w[:, 8], thick, w[:, 11], co2_on=co2_on)
        act = w[:, 9] * g.limited_pot
        g = dl.actual(s, g, PP, GP, act, w[:, 10], w[:, 11])
        recs.append(_record(s, g, GP, L))
        harv.append(torch.stack(dl.harvest(g, GP), -1).cpu())     # harvest on each case's last day
    res = []
    for i, c in enumerate(cases):
        days = [[float(v) for v in recs[d][i]] for d in range(c["N"])]
        res.append((days, [float(v) for v in harv[c["N"] - 1][i]]))
    return res


def _record(s, g, GP, L):
    c, r, y = g.canopy, g.root, g.yld
    e = c.exists
    zf = lambda x: torch.where(e, x, torch.zeros_like(x))
    cols = [s.stage.to(torch.float64), s.gdds, s.gdd, s.tmax_adj,
            g.et_max, g.pot_unadj, g.pot_co2, g.transp_factor, g.co2_tue_adjustment, g.intercepted,
            g.attainable, g.limited_pot, g.act_transp, g.act_growth, g.gai_rel_growth, g.wcef, g.ws,
            g.ws_index, g.overall_stress, g.tue_co2_adjusted,
            e.to(torch.float64), zf(cn.gai_vital(c)), zf(cn.gai_vital_effete(c)), zf(cn.lai(c)), zf(c.lai_ss),
            zf(c.peak_lai), zf(c.cover_total), zf(cn.interception_green(c, GP.kc)), zf(cn.biomass_vital(c)),
            zf(cn.biomass_vital(c)), c.height, g.srf,
            r.exists.to(torch.float64), r.length, r.depth, r.biomass, r.growth_biomass]
    cols += [r.fract[..., l] for l in range(L)]
    from torchvic.crop.cropsyst import yield_ as yd
    cols += [y.bm_filling, y.bm_maturity, yd.extreme_factor(y)]
    cols += [y.hourly[..., h] for h in range(24)]
    return torch.stack(cols, -1).cpu()


EXACT = {0, 20}
# root growth biomass is a difference of root biomasses: relative to the root biomass
SCALE_OF = {36: 35}
# degree days and temperatures: floor 1 as in test_cropsyst_phenology (the nonlinear thermal
# time cancels near 0)
FLOOR_OF = {1: 1.0, 2: 1.0, 3: 1.0}


def compare(ref, got, days=None):
    worst, where = 0.0, None
    for ci, ((rd, ry), (gd, gy)) in enumerate(zip(ref, got)):
        pick = range(len(rd)) if days is None else days[ci]
        root_exists_col = 32
        for d in pick:
            r, gg = rd[d], gd[d]
            assert len(r) == len(gg), (ci, d, len(r), len(gg))
            for f in range(len(r)):
                if f in EXACT or f == root_exists_col:
                    assert r[f] == gg[f], (ci, d, f, r[f], gg[f])
                    continue
                if math.isnan(r[f]) or math.isnan(gg[f]):
                    assert math.isnan(r[f]) and math.isnan(gg[f]), (ci, d, f, r[f], gg[f])
                    continue
                scale = abs(r[SCALE_OF.get(f, f)])
                e = abs(r[f] - gg[f]) / max(abs(r[f]), scale, FLOOR_OF.get(f, FLOOR))
                if e > worst:
                    worst, where = e, (ci, d, f, r[f], gg[f])
        for f in range(3):
            e = abs(ry[f] - gy[f]) / max(abs(ry[f]), FLOOR)
            if e > worst:
                worst, where = e, (ci, "Y", f, ry[f], gy[f])
    return worst, where


def _cols(L):
    """Column indices of the per-day record for L layers."""
    f = 37 + L
    return dict(wcef=15, ws=16, gai=21, lai_ss=24, root_depth=34, fill=f, mat=f + 1, factor=f + 2)


@pytest.mark.reference
@pytest.mark.skipif(not os.path.exists(ORACLE), reason="ref_cropsyst_growth oracle not built")
def test_live_oracle_random_and_edge_cases():
    cases = make_cases(120, seed=6060)
    ref = run_oracle(cases)
    got = torch_run(cases)
    worst, where = compare(ref, got)
    assert worst <= TOL, (worst, where)
    # the branches were reached
    stages = {int(d[0]) for r in ref for d in r[0]}
    assert {ph.NGS_ACCRESCENCE, ph.NGS_ANTHESIS, ph.NGS_FILLING, ph.NGS_MATURITY} <= stages
    col = [_cols(c["L"]) for c in cases]
    assert any(0.0 < r[0][-1][k["factor"]] < 1.0 for r, k in zip(ref, col))      # heat/cold stress
    assert any(r[0][-1][k["factor"]] == 0.0 for r, k in zip(ref, col))           # 0-hour quirk
    assert any(0.0 < d[k["wcef"]] < 1.0 for r, k in zip(ref, col) for d in r[0])  # leaf expansion
    assert any(r[0][-1][k["gai"]] < 0.5 * max(d[k["gai"]] for d in r[0]) for r, k in zip(ref, col))
    assert any(r[1][2] not in (0.0,) and r[1][2] != c["yl"][0] for r, c in zip(ref, cases))  # HI iterated


def _write_golden(n=10):
    import json
    cases = make_cases(n, seed=2026)
    ref = run_oracle(cases)
    days = [sorted(set(range(0, c["N"], 9)) | {c["N"] - 1}) for c in cases]
    out = [dict(days={str(d): r[0][d] for d in dd}, Y=r[1]) for r, dd in zip(ref, days)]
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with gzip.open(GOLDEN, "wt") as f:
        json.dump(dict(cases=cases, out=out), f, default=float)
    print(f"wrote {n} cases to {GOLDEN}")


def _golden():
    import json
    with gzip.open(GOLDEN, "rt") as f:
        g = json.load(f)
    cases = g["cases"]
    days = [sorted(int(k) for k in o["days"]) for o in g["out"]]
    ref = [([o["days"].get(str(d)) for d in range(c["N"])], o["Y"]) for c, o in zip(cases, g["out"])]
    return cases, days, ref


def test_golden_cases_match_reference():
    cases, days, ref = _golden()
    got = torch_run(cases)
    worst, where = compare(ref, got, days)
    assert worst <= TOL, (worst, where)


def test_params_from_crop_files():
    """growth_params() on the CRB crop files equals the reference parser (oracle CRP mode)."""
    db = os.path.join(SIM, "Database", "Crop", "name")
    if not os.path.isdir(db) or not os.path.exists(ORACLE):
        pytest.skip("CRB crop files or oracle not present")
    from torchvic.crop.cropsyst import readers as rd
    names = ["wheat_winter-high", "wheat_winter-intermediate", "wheat_winter-low", "wheat_spring-high",
             "wheat_spring-intermediate", "pea_spring-high"]
    paths = [f"{db}/{n}.crp" for n in names]
    r = subprocess.run([ORACLE], input="".join(f"CRP {p}\n" for p in paths), capture_output=True,
                       text=True, timeout=600)
    rows = [_parse(l) for l in r.stdout.splitlines()]
    gp = dl.growth_params(rd.crop_table(paths))
    for i, row in enumerate(rows):
        exp = dict(tue_1kpa=row[1], tue_scaling=row[2], rue=row[3], tl_min=row[5], tl_low=row[6],
                   tl_high=row[7], tl_max=row[8], et_coef=row[9], kc=row[10], reduce_lwp=row[11],
                   stop_lwp=row[12], max_height=row[13], initial_gai=row[14], regrowth_gai=row[15],
                   max_lai=row[16], sla=row[17], fract_lai_mature=row[18], stem_leaf_partition=row[19],
                   root_max_depth=row[20], root_surface_density=row[21], root_curvature=row[22],
                   root_sensitivity=row[23], root_length_emergence_cm=row[24], root_shoot_emergence=row[25],
                   root_shoot_full=row[26], hi_unstressed=row[27], translocation_max=row[30],
                   co2_tue_conc=row[31], co2_rue_conc=row[32], co2_alpha=row[33], co2_theta=row[34],
                   co2_gra=row[35])
        for k, v in exp.items():
            assert float(getattr(gp, k)[i]) == v, (names[i], k, float(getattr(gp, k)[i]), v)
        assert row[0] == 0 and row[4] == 1.0                  # TUE model, no cultivation factor
        assert bool(gp.hi_adjust[i]) == bool(row[28]) and bool(gp.hi_extreme[i]) == bool(row[29])
        ng = int(row[36])
        for j in range(ng):
            assert gp.gsp[i, j].tolist() == row[37 + 10 * j: 47 + 10 * j], (names[i], j)
        assert row[37 + 10 * ng:] == [1.0, 0.0, 0.0]       # leaf senescence, grain, not established
    assert gp.gsp_stage[0].tolist() == [ph.NGS_END_CANOPY_GROWTH, ph.NGS_ANTHESIS]
    assert gp.gsp_offset[0].tolist() == [1, 1]


def _day(s, g, PP, GP, day, w, co2, thick, co2_on):
    s, g = dl.potential(s, g, PP, GP, day, w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4], w[:, 5], co2,
                        w[:, 6], w[:, 7], w[:, 8], thick, w[:, 11], co2_on=co2_on)
    g = dl.actual(s, g, PP, GP, w[:, 9] * g.limited_pot, w[:, 10], w[:, 11])
    return s, g


def _setup(cases, device, P=400):
    PP, GP = params_from_cases(cases, device)
    n, L = len(cases), cases[0]["L"]
    W = torch.tensor([c["days"][:150] for c in cases], dtype=torch.float64, device=device).transpose(0, 1).contiguous()
    thick = torch.tensor([c["thick"] for c in cases], dtype=torch.float64, device=device)
    co2 = torch.tensor([c["co2"][5] for c in cases], dtype=torch.float64, device=device)
    co2_on = torch.tensor([bool(c["co2_on"]) for c in cases], device=device)
    ones = torch.ones(n, dtype=torch.bool, device=device)
    s = ph.sow(ph.initial_state(PP), PP, ones, 0)
    g = dl.sow(dl.initial_state((n,), L, GP.gsp_stage.shape[-1], P, device=device), ones, W[0, :, 0], W[0, :, 2])
    return PP, GP, W, thick, co2, co2_on, s, g


def _flat(x):
    out = []
    for v in x:
        out += _flat(v) if isinstance(v, tuple) else [v]
    return out


def _assert_close(a, b):
    for x, y in zip(_flat(a), _flat(b)):
        if x.dtype.is_floating_point:
            assert torch.allclose(x, y, rtol=1e-12, atol=1e-15, equal_nan=True), (x, y)
        else:
            assert torch.equal(x, y)


def test_compiles_without_graph_breaks():
    cases = [c for c in make_cases(30, seed=17) if c["L"] == 5][:6]
    PP, GP, W, thick, co2, co2_on, s, g = _setup(cases, "cpu", P=160)
    comp = torch.compile(_day, fullgraph=True, dynamic=False)
    se, ge, sc, gc = s, g, s, g
    for d in range(40):
        day = torch.tensor(d)
        se, ge = _day(se, ge, PP, GP, day, W[d], co2, thick, co2_on)
        sc, gc = comp(sc, gc, PP, GP, day, W[d], co2, thick, co2_on)
    _assert_close((se, ge), (sc, gc))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_matches_reference_and_cuda_graph():
    cases, days, ref = _golden()
    got = torch_run(cases, device="cuda")
    worst, where = compare(ref, got, days)
    assert worst <= TOL, (worst, where)
    # CUDA-graph capture of one crop day over a large batch: no host syncs inside
    base = [c for c in make_cases(40, seed=77) if c["L"] == 17][:8]
    big = base * 64
    PP, GP, W, thick, co2, co2_on, s0, g0 = _setup(big, "cuda", P=160)
    day = torch.zeros((), dtype=torch.long, device="cuda")
    w_in = torch.zeros_like(W[0])
    se, ge = s0, g0
    for d in range(30):
        day.fill_(d); w_in.copy_(W[d])
        se, ge = _day(se, ge, PP, GP, day, w_in, co2, thick, co2_on)
    st0 = _flat((s0, g0))
    buf = [x.clone() for x in st0]
    rebuild = lambda flat: _unflat((s0, g0), flat)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    day.fill_(0); w_in.copy_(W[0])
    with torch.cuda.stream(stream):
        for _ in range(2):
            _day(*rebuild(buf), PP, GP, day, w_in, co2, thick, co2_on)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _flat(_day(*rebuild(buf), PP, GP, day, w_in, co2, thick, co2_on))
        for b, o in zip(buf, out):
            b.copy_(o)
    for b, x in zip(buf, st0):
        b.copy_(x)
    for d in range(30):
        day.fill_(d); w_in.copy_(W[d])
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(_flat((se, ge)), buf):
        if a.dtype.is_floating_point:
            assert bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
        else:
            assert torch.equal(a, b)


def _unflat(template, flat):
    it = iter(flat)

    def build(t):
        if isinstance(t, tuple):
            vals = [build(v) for v in t]
            return type(t)(*vals) if hasattr(t, "_fields") else tuple(vals)
        return next(it)
    return build(template)


if __name__ == "__main__":
    if "--write-golden" in sys.argv:
        _write_golden()


# ----------------------------------------------------------------------------- replay

L2_CASES = (330492, 308200, 328617)
L2_COLS = ("GAI", "Biomass_kg_m2", "Root_depth_mm", "Green_Canopy_Cover", "crop_evapotransp_max_mm",
           "Act_Transp_mm", "Yield_kg_m2")


L2_RUNS = _paths.env_path(('TORCHVIC_CROPSYST_L2_RUNS',), 'build/vic_cropsyst/crb_noN')   # run_nitrogen=false


def _l2_case(case, device="cpu", lwp=None, runs=None, trace=None):
    """Drive E6 + E4-E9 over all rotation tiles of a CRB case with the reference weather of the
    crop csv and the reference water uptake (the water stress factor 1 - WSI of the csv, i.e.
    act = (1 - WSI) * limited potential transpiration). Returns per column the list of
    (ref, got) pairs on crop days."""
    import csv
    import glob
    from collections import defaultdict
    from torchvic.crop.cropsyst import readers as rdr
    from torchvic.crop.cropsyst import yield_ as yd
    from torchvic.crop.cropsyst import uptake as up
    lwp_prev = None
    f = glob.glob(f"{runs or L2_RUNS}/Liu_Linux_{case}_*/VIC_output/vic_crop_daily*")
    if not f:
        pytest.skip(f"reference crop csv of {case} not present")
    rows = list(csv.reader(open(f[0])))
    H = {h: i for i, h in enumerate(rows[0])}
    data = defaultdict(list)
    for r in rows[1:]:
        data[r[H["CroppingSyst_code"]]].append(r)
    tiles = sorted(data)
    names = sorted({r[H["Crop_name"]] for r in rows[1:]} - {"NOCROP"})
    tab = rdr.crop_table([f"{SIM}/Database/Crop/name/{c}.crp" for c in names], device=device)
    PPc, GPc = ph.phenology_params(tab, device=device), dl.growth_params(tab, device=device)
    soil = open(glob.glob(f"{SIM}/Database/Soil/soil_param_id_{case}_*")[0]).read().split()
    thick = torch.tensor([float(x) for x in soil[78:95]], dtype=torch.float64, device=device)
    sx = _saxton([float(x) for x in soil[114:131]], [float(x) for x in soil[273:290]], thick)
    TPc = up.transpiration_params(tab)
    nd, T = len(data[tiles[0]]), len(tiles)
    col = lambda name, fn=float: [[fn(data[k][d][H[name]]) for k in tiles] for d in range(nd)]
    # ET max uses YESTERDAY's CropSyst reference ET (land_unit_sim.cpp:1301-1303: the ET
    # reference is updated after the crop's ET max; the csv column is printed at day end)
    crop, lab = col("Crop_name", str), col("Grow_Stage", str)
    tmax, tmin, wsi, rhmin = col("Tmax"), col("Tmin"), col("water_stress_index"), col("RHUM_min_%")
    sw, etref, doy = col("SWRAD_w_m2"), col("CropSyst_refET_mm"), col("DOY", int)
    refs = {c: col(c) for c in L2_COLS}
    vwc = [[[float(data[k][d][H[f"soil_liq_vwc[{l}].liq"]]) for l in range(17)] for k in tiles]
           for d in range(nd)]
    act_csv = col("Act_Transp_mm")
    lat, lon = float(rows[1][H["lat"]]), float(rows[1][H["lon"]])
    # full-precision state of the dump binary (crb_noN/vic_cropsyst_i6dump.exe): leaf water
    # potential, GAI, canopy biomass, root depth, act transpiration, latest yield
    dump = {}
    dpath = os.path.join(os.path.dirname(f[0]), "i6_dump.txt")
    if os.path.exists(dpath):
        for line in open(dpath):
            a = line.split()
            dump[(int(a[0]), int(a[1]), a[2])] = [float.fromhex(x) for x in a[3:]]
    year = col("Year", int)
    tens = lambda v: torch.tensor(v, dtype=torch.float64, device=device)
    cid = torch.zeros(T, dtype=torch.long, device=device)
    thickT = thick.expand(T, 17)
    s = g = None
    pairs = {c: [] for c in L2_COLS}
    for d in range(nd):
        sow = [crop[d][k] != "NOCROP" and (d == 0 or crop[d - 1][k] == "NOCROP") for k in range(T)]
        for k in range(T):
            if sow[k]:
                cid[k] = names.index(crop[d][k])
        PP, GP = ph.gather_params(PPc, cid), dl.gather_params(GPc, cid)
        sowm = torch.tensor(sow, device=device)
        tx, tn = tens(tmax[d]), tens(tmin[d])
        tn_tom = tens(tmin[d + 1] if d + 1 < nd else tmin[d])
        if s is None:
            s, g = ph.initial_state(PP), dl.initial_state((T,), 17, GP.gsp_stage.shape[-1], 400, device=device)
        s = ph.sow(s, PP, sowm, d)
        g = dl.sow(g, sowm, tx, tn_tom)
        vmax = dl.gr.vpd_max_from_rhmin(tx, tens(rhmin[d]))
        sr = yd.sunrise_hour(lat, lon, doy[d][0]).to(device) + torch.zeros(T, dtype=torch.float64, device=device)
        s, g = dl.potential(s, g, PP, GP, d, tx, tn, tn_tom, tens(sw[d]) * 0.0864, vmax, tens(etref[max(d - 1, 0)]) / 1000.0,
                            370.0, 0.1, 0.065, 0.002, thickT, sr)
        ws_ref = 1.0 - tens(wsi[d])
        act = ws_ref * g.limited_pot
        key = lambda k: (year[d][k], doy[d][k], tiles[k])
        if dump and lwp is None:   # the reference's own leaf water potential of the day
            lw = tens([dump.get(key(k), [0.0])[0] for k in range(T)])
            g = dl.actual(s, g, PP, GP, act, lw, sr)
        elif lwp is None:   # leaf water potential from E1 on the csv soil water
            lwp, lwp_prev = _l2_lwp(g, GP, PP, up.TranspirationParams(*[x[cid] for x in TPc]), sx,
                                    tens(vwc[max(d - 1, 0)]), thickT, ws_ref, tens(act_csv[d]) / 1000.0,
                                    lwp_prev if d else torch.zeros(T, dtype=torch.float64))
            g = dl.actual(s, g, PP, GP, act, lwp, sr)
            lwp = None
        else:
            g = dl.actual(s, g, PP, GP, act, lwp, sr)
        gbm0 = cn.gai_related_biomass(g.canopy)
        if trace is not None:
            trace.append(dict(d=d, tiles=tiles, gbm0=gbm0.cpu(), act_growth=g.act_growth.cpu(),
                              gai_rel=g.gai_rel_growth.cpu(), sla=GP.sla.cpu(), slp=GP.stem_leaf_partition.cpu()))
        harvest = torch.tensor([crop[d][k] != "NOCROP" and "harvest" in lab[d][k] for k in range(T)], device=device)
        bfh, yl, hi = dl.harvest(g, GP)
        got = {"GAI": cn.gai_vital(g.canopy), "Biomass_kg_m2": cn.biomass_vital(g.canopy),
               "Root_depth_mm": g.root.depth * 1000.0,
               "Green_Canopy_Cover": cn.interception_green(g.canopy, GP.kc),
               "crop_evapotransp_max_mm": g.et_max * 1000.0, "Act_Transp_mm": act * 1000.0,
               "Yield_kg_m2": torch.where(harvest, yl, torch.zeros_like(yl))}
        got = {k: v.cpu() for k, v in got.items()}
        full = {"GAI": 1, "Biomass_kg_m2": 2, "Root_depth_mm": 3, "Act_Transp_mm": 4, "Yield_kg_m2": 5}
        scale = {"Root_depth_mm": 1000.0, "Act_Transp_mm": 1000.0}
        for k in range(T):
            if crop[d][k] == "NOCROP":
                continue
            for c in L2_COLS:
                if c == "Yield_kg_m2" and not bool(harvest[k]):
                    continue
                if bool(harvest[k]) and c in ("GAI", "Biomass_kg_m2", "Green_Canopy_Cover"):
                    continue    # harvest removes the canopy before the output (biomass fate, engine.py)
                ref = refs[c][d][k]
                if dump and c in full and key(k) in dump:
                    ref = dump[key(k)][full[c]] * scale.get(c, 1.0)
                pairs[c].append((d, tiles[k], ref, float(got[c][k])))
        s = ph.terminate(s, harvest)
    return pairs


def _saxton(quartz, clay, depth):
    """Soil hydraulics the CRB reference derives from texture with CLAY_INPUT
    (read_soilparam.c:735-781): porosity, Campbell b and air entry; max moisture = porosity
    depth (bulk density 2650 (1 - porosity))."""
    q = torch.tensor(quartz, dtype=torch.float64) * 100.0
    c = torch.tensor(clay, dtype=torch.float64) * 100.0
    por = 0.332 - 0.0007251 * q + torch.log10(c) * 0.1276
    b = -(-3.14 - 0.00222 * c ** 2 - 0.00003484 * q ** 2 * c)
    A = 100.0 * torch.exp(-4.396 - 0.0715 * c - 0.000488 * q ** 2 - 0.00004285 * q ** 2 * c)
    ae = -A * torch.pow(por, -b)
    bulk = 2650.0 * (1.0 - por)
    max_moist = depth * (1.0 - bulk / 2650.0) * 1000.0
    return dict(b=b, ae=ae, max_moist=max_moist)


def _l2_lwp(g, GP, PP, TP, sx, vwc, thick, ws_ref, act_ref_m, lwp_prev):
    """Leaf water potential of the day from E1 (uptake.root_uptake) driven by the csv soil water,
    this port's root fractions and green cover, and the reference's limited potential
    transpiration act/(1-WSI) (the csv's act transpiration and WSI); the CO2 transpiration factor
    is rescaled accordingly (it multiplies the limited potential transpiration). Canopy-stored
    rain is not in the csv and is taken as 0."""
    from torchvic.crop.cropsyst import uptake as up
    psi = up.soil_water_potential(vwc * thick * 1000.0, torch.zeros_like(vwc), thick, sx["max_moist"],
                                  sx["ae"], sx["b"])
    lim = g.limited_pot
    pot = torch.where(ws_ref > 0.0, act_ref_m / torch.where(ws_ref > 0, ws_ref, torch.ones_like(ws_ref)), lim)
    tf = g.transp_factor * torch.where(lim > 0.0, pot / torch.where(lim > 0, lim, torch.ones_like(lim)),
                                       torch.ones_like(lim))
    green = cn.interception_green(g.canopy, GP.kc)
    r = up.root_uptake(pot, green, torch.zeros_like(pot), g.root.fract, psi, torch.full_like(psi, -33.0),
                       TP, lwp_prev, co2_transp_factor=tf, active=g.canopy.exists)
    return r.leaf_water_pot, r.leaf_water_pot


def l2_stats(pairs, floor=1e-6):
    out = {}
    for c, v in pairs.items():
        errs = [abs(r - x) / max(abs(r), floor) for _, _, r, x in v]
        out[c] = (len(v), max(errs) if errs else 0.0, sorted(errs)[len(errs) // 2] if errs else 0.0,
                  sum(e <= 5e-7 for e in errs))
    return out


def l2_seasons(pairs, gai_tol=1e-5):
    """Per crop season (tile, sowing day): days after emergence on which GAI, biomass and green
    cover agree within `gai_tol` before the first disagreement, and the worst ET max and root
    depth errors on those days. The CRB runs have the nitrogen model on; once nitrogen limits,
    the reference growth falls below the water-limited attainable growth, so the
    agreement ends there by construction."""
    from collections import defaultdict
    idx = defaultdict(list)
    for i, (d, t, r, x) in enumerate(pairs["GAI"]):
        idx[t].append(i)
    out = []
    for t, ii in idx.items():
        seasons, cur = [], [ii[0]]
        for a, b in zip(ii, ii[1:]):
            if pairs["GAI"][b][0] == pairs["GAI"][a][0] + 1:
                cur.append(b)
            else:
                seasons.append(cur)
                cur = [b]
        seasons.append(cur)
        for s in seasons:
            em = [i for i in s if pairs["GAI"][i][2] > 0]
            agree, et_err = 0, 0.0
            for i in em:
                ok = all(abs(pairs[c][i][2] - pairs[c][i][3]) <= gai_tol * max(abs(pairs[c][i][2]), 1e-9)
                         for c in ("GAI", "Biomass_kg_m2", "Green_Canopy_Cover"))
                if not ok:
                    break
                agree += 1
                r, x = pairs["crop_evapotransp_max_mm"][i][2:]
                et_err = max(et_err, abs(r - x) / max(abs(r), 1e-9))
            rd = max(abs(pairs["Root_depth_mm"][i][2] - pairs["Root_depth_mm"][i][3]) /
                     max(abs(pairs["Root_depth_mm"][i][2]), 1e-3) for i in s)
            out.append(dict(tile=t, sow=pairs["GAI"][s[0]][0], days=len(s), emerged=len(em),
                            agree=agree, et_err=et_err, root_err=rd))
    return out


@pytest.mark.reference
@pytest.mark.parametrize("case", L2_CASES)
def test_l2_crb_spot(case):
    """Whole seasons of the 3 CRB cases rerun with run_nitrogen=false (crb_noN; the dump binary adds
    the leaf water potential and full-precision state): GAI, canopy biomass, green cover, ET max,
    root depth and every harvest's yield to print precision (7-8 digit csv inputs)."""
    pairs = _l2_case(case)
    st = l2_stats(pairs)
    for c in ("GAI", "Biomass_kg_m2", "Green_Canopy_Cover", "crop_evapotransp_max_mm",
              "Root_depth_mm", "Yield_kg_m2"):
        n, worst, _, _ = st[c]
        assert n > 0 and worst < 1e-7, (c, st[c])
    assert st["Yield_kg_m2"][0] >= 4 and st["GAI"][0] > 1400
