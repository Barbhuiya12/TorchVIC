"""CropSyst E6 thermal time and phenology. Oracle comparison against the compiled reference,
replay spot check.

Oracle comparison. tests/oracle/ref_cropsyst_phenology.cpp drives the compiled VIC-CropSyst
reference objects (Thermal_time_daily_linear/_nonlinear with Vernalization and Photoperiodization,
Phenology_2013 with its Period_thermal periods, calc_stress_adjusted_temperature) day by day,
in the Crop_complete order (transcribed, see the oracle header). Each case is one season from
sowing with random or edge parameters and weather; every day's thermal time, stage, periods,
vernalization factors and stage dates are compared with torchvic.crop.cropsyst.phenology, all
cases batched in one tensor run. Bar: 1e-12 (relative with a floor, see SCALE), stages and
dates exact. tests/data/cropsyst_phenology_l1.txt holds a frozen subset (sampled days), so the
comparison also runs without the reference build; `python tests/test_cropsyst_phenology.py
--write-golden` rewrites it.

Replay. The daily weather of the three working CRB cases (330492_rot_3001, 308200_rot_3002,
328617_rot_3003; reference crop csv) drives the batched phenology over all rotation
tiles; Grow_Stage must match every crop-day and Accum_DD to the csv print precision. Sowing and
harvest days, the previous day's water stress index and RHmin (for VPD max) come from the csv:
they are management and coupling inputs, not E6.
"""
import csv
import glob
import gzip
import math
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import _paths  # noqa: E402

from torchvic.crop.cropsyst import phenology as ph  # noqa: E402
from torchvic.crop.cropsyst import growth as gr  # noqa: E402

ORACLE = _paths.oracle("ref_cropsyst_phenology")
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "cropsyst_phenology_l1.txt.gz")
TOL = 1e-12
NSCAL = 29          # scalar fields after N, before the adjustment count
NDAY = 6            # tmax tmin day_length wsi vpd srf
SIM = _paths.env_path(('TORCHVIC_CROPSYST_SIM',), 'build/vic_cropsyst/VIC-CropSyst_v3_CRB/Simulation')
L2_CASES = (330492, 308200, 328617)
STAGES = (1, 3, 5, 6, 7, 8, 11)          # NGS stages an adjustment can synchronise on


def _f32(x):
    return float(np.float32(x))


def _same(a, b):
    """Bitwise-equal tensors, NaN equal to NaN (adjusted temperatures can make the reference's
    nonlinear thermal time NaN, and start == end makes its vernalization factor NaN)."""
    if a.dtype.is_floating_point:
        return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
    return torch.equal(a, b)


# ----------------------------------------------------------------------------- oracle cases

def make_cases(n, seed=0):
    """Random and edge seasons: one oracle input row (list of numbers) per case."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        edge = k % 12
        N = int(rng.integers(150, 330))
        base = _f32(rng.choice([0.0, 3.0, 5.0, rng.uniform(-2, 8)]))
        cutoff = _f32(base + rng.uniform(12, 25))
        opt = _f32(base + rng.uniform(10, 22))
        tmx = _f32(opt + rng.uniform(5, 15))
        nonlinear = 1 if edge == 3 or rng.uniform() < 0.1 else 0
        scale = rng.uniform(0.6, 1.4)
        emerg = _f32(rng.choice([50, 60, 70, 118]) * scale)
        flow = _f32(emerg + rng.uniform(400, 1300) * scale)
        fill = _f32(flow + rng.uniform(50, 250) * scale)
        mat = _f32(fill + rng.uniform(400, 800) * scale)
        tub = _f32(rng.uniform(emerg, mat))
        d_acc = _f32(rng.uniform(450, 1200) * scale)
        d_culm = _f32(rng.uniform(100, 300) * scale)
        d_sen = _f32(rng.uniform(300, 750) * scale)
        d_root = _f32(rng.choice([1040.0, rng.uniform(300, 1500)]))
        mat_sig, has_sen, tuber, fruit = 1, 1, 0, 0
        v_en, v_lo, v_hi, v_st, v_end, v_min = 0, 2.0, 6.0, 0.0, 50.0, 0.0
        photo, stop, unc = 0, 0.0, 0.0
        sens = _f32(rng.choice([0.5, 1.0, rng.uniform(0, 2)]))
        if edge in (1, 2, 11) or rng.uniform() < 0.2:
            v_en = 1
            v_lo, v_hi = _f32(rng.uniform(-2, 4)), _f32(rng.uniform(5, 10))
            v_end = _f32(rng.uniform(20, 60))
            v_min = _f32(rng.choice([0.0, 0.5, rng.uniform(0, 1)]))
            v_st = _f32(rng.uniform(1, 15)) if edge == 2 else 0.0
        if edge == 4:
            photo, stop, unc = 2, _f32(rng.uniform(8, 11)), _f32(rng.uniform(12, 16))
        if edge == 5:
            photo, stop, unc = 1, _f32(rng.uniform(13, 16)), _f32(rng.uniform(8, 12))
        if edge == 6:
            photo, stop, unc = int(rng.choice([1, 2, 3])), 12.0, 12.0
        if edge == 7:                                          # same-day transitions
            d_acc, d_culm, d_sen = (_f32(rng.uniform(1, 40)) for _ in range(3))
            flow = _f32(emerg + rng.uniform(0, 20))
            fill = _f32(flow + rng.uniform(-10, 10))
            mat = _f32(fill + rng.uniform(0, 30))
        if edge == 8:                                          # filling before flowering, zeros
            flow, fill = fill, flow
            if rng.uniform() < 0.5:
                flow = 0.0
            else:
                fill = 0.0
        if edge == 9:
            mat_sig, has_sen = 0, int(rng.uniform() < 0.5)
        if edge == 10:
            tuber, fruit = 1, int(rng.uniform() < 0.5)
        if edge == 11:
            cutoff = _f32(base - rng.uniform(0, 3))            # cutoff below base
            if v_en:
                v_end = v_st                                   # start == end
        row = [N, nonlinear, base, cutoff, opt, tmx, emerg, tub, flow, fill, mat,
               d_acc, d_culm, d_sen, d_root, mat_sig, has_sen, tuber, fruit,
               v_en, v_lo, v_hi, v_st, v_end, v_min, photo, stop, unc, sens]
        na = int(rng.choice([0, 0, 1, 2, 3]))
        row.append(na)
        for _ in range(na):
            row += [int(rng.choice(STAGES)), int(rng.integers(0, 4)), int(rng.integers(0, 4)),
                    _f32(rng.uniform(-2, 40))]
        t0 = rng.uniform(-6, 16)
        amp = rng.uniform(8, 18)
        ph0 = rng.uniform(0, 2 * math.pi)
        for d in range(N):
            tmax = t0 + amp * math.sin(ph0 + 2 * math.pi * d / 365.0) + rng.normal(0, 4) + 8
            tmin = tmax - rng.uniform(2, 16)
            if edge == 11 and d % 7 == 0:
                tmax = tmin = base                             # exactly at base
            if edge == 11 and d % 11 == 0:
                tmax, tmin = cutoff + 1.0, cutoff - 1.0
            u = rng.uniform()
            wsi = 0.0 if u < 0.6 else (rng.uniform(0, 1) if u < 0.97 else 1.0)
            vpd = rng.uniform(0, 4)
            srf = 0.0 if rng.uniform() < 0.1 else rng.uniform(0, 1.3)
            dl = rng.uniform(8, 16)
            row += [tmax, tmin, dl, wsi, vpd, srf]
        rows.append(row)
    return rows


def _is_int_field(i, row):
    if i in (0, 1, 15, 16, 17, 18, 19, 25, 29):
        return True
    na = int(row[29])
    if 30 <= i < 30 + 4 * na:
        return (i - 30) % 4 != 3
    return False


def _fmt(row):
    return " ".join(str(int(v)) if _is_int_field(i, row) else float(v).hex()
                    for i, v in enumerate(row))


def _parse_out(line):
    return [float(int(x)) if x.lstrip("-").isdigit() else float.fromhex(x) for x in line.split()]


def run_oracle(rows):
    inp = "\n".join(_fmt(r) for r in rows) + "\n"
    r = subprocess.run([ORACLE], input=inp, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = r.stdout.splitlines()
    enum = [int(x) for x in lines[0].split()[1:]]
    assert enum == [ph.NGS_NONE, ph.NGS_PLANTING, ph.NGS_GERMINATION, ph.NGS_EMERGENCE,
                    ph.NGS_ACCRESCENCE, ph.NGS_END_CANOPY_GROWTH, ph.NGS_ANTHESIS,
                    ph.NGS_FILLING, ph.NGS_MATURITY, ph.NGS_QUIESCENCE], enum
    out, i = [], 1
    for r_ in rows:
        n = int(r_[0])
        out.append([_parse_out(l) for l in lines[i:i + n]])
        i += n
    assert i == len(lines)
    return out


# ----------------------------------------------------------------------------- torch side

def params_from_rows(rows, device="cpu"):
    t = lambda col, dt=torch.float64: torch.tensor([r[col] for r in rows], dtype=dt, device=device)
    b = lambda col: t(col, torch.long) != 0
    A = max(1, max(int(r[29]) for r in rows))
    st = torch.full((len(rows), A), -1, dtype=torch.long)
    off = torch.zeros((len(rows), A), dtype=torch.long)
    par = torch.full((len(rows), A), -1, dtype=torch.long)
    val = torch.zeros((len(rows), A), dtype=torch.float64)
    for i, r in enumerate(rows):
        for a in range(int(r[29])):
            st[i, a], off[i, a], par[i, a], val[i, a] = r[30 + 4 * a: 34 + 4 * a]
    return ph.PhenologyParams(
        temps=torch.stack([t(2), t(3), t(4), t(5)], -1), nonlinear=b(1),
        emergence=t(6), tuberization=t(7), flowering=t(8), filling=t(9), maturity=t(10),
        dur_accrescence=t(11), dur_culminescence=t(12), dur_senescence=t(13),
        dur_root_elongation=t(14), maturity_significant=b(15), has_leaf_senescence=b(16),
        is_tuber=b(17), fruit_harvest=b(18),
        culminate_senescence=torch.zeros(len(rows), dtype=torch.bool, device=device),
        vern_enabled=b(19), vern_low=t(20), vern_high=t(21), vern_start=t(22), vern_end=t(23),
        vern_min=t(24), photo_crop=t(25, torch.long) > 0, photo_phen=t(25, torch.long) != 0,
        photo_stop=t(26), photo_unconstrained=t(27), sens_water_stress=t(28),
        adj_stage=st.to(device), adj_offset=off.to(device), adj_param=par.to(device),
        adj_value=val.to(device), is_fruit=torch.zeros(len(rows), dtype=torch.bool, device=device),
        rapid_fruit=torch.zeros(len(rows), dtype=torch.float64, device=device),
        grape=torch.zeros(len(rows), dtype=torch.bool, device=device))


def torch_run(rows, device="cpu"):
    """All cases batched; returns per case a list of per-day records laid out like the oracle."""
    P = params_from_rows(rows, device)
    n = len(rows)
    Nmax = max(int(r[0]) for r in rows)
    W = torch.zeros((Nmax, n, NDAY), dtype=torch.float64)
    for i, r in enumerate(rows):
        start = 30 + 4 * int(r[29])
        N = int(r[0])
        W[:N, i] = torch.tensor(r[start:start + N * NDAY], dtype=torch.float64).reshape(N, NDAY)
    W = W.to(device)
    s = ph.initial_state(P)
    s = ph.sow(s, P, torch.ones(n, dtype=torch.bool, device=device), 0)
    recs = []
    for d in range(Nmax):
        prev_y = s.gdds                     # GDDs_yesterday as printed (before end_day)
        prev_season = s.season_days
        w = W[d]
        s = ph.step(s, P, d, w[:, 0], w[:, 1], day_length=w[:, 2], water_stress_index=w[:, 3],
                    vpd_max=w[:, 4], senescence_reduction_factor=w[:, 5])
        recs.append((s, prev_y, prev_season))
    out = []
    for i, r in enumerate(rows):
        days = []
        for d in range(int(r[0])):
            s, py, pse = recs[d]
            rec = [float(s.tmax_adj[i]), float(s.gdd[i]), float(s.gdds[i]), float(py[i]),
                   float(s.stage[i])]
            for k in range(4):
                on = bool(s.period_on[i, k])
                tt = float(s.period_tt[i, k])
                dur = [P.dur_accrescence, P.dur_culminescence, P.dur_senescence,
                       P.dur_root_elongation][k][i]
                rec += ([1.0, tt, float(s.period_rel_elapsed[i, k]),
                         float(s.period_rel_remaining[i, k]), float(tt > float(dur))]
                        if on else [0.0, 0.0, 0.0, 0.0, 0.0])
            rec += [float(s.vfac_crop[i]), float(s.vfac_phen[i]), float(pse[i])]
            for code in (ph.NGS_PLANTING, ph.NGS_EMERGENCE, ph.NGS_ACCRESCENCE,
                         ph.NGS_END_CANOPY_GROWTH, ph.NGS_ANTHESIS, ph.NGS_FILLING,
                         ph.NGS_MATURITY):
                rec.append(float(s.init_day[i, code]))
            rec.append(float(s.tuber_day[i]))
            days.append(rec)
        out.append(days)
    return out


# field index -> absolute floor of the relative error (None: exact)
FIELDS = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: None}
for _k in range(4):
    FIELDS.update({5 + 5 * _k: None, 6 + 5 * _k: 1.0, 7 + 5 * _k: 1.0, 8 + 5 * _k: 1.0,
                   9 + 5 * _k: None})
FIELDS.update({25: 1.0, 26: 1.0, 27: None})
FIELDS.update({28 + j: None for j in range(8)})


def compare(ref, got, days=None):
    """Worst relative error over the float fields; asserts exact equality of the others."""
    worst = 0.0
    for c, (rc, gc) in enumerate(zip(ref, got)):
        pick = range(len(rc)) if days is None else days[c]
        for d in pick:
            r, g = rc[d], gc[d]
            for f, floor in FIELDS.items():
                if floor is None:
                    assert r[f] == g[f], (c, d, f, r[f], g[f])
                    continue
                if math.isnan(r[f]) or math.isnan(g[f]):
                    assert math.isnan(r[f]) and math.isnan(g[f]), (c, d, f)
                    continue
                worst = max(worst, abs(r[f] - g[f]) / (abs(r[f]) + floor))
    return worst


# ----------------------------------------------------------------------------- golden set

def _golden():
    rows, days, outs = [], [], []
    with gzip.open(GOLDEN, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.startswith("IN "):
                rows.append(_parse_out(line[3:]))
                days.append([])
                outs.append({})
            else:
                d, o = line.split("|")
                days[-1].append(int(d))
                outs[-1][int(d)] = _parse_out(o)
    return rows, days, outs


def test_golden_cases_match_reference():
    rows, days, outs = _golden()
    assert len(rows) >= 10
    got = torch_run(rows)
    ref = [[outs[c].get(d) for d in range(int(r[0]))] for c, r in enumerate(rows)]
    assert compare(ref, got, days) <= TOL


@pytest.mark.reference
@pytest.mark.skipif(not os.path.exists(ORACLE), reason="ref_cropsyst_phenology oracle not built")
def test_live_oracle_random_and_edge_cases():
    rows = make_cases(240, seed=4242)
    ref = run_oracle(rows)
    got = torch_run(rows)
    worst = compare(ref, got)
    assert worst <= TOL, worst
    # the branches were reached
    stages = {int(d[4]) for c in ref for d in c}
    assert {ph.NGS_GERMINATION, ph.NGS_ACCRESCENCE, ph.NGS_ANTHESIS, ph.NGS_FILLING,
            ph.NGS_MATURITY} <= stages
    assert any(d[19] == 1.0 for c in ref for d in c)                  # senescence expired
    assert any(0.0 < d[25] < 1.0 for c in ref for d in c)             # vernalization limiting
    assert any(c[-1][35] >= 0 for c in ref)                           # tuber initiation
    assert any(c[-1][32] < 0 and c[-1][33] >= 0 for c in ref)         # anthesis blocked by filling


def test_l1_batched_cell_tile_shape():
    """A [cell, tile] batch gives the same numbers as the flat batch."""
    rows = make_cases(12, seed=5)
    Nmax = max(int(r[0]) for r in rows)
    P = params_from_rows(rows)
    P2 = ph.PhenologyParams(*[x.reshape(3, 4, *x.shape[1:]) for x in P])
    s1 = ph.sow(ph.initial_state(P), P, torch.ones(12, dtype=torch.bool), 0)
    s2 = ph.sow(ph.initial_state(P2), P2, torch.ones(3, 4, dtype=torch.bool), 0)
    rng = np.random.default_rng(1)
    for d in range(min(Nmax, 200)):
        tx = torch.tensor(rng.uniform(-5, 30, 12))
        tn = tx - torch.tensor(rng.uniform(2, 12, 12))
        s1 = ph.step(s1, P, d, tx, tn)
        s2 = ph.step(s2, P2, d, tx.reshape(3, 4), tn.reshape(3, 4))
    for a, b in zip(s1, s2):
        assert _same(a, b.reshape(a.shape))


def test_sow_resets_and_terminate_freezes():
    rows = make_cases(2, seed=9)
    P = params_from_rows(rows)
    m = torch.tensor([True, False])
    s = ph.sow(ph.initial_state(P), P, m, 3)
    for d in range(3, 60):
        s = ph.step(s, P, d, 25.0, 10.0)
    assert float(s.gdds[1]) == 0.0 and int(s.stage[1]) == ph.NGS_NONE   # never sown: unchanged
    assert int(s.init_day[0, ph.NGS_PLANTING]) == 3
    s = ph.terminate(s, m)
    frozen = s
    s = ph.step(s, P, 60, 25.0, 10.0)
    for a, b in zip(s, frozen):
        assert _same(a, b)                                             # no crop: nothing moves
    # re-sowing gives the fresh-crop trajectory
    s = ph.sow(s, P, m, 61)
    t = ph.sow(ph.initial_state(P), P, m, 61)
    for d in range(61, 120):
        s = ph.step(s, P, d, 20.0, 5.0)
        t = ph.step(t, P, d, 20.0, 5.0)
    for a, b in zip(s, t):
        assert _same(a[0], b[0])


def test_params_from_crop_files():
    db = os.path.join(SIM, "Database")
    if not os.path.isdir(db):
        pytest.skip("CropSyst CRB build copy not present")
    from torchvic.crop.cropsyst import readers as rd
    t = rd.crop_table([f"{db}/Crop/name/pea_spring-high.crp", f"{db}/Crop/name/wheat_winter-high.crp"])
    p = ph.phenology_params(t)
    assert p.temps.tolist() == [[3.0, 25.0, 26.0, 40.0], [0.0, 23.0, 26.0, 40.0]]
    assert p.dur_accrescence.tolist() == [550.0, 1097.0]       # peak_LAI - emergence
    assert p.dur_culminescence.tolist() == [150.0, 252.0]      # senescence - peak_LAI
    assert p.dur_senescence.tolist() == [350.0, 716.0]         # full_senescence - senescence
    assert p.vern_enabled.tolist() == [False, True] and p.vern_min.tolist() == [0.0, 0.5]
    assert p.adj_stage.tolist() == [[8, 8], [8, 8]] and p.adj_offset.tolist() == [[1, 1], [1, 1]]
    assert p.adj_param.tolist() == [[0, 1], [2, 3]]
    assert p.adj_value.tolist() == [[5.0, 35.0], [30.0, 44.0]]


def _step_chain(s, P, day, tx, tn, dl, wsi, vpd, srf):
    return ph.step(s, P, day, tx, tn, dl, wsi, vpd, srf)


def test_compiles_without_graph_breaks():
    rows = make_cases(24, seed=17)
    P = params_from_rows(rows)
    s = ph.sow(ph.initial_state(P), P, torch.ones(24, dtype=torch.bool), 0)
    c = torch.compile(_step_chain, fullgraph=True, dynamic=False)
    rng = np.random.default_rng(3)
    se, sc = s, s
    for d in range(40):
        x = [torch.tensor(rng.uniform(lo, hi, 24)) for lo, hi in
             ((5, 30), (-5, 5), (8, 16), (0, 1), (0, 3), (0, 1))]
        x[1] = x[0] - 10 + x[1]
        day = torch.tensor(d)
        se = _step_chain(se, P, day, *x)
        sc = c(sc, P, day, *x)
    for a, b in zip(se, sc):
        if a.dtype.is_floating_point:
            assert torch.allclose(a, b, rtol=1e-13, atol=0, equal_nan=True)
        else:
            assert torch.equal(a, b)


# ----------------------------------------------------------------------------- replay

def _l2_case(case, device="cpu"):
    from torchvic.crop.cropsyst import readers as rd
    f = glob.glob(f"{SIM}/Scenarios/Liu_Linux_{case}_*/VIC_output/vic_crop_daily*")
    if not f:
        pytest.skip(f"reference crop csv of {case} not present")
    rows = list(csv.reader(open(f[0])))
    H = {h: i for i, h in enumerate(rows[0])}
    data = defaultdict(list)
    for r in rows[1:]:
        data[r[H["CroppingSyst_code"]]].append(r)
    tiles = sorted(data)
    names = sorted({r[H["Crop_name"]] for r in rows[1:]} - {"NOCROP"})
    P = ph.phenology_params(rd.crop_table([f"{SIM}/Database/Crop/name/{c}.crp" for c in names]),
                            device=device)
    nd, T = len(data[tiles[0]]), len(tiles)

    def col(name, fn=float):
        return [[fn(data[k][d][H[name]]) for k in tiles] for d in range(nd)]
    crop, lab = col("Crop_name", str), col("Grow_Stage", str)
    acc, tmax, tmin = col("Accum_DD"), col("Tmax"), col("Tmin")
    wsi, rhmin = col("water_stress_index"), col("RHUM_min_%")
    cid = torch.zeros(T, dtype=torch.long, device=device)
    st, n, worst, bad = None, 0, 0.0, []
    tens = lambda v: torch.tensor(v, dtype=torch.float64, device=device)
    for d in range(nd):
        sow = [crop[d][k] != "NOCROP" and (d == 0 or crop[d - 1][k] == "NOCROP") for k in range(T)]
        for k in range(T):
            if sow[k]:
                cid[k] = names.index(crop[d][k])
        Pt = ph.gather_params(P, cid)
        sowm = torch.tensor(sow, device=device)
        st = ph.initial_state(Pt) if st is None else st
        st = ph.sow(st, Pt, sowm, d)
        pw = tens([0.0 if (d == 0 or sow[k]) else wsi[d - 1][k] for k in range(T)])
        tx = tens(tmax[d])
        es = gr.saturation_vp_kpa(tx)     # VPD max = es(Tmax)(1 - RHmin), Murray es
        vpd = es * (1.0 - tens(rhmin[d]) / 100.0)
        st = ph.step(st, Pt, d, tx, tens(tmin[d]), water_stress_index=pw, vpd_max=vpd)
        stage, pon, tt = st.stage.cpu(), st.period_on.cpu(), st.period_tt.cpu()
        dsen = Pt.dur_senescence.cpu()
        for k in range(T):
            if crop[d][k] == "NOCROP":
                continue
            n += 1
            L = ph.stage_label(int(stage[k]), pon[k].tolist(), bool(tt[k, 2] > dsen[k]),
                               planting=sow[k], harvest="harvest" in lab[d][k])
            if L != lab[d][k]:
                bad.append((d, tiles[k], L, lab[d][k]))
            worst = max(worst, abs(float(st.gdds[k]) - acc[d][k]) / max(1.0, abs(acc[d][k])))
        harvest = torch.tensor([crop[d][k] != "NOCROP" and "harvest" in lab[d][k]
                                for k in range(T)], device=device)
        st = ph.terminate(st, harvest)
    return n, bad, worst


@pytest.mark.reference
@pytest.mark.parametrize("case", L2_CASES)
def test_l2_crb_stage_sequence(case):
    n, bad, worst = _l2_case(case)
    assert n > 1000
    assert not bad, bad[:5]
    assert worst < 3e-7, worst        # csv prints 8 significant digits (Accum_DD, WSI, RHmin)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_matches_reference_and_cuda_graph():
    rows, days, outs = _golden()
    got = torch_run(rows, device="cuda")
    ref = [[outs[c].get(d) for d in range(int(r[0]))] for c, r in enumerate(rows)]
    assert compare(ref, got, days) <= TOL
    if os.path.isdir(SIM):
        n, bad, worst = _l2_case(328617, device="cuda")
        assert not bad and worst < 3e-7
    # CUDA-graph capture of one day step over a large batch: no host syncs inside
    big = make_cases(48, seed=77)
    P = params_from_rows(big, device="cuda")
    P = ph.PhenologyParams(*[x.repeat(256, *([1] * (x.dim() - 1))) for x in P])
    n = P.emergence.shape[0]
    s0 = ph.sow(ph.initial_state(P), P, torch.ones(n, dtype=torch.bool, device="cuda"), 0)
    g_in = [torch.zeros(n, dtype=torch.float64, device="cuda") for _ in range(6)]
    day = torch.zeros((), dtype=torch.long, device="cuda")
    state = [s0]

    def fill(d):
        r = np.random.default_rng(d)
        vals = [r.uniform(5, 30, n), r.uniform(-8, 4, n), r.uniform(8, 16, n),
                r.uniform(0, 1, n), r.uniform(0, 3, n), r.uniform(0, 1, n)]
        vals[1] = vals[0] - 10 + vals[1]
        for t, v in zip(g_in, vals):
            t.copy_(torch.from_numpy(v))
        day.fill_(d)
    # eager reference trajectory
    se = s0
    for d in range(30):
        fill(d)
        se = ph.step(se, P, day, *g_in)
    # captured: the graph advances a static state buffer in place
    buf = [x.clone() for x in s0]
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    fill(0)
    with torch.cuda.stream(stream):
        for _ in range(2):
            ph.step(ph.PhenologyState(*buf), P, day, *g_in)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = ph.step(ph.PhenologyState(*buf), P, day, *g_in)
        for b, o in zip(buf, out):
            b.copy_(o)
    for b, x in zip(buf, s0):
        b.copy_(x)
    for d in range(30):
        fill(d)
        g.replay()
    torch.cuda.synchronize()
    for a, b in zip(se, buf):
        assert _same(a, b)
    # torch.compile (CUDA graphs, reduce-overhead) agrees with eager
    c = torch.compile(_step_chain, mode="reduce-overhead", fullgraph=True, dynamic=False)
    sc = s0
    for d in range(30):
        fill(d)
        sc = c(sc, P, day, *[x.clone() for x in g_in])
        sc = ph.PhenologyState(*[x.clone() for x in sc])
    for a, b in zip(se, sc):
        if a.dtype.is_floating_point:
            assert torch.allclose(a, b, rtol=1e-13, atol=0, equal_nan=True)
        else:
            assert torch.equal(a, b)


def _write_golden(n=10):
    rows = make_cases(n, seed=2026)
    ref = run_oracle(rows)
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with gzip.open(GOLDEN, "wt") as f:
        f.write("# ref_cropsyst_phenology L1 cases (I5): IN <input row>, then <day> | <oracle "
                "day row> on every 15th day and every day the stage or a period changes\n")
        for r, days in zip(rows, ref):
            f.write("IN " + _fmt(r) + "\n")
            for d, o in enumerate(days):
                prev = days[d - 1] if d else None
                change = prev is None or o[4] != prev[4] or any(
                    o[5 + 5 * k] != prev[5 + 5 * k] or o[9 + 5 * k] != prev[9 + 5 * k]
                    for k in range(4))
                if d % 15 == 0 or change or d == len(days) - 1:
                    f.write(f"{d} | " + " ".join(
                        (str(int(x)) if (i == 4 or i in (5, 9, 10, 14, 15, 19, 20, 24, 27)
                                         or i >= 28) else float(x).hex())
                        for i, x in enumerate(o)) + "\n")
    print(f"wrote {len(rows)} cases to {GOLDEN}")


if __name__ == "__main__":
    if "--write-golden" in sys.argv:
        _write_golden()
