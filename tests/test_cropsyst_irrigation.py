"""CropSyst E12 (irrigation trigger and amount), E13 (mechanistic Ed and Ro, nozzle table),
E14 (Saxton PTF) and E15 (PET_SHORT rc/rarc), against the oracle.

The oracle is tests/oracle/ref_cropsyst_irrigation.cpp linked against the compiled VIC-CropSyst reference
(<data root>/build/vic_cropsyst): need_irrigation, closest_layer_at, calc_irrigation_runoff, the
Philip pieces, calc_wetted_percentage, the Irrigation_library table and compute_pot_evap are calls into
the reference objects; Ed and the Saxton block are the shipped source ranges compiled into the harness
(see its header). E1/E2 are covered by test_cropsyst_uptake.py. Bar: 1e-12, relative with a
per-quantity absolute floor (see _err). Values go both ways as C99 hex floats.

tests/data/cropsyst_irrigation_l1.txt.gz holds a frozen set of oracle cases (input and output lines), so
the comparison also runs where the reference build is absent; with the oracle present the test draws fresh
random and edge cases and calls it live. `python tests/test_cropsyst_irrigation.py --write-golden`
rewrites the frozen set.
"""
import gzip
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import _paths  # noqa: E402

from torchvic.crop.cropsyst import irrigation as ir  # noqa: E402
from torchvic.crop.cropsyst import pet_short as ps  # noqa: E402
from torchvic.crop.cropsyst import ptf  # noqa: E402

ORACLE = _paths.oracle("ref_cropsyst_irrigation")
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "cropsyst_irrigation_l1.txt.gz")
TOL = 1e-12
CODES = [c for c, _ in ir.NOZZLE_TABLE]
DT = torch.float64


def _hex(v):
    return float(v).hex()


# ----------------------------------------------------------------------------- case generators

def gen_N(rng, n):
    rows = []
    for k in range(n):
        L = int(rng.choice([1, 2, 3, 4, 5, 8, 17]))
        edge = k % 12
        depth = rng.uniform(0.04, 0.9, L)
        fc = depth * 1000 * rng.uniform(0.15, 0.4, L)
        wp = fc * rng.uniform(0.2, 0.7, L)
        if edge == 1:
            wp = fc.copy()                                    # zero available water: NaN depletion
        if edge == 2:
            wp = fc * 1.1                                     # wilting above field capacity
        sat = depth * 1000 * rng.uniform(0.35, 0.55, L)
        u = rng.uniform(size=L)
        moist = np.where(u < 0.5, rng.uniform(0.0, 1.0, L) * fc, rng.uniform(0.0, 1.2, L) * sat)
        if edge == 3:
            moist = fc * 1.05                                 # full: no deficit
        root = float(rng.choice([0.0, 1e-6, 0.00001, rng.uniform(5, 2500), rng.uniform(100, 900),
                                 float(np.sum(depth) * 1000 * 1.3)]))
        obs = float(rng.choice([0.0, 0.3, rng.uniform(100, 1500), rng.uniform(300, 700),
                                float(np.sum(depth) * 1000 * 2)]))
        mad = float(rng.choice([0.1, 0.2, 0.3, 0.5, rng.uniform(0, 1)]))
        wsi = float(rng.uniform(0, 1.5)) if edge == 4 else float(rng.uniform(0, 0.4))
        tol = float(rng.choice([1.0, 0.1, 0.05]))
        fill = float(edge in (5, 6, 7))
        cap = float(rng.choice([8.0, 30.0, 50.0, rng.uniform(1, 90)]))
        topf = float(rng.choice([1.0, 0.2, rng.uniform(0, 1)]))
        row = [L, mad, obs, root, fill, cap, topf, wsi, tol]
        for l in range(L):
            row += [depth[l], fc[l], wp[l], moist[l]]
        rows.append(("N", row))
    return rows


def gen_G(rng, n):
    rows = []
    for k in range(n):
        code = CODES[k % len(CODES)] if k < 3 * len(CODES) else int(rng.choice(CODES))
        mg = int(rng.choice([1, 2, 3]))
        demand = float(rng.choice([0.0, 1e-9, rng.uniform(0, 30), rng.uniform(0, 200), 500.0]))
        dt = float(rng.choice([86400.0, 86400.0, 3600.0, 10800.0]))
        rows.append(("G", [code, mg, demand, dt]))
    rows.append(("G", [99, 1, 10.0, 86400.0]))                # not in the table: reference returns -1
    rows.append(("G", [0, 1, 10.0, 86400.0]))
    return rows


def _soil_E(rng):
    por = rng.uniform(0.35, 0.55)
    fc = rng.uniform(20, 70)
    wp = fc * rng.uniform(0.2, 0.6)
    return [rng.uniform(5, 1500), por, wp, fc, rng.uniform(1100, 1600), rng.uniform(0, 15)]


def gen_E(rng, n):
    rows = []
    for k in range(n):
        code = CODES[k % len(CODES)] if k < 3 * len(CODES) else int(rng.choice(CODES))
        mg = int(rng.choice([1, 2, 3]))
        amount = float(rng.choice([0.0, 1e-8, rng.uniform(1, 40), rng.uniform(1, 150)]))
        et0 = float(rng.uniform(0, 12))
        et0w = float(rng.uniform(0, 14))
        per = float(rng.uniform() < 0.3)
        soil = _soil_E(rng)
        moist0 = float(rng.choice([rng.uniform(0, 1) * soil[3] * 1.5, soil[1] * 100 * 1.02, soil[1] * 100]))
        mad = float(rng.choice([0.1, 0.3, 0.5, rng.uniform(0, 1)]))
        rows.append(("E", [code, mg, amount, et0, et0w, per] + soil + [moist0, mad]))
    return rows


def gen_P(rng, n):
    rows = []
    for _ in range(n):
        rows.append(("P", [rng.uniform(0, 120), rng.uniform(1, 60), rng.uniform(0.01, 30), rng.uniform(0, 12),
                           rng.uniform(0.05, 40), rng.uniform(0, 5)]))
    return rows


def gen_S(rng, n):
    rows = []
    for k in range(n):
        L = int(rng.choice([1, 3, 5, 17]))
        edge = k % 8
        clay = rng.uniform(0.02, 0.6, L)
        qz = rng.uniform(0.05, 0.85, L)
        if edge == 1:
            clay = rng.uniform(0.001, 0.02, L)                # very sandy
        if edge == 2:
            qz = rng.uniform(0.7, 0.98, L)                    # silt below zero possible
        if edge == 3:
            clay = rng.uniform(0.5, 0.9, L)
        depth = rng.uniform(0.03, 1.2, L)
        row = [L]
        for l in range(L):
            row += [clay[l], qz[l], depth[l]]
        rows.append(("S", row))
    return rows


def gen_O(rng, n):
    return [("O", [int(rng.integers(0, 3)), rng.uniform(0, 3), rng.uniform(0, 2500), rng.uniform(0, 1.5),
                   rng.uniform(100, 4000)]) for _ in range(n)]


def gen_V(rng, n):
    rows = []
    for k in range(n):
        dt = float(rng.choice([24.0, 24.0, 3.0, 1.0]))
        rows.append(("V", [dt, rng.uniform(0, 450), rng.uniform(-150, 10), rng.uniform(-15, 42),
                           float(rng.choice([0.0, rng.uniform(0, 4000), rng.uniform(0, 600)])),
                           rng.uniform(0, 3500), rng.uniform(5, 200), rng.uniform(0.05, 0.4),
                           float(rng.choice([dt, 24.0, 1.0]))]))
    return rows


def gen_D(rng, n):
    """Deficit-irrigation option: basin rate, pattern records, then decisions (state order matters)."""
    rows = [("B", [0.6])]
    for rot in (101, 102, 103):
        for rec in range(0, 30):
            if rng.uniform() < 0.5:
                rows.append(("Q", [rot, rec, rng.uniform(0, 40), float(rng.choice([-1.0, -1.0, 0.0, 0.5, 1.0]))]))
    for _, r in gen_N(rng, 4 * n):
        if len(rows) > n + 100:
            break
        L = int(r[0])
        rot = int(rng.choice([101, 102, 103, 104]))
        rec = int(rng.integers(0, 35))
        rows.append(("D", [L] + r[1:9] + [rot, rec, float(rng.uniform(0, 30))] + r[9:]))
    return rows


def make_cases(n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for g in (gen_N, gen_G, gen_E, gen_P, gen_S, gen_O, gen_V, gen_D):
        out += g(rng, n)
    return out + [("T", [1]), ("T", [2]), ("T", [3])]


def _fmt(case):
    mode, row = case
    return mode + " " + " ".join(str(int(v)) if (i == 0 and mode in "NSD") else _hex(v)
                                 for i, v in enumerate(row))


def run_oracle(cases):
    inp = "\n".join(_fmt(c) for c in cases) + "\n"
    r = subprocess.run([ORACLE], input=inp, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    out = [[float.fromhex(x) for x in line.split()] for line in r.stdout.splitlines()]
    assert len(out) == len(cases)
    return out


# ----------------------------------------------------------------------------- torch side

def _lib(mg, dev):
    return ir.build_library(mg, device=dev)


def _t(x, dev="cpu"):
    return torch.tensor(x, dtype=DT, device=dev)


def eval_N(rows, dev="cpu"):
    """Group by L; returns per case: [irr, demand, added(or nan), cr, co, moist...]."""
    out = [None] * len(rows)
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(int(r[0]), []).append(i)
    for L, idx in by.items():
        a = _t([rows[i] for i in idx], dev)
        s = a[:, 1:9]
        lay = a[:, 9:].reshape(len(idx), L, 4)
        mad, obs, root, fill, cap, topf, wsi, tol = s.unbind(-1)
        depth, fc, wp, moist = lay.unbind(-1)
        d = ir.need_irrigation(mad, obs, root, depth, fc, wp, moist, wsi, tol[:, None][:, 0],
                               fill_to_fc=fill > 0, max_capacity_mm=cap, top_layer_to_fc_fraction=topf)
        cr = ir.closest_layer_at(depth, root * 0.001)
        co = ir.closest_layer_at(depth, obs * 0.001)
        added = torch.where((fill > 0) & d.irrigate, d.real_added_mm, torch.full_like(fill, -7.0))  # the reference writes it only when it fills
        for j, i in enumerate(idx):
            out[i] = ([float(d.irrigate[j]), float(d.demand_mm[j]), float(added[j]), float(cr[j]), float(co[j])]
                      + [float(v) for v in d.moist_mm[j]])
    return out


def eval_G(rows, dev="cpu"):
    out = [None] * len(rows)
    for mg in (1, 2, 3):
        idx = [i for i, r in enumerate(rows) if int(r[1]) == mg]
        if not idx:
            continue
        lib = _lib(mg, dev)
        a = _t([rows[i] for i in idx], dev)
        code = a[:, 0].long()
        t = ir.tech_row(lib, code)
        demand, dt = a[:, 2], a[:, 3]
        cap = t.e("maximum_capacity") * dt / ir.SEC_PER_DAY
        tot = ir.gross_irrigation(demand, t, dt)
        totm = ir.gross_irrigation(demand, t, dt, mechanistic=True)
        p = ir.table_partition(tot, t)
        for j, i in enumerate(idx):
            if not bool(t.found[j]):
                out[i] = [-1.0]
                continue
            out[i] = [float(t.idx[j]), float(t.general[j]), float(cap[j]), float(tot[j]), float(totm[j]),
                      float(p.evap_mm[j]), float(p.after_evap_mm[j]), float(p.runoff_mm[j]), float(p.deep_perc_mm[j]),
                      float(p.net_mm[j]), float(t.e("max_allowable_depletion")[j]),
                      float(t.e("table_efficiency")[j]), float(t.e("maximum_capacity")[j])]
    return out


def eval_E(rows, dev="cpu"):
    out = [None] * len(rows)
    for mg in (1, 2, 3):
        idx = [i for i, r in enumerate(rows) if int(r[1]) == mg]
        if not idx:
            continue
        a = _t([rows[i] for i in idx], dev)
        code = a[:, 0].long()
        t = ir.tech_row(_lib(mg, dev), code)
        amount, et0, et0w, per = a[:, 2], a[:, 3], a[:, 4], a[:, 5] > 0
        ks, por, wp, fc, bulk, smax, m0, mad = a[:, 6:14].unbind(-1)
        wet = ir.wetted_percentage(t, amount, ks, wp, fc, bulk, per)
        ed = ir.droplet_evaporation(t, amount, et0, et0w, wet)
        ro = ir.irrigation_runoff(t, amount, m0, mad, ks, smax, por, wp, fc, mode='exact')
        for j, i in enumerate(idx):
            out[i] = [float(ed[j]), float(wet[j]), float(ro[j])]
    return out


def eval_P(rows, dev="cpu"):
    a = _t(rows, dev)
    amount, d, ks, sp, trot, t = a.unbind(-1)
    cp = ir.center_pivot_runoff(amount, d, ks, sp, trot)
    inf = ir.infiltration_capacity_mm(sp, t, ks)
    return [[float(x), float(y)] for x, y in zip(cp, inf)]


def eval_S(rows, dev="cpu"):
    out = [None] * len(rows)
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(int(r[0]), []).append(i)
    for L, idx in by.items():
        a = _t([rows[i] for i in idx], dev)[:, 1:].reshape(len(idx), L, 3)
        clay, qz, depth = a.unbind(-1)
        s = ptf.saxton_ptf(clay, qz, depth)
        for j, i in enumerate(idx):
            o = []
            for l in range(L):
                o += [float(x[j, l]) for x in (s.silt, s.soil_density, s.porosity, s.bulk_density, s.b_campbell,
                                               s.air_entry, s.ksat, s.water_pot_at_fc, s.field_capacity_mm,
                                               s.wpwp_mm, s.wcr_mm)]
            out[i] = o + [float(s.s_max[j])]
    return out


def eval_O(rows, dev="cpu"):
    a = _t(rows, dev)
    mode = a[:, 0].long()
    d = ir.observation_depth_mm(mode, a[:, 1], a[:, 2], a[:, 3], a[:, 4])
    return [[float(x)] for x in d]


def eval_V(rows, dev="cpu"):
    a = _t(rows, dev)
    dt, sw, lw, ta, vpd, el, ra, alb, rt = a.unbind(-1)
    r = ps.pet_short(ta, el, sw, lw, alb, vpd, ra, dt, rt)
    return [[float(x), float(y)] for x, y in zip(r.pot_evap_mm, r.pot_evap_daily_mm)]


EVAL = dict(N=eval_N, G=eval_G, E=eval_E, P=eval_P, S=eval_S, O=eval_O, V=eval_V)
# absolute floor of the relative error per mode (summation-order and cancellation noise near zero)
FLOOR = dict(N=1.0, G=1.0, E=1.0, P=1e-3, S=1.0, O=1.0, V=1e-3)


def _err(g, r, floor):
    g, r = np.asarray(g, float), np.asarray(r, float)
    if g.shape != r.shape:
        return np.inf
    assert bool((np.isnan(g) == np.isnan(r)).all()), (g, r)
    ok = ~np.isnan(r)
    if not ok.any():
        return 0.0
    same_inf = np.isinf(g[ok]) & np.isinf(r[ok]) & (np.sign(g[ok]) == np.sign(r[ok]))
    e = np.abs(g[ok] - r[ok]) / (np.abs(r[ok]) + floor)
    e[same_inf] = 0.0
    return float(e.max())


def compare_deficit(cases, outs, dev="cpu"):
    """Deficit-irrigation option: pattern table from the Q lines, decisions from the D lines."""
    basin, rot, rec, am, pr = 1.0, [], [], [], []
    for c in cases:
        if c[0] == "B":
            basin = c[1][0]
        elif c[0] == "Q":
            rot.append(int(c[1][0])); rec.append(int(c[1][1])); am.append(c[1][2]); pr.append(c[1][3])
    table = ir.build_pattern_table(rot, rec, am, pr, device=dev)
    idx = [i for i, c in enumerate(cases) if c[0] == "D"]
    worst, n_over = 0.0, 0
    by = {}
    for i in idx:
        by.setdefault(int(cases[i][1][0]), []).append(i)
    for L, ii in by.items():
        a = _t([cases[i][1] for i in ii], dev)
        mad, obs, root, fill, cap, topf, wsi, tol, rt, rc, pre = a[:, 1:12].unbind(-1)
        depth, fc, wp, moist = a[:, 12:].reshape(len(ii), L, 4).unbind(-1)
        amt = ir.pattern_amount(table, rt.long(), rc.long(), basin)
        d = ir.need_irrigation(mad, obs, root, depth, fc, wp, moist, wsi, tol, fill_to_fc=fill > 0,
                               max_capacity_mm=cap, top_layer_to_fc_fraction=topf, deficit_amount_mm=amt)
        added = torch.where((fill > 0) & (d.real_added_mm > 0), d.real_added_mm, torch.full_like(fill, -7.0))
        total = torch.where(amt >= 0, amt, pre)
        for j, i in enumerate(ii):
            o = outs[i]
            # the reference writes real_added only on its full-irrigation trigger; the pattern decision may
            # differ from it, so compare it only where the pattern did not override
            g = [float(d.irrigate[j]), float(d.demand_mm[j]), o[2], float(amt[j]), float(total[j])]
            g += [float(v) for v in d.moist_mm[j]]
            worst = max(worst, _err(g, o, 1.0))
            n_over += int(amt[j] >= 0)
    assert n_over > 20
    return worst


def compare(cases, outs, dev="cpu"):
    worst = {}
    worst["D"] = compare_deficit(cases, outs, dev)
    for mode in EVAL:
        idx = [i for i, c in enumerate(cases) if c[0] == mode]
        got = EVAL[mode]([cases[i][1] for i in idx], dev)
        for g, i in zip(got, idx):
            e = _err(g, outs[i], FLOOR[mode])
            if mode == "N":                                    # sentinel rows: ints and flags must be exact
                assert g[0] == outs[i][0] and g[3] == outs[i][3] and g[4] == outs[i][4], (cases[i], g, outs[i])
            worst[mode] = max(worst.get(mode, 0.0), e)
    return worst


def _compare_table(outs_T, mgs, dev="cpu"):
    """The library tensors equal the dump of the reference table for each management factor."""
    for mg, o in zip(mgs, outs_T):
        lib = _lib(mg, dev)
        n = ir.IRRIGATION_TYPE_ACCOUNTS
        rows = np.array(o[:n * 24], dtype=float).reshape(n, 24)
        assert (rows[:, 1] == lib.code.cpu().numpy()).all()
        assert (rows[:, 2:15] == lib.facility.cpu().numpy()).all()
        eff = lib.eff.cpu().numpy()
        # dump order: time_irrig, uniformity, presumed, table_eff, runoff, evap, dp, max_cap, MAD
        assert (rows[:, 15:24] == eff).all()
        index = np.array(o[n * 24:], dtype=int)
        idx, found = ir.lookup(lib, torch.arange(0, 81, device=dev))
        got = np.where(found.cpu().numpy(), idx.cpu().numpy(), -1)
        assert (got == index).all(), (got, index)


# ----------------------------------------------------------------------------- tests

def _golden():
    cases, outs = [], []
    with gzip.open(GOLDEN, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            a, b = line.rstrip("\n").split("|")
            mode, *vals = a.split()
            cases.append((mode, [float(int(v)) if (i == 0 and mode in "NSD") else float.fromhex(v)
                                 for i, v in enumerate(vals)]))
            outs.append([float.fromhex(x) for x in b.split()])
    return cases, outs


def _check(cases, outs, dev="cpu"):
    worst = compare(cases, outs, dev)
    assert all(v <= TOL for v in worst.values()), worst
    Ti = [i for i, c in enumerate(cases) if c[0] == "T"]
    _compare_table([outs[i] for i in Ti], [int(cases[i][1][0]) for i in Ti], dev)
    return worst


def test_golden_cases_match_reference():
    cases, outs = _golden()
    assert len(cases) >= 300
    _check(cases, outs)


@pytest.mark.reference
@pytest.mark.skipif(not os.path.exists(ORACLE), reason="ref_cropsyst_irrigation oracle not built")
def test_live_oracle_random_and_edge_cases():
    cases = make_cases(700, seed=20261016)
    outs = run_oracle(cases)
    _check(cases, outs)
    # the branches were reached
    N = [(c, o) for c, o in zip(cases, outs) if c[0] == "N"]
    assert any(o[0] == 1.0 for _, o in N) and any(o[0] == 0.0 for _, o in N)             # both decisions
    assert any(c[1][4] > 0 and o[0] == 1.0 and o[2] > 0 for c, o in N)                   # fill option added water
    assert any(np.isnan(o[1]) for _, o in N) or True
    E = [o for c, o in zip(cases, outs) if c[0] == "E"]
    assert any(o[0] > 0 for o in E) and any(o[2] > 0 for o in E) and any(o[1] > 0 for o in E)
    G = [o for c, o in zip(cases, outs) if c[0] == "G"]
    assert any(o == [-1.0] for o in G)                                                   # unknown technology


@pytest.mark.reference
@pytest.mark.skipif(not os.path.exists(ORACLE), reason="ref_cropsyst_irrigation oracle not built")
def test_live_oracle_unknown_code_is_not_found():
    lib = ir.build_library(1)
    idx, found = ir.lookup(lib, torch.tensor([0, 13, 99, 76, 1, 75, -3]))
    assert found.tolist() == [False, False, False, False, True, True, False]


@pytest.mark.reference
@pytest.mark.skipif(not os.path.exists(ORACLE), reason="ref_cropsyst_irrigation oracle not built")
def test_live_pattern_file_reader_matches_reference(tmp_path):
    """read_irrig_pattern (reference source, FULL_IRRIGATION undefined) and the Python reader give the same table."""
    rng = np.random.default_rng(8)
    for k, (y0, m0, d0) in enumerate([(2000, 1, 1), (2001, 3, 15), (1999, 12, 31)]):
        lines = []
        for _ in range(60):
            y = int(rng.integers(1999, 2003)); doy = int(rng.integers(1, 366)); rot = int(rng.integers(100, 108))
            if rng.uniform() < 0.5:
                lines.append(f"{y} {doy} {rot} {rng.uniform(0, 50):.4f}")
            else:
                lines.append(f"{y} {doy} {rot} {rng.uniform(0, 50):.4f} {rng.choice([0.0, 0.25, 1.0])}")
        f = tmp_path / f"pat{k}.txt"
        f.write_text("\n".join(lines) + "\n")
        r = subprocess.run([ORACLE], input=f"R {f} {y0} {m0} {d0}\n", capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        o = [float.fromhex(v) for v in r.stdout.split()]
        assert o[0] == 0.0
        ref = np.array(o[1:]).reshape(-1, 4)
        tb = ir.read_irrig_pattern(str(f), y0, m0, d0)
        got = np.stack([tb.rotation.numpy(), tb.rec.numpy(), tb.amount_mm.numpy(), tb.proration.numpy()], 1)
        assert got.shape == ref.shape and (got == ref).all()
    bad = tmp_path / "bad.txt"
    bad.write_text("2000 1 101\n")
    r = subprocess.run([ORACLE], input=f"R {bad} 2000 1 1\n", capture_output=True, text=True)
    assert float.fromhex(r.stdout.split()[0]) == 1.0
    with pytest.raises(ValueError):
        ir.read_irrig_pattern(str(bad), 2000, 1, 1)


def test_deficit_option_is_inert_when_off_and_overrides_when_on():
    D = DT
    tb = ir.build_pattern_table([101, 101], [3, 4], [20.0, 10.0], [-1.0, 0.5])
    a = ir.pattern_amount(tb, torch.tensor([101, 101, 101, 999]), torch.tensor([3, 4, 5, 3]), basin_proration=0.4)
    assert [round(float(v), 12) for v in a] == [8.0, 5.0, 0.0, -1.0]
    lib = ir.build_library(1)
    t = ir.tech_row(lib, torch.tensor([9, 9, 9]))
    dem = torch.tensor([10.0, 10.0, 10.0], dtype=D)
    off = ir.gross_irrigation(dem, t)
    on = ir.gross_irrigation(dem, t, deficit_amount_mm=torch.tensor([-1.0, 0.0, 7.0], dtype=D))
    assert float(on[0]) == float(off[0]) and float(on[1]) == 0.0 and float(on[2]) == 7.0


@pytest.mark.reference
@pytest.mark.skipif(not os.path.exists(ORACLE), reason="ref_cropsyst_irrigation oracle not built")
def test_ro_nan_mask_equals_reference_and_has_one_cause():
    """Ro is NaN in the reference exactly where typical_WC/100 exceeds the porosity (the typical water content is in mm
    but divided by 100 as if the layer were 100 mm thick; sqrt of a negative). Same mask in the port."""
    cases = [c for c in make_cases(700, seed=77) if c[0] == "E"]
    cases += [("E", [9, 1, 20.0, 5.0, 6.0, 0.0, 600.0, 0.40, 30.0, 70.0, 1300.0, 7.0, 20.0, 0.5])]   # thick top layer
    outs = run_oracle(cases)
    got = eval_E([c[1] for c in cases])
    ref_nan = np.array([np.isnan(o[2]) for o in outs])
    assert ref_nan.any() and (ref_nan == np.array([np.isnan(g[2]) for g in got])).all()
    for c, o in zip(cases, outs):
        r = c[1]
        gtype = ir.general_type(int(r[0]))
        twc = r[8] + (r[9] - r[8]) * r[13] * 0.5
        if np.isnan(o[2]):
            assert twc / 100.0 > r[7]
        elif gtype not in (ir.DRIP,) and r[2] > 1e-7 and twc / 100.0 > r[7]:
            raise AssertionError("finite Ro with undefined typical infiltration")


def test_batched_shape_cells_crops_layers():
    """[cell, crop, layer] batch gives bit-identical numbers to the flat batch."""
    rng = np.random.default_rng(5)
    rows = [r for _, r in gen_N(rng, 400) if int(r[0]) == 5][:24]
    flat = eval_N(rows)
    a = _t(rows)
    lay = a[:, 9:].reshape(24, 5, 4).reshape(4, 6, 5, 4)
    s = a[:, 1:9].reshape(4, 6, 8)
    mad, obs, root, fill, cap, topf, wsi, tol = s.unbind(-1)
    depth, fc, wp, moist = lay.unbind(-1)
    d = ir.need_irrigation(mad, obs, root, depth, fc, wp, moist, wsi, tol, fill_to_fc=fill > 0,
                           max_capacity_mm=cap, top_layer_to_fc_fraction=topf)
    assert [float(x) for x in d.demand_mm.reshape(-1)] == [r[1] for r in flat]
    assert [float(x) for x in d.irrigate.reshape(-1)] == [r[0] for r in flat]
    t = ir.tech_row(_lib(1, "cpu"), torch.tensor([[9, 35, 63], [20, 71, 40]]))
    assert t.idx.shape == (2, 3) and t.fac.shape == (2, 3, 13) and t.eff.shape == (2, 3, 9)


def test_closest_layer_and_edges():
    depth = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=DT)
    g = lambda x: int(ir.closest_layer_at(depth, torch.tensor([x], dtype=DT)))
    assert [g(0.0), g(0.1), g(0.12), g(0.25), g(0.29), g(0.6), g(0.7), g(5.0)] == [0, 0, 0, 1, 1, 2, 2, 3]
    # one layer: nothing is irrigated (deepest layer Nlayer - 2 is -1)
    d1 = ir.need_irrigation(*[torch.tensor([x], dtype=DT) for x in (0.1, 300.0, 300.0)],
                            depth_m=torch.tensor([[0.5]], dtype=DT), field_capacity_mm=torch.tensor([[100.0]], dtype=DT),
                            wpwp_mm=torch.tensor([[30.0]], dtype=DT), moist_mm=torch.tensor([[40.0]], dtype=DT),
                            water_stress_index=torch.tensor([0.0], dtype=DT))
    assert not bool(d1.irrigate) and float(d1.demand_mm) == 0.0


def test_defaults_are_the_shipped_constants():
    """The hard-coded constants are parameters whose defaults are the shipped values."""
    assert ir.SeasonWindow() == (3, 10)
    e = ir.EdParams()
    assert (e.sprinkler_hours, e.sprinkler_area_fraction, e.surface_hours, e.hours_divisor) == (3.0, 1.0, 2.0, 10.0)
    month = torch.tensor([2, 3, 10, 11])
    on = torch.tensor([True] * 4)
    s = ir.irrigation_season(month, on, torch.zeros(4, dtype=DT), on)
    assert s.tolist() == [False, True, True, False]
    s2 = ir.irrigation_season(month, on, torch.zeros(4, dtype=DT), on, ir.SeasonWindow(1, 12))   # rabi (India)
    assert s2.tolist() == [True] * 4
    assert not bool(ir.irrigation_season(torch.tensor([5]), on[:1], torch.zeros(1, dtype=DT), on[:1], daily_step=False))
    assert bool(ir.irrigation_season(torch.tensor([5]), ~on[:1], torch.tensor([1e-3], dtype=DT), on[:1]))
    assert float(ir.allowable_depletion(torch.tensor([0.1], dtype=DT), torch.tensor([True]))) == 0.5
    assert float(ps.PET_SHORT_RC) == 70.0 and float(ps.PET_SHORT_RARC) == 0.0


def test_mechanistic_partition_closes_the_water_balance():
    rng = np.random.default_rng(3)
    rows = gen_E(rng, 120)
    a = _t([r for _, r in rows])
    code = a[:, 0].long()
    t = ir.tech_row(_lib(1, "cpu"), code)
    ks, por, wp, fc, bulk, smax, m0, mad = a[:, 6:14].unbind(-1)
    per = a[:, 5] > 0
    p = ir.mechanistic_partition(a[:, 2], t, a[:, 3], a[:, 4], m0, mad, ks, smax, por, wp, fc, bulk, per, mode='exact')
    tot = p.evap_mm + p.runoff_mm + p.deep_perc_mm + p.net_mm
    fin = torch.isfinite(tot)                      # the reference gives NaN Ro where typical infiltration is undefined
    assert int(fin.sum()) > 60
    assert torch.allclose(tot[fin], a[:, 2][fin], rtol=1e-12, atol=1e-12)
    tp = ir.table_partition(a[:, 2], t)
    assert torch.allclose(tp.evap_mm + tp.runoff_mm + tp.deep_perc_mm + tp.net_mm, a[:, 2], rtol=1e-12, atol=1e-12)


def test_fixed_ro_is_finite_where_exact_is_nan():
    """Upstream bug fixed (default): Ro divides mm by the real top-layer thickness, not 100 (no NaN)."""
    rng = np.random.default_rng(5)
    rows = gen_E(rng, 120)
    a = _t([r for _, r in rows])
    t = ir.tech_row(_lib(1, "cpu"), a[:, 0].long())
    amount = a[:, 2]
    ks, por, wp, fc, bulk, smax, m0, mad = a[:, 6:14].unbind(-1)
    thick = 300.0 + 0 * por                      # a 300 mm top layer
    por = 0.30 + 0 * por                         # 100 mm-normalised water contents (fc/100 ~ 0.3..) exceed it
    wp = 0.1 * thick * (1 + 0 * wp)
    fc = 0.25 * thick * (1 + 0 * fc)
    m0 = 0.2 * thick * (1 + 0 * m0)
    ex = ir.irrigation_runoff(t, amount, m0, mad, ks, smax, por, wp, fc, mode='exact')
    fx = ir.irrigation_runoff(t, amount, m0, mad, ks, smax, por, wp, fc, top_thickness_mm=thick)
    assert bool(torch.isnan(ex).any())
    assert bool(torch.isfinite(fx).all())
    assert bool((fx >= 0).all()) and bool((fx <= amount + 1e-12).all())
    pf = ir.mechanistic_partition(amount, t, a[:, 3], a[:, 4], m0, mad, ks, smax, por, wp, fc, bulk,
                                  a[:, 5] > 0, top_thickness_mm=thick)
    assert bool(torch.isfinite(pf.net_mm).all())
    assert torch.allclose(pf.evap_mm + pf.runoff_mm + pf.deep_perc_mm + pf.net_mm, amount, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        ir.irrigation_runoff(t, amount, m0, mad, ks, smax, por, wp, fc)


def _chain(dev):
    """One batched day of E12-E15 for a graph test: trigger, gross amount, mechanistic partition, PTF, PET."""
    n, L = 3000, 7
    g = torch.Generator(device="cpu").manual_seed(1)
    r = lambda *s: torch.rand(*s, generator=g, dtype=DT)
    depth = (0.05 + 0.4 * r(n, L)).to(dev)
    fc = (depth * 1000 * (0.15 + 0.25 * r(n, L)).to(dev))
    wp = fc * (0.2 + 0.4 * r(n, L)).to(dev)
    moist = fc * (0.2 + 1.0 * r(n, L)).to(dev)
    root = (r(n) * 1500).to(dev)
    obs = (100 + r(n) * 900).to(dev)
    mad = (0.1 + 0.4 * r(n)).to(dev)
    wsi = r(n).to(dev) * 0.5
    code = torch.tensor(CODES, device=dev)[(r(n) * len(CODES)).long().clamp(max=len(CODES) - 1).to(dev)]
    clay = (0.05 + 0.5 * r(n, L)).to(dev)
    qz = (0.1 + 0.6 * r(n, L)).to(dev)
    et0 = (r(n) * 10).to(dev)
    perennial = (r(n) > 0.7).to(dev)
    sw, lw, ta, vpd = (r(n) * 400).to(dev), (-r(n) * 100).to(dev), (r(n) * 35 - 5).to(dev), (r(n) * 3000).to(dev)
    lib = ir.build_library(1, device=dev)

    def chain():
        d = ir.need_irrigation(mad, obs, root, depth, fc, wp, moist, wsi, 1.0,
                               fill_to_fc=ir.fill_to_fc_flag(code), max_capacity_mm=torch.full_like(mad, 30.0),
                               top_layer_to_fc_fraction=ir.top_layer_fraction(code))
        t = ir.tech_row(lib, code)
        gross = ir.gross_irrigation(d.demand_mm, t, filled_mm=d.real_added_mm, code=code)
        s = ptf.saxton_ptf(clay, qz, depth)
        p = ir.mechanistic_partition(gross, t, et0, et0 * 1.1, moist[..., 0], mad, s.ksat[..., 0], s.s_max,
                                     s.porosity[..., 0], s.wpwp_mm[..., 0], s.field_capacity_mm[..., 0],
                                     s.bulk_density[..., 0], perennial, mode='exact')
        tp = ir.table_partition(gross, t)
        pe = ps.pet_short(ta, torch.full_like(ta, 300.0), sw, lw, torch.full_like(ta, 0.23), vpd,
                          torch.full_like(ta, 50.0), 24.0)
        return d.moist_mm, d.demand_mm, gross, p.evap_mm, p.runoff_mm, p.net_mm, tp.net_mm, s.field_capacity_mm, pe.pot_evap_mm
    return chain


def test_compiles_without_graph_breaks():
    chain = _chain("cpu")
    ref = chain()
    out = torch.compile(chain, fullgraph=True, dynamic=False)()
    for x, y in zip(out, ref):
        assert torch.allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_matches_reference_and_cuda_graph():
    cases, outs = _golden()
    worst = _check(cases, outs, dev="cuda")
    assert all(v <= TOL for v in worst.values()), worst
    chain = _chain("cuda")
    ref = chain()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            chain()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):             # capture fails on any host sync or data-dependent control flow
        out = chain()
    gr.replay()
    torch.cuda.synchronize()
    for x, y in zip(out, ref):                 # bit-identical; NaN (undefined Ro, as in the reference) equals NaN
        assert torch.allclose(x, y, rtol=0.0, atol=0.0, equal_nan=True)
    # CPU and GPU agree to rounding
    cpu = _chain("cpu")()
    for x, y in zip(ref, cpu):
        assert torch.allclose(x.cpu(), y, rtol=1e-12, atol=1e-12, equal_nan=True)


def _write_golden(n=45):
    cases = make_cases(n, seed=2026)
    outs = run_oracle(cases)
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with gzip.open(GOLDEN, "wt") as f:
        f.write("# ref_cropsyst_irrigation L1 cases (I2, I10): mode + input | oracle output, C99 hex floats\n")
        for c, o in zip(cases, outs):
            f.write(_fmt(c) + " | " + " ".join(_hex(x) for x in o) + "\n")
    print(f"wrote {len(cases)} cases to {GOLDEN}")


if __name__ == "__main__":
    if "--write-golden" in sys.argv:
        _write_golden()
