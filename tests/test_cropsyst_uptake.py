"""CropSyst E1-E3 (root water uptake, soil water potential, uptake into VIC layers), against the oracle.

The oracle is tests/oracle/ref_cropsyst_uptake.cpp linked against the compiled VIC-CropSyst
reference build (<data root>/build/vic_cropsyst): E1, E2 and E3 are calls into the reference
objects. Inputs are random and edge states; values go both ways as C99 hex floats, so the
comparison is float64 bit-for-bit up to summation order. Bar: 1e-12 (relative, see _close).

tests/data/cropsyst_uptake_l1.txt holds a frozen set of oracle cases (input and output lines), so
the comparison also runs where the reference build is absent; with the oracle present the test
regenerates fresh cases and calls it live. `python tests/test_cropsyst_uptake.py --write-golden`
rewrites the frozen set from the oracle.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import _paths  # noqa: E402

from torchvic.crop.cropsyst import uptake as up  # noqa: E402

ORACLE = _paths.oracle("ref_cropsyst_uptake")
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "cropsyst_uptake_l1.txt")
TOL = 1e-12
N_SCAL = 12   # scalar inputs before the layer blocks
N_LAY = 9     # inputs per layer


def _f32(x):
    return float(np.float32(x))


def make_cases(n, seed=0, Ls=(1, 3, 5, 17)):
    """Random and edge E1-E3 states, one oracle input row (list of floats) per case."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        L = int(Ls[k % len(Ls)])
        edge = k % 10                                     # 1-6 edge states, 0 and 7-9 plain
        pot = rng.uniform(0, 0.012) if edge != 1 else 0.0
        green = rng.uniform(0, 1) if edge != 2 else (0.0 if k % 2 else 1.0)
        maxup = _f32(rng.choice([6.0, 8.0, 9.0, 10.0, 12.0, 14.0, rng.uniform(1, 20)]))
        icpt = rng.uniform(0, 0.002) * (rng.uniform() < 0.6)
        if edge == 3:
            icpt = pot + rng.uniform(0, 0.001)            # interception covers the demand
        co2 = rng.uniform(0.7, 1.3) if rng.uniform() < 0.5 else 1.0
        closure = _f32(rng.uniform(-1500, -600))
        wilt = _f32(closure - rng.uniform(100, 900))
        dry = _f32(rng.choice([8.0, 8.0, 4.0, 0.8, 1.0, rng.uniform(0.5, 12)]))
        sat = 8.0
        lwp_prev = rng.uniform(-2000, 0)
        dt = float(rng.choice([86400.0, 3600.0, 10800.0]))
        ebare = rng.uniform(0, 3)
        row = [L, pot, green, maxup, icpt, co2, closure, wilt, dry, sat, lwp_prev, dt, ebare]
        fr = rng.dirichlet(np.ones(L))
        fr[rng.uniform(size=L) < 0.2] = 0.0
        if fr.sum() > 0:
            fr = fr / fr.sum()                            # root length fractions sum to 1 ...
        if rng.uniform() < 0.3:
            fr = fr * rng.uniform(0.3, 1.0)               # ... or less (roots still growing in)
        if edge == 4:
            fr[:] = 0.0                                   # no roots: root_cond_adj <= 1e-5
        if edge == 5:
            fr[:] = 1e-7                                  # roots below the 1e-5 activity floor
        for l in range(L):
            depth = rng.uniform(0.05, 1.0)
            por = rng.uniform(0.35, 0.55)
            max_moist = por * depth * 1000.0
            ae = -rng.uniform(0.3, 6.0)
            b = rng.uniform(2.0, 13.0)
            psi_fc = -33.0 if rng.uniform() < 0.8 else -rng.uniform(5, 60)
            u = rng.uniform()
            if edge != 6 and u > 0.35:
                # target a potential between -5 and -3000 J/kg (the range crops live in)
                target = -np.exp(rng.uniform(np.log(5.0), np.log(3000.0)))
                frac = min((target / ae) ** (-1.0 / b), 1.0) if target < ae else 1.0
            elif edge == 6 or u < 0.15:
                frac = rng.uniform(0.02, 0.15)            # dry: psi below LWP wilt
            elif u < 0.2:
                frac = float(rng.choice([0.0, 1e-9, 1.0, 1.02]))
            else:
                frac = rng.uniform(0.1, 1.0)
            moist = frac * max_moist
            ice = 0.0
            v = rng.uniform()
            if v < 0.1:
                ice = rng.uniform(0, 0.5) * moist
            elif v < 0.13:
                ice = max_moist * rng.uniform(1.0, 1.2)   # ice above saturation: air entry 0
            row += [depth, max_moist, ae, b, psi_fc, moist, ice, fr[l], rng.uniform(0, 5)]
        rows.append(row)
    return rows


def _fmt(row):
    return " ".join(str(int(row[0])) if i == 0 else float(v).hex() for i, v in enumerate(row))


def run_oracle(rows):
    inp = "\n".join(_fmt(r) for r in rows) + "\n"
    r = subprocess.run([ORACLE], input=inp, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = [[float.fromhex(x) for x in line.split()] for line in r.stdout.splitlines()]
    assert len(out) == len(rows)
    return out


def split_out(L, o):
    i = 0
    def take(n):
        nonlocal i
        v = o[i:i + n]; i += n
        return v
    d = dict(psi=take(L))
    d["act"], d["uact"], d["lwp"], d["icpt"] = take(4)
    d["uptake"] = take(L)
    d["td"] = take(L)
    d["extracted"], d["stress"] = take(2)
    d["evap"] = take(L)
    d["top"] = take(1)[0]
    assert i == len(o)
    return d


def torch_eval(rows, device="cpu", shape=None):
    """Evaluate a group of same-L cases in one batched call; returns the same dict layout."""
    a = torch.tensor(rows, dtype=torch.float64, device=device)
    L = int(rows[0][0])
    s = a[:, 1:N_SCAL + 1]
    lay = a[:, N_SCAL + 1:].reshape(len(rows), L, N_LAY)
    if shape is not None:
        s = s.reshape(*shape, N_SCAL)
        lay = lay.reshape(*shape, L, N_LAY)
    pot, green, maxup, icpt, co2, clo, wil, dry, _sat, lwp_prev, dt, ebare = s.unbind(-1)
    depth, mm, ae, b, psifc, moist, ice, fr, tdp = lay.unbind(-1)
    psi = up.soil_water_potential(moist, ice, depth, mm, ae, b)
    p = up.TranspirationParams(maxup, clo, wil, dry)
    r = up.root_uptake(pot, green, icpt, fr, psi, psifc, p, lwp_prev, co2)
    dts = float(rows[0][11])
    e = up.uptake_to_vic(r.uptake_m, tdp, ebare, dt_sec=dts)
    flat = lambda t, last: t.reshape(len(rows), *( (L,) if last else ()))
    return dict(psi=flat(psi, 1), act=flat(r.act_transp_m, 0), uact=flat(r.uptake_act_m, 0),
                lwp=flat(r.leaf_water_pot, 0), icpt=flat(r.interception_m, 0),
                uptake=flat(r.uptake_m, 1), td=flat(e.transp_daily_mm, 1),
                extracted=flat(e.extracted_m, 0), stress=flat(r.water_stress, 0),
                evap=flat(e.layer_evap_mm, 1), top=flat(e.top_evap_mm, 0))


# per-quantity absolute floor of the relative test (summation-order noise on near-zero results)
SCALE = dict(psi=1.0, act=1e-3, uact=1e-3, lwp=1.0, icpt=1e-3, uptake=1e-3, td=1.0,
             extracted=1e-3, stress=1.0, evap=1.0, top=1.0)


def compare(rows, outs, device="cpu"):
    groups = {}
    for i, r in enumerate(rows):
        key = (int(r[0]), float(r[11]))
        groups.setdefault(key, []).append(i)
    worst = {}
    for (L, _dt), idx in groups.items():
        got = torch_eval([rows[i] for i in idx], device)
        for k in SCALE:
            ref = torch.tensor([split_out(L, outs[i])[k] for i in idx], dtype=torch.float64)
            g = got[k].cpu()
            same_nan = torch.isnan(ref) == torch.isnan(g)
            assert bool(same_nan.all()), k
            ok = ~torch.isnan(ref)
            err = ((g - ref).abs() / (ref.abs() + SCALE[k]))[ok]
            if err.numel():
                worst[k] = max(worst.get(k, 0.0), float(err.max()))
    return worst


def _golden():
    rows, outs = [], []
    with open(GOLDEN) as f:
        for line in f:
            if line.startswith("#"):
                continue
            a, b = line.split("|")
            rows.append([float(int(a.split()[0]))] + [float.fromhex(x) for x in a.split()[1:]])
            outs.append([float.fromhex(x) for x in b.split()])
    return rows, outs


def test_golden_cases_match_reference():
    rows, outs = _golden()
    assert len(rows) >= 100
    worst = compare(rows, outs)
    assert all(v <= TOL for v in worst.values()), worst


@pytest.mark.reference
@pytest.mark.skipif(not os.path.exists(ORACLE), reason="ref_cropsyst_uptake oracle not built")
def test_live_oracle_random_and_edge_cases():
    rows = make_cases(1400, seed=12345)
    outs = run_oracle(rows)
    worst = compare(rows, outs)
    assert all(v <= TOL for v in worst.values()), worst
    # the edge branches were reached
    stats = [split_out(int(r[0]), o) for r, o in zip(rows, outs)]
    assert any(s["uact"] == 0.0 and r[1] > 0 for s, r in zip(stats, rows))       # no-uptake paths
    assert any(0.0 < s["stress"] < 1.0 for s in stats)                            # stressed
    assert any(v < 0 for s in stats for v in s["uptake"])                         # hydraulic lift
    assert any(v == 0.0 for s, r in zip(stats, rows) for v in s["psi"][:1])       # air entry 0 (ice)


def test_batched_shape_cells_crops_layers():
    """[cell, crop, layer] batch gives the same numbers as the flat batch."""
    rows = [r for r in make_cases(84, seed=7, Ls=(3,)) if r[11] == 86400.0][:24]
    flat = torch_eval(rows)
    nd = torch_eval(rows, shape=(4, 6))
    for k in flat:
        assert torch.equal(flat[k], nd[k]), k


def test_inactive_crop_and_raf_edges():
    rows = make_cases(1, seed=3, Ls=(3,))
    a = torch.tensor(rows, dtype=torch.float64)
    L = 3
    s = a[:, 1:N_SCAL + 1]
    lay = a[:, N_SCAL + 1:].reshape(1, L, N_LAY)
    pot, green, maxup, icpt, co2, clo, wil, dry, _sat, lwp_prev, dt, ebare = s.unbind(-1)
    depth, mm, ae, b, psifc, moist, ice, fr, tdp = lay.unbind(-1)
    psi = up.soil_water_potential(moist, ice, depth, mm, ae, b)
    p = up.TranspirationParams(maxup, clo, wil, dry)
    r = up.root_uptake(pot, green, icpt, fr, psi, psifc, p, lwp_prev, co2,
                       active=torch.tensor([False]))
    assert float(r.act_transp_m) == 0.0 and float(r.uptake_m.abs().sum()) == 0.0
    assert float(r.leaf_water_pot) == float(lwp_prev) and float(r.interception_m) == float(icpt)
    # psi wetter than FC gives full root activity; psi below wilt gives none
    psi2 = torch.tensor([[-10.0, -40.0, -3000.0]], dtype=torch.float64)
    fr2 = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    r2 = up.root_uptake(torch.tensor([0.005]), torch.tensor([1.0]), torch.tensor([0.0]), fr2, psi2,
                        torch.full_like(psi2, -33.0), p, torch.tensor([-5.0]))
    assert float(r2.uptake_act_m) == 0.0 and float(r2.leaf_water_pot) == -5.0


def test_params_from_crop_files():
    db = _paths.env_path(('TORCHVIC_CROPSYST_DB',), 'build/vic_cropsyst/VIC-CropSyst_v3_CRB/Simulation/Database')
    if not os.path.isdir(db):
        pytest.skip("CropSyst CRB build copy not present")
    from torchvic.crop.cropsyst import readers as rd
    t = rd.crop_table([f"{db}/Crop/name/wheat_winter-low.crp", f"{db}/Crop/name/pea_spring-high.crp"])
    p = up.transpiration_params(t)
    assert p.max_water_uptake_mm.tolist() == [9.0, 8.0]
    assert p.lwp_stomatal_closure.tolist() == [-1200.0, -1200.0]
    assert p.lwp_wilt_point.tolist() == [-1600.0, -1800.0]
    assert p.dry_soil_root_activity_coef.tolist() == [8.0, 8.0]


def test_compiles_without_graph_breaks():
    """The E1-E3 chain is one graph under torch.compile (fullgraph) and matches eager."""
    rows = [r for r in make_cases(40, seed=11, Ls=(5,)) if r[11] == 86400.0]
    a = torch.tensor(rows, dtype=torch.float64)
    L = 5
    s = a[:, 1:N_SCAL + 1]
    lay = a[:, N_SCAL + 1:].reshape(len(rows), L, N_LAY)

    def chain(s, lay):
        pot, green, maxup, icpt, co2, clo, wil, dry, _sat, lwp_prev, dt, ebare = s.unbind(-1)
        depth, mm, ae, b, psifc, moist, ice, fr, tdp = lay.unbind(-1)
        psi = up.soil_water_potential(moist, ice, depth, mm, ae, b)
        r = up.root_uptake(pot, green, icpt, fr, psi, psifc, up.TranspirationParams(maxup, clo, wil, dry),
                           lwp_prev, co2)
        e = up.uptake_to_vic(r.uptake_m, tdp, ebare)
        return r.act_transp_m, r.uptake_m, r.leaf_water_pot, e.layer_evap_mm
    c = torch.compile(chain, fullgraph=True, dynamic=False)
    out = c(s, lay)
    ref = chain(s, lay)
    for x, y in zip(out, ref):
        assert torch.allclose(x, y, rtol=1e-12, atol=1e-18, equal_nan=True)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_matches_reference_and_cuda_graph():
    rows, outs = _golden()
    worst = compare(rows, outs, device="cuda")
    assert all(v <= TOL for v in worst.values()), worst
    # CUDA-graph capture of the batched chain: no host syncs inside
    big = [r for r in make_cases(4000, seed=99, Ls=(17,)) if r[11] == 86400.0]
    a = torch.tensor(big, dtype=torch.float64, device="cuda")
    L = 17
    s = a[:, 1:N_SCAL + 1].contiguous()
    lay = a[:, N_SCAL + 1:].reshape(len(big), L, N_LAY).contiguous()

    def chain(s, lay):
        pot, green, maxup, icpt, co2, clo, wil, dry, _sat, lwp_prev, dt, ebare = s.unbind(-1)
        depth, mm, ae, b, psifc, moist, ice, fr, tdp = lay.unbind(-1)
        psi = up.soil_water_potential(moist, ice, depth, mm, ae, b)
        r = up.root_uptake(pot, green, icpt, fr, psi, psifc, up.TranspirationParams(maxup, clo, wil, dry),
                           lwp_prev, co2)
        e = up.uptake_to_vic(r.uptake_m, tdp, ebare)
        return r.uptake_m, r.leaf_water_pot, e.layer_evap_mm
    ref = chain(s, lay)
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            chain(s, lay)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = chain(s, lay)
    g.replay()
    torch.cuda.synchronize()
    for x, y in zip(out, ref):
        assert torch.equal(x, y)
    # compiled with CUDA graphs (reduce-overhead) agrees with eager
    c = torch.compile(chain, mode="reduce-overhead", fullgraph=True, dynamic=False)
    for _ in range(3):
        oc = c(s, lay)
    for x, y in zip(oc, ref):
        assert torch.allclose(x, y, rtol=1e-12, atol=1e-18, equal_nan=True)


def _write_golden(n=140):
    rows = make_cases(n, seed=2026)
    outs = run_oracle(rows)
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with open(GOLDEN, "w") as f:
        f.write("# ref_cropsyst_uptake L1 cases (I4): input row | oracle output row, C99 hex floats\n")
        for r, o in zip(rows, outs):
            f.write(_fmt(r) + " | " + " ".join(float(x).hex() for x in o) + "\n")
    print(f"wrote {len(rows)} cases to {GOLDEN}")


if __name__ == "__main__":
    if "--write-golden" in sys.argv:
        _write_golden()
