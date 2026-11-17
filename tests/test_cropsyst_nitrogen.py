"""Crop and soil nitrogen of the VIC-CropSyst port: the oracle comparison against the compiled
reference and the replay against the original N-on CRB runs.

Replay data: the dump binary of <data root>/build/vic_cropsyst/crb_N (see tests/cropsyst_n_dump.py),
whose crop csv equals the original reference csv byte for byte. The component checks replay one
step of the land-unit day from the dumped state before it and compare with the dumped state
after it; the season check (test_l2_crb_seasons) runs the whole N chain day by day from the
dumped drivers only.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import _paths  # noqa: E402
from cropsyst_n_dump import read_dump, dump_path, L  # noqa: E402

from torchvic.crop.cropsyst import soil_n as sn
from torchvic.crop.cropsyst import organic_matter as om

L2_CASES = (330492, 308200, 328617)
N_RUNS = _paths.env_path(('TORCHVIC_CROPSYST_N_RUNS',), 'build/vic_cropsyst/crb_N')
T = lambda v: torch.tensor(v, dtype=torch.float64)

_CACHE = {}


def _load(case):
    if case not in _CACHE:
        p = dump_path(N_RUNS, case)
        if not p or os.path.getsize(p) == 0:
            pytest.skip(f"N-on dump of {case} not present")
        _CACHE[case] = read_dump(p)
    return _CACHE[case]


def _layers(h):
    return sn.SoilLayers(T(h["thickness"]), T(h["bulk_density"]), T(h["sat"]), T(h["fc"]), T(h["pwp"]),
                         T(h["ph_function"]))


def _state(r):
    return sn.SoilN(T(r["no3"]) * sn.N_TO_NO3, T(r["nh4"]) * sn.N_TO_NH4)


def _rel(a, b, floor=1e-12):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    both_nan = torch.isnan(a) & torch.isnan(b)
    d = (a - b).abs() / torch.maximum(b.abs(), torch.full_like(b, floor))
    d = torch.where(both_nan, torch.zeros_like(d), d)
    d = torch.where(torch.isnan(d), torch.full_like(d, float("inf")), d)
    return float(d.max()) if d.numel() else 0.0


def _env(h, p):
    lay = _layers(h)
    th = T(h["thickness"])
    liquid = T(p["moist"]) / (th * 1000.0)
    ice = T(p["ice"]) / (th * 1000.0)
    return lay, sn.abiotic_update(liquid, ice, T(p["T"]), lay)


@pytest.mark.reference
@pytest.mark.parametrize("case", L2_CASES)
def test_l2_abiotic_environment(case):
    """Abiotic_environment_profile::update on the dumped VIC moisture and temperature equals the
    reference's moisture and temperature functions (M record) on every tile day."""
    head, rec = _load(case)
    worst = 0.0
    n = 0
    for key, r in rec.items():
        if "M" not in r or "P" not in r:
            continue
        lay, env = _env(head[key[2]], r["P"])
        m = r["M"]
        worst = max(worst, _rel(env.moisture_daily, m["moisture"]), _rel(env.tf_daily_mean, m["tf_decomp"]),
                    _rel(env.tf_hour, m["tf_nitr"]))
        n += 1
    assert n > 1000 and worst <= 1e-15, (n, worst)


@pytest.mark.reference
@pytest.mark.parametrize("case", L2_CASES)
def test_l2_transformations(case):
    """Denitrification then nitrification (hourly) from the state after the crop's day (C) to the
    state after the transformations (T), with the day's organic-matter CO2 (M)."""
    head, rec = _load(case)
    worst = 0.0
    n = 0
    for key, r in rec.items():
        if not all(k in r for k in ("C", "T", "M", "P")):
            continue
        lay, env = _env(head[key[2]], r["P"])
        s, den, nit = sn.transformations(_state(r["C"]), lay, env, T(r["P"]["T"]), T(r["M"]["co2"]))
        t = r["T"]
        worst = max(worst, _rel(sn.no3_n(s), t["no3"], 1e-15), _rel(sn.nh4_n(s), t["nh4"], 1e-15),
                    _rel(den, t["x"][0], 1e-15), _rel(nit, t["x"][1], 1e-15))
        n += 1
    assert n > 1000 and worst <= 1e-12, (n, worst)


# ----------------------------------------------------------------------------- residue pools

def _pools(plist, surface, N):
    """Dumped pools (in list order) of one kind -> om.Pools with N slots."""
    sel = [p for p in plist if (p["position"] != om.SUBSURFACE) == surface]
    assert len(sel) <= N, (len(sel), N)
    p = om.empty_pools((), N)
    f = {k: list(getattr(p, k)) for k in om.Pools._fields if k != "n"}
    for j, q in enumerate(sel):
        f["active"][j] = True
        for k in ("type", "position", "layer", "cycling", "mass", "cfrac", "cn", "k", "kdet", "apm"):
            f[k][j] = q[k]
        f["env"][j] = q.get("env", 1 if surface else q["layer"])
    tb = lambda k, dt: torch.tensor([bool(x) for x in f[k]] if dt == torch.bool else [float(x) for x in f[k]], dtype=dt)
    return om.Pools(tb("active", torch.bool), *[torch.tensor([int(x) for x in f[k]]) for k in
                                                   ("type", "position", "layer", "cycling", "env")],
                    *[tb(k, torch.float64) for k in ("mass", "cfrac", "cn", "k", "kdet", "apm")],
                    torch.tensor(len(sel)))


def _residues(plist, flat_contact=0.3, S=96, B=192):
    return om.Residues(_pools(plist, True, S), _pools(plist, False, B), torch.tensor(float(flat_contact), dtype=torch.float64),
                       torch.tensor(False))


def _plist(p: om.Pools):
    out = []
    for j in range(p.active.shape[-1]):
        if bool(p.active[j]):
            out.append(dict(type=int(p.type[j]), position=int(p.position[j]), layer=int(p.layer[j]),
                            cycling=int(p.cycling[j]), env=int(p.env[j]), mass=float(p.mass[j]), cfrac=float(p.cfrac[j]),
                            cn=float(p.cn[j]), k=float(p.k[j]), kdet=float(p.kdet[j]), apm=float(p.apm[j])))
    return out


def _cmp_pools(mine, ref, what):
    """Same pools in the same order (per kind); returns the worst relative difference."""
    worst = 0.0
    for surface in (True, False):
        a = [p for p in mine if (p["position"] != om.SUBSURFACE) == surface]
        b = [p for p in ref if (p["position"] != om.SUBSURFACE) == surface]
        assert len(a) == len(b), (what, surface, len(a), len(b))
        for x, y in zip(a, b):
            for k in ("type", "position", "layer", "cycling"):
                assert x[k] == y[k], (what, k, x, y)
            if "env" in y and y["env"] >= 0:
                assert x["env"] == y["env"], (what, "env", x, y)
            for k in ("mass", "cfrac", "cn", "k", "kdet", "apm"):
                worst = max(worst, _rel(x[k], y[k], 1e-15))
    return worst


def _days(rec):
    keys = sorted(rec)
    return keys


def _prev(key):
    y, d, t = key
    return (y, d - 1, t) if d > 1 else (y - 1, 366 if (y - 1) % 4 == 0 else 365, t)


@pytest.mark.reference
@pytest.mark.parametrize("case", L2_CASES)
def test_l2_residue_start_day(case):
    """Detrition and deletion of negligible pools: E (yesterday) -> S (today)."""
    head, rec = _load(case)
    worst, n = 0.0, 0
    for key in _days(rec):
        r = rec[key]
        pk = _prev(key)
        if "S" not in r or pk not in rec or "E" not in rec[pk]:
            continue
        res = om.start_day(_residues(rec[pk]["E"]["pools"]))
        assert not bool(res.overflow)
        worst = max(worst, _cmp_pools(_plist(res.sur) + _plist(res.sub), r["S"]["pools"], key))
        n += 1
    assert n > 1000 and worst <= 1e-12, (n, worst)


@pytest.mark.reference
@pytest.mark.parametrize("case", L2_CASES)
def test_l2_residue_process(case):
    """Organic-matter process_day on days without operations: O -> M (N, CO2 by layer,
    mineralization NaN pattern) and the pools after exhaust (C has no pools; E when no harvest)."""
    head, rec = _load(case)
    worst, n = 0.0, 0
    for key in _days(rec):
        r = rec[key]
        pk = _prev(key)
        if not all(k in r for k in ("O", "M", "P")) or pk not in rec or "B" not in rec[pk]:
            continue
        lay, env = _env(head[key[2]], r["P"])
        surf = [q for q in r["O"]["pools"] if q["position"] != om.SUBSURFACE]
        fc = surf[0]["contact"] if surf else rec[pk]["B"]["flat_contact"]   # the shared flat contact variable
        res = _residues(r["O"]["pools"], fc)
        res2, s2, dec = om.process(res, env, _state(r["O"]))
        m = r["M"]
        worst = max(worst, _rel(sn.no3_n(s2), m["no3"], 1e-15), _rel(sn.nh4_n(s2), m["nh4"], 1e-15),
                    _rel(dec.co2, m["co2"], 1e-15), _rel(dec.mineralized, m["mineralized"], 1e-15))
        if "F" not in r and "Q" not in r and "E" in r:
            worst = max(worst, _cmp_pools(_plist(res2.sur) + _plist(res2.sub), r["E"]["pools"], key))
        n += 1
    assert n > 1000 and worst <= 1e-12, (n, worst)


SIM = _paths.data('build/vic_cropsyst/VIC-CropSyst_v3_CRB/Simulation')


def _crp_section(path, section):
    out, on = {}, False
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("["):
            on = line == f"[{section}]"
            continue
        if on and "=" in line:
            k, v = line.split("=", 1)
            try:
                out[k] = float(v.split()[0])
            except (ValueError, IndexError):
                out[k] = v.strip()
    return out


def _crop_names(case):
    """(year, doy, tile) -> crop name from the original crop csv."""
    import csv
    import glob
    f = glob.glob(f"{SIM}/Scenarios/Liu_Linux_{case}_*/VIC_output/vic_crop_daily*")[0]
    rows = list(csv.reader(open(f)))
    H = {h: i for i, h in enumerate(rows[0])}
    return {(int(r[H["Year"]]), int(r[H["DOY"]]), int(r[H["CroppingSyst_code"]])): r[H["Crop_name"]] for r in rows[1:]}


def _residue_params(name):
    return om.residue_params(_crp_section(f"{SIM}/Database/Crop/name/{name}.crp", "residue"))


@pytest.mark.reference
@pytest.mark.parametrize("case", L2_CASES)
def test_l2_harvest_residue(case):
    """distribute_any_crop_biomass_fate: the F pools plus the chaff, standing dead stubble and dead
    roots of the harvest (Y) give the E pools; and the shared flat contact fraction."""
    head, rec = _load(case)
    names = _crop_names(case)
    worst, n = 0.0, 0
    for key in _days(rec):
        r = rec[key]
        if "F" not in r or "chaff" not in r["F"]:
            continue
        f = r["F"]
        P = _residue_params(names[key])
        surf0 = [q for q in f["pools"] if q["position"] != om.SUBSURFACE]
        res = _residues(f["pools"], surf0[0]["contact"] if surf0 else r["B"]["flat_contact"])
        one = torch.tensor(True)
        if f["chaff"] > 0.0000001:
            res = om.add_surface(res, one, om.FLAT, T(f["chaff"]), T(f["residue_conc"]), P)
        if f["stubble"] > 0.0000001:
            res = om.add_surface(res, one, om.STUBBLE, T(f["stubble"]), T(f["residue_conc"]), P)
        if sum(f["roots_dead"]) > 0.0000001:
            res = om.add_subsurface(res, one, T(f["roots_dead"]), T(f["root_conc"]), P)
        worst = max(worst, _cmp_pools(_plist(res.sur) + _plist(res.sub), r["E"]["pools"], key))
        surf = [q for q in r["E"]["pools"] if q["position"] != om.SUBSURFACE]
        if surf:
            worst = max(worst, _rel(res.flat_contact, surf[0]["contact"], 1e-15))
        n += 1
    assert n >= 4 and worst <= 1e-12, (n, worst)


def _crop_params(names):
    from torchvic.crop.cropsyst import readers as rdr
    from torchvic.crop.cropsyst import phenology as ph
    from torchvic.crop.cropsyst import nitrogen as nt
    tab = rdr.crop_table([f"{SIM}/Database/Crop/name/{n}.crp" for n in names])
    return nt.n_params_from_table(tab), ph.phenology_params(tab), om.residue_params_from_table(tab)


@pytest.mark.reference
@pytest.mark.parametrize("case", L2_CASES)
def test_l2_crop_n_update(case):
    """Crop_nitrogen_V5::update_limited_growth + update_limited_pot_transpiration: crop N state
    and soil N before (N record) -> after (n record), on every crop day with a canopy."""
    from torchvic.crop.cropsyst import nitrogen as nt
    head, rec = _load(case)
    names = _crop_names(case)
    crops = sorted({v for v in names.values() if v != "NOCROP"})
    NP, PP, _ = _crop_params(crops)
    worst, n = {}, 0
    for key in _days(rec):
        r = rec[key]
        if "N" not in r or "n" not in r:
            continue
        i = crops.index(names[key])
        P = nt.gather_params(NP, torch.tensor(i))
        a, b = r["N"], r["n"]
        st = nt.initial_state(())._replace(
            canopy_n=T(a["canopy_n"]), root_n=T(a["root_n"]), canopy_conc=T(a["canopy_conc"]),
            root_conc=T(a["root_conc"]), store=T(a["store"]), n_max=T(a["n_max"]), n_crit=T(a["n_crit"]),
            n_min=T(a["n_min"]), max_uptake_per_length=T(a["max_uptake_per_length"]))
        lay = _layers(head[key[2]])
        fnd = nt.fract_n_demand(P, torch.tensor(False), torch.tensor(a["stage"]), T(a["gdds"]), PP.flowering[i],
                                PP.filling[i], PP.maturity[i], T(a["paw_n"]), lay.thickness, T(a["root_fract"]),
                                T(a["no3"]), T(a["nh4"]), T(r["P"]["T"][1]))
        st2, s2, lim, pot = nt.update_limited_growth(
            st, P, torch.tensor(True), T(a["attainable"]), T(a["raf"]), torch.tensor(False), PP.maturity_significant[i],
            T(a["gdds"]), PP.maturity[i], fnd, T(a["canopy_biomass_vital"]), T(a["root_biomass"]), T(a["root_growth"]),
            T(a["root_lengths"]), T(a["paw_n"]), lay, _state(a), T(a["tue_co2_adjusted"]))
        got = dict(canopy_n=st2.canopy_n, root_n=st2.root_n, canopy_conc=st2.canopy_conc, root_conc=st2.root_conc,
                   store=st2.store, limited_growth=st2.limited_growth, limited_pot=st2.limited_pot, stress=st2.stress,
                   stress32=st2.stress32, uptake=st2.uptake, fixation=st2.fixation)
        for k, v in got.items():
            worst[k] = max(worst.get(k, 0.0), _rel(v, b[k], 1e-15))
        worst["no3"] = max(worst.get("no3", 0.0), _rel(sn.no3_n(s2), b["no3"], 1e-15))
        worst["nh4"] = max(worst.get("nh4", 0.0), _rel(sn.nh4_n(s2), b["nh4"], 1e-15))
        n += 1
    assert n > 500 and max(worst.values()) <= 1e-12, (n, worst)


@pytest.mark.reference
@pytest.mark.parametrize("case", L2_CASES)
def test_l2_operations(case):
    """The day's operations (Q, in order: field-operation response of sowing, tillage and
    fertilizer operations, fertilizer N): S (after the organic-matter start_day) -> O. Days
    with an automatic N application are left to test_l2_crb_seasons (the amount comes from the
    soil N sampling days earlier)."""
    from torchvic.crop.cropsyst import n_management as nm
    head, rec = _load(case)
    names = _crop_names(case)
    sdr = sn.sdr_operations()
    worst, n = 0.0, 0
    for key in _days(rec):
        r = rec[key]
        if "Q" not in r or "S" not in r or "O" not in r:
            continue
        if any(q["type"] == 14 for q in r["Q"]):
            continue        # automatic N application: its amount comes from the sampling (season test)
        pk = _prev(key)
        h = head[key[2]]
        lay = _layers(h)
        po = rec[pk].get("O") if pk in rec else None
        sand = T(po["x"][0]) if po else T(h["sand"])
        clay = T(po["x"][1]) if po else T(h["clay"])
        dist = nm.initial_disturbance((), L)
        if po:
            dist = dist._replace(exists=torch.tensor(True), rating=T(po["dist"][1]), factor=T(po["dist"][0]),
                                 deepest=torch.tensor(max([l + 1 for l in range(L) if po["dist"][1][l] > 0] or [0])))
        if po and "P" in rec[pk]:       # yesterday's Disturbance::process (after the O record)
            liq = T(rec[pk]["P"]["moist"]) / (lay.thickness * 1000.0)
            aw = sn._macro_between((liq - lay.pwp) / (lay.fc - lay.pwp), torch.zeros_like(liq), torch.ones_like(liq))
            dist = nm.process_disturbance(dist, aw, clay)
        st = nm.SoilState(_state(r["S"]), _residues(r["S"]["pools"], 0.3), sand, clay, dist)
        sown = names.get(key, "NOCROP") != "NOCROP" and names.get(pk, "NOCROP") == "NOCROP"
        one = torch.tensor(True)
        for q in r["Q"]:
            if q["type"] in (1, 3, 6) and q["desc"] and not (q["type"] == 1 and not sown) and q["desc"] in sdr:
                st = nm.field_operation(st, one, lay, dict(sdr[q["desc"]], name=q["desc"]),
                                        (1.5, 0.5) if q["type"] == 3 else None)
            if q["type"] == 6:
                fr = sn.fract_inorganic_nh4_to_soil(q["nh4_source"], q["nh4_method"], q["vol_calculated"],
                                                     q["volatilization"], 20.0, 7.0)
                st = st._replace(n=sn.fertilizer(st.n, one, lay, T(q["no3_kg_ha"]), T(q["nh4_kg_ha"]), T(fr),
                                                 T(q["injection_depth_cm"])))
        o = r["O"]
        worst = max(worst, _rel(sn.no3_n(st.n), o["no3"], 1e-15), _rel(sn.nh4_n(st.n), o["nh4"], 1e-15),
                    _rel(st.sand, o["x"][0], 1e-15), _rel(st.clay, o["x"][1], 1e-15),
                    _rel(st.disturbance.rating, o["dist"][1], 1e-15),
                    _cmp_pools(_plist(st.residues.sur) + _plist(st.residues.sub), o["pools"], key))
        n += 1
    assert n > 10 and worst <= 1e-12, (n, worst)


@pytest.mark.reference
@pytest.mark.parametrize("case", L2_CASES)
def test_l2_crb_seasons(case):
    """Whole seasons with nitrogen on, every crop day of the original reference crop csv: GAI,
    canopy biomass, green cover, root depth, ET max, Accum_DD and every
    harvest's yield to print precision, with the
    crop N, soil mineral N, residues and operations of the port (driver: tests/cropsyst_n_l2.py).
    The port's soil NO3/NH4 also follows the reference's (dump E records) to 1e-6 relative (floor 1e-6 kg/m2 = 0.01 kg N/ha),
    the residual coming from the 7-8 digit csv drivers of the crop uptake."""
    import cropsyst_n_l2 as l2
    import math
    if not dump_path(N_RUNS, case):
        pytest.skip("N-on dump not present")
    pairs, ndiag = l2.run_case(case)
    for c, v in pairs.items():
        assert v, c
        if c == "Act_Transp_mm":
            continue        # the replayed driver ((1 - WSI) x limited potential, WSI printed to 8 digits)
        worst = max(abs(r - x) / max(abs(r), 1e-6) for _, _, r, x in v)
        assert worst < (3e-7 if c == "Accum_DD" else 1e-7), (c, worst)   # Accum_DD: 8-digit csv temperatures
    assert len(pairs["Yield_kg_m2"]) >= 4 and len(pairs["GAI"]) > 1400
    bad = 0
    for d, t, r3, r4, m3, m4 in ndiag:
        for a, b in zip(m3 + m4, r3 + r4):
            if math.isnan(a) != math.isnan(b) or (not math.isnan(b) and abs(a - b) > 1e-6 * max(abs(b), 1e-6)):
                bad += 1
    assert bad == 0, bad


# ----------------------------------------------------------------------------- oracle: soil day loop

def _l1_case(seed, L=8, D=40):
    """A random soil, initial N and D days of drivers and operations: (oracle line, python spec)."""
    import random
    rnd = random.Random(seed)
    sdr = sn.sdr_operations()
    names = [n for n, o in sdr.items() if o["depth_m"] > 0 and o["mixing"] > 0][:200]
    soil = []
    for _ in range(L):
        th = rnd.uniform(0.05, 0.3)
        mm = th * 1000.0 * rnd.uniform(0.38, 0.5)
        wcr = mm * rnd.uniform(0.55, 0.75)
        soil.append([th, rnd.uniform(1200, 1600), mm, wcr, wcr * rnd.uniform(0.35, 0.6),
                     rnd.uniform(0.1, 0.7), rnd.uniform(0.05, 0.4)])
    no3 = [rnd.uniform(0, 0.004) for _ in range(L)]
    nh4 = [rnd.uniform(0, 0.001) for _ in range(L)]
    rp = lambda: [rnd.uniform(0.2, 0.4), rnd.uniform(0.2, 0.4), 0.0, rnd.uniform(0.2, 0.4), rnd.uniform(0.2, 0.4),
                  0.0, rnd.uniform(5, 60), rnd.uniform(60, 400), rnd.uniform(400, 2000), rnd.uniform(100, 700),
                  rnd.uniform(0.38, 0.46), rnd.uniform(1, 6), rnd.uniform(1, 6)]
    days = []
    for d in range(D):
        moist = [s[2] * rnd.uniform(0.3, 0.95) for s in soil]
        ice = [m * rnd.uniform(0, 0.3) if rnd.random() < 0.15 else 0.0 for m in moist]
        tt = [rnd.uniform(-5, 30) for _ in soil]
        ops = []
        for _ in range(rnd.choice([0, 0, 1, 2, 3])):
            c = rnd.choice([1, 1, 2, 3, 3, 4, 5])
            if c == 1:
                p = rp()
                p[2] = 1.0 - p[0] - p[1]
                p[5] = 1.0 - p[3] - p[4]
                ops.append((1, rnd.choice([om.FLAT, om.STUBBLE]), rnd.uniform(0.01, 0.6), rnd.uniform(0.004, 0.02), p))
            elif c == 2:
                p = rp()
                p[2] = 1.0 - p[0] - p[1]
                p[5] = 1.0 - p[3] - p[4]
                ops.append((2, [rnd.uniform(0, 0.05) if rnd.random() < 0.8 else 0.0 for _ in range(L)],
                            rnd.uniform(0.004, 0.02), p))
            elif c == 3:
                ops.append((3, rnd.choice(names), rnd.uniform(0.5, 2.0), rnd.uniform(0.2, 1.0)))
            elif c == 4:
                ops.append((4, rnd.uniform(0, 80), rnd.uniform(0, 120), rnd.uniform(0.6, 1.0), rnd.choice([0.0, 5.0, 10.0, 25.0])))
            else:
                ops.append((5, [rnd.uniform(0, 0.0004) for _ in range(L)], [rnd.uniform(0, 0.0002) for _ in range(L)]))
        days.append((moist, ice, tt, ops))
    lay = sn.soil_layers(T([s[0] for s in soil]), T([s[1] for s in soil]), T([s[2] for s in soil]),
                         T([s[3] for s in soil]), T([s[4] for s in soil]))
    h = lambda x: float(x).hex()
    out = ["SOIL", str(L), str(D)]
    for s in soil:
        out += [h(x) for x in s]
    out += [h(x) for x in no3 + nh4]
    for moist, ice, tt, ops in days:
        out += [h(x) for x in moist + ice + tt]
        out.append(str(len(ops)))
        for o in ops:
            if o[0] == 1:
                out += ["1", str(o[1]), h(o[2]), h(o[3])] + [h(x) for x in o[4]]
            elif o[0] == 2:
                out += ["2"] + [h(x) for x in o[1]] + [h(o[2])] + [h(x) for x in o[3]]
            elif o[0] == 3:
                out += ["3", str(sdr[o[1]]["number"]), h(o[2]), h(o[3]), "2" if o[1] == "Seedbed finisher" else "1"]
            elif o[0] == 4:
                lyr = int(sn.layer_at_depth(lay.thickness, T(o[4] / 100.0)))
                out += ["4", h(o[1]), h(o[2]), h(o[3]), str(lyr)]
            else:
                out += ["5"] + [h(x) for x in o[1] + o[2]]
    return " ".join(out) + "\n", dict(L=L, soil=soil, no3=no3, nh4=nh4, days=days, lay=lay)


def _l1_python(spec):
    L, lay = spec["L"], spec["lay"]
    from torchvic.crop.cropsyst import n_management as nm
    sdr = sn.sdr_operations()
    one = torch.tensor(True)
    keys = ["fast_cycling_top_fraction", "slow_cycling_top_fraction", "lignified_top_fraction",
            "fast_cycling_root_fraction", "slow_cycling_root_fraction", "lignified_root_fraction",
            "fast_cycling_decomposition_time_50", "slow_cycling_decomposition_time_50",
            "lignified_decomposition_time_50", "detrition_time_50", "carbon_fraction",
            "stubble_area_covered_to_mass_ratio", "area_covered_to_mass_ratio"]
    st = nm.SoilState(sn.SoilN(T(spec["no3"]) * sn.N_TO_NO3, T(spec["nh4"]) * sn.N_TO_NH4),
                      om.initial_state((), 256, 512), T([s[5] for s in spec["soil"]]) * 100.0,
                      T([s[6] for s in spec["soil"]]) * 100.0, nm.initial_disturbance((), L))
    th = lay.thickness
    rows = []
    for moist, ice, tt, ops in spec["days"]:
        liquid = T(moist) / (th * 1000.0)
        st = st._replace(residues=om.start_day(st.residues))
        upt = None
        for o in ops:
            if o[0] in (1, 2):
                P = om.residue_params(dict(zip(keys, o[-1])))
                if o[0] == 1:
                    st = st._replace(residues=om.add_surface(st.residues, one, o[1], T(o[2]), T(o[3]), P))
                else:
                    st = st._replace(residues=om.add_subsurface(st.residues, one, T(o[1]), T(o[2]), P))
            elif o[0] == 3:
                st = nm.field_operation(st, one, lay, dict(sdr[o[1]], name=o[1]), (o[2], o[3]))
            elif o[0] == 4:
                st = st._replace(n=sn.fertilizer(st.n, one, lay, T(o[1]), T(o[2]), T(o[3]), T(o[4])))
            else:
                upt = (o[1], o[2])
        env = sn.abiotic_update(liquid, T(ice) / (th * 1000.0), T(tt), lay)
        res, n, dec = om.process(st.residues, env, st.n)
        if upt is not None:
            z = torch.zeros(1, dtype=torch.float64)
            n, _, _ = sn.uptake(n, torch.cat([z, T(upt[0][1:])]), torch.cat([z, T(upt[1][1:])]))
        n, _, _ = sn.transformations(n, lay, env, T(tt), dec.co2)
        st = st._replace(n=n, residues=om.balance(res))
        aw = sn._macro_between((liquid - lay.pwp) / (lay.fc - lay.pwp), torch.zeros_like(liquid), torch.ones_like(liquid))
        st = st._replace(disturbance=nm.process_disturbance(st.disturbance, aw, st.clay))
        rows.append(dict(no3=sn.no3_n(st.n), nh4=sn.nh4_n(st.n), co2=dec.co2, mn=dec.mineralized,
                         contact=st.residues.flat_contact, pools=_plist(st.residues.sur) + _plist(st.residues.sub),
                         sand=st.sand, clay=st.clay, rating=st.disturbance.rating))
    return rows


def _l1_parse(line, L):
    a = line.split()
    i = 2
    f = lambda k: [float.fromhex(x) for x in a[k:k + L]]
    r = dict(no3=f(i), nh4=f(i + L), co2=f(i + 2 * L), mn=f(i + 3 * L))
    i += 4 * L
    r["contact"] = float.fromhex(a[i])
    n = int(a[i + 1])
    i += 2
    pools = []
    for _ in range(n):
        b = a[i:i + 11]
        pools.append(dict(type=int(b[0]), position=int(b[1]), layer=int(b[2]), cycling=int(b[3]), env=int(b[4]),
                          mass=float.fromhex(b[5]), cfrac=float.fromhex(b[6]), cn=float.fromhex(b[7]),
                          k=float.fromhex(b[8]), kdet=float.fromhex(b[9]), apm=float.fromhex(b[10])))
        i += 11
    r["pools"] = pools
    sc = [float.fromhex(x) for x in a[i:i + 2 * L]]
    r["sand"], r["clay"] = sc[0::2], sc[1::2]
    r["rating"] = [float.fromhex(x) for x in a[i + 2 * L:i + 3 * L]]
    return r


@pytest.mark.parametrize("seed", range(6))
def test_l1_soil_day_loop_against_the_compiled_reference(seed):
    """Oracle comparison: random soils, drivers and operation sequences (residue additions, NRCS field
    operations, inorganic fertilizer, crop uptake) through the soil N / residue day loop of the
    compiled reference (oracle SOIL mode: VIC soil, Chemicals_profile, Organic_matter_residues_
    profile_common, Soil_tillage_effect) and of the port: mineral N, CO2 and N mineralization by
    layer, flat contact fraction, every residue pool, texture and disturbance rating to 1e-12."""
    import subprocess
    import _paths
    exe = _paths.oracle("ref_cropsyst_nitrogen")
    if not os.path.exists(exe):
        pytest.skip("oracle ref_cropsyst_nitrogen not built")
    line, spec = _l1_case(seed)
    r = subprocess.run([exe], input=line, capture_output=True, text=True, timeout=600)
    ref = [_l1_parse(x, spec["L"]) for x in r.stdout.splitlines() if x.startswith("D ")]
    assert len(ref) == len(spec["days"]), r.stdout[-2000:] + r.stderr[-2000:]
    mine = _l1_python(spec)
    worst = 0.0
    for d, (x, y) in enumerate(zip(mine, ref)):
        for k in ("no3", "nh4", "co2", "mn", "sand", "clay", "rating"):
            worst = max(worst, _rel(x[k], y[k], 1e-15))
        worst = max(worst, _rel(x["contact"], y["contact"]), _cmp_pools(x["pools"], y["pools"], (seed, d)))
        assert worst <= 1e-12, (seed, d, worst)
