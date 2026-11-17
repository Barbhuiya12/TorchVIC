"""Replay of the CRB orchard cases (315604 winegrape, 318377 apple) through the
crop connector, nitrogen off and on.

The reference runs are those of cropsyst_i8_l2 (<data root>/verification/cropsyst_i8i13/runs_off
and runs_N; the dump binary with patch P4, the null guard of Crop_nitrogen_common::restart for the
canopy an orchard drops at dormancy). The reference's water stress, leaf water potential and
(nitrogen on) VIC layer state are the drivers; everything else is the port. `compare` returns, per
crop csv column, the worst relative error on all days (ET max from day 1: on the sowing day the
reference reports its ET reference object before its first update), the growth stage of every day,
and the harvest, bud break and dormancy days.
"""
import csv
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cropsyst_i8_l2 as L  # noqa: E402

CASES = (315604, 318377)
COLS = {"GAI": ("gai", 1.0), "Biomass_kg_m2": ("biomass", 1.0), "Root_depth_mm": ("root_depth", 1000.0),
        "Green_Canopy_Cover": ("green_cover", 1.0), "crop_evapotransp_max_mm": ("et_max", 1.0),
        "Yield_kg_m2": ("yield", 1e-4), "Fruit_dry_biomass_kg_m2": ("fruit_biomass", 1.0),
        "chill_hours_remaining": ("chill_remaining", 1.0)}
# get_growth_stage_description (growth_stages.c:66-96) for fruit trees with dormancy, spaces dropped
PHRASES = {"germination": 2, "emergence": 3, "restart": 4, "Activegrowth": 5, "canopy(culmination)": 6,
           "flowering(anthesis)": 7, "fructescence(init)": 8, "veraison": 9, "fructescence(rapid)": 10,
           "maturity": 11, "harvestable": 12, "harvest": 13, "dormant": 15, "terminated": 16}


def ref_csv(case, nitrogen, runs=None):
    f = glob.glob(f"{runs or L.runs_dir(nitrogen)}/Liu_Linux_{case}_*/VIC_output/vic_crop_daily*")[0]
    rows = list(csv.reader(open(f)))
    H = {h: i for i, h in enumerate(rows[0])}
    return H, rows[1:]


def stage_of(label):
    for item in label.split("&"):
        if item in PHRASES:
            return PHRASES[item]
    raise ValueError(f"no growth stage in {label!r}")


def compare(case, nitrogen, device="cpu"):
    inp = L.inputs(case, nitrogen)
    out, _ = L.tc._run(L.connector(inp, nitrogen), inp["nd"], device=device)
    return compare_out(case, nitrogen, inp, out)


def compare_out(case, nitrogen, inp, out, runs=None):
    """The comparison of `compare` for a run already made (out: the connector series)."""
    H, rows = ref_csv(case, nitrogen, runs)
    nd = inp["nd"]
    assert len(rows) == nd and inp["T"] == 1
    worst = {c: 0.0 for c in COLS}
    where = {c: None for c in COLS}
    for d, r in enumerate(rows):
        for c, (key, fac) in COLS.items():
            if c == "crop_evapotransp_max_mm" and d == 0:
                continue
            ref = float(r[H[c]])
            x = float(out[key][d, 0]) * fac
            e = abs(ref - x) / max(abs(ref), 1e-6)
            e = e if not math.isnan(e) else 1e9
            if e > worst[c]:
                worst[c], where[c] = e, d
    st_ref = [stage_of(r[H["Grow_Stage"]]) for r in rows]
    st = [int(out["stage"][d, 0]) for d in range(nd)]
    bad_stage = [d for d in range(nd) if st[d] != st_ref[d]]
    harv_ref = [d for d, r in enumerate(rows) if float(r[H["Yield_kg_m2"]]) > 0.0]
    harv = [d for d in range(nd) if out["harvest"][d, 0] > 0]
    sown = [d for d in range(nd) if out["sown"][d, 0] > 0]
    res = dict(worst=worst, where=where, bad_stage=bad_stage[:10], n_bad_stage=len(bad_stage),
               harvest_ok=harv == harv_ref, harvest=harv, harvest_ref=harv_ref, sow_ok=sown == [0])
    if nitrogen:
        bad = n = 0
        t = inp["tiles"][0]
        for d in range(nd):
            e = inp["rec"].get((inp["year"][d], inp["doy"][d], int(t)), {}).get("E")
            if e is None:
                continue
            got = out["soil_no3_n"][d, 0].tolist() + out["soil_nh4_n"][d, 0].tolist()
            for a, b in zip(got, e["no3"] + e["nh4"]):
                n += 1
                if math.isnan(a) != math.isnan(b) or (not math.isnan(b) and abs(a - b) > 1e-6 * max(abs(b), 1e-6)):
                    bad += 1
        res.update(n_bad=bad, n_checked=n)
    return res


if __name__ == "__main__":
    for case in [int(x) for x in sys.argv[2:]] or CASES:
        for nit in ([sys.argv[1] == "N"] if len(sys.argv) > 1 else (False, True)):
            try:
                print(case, "N" if nit else "off", compare(case, nit), flush=True)
            except Exception as e:  # noqa
                import traceback
                traceback.print_exc()
                print(case, "N" if nit else "off", "ERROR", type(e).__name__, str(e)[:300], flush=True)
