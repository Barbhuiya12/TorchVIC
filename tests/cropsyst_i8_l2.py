"""Replay of the CRB cases through the crop connector, nitrogen on and off.

The reference runs (<data root>/verification/cropsyst_i8i13/runs_N and runs_off, made by the dump
binary build/vic_cropsyst_i8dump.exe: the VIC-CropSyst reference plus patches P3 (no delete of the
land unit at the end of the run, which crashes on a null vtable and loses the crop csv) and P4
(null guard in Crop_nitrogen_common::restart for orchard crops), and the dump of the reference's
leaf water potential, VIC layer moisture/ice/temperature and soil N) are replayed with the
reference's water stress, leaf water potential and VIC layer state as drivers; everything else is
the port. `compare` returns, per column, the worst relative error against the crop csv on crop
days and the sowing/harvest day lists.
"""
import csv
import glob
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths  # noqa: E402,F401
import test_cropsyst_connector as tc  # noqa: E402
from cropsyst_n_dump import read_dump, dump_path  # noqa: E402

RUNS_ROOT = _paths.env_path(('TORCHVIC_CROPSYST_I8_RUNS',), 'verification/cropsyst_i8i13')
CASES = (330492, 308200, 328617, 310974, 259094, 315604, 318377)
COLS = {"GAI": ("gai", 1.0), "Biomass_kg_m2": ("biomass", 1.0), "Root_depth_mm": ("root_depth", 1000.0),
        "Green_Canopy_Cover": ("green_cover", 1.0), "crop_evapotransp_max_mm": ("et_max", 1.0),
        "Yield_kg_m2": ("yield", 1e-4)}


def runs_dir(nitrogen):
    return os.path.join(RUNS_ROOT, "runs_N" if nitrogen else "runs_off")


def inputs(case, nitrogen, runs=None):
    import torch
    runs = runs or runs_dir(nitrogen)
    inp = tc._l2_inputs(case, runs=runs)
    if not nitrogen:
        return inp
    head, rec = read_dump(dump_path(runs, case))
    tiles, nd = inp["tiles"], inp["nd"]
    h = head[int(tiles[0])]
    soil = dict(inp["soil"], thickness=h["thickness"], bulk_density_g_cm3=h["bulk_density"],
                field_capacity=h["fc"], wilting_point=h["pwp"], saturation=h["sat"])
    L = len(h["thickness"])
    drv = {k: [] for k in ("moist", "ice", "T")}
    for d in range(nd):
        row = {k: [] for k in drv}
        for t in tiles:
            p = rec.get((inp["year"][d], inp["doy"][d], int(t)), {}).get("P")
            for k in drv:
                row[k].append(p[k] if p else [0.0] * L)
        for k in drv:
            drv[k].append(row[k])
    return dict(inp, soil=soil, replay_n=dict(moist=drv["moist"], ice=drv["ice"], soil_t=drv["T"]),
                head=head, rec=rec)


def connector(inp, nitrogen):
    c = tc._connector(inp, nitrogen=nitrogen)
    if nitrogen:
        c.cs_cfg["replay"].update(inp["replay_n"])
    return c


def compare(case, nitrogen, device="cpu"):
    inp = inputs(case, nitrogen)
    out, _ = tc._run(connector(inp, nitrogen), inp["nd"], device=device)
    nd = inp["nd"]
    f = glob.glob(f"{runs_dir(nitrogen)}/Liu_Linux_{case}_*/VIC_output/vic_crop_daily*")[0]
    data = list(csv.reader(open(f)))
    H = {h: i for i, h in enumerate(data[0])}
    by = defaultdict(list)
    for r in data[1:]:
        by[r[H["CroppingSyst_code"]]].append(r)
    worst = {c: 0.0 for c in COLS}
    count = {c: 0 for c in COLS}
    sow_ok = harv_ok = True
    for k, t in enumerate(inp["tiles"]):
        ref_sow = [d for d in range(nd) if inp["crop"][d][k] != "NOCROP"
                   and (d == 0 or inp["crop"][d - 1][k] == "NOCROP")]
        ref_harv = [d for d in range(nd) if inp["crop"][d][k] != "NOCROP" and "harvest" in inp["lab"][d][k]]
        sow_ok &= [d for d in range(nd) if out["sown"][d, k] > 0] == ref_sow
        harv_ok &= [d for d in range(nd) if out["harvest"][d, k] > 0] == ref_harv
        for d in range(nd):
            if inp["crop"][d][k] == "NOCROP":
                continue
            hv = d in ref_harv
            for c, (key, fac) in COLS.items():
                if c == "Yield_kg_m2" and not hv:
                    continue
                if hv and c in ("GAI", "Biomass_kg_m2", "Green_Canopy_Cover"):
                    continue
                r = float(by[t][d][H[c]])
                x = float(out[key][d, k]) * fac
                e = abs(r - x) / max(abs(r), 1e-6)
                worst[c] = max(worst[c], e if not math.isnan(e) else 1e9)
                count[c] += 1
    res = dict(worst=worst, count=count, sow_ok=bool(sow_ok), harv_ok=bool(harv_ok))
    if nitrogen:
        bad = n = 0
        for d in range(nd):
            for k, t in enumerate(inp["tiles"]):
                e = inp["rec"].get((inp["year"][d], inp["doy"][d], int(t)), {}).get("E")
                if e is None:
                    continue
                got = out["soil_no3_n"][d, k].tolist() + out["soil_nh4_n"][d, k].tolist()
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
                print(case, "N" if nit else "off", "ERROR", type(e).__name__, str(e)[:300], flush=True)
