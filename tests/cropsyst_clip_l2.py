"""Replay of the clipping reference cases through the crop connector, nitrogen off
and on.

Reference runs: <data root>/verification/cropsyst_clip/runs_<case>_<off|N> (cell 330492, 1979-1983,
alfalfa sown each spring on DOY 100; the dump binary with print-only clipping dump lines,
`clip_dump.txt`). The crop and management files are the case's own Database
(Simulation/Database: alfalfa_clip*.crp from the shipped alfalfa, alfalfa_clip_date.mgt with three
cuttings by date and a last clipping, alfalfa_clip_auto.mgt with an automatic clipping period).
Drivers as cropsyst_orchard_l2 (the reference's water stress, leaf water potential and, with
nitrogen, the layer state). `compare` returns the worst relative error of each crop csv column on
every crop day, the growth stage of every day, the clipping days and the relative error of every
clipping's Yield_kg_m2, and (nitrogen on) the soil NO3/NH4 against the dump.
"""
import csv
import glob
import math
import os
import _paths  # noqa: E402
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cropsyst_i8_l2 as L  # noqa: E402
import cropsyst_orchard_l2 as O  # noqa: E402

ROOT = _paths.env_path(('TORCHVIC_CROPSYST_CLIP',), 'verification/cropsyst_clip')
CASE = 330492
CASES = ("date", "auto")
COLS = {"GAI": ("gai", 1.0), "Biomass_kg_m2": ("biomass", 1.0), "Root_depth_mm": ("root_depth", 1000.0),
        "Green_Canopy_Cover": ("green_cover", 1.0), "crop_evapotransp_max_mm": ("et_max", 1.0)}


PHRASES = dict(O.PHRASES, yieldformation=8, ripening=9, quiescent=15)


def stage_of(label):
    for item in label.split("&"):
        if item in PHRASES:
            return PHRASES[item]
    raise ValueError(f"no growth stage in {label!r}")


def runs_dir(case, nitrogen):
    return os.path.join(ROOT, f"runs_{case}_{'N' if nitrogen else 'off'}")


def inputs(case, nitrogen):
    import test_cropsyst_connector as tc
    from cropsyst_n_dump import read_dump, dump_path
    runs = runs_dir(case, nitrogen)
    inp = tc._l2_inputs(CASE, runs=runs)
    inp = _exact_drivers(inp, runs)
    if not nitrogen:
        return inp
    head, rec = read_dump(dump_path(runs, CASE))
    tiles, nd = inp["tiles"], inp["nd"]
    h = head[int(tiles[0])]
    soil = dict(inp["soil"], thickness=h["thickness"], bulk_density_g_cm3=h["bulk_density"],
                field_capacity=h["fc"], wilting_point=h["pwp"], saturation=h["sat"])
    Ln = len(h["thickness"])
    drv = {k: [] for k in ("moist", "ice", "T")}
    for d in range(nd):
        row = {k: [] for k in drv}
        for t in tiles:
            p = rec.get((inp["year"][d], inp["doy"][d], int(t)), {}).get("P")
            for k in drv:
                row[k].append(p[k] if p else [0.0] * Ln)
        for k in drv:
            drv[k].append(row[k])
    return dict(inp, soil=soil, replay_n=dict(moist=drv["moist"], ice=drv["ice"], soil_t=drv["T"]),
                head=head, rec=rec)


def _exact_drivers(inp, runs):
    """The crop csv's drivers (Tmax, Tmin, shortwave, RH min, CropSyst reference ET, water stress
    index) at full precision from the dump binary's clip_wx.txt (W lines), when present: the csv
    prints 8 digits, enough for 5e-8 but not for decisions that compare rounding residuals."""
    f = glob.glob(f"{runs}/Liu_Linux_{CASE}_*/VIC_output/clip_wx.txt")
    if not f:
        return inp
    rec = {}
    for line in open(f[0]):
        a = line.split()
        if a[0] == "W":
            rec[(int(a[1]), int(a[2]))] = [float.fromhex(x) for x in a[4:10]]
    nd = inp["nd"]
    wx = dict(inp["wx"])
    for k, j in (("tmax", 0), ("tmin", 1), ("srad", 2), ("rhmin", 3), ("et_ref", 4)):
        wx[k] = [rec[(inp["year"][d], inp["doy"][d])][j] for d in range(nd)]
    act = [[1.0 - rec[(inp["year"][d], inp["doy"][d])][5]] for d in range(nd)]
    return dict(inp, wx=wx, act_frac=act)


def connector(inp, nitrogen):
    from torchvic.coupling.connectors.crop import CropConnector
    db = os.path.join(ROOT, "Simulation", "Database")
    cs = dict(db_dir=db, tiles=[int(t) for t in inp["tiles"]], tile_cell=[0] * inp["T"],
              soil=inp["soil"], lon=[inp["lon"]], replay=dict(act_frac=inp["act_frac"], lwp=inp["lwp"]),
              clip_mode="exact")                   # the reference's loss of retained portions
    if nitrogen:
        cs["nitrogen"] = True
        cs["replay"].update(inp["replay_n"])
    wx = dict(inp["wx"], year=inp["year"], doy=inp["doy"])
    return CropConnector(backend="cropsyst", cropsyst=cs, weather=wx, lat=[inp["lat"]], n_cells=1)


def run(case, nitrogen, device="cpu"):
    import test_cropsyst_connector as tc
    inp = inputs(case, nitrogen)
    out, _ = tc._run(connector(inp, nitrogen), inp["nd"], device=device)
    return inp, out


def compare(case, nitrogen, device="cpu", out=None, inp=None):
    if out is None:
        inp, out = run(case, nitrogen, device)
    f = glob.glob(f"{runs_dir(case, nitrogen)}/Liu_Linux_{CASE}_*/VIC_output/vic_crop_daily*")[0]
    rows = list(csv.reader(open(f)))
    H = {h: i for i, h in enumerate(rows[0])}
    rows = rows[1:]
    nd = inp["nd"]
    assert len(rows) == nd and inp["T"] == 1
    worst = {c: 0.0 for c in COLS}
    where = {c: None for c in COLS}
    crop_days = [d for d, r in enumerate(rows) if r[H["Crop_name"]] != "NOCROP"]
    for d in crop_days:
        r = rows[d]
        for c, (key, fac) in COLS.items():
            if c == "crop_evapotransp_max_mm" and (d == 0 or rows[d - 1][H["Crop_name"]] == "NOCROP"):
                continue                       # the sowing day reports the ET object before its update
            if key not in out:
                continue
            ref = float(r[H[c]])
            x = float(out[key][d, 0]) * fac
            e = abs(ref - x) / max(abs(ref), 1e-6)
            e = e if not math.isnan(e) else 1e9
            if e > worst[c]:
                worst[c], where[c] = e, d
    st_ref = {d: stage_of(rows[d][H["Grow_Stage"]]) for d in crop_days}
    bad_stage = [d for d in crop_days if int(out["stage"][d, 0]) != st_ref[d]]
    cut_ref = [d for d in range(nd) if float(rows[d][H["Yield_kg_m2"]]) > 0.0]
    cut = [d for d in range(nd) if out["harvest"][d, 0] > 0]
    errs = {d: abs(float(out["yield"][d, 0]) * 1e-4 - float(rows[d][H["Yield_kg_m2"]]))
            / float(rows[d][H["Yield_kg_m2"]]) for d in cut_ref}
    # a removal that takes the whole canopy compares a rounding residual with the GAI to retain
    # (Canopy_accumulation::remove_biomass): with drivers replayed to 1e-8 its sign is not
    # reproducible; such cuts (margin below 1e-12) are reported apart
    edge = [d for d in cut_ref if 0.0 <= float(out["clip_margin"][d, 0]) < 1e-12]
    ye = max([e for d, e in errs.items() if d not in edge] + [0.0])
    ye_edge = {d: errs[d] for d in edge}
    res = dict(worst=worst, where=where, n_bad_stage=len(bad_stage), bad_stage=bad_stage[:10],
               cuts_ok=cut == cut_ref, cuts=len(cut_ref), cut_days=cut[:6], cut_ref=cut_ref[:6], yield_err=ye,
               yield_edge=ye_edge)
    if nitrogen:
        bad = n = bad_pre = 0
        first = None
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
                    first = d if first is None else first
                    if not edge or d <= min(edge):
                        bad_pre += 1
        # the soil after a knife-edge cut gets a different residue: counted apart
        res.update(n_bad=bad, n_checked=n, first_bad_day=first, n_bad_before_edge=bad_pre)
    return res


if __name__ == "__main__":
    for case in sys.argv[2:] or CASES:
        for nit in ([sys.argv[1] == "N"] if len(sys.argv) > 1 and sys.argv[1] != "all" else (False, True)):
            try:
                print(case, "N" if nit else "off", compare(case, nit), flush=True)
            except Exception as e:  # noqa
                import traceback
                traceback.print_exc()
                print(case, "N" if nit else "off", "ERROR", type(e).__name__, str(e)[:300], flush=True)
