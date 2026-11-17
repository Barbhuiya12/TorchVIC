"""Replay of the CRB apple case 318377 with the orchard canopy architecture on
(crop file `[morphology] canopy_architecture=true`, tall-spindle geometry), each interception model
(scenario orchard_interception_model none / Kemanian-Stockle / Oyarzun-Stockle), nitrogen off and on.

Reference runs: <data root>/verification/cropsyst_apple_arch/runs_<model>_<off|N> (the dump binary
plus patch P5 and print-only dump lines in the three orchard units, `arch_dump.txt`: per crop
process_day the canopy extent "E tt begin end width breadth height", the model line, and "S has_model
interception"). The drivers are those of cropsyst_orchard_l2 (the reference's water stress, leaf
water potential and, with nitrogen, the layer state); the crop file comes from the run's own
Database (CROPSYST_WORKING_DIR). `compare` returns the crop csv comparison of cropsyst_orchard_l2
plus the worst relative error of the interception and of the canopy width, breadth and height on
every day.
"""
import glob
import math
import os
import _paths  # noqa: E402
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cropsyst_i8_l2 as L  # noqa: E402
import cropsyst_orchard_l2 as O  # noqa: E402

ROOT = _paths.env_path(('TORCHVIC_CROPSYST_APPLE_ARCH',), 'verification/cropsyst_apple_arch')
CASE = 318377
MODELS = {"oy": "Oyarzun-Stockle", "ks": "Kemanian-Stockle", "none": "none"}


def runs_dir(model, nitrogen):
    return os.path.join(ROOT, f"runs_{model}_{'N' if nitrogen else 'off'}")


def read_dump(model, nitrogen):
    """Per day (one S line per crop process_day): (interception, width, breadth, height or None)."""
    f = glob.glob(f"{runs_dir(model, nitrogen)}/Liu_Linux_{CASE}_*/VIC_output/arch_dump.txt")[0]
    days, ext = [], None
    for line in open(f):
        a = line.split()
        if a[0] == "E":
            ext = tuple(float(x) for x in a[4:7])
        elif a[0] == "S":
            days.append((float(a[2]), ext))
            ext = None
    return days


def connector(inp, nitrogen, model):
    from torchvic.coupling.connectors.crop import CropConnector
    db = os.path.join(ROOT, "Simulation", "Database")
    cs = dict(db_dir=db, tiles=[int(t) for t in inp["tiles"]], tile_cell=[0] * inp["T"],
              soil=inp["soil"], lon=[inp["lon"]], replay=dict(act_frac=inp["act_frac"], lwp=inp["lwp"]),
              orchard_interception_model=MODELS[model])
    if nitrogen:
        cs["nitrogen"] = True
        cs["replay"].update(inp["replay_n"])
    wx = dict(inp["wx"], year=inp["year"], doy=inp["doy"])
    return CropConnector(backend="cropsyst", cropsyst=cs, weather=wx, lat=[inp["lat"]], n_cells=1)


def compare(model, nitrogen, device="cpu"):
    runs = runs_dir(model, nitrogen)
    inp = L.inputs(CASE, nitrogen, runs=runs)
    out, _ = L.tc._run(connector(inp, nitrogen, model), inp["nd"], device=device)
    res = O.compare_out(CASE, nitrogen, inp, out, runs=runs)
    dump = read_dump(model, nitrogen)
    nd = inp["nd"]
    worst = dict(interception=0.0, width=0.0, breadth=0.0, height=0.0)
    where = dict(worst)
    for d in range(nd):
        sci, ext = dump[d]
        pairs = [("interception", sci, float(out["orchard_interception"][d, 0]))]
        if ext is not None:
            for k, r in zip(("width", "breadth", "height"), ext):
                pairs.append((k, r, float(out["canopy_" + k][d, 0])))
        for k, r, x in pairs:
            e = abs(r - x) / max(abs(r), 1e-6)
            e = e if not math.isnan(e) else 1e9
            if e > worst[k]:
                worst[k], where[k] = e, d
    res.update(arch_worst=worst, arch_where=where, arch_days=len(dump) == nd)
    return res


if __name__ == "__main__":
    for model in sys.argv[2:] or MODELS:
        for nit in ([sys.argv[1] == "N"] if len(sys.argv) > 1 else (False, True)):
            try:
                print(model, "N" if nit else "off", compare(model, nit), flush=True)
            except Exception as e:  # noqa
                import traceback
                traceback.print_exc()
                print(model, "N" if nit else "off", "ERROR", type(e).__name__, str(e)[:300], flush=True)
