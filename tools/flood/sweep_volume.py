"""Controlled-volume sweep of LISFLOOD-FP solvers (run on a GPU node).  Prints one JSON line per run."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import lisflood_run as L

N, DX = 32, 10.0
SOLVERS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(L.SOLVERS)
T = 200.0


def dem(kind):
    x = np.arange(N)[None, :] * np.ones((N, 1))
    if kind == "flat": return np.full((N, N), 10.0)
    if kind == "slope": return 10.0 - 0.01 * x * DX          # falls toward east
    if kind == "basin":
        yy, xx = np.mgrid[0:N, 0:N]
        return 10.0 + 0.002 * ((xx - N / 2) ** 2 + (yy - N / 2) ** 2) ** 0.5 * DX
    raise ValueError


def exp_cases():
    """(tag, dem_kind, kwargs, expected_inflow_volume_m3 or {hypothesis: V})."""
    px = py = N * DX / 2 - 5   # cell centre
    out = []
    out.append(("pt_QFIX_1", "flat", dict(bci=[f"P {px} {py} QFIX 1.0"]), {"m3/s": 1.0 * T, "m2/s": 1.0 * DX * T}))
    ser = {"s1": ("seconds", [(1.0, 0.0), (1.0, T * 2)])}
    out.append(("pt_QVAR_1", "flat", dict(bci=[f"P {px} {py} QVAR s1"], bdy=ser), {"m3/s": 1.0 * T, "m2/s": 1.0 * DX * T}))
    out.append(("edge_QFIX_W", "flat", dict(bci=[f"W 100 200 QFIX 0.5"]), {"m2/s*width": 0.5 * 100 * T, "m3/s": 0.5 * T}))
    ser2 = {"q": ("seconds", [(0.5, 0.0), (0.5, T * 2)])}
    out.append(("edge_QVAR_W", "flat", dict(bci=["W 100 200 QVAR q"], bdy=ser2), {"m2/s*width": 0.5 * 100 * T, "m3/s": 0.5 * T}))
    h = {"h": ("seconds", [(10.2, 0.0), (10.2, T * 2)])}
    out.append(("edge_HVAR_W", "flat", dict(bci=["W 100 200 HVAR h"], bdy=h), {"head_driven": None}))
    out.append(("edge_HFIX_W", "flat", dict(bci=["W 100 200 HFIX 10.2"]), {"head_driven": None}))
    rain_mm_hr = 36.0   # 1e-5 m/s -> 0.002 m in 200 s
    area = N * N * DX * DX
    out.append(("rain_ts", "flat", dict(rain_ts=[(rain_mm_hr, 0.0), (rain_mm_hr, T * 2)]), {"mm/hr": rain_mm_hr / 3.6e6 * T * area}))
    nt = 5
    dyn = np.ones((nt, N, N)) * 1.0           # 1 mm at every time level, levels 100 s apart
    out.append(("rain_dyn", "flat", dict(rain_dyn=(dyn, np.arange(nt) * 100.0 / 3600.0)),
                {"1mm_per_100s": 0.002 * area, "1mm_per_level_shifted": 0.003 * area}))
    return out


def run(tag, dkind, kw, hyp, solver, T=T, edges=()):
    bci = list(kw.get("bci", [])) + list(edges)
    c = L.Case(tag, solver, dem(dkind), DX, T, bci=bci, bdy=kw.get("bdy", {}),
               rain_ts=kw.get("rain_ts"), rain_dyn=kw.get("rain_dyn"), massint=T / 20, saveint=T, initial_tstep=1.0)
    r = L.run_case(c)
    rec = dict(tag=tag, dem=dkind, solver=solver, ok=r.ok, wall=round(r.wall, 2), hyp=hyp)
    if r.ok:
        rec.update(ledger_vol=r.ledger_volume, raster_vol=r.stored_volume if r.depth is not None else None)
        if r.mass:
            rec.update(qin_last=float(r.mass["Qin"][-1]), qout_last=float(r.mass["Qout"][-1]),
                       qin_int=r.inflow_volume(), qout_int=r.outflow_volume())
    else:
        rec["log"] = r.log[-300:]
    L.cleanup(r)
    return rec


if __name__ == "__main__":
    for s in SOLVERS:
        for tag, dk, kw, hyp in exp_cases():
            print(json.dumps(run(tag, dk, kw, hyp, s)), flush=True)
        # outflow: sloping plane, point source upstream, FREE east edge, closed else
        kw = dict(bci=["P 45 145 QFIX 1.0", "E 0 300 FREE 0.01"])
        print(json.dumps(run("slope_free_E", "slope", kw, {"m3/s": 1.0 * 600}, s, T=600.0)), flush=True)
        kw = dict(bci=["P 155 155 QFIX 1.0"], rain_ts=[(36.0, 0.0), (36.0, 1200.0)])
        print(json.dumps(run("basin_pt_rain", "basin", kw, {"sum": 1.0 * 200 + 36.0 / 3.6e6 * 200 * N * N * DX * DX}, s)), flush=True)
        kw = dict(bci=["P 155 155 QFIX 1.0", "E 0 300 FREE 0.0", "W 0 300 FREE 0.0", "N 0 300 FREE 0.0", "S 0 300 FREE 0.0"])
        print(json.dumps(run("flat_free_all", "flat", kw, {"m3/s": 1.0 * 600}, s, T=600.0)), flush=True)
