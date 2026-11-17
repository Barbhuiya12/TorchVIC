"""Distributed-source checks: rain_dyn volume exactness with a random field, cell orientation, point-source cell."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import lisflood_run as L

N, DX, T = 32, 10.0, 200.0
rng = np.random.default_rng(1)
for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["acc_cpu", "acc_gpu", "fv1_cpu", "fv1_gpu"]):
    # random depth field (mm per 100 s level), 4 levels covering 0..300 s; exact bowl so nothing leaves
    f = rng.uniform(0, 3, (4, N, N))
    yy, xx = np.mgrid[0:N, 0:N]
    bowl = 10.0 + 0.001 * (np.hypot(xx - N / 2, yy - N / 2)) * DX
    rec = dict(solver=s)
    for tag, field in (("rand", f),):
        c = L.Case("dr", s, bowl, DX, T, rain_dyn=(field, np.arange(4) * 100.0 / 3600.0), massint=T / 4)
        r = L.run_case(c)
        if r.ok:
            exp_a = field[:2].sum(axis=0) / 1000 * DX * DX    # intervals [0,100),[100,200)
            rec["rand_raster"] = r.stored_volume
            rec["rand_exp_levels01"] = float(exp_a.sum())
            rec["rand_exp_levels12"] = float(field[1:3].sum() / 1000 * DX * DX)
            rec["rand_ledger"] = r.ledger_volume
        else:
            rec["rand_log"] = r.log[-200:]
        L.cleanup(r)
    # orientation: one spike at array row 3 (top-first y), col 20 on flat DEM, short sim
    spike = np.zeros((3, N, N)); spike[1:, 3, 20] = 50.0
    c = L.Case("or", s, L.flat_dem(N), DX, 100.0, rain_dyn=(spike, np.arange(3) * 50.0 / 3600.0), massint=50.0)
    r = L.run_case(c)
    if r.ok:
        i = np.unravel_index(np.nanargmax(r.depth), r.depth.shape)
        rec["spike_row_col_expect_3_20"] = [int(i[0]), int(i[1])]
    L.cleanup(r)
    # point source cell: P x y with x=205,y=315 -> col 20; row from top = N-1-floor(y/DX)
    c = L.Case("pc", s, L.flat_dem(N), DX, 5.0, bci=["P 205 315 QFIX 1.0"], massint=1.0)
    r = L.run_case(c)
    if r.ok:
        i = np.unravel_index(np.nanargmax(r.depth), r.depth.shape)
        rec["pt_row_col_expect_0_20"] = [int(i[0]), int(i[1])]
    L.cleanup(r)
    print(json.dumps(rec), flush=True)
