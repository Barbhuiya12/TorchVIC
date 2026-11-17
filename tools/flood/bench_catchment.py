"""2000x2000 (2048 for adaptive) one-day benchmark: point source + uniform rain, closed domain, volume check."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import lisflood_run as L

solver = sys.argv[1]; n = int(sys.argv[2]); T = float(sys.argv[3]); dx = 30.0
mode = sys.argv[5] if len(sys.argv) > 5 else 'uniform'
x = np.arange(n)[None, :] * np.ones((n, 1))
dem = 100.0 - 0.001 * x * dx                       # plane falling east, closed walls
Q = 50.0 / dx                                       # per-width m2/s -> 50 m3/s... see sweep: per width
src = f"P {dx*20+1} {n*dx/2} QFIX {Q}"
bci = [src]
rain = [(2.0, 0.0), (2.0, T * 2)]                   # 2 mm/hr uniform
adaptive = solver.startswith(("hw", "mw"))
dyn = None
if mode == "dyn" and not adaptive:
    nl = int(T // 3600) + 1
    f = np.random.default_rng(0).uniform(0, 4, (nl, n, n)).astype("float32")   # mm per hour level
    dyn = (f, np.arange(nl) * 1.0)
    rain = None
c = L.Case(f"b_{solver}", solver, dem, dx, T, bci=bci, rain_ts=None if (adaptive or dyn) else rain, rain_dyn=dyn,
           massint=3600.0, saveint=T, initial_tstep=1.0)
t0 = time.time(); r = L.run_case(c, workdir=sys.argv[4]); wall = time.time() - t0
exp = Q * dx * T + (0 if adaptive else (dyn[0][:int(T // 3600)].astype('float64').sum() / 1000 * dx * dx if dyn else 2.0 / 3.6e6 * T * n * n * dx * dx))
print(json.dumps(dict(solver=solver, n=n, T=T, ok=r.ok, wall=round(r.wall, 1), comp_s=r.comp_s, mode=mode, expected=exp,
                      raster=r.stored_volume if r.depth is not None else None, ledger=r.ledger_volume,
                      log=None if r.ok else r.log[-300:])), flush=True)
