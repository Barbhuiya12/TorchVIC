# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""GPU cost of the CropSyst day step at ISC scale: 112,825 cells x crop tiles.

    python benchmarks/bench_cropsyst_isc.py --device cuda --mode graph --out r.json

The engine day (torchvic.crop.cropsyst.engine.day_step, nitrogen off or on) is run on synthetic
CRB-like drivers (rotation 3001: 3 tiles per cell, VIC-like layer moisture, constant weather) in
four ways: eager, torch.compile, a CUDA graph, and a compiled step inside a CUDA graph. Time per
day is measured with CUDA events over replays with no host synchronisation inside (eager and graph
steps run under torch.cuda.set_sync_debug_mode('error'), so a host sync would raise). Compare with
VIC alone: 0.076-0.082 s/day on an H200 at ISC scale.
"""
import os
import argparse
import json
import time

import torch

DB = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'build/vic_cropsyst/VIC-CropSyst_v3_CRB/Simulation/Database')


def build(cells, nday, layers, nitrogen, device, portions=400):
    from torchvic.crop.cropsyst import engine as cse
    year = [1980] * nday
    doy = list(range(100, 100 + nday))
    f = lambda v: torch.full((nday,), float(v), dtype=torch.float64)
    w = dict(tmax=f(18.0), tmin=f(6.0), srad=f(180.0), rhmin=f(35.0))
    L = layers
    th = [0.1, 0.3, 0.6] if L == 3 else [0.05] + [0.1] * (L - 1)
    soil = dict(thickness=th, quartz=[0.4] * L, clay=[0.2] * L)
    if nitrogen:
        soil.update(bulk_density_g_cm3=[1.3] * L, field_capacity=[0.3] * L, wilting_point=[0.12] * L,
                    saturation=[0.45] * L)
    cfg = dict(db_dir=DB, rotations=[3001] * cells, soil=soil, nitrogen=nitrogen, first_year=1979)
    lat = torch.full((cells,), 46.0, dtype=torch.float64)
    lon = torch.full((cells,), -119.0, dtype=torch.float64)
    run = cse.build_run(cfg, w, lat, lon, year, doy, cells, torch.device(device), P=portions)
    return run


def drivers(run, device, nitrogen):
    T, L = run.n_tile, run.n_layer
    g = torch.Generator(device="cpu").manual_seed(2)
    mx = run.S.max_moist
    moist = (mx * (0.5 + 0.3 * torch.rand(T, L, generator=g, dtype=torch.float64).to(device))).contiguous()
    et = torch.full((T,), 0.004, dtype=torch.float64, device=device)
    soil_t = torch.full((T, L), 12.0, dtype=torch.float64, device=device) if nitrogen else None
    return moist, et, soil_t


def seed_pools(st, n_sur, n_sub, seed=3):
    """The state with n_sur surface and n_sub subsurface residue pools in use in every tile (the
    CRB replays reach 27-56 and 63-97 after five years): flat and standing straw of the three
    cyclings, subsurface pools in layers 1-3, as harvests and tillage leave them."""
    from torchvic.crop.cropsyst import organic_matter as om
    r = st.n.soil.residues
    g = torch.Generator(device="cpu").manual_seed(seed)
    def fill(p, n, surface):
        T, N = p.active.shape
        dev = p.active.device
        i = torch.arange(N, device=dev).expand(T, N)
        on = i < n
        cyc = (i % 3) + 1
        rnd = torch.rand(T, N, generator=g, dtype=torch.float64).to(dev)
        lay = torch.zeros_like(i) if surface else (i // 3) % 3 + 1
        k = om.halflife_to_constant(torch.tensor([0.0, 3.0, 6.0, 30.0], dtype=torch.float64, device=dev))[cyc]
        return om.Pools(on, torch.zeros_like(i), torch.where(on, (i % 2) + 1 if surface else 0 * i, p.position),
                        lay, cyc, torch.where(lay > 0, lay, 1), 0.05 * rnd + 0.001, torch.full_like(rnd, 0.46),
                        30.0 + 40.0 * rnd, k + 0 * rnd, torch.full_like(rnd, 0.000963 if surface else 0.0),
                        torch.full_like(rnd, 4.0 if surface else 0.0), torch.full_like(p.n, n))
    r = r._replace(sur=fill(r.sur, n_sur, True), sub=fill(r.sub, n_sub, False))
    return st._replace(n=st.n._replace(soil=st.n.soil._replace(residues=r)))


def timed(fn, reps, dev):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def main():
    from torchvic.crop.cropsyst import engine as cse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=112825)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--nitrogen", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modes", default="eager,graph,compile,compile_graph")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--portions", type=int, default=400)
    ap.add_argument("--check", type=int, default=30)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--pools", default=None, help="n_sur,n_sub residue pools in use in every tile (nitrogen)")
    ap.add_argument("--fusion", action="store_true",
                    help="torchvic.core.fusion.setup() before compiling (eager-identical float64 arithmetic)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    if a.fusion:
        from torchvic.core import fusion
        fusion.setup()
    t0 = time.time()
    run = build(a.cells, a.days + 5, a.layers, a.nitrogen, a.device, a.portions)
    moist, et, soil_t = drivers(run, a.device, a.nitrogen)
    res = dict(cells=a.cells, tiles=run.n_tile, layers=a.layers, nitrogen=a.nitrogen, portions=a.portions,
               device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
               torch=torch.__version__, fusion=a.fusion, setup_s=time.time() - t0)
    step = lambda st: run.step(st, moist, None, et, soil_t)
    # warm up eagerly through the sowing days so crops are active
    st = run.state
    for _ in range(40):
        st, _o = step(st)
    if a.pools:
        ps, pb = (int(x) for x in a.pools.split(","))
        st = seed_pools(st, ps, pb)
        res["pools"] = [ps, pb]
    start = st
    flat0 = cse.flatten(start)
    mem = lambda: torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
    for mode in a.modes.split(","):
        try:
            if mode == "eager":
                st = start
                torch.cuda.set_sync_debug_mode("error")
                def f():
                    nonlocal st
                    st, _ = step(st)
                for _ in range(3):
                    f()
                torch.cuda.set_sync_debug_mode("default")
                st = start
                torch.cuda.set_sync_debug_mode("error")
                ms = timed(lambda: f(), a.reps, dev)
                torch.cuda.set_sync_debug_mode("default")
                res["eager_ms_per_day"] = ms
            elif mode == "graph":
                buf = [t.clone() for t in flat0]
                s_ = torch.cuda.Stream()
                s_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_):
                    for _ in range(2):
                        step(cse.unflatten(start, buf))
                torch.cuda.current_stream().wait_stream(s_)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    new, _o = step(cse.unflatten(start, buf))
                    for b, o in zip(buf, cse.flatten(new)):
                        b.copy_(o)
                for b, x in zip(buf, flat0):
                    b.copy_(x)
                for _ in range(3):
                    g.replay()
                res["graph_ms_per_day"] = timed(g.replay, a.reps, dev)
                res["graph_state_tensors"] = len(buf)
                if a.profile:
                    from torch.profiler import profile, ProfilerActivity
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(5):
                            g.replay()
                        torch.cuda.synchronize()
                    rows = prof.key_averages()
                    tot = sum(r.device_time_total for r in rows if r.device_time_total) or 1.0
                    top = sorted(rows, key=lambda r: -r.device_time_total)[:12]
                    res["profile_top_kernels_us_per_day"] = [
                        (r.key[:70], r.device_time_total / 5, r.count // 5) for r in top]
                    res["profile_kernels_per_day"] = sum(r.count for r in rows) // 5
            elif mode in ("compile", "compile_graph"):
                tc0 = time.time()
                cstep = torch.compile(lambda s: run.step(s, moist, None, et, soil_t), dynamic=False)
                st = start
                for _ in range(3):
                    st, _o = cstep(st)
                torch.cuda.synchronize()
                res[mode + "_compile_s"] = time.time() - tc0
                if mode == "compile":
                    st = start
                    def f():
                        nonlocal st
                        st, _ = cstep(st)
                    res[mode + "_ms_per_day"] = timed(f, a.reps, dev)
                    # difference to the eager step after `check` days (inductor scans are not sequential)
                    se, sc = start, start
                    for _ in range(a.check):
                        se, _o = step(se)
                        sc, _o = cstep(sc)
                    worst = 0.0
                    for x, y in zip(cse.flatten(se), cse.flatten(sc)):
                        if x.dtype.is_floating_point:
                            m = torch.isfinite(x) & torch.isfinite(y)
                            if m.any():
                                worst = max(worst, float(((x - y).abs() / x.abs().clamp(min=1e-3))[m].max()))
                    res["compile_vs_eager_max_rel_after_days"] = (a.check, worst)
                    nb = 0
                    for x, y in zip(cse.flatten(se), cse.flatten(sc)):
                        eq = x == y
                        if x.dtype.is_floating_point:
                            eq = (eq & (torch.signbit(x) == torch.signbit(y))) | (torch.isnan(x) & torch.isnan(y))
                        nb += int(not bool(eq.all()))
                    res["compile_vs_eager_tensors_not_bit_equal"] = nb
                else:   # the compiled day captured in a CUDA graph by hand
                    buf = [t.clone() for t in flat0]
                    s_ = torch.cuda.Stream()
                    s_.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s_):
                        for _ in range(2):
                            cstep(cse.unflatten(start, buf))
                    torch.cuda.current_stream().wait_stream(s_)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        new, _o = cstep(cse.unflatten(start, buf))
                        for b_, o in zip(buf, cse.flatten(new)):
                            b_.copy_(o)
                    for b_, x in zip(buf, flat0):
                        b_.copy_(x)
                    for _ in range(3):
                        g.replay()
                    res[mode + "_ms_per_day"] = timed(g.replay, a.reps, dev)
                    if a.profile:
                        from torch.profiler import profile, ProfilerActivity
                        with profile(activities=[ProfilerActivity.CUDA]) as prof:
                            for _ in range(5):
                                g.replay()
                            torch.cuda.synchronize()
                        rows = prof.key_averages()
                        top = sorted(rows, key=lambda r: -r.device_time_total)[:40]
                        res["compile_graph_profile_us_per_day"] = [(r.key[:90], r.device_time_total / 5, r.count // 5) for r in top]
                        res["compile_graph_kernels_per_day"] = sum(r.count for r in rows) // 5
        except Exception as e:  # noqa
            res[mode + "_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        res["peak_gib"] = mem()
        print(mode, {k: v for k, v in res.items() if k.startswith(mode)}, flush=True)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
