"""Clipping of the CropSyst backend (clipping.py), the forage cutting of alfalfa hay.

Oracle (tests/oracle/ref_cropsyst_clip.cpp, linked against the compiled VIC-CropSyst reference):
randomised canopies of the real Canopy_growth_leaf_area_index_based_V5 (growth, senescence die-back)
clipped by process_biomass_fate in every amount mode (fixed amount with a GAI to retain and a reserve,
percent, crop parameter with a yield, retain GAI), newest or oldest first, terminating or not, with
the regrowth canopy of respond_to_clipping, and the canopy N removed; every portion, the untouchable
portion, the self-shading and peak LAI, the cover, the fates and the N to 1e-12 after every step.

Replay (reference runs <data root>/verification/cropsyst_clip, tests/cropsyst_clip_l2.py): alfalfa
on cell 330492 over 1979-1983, three cuttings by date and a last clipping, or an automatic clipping
period, nitrogen off and on, through the crop connector.

GPU: the clipping cases on CUDA with no host sync, equal to the CPU; a CUDA graph replay and
torch.compile equal the eager step on a clipping day; the Triton walk equals the torch walk.
"""
import glob
import math
import os
import random
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths  # noqa: E402

from torchvic.crop.cropsyst import canopy as cn  # noqa: E402
from torchvic.crop.cropsyst import clipping as cl  # noqa: E402
from torchvic.crop.cropsyst import engine as cse  # noqa: E402

ORACLE = _paths.oracle("ref_cropsyst_clip")
CLIP_ROOT = _paths.env_path(('TORCHVIC_CROPSYST_CLIP',), 'verification/cropsyst_clip')
CLIP_DB = os.path.join(CLIP_ROOT, "Simulation", "Database")
TOL = 1e-12
P = 160


def _hex(x):
    return float(x).hex()


def _ops(seed, T):
    """A shared sequence of operation kinds, each tile with its own random parameters."""
    rng = random.Random(seed)
    kinds = ["D"] * rng.randint(20, 60)
    for _ in range(rng.randint(2, 4)):
        kinds += ["K"] if rng.random() < 0.6 else []
        kinds += ["D"] * rng.randint(0, 25)
        kinds += ["F", "R"]
        kinds += ["D"] * rng.randint(3, 30)
    kinds += ["F"]                                   # possibly terminating, last
    tiles = []
    for t in range(T):
        r = random.Random(seed * 1000 + t)
        new = dict(sla=r.choice([15.0, 20.0, 22.0, 28.0]), slp=r.choice([2.0, 2.5, 4.0]),
                   regrowth=r.choice([0.011, 0.02, 0.05]), initial=r.choice([0.011, 0.02, 0.04]),
                   kc=r.choice([0.5, 0.56, 0.6]))
        ops = []
        for i, k in enumerate(kinds):
            if k == "D":
                act = r.choice([0.0, r.uniform(0.0, 0.02), r.uniform(0.0, 0.003)])
                ops.append(dict(cont=int(r.random() < 0.85), act=act, rel=act * r.uniform(0.3, 1.0) + 1e-9))
            elif k == "K":
                ops.append(dict(rate=r.uniform(0.0005, 0.01), gdd=r.uniform(5.0, 25.0)))
            elif k == "F":
                last = i == len(kinds) - 1
                mode = r.choice([0, 1, 1, 2, 2, 3])
                pct = [0] * 10
                use = r.choice([0, 50, 70, 80, 100])
                pct[0] = use
                rest = 100 - use
                for j in (1, 2, 3, 4, 5):
                    v = r.randint(0, rest) if j != 5 else rest
                    pct[j] = v
                    rest -= v
                pct[6], pct[7], pct[8], pct[9] = 0, 0, 100, r.choice([0, 80])
                ops.append(dict(term=int(last and r.random() < 0.6), mode=mode, newest=int(r.random() < 0.7),
                                remove=r.uniform(0.0, 0.3), retain_gai=r.choice([0.0, 0.0, r.uniform(0.0, 1.0)]),
                                retain_bm=r.choice([0.0, r.uniform(0.0, 0.05)]),
                                yld=r.choice([0.0, r.uniform(0.0, 0.05)]) if mode == 0 else 0.0,
                                conc=r.uniform(0.0, 0.05), cn=r.uniform(0.0, 0.01), pct=pct))
            else:
                ops.append(dict(ia=int(r.random() < 0.6)))
        tiles.append((new, ops))
    return kinds, tiles


def _script(new, ops, kinds):
    L = [f"NEW {new['sla']} {new['slp']} {new['regrowth']} {new['initial']} {new['kc']}"]
    for k, o in zip(kinds, ops):
        if k == "D":
            L.append(f"D {o['cont']} {_hex(o['act'])} {_hex(o['rel'])}")
        elif k == "K":
            L.append(f"K {_hex(o['rate'])} {_hex(o['gdd'])}")
        elif k == "F":
            L.append("F %d %d %d %s %s %s %s %s %s %s" % (
                o["term"], o["mode"], o["newest"], _hex(o["remove"]), _hex(o["retain_gai"]), _hex(o["retain_bm"]),
                _hex(o["yld"]), _hex(o["conc"]), _hex(o["cn"]), " ".join(str(x) for x in o["pct"])))
        else:
            L.append(f"R {o['ia']}")
    return "\n".join(L) + "\n"


def _parse_state(line):
    a = line.split()
    assert a[0] == "S"
    n = int(a[1])
    v = [float.fromhex(x) for x in a[2:2 + 4 * n]]
    portions = [tuple(v[4 * i:4 * i + 4]) for i in range(n)]
    k = 2 + 4 * n
    assert a[k] == "U"
    u = (int(a[k + 1]),) + tuple(float.fromhex(x) for x in a[k + 2:k + 6])
    rest = [float.fromhex(x) for x in a[k + 6:]]
    return portions, u, rest


def _f32(x):
    return float(torch.tensor(x, dtype=torch.float32))


def _port_state(c, t):
    pg, pd, pb, pgb = (x[t].tolist() for x in (c.p_gai, c.p_dead, c.p_bm, c.p_gbm))
    portions = [(a, b, d, e) for a, b, d, e in zip(pg, pd, pb, pgb) if (a, b, d, e) != (0.0, 0.0, 0.0, 0.0)]
    has_u = float(c.u_bm[t]) != 0.0 or float(c.u_gai[t]) != 0.0
    u = (int(has_u), float(c.u_gai[t]), float(c.u_dead[t]), float(c.u_bm[t]), float(c.u_gbm[t]))
    rest = [float(c.lai_ss[t]), float(c.peak_lai[t]), float(cn.gai_vital(c)[t]), float(cn.biomass_vital(c)[t]),
            float(c.cover_total[t])]
    return portions, u, rest


def _close(a, b, what):
    assert abs(a - b) <= TOL * max(1.0, abs(a), abs(b)), (what, a, b)


@pytest.mark.reference
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_l1_clipping_canopy_and_fate_equal_the_reference(seed):
    if not os.path.exists(ORACLE):
        pytest.skip("ref_cropsyst_clip oracle not built (tests/oracle/build_ref_cropsyst_clip.sh)")
    T = 12
    kinds, tiles = _ops(seed, T)
    ref = []
    for new, ops in tiles:
        out = subprocess.run([ORACLE], input=_script(new, ops, kinds), capture_output=True, text=True,
                             timeout=600, check=True).stdout.splitlines()
        ref.append(out)
    dt = torch.float64
    t64 = lambda xs: torch.tensor(xs, dtype=dt)
    news = [n for n, _ in tiles]
    sla = t64([_f32(n["sla"]) for n in news])
    slp = t64([_f32(n["slp"]) for n in news])
    regrowth = t64([_f32(n["regrowth"]) for n in news])
    initial = t64([_f32(n["initial"]) for n in news])
    kc = t64([_f32(n["kc"]) for n in news])
    allm = torch.ones(T, dtype=torch.bool)
    c = cn.restart(cn.initial_state((T,), P, dt), allm, initial, sla, kc)
    lines = [0] * T

    def check(fate=None, canopy_n=None, term=None):
        for t in range(T):
            pr, ur, rr = _parse_state(ref[t][lines[t]])
            pp, up, rp = _port_state(c, t)
            assert len(pr) == len(pp), (seed, t, lines[t], len(pr), len(pp))
            for a, b in zip(pr, pp):
                for x, y in zip(a, b):
                    _close(x, y, ("portion", seed, t, lines[t]))
            assert ur[0] == up[0], (seed, t, lines[t], ur, up)
            for x, y in zip(ur[1:], up[1:]):
                _close(x, y, ("untouchable", seed, t, lines[t]))
            if term is None or not bool(term[t]):     # a terminated crop's canopy object is gone in the port
                for x, y in zip(rr, rp):
                    _close(x, y, ("canopy", seed, t, lines[t]))
            lines[t] += 1
            if fate is not None:
                a = ref[t][lines[t]].split()
                assert a[0] == "Fate"
                vals = [float.fromhex(x) for x in a[1:]]
                names = [f for f in cl.Fate._fields if f != "margin"] + ["canopy_n"]
                got = [float(getattr(fate, f)[t]) for f in names[:-1]] + [float(canopy_n[t])]
                assert len(vals) == len(got)
                for x, y, f in zip(vals, got, names):
                    _close(x, y, ("fate", f, seed, t, lines[t]))
                lines[t] += 1

    check()
    for i, k in enumerate(kinds):
        ops = [tiles[t][1][i] for t in range(T)]
        if k == "D":
            cont = torch.tensor([bool(o["cont"]) for o in ops])
            c = cn.develop_leaf(c, allm, cont, t64([o["act"] for o in ops]), t64([o["rel"] for o in ops]),
                                sla, slp, regrowth)
            check()
        elif k == "K":
            c0 = cl._rep(c, sen_known=allm, sen_rate=t64([o["rate"] for o in ops]))
            c = cl._rep(cn.age_leaf(c0, allm, t64([o["gdd"] for o in ops])), sen_known=c.sen_known,
                        sen_rate=c.sen_rate)
            check()
        elif k == "F":
            pct = t64([o["pct"] for o in ops])
            term = torch.tensor([bool(o["term"]) for o in ops])
            mode = torch.tensor([o["mode"] for o in ops])
            c, n_after, fate, _ = cl.biomass_fate(
                c, t64([o["cn"] for o in ops]), t64([o["conc"] for o in ops]), allm, mode,
                t64([o["remove"] for o in ops]), t64([o["retain_gai"] for o in ops]),
                t64([o["retain_bm"] for o in ops]), pct, term, t64([o["yld"] for o in ops]),
                torch.tensor([bool(o["newest"]) for o in ops]), exact=True)
            check(fate, n_after, term)
        else:
            ia = torch.tensor([bool(o["ia"]) for o in ops])
            c = cl.canopy_respond(c, allm, ia, initial, sla, kc)
            check()


def test_fixed_mode_keeps_the_portions_the_gai_to_retain_protects():
    """Upstream bug (Canopy_accumulation::remove_biomass, canopy_growth_portioned.cpp:117-146): a
    portion that would take the living index below the GAI to retain is popped and never put back,
    so it leaves the canopy uncounted (neither removed nor kept). Fixed (default): it stays on the
    plant; exact: lost as in the reference."""
    t = lambda x: torch.tensor([x], dtype=torch.float64)
    P = 4
    c = cn.initial_state((1,), P)
    pad = lambda x: torch.tensor([x + [0.0] * (P - len(x))], dtype=torch.float64)
    c = cl._rep(c, exists=torch.tensor([True]), p_gai=pad([0.4, 0.3, 0.2]), p_bm=pad([0.02, 0.015, 0.01]),
                p_gbm=pad([0.02, 0.015, 0.01]), n=torch.tensor([3]), u_gai=t(0.011), u_bm=t(0.0005), u_gbm=t(0.0005))
    pct = torch.tensor([[100, 0, 0, 0, 0, 0, 0, 0, 100, 80]], dtype=torch.float64)
    args = (c, None, None, torch.tensor([True]), torch.tensor([cl.FIXED]), t(0.03), t(0.5), t(0.0), pct,
            torch.tensor([False]), t(0.0), torch.tensor([True]))
    cf, _, ff, _ = cl.biomass_fate(*args)
    ce, _, fe, _ = cl.biomass_fate(*args, exact=True)
    # newest first: 0.2 (avail 0.911 -> 0.711) and 0.3 (0.711 -> 0.411 < 0.5: kept) ...
    assert abs(float(ff.useful[0]) - 0.01) < 1e-15 and abs(float(fe.useful[0]) - 0.01) < 1e-15
    assert float(cn.biomass_vital(cf)[0]) > 0.035                 # 0.4 and 0.3 portions still there
    assert float(cn.biomass_vital(ce)[0]) < 0.001                 # lost in the reference
    assert float(cn.gai_vital(cf)[0]) >= 0.5


# ----------------------------------------------------------------------------- setup

def test_clipping_events_and_the_automatic_clipping_period_are_read_not_refused():
    """The setup refusal is gone: a management with clipping events and an automatic clipping
    period loads; clipping events take the clipping path, the shipped CRB harvests keep theirs."""
    if not os.path.isdir(CLIP_DB):
        pytest.skip("cs-clip case database not present")
    from torchvic.crop.cropsyst import calendars as cal
    rs = cal.load_rotations(CLIP_DB, [3001, 3002], "cpu", torch.float64, start_doy=1)
    CS = cl.build_static(rs, [1979] * 365, list(range(1, 366)), "cpu")
    assert CS is not None
    i = [k for k, p in enumerate(rs.mgts.paths) if p.endswith("alfalfa_clip_date.mgt")][0]
    assert CS.ev.on[i].tolist()[:4] == [True] * 4
    assert CS.ev.mode[i].tolist()[:4] == [cl.FIXED, cl.PERCENT, cl.CROP_PARAM, cl.PERCENT]
    assert CS.ev.fate[i, 3] == cl.LAST_CLIPPING and bool(CS.ev.terminate[i, 3]) and bool(CS.ev.postpone[i, 0])
    assert abs(float(CS.ev.retain[i, 0]) - _f32(0.3)) == 0.0
    j = [k for k, p in enumerate(rs.mgts.paths) if p.endswith("alfalfa_clip_auto.mgt")][0]
    assert bool(CS.au.on[j, 0]) and abs(float(CS.au.max_bm[j, 0]) - 0.35) < 1e-15
    assert int(CS.au.flower[j, 0]) == 10
    sim = os.path.join(os.path.dirname(CLIP_ROOT), "..", "build", "vic_cropsyst", "VIC-CropSyst_v3_CRB", "Simulation",
                       "Database")
    if os.path.isdir(sim):
        crb = cal.load_rotations(sim, [3001, 3002, 3003], "cpu", torch.float64, start_doy=1)
        assert cl.build_static(crb, [1979], [1], "cpu") is None      # all CRB harvests stay classic


# ----------------------------------------------------------------------------- replay

@pytest.mark.reference
@pytest.mark.parametrize("nitrogen", [False, True], ids=["N-off", "N-on"])
@pytest.mark.parametrize("case", ["date", "auto"])
def test_l2_clipping_cases_through_the_connector(case, nitrogen):
    import cropsyst_clip_l2 as CL
    if not glob.glob(f"{CL.runs_dir(case, nitrogen)}/Liu_Linux_330492_*/VIC_output/vic_crop_daily*"):
        pytest.skip(f"reference run of {case} not present")
    r = CL.compare(case, nitrogen)
    assert r["cuts_ok"] and r["cuts"] >= 10, r            # every clipping day equal
    assert r["n_bad_stage"] == 0, r["bad_stage"]
    assert r["yield_err"] < 2e-7, r["yield_err"]       # every cut but the knife-edge ones (yield_edge)
    assert sum(e > 2e-7 for e in r["yield_edge"].values()) <= 2, r["yield_edge"]   # rounding-sign cuts
    for c, w in r["worst"].items():
        assert w < 2e-7, (c, w, r["where"][c])
    if nitrogen:
        assert r["n_checked"] > 50000 and r["n_bad_before_edge"] == 0, (r["n_bad"], r["first_bad_day"])


# ----------------------------------------------------------------------------- GPU

def _gpu_run(nitrogen, cells, nday, device):
    """The two clipping managements on `cells` cells each, a seasonal synthetic climate from
    1 January, VIC-like layer moisture (E1 uptake)."""
    L = 3
    th = [0.1, 0.3, 0.6]
    soil = dict(thickness=th, quartz=[0.4] * L, clay=[0.2] * L)
    if nitrogen:
        soil.update(bulk_density_g_cm3=[1.3] * L, field_capacity=[0.3] * L, wilting_point=[0.12] * L,
                    saturation=[0.45] * L)
    T = 2 * cells
    doy = [(d % 365) + 1 for d in range(nday)]
    year = [1980 + d // 365 for d in range(nday)]
    j = torch.tensor(doy, dtype=torch.float64)
    season = torch.cos(2 * math.pi * (j - 200) / 365.0)
    noise = torch.sin(j * 1.7) * 3.0
    w = dict(tmax=16 + 14 * season + noise, tmin=3 + 11 * season + noise, srad=180 + 140 * season,
             rhmin=torch.full((nday,), 35.0, dtype=torch.float64), et_ref=(3 + 3 * season).clamp(min=0.2))
    cfg = dict(db_dir=CLIP_DB, tiles=[3001, 3002] * cells, tile_cell=sorted(list(range(cells)) * 2), soil=soil,
               nitrogen=nitrogen)
    lat = torch.full((cells,), 47.0, dtype=torch.float64)
    lon = torch.full((cells,), -117.0, dtype=torch.float64)
    run = cse.build_run(cfg, w, lat, lon, year, doy, cells, torch.device(device), P=P)
    g = torch.Generator(device="cpu").manual_seed(5)
    moist = (run.S.max_moist * (0.45 + 0.3 * torch.rand(T, L, generator=g, dtype=torch.float64).to(device)))
    soil_t = torch.full((T, L), 14.0, dtype=torch.float64, device=device) if nitrogen else None
    return run, moist.contiguous(), soil_t


@pytest.mark.gpu
@pytest.mark.parametrize("nitrogen", [False, True], ids=["N-off", "N-on"])
def test_gpu_clipping_no_host_sync_graph_and_compile_equal_eager(nitrogen):
    """A year of clipped alfalfa tiles on CUDA: every eager step under set_sync_debug_mode('error'),
    the run equals the CPU run, and on a clipping day a CUDA graph replay and a torch.compile'd step
    equal the eager step."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not os.path.isdir(CLIP_DB):
        pytest.skip("cs-clip case database not present")
    nday, cells = 300, 64
    runs = {d: _gpu_run(nitrogen, cells, nday, d) for d in ("cpu", "cuda")}
    states, events, edge = {}, [], {}
    for dev, (run, moist, soil_t) in runs.items():
        st, _ = run.step(run.state, moist, None, None, soil_t)
        if dev == "cuda":
            torch.cuda.set_sync_debug_mode("error")
        keep = []
        try:
            for d in range(1, nday):
                prev = st
                st, out = run.step(st, moist, None, None, soil_t)
                e = out.clip.margin < 1e-12
                edge[dev] = e if dev not in edge else edge[dev] | e
                if dev == "cuda":
                    keep.append((prev, out.clip.event, out.clip.auto, out.clip.killed))
        finally:
            if dev == "cuda":
                torch.cuda.set_sync_debug_mode("default")
        states[dev] = st
        if dev == "cuda":
            events = keep
    # a removal of the whole canopy compares a rounding residual with the GAI to retain (clipping.py):
    # the CPU and CUDA canopy sums differ in the last bit, so tiles with such a knife-edge decision
    # may part; every other tile must agree
    keep_t = ~(edge["cpu"] | edge["cuda"].cpu())
    T = keep_t.shape[0]
    assert int(keep_t.sum()) >= T // 4, int(keep_t.sum())
    for a, b in zip(cse.flatten(states["cpu"]), cse.flatten(states["cuda"])):
        b = b.cpu()
        if a.dim() >= 1 and a.shape[0] == T:
            a, b = a[keep_t], b[keep_t]
        if a.dtype.is_floating_point:
            ok = torch.isclose(a, b, rtol=1e-11, atol=1e-13) | (torch.isnan(a) & torch.isnan(b))
            assert bool(ok.all()), (a[~ok][:4], b[~ok][:4])
        else:
            assert bool((a == b).all())
    ev = [i for i, e in enumerate(events) if bool(e[1].any())]
    au = [i for i, e in enumerate(events) if bool(e[2].any())]
    kl = [i for i, e in enumerate(events) if bool(e[3].any())]
    assert ev and au and kl, "the year had no event clipping, automatic clipping or last clipping"
    run, moist, soil_t = runs["cuda"]
    step = lambda s: run.step(s, moist, None, None, soil_t)
    same = lambda a, c: bool(((a == c) | (torch.isnan(a) & torch.isnan(c))).all())
    from torchvic.core import fusion
    fusion.setup()                       # eager-identical float64 arithmetic (no contraction)
    cstep = torch.compile(step, dynamic=False)
    for i in (ev[0], au[0], kl[0]):
        start = events[i][0]
        e, _ = step(start)
        flat0 = cse.flatten(start)
        buf = [t.clone() for t in flat0]
        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            step(cse.unflatten(start, buf))
        torch.cuda.current_stream().wait_stream(s_)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            new, _o = step(cse.unflatten(start, buf))
            for x, y in zip(buf, cse.flatten(new)):
                x.copy_(y)
        for x, y in zip(buf, flat0):
            x.copy_(y)
        g.replay()
        torch.cuda.synchronize()
        assert all(same(a, c) for a, c in zip(cse.flatten(e), buf))
        c, _ = cstep(start)
        for a, x in zip(cse.flatten(e), cse.flatten(c)):
            if a.dtype.is_floating_point:
                ok = torch.isclose(a, x, rtol=1e-12, atol=1e-14) | (torch.isnan(a) & torch.isnan(x))
                assert bool(ok.all())
            else:
                assert same(a, x)


@pytest.mark.gpu
def test_gpu_walk_kernel_equals_the_torch_walk():
    """The Triton walk of the canopy removal equals the torch walk bit for bit (random canopies,
    newest and oldest first, GAI to retain, partial portions, empty lists)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from torchvic.crop.cropsyst import clip_kernel
    if not clip_kernel.enabled():
        pytest.skip("Triton kernels disabled")
    g = torch.Generator().manual_seed(11)
    T, R = 3000, 81
    cnt = torch.randint(0, R + 1, (T,), generator=g)
    ar = torch.arange(R)
    live = ar[None, :] < cnt[:, None]
    rnd = lambda s: torch.rand((T, R), generator=g, dtype=torch.float64) * s
    pg = torch.where(live, rnd(0.05), 0.0)
    pd = torch.where(live, rnd(0.01), 0.0)
    pb = torch.where(live, rnd(0.003), 0.0)
    pgb = torch.where(live, rnd(0.003), 0.0)
    amount = torch.rand(T, generator=g, dtype=torch.float64) * 0.2 - 0.01
    retain = torch.where(torch.rand(T, generator=g) < 0.5, 0.0, torch.rand(T, generator=g, dtype=torch.float64))
    avail = pg.sum(-1) + torch.rand(T, generator=g, dtype=torch.float64) * 0.02
    newest = torch.rand(T, generator=g) < 0.6
    mask = torch.rand(T, generator=g) < 0.8
    cu = lambda x: x.cuda()
    for lose in (True, False):
        a = cl._walk_torch(pg, pd, pb, pgb, cnt, amount, retain, avail, newest, mask, lose)
        b = clip_kernel.walk(*[cu(x) for x in (pg, pd, pb, pgb, cnt, amount, retain, avail, newest, mask)], lose)
        for x, y in zip(a, b):
            assert torch.equal(x, y.cpu())
