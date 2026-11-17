"""Orchard and perennial fruit crops of the CropSyst backend (orchard.py).

Oracle (tests/oracle/ref_cropsyst_orchard.cpp, linked against the compiled VIC-CropSyst
reference): the reference parser's view of the orchard crop files, and whole multi-year orchard
runs (dormancy, chill, bud break, the catch-all quiescence end of the first winter, restart, fruit
growth to the maximum load, harvest by event and by the fruit harvest requirements, all four
harvest condition modes, grapes, both hemispheres, starts in and out of winter) of randomised crop
files through the port's engine, with the replayed water stress. Bar 1e-12 (relative) on every
field of every day.

Replay (reference runs <data root>/verification/cropsyst_i8i13, tests/cropsyst_orchard_l2.py):
315604 winegrape and 318377 apple through the crop connector, nitrogen off and on.
"""
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths  # noqa: E402

from torchvic.crop.cropsyst import engine as cse  # noqa: E402
from torchvic.crop.cropsyst import orchard as oc  # noqa: E402
from torchvic.crop.cropsyst import phenology as ph  # noqa: E402
from torchvic.crop.cropsyst import canopy as cn  # noqa: E402
from torchvic.crop.cropsyst import readers as rd  # noqa: E402

ORACLE = _paths.oracle("ref_cropsyst_orchard")
SIM = _paths.env_path(('TORCHVIC_CROPSYST_CRB',), 'build/vic_cropsyst/VIC-CropSyst_v3_CRB/Simulation')
DB = os.path.join(SIM, "Database")
TOL = 1e-12
FIELDS = ["stage", "gdds", "gdd", "tadj", "et_max", "pot", "attainable", "limited", "act_tr", "act_growth",
          "ws", "wsi", "canopy", "gai", "gai_ve", "lai", "lai_ss", "green", "biomass", "root_len",
          "root_depth", "root_bm", "fruit", "reserves", "chill_rem", "chill_on", "bb_on", "qend_on", "qend_n",
          "qstart_on", "qstart_n", "harvestable", "harvested", "latest", "max_load", "limit_lai", "em_count",
          "acc_on", "acc_tt", "culm_on", "culm_tt", "sen_on", "sen_tt", "root_tt", "harvest", "fh_on", "fh_n"]


def _need_oracle():
    if not os.path.exists(ORACLE):
        pytest.skip("ref_cropsyst_orchard oracle not built (tests/oracle/build_ref_cropsyst_orchard.sh)")
    if not os.path.isdir(DB):
        pytest.skip("CRB database not present")


def _parse(line):
    return [float.fromhex(x) if ("0x" in x or "p" in x) else float(x) for x in line.split()]


# ----------------------------------------------------------------------------- parameters

def test_orchard_params_equal_the_reference_parser():
    """orchard_params() and the fruit fields of phenology_params() on the CRB orchard files (and
    variants of the keys the parser treats specially) equal the compiled reference parser."""
    _need_oracle()
    tmp = tempfile.mkdtemp()
    try:
        base = [os.path.join(DB, "Crop", "Code", "2002.CS_crop"), os.path.join(DB, "Crop", "Code", "1401.CS_crop")]
        paths = list(base)
        text = open(base[0], encoding="latin-1").read()
        variants = [("grape=grape", "grape=true"), ("grape=grape", "grape=1"),
                    ("harvest_condition=management_scheduled", "harvest_condition=consider_avg_temperature"),
                    ("life_cycle=perennial_woody", "life_cycle=perennial"),
                    ("season_dormancy_mode=dormancy", "season_dormancy_mode=multiple"),
                    ("season2_adj=1.00000000", "season2_adj=0.75")]
        for i, (a, b) in enumerate(variants):
            p = os.path.join(tmp, f"v{i}.crp")
            open(p, "w", encoding="latin-1").write(text.replace(a, b))
            paths.append(p)
        r = subprocess.run([ORACLE], input="".join(f"CRP {p}\n" for p in paths), capture_output=True,
                           text=True, timeout=600)
        rows = [_parse(l) for l in r.stdout.splitlines()]
        assert len(rows) == len(paths), r.stderr[-2000:]
        crops = rd.crop_table(paths)
        OP = oc.orchard_params(crops)
        PP = ph.phenology_params(crops)
        for i, row in enumerate(rows):
            assert bool(OP.is_fruit[i]) == bool(row[6]) and bool(OP.perennial[i]) == bool(row[7])
            assert int(OP.continuation[i]) == int(row[3])
            assert bool(PP.has_leaf_senescence[i]) == bool(row[9])
            got = [float(OP.chill_hours[i]), float(OP.bb_sat[i]), float(OP.bb_not_sat[i]), float(OP.senesce[i]),
                   float(OP.q_temp[i]), int(OP.q_days[i]), int(PP.grape[i]), float(OP.fts[i]), float(OP.mfl[i]),
                   int(OP.h_mode[i]), float(OP.h_temp[i]), int(OP.h_days[i]), int(OP.h_early[i]),
                   int(OP.h_late[i]), float(OP.igf[i]), float(OP.rgf[i]), float(PP.rapid_fruit[i])]
            assert got == row[13:30], (paths[i], got, row[13:30])
            assert OP.season_adj[i].tolist() == row[31:37], (paths[i], OP.season_adj[i].tolist(), row[31:37])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- oracle seasons

def _set(text, key, value):
    """Replace the value of `key=` (first occurrence, the comment after the number dropped)."""
    pat = re.compile(r"^(%s=)[^\n]*$" % re.escape(key), re.M)
    assert pat.search(text), key
    return pat.sub(lambda m: m.group(1) + str(value), text, count=1)


def make_crop(rng, template):
    """A randomised orchard crop file (valid for the port: dormancy with 100 % senescence)."""
    t = template
    em = rng.uniform(60, 300)
    fl = em + rng.uniform(-40, 150)
    peak = em + rng.uniform(250, 600)
    fill = max(fl, em) + rng.uniform(100, 500)
    rapid = fill + rng.uniform(100, 400)
    mat = rapid + rng.uniform(100, 400)
    sen = max(peak, mat) + rng.uniform(50, 400)
    full = sen + rng.uniform(80, 300)
    vals = {"emergence": round(em), "flowering": round(fl), "peak_LAI": round(peak), "filling": round(fill),
            "maturity": round(mat, 2), "senescence": round(sen, 2), "full_senescence": round(full, 2),
            "rapid_fruit_growth": round(rapid, 2), "base_temp": round(rng.uniform(3, 8), 2),
            "cutoff_temp": round(rng.uniform(24, 32), 2),
            "chill_requirement": round(rng.choice([0, rng.uniform(0, 1200)]), 1),
            "deg_day_bud_break_chill_req_sat": round(rng.uniform(40, 200), 1),
            "inducement_temperature": round(rng.uniform(3, 12), 2), "consider_inactive_days": rng.randint(3, 9),
            "max_LAI": round(rng.uniform(0.4, 2.5), 2), "fract_total_solids": round(rng.uniform(0.15, 0.4), 3),
            "max_fruit_load": round(rng.choice([rng.uniform(500, 4000), rng.uniform(1e4, 4e5)]), 1),
            "initial_growth_fraction_to_fruit": round(rng.uniform(0.3, 0.95), 3),
            "rapid_growth_fraction_to_fruit": round(rng.uniform(0.3, 0.95), 3),
            "grape": rng.choice(["true", "false", "grape"]),
            "harvest_condition": rng.choice(["management_scheduled", "consider_fixed_date",
                                             "consider_min_temperature", "consider_avg_temperature"]),
            "harvest_temperature": round(rng.uniform(5, 18), 2), "harvest_temperature_days": rng.randint(1, 6),
            "earliest_harvest": rng.randint(200, 260), "latest_harvest": rng.randint(262, 330),
            "season2_adj": round(rng.uniform(0.7, 1.3), 3), "season3_adj": round(rng.uniform(0.7, 1.3), 3),
            "ET_crop_coef": round(rng.uniform(0.8, 1.6), 2), "initial_GAI": round(rng.uniform(0.05, 0.4), 3)}
    for k, v in vals.items():
        t = _set(t, k, v)
    return t


def weather(rng, n_day, doy0, south):
    tmax, tmin, srad, rh, et, delta, gamma, ra, uf, lw, doy = ([] for _ in range(11))
    off = rng.uniform(-4, 4)
    for d in range(n_day):
        j = (doy0 - 1 + d) % 365 + 1
        season = math.cos(2 * math.pi * (j - 200) / 365.0) * (-1 if south else 1)
        tx = 14 + off + 14 * season + rng.gauss(0, 4)
        tn = tx - rng.uniform(5, 16)
        tmax.append(tx), tmin.append(tn), doy.append(j)
        srad.append(max(20.0, 180 + 140 * season + rng.gauss(0, 40)))
        rh.append(rng.uniform(15, 80)), et.append(max(0.1, 3 + 3 * season + rng.gauss(0, 0.8)))
        delta.append(rng.uniform(0.05, 0.25)), gamma.append(rng.uniform(0.06, 0.07)), ra.append(rng.uniform(0.001, 0.004))
        uf.append(rng.choice([1.0, rng.uniform(0.2, 1.0)])), lw.append(rng.uniform(-2200, -300))
    return dict(tmax=tmax, tmin=tmin, srad=srad, rhmin=rh, et_ref=et, delta=delta, gamma=gamma, ra=ra,
                ufrac=uf, lwp=lw, doy=doy)


def _case_set(seed, n_case, doy0, n_day):
    rng = random.Random(seed)
    template = open(os.path.join(DB, "Crop", "Code", "2002.CS_crop"), encoding="latin-1").read()
    mgt_t = open(os.path.join(DB, "Management", "Code", "2002.mgt"), encoding="latin-1").read()
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "Crop", "Code"))
    os.makedirs(os.path.join(tmp, "Management", "Code"))
    cases = []
    for k in range(n_case):
        code = 2100 + k
        crop = make_crop(rng, template)
        mgt = _set(mgt_t, "event_offset", rng.randint(0, 12))
        mgt = _set(mgt, "min_LAI", round(rng.choice([0.0, 0.5, rng.uniform(0.5, 3.0)]), 2))
        mgt = _set(mgt, "min_biomass", round(rng.choice([0.0, rng.uniform(0, 8000)]), 1))
        mgt = _set(mgt, "event_phenologic(fruit)", rng.choice(["maturity", "rapid_fruit_development"]))
        open(os.path.join(tmp, "Crop", "Code", f"{code}.CS_crop"), "w", encoding="latin-1").write(crop)
        open(os.path.join(tmp, "Management", "Code", f"{code}.mgt"), "w", encoding="latin-1").write(mgt)
        south = rng.random() < 0.3
        w = weather(rng, n_day, doy0, south)
        stage = oc.FRUIT_STAGE_LABELS.index(re.search(r"event_phenologic\(fruit\)=(\S+)", mgt).group(1))
        cases.append(dict(code=code, south=south, w=w, mgt=mgt, stage=stage,
                          offset=int(re.search(r"^event_offset=(\d+)", mgt, re.M).group(1)),
                          min_lai=float(re.search(r"^min_LAI=(\S+)", mgt, re.M).group(1)),
                          min_bm=float(re.search(r"^min_biomass=(\S+)", mgt, re.M).group(1))))
    return tmp, cases


THICK = [0.1, 0.3, 0.4, 0.5, 0.7]


def port_run(tmp, cases, n_day, year0):
    T = len(cases)
    st = lambda k: torch.tensor([[c["w"][k][d] for c in cases] for d in range(n_day)], dtype=torch.float64)
    wx = {k: st(k) for k in ("tmax", "tmin", "srad", "rhmin", "et_ref", "delta", "gamma", "ra")}
    wx["co2"] = 370.0
    doy = cases[0]["w"]["doy"]
    year, y = [], year0
    for d in range(n_day):
        if d and doy[d] == 1:
            y += 1
        year.append(y)
    L = len(THICK)
    soil = dict(thickness=THICK, max_moist=[t * 450.0 for t in THICK], air_entry=[-2.0] * L, campbell_b=[5.0] * L)
    replay = dict(act_frac=st("ufrac"), lwp=st("lwp"))
    cfg = dict(db_dir=tmp, tiles=[c["code"] for c in cases], tile_cell=list(range(T)), soil=soil, replay=replay)
    lat = [(-40.0 if c["south"] else 45.0) for c in cases]
    run = cse.build_run(cfg, wx, lat, [-120.0] * T, year, doy, T, "cpu")
    state, rows = run.state, []
    for d in range(n_day):
        state, out = run.step(state)
        g, p, o = state.grow, state.phen, state.orc
        c, r = g.canopy, g.root
        neg = torch.full_like(p.gdds, -1.0)
        on = lambda k: p.period_on[:, k]
        vals = [p.stage.double(), p.gdds, p.gdd, p.tmax_adj, g.et_max, g.pot_unadj, g.attainable, g.limited_pot,
                g.act_transp, g.act_growth, g.ws, g.ws_index, c.exists.double(),
                torch.where(c.exists, cn.gai_vital(c), 0 * neg), torch.where(c.exists, cn.gai_vital(c), 0 * neg),
                torch.where(c.exists, cn.lai(c), 0 * neg), torch.where(c.exists, c.lai_ss, 0 * neg),
                torch.where(c.exists, cn.interception_green(c, run.S.gp.kc[state.cid]), 0 * neg),
                torch.where(c.exists, cn.biomass_vital(c), 0 * neg), r.length, out.root_depth, r.biomass,
                o.fruit_bm, o.reserves, torch.where(o.chill_on, o.chill_rem, 0 * neg), o.chill_on.double(),
                o.bb_on.double(), o.qend_on.double(), torch.where(o.qend_on, o.qend_n, 0 * o.qend_n).double(),
                o.qstart_on.double(), torch.where(o.qstart_on, o.qstart_n, 0 * o.qstart_n).double(),
                o.harvestable.double(), o.harvested.double(), o.latest_yield, o.max_load.double(),
                o.limit_lai.double(), o.em_count.double(),
                on(ph.ACC).double(), torch.where(on(ph.ACC), p.period_tt[:, ph.ACC], neg),
                on(ph.CULM).double(), torch.where(on(ph.CULM), p.period_tt[:, ph.CULM], neg),
                on(ph.SEN).double(), torch.where(on(ph.SEN), p.period_tt[:, ph.SEN], neg),
                torch.where(on(ph.ROOT), p.period_tt[:, ph.ROOT], neg), out.harvest.double(),
                o.fh_on.double(), torch.where(o.fh_on, o.fh_n, 0 * o.fh_n).double()]
        hours = oc.air_hours(o)
        rows.append(torch.cat([torch.stack(vals, -1), hours, r.fract], -1))
    return torch.stack(rows, 0), run


def oracle_run(tmp, cases, n_day, run):
    lines = []
    tmin_tom = run.tables.tmin_tom
    vpd = cse.dl.gr.vpd_max_from_rhmin(run.tables.tmax, run.tables.rhmin)
    for k, c in enumerate(cases):
        w = c["w"]
        head = [f"SEASON {tmp}/Crop/Code/{c['code']}.CS_crop", str(n_day), str(len(THICK)), str(int(c["south"])), "1"]
        head += [repr(float(x)) for x in THICK]
        head += [str(c["stage"]), str(c["offset"]), repr(c["min_lai"]), repr(c["min_bm"])]
        days = []
        for d in range(n_day):
            v = [float(w["doy"][d]), w["tmax"][d], w["tmin"][d], float(tmin_tom[d, k]),
                 float(run.tables.srad[d, k]), float(vpd[d, k]), float(run.tables.et_ref[d, k]),
                 w["delta"][d], w["gamma"][d], w["ra"][d], w["ufrac"][d], w["lwp"][d]]
            days.append(" ".join(float(x).hex() for x in v))
        lines.append(" ".join(head) + " " + " ".join(days) + "\n")
    r = subprocess.run([ORACLE], input="".join(lines), capture_output=True, text=True, timeout=3600)
    rows = [_parse(l) for l in r.stdout.splitlines()]
    assert len(rows) == n_day * len(cases), r.stderr[-3000:]
    t = torch.tensor(rows, dtype=torch.float64).reshape(len(cases), n_day, -1)
    return t.transpose(0, 1)


def compare(ref, got):
    names = FIELDS + [f"hour{h}" for h in range(24)] + [f"fract{l}" for l in range(len(THICK))]
    assert ref.shape == got.shape, (ref.shape, got.shape)
    err = (ref - got).abs() / ref.abs().clamp(min=1.0)
    bad = torch.nonzero(err > TOL)
    if bad.numel():
        d, k, f = bad[0].tolist()
        return float(err.max()), (d, k, names[f], float(ref[d, k, f]), float(got[d, k, f]))
    return float(err.max()), None


@pytest.mark.parametrize("seed,doy0,n_day", [(11, 1, 3 * 365), (12, 140, 3 * 365), (13, 300, 2 * 365)],
                         ids=["winter-start", "summer-start", "autumn-start"])
def test_l1_orchard_seasons_equal_the_reference(seed, doy0, n_day):
    _need_oracle()
    tmp, cases = _case_set(seed, 12, doy0, n_day)
    try:
        got, run = port_run(tmp, cases, n_day, 2001)
        ref = oracle_run(tmp, cases, n_day, run)
        worst, where = compare(ref, got)
        assert where is None, (worst, where)
        # the runs exercise what they are for
        st = ref[..., FIELDS.index("stage")]
        assert bool((st == ph.NGS_QUIESCENCE).any()) and bool((ref[..., FIELDS.index("harvest")] > 0).any())
        assert bool((ref[..., FIELDS.index("max_load")] > 0).any())
        assert bool((ref[..., FIELDS.index("limit_lai")] > 0).any())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- replay

@pytest.mark.reference
@pytest.mark.parametrize("nitrogen", [False, True], ids=["N-off", "N-on"])
@pytest.mark.parametrize("case", (315604, 318377))
def test_l2_orchard_cases_through_the_connector(case, nitrogen):
    import glob
    import cropsyst_orchard_l2 as OL
    if not glob.glob(f"{OL.L.runs_dir(nitrogen)}/Liu_Linux_{case}_*/VIC_output/vic_crop_daily*"):
        pytest.skip(f"reference run of {case} not present")
    r = OL.compare(case, nitrogen)
    assert r["sow_ok"] and r["harvest_ok"], r
    assert r["n_bad_stage"] == 0, r["bad_stage"]
    for c, w in r["worst"].items():
        assert w < 2e-7, (c, w, r["where"][c])
    if nitrogen:
        assert r["n_checked"] > 40000 and r["n_bad"] == 0, (r["n_bad"], r["n_checked"])


# ----------------------------------------------------------------------------- GPU

def _gpu_run(nitrogen, cells, nday, device, db=None, arch=None):
    """The two CRB orchard crops on `cells` cells each, a seasonal synthetic climate from 1 January,
    VIC-like layer moisture (E1 uptake, not a replay). db/arch: another CropSyst Database and the
    orchard interception model (test_cropsyst_orchard_arch.py)."""
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
    cfg = dict(db_dir=db or DB, tiles=[2002, 1401] * cells, tile_cell=sorted(list(range(cells)) * 2), soil=soil,
               nitrogen=nitrogen)
    if arch is not None:
        cfg["orchard_interception_model"] = arch
    lat = torch.full((cells,), 46.0, dtype=torch.float64)
    lon = torch.full((cells,), -119.0, dtype=torch.float64)
    run = cse.build_run(cfg, w, lat, lon, year, doy, cells, torch.device(device))
    g = torch.Generator(device="cpu").manual_seed(3)
    moist = (run.S.max_moist * (0.45 + 0.3 * torch.rand(T, L, generator=g, dtype=torch.float64).to(device)))
    soil_t = torch.full((T, L), 12.0, dtype=torch.float64, device=device) if nitrogen else None
    return run, moist.contiguous(), soil_t


@pytest.mark.gpu
@pytest.mark.parametrize("nitrogen", [False, True], ids=["N-off", "N-on"])
def test_gpu_orchard_day_step_no_host_sync_graph_and_compile_equal_eager(nitrogen):
    """Orchard tiles on CUDA through a whole year (dormancy, chill, bud break, fruit, harvest,
    dormancy again): every eager step under set_sync_debug_mode('error') (a host sync raises), the
    run equals the CPU run, and on days with a restart and with a dormancy start a CUDA graph replay
    and a torch.compile'd step equal the eager step."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not os.path.isdir(DB):
        pytest.skip("CRB database not present")
    gpu_check(nitrogen)


def gpu_check(nitrogen, **kw):
    """The checks of test_gpu_orchard_day_step_no_host_sync_graph_and_compile_equal_eager; kw: _gpu_run."""
    nday, cells = 400, 64
    runs = {d: _gpu_run(nitrogen, cells, nday, d, **kw) for d in ("cpu", "cuda")}
    states = {}
    events = []
    for dev, (run, moist, soil_t) in runs.items():
        st = run.state
        if dev == "cuda":
            st, _ = run.step(st, moist, None, None, soil_t)        # one day first: tables on the device
            torch.cuda.set_sync_debug_mode("error")
        else:
            st, _ = run.step(st, moist, None, None, soil_t)
        keep = []
        try:
            for d in range(1, nday):
                prev = st
                st, out = run.step(st, moist, None, None, soil_t)
                if dev == "cuda":
                    keep.append((prev, out.orc.restart, out.orc.dormant))
        finally:
            if dev == "cuda":
                torch.cuda.set_sync_debug_mode("default")
        states[dev] = st
        if dev == "cuda":
            events = keep
    for a, b in zip(cse.flatten(states["cpu"]), cse.flatten(states["cuda"])):
        b = b.cpu()
        if a.dtype.is_floating_point:
            ok = torch.isclose(a, b, rtol=1e-11, atol=1e-13) | (torch.isnan(a) & torch.isnan(b))
            assert bool(ok.all()), (a[~ok][:4], b[~ok][:4])
        else:
            assert bool((a == b).all())
    run, moist, soil_t = runs["cuda"]
    rs = [i for i, (_, r, _) in enumerate(events) if bool(r.any())]
    ds = [i for i, (_, _, dm) in enumerate(events) if bool(dm.any()) and not bool(events[i - 1][2].any())]
    assert rs and ds, "the year had no restart or no dormancy start"
    step = lambda s: run.step(s, moist, None, None, soil_t)
    same = lambda a, c: bool(((a == c) | (torch.isnan(a) & torch.isnan(c))).all())
    cstep = torch.compile(step, dynamic=False)
    for i in (rs[0], ds[0]):
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
