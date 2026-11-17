"""The crop connector's backend 'cropsyst' (mode A), the E10 daily order, restart.

CPU:
  * the default backend stays WOFOST; nitrogen=True needs the soil N parameters;
  * the sowing calendar and the harvest rule reproduce the sowing and harvest days of the CRB
    N-off reference runs (tiles of 330492, 308200, 328617);
  * the replay through the connector (crop csv weather, the reference's water stress and
    leaf water potential in place of E1) equals the direct replay harness of test_cropsyst_growth
    bit for bit, and meets its bar against the reference;
  * E1 from VIC-like layer moisture runs, is finite, and a restart at mid-season equals the
    continuous run bitwise (connector level and runtime level with a checkpoint file);
GPU (marker gpu): the connector on CUDA equals CPU, and one engine day is captured in a
CUDA graph and replayed bit-equal to eager (no host syncs).
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths  # noqa: E402,F401
import test_cropsyst_growth as tg  # noqa: E402

from torchvic.coupling import TorchVIC, registry  # noqa: E402
from torchvic.coupling.connector import Context, resolve_problems  # noqa: E402
from torchvic.coupling.connectors.crop import CropConnector  # noqa: E402
from torchvic.crop.cropsyst import engine as cse  # noqa: E402

DAY = 86400.0
SIM = tg.SIM
DB = os.path.join(SIM, "Database")
CASES = tg.L2_CASES


# ----------------------------------------------------------------------------- reference inputs

def _l2_inputs(case, runs=None):
    """The reference crop csv of a CRB N-off rerun as connector inputs."""
    f = glob.glob(f"{runs or tg.L2_RUNS}/Liu_Linux_{case}_*/VIC_output/vic_crop_daily*")
    if not f:
        pytest.skip(f"reference crop csv of {case} not present")
    with open(f[0]) as fh:
        rows = list(csv.reader(fh))
    H = {h: i for i, h in enumerate(rows[0])}
    data = defaultdict(list)
    for r in rows[1:]:
        data[r[H["CroppingSyst_code"]]].append(r)
    tiles = sorted(data)
    nd, T = len(data[tiles[0]]), len(tiles)
    col = lambda name, fn=float: [[fn(data[k][d][H[name]]) for k in tiles] for d in range(nd)]
    soil = open(glob.glob(f"{SIM}/Database/Soil/soil_param_id_{case}_*")[0]).read().split()
    thick = [float(x) for x in soil[78:95]]
    dpath = os.path.join(os.path.dirname(f[0]), "i6_dump.txt")
    if not os.path.exists(dpath):
        pytest.skip("i6 dump of the reference run not present")
    dump = {}
    with open(dpath) as fh:
        for line in fh:
            a = line.split()
            dump[(int(a[0]), int(a[1]), a[2])] = [float.fromhex(x) for x in a[3:]]
    year, doy = col("Year", int), col("DOY", int)
    lab = col("Grow_Stage", str)
    crop = col("Crop_name", str)
    lwp = [[dump.get((year[d][k], doy[d][k], tiles[k]), [0.0])[0] for k in range(T)]
           for d in range(nd)]
    one = lambda name, fn=float: [fn(data[tiles[0]][d][H[name]]) for d in range(nd)]
    wx = dict(tmax=one("Tmax"), tmin=one("Tmin"), srad=one("SWRAD_w_m2"),
              rhmin=one("RHUM_min_%"), et_ref=one("CropSyst_refET_mm"), co2=370.0)
    # the replay harness uses the reference ET of the day before (first day: its own); the engine
    # keeps yesterday's value itself, so it is given today's
    return dict(
        case=case, tiles=tiles, nd=nd, T=T, wx=wx,
        year=[y[0] for y in year], doy=[x[0] for x in doy],
        act_frac=[[1.0 - w for w in r] for r in col("water_stress_index")], lwp=lwp,
        soil=dict(thickness=thick, quartz=[float(x) for x in soil[114:131]],
                  clay=[float(x) for x in soil[273:290]]),
        lat=float(rows[1][H["lat"]]), lon=float(rows[1][H["lon"]]),
        crop=crop, lab=lab)


def _connector(inp, device="cpu", replay=True, nitrogen=False, **kw):
    cs = dict(db_dir=DB, tiles=[int(t) for t in inp["tiles"]], tile_cell=[0] * inp["T"],
              soil=inp["soil"], lon=[inp["lon"]])
    if nitrogen:
        cs["nitrogen"] = True
    if replay:
        cs["replay"] = dict(act_frac=inp["act_frac"], lwp=inp["lwp"])
    wx = dict(inp["wx"], year=inp["year"], doy=inp["doy"])
    return CropConnector(backend="cropsyst", cropsyst=cs, weather=wx, lat=[inp["lat"]],
                         n_cells=1, **kw)


def _run(conn, n_days, device="cpu", moisture=None, stop=None):
    ctx = Context(device=torch.device(device), dtype=torch.float64)
    conn.initialize(ctx)
    return _advance(conn, ctx, 0, n_days if stop is None else stop, moisture)


def _advance(conn, ctx, d0, d1, moisture=None):
    series = defaultdict(list)
    for d in range(d0, d1):
        ctx.t = d * DAY
        ins = {} if moisture is None else dict(soil_moisture=moisture[d])
        p = conn.compute(ins, ctx, DAY)
        assert not resolve_problems(conn.validate(p, ctx))
        conn.commit(p, ctx)
        for k, v in p.exports.items():
            series[k].append(v.clone())
    return {k: torch.stack(v) for k, v in series.items()}, ctx


# ----------------------------------------------------------------------------- defaults and errors

def test_default_backend_is_wofost_and_registry_has_both():
    c = CropConnector(params={"x": 1}, lat=0.0, n_cells=1)
    assert c.backend == "wofost" and c.engine is None and c.cs is None
    assert registry.get("crop").status == "available"
    assert {v.name for v in c.outputs} >= {"lai", "root_depth", "dvs", "yield", "crop_active", "rftra"}


def test_nitrogen_true_is_accepted_and_needs_the_soil_n_parameters():
    """nitrogen=True runs crop and soil nitrogen (tests/test_cropsyst_n_engine.py); it needs
    the soil's bulk density, wcr, wpwp, quartz and clay; a non-bool is refused."""
    CropConnector(backend="cropsyst", cropsyst=dict(nitrogen=True, db_dir=DB))
    with pytest.raises(cse.InputError, match="nitrogen must be True or False"):
        CropConnector(backend="cropsyst", cropsyst=dict(nitrogen="yes", db_dir=DB))
    inp = _l2_inputs(CASES[0])
    with pytest.raises(cse.InputError, match="needs soil bulk_density"):
        _run(_connector(inp, nitrogen=True), 1)


def test_unknown_backend_and_stray_cropsyst_block_are_refused():
    from torchvic.coupling.spec import CouplingError
    with pytest.raises(CouplingError, match="backend"):
        CropConnector(backend="nope")
    with pytest.raises(CouplingError, match="cropsyst"):
        CropConnector(params={}, cropsyst=dict(db_dir=DB))


# ----------------------------------------------------------------------------- calendar and harvest

@pytest.mark.parametrize("case", CASES)
def test_calendar_and_harvest_rule_reproduce_the_reference_days(case):
    inp = _l2_inputs(case)
    out, _ = _run(_connector(inp), inp["nd"])
    T, nd = inp["T"], inp["nd"]
    for k in range(T):
        ref_sow = [d for d in range(nd) if inp["crop"][d][k] != "NOCROP"
                   and (d == 0 or inp["crop"][d - 1][k] == "NOCROP")]
        ref_harv = [d for d in range(nd) if inp["crop"][d][k] != "NOCROP"
                    and "harvest" in inp["lab"][d][k]]
        got_sow = [d for d in range(nd) if out["sown"][d, k] > 0]
        got_harv = [d for d in range(nd) if out["harvest"][d, k] > 0]
        assert got_sow == ref_sow, (inp["tiles"][k], got_sow, ref_sow)
        assert got_harv == ref_harv, (inp["tiles"][k], got_harv, ref_harv)
        active = [d for d in range(nd) if out["crop_active"][d, k] > 0]
        ref_act = [d for d in range(nd) if inp["crop"][d][k] != "NOCROP"
                   and "harvest" not in inp["lab"][d][k]]
        assert active == ref_act


# ----------------------------------------------------------------------------- replay through the connector

# harness column -> (connector export, factor taking the export to the harness unit)
_COLS = {"GAI": ("gai", 1.0), "Biomass_kg_m2": ("biomass", 1.0), "Root_depth_mm": ("root_depth", 1000.0),
         "Green_Canopy_Cover": ("green_cover", 1.0), "crop_evapotransp_max_mm": ("et_max", 1.0),
         "Act_Transp_mm": ("act_transp", 1.0), "Yield_kg_m2": ("yield", 1e-4)}


def _pairs_from_connector(inp, out, ref_pairs):
    """The (day, tile, ref, got) lists of the direct harness with `got` read from the connector's
    exports, and per column whether every value is bit-identical to the direct harness's."""
    tix = {t: i for i, t in enumerate(inp["tiles"])}
    pairs, same = {}, {}
    for c, lst in ref_pairs.items():
        key, k = _COLS[c]
        new, eq = [], True
        for d, tile, ref, got_direct in lst:
            v = float(out[key][d, tix[tile]])
            if c == "Yield_kg_m2":          # kg ha-1 = kg m-2 * 1e4, the same product as the export's
                eq &= (v == got_direct * 1e4)
                v = v / 1e4
            else:
                v = v * k                   # m -> mm for the root depth
                eq &= (v == got_direct)
            new.append((d, tile, ref, v))
        pairs[c], same[c] = new, eq
    return pairs, same


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_l2_through_the_connector_equals_the_direct_l2(case):
    ref_pairs = tg._l2_case(case)                     # the direct harness (test_cropsyst_growth)
    inp = _l2_inputs(case)
    out, _ = _run(_connector(inp), inp["nd"])
    pairs, same = _pairs_from_connector(inp, out, ref_pairs)
    assert all(same.values()), same
    st = tg.l2_stats(pairs)
    for c in ("GAI", "Biomass_kg_m2", "Green_Canopy_Cover", "crop_evapotransp_max_mm",
              "Root_depth_mm", "Yield_kg_m2"):
        n, worst, _, _ = st[c]
        assert n > 0 and worst < 1e-7, (c, st[c])
    assert st["Yield_kg_m2"][0] >= 4 and st["GAI"][0] > 1400


# ----------------------------------------------------------------------------- E1 mode A and restart

def _vic_like_moisture(inp, nd):
    """A deterministic layer-moisture series [day][cell, layer] mm standing in for VIC: a wet
    profile that dries through the season."""
    thick = torch.tensor(inp["soil"]["thickness"], dtype=torch.float64)
    sx = cse.saxton_soil(torch.tensor(inp["soil"]["quartz"]), torch.tensor(inp["soil"]["clay"]), thick)
    frac = 0.55 + 0.35 * torch.sin(torch.arange(nd, dtype=torch.float64) / 40.0)
    return [(sx["max_moist"] * frac[d].clamp(0.3, 0.95))[None, :] for d in range(nd)]


def test_e1_from_layer_moisture_runs_and_restart_is_bitwise():
    inp = _l2_inputs(330492)
    nd = 520
    moist = _vic_like_moisture(inp, nd)
    a = _connector(inp, replay=False)
    full, _ = _run(a, nd, moisture=moist)
    for k, v in full.items():
        assert torch.isfinite(v).all(), k
    assert float(full["lai"].max()) > 0.5 and float(full["crop_active"].sum()) > 100
    assert float(full["act_transp"].max()) > 0 and float(full["uptake"].abs().sum()) > 0
    # restart at mid-season: the blob goes through torch.save like a checkpoint file
    b = _connector(inp, replay=False)
    s1, ctx = _run(b, nd, moisture=moist, stop=200)
    blob = b.save_state()
    assert blob["started"]
    import io
    buf = io.BytesIO()
    torch.save(blob, buf)
    buf.seek(0)
    c = _connector(inp, replay=False)
    cctx = Context(device=torch.device("cpu"), dtype=torch.float64)
    c.initialize(cctx)
    c.load_state(torch.load(buf, weights_only=False))
    s2, _ = _advance(c, cctx, 200, nd, moist)
    for k in full:
        assert torch.equal(torch.cat([s1[k], s2[k]]), full[k]), k
    # every state tensor equal too
    for x, y in zip(cse.flatten(a.cs.state), cse.flatten(c.cs.state)):
        assert torch.equal(x, y)


def test_proposal_does_not_change_the_accepted_state():
    inp = _l2_inputs(330492)
    moist = _vic_like_moisture(inp, 400)
    c = _connector(inp, replay=False)
    ctx = Context(device=torch.device("cpu"), dtype=torch.float64)
    c.initialize(ctx)
    _advance(c, ctx, 0, 150, moist)
    before = [t.clone() for t in cse.flatten(c.cs.state)]
    ctx.t = 150 * DAY
    p = c.compute(dict(soil_moisture=moist[150]), ctx, DAY)
    assert all(torch.equal(x, y) for x, y in zip(before, cse.flatten(c.cs.state)))
    ctx.t = 151 * DAY
    from torchvic.coupling.connector import ComputeError
    with pytest.raises(ComputeError):
        c.compute(dict(soil_moisture=moist[150]), ctx, DAY)       # the accepted state is at 150
    c.commit(p, ctx)
    assert c.day == 151


def test_runtime_checkpoint_restart_equals_continuous(tmp_path):
    inp = _l2_inputs(330492)
    n = 260

    def runtime():
        cs = dict(db_dir=DB, tiles=[int(t) for t in inp["tiles"]], tile_cell=[0] * inp["T"],
                  soil=inp["soil"], lon=[inp["lon"]],
                  replay=dict(act_frac=inp["act_frac"], lwp=inp["lwp"]))
        wx = dict(inp["wx"], year=inp["year"], doy=inp["doy"])
        return TorchVIC.from_dict(dict(
            run=dict(device="cpu", log=False, duration_days=n),
            modules=dict(crop=dict(connector="crop", backend="cropsyst", cropsyst=cs,
                                   weather=wx, lat=[inp["lat"]], n_cells=1))))

    with torch.inference_mode():
        a = runtime().initialize()
        a.run()
        b = runtime().initialize()
        b.run(until=130 * DAY)
        ck = b.save_checkpoint(str(tmp_path / "cs.pt"))
        c = runtime().initialize()
        c.load_checkpoint(ck)
        c.run()
        oa, ob, oc = a.outputs("crop"), b.outputs("crop"), c.outputs("crop")
    assert oa["lai"].shape[0] == n and float(oa["lai"].max()) > 0
    for k in oa:
        assert torch.equal(oa[k][:130], ob[k]), k
        assert torch.equal(oa[k][130:], oc[k]), k


# ----------------------------------------------------------------------------- GPU

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_connector_equals_cpu_and_cuda_graph_capture():
    inp = _l2_inputs(330492)
    nd = 300
    moist = _vic_like_moisture(inp, nd)
    cpu, _ = _run(_connector(inp, replay=False), nd, moisture=moist)
    gpu_c = _connector(inp, replay=False)
    gpu, _ = _run(gpu_c, nd, device="cuda", moisture=[m.cuda() for m in moist])
    for k in cpu:
        a, b = cpu[k], gpu[k].cpu()
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-12), (k, float((a - b).abs().max()))
    # CUDA graph: one engine day over the connector's own run object, 200 replays
    run = _connector(inp, replay=False)
    run.initialize(Context(device=torch.device("cuda"), dtype=torch.float64))
    cs = run.cs
    tc = cs.tile_cell
    mo = [m.cuda()[0][None, :].expand(cs.n_tile, -1).contiguous() for m in moist]
    et = torch.zeros(cs.n_tile, dtype=torch.float64, device="cuda")
    eager = cs.state
    for d in range(60):
        eager, _ = cs.step(eager, mo[d], None, et)
    template = cs.state
    buf = [t.clone() for t in cse.flatten(template)]
    m_in = mo[0].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            cs.step(cse.unflatten(template, buf), m_in, None, et)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        new, out = cs.step(cse.unflatten(template, buf), m_in, None, et)
        for b, o in zip(buf, cse.flatten(new)):
            b.copy_(o)
    for b, x in zip(buf, cse.flatten(template)):
        b.copy_(x)
    for d in range(60):
        m_in.copy_(mo[d])
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(cse.flatten(eager), buf):
        assert bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


# ----------------------------------------------------------------------------- coupled to VIC

def test_cropsyst_backend_coupled_to_vic_stehekin_leaves_vic_alone():
    """Mode A on VIC's Stehekin sample: weather from the case's forcing, layer moisture linked from
    vic.soil_moisture (lag 1) and the reference ET from vic.pet; VIC's own results are those of
    the VIC-only run."""
    import gc
    import test_conn_agri as ta
    if not (os.path.exists(ta.STEHE) and os.path.isdir(ta.STEHE_DATA)):
        pytest.skip("VIC's Stehekin sample is not present")
    from torchvic.coupling.vic import VicConnector
    drv = ta._driver("cpu")
    nd = drv.n_steps // drv.steps_per_day
    cs = dict(db_dir=DB, rotations=[3001] * drv.n_cells, soil=dict(
        thickness=[0.1, 0.3, 0.6], quartz=[0.4, 0.4, 0.4], clay=[0.2, 0.2, 0.2]))
    links = [{"from": "vic.soil_moisture", "to": "crop.soil_moisture", "lag": 1},
             {"from": "vic.pet", "to": "crop.pet"}]

    def run(with_crop):
        conns = dict(vic=VicConnector(driver=drv, exchange="day", soil_add_mode="cap"))
        mods = dict(vic=dict(connector="vic", global_param=ta.STEHE))
        if with_crop:
            conns["crop"] = CropConnector(driver=drv, backend="cropsyst", cropsyst=cs)
            mods["crop"] = dict(connector="crop")
        m = TorchVIC.from_dict(dict(run=dict(device="cpu", log=False, duration_days=nd),
                                    modules=mods, connections=links if with_crop else []),
                               connectors=conns)
        with torch.inference_mode():
            m.initialize()
            m.run()
            out = {k: m.outputs(k) for k in mods}
        return out

    gc.collect()
    try:
        a, b = run(False), run(True)
    finally:
        ta._DRIVERS.clear()             # the case's open netCDF files
        gc.collect()
    for k in a["vic"]:
        if torch.is_tensor(a["vic"][k]):
            assert torch.equal(a["vic"][k], b["vic"][k]), k
    for k, v in b["crop"].items():
        assert torch.isfinite(v).all(), k
    assert b["crop"]["lai"].shape[0] == nd
