"""nitrogen=True in the cropsyst backend (torchvic.crop.cropsyst.n_engine).

Reference (marker reference): the three CRB cases with nitrogen on, run through the crop
connector in replay mode (the reference's water stress, leaf water potential, layer moisture,
ice and temperature as drivers; everything else, including the management events read from the
.mgt files, is the port): every crop day of the original crop csv to print precision (GAI,
biomass, green cover, root depth, ET max, each harvest's yield), sowing and
harvest days, and the soil NO3/NH4 against the reference's (dump E records) to 1e-6.
GPU (marker gpu): the nitrogen run on CUDA equals CPU, and one nitrogen day is captured in a CUDA
graph and replayed bit-equal to eager (no host syncs).
"""
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths  # noqa: E402,F401
import test_cropsyst_connector as tc  # noqa: E402
from cropsyst_n_dump import read_dump, dump_path  # noqa: E402

from torchvic.coupling.connector import Context  # noqa: E402
from torchvic.crop.cropsyst import engine as cse  # noqa: E402

N_RUNS = _paths.env_path(('TORCHVIC_CROPSYST_N_RUNS',), 'build/vic_cropsyst/crb_N')
CASES = (330492, 308200, 328617)
_CACHE = {}


def _inputs(case):
    """Connector inputs of the N-on reference run plus the replay drivers and the soil N
    parameters (VIC soil as the reference sees it: dump H records)."""
    if case in _CACHE:
        return _CACHE[case]
    if not dump_path(N_RUNS, case):
        pytest.skip("N-on dump not present")
    inp = tc._l2_inputs(case, runs=N_RUNS)
    head, rec = read_dump(dump_path(N_RUNS, case))
    tiles, nd = inp["tiles"], inp["nd"]
    h = head[int(tiles[0])]
    th = h["thickness"]
    soil = dict(inp["soil"], thickness=th, bulk_density_g_cm3=h["bulk_density"], field_capacity=h["fc"],
                wilting_point=h["pwp"], saturation=h["sat"])
    L = len(th)
    drv = {k: [] for k in ("moist", "ice", "T")}
    for d in range(nd):
        row = {k: [] for k in drv}
        for t in tiles:
            p = rec.get((inp["year"][d], inp["doy"][d], int(t)), {}).get("P")
            for k in drv:
                row[k].append(p[k] if p else [0.0] * L)
        for k in drv:
            drv[k].append(row[k])
    inp = dict(inp, soil=soil, replay_n=dict(moist=drv["moist"], ice=drv["ice"], soil_t=drv["T"]),
               head=head, rec=rec)
    _CACHE[case] = inp
    return inp


def _connector(inp, device="cpu"):
    c = tc._connector(inp, nitrogen=True)
    c.cs_cfg["replay"].update(inp["replay_n"])
    return c


def test_nitrogen_static_from_the_crb_managements():
    """The management events of the CRB rotations become event tables: before-planting tillage and
    fertilizer, crop-stage events, sampling and split automatic N (308200); checked=false and
    relative-to-sowing-date events never run."""
    inp = _inputs(308200)
    c = _connector(inp)
    c.initialize(Context(device=torch.device("cpu"), dtype=torch.float64))
    N = c.cs.S.n
    assert N is not None
    assert int((N.crop_ev.nkind == 3).sum()) >= 2            # auto_N_application:1 of both crops
    assert int((N.scen_ev.nkind == 4).sum()) >= 2            # soil N sampling, before planting
    assert int((N.scen_ev.nkind == 2).sum()) == 0            # the fertilizations are checked=false
    assert int((N.day_mgt >= 0).sum()) > 0
    # the engine's own Saxton soil agrees with the saturation the reference hands CropSyst to
    # 1.5e-8 (VIC's soil file values); the replay runs pass the reference's values themselves
    sat = (c.cs.S.max_moist / 1000.0) / c.cs.S.thick
    ref = torch.tensor(inp["head"][int(inp["tiles"][0])]["sat"], dtype=torch.float64)
    assert float(((sat[0] - ref) / ref).abs().max()) < 1e-7


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_l2_nitrogen_through_the_connector(case):
    inp = _inputs(case)
    out, _ = tc._run(_connector(inp), inp["nd"])
    T, nd = inp["T"], inp["nd"]
    rows = tc._l2_inputs(case, runs=N_RUNS)
    import csv
    import glob
    f = glob.glob(f"{N_RUNS}/Liu_Linux_{case}_*/VIC_output/vic_crop_daily*")[0]
    data = list(csv.reader(open(f)))
    H = {h: i for i, h in enumerate(data[0])}
    by = {}
    for r in data[1:]:
        by.setdefault(r[H["CroppingSyst_code"]], []).append(r)
    cols = {"GAI": ("gai", 1.0), "Biomass_kg_m2": ("biomass", 1.0), "Root_depth_mm": ("root_depth", 1000.0),
            "Green_Canopy_Cover": ("green_cover", 1.0), "crop_evapotransp_max_mm": ("et_max", 1.0),
            "Yield_kg_m2": ("yield", 1e-4)}       # Act_Transp_mm is the replayed driver
    worst = {c: 0.0 for c in cols}
    count = {c: 0 for c in cols}
    for k, t in enumerate(inp["tiles"]):
        ref_sow = [d for d in range(nd) if rows["crop"][d][k] != "NOCROP"
                   and (d == 0 or rows["crop"][d - 1][k] == "NOCROP")]
        ref_harv = [d for d in range(nd) if rows["crop"][d][k] != "NOCROP" and "harvest" in rows["lab"][d][k]]
        assert [d for d in range(nd) if out["sown"][d, k] > 0] == ref_sow
        assert [d for d in range(nd) if out["harvest"][d, k] > 0] == ref_harv
        for d in range(nd):
            if rows["crop"][d][k] == "NOCROP":
                continue
            hv = d in ref_harv
            for c, (key, fac) in cols.items():
                if c == "Yield_kg_m2" and not hv:
                    continue
                if hv and c in ("GAI", "Biomass_kg_m2", "Green_Canopy_Cover", "Act_Transp_mm"):
                    continue
                r = float(by[t][d][H[c]])
                x = float(out[key][d, k]) * fac
                worst[c] = max(worst[c], abs(r - x) / max(abs(r), 1e-6))
                count[c] += 1
    for c in cols:
        assert count[c] > 0 and worst[c] < 1e-7, (c, worst[c], count[c])
    assert count["Yield_kg_m2"] >= 4 and count["GAI"] > 1400
    # soil mineral N against the reference's end-of-day state
    bad = 0
    n = 0
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
    assert n > 1000 and bad == 0, (bad, n)
    assert float(out["n_overflow"].max()) == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_nitrogen_equals_cpu_and_cuda_graph_capture():
    inp = _inputs(330492)
    nd = 400
    cpu, _ = tc._run(_connector(inp), nd)
    gpu, _ = tc._run(_connector(inp), nd, device="cuda")
    for k in cpu:
        a, b = cpu[k], gpu[k].cpu()
        same = torch.isclose(a, b, rtol=1e-9, atol=1e-12) | (torch.isnan(a) & torch.isnan(b))
        assert bool(same.all()), (k, float((a - b).abs().nan_to_num().max()))
    # CUDA graph: one nitrogen day of the connector's run object, replayed day by day
    c = _connector(inp)
    c.initialize(Context(device=torch.device("cuda"), dtype=torch.float64))
    cs = c.cs
    eager = cs.state
    for d in range(150):
        eager, _ = cs.step(eager)
    template = cs.state
    buf = [t.clone() for t in cse.flatten(template)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            cs.step(cse.unflatten(template, buf))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        new, out = cs.step(cse.unflatten(template, buf))
        for b, o in zip(buf, cse.flatten(new)):
            b.copy_(o)
    for b, x in zip(buf, cse.flatten(template)):
        b.copy_(x)
    for d in range(150):
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(cse.flatten(eager), buf):
        assert bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
