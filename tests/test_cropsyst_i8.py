"""Replay through the crop connector of every CRB case the reference can run, nitrogen
on and off (driver: tests/cropsyst_i8_l2.py; reference runs in <data root>/verification/cropsyst_i8i13).

Bar: GAI, canopy biomass, green cover, root depth, ET max and every harvest's yield equal the crop
csv (7-8 digits) on all crop days, sowing and harvest days equal, and with nitrogen the soil NO3/NH4
of every tile and day equal the reference dump to 1e-6. The orchard cases (315604 winegrape,
318377 apple) are tested in tests/test_cropsyst_orchard.py.
"""
import glob
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cropsyst_i8_l2 as L  # noqa: E402

ANNUAL = (330492, 308200, 328617, 310974, 259094)
BAR = 2e-7


def _need(case, nitrogen):
    if not glob.glob(f"{L.runs_dir(nitrogen)}/Liu_Linux_{case}_*/VIC_output/vic_crop_daily*"):
        pytest.skip(f"reference run of {case} not present ({L.runs_dir(nitrogen)})")


@pytest.mark.reference
@pytest.mark.parametrize("nitrogen", [False, True], ids=["N-off", "N-on"])
@pytest.mark.parametrize("case", ANNUAL)
def test_l2_through_the_connector(case, nitrogen):
    _need(case, nitrogen)
    r = L.compare(case, nitrogen)
    assert r["sow_ok"] and r["harv_ok"]
    for c, w in r["worst"].items():
        assert r["count"][c] > 0 and w < BAR, (c, w)
    if nitrogen:
        assert r["n_checked"] > 100000 and r["n_bad"] == 0, (r["n_bad"], r["n_checked"])


# ----------------------------------------------------------------------------- GPU day step

def _bench():
    import importlib.util
    p = os.path.join(HERE, "..", "benchmarks", "bench_cropsyst_isc.py")
    spec = importlib.util.spec_from_file_location("bench_cropsyst_isc", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.gpu
@pytest.mark.parametrize("nitrogen", [False, True], ids=["N-off", "N-on"])
def test_gpu_day_step_has_no_host_sync_and_compile_and_graph_equal_eager(nitrogen):
    """The CRB-like day step (3 tiles per cell) on CUDA: eager steps raise on any host sync
    (torch.cuda.set_sync_debug_mode('error')), a CUDA graph replay and a torch.compile'd step equal
    the eager one."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from torchvic.crop.cropsyst import engine as cse
    b = _bench()
    run = b.build(300, 40, 3, nitrogen, "cuda")
    moist, et, soil_t = b.drivers(run, "cuda", nitrogen)
    step = lambda s: run.step(s, moist, None, et, soil_t)
    st = run.state
    for _ in range(12):                                   # through the first sowings
        st, _o = step(st)
    start = st
    torch.cuda.set_sync_debug_mode("error")
    try:
        e = start
        for _ in range(4):
            e, _o = step(e)
    finally:
        torch.cuda.set_sync_debug_mode("default")
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
    for _ in range(4):
        g.replay()
    torch.cuda.synchronize()
    same = lambda a, c: bool(((a == c) | (torch.isnan(a) & torch.isnan(c))).all())
    assert all(same(a, c) for a, c in zip(cse.flatten(e), buf))
    cstep = torch.compile(step, dynamic=False)
    c = start
    for _ in range(4):
        c, _o = cstep(c)
    for a, x in zip(cse.flatten(e), cse.flatten(c)):
        if a.dtype.is_floating_point:
            ok = torch.isclose(a, x, rtol=1e-12, atol=1e-14) | (torch.isnan(a) & torch.isnan(x))
            assert bool(ok.all())
        else:
            assert same(a, x)
