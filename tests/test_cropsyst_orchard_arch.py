"""The orchard canopy architecture of the CropSyst backend (orchard_arch.py).

Oracle (tests/oracle/ref_cropsyst_orchard_arch.cpp, linked against the compiled VIC-CropSyst
reference): the parsed architecture parameters, and multi-year runs of the tree canopy extent and the
solid canopy interception of all three models (none, Kemanian-Stockle, Oyarzun-Stockle) for
randomised crop files (geometry, row azimuth, spacing smaller than the canopy), sites (both
hemispheres, slopes and aspects), radiation, thermal-time series and restarts. Bar 1e-12 (relative)
on every value of every day.

Replay (tests/cropsyst_apple_arch_l2.py): the CRB apple case 318377 with canopy_architecture=true, each
model, nitrogen off and on, against the reference runs of <data root>/verification/cropsyst_apple_arch.

GPU: the orchard day step with the architecture on (Oyarzun-Stockle and Kemanian-Stockle) on CUDA
with no host sync, equal to the CPU run; CUDA graph replay and torch.compile equal to eager.
"""
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths  # noqa: E402

from torchvic.crop.cropsyst import orchard_arch as oa  # noqa: E402
from torchvic.crop.cropsyst import phenology as ph  # noqa: E402
from torchvic.crop.cropsyst import readers as rd  # noqa: E402

ORACLE = _paths.oracle("ref_cropsyst_orchard_arch")
ARCH_ROOT = _paths.env_path(('TORCHVIC_CROPSYST_APPLE_ARCH',), 'verification/cropsyst_apple_arch')
ARCH_DB = os.path.join(ARCH_ROOT, "Simulation", "Database")
APPLE = os.path.join(ARCH_DB, "Crop", "Code", "1401.CS_crop")
TOL = 1e-12


def _need_oracle():
    if not os.path.exists(ORACLE):
        pytest.skip("ref_cropsyst_orchard_arch oracle not built (tests/oracle/build_ref_cropsyst_orchard_arch.sh)")
    if not os.path.exists(APPLE):
        pytest.skip("apple canopy-architecture crop file not present")


def _parse(line):
    return [float.fromhex(x) for x in line.split()]


def _set(text, key, value):
    import re
    pat = re.compile(r"^(%s=)[^\n\r]*" % re.escape(key), re.M)
    assert pat.search(text), key
    return pat.sub(lambda m: m.group(1) + str(value), text, count=1)


def make_crop(rng, template):
    """A randomised architecture: canopy dimensions (sometimes wider than the spacing, sometimes
    shrinking), spacing, row azimuth, the flowering / rapid fruit growth thermal times."""
    t = template
    fl = rng.uniform(30, 400)
    vals = {"initial_width": round(rng.uniform(0.3, 2.5), 3), "final_width": round(rng.uniform(0.3, 4.5), 3),
            "initial_breadth": round(rng.uniform(0.3, 2.5), 3), "final_breadth": round(rng.uniform(0.3, 4.5), 3),
            "initial_height": round(rng.uniform(0.8, 3.0), 3), "final_height": round(rng.uniform(0.8, 5.0), 3),
            "branch_insertion_height": round(rng.uniform(0.0, 0.8), 3),
            "row_spacing": round(rng.uniform(1.5, 5.0), 3), "col_spacing": round(rng.uniform(0.8, 4.0), 3),
            "row_azimuth": round(rng.choice([0.0, 90.0, rng.uniform(0, 360)]), 2),
            "flowering": round(fl), "rapid_fruit_growth": round(fl + rng.uniform(100, 1500), 2)}
    for k, v in vals.items():
        t = _set(t, k, v)
    return t


def test_arch_params_equal_the_reference_parser():
    """arch_params() on the apple architecture file and randomised variants equals the compiled
    reference parser (Crop_parameters + VV_File)."""
    _need_oracle()
    tmp = tempfile.mkdtemp()
    try:
        rng = random.Random(11)
        text = open(APPLE, encoding="latin-1").read()
        paths = [APPLE]
        for i in range(8):
            p = os.path.join(tmp, f"a{i}.crp")
            open(p, "w", encoding="latin-1").write(make_crop(rng, text))
            paths.append(p)
        p = os.path.join(tmp, "off.crp")                  # architecture off, defaults of absent keys
        open(p, "w", encoding="latin-1").write(text.replace("canopy_architecture=true\n", "")
                                              .replace("canopy_architecture=true\r\n", ""))
        paths.append(p)
        r = subprocess.run([ORACLE], input="".join(f"CRP {p}\n" for p in paths), capture_output=True,
                           text=True, timeout=600)
        rows = [l.split() for l in r.stdout.splitlines()]
        assert len(rows) == len(paths), r.stderr[-2000:]
        crops = rd.crop_table(paths)
        AP = oa.arch_params(crops, ph.phenology_params(crops))
        for i, row in enumerate(rows):
            v = [float.fromhex(x) for x in row[1:]]
            assert bool(AP.on[i]) == bool(int(row[0])), paths[i]
            got = [float(AP.w0[i]), float(AP.w1[i]), float(AP.b0[i]), float(AP.b1[i]), float(AP.h0[i]),
                   float(AP.h1[i]), float(AP.bih[i]), float(AP.row[i]), float(AP.col[i])]
            assert got == v[:9], (paths[i], got, v[:9])
            assert float(AP.row_az[i]) == float(oa._aspect_s0(torch.tensor(v[9], dtype=torch.float64))) * oa.DEG_TO_RAD
            assert [float(AP.begin0[i]), float(AP.end[i])] == v[10:12], paths[i]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _case_days(rng, n_day):
    """Radiation, thermal time and restarts of one case: tt accumulates by 0-22 a day, a restart
    (bud break) clears it about once a year."""
    days, tt = [], rng.uniform(0, 300)
    for d in range(n_day):
        srad = rng.uniform(0.5, 34.0)
        tt += rng.choice([0.0, rng.uniform(0, 22)])
        rs = rng.random() < 1 / 300.0
        days.append((srad, tt, rs, 0.0))
        if rs:
            tt = 0.0
    return days


def oracle_cases(cases, n_day):
    lines = []
    for c in cases:
        lines.append(f"ARCH {c['path']} {c['model']} {c['lat']!r} {c['lon']!r} {c['slope']!r} {c['aspect']!r} "
                     f"{n_day}\n")
        for d, (s, tt, rs, ta) in enumerate(c["days"]):
            year, doy = 2001 + d // 365, d % 365 + 1
            lines.append(f"{year} {doy} {s!r} {tt!r} {int(rs)} {ta!r}\n")
    r = subprocess.run([ORACLE], input="".join(lines), capture_output=True, text=True, timeout=3600)
    # the Kemanian-Stockle model prints a warning to stdout on each update (crop_orchard_Kemanian_Stockle.cpp:247)
    rows = [_parse(l) for l in r.stdout.splitlines() if not l.startswith("WARNING")]
    assert len(rows) == len(cases) * n_day, r.stderr[-2000:]
    return [rows[k * n_day:(k + 1) * n_day] for k in range(len(cases))]


def port_cases(cases, n_day):
    crops = rd.crop_table([c["path"] for c in cases])
    AP = oa.arch_params(crops, ph.phenology_params(crops))
    T = len(cases)
    AS = oa.arch_static([oa.MODELS[c["model"]] for c in cases], [c["lat"] for c in cases],
                        [c["lon"] for c in cases], torch.tensor([c["slope"] for c in cases], dtype=torch.float64),
                        torch.tensor([c["aspect"] for c in cases], dtype=torch.float64),
                        torch.tensor([c["days"][0][0] for c in cases], dtype=torch.float64), 1, AP, "cpu")
    on = torch.ones(T, dtype=torch.bool)
    AP = type(AP)(*[x for x in AP])                             # one crop per tile, in order
    a = oa.sow(oa.initial_state(T), AP, AS, on)
    out = []
    for d in range(n_day):
        col = lambda k, dt=torch.float64: torch.tensor([c["days"][d][k] for c in cases], dtype=dt)
        doy = torch.tensor(d % 365 + 1)
        a = oa.update(a, AP, AS, on, col(1), doy, col(0))
        out.append(torch.stack([a.width, a.breadth, a.height, a.interception], -1))
        a = oa.restart(a, col(2, torch.bool), col(3))
    return torch.stack(out, 1)                                  # [T, n_day, 4]


def test_l1_interception_and_canopy_extent_equal_the_reference():
    """Two years of 30 randomised orchards (ten per model) through orchard_arch.update / restart equal
    the compiled Orchard_common with its interception models on every day, to 1e-12."""
    _need_oracle()
    tmp = tempfile.mkdtemp()
    try:
        rng = random.Random(7)
        text = open(APPLE, encoding="latin-1").read()
        n_day = 730
        cases = []
        for k in range(30):
            p = os.path.join(tmp, f"c{k}.crp")
            open(p, "w", encoding="latin-1").write(make_crop(rng, text))
            lat = rng.choice([rng.uniform(-60, -15), rng.uniform(15, 60), rng.uniform(-15, 15)])
            cases.append(dict(path=p, model=k % 3, lat=round(lat, 5), lon=round(rng.uniform(-180, 180), 5),
                              slope=rng.choice([0.0, round(rng.uniform(0, 35), 2)]),
                              aspect=round(rng.uniform(0, 360), 2), days=_case_days(rng, n_day)))
        ref = oracle_cases(cases, n_day)
        got = port_cases(cases, n_day)
        worst = 0.0
        for k, c in enumerate(cases):
            r = torch.tensor(ref[k], dtype=torch.float64)
            g = got[k]
            cols = [3] if c["model"] == 0 else [0, 1, 2, 3]
            for j in cols:
                e = (g[:, j] - r[:, j]).abs() / r[:, j].abs().clamp(min=1e-6)
                bad = torch.nonzero(e > TOL)
                assert bad.numel() == 0, (k, c["model"], j, int(bad[0, 0]), float(g[bad[0, 0], j]),
                                          float(r[bad[0, 0], j]))
                worst = max(worst, float(e.max()))
        print("worst", worst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.reference
@pytest.mark.parametrize("nitrogen", [False, True], ids=["N-off", "N-on"])
@pytest.mark.parametrize("model", ["oy", "ks", "none"])
def test_l2_apple_architecture_through_the_connector(model, nitrogen):
    import glob
    import cropsyst_apple_arch_l2 as AL
    if not glob.glob(f"{AL.runs_dir(model, nitrogen)}/Liu_Linux_{AL.CASE}_*/VIC_output/arch_dump.txt"):
        pytest.skip("apple canopy-architecture reference run not present")
    r = AL.compare(model, nitrogen)
    assert r["sow_ok"] and r["harvest_ok"], r
    assert r["n_bad_stage"] == 0, r["bad_stage"]
    for c, w in r["worst"].items():
        assert w < 2e-7, (c, w, r["where"][c])
    assert r["arch_days"]
    for c, w in r["arch_worst"].items():
        assert w < 1e-7, (c, w, r["arch_where"][c])
    if nitrogen:
        assert r["n_checked"] > 40000 and r["n_bad"] == 0, (r["n_bad"], r["n_checked"])


@pytest.mark.gpu
@pytest.mark.parametrize("model,nitrogen", [("Oyarzun-Stockle", False), ("Kemanian-Stockle", False),
                                            ("Oyarzun-Stockle", True)], ids=["oy-N-off", "ks-N-off", "oy-N-on"])
def test_gpu_orchard_architecture_no_host_sync_graph_and_compile_equal_eager(model, nitrogen):
    """The orchard GPU checks of test_cropsyst_orchard.py (a year of 128 tiles, no host sync, CPU
    equality, CUDA graph and torch.compile on a restart and a dormancy day) with the apple's canopy
    architecture on."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not os.path.exists(APPLE):
        pytest.skip("apple canopy-architecture crop file not present")
    import test_cropsyst_orchard as TO
    TO.gpu_check(nitrogen, db=ARCH_DB, arch=model)
