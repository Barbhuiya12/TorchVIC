"""Controlled-volume tests for the external LISFLOOD-FP binary (tools/flood/lisflood_run.py).

Pins the volume-exact input paths found in <data root>/verification/flood_volume/run.log:
point source QFIX/QVAR (per-unit-width m2/s, so m3/s = value x cell size), uniform rainfall
file (mm/hr) and dynamic NetCDF rainfall, on the ACC local-inertial solver (CPU always, GPU
when a CUDA device is usable).  Skips when the binary is absent.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "flood"))
import lisflood_run as L  # noqa: E402

pytestmark = pytest.mark.skipif(not L.DEFAULT_BINARY.exists(), reason="LISFLOOD-FP binary not built")

N, DX, T = 32, 10.0, 200.0
AREA = N * N * DX * DX


def _gpu_ok():
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    except OSError:
        return False


SOLVERS = ["acc_cpu", pytest.param("acc_gpu", marks=pytest.mark.skipif(not _gpu_ok(), reason="no GPU"))]


def _run(solver, **kw):
    c = L.Case("t", solver, kw.pop("dem", L.flat_dem(N)), DX, T, massint=T / 4, **kw)
    r = L.run_case(c)
    assert r.ok, r.log
    return r


@pytest.mark.parametrize("solver", SOLVERS)
def test_point_qfix_is_per_unit_width(solver):
    r = _run(solver, bci=["P 155 155 QFIX 1.0"])
    expect = 1.0 * DX * T                       # value is m2/s: Q [m3/s] = value x dx
    assert r.stored_volume == pytest.approx(expect, rel=1e-9)
    assert r.ledger_volume == pytest.approx(expect, rel=1e-9)
    assert float(r.mass["Qin"][-1]) == pytest.approx(1.0 * DX)


@pytest.mark.parametrize("solver", SOLVERS)
def test_point_qvar_matches_qfix(solver):
    ser = {"s": ("seconds", [(1.0, 0.0), (1.0, 2 * T)])}
    r = _run(solver, bci=["P 155 155 QVAR s"], bdy=ser)
    assert r.stored_volume == pytest.approx(1.0 * DX * T, rel=1e-9)


@pytest.mark.parametrize("solver", SOLVERS)
def test_uniform_rain_file_mm_per_hour(solver):
    r = _run(solver, rain_ts=[(36.0, 0.0), (36.0, 2 * T)])
    assert r.stored_volume == pytest.approx(36.0 / 3.6e6 * T * AREA, rel=1e-9)


def test_dynamic_rain_cpu_exact_and_oriented():
    f = np.zeros((3, N, N)); f[1:, 3, 20] = 50.0       # mm per level (50 s), top-first rows
    c = L.Case("t", "acc_cpu", L.flat_dem(N), DX, 100.0, massint=50.0,
               rain_dyn=(f, np.arange(3) * 50.0 / 3600.0))
    r = L.run_case(c)
    assert r.ok, r.log
    assert np.unravel_index(np.nanargmax(r.depth), r.depth.shape) == (3, 20)
    assert r.stored_volume == pytest.approx(r.ledger_volume, rel=1e-6)


def test_point_source_cell_is_top_first():
    r = _run("acc_cpu", bci=["P 205 315 QFIX 1.0"])    # x=205 -> col 20, y=315 -> top row
    assert np.unravel_index(np.nanargmax(r.depth), r.depth.shape) == (0, 20)


def test_acc_free_edge_ledger_balance_cpu():
    x = np.arange(N)[None, :] * np.ones((N, 1))
    dem = 10.0 - 0.01 * x * DX
    c = L.Case("t", "acc_cpu", dem, DX, 600.0, massint=30.0, bci=["P 45 145 QFIX 1.0", "E 0 300 FREE 0.01"])
    r = L.run_case(c)
    assert r.ok, r.log
    v_in = 1.0 * DX * 600.0
    # outflow through FREE is reported in the ledger on CPU; in + out = stored within 2 percent
    assert r.outflow_volume() > 0
    assert r.stored_volume + r.outflow_volume() == pytest.approx(v_in, rel=0.02)
