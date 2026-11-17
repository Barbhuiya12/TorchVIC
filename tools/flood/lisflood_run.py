"""Thin driver for the external LISFLOOD-FP 8.x binary (GPL-2.0-only, run as a separate process).

Writes .par/.dem/.bci/.bdy/rain files, launches the binary, reads the .mass ledger
and depth rasters.  No LISFLOOD code is linked or copied.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_BINARY = Path(os.environ.get(
    "LISFLOOD_BIN",
    os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'static/upstream/lisflood-fp-8.2-zenodo/build_cuda/lisflood')))
NETCDF_LIB = os.environ.get("LISFLOOD_NETCDF_LIB", "")

# solver name -> (par keywords). 'cuda' keyword selects the GPU path.
SOLVERS = {
    "orig_cpu": [],
    "acc_cpu": ["acceleration"],
    "acc_gpu": ["acceleration", "cuda"],
    "fv1_cpu": ["fv1"],
    "fv1_gpu": ["fv1", "cuda"],
    "hwfv1_gpu": ["hwfv1", "cuda"],
    "mwdg2_gpu": ["mwdg2", "cuda"],
}


ADAPTIVE = ("hwfv1_gpu", "mwdg2_gpu")


def write_dem(path, z, cellsize, xll=0.0, yll=0.0, nodata=-9999):
    z = np.asarray(z, dtype=float)
    ny, nx = z.shape
    with open(path, "w") as f:
        f.write(f"ncols {nx}\nnrows {ny}\nxllcorner {xll}\nyllcorner {yll}\n"
                f"cellsize {cellsize}\nNODATA_value {nodata}\n")
        np.savetxt(f, z, fmt="%.6f")


def write_bdy(path, series):
    """series: {name: (units, [(value, t), ...])}; value is m (HVAR) or m2/s per unit width (QVAR)."""
    with open(path, "w") as f:
        f.write("LISFLOOD bdy\n")
        for name, (units, rows) in series.items():
            f.write(f"{name}\n{len(rows)} {units}\n")
            for v, t in rows:
                f.write(f"{v:.10g} {t:.10g}\n")


def write_bci(path, lines):
    Path(path).write_text("\n".join(lines) + "\n")


def write_rain_ts(path, rows, units="seconds"):
    """rows: [(rate_mm_per_hr, t)]; spatially uniform rainfall."""
    with open(path, "w") as f:
        f.write("# rainfall mm/hr\n")
        f.write(f"{len(rows)}\t{units}\n")
        for r, t in rows:
            f.write(f"{r:.10g}\t{t:.10g}\n")


def write_dynamic_rain_nc(path, depth_mm, times_h, cellsize, xll=0.0, yll=0.0):
    """depth_mm[t, y(top-first), x]: rainfall depth per interval (mm). Needs netCDF4."""
    import netCDF4
    nt, ny, nx = depth_mm.shape
    ds = netCDF4.Dataset(str(path), "w")
    ds.createDimension("time", nt)
    ds.createDimension("x", nx)
    ds.createDimension("y", ny)
    t = ds.createVariable("time", "f4", ("time",)); t.units = "hour"; t.axis = "T"
    x = ds.createVariable("x", "f8", ("x",)); x.units = "m"; x.axis = "X"
    y = ds.createVariable("y", "f8", ("y",)); y.units = "m"; y.axis = "Y"
    r = ds.createVariable("rainfall_depth", "f4", ("time", "y", "x")); r.units = "mm"
    t[:] = times_h
    x[:] = xll + (np.arange(nx) + 0.5) * cellsize
    y[:] = (yll + (np.arange(ny) + 0.5) * cellsize)[::-1]
    r[:] = depth_mm
    ds.close()


def read_asc(path):
    with open(path) as f:
        hdr = {}
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
        a = np.loadtxt(f)
    a = np.atleast_2d(a)
    a[a == hdr["nodata_value"]] = np.nan
    return a, hdr


def read_mass(path):
    """Parse the .mass ledger into {column: np.array}."""
    lines = Path(path).read_text().splitlines()
    cols = re.split(r"\s{2,}|\s+", lines[0].strip())
    # header has 'Rain-(Inf+Evap)' last; 12 columns
    cols = ["Time", "Tstep", "MinTstep", "NumTsteps", "Area", "Vol", "Qin", "Hds",
            "Qout", "Qerror", "Verror", "Rain"][:12]
    rows = [[float(v) for v in ln.split()] for ln in lines[1:] if ln.strip()]
    if not rows:
        return {}
    arr = np.array(rows)
    return {c: arr[:, i] for i, c in enumerate(cols[:arr.shape[1]])}


@dataclass
class Case:
    name: str
    solver: str
    dem: np.ndarray
    cellsize: float
    sim_time: float
    bci: list = field(default_factory=list)
    bdy: dict = field(default_factory=dict)
    rain_ts: list | None = None            # [(mm/hr, t_s)]
    rain_dyn: tuple | None = None          # (depth_mm[t,y,x], times_h)
    start_depth: np.ndarray | None = None  # initial water depth (m), enables .start
    fpfric: float = 0.03
    massint: float = 10.0
    saveint: float | None = None
    initial_tstep: float = 1.0
    extra: list = field(default_factory=list)
    epsilon: float = 1e-4


@dataclass
class Result:
    ok: bool
    returncode: int
    wall: float
    mass: dict
    depth: np.ndarray | None
    cellsize: float
    workdir: Path
    log: str
    comp_s: float | None = None   # solver-reported computation time (s)

    @property
    def stored_volume(self):
        """Sum(depth x cell area) from the final depth raster (m3)."""
        return float(np.nansum(self.depth) * self.cellsize ** 2)

    @property
    def ledger_volume(self):
        return float(self.mass["Vol"][-1]) if "Vol" in self.mass else None

    def outflow_volume(self):
        """Trapezoid integral of Qout (m3/s) over the ledger times."""
        t, q = self.mass["Time"], self.mass["Qout"]
        return float(np.trapezoid(q, t) + q[0] * t[0])

    def inflow_volume(self):
        t, q = self.mass["Time"], self.mass["Qin"]
        return float(np.trapezoid(q, t) + q[0] * t[0])


def run_case(case: Case, workdir=None, binary=DEFAULT_BINARY, timeout=None, keep=True):
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix=f"lf_{case.name}_"))
    wd.mkdir(parents=True, exist_ok=True)
    write_dem(wd / "case.dem", case.dem, case.cellsize)
    par = ["DEMfile case.dem", "resroot case", "dirroot out",
           f"sim_time {case.sim_time}", f"initial_tstep {case.initial_tstep}",
           f"massint {case.massint}", f"saveint {case.saveint or case.sim_time}",
           f"fpfric {case.fpfric}", "elevoff", "output_precision 10"]
    par += SOLVERS[case.solver]
    if case.solver in ADAPTIVE:
        # adaptive solvers need these keywords (no defaults; missing ones crash) and
        # a 2^L mesh; they write no .mass ledger, only rasters when raster_out is set.
        L = int(np.ceil(np.log2(max(case.dem.shape))))
        par += [f"max_ref_lvl {L}", f"epsilon {case.epsilon}", "wall_height 0.5", "raster_out"]
    if case.bci:
        write_bci(wd / "case.bci", case.bci)
        par.append("bcifile case.bci")
    if case.bdy:
        write_bdy(wd / "case.bdy", case.bdy)
        par.append("bdyfile case.bdy")
    if case.rain_ts:
        write_rain_ts(wd / "case.rain", case.rain_ts)
        par.append("rainfall case.rain")
    if case.rain_dyn:
        write_dynamic_rain_nc(wd / "rain.nc", case.rain_dyn[0], case.rain_dyn[1], case.cellsize)
        par.append("dynamicrainfile rain.nc")
    if case.start_depth is not None:
        write_dem(wd / "case.start", case.start_depth, case.cellsize)
        par.append("startfile case.start")
    par += case.extra
    (wd / "case.par").write_text("\n".join(par) + "\n")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = NETCDF_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    import time
    t0 = time.time()
    p = subprocess.run([str(binary), "case.par"], cwd=wd, env=env, capture_output=True,
                       text=True, timeout=timeout)
    wall = time.time() - t0
    out = wd / "out"
    mass = {}
    mf = sorted(out.glob("*.mass"))
    if mf:
        mass = read_mass(mf[0])
    depth = None
    wds = sorted(out.glob("*.wd"), key=lambda q: int(re.findall(r"(\d+)\.wd$", q.name)[0]))
    if wds:
        depth, _ = read_asc(wds[-1])
    comp = None
    m = re.search(r"Total computation time:\s*([\d.]+) mins", p.stdout)
    if m:
        comp = float(m.group(1)) * 60.0
    m = re.search(r"Loop time:\s*([\d.]+) s", p.stdout)
    if m:
        comp = float(m.group(1))
    return Result(p.returncode == 0 and (bool(mass) or depth is not None), p.returncode, wall, mass, depth,
                  case.cellsize, wd, p.stdout[-2000:] + p.stderr[-2000:], comp)


def flat_dem(n, z=10.0):
    return np.full((n, n), float(z))


def cleanup(res: Result):
    shutil.rmtree(res.workdir, ignore_errors=True)
