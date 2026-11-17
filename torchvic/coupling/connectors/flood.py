"""Flood connector: VIC runoff and routed inflow into a flood model (one way
by default; two-way with the engine backend, see "Two-way" below and
docs/connectors/flood.md).

Two backends (`backend`):

  engine    (default) TorchVIC's own local-inertial solver
            (torchvic/flood, validated against LISFLOOD-FP ACC), in
            process: depth and face momentum stay on the device between
            intervals (no restart from rest), and the interval's balance is
            the engine's float64 mass ledger (sources in, boundary outflow,
            storage). A basin DEM is conditioned by default (`condition`:
            depressions deeper than 5 m filled, rivers of the 1/16 degree
            flow map burnt in; plain `dem` arrays are used as given).
            `compute` runs the engine and restores its state; `commit`
            installs the proposed state, so a rejected step changes nothing.
  lisflood  LISFLOOD-FP 8.x as an external process (the reference; selected
            also by giving `solver`), described below.

Upstream inflow: `inflow` (m3 per inflow point over the interval, a flux),
or `upstream_discharge` (m3 s-1 at the routing connector's gauge nodes, a
state on grid `node_grid`): each node of `inflow_nodes` feeds all inflow
points, each [node, [point indices]] of `inflow_map` its own points; a
node's discharge x the interval is shared equally by its points.

LISFLOOD-FP 8.x (GPL-2.0-only) is never linked: it is an external process.
Per coupling interval the lisflood backend

  1. takes the water the runtime delivers (`runoff`: m3 per hydraulic cell
     over the interval, i.e. VIC depth x contributing VIC area, mapped by a
     conservative `Mapping`; `inflow`: m3 per boundary point over the
     interval, e.g. the routing connector's discharge at the domain edge),
  2. writes the LISFLOOD-FP inputs: the runoff as a dynamic rainfall file
     (depth in mm per sub-level), the inflow as point sources
     (`P x y QFIX q`, q in m2/s PER UNIT WIDTH = m3/s / dx, the unit pinned
     in a LISFLOOD-FP volume-check run log),
  3. starts the solver (ACC local-inertial on the GPU with LISFLOOD-FP's own
     CUDA code, or the CPU build) from the depth the previous interval ended
     on, for `dt` seconds,
  4. reads depth, maximum depth and the `.mass` record, closes the
     connector's water balance and proposes the new depth.

Hydraulic results never modify VIC (this is the one-way mode); the connector
declares only outputs.

Water accounting (m3, whole domain, one interval):

    I_ext = V_runoff + V_inflow          handed to the hydraulic model
    dS    = sum(depth_end - depth_start) x dx^2      stored change (rasters)
    Q_out = water that left through open boundaries
    R     = I_ext - Q_out - dS           closes to rounding

The CPU solver reports Q_out in its `.mass` record (independent check). The
CUDA ACC solver reports Q_out = 0 for FREE edges although water does leave
(seen in a volume-check run), so Q_out is derived from the rasters there
(I_ext - dS) and the balance is then a check only for closed boundaries; a
negative derived outflow (water created) is refused, and the run record keeps
both numbers. The `.mass` Qin record is compared with the handed inflow every
interval.

Restart: the state is the depth raster at the interval end (exactly the
numbers LISFLOOD-FP wrote), the clock and the running totals. Momentum is
re-initialised at every interval start; this is the model of record, so a
resumed run equals the uninterrupted one.

Two-way (engine backend; every option off by default, and then nothing of
this is declared or run):

  hand_over       export `river_request` (state, m3 per inflow point, the
                  "everything" request REQUEST_ALL) for the routing
                  connector's withdrawal_request (lag 1); the delivered
                  `routing.withdrawn` comes in as `inflow` (lag 0). The
                  reach inside the box is then the flood model's, not
                  routed twice.
  return_outlets  [[edge] or [edge, i0, i1], ...]: groups of open boundary
                  faces; the outflow of each group over the interval (the
                  engine's per-face float64 accumulators) is exported as
                  `returned` [outlet] (flux, m3) for routing.injection
                  (lag 1). Open faces in no group: `outflow` (leaves the
                  coupled system).
  infiltration_mm_day, open_water_evap_mm_day
                  interval-end losses of cells deeper than flood_threshold:
                  e = min(h, E dt), then i = min(h - e, I dt). Infiltration
                  is summed per VIC cell of `vic_map` and exported as
                  `infiltration` [VIC cell of the map] (flux, m3) for
                  vic.soil_add (lag 1); VIC's soil_add_unmet comes back as
                  `infiltration_unmet` (flux, m3) and is added to that VIC
                  cell's hydraulic cells (the map's fractions) in the next
                  interval. Evaporation is E_other.
  vic_map         (VIC cell, hydraulic cell, fraction) pairs (a list or a
                  csv with source,target,fraction): the runoff mapping.

Two-way balance (m3): I_ext = runoff + inflow + infiltration_unmet;
Q_out = returned + outflow + infiltration; E_other = evaporation;
dS = storage change; closure = -clamp.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..connector import (Balance, ComputeError, Connector, DeviceFlag,
                         Proposal)
from ..spec import CouplingError, GridSpec, VarSpec

#: the LISFLOOD-FP executable (a user-built external program) and the netCDF
#: library directory it needs, if any
DEFAULT_BINARY = (os.environ.get('TORCHVIC_LISFLOOD_BIN')
                  or os.environ.get('LISFLOOD_BIN') or 'lisflood')
NETCDF_LIB = os.environ.get('LISFLOOD_NETCDF_LIB', '')

#: relative precision of the LISFLOOD-FP data path (see compute)
EPS_PATH = 1.0e-12

GRID = 'flood_cells'
INFLOW_GRID = 'flood_inflow'

#: solver -> LISFLOOD-FP .par keywords ('cuda' selects the GPU code)
SOLVERS = {'acc_cpu': ['acceleration'],
           'acc_gpu': ['acceleration', 'cuda']}
BACKENDS = ('engine', 'lisflood')
#: the 1/16 degree D8 map whose rivers are burnt into a basin DEM
FDIR = os.environ.get('TORCHVIC_FLOOD_FDIR', 'isc_fdir.nc')
#: the request that takes everything that passes a routing site (finite, so
#: no inf reaches a mapping or a ledger; the router caps it)
REQUEST_ALL = 1.0e30
OUTLET_GRID = 'flood_outlets'
VIC_GRID = 'flood_vic'


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------
def write_asc(path, a, dx, nodata=-9999, digits=None):
    """ESRI ASCII raster, row 0 = north (the layout LISFLOOD-FP reads)."""
    a = np.asarray(a, dtype=np.float64)
    ny, nx = a.shape
    import pandas as pd
    with open(path, 'w') as f:
        f.write('ncols %d\nnrows %d\nxllcorner 0.0\nyllcorner 0.0\n'
                'cellsize %.17g\nNODATA_value %d\n' % (nx, ny, dx, nodata))
        pd.DataFrame(a).to_csv(f, sep=' ', header=False, index=False,
                               float_format='%.17g' if digits is None
                               else '%%.%dg' % digits)


def read_asc(path):
    import pandas as pd
    with open(path) as f:
        hdr = {}
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
        a = pd.read_csv(f, sep=r'\s+', header=None,
                        engine='c').to_numpy(dtype=np.float64)
    a[a == hdr['nodata_value']] = np.nan
    return a


def write_rain_nc(path, depth_mm, times_h, dx):
    """depth_mm[level, row(north first), col]: the depth (mm) that falls
    between a level's time and the next one's (verified in a volume-check
    run). Double precision throughout."""
    import netCDF4
    nt, ny, nx = depth_mm.shape
    ds = netCDF4.Dataset(str(path), 'w')
    ds.createDimension('time', nt)
    ds.createDimension('x', nx)
    ds.createDimension('y', ny)
    t = ds.createVariable('time', 'f8', ('time',))
    t.units = 'hour'
    t.axis = 'T'
    x = ds.createVariable('x', 'f8', ('x',))
    x.units = 'm'
    x.axis = 'X'
    y = ds.createVariable('y', 'f8', ('y',))
    y.units = 'm'
    y.axis = 'Y'
    r = ds.createVariable('rainfall_depth', 'f8', ('time', 'y', 'x'))
    r.units = 'mm'
    t[:] = times_h
    x[:] = (np.arange(nx) + 0.5) * dx
    y[:] = ((np.arange(ny) + 0.5) * dx)[::-1]
    r[:] = depth_mm
    ds.close()


_MASS_COLS = ['Time', 'Tstep', 'MinTstep', 'NumTsteps', 'Area', 'Vol', 'Qin',
              'Hds', 'Qout', 'Qerror', 'Verror', 'Rain']


def read_mass(path):
    """The `.mass` record as {column: array}."""
    lines = Path(path).read_text().splitlines()
    rows = []
    for ln in lines[1:]:
        p = ln.split()
        if not p or p[0].startswith('#'):
            continue
        try:
            rows.append([float(v) for v in p])
        except ValueError:
            continue
    if not rows:
        return {}
    a = np.array(rows)
    return {c: a[:, i] for i, c in enumerate(_MASS_COLS[:a.shape[1]])}


def integrate(t, q, T=None):
    """Volume from a rate record that starts at t = 0 (the first row holds
    the rate over [0, t0)); the last rate is held to T when the record stops
    short of it (the .mass record is written every `massint`)."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    if len(t) == 0:
        return 0.0
    v = q[0] * t[0]
    if len(t) > 1:
        v += float(np.sum(0.5 * (q[1:] + q[:-1]) * np.diff(t)))
    if T is not None and t[-1] < T:
        v += q[-1] * (T - t[-1])
    return float(v)


def cell_speed(vx, vy):
    """Speed (m/s) at cell centres from the face velocities LISFLOOD-FP
    writes (.Vx is ny x (nx+1), .Vy (ny+1) x nx; NaN = no data -> 0)."""
    vx, vy = np.nan_to_num(vx), np.nan_to_num(vy)
    u = 0.5 * (vx[:, :-1] + vx[:, 1:])
    v = 0.5 * (vy[:-1, :] + vy[1:, :])
    return np.hypot(u, v)


# ---------------------------------------------------------------------------
# geometry: hydraulic grid <-> VIC cells
# ---------------------------------------------------------------------------
def load_basin(path):
    """A basin file written by tools/flood/make_basin.py: DEM and the
    georeference of the metric grid."""
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def hydraulic_lonlat(meta):
    """(lon, lat) of every hydraulic cell centre, [ny, nx], row 0 north."""
    nx, ny, dx = int(meta['nx']), int(meta['ny']), float(meta['dx'])
    j = np.arange(nx)[None, :] + 0.5
    i = np.arange(ny)[:, None] + 0.5
    x = j * dx * np.ones((ny, 1))
    y = (ny - i) * dx * np.ones((1, nx))
    lon = float(meta['lon0']) + x / float(meta['m_lon'])
    lat = float(meta['lat0']) + y / float(meta['m_lat'])
    return lon, lat


def vic_pairs(meta, vic_lon, vic_lat, dvic=0.0625):
    """Conservative mapping pairs (vic cell, hydraulic cell, fraction): each
    hydraulic cell belongs to the VIC cell whose box (centre +- dvic/2)
    contains its centre, and a VIC cell's water is shared equally between
    its hydraulic cells (equal cell areas), so every VIC cell's fractions
    sum to 1. A VIC cell with no hydraulic cell inside is refused (its
    water would have nowhere to go)."""
    lon, lat = hydraulic_lonlat(meta)
    vl, vb = np.asarray(vic_lon, float), np.asarray(vic_lat, float)
    key = {}
    for k, (lo, la) in enumerate(zip(vl, vb)):
        key[(int(round((lo - dvic / 2) / dvic)),
             int(round((la - dvic / 2) / dvic)))] = k
    ci = np.floor(lon / dvic + 1e-12).astype(int).ravel()
    cj = np.floor(lat / dvic + 1e-12).astype(int).ravel()
    owner = np.array([key.get((a, b), -1) for a, b in zip(ci, cj)])
    cnt = np.bincount(owner[owner >= 0], minlength=len(vl))
    if (cnt == 0).any():
        raise CouplingError(
            'VIC cell(s) %s have no hydraulic cell inside: their water '
            'could not enter the flood domain'
            % np.nonzero(cnt == 0)[0][:5].tolist())
    cells = np.nonzero(owner >= 0)[0]
    return (owner[cells], cells, 1.0 / cnt[owner[cells]])


def write_pairs_csv(path, pairs):
    with open(path, 'w') as f:
        f.write('source,target,fraction\n')
        for s, t, fr in zip(*pairs):
            f.write('%d,%d,%.17g\n' % (s, t, fr))


# ---------------------------------------------------------------------------
# the external run
# ---------------------------------------------------------------------------
class RunResult(object):
    pass


def run_lisflood(workdir, dem_path, ny, nx, dx, T, solver, depth0,
                 rain_mm=None, rain_times_h=None, points=(), free_edges=(),
                 fpfric=0.03, initial_tstep=1.0, massint=None, binary=None,
                 extra=(), timeout=None, digits=14, velocity=False):
    """Run LISFLOOD-FP for T seconds from `depth0` [ny, nx] (m). `points`:
    [(row, col, Q m3/s)]; rain: depth (mm) per level (see write_rain_nc)."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    par = ['DEMfile %s' % dem_path, 'resroot case', 'dirroot out',
           'sim_time %.17g' % T, 'initial_tstep %.17g' % initial_tstep,
           'massint %.17g' % (massint or T / 24.0), 'saveint %.17g' % T,
           'fpfric %.17g' % fpfric, 'elevoff',
           'output_precision %d' % digits]
    par += SOLVERS[solver]
    bci = []
    for r, c, Q in points:
        x, y = (c + 0.5) * dx, (ny - 1 - r + 0.5) * dx
        bci.append('P %.17g %.17g QFIX %.17g' % (x, y, Q / dx))
    for edge, slope in free_edges:
        L = (ny if edge in 'EW' else nx) * dx
        bci.append('%s 0 %.17g FREE %.17g' % (edge, L, slope))
    if bci:
        (wd / 'case.bci').write_text('\n'.join(bci) + '\n')
        par.append('bcifile case.bci')
    if rain_mm is not None:
        write_rain_nc(wd / 'rain.nc', rain_mm, rain_times_h, dx)
        par.append('dynamicrainfile rain.nc')
    if depth0 is not None and np.any(depth0 > 0):
        write_asc(wd / 'case.start', depth0, dx, digits=None)
        par.append('startfile case.start')
    if velocity:
        par.append('voutput')
    par += list(extra)
    (wd / 'case.par').write_text('\n'.join(par) + '\n')
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = NETCDF_LIB + ':' + env.get('LD_LIBRARY_PATH', '')
    t0 = time.time()
    p = subprocess.run([str(binary or DEFAULT_BINARY), 'case.par'], cwd=wd,
                       env=env, capture_output=True, text=True,
                       timeout=timeout)
    wall = time.time() - t0
    res = RunResult()
    res.returncode, res.wall, res.workdir = p.returncode, wall, wd
    res.log = (p.stdout[-1500:] + p.stderr[-1500:])
    out = wd / 'out'
    res.mass = {}
    mf = sorted(out.glob('*.mass')) if out.exists() else []
    if mf:
        res.mass = read_mass(mf[0])
    wds = sorted(out.glob('*.wd'), key=lambda q: int(
        re.findall(r'(\d+)\.wd$', q.name)[0])) if out.exists() else []
    res.depth = read_asc(wds[-1]) if wds else None
    mx = sorted(out.glob('*.max')) if out.exists() else []
    res.max_depth = read_asc(mx[0]) if mx else None
    res.vx = res.vy = None
    for ax in ('Vx', 'Vy'):
        fs = sorted(out.glob('*.' + ax)) if out.exists() else []
        if fs:
            setattr(res, ax.lower(), read_asc(fs[-1]))
    res.files = sorted(q.name for q in out.iterdir()) if out.exists() else []
    m = re.search(r'Loop time:\s*([\d.]+) s', p.stdout)
    res.loop_s = float(m.group(1)) if m else None
    return res


# ---------------------------------------------------------------------------
# the connector
# ---------------------------------------------------------------------------
class SegmentSum(object):
    """out[t] = sum of x[j] over the j with idx[j] == t, in a fixed order.

    A CUDA index_add_ adds with atomics, so with three or more terms per
    target the rounding depends on thread timing and two identical runs
    differ (here: the floodplain infiltration per VIC cell, a few thousand
    hydraulic cells each; the run then drifts at wet/dry fronts and a
    restart cannot equal the continuous run). The terms are gathered into
    a padded [target, k] matrix once and summed along k: a plain reduction,
    the same on every run."""

    def __init__(self, idx, n, device):
        idx = torch.as_tensor(idx, dtype=torch.long).cpu()
        m = int(idx.numel())
        order = torch.argsort(idx, stable=True)
        cnt = torch.bincount(idx, minlength=int(n))
        k = max(int(cnt.max()) if m else 0, 1)
        start = torch.cumsum(cnt, 0) - cnt
        si = idx[order]
        pos = torch.arange(m) - start[si]
        P = torch.full((int(n), k), m, dtype=torch.long)
        P[si, pos] = order
        self.P = P.to(device)
        self.m = m

    def __call__(self, x):
        xz = torch.cat([x, x.new_zeros(1)])
        return xz[self.P].sum(1)


class FloodConnector(Connector):
    """A flood model behind the Connector API, one way: TorchVIC's engine
    (default) or LISFLOOD-FP ACC as an external process."""
    name = 'flood'
    version = '1'
    backends = ('cpu', 'cuda')
    checkpoint = 'full'
    validation = ('engine: LISFLOOD-FP ACC to round-off (tools/flood/'
                  'validate_engine.py), float64 mass ledger; lisflood: '
                  'volume-exact against the LISFLOOD-FP mass record and the '
                  'depth rasters (tests/test_coupling_flood.py); hydraulics '
                  'are LISFLOOD-FP\'s own (not re-verified here)')
    imported_fluxes = ('runoff', 'inflow')
    exported_fluxes = ('outflow',)

    @property
    def owned_state(self):
        return (self.name + '.depth',)

    def __init__(self, basin=None, dem=None, dx=None, interval_s=86400.0,
                 solver=None, inflow_points=(), free_edges=(), fpfric=0.03,
                 rain_levels=1, initial_tstep=1.0, binary=None, workdir=None,
                 keep_work=False, flood_threshold=0.01,
                 initial_depth=None, timeout=None, extra=(), velocity=True,
                 dependencies=(), name=None, backend=None, dtype='float64',
                 max_dt=10.0, condition='auto', fdir=None,
                 fill_min_depth=5.0, burn_min_area_km2=5000.0,
                 inflow_nodes=None, inflow_map=None,
                 node_grid='river_nodes', block=64, devices=None,
                 hand_over=False, return_outlets=None,
                 infiltration_mm_day=0.0, open_water_evap_mm_day=0.0,
                 vic_map=None, **config):
        super().__init__(**config)
        if name:
            self.name = name
        if basin is not None:
            b = load_basin(basin)
            dem, dx = b['dem'], float(b['dx'])
            self.meta = b
        else:
            self.meta = None
        if dem is None or dx is None:
            raise CouplingError('flood: give `basin` (a basin .npz) or '
                                '`dem` and `dx`')
        self.dem = np.asarray(dem, dtype=np.float64)
        self.ny, self.nx = self.dem.shape
        self.dx = float(dx)
        self.interval_s = float(interval_s)
        self.solver = solver
        self.points = [tuple(int(v) for v in p) for p in inflow_points]
        self.free_edges = [(str(e).upper(), float(s)) for e, s in free_edges]
        for e, _ in self.free_edges:
            if e not in ('N', 'S', 'E', 'W'):
                raise CouplingError('flood: free edge %r is not N/S/E/W' % e)
        for r, c in self.points:
            if not (0 <= r < self.ny and 0 <= c < self.nx):
                raise CouplingError('flood: inflow point (%d, %d) is '
                                    'outside the %d x %d grid'
                                    % (r, c, self.ny, self.nx))
        if len(set(self.points)) != len(self.points):
            raise CouplingError('flood: two inflow points in one cell')
        self.fpfric = float(fpfric)
        self.rain_levels = max(int(rain_levels), 1)
        self.initial_tstep = float(initial_tstep)
        self.binary = binary
        self.workdir = workdir
        self.keep_work = bool(keep_work)
        self.threshold = float(flood_threshold)
        self.timeout = timeout
        self.extra = list(extra)
        self.velocity = bool(velocity)
        self.dependencies = tuple(dependencies)
        self.initial_depth = initial_depth
        if backend is None:
            backend = 'lisflood' if solver is not None else 'engine'
        if backend not in BACKENDS:
            raise CouplingError('flood: backend %r (have %s)'
                                % (backend, ', '.join(BACKENDS)))
        self.backend = backend
        self.dtype_name = str(dtype)
        if self.dtype_name not in ('float64', 'float32', 'mixed'):
            raise CouplingError('flood: dtype %r: float64, float32 or mixed'
                                % dtype)
        self.max_dt = float(max_dt)
        self.condition = condition
        self.fdir = fdir or FDIR
        self.fill_min_depth = float(fill_min_depth)
        self.burn_min_area_km2 = float(burn_min_area_km2)
        self.block = int(block)
        # opt-in multi-GPU engine (flood/multigpu.py): e.g. [cuda:0, cuda:1];
        # the first is the run's device, results equal one GPU bit for bit
        self.devices = [str(d) for d in devices] if devices else None
        # node -> the inflow points that share its discharge
        imap = []
        if inflow_nodes is not None:
            imap += [(int(k), list(range(len(self.points))))
                     for k in inflow_nodes]
        for k, pts in (inflow_map or ()):
            pts = [int(i) for i in pts]
            if not pts or min(pts) < 0 or max(pts) >= len(self.points):
                raise CouplingError('flood: inflow_map node %s: point '
                                    'indices outside the inflow points'
                                    % (k,))
            imap.append((int(k), pts))
        if imap and not self.points:
            raise CouplingError('flood: inflow_nodes / inflow_map need '
                                'inflow_points')
        self.inflow_map = imap
        self.inflow_nodes = [k for k, _ in imap] if imap else None
        self.node_grid = node_grid
        self._init_two_way(hand_over, return_outlets, infiltration_mm_day,
                           open_water_evap_mm_day, vic_map)
        self.engine = None
        self.condition_info = None
        self.inputs = (
            VarSpec('runoff', 'm3', ('cell',), kind='flux', grid=GRID,
                    description='water entering each hydraulic cell over '
                    'the interval (VIC runoff x contributing area, mapped)'),
        )
        if self.points:
            self.inputs += (
                VarSpec('inflow', 'm3', ('point',), kind='flux',
                        grid=INFLOW_GRID,
                        description='routed upstream water entering at the '
                        'inflow points over the interval'),)
        if self.inflow_nodes is not None:
            self.inputs += (
                VarSpec('upstream_discharge', 'm3 s-1', ('node',),
                        kind='state', grid=self.node_grid,
                        description='discharge at the routing gauge nodes; '
                        'the nodes in inflow_nodes feed the inflow points'),)
        self.outputs = (
            VarSpec('depth', 'm', ('cell',), kind='state', grid=GRID,
                    description='water depth at the interval end'),
            VarSpec('max_depth', 'm', ('cell',), kind='state', grid=GRID,
                    description='maximum depth during the interval'),
            VarSpec('speed', 'm s-1', ('cell',), kind='state', grid=GRID,
                    description='flow speed at the interval end (cell '
                    'centre, from the face velocities)'),
            VarSpec('outflow', 'm3', ('domain',), kind='flux',
                    grid='flood_domain',
                    description='volume that left through open boundaries'
                    + (' outside every return outlet'
                       if self.outlets else '')),
        )
        self._declare_two_way()
        self.depth = None
        self.t = 0.0
        self.device = 'cpu'
        self.totals = None
        self.n_runs = 0
        self._tmp = None
        self.last = {}
        self._dem_path = None
        self.timing = dict(run_s=0.0, solver_s=0.0)

    # -- declarations ------------------------------------------------------
    def grids(self):
        g = [GridSpec(GRID, self.ny * self.nx,
                      area=torch.full((self.ny * self.nx,), self.dx ** 2,
                                      dtype=torch.float64),
                      description='hydraulic cells, row-major, row 0 north'),
             GridSpec('flood_domain', 1,
                      area=torch.ones(1, dtype=torch.float64))]
        if self.points:
            g.append(GridSpec(INFLOW_GRID, len(self.points)))
        if self.outlets:
            g.append(GridSpec(OUTLET_GRID, len(self.outlets),
                              description='return outlets (groups of open '
                              'boundary faces)'))
        if self.losses and self.vic_pairs is not None:
            src, tgt, fr = self.vic_pairs
            tot = np.zeros(self.ny * self.nx)
            np.add.at(tot, tgt, fr)
            w = fr / np.where(tot[tgt] > 0, tot[tgt], 1.0)
            ar = np.zeros(self.n_vic)
            np.add.at(ar, src, w * self.dx ** 2)
            g.append(GridSpec(VIC_GRID, self.n_vic,
                              area=torch.as_tensor(ar, dtype=torch.float64),
                              description='the VIC cells of vic_map (area: '
                              'the hydraulic area each one owns)'))
        return g

    # -- two-way -----------------------------------------------------------
    def _init_two_way(self, hand_over, return_outlets, infil, evap,
                      vic_map):
        self.hand_over = bool(hand_over)
        self.infil_mm_day = float(infil or 0.0)
        self.evap_mm_day = float(evap or 0.0)
        if self.infil_mm_day < 0 or self.evap_mm_day < 0:
            raise CouplingError('flood: infiltration_mm_day and '
                                'open_water_evap_mm_day must be >= 0')
        outs = []
        for o in (return_outlets or ()):
            o = [o] if isinstance(o, str) else list(o)
            e = str(o[0]).upper()
            if e not in ('N', 'S', 'E', 'W') or len(o) not in (1, 3):
                raise CouplingError('flood: return outlet %r: [edge] or '
                                    '[edge, i0, i1]' % (o,))
            if e not in dict(self.free_edges):
                raise CouplingError('flood: return outlet on edge %s, which '
                                    'is not open (free_edges)' % e)
            m = self.ny if e in 'WE' else self.nx
            i0, i1 = (0, m) if len(o) == 1 else (int(o[1]), int(o[2]))
            if not 0 <= i0 < i1 <= m:
                raise CouplingError('flood: return outlet %r outside the '
                                    'edge (0..%d)' % (o, m))
            outs.append((e, i0, i1))
        self.outlets = outs
        self.losses = self.infil_mm_day > 0 or self.evap_mm_day > 0
        self.vic_pairs = None
        self.n_vic = 0
        if vic_map is not None:
            if isinstance(vic_map, (str, Path)):
                a = np.loadtxt(str(vic_map), delimiter=',', skiprows=1,
                               ndmin=2)
            else:
                a = np.asarray(vic_map, dtype=np.float64).reshape(-1, 3)
            src, tgt, fr = a[:, 0].astype(np.int64), a[:, 1].astype(
                np.int64), a[:, 2]
            if tgt.min() < 0 or tgt.max() >= self.ny * self.nx or \
                    src.min() < 0 or (fr < 0).any():
                raise CouplingError('flood: vic_map pairs outside the grid')
            self.vic_pairs = (src, tgt, fr)
            self.n_vic = int(src.max()) + 1
        if self.infil_mm_day > 0 and self.vic_pairs is None:
            raise CouplingError('flood: infiltration needs vic_map (which '
                                'VIC cell a hydraulic cell belongs to)')
        self.two_way = bool(self.hand_over or self.outlets or self.losses)
        if self.two_way and self.backend != 'engine':
            raise CouplingError('flood: two-way coupling needs the engine '
                                'backend (LISFLOOD-FP keeps no per-face '
                                'outflow record)')
        if self.hand_over and not self.points:
            raise CouplingError('flood: hand_over needs inflow_points')

    def _declare_two_way(self):
        if not self.two_way:
            return
        ex, im = ['outflow'], list(self.imported_fluxes)
        if self.hand_over:
            self.outputs += (
                VarSpec('river_request', 'm3', ('point',), kind='state',
                        grid=INFLOW_GRID,
                        description='take everything that passes the '
                        'entry sites (for routing.withdrawal_request)'),)
        if self.outlets:
            self.outputs += (
                VarSpec('returned', 'm3', ('outlet',), kind='flux',
                        grid=OUTLET_GRID,
                        description='water that left through each return '
                        'outlet over the interval (for routing.injection)'),)
            ex.append('returned')
        if self.infil_mm_day > 0:
            self.outputs += (
                VarSpec('infiltration', 'm3', ('vic_cell',), kind='flux',
                        grid=VIC_GRID,
                        description='floodplain infiltration per VIC cell '
                        '(for vic.soil_add)'),)
            self.inputs += (
                VarSpec('infiltration_unmet', 'm3', ('vic_cell',),
                        kind='flux', grid=VIC_GRID,
                        description='the infiltration VIC could not hold '
                        '(vic.soil_add_unmet), back onto the floodplain'),)
            ex.append('infiltration')
            im.append('infiltration_unmet')
        self.exported_fluxes = tuple(ex)
        self.imported_fluxes = tuple(im)

    def _two_way_setup(self, dev):
        """Device index tensors of the outlets and the VIC map."""
        e = self.engine
        n_out = len(self.outlets)
        nf = 2 * (self.ny + self.nx)
        fo = torch.full((nf,), n_out, dtype=torch.long, device=dev)
        for k, (ed, i0, i1) in enumerate(self.outlets):
            idx = e.edge_index(ed, i0, i1)
            if bool((fo[idx] != n_out).any()):
                raise CouplingError('flood: two return outlets share a face')
            fo[idx] = k
        self._face_outlet = fo
        self._ret_sum = SegmentSum(fo, n_out + 1, dev)
        if self.vic_pairs is not None:
            src, tgt, fr = self.vic_pairs
            n = self.ny * self.nx
            # a hydraulic cell's infiltration goes to its VIC cells in the
            # proportion of their fractions there (one owner: all of it)
            tot = np.zeros(n)
            np.add.at(tot, tgt, fr)
            w = np.where(tot[tgt] > 0, fr / np.where(tot[tgt] > 0,
                                                     tot[tgt], 1.0), 0.0)
            T = lambda x, d=torch.float64: torch.as_tensor(x, dtype=d,
                                                           device=dev)
            self._vm = dict(src=T(src, torch.long), tgt=T(tgt, torch.long),
                            fr=T(fr), w=T(w),
                            owned=T(tot > 0, torch.bool).reshape(
                                self.ny, self.nx),
                            # fixed-order sums (no atomics): per VIC cell
                            # and per hydraulic cell
                            sum_vic=SegmentSum(src, self.n_vic, dev),
                            sum_hyd=SegmentSum(tgt, n, dev))

    # -- life cycle --------------------------------------------------------
    def initialize(self, context):
        self.device = context.device if context.device is not None else 'cpu'
        dev = torch.device(self.device)
        if self.backend == 'engine':
            self._init_engine(context, dev)
            return
        if self.solver is None:
            self.solver = 'acc_gpu' if dev.type == 'cuda' else 'acc_cpu'
        if self.solver not in SOLVERS:
            raise CouplingError('flood: solver %r (have %s)'
                                % (self.solver, ', '.join(SOLVERS)))
        if self.solver == 'acc_gpu' and dev.type != 'cuda':
            raise CouplingError('flood: acc_gpu needs the run on a CUDA '
                                'device (run.device: cuda)')
        binary = Path(self.binary or DEFAULT_BINARY)
        if not binary.exists():
            raise CouplingError('flood: LISFLOOD-FP binary not found: %s '
                                '(set `binary` or LISFLOOD_BIN)' % binary)
        base = self.workdir or tempfile.mkdtemp(prefix='flood_')
        self._tmp = None if self.workdir else base
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
        self._dem_path = str(self.base / 'dem.asc')
        write_asc(self._dem_path, self.dem, self.dx)
        d0 = np.zeros((self.ny, self.nx))
        if self.initial_depth is not None:
            d0 = np.asarray(self.initial_depth, float).reshape(
                self.ny, self.nx).copy()
        self.depth = torch.as_tensor(d0, dtype=torch.float64)
        self.t = float(getattr(context, 't', 0.0))
        v0 = float(self.depth.sum()) * self.dx ** 2
        self.totals = dict(rain=0.0, inflow=0.0, outflow=0.0, stored=v0,
                           stored_start=v0)

    def conditioned_dem(self):
        """The DEM the engine runs on: a basin's DEM with its deep pits
        filled and the flow map's rivers burnt in (condition 'auto' or
        True), or the given DEM (condition False / a plain `dem`)."""
        cond = self.condition
        if cond == 'auto':
            cond = self.meta is not None
        if not cond:
            return self.dem, None
        from ...flood import dem as D
        z = self.dem
        info = dict(fill_min_depth=self.fill_min_depth)
        if self.meta is not None and 'lon0' in self.meta and self.fdir \
                and self.burn_min_area_km2 > 0:
            m = self.meta
            lon0, lat0 = float(m['lon0']), float(m['lat0'])
            ml, mt = float(m['m_lon']), float(m['m_lat'])
            ny, dx = self.ny, self.dx

            def rc(lon, lat):
                return (ny - (lat - lat0) * mt / dx - 0.5,
                        (lon - lon0) * ml / dx - 0.5)
            box = (lon0, float(m['lon1']), lat0, float(m['lat1']))
            riv = D.river_cells(self.fdir, box, self.burn_min_area_km2)
            z, burnt = D.burn_rivers(z, rc, riv)
            info.update(river_cells=len(riv), burnt_cells=int(burnt.sum()))
        z, ci = D.condition(z, 'fill', self.fill_min_depth)
        info.update(ci)
        return z, info

    def _init_engine(self, context, dev):
        from ...flood import FloodEngine
        z, self.condition_info = self.conditioned_dem()
        self.dem_used = z
        mixed = self.dtype_name == 'mixed'
        dt_ = torch.float64 if mixed else getattr(torch, self.dtype_name)
        mg = {}
        if self.devices and len(self.devices) > 1:
            def idx(d):
                d = torch.device(d)
                return (d.type, d.index if d.index is not None or
                        d.type != 'cuda' else torch.cuda.current_device())
            if self.dtype_name != 'float64':
                raise CouplingError('flood: devices (multi-GPU) needs dtype '
                                    'float64')
            if idx(self.devices[0]) != idx(dev):
                raise CouplingError('flood: devices[0] %s must be the run '
                                    'device %s' % (self.devices[0], dev))
            mg = dict(devices=self.devices)
        self.engine = FloodEngine(
            z, self.dx, self.fpfric, free_edges=dict(self.free_edges),
            inflow_points=self.points, max_dt=self.max_dt, device=dev,
            dtype=dt_, block=self.block, mixed=mixed, **mg,
            **(dict(track_edges=True) if self.two_way else {}))
        d0 = np.zeros((self.ny, self.nx))
        if self.initial_depth is not None:
            d0 = np.asarray(self.initial_depth, float).reshape(
                self.ny, self.nx).copy()
        self.engine.reset(depth=d0)
        self.depth = self.engine.h
        self.t = float(getattr(context, 't', 0.0))
        v0 = self.engine.storage()
        self.totals = dict(rain=0.0, inflow=0.0, outflow=0.0, stored=v0,
                           stored_start=v0)
        if self.two_way:
            self._two_way_setup(dev)
            self.totals.update(returned=0.0, lost=0.0, infiltration=0.0,
                               evaporation=0.0, unmet=0.0)
        if self.condition_info:
            context.say('%s: DEM conditioned %s' % (self.name,
                                                    self.condition_info))

    def finalize(self, context):
        if self._tmp and not self.keep_work:
            shutil.rmtree(self._tmp, ignore_errors=True)

    # -- one interval ------------------------------------------------------
    @staticmethod
    def _as_np(x, n):
        if x is None:
            return np.zeros(n)
        return torch.as_tensor(x).detach().to(
            'cpu', torch.float64).numpy().reshape(-1)

    def _rain_levels(self, V_rain, dt):
        """Rainfall levels (mm, falling until the next level's time) for
        the interval: a dry first step of `initial_tstep` seconds, then
        `rain_levels` equal levels that carry the whole volume, then a
        closing zero level at dt.

        Why a dry first step: the CUDA solver skips the rainfall of its
        first time step (seen in a volume-check run: exactly rate x the first
        step, 12.669 m3 of 2533.9 m3 in a 200 s test, and
        initial_tstep 0.001 s shrank the loss 1000 fold). The first step
        is initial_tstep long on both solvers, so putting no water in
        [0, initial_tstep) loses nothing and hands the same volume to
        either."""
        nl, tau = self.rain_levels, min(self.initial_tstep, 0.5 * dt)
        mm = (V_rain / self.dx ** 2 * 1000.0).reshape(self.ny, self.nx)
        lev = np.zeros((nl + 2,) + mm.shape)
        lev[1:nl + 1] = mm / nl
        times = np.concatenate([[0.0], tau + (dt - tau) * np.arange(nl + 1)
                                / nl]) / 3600.0
        return lev, times

    def _as_dev(self, x, n):
        dev = self.engine.device
        if x is None:
            return torch.zeros(n, dtype=torch.float64, device=dev)
        return torch.as_tensor(x).detach().to(dev, torch.float64).reshape(-1)

    def _inputs_engine(self, inputs, dt):
        """The interval's volumes on the engine's device, and the device
        flags for their validity (no host read)."""
        n = self.ny * self.nx
        V_rain = self._as_dev(inputs.get('runoff'), n)
        if V_rain.shape[0] != n:
            raise ComputeError('flood: runoff has %d cells, the grid %d'
                               % (V_rain.shape[0], n))
        flags = [DeviceFlag('flood: runoff must be finite and >= 0',
                            ~torch.isfinite(V_rain).all()
                            | (V_rain < 0).any())]
        V_pt = self._as_dev(inputs.get('inflow'), len(self.points))
        if self.inflow_nodes is not None and \
                inputs.get('upstream_discharge') is not None:
            q = self._as_dev(inputs['upstream_discharge'], -1)
            if max(self.inflow_nodes) >= q.shape[0]:
                raise ComputeError('flood: inflow node outside the %d '
                                   'nodes' % q.shape[0])
            V_pt = V_pt.clone()
            if not hasattr(self, '_pt_idx'):
                self._pt_idx = [(k, torch.as_tensor(
                    pts, dtype=torch.long, device=V_pt.device))
                    for k, pts in self.inflow_map]
            for k, pidx in self._pt_idx:
                # true division by a device scalar (a CUDA division by a
                # host scalar multiplies by its reciprocal; numpy divides)
                n_k = torch.full((), float(len(pidx)), dtype=torch.float64,
                                 device=V_pt.device)
                V_pt.index_add_(0, pidx, ((q[k] * dt) / n_k).expand(
                    len(pidx)))
        flags.append(DeviceFlag('flood: inflow must be finite and >= 0',
                                ~torch.isfinite(V_pt).all()
                                | (V_pt < 0).any()))
        return V_rain, V_pt, flags

    def compute(self, inputs, context, dt):
        if self.backend == 'engine':
            V_rain, V_pt, flags = self._inputs_engine(inputs, dt)
            if self.two_way:
                prop = self._compute_two_way(V_rain, V_pt, inputs, context,
                                             dt, flags)
            else:
                prop = self._compute_engine(V_rain, V_pt, context, dt)
            prop.flags += flags
            return prop
        n = self.ny * self.nx
        V_rain = self._as_np(inputs.get('runoff'), n)
        if V_rain.shape[0] != n:
            raise ComputeError('flood: runoff has %d cells, the grid %d'
                               % (V_rain.shape[0], n))
        if not np.all(np.isfinite(V_rain)) or (V_rain < 0).any():
            raise ComputeError('flood: runoff must be finite and >= 0')
        V_pt = self._as_np(inputs.get('inflow'), len(self.points))
        if self.inflow_nodes is not None and \
                inputs.get('upstream_discharge') is not None:
            q = self._as_np(inputs['upstream_discharge'], -1)
            if max(self.inflow_nodes) >= q.shape[0]:
                raise ComputeError('flood: inflow node outside the %d '
                                   'nodes' % q.shape[0])
            V_pt = V_pt.copy()
            for k, pts in self.inflow_map:
                V_pt[pts] += q[k] * dt / len(pts)
        if not np.all(np.isfinite(V_pt)) or (V_pt < 0).any():
            raise ComputeError('flood: inflow must be finite and >= 0')
        if self.backend == 'engine':
            return self._compute_engine(V_rain, V_pt, context, dt)
        d0 = self.depth.numpy()
        rain = times = None
        if V_rain.any():
            rain, times = self._rain_levels(V_rain, dt)
        pts = [(r, c, V_pt[i] / dt) for i, (r, c) in enumerate(self.points)]
        wd = self.base / ('step_%05d' % self.n_runs)
        if wd.exists():
            shutil.rmtree(wd)
        res = run_lisflood(
            wd, self._dem_path, self.ny, self.nx, self.dx, dt, self.solver,
            d0, rain_mm=rain, rain_times_h=times, points=pts,
            free_edges=self.free_edges, fpfric=self.fpfric,
            initial_tstep=self.initial_tstep, massint=dt / 200.0,
            binary=self.binary, extra=self.extra, timeout=self.timeout,
            velocity=self.velocity)
        if res.returncode != 0 or res.depth is None:
            raise ComputeError('flood: LISFLOOD-FP failed (exit %d): %s'
                               % (res.returncode, res.log[-400:]))
        if not np.all(np.isfinite(res.depth)):
            raise ComputeError('flood: LISFLOOD-FP returned non-finite depth')
        d1 = res.depth
        a = self.dx ** 2
        S0, S1 = float(d0.sum()) * a, float(d1.sum()) * a
        I_rain, I_pt = float(V_rain.sum()), float(V_pt.sum())
        I_ext = I_rain + I_pt
        dS = S1 - S0
        mass = res.mass
        q_in = float(mass['Qin'][-1]) * dt if mass else None   # constant QFIX
        q_out = integrate(mass['Time'], mass['Qout'], dt) if mass else None
        # closed edges: nothing can leave, so the balance is a real check;
        # open edges: the outflow is what the rasters show left (the CUDA
        # solver's own record cannot be used, see the module docstring)
        # and, where the solver reports it (CPU), it is cross-checked in
        # validate
        if not self.free_edges:
            Q_out, mode = 0.0, 'closed'
        else:
            Q_out, mode = I_ext - dS, 'derived'
        info = dict(wall_s=res.wall, loop_s=res.loop_s, solver=self.solver,
                    I_rain=I_rain, I_inflow=I_pt, stored0=S0, stored1=S1,
                    outflow_mode=mode, mass_qin=q_in, mass_qout=q_out,
                    mass_vol=float(mass['Vol'][-1]) if mass else None)
        # precision of this connector's data path: LISFLOOD-FP accumulates
        # in double over thousands of steps and exchanges rasters as text
        # (14 decimals), so its balance holds to ~1e-12 of the throughput,
        # not to machine epsilon; the ledger tolerance is 4 x eps x scale
        bal = Balance(I_ext=I_ext, Q_out=Q_out, dS=dS, n_terms=1,
                      eps=EPS_PATH, scale=I_ext + S0 + S1 + abs(Q_out),
                      detail=dict(rain=I_rain, inflow=I_pt))
        dev = torch.device(self.device)
        d1t = torch.as_tensor(d1, dtype=torch.float64)
        mxd = res.max_depth if res.max_depth is not None else d1
        exports = dict(
            depth=d1t.reshape(-1).to(dev),
            max_depth=torch.as_tensor(np.maximum(mxd, d1),
                                      dtype=torch.float64).reshape(-1)
            .to(dev),
            speed=torch.as_tensor(
                cell_speed(res.vx, res.vy) if res.vx is not None
                and res.vy is not None else np.zeros_like(d1),
                dtype=torch.float64).reshape(-1).to(dev),
            outflow=torch.full((1,), Q_out, dtype=torch.float64, device=dev))
        f64 = torch.float64
        series = dict(
            stored_m3=torch.tensor([S1], dtype=f64),
            handed_m3=torch.tensor([I_ext], dtype=f64),
            outflow_m3=torch.tensor([Q_out], dtype=f64),
            flooded_area_m2=torch.tensor(
                [float((d1 > self.threshold).sum()) * a], dtype=f64),
            max_depth_m=torch.tensor([float(np.max(d1))], dtype=f64))
        if not self.keep_work:
            shutil.rmtree(wd, ignore_errors=True)
        return Proposal(connector=self.name, t0=context.t,
                        t1=context.t + dt,
                        state=dict(depth=d1t, t=context.t + dt),
                        exports=exports, balance=bal, series=series,
                        backend='cuda' if self.solver == 'acc_gpu' else 'cpu',
                        info=info)

    def _compute_engine(self, V_rain, V_pt, context, dt):
        e = self.engine
        before = e.state_dict()
        # the handed volumes are summed on the host as the LISFLOOD path
        # does (numpy order): their copies are queued before the run and
        # are complete after the run's own bulk read (no extra sync)
        hv = [x.to('cpu', non_blocking=True) for x in (V_rain, V_pt)]
        ev = None
        if V_rain.device.type == 'cuda':
            ev = torch.cuda.Event()
            ev.record()
        # rate = volume / (A T) divided element-wise as on the host (a CUDA
        # division by a host scalar is a multiply by its reciprocal)
        AT = torch.full((), e.area * float(dt), dtype=torch.float64,
                        device=V_rain.device)
        try:
            out = e.run(dt, rate=V_rain / AT,
                        inflow=V_pt if len(self.points) else None,
                        check=False)
            after = e.state_dict()
            exports = dict(depth=out['depth'].to(torch.float64).clone(),
                           max_depth=out['max_depth'].to(
                               torch.float64).clone(),
                           speed=out['speed'].to(torch.float64).clone())
        finally:
            e.load_state_dict(before)          # a proposal only
        L = out['ledger']
        a = self.dx ** 2
        if ev is not None:
            ev.synchronize()                   # already complete
        I_rain = float(hv[0].numpy().sum())
        I_pt = float(hv[1].numpy().sum())
        I_ext = L['rain'] + L['inflow']
        S0, S1 = L['stored0'], L['stored1']
        Q_out = L['outflow']
        dev = e.device
        exports['outflow'] = torch.full((1,), Q_out, dtype=torch.float64,
                                        device=dev)
        eps = torch.finfo(e.dtype).eps
        # the ledger's terms: sources as the engine added them (rate x
        # sum of steps = the handed volume to rounding), boundary outflow,
        # storage summed over ny x nx cells; clamp = water a negative depth
        # reset created (0 with the wet/dry limiter)
        bal = Balance(I_ext=I_ext, Q_out=Q_out, dS=S1 - S0,
                      closure=-L['clamp'], n_terms=self.ny * self.nx,
                      eps=eps, scale=I_ext + S0 + S1 + abs(Q_out),
                      detail=dict(rain=L['rain'], inflow=L['inflow'],
                                  handed_rain=I_rain, handed_inflow=I_pt,
                                  steps=L['steps']))
        d1 = exports['depth']
        f64 = torch.float64
        series = dict(
            stored_m3=torch.tensor([S1], dtype=f64),     # host values: no copy
            handed_m3=torch.tensor([I_rain + I_pt], dtype=f64),
            outflow_m3=torch.tensor([Q_out], dtype=f64),
            flooded_area_m2=((d1 > self.threshold).sum().to(f64)
                             * a).reshape(1),
            max_depth_m=d1.max().reshape(1))
        info = dict(wall_s=None, loop_s=None, solver='engine',
                    I_rain=I_rain, I_inflow=I_pt, stored0=S0, stored1=S1,
                    outflow_mode='ledger', mass_qin=None, mass_qout=None,
                    mass_vol=None, steps=L['steps'], dt_min=L['dt_min'],
                    rel_residual=L['rel_residual'])
        return Proposal(connector=self.name, t0=context.t, t1=context.t + dt,
                        state=dict(engine=after, t=context.t + dt),
                        exports=exports, balance=bal, series=series,
                        backend=torch.device(dev).type, info=info)

    def _compute_two_way(self, V_rain, V_pt, inputs, context, dt, flags):
        """One interval of the two-way connector: the engine run with
        per-face outflow, the return outlets, the interval-end losses, the
        balance. One bulk host read (besides the engine's own)."""
        e, f64 = self.engine, torch.float64
        dev = e.device
        n, A = self.ny * self.nx, self.dx ** 2
        V_un = None
        if 'infiltration_unmet' in [v.name for v in self.inputs] and \
                inputs.get('infiltration_unmet') is not None:
            un = self._as_dev(inputs['infiltration_unmet'], self.n_vic)
            # VIC's unmet = added - applied carries its rounding (a few
            # ulp below 0 where everything fitted): kept as it is (an
            # overdrawn dry cell is the engine's ledgered clamp); a clearly
            # negative value is refused
            flags.append(DeviceFlag('flood: infiltration_unmet must be '
                                    'finite and >= 0',
                                    ~torch.isfinite(un).all()
                                    | (un < -1e-6).any()))
            vm = self._vm
            V_un = vm['sum_hyd'](un[vm['src']] * vm['fr'])
            V_rain = V_rain + V_un
        before = e.state_dict()
        AT = torch.full((), e.area * float(dt), dtype=f64, device=dev)
        try:
            out = e.run(dt, rate=V_rain / AT,
                        inflow=V_pt if len(self.points) else None,
                        check=False)
            after = e.state_dict()
        finally:
            e.load_state_dict(before)          # a proposal only
        L = out['ledger']
        n_out = len(self.outlets)
        ret = self._ret_sum(out['edge_out'].to(f64))
        h = after['h']
        zero = torch.zeros((), dtype=f64, device=dev)
        V_e = V_i = zero
        infil_vic = None
        if self.losses:
            wet = h > self.threshold
            hd = h.to(f64)
            E_ = self.evap_mm_day * 1e-3 * dt / 86400.0
            I_ = self.infil_mm_day * 1e-3 * dt / 86400.0
            ev = torch.where(wet, torch.clamp(hd, max=E_), 0.0)
            r = hd - ev
            if I_ > 0:
                can = wet & self._vm['owned']
                inf = torch.where(can, torch.clamp(r, max=I_), 0.0)
            else:
                inf = torch.zeros_like(r)
            hn = r - inf
            after['h'] = hn.to(h.dtype)
            # the losses are what the stored depth lost (working precision)
            lost_cell = hd - after['h'].to(f64)
            inf = torch.where(lost_cell > 0, torch.minimum(inf, lost_cell),
                              torch.zeros_like(inf))
            ev = lost_cell - inf
            V_e = ev.sum() * A
            V_i = inf.sum() * A
            if I_ > 0:
                vm = self._vm
                vi = inf.reshape(-1) * A
                infil_vic = vm['sum_vic'](vi[vm['tgt']] * vm['w'])
        S1t = after['h'].to(f64).sum() * A
        # one bulk read: outlets + lost, the losses, the end storage, the
        # handed volumes (summed in float64 on the device)
        rd = torch.cat([ret, V_e.reshape(1), V_i.reshape(1),
                        S1t.reshape(1), V_rain.sum().reshape(1),
                        V_pt.sum().reshape(1),
                        (V_un.sum() if V_un is not None else zero).reshape(1)
                        ]).tolist()
        returned, lost = rd[:n_out], rd[n_out]
        Ve, Vi, S1, I_rain, I_pt, I_un = rd[n_out + 1:]
        S0 = L['stored0']
        I_ext = L['rain'] + L['inflow']
        Q_out = sum(returned) + lost + Vi
        exports = dict(depth=after['h'].reshape(-1).to(f64).clone(),
                       max_depth=out['max_depth'].to(f64).clone(),
                       speed=out['speed'].to(f64).clone(),
                       outflow=ret[n_out:].clone())
        if self.outlets:
            exports['returned'] = ret[:n_out].clone()
        if self.hand_over:
            exports['river_request'] = torch.full(
                (len(self.points),), REQUEST_ALL, dtype=f64, device=dev)
        if infil_vic is not None:
            exports['infiltration'] = infil_vic
        eps = torch.finfo(e.dtype).eps
        bal = Balance(I_ext=I_ext, Q_out=Q_out, E_other=Ve, dS=S1 - S0,
                      closure=-L['clamp'], n_terms=n + 2 * (self.ny + self.nx),
                      eps=eps, scale=I_ext + S0 + S1 + abs(Q_out) + Ve,
                      detail=dict(rain=L['rain'], inflow=L['inflow'],
                                  handed_rain=I_rain - I_un,
                                  handed_inflow=I_pt, unmet=I_un,
                                  returned=sum(returned), lost=lost,
                                  infiltration=Vi, evaporation=Ve,
                                  engine_outflow=L['outflow'],
                                  steps=L['steps']))
        d1 = exports['depth']
        series = dict(
            stored_m3=torch.tensor([S1], dtype=f64),
            handed_m3=torch.tensor([I_rain + I_pt], dtype=f64),
            outflow_m3=torch.tensor([lost + sum(returned)], dtype=f64),
            returned_m3=torch.tensor([sum(returned)], dtype=f64),
            infiltration_m3=torch.tensor([Vi], dtype=f64),
            evaporation_m3=torch.tensor([Ve], dtype=f64),
            flooded_area_m2=((d1 > self.threshold).sum().to(f64)
                             * A).reshape(1),
            max_depth_m=d1.max().reshape(1))
        info = dict(wall_s=None, loop_s=None, solver='engine',
                    I_rain=I_rain, I_inflow=I_pt, stored0=S0, stored1=S1,
                    outflow_mode='edges', mass_qin=None, mass_qout=None,
                    mass_vol=None, steps=L['steps'], dt_min=L['dt_min'],
                    rel_residual=L['rel_residual'], returned=returned,
                    lost=lost, infiltration=Vi, evaporation=Ve, unmet=I_un)
        return Proposal(connector=self.name, t0=context.t, t1=context.t + dt,
                        state=dict(engine=after, t=context.t + dt),
                        exports=exports, balance=bal, series=series,
                        backend=torch.device(dev).type, info=info)

    def validate(self, proposal, context):
        p = []
        i = proposal.info
        scale = max(i['I_rain'] + i['I_inflow'] + i['stored0'], 1.0)
        if proposal.balance.Q_out < -1e-9 * scale:
            p.append('%s: LISFLOOD-FP created water (derived outflow %.6g m3)'
                     % (self.name, proposal.balance.Q_out))
        dt = proposal.t1 - proposal.t0
        if i['mass_qin'] is not None and i['I_inflow'] > 0:
            # the .mass record prints Qin with 3 decimals (m3/s)
            err = abs(i['mass_qin'] - i['I_inflow']) / dt
            if err > 6e-4 + 1e-6 * i['I_inflow'] / dt:
                p.append('%s: the .mass inflow (%.9g m3) differs from the '
                         'handed inflow (%.9g m3)'
                         % (self.name, i['mass_qin'], i['I_inflow']))
        if self.solver == 'acc_cpu' and self.free_edges and \
                i['mass_qout'] is not None:
            tol = 5e-3 * (i['I_rain'] + i['I_inflow']) + 1e-3 * dt
            if abs(i['mass_qout'] - proposal.balance.Q_out) > tol:
                p.append('%s: outflow from the rasters (%.9g m3) and from '
                         'the .mass record (%.9g m3) disagree'
                         % (self.name, proposal.balance.Q_out,
                            i['mass_qout']))
        p.append(DeviceFlag('%s: negative depth' % self.name,
                            (proposal.exports['depth'] < -1e-12).any()))
        return p

    def commit(self, proposal, context):
        if self.backend == 'engine':
            self.engine.load_state_dict(proposal.state['engine'])
            self.depth = self.engine.h
            self.t = proposal.state['t']
            i, T = proposal.info, self.totals
            T['rain'] += proposal.balance.detail['rain']
            T['inflow'] += proposal.balance.detail['inflow']
            T['outflow'] += proposal.balance.Q_out
            T['stored'] = i['stored1']
            if self.two_way:
                d = proposal.balance.detail
                T['returned'] += d['returned']
                T['lost'] += d['lost']
                T['infiltration'] += d['infiltration']
                T['evaporation'] += d['evaporation']
                T['unmet'] += d['unmet']
            self.n_runs += 1
            self.last = i
            return
        self.depth = proposal.state['depth']
        self.t = proposal.state['t']
        i, T = proposal.info, self.totals
        T['rain'] += i['I_rain']
        T['inflow'] += i['I_inflow']
        T['outflow'] += proposal.balance.Q_out
        T['stored'] = i['stored1']
        self.n_runs += 1
        self.last = i
        self.timing['run_s'] += i['wall_s']
        self.timing['solver_s'] += i['loop_s'] or 0.0

    # -- restart -----------------------------------------------------------
    def save_state(self):
        if self.backend == 'engine':
            st = {k: (v.cpu().clone() if torch.is_tensor(v) else v)
                  for k, v in self.engine.state_dict().items()}
            return dict(engine=st, t=self.t, totals=dict(self.totals),
                        n_runs=self.n_runs)
        return dict(depth=self.depth.cpu().clone(), t=self.t,
                    totals=dict(self.totals), n_runs=self.n_runs)

    def load_state(self, blob):
        if self.backend == 'engine':
            self.engine.load_state_dict(blob['engine'])
            self.depth = self.engine.h
            self.t = blob['t']
            self.totals = dict(blob['totals'])
            self.n_runs = int(blob['n_runs'])
            return
        self.depth = blob['depth'].to(torch.float64).cpu()
        self.t = blob['t']
        self.totals = dict(blob['totals'])
        self.n_runs = int(blob['n_runs'])

    # -- diagnostics -------------------------------------------------------
    def volume_record(self):
        """Handed / stored / out over the run and the closing error (m3)."""
        T = self.totals
        handed = T['rain'] + T['inflow']
        ds = T['stored'] - T['stored_start']
        if 'evaporation' in T:
            # two-way: outflow = returned + lost + infiltration
            return dict(handed=handed, stored_change=ds,
                        outflow=T['outflow'], returned=T['returned'],
                        lost=T['lost'], infiltration=T['infiltration'],
                        evaporation=T['evaporation'], unmet=T['unmet'],
                        residual=handed - T['outflow'] - T['evaporation']
                        - ds)
        return dict(handed=handed, stored_change=ds, outflow=T['outflow'],
                    residual=handed - T['outflow'] - ds)
