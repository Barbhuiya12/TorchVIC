"""VIC + routing + flood, one way or two way (--two-way), on the
Brahmaputra/Guwahati test basin.

VIC is the stored TorchVIC ISC run (<data root>/benchmark_isc/runs/
b112825_h200daily_r1, 112825 cells, daily OUT_RUNOFF + OUT_BASEFLOW)
replayed as the land-surface module (the VIC connector's global-file path
expands the ISC forcing at once and does not fit one GPU; the ISC driver
benchmarks/run_isc_torchvic.py is not runtime-pluggable); the routing
connector routes VIC's daily runoff on the ISC 1/16 degree network and its
discharge at the gauge cells just upstream of the flood box (the
Brahmaputra main stem and two tributaries that enter it) feeds the flood
connector's inflow points; VIC runoff + baseflow of the box's 256 cells
enters the hydraulic grid by the conservative mapping. The flood connector
runs its default backend (TorchVIC engine, float64, DEM with pits > 5 m
filled and the flow map's rivers burnt in), daily intervals. Prints one JSON
line per day and a RESULT line (wall time per module per day, the coupled
ledger).

    python benchmarks/bench_flood_coupled.py --start 2004-06-01 \
        --end 2004-07-31 --out DIR [--two-way] [--exit-gauge]

Two way (docs/connectors/flood.md): routing hands the river
over at the entry gauges (withdrawal of everything that passes, lag-1
request) into the inflow points, the flood model returns what leaves its
west edge to the routing site at the cell where the Brahmaputra's D8 path
leaves the box (injection, lag 1); `--evap-mm-day` adds open-water
evaporation (E_other). `--exit-gauge` (implied by --two-way) adds that cell
as a routing gauge, so one-way and two-way runs report the discharge
downstream of the box.
"""
import os
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchvic.coupling import (Connector, TorchVIC, VarSpec,    # noqa: E402
                               registry)
from torchvic.coupling.connector import Balance, Proposal       # noqa: E402

AD = Path(os.environ.get('TORCHVIC_DATA_ROOT', 'data'))
GLOBAL = AD / 'benchmark_isc/runs/b112825_h200daily_r1/global.txt'
FDIR = AD / 'static/isc_fdir_grit.nc'
DOMAIN = AD / 'vic_params_isc/domain.nc'
BASIN = AD / 'static/flood_basins/brahmaputra_guwahati_100m.npz'
PAIRS = AD / 'static/flood_basins/brahmaputra_guwahati_100m_vic_pairs.csv'
ESRI = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1),
        32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


class Replay(Connector):
    """The stored TorchVIC ISC run as the land-surface module: each day's
    runoff + baseflow (mm) of the domain's active cells."""
    name = 'vic'
    exported_fluxes = ('total_runoff',)
    backends = ('cpu', 'cuda')
    interval_s = 86400.0

    def __init__(self, fluxes, day0, mask, area, **c):
        super().__init__(**c)
        import netCDF4
        self.nc = netCDF4.Dataset(str(fluxes))
        self.day0, self.mask = int(day0), mask
        self.area = torch.as_tensor(area, dtype=torch.float64)
        self.outputs = (VarSpec('total_runoff', 'mm', ('cell',),
                                kind='flux', grid='vic_cells'),)

    def grids(self):
        from torchvic.coupling import GridSpec
        return [GridSpec('vic_cells', len(self.area), area=self.area,
                         kind='cells')]

    def compute(self, inputs, context, dt):
        k = self.day0 + int(round(context.t / 86400.0))
        r = (np.asarray(self.nc['OUT_RUNOFF'][k], dtype=np.float64)
             + np.asarray(self.nc['OUT_BASEFLOW'][k], dtype=np.float64))
        x = torch.as_tensor(np.nan_to_num(r[self.mask]),
                            device=context.device)
        v = float((x.cpu() * 1e-3 * self.area).sum())
        return Proposal(self.name, context.t, context.t + dt,
                        exports=dict(total_runoff=x),
                        balance=Balance(P=v, Q_out=v, n_terms=x.numel(),
                                        scale=2 * v + 1e-30),
                        backend=x.device.type)

    def commit(self, proposal, context):
        pass


class Split(Connector):
    """Partitions VIC's runoff between the river network and the flood
    box: each cell's water goes whole to one of them (the box's 256 cells to
    the flood model, the rest to routing), so no water is delivered
    twice."""
    name = 'split'
    imported_fluxes = ('runoff',)
    exported_fluxes = ('routed', 'box')
    backends = ('cpu', 'cuda')
    interval_s = 86400.0

    def __init__(self, box_cells, box_area, **c):
        super().__init__(**c)
        self.box_cells = torch.as_tensor(box_cells, dtype=torch.long)
        self.box_area = torch.as_tensor(box_area, dtype=torch.float64)
        self.inputs = (VarSpec('runoff', 'mm', ('cell',), kind='flux',
                               grid='vic_cells'),)
        self.outputs = (VarSpec('routed', 'mm', ('cell',), kind='flux',
                                grid='vic_cells'),
                        VarSpec('box', 'mm', ('cell',), kind='flux',
                                grid='box_cells'))

    def grids(self):
        from torchvic.coupling import GridSpec
        return [GridSpec('box_cells', len(self.box_cells),
                         area=self.box_area, kind='cells',
                         description='the VIC cells of the flood box')]

    def compute(self, inputs, context, dt):
        x = inputs['runoff']
        bi = self.box_cells.to(x.device)
        inbox = torch.zeros_like(x, dtype=torch.bool)
        inbox[bi] = True
        box = x[bi].clone()
        routed = torch.where(inbox, torch.zeros_like(x), x)
        n = float(x.sum())
        return Proposal(self.name, context.t, context.t + dt,
                        exports=dict(routed=routed, box=box),
                        balance=Balance(I_ext=n, Q_out=float(box.sum())
                                        + float(routed.sum()),
                                        n_terms=x.numel(), scale=2 * abs(n)
                                        + 1e-30),
                        backend=x.device.type)

    def commit(self, proposal, context):
        pass


def write_global(out, start, end):
    """The ISC global file with this run's dates and result directory."""
    y0, m0, d0 = start.split('-')
    y1, m1, d1 = end.split('-')
    rep = dict(STARTYEAR=y0, STARTMONTH=str(int(m0)), STARTDAY=str(int(d0)),
               ENDYEAR=y1, ENDMONTH=str(int(m1)), ENDDAY=str(int(d1)),
               RESULT_DIR=str(out / 'vic_out') + '/',
               LOG_DIR=str(out / 'vic_log') + '/')
    (out / 'vic_out').mkdir(parents=True, exist_ok=True)
    (out / 'vic_log').mkdir(parents=True, exist_ok=True)
    lines = []
    for ln in GLOBAL.read_text().splitlines():
        k = ln.split()[0] if ln.split() else ''
        lines.append('%-12s %s' % (k, rep[k]) if k in rep else ln)
    p = out / 'global.txt'
    p.write_text('\n'.join(lines) + '\n')
    return p


def entry_gauges(box):
    """Cells just outside the box whose D8 link enters it, by upstream
    area: [(upstream km2, lat, lon, entry lat, entry lon)]."""
    import netCDF4
    d = netCDF4.Dataset(str(FDIR))
    lat, lon = np.asarray(d['lat'][:]), np.asarray(d['lon'][:])
    fd, ua = np.asarray(d['fdir'][:]), np.asarray(d['upstream_area'][:])
    lon0, lon1, lat0, lat1 = box

    def inb(i, j):
        return lon0 < lon[j] < lon1 and lat0 < lat[i] < lat1
    ii = np.nonzero((lat > lat0 - 0.1) & (lat < lat1 + 0.1))[0]
    jj = np.nonzero((lon > lon0 - 0.1) & (lon < lon1 + 0.1))[0]
    out = []
    for i in ii:
        for j in jj:
            if inb(i, j) or int(fd[i, j]) not in ESRI:
                continue
            di, dj = ESRI[int(fd[i, j])]
            if inb(i + di, j + dj):
                out.append((float(ua[i, j]), float(lat[i]), float(lon[j]),
                            float(lat[i + di]), float(lon[j + dj])))
    return sorted(out, reverse=True)


def exit_cell(box, start):
    """(lat, lon) of the first cell outside the box on the D8 path from
    `start` (lat, lon)."""
    import netCDF4
    d = netCDF4.Dataset(str(FDIR))
    lat, lon = np.asarray(d['lat'][:]), np.asarray(d['lon'][:])
    fd = np.asarray(d['fdir'][:])
    lon0, lon1, lat0, lat1 = box
    i = int(np.argmin(abs(lat - start[0])))
    j = int(np.argmin(abs(lon - start[1])))
    seen_box = False
    for _ in range(10000):
        inb = lon0 < lon[j] < lon1 and lat0 < lat[i] < lat1
        if inb:
            seen_box = True
        elif seen_box:
            return float(lat[i]), float(lon[j])
        c = int(fd[i, j])
        if c not in ESRI:
            break
        i, j = i + ESRI[c][0], j + ESRI[c][1]
    raise SystemExit('no exit cell found from %s' % (start,))


def main():
    import netCDF4
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2004-05-01')
    ap.add_argument('--fluxes', default=str(
        AD / 'benchmark_isc/runs/b112825_h200daily_r1/out/'
        'fluxes.2004-01-01.nc'))
    ap.add_argument('--end', default='2004-07-31')
    ap.add_argument('--min-gauge-km2', type=float, default=10000.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    ap.add_argument('--two-way', action='store_true')
    ap.add_argument('--exit-gauge', action='store_true')
    ap.add_argument('--evap-mm-day', type=float, default=0.0)
    a = ap.parse_args()
    if a.two_way:
        a.exit_gauge = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import datetime as dtm
    d0 = dtm.date.fromisoformat(a.start)
    ndays = (dtm.date.fromisoformat(a.end) - d0).days + 1
    day0 = (d0 - dtm.date(2004, 1, 1)).days
    b = np.load(BASIN)
    dem = b['dem']
    ny, nx = dem.shape
    dx = float(b['dx'])
    box = (float(b['lon0']), float(b['lon1']), float(b['lat0']),
           float(b['lat1']))
    # VIC cell index (row-major active cells of the domain) of the box cells
    dom = netCDF4.Dataset(str(DOMAIN))
    dlat, dlon = np.asarray(dom['lat'][:]), np.asarray(dom['lon'][:])
    mask = np.asarray(dom['mask'][:]) > 0
    idx = -np.ones(mask.shape, dtype=np.int64)
    cells = np.argwhere(mask)
    idx[mask] = np.arange(len(cells))
    box_idx = []
    for lo, la in zip(b['vic_lon'], b['vic_lat']):
        i = int(np.argmin(abs(dlat - la)))
        j = int(np.argmin(abs(dlon - lo)))
        if idx[i, j] < 0:
            raise SystemExit('box cell %g %g not active' % (lo, la))
        box_idx.append(int(idx[i, j]))
    area = (np.asarray(dom['area'][:], dtype=np.float64)
            * np.asarray(dom['frac'][:], dtype=np.float64))[mask]
    pr = np.loadtxt(PAIRS, delimiter=',', skiprows=1)
    pairs = [[int(s), int(t), float(f)] for s, t, f in pr]
    # gauges: the river cells that enter the box
    gs = [g for g in entry_gauges(box) if g[0] >= a.min_gauge_km2]
    ex = exit_cell(box, (gs[0][1], gs[0][2])) if a.exit_gauge else None
    gcsv = out / 'gauges.csv'
    gcsv.write_text('lat,lng\n' + ''.join('%.5f,%.5f\n' % (g[1], g[2])
                                          for g in gs)
                    + ('%.5f,%.5f\n' % ex if ex else ''))
    # inflow points: the east main-stem river cells (as the connector
    # benchmark) for gauges entering through the east edge, the lowest
    # edge cells of the entry VIC cell for the others
    r_in, c_in = [int(v) for v in b['inflow_rc'][0]]
    col = dem[:, c_in]
    win = col[max(r_in - 80, 0):r_in + 80]
    lo = float(np.percentile(win, 15))
    east = [(k, c_in) for k in range(max(r_in - 80, 0), min(r_in + 80, ny))
            if abs(col[k] - lo) <= 0.7]
    points, groups = list(east), []
    ml, mt = float(b['m_lon']), float(b['m_lat'])

    def rc(lon, lat):
        return (ny - (lat - box[2]) * mt / dx - 0.5,
                (lon - box[0]) * ml / dx - 0.5)
    for g in gs:
        ua, la, lo_, ela, elo = g
        if lo_ > box[1]:                             # enters from the east
            groups.append(list(range(len(east))))
            continue
        r0, c0 = rc(elo - 0.03125, ela + 0.03125)
        r1, c1 = rc(elo + 0.03125, ela - 0.03125)
        r0, r1 = int(max(r0, 0)), int(min(r1, ny))
        c0, c1 = int(max(c0, 0)), int(min(c1, nx))
        if la > box[3]:                              # from the north
            line = [(3, c) for c in range(c0, c1)]
        elif la < box[2]:
            line = [(ny - 4, c) for c in range(c0, c1)]
        else:
            line = [(r, 3) for r in range(r0, r1)]
        zl = np.array([dem[p] for p in line])
        pick = [line[k] for k in np.nonzero(zl <= zl.min() + 2.0)[0]]
        g_ix = []
        for p in pick:
            if p not in points:
                points.append(p)
            g_ix.append(points.index(p))
        groups.append(g_ix)
    # routing node of each gauge: the router sorts its gauge cells
    cl, cc = dlat[cells[:, 0]], dlon[cells[:, 1]]
    gcell = [int(np.argmin((cl - g[1]) ** 2 + ((cc - g[2]) * np.cos(
        np.radians(g[1]))) ** 2)) for g in gs]
    ecell = None
    if ex is not None:
        ecell = int(np.argmin((cl - ex[0]) ** 2 + ((cc - ex[1]) * np.cos(
            np.radians(ex[0]))) ** 2))
    order = sorted(set(gcell) | ({ecell} if ecell is not None else set()))
    imap = [[order.index(c), grp] for c, grp in zip(gcell, groups)]
    exit_node = order.index(ecell) if ecell is not None else None
    entry_nodes = sorted(set(order.index(c) for c in gcell))
    # two way: routing sites = the entry gauge cells, then the exit cell
    sites = list(dict.fromkeys(gcell)) + ([ecell] if a.two_way else [])
    registry.register('split_box', lambda **k: Split(
        box_idx, area[box_idx], **k),
                      'test', replace=True)
    registry.register('vic_replay', lambda **k: Replay(
        a.fluxes, day0, mask, area, **k), 'test', replace=True)
    cfg = dict(
        run=dict(device=a.device, dtype='float64', log=False,
                 duration_days=ndays),
        modules=dict(
            vic=dict(connector='vic_replay'),
            split=dict(connector='split_box'),
            routing=dict(connector='routing', fdir=str(FDIR) + ':fdir',
                         domain=str(DOMAIN), gauges=str(gcsv),
                         step_s=86400.0),
            flood=dict(connector='flood', basin=str(BASIN),
                       interval_s=86400.0, free_edges=[['W', 1e-4]],
                       inflow_points=[list(p) for p in points],
                       inflow_map=imap)),
        connections=[
            {'from': 'vic.total_runoff', 'to': 'split.runoff'},
            {'from': 'split.routed', 'to': 'routing.inflow'},
            {'from': 'split.box', 'to': 'flood.runoff',
             'mapping': 'vic_to_flood'},
            {'from': 'routing.discharge', 'to': 'flood.upstream_discharge'}],
        mappings=dict(vic_to_flood=dict(source='box_cells',
                                        target='flood_cells', pairs=pairs)),
        ledger=dict(enabled=True, on_violation='reject'))
    if a.two_way:
        fm = cfg['modules']['flood']
        fm.pop('inflow_map')
        fm.update(hand_over=True, return_outlets=[['W']],
                  dependencies=['routing'],
                  open_water_evap_mm_day=a.evap_mm_day)
        cfg['modules']['routing']['site_cells'] = [int(c) for c in sites]
        si = {c: k for k, c in enumerate(sites)}
        # river_sites -> points: an entry site feeds its group's points
        # equally; the exit site hands nothing over (maps to point 0)
        s2p, p2s = [], {}
        for c, grp in zip(gcell, groups):
            if any(x[0] == si[c] for x in s2p):
                continue
            s2p += [[si[c], i, 1.0 / len(grp)] for i in grp]
            for i in grp:
                p2s.setdefault(i, si[c])
        s2p.append([si[ecell], 0, 1.0])
        cfg['connections'] = cfg['connections'][:3] + [
            {'from': 'flood.river_request',
             'to': 'routing.withdrawal_request', 'lag': 1,
             'mapping': 'pt_site'},
            {'from': 'routing.withdrawn', 'to': 'flood.inflow',
             'mapping': 'site_pt'},
            {'from': 'flood.returned', 'to': 'routing.injection',
             'lag': 1, 'mapping': 'out_site'}]
        cfg['mappings'].update(
            pt_site=dict(source='flood_inflow', target='river_sites',
                         pairs=[[i, p2s[i], 1.0]
                                for i in range(len(points))]),
            site_pt=dict(source='river_sites', target='flood_inflow',
                         pairs=s2p),
            out_site=dict(source='flood_outlets', target='river_sites',
                          pairs=[[0, si[ecell], 1.0]]))
    t0 = time.time()
    m = TorchVIC.from_dict(cfg).initialize()
    t_init = time.time() - t0
    rt = m.runtime
    rcn, fl = m.connector('routing'), m.connector('flood')
    gcells = gcell
    timing = {k: 0.0 for k in ('vic', 'routing', 'flood')}
    calls = {k: 0 for k in timing}
    dev = torch.device(a.device)

    def wrap(name):
        c = m.connector(name)
        orig = c.compute

        def timed(inputs, context, dt, _o=orig):
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t_ = time.time()
            r_ = _o(inputs, context, dt)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            timing[name] += time.time() - t_
            calls[name] += 1
            return r_
        c.compute = timed
    for n in timing:
        wrap(n)
    days, rows = 0, []
    t1 = time.time()
    while True:
        tf = timing['flood']
        if not m.step():
            break
        if calls['flood'] > days:
            days = calls['flood']
            i = fl.last
            q_in = i['I_inflow'] / 86400.0
            qd = rt.state.get('routing', 'discharge')
            row = dict(day=days, flood_s=round(timing['flood'] - tf, 3),
                       steps=i.get('steps'), inflow_m3s=q_in,
                       runoff_m3=i['I_rain'], stored=i['stored1'],
                       outflow=fl.totals['outflow'],
                       flooded_km2=float((fl.depth > 0.01).sum()) * 1e-2,
                       max_depth=float(fl.depth.max()),
                       q_exit=(float(qd[exit_node]) if exit_node is not None
                               else None),
                       q_entry=[float(qd[k]) for k in entry_nodes],
                       returned=i.get('returned'), lost=i.get('lost'),
                       evaporation=i.get('evaporation'))
            rows.append(row)
            print(json.dumps(row), flush=True)
    wall = time.time() - t1
    led = rt.ledger
    bal = led.balances
    by = {}
    for r in bal:
        d = by.setdefault(r['module'], dict(rows=0, ok=0, worst=0.0))
        d['rows'] += 1
        d['ok'] += bool(r['ok'])
        tol = r.get('tolerance') or 0.0
        if tol > 0:
            d['worst'] = max(d['worst'], abs(r['residual']) / tol)
    cp = led.coupled
    rec = fl.volume_record()
    tot = led.totals()
    comb = (sum(r['P'] + r['I_boundary'] - r['ET'] - r['Q_boundary']
                - r['dS'] for r in cp) - tot['E_other'] - tot['closure'])
    P_all = sum(r['P'] for r in cp)
    two = {}
    if a.two_way:
        rr = rcn.router
        wd = float(rr.withdrawn.sum())
        passed = sum(r['q_entry'][0] for r in rows) * 86400.0
        two = dict(withdrawn_m3=wd, injected_m3=float(rr.injected.sum()),
                   returned_m3=fl.totals['returned'],
                   lost_m3=fl.totals['lost'],
                   evaporation_m3=fl.totals['evaporation'],
                   entry_left_in_river_m3=passed)
    res = dict(start=a.start, end=a.end, days=days,
               device=torch.cuda.get_device_name() if dev.type == 'cuda'
               else 'cpu', init_s=round(t_init, 1), run_s=round(wall, 1),
               ms_per_day={k: round(1e3 * v / max(days, 1), 1)
                           for k, v in timing.items()},
               module_s={k: round(v, 1) for k, v in timing.items()},
               wall_ms_per_day=round(1e3 * wall / max(days, 1), 1),
               flood_ms_per_day=round(1e3 * timing['flood'] / max(days, 1), 1),
               gauges=gs, gauge_cells=gcells, inflow_points=len(points),
               inflow_map=[[k, len(g)] for k, g in imap],
               condition=fl.condition_info, ledger_by_module=by,
               coupled_rows=len(cp), coupled_ok=all(r['ok'] for r in cp),
               coupled_worst=max((abs(r['residual']) / r['tolerance']
                                  for r in cp if r.get('tolerance')),
                                 default=0.0),
               flood_volume=rec,
               flood_handed_runoff_m3=sum(r['runoff_m3'] for r in rows),
               flood_inflow_m3=sum(r['inflow_m3s'] for r in rows) * 86400.0,
               peak_inflow_m3s=max(r['inflow_m3s'] for r in rows),
               max_depth_m=float(fl.depth.max()),
               flooded_km2=float((fl.depth > 0.01).sum()) * 1e-2,
               mode='two-way' if a.two_way else 'one-way',
               combined_residual_m3=comb, combined_P_m3=P_all,
               flooded_km2_mean=float(np.mean([r['flooded_km2']
                                               for r in rows])),
               flooded_km2_max=float(max(r['flooded_km2'] for r in rows)),
               q_exit_mean=(float(np.mean([r['q_exit'] for r in rows]))
                            if exit_node is not None else None),
               q_exit_max=(float(max(r['q_exit'] for r in rows))
                           if exit_node is not None else None),
               exit_cell=ex, **two)
    m.finalize(write=False)
    (out / 'result.json').write_text(json.dumps(res, indent=1, default=str))
    (out / 'days.json').write_text(json.dumps(rows))
    print('RESULT ' + json.dumps(res, default=str), flush=True)


if __name__ == '__main__':
    main()
