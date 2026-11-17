#!/usr/bin/env python3
"""Combination matrix of the coupled TorchVIC system on the ISC domain.

    run_combos.py prepare              inputs (dams, irrigation, flood wiring)
    run_combos.py prepare-2way         two-way flood wiring (after prepare)
    run_combos.py configs [--repo]     write the combination configs
    run_combos.py global OUT_DIR [--end YYYY-MM-DD]   ISC global file
    run_combos.py run N --out DIR [--end ..] [--stop-day D] [--restart-from F]
                          [--ckpt-end F] [--history] [--syncs]
    run_combos.py table                TEST_RESULTS matrix rows from results

Combinations (N): 1 vic_only, 2 vic_routing, 3 vic_irrigation, 4 vic +
reservoir (on the router; identical to 7), 5 vic_groundwater, 6
vic_groundwater_irrigation, 7 vic_reservoir_routing, 8 irrigation +
reservoir + routing, 9 + crop, 10 + quality, 11 vic_flood, 12
full_coupled_system, 16 full_coupled_system_2way (12 with the flood
returned to routing and floodplain infiltration into VIC's soil next to
irrigation). COMBO_RUNS (env) moves the per-combination global files out of
$TORCHVIC_DATA_ROOT/verification/combos/runs. Where no data exist the inputs `prepare` writes
are synthetic (a uniform aquifer, one crop parameter set, constant quality loads and
meteorology, a constant floodplain infiltration); the configuration headers mark them SYNTHETIC.
"""
import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

AD = Path(os.environ.get('TORCHVIC_DATA_ROOT', 'data'))
COMBO = AD / 'verification/combos'
INP = COMBO / 'inputs'
REPO = Path(__file__).resolve().parents[1]
GLOBAL_T = REPO / 'benchmarks/isc/isc_vic5_3h.txt'
GLOBAL_EB = REPO / 'benchmarks/isc/isc_vic5_3h_eb.txt'
#: combinations that need VIC's full energy balance (OUT_PET for the crop)
EB = {9, 10, 12, 13, 15, 16}
FDIR = AD / 'static/isc_fdir.nc'
DOMAIN = AD / 'vic_params_isc/domain.nc'
GAUGES = AD / 'discharge/final_basins/basin_index.csv'
BASIN = AD / 'static/flood_basins/brahmaputra_guwahati_100m.npz'
PAIRS = AD / 'static/flood_basins/brahmaputra_guwahati_100m_vic_pairs.csv'
GDW = AD / 'water_use/dams/GDW_barriers_v1_delta.dbf'
GMIA = AD / 'water_use/gmia_v5'
ESRI = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1),
        32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}

#: n -> (key, config file, in the repo's configs/, VIC exchange)
COMBOS = {
    1: ('vic_only', 'vic_only', True, 'w31'),
    2: ('vic_routing', 'vic_routing', True, 'w31'),
    3: ('vic_irrigation', 'vic_irrigation', True, 'd1'),
    4: ('vic_reservoir', 'vic_reservoir_routing', True, 'w31'),
    5: ('vic_groundwater', 'vic_groundwater_isc', False, 'd1'),
    6: ('vic_groundwater_irrigation', 'vic_groundwater_irrigation', True,
        'd1'),
    7: ('vic_reservoir_routing', 'vic_reservoir_routing', True, 'w31'),
    8: ('vic_irrigation_reservoir_routing', 'vic_irrigation_reservoir_routing',
        False, 'd1'),
    9: ('plus_crop', 'vic_irr_res_routing_crop', False, 'd1'),
    10: ('plus_quality', 'vic_irr_res_routing_crop_quality', False, 'd1'),
    11: ('vic_flood', 'vic_flood', True, 'd1'),
    12: ('full_coupled_system', 'full_coupled_system', True, 'd1'),
    13: ('vic_only_eb', 'vic_only_eb', False, 'w31'),     # EB timing baseline
    14: ('vic_only_daily', 'vic_only_daily', False, 'd1'),  # daily-window baseline
    15: ('vic_only_eb_daily', 'vic_only_eb_daily', False, 'd1'),
    16: ('full_coupled_system_2way', 'full_coupled_system_2way', True, 'd1'),
}
#: floodplain infiltration of the two-way full system (mm/day, interval-end
#: loss of hydraulic cells deeper than flood_threshold; SYNTHETIC constant)
INFIL_MM_DAY = 10.0
ENV = dict(
    TORCHVIC_GLOBAL='(per run, from `global`)',
    TORCHVIC_FDIR=str(FDIR) + ':fdir', TORCHVIC_DOMAIN=str(DOMAIN),
    TORCHVIC_GAUGES=str(GAUGES), TORCHVIC_DAMS=str(INP / 'dams.csv'),
    TORCHVIC_COMBO_DATA=str(INP), TORCHVIC_ALL_DATA=str(AD))


def say(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- prepare
def snap_gauges(rows, cl, cc):
    gl = []
    for la, lo in rows:
        gl.append(int(np.argmin((cl - la) ** 2
                                + ((cc - lo) * np.cos(np.radians(la))) ** 2)))
    return gl


def read_asc(p):
    with open(p) as f:
        h = {}
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
        a = np.loadtxt(f, dtype=np.float32)
    return h, a


def prepare():
    import netCDF4
    from torchvic.model import river_network as rn
    from torchvic.reservoir import read_gdw_dams, place_dams, merge_by_cell
    INP.mkdir(parents=True, exist_ok=True)
    d = netCDF4.Dataset(str(DOMAIN))
    lat, lon = np.asarray(d['lat'][:]), np.asarray(d['lon'][:])
    mask = np.asarray(d['mask'][:]) > 0
    f = netCDF4.Dataset(str(FDIR))
    net, cells = rn.d8_downstream(np.asarray(f['fdir'][:]), mask, lat, lon)
    idx = -np.ones(mask.shape, np.int64)
    idx[mask] = np.arange(mask.sum())
    # --- dams (GDW v1, India >= 10 MCM, placed by upstream area)
    dm = read_gdw_dams(str(GDW), min_capacity_mcm=10)
    iy, ix, ok = place_dams(dm, np.asarray(f['lat'][:]),
                            np.asarray(f['lon'][:]),
                            np.asarray(f['upstream_area'][:]))
    y, x, cap, yr, irr, area, n = merge_by_cell(dm, iy, ix, ok,
                                                year_default=1900)
    keep = idx[y, x] >= 0
    with open(INP / 'dams.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['cell', 'capacity_m3', 'build_year', 'irrigation',
                    'init_frac', 'area_max'])
        for k in np.nonzero(keep)[0]:
            w.writerow([int(idx[y[k], x[k]]), '%.6g' % cap[k],
                        int(yr[k]), str(bool(irr[k])).lower(), 0.5,
                        '%.6g' % area[k]])
    say('dams: %d cells, %.1f km3' % (keep.sum(), cap[keep].sum() / 1e9))
    # --- irrigation (GMIA v5) on the ISC grid
    ha, a_aei = read_asc(GMIA / 'gmia_v5_aei_pct.asc')
    _, a_aai = read_asc(GMIA / 'gmia_v5_aai_pct_aei.asc')
    _, a_gw = read_asc(GMIA / 'gmia_v5_aeigw_pct_aei.asc')
    cs = ha['cellsize']
    ny = a_aei.shape[0]
    ri = np.clip(((90.0 - lat) / cs).astype(int), 0, ny - 1)
    cj = np.clip(((lon + 180.0) / cs).astype(int), 0, a_aei.shape[1] - 1)

    def samp(a, fill):
        v = a[np.ix_(ri, cj)].astype(np.float64)
        return np.where(v < 0, fill, v)
    aei = samp(a_aei, 0.0) / 100.0
    aai = samp(a_aai, 100.0) / 100.0
    gw = samp(a_gw, 0.0) / 100.0
    with netCDF4.Dataset(str(INP / 'isc_irrigation.nc'), 'w') as o:
        o.createDimension('lat', len(lat))
        o.createDimension('lon', len(lon))
        for nm, v in (('lat', lat), ('lon', lon)):
            o.createVariable(nm, 'f8', (nm,))[:] = v
        o.createVariable('irr_frac', 'f8', ('lat', 'lon'))[:] = \
            np.clip(aei * aai, 0, 1)
        o.createVariable('gw_frac', 'f8', ('lat', 'lon'))[:] = \
            np.clip(gw, 0, 1)
        o.source = ('GMIA v5 (FAO) aei_pct x aai_pct_aei, aeigw_pct_aei, '
                    'nearest 5 arcmin cell')
    m = mask
    say('irrigation: irrigated cells %d, mean frac %.4f' % (
        ((aei * aai)[m] > 0).sum(), (aei * aai)[m].mean()))
    # --- flood wiring
    sys.path.insert(0, str(REPO / 'benchmarks'))
    import bench_flood_coupled as bfc
    bfc.FDIR = FDIR
    b = np.load(BASIN)
    dem = b['dem']
    nyb, nxb = dem.shape
    dx = float(b['dx'])
    box = (float(b['lon0']), float(b['lon1']), float(b['lat0']),
           float(b['lat1']))
    box_idx = []
    for lo, la in zip(b['vic_lon'], b['vic_lat']):
        i = int(np.argmin(abs(lat - la)))
        j = int(np.argmin(abs(lon - lo)))
        box_idx.append(int(idx[i, j]))
    assert min(box_idx) >= 0
    with open(INP / 'box_cells.csv', 'w') as fh:
        fh.write('cell\n' + ''.join('%d\n' % c for c in box_idx))
    gs = [g for g in bfc.entry_gauges(box) if g[0] >= 10000.0]
    rows = [(float(r['lat']), float(r['lng']))
            for r in csv.DictReader(open(GAUGES))]
    union = rows + [(g[1], g[2]) for g in gs]
    with open(INP / 'gauges_flood.csv', 'w') as fh:
        fh.write('lat,lng\n' + ''.join('%.5f,%.5f\n' % p for p in union))
    cl, cc = lat[cells[:, 0]], lon[cells[:, 1]]
    order = sorted(set(snap_gauges(union, cl, cc)))
    gcell = snap_gauges([(g[1], g[2]) for g in gs], cl, cc)
    r_in, c_in = [int(v) for v in b['inflow_rc'][0]]
    col = dem[:, c_in]
    win = col[max(r_in - 80, 0):r_in + 80]
    lo = float(np.percentile(win, 15))
    east = [(k, c_in) for k in range(max(r_in - 80, 0), min(r_in + 80, nyb))
            if abs(col[k] - lo) <= 0.7]
    points, groups = list(east), []
    ml, mt = float(b['m_lon']), float(b['m_lat'])

    def rc(lon_, lat_):
        return (nyb - (lat_ - box[2]) * mt / dx - 0.5,
                (lon_ - box[0]) * ml / dx - 0.5)
    for g in gs:
        ua, la, lo_, ela, elo = g
        if lo_ > box[1]:
            groups.append(list(range(len(east))))
            continue
        r0, c0 = rc(elo - 0.03125, ela + 0.03125)
        r1, c1 = rc(elo + 0.03125, ela - 0.03125)
        r0, r1 = int(max(r0, 0)), int(min(r1, nyb))
        c0, c1 = int(max(c0, 0)), int(min(c1, nxb))
        if la > box[3]:
            line = [(3, c) for c in range(c0, c1)]
        elif la < box[2]:
            line = [(nyb - 4, c) for c in range(c0, c1)]
        else:
            line = [(r, 3) for r in range(r0, r1)]
        zl = np.array([dem[p] for p in line])
        pick = [line[k] for k in np.nonzero(zl <= zl.min() + 2.0)[0]]
        gi = []
        for p in pick:
            if p not in points:
                points.append(p)
            gi.append(points.index(p))
        groups.append(gi)
    imap = [[order.index(c), grp] for c, grp in zip(gcell, groups)]
    json.dump(dict(inflow_points=[list(map(int, p)) for p in points],
                   inflow_map=imap, entry_gauges=gs,
                   n_gauges=len(order)),
              open(INP / 'flood_wiring.json', 'w'))
    say('flood: %d entry gauges, %d inflow points, %d routing gauge nodes'
        % (len(gs), len(points), len(order)))


def prepare_2way():
    """Two-way flood wiring (bench_flood_coupled --two-way on the combo
    inputs): routing sites = the entry gauge cells, then the exit cell where
    the main stem leaves the box; the hand-over and return mappings; the
    box VIC cells of the flood's vic_map (infiltration -> vic.soil_add)."""
    import netCDF4
    sys.path.insert(0, str(REPO / 'benchmarks'))
    import bench_flood_coupled as bfc
    bfc.FDIR = FDIR
    w = json.load(open(INP / 'flood_wiring.json'))
    gs = [tuple(g) for g in w['entry_gauges']]
    groups = [grp for _, grp in w['inflow_map']]
    npts = len(w['inflow_points'])
    d = netCDF4.Dataset(str(DOMAIN))
    lat, lon = np.asarray(d['lat'][:]), np.asarray(d['lon'][:])
    mask = np.asarray(d['mask'][:]) > 0
    cells = np.argwhere(mask)
    cl, cc = lat[cells[:, 0]], lon[cells[:, 1]]
    b = np.load(BASIN)
    box = (float(b['lon0']), float(b['lon1']), float(b['lat0']),
           float(b['lat1']))
    ex = bfc.exit_cell(box, (gs[0][1], gs[0][2]))
    gcell = snap_gauges([(g[1], g[2]) for g in gs], cl, cc)
    ecell = snap_gauges([ex], cl, cc)[0]
    sites = list(dict.fromkeys(gcell)) + [ecell]
    si = {c: k for k, c in enumerate(sites)}
    s2p, p2s = [], {}
    for c, grp in zip(gcell, groups):
        if any(x[0] == si[c] for x in s2p):
            continue
        s2p += [[si[c], int(i), 1.0 / len(grp)] for i in grp]
        for i in grp:
            p2s.setdefault(int(i), si[c])
    s2p.append([si[ecell], 0, 1.0])
    box_idx = [int(x) for x in np.loadtxt(INP / 'box_cells.csv', skiprows=1,
                                          ndmin=1)]
    pr = np.loadtxt(PAIRS, delimiter=',', skiprows=1, ndmin=2)
    n_vic = int(pr[:, 0].max()) + 1
    dams = [int(r['cell']) for r in csv.DictReader(open(INP / 'dams.csv'))]
    out = dict(site_cells=[int(c) for c in sites], exit_latlon=list(ex),
               pt_site=[[i, p2s[i], 1.0] for i in range(npts)],
               site_pt=s2p, out_site=[[0, si[ecell], 1.0]],
               flood_vic=[[k, box_idx[k], 1.0] for k in range(n_vic)],
               sites_with_dam=sorted(set(sites) & set(dams)))
    json.dump(out, open(INP / 'flood_wiring_2way.json', 'w'))
    say('flood 2-way: sites %s (exit %s), %d VIC cells in vic_map, sites '
        'on a dam cell: %s' % (sites, ex, n_vic, out['sites_with_dam']))


# ---------------------------------------------------------------- configs
def vic_block(mode, extra=None, eb=False):
    v = dict(connector='vic', global_param='${TORCHVIC_GLOBAL}',
             core='5.1.0')
    if not eb:
        v['snow_bands'] = 'mean'
    if mode == 'w31':
        v['window_days'] = 31
    else:
        v.update(window_days=1, exchange='day')
    v.update(extra or {})
    return v


def blocks(eb=False):
    vicref = dict(global_param='${TORCHVIC_GLOBAL}', core='5.1.0')
    if not eb:
        vicref['snow_bands'] = 'mean'
    cd = '${TORCHVIC_COMBO_DATA}'
    B = {}
    B['routing'] = dict(connector='routing', fdir='${TORCHVIC_FDIR}',
                        domain='${TORCHVIC_DOMAIN}',
                        gauges='${TORCHVIC_GAUGES}', step_s=86400)
    B['reservoir'] = dict(connector='reservoir', dams='${TORCHVIC_DAMS}',
                          routing='routing', start_date='2004-01-01',
                          relax_steps=10, evap_m=0.004)
    B['irrigation'] = dict(
        connector='irrigation', vic=vicref,
        irrigated_fraction={'file': cd + '/isc_irrigation.nc',
                            'var': 'irr_frac'},
        gw_fraction={'file': cd + '/isc_irrigation.nc', 'var': 'gw_frac'},
        efficiency=0.7, conveyance_loss=0.1, allocation='priority',
        loss_return='river')
    B['groundwater'] = dict(
        connector='groundwater', global_param='${TORCHVIC_GLOBAL}',
        interval_days=1,
        aquifer=dict(k=5.0, sy=0.10, ss=1.0e-5, thickness=100.0,
                     top={'file': '${TORCHVIC_DOMAIN}', 'var': 'elev'},
                     depth_to_water=10.0),
        drain=dict(depth=3.0), pumping=True, afr=0.1)
    B['crop'] = dict(connector='crop', vic=vicref,
                     params_file='${TORCHVIC_CROP_PARAMS}', start_day=60)
    B['quality'] = dict(
        connector='quality', routing='routing',
        constituents=dict(nitrate=2.0e-7, tracer=0.0),
        temperature=dict(depth=1.0, k_ref=30.0),
        loads=dict(nitrate=3.0, tracer=1.5),
        met=dict(t_air=12.0, vp=0.9, wind=2.0, sw=180.0), t_runoff=4.0)
    return B


C_IRR = [{'from': 'vic.soil_moisture', 'to': 'irrigation.soil_moisture'},
         {'from': 'vic.soil_add_unmet', 'to': 'irrigation.soil_unmet'},
         {'from': 'irrigation.soil_add', 'to': 'vic.soil_add', 'lag': 1}]
C_GW = [{'from': 'vic.baseflow', 'to': 'groundwater.recharge'}]
C_PUMP = [{'from': 'irrigation.gw_request',
           'to': 'groundwater.pumping_request'},
          {'from': 'groundwater.pumping', 'to': 'irrigation.gw_pumped',
           'lag': 1}]
C_CROP = [{'from': 'vic.transp', 'to': 'crop.transp'},
          {'from': 'vic.pet', 'to': 'crop.pet'},
          {'from': 'crop.crop_active', 'to': 'irrigation.season'},
          {'from': 'crop.root_depth', 'to': 'irrigation.root_depth'}]


def config_for(n):
    key, name, _, mode = COMBOS[n]
    eb = n in EB
    B = blocks(eb)
    mods = {'vic': vic_block(mode, dict(soil_add_mode='cap'), eb)}
    conns = []
    full = n in (12, 16)
    gw = n in (5, 6) or full
    irr = n in (3, 6, 8, 9, 10) or full
    res = n in (4, 7, 8, 9, 10) or full
    routing = n in (2, 4, 7, 8, 9, 10, 11) or full
    crop = n in (9, 10) or full
    qual = n == 10 or full
    flood = n == 11 or full
    two_way = n == 16
    if n in (1, 13, 14, 15):
        mods['vic'].pop('soil_add_mode')
    if irr:
        mods['irrigation'] = B['irrigation']
        conns += C_IRR
        if gw:
            mods['irrigation'] = dict(B['irrigation'], groundwater=True)
    if crop:
        mods['crop'] = B['crop']
        conns += C_CROP
    if gw:
        mods['groundwater'] = B['groundwater']
        conns += C_GW + (C_PUMP if irr else [])
    if routing:
        r = dict(B['routing'])
        if res:
            r['reservoirs'] = 'reservoir'
            mods['reservoir'] = B['reservoir']
        if qual:
            r['quality'] = 'quality'
            mods['quality'] = B['quality']
        if flood:
            r['gauges'] = '${TORCHVIC_COMBO_DATA}/gauges_flood.csv'
        mods['routing'] = r
        src = 'vic.runoff' if gw else (
            'vic.total_runoff' if flood else 'vic.total_runoff_daily')
        if flood:
            mods['split'] = dict(connector='split',
                                 cells='${TORCHVIC_COMBO_DATA}/box_cells.csv',
                                 domain='${TORCHVIC_DOMAIN}')
            conns.append({'from': src, 'to': 'split.runoff'})
            conns.append({'from': 'split.routed', 'to': 'routing.inflow'})
        else:
            conns.append({'from': src, 'to': 'routing.inflow'})
        if gw:
            conns.append({'from': 'groundwater.discharge',
                          'to': 'routing.inflow'})
    cfg = dict(version=1, run=dict(device='auto', dtype='float64'),
               modules=mods, connections=conns)
    if two_way:
        # flood returns to routing (hand-over at the entry sites, return at
        # the exit site) and floodplain infiltration enters VIC's soil next
        # to irrigation: two soil_add producers, each with its own share of
        # VIC's unmet (VicConnector soil_add_sources)
        w2 = json.load(open(INP / 'flood_wiring_2way.json'))
        mods['vic']['soil_add_sources'] = ['irrigation', 'flood']
        for c in conns:
            if c['to'] == 'vic.soil_add':
                c['to'] = 'vic.soil_add_irrigation'
            if c['from'] == 'vic.soil_add_unmet':
                c['from'] = 'vic.soil_add_unmet_irrigation'
        mods['routing']['site_cells'] = w2['site_cells']
    if flood:
        w = json.load(open(INP / 'flood_wiring.json'))
        mods['flood'] = dict(
            connector='flood', basin='${TORCHVIC_ALL_DATA}/static/'
            'flood_basins/brahmaputra_guwahati_100m.npz',
            interval_s=86400, free_edges=[['W', 0.0001]],
            inflow_points=w['inflow_points'], inflow_map=w['inflow_map'])
        conns.append({'from': 'split.box', 'to': 'flood.runoff',
                      'mapping': 'vic_to_flood'})
        cfg['mappings'] = dict(vic_to_flood=dict(
            source='box_cells', target='flood_cells',
            file='${TORCHVIC_ALL_DATA}/static/flood_basins/'
            'brahmaputra_guwahati_100m_vic_pairs.csv'))
        if not two_way:
            conns.append({'from': 'routing.discharge',
                          'to': 'flood.upstream_discharge'})
    if two_way:
        fm = mods['flood']
        fm.pop('inflow_map')
        fm.update(hand_over=True, return_outlets=[['W']],
                  dependencies=['routing'], infiltration_mm_day=INFIL_MM_DAY,
                  vic_map='${TORCHVIC_ALL_DATA}/static/flood_basins/'
                  'brahmaputra_guwahati_100m_vic_pairs.csv')
        conns += [
            {'from': 'flood.river_request',
             'to': 'routing.withdrawal_request', 'lag': 1,
             'mapping': 'pt_site'},
            {'from': 'routing.withdrawn', 'to': 'flood.inflow',
             'mapping': 'site_pt'},
            {'from': 'flood.returned', 'to': 'routing.injection', 'lag': 1,
             'mapping': 'out_site'},
            {'from': 'flood.infiltration', 'to': 'vic.soil_add_flood',
             'lag': 1, 'mapping': 'fv2v'},
            {'from': 'vic.soil_add_unmet_flood',
             'to': 'flood.infiltration_unmet', 'mapping': 'v2fv'}]
        cfg['mappings'].update(
            pt_site=dict(source='flood_inflow', target='river_sites',
                         pairs=w2['pt_site']),
            # the other river sites (reservoir and quality sites routing
            # adds after the configured ones) never carry a flood request:
            # their (zero) withdrawn goes to point 0
            site_pt=dict(source='river_sites', target='flood_inflow',
                         pairs=w2['site_pt'], rest_to=0),
            out_site=dict(source='flood_outlets', target='river_sites',
                          pairs=w2['out_site']),
            fv2v=dict(source='flood_vic', target='vic_cells',
                      pairs=w2['flood_vic']),
            # VIC cells outside the box get no floodplain water, so their
            # (zero) soil_add_unmet_flood goes to flood_vic element 0
            v2fv=dict(source='vic_cells', target='flood_vic',
                      pairs=[[t, s_, f] for s_, t, f in w2['flood_vic']],
                      rest_to=0))
    cfg['ledger'] = dict(enabled=True, on_violation='reject')
    cfg['output'] = dict(dir='results_' + name, history=True, ledger=True)
    return cfg


HEADER = """# {title}
# ISC domain (112,825 cells, 1/16 degree, India), one year (2004), VIC 5.1.0.
#   torchvic validate --config configs/{name}.yaml
#   torchvic run --config configs/{name}.yaml --out results_{name}
# Environment (data are not in the package):
#   TORCHVIC_GLOBAL      ISC global file (benchmarks/isc/isc_vic5_3h.txt with
#                        RESULT_DIR / dates filled; `run_combos.py global`)
#   TORCHVIC_FDIR        flow map, FILE.nc:fdir (<data root>/static/isc_fdir.nc)
#   TORCHVIC_DOMAIN      domain.nc (lat, lon, mask, area, frac, elev)
#   TORCHVIC_GAUGES      gauges csv (lat, lng)
#   TORCHVIC_DAMS        dam table (`run_combos.py prepare`)
#   TORCHVIC_COMBO_DATA  inputs from `run_combos.py prepare`
#   TORCHVIC_ALL_DATA    <data root> (flood basin)
#   TORCHVIC_CROP_PARAMS configs/crop_params_sbarley.json
{notes}"""

TITLES = {
    1: 'VIC alone through the runtime (equals the direct run).',
    2: 'VIC -> river routing (daily routing steps, gauge discharge).',
    3: 'VIC + irrigation (daily; sources are stub stores, see notes).',
    4: 'VIC + reservoirs on the router.',
    5: 'VIC + groundwater (recharge = VIC baseflow).',
    6: 'VIC + irrigation + groundwater (pumping).',
    7: 'VIC + reservoirs + routing.',
    8: 'VIC + irrigation + reservoirs + routing.',
    9: 'VIC + irrigation + reservoirs + routing + crop.',
    10: 'VIC + irrigation + reservoirs + routing + crop + quality.',
    11: 'VIC + routing + flood (Guwahati box, one way).',
    13: 'VIC alone, full energy balance (timing baseline of 9, 10, 12).',
    14: 'VIC alone with daily windows (timing baseline of 3, 5, 6, 8, 11).',
    15: 'VIC alone, energy balance, daily windows (baseline of 9, 10, 12).',
    12: 'Full coupled system: VIC, routing, reservoirs, quality, '
        'irrigation, crop, groundwater, flood.',
    16: 'Full coupled system with a two-way flood: the box river is handed '
        'over by routing and returned to it; floodplain infiltration enters '
        'VIC\'s soil next to irrigation.',
}
EBNOTE = ('# TORCHVIC_GLOBAL here is benchmarks/isc/isc_vic5_3h_eb.txt (FULL_ENERGY:\n'
          '# the crop needs VIC\'s potential evaporation, OUT_PET, an energy-balance\n'
          '# output).\n')
NOTES = {
    3: '# Irrigation withdraws from stub river/reservoir/groundwater stores\n'
       '# (unlimited); irrigated fraction and groundwater share from GMIA v5.\n',
    6: '# Pumping loop: irrigation.gw_request -> groundwater.pumping_request;\n'
       '# groundwater.pumping -> irrigation.gw_pumped (lag 1). Aquifer\n'
       '# parameters are SYNTHETIC (uniform k, sy, thickness).\n',
    9: EBNOTE, 10: EBNOTE,
    12: EBNOTE + '# Irrigation river/reservoir sources are stub stores; groundwater is\n'
        '# wired (pumping). Crop is one spring barley set for all cells\n'
        '# (synthetic). Quality loads/meteorology are synthetic constants.\n',
}
NOTES[16] = NOTES[12] + (
    '# Two-way flood (docs/connectors/flood.md): routing hands the\n'
    '# river over at the entry sites (flood.river_request ->\n'
    '# routing.withdrawal_request, routing.withdrawn -> flood.inflow) and gets\n'
    '# the west-edge outflow back at the exit site (flood.returned ->\n'
    '# routing.injection, lag 1). Floodplain infiltration (%g mm/day,\n'
    '# SYNTHETIC) -> vic.soil_add_flood next to irrigation -> vic.soil_add_irrigation:\n'
    '# VIC adds their sum once (cap) and hands the unmet back split in\n'
    '# proportion to what each added (soil_add_unmet_irrigation,\n'
    '# soil_add_unmet_flood), so nothing is handed back twice.\n' % INFIL_MM_DAY)


def write_configs(repo_only=False):
    import yaml
    COMBO.joinpath('configs').mkdir(parents=True, exist_ok=True)
    done = set()
    for n, (key, name, in_repo, mode) in COMBOS.items():
        if name in done:
            continue
        done.add(name)
        cfg = config_for(n)
        txt = HEADER.format(title=TITLES[n], name=name,
                            notes=NOTES.get(n, ''))
        txt += yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None,
                              width=100)
        outs = []
        if in_repo:
            outs.append(REPO / 'configs' / (name + '.yaml'))
        if not repo_only:
            outs.append(COMBO / 'configs' / (name + '.yaml'))
        for p in outs:
            p.write_text(txt)
        say('wrote', name)


# ---------------------------------------------------------------- global
def write_global(outdir, start='2004-01-01', end='2004-12-31', eb=False):
    outdir = Path(outdir)
    (outdir / 'vic_out').mkdir(parents=True, exist_ok=True)
    (outdir / 'vic_log').mkdir(parents=True, exist_ok=True)
    y0, m0, d0 = start.split('-')
    y1, m1, d1 = end.split('-')
    rep = dict(STARTYEAR=y0, STARTMONTH=str(int(m0)), STARTDAY=str(int(d0)),
               ENDYEAR=y1, ENDMONTH=str(int(m1)), ENDDAY=str(int(d1)),
               RESULT_DIR=str(outdir / 'vic_out') + '/',
               LOG_DIR=str(outdir / 'vic_log') + '/')
    lines = []
    for ln in (GLOBAL_EB if eb else GLOBAL_T).read_text().splitlines():
        k = ln.split()[0] if ln.split() else ''
        ln = ln.replace('@DATA_ROOT@', str(AD))
        lines.append('%-12s %s' % (k, rep[k]) if k in rep else ln)
    p = outdir / 'global.txt'
    p.write_text('\n'.join(lines) + '\n')
    return p


# ---------------------------------------------------------------- run
def set_env(gfile):
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    os.environ['TORCHVIC_GLOBAL'] = str(gfile)
    os.environ.setdefault('TORCHVIC_CROP_PARAMS',
                          str(REPO / 'configs/crop_params_sbarley.json'))


def walk_min(x, pre, out):
    import torch
    if isinstance(x, dict):
        for k, v in x.items():
            walk_min(v, pre + str(k) + '.', out)
    elif torch.is_tensor(x) and x.is_floating_point() and x.numel():
        out[pre.rstrip('.')] = x


def store_minima(m):
    """Smallest value of each water store (negative = a violation)."""
    import torch
    out = {}
    c = m.runtime.connectors
    try:
        st = {}
        walk_min(c['vic'].state, '', st)
        for k, v in st.items():
            if k.split('.')[0] in ('moist', 'swe', 'wdew', 'snow_canopy',
                                   'pack_water', 'canopy_water'):
                out['vic.' + k] = float(torch.nan_to_num(v, nan=0.0).min())
    except Exception as e:                                # noqa: BLE001
        out['vic'] = 'n/a: %s' % e
    if 'irrigation' in c:
        e = c['irrigation'].engine
        out['irrigation.leftover'] = float(e.leftover_m3.min())
        out['irrigation.payable'] = float(c['irrigation'].payable.min())
    if 'routing' in c:
        out['routing.ring'] = float(c['routing'].router.ring.min())
    if 'groundwater' in c:
        g = c['groundwater']
        h = g.model.h[0].reshape(-1)[g._flat]
        # a dry cell carries MODFLOW's hdry sentinel; the connector reports
        # it at the aquifer bottom (no water, not a negative store)
        wet = h > g.model.hdry * 0.5
        out['groundwater.head_above_bottom'] = float(
            (torch.where(wet, h, g.bot) - g.bot).min())
        out['groundwater.dry_cells'] = int((~wet).sum())
    if 'flood' in c:
        out['flood.depth'] = float(c['flood'].engine.h.min())
    if 'reservoir' in c:
        try:
            r = c['reservoir'].engine
            for nm in ('S', 'storage', 'stor'):
                if hasattr(r, nm):
                    out['reservoir.storage'] = float(getattr(r, nm).min())
                    break
        except Exception as e:                            # noqa: BLE001
            out['reservoir.storage'] = 'n/a: %s' % e
    return out


def tree_equal(a, b, pre='', bad=None, cnt=None):
    import torch
    bad = [] if bad is None else bad
    cnt = [0] if cnt is None else cnt
    if isinstance(a, dict):
        for k in a:
            tree_equal(a[k], b.get(k) if isinstance(b, dict) else None,
                       pre + str(k) + '.', bad, cnt)
    elif isinstance(a, (list, tuple)) and a and torch.is_tensor(a[0]):
        for i, x in enumerate(a):
            tree_equal(x, b[i] if b is not None and i < len(b) else None,
                       pre + '%d.' % i, bad, cnt)
    elif torch.is_tensor(a):
        cnt[0] += 1
        if b is None or not torch.is_tensor(b) or not torch.equal(a, b):
            bad.append(pre)
    return bad, cnt[0]


def run(n, out, end, stop_day, restart_from, ckpt_end, history, syncs,
        days_hint, flood_dtype=None, flood_devices=None):
    import torch
    from torchvic.coupling import TorchVIC
    key, name, in_repo, mode = COMBOS[n]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # one global file (one result path) per combination: a checkpoint is
    # refused by a run whose configuration differs, the path included
    gdir = Path(os.environ.get('COMBO_RUNS', str(COMBO / 'runs'))) / (
        'glob_c%d_%s' % (n, end))
    g = write_global(gdir, end=end, eb=n in EB)
    set_env(g)
    cfgp = (REPO / 'configs' / (name + '.yaml')) if in_repo else \
        COMBO / 'configs' / (name + '.yaml')
    t00 = time.time()
    m = TorchVIC.from_config(str(cfgp), log=say)
    m.cfg['output']['dir'] = str(out / 'results')
    m.cfg['output']['history'] = bool(history)
    if flood_dtype:
        # the connectors exist after from_config; the engine is built at
        # initialize from this attribute
        m.runtime.connectors['flood'].dtype_name = flood_dtype
    if flood_devices:
        m.runtime.connectors['flood'].devices = flood_devices.split(',')
    sink_runoff = []
    if n == 2:
        def sink(nm, prop):
            if nm != 'vic':
                return
            s = prop.series
            for i in range(0, s['runoff'].shape[0], 8):
                sink_runoff.append((s['runoff'][i:i + 8].sum(0)
                                    + s['baseflow'][i:i + 8].sum(0)).clone())
        m.runtime.outputs_mgr.add_sink(sink)
    res = dict(n=n, key=key, config=name, vic_exchange=mode, end=end)
    with torch.inference_mode():
        m.initialize()
        t1 = time.time()
        if restart_from:
            m.load_checkpoint(restart_from)
        day0 = int(round(m.runtime.time.t / 86400.0))
        if stop_day:
            m.run(until=stop_day * 86400.0)
            m.save_checkpoint(ckpt_end)
            res['stopped_day'] = stop_day
        else:
            sc = None
            if syncs:
                sys.path.insert(0, str(REPO / 'benchmarks'))
                from bench_gpu_audit import SyncCounter
                mid = day0 + 60
                m.run(until=mid * 86400.0)
                torch.cuda.synchronize()
                sc = SyncCounter()
                with sc:
                    m.run(until=(mid + 3) * 86400.0)
                res['syncs_per_day'] = sum(sc.hits.values()) / 3.0
                res['sync_sites'] = {k: v / 3.0 for k, v in
                                     sc.hits.most_common(6)}
            m.run()
            if ckpt_end:
                m.save_checkpoint(ckpt_end)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        paths = m.finalize()
    t3 = time.time()
    days = int(round(m.runtime.time.t / 86400.0)) - day0
    rep = m.report()
    bal, cp = m.ledger.balances, m.ledger.coupled
    mods = sorted({r['module'] for r in bal})
    res.update(
        days=days, init_s=round(t1 - t00, 2), run_s=round(t2 - t1, 2),
        finalize_s=round(t3 - t2, 2), wall_s=round(t3 - t00, 2),
        ms_per_day=round(1e3 * (t2 - t1) / max(days, 1), 2),
        peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available() else None,
        peak_gpu_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2)
        if torch.cuda.is_available() else None,
        peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                          / 1e6, 2),
        backends=rep['backends'], timing=rep['timing'],
        modules=mods, ledger_rows=len(bal),
        ledger_rows_ok=sum(1 for r in bal if r['ok']),
        coupled_rows=len(cp), coupled_rows_ok=sum(1 for r in cp if r['ok']),
        coupled_max_abs_R_m3=max([abs(r['residual']) for r in cp] or [0.0]),
        coupled_max_R_over_tol=max([abs(r['residual']) / max(r['tolerance'],
                                                             1e-300)
                                    for r in cp] or [0.0]),
        coupled_sum_R_m3=sum(r['residual'] for r in cp),
        coupled_P_m3=sum(r['P'] for r in cp),
        coupled_ET_m3=sum(r['ET'] for r in cp),
        coupled_Q_boundary_m3=sum(r['Q_boundary'] for r in cp),
        coupled_I_boundary_m3=sum(r['I_boundary'] for r in cp),
        coupled_dS_m3=sum(r['dS'] for r in cp),
        module_totals={k: m.ledger.totals(k) for k in mods},
        module_max_R_over_tol={
            k: max([abs(r['residual']) / max(r['tolerance'], 1e-300)
                    for r in bal if r['module'] == k] or [0.0])
            for k in mods},
        stores_min=store_minima(m))
    if 'flood' in m.runtime.connectors:
        res['flood_totals'] = {
            k: float(v) for k, v in m.connector('flood').totals.items()
            if isinstance(v, (int, float)) or getattr(v, 'numel', lambda: 0)()
            == 1}
    vc = m.connector('vic')
    res['vic_driver'] = type(vc.driver).__name__
    if n == 2 and not stop_day:
        from torchvic.coupling import Context
        from torchvic.coupling.connectors import RoutingConnector
        ref = RoutingConnector(fdir=os.environ['TORCHVIC_FDIR'],
                               domain=os.environ['TORCHVIC_DOMAIN'],
                               gauges=os.environ['TORCHVIC_GAUGES'])
        ref.initialize(Context(device=torch.device('cuda')))
        qs = torch.stack([ref.router.step(x.to(torch.float64))[0]
                          for x in sink_runoff])
        got = m.outputs('routing')['discharge']
        res['routing_check'] = dict(
            days=int(qs.shape[0]), bitwise_equal=bool(torch.equal(got, qs)),
            max_abs_diff=float((got - qs).abs().max()),
            peak_m3s=float(got.max()))
    dump = out / 'vic_state.pt'
    from torchvic.model import state as ms
    if vc.state is not None and not stop_day:
        torch.save(ms.map_state(vc.state, lambda x: x.cpu()), dump)
    (out / 'result.json').write_text(json.dumps(res, indent=1, default=str))
    try:
        for c_ in m.runtime.connectors.values():
            if hasattr(getattr(c_, 'driver', None), 'close'):
                c_.driver.close()
    except Exception:                                     # noqa: BLE001
        pass
    say('RESULT', json.dumps({k: res[k] for k in (
        'n', 'days', 'wall_s', 'run_s', 'ms_per_day', 'peak_gpu_gb',
        'ledger_rows_ok', 'ledger_rows', 'coupled_rows_ok', 'coupled_rows',
        'coupled_max_R_over_tol')}))
    return res


def compare_ckpt(a, b):
    import torch
    A = torch.load(a, weights_only=False, map_location='cpu')
    B = torch.load(b, weights_only=False, map_location='cpu')
    bad, cnt = tree_equal(A['connectors'], B['connectors'])
    bb, cb = tree_equal(A.get('buffers', {}), B.get('buffers', {}))
    la = [(r['module'], r['t0'], r['residual']) for r in
          A['ledger']['balances']]
    lb = [(r['module'], r['t0'], r['residual']) for r in
          B['ledger']['balances']]
    r = dict(arrays=cnt + cb, differ=bad + bb, ledger_rows=len(la),
             ledger_equal=la == lb, t_equal=A['t'] == B['t'])
    r['equal'] = not r['differ'] and r['ledger_equal'] and r['t_equal']
    return r


def gate(path):
    """0 when a run passes the common criteria: every module and coupled
    ledger row closed, no negative store."""
    r = json.load(open(path))
    bad = []
    if r['ledger_rows_ok'] != r['ledger_rows']:
        bad.append('module ledger rows')
    if r['coupled_rows_ok'] != r['coupled_rows']:
        bad.append('coupled ledger rows')
    for k, v in r['stores_min'].items():
        if isinstance(v, (int, float)) and v < -1e-6:
            bad.append('negative store %s = %g' % (k, v))
    rc = r.get('routing_check')
    if rc is not None and not rc['bitwise_equal']:
        bad.append('routing check')
    print('gate %s: %s' % (path, 'PASS' if not bad else 'FAIL ' + str(bad)))
    return 0 if not bad else 1


def table():
    """Markdown rows of the matrix from runs/c*/result.json."""
    R = COMBO / 'runs'
    rows = []
    base = {}
    for n in (1, 13, 14, 15):
        f = R / ('c%d/result.json' % n)
        if f.exists():
            base[n] = json.load(open(f))
    print('| # | combination | exch. | ms/day | vs VIC alone | wall s | peak GPU GB '
          '| syncs/day | module rows ok | coupled rows ok | max R/tol | coupled sum R m3 | min store |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for n in sorted(COMBOS):
        f = R / ('c%d/result.json' % n)
        if not f.exists():
            continue
        r = json.load(open(f))
        eb = n in EB
        d1 = COMBOS[n][3] == 'd1'
        b = base.get(15 if (eb and d1) else 13 if eb else 14 if d1 else 1)
        vs = '%.2fx' % (r['ms_per_day'] / b['ms_per_day']) if b else ''
        mn = [v for v in r['stores_min'].values() if isinstance(v, (int, float))]
        print('| %d | %s | %s | %.1f | %s | %.0f | %.1f | %s | %d/%d | %d/%d | %.1e | %.2e | %.2e |' % (
            n, r['key'], COMBOS[n][3] + ('+EB' if eb else ''), r['ms_per_day'], vs,
            r['wall_s'], r['peak_gpu_gb'],
            ('%.1f' % r['syncs_per_day']) if 'syncs_per_day' in r else '-',
            r['ledger_rows_ok'], r['ledger_rows'], r['coupled_rows_ok'],
            r['coupled_rows'], r['coupled_max_R_over_tol'],
            r['coupled_sum_R_m3'], min(mn) if mn else float('nan')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    gt = sub.add_parser('gate')
    gt.add_argument('result')
    sub.add_parser('prepare')
    sub.add_parser('prepare-2way')
    sub.add_parser('table')
    c = sub.add_parser('configs')
    c.add_argument('--repo', action='store_true')
    g = sub.add_parser('global')
    g.add_argument('out')
    g.add_argument('--end', default='2004-12-31')
    g.add_argument('--eb', action='store_true')
    r = sub.add_parser('run')
    r.add_argument('n', type=int)
    r.add_argument('--out', required=True)
    r.add_argument('--end', default='2004-12-31')
    r.add_argument('--stop-day', type=int, default=0)
    r.add_argument('--restart-from', default=None)
    r.add_argument('--ckpt-end', default=None)
    r.add_argument('--history', action='store_true')
    r.add_argument('--syncs', action='store_true')
    r.add_argument('--flood-dtype', default=None,
                   help="the flood connector's dtype (float64 default; "
                   'mixed)')
    r.add_argument('--flood-devices', default=None,
                   help='multi-GPU flood engine, e.g. cuda:0,cuda:1')
    k = sub.add_parser('cmp-ckpt')
    k.add_argument('a')
    k.add_argument('b')
    a = ap.parse_args()
    if a.cmd == 'gate':
        sys.exit(gate(a.result))
    if a.cmd == 'table':
        table()
    elif a.cmd == 'prepare':
        prepare()
    elif a.cmd == 'prepare-2way':
        prepare_2way()
    elif a.cmd == 'configs':
        write_configs(a.repo)
    elif a.cmd == 'global':
        print(write_global(a.out, end=a.end, eb=a.eb))
    elif a.cmd == 'run':
        run(a.n, a.out, a.end, a.stop_day, a.restart_from, a.ckpt_end,
            a.history, a.syncs, None, a.flood_dtype, a.flood_devices)
        # the interpreter's own teardown aborted once after a finished run
        # (bool_dealloc refcount error in a C extension, combo 3): the
        # results are written, so leave without it
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    elif a.cmd == 'cmp-ckpt':
        print(json.dumps(compare_ckpt(a.a, a.b)))


if __name__ == '__main__':
    main()
