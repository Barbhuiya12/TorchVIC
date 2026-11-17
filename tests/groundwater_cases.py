"""Groundwater validation cases, shared by the tests (torch engine) and the
MODFLOW 6 reference generator (<data root>/verification/gw_engine/mf6_ref.py).
Pure numpy so both environments import the same definitions.

* ``twri`` and ``fhb``: MODFLOW 6 official examples ex-gwf-twri01 and
  ex-gwf-fhb (modflow6-examples, scripts/ex-gwf-twri.py, ex-gwf-fhb.py),
  parameters copied verbatim; the reference runs the example as published.
* ``mixed``: 2-layer DIS model with every supported package and option
  (convertible over confined, VARIABLECV, Ss+Sy, RCH, WEL with
  AUTO_FLOW_REDUCE, RIV, DRN, GHB, CHD), 30 daily steps.
* ``latlon``: a small one-VIC-cell-per-model-cell lat/lon grid (MODFLOW 6
  DISU), unconfined, RCH/WEL/RIV/DRN, 30 daily steps.
* ``isc``: synthetic ISC model on the VIC domain (112,825 cells), one
  unconfined layer, recharge, wells, rivers, drains, coastal GHB, EVT, 365
  daily steps (built from the domain arrays).

EVT (MODFLOW 6 evt_cf): ``mixed`` uses 3 segments (PXDP/PETM), ``latlon``
and ``isc`` the linear form.

A generic case is a dict with ``grid`` (kwargs of Grid.dis / Grid.latlon,
``kind``), ``npf``, ``sto``, ``strt``, ``ims`` and ``nstep``/``dt``; the
boundary cells are fixed and ``stress(case, kstp)`` gives the values of
step ``kstp`` (0-based).
"""
import numpy as np

# MODFLOW 6 IMS used for the generic references; the engine uses the same
# closure (outer_dvclose, inner_dvclose, inner_rclose).
IMS_TIGHT = dict(outer_dvclose=1e-9, inner_dvclose=1e-10, inner_rclose=1e-6,
                 outer_maximum=200, inner_maximum=2000)
# lat/lon grids (km-sized cells): storage terms reach 1e9 m3/d, so the
# residual closure is set above float64 round-off of the row sums
IMS_LATLON = dict(IMS_TIGHT, inner_rclose=1e-2)


# --------------------------------------------------------------- official
def twri():
    """ex-gwf-twri01 (steady state, 5 layers, CHD, DRN, WEL, RCH)."""
    nlay, nrow, ncol = 5, 15, 15
    chd = [(k, i, 0, 0.0) for k in (0, 2) for i in range(nrow)]
    wel = [(4, 4, 10, -5.0), (2, 3, 5, -5.0), (2, 5, 11, -5.0),
           (0, 8, 7, -5.0), (0, 8, 9, -5.0), (0, 8, 11, -5.0),
           (0, 8, 13, -5.0), (0, 10, 7, -5.0), (0, 10, 9, -5.0),
           (0, 10, 11, -5.0), (0, 10, 13, -5.0), (0, 12, 7, -5.0),
           (0, 12, 9, -5.0), (0, 12, 11, -5.0), (0, 12, 13, -5.0)]
    drn = [(0, 7, 1, 0.0, 1.0), (0, 7, 2, 0.0, 1.0), (0, 7, 3, 10.0, 1.0),
           (0, 7, 4, 20.0, 1.0), (0, 7, 5, 30.0, 1.0), (0, 7, 6, 50.0, 1.0),
           (0, 7, 7, 70.0, 1.0), (0, 7, 8, 90.0, 1.0), (0, 7, 9, 100.0, 1.0)]
    return dict(
        name='twri', kind='official',
        grid=dict(kind='dis', nlay=nlay, nrow=nrow, ncol=ncol, delr=5000.0,
                  delc=5000.0, top=200.0,
                  botm=[-150.0, -200.0, -300.0, -350.0, -450.0]),
        npf=dict(icelltype=[1, 0, 0, 0, 0],
                 k11=[1.0e-3, 1.0e-8, 1.0e-4, 5.0e-7, 2.0e-4],
                 k33=[1.0e-3, 1.0e-8, 1.0e-4, 5.0e-7, 2.0e-4],
                 variablecv=True, dewatered=True),  # flopy cvoptions='perched' writes VARIABLECV DEWATERED
        sto=None, strt=0.0, recharge=3e-8, chd=chd, wel=wel, drn=drn,
        perlen=8.640e04, nstp=1,
        ims=dict(outer_maximum=50, inner_maximum=100, outer_dvclose=1e-9,
                 inner_dvclose=1e-9, inner_rclose=1e-6))


def fhb():
    """ex-gwf-fhb (transient, confined, storage coefficient, CHD and WEL
    time series with LINEAREND interpolation)."""
    return dict(
        name='fhb', kind='official',
        grid=dict(kind='dis', nlay=1, nrow=3, ncol=10, delr=1000.0,
                  delc=1000.0, top=50.0, botm=[-200.0]),
        npf=dict(icelltype=[0], k11=[20.0], k33=None, variablecv=False),
        sto=dict(ss=1.0e-6, sy=0.0, iconvert=0, storagecoefficient=True),
        strt=0.0,
        perlen=[400.0, 200.0, 400.0], nstp=[10, 4, 6],
        chd_cells=[(0, i, 9) for i in range(3)],
        chd_ts=[(0.0, 0.0), (307.0, 1.0), (791.0, 5.0), (1000.0, 2.0)],
        wel_cells=[(0, 1, 0)],
        wel_ts=[(0.0, 2000.0), (307.0, 6000.0), (791.0, 5000.0),
                (1000.0, 9000.0)],
        ims=dict(outer_maximum=50, inner_maximum=100, outer_dvclose=1e-9,
                 inner_dvclose=1e-9, inner_rclose=1e-6))


def fhb_steps(c):
    """(dt, chd head, wel q) per time step: LINEAREND = time-series value
    at the end of the step (MODFLOW 6 TimeSeries LINEAREND)."""
    out, t = [], 0.0
    ct, cv = np.array(c['chd_ts']).T
    wt, wv = np.array(c['wel_ts']).T
    for perlen, nstp in zip(c['perlen'], c['nstp']):
        dt = perlen / nstp
        for _ in range(nstp):
            t1 = t + dt
            out.append((dt, float(np.interp(t1, ct, cv)),
                        float(np.interp(t1, wt, wv))))
            t = t1
    return out


# --------------------------------------------------------------- generic
def mixed():
    rng = np.random.default_rng(7)
    nlay, nrow, ncol = 2, 20, 25
    delr = np.linspace(400.0, 600.0, ncol)
    delc = np.linspace(500.0, 300.0, nrow)
    top = 50.0 + 5.0 * rng.random((nrow, ncol))
    botm = np.stack([np.zeros((nrow, ncol)) + rng.random((nrow, ncol)),
                     np.full((nrow, ncol), -50.0)])
    idomain = np.ones((nlay, nrow, ncol), int)
    idomain[:, :3, :2] = 0
    rows = np.arange(3, nrow)
    return dict(
        name='mixed', kind='generic',
        grid=dict(kind='dis', nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
                  delc=delc, top=top, botm=botm, idomain=idomain),
        npf=dict(icelltype=[1, 0], k11=np.stack([
            10.0 * np.exp(0.5 * rng.standard_normal((nrow, ncol))),
            np.full((nrow, ncol), 5.0)]), k33=[1.0, 0.5], variablecv=True),
        sto=dict(ss=1e-5, sy=0.1 + 0.05 * rng.random((nlay, nrow, ncol)),
                 iconvert=[1, 0], storagecoefficient=False),
        strt=30.0 + rng.random((nlay, nrow, ncol)),
        nstep=30, dt=1.0, ims=IMS_TIGHT,
        rch_base=1e-3 * (1 + rng.random((nrow, ncol))),
        wel=dict(cells=([1, 0, 0], [10, 12, 15], [5, 5, 20]),
                 q=[-2e4, -1e4, -3e3], afr=0.1),
        riv=dict(cells=(np.zeros(rows.size, int), rows,
                        np.zeros(rows.size, int)),
                 stage=25.0, cond=1e4, rbot=20.0),
        drn=dict(cells=([0, 0], [5, 6], [20, 20]), elev=[31.0, 29.5],
                 cond=[1e3, 2e3]),
        ghb=dict(cells=([1], [19], [24]), bhead=28.0, cond=500.0),
        chd=dict(cells=([0], [3], [24]), head=35.0),
        evt=dict(surface=top, rate=1e-3, depth=30.0, pxdp=[0.2, 0.5],
                 petm=[0.3, 0.1]))


def latlon():
    rng = np.random.default_rng(11)
    lat = 25.0 + 0.0625 * np.arange(12)
    lon = 80.0 + 0.0625 * np.arange(15)
    nrow, ncol = lat.size, lon.size
    mask = np.ones((nrow, ncol), int)
    mask[0, :4] = 0
    mask[-1, -3:] = 0
    top = 100.0 + 20.0 * rng.random((nrow, ncol))
    botm = top - 80.0
    cells = np.argwhere(mask > 0)
    riv = cells[rng.random(len(cells)) < 0.2]
    wel = cells[rng.choice(len(cells), 6, replace=False)]
    z = lambda a: np.zeros(len(a), int)
    return dict(
        name='latlon', kind='generic',
        grid=dict(kind='latlon', lat=lat, lon=lon, top=top, botm=botm,
                  mask=mask),
        npf=dict(icelltype=[1], k11=8.0 * np.exp(0.4 * rng.standard_normal(
            (1, nrow, ncol))), k33=None, variablecv=False),
        sto=dict(ss=1e-5, sy=0.12, iconvert=[1], storagecoefficient=False),
        strt=(top - 10.0)[None],
        nstep=30, dt=1.0, ims=IMS_LATLON,
        rch_base=2e-3 * rng.random((nrow, ncol)),
        wel=dict(cells=(z(wel), wel[:, 0], wel[:, 1]),
                 q=-5e3 * (1 + rng.random(len(wel))), afr=0.1),
        riv=dict(cells=(z(riv), riv[:, 0], riv[:, 1]),
                 stage=top[riv[:, 0], riv[:, 1]] - 8.0, cond=2e4,
                 rbot=top[riv[:, 0], riv[:, 1]] - 12.0),
        drn=dict(cells=(z(cells), cells[:, 0], cells[:, 1]),
                 elev=top[cells[:, 0], cells[:, 1]] - 1.0, cond=1e4),
        ghb=None, chd=None,
        evt=dict(surface=top, rate=2e-3, depth=15.0))


def isc(lat, lon, mask, area, elev, nstep=365):
    """Synthetic ISC aquifer on the VIC domain (one cell per VIC cell)."""
    rng = np.random.default_rng(2026)
    nrow, ncol = mask.shape
    act = mask > 0
    elev = np.where(act, elev, 0.0)
    top = elev
    botm = (elev - 300.0)[None]
    k = 10 ** (0.7 + 0.3 * rng.standard_normal((nrow, ncol)))
    k = np.where(elev > 1500.0, k / 5.0, k)
    cells = np.argwhere(act)
    e = elev[act]
    # rivers on 12 % of the cells below 2000 m
    rsel = (rng.random(len(cells)) < 0.12) & (e < 2000.0)
    riv = cells[rsel]
    # 8000 wells in the plains (< 400 m)
    plains = np.flatnonzero(e < 400.0)
    wsel = rng.choice(plains, 8000, replace=False)
    wel = cells[np.sort(wsel)]
    # coastal GHB: active cells below 10 m with an inactive neighbour
    pad = np.pad(act, 1)
    edge = ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    gsel = edge[act] & (e < 10.0)
    ghb = cells[gsel]
    z = lambda a: np.zeros(len(a), int)
    tv = lambda a: top[a[:, 0], a[:, 1]]
    return dict(
        name='isc', kind='generic',
        grid=dict(kind='latlon', lat=lat, lon=lon, top=top, botm=botm,
                  mask=mask, area=area),
        npf=dict(icelltype=[1], k11=k[None], k33=None, variablecv=False),
        sto=dict(ss=1e-5, sy=(0.05 + 0.1 * rng.random((nrow, ncol)))[None],
                 iconvert=[1], storagecoefficient=False),
        strt=(top - 15.0)[None],
        nstep=nstep, dt=1.0, ims=IMS_LATLON,
        rch_base=1e-3 * (0.3 + 2.7 * rng.random((nrow, ncol))),
        wel=dict(cells=(z(wel), wel[:, 0], wel[:, 1]),
                 q=-(2e3 + 1.8e4 * rng.random(len(wel))), afr=0.1),
        riv=dict(cells=(z(riv), riv[:, 0], riv[:, 1]), stage=tv(riv) - 3.0,
                 cond=5e4, rbot=tv(riv) - 6.0),
        drn=dict(cells=(z(cells), cells[:, 0], cells[:, 1]),
                 elev=tv(cells) - 1.0, cond=1e4),
        ghb=dict(cells=(z(ghb), ghb[:, 0], ghb[:, 1]), bhead=0.0, cond=1e3),
        chd=None,
        evt=dict(surface=top, rate=2e-3, depth=10.0))


def stress(c, kstp):
    """Values that change per step for generic cases: recharge (seasonal,
    monsoon peak near day 200) and pumping (dry-season peak)."""
    d = float(kstp)
    season = 0.2 + 2.5 * np.exp(-((d % 365.0 - 200.0) / 40.0) ** 2)
    pump = 1.0 + 0.5 * np.cos(2.0 * np.pi * (d - 60.0) / 365.0)
    rch = c['rch_base'] * season
    q = np.asarray(c['wel']['q'], float) * pump
    return rch, q


GENERIC = {'mixed': mixed, 'latlon': latlon}


# ------------------------------------------------------- engine side
def make_grid(g):
    from torchvic.groundwater import Grid
    g = dict(g)
    kind = g.pop('kind')
    return Grid.dis(**g) if kind == 'dis' else Grid.latlon(**g)


def _solver(ims, solver):
    s = dict(outer_dvclose=ims['outer_dvclose'],
             inner_dvclose=ims['inner_dvclose'],
             inner_rclose=ims['inner_rclose'],
             outer_maximum=max(ims['outer_maximum'], 100),
             inner_maximum=max(ims['inner_maximum'], 2000))
    s.update(solver)
    return s


def build_engine(c, device='cpu', **solver):
    """GroundwaterModel for a case (stresses of step 0 for generic)."""
    from torchvic.groundwater import GroundwaterModel
    grid = make_grid(c['grid'])
    npf, sto = c['npf'], c['sto'] or {}
    m = GroundwaterModel(
        grid, k11=npf['k11'], k33=npf['k33'], icelltype=npf['icelltype'],
        ss=sto.get('ss', 0.0), sy=sto.get('sy', 0.0),
        iconvert=sto.get('iconvert'), strt=c['strt'],
        storagecoefficient=sto.get('storagecoefficient', False),
        variablecv=npf['variablecv'], dewatered=npf.get('dewatered', False),
        device=device, **_solver(c['ims'], solver))
    if c['name'] == 'twri':
        m.set_recharge(c['recharge'])
        k, i, j, h = zip(*c['chd'])
        m.set_chd((k, i, j), h)
        k, i, j, e, cd = zip(*c['drn'])
        m.set_drains((k, i, j), e, cd)
        k, i, j, q = zip(*c['wel'])
        m.set_wells((k, i, j), q)
    elif c['name'] == 'fhb':
        st = fhb_steps(c)
        m.set_chd(tuple(zip(*c['chd_cells'])), st[0][1])
        m.set_wells(tuple(zip(*c['wel_cells'])), st[0][2])
    else:
        rch, q = stress(c, 0)
        m.set_recharge(rch)
        w = c['wel']
        m.set_wells(w['cells'], q, auto_flow_reduce=w['afr'])
        if c.get('riv'):
            b = c['riv']
            m.set_rivers(b['cells'], b['stage'], b['cond'], b['rbot'])
        if c.get('drn'):
            b = c['drn']
            m.set_drains(b['cells'], b['elev'], b['cond'])
        if c.get('ghb'):
            b = c['ghb']
            m.set_ghb(b['cells'], b['bhead'], b['cond'])
        if c.get('chd'):
            b = c['chd']
            m.set_chd(b['cells'], b['head'])
        if c.get('evt'):
            e = c['evt']
            R, C = m.shape[1:]
            seg = lambda v: (None if v is None else
                             np.broadcast_to(np.asarray(v, float)[:, None, None],
                                             (len(v), R, C)))
            m.set_evt(e['surface'], e['rate'], e['depth'],
                      seg(e.get('pxdp')), seg(e.get('petm')))
    return m


def run_engine(c, m, nstep=None, on_step=None):
    """Run a case; returns heads (nstep, L, R, C) numpy (NaN inactive/dry)
    and budgets {package: (nstep, 3) in, out, net}."""
    import torch
    heads, bud, res = [], {}, []
    if c['name'] == 'twri':
        steps = [(c['perlen'], None)]
    elif c['name'] == 'fhb':
        steps = fhb_steps(c)
    else:
        steps = [(c['dt'], k) for k in range(c['nstep'])]
    for kstp, st in enumerate(steps[:nstep]):
        if c['name'] == 'fhb':
            m.set_chd(head=st[1])
            m.set_wells(q=st[2])
        elif c['name'] != 'twri':
            rch, q = stress(c, kstp)
            m.set_recharge(rch)
            m.set_wells(q=q)
        r = m.step(st[0], steady=(c['name'] == 'twri'))
        h = m.h.clone()
        h = torch.where(m.ibound != 0, h, torch.full_like(h, float('nan')))
        heads.append(h.cpu().numpy())
        for k, (a, b) in r.rates().items():
            net = float(r.flows[k].sum()) if k in r.flows else a - b
            bud.setdefault(k, []).append((a, b, net))
        res.append(r)
        if on_step:
            on_step(kstp, r)
    return np.array(heads), {k: np.array(v) for k, v in bud.items()}, res


def compare(heads, bud, ref, official=False):
    """Max head difference (m) and max budget difference per package,
    relative to the step's total inflow. Official runs read the cell budget
    file, whose CHD record is the net rate per cell: CHD is compared net."""
    hr = ref['heads']
    act = np.isfinite(heads) & np.isfinite(hr) & (np.abs(hr) < 1e29)
    dh = float(np.max(np.abs(heads - hr)[act])) if act.any() else 0.0
    tot = sum(bud[k][:, 0] for k in bud)
    out = {}
    for k in bud:
        rb = ref.get('b_' + k, ref.get('b_' + k + 'A'))
        if rb is None:
            continue
        if official and k == 'CHD':
            d = np.abs(bud[k][:, 2] - rb[:, 2])
        else:
            d = np.abs(bud[k][:, :2] - rb[:, :2]).max(1)
        out[k] = float(np.max(d / np.maximum(tot, 1e-300)))
    return dh, out
