"""TorchVIC's VIC 5.1.0 core (torchvic.core.version) pinned against VIC 5.1.0.

core_version.set_core('5.1.0') switches every 5.0.1 -> 5.1.0 difference in
the core physics where it lives (torchvic/core/version.py); 5.1.0 has been
the default core since 0.4.8, and test_wb_vs_vic.py pins the 5.0.1 core
(vic_core marker) against VIC 5.0.1.
Here the same comparisons run against a pristine VIC 5.1.0 image driver
(reference/VIC-5.1.0, built from the release tarball), and skip where it is
not built.

The change that moves the most water on these sites is vic_force.c
5.1.0:417-440 (GH#823): a tile in a month with LAI 0 has fcanopy 0 and runs
as bare soil, where 5.0.1 ran a closed canopy of no leaves that evaporated
nothing from its soil. Walnut Gulch has no leafless class, so two of the
cases below give it one (a class's LAI set to 0 in February, which also
exercises the LAI > 0 -> 0 transition that deletes the canopy store), and
the last-class cases of test_wb_vs_vic.py are repeated: under 5.1.0 the
FROM_DEFAULT MIN_FCANOPY of a leafless last class is 0.

Measured (annual or run totals, TorchVIC minus VIC over VIC) are in each
test's docstring.
"""
import glob
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import _paths                                           # noqa: E402
from torchvic.core import version as core_version, vegetation  # noqa: E402
from torchvic.core.vegetation import VegMosaic  # noqa: E402

D = torch.float64
CPU = torch.device('cpu')

REF = _paths.REFERENCE
VIC51 = _paths.VIC51_IMAGE
WG_FULL = _paths.WG_FULL
WG_DOMAIN = _paths.WG_DOMAIN
SAMPLES = os.path.join(_paths.SAMPLES, 'image', 'Stehekin')
STEH_GLOBAL = os.path.join(SAMPLES, 'parameters',
                           'Stehekin_image_test.global.txt')
STEH_DOMAIN = os.path.join(SAMPLES, 'parameters',
                           'domain.stehekin.20151028.nc')


@pytest.fixture(autouse=True)
def _core_510_on_cpu():
    # The core is read when a case is loaded (image_run.build_soil,
    # build_mosaic) and at every step, so it is set for the whole test and
    # put back after; the CPU default as in test_wb_vs_vic.py.
    was = torch.empty(0).device
    torch.set_default_device('cpu')
    old = core_version.CORE
    core_version.set_core('5.1.0')
    yield
    core_version.set_core(old)
    torch.set_default_device(was)


def T(v, *shape):
    return torch.full(shape, float(v), dtype=D, device=CPU)


# ---------------------------------------------------------------------------
# The rule itself, on one cell
# ---------------------------------------------------------------------------
def _mosaic(fc, lai=3.0, bare_tile=None):
    nt = fc.shape[1]
    return VegMosaic(cv=T(1.0 / nt, 1, nt), lai=T(lai, 1, nt, 12),
                     rmin=T(100.0, 1, nt), rgl=T(100.0, 1, nt),
                     rarc=T(60.0, 1, nt),
                     root_frac=torch.tensor([[[0.2, 0.2, 0.6]] * nt],
                                            dtype=D, device=CPU),
                     dtype=D, device=CPU, fcanopy=fc, bare_tile=bare_tile)


def _bare_soil(nt, moist=30.0):
    return dict(moist=T(moist, 1, nt), max_moist=T(40.0, 1, nt),
                b_infilt=T(0.2, 1, nt), depth=T(0.1, 1, nt),
                resid_frac=T(0.02, 1, nt), resid_mm=T(2.0, 1, nt),
                ra=T(60.0, 1, nt))


def _step(mos, wdew, rain, **kw):
    avail = torch.tensor([[40.0, 60.0, 200.0]], dtype=D, device=CPU)
    wcr = torch.tensor([[30.0, 45.0, 120.0]], dtype=D, device=CPU)
    wpwp = torch.tensor([[15.0, 20.0, 60.0]], dtype=D, device=CPU)
    return mos.step(0, wdew, rain, avail, wcr, wpwp,
                    T(300.0, 1), T(30.0, 1), T(3000.0, 1), T(30.0, 1),
                    T(250.0, 1), T(300.0, 1), dt=3 * 3600.0,
                    per_tile=True, wilting_clamp=True, exact_wetness=True,
                    **kw)


def test_a_leafless_tile_is_bare_soil_and_its_store_is_deleted():
    """func_surf_energy_bal.c 5.1.0:730-761 with VEG false (fcanopy 0 after
    vic_force.c 5.1.0:419-439): no interception, no transpiration, the rain
    straight through, Wdew *= fcanopy (0: a store left from a leafy month is
    deleted, not drained), and the whole tile ARNO bare soil at the
    unattenuated NetBareRad. A VEG tile at fcanopy 1 is the closed canopy
    exactly, with no bare-soil term."""
    fc = torch.tensor([[[0.0] * 12, [1.0] * 12]], dtype=D, device=CPU)
    mos = _mosaic(fc)
    mos.lai = torch.where(fc == 0.0, torch.zeros_like(mos.lai), mos.lai)
    wdew = torch.tensor([[0.2, 0.2]], dtype=D, device=CPU)
    rain = torch.tensor([[1.5, 1.5]], dtype=D, device=CPU)
    tr, ce, tf, nw, bare = _step(mos, wdew, rain,
                                 bare_soil=_bare_soil(2))
    assert torch.equal(tr[0, 0], torch.zeros(3, dtype=D, device=CPU))
    assert float(ce[0, 0]) == 0.0 and float(tf[0, 0]) == 1.5
    assert float(nw[0, 0]) == 0.0
    assert float(bare[0, 0]) > 0.0
    assert float(bare[0, 1]) == 0.0 and float(tr[0, 1].sum()) > 0.0
    # the closed canopy is what the fcanopy-less (5.0.1-default) mosaic gives
    closed = _mosaic(T(1.0, 1, 1, 12))
    closed.fcanopy = None
    ref = _step(closed, wdew[:, 1:], rain[:, 1:])
    for a, b in zip((tr, ce, tf, nw), ref):
        assert torch.equal(a[:, 1:], b)
    # Under snow the leafless tile's store goes too (solve_snow.c 5.1.0:251)
    _, _, _, nw_s, _ = _step(mos, wdew, rain, bare_soil=_bare_soil(2),
                             snowing=torch.tensor([[True, True]]))
    assert float(nw_s[0, 0]) == 0.0 and float(nw_s[0, 1]) == 0.2


def test_the_bare_tile_evaporates_one_minus_fcanopy():
    """5.0.1 evaporated its bare-soil tile (iveg == Nveg) whole
    (func_surf_energy_bal.c 5.0.1:797-805); 5.1.0 weights it by
    (1 - fcanopy) like any tile that is not VEG (5.1.0:749-752)."""
    fc = torch.tensor([[[0.25] * 12]], dtype=D, device=CPU)
    bt = torch.tensor([[True]])
    mos = _mosaic(fc, bare_tile=bt)
    wdew, rain = T(0.0, 1, 1), T(0.0, 1, 1)
    tr, ce, tf, nw, bare = _step(mos, wdew, rain, bare_soil=_bare_soil(1))
    core_version.set_core('5.0.1')
    try:
        _, _, _, _, whole = _step(mos, wdew, rain, bare_soil=_bare_soil(1))
    finally:
        core_version.set_core('5.1.0')
    assert float(tr.abs().sum()) == 0.0 and float(ce) == 0.0
    assert float(whole) > 0.0
    assert abs(float(bare) - 0.75 * float(whole)) < 1e-15 * float(whole)


def test_build_mosaic_applies_the_leafless_rule_and_renormalises_cover():
    """image_run.build_mosaic under 5.1.0: fcanopy and LAI 0 wherever LAI
    is 0 or fcanopy below MIN_FCANOPY (vic_force.c 5.1.0:419-439), and the
    cover divided by its sum where that is not exactly 1 (vic_init.c 5.1.0
    :1265-1277)."""
    from torchvic.io import build as image_run

    class _P(object):
        path = 'synthetic'
        ds = object()           # an image-driver parameter set
        nveg = None

    p = _P()
    lai = np.ones((3, 12, 2))
    lai[1, 1:4, 0] = 0.0                      # class 1 leafless Feb-Apr
    p.veg = dict(Cv=np.array([[0.5, 0.3], [0.3, 0.7], [0.2 + 1e-7, 0.0]]),
                 LAI=lai, root_fract=np.full((3, 3, 2), 1.0 / 3.0),
                 root_depth=np.full((3, 3, 2), 0.3),
                 rmin=np.full((3, 2), 100.0), RGL=np.full((3, 2), 100.0),
                 rarc=np.full((3, 2), 60.0))
    p.soil = dict(depth=np.full((3, 2), 0.3))
    m = image_run.build_mosaic(p, D, CPU, log=lambda *a: None)
    fc = m.fcanopy
    assert fc is not None
    assert torch.equal(fc[0, 1, 1:4], torch.zeros(3, dtype=D))
    assert torch.equal(m.lai[0, 1, 1:4], torch.zeros(3, dtype=D))
    assert float(fc[0, 1, 0]) == 1.0 and float(fc[0, 0].min()) == 1.0
    # class 2 is the last: FROM_DEFAULT's MIN_FCANOPY, its LAI is 1
    assert float(fc[0, 2, 0]) == image_run.MIN_FCANOPY
    s = m.cv[0, 0] + m.cv[0, 1] + m.cv[0, 2]
    assert abs(float(s) - 1.0) < 1e-15
    assert torch.equal(m.cv[1], torch.tensor([0.3, 0.7, 0.0], dtype=D))


def test_new_snow_density_is_capped_and_a_trace_pack_conducts_nothing():
    """snow_utility.c 5.1.0:218-222 caps new-snow density at
    SNOW_NEW_SNOW_DENS_MAX = 400 (5.0.1: 789.6 kg/m3 at +30 C, uncapped);
    SnowPackEnergyBalance.c 5.1.0:236 conducts no ground heat through a
    pack under SNOW_DEPTH_THRES = 1e-8 m (5.0.1: any depth > 0)."""
    from torchvic.core import snow
    t = torch.tensor([-5.0, 0.0, 30.0], dtype=D)
    capped = snow.new_snow_density(t)
    core_version.set_core('5.0.1')
    try:
        free = snow.new_snow_density(t)
    finally:
        core_version.set_core('5.1.0')
    assert torch.equal(capped[:2], free[:2])
    assert float(free[2]) > 700.0 and float(capped[2]) == 400.0
    st = dict(t_air=T(-2.0, 2), rain=T(0.0, 2), dt=3600.0,
              swe_surface=T(1e-6, 2), old_t_surf=T(-1.0, 2),
              snow_depth=torch.tensor([5e-9, 1e-3], dtype=D),
              snow_density=T(100.0, 2), t_grnd=T(1.0, 2),
              surface_liquid_water=T(0.0, 2), net_short=T(0.0, 2),
              long_snow_in=T(250.0, 2), air_dens=T(1.2, 2),
              eact_air=T(400.0, 2), lv=T(2.5e6, 2), press=T(9e4, 2),
              vpd=T(100.0, 2), z=T(2.03, 2), z0=T(0.03, 2), ra=T(50.0, 2),
              wind=T(2.0, 2), blowing_mass_flux=T(0.0, 2))
    _, _, _, terms = snow.snow_pack_energy_balance(T(-1.5, 2), st,
                                                   return_terms=True)
    assert float(terms['grnd_flux'][0]) == 0.0
    assert float(terms['grnd_flux'][1]) != 0.0


# ---------------------------------------------------------------------------
# Against pristine VIC 5.1.0 on Walnut Gulch
# ---------------------------------------------------------------------------
PAIRS = (('OUT_EVAP', 'et'), ('OUT_TRANSP_VEG', 'transp'),
         ('OUT_EVAP_CANOP', 'evap_canop'), ('OUT_EVAP_BARE', 'evap_bare'),
         ('OUT_RUNOFF', 'runoff'), ('OUT_BASEFLOW', 'baseflow'),
         ('OUT_SUB_SNOW', 'sub_snow'))


def _need_vic():
    if not (os.path.exists(VIC51) and os.path.exists(WG_FULL)
            and os.path.exists(WG_DOMAIN)):
        pytest.skip('pristine VIC 5.1.0 (reference/VIC-5.1.0) or the Walnut '
                    'Gulch sample data is not present')
    pytest.importorskip('xarray')


def _run_vic(tmp, subs, src=WG_FULL):
    """VIC 5.1.0 on the Walnut Gulch global file with `subs` applied;
    returns the file and its daily history as an xarray Dataset."""
    import xarray as xr
    res, logs = os.path.join(tmp, 'results'), os.path.join(tmp, 'logs')
    os.makedirs(res), os.makedirs(logs)
    lines = []
    for ln in open(src).read().splitlines():
        key = ln.split()[0] if ln.split() else ''
        if key in subs:
            ln = '%s %s' % (key, subs[key])
        elif key == 'RESULT_DIR':
            ln = 'RESULT_DIR %s' % res
        elif key == 'LOG_DIR':
            ln = 'LOG_DIR %s/' % logs
        lines.append(ln)
    gf = os.path.join(tmp, 'global.txt')
    open(gf, 'w').write('\n'.join(lines) + '\n')
    env = dict(os.environ, OMPI_MCA_btl_vader_single_copy_mechanism='none')
    r = subprocess.run([VIC51, '-g', gf], env=env, capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return gf, _history(res)


def _history(res):
    """VIC's history files, concatenated in time (no dask needed)."""
    import xarray as xr
    files = sorted(glob.glob(os.path.join(res, '*.nc')))
    assert files, 'VIC wrote no history'
    parts = [xr.open_dataset(f).load() for f in files]
    return xr.concat(parts, dim='time') if len(parts) > 1 else parts[0]


def _basin_daily(vic, ours, spd, pairs=PAIRS):
    import xarray as xr
    dom = xr.open_dataset(WG_DOMAIN)
    mask = dom['mask'].values.astype(bool)
    w = dom['area'].values[mask]
    w = w / w.sum()
    out = {}
    for vk, ok in pairs:
        if vk not in vic or ok not in ours:
            continue
        a = vic[vk].values[:, mask]
        b = ours[ok].detach().cpu().numpy()
        b = b.reshape(b.shape[0] // spd, spd, b.shape[1]).sum(1)
        out[vk] = ((a * w).sum(-1), (b * w).sum(-1))
    return out


def _compare(series, label, tol_rel=1e-4):
    lines, bad = [], []
    for vk, (a, b) in series.items():
        assert a.shape == b.shape, (vk, a.shape, b.shape)
        A, B = float(a.sum()), float(b.sum())
        rel = (B - A) / A if abs(A) > 1e-9 else 0.0
        p99 = float(np.percentile(np.abs(a - b), 99))
        lines.append('  %-15s VIC %10.4f  TorchVIC %10.4f  %+.4f%%  '
                     'daily p99|d| %.2e mm' % (vk, A, B, 100 * rel, p99))
        if abs(A) > 0.05 and abs(rel) > tol_rel:
            bad.append(vk)
    print(label)
    print('\n'.join(lines))
    assert not bad, '%s: %s off VIC by more than %.2g' % (label, bad, tol_rel)


def _leafless_params(tmp, month=1):
    """Walnut Gulch's parameters with the largest class's LAI set to 0 in
    `month` (0-based; February): VIC 5.1.0 runs that class as bare soil for
    the month, and deletes its canopy store at the start of it."""
    import xarray as xr
    src = [ln.split()[1] for ln in open(WG_FULL)
           if ln.split() and ln.split()[0] == 'PARAMETERS'][0]
    d = xr.open_dataset(src).load()
    cv = np.nan_to_num(d['Cv'].values)
    big = int(np.argmax(cv.sum(axis=(1, 2))))
    d['LAI'].values[big, month] = 0.0
    out = os.path.join(tmp, 'params_leafless.nc')
    d.to_netcdf(out)
    return out, big


@pytest.mark.parametrize('leafless', [None, 2],
                         ids=['sample', 'largest_class_leafless_in_march'])
def test_walnut_gulch_three_hourly_water_balance_is_vic51s(tmp_path,
                                                           leafless):
    """Walnut Gulch, January-March 2008, 3-hourly, FULL_ENERGY FALSE, VIC
    5.1.0 against TorchVIC's 5.1.0 core: the thin packs and the snow gate,
    and (second case) the largest class (9, 71% of the basin) leafless in
    March, so run as bare soil for the month with its canopy store deleted
    on 1 March. Every flux within 1e-4 of VIC's; measured, every flux
    within 0.0000% and no day off by more than 1e-6 mm in either case."""
    _need_vic()
    from torchvic import case
    subs = {'FULL_ENERGY': 'FALSE', 'ENDMONTH': '03', 'ENDDAY': '31'}
    if leafless is not None:
        subs['PARAMETERS'], big = _leafless_params(str(tmp_path), leafless)
    gf, vic = _run_vic(str(tmp_path), subs)
    c = case.load(gf, device=CPU, steps=8 * 91)
    assert not c.full_energy and c.model_steps == 8
    if leafless is not None:
        assert c.mosaic.fcanopy is not None
        assert float(c.mosaic.fcanopy[:, :, leafless].min()) == 0.0
    with torch.no_grad():
        out = c.run(p=c.params())
    series = _basin_daily(vic, out, 8)
    if leafless is not None:
        assert float(series['OUT_EVAP_BARE'][0].sum()) > 1.0
    _compare(series, 'Walnut Gulch Jan-Mar 2008, 3-hourly water balance, '
             'VIC 5.1.0%s' % ('' if leafless is None else
                              ', class %d leafless in month %d'
                              % (big, leafless + 1)))


@pytest.mark.parametrize('leafless', [None, 6],
                         ids=['sample', 'largest_class_leafless_in_july'])
def test_walnut_gulch_summer_energy_balance_is_vic51s(tmp_path, leafless):
    """Walnut Gulch, June-August 2008, 3-hourly, FULL_ENERGY TRUE, VIC
    5.1.0 against TorchVIC's 5.1.0 core, and class 9 leafless in July: the
    surface energy balance with the tile ARNO bare soil at Ra_used[0], the
    conductance average with the bare soil's resistance at SOIL_WINDH (10
    m), stability-corrected at each trial temperature. Measured: every flux
    within 0.0000% in both cases (bare soil 45.7007 mm both); with the bare
    soil's reference height at 2 m, bare soil was -4.9%."""
    _need_vic()
    from torchvic import case
    subs = {'STARTMONTH': '06', 'STARTDAY': '01', 'ENDMONTH': '08',
            'ENDDAY': '31'}
    if leafless is not None:
        subs['PARAMETERS'], big = _leafless_params(str(tmp_path), leafless)
    gf, vic = _run_vic(str(tmp_path), subs)
    # the global file's STARTMONTH 06 starts the run (case.window)
    c = case.load(gf, device=CPU, steps=8 * 92)
    assert c.full_energy and c.model_steps == 8
    with torch.no_grad():
        out = c.run(p=c.params())
    series = _basin_daily(vic, out, 8)
    if leafless is not None:
        assert float(series['OUT_EVAP_BARE'][0].sum()) > 40.0
    _compare(series, 'Walnut Gulch Jun-Aug 2008, 3-hourly energy balance, '
             'VIC 5.1.0%s' % ('' if leafless is None else
                              ', class %d leafless in month %d'
                              % (big, leafless + 1)))


def test_a_leafless_tile_under_rain_on_snow_is_not_yet_vics(tmp_path):
    """The same with class 9 leafless in February: on 16 February rain
    falls on a pack that, with no canopy store dumped into it, is thin
    enough to be solved differently. Measured against VIC 5.1.0: ET
    -0.0009%, transpiration and canopy evaporation 0.0000%, bare soil
    +0.07%, runoff +0.11%, snow sublimation -1.54%, all from that one day
    on one cell. NOT a 5.1.0 port defect: VIC 5.0.1 against the 5.0.1 core
    with the class at fcanopy 0.3 in February (FROM_VEGPARAM) gives the
    same -1.51%. VIC's own trace shows the step differ
    in the full pack solve (snow_melt.c), which on a tile without an
    overstory takes its resistance from the global WIND_H here
    (basin.snow_forcing_at; Ra 39.36 against VIC's 31.20 at the class's
    wind_h of 3 m) and whose density the step before already differs
    (132.9 against 121.6 kg/m3). Pinned so a regression shows; tighten
    when the pack resistance is per tile."""
    _need_vic()
    from torchvic import case
    subs = {'FULL_ENERGY': 'FALSE', 'ENDMONTH': '03', 'ENDDAY': '31'}
    subs['PARAMETERS'], big = _leafless_params(str(tmp_path), 1)
    gf, vic = _run_vic(str(tmp_path), subs)
    c = case.load(gf, device=CPU, steps=8 * 91)
    with torch.no_grad():
        out = c.run(p=c.params())
    series = _basin_daily(vic, out, 8)
    rel = {}
    for vk, (a, b) in series.items():
        A, B = float(a.sum()), float(b.sum())
        rel[vk] = (B - A) / A if abs(A) > 1e-9 else 0.0
        print('  %-15s VIC %10.4f  TorchVIC %10.4f  %+.4f%%'
              % (vk, A, B, 100 * rel[vk]))
    assert abs(rel['OUT_EVAP']) < 1e-4
    assert abs(rel['OUT_TRANSP_VEG']) < 1e-5
    assert abs(rel['OUT_BASEFLOW']) < 1e-5
    assert abs(rel['OUT_EVAP_BARE']) < 2e-3
    assert abs(rel['OUT_RUNOFF']) < 3e-3
    assert abs(rel['OUT_SUB_SNOW']) < 0.03


def _on_the_file_day(monkeypatch):
    from torchvic import case
    from torchvic.io.forcing import aggregate_to_daily
    monkeypatch.setattr(
        case, 'daily_by_off_gmt',
        lambda fields, times, off_gmt, log=None: aggregate_to_daily(
            fields, times, utc_offset=0.0, log=log))


def _bare_params(tmp, count_it):
    """test_wb_vs_vic._bare_params: 30% of each cell's largest class moved
    to the last class (LAI 0), Nveg one short or counting it."""
    import xarray as xr
    src = [ln.split()[1] for ln in open(WG_FULL)
           if ln.split() and ln.split()[0] == 'PARAMETERS'][0]
    d = xr.open_dataset(src).load()
    cv = d['Cv'].values.copy()
    last = cv.shape[0] - 1
    big = np.argmax(np.nan_to_num(cv), axis=0)
    move = np.zeros_like(cv[0])
    for j in range(cv.shape[1]):
        for i in range(cv.shape[2]):
            k = big[j, i]
            if cv[k, j, i] > 0:
                move[j, i] = 0.3 * cv[k, j, i]
                cv[k, j, i] -= move[j, i]
    cv[last] = move
    d['Cv'].values[:] = cv
    if count_it:
        d['Nveg'].values[:] = d['Nveg'].values + (move > 0)
        for v in ('root_depth', 'root_fract'):
            a = d[v].values
            for j in range(cv.shape[1]):
                for i in range(cv.shape[2]):
                    a[last, :, j, i] = a[big[j, i], :, j, i]
    out = os.path.join(tmp, 'params_bare.nc')
    d.to_netcdf(out)
    return out


@pytest.mark.parametrize('count_it', [False, True],
                         ids=['bare_tile', 'last_class_counted'])
def test_the_leafless_last_class_is_vic51s(tmp_path, monkeypatch, count_it):
    """test_wb_vs_vic's last-class case under 5.1.0, June-August 2008,
    daily: the last class has LAI 0, so its FROM_DEFAULT MIN_FCANOPY goes
    to 0 (vic_force.c 5.1.0:419-439) and the tile is wholly bare soil
    whether VIC runs it as its bare-soil tile or as vegetation."""
    _need_vic()
    from torchvic import case
    pf = _bare_params(str(tmp_path), count_it)
    gf, vic = _run_vic(str(tmp_path), {'FULL_ENERGY': 'FALSE',
                                        'MODEL_STEPS_PER_DAY': '1',
                                        'STARTMONTH': '06', 'STARTDAY': '01',
                                        'ENDMONTH': '08', 'ENDDAY': '31',
                                        'PARAMETERS': pf})
    first = 0                # the global file's STARTMONTH 06 starts it
    _on_the_file_day(monkeypatch)
    c = case.load(gf, device=CPU, daily=True, first_step=first, steps=8 * 92)
    fc = c.mosaic.fcanopy
    assert torch.equal(fc[:, -1], torch.zeros_like(fc[:, -1]))
    with torch.no_grad():
        out = c.run(p=c.params())
    series = _basin_daily(vic, out, 1)
    assert float(series['OUT_EVAP_BARE'][0].sum()) > 30.0
    _compare(series, 'Walnut Gulch Jun-Aug 2008 daily, VIC 5.1.0, 30%% of '
             'cover in the leafless last class, Nveg %s'
             % ('counting it' if count_it else 'one short'))


# ---------------------------------------------------------------------------
# Stehekin: an overstory under snow
# ---------------------------------------------------------------------------
def _stehekin_global(tmp):
    sd = os.path.dirname(os.path.dirname(SAMPLES))
    res, logs = os.path.join(tmp, 'results'), os.path.join(tmp, 'logs')
    os.makedirs(res), os.makedirs(logs)
    lines = []
    for ln in open(STEH_GLOBAL).read().splitlines():
        key = ln.split()[0] if ln.split() else ''
        ln = ln.replace('${VIC_SAMPLE_DATA}', sd)
        if key == 'SNOW_BAND':
            ln = 'SNOW_BAND FALSE'
        elif key == 'RESULT_DIR':
            ln = 'RESULT_DIR %s/' % res
        elif key == 'COMPRESS':
            ln = 'COMPRESS FALSE'
        lines.append(ln)
    lines.append('LOG_DIR %s/' % logs)
    lines.append('WIND_H 10.0')
    for v in ('OUT_TRANSP_VEG', 'OUT_EVAP_CANOP', 'OUT_SUB_SNOW',
              'OUT_SUB_CANOP', 'OUT_SNOW_CANOPY'):
        lines.append('OUTVAR %s' % v)
    gf = os.path.join(tmp, 'global.txt')
    open(gf, 'w').write('\n'.join(lines) + '\n')
    return gf, res


def test_stehekin_water_balance_against_vic51(tmp_path):
    """Stehekin, January 1949, hourly, FULL_ENERGY FALSE, against VIC
    5.1.0: an overstory under snow, every water row within 0.005% as
    under 5.0.1 (test_wb_vs_vic), same bounds, so a 5.1.0-only regression
    shows."""
    if not (os.path.exists(VIC51) and os.path.exists(STEH_GLOBAL)):
        pytest.skip('pristine VIC 5.1.0 or the Stehekin sample is not present')
    xr = pytest.importorskip('xarray')
    from torchvic import case
    gf, res = _stehekin_global(str(tmp_path))
    env = dict(os.environ, OMPI_MCA_btl_vader_single_copy_mechanism='none')
    r = subprocess.run([VIC51, '-g', gf], env=env, capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    vic = _history(res)
    c = case.load(gf, device=CPU)
    c.batch.elevation = torch.as_tensor(np.asarray(c.par.soil['elev']),
                                        dtype=c.dtype, device=c.device)
    with torch.no_grad():
        out = c.run(p=c.params())
    dom = xr.open_dataset(STEH_DOMAIN)
    mask = (dom['mask'].values.astype(bool)
            & np.isfinite(vic['OUT_EVAP'].values[0]))
    w = dom['area'].values[mask]
    w = w / w.sum()
    rel = {}
    for vk, ok in (('OUT_EVAP', 'et'), ('OUT_SUB_SNOW', 'sub_snow'),
                   ('OUT_SUB_CANOP', 'sub_canop'),
                   ('OUT_EVAP_CANOP', 'evap_canop'),
                   ('OUT_BASEFLOW', 'baseflow'), ('OUT_SWE', 'swe')):
        a = (vic[vk].values[:, mask] * w).sum(-1)
        b = out[ok].detach().cpu().numpy()
        b = (b.reshape(10, 24, -1).sum(1) if vk != 'OUT_SWE'
             else b.reshape(10, 24, -1)[:, -1]) @ w
        A, B = (a.sum(), b.sum()) if vk != 'OUT_SWE' else (a.mean(), b.mean())
        rel[vk] = (B - A) / A
        print('  %-15s VIC %9.5f  TorchVIC %9.5f  %+.2f%%'
              % (vk, A, B, 100 * rel[vk]))
    assert abs(rel['OUT_BASEFLOW']) < 1e-6
    # Measured after the overstory-under-snow fixes (opening Tfoliage and
    # LongUnderOut, the ground's LongUnderOut, IntRainOrg, saturated vp,
    # thin pack, albedo age): every row within 0.005%. Were ET +11%,
    # canopy sublimation +55%, canopy evaporation -23%.
    for k in ('OUT_SWE', 'OUT_EVAP', 'OUT_SUB_SNOW', 'OUT_SUB_CANOP',
              'OUT_EVAP_CANOP'):
        assert abs(rel[k]) < 5e-4, (k, rel[k])


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------
def test_gradients_through_a_leafless_tile_are_finite(tmp_path):
    """30 days of Walnut Gulch 3-hourly water balance under 5.1.0 with the
    largest class leafless in January (fcanopy 0 there): every fcanopy
    division is guarded by a 1 in place of 0 BEFORE any where, so every
    parameter gradient is finite, and the leafless tile's bare soil still
    carries a gradient."""
    if not os.path.exists(WG_FULL):
        pytest.skip('the Walnut Gulch global file is not present')
    pytest.importorskip('xarray')
    from torchvic import case
    pf, _ = _leafless_params(str(tmp_path), month=0)
    lines = []
    for ln in open(WG_FULL).read().splitlines():
        if ln.split() and ln.split()[0] == 'PARAMETERS':
            ln = 'PARAMETERS %s' % pf
        lines.append(ln)
    gf = os.path.join(str(tmp_path), 'global.txt')
    open(gf, 'w').write('\n'.join(lines) + '\n')
    c = case.load(gf, device=CPU, no_energy=True, steps=8 * 30)
    assert float(c.mosaic.fcanopy[:, :, 0].min()) == 0.0
    p = c.params()
    names = [k for k, v in p.items()
             if torch.is_tensor(v) and v.is_floating_point()]
    for k in names:
        p[k] = p[k].detach().clone().requires_grad_(True)
    out = c.run(p=p)
    assert float(out['evap_bare'].sum()) > 0.0
    loss = (out['et'].sum() + out['runoff'].sum() + out['baseflow'].sum()
            + out['soil_moisture'][-1].sum())
    loss.backward()
    bad = [k for k in names if p[k].grad is not None
           and not bool(torch.isfinite(p[k].grad).all())]
    assert not bad, bad
    assert float(p['lai_scale'].grad.abs().max()) > 0.0


def test_gradients_through_a_leafless_tile_energy_balance_are_finite(
        tmp_path):
    """Five July days of Walnut Gulch 3-hourly energy balance under 5.1.0
    with class 9 leafless: the bare soil's stability-corrected resistance
    sits inside the surface temperature's implicit solve, and every
    parameter gradient through it must be finite."""
    if not os.path.exists(WG_FULL):
        pytest.skip('the Walnut Gulch global file is not present')
    pytest.importorskip('xarray')
    from torchvic import case
    pf, _ = _leafless_params(str(tmp_path), month=6)
    lines = []
    for ln in open(WG_FULL).read().splitlines():
        key = ln.split()[0] if ln.split() else ''
        if key == 'PARAMETERS':
            ln = 'PARAMETERS %s' % pf
        elif key == 'STARTMONTH':
            ln = 'STARTMONTH 07'
        lines.append(ln)
    gf = os.path.join(str(tmp_path), 'global.txt')
    open(gf, 'w').write('\n'.join(lines) + '\n')
    # the global file's STARTMONTH 07 starts the run (case.window)
    c = case.load(gf, device=CPU, steps=8 * 5)
    assert c.full_energy
    assert float(c.mosaic.fcanopy[:, :, 6].min()) == 0.0
    p = c.params()
    names = [k for k, v in p.items()
             if torch.is_tensor(v) and v.is_floating_point()]
    for k in names:
        p[k] = p[k].detach().clone().requires_grad_(True)
    out = c.run(p=p)
    assert float(out['evap_bare'].sum()) > 0.0
    loss = (out['et'].sum() + out['runoff'].sum() + out['baseflow'].sum()
            + out['soil_moisture'][-1].sum())
    loss.backward()
    bad = [k for k in names if p[k].grad is not None
           and not bool(torch.isfinite(p[k].grad).all())]
    assert not bad, bad
    assert float(p['lai_scale'].grad.abs().max()) > 0.0
