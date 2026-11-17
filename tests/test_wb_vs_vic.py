"""VIC 5.0.1's water-balance mode (FULL_ENERGY FALSE), pinned against VIC.

The water-balance path used to be VIC's daily water balance to +0.08% on
annual ET and a different model at every sub-daily step: +3.6% ET on Walnut
Gulch 3-hourly and +7.5% transpiration, -11.7% canopy evaporation on India
basin 37 hourly. Four things were missing or wrong against the C:

  * VIC still runs the QUICK_FLUX ground heat flux with Tsurf = Tair
    (calc_surf_energy_bal.c:542-552) and adds grnd_flux + deltaH to the
    NetBareRad that drives canopy_evap and transpiration
    (func_surf_energy_bal.c:398-429, :529-531, :663-664);
  * WDEW_FLOOR made an empty canopy wet, (1e-4)^(2/3) = 0.00215, where
    canopy_evap.c:99 gives exactly 0 -- on a daily step that ate small
    rains whole (cell 39, basin 37: 9.23 mm off the canopy on a day VIC's
    whole ET was 0.658 mm);
  * overstory tiles took the no-overstory resistance, not Ra_veg[1];
  * no wilting-point cap on transpiration (canopy_evap.c:500-524), no
    fcanopy, and evaporation on tiles under snow.

Measured after the fix against pristine vic_image.exe (annual basin totals,
TorchVIC minus VIC over VIC): basin 37 2001 daily ET +0.0006%, 3-hourly
+0.0012%, hourly +0.0008%, transpiration, canopy evaporation, runoff and
baseflow all within 0.005%; Walnut Gulch 2008 3-hourly every flux within
0.0001%.

The first tests pin the physics properties on their own; the last two run
pristine VIC 5.0.1 (TORCHVIC_REFERENCE_DIR) on
Walnut Gulch and compare, and skip where it is not built.
"""
import glob
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import _paths                                           # noqa: E402
from torchvic.core import energy, vegetation  # noqa: E402
from torchvic.core.vegetation import VegMosaic  # noqa: E402

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')

D = torch.float64
CPU = torch.device('cpu')
REF = _paths.REFERENCE
VIC_IMAGE = _paths.VIC501_IMAGE
WG_FULL = _paths.WG_FULL
WG_DOMAIN = _paths.WG_DOMAIN


def _open_history(files):
    """VIC's history files, one dataset concatenated along time, loaded
    (xarray.open_mfdataset without needing dask)."""
    import xarray as xr
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            parts.append(ds.load())
    return xr.combine_by_coords(parts, combine_attrs='override')


@pytest.fixture(autouse=True)
def _on_cpu():
    # Ten standalone scripts in this directory call torch.set_default_device
    # at import, and pytest imports them all when it collects, so in the full
    # suite the process default may be CUDA; torch's own factory functions
    # inside VegMosaic then mixed devices with these CPU tensors (four
    # failures in the first full run, none alone). test_soil_knobs.py pins
    # the default the same way.
    was = torch.empty(0).device
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(was)


def T(v, *shape):
    return torch.full(shape, float(v), dtype=D, device=CPU)


# ---------------------------------------------------------------------------
# Physics properties
# ---------------------------------------------------------------------------
def test_an_empty_canopy_is_dry_and_its_gradient_is_finite():
    """canopy_evap.c:99, (Wdew / Wdmax)^(2/3): exactly 0 at Wdew = 0 and the
    true power for any positive store however small -- a canopy dried by
    `f < 1` leaves a residue of a few 1e-18 mm, and flooring that brings the
    wet empty canopy back as surely as flooring an exact zero. The
    derivative at the empty store is where d/dx x^(2/3) diverges; it has to
    come back finite ('torch.where is not an if')."""
    held = torch.tensor([0.0, 1e-30, 3e-18, 1e-8, 2e-5, 0.05, 0.3, 0.4],
                        dtype=D, device=CPU, requires_grad=True)
    wdmax = torch.full_like(held, 0.3).requires_grad_(True)
    wet = vegetation.canopy_wetness(held, wdmax, exact=True)
    assert wet[0].item() == 0.0
    ratio = (held / wdmax).detach().clamp(max=1.0)
    want = ratio ** vegetation.INTERCEPT_EXP
    assert torch.equal(wet.detach(), want), (wet, want)
    wet.sum().backward()
    assert torch.isfinite(held.grad).all(), held.grad
    assert torch.isfinite(wdmax.grad).all(), wdmax.grad
    # Above the floor the derivative is the true one.
    i = 5
    true = (2.0 / 3.0) * (held[i] / wdmax[i]).item() ** (-1.0 / 3.0) \
        / wdmax[i].item()
    assert abs(held.grad[i].item() - true) < 1e-12 * abs(true)
    # The floored value the energy path keeps is what made the empty canopy
    # wet: 1e-4 ** (2/3) = 0.00215.
    floored = vegetation.canopy_wetness(held.detach(), wdmax.detach(),
                                        exact=False)
    assert abs(floored[0].item() - 1e-4 ** (2.0 / 3.0)) < 1e-15


def test_a_daily_step_does_not_evaporate_rain_off_an_empty_canopy():
    """On a daily step VIC lets the day's rain evaporate as it lands, but
    the demand is (Wdew / Wdmax)^(2/3) of the wet-canopy Penman with Wdew
    the store *before* the rain (canopy_evap.c:99-111). An empty canopy
    therefore evaporates nothing that day and ends it full; the floored
    wetness evaporated 0.00215 of a Penman of several mm, i.e. the whole of
    a small rain, and flipped the next day's store."""
    n = 3
    lwf = torch.tensor(0.1, dtype=D, device=CPU, requires_grad=True)
    wdew = T(0.0, n).requires_grad_(True)
    rain = torch.tensor([0.004, 0.2, 5.0], dtype=D, device=CPU)
    args = dict(lai=T(3.0, n), rgl=T(100.0, n), net_short=T(250.0, n),
                tair=T(28.0, n), vpd=T(2500.0, n), ra=T(40.0, n),
                rarc=T(60.0, n), rad=T(180.0, n), elevation=T(300.0, n))
    ce, new_wdew, tf, dry = vegetation.canopy_interception(
        wdew, rain, dt=vegetation.CONST_CDAY, lai_water_factor=lwf,
        exact_wetness=True, **args)
    assert torch.equal(ce, torch.zeros_like(ce))
    assert torch.equal(dry, torch.ones_like(dry))
    wdmax = 0.1 * 3.0
    assert torch.allclose(new_wdew, torch.clamp(rain, max=wdmax), rtol=0,
                          atol=1e-15)
    assert torch.allclose(tf, torch.clamp(rain - wdmax, min=0.0), rtol=0,
                          atol=1e-15)
    (ce.sum() + new_wdew.sum() + tf.sum() + dry.sum()).backward()
    assert torch.isfinite(wdew.grad).all() and torch.isfinite(lwf.grad)
    # The floor would have taken all of the small rain.
    ce_f, _, _, _ = vegetation.canopy_interception(
        wdew.detach(), rain, dt=vegetation.CONST_CDAY, lai_water_factor=0.1,
        exact_wetness=False, **args)
    assert ce_f[0].item() == pytest.approx(0.004, abs=1e-15)


def _one_tile_mosaic(n_tiles=1, fcanopy=None):
    nt = n_tiles
    return VegMosaic(cv=T(1.0 / nt, 1, nt), lai=T(3.0, 1, nt, 12),
                     rmin=T(100.0, 1, nt), rgl=T(100.0, 1, nt),
                     rarc=T(60.0, 1, nt),
                     root_frac=torch.tensor([[[0.2, 0.2, 0.6]] * nt],
                                            dtype=D, device=CPU),
                     dtype=D, device=CPU, fcanopy=fcanopy)


def _step(mos, avail, wcr, wpwp, **kw):
    n = 1
    rain = kw.pop('rain', T(0.0, n))
    wdew = kw.pop('wdew', T(0.0, n, mos.cv.shape[1]))
    return mos.step(0, wdew, rain, avail, wcr, wpwp,
                    T(300.0, n), T(30.0, n), T(3000.0, n), T(30.0, n),
                    T(250.0, n), T(300.0, n), dt=vegetation.CONST_CDAY,
                    per_tile=True, **kw)


def test_transpiration_stops_at_the_wilting_point():
    """canopy_evap.c:500-524: after the CASE 1 / CASE 2 split no layer gives
    up water below its wilting point, and a negative result is zero. CASE 1
    hands a dry layer's shortfall to the wet ones, so a thin wet layer --
    here 0.2 mm above Wpwp -- is asked for more than it holds. Without the
    cap the column's `evap_withheld` quietly absorbed it: 835
    tile-layer-days, 0.156% of the transpiration, basin 37, May-October
    2002, daily."""
    mos = _one_tile_mosaic()
    # Layer 3 holds 0.6 of the roots and sits at Wcr, so CASE 1 applies
    # (canopy_evap.c:290-292); layer 2 is below Wcr and hands its shortfall
    # to layers 1 and 3; layer 1 is at Wcr with 0.2 mm above Wpwp.
    avail = torch.tensor([[10.2, 30.0, 20.3]], dtype=D, device=CPU,
                         requires_grad=True)
    wcr = torch.tensor([[10.2, 90.0, 20.3]], dtype=D, device=CPU)
    wpwp = torch.tensor([[10.0, 15.0, 20.0]], dtype=D, device=CPU,
                        requires_grad=True)
    free = _step(mos, avail, wcr, wpwp)[0]
    capped = _step(mos, avail, wcr, wpwp, wilting_clamp=True)[0]
    room = (avail - wpwp).unsqueeze(1)
    assert float(free[0, 0, 0]) > float(room[0, 0, 0]) + 0.1, free
    assert float(free[0, 0, 2]) > float(room[0, 0, 2]) + 0.1, free
    assert torch.equal(capped[..., 0], room[..., 0])
    assert torch.equal(capped[..., 2], room[..., 2])
    assert torch.equal(capped[..., 1], free[..., 1])
    capped.sum().backward()
    assert torch.isfinite(avail.grad).all() and torch.isfinite(wpwp.grad).all()
    # A layer already below its wilting point gives nothing, not a negative.
    below = torch.tensor([[9.9, 30.0, 20.3]], dtype=D, device=CPU)
    out = _step(mos, below, wcr, wpwp.detach(), wilting_clamp=True)[0]
    assert float(out[0, 0, 0]) == 0.0


def test_a_tile_under_snow_neither_evaporates_nor_transpires():
    """func_surf_energy_bal.c:756 and :798 gate canopy_evap and arno_evap on
    !SNOWING, and SNOWING is solve_snow.c:164-171's snow->snow: a pack, snow
    falling, or snow in an overstory. The tile's canopy store is left as it
    was and what reached it goes to the soil as it is
    (calc_surf_energy_bal.c:605-612 skips the throughfall assignment)."""
    mos = _one_tile_mosaic(n_tiles=2)
    avail = torch.tensor([[40.0, 60.0, 200.0]], dtype=D, device=CPU)
    wcr = torch.tensor([[30.0, 45.0, 120.0]], dtype=D, device=CPU)
    wpwp = torch.tensor([[15.0, 20.0, 60.0]], dtype=D, device=CPU)
    wdew = torch.tensor([[0.1, 0.1]], dtype=D, device=CPU)
    rain = torch.tensor([[2.0, 2.0]], dtype=D, device=CPU)
    snowing = torch.tensor([[True, False]], device=CPU)
    tr, ce, tf, nw = _step(mos, avail, wcr, wpwp, rain=rain, wdew=wdew,
                           snowing=snowing)
    assert torch.equal(tr[0, 0], torch.zeros(3, dtype=D, device=CPU))
    assert float(ce[0, 0]) == 0.0 and float(tf[0, 0]) == 2.0
    assert float(nw[0, 0]) == 0.1
    assert float(tr[0, 1].sum()) > 0.0 and float(ce[0, 1]) > 0.0


def test_a_whole_canopy_fcanopy_is_the_identity_and_a_partial_one_conserves():
    """vic_run.c:136-140 runs the canopy per plant, LAI and Wdew divided by
    fcanopy, and func_surf_energy_bal.c:762-791 scales every canopy output
    back and lets the bare fraction's rain through. With fcanopy exactly 1
    every one of those operations is an identity in floating point, so a
    FROM_DEFAULT run cannot move by a bit; below 1 the tile's canopy still
    closes: rain = throughfall + canopy evaporation + change in store."""
    avail = torch.tensor([[40.0, 60.0, 200.0]], dtype=D, device=CPU)
    wcr = torch.tensor([[30.0, 45.0, 120.0]], dtype=D, device=CPU)
    wpwp = torch.tensor([[15.0, 20.0, 60.0]], dtype=D, device=CPU)
    wdew = torch.tensor([[0.05]], dtype=D, device=CPU)
    rain = torch.tensor([[1.5]], dtype=D, device=CPU)
    none = _step(_one_tile_mosaic(), avail, wcr, wpwp, rain=rain, wdew=wdew)
    ones = _step(_one_tile_mosaic(fcanopy=T(1.0, 1, 1, 12)), avail, wcr,
                 wpwp, rain=rain, wdew=wdew)
    for a, b in zip(none, ones):
        assert torch.equal(a, b)
    part = _step(_one_tile_mosaic(fcanopy=T(0.4, 1, 1, 12)), avail, wcr,
                 wpwp, rain=rain, wdew=wdew)
    tr, ce, tf, nw = part
    assert abs(float(rain - tf - ce - (nw - wdew))) < 1e-14
    # And it is a different canopy: the plants' LAI is 3 / 0.4, so their
    # stomata and their wet fraction are not the tile's.
    assert float((tr - ones[0]).abs().max()) > 1e-3


def test_the_quick_flux_at_tair_vanishes_in_a_steady_profile():
    """energy.water_balance_ground_flux is estimate_T1 plus the ground flux
    and heat-storage lines of func_surf_energy_bal.c (:405, :419-429,
    :530-531) at TMean = Tair. A profile at one temperature everywhere
    carries no flux and stores nothing; air warmer than the soil drives heat
    down, which VIC's sign takes out of NetBareRad (grnd_flux < 0), and the
    storage term then says the top layer warmed (deltaH < 0)."""
    kw = dict(d1=T(0.1, 2), kappa1=T(1.2, 2), kappa2=T(1.1, 2),
              cs1=T(2.0e6, 2), cs2=T(2.1e6, 2), dp=T(4.0, 2),
              dt=3 * 3600.0, surf_atten=T(1.0, 2))
    g, dh, t1 = energy.water_balance_ground_flux(
        T(15.0, 2), T(15.0, 2), T(15.0, 2), T(15.0, 2), **kw)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-12)
    assert torch.allclose(dh, torch.zeros_like(dh), atol=1e-12)
    assert torch.allclose(t1, T(15.0, 2), atol=1e-12)
    g, dh, t1 = energy.water_balance_ground_flux(
        T(25.0, 2), T(15.0, 2), T(15.0, 2), T(15.0, 2), **kw)
    assert bool((g < 0).all()) and bool((dh < 0).all())
    assert bool((t1 > 15.0).all()) and bool((t1 < 25.0).all())


# ---------------------------------------------------------------------------
# Against pristine VIC 5.0.1 on Walnut Gulch
# ---------------------------------------------------------------------------
PAIRS = (('OUT_EVAP', 'et'), ('OUT_TRANSP_VEG', 'transp'),
         ('OUT_EVAP_CANOP', 'evap_canop'), ('OUT_RUNOFF', 'runoff'),
         ('OUT_BASEFLOW', 'baseflow'), ('OUT_SUB_SNOW', 'sub_snow'))


def _need_vic():
    if not (os.path.exists(VIC_IMAGE) and os.path.exists(WG_FULL)
            and os.path.exists(WG_DOMAIN)):
        pytest.skip('pristine VIC 5.0.1 (TORCHVIC_REFERENCE_DIR) or '
                    'the Walnut Gulch sample data is not present')
    pytest.importorskip('xarray')


def _run_vic(tmp, subs):
    """VIC 5.0.1 on the Walnut Gulch global file with `subs` applied; returns
    its daily history (AGGFREQ NDAYS 1) as an xarray Dataset."""
    import xarray as xr
    g = open(WG_FULL).read()
    res, logs = os.path.join(tmp, 'results'), os.path.join(tmp, 'logs')
    os.makedirs(res), os.makedirs(logs)
    lines = []
    for ln in g.splitlines():
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
    r = subprocess.run([VIC_IMAGE, '-g', gf], env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    files = sorted(glob.glob(os.path.join(res, '*.nc')))
    assert files, 'VIC wrote no history'
    return gf, _open_history(files)


def _basin_daily(vic, ours, spd):
    """Area-weighted basin means per day, VIC's and ours, for PAIRS."""
    import xarray as xr
    dom = xr.open_dataset(WG_DOMAIN)
    mask = dom['mask'].values.astype(bool)
    w = dom['area'].values[mask]
    w = w / w.sum()
    out = {}
    for vk, ok in PAIRS:
        a = vic[vk].values[:, mask]
        b = ours[ok].detach().cpu().numpy()
        b = b.reshape(b.shape[0] // spd, spd, b.shape[1]).sum(1)
        out[vk] = ((a * w).sum(-1), (b * w).sum(-1))
    return out


def _compare(series, label, tol_rel=1e-3):
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


def test_walnut_gulch_three_hourly_water_balance_is_vics(tmp_path):
    """Walnut Gulch, January-March 2008, 3-hourly, FULL_ENERGY FALSE: the
    configuration with thin packs (snow_melt.c:322-324 hands them to the
    soil surface, which water balance evaluates at Tsurf = Tair) and snow
    gating the canopy. Before the fix ET was +3.6% and transpiration +5.5%
    over the year; after it, every annual flux within 0.0001%."""
    _need_vic()
    from torchvic import case
    gf, vic = _run_vic(str(tmp_path), {'FULL_ENERGY': 'FALSE',
                                        'ENDMONTH': '03', 'ENDDAY': '31'})
    c = case.load(gf, device=CPU, steps=8 * 91)
    assert not c.full_energy and c.model_steps == 8
    with torch.no_grad():
        out = c.run(p=c.params())
    _compare(_basin_daily(vic, out, 8), 'Walnut Gulch Jan-Mar 2008, 3-hourly '
             'water balance')


def _on_the_file_day(monkeypatch):
    """Bin a daily step on the forcing file's own day, as vic_image.exe
    does: vic_force.c:89-90 reads the step's records by their index in the
    file (forceskip + forceoffset + j), with no off_gmt shift. case.load
    bins on the cell's local day (case.daily_by_off_gmt), which on Walnut
    Gulch (off_gmt -7 h) is seven hours off VIC's day and gave 93 partial
    days for VIC's 92."""
    from torchvic import case
    from torchvic.io.forcing import aggregate_to_daily
    monkeypatch.setattr(
        case, 'daily_by_off_gmt',
        lambda fields, times, off_gmt, log=None: aggregate_to_daily(
            fields, times, utc_offset=0.0, log=log))


def test_walnut_gulch_daily_water_balance_is_vics_without_snow(tmp_path,
                                                                 monkeypatch):
    """Walnut Gulch, June-August 2008, MODEL_STEPS_PER_DAY 1: VIC's daily
    water balance, which averages the eight 3-hourly records itself
    (vic_force.c:404-412) and lets the day's rain evaporate off the canopy
    (canopy_evap.c:108-110) -- where the floored wetness made a dry canopy
    evaporate.     Summer has no snow, so the snow sub-step inside a daily step
    (surface_fluxes.c:258-269) does not enter; the day stays the daily mean."""
    _need_vic()
    from torchvic import case
    gf, vic = _run_vic(str(tmp_path), {'FULL_ENERGY': 'FALSE',
                                        'MODEL_STEPS_PER_DAY': '1',
                                        'STARTMONTH': '06', 'STARTDAY': '01',
                                        'ENDMONTH': '08', 'ENDDAY': '31'})
    first = 0                # the global file's STARTMONTH 06 starts it
    _on_the_file_day(monkeypatch)
    c = case.load(gf, device=CPU, daily=True, first_step=first, steps=8 * 92)
    assert c.model_steps == 1 and c.n_steps == 92
    assert str(c.dates[0])[:10] == '2008-06-01'
    with torch.no_grad():
        out = c.run(p=c.params())
    _compare(_basin_daily(vic, out, 1), 'Walnut Gulch Jun-Aug 2008, daily '
             'water balance')


def test_gradients_through_the_water_balance_with_snow_are_finite():
    """The new terms -- exact wetness, the wilting cap, the quick flux, the
    thin-pack balance and the snow gate -- all sit behind torch.where or
    clamp; every parameter gradient through 30 days of Walnut Gulch
    3-hourly water balance with January's snow has to be finite."""
    if not os.path.exists(WG_FULL):
        pytest.skip('the Walnut Gulch global file is not present')
    from torchvic import case
    c = case.load(WG_FULL, device=CPU, no_energy=True, steps=8 * 30)
    p = c.params()
    names = ('pcf', 'lai_scale', 'rmin_scale', 'lai_water_factor',
             'wcr_scale', 'wpwp_scale', 't_max_snow', 'top_depth_scale')
    for k in names:
        p[k] = p[k].clone().requires_grad_(True)
    out = c.run(p=p)
    loss = (out['et'].sum() + out['runoff'].sum() + out['baseflow'].sum()
            + out['soil_moisture'][-1].sum())
    loss.backward()
    for k in names:
        assert p[k].grad is not None and torch.isfinite(p[k].grad), k
    assert float(p['lai_water_factor'].grad.abs()) > 0.0


# ---------------------------------------------------------------------------
# FCAN_SRC FROM_DEFAULT and the last (bare-soil) class
# ---------------------------------------------------------------------------
def _bare_params(tmp, count_it):
    """Walnut Gulch's parameters with 30% of each cell's largest class moved
    to the last class, the file's bare soil (LAI 0). With `count_it` false
    the file's Nveg is left as it was, one below the covered classes, and
    VIC runs that class as its bare-soil tile (iveg == Nveg, VEG false,
    vic_alloc.c:128); with it true Nveg counts it (roots copied from the
    largest class, which calc_root_fraction.c:163 then requires) and it runs
    as vegetation with FROM_DEFAULT's fcanopy, MIN_FCANOPY."""
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
                         ids=['bare_tile', 'last_class_as_vegetation'])
def test_from_default_evaporates_the_last_class_as_bare_soil(
        tmp_path, monkeypatch, count_it):
    """vic_init.c:262-273: FCAN_SRC FROM_DEFAULT gives every class fcanopy 1
    except the last, which it takes to be bare soil and gives MIN_FCANOPY,
    so VIC evaporates it (almost) wholly as ARNO bare soil
    (func_surf_energy_bal.c:762-805). This model read 1 for every class and
    evaporated none of it: on these parameters, June-August 2008 daily,
    OUT_EVAP_BARE 0 against VIC's 31.54 mm, ET -11.3%, baseflow +265%.
    Both of VIC's readings of the class are pinned -- its bare-soil tile
    (Nveg one short) and vegetation at MIN_FCANOPY (Nveg counting it)."""
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
    assert c.model_steps == 1 and c.n_steps == 92
    fc = c.mosaic.fcanopy
    assert fc is not None
    assert torch.equal(fc[:, -1], torch.full_like(fc[:, -1], 1e-4))
    assert torch.equal(fc[:, :-1], torch.ones_like(fc[:, :-1]))
    assert (c.mosaic.bare_tile is not None) == (not count_it)
    with torch.no_grad():
        out = c.run(p=c.params())
    series = _basin_daily(vic, out, 1)
    a, b = _basin_daily_one(vic, out, 1, 'OUT_EVAP_BARE', 'evap_bare')
    series['OUT_EVAP_BARE'] = (a, b)
    assert float(a.sum()) > 30.0
    _compare(series, 'Walnut Gulch Jun-Aug 2008 daily, 30%% of cover in '
             'the last class, Nveg %s' % ('counting it' if count_it
                                          else 'one short'), tol_rel=1e-4)


def _basin_daily_one(vic, ours, spd, vk, ok):
    import xarray as xr
    dom = xr.open_dataset(WG_DOMAIN)
    mask = dom['mask'].values.astype(bool)
    w = dom['area'].values[mask]
    w = w / w.sum()
    a = vic[vk].values[:, mask]
    b = ours[ok].detach().cpu().numpy()
    b = b.reshape(b.shape[0] // spd, spd, b.shape[1]).sum(1)
    return (a * w).sum(-1), (b * w).sum(-1)


# ---------------------------------------------------------------------------
# Mass closure of the water-balance path where its new branches run
# ---------------------------------------------------------------------------
def _closure(c, out):
    """Per cell-step, the model's own put_data.c:531-544 identity; over the
    run, an independent one from the stores themselves: precipitation that
    entered, less ET, runoff and baseflow, less the change from the opening
    soil moisture (init_moist capped at capacity, as simulate starts) to the
    stores the run ends with."""
    per_step = float(out['water_error'].abs().max())
    sm0 = torch.minimum(c.init_moist.expand_as(c.soil.max_moist),
                        c.soil.max_moist).sum(-1)
    end = (out['soil_moisture'].sum(-1) + out['swe'][-1]
           + out['wdew'][-1]
           + (out['snow_canopy'][-1] if 'snow_canopy' in out else 0.0))
    inflow = (out['rainf'] + out['snowf']).sum(0)
    outflow = (out['et'] + out['runoff'] + out['baseflow']).sum(0)
    run = float((inflow - outflow - (end - sm0)).abs().max())
    return per_step, run


def test_water_balance_closes_through_walnut_gulch_snow_three_hourly():
    """January-February 2008, 3-hourly: the thin packs handed to the soil
    surface (_wb_thin_pack), the snow gate, the canopy dump under snow and
    VIC's 1e-5 mm snowfall cut all run here; the conservation suite's own
    water-balance run stops before the first snow (day 25). Measured 1e-13
    mm per cell-step."""
    if not os.path.exists(WG_FULL):
        pytest.skip('the Walnut Gulch global file is not present')
    from torchvic import case
    c = case.load(WG_FULL, device=CPU, no_energy=True, steps=8 * 60)
    with torch.no_grad():
        out = c.run(p=c.params())
    assert float(out['sub_snow'].sum()) > 0.0
    assert float(out['swe'].max()) > 0.0      # thin packs: 0.005 mm at most
    per_step, run = _closure(c, out)
    print('Walnut Gulch Jan-Feb 3-hourly: water_error %.2e mm, whole-run '
          '%.2e mm' % (per_step, run))
    assert per_step < 1e-9 and run < 1e-9, (per_step, run)


BASIN37 = _paths.B37_GLOBAL


def test_water_balance_closes_with_fcan_from_vegparam():
    """Basin 37, July 2000, daily, FCAN_SRC FROM_VEGPARAM: the fractional
    canopy's per-plant canopy and the bare soil between plants
    (func_surf_energy_bal.c:762-791), 216 mm a year of bare-soil evaporation
    against VIC's. Closed per cell-step and over the run."""
    if not os.path.exists(BASIN37):
        pytest.skip('India_basin37_Nandgaon is not built')
    from torchvic import case
    from torchvic.io import build as image_run
    c = case.load(BASIN37, device=CPU, no_energy=True, daily=True,
                  first_step=182, steps=31)
    c.mosaic = image_run.build_mosaic(c.par, c.dtype, c.device,
                                      log=lambda *a: None,
                                      fcan_src='FROM_VEGPARAM')
    assert c.mosaic.fcanopy is not None
    assert float(c.mosaic.fcanopy.min()) < 1.0
    with torch.no_grad():
        out = c.run(p=c.params())
    assert float(out['evap_bare'].sum()) > 0.0
    per_step, run = _closure(c, out)
    print('basin 37 July 2000 daily FROM_VEGPARAM: water_error %.2e mm, '
          'whole-run %.2e mm' % (per_step, run))
    assert per_step < 1e-9 and run < 1e-9, (per_step, run)


# ---------------------------------------------------------------------------
# Stehekin: VIC's own water-balance sample, an overstory under snow
# ---------------------------------------------------------------------------
STEH_GLOBAL = os.path.join(_paths.SAMPLES, 'image', 'Stehekin',
                           'parameters', 'Stehekin_image_test.global.txt')
STEH_DOMAIN = os.path.join(_paths.SAMPLES, 'image', 'Stehekin',
                           'parameters', 'domain.stehekin.20151028.nc')


def _stehekin_global(tmp):
    """The sample's global file, SNOW_BAND FALSE (no band file is carried),
    with its paths resolved and the water fluxes compared written out."""
    sd = _paths.SAMPLES
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
    # Declared, because case.load now refuses to guess a wind height where
    # snow is possible (case.resolve_wind_h). 10 m is what this comparison
    # was measured with; VIC's image driver ignores the global WIND_H for
    # the land surface (vic_run.c:219, 259-264), so VIC's side is unchanged.
    lines.append('WIND_H 10.0')
    for v in ('OUT_TRANSP_VEG', 'OUT_EVAP_CANOP', 'OUT_SUB_SNOW',
              'OUT_SUB_CANOP', 'OUT_SNOW_CANOPY'):
        lines.append('OUTVAR %s' % v)
    gf = os.path.join(tmp, 'global.txt')
    open(gf, 'w').write('\n'.join(lines) + '\n')
    return gf, res


def test_stehekin_water_balance_against_vic(tmp_path):
    """Stehekin, January 1949, hourly, FULL_ENERGY FALSE, about 80%
    overstory under snow. Ten-day basin totals against vic_image.exe:
    ET, snow and canopy sublimation, canopy evaporation and SWE within
    0.005%, baseflow 0.0000%. The canopy air temperature is the air's in
    VIC's image driver (CLOSE_ENERGY is never read, calc_atmos_energy_bal.c
    5.1.0:73-103); the gap was elsewhere (see the bounds below)."""
    if not (os.path.exists(VIC_IMAGE) and os.path.exists(STEH_GLOBAL)):
        pytest.skip('pristine VIC 5.0.1 or the Stehekin sample is not present')
    xr = pytest.importorskip('xarray')
    from torchvic import case
    gf, res = _stehekin_global(str(tmp_path))
    env = dict(os.environ, OMPI_MCA_btl_vader_single_copy_mechanism='none')
    r = subprocess.run([VIC_IMAGE, '-g', gf], env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    vic = _open_history(sorted(glob.glob(os.path.join(res, '*.nc'))))
    c = case.load(gf, device=CPU)
    assert not c.full_energy and c.model_steps == 24 and c.n_steps == 240
    # VIC runs SNOW_BAND FALSE at the cell's own elevation.
    c.batch.elevation = torch.as_tensor(np.asarray(c.par.soil['elev']),
                                        dtype=c.dtype, device=c.device)
    with torch.no_grad():
        out = c.run(p=c.params())
    dom = xr.open_dataset(STEH_DOMAIN)
    # the domain file's mask has 20 cells, the parameter file's run 16
    mask = (dom['mask'].values.astype(bool)
            & np.isfinite(vic['OUT_EVAP'].values[0]))
    assert int(mask.sum()) == 16
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


def test_gradients_with_an_overstory_under_snow_are_finite():
    """Stehekin, 24 hourly water-balance steps: the overstory profile of a
    tile with no overstory (height 0) is NaN in CalcAerodynamic's canopy
    terms, and canopy_step, which runs on every tile and masks afterwards,
    sent 0 * NaN into Tfoliage's gradient: pcf, new_snow_albedo and
    lai_scale came back NaN here (and t_max_snow, t_min_rain and
    lai_water_factor as well over 240 steps)."""
    if not os.path.exists(STEH_GLOBAL):
        pytest.skip('the Stehekin sample is not present')
    import tempfile
    from torchvic import case
    with tempfile.TemporaryDirectory() as tmp:
        gf, _ = _stehekin_global(tmp)
        c = case.load(gf, device=CPU, steps=24)
    assert not c.full_energy
    p = c.params()
    names = [k for k, v in p.items()
             if torch.is_tensor(v) and v.is_floating_point()]
    for k in names:
        p[k] = p[k].detach().clone().requires_grad_(True)
    out = c.run(p=p)
    assert float(out['snow_canopy'].max()) > 0.0
    loss = (out['et'].sum() + out['runoff'].sum() + out['baseflow'].sum()
            + out['swe'].sum() + out['sub_snow'].sum())
    loss.backward()
    bad = [k for k in names if p[k].grad is not None
           and not bool(torch.isfinite(p[k].grad).all())]
    assert not bad, bad
    for k in ('pcf', 'new_snow_albedo', 'lai_scale'):
        assert float(p[k].grad.abs().max()) > 0.0, k


def test_a_trace_snowfall_is_not_snow():
    """solve_snow.c:139-141 sets a snowfall under 1e-5 mm to zero before
    the snow->snow test at :164, so a trace neither gates the step's
    evaporation nor enters the water balance (out_prec is rain plus what is
    left). India's daily cache, 2000-2009: 25,745 such cell-days, each of
    which zeroed a whole day's ET here. A 5e-6 mm snowfall on a snow-free
    Walnut Gulch step must be the dry step to the bit; 2e-5 mm is snow and
    gates every tile's transpiration and canopy evaporation."""
    if not os.path.exists(WG_FULL):
        pytest.skip('the Walnut Gulch global file is not present')
    from torchvic import case
    c = case.load(WG_FULL, device=CPU, no_energy=True, steps=8,
                  first_step=8 * 180)
    t = 4                                  # 12 UTC, midday, evaporating
    c.atm['tair'][t] = torch.full_like(c.atm['tair'][t], -5.0)
    runs = {}
    for mm in (0.0, 5e-6, 2e-5):
        c.atm['prcp'][t] = torch.full_like(c.atm['prcp'][t], mm)
        with torch.no_grad():
            runs[mm] = c.run(p=c.params())
    for k in ('et', 'transp', 'evap_canop', 'runoff', 'baseflow', 'swe',
              'water_error'):
        assert torch.equal(runs[5e-6][k], runs[0.0][k]), k
    assert float(runs[5e-6]['snowf'].abs().max()) == 0.0
    assert float(runs[0.0]['transp'][t].min()) > 0.0
    assert float(runs[2e-5]['snowf'][t].min()) > 0.0
    # snow: no transpiration and no canopy evaporation on any tile (the new
    # pack's own vapour flux is snow sublimation, not gated)
    assert float(runs[2e-5]['transp'][t].abs().max()) == 0.0
    assert float(runs[2e-5]['evap_canop'][t].abs().max()) == 0.0
