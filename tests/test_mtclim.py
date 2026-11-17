"""torchvic.forcing.mtclim against an independent reading of MetSim v2.0.1.

The module re-implements MetSim's MTCLIM (metsim/physics.py,
metsim/methods/mtclim.py, metsim/disaggregate.py; UW-Hydro, GPL-3.0) as
batched tensor code. This file holds a second implementation of the same
algorithms in plain numpy, written one cell and one day at a time the way
MetSim writes them, and checks the tensor code against it. MetSim itself is
not imported or run: it is third-party code kept as untrusted reading
material (a checkout of MetSim v2.0.1) and needs numba, netCDF4 and dask.
The PCHIP reference is scipy's, which is what MetSim calls.

The last test measures, over ten sites from 44 N to 76 N, how far this
algorithm is from VIC 4.2.d's compiled MTCLIM (tests/oracle/ref_mtclim.c),
which an earlier version of the module reproduced exactly. Its bounds sit
just above the measured differences; they are not claims of agreement.
"""
import pathlib
import subprocess

import numpy as np
import pytest
import torch

from torchvic.forcing import mtclim

DT = torch.float64
#: Ten standalone scripts in this directory call torch.set_default_device at
#: module scope and pytest imports every test_*.py when it collects, so by
#: the time these run the process default may already be CUDA. Take the live
#: default rather than fight it.
DEV = torch.empty(0).device

import _paths                                           # noqa: E402
REF = pathlib.Path(_paths.oracle('ref_mtclim'))

SEEDS = [0, 1, 2, 3, 4, 5, 7, 11, 13, 19]


# ---------------------------------------------------------------------------
# The numpy reference, from MetSim v2.0.1's source
# ---------------------------------------------------------------------------
C = mtclim
OPTAM = list(mtclim.OPTAM)


def r_svp(t):
    t = np.asarray(t, dtype=float)
    s = 0.61078 * np.exp(17.269 * t / (237.3 + t))
    s = np.where(t < 0.0, s * (1.0 + 0.00972 * t + 0.000042 * t ** 2), s)
    return s * 1000.0


def r_atm_pres(elev, lr=0.0065):
    t1 = 1.0 - lr * elev / C.T_STD
    t2 = C.G_STD / (lr * (C.R_GAS / C.MA))
    return C.P_STD * t1 ** t2


def r_pet(rad, ta, pa, dayl, dt=0.2):
    rnet = rad * 0.72
    lhvap = 2.5023e6 - 2430.54 * ta
    gamma = C.CP * pa / (lhvap * C.EPS)
    t1, t2 = ta + dt, ta - dt
    s = (r_svp(t1) - r_svp(t2)) / (t1 - t2)
    return (1.26 * (s / (s + gamma)) * rnet * dayl) / lhvap / 10.0


def r_solar_geom(elev, lat_deg, lr=0.0065):
    """metsim.physics.solar_geom, day by day."""
    nd = 366
    tt_max0 = np.zeros(nd)
    daylength = np.zeros(nd)
    flat = np.zeros(nd)
    tiny_n = 2880
    tiny = np.zeros((nd, tiny_n))
    t1 = 1.0 - lr * elev / C.T_STD
    t2 = C.G_STD / (lr * (C.R_GAS / C.MA))
    trans = C.TBASE ** (t1 ** t2)
    lat = min(max(lat_deg * C.RAD_PER_DEG, -np.pi / 2), np.pi / 2)
    coslat, sinlat = np.cos(lat), np.sin(lat)
    dt = C.SW_RAD_DT
    dh = dt / C.SEC_PER_RAD
    for i in range(nd - 1):
        decl = C.MIN_DECL * np.cos((i + C.DAYS_OFF) * C.RAD_PER_DAY)
        cosegeom = coslat * np.cos(decl)
        sinegeom = sinlat * np.sin(decl)
        coshss = min(max(-sinegeom / cosegeom, -1), 1)
        hss = np.arccos(coshss)
        daylength[i] = min(2.0 * hss * C.SEC_PER_RAD, 86400.0)
        dir_beam = (1368.0 + 45.5 * np.sin(2.0 * np.pi * i / 365.25 + 1.7)) * dt
        sum_trans = 0.0
        sum_flat = 0.0
        for h in np.arange(-hss, hss, dh):
            cza = cosegeom * np.cos(h) + sinegeom
            if cza > 0:
                dft = dir_beam * cza
                am = 1.0 / (cza + 0.0000001)
                if am > 2.9:
                    ami = min(max(int(np.arccos(cza) / C.RAD_PER_DEG) - 69, 0),
                              20)
                    am = OPTAM[ami]
                sum_trans += trans ** am * dft
                sum_flat += dft
            else:
                dft = 0.0
            step = int(min(max((12 * 3600 + h * C.SEC_PER_RAD) / dt, 0),
                           tiny_n - 1))
            tiny[i][step] = dft
        if daylength[i] and sum_flat > 0:
            tiny[i] /= sum_flat
        if daylength[i]:
            tt_max0[i] = sum_trans / sum_flat
            flat[i] = sum_flat / daylength[i]
    for a in (tt_max0, flat, daylength, tiny):
        a[nd - 1] = a[nd - 2]
    return tiny, daylength, flat, tt_max0


def r_trailing(x, w):
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def r_mtclim(tmax, tmin, prec, yday, geom, elev, tol=1e-6):
    """metsim.methods.mtclim.run for one cell, the loop as written."""
    _tiny, dayl, flat, tt0 = geom
    ix = np.asarray(yday) - 1
    dayl, potrad, tt_max = dayl[ix], flat[ix], tt0[ix]
    dtr = tmax - tmin
    sm = r_trailing(dtr, 30)
    seasonal = r_trailing(prec, 90) * 365.25
    mean = (tmax + tmin) / 2
    t_day = (tmax - mean) * 0.45 + mean
    b = C.B0 + C.B1 * np.exp(-C.B2 * sm)
    tfmax = 1.0 - 0.9 * np.exp(-b * dtr ** C.C_RAD)
    tfmax = np.where(prec > 0.0, tfmax * 0.75, tfmax)
    pa = r_atm_pres(elev)

    def sw_of(vp):
        return potrad * np.maximum(tt_max + C.ABASE * vp, 0.0001) * tfmax

    def tdew_of(sw):
        pet = r_pet(sw, t_day, pa, dayl) * 10.0
        ratio = pet / np.where(seasonal < 80.0, 80.0, seasonal)
        return ((tmin + C.KELVIN) * (-0.127 + 1.121 * (
            1.003 - 1.444 * ratio + 12.312 * ratio ** 2
            - 32.766 * ratio ** 3) + 0.0006 * dtr) - C.KELVIN)

    old = tmin
    new = tdew_of(sw_of(r_svp(tmin)))
    while np.sqrt(np.mean((new - old) ** 2)) > tol:
        old = new
        new = tdew_of(sw_of(r_svp(old)))
    vp = r_svp(new)
    return dict(tdew=new, vp=vp, sw=sw_of(vp), tskc=1.0 - tfmax)


def r_rise_set_minutes(tiny):
    """metsim.disaggregate.set_min_max_hour's sunrise and sunset, minutes."""
    mask = np.diff(1 * (tiny > 0))
    return np.where(mask > 0)[1] * 0.5, np.where(mask < 0)[1] * 0.5


# ---------------------------------------------------------------------------
def _weather(n, seed, lat):
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    base = 11.0 - (14.0 if abs(lat) > 30 else 6.0) * np.cos(
        2 * np.pi * (i + 10) / 365.0)
    wet = rng.random(n) < 0.3
    tmax = base + 4 + 6 * rng.random(n)
    tmin = base - 4 - 6 * rng.random(n)
    prec = wet * 25.0 * rng.random(n)
    return tmax, tmin, prec


def _t(a):
    return torch.as_tensor(a, dtype=DT, device=DEV)


def test_svp_atm_pres_and_pet_match_the_reference():
    t = np.linspace(-40.0, 40.0, 161)
    assert np.abs(mtclim.svp(_t(t)).cpu().numpy() - r_svp(t)).max() < 1e-9
    el = np.array([0.0, 300.0, 1500.0, 3000.0])
    got = mtclim.atm_pres(_t(el)).cpu().numpy()
    assert np.abs(got - r_atm_pres(el)).max() < 1e-7
    rad = np.array([20.0, 150.0, 400.0])
    ta = np.array([-5.0, 12.0, 28.0])
    got = mtclim.calc_pet(_t(rad), _t(ta), _t(90000.0), _t(43200.0))
    want = r_pet(rad, ta, 90000.0, 43200.0)
    assert np.abs(got.cpu().numpy() - want).max() < 1e-12


@pytest.mark.parametrize('lat,elev', [(47.8, 1300.0), (31.7, 1500.0),
                                      (68.0, 300.0), (-33.0, 50.0)])
def test_solar_geometry_matches_the_reference(lat, elev):
    g = mtclim.solar_geometry(_t(lat), _t(elev), device=DEV)
    tiny, dayl, flat, tt0 = r_solar_geom(elev, lat)
    got = {k: g[k].cpu().numpy() for k in ('daylength', 'flat_potrad',
                                           'ttmax0', 'tiny_radfract')}
    assert np.abs(got['daylength'] - dayl).max() < 1e-6
    assert np.abs(got['flat_potrad'] - flat).max() < 1e-7
    assert np.abs(got['ttmax0'] - tt0).max() < 1e-12
    # A 30-second slot can be claimed by either of two neighbouring hour
    # angles when their position lands on an integer within rounding; allow
    # a handful of such slots over the 366 x 2880.
    bad = (np.abs(got['tiny_radfract'] - tiny) > 1e-12).sum()
    assert bad <= 4, bad


@pytest.mark.parametrize('lat,elev,seed', [(47.8, 1300.0, 1),
                                           (31.7, 1500.0, 2),
                                           (60.0, 100.0, 3)])
def test_mtclim_run_matches_the_reference(lat, elev, seed):
    """The dewpoint iteration, the transmittance and the cloud fraction."""
    n = 200
    tmax, tmin, prec = _weather(n, seed, lat)
    yday = [(40 + i) % 365 + 1 for i in range(n)]
    geom = mtclim.solar_geometry(_t(lat), _t(elev), device=DEV)
    s_tmax, s_tmin, s_tday = mtclim.calc_tair(_t(tmax)[None], _t(tmin)[None])
    res = mtclim.srad_humidity(s_tmax, s_tmin, s_tday, _t(prec)[None], yday,
                               geom)
    ref = r_mtclim(tmax, tmin, prec, yday, r_solar_geom(elev, lat), elev)
    assert np.abs(res['s_tdew'][0].cpu().numpy() - ref['tdew']).max() < 1e-5
    assert np.abs(res['s_vp'][0].cpu().numpy() - ref['vp']).max() < 1e-3
    assert np.abs(res['s_srad'][0].cpu().numpy() - ref['sw']).max() < 1e-4
    assert np.abs(res['s_tskc'][0].cpu().numpy() - ref['tskc']).max() < 1e-12


def test_a_cells_answer_does_not_depend_on_its_neighbours():
    """Convergence is tested per cell and a converged cell is frozen, so the
    same cell in a block of one and in a block of three agrees exactly."""
    n = 150
    yday = [i % 365 + 1 for i in range(n)]
    rows = [_weather(n, s, 45.0) for s in (4, 5, 6)]
    tmax = _t([r[0] for r in rows])
    tmin = _t([r[1] for r in rows])
    prec = _t([r[2] for r in rows])
    lat, elev = _t([45.0, 45.0, 45.0]), _t([200.0, 900.0, 1800.0])
    both = mtclim.disaggregate(yday, prec, tmax, tmin, lat, elev, device=DEV)
    one = mtclim.disaggregate(yday, prec[1:2], tmax[1:2], tmin[1:2],
                              lat[1:2], elev[1:2], device=DEV)
    for k in ('s_vp', 's_srad', 'tair'):
        assert torch.equal(both[k][1:2], one[k]), k


def test_sunrise_sunset_and_temperature_knots_match_the_reference():
    lat, elev, n = 47.8, 600.0, 120
    geom = mtclim.solar_geometry(_t([lat]), _t([elev]), device=DEV)
    yday = [(100 + i) % 365 + 1 for i in range(n)]
    t_tmin, t_tmax = mtclim.set_max_min_hour(geom['tiny_radfract'], yday)
    rise, sset = r_rise_set_minutes(r_solar_geom(elev, lat)[0][:365])
    ix = np.array(yday) - 1
    want_min = rise[ix] - 60.0
    want_max = 0.67 * (sset[ix] - rise[ix]) + rise[ix] - 60.0
    assert np.abs(t_tmin[0].cpu().numpy() - want_min).max() < 1e-9
    assert np.abs(t_tmax[0].cpu().numpy() - want_max).max() < 1e-9


def test_hourly_temperature_is_scipys_pchip_through_the_knots():
    """metsim.disaggregate.temp: PchipInterpolator over the minima and
    maxima, the end days repeated one day outside the record."""
    PchipInterpolator = pytest.importorskip('scipy.interpolate').PchipInterpolator
    lat, elev, n = 47.8, 600.0, 60
    tmax, tmin, _p = _weather(n, 9, lat)
    geom = mtclim.solar_geometry(_t([lat]), _t([elev]), device=DEV)
    yday = [(30 + i) % 365 + 1 for i in range(n)]
    t_tmin, t_tmax = mtclim.set_max_min_hour(geom['tiny_radfract'], yday)
    got = mtclim.hourly_t(t_tmin, _t(tmin)[None], t_tmax, _t(tmax)[None])
    tn, tx = t_tmin[0].cpu().numpy(), t_tmax[0].cpu().numpy()
    d0 = np.arange(n) * 1440.0
    time = np.ravel(np.column_stack((tn + d0, tx + d0)))
    temp = np.ravel(np.column_stack((tmin, tmax)))
    time = np.append(np.insert(time, 0, time[0:2] - 1440.0),
                     time[-2:] + 1440.0)
    temp = np.append(np.insert(temp, 0, temp[0:2]), temp[-2:])
    want = PchipInterpolator(time, temp, extrapolate=True)(
        60.0 * np.arange(n * 24))
    assert np.abs(got[0].cpu().numpy() - want).max() < 1e-10


def test_hourly_temperature_never_leaves_the_daily_extremes():
    """PCHIP is monotone between knots, so the hourly series cannot exceed
    the daily maximum or fall below the daily minimum it was built from, and
    it passes through each knot."""
    n = 40
    tmax = torch.linspace(5.0, 25.0, n, dtype=DT, device=DEV).unsqueeze(0)
    tmin = tmax - 9.0
    tn = torch.full((1, n), 180.0, dtype=DT, device=DEV)     # 03:00
    tx = torch.full((1, n), 840.0, dtype=DT, device=DEV)     # 14:00
    tair = mtclim.hourly_t(tn, tmin, tx, tmax)
    assert float(tair.max()) <= float(tmax.max()) + 1e-9
    assert float(tair.min()) >= float(tmin.min()) - 1e-9
    hourly = tair.reshape(1, n, 24)
    assert torch.allclose(hourly[0, :, 14], tmax[0], atol=1e-10)
    assert torch.allclose(hourly[0, :, 3], tmin[0], atol=1e-10)


def test_polar_days_fall_back_to_a_six_and_eighteen_hour_sun():
    """MetSim fails where the sun neither rises nor sets; here the knots stay
    ordered so the spline is defined."""
    geom = mtclim.solar_geometry(_t([80.0]), _t([0.0]), device=DEV)
    t_tmin, t_tmax = mtclim.set_max_min_hour(geom['tiny_radfract'],
                                             [172, 355])
    assert bool((t_tmax > t_tmin).all())
    assert torch.allclose(t_tmin, torch.full_like(t_tmin, 300.0))


def test_trailing_mean_uses_the_days_available():
    x = torch.arange(1.0, 11.0, dtype=DT, device=DEV)
    got = mtclim.trailing_mean(x, 4)
    want = [1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    assert np.allclose(got.cpu().numpy(), want)


def test_the_hourly_shortwave_keeps_the_daily_total():
    n, lat = 30, 47.8
    tmax, tmin, prec = _weather(n, 2, lat)
    yday = [i % 365 + 1 for i in range(n)]
    args = (yday, _t(prec)[None], _t(tmax)[None], _t(tmin)[None], lat, 500.0)
    res = mtclim.disaggregate(*args, device=DEV)
    day_mean = res['s_srad'] * res['s_dayl'] / 86400.0
    assert float((res['hourlyrad'].mean(-1) - day_mean).abs().max()) < 1e-9
    # a rotated day keeps it too, up to the neighbouring days' radiation the
    # rotation reaches into
    rot = mtclim.disaggregate(*args, hour_offset=0.4, device=DEV)
    assert float((rot['hourlyrad'].mean(-1) - day_mean).abs().max()) < 0.5


def test_svp_matches_the_physics_modules_svp():
    """MTCLIM's humidity and the snow model's must use one saturation curve."""
    from torchvic.core import snow
    t = torch.linspace(-40.0, 40.0, 81, dtype=DT, device=DEV)
    assert torch.allclose(mtclim.svp(t), snow.svp(t), rtol=1e-14, atol=0.0)
    plain = 0.61078 * torch.exp((17.269 * t) / (237.3 + t)) * 1000.0
    ratio = mtclim.svp(t) / plain
    assert float(ratio[t >= 0.0].min()) == 1.0
    assert float(ratio[0]) < 0.7          # -40 C


def test_an_observed_vapour_pressure_and_shortwave_are_taken_as_given():
    n, lat = 40, 47.8
    tmax, tmin, prec = _weather(n, 5, lat)
    yday = [i % 365 + 1 for i in range(n)]
    args = (yday, _t(prec)[None], _t(tmax)[None], _t(tmin)[None], lat, 500.0)
    vp = torch.full((1, n), 600.0, dtype=DT, device=DEV)
    got = mtclim.disaggregate(*args, vp_in=vp, device=DEV)
    assert float((got['s_vp'] - vp).abs().max()) == 0.0
    sw = torch.full((1, n), 150.0, dtype=DT, device=DEV)
    got = mtclim.disaggregate(*args, srad_in=sw, device=DEV)
    # the 24-hour mean of the hourly shortwave is the observation
    assert float((got['hourlyrad'].mean(-1) - sw).abs().max()) < 1e-9
    assert float(got['s_tskc'].min()) >= 0.0
    assert float(got['s_tskc'].max()) <= 1.0


def test_direct_fraction_follows_the_clearness_index():
    kt = torch.tensor([0.05, 0.3, 0.5, 0.8], dtype=DT, device=DEV)
    short_day = torch.full_like(kt, 30000.0)     # sunset hour angle < 81.4 deg
    kd = mtclim._erbs_diffuse_fraction(kt, short_day)
    assert bool((kd[1:] < kd[:-1]).all())          # clearer sky, less diffuse
    assert float(kd[-1]) == 0.143 and 0.9 < float(kd[0]) <= 1.0
    long_day = mtclim._erbs_diffuse_fraction(kt, short_day * 1.5)
    assert float(long_day[-1]) == 0.175


def test_daily_shortwave_is_differentiable_in_the_daily_temperature():
    """A daily forcing record carries a gradient into the physics that
    consumes it. Checked against a central finite difference for single
    days rather than asserting a sign: raising a day's maximum widens that
    day's range (clearer sky, more shortwave) and raises the 30-day smoothed
    range for the next 30 days (the opposite)."""
    n, lat = 120, 47.8
    tmax, tmin, prec = _weather(n, 3, lat)
    yday = [i % 365 + 1 for i in range(n)]
    tx = _t(tmax)[None].clone().requires_grad_(True)

    def total(x):
        return mtclim.disaggregate(yday, _t(prec)[None], x, _t(tmin)[None],
                                   lat, 500.0, device=DEV)['s_srad'].sum()

    total(tx).backward()
    g = tx.grad
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0.0
    eps = 1.0e-6
    for day in (10, 60):
        up = tx.detach().clone()
        up[0, day] += eps
        dn = tx.detach().clone()
        dn[0, day] -= eps
        fd = float((total(up) - total(dn)) / (2 * eps))
        # the dewpoint iteration stops at a 1e-6 rms change, which limits
        # how well a finite difference of its result can agree
        assert abs(fd - float(g[0, day])) < 1e-3 * max(1.0, abs(fd))


# ---------------------------------------------------------------------------
# How far this is from VIC 4.2.d's compiled MTCLIM
# ---------------------------------------------------------------------------
def _oracle(mode, seed):
    if not REF.exists():
        pytest.skip('ref_mtclim missing; run tests/oracle/build_ref_mtclim.sh')
    out = subprocess.run([str(REF), mode, str(seed)], capture_output=True,
                         text=True, check=True).stdout
    lines = [ln for ln in out.splitlines() if not ln.startswith('#')]
    head = [float(x) for x in lines[0].split(',')]
    rows = [[float(x) for x in ln.replace('|', ',').split(',')]
            for ln in lines[1:]]
    return head, rows


def _versus_vic42d():
    """Differences (this minus VIC 4.2.d), pooled over all ten sites."""
    pool = {}

    def col(rows, j):
        return torch.tensor([r[j] for r in rows], dtype=DT, device=DEV)

    for seed in SEEDS:
        head, dr = _oracle('daily', seed)
        _h, hr = _oracle('hourly', seed)
        lat, elev = head[0], head[2]
        n = len(dr)
        yday = [int(r[1]) for r in dr]
        res = mtclim.disaggregate(yday, col(dr, 2)[None], col(dr, 3)[None],
                                  col(dr, 4)[None], lat, elev, device=DEV)
        d = dict(
            sw=res['hourlyrad'].reshape(n, 24) - col(hr, 2).reshape(n, 24),
            tair=res['tair'].reshape(n, 24) - col(hr, 3).reshape(n, 24),
            vp=res['s_vp'].reshape(n) - col(dr, 6),
            tskc=res['s_tskc'].reshape(n) - col(dr, 5))
        for k, v in d.items():
            pool.setdefault(k, []).append(v.reshape(-1))
    return {k: torch.cat(v) for k, v in pool.items()}


def test_distance_from_vic42d_is_the_measured_one():
    """Ten sites, 44-76 N, 120 days each. Measured when MTCLIM was rebuilt
    on MetSim (this minus VIC 4.2.d):

        hourly SWdown   rmse 1.46 W/m2   max 57.0 W/m2   mean +0.06
        hourly Tair     rmse 0.82 C      max 4.41 C      mean -0.001
        daily vapour p. rmse 18.3 Pa     max 173 Pa      mean -3.8
        daily cloud     rmse 0.0034      max 0.075       (tskc)

    The bounds are set 10% above those. Where they come from: the PCHIP
    spline replaces VIC's zero-derivative Hermite spline and puts the
    minimum and maximum at MetSim's sunrise and 67%-of-daylight times
    (Tair); the dewpoint iterates to a 1e-6 rms tolerance rather than twice,
    and the trailing 30- and 90-day means start from the days at hand
    instead of VIC's pulled boxcar and wrapped precipitation window (vapour
    pressure, cloud, shortwave); and the radiation sums the same
    flat-surface geometry with different constants.
    """
    d = _versus_vic42d()
    rm = {k: float(v.pow(2).mean().sqrt()) for k, v in d.items()}
    mx = {k: float(v.abs().max()) for k, v in d.items()}
    assert rm['sw'] < 1.61 and mx['sw'] < 62.7, (rm, mx)
    assert rm['tair'] < 0.91 and mx['tair'] < 4.85, (rm, mx)
    assert rm['vp'] < 20.1 and mx['vp'] < 191.0, (rm, mx)
    assert rm['tskc'] < 0.0038 and mx['tskc'] < 0.083, (rm, mx)


# ---------------------------------------------------------------------------
# Real MetSim 2.0.1 output (tests/data/metsim_2_0_1_fixture.npz)
# ---------------------------------------------------------------------------
FIXTURE = pathlib.Path(__file__).parent / 'data' / 'metsim_2_0_1_fixture.npz'


@pytest.mark.parametrize('site', ['seed0', 'seed2'])
@pytest.mark.parametrize('step,spd', [(60, 24), (180, 8)])
def test_daily_path_matches_real_metsim_2_0_1(site, step, spd):
    """The daily-forcing path against output of MetSim 2.0.1 itself (PyPI
    package, options in the fixture's `options` entry; regenerate with
    tools/metsim_validation). Both were run on 90 leading days plus 120
    days; the last 60 are compared. Shortwave, cloud, precipitation and wind
    agree to round-off; air temperature, vapour pressure, pressure and
    longwave to the half-minute sunrise slot MetSim's numpy cosine puts a
    different side of zero than torch's (measured max over ten sites: Tair
    0.028 C, vp 1.4 Pa, pressure 1.6 Pa, LWdown 0.11 W/m2), and to the
    fixture's float32 storage."""
    import datetime
    from torchvic.forcing import daily
    z = np.load(FIXTURE)
    assert str(z['metsim_version']) == '2.0.1'
    lat, lng, elev, o = z[site + '/meta']
    lead, keep = int(z['lead']), int(z['keep'])
    start = datetime.date.fromordinal(int(o)) - datetime.timedelta(days=90)
    days = {k: torch.as_tensor(z['%s/%s' % (site, v)], dtype=DT,
                               device=DEV)[None]
            for k, v in (('tmax', 'tmax'), ('tmin', 'tmin'),
                         ('prcp', 'prec'), ('wind', 'wind'))}
    out = daily.daily_to_subdaily(days, start, float(lat), float(lng),
                                  float(elev), steps_per_day=spd,
                                  device=DEV)
    pairs = [('shortwave', 'shortwave', 1.0, 2e-3),
             ('longwave', 'longwave', 1.0, 0.25),
             ('tair', 'temp', 1.0, 0.05),
             ('vp', 'vapor_pressure', 1000.0, 2.5),
             ('press', 'air_pressure', 1000.0, 3.0),
             ('wind', 'wind', 1.0, 1e-5)]
    for kt, km, sc, tol in pairs:
        got = out[kt][0].cpu().numpy()[-keep * spd:]
        want = z['%s/%d/%s' % (site, step, km)].astype(float) * sc
        assert np.abs(got - want).max() < tol, (kt, np.abs(got - want).max())
