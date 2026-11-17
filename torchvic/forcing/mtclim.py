"""Daily forcings into sub-daily ones: MTCLIM as implemented in MetSim.

VIC 5 requires sub-daily forcings and will not run on a daily dataset. Most
of the world's forcing archives are daily, so this module builds the
sub-daily shortwave, vapour pressure, cloud fraction and air temperature
from daily maximum and minimum temperature and daily precipitation, and
`torchvic.forcing.daily` assembles the rest of the forcing set around it.

**Provenance.** The algorithm follows MetSim v2.0.1, the Meteorology
Simulator of the Computational Hydrology Group, University of Washington
(https://github.com/UW-Hydro/MetSim, GPL-3.0-or-later; tag v2.0.1, commit
742ef6897f7c3bb35649568f63cb30451e9ba4b9), whose `metsim/physics.py`,
`metsim/methods/mtclim.py` and `metsim/disaggregate.py` it re-implements as
batched PyTorch tensor code, and the published papers behind them:

  * Thornton, P. E. and Running, S. W. (1999), An improved algorithm for
    estimating incident daily solar radiation from measurements of
    temperature, humidity, and precipitation, Agric. For. Meteorol. 93,
    211-228 -- the transmittance from the daily temperature range;
  * Kimball, J. S., Running, S. W. and Nemani, R. (1997), An improved method
    for estimating surface humidity from daily minimum temperature, Agric.
    For. Meteorol. 85, 87-98 -- the dewpoint from the ratio of potential
    evapotranspiration to precipitation;
  * Bohn, T. J., Livneh, B., Oyler, J. W., Running, S. W., Nijssen, B. and
    Lettenmaier, D. P. (2013), Global evaluation of MTCLIM and related
    algorithms for forcing of ecological and hydrological models, Agric.
    For. Meteorol. 176, 38-49 -- the disaggregation MetSim implements.

Nothing here is derived from VIC 4.2.d's C source (GPL-2.0-only, which cannot
be combined with this project's GPL-3.0-or-later). An earlier version of this
module transcribed it; this one replaces it, and `THIRD_PARTY_NOTICES.md`
records the change.

The chain, all vectorised over a leading cell axis with days second:

    solar_geometry     once per site: potential radiation on a flat surface,
                       the daily transmittance, the daylength, and the
                       fraction of the day's radiation in each 30-second
                       interval (metsim.physics.solar_geom)
    srad_humidity      the transmittance from the daily temperature range,
                       iterated against a dewpoint estimated from potential
                       evapotranspiration until it stops changing
                       (metsim.methods.mtclim.run)
    to_hourly          the 30-second fractions summed into hourly shortwave
                       (metsim.disaggregate.shortwave)
    set_max_min_hour   the times of the daily minimum and maximum
                       temperature from sunrise and sunset
                       (metsim.disaggregate.set_min_max_hour)
    hourly_t           a monotone piecewise-cubic Hermite (PCHIP) spline
                       through those points (metsim.disaggregate.temp)

**What differs from MetSim.** MetSim loops over cells and runs in numpy; this
runs a block of cells at once and is differentiable. Convergence of the
dewpoint iteration is tested per cell, and a converged cell is frozen, so a
cell's answer does not depend on which cells share its block. The trailing
means that MetSim takes over a state file (90 days before the run) are
expanding means over the days available when the record is shorter, so a run
over a window should be given its margin of earlier days
(`torchvic.case.MTCLIM_MARGIN_DAYS`). The hourly temperature is always built
on an hourly grid and averaged to the model step by the caller. MetSim's
flat-surface radiation is the only radiation here: the slope, aspect and
horizon corrections and the snow-cover correction that VIC 4.2.d offered are
not part of MetSim's MTCLIM and are not provided. `CHANGELOG.md` lists every
numerical difference from VIC 4.2.d and its measured size.
"""
import math

import torch

# --- metsim/constants.py ----------------------------------------------------
SEC_PER_RAD = 13750.9871   #: seconds per radian of hour angle
RAD_PER_DAY = 0.017214     #: radians of Earth orbit per julian day
RAD_PER_DEG = 0.01745329   #: radians per degree
MIN_DECL = -0.4092797      #: minimum declination, radians
DAYS_OFF = 11.25           #: julian day offset of the winter solstice
SW_RAD_DT = 30.0           #: timestep of the radiation routine, seconds
MA = 28.9644e-3            #: kg/mol, molecular weight of air
R_GAS = 8.3143             #: m3 Pa /mol /K
R_DRY = 287.0              #: J /kg /K, gas constant of dry air
G_STD = 9.80665            #: m/s2
P_STD = 101325.0           #: Pa at 0 m
T_STD = 288.15             #: K at 0 m
CP = 1010.0                #: J /kg /K, specific heat of air
KELVIN = 273.15
EPS = 0.62196351           #: ratio of molecular weights, water to dry air
DAYS_PER_YEAR = 365.25
SEC_PER_DAY = 86400.0
SEC_PER_HOUR = 3600.0
MIN_PER_DAY = 1440.0
STEFAN_B = 5.669e-8        #: W /m2 /K4
LAPSE_RATE = 0.0065        #: K/m, MetSim's `lapse_rate` default

# the radiation algorithm (Thornton and Running 1999)
C_RAD = 1.5                #: radiation parameter
B0 = 0.031
B1 = 0.201
B2 = 0.185
TBASE = 0.870              #: maximum instantaneous transmittance, dry, nadir
ABASE = -6.1e-5            #: vapour pressure effect on transmittance, 1/Pa

# --- metsim.metsim.MetSim.params defaults ------------------------------------
TDAY_COEF = 0.45           #: daylight air temperature coefficient
RAIN_SCALAR = 0.75         #: transmittance correction on a rain day
TDEW_TOL = 1e-6            #: dewpoint iteration tolerance, C (rmse)
TMAX_DAYLENGTH_FRACTION = 0.67  #: where in the daylight the maximum falls
SW_PREC_THRESH = 0.0       #: mm/day above which a day is a rain day

#: Safety cap on the dewpoint iteration. MetSim's loop has none; on every
#: record tried it stops in well under thirty passes.
MAX_TDEW_ITER = 200

#: Floor under the 90-day precipitation total, mm/yr, which keeps the
#: potential-evapotranspiration ratio finite in an arid climate
#: (metsim.methods.mtclim.tdew).
MIN_SEASONAL_PREC = 80.0

#: Optical air mass above 2.9, indexed by the zenith angle in degrees minus
#: 69 (metsim.physics.solar_geom).
OPTAM = (2.90, 3.05, 3.21, 3.39, 3.69, 3.82, 4.07, 4.37, 4.72, 5.12, 5.60,
         6.18, 6.88, 7.77, 8.90, 10.39, 12.44, 15.36, 19.79, 26.96, 30.00)

#: 86400 / SW_RAD_DT, the number of radiation sub-steps in a day.
TINYSTEPS_PER_DAY = int(SEC_PER_DAY // SW_RAD_DT)


def svp(temp):
    """Saturation vapour pressure, Pa (Maidment 1992, eq. 4.2.2, with the
    sub-freezing correction). metsim.physics.svp."""
    s = 0.61078 * torch.exp((17.269 * temp) / (237.3 + temp))
    cold = 1.0 + 0.00972 * temp + 0.000042 * temp * temp
    return torch.where(temp < 0.0, s * cold, s) * 1000.0


def atm_pres(elev, lr=LAPSE_RATE):
    """Air pressure from elevation, Pa (Iribane and Godson 1981, p. 168).
    metsim.physics.atm_pres."""
    t1 = 1.0 - (lr * elev) / T_STD
    t2 = G_STD / (lr * (R_GAS / MA))
    return P_STD * torch.pow(t1, t2)


def calc_pet(rad, ta, pa, dayl, dt=0.2):
    """Priestley-Taylor potential evapotranspiration, cm/day (Kimball et al.
    1997). metsim.physics.calc_pet.

    `rad` is the daylight-average shortwave in W/m2, `ta` the daylight
    average air temperature in C, `pa` the air pressure in Pa and `dayl` the
    daylength in seconds.
    """
    rnet = rad * 0.72                       # albedo 0.2, ground flux 10%
    lhvap = 2.5023e6 - 2430.54 * ta
    gamma = CP * pa / (lhvap * EPS)
    t1, t2 = ta + dt, ta - dt
    s = (svp(t1) - svp(t2)) / (t1 - t2)
    pet = (1.26 * (s / (s + gamma)) * rnet * dayl) / lhvap
    return pet / 10.0


def trailing_mean(x, w):
    """Trailing mean of width `w` along the last axis; where fewer than `w`
    days exist, the mean of those that do.

    MetSim takes this mean (30 days for the temperature range, 90 for the
    precipitation) over the record preceded by a state file of earlier days.
    Without one the first days see a shorter window.
    """
    n = x.shape[-1]
    zero = torch.zeros_like(x[..., :1])
    cs = torch.cat([zero, torch.cumsum(x, -1)], -1)          # cs[i] = sum x[:i]
    i = torch.arange(1, n + 1, device=x.device)
    lo = torch.clamp(i - w, min=0)
    cnt = (i - lo).to(x.dtype)
    return (cs[..., 1:] - torch.index_select(cs, -1, lo)) / cnt


#: How many days of the year `solar_geometry` computes at once.
#:
#: Its working arrays are [cell, day, 2881] -- the hour-angle axis at the
#: radiation routine's 30-second resolution -- so at 365 days they cost 8.4
#: MiB per cell each in float64, about a dozen live at once. Days are
#: independent (the only reduction is over the hour angle within a day), so
#: chunking changes peak memory and nothing else, which
#: `tests/test_mtclim_chunking.py` asserts bit-exactly against
#: `day_chunk=None`. 16 days is a tenth of the memory for a negligible time
#: cost; the floor is `tiny_radfract` itself, [cell, 366, 2880] float64.
DAY_CHUNK = 16


def solar_geometry(lat_deg, elev, dtype=torch.float64, device=None,
                   day_chunk=DAY_CHUNK):
    """The part of the radiation that depends only on where the site is
    (flat surface). metsim.physics.solar_geom.

    Returns a dict with, per cell and per day of year (366 of them, the last
    a copy of the 365th as MetSim does):

        ttmax0        daily mean atmospheric transmittance at nadir, dry
        flat_potrad   potential radiation on a flat surface, W/m2 of daylight
        daylength     seconds
        tiny_radfract [cell, 366, 2880] the fraction of the day's radiation
                      arriving in each 30-second interval
        pa            air pressure at the site, Pa
    """
    kw = dict(dtype=dtype, device=device)
    lat = torch.as_tensor(lat_deg, **kw) * RAD_PER_DEG
    lat = torch.clamp(lat, -math.pi / 2.0, math.pi / 2.0)
    elev = torch.as_tensor(elev, **kw)
    cell = lat.shape

    t1 = 1.0 - (LAPSE_RATE * elev) / T_STD
    t2 = G_STD / (LAPSE_RATE * (R_GAS / MA))
    trans1 = torch.pow(TBASE, torch.pow(t1, t2))
    trans1_c = trans1.reshape(cell + (1, 1))

    coslat, sinlat = torch.cos(lat), torch.sin(lat)
    dt = SW_RAD_DT
    dh = dt / SEC_PER_RAD

    nday = 365
    step = nday if day_chunk is None else max(1, int(day_chunk))
    nk = TINYSTEPS_PER_DAY + 1
    k = torch.arange(nk, **kw)
    optam = torch.tensor(OPTAM, **kw)

    daylength = torch.empty(cell + (nday,), **kw)
    sum_trans = torch.empty(cell + (nday,), **kw)
    sum_flat_potrad = torch.empty(cell + (nday,), **kw)
    tiny = torch.empty(cell + (nday + 1, TINYSTEPS_PER_DAY), **kw)

    for d0 in range(0, nday, step):
        d1 = min(d0 + step, nday)
        nd = d1 - d0
        i = torch.arange(d0, d1, **kw)
        decl = MIN_DECL * torch.cos((i + DAYS_OFF) * RAD_PER_DAY)
        cosdecl, sindecl = torch.cos(decl), torch.sin(decl)

        def outer(a, b, _nd=nd):
            """[cell] against [day] -> [cell, day]."""
            return (a.reshape(cell + (1,))
                    * b.reshape((1,) * len(cell) + (_nd,)))

        cosegeom = outer(coslat, cosdecl)
        sinegeom = outer(sinlat, sindecl)
        coshss = torch.clamp(-sinegeom / cosegeom, -1.0, 1.0)
        hss = torch.acos(coshss)
        daylength[..., d0:d1] = torch.clamp(2.0 * hss * SEC_PER_RAD,
                                            max=SEC_PER_DAY)

        # Extraterrestrial radiation perpendicular to the beam, J/m2 per step.
        sc = 1368.0 + 45.5 * torch.sin(2.0 * math.pi * i / DAYS_PER_YEAR + 1.7)
        dir_beam_topa = (sc * dt).reshape((1,) * len(cell) + (nd,))

        # np.arange(-hss, hss, dh): the half-open hour-angle grid.
        h = hss.unsqueeze(-1).neg() + k * dh                # [cell, day, k]
        live = h < hss.unsqueeze(-1)
        cza = cosegeom.unsqueeze(-1) * torch.cos(h) + sinegeom.unsqueeze(-1)
        up = live & (cza > 0.0)
        dir_flat_topa = dir_beam_topa.unsqueeze(-1) * cza

        am_simple = 1.0 / (cza + 0.0000001)
        ami = torch.clamp((torch.acos(torch.clamp(cza, -1.0, 1.0))
                           / RAD_PER_DEG).floor().long() - 69, 0, 20)
        am = torch.where(am_simple > 2.9, optam[ami], am_simple)
        trans2 = torch.pow(trans1_c, am)

        zero = torch.zeros_like(cza)
        sum_trans[..., d0:d1] = torch.where(
            up, trans2 * dir_flat_topa, zero).sum(-1)
        sum_flat_potrad[..., d0:d1] = torch.where(
            up, dir_flat_topa, zero).sum(-1)

        # Each step's radiation goes to its 30-second slot; where two hour
        # angles land in one slot the later one wins, as in MetSim's loop,
        # which overwrites.
        base = ((12.0 * SEC_PER_HOUR - hss * SEC_PER_RAD) / dt).floor().long()
        tinystep = torch.clamp(base.unsqueeze(-1) + k.long(),
                               0, TINYSTEPS_PER_DAY - 1)
        frac_src = torch.where(live & up, dir_flat_topa, zero)
        kk = k.long().expand_as(tinystep)
        winner = torch.full(cell + (nd, TINYSTEPS_PER_DAY), -1,
                            dtype=torch.long, device=kw['device'])
        winner.scatter_reduce_(-1, tinystep, kk, reduce='amax',
                               include_self=True)
        t = torch.where(
            winner >= 0,
            torch.gather(frac_src, -1, torch.clamp(winner, min=0)),
            torch.zeros(cell + (nd, TINYSTEPS_PER_DAY), **kw))

        sfp_c = sum_flat_potrad[..., d0:d1]
        lit_c = (daylength[..., d0:d1] > 0.0) & (sfp_c > 0.0)
        safe_c = torch.where(lit_c, sfp_c, torch.ones_like(sfp_c))
        t = t / safe_c.unsqueeze(-1)
        tiny[..., d0:d1, :] = torch.where(lit_c.unsqueeze(-1), t,
                                          torch.zeros_like(t))

    day = daylength > 0.0
    dl = torch.where(day, daylength, torch.ones_like(daylength))
    sfp = torch.where(sum_flat_potrad > 0.0, sum_flat_potrad,
                      torch.ones_like(sum_flat_potrad))
    z = torch.zeros_like(daylength)
    ttmax0 = torch.where(day, sum_trans / sfp, z)
    flat_potrad = torch.where(day, sum_flat_potrad / dl, z)

    def pad366(x):
        """Day 366 repeats day 365."""
        return torch.cat([x, x[..., -1:]], -1)

    tiny[..., nday, :] = tiny[..., nday - 1, :]
    return dict(ttmax0=pad366(ttmax0), flat_potrad=pad366(flat_potrad),
                daylength=pad366(daylength), tiny_radfract=tiny,
                pa=atm_pres(elev))


def calc_tair(tmax, tmin, tday_coef=TDAY_COEF):
    """Maximum, minimum and daylight-average air temperature.
    metsim.methods.mtclim.t_day."""
    tmean = (tmax + tmin) / 2.0
    return tmax, tmin, ((tmax - tmean) * tday_coef) + tmean


def _erbs_diffuse_fraction(kt, daylength):
    """Daily diffuse fraction of the shortwave from the clearness index.

    MetSim has no direct/diffuse partition. A CARBON run needs one, so the
    daily correlation of Erbs, Klein and Duffie (1982, "Estimation of the
    diffuse radiation fraction for hourly, daily and monthly-average global
    radiation", Solar Energy 28(4), 293-302), eqn (6), is used. Checked
    against the paper's text; as printed there, with Kt the daily clearness
    index and ws the sunset hour angle:

        ws < 1.4208 rad (81.4 deg):
            Hd/H = 1.0 - 0.2727 Kt + 2.4495 Kt^2 - 11.9514 Kt^3
                   + 9.3879 Kt^4                  for Kt < 0.715
            Hd/H = 0.143                          for Kt >= 0.715
        ws >= 1.4208 rad:
            Hd/H = 1.0 + 0.2832 Kt - 2.5557 Kt^2 + 0.8448 Kt^3
                                                  for Kt < 0.722
            Hd/H = 0.175                          for Kt >= 0.722

    The sunset hour angle is half the daylength over SEC_PER_RAD.
    """
    kt = torch.clamp(kt, 0.0, 1.0)
    short = (1.0 - 0.2727 * kt + 2.4495 * kt ** 2 - 11.9514 * kt ** 3
             + 9.3879 * kt ** 4)
    short = torch.where(kt < 0.715, short, torch.full_like(kt, 0.143))
    long_ = 1.0 + 0.2832 * kt - 2.5557 * kt ** 2 + 0.8448 * kt ** 3
    long_ = torch.where(kt < 0.722, long_, torch.full_like(kt, 0.175))
    kd = torch.where(daylength >= 2.0 * 1.4208 * SEC_PER_RAD, long_, short)
    return torch.clamp(kd, 0.0, 1.0)


def srad_humidity(tmax, tmin, tday, prec_mm, yday, geom, vp_in=None,
                  srad_in=None, lw_cloud_deardorff=True,
                  sw_prec_thresh=SW_PREC_THRESH, tdew_tol=TDEW_TOL,
                  rain_scalar=RAIN_SCALAR, max_iter=MAX_TDEW_ITER):
    """Daylight shortwave, cloud fraction and vapour pressure from the daily
    temperatures and precipitation. metsim.methods.mtclim.run.

    The transmittance depends on vapour pressure through ABASE and the
    dewpoint is estimated from the ratio of potential evapotranspiration --
    which needs the radiation -- to the 90-day precipitation total. MetSim
    settles this by iterating until the dewpoint's rms change between passes
    is below `tdew_tol`. Here that is tested per cell and a converged cell is
    held fixed.

    `vp_in` (Pa) is an observed daily vapour pressure; the humidity
    estimate is then skipped. `srad_in` is an observed daily-mean shortwave
    (W/m2 over 24 h); the cloud transmittance is then inferred from it
    through MetSim's own relation shortwave = potrad * t_tmax * tfmax.

    Every array is [..., ndays]; `yday` is a 1-based day-of-year sequence.
    """
    idx = torch.as_tensor([int(d) - 1 for d in yday], dtype=torch.long,
                          device=tmax.device)
    daylength = geom['daylength'].index_select(-1, idx)
    ttmax0 = geom['ttmax0'].index_select(-1, idx)
    potrad = geom['flat_potrad'].index_select(-1, idx)
    pa = geom['pa'].unsqueeze(-1)

    dtr = torch.clamp(tmax - tmin, min=0.0)
    sm_dtr = trailing_mean(dtr, 30)
    seasonal = torch.clamp(trailing_mean(prec_mm, 90) * DAYS_PER_YEAR,
                           min=MIN_SEASONAL_PREC)

    b = B0 + B1 * torch.exp(-B2 * sm_dtr)
    tfmax = 1.0 - 0.9 * torch.exp(-b * torch.pow(dtr, C_RAD))
    tfmax = torch.where(prec_mm > sw_prec_thresh, tfmax * rain_scalar, tfmax)

    if srad_in is not None:
        # a 24-hour mean as observed, as a daylight mean
        lit = daylength > 0.0
        sw_obs = torch.where(lit, srad_in * SEC_PER_DAY
                             / torch.where(lit, daylength,
                                           torch.ones_like(daylength)),
                             torch.zeros_like(srad_in))

    def shortwave(vp):
        t_tmax = torch.clamp(ttmax0 + ABASE * vp, min=0.0001)
        if srad_in is None:
            return potrad * t_tmax * tfmax, tfmax, t_tmax
        tf = torch.where(
            (potrad > 0.0) & (sw_obs > 0.0),
            torch.clamp(sw_obs / torch.clamp(potrad * t_tmax, min=1e-30),
                        max=1.0),
            torch.ones_like(tfmax))
        return sw_obs, tf, t_tmax

    def tdew_of(sw):
        pet = calc_pet(sw, tday, pa, daylength) * 10.0         # mm/day
        ratio = pet / seasonal
        tdew = ((tmin + KELVIN)
                * (-0.127 + 1.121 * (1.003 - 1.444 * ratio
                                     + 12.312 * ratio ** 2
                                     - 32.766 * ratio ** 3)
                   + 0.0006 * dtr) - KELVIN)
        return tdew, pet

    if vp_in is not None:
        vp = vp_in
        tdew = tmin
    else:
        tdew_old = tmin
        sw, _tf, _tt = shortwave(svp(tdew_old))
        tdew, _ = tdew_of(sw)
        active = torch.sqrt(((tdew - tdew_old) ** 2).mean(-1)) > tdew_tol
        it = 1
        while bool(active.any()) and it < max_iter:
            sw, _tf, _tt = shortwave(svp(tdew))
            new, _ = tdew_of(sw)
            rm = torch.sqrt(((new - tdew) ** 2).mean(-1))
            tdew = torch.where(active.unsqueeze(-1), new, tdew)
            active = active & (rm > tdew_tol)
            it += 1
        vp = svp(tdew)

    sw, tfmax_out, t_tmax = shortwave(vp)
    _, pet = tdew_of(sw)
    if lw_cloud_deardorff:
        tskc = 1.0 - tfmax_out
    else:
        tskc = torch.sqrt(torch.clamp((1.0 - tfmax_out) / 0.65, min=0.0))
    kt = torch.where(potrad > 0.0, sw / torch.clamp(potrad, min=1e-30),
                     torch.zeros_like(sw))
    fdir = 1.0 - _erbs_diffuse_fraction(kt, daylength)
    vpd = torch.clamp(svp(tday) - vp, min=0.0)
    return dict(s_srad=sw, s_tfmax=tfmax_out, s_tskc=tskc, s_fdir=fdir,
                s_tdew=tdew, s_vp=vp, s_vpd=vpd, s_dayl=daylength, pet=pet,
                s_ttmax=t_tmax)


def to_hourly(s_srad, s_dayl, yday, tiny_radfract, hour_offset=0.0):
    """Spread the daily shortwave over the 24 hours of the day.
    metsim.disaggregate.shortwave.

    `tiny_radfract` holds, for each of 366 ydays, the fraction of the day's
    radiation falling in each 30-second interval; summing 120 of them gives
    the hour. `s_srad` is the daylight mean, so it scales by the daylength.
    `hour_offset` (hours, positive = later) rotates the fractions through the
    flattened year, as MetSim does for its UTC offset.
    """
    tiny_ph = int(SEC_PER_HOUR // SW_RAD_DT)               # 120
    shift = int(hour_offset * tiny_ph)                     # truncated, as int()
    frac = tiny_radfract
    if shift:
        flat = frac.reshape(*frac.shape[:-2], -1)
        frac = torch.roll(flat, shifts=shift, dims=-1).reshape(frac.shape)
    hourly_frac = frac.reshape(*frac.shape[:-1], 24, tiny_ph).sum(-1)
    idx = torch.as_tensor([int(d) - 1 for d in yday], dtype=torch.long,
                          device=s_srad.device)
    hourly_frac = hourly_frac.index_select(-2, idx)        # [cell, ndays, 24]
    return hourly_frac * (s_srad * s_dayl / SEC_PER_HOUR).unsqueeze(-1)


def set_max_min_hour(tiny_radfract, yday, hour_offset=0.0, ts=60.0,
                     tmax_daylength_fraction=TMAX_DAYLENGTH_FRACTION):
    """Times of the daily minimum and maximum temperature, in minutes after
    the start of the day. metsim.disaggregate.set_min_max_hour.

    Sunrise is the 30-second step before the first lit one and sunset the
    last lit one. The maximum falls `tmax_daylength_fraction` of the way from
    sunrise to sunset and the minimum at sunrise, both one output step `ts`
    (minutes) earlier, as in MetSim. `hour_offset` moves both later.

    MetSim assumes one sunrise and one sunset every day and fails where the
    sun does not rise or set. There the times fall back to a 06:00 sunrise
    and 18:00 sunset so the spline stays ordered.

    Returns `(t_tmin, t_tmax)`, each [..., ndays].
    """
    lit = tiny_radfract > 0.0                              # [cell, 366, 2880]
    d = lit[..., 1:].to(torch.int8) - lit[..., :-1].to(torch.int8)
    n = d.shape[-1]
    pos = torch.arange(n, device=lit.device)
    big = torch.full_like(pos, n + 1)
    rise_i = torch.where(d > 0, pos, big).amin(-1)
    set_i = torch.where(d < 0, pos, -torch.ones_like(pos)).amax(-1)
    ok = (rise_i <= n) & (set_i >= 0)
    step_min = SW_RAD_DT / 60.0
    dtype = tiny_radfract.dtype
    rise = torch.where(ok, rise_i.to(dtype) * step_min,
                       torch.full(ok.shape, 360.0, dtype=dtype,
                                  device=lit.device))
    sset = torch.where(ok, set_i.to(dtype) * step_min,
                       torch.full(ok.shape, 1080.0, dtype=dtype,
                                  device=lit.device))
    rise = rise + hour_offset * 60.0
    sset = sset + hour_offset * 60.0
    idx = torch.as_tensor([int(y) - 1 for y in yday], dtype=torch.long,
                          device=lit.device)
    rise = rise.index_select(-1, idx)
    sset = sset.index_select(-1, idx)
    t_tmax = tmax_daylength_fraction * (sset - rise) + rise - ts
    t_tmin = rise - ts
    return t_tmin, t_tmax


def _pchip_slopes(x, y):
    """Derivatives of the monotone piecewise-cubic Hermite interpolant
    (Fritsch and Carlson 1980) at the knots, along the last axis; the same
    rule as scipy.interpolate.PchipInterpolator, which MetSim calls."""
    h = x[..., 1:] - x[..., :-1]
    m = (y[..., 1:] - y[..., :-1]) / h
    h0, h1 = h[..., :-1], h[..., 1:]
    m0, m1 = m[..., :-1], m[..., 1:]
    w1, w2 = 2.0 * h1 + h0, h1 + 2.0 * h0
    same = (m0 * m1) > 0.0
    den = (w1 / torch.where(same, m0, torch.ones_like(m0))
           + w2 / torch.where(same, m1, torch.ones_like(m1)))
    inner = torch.where(same, (w1 + w2) / den, torch.zeros_like(m0))

    def edge(ha, hb, ma, mb):
        e = ((2.0 * ha + hb) * ma - ha * mb) / (ha + hb)
        e = torch.where(torch.sign(e) != torch.sign(ma), torch.zeros_like(e), e)
        clip = (torch.sign(ma) != torch.sign(mb)) & (e.abs() > 3.0 * ma.abs())
        return torch.where(clip, 3.0 * ma, e)

    first = edge(h[..., 0], h[..., 1], m[..., 0], m[..., 1])
    last = edge(h[..., -1], h[..., -2], m[..., -1], m[..., -2])
    return torch.cat([first.unsqueeze(-1), inner, last.unsqueeze(-1)], -1)


def hourly_t(t_tmin, tmin, t_tmax, tmax, dt=60.0):
    """The sub-daily air temperature: a PCHIP through the daily extremes.
    metsim.disaggregate.temp.

    `t_tmin`, `t_tmax` are minutes after the start of each day ([..., ndays]);
    the knots are the minimum and maximum of every day, in time order, with
    the first and last day repeated one day before and after the record so
    the interpolant is tied down at both ends (MetSim's default when no
    neighbouring days are given). The output is on a `dt`-minute grid from
    the start of the first day, [..., ndays * 1440 / dt].
    """
    ndays = tmax.shape[-1]
    dev, dtype = tmax.device, tmax.dtype
    day0 = torch.arange(ndays, device=dev, dtype=dtype) * MIN_PER_DAY
    x = torch.stack([t_tmin + day0, t_tmax + day0], -1).reshape(
        *tmax.shape[:-1], ndays * 2)
    y = torch.stack([tmin, tmax], -1).reshape(*tmax.shape[:-1], ndays * 2)
    x = torch.cat([x[..., :2] - MIN_PER_DAY, x, x[..., -2:] + MIN_PER_DAY], -1)
    y = torch.cat([y[..., :2], y, y[..., -2:]], -1)

    d = _pchip_slopes(x, y)
    nsteps = int(round(MIN_PER_DAY / dt)) * ndays
    xb = (torch.arange(nsteps, device=dev, dtype=dtype) * dt).expand(
        *x.shape[:-1], nsteps).contiguous()
    n = x.shape[-1]
    k = torch.clamp(torch.searchsorted(x.contiguous(), xb, right=True) - 1,
                    0, n - 2)
    x0, x1 = torch.gather(x, -1, k), torch.gather(x, -1, k + 1)
    y0, y1 = torch.gather(y, -1, k), torch.gather(y, -1, k + 1)
    d0, d1 = torch.gather(d, -1, k), torch.gather(d, -1, k + 1)
    hh = x1 - x0
    s = (xb - x0) / hh
    h00 = (1.0 + 2.0 * s) * (1.0 - s) ** 2
    h10 = s * (1.0 - s) ** 2
    h01 = s * s * (3.0 - 2.0 * s)
    h11 = s * s * (s - 1.0)
    return h00 * y0 + h10 * hh * d0 + h01 * y1 + h11 * hh * d1


def disaggregate(yday, prec_mm, tmax, tmin, lat_deg, elev, hour_offset=0.0,
                 vp_in=None, srad_in=None, geom=None,
                 lw_cloud_deardorff=True, sw_prec_thresh=SW_PREC_THRESH,
                 step_minutes=60.0, dtype=torch.float64, device=None):
    """A daily forcing record turned into the sub-daily one VIC's physics
    runs on, by MetSim's MTCLIM and disaggregation.

    Inputs are [cell, ndays] tensors (or broadcastable), `yday` a 1-based
    day-of-year sequence of length ndays, precipitation in mm and temperature
    in C. Returns the daily fields of `srad_humidity`, the hourly shortwave
    [cell, ndays, 24] and air temperature [cell, ndays * 24], and the times of
    the daily minimum and maximum in minutes after the start of the day.

    `step_minutes` is the length of the model step the series will be read
    at. MetSim places the daily minimum and maximum `time_step` minutes
    before sunrise and the 67%-of-daylight point (set_min_max_hour), so the
    temperature knots move with the step; the air temperature and the
    interpolated vapour pressure are then read at the instants the steps
    begin. The series here stay hourly: with a longer step the caller takes
    every (step_minutes / 60)-th hour (see daily.daily_to_subdaily).

    `geom` lets a caller reuse a solar_geometry() result across a long run:
    it depends only on the site, never on the weather.
    """
    def T(v):
        t = torch.as_tensor(v, dtype=dtype, device=device)
        return t.unsqueeze(-1) if t.dim() == 0 else t

    tmax, tmin, prec_mm = T(tmax), T(tmin), T(prec_mm)
    if geom is None:
        geom = solar_geometry(lat_deg, elev, dtype=dtype, device=tmax.device)

    s_tmax, s_tmin, s_tday = calc_tair(tmax, tmin)
    daily = srad_humidity(s_tmax, s_tmin, s_tday, prec_mm, yday, geom,
                          vp_in=vp_in, srad_in=srad_in,
                          lw_cloud_deardorff=lw_cloud_deardorff,
                          sw_prec_thresh=sw_prec_thresh)

    hourlyrad = to_hourly(daily['s_srad'], daily['s_dayl'], yday,
                          geom['tiny_radfract'], hour_offset=hour_offset)
    t_tmin, t_tmax = set_max_min_hour(geom['tiny_radfract'], yday,
                                      hour_offset=hour_offset,
                                      ts=float(step_minutes))
    t_tmin, t_tmax = t_tmin.expand_as(s_tmin), t_tmax.expand_as(s_tmax)
    tair = hourly_t(t_tmin, s_tmin, t_tmax, s_tmax)

    out = dict(daily)
    out.update(hourlyrad=hourlyrad, tair=tair, t_tmin=t_tmin, t_tmax=t_tmax,
               s_tmax=s_tmax, s_tmin=s_tmin, s_tday=s_tday, s_prcp=prec_mm,
               geom=geom)
    return out
