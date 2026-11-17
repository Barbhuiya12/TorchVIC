"""A daily forcing record turned into the forcing set VIC 5 requires.

VIC 5 will not run on daily data. Its drivers demand AIR_TEMP, PREC,
PRESSURE, SWDOWN, LWDOWN, VP and WIND at the model's own time step and
stop with an error otherwise. `torchvic.forcing.mtclim` builds the
shortwave, vapour pressure, cloud fraction and hourly air temperature from
daily temperature and precipitation, following MetSim (UW-Hydro, GPL-3.0;
see that module). This module is the rest: it takes those and assembles the
full sub-daily forcing set, including the pieces MTCLIM has nothing to do
with (precipitation spread over the day, wind held constant, pressure from
elevation and temperature, longwave from cloud fraction and humidity), each
as MetSim's `metsim/disaggregate.py` does it.

Nothing here is physics. It produces the arrays a sub-daily forcing file
would have contained, and the model that consumes them is unchanged -- a
run driven from daily data takes the same code path as a run driven from an
hourly file.

    >>> sub = daily_to_subdaily(
    ...     dict(tmax=tmax, tmin=tmin, prcp=prcp, wind=wind),
    ...     start=datetime.date(1970, 1, 1), lat=47.8, lng=-121.5,
    ...     elev=1300.0, steps_per_day=24)
    >>> sorted(sub)
    ['longwave', 'prcp', 'press', 'shortwave', 'tair', 'vp', 'wind']

**Local time.** A daily record is a local-solar day, and a forcing file
usually is not on that clock: a cell at 121.5 W inside the -120 time zone is
6 minutes off centre, and one recorded in UTC is eight hours off. MetSim
handles this with `utc_offset`, rotating the sub-daily series by the
longitude's distance from the time-zone meridian. Here that offset is split
into whole hours and a remainder. The remainder rotates the 30-second
radiation fractions inside the day and moves the temperature extremes; the
whole hours shift the whole record and pad it with one extra day taken from
the record's own first or last day, so that no precipitation is wrapped
round the ends of the record (MetSim's rotation would wrap it). Pass
`time_zone_lng` to say where the clock is; leaving it out puts the clock at
the cell.
"""
import datetime
import math

import torch

from . import mtclim

#: The floor put under a wind speed, m/s (applied at every step length).
MIN_WIND_SPEED = 0.1
#: The wind used when none is supplied, m/s.
DEFAULT_WIND_SPEED = 1.5
#: metsim/constants.py -- the constants of metsim.disaggregate.pressure and
#: .longwave (mtclim.R_DRY, G_STD, P_STD, KELVIN, STEFAN_B, LAPSE_RATE are
#: the same values, re-exported for the callers that read them here).
STEFAN_B = mtclim.STEFAN_B
KELVIN = mtclim.KELVIN

#: The CO2 concentration assumed when none is supplied, ppm, and the
#: empirical fraction of shortwave that is photosynthetically active.
CATM_CURRENT_PPM = 383.0
SW2PAR = 0.45

#: The clear-sky emissivity formulations of metsim.disaggregate.longwave
#: (Bohn et al. 2013). The default here is Prata, which is also MetSim's
#: default in `MetSim.params`.
LW_TYPES = ('TVA', 'ANDERSON', 'BRUTSAERT', 'SATTERLUND', 'IDSO', 'PRATA')


def local_time_offset(lng, time_zone_lng=None):
    """The cell's distance from its time zone's centre, split into whole
    hours and a remainder (the offset of MetSim's `utc_offset` option,
    metsim.disaggregate.set_min_max_hour).

    Returns `(fractional_hours, whole_hours)`. The whole part rounds half
    away from zero rather than through `round()`, which sends halves to
    even: a cell exactly 7.5 hours from its meridian would otherwise land
    on a different day depending on its sign.
    """
    if time_zone_lng is None:
        time_zone_lng = lng
    off = (float(time_zone_lng) - float(lng)) * 24.0 / 360.0
    whole = int(off - 0.5) if off < 0 else int(off + 0.5)
    return off - whole, whole


def calc_longwave(tskc, air_temp, vp, lw_type='PRATA',
                  lw_cloud_deardorff=True):
    """Incoming longwave from cloud fraction, air temperature and vapour
    pressure, W/m2.  metsim.disaggregate.longwave (Bohn et al. 2013), whose
    clear-sky emissivities are TVA 1972, Anderson 1954, Brutsaert 1975,
    Satterlund 1979, Idso 1981 and Prata 1996.

    `air_temp` is C and `vp` Pa; the emissivity formulas are stated for
    Kelvin and millibar, so both are converted on entry.
    """
    tk = air_temp + KELVIN
    mbar = vp / 100.0
    lw_type = lw_type.upper()
    if lw_type == 'TVA':
        eps_clear = 0.740 + 0.0049 * mbar
    elif lw_type == 'ANDERSON':
        eps_clear = 0.68 + 0.036 * torch.sqrt(torch.clamp(mbar, min=0.0))
    elif lw_type == 'BRUTSAERT':
        eps_clear = 1.24 * torch.pow(torch.clamp(mbar / tk, min=1e-30),
                                     0.14285714)
    elif lw_type == 'SATTERLUND':
        eps_clear = 1.08 * (1.0 - torch.exp(
            -torch.pow(torch.clamp(mbar, min=1e-30), tk / 2016.0)))
    elif lw_type == 'IDSO':
        eps_clear = 0.7 + 5.95e-5 * mbar * torch.exp(1500.0 / tk)
    elif lw_type == 'PRATA':
        x = 46.5 * mbar / tk
        eps_clear = 1.0 - (1.0 + x) * torch.exp(
            -torch.sqrt(torch.clamp(1.2 + 3.0 * x, min=0.0)))
    else:
        raise ValueError('LW_TYPE %r is not one of %s'
                         % (lw_type, ', '.join(LW_TYPES)))
    if lw_cloud_deardorff:
        # Deardorff 1978: clouds radiate as black bodies over the fraction
        # of sky they cover.
        eps = tskc * 1.0 + (1.0 - tskc) * eps_clear
    else:
        eps = (1.0 + 0.17 * tskc * tskc) * eps_clear
    return eps * STEFAN_B * tk * tk * tk * tk


def _ydays(start, n):
    """A 1-based day-of-year sequence of length n beginning at `start`."""
    return [(start + datetime.timedelta(days=i)).timetuple().tm_yday
            for i in range(n)]


def _at_steps(hourly, steps_per_day, offset_hours, nsteps):
    """One local hourly series read at the instants the model steps begin
    (MetSim samples air temperature, vapour pressure and so pressure and
    longwave at the start of each step, not over it)."""
    per_step = 24 // steps_per_day
    return hourly[..., offset_hours:offset_hours + nsteps * per_step:per_step]


def _to_steps(hourly, steps_per_day, offset_hours, nsteps, reduce='mean'):
    """One local hourly series [..., nlocal*24] read out as model steps.

    `offset_hours` is where global hour 0 sits in the local array; the hours
    inside each sub-step are averaged (summed for a flux per step).
    """
    per_step = 24 // steps_per_day
    take = hourly[..., offset_hours:offset_hours + nsteps * per_step]
    take = take.reshape(*take.shape[:-1], nsteps, per_step)
    return take.sum(-1) if reduce == 'sum' else take.mean(-1)


def daily_to_subdaily(days, start, lat, lng, elev, steps_per_day=24,
                      time_zone_lng=None, slope=0.0, aspect=0.0,
                      ehoriz=0.0, whoriz=0.0, start_hour=0,
                      vp_interp=True, plapse=True,
                      min_wind_speed=MIN_WIND_SPEED, lw_type='PRATA',
                      lw_cloud_deardorff=True, vp_iter=None,
                      swe_corr=False, sw_prec_thresh=0.0,
                      annual_prcp_mm=None, geom=None, carbon=False,
                      catm_ppm=CATM_CURRENT_PPM, start_day_of_year=None,
                      dtype=torch.float64, device=None):
    """Daily forcings in, VIC 5's sub-daily forcing set out.

    `days` is a dict of `[cell, nday]` tensors. `tmax`, `tmin` and `prcp`
    are required -- degrees C and millimetres per day. `wind` (m/s) is used
    if present and replaced by VIC's 1.5 m/s default if not. `vp` (Pa) and
    `shortwave` (daily-mean W/m2) are passed to MTCLIM as observations if
    present: an observed vapour pressure replaces the humidity estimate and
    an observed shortwave replaces the radiation estimate (the cloud
    fraction is then inferred from it).

    `slope`, `aspect`, `ehoriz`, `whoriz`, `swe_corr` and `vp_iter` belong to
    VIC 4.2.d's terrain and snow corrections, which MetSim's MTCLIM does not
    have; they are accepted only at their neutral values (a flat surface, no
    snow correction, no iteration setting) and anything else is refused.
    `annual_prcp_mm` is accepted and unused: the precipitation total is the
    trailing 90 days of the record itself.

    `start` is a `datetime.date`, `lat`/`lng`/`elev` are scalars or
    `[cell]`, and the returned dict holds `[cell, nday * steps_per_day]`
    tensors keyed by the names `torchvic`'s readers use: tair, prcp, press,
    shortwave, longwave, vp, wind.

    The whole-hour part of the time-zone offset must be the same for every
    cell in one call, because it changes how many days the local record has.
    Group cells by offset and call once per group; they do not interact.
    """
    if (slope or aspect or ehoriz or whoriz or swe_corr
            or vp_iter is not None):
        raise ValueError('slope, aspect, horizon, snow-cover and vp_iter '
                         'options are VIC 4.2.d corrections that MetSim\'s '
                         'MTCLIM (this module\'s basis) does not have; '
                         'only a flat surface is supported')
    if steps_per_day not in (1, 2, 3, 4, 6, 8, 12, 24):
        raise ValueError('steps_per_day must divide 24; got %r'
                         % (steps_per_day,))

    def T(v):
        return torch.as_tensor(v, dtype=dtype, device=device)

    tmax, tmin = T(days['tmax']), T(days['tmin'])
    prcp = T(days['prcp'])
    if tmax.dim() == 1:
        tmax, tmin, prcp = tmax[None], tmin[None], prcp[None]
    ndays = tmax.shape[-1]

    fracs, wholes = zip(*[local_time_offset(x, z) for x, z in zip(
        torch.as_tensor(lng).reshape(-1).tolist(),
        (torch.as_tensor(time_zone_lng).reshape(-1).tolist()
         if time_zone_lng is not None
         else torch.as_tensor(lng).reshape(-1).tolist()))])
    if len(set(wholes)) != 1:
        raise ValueError('the whole-hour time-zone offsets differ across '
                         'cells (%s); one call must cover one offset, '
                         'because it decides how many local days the record '
                         'has' % sorted(set(wholes)))
    off_int = wholes[0]
    if len(set('%.12g' % f for f in fracs)) != 1:
        raise ValueError('the fractional time-zone offsets differ across '
                         'cells; call once per distinct longitude')
    off_frac = fracs[0]

    # An offset record straddles local midnight, so MTCLIM runs over one
    # extra day, and the padding day repeats the record's own end.
    nlocal = ndays + (1 if off_int else 0)
    shift = 1 if off_int > 0 else 0
    idx = torch.tensor([min(max(i - shift, 0), ndays - 1)
                        for i in range(nlocal)], dtype=torch.long,
                       device=tmax.device)
    l_tmax = tmax.index_select(-1, idx)
    l_tmin = tmin.index_select(-1, idx)
    l_prcp = prcp.index_select(-1, idx)
    local_start = start - datetime.timedelta(days=shift)
    yday = _ydays(local_start, nlocal)

    vp_in = (T(days['vp']).index_select(-1, idx) if 'vp' in days else None)
    sw_in = (T(days['shortwave']).index_select(-1, idx)
             if 'shortwave' in days else None)

    mt = mtclim.disaggregate(yday, l_prcp, l_tmax, l_tmin, lat, elev,
                             hour_offset=off_frac, vp_in=vp_in,
                             srad_in=sw_in, geom=geom,
                             step_minutes=60.0 * (24 // steps_per_day),
                             lw_cloud_deardorff=lw_cloud_deardorff,
                             sw_prec_thresh=sw_prec_thresh, dtype=dtype,
                             device=device)

    # Where global hour zero sits in the local hourly array.
    base = start_hour - off_int
    offset_hours = base + (24 if base < 0 else 0)
    nsteps = ndays * steps_per_day
    per_step = 24 // steps_per_day

    def hourly(x):
        return x.reshape(*x.shape[:-2], nlocal * 24)

    def daily_as_hourly(x):
        return x.unsqueeze(-1).expand(*x.shape, 24).reshape(
            *x.shape[:-1], nlocal * 24)

    tair = _at_steps(hourly(mt['tair'].reshape(-1, nlocal, 24)),
                     steps_per_day, offset_hours, nsteps)
    swdown = _to_steps(hourly(mt['hourlyrad'].reshape(-1, nlocal, 24)),
                       steps_per_day, offset_hours, nsteps)

    # A daily total spread evenly over the day's model steps (MetSim's
    # UNIFORM precipitation).
    prec = _to_steps(daily_as_hourly(l_prcp / 24.0), steps_per_day,
                     offset_hours, nsteps, reduce='sum')

    # Wind held constant through the day (metsim.disaggregate.wind), floored
    # at `min_wind_speed` at every step length.
    if 'wind' in days:
        wind_d = T(days['wind'])
        if wind_d.dim() == 1:
            wind_d = wind_d[None]
        wind_d = wind_d.index_select(-1, idx)
    else:
        wind_d = torch.full_like(l_prcp, DEFAULT_WIND_SPEED)
    wind = _to_steps(daily_as_hourly(wind_d), steps_per_day, offset_hours,
                     nsteps)
    wind = torch.clamp(wind, min=min_wind_speed)

    # Daily vapour pressure placed at the time of the daily minimum
    # temperature and interpolated linearly between those times
    # (metsim.disaggregate.vapor_pressure), or held flat.
    vp_hourly = _vp_to_hourly(mt['s_vp'].reshape(-1, nlocal),
                              mt['t_tmin'].reshape(-1, nlocal), vp_interp)
    vp = _at_steps(vp_hourly, steps_per_day, offset_hours, nsteps)

    # Pressure from elevation through the temperature of the air column
    # (metsim.disaggregate.pressure), or from elevation alone.
    elev_t = T(elev).reshape(-1, 1)
    if plapse:
        temp_corr = mtclim.KELVIN + tair - 0.5 * elev_t * mtclim.LAPSE_RATE
        press = mtclim.P_STD * torch.exp(
            -(elev_t * mtclim.G_STD) / (mtclim.R_DRY * temp_corr))
    else:
        press = mtclim.atm_pres(elev_t).expand_as(tair).contiguous()

    # A vapour pressure above saturation is pulled back to it, and the
    # longwave below sees the pulled-back value (metsim.disaggregate
    # .vapor_pressure).
    svp_t = mtclim.svp(tair)
    vp = torch.where(svp_t - vp < 0.0, svp_t, vp)

    # Cloud fraction held constant over the day (metsim.disaggregate.tskc),
    # then the longwave.
    tskc = _to_steps(daily_as_hourly(mt['s_tskc'].reshape(-1, nlocal)),
                     steps_per_day, offset_hours, nsteps)
    lwdown = calc_longwave(tskc, tair, vp, lw_type=lw_type,
                           lw_cloud_deardorff=lw_cloud_deardorff)

    out = dict(tair=tair, prcp=prec, press=press, shortwave=swdown,
               longwave=lwdown, vp=vp, wind=wind, tskc=tskc,
               vpd=torch.clamp(svp_t - vp, min=0.0), mtclim=mt)
    if carbon:
        out.update(_carbon_forcings(mt, swdown, start, lat, lng,
                                    time_zone_lng, off_int, ndays,
                                    steps_per_day, nlocal, offset_hours,
                                    catm_ppm, dtype))
    return out


def _carbon_forcings(mt, swdown, start, lat, lng, time_zone_lng, off_int,
                     ndays, steps_per_day, nlocal, offset_hours, catm_ppm,
                     dtype):
    """The four extra fields VIC reads when CARBON is on.

    VIC 5 demands CATM, FDIR and PAR as forcing columns and stops without
    them, which is why a CARBON run needs a dataset almost nobody has.
    MetSim has none of these, so they are derived here:

        fdir    the direct fraction of the shortwave, one minus Erbs et al.'s
                daily diffuse fraction at the day's clearness index
                (mtclim._erbs_diffuse_fraction), held over the day
        par     SW2PAR times the shortwave
        Catm    a single global concentration as a mixing ratio
        coszen  computed, never read from a file

    `coszen` follows VIC 5's convention: the zenith angle at the start
    second of the record. The physics that consumes it is VIC 5's, so it
    gets VIC 5's value.
    """
    from ..core import atmos as _atmos

    nsteps = ndays * steps_per_day
    dev = swdown.device
    fdir_daily = mt['s_fdir'].reshape(-1, nlocal)
    fdir = _to_steps(fdir_daily.unsqueeze(-1).expand(*fdir_daily.shape, 24)
                     .reshape(*fdir_daily.shape[:-1], nlocal * 24),
                     steps_per_day, offset_hours, nsteps)
    par = SW2PAR * swdown
    catm = torch.full_like(swdown, float(catm_ppm) * 1e-6)

    step_s = 86400 // steps_per_day
    doy = torch.tensor(
        [(start + datetime.timedelta(days=i)).timetuple().tm_yday
         for i in range(ndays) for _ in range(steps_per_day)],
        dtype=dtype, device=dev)
    sec = torch.tensor([(j * step_s) for _ in range(ndays)
                        for j in range(steps_per_day)], dtype=dtype,
                       device=dev)
    tz = lng if time_zone_lng is None else time_zone_lng
    cz = _atmos.coszen(torch.as_tensor(lat, dtype=dtype, device=dev)
                       .reshape(-1, 1),
                       torch.as_tensor(lng, dtype=dtype, device=dev)
                       .reshape(-1, 1),
                       torch.as_tensor(tz, dtype=dtype, device=dev)
                       .reshape(-1, 1),
                       doy.reshape(1, -1), sec.reshape(1, -1))
    return dict(fdir=fdir, par=par, Catm=catm,
                coszen=cz.expand_as(swdown).contiguous())


def _vp_to_hourly(daily_vp, t_tmin, interp):
    """Daily vapour pressure spread over the hours.
    metsim.disaggregate.vapor_pressure.

    The daily value is placed at the time of the daily minimum temperature
    -- the coldest time, where the air is nearest saturation -- and a
    straight line runs between consecutive minima, continued with the end
    segments' slopes before the first and after the last (scipy's
    `interp1d(fill_value='extrapolate')`, which MetSim calls). The segments
    have different lengths because the minimum moves with sunrise.

    `t_tmin` is minutes after the start of each day, [..., nlocal]. With
    `interp` false, or a single day, the value is held flat.
    """
    nlocal = daily_vp.shape[-1]
    dev, dtype = daily_vp.device, daily_vp.dtype
    if not interp or nlocal == 1:
        return daily_vp.unsqueeze(-1).expand(*daily_vp.shape, 24).reshape(
            *daily_vp.shape[:-1], nlocal * 24)

    day0 = torch.arange(nlocal, device=dev, dtype=dtype) * mtclim.MIN_PER_DAY
    x = (t_tmin.to(dtype) + day0).contiguous()
    xq = (torch.arange(nlocal * 24, device=dev, dtype=dtype) * 60.0).expand(
        *x.shape[:-1], nlocal * 24).contiguous()
    k = torch.clamp(torch.searchsorted(x, xq, right=True) - 1, 0, nlocal - 2)
    x0, x1 = torch.gather(x, -1, k), torch.gather(x, -1, k + 1)
    y0, y1 = (torch.gather(daily_vp, -1, k),
              torch.gather(daily_vp, -1, k + 1))
    return y0 + (y1 - y0) * (xq - x0) / (x1 - x0)


def expand_forcing(fields, stamps, steps_per_day, lat, lng, elev,
                   time_zone_lng=None, log=None, **kw):
    """A reader's daily output expanded to a sub-daily forcing set.

    `ClassicForcing.read` and `ImageForcing.read` both return
    `{name: [time, cell]}` plus a `stamps` array, and a daily file gives one
    record a day. This turns that into the `[time, cell]` fields and stamps a
    sub-daily file would have produced, so the rest of the run is unchanged:
    the caller does not learn that the forcing was daily.

    Requires `tmax`, `tmin` and `prcp`; uses `wind`, `vp` and `shortwave` if
    the file has them. Anything else the reader produced is carried through
    by repeating each day's value across its steps, which is what
    is done with a daily forcing of any other type.
    """
    import datetime as _dt

    def say(msg):
        if log is None:
            print(msg)
        else:
            log(msg)

    need = [k for k in ('tmax', 'tmin', 'prcp') if k not in fields]
    if need:
        raise ValueError('a daily forcing needs TMAX, TMIN and PREC to be '
                         'disaggregated; missing %s'
                         % ', '.join(sorted(need)))
    start = stamps[0].astype('datetime64[s]').astype(_dt.datetime).date()
    ndays = len(stamps)

    days = {k: fields[k].transpose(0, 1).contiguous()
            for k in ('tmax', 'tmin', 'prcp', 'wind', 'vp', 'shortwave')
            if k in fields}
    ref = days['tmax']
    out = daily_to_subdaily(days, start, lat, lng, elev,
                            steps_per_day=steps_per_day,
                            time_zone_lng=time_zone_lng,
                            dtype=ref.dtype, device=ref.device, **kw)

    sub = {}
    for name in ('tair', 'prcp', 'press', 'shortwave', 'longwave', 'vp',
                 'wind', 'fdir', 'par', 'Catm', 'coszen', 'tskc'):
        if name in out:
            sub[name] = out[name].transpose(0, 1).contiguous()
    # An observation beats a parameterisation.
    #
    # MTCLIM estimates longwave and surface pressure, and until now its
    # estimate overwrote a measured one: `out` was copied in above for
    # every name it produces, so a file carrying LWDOWN and PRESSURE had
    # both silently discarded. `vp` and `shortwave` are different -- they
    # are INPUTS to the chain, so what comes back is the measured daily
    # value distributed over the day, which is what should be used.
    #
    # Held constant across the day, like any other daily field.
    measured = []
    for name in ('longwave', 'press'):
        if name in fields:
            sub[name] = fields[name].repeat_interleave(steps_per_day, dim=0)
            measured.append(name)

    # Any other daily field is held constant over its day.
    for name, v in fields.items():
        if name in sub or name in ('tmax', 'tmin'):
            continue
        sub[name] = v.repeat_interleave(steps_per_day, dim=0)

    step = 86400 // steps_per_day
    import numpy as _np
    new_stamps = (stamps[0] + _np.arange(ndays * steps_per_day)
                  * _np.timedelta64(step, 's')).astype('datetime64[ns]')
    # Say what was actually read rather than a blanket caveat. The old
    # message claimed shortwave, longwave and vapour pressure were all
    # estimated even when the file supplied them, which is the opposite of
    # what matters: driving VIC from daily data costs +33% evaporation
    # when vapour pressure is estimated and -3% when it is read.
    est = [n for n in ('shortwave', 'vp', 'longwave', 'press')
           if n not in fields]
    say('  daily forcing disaggregated to %d steps/day over %d days by the '
        'MetSim MTCLIM chain (torchvic.forcing.mtclim)' % (steps_per_day, ndays))
    read_in = sorted(set(fields) & {'shortwave', 'vp', 'longwave', 'press',
                                    'wind'})
    say('    read from the file: %s' % (', '.join(read_in) or 'none'))
    if measured:
        say('    %s held at the measured daily value rather than MTCLIM\'s '
            'estimate' % ', '.join(measured))
    say('    estimated by MTCLIM: %s' % (', '.join(est) or 'none'))
    return sub, new_stamps


#: MTCLIM's device memory per cell (expand_forcing): solar_geometry's
#: [366, 2880] tiny_radfract with to_hourly's rolled copy and its day-chunk
#: working arrays, plus per day of the window about thirty [24]-hour rows.
#: Measured peak on the ISC daily run (911 days, float64): 15.4 MB/cell.
_MT_FIXED_BYTES = 10 * 2 ** 20
_MT_DAY_BYTES = 30 * 24 * 8


def auto_cell_chunk(n_days, device, dtype=torch.float64, cap=4096,
                    floor=64, use=0.8):
    """The largest power-of-two block of cells, at most `cap`, whose
    MTCLIM (expand_forcing over `n_days`) fits in `use` of the device's
    free memory. On a CPU, `cap`. Every cell is computed alone, so the
    block size changes the memory and the kernels' shapes only."""
    device = torch.device(device)
    if device.type != 'cuda':
        return int(cap)
    free, _ = torch.cuda.mem_get_info(device)
    scale = torch.tensor([], dtype=dtype).element_size() / 8.0
    per_cell = (_MT_FIXED_BYTES + n_days * _MT_DAY_BYTES) * scale
    n = int(cap)
    while n > floor and n * per_cell > use * free:
        n //= 2
    return max(n, int(floor))
