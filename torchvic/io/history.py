"""VIC 5's history output -- OUTFILE, AGGFREQ, OUTVAR, HISTFREQ -- for TorchVIC.

TorchVIC steps at the model step and `basin.simulate` returns every output at
that step: eight three-hourly values a day. VIC 5 does not write what it
steps. Its image driver aggregates each output stream over the period the
global file's AGGFREQ names, and each variable by an aggregation type: a
water flux is summed (OUT_EVAP of a day is the day's evaporation, in mm), a
store is taken at the end of the period (OUT_SOIL_MOIST, OUT_SWE), and
everything else is averaged (OUT_AIR_TEMP, OUT_LATENT). This module does the
same, from the same global file, so a user who asks VIC for daily output
gets VIC's daily output from TorchVIC.

Every rule below is read from VIC 5's C and was measured against the
compiled image driver (reference/VIC-5.1.0; the aggregation code is the same
in 5.0.1). tests/test_output_aggregation.py holds both.

**Streams** (parse_output_info.c). Each `OUTFILE prefix` opens a stream; the
AGGFREQ, HISTFREQ, OUT_FORMAT, COMPRESS and OUTVAR lines after it, up to the
next OUTFILE, belong to it. VIC refuses any of them before the first OUTFILE.
An OUTVAR line is `OUTVAR name [format [type [mult [aggtype]]]]`, positional,
`*` for the default. The format is for the classic driver's ASCII files and
is not read here; `mult` is parsed by VIC's image driver and never applied
(vic_write.c writes `aggdata` as it is) -- measured, a mult of 1000 changes
nothing -- so it is not applied here either.

**When a period closes** (alarms.c). The aggregation alarm is anchored at the
first model step, not at calendar midnight:

* NSTEPS n: after every n steps.
* NSECONDS/NMINUTES/NHOURS/NDAYS n: the alarm date starts at (first step
  - dt) + n units and the period closes on the step whose start equals it;
  then the next alarm is that step's start + n units. With the run starting
  at 00:00 this is the calendar day; with STARTSEC 10800 it is 03:00-03:00.
  A period that is not a whole number of steps (NHOURS 2 at a three-hour
  step) never closes: VIC writes nothing at all (measured). This module
  refuses it instead.
* NMONTHS n: the same, in months, by shifting a step forward, adding the
  months to the day of the month, and shifting back (reset_alarm): a run
  from 15 January writes 15 Jan - 14 Feb, 15 Feb - 14 Mar, ... (measured).
  Month arithmetic is VIC's time_delta: a date the month lacks (31 Jan + 1
  month) is refused, as VIC log_errs on it.
* NYEARS n: whole years from the first step; a start on 29 February is
  refused (time_delta.c's own message).
* DATE yyyy-mm-dd: one period, from the first step up to and including the
  step that starts at 00:00 of that date; nothing after it is written
  (measured: 2008-01-01 00:00 .. 2008-01-04 03:00).
* NEVER: no record.
* END: VIC 5's image driver never raises this alarm -- `set_alarm` leaves
  the date it compares against on the placeholder from `setup_stream`, so
  VIC writes no file at all (measured). The documented meaning is one
  record over the whole run, and that is what this module writes, saying
  that it differs from what VIC would have written.
* An OUTFILE with no AGGFREQ: the same placeholder date, and VIC again
  writes nothing (measured). This module writes that stream at every model
  step (NSTEPS 1) -- what TorchVIC always wrote -- and says so.

**A partial last period is dropped.** A record is written only when its
alarm is raised, so ten days at NDAYS 7 give one record, and a 10-day run
at NMONTHS 1 gives none (measured).

**What each AGG_TYPE does** (agg_data.c, agg_stream_data):

* SUM -- the sum over the period's steps, accumulated from 0 in step order.
* AVG -- that sum divided by the number of steps.
* END -- the value at the last step of the period.
* BEG -- the value at the first step of the period.
* MAX / MIN -- the running max/min, **started from 0**: `aggdata` is zeroed
  by calloc and by reset_stream, and agg_stream_data takes max(aggdata,
  value). VIC's AGG_TYPE_MAX of a quantity that stays below zero is 0, and
  its AGG_TYPE_MIN of one that stays above zero -- evaporation, soil
  moisture, a summer air temperature -- is 0 (measured on Walnut Gulch:
  OUT_AIR_TEMP's daily MIN reads 0.0 on days whose coldest step was
  +9.7 C). `extremes='vic'` (the default) writes what VIC writes and
  counts the records the zero start decided; `extremes='true'` writes the
  period's own extremes.

**The default AGG_TYPE of each variable** (get_default_outvar_aggtype,
vic_history.c): END for the states in DEFAULT_END, SUM for the water fluxes
in DEFAULT_SUM, AVG for every other variable. The two sets below are VIC's
switch statement written out; the test re-reads the C and fails if they
drift.

**What is written** (vic_write.c, vic_init_output.c). One record per closed
period. `time` is the start of the period's first step ("timestamp is the
beginning of the aggregation window"); `time_bnds` is [that, the start of
its last step + dt]. Times are `<OUT_TIME_UNITS> since 0001-01-01 00:00:00`
in the global file's CALENDAR, as VIC writes them. Variables are
(time, y, x), with nlayer / node / snow_band / front between time and y
for the layered ones, filled outside the active cells, and carry VIC's
long_name, standard_name, units, description and cell_methods. OUT_TYPE
defaults to float (VIC: OUT_TYPE_FLOAT, so a history value is a float32
rounding of the aggregated double); OUT_TYPE_DOUBLE and OUT_TYPE_INT are
honoured, and the types VIC's image driver refuses (SINT, USINT, CHAR) are
refused. A file holds the records up to its HISTFREQ alarm (default: the
whole run) and is named `prefix.YYYY-MM-DD.nc` (NDAYS), `prefix.YYYY-MM.nc`
(NMONTHS), `prefix.YYYY.nc` (NYEARS) or `prefix.YYYY-MM-DD-SSSSS.nc`, from
its first record. OUT_FORMAT NETCDF4_CLASSIC, VIC's default, is written as
netCDF-3 classic, because that is what VIC's get_nc_mode asks the library
for (NC_CLASSIC_MODEL without NC_NETCDF4; measured).

A variable TorchVIC does not compute (OUTVAR_SOURCES has no entry, or the
configuration did not produce it -- OUT_LATENT with the energy balance off)
is left out of the file and named; a name that is not a VIC output at all
is refused, as VIC refuses it.

**VIC 4.2.d's convention** (`convention='vic4.2d'`; the section "VIC 4.2.d"
below states every rule with its line in VIC 4.2.d's C). Selected by
itself when the global file is a VIC 4.2.d one (TIME_STEP / OUT_STEP /
N_OUTFILES), or by `--agg-convention vic4.2d`. The default aggregation of
every variable the two versions share is the same; what differs is how a
period is formed and written: OUT_STEP hours (<= 24) counted in steps from
the first step of the run, AVG accumulated as the sum of x/n rather than
the sum divided by n, OUT_AERO_RESIST as 1/(mean OUT_AERO_COND), only AVG,
END and SUM (no BEG, MAX or MIN), the record stamped with the date of the
period's LAST step, SKIPYEAR, ALMA_OUTPUT's unit conversions after the
aggregation, and one ASCII (or BINARY) file per cell and OUTFILE,
`<prefix>_<lat>_<lon>`, instead of netCDF. The default, `'vic5'`, is
everything above, unchanged.
"""
import datetime as _dt
import json
import math
import os
import re
import struct
import warnings

import numpy as np
import torch

from .global_param import _announcer

AGG_TYPES = ('AVG', 'BEG', 'END', 'MAX', 'MIN', 'SUM')
FREQS = ('NEVER', 'NSTEPS', 'NSECONDS', 'NMINUTES', 'NHOURS', 'NDAYS',
         'NMONTHS', 'NYEARS', 'DATE', 'END')
#: Seconds in one unit of the uniform frequencies (time_delta, vic_time.c).
_UNIT_SECONDS = {'NSECONDS': 1, 'NMINUTES': 60, 'NHOURS': 3600,
                 'NDAYS': 86400}

#: get_default_outvar_aggtype (vic_history.c 5.1.0:209-303, identical in
#: 5.0.1): the variables VIC takes at the END of a period ...
DEFAULT_END = frozenset((
    'OUT_ASAT', 'OUT_LAKE_AREA_FRAC', 'OUT_LAKE_DEPTH', 'OUT_LAKE_ICE',
    'OUT_LAKE_ICE_FRACT', 'OUT_LAKE_ICE_HEIGHT', 'OUT_LAKE_MOIST',
    'OUT_LAKE_SURF_AREA', 'OUT_LAKE_SWE', 'OUT_LAKE_SWE_V', 'OUT_LAKE_VOLUME',
    'OUT_ROOTMOIST', 'OUT_SMFROZFRAC', 'OUT_SMLIQFRAC', 'OUT_SNOW_CANOPY',
    'OUT_SNOW_COVER', 'OUT_SNOW_DEPTH', 'OUT_SOIL_ICE', 'OUT_SOIL_LIQ',
    'OUT_SOIL_MOIST', 'OUT_SOIL_WET', 'OUT_SURFSTOR', 'OUT_SURF_FROST_FRAC',
    'OUT_SWE', 'OUT_WDEW', 'OUT_ZWT', 'OUT_ZWT_LUMPED',
    'OUT_SNOW_CANOPY_BAND', 'OUT_SNOW_COVER_BAND', 'OUT_SNOW_DEPTH_BAND',
    'OUT_SWE_BAND'))
#: ... the ones it SUMS over the period ...
DEFAULT_SUM = frozenset((
    'OUT_BASEFLOW', 'OUT_DELINTERCEPT', 'OUT_DELSOILMOIST', 'OUT_DELSWE',
    'OUT_DELSURFSTOR', 'OUT_EVAP', 'OUT_EVAP_BARE', 'OUT_EVAP_CANOP',
    'OUT_INFLOW', 'OUT_LAKE_BF_IN', 'OUT_LAKE_BF_IN_V', 'OUT_LAKE_BF_OUT',
    'OUT_LAKE_BF_OUT_V', 'OUT_LAKE_CHAN_IN', 'OUT_LAKE_CHAN_IN_V',
    'OUT_LAKE_CHAN_OUT', 'OUT_LAKE_CHAN_OUT_V', 'OUT_LAKE_DSTOR',
    'OUT_LAKE_DSTOR_V', 'OUT_LAKE_DSWE', 'OUT_LAKE_DSWE_V', 'OUT_LAKE_EVAP',
    'OUT_LAKE_EVAP_V', 'OUT_LAKE_PREC_V', 'OUT_LAKE_RCHRG',
    'OUT_LAKE_RCHRG_V', 'OUT_LAKE_RO_IN', 'OUT_LAKE_RO_IN_V',
    'OUT_LAKE_VAPFLX', 'OUT_LAKE_VAPFLX_V', 'OUT_PET', 'OUT_PREC',
    'OUT_RAINF', 'OUT_REFREEZE', 'OUT_RUNOFF', 'OUT_SNOWF',
    'OUT_SUB_BLOWING', 'OUT_SUB_CANOP', 'OUT_SUB_SNOW', 'OUT_SUB_SURFACE',
    'OUT_TRANSP_VEG', 'OUT_DELTACC_BAND', 'OUT_SNOW_MELT',
    'OUT_SNOWT_FBFLAG', 'OUT_SOILT_FBFLAG', 'OUT_SURFT_FBFLAG',
    'OUT_TCAN_FBFLAG', 'OUT_TFOL_FBFLAG'))
#: ... and every other variable is averaged (the switch's default branch).
DEFAULT_OTHER = 'AVG'


def default_agg(name, convention='vic5'):
    """VIC's default AGG_TYPE for the output variable `name` (OUT_*):
    VIC 5's (get_default_outvar_aggtype), or VIC 4.2.d's
    (create_output_list) under convention='vic4.2d'."""
    if convention != 'vic5' and check_convention(convention) == 'vic4.2d':
        return default_agg_42d(name)
    if name in DEFAULT_END:
        return 'END'
    if name in DEFAULT_SUM:
        return 'SUM'
    return DEFAULT_OTHER


#: cell_method_from_agg_type (input_tools.c).
CELL_METHODS = {'AVG': 'time: mean', 'MAX': 'time: maximum',
                'MIN': 'time: minimum', 'SUM': 'time: sum',
                'END': 'time: end', 'BEG': 'time: beg'}

#: VIC output variable -> the basin.simulate output that carries it, each row
#: scored against VIC's own history (tests/test_output_aggregation.py).
OUTVAR_SOURCES = {
    'OUT_EVAP': 'et', 'OUT_TRANSP_VEG': 'transp',
    'OUT_EVAP_CANOP': 'evap_canop', 'OUT_RUNOFF': 'runoff',
    'OUT_BASEFLOW': 'baseflow', 'OUT_SWE': 'swe', 'OUT_SUB_SNOW': 'sub_snow',
    'OUT_SWNET': 'eb_net_short', 'OUT_LWNET': 'eb_net_long',
    'OUT_R_NET': 'eb_net_rad', 'OUT_LATENT': 'eb_latent',
    'OUT_SENSIBLE': 'eb_sensible', 'OUT_GRND_FLUX': 'eb_grnd_flux',
    'OUT_DELTAH': 'eb_delta_h', 'OUT_SURF_TEMP': 'eb_t_surf',
    'OUT_AERO_RESIST': 'eb_aero_resist', 'OUT_ENERGY_ERROR': 'eb_energy_error',
    'OUT_SOIL_TEMP': 'eb_layer_temp',
    'OUT_RAINF': 'rainf', 'OUT_SNOWF': 'snowf',
    'OUT_SNOW_MELT': 'snow_melt', 'OUT_INFLOW': 'inflow',
    'OUT_WDEW': 'wdew', 'OUT_SNOW_COVER': 'snow_cover',
    'OUT_SALBEDO': 'salbedo', 'OUT_SNOW_SURF_TEMP': 'snow_surf_temp',
    'OUT_SNOW_PACK_TEMP': 'snow_pack_temp',
    'OUT_SOIL_LIQ': 'soil_liq', 'OUT_SOIL_ICE': 'soil_ice',
    'OUT_SOIL_WET': 'soil_wet', 'OUT_ROOTMOIST': 'rootmoist',
    'OUT_ASAT': 'asat',
    'OUT_DELSOILMOIST': 'delsoilmoist', 'OUT_DELSWE': 'delswe',
    'OUT_DELINTERCEPT': 'delintercept', 'OUT_WATER_ERROR': 'water_error',
    'OUT_SURF_COND': 'surf_cond',
    'OUT_AIR_TEMP': 'air_temp', 'OUT_SWDOWN': 'swdown',
    'OUT_LWDOWN': 'lwdown', 'OUT_PRESSURE': 'pressure',
    'OUT_IN_LONG': 'in_long',
    'OUT_VP': 'vp', 'OUT_VPD': 'vpd', 'OUT_WIND': 'wind',
    'OUT_REL_HUMID': 'rel_humid', 'OUT_DENSITY': 'density',
    'OUT_LAI': 'lai', 'OUT_FCANOPY': 'fcanopy',
    'OUT_ALBEDO': 'albedo',
    'OUT_PET': 'pet', 'OUT_EVAP_BARE': 'evap_bare',
    'OUT_SNOW_DEPTH': 'snow_depth',
    'OUT_LATENT_SUB': 'eb_latent_sub', 'OUT_ADVECTION': 'eb_advection',
    'OUT_DELTACC': 'eb_deltacc', 'OUT_RFRZ_ENERGY': 'eb_rfrz_energy',
    'OUT_MELT_ENERGY': 'eb_melt_energy', 'OUT_REFREEZE': 'refreeze',
    'OUT_SNOW_FLUX': 'eb_snow_flux',
    'OUT_FUSION': 'eb_fusion', 'OUT_SNOW_CANOPY': 'snow_canopy',
    'OUT_SUB_CANOP': 'sub_canop',
    'OUT_AERO_COND': 'eb_aero_cond', 'OUT_RAD_TEMP': 'eb_rad_temp',
    'OUT_VEGT': 'eb_vegt', 'OUT_BARESOILT': 'eb_baresoilt',
    # Derived, and the two places this table departs from the scoreboard's:
    # put_data.c:691-703 makes OUT_SOIL_MOIST the TOTAL layer moisture and
    # OUT_SOIL_LIQ the total less the ice, and basin.simulate returns the
    # latter two (with FROZEN_SOIL off the ice is 0 and they coincide,
    # which is why the scoreboard can use soil_liq). OUT_PREC is
    # force->out_prec, which solve_snow.c:124-126 sets to snowfall +
    # rainfall, the two OUT_SNOWF and OUT_RAINF carry.
    'OUT_SOIL_MOIST': ('soil_liq', '+', 'soil_ice'),
    'OUT_PREC': ('rainf', '+', 'snowf'),
}
#: Added by out-vars (torchvic.model.outvars and the rows noted there): a
#: tuple (a, '|', b) is `a` where the run returned it and `b` otherwise
#: (OUT_ENERGY_ERROR is the solver or VIC's own combination with the energy
#: balance on, VIC's MISSING without it).
OUTVAR_SOURCES_OUTVARS = {
    'OUT_QAIR': 'qair', 'OUT_SMLIQFRAC': 'smliqfrac',
    'OUT_SMFROZFRAC': 'smfrozfrac', 'OUT_SOIL_LIQ_FRAC': 'soil_liq_frac',
    'OUT_SOIL_ICE_FRAC': 'soil_ice_frac', 'OUT_DELSURFSTOR': 'delsurfstor',
    'OUT_CATM': 'catm', 'OUT_FDIR': 'fdir', 'OUT_PAR': 'par',
    'OUT_ADV_SENS': 'eb_advected_sensible', 'OUT_SUB_BLOWING': 'sub_blowing',
    'OUT_ZWT': 'vic_zwt', 'OUT_ZWT_LUMPED': 'vic_zwt_lumped',
    'OUT_AERO_COND1': 'aero_cond1', 'OUT_AERO_COND2': 'aero_cond2',
    'OUT_AERO_RESIST1': 'aero_resist1', 'OUT_AERO_RESIST2': 'aero_resist2',
    'OUT_SURF_FROST_FRAC': 'surf_frost_frac', 'OUT_SUB_SURFACE': 'sub_surface',
    'OUT_FDEPTH': 'fdepth', 'OUT_TDEPTH': 'tdepth',
    'OUT_SWE_BAND': 'band_swe', 'OUT_SNOW_DEPTH_BAND': 'band_snow_depth',
    'OUT_SNOW_CANOPY_BAND': 'band_snow_canopy',
    'OUT_SNOW_MELT_BAND': 'band_snow_melt',
    'OUT_SNOW_COVER_BAND': 'band_snow_cover',
    'OUT_DELTACC_BAND': 'band_deltacc', 'OUT_SNOW_FLUX_BAND': 'band_snow_flux',
    'OUT_MELT_ENERGY_BAND': 'band_melt_energy',
    'OUT_SWNET_BAND': 'band_swnet', 'OUT_LWNET_BAND': 'band_lwnet',
    'OUT_ALBEDO_BAND': 'band_albedo', 'OUT_GRND_FLUX_BAND': 'band_grnd_flux',
    'OUT_ADV_SENS_BAND': 'band_adv_sens', 'OUT_ADVECTION_BAND': 'band_adv', 'OUT_RFRZ_ENERGY_BAND': 'band_rfz',
    'OUT_LATENT_BAND': 'band_lat', 'OUT_LATENT_SUB_BAND': 'band_lsub',
    'OUT_SENSIBLE_BAND': 'band_sen', 'OUT_SNOW_SURFT_BAND': 'band_surft',
    'OUT_SNOW_PACKT_BAND': 'band_packt', 'OUT_IN_LONG_BAND': 'band_in_long',
    'OUT_GPP': 'gpp', 'OUT_RAUT': 'raut', 'OUT_NPP': 'npp', 'OUT_APAR': 'apar',
    'OUT_LITTERFALL': 'litterfall', 'OUT_CLITTER': 'clitter',
    'OUT_CINTER': 'cinter', 'OUT_CSLOW': 'cslow', 'OUT_RHET': 'rhet',
    'OUT_NEE': 'nee',
    'OUT_DISCHARGE': 'discharge_rvic',
    'OUT_SURFT_FBFLAG': 'surft_fbflag', 'OUT_SNOWT_FBFLAG': 'snowt_fbflag',
    'OUT_TFOL_FBFLAG': 'tfol_fbflag', 'OUT_TCAN_FBFLAG': 'tcan_fbflag',
    'OUT_SOILT_FBFLAG': 'soilt_fbflag',
    'OUT_SOIL_TNODE': 'eb_tnode', 'OUT_SOIL_TNODE_WL': 'soil_tnode_wl',
    'OUT_ENERGY_ERROR': ('eb_energy_error', '|', 'energy_error_missing'),
}
OUTVAR_SOURCES.update(OUTVAR_SOURCES_OUTVARS)
# LAKES TRUE: put_data.c 5.1.0:305-400 (torchvic.model.lakecell.OUT_LAKE).
# A run without lakes returns none of these keys and writes none of them.
from ..model.lakecell import OUT_LAKE as _OUT_LAKE        # noqa: E402
OUTVAR_SOURCES.update(_OUT_LAKE)

#: TorchVIC outputs with no OUT_* counterpart, by how a period of them
#: aggregates. `discharge_mm` is the routed outflow as a depth over the
#: basin per step, so a day of it is a sum, as OUT_RUNOFF's is (VIC's own
#: OUT_DISCHARGE is a rate in m3 s-1 and is averaged); `evap_withheld` is a
#: depth per step too.
EXTRA_AGG = {'discharge_mm': 'SUM', 'evap_withheld': 'SUM',
             'eb_fallback': 'SUM'}

#: OUT_TYPE_* -> numpy dtype, for the three get_nc_dtype types VIC's image
#: driver can write (vic_init_output.c refuses NC_SHORT and NC_CHAR, and
#: vic_write.c has no branch for NC_UINT).
OUT_TYPES = {'OUT_TYPE_FLOAT': np.float32, 'OUT_TYPE_DOUBLE': np.float64,
             'OUT_TYPE_INT': np.int32}
OUT_TYPES_REFUSED = ('OUT_TYPE_SINT', 'OUT_TYPE_USINT', 'OUT_TYPE_CHAR')
#: netCDF's default fill values, which VIC writes (NC_FILL_*).
FILL = {np.float32: np.float32(9.969209968386869e36),
        np.float64: 9.969209968386869e36, np.int32: np.int32(-2147483647)}
#: OUT_FORMAT -> the file format VIC's get_nc_mode actually creates.
OUT_FORMATS = {'NETCDF3_CLASSIC': 'NETCDF3_CLASSIC',
               'NETCDF3_64BIT_OFFSET': 'NETCDF3_64BIT_OFFSET',
               'NETCDF4_CLASSIC': 'NETCDF3_CLASSIC',
               'NETCDF4': 'NETCDF4'}
#: COMPRESS TRUE (vic_driver_shared_all.h COMPRESSION_LVL_DEFAULT).
COMPRESSION_LVL_DEFAULT = 5
#: global_param.time_origin_str (vic_time.c:681).
TIME_ORIGIN = '0001-01-01 00:00:00'

#: VIC's output metadata as tools/extract_vic_outvars.py read it, shipped as
#: package data.
OUTVAR_INVENTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
    'vic5_outvars.json')
_META = None


def outvar_metadata():
    """{OUT_*: {long_name, standard_name, units, description, dim, ...}}
    from VIC's own tables, or {} when the inventory is not shipped."""
    global _META
    if _META is None:
        try:
            with open(OUTVAR_INVENTORY) as fh:
                _META = json.load(fh).get('variables', {})
        except (IOError, OSError, ValueError):
            _META = {}
    return _META


def is_vic_outvar(name):
    """Whether VIC 5 defines `name` (the OUT_* enum)."""
    meta = outvar_metadata()
    if meta:
        return name in meta
    return (name in DEFAULT_END or name in DEFAULT_SUM
            or name in OUTVAR_SOURCES)


# ---------------------------------------------------------------------------
# The global file's output section
# ---------------------------------------------------------------------------
def _atoi(s):
    """C's atoi: the leading integer of `s`, 0 if there is none."""
    m = re.match(r'\s*([+-]?\d+)', str(s))
    return int(m.group(1)) if m else 0


class Freq(object):
    """An AGGFREQ or HISTFREQ: `kind` from FREQS, `n`, and `date` for DATE."""

    def __init__(self, kind, n=1, date=None):
        kind = str(kind).upper()
        if kind not in FREQS:
            raise ValueError('unknown frequency %r (VIC: "Unknown frequency '
                             'flag found"); one of %s'
                             % (kind, ', '.join(FREQS)))
        if kind in _UNIT_SECONDS or kind in ('NSTEPS', 'NMONTHS', 'NYEARS'):
            if int(n) <= 0:
                raise ValueError('invalid n (%d) for %s (VIC: set_alarm '
                                 'log_err)' % (int(n), kind))
        if kind == 'DATE' and date is None:
            raise ValueError('DATE was given no date string (VIC: "AGGFREQ '
                             'was set to DATE but no date string was found")')
        self.kind, self.n, self.date = kind, int(n), date

    @classmethod
    def parse(cls, tokens, what='AGGFREQ'):
        """`AGGFREQ kind [value]` as parse_output_info.c reads it: the value
        is atoi'd (default 1), or a %Y-%m-%d date for DATE."""
        if not tokens:
            raise ValueError('No arguments found after %s' % what)
        kind = tokens[0].upper()
        if kind == 'DATE':
            if len(tokens) < 2:
                return cls(kind)             # raises VIC's message
            try:
                d = _dt.datetime.strptime(tokens[1], '%Y-%m-%d')
            except ValueError:
                raise ValueError('%s DATE %r is not a %%Y-%%m-%%d date'
                                 % (what, tokens[1]))
            return cls(kind, date=d)
        return cls(kind, _atoi(tokens[1]) if len(tokens) > 1 else 1)

    def __eq__(self, other):
        return (isinstance(other, Freq) and self.kind == other.kind
                and self.n == other.n and self.date == other.date)

    def __repr__(self):
        if self.kind == 'DATE':
            return 'DATE %s' % self.date.strftime('%Y-%m-%d')
        if self.kind in ('NEVER', 'END'):
            return self.kind
        return '%s %d' % (self.kind, self.n)


class OutVar(object):
    """One OUTVAR line: the variable, its output type and its aggregation."""

    def __init__(self, name, agg=None, out_type='OUT_TYPE_FLOAT', mult=None,
                 fmt=None):
        self.name = name
        self.agg_given = agg is not None
        self.agg = agg if agg is not None else default_agg(name)
        self.out_type = out_type
        self.mult = mult
        self.fmt = fmt

    def __repr__(self):
        return '<OutVar %s %s%s %s>' % (self.name, self.agg,
                                         '' if self.agg_given else ' (default)',
                                         self.out_type)


class Stream(object):
    """One OUTFILE and everything declared under it."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.agg = None            # None: the file gave no AGGFREQ
        self.hist = Freq('END')    # setup_stream's write alarm
        self.out_format = 'NETCDF4_CLASSIC'
        self.compress = 0
        self.vars = []

    def __repr__(self):
        return '<Stream %s AGGFREQ %s HISTFREQ %s, %d variables>' % (
            self.prefix, self.agg, self.hist, len(self.vars))


def _agg_type(tok):
    if tok in ('', '*'):
        return None
    t = tok.upper()
    if t.startswith('AGG_TYPE_') and t[len('AGG_TYPE_'):] in AGG_TYPES:
        return t[len('AGG_TYPE_'):]
    raise ValueError('Unknown aggregation type found: %s (VIC log_errs; one '
                     'of %s)' % (tok, ', '.join('AGG_TYPE_' + a
                                                for a in AGG_TYPES)))


def _out_type(tok):
    if tok in ('', '*'):
        return 'OUT_TYPE_FLOAT'                 # set_output_var's default
    t = tok.upper()
    if t in OUT_TYPES:
        return t
    if t in OUT_TYPES_REFUSED:
        raise ValueError(
            '%s: VIC 5\'s image driver cannot write it (vic_init_output.c '
            'log_errs on NC_SHORT and NC_CHAR, and vic_write.c has no NC_UINT '
            'branch); use OUT_TYPE_FLOAT, OUT_TYPE_DOUBLE or OUT_TYPE_INT' % t)
    raise ValueError('Unknown out type found: %s' % tok)


def parse_streams(gp, log=None):
    """The global file's output streams, as parse_output_info.c reads them.

    `gp` is a vic_image.GlobalParam. Returns [] when the file has no
    OUTFILE: VIC then writes its default daily streams (set_output_defaults),
    which TorchVIC does not reproduce -- its outputs stay at the model step.
    """
    say = _announcer(log)
    streams, cur = [], None
    for key, rest in getattr(gp, 'lines', []):
        if key == 'OUTFILE':
            if not rest:
                raise ValueError('Invalid specification for OUTFILE')
            cur = Stream(rest[0])
            streams.append(cur)
            continue
        if key not in ('AGGFREQ', 'HISTFREQ', 'COMPRESS', 'OUT_FORMAT',
                       'OUTVAR'):
            continue
        if cur is None:
            raise ValueError('Error in global param file: "OUTFILE" must be '
                             'specified before you can specify "%s".' % key)
        if key == 'AGGFREQ':
            cur.agg = Freq.parse(rest, 'AGGFREQ')
        elif key == 'HISTFREQ':
            cur.hist = Freq.parse(rest, 'HISTFREQ')
        elif key == 'COMPRESS':
            flag = (rest[0] if rest else '').upper()
            cur.compress = (COMPRESSION_LVL_DEFAULT if flag == 'TRUE' else
                            0 if flag == 'FALSE' else _atoi(flag))
        elif key == 'OUT_FORMAT':
            fmt = (rest[0] if rest else '').upper()
            if fmt not in OUT_FORMATS:
                raise ValueError('Image driver file format must be a valid '
                                 'NETCDF format, not %r (%s)'
                                 % (fmt, ', '.join(OUT_FORMATS)))
            cur.out_format = fmt
        else:                                           # OUTVAR
            if not rest:
                raise ValueError('OUTVAR specified but no variable was listed')
            tok = list(rest[:5]) + [''] * (5 - len(rest[:5]))
            name, fmt, typ, mult, agg = tok
            if not is_vic_outvar(name):
                raise ValueError(
                    'set_output_var: "%s" was not found in the list of '
                    'supported output variable names (VIC log_errs)' % name)
            for pos, t in (('format', fmt), ('type', typ), ('mult', mult)):
                if t.upper().startswith('AGG_TYPE_'):
                    say('OUTVAR %s: %s is in the %s position; VIC reads the '
                        'aggregation type only as the fifth token '
                        '(OUTVAR name format type mult aggtype), so this '
                        'line keeps the default %s. Write "OUTVAR %s * * * '
                        '%s"' % (name, t, pos, default_agg(name), name, t))
            try:
                out_type = _out_type(typ)
            except ValueError as e:
                raise ValueError('OUTVAR %s: %s' % (name, e))
            m = None
            if mult not in ('', '*'):
                try:
                    m = float(mult)
                except ValueError:
                    m = None
                if m not in (None, 1.0):
                    say('OUTVAR %s: mult %s is parsed by VIC 5\'s image '
                        'driver and never applied (vic_write.c writes the '
                        'aggregate as it is); it is not applied here either'
                        % (name, mult))
            if any(v.name == name for v in cur.vars):
                raise ValueError(
                    'OUTFILE %s names %s twice; VIC cannot define one netCDF '
                    'variable twice in a file ("NetCDF: String match to name '
                    'in use") -- put the second in its own OUTFILE'
                    % (cur.prefix, name))
            cur.vars.append(OutVar(name, _agg_type(agg), out_type, m,
                                   fmt if fmt not in ('', '*') else None))
    for s in streams:
        if not s.vars:
            # validate_streams (vic_history.c) log_errs on it
            raise ValueError('OUTFILE %s has no OUTVAR: "Number of variables '
                             'in stream is less than 1"' % s.prefix)
    return streams


# ---------------------------------------------------------------------------
# The alarm: which steps close a period
# ---------------------------------------------------------------------------
def as_datetimes(times):
    """Each model step's start as a datetime (cftime objects kept as they
    are): case.dates holds datetimes for a sub-daily model and dates for a
    daily one."""
    out = []
    for t in times:
        if isinstance(t, np.datetime64):
            t = t.astype('datetime64[us]').astype(_dt.datetime)
        elif hasattr(t, 'to_pydatetime'):
            t = t.to_pydatetime()
        elif isinstance(t, _dt.date) and not isinstance(t, _dt.datetime):
            t = _dt.datetime(t.year, t.month, t.day)
        out.append(t)
    return out


def _replace(t, year, month):
    try:
        return t.replace(year=year, month=month)
    except ValueError:
        raise ValueError('Invalid date found during time_delta computation '
                         '(VIC log_errs): %04d-%02d-%02d' % (year, month, t.day))


def _add_months(t, n):
    # time_delta (vic_time.c): month += n, wrapped ONCE past December
    month, year = t.month + n, t.year
    if month > 12:
        month -= 12
        year += 1
    if month > 12:
        raise ValueError('Invalid date found during time_delta computation '
                         '(VIC log_errs): NMONTHS %d from %s wraps the year '
                         'more than once' % (n, t))
    return _replace(t, year, month)


def _add_years(t, n):
    try:
        return t.replace(year=t.year + n)
    except ValueError:
        raise ValueError('VIC does not support a simulation starting from '
                         'Feb 29 of a leap year with yearly AGGFREQ or '
                         'HISTFREQ.')


class _Alarm(object):
    """alarms.c: set_alarm at the first step, raise_alarm, reset_alarm."""

    def __init__(self, freq, t0, dt):
        self.freq, self.dt, self.count = freq, _dt.timedelta(seconds=dt), 0
        self.next = None
        if freq.kind in _UNIT_SECONDS or freq.kind in ('NMONTHS', 'NYEARS'):
            # set_alarm shifts the first step back one step before advancing
            self.next = self._advance(t0 - self.dt)
        elif freq.kind == 'DATE':
            self.next = freq.date

    def _advance(self, t):
        f = self.freq
        if f.kind in _UNIT_SECONDS:
            return t + _dt.timedelta(seconds=f.n * _UNIT_SECONDS[f.kind])
        if f.kind == 'NMONTHS':
            # forward one step, add the months, back one step (reset_alarm)
            return _add_months(t + self.dt, f.n) - self.dt
        if f.kind == 'NYEARS':
            return _add_years(t, f.n)
        return None

    def tick(self, t):
        """Count step `t` into the period; True if the alarm is raised."""
        self.count += 1
        return ((self.freq.kind == 'NSTEPS' and self.count == self.freq.n)
                or (self.next is not None and t == self.next))

    def reset(self, t):
        self.count = 0
        if self.freq.kind in _UNIT_SECONDS or \
                self.freq.kind in ('NMONTHS', 'NYEARS'):
            self.next = self._advance(t)


def windows(times, dt, freq):
    """The periods VIC writes, as [(first step, last step)], and the steps
    it never writes, (first, last) or None.

    `times` are the steps' starts (as_datetimes), `dt` the step in seconds,
    `freq` a Freq. Freq END is one period over the whole run (see the module
    notes: VIC 5 itself writes nothing for it).
    """
    t = as_datetimes(times)
    if not t:
        return [], None
    if freq.kind == 'END':
        return [(0, len(t) - 1)], None
    if freq.kind == 'NEVER':
        return [], (0, len(t) - 1)
    if freq.kind in _UNIT_SECONDS:
        span = freq.n * _UNIT_SECONDS[freq.kind]
        if span % int(dt):
            raise ValueError(
                'AGGFREQ %s is not a whole number of %d-second model steps: '
                'VIC closes a period only on a step that starts exactly at '
                'the period\'s end, so it would never close one and would '
                'write nothing at all (measured, NHOURS 2 at a 3-hour step)'
                % (freq, int(dt)))
    alarm = _Alarm(freq, t[0], dt)
    recs, first = [], 0
    for i, ti in enumerate(t):
        if alarm.tick(ti):
            recs.append((first, i))
            first = i + 1
            alarm.reset(ti)
    tail = (first, len(t) - 1) if first < len(t) else None
    return recs, tail


def file_groups(records, times, dt, hist):
    """Which records share a history file: the write alarm (vic_write.c
    226-235), counted in records and checked on each record's closing step."""
    t = as_datetimes(times)
    if not records:
        return []
    alarm = _Alarm(hist, t[0], dt)
    groups, cur = [], []
    for r, (_, last) in enumerate(records):
        cur.append(r)
        if alarm.tick(t[last]):
            groups.append(cur)
            cur = []
            alarm.reset(t[last])
    if cur:
        groups.append(cur)
    return groups


# ---------------------------------------------------------------------------
# The aggregation itself
# ---------------------------------------------------------------------------
def normalize_agg(agg):
    """One of AGG_TYPES from 'sum', 'AGG_TYPE_SUM', 'mean', 'last', ..."""
    a = str(agg).upper()
    a = a[len('AGG_TYPE_'):] if a.startswith('AGG_TYPE_') else a
    a = {'MEAN': 'AVG', 'LAST': 'END', 'FIRST': 'BEG'}.get(a, a)
    if a not in AGG_TYPES:
        raise ValueError('an aggregation is one of sum, avg (mean), end, beg, '
                         'max, min (VIC\'s AGG_TYPE_*), not %r' % (agg,))
    return a


#: (records, device) -> the index tensors aggregate() needs, made once:
#: a run's windows of one length have the same records, and building them
#: is a host-to-device copy (a host sync) per variable and window
_RECORD_INDEX = {}


def _record_index(records, dev):
    key = (tuple(records), str(dev), torch.is_inference_mode_enabled())
    ix = _RECORD_INDEX.get(key)
    if ix is not None:
        return ix
    if len(_RECORD_INDEX) > 64:
        _RECORD_INDEX.clear()
    n = len(records)
    first = torch.as_tensor([f for f, _ in records], dtype=torch.long,
                            device=dev)
    last = torch.as_tensor([l for _, l in records], dtype=torch.long,
                           device=dev)
    length = last - first + 1
    back = all(records[i + 1][0] == records[i][1] + 1 for i in range(n - 1))
    if back:
        # windows() gives back-to-back periods: one range covers them
        steps = torch.arange(records[0][0], records[-1][1] + 1, device=dev)
    else:
        steps = torch.cat([torch.arange(f, l + 1, device=dev)
                           for f, l in records])
    lens = [l - f + 1 for f, l in records]
    wid = torch.repeat_interleave(torch.arange(n, device=dev), length,
                                  output_size=sum(lens))
    flt = {}

    def length_f(dt):
        if dt not in flt:
            flt[dt] = length.to(dt)
        return flt[dt]
    ix = dict(first=first, last=last, length=length, steps=steps, wid=wid,
              length_f=length_f,
              equal=lens[0] if back and len(set(lens)) == 1 else None)
    _RECORD_INDEX[key] = ix
    return ix


def aggregate(x, records, agg, extremes='vic'):
    """x [step, ...] -> [record, ...], as agg_stream_data forms each record.

    SUM is accumulated from zero in step order -- index_add on the CPU,
    segment_reduce on the GPU, the same additions in the same order, which
    is VIC's own -- and AVG divides that sum by the count.
    MAX and MIN start from 0 under extremes='vic' (VIC's aggdata is zeroed),
    from the period's first value under extremes='true'. Differentiable.
    """
    agg = normalize_agg(agg)
    if extremes not in ('vic', 'true'):
        raise ValueError("extremes is 'vic' or 'true', not %r" % (extremes,))
    dev = x.device
    n = len(records)
    if n == 0:
        return x[:0]
    ix = _record_index(records, dev)
    first, last = ix['first'], ix['last']
    if agg == 'BEG':
        return x.index_select(0, first)
    if agg == 'END':
        return x.index_select(0, last)
    length = ix['length']
    shape = (n,) + tuple(x.shape[1:])
    L = ix['equal']
    if agg in ('SUM', 'AVG') and x.is_cuda and L is not None:
        # back-to-back periods of L steps each (a day of model steps): the
        # sum from 0 in step order, one add per step over all records at
        # once -- segment_reduce's additions in its order (one thread per
        # record adding its steps from 0), so the same bits, without its
        # fixed cost (about 4 ms a call on the ISC domain) and its host
        # copy of the lengths
        r0 = records[0][0]
        xs = x[r0:r0 + n * L].reshape((n, L) + tuple(x.shape[1:]))
        acc = torch.zeros(shape, dtype=x.dtype, device=dev)
        for j in range(L):
            acc = acc + xs[:, j]
        if agg == 'SUM':
            return acc
        return acc / ix['length_f'](x.dtype).reshape(
            (n,) + (1,) * (x.dim() - 1))
    steps, wid = ix['steps'], ix['wid']
    xs = x.index_select(0, steps)
    if agg in ('SUM', 'AVG'):
        if x.is_cuda:
            # index_add on the GPU adds with atomics, in whatever order the
            # threads arrive: two runs of the same model differed in the
            # last bit of a daily sum. segment_reduce gives each record one
            # thread that adds its steps from 0 in step order -- the CPU's
            # index_add order, so the GPU's sums equal the CPU's bit for bit.
            # (unsafe: the lengths are the records' own, so the checks it
            # would make -- host syncs -- hold; the same kernel)
            acc = torch.segment_reduce(xs, 'sum', lengths=length, axis=0,
                                       unsafe=True)
        else:
            acc = torch.zeros(shape, dtype=x.dtype, device=dev).index_add(
                0, wid, xs)
        if agg == 'SUM':
            return acc
        return acc / length.to(x.dtype).reshape((n,) + (1,) * (x.dim() - 1))
    red = 'amax' if agg == 'MAX' else 'amin'
    # VIC's running max/min starts from the zeroed aggdata; the period's own
    # extreme starts from its first value, which the reduction then meets
    # again (max(a, a) = a).
    base = (torch.zeros(shape, dtype=x.dtype, device=dev)
            if extremes == 'vic' else x.index_select(0, first))
    with warnings.catch_warnings():
        # "index_reduce() is in beta": said once per process, and about the
        # API, not about these values
        warnings.simplefilter('ignore', UserWarning)
        return base.index_reduce(0, wid, xs, red, include_self=True)


def key_agg(key, convention='vic5'):
    """VIC's default aggregation for a basin.simulate output `key`: that of
    the OUT_* variable it carries, EXTRA_AGG for the few with none, and
    VIC's fallback (AVG) otherwise. Returns (agg, OUT_* name or None)."""
    if key in EXTRA_AGG:
        return EXTRA_AGG[key], None
    for name, src in OUTVAR_SOURCES.items():
        if src == key or (isinstance(src, tuple) and src[1] == '|'
                          and key in (src[0], src[2])):
            return default_agg(name, convention), name
    return DEFAULT_OTHER, None


def aggregate_output(out, times, steps_per_day, freq='NDAYS 1', how=None,
                     extremes='vic', log=None, convention='vic5'):
    """Every per-step output of basin.simulate aggregated as VIC would.

        day = history.aggregate_output(out, case.dates, case.model_steps)
        day['et']        # [day, cell] mm per day: summed (OUT_EVAP)
        day['swe']       # the end-of-day state (OUT_SWE)
        day['air_temp']  # the day's mean (OUT_AIR_TEMP)

    `freq` is an AGGFREQ ('NDAYS 1', 'NMONTHS 1', 'NHOURS 6', ...) or a
    Freq; `how` overrides the aggregation per key ({'soil_liq': 'AVG'}).
    Returns (dict, record start times, the steps left in a partial last
    period or None). Outputs without a time axis are left out.

    convention='vic4.2d' aggregates as VIC 4.2.d's put_data.c does
    (aggregate_output_42d): `freq` is then an OUT_STEP of at most a day
    ('NDAYS 1', 'NHOURS n', 'NSTEPS n', or hours as an int), and the times
    returned are 4.2.d's stamps -- the start of each period's LAST step.
    """
    if check_convention(convention) == 'vic4.2d':
        return aggregate_output_42d(out, times, steps_per_day, freq=freq,
                                    how=how, log=log)
    say = _announcer(log)
    f = freq if isinstance(freq, Freq) else Freq.parse(str(freq).split())
    dt = 86400 // int(steps_per_day)
    recs, tail = windows(times, dt, f)
    if tail is not None:
        say('aggregate_output: steps %d..%d are a partial last period under '
            'AGGFREQ %s and are not written, as VIC does not write them'
            % (tail[0], tail[1], f))
    t = as_datetimes(times)
    res, fallback = {}, []
    for k, v in out.items():
        if not isinstance(v, torch.Tensor) or v.dim() < 1 or \
                v.shape[0] != len(t):
            continue
        agg = (how or {}).get(k)
        if agg is None:
            agg, name = key_agg(k)
            if name is None and k not in EXTRA_AGG:
                fallback.append(k)
        res[k] = aggregate(v, recs, agg, extremes=extremes)
    if fallback:
        say('aggregate_output: %s carry no VIC output variable and are '
            'averaged, VIC\'s default for a variable it does not list; pass '
            'how={name: agg} to say otherwise' % ', '.join(sorted(fallback)))
    return res, [t[a] for a, _ in recs], tail


def outvar_series(out, name):
    """(tensor [step, cell, ...], the output key(s) it came from) for VIC
    variable `name`, or (None, reason)."""
    src = OUTVAR_SOURCES.get(name)
    if src is None:
        return None, 'TorchVIC does not compute %s' % name
    if isinstance(src, tuple) and src[1] == '|':
        for k in (src[0], src[2]):
            if k in out:
                return out[k], k
        return None, 'this run returned neither %s nor %s' % (src[0], src[2])
    if isinstance(src, tuple):
        a, _, b = src
        if a not in out:
            return None, 'this run returned no %s' % a
        if b not in out:
            return out[a], a
        return out[a] + out[b], '%s + %s' % (a, b)
    if src not in out:
        return None, ('this run returned no %s (a configuration that does '
                      'not compute it: the energy balance off, no vegetation '
                      'mosaic, ...)' % src)
    return out[src], src


class Record(object):
    """One stream's aggregated output, before it is written."""

    def __init__(self, stream, freq, records, tail, times, dt, data, sources,
                 missing, notes):
        self.stream, self.freq = stream, freq
        self.records, self.tail = records, tail
        self.times, self.dt = times, dt
        self.data, self.sources = data, sources
        self.missing, self.notes = missing, notes

    @property
    def starts(self):
        return [self.times[a] for a, _ in self.records]

    @property
    def bounds(self):
        d = _dt.timedelta(seconds=self.dt)
        return [(self.times[a], self.times[b] + d) for a, b in self.records]


def stream_records(out, stream, times, steps_per_day, extremes='vic',
                   log=None):
    """One stream of the global file, aggregated: a Record."""
    say = _announcer(log)
    dt = 86400 // int(steps_per_day)
    freq = stream.agg
    notes = []
    if freq is None:
        freq = Freq('NSTEPS', 1)
        notes.append('OUTFILE %s has no AGGFREQ: VIC 5\'s image driver '
                     'would write no record for it (its alarm stays on '
                     'setup_stream\'s placeholder date; measured), and '
                     'TorchVIC writes it at every model step (NSTEPS 1)'
                     % stream.prefix)
    elif freq.kind == 'END':
        notes.append('OUTFILE %s AGGFREQ END: one record over the whole run. '
                     'VIC 5\'s image driver never raises the END alarm and '
                     'writes no file for this stream (measured); this is '
                     'the documented meaning of END, not VIC 5\'s output'
                     % stream.prefix)
    recs, tail = windows(times, dt, freq)
    t = as_datetimes(times)
    if tail is not None and freq.kind != 'NEVER':
        notes.append('OUTFILE %s: the last %d step(s), %s .. %s, are a '
                     'partial AGGFREQ %s period and are not written, as VIC '
                     'does not write them'
                     % (stream.prefix, tail[1] - tail[0] + 1, t[tail[0]],
                        t[tail[1]], freq))
    if freq.kind == 'DATE' and not recs:
        notes.append('OUTFILE %s: no step starts at %s 00:00, so the DATE '
                     'alarm is never raised and nothing is written'
                     % (stream.prefix, freq.date.date()))
    data, sources, missing = {}, {}, {}
    for v in stream.vars:
        x, src = outvar_series(out, v.name)
        if x is None:
            missing[v.name] = src
            continue
        if x.shape[0] != len(t):
            raise ValueError('%s (%s) has %d steps and the run %d'
                             % (v.name, src, x.shape[0], len(t)))
        data[v.name] = aggregate(x, recs, v.agg, extremes=extremes)
        sources[v.name] = src
        if v.agg in ('MAX', 'MIN') and recs:
            true = aggregate(x, recs, v.agg, extremes='true')
            moved = int((data[v.name] != true).reshape(len(recs), -1)
                        .any(-1).sum()) if extremes == 'vic' else 0
            if moved:
                notes.append(
                    '%s AGG_TYPE_%s: in %d of %d records, cells whose '
                    'period %s is %s zero read 0 -- VIC starts MAX/MIN from 0 '
                    '(reset_stream zeroes aggdata) and so does this file; '
                    'extremes=\'true\' gives the period\'s own extremes'
                    % (v.name, v.agg, moved, len(recs),
                       'maximum' if v.agg == 'MAX' else 'minimum',
                       'below' if v.agg == 'MAX' else 'above'))
    if missing:
        notes.append('OUTFILE %s: not written, TorchVIC has no value for: %s'
                     % (stream.prefix, '; '.join('%s (%s)' % kv
                                                 for kv in missing.items())))
    for n in notes:
        say(n)
    return Record(stream, freq, recs, tail, t, dt, data, sources, missing,
                  notes)


# ---------------------------------------------------------------------------
# Writing the history files
# ---------------------------------------------------------------------------
_CALENDARS = {'STANDARD': 'standard', 'GREGORIAN': 'gregorian',
              'PROLEPTIC_GREGORIAN': 'proleptic_gregorian',
              'NOLEAP': 'noleap', 'NO_LEAP': 'noleap', '365_DAY': '365_day',
              '360_DAY': '360_day', 'JULIAN': 'julian',
              'ALL_LEAP': 'all_leap', '366_DAY': '366_day'}
_TIME_UNITS = {'SECONDS': 'seconds', 'MINUTES': 'minutes', 'HOURS': 'hours',
               'DAYS': 'days'}


def file_name(prefix, freq, start):
    """initialize_history_file's naming, from the file's first record."""
    if freq.kind == 'NDAYS':
        return '%s.%04d-%02d-%02d.nc' % (prefix, start.year, start.month,
                                         start.day)
    if freq.kind == 'NMONTHS':
        return '%s.%04d-%02d.nc' % (prefix, start.year, start.month)
    if freq.kind == 'NYEARS':
        return '%s.%04d.nc' % (prefix, start.year)
    sec = start.hour * 3600 + start.minute * 60 + start.second
    return '%s.%04d-%02d-%02d-%05d.nc' % (prefix, start.year, start.month,
                                          start.day, sec)


class _FileCtx(object):
    """What the history files of one stream share: the calendar and time
    units, the grid and the file groups (write_record, StreamFiles)."""

    def __init__(self, stream, freq, records, times, dt, domain, path_dir,
                 gp):
        import cftime
        self.stream, self.freq, self.gp = stream, freq, gp
        calendar = _CALENDARS.get(str((gp.get('CALENDAR') if gp is not None
                                       else None) or 'STANDARD').upper())
        if calendar is None:
            raise ValueError('Unknown calendar specified: %s'
                             % gp.get('CALENDAR'))
        tu = _TIME_UNITS.get(str((gp.get('OUT_TIME_UNITS') if gp is not None
                                  else None) or 'DAYS').upper())
        if tu is None:
            raise ValueError('Invalid value, or no value for OUT_TIME_UNITS '
                             '(%s)' % gp.get('OUT_TIME_UNITS'))
        self.calendar = calendar
        self.units = units = '%s since %s' % (tu, TIME_ORIGIN)
        self.fmt = OUT_FORMATS[stream.out_format]
        self.comp = int(stream.compress or 0)
        if self.comp and not self.fmt.startswith('NETCDF4'):
            raise ValueError(
                'OUTFILE %s: COMPRESS needs a netCDF-4 file and OUT_FORMAT '
                '%s writes %s (VIC fails in nc_def_var_deflate the same way)'
                % (stream.prefix, stream.out_format, self.fmt))
        names = self.names = domain.names
        self.ny, self.nx = domain.ny, domain.nx
        self.rows, self.cols = np.asarray(domain.rows), np.asarray(domain.cols)
        self.lat = np.asarray(domain.ds[names['lat_var']].values,
                              dtype=np.float64)
        self.lon = np.asarray(domain.ds[names['lon_var']].values,
                              dtype=np.float64)
        self.meta = outvar_metadata()
        self.groups = file_groups(records, times, dt, stream.hist)
        self.starts = [times[a] for a, _ in records]
        d = _dt.timedelta(seconds=dt)
        self.bounds = [(times[a], times[b] + d) for a, b in records]
        self.tnum = np.asarray(cftime.date2num(self.starts, units, calendar),
                               dtype=np.float64)
        self.bnum = np.asarray(cftime.date2num([b for _, b in self.bounds],
                                               units, calendar),
                               dtype=np.float64)
        self.path_dir = path_dir
        os.makedirs(path_dir, exist_ok=True)

    def path(self, g):
        return os.path.join(self.path_dir, file_name(
            self.stream.prefix, self.freq, self.starts[g[0]]))


class _GroupFile(object):
    """One history file, written record range by record range: created at
    the first `append`, closed by `close`."""

    def __init__(self, ctx, g):
        import netCDF4
        self.ctx, self.g = ctx, g
        self.path = path = ctx.path(g)
        self.nc = None
        self.vars = {}
        # VIC writes the same names into the same RESULT_DIR: running both
        # models from one global file (this project's usual comparison)
        # used to overwrite VIC's own history silently. Only a file this
        # writer made may be replaced.
        if os.path.exists(path):
            try:
                with netCDF4.Dataset(path, 'r') as old:
                    ours = getattr(old, 'title', '') == 'TorchVIC history file'
            except OSError:
                ours = False
            if not ours:
                raise FileExistsError(
                    '%s exists and was not written by TorchVIC (VIC\'s own '
                    'history file?); not overwriting it -- give run '
                    '--result-dir DIR to write TorchVIC\'s files elsewhere'
                    % path)

    def _create(self, arrays):
        import netCDF4
        c, g = self.ctx, self.g
        stream, rec_freq = c.stream, c.freq
        ydim, xdim = c.names['y_dim'], c.names['x_dim']
        nc = self.nc = netCDF4.Dataset(self.path, 'w', format=c.fmt)
        nc.title = 'TorchVIC history file'
        nc.source = ('TorchVIC (torchvic.io.history), in the layout of the VIC '
                     '5 image driver\'s history files')
        nc.history = 'Created %s' % _dt.datetime.now().isoformat(
            timespec='seconds')
        nc.Conventions = 'CF-1.6'
        if c.gp is not None:
            nc.global_param = os.path.abspath(c.gp.path)
        nc.aggfreq = str(rec_freq)
        nc.createDimension('time', None)
        nc.createDimension('nv', 2)
        nc.createDimension(ydim, c.ny)
        nc.createDimension(xdim, c.nx)
        tv = nc.createVariable('time', 'f8', ('time',))
        tv.standard_name = 'time'
        tv.units = c.units
        tv.calendar = c.calendar
        tv.bounds = 'time_bnds'
        bv = nc.createVariable('time_bnds', 'f8', ('time', 'nv'))
        bv.standard_name = 'time_bounds'
        bv.units = c.units
        bv.calendar = c.calendar
        if c.lon.ndim == 1:
            lo = nc.createVariable(c.names['lon_var'], 'f8', (xdim,))
            la = nc.createVariable(c.names['lat_var'], 'f8', (ydim,))
        else:
            lo = nc.createVariable(c.names['lon_var'], 'f8', (ydim, xdim))
            la = nc.createVariable(c.names['lat_var'], 'f8', (ydim, xdim))
        lo.long_name, lo.units, lo.standard_name = ('longitude',
                                                    'degrees_east',
                                                    'longitude')
        la.long_name, la.units, la.standard_name = ('latitude',
                                                    'degrees_north',
                                                    'latitude')
        lo[:] = c.lon
        la[:] = c.lat
        gi = np.asarray(g)
        tv[:] = c.tnum[gi]
        bv[:] = np.stack([c.tnum[gi], c.bnum[gi]], axis=-1)
        ncell = len(c.rows)
        for v in stream.vars:
            if v.name not in arrays:
                continue
            a = arrays[v.name]
            if a.shape[1] != ncell:
                raise ValueError(
                    '%s has %d cells and the domain %d active cells; an '
                    'ensemble cannot be written as one VIC history file'
                    % (v.name, a.shape[1], ncell))
            md = c.meta.get(v.name, {})
            dims = ('time',)
            if a.ndim == 3:
                dim = md.get('dim') or 'nlayer'
                if dim not in nc.dimensions:
                    nc.createDimension(dim, a.shape[2])
                dims += (dim,)
            dims += (ydim, xdim)
            dtype = OUT_TYPES[v.out_type]
            fill = FILL[dtype]
            kw = dict(fill_value=fill)
            if c.comp:
                kw.update(zlib=True, complevel=c.comp, shuffle=True)
            var = nc.createVariable(v.name, dtype, dims, **kw)
            for attr in ('long_name', 'standard_name', 'units',
                         'description'):
                if md.get(attr):
                    var.setncattr(attr, md[attr])
            var.cell_methods = CELL_METHODS[v.agg]
            var.torchvic_source = c.sources[v.name]
            # netCDF4 masks the fill value itself; write raw values
            var.set_auto_maskandscale(False)
            self.vars[v.name] = (var, dtype, fill)

    def append(self, arrays, r0):
        """`arrays` ({variable: [record, cell, (elem)] numpy}) hold the
        stream's records r0, r0 + 1, ...; those of this file are written."""
        c, g = self.ctx, self.g
        n = len(next(iter(arrays.values())))
        lo, hi = max(g[0], r0), min(g[-1] + 1, r0 + n)
        if lo >= hi:
            return
        if self.nc is None:
            self._create(arrays)
        for name, (var, dtype, fill) in self.vars.items():
            a = arrays[name][lo - r0:hi - r0]       # [rec, cell, (elem)]
            if a.ndim == 3:
                grid = np.full((a.shape[0], a.shape[2], c.ny, c.nx), fill,
                               dtype=dtype)
                vals = np.moveaxis(a, 1, 2)          # [rec, elem, cell]
                grid[:, :, c.rows, c.cols] = _cast(vals, dtype)
            else:
                grid = np.full((a.shape[0], c.ny, c.nx), fill, dtype=dtype)
                grid[:, c.rows, c.cols] = _cast(a, dtype)
            var[lo - g[0]:hi - g[0]] = grid

    def close(self):
        if self.nc is not None:
            self.nc.close()
            self.nc = None
        return self.path


def write_record(rec, domain, path_dir, gp=None, log=None, only=None):
    """Write one stream's Record as VIC's history file(s); returns paths.
    `only`: the indices (into file_groups) of the files to write, all by
    default; independent files can be written by different processes."""
    say = _announcer(log)
    stream = rec.stream
    if not rec.records or not rec.data:
        return []
    ctx = _FileCtx(stream, rec.freq, rec.records, rec.times, rec.dt, domain,
                   path_dir, gp)
    ctx.sources = rec.sources
    arrays = {k: v.detach().cpu().numpy() for k, v in rec.data.items()}
    written = []
    for gn, g in enumerate(ctx.groups):
        if only is not None and gn not in only:
            continue
        f = _GroupFile(ctx, g)
        try:
            f.append(arrays, 0)
        finally:
            f.close()
        written.append(f.path)
        say('  wrote %s (%d record%s, AGGFREQ %s, %d variable%s)'
            % (f.path, len(g), '' if len(g) == 1 else 's', rec.freq,
               len(arrays), '' if len(arrays) == 1 else 's'))
    return written


def _cast(a, dtype):
    """The C cast vic_write.c makes: (float), (double), or (int) --
    truncation toward zero."""
    if dtype is np.int32:
        return np.trunc(a).astype(np.int32)
    return a.astype(dtype)


def resolve_result_dir(gp, result_dir=None, log=None):
    """Where the history goes: `result_dir`, else the global file's
    RESULT_DIR, else the current directory (said). A RESULT_DIR naming an
    environment variable that is not set -- UW-Hydro's samples say
    ${VIC_SAMPLE_RESULTS}/sample_image -- is refused rather than created
    as a directory called '${VIC_SAMPLE_RESULTS}'."""
    if result_dir is not None:
        return result_dir
    say = _announcer(log)
    raw = gp.get('RESULT_DIR')
    if raw is None:
        say('the global file has no RESULT_DIR: history goes to %s'
            % os.getcwd())
        return os.getcwd()
    path = gp.path_of('RESULT_DIR', getattr(gp, 'sample_data', None))
    if '$' in path:
        raise ValueError('RESULT_DIR %s names an environment variable that '
                         'is not set; set it, or give the history a '
                         'directory (--result-dir, result_dir=)' % raw)
    return path


def write_history(out, case, result_dir=None, extremes='vic', streams=None,
                  log=None, convention=None, output_global=None, spec=None,
                  carry=False):
    """Every OUTFILE stream of the case's global file, aggregated as VIC 5
    aggregates it and written as VIC 5 writes it. Returns the paths.

    `result_dir` defaults to the global file's RESULT_DIR (the current
    directory, said, when it has none). `streams` defaults to
    parse_streams(case.gp).

    `convention` is 'vic5' or 'vic4.2d'; None picks 'vic4.2d' when the
    global file that defines the output -- `output_global` (a path or a
    GlobalParam) if given, else the case's -- is a VIC 4.2.d file, and
    'vic5' otherwise. Under 'vic4.2d' the output is VIC 4.2.d's per-cell
    ASCII/BINARY files (write_history_42d; `spec`, `carry` there).
    """
    conv = check_convention(convention) if convention is not None else None
    ogp = _as_gp(output_global) if output_global is not None else None
    if conv is None:
        conv = convention_of(ogp if ogp is not None else case.gp)
    if conv == 'vic4.2d':
        return write_history_42d(out, case, result_dir=result_dir,
                                 output_global=ogp, spec=spec, carry=carry,
                                 streams=streams, log=log)
    if ogp is not None:
        raise ValueError('output_global names the file the output section is '
                         'read from under the VIC 4.2.d convention; %s is not '
                         'a VIC 4.2.d file and convention is %s'
                         % (ogp.path, conv))
    say = _announcer(log)
    gp = case.gp
    streams = parse_streams(gp, log=say) if streams is None else streams
    if not streams:
        say('the global file declares no OUTFILE: no history is written '
            '(VIC would write its default daily fluxes/snow streams; '
            'TorchVIC\'s outputs stay at the model step)')
        return []
    if int(case.n_cells) != int(case.dom.n_active):
        raise ValueError('the case has %d cells and the domain %d active '
                         'ones: an ensemble has no VIC history layout'
                         % (case.n_cells, case.dom.n_active))
    result_dir = resolve_result_dir(gp, result_dir, log=say)
    paths = []
    for s in streams:
        rec = stream_records(out, s, case.dates, case.model_steps,
                             extremes=extremes, log=say)
        paths += write_record(rec, case.dom, result_dir, gp=gp, log=say)
    return paths


# ===========================================================================
# VIC 4.2.d
# ===========================================================================
# Every rule here is read from VIC 4.2.d's C (reference/VIC-VIC.4.2.d/src,
# identical to the release tarball) and measured against the vicNl it
# compiles to, on Stehekin and FindleyLake (tests/
# test_output_aggregation.py, section (c)).
#
# * Default aggregation -- create_output_list (output_list_utils.c:297-385):
#   every variable AGG_TYPE_AVG, then the 31 END and 53 SUM below. No global
#   file can change it: parse_output_info.c:93 reads OUTVAR as
#   `name [format [type [mult]]]`, four tokens and no aggregation type.
# * A period -- put_data.c:225-229, 694, 777: out_step_ratio = OUT_STEP /
#   TIME_STEP model steps, counted by a static step_count from the first
#   step of the run, not from midnight. OUT_STEP 0 (or none) is TIME_STEP;
#   otherwise a whole multiple of TIME_STEP and <= 24 (get_global_param.c:
#   754-759). A daily model (TIME_STEP 24) writes every step as it is.
# * END: the period's last step (aggdata = data at every step).
#   SUM: aggdata += data, from 0, in step order.
#   AVG: aggdata += data / out_step_ratio -- each step divided BEFORE it is
#   added, which is not VIC 5's sum / count to the last bit.
#   BEG, MAX, MIN: #defined (vicNl_def.h:671-676), never computed: put_data.c
#   has no branch for them and nothing can select them.
# * OUT_AERO_RESIST, _RESIST1, _RESIST2 (put_data.c:686-688): set after every
#   step to 1 / aggdata of OUT_AERO_COND, _COND1, _COND2 -- written, it is
#   1 / (the period's mean conductance), not the mean resistance.
# * The stamp (write_data.c:202-211, called from put_data.c:773 with
#   dmy[rec]): the date of the step that closes the period, i.e. of its LAST
#   step -- year, month, day, and hour when OUT_STEP < 24. From a 00:00
#   start a day is stamped with its own date; OUT_STEP 6 at a 1-hour step is
#   stamped 05, 11, 17, 23; from STARTHOUR 12 the day 12:00-11:00 carries
#   the second date (all measured).
# * A partial last period is never written: step_count never reaches the
#   ratio. And step_count and aggdata are static and never reset between
#   cells, so the partial period is CARRIED into the next cell's first
#   record (measured: Stehekin, OUT_STEP 7, and OUT_STEP 24 from STARTHOUR
#   12). carry=True reproduces that, with the cells in the order of the cell
#   axis (4.2.d's is the soil file's); by default each cell starts clean,
#   and the carry is said.
# * SKIPYEAR n (make_dmy.c:163-169, put_data.c:765): skiprec is n years of
#   365 or 366 days of steps, each year's length that of the calendar year
#   of the record where the skipped year starts; a period is written when
#   its last step's record is >= skiprec. From 1971-07-01, SKIPYEAR 1 skips
#   365 days (1971 is not a leap year) and writes from 1972-06-30
#   (measured).
# * ALMA_OUTPUT TRUE (put_data.c:699-760): after the aggregation, a fixed
#   list of water fluxes / out_dt_sec (mm per period -> mm/s; OUT_PET_*,
#   OUT_DEL*, OUT_DELTACC_BAND and *_FBFLAG are summed, not converted),
#   temperatures + 273.15,
#   pressure and vapour pressure * 1000 (kPa -> Pa), element 0 of OUT_FDEPTH
#   and OUT_TDEPTH / 100, OUT_DELTACC and OUT_DELTAH * out_dt_sec, and
#   OUT_SUB_SNOW += OUT_SUB_CANOP (both already converted). ALMA_42D is that
#   block, in order. Without it nothing is converted: a SUM is mm per
#   period, an AVG the mean in the step's own units.
# * MOISTFRACT TRUE (put_data.c:902-912): OUT_SOIL_LIQ and OUT_SOIL_ICE (so
#   OUT_SOIL_MOIST) divided by depth * 1000 at each step, before the
#   aggregation, with 4.2.d's depth -- rounded to the millimetre in single
#   precision (read_soilparam.c:342, depth_42d). 4.2.d divides each tile's
#   value before the area-weighted sum; this divides the cell's, which
#   agrees to rounding, not bit for bit. OUT_DELSOILMOIST (put_data.c:
#   586-607) is then the change in the summed fractions, so its first step
#   needs the initial layer moisture (records_42d's init_moist).
# * Files (make_in_and_outfiles.c, write_data.c, write_header.c): one per
#   OUTFILE and cell, <RESULT_DIR>/<prefix>_<lat>_<lon>, lat and lon printed
#   with GRID_DECIMAL decimals (default 2). ASCII: the date fields
#   "%04i\t%02i\t%02i[\t%02i]\t", then each value by its OUTVAR format
#   (default %.4f), separated by "\t "; mult is NOT applied. BINARY_OUTPUT:
#   int32 date fields, then each value * mult cast to its OUT_TYPE. A
#   variable's format, type and mult are ONE setting (set_output_var writes
#   them into out_data[varid]): the last OUTVAR line that gives one holds in
#   every OUTFILE. PRT_HEADER: write_header.c's ASCII or binary header. No
#   N_OUTFILES: set_output_defaults.c's fluxes and snow files (and fdepth,
#   snowband, lake).

CONVENTIONS = ('vic5', 'vic4.2d')


def check_convention(convention):
    """'vic5' or 'vic4.2d' (any case), or ValueError."""
    c = str(convention).lower()
    if c not in CONVENTIONS:
        raise ValueError('an aggregation convention is %s, not %r'
                         % (' or '.join(CONVENTIONS), convention))
    return c


#: create_output_list (VIC 4.2.d output_list_utils.c:301-331): END ...
DEFAULT_END_42D = frozenset((
    'OUT_ASAT', 'OUT_LAKE_AREA_FRAC', 'OUT_LAKE_DEPTH', 'OUT_LAKE_ICE',
    'OUT_LAKE_ICE_FRACT', 'OUT_LAKE_ICE_HEIGHT', 'OUT_LAKE_MOIST',
    'OUT_LAKE_SURF_AREA', 'OUT_LAKE_SWE', 'OUT_LAKE_SWE_V', 'OUT_LAKE_VOLUME',
    'OUT_ROOTMOIST', 'OUT_SMFROZFRAC', 'OUT_SMLIQFRAC', 'OUT_SNOW_CANOPY',
    'OUT_SNOW_COVER', 'OUT_SNOW_DEPTH', 'OUT_SOIL_ICE', 'OUT_SOIL_LIQ',
    'OUT_SOIL_MOIST', 'OUT_SOIL_WET', 'OUT_SURFSTOR', 'OUT_SURF_FROST_FRAC',
    'OUT_SWE', 'OUT_WDEW', 'OUT_ZWT', 'OUT_ZWT_LUMPED',
    'OUT_SNOW_CANOPY_BAND', 'OUT_SNOW_COVER_BAND', 'OUT_SNOW_DEPTH_BAND',
    'OUT_SWE_BAND'))
#: ... SUM (:332-385) ...
DEFAULT_SUM_42D = frozenset((
    'OUT_BASEFLOW', 'OUT_DELINTERCEPT', 'OUT_DELSOILMOIST', 'OUT_DELSWE',
    'OUT_DELSURFSTOR', 'OUT_EVAP', 'OUT_EVAP_BARE', 'OUT_EVAP_CANOP',
    'OUT_INFLOW', 'OUT_LAKE_BF_IN', 'OUT_LAKE_BF_IN_V', 'OUT_LAKE_BF_OUT',
    'OUT_LAKE_BF_OUT_V', 'OUT_LAKE_CHAN_IN', 'OUT_LAKE_CHAN_IN_V',
    'OUT_LAKE_CHAN_OUT', 'OUT_LAKE_CHAN_OUT_V', 'OUT_LAKE_DSTOR',
    'OUT_LAKE_DSTOR_V', 'OUT_LAKE_DSWE', 'OUT_LAKE_DSWE_V', 'OUT_LAKE_EVAP',
    'OUT_LAKE_EVAP_V', 'OUT_LAKE_PREC_V', 'OUT_LAKE_RCHRG',
    'OUT_LAKE_RCHRG_V', 'OUT_LAKE_RO_IN', 'OUT_LAKE_RO_IN_V',
    'OUT_LAKE_VAPFLX', 'OUT_LAKE_VAPFLX_V', 'OUT_PET_SATSOIL',
    'OUT_PET_H2OSURF', 'OUT_PET_SHORT', 'OUT_PET_TALL', 'OUT_PET_NATVEG',
    'OUT_PET_VEGNOCR', 'OUT_PREC', 'OUT_RAINF', 'OUT_REFREEZE', 'OUT_RUNOFF',
    'OUT_SNOW_MELT', 'OUT_SNOWF', 'OUT_SUB_BLOWING', 'OUT_SUB_CANOP',
    'OUT_SUB_SNOW', 'OUT_SUB_SURFACE', 'OUT_TRANSP_VEG', 'OUT_DELTACC_BAND',
    'OUT_SNOWT_FBFLAG', 'OUT_SOILT_FBFLAG', 'OUT_SURFT_FBFLAG',
    'OUT_TCAN_FBFLAG', 'OUT_TFOL_FBFLAG'))
#: ... and AGG_TYPE_AVG for every other one (:298-300).
DEFAULT_OTHER_42D = 'AVG'
#: The aggregations put_data.c:669-684 has a branch for.
AGG_TYPES_42D = ('AVG', 'END', 'SUM')

#: VIC 4.2.d names VIC 5 renamed -> the basin.simulate output that carries
#: the VIC 5 name (OUTVAR_SOURCES): OUT_NET_SHORT is VIC 5's OUT_SWNET,
#: OUT_SHORTWAVE its OUT_SWDOWN, OUT_VEGCOVER its OUT_FCANOPY.
OUTVAR_SOURCES_42D = {'OUT_NET_SHORT': 'eb_net_short',
                      'OUT_NET_LONG': 'eb_net_long',
                      'OUT_SHORTWAVE': 'swdown', 'OUT_LONGWAVE': 'lwdown',
                      'OUT_VEGCOVER': 'fcanopy'}

#: put_data.c:699-760, the ALMA_OUTPUT block, statement by statement:
#: (variable, element -- '0' or 'all' --, operation, operand).
#: tools/extract_vic42d_outvars.py reads the same block from the C.
ALMA_42D = tuple(
    [(v, '0', '/=', 'out_dt_sec') for v in (
        'OUT_BASEFLOW', 'OUT_EVAP', 'OUT_EVAP_BARE', 'OUT_EVAP_CANOP',
        'OUT_INFLOW', 'OUT_LAKE_BF_IN', 'OUT_LAKE_BF_IN_V', 'OUT_LAKE_BF_OUT',
        'OUT_LAKE_BF_OUT_V', 'OUT_LAKE_CHAN_IN', 'OUT_LAKE_CHAN_IN_V',
        'OUT_LAKE_CHAN_OUT', 'OUT_LAKE_CHAN_OUT_V', 'OUT_LAKE_DSTOR',
        'OUT_LAKE_DSTOR_V', 'OUT_LAKE_DSWE', 'OUT_LAKE_DSWE_V',
        'OUT_LAKE_EVAP', 'OUT_LAKE_EVAP_V')]
    + [('OUT_LAKE_ICE_TEMP', '0', '+=', 'KELVIN')]
    + [(v, '0', '/=', 'out_dt_sec') for v in (
        'OUT_LAKE_PREC_V', 'OUT_LAKE_RCHRG', 'OUT_LAKE_RCHRG_V',
        'OUT_LAKE_RO_IN', 'OUT_LAKE_RO_IN_V', 'OUT_LAKE_VAPFLX',
        'OUT_LAKE_VAPFLX_V')]
    + [('OUT_LAKE_SURF_TEMP', '0', '+=', 'KELVIN')]
    + [(v, '0', '/=', 'out_dt_sec') for v in (
        'OUT_PREC', 'OUT_RAINF', 'OUT_REFREEZE', 'OUT_RUNOFF',
        'OUT_SNOW_MELT', 'OUT_SNOWF', 'OUT_SUB_BLOWING', 'OUT_SUB_CANOP',
        'OUT_SUB_SNOW')]
    + [('OUT_SUB_SNOW', '0', '+=', 'OUT_SUB_CANOP')]
    + [(v, '0', '/=', 'out_dt_sec') for v in ('OUT_SUB_SURFACE',
                                              'OUT_TRANSP_VEG')]
    + [(v, '0', '+=', 'KELVIN') for v in (
        'OUT_BARESOILT', 'OUT_SNOW_PACK_TEMP', 'OUT_SNOW_SURF_TEMP')]
    + [(v, 'all', '+=', 'KELVIN') for v in (
        'OUT_SOIL_TEMP', 'OUT_SOIL_TNODE', 'OUT_SOIL_TNODE_WL')]
    + [(v, '0', '+=', 'KELVIN') for v in ('OUT_SURF_TEMP', 'OUT_VEGT')]
    + [(v, '0', '/=', '100') for v in ('OUT_FDEPTH', 'OUT_TDEPTH')]
    + [(v, '0', '*=', 'out_dt_sec') for v in ('OUT_DELTACC', 'OUT_DELTAH')]
    + [('OUT_AIR_TEMP', '0', '+=', 'KELVIN')]
    + [(v, '0', '*=', '1000') for v in ('OUT_PRESSURE', 'OUT_VP',
                                        'OUT_VPD')])
KELVIN_42D = 273.15          # vicNl_def.h:288
SECPHOUR = 3600              # vicNl_def.h:277
#: OUT_TYPE_* codes (vicNl_def.h:662-668), as the binary header stores them
OUT_TYPE_CODES_42D = {'OUT_TYPE_DEFAULT': 0, 'OUT_TYPE_CHAR': 1,
                      'OUT_TYPE_SINT': 2, 'OUT_TYPE_USINT': 3,
                      'OUT_TYPE_INT': 4, 'OUT_TYPE_FLOAT': 5,
                      'OUT_TYPE_DOUBLE': 6}
#: the types parse_output_info.c:100-109 recognises on an OUTVAR line; any
#: other string (OUT_TYPE_INT and OUT_TYPE_CHAR included) is DEFAULT, which
#: leaves the variable's type as it was
OUT_TYPES_PARSED_42D = ('OUT_TYPE_USINT', 'OUT_TYPE_SINT', 'OUT_TYPE_FLOAT',
                        'OUT_TYPE_DOUBLE')
#: numpy type, and the range a C cast from double is defined on
_BIN_DTYPES_42D = {'OUT_TYPE_SINT': (np.int16, -32768, 32767),
                   'OUT_TYPE_USINT': (np.uint16, 0, 65535),
                   'OUT_TYPE_FLOAT': (np.float32, None, None),
                   'OUT_TYPE_DOUBLE': (np.float64, None, None)}

OUTVAR_INVENTORY_42D = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
    'vic42d_outvars.json')
_META42 = None


def outvar_metadata_42d():
    """{OUT_*: {index, default_agg, nelem, units, ...}} of VIC 4.2.d, from
    data/vic42d_outvars.json, or {} when it is not shipped."""
    global _META42
    if _META42 is None:
        try:
            with open(OUTVAR_INVENTORY_42D) as fh:
                _META42 = json.load(fh).get('variables', {})
        except (IOError, OSError, ValueError):
            _META42 = {}
    return _META42


#: The VIC 5 names VIC 4.2.d does not have (data/vic42d_outvars.json
#: vs_vic5.only_in_vic5; the test keeps the two equal).
VIC5_ONLY = frozenset((
    'OUT_DISCHARGE', 'OUT_FCANOPY', 'OUT_LWDOWN', 'OUT_LWNET',
    'OUT_LWNET_BAND', 'OUT_PET', 'OUT_SOIL_ICE_FRAC', 'OUT_SOIL_LIQ_FRAC',
    'OUT_SWDOWN', 'OUT_SWNET', 'OUT_SWNET_BAND', 'OUT_TIME_VICRUN_CPU',
    'OUT_TIME_VICRUN_WALL'))


def is_vic42d_outvar(name):
    """Whether VIC 4.2.d defines `name` (vicNl_def.h's OUT_* list)."""
    meta = outvar_metadata_42d()
    if meta:
        return name in meta
    return (name in DEFAULT_END_42D or name in DEFAULT_SUM_42D
            or name in OUTVAR_SOURCES_42D
            or (name in OUTVAR_SOURCES and name not in VIC5_ONLY
                and name not in OUTVAR_SOURCES_OUTVARS))


def default_agg_42d(name):
    """VIC 4.2.d's aggregation of `name`. A VIC 5 name 4.2.d does not have
    (OUT_SWDOWN, OUT_PET, ...) keeps VIC 5's default, which is also that of
    its 4.2.d counterpart where there is one."""
    if name in DEFAULT_END_42D:
        return 'END'
    if name in DEFAULT_SUM_42D:
        return 'SUM'
    if not is_vic42d_outvar(name) and is_vic_outvar(name):
        return default_agg(name)
    return DEFAULT_OTHER_42D


def _as_gp(g):
    from .global_param import GlobalParam
    return g if hasattr(g, 'lines') else GlobalParam(g)


#: Keys only a VIC 4.2.d global file has: VIC 5 log_errs on each
#: (TIME_STEP -> MODEL_STEPS_PER_DAY, OUT_STEP -> AGGFREQ, N_OUTFILES gone).
VIC42D_KEYS = ('TIME_STEP', 'OUT_STEP', 'N_OUTFILES')


def is_vic42d_global(gp):
    """Whether the global file is a VIC 4.2.d one."""
    keys = set(k for k, _ in getattr(gp, 'lines', []))
    return 'MODEL_STEPS_PER_DAY' not in keys and bool(keys & set(VIC42D_KEYS))


def convention_of(gp):
    """'vic4.2d' for a VIC 4.2.d global file, 'vic5' for any other."""
    return 'vic4.2d' if is_vic42d_global(gp) else 'vic5'


def _leap(y):
    # LEAPYR, make_dmy.c:6
    return (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0)


def _atof(s):
    """C's atof: the leading number of `s`, 0 if there is none."""
    m = re.match(r'\s*([+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)', str(s))
    return float(m.group(1)) if m else 0.0


def step_hours_42d(steps_per_day):
    """The model step in whole hours, as VIC 4.2.d's TIME_STEP is."""
    spd = int(steps_per_day)
    if spd < 1 or 24 % spd:
        raise ValueError('%d steps a day is not a whole number of hours: '
                         'VIC 4.2.d\'s TIME_STEP is an integer number of '
                         'hours, and its OUT_STEP a multiple of it' % spd)
    return 24 // spd


def check_out_step_42d(out_dt, dt):
    """get_global_param.c:757 -- VIC 4.2.d's own refusal."""
    if out_dt < dt or out_dt > 24 or out_dt % dt:
        raise ValueError('Invalid output step specified.  Output step must '
                         'be an integer multiple of the model time step; >= '
                         'model time step and <= 24 (VIC 4.2.d get_global_'
                         'param.c:757): OUT_STEP %s at a %d-hour step'
                         % (out_dt, dt))
    return int(out_dt)


def out_step_42d(freq, dt):
    """VIC 4.2.d's OUT_STEP, in hours, from hours (0: the model step) or an
    AGGFREQ ('NDAYS 1' is 24, 'NHOURS n' n, 'NSTEPS n' n steps). A period
    longer than a day cannot be one: 4.2.d refuses OUT_STEP > 24."""
    if freq is None:
        return check_out_step_42d(dt, dt)
    if isinstance(freq, (int, np.integer)):
        return check_out_step_42d(int(freq) or dt, dt)
    f = freq if isinstance(freq, Freq) else Freq.parse(str(freq).split())
    if f.kind == 'NSTEPS':
        h = f.n * dt
    elif f.kind in _UNIT_SECONDS:
        sec = f.n * _UNIT_SECONDS[f.kind]
        if sec % SECPHOUR:
            raise ValueError('AGGFREQ %s is not a whole number of hours, and '
                             'VIC 4.2.d\'s OUT_STEP is in hours' % f)
        h = sec // SECPHOUR
    else:
        raise ValueError('AGGFREQ %s has no VIC 4.2.d equivalent: 4.2.d '
                         'aggregates over OUT_STEP hours, at most 24 '
                         '(get_global_param.c:757)' % f)
    return check_out_step_42d(h, dt)


def windows_42d(n_steps, ratio, n_cells=1, carry=False):
    """The periods VIC 4.2.d writes, as [(first, last)] in the flattened
    (cell, step) index c * n_steps + i, in cell order, and the runs of
    steps no record covers.

    Each cell's periods are `ratio` steps from its own first step; with
    carry=True they run on across the cells, as 4.2.d's static step_count
    and aggdata do (put_data.c:198, 229, 694, 777), so a cell's partial
    last period becomes the start of the next cell's first."""
    n, r = int(n_steps), int(ratio)
    if carry:
        total = n * int(n_cells)
        k = total // r
        recs = [(j * r, j * r + r - 1) for j in range(k)]
        tails = [(k * r, total - 1)] if k * r < total else []
        return recs, tails
    recs, tails = [], []
    for c in range(int(n_cells)):
        base, k = c * n, n // r
        recs += [(base + j * r, base + j * r + r - 1) for j in range(k)]
        if k * r < n:
            tails.append((base + k * r, base + n - 1))
    return recs, tails


def aggregate_42d(x, records, agg, ratio):
    """x [step, ...] -> [record, ...] as put_data.c:669-684 forms each:
    END the last step, SUM accumulated from 0 in step order, AVG the sum of
    x / ratio in step order. 4.2.d has no other aggregation."""
    agg = normalize_agg(agg)
    if agg == 'AVG':
        return aggregate(x / float(ratio), records, 'SUM')
    if agg in ('SUM', 'END'):
        return aggregate(x, records, agg)
    raise ValueError('AGG_TYPE_%s: VIC 4.2.d aggregates only AVG, END and SUM '
                     '(put_data.c:669-684 has no branch for BEG, MAX or MIN, '
                     'so their aggdata would stay 0)' % agg)


def skip_records_42d(t0, dt, skipyear):
    """make_dmy.c:163-169: the records SKIPYEAR years cover -- 365 or 366
    days of steps each, by the year of the record where that year starts."""
    t0 = as_datetimes([t0])[0]
    skiprec = 0
    for _ in range(int(skipyear)):
        y = (t0 + _dt.timedelta(hours=int(dt) * skiprec)).year
        skiprec += (366 if _leap(y) else 365) * 24 // int(dt)
    return skiprec


def _flatten(x):
    """[step, cell, ...] -> [cell * step, ...], cell-major."""
    return x.transpose(0, 1).reshape((-1,) + tuple(x.shape[2:]))


def aggregate_output_42d(out, times, steps_per_day, freq='NDAYS 1', how=None,
                         log=None):
    """aggregate_output under VIC 4.2.d's convention (the rules above).

    `freq` is OUT_STEP: hours (0 = the model step) or an AGGFREQ of at most
    a day. Returns (dict of [record, cell, ...], each record's 4.2.d stamp
    -- the start of its LAST step --, the steps of a partial last period or
    None). eb_aero_resist is 1 / (mean eb_aero_cond) when the run has the
    conductance, as 4.2.d writes OUT_AERO_RESIST.
    """
    say = _announcer(log)
    dt = step_hours_42d(steps_per_day)
    ratio = out_step_42d(freq, dt) // dt
    t = as_datetimes(times)
    recs, tails = windows_42d(len(t), ratio)
    tail = tails[0] if tails else None
    if tail is not None:
        say('aggregate_output (VIC 4.2.d): steps %d..%d are a partial last '
            'OUT_STEP period and are not written, as VIC 4.2.d does not '
            'write them' % tail)
    res, fallback = {}, []
    for k, v in out.items():
        if not isinstance(v, torch.Tensor) or v.dim() < 1 or \
                v.shape[0] != len(t):
            continue
        agg = (how or {}).get(k)
        if agg is None and k == 'eb_aero_resist' and \
                isinstance(out.get('eb_aero_cond'), torch.Tensor):
            c = aggregate_42d(out['eb_aero_cond'], recs, 'AVG', ratio)
            res[k] = torch.ones_like(c) / c
            continue
        if agg is None:
            agg, name = key_agg(k, 'vic4.2d')
            if name is None and k not in EXTRA_AGG:
                fallback.append(k)
        res[k] = aggregate_42d(v, recs, agg, ratio)
    if fallback:
        say('aggregate_output (VIC 4.2.d): %s carry no VIC output variable '
            'and are averaged, 4.2.d\'s default; pass how={name: agg} to say '
            'otherwise' % ', '.join(sorted(fallback)))
    return res, [t[b] for _, b in recs], tail


# -- the output section of a VIC 4.2.d global file --------------------------
class OutFile42d(object):
    """One OUTFILE: its prefix, declared variable count, variables in
    column order, OUT_STEP in hours, and any aggregation given explicitly
    (only a VIC 5 stream can give one)."""

    def __init__(self, prefix, nvars, names=None, out_dt=None, aggs=None):
        self.prefix, self.nvars = prefix, int(nvars)
        self.names = list(names or [])
        self.out_dt = out_dt
        self.aggs = dict(aggs or {})

    def agg_of(self, name):
        return self.aggs.get(name) or default_agg_42d(name)

    def __repr__(self):
        return '<OutFile42d %s: %d variables, OUT_STEP %s>' % (
            self.prefix, len(self.names), self.out_dt)


class Spec42d(object):
    """What VIC 4.2.d writes, as vicNl reads it from the global file."""

    def __init__(self):
        self.dt = None                 # the model step, hours
        self.out_dt = None             # OUT_STEP, hours
        self.skipyear = 0
        self.binary = self.alma = self.moistfract = False
        self.prt_header = self.compress = False
        #: OUTPUT_FORCE TRUE: write the forcing (write_forcing_file.c)
        self.output_force = False
        self.grid_decimal = 2          # initialize_global.c:189
        self.files = []
        # format, type and mult are per VARIABLE (out_data[varid]), the last
        # OUTVAR line that set one holding in every file
        self.fmt, self.type, self.mult = {}, {}, {}
        self.source = None
        self.notes = []

    def format_of(self, name):
        return self.fmt.get(name, '%.4f')

    def type_of(self, name):
        return self.type.get(name, 'OUT_TYPE_FLOAT')

    def mult_of(self, name):
        return self.mult.get(name, 1.0)

    def __repr__(self):
        return ('<Spec42d OUT_STEP %s at a %s-hour step, %s%s, files %s>'
                % (self.out_dt, self.dt,
                   'BINARY' if self.binary else 'ASCII',
                   ', ALMA' if self.alma else '',
                   ', '.join(f.prefix for f in self.files)))


def _flag42(gp, key):
    # strcasecmp("TRUE", flgstr): anything else is FALSE (a "1" included)
    v = gp.get(key)
    return v is not None and str(v).upper() == 'TRUE'


_C_FLOAT_FORMAT = re.compile(r'^%[-+ #0]*\d*(\.\d*)?[hlL]?[feEgG]$')


def _check_c_format(fmt, name):
    """An OUTVAR format VIC 4.2.d would pass a double to (write_data.c:
    222). parse_output_info.c:42 reads it into char[10]."""
    if len(fmt) > 9:
        raise ValueError('OUTVAR %s: format %r is longer than the 9 characters '
                         'VIC 4.2.d reads it into (char format[10], '
                         'parse_output_info.c:42)' % (name, fmt))
    if not _C_FLOAT_FORMAT.match(fmt):
        raise ValueError('OUTVAR %s: format %r is not a single floating-point '
                         'conversion; VIC 4.2.d passes a double to it '
                         '(write_data.c:222), which only %%f/%%e/%%g print'
                         % (name, fmt))
    return fmt


def default_files_42d(full_energy=False, frozen_soil=False, blowing=False,
                      prt_snow_band=False, lakes=False):
    """set_output_defaults.c:63-222: the files VIC 4.2.d writes when the
    global file has no N_OUTFILES."""
    eb = full_energy or frozen_soil
    fluxes = (['OUT_PREC', 'OUT_EVAP', 'OUT_RUNOFF', 'OUT_BASEFLOW',
               'OUT_WDEW', 'OUT_SOIL_LIQ']
              + (['OUT_RAD_TEMP'] if eb else [])
              + ['OUT_NET_SHORT', 'OUT_R_NET']
              + (['OUT_LATENT'] if eb else [])
              + ['OUT_EVAP_CANOP', 'OUT_TRANSP_VEG', 'OUT_EVAP_BARE',
                 'OUT_SUB_CANOP', 'OUT_SUB_SNOW']
              + (['OUT_SENSIBLE', 'OUT_GRND_FLUX', 'OUT_DELTAH',
                  'OUT_FUSION'] if eb else [])
              + ['OUT_AERO_RESIST', 'OUT_SURF_TEMP', 'OUT_ALBEDO',
                 'OUT_REL_HUMID', 'OUT_IN_LONG', 'OUT_AIR_TEMP', 'OUT_WIND'])
    snow = (['OUT_SWE', 'OUT_SNOW_DEPTH', 'OUT_SNOW_CANOPY', 'OUT_SNOW_COVER']
            + (['OUT_ADVECTION', 'OUT_DELTACC', 'OUT_SNOW_FLUX',
                'OUT_RFRZ_ENERGY', 'OUT_MELT_ENERGY', 'OUT_ADV_SENS',
                'OUT_LATENT_SUB', 'OUT_SNOW_SURF_TEMP', 'OUT_SNOW_PACK_TEMP',
                'OUT_SNOW_MELT'] if eb else [])
            + (['OUT_SUB_BLOWING', 'OUT_SUB_SURFACE', 'OUT_SUB_SNOW']
               if blowing else []))
    files = [('fluxes', fluxes), ('snow', snow)]
    if frozen_soil:
        files.append(('fdepth', ['OUT_FDEPTH', 'OUT_TDEPTH', 'OUT_SOIL_MOIST',
                                 'OUT_SURF_FROST_FRAC']))
    if prt_snow_band:
        files.append(('snowband',
                      ['OUT_SWE_BAND', 'OUT_SNOW_DEPTH_BAND',
                       'OUT_SNOW_CANOPY_BAND']
                      + (['OUT_ADVECTION_BAND', 'OUT_DELTACC_BAND',
                          'OUT_SNOW_FLUX_BAND', 'OUT_RFRZ_ENERGY_BAND']
                         if full_energy else [])
                      + ['OUT_NET_SHORT_BAND', 'OUT_NET_LONG_BAND',
                         'OUT_ALBEDO_BAND', 'OUT_LATENT_BAND',
                         'OUT_SENSIBLE_BAND', 'OUT_GRND_FLUX_BAND']))
    if lakes:
        files.append(('lake', ['OUT_LAKE_ICE_TEMP', 'OUT_LAKE_ICE_HEIGHT',
                               'OUT_LAKE_ICE_FRACT', 'OUT_LAKE_DEPTH',
                               'OUT_LAKE_SURF_AREA', 'OUT_LAKE_VOLUME',
                               'OUT_LAKE_SURF_TEMP', 'OUT_LAKE_EVAP']))
    return [OutFile42d(p, len(n), n) for p, n in files]


#: set_output_defaults.c:41-52: the file OUTPUT_FORCE writes unless the global
#: file gives N_OUTFILES -- (variable, format, type, multiplier).
FORCE_DEFAULTS_42D = (
    ('OUT_PREC', '%.4f', 'OUT_TYPE_USINT', 40),
    ('OUT_AIR_TEMP', '%.4f', 'OUT_TYPE_SINT', 100),
    ('OUT_SHORTWAVE', '%.4f', 'OUT_TYPE_USINT', 50),
    ('OUT_LONGWAVE', '%.4f', 'OUT_TYPE_USINT', 80),
    ('OUT_DENSITY', '%.4f', 'OUT_TYPE_USINT', 100),
    ('OUT_PRESSURE', '%.4f', 'OUT_TYPE_USINT', 100),
    ('OUT_VP', '%.4f', 'OUT_TYPE_SINT', 100),
    ('OUT_WIND', '%.4f', 'OUT_TYPE_USINT', 100))


def parse_output_42d(gp, dt=None, log=None, skip_vic5_streams=False):
    """The output section of a VIC 4.2.d global file, read as
    get_global_param.c and parse_output_info.c read it: a Spec42d.

    `dt` is the model step in hours TorchVIC ran at (default: the file's
    TIME_STEP). OUT_STEP 0 (or none) is the file's TIME_STEP, and each
    OUT_STEP period is formed from `dt`-hour steps.

    `skip_vic5_streams`: OUTFILE/OUTVAR lines with no N_OUTFILES before them
    are a VIC 5 global file's model-output streams, which 4.2.d would stop
    on; OUTPUT_FORCE (nothing to aggregate) skips them and writes the
    forcing defaults.
    """
    say = _announcer(log)
    gp = _as_gp(gp)
    s = Spec42d()
    s.source = gp.path
    step = _atoi(gp.get('TIME_STEP')) if gp.get('TIME_STEP') else None
    s.dt = int(dt) if dt is not None else step
    if s.dt is None:
        raise ValueError('%s has no TIME_STEP ("Model time step has not been '
                         'defined") and no model step was given' % gp.path)
    if step is not None and dt is not None and step != int(dt):
        s.notes.append('%s says TIME_STEP %d; TorchVIC stepped at %d h, and '
                       'each OUT_STEP period is formed from its steps'
                       % (os.path.basename(gp.path), step, int(dt)))
    out = _atoi(gp.get('OUT_STEP')) if gp.get('OUT_STEP') else 0
    s.out_dt = check_out_step_42d(out or step or s.dt, s.dt)
    s.skipyear = _atoi(gp.get('SKIPYEAR')) if gp.get('SKIPYEAR') else 0
    s.binary = _flag42(gp, 'BINARY_OUTPUT')
    s.alma = _flag42(gp, 'ALMA_OUTPUT')
    s.moistfract = _flag42(gp, 'MOISTFRACT')
    s.prt_header = _flag42(gp, 'PRT_HEADER')
    s.compress = _flag42(gp, 'COMPRESS')
    if gp.get('GRID_DECIMAL') is not None:
        s.grid_decimal = _atoi(gp.get('GRID_DECIMAL'))
    s.output_force = _flag42(gp, 'OUTPUT_FORCE')
    if s.compress:
        s.notes.append('COMPRESS TRUE: VIC 4.2.d gzips each output file as it '
                       'closes it (close_files.c:51); TorchVIC writes them '
                       'uncompressed')
    n_out, cur = None, None
    for key, rest in gp.lines:
        if key == 'N_OUTFILES':
            # parse_output_info.c:63-70: the defaults are dropped and every
            # variable's format/type/mult reset (init_output_list)
            n_out = _atoi(rest[0]) if rest else 0
            s.files, s.fmt, s.type, s.mult, cur = [], {}, {}, {}, None
        elif key in ('OUTFILE', 'OUTVAR') and n_out is None and \
                skip_vic5_streams:
            continue
        elif key == 'OUTFILE':
            if n_out is None:
                raise ValueError('Error in global param file: "N_OUTFILES" '
                                 'must be specified before you can specify '
                                 '"OUTFILE".')
            if len(s.files) >= n_out:
                raise ValueError('Error in global param file: number of output '
                                 'files specified in N_OUTFILES (%d) is less '
                                 'than actual number of output files defined '
                                 'in the global param file.' % n_out)
            cur = OutFile42d(rest[0] if rest else '',
                             _atoi(rest[1]) if len(rest) > 1 else 0)
            s.files.append(cur)
        elif key == 'OUTVAR':
            if cur is None:
                raise ValueError('Error in global param file: "OUTFILE" must '
                                 'be specified before you can specify '
                                 '"OUTVAR".')
            if not rest:
                raise ValueError('OUTVAR with no variable name')
            name = rest[0]
            f, t, m = (list(rest[1:4]) + ['', '', ''])[:3]
            if len(rest) > 4:
                say('OUTVAR %s: %s is past the fourth token and VIC 4.2.d does '
                    'not read it (parse_output_info.c:93 reads name, format, '
                    'type, mult); the aggregation stays %s'
                    % (name, ' '.join(rest[4:]), default_agg_42d(name)))
            if not is_vic42d_outvar(name):
                raise ValueError('Error: set_output_var: "%s" was not found in '
                                 'the list of supported output variable names '
                                 '(VIC 4.2.d vicNl_def.h)' % name)
            if f:
                tu = t.upper()
                if tu in OUT_TYPES_PARSED_42D:
                    s.type[name] = tu
                elif tu:
                    say('OUTVAR %s: VIC 4.2.d\'s parse_output_info does not '
                        'recognise %s (only %s) and leaves the type as it was'
                        % (name, t, ', '.join(OUT_TYPES_PARSED_42D)))
                mv = 0.0 if m == '*' else _atof(m)
                if mv != 0:
                    # out_data[].mult is a float (vicNl_def.h)
                    s.mult[name] = float(np.float32(mv))
                if f != '*':
                    s.fmt[name] = _check_c_format(f, name)
            cur.names.append(name)
    if n_out is None and s.output_force:
        # set_output_defaults.c:41-52
        s.files = [OutFile42d('full_data', len(FORCE_DEFAULTS_42D),
                              [n for n, *_ in FORCE_DEFAULTS_42D])]
        for nm, fmt, typ, mult in FORCE_DEFAULTS_42D:
            s.fmt[nm], s.type[nm], s.mult[nm] = fmt, typ, float(mult)
        s.notes.append('no N_OUTFILES: VIC 4.2.d\'s OUTPUT_FORCE default file '
                       '(full_data)')
    elif n_out is None:
        lakes = gp.get('LAKES')
        s.files = default_files_42d(
            full_energy=_flag42(gp, 'FULL_ENERGY'),
            frozen_soil=_flag42(gp, 'FROZEN_SOIL'),
            blowing=_flag42(gp, 'BLOWING'),
            prt_snow_band=_flag42(gp, 'PRT_SNOW_BAND'),
            lakes=lakes is not None and str(lakes).upper() != 'FALSE')
        s.notes.append('no N_OUTFILES: VIC 4.2.d\'s default files (%s)'
                       % ', '.join(f.prefix for f in s.files))
    else:
        if len(s.files) < n_out:
            raise ValueError('N_OUTFILES %d but %d OUTFILE line(s): VIC 4.2.d '
                             'would open %d more file(s) with an empty prefix '
                             'and write only dates to them'
                             % (n_out, len(s.files), n_out - len(s.files)))
        for f in s.files:
            if len(f.names) < f.nvars:
                raise ValueError(
                    'OUTFILE %s declares %d variables and lists %d: VIC 4.2.d '
                    'fills the missing column(s) with varid 0, OUT_ASAT '
                    '(parse_output_info.c:83 callocs varid); list them or '
                    'correct the count' % (f.prefix, f.nvars, len(f.names)))
            if len(f.names) > f.nvars:
                raise ValueError(
                    'OUTFILE %s declares %d variables and lists %d: VIC 4.2.d '
                    'writes past the end of its varid array (undefined); '
                    'correct the count' % (f.prefix, f.nvars, len(f.names)))
    for f in s.files:
        f.out_dt = s.out_dt
    for n in s.notes:
        say(n)
    return s


def spec_from_streams_42d(streams, dt, gp=None, log=None):
    """VIC 5 OUTFILE streams written in VIC 4.2.d's convention: each
    stream's AGGFREQ becomes an OUT_STEP (at most a day), its variables
    are aggregated by 4.2.d's arithmetic and written as 4.2.d's ASCII
    files. An explicit AGG_TYPE is kept if 4.2.d has it (AVG, END, SUM)."""
    say = _announcer(log)
    s = Spec42d()
    s.dt = int(dt)
    s.source = ('the OUTFILE streams of %s' % gp.path) if gp is not None \
        else 'OUTFILE streams'
    if gp is not None and gp.get('GRID_DECIMAL') is not None:
        s.grid_decimal = _atoi(gp.get('GRID_DECIMAL'))
    for st in streams:
        if st.agg is None:
            s.notes.append('OUTFILE %s has no AGGFREQ: written at every model '
                           'step (VIC 4.2.d\'s OUT_STEP 0)' % st.prefix)
        out_dt = out_step_42d(st.agg, s.dt)
        f = OutFile42d(st.prefix, len(st.vars), out_dt=out_dt)
        for v in st.vars:
            if v.agg_given:
                if v.agg not in AGG_TYPES_42D:
                    raise ValueError(
                        'OUTFILE %s OUTVAR %s AGG_TYPE_%s: VIC 4.2.d '
                        'aggregates only AVG, END and SUM (put_data.c:669-684)'
                        % (st.prefix, v.name, v.agg))
                f.aggs[v.name] = v.agg
            if v.fmt:
                s.fmt[v.name] = _check_c_format(v.fmt, v.name)
            f.names.append(v.name)
        s.files.append(f)
    outs = sorted(set(f.out_dt for f in s.files))
    s.out_dt = outs[0] if len(outs) == 1 else None
    if len(outs) > 1:
        s.notes.append('the streams aggregate over %s hours; VIC 4.2.d has one '
                       'OUT_STEP for every file' % ', '.join(map(str, outs)))
    s.notes.append('%s: written as VIC 4.2.d\'s per-cell ASCII files'
                   % s.source)
    for n in s.notes:
        say(n)
    return s


def output_spec_42d(case, output_global=None, streams=None, log=None):
    """The Spec42d a run of `case` is written by: the output section of
    `output_global`, or of the case's global file, when that is a VIC 4.2.d
    file; else its VIC 5 streams (`streams`, or parsed from it)."""
    say = _announcer(log)
    dt = step_hours_42d(case.model_steps)
    gp = _as_gp(output_global) if output_global is not None else case.gp
    if is_vic42d_global(gp):
        return parse_output_42d(gp, dt=dt, log=say)
    if streams is None:
        streams = parse_streams(gp, log=say)
    if not streams:
        raise ValueError('%s declares no OUTFILE: there is nothing to write in '
                         'VIC 4.2.d\'s convention' % gp.path)
    return spec_from_streams_42d(streams, dt, gp=gp, log=say)


# -- the records, and the files ----------------------------------------------
def _series_42d(out, name):
    # `out` may be keyed by VIC's own names -- VIC 4.2.d's per-step output,
    # read back, aggregated as vicNl would have aggregated it
    if isinstance(out.get(name), torch.Tensor):
        return out[name], name
    src = OUTVAR_SOURCES_42D.get(name)
    if src is not None:
        return (out[src], src) if src in out else \
            (None, 'this run returned no %s' % src)
    if name in OUTVAR_SOURCES_OUTVARS and name != 'OUT_ENERGY_ERROR':
        # VIC 5 semantics (out-vars); 4.2.d forms these differently
        return None, 'TorchVIC does not compute %s in VIC 4.2.d\'s convention' % name
    return outvar_series(out, name)


def _apply_alma_42d(data, out_dt):
    """put_data.c:699-760 on {OUT_*: [record(, elem)]}, in its order."""
    out_dt_sec = float(int(out_dt) * SECPHOUR)
    for name, elem, op, operand in ALMA_42D:
        if name not in data:
            continue
        x = data[name]
        if operand == 'out_dt_sec':
            y = out_dt_sec
        elif operand == 'KELVIN':
            y = KELVIN_42D
        elif operand.startswith('OUT_'):
            y = data[operand]                  # records_42d computed it
        else:
            y = float(operand)
        first = elem == '0' and x.dim() > 1
        part = x[:, 0] if first else x
        if op == '/=':
            part = part / y
        elif op == '*=':
            part = part * y
        else:
            part = part + y
        if first:
            x = x.clone()
            x[:, 0] = part
            data[name] = x
        else:
            data[name] = part
    return data


def depth_42d(depth):
    """The layer depths VIC 4.2.d holds: read_soilparam.c:342 rounds each to
    the millimetre IN SINGLE PRECISION, (float)(int)(d * 1000 + 0.5) / 1000,
    so 0.3 m is 0.30000001192... (VIC 5 rounds in double). MOISTFRACT
    divides by these."""
    d = np.asarray(depth, dtype=np.float64)
    k = np.floor(d * 1000.0 + 0.5).astype(np.int64)      # (int), d > 0
    return (k.astype(np.float32) / np.float32(1000.0)).astype(np.float64)


def records_42d(out, times, spec, names=None, depth=None, carry=False,
                aggs=None, out_dt=None, init_moist=None):
    """VIC 4.2.d's records of `names` (OUT_*) from basin.simulate's `out`
    [step, cell, ...] (or a dict keyed by OUT_* names, such as VIC 4.2.d's
    own per-step output read back), over the flattened (cell, step) index:

        (data {OUT_*: [record(, elem)]}, records [(first, last)],
         tails, missing {OUT_*: reason})

    A record belongs to cell last // n_steps and is stamped with step
    last % n_steps. OUT_AERO_RESIST* are 1 / the mean OUT_AERO_COND*;
    MOISTFRACT and ALMA_OUTPUT are applied as `spec` says. Under MOISTFRACT
    OUT_DELSOILMOIST is the change in the layers' summed volumetric
    fractions, which for the first step needs `init_moist` (mm, [layer,
    cell] like `depth`: the state the run started from).
    """
    t = as_datetimes(times)
    n = len(t)
    out_dt = int(out_dt or spec.out_dt)
    ratio = out_dt // int(spec.dt)
    names = list(names) if names is not None else sorted(
        set(v for f in spec.files for v in f.names))
    aggs = dict(aggs or {})
    need = list(names)
    # put_data.c:586-607: OUT_DELSOILMOIST is the sum of OUT_SOIL_MOIST over
    # the layers minus the previous step's sum, and under MOISTFRACT those
    # are already fractions (put_data.c:906-909)
    frac_del = bool(spec.moistfract) and 'OUT_DELSOILMOIST' in names
    for nm in names:            # what a written variable is computed from
        if nm.startswith('OUT_AERO_RESIST'):
            need.append(nm.replace('RESIST', 'COND'))
        if spec.alma and nm == 'OUT_SUB_SNOW':
            need.append('OUT_SUB_CANOP')
    if frac_del:
        need.append('OUT_SOIL_MOIST')
    n_cells = None
    series, missing = {}, {}
    for nm in dict.fromkeys(need):
        if nm.startswith('OUT_AERO_RESIST') or (
                frac_del and nm == 'OUT_DELSOILMOIST'):
            continue
        x, src = _series_42d(out, nm)
        if x is None:
            missing[nm] = src
            continue
        if x.shape[0] != n:
            raise ValueError('%s (%s) has %d steps and the run %d'
                             % (nm, src, x.shape[0], n))
        if spec.moistfract and nm in ('OUT_SOIL_LIQ', 'OUT_SOIL_ICE',
                                      'OUT_SOIL_MOIST'):
            if depth is None:
                raise ValueError('MOISTFRACT TRUE needs the layer depths')
            d = torch.as_tensor(depth_42d(depth), dtype=x.dtype,
                                device=x.device)
            # put_data.c:906-909: moist / (depth[layer] * 1000.)
            x = x / (d.T * 1000.0)
        series[nm] = x
        n_cells = x.shape[1]
    if frac_del:
        if 'OUT_SOIL_MOIST' not in series:
            missing['OUT_DELSOILMOIST'] = (
                'under MOISTFRACT VIC 4.2.d writes it from OUT_SOIL_MOIST '
                '(%s)' % missing.get('OUT_SOIL_MOIST'))
        elif init_moist is None:
            missing['OUT_DELSOILMOIST'] = (
                'under MOISTFRACT VIC 4.2.d writes the change in the summed '
                'volumetric fractions (put_data.c:586-607); its first step '
                'needs the initial layer moisture, which was not given')
        else:
            f = series['OUT_SOIL_MOIST']                 # [step, cell, layer]
            d = torch.as_tensor(depth_42d(depth), dtype=f.dtype,
                                device=f.device)
            f0 = torch.as_tensor(np.asarray(init_moist, np.float64),
                                 dtype=f.dtype, device=f.device).T \
                / (d.T * 1000.0)                        # [cell, layer]
            tot, tot0 = torch.zeros_like(f[..., 0]), torch.zeros_like(f0[:, 0])
            for k in range(f.shape[-1]):                # the C's layer order
                tot, tot0 = tot + f[..., k], tot0 + f0[:, k]
            series['OUT_DELSOILMOIST'] = torch.cat(
                [(tot[0] - tot0).unsqueeze(0), tot[1:] - tot[:-1]], 0)
    if n_cells is None:
        return {}, [], [], missing
    recs, tails = windows_42d(n, ratio, n_cells, carry=carry)
    data = {}
    for nm, x in series.items():
        data[nm] = aggregate_42d(_flatten(x), recs,
                                 aggs.get(nm) or default_agg_42d(nm), ratio)
    for nm in names:
        if nm.startswith('OUT_AERO_RESIST'):
            cond = nm.replace('RESIST', 'COND')
            if cond in data:
                data[nm] = torch.ones_like(data[cond]) / data[cond]
            else:
                missing[nm] = ('VIC 4.2.d writes it as 1 / the mean %s, which '
                               'this run does not have' % cond)
    if spec.alma:
        if 'OUT_SUB_SNOW' in data and 'OUT_SUB_CANOP' not in data:
            del data['OUT_SUB_SNOW']
            missing['OUT_SUB_SNOW'] = ('ALMA_OUTPUT adds OUT_SUB_CANOP to it '
                                       '(put_data.c:737), which this run does '
                                       'not have')
        data = _apply_alma_42d(data, out_dt)
    return ({k: v for k, v in data.items() if k in names}, recs, tails,
            {k: v for k, v in missing.items() if k in names})


def _c_printf(fmt, v):
    """printf of one double, as glibc prints it (a NaN with its sign)."""
    if v != v:
        return '-nan' if math.copysign(1.0, v) < 0 else 'nan'
    return fmt % v


def _date_fields(t, out_dt):
    return (t.year, t.month, t.day, t.hour) if out_dt < 24 else \
        (t.year, t.month, t.day)


def _header_42d(f, spec, t0, nrecs, nelem):
    """write_header.c for one file: NRECS is the run's model steps, the
    start date its first step's, NVARS the declared count + the dates."""
    nvars = f.nvars + (4 if f.out_dt < 24 else 3)
    if not spec.binary:
        cols = []
        for nm in f.names:
            k = nelem.get(nm, 1)
            cols += ['%s_%d' % (nm, e) for e in range(k)] if k > 1 else [nm]
        head = ('# NRECS: %d\n# DT: %d\n# STARTDATE: %04d-%02d-%02d '
                '%02d:00:00\n# ALMA_OUTPUT: %d\n# NVARS: %d\n'
                % (nrecs, f.out_dt, t0.year, t0.month, t0.day, t0.hour,
                   int(spec.alma), nvars))
        head += '# ' + ('YEAR\tMONTH\tDAY\tHOUR\t' if f.out_dt < 24
                        else 'YEAR\tMONTH\tDAY\t')
        return (head + '\t '.join(cols) + '\n').encode()
    dates = ['YEAR', 'MONTH', 'DAY'] + (['HOUR'] if f.out_dt < 24 else [])
    ents = [(d, OUT_TYPE_CODES_42D['OUT_TYPE_INT'], 1.0) for d in dates]
    for nm in f.names:
        k = nelem.get(nm, 1)
        for e in range(k):
            ents.append(('%s_%d' % (nm, e) if k > 1 else nm,
                         OUT_TYPE_CODES_42D[spec.type_of(nm)],
                         spec.mult_of(nm)))
    part2 = b''.join(struct.pack('=B', len(s)) + s.encode()
                     + struct.pack('=Bf', c, m) for s, c, m in ents)
    nbytes2 = 2 + len(part2)
    nbytes1 = 2 + 4 + 4 + 16 + 1 + 1
    nbytes = 4 * 2 + 2 + nbytes1 + nbytes2
    return (struct.pack('=4H', *([0xFFFF] * 4)) + struct.pack('=H', nbytes)
            + struct.pack('=H', nbytes1)
            + struct.pack('=6i', nrecs, f.out_dt, t0.year, t0.month, t0.day,
                          t0.hour)
            # char tmp_ALMA_OUTPUT, char Nvars (a byte, as C stores it)
            + struct.pack('=BB', int(spec.alma), nvars & 0xFF)
            + struct.pack('=H', nbytes2) + part2)


MANIFEST_42D = '.torchvic_written'


def _guard_42d(result_dir, paths):
    """Only a file TorchVIC wrote may be replaced: VIC 4.2.d's own output has
    the same names in the same RESULT_DIR. What TorchVIC writes is listed in
    RESULT_DIR/.torchvic_written."""
    man = os.path.join(result_dir, MANIFEST_42D)
    ours = set(open(man).read().split()) if os.path.exists(man) else set()
    for p in paths:
        if os.path.exists(p) and os.path.basename(p) not in ours:
            raise FileExistsError(
                '%s exists and was not written by TorchVIC (VIC 4.2.d\'s own '
                'output?); not overwriting it -- give run --result-dir DIR to '
                'write TorchVIC\'s files elsewhere' % p)
    with open(man, 'w') as fh:
        fh.write('\n'.join(sorted(ours | set(os.path.basename(p)
                                             for p in paths))) + '\n')


def _nelem_42d(name, out):
    """How many columns VIC 4.2.d writes for a variable TorchVIC lacks."""
    e = outvar_metadata_42d().get(name, {}).get('nelem', '1')
    if e == 'options.Nlayer':
        for k in ('soil_liq', 'soil_ice'):
            if isinstance(out.get(k), torch.Tensor) and out[k].dim() == 3:
                return int(out[k].shape[2])
    return 1


def write_42d(out, times, lat, lon, spec, result_dir, depth=None,
              carry=False, log=None, init_moist=None):
    """Write VIC 4.2.d's per-cell output files for `out` (basin.simulate's,
    [step, cell, ...]) by `spec`: one file per OUTFILE and cell,
    <result_dir>/<prefix>_<lat>_<lon>. Returns the paths. `depth` and
    `init_moist` ([layer, cell], m and mm) serve MOISTFRACT."""
    say = _announcer(log)
    t = as_datetimes(times)
    n = len(t)
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    ncell = len(lat)
    dec = int(spec.grid_decimal)
    tags = ['%.*f_%.*f' % (dec, a, dec, b) for a, b in zip(lat, lon)]
    if len(set(tags)) < ncell:
        raise ValueError('GRID_DECIMAL %d gives two cells the same file name; '
                         'VIC 4.2.d would overwrite one with the other -- '
                         'set GRID_DECIMAL higher' % dec)
    skiprec = skip_records_42d(t[0], spec.dt, spec.skipyear) \
        if spec.skipyear else 0
    if spec.skipyear:
        say('SKIPYEAR %d: the periods ending before record %d (%s) are not '
            'written' % (spec.skipyear, skiprec,
                         t[skiprec] if skiprec < n else 'after the run'))
    os.makedirs(result_dir, exist_ok=True)
    paths, blobs = [], {}
    for f in spec.files:
        data, recs, tails, missing = records_42d(
            out, t, spec, names=f.names, depth=depth, carry=carry,
            aggs=f.aggs, out_dt=f.out_dt, init_moist=init_moist)
        if not data:
            say('OUTFILE %s: TorchVIC has none of its variables (%s); not '
                'written' % (f.prefix, '; '.join('%s: %s' % kv
                                                 for kv in missing.items())))
            continue
        if tails and ncell > 1 and not carry:
            say('OUTFILE %s: each cell ends with a partial OUT_STEP %d period '
                '(%d step(s)); VIC 4.2.d carries it into the NEXT cell\'s '
                'first record (its step_count and aggdata are static, '
                'put_data.c:198) -- written here with each cell starting '
                'clean; carry=True reproduces 4.2.d'
                % (f.prefix, f.out_dt, tails[0][1] - tails[0][0] + 1))
        elif tails:
            say('OUTFILE %s: the last %d step(s) are a partial OUT_STEP period '
                'and are not written, as VIC 4.2.d does not write them'
                % (f.prefix, tails[-1][1] - tails[-1][0] + 1))
        if missing:
            say('OUTFILE %s: TorchVIC has no %s; written as nan so the columns '
                'stay where VIC 4.2.d puts them'
                % (f.prefix, '; '.join('%s (%s)' % kv
                                       for kv in missing.items())))
        nrec = len(recs)
        nelem, cols = {}, []
        for nm in f.names:
            if nm in data:
                a = data[nm].detach().cpu().numpy().astype(np.float64)
                a = a.reshape(nrec, -1)
            else:
                a = np.full((nrec, _nelem_42d(nm, out)), np.nan)
            nelem[nm] = a.shape[1]
            if spec.binary:
                a = a * np.float64(spec.mult_of(nm))    # put_data.c:769
            cols.append((nm, a))
        cell_of = [b // n for _, b in recs]
        step_of = [b % n for _, b in recs]
        head = _header_42d(f, spec, t[0], n, nelem) if spec.prt_header \
            else b''
        by_cell = [[] for _ in range(ncell)]
        for r in range(nrec):
            if step_of[r] >= skiprec:
                by_cell[cell_of[r]].append(r)
        for c in range(ncell):
            path = os.path.join(result_dir, '%s_%s' % (f.prefix, tags[c]))
            parts = [head]
            for r in by_cell[c]:
                d = _date_fields(t[step_of[r]], f.out_dt)
                if spec.binary:
                    parts.append(np.asarray(d, dtype=np.int32).tobytes())
                    for nm, a in cols:
                        dtype, lo, hi = _BIN_DTYPES_42D[spec.type_of(nm)]
                        v = a[r]
                        if lo is not None:
                            v = np.trunc(v)       # the C cast truncates
                            if not np.all(np.isfinite(v)) or \
                                    np.any(v < lo) or np.any(v > hi):
                                raise ValueError(
                                    '%s * %g = %s does not fit %s; VIC '
                                    '4.2.d\'s C cast is undefined there'
                                    % (nm, spec.mult_of(nm), a[r],
                                       spec.type_of(nm)))
                        parts.append(v.astype(dtype).tobytes())
                else:
                    vals = []
                    for nm, a in cols:
                        fmt = spec.format_of(nm)
                        vals += [_c_printf(fmt, float(v)) for v in a[r]]
                    parts.append((('%04i\t%02i\t%02i\t%02i\t' % d
                                   if len(d) == 4 else
                                   '%04i\t%02i\t%02i\t' % d)
                                  + '\t '.join(vals) + '\n').encode())
            blobs[path] = b''.join(parts)
            paths.append(path)
    _guard_42d(result_dir, paths)
    for p in paths:
        with open(p, 'wb') as fh:
            fh.write(blobs[p])
    say('  wrote %d VIC 4.2.d %s file(s) in %s (OUT_STEP %s h at a %d-hour '
        'step: %s)' % (len(paths), 'BINARY' if spec.binary else 'ASCII',
                       result_dir, spec.out_dt, spec.dt,
                       ', '.join(f.prefix for f in spec.files)))
    return paths


def write_history_42d(out, case, result_dir=None, output_global=None,
                      spec=None, carry=False, streams=None, log=None):
    """write_history under VIC 4.2.d's convention: `spec` (default
    output_spec_42d(case, output_global, streams)) written as VIC 4.2.d's
    per-cell files. Returns the paths."""
    say = _announcer(log)
    if spec is None:
        spec = output_spec_42d(case, output_global=output_global,
                               streams=streams, log=say)
    if int(case.n_cells) != int(case.dom.n_active):
        raise ValueError('the case has %d cells and the domain %d active '
                         'ones: an ensemble has no VIC 4.2.d file layout'
                         % (case.n_cells, case.dom.n_active))
    gp = _as_gp(output_global) if output_global is not None else case.gp
    result_dir = resolve_result_dir(gp, result_dir, log=say)
    depth = init = None
    if spec.moistfract:
        depth = np.asarray(case.par.soil['depth'], np.float64)
        # the state basin.initial_states starts from: the file's init_moist
        # clipped at max_moist, [cell, layer] -> [layer, cell]
        init = torch.minimum(case.init_moist,
                             case.soil.max_moist.expand_as(case.init_moist))
        init = init.detach().cpu().double().numpy().T
    return write_42d(out, case.dates, np.asarray(case.dom.lat),
                     np.asarray(case.dom.lon), spec, result_dir, depth=depth,
                     carry=carry, log=say, init_moist=init)


def read_ascii_42d(path, columns=None):
    """One VIC 4.2.d ASCII output file: (date rows [(y, m, d[, h])],
    {column: float64 array}). Column names come from its PRT_HEADER header
    (`OUT_SOIL_MOIST_0`, ... for a layered variable), or from `columns`."""
    head, rows = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                head = line[1:].split()
                continue
            if line.strip():
                rows.append(line.split())
    names = list(columns) if columns is not None else head
    if names is None:
        raise ValueError('%s has no header (PRT_HEADER FALSE): give columns'
                         % path)
    nd = sum(1 for c in names if c in ('YEAR', 'MONTH', 'DAY', 'HOUR'))
    if columns is not None and not nd and rows:
        nd = len(rows[0]) - len(names)
    names = [c for c in names if c not in ('YEAR', 'MONTH', 'DAY', 'HOUR')]
    dates = [tuple(int(x) for x in r[:nd]) for r in rows]
    vals = np.array([[float(x) for x in r[nd:]] for r in rows],
                    dtype=np.float64).reshape(len(rows), len(names))
    return dates, {c: vals[:, i] for i, c in enumerate(names)}


def outputs_from_42d(paths, columns=None, dtype=torch.float64):
    """VIC 4.2.d's per-step ASCII files of one OUTFILE, one path per cell,
    as {OUT_*: tensor [step, cell(, elem)]} and the steps' start times --
    the `out` and `times` records_42d / write_42d take."""
    per = [read_ascii_42d(p, columns) for p in paths]
    dates = per[0][0]
    for p, (d, _) in zip(paths, per):
        if d != dates:
            raise ValueError('%s has other dates than %s' % (p, paths[0]))
    out, groups = {}, {}
    for c in per[0][1]:
        m = re.match(r'(OUT_\w+?)_(\d+)$', c)
        if m and m.group(1) not in per[0][1] and is_vic42d_outvar(m.group(1)):
            groups.setdefault(m.group(1), []).append(c)
        else:
            groups.setdefault(c, []).append(c)
    for name, cols in groups.items():
        a = np.stack([np.stack([v[1][c] for c in cols], -1) for v in per], 1)
        out[name] = torch.as_tensor(a if len(cols) > 1 else a[..., 0],
                                    dtype=dtype)
    times = [_dt.datetime(*(d + (0,) * (4 - len(d)))[:4]) for d in dates]
    return out, times


# -- OUTPUT_FORCE: VIC 4.2.d's write_forcing_file.c ---------------------------
#: write_forcing_file.c:46-58: the variables it fills; any other variable a
#: global file lists under OUTPUT_FORCE is written as the 0 the C leaves in it.
FORCE_VARS_42D = ('OUT_AIR_TEMP', 'OUT_CATM', 'OUT_COSZEN', 'OUT_DENSITY',
                  'OUT_FDIR', 'OUT_LONGWAVE', 'OUT_PAR', 'OUT_PREC',
                  'OUT_PRESSURE', 'OUT_QAIR', 'OUT_REL_HUMID',
                  'OUT_SHORTWAVE', 'OUT_TSKC', 'OUT_VP', 'OUT_VPD',
                  'OUT_WIND', 'OUT_RAINF', 'OUT_SNOWF')
#: vicNl_def.h: EPS (:300), kPa2Pa (:306), SECPHOUR (:277), KELVIN (:288)
_EPS_42D, _KPA2PA_42D, _KELVIN_42D = 0.62196351, 1000., 273.15
#: vicNl_def.h Rd (:302), and PS_PM, G, T_LAPSE (initialize_atmos.c:1075)
_RD_42D, _PS_PM_42D, _G_42D, _T_LAPSE_42D = 287.0, 101300., 9.81, -0.0065


def force_data_42d(fields, elevation=None, dt_hours=1, max_snow_temp=0.5,
                   min_rain_temp=-0.5, min_wind_speed=0.1, plapse=True,
                   alma=False):
    """write_forcing_file.c:46-91 on the forcing `fields` ({name: [step,
    cell]} in TorchVIC's canonical names and units: tair C, prcp mm,
    shortwave/longwave W/m2, press and vp in Pa, wind m/s, and density
    kg/m3 when the forcing supplies it), as {OUT_*: float64 [step, cell]}.

    Everything VIC 4.2.d derives is derived as it does, in double
    arithmetic and the C's order of operations: the wind floor
    (initialize_atmos.c:655, :669), the pressure from the density or the
    elevation when none is supplied (:1069-1109), the density from the
    pressure (:1153-1167), the vapour pressure deficit (:1340-1345), the
    rain/snow partition (write_forcing_file.c:60-71) and the ALMA units
    (:82-91). `plapse` is the global file's PLAPSE (default TRUE).
    """
    import numpy as _np

    def a(name):
        v = fields.get(name)
        return None if v is None else _np.asarray(
            v.detach().cpu().numpy() if hasattr(v, 'detach') else v,
            dtype=_np.float64)
    tair, prcp = a('tair'), a('prcp')
    sw, lw, wind = a('shortwave'), a('longwave'), a('wind')
    press, vp, dens = a('press'), a('vp'), a('density')
    for nm, v in (('AIR_TEMP', tair), ('PREC', prcp), ('SWDOWN', sw),
                  ('LWDOWN', lw), ('VP', vp)):
        if v is None:
            raise ValueError('OUTPUT_FORCE: the forcing has no %s, which VIC '
                             '4.2.d would derive with MTCLIM from other '
                             'variables (initialize_atmos.c); give it, or a '
                             'daily forcing' % nm)
    if wind is None:
        wind = _np.full_like(tair, 1.5)       # DEFAULT_WIND_SPEED, :685
    else:
        wind = _np.where(wind < min_wind_speed, min_wind_speed, wind)
    if press is None:
        if dens is not None:
            press = (_KELVIN_42D + tair) * dens * _RD_42D if plapse else \
                (275.0 + tair) * dens / 0.003486
        elif plapse:
            if elevation is None:
                raise ValueError('OUTPUT_FORCE: no pressure and no cell '
                                 'elevation to estimate it from')
            el = _np.asarray(elevation, dtype=_np.float64)[None, :]
            press = _PS_PM_42D * _np.frompyfunc(math.exp, 1, 1)(
                -el * _G_42D / (_RD_42D * (_KELVIN_42D + tair
                                           + 0.5 * el * _T_LAPSE_42D))
            ).astype(_np.float64)
        else:
            press = _np.full_like(tair, 95500.)
    if dens is None:
        dens = (press / (_RD_42D * (_KELVIN_42D + tair)) if plapse
                else 0.003486 * press / (275.0 + tair))
    # svp.c, in Pa; a vapour pressure above saturation is set to it and the
    # deficit to 0 (initialize_atmos.c:1340-1344)
    A_, B_, C_ = 0.61078, 17.269, 237.3
    exp_ = _np.frompyfunc(math.exp, 1, 1)      # the C library's, as vicNl's
    es = A_ * exp_((B_ * tair) / (C_ + tair)).astype(_np.float64)
    es = _np.where(tair < 0, es * (1.0 + .00972 * tair
                                   + .000042 * tair * tair), es)
    es = es * 1000.
    over = (es - vp) < 0
    vpd = _np.where(over, 0.0, es - vp)
    vp = _np.where(over, es, vp)          # initialize_atmos.c:1340-1344
    z = _np.zeros_like(tair)
    d = {'OUT_AIR_TEMP': tair.copy(), 'OUT_PREC': prcp.copy(),
         'OUT_SHORTWAVE': sw.copy(), 'OUT_LONGWAVE': lw.copy(),
         'OUT_DENSITY': dens, 'OUT_PRESSURE': press / _KPA2PA_42D,
         'OUT_QAIR': _EPS_42D * vp / press,
         'OUT_REL_HUMID': 100. * vp / (vp + vpd),
         'OUT_VP': vp / _KPA2PA_42D, 'OUT_VPD': vpd / _KPA2PA_42D,
         'OUT_WIND': wind}
    for nm, key, scale in (('OUT_CATM', 'catm', 1e6), ('OUT_COSZEN', 'coszen', 1),
                           ('OUT_FDIR', 'fdir', 1), ('OUT_PAR', 'par', 1),
                           ('OUT_TSKC', 'tskc', 1)):
        v = a(key)
        d[nm] = z.copy() if v is None else (v * scale if scale != 1 else v)
    t = d['OUT_AIR_TEMP']
    if min_rain_temp > max_snow_temp:
        raise ValueError('MIN_RAIN_TEMP above MAX_SNOW_TEMP')
    rain = _np.where(t >= max_snow_temp, d['OUT_PREC'],
                     _np.where(t <= min_rain_temp, 0.0,
                               ((t - min_rain_temp)
                                / (max_snow_temp - min_rain_temp))
                               * d['OUT_PREC']))
    d['OUT_RAINF'] = rain
    d['OUT_SNOWF'] = d['OUT_PREC'] - rain
    d['OUT_SNOWF'] = _np.where(t >= max_snow_temp, 0.0,
                               _np.where(t <= min_rain_temp, d['OUT_PREC'],
                                         d['OUT_SNOWF']))
    if alma:
        dt_sec = int(dt_hours * 3600)
        for nm in ('OUT_PREC', 'OUT_RAINF', 'OUT_SNOWF'):
            d[nm] = d[nm] / dt_sec
        d['OUT_AIR_TEMP'] = d['OUT_AIR_TEMP'] + _KELVIN_42D
        for nm in ('OUT_PRESSURE', 'OUT_VP', 'OUT_VPD'):
            d[nm] = d[nm] * 1000
    return d


def write_forcing_42d(data, times, nrecs, lat, lon, spec, result_dir,
                      dt_hours=1, log=None):
    """VIC 4.2.d's forcing files, write_forcing_file.c + write_data.c +
    write_header.c under OUTPUT_FORCE: one file per cell,
    <result_dir>/<prefix>_<lat>_<lon>, every record of `data` ({OUT_*:
    [step, cell]}, from force_data_42d) as one line (ASCII: the values
    only -- no date columns under OUTPUT_FORCE, write_data.c:126 -- joined
    by "\\t ") or one run of typed values (BINARY_OUTPUT, each multiplied
    by its mult and cast as the C does). `nrecs` is the model's record
    count for PRT_HEADER's NRECS. Returns the paths."""
    say = _announcer(log)
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    dec = int(spec.grid_decimal)
    tags = ['%.*f_%.*f' % (dec, a, dec, b) for a, b in zip(lat, lon)]
    if len(set(tags)) < len(lat):
        raise ValueError('GRID_DECIMAL %d gives two cells the same file name; '
                         'VIC 4.2.d would overwrite one with the other -- '
                         'set GRID_DECIMAL higher' % dec)
    t = as_datetimes(times)
    n = len(t)
    os.makedirs(result_dir, exist_ok=True)
    paths, blobs = [], {}
    for f in spec.files:
        cols = []
        for nm in f.names:
            a = data.get(nm)
            a = np.zeros((n, len(lat))) if a is None else np.asarray(a)
            if spec.binary:
                a = a * np.float64(spec.mult_of(nm))      # write_forcing:98
            cols.append((nm, a))
        head = b''
        if spec.prt_header:
            head = _force_header_42d(f, spec, t[0], nrecs, dt_hours)
        for c in range(len(lat)):
            path = os.path.join(result_dir, '%s_%s' % (f.prefix, tags[c]))
            parts = [head]
            for r in range(n):
                if spec.binary:
                    for nm, a in cols:
                        dtype, lo, hi = _BIN_DTYPES_42D[spec.type_of(nm)]
                        v = np.asarray(a[r, c])
                        if lo is not None:
                            v = np.trunc(v)
                            if not np.isfinite(v) or v < lo or v > hi:
                                raise ValueError(
                                    '%s * %g = %s does not fit %s; VIC '
                                    '4.2.d\'s C cast is undefined there'
                                    % (nm, spec.mult_of(nm), a[r, c],
                                       spec.type_of(nm)))
                        parts.append(v.astype(dtype).tobytes())
                else:
                    parts.append(('\t '.join(
                        _c_printf(spec.format_of(nm), float(a[r, c]))
                        for nm, a in cols) + '\n').encode())
            blobs[path] = b''.join(parts)
            paths.append(path)
    _guard_42d(result_dir, paths)
    for p in paths:
        with open(p, 'wb') as fh:
            fh.write(blobs[p])
    say('  wrote %d VIC 4.2.d forcing %s file(s) in %s (%d records each: %s)'
        % (len(paths), 'BINARY' if spec.binary else 'ASCII', result_dir, n,
           ', '.join(f.prefix for f in spec.files)))
    return paths


def _force_header_42d(f, spec, t0, nrecs, dt_hours):
    """write_header.c under OUTPUT_FORCE: no date fields (:108, :165, :263,
    :279); NRECS is global.nrecs, DT the output step."""
    nvars = f.nvars
    if not spec.binary:
        head = ('# NRECS: %d\n# DT: %d\n# STARTDATE: %04d-%02d-%02d '
                '%02d:00:00\n# ALMA_OUTPUT: %d\n# NVARS: %d\n# '
                % (nrecs, f.out_dt or dt_hours, t0.year, t0.month, t0.day,
                   t0.hour, int(spec.alma), nvars))
        return (head + '\t '.join(f.names) + '\n').encode()
    ents = [(nm, OUT_TYPE_CODES_42D[spec.type_of(nm)], spec.mult_of(nm))
            for nm in f.names]
    part2 = b''.join(struct.pack('=B', len(s)) + s.encode()
                     + struct.pack('=Bf', c, m) for s, c, m in ents)
    nbytes2 = 2 + len(part2)
    nbytes1 = 2 + 4 + 4 + 16 + 1 + 1
    nbytes = 4 * 2 + 2 + nbytes1 + nbytes2
    return (struct.pack('=4H', *([0xFFFF] * 4)) + struct.pack('=H', nbytes)
            + struct.pack('=H', nbytes1)
            + struct.pack('=6i', nrecs, f.out_dt or dt_hours, t0.year,
                          t0.month, t0.day, t0.hour)
            + struct.pack('=BB', int(spec.alma), nvars & 0xFF)
            + struct.pack('=H', nbytes2) + part2)


def coszen_42d(lat, lng, time_zone_lng, day_in_year, hour):
    """compute_coszen.c, in the C's double arithmetic: the cosine of the
    solar zenith angle at (day_in_year, hour) for a cell at `lat`, `lng`
    in a time zone centred on `time_zone_lng` (degrees). `hour` is the C
    int dmy.hour (initialize_atmos.c:1592-1606 truncates hour + 0.5 *
    SNOW_STEP into it) counted from the start of the run, not the day:
    cos((hour + offset - 12) * PI / 12) is not periodic in the C. PI is the
    3.1415927 of vicNl_def.h:309, which is what compute_coszen.c sees (the
    3.14159265 of mtclim_constants_vic.h does not replace it; checked
    against vicNl to the last digit), not pi. Arrays broadcast; `lat`,
    `lng` and `time_zone_lng` are per cell."""
    PI, MINDECL, DAYSOFF, RADPERDAY = 3.1415927, -0.4092797, 11.25, 0.017214
    cos_ = np.frompyfunc(math.cos, 1, 1)      # the C library's, as vicNl's
    sin_ = np.frompyfunc(math.sin, 1, 1)

    def f(fn, x):
        return np.asarray(fn(np.asarray(x, np.float64)), dtype=np.float64)
    lat = np.asarray(lat, np.float64)
    coslat = f(cos_, lat * PI / 180)
    sinlat = f(sin_, lat * PI / 180)
    decl = MINDECL * f(cos_, (np.asarray(day_in_year, np.float64) + DAYSOFF)
                       * RADPERDAY)
    cosdecl, sindecl = f(cos_, decl), f(sin_, decl)
    cosegeom = coslat * cosdecl
    sinegeom = sinlat * sindecl
    hour_offset = (np.asarray(time_zone_lng, np.float64)
                   - np.asarray(lng, np.float64)) * 24 / 360
    cosh = f(cos_, (np.asarray(hour, np.float64) + hour_offset - 12) * PI / 12)
    return cosegeom * cosh + sinegeom
