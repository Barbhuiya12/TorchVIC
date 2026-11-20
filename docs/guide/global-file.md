# The global file

The global parameter file is VIC 5's run description, and TorchVIC reads
it as VIC 5's image driver does: one option per line, `KEY value`,
comments after `#`, the last line of a repeated key winning. This page
goes through UW-Hydro's Walnut Gulch file block by block. The exhaustive
table of every key and value, with what TorchVIC does for each, is
[Options, value by value](../OPTIONS.md). The few options that have no
global-file key, and are given from Python, are described in
[Python API](python-api.md#options-with-no-global-file-key).

## Time

```text
MODEL_STEPS_PER_DAY   8        # the model step: 8 = three-hourly, 24 = hourly, 1 = daily
SNOW_STEPS_PER_DAY    8        # snow sub-steps per day (>= MODEL_STEPS_PER_DAY)
RUNOFF_STEPS_PER_DAY  24       # runoff sub-steps per day (>= MODEL_STEPS_PER_DAY)
STARTYEAR   2008
STARTMONTH  01
STARTDAY    01
ENDYEAR     2008
ENDMONTH    01
ENDDAY      10
CALENDAR    PROLEPTIC_GREGORIAN
```

The run covers `STARTYEAR/MONTH/DAY` 00:00 (or `STARTSEC` seconds after)
through the end of `ENDDAY`; `NRECS n` instead of the end date runs `n`
model steps. The forcing must cover the window, or the run is refused
before it starts. `CALENDAR` must agree with the forcing's calendar.
`--daily` on the command line overrides `MODEL_STEPS_PER_DAY` with 1.

## Physics

```text
FULL_ENERGY TRUE               # surface energy balance (FALSE: water balance only)
QUICK_FLUX  TRUE               # the two-node ground-heat-flux scheme (FALSE: the full soil thermal profile)
FROZEN_SOIL FALSE              # soil ice (TRUE runs with QUICK_FLUX TRUE after it, or IMPLICIT/EXP_TRANS)
NODES       3                  # soil thermal nodes (QUICK_FLUX runs 3 whatever this says, as VIC)
AERO_RESIST_CANSNOW AR_406     # the canopy-snow aerodynamic resistance variant
```

Other physics keys the file may carry, each run as VIC runs it:
`SNOW_DENSITY`, `BLOWING` and its five sub-options, `SPATIAL_SNOW`,
`SPATIAL_FROST`, `CARBON`, `CLOSE_ENERGY`, `COMPUTE_TREELINE`,
`CORRPREC`, `EXP_TRANS`, `IMPLICIT`, `NOFLUX`, `QUICK_SOLVE`,
`GRND_FLUX_TYPE`, `TFALLBACK`, `SHARE_LAYER_MOIST`, `CANOPY_LAYERS`,
`LAKES`, `LAKE_PROFILE`, `MAX_SNOW_ALBEDO`, `ORGANIC_FRACT`,
`BULK_DENSITY_COMB`, `JULY_TAVG_SUPPLIED`, `RC_MODE`, `VEGLIB_PHOTO`,
`BASEFLOW` (`ARNO` or `NIJSSEN2001`). VIC's own defaults apply when a key
is absent; note that VIC's `IMPLICIT` and `EXP_TRANS` default to TRUE
until a `FROZEN_SOIL FALSE` line resets them, so a file without frozen
soil should carry that line, as VIC itself requires.

The GPU runs the water balance and the energy balance with `QUICK_FLUX`
(TRUE or FALSE), `FROZEN_SOIL` and `IMPLICIT` inside CUDA graphs; the
other energy-balance configurations run on the GPU step by step, correct
and slower.

## The domain

```text
DOMAIN         ${VIC_SAMPLE_DATA}/image/WalnutGulch/parameters/domain.WalnutGulch.0.0625_deg.nc
DOMAIN_TYPE    LAT     lat
DOMAIN_TYPE    LON     lon
DOMAIN_TYPE    MASK    mask
DOMAIN_TYPE    AREA    area
DOMAIN_TYPE    FRAC    frac
DOMAIN_TYPE    YDIM    lat
DOMAIN_TYPE    XDIM    lon
```

`DOMAIN_TYPE` maps VIC's names to the file's variable and dimension names.

## The forcing

```text
FORCING1      ${VIC_SAMPLE_DATA}/image/WalnutGulch/forcings/livneh_NAmerExt_15Oct2014.WalnutGulch.
FORCE_TYPE    AIR_TEMP     temp
FORCE_TYPE    PREC         prec
FORCE_TYPE    PRESSURE     air_pressure
FORCE_TYPE    SWDOWN       shortwave
FORCE_TYPE    LWDOWN       longwave
FORCE_TYPE    VP           vapor_pressure
FORCE_TYPE    WIND         wind
WIND_H        10.0
```

`FORCING1` is the file prefix: the year and `.nc` are appended, so this
prefix names `livneh_NAmerExt_15Oct2014.WalnutGulch.2008.nc`. A prefix
ending in `.zarr` is one Zarr store. Each `FORCE_TYPE` line pairs a VIC
quantity with a variable of the file; units come from the file's
attributes ([Prepare your inputs](inputs.md)). `FORCING2` with
`ALB_SRC FROM_VEGHIST` (and `LAI_SRC`, `FCAN_SRC`) reads a per-step
vegetation history.

`WIND_H` is the wind measurement height. With it in the global file,
TorchVIC uses it for the land surface and says so; VIC 5's image driver
would use the parameter file's per-class heights there
([Differences from VIC](../differences.md)).

## The parameters

```text
PARAMETERS      ${VIC_SAMPLE_DATA}/image/WalnutGulch/parameters/params.WalnutGulch.L2015.0.0625_deg.nc
LAI_SRC         FROM_VEGPARAM     # LAI from the parameter file's per-cell values (FROM_VEGLIB: the class values)
FCAN_SRC        FROM_DEFAULT      # canopy fraction 1 (FROM_VEGPARAM/FROM_VEGLIB need `fcanopy` in the file)
ALB_SRC         FROM_VEGPARAM
NODES           3
SNOW_BAND       TRUE              # read the file's bands; every band with area runs as its own column
```

## Initial and saved state

```text
INIT_STATE   state/state.20031231_00000.nc    # start from a VIC state file (absent: VIC's cold start)
STATENAME    state/state                      # write state.YYYYMMDD_SSSSS.nc ...
STATEYEAR    2006
STATEMONTH   1
STATEDAY     1
STATESEC     0                                # ... holding the model at the end of the step that ends at this time
```

The `STATE*` date is the end of a step (check_save_state_flag.c): the
state after the last step of 2005 is dated 2006-01-01 00000.

Paths resolve as the other files' do: relative to the global file's
directory, environment variables expanded. [Outputs and state files](outputs.md).

## Output

```text
RESULT_DIR  ${VIC_SAMPLE_RESULTS}/sample_image     # --out overrides it
OUTFILE     fluxes.WalnutGulch.L2015              # a stream: this file prefix ...
COMPRESS    FALSE
OUT_FORMAT  NETCDF4
AGGFREQ     NDAYS 1                               # ... one record per day ...
HISTFREQ    NMONTHS 1                             # ... one file per month ...
OUTVAR OUT_PREC                                   # ... holding these variables
OUTVAR OUT_EVAP
OUTVAR OUT_RUNOFF
OUTVAR OUT_BASEFLOW
OUTVAR OUT_SWE
OUTVAR OUT_SOIL_MOIST
OUTVAR OUT_SOIL_TEMP
OUTVAR OUT_AIR_TEMP
```

Every `OUTFILE` opens a stream; the `AGGFREQ`, `HISTFREQ`, `OUT_FORMAT`,
`COMPRESS` and `OUTVAR` lines after it belong to it. `AGGFREQ` takes
`NSTEPS`, `NHOURS`, `NDAYS`, `NMONTHS`, `NYEARS` with a count, `DATE
yyyy-mm-dd`, or `END`; `OUTVAR name [format [type [mult [aggtype]]]]` as
VIC 5 (`*` for a default). The 165 `OUT_*` variables VIC 5.1.0 defines
are all known, each with VIC's default aggregation. A `RESULT_DIR` naming
an environment variable that is not set is refused rather than created as
a directory called `${...}`; `--out DIR` is the usual way to say where the
results go.

## Keys read to no effect, and refusals

`ATMOS_STEPS_PER_DAY`, `LAKE_NODES`, `LOG_DIR`, `ROUT_PARAM` and
`STATE_FORMAT` are read, cannot change a result in a configuration TorchVIC
runs, and are announced in the log. A key that is not an image-driver
option is announced and ignored, as VIC warns and ignores it. A value
TorchVIC does not run as VIC does stops the run with a message naming the
key, the value and the reason; the list is [Options](../OPTIONS.md).

## Next step

[Run the model](running.md). The value-by-value list of options is [Options](../OPTIONS.md).
