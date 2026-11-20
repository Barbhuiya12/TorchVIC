# Outputs and state files

A run produces history files (the variables you ask for, aggregated as VIC
aggregates them), optionally a state file that restarts the model, and a log on
the screen. The variables are listed in [Output variables](../OUTPUT_VARIABLES.md).

## History files

TorchVIC writes the global file's `OUTFILE` streams as VIC 5's image
driver writes them: NetCDF on the domain grid, one file per `HISTFREQ`
period, named `<OUTFILE prefix>.<start date>.nc`, with the domain's
`lat`/`lon`, fill value off the mask, `time` and `time_bnds`, and one
variable per `OUTVAR` line with VIC's `long_name`, `standard_name` and
`units`.

**Aggregation.** TorchVIC steps at the model step and produces every
output at that step; the files hold what `AGGFREQ` asks for. Each
variable is aggregated by VIC's own type:

| type of variable | aggregated as | example |
|---|---|---|
| water flux | summed over the period | `OUT_EVAP` of a day is the day's evaporation in mm |
| state | taken at the period's end | `OUT_SOIL_MOIST`, `OUT_SWE` |
| everything else | averaged | `OUT_AIR_TEMP`, `OUT_LATENT` |

`AGG_TYPE_MAX` and `AGG_TYPE_MIN` start from 0 as VIC 5's do
(`--true-extremes` gives the period's own extremes).
The aggregation period is anchored at the first model step, as VIC's
alarms are: a run from 00:00 gives calendar days; a run from 03:00 gives
03:00-to-03:00 days. A period that is not a whole number of steps is
refused (VIC writes nothing at all in that case). Every rule was measured
against the compiled image driver (`tests/test_output_aggregation.py`).

**Reading a file.**

```python
import xarray as xr
ds = xr.open_dataset('results/fluxes.2005-01-01.nc')
evap = ds['OUT_EVAP']                                   # [time, lat, lon], mm per period
print(ds.time.values[:3], ds['OUT_EVAP'].units)
basin_mean = evap.weighted(ds_domain['area'].fillna(0)).mean(('lat', 'lon'))
```

`OUT_SOIL_MOIST` has a `nlayer` dimension, `OUT_*_BAND` variables a
`snow_band` dimension, as in VIC.

**Several files.** `HISTFREQ NMONTHS 1` writes one file per month,
`NYEARS 1` one per year, `END` one file for the run; open them together
with `xr.open_mfdataset('results/fluxes.*.nc')`.

**The VIC 4.2.d convention.** A VIC 4.2.d global file (or
`--agg-convention vic4.2d`) writes 4.2.d's per-cell ASCII or BINARY files
with its `OUT_STEP` aggregation instead.

## State files

A state file holds the model at the end of one step for every (vegetation
class, snow band, cell): soil moisture and ice, canopy water, the snow
pack (age, cover, water equivalent, temperatures, density, cold content,
canopy snow), soil node temperatures, foliage temperature and the
energy-balance carry. TorchVIC writes and reads VIC 5.1.0's image-driver
layout (`vic_store.c`, `vic_restore.c`), NetCDF4 whatever `STATE_FORMAT`
says. In the verification matrix the variables of a state file written by
TorchVIC equal VIC's to 1e-12, and a run restarted from a state file written by
VIC agrees with VIC's own restart ([Verification](../verification.md)).

**Writing one**:

```text
STATENAME    states/state
STATEYEAR    2006
STATEMONTH   1
STATEDAY     1
STATESEC     0
```

writes `states/state.20060101_00000.nc`, the model at the end of
the step that ends at that time (here the last step of 2005). VIC writes
no file when no step ends there, and neither does TorchVIC (the log says
so). `STATENAME` resolves like the other paths (relative to the global
file's directory).

**Starting from one**:

```text
INIT_STATE   states/state.20060101_00000.nc
```

A state dated at the run's start continues the run without a gap; VIC does
not compare the two dates and neither refuses TorchVIC, which announces a
mismatch and starts from the file all the same. The state's cells,
layers, classes and bands must match the parameter file. Without
`INIT_STATE` the run starts from VIC's cold start (the parameter file's
`init_moist`, no snow, soil temperatures from `avg_T`).

A typical spin-up: run a few years ending at a `STATE*` time, then start the
production run from that file. On a GPU, reading a VIC state file has a known
problem in 0.4.8 with a workaround ([Known limitations](../about/limitations.md)).

## Outputs from Python

`torchvic.run(case)` returns a dict of tensors at the model step, keyed by
TorchVIC's own names (`runoff`, `baseflow`, `et`, `swe`, ... `[time,
cell]` on the active cells). `case.aggregate(out, 'NDAYS 1')` applies
VIC's aggregation and returns tensors plus the period start times;
`case.write_history(out, result_dir=...)` writes the files. The
[Python API](python-api.md) page lists the calls.

## Next step

[Route runoff to gauges](routing.md), compare with observations in [Observations and ensembles](study.md), or [couple VIC with other models](../coupled/overview.md).
