# Prepare your inputs

A TorchVIC run is a VIC 5 image-driver run, and it needs the same four
inputs. If you already run VIC 5's image driver, you have them and can go
straight to [Running the model](running.md). This page is for everyone
else: what each file holds, how TorchVIC reads it, and the ways to make
one.

| # | input | file | grid |
|---|---|---|---|
| 1 | domain | one NetCDF, `domain.nc` | `[lat, lon]` |
| 2 | parameters | one NetCDF, `params.nc` | `[..., lat, lon]`, the domain's grid |
| 3 | forcing | one NetCDF per year, `prefix.YYYY.nc`, or a Zarr store | `[time, lat, lon]`, the domain's grid |
| 4 | global parameter file | one text file | names the three above and the run |

The domain, parameter and forcing files must share one grid: the same
`lat` and `lon` coordinate values, in the same order. TorchVIC checks the
coordinates of the active cells and refuses a file on another grid.

## 1. The domain file

The grid and which of its cells to run.

| variable | dims | meaning |
|---|---|---|
| `lat`, `lon` | 1-D | cell-centre coordinates, degrees |
| `mask` | `(lat, lon)` | 1 where a cell may run, 0 elsewhere |
| `area` | `(lat, lon)` | cell area, m2 |
| `frac` | `(lat, lon)` | land fraction of the cell (1 for a land cell) |
| `elev` | `(lat, lon)` | elevation, m (used where the parameter file has no bands) |

The global file maps these names with `DOMAIN_TYPE` lines (`LAT lat`,
`LON lon`, `MASK mask`, `AREA area`, `FRAC frac`, `YDIM lat`, `XDIM lon`),
so other names are fine as long as the lines say so. The cells that run
are those with `run_cell = 1` in the parameter file, bounded by the
domain's mask; a cell with `run_cell = 1` outside the mask stops the run,
as it stops VIC.

A domain file is easy to write from any gridded land mask:

```python
import numpy as np, xarray as xr
lat = np.arange(8.03125, 37.6, 1/16); lon = np.arange(68.03125, 97.6, 1/16)
mask = land_mask_on_this_grid.astype('int32')                 # your [lat, lon] array
R = 6371008.8
dlat = dlon = np.radians(1/16)
area = (R**2 * dlon * (np.sin(np.radians(lat + 1/32)) - np.sin(np.radians(lat - 1/32))))[:, None] * np.ones_like(lon)
ds = xr.Dataset(
    dict(mask=(('lat', 'lon'), mask), area=(('lat', 'lon'), area), frac=(('lat', 'lon'), mask.astype('f4')),
         elev=(('lat', 'lon'), elevation_m)),
    coords=dict(lat=lat, lon=lon))
ds['area'].attrs['units'] = 'm2'
ds.to_netcdf('domain.nc')
```

## 2. The parameter file

Everything VIC knows about the land surface of each cell, on the domain
grid. VIC 5 puts the soil parameters, the vegetation mosaic, the vegetation
library and the snow bands in this one file. TorchVIC reads the variables
VIC 5.1.0's `vic_init.c` reads; the ones a configuration needs that are
absent stop the run with the variable's name.

**Per cell**: `run_cell`, `gridcell`, `infilt`, `Ds`, `Dsmax`, `Ws`, `c`,
`elev`, `avg_T`, `dp`, `off_gmt`, `rough`, `snow_rough`, `annual_prec`,
`fs_active`, `Nveg`; with `NIJSSEN2001_BASEFLOW`, `d1`..`d4` instead of
`Ds`..`c`.

**Per layer** (`nlayer, lat, lon`, usually three layers): `expt`, `Ksat`,
`phi_s`, `init_moist`, `depth`, `bubble`, `quartz`, `bulk_density`,
`soil_density`, `Wcr_FRACT`, `Wpwp_FRACT`, `resid_moist`; with
`ORGANIC_FRACT`, `organic`, `bulk_density_org`, `soil_density_org`.

**Per vegetation class** (`veg_class, lat, lon`): `Cv` (the cover fraction
of the class in the cell; the fractions of a cell sum to 1 with bare soil
as the last class), `overstory`, `rarc`, `rmin`, `wind_h`, `RGL`,
`rad_atten`, `wind_atten`, `trunk_ratio`; per class and month: `LAI`,
`albedo`, `veg_rough`, `displacement`, optionally `fcanopy`; per class and
root zone: `root_depth`, `root_fract`.

**Per snow band** (`snow_band, lat, lon`): `AreaFract`, `elevation`,
`Pfactor`. With `SNOW_BAND TRUE` every band with area runs as its own column,
at its lapsed air temperature and scaled precipitation, and every output is
area-weighted back to the cell; a cell with one band with area is the cell
itself. As VIC 5 does, TorchVIC then replaces the cell's `elev` by the
area-weighted mean of the band elevations and says so in the log
(`band_mean_elevation` in `torchvic/io/build.py`). The file needs `AreaFract`,
and `Pfactor` and `elevation` where a cell has more than one band with area; a
file without them is refused by name.

Three ways to get a parameter file:

* **An existing VIC 5 image-driver file.** VICGlobal and the UW-Hydro
  sample sets ship `params.nc` files on 1/16-degree grids; a regional cut
  on your domain grid works as it is.
* **VIC classic files.** `torchvic convert-classic` turns a classic soil
  file, vegetation parameter file, vegetation library and snow-band file
  into `domain.nc` and `params.nc`, read exactly as `vic_classic` reads
  them:

        torchvic convert-classic soil.txt vegparam.txt veglib.txt \
            --snowband snowbands.txt --snow-bands 5 --root-zones 3 \
            --veglib-fcan --fcan-src FROM_VEGLIB \
            --grid forcing.nc --domain domain.nc --params params.nc

    or `--global global.classic.txt` to take the files and options from a
    classic global file. `--grid` names a NetCDF file or Zarr store whose
    1-D `lat`/`lon` are the target grid (default: the cells' own extent);
    `--drop-outside` lists and omits cells beyond it. The conversion was
    checked against VIC's Stehekin sample (every variable equals UW-Hydro's own
    converted file except the classic driver's normalisations); a comparison of
    VIC's output on converted and on shipped files was not repeated for
    0.4.7.
* **Your own.** Write the variables above with xarray on the domain grid.
  `tools/build_isc_params.py` is an example that assembles a regional set
  from a soil parameter file and VICGlobal's vegetation and snow bands.

## 3. The forcing

Meteorology at every active cell and every model step, `[time, lat, lon]`
on the domain grid, one NetCDF file per calendar year named
`<prefix><year>.nc` (the global file gives `FORCING1 <prefix>`), or a Zarr
store (`FORCING1 /path/store.zarr`, `pip install -e .[zarr]`). The time
axis is a CF time coordinate (`hours since ...`); records must be
contiguous and equally spaced, and a file whose dates repeat or jump is
refused. Only the active cells and the run's time window are read, in
blocks, with the next block read on a background thread while the current
one computes.

The variable names are yours. The global file's `FORCE_TYPE` lines say
which variable is which:

```text
FORCE_TYPE    AIR_TEMP     temp
FORCE_TYPE    PREC         prec
FORCE_TYPE    PRESSURE     air_pressure
FORCE_TYPE    SWDOWN       shortwave
FORCE_TYPE    LWDOWN       longwave
FORCE_TYPE    VP           vapor_pressure
FORCE_TYPE    WIND         wind
```

### Sub-daily forcing (VIC 5's own convention)

| FORCE_TYPE | quantity | unit TorchVIC works in | units it converts from (the file's `units` attribute) |
|---|---|---|---|
| `AIR_TEMP` | air temperature | degC | `C` |
| `PREC` | precipitation per step | mm | `mm`, `mm/step`; rates (`mm/s`, `mm/h`, `mm/day`, `kg m-2 s-1`) are multiplied by the step length |
| `PRESSURE` | surface pressure | Pa | `Pa`, `kPa` |
| `SWDOWN` | downward shortwave | W m-2 | `W/m2` |
| `LWDOWN` | downward longwave | W m-2 | `W/m2` |
| `VP` | vapour pressure | Pa | `Pa`, `kPa` |
| `WIND` | wind speed | m s-1 | `m/s` |
| `QAIR`, `REL_HUMID` | specific / relative humidity | kg/kg, % | instead of `VP` |
| `CATM`, `PAR`, `FDIR`, `ALBEDO`, `LAI_IN`, `FCANOPY` | optional, as VIC 5 defines them | | |

The conversion is driven by the file's `units` attribute, never assumed:
a file in kPa is multiplied by 1000, a file with no units attribute for a
quantity that needs one is refused, and a unit TorchVIC does not know is
refused with the file and variable named. Temperatures in Kelvin are
refused (an offset, not a factor); convert them first. A `FORCE_TYPE` name
outside VIC 5's list stops the run.

### Daily forcing

TorchVIC accepts a daily forcing and disaggregates it on the device with
the MTCLIM algorithms that VIC 4.2 used and VIC 5 dropped, re-implemented
from MetSim 2.0.1 and compared with it
([Verification](../verification.md#daily-forcing-mtclim-against-metsim)). The
file holds one record per day with

```text
FORCE_TYPE    TMAX    tasmax      # daily maximum temperature, degC
FORCE_TYPE    TMIN    tasmin      # daily minimum temperature, degC
FORCE_TYPE    PREC    prcp        # daily total, mm
FORCE_TYPE    WIND    wind        # optional daily mean, m/s
FORCE_TYPE    VP      vp          # optional daily mean; estimated from TMIN when absent
FORCE_TYPE    SWDOWN  swdown      # optional daily mean; estimated when absent
FORCE_TYPE    LWDOWN  lwdown      # optional; estimated when absent
FORCE_TYPE    PRESSURE sp         # optional; from elevation when absent
```

Daily records are stamped at 00:00 of their day. MTCLIM produces the
sub-daily temperature, radiation, humidity and pressure at
`MODEL_STEPS_PER_DAY`, reading 90 days of the file around the run window
so that the first days see the same temperature-range history as inside a
long run. An observed daily shortwave, vapour pressure, longwave or
pressure is used in preference to the estimate, and the log says which
fields were read and which were estimated. Supply vapour pressure, radiation
and pressure when you have them rather than leaving them to the estimate.

With `MODEL_STEPS_PER_DAY 1` the model itself steps daily; a day with
snow is sub-stepped inside at `SNOW_STEPS_PER_DAY`, as VIC does.

### Making a forcing

Any gridded product works once it is on the domain grid with a CF time
axis. `tools/edh_era5_daily.py` reduces hourly ERA5 to the daily fields
above and `tools/build_isc_forcing.py` assembles them into a Zarr store;
`benchmarks/make_isc_forcing.py subdaily` writes a three-hourly VIC 5
forcing from a daily one through the same MTCLIM chain. A quick check of
a new forcing before a long run:

```bash
torchvic run global_param.txt --core 5.1.0 --steps 8 --device cpu --out check_run
```

reads one day, prints every unit decision and writes nothing it cannot
justify.

## 4. The global parameter file

One text file that names the three files above and describes the run. It
is VIC 5's own format, and every line is explained in
[The global parameter file](global-file.md). The shortest useful file:

```text
MODEL_STEPS_PER_DAY   8
SNOW_STEPS_PER_DAY    8
RUNOFF_STEPS_PER_DAY  8
STARTYEAR 2004   STARTMONTH 1   STARTDAY 1
ENDYEAR   2005   ENDMONTH  12   ENDDAY   31
CALENDAR  PROLEPTIC_GREGORIAN
FULL_ENERGY FALSE
FROZEN_SOIL FALSE

DOMAIN       domain.nc
DOMAIN_TYPE  LAT lat
DOMAIN_TYPE  LON lon
DOMAIN_TYPE  MASK mask
DOMAIN_TYPE  AREA area
DOMAIN_TYPE  FRAC frac
DOMAIN_TYPE  YDIM lat
DOMAIN_TYPE  XDIM lon

FORCING1     forcing/isc_3h.
FORCE_TYPE   AIR_TEMP air_temp
FORCE_TYPE   PREC     prec
FORCE_TYPE   PRESSURE pressure
FORCE_TYPE   SWDOWN   swdown
FORCE_TYPE   LWDOWN   lwdown
FORCE_TYPE   VP       vp
FORCE_TYPE   WIND     wind
WIND_H       10.0

PARAMETERS   params.nc
SNOW_BAND    TRUE
BASEFLOW     ARNO
LAI_SRC      FROM_VEGLIB
FCAN_SRC     FROM_VEGLIB
ALB_SRC      FROM_VEGLIB

RESULT_DIR   results
OUTFILE      fluxes
AGGFREQ      NDAYS 1
HISTFREQ     NYEARS 1
OUT_FORMAT   NETCDF4
OUTVAR       OUT_PREC
OUTVAR       OUT_EVAP
OUTVAR       OUT_RUNOFF
OUTVAR       OUT_BASEFLOW
OUTVAR       OUT_SWE
OUTVAR       OUT_SOIL_MOIST
```

(Each option goes on its own line; the two date lines above are compressed
for the page.) Relative paths resolve against the global file's
directory; `${VIC_SAMPLE_DATA}` and other environment variables are
expanded.

## Checking an input set

* `torchvic run global_param.txt --core 5.1.0 --steps 8 --device cpu --no-history`
  loads everything, runs one day and prints every decision. A missing
  variable, a unit TorchVIC does not accept, a grid mismatch or an option
  it refuses stops here, with the name of what is wrong.
* The log line `<ImageParams N cells, 3 layers, 12 veg classes, 5 snow bands, 3 root zones | soil ... | veg ...>`
  lists the variables found; compare it with the list above for the
  options you set.
* `tools/compare_outputs.py A B` compares two result directories variable
  by variable, for before-and-after checks.

## Next step

Describe the run in [The global file](global-file.md), then [run the model](running.md).
