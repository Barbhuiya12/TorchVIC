# Walnut Gulch, end to end

A complete walk through one VIC 5 case: what is in the input files, what
the run decides, what the outputs hold and how to change the run. The
case is UW-Hydro's Walnut Gulch sample (Arizona, 15 cells of a
1/16-degree grid, January 2008, three-hourly forcing, full energy
balance). Nothing here needs a GPU.

## 1. Get the data

```bash
git clone --depth 1 https://github.com/UW-Hydro/VIC_sample_data.git
export TORCHVIC_SAMPLE_DATA=$PWD/VIC_sample_data
cd $TORCHVIC_SAMPLE_DATA/image/WalnutGulch && ls parameters forcings
```

```
parameters/domain.WalnutGulch.0.0625_deg.nc          the grid and mask
parameters/params.WalnutGulch.L2015.0.0625_deg.nc    soil, vegetation, snow bands
parameters/global_param.WalnutGulch.L2015.txt        the run
forcings/livneh_NAmerExt_15Oct2014.WalnutGulch.2008.nc   the year's forcing
```

## 2. Look at the inputs

```python
import xarray as xr
d = xr.open_dataset('parameters/domain.WalnutGulch.0.0625_deg.nc')
print(d)                      # lat (3), lon (5); mask, area, frac, elev
print(int(d.mask.sum()))      # 15 active cells

p = xr.open_dataset('parameters/params.WalnutGulch.L2015.0.0625_deg.nc')
print(p.dims)                 # nlayer 3, veg_class 12, snow_band 5, root_zone 3, month 12
print(p.Cv.sum('veg_class').values)      # cover fractions sum to 1 per cell

f = xr.open_dataset('forcings/livneh_NAmerExt_15Oct2014.WalnutGulch.2008.nc')
print(f)                      # time 2928 (8 a day), temp, prec, air_pressure, shortwave, longwave, vapor_pressure, wind
print(f.vapor_pressure.units, f.wind.units)   # kPa, and a wrongly declared 'mm s-1'
```

The forcing variables are named by the file, not by VIC; the global
file's `FORCE_TYPE` lines pair them with VIC's quantities. The file
declares pressure and vapour pressure in kPa and wind in `mm s-1` (an
error in the sample's metadata). Both matter below.

## 3. Read the global file

`parameters/global_param.WalnutGulch.L2015.txt`, without comments:

```
MODEL_STEPS_PER_DAY 8        SNOW_STEPS_PER_DAY 8        RUNOFF_STEPS_PER_DAY 24
STARTYEAR 2008 STARTMONTH 01 STARTDAY 01    ENDYEAR 2008 ENDMONTH 01 ENDDAY 10
CALENDAR PROLEPTIC_GREGORIAN
FULL_ENERGY TRUE    QUICK_FLUX TRUE    FROZEN_SOIL FALSE    AERO_RESIST_CANSNOW AR_406
DOMAIN ${VIC_SAMPLE_DATA}/image/WalnutGulch/parameters/domain.WalnutGulch.0.0625_deg.nc
DOMAIN_TYPE LAT lat / LON lon / MASK mask / AREA area / FRAC frac / YDIM lat / XDIM lon
FORCING1 ${VIC_SAMPLE_DATA}/image/WalnutGulch/forcings/livneh_NAmerExt_15Oct2014.WalnutGulch.
FORCE_TYPE AIR_TEMP temp / PREC prec / PRESSURE air_pressure / SWDOWN shortwave / LWDOWN longwave / VP vapor_pressure / WIND wind
WIND_H 10.0
PARAMETERS ${VIC_SAMPLE_DATA}/image/WalnutGulch/parameters/parameters.WalnutGulch.L2015.0.0625_deg.nc
LAI_SRC FROM_VEGPARAM    FCAN_SRC FROM_DEFAULT    ALB_SRC FROM_VEGPARAM    NODES 3    SNOW_BAND TRUE
RESULT_DIR ${VIC_SAMPLE_RESULTS}/sample_image
OUTFILE fluxes.WalnutGulch.L2015    COMPRESS FALSE    OUT_FORMAT NETCDF4    AGGFREQ NDAYS 1    HISTFREQ NMONTHS 1
OUTVAR OUT_PREC ... OUT_ALBEDO   (17 variables)
```

(Compressed onto fewer lines for the page; in the file every option has
its own line.) Ten days at eight steps a day, energy balance on with the
quick ground-flux scheme, daily output of 17 variables into one file per
month. [The global parameter file](../guide/global-file.md) explains each
block.

## 4. Run

```bash
torchvic run $TORCHVIC_SAMPLE_DATA/image/WalnutGulch/parameters/global_param.WalnutGulch.L2015.txt \
    --core 5.1.0 --device cpu --out wg_results
```

`--core 5.1.0` runs the physics of VIC 5.1.0, the release that the
[verification](../verification.md) is for. It is the default, so the flag only
makes the choice explicit. `--core 5.0.1` runs the physics of VIC 5.0.1.

Read the log from the top; every line is a decision you may want to know
about:

```
VIC core 5.1.0
global file: MODEL_STEPS_PER_DAY 8, RUNOFF_STEPS_PER_DAY 24
options: FROZEN_SOIL FALSE, BLOWING FALSE, SPATIAL_SNOW FALSE, CARBON FALSE, SHARE_LAYER_MOIST TRUE,
         AERO_RESIST_CANSNOW AR_406, GRND_FLUX_TYPE GF_410, CLOSE_ENERGY FALSE, SNOW_DENSITY DENS_BRAS
  NODES 3: read, no effect here: QUICK_FLUX runs 3 nodes whatever this says (VIC warns and resets it ...)
.../parameters.WalnutGulch.L2015.0.0625_deg.nc does not exist; using .../params.WalnutGulch.L2015.0.0625_deg.nc (the global file names it wrongly)
<ImageDomain 3x5 grid, 15 active cells on mask, lat 31.6562..31.7812>
<ImageParams 15 cells, 3 layers, 12 veg classes, 5 snow bands, 3 root zones | soil Ds,Dsmax,... | veg Cv,LAI,...>
  livneh_NAmerExt_15Oct2014.WalnutGulch.2008.nc declares wind in 'mm s-1'; reading it as m/s -- ...
  livneh_NAmerExt_15Oct2014.WalnutGulch.2008.nc  2928 steps
simulation window from the global file: forcing records 0..80 (2008-01-01T00:00:00 .. 2008-01-10T21:00:00)
forcing window: steps 0..80 of the file
80 steps at 8/day (2008-01-01 00:00:00 .. 2008-01-10 21:00:00)
elevation: the band-weighted mean VIC uses (1399.6 m) rather than the file's elev (1402.4 m)
  press      forcing file
  longwave   forcing file
  shortwave  forcing file, 24-hour mean
  wind       forcing file
vegetation: 12 tiles per cell, mean cover 1.000 (FCAN_SRC FROM_DEFAULT)
WIND_H 10 m from the global file; VIC 5's image driver would give the land surface the parameter file's heights ...
wind height 10 m (global file)
surface energy balance: ON (FULL_ENERGY TRUE in the global file)
<Case 15 cells, 80 steps at 8/day, cpu, energy on, core 5.1.0>
ran 80 steps over 15 cells in 2.5 s

history (the global file's OUTFILE streams):
  wrote wg_results/fluxes.WalnutGulch.L2015.2008-01-01.nc (10 records, AGGFREQ NDAYS 1, 17 variables)
```

* The first line says which VIC release's physics runs.
* The options block states every physics switch in force, including the
  defaults VIC applies for keys the file does not carry.
* The parameter file is named wrongly in the sample's global file;
  TorchVIC finds the one that is there and says so.
* Pressure and vapour pressure were converted from kPa (the file's
  attribute), the wind's wrong unit was recognised and the values taken as
  m/s. Had the file declared nothing, the run would have stopped.
* `WIND_H` from the global file is used for the land surface, a documented
  difference from VIC 5's image driver.

## 5. Read the output

```python
import xarray as xr
ds = xr.open_dataset('wg_results/fluxes.WalnutGulch.L2015.2008-01-01.nc')
print(ds.data_vars)                                 # OUT_PREC, OUT_EVAP, ..., OUT_ALBEDO
print(ds.time.values)                               # 10 daily records
daily = ds[['OUT_PREC', 'OUT_EVAP', 'OUT_RUNOFF', 'OUT_BASEFLOW']].mean(('lat', 'lon')).to_dataframe()
print(daily)                                        # mm per day, domain mean
print(ds.OUT_SOIL_MOIST.isel(time=-1).values)       # [nlayer, lat, lon], mm, end of the last day
```

Water balance of the ten days, per cell: precipitation minus evaporation,
runoff and baseflow equals the change in soil moisture, snow water
equivalent and canopy storage, to round-off; `OUT_DELSOILMOIST`,
`OUT_DELSWE` and `OUT_DELINTERCEPT` are the changes if you add them to
the `OUTVAR` list.

## 6. Change the run

Copy the global file and edit it; the inputs stay where they are.

```bash
cp $TORCHVIC_SAMPLE_DATA/image/WalnutGulch/parameters/global_param.WalnutGulch.L2015.txt wg.txt
```

* **A full year**: `ENDMONTH 12`, `ENDDAY 31` (the forcing covers 2008).
* **Water balance only**: `FULL_ENERGY FALSE` (or `--no-energy`), faster.
* **Hourly output of the energy terms**: `AGGFREQ NSTEPS 1` and
  `OUTVAR OUT_NET_SHORT`, `OUT_NET_LONG`, `OUT_SENSIBLE`, `OUT_LATENT`,
  `OUT_GRND_FLUX`, `OUT_SURF_TEMP`.
* **A restart file at the end**: add `STATENAME wg_results/state`,
  `STATEYEAR 2009`, `STATEMONTH 1`, `STATEDAY 1`, `STATESEC 0` (the state
  date is the end of the last step).
* **VIC 5.0.1's physics**: `--core 5.0.1` (without `--core`, VIC 5.1.0 runs).
* **On a GPU**: `--device cuda`. Fifteen cells are too few to occupy a
  GPU; the GPU pays on large domains ([Performance](../performance.md)).

```bash
torchvic run wg.txt --core 5.1.0 --out wg_year --device cpu
```

## 7. Compare with observations (optional)

A study file next to `wg.txt`:

```yaml
case:
  global_param: wg.txt
  core: 5.1.0
observations:
  runoff:
    file: obs_runoff_daily.csv      # date,value in mm/day
    freq: daily
    cells: mean
report:
  metrics: [kge, nse, pbias]
```

```bash
torchvic run study.yaml --out wg_year
```

prints the metrics after the run. [Study files](../guide/study.md).
