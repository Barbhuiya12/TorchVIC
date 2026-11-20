# Your own domain, from scratch

This tutorial builds a TorchVIC case for a region where you have no VIC 5
files yet. It is the path the Indian-subcontinent (ISC) case took,
generalised. Each step names the tool in `tools/` that did it there and what
you would replace it with.

The end state is four things: `domain.nc`, `params.nc`, a forcing on the
same grid, and a global parameter file. [Prepare your inputs](../guide/inputs.md)
describes each file; this page is the order of work.

## 1. Choose the grid

A regular latitude-longitude grid with cell centres at
`lat0 + i*res`, `lon0 + j*res`. VIC's parameter sets are usually at 1/16
degree (about 6 km) or 1/4 degree. Write down `lat`, `lon`, `res` once and
build every file on them; TorchVIC refuses files whose coordinates differ.

## 2. The land mask and the domain file

A `mask` of 1 where the model runs. Take it from your parameter source (a
cell with soil parameters is a land cell), or from a land-sea mask on the
grid. Cell `area` in m2 from the spherical formula or from
`compute_cell_area.c`'s (`torchvic.io.classic.cell_area_vic`),
`frac` 1 for land, `elev` from a DEM averaged to the cell. Write them with
xarray ([Prepare your inputs](../guide/inputs.md) has the snippet).

## 3. Parameters

**You have VIC classic files** (soil, vegetation parameter, vegetation
library, snow bands) for the region, for example from an earlier VIC 4
calibration:

```bash
torchvic convert-classic soil.txt vegparam.txt veglib.txt --snowband snowbands.txt \
    --snow-bands 5 --root-zones 3 --veglib-fcan --fcan-src FROM_VEGLIB \
    --grid domain.nc --domain domain.nc --params params.nc
```

writes both files on the grid of `domain.nc` (`--grid` takes any file with
1-D `lat`/`lon`). Cells of the soil file outside the grid stop the
conversion unless `--drop-outside` lists and omits them. With a classic
global file, `--global global.classic.txt` takes the file names and the
layout options (layers, root zones, `LAI_SRC`, ...) from it.

**You have a global parameter set** such as VICGlobal: cut its `params.nc`
to the grid with xarray (`sel(lat=slice(...), lon=slice(...))`) and keep
the variable names. `tools/build_isc_params.py` is the ISC version: it
merged a soil parameter file with VICGlobal's vegetation and snow bands
and filled coastal cells VICGlobal treats as sea from their nearest land
cell.

**Snow bands.** With `SNOW_BAND TRUE` every band with area runs as its own
column, so a cell with five bands costs five columns. Either write the bands
(`AreaFract`, `elevation` and `Pfactor` per band), or write one band per cell
(`AreaFract` 1, `elevation` the cell's), or set `SNOW_BAND FALSE`. As VIC does
with `SNOW_BAND TRUE`, TorchVIC replaces the cell's `elev` by the area-weighted
mean of the band elevations.

## 4. Forcing

**Sub-daily reanalysis or product at the model step** (ERA5-Land hourly,
a 3-hourly product): regrid to the domain grid (bilinear for state
variables, conservative for precipitation), write one file per year
`forcing.YYYY.nc` with `[time, lat, lon]` variables and a CF time axis,
and declare the names and units. Pressure in Pa or kPa, vapour pressure in
Pa or kPa, temperature in degC (not Kelvin), precipitation in mm per step
or as a rate with its unit stated, radiation in W m-2, wind in m/s.

**Daily data** (ERA5 daily reductions, station-based gridded products
such as IMD): daily `tasmax`, `tasmin`, `prcp`, and whichever of `wind`,
`vp`, `sp`, `swdown`, `lwdown` you have, one record per day stamped at
00:00. TorchVIC's MTCLIM chain produces the sub-daily forcing on the
device. For ISC, `tools/edh_era5_daily.py` reduced hourly ERA5 from the
Earth Data Hub to these daily fields and `tools/build_isc_forcing.py`
assembled them into a Zarr store; with no `FORCE_TYPE` lines a store
carrying those eight ERA5 names is recognised and run through MTCLIM.

To give VIC 5 itself the same sub-daily forcing (for a side-by-side
comparison), `benchmarks/make_isc_forcing.py subdaily` writes the
three-hourly NetCDF files MTCLIM produces.

## 5. The global parameter file

Start from the template in [Prepare your inputs](../guide/inputs.md) or
from `benchmarks/isc/isc_vic5_3h.txt` (ISC, water balance, three-hourly)
and `benchmarks/isc/isc_vic5_3h_eb.txt` (energy balance). Set the dates,
the step, the three paths, the `FORCE_TYPE` lines and the output streams.

## 6. Smoke test, then run

```bash
torchvic run global.txt --core 5.1.0 --device cpu --steps 8 --no-history      # one day: every check and decision printed
torchvic run global.txt --core 5.1.0 --device cuda --steps 240 --out month   # one month on the GPU
torchvic run global.txt --core 5.1.0 --device cuda --out results/            # the run
```

`--core 5.1.0` runs the physics of VIC 5.1.0, the release that the
[verification](../verification.md) is for. It is the default, so the flag only
makes the choice explicit. `--core 5.0.1` runs the physics of VIC 5.0.1.

The first GPU run for a new cell count compiles the kernels and is slower;
every later run reuses the cache. A spin-up ends in a state file
(`STATENAME`, `STATE*`) that the production run starts from (`INIT_STATE`). On a
GPU, reading a VIC state file has a known problem in 0.4.8, with workarounds
([Known limitations](../about/limitations.md)).

## 7. Flow directions and gauges (optional)

Discharge at gauges needs a D8 flow-direction grid on the domain and the
gauge locations. Sources: HydroSHEDS 15 arc-second or MERIT-Hydro
directions upscaled to the grid (`tools/fdir_sources.py`, outlet-cell
tracing or `pyflwdir`), or a vector river network such as GRIT
(`tools/build_isc_fdir_grit.py`). `tools/fdir_multi.py` scores several
candidate maps against gauge catchment polygons and combines the best per
gauge. Then route the run's runoff and baseflow with
[Routing to gauges](../guide/routing.md); `tools/route_isc.py` is the ISC
driver, reading the history files and writing gauge discharge as NetCDF
and CSV.

## 8. Compare with observations and run ensembles

Observed discharge as a CSV, a study file with the metrics you want
([Study files](../guide/study.md)), and `case.ensemble(n)` to evaluate
many parameter sets in one GPU run ([Python API](../guide/python-api.md)).

## Next step

Run VIC with other models in one time loop: [Coupled runs](../coupled/overview.md).
