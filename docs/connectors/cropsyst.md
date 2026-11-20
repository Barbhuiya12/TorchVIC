# CropSyst

The `cropsyst` backend of the `crop` connector runs the VIC-CropSyst crop model (Malek et al., 2017; see [References](../about/references.md)) inside the coupled system.

## What it does

The engine simulates crop growth per crop tile (sowing, canopy and root growth, phenology, yield), water uptake from VIC's layer moisture, and optionally nitrogen and irrigation. It is the `cropsyst` backend of the `crop` connector (source `torchvic/coupling/connectors/crop.py`; engine in `torchvic/crop/cropsyst/`). As a backend of the `crop` connector it does not change VIC's results. Run as the `vicwur_cropsyst` connector inside a VIC-WUR group (`torchvic/coupling/connectors/vicwur_cropsyst.py`), the same engine replaces VIC's transpiration on crop tiles with CropSyst's crop uptake. The WOFOST backend is described in [Crops](crop.md).

## When to use it

* Use it to run the VIC-CropSyst crop model, including its nitrogen, irrigation, orchard and forage-clipping parts, on VIC's soil water.
* Use the `vicwur_cropsyst` connector when CropSyst's uptake should replace VIC's transpiration on crop tiles ([VIC-WUR](vicwur.md)).
* Do not use it when a WOFOST crop with one parameter set is enough: use the default backend of [Crops](crop.md).
* Do not use it for crop parameters of a new region without supplying them: the repository has no CropSyst parameter files.

## Inputs

Data inputs of the engine:

| input | file or variable | unit | meaning |
|---|---|---|---|
| crop, management and rotation files | CropSyst database of the VIC-CropSyst package (`db_dir`) | none (parameter files) | crop parameters, management operations (fertiliser, tillage) and the sowing calendar; the rotation files give the sowing calendar (the WOFOST backend's `start_doy` does not apply) |
| irrigation parameter file | file of the VIC-CropSyst package | none (parameter file) | irrigation technology table: allowable depletion and losses of the applied water |
| weather arrays | `weather` option | tmax, tmin: C; srad: W m-2; rhmin: % | daily weather (tmax, tmin, srad, rhmin; optional et_ref); if absent, built from the VIC case's forcing. The reference evapotranspiration comes from the arrays or from `vic.pet` |

Connection inputs of the `crop` connector:

| input | file or variable | unit | meaning |
|---|---|---|---|
| `soil_moisture` | `vic.soil_moisture`, link with `lag: 1` | mm | VIC layer moisture (liquid and ice) at the start of the day |
| `soil_ice` | optional | mm | VIC layer ice |
| `soil_temperature` | VIC layer temperature (nitrogen only) | C | layer temperature at the start of the day |
| `pet` | `vic.pet` | mm | VIC potential evaporation of the day; the reference evapotranspiration unless the weather arrays supply it |

## Config block

```yaml
modules:
  vic:
    connector: vic
    global_param: path/to/global_param.txt
    core: 5.1.0
  crop:
    connector: crop
    global_param: path/to/global_param.txt
    core: 5.1.0
    backend: cropsyst
    cropsyst:             # the engine's options: db_dir, tiles, tile_cell, soil,
      db_dir: path/to/cropsyst_db   # nitrogen, clip_mode, ...
connections:
  - {from: vic.soil_moisture, to: crop.soil_moisture, lag: 1}
  - {from: vic.pet,           to: crop.pet}
```

`configs/` has no CropSyst example. See [Configuration](../CONFIGURATION.md). To run CropSyst inside VIC-WUR's daily step, use the `vicwur_cropsyst` connector as a member of a VIC-WUR group ([VIC-WUR](vicwur.md)).

## Outputs

Outputs of the `crop` connector with the `cropsyst` backend, on the grid `crop_tiles`:

| output | unit | meaning |
|---|---|---|
| `lai` | 1 | leaf area index (0 without a crop) |
| `gai` | 1 | green area index of the day record |
| `green_cover` | 1 | green canopy solar interception |
| `root_depth` | m | rooting depth |
| `harvest`, `sown` | 1 | 1 on a harvest or sowing day |
| `stage` | 1 | normal growth stage code |
| `biomass` | kg m-2 | vital canopy biomass |
| `yield` | kg ha-1 | yield on the harvest day (fruit dry biomass for orchard crops) |
| `crop_active` | 1 | 1 while a crop stands on the tile |
| `water_stress` | 1 | actual / limited potential transpiration |
| `act_transp` | mm | uptake diagnosed from the VIC layer moisture (per day) |
| `et_max` | mm | crop ET max (per day) |
| `leaf_water_pot` | J kg-1 | leaf water potential |
| `uptake` | mm | uptake per VIC layer (per day) |
| `fruit_biomass`, `chill_remaining` | kg m-2, 1 | orchard fruit dry biomass; chill units still needed |
| `clip_margin`, `orchard_interception`, `canopy_width`, `canopy_breadth`, `canopy_height` | 1, 1, m, m, m | clipping diagnostic and orchard canopy architecture |

With nitrogen on, the connector also exports `n_stress` (1), `soil_no3_n` and `soil_nh4_n` (kg m-2, per layer) and `n_overflow` (1).

## Technical details

### Features

* **Crop growth** per crop tile: sowing, canopy and root growth, phenology and yield; with `vicwur_cropsyst`, per snow band.
* **Water uptake** from VIC's layer moisture. With the `crop` connector it is an output and VIC keeps its transpiration. With `vicwur_cropsyst` it is written into VIC's soil layers as transpiration (hook `transp`) and reaches VIC on the next day, as in VIC-CropSyst.
* **Soil layers.** The engine takes its layer count from the `soil` option and VIC from `NLAYER`; VIC-CropSyst uses 17 layers. Its wilting-point floor below the top layer is the hook `wpwp_floor` of `vicwur_cropsyst`.
* **Crop properties in VIC** (`vicwur_cropsyst`, option `hooks`): the daily LAI replaces the vegetation library value on crop tiles (hook `lai`); canopy evaporation of crop tiles can be handed to CropSyst (hook `canopy_evap`).
* **Nitrogen** (optional): crop nitrogen and the soil nitrate and ammonium of each layer, with residue mineralisation, nitrification, denitrification, crop uptake, and the fertiliser and tillage operations of the management files.
* **Irrigation**: demand when the soil water depletion exceeds the allowable depletion (by default from the irrigation technology table) or the water stress index exceeds its tolerance; losses of the applied water from the table or a mechanistic evaporation and runoff model; a scheme for VIC-WUR's irrigation component (`scheme: cropsyst` of `vicwur_irrigation`).
* **Orchards and perennial fruit crops** with canopy architecture (apple), and **forage clipping** (alfalfa hay).

The coupling is daily: with sub-daily or energy-balance VIC runs the crop steps once per day.

### Modes

Two defects of upstream VIC-CropSyst are corrected by default. Each has an option, `fixed` (default) or `exact`; `exact` reproduces the upstream behaviour, for comparison with the compiled VIC-CropSyst.

| upstream behaviour | default (`fixed`) | `exact` |
|---|---|---|
| the mechanistic irrigation runoff divides mm water contents by 100, as if the top layer were 100 mm thick, giving NaN where the water content exceeds porosity | divides by the top-layer thickness, which must be given (`ro_mode='fixed'`) | keeps the NaN (`ro_mode='exact'`) |
| clipping removes a canopy portion that it may not remove and loses it (neither yield, residue nor standing) | the portion stays on the plant (`clip_mode='fixed'`) | loses it (`clip_mode='exact'`) |

### Setting up

Select the backend in the `crop` connector, or use the `vicwur_cropsyst` connector as a member of a VIC-WUR group, where CropSyst runs inside VIC-WUR's daily step ([VIC-WUR](vicwur.md)).

The `crop` backend reads VIC's layer moisture of the start of the day (link `vic.soil_moisture` with `lag: 1`) and VIC's potential evaporation (link `vic.pet`) unless the weather arrays supply the reference evapotranspiration. Inputs are the crop, management and rotation files of the CropSyst database of the VIC-CropSyst package (`db_dir`) and its irrigation parameter file; the rotation files give the sowing calendar (the WOFOST backend's `start_doy` does not apply). Crop parameters for a new region have to be supplied by the user.

## Limitations

* The repository has no CropSyst parameter files: the crop, management and rotation files come from the VIC-CropSyst package or, for a new region, from the user.
* `vicwur_cropsyst` builds its daily weather from a VIC case with one model step per day; for a sub-daily case give the `weather` arrays.
* License and credits of the CropSyst components: see [License and credits](../downloads/code.md#license-and-credits) and `THIRD_PARTY_NOTICES.md`.
* General limits: [Limits of the coupled system](../coupled/overview.md#limits).

## Verification status

Checked against the VIC-CropSyst package in the test suite; not part of the VIC verification on the Verification page. The reference runs VIC 4.1.2.e in water-balance mode with 17 soil layers, daily forcing and one snow band for crop tiles, so a coupled run cannot agree with pristine VIC 5 and with VIC-CropSyst at once. The comparison has three levels:

* **Function level.** Each CropSyst component is compared with the compiled reference functions on the same inputs (`tests/test_cropsyst_uptake.py`, `test_cropsyst_phenology.py`, `test_cropsyst_growth.py`, `test_cropsyst_nitrogen.py`, `test_cropsyst_irrigation.py`, `test_cropsyst_orchard.py`, `test_cropsyst_orchard_arch.py`, `test_cropsyst_clip.py`).
* **Season replays.** Driven by the reference run's own water stress, leaf water potential and, with nitrogen, soil state, the engine's crop output is compared with the reference's on the annual cases of the Columbia River Basin (CRB) set (`tests/test_cropsyst_i8.py`, `tests/test_cropsyst_n_engine.py`).
* **Whole runs.** VIC 5 with `vicwur_cropsyst` was started from the reference run's own inputs on the annual CRB cases and two orchards, with nitrogen off, and compared with the reference run (`tests/cropsyst_i12_l3.py`). The comparison is statistical, not a match, because the reference is a different VIC.


## Next step

[VIC-WUR](vicwur.md).
