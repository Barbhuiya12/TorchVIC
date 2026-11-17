# Crop connector

The crop connector (`CropConnector`, registry name `crop`,
`torchvic/coupling/connectors/crop.py`) runs the WOFOST 7.2 crop model on the
cells of a VIC case, once a day. The model is `torchvic/crop/`, a PyTorch port
of WOFOST 7.2 as implemented in PCSE (Python Crop Simulation Environment)
6.0.13; PCSE is the reference in the tests. The crop's daily water stress comes
from VIC, and the connector returns leaf area index, rooting depth, development
stage and yield. The `cropsyst` backend of the same connector is described in
[CropSyst](cropsyst.md). Example configuration: `configs/vic_crop.yaml`.

## Process responsibility

VIC is responsible for all water and energy. The crop model is responsible for
biomass, phenology, leaf area and roots, and computes no water balance of its
own: WOFOST runs in `external` mode, where its soil-water module is replaced by
a transpiration reduction factor (RFTRA) taken from VIC. There is no second soil
bucket, so nothing is counted twice, and the water ledger is unchanged (ET stays
VIC's).

| Quantity | Responsible model | Direction |
|---|---|---|
| Soil water, evaporation, transpiration | VIC | VIC to crop: RFTRA, potential evaporation |
| LAI, root depth, development stage, yield | crop | crop outputs |

Each day VIC is stepped first and the crop second (the example configuration
runs VIC with `exchange: day`). VIC uses the crop's LAI only when it is fed back
([Links to other modules](#links-to-other-modules)).

## Sowing

One crop starts on the run day `start_day` (default 0); before that day
`crop_active` is 0. `start_doy` sows a fresh crop every year on that day of the
year: each sowing replaces the previous crop with a new one, and the first
sowing is the first such day on or after `start_day`. `crop_active` is 1 while a
crop is growing.

## RFTRA: the water-stress factor

RFTRA is WOFOST's TRA / TRAMX, actual over potential crop transpiration, which
without oxygen stress is WOFOST's water-stress factor RFWS. With
`vic.soil_moisture` linked (`rftra_from: soil`, or `auto`) the connector forms

```
RFWS = (W - Wwp) / (Wcr - Wwp)    over the root zone [0, RD]
```

from VIC's layer moisture W (liquid plus ice, at the end of the day) and VIC's
critical and wilting points Wcr and Wwp (`Wcr_FRACT` and `Wpwp_FRACT` times the
layer's maximum moisture). The terms are summed over the layers, each layer
weighted by the part of its thickness above the crop's rooting depth RD (the
depth at the start of the day, the initial depth for a new crop), and RFWS is
limited to [0, 1] (`CropConnector.rftra_soil`). It is 1 where the root zone has
no range (Wcr = Wwp).

Other options:

* `rftra_from: transp_pet` uses `vic.transp / vic.pet` (limited to [0, 1], and 1
  where `vic.pet` is zero). Kept for comparison only; not a stress factor,
  because VIC's OUT_PET is the potential evaporation of the whole cell (an
  unstressed canopy plus a bare-soil part with no surface resistance), while
  OUT_TRANSP_VEG is only the dry canopy's transpiration at the library LAI. With
  wet soil the ratio is therefore low, and it would stress a crop whose soil
  water is not limiting.
* a direct `rftra` input, which replaces both and takes precedence over
  `rftra_from`.

`rftra_from: auto`, the default, uses a direct `rftra` input if one is linked,
otherwise the soil-water factor if `vic.soil_moisture` is linked, otherwise
`transp_pet`. `configs/vic_crop.yaml` links `vic.transp` and `vic.pet` only, so
it runs with `transp_pet`; also link `vic.soil_moisture` to
`crop.soil_moisture` to use the soil-water factor.

## Inputs and outputs

The crop parameters are one WOFOST 7.2 parameter set, given as `params_file` (a
JSON or YAML file of parameter names and values, relative to the configuration
file) or as `params`. `configs/crop_params_sbarley.json` (spring barley) is the
example. A variety of the WOFOST crop parameter repository is read with
`torchvic.crop.load_wofost_crop_yaml(path, variety)` and passed as `params`.

Weather comes from the VIC case's forcing (daily minimum and maximum air
temperature, mean shortwave radiation, precipitation used as rain), with
E0 = ES0 = ET0 = VIC's `pet` (WOFOST's open-water, bare-soil and reference-crop
evaporation). Or give `weather` arrays: daily `tmin` and `tmax` (degrees C),
`irrad` (J m-2 d-1), `rain` (cm d-1), `doy` and `et0` (cm d-1; `e0` and `es0`
default to it), each `[day, cell]` or `[day, 1]`; without `et0` VIC's `pet` is
used. The latitude is taken from the VIC case or given as `lat`.

Inputs:

| Input | Unit | Meaning |
|---|---|---|
| `soil_moisture` | mm | VIC layer moisture, `[cell, layer]`; gives RFTRA from the soil water |
| `transp` | mm | VIC transpiration over the day; for `transp_pet` |
| `pet` | mm | VIC potential evaporation over the day; the reference evaporation, and for `transp_pet` |
| `rftra` | 1 | RFTRA supplied directly |

Outputs, all states on the VIC cell grid:

| Output | Unit | Meaning |
|---|---|---|
| `lai` | 1 | leaf, stem and pod area index |
| `root_depth` | m | rooting depth (WOFOST RD) |
| `dvs` | 1 | development stage |
| `yield` | kg ha-1 | weight of storage organs (TWSO) |
| `crop_active` | 1 | 1 while the crop stands (the irrigation season) |
| `rftra` | 1 | transpiration reduction used on the day |
| `crop_height` | m | `height_table` (an AFGEN table of DVS, as a flat list of DVS and height pairs) evaluated at DVS; only with `height_table`, because WOFOST 7.2 has no height |

## Links to other modules

VIC reads a per-step LAI only from a vegetation history (`LAI_SRC FROM_VEGHIST`
with `FORCING2`, see [Options](../OPTIONS.md)). The crop's LAI can reach VIC in
one of two ways.

* **Same run.** The VIC connector has an optional `lai` input (`[cell, tile]`,
  the LAI of the next interval; only for cases with `LAI_SRC FROM_VEGHIST`, see
  [Connector API](../CONNECTOR_API.md#injection-points)). The crop connector
  exports one LAI per cell and this input takes one value per cell and tile, so
  the connector's `lai` output cannot feed it directly.
* **Two passes.** `write_veghist(case, lai_days, crop_tiles, prefix)` in
  `torchvic/coupling/connectors/crop.py` writes a daily LAI series, `[day, cell]`
  (for example the connector's `lai` series), on the given crop tiles as the
  vegetation-history files `<prefix><year>.nc`; the other tiles keep the
  parameter file's monthly LAI. A second VIC run with `LAI_SRC FROM_VEGHIST` and
  `FORCING2 <prefix>` then reads them.

Irrigation demand can follow the crop calendar: `crop.crop_active` to
`irrigation.season` and `crop.root_depth` to `irrigation.root_depth`
([Irrigation](irrigation.md)); these connections are commented out in
`configs/vic_crop.yaml`.

## State

Per cell, the state is that of the WOFOST engine: development stage and DVS,
temperature sums, LAI with the leaf cohorts, the biomass pools (leaf, stem,
root, storage organ), root depth, the days of emergence, anthesis and maturity,
and whether the crop has finished; the connector adds the day counter. A
checkpoint stores all of it. `compute` leaves the accepted state unchanged and
`commit` installs the new state. Crop carbon is not part of the water ledger.

## Limits

* Crop calendars and cultivar parameters are inputs and are used as given. No
  regional values are provided; they must be supplied by the user. For India,
  possible sources are the SAGE (Sacks et al., 2010) and MIRCA crop calendars,
  and the WOFOST crop parameter repository for parameters; check each licence.
* Both ways of feeding LAI to VIC are tested with a prescribed LAI, not with the
  LAI of the crop connector: the `lai` input with a stand-in source that exports
  LAI per tile, and `write_veghist` with a prescribed LAI series.
* `torchvic/crop/` is a port of PCSE 6.0.13, which is distributed under the
  EUPL-1.2; the port is distributed with TorchVIC under GPL-3.0-or-later, which
  the EUPL-1.2 appendix lists as compatible. The notice is
  `torchvic/crop/NOTICE_PCSE.txt`; see also
  [License and credits](../downloads/code.md#license-and-credits) and
  `THIRD_PARTY_NOTICES.md`. Cite de Wit et al. (2019) and van Diepen et al.
  (1989) ([References](../about/references.md)).
* The limits of the coupled system as a whole are listed under
  [Limits of the coupled system](../coupled/overview.md#limits).

## Tests

`tests/test_conn_agri.py` (it needs VIC's Stehekin sample data, see
[Testing](../development/testing.md)) checks that the coupled run equals the
standalone engine given the same RFTRA series, bit for bit, on the CPU and on
CUDA; that a run restarted from a checkpoint equals the continuous run; the
`start_day` and `start_doy` sowing rules; that the crop connector leaves VIC's
results unchanged and the module and coupled ledgers close; and that RFTRA from
the soil water is WOFOST's water-stress factor over the root zone.
`tests/test_coupling_vic.py` tests the `lai` input of the VIC connector with a
stand-in LAI source. `tests/test_crop_pcse.py` compares the port with stored
PCSE 6.0.13 reference output (development stage, LAI, biomass pools and rooting
depth on every day, to a relative tolerance of 1e-9) and replays PCSE's
water-limited crop from its stress series; PCSE itself is not needed, and the
GPU comparison needs CUDA.
