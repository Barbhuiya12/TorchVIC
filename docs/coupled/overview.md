# Coupled runs: overview

**What you will learn:** what a coupled run is, which connectors exist and which
page to read for each.

A coupled run runs VIC and other water models in one time loop on the GPU.
A YAML file names the modules (connectors) and the connections between them; a
runtime orders them, exchanges water between them and a ledger checks that every
transfer closes. VIC alone is run with `torchvic run global_param.txt --core 5.1.0`; the
coupled system is a separate entry point, `torchvic run --config`, that uses the
same VIC code.

```bash
torchvic validate --config coupled.yaml [--deep]    # check without running
torchvic run --config coupled.yaml [--device cuda] [--out DIR]
```

```python
from torchvic.coupling import TorchVIC
app = TorchVIC.from_config('configs/vic_routing.yaml')
app.initialize().run()
```

`validate` checks a configuration without running it; `--deep` also reads the
data. A run writes VIC's history and the ledger to `output.dir`. Example
configurations are in `configs/`. The file is described in
[Configuration](../CONFIGURATION.md), the connector interface in
[Connector API](../CONNECTOR_API.md), the dependency graph in
[Coupling graph](../COUPLING_GRAPH.md) and the budget in
[Water accounting](../WATER_ACCOUNTING.md). The design is in
[Architecture](../ARCHITECTURE.md).

## Your first coupled run

1. Install TorchVIC and run the [Quick start](../quickstart.md) first.
2. Write a configuration file with the VIC module only; the minimal file is in
   [Configuration file](../CONFIGURATION.md#config-block).
3. Check it: `torchvic validate --config coupled.yaml`.
4. Run it and compare the history with a plain `torchvic run`: the two are
   equal.
5. Add one connector at a time: take its `modules:` entry and `connections:`
   from its page under [Connectors](../connectors/index.md).

## Connectors

| connector | does | page |
|---|---|---|
| `vic` | the land surface model (also streamed for large domains) | [Connector API](../CONNECTOR_API.md) |
| `routing` | river routing (Lohmann/RVIC convolution) | [River routing](../connectors/river.md) |
| `reservoir` | dams and operating rules | [Reservoirs](../connectors/reservoir.md) |
| `quality` | water temperature and constituents | [Water quality](../connectors/water_quality.md) |
| `groundwater` | aquifer flow (MODFLOW 6 GWF equations) and river-aquifer exchange | [Groundwater](../connectors/groundwater.md) |
| `flood` | GPU flood hydraulics, one or two way; LISFLOOD-FP as a reference backend | [Flood](../connectors/flood.md) |
| `split` | sends the runoff of a box of VIC cells to the flood module and the rest to the router, never both | [Coupling graph](../COUPLING_GRAPH.md) |
| `irrigation` | water use and irrigation | [Irrigation](../connectors/irrigation.md) |
| `crop` | WOFOST 7.2 crop growth; backend `cropsyst` for VIC-CropSyst | [Crops](../connectors/crop.md), [CropSyst](../connectors/cropsyst.md) |
| `vicwur_routing`, `vicwur_dams`, `vicwur_wateruse_local`, `vicwur_wateruse_remote`, `vicwur_nonrenew`, `vicwur_efr`, `vicwur_irrigation`, `vicwur_co2`, `vicwur_landuse`, `vicwur_crop`, `vicwur_cropsyst` | VIC-WUR package: cascade router, dams, water use, non-renewable abstraction, environmental flow, irrigation, CO2, land use, crops | [VIC-WUR](../connectors/vicwur.md) |
| ensembles and EnKF/ETKF/LETKF | Python engines, not registry connectors | [Ensembles and DA](../connectors/ensemble_da.md) |

## Inputs of each connector

| input | format |
|---|---|
| VIC case | VIC 5 global parameter file, image-driver domain, parameter and forcing NetCDF (or Zarr forcing) |
| coupled run | YAML, `${ENV}` substitution; path keys are resolved against the file's directory |
| routing | flow-direction NetCDF (`fdir`), domain NetCDF, gauge table |
| reservoirs | dam table (GDW/GRanD columns) |
| irrigation | irrigated fraction per cell, season window or crop calendar |
| crop | WOFOST parameter JSON (`configs/crop_params_sbarley.json`), optional `height_table` |
| groundwater | per-cell top, bottom, conductivity, specific yield, initial head |
| flood | basin `.npz` plus a VIC-pairs CSV (`tools/flood/make_basin.py`) |

## GPU requirements

A CUDA GPU with good float64 throughput, a driver for the torch 2.6 CUDA build and
Triton (shipped with the CUDA wheel) for the flood engine's fast path; plain
torch is the fallback. A coupled run uses one GPU. The first run on a new cell
count compiles the kernels and is slower. The LISFLOOD-FP backend needs its own CUDA
build (`LISFLOOD_BIN`). Device memory grows with the domain and the connectors
(see [Install](../install.md) for the memory of VIC alone).

## Limits

* The coupled system is checked by the test suite for plumbing, water
  conservation and restart; each connector page lists its tests. It is not part
  of the evaluation of VIC reported on the [Verification](../verification.md)
  and [Performance](../performance.md) pages.
* Results are not calibrated. Where a connector page says that input data must
  be supplied, a run without them uses default values that are not regional data.
* `torchvic run` does not run the router; routing is reached through a coupled
  configuration, `tools/route_isc.py` or the Python functions of
  [Routing to gauges](../guide/routing.md).
* The VIC connector's `soil_add` refuses `FROZEN_SOIL` and snow bands, unless
  the VIC-WUR options `soil_add_frozen` and `soil_add_bands` are set; the
  streamed path does not build `LAKES`, `CARBON`, `FROM_VEGHIST` or
  `INIT_STATE`.
* A VIC connector whose global file names a forcing larger than 2 GiB is loaded
  by the streamed path, which reads the forcing one exchange window at a time.
* Flood results are not identical between CPU and GPU: round-off is amplified
  at wet/dry fronts by the wet/dry thresholds ([Flood](../connectors/flood.md)).

## Not included

A lakes connector, carbon and land-cover coupling, coupling to Earth-system
models, and ensembles inside the runtime.

## Next step

[Configuration file](../CONFIGURATION.md), then [Connectors](../connectors/index.md).
