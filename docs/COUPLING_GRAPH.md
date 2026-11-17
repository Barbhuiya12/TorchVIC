# Coupling graph

Which connector talks to which, what is exchanged, and in what order. Every
edge is a `connections:` entry of a configuration (`from: module.output`,
`to: module.input`, optional `lag:` and `mapping:`); the runtime checks it at
construction and initialisation (see [Connector API](CONNECTOR_API.md)).
Names below are the registry names (`torchvic/coupling/registry.py`).

## Modules

| registry name | class | exchange interval | state it holds | needs a GPU |
|---|---|---|---|---|
| `vic` | `VicConnector` | `window_days` (default 31), or `exchange: day` / `step` | soil, snow, canopy | no (CPU path exists), fast on GPU |
| `routing` | `RoutingConnector` | one day | channel ring buffers | no (`StepRouter` on CPU, `FastStepRouter` on CUDA) |
| `reservoir`, `quality` | sweep plugins of `routing` | with routing | dam storage; constituents and heat | as routing |
| `irrigation` | `WaterUseConnector` | one day | leftover-demand store | no |
| `crop` | `CropConnector` (WOFOST 7.2) | one day | crop cohorts | no |
| `groundwater` | `GroundwaterConnector` | one day | aquifer head | no (fused CUDA graphs on GPU) |
| `flood` | `FloodConnector` | `interval_s` (86400) | water on the hydraulic grid | engine backend: CPU and GPU (Triton step); `lisflood` backend: external process |
| `split` | `SplitConnector` | with VIC | none; sends each VIC cell's runoff whole to the flood box or to the router | no |
| `two_store` | `TwoStoreConnector` | any | test stores | no |

Ensembles and data assimilation (`torchvic.ensemble`) are
engines used from Python, not registry connectors (see
[Coupled system overview](coupled/overview.md)).

## Edges that exist

```
                 vic.total_runoff / total_runoff_daily
   VIC  ---------------------------------------------->  routing  ----> discharge (gauges)
    ^  \                                                  ^  |   ^
    |   \  vic.total_runoff (mm) -> flood.runoff           |  |   | injection (lag 1)
    |    \_____________________________> flood  <----------+  |   |
    |                                      |   withdrawn ->    |   |
    |     infiltration -> vic.soil_add     |   flood.inflow    |   |
    +<------------(lag 1, cap)-------------+-- returned -------+---+
    |
    |  vic.soil_moisture, vic.soil_add_unmet
    +------------------------> irrigation --soil_add (lag 1)--> VIC layer 0
    |  vic.baseflow -> groundwater.recharge        ^   |
    +------------------------> groundwater         |   +-- pumping_request -> groundwater
    |  vic.transp, vic.pet -> crop.transp/pet      |
    +------------------------> crop --crop_active, root_depth--> irrigation.season / root_depth
```

| from | to | kind | lag | see |
|---|---|---|---|---|
| `vic.total_runoff_daily` | `routing.inflow` | flux, mm per day | 0 | [river connector](connectors/river.md); equals standalone routing bit for bit |
| `routing.withdrawn` | `flood.inflow` | flux, m3 | 0 | [flood connector](connectors/flood.md#exchange-frequency-and-lags) |
| `flood.returned` | `routing.injection` | flux, m3 | 1 | same |
| `flood.river_request` | `routing.withdrawal_request` | state | 1 | same |
| `vic.total_runoff` | `flood.runoff` | flux, mm to m3 per hydraulic cell (mapping `vic_to_flood`) | 0 | `configs/vic_flood.yaml` |
| `flood.infiltration` | `vic.soil_add` | flux, optional | 1 | [flood connector](connectors/flood.md#floodplain-to-vic-soil-optional-off-by-default) |
| `vic.soil_add_unmet` | `flood.infiltration_unmet`, `irrigation.soil_unmet` | flux | 0 | same, `configs/vic_irrigation.yaml` |
| `vic.soil_moisture` | `irrigation.soil_moisture` | state | 0 | `configs/vic_irrigation.yaml` |
| `irrigation.soil_add` | `vic.soil_add` | flux, mm | 1 | `configs/vic_irrigation.yaml` |
| `crop.crop_active`, `crop.root_depth` | `irrigation.season`, `irrigation.root_depth` | state | 0 | `configs/vic_crop.yaml` (commented, enabled by the user) |
| `vic.transp`, `vic.pet` | `crop.transp`, `crop.pet` | state | 0 | `configs/vic_crop.yaml` |
| `vic.baseflow` | `groundwater.recharge` | flux | 0 | `configs/vic_groundwater.yaml` |
| `irrigation.gw_request` | `groundwater.pumping_request` | state | 0 | `configs/vic_groundwater_irrigation.yaml` |
| `groundwater.pumping` | `irrigation.gw_pumped` | flux | 1 | same |
| `groundwater.discharge` | `routing.inflow` | flux | 0 | `configs/vic_groundwater.yaml` (comment) |

The river and reservoir sources of the irrigation connector are stores held
inside the connector; only the groundwater share is wired to another connector
(`configs/vic_groundwater_irrigation.yaml`). A configuration that wires
irrigation withdrawals to the routing or reservoir connectors is not provided
([Irrigation](connectors/irrigation.md)).

## Rules the runtime enforces

* a flux feeds a flux and a state feeds a state; units convert or the
  connection is refused; grids must match or be joined by a conservative
  mapping (each source element's fractions sum to 1);
* a flux output feeds at most one connection (no water delivered twice);
* two modules that write one store are refused (`owned_state` in the
  [Connector API](CONNECTOR_API.md));
* a lag-0 cycle needs `iterate`; otherwise one edge of the cycle must carry
  `lag: 1` (the flood two-way loop uses the lagged return);
* a declared `dependencies` entry is also an ordering edge (the dependency
  computes first); a cycle of dependencies alone is refused;
* a step is committed only when every due connector validated and the water
  ledger closed; otherwise nothing changes (transactional commit).

## Order of work in one day

`vic` (window or day) -> `split` (sends the runoff of a box of VIC cells to
flood and the rest to routing, never both) -> `routing` (with `reservoir` / `quality`
inside the sweep) -> `flood` -> next day's `vic` sees `soil_add` from
irrigation or flood infiltration. Water not delivered in a step waits in the
exchange buffer, so each transfer closes (sent = delivered + buffered).

`benchmarks/run_combos.py` runs combinations of these modules together.
