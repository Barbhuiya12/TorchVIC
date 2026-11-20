# Connectors

A connector is one module of a coupled run. It takes water or state from other
connectors, computes one process and hands its results on. VIC is the first
connector. A coupled run names the connectors and the connections between them
in one YAML file; a runtime orders them and a ledger checks that the water
closes ([Coupled runs](../coupled/overview.md)).

This page lists every connector. To set up a
coupled run for the first time, read the [overview](../coupled/overview.md) and
the [configuration file](../CONFIGURATION.md), then open the page of the
connector you need.

## Connectors at a glance

| connector (registry name) | what it computes | when to use it | page |
|---|---|---|---|
| `vic` | the land surface model | every coupled run | [VIC (connector API)](../CONNECTOR_API.md) |
| `routing` | river routing, with withdrawal and injection services | discharge at gauges inside a coupled run | [River routing](river.md) |
| `reservoir` | dams and operating rules, in the routing step | regulated rivers | [Reservoirs](reservoir.md) |
| `quality` | water temperature and constituents, in the routing step | stream temperature or constituent loads | [Water quality](water_quality.md) |
| `groundwater` | aquifer flow and river-aquifer exchange | recharge, pumping, water table | [Groundwater](groundwater.md) |
| `flood` | GPU flood hydraulics, one or two way | inundation of a floodplain | [Flood](flood.md) |
| `split` | sends the runoff of a box of VIC cells to the flood connector and the rest to the router | flood and routing in one domain | [Coupling graph](../COUPLING_GRAPH.md) |
| `irrigation` | water use and irrigation | irrigated areas | [Irrigation](irrigation.md) |
| `crop` | WOFOST 7.2 crop growth | crop yield and water stress | [Crops (WOFOST)](crop.md) |
| `crop` with backend `cropsyst` | VIC-CropSyst crop model | CropSyst crop growth | [CropSyst](cropsyst.md) |
| `vicwur_*` | the VIC-WUR water-use and water-management package | reproduce VIC-WUR 3.0 | [VIC-WUR](vicwur.md) |
| ensembles, EnKF, ETKF, LETKF | Python engines, not registry connectors | parameter ensembles and data assimilation | [Ensembles and assimilation](ensemble_da.md) |

## Reading a connector page

Every connector page has the same sections, in this order.

| section | content |
|---|---|
| What it does | the process, the engine and the source file |
| When to use it | the cases it fits and the cases it does not |
| Inputs | every file, table column and connection input, with units |
| Config block | a minimal YAML snippet for the `modules:` entry and its connections |
| Outputs | what it hands on and what it writes, with units |
| Technical details | equations, wiring, ledger terms, state and restart |
| Limitations | what it does not do |
| Verification status | what was checked, against what, and the result |

!!! note "Verification status"
    The coupled system is checked by the test suite for plumbing, water
    conservation and restart. It is not part of the evaluation of VIC reported
    on the [Verification](../verification.md) and
    [Performance](../performance.md) pages. Each connector page says what was
    checked for that connector.

## Next step

* Wire connectors together: [Configuration file](../CONFIGURATION.md) and
  [Coupling graph](../COUPLING_GRAPH.md).
* Check the water budget: [Water accounting](../WATER_ACCOUNTING.md).
* Write your own connector: [VIC (connector API)](../CONNECTOR_API.md).
