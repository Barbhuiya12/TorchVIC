# VIC-WUR water use and management

The VIC-WUR connectors reproduce the water-use and water-management package
of VIC-WUR 3.0 (Droppers et al., 2020; see [References](../about/references.md))
inside the coupled system.

## What it does

VIC-WUR couples VIC with river routing, dams, sector water use, irrigation,
environmental flow requirements (EFR), land-use change, the WOFOST-NPK crop
model and a MODFLOW aquifer. TorchVIC implements each part as its own module
and connector (registry names `vicwur_routing`, `vicwur_dams`, `vicwur_efr` and
so on, in `torchvic/coupling/connectors/vicwur_*.py`), so a configuration can
use any subset. The reference version is VIC-WUR 3.0.0. The connectors of a run
join one phase group and run on the `vicwur3` VIC core.

## When to use it

* Use it to reproduce VIC-WUR 3.0: dams, sector water use, non-renewable
  abstraction, EFR, irrigation, crops, land use and the MODFLOW coupling, with
  inputs from a VIC-WUR global file.
* Use `exact` mode ([Modes](#modes)) to compare with the compiled VIC-WUR and
  `fixed` mode (default) for the corrected model.
* Do not use it when the lumped irrigation connector is enough: see
  [Irrigation](irrigation.md), which remains available alongside
  `vicwur_irrigation`.
* Do not use it for the CropSyst crop engine alone ([CropSyst](cropsyst.md)) or
  for the aquifer alone ([Groundwater](groundwater.md)).

## Inputs

The inputs are those of VIC-WUR 3.0 and are named in a VIC-WUR global file.
`torchvic.coupling.connectors.vicwur_setup.modules` reads them.

| input | file or variable | unit | meaning |
|---|---|---|---|
| sector water demand | plugin forcing files of the sector demands | mm per model step, per sector (MUN, ENE, MAN, LIV, IRR) | demand of each sector, held between records |
| groundwater fraction | plugin forcing files | - (fraction) | share of the demand taken from groundwater, held between records |
| consumption fraction | plugin forcing files | - (fraction) | share of the withdrawal consumed, held between records |
| EFR discharge | plugin forcing file `EFR_DISCHARGE` | m3 s-1 | discharge for the environmental flow requirement |
| EFR baseflow | plugin forcing file `EFR_BASEFLOW` | mm per model step | baseflow for the environmental flow requirement |
| routing network | parameter file of the routing network | - | flow network and routing levels of the cascade router |
| dams | parameter file of the dams | capacity in hm3 | register of global and local dams |
| remote water use | parameter file of remote water use | - | source and receiver cells |
| irrigation | parameter file of irrigation | - | irrigated tiles, efficiency, groundwater fraction, offset |
| crops | parameter file of the crops | - | WOFOST-NPK crop parameters |
| land cover | land-cover (`Cv`) records | - (area fraction) | land-cover fraction per vegetation class, for `vicwur_landuse` |
| CO2 | plugin forcing file `CO2` | ppm | atmospheric CO2 for `vicwur_co2` |
| MODFLOW static values | `groundwater` argument of `modules` (needed with `GWM TRUE`) | as on the [Groundwater](groundwater.md) page | static values per cell for the `vicwur3` profile |

The sample configuration has no `connections:` entries (`connections: []`): the
connectors of the phase group exchange data in the order of the VIC-WUR driver
([Order of the step](#order-of-the-step)) and act inside the VIC step through
the hooks of the `vicwur3` core ([VIC core and hooks](#vic-core-and-hooks)).

## Config block

No example YAML file for the VIC-WUR connectors ships in `configs/`; the blocks
are the `modules` of a configuration ([Configuration](../CONFIGURATION.md)).
`vicwur_setup.modules` builds them from the global file and fills the other
options (parameter tables and forcing streams). A block has this form:

```yaml
modules:
  vic:
    connector: vic
    global_param: path/to/global_vicwur.txt
    core: vicwur3
    phase_group: vicwur
    day_coupled: [vicwur_routing, vicwur_dams]
  vicwur_routing:
    connector: vicwur_routing
    group: vicwur
  vicwur_dams:
    connector: vicwur_dams
    group: vicwur
    exact: false          # true: reproduce the C code of VIC-WUR 3.0
connections: []
```

## Outputs

| output | unit | meaning |
|---|---|---|
| `vicwur_routing`: `discharge` (`OUT_DISCHARGE`) | m3 s-1 | routed discharge per cell |
| `vicwur_routing`: `stream_moist` (`OUT_STREAM_MOIST`) | mm | in-stream water per cell |
| `vicwur_dams`: `storage` | m3 | storage per dam |
| `vicwur_wateruse_local`: `withdrawn` | mm | withdrawal per cell |
| `vicwur_efr`: `efr_discharge` (`OUT_EFR_*`) | m3 s-1 | EFR discharge per cell |
| `vicwur_irrigation`: `OUT_REQUIREMENT`, `OUT_SHORTAGE`, `OUT_RECEIVED`, `OUT_APPLIED`, `OUT_LEFTOVER` | mm | irrigation requirement, shortage, received, applied and leftover water per cell |
| `vicwur_landuse`: `cv` (`OUT_CV`) | - | land-cover fraction per cell and class |
| `vicwur_co2`: `catm` (`OUT_CATM`) | ppm | atmospheric CO2 per cell |

The outputs also include the plugin water balance.

## Technical details

### Connectors

| connector (registry name) | what it does |
|---|---|
| `vicwur_routing` | the VIC-WUR cascade router: runoff and baseflow routed over the river network by level sweep; the connector that all water-use withdrawals and returns act on |
| `vicwur_dams` | global dams (register, operation, release) and local dams |
| `vicwur_wateruse_local` | local water use: withdrawals from soil water, the river and dams, by sector, with returns |
| `vicwur_wateruse_remote` | remote water use: unmet demand served from a source cell and returned there |
| `vicwur_nonrenew` | non-renewable abstraction of residual demand, a deficit repaid from later runoff and baseflow |
| `vicwur_efr` | environmental flow requirements: the share of the liquid water whose baseflow stays below the required baseflow |
| `vicwur_irrigation` | irrigation demand, application, shortage and leftover per tile and band (the lumped connector is described in [Irrigation](irrigation.md)) |
| `vicwur_crop` | WOFOST-NPK crop growth with a feedback of LAI, canopy fraction, albedo and roughness to VIC |
| `vicwur_cropsyst` | the CropSyst crop engine run inside the VIC-WUR group ([CropSyst](cropsyst.md)) |
| `vicwur_co2` | the CO2 effect on canopy resistance (see [VIC core and hooks](#vic-core-and-hooks)) |
| `vicwur_landuse` | changes in land-cover fraction between records, with the redistribution of the tile water and energy states |
| `groundwater` with the `vicwur3` profile | MODFLOW coupling of VIC-WUR 3.0 ([Groundwater](groundwater.md)) |

Each connector is registered under its own name. The connectors of a run join
one phase group, which runs them in the order of the VIC-WUR driver
([Order of the step](#order-of-the-step)).

### VIC core and hooks

To reproduce VIC-WUR 3.0 the connectors run on the `vicwur3` VIC core, the
`vic_run` of VIC-WUR 3.0 (`core: vicwur3` in the `vic` block, or
`torchvic run --core vicwur3`). The connectors that act inside the VIC step do
so through optional hooks, held by a `VicwurCore` object that is passed to the
time loop as `vicwur_core` and set by the connectors that need them. The hooks
include the saturated conductivity and the critical moisture of each layer and
tile, the vegetation properties and land-cover fractions of the tiles, the
stored water stress, the CO2 effect on canopy resistance and the soil of the
groundwater coupling (`GWM`) with its recharge. With every hook off, VIC runs
as it would without the connectors.

### Modes

The components that reproduce a defect of the C code of VIC-WUR 3.0 have two
modes. There is no single configuration key: the mode is set by the `mode`
argument of
`torchvic.coupling.connectors.vicwur_setup.modules(global_path, case, ...)`
(`'fixed'` or `'exact'`), which builds the module blocks from a VIC-WUR global
file, or per connector (for example `exact` on `vicwur_dams` and `vicwur_efr`,
`mode` on `vicwur_landuse` and on `groundwater` with the `vicwur3` profile).

* **`fixed`** (default). Corrects the upstream defects listed below. Use it
  unless the aim is a comparison with the compiled VIC-WUR.
* **`exact`**. Reproduces the C code of VIC-WUR 3.0 including its quirks. Use
  it to compare with the compiled VIC-WUR.

Defects corrected in `fixed` mode:

| upstream behaviour (`exact` mode) | `fixed` mode |
|---|---|
| land use writes past a two-element array, giving node 0 the heat capacity of layer 1, and hands the irrigation leftover of a changed tile to the river once per snow band instead of weighted by area, which creates water | node 0 keeps its own heat capacity; the leftover is weighted by the area of the band |
| pumping ignores the ice and can leave soil moisture below the ice content (the ice clamp of the C code is dead code) | pumping stops at the ice content: only liquid water is taken |
| EFR non-linear baseflow uses cell 0's parameters | each cell's own |
| dam demand uses the dam cell's area for every service cell and a calendar-indexed month count | the service cell's area and a month-aligned step count |
| the fcanopy share of rain is lost as throughfall on a leafless vegetated tile | the rain passes and water is conserved |
| a losing river's aquifer exchange is added to the cell's inflow whatever the river holds, so routed discharge can go negative | solved in the routing sweep (slot `river_exchange`), taken from the water in the reach; see [Groundwater](groundwater.md) |

Opt-in (a choice of model behaviour, not a correction): `vicwur_nonrenew` with
`return_to='aquifer'` returns the unused part of the pumped water to the aquifer
of the same cell. The default, `return_to='river'`, returns it to the river as
VIC-WUR 3.0 does.

### Order of the step

In each model step the connectors run in the order of the VIC-WUR driver:
forcing, update of the step variables, the VIC step, a first pass over the
cells, the sweep over the routing levels, a last pass over the cells and the
outputs. The first pass holds the irrigation requirement, the sector demand and
routing's conversion of runoff and baseflow with the non-renewable payback. The
sweep visits the routing levels upstream first, with local dams, routing, local
water use, global dams and remote water use. The last pass holds non-renewable
abstraction, irrigation withdrawal and shortage, and crops; the outputs include
the plugin water balance. The level sweep is equivalent to the serial order of
the C code. The runtime that runs the group is described in
[Architecture](../ARCHITECTURE.md) and [Coupling graph](../COUPLING_GRAPH.md).

### Setting up

The inputs are those listed under [Inputs](#inputs). `modules` builds the module
blocks of the switched-on components, plus the `vic` block. With `GWM TRUE` it
also needs the `groundwater` argument (the MODFLOW static values per cell) and
adds the `groundwater` member with the `vicwur3` profile.

```python
from torchvic.coupling import TorchVIC
from torchvic.coupling.connectors import vicwur_setup as WS

case = WS.load_case('global_vicwur.txt', 'scratch')    # 'scratch': an existing directory
mods = WS.modules('global_vicwur.txt', case)           # includes the 'vic' block
app = TorchVIC.from_dict(dict(run=dict(device='cpu'), modules=mods,
                              connections=[]))
app.initialize().run()
```

### Large domains

Two options trade device memory for host memory. `run.keep_outputs_on`
(`device`, `host` or `none`) sets where the kept output series live: with `host`
they are copied off the GPU at each commit, with `none` they are not kept.
`vic: forcing_on: host` keeps the VIC forcing in pinned host memory and copies
each window to the device when it runs, so the device holds one window of
forcing whatever the length of the run. See
[Configuration](../CONFIGURATION.md).

States and outputs of repeated runs are bit-identical on the CPU and on the
GPU: the sums over bands and cells use a fixed order instead of GPU atomics.

## Limitations

* The inputs (irrigation efficiency, sector water demand, groundwater and
  consumption fractions, and the other files listed under
  [Inputs](#inputs)) are those of VIC-WUR and must be supplied by the
  user. Results are not calibrated.
* The land-use connector does not redistribute carbon pools (`CARBON` is not
  wired with the VIC hooks).
* Licensing of the VIC-WUR-derived components: see
  [License and credits](../downloads/code.md#license-and-credits) and
  `THIRD_PARTY_NOTICES.md`.
* Limits that apply to every coupled run:
  [Limits of the coupled system](../coupled/overview.md#limits).

## Verification status

Checked against reference runs of VIC-WUR 3.0 in the test suite, in `exact`
mode. The tests are skipped when the reference runs are absent.

### Agreement with VIC-WUR 3.0

The reference runs of VIC-WUR 3.0 were replayed in `exact` mode, component by
component (through the connectors and at kernel level) and end to end, with VIC
in the loop. The outputs agree with the reference output files to the precision
of those files (float32), except after the first day on which the discontinuity
of the C code described below applies. A run restarted from a checkpoint
reproduces the continuous run bit for bit (on CUDA, states and outputs bit for
bit and ledger rows to rounding). The reference runs are not distributed with
the repository; the tests that use them are skipped when they are absent
([Testing](../development/testing.md)).

The tests are the `tests/test_vicwur_*.py` files: replays of routing, dams,
local and remote water use, non-renewable abstraction, EFR, irrigation, land use
and WOFOST-NPK component by component (`*_l2.py` and the kernel-level tests),
end-to-end runs with the connectors built from the reference global files
(`test_vicwur_l3.py`, `test_vicwur_connectors_b.py`,
`test_vicwur_crop_feedback.py`), the `vicwur3` core (`test_vicwur_core.py`) and
restart tests (`test_vicwur_l3.py`, `test_vicwur_connectors.py`,
`test_vicwur_connectors_b.py`, `test_vicwur_w3_restart.py`).

A known cause of divergence between `exact` mode and the compiled model is a
discontinuity in the C code. When a withdrawal empties a river ring to
round-off, VIC-WUR adds the return flow to the ring in proportion to the
contents of its bins if their sum is positive, and evenly over the bins if the
ring holds no water. `exact` mode copies this rule, so the proportional or the
even branch is taken according to round-off, and beyond the first day on which
the two codes take different branches the river can agree only if VIC's states
are bit-identical.

The lumped irrigation connector ([Irrigation](irrigation.md)) remains available
alongside `vicwur_irrigation`.


## Next step

[Irrigation](irrigation.md) for the lumped irrigation connector, or
[Groundwater](groundwater.md) for the MODFLOW coupling.
