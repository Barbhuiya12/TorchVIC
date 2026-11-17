# Connector API

Per-connector pages:
[routing, reservoirs and water quality](connectors/river.md),
[irrigation](connectors/irrigation.md), [reservoirs](connectors/reservoir.md),
[crops](connectors/crop.md), [groundwater](connectors/groundwater.md),
[flood](connectors/flood.md), [ensembles and assimilation](connectors/ensemble_da.md),
[water quality](connectors/water_quality.md). Overview of the edges between
them: [Coupling graph](COUPLING_GRAPH.md). Limits of the coupled system:
[Coupled system overview](coupled/overview.md#limits).

Every module of a coupled run is a `torchvic.coupling.Connector`. The
runtime calls, in this order:

| method | when | contract |
|---|---|---|
| `initialize(context)` | once | load data, allocate state on `context.device`; set `interval_s` if it depends on the data |
| `prepare_step(context, dt)` | start of each interval | optional; must not change accepted state |
| `compute(inputs, context, dt) -> Proposal` | each interval | from the **accepted** state and `inputs` over `[context.t, context.t + dt)`; must not mutate the accepted state; raise `ComputeError` to refuse |
| `validate(proposal, context) -> [problems]` | each interval | the connector's own physical checks; `[]` when fine |
| `commit(proposal, context)` | only when every due connector validated and the ledger closed | make the proposal the accepted state (swap references, do not copy) |
| `save_state() / load_state(blob)` | checkpoint / restart | everything a restart needs; tensors on the CPU in the blob |
| `finalize(context)` | once | flush, close |

## Declarations

| attribute | meaning |
|---|---|
| `name`, `version` | the runtime sets `name` to the module's name in the configuration |
| `enabled` | from the configuration (`enabled: false` lists it without running it) |
| `dependencies` | modules it needs; a missing one is refused at construction |
| `inputs`, `outputs` | `VarSpec(name, units, dims, dtype, kind, grid, description)`; `kind` is `flux` (an amount over the interval, ledgered, buffered) or `state` (a value at the interval end, sampled) |
| `grids()` | `GridSpec(name, n, area)` (area in m2 per element; needed to turn depths into volumes) |
| `interval_s` | exchange interval in seconds (0: set in `initialize`) |
| `owned_state` | stores this connector alone writes; two modules that write one store are refused |
| `imported_fluxes`, `exported_fluxes` | which inputs/outputs are water transfers |
| `backends` | `('cpu',)`, `('cpu', 'cuda')` |
| `graph_safe` | its compute can be captured in a CUDA graph (static shapes, no host sync) |
| `checkpoint` | `'full'`: `save_state` is a complete restart |
| `validation` | how far its results are verified against the model it ports |

A connection is checked at construction and initialisation: both ends exist,
`flux` feeds `flux` and `state` feeds `state`, units convert (`unit
mismatch` otherwise; depth <-> volume needs grid areas), the grids are the
same or a declared conservative mapping joins them (`grid mismatch`
otherwise), and a flux output feeds at most one connection (the same water
cannot be delivered twice).

## Proposal and Balance

`Proposal(connector, t0, t1, state, exports, balance, series, backend, info)`.
`exports` holds the declared outputs the run's connections consume (device
tensors on the output's grid, in its units). `series` is the per-step output
for the output manager. `balance` is a `Balance`:

```
R = P + I_ext - ET - Q_out - E_other - dS - closure        (m3, whole domain)
```

with optional `per_cell` (`residual`, `scale` in mm, `n_terms`, `eps`) for
the element-wise check. A connector that receives water through an input
counts it in `I_ext`; one that sends water through a flux output counts it
in `Q_out`. Terms may be device tensors; the ledger reads them once at its
check. See [Water accounting](WATER_ACCOUNTING.md).

## The VIC connector

`VicConnector` wraps the existing run path; it adds no physics.

* **intervals** (`window_days` / `exchange`): `window_days: 31` (default), `0` or `exchange: whole` (one `Case.run`),
  `exchange: day`, `exchange: step`. Each interval is `Case.run` (or, for
  a pipeline with its own forcing reader, `basin.simulate` through
  `KwargsDriver`) on the interval's slice of the forcing, started from the
  state the previous interval exported. VIC through the runtime equals the
  direct run bit for bit (tests/test_coupling_vic.py; on the CPU every
  interval length equals the single run).
* **large domains** (`stream: auto | true | false`): a global file whose
  forcing exceeds 2 GiB (or `stream: true`) is not loaded by `case.load`
  but by `StreamDriver` (`torchvic/coupling/vic_stream.py`), the fast
  path of `benchmarks/run_isc_torchvic.py`: the forcing is read
  one exchange window at a time ahead of the compute (reader processes into
  shared memory, `forcing_feed: true`, default on a GPU; or a prefetch
  thread), at most `prefetch` windows on the device; a daily forcing is
  expanded by MTCLIM in cell blocks into host memory; VIC's CUDA graphs and
  fused regions run as in `torchvic run`; the history streams are
  aggregated per window and written at `finalize` (the per-step series is
  not kept). Cases it does not build (snow-band expansion, LAKES, CARBON,
  FROM_VEGHIST, COMPUTE_TREELINE, INIT_STATE, classic files, a daily model
  step) use `case.load`; multi-band cells stream only with `snow_bands:
  mean` (one column per cell at the band-mean elevation, as the benchmark
  runs them).
* **drivers**: `CaseDriver(case)` (a `torchvic.case.Case`), `KwargsDriver(
  static, window, dates, steps_per_day, area, frac)` (simulate's static
  arguments and a window callback -- `benchmarks/run_isc_torchvic.py
  --runtime` uses it with its prefetched forcing).
* **outputs** (only what VIC computes): fluxes summed over the interval, mm:
  `runoff`, `baseflow` (drainage out of the bottom layer: the recharge
  candidate a groundwater model takes), `total_runoff` (= runoff +
  baseflow, what routing takes), `et`, `transp`, `evap_canop`, `evap_bare`,
  `sub_snow`, `sub_canop`, `rainf`, `snowf`, `snow_melt`, `pet`,
  `soil_add_unmet`; states at the interval end: `swe`, `soil_moisture`
  [cell, layer], `wdew`, `snow_canopy`. The full per-step output (every
  OUT_* TorchVIC has) goes to the output manager and the history writer.
  Inter-layer drainage and the moisture flux between layers are computed
  inside `basin.simulate` and are not outputs; exposing them would need a change to
  the step code (not done).

### Injection points

Available now, without changing VIC's equations:

| input | how it enters | limits |
|---|---|---|
| `precip_add` [cell] (interval total, spread evenly over the steps) or [step, cell], mm | added to the forcing precipitation before the interval runs. It is then precipitation in every respect: split into rain and snow by air temperature, intercepted, infiltrated or run off. Tested: identical bit for bit to a run whose forcing carries the same extra precipitation. | refused under a precipitation correction (pcf != 1, CORRPREC), which would scale it; must be >= 0 |
| `soil_add` [cell] (into layer 0, as VIC-WUR's irrigation, `irr_use.c:265`), [cell, layer] or [cell, tile, layer], mm | added to the soil moisture of the state the interval starts from -- i.e. after the previous interval's last step, as VIC-WUR adds irrigation after `vic_run`. VIC's report totals (the opening of the first DEL terms) move by the same amount, so VIC's OUT_WATER_ERROR stays VIC's. Negative values withdraw. | above a layer's `max_moist`: refused, or with `soil_add_mode: over` accepted whole (the next `vic_run`'s runoff moves the excess on, `runoff.c:242-268`; the VIC-WUR groundwater group sets it, as VIC-WUR 3.0's capillary-rise edit leaves tiles above `max_moist`), or with `soil_add_mode: cap` the excess is not added and is exported as `soil_add_unmet` (handed back; ledgered). Below `resid_moist`: refused. Works on the GPU's lane-compacted state (covered tiles only) as on the padded grid. Not with FROZEN_SOIL or snow bands unless the VIC-WUR options `soil_add_frozen` / `soil_add_bands` are set. Not before the first interval (no state yet). |
| `soil_add_<src>` (with `soil_add_sources: [src, ...]`), mm, signed as soil_add | one more producer of soil_add (e.g. `irrigation` and floodplain infiltration `flood` on the same cells): VIC adds the sum of `soil_add` and every `soil_add_<src>` once, with one capacity check. | the unmet (cap mode) is split back in proportion to what each producer added to the cell (positive parts only): `soil_add_unmet_<src>` (and `soil_add_unmet` for the plain `soil_add`); the parts add up to the cell's unmet, none exceeds its producer's addition. One flux has one consumer (the runtime refuses `soil_add_unmet` feeding two). Not with `day_coupled`. |
| `lai` [cell, tile] or [cell, tile, step] (state input) | replaces the vegetation history's per-step LAI for the interval (the existing FORCING2 path VIC reads per step), with VIC's leafless rule (vic_force.c 5.1.0:419-439: LAI 0 or fcanopy below MIN_FCANOPY sets both to 0) | only with LAI_SRC FROM_VEGHIST (refused otherwise, with the reason); same-run crop -> VIC feedback at the interval boundary |

For VIC-WUR-style irrigation at daily granularity, run VIC with `exchange:
day` (a daily model step then gives exactly VIC-WUR's "after each step");
a sub-daily model step gets the addition once a day, at the day boundary.

Daily coupling inside VIC's windows (`day_coupled`): a module listed in
VIC's `day_coupled` runs between the days of each VIC window, from inside
the window's single `basin.simulate` call (its `day_hook`; the runtime's
`DayCoupler`). Each day it reads VIC's soil moisture at the day's start
(`vic.soil_moisture`, lag 1) and returns that day's `soil_add` /
`precip_add` (lag 0), which the hook writes into the loop's soil moisture
and the coming steps' precipitation buffer -- on the GPU into the captured
graphs' own buffers, so the window keeps its graphs. The result is, bit for
bit, the run with one-day VIC windows exchanging at every boundary
(`tests/test_coupling_daily.py`). The day-coupled modules commit day by day inside the window
and are restored from a snapshot if the step is rejected.

Vegetation (`lai`) still changes at window boundaries (a whole window's
per-step series may be given ahead).

Not available, because each would need a change to VIC's step code (a
`basin.simulate` hook, a device tensor read inside the step loop, not a new
equation):

* water added *inside* an interval at the model step (per-step soil or
  surface addition within a 31-day captured window);
* throughfall-like addition below the canopy (bypassing interception);
* a ponded surface store or a surface-water body fed by modules;
* groundwater as VIC's lower boundary (two-way: a water table that changes
  baseflow or capillary rise) -- this replaces ARNO baseflow physics and is
  a physics change, not available;
* withdrawals from runoff before routing within the step (today: take from
  the `total_runoff` export, i.e. after VIC, which is what VIC-WUR does);
* crop-driven rooting depth or fraction (LAI is available, see `lai`);
  LAI that changes inside an interval from a model running at the same
  step (the `lai` input takes a whole interval's per-step series ahead).

## A stepping connector (routing, reservoirs)

The API supports a connector that steps at the model step with per-step
state: `interval_s` = the model step, state on the device (e.g. the
`StepRouter` ring buffer and head), `compute` = one `begin -> withdraw /
inject -> end` sweep returning the new ring as the proposal (no copy:
compute writes into a second buffer and commit swaps the two), exports such
as `discharge` (state) and `withdrawn` (flux), inputs such as `inflow` (flux
from VIC's `total_runoff`, taken from VIC's window buffer prorated per step)
and `withdrawal_request`. With VIC on 31-day windows and routing per step,
VIC's runoff is pushed once per window and delivered per step (lag 0, VIC is
the slower producer). A reservoir release that must change runoff routing in
the same step is a lag-0 connection routing -> reservoir -> routing, i.e. an
iterated cycle, or two halves of the routing step as two connectors.

## The river connectors (routing, reservoirs, water quality)

`torchvic/coupling/connectors/{routing,reservoir,quality}.py`, configurations
`configs/vic_routing.yaml`, `vic_reservoir_routing.yaml`,
`vic_quality_routing.yaml`; details in [river connector](connectors/river.md).

* **RoutingConnector** (`routing`): the stepping router (`StepRouter` on the
  CPU, `FastStepRouter` on CUDA), `interval_s` = `step_s` (one day). Inputs
  `inflow` (flux, mm, VIC's `total_runoff`), `withdrawal_request` (state, m3
  per step at each site), `injection` (flux, m3 at each site); outputs
  `discharge` (state, m3 s-1 at the gauge nodes) and `withdrawn` (flux, m3
  delivered to the requests). The withdraw / inject services are these
  connections: a user connector sends its request as `withdrawal_request`
  (usually `lag: 1`), reads the delivered water from `withdrawn` (lag 0, it
  must declare `dependencies = ('routing',)` so it computes after the
  router) and returns water through `injection`.
* **Sweep plugins** (`ReservoirConnector`, `QualityConnector`): their
  computation happens inside the router's level-by-level sweep, so the
  routing connector lists them (`reservoirs:`, `quality:`; a declared
  dependency, so they compute first). Their `compute` stages the inputs and
  returns a `Proposal` with empty exports; the routing connector's compute
  runs the shared step and fills that proposal (exports, balance, series)
  before the runtime validates and checks. A consumer of a plugin's output
  at lag 0 must declare `dependencies = ('routing',)`.
* **Transactions**: the engines update their rings in place. `compute` takes
  an undo snapshot of every engine tensor (one foreach copy) and `commit`
  drops it; any later call (the next `compute`, `save_state`) restores a
  snapshot that was not committed. The accepted state is therefore always the
  state before a rejected step.
* **Balance**: routing's `Balance` counts the volume handed over as `I_ext`
  (plus injections), the discharge x dt at the outlet gauges and the
  withdrawals as `Q_out`, the change of the water in the ring as `dS`, and
  the water of cells that drain to no gauge as `E_other`. A reservoir's
  balance: the water it takes is `I_ext`, evaporation `ET`, release + spill
  + supplied water `Q_out`, storage change `dS`; routing counts the same take
  as a withdrawal and the release as an injection, so they cancel in the
  coupled balance.
* **Window runoff**: connect `vic.total_runoff_daily` (flux, mm, [day,
  cell]: VIC's own daily series of runoff + baseflow, summed from the
  model steps of each day) to `routing.inflow`. The exchange buffer of such
  an output (`ExchangeBuffer(profile=True)`, chosen by the output's first
  dimension `day`) hands the consumer the days its interval covers, so the
  router gets each day's runoff on that day and the flood peak VIC
  computed. Routing through VIC's 31-day windows then equals, bit for bit,
  the standalone router fed VIC's daily series (`tests/test_coupling_routing.py`). The window total
  `vic.total_runoff` is still available and is spread evenly (it smooths
  peaks). The router is one-way from VIC, so it is not a day-coupled module;
  it runs after the window.
* **Reservoirs and quality together**: the sweep is then
  `connectors/_combined.py` (`ReservoirQuality`): water as in a
  reservoir-only router (bit for bit), each dam a well-mixed store of
  constituents and heat (first-order decay, no heat exchange, clean initial
  storage); mass removed with the take is the mass that leaves the outlets'
  rings; the store identity `store = removed - released - supplied -
  decayed` and the per-node ledgers close.

## Adding a connector, step by step

1. **Write the class** in `torchvic/coupling/connectors/<name>.py`,
   subclassing `Connector`. Set `name`, `version`, `backends`
   (`('cpu',)` or `('cpu', 'cuda')`), `validation` (what it was checked
   against), `owned_state`, `imported_fluxes` / `exported_fluxes`.
   `torchvic/coupling/toy.py` (`TwoStoreConnector`) is the smallest worked
   example, with a balance that closes to rounding.
2. **Declare the interface**: `inputs` / `outputs` as `VarSpec(name, units,
   dims, dtype, kind, grid, description)` (`kind='flux'` for an amount over
   the interval, `'state'` for a value at its end) and `grids()` as
   `GridSpec(name, n, area)`; `interval_s` (or set it in `initialize`).
3. **Implement the methods** in the order the runtime calls them:
   `initialize`, `compute(inputs, context, dt)` returning a `Proposal` built
   from the accepted state without mutating it, `validate`, `commit` (swap
   references), `save_state` / `load_state` (tensors on the CPU),
   `finalize`. Raise `ComputeError` to refuse a step.
4. **Fill the `Balance`** of every proposal: `R = P + I_ext - ET - Q_out -
   E_other - dS`. Count water received through an input as `I_ext` and water
   sent through a flux output as `Q_out`. On a GPU, return device tensors and
   let the ledger read them once; do not call `.item()` in `compute`
   (`DeviceFlag` and `hostfetch.py` exist for the one host copy per step).
5. **Register it**: in `torchvic/coupling/registry.py` add a factory function
   and `register('<name>', _factory, 'available')`, or call
   `registry.register` from your own package.
6. **Configure it**: add a module block (`connector: <name>`, options,
   `enabled`) and `connections:` to a YAML file; run `torchvic validate
   --config file.yaml` first (unit, grid, dependency and lag-0 cycle errors
   are reported here).
7. **Test it**, following `tests/test_coupling_*.py`: ledger closes, a
   rejected step changes nothing (`validate` refusing), restart equals the
   uninterrupted run, a missing dependency and a unit mismatch are refused,
   and your outputs equal the standalone engine bit for bit when it exists.
8. **Document it** in `docs/connectors/<name>.md` and add a row to
   [Coupled system overview](coupled/overview.md) and [Coupling graph](COUPLING_GRAPH.md).
   Credit any upstream model in `THIRD_PARTY_NOTICES.md`; GPL-2.0-only code
   is only used as an external process.

## Input formats

| input | format |
|---|---|
| VIC case | VIC 5 global parameter file, image-driver domain, parameter and forcing NetCDF (or Zarr forcing) |
| coupled run | YAML (`torchvic run --config`, `torchvic validate --config`), `${ENV}` substitution; path keys resolved against the file's directory |
| routing | flow-direction NetCDF (`fdir`), domain NetCDF, gauge table (`gauges`) |
| reservoirs | dam table (`dams`, GDW/GRanD columns) |
| irrigation | irrigated fraction per cell, season window or crop calendar, Wcr/Wfc from the parameter file |
| crop | WOFOST parameter JSON (`configs/crop_params_sbarley.json`), optional `height_table` |
| groundwater | per-cell top, bottom, K, Sy, initial head |
| flood | basin `.npz` plus `*_vic_pairs.csv` (`tools/flood/make_basin.py`) |
