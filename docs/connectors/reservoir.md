# Reservoir connector

The reservoir connector (registry name `reservoir`; class `ReservoirConnector`
in `torchvic/coupling/connectors/reservoir.py`) operates dams on the river
network inside the routing step. Its engine, `torchvic.reservoir.Reservoirs`
(`torchvic/reservoir/system.py`), is vectorised over dams and runs on the device
of the routing connector (CPU or CUDA); the dam-table helpers are in
`torchvic/reservoir/data.py`. A dam sits at a router site: each step it takes
the water reaching the site (never more than passes without driving the river
negative), updates its storage and returns release plus spill to the river at
the same site, level by level, so a dam downstream sees the release of the dam
above it in the same step. The routing module lists the connector in its
`reservoirs` option ([routing connector](river.md)); the connector registers its
cells with the routing connector and owns the storage state. Example
configuration: `configs/vic_reservoir_routing.yaml` (see
[Configuration](../CONFIGURATION.md)).

## Dam table

The `dams` option is a list of mappings, or the path of a csv file with the same
columns. `cell` and a capacity are required; the other columns are optional.

| column | meaning |
|---|---|
| `cell` | index of the river cell (the router's numbering) |
| `capacity_m3` (or `capacity_hm3`) | total storage, m3 (`capacity_hm3` in hm3, 1 hm3 = 10^6 m3) |
| `dead_m3` | dead storage, m3 (default 0) |
| `q_env` | minimum release, m3 s-1 (default 0) |
| `q_max` | maximum rule release, m3 s-1 (default unlimited) |
| `build_year` | active from this year (default always); a dam not yet built passes its inflow unchanged |
| `init_frac` | initial storage / capacity (default 0.5) |
| `irrigation` | true selects the irrigation branch of the Hanasaki rule (default false; in a csv `1`, `true` or `yes`) |
| `area_max` | surface area at full storage, m2 (default 0: no evaporation) |
| `area_exp` | exponent p of the area-storage relation, A = `area_max` (S / capacity)^p (default 1.5) |
| `rule` | `hanasaki` (default), `target` or `pass` |
| `target_frac` | 12 monthly target fractions of capacity, January first (rule `target`; default 1 in every month; in a csv one field of 12 numbers separated by spaces) |

The capacity, build year, irrigation and area columns correspond to the
capacity, year, irrigation use and reservoir area of the Global Dam Watch (GDW,
which includes GRanD) records (Lehner et al., 2011; 2024). In
`torchvic/reservoir/data.py`, `read_gdw_dams` reads the GDW barriers file (India
by default), `place_dams` places each dam on a river cell by matching upstream
area, and `merge_by_cell` merges dams that share a cell.

## Rules

The `rule` column selects one of three rules.

* `hanasaki` (default): the monthly scheme of Hanasaki, Kanae and Oki (2006,
  J. Hydrol. 327, 22-41), with the release and capacity-correction formulas of
  VIC-WUR (Droppers et al., 2020); see below.
* `target`: the release per step is the storage above the target fraction of
  capacity for the month (`target_frac`) divided by `relax_steps` (connector
  option, default 10).
* `pass`: run of river.

The `hanasaki` rule computes twelve monthly target releases from the monthly
climatology of inflow and demand and from the environmental flow `q_env`. The
climatology is prescribed with the connector options `inflow_clim` and
`demand_clim` (12 monthly values in m3 s-1, January first, shared by all dams or
one set per dam) or learned online as a running mean over the simulated years;
until it covers all twelve months a dam passes its inflow through. The release
is scaled by the storage at the start of the operational year relative to the
desired maximum (alpha = 0.85, Hanasaki et al., 2006), and blended with the
monthly inflow for a dam whose capacity is below half its mean annual inflow
volume. The operational year starts in the month after the minimum of the
cumulative inflow anomaly (the running sum of the monthly inflow minus its
annual mean), that is at the start of the filling season; this is a
simplification of the rule of the paper.

Per step, storage is updated by the inflow, evaporation, withdrawal and release.
Withdrawal and release are limited to the water above dead storage, and storage
above capacity is spilled and continues downstream with the release, so the
storage balance closes to rounding error.

## Inputs and outputs

| name | kind | meaning |
|---|---|---|
| in `request` | state, m3 per routing step | water-use demand on each dam (default the configuration's `request_m3`, a scalar or one value per dam; none if neither is given) |
| in `evap` | state, m | potential evaporation depth per routing step (default the configuration's `evap_m`, a scalar or one value per dam; none if neither is given) |
| out `storage` | state, m3 | storage at the step end |
| out `supplied` | flux, m3 | water delivered to `request` (a conservative transfer) |

Release and spill return to the river inside the sweep. The connector's `series`
(see [Connector API](../CONNECTOR_API.md)) holds, per step, `storage`, `inflow`,
`release` (environmental flow plus rule release), `spill`, `evap` and
`supplied`.

## Calendar, state and restart

The rules are monthly. `start_date` (YYYY-MM-DD, default 2000-01-01) is the date
of t = 0; the engine's `new_month(month, year)` runs at the first step and
whenever the date enters a new month. State (storage, histories, targets,
accumulators) is checkpointed, so a run can be restarted.

## Ledger

Per step, inflow I (water taken from the river) = evaporation + supplied +
release + spill + dS, where release is the environmental flow plus the rule
release and dS is the change of storage; a dam not yet built passes its inflow
through unchanged. The routing ledger counts the take as water leaving to a
connector and the injection as water entering from one, so the transfers cancel
in the coupled balance ([Water accounting](../WATER_ACCOUNTING.md)).

## Tests

`tests/test_reservoir.py` tests the engine on small constructed river networks:
filling, spilling and emptying down to dead storage, environmental flow,
withdrawals, evaporation from the area-storage relation, the `target` rule, the
Hanasaki schedule against hand-computed cases, conservation with the router in
float64, dams before their build year, restart equality, and the batched engine
against dam-by-dam application. `tests/test_reservoir_data.py` tests placing dams
on river cells by upstream area and merging dams that share a cell (the test that
reads the GDW file is skipped when the data are absent). `tests/test_vicwur_dams.py`
compares the Hanasaki release and capacity-correction functions with VIC-WUR's C
code (it builds the C reference and is skipped when that fails).
`tests/test_coupling_routing.py` tests the connector in the coupled runtime on a
constructed river network and on VIC's Stehekin sample: discharge and storage
equal the standalone engine bit for bit, the module and coupled ledgers close,
restart across a month boundary and rollback of a rejected step, and dams
combined with water quality in one sweep. See [Testing](../development/testing.md)
for how to run the tests and the data some of them need.

## Limits

Limits that apply to the coupled system as a whole are listed in
[Limits of the coupled system](../coupled/overview.md#limits).

* A cell holds at most one dam. The connector stops with an error when two dams
  of the table share a cell; `merge_by_cell` (`torchvic/reservoir/data.py`)
  merges dams that fall in one cell (capacities and surface areas add, the
  earliest build year is used, and the dam is an irrigation dam if any of them
  is).
* Seepage is not represented. VIC-WUR's local dams are not part of this
  connector; the `vicwur_dams` connector covers the global and local dams of
  VIC-WUR ([VIC-WUR](vicwur.md)).
* The irrigation connector does not draw on the dams: its reservoir source is a
  store held inside the irrigation connector ([Irrigation](irrigation.md)). Water use on the dams is given by
  `request_m3`, or by a connection to `request`.
* The VICRes rule curves and operating rules (Dang et al., 2020, HESS 24,
  397-416) are not implemented; no VICRes code is used.
* For the code licences see [License and credits](../downloads/code.md#license-and-credits) and
  `THIRD_PARTY_NOTICES.md`.
