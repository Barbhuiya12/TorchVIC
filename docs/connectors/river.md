# River system connectors: routing, reservoirs, water quality

The `routing` connector (`torchvic/coupling/connectors/routing.py`, registry
name `routing`) routes the runoff and baseflow that VIC produces for each cell
down a river network, returns the discharge at the gauge nodes and offers the
router's withdrawal and injection services to other connectors. Its engine is
`torchvic.model.routing.StepRouter` (`torchvic.model.routing_fast.FastStepRouter`
on CUDA unless `fast: false`). The `reservoir` (`torchvic.reservoir.Reservoirs`)
and `quality` (`torchvic.quality.WaterQuality`) connectors, in `reservoir.py`
and `quality.py` beside it, are plug-ins that run inside the routing step. The
connectors add no routing, reservoir or water-quality physics: they declare
the engines' inputs and outputs, and the routing and reservoir connectors
report the water balance that the ledger checks. Further reading:
[Reservoirs](reservoir.md), [Water quality](water_quality.md) and
[Routing to gauges](../guide/routing.md). Limits that apply to every module:
[Limits of the coupled system](../coupled/overview.md#limits).

## Wiring

```
vic.total_runoff_daily --(mm, [day, cell])--> routing.inflow
user.request ------(state, lag 1)----> routing.withdrawal_request
routing.withdrawn --(m3, lag 0)------> user.received
user.return_flow --(m3, lag 1)-------> routing.injection
routing.discharge  (state, m3 s-1, per gauge node)
reservoir  (plug-in of routing: request, evap in; storage, supplied out)
quality    (plug-in of routing: loads, met in; conc_<name>, temperature out)
```

`user` stands for any connector that takes water from the river; its request
is in m3 per routing step at each site. Requests and injections must not be
negative, and `withdrawn` is never more than passes the site without a
discharge becoming negative. The request and the return flow carry `lag: 1`; a
connector that reads `routing.withdrawn`, or a plug-in's output, at lag 0 must
declare `dependencies = ('routing',)` so that it computes after the router.

The `routing` module builds the network from the files `fdir` (D8 flow
directions, `file.nc:variable`), `domain` (`lat`, `lon`, `mask`, `area`,
optional `frac`) and `gauges` (a csv with `lat`, `lng`) or `gauge_cells`; or,
when embedded in Python, from the objects `net`, `gauge_cells` and `area`. It
builds the router at `step_s` (default 86400 s) and the sites: its own
`site_cells` and the cells of its plug-ins, which it lists with `reservoirs`
and `quality`. Channel parameters: `velocity` (default 1.5 m s-1), `diffusion`
(800 m2 s-1) and the within-cell unit hydrograph `cell_uh` (weights that sum to
1; default: no delay). Gauge nodes: channel gauges, extra channel nodes at
sites, sub-cell gauges (`subcell`). Outlets are channel nodes with no channel
node downstream; the ledger counts the discharge at the outlets. Example
configurations: `configs/vic_routing.yaml`,
`configs/vic_reservoir_routing.yaml` and `configs/vic_quality_routing.yaml`.

## Water ledger

Per step, in m3. The terms are defined in
[Water accounting](../WATER_ACCOUNTING.md), which also gives VIC's; P, ET and
closure are zero for the router. The ring is the router's ring buffer, the
discharge that each node will still receive from the water already routed.

| module | I_ext | Q_out | E_other | dS | ET |
|---|---|---|---|---|---|
| routing | volume received as `inflow` (depth x area) + injections | outlet discharge x step + withdrawals | water of cells that drain to no gauge or site | change of the ring at the outlets | |
| reservoir | water taken from the river | release + environmental release + spill + water passed through (dams not yet built) + supplied | | storage change | evaporation |

The tolerance is the ledger's rule (4 eps x n_terms x magnitudes, see
[Water accounting](../WATER_ACCOUNTING.md#tolerances)) with n_terms = cells +
ring length (the length of the unit hydrograph) + 4. The magnitude includes the router's cumulative counters
(their increments carry the rounding of the totals) and a floor of 1e-3 m3
(one litre), so that the tiny amount of water in a drained river is not read
as a residual.

## Tests

`tests/test_coupling_routing.py` tests the connectors on random daily runoff over
a random test network, on the CPU and, where available, the GPU: the discharge and
the ring equal the standalone `StepRouter` bit for bit, with and without a user
connector that requests, receives and returns water through the three
connections; the reservoir and water-quality connectors equal their engines
(`Reservoirs.step`, `WaterQuality.step`) and, together in one router, the water
equals the reservoir-only router; the module, coupled, mass and heat ledgers
close; dams not yet built pass water and mass; a rejected step is undone and a
restart equals the continuous run. With
VIC's Stehekin sample (marked `reference`, it needs the sample, see
[Testing](../development/testing.md)) on a small test network over its 16 cells,
which is not a river, the gauge discharge equals the standalone router fed VIC's
daily runoff plus baseflow bit for bit, with VIC at one-day windows and, through
`total_runoff_daily`, at 5-day windows, and the whole-run budget closes. A network read from
`fdir`, `domain` and `gauges` files equals the same network given as objects, and
the shipped configurations validate. The engines have their own tests:
`tests/test_river_network.py`, `tests/test_step_router.py`,
`tests/test_routing_fast.py`, `tests/test_reservoir.py` and
`tests/test_quality.py`.

## Limits

* VIC does not see the river: the coupling is one-way, so there is no flood or
  lake feedback to VIC through this connector, and water is withdrawn from the
  river at the sites, not from a cell's runoff before routing.
* With reservoirs and quality on one router each dam is a well-mixed store:
  constituents decay in it at their first-order rate, heat is not exchanged
  with the atmosphere (evaporation removes water only), and the initial
  storage holds no mass and no heat. A dam receives the mass removed from the
  river with the water it takes; a dam not yet built neither takes nor
  releases mass.
* Quality inputs (loads, meteorology) are `state` values per step, set in the
  configuration or connected. Mass is not part of the water ledger; its own
  per-node ledgers are checked in the quality connector's `validate`.
* Canal gauges (`node_inject`) are a router feature that the connector does
  not expose. The discharge of sub-cell gauges (`subcell`) is an output only,
  not part of the ledger.
* Use `vic.total_runoff_daily`; the window total `vic.total_runoff` is spread
  evenly over the days and smooths peaks.
