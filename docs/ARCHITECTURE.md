# Architecture

TorchVIC runs VIC alone (`torchvic run`) or as the hydrological core of a
coupled system of connectors (`torchvic run --config x.yaml`). VIC's physics
(`torchvic/model/basin.py`, `torchvic/core/*`) is the same in both: the VIC
connector calls the same `Case.run` / `basin.simulate` that `torchvic run`
uses, one exchange interval at a time, with the loop state carried through
`return_state` / `initial_state`.

## Layers

```
torchvic run --config x.yaml        torchvic validate --config x.yaml
        |                                     |
TorchVIC (runtime.py)  from_config / initialize / run / save_checkpoint / finalize
        |
Runtime
  registry      which connectors exist (registry.py)
  schema        YAML/JSON configuration, every problem listed (schema.py)
  graph         dependency graph, lag-0 ordering, cycles (graph.py)
  managers      TimeManager       multi-rate clock (gcd of intervals)
                StateManager      which module writes which store, accepted exports
                SpatialMapper     conservative grid-to-grid mappings
                ExchangeBuffer    water in transit between two modules
                BackendManager    device per connector, what actually ran
                OutputManager     per-step series, sinks
                CheckpointManager states + clock + buffers + ledger
  ledger        module and coupled water balances (ledger.py)
        |
Connectors (connector.py API)
  vic, routing, reservoir, quality, irrigation, crop, groundwater, flood,
  and a conservative test module (two_store)
```

The connector interface is described in [Connector API](CONNECTOR_API.md),
the edges between connectors in [Coupling graph](COUPLING_GRAPH.md), the
configuration keys in [Configuration](CONFIGURATION.md), and the limits of the
coupled system in [Coupled system overview](coupled/overview.md#limits).

## Runtime: one step

The runtime advances by a base tick, the greatest common divisor of the
connectors' exchange intervals. At each tick the connectors whose interval
starts there are due; each computes over its own interval. The step is a
transaction:

1. **read accepted**: inputs come from producers' accepted exports (lag >= 1)
   or from producers earlier in the order that proposed this tick (lag 0);
   flux inputs come out of the connection's exchange buffer, prorated in time
   and mapped to the consumer's grid and units;
2. **compute**: every due connector returns a `Proposal` (new state, exports,
   balance, per-step series) without changing its accepted state;
3. **validate**: each connector's own checks, then the ledger (each module's
   balance, the coupled domain's balance);
4. **commit**: all connectors, buffers, ledger rows and the clock advance
   together; if anything raised or failed, nothing does (`StepRejected` lists
   why).

Commit swaps references: the proposal's device tensors become the accepted
state, so exchanged fields stay on the device.

## Multi-rate exchange

Each flux connection has a buffer. A producer pushes its interval's amount
`[t0, t1, amount]`; a consumer takes the part that overlaps its own interval
(amounts are spread uniformly in time), so a 31-day VIC window can feed a
daily consumer, and a daily producer can feed a slower consumer with
`lag: 1` (what is not yet delivered is in transit, a store of the coupled
domain). At lag 0 a producer slower than its consumer is fine; a faster one is
refused at initialisation.

A cycle of lag-0 connections is refused unless every connection in it declares
`iterate: {max_iter, tol}`; the members are then computed in turn until their
exports change by at most `tol`.

## Ledger

The ledger holds each module's water balance and every transfer between
modules, and rejects a step whose water does not add up. The sums are formed
on the device and read once per check. `ledger.check_every: N` batches N
steps' checks (a violation is then reported at the check point and cannot be
rolled back). `ledger.speculate: N` (by default 31 days on a GPU) checks N
steps in one copy from an in-memory snapshot of the window's start: a failed
check, or a device flag that asks for a redo, puts the run back there and runs
the steps again checked one by one, so the results are those of checking every
step. See [Water accounting](WATER_ACCOUNTING.md).

## Checkpoints

`CheckpointManager` stores every connector's state, the clock, the exchange
buffers and the ledger, so a resumed run continues where the checkpoint was
written (`save_checkpoint`). Connectors that support a snapshot (`snapshot()`,
else `save_state()` copied on the device) can also be restored in memory.

## GPU

Connectors declare their backends (`cpu`, `cuda`); the backend manager gives
each the run's device or refuses (`run.backend_fallback: cpu` runs a CPU-only
connector on the CPU instead) and records what ran each interval (VIC reports
`cuda cuda-graphs` when its window ran on captured CUDA graphs). VIC captures
its own graphs inside a window; the connector's compute is one `simulate` call
and is not itself captured (`graph_safe = False`). A connector whose step is
graph-safe (fixed shapes, no host decisions, as the `StepRouter`) declares
`graph_safe = True`.

## Source files

| file | what |
|---|---|
| `torchvic/coupling/spec.py` | VarSpec, GridSpec, TimeSpec, ConnectorInfo, units |
| `torchvic/coupling/connector.py` | Connector, Context, Proposal, Balance |
| `torchvic/coupling/vic.py` | VicConnector, CaseDriver, KwargsDriver |
| `torchvic/coupling/toy.py` | TwoStoreConnector |
| `torchvic/coupling/ledger.py` | Ledger, tolerances |
| `torchvic/coupling/managers.py` | time, state, mapping, buffers, backends, outputs, checkpoints |
| `torchvic/coupling/graph.py` | DependencyGraph |
| `torchvic/coupling/schema.py` | configuration |
| `torchvic/coupling/registry.py` | registry |
| `torchvic/coupling/runtime.py` | Runtime, TorchVIC, validate_config |
| `torchvic/coupling/connectors/` | the other connectors |
| `configs/` | example configurations |
| `tests/test_coupling_*.py` | framework, conservation, VIC-through-runtime equals direct run, connectors |
