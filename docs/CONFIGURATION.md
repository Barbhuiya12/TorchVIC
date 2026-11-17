# Coupled-run configuration

One YAML (or JSON) file describes a coupled run: the modules, the connections
between them, the ledger, checkpoints and outputs. The coupled system is
introduced in the [overview](coupled/overview.md). Check it without running:

    torchvic validate --config configs/vic_only.yaml          # schema, registry, graph, units
    torchvic validate --config configs/vic_only.yaml --deep   # also reads the data: grids, mappings, span

Run it:

    torchvic run --config configs/vic_only.yaml [--device cuda] [--out DIR]

or from Python:

    from torchvic import TorchVIC
    m = TorchVIC.from_config('configs/vic_only.yaml')
    m.initialize(); m.run(); m.finalize()

## Sections

```yaml
version: 1
run:
  device: auto            # auto | cpu | cuda | cuda:N
  dtype: float64          # float64 | float32
  backend_fallback: error # error | cpu: a connector without the run's backend
  duration_days: 0        # the run's length when no connector defines it
  log: true
  keep_outputs_on: device # device | host | none: where the kept output series live
modules:                  # name -> connector options
  vic:
    connector: vic        # registry name (default: the module name)
    enabled: true
    global_param: path/global_param.txt   # VIC 5 image-driver global file
    core: 5.1.0           # 5.1.0 | 5.0.1 (absent: 5.1.0)
    window_days: 31       # 0: the whole run in one call
    exchange: window      # window | whole | day | step
    day_session: true     # exchange day, a streamed case on a GPU: the days
    session_days: 62      #   of a block in one simulate call, handed out a
                          #   day at a time (the same bits; false: one call
                          #   a day)
    steps: 0              # case.load's options: steps, first_step, daily,
    first_step: 0         #   no_energy, sample_data
    soil_add_mode: refuse # refuse | cap | over (see CONNECTOR_API.md)
    stream: auto          # auto | true | false: read the forcing a window at a time
    day_coupled: []       # modules run between VIC's days inside its window
                          # (each: interval 1 day; reads vic.soil_moisture at
                          # lag 1, feeds vic.soil_add / precip_add at lag 0)
    ledger: true          # the VIC balance in the ledger
  routing:                # see connectors/river.md for the options
    connector: routing
    fdir: net/fdir.nc:fdir      # D8 flow directions, file:variable
    domain: net/domain.nc       # lat, lon, mask, area, frac
    gauges: net/gauges.csv      # lat, lng
    step_s: 86400
    reservoirs: reservoir       # a module that shares the routing step
    quality: quality            # may be combined with reservoirs
  irrigation:
    enabled: false        # a listed module that is switched off: not run
connections:              # producer output -> consumer input
  - from: a.outflow
    to: b.inflow
    lag: 0                # in consumer intervals
    mapping: a_to_b       # when the grids differ
    iterate: {max_iter: 20, tol: 1.0e-9}   # only inside a lag-0 cycle
mappings:
  a_to_b:
    source: grid_a
    target: grid_b
    pairs: [[0, 0, 1.0], [1, 0, 0.4], [1, 1, 0.6]]  # source, target, fraction
    # or file: weights.csv with columns source,target,fraction
    rest_to: 0            # optional: target index that takes, whole, every source element the pairs leave out
ledger:
  enabled: true
  factor: 4               # tolerance = factor * eps * n_terms * sum |terms|
  on_violation: reject    # reject (the step is not committed) | warn
  check_every: 1          # N > 1: batch N steps' checks (no rollback)
  speculate: auto         # N steps checked in one host copy; a failed check
                          # rolls them back and runs them again checked one
                          # by one (the results of checking every step);
                          # auto: 31 days on a GPU, off on the CPU; 0: off
checkpoint:
  path: null              # default path of save_checkpoint
  every: 0                # write one every N steps
  restart_from: null      # start from this checkpoint
  w3_restart: false       # VIC-WUR exact: on restart keep only what VIC-WUR's state file holds
output:
  dir: results            # VIC history streams, ledger.csv, run_report.json
  history: true
  ledger: true
```

`${VAR}` in any string is replaced from the environment (unset: an error).
Relative `global_param`, `dir`, `path`, `restart_from` and `file` are taken
from the configuration file's directory.

## What is checked, and when

* **load** (`schema.load`): unknown keys, types, ranges, modules named by
  connections, mappings named by connections, files that must exist,
  environment variables; every problem is listed at once (`ConfigError`).
  A registry entry that is marked `planned` is refused with its reason.
* **construction** (`Runtime`): connector options, missing dependencies,
  lag-0 cycles without `iterate`, flux/state kind mismatch, unit mismatch,
  grid mismatch without a mapping, a flux output feeding two connections,
  two modules that write one store.
* **initialisation**: the data; grids and their sizes, mapping conservation
  (each source element's fractions sum to 1), depth <-> volume needing
  areas, connectors agreeing on the run's length, a fast producer feeding a
  slower consumer at lag 0, unavailable backends.

## Example configurations

| file | what it does |
|---|---|
| `configs/vic_only.yaml` | runnable; tested (tests/test_coupling_vic.py: its history equals `torchvic run`'s) |
| `configs/vic_routing.yaml` | VIC -> river routing; network files from the environment (`TORCHVIC_GLOBAL`, `_FDIR`, `_DOMAIN`, `_GAUGES`); validates, run on the ISC domain by `benchmarks/run_isc_torchvic.py --runtime --routing` |
| `configs/vic_reservoir_routing.yaml` | the same with reservoirs in the routing step (`TORCHVIC_DAMS`) |
| `configs/vic_quality_routing.yaml` | the same with water temperature and constituents on the router |
| `configs/vic_flood.yaml` | validates; VIC runoff into the flood connector (one way) on a basin file you build with `tools/flood/make_basin.py`. The default backend is TorchVIC's own GPU engine (float64, in process); `backend: lisflood` runs LISFLOOD-FP ACC as an external reference process (`LISFLOOD_BIN`). Two-way options (`hand_over`, `return_outlets`, `infiltration_mm_day`, `open_water_evap_mm_day`) are off by default. Tested: `tests/test_coupling_flood.py`, `test_coupling_flood_2way.py`, `test_flood_gpu.py` |
| `configs/vic_irrigation.yaml` | VIC (`exchange: day`, `soil_add_mode: cap`) + irrigation: `vic.soil_moisture`, `vic.soil_add_unmet` into irrigation, `irrigation.soil_add` back to `vic.soil_add` with `lag: 1`; tested in `tests/test_conn_agri.py` |
| `configs/vic_crop.yaml` | VIC + WOFOST crop (`vic.transp`, `vic.pet` into the crop; parameters `configs/crop_params_sbarley.json`); the irrigation connections are commented and enabled by the user |
| `configs/vic_groundwater_irrigation.yaml` | VIC + irrigation + groundwater: pumping requests to the aquifer, pumping back with `lag: 1`; uniform aquifer values set in the file (not regional data) |
| `configs/full_coupled_system.yaml`, `configs/full_coupled_system_2way.yaml` | VIC, routing, reservoirs, quality, irrigation, crop, groundwater and flood on the ISC domain (data from the environment: `TORCHVIC_GLOBAL`, `TORCHVIC_COMBO_DATA`, ...) |
| `configs/vic_groundwater.yaml` | VIC + groundwater (`vic.baseflow` into `groundwater.recharge`); routing connections shown as comments; tested in `tests/test_coupling_groundwater.py` |

Other module combinations are not provided as examples.
