# Flood connector

The flood connector (`FloodConnector`, registry name `flood`,
`torchvic/coupling/connectors/flood.py`) turns VIC runoff and routed river water
into water on a two-dimensional hydraulic grid. In one-way mode the hydraulic
results are outputs only; in two-way mode the connector also takes river water
from routing, returns the water that leaves the grid to routing and can pass
floodplain infiltration to the VIC soil. Example configurations:
`configs/vic_flood.yaml` (one way) and `configs/full_coupled_system_2way.yaml`
(two way). Tests: `tests/test_coupling_flood.py` (connector, both backends),
`tests/test_coupling_flood_2way.py` (two-way mode), `tests/test_flood_gpu.py`
(engine, DEM conditioning, comparison with LISFLOOD-FP),
`tests/test_flood_multigpu.py` (multi-GPU engine) and
`tests/test_flood_lisflood_run.py` (LISFLOOD-FP input conventions).

| backend | what |
|---|---|
| `engine` (default) | TorchVIC's own local-inertial solver, `torchvic/flood/` (`engine.py`, `dem.py`, `triton_step.py`, `multigpu.py`), in process on the CPU or a CUDA GPU, float64 by default; depth and face momentum persist between intervals. One-way and two-way. |
| `lisflood` | LISFLOOD-FP 8.x as an external process (GPL-2.0-only, never linked). One-way only; needs `LISFLOOD_BIN` (or `binary`). Selected by `backend: lisflood` or by giving `solver`. |

LISFLOOD-FP is the reference the engine is checked against (see
[Validation against LISFLOOD-FP](#validation-against-lisflood-fp)).

## Inputs and outputs

* Basin file `*.npz` built by `tools/flood/make_basin.py`: DEM (Copernicus
  GLO-30 resampled to a metric grid, 100 m by default), cell size and
  georeference. The tool also writes the VIC-to-hydraulic cell pairs
  (`*_vic_pairs.csv`). `configs/vic_flood.yaml` uses a basin of the Brahmaputra
  near Guwahati. `dem` and `dx` can be given instead of `basin`.
* Conditioning (`condition: auto`): depressions deeper than `fill_min_depth`
  (5 m) are filled and the rivers of the flow-direction file `fdir` with an
  upstream area of at least `burn_min_area_km2` (5000 km2) are burnt in; a deep
  pit would otherwise fill with water and set the global CFL step. A DEM given
  as `dem` is used as given.
* Manning n: one value (`fpfric`, default 0.03). A raster is not passed through
  the connector.
* Runoff: `vic.total_runoff` (mm, mapped to hydraulic cells in m3; each
  hydraulic cell belongs to the VIC cell containing its centre). Upstream water
  enters at `inflow_points` (`[row, col]` cells), as `inflow` (m3 per point) or
  from routing nodes (`upstream_discharge`, `inflow_nodes`, `inflow_map`). Edges
  are closed unless listed in `free_edges` (`[edge, slope]`), where water leaves
  at normal depth.
* Outputs: states `depth`, `max_depth`, `speed`; flux `outflow` (m3, through open
  edges; in two-way mode outside every return outlet); series of stored volume,
  handed volume, outflow, flooded area (depth above `flood_threshold`, default
  0.01 m) and maximum depth. In one-way mode nothing is returned to VIC.
* Interval: `interval_s` (default 86400 s).

## Hydraulic scheme

Local-inertial (ACC) equations on a regular grid with Manning friction, as in
LISFLOOD-FP ACC (Bates, Horritt and Fewtrell, 2010, J. Hydrol. 387; de Almeida
et al., 2012, Water Resour. Res. 48). The engine has three schemes (engine
argument `scheme`): `torchvic` (the default, used by the connector;
mass-conservative: a cell never gives more water than it holds), `lisflood_gpu`
and `lisflood_cpu` (compatibility schemes that reproduce the respective
LISFLOOD-FP limiter behaviour). The CUDA fast path is Triton (engine argument
`backend='triton'`; it needs CUDA, scheme `torchvic` and theta = 1, and is
chosen by default when these hold); the portable path is plain torch. The
sub-step is one global CFL step set by the deepest water on the grid.

The float64 Triton step (`torchvic/flood/triton_step.py`) has five launches per
sub-step, all in one CUDA graph per block of sub-steps: `k_faces_mark` (wet test
of every interior face; dry faces get Q = 0, wet faces go to a list; open-edge
faces), `k_faces_wet` (the listed faces only: the float64 power and divisions
run on wet faces), `k_limiter2` (donor factors), `k_update2` (continuity,
momentum, interval maximum, in place) and `k_clock2` (ledger, clock, next CFL
step). It is bit-identical to the reference five-kernel step (engine argument
`triton_cfg={'fused': False}`): every face and cell gets the same expression on
the same numbers, `k_update2` is compiled with FMA contraction off, and the
contractions the compiler made in the reference kernel are written out as `fma`
calls. Further variants are kept as options in `triton_cfg` (`seg`, `quiet`,
`wfuse`, `fclock`, `pms`, `wflag`, `qflag`, `mskip`); all are bit-identical to
the default.

### Precision options

| `dtype` (connector) | what | water balance | use |
|---|---|---|---|
| `float64` (default) | everything float64 | closes to float64 round-off | accounting; the default |
| `mixed` | float32 face discharges, momentum and donor factors; depth as a float32 pair (about 48 bits), continuity in pair arithmetic; ledger and clock float64 | closes to float64 round-off | speed, mainly on cards with slow float64; Triton only |
| `float32` | everything float32 | limited by float32 precision | not for accounting |

In `mixed`, the face takes the depth and bed differences from float32 pairs
(y0 - y1 = dz + (h0 - h1), hflow = max(h0 + min(dz, 0), h1 - max(dz, 0)), dz from
the float64 DEM), so a deep, nearly flat water surface keeps its slope and a lake
at rest stays at rest; the power h^(10/3) is h^3 cbrt(h). The depth of `mixed`
differs from the float64 depth by the float32 rounding of the face discharges.

## Water accounting

Per interval, in m3: `R = I_ext - Q_out - dS` with `I_ext` = runoff + inflow
handed over, `Q_out` = open-edge outflow and `dS` = change of depth x cell area,
summed over the grid. The ledger also reports `clamp`, the water created when a
negative depth is reset (0 with the wet/dry limiter). `R` closes to float64
round-off for `float64` and `mixed` (`tests/test_flood_gpu.py`). In two-way mode
`I_ext` also includes `infiltration_unmet`, `Q_out` includes the returned water
and the infiltration, and evaporation is `E_other`
([Water accounting](../WATER_ACCOUNTING.md)).

## Validation against LISFLOOD-FP

LISFLOOD-FP `acc_gpu` and `acc_cpu` are the reference for the engine. When a
LISFLOOD-FP executable is available, `tests/test_flood_gpu.py` runs the engine
schemes `lisflood_gpu` and `lisflood_cpu` and LISFLOOD-FP on idealised cases
(`flat`, `slope` and `basin`, defined in `tools/flood/validate_engine.py`) and
asserts that the depths agree to better than 1e-9 m. Where the wet/dry limiter
does not act, the `torchvic` scheme and `lisflood_cpu` use the same arithmetic.

Flood results are not identical between CPU and GPU: round-off is amplified at
wet/dry fronts by the wet/dry thresholds. In the tests the CPU and GPU engines
agree to round-off on small cases.

## The LISFLOOD-FP backend

LISFLOOD-FP 8.2 is distributed under GPL-2.0-only (Zenodo record). TorchVIC is
GPL-3.0-or-later, so LISFLOOD-FP is used only as a separate process exchanging
files, never linked. See [License and credits](../downloads/code.md#license-and-credits)
and `THIRD_PARTY_NOTICES.md`.

LISFLOOD-FP ACC runs once per coupling interval (default one day), on the GPU
(`acc_gpu`) or the CPU (`acc_cpu`).

| item | rule |
|---|---|
| Inputs | `runoff`: m3 per hydraulic cell over the interval. `inflow`: m3 per boundary point (routing discharge at the domain edge, optional). |
| Runoff into LISFLOOD-FP | dynamic rainfall file, double precision, depth in mm per level. The first `initial_tstep` seconds carry no water: the CUDA solver skips the rain of its first step (exactly rate x `initial_tstep`), so the volume is carried by the following level and both solvers receive the same total. |
| Inflow into LISFLOOD-FP | `P x y QFIX q` with q = (V / dt) / dx in m2/s per unit width; x, y from (row, col): x = (col + 0.5) dx, y = (ny - 1 - row + 0.5) dx. No edge inflow conditions are written. |
| Ledger | I_ext = runoff + inflow; dS = change of the raster sum x dx^2; Q_out = 0 for closed edges (the balance is then a real check), else I_ext - dS because the CUDA `.mass` Qout is 0 even though water leaves. On the CPU solver the derived outflow is cross-checked against the `.mass` record (0.5 %), and every interval the `.mass` Qin is compared with the handed inflow. The tolerance uses eps = 1e-12 (14-decimal text rasters and thousands of solver steps). |
| Restart | state = depth raster, clock, running totals; momentum is re-initialised at every interval start, so a resumed run equals the uninterrupted one bit for bit. |

## Two-way coupling

Engine backend only. Every two-way feature is off by default; without them the
connector declares exactly the one-way inputs and outputs. Switching them on does
not change the hydraulics: with a return outlet and no losses, depth, momentum
and stored volume equal those of the one-way connector bit for bit.

### Which module accounts for the water

Each volume of water is accounted for by exactly one module at every instant:

| water | accounted by |
|---|---|
| soil, snow, canopy of every VIC cell | VIC |
| runoff + baseflow of a VIC cell outside the box | routing |
| runoff + baseflow of a VIC cell inside the box | flood (the `split` connector sends it to one of the two, never both) |
| river water upstream of an entry site | routing |
| river water handed over at an entry site | flood, from the moment of the hand-over |
| water on the hydraulic grid (channel and floodplain) | flood |
| water leaving the grid through a return outlet | routing, from the next routing step (injected at the outlet's site) |
| water leaving through an open face outside every return outlet | no module: it leaves the coupled system (`outflow`) |
| floodplain infiltration | VIC soil (through `soil_add`, next VIC interval) |
| the part of it VIC cannot hold (`soil_add_unmet`, cap mode) | flood again (added to the same VIC cell's hydraulic cells) |
| open-water evaporation | the atmosphere |

### Channel-floodplain exchange: hand-over, not double routing

The reach inside the box is bypassed in routing by a hand-over. At each entry
site (a routing site at the cell just upstream of the box) the flood
connector asks for everything (`river_request`, a state, lag 1); the router
delivers what passes the site (its `withdraw` service, capped so no ring slot
goes negative) as `routing.withdrawn` -> `flood.inflow` (flux, lag 0). The
router's downstream rings lose exactly that water, so it is not also routed
through the box. What the cap leaves in the river continues in routing and is
counted once.

Return: the faces of the open edges are grouped into return outlets
(`return_outlets`: `[edge]` for the whole edge, or `[edge, i0, i1]` for a face
range). Each outlet's outflow over the interval, accumulated per face in
float64, is exported as `returned` (flux, m3) -> `routing.injection` (lag 1)
and injected at the site's cell.

Ledger lines: routing Q_out includes the hand-over, flood I_ext includes it
(`inflow`); flood Q_out includes `returned` (+ lost outflow + infiltration),
routing I_ext includes it (`injection`). The runtime's coupled check (sent =
delivered + still in a buffer) closes each transfer.

### Floodplain to VIC soil (optional, off by default)

`infiltration_mm_day` (> 0 switches it on) and `open_water_evap_mm_day` act at
the interval end on cells deeper than `flood_threshold`: evaporation
e = min(h, E dt) first, then infiltration i = min(h - e, I dt), so depth never
goes negative. Infiltration is summed per VIC cell (`vic_map`) and exported as
`infiltration` (flux, m3) -> `vic.soil_add` (lag 1), into layer 0, as
irrigation does. With `soil_add_mode: cap` VIC adds what fits and exports the
rest as `soil_add_unmet` -> `flood.infiltration_unmet` (lag 0), which is put
back on the same VIC cell's hydraulic cells as a source of the next interval.
Evaporation leaves as `E_other` of the flood balance; VIC's own ET is not
changed. Not available with FROZEN_SOIL or snow bands (`soil_add` refuses
them).

Together with irrigation (`configs/full_coupled_system_2way.yaml`) both write
VIC's soil on the same cells, so the VIC connector takes them as two named
producers (`soil_add_sources: [irrigation, flood]`): `irrigation.soil_add ->
vic.soil_add_irrigation`, `flood.infiltration -> vic.soil_add_flood`. VIC adds
their sum once and splits the unmet part back in proportion to each producer's
addition (`vic.soil_add_unmet_irrigation -> irrigation.soil_unmet`,
`vic.soil_add_unmet_flood -> flood.infiltration_unmet`).

### Exchange frequency and lags

Daily exchange, no sub-daily iteration. Order in a day: VIC -> split ->
routing -> flood.

| connection | lag | why |
|---|---|---|
| routing.withdrawn -> flood.inflow | 0 | the box receives the day's river water the same day |
| flood.returned -> routing.injection | 1 | the cycle routing -> flood -> routing needs one lagged edge; the router's step is one day |
| flood.river_request -> routing.withdrawal_request | 1 | constant "everything"; lagged because routing computes first, so the first interval hands nothing over |
| flood.infiltration -> vic.soil_add | 1 | VIC's day is computed before flood; the water enters VIC's soil at the start of the next day |
| vic.soil_add_unmet -> flood.infiltration_unmet | 0 | VIC runs before flood on the next day |

### Combined balance

Over VIC + routing + flood (+ buffers in transit), in m3:

    P + I_ext(boundary) - ET - Q_out(outlets, lost edge outflow)
        - E_other(unrouted cells, open-water evaporation) - dS - closure = R

with dS = VIC storage + routing ring + flood depth + buffered transfers. Tested
at float64 rounding (`tests/test_coupling_flood_2way.py`).

## Multi-GPU engine (opt-in)

`FloodEngine(..., devices=['cuda:0', 'cuda:1'])`, or in a configuration
`flood: {devices: [cuda:0, cuda:1]}` (the first device is the run's device).
One GPU is the default. Float64, scheme `torchvic`, Triton path only (mixed and
float32 are refused). Code: `torchvic/flood/multigpu.py`.

| item | rule |
|---|---|
| decomposition | K equal row strips; each GPU steps its rows plus 2 halo rows per inner side with the unchanged single-GPU kernels; the frame edges are closed |
| halo | after every sub-step each GPU writes the depth and qx rows and the qy faces of its own rows into its neighbours' halo, into peer memory (default, where the GPUs allow it) or through a pinned host buffer (`triton_cfg` option `copy: host`) |
| faces | every face is evaluated with the single-GPU expression by the GPU owning its row; nothing is counted twice |
| dt | each GPU's own-row maximum depth goes to every GPU, and every GPU takes the maximum over all of them, so all take the same dt |
| ledger | boundary-face arrays are summed in the single-GPU order; the limiter clamp is the per-GPU sums added in GPU order |
| coupling, restart | sources are scattered from the primary (first) device at the interval start; own rows of the state are gathered at its end, so connectors, two-way exchange and `state_dict` see one engine on the primary |
| graph | a block of sub-steps is one CUDA graph over all GPUs; barriers are flag-based (`triton_cfg` option `sync: flag`, the default with peer copies on distinct GPUs) or CUDA events (`sync: event`) |

`tests/test_flood_multigpu.py` checks that two and three strips give the same
states, per-face edge outflow and ledger as the single-GPU engine bit for bit,
that a restart from `state_dict` equals the continuous run, and that the two-way
coupled system gives the same results with several strips.

## Limitations

* One hydraulic box per run; no multi-basin flood domain.
* The DEM is a surface model, not bare-earth terrain, and has no channel
  bathymetry, so depths in channels are qualitative; one Manning n.
* The sub-step is one global CFL step set by the deepest water on the grid, so
  deep water in one cell slows the whole grid.
* The `lisflood` backend is one-way only and needs `LISFLOOD_BIN` (or `binary`).
  Two-way coupling needs the `engine` backend.
* For the limits that apply to every connector see
  [Limits of the coupled system](../coupled/overview.md#limits).
* DEM and data licences: see [License and credits](../downloads/code.md#license-and-credits).
