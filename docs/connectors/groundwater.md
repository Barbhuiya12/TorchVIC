# Groundwater connector

The groundwater connector adds a MODFLOW 6-style aquifer to a coupled run: VIC's baseflow is the recharge, and the aquifer returns discharge to the rivers and serves pumping requests.

## What it does

The groundwater engine (`torchvic/groundwater/`) is a PyTorch implementation, for CPU and CUDA, of the MODFLOW 6 groundwater flow (GWF) equations; MODFLOW 6 is the validation reference and is not a dependency. The connector (`torchvic/coupling/connectors/groundwater.py`, registry name `groundwater`) wraps the engine for the coupling runtime. One VIC cell is one aquifer cell. A second profile, `vicwur3`, is the MODFLOW coupling of VIC-WUR 3.0 ([VIC-WUR](vicwur.md)); this page describes the default profile unless it says otherwise. See also [Limits of the coupled system](../coupled/overview.md#limits).

## When to use it

* Use it when VIC's baseflow should pass through an aquifer with its own storage before it reaches the rivers.
* Use it when irrigation pumping should draw on a groundwater store ([Irrigation](irrigation.md)).
* Use it when river-aquifer exchange is needed (RIV and DRN packages).
* Do not use it when the aquifer must feed back to VIC (capillary rise, water table in the soil column); mode G2 is not built.
* Do not use it for an aquifer grid finer than the VIC grid, or for several aquifer layers through the connector; it sets one unconfined layer.
* For the MODFLOW coupling of VIC-WUR 3.0, use the `vicwur3` profile and read [VIC-WUR](vicwur.md).

## Inputs

Connection inputs (see [Exchanges](#exchanges)):

| input | file or variable | unit | meaning |
|---|---|---|---|
| `recharge` | `vic.baseflow` (flux) | mm over the interval | VIC bottom-layer baseflow; applied as RCH at the rate mm / 1000 / dt (dt in days) |
| `pumping_request` | irrigation connector (state) | m3 | volume wanted per cell over the interval |
| `river_stage` | channel stage (state, optional) | m | channel level for RIV, lagged by one interval; without a connection the configured `river.stage` holds |

Configuration inputs:

| input | file or variable | unit | meaning |
|---|---|---|---|
| cell geometry | `global_param` (the global file of the VIC grid) | - | the VIC cells are the aquifer cells; the area is the VIC land area (`area x frac`) |
| `aquifer.k` | scalar, per-cell list or `{file, var, scale}` | m/d | hydraulic conductivity |
| `aquifer.sy` | same | none | specific yield |
| `aquifer.ss` | same | 1/m | specific storage |
| `aquifer.thickness` | same | m | aquifer thickness |
| `aquifer.top` | same | m | land surface elevation |
| `aquifer.depth_to_water` or `aquifer.head0` | same | m | initial water table depth or initial head |
| `interval_days` | scalar | days | aquifer step (default 1) |
| `river.cond` | same | m2/d | riverbed conductance; RIV cells where `cond` > 0 |
| `river.stage` | same | m | configured river stage (used when no `river_stage` is connected) |
| `river.bottom` | same | m | riverbed bottom |
| `drain.depth` | same | m | drain level below the surface (default 2 m) |
| `drain.cond` | same | m2/d per cell | drain conductance (default K sqrt(area)) |
| `evt.rate`, `evt.depth` (optional `evt.surface`) | same | m/d, m | groundwater ET rate and extinction depth |
| `pump_cap`, `afr` | scalar | none | capping factor (default 1) and well-reduction fraction (default 0.1) of [Pumping](#pumping) |
| `max_discrepancy_pct` | scalar | % | bound on the engine budget discrepancy checked by `validate` (default 1e-4 %) |

## Config block

```yaml
modules:
  vic:
    connector: vic
    global_param: path/to/global_param.txt
    core: 5.1.0
    window_days: 1            # daily exchange: daily recharge
  groundwater:
    connector: groundwater
    global_param: path/to/global_param.txt   # the VIC cells are the aquifer cells
    interval_days: 1
    aquifer:                  # scalar, per-cell list, or {file, var, scale}
      k: {file: path/to/aquifer.nc, var: k}
      sy: {file: path/to/aquifer.nc, var: sy}
      thickness: {file: path/to/aquifer.nc, var: thickness}
      depth_to_water: 5.0
    drain:
      depth: 2.0
    pumping: true
    afr: 0.1
connections:
  - {from: vic.baseflow, to: groundwater.recharge}
```

With a routing module, add `vic.runoff -> routing.inflow` and `groundwater.discharge -> routing.inflow`; see [Counting the baseflow once](#counting-the-baseflow-once). The example configuration is `configs/vic_groundwater.yaml`.

## Outputs

| output | unit | meaning |
|---|---|---|
| `discharge` | m3 | flux: aquifer to river, DRN outflow + RIV gains - RIV losses (negative where the river loses water) |
| `pumping` | m3 | flux: what the wells delivered |
| `pumping_unmet` | m3 | state: request - delivered, returned to the requester |
| `head` | m | state: aquifer head at the interval end |
| `water_table_depth` | m | state: depth to the water table at the interval end |

The engine's `step` also returns the budget rates in and out per package and the percent discrepancy. Per-interval ledger rows carry the `detail_*` terms listed under [Ledger](#ledger).

## Technical details

### Equations

The engine solves the control-volume finite-difference form of the MODFLOW 6 GWF equations. For cell n of
area A_n and saturated thickness b_n = min(h_n, top_n) - bot_n:

```text
S_n A_n dh_n/dt = sum_m C_nm (h_m - h_n) + Q_rch,n + Q_wel,n + Q_riv,n + Q_drn,n
```

The Q terms are flows into the aquifer.

* Conductance between neighbours (NPF: harmonic mean, cell-centred, `icelltype=1` unconfined):
  C_nm = 1/(1/C_n + 1/C_m), with C_n = K_n b_n w / (d/2) (w the width of the shared face, d the cell size
  normal to it). The saturated thickness follows the standard NPF formulation of MODFLOW 6 (no Newton).
* Storage (STO): S_n = Ss b_n when confined, S_n = Sy when unconfined and converting (`iconvert`). Backward
  Euler in time, as in MODFLOW 6.
* RCH: Q = R A, with R in m/d. WEL: Q is prescribed (negative = extraction) and is reduced when the cell
  would go dry (the MODFLOW 6 AUTO_FLOW_REDUCE option).
* RIV: Q = Cr (Hriv - h) if h > Rbot, else Cr (Hriv - Rbot). DRN: Q = Cd (Helev - h) only if h > Helev.
  Cr = K_bed L W / thickness_bed (riverbed conductivity, reach length and width, bed thickness). RIV and DRN
  exchange water with the rivers.
* The nonlinearities (dry cells, the RIV and DRN switches, the saturated thickness) are solved by Picard
  iteration.

One VIC cell is one aquifer cell, on the regular lat/lon grid of the VIC mask: A_n is the land area of the
VIC cell (the area VIC uses for runoff volumes), the stencil has five points, cells outside the mask are
inactive and the domain edges are no-flow. The engine supports several layers with vertical (Kv) leakage;
the connector sets one layer.

### Responsibility for each flux

Each flux has one responsible module, so no water is counted twice.

| Flux | From -> to | Unit in the engine | Notes |
|---|---|---|---|
| Recharge | VIC bottom-layer baseflow -> RCH | m/d over the cell (= mm/d / 1000) | cell mean of the VIC tile fluxes, area weighted |
| Pumping | irrigation connector -> WEL | m3/d, negative | capped by the water available in the cell; the unmet part returns to the requester |
| River exchange | channel stage -> RIV | m3/d, sign from the aquifer side | channel stage if connected, else the configured stage; needs the riverbed conductance |
| Groundwater discharge | DRN and RIV outflow -> routing | m3/d to the cell's channel | becomes baseflow to routing |

VIC's bottom-layer outflow (the ARNO baseflow in `runoff.c`) is the only VIC term that represents the
groundwater system, so it is the only flux handed to the aquifer. The connections of the configuration
select G0 or G1; there is no separate option:

* Mode G0 (groundwater off): VIC is unchanged and its baseflow goes to routing.
* Mode G1 (coupled, the default): the VIC baseflow is diverted from routing into RCH (same value and
  unit; the ledger books it as a transfer to the aquifer, not as runoff leaving the domain). The
  groundwater engine holds the aquifer storage and produces the discharge, which goes to routing as
  baseflow. VIC no longer routes its own baseflow, so the water is counted once. VIC's soil physics is
  untouched.
* Mode G2 (head feedback to VIC: capillary rise, water table in the soil column): not built, because it
  would change VIC's physics.

### Engine formulation

Code: `torchvic/groundwater/` (`grid.py`, `solver.py`, `model.py`).

The engine reproduces one MODFLOW 6 option set, matched term by term to the MODFLOW 6.7.0 source:

* NPF: standard formulation (no NEWTON), harmonic-mean conductance, `icelltype` 0 and 1, constant vertical
  conductance or VARIABLECV [DEWATERED].
* STO: Ss (default formulation or STORAGECOEFFICIENT) and Sy with `iconvert`.
* Packages: RCH (array input, highest active cell); WEL with optional AUTO_FLOW_REDUCE; RIV; DRN (no DDRN);
  GHB; CHD; EVT (array input, highest active cell, linear or PXDP/PETM segments, optional PETM0).
* Time: backward Euler and steady steps; Picard outer iterations with an IMS-style closure
  (`outer_dvclose`; inner `inner_dvclose` and `inner_rclose`, infinity norms).

Dry convertible cells are deactivated at every outer iteration and stay dry. Rewetting is not supported:
MODFLOW 6 rewets cells in node order within one pass, so a cell rewetted earlier in the loop can trigger its
neighbours, which a parallel solver would not reproduce exactly. PERCHED, the Newton formulation and
upstream weighting are not supported either.

### Engine grid

A structured `nlay x nrow x ncol` grid with per-cell area and per-face distances and widths (CL1, CL2,
width), so the same solver runs MODFLOW 6 DIS grids (`Grid.dis`) and a regular lat/lon grid with one VIC
cell per model cell (`Grid.latlon`: spherical distances, the VIC domain area as the cell area). MODFLOW 6
needs DISU for the lat/lon geometry; `Grid.disu_arrays` exports it.

### Engine solver

The 7-point stencil is stored as diagonals (no sparse matrix, no atomics) in float64. The linear solver is
PCG with a Jacobi (default) or a red-black symmetric Gauss-Seidel preconditioner.

The same algorithm runs on every device. An outer (Picard) iteration is drying, assembly and PCG start,
then `inner_per_outer` PCG iterations (the IMS inner closure is tested on the last one, and the iterate is
frozen once it closes), then the head update and the outer closure, all as device-side flags.
`outer_per_check` outer iterations form a chunk, and the host reads one flag per chunk (iterations after
convergence are masked no-ops). An outer iteration whose inner solve has not closed is never accepted, so
short inner solves trade outer for inner iterations without changing the converged answer. On CUDA the
functions are fused by `torch.compile` and a chunk is one CUDA graph; the budget is a second graph.

The defaults on CUDA are 3 PCG iterations per outer iteration, 4 outer iterations per host read, and 50 PCG
iterations per outer iteration for steady steps. A step that does not converge is retried with 8x longer
inner solves, which are kept for later steps; this covers stiff confined systems. List boundaries scatter
without atomics, so runs are deterministic, and graph and eager results are bitwise equal.

### Engine API

`GroundwaterModel(grid, k11, k33, icelltype, ss, sy, iconvert, strt, ...)` with the setters
`set_recharge(rate)`, `set_wells(cells, q, auto_flow_reduce)`, `set_rivers(cells, stage, cond, rbot)`,
`set_drains`, `set_ghb`, `set_chd` and `set_evt(surface, rate, depth, pxdp, petm, petm0)`. A setter called
with values only (cells omitted) updates them in place, without re-capturing the CUDA graph.

`step(dt, steady)` returns a `StepResult` with the budget rates in and out per package (`STO-SS`, `STO-SY`,
`RCH`, `WEL`, `RIV`, `DRN`, `GHB`, `EVT`, `CHD`), the flows per entry used by the ledger and the coupling
(for example the RIV and DRN discharge to routing), and the percent discrepancy. A step that fails to
converge raises `GroundwaterNotConverged` and the state is not committed. The restart state (`state_dict`,
`save`, `load`) is the heads, the wet/dry flags and the clock; a restarted run reproduces the continuous run
bit for bit.

### Recharge from VIC

VIC 5 has no recharge output (`OUT_RECHARGE` does not exist; only the lake variable `OUT_LAKE_RCHRG` does),
and its inter-layer drainage `Q12` is computed inside the step. The one flux VIC computes that leaves the soil
column downward is the ARNO baseflow taken from the bottom layer (`runoff.c:284-345`: `dt_baseflow` from the
relative moisture of the last layer with `Ds`, `Dsmax`, `Ws` and `c`, summed over frost areas into
`cell->baseflow`; output as `OUT_BASEFLOW`, `put_data.c:633`). It is exposed as the VIC connector's
`baseflow` (mm over the interval) and is the recharge.

It represents the drainage from the bottom soil layer into the saturated zone, which VIC does not model and
returns to the river as slow flow. It is not infiltration to the water table of a deeper aquifer with its own
physics, and it responds only to the moisture of the bottom layer; no head feedback goes back. VIC has no
`Q12` out of the last layer (the bottom layer loses water by baseflow only), so `baseflow` is the only
candidate, and no VIC code is changed.

### Counting the baseflow once

VIC alone sends runoff plus baseflow to routing (`total_runoff`). With groundwater the baseflow goes to the
aquifer and returns as discharge:

```text
vic.baseflow          -> groundwater.recharge
vic.runoff            -> routing.inflow           (not vic.total_runoff)
groundwater.discharge -> routing.inflow
```

The runtime refuses a flux that feeds two connections ([Coupling graph](../COUPLING_GRAPH.md)), but it
cannot see that `total_runoff` contains `baseflow`. `ownership_problems(config)` reports a configuration in
which the baseflow feeds the aquifer and `total_runoff` also feeds a consumer, or in which `recharge` is fed
by a flux other than `baseflow`; the rule and `configs/vic_groundwater.yaml` are tested in
`tests/test_coupling_groundwater.py`.

In the ledger, VIC's module balance is unchanged (runoff and baseflow are both `Q_out`). The baseflow is a
transfer between modules (sent by VIC, delivered to the aquifer's `I_ext`), so it drops out of the boundary outflow
of the coupled domain; the aquifer's discharge, pumping and groundwater ET carry the water on. Without
routing, `discharge` leaves the domain (boundary outflow).

### Exchanges

The area of an aquifer cell is the land area of the VIC cell (`area x frac`), so a depth over the VIC cell
and a volume into the aquifer are the same water. The lateral conductances follow from the lat/lon spacing,
so the domain must be a regular lat/lon grid.

| name | kind | meaning |
|---|---|---|
| in `recharge` | flux, mm | VIC `baseflow`; the rate mm / 1000 / dt (dt in days) is applied as RCH |
| in `pumping_request` | state, m3 | volume wanted per cell over the interval (a state, not water: the irrigation connector sends the demand and the aquifer returns what it delivered) |
| in `river_stage` | state, m | channel level for RIV, lagged by one interval; without a connection the configured `river.stage` holds |
| out `discharge` | flux, m3 | aquifer to river: DRN outflow + RIV gains - RIV losses (negative where the river loses water) |
| out `pumping` | flux, m3 | what the wells delivered |
| out `pumping_unmet` | state, m3 | request - delivered, returned to the requester |
| out `head`, `water_table_depth` | state, m | at the interval end |

### Pumping

The request is first capped at the water the cell holds above the level where wells start to reduce,
`pump_cap` x Sy x area x (head - (bottom + `afr` x thickness)) (defaults: `pump_cap` 1, `afr` 0.1). The
engine's well with the AUTO_FLOW_REDUCE option then reduces the rate smoothly as the head approaches the
bottom; the delivered volume is the engine's actual well flow. Lateral inflow and the recharge of the same
step are not added to the cap, so a cell is never asked for more than it stores. The undelivered part of the
request is returned to the requester as `pumping_unmet`.

### River-aquifer exchange

`river: {cond, stage, bottom}` places RIV cells where `cond` > 0. When `river_stage` is connected it replaces
`stage` at each interval (one-interval lag); otherwise the configured `stage` holds. DRN (`drain`) carries
the groundwater discharge where no river data exist; by default the drain level is 2 m below the surface and
the conductance is K sqrt(area).

### Groundwater evapotranspiration

`evt: {rate, depth[, surface]}` is the engine's EVT (linear to the extinction depth). It leaves the coupled
domain as `ET`.

### Time

The aquifer step is `interval_days` (default 1 day; K is in m/d). VIC runs ahead of the aquifer (lag 0): the
aquifer takes the water VIC computed for the interval, without iteration. With lag-0 connections the order
within an interval is VIC, the irrigation connector (pumping request), the groundwater step, then routing. A
VIC window total is spread uniformly over the aquifer's steps, so with VIC's default 31-day window the
recharge is flat within each window; use `window_days: 1` or `exchange: day` on the `vic` module for daily
recharge.

### Transaction and restart

`compute` runs the engine step and then puts the committed state back; `commit` makes the heads of the
proposal the accepted ones. A step that does not converge raises `ComputeError`, the runtime rejects the
step and nothing changes. `save_state` holds the heads, the wet/dry flags and the engine clock; a restarted
run equals the continuous run bit for bit (tested).

### Ledger

The module balance follows [Water accounting](../WATER_ACCOUNTING.md). Per interval, in m3: `I_ext` =
recharge + general-head inflow; `ET` = groundwater ET; `Q_out` = pumping + net discharge + general-head
outflow; `dS` = the engine's storage terms (STO-SS and STO-SY); `closure` = the engine's own budget
discrepancy (the solver's convergence error, as VIC declares its water error). The module row therefore
closes by construction. Three checks stand apart from it: `validate` bounds the discrepancy
(`max_discrepancy_pct`, default 1e-4 %); the tests compare `dS` with the storage change formed from the
heads (Sy x area x dh); and the coupled row (sent = delivered across VIC -> aquifer -> consumer) is
independent of both. Per-interval ledger rows carry the `detail_*` terms recharge, pumping, discharge,
river_in, river_out, ghb_in, ghb_out, pumping_request, pumping_capped and discrepancy_pct.

### Parameters

Every aquifer field is a scalar, a list with one value per cell, or `{file, var, scale}` (sampled at the cell
centres, nearest neighbour). The fields are `k` (m/d), `sy`, `ss`, `thickness` (m), `top` (m), and
`depth_to_water` or `head0`.

A field that is not given takes a default value: `k` 1 m/d, `sy` 0.1, `ss` 1e-5, `thickness` 50 m, `top` 0 m,
`depth_to_water` 5 m, and drains 2 m below the surface. These default values are not regional data. When none
of `k`, `sy` and `thickness` is given, the run reports the defaults in its log and says that they are not
data. The result is not a calibrated groundwater model; supply regional aquifer data for any application.
`configs/vic_groundwater.yaml` and the Stehekin tests use uniform aquifer values set in the file or the test,
because no regional aquifer data are supplied for those domains.

### Parameter sources for India

Not bundled; read each licence before use.

* `k` and porosity: GLHYMPS 2.0 (Huscroft et al., 2018).
* `thickness`: global depth to bedrock (Pelletier et al., 2016) and the CGWB aquifer atlas / India-WRIS.
* `sy`: CGWB / GEC-2015 norms by lithology.
* `head0` or `depth_to_water`: CGWB observation wells, or a spin-up from the recharge climatology.
* `top`: the elevation in the parameter file.
* River `cond` and `bottom`: channel width and length of the routing network, with an assumed bed K that
  has to be calibrated.
* Pumping: the Minor Irrigation Census and CGWB dynamic resources, through the irrigation connector.
* Data for evaluating results: CGWB water levels and GRACE storage.

### River-aquifer exchange in the routing sweep

In the `vicwur3` profile, in mode `fixed` with a daily period and a router in the VIC-WUR group, the
exchange of the river cells is solved inside the routing sweep (slot `river_exchange`; connector option
`river_exchange`, on by default in that mode). Each river cell is an implicit reach: the loss L to the
aquifer solves, by bisection, L = C (stage(Q - L) - head), where C is the riverbed conductance, `stage(.)`
the channel stage from the discharge and `head` the aquifer head, and the discharge Q is that of the same
day, taken from the water in the reach. A river cannot lose more than it holds; the shortfall is a debt
booked as unmet in the groundwater ledger (`river_exchange_unmet`) and paid in the following days.

The bisection has `exchange_iterations` steps (default 50); the result does not depend on the number beyond
2^-n of the bracket. On CUDA the solver is a Triton kernel replayed from a CUDA graph; on the CPU it runs in
torch. Using the discharge of the same day avoids the one-day lag between channel stage and exchange flux
that a daily exchange has otherwise, which can make the discharge oscillate with a period of two days. In
mode `exact` the behaviour of VIC-WUR 3.0 is kept: the forced inflow of a losing river is added whatever the
river holds, so the discharge can become negative. The slot is tested in `tests/test_river_exchange.py`; the
modes are described in [VIC-WUR](vicwur.md).

## Limitations

Unless stated otherwise, these are the limits of the default profile.

* Regional aquifer fields are supplied by the user; none are bundled (see
  [Parameter sources for India](#parameter-sources-for-india)). Without them the connector uses default
  values that are not regional data, and the result is not a calibrated groundwater model.
* The aquifer does not feed back to VIC (mode G2: capillary rise, water table in the soil column).
* The aquifer grid is the VIC grid (one cell per VIC cell, regular lat/lon); a finer aquifer grid is not
  supported. The engine supports several layers; the connector sets one unconfined layer.
* The engine does not support rewetting, PERCHED or the Newton formulation of MODFLOW 6. It supports GHB and
  CHD boundaries, which the connector does not set (its general-head ledger terms are zero).
* Recharge inside a VIC exchange window is flat (the window total is spread evenly); daily recharge needs
  `window_days: 1` or `exchange: day`.
* The `routing` connector exports no channel stage, so `river_stage` has no producer among the connectors
  and the configured `river.stage` holds. It adds a negative `inflow` as given: a negative `discharge` (a
  losing river) is not withdrawn from the channel or limited by the water in it. The `vicwur3` profile
  does this in the [routing sweep](#river-aquifer-exchange-in-the-routing-sweep).
* The water balance is checked for the whole aquifer in each interval, not cell by cell.

## Verification status

The engine is checked against MODFLOW 6 6.7.0 and against analytic solutions in the test suite; the connector is checked in its own tests. This is not part of the VIC verification on the Verification page.

* `tests/test_groundwater_mf6.py` compares the engine with MODFLOW 6 6.7.0 on four cases: the MODFLOW 6 examples `ex-gwf-twri01` (steady state, five layers) and `ex-gwf-fhb` (transient, confined), a two-layer case with every supported package and option (`mixed`), and a lat/lon case with one VIC cell per model cell (`latlon`, MODFLOW 6 DISU). Both solvers use the same IMS closure settings (`tests/groundwater_cases.py`); the test asserts a maximum head difference of 1e-7 (in the model's length unit) and a difference of 1e-9 of the step's inflow for each package budget term. It is skipped when the MODFLOW 6 reference results are not available.
* `tests/test_groundwater.py` checks:
    * analytic solutions (a confined line between two fixed heads, linear without and parabolic with recharge; the Dupuit solution for unconfined flow);
    * conservation in a closed domain with recharge, wells and a river, and the storage rates against the head change;
    * dry cells and AUTO_FLOW_REDUCE: a cell that goes dry stays dry, and a well is reduced rather than cut off;
    * restart (bit-identical to the continuous run) and a failed step (the state is unchanged);
    * agreement between the two preconditioners, between CPU and GPU, and between CUDA-graph and eager execution;
    * the lat/lon geometry and the DISU export.
* `tests/test_coupling_groundwater.py` checks ledger conservation, the baseflow rule, pumping limits, river exchange and ET, restart, failed steps, and VIC with groundwater on VIC's Stehekin sample.
* The `river_exchange` slot of the `vicwur3` profile is tested in `tests/test_river_exchange.py`.


## Next step

[VIC-WUR connector](vicwur.md) for the `vicwur3` profile and its river-exchange modes, or [Irrigation](irrigation.md) for the pumping requests.
