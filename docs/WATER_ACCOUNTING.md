# Water accounting

The runtime keeps a ledger (`torchvic/coupling/ledger.py`) of every module's
water balance and of every transfer between modules, per exchange interval,
and refuses to commit a step whose water does not add up.

## Module balance

Over an interval, in m3 over the module's domain (and in mm per element):

    R = P + I_ext - ET - Q_out - E_other - dS - closure

| term | meaning |
|---|---|
| P | precipitation from outside the coupled system |
| I_ext | water another module or the boundary added (delivered inputs) |
| ET | evaporation, transpiration, sublimation |
| Q_out | water that left: to another module (a flux export) or out of the domain |
| E_other | other losses |
| dS | the module's storage change |
| closure | the model's own reported balance error, declared so the ledger does not call a model's known residual a coupling error |

### VIC

From VIC's own per-step outputs over the interval (mm per cell; volumes with
the domain's cell area x land fraction):

    P      = sum(OUT_RAINF + OUT_SNOWF) - precip_add
    I_ext  = precip_add + soil_add (+ soil_add_unmet)
    ET     = sum(OUT_EVAP)                    (incl. sublimation, put_data.c:626)
    Q_out  = sum(OUT_RUNOFF + OUT_BASEFLOW) (+ soil_add_unmet)
    dS     = sum(OUT_DELSOILMOIST + OUT_DELSWE + OUT_DELINTERCEPT) + soil_add
    closure= sum(OUT_WATER_ERROR)             (put_data.c:531-544)

So VIC alone closes to VIC's own closure: R is VIC's water error recomputed
from its reported fluxes and storage changes, minus VIC's reported error,
and is zero to rounding. VIC's OUT_WATER_ERROR is not zero everywhere (VIC
withholds evaporation from a layer at residual moisture and still reports it,
runoff.c:108-133); the ledger reports it as `closure` and does not hide it.

Between intervals the VIC connector also checks continuity: an interval must
open on the storage the previous one closed on (VIC's report totals carried
in its state) plus any `soil_add`; a state that was not carried shows here.

Routing inside `basin.simulate` (the unit-hydrograph discharge) is a
diagnostic of the runoff that left the cells, not a store of the cell, as in
VIC; runoff and baseflow leave VIC's balance in Q_out.

The lake store is not in the ledger: with LAKES TRUE the VIC balance check is
switched off for the run, and the log says so.

### Flood (two-way)

From the engine's float64 accumulators over the interval (m3, whole
hydraulic grid; [flood connector](connectors/flood.md#two-way-coupling)):

    I_ext  = runoff + river inflow handed over + infiltration_unmet
    Q_out  = returned (per outlet, to routing) + lost edge outflow
             + infiltration (to VIC soil_add)
    E_other= open-water evaporation
    dS     = change of sum(depth) x dx^2
    closure= -clamp (water a negative-depth reset created; 0 normally)

Routing counts the hand-over as a withdrawal (Q_out) and the return as an
injection (I_ext); VIC counts the infiltration as soil_add (I_ext, dS) and
the unmet part as Q_out. All three transfers cancel in the coupled balance.

## Coupled balance

Transfers between modules cancel. What a producer sends into a flux
connection (its Q_out) is either delivered (the consumer's I_ext, measured
after spatial mapping, at the consumer) or still in transit in the
connection's buffer:

    R_coupled = sum_m R_m + sent - delivered - dS_buffers

with dS_buffers from the buffers' own contents before mapping. A
non-conservative mapping, a buffer that loses an entry, or a module that
counts a transfer differently from its partner does not close. Every ledger
row records source, destination, quantity, amount (m3), units, location
(grid), and the interval.

## Tolerances

From the precision of what was summed: |R| <= factor x eps x n_terms x
sum|terms|, eps the dtype's machine epsilon (2.2e-16 for float64), n_terms
the number of values summed (rounding of a sum of n values grows at most
like n eps), factor 4 by default (`ledger.factor`). The same rule applies
per element. For VIC the magnitudes include the storage totals (soil, pack,
canopy) at every step: the DEL terms and the water error are differences of
those totals, so their rounding is relative to the storage, not to the small
change. A violation rejects the step (`ledger.on_violation: reject`) or
is recorded (`warn`).

## Cost

The ledger's sums are formed on the device; the host reads one small vector
per check.

## Outputs

`output.ledger: true` writes `ledger.csv` (one row per module and interval)
and `run_report.json` (backends that ran, timing of compute / validate /
ledger / commit, totals per module, warnings) to `output.dir`.
