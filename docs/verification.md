# Verification

TorchVIC is verified against VIC itself: the question is whether it computes
what VIC computes, not whether either matches observations. The claim is that
TorchVIC agrees with compiled VIC to round-off. It is not a claim of bitwise
identity ([Differences from VIC](differences.md) explains why the bits differ).

The evidence has three levels:

* **Function by function.** Small C programs linked to the pristine VIC sources
  print the results of VIC's routines to 17 digits, and TorchVIC's functions
  are tested against them (`tests/oracle`, compiled on first use). They cover
  the soil, snow, canopy, energy-balance and routing routines.
* **Whole runs.** `vic_image.exe` of VIC 5.0.1 and 5.1.0 and TorchVIC run on the
  same files: Walnut Gulch, Stehekin and small windows of the
  Indian-subcontinent domain (`tests/`, marker `reference`) and, for the main
  evidence, the whole domain (`benchmarks/run_isc.sh` runs both programs on it
  and `benchmarks/compare_isc.py` compares the annual totals cell by cell).
* **The verification matrix** (`verification/`): small-window configurations
  covering the options, both releases and every output variable at every model
  step, in double precision.

All numbers on this page were measured with TorchVIC 0.4.7. With `--core 5.1.0`
(the default), the results of release 0.4.8 are bit-identical to those of 0.4.7.

## Method of the whole-domain comparison

The reference is the pristine `vic_image.exe` of VIC 5.1.0, compiled from the
released sources (commit `14a371a8dbee7ea834152f8ac65e2d6d54a9c30a`) with GCC
13.3.0 at `-O3`. Both programs read the same global parameter file, domain and
parameter files and 3-hourly forcing, simulate 2004 and 2005 (731 days) on all
112,825 cells, and write daily history in double precision. TorchVIC ran on an
NVIDIA H200 NVL in float64.

Every output variable that VIC writes in the configuration and that TorchVIC
also writes is compared: 85 variables in the water balance, 88 with the energy
balance and with frozen soil, and 106 with five elevation bands. The comparison
covers every active cell, every day, every soil layer and every elevation band;
no cell or day is left out and no subset is used. For each variable it records
the share of values that are bit-identical in double precision, the share that
are identical after both values are rounded to single precision (the precision
of VIC's own history files), the largest absolute difference over all cells
and days, and the differences of the annual totals (the sum over the days for
water fluxes, the mean for other variables). Domain means are weighted by cell
area times land fraction.

The domain is the Indian subcontinent (ISC): 112,825 land cells at 1/16 degree
on the VICGlobal grid ([Schaperow et al., 2021](about/references.md#data)),
covering the river basins of the subcontinent, the coasts and the cells of
political India and Bangladesh. It has 18 vegetation classes including bare
soil and five snow bands per cell, and spans the humid tropics, the monsoon
plains, a desert and the Himalaya, so that every branch of the model is active
somewhere. The daily forcing was built from ERA5
([Hersbach et al., 2020](about/references.md#data)) and the 3-hourly forcing
that VIC reads was produced from it by TorchVIC's MTCLIM disaggregation, so both
models see identical sub-daily forcing. The comparison therefore verifies the
land surface model, not the disaggregation, which is tested separately
([below](#daily-forcing-mtclim-against-metsim)).

| configuration | VIC release | description |
|---|---|---|
| water balance | 5.1.0 | `--core 5.1.0`, `FULL_ENERGY FALSE`, one band per cell |
| energy balance | 5.1.0 | `--core 5.1.0`, `FULL_ENERGY TRUE`, `QUICK_FLUX TRUE` |
| frozen soil | 5.1.0 | `--core 5.1.0`, `FROZEN_SOIL TRUE`, `QUICK_FLUX TRUE` |
| water balance, five bands | 5.1.0 | `--core 5.1.0`, `SNOW_BAND TRUE`, five bands per cell |
| water balance | 5.0.1 | `--core 5.0.1`, one band per cell |
| energy balance | 5.0.1 | `--core 5.0.1`, `QUICK_FLUX TRUE` |

## Whole domain

**Annual totals.** The table gives, for each configuration, the largest
difference in any cell of the annual value of 2005. The largest is 1.91 mm, the
annual mean soil moisture of the energy balance with VIC 5.1.0. The
domain-mean relative difference of the annual precipitation, evapotranspiration,
runoff, baseflow, snow water equivalent and soil moisture rounds to 0.0000 % in
all six configurations; precipitation, which is forcing, differs by 0.0000 mm.
The 99th percentile of the per-cell relative differences is below 0.007 % for
every variable listed. No cell differs by more than 1 % in any of the six
variables, except one cell with snow water equivalent in the frozen-soil
configuration and one in the energy balance with VIC 5.0.1.

| configuration | evapo-transpiration | runoff | baseflow | snow water equivalent | soil moisture | identical in float64 | identical in float32 |
|---|---|---|---|---|---|---|---|
| water balance, 5.1.0 | 0.897 mm | 0.281 mm | 0.260 mm | 0.070 mm | 0.519 mm | 71 % | 98.6 % |
| energy balance, 5.1.0 | 1.535 mm | 1.218 mm | 0.225 mm | 0.476 mm | 1.906 mm | 63 % | 97.6 % |
| frozen soil, 5.1.0 | 1.246 mm | 0.535 mm | 0.251 mm | 0.250 mm | 0.602 mm | 62 % | 97.6 % |
| five bands, 5.1.0 | 0.479 mm | 0.201 mm | 0.172 mm | 0.198 mm | 0.453 mm | 79 % | 99.0 % |
| water balance, 5.0.1 | 0.669 mm | 0.301 mm | 0.352 mm | 0.071 mm | 0.606 mm | 74 % | 98.7 % |
| energy balance, 5.0.1 | 1.112 mm | 0.711 mm | 0.274 mm | 0.447 mm | 0.862 mm | 65 % | 97.7 % |

**Bitwise agreement.** Between 62 and 79 % of all compared values are
bit-identical in float64, and between 97.6 and 99.0 % are identical after
rounding to single precision. The shares are weighted by value and include
forcing variables that are written back as outputs. Of the compared variables
15, 17, 13 and 16 are bit-identical in every cell and on every day in the four
configurations with VIC 5.1.0 (of 85, 88, 88 and 106). The variable with the
largest absolute difference is `OUT_AERO_RESIST1` in all configurations except
the one with five bands, where it is `OUT_SNOW_FLUX_BAND`.

**Distribution of the differences.** Pooled over all compared variables, cells
and days of the four configurations with VIC 5.1.0, 65 to 74 % of the values
have a difference of exactly zero (74 % water balance, 65 % energy balance, 65 %
frozen soil, 74 % five bands), 95 to 96 % have a difference of at most 1e-12 and
99.6 to 99.9 % of at most 1e-6, in the units of the variable.

**Water and energy closure.** In VIC 5.1.0 the water balance
P - E - R - B - dS is not closed in some cells: the largest daily residual in a
cell is 0.88 mm (0.79 mm with five bands), and 7,287 to 7,290 cells exceed
0.01 mm on some day. TorchVIC has the same non-closure in the same cells and
does not add to it. The energy error is a different case: the value TorchVIC
reports in `OUT_ENERGY_ERROR` follows its own definition and is not comparable
with VIC's ([Differences from VIC](differences.md#out_energy_error)). When the
energy balance is recomputed from the output fluxes by one formula for both
programs, the largest error is 111.99 W m-2 for TorchVIC and 111.98 W m-2 for
VIC with the energy balance, and 185.5 W m-2 for both with frozen soil.

**Ten years.** A ten-year water-balance run (2000 to 2009, whole domain, H200)
compared with VIC 5.1.0 in 53 variables has 97.7 % of the values identical in
single precision and 67.9 % bit-identical in float64. The annual domain mean of
evapotranspiration, runoff, baseflow and soil moisture differs by at most
3e-5 % in each year, with the exception of the diagnostics `OUT_WATER_ERROR`
(up to 1.6e-2 %) and `OUT_SUB_CANOP` (6.6e-3 %). The largest daily difference
in a single cell is larger than the annual figures suggest: 0.52 to 9.84 mm in
soil moisture (9.84 mm in 2006), up to 1.05 mm in evapotranspiration, 5.54 mm in
runoff and 15.1 mm in snow water equivalent (both in 2006). On the last day of a
year, 1,539 to 2,567 cells have a soil-moisture difference above 1e-6 mm; a small
number of cells carry a state offset that persists and does not spread to other
cells.

## Streamflow

Runoff and baseflow of TorchVIC and of VIC 5.1.0 were routed by the same routing
code (the routing connector, with the same unit hydrographs and flow
directions) for 2000 to 2009, and the daily discharge was compared at four
snow-fed gauges and one rain-fed control.

| gauge | mean discharge in VIC | largest daily difference |
|---|---|---|
| Beas at Pandoh | 214 m3 s-1 | 0.0052 m3 s-1 |
| Sutlej at Bhakra | 570 m3 s-1 | 0.0036 m3 s-1 |
| Bhagirathi at Devprayag | 412 m3 s-1 | 0.0135 m3 s-1 |
| Ganga at Rishikesh | 436 m3 s-1 | 0.0183 m3 s-1 |
| Mahanadi at Seorinarayan (rain-fed control) | 465 m3 s-1 | 3e-9 m3 s-1 |

The Nash-Sutcliffe efficiency is at least 0.99999999998 at every gauge, the
largest relative daily difference is 4.5e-4 (Ganga), and the mean annual volume
differs by at most 5.3e-5 %. The gauges are nominal points snapped to the cell
of largest upstream area within 0.2 degree; the snapped areas are 1.3 times
(Beas) and 2.8 times (Bhagirathi) the nominal catchment areas, so the gauge
names are approximate. The result, that the differences are negligible, does
not depend on this.

## Restart from VIC state files

A run of 2005 started from the VIC state file of 31 December 2004 (whole domain,
water balance) was compared with the continuous run of 2004 and 2005. At day 0
the largest per-cell difference in soil moisture between the TorchVIC restart and
the VIC restart is 0.007 mm, while VIC's own restart differs from VIC's
continuous run by 0.063 mm: TorchVIC's restart matches VIC's restart as closely
as VIC's restart matches VIC. A native split of a run (1,000 cells, GPU)
reproduces the continuous run in all of 270,000 values, and on the CPU a restart
from the VIC file differs from VIC's restart by at most 1.4e-10 mm over 90
days. On a GPU, reading a VIC state file has a known problem and a workaround
([Known limitations](about/limitations.md)).

## Daily forcing: MTCLIM against MetSim

The MTCLIM routines on the device were compared with MetSim 2.0.1 for 2004 on
five windows of 6 x 6 cells with a 3-hourly step. The first day is dropped
because MetSim extrapolates the vapour pressure on it. The largest differences
over the windows are 0.030 degC in air temperature, 0.103 W m-2 in longwave
radiation, 0.273 Pa in vapour pressure and 4.49 Pa in pressure; those of
shortwave radiation (1.5e-6 W m-2), wind (0) and precipitation (4e-15 mm) are of
round-off size. Between 87 and 99 % of the values of air temperature, longwave
radiation, vapour pressure and pressure are identical in float32, and 99.998 to
100 % of the others. The comparison is with MetSim, not with VIC 4.2.d, and the guard
against `Tmin > Tmax` in `calc_tair` was not examined.

## Paths not compared over the whole domain

Five option sets that the whole-domain configurations do not exercise were run
for 2004 and 2005 on two windows (Himalaya and forest) against VIC 5.1.0:
`QUICK_FLUX FALSE`, a finite-difference soil profile with 12 nodes (explicit and
with `IMPLICIT`), and a daily step with eight snow sub-steps in the
energy-balance and the water-balance mode. The five rows of the forest window
pass and the five rows of the Himalaya window warn; none fails. On the CPU
TorchVIC is slower than VIC on these windows (for example 2,400 s against 157 s
for `QUICK_FLUX FALSE` in the Himalaya window). Lakes (`LAKES TRUE`) were not
run on this domain, because it has no lake parameters.

## The verification matrix

The matrix has 95 configurations of eight small cases: three of VIC's own
image-driver samples (Walnut Gulch, 15 cells; Stehekin, 16 cells; Findley Lake,
one cell with lakes off) and five windows of 6 x 6 cells cut from the
Indian-subcontinent domain to cover its climates (the Indo-Gangetic plain, the
Thar desert, a Western Ghats forest, the southern tip of India and the
Himalaya at Gangotri, 3,450 to 5,600 m with seasonal snow). Each row is one case,
a physics mode (water balance, energy balance, frozen soil), a VIC release, a
model step and optional global-file options, run by the pristine
`vic_image.exe` and by TorchVIC on the CPU.

A variable's score is the larger of the relative difference of its area-weighted
total or mean and its largest per-cell relative difference. It **passes** at 1e-4
(0.01 %) or less and **warns** at 1e-2 (1 %) or less; above that it fails. The
closure test is TorchVIC's residual P - E - R - B - dS, which passes at 1e-10 mm
or less and warns at 1e-6 mm or less, or when it is no larger than VIC's own
residual on the same row. A row takes the worst grade of its variables.

Of the 90 supported rows 53 pass, 34 warn, 1 fails and 2 cannot be compared
because VIC itself returns NaN. The five rows that are documented departures
give 3 passes, 1 warning and 1 row on which both programs reject the input. The
failing row is `stehe_EB_510_1h` with `BLOWING TRUE`; its cause was not
examined. The warnings are concentrated in the energy-balance rows at the
3-hourly step (20 of the 35 warnings) and in the frozen-soil rows at the
3-hourly step (4 of 4). In an earlier matrix, most warnings were in the Himalaya window and were
traced to snow masses of the order of 1e-16 mm that remain in a pack and whose
presence differs between the two programs because of the last bit of the
exponential function; this was not re-examined for 0.4.7.

### Options and probes

VIC 5's image driver reads 82 global-file keywords. TorchVIC classifies every
keyword and value in one table (`torchvic/io/options.py`, from which
[Options](OPTIONS.md) is generated, and a test fails when the two disagree).
77 keywords have at least one value that TorchVIC runs as VIC does and 6 have at
least one value that is read and announced as having no effect (one keyword has
both kinds of value). The no-effect claim is checked by running VIC 5.1.0 with
and without each option and finding bit-identical output.

A separate set of 28 probes, each changing one option of a base case, compares
the response of VIC and of TorchVIC:

| VIC | TorchVIC | probes |
|---|---|---|
| accepts | runs as VIC does | 12 |
| accepts | refuses by name | 1 |
| rejects | refuses by name | 9 |
| rejects | stops with an error (`LAKES TRUE` is one) | 2 |
| rejects | accepts without a message | 3 |
| rejects | no effect, announced | 1 |

### GPU against CPU

Three rows of the matrix were also run with TorchVIC on a GPU (RTX 6000 Ada) and
compared with the CPU run of TorchVIC, in float64:

| row | variables | identical | largest absolute difference |
|---|---|---|---|
| Indo-Gangetic plain, energy balance, 3-hourly | 128 | 86 | 1.85e-8 |
| Himalaya, frozen soil, 3-hourly | 130 | 38 | 934.8 |
| Stehekin, energy balance, hourly | 128 | 50 | 7.0e-12 |

The 934.8 is the aerodynamic resistance `OUT_AERO_RESIST1` (s m-1) in one cell
at one step, a resistance that is very large when the wind is near zero. It is a
threshold case, not a water or energy result; see
[Known limitations](about/limitations.md).

## Single precision

`--dtype float32` does not reproduce VIC: see
[Known limitations](about/limitations.md).

## Rerun the matrix

CPU only, exact-math mode:

```bash
bash verification/run_all.sh
```

It builds the ISC windows, writes the variant global files, runs both VIC
releases and TorchVIC for every row, the refusal probes and the classic rows,
and regenerates the scoreboard in `verification/results/SCOREBOARD.md`. Run the
GPU-versus-CPU equality check with `python verification/gpu_check.py` on a GPU
node; a wrapper for batch schedulers is provided in `verification/`.

## Unit tests

```bash
pytest tests                                    # all; skips what lacks data
pytest -m "not reference and not slow and not gpu" tests   # what CI runs
pytest -m gpu tests                              # needs a CUDA device
```

Tests find their inputs through environment variables; see
[Testing](development/testing.md).

## Next step

Run times: [Performance](performance.md). What differs from VIC on purpose: [Differences from VIC](differences.md).
