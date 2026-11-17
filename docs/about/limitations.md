# Known limitations of TorchVIC 0.4.8

This page lists what TorchVIC 0.4.8 does not do, or does differently from
VIC, and gives a workaround wherever there is one. The measurements quoted in
these pages were made with TorchVIC 0.4.7; the changes of 0.4.8 are in the
[release notes](../development/release-notes.md). With `--core 5.1.0` (the
default), the results of 0.4.8 are bit-identical to those of 0.4.7.

## Known issues and workarounds

### Starting from a VIC state file on a GPU

**Symptom.** A run that starts from a VIC state file (`INIT_STATE` in the
global file) on a CUDA device stops in the state reader with a tensor-size
mismatch. On the 112,825-cell Indian-subcontinent domain the message is
`size of tensor a (112825) must match tensor b (273370)`.

**Cause.** On a GPU the step normally runs on the covered tiles only: each
covered cell and vegetation tile is one lane. Reading a VIC state file onto
this layout fails. Reading a VIC state file on a CPU is not affected.

**Workarounds.**

* Set `TORCHVIC_LANES=0`. The step then runs on the padded cell-by-tile grid,
  which has about 2.4 times as many lanes for the whole domain, so the run
  needs more memory and time.
* Carry the state in memory instead of through a file: run the period in
  windows and pass the state of one window to the next (`return_state` and
  `initial_state` of `basin.simulate`; [Running the model](../guide/running.md),
  [Python API](../guide/python-api.md)).
* Run on the CPU (`--device cpu`).

### Single precision (float32) is not recommended

`--dtype float32` runs, but it does not reproduce VIC. On the whole domain
(2004 to 2005, against VIC 5.1.0) the domain-mean annual runoff is 2.565 %
too high in the water-balance configuration (2.458 % with the energy
balance), the 99th percentile of the per-cell relative difference is 351 %,
and the worst cell differs by 18,976 mm in a year, against 0.28 mm in float64.
Runoff differs by more than 1 % in 26,414 of the 112,825 cells (about 23 %).
Only 11 of 88 variables are bit-identical, and there is no float32 test in the
release gate. About 670 cells (0.6 % of the
domain) carry almost all of the bias: in VIC 5.1.0's literal form of the
drainage between soil layers one term overflows in float32 in cells with a
large Brooks-Corey exponent. The float32 run is faster and smaller: the two-year water-balance run
(all output variables written) takes 177 s against 332 s in float64 and 14.5 GB
against 28.6 GB of peak physics memory (H200, not exclusively timed).

Two further points apply on a GPU. With the default fusion mode the compiler
stops with an error; setting `TORCHVIC_FUSION_FP=legacy` lets a float32 run
start. Float32 runs also failed intermittently at start without a traceback
in 5 of 6 plain water-balance jobs and did not fail with Python's fault
handler enabled (`PYTHONFAULTHANDLER=1`); the cause is not identified.

**Workaround.** Run in float64, the default. TorchVIC's agreement with VIC is
established in float64 only.

### Long runs of a large domain keep their outputs on the GPU

`torchvic run` and `basin.simulate` keep the outputs of all steps on the
device until the run ends. A single long call over a large domain can
therefore exhaust the device memory: the whole domain with all output
variables needs about 130 GB for 2,920 steps.

**Workaround.** Run the period in windows with the state carried between
them, as the benchmark driver `benchmarks/run_isc_torchvic.py` does
(`--chunk-days 31`), and ask for only the output variables you need: rows
beyond VIC's standard outputs are formed only for the `OUTVAR` names
requested.

### Three option values that VIC rejects are accepted

TorchVIC refuses by name every option value it does not run as VIC does, and
stops on values that VIC itself rejects. In a test of 28 single-option probes
against VIC, three values that VIC's image driver rejects were accepted by
TorchVIC without a message. If a run depends on an unusual option, check it
against VIC on a small window.

### Several GPUs with the 5.1.0 core agree with one GPU to rounding

With the default core, 5.1.0, a run split over several GPUs or processes
(`--gpus N`) is not always bit-identical to the same run on one device: for
some N, values differ at the last bits (at most about 1e-13 mm in soil
moisture in a 16-cell, 10-day test). The 5.1.0 code path rounds differently for
different numbers of cells in a batch; the physics is the same. With
`--core 5.0.1` sharded runs are bit-identical to the single run.

### A water-balance run with snow bands can differ between two GPU runs in the last bit

Two runs of the water balance with snow bands, started with `torchvic run` on
one GPU, gave values that differ by about 1e-16 when every output variable was
written in float64 at every step (40 steps of the ISC test case; 0.4.7 does the
same). The order of a floating-point sum differs between the runs. The daily
float32 history files of the same configuration were identical over 2004-2005,
and the energy-balance run was identical in float64.

### Constants file

A `CONSTANTS` file that sets any constant other than `VEG_LAI_WATER_FACTOR` is
refused.

## Agreement with VIC

* **Round-off, not bit-identity.** 62 to 79 % of all compared values are
  bit-identical in float64 on the whole domain, and the largest difference of
  an annual value in a single cell is 1.91 mm. In a 7,154-cell subset the
  largest daily soil-moisture difference in a cell is 3.5 mm on the GPU and
  0.45 to 2.7 mm on the CPU; in a ten-year water-balance run it reaches
  9.8 mm in one cell, and 1,539 to 2,567 cells differ in soil moisture by more
  than 1e-6 mm on the last day of a year. Check the configuration you care
  about against VIC on a window of your domain before relying on single-cell
  results ([Verification](../verification.md)).
* **GPU and threshold cells.** Where VIC compares a quantity with a threshold
  (a snow pack of 1e-16 mm, a frozen fraction) a last-bit difference can flip a
  branch for one step. In the Himalayan frozen-soil window of the small-window
  matrix the GPU is less faithful to VIC than the CPU: the aerodynamic
  resistance differs by up to 934.8 s m-1 in one cell and the soil moisture by
  1.11 mm between GPU and CPU, against 0.034 mm between the CPU run and VIC.
  The cause is open. `--exact-math` (CPU, float64) removes the last-bit
  differences of `pow`, `exp` and `sqrt` at the threshold sites
  ([Differences from VIC](../differences.md)).
* **`OUT_ENERGY_ERROR`.** By default TorchVIC reports the residual of its own
  surface energy balance, which is not comparable with VIC's value
  (0.017 against 112 W m-2 in the largest cell of the whole-domain
  energy-balance run). `--energy-error vic` forms the output as VIC 5 does
  from the reported fluxes.
* **Verification matrix.** Of the 90 supported rows of the small-window matrix
  53 pass, 34 warn, 1 fails and 2 cannot be compared because VIC itself
  returns NaN. The failing row is `BLOWING TRUE` with the hourly energy
  balance on the Stehekin sample; its cause was not examined.
* **Paths compared only on small windows.** The finite-difference frozen soil
  (`QUICK_FLUX FALSE`) and the daily model step with snow sub-steps were
  compared with VIC on two windows only: the five rows of the forest window
  pass and the five rows of the Himalayan window warn. Lakes (`LAKES TRUE`)
  were not part of the whole-domain tests because the Indian-subcontinent
  data contain no lake parameters.
* **Lake channel inflow.** The `CHANNEL_IN` forcing is accepted but not used:
  the channel inflow of a lake is always 0, and the log says that `CHANNEL_IN`
  has no canonical unit and is passed through unconverted. A run with
  `LAKES TRUE` therefore has no channel inflow into the lake.
* **Daily forcing.** The MTCLIM disaggregation was compared with MetSim 2.0.1
  and not with VIC 4.2.d; the guard against `Tmin > Tmax` in `calc_tair` was
  not examined. The differences from VIC 4.2.d's MTCLIM are listed in the
  [change log](https://github.com/Barbhuiya12/TorchVIC/blob/main/CHANGELOG.md).

## Performance

* Every time is for one simulated year of one domain on one cluster. VIC is
  faster for domains of 20,000 cells or fewer ([Performance](../performance.md)).
* On a CPU TorchVIC is slower than VIC. In the Himalayan window, with
  `QUICK_FLUX FALSE`, TorchVIC on the CPU took 2,400 s against 157 s for VIC.
* The H200 was timed with one GPU dedicated to the job on a node shared with
  two other jobs, so its times are an upper bound for an exclusive node.
* Multi-GPU runs (`--gpus N`) were not timed. Frozen soil and five elevation
  bands per cell were not timed for 0.4.7.
* The energy of the CPU runs was not measured, so no energy comparison between
  CPU and GPU is made.
* Workstation GPUs have a much lower double-precision throughput and less
  memory than data-centre GPUs. A card with 48 GB holds about 1.5e5
  (water balance) or 1.3e5 (energy balance) cells of this model, extrapolated
  linearly from the measured 309 and 379 kB per cell; this was not measured.

## Platform and scope

* All tests and timings were made on Linux (Ubuntu 24.04) with Python 3.11 and
  torch 2.6.0 (CUDA 12.4 on the GPUs). Other platforms have not been tested.
  Torch 2.7.1, which pip installs within the required range, passed the quick
  start on a CPU and a GPU, the GPU fusion and CUDA-graph tests and a five-day
  comparison of history files with 2.6.0, but not the full test suites.
* The flood connector's Triton backend, its mixed-precision scheme and its
  multi-GPU mode need a CUDA GPU; on a CPU they stop with an error.
* Only the Indian-subcontinent domain was used for the whole-domain
  comparisons. Global domains and other climates have not been tested.
