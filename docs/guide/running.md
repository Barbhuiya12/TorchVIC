# Run the model

```bash
torchvic run INPUT [options]
```

This page lists the command-line options, what a run does, the environment
variables that change how it runs and how to use several GPUs.

`INPUT` is a VIC 5 image-driver global parameter file, or a study file
(`.yaml`, `.yml`, `.json`; see [Study files](study.md)). The global file
decides the run, as it does VIC's: the physics options, the model step
and the simulation window. The command-line options choose the hardware
and precision, override the output directory, cut the window for a quick
test, and control what is printed.

**Choose the VIC release with `--core`.** Without `--core` (and without a study
file's `case.core`), TorchVIC runs the physics of VIC 5.1.0, the release that
the whole-domain verification and the timings of these pages are for. Pass
`--core 5.0.1` for VIC 5.0.1. The examples on this site pass `--core 5.1.0`,
which only makes the default explicit.

## Options

| option | meaning |
|---|---|
| `--device auto\|cpu\|cuda\|cuda:N` | `auto` (default) picks the GPU with the most free memory and falls back to the CPU |
| `--gpus N` | split the cells over N processes, one per GPU (`cuda:0..N-1`); with `--device cpu`, N CPU processes. [Several GPUs](#several-gpus) |
| `--core 5.0.1\|5.1.0\|vicwur3` | the VIC release whose physics to run. The default is `5.1.0`, the release verified on the [Verification](../verification.md) page (a study file's `case.core` replaces the default). `5.0.1` runs the physics of VIC 5.0.1. `vicwur3` is the VIC-WUR 3.0 core used by the [VIC-WUR connectors](../connectors/vicwur.md) |
| `--dtype float64\|float32` | floating-point precision (default float64; float32 is not recommended, see [Known limitations](../about/limitations.md)) |
| `--out DIR`, `--result-dir DIR` | where the history files go (default: the global file's `RESULT_DIR`) |
| `--steps N` | run only the first N forcing records of the window (0: all) |
| `--first-step K` | skip the first K records of the window (a spin-up, or a slice of a long run) |
| `--daily` | override the global file's `MODEL_STEPS_PER_DAY` with a daily step (`MODEL_STEPS_PER_DAY 1` needs no flag) |
| `--no-energy` | the water-balance path even where `FULL_ENERGY` is TRUE |
| `--no-history` | do not write the `OUTFILE` streams |
| `--true-extremes` | `AGG_TYPE_MAX/MIN` as the period's own extremes; by default they start from 0 as VIC 5's do |
| `--agg-convention vic5\|vic4.2d` | aggregate and write as VIC 5 (NetCDF streams) or VIC 4.2.d (per-cell ASCII/BINARY); default by the global file's kind |
| `--output-global FILE` | a VIC 4.2.d global file whose output section the history follows |
| `--vic42d-carry` | VIC 4.2.d convention only: carry a cell's partial last `OUT_STEP` period into the next cell's first record, as `vicNl` does |
| `--energy-error solver\|vic` | what `OUT_ENERGY_ERROR` holds on a full-energy run: the surface balance's residual at the solved temperature (`solver`, default), or formed as VIC 5 forms it from the reported rows (`vic`) |
| `--config FILE` | run a coupled-system configuration instead of a global file ([Configuration file](../CONFIGURATION.md)) |
| `--exact-math` | CPU verification mode: glibc `pow`/`exp`/`sqrt` at threshold sites, as VIC gets them ([Differences from VIC](../differences.md)) |
| `--sample-data DIR` | what `${VIC_SAMPLE_DATA}` in the global file resolves to (default `$TORCHVIC_SAMPLE_DATA`) |
| `--progress` | a progress bar over the time loop (`pip install -e .[progress]`) |
| `--quiet` | print only the summary and the files written |

`torchvic --version`, `torchvic run --help`.

## What a run does

1. **Reads the global file** and resolves every path (relative to the
   global file's directory, environment variables expanded).
2. **Checks every option** against what TorchVIC runs: run as VIC, no
   effect (announced), or refused by name. A refusal stops here, before
   any data is read.
3. **Reads the domain and the parameter file** for the active cells, and
   prints what it found (`<ImageDomain ...>`, `<ImageParams ...>`).
4. **Plans the forcing**: the files for the window, the variable for each
   `FORCE_TYPE`, the unit conversion for each. A daily forcing is set up
   for MTCLIM. Every conversion is printed.
5. **Runs the time loop** on the device. On a GPU the first run for a new
   cell count compiles the fused kernels (cached afterwards, see
   [Install](../install.md)); the step itself is replayed as CUDA graphs.
6. **Writes the state file** at the `STATE*` time, if asked.
7. **Aggregates and writes the history streams**, one file per `HISTFREQ`
   period, and prints their paths.

The summary line is `ran N steps over M cells in T s`. With
`TFALLBACK TRUE` the number of temperature solves that found no root and
kept the previous temperature is printed after it.

`torchvic run` keeps the outputs of all steps on the device until the run ends.
For a long run of a large domain, run the period in windows with the state
carried between them (the benchmark driver `benchmarks/run_isc_torchvic.py` does
this with `--chunk-days`; from Python, call `basin.simulate` per window with
`return_state` and `initial_state`), and ask for only the output variables you
need ([Known limitations](../about/limitations.md)).

## Typical commands

```bash
# an existing VIC 5 setup, on the GPU, results next to the global file's RESULT_DIR
torchvic run global_param.txt --core 5.1.0 --device cuda

# the same, results elsewhere
torchvic run global_param.txt --core 5.1.0 --device cuda --out results/run1

# a one-day smoke test of a new input set on the CPU, no output files
torchvic run global_param.txt --core 5.1.0 --device cpu --steps 8 --no-history

# a one-year slice out of a long window (three-hourly: 2920 records a year)
torchvic run global_param.txt --core 5.1.0 --first-step 2920 --steps 2928 --out year2

# daily model step with a daily forcing
torchvic run global_daily.txt --core 5.1.0 --daily --device cuda

# four GPUs
torchvic run global_param.txt --core 5.1.0 --gpus 4 --out results/
```

## Environment variables

| variable | effect |
|---|---|
| `TORCHVIC_SAMPLE_DATA` | what `${VIC_SAMPLE_DATA}` resolves to |
| `TORCHVIC_EXACT_MATH=1` | exact-math mode (same as `--exact-math`) |
| `TORCHVIC_FUSION=0` | run the GPU step without `torch.compile` kernel fusion (slower, same numbers) |
| `TORCHVIC_LANES=0` | run the GPU step on the padded `[cell, tile]` grid instead of the covered tiles (slower, same numbers); the workaround for starting from a VIC state file on a GPU ([Known limitations](../about/limitations.md)) |
| `TORCHVIC_FUSION_FP=legacy` | the earlier floating-point scheme of the fused kernels; needed to run float32 on a GPU, which is not recommended |
| `TORCHVIC_PREREAD=0` | read the parameter file in one process instead of a pool (the streamed path of the coupled runtime and the benchmark driver) |
| `TORCHVIC_COMPILE_CACHE`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | where compiled kernels are cached (by default `~/.cache/torchvic/compile`; `TORCHVIC_COMPILE_CACHE=0` leaves PyTorch's own locations) |
| `CUDA_VISIBLE_DEVICES` | which GPUs `cuda:N` counts |

## Precision and determinism

Float64 is the default on both devices and the mode in which the agreement
with VIC is established. The GPU's results equal the CPU's to rounding, with
exceptions at threshold flips ([Differences from VIC](../differences.md)). Two
runs on the same device with the same inputs gave identical files in the tests,
and runs on an H200 and an RTX 6000 Ada were bit-identical; the one exception
found, in the last bit, is in [Known limitations](../about/limitations.md).
`--dtype float32` roughly halves
the memory and is faster, but it does not reproduce VIC and is not
recommended ([Known limitations](../about/limitations.md)).

## Several GPUs

```bash
torchvic run global_param.txt --core 5.1.0 --gpus 4 --out results/
```

`--gpus N` splits the active cells into N shards of about equal work (the
cells are dealt out cyclically, so each shard gets the same mix of cheap
and expensive cells), runs one process per GPU (`cuda:0..N-1` of the
job's allocation) and merges the shards' pieces into the history and
state files a single run writes. VIC's cells do not read each other, so a
sharded run gives the single run's values. With `--core 5.0.1` they are
bit-identical (checked array-equal in `tests/test_parallel_shards.py`). With
the default core, 5.1.0, some shard counts differ from the single run at the
last bits (at most about 1e-13 mm in soil moisture in a 16-cell test), because
the 5.1.0 code path rounds differently for different numbers of cells in a
batch; the test checks them to 1e-12.

* Each process loads the global file with its own share of the active cells
  and runs the single-device path on it; the parent merges the pieces.
* With `--device cpu`, N CPU processes with the cores divided between
  them.
* Global-file input and the VIC 5 history only: study files,
  `OUTPUT_FORCE` runs and the VIC 4.2.d history convention are refused
  with a one-line message.

Multi-GPU runs were not timed for 0.4.7. Each process repeats the fixed costs of
start-up and input reading, so several GPUs pay off for much larger domains, or
for independent runs, one per GPU, rather than for a domain the size of the
Indian subcontinent (112,825 cells; [Performance](../performance.md)).

## Next step

Read the results: [Outputs and state files](outputs.md). For discharge at gauges, see [Route runoff to gauges](routing.md); to run VIC with other models, see [Coupled runs](../coupled/overview.md).
