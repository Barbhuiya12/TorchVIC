# Performance

This page compares the run time of TorchVIC on a GPU with that of compiled VIC
5.1.0 on a full CPU node. Every time is the end-to-end wall time of one
simulated year of one domain, measured with TorchVIC 0.4.7 in float64. With
`--core 5.1.0` (the default), the results of release 0.4.8 are bit-identical to
those of 0.4.7.

## Summary

On the 112,825-cell Indian-subcontinent domain one H200 simulates a year in
33.0 s with the water balance and 49.1 s with the full energy balance. That is
2.88 and 1.97 times faster than the best layout of VIC on a full, exclusive
384-core node (48 MPI ranks with 8 OpenMP threads each: 94.9 and 96.8 s), and
3.37 and 2.31 times faster than VIC on all 384 cores as MPI ranks. An RTX 6000
Ada is 1.87 times faster in the water balance and only 1.08 times with the
energy balance. VIC is faster for domains of 20,000 cells or fewer. The GPU
needs 35 to 43 GB for the whole domain and, for one year, 1.2 to 2.6 Wh of
board energy.

## Protocol

**Benchmark.** Each run simulates the year 2005 (365 days, 2,920 three-hourly
steps) of the Indian-subcontinent domain, 112,825 land cells, which is 3.3e8
cell-steps. It starts from the state written by an untimed one-year warm-up of
2004, uses VIC 5.1.0 physics and three soil layers, and writes the daily
history of six variables (precipitation, evapotranspiration, runoff, baseflow,
snow water equivalent and the soil moisture of each layer). The two modes are
the water balance (`FULL_ENERGY FALSE`) and the full energy balance
(`FULL_ENERGY TRUE`, `QUICK_FLUX TRUE`). Both programs read the same domain,
parameter files, 3-hourly forcing and global file. A run is timed from the start
of the program to the last output file written (`/usr/bin/time`): imports,
reading the inputs, graph capture and kernel compilation at start-up, the time
loop and writing are all included.

**VIC.** VIC 5.1.0 was compiled from the released sources with GCC 13.3.0 at
`-O3` with OpenMP enabled, so that it runs as pure MPI (one rank per core) and as
hybrid MPI x OpenMP. Every run was the only job on an exclusive compute node,
with input and output staged to node-local disk; the median of three
repetitions is reported. The baseline is the best layout found on a full node,
not a fixed one. The layouts were pure MPI with 2, 20, 50, 100, 192, 256 and 384
ranks on one node, hybrid layouts of 48 x 8, 96 x 4 and 192 x 2 (ranks x
threads), and pure MPI on two and three nodes (768 and 1,152 ranks). A repeat of
the best layout with all files on the shared parallel file system measures the
effect of the file system.

**TorchVIC.** Float64, windows of 31 days. One untimed warm-up run of a full
year makes the compile caches warm; then five timed runs, median and range. The
RTX 6000 Ada runs were made on an exclusive node. The H200 runs were made with
one GPU dedicated to the job on a node that was shared: two other jobs ran on
the other GPUs during every timed run and share the host CPU, memory bandwidth
and file system, so the H200 times are an upper bound for an exclusive node.

**Hardware and software.**

| | CPU compute node | GPU node 1 | GPU node 2 |
|---|---|---|---|
| CPU | 2 x AMD EPYC 9965 (384 cores, 768 threads) | the same | the same |
| host memory | 502 GB | 502 GB | 502 GB |
| GPU | none | NVIDIA H200 NVL, 141 GB; one GPU dedicated, node shared | NVIDIA RTX 6000 Ada, 48 GB; exclusive node |
| operating system | Ubuntu 24.04 | the same | the same |
| compiler, MPI, netCDF | GCC 13.3.0 at `-O3` with OpenMP, Open MPI 5.0.11, netCDF 4.9.3 | | |
| Python, PyTorch, CUDA, Triton | | 3.11.13, 2.6.0+cu124, 12.4, 3.2.0 | the same |
| program | VIC 5.1.0, commit `14a371a8dbee7ea834152f8ac65e2d6d54a9c30a` | TorchVIC 0.4.7 | TorchVIC 0.4.7 |

## End-to-end time of one simulated year

The table gives the median wall time and the cost in node-hours (VIC, exclusive
nodes) or GPU-hours (TorchVIC, one GPU) per simulated year.

| program and layout | water balance | energy balance | node-hours or GPU-hours (water balance, energy balance) |
|---|---|---|---|
| VIC, MPI, 2 ranks | 1,718.2 s | 3,450.6 s | 0.4773, 0.9585 |
| VIC, MPI, 20 ranks | 383.8 s | 472.1 s | 0.1066, 0.1311 |
| VIC, MPI, 50 ranks | 215.6 s | 238.0 s | 0.0599, 0.0661 |
| VIC, MPI, 100 ranks | 166.2 s | 169.9 s | 0.0462, 0.0472 |
| VIC, MPI, 192 ranks | 159.2 s | 160.5 s | 0.0442, 0.0446 |
| VIC, MPI, 256 ranks | 124.0 s | 125.0 s | 0.0344, 0.0347 |
| VIC, MPI, 384 ranks (one node) | 110.9 s | 113.3 s | 0.0308, 0.0315 |
| VIC, MPI, 768 ranks (2 nodes) | 147.8 s | 134.8 s | 0.0821, 0.0749 |
| VIC, MPI, 1,152 ranks (3 nodes) | 255.9 s | 254.4 s | 0.2133, 0.2120 |
| VIC, hybrid 96 x 4 | 96.1 s | 99.7 s | 0.0267, 0.0277 |
| **VIC, hybrid 48 x 8 (best)** | **94.9 s** | **96.8 s** | 0.0264, 0.0269 |
| VIC, hybrid 192 x 2 | 109.7 s | 112.8 s | 0.0305, 0.0313 |
| VIC, hybrid 48 x 8, files on the parallel file system | 101.8 s | 100.7 s | 0.0283, 0.0280 |
| TorchVIC, RTX 6000 Ada | 50.81 s | 89.8 s | 0.0141, 0.0249 |
| TorchVIC, H200 (one GPU dedicated, node shared) | 32.96 s | 49.11 s | 0.0092, 0.0136 |

The best layout of VIC on one full node is the hybrid 48 ranks x 8 threads, with
94.87 s (water balance) and 96.79 s (energy balance); the ranges of the three
repetitions are 92.9 to 95.2 s and 94.7 to 96.9 s. The best pure-MPI layout is
384 ranks. Moving the files of the best layout from node-local disk to the
parallel file system changes the time by 4 to 7 %, so the file system does not
decide the comparison.

One H200 takes 32.96 s (range 32.49 to 37.37 s) with the water balance and
49.11 s (48.58 to 50.28 s) with the energy balance; the RTX 6000 Ada takes
50.81 s (50.23 to 51.76 s) and 89.8 s (89.13 to 90.03 s).

| speed-up of one GPU over | water balance | energy balance |
|---|---|---|
| H200 over the best VIC layout (48 x 8, one node) | 2.88 | 1.97 |
| H200 over VIC on 384 MPI ranks | 3.37 | 2.31 |
| RTX 6000 Ada over the best VIC layout | 1.87 | 1.08 |
| RTX 6000 Ada over VIC on 384 MPI ranks | 2.18 | 1.26 |

These are measured against a full node, not against a part of one. They apply to
this domain and to the year 2005.

### Scaling of VIC with the number of cores

The time of VIC falls with the number of ranks on one node, from 1,718 s on two
ranks to 111 s on 384 (water balance), but the total stops improving because the
other parts of the run do not shrink: reading the forcing takes 34 to 60 s at
every layout, and the start-up outside the loop grows from 12.4 s on two ranks to
31.2 s on 384, against 10.9 s for the hybrid layout. The compute loop alone keeps
falling (1,551 s on two ranks, 38.6 s on 384). Beyond one node VIC becomes
slower: 148 s on two nodes and 256 s on three (water balance). This is the
behaviour [Hamman et al. (2018)](about/references.md#vic) describe for VIC 5: a
simple configuration is limited by input and output, a physics-heavy one scales
further. On this domain the scaling of VIC is limited by input and start-up, not
by the physics.

### Where the time goes

| stage (seconds) | VIC, best layout | TorchVIC, RTX 6000 Ada | TorchVIC, H200 |
|---|---|---|---|
| water balance: start-up and set-up | 10.9 | 5.5 | 5.8 |
| water balance: reading the forcing | 38.6 | 0.9 | 1.7 |
| water balance: physics | 42.0 | 43.9 | 24.9 |
| water balance: writing | 3.3 | 0.5 | 0.5 |
| water balance: total | 94.9 | 50.8 | 33.0 |
| energy balance: start-up and set-up | 11.0 | 5.4 | 5.6 |
| energy balance: reading the forcing | 36.7 | 0.9 | 1.8 |
| energy balance: physics | 45.8 | 83.0 | 41.1 |
| energy balance: writing | 3.2 | 0.5 | 0.5 |
| energy balance: total | 96.8 | 89.8 | 49.1 |

VIC reads the forcing in series with the compute loop. TorchVIC reads the next
window in the background while the current one computes, so only the initial
wait appears. The physics of TorchVIC alone is 1.7 times (water balance) and 1.1
times (energy balance) as fast as VIC's compute loop on a full node, so most of
the speed-up in end-to-end time comes from reading the forcing in the background
and from the shorter start-up.

## Domain size and memory

The comparison depends on the size of the domain. The table compares one H200
with VIC in its best layout (48 x 8, exclusive node) on nested subsets of the
domain. The last column is the time of VIC divided by the time of TorchVIC; above
1 the GPU is faster.

| cells | water balance: TorchVIC | VIC | ratio | energy balance: TorchVIC | VIC | ratio |
|---|---|---|---|---|---|---|
| 1,000 | 22.4 s | 13.0 s | 0.58 | 30.2 s | 13.1 s | 0.43 |
| 5,000 | 24.3 s | 15.1 s | 0.62 | 32.9 s | 16.2 s | 0.49 |
| 20,000 | 38.5 s | 19.6 s | 0.51 | 35.5 s | 22.4 s | 0.63 |
| 50,000 | 28.4 s | 42.8 s | 1.51 | 40.5 s | 44.5 s | 1.10 |
| 112,825 | 33.2 s | 99.1 s | 2.99 | 53.4 s | 99.4 s | 1.86 |

Below the crossover VIC is faster, because TorchVIC pays a fixed cost of start-up
and set-up. The crossover lies between 20,000 and 50,000 cells in both modes. The
water-balance time of the H200 at 20,000 cells is inflated by one slow run
(range 24.2 to 42.2 s); with its minimum the ratio is 0.81, still below 1. The
whole-domain times of this series differ slightly from those of the previous
section, which were timed in another job.

The memory of the GPU grows linearly with the number of cells: the peak memory
allocated by PyTorch with windows of 31 days is 309.1 kB per cell for the water
balance and 379.0 kB per cell for the energy balance, with an intercept below
0.02 GB. On the whole domain it is 34.9 GB and 42.8 GB, which fit on both GPUs.
Extrapolated linearly (not measured), a card of 48 GB holds about 1.5e5 cells
with the water balance and 1.3e5 with the energy balance, and a card of 141 GB
about 4.6e5 and 3.7e5 cells.

## Energy to solution

The power of the GPU board was read with `nvidia-smi` every 200 ms and integrated
over the run. For one simulated year of the domain the board needs 1.17 Wh
(water balance) and 2.43 Wh (energy balance) on the RTX 6000 Ada at a mean power
of 83 and 98 W, and 1.30 and 2.63 Wh on the H200 at 143 and 193 W (medians of
five runs; idle power 72 to 74 W for the H200 and 25 to 26 W for the RTX 6000 Ada).
The energy of the VIC runs and of the host cores of the TorchVIC runs was not
measured, so no ratio of CPU to GPU energy is given.

## An ensemble on one GPU

One H200 holds the whole domain with room to spare, so an ensemble of parameter
perturbations runs on one device. As a demonstration, 100 members of the water
balance for 2005 were run on the whole domain on one H200, each with five soil
parameters multiplied by factors between 0.7 and 1.3. The members start from the VIC state at the end of 2004 and
run in chunks of 365 steps with the state carried between chunks; only the
per-cell sums of runoff, baseflow, evapotranspiration and precipitation are kept.
The loop takes 1,271 s (21.2 min) and the whole job 1,355 s including 83 s of
loading, at a peak GPU memory of 85.8 GB. A member takes 12.5 s (median of
members 2 to 100); the first takes 32.8 s because it captures the graphs. The
estimate for VIC, 100 times the 94.9 s of the best layout, is 9,490 s or 2.64
node-hours; it is estimated, not measured. The comparison is not like for like:
the members write no gridded output, so 12.5 s per member is below the 24.9 s of
the physics of a run that writes six variables.

## Notes

* The RTX 6000 Ada is a workstation card with a much lower double-precision
  throughput than the H200, which is consistent with the larger gap between the
  two cards in the energy balance. Use a data-centre GPU where possible.
* The first run for a new cell count compiles the fused kernels and caches them.
  The timings above are with warm caches (one untimed warm-up year precedes the
  timed runs).
* With daily forcing the MTCLIM disaggregation runs on the device, which adds to
  the run time. The benchmark above uses 3-hourly forcing.
* On a CPU TorchVIC is slower than VIC ([Known limitations](about/limitations.md)).
* `--gpus N` splits the cells into N shards, one process per GPU, and merges the
  output ([Running the model](guide/running.md)). Multi-GPU runs were not timed
  for 0.4.7. Each process repeats the fixed costs of start-up and input reading,
  so several GPUs pay off for much larger domains, not for this one; the notes of
  the benchmark driver put the break-even at several hundred thousand cells.

## Limits of the comparison

* The CPU baseline is complete for one, two and three nodes; a four-node run was
  not made because the cluster has only three CPU nodes.
* The H200 was not timed on an exclusive node, so its times may be too long.
* The RTX 6000 Ada is a workstation card whose double-precision throughput is much
  lower than that of the H200.
* The energy of the CPU runs was not measured.
* Frozen soil and five elevation bands per cell were not timed for 0.4.7.
* The GPU timings of the size series were not made on exclusive nodes.
* Every time is for one simulated year of one domain on one cluster; other
  hardware will differ.

## Reproduce

`benchmarks/` holds the drivers and the global-file templates: `run_isc.sh` and
`run_isc_torchvic.py` run the Indian-subcontinent case, and `isc/` holds the
VIC global files for the water and energy balance. See `benchmarks/README.md`
for the environment variables. The input parameters and forcing of the domain
are built by the scripts in `tools/` ([Your own domain](tutorials/own-domain.md)).

## Next step

The accuracy side of the comparison: [Verification](verification.md).
