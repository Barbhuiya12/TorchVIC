# Model overview

TorchVIC runs the Variable Infiltration Capacity (VIC) macroscale hydrologic
model ([Liang et al., 1994](references.md#vic)), release 5 ([Hamman et al.,
2018](references.md#vic)), as one batched tensor program in PyTorch. It
implements the physics of VIC 5.0.1 and VIC 5.1.0 and reads and writes VIC 5's
image-driver files. It changes how VIC is executed, not what it computes. This
page summarises what the model solves and where TorchVIC differs in execution;
the physics itself is described in the
[VIC 5 documentation](https://vic.readthedocs.io/en/master/Overview/ModelOverview/)
and in the papers on the [references](references.md) page.

## Main features

VIC is a large-scale, semi-distributed hydrologic model. Like the land surface
models that are coupled to climate models:

1. The land surface is a grid of cells. Each cell is divided into vegetation
   tiles (a vegetation mosaic) and, optionally, elevation bands for snow.
   Sub-grid heterogeneity is handled with statistical distributions.
2. The inputs are time series of meteorological drivers (precipitation, air
   temperature, wind speed, radiation, humidity, pressure), sub-daily or daily.
3. Water and energy balances at the land surface are simulated at a daily or
   sub-daily time step.
4. Water enters a cell only from the atmosphere. Cells do not exchange water or
   energy within a time step, so they are simulated independently of each
   other, and runoff is routed to gauges afterwards.

The last point is what makes VIC fit a GPU: a time step is the same small
calculation repeated over thousands of independent columns.

## What the model solves

For each tile and model step VIC solves:

* **Water balance.** Canopy interception, a multi-layer soil column (usually
  three layers) with the variable infiltration curve (`b_infilt`), fast runoff
  and ARNO or Nijssen baseflow, evapotranspiration (canopy evaporation,
  transpiration, bare-soil evaporation) and the snow pack, a two-layer pack
  with canopy snow ([Andreadis et al., 2009](references.md#vic-physics)).
* **Energy balance** (`FULL_ENERGY TRUE`). The surface energy balance with the
  surface temperature solved iteratively, snow surface and pack energy, canopy
  air, ground heat flux through the soil thermal nodes, and optionally frozen
  soil (`FROZEN_SOIL`; [Cherkauer and Lettenmaier, 1999](references.md#vic-physics)),
  lakes (`LAKES`), blowing snow (`BLOWING`), spatial snow and frost
  (`SPATIAL_SNOW`, `SPATIAL_FROST`) and a carbon module for photosynthesis
  (`CARBON`).
* **Forcing preparation.** Sub-daily forcing is used as given. Daily maximum
  and minimum temperature and precipitation are disaggregated to the model
  step by the MTCLIM algorithms
  ([Thornton and Running, 1999](references.md#forcing-disaggregation-and-routing);
  [Bohn et al., 2013](references.md#forcing-disaggregation-and-routing)). This
  is an input that VIC 5's image driver cannot read.

Runoff and baseflow are routed to gauges with a unit-hydrograph convolution on
a D8 flow network ([Routing to gauges](../guide/routing.md)).

## What TorchVIC changes

| | VIC 5 image driver | TorchVIC |
|---|---|---|
| input files | global file, `domain.nc`, `params.nc`, forcing | the same; daily forcing is also accepted |
| history and state files | VIC 5 NetCDF | the same |
| execution | a loop over cells, in parallel with MPI and OpenMP | all cells of the domain as tensors, on a GPU or a CPU |
| precision | double | double by default |
| options VIC accepts | run | run as VIC runs them, or read to no effect and announced |
| options TorchVIC cannot run as VIC does | | refused by name before the run starts |

Beyond VIC 5, TorchVIC adds:

* one time loop that runs every cell at once, reading the forcing of the next
  window in the background while the device computes the current one;
* the option check at start-up ([Options](../OPTIONS.md));
* batched parameter ensembles and goodness-of-fit metrics
  ([Study files](../guide/study.md));
* a coupled system in which routing, reservoirs, water quality, groundwater,
  flood hydraulics, irrigation and crop growth run in the same GPU time loop
  ([Coupled runs](../coupled/overview.md)).

## Fidelity

TorchVIC agrees with the compiled VIC 5 image drivers to round-off. It is not
bit-identical: sums are added in a different order, PyTorch's `pow`, `exp` and
`log` differ from the C library's in the last bit for a share of the
arguments, and a VIC build with fused multiply-add differs from one without.
The release whose physics runs is chosen with `--core`: VIC 5.1.0, the default
and the release that the results on these pages are for, or VIC 5.0.1 with
`--core 5.0.1`. In the small-window verification matrix a variable passes when
it agrees to 1e-4 (0.01 %) and warns when it agrees to 1e-2 (1 %). The
agreement over a whole domain, the sources of the differences and the cases
that warn are described in [Verification](../verification.md) and
[Differences from VIC](../differences.md); what is not covered is in
[Known limitations](limitations.md).

## Where to go next

[Install](../install.md), then the [Quick start](../quickstart.md). The
[Process-to-C mapping](../mapping.md) lists the VIC C function behind each
TorchVIC function.
