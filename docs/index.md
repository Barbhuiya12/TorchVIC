# TorchVIC

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://github.com/Barbhuiya12/TorchVIC/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](install.md)

TorchVIC is the Variable Infiltration Capacity (VIC) macroscale hydrologic
model, release 5 ([Liang et al., 1994](about/references.md#vic);
[Hamman et al., 2018](about/references.md#vic)), re-implemented in PyTorch. It
reads the files of VIC 5's image driver (the global file, `domain.nc`,
`params.nc` and the forcing), solves every grid cell of the domain at once on a
GPU or a CPU, and writes VIC 5 NetCDF history and state files. The physics is
VIC's; what changes is how VIC is executed. A coupled run adds river routing,
reservoirs, water quality, groundwater, flood hydraulics, irrigation and crops
to the same time loop through connectors.

!!! note "Version"
    This is the documentation for TorchVIC 0.4.8. It computes the physics of
    VIC 5.1.0 (core 5.1.0, the default, and the release that the results below
    are for) and of VIC 5.0.1 (`--core 5.0.1`), on a GPU or a CPU. The results
    quoted on this site were measured with TorchVIC 0.4.7 against compiled VIC
    ([Verification](verification.md), [Performance](performance.md)).

## Start here

<div class="card-grid" markdown>

<div class="card" markdown>
<p class="card-title">1. Install</p>

`pip install torchvic` or install from source.
Choose the CPU or the CUDA build of PyTorch first.

[Install TorchVIC](install.md)
</div>

<div class="card" markdown>
<p class="card-title">2. Quick start</p>

Run the 15-cell Walnut Gulch sample on a CPU in a few minutes and read the
output file.

[Quick start](quickstart.md)
</div>

<div class="card" markdown>
<p class="card-title">3. Connectors</p>

Couple VIC with routing, reservoirs, groundwater, flood, irrigation and crops
in one coupled run.

[Connectors](connectors/index.md)
</div>

</div>

The path through the documentation is: [Install](install.md), then
[Quick start](quickstart.md), then [Running VIC](guide/index.md) (inputs,
global file, outputs), then [Coupled runs](coupled/overview.md) and their
[connectors](connectors/index.md), then the [Python API](guide/python-api.md),
then [Verification](verification.md) and [Performance](performance.md). The
[About](about/overview.md) section holds the model overview, limitations and
how to cite.

VIC is a research model developed by the Computational Hydrology Group at the
University of Washington. TorchVIC is written and maintained by Siddik
Barbhuiya. If you use TorchVIC, please acknowledge VIC and TorchVIC as
described in [How to cite](about/citing.md).

## Features

* **VIC's own inputs and outputs.** A VIC 5 image-driver setup runs as it is; a
  VIC classic setup is converted with one command (`torchvic convert-classic`).
  The history files are VIC 5's streams, aggregated as VIC aggregates them;
  VIC's state files restart TorchVIC, and TorchVIC writes the same state-file
  layout.
* **Every option accounted for.** VIC 5.1.0's image driver reads 82 global-file
  keywords. 77 have at least one value that TorchVIC runs as VIC does, and 6
  are read to no effect and announced. A value TorchVIC does not run as VIC does
  is refused by name before the run starts ([Options](OPTIONS.md)).
* **Daily forcing.** Daily maximum and minimum temperature and precipitation
  are disaggregated on the device by the MTCLIM algorithms, so a reanalysis or
  a station-based daily product drives the model directly.
* **Hardware.** One GPU, several GPUs (`--gpus N`) or a CPU, from one code.
* **Tools around VIC.** Zarr forcing, GPU river routing to many nested gauges,
  batched parameter ensembles and goodness-of-fit metrics.
* **Coupled runs.** VIC, routing, reservoirs, water quality, groundwater,
  flood hydraulics, irrigation and crops run in one GPU time loop through a
  connector API, with a water ledger that must close
  ([Coupled runs](coupled/overview.md)).

## Key results

The results below are for a 112,825-cell Indian-subcontinent domain (1/16
degree, 3-hourly forcing, 2004 to 2005) and compiled VIC 5.1.0. They are
measured with TorchVIC 0.4.7 in double precision.

**Agreement with VIC.** TorchVIC agrees with VIC 5.1.0 to round-off, in every
cell, on every day and for every output variable VIC writes (85 to 106
variables), for the water balance, the full energy balance, frozen soil and
five elevation bands per cell:

* the domain-mean annual precipitation, evapotranspiration, runoff, baseflow,
  snow water equivalent and soil moisture differ by 0.0000 %;
* the largest difference of an annual value in any single cell is 1.91 mm
  (soil moisture, full energy balance);
* 62 to 79 % of all compared values are bit-identical in float64, and 97.6 to
  99.0 % after rounding to single precision;
* where VIC's own water balance does not close, TorchVIC's does not close in
  the same cells;
* routed discharge at four snow-fed gauges differs by at most 0.018 m3/s.

This is agreement to round-off, not bit-identity
([Verification](verification.md), [Differences from VIC](differences.md)).

**Speed.** One simulated year of the same domain, end to end (reading, set-up,
time loop and writing), with daily output of six variables:

| | one H200 | one RTX 6000 Ada | VIC 5.1.0, best layout on a full CPU node |
|---|---|---|---|
| water balance | 33.0 s (2.88 times faster) | 50.8 s (1.87 times) | 94.9 s |
| full energy balance | 49.1 s (1.97 times) | 89.8 s (1.08 times) | 96.8 s |

The VIC reference is the best layout found on a full, exclusive 384-core node
(2 x AMD EPYC 9965): 48 MPI ranks with 8 OpenMP threads each. The H200 was
timed with one GPU dedicated to the job on a node that was shared, so its times
are an upper bound. VIC is faster for domains of 20,000 cells or fewer
([Performance](performance.md)).

**Repeatability.** Two runs on one H200, and runs on an H200 and on an RTX
6000 Ada, give bit-identical output (89 of 89 variables in the water balance,
92 of 92 with the energy balance). One exception, in the last bit of
double-precision values, is in [Known limitations](about/limitations.md).

**Precision.** Float64 is the default and the verified mode. Float32 is not
recommended ([Known limitations](about/limitations.md)).

## Where to start

| you want to | read |
|---|---|
| install TorchVIC | [Install](install.md) |
| run the example and see an output file | [Quick start](quickstart.md), [Walnut Gulch, end to end](tutorials/walnut-gulch.md) |
| run your existing VIC 5 image-driver setup | [Run the model](guide/running.md), [Options](OPTIONS.md) |
| run a VIC classic setup | [Prepare your inputs](guide/inputs.md), section *The parameter file* |
| build a domain from scratch (reanalysis forcing, new parameters, routing) | [Your own domain, from scratch](tutorials/own-domain.md) |
| understand every line of the global file | [The global file](guide/global-file.md) |
| read the history and state files, restart a run | [Outputs and state files](guide/outputs.md) |
| get discharge at gauges | [Route runoff to gauges](guide/routing.md) |
| compare a run with observations, run ensembles | [Observations and ensembles](guide/study.md), [Python API](guide/python-api.md) |
| use more than one GPU | [Several GPUs](guide/running.md#several-gpus) |
| run VIC with routing, reservoirs, groundwater, flood, irrigation or crops | [Coupled runs](coupled/overview.md), [Connectors](connectors/index.md) |
| know how far TorchVIC is from VIC, and why | [Differences from VIC](differences.md), [Verification](verification.md) |
| know what does not work yet | [Known limitations](about/limitations.md) |
| something refused or wrong | [Troubleshooting](troubleshooting.md) |

## Scientific use, license and support

TorchVIC changes how VIC is executed, not what it computes. Where it departs
from VIC on purpose, for example in the definition of `OUT_ENERGY_ERROR` or
in the reference height of the snow surface, the departure is documented in
[Differences from VIC](differences.md); the departure in the wind height is
also announced in the run log.

TorchVIC is free software under the GNU General Public License, version 3 or
(at your option) any later version. VIC 5.1.0 is MIT-licensed and VIC 5.0.1 is
GPL-2.0-or-later ([Code and license](downloads/code.md)). It is distributed
without any warranty. Questions and bug reports go to the
[issue tracker](https://github.com/Barbhuiya12/TorchVIC/issues)
([Contact](about/contact.md)).
