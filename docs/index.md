# TorchVIC

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://github.com/Barbhuiya12/TorchVIC/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](install.md)

!!! note
    You are viewing the documentation for TorchVIC 0.4.8. It computes the
    physics of VIC 5.1.0 (the default, and the release that the results below
    are for) and of VIC 5.0.1 (`--core 5.0.1`), on a GPU or a CPU. The results
    quoted on this site were measured with TorchVIC 0.4.7 against compiled VIC
    ([Verification](verification.md), [Performance](performance.md)).

TorchVIC is the Variable Infiltration Capacity (VIC) macroscale hydrologic
model, release 5 ([Liang et al., 1994](about/references.md#vic);
[Hamman et al., 2018](about/references.md#vic)), re-implemented in PyTorch. It
reads the files of VIC 5's image driver (the global parameter file,
`domain.nc`, `params.nc` and the forcing), solves every grid cell of the
domain at once on a GPU or a CPU, and writes VIC 5 NetCDF history and state
files. The physics is VIC's; what changes is how VIC is executed.

VIC is a research model developed by the Computational Hydrology Group at the
University of Washington. TorchVIC is written and maintained by Siddik
Barbhuiya. If you use TorchVIC, please acknowledge VIC and TorchVIC as
described in [How to cite](about/citing.md).

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
* **Coupled system.** VIC, routing, reservoirs, water quality, groundwater,
  flood hydraulics, irrigation and crops run in one GPU time loop through a
  connector API, with a water ledger that must close
  ([Coupled system](coupled/overview.md)).

## Install and first run

TorchVIC is installed from source (Python 3.10 or later; torch 2.6 or 2.7,
2.6.0 tested):

```bash
git clone https://github.com/Barbhuiya12/TorchVIC.git
cd TorchVIC
python -m venv .venv && source .venv/bin/activate
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel for a GPU
pip install -e .
torchvic --version
```

Then run UW-Hydro's Walnut Gulch sample (15 cells, ten days):

```bash
git clone --depth 1 https://github.com/UW-Hydro/VIC_sample_data.git
export TORCHVIC_SAMPLE_DATA=$PWD/VIC_sample_data
torchvic run $TORCHVIC_SAMPLE_DATA/image/WalnutGulch/parameters/global_param.WalnutGulch.L2015.txt \
    --core 5.1.0 --device cpu --out wg_results
```

The output is a VIC 5 history file in `wg_results/`. VIC 5.1.0 is the default
core and the release verified on the [Verification](verification.md) page;
`--core 5.1.0` only makes it explicit. Pass `--core 5.0.1` for VIC 5.0.1.
From Python:

```python
import torchvic
case = torchvic.load_case('global_param.txt', device='cuda', core='5.1.0')
out = torchvic.run(case)
torchvic.write_history(out, case, result_dir='results')
```

See [Install](install.md) and the [Quick start](quickstart.md).

## Where to start

| you want to | read |
|---|---|
| install TorchVIC | [Install](install.md) |
| run the example and see an output file | [Quick start](quickstart.md), [Walnut Gulch, end to end](tutorials/walnut-gulch.md) |
| run your existing VIC 5 image-driver setup | [Running the model](guide/running.md), [Options](OPTIONS.md) |
| run a VIC classic setup | [Prepare your inputs](guide/inputs.md), section *The parameter file* |
| build a domain from scratch (reanalysis forcing, new parameters, routing) | [Your own domain, from scratch](tutorials/own-domain.md) |
| understand every line of the global file | [The global parameter file](guide/global-file.md) |
| read the history and state files, restart a run | [Outputs and state files](guide/outputs.md) |
| get discharge at gauges | [Routing to gauges](guide/routing.md) |
| compare a run with observations, run ensembles | [Study files, observations, ensembles](guide/study.md), [Python API](guide/python-api.md) |
| use more than one GPU | [Several GPUs](guide/running.md#several-gpus) |
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
