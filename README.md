# TorchVIC

**The VIC 5 land surface model in PyTorch, on a GPU or a CPU, from VIC's own input files.**

| | |
|---|---|
| Documentation | <https://barbhuiya12.github.io/TorchVIC/> |
| Source and issues | <https://github.com/Barbhuiya12/TorchVIC> |
| License | [GPL-3.0-or-later](https://github.com/Barbhuiya12/TorchVIC/blob/main/LICENSE) |
| Citation | [CITATION.cff](https://github.com/Barbhuiya12/TorchVIC/blob/main/CITATION.cff) (see also [Citing](#citing)) |
| Release | 0.4.8 |

TorchVIC is the Variable Infiltration Capacity (VIC) macroscale hydrologic
model, release 5, re-implemented in PyTorch. It computes the physics of VIC
5.1.0, the default core and the release that the results below are for, and of
VIC 5.0.1 (`--core 5.0.1`). It reads the files a VIC 5 image-driver user
already has (the global parameter file, `domain.nc`, `params.nc` and the
forcing), solves every grid cell of the domain at once, and writes VIC 5 NetCDF
history and state files. The physics is VIC's; what changes is how VIC is
executed.

On a 112,825-cell Indian-subcontinent domain (3-hourly forcing, measured with
TorchVIC 0.4.7 in double precision):

* TorchVIC agrees with compiled VIC 5.1.0 to round-off in every cell, on every
  day and for every output variable: the domain-mean annual totals differ by
  0.0000 %, the largest difference of an annual value in a single cell is
  1.91 mm, and 62 to 79 % of all values are bit-identical in float64. It is
  agreement to round-off, not bit-identity.
* One H200 runs a simulated year 2.88 times (water balance) and 1.97 times (full
  energy balance) faster than the best layout of VIC on a full, exclusive
  384-core CPU node; an RTX 6000 Ada 1.87 and 1.08 times. The H200 was timed with
  one GPU dedicated to the job on a shared node, so its times are an upper
  bound. VIC is faster for domains of 20,000 cells or fewer.
* Repeated runs, and runs on the H200 and the RTX 6000 Ada, gave bit-identical
  output in the tests, with one exception in the last bit
  ([Known limitations](https://barbhuiya12.github.io/TorchVIC/about/limitations/)). Float32 is not recommended.

## Features

* VIC 5 image-driver inputs and outputs; VIC classic files via
  `torchvic convert-classic`; sub-daily or daily forcing (MTCLIM disaggregation
  on the device); NetCDF or Zarr forcing.
* Every option of VIC 5.1.0's image driver is run as VIC runs it, read to no
  effect and announced, or refused by name before the run starts.
* One GPU, several GPUs (`--gpus N`) or a CPU; float64 by default.
* GPU river routing to gauges, batched parameter ensembles and goodness-of-fit
  metrics.
* Coupled runs: routing, reservoirs, water quality, groundwater, flood
  hydraulics, irrigation and crops in one GPU time loop.

## Install

TorchVIC needs Python 3.10 or later and torch 2.6 or 2.7 (`torch>=2.6,<2.8`;
2.6.0 is tested), on Linux. Install the torch wheel for your hardware first.

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu   # CPU; for a GPU use the wheel for your CUDA driver
```

Then install TorchVIC, by one of two routes.

**pip** (install the torch wheel for your hardware first):

```bash
pip install torchvic
pip install "torchvic[flood,progress,zarr]"    # optional extras: progress, zarr, flood, test
```

**Source** (the developer route, and the route for release 0.4.8):

```bash
git clone https://github.com/Barbhuiya12/TorchVIC.git && cd TorchVIC
pip install -e .
```

Details, the GPU wheel and cluster notes are on the [Install](https://barbhuiya12.github.io/TorchVIC/install/) page.

## Quick start

The example is UW-Hydro's Walnut Gulch case (15 cells, Arizona):

```bash
git clone --depth 1 https://github.com/UW-Hydro/VIC_sample_data.git
export TORCHVIC_SAMPLE_DATA=$PWD/VIC_sample_data   # required: fills ${VIC_SAMPLE_DATA} in the global file
torchvic run $TORCHVIC_SAMPLE_DATA/image/WalnutGulch/parameters/global_param.WalnutGulch.L2015.txt \
    --core 5.1.0 --device cpu --out wg_results
```

The output is a VIC 5 history file in `wg_results/`. On a GPU use
`--device cuda`. VIC 5.1.0 is the default core and the release verified on the
[Verification](https://barbhuiya12.github.io/TorchVIC/verification/) page; `--core 5.1.0` only makes it
explicit. Pass `--core 5.0.1` for VIC 5.0.1. From Python:

```python
import torchvic
case = torchvic.load_case('global_param.txt', device='cuda', core='5.1.0')
out = torchvic.run(case)
torchvic.write_history(out, case, result_dir='results')
```

## Documentation

The documentation is at <https://barbhuiya12.github.io/TorchVIC/>. It is in
`docs/` and is built with MkDocs (`pip install -r docs/requirements.txt &&
mkdocs serve`). Read in this order: install, quick start, running, inputs and
outputs, coupled runs and connectors, the Python API, verification and
performance.

* [Install](https://barbhuiya12.github.io/TorchVIC/install/), [Quick start](https://barbhuiya12.github.io/TorchVIC/quickstart/) and the [Walnut Gulch tutorial](https://barbhuiya12.github.io/TorchVIC/tutorials/walnut-gulch/)
* [Prepare your inputs](https://barbhuiya12.github.io/TorchVIC/guide/inputs/), [the global parameter file](https://barbhuiya12.github.io/TorchVIC/guide/global-file/),
  [running the model](https://barbhuiya12.github.io/TorchVIC/guide/running/), [outputs](https://barbhuiya12.github.io/TorchVIC/guide/outputs/)
* [Your own domain, from scratch](https://barbhuiya12.github.io/TorchVIC/tutorials/own-domain/)
* [Options, value by value](https://barbhuiya12.github.io/TorchVIC/OPTIONS/) and [differences from VIC](https://barbhuiya12.github.io/TorchVIC/differences/)
* [Verification](https://barbhuiya12.github.io/TorchVIC/verification/) and [performance](https://barbhuiya12.github.io/TorchVIC/performance/)
* [Known limitations](https://barbhuiya12.github.io/TorchVIC/about/limitations/)
* [Coupled runs](https://barbhuiya12.github.io/TorchVIC/coupled/overview/), the [connectors](https://barbhuiya12.github.io/TorchVIC/connectors/index/) and the [connector API](https://barbhuiya12.github.io/TorchVIC/CONNECTOR_API/)
* [Python API](https://barbhuiya12.github.io/TorchVIC/guide/python-api/)
* [FAQ](https://barbhuiya12.github.io/TorchVIC/about/faq/), [troubleshooting](https://barbhuiya12.github.io/TorchVIC/troubleshooting/),
  [release notes](https://github.com/Barbhuiya12/TorchVIC/blob/main/CHANGELOG.md), [testing](https://barbhuiya12.github.io/TorchVIC/development/testing/)

## Citing

If you use TorchVIC, please cite the software (see `CITATION.cff`):

> Barbhuiya, S. and Gupta, V.: TorchVIC: the VIC 5 land surface model in PyTorch, version
> 0.4.8, GPL-3.0-or-later, <https://github.com/Barbhuiya12/TorchVIC>.

A paper that describes the model and its evaluation is in preparation. Please
also acknowledge VIC, as its developers ask: Liang et al. (1994) and Hamman et
al. (2018), listed with the other references and the third-party credits on the
[How to cite](https://barbhuiya12.github.io/TorchVIC/about/citing/) page.

## License and acknowledgements

Copyright (C) 2026 Siddik Barbhuiya. TorchVIC is free software under the
GNU General Public License, version 3 or (at your option) any later version (see
`LICENSE`); it is distributed without any warranty.

TorchVIC implements the physics of VIC, the Variable Infiltration Capacity model
of the Computational Hydrology Group at the University of Washington
(<https://github.com/UW-Hydro/VIC>; VIC 5.1.0 is MIT-licensed, VIC 5.0.1
GPL-2.0-or-later). The connectors re-implement or are checked against further
codes, among them RVIC, VIC-WUR, MODFLOW 6, LISFLOOD-FP, WOFOST and CropSyst.
Upstream credits, licenses and the data that the tools read are in `NOTICE` and
`THIRD_PARTY_NOTICES.md`.

Questions and bug reports go to the
[issue tracker](https://github.com/Barbhuiya12/TorchVIC/issues).
