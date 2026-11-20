# Running VIC: user guide

A TorchVIC run is a VIC 5 image-driver run. If you already run VIC's image
driver you have every input and can go straight to
[Running the model](running.md). Otherwise the steps are, in order:

1. [Install](../install.md) TorchVIC and, for a GPU, the matching PyTorch
   build, and check the installation with the [Quick start](../quickstart.md).
2. [Prepare your inputs](inputs.md): the domain file, the parameter file and
   the forcing, from VIC 5 files, from VIC classic files or from your own data.
3. Describe the run in [the global parameter file](global-file.md). The
   exhaustive list of keys and values, and what TorchVIC does for each, is
   [Options, value by value](../OPTIONS.md).
4. [Run the model](running.md) on a CPU, one GPU or several GPUs.
5. Read the history files and restart from state files
   ([Outputs and state files](outputs.md); the variables are listed in
   [Output variables](../OUTPUT_VARIABLES.md)).
6. Optionally route runoff to gauges ([Routing to gauges](routing.md)),
   compare a run with observations and run parameter ensembles
   ([Study files, observations, ensembles](study.md)).

If something is refused or looks wrong, read
[Troubleshooting](../troubleshooting.md), then
[Differences from VIC](../differences.md) and
[Known limitations](../about/limitations.md).

## Tutorials

* [Walnut Gulch, end to end](../tutorials/walnut-gulch.md): one sample case,
  every input, log line and output explained.
* [Your own domain, from scratch](../tutorials/own-domain.md): the order of
  work for a region where you have no VIC 5 files yet.

## Programming interface

The command line is a thin layer over a few Python calls
([Python API](python-api.md)). Several models can also run together in one
GPU time loop through the [coupled system](../coupled/overview.md).
