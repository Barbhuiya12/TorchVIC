# Install

TorchVIC is a Python package that you install from source. It needs

* Python 3.10 or later (the package lists 3.10, 3.11 and 3.12; the tested stack
  is Python 3.11);
* PyTorch 2.6 or 2.7. `pyproject.toml` requires `torch>=2.6,<2.8`. The test
  suites ran on 2.6.0 (2.6.0+cu124 with CUDA 12.4 and Triton 3.2.0 on the GPUs).
  2.7.1, which pip picks within that range, passed the quick start on a CPU and
  a GPU, the GPU fusion and CUDA-graph tests and a five-day comparison of history
  files with 2.6.0;
* Linux. All tests and timings were made on Ubuntu 24.04; other platforms have
  not been tested.

NumPy, xarray, netCDF4, cftime, pandas and PyYAML are installed with TorchVIC.
A GPU is optional ([Performance](performance.md) says when it pays off).

## 1. Get the code

```bash
git clone https://github.com/Barbhuiya12/TorchVIC.git
cd TorchVIC
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install PyTorch for your hardware

PyTorch is installed first, because the right wheel depends on whether you
have a GPU and which CUDA driver it runs.

**CPU only** (laptops, small cases, verification runs):

```bash
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
```

The CPU wheels do not include Triton. TorchVIC does not need it on a CPU; only
the flood connector's GPU options do (see [Known limitations](about/limitations.md)).

**NVIDIA GPU**: pick the torch 2.6 wheel for your CUDA driver at
<https://pytorch.org/get-started/locally/>, for example

```bash
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
```

Check that it sees the card:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

The GPU path uses CUDA graphs and `torch.compile` kernel fusion. Triton is part
of the CUDA torch wheel. The first run for a new cell count compiles the
kernels and is slower; the compiled code is cached in
`~/.cache/torchvic/compile` (`TORCHVIC_COMPILE_CACHE` moves it, and a
`TORCHINDUCTOR_CACHE_DIR` or `TRITON_CACHE_DIR` that you set wins). A C compiler
(`gcc`) on the `PATH` speeds up the `IMPLICIT` soil-temperature solve on the
CPU; without one the run is correct and slower.

A GPU with good float64 throughput is preferable, because the model runs in
double precision. The H200 and the RTX 6000 Ada were tested. The device memory
needed grows linearly with the number of cells (309 kB per cell with the water
balance and 379 kB with the energy balance); the 112,825-cell domain needs 34.9
and 42.8 GB. The [coupled system](coupled/overview.md) needs more.

## 3. Install TorchVIC

The installed torch must be 2.6 or 2.7. The fusion of the GPU kernels uses
private interfaces of PyTorch's inductor compiler. On a torch that lacks them
TorchVIC switches the fusion off with a warning and runs the unfused path: the
same results, slower on a GPU.

```bash
pip install -e .              # torch, numpy, xarray, netCDF4, cftime, pandas, pyyaml
pip install -e .[progress]    # optional: a progress bar over the time loop (tqdm)
pip install -e .[zarr]        # optional: Zarr forcing stores (zarr>=3, numcodecs)
pip install -e .[flood]       # optional: DEM conditioning for the flood connector (scipy)
pip install -e .[test]        # optional: pytest and scipy, to run the tests
torchvic --version
```

`-e` installs in place so that `git pull` updates the installation. TorchVIC is
not on PyPI; install it from the repository as above. The DEM conditioning of the
flood connector needs SciPy, which is not installed with TorchVIC
(`pip install -e .[flood]` installs it).

### Building VIC and fetching the test data (optional)

Only needed to rerun the comparisons against VIC itself, not to use TorchVIC.
`tools/get_test_data.sh` downloads the VIC releases and builds their image
drivers, so it needs `nc-config` (NetCDF-C development files) and `mpicc` (an
MPI C compiler) on the `PATH`, besides `curl` and `git`; the script checks for
them and says which one is missing. On a cluster load the netCDF and MPI modules
first.

## 4. Check the installation

The unit tests that need no external data run on a CPU:

```bash
pip install -e .[test]
CUDA_VISIBLE_DEVICES= python -m pytest -q -m "not reference and not slow and not gpu" tests
```

With a GPU, `python -m pytest -q -m gpu tests` runs the device tests (CUDA
graphs, fusion, the tile-sum kernel) on it.

Then run the [Quick start](quickstart.md).

## Clusters

* **Modules.** Load a CUDA driver module at least as new as the torch wheel's
  CUDA; the wheel brings its own CUDA runtime libraries.
* **netCDF.** `netCDF4` comes as a wheel with its own libnetcdf. On a cluster
  that builds VIC against a system netCDF, put that library first on
  `LD_LIBRARY_PATH` and, if the wheel clashes with it, install `netCDF4` from
  source against the same library.
* **Batch schedulers.** Ask for the GPUs (one, or N with `--gpus N`) and a few
  CPU cores: the forcing reader and the history writer run on the CPU while the
  GPU computes.
* **Caches.** The compile caches are under `~/.cache/torchvic/compile`.
  `TORCHVIC_COMPILE_CACHE`, or `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR`,
  move them to a shared, persistent file system so that a warm compile survives
  a new job.

## Building this documentation

```bash
pip install -r docs/requirements.txt
mkdocs serve                      # http://127.0.0.1:8000
mkdocs build --strict
```
