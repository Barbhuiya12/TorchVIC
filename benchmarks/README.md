# Benchmarks

Scripts that time TorchVIC against VIC 5 and check the GPU against the CPU.
They write their results below `benchmarks/results*/` (not tracked) and read
their large inputs from a data directory you provide.

Environment shared by the scripts:

| variable | meaning |
|---|---|
| `TORCHVIC_DATA_ROOT` | root of the input data (domain, parameters, forcing, run directories) |
| `TORCHVIC_PYTHON` | Python interpreter (default `python`) |
| `TORCHVIC_REFERENCE_DIR` | pristine VIC trees, for the VIC 5 runs (default `reference`) |
| `NETCDF_HOME` | netCDF / MPI install used by the VIC 5 runs (default `/usr`) |
| `TORCHVIC_COMPILE_CACHE` | Inductor and Triton cache (default `~/.cache/torchvic`) |

`isc/*.txt` are VIC global-parameter templates; `@DATA_ROOT@`, `@RESULT_DIR@`
and `@LOG_DIR@` are filled by `run_isc.sh` and `run_combos.py`.

## Example batch script (Slurm)

Fill in the placeholders for your cluster.

```bash
#!/bin/bash
#SBATCH --job-name=torchvic-bench
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=<memory>
#SBATCH --time=<hh:mm:ss>
#SBATCH --output=<log-dir>/%x.%j.out

export TORCHVIC_DATA_ROOT=<path/to/data>
export TORCHVIC_PYTHON=<path/to/python>
cd <path/to/TorchVIC>
benchmarks/run_isc.sh torchvic my_run 3h cuda
```
