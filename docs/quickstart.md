# Quick start

This page runs TorchVIC on a public VIC 5 example and shows what comes
out. It takes a few minutes, needs no GPU and no VIC installation.

## 1. Install

See [Install](install.md). In short:

```bash
git clone https://github.com/Barbhuiya12/TorchVIC.git && cd TorchVIC
python -m venv .venv && source .venv/bin/activate
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## 2. Get the example data

The example is UW-Hydro's Walnut Gulch image-driver case from the public
`VIC_sample_data` repository (15 cells of a 1/16-degree grid in Arizona,
January 2008, three-hourly forcing). It is not shipped with the package.

```bash
git clone --depth 1 https://github.com/UW-Hydro/VIC_sample_data.git
export TORCHVIC_SAMPLE_DATA=$PWD/VIC_sample_data
```

The sample's global file names its inputs as `${VIC_SAMPLE_DATA}/...`.
TorchVIC resolves `${VIC_SAMPLE_DATA}` from `TORCHVIC_SAMPLE_DATA` (or
`VIC_SAMPLE_DATA`), or from `--sample-data DIR` on the command line. **Set
one of them before the run below**: without it the global file's paths stay
unresolved and the run stops at the first input. The `export` above is
enough; it must be made in the shell that runs `torchvic`.

## 3. Run

```bash
torchvic run $TORCHVIC_SAMPLE_DATA/image/WalnutGulch/parameters/global_param.WalnutGulch.L2015.txt \
    --core 5.1.0 --device cpu --out wg_results
```

`--core 5.1.0` selects the physics of VIC 5.1.0, the release that the
[verification](verification.md) is for. It is the default, so the flag only
makes the choice explicit. `--core 5.0.1` selects the physics of VIC 5.0.1.

The log says what was read and every decision taken from the global file.
The end of it:

```
<Case 15 cells, 80 steps at 8/day, cpu, energy on, core 5.1.0>
ran 80 steps over 15 cells in 2.5 s

history (the global file's OUTFILE streams):
  wrote wg_results/fluxes.WalnutGulch.L2015.2008-01-01.nc (10 records, AGGFREQ NDAYS 1, 17 variables)
```

Ten days at eight steps a day is 80 steps; the global file asks for daily
output (`AGGFREQ NDAYS 1`), so the file holds 10 records of the 17
variables its `OUTVAR` lines name.

Two of the log lines are worth knowing about now. The global file names
the parameter file with a name the sample does not use; TorchVIC says so
and uses the file that is there. The forcing file declares its wind in
`mm s-1`; TorchVIC reads it as m/s and says so. Everything TorchVIC
decides on your behalf is printed; nothing is silent.

## 4. Look at the output

```python
import xarray as xr
ds = xr.open_dataset('wg_results/fluxes.WalnutGulch.L2015.2008-01-01.nc')
print(ds)
print(ds['OUT_EVAP'].mean(('lat', 'lon')).values)      # domain-mean daily evaporation, mm/day
```

The variables are VIC's (`OUT_PREC`, `OUT_EVAP`, `OUT_RUNOFF`,
`OUT_BASEFLOW`, `OUT_SWE`, `OUT_SOIL_MOIST`, ...), on the domain grid with
the domain's `lat`/`lon`, fill value off the mask, `time` and `time_bnds`
as VIC 5 writes them. [Outputs and state files](guide/outputs.md) explains
the aggregation rules.

## 5. Variations

```bash
torchvic run ... --device cuda                 # on a GPU
torchvic run ... --core 5.0.1                  # VIC 5.0.1's physics (the default is VIC 5.1.0)
torchvic run ... --dtype float32               # single precision; not recommended
torchvic run ... --no-energy                   # water balance only, even with FULL_ENERGY TRUE
torchvic run ... --steps 16                    # the first two days only
torchvic run ... --progress                    # a progress bar (pip install -e .[progress])
```

From Python the same run is

```python
import torchvic
case = torchvic.load_case(global_file, device='cpu', core='5.1.0')
out = torchvic.run(case)                              # {name: tensor [time, cell]}
paths = torchvic.write_history(out, case, result_dir='wg_results')
```

## Next

* Your own VIC 5 setup: [Running the model](guide/running.md).
* Your own data: [Prepare your inputs](guide/inputs.md), then
  [The global parameter file](guide/global-file.md).
* The example in depth: [Walnut Gulch, end to end](tutorials/walnut-gulch.md).
