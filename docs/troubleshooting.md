# Troubleshooting

TorchVIC prints every decision it takes from the inputs and stops with a named
reason rather than running on with a difference from VIC. Most problems are
therefore a message to read. The common ones are below; known problems with
workarounds are on the [Known limitations](about/limitations.md) page.

## The run is refused before it starts

**`... selects what TorchVIC does not run as VIC does (docs/OPTIONS.md)`**
followed by a list. Each item names an option, its value and why. The table in
[Options](OPTIONS.md) has the value-by-value status; change the option, or run
VIC for that configuration.

**`SNOW_BAND TRUE: the parameter file has no AreaFract` (or no `Pfactor` or band
`elevation`)**: with `SNOW_BAND TRUE` TorchVIC runs every band with area as its
own column, so the parameter file needs the band variables VIC reads for it
(`AreaFract`, and `Pfactor` and `elevation` where a cell has more than one band
with area). Add them to `params.nc`, or set `SNOW_BAND FALSE`.

**`WIND_H is not in the global file, and no single wind height is VIC's here`**:
VIC 5's image driver gives each vegetation class its own wind reference height
from the parameter file, and the parameter file's heights differ while snow can
form. Declare `WIND_H` (the measurement height of the forcing wind) in the
global file. See [Differences from VIC](differences.md#reference-height-of-the-snow-surface-wind_h).

**`FROZEN_SOIL TRUE ... QUICK_FLUX TRUE`**: VIC's defaults make `IMPLICIT` and
`EXP_TRANS` TRUE until a `FROZEN_SOIL FALSE` line; follow the message (put
`QUICK_FLUX TRUE` after `FROZEN_SOIL TRUE`, or run the implicit solve).

**`FORCE_TYPE declares X and no file matching ... contains it`** or `only some
of the files ... contain it`: the forcing variable is missing from all, or from
some, of the yearly files. Check the names with `ncdump -h` and the
`FORCE_TYPE` lines.

**A unit TorchVIC does not accept** (Kelvin, or a spelling it does not know): it
converts only the units it knows (Pa/kPa, mm and rates, degC, W/m2, m/s) and
refuses the rest, naming the file and variable. Convert the file, or fix its
`units` attribute if the numbers are already right.

**`simulation window ... is not covered by the forcing`**: the dates in the
global file reach beyond the files. Check `STARTYEAR..ENDDAY` (or `NRECS`, in
model steps) and the yearly files present.

**`run_cell = 1 ... where mask != 1`**: the parameter file runs a cell the
domain masks out. Fix one of the two; VIC stops on it too.

**A file on another grid**: a forcing store or parameter file whose `lat`/`lon`
values differ from the domain's at the active cells is refused. Regrid, or check
the ordering (both ascending).

**`${VIC_SAMPLE_DATA}` or another path stays unresolved**: set
`TORCHVIC_SAMPLE_DATA` (or `VIC_SAMPLE_DATA`), or pass `--sample-data DIR`
([Quick start](quickstart.md)).

**`history NOT written: RESULT_DIR ... names an environment variable that is not
set`**: give `--out DIR` or set the variable.

## The run starts but

**The first GPU run takes a long time before stepping.** Kernel compilation for a
new cell count; the compiled code is cached afterwards in
`~/.cache/torchvic/compile`. Set `TORCHVIC_COMPILE_CACHE` (or
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR`) on a cluster so that the cache
persists.

**A warning that torch lacks inductor interfaces that kernel fusion needs.** The
installed torch is outside the supported range (2.6 and 2.7). TorchVIC runs
unfused: the same results, slower on a GPU. Install torch 2.6.0
([Install](install.md)).

**Out of GPU memory.** `torchvic run` keeps the outputs of all steps on the
device. Run long periods in windows with the state carried between them, ask for
only the output variables you need, split the cells over more GPUs (`--gpus N`),
or use the water balance, which keeps less state per cell
([Known limitations](about/limitations.md)). Float32 is not a remedy: it is not
recommended.

**`RuntimeError: size of tensor a (...) must match tensor b (...)` when starting
from `INIT_STATE` on a GPU.** A known problem of reading a VIC state file on the
GPU's lane layout; set `TORCHVIC_LANES=0` or follow the other workarounds on
[Known limitations](about/limitations.md).

**The compiler stops with an error in a float32 run on a GPU.** Float32 is not
recommended. If you must, `TORCHVIC_FUSION_FP=legacy` lets it run
([Known limitations](about/limitations.md)).

**The log says a field is `estimated` that the file has.** The `FORCE_TYPE` line
for it is missing or misnamed; TorchVIC only reads what the global file declares.

**`TFALLBACK TRUE: temperature solves that found no root ...`.** Information, not
an error: VIC's own fallback was taken that many times.

## Results differ from VIC

**By more than rounding, against VIC 5.1.0.** Check `--core`. VIC 5.1.0 is the
default; `--core 5.0.1` (or `core: 5.0.1` in a study file, or `core='5.0.1'`
from Python) runs VIC 5.0.1, whose physics differs from that of VIC 5.1.0
([Running the model](guide/running.md)).

**By 1e-12 or less.** Rounding; the two are the same computation in a different
order.

**In the last bits of a threshold variable (melt, snow cover) on a CPU.**
PyTorch's vectorised `pow` and `exp` differ from the C library's in the last bit;
`--exact-math` (CPU, float64) uses the C library at the threshold sites, which is
how the verification matrix is run. [Differences from VIC](differences.md).

**Wind-driven fluxes.** `WIND_H` in the global file is used for the snow surface;
VIC 5's image driver would use the parameter file's per-class heights there.
Remove `WIND_H` to match VIC where the heights agree.

**`OUT_ENERGY_ERROR` is far smaller than VIC's.** TorchVIC reports the residual
of its own surface energy balance by default; `--energy-error vic` forms the
output as VIC does ([Differences from VIC](differences.md#out_energy_error)).

**Per cent or more.** Compare the log's options block with VIC's
`display_current_settings` output; a default VIC applies that the file does not
carry is the usual cause. If the configuration is covered by the
[verification](verification.md) and the difference remains, open an issue with
the global file and the log ([Contact](about/contact.md)).

## Tests

`pytest tests` skips every test whose data is absent and says which;
`tools/get_test_data.sh` downloads the sample data and the VIC releases and
builds their image drivers (needs `nc-config` and `mpicc`).

## Next step

If the message is not listed here, open an issue as described in [Contact](about/contact.md).
