# Python API

The command line is a thin layer over a few calls.

## Top level

```python
import torchvic

case = torchvic.load_case('global_param.txt', device='cuda', dtype=torch.float64,
                          core='5.1.0', steps=0, first_step=0, daily=False,
                          no_energy=False, sample_data=None, log=print)
out = torchvic.run(case)                          # under torch.inference_mode()
paths = torchvic.write_history(out, case, result_dir='results')
study = torchvic.load_study('study.yaml')         # study.case, study.report(out)
torchvic.set_core('5.1.0')                        # the VIC physics for later cases
torchvic.default_device()                         # 'cuda' when a GPU is visible
```

`load_case` reads the global file and everything it names, with the same
checks and log lines as `torchvic run`; a refused option raises
`ValueError` naming it. `core` is the VIC release whose physics runs: without
it (and without `torchvic.set_core`) the physics of VIC 5.1.0 runs, the release
verified on the [Verification](../verification.md) page. `'5.0.1'` selects the
physics of VIC 5.0.1.

## The case

`torchvic.case.Case` (returned by `load_case`):

| | |
|---|---|
| `case.n_cells`, `case.n_steps`, `case.device`, `case.dtype`, `case.dates` | the run's shape |
| `case.gp`, `case.dom` | the global file and the domain as read |
| `case.run(p=None, **kw)` | `basin.simulate` over the whole case; returns `{name: tensor}` at the model step. Call it yourself (not through `torchvic.run`) when you need gradients |
| `case.params(**overrides)` | the run's parameter dict, with overrides |
| `case.aggregate(out, freq='NDAYS 1', how=None, extremes='vic')` | VIC's aggregation as tensors: `(dict, period start times, dropped tail)` |
| `case.write_history(out, result_dir=None, streams=None, extremes='vic', convention=None)` | the `OUTFILE` streams; returns the paths |
| `case.ensemble(n)`, `case.member_params(values, n)`, `case.split_members(x, n)` | batched parameter sets ([Study files](study.md)) |

`torchvic.run` wraps `case.run` in `torch.inference_mode()`: no autograd
graph, less memory. The outputs are tensors on the case's device; move
them with `.cpu()`.

## State files

```python
from torchvic.io import state_file
plan = state_file.save_plan(case)                 # None, or (n_steps, when, path) from the STATE* lines
out = case.run(return_state=True)
state_file.save_state(case, plan, out=out)
```

`INIT_STATE` in the global file is read once and applied by `case.run`.

## Forcing, domain, parameters

```python
from torchvic.io.domain import ImageDomain
from torchvic.io.params import ImageParams
from torchvic.io.forcing import ImageForcing, aggregate_to_daily

dom = ImageDomain('domain.nc')                                 # .rows, .cols, .lat, .lon, .area, .n_active
par = ImageParams('params.nc', dom)
fc = ImageForcing('forcing/isc_3h.', dom, force_type={'AIR_TEMP': 'air_temp', ...})
stamps = fc.plan(); block = fc.read_steps(0, 8 * 31)           # {field: numpy [step, cell]} in SI units
```

## Options with no global-file key

`case.run(**kwargs)` (and `torchvic.run(case, **kwargs)`) passes its keyword
arguments on to `basin.simulate`. A few options of `basin.simulate` have no key
in VIC's global file, so the table of [Options](../OPTIONS.md), which lists the
keys of the global file, does not contain them. A keyword argument replaces the
value that the case built from the global file.

### Carbon: `carbon_options`

`carbon_options` is a dict that matters with `CARBON TRUE`.
`case.options['carbon_options']` holds what the global file gave (`ncanopy`
from `CANOPY_LAYERS` and `rc_mode` from `RC_MODE`); a dict passed to `case.run`
replaces it, so copy it and add the keys you want. The defaults reproduce VIC;
the alternatives correct two defects of VIC 5, which are described under
[Known issues in VIC 5](../about/vic5-issues.md).

| key | values | meaning |
|---|---|---|
| `gas_constant` | `'vic5'` (default), `'per_mol'` | VIC 5 puts a gas constant per kilomole into the nine places of `photosynth.c` whose formulas are written per mole. `'vic5'` does the same; `'per_mol'` uses the constant per mole, which is VIC 4.2.d's arithmetic and the unit in which the activation energies are stated. |
| `soil_respiration` | `'vic5'` (default), `'per_second'` | `compute_soil_resp.c` divides the turnover time of the litter pool by `SEC_PER_DAY / dt` and those of the other two pools by `HOURS_PER_DAY / dt`, with `dt` in seconds. `'vic5'` does the same; `'per_second'` applies the unit of the litter line to all three pools. |
| `initial_pools`, `annual_npp_prev` | the three soil carbon pools (litter, intermediate, slow) and last year's production, each broadcast to `[cell, tile]` | start a run from the pools that VIC would read from its state file. By default a run starts with empty pools and grows no litter until a year has passed, as VIC does from bare ground. The pools need the energy balance and a soil file with `bubble`. |

```python
opts = dict(case.options['carbon_options'])
opts.update(gas_constant='per_mol', soil_respiration='per_second')
out = torchvic.run(case, carbon_options=opts)
```

## Routing

`torchvic.model.river_network` and `torchvic.model.routing`; see
[Routing to gauges](routing.md).

## Metrics

`torchvic.metrics.kge`, `nse`, `log_nse`, `pbias`, ... and
`torchvic.metrics.score(name, sim, obs)`; see [Study files](study.md).

## Conversion

```python
from torchvic.io.classic_to_image import convert
cv = convert('soil.txt', 'vegparam.txt', 'veglib.txt', 'snowbands.txt',
             out_domain='domain.nc', out_params='params.nc', grid='forcing.nc')
```

## Where the physics is

`torchvic.core` holds the process functions (snow, canopy, vegetation,
soil, energy balance, frozen soil, lakes, carbon, blowing snow, MTCLIM),
each documented with the VIC C file and lines it implements;
[Process-to-C mapping](../mapping.md) is the generated index.
`torchvic.model.basin.simulate` is the time loop that calls them.
