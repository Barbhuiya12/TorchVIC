# Study files, observations, ensembles

## Study files

A study file is a small YAML (or JSON) file that names a global parameter
file and, optionally, observations to report the run against. The data
stay in the global file, so nothing is duplicated.

```yaml
case:
  global_param: forcing/global_param.WalnutGulch.txt
  device: auto             # auto | cpu | cuda | cuda:0
  dtype: float64           # float64 | float32
  core: 5.1.0              # 5.1.0 | 5.0.1 (5.1.0 when core is absent)
  exact_math: false        # true: glibc pow/exp/sqrt on the CPU
  steps: 2928              # 0 or absent: the whole window
  first_step: 0
  no_energy: false
  daily: false             # a daily model step (MODEL_STEPS_PER_DAY 1)
  daily_day: forcing       # forcing | off_gmt
  mtclim_margin: 90        # days of daily forcing read around a window

observations:              # optional
  runoff:
    file: obs/runoff_daily.csv
    freq: daily            # daily | step
    cells: mean            # mean | a cell index | a list of them

report:                    # optional
  metrics: [kge, nse, pbias]
```

Everything has a default except `case.global_param`, so the smallest
study file is two lines. Without `core` it runs VIC 5.1.0; add
`core: 5.0.1` for VIC 5.0.1. Run it like a global file; command-line options
override the file's own:

```bash
torchvic run study.yaml --device cuda --out results/
```

With observations the run ends with a table:

```
against the observations:
  runoff         kge      0.7312
  runoff         nse      0.6880
  runoff         pbias   -4.1200
```

### Observation files

Three forms, all placed on the run's own days (or steps):

* a CSV with a header and a date column (`date,value`), dates in ISO
  8601 unless `date_format` says otherwise; a missing row is a missing
  day. This is the form for any record with gaps.
* one value per line, by position from the run's first day; a blank line
  or `nan` is a missing value.
* two columns `index value`, the index counted from the run's first day.

`freq: daily` compares with VIC's daily aggregation of the variable
(fluxes summed, states at the end of the day, rates averaged; `agg:` to
override); `cells: mean` averages the simulation over the cells,
otherwise one cell or a list of them is taken.

## Metrics

`torchvic.metrics` scores a simulation against an observation along time,
ignoring every position where the observation is not finite: `kge`
(Gupta et al. 2009), `kge2012` (Kling et al. 2012), `nse`, `lognse`
(low flows), `r`, `mse`, `rmse`, `mae`, `pbias`. A series with no finite
observation scores NaN.

```python
from torchvic import metrics
metrics.kge(sim, obs)                 # sim, obs [time] or [time, n]; obs may hold NaN
metrics.score('nse', sim, obs)
metrics.kge(sim, obs, components=True)   # (kge, {'r', 'alpha', 'beta'})
```

## Ensembles: many parameter sets in one run

A GPU is idle on a few cells: the cost of a step is in launching kernels,
not in the arithmetic. `case.ensemble(n)` repeats every cell `n` times so
that one run evaluates `n` parameter sets, and `case.member_params`
builds the parameter tensor for them.

```python
import torch, torchvic
case = torchvic.load_case('global_param.txt', device='cuda', core='5.1.0')
ens = case.ensemble(64)                                     # 64 members, member-major
p = ens.member_params({'b_infilt': torch.linspace(0.05, 0.5, 64),
                       'Dsmax': torch.linspace(1.0, 30.0, 64)})
out = torchvic.run(ens, p=p)
runoff = ens.split_members(out['runoff'], 64)               # [member, time, cell]
```

Each member routes to its own outlet, so member `m`'s discharge is column
`m`. Members are extra cells on one longer cell axis, so they run on one
device: 100 members of the water balance for the whole 112,825-cell
Indian-subcontinent domain ran in 21 minutes on one H200
([Performance](../performance.md#an-ensemble-on-one-gpu)). `member_params` accepts the names `case.params()` lists (such as
`rmin_scale`) and the per-cell soil parameters `b_infilt`, `Ds`, `Dsmax`,
`Ws` and `c`; any other name raises a `ValueError` (it would change
nothing). The soil parameters are written into the ensemble case `ens`
itself, so run `ens` (`torchvic.run(ens, p=p)`, `ens.run(p=p)`), not the
original case. Anything not named keeps the parameter file's value. Gradients flow back
to whatever tensor was passed in when the run is made outside
`inference_mode` (`case.run(p=p)` directly), which is the basis for
gradient-based methods.
