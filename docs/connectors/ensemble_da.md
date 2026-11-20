# Ensembles and data assimilation

`torchvic.ensemble` runs the members of an ensemble as extra cells of one model run and
assimilates observations of soil moisture and snow water equivalent (SWE) into the member
states.

## What it does

`Ensemble` builds and runs the members. `ForcingPerturbation` and `sample_parameters`
generate seeded forcing errors and parameter sets. `Assimilator` applies an ensemble Kalman
filter (EnKF), ensemble transform Kalman filter (ETKF) or local ensemble transform Kalman
filter (LETKF) analysis between two windows of a run and records the water that the analysis
adds or removes in a ledger. The code adds no model physics: it uses `Case.ensemble`,
`basin.simulate`, `state.time_window` and the in-memory state that `simulate` returns. It is
a set of Python engines, not registry connectors of the coupled runtime
([Coupled system overview](../coupled/overview.md)). The code is in `torchvic/ensemble/`
(`perturb.py`, `members.py`, `assim.py`, `bench.py`).

## When to use it

* Use it to run `n` members of one case on the GPU with seeded forcing and parameter
  perturbations.
* Use it to assimilate soil moisture or SWE observations into the member states between two
  windows of a run, with a water-balance ledger of the analysis increment.
* Do not use it for a single deterministic run: run the case directly. A one-member ensemble
  without perturbation gives the same result bit for bit.
* Do not use it for snow-band (pseudo-cell) cases, `FROZEN_SOIL` ice or lakes; they are not
  supported (see Limitations).
* For `Case.ensemble` and `Case.member_params` on their own, see
  [Study files, observations, ensembles](../guide/study.md).

## Inputs

Arguments of `Ensemble`, `ForcingPerturbation`, `sample_parameters` and `Assimilator`:

| input | file or variable | unit | meaning |
|---|---|---|---|
| case | `case` (`Case`) | - | the cell batch that is repeated `n` times |
| member count | `n` | none | number of members |
| forcing error | `ForcingPerturbation` (`forcing`) | - | seeded perturbation of the repeated atmosphere; `None` gives every member the same forcing |
| precipitation error | `prcp_sigma` | none | `s` of the multiplicative lognormal factor `exp(s z - s^2/2)` |
| air temperature error | `tair_sigma` | K | additive error on `tair`, `tmax` and `tmin` |
| time correlation | `phi` | none | lag-1 autocorrelation of the noise (AR(1)), `0 <= phi < 1` |
| noise sharing | `spatial` | none | `'member'` or `'cell'` |
| control member | `control` | none | `True` keeps member 0 unperturbed |
| seed | `seed` | none | seed of the generator |
| parameter ranges | `ranges` (`{name: (low, high)}`) | unit of each parameter | bounds of the Latin hypercube |
| log-sampled names | `log` | none | names sampled uniformly in log space (bounds must be positive) |
| nominal values | `control` (`sample_parameters`) | unit of each parameter | values that member 0 takes |
| member parameters | `params` (`{name: [n]}` or `[n, cells]`) | unit of each parameter | per-member parameters; `None` uses the case's default parameters |
| window | `t0`, `t1` | time steps | steps `[t0, t1)` of the run |
| state | `state` | - | in-memory state of the previous window (`None` for the first) |
| output selection | `keep` | none | names of the outputs to return (default: all) |
| analysed layers | `layers` | none | soil layers whose tile-mean moisture is analysed |
| SWE switch | `swe` | none | `True` analyses the snow water equivalent too |
| method | `method` | none | `'enkf'`, `'etkf'` or `'letkf'` |
| localisation radius | `radius_km` | km | Gaspari-Cohn radius; required by `letkf` |
| inflation | `inflation` | none | factor >= 1 on the prior covariance (1.0 is no inflation) |
| local observations | `max_local_obs` | none | nearest observations used per cell by `letkf` |
| lower bound | `lower` | none | `'zero'` or `'resid'` (`resid_moist` as the lower bound of the moisture) |

Observations passed to `Assimilator.analysis(state, obs, time=None)`:

| input | file or variable | unit | meaning |
|---|---|---|---|
| soil moisture | `obs_soil_moisture(cells, value, sigma, layers=None, scale=1.0)` | mm | `scale` times the sum of the moisture of `layers` (default: the analysed layers); `value` and `sigma` are the observation and its error |
| SWE | `obs_swe(cells, value, sigma)` | mm | snow water equivalent; `value` and `sigma` are the observation and its error |
| any linear operator | `ObsSet(cell, coef, value, sigma)` | mm | `coef` has one weight per observation and analysed variable |
| observation time | `time` | time step | step of the analysis, recorded in the ledger |

## Config block

These engines are not registry connectors, so there is no `modules:` entry. The config block
is a short Python snippet; the full loop is under Technical details.

```python
fp = ForcingPerturbation(prcp_sigma=0.0, tair_sigma=0.0, phi=0.0, spatial='member',
                         control=False, seed=0)
prm = sample_parameters(ranges, n, seed=0)
ens = Ensemble(case, n, forcing=fp, params=prm)
A = Assimilator(ens, layers=(0,), swe=False, method='enkf', radius_km=None,
                inflation=1.0, max_local_obs=64, seed=0, lower='zero')
```

`ranges`, `n` and `case` are supplied by the caller. The values shown for `ForcingPerturbation`
and `Assimilator` are the defaults of their signatures.

## Outputs

| output | unit | meaning |
|---|---|---|
| `Ensemble.run` outputs, `[member, cell, time, ...]` | as in the deterministic run | the outputs of `[t0, t1)` with the member axis exposed |
| end-of-run fields, `[member, cell, ...]` | as in the deterministic run | end-of-run fields with the member axis |
| `discharge_mm`, `[member, outlet, time]` | mm | discharge of every member |
| `out['state']` (with `return_state=True`) | - | state on the cell axis; the `state` of the next window |
| `Assimilator.analysis` returns `(new state, info)` | - | state of the same layout, from which the next window starts, and the analysis record (`info['external_m3']`) |
| `ledger.external_m3(i)` | m3 | ensemble-mean external water input `I_ext` of the step that analysis `i` follows |
| `ledger.rows()` | see Ledger | one dict per analysis (`term='da_increment'`) |
| `ledger.rows()` fields `external_m3`, `member_min_m3`, `member_max_m3` | m3 | ensemble mean, minimum and maximum over the members |
| `ledger.rows()` field `clipped_mm_abs` | mm | summed absolute clipped increment |

## Technical details

### Ensemble construction

```python
Ensemble(case, n, forcing=None, params=None)
Ensemble.run(t0=0, t1=None, state=None, return_state=False, keep=None)
```

`Ensemble` builds `Case.ensemble(n)` (the cell batch repeated `n` times, member-major: row
`m * n_cells + c` is member `m`, cell `c`), perturbs the repeated atmosphere with `forcing`
and builds the member parameters from `params`. With `forcing=None` every member has the
same forcing, and with `params=None` the members use the case's default parameters. The run
is the existing GPU path (lanes, CUDA graphs, fusion) on a longer cell axis, unchanged.

`Ensemble.run` runs steps `[t0, t1)` from an in-memory state and returns the outputs with
the member axis exposed: `[member, cell, time, ...]`, `[member, cell, ...]` for end-of-run
fields and `[member, outlet, time]` for `discharge_mm`. `keep` names the outputs to return
(default: all). With `return_state=True`, `out['state']` stays on the cell axis and is the
`state` of the next window.

### Forcing perturbation

```python
ForcingPerturbation(prcp_sigma=0.0, tair_sigma=0.0, phi=0.0, spatial='member',
                    control=False, seed=0)
```

| argument | meaning |
|---|---|
| `prcp_sigma` | multiplicative lognormal precipitation error: the factor is `exp(s z - s^2/2)` with `z` drawn from N(0, 1) and `s` = `prcp_sigma`, so it has mean 1 and precipitation is never negative |
| `tair_sigma` | additive air temperature error (K), applied to `tair`, `tmax` and `tmin`; air density, latent heat and vapour pressure deficit are recomputed from the new temperature (vapour pressure is kept) |
| `phi` | lag-1 autocorrelation of the noise in time (AR(1)), `0 <= phi < 1` |
| `spatial` | `'member'`: one noise series per member, shared by all its cells; `'cell'`: independent noise per cell |
| `control` | `True` keeps member 0 unperturbed |
| `seed` | seed of the generator |

A zero sigma draws nothing: the input tensors of that variable are returned themselves.

### Parameter sampling

```python
sample_parameters(ranges, n, seed=0, log=(), control=None, dtype=torch.float64, device=None)
```

`ranges` maps a parameter name to `(low, high)`. The function returns `{name: tensor [n]}`,
a Latin hypercube over the ranges. The names in `log` are sampled uniformly in log space
(their bounds must be positive). When `control` maps names to nominal values, member 0 takes
them and the hypercube is drawn over the other members.

The result is passed as `params={name: [n]}` (or `[n, cells]` for a parameter that varies by
cell) to `Ensemble`, which hands it to `Case.member_params`. `Case.member_params` takes the
names of `case.params()` and the soil parameters `b_infilt`, `Ds`, `Dsmax`, `Ws`, `c`, and
raises a `ValueError` on any other name. `Case.ensemble` and `Case.member_params` are also
described in [Study files, observations, ensembles](../guide/study.md).

### Reproducibility

All draws come from a CPU `torch.Generator` seeded per call: the same seed gives the same
members on any machine and device. The observation perturbations of the EnKF are drawn the
same way, from the `seed` of the `Assimilator`.

### Data assimilation

```python
Assimilator(ens, layers=(0,), swe=False, method='enkf', radius_km=None,
            inflation=1.0, max_local_obs=64, seed=0, lower='zero')
Assimilator.analysis(state, obs, time=None)      # returns (new state, info)
```

`Assimilator` analyses the state that an ensemble run leaves
(`Ensemble.run(..., return_state=True)['state']`, member-major on the cell axis) at an
observation time and returns a state of the same layout, from which the next window starts.

#### Methods

| `method` | algorithm |
|---|---|
| `enkf` | stochastic EnKF with perturbed observations (zero-mean per analysis); covariance localisation by Schur product with the Gaspari-Cohn function when `radius_km` is set |
| `etkf` | deterministic ensemble-space transform ([Hunt et al., 2007](../about/references.md)), global |
| `letkf` | the same transform per cell on the nearest `max_local_obs` observations within `radius_km` (which `letkf` requires), Gaspari-Cohn weights on R^-1, batched `eigh` over cells (chunks of 2048) |

#### Analysed variables and observations

Per cell, the analysed variables are the tile-mean soil moisture of the layers in `layers`
(mm) and, with `swe=True`, the snow water equivalent (mm). Observation operators are linear
in these variables:

* `obs_soil_moisture(cells, value, sigma, layers=None, scale=1.0)`: `scale` times the sum of
  the moisture of `layers` (default: the analysed layers), mm;
* `obs_swe(cells, value, sigma)`: the snow water equivalent, mm;
* `ObsSet(cell, coef, value, sigma)`: any linear operator (`coef` has one weight per
  observation and analysed variable).

`analysis` takes an `ObsSet` or a list of them. `inflation` (>= 1; 1.0 is no inflation)
multiplies the prior covariance.

#### Constraints

A cell increment is spread over the cell's covered tiles. Moisture is added to the layer of
every tile; for SWE, ice, pack water, surface water, `swq_c`, depth and cold content are
scaled by one factor, so the density of the pack is kept. Per tile and layer,
`0 <= moisture <= max_moist` (with `lower='resid'`, `resid_moist` is the lower bound); SWE is
not negative, and no pack is created where a member has none. The state is handled in
either layout: padded `[cell, tile, layer]`, or the covered-tile lanes of a CUDA run.

#### Ledger

`Assimilator.ledger` (a `DALedger`) gets one entry per analysis. For every member and cell it
holds the increment requested by the filter and the increment realised after the
constraints (soil layers plus SWE, mm); `clipped = realised - requested` is the share of the
constraints. The increment is an external source (+) or sink (-) of water:
`ledger.external_m3(i)` (ensemble mean, m3) is the external input `I_ext` of the step that
the analysis follows ([Water accounting](../WATER_ACCOUNTING.md)). `ledger.rows()` gives one
dict per analysis (`term='da_increment'`) with `time`, `external_m3`, `member_min_m3`,
`member_max_m3` and `clipped_mm_abs` (the summed absolute clipped increment, mm). The change
of the soil moisture and SWE of every member equals its realised increment (see
[Verification status](#verification-status)).

### Forecast-analysis loop

```python
ens = Ensemble(case, 16, forcing=fp, params=prm)
A = Assimilator(ens, layers=(0, 1), method='letkf', radius_km=100.0, inflation=1.05)
state = None
for t0, t1 in windows:
    out = ens.run(t0, t1, state=state, return_state=True)
    state = out['state']
    if observations_at(t1):
        state, info = A.analysis(state, obs, time=t1)    # new state, same layout
        # info['external_m3'] -> I_ext of the step;  A.ledger.rows()
```

`fp` is a `ForcingPerturbation` and `prm` the result of `sample_parameters`. `windows`,
`observations_at` and `obs` are supplied by the caller: the step ranges between observation
times, a test for observations at step `t1`, and an `ObsSet` (or a list of them) built with
`A.obs_soil_moisture` and `A.obs_swe`. The state is a dict of tensors on the ensemble's
device; save it with `torch.save` or pass it on in memory.

### Performance

Members are extra cells on one longer cell axis, so kernel launches do not multiply with the
members. `python -m torchvic.ensemble.bench GLOBAL_FILE` times the run for each member count
given with `--members` and, with `--da`, one LETKF analysis. A whole-domain demonstration with
100 members on one GPU is described in
[Performance](../performance.md#an-ensemble-on-one-gpu).

## Limitations

* Snow-band (pseudo-cell) cases, `FROZEN_SOIL` ice and lakes are not supported.
* An `INIT_STATE` file is read for the case's single-member size only.
* The VIC state-file reader and writer (`torchvic/io/state_file.py`) are not used: states
  are passed on in memory.

## Verification status

Checked in the test suite (`tests/test_ensemble.py`, `tests/test_ensemble_da.py` and
`tests/test_case_ensemble.py`); not part of the VIC verification on the Verification page.
The tests check:

* a one-member ensemble without perturbation reproduces the deterministic run bit for bit
  (CPU and CUDA);
* members agree with standalone runs to rounding (the reductions over a longer cell axis run
  in a different order);
* a run split into windows with the state carried in memory equals the whole run (CUDA);
* the seeded Latin-hypercube sampling and forcing perturbations;
* the analysis of the EnKF, ETKF and LETKF matches the Kalman filter of the sample for a
  linear-Gaussian problem;
* the ETKF and the LETKF agree at a wide localisation radius, and localisation leaves
  distant cells alone;
* the constraints and the ledger account for every millimetre of an increment;
* a twin experiment on the Walnut Gulch sample case.

Tests marked `gpu` need a CUDA device, and tests that read the sample data are skipped when
it is absent ([Testing](../development/testing.md)).


## Next step

[Observations and ensembles](../guide/study.md).
