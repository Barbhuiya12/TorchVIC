# Known issues in VIC 5

TorchVIC was built by porting VIC 5.1.0 routine by routine and testing each
routine, and whole runs, against the compiled C code. These comparisons
exposed a number of defects in VIC itself. This page lists them, with their
location in the VIC source, their effect, their status upstream, and what
TorchVIC does about each one.

Line numbers refer to VIC 5.1.0 (tag `5.1.0`, commit `14a371a`). The
`develop` branch of [UW-Hydro/VIC](https://github.com/UW-Hydro/VIC) has the
same source code (its later commits change only the documentation build), so
every entry below also applies to `develop`.

## How TorchVIC handles VIC's defects

By default TorchVIC reproduces VIC as released, defects included, so that a
TorchVIC run can be compared with a VIC run to round-off. Where a defect
changes the physics, TorchVIC offers the corrected behaviour as an option.
Where reproducing a defect would lose water or misreport the energy balance,
TorchVIC departs from VIC on purpose, and the entry says so.

## Summary

| # | VIC source | Defect | Affects | Upstream | TorchVIC |
|---|------------|--------|---------|----------|----------|
| 1 | `photosynth.c` | gas constant per kmol in expressions per mol | `CARBON TRUE` | issue [#965](https://github.com/UW-Hydro/VIC/issues/965), pull request [#966](https://github.com/UW-Hydro/VIC/pull/966) | reproduced; corrected by an option |
| 2 | `compute_soil_resp.c` | turnover times in hours divided by a step in seconds | `CARBON TRUE` | issue [#967](https://github.com/UW-Hydro/VIC/issues/967), pull request [#968](https://github.com/UW-Hydro/VIC/pull/968) | reproduced; corrected by an option |
| 3 | `generate_default_lake_state.c` | lake temperature profile left at 0 °C | `LAKES TRUE`, no initial state file | not yet reported | reproduced (default core 5.1.0) |
| 4 | `lake_utils.c` | `get_volume` returns twice the capacity | latent | pull request [#973](https://github.com/UW-Hydro/VIC/pull/973) | value reproduced |
| 5 | image driver, `vic_force.c` | `CHANNEL_IN` read with a stale time index | `LAKES TRUE` with `CHANNEL_IN` | issue [#969](https://github.com/UW-Hydro/VIC/issues/969), pull request [#970](https://github.com/UW-Hydro/VIC/pull/970) | `CHANNEL_IN` not used |
| 6 | RVIC extension, `rout_convolution.c` | water lost with a source time offset | RVIC routing | issue [#971](https://github.com/UW-Hydro/VIC/issues/971), pull request [#972](https://github.com/UW-Hydro/VIC/pull/972) | own router conserves water |
| 7 | `put_data.c` | `OUT_ENERGY_ERROR` does not close where there is snow | full energy balance | issue [#974](https://github.com/UW-Hydro/VIC/issues/974) | own residual by default; VIC's by an option |
| 8 | `vic_version.h` | VIC 5.1.0 reports itself as 5.0.1 | every run | issue [#941](https://github.com/UW-Hydro/VIC/issues/941), pull request [#975](https://github.com/UW-Hydro/VIC/pull/975) | not copied |

## Patches for VIC 5.1.0

Patches for six of the defects are provided here and have been submitted to
UW-Hydro/VIC as pull requests. They change source files only, apply to VIC
5.1.0 and to `develop`, and can be applied together, so they can be used
before they are merged upstream:

| Patch | Fixes | Pull request |
|-------|-------|--------------|
| [01-photosynth-gas-constant.patch](vic5-patches/01-photosynth-gas-constant.patch) | [Gas constant in photosynthesis](#1-photosynthesis-uses-the-gas-constant-per-kmol) | [#966](https://github.com/UW-Hydro/VIC/pull/966) |
| [02-soil-resp-time-units.patch](vic5-patches/02-soil-resp-time-units.patch) | [Soil respiration time units](#2-soil-respiration-divides-hours-by-seconds) | [#968](https://github.com/UW-Hydro/VIC/pull/968) |
| [03-lake-get-volume-return.patch](vic5-patches/03-lake-get-volume-return.patch) | [Lake volume above the basin top](#4-get_volume-does-not-return-above-the-basin-top) | [#973](https://github.com/UW-Hydro/VIC/pull/973) |
| [04-image-channel-in-time-index.patch](vic5-patches/04-image-channel-in-time-index.patch) | [`CHANNEL_IN` time index](#5-channel_in-is-read-with-a-stale-time-index) | [#970](https://github.com/UW-Hydro/VIC/pull/970) |
| [05-rout-rvic-ring-rotation.patch](vic5-patches/05-rout-rvic-ring-rotation.patch) | [Routing ring buffer](#6-rvic-routing-loses-water-with-a-source-time-offset) | [#972](https://github.com/UW-Hydro/VIC/pull/972) |
| [06-version-strings-5.1.0.patch](vic5-patches/06-version-strings-5.1.0.patch) | [Version strings](#8-vic-510-reports-itself-as-501) | [#975](https://github.com/UW-Hydro/VIC/pull/975) |

To apply them to a clone of VIC:

```bash
git clone https://github.com/UW-Hydro/VIC.git
cd VIC
git checkout 5.1.0
git apply /path/to/01-photosynth-gas-constant.patch   # and so on
```

Each patch was compiled with every driver it touches (the image driver for
all six, the classic driver for those that change shared code, and the RVIC
extension for patch 05) with gcc 13.3, without new warnings. In a regression
run with the default options (five cells, ten days, three-hourly, full energy
balance, five snow bands) patches 01 to 04 leave the output bit-identical to
that of VIC 5.1.0, and patch 06 changes only the version attribute. Patches
01, 02, 03 and 05 were also tested with function-level programs that show the
defect and its correction. Patch 04 was checked by inspection, because no
test case with `CHANNEL_IN` data was available.

## Defects in VIC 5.1.0

### 1. Photosynthesis uses the gas constant per kmol

* **What.** `photosynth()` uses `CONST_RGAS`, the universal gas constant per
  kmol (8314.47 J K⁻¹ kmol⁻¹), in nine expressions that are written per mole:
  the Arrhenius factors of Vcmax, KC, KO, Rdark and K, whose activation
  energies are in J mol⁻¹, and the conversions between stomatal resistance
  and conductance. VIC 4.2 used 8.3143 J K⁻¹ mol⁻¹ here.
* **Where.** `vic/vic_run/src/photosynth.c`, lines 69, 92, 94, 135, 167, 177,
  264, 323 and 421.
* **Effect.** The temperature response of photosynthesis is almost switched
  off. The temperature factor of Vcmax at 5 °C is 0.998304 of its value at
  25 °C, against 0.183158 with the gas constant per mole. With the stomatal
  resistance given (the mode used by `surface_fluxes.c`, 80 s m⁻¹), a C3
  canopy has negative net primary production at 5 and at 25 °C
  (−2.51 × 10⁻⁷ and −2.82 × 10⁻⁷ gC m⁻² s⁻¹, against +5.57 × 10⁻⁶ and
  +1.42 × 10⁻⁵ with the gas constant per mole). Measured with the compiled
  routine, for the forcing of one grid cell of the Indian subcontinent and a
  C3 canopy with Vcmax25 = 33 × 10⁻⁶ and Jmax25 = 75 × 10⁻⁶ mol m⁻² s⁻¹.
* **Upstream.** Reported as [#965](https://github.com/UW-Hydro/VIC/issues/965); fixed by pull request [#966](https://github.com/UW-Hydro/VIC/pull/966)
  ([patch 01](vic5-patches/01-photosynth-gas-constant.patch)), which adds
  `CONST_RMOL` (J K⁻¹ mol⁻¹) and uses it in the nine expressions.
* **TorchVIC.** Reproduces VIC by default. The corrected arithmetic is
  available through the Python API:
  `carbon_options={'gas_constant': 'per_mol'}` in `basin.simulate`.

### 2. Soil respiration divides hours by seconds

* **What.** The time step `dt` is in seconds. The litter pool converts its
  turnover time with `SEC_PER_DAY / dt`, but the intermediate and the slow
  pool use `HOURS_PER_DAY / dt`, a leftover from VIC 4, where `dt` was in
  hours.
* **Where.** `vic/vic_run/src/compute_soil_resp.c`, lines 168 and 172 (the
  litter pool, line 160, is correct).
* **Effect.** The intermediate and the slow pool decay 3600 times too fast:
  their turnover times of 33.3 and 1000 years act as about 3.4 and 101 days.
* **Upstream.** Reported as [#967](https://github.com/UW-Hydro/VIC/issues/967); fixed by pull request [#968](https://github.com/UW-Hydro/VIC/pull/968)
  ([patch 02](vic5-patches/02-soil-resp-time-units.patch)).
* **TorchVIC.** Reproduces VIC by default.
  `carbon_options={'soil_respiration': 'per_second'}` in `basin.simulate`
  uses `SEC_PER_DAY` for all three pools.

### 3. A lake started from the default state has a profile at 0 °C

* **What.** When no initial state file is given,
  `generate_default_lake_state()` sets the lake depth from the parameter file
  and then sets the temperature of the lake nodes in a loop over
  `lake->activenod`. At that point `activenod` is still 0, because
  `initialize_lake()` left it there and `compute_derived_lake_dimensions()`
  is called only afterwards. The loop therefore does nothing, and every node
  keeps the 0 °C that `initialize_lake()` set, instead of the soil
  temperature the routine intends.
* **Where.** `vic/drivers/shared_all/src/generate_default_lake_state.c`; call
  order in `vic_populate_model_state.c` of the image driver (lines 47–51) and
  of the classic driver.
* **Effect.** On its Findley Lake sample, VIC 5.1.0 reports after the first
  hour a lake depth of 23.3993 m, close to the file's initial depth, and a
  surface temperature of −0.0065 °C, with ice forming. Found in the source and
  confirmed by TorchVIC's comparison with VIC 5.1.0 on this sample.
* **Upstream.** Not yet reported. The related issue
  [#406](https://github.com/UW-Hydro/VIC/issues/406) (closed) asked for a
  sensible default lake temperature.
* **TorchVIC.** With the default core (5.1.0) it reproduces VIC 5.1.0.
  `torchvic.core.lake.initial_lake_state(..., honour_depth_in=True)` gives the
  intended start (the file's depth, the profile at the soil temperature); no
  global-file keyword selects it.

### 4. `get_volume` does not return above the basin top

* **What.** For a depth above the top node of the lake basin, `get_volume()`
  sets the volume to the maximum volume and the warning status, but does not
  return. The loop over the nodes then adds the volume of the whole basin a
  second time.
* **Where.** `vic/vic_run/src/lake_utils.c`, lines 65–68 and the loop in
  lines 70–87.
* **Effect.** The function returns twice the capacity (volume divided by
  maximum volume 2.000000 in a function-level test). In VIC 5.1.0 the callers
  stop or refuse the input before the value is used, so valid runs are not
  affected.
* **Upstream.** Fixed by pull request [#973](https://github.com/UW-Hydro/VIC/pull/973)
  ([patch 03](vic5-patches/03-lake-get-volume-return.patch)).
* **TorchVIC.** `torchvic.core.lake.get_volume` returns the same value. During
  a run the volume is capped before it is used, so the case does not arise.
  Unlike VIC, TorchVIC does not refuse an initial lake depth above the basin
  top: such an input gives about twice the volume without a message, so check
  `depth_in` against the basin depth of the lake parameter file.

### 5. `CHANNEL_IN` is read with a stale time index

* **What.** With lakes on and a declared `CHANNEL_IN` forcing, the image
  driver computes the time index of the read before the loop over the forcing
  steps, with the value of `j` left by the previous loop (`NF`). It therefore
  reads the record of the next forcing block (and, at the last step, a record
  beyond the end of the file), and copies that one record to all `NF` steps.
* **Where.** `vic/drivers/image/src/vic_force.c`, lines 166–182 (the index is
  set in lines 171–172).
* **Effect.** Each step uses the channel inflow of a later step. Found in the
  source.
* **Upstream.** The unchecked read of VIC 5.0.1 was reported as
  [#813](https://github.com/UW-Hydro/VIC/issues/813) and fixed in 5.1.0; the
  time index was not. Reported as [#969](https://github.com/UW-Hydro/VIC/issues/969); fixed by pull request [#970](https://github.com/UW-Hydro/VIC/pull/970)
  ([patch 04](vic5-patches/04-image-channel-in-time-index.patch)), which reads
  `CHANNEL_IN` inside the loop with the same index as the other forcings.
* **TorchVIC.** Does not use `CHANNEL_IN`: the field is read but not applied,
  so the channel inflow to lakes is zero (see
  [Known limitations](limitations.md)).

### 6. RVIC routing loses water with a source time offset

* **What.** The routing extension keeps a ring buffer of `full_time_length`
  rows, but `convolution()` rotates only the first `n_timesteps` rows. A
  source with `source_time_offset > 0` adds its unit hydrograph to rows
  `offset` to `offset + n_timesteps − 1`. Whatever lands on rows at or beyond
  `n_timesteps` never reaches row 0, which is the row read as discharge, and
  is lost. `rout_init()` also zeroes only `n_timesteps` rows.
* **Where.** `vic/extensions/rout_rvic/src/rout_convolution.c`, lines 33–34
  and 44–55; `rout_init.c`, lines 50–55.
* **Effect.** The loss is the part of the unit hydrograph that falls in its
  last `offset` steps. For a source whose offset equals the unit-hydrograph
  length all of its water is lost. For a single gamma-shaped unit hydrograph
  of 20 steps, 0.29 % of the volume is lost at offset 1 and 12.94 % at offset
  10. Measured with the unmodified `rout_convolution.c` in a function-level
  test. RVIC writes non-zero offsets when the unit hydrographs are subset
  (`SUBSET_DAYS`, 10 in its sample configurations).
* **Upstream.** Reported as [#971](https://github.com/UW-Hydro/VIC/issues/971); fixed by pull request [#972](https://github.com/UW-Hydro/VIC/pull/972)
  ([patch 05](vic5-patches/05-rout-rvic-ring-rotation.patch)), which rotates and
  zeroes all `full_time_length` rows, as RVIC itself does.
* **TorchVIC.** Departs on purpose. It does not read RVIC parameter files.
  Its own router (`torchvic.model.routing`) rotates the whole ring and
  conserves water.

### 7. `OUT_ENERGY_ERROR` does not close where there is snow

* **What.** `put_data()` forms `OUT_ENERGY_ERROR` from the cell-averaged
  output terms: `(SWNET + LWNET) − (LATENT + LATENT_SUB) − (SENSIBLE +
  ADV_SENS) − (GRND_FLUX + DELTAH + FUSION) + (ADVECTION − DELTACC +
  SNOW_FLUX + RFRZ_ENERGY)`. These terms belong to different control volumes
  (canopy, snow pack and ground) averaged over tiles and bands, so their sum
  is not the residual of one energy balance.
* **Where.** `vic/drivers/shared_all/src/put_data.c`, lines 538–553, and
  `calc_energy_balance_error()`, lines 1071–1082.
* **Effect.** The value is zero in snow-free cells and not zero where there
  is snow: 167 W m⁻² on the Stehekin sample; on the Indian subcontinent
  (2004–2005, daily, 112 825 cells) up to 111.98 W m⁻², and above
  1 W m⁻² in 7 557 cells. Adding or removing any single output term does not
  close it.
* **Upstream.** Reported as [#974](https://github.com/UW-Hydro/VIC/issues/974) (the closed issue
  [#775](https://github.com/UW-Hydro/VIC/issues/775) is a different problem).
  No patch: a fix needs a decision on what the output should measure.
* **TorchVIC.** Departs on purpose. By default `OUT_ENERGY_ERROR` is the
  residual of TorchVIC's own surface energy balance at the solved
  temperature, about 10⁻⁶ W m⁻². `torchvic run --energy-error vic` writes
  VIC's expression instead (to 2 × 10⁻¹³ of VIC's value). VIC's expression is
  always written with `CLOSE_ENERGY TRUE`, with `SPATIAL_SNOW TRUE`, and for a
  daily step with snow sub-steps.

### 8. VIC 5.1.0 reports itself as 5.0.1

* **What.** The release commit "bump version to 5.1.0" changed no file.
  `vic_version.h` still defines `VERSION "5.0.1 February 1, 2017"` and
  `SHORT_VERSION "5.0.1"`, and the Python driver's `setup.py` still has 5.0.1.
* **Where.** `vic/drivers/shared_all/include/vic_version.h`, lines 14 and 18;
  `vic/drivers/python/setup.py`.
* **Effect.** `vic_image.exe -v` and `vic_classic.exe -v` print
  "5.0.1 February 1, 2017", and so do the log and the `VIC_Model_Version`
  attribute of every netCDF output file. Only the "VIC Git Tag" line of the
  log identifies a build from a clone correctly.
* **Upstream.** Reported as
  [#941](https://github.com/UW-Hydro/VIC/issues/941); fixed by pull request [#975](https://github.com/UW-Hydro/VIC/pull/975)
  ([patch 06](vic5-patches/06-version-strings-5.1.0.patch)).
* **TorchVIC.** Does not copy the string. `torchvic --version` prints the
  TorchVIC version. History files carry no VIC version attribute; state files
  carry `VIC_Model_Version = 5.1.0`, whichever core is emulated.

## Documented behaviour worth knowing

### Cell elevation replaced by the band mean

With `SNOW_BAND TRUE` the image driver of VIC 5.1.0 replaces the grid-cell
elevation by the area-weighted mean of the snow-band elevations when the two
differ, prints a warning, and computes the band temperature offsets from the
replaced value. The comparison has a tolerance of 10⁻²⁰, so in practice any
difference triggers it. This is intended: the release notes of VIC 5.1.0
require the mean band elevation to match the cell elevation
([GH#701](https://github.com/UW-Hydro/VIC/pull/701)). The classic driver
replaces the elevation only when the two differ by more than 1 m
(`read_snowband.c`, lines 79–85).

The effect depends on the parameter file. In the 1/16° parameter file used
for TorchVIC's verification over the Indian subcontinent, 65 634 of 112 825
cells differ by more than 1 m (up to 255 m), so their elevation is replaced.

TorchVIC reproduces this (`torchvic.io.build.band_mean_elevation`): with
`SNOW_BAND TRUE` the cell elevation is the band mean, and classic-format
inputs follow the classic 1 m rule. It writes one summary line to the log
instead of one warning per cell.

## Observation without a known cause

With `LAKES FALSE`, the Findley Lake image sample gives NaN in several outputs
(for example `OUT_ASAT`, `OUT_DELSWE` and `OUT_DELSOILMOIST`) from the first
step in VIC 5.0.1, and also in VIC 5.1.0 with a daily model step. The cause
has not been isolated, so nothing has been reported upstream. These cases
are marked "VIC NaN" in TorchVIC's verification matrix and are not graded.

## Fixed in VIC 5.1.0, or not part of VIC 5

These were found in earlier VIC releases. They need no action in VIC 5.1.0.

* **Initial lake state lost (5.0.1).** `generate_default_lake_state()` wrote
  the initial depth and temperatures into a local copy of the lake state.
  Reported as [#827](https://github.com/UW-Hydro/VIC/issues/827) and fixed in
  5.1.0 by [#826](https://github.com/UW-Hydro/VIC/pull/826), except for
  [entry 3](#3-a-lake-started-from-the-default-state-has-a-profile-at-0-c).
  With `--core 5.0.1` TorchVIC reproduces 5.0.1.
* **`CHANNEL_IN` read when not declared (5.0.1).** The image driver read
  `CHANNEL_IN` whenever lakes were on, so the Findley Lake image sample
  failed. Reported as [#813](https://github.com/UW-Hydro/VIC/issues/813) and
  fixed in 5.1.0, except for
  [entry 5](#5-channel_in-is-read-with-a-stale-time-index).
* **Band elevation read past the last band (5.0.1).** `vic_init.c` compared
  the cell elevation with `BandElev[SNOW_BAND]`, one element past the last
  band, a value that is practically zero, so the warning and the replacement
  applied in every cell above 1 m. Replaced before 5.1.0 (commit `070bd929`)
  by the comparison described above.
* **`calc_tair` guard (VIC 4.2.d MTCLIM).** The comment of `calc_tair()` says
  that the minimum temperature is limited to 0.5 °C below the maximum after
  the lapse-rate corrections; the code does not do it. It has no effect in
  VIC 4.2.d, whose wrapper makes the corrections zero, and VIC 5 has no
  MTCLIM. TorchVIC's daily-to-sub-daily forcing follows MetSim, applies no
  lapse correction and has no such guard.
