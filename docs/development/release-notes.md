# Release notes

## 0.4.9 (2026-11-20)

A packaging and documentation release. The model code is unchanged from 0.4.8 (only the version string differs), so every result of 0.4.8 holds.

**Install**
- TorchVIC is on PyPI as `torchvic`: `pip install torchvic`, with the extras `flood`, `progress`, `zarr` and `test`. Install the torch wheel for your hardware first; otherwise pip installs the CUDA build of torch from PyPI. A workflow builds and publishes each tagged release to PyPI with trusted publishing.
- Package metadata for PyPI: description, authors, classifiers and project links.

**Documentation**
- The site is reorganised: a landing page, two install routes (pip and source), the quick start, an overview of all connectors, and one layout for every connector page (what it does, when to use it, inputs with units, configuration, outputs, limitations, verification status).
- Known issues in VIC 5: the default lake state of VIC 5.1.0 is reported upstream (UW-Hydro/VIC issue #976, pull request #977), with its effect on VIC's Findley Lake sample.
- The links of the README are absolute, so they work on the PyPI page.

## 0.4.8 (2026-11-17)

The default VIC core is now 5.1.0 (it was 5.0.1); pass `--core 5.0.1` for the previous physics. With `--core 5.1.0`, results are bit-identical to 0.4.7. Checked on the ISC test dataset in float64 (WB1 and EB1; every variable of the history files compared byte for byte; 0.4.8 run without `--core`, 0.4.7 run with `--core 5.1.0`): the whole period 2004-2005 on one RTX 6000 Ada, and the first 80 steps (ten days) on the CPU. The rest of this release is installation and usability fixes.

**Default core**
- `torchvic run`, `torchvic.load_case` and `torchvic.run` start on the VIC 5.1.0 core. `--core 5.0.1`, `core='5.0.1'`, `torchvic.set_core('5.0.1')` and `core: 5.0.1` in a study file select the previous default; `vicwur3` is unchanged.

**Install**
- `torch` is constrained to `>=2.6,<2.8` in `pyproject.toml`. 0.4.7 asked only for `torch>=2.6`, so pip installed the newest torch, whose private inductor interface (the one `core/fusion.py` patches) had changed, and the GPU run failed. The test suites run on torch 2.6.0. Torch 2.7.1, the release pip resolves inside the range, passed the quick start on a CPU and on a GPU, gave the same history files as 2.6.0 on the ISC test dataset (five days, WB1 and EB1, one RTX 6000 Ada) and passed the GPU fusion and CUDA-graph tests.
- The wheel carries `torchvic/crop/cropsyst/sdr_field_ops.txt`, the NRCS field-operation table that the CropSyst nitrogen code reads at run time. The 0.4.7 wheel left it out, so only an editable install could read it. `tests/test_package_data.py` checks that every data file of the package is listed in `package-data`.
- `scipy` is an optional dependency (extras `flood` and `test`): `torchvic.flood.dem` imports it when the flood connector conditions a DEM, and one MTCLIM test compares with scipy's PCHIP and now skips without it.

**Fixes**
- Without Triton (CPU-only torch wheels do not ship it) the modules that define Triton kernels now import, so `import torchvic.coupling.connectors` and the whole test suite work on a CPU-only install; a Triton kernel launched there raises one clear error. `tests/test_triton_optional.py`.
- `core/fusion.py` raised `AttributeError` on a torch without the private inductor name `disable_pointwise_autotuning` (torch 2.15). It now checks each private name it patches; when one is missing it switches fusion off with a one-time warning and runs the unfused path (same results, slower on a GPU). `tests/test_fusion_api_guard.py`.
- `Case.member_params` accepted any name, but `basin.simulate` reads only the keys of `case.params()`, so the documented `b_infilt`/`Dsmax` ensemble example changed nothing. It now applies `b_infilt`, `Ds`, `Dsmax`, `Ws` and `c` per member to the ensemble case's own soil tensors (the shared ones are never modified) and raises `ValueError` on any other name. The example in `docs/guide/study.md` is corrected (`b_infilt`, not `infilt`). `tests/test_case_ensemble.py`.
- `tools/get_test_data.sh` fetched the non-existent tag `VIC.5.1.0` (the tag is `5.1.0`); it also checks for `nc-config`, `mpicc`, `curl` and `git` and names the missing one.
- Added `verification/gpu_check.sbatch`, the wrapper the verification page names.

**Documentation and tree**
- The install and quick-start pages state the requirements and that `TORCHVIC_SAMPLE_DATA` must be set for the sample global file.
- The skip messages of the sample-data tests point to `docs/downloads/data.md`, which says how to get the data. `THIRD_PARTY_NOTICES.md` lists the test fixtures, oracle drivers and patches that the repository carries, and the optional dependencies. `CHANGELOG.md` is the same text as these release notes.

## 0.4.7 (2026-11-04)

- Documentation rebuilt as a MkDocs site laid out like the VIC documentation (Home, About, Documentation, Development, Downloads) and a short README; data locations read from `TORCHVIC_DATA_ROOT` (the LISFLOOD-FP binary from `TORCHVIC_LISFLOOD_BIN`); `torchvic/core/diaglog.py` for diagnostic logs; packaging with the SPDX licence expression and the notice files in the wheel. `THIRD_PARTY_NOTICES.md` credits the VIC 4.2.d output convention in `torchvic/io/history.py` and the CropSyst sources.
- MTCLIM validated against real MetSim 2.0.1 (PyPI package, its own venv, CPU; `tools/metsim_validation/`). Ten weather series (120 days, 44-60 N and four at 62 N in place of the 68/76 N series, on which MetSim 2.0.1 misindexes or fails at polar day) plus Walnut Gulch 2008, hourly and 3-hourly, MetSim defaults with float64 output. Daily path agrees to round-off for SWdown (5.7e-8 W/m2), cloud fraction, precipitation and wind; air temperature 0.028 C, vapour pressure 1.4 Pa, pressure 1.6 Pa, LWdown 0.11 W/m2 at most, from one 30-second sunrise slot where MetSim's numpy cosine and torch's land on opposite sides of zero. Fixed: with a model step longer than one hour MetSim places the temperature extremes `time_step` minutes (not 60) before sunrise and 67 % of daylight, and reads temperature, vapour pressure, pressure and so longwave at the instant each step begins; TorchVIC averaged hourly values over the step (3-hourly Tair rmse 1.4 C, LWdown 5.4 W/m2). `mtclim.disaggregate(step_minutes=)` and `daily._at_steps` now do the same. The Erbs, Klein and Duffie (1982) daily diffuse-fraction coefficients were checked against the paper (eqn 6) and are correct; the sunset-hour-angle split is now exactly 1.4208 rad. `benchmarks/gpu_graphs/to_hourly_same.py` no longer passes the removed `insw`. New regression test on a 80 kB fixture of MetSim 2.0.1 output (`tests/data/metsim_2_0_1_fixture.npz`).

- Licence fix: MTCLIM daily-to-sub-daily disaggregation rebuilt on MetSim v2.0.1 (UW-Hydro, GPL-3.0; commit 742ef689) in place of the transcription of VIC 4.2.d's MTCLIM (GPL-2.0 only, incompatible with GPL-3.0-or-later). `torchvic/forcing/mtclim.py` and `daily.py` keep the batched tensor design, the per-cell independence and the `daily_to_subdaily`/`expand_forcing` API; no 4.2.d-derived code path remains. Differences from VIC 4.2.d, each measured over ten sites 44-76 N, 120 days (this minus 4.2.d; hourly SWdown rmse 1.46 W/m2, max 57; Tair rmse 0.82 C, max 4.4; daily vapour pressure rmse 18 Pa, max 173; cloud fraction rmse 0.003, max 0.075; longwave only the constant below): (1) the humidity-radiation loop iterates until the dewpoint's rms change is below 1e-6 C (MetSim) instead of VIC's two passes; (2) the 30-day temperature-range mean and the 90-day precipitation total are plain trailing means over the days available (MetSim, with its state file replaced by an expanding mean) instead of VIC's pulled boxcar and wrapped 90-day window, and the precipitation floor is 80 mm/yr as before; (3) air temperature is a PCHIP through the daily minimum (sunrise minus one step) and maximum (67 % of daylight) instead of VIC's zero-derivative Hermite spline on hourly knots, sunrise taken from the 30-second radiation fractions instead of hourly shortwave; (4) vapour pressure is linear in time between the daily minimum-temperature times with end segments extrapolated, VIC held the end days flat; (5) constants: pi exact (VIC 4 used 3.1415927), Stefan-Boltzmann 5.669e-8 (VIC 4: 5.6696e-8, longwave lower by a constant 1.06e-4 relative), pressure P0 101325 Pa, g 9.80665, R 287, lapse 0.0065 (VIC 4: 101300, 9.81; mean pressure +26 Pa); (6) the wind floor applies at every step length (VIC 4 floored only a whole-day step's daily mean); (7) the direct-beam fraction for `carbon: true` is one minus the Erbs et al. (1982) daily diffuse fraction (MetSim has none; VIC's was a linear clip of the transmittance; difference rmse 0.09); (8) not provided because MetSim's MTCLIM has none: slope, aspect and horizon radiation, the snow-pack radiation correction (off by default in VIC), `vp_iter`; they are refused with an error unless neutral. The same ten-site comparison no longer agrees with 4.2.d to 1e-11; `tests/test_mtclim.py` now checks the tensor code against an independent numpy re-reading of MetSim and bounds the distance from 4.2.d at its measured value. `tests/test_mtclim5_crosscheck.py` (a comparison with jhamman/mtclim5, itself an extraction of VIC 4.2) is removed.
Other measured effects. Walnut Gulch 2008 (15 cells, 3-hourly Livneh forcing aggregated to local days and rebuilt at 8 steps a day), annual means against the observed 230.3 W/m2 SWdown, 321.9 W/m2 LWdown, 17.20 C, 870 Pa: 4.2.d 225.8 / 322.3 / 17.09 / 842; MetSim 226.1 / 322.1 / 17.09 / 844. OUTPUT_FORCE against vicNl (Stehekin, 16 cells): daily input, AIR_TEMP rmse 0.39 C (max 1.85), SHORTWAVE 2.4 W/m2 (17.2), LONGWAVE 1.9 W/m2 (6.3), VP 0.005 kPa (0.020), PRESSURE +0.026 kPa mean; precipitation and wind unchanged; hourly input, only FDIR differs (rmse 0.076, max 0.156). `tests/test_refused_zero.py`: the two OUTPUT_FORCE tests that compared those files to vicNl byte for byte now require the byte match on every unchanged column and bound the changed ones at their measured maxima (+10 %). The `torchvic.io.history` VIC 4.2.d output convention (`force_data_42d`, put_data rules) is separate 4.2.d-derived code and is not part of this change.

## 0.4.6 (2026-11-03)

- Fixed: restarts of coupled runs with the MODFLOW member were not bitwise. `core.fusion.setup()` rewires inductor for the whole process and ran lazily at the first fused region, so the groundwater solver compiled under one configuration in a continuous run and another after a restart (all_wur_cropsyst, Nandgaon: 33 of 73 series differed from 1e-13 on day 8). `fusion.ensure_setup()` now runs before the groundwater, reservoir, snow and energy compiles; restart now bitwise (0 of 73). The default flood scheme does not compile and is unaffected. `tests/test_gw_compile_order.py`, `tests/test_flood_setup_order.py`.
- Fixed: the WOFOST 7.2 crop connector sowed once, on the first sowing date of the run; `start_doy` sows a fresh crop every year. Its transpiration reduction (RFTRA) was transp / OUT_PET, which mixes a bare-soil potential without resistance with the dry canopy's transpiration (0.19-0.24 on wet soil, near-zero yields); it is now WOFOST's water-stress factor over the root zone from VIC's soil moisture and its Wcr/Wpwp points. Nandgaon cotton 2000-2004: 0 -> 1907-2898 kg/ha. `docs/connectors/crop.md`, `tests/test_conn_agri.py`.
- Fixed (VIC-WUR mode exact): a routing-only group lost VIC-WUR 3.0's throughfall leak (the core is now built for it); the land-use connector keeps the states of tiles without cover between records, as VIC-WUR 3.0 does not run a tile whose Cv is 0 (vic_run.c:137, :388). Nandgaon against compiled VIC-WUR 3.0: routing only 6e-8; land use alone 4.8e-7; full stack discharge 2.0 -> 0.63 m3/s, first divergence day 151 -> 721, the remaining difference starting from VIC-WUR 3.0's even-vs-proportional return split on a 1e-35 river residue (wu_run_local.c:505-520; documented in `docs/connectors/vicwur.md`). `tests/test_vicwur_connectors_b.py`.

## 0.4.5 (2026-11-01)

- Fixed: the 0.1 m/s wind floor TorchVIC put under the surface aerodynamics (not VIC's: 5.1.0 passes any positive forcing wind to CalcAerodynamic and special-cases only wind == 0, vic_run.c:171, CalcAerodynamic.c:177-186) is now a guard against zero only (`basin.MIN_WIND` 1e-12). A cold, still, deep pack with forcing wind under 0.1 m/s (ISC cell y451 x118, 2004 day 341: 0.0992 m/s) had AERO_RESIST 0.86 % and snow surface temperature 2.8e-4 K off VIC 5.1.0; that cell now matches VIC to 1e-9 s/m and 1e-13 K over 2004. `tests/test_calm_wind_not_floored.py`.
- Faster set-up (bit-identical): the soil bubbling pressure is read with the image parameters and kept in their cache (cache version 4), so set-up no longer opens the parameter file for it (0.4.3 added the cache and moved a 7 s file open from read to set-up). Whole ISC, H200, one month: set-up 8.75 -> 1.65 s; histories byte-identical.

## 0.4.4 (2026-10-31)

- Fixed (speed regression in 0.4.3): the rows beyond the legacy outputs are formed only for the OUT_* names asked for (global-file OUTVAR, or `out_vars` from Python). Whole ISC, 1 year, H200: water balance 49.4 -> 38.9 s, peak memory back to the 0.4.2 level; the RTX 6000 Ada (48 GB) runs water and energy balance again (out of memory in 0.4.3). Requested rows bitwise identical to 0.4.3.
- The coupling's streamed forcing (`stream: true`) asks for the same OUT_* rows as the loaded case, so both write the same history.

## 0.4.3 (2026-10-30)

**VIC output variables**
- All 165 VIC `OUT_*` names are written (snow-band rows, energy rows in water-balance mode, soil nodes, water table, blowing snow, aerodynamic terms, ...); reference table `docs/OUTPUT_VARIABLES.md`. New rows agree with pristine VIC to <= 1e-12; existing outputs bitwise unchanged.
- `--energy-error vic` (`energy_error='vic'`): OUT_ENERGY_ERROR formed from the reported rows as VIC's `put_data.c`; the default stays the solver residual.
- The ISC driver (`benchmarks/run_isc_torchvic.py`) honours SNOW_BAND with the same band expansion as `torchvic run` (bands off bitwise; bands on bitwise to `torchvic run`); whole ISC with 5 bands on one H200: 85 GB.

**ISC campaign (VIC-WUR, CropSyst)**
- Irrigation calendar: MIRCA-OS monthly areas with GGCMI Phase 3 growing seasons (rice main season; second rice only in second-season states); India 2005 withdrawals 1038 km3 (AQUASTAT 688), groundwater 449 km3 (CGWB 243); still high, under study.
- NONRENEWABLE_LIMIT read from CONSTANTS only (as VIC-WUR); fixed-mode hold only after a short-supply day; Jaegermeyr irrigation efficiency.
- Crop suitability without substitution: no rabi crop where Nov-Dec tasmin < 0 C, no kharif chickpea, thermal maturity rule; rainfed failure 29.8 -> 8.7 %, irrigated 23.0 -> 9.8 %.
- 10-cycle VIC spin-up as the default start (`--no-spun` to skip); recharge sanity bound by P + I + capillary rise + soil storage lost.

## 0.4.2 (2026-10-27)

Full CPU and GPU suites on fbb6c566: CPU 1847 passed, GPU 379 passed, 0 failed.

**Speed (outputs unchanged)**
- VIC-WUR full stack on ISC (112,825 cells, H200): 6.70 -> 0.13 s/day; with WOFOST-NPK 1.33 -> 0.25 s/day (WOFOST ~11x faster, peak memory 53 -> 46 GB).
- VIC step with VIC-WUR hooks on the lane path and CUDA graphs: 244 -> 15 ms/day; deterministic fixed-order sums.
- VIC energy balance: RTX 6000 ISC 2 years 305 -> 176 s, now faster than the 384-core CPU node (199 s); H200 step about 35% faster.
- Multi-GPU flood engine (opt-in), bit-identical to one GPU: Guwahati H200 2.23 -> 1.41 s/day.
- Snow: snow-free steps under a snow graph no longer redo the day (small domains up to 8.6x faster); per-cell snow thresholds.

**VIC-WUR**
- every R1 reference run end to end; restarts bitwise; VIC-WUR restart-file emulation.
- Upstream VIC-WUR land-use water leak quantified (124-214 mm/yr), fixed by default (exact mode keeps it).
- Irrigation alternate-day pulse (VIC-WUR quirk) fixed by default, exact option kept.

**VIC-CropSyst (complete)**
- Core hooks, live coupled mode inside VIC-WUR, drainage law, canopy and soil evaporation, 17 layers, sprinkler loss as ET (baseflow within -4..+6% of the reference on 6 of 7 cases).
- Orchard canopy architecture (apple) and forage clipping (alfalfa).

**India (ISC) real-world testing**
- Physical sanity checker and a permanent 31-day ISC sanity test.
- Groundwater-river coupling fixed for real topography (implicit river stage, losses limited by available water; negative discharge eliminated).
- ISC input fixes: missing roots, zero-area crop tiles, groundwater statics, compile hang.
- Known open items: ISC irrigation demand above published totals, crop mapping in cold regions, long spin-up for deep soils.

**Details (shipped in 0.4.2)**
- ISC groundwater, river exchange and the Krishna dry season (mode fixed): the river-aquifer exchange of the river cells is solved in the routing sweep (slot `river_exchange`, an implicit reach: bisection on L = C (stage(Q - L) - head) with the discharge of the same day, taken from the water in the reach, the shortfall a debt booked as unmet in the groundwater ledger; Triton kernel, CUDA-graph replay). It replaces the stage relaxation 0.2, the linearised stage and the loss cap, whose one-day lag made a two-day limit cycle. ISC 2005 after spin-up: cell-days below zero 0.39 % -> 0, day-to-day roughness at the Godavari, Krishna, Mahanadi, Narmada gauges 655/522/275/311 -> 107/28/37/21 m3/s; 20, 50 and 90 bisection steps agree to 1.4e-4 of the mean. Mode exact keeps VIC-WUR 3.0 (a losing river's forced inflow can make the discharge negative; documented and tested). `vicwur_nonrenew` `return_to='aquifer'` (opt-in, default VIC-WUR 3.0's river return): the unused part of the pumped water recharges the aquifer; then the Krishna's February-June flow at Vijayawada 700-800 -> 46-104 m3/s. `docs/connectors/vicwur.md`, `tests/test_river_exchange.py`.
- Fixed: a CUDA graph captured outside `torch.inference_mode` after `basin.simulate` had captured inside it (first capture in the process) failed in `capture_begin` with "Inplace update to inference tensor outside InferenceMode": the generator's graph state had been allocated as inference tensors (test suites run in another file order failed 17 GPU tests this way, main included). `graphsafe.ensure_generator_graph_state` makes the state outside inference mode before the first capture; `tests/test_graph_state.py`.
- Energy-balance speed (bit-identical): the surface residual's temperature-independent part evaluated once per solve, the late iterations of the bracket widening and Brent's search on the live lanes only (compact graph, timed against the full iteration), one dynamo graph for the residual, products as `mul.rn` with Triton contraction on so libdevice `pow` fuses, the evaporation re-evaluation at the root dropped (it equals the residual's), no speculative extra loop iteration for iterations over 120 us. ISC 2 years: RTX 305 -> 176 s; H200 steady step 12.6-13.7 -> 8.1-9.2 ms, run 131.5/132.9 -> 110 s (same node and day). `core/compact.py`, `tests/test_compact.py`, docs/performance.md.
- Flood engine on several GPUs (opt-in, `devices: [cuda:0, cuda:1]` on the flood connector, `FloodEngine(devices=...)`): row strips with a 2-row halo exchanged every sub-step inside one multi-GPU CUDA graph (peer stores and in-graph flag barriers); depths, fluxes, edge outflow and ledger equal the single-GPU float64 engine bit for bit (Guwahati 31 days, combo 16 one year). Two RTX 6000 Ada: Guwahati 8.01 -> 4.91 s/day (82 % scaling efficiency), combo-16 flood 8.79 -> 5.35 s/day; two H200: 2.23 -> 1.41 s/day (79 %), combo-16 flood 2.46 -> 1.53 s/day (docs/connectors/flood.md section 11). One GPU stays the default.
- WOFOST-NPK on ISC (H200): the full VIC-WUR stack 1158 -> 248 ms/day in steady state (398 over 365 days with the first day's compile), WOFOST 88 ms/day, peak GPU 53.2 -> 46.2 GB, bit-identical (docs/connectors/vicwur.md 5.5): the day on the running units only (state included), the leaf classes on a ring with eager's sum and scan trees on the live classes (Triton), fused regions with float32 division and casts kept eager's.
- VIC-WUR hooks (`vicwur_core`) on the covered tiles and inside VIC's CUDA graphs, bit-identical to the padded run on the covered tiles: ISC VIC step 243.8 -> 14.8 ms/day; full stack without WOFOST 377.7 -> 158.7 ms/day over 365 days (130.8 steady), 29.5 -> 20.9 GB; with WOFOST-NPK 1,332 -> 1,158 ms/day, 63.7 -> 53.2 GB. Coupled GPU runs are deterministic without deterministic mode: `core.ordered` replaces the GPU `index_add` atomics in the band-to-cell sums, the outlet areas and MODFLOW's river/drain flows (the forcing loads were already bitwise).
- VIC-WUR on ISC (112,825 cells, H200, 365 days): without WOFOST 6.70 s/day -> 0.38 s/day, peak GPU 110.8 -> 29.5 GB; with WOFOST-NPK out of memory -> 1.33 s/day, 63.7 GB (runs under a 78 GiB cap). Results bit-identical (docs/connectors/vicwur.md 5.4): CUDA kernels for local water use with its ring edit, the dams and the EFR search; the P2 sweep replayed from a CUDA graph; MODFLOW's deferred steps launch the chunks converged steps need (306 -> 14 ms/day); WOFOST's leaf classes kept for the units that grow, its day written in place by chunks of cells from CUDA graphs. `run.keep_outputs_on` (device, host, none); `vic: forcing_on: host`.

## 0.4.1 (2026-10-20)

- Day-coupled runtime about 3.5x faster and bit-identical: VIC in day-coupled mode 121 -> 34.7 ms/day on the H200 over ISC, host syncs 82 -> 0.85 per day (one 62-day simulate handed out a day at a time; ledger flags read once per 31 days with rollback and replay). Full systems: combo 12 (flood off) 277 -> 155 ms/day, combo 16 (flood off) 387 -> 265 ms/day.
- ISC aquifer thickness rebuilt from de Graaf et al. (unconsolidated sediments) and Pelletier et al. 2016 (elsewhere): Indo-Gangetic median 27 -> 184 m.
- Full suites on 6a4beadd: CPU 1757 passed, GPU 298 passed, 0 failed.

## 0.4.0 (2026-10-18)

The coupled system, the VIC-WUR and VIC-CropSyst packages, a faster flood
engine and fixes. Single-model behaviour is unchanged: `torchvic run`
without `--config` gives the same outputs as 0.3.0. The licence is
GPL-3.0-or-later (NOTICE, THIRD_PARTY_NOTICES.md).

### Coupled system and runtime

- Upstream bugs fixed by default, bug-exact modes kept (tables in connectors/vicwur.md and cropsyst.md): VIC-WUR
  land use (`Cs[2]` overflow and the other lu_apply quirks), pumping floor
  (`liquid`), EFR cell-0 parameters, dam area and month count; CropSyst
  mechanistic Ro (real top-layer thickness, no NaN). Replays against the
  references select `exact`. The VIC-WUR 3.0 throughfall leak is fixed
  too: `VicwurCore(throughfall='fixed')` conserves water and is the vicwur3
  default; `True`/`'exact'` keeps the leak.
- Connector API and runtime (`torchvic.coupling`): connectors register by
  name, declare inputs and outputs, and exchange fluxes through the runtime
  (dependency graph, transactional step, checkpoints and restarts); a water
  ledger books every transfer and checks closure
  (CONNECTOR_API.md, CONFIGURATION.md).
- `TorchVIC.from_config(path)`; CLI `torchvic run --config coupled.yaml` and
  `torchvic validate --config coupled.yaml [--deep]`; example configurations
  in `configs/`.
- VIC connector: `precip_add`, capped `soil_add` (also on GPU lanes),
  `soil_add_sources`, `lai` input, mapping `rest_to`, day-coupled modules
  through an optional `basin.simulate day_hook`, `StreamDriver` for
  large-domain forcing per window.
- GPU-native connectors: one host copy per step or window (`hostfetch`,
  `DeviceFlag`), deferred convergence reads; the same outputs as before on CPU
  and CUDA. netCDF inputs are read into memory and closed (HDF5 segfault); forcing
  handles are shared and reference-counted.
- The ISC combination matrix (16 combinations, 366 days on an H200); the
  full system, with and without two-way flood, closes every ledger row and
  restarts equal to the continuous run.
- Documentation for the coupled system and connectors, a user guide, tutorials and the MkDocs site.

### Connectors

- `routing` (`StepRouter`, fused Triton `FastStepRouter`), `reservoir`
  (Hanasaki 2006, rule curve, run-of-river), `quality` (water temperature
  and constituents), all in one sweep.
- `groundwater`: GPU engine of the MODFLOW 6 GWF equations (EVT, VARIABLECV
  DEWATERED), equal to MODFLOW 6 6.7.0 to 4.5e-9 m, with pumping and
  river-aquifer exchange; 0.98 ms per day on an H200.
- `irrigation` (water use, VIC-WUR method) and `crop` (WOFOST 7.2, equal to
  pcse 6.0.13 to 4e-15).
- `flood`: own GPU local-inertial engine (`torchvic/flood`, Triton), checked
  against LISFLOOD-FP 8.2, which stays an external reference backend
  (GPL-2.0-only, never linked); two-way coupling with routing and optional
  floodplain infiltration; fixed-order deterministic sums (`SegmentSum`).
- Ensembles (perturbed members as cells) and EnKF/ETKF/LETKF data
  assimilation with ledgered increments (Python engines, not registry
  connectors).
- `two_store`, a test connector.

### VIC-WUR package

- Core hooks (layer Ksat, Wcr, vegetation override, `b_co2`, land-use `Cv`
  and state redistribution, NBARE and SHARE_LAYER_MOIST checks), phase
  scheduler, step view.
- Connectors `vicwur_routing` (CascadeRouter, bitwise against the VIC-WUR 3.0 routing
  C code), `dams`, `wateruse_local`, `wateruse_remote`, `nonrenew`, `efr`;
  balance (OUT_ROUTING_ERROR, OUT_WATER_ERROR); WOFOST-NPK kernel; runoff_GWM
  with the GPU MODFLOW 6 groundwater coupling; ISC input generators.
  Level-1 checks 1e-12 against the unmodified C, level-2 replays to float32
  output precision or better (docs/connectors/vicwur.md). Opt-in `vicwur3`
  core; VIC 5 is unchanged when it is off.

### VIC-CropSyst

- Readers, uptake, phenology, growth, canopy, root, yield, daily engine and
  the crop backend `cropsyst`; nitrogen (soil N, organic matter, crop N);
  irrigation and deficit irrigation; orchard and perennial crops. Level-1
  1e-12 or better against the reference; whole-season replays on CRB cases,
  nitrogen on and off, with sowing and harvest exact; ISC-scale GPU
  benchmark (docs/connectors/cropsyst.md).

### Flood speed

- The default float64 Triton step evaluates the power and divisions on wet
  faces only, skips limiter divisions in blocks without a limited cell and
  updates on row segments, bit-identical to the previous step; combo 16 on
  the H200 3.75 to 3.18 s per day, RTX 6000 Ada 21.98 to 10.83 s per day.
  Option `dtype: mixed` (float32 faces, float64 ledger; bound in
  connectors/flood.md). Blocks are launched ahead of the host poll.
- CascadeRouter ISC speed (n=1 3.86 to 0.93 ms per step, n=4 19.9 to
  3.44 ms on an H200, bitwise kept); groundwater daily step 6 ms to under
  1 ms.

### Fixes

- Lane path: the bool refcount loss (`bool_dealloc`, torch 2.6
  `get_eval_frame_callback` borrowed reference; probe-gated own reference in
  `core/fusion.py`) is fixed, and `band_tfactor` runs on the LANE_COMPACT
  path, bit-identical to the padded path (A19, A20).
- Routing (VIC-WUR): a negative forced inflow is not clipped, as the VIC-WUR 3.0
  C code, bitwise against the unchanged C.
- Flood: the Triton kernels follow the process FMA setting; the
  order-dependent `lisflood_gpu` compat test and the LISFLOOD-FP `acc_gpu`
  SIGSEGV (about 5 percent of runs) no longer fail the suite.
- Tests restore the default device and dtype per module; `test_vicwur_day`
  uses a private scratch directory.
- Dry groundwater cells are not negative stores in `run_combos`.

## 0.3.0

Speed, GPU routing and docs; physics options as in 0.2.0, outputs of every
accelerated path bit-identical to the eager step before it. Times are the ISC
domain (112,825 cells), 2004-2005, 3-hourly, one H200 with 4 CPU cores,
against VIC 5.1.0 (384 MPI ranks, one band per cell) on a compute node
(docs/performance.md).

- Full energy balance in CUDA graphs: `FULL_ENERGY` (QUICK_FLUX, FROZEN_SOIL,
  QUICK_FLUX FALSE explicit and IMPLICIT) on the lanes, with the surface
  solve in fused regions. Energy balance, H200: 108 s against 199 s for VIC;
  RTX 6000 Ada 305 s. On 112,680 cells the eager step took 3256 s on the H200
  (6101 s RTX); the output is bit-identical. Eager by design: CLOSE_ENERGY,
  LAKES, daily sub-steps, QUICK_FLUX FALSE with FROZEN_SOIL or QUICK_SOLVE,
  SPATIAL_SNOW, BLOWING, CARBON.
- P3 kernel fusion (`torch.compile` regions inside the CUDA graph, FP
  contraction off) and P4 fused step with per-cell tile sums in eager CUDA
  order: water balance on the H200 from 179 s to 61 s.
- Snow-lane compaction: the snow and canopy stages run on the window's snow
  lanes only (solve_snow.c:117-150 gates).
- Water balance on the 112,825-cell domain takes
  55.7 s on the H200 (VIC 191 s, 3.4x), 99.2 s on the RTX 6000 Ada, 119.8 s
  with daily forcing and MTCLIM on the H200.
- `IMPLICIT TRUE` soil temperature: a compiled per-column Newton loop on the
  CPU (4.5 to 0.12 s per step, VIC C 0.15 s) and one NVRTC CUDA kernel per
  profile solve on the GPU (14-day rows 0.76-1.75 to 0.16-0.23 s per step),
  both bit-identical.
- Multi-GPU: `torchvic run --gpus N` and `run_isc_torchvic.py --gpus N`, cells
  split over N processes, history and state merged, outputs equal the
  single-GPU run (torchvic/parallel). Feed I/O: the forcing is read once by
  reader processes into shared memory windows, parameters inflated by a
  process pool, shards forked, history written in parallel (112,680 cells:
  1 GPU 59.7 to 51.7 s, 2 GPUs 69.9 to 51.6 s).
- Zarr forcing input (single store or per year), bit-identical to NetCDF.
- GPU river routing (RVIC source lists, Lohmann channel routing, nested
  gauges) for the ISC network, 2 years in 0.09 s on an H200. Flow directions
  from HydroSHEDS 15s with transplants from MERIT-Hydro and GRIT, upscaled to
  1/16 degree (tools/); all 372 gauges are routable: 358 natural, 10 sub-cell
  (area-weighted cell runoff) and 4 canal gauges, flagged.
- ISC domain 112,825 cells (was 112,680); benchmark global files for the
  energy balance (`benchmarks/isc/isc_vic5_3h_eb.txt`).
- Documentation site (MkDocs: install, usage, options, VIC differences,
  process-to-C mapping, verification, performance).
- Energy to solution of the ISC run: GPU power by `nvidia-smi`, CPU share and
  VIC estimated (RAPL, perf and IPMI need root); docs/performance.md,
  `benchmarks/results_energy`.

## 0.2.0

Option coverage of VIC 5's image driver, each option checked against the
pristine VIC 5.0.1 / 5.1.0 `vic_image.exe` (matrix rows in `verification/`;
numbers in docs/OPTIONS.md). The version string lives in
`torchvic/__init__.py`; `pyproject.toml` reads it.

Options implemented (previously refused by name):

- `SNOW_BAND TRUE` with several elevation bands per cell.
- `AERO_RESIST_CANSNOW` `AR_410` (and `AR_406_LS`).
- `NIJSSEN2001_BASEFLOW`, `ARNO_PARAMS`.
- `GRND_FLUX_TYPE GF_406`.
- `FROZEN_SOIL` with `FULL_ENERGY FALSE` (water balance mode) and
  `SPATIAL_FROST`.
- `CLOSE_ENERGY`.
- `SPATIAL_SNOW`.
- `SNOW_DENSITY DENS_SNTHRM` (SNTHERM89 densification, snow_utility.c).
- `ORGANIC_FRACT`, `BULK_DENSITY_COMB`, `MAX_SNOW_ALBEDO`, `CORRPREC`.
- `RC_MODE RC_PHOTO` (canopy_evap.c, penman.c).
- `TFALLBACK FALSE`: the run stops at the first failed temperature solve.
- `FORCING2` and `LAI_SRC`/`FCAN_SRC`/`ALB_SRC FROM_VEGHIST`
  (vegetation history forcing).
- `QUICK_FLUX FALSE`: finite-difference soil temperature profile, with
  `IMPLICIT`, `EXP_TRANS`, `NOFLUX`, `QUICK_SOLVE` and the node grid.
- `LAKES TRUE` (lake energy and water balance, checked on Findley Lake).
- `COMPUTE_TREELINE`, `OUTPUT_FORCE`, `ATMOS_STEPS_PER_DAY` run instead of
  being refused; `torchvic run` reads a classic-driver global file.

Also in this release: refused-by-name options that other drivers run
(`COMPUTE_TREELINE` with `JULY_TAVG_SUPPLIED`, `OUTPUT_FORCE`,
`ATMOS_STEPS_PER_DAY`) run, the classic driver's global file is read, and
the lake and finite-difference soil temperature paths are verified.

Scoreboard (verification/, pristine VIC 5.0.1 / 5.1.0, graded at 1e-4,
exact-math mode): supported rows 71 pass, 32 warn, 0 fail, 4 other; 15 classic-driver
rows pass. Every VIC 5.1.0 image-driver option is implemented or has no
effect (docs/OPTIONS.md: 82 options, 0 refused).

Changes to existing behaviour:

- Drainage between layers is VIC's literal `calc_Q12` (runoff.c:478-481).
- A daily model step sub-steps a snow day at `SNOW_STEPS_PER_DAY`
  (surface_fluxes.c:258-269).
- Grades use a 1e-8 mm absolute floor in the denominator for mm variables.
- Shared physics rounds in VIC's C order: surface energy residual,
  stability correction, aerodynamic resistance, evaporation and
  transpiration (scaled before the root split), VIC's own root-fraction
  walk, snow_melt's ice and cold content carried, precipitation blending,
  and `number / tensor` written as IEEE division. With it the Himalaya
  daily rows (EB and WB) pass on every variable (<= 1.5e-12).
- Exact-math mode (`torchvic run --exact-math`, `TORCHVIC_EXACT_MATH=1`,
  `exact_math: true` in a study file): glibc pow/exp/sqrt on the CPU at the
  sites that feed VIC's thresholds (torchvic/core/xmath.py). Official
  verification rows run in it; the default and GPU paths are unchanged
  and may differ from it in the last bit.
- The implicit finite-difference soil temperature solve (QUICK_FLUX FALSE,
  IMPLICIT TRUE) is batched over nodes and Jacobian columns, bit for bit;
  the 12-node implicit rows now run 14 days.
- `torchvic convert-classic` turns classic-driver soil, vegetation and
  snow band files into image-driver `domain.nc` and `params.nc`.

Still refused: a split run with `LAKES TRUE` until the lake state is carried.
`INIT_STATE` and `STATENAME` are implemented (state files in every configuration).

Tests and CI: slow and reference-dependent tests skip when VIC trees or
sample data are absent; a GitHub Actions workflow runs the data-free subset
on CPU torch.

## 0.1.0

First release: the forward VIC 5 model as a package.

- VIC 5.0.1 and 5.1.0 core physics (`--core`): snow, canopy, vegetation
  tiles, soil column, surface energy balance, frozen soil (QUICK_FLUX),
  blowing snow, carbon, spatial snow, Lohmann/RVIC routing.
- Native VIC 5 image-driver inputs; sub-daily forcing, or daily forcing
  disaggregated by MTCLIM; the classic driver's ASCII inputs are read too.
- `FROZEN_SOIL`, `BLOWING`, `CARBON`, `CANOPY_LAYERS` and
  `SHARE_LAYER_MOIST` are read from the global parameter file; options
  selecting physics that is not implemented are refused by name.
- VIC 5 NetCDF history (OUTFILE streams, every AGGFREQ and AGG_TYPE), and
  VIC 4.2.d's per-cell files on request.
- `torchvic run` command; forward runs execute under
  `torch.inference_mode()`.
