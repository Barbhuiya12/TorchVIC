# Differences from VIC

TorchVIC computes what VIC computes, including VIC's defects where reproducing
them is meaningful. It is compared with the compiled VIC 5.0.1 and 5.1.0 image
drivers ([Verification](verification.md)). This page lists what still differs:
the round-off that remains, the places where TorchVIC departs from VIC on
purpose, and how options are treated.

## Round-off, not bit-identity

TorchVIC and VIC evaluate the same expressions, but the two programs do not
produce the same bits. Three sources were identified and are not removed:

1. **The order of floating-point sums.** TorchVIC sums over time and over cells
   in a fixed order (ordered segment sums, not atomic additions); the C code
   adds in its own order. Sums of five to eight products, added from left to
   right as the C code does, differ from `torch.sum` in 35 to 44 % of cases on
   both devices (up to four units in the last place).
2. **Library functions.** PyTorch's `pow`, `exp` and `log` differ from the C
   library's results in the last bit for a share of the arguments. In
   micro-tests with 4 million random arguments `pow` differs in 6.0 % of the
   arguments on the CPU and 25.9 % on the GPU, `exp` in 4.5 and 8.3 %, `log` in
   0.34 and 0.53 %, and `sqrt` in 1.4 % on the CPU and never on the GPU.
3. **Fused multiply-add.** A VIC binary compiled with contraction into fused
   multiply-add operations differs from the reference build of VIC itself, a
   difference of the same kind as that between TorchVIC and VIC. TorchVIC
   disables the contraction in its fused GPU kernels. VIC compiled at `-O0` is bit-identical to the `-O3`
   reference in all 88 variables, so the optimisation level does not matter.

A last-bit difference grows into a visible one only where VIC compares a
quantity with a threshold, for example a snow pack of 1e-16 mm or a frozen
fraction, and the comparison flips a branch for one step.

The table shows the effect on a subset of 7,154 cells of the
Indian-subcontinent domain (2004 to 2005): every cell with a soil-moisture
difference above 1e-9 mm in the water balance or a sensible-heat difference
above 1e-6 W m-2 in the energy balance, plus 3,000 cells chosen at random. The
shares are therefore lower than those of the whole domain. The reference is
VIC 5.1.0 compiled at `-O3`.

| compared with VIC at `-O3` | water balance: values identical in float32, largest daily soil-moisture difference | energy balance: the same |
|---|---|---|
| VIC at `-O0` | 100.0 %, 0 mm | 100.0 %, 0 mm |
| VIC with FMA contraction | 97.5 %, 0.535 mm | 94.4 %, 0.818 mm |
| TorchVIC, CPU | 98.2 %, 0.451 mm | 94.5 %, 2.73 mm |
| TorchVIC, CPU, exact-math | 98.7 %, 0.384 mm | 97.6 %, 1.31 mm |
| TorchVIC, GPU (H200) | 95.0 %, 3.48 mm | 92.1 %, 3.5 mm |

The GPU run differs from VIC by more than the CPU run does, and in the largest
daily difference by more than two builds of VIC differ from each other (by
about a factor of six in the water balance). The GPU evaluates `pow` and `exp`
differently from the C library and adds in another order. The whole-domain
comparison, with the largest annual differences, is on the
[Verification](verification.md) page.

### Exact-math mode

`--exact-math` (or `TORCHVIC_EXACT_MATH=1`) is a verification mode for the CPU
in float64. At the sites that feed VIC's thresholds it evaluates `pow`, `exp`
and `sqrt` with the C library, as VIC gets them. On the subset above it raises
the share of identical single-precision values from 98.2 to 98.7 % in the
water balance and from 94.5 to 97.6 % in the energy balance, and lowers the
largest daily soil-moisture difference from 0.451 to 0.384 mm and from 2.73 to
1.31 mm. The default CPU path and the GPU path differ from it only in the last
bit.

### Repeatability

TorchVIC's own results are repeatable. Two runs of one configuration on one
H200, and runs on an H200 and an RTX 6000 Ada, are bit-identical in all 89
(water balance) and 92 (energy balance) output variables compared. Two runs on
the same device with the same inputs gave identical files in these tests; the
one exception found is in [Known limitations](about/limitations.md).

## Where TorchVIC departs from VIC

### OUT_ENERGY_ERROR

By default TorchVIC reports in `OUT_ENERGY_ERROR` the residual of its own
surface energy balance at the solved temperature, which closes. VIC 5 forms
the output from the reported fluxes of different control volumes (canopy,
snow pack, ground), so its value contains an imbalance that is part of VIC's
definition. On the whole-domain energy-balance run VIC reports a largest value
of 111.98 W m-2 with 7,557 cells above 1 W m-2; TorchVIC reports 0.0169 W m-2
and no cell above 1 W m-2. The two reported values are therefore not
comparable. When the energy balance is recomputed from the output fluxes by one
formula for both programs the largest error is 111.99 W m-2 for TorchVIC and
111.98 W m-2 for VIC: TorchVIC's fluxes carry the same imbalance as VIC's.

`--energy-error vic` (`energy_error='vic'` in Python) forms the output from the
reported rows as VIC 5 does (`put_data.c`).

### Water balance closure

In VIC the water balance P - E - R - B - dS is not closed in some cells (the
largest daily residual in a cell is 0.88 mm, or 0.79 mm with five bands, and
7,287 to 7,290 cells exceed 0.01 mm on some day). TorchVIC reproduces this non-closure in the same cells
and does not add to it.

### Reference height of the snow surface (`WIND_H`)

VIC 5's image driver gives each vegetation class its own reference height for
the wind from the parameter file, and uses `WIND_H` of the global file only for
lakes and the precipitation correction. TorchVIC gives the snow surface of the
tiles without an overstory one height. The rules are:

1. `WIND_H` is in the global file: that value is used, and the log says so if
   the parameter file's heights differ.
2. `WIND_H` is absent and the parameter file gives every such tile the same
   height: that height, which is VIC's.
3. `WIND_H` is absent, the heights differ and no step of the case has air below
   3 degC, so that no snow can form: the height is never read, and VIC's
   default of 10 m is recorded as inert.
4. Otherwise the run is refused with a message asking for `WIND_H`.

Declare `WIND_H` (the measurement height of the forcing wind) in the global
file, as in the example of [Prepare your inputs](guide/inputs.md).

### Variables not written

In the water-balance configurations VIC writes three variables that TorchVIC
does not write (`OUT_ADV_SENS`, `OUT_BARESOILT` and `OUT_FUSION`); VIC's values
for them are zero.

### Options

Every value of every option of VIC 5.1.0's image driver that VIC accepts is run
as VIC runs it, or has no effect on the result and is announced in the log
([Options](OPTIONS.md)). The exceptions and notable cases are:

* `STATE_FORMAT` has no effect: state files are always NetCDF4. `ROUT_PARAM`,
  `LAKE_NODES`, `LOG_DIR` and `ATMOS_STEPS_PER_DAY` are read and have no
  effect.
* A `CONSTANTS` file may only set `VEG_LAI_WATER_FACTOR`; any other constant is
  refused by name.
* A value that is not in the options table, and a value that VIC itself
  rejects, is refused by name or stops the run as it stops VIC. Three values
  that VIC rejects are accepted without a message
  ([Known limitations](about/limitations.md)).
* `AGG_TYPE_MAX` and `AGG_TYPE_MIN` start from 0 as VIC 5's do;
  `--true-extremes` gives the period's own extremes.

The probes behind this classification are listed in the
[verification](verification.md#options-and-probes) page.

## Additions to VIC 5

* **Daily forcing.** The forcing types `TMAX` and `TMIN` (the names VIC 4.2
  used for daily extremes) mark a forcing as daily; TorchVIC then disaggregates
  it on the device with the MTCLIM algorithms and reads 90 days beyond each end
  of the run window. VIC 5's image driver does not read daily forcing. The
  disaggregation is re-implemented from MetSim 2.0.1; its differences from the
  MTCLIM code of VIC 4.2.d are listed in the
  [change log](https://github.com/Barbhuiya12/TorchVIC/blob/main/CHANGELOG.md).
* **Physics release.** VIC 5.1.0 is the default core and the release that the
  verification is for (`--core 5.1.0`). `--core 5.0.1` selects the physics of
  VIC 5.0.1.
* **Precision and device.** `--dtype` and `--device` choose what VIC fixes at
  compile time or does not have.
* **Per-cell snow thresholds.** VIC 5's `MAX_SNOW_TEMP` and `MIN_RAIN_TEMP` are
  global scalars, and so are TorchVIC's `t_max_snow` and `t_min_rain` by
  default. `basin.simulate` also takes either as a `[cell]` tensor, for a
  per-basin or per-cell rain and snow ramp. The scalar run is unchanged bit for
  bit; a `[cell]` tensor holding the scalar gives the scalar run bit for bit,
  and a mixed tensor gives each cell the run with its own scalar
  (`tests/test_snow_thresholds_cell.py`).
