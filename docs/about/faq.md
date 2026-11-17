# Frequently asked questions

Here are some commonly asked questions and their answers. Questions about VIC
itself are best asked in the [VIC project](https://github.com/UW-Hydro/VIC);
questions about TorchVIC go to the [issue tracker](contact.md).

1.  **Do I need a GPU?**

    No. `--device cpu` runs the same program on a CPU. A GPU pays off for large
    domains: for the 112,825-cell Indian-subcontinent domain one H200 runs a
    simulated year 2.88 times (water balance) and 1.97 times (energy balance)
    faster than the best layout of VIC on a full 384-core CPU node. VIC is
    faster for domains of 20,000 cells or fewer, and on a CPU TorchVIC is slower
    than VIC ([Performance](../performance.md)).

2.  **Is TorchVIC the same model as VIC?**

    It computes the physics of VIC 5.1.0 (the default) or VIC 5.0.1
    (`--core 5.1.0` or `--core 5.0.1`) and agrees with the compiled image
    drivers to round-off. It is not bit-identical
    ([Verification](../verification.md)). Where it departs from VIC on purpose,
    the departure is listed in [Differences from VIC](../differences.md).

3.  **Which core should I use?**

    Use VIC 5.1.0, the default (`--core 5.1.0` only makes it explicit). It is
    the release that the verification and the timings of these pages are for:
    four of the six whole-domain comparisons are with VIC 5.1.0 and two with
    VIC 5.0.1. Pass `--core 5.0.1` to compare with a run of VIC 5.0.1. The two
    releases differ, among other things, in the exact integral of the drainage
    between soil layers, a cap on the density of new snow and the reference
    heights of the land surface.

4.  **Why float64? Can I use float32?**

    VIC is a double-precision model and the agreement with it is established
    in float64, which is the default. `--dtype float32` is available but is not
    recommended: on the whole domain it changes the domain-mean runoff by 2.6 %
    ([Known limitations](limitations.md)).

5.  **Is the output bit-identical to VIC's?**

    No. Between 62 and 79 % of the compared values are bit-identical in
    float64 and 97.6 to 99.0 % are identical after rounding to single
    precision. The differences come from the order of floating-point sums and
    the last bit of library functions; see
    [Differences from VIC](../differences.md).

6.  **Are two runs identical?**

    Yes, in the configurations tested. Two runs of one configuration on one
    H200, and runs on an H200 and an RTX 6000 Ada, give bit-identical output
    (float64, tested for the water balance and the energy balance over two
    years). One exception is known: a water-balance run with snow bands can
    differ between two GPU runs in the last bit of double-precision values
    ([Known limitations](limitations.md)).

7.  **My run was refused. Why?**

    An option or input is one TorchVIC does not run as VIC does. The message
    names it. See [Troubleshooting](../troubleshooting.md) and
    [Options](../OPTIONS.md).

8.  **Can I run my VIC classic setup?**

    Yes: `torchvic convert-classic` turns soil, vegetation and snow-band files
    into `domain.nc` and `params.nc` ([Prepare your inputs](../guide/inputs.md)).

9.  **Can I use daily forcing?**

    Yes. Daily maximum and minimum temperature and precipitation are
    disaggregated to the model step by the MTCLIM algorithms on the device.
    VIC 5's image driver cannot read daily forcing.

10. **Does it write the same files as VIC?**

    It writes VIC 5 history and state files, in the layout of VIC 5.1.0's image
    driver, and reads VIC's state files ([Outputs](../guide/outputs.md)). Reading
    a VIC state file on a GPU has a known problem in 0.4.8, with a workaround
    ([Known limitations](limitations.md)).

11. **Does TorchVIC reproduce the bugs of VIC?**

    Where that is meaningful, yes: TorchVIC computes what VIC computes, and
    its agreement with VIC is with VIC as released. Where reproducing a defect
    would not be meaningful (for example a routing that loses water) the
    departure is deliberate and documented. See
    [Known issues in VIC 5](vic5-issues.md).

12. **Why is the first GPU run on a new domain slow?**

    The GPU path compiles fused kernels for a new cell count and caches them.
    Even with warm caches TorchVIC pays a fixed
    start-up and set-up cost (22.4 s for the water balance and 30.2 s with the
    energy balance on a 1,000-cell domain, against about 13 s for VIC), which
    is why VIC is faster on small domains.

13. **Which GPU do I need?**

    The model runs in double precision, so double-precision throughput
    matters. An H200 (141 GB) and an RTX 6000 Ada (48 GB) were tested. The
    whole 112,825-cell domain needs about 34.9 GB with the water balance and
    42.8 GB with the energy balance; memory grows linearly with the number of
    cells (309 and 379 kB per cell).

14. **Does it run on Windows or macOS?**

    The tests and timings were made on Linux (Ubuntu 24.04). Other platforms
    have not been tested.

15. **Does TorchVIC calibrate the model?**

    No. It provides batched ensembles and goodness-of-fit metrics
    ([Study files](../guide/study.md)); calibration is done outside it.

16. **How do I cite it?**

    See [How to cite](citing.md).
