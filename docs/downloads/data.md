# Sample data

TorchVIC ships no data. The examples and tests use public data sets.

| data | source | used for |
|---|---|---|
| `VIC_sample_data` (Walnut Gulch, Stehekin, Findley Lake) | <https://github.com/UW-Hydro/VIC_sample_data> | [Quick start](../quickstart.md), tutorials, tests |
| VIC releases 5.0.1, 5.1.0 and 4.2.d | <https://github.com/UW-Hydro/VIC/releases> | reference runs and oracles in the tests |

```bash
git clone --depth 1 https://github.com/UW-Hydro/VIC_sample_data.git
export TORCHVIC_SAMPLE_DATA=$PWD/VIC_sample_data
```

The sample's global file names its inputs as `${VIC_SAMPLE_DATA}/...`; TorchVIC
resolves `${VIC_SAMPLE_DATA}` from `TORCHVIC_SAMPLE_DATA` (or `VIC_SAMPLE_DATA`)
or from `--sample-data DIR`.

`tools/get_test_data.sh` fetches everything the test suite needs: the VIC
releases (the tags are `VIC.5.0.1`, `5.1.0` and `VIC.4.2.d`), whose image drivers
it builds, and the sample data ([Testing](../development/testing.md)). It needs
`nc-config`, `mpicc`, `curl` and `git` on the `PATH`. To build input files for
your own domain, see [Your own domain](../tutorials/own-domain.md).

Datasets that the tools and connectors read (VICGlobal parameters, ERA5,
HydroSHEDS, MERIT-Hydro, dam and irrigation databases) are not distributed with
TorchVIC and keep their own terms; they are credited in
[How to cite](../about/citing.md#data-used-by-the-tools).
