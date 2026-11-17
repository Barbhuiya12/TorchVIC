# Testing

```bash
pip install -e .[test]
pytest tests                                              # all; skips what lacks data
pytest -m "not reference and not slow and not gpu" tests  # what CI runs
pytest -m gpu tests                                       # needs a CUDA device
```

Markers: `slow`, `reference` (needs VIC's source trees or executables), `gpu`.
A test whose data is absent is skipped and says which.

On an RTX 6000 Ada with torch 2.6.0, running the GPU tests in parallel workers
(`-n 4`) can fail up to six tests of `test_fusion.py`, `test_graph_capture.py`
and `test_lane_refcount.py`, also with 0.4.7. They pass when run alone with an
empty compile cache (`TORCHVIC_COMPILE_CACHE` pointing to a new directory), and
with torch 2.7.1.

## Test data

Tests that need VIC's trees, oracles or sample data find them through
environment variables:

| variable | points to |
|---|---|
| `TORCHVIC_REFERENCE_DIR` | pristine VIC releases (5.0.1, 5.1.0, 4.2.d) with their image drivers built |
| `TORCHVIC_SAMPLE_DATA` | UW-Hydro's `VIC_sample_data` (Walnut Gulch, Findley Lake, ...) |
| `TORCHVIC_SAMPLES` | VIC's own Stehekin samples |
| `TORCHVIC_ORACLE_DIR` | where the small C oracle programs are compiled |
| `TORCHVIC_DEVICE` | device the tests use (for example `cpu` or `cuda:0`) |

`tools/get_test_data.sh` downloads the three VIC releases, builds the image
drivers (it needs `nc-config` and `mpicc` on the `PATH`) and clones the sample
data into `reference/` and `data/`:

```bash
tools/get_test_data.sh                               # download and build
tools/get_test_data.sh --link REF SAMPLE_DATA SAMPLES   # link existing copies
```

See [Sample data](../downloads/data.md).

## Verification matrix

The matrix against VIC 5.0.1 and 5.1.0 is run from `verification/`
([Verification](../verification.md)).
