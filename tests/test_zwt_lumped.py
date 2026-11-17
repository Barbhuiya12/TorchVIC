"""The lumped water table, against pristine VIC 5.0.1.

`column.zwt_moisture_curve` builds one zwt-versus-moisture curve per soil
layer, which is what OUT_ZWT reads. VIC builds a second kind: the whole
column filled from the bottom up (soil_moisture_from_water_table.c:155-224).
That row is what OUT_ZWT_LUMPED reports, and it is also what the soil carbon
cycle needs -- soil_carbon_balance.c:93 takes every thermal node's relative
moisture from `cell->zwt_lumped`, so the pools cannot be stepped without it.

The oracle is `ref_snow zwtlump`, which calls upstream's own
soil_moisture_from_water_table() on six soil columns.
"""
import os
import pathlib
import subprocess

import pytest
import torch

from torchvic.core import column

import _paths                                           # noqa: E402

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')
REF = pathlib.Path(_paths.oracle('ref_snow'))
DT = torch.float64
#: The live default; see test_image_physics.py for why this is not 'cpu'.
DEV = torch.empty(0).device


class _Soil(object):
    """Only the fields the curve reads."""
    def __init__(self, depth, expt, bubble, max_moist, resid_moist):
        self.depth, self.expt, self.bubble = depth, expt, bubble
        self.max_moist, self.resid_moist = max_moist, resid_moist


@pytest.fixture(scope='module')
def oracle():
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'zwtlump'], capture_output=True,
                         text=True, check=True).stdout
    layers, curves, queries = {}, {}, {}
    for line in out.splitlines():
        if line.startswith('#'):
            continue
        if '|' in line:
            head, tail = line.split('|')
            c = int(head.split(',')[0])
            curves.setdefault(c, []).append(
                tuple(float(x) for x in tail.split(',')))
        elif ';' in line:
            head, tail = line.split(';')
            c = int(head.split(',')[0])
            queries.setdefault(c, []).append(
                tuple(float(x) for x in tail.split(',')))
        else:
            v = [float(x) for x in line.split(',')]
            layers.setdefault(int(v[0]), []).append(v[1:])
    cases = sorted(layers)
    t = lambda col: torch.tensor([[r[col] for r in layers[c]] for c in cases],
                                 dtype=DT, device=DEV)
    soil = _Soil(depth=t(1), expt=t(2), bubble=t(3), max_moist=t(4),
                 # VIC's resid_moist is a fraction; ours is the depth-weighted
                 # millimetres the rest of column.py uses.
                 resid_moist=t(5) * t(1) * 1000.0)
    ref_z = torch.tensor([[x[0] for x in curves[c]] for c in cases],
                         dtype=DT, device=DEV)
    ref_m = torch.tensor([[x[1] for x in curves[c]] for c in cases],
                         dtype=DT, device=DEV)
    q_moist = torch.tensor([[x[0] for x in queries[c]] for c in cases],
                           dtype=DT, device=DEV)
    q_zwt = torch.tensor([[x[1] for x in queries[c]] for c in cases],
                         dtype=DT, device=DEV)
    return soil, ref_z, ref_m, q_moist, q_zwt


def test_lumped_curve_matches_vic(oracle):
    soil, ref_z, ref_m, _qm, _qz = oracle
    z, m = column.zwt_lumped_curve(soil)
    assert float((z - ref_z).abs().max()) < 1e-10
    rel = (m - ref_m).abs() / ref_m.abs().clamp(min=1e-9)
    assert float(rel.max()) < 1e-11, float(rel.max())


def test_the_curve_is_monotonic_and_starts_saturated(oracle):
    """Sample 0 is the table at the surface, so the column holds its whole
    max_moist; every later sample lowers the table and holds less."""
    soil, _z, _m, _qm, _qz = oracle
    z, m = column.zwt_lumped_curve(soil)
    assert float((m[:, 0] - soil.max_moist.sum(-1)).abs().max()) < 1e-9
    assert bool((z[:, 1:] < z[:, :-1]).all())
    assert float(z[:, 0].abs().max()) == 0.0


def test_it_is_not_the_per_layer_curve(oracle):
    """The distinction this exists for. Lumped, the layers above the water
    table are drawn against an *effective* table (:206-209) -- the depth that
    gives the same suction through their own pore-size distribution -- so a
    sand-over-clay column has one capillary fringe per horizon, not one. A
    caller that reached for the per-layer curve instead would get a different
    number, and silently.
    """
    soil, _z, _m, _qm, _qz = oracle
    _zl, ml = column.zwt_lumped_curve(soil)
    _zp, mp = column.zwt_moisture_curve(soil)
    # The per-layer curves summed are not the column curve.
    assert float((ml - mp.sum(1)).abs().max()) > 1.0


def test_reading_the_curve_matches_vics_own_compute_zwt(oracle):
    """compute_zwt.c:41-70 against the same row, over the whole moisture
    range and past the top of it.

    This is where the non-monotonicity bites. The C walks *down* from the
    last sample while the moisture is still covered, so it stops at the end
    of a trailing run; counting every covered sample anywhere in the table
    gives a different index as soon as the curve turns back on itself, which
    the lumped curve does whenever a coarse horizon sits under a fine one.
    The per-layer curves never do, which is why nothing caught it before.
    """
    soil, _z, _m, q_moist, q_zwt = oracle
    n = q_moist.shape[-1]
    for k in range(n):
        moist = torch.zeros(q_moist.shape[0], 3, dtype=DT, device=DEV)
        moist[:, 2] = q_moist[:, k]
        got = column.compute_zwt_lumped(moist, soil)
        ref = q_zwt[:, k]
        # 999 is the C's "no water table here"; compute_zwt_lumped turns it
        # into the column base, as compute_zwt.c:134-135 does.
        base = -(soil.depth * 100.0).sum(-1)
        ref = torch.where(ref == 999.0, base, ref)
        # Relative, because VIC's own curve has a singularity and this test
        # keeps a column that hits it: with expt below 5 the exponent
        # (b-1)/b is negative, so the capillary term diverges as its argument
        # approaches zero and the tabulated moisture reaches 1e6 mm. That is
        # VIC's arithmetic, reproduced rather than smoothed, and comparing it
        # absolutely would only be measuring the size of the numbers.
        rel = (got - ref).abs() / ref.abs().clamp(min=1.0)
        assert float(rel.max()) < 1e-10, (k, float(q_moist[0, k]))


def test_a_dry_column_reports_its_own_base_not_the_sentinel(oracle):
    """compute_zwt.c:134-135 -- 999 becomes the depth of the column. The soil
    carbon cycle downstream reads this without testing for a sentinel, so a
    999 leaking through would be a water table 10 metres in the air."""
    soil, _z, _m, _qm, _qz = oracle
    dry = torch.zeros(soil.depth.shape[0], 3, dtype=DT, device=DEV)
    got = column.compute_zwt_lumped(dry, soil)
    base = -(soil.depth * 100.0).sum(-1)
    assert float((got - base).abs().max()) < 1e-9
    assert float(got.max()) < 0.0


def test_the_curve_carries_a_gradient(oracle):
    """The water table is a function of calibrated soil parameters; a run
    optimising them has to be able to differentiate through it."""
    soil, _z, _m, _qm, _qz = oracle
    b = soil.bubble.clone().requires_grad_(True)
    s2 = _Soil(soil.depth, soil.expt, b, soil.max_moist, soil.resid_moist)
    _z2, m2 = column.zwt_lumped_curve(s2)
    m2.sum().backward()
    assert torch.isfinite(b.grad).all()
    assert float(b.grad.abs().max()) > 0.0
