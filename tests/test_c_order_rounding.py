"""Bit-level agreement with VIC 5.1.0's own order of operations.

Each case is a formula whose value TorchVIC must round exactly as the C
does (himal daily rows: an ulp in any of these was traced to a
threshold flip downstream). The reference value is the C expression
evaluated in Python doubles, left to right as the C source writes it.
"""
import math

import torch

from torchvic.core import atmos, column, energy, vegetation

T = lambda x: torch.tensor([x], dtype=torch.float64)  # noqa: E731


def test_svp_slope_is_an_ieee_quotient():
    # svp.c 5.1.0:40-41; torch's `number / tensor` is reciprocal-multiply
    ta = -3.0968422591686249
    c = (17.269 * 237.3) / ((237.3 + ta) * (237.3 + ta))
    assert (atmos.svp_slope(T(ta)).item()
            == c * atmos.svp(T(ta)).item())


def test_stability_correction_c_order():
    # StabilityCorrection.c 5.1.0:34-47
    def c(Z, d, TS, Ta, W, Z0):
        tk = 273.15
        Ri = (energy.CONST_G * (Ta - TS) * (Z - d)
              / (((Ta + tk) + (TS + tk)) / 2.0 * W * W))
        lim = (Ta + tk) / (((Ta + tk) + (TS + tk)) / 2.0
                           * (math.log((Z - d) / Z0) + 5))
        Ri = min(Ri, lim)
        if Ri > 0.0:
            return (1 - Ri / 0.2) * (1 - Ri / 0.2)
        return math.sqrt(1 - 16 * max(Ri, -0.5))
    for ts, ta, w in ((-6.3481542696352085, -5.615902423858643,
                       1.1863331024140582),
                      (-3.3476857590620868, -12.667128562927246,
                       1.1900214216459686)):
        got = energy.stability_correction(T(10.0), T(0.0054471544715447165),
                                          T(ts), T(ta), T(w), T(0.001))
        assert got.item() == c(10.0, 0.0054471544715447165, ts, ta, w,
                               0.001)


def test_root_fractions_follow_vics_walk():
    # calc_root_fraction.c 5.1.0:56-177: himal tile, overlap form 1 ulp off
    zd = torch.tensor([[0.1, 0.6, 0.7]], dtype=torch.float64)
    zf = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
    ld = torch.tensor([[0.256, 0.5, 1.2]], dtype=torch.float64)
    got = vegetation.root_fractions(zd, zf, ld)[0].tolist()
    ref = vegetation._root_walk([0.1, 0.6, 0.7], [0.2, 0.5, 0.3],
                                [0.256, 0.5, 1.2])
    assert got == ref
    assert abs(sum(got) - 1.0) < 1e-12


def test_arno_bare_soil_round_trips_like_the_c():
    # arno_evap.c 5.1.0:84-139, with the ratio's round trips
    moist, mx, b, epot = 30.0, 70.0, 0.3, 1.0
    ratio = (1.0 - moist / mx) ** (1.0 / (b + 1.0))
    mi = (1.0 + b) * mx
    tmp = mi * (1.0 - ratio)
    r = 1.0 - tmp / mi
    r = r ** b
    a_s = 1 - r
    r = r ** (1.0 / b)
    tmpsum = 1.0
    dummy = 1.0
    for k in range(1, 31):
        tmpsum *= r
        dummy += b * tmpsum / (b + k)
    ref = epot * (a_s + (1.0 - a_s) * (1.0 - r) * dummy)
    got = column.bare_soil_evap(T(epot), T(moist), T(mx), T(b), T(0.3),
                                T(0.0), resid_mm=T(0.0))
    assert abs(got.item() - ref) <= 2 * math.ulp(ref)
