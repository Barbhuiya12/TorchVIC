"""snow_intercept.c's mass part (:143-289) against a scalar transcription of the
C, and mass closure: what the canopy held plus what fell equals what it holds
plus what fell through (F = 1)."""

import torch

from torchvic.core import canopy as C  # noqa: E402

D = torch.float64
CPU = torch.device('cpu')
EPS = 2.220446049250313e-16


def c_mass(rain, snow, int_rain, int_snow, tf, lai, max_int, wind1, f=1.0, thres=0.001):
    mult = C.VEG_LAI_SNOW_MULTIPLIER
    cap = C.SNOW_LIQUID_WATER_CAPACITY
    imax1 = 4.0 * mult * lai
    if tf < -1.0 and tf > -3.0:
        msi = tf * 3.0 / 2.0 + 11.0 / 2.0
    elif tf > -1.0:
        msi = 4.0
    else:
        msi = 1.0
    msi *= mult * lai
    if msi > 0:
        d = (1 - int_snow / msi) * snow
        if d + int_snow > msi:
            d = msi - int_snow
        if d < 0.0:
            d = 0.0
    else:
        d = 0.0
    if tf < -3.0 and d > 0.0 and wind1 > 1.0:
        b = (0.2 * wind1 - 0.2) * d
        if b >= d:
            b = d
        d -= b
    if int_snow + d > imax1:
        d = 0.0
    snow_through = (snow - d) * f + snow * (1 - f)
    if snow == 0 and int_snow < thres:
        snow_through += int_snow
        d -= int_snow
    int_snow += d
    if int_snow < EPS:
        int_snow = 0.0
    mwi = cap * int_snow + max_int
    if int_rain + rain <= mwi:
        int_rain += rain
        rain_through = rain * (1 - f)
    else:
        rain_through = (int_rain + rain - mwi) * f + rain * (1 - f)
        int_rain = mwi
    if rain == 0 and int_rain < thres:
        rain_through += int_rain
        int_rain = 0.0
    if int_rain + int_snow > imax1:
        ov = (int_snow + int_rain) - imax1
        fr = int_rain / (int_rain + int_snow)
        fs = int_snow / (int_rain + int_snow)
        int_rain -= ov * fr
        int_snow -= ov * fs
        rain_through += ov * fr * f
        snow_through += ov * fs * f
    return int_rain, int_snow, rain_through, snow_through, d


def test_matches_the_c_loop_and_closes_on_random_canopies():
    g = torch.Generator(device=CPU).manual_seed(11)
    n = 600
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g, dtype=D, device=CPU)
    rain = torch.where(torch.rand(n, generator=g, dtype=D, device=CPU) < 0.3, torch.zeros(n, dtype=D, device=CPU), r(0, 0.01))
    snow = torch.where(torch.rand(n, generator=g, dtype=D, device=CPU) < 0.3, torch.zeros(n, dtype=D, device=CPU), r(0, 0.02))
    lai = r(0.5, 6.0)
    int_snow = torch.where(torch.rand(n, generator=g, dtype=D, device=CPU) < 0.25, torch.zeros(n, dtype=D, device=CPU), r(0, 0.012)) * lai / 3
    int_rain = torch.where(torch.rand(n, generator=g, dtype=D, device=CPU) < 0.25, torch.zeros(n, dtype=D, device=CPU), r(0, 0.002))
    tf = r(-8.0, 4.0)
    max_int = 0.0002 * lai
    wind1 = r(0.0, 4.0)
    is_, ir, st, rt, blown = C.intercept(tf, lai, snow, rain, int_snow, int_rain, wind1,
                                         torch.ones_like(lai), max_int)
    got = (ir, is_, rt, st)
    worst = 0.0
    for k in range(n):
        ref = c_mass(float(rain[k]), float(snow[k]), float(int_rain[k]), float(int_snow[k]),
                     float(tf[k]), float(lai[k]), float(max_int[k]), float(wind1[k]))
        for a, b in zip(got, ref[:4]):
            worst = max(worst, abs(float(a[k]) - b))
    assert worst < 1e-15, 'vectorised interception differs from the C loop by %.3e m' % worst
    # mass closure with F = 1: stores + throughfall after == stores + falls before
    before = int_rain + int_snow + rain + snow
    after = ir + is_ + rt + st
    assert float((after - before).abs().max()) < 1e-14
    # the branches were reached
    assert int((st > snow).sum()) > 10 and int((rt > rain).sum()) > 10   # unloading / thin fall-through
    assert int(((is_ > int_snow) & (snow > 0)).sum()) > 100               # interception
