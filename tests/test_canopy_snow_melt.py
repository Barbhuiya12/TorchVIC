"""snow_intercept.c:421-535 and MassRelease.c against scalar transcriptions."""

import torch

from torchvic.core import canopy as C  # noqa: E402
from torchvic.core import snow as S  # noqa: E402

D = torch.float64
CPU = torch.device('cpu')
LAT = S.CONST_LATICE
RHO = S.CONST_RHOFW
MINS = C.VEG_MIN_INTERCEPTION_STORAGE
CAP = C.SNOW_LIQUID_WATER_CAPACITY


def c_mass_release(snow, temp, released, drip):
    if snow > MINS:
        thr = 0.10 * snow
        mx = 0.17 * snow
        if temp >= thr:
            drip += thr
            snow -= thr
            temp -= thr
            rel = 0.0 if snow < MINS else min(snow - MINS, mx)
            released += rel
            snow -= rel
            return c_mass_release(snow, temp, released, drip)
        else:
            td = min(temp, snow)
            drip += td
            snow -= td
    else:
        td = min(temp, snow)
        drip += td
        snow -= td
        temp = 0.0
    return snow, temp, released, drip


def c_melt(tf, rf, v, snow, rain, temp, snow_through, init_snow, max_int, dt):
    mwi = CAP * snow + max_int
    melt_energy = 0.0
    drip = 0.0
    released = 0.0
    if tf == 0:
        if -v > rain:
            v = -rain
            rain = 0.0
        else:
            rain += v
        if rf < 0:
            pot = min(-rf / LAT / RHO, snow)
            melt_energy -= LAT * pot * RHO / dt
        else:
            pot = 0.0
        if rain + pot <= mwi:
            snow -= pot
            rain += pot
        else:
            excess = pot + rain - mwi
            snow -= mwi - rain
            rain = mwi
            if snow < 0:
                snow = 0.0
            if snow_through > 0 and init_snow <= MINS:
                drip += excess
                snow -= excess
                if snow < 0:
                    snow = 0.0
            else:
                temp += excess
            snow, temp, released, drip = c_mass_release(snow, temp, released, drip)
        mwi = CAP * snow + max_int
        if rain > mwi:
            drip += rain - mwi
            rain = mwi
    else:
        temp = 0.0
        if -rf > -rain * LAT:
            snow += abs(rf) / LAT
            rain -= abs(rf) / LAT
            melt_energy += abs(rf) * RHO / dt
        else:
            snow += rain
            melt_energy += LAT * rain * RHO / dt
            rain = 0.0
        if -v > snow:
            v = -snow
            snow = 0.0
        else:
            snow += v
    return snow, rain, temp, v, melt_energy, drip, released


def test_melt_and_release_match_the_c_on_random_canopies():
    g = torch.Generator(device=CPU).manual_seed(5)
    n = 800
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g, dtype=D, device=CPU)
    coin = lambda p: torch.rand(n, generator=g, dtype=D, device=CPU) < p
    tf = torch.where(coin(0.5), torch.zeros(n, dtype=D, device=CPU), r(-12.0, -0.1))
    rf = torch.where(coin(0.5), r(-400.0, 0.0), r(0.0, 200.0))
    v = torch.where(coin(0.7), r(-0.0008, 0.0), r(0.0, 0.0002))
    snow = torch.where(coin(0.2), torch.zeros(n, dtype=D, device=CPU), r(0.0, 0.03))
    rain = torch.where(coin(0.3), torch.zeros(n, dtype=D, device=CPU), r(0.0, 0.002))
    temp = torch.where(coin(0.5), torch.zeros(n, dtype=D, device=CPU), r(0.0, 0.01))
    snow_through = torch.where(coin(0.5), torch.zeros(n, dtype=D, device=CPU), r(0.0, 0.01))
    init_snow = torch.where(coin(0.5), snow, r(0.0, 0.004))
    max_int = r(0.0001, 0.001)
    dt = torch.full((n,), 3600.0, dtype=D, device=CPU)
    got = C.melt_and_release(tf, rf, v, snow, rain, temp, snow_through, init_snow, max_int, dt)
    worst = 0.0
    for k in range(n):
        ref = c_melt(float(tf[k]), float(rf[k]), float(v[k]), float(snow[k]), float(rain[k]),
                     float(temp[k]), float(snow_through[k]), float(init_snow[k]),
                     float(max_int[k]), float(dt[k]))
        for a, b in zip(got, ref):
            worst = max(worst, abs(float(a[k]) - b) / max(abs(b), 1.0))
    assert worst < 1e-12, 'melt_and_release differs from the C by %.3e relative' % worst
    # both branches and the release path were exercised
    assert int((got[6] > 0).sum()) > 5, 'no mass release reached'
    assert int((got[5] > 0).sum()) > 20, 'no drip reached'
