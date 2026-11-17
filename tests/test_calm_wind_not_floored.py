"""A calm but positive forcing wind reaches the snow aerodynamics unchanged.

VIC 5.1.0 has no wind floor in the image driver or in vic_run (vic_run.c:171
passes force->wind[NR] to CalcAerodynamic, which divides by it, :177-186);
only wind == 0 is special-cased (Ra = HUGE_RESIST). TorchVIC used to floor the
wind at 0.1 m/s, which changed the stable-snow aerodynamic resistance and the
snow surface temperature of a still, cold, deep pack (ISC cell y451 x118,
2004 day 341, forcing wind 0.0992 m/s: AERO_RESIST 0.86 % and snow surface
temperature 2.8e-4 K off VIC 5.1.0).
"""
import math

import torch

from torchvic.core import vegetation
from torchvic.model import basin

K2 = basin.VON_KARMAN_SQ


def _atm(wind):
    one = lambda v: torch.full((1, 1), v, dtype=torch.float64)
    return dict(tair=one(-19.678279876708984), wind=one(wind),
                shortwave=one(0.0), longwave=one(202.7357330322),
                air_dens=one(0.8833401059390), eact_air=one(73.2367634773),
                lv=one(2547460.41878891), press=one(64269.27185058594),
                vpd=one(32.485732756061))


def _snow(wind, z0s=0.0005, zref=3.0):
    z = basin.snow_forcing_at(_atm(wind), 0, torch.zeros(1, dtype=torch.float64),
                              torch.zeros(1, dtype=torch.float64),
                              torch.zeros(1, dtype=torch.float64))
    return z


def test_wind_guard_is_below_any_real_wind():
    assert basin.MIN_WIND <= 1e-6


def test_calm_forcing_wind_is_not_floored_in_the_open_snow_aerodynamics():
    lr = math.log((2.0 + basin.SNOW_Z0) / basin.SNOW_Z0)
    for w in (0.0992, 0.05, 0.0123):
        z = _snow(w)
        # the pack is handed U[2] = w * ln((2+z0)/z0) / ln(zref/z0)
        # (CalcAerodynamic.c:97-104, :178-185); no 0.1 m/s floor on w
        want_u = lr / math.log(basin.MEAS_HEIGHT / basin.SNOW_Z0) * w
        got_u = float(z['wind'])
        assert abs(got_u - want_u) <= 1e-15 * abs(want_u), (w, got_u, want_u)
        want_ra = lr * math.log(basin.MEAS_HEIGHT / basin.SNOW_Z0) / K2 / w
        got_ra = float(z['ra'])
        assert abs(got_ra - want_ra) <= 1e-12 * want_ra, (w, got_ra, want_ra)


def test_vegetation_resistance_uses_the_calm_wind_as_given():
    z0 = torch.tensor([0.0738], dtype=torch.float64)
    d = torch.tensor([0.402], dtype=torch.float64)
    ra = {}
    for w in (0.0992, 0.2):
        ra[w] = float(vegetation.aerodynamic_resistance(
            z0, d, torch.tensor([w], dtype=torch.float64)))
    # Ra scales as 1/wind, with no floor between 0.0992 and 0.2
    assert abs(ra[0.0992] * 0.0992 - ra[0.2] * 0.2) <= 1e-12 * ra[0.2] * 0.2
