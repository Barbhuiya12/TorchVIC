"""Equilibrium stream temperature (surface heat exchange), tensor code.

Edinger, Duttweiler and Geyer (1968, J. Water Pollut. Control Fed. 40:1537)
and Brady, Graves and Geyer (1969): the net surface heat flux is linearised
about the equilibrium temperature Te,

    dT/dt = -(K / (rho cp h)) (T - Te),     Te = Td + Rs_net / K,

with Td the dew-point temperature, Rs_net the net shortwave at the water
surface, K the surface exchange coefficient (W m-2 K-1)

    K = 4.5 + 0.05 Tw + (beta + 0.47) f(W),
    beta = 0.35 + 0.015 Tm + 0.0012 Tm^2,   f(W) = 9.2 + 0.46 W^2,

Tm = (Td + Tw) / 2, W the wind speed (m s-1).  RBM (Yearsley 2009, Water
Resour. Res. 45:W12421; Yearsley 2012, Hydrol. Earth Syst. Sci. 16:4151)
solves the full energy budget (shortwave, longwave, latent, sensible heat)
with air temperature, vapour pressure, wind and radiation from the VIC forcing;
the linearisation here keeps those same forcings and is exact for the steady
analytic case.  Tw is replaced by the air temperature in K (no iteration).
"""
import torch

RHO_W = 1000.0          # kg m-3
CP_W = 4181.3           # J kg-1 K-1


def dew_point(vp_kpa):
    """Dew point [C] from vapour pressure [kPa] (Magnus inversion, Tetens constants)."""
    g = torch.log(vp_kpa.clamp(min=1e-6) / 0.6108)
    return 237.3 * g / (17.27 - g)


def exchange_coefficient(t_air, t_dew, wind):
    """K [W m-2 K-1], Edinger et al. (1968) with Tw ~ Ta."""
    tm = 0.5 * (t_dew + t_air)
    beta = 0.35 + 0.015 * tm + 0.0012 * tm * tm
    fw = 9.2 + 0.46 * wind * wind
    return 4.5 + 0.05 * t_air + (beta + 0.47) * fw


def equilibrium_temperature(t_air, vp_kpa, wind, shortwave, albedo=0.06,
                            shade=0.0, floor=0.0):
    """(Te [C], K [W m-2 K-1]).

    shortwave: incoming shortwave [W m-2]; albedo: water albedo; shade: the
    fraction of shortwave blocked by riparian cover.  Te is floored at the
    freezing point (water cannot be colder)."""
    td = dew_point(vp_kpa)
    k = exchange_coefficient(t_air, td, wind)
    te = td + shortwave * (1.0 - albedo) * (1.0 - shade) / k
    return te.clamp(min=floor), k


def relaxation_rate(k_exchange, depth):
    """1/s: K / (rho cp h)."""
    return k_exchange / (RHO_W * CP_W * depth)
