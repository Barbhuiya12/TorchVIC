"""The energy rows VIC reports in water-balance mode, per tile.

VIC 5.1.0 with FULL_ENERGY FALSE still runs func_surf_energy_bal once, at
Tsurf = Tair (calc_surf_energy_bal.c:542-545), so every energy term of that
balance exists and put_data.c writes it. The water-balance path of
`basin.simulate` computes the hydrology of that evaluation exactly; this
module forms the rest of it from the per-tile quantities that path holds, with
the expressions the energy-balance path uses (`energy._residual`,
`basin._eb_rows`), at TMean = Tair:

* the stability correction is exactly 1 (StabilityCorrection.c returns 1 when
  the two temperatures are equal), so Ra_veg[0] is the neutral resistance of
  the understory level and the sensible heat of the surface,
  calc_sensible_heat(Tair, Tair), is zero;
* NetShortBare, NetLongBare, NetShortSnow and NetLongSnow are
  calc_surf_energy_bal.c:211-238 with the surface emitting EMISS_GRND Tair^4
  (func_surf_energy_bal.c:632-636);
* latent heat is -RHOFW Le Evap (1 - snow_coverage) (:766, :838) with Evap the
  water balance's own evaporation, plus a thin pack's latent heat;
* the atmospheric terms of an overstory tile with snow are
  calc_atmos_energy_bal.c's at Tcanopy = Tair.

Nothing here changes a state or an existing output.
"""
import torch

from ..core import snow, energy, vegetation

CONST_TKFRZ = snow.CONST_TKFRZ
DBL_EPSILON = 2.220446049250313e-16
SOIL_RARC = 100.0


def surface_rows(*, tair, lw_in, sw_in, cov, dcov, spatial, cov_eff, thin,
                 alb_t, albedo, alb_under, pack_ns, pack_nl, pack_lat,
                 pack_sen, pack_w, evap_ms, thin_latent, ra_veg0, ra_bare,
                 fc, air_dens, le):
    """The surface balance at Tsurf = Tair for every tile.

    All arguments are [cell, tile] tensors (booleans where named so) except
    `fc` and `ra_bare`, which may be None (no fractional canopy). Returns the
    reported rows (put_data.c's sign: latent and sensible negated, radiation
    downward), before any overstory canopy joins them:

        net_short, net_long    NetShortUnder, NetLongUnder (pack included)
        latent, sensible       energy.latent / energy.sensible, negated
        aero_resist            Ra_used[0]
        long_out               the surface's emission
    """
    bf = 1.0 - cov
    long_out = snow.calc_outgoing_longwave(tair + CONST_TKFRZ,
                                           energy.EMISS_GRND)
    # calc_surf_energy_bal.c 5.1.0:229-231
    if spatial:
        ns_bare = (sw_in * (1.0 - alb_t) * (1.0 - cov_eff)
                   + sw_in * dcov * (1.0 - albedo))
        ns_snow = sw_in * (1.0 - alb_under) * cov
    else:
        ns_bare = (sw_in * (1.0 - alb_t) * (bf - dcov)
                   + sw_in * dcov * (1.0 - albedo))
        ns_snow = sw_in * (1.0 - albedo)
    # func_surf_energy_bal.c:658-662 and calc_surf_energy_bal.c:214, :218
    nl_bare = (1.0 - cov) * lw_in - (1.0 - cov) * long_out
    nl_snow = cov * lw_in - cov * long_out
    ns_surf = torch.where(thin, (ns_bare + ns_snow) + 0.0, ns_bare)
    nl_surf = torch.where(thin, nl_bare + nl_snow, bf * (lw_in - long_out))
    net_short = ns_surf + pack_ns
    net_long = nl_surf + pack_nl

    # Ra_used[0] (func_surf_energy_bal.c 5.1.0:677-745) at corr == 1
    ra_used = ra_veg0
    if fc is not None and ra_bare is not None:
        one = torch.ones_like(ra_used)
        pos = (ra_used > 0.0) & (ra_bare > 0.0)
        ga = (fc * (1.0 / torch.where(ra_used > 0.0, ra_used, one))
              + (1.0 - fc) * (1.0 / torch.where(ra_bare > 0.0, ra_bare, one)))
        blended = torch.where(pos, 1.0 / torch.where(pos, ga, one),
                              torch.zeros_like(ga))
        ra_used = torch.where(fc < 1.0, blended, ra_used)

    latent_int = -snow.CONST_RHOFW * le * evap_ms * bf
    latent_int = torch.where(thin, latent_int + thin_latent, latent_int)
    sensible_int = snow.calc_sensible_heat(air_dens, tair, tair, ra_used)
    sensible_int = torch.where(thin, sensible_int, sensible_int * bf)
    # calc_surf_energy_bal.c:600-612 with the pack's own terms added
    latent = -latent_int - pack_lat * pack_w
    sensible = -sensible_int - pack_sen * pack_w
    return dict(net_short=net_short, net_long=net_long, latent=latent,
                sensible=sensible, aero_resist=ra_used, long_out=long_out)


def potential_evap(*, tair, vpd, elev, sw_sky, alb_t, nl_atm, rmin, rgl,
                   rarc, lai, ra_veg, ra_used, fc):
    """compute_pot_evap.c per tile, mm/day (the caller scales by dt / day):
    the vegetated Penman with the library resistance and the bare soil's with
    SOIL_RARC, mixed by fcanopy. `ra_veg` is Ra_veg[0] (Ra_veg[1] on an
    overstory tile), `ra_used` Ra_used[0]."""
    ns_pot = sw_sky * (1.0 - alb_t)
    rc_pot = vegetation.calc_rc(rmin, ns_pot, rgl, tair, vpd, lai,
                                torch.ones_like(lai))
    epot_veg = vegetation.penman(tair, elev, ns_pot + nl_atm, vpd, ra_veg,
                                 rc_pot, rarc)
    epot_soil = vegetation.penman(tair, elev, ns_pot + nl_atm, vpd, ra_used,
                                  torch.zeros_like(rc_pot),
                                  torch.full_like(rc_pot, SOIL_RARC))
    fc_pot = torch.ones_like(alb_t) if fc is None else fc
    return fc_pot * epot_veg + (1.0 - fc_pot) * epot_soil
