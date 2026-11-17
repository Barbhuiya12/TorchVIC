"""The carbon cycle: photosynthesis, autotrophic respiration and the soil
carbon pools.  VIC's CARBON option, vectorised and differentiable.

Five files of VIC become this one module:

    faparl.c                absorbed photosynthetically active radiation per
                            canopy layer, by Sellers' (1985) two-stream
                            approximation
    calc_Nscale_factors.c   how the plant distributes its nitrogen, and so
                            its Rubisco, down through the canopy
    photosynth.c            Farquhar (1980) for C3 plants and Collatz (1992)
                            for C4, at a given leaf-internal CO2 or a given
                            stomatal resistance
    canopy_assimilation.c   the layers summed into gross primary production,
                            respiration and a canopy resistance
    compute_soil_resp.c     heterotrophic respiration from three soil carbon
                            pools
    soil_carbon_balance.c   those pools advanced one step

What it is for, beyond reporting carbon: with RC_MODE = RC_PHOTO the canopy
resistance the water balance uses is the one photosynthesis implies, rather
than Jarvis's empirical function of light, temperature and vapour pressure
deficit. Turning CARBON on therefore changes transpiration, not only the
carbon outputs.

Two shapes run through here. Most quantities are per cell and tile, and the
canopy is `Ncanopy` layers deep; layer-resolved quantities carry that as a
*leading* axis, so a [layer, cell, tile] tensor broadcasts against a
[cell, tile] one without any reshaping. `soil_resp` is the same idea over
thermal nodes.

The C's branches on plant type (C3 or C4) are `where` here rather than
`if`, so both are evaluated and selected. That is what makes the module
differentiable, and it is safe because each branch is finite on the other's
inputs -- the guards below say where that needed care.
"""
import math

import torch

# Taken from torchvic.core.atmos rather than restated, so there is one
# definition of each. VIC derives the gas constant from Avogadro's
# number and Boltzmann's constant rather than quoting it, and a value
# quoted to more digits than that product has is a 6e-07 relative error
# that walks straight into every rate below.
from .atmos import (CONST_RGAS, CONST_PSTD, CONST_TKFRZ,
                    scale_height as _scale_height)
CONST_DDAYS_PER_YEAR = 365.2425
SEC_PER_DAY = 86400.0
HOURS_PER_DAY = 24.0
MM_PER_M = 1000.0

#: initialize_parameters.c:105-133.
PHOTO_OMEGA = 0.12             #: single-leaf albedo in the PAR band
PHOTO_LAILIMIT = 3.0
PHOTO_LAIMIN = 1.0e-9
PHOTO_EPAR = 2.2e5             #: J per mol of photons in the PAR band
PHOTO_FCMAX = 0.9
PHOTO_FCMIN = 1.0e-3
PHOTO_ZENITHMINPAR = 1.0e-3
PHOTO_MINMAXETRANS = 1.0e-12
PHOTO_FCI1C3 = 0.87
PHOTO_FCI1C4 = 0.67
PHOTO_OX = 0.21                #: oxygen partial pressure, mol/mol
PHOTO_KC = 460.0e-6            #: Michaelis-Menten constant for CO2 at 25 C
PHOTO_KO = 330.0e-3            #: Michaelis-Menten constant for O2 at 25 C
#: VIC 4.2.d's gas constant, vicNl_def.h:303, in J/(mol K) -- the unit every
#: activation energy below is stated in. VIC 5 substituted CONST_RGAS, which
#: is J/(kmol K), in all nine places photosynth.c uses it; see
#: `PHOTOSYNTHESIS_RGAS` and `photosynth`.
PHOTO_RGAS_PER_MOL = 8.3143

PHOTO_EC = 59356.0             #: activation energies, J/mol
PHOTO_EO = 35948.0
PHOTO_EV = 58520.0
PHOTO_ER = 45000.0
PHOTO_ALC3 = 0.28              #: quantum efficiency, C3
PHOTO_FRDC3 = 0.011            #: dark respiration as a fraction of Vcmax, C3
PHOTO_EK = 50967.0
PHOTO_ALC4 = 0.04              #: quantum efficiency, C4
PHOTO_FRDC4 = 0.042
PHOTO_THETA = 0.83             #: Collatz's curvature parameter
PHOTO_FRLEAF = 0.4             #: leaf share of maintenance respiration
PHOTO_FRGROWTH = 0.25

#: initialize_parameters.c:136-147.
SRESP_E0_LT = 308.56           #: Lloyd and Taylor activation energy, K
SRESP_T0_LT = 227.13           #: Lloyd and Taylor lower limit, K
SRESP_WMINFM = 0.0
SRESP_WMAXFM = 1.0
SRESP_WOPTFM = 0.5
SRESP_RHSAT = 0.15
SRESP_RFACTOR = 0.5
SRESP_TAULITTER = 2.86         #: pool turnover times, years
SRESP_TAUINTER = 33.3
SRESP_TAUSLOW = 1000.0
#: vic_physical_constants.h:88, kg per kmol -- so dividing by MOLE_PER_KMOLE
#: as put_data.c:750 does gives grams of carbon per mole of CO2 fixed.
CONST_MWC = 12.01

SRESP_FAIR = 0.7               #: litter respiration that goes to the air
SRESP_FINTER = 0.985

HUGE_RESIST = 1.0e20           #: param.HUGE_RESIST, initialize_parameters.c

#: vic_def.h:160-161. Kept as numbers so a plant type can be a tensor and a
#: mosaic can hold both kinds at once.
C3 = 0
C4 = 1


def _tiny(x):
    return torch.finfo(x.dtype).tiny


def canopy_layer_lai(lai_total, bnd):
    """Leaf area in each canopy layer.  faparl.c:52-64.

    `bnd` is the cumulative fraction of the canopy at each layer's base,
    [layer] or [layer, ...]; the layers partition it, so each one's share is
    the difference from the layer above.
    """
    upper = torch.cat([torch.zeros_like(bnd[:1]), bnd[:-1]], 0)
    lai = lai_total * (bnd - upper)
    return torch.clamp(lai, min=PHOTO_LAIMIN)


def fapar(lai_total, bnd, coszen, albsoil_par, fdir):
    """Absorbed PAR per unit ground area per canopy layer, normalised to an
    incoming PAR of 1 W.  faparl.c, Sellers' (1985) two-stream solution
    under the C's two assumptions -- isotropic leaf angles, and leaf
    reflectivity equal to transmissivity -- which reduce b = b0 = 0.5 and
    muq = 1.

    Returns [layer, ...], zero everywhere the sun is below
    PHOTO_ZENITHMINPAR, as the C leaves it.
    """
    n = bnd.shape[0]
    # :39-50, the fractional cover: how much bare ground shows between the
    # plants, which is what relates a flux per leaf area to one per ground
    # area. Not the tile's own area fraction.
    fc = torch.where(lai_total < PHOTO_LAILIMIT,
                     lai_total / PHOTO_LAILIMIT * PHOTO_FCMAX,
                     torch.full_like(lai_total, PHOTO_FCMAX))
    fc = torch.clamp(fc, min=PHOTO_FCMIN)

    zh = math.sqrt(1.0 - PHOTO_OMEGA)
    zp1 = (1.0 - PHOTO_OMEGA / 2.0 + zh) / PHOTO_OMEGA * 2.0
    zp0 = 1.0 / zp1
    # :159-166. The C nudges k off +/- h to keep the solution non-singular;
    # the same nudge here, and a floor under coszen so the division is
    # finite on the branch the `where` below discards.
    cz = torch.clamp(coszen, min=PHOTO_ZENITHMINPAR)
    k0 = 0.5 / cz
    k0 = torch.where((k0 - zh).abs() < 1e-12, k0 + 1e-12, k0)
    k0 = torch.where((k0 + zh).abs() < 1e-12, k0 + 1e-12, k0)

    x0d = 1.0 - 4.0 * cz * cz * zh * zh
    # x0d vanishes where 2*cz*zh == 1, a single zenith angle; the C divides
    # regardless and would return an infinity there.
    x0d = torch.where(x0d.abs() > 0.0, x0d, torch.full_like(x0d, 1e-300))
    q1 = ((1.0 + 2.0 * cz) * PHOTO_OMEGA / 2.0) / x0d
    q0 = ((1.0 - 2.0 * cz) * PHOTO_OMEGA / 2.0) / x0d

    ekl = torch.exp(-k0 / fc * lai_total)
    ehl = torch.exp(-zh / fc * lai_total)
    x1 = (zp1 - albsoil_par) / ehl
    x0 = (zp0 - albsoil_par) * ehl
    x2 = (albsoil_par * (1.0 - q1) - q0) * ekl
    f = 1.0 - fdir + q1 * fdir
    den = x1 - x0
    den = torch.where(den.abs() > 0.0, den, torch.full_like(den, 1e-300))
    b1 = (x2 * fdir - f * x0) / den
    b0 = (x2 * fdir - f * x1) / (-den)
    b4 = (1.0 - q0 - q1) * fdir + (1.0 - zp1) * b1 + (1.0 - zp0) * b0

    out = []
    for cidx in range(n - 1):
        ekl0 = torch.exp(-k0 / fc * bnd[cidx] * lai_total)
        ehl0 = torch.exp(-zh / fc * bnd[cidx] * lai_total)
        xi = ((1.0 - q0 - q1) * ekl0 * fdir
              + (1.0 - zp1) * b1 / ehl0 + (1.0 - zp0) * b0 * ehl0)
        out.append(b4 - xi)
        b4 = xi
    xi = ((1.0 - q0 - q1) * ekl * fdir
          + (1.0 - zp1) * b1 / ehl + (1.0 - zp0) * b0 * ehl)
    out.append(b4 - xi)
    apar = torch.stack(out, 0) * fc
    # :155, nothing is absorbed with the sun below the horizon.
    lit = coszen >= PHOTO_ZENITHMINPAR
    return torch.where(lit.expand_as(apar), apar, torch.zeros_like(apar))


def apar_per_leaf(par, fapar_layer, lai_layer):
    """Absorbed PAR per unit LEAF area, mol(photons)/m2/s.
    surface_fluxes.c:419-434.

    `fapar` returns a fraction of the incoming PAR absorbed per unit GROUND
    area; photosynthesis works per leaf area and in moles of photons, and
    this is the conversion between them: multiply by the incoming PAR,
    divide by PHOTO_EPAR to get moles, divide by the layer's leaf area.

    The distinction matters by three orders of magnitude, so it is worth
    saying plainly: `fapar`'s output is order 0.05 and this function's is
    order 1e-5, and photosynth.c multiplies it by a quantum efficiency of
    0.28 mol CO2 per mol photons. Handing the fraction straight to
    photosynthesis puts every leaf permanently in the carboxylation-limited
    regime.

    The C floors the leaf area at 1e-10 rather than skipping the layer
    (:429-433), which is what `canopy_layer_lai` already does.
    """
    return (par / PHOTO_EPAR) * fapar_layer / torch.clamp(lai_layer,
                                                          min=PHOTO_LAIMIN)


def nscale_factors(nscale_flag, bnd, lai_total, coszen):
    """Nitrogen scaling down the canopy.  calc_Nscale_factors.c:38-64.

    The plant puts its Rubisco where the light is, so the capacity falls
    exponentially with the leaf area above. Returns [layer, ...], all ones
    where the flag is off or the canopy is thin.
    """
    cz = torch.clamp(coszen, min=PHOTO_ZENITHMINPAR)
    k12 = 0.5 / cz
    upper = torch.cat([torch.zeros_like(bnd[:1]), bnd[:-1]], 0)
    scaled = torch.clamp(torch.exp(-k12 * upper * lai_total), min=1e-10)
    deep = torch.ones_like(scaled, dtype=torch.bool)
    deep[0] = False                       # the C's `cidx > 0`
    on = (nscale_flag & (lai_total > PHOTO_LAILIMIT)) & deep
    return torch.where(on, scaled, torch.ones_like(scaled))


def hi_t_inhib(t_c):
    """Inhibition of photosynthesis at high leaf temperature.
    photosynth.c:447-455."""
    return 1.0 / (1.0 + torch.exp(1.3 * (t_c - 55.0)))


def dark_inhib(irr):
    """Inhibition of dark respiration by light.  photosynth.c:461-475.

    `irr` is the total irradiance at the surface in mol/m2/s.
    """
    return torch.where(irr < 0.0, torch.zeros_like(irr),
                       0.5 + 0.5 * torch.exp(-irr * 1.0e6 / 10.0))


def photosynth(ctype, max_carbox_rate, max_e_transport, co2_specificity,
               nscale, t_foliage, pirrin, apar, psurf, catm, mode,
               rs=None, ci=None, rgas=CONST_RGAS):
    """Leaf photosynthesis.  photosynth.c:48-440.  Returns a dict with
    rs, Ci, Rdark, Rphoto and Agross.

    **On `rgas`.** VIC 4.2.d wrote every one of these expressions with
    `Rgas = 8.3143 J/(mol K)` (vicNl_def.h:303). VIC 5.0.1 replaced it with
    `CONST_RGAS`, which is J/(**k**mol K) -- a thousand times larger -- in
    all nine places, and nothing else changed to match. Two things follow,
    and both are large:

    * The Arrhenius factors `exp(E * (T0/T1) / (Rgas * T))` lose their
      exponent. Vcmax at 5 C comes out at 0.998 of its 25 C value instead of
      0.183; dark respiration at 0.999 instead of 0.271. Photosynthesis and
      respiration stop responding to temperature at all.
    * `r0 = rs * 1.6 * Rgas * T / Psurf` comes out a thousand times too
      large, so the CO2 the stomata can supply is a thousandth of what it
      should be and assimilation is supply-starved everywhere. With a
      canopy of LAI 3 in full sun this is the difference between net primary
      production of -0.2 and +5 gC/m2/day: VIC 5's sign is wrong.

    The default is VIC 5's value, because this model's contract is to
    reproduce VIC 5 and `tests/test_carbon.py` holds it to VIC 5's
    own compiled output. Pass `rgas=PHOTO_RGAS_PER_MOL` for the arithmetic
    VIC 4.2.d did and the units the activation energies are stated in.

    `mode` is 'ci' or 'rs': either the leaf-internal CO2 is given and the
    stomatal resistance follows, or the resistance is given and Ci follows.
    VIC uses the first to initialise and the second inside its canopy
    iteration.

    `ctype` is C3 or C4 elementwise. The C branches on it; here both
    formulations are evaluated and selected, which is why every division
    below is guarded even where one branch could not reach it.
    """
    t1 = 25.0 + CONST_TKFRZ
    t = t_foliage + CONST_TKFRZ
    t0 = t - t1
    is_c3 = ctype == C3

    # :108-112. Rubisco capacity, temperature- and nitrogen-scaled.
    vcmax = max_carbox_rate * nscale * torch.exp(
        PHOTO_EV * (t0 / t1) / (rgas * t))

    # ---- C3 rates, :130-190 ------------------------------------------------
    kc = PHOTO_KC * torch.exp(PHOTO_EC * (t0 / t1) / (rgas * t))
    ko = PHOTO_KO * torch.exp(PHOTO_EO * (t0 / t1) / (rgas * t))
    gamma = torch.clamp(1.7e-6 * t_foliage, min=0.0)
    k2 = kc * (1.0 + PHOTO_OX / ko)
    jmax = torch.clamp(max_e_transport * nscale * t_foliage / 25.0,
                       min=PHOTO_MINMAXETRANS)
    # :176-182. The C forms J1 only where Jmax exceeds the floor; at the
    # floor the denominator is still positive, so the same expression is
    # evaluated and then selected -- no division by zero on either side.
    al_apar = PHOTO_ALC3 * apar
    j1 = torch.where(jmax > PHOTO_MINMAXETRANS,
                     al_apar * jmax / torch.sqrt(jmax * jmax
                                                 + al_apar * al_apar),
                     torch.zeros_like(jmax))
    rdark_c3 = (PHOTO_FRDC3 * max_carbox_rate * nscale
                * torch.exp(PHOTO_ER * (t0 / t1) / (rgas * t))
                * hi_t_inhib(t_foliage) * dark_inhib(pirrin))

    # ---- C4 rates, :196-230 ------------------------------------------------
    k = (co2_specificity * 1.0e3 * nscale
         * torch.exp(PHOTO_EK * (t0 / t1) / (rgas * t)))
    rdark_c4 = (PHOTO_FRDC4 * max_carbox_rate * nscale
                * torch.exp(PHOTO_ER * (t0 / t1) / (rgas * t))
                * hi_t_inhib(t_foliage) * dark_inhib(pirrin))

    rdark = torch.where(is_c3, rdark_c3, rdark_c4)

    # C4's JE is the smaller root of Collatz's quadratic; its discriminant is
    # non-negative analytically but rounds below zero when Vcmax and Ji are
    # nearly equal, so it is floored.
    j0 = (PHOTO_ALC4 * apar + vcmax) / 2.0 / PHOTO_THETA
    disc = torch.clamp(j0 * j0 - vcmax * PHOTO_ALC4 * apar / PHOTO_THETA,
                       min=0.0)
    je_c4 = j0 - torch.sqrt(disc)

    if mode == 'ci':
        # :235-290, gross photosynthesis at a given leaf-internal CO2.
        ci_v = ci
        den_e = 4.0 * (ci_v + 2.0 * gamma)
        den_e = torch.where(den_e.abs() > 0.0, den_e,
                            torch.full_like(den_e, _tiny(den_e)))
        je_c3 = j1 * (ci_v - gamma) / den_e
        den_c = ci_v + k2
        den_c = torch.where(den_c.abs() > 0.0, den_c,
                            torch.full_like(den_c, _tiny(den_c)))
        jc_c3 = vcmax * (ci_v - gamma) / den_c
        jc_c4 = k * ci_v
        je = torch.where(is_c3, je_c3, je_c4)
        jc = torch.where(is_c3, jc_c3, jc_c4)
        r0 = None
    else:
        # :292-400, gross photosynthesis at a given stomatal resistance. Ci
        # is eliminated with the diffusion equation and each of JE and JC
        # becomes a quadratic; the C takes the smaller root of each.
        r0 = rs * 1.6 * rgas * t / psurf
        r0 = torch.where(r0.abs() > 0.0, r0, torch.full_like(r0, _tiny(r0)))
        k1 = 2.0 * gamma
        w1 = j1 / 4.0
        b = rdark + w1 + (catm + k1) / r0
        c = w1 * (catm - gamma) / r0 + w1 * rdark
        je_c3 = b / 2.0 - torch.sqrt(torch.clamp(b * b / 4.0 - c, min=0.0))
        b = rdark + vcmax + (catm + k2) / r0
        c = vcmax * (catm - gamma) / r0 + vcmax * rdark
        jc_c3 = b / 2.0 - torch.sqrt(torch.clamp(b * b / 4.0 - c, min=0.0))
        # :396, C4: JC = (Catm/r0 + Rdark) / (1 + 1/(K*r0))
        kr0 = k * r0
        kr0 = torch.where(kr0.abs() > 0.0, kr0,
                          torch.full_like(kr0, _tiny(kr0)))
        jc_c4 = (catm / r0 + rdark) / (1.0 + 1.0 / kr0)
        je = torch.where(is_c3, je_c3, je_c4)
        jc = torch.where(is_c3, jc_c3, jc_c4)

    # :404-412, the assimilation is the smaller of the two, inhibited at
    # high temperature.
    agross = torch.minimum(je, jc) * hi_t_inhib(t_foliage)

    if mode == 'rs':
        # :425-437, Ci from the diffusion equation, with the C's own cap on
        # r0 and floor on Ci.
        r0c = torch.clamp(r0, max=1.0e6)
        ci_out = torch.clamp(catm - (agross - rdark) * r0c, min=0.0)
    else:
        ci_out = ci

    # :443-470, photorespiration. Zero for C4.
    den_p = ci_out + k2
    den_p = torch.where(den_p.abs() > 0.0, den_p,
                        torch.full_like(den_p, _tiny(den_p)))
    rphoto = torch.where(is_c3, vcmax * gamma / den_p * hi_t_inhib(t_foliage),
                         torch.zeros_like(vcmax))

    if mode == 'ci':
        # :486-510, the stomatal resistance the assimilation implies.
        net = agross - rdark
        eps = torch.finfo(net.dtype).eps
        safe_net = torch.where(net.abs() > 0.0, net,
                               torch.full_like(net, _tiny(net)))
        rs_out = (0.625 * (catm - ci_out) / safe_net
                  * (psurf / (rgas * t)))
        rs_out = torch.where(net < eps, torch.full_like(rs_out, HUGE_RESIST),
                             rs_out)
        rs_out = torch.clamp(rs_out, max=HUGE_RESIST)
    else:
        rs_out = rs

    return dict(rs=rs_out, Ci=ci_out, Rdark=rdark, Rphoto=rphoto,
                Agross=agross)


def calc_scale_height(tair, elevation):
    """Atmospheric scale height, m.  physics.c:76-86, which is what
    canopy_assimilation.c:109 calls to get the surface pressure.

    Delegates to torchvic.core.atmos, which already had it verified against the C.
    """
    return _scale_height(tair, elevation)


def canopy_assimilation(ctype, max_carbox_rate, max_e_transport,
                        co2_specificity, nscale, t_foliage, swdown, apar,
                        elevation, catm, bnd, lai_total, mode, rs_layer=None,
                        press=None, rgas=CONST_RGAS):
    """The canopy's layers summed into whole-canopy fluxes.
    canopy_assimilation.c:56-193.  Returns a dict with rsLayer, rc, Ci, GPP,
    Rdark, Rphoto, Rmaint, Rgrowth, Raut and NPP.

    `nscale`, `apar` and `rs_layer` are [layer, ...]; everything else is per
    cell and tile and broadcasts onto them.

    `press` lets a caller pass the surface pressure it already has. The C
    rebuilds it from the elevation and a scale height at :109-111, which is
    what happens when it is left out.
    """
    if press is None:
        h = calc_scale_height(t_foliage, elevation)
        press = CONST_PSTD * torch.exp(-elevation / h)
    pirrin = swdown / PHOTO_EPAR
    d_lai = canopy_layer_lai(lai_total, bnd)

    if mode == 'ci':
        # :120-128, the first guess: a fixed fraction of the ambient CO2.
        ci0 = torch.where(ctype == C3, PHOTO_FCI1C3 * catm,
                          PHOTO_FCI1C4 * catm)
        out = photosynth(ctype, max_carbox_rate, max_e_transport,
                         co2_specificity, nscale, t_foliage, pirrin, apar,
                         press, catm, 'ci', ci=ci0.expand_as(nscale),
                         rgas=rgas)
        gpp = (out['Agross'] * d_lai).sum(0)
        rdark = (out['Rdark'] * d_lai).sum(0)
        rphoto = (out['Rphoto'] * d_lai).sum(0)
        gc = ((1.0 / out['rs']) * d_lai).sum(0)
        gc = torch.clamp(gc, min=torch.finfo(gc.dtype).eps)
        rc = torch.clamp(1.0 / gc, max=HUGE_RESIST)
        ci = ci0
        rs_out = out['rs']
    else:
        out = photosynth(ctype, max_carbox_rate, max_e_transport,
                         co2_specificity, nscale, t_foliage, pirrin, apar,
                         press, catm, 'rs', rs=rs_layer, rgas=rgas)
        gpp = (out['Agross'] * d_lai).sum(0)
        rdark = (out['Rdark'] * d_lai).sum(0)
        rphoto = (out['Rphoto'] * d_lai).sum(0)
        # :169, the C sums Ci weighted by leaf area and does NOT divide by
        # the total -- so with a deep canopy this is larger than any layer's
        # own Ci. Reproduced as written.
        ci = (out['Ci'] * d_lai).sum(0)
        rc = None
        rs_out = rs_layer

    # :183-190.
    rmaint = rdark / PHOTO_FRLEAF
    rgrowth = (PHOTO_FRGROWTH / (1.0 + PHOTO_FRGROWTH)) * (gpp - rmaint)
    raut = rmaint + rgrowth
    npp = gpp - raut
    return dict(rsLayer=rs_out, rc=rc, Ci=ci, GPP=gpp, Rdark=rdark,
                Rphoto=rphoto, Rmaint=rmaint, Rgrowth=rgrowth, Raut=raut,
                NPP=npp)


def _moisture_factor(w):
    """The soil-moisture control on respiration.  compute_soil_resp.c:96-155,
    the same expression for the litter and for every node."""
    wc = torch.clamp(w, SRESP_WMINFM, SRESP_WMAXFM)
    num = (wc - SRESP_WMINFM) * (wc - SRESP_WMAXFM)
    den = num - (wc - SRESP_WOPTFM) * (wc - SRESP_WOPTFM)
    den = torch.where(den.abs() > 0.0, den, torch.full_like(den, 1e-300))
    wet = SRESP_RHSAT + (1.0 - SRESP_RHSAT) * num / den
    dry = num / den
    return torch.clamp(torch.where(wc <= SRESP_WOPTFM, dry, wet), 0.0, 1.0)


def soil_resp(dz, dz_tot, dt, t_node, w_node, c_litter, c_inter,
              c_slow, per_second=False):
    """Heterotrophic respiration from the three soil carbon pools.
    compute_soil_resp.c:47-203.  Returns (RhLitter, RhInter, RhSlow).

    `dz`, `t_node` and `w_node` are [node, ...]; `dz` is in mm and `dt` in
    seconds. The litter pool sits at the surface and sees the top node's
    temperature and moisture; the intermediate and slow pools are spread
    down the profile by node thickness.

    The C's own unit inconsistency is reproduced: the litter turnover time
    is converted with SEC_PER_DAY and the other two with HOURS_PER_DAY
    (:183-200), so the three pools are not on the same clock. It is VIC's,
    and changing it would change every carbon number VIC produces.
    """
    tref = 10.0 + CONST_TKFRZ
    tk = torch.clamp(t_node + CONST_TKFRZ, min=SRESP_T0_LT)
    # Lloyd and Taylor. The clamp above keeps the denominator positive.
    denom = tk - SRESP_T0_LT
    denom = torch.where(denom.abs() > 0.0, denom,
                        torch.full_like(denom, 1e-300))
    f_t = torch.exp(SRESP_E0_LT * (1.0 / (tref - SRESP_T0_LT) - 1.0 / denom))
    f_m = _moisture_factor(w_node)

    frac = dz / dz_tot
    c_inter_node = c_inter * frac
    c_slow_node = c_slow * frac

    rh_litter = (SRESP_RFACTOR
                 * (f_t[0] * f_m[0]
                    / (SRESP_TAULITTER * CONST_DDAYS_PER_YEAR * SEC_PER_DAY
                       / dt))
                 * c_litter)
    # compute_soil_resp.c:187-192 -- HOURS_PER_DAY here, SEC_PER_DAY on the
    # litter line four lines above. `dt` is seconds in VIC 5, so the litter
    # line converts years to steps correctly and these two do not: they
    # treat dt as hours and so shorten the intermediate and slow pools'
    # lifetimes by a factor of 3600. VIC 4.2.d had `dt` in hours
    # (compute_soil_resp.c:8) and all three lines read `365.25*24/dt`,
    # consistently; the 4-to-5 port updated one of the three.
    #
    # Reproduced by default, because that is what VIC 5 computes; pass
    # `per_second=True` for the arithmetic the other two lines imply.
    day = SEC_PER_DAY if per_second else HOURS_PER_DAY
    rh_inter = (SRESP_RFACTOR
                * (f_t * f_m
                   / (SRESP_TAUINTER * CONST_DDAYS_PER_YEAR * day / dt))
                * c_inter_node).sum(0)
    rh_slow = (SRESP_RFACTOR
               * (f_t * f_m
                  / (SRESP_TAUSLOW * CONST_DDAYS_PER_YEAR * day / dt))
               * c_slow_node).sum(0)
    return rh_litter, rh_inter, rh_slow


def node_relative_moisture(zwt_lumped_cm, dz_cum, bubble_node, expt_node):
    """Relative moisture at each thermal node from the water table depth.
    soil_carbon_balance.c:88-160.

    Three cases per node, by where the water table sits relative to the
    node: entirely below it, inside it, or above it. The C writes them as
    an if/else chain; they are `where` here.

    `zwt_lumped_cm` is the water table depth in centimetres, negative
    downwards as the column reports it; `dz_cum` is the cumulative node
    depth in mm, [node, ...].
    """
    b = 0.5 * (expt_node - 3.0)
    b = torch.where(b.abs() > 0.0, b, torch.full_like(b, 1e-300))
    wtd = -zwt_lumped_cm * 10.0                      # mm, positive downwards
    upper = torch.cat([torch.zeros_like(dz_cum[:1]), dz_cum[:-1]], 0)
    bub = bubble_node
    bub = torch.where(bub.abs() > 0.0, bub, torch.full_like(bub, 1e-300))

    def pw(x):
        # The C's pow(x, -1/b) with x <= 0 is a NaN it never reaches, because
        # the branch guards put the water table below the depth in question.
        return torch.pow(torch.clamp(x / bub, min=_tiny(bub)), -1.0 / b)

    # Water table below the node: the whole node is unsaturated.
    w0_below = pw(wtd + bub - upper)
    w1_below = pw(wtd + bub - dz_cum)
    w_below = 0.5 * (w0_below + w1_below)

    # Water table inside the node: unsaturated above it, saturated below.
    mid = 0.5 * (upper + wtd)
    w0_in = pw(wtd + bub - upper)
    w1_in = pw(wtd + bub - mid)
    span = dz_cum - upper
    span = torch.where(span.abs() > 0.0, span, torch.full_like(span, 1e-300))
    w_in = (0.5 * (w0_in + w1_in) * (mid - upper)
            + 0.5 * (w1_in + 1.0) * (wtd - mid)
            + 1.0 * (dz_cum - wtd)) / span

    inside = (wtd > upper) & (wtd <= dz_cum)
    return torch.where(wtd > dz_cum, w_below,
                       torch.where(inside, w_in, torch.ones_like(w_below)))


def soil_carbon_balance(dz, dz_cum, dz_tot, dt, t_node, w_node, c_litter,
                        c_inter, c_slow, annual_npp_prev, per_second=False):
    """One step of the three soil carbon pools.  soil_carbon_balance.c
    :162-186.  Returns a dict with the new pools, the respiration terms and
    the litterfall.

    Last year's net primary production enters the litter evenly through
    this year, which is the C's own assumption at :170-173.
    """
    rh_litter, rh_inter, rh_slow = soil_resp(dz, dz_tot, dt, t_node, w_node,
                                             c_litter, c_inter, c_slow,
                                             per_second=per_second)
    rh_litter2atm = SRESP_FAIR * rh_litter
    rh_tot = rh_litter2atm + rh_inter + rh_slow
    litterfall = annual_npp_prev / (CONST_DDAYS_PER_YEAR * SEC_PER_DAY / dt)
    return dict(
        CLitter=c_litter + litterfall - rh_litter,
        CInter=(c_inter + (1.0 - SRESP_FAIR) * rh_litter * SRESP_FINTER
                - rh_inter),
        CSlow=(c_slow + (1.0 - SRESP_FAIR) * rh_litter * (1.0 - SRESP_FINTER)
               - rh_slow),
        RhLitter=rh_litter, RhLitter2Atm=rh_litter2atm, RhInter=rh_inter,
        RhSlow=rh_slow, RhTot=rh_tot, Litterfall=litterfall)


#: initialize_parameters.c:71-73, re-exported from torchvic.core.vegetation so
#: there is one definition of each.
from .vegetation import CANOPY_CLOSURE, CANOPY_RSMAX, CANOPY_VPDMINFACTOR


def npp_factor(moist_top, w_max_top, wnpp_inhib, npp_factor_sat):
    """The inhibition of photosynthesis by a saturated top soil layer.
    canopy_evap.c:263-275.

    Above `wnpp_inhib` of the layer's capacity the plant's production falls
    linearly to `npp_factor_sat` at saturation; below it there is no
    inhibition. surface_fluxes.c:760 multiplies the dry canopy fraction by
    this before scaling every carbon flux.
    """
    thresh = wnpp_inhib * w_max_top
    span = w_max_top - thresh
    span = torch.where(span.abs() > 0.0, span, torch.full_like(span, 1e-300))
    inhibited = npp_factor_sat + (1.0 - npp_factor_sat) * (w_max_top
                                                           - moist_top) / span
    return torch.where(moist_top > thresh, inhibited,
                       torch.ones_like(inhibited))


def rs_layer_from_rc(rc, lai, ncanopy):
    """The per-layer stomatal resistance RC_JARVIS implies.
    canopy_evap.c:300-310, :407-417 and :461-471.

    With CARBON on but RC_MODE at its default of RC_JARVIS, the canopy
    resistance is still Jarvis's empirical function and every layer simply
    gets it divided by the leaf area, capped at CANOPY_RSMAX. A tile with no
    leaves gets HUGE_RESIST. Returns [layer, ...].
    """
    per = torch.where(lai > 0.0, rc / torch.clamp(lai, min=_tiny(lai)),
                      torch.full_like(rc, HUGE_RESIST))
    per = torch.clamp(per, max=CANOPY_RSMAX)
    return per.unsqueeze(0).expand((int(ncanopy),) + tuple(per.shape))


def rc_ps_unstressed(ctype, max_carbox_rate, max_e_transport,
                     co2_specificity, nscale, tair, shortwave, apar,
                     elevation, catm, bnd, lai, vpd, press=None,
                     rgas=CONST_RGAS):
    """The part of penman.c calc_rc_ps that does not depend on gsm_inv:
    canopy_assimilation in 'ci' mode (:111-129) and the vapour pressure
    deficit factor (:139-143). Returns (rc0, rsLayer0, vpdfactor).

    Split out because transpiration calls calc_rc_ps once per soil layer
    with a different gsm_inv each time and everything here is the same for
    all of them.
    """
    res = canopy_assimilation(ctype, max_carbox_rate, max_e_transport,
                              co2_specificity, nscale, tair, shortwave, apar,
                              elevation, catm, bnd, lai, 'ci', press=press,
                              rgas=rgas)
    vpdfactor = torch.clamp(1.0 - vpd / CANOPY_CLOSURE,
                            min=CANOPY_VPDMINFACTOR)
    return res['rc'], res['rsLayer'], vpdfactor


def rc_ps_stressed(rc0, rs0, vpdfactor, gsm_inv):
    """penman.c:145-157: the soil moisture stress applied to the unstressed
    resistance. Returns (rc, rsLayer)."""
    den = gsm_inv * vpdfactor
    den = torch.where(den.abs() > 0.0, den, torch.full_like(den, _tiny(den)))
    rc = torch.clamp(rc0 / den, max=CANOPY_RSMAX)
    # :100, the ratio the layers are scaled by. rc0 cannot be zero -- it is
    # 1/gc with gc floored at the machine epsilon -- but the guard keeps the
    # backward pass finite if a caller hands one in.
    safe_rc0 = torch.where(rc0.abs() > 0.0, rc0,
                           torch.full_like(rc0, _tiny(rc0)))
    rs_layer = torch.clamp(rs0 * (rc / safe_rc0), max=CANOPY_RSMAX)
    return rc, rs_layer


def calc_rc_ps(ctype, max_carbox_rate, max_e_transport, co2_specificity,
               nscale, tair, shortwave, apar, elevation, catm, bnd, lai,
               gsm_inv, vpd, press=None, rgas=CONST_RGAS):
    """The canopy resistance photosynthesis implies.  penman.c calc_rc_ps.

    This is RC_MODE = RC_PHOTO: instead of Jarvis's empirical function of
    light, temperature and vapour pressure deficit, the resistance is the
    one the assimilation at an unstressed leaf-internal CO2 would need, then
    opened or closed by the soil moisture stress and the vapour pressure
    deficit. Returns (rc, rsLayer).

    No iteration: canopy_assimilation in 'ci' mode is a closed form, and the
    stress enters afterwards as a ratio.
    """
    rc0, rs0, vpdfactor = rc_ps_unstressed(
        ctype, max_carbox_rate, max_e_transport, co2_specificity, nscale,
        tair, shortwave, apar, elevation, catm, bnd, lai, vpd, press=press,
        rgas=rgas)
    return rc_ps_stressed(rc0, rs0, vpdfactor, gsm_inv)


def canopy_light(nscale_flag, lai, coszen, fdir, par, albedo_par, ncanopy):
    """The canopy's layer bounds, leaf area, absorbed PAR per leaf area and
    nitrogen scaling for one step.  surface_fluxes.c:375-407 and
    calc_Nscale_factors.c. Returns (bnd, lai_layer, fapar, aPAR, nscale).

    `lai`, `coszen`, `fdir`, `par` are [cell, tile]; the layer axis leads.
    """
    bnd = ((torch.arange(int(ncanopy), device=lai.device, dtype=lai.dtype)
            + 1.0) / float(ncanopy)).reshape((int(ncanopy),)
                                             + (1,) * lai.dim())
    lai_layer = canopy_layer_lai(lai, bnd)
    fap = fapar(lai, bnd, coszen, albedo_par, fdir)
    apar = apar_per_leaf(par, fap, lai_layer)
    nscale = nscale_factors(nscale_flag, bnd, lai, coszen)
    return bnd, lai_layer, fap, apar, nscale
