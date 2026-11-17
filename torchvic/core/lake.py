"""VIC's lake and wetland model, vectorised and differentiable.

`options.LAKES` is the last of VIC's five big physics switches this model did
not have. The C is `vic/vic_run/src/lakes.eb.c` -- 2,683 lines around a
one-dimensional thermal lake after Hostetler (1991), with an ice cover, a
snow cover on the ice, a stage-area-volume relation the lake grows and
shrinks along, and machinery for moving soil, snow and carbon storage
between the lake tile and the land as it does.

This module is being built from the bottom up, each piece against upstream's
own compiled function (`build/ref_snow lake*`). What is here so far is the
closed-form layer: the state functions of water, the drag and turbulent
fluxes at the surface, and the albedos. Nothing here iterates or holds
state, so every one of them is a `where` away from being a batched tensor
function, and all of them are exact against VIC.

**On `calc_density`.** It returns the *difference* from fresh water, not a
density -- `rhow - CONST_RHOFW` (lakes.eb.c:791-796). Every caller adds
CONST_RHOFW back or uses the difference directly for buoyancy, so the name
is the trap: a caller that treated the return value as a density would find
lakes about a thousand times too light.
"""
import math

import torch

from .snow import CONST_TKFRZ, snow_albedo

#: vic_physical_constants.h
CONST_RHOFW = 1.000e3           #: density of fresh water, kg/m3
CONST_G = 9.80616               #: m/s2
CONST_KARMAN = 0.4
CONST_MWWV = 18.016             #: kg/kmol
CONST_MWAIR = 28.97
CONST_EPS = CONST_MWWV / CONST_MWAIR
CONST_CPMAIR = 1.013e3          #: J/kg/K, moist air
BAR_PER_KPA = 100.0

#: initialize_parameters.c:76-97, the lake model's own parameters.
LAKE_TMELT = 0.0
LAKE_MAX_SURFACE = 0.6          #: m, thickest surface layer
LAKE_BETA = 0.001
LAKE_FRACMIN = 0.10
LAKE_FRACLIM = 0.02
LAKE_DM = 1.38889e-07           #: m2/s, molecular diffusivity of water
LAKE_SNOWCRIT = 0.05
LAKE_ZWATER = 0.0045            #: m, roughness of open water
LAKE_ZSNOW = 0.005              #: m, roughness of snow on ice
LAKE_RHOSNOW = 250.0            #: kg/m3
LAKE_CONDI = 2.3                #: W/m/K, ice
LAKE_CONDS = 0.7                #: W/m/K, snow
LAKE_LAMISW = 1.5               #: attenuation of shortwave in ice, 1/m
LAKE_LAMILW = 20.0
LAKE_LAMSSW = 6.0
LAKE_LAMSLW = 20.0
LAKE_LAMWSW = 0.3
LAKE_LAMWLW = 1.4
LAKE_A1 = 0.7                   #: the visible fraction of shortwave
LAKE_A2 = 0.3
LAKE_QWTAU = 43200.0            #: s, the lake's mixing timescale
LAKE_MAX_ITER = 50

#: initialize_parameters.c:177, the albedo of fresh snow.
SNOW_NEW_SNOW_ALB = 0.85
#: :182, the snowfall that counts as significant, in MILLIMETRES of water
#: equivalent -- which is the unit `solve_lake` carries snowfall in, as
#: lakes.eb.c:160 and :172 show by dividing it by MM_PER_M to get metres.
#:
#: This is NOT `torchvic.core.snow.TRACE_SNOW_M`. That module works in metres, so
#: the same VIC constant appears there as 3e-5; here the number is VIC's own
#: 0.03 because the quantity it is compared against is VIC's own millimetres.
#: The constant's unit is written down nowhere in VIC, which is exactly how
#: it came to be wrong in the snow module once already, so `alblake` below
#: states the unit of its `newsnow` argument rather than leaving it to be
#: inferred.
SNOW_TRACESNOW = 0.03


def calc_density(t):
    """Water density MINUS fresh water's, kg/m3.  lakes.eb.c:774-798.

    A fifth-order polynomial in temperature, offset by CONST_RHOFW so the
    buoyancy terms downstream work with a small number. The offset is VIC's,
    not a convenience, and the module docstring says why it matters.
    """
    return (999.842594 + 0.06793952 * t - 0.009095290 * t * t
            + 0.0001001685 * t ** 3 - 0.000001120083 * t ** 4
            + 0.000000006536332 * t ** 5) - CONST_RHOFW


def specheat(t):
    """Specific heat of fresh water, J/kg/K.  lakes.eb.c:1280-1292."""
    return (4217.4 - 3.720283 * t + 0.1412855 * t * t
            - 2.654387e-3 * t ** 3 + 2.093236e-5 * t ** 4)


def freezing_point(pressure_kpa):
    """The temperature at which the lake freezes, C.  rhoinit,
    lakes.eb.c:1255-1278.

    Salinity is zero throughout VIC's lake model, so the three salinity
    terms vanish and only the pressure term survives -- but they are written
    out because the C writes them out, and because a reader who sees only
    `-7.53e-3 * p / 100` has no way to know that is a freezing-point
    depression and not a units slip.
    """
    salinity = torch.zeros_like(pressure_kpa)
    return (-0.0575 * salinity
            + 1.710523e-3 * torch.pow(salinity, 1.5)
            - 2.154996e-4 * salinity * salinity
            - 7.53e-3 * pressure_kpa / BAR_PER_KPA)


def lkdrag(t_surf_k, t_air_k, wind, roughness, wind_h):
    """The surface drag coefficient over a lake.  lakes.eb.c:1197-1248.

    A neutral coefficient from the roughness, corrected by the bulk
    Richardson number, floored at whichever is larger of a quarter of the
    neutral value and 6e-4. The two branches for the Richardson denominator
    -- a 0.1 m/s gustiness under stable conditions and 1.0 m/s under
    unstable -- are the C's, at :1215-1220.
    """
    cdrn = (CONST_KARMAN / torch.log(wind_h / roughness)) ** 2
    ribn = wind_h * CONST_G * (1.0 - t_surf_k / t_air_k)
    stable = (t_surf_k / t_air_k) <= 1.0
    ribd = torch.where(stable, wind * wind + 0.01, wind * wind + 1.0)
    rib = ribn / ribd
    # :1229-1234. Both branches are evaluated, so the unstable one's sqrt
    # must be finite where it is not selected: -cdrn*rib is negative there.
    unstable_arg = torch.clamp(-cdrn * rib, min=0.0)
    cdr = torch.where(rib < 0.0,
                      cdrn * (1.0 + 24.5 * torch.sqrt(unstable_arg)),
                      cdrn / (1.0 + 11.5 * rib))
    cdrmin = torch.clamp(0.25 * cdrn, min=6.0e-4)
    return torch.maximum(cdr, cdrmin)


def latsens(t_surf_k, t_cutoff_k, hice, tair, wind, pressure, vp,
            air_density, wind_h):
    """Evaporation and sensible heat from the lake surface.
    lakes.eb.c:533-614.  Returns (evap in mm/s, sensible in W/m2).

    The saturation vapour pressure over the surface takes one of two forms:
    water when the lake is open and above freezing, ice otherwise
    (:571-587). They are not the same curve and the difference is what makes
    a frozen lake stop evaporating.

    `evap` is negative of the humidity gradient times the drag, so a lake
    drier than the air gains water -- condensation -- and the sign is VIC's.
    """
    ice = hice > 0.0
    rough = torch.where(ice, torch.full_like(hice, LAKE_ZSNOW),
                        torch.full_like(hice, LAKE_ZWATER))
    drag = lkdrag(t_surf_k, tair + CONST_TKFRZ, wind, rough, wind_h)

    tc = t_surf_k - CONST_TKFRZ
    # :575-578 over open water, :583-586 over ice. Both evaluated, so the
    # ice form's denominator (Tsurf - 7.66 K) must not be met at zero: it
    # is -273 K away from it at every temperature a lake reaches.
    eog_water = 0.611 * torch.exp(17.269 * tc / (tc + 237.3))
    eog_ice = 0.611 * torch.exp(21.874 * tc / (t_surf_k - 7.66))
    open_water = (hice <= 0.0) & (t_surf_k > t_cutoff_k)
    eog = torch.where(open_water, eog_water, eog_ice)

    qlake = CONST_EPS * (eog / (pressure - 0.378 * eog))
    qair = CONST_EPS * (vp / (pressure - 0.378 * vp))
    evap = -drag * wind * air_density * (qair - qlake)
    qsen = drag * wind * air_density * CONST_CPMAIR * (
        tair + CONST_TKFRZ - t_surf_k)
    return evap, qsen


def alblake(t_cutoff, tair, snowalbedo, newsnow, coldcontent, dt, last_snow,
            swq, melting, day_in_year, latitude):
    """The lake's three albedos -- snow, ice and water -- and the snow-age
    bookkeeping that goes with them.  lakes.eb.c:621-706.

    Returns (snowalbedo, albs, albi, albw, last_snow, melting).

    **`newsnow` is in millimetres of water equivalent**, which is what
    solve_lake carries (lakes.eb.c:160). The rest of this model works in
    metres; the lake does not, and passing metres here would put every
    snowfall three orders of magnitude below SNOW_TRACESNOW and leave the
    albedo clock permanently running.

    The ice albedo falls with how far the air is above the melting point,
    capped at 20 degrees (:640-652), and is the mean of a "glare" and a
    "snow" value. The water albedo is the constant 0.15 at :705. The snow
    albedo is the ordinary snow-pack decay, which is why `last_snow` and
    `MELTING` have to be carried: they are the decay's state, and VIC keeps
    them on the lake as well as the land.
    """
    above = tair - t_cutoff
    warm = torch.clamp(above, min=0.0, max=20.0)
    positive = above > 0.0
    albgl = torch.where(positive, 0.4 - 0.011 * warm,
                        torch.full_like(warm, 0.4))
    albgs = torch.where(positive, 0.6 - 0.0245 * warm,
                        torch.full_like(warm, 0.6))
    albi = 0.5 * albgs + 0.5 * albgl

    # :657-666 -- days since the last significant snowfall.
    fresh = newsnow > SNOW_TRACESNOW
    bare = swq == 0.0
    last_snow = torch.where(
        fresh, torch.ones_like(last_snow),
        torch.where(bare, torch.zeros_like(last_snow), last_snow + 1.0))

    # :669-686 -- the seasonal melting flag, and fresh snow ends it.
    has_snow = swq > 0.0
    season = torch.where(
        latitude >= 0.0,
        (day_in_year > 60) & (day_in_year < 273),
        (day_in_year < 60) | (day_in_year > 273))
    melting = has_snow & (coldcontent >= 0.0) & season & ~fresh

    # :688-698.
    # snow_utility.c's own decay. This model's `snow_albedo` takes the
    # fresh-snow albedo where VIC takes the current one, because VIC's
    # current value is overwritten in every branch that uses it and the
    # fresh-snow value is the run parameter here; the arithmetic is the
    # same function.
    decayed = snow_albedo(newsnow, torch.full_like(snowalbedo,
                                                   SNOW_NEW_SNOW_ALB),
                          swq, coldcontent, dt, last_snow, melting,
                          trace_snow=SNOW_TRACESNOW)
    snowalbedo = torch.where(
        has_snow, decayed,
        torch.where(newsnow > 0.0,
                    torch.full_like(snowalbedo, SNOW_NEW_SNOW_ALB),
                    torch.zeros_like(snowalbedo)))
    albs = torch.where(newsnow > 0.0,
                       torch.full_like(snowalbedo, SNOW_NEW_SNOW_ALB),
                       snowalbedo)
    albw = torch.full_like(albi, 0.15)
    return snowalbedo, albs, albi, albw, last_snow, melting


#: vic_def.h:68.
MAX_LAKE_NODES = 20
#: vic_physical_constants.h -- the pi the lake's Ekman profile uses.
CONST_PI = 3.141592653589793
#: float.h's DBL_EPSILON, which VIC uses as its threshold for "no water".
DBL_EPSILON = 2.220446049250313e-16


def tridia(a, b, c, y):
    """Solve a tridiagonal system.  lakes.eb.c:1716-1758.

    `a`, `b`, `c` and `y` are [node, ...]; every trailing axis is a separate
    system solved at once. Returns x with the same shape.

    An LU decomposition and back-substitution, written as the C writes it:
    the forward sweeps stop one short of the last node and that node is
    eliminated on its own (:1752-1755). Pulling it into the loop would give
    the same answer in exact arithmetic and a different one in floating
    point, and this is the inner solve of the lake's temperature profile.
    """
    n = a.shape[0]
    alpha = [1.0 / b[0]]
    gamma = [c[0] * alpha[0]]
    for i in range(1, n - 1):
        alpha.append(1.0 / (b[i] - a[i] * gamma[i - 1]))
        gamma.append(c[i] * alpha[i])

    x = [y[0] * alpha[0]]
    for i in range(1, n - 1):
        x.append((y[i] - a[i] * x[i - 1]) * alpha[i])
    # :1752-1755 -- the last row, eliminated against the second last.
    x.append((y[n - 1] - a[n - 1] * x[n - 2])
             / (b[n - 1] - a[n - 1] * gamma[n - 2]))
    for i in range(n - 2, -1, -1):
        x[i] = x[i] - gamma[i] * x[i + 1]
    return torch.stack(x, 0)


def eddy(freezeflag, wind, water_density, lat, numnod, dz, surfdz):
    """Eddy diffusivity at each lake node, m2/s.  lakes.eb.c:802-926,
    Hostetler and Bartlein (1990) equations 5 to 9.

    `water_density` is [node, ...] and is the *difference* from fresh water,
    as `calc_density` returns it; the C adds 1e3 back inside the
    Brunt-Vaisala frequency at :883 and so does this.

    Under ice -- `freezeflag != 1` -- there is no wind stirring and the
    whole column falls back to molecular diffusivity. Note that the C's
    branch reads the flag the opposite way round from its own comment;
    what is reproduced is the code.
    """
    dm = torch.full_like(water_density, LAKE_DM)
    molecular = freezeflag != 1

    # :838-845 -- a floor under the wind, for the stability of the
    # Richardson number below rather than for any physical reason.
    w = torch.clamp(wind, min=1.0)
    ks = 6.6 * torch.sqrt(torch.sin(torch.abs(lat) * CONST_PI / 180.0)) \
        * torch.pow(w, -1.84)
    ws = 0.0012 * w
    po = 1.0
    radmax = 4.0e4

    # :870 -- the spacing between node centres; the surface layer is
    # thinner, so its half-spacing to the next node is its own.
    zhalf0 = (surfdz + dz) * 0.5

    out = []
    for k in range(numnod - 1):
        zh = zhalf0 if k == 0 else dz
        dpdz = (water_density[k + 1] - water_density[k]) / zh
        n2 = (dpdz / (1.0e3 + water_density[k])) * 9.8
        z = surfdz + float(k) * dz
        # :890-907 -- the second Richardson term, capped both ways. The
        # `> 1e8` test short-circuits an exponential that would overflow.
        big = (z * torch.exp(ks * z) / ws) > 1.0e8
        rad = 1.0 + 40.0 * n2 * (CONST_KARMAN * z) ** 2 / (
            ws * ws * torch.exp(-2.0 * ks * z))
        rad = torch.clamp(rad, 1.0, radmax)
        rad = torch.where(big, torch.full_like(rad, radmax), rad)
        ri = (-1.0 + torch.sqrt(rad)) / 20.0
        out.append(LAKE_DM + (CONST_KARMAN * ws * z / po)
                   * torch.exp(-ks * z) / (1.0 + 37.0 * ri * ri))
    # :921-925 -- the deepest node takes the one above it.
    out.append(out[-1] if out else dm[0])
    de = torch.stack(out, 0)
    return torch.where(molecular.expand_as(de), dm, de)


def colavg(t_water, t_ice, lake_fraction, numnod, dz, surfdz):
    """The column temperature as a mass-weighted mean of the water and ice
    fractions, and the density it implies.  lakes.eb.c:717-772.
    Returns (finaltemp, density), each [node, ...].

    `lake_fraction` is the ice-covered fraction -- the C calls it `lakeprv`
    and weights the ICE column by it, which is the opposite of what the name
    suggests; see :752-758.
    """
    z = torch.stack([(surfdz if k == 0 else dz).expand_as(t_water[0])
                     if torch.is_tensor(surfdz) else
                     torch.full_like(t_water[0], surfdz if k == 0 else dz)
                     for k in range(numnod)], 0)
    rho_w = calc_density(t_water) + CONST_RHOFW
    rho_i = calc_density(t_ice) + CONST_RHOFW
    f = lake_fraction
    num = (1.0 - f) * t_water * z * rho_w + f * t_ice * z * rho_i
    den = (1.0 - f) * z * rho_w + f * z * rho_i
    final = num / den
    return final, calc_density(final)


def energycalc(finaltemp, numnod, dz, surfdz, surface, cp, density):
    """The thermal energy the column holds, J.  lakes.eb.c:1765-1806.

    `surface` is [node + 1, ...] -- the horizontal area of water at each
    node boundary -- and each layer takes the mean of the two that bound it,
    except the deepest, which takes half of its upper boundary alone
    (:1788-1793) because the lake closes to a point below it.
    """
    total = None
    for k in range(numnod):
        thick = surfdz if k == 0 else dz
        # :1779-1799 tests `k == 0` BEFORE `k == numnod - 1`, so on a
        # one-node lake the surface layer's rule wins, not the deepest
        # layer's. They agree only because surface[1] is zero there.
        area = (surface[k] / 2.0 if (k == numnod - 1 and k != 0)
                else (surface[k] + surface[k + 1]) / 2.0)
        e = ((finaltemp[k] + CONST_TKFRZ) * thick
             * (CONST_RHOFW + density[k]) * cp[k] * area)
        total = e if total is None else total + e
    return total


def _bc(v, like):
    """Broadcast a layer thickness -- a float or a per-column tensor --
    to the shape of a [node, ...] field."""
    return (v.expand_as(like) if torch.is_tensor(v)
            else torch.full_like(like, v))


def _exp(x):
    """exp() for a value that may be a float or a tensor.

    The depth attenuations in `temp_area` are exponentials of a layer
    thickness. `math.exp` refuses a tensor, which pinned the lake's geometry
    to a Python scalar and so to one lake per call; this lets `dz` and
    `surfdz` vary down the batch like everything else.
    """
    return torch.exp(x) if torch.is_tensor(x) else math.exp(x)


def temp_area(sw_visible, sw_nir, surface_force, t, water_density, de, dt,
              surface, numnod, dz, surfdz, cp):
    """One step of the lake's temperature profile.  lakes.eb.c:1294-1560.
    Returns (Tnew, energy_out_bottom, joulenew), with Tnew [node, ...].

    A Crank-Nicolson diffusion equation on a column whose horizontal area
    shrinks with depth, forced by shortwave absorbed in each layer and by
    whatever the surface energy balance hands down. Shortwave is split into
    a visible and a near-infrared band with different attenuation lengths
    (LAKE_LAMWSW and LAKE_LAMWLW), which is why a lake warms at the surface
    and a soil column does not.

    Three details are the C's and are reproduced rather than tidied:

    * The deepest node uses `surface[numnod-1]` for both of its boundary
      areas (:1463-1465), so it has no area convergence -- the lake is
      treated as a cylinder below the last node.
    * `energy_out_bottom` is accumulated over the column and then thrown
      away and recomputed from the deepest node alone (:1487-1493). The
      accumulation is dead code in the C; it is not reproduced, and this
      note is why.
    * `tridia` is called with (c, a, b) in that order (:1541) -- the C's
      `a` is the diagonal and its `b` and `c` are the off-diagonals, which
      is the opposite naming from the solver's own arguments.
    """
    # :1336-1352 -- layer thicknesses and the spacing between node centres.
    z = [surfdz if k == 0 else dz for k in range(numnod)]
    zhalf = [dz] * numnod
    zhalf[0] = 0.5 * (z[0] + z[1]) if numnod > 1 else 0.5 * z[0]

    lam_sw, lam_lw = LAKE_LAMWSW, LAKE_LAMWLW

    def sfc(k):
        return surface[k]

    # --- the surface layer, :1359-1380 ---------------------------------
    # :1359-1361 reads surface[1] whatever `numnod` is. On a one-node lake
    # that entry is ZERO -- compute_derived_lake_dimensions fills
    # activenod + 1 entries and the second is the area at the bottom, which
    # for a lake one layer deep is the point the basin closes to. So
    # surface_avg is half surface[0] and the surface forcing is doubled.
    # Substituting surface[0] for it, which looks like the safe reading,
    # halves the energy into a shallow lake: case 0 of the solve_lake
    # oracle then cooled to +0.17 C where VIC freezes to -0.0072 and puts
    # 203 m3 of ice on the water.
    s1, s2 = sfc(0), sfc(1)
    savg = (s1 + s2) / 2.0
    t1 = ((sw_visible * (s1 - s2 * _exp(-lam_sw * surfdz))
           + sw_nir * (s1 - s2 * _exp(-lam_lw * surfdz))) / savg
          + surface_force * s1 / savg)

    if numnod == 1:
        tnew = t[0] + (t1 * dt) / ((1.0e3 + water_density[0]) * cp[0] * z[0])
        tnew = tnew.unsqueeze(0)
        eob = torch.zeros_like(t1)
        return tnew, eob, energycalc(tnew, numnod, dz, surfdz, surface, cp,
                                     water_density)

    cnextra = 0.5 * (s2 / savg) * (de[0] / zhalf[0]) * (
        (t[1] - t[0]) / z[0])
    d = [t[0] + (t1 * dt) / ((1.0e3 + water_density[0]) * cp[0] * z[0])
         + cnextra * dt]

    # --- every node but the deepest, :1404-1454 -------------------------
    for k in range(1, numnod - 1):
        top = surfdz + (k - 1) * dz
        bot = surfdz + k * dz
        s1, s2 = sfc(k), sfc(k + 1)
        savg = (s1 + s2) / 2.0
        t1 = ((sw_visible * (s1 * _exp(-lam_sw * top)
                             - s2 * _exp(-lam_sw * bot))
               + sw_nir * (s1 * _exp(-lam_lw * top)
                           - s2 * _exp(-lam_lw * bot))) / savg)
        term1 = 0.5 * (1.0 / savg) * ((de[k] / zhalf[k])
                                      * ((t[k + 1] - t[k]) / z[k])) * s2
        term2 = 0.5 * (-1.0 / savg) * ((de[k - 1] / zhalf[k - 1])
                                       * ((t[k] - t[k - 1]) / z[k])) * s1
        d.append(t[k]
                 + (t1 * dt) / ((1.0e3 + water_density[k]) * cp[k] * z[k])
                 + (term1 + term2) * dt)

    # --- the deepest node, :1460-1500 -----------------------------------
    k = numnod - 1
    s1 = s2 = savg = sfc(k)
    top, bot = surfdz + (k - 1) * dz, surfdz + k * dz
    t1 = ((sw_visible * (s1 * _exp(-lam_sw * top) - s2 * _exp(-lam_sw * bot))
           + sw_nir * (s1 * _exp(-lam_lw * top) - s2 * _exp(-lam_lw * bot)))
          / savg)
    cnextra = 0.5 * (-1.0 * s1 / savg) * ((de[k - 1] / zhalf[k - 1])
                                          * ((t[k] - t[k - 1]) / z[k]))
    # :1487-1493 -- reset, then the deepest node alone, per unit surface area.
    eob = s2 * (sw_visible * _exp(-lam_sw * bot)
                + sw_nir * _exp(-lam_lw * bot)) / sfc(0)
    d.append(t[k] + (t1 * dt) / ((1.0e3 + water_density[k]) * cp[k] * z[k])
             + cnextra * dt)

    # --- the tridiagonal coefficients, :1505-1538 -----------------------
    zero = torch.zeros_like(d[0])
    a = [None] * numnod          # the diagonal, the C's `a`
    b = [zero] * numnod          # the upper off-diagonal, the C's `b`
    c = [zero] * numnod          # the lower off-diagonal, the C's `c`
    s2, savg = sfc(1), (sfc(0) + sfc(1)) / 2.0
    b[0] = -0.5 * (de[0] / zhalf[0]) * (dt / z[0]) * s2 / savg
    a[0] = 1.0 - b[0]
    for k in range(1, numnod - 1):
        s1, s2 = sfc(k), sfc(k + 1)
        savg = (s1 + s2) / 2.0
        b[k] = -0.5 * (de[k] / zhalf[k]) * (dt / z[k]) * s2 / savg
        c[k] = -0.5 * (de[k - 1] / zhalf[k - 1]) * (dt / z[k]) * s1 / savg
        a[k] = 1.0 - b[k] - c[k]
    k = numnod - 1
    c[k] = -0.5 * (de[k] / zhalf[k]) * (dt / z[k])      # s1 / savg is 1 here
    a[k] = 1.0 - c[k]

    tnew = tridia(torch.stack(c, 0), torch.stack(a, 0), torch.stack(b, 0),
                  torch.stack(d, 0))
    joulenew = energycalc(tnew, numnod, dz, surfdz, surface, cp,
                          water_density)
    return tnew, eob, joulenew


#: How many times `tracer_mixer`'s scan may restart before this gives up.
#: The C restarts whenever mixing leaves something denser above the mixed
#: block, and each restart mixes at least one more node into it, so a column
#: of N nodes cannot restart more than N times. The cap is generous against
#: that argument rather than tuned, and being hit is an error, not a limit.
MIXER_MAX_PASSES = 4 * MAX_LAKE_NODES


def tracer_mixer(t, mixdepth, surface, numnod, dz, surfdz, cp):
    """Convective mixing.  lakes.eb.c:1567-1692.
    Returns (T, density, mixdepth), with T and density [node, ...].

    Water that is denser than the water beneath it sinks, and the column
    overturns until it is stable. VIC scans down for the first inversion,
    mixes everything from the last stable node to one below the inversion
    into a single mass-weighted temperature, and -- if that leaves something
    denser still above the mixed block -- starts the scan again from the top
    (:1674-1678, `mixprev = 0; k = -1`).

    That restart is the whole difficulty in vectorising this. The scan is
    data-dependent per column, so what runs here is the C's scan with a
    per-column mask: a column that restarts drops out of the current pass
    and rejoins at the top of the next one, which is exactly what `k = -1`
    does to it. Columns that reach the end of a scan without restarting are
    finished and are not touched again.

    `mixdepth` is carried in and out because VIC does: it records how deep
    the mixing reached, and `solve_lake` uses it to decide where to put the
    energy the surface lost.
    """
    idx = torch.arange(numnod, device=t.device).reshape(
        (numnod,) + (1,) * (t.dim() - 1))
    z = torch.where(idx == 0, _bc(surfdz, t), _bc(dz, t))
    area = surface[:numnod]

    t = t.clone()
    rho = calc_density(t)
    mixdepth = mixdepth.clone()
    needs_scan = torch.ones_like(t[0], dtype=torch.bool)

    for _pass in range(MIXER_MAX_PASSES):
        if not bool(needs_scan.any()):
            break
        active = needs_scan.clone()
        needs_scan = torch.zeros_like(needs_scan)
        mixprev = torch.zeros_like(t[0], dtype=torch.long)
        for k in range(numnod - 1):
            unstable = active & (rho[k] > rho[k + 1])
            # :1683-1689 -- every active column with no inversion HERE
            # advances its stable depth by one node, whether or not some
            # other column in the batch is unstable at this k. Skipping the
            # whole node when any column was unstable -- which is what an
            # `if not unstable.any(): continue` does -- leaves the stable
            # columns with mixprev = 0, and they then mix from the surface
            # down. On case 20 of the solve_lake oracle that mixed the whole
            # column to 3.94 C where VIC leaves the top node at 3.76 and
            # mixes only the bottom three. Alone the column was right;
            # only in a batch with an unstable neighbour was it wrong.
            mixprev = torch.where(active & ~unstable,
                                  torch.full_like(mixprev, k + 1), mixprev)
            if not bool(unstable.any()):
                continue
            # :1604-1609 -- record how deep the mixing reached.
            deeper = unstable & (mixprev == 0) & ((k + 1) > mixdepth)
            mixdepth = torch.where(deeper, torch.full_like(mixdepth, k + 1),
                                   mixdepth)
            # :1616-1636 -- one mass-weighted temperature over mixprev..k+1.
            inblock = (idx >= mixprev.unsqueeze(0)) & (idx <= k + 1)
            heatcon = z * (1.0e3 + rho) * cp * area
            hc = torch.where(inblock, heatcon, torch.zeros_like(heatcon))
            tav = (t * hc).sum(0) / hc.sum(0)
            densnew = calc_density(tav)
            # :1650-1656 -- the densest water above the block, before it is
            # overwritten.
            above = idx < mixprev.unsqueeze(0)
            rho_max = torch.where(above, CONST_RHOFW + rho,
                                  torch.zeros_like(rho)).amax(0)
            write = inblock & unstable.unsqueeze(0)
            t = torch.where(write, tav.unsqueeze(0).expand_as(t), t)
            rho = torch.where(write, densnew.unsqueeze(0).expand_as(rho), rho)
            # :1674-1678 -- mixing made a new inversion higher up; restart.
            restart = unstable & (rho_max > (CONST_RHOFW + densnew))
            needs_scan = needs_scan | restart
            active = active & ~restart
            # A column that mixed but did not restart keeps its mixprev, as
            # the C does: the `else` that advances it is not taken.
        # Columns still active at the end of a scan are stable.
    else:                                                # pragma: no cover
        if bool(needs_scan.any()):
            raise RuntimeError(
                'tracer_mixer did not settle in %d scans; the column keeps '
                'making new inversions, which means the density profile or '
                'the mixing block is wrong rather than that the cap is too '
                'low' % MIXER_MAX_PASSES)

    # :1690-1692 -- the density is rebuilt from the mixed temperatures.
    return t, calc_density(t), mixdepth


#: vic_physical_constants.h -- ice, and the latent heat of fusion.
CONST_RHOICE = 0.917e3          #: kg/m3
CONST_LATICE = 3.337e5          #: J/kg


def iceform(t, t_cutoff, fracprv, numnod, dt, dz, surfdz, cp, surface,
            lvolume):
    """New ice forming on the open part of the lake.  lakes.eb.c:936-1038.
    Returns (T, qfusion, areaadd, new_ice_height, new_ice_water_eq).

    Every node below the freezing point gives up the heat that would take it
    back to freezing, and is set to freezing; the heat is turned into ice.
    The ice spreads at a fixed thickness LAKE_FRACMIN until it covers the
    open water, and thickens after that (:1027-1037).

    Three branches decide how much ice there can be (:1002-1022): enough
    unfrozen water for all of it, only enough for some, or none. VIC's note
    in the middle branch -- that the ice temperature should be adjusted for
    the extra fusion heat and is not -- is left as it stands.
    """
    idx = torch.arange(numnod, device=t.device).reshape(
        (numnod,) + (1,) * (t.dim() - 1))
    freezing = t < t_cutoff
    thick = torch.where(idx == 0, _bc(surfdz, t), _bc(dz, t))
    # :970-984 -- the deepest node has no lower boundary to average with.
    # The C tests `j == 0` first, so a one-node lake takes the SURFACE
    # layer's rule, half of surface[0] rather than all of it. Taking the
    # deepest layer's rule there put twice as much ice on a shallow lake:
    # 406 m3 against VIC's 203 on case 0 of the solve_lake oracle.
    deepest = (idx == numnod - 1) & (idx != 0)
    area = torch.where(deepest, surface[:numnod],
                       (surface[:numnod] + surface[1:numnod + 1]) / 2.0)
    extra = ((t_cutoff - t) * thick * CONST_RHOFW * cp * (1.0 - fracprv)
             * area)
    total = torch.where(freezing, extra, torch.zeros_like(extra)).sum(0)
    t = torch.where(freezing, t_cutoff.expand_as(t), t)

    water_eq = total / (CONST_RHOFW * CONST_LATICE)
    open_area = (1.0 - fracprv) * surface[0]
    enough = lvolume > water_eq
    some = (~enough) & (lvolume > 0.0)

    qf_full = total / (dt * surface[0] * (1.0 - fracprv))
    di_full = total / (CONST_LATICE * CONST_RHOICE)
    # :1012-1019 -- limited by the water there is.
    eq_some = lvolume
    di_some = eq_some * CONST_RHOFW / CONST_RHOICE
    qf_some = di_some / (dt * surface[0] * (1.0 - fracprv))

    zero = torch.zeros_like(total)
    new_ice_water_eq = torch.where(enough, water_eq,
                                   torch.where(some, eq_some, zero))
    di = torch.where(enough, di_full, torch.where(some, di_some, zero))
    qfusion = torch.where(enough, qf_full, torch.where(some, qf_some, zero))

    # :1027-1037 -- spread at LAKE_FRACMIN until the open water is covered.
    areaadd = di / LAKE_FRACMIN
    spread = areaadd > open_area
    new_ice_height = torch.where(
        spread, di / torch.where(open_area > 0.0, open_area,
                                 torch.ones_like(open_area)),
        torch.full_like(di, LAKE_FRACMIN))
    areaadd = torch.where(spread, open_area, areaadd)
    return t, qfusion, areaadd, new_ice_height, new_ice_water_eq


def icerad(sw, hi, hs):
    """Shortwave through the snow and ice cover.  lakes.eb.c:1043-1106,
    Patterson and Hamblin (1988) equations 7 and 8.
    Returns (avgcond, SWnet, SW_under_ice).

    `avgcond` is the combined snow-and-ice thermal resistance, and the C
    writes it as `(hs*CONDI + hi*CONDS) / (CONDI*CONDS)` -- the two
    thicknesses cross-multiplied by the *other* layer's conductivity, which
    is the series resistance hs/CONDS + hi/CONDI written over a common
    denominator. It reads like a transposition and is not one.

    The two bands attenuate at different rates in snow and in ice, which is
    why a few centimetres of snow shut the light off and a metre of clear
    ice does not.
    """
    avgcond = (hs * LAKE_CONDI + hi * LAKE_CONDS) / (LAKE_CONDI * LAKE_CONDS)
    a = -(1.0 - torch.exp(-LAKE_LAMSSW * hs)) / (LAKE_CONDS * LAKE_LAMSSW)
    b = -(torch.exp(-LAKE_LAMSSW * hs)
          * (1.0 - torch.exp(-LAKE_LAMISW * hi))) / (LAKE_CONDI * LAKE_LAMISW)
    c = -(1.0 - torch.exp(-LAKE_LAMSLW * hs)) / (LAKE_CONDS * LAKE_LAMSLW)
    d = -(torch.exp(-LAKE_LAMSLW * hs)
          * (1.0 - torch.exp(-LAKE_LAMILW * hi))) / (LAKE_CONDI * LAKE_LAMILW)
    swnet = sw * LAKE_A1 * (a + b) + sw * LAKE_A2 * (c + d)
    sw_under = (LAKE_A1 * sw * (1.0 - torch.exp(
        -(LAKE_LAMSSW * hs + LAKE_LAMISW * hi)))
        + LAKE_A2 * sw * (1.0 - torch.exp(
            -(LAKE_LAMSLW * hs + LAKE_LAMILW * hi))))
    return avgcond, swnet, sw_under


def lakeice(sw_ice, fracice, dt, snowflux, qw, sw_absorbed, ice_water_eq,
            volume, sarea):
    """Ice growing or melting at its base.  lakes.eb.c:1111-1192.
    Returns (qf, energy_ice_melt_bot, ice_water_eq, dibot).

    The heat that reaches the ice-water interface either freezes water onto
    the bottom of the ice or melts it off. Freezing is limited by the water
    there is left to freeze (:1170-1183); melting is not limited by the ice
    there is, and the C says so in a comment at :1190-1191 -- the energy
    that should have gone to cooling the ice or warming the water is simply
    lost. That is reproduced.
    """
    qf = snowflux + sw_ice - sw_absorbed
    energy_melt = qf - qw
    dibot = (energy_melt / (CONST_RHOICE * CONST_LATICE)) * dt
    per_m = sarea * fracice * CONST_RHOICE / CONST_RHOFW
    new_water_eq = dibot * per_m

    freezing = dibot > 0.0
    room = volume - ice_water_eq
    fits = room >= new_water_eq
    has_room = room > 0.0
    safe_per_m = torch.where(per_m > 0.0, per_m, torch.ones_like(per_m))

    # :1170-1183, the freezing branch.
    di_freeze = torch.where(fits, dibot,
                            torch.where(has_room, room / safe_per_m,
                                        torch.zeros_like(dibot)))
    eq_freeze = torch.where(fits, ice_water_eq + new_water_eq,
                            torch.where(has_room, volume, ice_water_eq))
    # :1185-1191, the melting branch.
    eq_melt = torch.clamp(ice_water_eq + new_water_eq, min=0.0)

    dibot = torch.where(freezing, di_freeze, dibot)
    ice_water_eq = torch.where(freezing, eq_freeze, eq_melt)
    return qf, energy_melt, ice_water_eq, dibot


def get_sarea(z, basin, depth):
    """Surface area of liquid water at a given depth, m2.  lake_utils.c:30-66.

    `z` and `basin` are [level, ...] -- the depth and area of each level of
    the basin's hypsometry, deepest last -- and `depth` is the current
    liquid depth. Linear between the levels, and the full basin area above
    the top one (:44-46).

    The C returns an error when the depth is non-zero and no level bracketed
    it, which cannot happen for a depth inside the basin; here the bracket
    is a mask and a depth below the deepest level gives zero, which is the
    same answer the C's initialisation leaves.
    """
    n = z.shape[0] - 1
    above = depth > z[0]
    area = torch.zeros_like(depth)
    for i in range(n):
        inside = (depth <= z[i]) & (depth > z[i + 1])
        span = z[i] - z[i + 1]
        span = torch.where(span.abs() > 0.0, span, torch.ones_like(span))
        interp = basin[i + 1] + (depth - z[i + 1]) * (basin[i]
                                                      - basin[i + 1]) / span
        area = torch.where(inside, interp, area)
    return torch.where(above, basin[0].expand_as(area), area)


def get_volume(z, basin, depth, maxvolume):
    """Liquid water the basin holds at a given depth, m3.
    lake_utils.c:71-113.

    The basin is a stack of truncated cones: a level entirely below the
    water contributes its whole trapezoidal volume, the one the surface
    falls in contributes the part below the surface, and the C accumulates
    both in the same downward loop.
    """
    n = z.shape[0] - 1
    total = torch.zeros_like(depth)
    for i in range(n - 1, -1, -1):
        full = depth >= z[i]
        part = (depth < z[i]) & (depth >= z[i + 1])
        span = z[i] - z[i + 1]
        safe = torch.where(span.abs() > 0.0, span, torch.ones_like(span))
        vol_full = (basin[i] + basin[i + 1]) * span / 2.0
        m = (basin[i] - basin[i + 1]) / safe
        d = depth - z[i + 1]
        vol_part = d * (m * d / 2.0 + basin[i + 1])
        total = total + torch.where(full, vol_full,
                                    torch.where(part, vol_part,
                                                torch.zeros_like(total)))
    # :84-87 -- above the basin's top the volume is capped, and the C then
    # carries on adding to it in the loop below, so the cap is a floor on
    # the accumulation rather than a replacement for it.
    return torch.where(depth > z[0], maxvolume + total, total)


def get_depth(z, basin, volume, maxvolume, maxdepth):
    """The depth a given volume of water stands at, m.  lake_utils.c:118-192.

    The inverse of `get_volume`, filled from the bottom up. Within a level
    the area varies linearly with depth, so the volume is quadratic in it
    and the depth is a root -- the C takes the positive one (:163-170).
    Where a level has parallel sides the quadratic degenerates and the C
    divides instead (:157-160); both branches are evaluated here, so the
    degenerate one is fed a non-zero slope it does not use.

    Above the basin's capacity the lake stands on its own full area
    (:129-132), which is what lets a lake flood beyond its banks; below
    DBL_EPSILON it is dry (:133-135), and that guard is not cosmetic --
    a lake whose water is all ice has a liquid volume of a few times
    1e-16 rather than zero, and without the guard it reports a depth and
    therefore a surface area where VIC reports none.
    """
    n = z.shape[0] - 1
    volume = torch.clamp(volume, min=0.0)
    depth = torch.zeros_like(volume)
    left = volume.clone()
    for k in range(n - 1, -1, -1):
        span = z[k] - z[k + 1]
        cap = span * (basin[k] + basin[k + 1]) / 2.0
        fills = left > cap
        partial = (~fills) & (left > 0.0)
        flat = basin[k] == basin[k + 1]
        m = basin[k] - basin[k + 1]
        safe_m = torch.where(m.abs() > 0.0, m, torch.ones_like(m)) / \
            torch.where(span.abs() > 0.0, span, torch.ones_like(span))
        safe_b = torch.where(basin[k + 1].abs() > 0.0, basin[k + 1],
                             torch.ones_like(basin[k + 1]))
        d_flat = left / safe_b
        # The deepest level closes to a point, so basin[k+1] is zero there
        # and with no water left the discriminant is exactly zero -- where
        # sqrt is finite and its derivative is not. The branch is selected
        # rather than clamped so the zero carries a zero gradient.
        disc = basin[k + 1] * basin[k + 1] + 2.0 * safe_m * left
        pos = disc > 0.0
        root = torch.where(pos, torch.sqrt(torch.where(
            pos, disc, torch.ones_like(disc))), torch.zeros_like(disc))
        d_slope = (-basin[k + 1] + root) / safe_m
        d_part = torch.where(flat, d_flat, d_slope)
        depth = depth + torch.where(fills, span,
                                    torch.where(partial, d_part,
                                                torch.zeros_like(depth)))
        left = torch.where(fills, left - cap, torch.zeros_like(left))
    over = volume >= maxvolume
    depth = torch.where(over,
                        maxdepth + (volume - maxvolume) / basin[0], depth)
    # :133-135 -- DBL_EPSILON, the C's own threshold for "no water".
    return torch.where(volume < DBL_EPSILON, torch.zeros_like(depth), depth)


def derived_lake_dimensions(ldepth, ice_water_eq, z, basin, maxdepth,
                            maxvolume, max_nodes=MAX_LAKE_NODES):
    """The node grid and the areas a lake of a given depth is solved on.
    compute_derived_lake_dimensions.c:34-104.
    Returns (surfdz, dz, activenod, surface, sarea, volume).

    VIC re-derives this every step, because the lake's depth changes and
    with it the number of layers it is worth solving. The rule
    (:45-72) is: a surface layer of at most LAKE_MAX_SURFACE, then as many
    equal layers as fit, capped at MAX_LAKE_NODES; a lake too shallow for
    two layers is split in half; a lake shallower than that is one layer;
    and a dry basin has none.

    `activenod` therefore varies from cell to cell and from step to step,
    which a batched model cannot have as an array length. It is returned as
    an integer tensor and `surface` is built to `max_nodes + 1` with the
    inactive entries left at zero, so the caller carries one shape and masks.
    That is a structural difference from the C and it is named here rather
    than discovered: a routine that sums over `surface` without masking will
    read zeros, not garbage, which is the safe direction.
    """
    two_layers = (ldepth > LAKE_MAX_SURFACE) & (ldepth < 2 * LAKE_MAX_SURFACE)
    many = ldepth >= 2 * LAKE_MAX_SURFACE
    one = (~two_layers) & (~many) & (ldepth > 0.0)

    nod_many = torch.clamp((ldepth / LAKE_MAX_SURFACE).to(torch.long),
                           max=max_nodes)
    activenod = torch.where(many, nod_many,
                            torch.where(two_layers,
                                        torch.full_like(nod_many, 2),
                                        torch.where(one,
                                                    torch.ones_like(nod_many),
                                                    torch.zeros_like(nod_many))))
    surfdz = torch.where(many, torch.full_like(ldepth, LAKE_MAX_SURFACE),
                         torch.where(two_layers, ldepth / 2.0,
                                     torch.where(one, ldepth,
                                                 torch.zeros_like(ldepth))))
    # :61 -- the remaining depth split over the layers below the surface one.
    below = torch.clamp(activenod - 1, min=1).to(ldepth.dtype)
    dz = torch.where(many, (ldepth - surfdz) / below,
                     torch.where(two_layers, ldepth / 2.0,
                                 torch.zeros_like(ldepth)))
    ldepth = torch.where(activenod > 0, ldepth, torch.zeros_like(ldepth))

    # :77-90 -- the area at the top of each solution layer. Node 0 is the
    # lake surface; node k below it sits dz*(activenod - k) above the bottom.
    rows = []
    for k in range(max_nodes + 1):
        depth_k = (ldepth if k == 0
                   else dz * (activenod - k).clamp(min=0).to(ldepth.dtype))
        area = get_sarea(z, basin, depth_k)
        rows.append(torch.where(activenod >= k, area,
                                torch.zeros_like(area)))
    surface = torch.stack(rows, 0)

    # :92-104 -- the stored volume includes the ice, which is water too.
    volume = get_volume(z, basin, ldepth, maxvolume) + ice_water_eq
    return surfdz, dz, activenod, surface, surface[0], volume


def lake_outflow(volume, ice_water_eq, evapw, ice_throughfall, inflow_volume,
                 snowmlt, z, basin, maxvolume, mindepth, minvolume, wfrac,
                 dt, dsmax_step, ds, ws, c_expt, max_moist_bot,
                 resid_moist_bot, ice_bot, new_ice_area, recharge=None):
    """The lake's own water balance and what leaves it.
    lakes.eb.c:1873-1897 and :2010-2118.  Returns
    (volume, ldepth, sarea, baseflow_out, runoff_out, evapw).

    Three things happen, in VIC's order. Evaporation is taken, and is not
    allowed to take more liquid water than there is (:1876-1891). Baseflow
    leaves as the ARNO curve of the *wetland's* bottom-layer moisture --
    the lake drains through the soil beside it, not through its own bed --
    over the lake's own area. Runoff leaves over a broad-crested weir round
    the lake's circumference, and only above `mindepth`.

    `recharge` is the water the lake gives to newly flooded wetland
    (:1953-2005). It is passed in rather than computed here because it
    needs the wetland tile's soil, snow and canopy storage; pass None and
    the lake is treated as not having changed area, which is the C's path
    when `max_newfraction <= lakefrac`.

    Every branch that clips a flux to the water available is reproduced,
    including the two that can make `runoff_out` negative when the lake is
    already below `minvolume` (:2078-2093) -- VIC subtracts the shortfall
    from the runoff rather than refusing to drain, and a caller that sums
    runoff without expecting that will see it.
    """
    volume, evapw = lake_inflow(volume, ice_water_eq, evapw,
                                ice_throughfall, inflow_volume, snowmlt)
    volume_save = volume

    if recharge is not None:
        # :1980-2005 -- the lake fills the newly flooded soil, unless it
        # runs out of liquid water doing so.
        enough = (volume - ice_water_eq) > recharge
        volume = torch.where(enough, volume - recharge, ice_water_eq)

    volume, ldepth, sarea, baseflow_out, runoff_out = lake_drain(
        volume, volume_save, ice_water_eq, z, basin, maxvolume, mindepth,
        minvolume, wfrac, dt, dsmax_step, ds, ws, c_expt, max_moist_bot,
        resid_moist_bot, ice_bot, new_ice_area)
    return volume, ldepth, sarea, baseflow_out, runoff_out, evapw


def lake_inflow(volume, ice_water_eq, evapw, ice_throughfall, inflow_volume,
                snowmlt):
    """What the lake gains and loses before anything drains out of it.
    lakes.eb.c:1873-1891.  Returns (volume, evapw).

    Split out of `lake_outflow` so that `water_balance` can put the
    wetland recharge between this and the draining, which is where the C
    puts it: the recharge needs the lake's new area, and that needs the
    volume this step leaves.
    """
    liquid = volume - ice_water_eq
    available = liquid + ice_throughfall + inflow_volume + snowmlt
    # :1876-1891 -- evaporation cannot exceed the liquid water there is.
    too_much = (evapw.abs() > 0.0) & (evapw > available)
    evapw = torch.where(too_much, available, evapw)
    volume = torch.where(too_much, ice_water_eq,
                         volume + (ice_throughfall + inflow_volume + snowmlt
                                   - evapw))
    return volume, evapw


def lake_drain(volume, volume_save, ice_water_eq, z, basin, maxvolume,
               mindepth, minvolume, wfrac, dt, dsmax_step, ds, ws, c_expt,
               max_moist_bot, resid_moist_bot, ice_bot, new_ice_area):
    """Baseflow and runoff out of the lake, and the caps on the result.
    lakes.eb.c:2010-2118.
    Returns (volume, ldepth, sarea, baseflow_out, runoff_out).
    """
    # :2016-2041 -- ARNO baseflow from the wetland's bottom layer.
    rel_moist = ((max_moist_bot - ice_bot) - resid_moist_bot) / (
        max_moist_bot - resid_moist_bot)
    base = dsmax_step * ds / ws * rel_moist
    wet = rel_moist > ws
    frac = torch.clamp((rel_moist - ws) / (1.0 - ws), min=0.0)
    base = base + torch.where(wet,
                              dsmax_step * (1.0 - ds / ws)
                              * torch.pow(frac, c_expt),
                              torch.zeros_like(base))
    baseflow_out_mm = torch.clamp(base, min=0.0)

    # :2046-2062 -- over the liquid lake's area, and no more than there is.
    ldepth = get_depth(z, basin, volume - ice_water_eq, maxvolume,
                       z[0].expand_as(volume))
    sarea = get_sarea(z, basin, ldepth)
    baseflow_out = baseflow_out_mm * sarea / 1000.0
    room = volume - ice_water_eq
    baseflow_out = torch.where(room >= baseflow_out, baseflow_out, room)
    volume = volume - baseflow_out

    # :2065-2093 -- a broad-crested weir round the lake's circumference.
    ldepth = get_depth(z, basin, volume - ice_water_eq, maxvolume,
                       z[0].expand_as(volume))
    # sqrt(0) is 0 but its derivative is infinite, and a lake drained to its
    # ice has sarea exactly 0 -- so the branch has to be selected, not
    # clamped, or the gradient of every column sharing the batch is NaN.
    inner = sarea / CONST_PI
    circum = torch.where(
        inner > 0.0,
        2.0 * CONST_PI * torch.sqrt(torch.where(inner > 0.0, inner,
                                                torch.ones_like(inner))),
        torch.zeros_like(inner))
    weir = wfrac * circum * dt * 1.6 * torch.pow(
        torch.clamp(ldepth - mindepth, min=0.0), 1.5)
    liquid = volume - ice_water_eq
    have = liquid >= weir
    runoff_out = torch.where(have, weir, liquid)
    # :2079-2081 and :2088-2090 -- and never below minvolume, even if that
    # means a negative runoff.
    below = (volume - runoff_out) < minvolume
    runoff_out = torch.where(below, volume - minvolume, runoff_out)
    shallow = ldepth <= mindepth
    runoff_out = torch.where(shallow, torch.zeros_like(runoff_out),
                             runoff_out)
    volume = torch.where(shallow, volume, volume - runoff_out)

    # :2096-2101 -- the lake may not end the step fuller than it started.
    grew = volume > volume_save
    runoff_out = torch.where(grew, runoff_out + volume - volume_save,
                             runoff_out)
    volume = torch.where(grew, volume_save, volume)

    # :2104-2118 -- nor fuller than the basin, nor negative.
    over = (volume - maxvolume) > 0.0
    icy = ice_water_eq > maxvolume
    spill = torch.where(icy, volume - ice_water_eq, volume - maxvolume)
    runoff_out = torch.where(over, runoff_out + spill, runoff_out)
    volume = torch.where(over, torch.where(icy, ice_water_eq, maxvolume),
                         volume)
    volume = torch.where((~over) & (volume < 0.0), torch.zeros_like(volume),
                         volume)

    ldepth = get_depth(z, basin, volume - ice_water_eq, maxvolume,
                       z[0].expand_as(volume))
    sarea = torch.maximum(get_sarea(z, basin, ldepth), new_ice_area)
    return volume, ldepth, sarea, baseflow_out, runoff_out


def advect_soil_veg_storage(lakefrac, max_newfraction, newfraction,
                            delta_moist, moist, max_moist, wdew, basin0,
                            baseflow):
    """Wetland soil moisture and canopy storage carried across a change in
    the lake's area.  lakes.eb.c:2313-2432.
    Returns (moist, wdew, baseflow), all over the wetland's *final* area.

    Three things happen, and the order is VIC's. The recharge the lake gave
    the newly flooded soil is added layer by layer, and what will not fit
    cascades -- downward first (:2333-2346), then back upward (:2348-2364),
    then out as baseflow if it still will not fit (:2367-2371). The moisture
    is then rescaled from the wetland's starting area to its final one, with
    the strip between the two added at saturation (:2374-2381): the C's
    comment says this works whether the lake grew or shrank, and it does,
    because the strip's sign follows `lakefrac - newfraction`.

    `moist` and `delta_moist` are [layer, ...] in millimetres over the
    wetland's *starting* area; `baseflow` is in cubic metres and is added to,
    as the C adds to `cell->baseflow`.

    A lake that covered the whole cell at the start of the step has no
    wetland to carry anything from, so the new wetland is created saturated
    and dry-canopied (:2404-2419).
    """
    n = moist.shape[0]
    had_wetland = lakefrac < 1.0

    dm = [delta_moist[i] for i in range(n)]
    new = [None] * n
    # :2333-2346 -- fill each layer in turn, spilling the excess downward.
    for i in range(n):
        v = moist[i] + dm[i]
        dm[i] = torch.zeros_like(v)
        over = torch.clamp(v - max_moist[i], min=0.0)
        if i < n - 1:
            dm[i + 1] = dm[i + 1] + over
        else:
            dm[i] = dm[i] + over
        new[i] = torch.minimum(v, max_moist[i])
    # :2348-2364 -- and back up, spilling what the bottom would not take.
    for i in range(n - 1, -1, -1):
        v = new[i] + dm[i]
        dm[i] = torch.zeros_like(v)
        over = torch.clamp(v - max_moist[i], min=0.0)
        if i > 0:
            dm[i - 1] = dm[i - 1] + over
        else:
            dm[i] = dm[i] + over
        new[i] = torch.minimum(v, max_moist[i])

    # :2367-2371 -- whatever the column will not hold leaves as baseflow.
    spill = torch.clamp(dm[0], min=0.0)
    baseflow = baseflow + torch.where(
        had_wetland, spill / 1000.0 * (1.0 - lakefrac) * basin0,
        torch.zeros_like(spill))

    # :2374-2381 -- rescale to the wetland's final area.
    denom = 1.0 - newfraction
    denom = torch.where(denom.abs() > 0.0, denom, torch.ones_like(denom))
    rescaled = []
    for i in range(n):
        v = new[i] * (1.0 - lakefrac)
        v = v + max_moist[i] * (lakefrac - newfraction)
        rescaled.append(v / denom)

    # :2404-2419 -- no wetland before this step: create it saturated.
    out = torch.stack([torch.where(had_wetland, rescaled[i],
                                   max_moist[i].expand_as(rescaled[i]))
                       for i in range(n)], 0)

    # :2392-2401 -- the canopy store follows the area it sat on, which is
    # the *pre-flood* area when the lake grew and the starting one when it
    # receded.
    receded = max_newfraction <= lakefrac
    scale = torch.where(receded, (1.0 - lakefrac) / denom,
                        (1.0 - max_newfraction) / denom)
    wdew = torch.where(had_wetland, wdew * scale, torch.zeros_like(wdew))
    return out, wdew, baseflow


def rescale_fluxes(oldfrac, newfrac, *fluxes):
    """Fluxes reported over one area, restated over another.
    lakes.eb.c:2441-2477, :2523-2537 and :2543-2640.

    Every flux VIC carries per unit area of a tile has to be rescaled when
    that tile's area changes, or the cell's total would change with it.
    A tile that did not exist at the start of the step has no flux to
    rescale and its fluxes are zeroed (:2465-2476).

    `newfrac` is floored at DBL_EPSILON as the C floors it, which is what
    keeps a tile that has just vanished from dividing by zero.
    """
    newfrac = torch.clamp(newfrac, min=DBL_EPSILON)
    existed = oldfrac > 0.0
    scale = oldfrac / newfrac
    return tuple(torch.where(existed, f * scale, torch.zeros_like(f))
                 for f in fluxes)


def advect_snow_storage(lakefrac, max_newfraction, newfraction, *stores):
    """Wetland snow storage carried across a change in the wetland's area.
    lakes.eb.c:2484-2517.

    Unlike `rescale_fluxes` this scales by the area the snow *was lying on*:
    the pre-flood area when the lake grew, the starting area when it
    receded (:2494-2509). Snow in the strip the lake took is gone with it.
    """
    newfraction = torch.where(1.0 - newfraction < DBL_EPSILON,
                              torch.full_like(newfraction, 1.0 - DBL_EPSILON),
                              newfraction)
    had_wetland = lakefrac < 1.0
    grew = max_newfraction > lakefrac
    num = torch.where(grew, 1.0 - max_newfraction, 1.0 - lakefrac)
    scale = num / (1.0 - newfraction)
    return tuple(torch.where(had_wetland, s * scale, torch.zeros_like(s))
                 for s in stores)


def rescale_snow_storage(oldfrac, newfrac, *stores):
    """Lake-ice snow storage restated over a new area.  lakes.eb.c:2523-2537.

    The lake's own snowpack, unlike the wetland's, is rescaled
    unconditionally -- there is no "did not exist" branch, because the C
    only calls this when the lake was there.
    """
    newfrac = torch.clamp(newfrac, min=DBL_EPSILON)
    scale = oldfrac / newfrac
    return tuple(s * scale for s in stores)


def advect_carbon_storage(lakefrac, newfraction, lake_pool, cell_pool):
    """A soil carbon pool carried across a change in the lake's area.
    lakes.eb.c:2647-2683.  Returns (lake_pool, cell_pool).

    Whichever tile grew takes a mass-weighted share of the other's carbon.
    The two branches are not symmetric in the C and are not made so here:
    when the lake grows only the lake's pool is rewritten, and when it
    shrinks only the wetland's is -- so the tile that lost area keeps its
    concentration and the one that gained it is diluted.
    """
    grew = newfraction > lakefrac
    shrank = newfraction < lakefrac
    nf_grow = torch.clamp(newfraction, min=DBL_EPSILON)
    nf_shrink = torch.where(1.0 - newfraction < DBL_EPSILON,
                            torch.full_like(newfraction, 1.0 - DBL_EPSILON),
                            newfraction)
    lake_new = (lakefrac * lake_pool
                + (newfraction - lakefrac) * cell_pool) / nf_grow
    cell_new = ((lakefrac - newfraction) * lake_pool
                + (1.0 - lakefrac) * cell_pool) / (1.0 - nf_shrink)
    return (torch.where(grew, lake_new, lake_pool),
            torch.where(shrank, cell_new, cell_pool))


#: initialize_parameters.c:55, the lake surface's longwave emissivity.
EMISS_H2O = 0.98
#: vic_physical_constants.h:79 and :115, and physics.c:49-56. VIC 5's
#: Stefan-Boltzmann is 5.67e-8 where the lake's own longwave in VIC 4 used
#: 5.6696e-8; these two functions are VIC 5's `calc_outgoing_longwave` and
#: `calc_latent_heat_of_vaporization`, which the lake calls, so they take
#: VIC 5's values.
CONST_STEBOL = 5.67e-8
CONST_LATVAP = 2.501e6
#: initialize_parameters.c:97, the cap on the lake's two fixed-point loops.
LAKE_MAX_ITER_N = 50
#: The convergence tolerance both loops use (water_energy_balance.c:78,
#: water_under_ice.c:60). It is an absolute tolerance on a temperature in
#: one and on a heat flux in the other, which is the C's, not a slip.
LAKE_EPSILON = 0.0001


def water_energy_balance(surface, evap_out, dt, dz, surfdz, lat, t_cutoff,
                         tair, wind, pressure, vp, air_density, longwave,
                         shortwave, wind_h, t, water_density, fracprv, cp,
                         lvolume, numnod):
    """The open lake's surface energy balance.  water_energy_balance.c:33-205.
    Returns a dict with T, Qh, Qle, LWnet, deltaH, evapw, energy_ice_formation,
    new_ice_area, new_ice_height, new_ice_water_eq and energy_out_bottom.

    A fixed point on the surface temperature: guess it, compute the turbulent
    and radiative fluxes it implies, solve the column, take the mean of the
    old and new surface temperatures, and repeat. The C iterates at most
    LAKE_MAX_ITER times and, if it has not converged, throws the whole
    iteration away and recomputes the fluxes at the *starting* temperature
    with deltaH zero (:176-203). Both outcomes are reproduced, per column,
    because in a batch some columns converge and others do not.

    Ice forms inside the loop whenever the new surface temperature falls
    below freezing (:148-160), and the heat that formed it is what the
    column's energy change is measured against.
    """
    ts = t[0]
    tnew = t
    jouleold = energycalc(t, numnod, dz, surfdz, surface, cp, water_density)
    zero = torch.zeros_like(ts)
    out = dict(Qh=zero.clone(), Qle=zero.clone(), LWnet=zero.clone(),
               deltaH=zero.clone(), evapw=zero.clone(),
               energy_ice_formation=zero.clone(), new_ice_area=zero.clone(),
               new_ice_height=zero.clone(), new_ice_water_eq=zero.clone(),
               energy_out_bottom=zero.clone())
    converged = torch.zeros_like(ts, dtype=torch.bool)
    t_final = t.clone()

    def fluxes(ts_k):
        tskin = ts_k + CONST_TKFRZ
        tcutk = t_cutoff + CONST_TKFRZ
        evapw, qh = latsens(tskin, tcutk, torch.zeros_like(ts_k), tair, wind,
                            pressure, vp, air_density, wind_h)
        # physics.c:49-56 -- note 2361, not 2.361e3 scaled by anything.
        le = CONST_LATVAP - 2361.0 * tair
        qle = -evapw * le
        # physics.c:62-70.
        lwnet = longwave - EMISS_H2O * CONST_STEBOL * tskin ** 4
        return evapw, qh, qle, lwnet

    tmean = torch.full_like(ts, 1.0e30)
    for it in range(LAKE_MAX_ITER_N):
        ts_k = t[0] if it == 0 else tmean
        evapw, qh, qle, lwnet = fluxes(ts_k)
        de = eddy(torch.ones_like(fracprv, dtype=torch.long), wind,
                  water_density, lat, numnod, dz, surfdz)
        tnew, eob, joulenew = temp_area(
            shortwave * LAKE_A1, shortwave * LAKE_A2, qle + qh + lwnet, t,
            water_density, de, dt, surface, numnod, dz, surfdz, cp)

        freezing = tnew[0] < t_cutoff
        tfrozen, qfus, aadd, nih, niwe = iceform(
            tnew, t_cutoff, fracprv, numnod, dt, dz, surfdz, cp, surface,
            lvolume)
        sumjouli = energycalc(tfrozen, numnod, dz, surfdz, surface, cp,
                              water_density)
        dh = torch.where(freezing, (sumjouli - jouleold) / (surface[0] * dt),
                         (joulenew - jouleold) / (surface[0] * dt))
        tnew = torch.where(freezing.unsqueeze(0), tfrozen, tnew)
        # :148-160 -- iceform is called only when the surface went below
        # freezing, so on a step where it did not, the C leaves these four
        # at whatever the caller set. solve_lake:102 zeroes them each step,
        # so zero is what "untouched" means here.
        zmask = lambda v: torch.where(freezing, v, torch.zeros_like(v))
        qfus, aadd, nih, niwe = (zmask(qfus), zmask(aadd), zmask(nih),
                                 zmask(niwe))

        tmean_new = (tnew[0] + t[0]) / 2.0
        # :92 -- the test is on the *previous* pair, so a column converges on
        # the iteration after the one that settled it.
        settled = (tmean_new - ts_k).abs() <= LAKE_EPSILON
        first = ~converged & settled
        if bool(first.any()):
            t_final = torch.where(first.unsqueeze(0), tnew, t_final)
            for k, v in (('Qh', qh), ('Qle', qle), ('LWnet', lwnet),
                         ('deltaH', dh), ('evapw', evapw),
                         ('energy_ice_formation', qfus),
                         ('new_ice_area', aadd), ('new_ice_height', nih),
                         ('new_ice_water_eq', niwe),
                         ('energy_out_bottom', eob)):
                out[k] = torch.where(first, v, out[k])
            converged = converged | settled
        if bool(converged.all()):
            break
        tmean = tmean_new

    # :176-203 -- a column that never converged keeps its starting profile
    # and is re-evaluated there, with no energy change at all.
    if not bool(converged.all()):
        evapw, qh, qle, lwnet = fluxes(t[0])
        tfrozen, qfus, aadd, nih, niwe = iceform(
            t, t_cutoff, fracprv, numnod, dt, dz, surfdz, cp, surface,
            lvolume)
        freezing = t[0] < t_cutoff
        zmask = lambda v: torch.where(freezing, v, torch.zeros_like(v))
        qfus, aadd, nih, niwe = (zmask(qfus), zmask(aadd), zmask(nih),
                                 zmask(niwe))
        bad = ~converged
        t_final = torch.where(bad.unsqueeze(0),
                              torch.where(freezing.unsqueeze(0), tfrozen, t),
                              t_final)
        for k, v in (('Qh', qh), ('Qle', qle), ('LWnet', lwnet),
                     ('deltaH', torch.zeros_like(qh)), ('evapw', evapw),
                     ('energy_ice_formation', qfus), ('new_ice_area', aadd),
                     ('new_ice_height', nih), ('new_ice_water_eq', niwe)):
            out[k] = torch.where(bad, v, out[k])
    out['T'] = t_final
    out['converged'] = converged
    return out


def water_under_ice(freezeflag, sw_ice, wind, t_ice, water_density, lat,
                    numnod, dz, surfdz, t_cutoff, surface, water_cp,
                    mixdepth, hice, sdepth, dt):
    """The water column beneath the ice.  water_under_ice.c:31-142.
    Returns a dict with Ti, qw, deltaH, mixdepth and energy_out_bottom.

    A fixed point on the heat flux the water gives up to the ice, which the
    C estimates from the top node's temperature gradient over half the
    surface layer (:73). Shortwave that got through the snow and ice drives
    the column, the column is mixed, and the flux is re-estimated.

    The non-convergence branch (:130-139) is not a fallback but a different
    answer: the whole column is set to the freezing point and the flux to
    zero. Reproduced per column.

    `mixdepth` comes back exactly as it went in. The C takes it by value
    (:45, `int mixdepth`) and passes its address to `tracer_mixer`, so the
    mixing depth the convective adjustment finds is written into a local
    copy and discarded when the function returns. It is returned here so
    the caller sees what VIC's caller sees, which is its own value back.
    """
    tnew = t_ice
    de = eddy(freezeflag, wind, water_density, lat, numnod, dz, surfdz)
    qw_init = 0.57 * (t_ice[0] - t_cutoff) / (surfdz / 2.0)
    jouleold = energycalc(t_ice, numnod, dz, surfdz, surface, water_cp,
                          water_density)

    qw = qw_init
    qw_mean = torch.full_like(qw_init, 1.0e30)
    converged = torch.zeros_like(qw_init, dtype=torch.bool)
    t_final = t_ice.clone()
    out = dict(qw=torch.zeros_like(qw_init),
               deltaH=torch.zeros_like(qw_init),
               energy_out_bottom=torch.zeros_like(qw_init))
    md = mixdepth.clone()

    for it in range(LAKE_MAX_ITER_N):
        qw_k = qw_init if it == 0 else qw_mean
        sw_vis = LAKE_A1 * sw_ice * torch.exp(
            -(LAKE_LAMISW * hice + LAKE_LAMSSW * sdepth))
        sw_nir = LAKE_A2 * sw_ice * torch.exp(
            -(LAKE_LAMILW * hice + LAKE_LAMSLW * sdepth))
        tnew, eob, joulenew = temp_area(sw_vis, sw_nir, -qw_k, t_ice,
                                        water_density, de, dt, surface,
                                        numnod, dz, surfdz, water_cp)
        dh = (joulenew - jouleold) / (surface[0] * dt)
        tnew, _rho, _md_discarded = tracer_mixer(tnew, md, surface, numnod,
                                                 dz, surfdz, water_cp)
        qw_final = 0.57 * (tnew[0] - t_cutoff) / (surfdz / 2.0)
        qw_mean_new = (qw_final + qw_k) / 2.0

        settled = (qw_mean_new - qw_k).abs() <= LAKE_EPSILON
        first = ~converged & settled
        if bool(first.any()):
            t_final = torch.where(first.unsqueeze(0), tnew, t_final)
            out['qw'] = torch.where(first, qw_mean_new, out['qw'])
            out['deltaH'] = torch.where(first, dh, out['deltaH'])
            out['energy_out_bottom'] = torch.where(first, eob,
                                                   out['energy_out_bottom'])
            converged = converged | settled
        if bool(converged.all()):
            break
        qw_mean = qw_mean_new

    if not bool(converged.all()):
        bad = ~converged
        frozen = t_cutoff.expand_as(t_ice)
        t_final = torch.where(bad.unsqueeze(0), frozen, t_final)
        joulenew = energycalc(frozen, numnod, dz, surfdz, surface, water_cp,
                              water_density)
        out['qw'] = torch.where(bad, torch.zeros_like(out['qw']), out['qw'])
        out['deltaH'] = torch.where(bad,
                                    (joulenew - jouleold) / (surface[0] * dt),
                                    out['deltaH'])
    out['Ti'] = t_final
    out['mixdepth'] = md
    out['converged'] = converged
    return out


#: initialize_parameters.c:54 and vic_physical_constants.h -- the snow
#: surface's emissivity and the heat capacity of fresh water.
EMISS_SNOW = 0.97
CONST_CPFW = 4.188e3
#: initialize_parameters.c:45, the resistance of a decoupled surface.
HUGE_RESIST = 1.0e20


def ice_energy_balance(t_surf, st):
    """The surface energy balance over lake ice.  IceEnergyBalance.c:34-214.
    Returns (residual, terms).

    This is the snow pack's surface balance with two differences, and both
    are the lake's:

    * There is no cold-content term. The C says so at :185-186 -- the ice
      has no pack to cool -- so the `deltaCC` that appears in
      `snow_pack_energy_balance` is simply absent here rather than zero.
    * There is a conduction term through the ice and its snow cover,
      `qnull = (1/AvgCond) * (Tfreeze - TMean + SWconducted)` (:188), which
      is what couples the surface to the water beneath. `AvgCond` is
      `icerad`'s series resistance and `SWconducted` the shortwave that
      reached the ice-water interface.

    `st` carries the same keys `snow.snow_pack_energy_balance` takes, plus
    `avg_cond`, `t_freeze` and `sw_conducted`. The pieces underneath --
    the stability correction, the sensible heat, the sublimation split --
    are the snow module's, because they are the same functions.
    """
    from . import snow as _snow

    tmean = t_surf
    density = CONST_RHOFW

    # :145-150 -- a windless step decouples the surface entirely.
    ra_used = torch.where(
        st['wind'] > 0.0,
        st['ra'] / _snow.stability_correction(st['z'], torch.zeros_like(
            st['z']), tmean, st['t_air'], st['wind'], st['z0']),
        torch.full_like(st['ra'], HUGE_RESIST))

    long_out = st['long_rad_in'] - _snow.calc_outgoing_longwave(
        tmean + CONST_TKFRZ, EMISS_SNOW)
    net_rad = st['short_rad'] + long_out
    sensible = _snow.calc_sensible_heat(st['air_dens'], st['t_air'], tmean,
                                        ra_used)

    # :169-176 -- the blowing flux was fixed before the root find, so only
    # the surface term moves with the temperature.
    blowing_mass = st.get('blowing_flux', torch.zeros_like(tmean)) * \
        density / st['dt']
    latent, latent_sub, vapor_mass, surface_mass = \
        _snow.latent_heat_from_snow(st['air_dens'], st['eact_air'], st['lv'],
                                    st['press'], st['ra'], tmean, st['vpd'],
                                    blowing_mass)
    vapor_flux = vapor_mass * st['dt'] / density
    surface_flux = surface_mass * st['dt'] / density

    # :180-181 -- rain arrives at the air temperature, not the surface's.
    advected = (CONST_CPFW * CONST_RHOFW * st['t_air'] * st['rain']) / st['dt']

    # :188 -- conduction through the ice and its snow.
    qnull = (1.0 / st['avg_cond']) * (st['t_freeze'] - tmean
                                      + st['sw_conducted'])

    rest = net_rad + sensible + latent + latent_sub + advected + qnull
    refreeze_energy = (st['surface_liquid_water'] * CONST_LATICE
                       * density) / st['dt']

    # :201-210 -- at exactly 0 C with energy to spare, the surplus melts
    # rather than warming a surface that cannot get warmer.
    melting = (t_surf == 0.0) & (rest > -refreeze_energy)
    refreeze_energy = torch.where(melting, -rest, refreeze_energy)
    rest = torch.where(melting, torch.zeros_like(rest),
                       rest + refreeze_energy)

    return rest, dict(ra_used=ra_used, long_out=long_out, net_rad=net_rad,
                      sensible=sensible, latent=latent,
                      latent_sub=latent_sub, vapor_flux=vapor_flux,
                      surface_flux=surface_flux, advected=advected,
                      qf=qnull, refreeze_energy=refreeze_energy)


#: initialize_parameters.c:150-201 and vic_physical_constants.h:111-112.
SNOW_MAX_SURFACE_SWE = 0.125        #: m, the thickest active surface layer
SNOW_LIQUID_WATER_CAPACITY = 0.035  #: fraction of the ice it can hold
SNOW_MIN_SWQ_EB_THRES = 0.0010      #: m, below which the surface is not solved
SNOW_DT = 5.0                       #: C, the bracket the root find opens with
ROOT_BRENT_TSTEP = 10.0             #: C, how far root_brent widens it each try
ROOT_BRENT_MAXTRIES = 5             #: and how many times
CONST_CPICE = 2.11727e3             #: J/kg/K, vic_physical_constants.h:110
CONST_VCPICE_WQ = CONST_CPICE * CONST_RHOFW


MM_PER_M = 1000.0


def _ice_eb_residual(t_surf, st):
    """`ice_energy_balance`'s residual alone, for the root solver."""
    return ice_energy_balance(t_surf, st)[0]


def _widen_bracket(fn, lo, hi, st, tstep=ROOT_BRENT_TSTEP,
                   maxtries=ROOT_BRENT_MAXTRIES):
    """root_brent.c:168-179's bracket search for a residual taking `st`.

    The search itself is `torchvic.core.implicit.widen_bracket`, shared with the
    soil thermal profile, which opens its own brackets the same way and
    needs the same widening.
    """
    from . import implicit
    return implicit.widen_bracket(fn, lo, hi, args=(st,), tstep=tstep,
                                  maxtries=maxtries)


def ice_melt(state, f):
    """One step of the snow-and-ice pack on a lake.  ice_melt.c:34-611.
    Returns (new_state, diagnostics).

    This is `torchvic.core.snow.snow_step`'s job on a lake: the same two-layer
    bookkeeping -- a surface layer capped at SNOW_MAX_SURFACE_SWE over a
    pack layer, cold contents, refreezing, melting, a liquid-water capacity
    -- but the ice beneath the snow is the LAKE's, and the three stores
    have to be kept apart:

        SnowIce   snow that fell on the ice, m water equivalent
        LakeIce   the lake's own ice, `ice_water_eq / areai`
        Ice       the two together, which is what the layers hold

    Melt takes them bottom-up (:283-322): the pack layer first, then the
    surface layer's snow, and only when the snow is gone does it eat into
    the lake's ice. That last part is the only melt that leaves the lake,
    which is why `IceMelt` is tracked apart from `SnowMelt` and is the
    term that moves `lake->volume` at :576.

    The surface temperature follows the C's two branches. The balance is
    evaluated at 0 C first; `IceEnergyBalance` returns exactly zero when
    there is energy to spare, so `Qnet == 0.0` (:239) is the melting test,
    not a positive residual. Only when it is non-zero is there a
    sub-freezing root, and that root comes from `implicit.solve` on the
    C's own bracket `[surf_temp - SNOW_DT, surf_temp + SNOW_DT]` rather
    than from unrolling `root_brent`.

    `state` keys: swq, surf_water, pack_water, surf_temp, pack_temp,
    ice_water_eq, areai, hice, volume.
    `f` keys: dt, rainfall and snowfall in MILLIMETRES (the C divides by
    MM_PER_M at :102-103, and solve_lake hands it mm), air_temp, wind,
    net_short, long_rad_in, air_dens, lv, press and vpd and eact_air in
    PASCALS (the C's callers pass kPa and it multiplies at :233-235),
    t_cutoff, ra, z, z0.

    Not implemented, and not approximated: BLOWING (`CalcBlowingSnow`),
    whose flux the C folds into the vapour split at :196-215; it is held
    at zero, which is what VIC leaves `snow->blowing_flux` at with the
    option off. TFALLBACK and the -998 error path are not taken: a column
    whose residual does not change sign inside the C's bracket comes back
    in `diagnostics['bracket_failed']` instead, because a bracketed solver
    returns a bound there and VIC returns ERROR, and the caller has to
    know which happened rather than be handed a number.
    """
    from . import implicit

    dt = f['dt']
    snowfall = f['snowfall'] / MM_PER_M
    rainfall = f['rainfall'] / MM_PER_M
    zero = torch.zeros_like(state['swq'])
    one = torch.ones_like(state['swq'])

    initial_swq = state['swq']
    snow_ice = state['swq'] - state['pack_water'] - state['surf_water']
    lake_ice = state['ice_water_eq'] / state['areai']
    initial_ice = lake_ice
    ice = snow_ice + lake_ice

    # :121-135 -- rebuild the two layers. The lake's ice sits in the pack
    # unless the surface layer is thicker than the snow on top of it, in
    # which case the surface layer reaches down into the ice.
    surface_swq = torch.clamp(ice, max=SNOW_MAX_SURFACE_SWE)
    in_snow = surface_swq <= snow_ice
    pack_swq = torch.where(in_snow, snow_ice - surface_swq, zero)
    pack_ice = torch.where(in_snow, lake_ice, ice - surface_swq)

    surface_cc = CONST_VCPICE_WQ * surface_swq * state['surf_temp']
    pack_cc = CONST_VCPICE_WQ * (pack_swq + pack_ice) * state['pack_temp']
    snowfall_cc = torch.where(f['air_temp'] > 0.0, zero,
                              CONST_VCPICE_WQ * snowfall * f['air_temp'])

    # :149-173 -- fresh snow fills the surface layer; the surplus and its
    # cold content are handed down. Note :153-155: the numerator is
    # (SnowFall - MAX), not (SurfaceSwq + SnowFall - MAX), so the fraction
    # of the new snow's cold content that goes down is not the fraction of
    # the new snow that goes down. That is VIC's, and it is reproduced.
    spill = snowfall > (SNOW_MAX_SURFACE_SWE - surface_swq)
    d_pack_swq = surface_swq + snowfall - SNOW_MAX_SURFACE_SWE
    deep = d_pack_swq > surface_swq
    sf_safe = torch.where(snowfall > 0.0, snowfall, one)
    ss_safe = torch.where(surface_swq > 0.0, surface_swq, one)
    d_pack_cc = torch.where(
        deep,
        surface_cc + (snowfall - SNOW_MAX_SURFACE_SWE) / sf_safe * snowfall_cc,
        d_pack_swq / ss_safe * surface_cc)
    d_pack_cc = torch.where(spill, d_pack_cc, zero)
    d_pack_swq = torch.where(spill, d_pack_swq, zero)
    surface_swq = torch.where(
        spill, torch.full_like(surface_swq, SNOW_MAX_SURFACE_SWE),
        surface_swq + snowfall)
    surface_cc = surface_cc + snowfall_cc - d_pack_cc
    pack_swq = pack_swq + d_pack_swq
    pack_cc = pack_cc + d_pack_cc

    # :169-180 -- and the layer temperatures follow the cold contents.
    # There is no clamp to 0 C here; solve_snow's accumulation has one and
    # this does not.
    ss_safe = torch.where(surface_swq > 0.0, surface_swq, one)
    surf_temp = torch.where(surface_swq > 0.0,
                            surface_cc / (CONST_VCPICE_WQ * ss_safe), zero)
    packtot = pack_swq + pack_ice
    pt_safe = torch.where(packtot > 0.0, packtot, one)
    pack_temp = torch.where(packtot > 0.0,
                            pack_cc / (CONST_VCPICE_WQ * pt_safe), zero)

    snow_ice = snow_ice + snowfall
    ice = ice + snowfall
    surf_water = state['surf_water'] + rainfall
    pack_water = state['pack_water']

    # :185-187 -- the snow depth icerad wants is the water equivalent
    # converted at the LAKE's snow density, not the snow module's.
    avg_cond, sw_conducted, delta_cc = icerad(
        f['net_short'], state['hice'],
        snow_ice * CONST_RHOFW / LAKE_RHOSNOW)

    blowing_flux = zero
    st = dict(dt=dt, ra=f['ra'], z=f['z'], z0=f['z0'], wind=f['wind'],
              short_rad=f['net_short'], long_rad_in=f['long_rad_in'],
              air_dens=f['air_dens'], lv=f['lv'], t_air=f['air_temp'],
              press=f['press'], vpd=f['vpd'], eact_air=f['eact_air'],
              rain=rainfall, surface_liquid_water=surf_water,
              blowing_flux=blowing_flux, t_freeze=f['t_cutoff'],
              avg_cond=avg_cond, sw_conducted=sw_conducted)

    qnet0, terms0 = ice_energy_balance(zero, st)
    melting = qnet0 == 0.0

    # ----- :239-332, the melting branch --------------------------------
    re0 = terms0['refreeze_energy']
    freezing = re0 >= 0.0
    refrozen = re0 / (CONST_LATICE * CONST_RHOFW) * dt
    capped = refrozen > surf_water
    refrozen_a = torch.where(capped, surf_water, refrozen)
    re_a = torch.where(capped,
                       refrozen_a * CONST_LATICE * CONST_RHOFW / dt, re0)
    refrozen_a = torch.where(freezing, refrozen_a, zero)
    re_a = torch.where(freezing, re_a, re0)
    snowmelt = torch.where(freezing, zero,
                           re0.abs() / (CONST_LATICE * CONST_RHOFW) * dt)

    a_surface_swq = surface_swq + refrozen_a
    a_snow_ice = snow_ice + refrozen_a
    a_ice = ice + refrozen_a
    a_surf_water = torch.clamp(surf_water - refrozen_a, min=0.0)

    # :275-286 -- when the vapour flux is larger than the liquid there is,
    # all three flux components are rescaled, not just the total.
    vf0 = terms0['vapor_flux']
    vf0_safe = torch.where(vf0 != 0.0, vf0, one)
    a_exhaust = a_surf_water < -vf0
    a_blowing = torch.where(a_exhaust, blowing_flux * (-a_surf_water)
                            / vf0_safe, blowing_flux)
    a_vapor = torch.where(a_exhaust, -a_surf_water, vf0)
    a_surface_flux = torch.where(a_exhaust, -a_surf_water - a_blowing,
                                 terms0['surface_flux'])
    a_surf_water = torch.where(a_exhaust, zero, a_surf_water + vf0)

    # :288-322 -- melt comes off the bottom layer first.
    incomplete = snowmelt < a_ice
    c1 = snowmelt <= pack_swq                       # pack layer only
    c2 = (~c1) & (snowmelt <= a_snow_ice)           # pack plus surface snow
    c3 = (~c1) & (~c2)                              # and into the lake ice
    keep = a_surface_swq > snowmelt

    a_surf_water = torch.where(
        incomplete,
        torch.where(c1, a_surf_water + snowmelt,
                    torch.where(c2, a_surf_water + snowmelt + pack_water,
                                a_surf_water + a_snow_ice + pack_water)),
        a_surf_water + a_snow_ice + pack_water)
    a_pack_water = torch.where(incomplete & c1, pack_water, zero)
    a_pack_swq = torch.where(incomplete & c1, pack_swq - snowmelt, zero)
    # :313-319 -- PackSwq and SurfaceSwq are already zero on the line that
    # reads them back, so the subtraction is PackIce -= SnowMelt.
    a_pack_ice = torch.where(incomplete,
                             torch.where(c3 & (~keep), pack_ice - snowmelt,
                                         pack_ice), zero)
    a_surface_swq = torch.where(
        incomplete,
        torch.where(c1, a_surface_swq,
                    torch.where(c2, a_surface_swq - (snowmelt - pack_swq),
                                torch.where(keep, a_surface_swq - snowmelt,
                                            zero))),
        zero)
    a_snow_ice2 = torch.where(incomplete & (c1 | c2), a_snow_ice - snowmelt,
                              zero)
    a_lake_ice = torch.where(incomplete,
                             torch.where(c3, lake_ice - (snowmelt - a_snow_ice),
                                         lake_ice), zero)
    a_ice_melt = torch.where(incomplete,
                             torch.where(c3, snowmelt - a_snow_ice, zero),
                             lake_ice)
    a_snowmelt = torch.where(incomplete, snowmelt, a_ice)
    a_ice2 = torch.where(incomplete, a_ice - snowmelt, zero)
    a_surf_temp = zero
    a_pack_temp = torch.where(incomplete, pack_temp, zero)
    # :339-341 -- on a complete melt the energy is rescaled to the melt
    # that was actually available, keeping its sign.
    re_abs = torch.where(re_a != 0.0, re_a.abs(), one)
    a_re = torch.where(incomplete, re_a,
                       re_a / re_abs * a_snowmelt * CONST_LATICE
                       * CONST_RHOFW / dt)

    # ----- :345-464, the sub-freezing branch ---------------------------
    # root_brent.c:168-179 does NOT require its caller's bracket to hold the
    # root: when f(a) and f(b) share a sign it widens the interval by
    # ROOT_BRENT_TSTEP on BOTH sides and tries again, up to
    # ROOT_BRENT_MAXTRIES times. ice_melt.c:348-349 opens with
    # [surf_temp - 5, surf_temp + 5], which is often nowhere near the root
    # -- a surface at 0 C over a cold night solves to -11.6 C -- so a
    # solver handed that interval as given returns a bound, not a root, and
    # every energy term downstream of it is wrong by hundreds of W/m2.
    lo, hi, bracket_failed = _widen_bracket(_ice_eb_residual,
                                            surf_temp - SNOW_DT,
                                            surf_temp + SNOW_DT, st)
    bracket_failed = (~melting) & bracket_failed
    t_solved = implicit.solve(_ice_eb_residual, lo, hi, args=(st,),
                              tol=implicit.ROOT_BRENT_T)
    # :347 -- a surface layer this thin is not solved here at all; the C
    # flags it with 999 so calc_surf_energy_bal solves it with the ground.
    thin = surface_swq <= SNOW_MIN_SWQ_EB_THRES
    qnet_b, terms_b = ice_energy_balance(t_solved, st)

    # :425-433 -- the surface is below freezing, so every drop of liquid
    # in the surface layer has frozen.
    b_snow_ice = snow_ice + surf_water
    b_ice = ice + surf_water
    b_surf_water = zero
    vfb = terms_b['vapor_flux']
    vfb_safe = torch.where(vfb != 0.0, vfb, one)

    # :435-461 -- and the vapour exchange then comes off the ice.
    b_exhaust = surface_swq < -vfb
    b_into_lake = surface_swq > b_snow_ice
    b_blowing = torch.where(b_exhaust,
                            blowing_flux * (-surface_swq) / vfb_safe,
                            blowing_flux)
    b_vapor = torch.where(b_exhaust, -surface_swq, vfb)
    b_surface_flux = torch.where(b_exhaust, -surface_swq - b_blowing,
                                 terms_b['surface_flux'])
    b_lake_ice = torch.where(
        b_exhaust,
        torch.where(b_into_lake, lake_ice - (surface_swq - b_snow_ice),
                    lake_ice),
        torch.where(b_snow_ice > -vfb, lake_ice, lake_ice + (vfb + b_snow_ice)))
    b_snow_ice2 = torch.where(
        b_exhaust,
        torch.where(b_into_lake, zero, b_snow_ice),
        torch.where(b_snow_ice > -vfb, b_snow_ice + vfb, zero))
    b_surface_swq = torch.where(
        b_exhaust,
        torch.where(b_into_lake, surface_swq, zero),
        surface_swq + vfb)
    b_ice2 = torch.where(
        b_exhaust,
        torch.where(b_into_lake, pack_ice, pack_swq + pack_ice),
        b_ice + vfb)

    # A thin surface layer takes none of that: the C skips the whole block
    # and leaves surf_temp at 999.
    b_surf_temp = torch.where(thin, torch.full_like(surf_temp, 999.0),
                              t_solved)
    b_snow_ice2 = torch.where(thin, snow_ice, b_snow_ice2)
    b_surface_swq = torch.where(thin, surface_swq, b_surface_swq)
    b_ice2 = torch.where(thin, ice, b_ice2)
    b_surf_water2 = torch.where(thin, surf_water, b_surf_water)
    b_lake_ice = torch.where(thin, lake_ice, b_lake_ice)
    b_vapor = torch.where(thin, vf0, b_vapor)
    b_surface_flux = torch.where(thin, terms0['surface_flux'], b_surface_flux)
    b_blowing = torch.where(thin, blowing_flux, b_blowing)
    b_re = torch.where(thin, re0, terms_b['refreeze_energy'])

    # ----- select the branch -------------------------------------------
    def pick(a, b):
        return torch.where(melting, a, b)

    surface_swq = pick(a_surface_swq, b_surface_swq)
    pack_swq = pick(a_pack_swq, pack_swq)
    pack_ice = pick(a_pack_ice, pack_ice)
    snow_ice = pick(a_snow_ice2, b_snow_ice2)
    ice = pick(a_ice2, b_ice2)
    lake_ice = pick(a_lake_ice, b_lake_ice)
    surf_water = pick(a_surf_water, b_surf_water2)
    pack_water = pick(a_pack_water, pack_water)
    surf_temp = pick(a_surf_temp, b_surf_temp)
    pack_temp = pick(a_pack_temp, pack_temp)
    ice_melt_out = pick(a_ice_melt, zero)
    snowmelt = pick(a_snowmelt, zero)
    vapor_flux = pick(a_vapor, b_vapor)
    surface_flux = pick(a_surface_flux, b_surface_flux)
    blowing_flux = pick(a_blowing, b_blowing)
    refreeze_energy = pick(a_re, b_re)
    qnet = pick(qnet0, torch.where(thin, qnet0, qnet_b))
    use_b = (~melting) & (~thin)
    terms = {k: torch.where(use_b, terms_b[k], terms0[k]) for k in terms0}

    # ----- :467-479, the surface layer's liquid capacity ---------------
    max_liquid = SNOW_LIQUID_WATER_CAPACITY * torch.minimum(surface_swq,
                                                            snow_ice)
    spilled = surf_water > max_liquid
    melt = torch.where(spilled, surf_water - max_liquid, zero)
    surf_water = torch.where(spilled, max_liquid, surf_water)

    # ----- :489-525, refreeze what reached the pack --------------------
    pack_water = pack_water + melt
    pack_refreeze = pack_water * CONST_LATICE * CONST_RHOFW
    all_freeze = pack_cc < -pack_refreeze
    packtot = pack_swq + pack_water + pack_ice
    pt_safe = torch.where(packtot > 0.0, packtot, one)
    pack_cc_full = ((pack_swq + pack_water + pack_ice) * CONST_VCPICE_WQ
                    * pack_temp + pack_refreeze)
    pack_temp_full = torch.clamp(
        torch.where(packtot > 0.0,
                    pack_cc_full / (CONST_VCPICE_WQ * pt_safe), zero), max=0.0)
    d_pack = -pack_cc / (CONST_LATICE * CONST_RHOFW)
    pack_swq = torch.where(all_freeze, pack_swq + pack_water,
                           pack_swq + d_pack)
    ice = torch.where(all_freeze, ice + pack_water, ice + d_pack)
    snow_ice = torch.where(all_freeze, snow_ice + pack_water,
                           snow_ice + d_pack)
    pack_temp = torch.where(all_freeze, pack_temp_full, zero)
    pack_water = torch.where(all_freeze, zero, pack_water - d_pack)

    # ----- :528-535, the pack layer's liquid capacity ------------------
    max_liquid = SNOW_LIQUID_WATER_CAPACITY * pack_swq
    spilled = pack_water > max_liquid
    melt = torch.where(spilled, pack_water - max_liquid, zero)
    pack_water = torch.where(spilled, max_liquid, pack_water)

    # ----- :538-570, rebalance the two layers --------------------------
    ice = pack_ice + pack_swq + surface_swq
    big = ice > SNOW_MAX_SURFACE_SWE
    cc_s = CONST_VCPICE_WQ * surf_temp * surface_swq
    cc_p = CONST_VCPICE_WQ * pack_temp * (pack_swq + pack_ice)
    ss_safe = torch.where(surface_swq > 0.0, surface_swq, one)
    ptot = pack_swq + pack_ice
    pt_safe = torch.where(ptot > 0.0, ptot, one)
    over = surface_swq > SNOW_MAX_SURFACE_SWE
    under = surface_swq < SNOW_MAX_SURFACE_SWE
    share_o = cc_s * (surface_swq - SNOW_MAX_SURFACE_SWE) / ss_safe
    # :558-565 -- PackCC is updated first and SurfaceCC then takes its
    # share of the NEW PackCC, not the old one.
    cc_p_u = cc_p - cc_p * (SNOW_MAX_SURFACE_SWE - surface_swq) / pt_safe
    share_u = cc_p_u * (SNOW_MAX_SURFACE_SWE - surface_swq) / pt_safe
    cc_p_new = torch.where(over, cc_p + share_o,
                           torch.where(under, cc_p_u, cc_p))
    cc_s_new = torch.where(over, cc_s - share_o,
                           torch.where(under, cc_s + share_u, cc_s))
    # :562-565 and :571-572 -- the C writes these as compound assignments,
    # `PackSwq += SurfaceSwq - MAX` and `SurfaceSwq -= SurfaceSwq - MAX`, and
    # the parentheses are not cosmetic: `(a + b) - c` and `a + (b - c)` are
    # different doubles, and `x - (x - MAX)` is not MAX. Written the other
    # way round this came out one ulp from VIC on the layer temperatures.
    pack_swq_new = torch.where(
        over, pack_swq + (surface_swq - SNOW_MAX_SURFACE_SWE),
        torch.where(under, pack_swq - (SNOW_MAX_SURFACE_SWE - surface_swq),
                    pack_swq))
    surface_swq_new = torch.where(
        over, surface_swq - (surface_swq - SNOW_MAX_SURFACE_SWE),
        torch.where(under, surface_swq + (SNOW_MAX_SURFACE_SWE - surface_swq),
                    surface_swq))
    ptot_new = pack_swq_new + pack_ice
    ptn_safe = torch.where(ptot_new != 0.0, ptot_new, one)
    ssn_safe = torch.where(surface_swq_new != 0.0, surface_swq_new, one)

    pack_swq = torch.where(big, pack_swq_new, zero)
    surface_swq = torch.where(big, surface_swq_new, surface_swq)
    pack_ice = torch.where(big, pack_ice, zero)
    pack_temp = torch.where(big, cc_p_new / (CONST_VCPICE_WQ * ptn_safe), zero)
    surf_temp = torch.where(big, cc_s_new / (CONST_VCPICE_WQ * ssn_safe),
                            surf_temp)
    # :568 -- in the small-pack branch SurfaceCC is never recomputed, so
    # what comes back as snow->coldcontent is the value left by the
    # snowfall distribution 400 lines earlier. Reproduced.
    cold_content = torch.where(big, cc_s_new, surface_cc)

    swq = snow_ice + surf_water + pack_water
    ice_water_eq = torch.clamp(lake_ice * state['areai'], min=0.0)
    volume = state['volume'] - ((initial_ice - lake_ice - ice_melt_out)
                                * state['areai'])
    coverage = torch.where(swq > 0.0, one, zero)

    # :583-586 -- VIC's own mass balance test, kept as an output because it
    # is the thing that says whether the bookkeeping above closed.
    mass_error = ((initial_swq - swq) + (initial_ice - lake_ice)
                  + (rainfall + snowfall) - ice_melt_out - melt + vapor_flux)

    new_state = dict(state)
    new_state.update(
        swq=swq, surf_water=surf_water, pack_water=pack_water,
        surf_temp=surf_temp, pack_temp=pack_temp, coldcontent=cold_content,
        ice_water_eq=ice_water_eq, volume=volume, coverage=coverage,
        vapor_flux=-vapor_flux, surface_flux=surface_flux,
        blowing_flux=blowing_flux, mass_error=mass_error)

    diagnostics = dict(
        melt=melt * MM_PER_M, snowmelt=snowmelt, ice_melt=ice_melt_out,
        qnet=qnet, advection=terms['advected'], deltaCC=delta_cc,
        snow_flux=terms['qf'], latent=terms['latent'] + terms['latent_sub'],
        sensible=terms['sensible'], refreeze_energy=refreeze_energy,
        lwnet=terms['long_out'], lake_ice=lake_ice, snow_ice=snow_ice,
        surface_swq=surface_swq, pack_swq=pack_swq, pack_ice=pack_ice,
        avg_cond=avg_cond, sw_conducted=sw_conducted, melting=melting,
        ra_used=terms['ra_used'],
        thin=thin, bracket_failed=bracket_failed)
    return new_state, diagnostics


PA_PER_KPA = 1000.0


def solve_lake(lake, f, soil, dt, wind_h, day_in_year, fracprv):
    """One step of the whole lake.  lakes.eb.c:33-532.
    Returns (new_lake, energy).

    This is the driver that calls everything else in this module in the
    order the C calls it, and the order is the physics:

      1. freezing point, the initial temperature and density profiles, and
         the heat the column holds
      2. precipitation split between the ice and the open water, which is
         also where the snow's water equivalent is rescaled from the whole
         lake's area onto the ice's
      3. the three albedos and the shortwave each fraction absorbs
      4. the open water's surface energy balance, then convective mixing
      6. the snow-and-ice pack on the ice fraction (`ice_melt`)
      7. the water column beneath the ice
      8. ice grown or melted at its base (`lakeice`)
      9. the two columns averaged back into one
     10. the energy balance closure
     11. everything rescaled from per-fraction back to per-lake

    Two things about the shape of this are worth stating because they are
    what makes a batched port different from the C.

    The C runs section 4 only when there is open water and section 6 only
    when there is ice, and in a batch both are true of different columns at
    once. Both are therefore evaluated for every column and selected, which
    means every argument has to stay finite where the branch is not taken:
    `areai` is floored before `ice_melt` divides by it, and `fracprv` before
    the snow's water equivalent is divided by it. A branch that is not
    selected must not be allowed to produce a NaN, because `torch.where`
    evaluates it anyway and the gradient would carry the NaN out.

    `lake['temp']`, `lake['surface']` and the other profiles are node-first,
    [node, batch]; `activenod` is an integer tensor per column and the node
    functions mask on it rather than slicing, because a batch has no single
    active depth.

    `f` carries snowfall and rainfall in MILLIMETRES and pressure, vp and
    vpd in KILOPASCALS, which is what solve_lake's own callers pass
    (:33-48); only `ice_melt` wants pascals and this converts for it.

    **One limitation, stated rather than hidden.** Every column in a batch
    must have the same `activenod`, or none (a column with no active nodes
    is carried through untouched). The node functions underneath --
    `energycalc`, `temp_area`, `eddy`, `colavg` -- take the node count as a
    Python integer because the tridiagonal system they assemble has a
    different last row for the deepest node, and that row's index is the
    node count. Two columns of different depth in one batch would need two
    different systems. This raises rather than silently solving the wrong
    depth for some of them; making the node count per-column is its own
    piece of work.
    """
    from .snow import calc_latent_heat_of_sublimation

    zero = torch.zeros_like(lake['sarea'])
    one = torch.ones_like(lake['sarea'])
    sarea = lake['sarea']
    areai = lake['areai']
    activenod = lake['activenod']
    nodes_on = activenod > 0
    numnod = int(activenod.max().item())
    if not bool(((activenod == numnod) | (activenod == 0)).all()):
        raise ValueError(
            'solve_lake needs one active node count across the batch; got '
            '%s. See the docstring.' % sorted(set(activenod.tolist())))
    numnod = max(numnod, 1)
    # A column with no active nodes still goes through the node functions
    # -- torch.where evaluates the branch it does not take -- so its zero
    # thicknesses and zero areas are replaced with ones first. Without
    # this the division inside colavg is 0/0 and the NaN reaches the
    # gradient of every column that shares the batch.
    dz = torch.where(nodes_on, lake['dz'], one)
    surfdz = torch.where(nodes_on, lake['surfdz'], one)
    surface = torch.where(nodes_on, lake['surface'], torch.ones_like(
        lake['surface']))[:numnod + 1]
    #: the profile the node functions see, sliced to the active depth; the
    #: caller's array is MAX_LAKE_NODES long and the tail is padded back on
    #: at the end so the state keeps one shape from step to step.
    n_slots = lake['temp'].shape[0]
    temp_in = lake['temp'][:numnod]

    # ----- 1. initialise ------------------------------------------------
    t_cutoff = freezing_point(f['pressure'])
    t_open = temp_in.clone()
    t_ice = temp_in.clone()
    water_density = calc_density(t_open)
    water_cp = specheat(t_open)
    sumjoulb = energycalc(temp_in, numnod, dz, surfdz, surface,
                          water_cp, water_density)

    # ----- 2. precipitation, and swq onto the ice's area -----------------
    # :143-155. The snow pack is stored over the whole lake and solved over
    # the ice, so it is divided by the ice fraction here and multiplied
    # back at :467. With no ice at all the pack has nowhere to stand and
    # falls into the lake.
    swq = lake['swq']
    has_snow = swq > 0.0
    frac_safe = torch.where(fracprv > 0.0, fracprv, one)
    ice_throughfall = torch.where(has_snow & (fracprv == 0.0), sarea * swq,
                                  zero)
    swq = torch.where(has_snow,
                      torch.where(fracprv > 0.0, swq / frac_safe, zero), swq)

    rainfall = f['rainfall']
    snowfall = f['snowfall']
    full_ice = fracprv >= 1.0
    part_ice = (fracprv > LAKE_FRACLIM) & (fracprv < 1.0)
    open_lake = ~(full_ice | part_ice)
    bare_rain = (swq <= 0.0) & (rainfall > 0.0)

    itf_full = torch.where(bare_rain, rainfall / MM_PER_M * areai, zero)
    itf_part = ((snowfall / MM_PER_M + rainfall / MM_PER_M)
                * (1.0 - fracprv) * sarea
                + torch.where(bare_rain,
                              rainfall / MM_PER_M * fracprv * sarea, zero))
    itf_open = (rainfall + snowfall) / MM_PER_M * sarea
    ice_throughfall = ice_throughfall + torch.where(
        full_ice, itf_full, torch.where(part_ice, itf_part, itf_open))
    rainfall = torch.where(open_lake, zero,
                           torch.where(bare_rain, zero, rainfall))
    snowfall = torch.where(open_lake, zero, snowfall)

    # ----- 3. albedo and the shortwave each fraction absorbs -------------
    salbedo, tempalbs, albi, albw, last_snow, melting = alblake(
        t_cutoff, f['tair'], lake['SAlbedo'], snowfall, lake['coldcontent'],
        dt, lake['last_snow'], swq, lake['MELTING'], day_in_year, soil['lat'])

    crit = LAKE_SNOWCRIT * LAKE_RHOSNOW / CONST_RHOFW
    deep_snow = swq > crit
    thin_snow = (swq > 0.0) & (swq <= crit)
    bare_ice = (fracprv > 0.0) & (swq <= 0.0)
    sw_ice = torch.where(
        deep_snow, f['shortin'] * (1.0 - tempalbs),
        torch.where(thin_snow, f['shortin'] * (1.0 - (albi + tempalbs) / 2.0),
                    torch.where(bare_ice, f['shortin'] * (1.0 - albi), zero)))
    albedo_lake = torch.where(
        deep_snow, fracprv * tempalbs + (1.0 - fracprv) * albw,
        torch.where(thin_snow,
                    fracprv * (albi + tempalbs) / 2.0 + (1.0 - fracprv) * albw,
                    torch.where(bare_ice,
                                fracprv * albi + (1.0 - fracprv) * albw,
                                albw)))
    sw_water = f['shortin'] * (1.0 - albw)

    # ----- 4. the open water --------------------------------------------
    is_open = ((1.0 - fracprv) > DBL_EPSILON) & nodes_on
    windw = (f['wind'] * math.log((2.0 + LAKE_ZWATER) / LAKE_ZWATER)
             / torch.log(wind_h / LAKE_ZWATER))
    web = water_energy_balance(
        surface, zero, dt, dz, surfdz, soil['lat'],
        t_cutoff, f['tair'], windw, f['pressure'], f['vp'], f['air_density'],
        f['longin'], sw_water, wind_h, t_open, water_density, fracprv,
        water_cp, lake['volume'] - lake['ice_water_eq'], numnod)
    t_mixed, _dens, mixdepth = tracer_mixer(
        web['T'], torch.zeros_like(activenod), surface, numnod,
        dz, surfdz, water_cp)

    t_open = torch.where(is_open, t_mixed, t_open)
    mixdepth = torch.where(is_open, mixdepth, torch.zeros_like(mixdepth))
    sel = lambda v: torch.where(is_open, v, zero)
    evapw = sel(web['evapw'])
    new_ice_area = sel(web['new_ice_area'])
    new_ice_height = sel(web['new_ice_height'])
    new_ice_water_eq = sel(web['new_ice_water_eq'])
    energy_out_bottom = sel(web['energy_out_bottom'])
    atmos_latent = sel((1.0 - fracprv) * web['Qle'])
    atmos_sensible = sel((1.0 - fracprv) * web['Qh'])
    net_long = sel((1.0 - fracprv) * web['LWnet'])
    net_short = sel((1.0 - fracprv) * sw_water)
    refreeze_energy = sel(web['energy_ice_formation'] * (1.0 - fracprv))
    delta_h = sel(web['deltaH'] * (1.0 - fracprv))
    grnd_flux = sel(-1.0 * (energy_out_bottom * (1.0 - fracprv)))
    t_surf_lake = sel((1.0 - fracprv) * t_open[0])

    # ----- 6. the ice ----------------------------------------------------
    snow_rough = soil['snow_rough']
    windi = torch.clamp(
        f['wind'] * torch.log((2.0 + snow_rough) / snow_rough)
        / torch.log(wind_h / snow_rough), min=1.0)
    has_ice = fracprv >= LAKE_FRACLIM
    lv_sub = calc_latent_heat_of_sublimation(f['tair'])
    ra_ice = (torch.log((2.0 + snow_rough) / snow_rough)
              * torch.log(wind_h / snow_rough)
              / (CONST_KARMAN * CONST_KARMAN)) / windi

    im_state = dict(
        swq=swq, surf_water=lake['surf_water'], pack_water=lake['pack_water'],
        surf_temp=lake['surf_temp'], pack_temp=lake['pack_temp'],
        ice_water_eq=lake['ice_water_eq'],
        # ice_melt divides by areai; the columns that have no ice never use
        # the result, but the division happens anyway.
        areai=torch.where(areai > 0.0, areai, one),
        hice=lake['hice'], volume=lake['volume'])
    im_f = dict(
        dt=dt, rainfall=rainfall, snowfall=snowfall, air_temp=f['tair'],
        wind=windi, net_short=sw_ice, long_rad_in=f['longin'],
        air_dens=f['air_density'], lv=lv_sub,
        press=f['pressure'] * PA_PER_KPA, vpd=f['vpd'] * PA_PER_KPA,
        eact_air=f['vp'] * PA_PER_KPA, ra=ra_ice, z=wind_h + snow_rough,
        z0=snow_rough, t_cutoff=t_cutoff)
    ice_state, ice_d = ice_melt(im_state, im_f)

    snowmlt = ice_d['melt']
    advection = ice_d['advection']
    delta_cc = ice_d['deltaCC']
    snow_flux = ice_d['snow_flux']
    qei, qhi = ice_d['latent'], ice_d['sensible']
    lw_neti = ice_d['lwnet']
    temp_refreeze_energy = ice_d['refreeze_energy']
    # :341 passes `&(lake->aero_resist)` as ice_melt's `aero_resist_used`,
    # and IceEnergyBalance.c:145-150 writes the stability-corrected
    # resistance into it on every call. What comes back is therefore not
    # the resistance that went in: it is the corrected one from the last
    # evaluation the solver made. Reading the call site's name instead put
    # this 3x high.
    ra_ice_used = ice_d['ra_used']

    # ----- 7. the water under the ice ------------------------------------
    wui = water_under_ice(
        torch.zeros_like(activenod), sw_ice, f['wind'], t_ice, water_density,
        soil['lat'], numnod, dz, surfdz, t_cutoff,
        surface, water_cp, mixdepth, lake['hice'],
        ice_state['swq'] * CONST_RHOFW / LAKE_RHOSNOW, dt)
    # :352-365 -- water_under_ice runs only inside the ice branch, so a
    # column with too little ice keeps the temperature profile it came in
    # with and contributes nothing to deltaH.
    under_ice = has_ice & nodes_on
    t_ice = torch.where(under_ice, wui['Ti'], t_ice)
    qw = torch.where(under_ice, wui['qw'], zero)
    temphi = torch.where(nodes_on, wui['deltaH'], -sumjoulb)
    energy_out_bottom_ice = torch.where(under_ice, wui['energy_out_bottom'],
                                        zero)

    # ----- 8. ice grown or melted at the base ----------------------------
    iwe = ice_state['ice_water_eq']
    qf, e_melt_bot, iwe_after, _dibot = lakeice(
        sw_ice, fracprv, dt, snow_flux, qw, delta_cc, iwe,
        ice_state['volume'] - new_ice_water_eq, lake['surface'][0])
    grew = iwe > 0.0
    iwe = torch.where(grew, iwe_after, iwe)
    e_melt_bot = torch.where(grew, e_melt_bot, zero)
    qf = torch.where(grew, qf, zero)

    # ...and the ice fraction's share of every term.
    i_atmos_latent = atmos_latent + fracprv * qei
    i_advection = advection * fracprv
    i_atmos_sensible = atmos_sensible + fracprv * qhi
    i_net_long = net_long + fracprv * lw_neti
    i_net_short = net_short + fracprv * sw_ice
    i_delta_h = delta_h + fracprv * temphi
    i_grnd_flux = grnd_flux + -1.0 * (energy_out_bottom_ice * fracprv)
    i_refreeze = (refreeze_energy + temp_refreeze_energy * fracprv
                  + e_melt_bot * fracprv)
    i_t_surf = t_surf_lake + fracprv * ice_state['surf_temp']

    # :384-411, the no-ice branch. What was ice is melted outright and the
    # energy that took is booked; the pack is left exactly as it came in,
    # because ice_melt never ran.
    e_melt_no_ice = torch.where(
        fracprv > 0.0,
        (lake['hice'] * CONST_RHOICE
         + (snowfall / MM_PER_M) * CONST_RHOFW) * CONST_LATICE / dt, zero)
    ra_water = (math.log((2.0 + LAKE_ZWATER) / LAKE_ZWATER)
                * torch.log(wind_h / LAKE_ZWATER)
                / (CONST_KARMAN * CONST_KARMAN)) / windi

    pick = lambda a, b: torch.where(has_ice, a, b)
    atmos_latent = pick(i_atmos_latent, atmos_latent)
    atmos_sensible = pick(i_atmos_sensible, atmos_sensible)
    net_long = pick(i_net_long, net_long)
    net_short = pick(i_net_short, net_short)
    delta_h = pick(i_delta_h, delta_h)
    grnd_flux = pick(i_grnd_flux, grnd_flux)
    refreeze_energy = pick(i_refreeze, zero)
    advection = pick(i_advection, zero)
    delta_cc = pick(delta_cc, zero)
    snow_flux = pick(snow_flux, zero)
    t_surf_lake = pick(i_t_surf, t_surf_lake)
    qei = pick(qei, zero)
    qhi = pick(qhi, zero)
    lw_neti = pick(lw_neti, zero)
    qf = pick(qf, zero)
    temphi = pick(temphi, zero)
    energy_ice_melt_bot = pick(e_melt_bot, e_melt_no_ice)
    snowmlt = pick(snowmlt, zero)
    aero_resist = pick(ra_ice_used, ra_water)
    areai = pick(areai, zero)
    hice = pick(lake['hice'], zero)
    ice_water_eq = pick(iwe, zero)
    volume = pick(ice_state['volume'], lake['volume'])
    tempi = pick(ice_state['surf_temp'], lake.get('tempi', zero))
    snow = {k: pick(ice_state[k], lake[k]) for k in
            ('swq', 'surf_water', 'pack_water', 'surf_temp', 'pack_temp',
             'coldcontent')}
    snow['swq'] = pick(ice_state['swq'], swq)
    vapor_flux = pick(ice_state['vapor_flux'], zero)
    surface_flux = pick(ice_state['surface_flux'], zero)
    blowing_flux = pick(ice_state['blowing_flux'], zero)

    # ----- 9. one column again -------------------------------------------
    temp, density = colavg(t_open, t_ice, fracprv, numnod, dz, surfdz)
    if n_slots > numnod:
        pad = lake['temp'][numnod:]
        temp = torch.cat([temp, pad], 0)
        density = torch.cat([density, torch.zeros_like(pad)], 0)
    node = torch.arange(temp.shape[0], device=temp.device).view(
        -1, *([1] * (temp.dim() - 1)))
    inside = node < activenod
    n_safe = torch.where(nodes_on, activenod.to(temp.dtype), one)
    tempavg = torch.where(nodes_on,
                          (torch.where(inside, temp, torch.zeros_like(temp))
                           / n_safe).sum(0), torch.full_like(sarea, -99.0))

    # ----- 10. closure ----------------------------------------------------
    # :434-444 computes an `error` from the ice fraction's terms alone and
    # then overwrites it at :450 with the whole-lake sum, so the first is
    # dead. It is not reproduced; the value VIC reports is the second.
    snow_flux = torch.zeros_like(snow_flux)
    delta_h = delta_h * -1.0
    error = (net_short + net_long + atmos_sensible + atmos_latent + delta_h
             + grnd_flux + refreeze_energy + advection)

    # ----- 11. back to whole-lake units ----------------------------------
    snow['swq'] = snow['swq'] * fracprv
    snow['surf_water'] = snow['surf_water'] * fracprv
    snow['pack_water'] = snow['pack_water'] * fracprv
    snow_depth = snow['swq'] * CONST_RHOFW / LAKE_RHOSNOW
    snow['coldcontent'] = snow['coldcontent'] * fracprv
    vapor_flux = vapor_flux * fracprv
    blowing_flux = blowing_flux * fracprv
    surface_flux = surface_flux * fracprv
    snow_melt_mm = snowmlt * fracprv

    evapw_m3 = evapw * ((1.0 - fracprv) * dt) / MM_PER_M * sarea
    lake_vapor_flux = vapor_flux * sarea
    swe = snow['swq'] * sarea
    snowmlt_m3 = snowmlt * fracprv / MM_PER_M * sarea
    pack_water_m3 = snow['pack_water'] * sarea
    surf_water_m3 = snow['surf_water'] * sarea
    sdepth = snow_depth * sarea

    # :492-513 -- the ice the open fraction grew is added to the area and to
    # the water equivalent, and the thickness follows from the two. A cover
    # thinner than LAKE_FRACMIN is spread out rather than thinned further,
    # which is what keeps hice from going to zero over a whole lake.
    grew_open = new_ice_area > 0.0
    area_total = areai + torch.where(grew_open, new_ice_area, zero)
    ice_water_eq = ice_water_eq + torch.where(grew_open, new_ice_water_eq,
                                              zero)
    at_safe = torch.where(area_total > 0.0, area_total, one)
    hice = torch.where((ice_water_eq > 0.0) & (area_total > 0.0),
                       (ice_water_eq / at_safe) * CONST_RHOFW / CONST_RHOICE,
                       zero)
    gone = hice <= 0.0
    spread = (~gone) & (hice < LAKE_FRACMIN)
    area_total = torch.where(gone, zero,
                             torch.where(spread,
                                         area_total * hice / LAKE_FRACMIN,
                                         area_total))
    hice = torch.where(gone, zero,
                       torch.where(spread,
                                   torch.full_like(hice, LAKE_FRACMIN), hice))
    coverage = torch.where(snow['swq'] > 0.0,
                           area_total / torch.where(sarea > 0.0, sarea, one),
                           zero)

    # :137 -- with no active nodes and no ice, solve_lake does nothing at all.
    run = nodes_on | (lake['areai'] > 0.0)
    keep = lambda new, old: torch.where(run, new, old)
    keep_n = lambda new, old: torch.where(run, new, old)

    out = dict(lake)
    out.update(
        sarea_save=lake['sarea'], volume_save=lake['volume'],
        swe_save=lake['swe'],
        temp=keep_n(temp, lake['temp']), density=keep_n(density,
                                                        lake.get('density',
                                                                 temp)),
        tempavg=keep(tempavg, lake.get('tempavg', zero)),
        tempi=keep(tempi, lake.get('tempi', zero)),
        volume=keep(volume, lake['volume']),
        ice_water_eq=keep(ice_water_eq, lake['ice_water_eq']),
        areai=keep(areai, lake['areai']),
        new_ice_area=keep(area_total, lake.get('new_ice_area', zero)),
        hice=keep(hice, lake['hice']),
        SAlbedo=keep(salbedo, lake['SAlbedo']),
        last_snow=torch.where(run, last_snow, lake['last_snow']),
        MELTING=torch.where(run, melting, lake['MELTING']),
        swq=keep(snow['swq'], lake['swq']),
        surf_water=keep(snow['surf_water'], lake['surf_water']),
        pack_water=keep(snow['pack_water'], lake['pack_water']),
        surf_temp=keep(snow['surf_temp'], lake['surf_temp']),
        pack_temp=keep(snow['pack_temp'], lake['pack_temp']),
        coldcontent=keep(snow['coldcontent'], lake['coldcontent']),
        snow_depth=keep(snow_depth, lake.get('snow_depth', zero)),
        snow_melt=keep(snow_melt_mm, zero),
        coverage=keep(coverage, lake.get('coverage', zero)),
        aero_resist=keep(aero_resist, lake.get('aero_resist', zero)),
        evapw=keep(evapw_m3, zero), vapor_flux=keep(lake_vapor_flux, zero),
        snow_vapor_flux=keep(vapor_flux, zero),
        blowing_flux=keep(blowing_flux, zero),
        surface_flux=keep(surface_flux, zero),
        swe=keep(swe, lake['swe']), snowmlt=keep(snowmlt_m3, zero),
        lake_pack_water=keep(pack_water_m3, zero),
        lake_surf_water=keep(surf_water_m3, zero),
        sdepth=keep(sdepth, lake.get('sdepth', zero)),
        ice_throughfall=keep(ice_throughfall, zero))

    energy = dict(
        AlbedoLake=keep(albedo_lake, zero),
        AlbedoUnder=keep(albedo_lake, zero),
        AtmosLatent=keep(atmos_latent, zero),
        AtmosSensible=keep(atmos_sensible, zero),
        NetLongAtmos=keep(net_long, zero),
        NetShortAtmos=keep(net_short, zero),
        deltaH=keep(delta_h, zero), grnd_flux=keep(grnd_flux, zero),
        refreeze_energy=keep(refreeze_energy, zero),
        advection=keep(advection, zero), deltaCC=keep(delta_cc, zero),
        snow_flux=keep(snow_flux, zero), Tsurf=keep(t_surf_lake, zero),
        error=keep(error, zero), Qnet_ice=keep(ice_d['qnet'], zero),
        qf=qf, qw=qw, temphi=temphi,
        energy_ice_melt_bot=energy_ice_melt_bot,
        energy_out_bottom=energy_out_bottom,
        energy_out_bottom_ice=energy_out_bottom_ice,
        new_ice_area=new_ice_area, new_ice_height=new_ice_height,
        new_ice_water_eq=new_ice_water_eq, Tcutoff=t_cutoff,
        sw_ice=sw_ice, sw_water=sw_water, mixdepth=mixdepth)
    return out, energy


def lake_recharge(lakefrac, max_newfraction, volume, ice_water_eq, moist,
                  max_moist, wdew, snow_canopy, swq, basin0):
    """Water the lake gives to wetland it has just flooded.
    lakes.eb.c:1944-2005.  Returns (recharge, delta_moist, volume).

    A lake that grows takes wetland with it. VIC settles that by filling
    the newly flooded soil to saturation out of the lake, and taking the
    above-ground storage that was standing there -- the canopy store, the
    snow on the canopy and the snowpack -- into the lake in exchange
    (:1966-1976). The exchange can be either way round: `recharge` is
    negative when the snow that came in is worth more than the water the
    soil took.

    If the lake has not got the liquid water to fill the soil, it gives
    what it has and the soil is filled layer by layer from the top until
    that runs out (:1983-2005).

    `moist`, `max_moist` and `delta_moist` are [layer, ...] in millimetres
    over the wetland's starting area; `recharge` is cubic metres.
    """
    n = moist.shape[0]
    zero = torch.zeros_like(volume)
    grew = max_newfraction > lakefrac
    strip = max_newfraction - lakefrac
    rest = 1.0 - lakefrac
    rest_safe = torch.where(rest.abs() > 0.0, rest, torch.ones_like(rest))
    room = [max_moist[i] - moist[i] for i in range(n)]

    # :1953-1961 -- fill the flooded strip to saturation.
    dm_full = [room[i] * strip / rest_safe for i in range(n)]
    recharge = sum(dm / MM_PER_M * rest * basin0 for dm in dm_full)
    # :1966-1976 -- and take what was standing on it.
    abovegrnd = (wdew / MM_PER_M + snow_canopy + swq) * strip * basin0
    recharge = recharge - abovegrnd

    enough = (volume - ice_water_eq) > recharge
    # :1983-2005 -- not enough liquid water: give what there is and fill
    # the soil from the top until it runs out.
    short = volume - ice_water_eq
    strip_vol = torch.where(strip.abs() > 0.0, strip * basin0,
                            torch.ones_like(strip))
    left = (MM_PER_M * short / strip_vol
            + (wdew + snow_canopy * MM_PER_M + swq * MM_PER_M))
    dm_short = []
    for i in range(n):
        fits = left > room[i]
        dm_short.append(torch.where(fits, room[i] * strip / rest_safe,
                                    left * strip / rest_safe))
        left = torch.where(fits, left - room[i], torch.zeros_like(left))

    recharge = torch.where(grew, torch.where(enough, recharge, short), zero)
    volume = torch.where(grew,
                         torch.where(enough, volume - recharge, ice_water_eq),
                         volume)
    delta_moist = torch.stack(
        [torch.where(grew, torch.where(enough, dm_full[i], dm_short[i]),
                     torch.zeros_like(dm_full[i])) for i in range(n)], 0)
    return recharge, delta_moist, volume


def redistribute_lake_nodes(temp, activenod, isave_n):
    """The temperature profile carried across a change in the number of
    nodes.  lakes.eb.c:2154-2171.  Returns the new profile.

    VIC does not interpolate: it walks a cursor down the OLD profile in
    steps of 1/activenod, taking `isave_n` samples for each new node and
    averaging them. The cursor is never reset between nodes, so after
    activenod * isave_n steps it has advanced by exactly isave_n and the
    last sample reads `temp[isave_n]` -- one past the old profile's last
    node. That entry is whatever the array held from an earlier, deeper
    step. It is reproduced rather than clamped, because the question is
    what VIC computes; the C's own comment at :2151-2152 says the scheme
    does not conserve the column's heat either.

    `activenod` and `isave_n` are integers, not tensors, for the same
    reason `solve_lake` takes an integer node count.
    """
    if activenod == isave_n or activenod <= 0:
        return temp
    # The cursor is a running sum of 1/activenod in a double, not the
    # closed form (j + 1) / activenod, and at a boundary the two floor to
    # different nodes. The sum is reproduced.
    index = 0.0
    step = 1.0 / activenod
    out = []
    for _k in range(activenod):
        total = None
        for _i in range(isave_n):
            index += step
            src = temp[min(int(math.floor(index)), temp.shape[0] - 1)]
            total = src if total is None else total + src
        out.append(total / isave_n if isave_n > 0 else total)
    new = temp.clone()
    for k in range(activenod):
        new[k] = out[k]
    return new


def water_balance(lake, wetland, soil, lake_con, dt, lakefrac, isave_n,
                  carbon=False):
    """The lake's water balance and the area it trades with the wetland.
    lakes.eb.c:1808-2300.  Returns (new_lake, new_wetland, diagnostics).

    solve_lake settles the energy; this settles the water, and with it the
    lake's area, which is the part that reaches the rest of the cell. In
    order:

      1. inflow from the rest of the grid cell, and evaporation, capped at
         the liquid water there is
      2. an estimate of the new area, and from it the strip of wetland the
         lake is about to flood or expose
      3. `lake_recharge`: the lake fills the newly flooded soil to
         saturation and takes the snow and canopy water that was standing
         there
      4. `lake_drain`: ARNO baseflow through the wetland's bottom layer and
         weir runoff round the lake's circumference
      5. the depth, the node grid and the areas re-derived for the next
         step, and the temperature profile carried across any change in the
         number of nodes
      6. every storage and flux on both tiles restated over its new area

    `lakefrac` is the lake's fraction of the cell at the START of the step
    and `isave_n` the node count solve_lake used, both of which the caller
    has to have kept.

    Two things the C does here are NOT done here, and are named rather than
    approximated. `distribute_node_moisture_properties` (:2199-2217) is the
    soil thermal solver's, not the lake's, and belongs to the caller that
    owns the soil column. `initialize_lake` (:2262) builds a lake where
    there was none at the start of the step; that needs the lake parameter
    file's initial state and is the reader's job, so this returns
    `diagnostics['lake_is_new']` and leaves it to the caller.
    """
    zero = torch.zeros_like(lake['volume'])
    basin0 = lake_con['basin'][0]
    z, basin = lake_con['z'], lake_con['basin']
    maxvolume, maxdepth = lake_con['maxvolume'], lake_con['maxdepth']

    # ----- 1-2, :1865-1914 ----------------------------------------------
    inflow_volume = (lake['runoff_in'] + lake['baseflow_in']
                     + lake['channel_in'])
    volume, evapw = lake_inflow(
        lake['volume'], lake['ice_water_eq'], lake['evapw'],
        lake['ice_throughfall'], inflow_volume, lake['snowmlt'])
    volume_save = volume

    ldepth = get_depth(z, basin, volume - lake['ice_water_eq'], maxvolume,
                       maxdepth)
    surfacearea = torch.maximum(get_sarea(z, basin, ldepth),
                                lake['new_ice_area'])
    # :1914 -- the estimate the recharge is based on, kept because the area
    # will move again below and the wetland exchange was sized on this one.
    max_newfraction = surfacearea / basin0

    # ----- 3, :1944-2005 -------------------------------------------------
    recharge, delta_moist, volume = lake_recharge(
        lakefrac, max_newfraction, volume, lake['ice_water_eq'],
        wetland['moist'], soil['max_moist'], wetland['wdew'],
        wetland['snow_canopy'], wetland['swq'], basin0)

    # ----- 4, :2010-2118 -------------------------------------------------
    volume, ldepth, _sarea, baseflow_out, runoff_out = lake_drain(
        volume, volume_save, lake['ice_water_eq'], z, basin, maxvolume,
        lake_con['mindepth'], lake_con['minvolume'], lake_con['wfrac'], dt,
        soil['Dsmax_step'], soil['Ds'], soil['Ws'], soil['c'],
        soil['max_moist'][-1], soil['resid_moist_bot'], wetland['ice_bot'],
        lake['new_ice_area'])

    # ----- 5, :2124-2175 -------------------------------------------------
    ldepth = get_depth(z, basin, volume - lake['ice_water_eq'], maxvolume,
                       maxdepth)
    surfdz, dz, activenod, surface, _s0, volume = derived_lake_dimensions(
        ldepth, lake['ice_water_eq'], z, basin, maxdepth, maxvolume)
    sarea = torch.maximum(lake['new_ice_area'], surface[0])
    newfraction = sarea / basin0

    nodes = int(activenod.max().item())
    if not bool(((activenod == nodes) | (activenod == 0)).all()):
        raise ValueError(
            'water_balance needs one active node count across the batch; '
            'got %s. See solve_lake.' % sorted(set(activenod.tolist())))
    temp = redistribute_lake_nodes(lake['temp'], nodes, isave_n)

    # ----- 6, :2179-2290 -------------------------------------------------
    moist, wdew, wet_baseflow = advect_soil_veg_storage(
        lakefrac, max_newfraction, newfraction, delta_moist,
        wetland['moist'], soil['max_moist'], wetland['wdew'], basin0,
        wetland['baseflow_m3'])
    wet_fluxes = {k: v for k, v in wetland.get('fluxes', {}).items()}
    if wet_fluxes:
        keys = list(wet_fluxes)
        scaled = rescale_fluxes(1.0 - lakefrac, 1.0 - newfraction,
                                *[wet_fluxes[k] for k in keys])
        wet_fluxes = dict(zip(keys, scaled))
    snow_keys = ('snow_depth', 'snow_pack_water', 'snow_canopy',
                 'snow_surf_water', 'swq')
    wet_snow = advect_snow_storage(
        lakefrac, max_newfraction, newfraction,
        *[wetland[k] for k in snow_keys])
    wet_snow = dict(zip(snow_keys, wet_snow))

    # :2288-2291 -- the lake's own fluxes, restated over its end-of-step
    # area in millimetres, which is the form the rest of VIC reads.
    area_end = torch.clamp(newfraction * basin0, min=DBL_EPSILON)
    has_lake = newfraction > 0.0
    lake_soil = dict(
        runoff=torch.where(has_lake, runoff_out * MM_PER_M / area_end, zero),
        baseflow=torch.where(has_lake, baseflow_out * MM_PER_M / area_end,
                             zero),
        inflow=torch.where(has_lake, baseflow_out * MM_PER_M / area_end,
                           zero),
        evap=torch.where(has_lake, evapw * MM_PER_M / area_end, zero))

    lake_snow_keys = ('snow_depth', 'pack_water', 'snow_canopy',
                      'surf_water', 'swq')
    had_lake = lakefrac > 0.0
    rescaled = rescale_snow_storage(lakefrac, newfraction,
                                    *[lake[k] for k in lake_snow_keys])
    lake_snow = {k: torch.where(had_lake & has_lake, v, lake[k])
                 for k, v in zip(lake_snow_keys, rescaled)}
    coverage = torch.where(lake_snow['swq'] > 0.0,
                           lake['new_ice_area']
                           / torch.where(sarea > 0.0, sarea,
                                         torch.ones_like(sarea)), zero)
    coverage = torch.where(had_lake & has_lake, coverage, lake['coverage'])

    # :2260-2280 -- a lake where there was none at the start of the step is
    # built from scratch by initialize_lake(preserve_essentials = true).
    # That call comes AFTER the four lake->soil fluxes above were assigned,
    # and it sets them back to zero (initialize_lake.c:202-222), so a
    # brand-new lake reports no runoff, baseflow or evaporation however
    # much water left it this step. Reproduced, because the caller's water
    # balance is written around it. The depth, the volume and the outflow
    # volumes themselves are `essential` and survive.
    fresh = (~had_lake) & has_lake
    for k in lake_snow:
        lake_snow[k] = torch.where(fresh, torch.zeros_like(lake_snow[k]),
                                   lake_snow[k])
    coverage = torch.where(fresh, zero, coverage)
    lake_soil = {k: torch.where(fresh, zero, v) for k, v in lake_soil.items()}
    node = torch.arange(temp.shape[0], device=temp.device).view(
        -1, *([1] * (temp.dim() - 1)))
    soil_t0 = wetland.get('soil_T0', zero)
    temp = torch.where(fresh,
                       torch.where(node < activenod,
                                   soil_t0.expand_as(temp),
                                   torch.zeros_like(temp)),
                       temp)

    # :2270-2285 -- a wetland that is gone at the end of the step hands its
    # evaporation to the lake, and :2291-2300 the other way round.
    wetland_gone = (newfraction >= 1.0) & (lakefrac < 1.0) & (lakefrac > 0.0)
    evapw = evapw + torch.where(
        wetland_gone,
        (wetland['evap_total'] / MM_PER_M
         + wetland['canopyevap'] / MM_PER_M
         + wetland['canopy_vapor_flux'] + wetland['vapor_flux'])
        * (1.0 - lakefrac) * basin0, zero)
    lake_gone = (newfraction <= 0.0) & (lakefrac > 0.0) & (lakefrac < 1.0)
    wet_area_end = torch.clamp((1.0 - newfraction) * basin0,
                               min=DBL_EPSILON)
    hand_back = dict(
        evap=torch.where(lake_gone, MM_PER_M * evapw / wet_area_end, zero),
        runoff=torch.where(lake_gone, MM_PER_M * runoff_out / wet_area_end,
                           zero),
        baseflow=torch.where(lake_gone,
                             MM_PER_M * baseflow_out / wet_area_end, zero),
        inflow=torch.where(lake_gone,
                           MM_PER_M * baseflow_out / wet_area_end, zero))

    new_lake = dict(lake)
    new_lake.update(volume=volume, ldepth=ldepth, sarea=sarea,
                    surface=surface, surfdz=surfdz, dz=dz,
                    activenod=activenod, temp=temp, evapw=evapw,
                    recharge=recharge, baseflow_out=baseflow_out,
                    runoff_out=runoff_out, coverage=coverage,
                    # NOT the local volume_save above: that one is
                    # water_balance's own (lakes.eb.c:1824, :1877) and
                    # lake->volume_save, which OUT_LAKE_DSTOR reads, is the
                    # start-of-step volume solve_lake set (:69-71)
                    tempavg=torch.where(fresh, temp[0],
                                        lake.get('tempavg', zero)),
                    areai=torch.where(fresh, zero, lake.get('areai', zero)),
                    hice=torch.where(fresh, zero, lake.get('hice', zero)),
                    new_ice_area=torch.where(fresh, zero,
                                             lake['new_ice_area']),
                    SAlbedo=torch.where(fresh, zero,
                                        lake.get('SAlbedo', zero)),
                    swe=torch.where(fresh, zero, lake.get('swe', zero)),
                    surf_temp=torch.where(fresh, zero,
                                          lake.get('surf_temp', zero)),
                    pack_temp=torch.where(fresh, zero,
                                          lake.get('pack_temp', zero)),
                    coldcontent=torch.where(fresh, zero,
                                            lake.get('coldcontent', zero)),
                    ice_throughfall=torch.where(
                        fresh, zero, lake.get('ice_throughfall', zero)))
    new_lake.update(lake_snow)
    new_wetland = dict(wetland)
    new_wetland.update(moist=moist, wdew=wdew, baseflow_m3=wet_baseflow)
    new_wetland.update(wet_snow)
    new_wetland.update(wet_fluxes)

    if carbon:
        for pool in ('litter', 'inter', 'slow'):
            key = 'soil_c_' + pool
            if key in wetland:
                new_lake[key], new_wetland[key] = advect_carbon_storage(
                    lakefrac, newfraction, lake.get(key, zero),
                    wetland[key])

    diagnostics = dict(
        newfraction=newfraction, max_newfraction=max_newfraction,
        inflow_volume=inflow_volume, delta_moist=delta_moist,
        lake_soil=lake_soil, hand_back=hand_back,
        lake_is_new=(~had_lake) & has_lake,
        wetland_is_new=(lakefrac >= 1.0) & (newfraction < 1.0),
        nodes_changed=nodes != isave_n)
    return new_lake, new_wetland, diagnostics


# ---------------------------------------------------------------------------
# The lake parameter file and the state a run starts from
# ---------------------------------------------------------------------------

LAKE_BETA = 0.001               #: exponent in VIC's default depth-area curve


def compute_lake_params(z, cl, numnod, cell_area, mindepth, rpercent,
                        lake_profile):
    """The basin geometry a lake parameter file implies.
    compute_lake_params.c:34-90.  Returns a dict of lake_con fields.

    `z` and `cl` are [numnod + 1, ...] -- the elevation of each basin level
    above the deepest point and the fraction of the grid cell it covers.
    With LAKE_PROFILE off only z[0] and cl[0] are read from the file and
    the rest of the curve is generated as a paraboloid of exponent
    LAKE_BETA; with it on, every level is given.

    LAKE_BETA is 0.001, not the 0.5 that "depth-area profile" suggests
    (initialize_parameters.c:78). At that exponent the generated basin has
    very nearly vertical walls: a level at a tenth of the maximum depth
    still covers 99.8% of the surface area. A port that guessed 0.5 would
    make every generated lake a shallow bowl, and the depth a given volume
    stands at would be wrong by metres rather than percent.
    """
    z = z.clone()
    cl = cl.clone()
    maxdepth = z[0]
    basin = torch.zeros_like(cl)
    basin[0] = cl[0] * cell_area
    if not lake_profile:
        # :50-66 -- the generated curve, a paraboloid of revolution.
        tempdz = maxdepth / float(numnod)
        radius = torch.sqrt(basin[0] / CONST_PI)
        for i in range(1, numnod + 1):
            z[i] = torch.clamp((numnod - i) * tempdz, min=0.0)
            x = torch.pow(z[i] / maxdepth, LAKE_BETA) * radius
            basin[i] = CONST_PI * x * x
            cl[i] = basin[i] / cell_area
    else:
        # :68-78 -- the given curve, with the bottom closing to a point.
        z[numnod] = torch.zeros_like(z[numnod])
        cl[numnod] = torch.zeros_like(cl[numnod])
        for i in range(1, numnod + 1):
            basin[i] = cl[i] * cell_area

    # :81-86 -- the basin's capacity, level by level.
    maxvolume = torch.zeros_like(basin[0])
    for i in range(1, numnod + 1):
        maxvolume = maxvolume + ((basin[i] + basin[i - 1])
                                 * (z[i - 1] - z[i]) / 2.0)
    minvolume = get_volume(z, basin, mindepth, maxvolume)
    return dict(z=z, basin=basin, Cl=cl, numnod=numnod, maxdepth=maxdepth,
                maxvolume=maxvolume, mindepth=mindepth, minvolume=minvolume,
                rpercent=rpercent, bpercent=rpercent, basin0=basin[0])


def read_lakeparam(path, gridcel, cv, cell_area, lake_profile,
                   dtype=torch.float64, device=None, log=None):
    """The classic driver's lake parameter file.  read_lakeparam.c:36-206.
    Returns (lake_con, lake_idx), or (None, -1) where the cell has no lake.

    Format, per cell: a line `gridcel lake_idx numnod mindepth wfrac
    depth_in rpercent`, and -- when `lake_idx >= 0` -- a second line with
    the depth-area curve. With LAKE_PROFILE off that line is one
    `z[0] Cl[0]` pair; with it on it is `numnod` pairs, deepest last.

    `Cl[0]` has to equal the vegetation tile's own area fraction to within
    1%, and the C then replaces it with that fraction outright (:141-150),
    so a file that is within tolerance but not equal does not shift the
    basin. That substitution is reproduced.
    """
    device = torch.empty(0).device if device is None else device
    tokens = []
    with open(path) as fh:
        for line in fh:
            tokens.append(line.split())
    i = 0
    while i < len(tokens):
        head = tokens[i]
        if not head:
            i += 1
            continue
        cell = int(head[0])
        idx = int(head[1])
        if cell == gridcel:
            break
        i += 2 if idx >= 0 else 1
    else:
        raise ValueError('cell %d is not in %s' % (gridcel, path))
    head = tokens[i]
    idx = int(head[1])
    if idx < 0:
        return None, -1
    numnod = int(head[2])
    if numnod < 1 or numnod > MAX_LAKE_NODES:
        raise ValueError('%s: cell %d has %d lake nodes; VIC allows 1 to %d'
                         % (path, gridcel, numnod, MAX_LAKE_NODES))
    mindepth, wfrac, depth_in, rpercent = [float(x) for x in head[3:7]]
    for name, value, lo, hi in (('mindepth', mindepth, 0.0, None),
                                ('wfrac', wfrac, 0.0, 1.0),
                                ('depth_in', depth_in, 0.0, None),
                                ('rpercent', rpercent, 0.0, 1.0)):
        if value < lo or (hi is not None and value > hi):
            raise ValueError('%s: cell %d has %s = %g, outside VIC\'s range'
                             % (path, gridcel, name, value))

    pairs = [float(x) for x in tokens[i + 1]]
    zs = torch.zeros(MAX_LAKE_NODES + 1, dtype=dtype, device=device)
    cls = torch.zeros(MAX_LAKE_NODES + 1, dtype=dtype, device=device)
    n_pairs = numnod if lake_profile else 1
    for k in range(n_pairs):
        zs[k] = pairs[2 * k]
        cls[k] = pairs[2 * k + 1]
    # :141-150 -- Cl[0] must match the tile's area fraction, and is then
    # replaced by it.
    if abs(1.0 - float(cls[0]) / cv) > 0.01:
        raise ValueError(
            '%s: cell %d gives a lake area fraction of %g at the top of the '
            'basin; the vegetation tile that holds it covers %g, and VIC '
            'requires them equal to 1%%'
            % (path, gridcel, float(cls[0]), cv))
    cls[0] = cv

    con = compute_lake_params(zs[:numnod + 1], cls[:numnod + 1], numnod,
                              torch.tensor(cell_area, dtype=dtype,
                                           device=device),
                              torch.tensor(mindepth, dtype=dtype,
                                           device=device), rpercent,
                              lake_profile)
    # :190-203 -- the initial depth is clipped to the basin.
    if depth_in > float(con['maxdepth']):
        if log is not None:
            log('  %s: initial lake depth %g m exceeds the basin depth %g m; '
                'using the basin depth' % (path, depth_in,
                                           float(con['maxdepth'])))
        depth_in = float(con['maxdepth'])
    con['depth_in'] = depth_in
    con['wfrac'] = wfrac
    con['lake_idx'] = idx
    return con, idx


def initial_lake_state(con, avg_temp, dtype=torch.float64, device=None,
                       honour_depth_in=False, vic_core=None):
    """The state a lake run starts in.  initialize_lake.c:35-235 and
    generate_default_state.c's `generate_default_lake_state`.

    **VIC discards the initial depth the parameter file gives.**
    `generate_default_lake_state` takes `all_vars_struct *` and then does
    `lake_var_struct lake; lake = all_vars->lake_var;` -- a copy of the
    whole struct, arrays included -- and writes `lake.ldepth =
    lake_con.depth_in` and the initial temperature profile into the copy.
    Nothing is written back. So the lake starts at the depth
    `initialize_lake` left it, which is zero, and at 0 C, whatever the
    file says and whatever the cell's mean annual temperature is.

    Measured, not inferred: FindleyLake's parameter file gives
    `depth_in = 23.4` m in a basin 24.4 m deep, and pristine VIC 5.0.1 run
    on that cell reports OUT_LAKE_DEPTH = 1.9938 m at the end of the first
    day -- an empty basin filling from the wetland's runoff at about a
    metre a day, not a full lake. The spin-up that costs is real: the lake
    is still filling months later.

    With `vic_core` None or '5.0.1' (this function's own default) it
    reproduces VIC 5.0.1. The lake model (model/lakecell.py) passes the run's
    core, so a run on the default core 5.1.0 takes the 5.1.0 start described
    below and one on `--core 5.0.1` this one. `honour_depth_in=True` starts
    the lake where the file says and the profile at `avg_temp`, which is what
    the C was written to do; with the 5.0.1 core a run that uses it is not
    comparable with VIC's.

    **VIC 5.1.0 fixed half of it** (`vic_core='5.1.0'`).
    generate_default_lake_state.c 5.1.0:15-35 takes the lake through a
    pointer, so `ldepth = depth_in` now sticks. The temperature loop below
    it still runs over `lake->activenod`, which is the 0 that
    initialize_lake.c:45-48 (ldepth 0, vic_init.c 5.1.0:1548) left, so the
    profile stays at initialize_lake's 0 C; compute_derived_lake_dimensions
    then grows the node grid to the full depth (vic_populate_model_state.c
    5.1.0:47-51). Measured: pristine 5.1.0 on FindleyLake reports
    OUT_LAKE_DEPTH 23.3993 m and a surface at -0.0065 C with ice forming
    after its first hour.
    """
    device = torch.empty(0).device if device is None else device
    one = torch.zeros((), dtype=dtype, device=device)
    if vic_core == '5.1.0':
        honour_depth_in = True
    ldepth = (torch.as_tensor(con['depth_in'], dtype=dtype, device=device)
              if honour_depth_in else one.clone())
    surfdz, dz, activenod, surface, sarea, volume = derived_lake_dimensions(
        ldepth.reshape(1), one.reshape(1), con['z'].unsqueeze(-1),
        con['basin'].unsqueeze(-1), con['maxdepth'].reshape(1),
        con['maxvolume'].reshape(1))
    nod = int(activenod)
    temp = torch.zeros(MAX_LAKE_NODES, 1, dtype=dtype, device=device)
    if honour_depth_in and vic_core != '5.1.0':
        temp[:nod] = torch.as_tensor(avg_temp, dtype=dtype, device=device)
    z1 = lambda v: torch.full((1,), float(v), dtype=dtype, device=device)
    return dict(
        ldepth=ldepth.reshape(1), surfdz=surfdz, dz=dz, activenod=activenod,
        surface=surface, sarea=sarea, volume=volume, temp=temp,
        density=torch.full((MAX_LAKE_NODES, 1), CONST_RHOFW, dtype=dtype,
                           device=device),
        ice_water_eq=z1(0), areai=z1(0), new_ice_area=z1(0), hice=z1(0),
        tempi=z1(0), tempavg=z1(0), swe=z1(0), sdepth=z1(0), SAlbedo=z1(0),
        coldcontent=z1(0), surf_temp=z1(0), pack_temp=z1(0),
        surf_water=z1(0), pack_water=z1(0), swq=z1(0), snow_depth=z1(0),
        snow_canopy=z1(0), coverage=z1(0), aero_resist=z1(0),
        last_snow=torch.zeros(1, dtype=torch.long, device=device),
        MELTING=torch.zeros(1, dtype=torch.bool, device=device),
        evapw=z1(0), ice_throughfall=z1(0), snowmlt=z1(0), recharge=z1(0),
        runoff_in=z1(0), baseflow_in=z1(0), channel_in=z1(0),
        runoff_out=z1(0), baseflow_out=z1(0), vapor_flux=z1(0),
        snow_vapor_flux=z1(0), blowing_flux=z1(0), surface_flux=z1(0),
        snow_melt=z1(0), volume_save=volume.clone(), swe_save=z1(0),
        sarea_save=sarea.clone())


def calc_rainonly(prec, tair, t_max_snow, t_min_rain):
    """(rain, snow) of `prec`.  calc_rainonly.c:15-35, as vic_run.c
    5.1.0:427-437 uses it for the lake: rain is the ramp's share counted up
    from MIN_RAIN_TEMP, snow is what is left."""
    t_max_snow = torch.as_tensor(t_max_snow, dtype=prec.dtype,
                                 device=prec.device)
    t_min_rain = torch.as_tensor(t_min_rain, dtype=prec.dtype,
                                 device=prec.device)
    ramp = (tair < t_max_snow) & (tair > t_min_rain)
    rain = torch.where(ramp, (tair - t_min_rain) / (t_max_snow - t_min_rain)
                       * prec,
                       torch.where(tair >= t_max_snow, prec,
                                   torch.zeros_like(prec)))
    return rain, prec - rain


def lake_step(lake, wetland, soil, lake_con, f, dt, wind_h, day_in_year,
              carbon=False):
    """One whole model step for a lake tile.  vic_run.c:155-482.
    Returns (new_lake, new_wetland, diagnostics).

    The sequence VIC runs between the two halves of the lake model, which
    is where the state that ties them together lives:

    * :156-172 -- the ice area is carried forward from the END of the
      previous step (`areai = new_ice_area`) and the two fractions derived
      from it. `fraci` is the ice's share of the lake's surface and is what
      `solve_lake` splits its energy balance on; `lakefrac` is the lake's
      share of the grid cell and is what `water_balance` moves. They are
      both taken before anything else happens, so they describe the lake as
      it was at the start of the step.
    * :435-447 -- the wetland's and the rest of the cell's runoff and
      baseflow, converted from millimetres over their own tiles to cubic
      metres, with `rpercent` of the non-lake tiles' share reaching the
      lake and the wetland's whole share reaching it.
    * :454-462 -- precipitation split into rain and snow at the air
      temperature, and handed to `solve_lake` in millimetres.
    * :462-482 -- solve_lake, then water_balance.

    `f` carries `prec` (mm over the lake's surface), `tair`, `wind`, `vp`
    and `vpd` and `pressure` in kPa, `shortin`, `longin`, `air_density`,
    and the cell's `sum_runoff`, `sum_baseflow`, `wetland_runoff`,
    `wetland_baseflow` in millimetres plus `channel_in`.
    """
    zero = torch.zeros_like(lake['volume'])
    sarea = lake['sarea']
    basin0 = lake_con['basin'][0]
    cell_area = soil['cell_area']

    # :156-172 -- the ice the previous step left, and the two fractions.
    areai = torch.clamp(lake['new_ice_area'], min=0.0)
    fraci = torch.where(sarea > 0.0,
                        torch.clamp(areai / torch.where(sarea > 0.0, sarea,
                                                        torch.ones_like(sarea)),
                                    max=1.0), zero)
    lakefrac = sarea / basin0

    # :435-447 -- what reaches the lake, in cubic metres.
    rp = lake_con['rpercent']
    runoff_in = ((f['sum_runoff'] * rp + f['wetland_runoff'])
                 * cell_area / MM_PER_M)
    baseflow_in = ((f['sum_baseflow'] * rp + f['wetland_baseflow'])
                   * cell_area / MM_PER_M)
    channel_in = f.get('channel_in', zero) * cell_area / MM_PER_M

    # :420-437 -- the precipitation split, in millimetres over the lake:
    # calc_rainonly.c:15-35 literally (rain over the ramp from the MIN_RAIN
    # end, snow the remainder), not the land's partition_precipitation,
    # which forms the same split the other way round and can differ in the
    # last bit. No gauge correction (CORRPREC is refused) and no trace-snow
    # cut: solve_lake is handed force->prec as it is.
    rain, snow = calc_rainonly(f['prec'], f['tair'],
                               f.get('t_max_snow', 0.5),
                               f.get('t_min_rain', -0.5))

    lk = dict(lake)
    lk.update(areai=areai, runoff_in=runoff_in, baseflow_in=baseflow_in,
              channel_in=channel_in,
              prec=f['prec'] * sarea / MM_PER_M)
    sf = dict(f)
    sf.update(snowfall=snow, rainfall=rain)
    lk, energy = solve_lake(lk, sf, soil, dt, wind_h, day_in_year, fraci)

    isave_n = int(lk['activenod'].max().item())
    new_lake, new_wetland, diag = water_balance(
        lk, wetland, soil, lake_con, dt, lakefrac, isave_n, carbon=carbon)
    diag.update(energy=energy, fraci=fraci, lakefrac=lakefrac,
                rainfall=rain, snowfall=snow, runoff_in=runoff_in,
                baseflow_in=baseflow_in, channel_in=channel_in)
    return new_lake, new_wetland, diag
