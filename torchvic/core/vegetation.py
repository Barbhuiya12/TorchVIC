"""Canopy resistance and Penman-Monteith, ported from VIC.

An earlier, vegetation-free version of this model measured the two most important
parameters for evapotranspiration in this model to be `et_beta` and `rs`, and
**both of them exist only because vegetation is not represented**. `rs` is one
number standing where VIC computes a canopy resistance that varies with
radiation, temperature, vapour pressure deficit, leaf area and soil moisture;
`et_beta` is an exponent standing in for root-weighted extraction. Measuring
the two stand-ins as the leading controls is the clearest possible argument
for replacing them with the thing they stand in for.

What is ported here, from vic/vic_run/src/penman.c:

    calc_rc   canopy resistance, Wigmosta et al. (1994) factors
    penman    Penman-Monteith evapotranspiration, VIC's exact algebra

VIC's own `penman` differs from the textbook form in ways that matter for
agreement: it uses CONST_CPMAIR rather than a rounded 1013, adds the
architectural resistance `rarc` in series with the canopy resistance, converts
with CONST_CDAY, and clips a negative evaporation to zero only when the
vapour pressure deficit is non-negative. The version previously in
`torchvic.model.basin` did none of those.

Not yet ported, and named so nobody assumes otherwise: the vegetation library
itself (monthly leaf area index, albedo, roughness and displacement per
class), the per-cell fractional cover and root distribution, and
`canopy_evap.c`'s partition of demand between intercepted water and
transpiration. This module is the resistance layer those will sit on.
"""
import torch

from . import atmos, version as core_version, snow, xmath
from .snow import CONST_CPMAIR, CONST_TKFRZ

# vic/drivers/shared_all/src/initialize_parameters.c
CANOPY_CLOSURE = 4000.0        # Pa, vpd at which the canopy shuts
CANOPY_RSMAX = 5000.0          # s/m, the resistance of a closed canopy
HUGE_RESIST = 1.0e20           # s/m, a canopy that is not conducting at all
CANOPY_VPDMINFACTOR = 0.1      # floor on the vpd factor
CONST_CDAY = 86400.0           # s/day


def calc_rc(rs, net_short, rgl, tair, vpd, lai, gsm_inv, ref_crop=False,
            co2factor=None):
    """Canopy resistance, s/m.  penman.c calc_rc.

    The C is a four-way branch. Two of those branches are structural -- a zero
    minimum resistance means no canopy resistance at all, and zero leaf area
    means a closed one -- and are reproduced with `where`. The reference-crop
    branch is a flag rather than data and is selected at trace time.

    `gsm_inv` is the soil-moisture stress factor, 1 when the soil is wet and
    falling towards 0 as it dries. It is the quantity `et_beta` was
    approximating.

    `co2factor` is the VIC-WUR 3.0 hook (vic_run/src/penman.c:83-87, option
    `vicwur_core.b_co2`): the canopy resistance factor
    1 / (1 + b_co2 (CO2 / CANOPY_CO2REF - 1)) of Li et al. (2019), built by
    `torchvic.vicwur.co2.co2factor`. It multiplies the denominator in the C's
    order and is not applied in the reference-crop branch. None (the default)
    leaves the VIC 5 expression untouched, bit for bit.
    """
    if ref_crop:
        rc = torch.clamp(rs / torch.clamp(lai * 0.5, min=1e-12),
                         max=CANOPY_RSMAX)
        return torch.where(rs == 0, torch.zeros_like(rc),
                           torch.where(lai == 0,
                                       torch.full_like(rc, CANOPY_RSMAX), rc))

    # Wigmosta et al. (1994) factors. Each is guarded exactly where the C
    # guards it and nowhere else.
    f = net_short / torch.clamp(rgl, min=1e-12)
    dayfactor = torch.where(
        rs > 0.0,
        (1.0 + f) / torch.clamp(f + rs / CANOPY_RSMAX, min=1e-12),
        torch.ones_like(f))

    tfactor = 0.08 * tair - 0.0016 * tair * tair
    tfactor = torch.where(tfactor <= 0.0,
                          torch.full_like(tfactor, 1e-10), tfactor)

    vpdfactor = 1.0 - vpd / CANOPY_CLOSURE
    vpdfactor = torch.where(vpdfactor < CANOPY_VPDMINFACTOR,
                            torch.full_like(vpdfactor, CANOPY_VPDMINFACTOR),
                            vpdfactor)

    if co2factor is None:
        denom = torch.clamp(lai * gsm_inv * tfactor * vpdfactor, min=1e-12)
    else:
        # penman.c:87. A non-positive factor (b_co2 (1 - CO2/REF) >= 1) gives
        # a negative or infinite rc in C; only an exact zero is guarded.
        denom = lai * gsm_inv * tfactor * vpdfactor * co2factor
        denom = torch.where(denom == 0.0, torch.full_like(denom, 1e-12),
                            denom)
    rc = torch.clamp(rs / denom * dayfactor, max=CANOPY_RSMAX)
    return torch.where(rs == 0, torch.zeros_like(rc),
                       torch.where(lai == 0,
                                   torch.full_like(rc, CANOPY_RSMAX), rc))


def penman(tair, elevation, rad, vpd, ra, rc, rarc):
    """Penman-Monteith evapotranspiration, mm/day.  penman.c penman.

    VIC's exact algebra, including the architectural resistance in series and
    the asymmetric clip: a negative evaporation is zeroed only where the
    vapour pressure deficit is non-negative, so condensation under a negative
    deficit is allowed through.
    """
    slope = atmos.svp_slope(tair)
    pz = atmos.air_pressure(tair, elevation)
    lv = atmos.latent_heat_of_vaporization(tair)
    gamma = 1628.6 * pz / lv
    r_air = 0.003486 * pz / (275.0 + tair)

    evap = ((slope * rad + r_air * CONST_CPMAIR * vpd / ra)
            / (lv * (slope + gamma * (1.0 + (rc + rarc) / ra)))
            * CONST_CDAY)
    return torch.where((vpd >= 0.0) & (evap < 0.0),
                       torch.zeros_like(evap), evap)


def soil_moisture_stress(avail_moist, w_crit, w_wilt):
    """VIC's transpiration restriction factor, gsm_inv.  canopy_evap.c.

        moist >= Wcr                 ->  1
        Wpwp <= moist < Wcr          ->  (moist - Wpwp) / (Wcr - Wpwp)
        moist <  Wpwp                ->  0

    This is the quantity `et_beta` was standing in for, and the difference is
    not cosmetic. `et_beta` raised a normalised availability to a fitted power,
    which is monotonic and smooth and has no physical anchor. VIC's version is
    piecewise linear between two measurable soil properties: the critical
    moisture below which a plant begins to close its stomata, and the wilting
    point below which it cannot extract water at all. A fitted exponent can
    mimic the curve; it cannot place the thresholds, and the thresholds are
    what carry to a basin that was never fitted.

    The denominator is guarded because a soil parameter set may give Wcr equal
    to Wpwp, which is degenerate rather than wet or dry; the guard keeps the
    gradient finite there instead of letting a zero divide poison the whole
    backward pass.
    """
    span = torch.clamp(w_crit - w_wilt, min=1e-9)
    ramp = torch.clamp((avail_moist - w_wilt) / span, min=0.0, max=1.0)
    return torch.where(avail_moist >= w_crit, torch.ones_like(ramp), ramp)


def root_fractions(root_depth, root_fract, layer_depth):
    """Fraction of a vegetation class's roots in each soil layer.

    `root_depth` and `root_fract` are the root zones from VIC's vegetation
    parameter file, each [zone]; `layer_depth` is the soil layer thickness,
    [layer]. All in metres, all broadcastable over cells.

    A root zone and a soil layer are two different partitions of the same
    column, so the fraction of roots in a layer is the overlap between them.
    VIC computes this once in calc_root_fractions.c; it is reproduced here as
    an overlap integral, which is the same quantity and is differentiable with
    respect to both partitions.
    """
    zone_top = torch.cumsum(root_depth, dim=-1) - root_depth
    zone_bot = zone_top + root_depth
    lay_top = torch.cumsum(layer_depth, dim=-1) - layer_depth
    lay_bot = lay_top + layer_depth
    # Roots below the soil column belong to the bottom layer, not to nowhere.
    # calc_root_fraction.c does this twice over: `ltmp = Nlayer - 1` caps the
    # layer index so anything deeper accumulates into the last one, and when
    # the loop runs past the column it sets `Lsum = Zsum` to "extend Lsum to
    # include all remaining roots within the final soil layer". Renormalising
    # instead spreads those roots over every layer in proportion to the ones
    # already there, which moves water uptake upward -- for Walnut Gulch
    # grassland, zones reach 1.6 m and the column stops at 1.4 m, so 0.2 m of
    # roots are at stake and the bottom layer's share falls from 0.69 to 0.67
    # while the middle layer's rises.
    # The layers may be per cell while the zones are per cell and per tile,
    # so bring both onto the common leading shape before touching the last
    # layer's floor -- otherwise the concatenation below sees [cell, 1, layer]
    # against [cell, tile, 1].
    lead = torch.broadcast_shapes(lay_bot.shape[:-1], zone_bot.shape[:-1])
    lay_top = lay_top.expand(*lead, lay_top.shape[-1])
    lay_bot = lay_bot.expand(*lead, lay_bot.shape[-1])
    deepest = zone_bot[..., -1:].expand(*lead, 1)
    lay_bot = torch.cat([lay_bot[..., :-1],
                         torch.maximum(lay_bot[..., -1:], deepest)], dim=-1)
    # [..., layer, zone] overlap
    overlap = (torch.minimum(lay_bot.unsqueeze(-1), zone_bot.unsqueeze(-2))
               - torch.maximum(lay_top.unsqueeze(-1), zone_top.unsqueeze(-2)))
    overlap = torch.clamp(overlap, min=0.0)
    share = overlap / torch.clamp(root_depth.unsqueeze(-2), min=1e-9)
    frac = (share * root_fract.unsqueeze(-2)).sum(-1)
    # VIC discards a layer holding less than 1e-4 of the roots before
    # rescaling (calc_root_fraction.c:165-168). Kept so the two agree exactly
    # rather than to four decimals.
    frac = torch.where(frac < 1e-4, torch.zeros_like(frac), frac)
    frac = frac / torch.clamp(frac.sum(-1, keepdim=True), min=1e-9)
    if any(isinstance(x, torch.Tensor) and x.requires_grad
           for x in (root_depth, root_fract, layer_depth)):
        return frac
    # Parameters, not a gradient: VIC's own walk, so the fractions carry
    # its rounding (the overlap form lands an ulp off on some tiles, e.g.
    # 0.3890909090909092 against VIC's 0.38909090909090915 on himal, and
    # that ulp reaches every transpiration).
    return _root_fractions_c(root_depth, root_fract, layer_depth, frac)


def _root_fractions_c(root_depth, root_fract, layer_depth, fallback):
    """calc_root_fraction.c 5.1.0:56-177 transcribed, per tile, in Python
    doubles; `fallback` (the overlap form) where the inputs are not finite
    or VIC itself would stop (fractions not summing to 1)."""
    lead = fallback.shape[:-1]
    nl = fallback.shape[-1]
    zd = torch.broadcast_to(root_depth, lead + root_depth.shape[-1:])
    zf = torch.broadcast_to(root_fract, lead + root_fract.shape[-1:])
    ld = torch.broadcast_to(layer_depth, lead + (nl,))
    nz = zd.shape[-1]
    rows = torch.cat([zd.detach().cpu().reshape(-1, nz).double(),
                      zf.detach().cpu().reshape(-1, nz).double(),
                      ld.detach().cpu().reshape(-1, nl).double()], 1)
    # The walk is a function of its inputs alone, so it runs once per
    # distinct row (by bit pattern: 0.0 and -0.0, or two NaNs, stay apart)
    # -- the ISC domain's 1,986,354 tiles have 125,160 distinct rows.
    import numpy as np
    a = np.ascontiguousarray(rows.numpy())
    key = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    inv = torch.from_numpy(inv.reshape(-1).astype(np.int64))
    u_l = a[first].tolist()
    res = torch.zeros(len(u_l), nl, dtype=torch.float64)
    found = torch.zeros(len(u_l), dtype=torch.bool)
    for k, u in enumerate(u_l):
        r = _root_walk(u[:nz], u[nz:2 * nz], u[2 * nz:])
        if r is not None:
            res[k] = torch.tensor(r, dtype=torch.float64)
            found[k] = True
    out = fallback.detach().cpu().reshape(-1, nl).clone()
    hit = found[inv]
    out[hit] = res[inv[hit]].to(out.dtype)
    return out.reshape(fallback.shape).to(fallback.device)


def _root_walk(zone_depth, zone_fract, depth):
    import math
    nz, nl = len(zone_depth), len(depth)
    if not all(math.isfinite(x) for x in zone_depth + zone_fract + depth):
        return None
    dum = 0.0
    for z in range(nz):
        dum += zone_fract[z]
    if abs(dum - 1.0) > 1e-4:
        return None
    dens = [zone_fract[z] / zone_depth[z] if zone_depth[z] > 0 else 1.0
            for z in range(nz)]
    root = [0.0] * nl
    layer = zone = 0
    Lsum = depth[0]
    Lsumprev = 0.0
    Zsum = zone_depth[0]
    Dsumprev = 0.0
    sum_fract = 0.0
    guard = 0
    while layer < nl or zone < nz:
        guard += 1
        if guard > 10 * (nl + nz) + 10:
            return None
        Dsum = Lsum if (Lsum <= Zsum and layer < nl) else Zsum
        dD = Dsum - Dsumprev
        Dsumprev = Dsum
        if zone < nz:
            sum_fract += dD * dens[zone]
        if (Lsum > Lsumprev and Dsum == Lsum
                and (layer < nl - 1
                     or (zone >= nz - 1 and Dsum == Zsum))):
            root[min(layer, nl - 1)] = sum_fract
            sum_fract = 0.0
        if Lsum < Zsum:
            layer += 1
            if layer < nl:
                Lsumprev = Lsum
                Lsum += depth[layer]
            else:
                Lsum = Zsum
        elif Lsum > Zsum:
            zone += 1
            if zone < nz:
                Zsum += zone_depth[zone]
            else:
                Zsum = Lsum
        else:
            layer += 1
            if layer < nl:
                Lsumprev = Lsum
                Lsum += depth[layer]
            zone += 1
            if zone < nz:
                Zsum += zone_depth[zone]
    dum = 0.0
    for i in range(nl):
        if root[i] < 1.e-4:
            root[i] = 0.0
        dum += root[i]
    if abs(dum - 1.0) > 1e-4:
        return None
    return [x / dum for x in root]


def root_fractions_w3(root_depth, root_fract, layer_depth):
    """Root fractions as VIC-WUR 3.0.0 computes them (core 'vicwur3').

    VIC-WUR 3.0's calc_root_fraction.c (<data root>/build/vicwur3/src_tree/
    vic/drivers/shared_all/src/calc_root_fraction.c:58-183) is VIC 5.0.1's zone
    walk with an iteration guard (:67-75) and one more line, `zone++` at :137:
    a root zone that reaches past the bottom of the current soil layer is left
    once its part in that layer is counted, so the part below is lost, not
    carried into the next layer, and the profile is then renormalised
    (:166-182). It is not the overlap (5.0.1, 5.1.0): R1 class 2, zones
    0.1/0.6/2.3 m over layers 0.252/6/6 m, is 0.6676/0.3324/0 here and
    0.4358/0.5642/0 there. A zone_fract summing to more than 1e-5 off 1 is
    first divided by its sum (VIC-WUR 3.0 vic_init.c:1259-1267). Per tile, in
    Python doubles, once per distinct row; returns [..., layer] on the inputs'
    device. A tile whose walk leaves no layer with 1e-4 of the roots
    (VIC-WUR 3.0 stops there, :173-178; the `zone++` can do it) comes back all
    zero, for the caller to refuse where the tile is vegetation.
    """
    import numpy as np
    lead = torch.broadcast_shapes(root_depth.shape[:-1],
                                  root_fract.shape[:-1],
                                  layer_depth.shape[:-1])
    nz, nl = root_depth.shape[-1], layer_depth.shape[-1]
    zd = torch.broadcast_to(root_depth, lead + (nz,))
    zf = torch.broadcast_to(root_fract, lead + (nz,))
    ld = torch.broadcast_to(layer_depth, lead + (nl,))
    rows = torch.cat([zd.detach().cpu().reshape(-1, nz).double(),
                      zf.detach().cpu().reshape(-1, nz).double(),
                      ld.detach().cpu().reshape(-1, nl).double()], 1)
    a = np.ascontiguousarray(rows.numpy())
    key = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    res = np.zeros((len(first), nl))
    for k, u in enumerate(a[first].tolist()):
        res[k] = _root_walk_w3(u[:nz], u[nz:2 * nz], u[2 * nz:])
    out = torch.from_numpy(res[inv.reshape(-1)]).reshape(lead + (nl,))
    return out.to(dtype=root_depth.dtype, device=root_depth.device)


def _root_walk_w3(zone_depth, zone_fract, depth):
    """VIC-WUR 3.0 calc_root_fraction.c:58-183 for one tile, in doubles. A tile
    whose walk leaves no roots (VIC-WUR 3.0 stops there, :173-178, for a
    vegetated class; the bare tile is never walked, :58) returns zeros."""
    import math
    nz, nl = len(zone_depth), len(depth)
    if not all(math.isfinite(x) for x in zone_depth + zone_fract + depth):
        return [0.0] * nl
    s = 0.0
    for z in range(nz):
        s += zone_fract[z]
    if s != 0.0 and abs(s - 1.0) > 1e-5:
        # VIC-WUR 3.0 vic_init.c:1259-1267: assert_close_double(sum, 1, 0,
        # AREA_SUM_ERROR_THRESH), 1e-5 in VIC-WUR 3.0
        # (vic_driver_shared_image.h:37); a rootless class (the bare one, never
        # walked) is left
        zone_fract = [f / s for f in zone_fract]

    import numpy as _np

    def interp(x, lx, ux, ly, uy):         # interpoloation.c:33-40
        # IEEE doubles as in C: a zero-depth zone divides by 0 there too
        with _np.errstate(all='ignore'):
            return float(_np.float64(x - lx) / _np.float64(ux - lx)
                         * (uy - ly) + ly)

    root = [0.0] * nl
    sum_fract = 0.0
    layer = 0
    Lstep = depth[0]
    Lsum = Lstep
    Zsum = 0.0
    zone = 0
    n_iter = 0
    while zone < nz:
        n_iter += 1
        if n_iter > 9999:   # MAX_ROOT_ITER, vic_driver_shared_all.h:48
            raise ValueError('calc_root_fraction (VIC-WUR 3.0): stuck in an infinite '
                             'loop (zones %r, layers %r)'
                             % (zone_depth, depth))
        Zstep = zone_depth[zone]
        if (Zsum + Zstep) <= Lsum and Zsum >= Lsum - Lstep:
            sum_fract += zone_fract[zone]                       # :78-80
        else:
            if Zsum < Lsum - Lstep:                             # :82-106
                Zmin_fract = interp(Lsum - Lstep, Zsum, Zsum + Zstep, 0,
                                    zone_fract[zone])
            else:
                Zmin_fract = 0.0
            Zmax = Zsum + Zstep if Zsum + Zstep <= Lsum else Lsum
            sum_fract += (interp(Zmax, Zsum, Zsum + Zstep, 0,
                                 zone_fract[zone]) - Zmin_fract)
        if Zsum + Zstep < Lsum:                                 # :108-112
            Zsum += Zstep
            zone += 1
        elif Zsum + Zstep == Lsum:                              # :113-135
            Zsum += Zstep
            zone += 1
            if layer < nl:
                root[layer] = sum_fract
                sum_fract = 0.0
            layer += 1
            if layer < nl:
                Lstep = depth[layer]
                Lsum += Lstep
            elif layer == nl and zone < nz:
                Zstep = zone_depth[zone]
                Lstep = Zsum + Zstep - Lsum
                for i in range(zone + 1, nz):
                    Lstep += zone_depth[i]
                Lsum += Lstep
        else:                                                   # :136-155
            # VIC-WUR 3.0 calc_root_fraction.c:137
            zone += 1
            if layer < nl:
                root[layer] = sum_fract
                sum_fract = 0.0
            layer += 1
            if layer < nl:
                Lstep = depth[layer]
                Lsum += Lstep
            elif layer == nl:
                Lstep = Zsum + Zstep - Lsum
                for i in range(zone + 1, nz):
                    Lstep += zone_depth[i]
                Lsum += Lstep
    if sum_fract > 0 and layer >= nl:                           # :159-164
        root[nl - 1] += sum_fract
    elif sum_fract > 0:
        root[layer] += sum_fract
    dum = 0.0
    for i in range(nl):                                         # :166-172
        if root[i] < 1.e-4:
            root[i] = 0.0
        dum += root[i]
    if dum == 0.0:
        return [0.0] * nl
    return [x / dum for x in root]                              # :180-182


#: options.SHARE_LAYER_MOIST, initialize_options.c:67 -- true unless the
#: global file says otherwise. Selects canopy_evap.c's CASE 1.
SHARE_LAYER_MOIST = True


def cellwise(x, ref):
    """A parameter given per cell broadcast against a [cell, tile] tensor.

    Calibration hands the model one value per cell (and, in an ensemble, one
    per member per cell); the library quantities it scales are per tile. A
    [cell] tensor aligns to the *last* axis under torch's rules, which is
    the tile axis, so it has to be unsqueezed. A scalar and an already
    per-tile tensor pass through.
    """
    if isinstance(x, torch.Tensor) and x.dim() == 1 and ref.dim() >= 2 \
            and x.shape[0] == ref.shape[0]:
        return x.reshape((-1,) + (1,) * (ref.dim() - 1))
    return x


def transpiration(avail_moist, w_crit, w_wilt, root_frac, lai, rmin, rgl,
                  net_short, tair, vpd, ra, rarc, rad, elevation,
                  share_layer_moist=SHARE_LAYER_MOIST, return_rc=False,
                  rc_ps=None, rs_sink=None, step=None, co2factor=None,
                  share_fix=None, stress=None):
    """Per-layer transpiration, mm/day.  canopy_evap.c transpiration().

    `co2factor` [tile], like lai, is the VIC-WUR b_co2 hook of both calc_rc
    calls (VIC-WUR 3.0 canopy_evap.c:302, :414); None leaves VIC 5 untouched.

    `share_fix` is VIC-WUR 3.0.0's CASE 1 (canopy_evap.c:341-372):
    `spare_transp +=`, so every dry layer's shortfall is handed on, and
    the hand-on only where root_sum > 0. None follows the core (on under
    'vicwur3' only, torchvic.core.version).

    `stress`, a dict, receives stress['gsm'] [..., layer]: the gsm_inv VIC-WUR
    stores in layer[i].water_stress -- 1 on every layer in CASE 1 (VIC-WUR 3.0
    canopy_evap.c:290-296), each layer's own in CASE 2 (:392-405), from the
    moisture CASE 2 sees (an upper layer without roots has 0, :234-247).

    With `step` = (delta_t, dryFrac) the result is per step instead, scaled
    where the C scales it: penman(...) * delta_t / CONST_CDAY * dryFrac,
    then the root fraction (canopy_evap.c 5.1.0:306-330, :402-405), which
    rounds differently from scaling the per-layer split afterwards.

    Every argument broadcasts against `root_frac`, whose last axis is the
    soil layer; the scalars (lai, rmin, rgl, forcing, ra, rarc) carry no
    layer axis. Returns [..., layer], before the dry-fraction scaling and
    the step scaling the C applies as `* delta_t / CONST_CDAY * dryFrac`.

    CASE 1 (SHARE_LAYER_MOIST, :112-120): transpiration is unhindered when
    the root-bearing upper layers combined hold at least their combined Wcr
    and so does the bottom layer, or when the side of the split holding at
    least half the roots does. One Penman at gsm_inv = 1; a layer at or
    above its Wcr takes `evap * root`, a drier one `evap * gsm_inv * root`,
    and the shortfall is handed to the wet layers in proportion to their
    roots. Two things the C does that are reproduced rather than tidied:
    `spare_evap` is *assigned* in the layer loop (:143), so only the last
    dry layer's shortfall is redistributed; and an upper layer with no roots
    has its available moisture set to 0 (:80), which makes it "dry" and
    resets spare_evap to 0 if it comes after a dry rooted layer.

    CASE 2 (:167-215): each layer's gsm_inv goes into calc_rc for its own
    canopy resistance and the layer takes `penman(rc_i) * root`; a layer at
    gsm_inv = 0 takes nothing (the `if (gsm_inv > 0.0)` is a branch).

    `return_rc` also returns the canopy resistance the function leaves in
    `veg_var->rc` -- one number per tile, the CASE 1 Penman's own resistance
    or CASE 2's series combination at :463-475. Nothing in the water balance
    reads it; the carbon step does, which is why it is returned rather than
    recomputed by a caller that would have to repeat the case test to get it
    right.

    `rc_ps` is RC_MODE = RC_PHOTO (canopy_evap.c:296-307 and :400-411): a
    callable `rc_ps(gsm_inv [...]) -> (rc [...], rsLayer [layer, ...])`
    (penman.c calc_rc_ps) that replaces calc_rc in both cases. CASE 2 then
    also aggregates the per-layer-canopy stomatal resistances,
    rsLayer = 1 / sum_i(1 / rsLayer_i) (:419-432, :446-460), and `rs_sink`
    (a dict) receives them as rs_sink['rs'], [canopy layer, ...], which is
    what the carbon step's 'rs' assimilation reads (surface_fluxes.c:737).

    Before this the function was CASE 2 alone. Against VIC's own year of
    Walnut Gulch at every step, transpiration was 1.00 of VIC's in January
    and 1.30 in June: in the dry season the bottom layer stays above Wcr,
    VIC takes CASE 1 and the dry upper layers transpire `evap * gsm * root`
    ~ 0, while CASE 2 let them transpire through the CANOPY_RSMAX cap --
    about 0.7 mm/day times their root fraction, which is the size of the
    excess.
    """
    if share_fix is None:
        share_fix = core_version.vicwur3()
    root = root_frac
    shape = torch.broadcast_shapes(avail_moist.shape, w_crit.shape,
                                   w_wilt.shape, root.shape)
    am = avail_moist.expand(shape)
    wc = w_crit.expand(shape)
    wp = w_wilt.expand(shape)
    root = root.expand(shape)
    lai_l = lai.unsqueeze(-1)

    # --- CASE 2, each layer on its own ------------------------------------
    gsm = soil_moisture_stress(am, wc, wp)
    # calc_rc divides by gsm and by LAI: the masked branch is fed finite
    # stand-ins so autograd does not meet 0 * inf (see stability_correction).
    safe_gsm = torch.where(gsm > 0.0, gsm, torch.ones_like(gsm))
    safe_lai = torch.where(lai > 0.0, lai, torch.ones_like(lai))
    if rc_ps is None:
        rc2 = calc_rc(rmin.unsqueeze(-1), net_short.unsqueeze(-1),
                      rgl.unsqueeze(-1), tair.unsqueeze(-1),
                      vpd.unsqueeze(-1), safe_lai.unsqueeze(-1), safe_gsm,
                      co2factor=(None if co2factor is None
                                 else co2factor.unsqueeze(-1)))
    else:
        _pairs = [rc_ps(safe_gsm[..., i]) for i in range(safe_gsm.shape[-1])]
        rc2 = torch.stack([a for a, _ in _pairs], dim=-1)
        rs2 = torch.stack([b for _, b in _pairs], dim=-1)  # [canopy, ..., layer]
    et2 = penman(tair.unsqueeze(-1), elevation.unsqueeze(-1),
                 rad.unsqueeze(-1), vpd.unsqueeze(-1), ra.unsqueeze(-1), rc2,
                 rarc.unsqueeze(-1))
    keep = (gsm > 0.0) & (lai_l > 0.0)
    if step is not None:
        _dt, _dry = step
        _sc = lambda x: x * _dt / CONST_CDAY * _dry.unsqueeze(-1)
    else:
        _sc = lambda x: x
    case2 = torch.where(keep, _sc(et2) * root, torch.zeros_like(et2))
    if return_rc:
        # :437-441 and :463-475 -- the layers' conductances in parallel,
        # inverted, then capped. A tile with no conducting layer gets
        # HUGE_RESIST, which the cap immediately pulls to CANOPY_RSMAX.
        gc = torch.where(keep & (rc2 > 0.0), 1.0 / torch.where(rc2 > 0.0, rc2,
                                                              torch.ones_like(rc2)),
                         torch.zeros_like(rc2)).sum(-1)
        rc_case2 = torch.where(gc > 0.0,
                               1.0 / torch.where(gc > 0.0, gc,
                                                 torch.ones_like(gc)),
                               torch.full_like(gc, HUGE_RESIST))
        rc_case2 = torch.clamp(rc_case2, max=CANOPY_RSMAX)
    if rc_ps is not None and rs_sink is not None:
        # :419-432, :446-460 -- `if rsLayer > 0: += 1/rsLayer else +=
        # HUGE_RESIST`, then the inverse, then the cap.
        _inv = torch.where(rs2 > 0.0,
                           1.0 / torch.where(rs2 > 0.0, rs2,
                                             torch.ones_like(rs2)),
                           torch.full_like(rs2, HUGE_RESIST))
        _gs = torch.where(keep.unsqueeze(0), _inv,
                          torch.zeros_like(_inv)).sum(-1)
        rs_case2 = torch.clamp(
            torch.where(_gs > 0.0, 1.0 / torch.where(_gs > 0.0, _gs,
                                                     torch.ones_like(_gs)),
                        torch.full_like(_gs, HUGE_RESIST)),
            max=CANOPY_RSMAX)
        rs_sink['rs'] = rs_case2
    if stress is not None:
        # VIC-WUR 3.0 canopy_evap.c:234-258: avail_moist of an unrooted upper
        # layer is 0, so CASE 2 stores its gsm_inv at 0 moisture
        _ur = root[..., :-1] > 0.0
        _az = torch.cat([torch.where(_ur, am[..., :-1],
                                     torch.zeros_like(am[..., :-1])),
                         am[..., -1:]], dim=-1)
        stress['gsm'] = soil_moisture_stress(_az, wc, wp)
    if not share_layer_moist:
        return (case2, rc_case2) if return_rc else case2

    # --- CASE 1, the layers share -----------------------------------------
    # :74-84: upper layers with roots pool their moisture and their Wcr;
    # those without count as holding nothing. :88-95: the bottom layer
    # always counts.
    upper_rooted = root[..., :-1] > 0.0
    avail_c = torch.cat([torch.where(upper_rooted, am[..., :-1],
                                     torch.zeros_like(am[..., :-1])),
                         am[..., -1:]], dim=-1)
    moist1 = avail_c[..., :-1].sum(-1)
    wcr1 = torch.where(upper_rooted, wc[..., :-1],
                       torch.zeros_like(wc[..., :-1])).sum(-1)
    moist2 = avail_c[..., -1]
    wcr2 = wc[..., -1]
    root2 = root[..., -1]
    case1 = (((moist1 >= wcr1) & (moist2 >= wcr2) & (wcr1 > 0.0))
             | ((moist1 >= wcr1) & ((1.0 - root2) >= 0.5))
             | ((moist2 >= wcr2) & (root2 >= 0.5)))

    if rc_ps is None:
        rc1 = calc_rc(rmin, net_short, rgl, tair, vpd, safe_lai,
                      torch.ones_like(safe_lai), co2factor=co2factor)
    else:
        rc1, rs1 = rc_ps(torch.ones_like(safe_lai))
    evap1 = _sc(penman(tair, elevation, rad, vpd, ra, rc1, rarc)
                .unsqueeze(-1))
    wet = avail_c >= wc
    # :131-139, VIC's own ramp rather than soil_moisture_stress, so that the
    # layer with avail == Wcr sits on the wet side exactly as the C's
    # `>=` puts it.
    span = torch.where(wc > wp, wc - wp, torch.ones_like(wc))
    ramp = torch.where(avail_c >= wp, (avail_c - wp) / span,
                       torch.zeros_like(avail_c))
    gsm_c = torch.where(wet, torch.ones_like(ramp), ramp)
    lev1 = torch.where(wet, evap1 * root, evap1 * gsm_c * root)
    root_sum = 1.0 - torch.where(wet, torch.zeros_like(root), root).sum(-1)
    spare = torch.zeros_like(root_sum)
    for i in range(root.shape[-1]):
        if share_fix:
            # VIC-WUR 3.0 canopy_evap.c:355: spare_transp += ...
            spare = torch.where(wet[..., i], spare,
                                spare + (evap1[..., 0] * root[..., i]
                                         * (1.0 - gsm_c[..., i])))
        else:
            spare = torch.where(wet[..., i], spare,
                                (evap1[..., 0] * root[..., i]
                                 * (1.0 - gsm_c[..., i])))
    safe_root_sum = torch.where(root_sum != 0.0, root_sum,
                                torch.ones_like(root_sum))
    share = (root * spare.unsqueeze(-1) / safe_root_sum.unsqueeze(-1))
    if share_fix:
        # VIC-WUR 3.0 canopy_evap.c:360-372: handed on only where root_sum > 0
        # (else `transp -= spare_transp`, which no layer reads)
        lev1 = lev1 + torch.where(
            wet & ((spare > 0.0) & (root_sum > 0.0)).unsqueeze(-1), share,
            torch.zeros_like(share))
    else:
        lev1 = lev1 + torch.where(wet & (spare > 0.0).unsqueeze(-1), share,
                                  torch.zeros_like(share))
    lev1 = torch.where(lai_l > 0.0, lev1, torch.zeros_like(lev1))
    out = torch.where(case1.unsqueeze(-1), lev1, case2)
    if stress is not None:
        stress['gsm'] = torch.where(case1.unsqueeze(-1),
                                    torch.ones_like(stress['gsm']),
                                    stress['gsm'])
    if rc_ps is not None and rs_sink is not None:
        rs_sink['rs'] = torch.where(case1.unsqueeze(0), rs1, rs_sink['rs'])
    if return_rc:
        return out, torch.where(case1, torch.clamp(rc1, max=CANOPY_RSMAX),
                                rc_case2)
    return out


# ---------------------------------------------------------------------------
# VIC's vegetation library and vegetation parameter file.
#
# These are the files a VIC user already has, in the format VIC already reads,
# so bringing an existing set-up across is a path and not a conversion. The
# repository ships a real pair for the Stehekin basin under
# samples/data/classic/Stehekin/parameters/, so the vegetation used here is the vegetation VIC itself was run
# with rather than a table invented for the purpose.
# ---------------------------------------------------------------------------
MONTHS = 12


class VegLibrary(object):
    """VIC's vegetation library: one record per land-cover class.

    Monthly leaf area index, albedo, roughness and displacement, plus the
    scalars the canopy needs: architectural resistance, minimum stomatal
    resistance, the radiation threshold for transpiration, and the attenuation
    coefficients.
    """

    FIELDS = ('overstory', 'rarc', 'rmin')
    MONTHLY = ('lai', 'albedo', 'roughness', 'displacement')
    TRAILING = ('wind_h', 'rgl', 'rad_atten', 'wind_atten', 'trunk_ratio')

    def __init__(self, classes):
        self.classes = classes            # {class_id: {field: value}}

    @classmethod
    def read(cls, path):
        out = {}
        for line in open(path):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            f = line.split()
            i = 0

            def take(n):
                nonlocal i
                v = [float(x) for x in f[i:i + n]]
                i += n
                return v
            cid = int(f[0]); i = 1
            rec = {'overstory': bool(take(1)[0]),
                   'rarc': take(1)[0], 'rmin': take(1)[0]}
            for m in cls.MONTHLY:
                rec[m] = take(MONTHS)
            for t in cls.TRAILING:
                rec[t] = take(1)[0]
            out[cid] = rec
        return cls(out)

    def tensor(self, field, class_ids, dtype, device, month=None):
        """`field` for the given classes as a tensor, monthly fields indexed."""
        import torch as _t
        vals = []
        for c in class_ids:
            v = self.classes[int(c)][field]
            vals.append(v[month] if isinstance(v, list) else float(v))
        return _t.tensor(vals, dtype=dtype, device=device)

    def __repr__(self):
        return ('<VegLibrary %d classes: %s>'
                % (len(self.classes), sorted(self.classes)))


class VegParam(object):
    """Per-cell vegetation tiles: class, fractional cover and root zones.

    A VIC cell is a mosaic. Each tile has its own land-cover class, the
    fraction of the cell it occupies, and its own root distribution, and the
    cell's flux is the area-weighted sum over tiles. That mosaic is the thing
    this model has been missing, and it is why a single surface resistance had
    to stand in for a library.
    """

    def __init__(self, cells):
        self.cells = cells                # {cellnum: [tile, ...]}

    @classmethod
    def read(cls, path, n_root_zones=3, lai_from_param=True):
        cells = {}
        toks = []
        for line in open(path):
            if line.strip() and not line.startswith('#'):
                toks.extend(line.split())
        i = 0
        while i < len(toks):
            cellnum = int(toks[i]); nveg = int(toks[i + 1]); i += 2
            tiles = []
            for _ in range(nveg):
                veg_class = int(toks[i]); cv = float(toks[i + 1]); i += 2
                depths, fracts = [], []
                for _ in range(n_root_zones):
                    depths.append(float(toks[i]))
                    fracts.append(float(toks[i + 1]))
                    i += 2
                tile = {'veg_class': veg_class, 'Cv': cv,
                        'root_depth': depths, 'root_fract': fracts}
                if lai_from_param:
                    tile['lai'] = [float(x) for x in toks[i:i + MONTHS]]
                    i += MONTHS
                tiles.append(tile)
            cells[cellnum] = tiles
        return cls(cells)

    def coverage(self, cellnum):
        """Sum of tile fractions. VIC allows it below 1; the remainder is bare."""
        return sum(t['Cv'] for t in self.cells[cellnum])

    def __repr__(self):
        n = len(self.cells)
        tiles = sum(len(v) for v in self.cells.values())
        return ('<VegParam %d cells, %d tiles, %.2f tiles per cell>'
                % (n, tiles, tiles / max(n, 1)))


class VegMosaic(object):
    """A cell's vegetation tiles as padded tensors, ready for the basin loop.

    A cell is a mosaic of tiles and its flux is the area-weighted sum over
    them. Tiles are padded to a common count with zero cover so the whole
    domain is one [cell, tile] problem rather than a Python loop over a ragged
    structure; a padded tile contributes nothing because its Cv is zero, and
    nothing downstream needs to know it is there.

    Everything is a tensor, so leaf area, root depths, stomatal resistance and
    fractional cover can all be varied. That is the point: these are the
    quantities `rs` and `et_beta` were compressing into two numbers.
    """

    #: True when the five vegetation histories (lai, albedo, fcanopy,
    #: displacement, veg_rough) are [cell, tile, step] rather than
    #: [cell, tile, 12]: a *_SRC FROM_VEGHIST run (torchvic.io.veghist),
    #: where VIC sets them every model step (vic_force.c:264-363,
    #: update_step_vars.c:35-44). The step loop then indexes them by step.
    step_axis = False

    def __init__(self, cv, lai, rmin, rgl, rarc, root_frac, dtype, device,
                 veg_rough=None, displacement=None, albedo=None,
                 wind_h=None, rad_atten=None, overstory=None,
                 wind_atten=None, trunk_ratio=None, fcanopy=None,
                 sigma_slope=None, lag_one=None, fetch=None,
                 carbon=None, bare_tile=None, max_snow_albedo=None):
        self.cv = cv                  # [cell, tile]
        self.lai = lai                # [cell, tile, 12]
        self.rmin = rmin              # [cell, tile]
        self.rgl = rgl                # [cell, tile]
        self.rarc = rarc              # [cell, tile]
        self.root_frac = root_frac    # [cell, tile, layer]
        # Monthly, [cell, tile, 12]. The parameter file carries all three and
        # they were read and unused: the aerodynamic resistance was built from
        # one snow roughness for every surface, and the albedo was a single
        # constant for the whole domain.
        self.veg_rough = veg_rough
        self.displacement = displacement
        self.albedo = albedo
        # Monthly fractional canopy cover, [cell, tile, 12], or None when
        # the global file's FCAN_SRC is FROM_DEFAULT. VIC runs the whole
        # canopy on a per-plant basis when this is below 1: vic_run.c
        # :136-140 divides LAI, Wdew and the canopy snow by it on the way
        # in, :222-225 builds surf_atten from the divided LAI, and
        # solve_snow.c:255-277 scales every canopy output back to the
        # whole tile on the way out.
        self.fcanopy = fcanopy
        # [cell, tile] bool, or None: the classic driver's bare-soil tile,
        # vic_run's iveg == Nveg, which has no vegetation at all (VEG false,
        # calc_surf_energy_bal.c:165-175): no interception, no transpiration,
        # the whole tile ARNO bare soil (func_surf_energy_bal.c:797-805).
        # image_run.build_mosaic marks it; the image driver has none.
        self.bare_tile = bare_tile
        # BLOWING's three per-tile fields, [cell, tile], or None when the
        # file was not read with the option on: the slope's standard
        # deviation, the lag-one autocorrelation of the topography and
        # the fetch (read_vegparam.c:444-456).
        self.sigma_slope = sigma_slope
        self.lag_one = lag_one
        self.fetch = fetch
        # MAX_SNOW_ALBEDO's per-class fresh-snow albedo, [cell, tile], or
        # None when the option is off (vic_init.c:186-196); basin.simulate
        # uses it on every tile but the bare-soil one (surface_fluxes.c:524).
        self.max_snow_albedo = max_snow_albedo
        # CARBON's per-class vegetation-library fields, as a dict of
        # [cell, tile] tensors, or None when the library was not read
        # with VEGLIB_PHOTO on: Ctype, MaxCarboxRate, MaxETransport,
        # CO2Specificity, LightUseEff, NscaleFlag, Wnpp_inhib and
        # NPPfactor_sat (read_veglib.c:162-192). A dict rather than
        # eight attributes because they arrive and leave together.
        self.carbon = carbon
        # Per class, [cell, tile]. Both are library scalars VIC uses in the
        # energy balance and nowhere else: wind_h sets the reference height
        # of the wind profile (vic_run.c:259-264 -- the *library's*, not the
        # global file's WIND_H) and rad_atten sets the surface attenuation
        # that scales the ground flux (vic_run.c:222-225).
        self.wind_h = wind_h
        self.rad_atten = rad_atten
        # [cell, tile], 0 or 1. Not used by any physics here yet -- it is
        # carried so the surface energy balance can refuse a tile it would
        # otherwise solve on the wrong branch (CalcAerodynamic.c:83-160 and
        # the canopy solve are the overstory path, and neither exists here).
        self.overstory = overstory
        # CalcAerodynamic.c's overstory branch: the canopy wind attenuation
        # and the trunk-space ratio (veg library wind_atten, trunk_ratio).
        self.wind_atten = wind_atten
        self.trunk_ratio = trunk_ratio
        self.dtype, self.device = dtype, device

    @property
    def n_tiles(self):
        return int(self.cv.shape[1])

    @classmethod
    def uniform(cls, library, vegparam, cellnum, n_cells, layer_depth,
                dtype, device):
        """One mosaic, repeated over every cell.

        For a domain with no per-cell land cover: every cell is given the
        same tile set, taken from a real VIC cell rather than
        invented. This is a weaker claim than per-cell cover and is not the
        same thing -- it captures the seasonality of leaf area and the spread
        of stomatal resistance across classes, and not the spatial pattern of
        vegetation. Replacing it means supplying a mosaic per cell; nothing
        else here changes.
        """
        import torch as _t
        tiles = vegparam.cells[cellnum]
        nt = len(tiles)
        cv = _t.tensor([[t['Cv'] for t in tiles]], dtype=dtype, device=device)
        lai = _t.tensor([[t['lai'] for t in tiles]], dtype=dtype,
                        device=device)
        rmin = _t.tensor([[library.classes[t['veg_class']]['rmin']
                           for t in tiles]], dtype=dtype, device=device)
        rgl = _t.tensor([[library.classes[t['veg_class']]['rgl']
                          for t in tiles]], dtype=dtype, device=device)
        rarc = _t.tensor([[library.classes[t['veg_class']]['rarc']
                           for t in tiles]], dtype=dtype, device=device)
        rd = _t.tensor([[t['root_depth'] for t in tiles]], dtype=dtype,
                       device=device)
        rf = _t.tensor([[t['root_fract'] for t in tiles]], dtype=dtype,
                       device=device)
        ld = layer_depth.reshape(1, 1, -1).expand(1, nt, -1)
        root_frac = root_fractions(rd, rf, ld)
        rep = (n_cells, -1, -1)
        return cls(cv.expand(n_cells, -1), lai.expand(*rep),
                   rmin.expand(n_cells, -1), rgl.expand(n_cells, -1),
                   rarc.expand(n_cells, -1), root_frac.expand(*rep),
                   dtype, device)

    def select(self, cell_index):
        """The mosaic restricted to some cells, for a minibatch view.

        Epoch-wise training draws a few basins at a time, and the mosaic has
        to follow the same cells or a tile will be applied to the wrong
        column.
        """
        def sub(x):
            return None if x is None else x[cell_index]

        out = VegMosaic(self.cv[cell_index], self.lai[cell_index],
                         self.rmin[cell_index], self.rgl[cell_index],
                         self.rarc[cell_index], self.root_frac[cell_index],
                         self.dtype, self.device,
                         veg_rough=sub(self.veg_rough),
                         displacement=sub(self.displacement),
                         albedo=sub(self.albedo),
                         wind_h=sub(self.wind_h),
                         rad_atten=sub(self.rad_atten),
                         fcanopy=sub(self.fcanopy),
                         bare_tile=sub(self.bare_tile),
                         sigma_slope=sub(self.sigma_slope),
                         lag_one=sub(self.lag_one),
                         fetch=sub(self.fetch),
                         max_snow_albedo=sub(self.max_snow_albedo),
                         carbon=(None if self.carbon is None else
                                 {k: sub(v) for k, v in self.carbon.items()}),
                         overstory=sub(self.overstory),
                         wind_atten=sub(self.wind_atten),
                         trunk_ratio=sub(self.trunk_ratio))
        out.step_axis = self.step_axis
        return out

    def tile(self, n):
        """The mosaic with every cell repeated `n` times member-major, for
        an ensemble of parameter sets (see torchvic.case.Case.ensemble)."""
        import torch as _t

        def rep(x):
            if isinstance(x, _t.Tensor) and x.dim() >= 1:
                return x.repeat((n,) + (1,) * (x.dim() - 1))
            return x
        out = VegMosaic(rep(self.cv), rep(self.lai), rep(self.rmin),
                        rep(self.rgl), rep(self.rarc), rep(self.root_frac),
                        self.dtype, self.device,
                        veg_rough=rep(self.veg_rough),
                        displacement=rep(self.displacement),
                        albedo=rep(self.albedo), wind_h=rep(self.wind_h),
                        rad_atten=rep(self.rad_atten),
                        fcanopy=rep(self.fcanopy),
                        bare_tile=rep(self.bare_tile),
                        sigma_slope=rep(self.sigma_slope),
                        lag_one=rep(self.lag_one),
                        fetch=rep(self.fetch),
                        max_snow_albedo=rep(self.max_snow_albedo),
                        carbon=(None if self.carbon is None else
                                {k: rep(v) for k, v in self.carbon.items()}),
                        overstory=rep(self.overstory),
                        wind_atten=rep(self.wind_atten),
                        trunk_ratio=rep(self.trunk_ratio))
        out.step_axis = self.step_axis
        return out

    def initial_wdew(self):
        """Canopy storage at the start of a run: dry."""
        import torch as _t
        return _t.zeros_like(self.cv)

    def step(self, month, wdew, rain, avail_moist, w_crit, w_wilt,
             net_short, tair, vpd, ra, rad, elevation,
             lai_scale=1.0, rmin_scale=1.0, dt=CONST_CDAY,
             wind=None, shortwave=None, longwave=None, per_tile=False,
             return_rc=False, share_layer_moist=SHARE_LAYER_MOIST,
             lai_water_factor=None, ground_heat=None, ra_over=None,
             over=None, soil_rough=None, snowing=None, wilting_clamp=False,
             exact_wetness=False, bare_soil=None, atten_one=None,
             wdew_per_plant=False, ice=None):
        """One step of the canopy: interception, then transpiration.

        Returns (transp_per_layer, canopy_evap, throughfall, new_wdew), all
        area-weighted over tiles except `new_wdew`, which stays per tile
        because it is state. With `per_tile` the first three come back per
        tile, unweighted, for a caller that keeps a soil column per tile.

        The order is VIC's and it matters. Rain reaches the canopy first, some
        is held and evaporates at the potential rate with no stomatal
        resistance, and only the rest falls through to the ground. What is
        left of the leaf surface -- the dry fraction -- is what transpires.
        Running transpiration on the whole canopy while also evaporating
        intercepted water counts the same leaf twice.

        The keywords after `lai_water_factor` are what VIC's water-balance
        mode does and the defaults leave out; basin.simulate passes all of
        them on its water-balance path:

        ground_heat    [cell, tile] W/m2, grnd_flux + deltaH at Tsurf = Tair,
                       added to NetBareRad (func_surf_energy_bal.c:663-664);
                       energy.water_balance_ground_flux forms it.
        ra_over, over  overstory tiles hand canopy_evap Ra_veg[1], the
                       canopy resistance of CalcAerodynamic's overstory
                       branch (:120-124), not the no-overstory formula.
        soil_rough     [cell]; a tile with zero roughness takes the soil's
                       (vic_run.c:245-247).
        snowing        [cell, tile] bool, snow->snow: with a pack, falling
                       snow or (overstory) canopy snow the tile neither
                       evaporates nor transpires and its canopy store is
                       untouched (func_surf_energy_bal.c:756, :798, :807-809);
                       what reaches it (`rain`: melt or rain) goes to the
                       soil as it is (calc_surf_energy_bal.c:605-612).
        wilting_clamp  canopy_evap.c:500-524: no layer transpires below its
                       wilting point.
        exact_wetness  canopy_wetness(exact=True): an empty canopy is dry.
        bare_soil      dict(moist, max_moist, b_infilt, depth, resid_frac,
                       ra) for the (1 - fcanopy) bare fraction of each tile
                       (func_surf_energy_bal.c:762-791); `ra` is Ra_used[0].
                       When given, a fifth element -- that bare-soil
                       evaporation, mm per step per tile, zero without a
                       fractional canopy -- follows `new_wdew`.
        atten_one      [cell, tile] bool, surf_atten is 1 for the bare
                       fraction's radiation (solve_snow.c:154-156).
        wdew_per_plant 5.1.0: `wdew` is read as per-plant water as it
                       stands. vic_run.c:219-225 divides by fcanopy once per
                       model step and every snow step multiplies back
                       (solve_snow.c:251, func_surf_energy_bal.c:760), so a
                       later snow step of a model step starts from the
                       tile-unit store.
        ice            [cell, tile, layer] (or [cell, layer]) soil ice, mm,
                       or None. canopy_evap.c:201-241 transpires the liquid,
                       layer moist - ice, and :478-508 caps no layer below
                       its wilting point (the unfrozen moisture where the
                       ice is at or above it). As evap_per_tile's.

        With a fractional canopy (FCAN_SRC other than FROM_DEFAULT) the
        canopy runs per plant as vic_run.c:136-140 sets it up: LAI and Wdew
        divided by fcanopy (so Wdmax is per plant too), every canopy output
        multiplied back by it, the bare fraction's rain passing straight
        through (func_surf_energy_bal.c:785-790). With fcanopy exactly 1
        each of those is an exact identity in floating point.
        """
        import torch as _t
        _vw = getattr(self, 'vicwur', None)
        take = None
        if _vw is not None and _vw.get('canopy_take') is not None:
            # vicwur_core canopy_crop: the crop's evaporation of the store it
            # was handed (VIC_land_unit_simulation.cpp:435-446: Wdew goes to
            # what is left, canopyevap to what was used)
            take = _t.minimum(_vw['canopy_take'].expand_as(wdew).clamp(min=0.0),
                              wdew)
            wdew = wdew - take
        lai = self.lai[:, :, month] * cellwise(lai_scale, self.cv)
        # A quantity the cell supplies gets a tile axis; one that is
        # already per tile -- the melt from a pack per tile, say -- is
        # used as it is.
        c = lambda x: x.unsqueeze(1) if x.dim() == 1 else x
        fc = None if self.fcanopy is None else self.fcanopy[:, :, month]
        v51 = core_version.at_least('5.1.0')
        veg = None
        if fc is not None and v51:
            # vic_run.c 5.1.0:219-226: LAI and Wdew are divided by fcanopy
            # only `if (veg_var->fcanopy > 0)` -- vic_force.c 5.1.0:419-439
            # sets a leafless tile's fcanopy to 0 -- and for the bare-soil
            # tile too. Divided by 1 where fcanopy is 0 (LAI is 0 there and
            # Wdew stays as it is, as in the C), so no 0/0 reaches either
            # branch of a where below.
            fc_div = _t.where(fc > 0.0, fc, _t.ones_like(fc))
            lai_p = lai / fc_div
            wdew_p = wdew if wdew_per_plant else wdew / fc_div
            # calc_surf_energy_bal.c 5.1.0:145-155: VEG is false on the
            # bare-soil tile and wherever fcanopy or the (per-plant) LAI is
            # 0; in 5.0.1 (:165-175) fcanopy > 0 alone decided it.
            veg = (fc > 0.0) & (lai_p > 0.0)
            if self.bare_tile is not None:
                veg = veg & ~self.bare_tile
        else:
            lai_p = lai if fc is None else lai / fc
            wdew_p = wdew if fc is None else wdew / fc

        # Per-tile aerodynamic resistance from the tile's own roughness and
        # displacement, when the file supplies them and a wind is given.
        # Otherwise the caller's single `ra`, which is what this did for every
        # surface: one snow roughness, 70.5 s/m against VIC's 13.7 for shrub
        # and 40.7 for grass at this site's mean wind.
        if wind is not None and self.veg_rough is not None:
            z0 = self.veg_rough[:, :, month]
            if soil_rough is not None:
                z0 = _t.where(z0 > 0.0, z0, c(soil_rough).expand_as(z0))
            ra_t = aerodynamic_resistance(z0, self.displacement[:, :, month],
                                          c(wind))
        else:
            ra_t = c(ra)
        if ra_over is not None:
            # func_surf_energy_bal.c:757-761 hands canopy_evap Ra_veg[1];
            # on an overstory tile that is the canopy resistance, which for
            # the India forest classes is 1.58-1.72 times the no-overstory
            # value at 1 m/s -- the energy path already takes it.
            ra_t = _t.where(over, ra_over, ra_t)

        # Per-tile net radiation from the tile's own albedo. VIC hands
        # canopy_evap `NetShortBare`, which is the incoming shortwave with the
        # albedo already taken off (calc_surf_energy_bal.c:215), and
        # `NetBareRad`, the net radiation. Passing the *incoming* shortwave as
        # net_short gave calc_rc about 23 per cent more radiation than VIC
        # sees, which lowers the canopy resistance and raises transpiration.
        if shortwave is not None and self.albedo is not None:
            alb = self.albedo[:, :, month]
            ns_t = c(shortwave) * (1.0 - alb)
            if longwave is not None:
                rad_t = (ns_t + c(longwave)
                         - SURFACE_EMISSIVITY * snow.CONST_STEBOL
                         * (c(tair) + CONST_TKFRZ) ** 4)
            else:
                rad_t = c(rad)
        else:
            ns_t, rad_t = c(net_short), c(rad)
        if ground_heat is not None:
            # func_surf_energy_bal.c:663-664: NetBareRad is net radiation
            # PLUS the ground heat flux and the change in heat storage of
            # the top layer, and it is what drives canopy_evap and
            # transpiration (:757) in water-balance mode too, where Tsurf =
            # Tair (calc_surf_energy_bal.c:542-545) but QUICK_FLUX still
            # runs. Leaving it out was +7.5% transpiration and -11.7%
            # canopy evaporation against VIC on basin 37, hourly, 2001: by
            # day the ground takes up to 95 W/m2, by night it gives it back.
            rad_t = rad_t + ground_heat

        ce, new_wdew, tf, dry = canopy_interception(
            wdew_p, c(rain), lai_p, self.rgl, ns_t, c(tair), c(vpd),
            ra_t, self.rarc, rad_t, c(elevation), dt=dt,
            lai_water_factor=_lwf(lai_water_factor),
            exact_wetness=exact_wetness,
            **({} if _vw is None or _vw.get('canopy_mask') is None
               else dict(no_evap=_vw['canopy_mask'])))

        # canopy_evap.c 5.1.0:201-241: the moisture transpiration sees is
        # the liquid, layer moist - ice (the clamp below keeps the totals).
        # As evap_per_tile's am_liq.
        am_liq = avail_moist if ice is None else avail_moist - ice
        transp = self._transpiration(month, am_liq, w_crit, w_wilt,
                                     ns_t, tair, vpd, ra_t, rad_t, elevation,
                                     lai_scale, rmin_scale,
                                     return_rc=return_rc,
                                     share_layer_moist=share_layer_moist,
                                     lai=None if fc is None else lai_p,
                                     step=(dt, dry))
        rc = None
        if return_rc:
            transp, rc = transp
        # penman returns a rate per day, and canopy_evap.c scales it to the
        # step exactly as it scales the canopy term:
        #     layertransp[i] = penman(...) * delta_t / CONST_CDAY
        #                      * dryFrac * root[i]
        # `dt` reached the canopy here and never reached transpiration. On a
        # daily step the factor is 1, so the omission was invisible for as
        # long as this model only ever stepped daily; at eight steps a day it
        # makes transpiration eight times too large per step. It surfaced the
        # first time the model ran sub-daily -- 442 mm against VIC's 311 --
        # and nothing in the water balance could have caught it, because
        # over-transpiring conserves mass perfectly. (`step` above applies
        # it inside the case split, in the C's order.)
        if wilting_clamp:
            # canopy_evap.c:500-524, after the case split and the dryFrac
            # and step scaling: no layer gives up water below its wilting
            # point (with FROZEN_SOIL off there is no ice, so the C's
            # `layer[i].moist - Wpwp[i]` branch), and a negative result is
            # zero. CASE 1 hands a dry layer's shortfall to the wet ones and
            # CASE 2 can run a layer near Wpwp at a gsm_inv of a few per
            # cent of a large Penman, so the cap binds: 835 tile-layer-days,
            # 0.156% of the transpiration, basin 37, May-October 2002, daily.
            # With ice (:478-508): where the layer's ice is at or above
            # Wpwp the cap is the unfrozen moisture instead. As
            # evap_per_tile's.
            am = (avail_moist.unsqueeze(1) if avail_moist.dim() == 2
                  else avail_moist)
            wp = w_wilt.unsqueeze(1) if w_wilt.dim() == 2 else w_wilt
            cap = am - wp
            if ice is not None:
                ic = ice.unsqueeze(1) if ice.dim() == 2 else ice
                cap = _t.where((ic > 0) & (ic >= wp), am - ic, cap)
            transp = _t.clamp(_t.minimum(transp, cap), min=0.0)
        bare = None
        if veg is not None:
            # VIC 5.1.0's evapotranspiration, func_surf_energy_bal.c 5.1.0
            # :722-761, one rule for every tile that is not snowing:
            #     if (VEG) { Evap = fc * canopy_evap(...); transp *= fc;
            #                SurfRad = surf_atten * NetBareRad; }
            #     else       SurfRad = NetBareRad;
            #     Evap += (1 - fc) * arno_evap(..., SurfRad, Ra_used[0], ...);
            #     esoil *= (1 - fc);
            #     throughfall = (1 - fc) * rainfall + fc * throughfall;
            #     canopyevap *= fc;  Wdew *= fc;
            # On a VEG tile this is 5.0.1's 0 < fcanopy < 1 branch (and, at
            # fcanopy 1, its closed-canopy branch exactly: the bare term is
            # 0 * finite). A tile that is not VEG runs no canopy at all:
            # nothing intercepted or transpired, its rain straight through
            # (fcanopy 0), its Wdew times fcanopy -- deleted, not drained,
            # when a month's LAI goes to 0 -- and (1 - fcanopy) of it ARNO
            # bare soil at the unattenuated NetBareRad. That includes the
            # bare-soil tile, which 5.0.1 evaporated whole (:797-805) and
            # 5.1.0 weights by (1 - fcanopy) like any other.
            f3 = fc.unsqueeze(-1)
            z = _t.zeros_like(ce)
            transp = _t.where(veg.unsqueeze(-1), f3 * transp,
                              _t.zeros_like(transp))
            ce = _t.where(veg, fc * ce, z)
            _vwt = getattr(self, 'vicwur', None)
            tf_nv = c(rain).expand_as(tf)
            if _vwt is not None and _vwt.get('w3_tf'):
                # torchvic.vicwur.core throughfall (VIC-WUR 3.0): a vegetated
                # tile that is not VEG but has fcanopy > 0 (VIC-WUR 3.0 floors
                # it at MIN_FCANOPY, vic_force.c:428-441) passes (1 - fc) rain
                # + fc veg_var->throughfall (func_surf_energy_bal.c:785-786),
                # and that throughfall is the step's copy, zeroed at
                # surface_fluxes.c:267: the fc share of the rain is lost. ppt
                # is veg_var->throughfall for iveg < Nveg
                # (calc_surf_energy_bal.c:609-611); the bare tile takes its
                # rain.
                tf_w3 = (1.0 - fc) * c(rain)
                if self.bare_tile is not None:
                    tf_w3 = _t.where(self.bare_tile, tf_nv, tf_w3)
                tf_nv = tf_w3
            tf = _t.where(veg, (1.0 - fc) * c(rain) + fc * tf, tf_nv)
            new_wdew = _t.where(veg, fc * new_wdew, fc * wdew_p)
            if bare_soil is not None:
                from . import column as _column
                from .energy import surface_attenuation
                atten = surface_attenuation(
                    lai_p, (self.rad_atten if self.rad_atten is not None
                            else 0.5), fc)
                atten = _t.where(veg, atten, _t.ones_like(atten))
                if atten_one is not None:
                    atten = _t.where(atten_one, _t.ones_like(atten), atten)
                b = bare_soil
                bare = (1.0 - fc) * _column.bare_soil_evap(
                    penman(c(tair), c(elevation), atten * rad_t, c(vpd),
                           b['ra'], _t.zeros_like(b['ra']),
                           _column.SOIL_RARC) * dt / CONST_CDAY,
                    b['moist'], b['max_moist'], b['b_infilt'], b['depth'],
                    b['resid_frac'], resid_mm=b.get('resid_mm'))
        elif fc is not None:
            # func_surf_energy_bal.c:762-791 with fcanopy < 1: the plants'
            # fluxes scaled back to the tile, the rest of the tile's rain
            # falling straight through, and the tile's Wdew back in tile
            # units (vic_run.c:136-140 divided it on the way in).
            f3 = fc.unsqueeze(-1)
            transp = f3 * transp
            ce = fc * ce
            tf = (1.0 - fc) * c(rain) + fc * tf
            new_wdew = fc * new_wdew
            if bare_soil is not None:
                # :768-772: ARNO evaporation from the gaps, driven by
                # surf_atten * NetBareRad and Ra_used[0], weighted by the
                # bare fraction. It draws on the top layer only
                # (arno_evap.c:192).
                from . import column as _column
                from .energy import surface_attenuation
                atten = surface_attenuation(
                    lai_p, (self.rad_atten if self.rad_atten is not None
                            else 0.5), fc)
                if atten_one is not None:
                    atten = _t.where(atten_one, _t.ones_like(atten), atten)
                b = bare_soil
                bare = (1.0 - fc) * _column.bare_soil_evap(
                    penman(c(tair), c(elevation), atten * rad_t, c(vpd),
                           b['ra'], _t.zeros_like(b['ra']),
                           _column.SOIL_RARC) * dt / CONST_CDAY,
                    b['moist'], b['max_moist'], b['b_infilt'], b['depth'],
                    b['resid_frac'])
        if bare_soil is not None and bare is None:
            bare = _t.zeros_like(ce)
        if (bare_soil is not None and self.bare_tile is not None
                and veg is None):
            # (VIC 5.1.0's rule above already covers this tile.)
            # The classic driver's bare-soil tile, VEG false: no canopy
            # (calc_surf_energy_bal.c:165-175 -- func_surf_energy_bal.c:756
            # needs VEG), so nothing is intercepted, transpired or held,
            # and :797-805 evaporates the whole tile, not (1 - fcanopy) of
            # it, as ARNO bare soil with NetBareRad unattenuated and
            # Ra_used[0] (which :703 still averages with its MIN_FCANOPY).
            # Weighting it (1 - MIN_FCANOPY) instead left 1e-4 of the
            # tile's evaporation out and let a leafless canopy with
            # rmin 0 (read_veglib.c:215) transpire 1e-4 of the Penman.
            from . import column as _column
            vegless = self.bare_tile
            b = bare_soil
            whole = _column.bare_soil_evap(
                penman(c(tair), c(elevation), rad_t, c(vpd), b['ra'],
                       _t.zeros_like(b['ra']), _column.SOIL_RARC)
                * dt / CONST_CDAY,
                b['moist'], b['max_moist'], b['b_infilt'], b['depth'],
                b['resid_frac'])
            bare = _t.where(vegless, whole, bare)
            transp = _t.where(vegless.unsqueeze(-1), _t.zeros_like(transp),
                              transp)
            ce = _t.where(vegless, _t.zeros_like(ce), ce)
            tf = _t.where(vegless, c(rain).expand_as(tf), tf)
            new_wdew = _t.where(vegless, wdew.expand_as(new_wdew), new_wdew)
        if snowing is not None:
            # func_surf_energy_bal.c:756 / :798 / :807-809: `!SNOWING` gates
            # both canopy_evap and arno_evap, so a tile under snow -- a pack,
            # snow falling, or an overstory holding snow (solve_snow.c
            # :164-171) -- evaporates nothing, transpires nothing, leaves
            # its canopy store as it was, and passes what reached it to the
            # soil (calc_surf_energy_bal.c:605-612 skips the throughfall
            # assignment; solve_snow's ppt is the pack's outflow).
            z = _t.zeros_like(ce)
            transp = _t.where(snowing.unsqueeze(-1), _t.zeros_like(transp),
                              transp)
            ce = _t.where(snowing, z, ce)
            tf = _t.where(snowing, c(rain).expand_as(tf), tf)
            kept = wdew.expand_as(new_wdew)
            if veg is not None:
                # solve_snow.c 5.1.0:251 rescales the store under snow too,
                # Wdew *= fcanopy, on every tile but the bare-soil one
                # (:161, iveg != Nveg): the round trip where fcanopy > 0,
                # and a leafless tile's store deleted.
                gone = fc <= 0.0
                if self.bare_tile is not None:
                    gone = gone & ~self.bare_tile
                kept = _t.where(gone, _t.zeros_like(kept), kept)
                if wdew_per_plant:
                    # An overstory tile's store comes from the canopy
                    # stage, which has applied :251 already.
                    scaled = fc * wdew_p
                    if self.bare_tile is not None:
                        scaled = _t.where(self.bare_tile, kept, scaled)
                    if over is not None:
                        scaled = _t.where(over, kept, scaled)
                    kept = scaled
            new_wdew = _t.where(snowing, kept, new_wdew)
            if bare is not None:
                bare = _t.where(snowing, z, bare)
        if take is not None:
            ce = ce + take
        if _vw is not None and _vw.get('transp_ov') is not None:
            # torchvic.vicwur.core transp: the crop's per-layer uptake
            # replaces the tile's transpiration (VIC-CropSyst
            # VIC_soil.cpp:231-232 -> runoff.c:370-374)
            ov = _vw['transp_ov'].expand_as(transp)
            transp = _t.where(_vw['transp_mask'].unsqueeze(-1)
                              if _vw['transp_mask'].dim() == transp.dim() - 1
                              else _vw['transp_mask'], ov, transp)
        if _vw is not None and _vw.get('stress') is not None:
            # torchvic.vicwur.core water_stress: canopy_evap, and the
            # stress it stores, runs only where !SNOWING && VEG
            # (VIC-WUR 3.0 func_surf_energy_bal.c:756-762; VEG,
            # calc_surf_energy_bal.c:172-182)
            if veg is not None:
                ran = veg
            else:
                ran = lai_p > 0.0
                if fc is not None:
                    ran = ran & (fc > 0.0)
                if self.bare_tile is not None:
                    ran = ran & ~self.bare_tile
            if snowing is not None:
                ran = ran & ~snowing
            _vw['stress']['ran'] = ran
        if _vw is not None and _vw.get('diag') is not None:
            # the tile's layer transpiration and bare-soil evaporation
            # (layer[l].transp, layer[0].esoil), per step
            _vw['diag']['transp'] = transp
            _vw['diag']['bare'] = bare
        if _vw is not None and _vw.get('tflux') is not None:
            # torchvic.vicwur.core tile_fluxes: what the crop plugin reads of
            # the tile after vic_run (VIC-WUR 3.0 crop_run.c:68-71,
            # crop_output.c:1003-1031): layer transp, layer[0].esoil,
            # canopyevap, and cell_data pot_evap, VIC-WUR 3.0
            # compute_pot_evap.c:33-88 as surface_fluxes.c:788-800 calls it for
            # every tile: net_short = (1 - albedo) shortwave, the net longwave
            # NetLongAtmos (no overstory: NetLongUnder = NetLongBare =
            # LongUnderIn - EMISS_GRND Tair^4 at Tsurf = Tair, VIC-WUR 3.0
            # func_surf_energy_bal.c:658-662, calc_surf_energy_bal.c:630),
            # calc_rc unstressed with the library rmin, RGL and the b_co2
            # factor, Ra_veg[0] (an overstory's Ra_veg[1]) for the vegetation
            # and Ra_used[0] with SOIL_RARC for the soil, mixed by fcanopy. A
            # tile under snow is not formed as the C forms it (NetLongAtmos
            # then carries the pack); the R1 runs never snow.
            if shortwave is not None and self.albedo is not None \
                    and longwave is not None:
                nl_pot = (c(longwave) - SURFACE_EMISSIVITY
                          * snow.CONST_STEBOL * (c(tair) + CONST_TKFRZ) ** 4)
                net_pot = ns_t + nl_pot
            else:
                net_pot = c(rad)
            lai_pot = lai if fc is None else lai_p
            rc_pot = calc_rc(self.rmin * cellwise(rmin_scale, self.rmin),
                             ns_t, self.rgl, c(tair), c(vpd), lai_pot,
                             _t.ones_like(lai_pot),
                             co2factor=_vw.get('co2factor'))
            msd = CONST_CDAY / dt
            ev_pot = penman(c(tair), c(elevation), net_pot, c(vpd), ra_t,
                            rc_pot, self.rarc) / msd
            from . import column as _column
            ra_soil = ra_t if bare_soil is None else bare_soil['ra']
            es_pot = penman(c(tair), c(elevation), net_pot, c(vpd), ra_soil,
                            _t.zeros_like(ra_soil),
                            _t.full_like(ra_soil, _column.SOIL_RARC)) / msd
            if fc is None:
                pot = ev_pot
            else:
                pot = fc * ev_pot + (1 - fc) * es_pot
            _vw['tflux'].update(pet=pot, transp=transp, canop=ce,
                                esoil=(_t.zeros_like(ce) if bare is None
                                       else bare))
            if _vw.get('canopy_mask') is not None:
                # the store after the step: what the crop is handed
                _vw['tflux']['wdew'] = new_wdew
            if _vw.get('pet_short'):
                if (shortwave is None or longwave is None or wind is None):
                    raise ValueError('vicwur_core pet_short needs the '
                                     "step's wind, shortwave and longwave")
                # compute_pot_evap.c:70-75, PET_SHORT: rc = 70, rarc = 0, the
                # reference crop's albedo, roughness and displacement
                # (global.c:50-52 [2]); same radiation and forcing
                from ..crop.cropsyst import pet_short as _ps
                ra_ref = aerodynamic_resistance(
                    _t.full_like(c(tair), _ps.REF_ROUGH),
                    _t.full_like(c(tair), _ps.REF_DISPL), c(wind))
                _vw['tflux']['pet_short'] = _t.zeros_like(pot) + \
                    _ps.pet_short(c(tair), c(elevation), c(shortwave),
                                  nl_pot, _ps.REF_ALBEDO, c(vpd), ra_ref,
                                  24.0 / msd).pot_evap_mm
            if _vw.get('se_mask') is not None and bare_soil is not None \
                    and _vw['tflux'].get('pet_short') is not None:
                # vicwur_core crop_soil_evap (VIC-CropSyst canopy_evap.c:
                # 760-831): on a crop tile the crop evaporates the soil
                # under the gaps of its canopy, not ARNO's (1 - fcanopy)
                from . import column as _column
                b = bare_soil
                cg, ct = _vw['se_green'], _vw['se_total']
                # potential from the reference ET and the cover (VIC_soil_
                # hydrology.c:408-411; no residue cover without the OM pools)
                pot_ev = _vw['tflux']['pet_short'] * (1.0 - ct)
                # the soil's own potential in the canopy gaps (:812-819):
                # ARNO from the tile's net radiation, Ra, rc = 0, rarc = 100
                tmp_ev = (1.0 - cg) * _column.bare_soil_evap(
                    penman(c(tair), c(elevation), rad_t, c(vpd), ra_t,
                           _t.zeros_like(ra_t),
                           _t.full_like(ra_t, _column.SOIL_RARC))
                    * dt / CONST_CDAY,
                    b['moist'], b['max_moist'], b['b_infilt'], b['depth'],
                    b['resid_frac'], resid_mm=b.get('resid_mm'))
                ww = w_wilt[..., 0]
                ww = ww.unsqueeze(1) if ww.dim() == 1 else ww
                e_act = _t.minimum(
                    (1.0 - cg) * (b['moist'] - 0.5 * ww),
                    _t.minimum(tmp_ev, pot_ev))
                e_act = _t.clamp(e_act, min=0.0)
                take_ = _vw['se_mask']
                if snowing is not None:
                    take_ = take_ & ~snowing
                bare = _t.where(take_, e_act, bare)
                _vw['tflux']['esoil'] = bare
        if per_tile:
            out = (transp, ce, tf, new_wdew)
        else:
            out = ((transp * self.cv.unsqueeze(-1)).sum(1),
                   (ce * self.cv).sum(1),
                   (tf * self.cv).sum(1),
                   new_wdew)
        if bare_soil is not None:
            out = out + ((bare if per_tile else (bare * self.cv).sum(1)),)
        # `dry` is canopy_evap.c's dryFrac, which the carbon step needs as
        # well: :313-317 scales every carbon flux by it, because a wet leaf
        # is not assimilating.
        return out + (rc, dry) if return_rc else out

    def evap_per_tile(self, month, wdew, rain, avail_moist, w_crit, w_wilt,
                      net_short, tair, vpd, ra, rad, elevation,
                      lai_scale=1.0, rmin_scale=1.0, dt=CONST_CDAY,
                      return_rc=False,
                      share_layer_moist=SHARE_LAYER_MOIST,
                      lai_water_factor=None, bare_soil=None,
                      wilting_clamp=False, ice=None, atten_one=None,
                      wdew_per_plant=False):
        """Interception then transpiration, returned **per tile**, unweighted.

        `step` returns the cell -- tiles weighted by cover and summed --
        which is what the water balance wants. The surface energy balance
        wants the opposite: it solves a temperature *per tile*, and inside
        that solve it needs each tile's own evaporation as a function of that
        tile's net radiation, so it can put latent heat into the residual.
        VIC does exactly this: canopy_evap is called from inside
        func_surf_energy_bal (:757) for one tile at a time.

        `net_short`, `rad` and `ra` are therefore already per tile,
        [cell, tile], and are used as given -- nothing is recomputed from
        albedo or roughness here, because the caller has done that at the
        temperature it is solving for. Returns
        (transp [cell, tile, layer], canopy_evap [cell, tile],
         throughfall [cell, tile], new_wdew [cell, tile]), all mm per step.

        Under the VIC 5.1.0 core with a canopy fraction on the mosaic, the
        tile runs as `step` runs it (func_surf_energy_bal.c 5.1.0:722-761:
        per plant where VEG, nothing where not, every output back to the
        tile), and `bare_soil` (step's dict, `ra` being Ra_used[0]) adds
        the (1 - fcanopy) ARNO bare soil as a fifth element, mm per step
        per tile, ahead of `rc` and `dry`. `atten_one` and `wdew_per_plant`
        are `step`'s.
        """
        self.last_evap_ms = None
        lai = self.lai[:, :, month] * cellwise(lai_scale, self.cv)
        c = lambda x: x.unsqueeze(1) if x.dim() == 1 else x
        fc = None if self.fcanopy is None else self.fcanopy[:, :, month]
        veg = None
        lai_p, wdew_p = lai, wdew
        if fc is not None and core_version.at_least('5.1.0'):
            # vic_run.c 5.1.0:219-226 and calc_surf_energy_bal.c 5.1.0
            # :145-155, as in `step`.
            fc_div = torch.where(fc > 0.0, fc, torch.ones_like(fc))
            lai_p = lai / fc_div
            wdew_p = wdew if wdew_per_plant else wdew / fc_div
            veg = (fc > 0.0) & (lai_p > 0.0)
            if self.bare_tile is not None:
                veg = veg & ~self.bare_tile
        elif fc is not None:
            # VIC 5.0.1: vic_run.c 5.0.1:132-141 divides LAI and Wdew by
            # fcanopy for iveg < Nveg (the bare-soil tile keeps its own),
            # and calc_surf_energy_bal.c 5.0.1:165-175 makes VEG
            # `iveg != Nveg && fcanopy > 0` -- every tile but the bare-soil
            # one, fcanopy being at least MIN_FCANOPY (vic_init.c
            # 5.0.1:271). The energy path ran the canopy on the tile's LAI
            # and never formed the bare soil between the plants: on the
            # ISC windows TRANSP_VEG came out 2 to 80 times VIC's and
            # EVAP_BARE 0 (forest, step 1: VIC 0.066 mm).
            lai_p = lai / fc
            wdew_p = wdew / fc
            veg = fc > 0.0
            if self.bare_tile is not None:
                lai_p = torch.where(self.bare_tile, lai, lai_p)
                wdew_p = torch.where(self.bare_tile, wdew, wdew_p)
                veg = veg & ~self.bare_tile
        # canopy_evap.c:78-98. Wetness is (Wdew/Wdmax)^(2/3) of the store
        # *before* this step's rain, and an empty store is exactly 0.
        # The floored wetness (canopy_wetness exact=False) made an empty
        # canopy 0.00215 wet, and on a daily step -- where the supply is
        # the store plus the rain (canopy_evap.c:88-90) -- that demand
        # evaporated the whole of a small rain. igp_EB_510_D, 2004-01-01,
        # cell 0: VIC stored 0.000794 mm and evaporated 0; this path
        # evaporated 0.000794 mm and stored 0. Sub-daily the supply is
        # the store alone, so the floor could not take rain that had not
        # landed, which is why the 3-hourly rows did not show it.
        ce, new_wdew, tf, dry = canopy_interception(
            wdew_p, c(rain), lai_p, self.rgl, net_short, c(tair), c(vpd),
            ra, self.rarc, rad, c(elevation), dt=dt,
            lai_water_factor=_lwf(lai_water_factor),
            exact_wetness=True)
        # canopy_evap.c 5.1.0:201-241: the moisture transpiration sees is
        # the liquid, layer moist - ice (the clamp below keeps the totals)
        am_liq = avail_moist if ice is None else avail_moist - ice
        transp = self._transpiration(month, am_liq, w_crit, w_wilt,
                                     net_short, tair, vpd, ra, rad, elevation,
                                     lai_scale, rmin_scale,
                                     return_rc=return_rc,
                                     share_layer_moist=share_layer_moist,
                                     lai=None if veg is None else lai_p,
                                     step=(dt, dry))
        rc = None
        if return_rc:
            transp, rc = transp
        if wilting_clamp:
            # canopy_evap.c 5.1.0:478-508 (5.0.1 the same), per plant, after
            # the dryFrac and step scaling and before func_surf_energy_bal
            # scales by fcanopy: no layer transpires below its wilting point,
            # layer[i].moist - Wpwp[i]; where the layer's ice is at or above
            # Wpwp the cap is the unfrozen moisture instead; never negative.
            # The water-balance path (`step`) had it, the energy path did
            # not: a layer drying through Wpwp transpired 0.0251 mm in a
            # step where VIC's cap allowed 0.0106 (igp window, 2004-03-12,
            # both cores).
            am = (avail_moist.unsqueeze(1) if avail_moist.dim() == 2
                  else avail_moist)
            wp = w_wilt.unsqueeze(1) if w_wilt.dim() == 2 else w_wilt
            cap = am - wp
            if ice is not None:
                ic = ice.unsqueeze(1) if ice.dim() == 2 else ice
                cap = torch.where((ic > 0) & (ic >= wp), am - ic, cap)
            transp = torch.clamp(torch.minimum(transp, cap), min=0.0)
        bare = None
        if veg is not None and not core_version.at_least('5.1.0'):
            # func_surf_energy_bal.c 5.0.1:756-810. VEG: canopy_evap per
            # plant, and where fcanopy < 1 the plants' fluxes back to the
            # tile (:762-790: Evap, layer evap, canopyevap and Wdew times
            # fcanopy, throughfall (1 - fc) rainfall + fc throughfall) plus
            # (1 - fc) arno_evap at surf_atten * NetBareRad and Ra_used[0]
            # (:768-772). At fcanopy 1 each is an exact identity and the
            # bare term 0 * finite. Not VEG (the bare-soil tile): no canopy,
            # nothing intercepted or transpired, Wdew as it was, and the
            # whole tile arno_evap at the unattenuated NetBareRad
            # (:797-805).
            f3 = fc.unsqueeze(-1)
            transp = torch.where(veg.unsqueeze(-1), f3 * transp,
                                 torch.zeros_like(transp))
            ce = torch.where(veg, fc * ce, torch.zeros_like(ce))
            tf = torch.where(veg, (1.0 - fc) * c(rain) + fc * tf,
                             c(rain).expand_as(tf))
            new_wdew = torch.where(veg, fc * new_wdew,
                                   wdew.expand_as(new_wdew))
            if bare_soil is not None:
                from . import column as _column
                from .energy import surface_attenuation
                atten = surface_attenuation(
                    lai_p, (self.rad_atten if self.rad_atten is not None
                            else 0.5), fc)
                b = bare_soil

                def arno(r_):
                    return _column.bare_soil_evap(
                        penman(c(tair), c(elevation), r_, c(vpd), b['ra'],
                               torch.zeros_like(b['ra']), _column.SOIL_RARC)
                        * dt / CONST_CDAY,
                        b['moist'], b['max_moist'], b['b_infilt'],
                        b['depth'], b['resid_frac'],
                        resid_mm=b.get('resid_mm'))
                bare = torch.where(veg, (1.0 - fc) * arno(atten * rad),
                                   arno(rad))
        elif veg is not None:
            # func_surf_energy_bal.c 5.1.0:734-751 forms Evap in m/s from
            # the unweighted per-plant sums: canopy_evap.c:134-140,
            # (canopyevap + transp[0] + ...) / (MM_PER_M * delta_t), times
            # fcanopy, then + (1 - fcanopy) * arno_evap's esoil / MM_PER_M /
            # delta_t (arno_evap.c:164). Kept for the residual's latent
            # heat, which a mm sum converted afterwards rounds apart from.
            _tmp = ce
            for _i in range(transp.shape[-1]):
                _tmp = _tmp + transp[..., _i]
            evap_ms = torch.where(veg, _tmp / (MM_PER_M * dt) * fc,
                                  torch.zeros_like(ce))
            f3 = fc.unsqueeze(-1)
            transp = torch.where(veg.unsqueeze(-1), f3 * transp,
                                 torch.zeros_like(transp))
            ce = torch.where(veg, fc * ce, torch.zeros_like(ce))
            tf = torch.where(veg, (1.0 - fc) * c(rain) + fc * tf,
                             c(rain).expand_as(tf))
            new_wdew = torch.where(veg, fc * new_wdew, fc * wdew_p)
            if bare_soil is not None:
                from . import column as _column
                from .energy import surface_attenuation
                atten = surface_attenuation(
                    lai_p, (self.rad_atten if self.rad_atten is not None
                            else 0.5), fc)
                atten = torch.where(veg, atten, torch.ones_like(atten))
                if atten_one is not None:
                    atten = torch.where(atten_one, torch.ones_like(atten),
                                        atten)
                b = bare_soil
                _es = _column.bare_soil_evap(
                    penman(c(tair), c(elevation), atten * rad, c(vpd),
                           b['ra'], torch.zeros_like(b['ra']),
                           _column.SOIL_RARC) * dt / CONST_CDAY,
                    b['moist'], b['max_moist'], b['b_infilt'], b['depth'],
                    b['resid_frac'], resid_mm=b.get('resid_mm'))
                bare = (1.0 - fc) * _es
                evap_ms = evap_ms + (1.0 - fc) * (_es / MM_PER_M / dt)
                self.last_evap_ms = evap_ms
        out = (transp, ce, tf, new_wdew)
        if bare_soil is not None:
            out = out + ((bare if bare is not None
                          else torch.zeros_like(ce)),)
        if return_rc:
            return out + (rc, dry)
        return out

    def _transpiration(self, month, avail_moist, w_crit, w_wilt,
                       net_short, tair, vpd, ra, rad, elevation,
                       lai_scale=1.0, rmin_scale=1.0, return_rc=False,
                       share_layer_moist=SHARE_LAYER_MOIST, lai=None,
                       step=None):
        """Per-tile, per-layer transpiration before the dry-fraction scaling
        (or after it, in the C's order, with `step`; see `transpiration`).

        canopy_evap.c transpiration(), both of its cases, in `transpiration`
        above; this only lays the per-cell moisture and forcing alongside
        the per-tile roots and library values. [cell, 1, layer] broadcasts
        against root_frac's [cell, tile, layer]. `lai`, when given, is the
        leaf area to use instead of the month's scaled one -- the per-plant
        LAI / fcanopy that calc_rc sees under a fractional canopy.
        """
        if lai is None:
            lai = (self.lai[:, :, month]
                   * cellwise(lai_scale, self.lai[:, :, month]))
        rmin = self.rmin * cellwise(rmin_scale, self.rmin)
        nt = self.root_frac.shape[1]

        def tile(x):
            # [cell] -> [cell, tile]; an already per-tile [cell, tile] stays.
            return x.unsqueeze(1).expand(-1, nt) if x.dim() == 1 else x

        # The moisture is per cell [cell, layer] or, with a column per
        # tile, already [cell, tile, layer].
        am = avail_moist.unsqueeze(1) if avail_moist.dim() == 2 else avail_moist
        rc_ps = rs_sink = None
        photo = getattr(self, 'photo', None)
        if photo is not None:
            # RC_MODE = RC_PHOTO. `photo` is set by the step loop with the
            # step's PAR, CO2, direct fraction, zenith and shortwave.
            from . import carbon as _carbon
            lib = self.carbon
            lai_ = lai.expand_as(self.cv)
            bnd, _, _, apar, nscale = _carbon.canopy_light(
                lib['NscaleFlag'] > 0.5, lai_, tile(photo['coszen']),
                tile(photo['fdir']), tile(photo['par']),
                photo['albedo_par'], photo['ncanopy'])
            kw = {} if photo.get('rgas') is None else {'rgas':
                                                      photo['rgas']}
            rc0, rs0, vf = _carbon.rc_ps_unstressed(
                lib['Ctype'], lib['MaxCarboxRate'], lib['MaxETransport'],
                lib['CO2Specificity'], nscale, tile(tair),
                tile(photo['shortwave']), apar, tile(elevation),
                tile(photo['Catm']), bnd, lai_, tile(vpd), **kw)

            def rc_ps(gsm):
                return _carbon.rc_ps_stressed(rc0, rs0, vf, gsm)
            rs_sink = {}
        vw = getattr(self, 'vicwur', None)
        if vw is None:
            out = transpiration(am, w_crit.unsqueeze(1),
                                w_wilt.unsqueeze(1), self.root_frac, lai,
                                rmin, self.rgl, tile(net_short), tile(tair),
                                tile(vpd), tile(ra), self.rarc, tile(rad),
                                tile(elevation), return_rc=return_rc,
                                share_layer_moist=share_layer_moist,
                                rc_ps=rc_ps, rs_sink=rs_sink, step=step)
        else:
            # torchvic.vicwur.core: the layer Wcr per tile, the b_co2
            # factor, the stored stress (basin.simulate sets `vicwur`)
            wc_ = vw.get('wcr')
            out = transpiration(am, (w_crit.unsqueeze(1) if wc_ is None
                                     else wc_),
                                w_wilt.unsqueeze(1), self.root_frac, lai,
                                rmin, self.rgl, tile(net_short), tile(tair),
                                tile(vpd), tile(ra), self.rarc, tile(rad),
                                tile(elevation), return_rc=return_rc,
                                share_layer_moist=share_layer_moist,
                                rc_ps=rc_ps, rs_sink=rs_sink, step=step,
                                co2factor=vw.get('co2factor'),
                                stress=vw.get('stress'))
        if rs_sink is not None:
            # canopy_evap leaves veg_var->rsLayer here; the carbon step
            # reads the converged evaluation's
            self.rs_last = rs_sink['rs']
        return out

    def evapotranspiration(self, month, avail_moist, w_crit, w_wilt,
                           net_short, tair, vpd, ra, rad, elevation,
                           lai_scale=1.0, rmin_scale=1.0):
        """Area-weighted transpiration per soil layer, [cell, layer].

        `lai_scale` and `rmin_scale` multiply the library values, so the
        library supplies the pattern -- which class, which month, which
        relative resistance -- and the multipliers set the level. That is the
        division of labour the vegetation library exists for: its numbers are
        a continental average and the correction is local.
        """
        import torch as _t
        lai = self.lai[:, :, month] * cellwise(lai_scale, self.cv)
        rmin = self.rmin * cellwise(rmin_scale, self.cv)
        c = lambda x: x.unsqueeze(1)                       # [cell] -> [cell,1]
        # [cell, tile, layer]
        am = avail_moist.unsqueeze(1)
        wc = w_crit.unsqueeze(1)
        wp = w_wilt.unsqueeze(1)
        gsm = soil_moisture_stress(am, wc, wp)
        # Canopy resistance is per tile; the stress that sets it is the
        # root-weighted mean over layers, which is what the canopy as a whole
        # experiences.
        gsm_t = (gsm * self.root_frac).sum(-1)
        rc = calc_rc(rmin, c(net_short), self.rgl, c(tair), c(vpd), lai, gsm_t)
        et = penman(c(tair), c(elevation), c(rad), c(vpd), c(ra), rc,
                    self.rarc)                             # [cell, tile]
        per_layer = et.unsqueeze(-1) * self.root_frac * gsm
        return (per_layer * self.cv.unsqueeze(-1)).sum(1)  # [cell, layer]

    def __repr__(self):
        return ('<VegMosaic %d cells x %d tiles, cover %.3f..%.3f>'
                % (self.cv.shape[0], self.n_tiles,
                   float(self.cv.sum(1).min()), float(self.cv.sum(1).max())))


# vic/drivers/shared_all/src/initialize_parameters.c
VEG_LAI_WATER_FACTOR = 0.1     # Wdmax = this * LAI, mm of canopy storage


def _lwf(x):
    """The interception capacity per unit LAI: the run's value if given
    (p['lai_water_factor'], a run parameter -- VIC reads it from its parameter
    file as VEG_LAI_WATER_FACTOR), else VIC's default."""
    return VEG_LAI_WATER_FACTOR if x is None else x
INTERCEPT_EXP = 2.0 / 3.0      # canopy_evap.c: (Wdew/Wdmax)^(2/3)
# Smallest relative canopy storage the 2/3 power may see. d/dx of x**(2/3) is
# (2/3) x**(-1/3), which is infinite at zero, and a canopy is empty most of
# the year. Flooring the base keeps the worst derivative near 145 instead of
# unbounded. This is the same failure that collapsed training through
# avail**et_beta and through the Brooks-Corey powers; a floor has to be chosen
# against the operation that follows it.
#
# The floor is for the derivative, not the value. As a value it made an empty
# canopy wet: 1e-4 ** (2/3) = 0.00215 where canopy_evap.c:99 gives 0, and on a
# daily step -- where the day's rain may evaporate as it lands
# (canopy_evap.c:108-110) -- that demand ate small rains whole. Basin 37,
# 2000-2002, daily water balance: cell 39 evaporated 9.23 mm off its canopy on
# day 140 where VIC's whole ET was 0.658 mm, and basin-mean daily ET p99
# |diff| against VIC was 0.72 mm. `canopy_wetness(exact=True)` keeps VIC's
# value and takes the gradient through the floor.
WDEW_FLOOR = 1e-4


def canopy_wetness(held, wdmax, exact=True):
    """(Wdew / Wdmax)^(2/3), canopy_evap.c:99 and :123.

    `exact=True` is VIC's value for every store: exactly 0 for an empty
    canopy and the true power for any positive one, however small -- a
    canopy dried through `f < 1` leaves a residue of a few 1e-18 mm, and
    flooring that residue brings back the wet empty canopy just as surely
    as flooring an exact zero. The derivative is taken through
    clamp(ratio, WDEW_FLOOR) where the ratio is at or above the floor (the
    true derivative there) and is zero below it, where the true one
    diverges; the unselected pow always sees a base of at least the floor,
    so no branch is non-finite and the gradient at held = 0 is finite
    ('torch.where is not an if').

    `exact=False` is the floored value this module used before, kept for
    the surface energy balance, whose VIC-comparison tests are bit-exact.
    There every step is sub-daily and the supply is the store alone, so
    where the demand exceeds the store `f * wet` is held / penman whichever
    wetness is used. The two differ only while the store is under
    WDEW_FLOOR of Wdmax: the canopy term by less than that store, the dry
    fraction by at most 1e-4 ** (2/3) = 0.00215.
    """
    if not exact:
        return xmath.pow(torch.clamp(held / wdmax, min=WDEW_FLOOR,
                                     max=1.0), INTERCEPT_EXP)
    stored = held > 0.0
    ratio = torch.where(stored, held / wdmax, torch.ones_like(held))
    smooth = xmath.pow(torch.clamp(ratio, min=WDEW_FLOOR, max=1.0),
                       INTERCEPT_EXP)
    steep = xmath.pow(torch.clamp(ratio, max=1.0).detach(), INTERCEPT_EXP)
    wet = torch.where(ratio >= WDEW_FLOOR, smooth, steep)
    return torch.where(stored, wet, torch.zeros_like(wet))


#: von Karman squared, as VIC's K2.
#: Surface emissivity for the per-tile net radiation. The Stefan-Boltzmann
#: constant and the freezing point come from the modules that already own
#: them, so there is one definition of each in the package rather than a copy
#: here that can drift.
SURFACE_EMISSIVITY = 0.97
#: von Karman squared. VIC defines CONST_KARMAN as **0.4**, not 0.41
#: (vic_physical_constants.h:82), and CalcAerodynamic.c:67 squares that to
#: 0.16. Using 0.41 makes every aerodynamic resistance 4.8 per cent too small
#: -- which biases sensible and latent heat and closes every water balance
#: perfectly, because a resistance is not a mass term.
CONST_KARMAN = 0.4
VON_KARMAN_SQ = CONST_KARMAN ** 2
#: (1/0.63 - 1), the displacement factor in VIC's lower-boundary resistance.
MM_PER_M = 1000.0
DISPL_FACTOR = 1.0 / 0.63 - 1.0
#: VIC's param.HUGE_RESIST, used when the wind is calm.
HUGE_RESIST = 1.0e20


def aerodynamic_resistance(veg_rough, displacement, wind, min_wind=1e-12):
    """Aerodynamic resistance over vegetation, s/m.  CalcAerodynamic.c.

    The no-overstory branch, which is every class at Walnut Gulch and most
    non-forest classes anywhere:

        Ra[0] = log((2 + (1/0.63 - 1) d) / z0)
              * log((2 + (1/0.63 - 1) d) / (0.1 z0)) / K2      then / wind

    and `canopy_evap` is handed `Ra_veg[1]`, which the no-overstory branch
    sets equal to `Ra[0]`.

    Two things are easy to get wrong here. The reference height does not
    appear: VIC evaluates the profile 2 m above the displacement height, not
    at the forcing's measurement height, so WIND_H belongs to the snow surface
    and not to this. And the roughness is the **tile's** -- 0.495 m for the
    shrub classes, 0.074 m for grassland -- not the snow roughness. Using
    snow's 0.03 m for vegetation gave 70.5 s/m where VIC gives 13.7 for shrub
    and 40.7 for grass at this site's mean wind.

    Overstory is not implemented: a forest needs the canopy wind profile and
    the trunk space, and a configuration with overstory classes must not rely
    on this.
    """
    z0 = torch.clamp(veg_rough, min=1e-4)
    top = 2.0 + DISPL_FACTOR * displacement
    ra = (torch.log(top / z0) * torch.log(top / (0.1 * z0))
          / VON_KARMAN_SQ)
    return torch.where(wind > 0.0,
                       ra / torch.clamp(wind, min=min_wind),
                       torch.full_like(ra, HUGE_RESIST))


def aerodynamic_resistance_overstory(height, trunk_ratio, z0_snow, z0_soil,
                                     wind_atten, roughness, displacement,
                                     ref_height, wind):
    """CalcAerodynamic.c, the overstory branch (:50-112), per tile.

    Returns a dict: `ra` [..., 3] and `u` [..., 3] -- VIC's Ra[0..2] and
    U[0..2] after the division by the wind (:120-131) -- and the
    reference heights, roughnesses and displacements the C rewrites for the
    understory ([0]: 2 m over the soil roughness, no displacement), the
    canopy ([1]: the tile's own) and the snow surface ([2]: 2 m over the
    snow roughness, :105-112 and :30-35).

    The canopy wind profile is exponential inside the canopy with
    attenuation `wind_atten`, logarithmic above it; `Zw = 1.5 h - 0.5 d`
    is the top of the roughness sublayer and `Zt = trunk_ratio * h` the top
    of the trunk space. The resistance to the snow surface takes one of
    three forms by where Zt and h sit against 2 m over the snow (:73-104),
    selected here with `where` on finite branches: the logarithms are
    guarded so an unselected branch cannot put NaN into the gradient.

    `height` is calc_veg_height's displacement / VEG_RATIO_DH_HEIGHT
    (vic_run.c:256); `ref_height` is vic_run.c:258-263, wind_h or
    d + wind_h + z0. A wind of 0 gives HUGE_RESIST for all three, as the C's
    else branch does.
    """
    k2 = VON_KARMAN_SQ
    h = height
    z0u = roughness
    du = displacement
    z0l = z0_soil
    n = wind_atten
    zw = 1.5 * h - 0.5 * du
    zt = trunk_ratio * h
    rh = ref_height
    lg = lambda x: torch.log(torch.clamp(x, min=1e-12))

    ra1 = (lg((rh - du) / z0u) / k2
           * (h / (n * (zw - du)) * (xmath.exp(n * (1.0 - (du + z0u) / h)) - 1.0)
              + (zw - h) / (zw - du) + lg((rh - du) / (zw - du))))
    uw = lg((zw - du) / z0u) / lg((rh - du) / z0u)
    uh = uw - (1.0 - (h - du) / (zw - du)) / lg((rh - du) / z0u)
    u1 = uh * xmath.exp(n * ((z0u + du) / h - 1.0))
    ut = uh * xmath.exp(n * (zt / h - 1.0))
    u0 = lg((2.0 + z0u) / z0u) / lg((rh - du) / z0u)
    ra0 = (lg((2.0 + DISPL_FACTOR * du) / z0u)
           * lg((2.0 + DISPL_FACTOR * du) / (0.1 * z0u)) / k2)

    # The C divides by Ut as it is, negative included (a roughness above
    # the displacement gives one); only an exact zero is guarded.
    safe_ut = torch.where(ut != 0.0, ut, torch.ones_like(ut))
    # :73-78, trunk space above 2 m over the snow: wholly logarithmic
    u2_a = ut * lg((2.0 + z0_snow) / z0_snow) / lg(zt / z0_snow)
    ra2_a = lg((2.0 + z0_snow) / z0_snow) * lg(zt / z0_snow) / (k2 * safe_ut)
    # :83-93, canopy top above 2 m but trunk space below: log then exponential
    u2_b = uh * xmath.exp(n * ((2.0 + z0_snow) / h - 1.0))
    ra2_b = (lg(zt / z0_snow) * lg(zt / z0_snow) / (k2 * safe_ut)
             + h * lg((rh - du) / z0u) / (n * k2 * (zw - du))
             * (xmath.exp(n * (1.0 - zt / h))
                - xmath.exp(n * (1.0 - (z0_snow + 2.0) / h))))
    # :98-104, canopy top under 2 m over the snow: the profile ends at h
    u2_c = uh
    ra2_c = (lg(zt / z0_snow) * lg(zt / z0_snow) / (k2 * safe_ut)
             + h * lg((rh - du) / z0u) / (n * k2 * (zw - du))
             * (xmath.exp(n * (1.0 - zt / h)) - 1.0))
    top = zt > (2.0 + z0_snow)
    mid = (~top) & (h > (2.0 + z0_snow))
    u2 = torch.where(top, u2_a, torch.where(mid, u2_b, u2_c))
    ra2 = torch.where(top, ra2_a, torch.where(mid, ra2_b, ra2_c))

    # :120-131: scale by the wind, or HUGE_RESIST without one
    blowing = wind > 0.0
    safe_w = torch.where(blowing, wind, torch.ones_like(wind))
    huge = torch.full_like(ra0, HUGE_RESIST)
    ra = torch.stack([torch.where(blowing, ra0 / safe_w, huge),
                      torch.where(blowing, ra1 / safe_w, huge),
                      torch.where(blowing, ra2 / safe_w, huge)], dim=-1)
    u = torch.stack([u0 * wind, u1 * wind, u2 * wind], dim=-1)
    two = torch.full_like(h, 2.0)
    zero = torch.zeros_like(h)
    return dict(ra=ra, u=u,
                ref_height=torch.stack([two, rh, 2.0 + z0_snow], -1),
                roughness=torch.stack([z0l, z0u, z0_snow], -1),
                displacement=torch.stack([zero, du, zero], -1))


def canopy_interception_evap(wdew, ppt, lai, rgl, net_short, tair, vpd, ra,
                             rarc, elevation, dt=CONST_CDAY,
                             lai_water_factor=VEG_LAI_WATER_FACTOR,
                             exact_wetness=False):
    """`canopy_interception(...)[0]`, the canopy evaporation, as a function
    of the net radiation `rad` alone: everything that does not depend on
    `rad` is formed once, here, and the returned function forms the rest
    with the same operations in the same order -- the same number, bit for
    bit. For a solver that evaluates the evaporation at many trial canopy
    temperatures with the rest of the arguments fixed."""
    wdmax = torch.clamp(lai * cellwise(lai_water_factor, lai), min=1e-9)
    # canopy_evap.c 5.1.0:68-71: the store is capped at Wdmax by assignment
    # (wdew - (wdew - Wdmax) is not Wdmax in floating point)
    over = wdew > wdmax
    spill = torch.where(over, wdew - wdmax, torch.zeros_like(wdew))
    held = torch.where(over, wdmax, wdew)
    rc = calc_rc(torch.zeros_like(lai), net_short, rgl, tair, vpd, lai,
                 torch.ones_like(lai))
    # penman(), less the radiation term
    slope = atmos.svp_slope(tair)
    pz = atmos.air_pressure(tair, elevation)
    lv = atmos.latent_heat_of_vaporization(tair)
    gamma = 1628.6 * pz / lv
    r_air = 0.003486 * pz / (275.0 + tair)
    aero = r_air * CONST_CPMAIR * vpd / ra
    den = lv * (slope + gamma * (1.0 + (rc + rarc) / ra))
    vpd_ok = vpd >= 0.0
    wet = canopy_wetness(held, wdmax, exact=exact_wetness)
    leafy = lai > 0.0
    supply = held + ppt if dt == CONST_CDAY else held

    def evap(rad):
        e = (slope * rad + aero) / den * CONST_CDAY
        e = torch.where(vpd_ok & (e < 0.0), torch.zeros_like(e), e)
        # canopy_evap.c 5.1.0:75-78, left to right
        demand = torch.where(leafy, wet * e * dt / CONST_CDAY,
                             torch.zeros_like(e))
        f = torch.where(demand > 0.0,
                        torch.clamp(supply / torch.clamp(demand, min=1e-12),
                                    max=1.0),
                        torch.ones_like(demand))
        return demand * f
    return evap


def canopy_interception(wdew, ppt, lai, rgl, net_short, tair, vpd, ra, rarc,
                        rad, elevation, dt=CONST_CDAY,
                        lai_water_factor=VEG_LAI_WATER_FACTOR,
                        exact_wetness=False, no_evap=None):
    """Evaporation from intercepted water, and the dry fraction of the canopy.

    canopy_evap.c, the part before transpiration. Returns
    (canopy_evap, new_wdew, throughfall, dry_fraction), all in mm.

    The canopy holds `Wdmax = lai_water_factor * LAI` millimetres. Anything
    above that falls through. What remains evaporates at the potential rate
    with **no stomatal resistance** -- open water on a leaf does not have to
    pass through a stoma, which is why `calc_rc` is called here with a minimum
    resistance of zero -- scaled by the two-thirds power of relative wetness.

    `dry_fraction` is the share of the canopy not covered by water, and it is
    what transpiration is multiplied by afterwards. A canopy cannot both
    evaporate free water and transpire through the same leaf surface at the
    same time, and this is the term that stops it doing so. Nothing in the
    previous scheme expressed that at all.

    `exact_wetness` selects `canopy_wetness(exact=True)`, VIC's value, which
    the water-balance path uses; the default keeps the floored value the
    energy balance's bit-exact tests were recorded with.

    `no_evap` (bool, broadcast to the tiles; None: VIC 5): the canopy does
    not evaporate its store there, VIC-CropSyst's CROPSYST_HANDLE_CANOPY_EVAP
    (canopy_evap.c:609-611, `canopyevap = 0`; the crop evaporates it,
    vicwur_core canopy_crop).
    """
    wdmax = torch.clamp(lai * cellwise(lai_water_factor, lai), min=1e-9)
    # canopy_evap.c 5.1.0:68-71: the store is capped at Wdmax by assignment
    over = wdew > wdmax
    spill = torch.where(over, wdew - wdmax, torch.zeros_like(wdew))
    held = torch.where(over, wdmax, wdew)

    # rs = 0 gives rc = 0: free water, no stomatal control.
    rc = calc_rc(torch.zeros_like(lai), net_short, rgl, tair, vpd, lai,
                 torch.ones_like(lai))
    pet = penman(tair, elevation, rad, vpd, ra, rc, rarc)
    wet = canopy_wetness(held, wdmax, exact=exact_wetness)
    # :75-78, pow(...) * penman(...) * delta_t / CONST_CDAY, left to right
    demand = torch.where(lai > 0.0, wet * pet * dt / CONST_CDAY,
                         torch.zeros_like(pet))
    if no_evap is not None:
        demand = torch.where(no_evap, torch.zeros_like(demand), demand)

    # The supply limit. On a daily step VIC lets the falling precipitation be
    # evaporated as it lands; on a shorter step only what is already held.
    supply = held + ppt if dt == CONST_CDAY else held
    f = torch.where(demand > 0.0,
                    torch.clamp(supply / torch.clamp(demand, min=1e-12),
                                max=1.0),
                    torch.ones_like(demand))
    canopy_evap = demand * f
    dry_fraction = torch.where(wdmax > 0.0, 1.0 - f * wet,
                               torch.zeros_like(wet))

    # :104, tmp_Wdew += ppt - canopyevap
    new_wdew = torch.clamp(held + (ppt - canopy_evap), min=0.0)
    spill = spill + torch.clamp(new_wdew - wdmax, min=0.0)
    new_wdew = torch.minimum(new_wdew, wdmax)
    return canopy_evap, new_wdew, spill, dry_fraction


# fused regions on the CUDA graph path (core.fusion)
from . import fusion as _fusion                                # noqa: E402
VegMosaic.step = _fusion.region(VegMosaic.step)
canopy_interception = _fusion.region(canopy_interception)
aerodynamic_resistance_overstory = _fusion.region(
    aerodynamic_resistance_overstory)
