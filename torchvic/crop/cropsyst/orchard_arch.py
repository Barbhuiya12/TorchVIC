"""Orchard canopy architecture of the VIC-CropSyst port: the tree canopy extent
and the solid canopy interception of the orchard geometry (Orchard_common with its
Interception_architectural models, crop_orchard_common.cpp, crop_orchard_Kemanian_Stockle.cpp,
crop_orchard_Oyarzun_Stockle.cpp).

Switch (as CropSyst's own parameter files do it):
- crop file `[morphology] canopy_architecture=true` (crop_param_class.cpp:346, 353); the geometry is
  `[canopy_characteristics]` initial/final width (across the row), breadth (along the row) and
  height, branch_insertion_height, and `[layout]` row_spacing, col_spacing (tree spacing along the
  row), row_azimuth (true north, degrees) (crop_param_V5.cpp:755-766; defaults
  crop_param_class.cpp:784-819);
- the scenario's `[submodels] orchard_interception_model` = none / Kemanian-Stockle /
  Oyarzun-Stockle (model_options.cpp:46-50, 204; Oyarzun-Stockle when the key is absent,
  model_options.cpp:132). The CRB scenarios ship `none`. Port: the cropsyst block's
  `orchard_interception_model` (same labels, default Oyarzun-Stockle).

What the reference does with it (perform_op.cpp:430-436, 1761-1826; crop_fruit.cpp:646-690):
- the fruit crop gets an Orchard_common; with an interception model also a Tree_canopy_extent whose
  three dimensions start at their initial sizes and expand between the thermal times
  deg_days_begin (flowering at creation; the accumulated thermal time at each restart, i.e. 0
  because the restart clears it, crop_fruit.cpp:262-270) and deg_days_end (rapid_fruit_growth):
  size = initial + (final - initial) (tt - begin) / (end - begin) while begin < tt < end, else kept
  (crop_orchard_common.cpp:52-99);
- every process_day of the crop, before the fruit harvest requirements and the rest of the day,
  the canopy expands with the crop's accumulated thermal time and the solid canopy interception is
  updated (crop_fruit.cpp:659-671):
  * Oyarzun-Stockle (crop_orchard_Oyarzun_Stockle.cpp:81-334): for each hour with an insolation
    fraction > 0.001 (Solar_radiation_extraterrestrial_hourly, irradiance_extraterrestrial.cpp:
    123-164, on the zenith of Sun_hours, sun_hours.cpp:182-245, and the solar noon of Sun_days,
    sun_days.cpp:5-31 with longitude = standard meridian, CS_land_unit_meteorological.cpp:129-137),
    the shaded fraction of the orchard floor from the shadow of a box canopy of the current height
    above the branch insertion height, across and along the rows, the canopy width and breadth
    limited to the row and tree spacing; the diffuse transmittance from the shaded fraction
    integrated over the day's hours from sunrise + 1 to sunset (Sun::provide_sun_rise_set_times,
    sun.cpp:164-209, integer hours), at most 0.9; the hourly direct and diffuse radiation of
    Solar_radiation_hour (solar_irrad_hour.cpp:20-35); daily = sum of insolation fraction x hourly
    interception;
  * Kemanian-Stockle (crop_orchard_Kemanian_Stockle.cpp:49-250): the hourly row shadow of a
    hedgerow on a (sloped) floor, weighted by the potential direct and diffuse radiation, mixed with
    the diffuse fraction of the land unit's cloud cover (Sky::update_clouds, sky.cpp:98-150: set
    once, from the first simulated day's radiation, never invalidated in the VIC build);
  * none: 1.0.
- The interception is not used by any crop process in this build: Orchard_common::Canopy_
  architectural::calc_cover_fract (crop_orchard_common.cpp:161-203) is not an override of the
  canopy's cover (canopy_growth_LAI_based.cpp:483-511 keeps the field-crop cover), and nothing else
  reads get_solid_canopy_interception (only the V4 report, cs_report_V4.cpp:416). With the
  architecture on, every crop and soil output of the reference equals the run without it (318377,
  N off and N on, all three models: the crop csv, fluxes and dumps byte-equal but the crop name).
  The port reports the interception and the canopy extent (OrchardDayOutput; connector outputs
  orchard_interception, canopy_width, canopy_breadth, canopy_height).

Reference defects (patched only in the dump build copy of the reference):
Orchard_common::update_solid_canopy_interception stores the value in
canopy_architectural, which exists only once the crop has provided a canopy
(crop_orchard_common.cpp:48, 205-212); a fruit tree started dormant has none, so the shipped build
dies with SIGSEGV on the first day of every model (P5: null guard; output-only). Not reached: the
Orchard_common constructor never sets its light extinction coefficient (used only by the dead
calc_cover_fract); Crop_orchard_fruit::get_orchard_solid_canopy_interception calls itself
(crop_fruit.cpp:765-766, output only); the slope aspect is an uninitialised, unread scenario member
(cs_scenario.h:138) - harmless at the VIC build's zero slope.

Terrain: the VIC build's soil steepness is 0 (the dumped shadow correction factor is 1 on every
hour); `slope_percent` and `slope_aspect` (degrees, 0 north, 90 east) of the cropsyst block set
them (default 0 and 180).

Batched over tiles x (static) hours with masks. Everything that depends only on the site (latitude,
longitude, slope, aspect), the day of year and the row azimuth - the sun's position, sunrise and
sunset, insolation fractions, the trigonometry of the shadows - is tabulated once at setup with the
C library's functions (python math), so the day step is plain arithmetic on the canopy size plus
exp of the diffuse fraction: no host syncs, no data-dependent control flow. Polar days and nights
(|lat| > 66.5) are outside the reference's formulas (sun.cpp:192-197).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

MODELS = ["none", "Kemanian-Stockle", "Oyarzun-Stockle"]   # model_options.cpp:46-50
NONE, KEMANIAN, OYARZUN = 0, 1, 2
DEFAULT_MODEL = "Oyarzun-Stockle"                          # model_options.cpp:132
DEG_TO_RAD = 0.017453293                                   # measures.h:18 deg_to_rad
PI = math.pi                                               # moremath.h:101 CORN_pi = M_PI
PI_KS = 3.14159                                            # literal of crop_orchard_Kemanian_Stockle.cpp
DEFAULTS = dict(w0=1.0, w1=1.0, b0=2.0, b1=2.0, h0=1.2, h1=1.2, bih=0.6,   # crop_param_class.cpp:784-792
                row=2.7, col=2.5, azim=0.0)                                 # :813-818


def _f32(x):
    from .growth import round_f32_bits
    return round_f32_bits(x.contiguous())


class ArchParams(NamedTuple):
    """Per-crop canopy architecture (float32 parameters as float64)."""
    on: torch.Tensor                # bool, [morphology] canopy_architecture
    w0: torch.Tensor                # initial / final width across the row (m)
    w1: torch.Tensor
    b0: torch.Tensor                # initial / final breadth along the row (m)
    b1: torch.Tensor
    h0: torch.Tensor                # initial / final height (m)
    h1: torch.Tensor
    bih: torch.Tensor               # branch insertion height (m)
    row: torch.Tensor               # row spacing (m)
    col: torch.Tensor               # tree spacing along the row (m)
    row_az: torch.Tensor            # row azimuth, radians, 0 south, east positive (crop_orchard_common.cpp:127)
    az_idx: torch.Tensor            # long, the crop's row azimuth among the distinct ones (site tables)
    begin0: torch.Tensor            # thermal time of flowering (deg_days_begin at creation)
    end: torch.Tensor               # thermal time of rapid fruit growth (deg_days_end)


def _aspect_s0(deg_n0):
    """convert_aspect_from_0_north_with_east_90_to_0_south_with_east_90 (crop_orchard_common.cpp:
    136-143), float32 result."""
    d = torch.as_tensor(deg_n0, dtype=torch.float64)
    return _f32(torch.where(d < 180.0, 180.0 - d, 540.0 - d))


def arch_params(crops, PP, device=None, dtype=torch.float64) -> ArchParams:
    """[n_crop] from the crop table and the phenology parameters (flowering, rapid_fruit)."""
    n = crops.n
    dev = device if device is not None else PP.flowering.device

    def get(key, default):
        t = crops.params.get(key)
        full = torch.full((n,), float(default), dtype=dtype)
        if t is None:
            return full
        t = t.cpu()
        missing = torch.isnan(t) if t.dtype.is_floating_point else t == -1
        return torch.where(missing, full, t.to(dtype))

    on = crops.params.get("morphology.canopy_architecture")
    on = torch.zeros(n, dtype=torch.bool) if on is None else (on.cpu() == 1)
    f = lambda k, s, d: _f32(get(f"{s}.{k}", DEFAULTS[d]))
    az = _aspect_s0(f("row_azimuth", "layout", "azim")) * DEG_TO_RAD
    uaz = sorted(set(az.tolist()))
    idx = torch.tensor([uaz.index(v) for v in az.tolist()], dtype=torch.long)
    to = lambda x: x.to(dev)
    return ArchParams(on=to(on), w0=to(f("initial_width", "canopy_characteristics", "w0")),
                      w1=to(f("final_width", "canopy_characteristics", "w1")),
                      b0=to(f("initial_breadth", "canopy_characteristics", "b0")),
                      b1=to(f("final_breadth", "canopy_characteristics", "b1")),
                      h0=to(f("initial_height", "canopy_characteristics", "h0")),
                      h1=to(f("final_height", "canopy_characteristics", "h1")),
                      bih=to(f("branch_insertion_height", "canopy_characteristics", "bih")),
                      row=to(f("row_spacing", "layout", "row")), col=to(f("col_spacing", "layout", "col")),
                      row_az=to(az), az_idx=to(idx), begin0=PP.flowering.to(dev, dtype),
                      end=PP.rapid_fruit.to(dev, dtype))


# ----------------------------------------------------------------------------- tables (setup, libm)

def _sdec(d):
    return 0.39785 * math.sin(4.869 + 0.0172 * d + 0.03345 * math.sin(6.224 + 0.0172 * d))


def _sun_tables(lat_deg32: float):
    """Per latitude (float32 degrees), [367][24] lists as the land unit's Sun_days / Sun_hours /
    Solar_radiation_extraterrestrial_hourly fill them: zenith, azimuth, insolation fraction, and
    [367] the daily extraterrestrial radiation (MJ m-2)."""
    lat = lat_deg32 * DEG_TO_RAD                                # Sun_hours(latitude_deg) deg_to_rad
    Z = [[0.0] * 24 for _ in range(367)]
    A = [[0.0] * 24 for _ in range(367)]
    F = [[0.0] * 24 for _ in range(367)]
    E = [0.0] * 367
    sl, cl = math.sin(lat), math.cos(lat)
    for d in range(1, 367):
        dec = math.asin(_sdec(d))                               # sun_days.cpp:5-12
        adj = 1.0 + 0.0334 * math.cos(0.01721 * d - 0.0552)
        f = (279.575 + 0.9856 * d) * math.pi / 180.0            # sun_days.cpp:18-31
        et = (-104.7 * math.sin(f) + 596.2 * math.sin(2.0 * f) + 4.3 * math.sin(3.0 * f)
              - 12.7 * math.sin(4.0 * f) - 429.3 * math.cos(f) - 2.0 * math.cos(2.0 * f)
              + 19.3 * math.cos(3.0 * f)) / 3600.0
        t0 = 12.0 - 0.0 - et                                    # standard meridian = longitude
        sd, cd = math.sin(dec), math.cos(dec)
        R = []
        for h in range(24):
            h1 = h + 1                                          # sun_hours.cpp:182-195
            cosz = sl * sd + cl * cd * math.cos(0.2618 * (h1 - t0))
            sinz = math.sqrt(1.0 - cosz * cosz)
            z = math.pi / 2 - math.atan(cosz / sinz)
            if cosz < 0:
                z = math.pi / 2
            Z[d][h] = z
            noon = 12.0                                         # sun_hours.cpp:198-245
            ca = (-(sd - math.cos(z) * sl) / (cl * math.sin(z))) if z < math.pi / 2 else 0.0
            ca = ca if h1 <= noon else -ca
            if abs(ca) > 1.0:
                ca = 0.9999
            sa = math.sqrt(1.0 - ca * ca)
            if abs(0.0 - sa) <= 0.00001:
                sa = 0.000001
            A[d][h] = (PI / 2.0 - math.atan(ca / sa)) if h1 < noon else ((3.0 / 2.0) * PI - math.atan(ca / sa))
            r = 1360.0 * (adj * adj) * math.cos(z)              # irradiance_extraterrestrial.cpp:150-164
            R.append(0.0 if r < 0.0 else r)
        s = 0.0
        for h in range(24):
            s += R[h]
        for h in range(24):
            F[d][h] = R[h] / s
        E[d] = ((s / 24.0) / 1000000.0) * 86400                 # get_MJ_per_m2_day of the hourly object
    return Z, A, F, E


def _rise_set(lat_deg32, lon_deg32):
    """[367] x 2 integer hours of Sun::get_sun_rise_time / get_sun_set_time (sun.cpp:40-55, 143-224;
    time_A.cpp:24-25): Sun::calc_angles' declination of the day, the solar noon with the standard
    meridian (int)(longitude), seconds truncated."""
    lat = lat_deg32 * DEG_TO_RAD                                # Geocoordinate::get_latitude_radians
    out = [[0, 0] for _ in range(367)]
    for d in range(1, 367):
        dec = math.asin(0.39785 * math.sin(4.869 + 0.0172 * d + 0.03345 * math.sin(6.224 + 0.0172 * d)))
        coshs = (-math.sin(lat) * math.sin(dec)) / (math.cos(lat) * math.cos(dec))
        sinhs = math.sqrt(1.0 - coshs * coshs)
        half = math.pi / 2 - math.atan(coshs / sinhs)
        f = (279.575 + 0.9856 * d) * math.pi / 180.0
        et = (-104.7 * math.sin(f) + 596.2 * math.sin(2.0 * f) + 4.3 * math.sin(3.0 * f)
              - 12.7 * math.sin(4 * f) - 429.3 * math.cos(f) - 2.0 * math.cos(2.0 * f)
              + 19.3 * math.cos(3.0 * f)) / 3600.0
        noon = 12.0 - (float(int(lon_deg32)) - lon_deg32) / 15.0 - et
        hfn = (half * 180.0 / math.pi) / 15.0
        hr = lambda x: math.fmod(math.trunc(math.trunc(x * (60.0 * 60.0)) / 3600), 24)   # C int / and %
        out[d] = [int(hr(noon - hfn)), int(hr(noon + hfn))]
    return out


def _scf_table(azi, slope, aspect):
    """calc_shadow_correction_factor (crop_orchard_Oyarzun_Stockle.cpp:292-309) of every hour."""
    out = [[0.0] * 24 for _ in range(367)]
    cs = math.cos(slope)
    for d in range(1, 367):
        for h in range(24):
            x = abs(azi[d][h] - aspect)
            v = cs if x < PI * 0.25 else 1.0 if x < PI * 0.75 else 1.0 / cs if x < PI * 1.25 else \
                1.0 if x < PI * 1.75 else cs
            out[d][h] = max(v, 0.0)
    return out


def _pot(s):
    """calc_hourly_direct_radiation_potential (crop_orchard_Kemanian_Stockle.cpp:143-147)."""
    return max(1360.0 * s * math.pow(0.75, (1.0 / s)), 0.0)


def _ks_sun(lat):
    """[367][24] (sun up, sine of elevation, elevation, azimuth) of Interception_architectural_
    Kemanian_Stockle (crop_orchard_Kemanian_Stockle.cpp:49-66, 156-167)."""
    out = [[[0.0] * 24 for _ in range(367)] for _ in range(4)]
    for d in range(1, 367):
        ssd = _sdec(d)
        for h in range(24):
            asd = math.asin(ssd)
            sse = math.sin(lat) * ssd + math.cos(lat) * math.cos(asd) * math.cos(DEG_TO_RAD * (15 * (h - 11)))
            sse = min(max(sse, -0.9999), 0.9999)
            elev = max(math.asin(sse), 0.0)
            az = 0.0
            if sse > 0:
                csa = -(ssd - sse * math.sin(lat)) / (math.cos(lat) * math.cos(math.asin(sse)))
                csa = min(max(csa, -0.9999), 0.9999)
                az = math.acos(csa) if h <= 11 else (1.5 * PI_KS + math.asin(csa))
            for k, v in enumerate((1.0 if sse > 0 else 0.0, sse, elev, az)):
                out[k][d][h] = v
    return out


def _ks_site(sun, slope, aspect):
    """[367][24] (equivalent elevation, shadow length correction, potential direct and diffuse
    radiation) of a Kemanian-Stockle site (crop_orchard_Kemanian_Stockle.cpp:105-201); 0.0001 for
    the potentials with the sun down."""
    up, _, elev_t, az_t = sun
    out = [[[0.0] * 24 for _ in range(367)] for _ in range(4)]
    for d in range(1, 367):
        for h in range(24):
            v = (0.0, 0.0, 0.0001, 0.0001)
            if up[d][h] > 0.5:
                elev, az = elev_t[d][h], az_t[d][h]
                c = (math.sin(slope) * math.cos(elev) * math.cos(-(aspect - az))
                     + math.cos(slope) * math.sin(elev))
                ia = math.atan(-c / math.sqrt(-c * c + 1.0)) + 2.0 * math.atan(1.0)
                eq = PI_KS / 2 - ia
                cor = math.sin(elev) / math.sin(3.14159 - elev - slope * math.cos(aspect - az))
                v = (eq, cor, _pot(math.sin(eq)) if eq > 0 else 0.0, _pot(math.sin(elev)) if elev > 0 else 0.0)
            for k in range(4):
                out[k][d][h] = v[k]
    return out


def _ks_row(sun, raz):
    """[367][24] cos and tan of the shadow angle (calc_shadow_angle, crop_orchard_Kemanian_Stockle.cpp:
    68-78) for a row azimuth; 1 and 0 with the sun down (unused)."""
    up, _, elev_t, az_t = sun
    cs = [[1.0] * 24 for _ in range(367)]
    tn = [[0.0] * 24 for _ in range(367)]
    for d in range(1, 367):
        for h in range(24):
            if up[d][h] > 0.5:
                s1 = abs(math.sin(az_t[d][h] - raz))
                sa = 1.570796 if s1 == 0 else math.atan(math.tan(elev_t[d][h]) / s1)
                cs[d][h], tn[d][h] = math.cos(sa), math.tan(sa)
    return cs, tn


class ArchStatic(NamedTuple):
    """Per-tile site of the interception models and the setup tables. Rows: li per latitude, si per
    site (latitude, slope, aspect), ri per (latitude, longitude); the second axis of the row tables
    is the crop's az_idx."""
    model: torch.Tensor             # [T] long NONE / KEMANIAN / OYARZUN
    cloud: torch.Tensor             # [T] cloud cover fraction of the first day (Kemanian-Stockle)
    li: torch.Tensor                # [T] long
    si: torch.Tensor                # [T] long
    ri: torch.Tensor                # [T] long
    rise_set: torch.Tensor          # [n_ll, 367, 2] long sunrise / sunset hour
    frac: torch.Tensor              # [n_lat, 367, 24] insolation fraction
    etday: torch.Tensor             # [n_lat, 367] MJ m-2 extraterrestrial radiation
    oy: torch.Tensor                # [n_lat, 4, 367, 24] tan z, sin z, cos z, |z(h-1) - z(h)| (Oyarzun-Stockle)
    oy_row: torch.Tensor            # [n_lat, n_az, 2, 367, 24] |sin(az - row az)|, |cos(az - row az)|
    scf: torch.Tensor               # [n_site, 367, 24] shadow correction factor (Oyarzun-Stockle)
    ks: torch.Tensor                # [n_lat, 4, 367, 24] sun up, sine elevation, elevation, azimuth
    ks_site: torch.Tensor           # [n_site, 4, 367, 24] equivalent elevation, correction, potentials
    ks_row: torch.Tensor            # [n_lat, n_az, 2, 367, 24] cos, tan of the shadow angle
    n_az: int = 1                   # distinct row azimuths (second axis of the row tables)
                                    # (all tables are stored flattened day-major, see arch_static)


def arch_static(model, tile_lat, tile_lon, slope_percent, aspect_deg, srad0_mj, doy0, AP: ArchParams,
                device, dtype=torch.float64) -> ArchStatic:
    """Setup (host): model label (one per run or per tile), tile latitude/longitude (degrees), slope
    percent and aspect (scalars or [T]), the first day's radiation [T] (MJ m-2) and day of year, the
    per-crop architecture (its distinct row azimuths)."""
    lat32 = _f32(torch.as_tensor(tile_lat, dtype=torch.float64).reshape(-1).cpu())
    T = lat32.shape[0]
    lon32 = _f32(torch.as_tensor(tile_lon, dtype=torch.float64).reshape(-1).cpu().expand(T).contiguous())
    labels = [model] * T if isinstance(model, str) else list(model)
    for m in labels:
        if m not in MODELS:
            raise ValueError(f"cropsyst: orchard_interception_model {m!r} not one of {MODELS}")
    code = [MODELS.index(m) for m in labels]
    use_oy, use_ks = OYARZUN in code, KEMANIAN in code
    lats, lons = lat32.tolist(), lon32.tolist()
    # terrain (crop_orchard_common.cpp:9, 126-133): slope asin(percent / 100), aspect from south
    slope = [math.asin(x / 100) for x in torch.as_tensor(slope_percent, dtype=torch.float64).expand(T).tolist()]
    aspect = (_aspect_s0(torch.as_tensor(aspect_deg, dtype=torch.float64).expand(T)) * DEG_TO_RAD).tolist()
    raz = sorted(set(AP.row_az.cpu().tolist()))
    ulat = sorted(set(lats))
    usite = sorted(set(zip(lats, slope, aspect)))
    ull = sorted(set(zip(lats, lons)))
    li = [ulat.index(v) for v in lats]
    si = [usite.index(v) for v in zip(lats, slope, aspect)]
    ri = [ull.index(v) for v in zip(lats, lons)]
    sun = [_sun_tables(v) for v in ulat]
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    frac, etday = t([s[2] for s in sun]), t([s[3] for s in sun])
    empty = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    oy = oy_row = scf = ks = ks_site = ks_row = empty
    if use_oy:
        oyl, oyr = [], []
        for (Z, A, _, _) in sun:
            tz = [[math.tan(z) for z in r] for r in Z]
            sz = [[math.sin(z) for z in r] for r in Z]
            cz = [[math.cos(z) for z in r] for r in Z]
            dz = [[abs(r[h - 1] - r[h]) if h else 0.0 for h in range(24)] for r in Z]
            oyl.append([tz, sz, cz, dz])
            oyr.append([[[[abs(math.sin(a - rz)) for a in r] for r in A],
                         [[abs(math.cos(a - rz)) for a in r] for r in A]] for rz in raz])
        oy, oy_row = t(oyl), t(oyr)
        scf = t([_scf_table(sun[ulat.index(la)][1], sl, asp) for la, sl, asp in usite])
    if use_ks:
        kss = [_ks_sun(v * DEG_TO_RAD) for v in ulat]
        ks = t(kss)
        ks_site = t([_ks_site(kss[ulat.index(la)], sl, asp) for la, sl, asp in usite])
        ks_row = t([[list(_ks_row(s, rz)) for rz in raz] for s in kss])
    rise_set = torch.tensor([_rise_set(la, lo) for la, lo in ull], dtype=torch.long)
    # Sky::calc_cloud_cover_fraction of the first day (sky.cpp:98-114, min transmission 0.4)
    s0 = torch.as_tensor(srad0_mj, dtype=torch.float64).reshape(-1).cpu().expand(T)
    r = s0 / etday[torch.tensor(li), int(doy0)]
    tc = torch.clamp(r, min=0.4).clamp(max=0.75)
    cloud = torch.clamp(2.33 - 3.33 * tc, min=0.0).clamp(max=1.0)
    L = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    to = lambda x: x.to(device=device)

    def flat(x, rows):
        """Day axis right after the row axes, rows flattened with it: [rows x 367, ...] (gathered by
        index_select with row * 367 + doy, no host sync)."""
        if x.shape[-1] != 24:
            return to(x.reshape(-1, 1))
        d = x.dim() - 2                                         # day axis of the stored layout
        perm = list(range(rows)) + [d] + [i for i in range(rows, x.dim()) if i != d]
        y = x.permute(*perm).contiguous()
        return to(y.reshape(-1, *y.shape[rows + 1:]))
    return ArchStatic(L(code), to(cloud), L(li), L(si), L(ri), to(rise_set.reshape(-1, 2)),
                      flat(frac, 1), to(etday.reshape(-1)), flat(oy, 1), flat(oy_row, 2), flat(scf, 1),
                      flat(ks, 1), flat(ks_site, 1), flat(ks_row, 2), len(raz))


# ----------------------------------------------------------------------------- state

class ArchState(NamedTuple):
    width: torch.Tensor             # current canopy width across the row (m)
    breadth: torch.Tensor           # along the row
    height: torch.Tensor
    begin: torch.Tensor             # deg_days_begin_canopy_development
    interception: torch.Tensor      # solid canopy interception (last update)


def initial_state(T, dtype=torch.float64, device=None) -> ArchState:
    z = torch.zeros(T, dtype=dtype, device=device)
    return ArchState(z, z, z, z, z)


def sow(a: ArchState, AP: ArchParams, AS: ArchStatic, m) -> ArchState:
    """Orchard_common created with the crop (perform_op.cpp:430-436): dimensions at their initial
    sizes, begin = flowering (crop_orchard_common.cpp:12-28, 52-74); the interception as the
    model objects start (none 1.0, Kemanian-Stockle 1.0, Oyarzun-Stockle 0.0)."""
    m = m & AP.on
    sci = torch.where(AS.model == OYARZUN, torch.zeros_like(a.width), torch.ones_like(a.width))
    w = lambda new, old: torch.where(m, new, old)
    return ArchState(w(AP.w0, a.width), w(AP.b0, a.breadth), w(AP.h0, a.height), w(AP.begin0, a.begin),
                     w(sci, a.interception))


def restart(a: ArchState, m, tt) -> ArchState:
    """set_deg_days_begin_canopy_development(accumulated thermal time) at the restart
    (crop_fruit.cpp:262-270, crop_orchard_common.cpp:100-110)."""
    return ArchState(a.width, a.breadth, a.height, torch.where(m, tt, a.begin), a.interception)


def _expand(cur, ini, fin, tt, begin, end, m):
    """Tree_canopy_extent::Dimension::expand (crop_orchard_common.cpp:76-87)."""
    go = m & (tt > begin) & (tt < end)
    den = torch.where(go, end - begin, torch.ones_like(end))
    return torch.where(go, ini + ((fin - ini) * (tt - begin) / den), cur)


def _take(table, row, doy):
    """table[row, doy] of a flattened [rows x 367, ...] table, by index_select (no host sync)."""
    return torch.index_select(table, 0, row * 367 + doy)


def _shade(tz, s1, s3, scf, H, bih, lw, lb, row, col):
    """calc_fraction_of_shade (crop_orchard_Oyarzun_Stockle.cpp:226-334): shadow lengths across and
    along the rows above the branch insertion height, slope-corrected, over the plot."""
    sl = torch.clamp((H * tz) * s1, min=0.0)
    gsl = torch.clamp((bih * tz) * s1, min=0.0)
    sla = torch.clamp((H * tz) * s3, min=0.0)
    gsla = torch.clamp((bih * tz) * s3, min=0.0)
    sh = (sl - gsl) * scf
    sha = (sla - gsla) * scf
    t1 = ((sh + lw) * (lb + sha) - (sh * sha)) / (row * col)
    return torch.clamp(t1, max=1.0)


def oyarzun(a: ArchState, AP: ArchParams, AS: ArchStatic, doy, srad):
    """Interception_architectural_Oyarzun_Stockle::update_solid_canopy_interception
    (crop_orchard_Oyarzun_Stockle.cpp:81-224) with Solar_radiation_hour::update
    (solar_irrad_hour.cpp:20-35). [T]."""
    o = _take(AS.oy, AS.li, doy)                                # [T, 4, 24]
    tz, sz, cz, dz = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
    r = _take(AS.oy_row, AS.li * AS.n_az + AP.az_idx, doy)      # [T, 2, 24]
    s1, s3 = r[:, 0], r[:, 1]
    scf = _take(AS.scf, AS.si, doy)                             # [T, 24]
    frac = _take(AS.frac, AS.li, doy)
    etd = _take(AS.etday, AS.li, doy)
    rs = _take(AS.rise_set, AS.ri, doy)
    sunrise, sunset = rs[:, 0:1], rs[:, 1:2]
    ex = lambda x: x[..., None]
    lw = torch.minimum(a.width, AP.row)
    lb = torch.minimum(a.breadth, AP.col)
    geo = (ex(a.height), ex(AP.bih), ex(lw), ex(lb), ex(AP.row), ex(AP.col))
    fs = _shade(tz, s1, s3, scf, *geo)                          # [T, 24]
    # diffuse transmittance of each hour h: local hours from sunrise + 1 to sunset, with hour h's
    # shadow correction (calc_daily_diffuse_transmittance_of_Rs_for_hour, :197-224)
    gh = tuple(x[..., None] for x in geo)
    fl = _shade(tz[:, None, :], s1[:, None, :], s3[:, None, :], scf[:, :, None], *gh)   # [T, h, lh]
    pseudo = (1.0 - fl) * sz[:, None, :] * cz[:, None, :] * dz[:, None, :]
    lh = torch.arange(24, device=frac.device)
    use = ((lh >= sunrise + 1) & (lh <= sunset) & (lh >= 1))[:, None, :]
    s = torch.zeros_like(fs)
    for k in range(1, 24):                                      # sequential (static) sum
        s = torch.where(use[..., k], s + pseudo[..., k], s)
    adt = torch.clamp(2.0 * s, max=0.9)
    fd = 1.0 - torch.exp(1.0 - 0.96 / torch.clamp(srad / etd, max=0.8))
    sr = ex(srad) * frac
    dif = sr * ex(fd)
    beam = sr - dif
    tot = torch.clamp(beam + dif, min=0.000001)
    hourly = fs * (beam / tot) + (1.0 - adt) * (dif / tot)
    daily = torch.zeros_like(srad)
    for h in range(24):
        daily = torch.where(frac[..., h] > 0.001, daily + frac[..., h] * hourly[..., h], daily)
    return daily


def kemanian(a: ArchState, AP: ArchParams, AS: ArchStatic, doy):
    """Interception_architectural_Kemanian_Stockle::update_solid_canopy_interception
    (crop_orchard_Kemanian_Stockle.cpp:80-250). [T]."""
    k = _take(AS.ks, AS.li, doy)                                # [T, 4, 24]
    up, elev = k[:, 0] > 0.5, k[:, 2]
    q = _take(AS.ks_site, AS.si, doy)
    eq, cor, drp, frp = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = _take(AS.ks_row, AS.li * AS.n_az + AP.az_idx, doy)
    cs, tn = r[:, 0], r[:, 1]
    H, W, row = a.height[..., None], a.width[..., None], AP.row[..., None]
    shl = (H - W + W / cs) / tn                                 # calc_shadow_length (:80-88)
    csl = shl * cor                                             # calc_corrected_shadow_length (:123-140)
    csl = torch.where((csl < W) & (csl > 0), W + torch.zeros_like(csl), csl)
    csl = torch.clamp(csl, min=0.0)
    z = torch.zeros_like(csl)
    csl = torch.where(eq > 0, csl, z)
    dsl = torch.where(elev > 0, shl, z)
    dri = torch.where(up, torch.clamp(csl / row, min=0.0).clamp(max=1.0), z)
    fri = torch.where(up, torch.clamp(dsl / row, min=0.0).clamp(max=1.0), z)
    wd = sd = wf = sf = torch.zeros_like(a.width)
    for h in range(24):                                         # get_daily_interception (:203-236)
        wd = wd + dri[..., h] * drp[..., h]
        sd = sd + drp[..., h]
        wf = wf + fri[..., h] * frp[..., h]
        sf = sf + frp[..., h]
    dfr = torch.clamp(0.3 + 0.7 * AS.cloud, min=0.0).clamp(max=1.0)
    return (wd / sd) * (1.0 - dfr) + (wf / sf) * dfr


def update(a: ArchState, AP: ArchParams, AS: ArchStatic, on, tt, doy, srad) -> ArchState:
    """Crop_orchard_fruit::process_day, first part (crop_fruit.cpp:659-671): expand_canopy with the
    accumulated thermal time, update_solid_canopy_interception. on: [T] the fruit crop is present;
    AP gathered to the tiles; doy 0-d long."""
    m = on & AP.on
    tree = m & (AS.model != NONE)                               # a Tree_canopy_extent exists
    w = _expand(a.width, AP.w0, AP.w1, tt, a.begin, AP.end, tree)
    b = _expand(a.breadth, AP.b0, AP.b1, tt, a.begin, AP.end, tree)
    h = _expand(a.height, AP.h0, AP.h1, tt, a.begin, AP.end, tree)
    a2 = ArchState(w, b, h, a.begin, a.interception)
    sci = torch.ones_like(w)
    if AS.oy.shape[-1] == 24:                                   # a tile of the run uses Oyarzun-Stockle
        sci = torch.where(AS.model == OYARZUN, oyarzun(a2, AP, AS, doy, srad), sci)
    if AS.ks.shape[-1] == 24:                                   # ... Kemanian-Stockle
        sci = torch.where(AS.model == KEMANIAN, kemanian(a2, AP, AS, doy), sci)
    return ArchState(w, b, h, a.begin, torch.where(m, sci, a.interception))
