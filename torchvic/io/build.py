"""Build what `basin.simulate` consumes from a VIC 5 image-driver set-up:
the batch (ImageBatch), the atmosphere (build_atmosphere), the soil column
parameters (build_soil) and the vegetation mosaic (build_mosaic), from the
files VIC itself reads -- domain, parameters, forcing -- so a user with a
working image set-up gets a run without a conversion step.

Nothing is estimated that the forcing measures: longwave, pressure and wind
are the file's own where it carries them, and shortwave is the 24-hour mean
VIC forcing already is.
"""
import math
import os
import warnings

import torch

from ..core import atmos, column, version as core_version, snow, soil
from ..model.basin import EARTH_RADIUS_M
from ..core.vegetation import VegMosaic, root_fractions

#: vic_def.h:73, the floor VIC puts under a canopy fraction so that
#: dividing LAI by it cannot blow up.
MIN_FCANOPY = 1.0e-4
#: param.ALBEDO_BARE_SOIL, initialize_parameters.c:48.
ALBEDO_BARE_SOIL = 0.2


def band_mean_elevation(params, snow_band=True, classic=False):
    """The elevation VIC actually uses, which is not the file's `elev`.

    With `snow_band` False (SNOW_BAND FALSE in the image global file, VIC's
    default) VIC reads no bands and keeps the file's `elev`: vic_init.c
    5.1.0:957-962 sets BandElev[0] = elevation and never resets it. The
    band mean below is the SNOW_BAND TRUE case (vic_init.c 5.1.0:1020-1034).

    With bands the image driver replaces the cell elevation by the
    area-weighted mean of the band elevations, and logs a warning for every
    cell it changes, for example on Walnut Gulch: "average band elevation
    1241.300000 not equal to grid_cell average elevation 1283.400000; setting
    grid cell elevation to average band elevation." The two releases reach
    that differently:

    * VIC 5.0.1 (tag VIC.5.0.1, vic_init.c:1003-1007) tests

          if (fabs(soil_con[i].elevation - soil_con[i].BandElev[j]) > 1.0)

      where `j` is `options.SNOW_BAND`: one past the last band written, a read
      out of the loop's range. `BandElev` beyond the declared bands is zero,
      so the test is true for any cell above 1 m and the cell elevation is
      replaced by the band mean every time. That is an upstream defect.
    * VIC 5.1.0 replaced that comparison (upstream commit 070bd929,
      2017-03-15, in 5.1.0 rc1) by one against the area-weighted band mean
      itself, with a tolerance of 1e-20. Replacing the cell elevation by the
      band mean is there documented, intended behaviour (the 5.1.0 release
      notes, GH#701).

    Both releases end up with the band mean, so this function returns it.
    Using the file's own `elev` instead would disagree with VIC by about 40 m
    everywhere at this site, which moves air pressure, air density and the
    psychrometric constant in penman, and the band temperature lapse.

    The classic driver (`classic=True`; read_snowband.c:79-85) replaces the
    elevation only when the band mean is more than 1 m away from it.
    """
    import numpy as np
    bands = getattr(params, 'bands', None) or {}
    if not snow_band or 'elevation' not in bands or 'AreaFract' not in bands:
        return np.asarray(params.soil['elev'])
    elev = np.asarray(bands['elevation'])          # [band, cell]
    frac = np.asarray(bands['AreaFract'])
    mean = (elev * frac).sum(axis=0)
    if classic:
        # read_snowband.c:79-85: the classic driver replaces the cell's
        # elevation only when the band mean is more than 1 m away from it.
        cell = np.asarray(params.soil['elev'], dtype=np.float64)
        mean = np.where(np.abs(mean - cell) > 1.0, mean, cell)
    return mean


def pick_device(prefer=None):
    """The least-loaded CUDA device, or what the caller asked for.

    A shared machine usually has one card busy with somebody else's work and
    another sitting idle. Taking cuda:0 unconditionally puts this model behind
    whatever is already there, and at this domain size the model is
    launch-bound rather than compute-bound, so it is another process's kernels
    it waits behind rather than its own arithmetic.

    Free memory is the signal used, because it is the one thing available
    without nvidia-smi and it tracks "somebody else is here" closely enough
    for this purpose. `prefer` wins outright when given, so an explicit
    --device is never second-guessed.
    """
    import torch as _t
    if prefer not in (None, 'cuda', 'auto'):
        return _t.device(prefer)
    if not _t.cuda.is_available():
        return _t.device('cpu')
    n = _t.cuda.device_count()
    if n == 1:
        return _t.device('cuda:0')
    free = []
    for i in range(n):
        try:
            f, _tot = _t.cuda.mem_get_info(i)
        except Exception:                                  # noqa: BLE001
            f = 0
        free.append((f, i))
    return _t.device('cuda:%d' % max(free)[1])


class ImageBatch(object):
    """What `basin.simulate` needs from a domain, and nothing more.

    An image domain is a grid with a mask; it may have no gauge at all.
    This carries the grid, its cells' latitude, elevation and area, and the
    routing map to the outlet(s).
    """

    def __init__(self, domain, dates, grid_deg=None, dtype=torch.float64,
                 device=None, elevation=None):
        kw = dict(dtype=dtype, device=device)
        self.domain = domain
        self.dates = dates
        self.dtype, self.device = dtype, device
        self.n_cells = int(domain.n_active)
        self.n_days = len(dates)
        self.latitude = torch.as_tensor(domain.lat, **kw)
        self.longitude = torch.as_tensor(domain.lon, **kw)
        if elevation is None:
            raise ValueError('elevation is required; it comes from the '
                             'parameter file (`elev`), not the domain')
        self.elevation = torch.as_tensor(elevation, **kw)

        # The domain file states each cell's area in m2. Prefer it over a
        # spherical formula: it is what VIC itself uses to weight the cell,
        # and a domain built from a projected grid need not match the formula.
        if getattr(domain, 'area', None) is not None:
            self.cell_area = torch.as_tensor(domain.area, **kw)
            self.area_source = 'domain file'
        else:
            g = grid_deg or self._infer_grid_deg()
            half = math.radians(g / 2.0)
            latr = self.latitude * math.pi / 180.0
            self.cell_area = (EARTH_RADIUS_M ** 2 * math.radians(g)
                              * (torch.sin(latr + half)
                                 - torch.sin(latr - half)))
            self.area_source = 'spherical, %.6f deg' % g

        # One outlet for the whole domain: every cell drains to it. Without a
        # flow-direction field there is no network to route through, so this
        # is an area-weighted basin total, not a routed hydrograph, and the
        # output says so.
        self.n_outlets = 1
        self.source2outlet = torch.zeros(self.n_cells, dtype=torch.long,
                                         device=device)
        # With a flow network (domain `flow_distance`), each cell gets its
        # own Lohmann unit hydrograph to the outlet in basin.simulate.
        fd = getattr(domain, 'flow_distance', None)
        self.flow_distance = (None if fd is None
                              else torch.as_tensor(fd, **kw))
        self.t0, self.t1 = 0, self.n_days
        self.cell_index = torch.arange(self.n_cells, device=device)

    def _infer_grid_deg(self):
        import numpy as np
        lat = np.unique(self.domain.lat)
        return float(np.diff(lat)[0]) if lat.size > 1 else 1.0 / 16.0

    @property
    def area_m2(self):
        return float(self.cell_area.sum())

    def __repr__(self):
        return ('<ImageBatch %d cells x %d steps, area %.1f km2 (%s)>'
                % (self.n_cells, self.n_days, self.area_m2 / 1e6,
                   self.area_source))


def build_soil(params, dtype=torch.float64, device=None,
               runoff_steps_per_day=24, model_steps_per_day=24,
               baseflow='ARNO'):
    """`column.SoilParams` from the image parameter file, VIC's own names.

    `infilt` in the file is VIC's b_infilt. The maximum and residual moisture
    are absolute millimetres derived exactly as `read_soilparam.c` does, from
    depth and the bulk/soil density pair, rather than assumed.

    Under BASEFLOW NIJSSEN2001 the file's Ds, Dsmax, Ws, c are d1..d4 and
    are converted to ARNO's as VIC converts them on read (image driver
    vic_init.c:815-829, classic driver read_soilparam.c:693-707), with the
    bottom layer's maximum moisture. Both drivers reach this builder, so
    the conversion lives here and nowhere else.
    """
    def t(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    s = params.soil
    max_moist = t(params.max_moist()).T.contiguous()         # [cell, layer]
    # resid_moist is a fraction of the layer in the file and millimetres here.
    # VIC 5.0.1 forms the millimetres where it uses them, as
    # resid * depth * MM_PER_M, left to right (runoff.c:86-87,
    # arno_evap.c:172-175). VIC 5.1.0 premultiplies once at load,
    # `resid_moist[j] *= depth[j] * MM_PER_M` (vic_init.c:775), which is
    # resid * (depth * 1000): the same number to within an ulp, rounded
    # differently. VIC-WUR 3.0.0 keeps the fraction (vic_init.c:815-816)
    # and forms the millimetres where it uses them as 5.0.1 does
    # (VIC-WUR 3.0 runoff.c:88-89, arno_evap.c:172-175).
    if core_version.at_least('5.1.0') and not core_version.vicwur3():
        resid = t(s['resid_moist']).T.contiguous() * (
            t(s['depth']).T.contiguous() * 1000.0)
    else:
        resid = t(s['resid_moist']).T.contiguous() \
            * t(s['depth']).T.contiguous() * 1000.0
    w_crit, w_wilt = params.critical_moisture()
    # The bubbling pressure is not in ImageParams.SOIL_3D -- that list is
    # what the column consumes -- so it is taken from the open dataset by the
    # same `domain.take` the reader uses for everything else, in the file's
    # own cm. A file without it gets no water table, which column.step
    # reports as an absent diagnostic rather than a guessed one. The classic
    # reader (ClassicParams) has no dataset at all -- its `ds` is None --
    # and its soil table carries no bubbling pressure, so it takes that
    # same absent-diagnostic path.
    _ex = getattr(params, 'extra', None) or {}
    bubble = None
    if 'bubble' in s:
        bubble = t(s['bubble']).T.contiguous()
    elif 'bubble' in _ex:
        # The classic reader keeps columns the image layout has no slot
        # for in `extra`, and the bubbling pressure is one of them. It is
        # read (read_soilparam.c:287-322) and it is in the file, so an
        # ASCII set-up should get the water-table diagnostic and
        # FROZEN_SOIL's unfrozen-water curve like any other.
        bubble = t(_ex['bubble']).T.contiguous()
    elif getattr(params, 'bubble', None) is not None:
        # ImageParams: taken from the file at read (and cached with the rest)
        bubble = t(params.bubble).T.contiguous()
    elif (not hasattr(params, 'bubble')
          and getattr(params, 'ds', None) is not None
          and 'bubble' in params.ds):
        bubble = t(params.domain.take(params.ds['bubble'])).T.contiguous()
    Ds, Dsmax, Ws, c = (t(s['Ds']), t(s['Dsmax']), t(s['Ws']),
                        t(s['c']))
    if str(baseflow).upper() == 'NIJSSEN2001':
        Ds, Dsmax, Ws, c = soil.nijssen2001_to_arno(
            Ds, Dsmax, Ws, c, max_moist[:, -1])
    sp = column.SoilParams(
        b_infilt=t(s['infilt']), Ds=Ds, Dsmax=Dsmax, Ws=Ws, c=c,
        expt=t(s['expt']).T.contiguous(),
        Ksat=t(s['Ksat']).T.contiguous(),
        max_moist=max_moist, resid_moist=resid,
        runoff_steps_per_day=runoff_steps_per_day,
        model_steps_per_day=model_steps_per_day,
        # Read by the water-table diagnostic only (column.wrap_compute_zwt,
        # OUT_ZWT), which refuses to run without them rather than assume a
        # bubbling pressure. The file carries both, in VIC's own units: depth
        # in m and bubble in cm (read_soilparam.c reads them unscaled).
        depth=t(s['depth']).T.contiguous(), bubble=bubble)
    # Carried so the canopy's moisture stress and the column agree on one
    # soil rather than describing it twice.
    sp.w_crit_mm = t(w_crit).T.contiguous()
    sp.w_wilt_mm = t(w_wilt).T.contiguous()
    # What the surface energy balance reads from the soil file and nothing
    # else does: the deep temperature the profile is asymptotic to, the
    # damping depth, and the thermal-property inputs. Attached rather than
    # added to SoilParams' slots so the column is untouched.
    sp.thermal = dict(
        avg_temp=t(s['avg_T']), dp=t(s['dp']),
        depth=t(s['depth']).T.contiguous(),
        bulk_density=t(s['bulk_density']).T.contiguous(),
        soil_density=t(s['soil_density']).T.contiguous(),
        quartz=t(s['quartz']).T.contiguous(),
        rough=t(s['rough']),
        # CalcAerodynamic.c's snow-surface resistance takes the cell's snow
        # roughness (soil_con->snow_rough)
        snow_rough=t(s['snow_rough']) if 'snow_rough' in s else None)
    # ORGANIC_FRACT / BULK_DENSITY_COMB (ImageParams.apply_density_options):
    # the thermal properties take the mineral densities and the organic
    # fraction beside the combined pair (soil_conduction.c:17-99)
    if 'bulk_dens_min' in s:
        for _k in ('bulk_dens_min', 'soil_dens_min', 'organic'):
            sp.thermal[_k] = t(s[_k]).T.contiguous()
    # SPATIAL_SNOW's only soil-file input: the deepest linear depth
    # distribution the pack is spread over (vic_def.h:584). The classic
    # reader takes it as an extra soil column when the global file sets the
    # option (read_soilparam.c:542-552); the image driver reads the
    # parameter file's max_snow_distrib_slope under the option
    # (vic_init.c 5.1.0:691-700, 5.0.1 the same shape) and stops on a
    # negative value (:799-802). The classic reader keeps columns the image
    # layout has no slot for in `extra` rather than in `soil`, so that
    # `soil` has the same key set on both paths; this is one of them.
    _extra = getattr(params, 'extra', None) or {}
    if 'max_snow_distrib_slope' in s:
        sp.max_snow_distrib_slope = t(s['max_snow_distrib_slope'])
    elif 'max_snow_distrib_slope' in _extra:
        sp.max_snow_distrib_slope = t(_extra['max_snow_distrib_slope'])
    # FS_ACTIVE, the soil file's per-cell frozen-soil switch.
    for _src in (s, _extra):
        if 'fs_active' in _src:
            sp.fs_active = t(_src['fs_active']) > 0.5
            break
    # SPATIAL_FROST's one soil-file input, read when present; the option
    # refuses to run without it (case.check_inputs).
    for _src in (s, _extra):
        if 'frost_slope' in _src:
            sp.frost_slope = t(_src['frost_slope'])
            break
    return sp


def build_mosaic(params, dtype=torch.float64, device=None, log=None,
                 fcan_src='FROM_DEFAULT', max_snow_albedo=False,
                 keep_raw=False, nbare=1):
    """`VegMosaic` straight from the parameter file's veg_class dimension.

    Tiles come as a full [veg_class, cell] cover array with zeros where a
    class is absent, which is already the padded layout the mosaic wants, so
    no ragged structure is built and then flattened.

    Two kinds of missing value have to be told apart, because they mean
    opposite things:

    * A class with **zero cover** may carry NaN in every field -- MOD_IGBP
      leaves all fifteen unset for the fourteen classes Walnut Gulch does not
      have. That is padding. NaN times zero cover is still NaN and would
      poison the cell, so those entries take a finite value the zero cover
      then discards. Nothing is invented: the tile contributes nothing either
      way.
    * A class with **real cover and no parameters** is a defect in the file,
      not padding. MOD_IGBP gives Walnut Gulch up to 1.5 per cent Urban cover
      with LAI, roots, rmin, rarc and RGL all unset. Substituting numbers
      there would be inventing vegetation that the file does not describe, so
      the tile is dropped and the cell's remaining cover renormalised to one
      -- the water balance needs a fully tiled cell -- and exactly how much
      area moved is reported rather than absorbed silently.
    """
    import numpy as np

    nbare = int(nbare)
    if nbare != 1 and not core_version.vicwur3():
        raise ValueError('NBARE %d is VIC-WUR\'s (get_global_param.c:'
                         '76-77 of VIC-WUR 3.0); it needs the vicwur3 core' % nbare)

    def t(a):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    v = params.veg
    cv = t(v['Cv']).T.contiguous()                     # [cell, tile]

    # Cover that is not a number is not cover. VIC's own image parameter
    # files write 0 into the vegetation classes a cell does not have; the
    # Indian 6 km file writes NaN, and so does anything built with xarray
    # from a per-cell class list, because an absent class has no value to
    # write and NaN is what fills the rest of the rectangle. The two files
    # say the same thing, but NaN says it in a way that spreads: a tile
    # weighted by NaN contributes NaN to every sum it enters however finite
    # everything else is, so a cell with two real classes out of seventeen
    # returns NaN for evaporation, runoff, storage and the water balance
    # alike -- which is what it did, on all 100 cells, before this.
    #
    # Coercing NaN to zero is only correct where the finite cover already
    # accounts for the whole cell. That is the test, applied per cell, and
    # the two outcomes are reported apart: an absent class is padding and
    # is silent once counted, while a cell whose finite cover does not
    # reach one is a file missing vegetation, and completing it with bare
    # soil would be inventing the land surface rather than reading it.
    nan_cv = ~torch.isfinite(cv)
    if bool(nan_cv.any()):
        zero = torch.zeros_like(cv)
        finite_cover = torch.where(nan_cv, zero, cv).sum(1)
        complete = (finite_cover - 1.0).abs() < 1e-4
        cv = torch.where(nan_cv, zero, cv)
        n_pad = int((nan_cv & complete.unsqueeze(1)).sum())
        n_cell = int(nan_cv.any(1).sum())
        (log or warnings.warn)(
            '  %s marks absent vegetation classes with NaN rather than 0; '
            '%d such tiles over %d cells read as no cover'
            % (os.path.basename(params.path), n_pad, n_cell))
        short = ~complete & nan_cv.any(1)
        if bool(short.any()):
            worst = float(finite_cover[short].min())
            raise ValueError(
                '%s: %d cell(s) have NaN vegetation cover AND finite cover '
                'that does not sum to one (worst %.6f). NaN cannot be read '
                'as an absent class there -- the cover is genuinely missing, '
                'and filling it would invent vegetation. Fix the file or '
                'exclude these cells.'
                % (os.path.basename(params.path), int(short.sum()), worst))
    # The classes VIC maps to tiles (vic_init.c:1094-1098, Cv > 0), before
    # anything below drops one; the bare-soil tile below is counted on them.
    covered_file = cv > 0.0
    lai = t(v['LAI']).permute(2, 0, 1).contiguous()    # [cell, tile, month]

    # A root *zone* is not a soil *layer*. The parameter file gives root
    # fractions over zones of its own choosing (0.1/1.0/0.5 m at Walnut
    # Gulch); the soil column has its own layers (0.1/0.3/1.0 m). VIC computes
    # the overlap once in calc_root_fraction.c and it is not the identity:
    # grassland's [0.10, 0.70, 0.20] over zones is [0.10, 0.21, 0.69] over
    # layers. Passing the zone fractions straight through gave soil layer 2 --
    # 129 mm of capacity -- 3.3 times the root share VIC gives it, and layer 3
    # -- 430 mm -- 0.29 times, which draws transpiration from the small store
    # instead of the large one. It went unnoticed because n_root and n_layer
    # are both 3 in every VIC file, so the array shape is legal either way.
    # FCAN_SRC, get_global_param.c:82. FROM_VEGPARAM / FROM_VEGLIB take the
    # file's monthly values. FROM_DEFAULT is not "fcanopy 1 everywhere":
    # vic_init.c:262-273 gives every class 1.0 except the last, which it
    # takes to be bare soil and gives MIN_FCANOPY, and the classic driver
    # gives its own bare-soil tile MIN_FCANOPY too (read_veglib.c:219). So
    # VIC evaporates that class almost wholly as ARNO bare soil
    # (func_surf_energy_bal.c:762-783) and this model, reading 1, did not
    # evaporate it at all: basin 37 with 30% of class 13's cover moved to
    # India's class 17, daily water balance, OUT_EVAP_BARE 99.30 mm against
    # 0 (India's class 17 has cover in 26,675 cells). With no cell covering
    # the last class every covered tile has fcanopy 1, and None -- the
    # identity everywhere it is read -- is that same model at no cost.
    # Default FROM_DEFAULT so a caller that does not read the global file
    # gets VIC's default rather than a silent change of physics.
    fcanopy = None
    if str(fcan_src).upper() != 'FROM_DEFAULT' and 'fcanopy' in v:
        fcanopy = t(v['fcanopy']).permute(2, 0, 1).contiguous()
        # VIC 5.0.1 raises a canopy fraction below MIN_FCANOPY to it
        # (vic_force.c 5.0.1:386-395); VIC 5.1.0 instead sets it, and the
        # LAI, to 0 (vic_force.c 5.1.0:419-439, below), so it is not
        # floored here under that core. VIC-WUR 3.0.0 floors it as 5.0.1
        # does (VIC-WUR 3.0 vic_force.c:428-441).
        fcanopy = torch.clamp(torch.nan_to_num(fcanopy, nan=1.0),
                              min=(None if (core_version.at_least('5.1.0')
                                            and not core_version.vicwur3())
                                   else MIN_FCANOPY), max=1.0)
    zone_frac = t(v['root_fract']).permute(2, 0, 1).contiguous()
    if 'root_depth' in v:
        zone_depth = t(v['root_depth']).permute(2, 0, 1).contiguous()
        layer_depth = t(params.soil['depth']).T.contiguous()   # [cell, layer]
        if core_version.vicwur3():
            # VIC-WUR 3.0 calc_root_fraction.c:58-183, the zone walk with its
            # `zone++` at :137 (vegetation.root_fractions_w3)
            from ..core.vegetation import root_fractions_w3
            root = root_fractions_w3(zone_depth, zone_frac,
                                     layer_depth.unsqueeze(1))
            # a walk that left no roots stops VIC-WUR 3.0 (:173-178), checked
            # below on the vegetated tiles once the bare ones are known
            w3_rootless = covered_file & ~(root.sum(-1) > 0.0)
        else:
            root = root_fractions(zone_depth, zone_frac,
                                  layer_depth.unsqueeze(1))
    else:
        # Without zone depths the overlap cannot be computed. Say so rather
        # than silently treating zones as layers, which is the defect above.
        (log or warnings.warn)(
            '  %s has root_fract but no root_depth; root zones cannot be '
            'mapped onto soil layers and are used as layers directly, which '
            'is what VIC does NOT do' % os.path.basename(params.path))
        root = zone_frac

    missing = []

    def scalar(name, fill):
        if name not in v:
            missing.append(name)
            return torch.full_like(cv, fill)
        return t(v[name]).T.contiguous()

    rmin = scalar('rmin', 100.0)
    rgl = scalar('RGL', 100.0)
    rarc = scalar('rarc', 50.0)
    if missing:
        # In image mode VIC keeps the vegetation library in the parameter
        # file. A file without it is readable, and every class then shares one
        # resistance -- which is exactly the defect that once ran here for
        # weeks while the water balance closed to 3e-16. It may not be silent.
        (log or warnings.warn)(
            '  %s carries no %s; every vegetation class will share one value '
            'and the canopy will not be the file\'s'
            % (os.path.basename(params.path), ', '.join(missing)))

    # A tile is describable only if every field the canopy reads is finite.
    ok = (torch.isfinite(rmin) & torch.isfinite(rgl) & torch.isfinite(rarc)
          & torch.isfinite(lai).all(-1) & torch.isfinite(root).all(-1))
    undescribed = (cv > 0) & ~ok
    moved = float((cv * undescribed).sum())
    if moved > 0:
        # Renormalise only the cells that actually lost a tile. Doing it
        # domain-wide rewrote every cell's cover to sum to one, which turned a
        # legitimate bare-soil fraction elsewhere in the domain into
        # vegetation.
        touched = undescribed.any(1, keepdim=True)
        cv = torch.where(undescribed, torch.zeros_like(cv), cv)
        total = cv.sum(1, keepdim=True)
        cv = torch.where(touched, cv / total.clamp(min=1e-12), cv)
        n = int(undescribed.any(0).sum())
        msg = ('  %d vegetation class(es) carry cover but no parameters in '
               '%s; dropped and cover renormalised, %.4f of %d cells of area '
               'reassigned (%.3f%%)'
               % (n, os.path.basename(params.path), moved, cv.shape[0],
                  100.0 * moved / cv.shape[0]))
        (log or warnings.warn)(msg)

    fill = lambda a, x: torch.where(torch.isfinite(a), a,
                                    torch.full_like(a, x))
    def monthly(name, default):
        if name not in v:
            return None
        return fill(t(v[name]).permute(2, 0, 1).contiguous(), default)

    # read_vegparam.c:444-456, BLOWING's three per-tile fields. Present
    # only when the file was read with the option on.
    # read_veglib.c:162-192, CARBON's per-class library fields.
    carb = {}
    for _k in ('Ctype', 'MaxCarboxRate', 'MaxETransport', 'CO2Specificity', 'LightUseEff', 'NscaleFlag', 'Wnpp_inhib', 'NPPfactor_sat'):
        if _k in v:
            carb[_k] = fill(t(v[_k]).T.contiguous(), 0.0)
    blow = {}
    for _k in ('sigma_slope', 'lag_one', 'fetch'):
        if _k in v:
            blow[_k] = fill(t(v[_k]).T.contiguous(), 0.0)
    # vic_init.c:186-196: MAX_SNOW_ALBEDO reads the class's max_snow_albedo
    msa = None
    if max_snow_albedo:
        if 'max_snow_albedo' not in v:
            raise ValueError('MAX_SNOW_ALBEDO TRUE: the parameter file has '
                             'no max_snow_albedo, which VIC reads '
                             '(vic_init.c:186-196)')
        msa = fill(t(v['max_snow_albedo']).T.contiguous(), 0.85)
    # A tile with no cover is still evaluated. `torch.where` computes both
    # branches, so a zero-cover tile whose root fractions are all zero
    # divides by its own root sum and returns NaN, which then survives
    # multiplication by a cover of zero. Absent tiles therefore get a
    # *valid* profile -- all roots in the top layer -- rather than zeros.
    # It weighs nothing either way; the difference is finite versus NaN.
    root = fill(root, 0.0)
    rootless = root.sum(-1, keepdim=True) <= 0.0
    top = torch.zeros_like(root)
    top[..., 0] = 1.0
    root = torch.where(rootless, top, root)

    # vic_run's bare-soil tile, iveg == Nveg. calc_surf_energy_bal.c:165-175
    # sets VEG false there, so it has no canopy at all: nothing is
    # intercepted or transpired and func_surf_energy_bal.c:797-805
    # evaporates the whole tile as ARNO bare soil (Ra_used[0] still averaged
    # with the tile's fcanopy at :703), with ALBEDO_BARE_SOIL rather than the
    # class's albedo (vic_run.c:231-236) and its LAI never divided by
    # fcanopy (vic_run.c:133 stops at Nveg). Which tile that is depends on
    # the driver:
    #   * classic: the record read_veglib.c:210-229 appends, last here
    #     (vic_classic.py);
    #   * image: the tile at index Nveg of the cell's covered classes, and
    #     vic_alloc.c:128 allocates Nveg + 1 of them from the file's Nveg.
    #     Where Nveg counts every covered class (India's file, FindleyLake)
    #     that index holds no class and the last class runs as vegetation
    #     with its MIN_FCANOPY; where Nveg leaves the last covered class out
    #     (Stehekin's four cells with bare soil) that class is the bare tile.
    #     Basin 37 with India's class 17 added and Nveg left as it was:
    #     VIC's OUT_EVAP_BARE 99.30 mm, and 92.21 with the tile run as
    #     vegetation at fcanopy MIN_FCANOPY, 0 before that.
    classes = getattr(params, 'veg_classes', None)
    nveg = getattr(params, 'nveg', None)
    bare_tile = None
    if core_version.vicwur3() and nveg is not None and not classes:
        # VIC-WUR 3.0.0 (image driver): nv_active = Nveg + NBARE tiles
        # (vic_alloc.c:128-129), the covered classes in class order
        # (vic_init.c:1145-1183), and every tile at an index >= Nveg is bare
        # soil, iveg < Nveg false (vic_run.c:135,250;
        # calc_surf_energy_bal.c:172-182). NBARE 1 is 5.1.0's iveg == Nveg, the
        # `one_short` rule below; more covered classes than Nveg + NBARE stop
        # VIC (VIC-WUR 3.0 vic_init.c:1186-1191).
        import numpy as np
        nv = torch.as_tensor(np.asarray(nveg, dtype=np.float64),
                             dtype=dtype, device=cv.device)
        n_cov = covered_file.sum(1).to(dtype)
        if bool((n_cov > nv + nbare).any()):
            raise ValueError(
                '%s: %d cell(s) have more covered classes than Nveg + NBARE '
                '(%d); VIC-WUR stops (VIC-WUR 3.0 vic_init.c:1186-1191)'
                % (os.path.basename(params.path),
                   int((n_cov > nv + nbare).sum()), nbare))
        rank = torch.cumsum(covered_file.to(torch.long), 1) - 1
        bare_tile = covered_file & (rank >= nv.to(torch.long).unsqueeze(1))
        _dead = w3_rootless & ~bare_tile if 'root_depth' in v else None
        if _dead is not None and bool(_dead.any()):
            raise ValueError(
                '%s: %d vegetated tile(s) get no roots from VIC-WUR\'s '
                'calc_root_fraction (VIC-WUR 3.0 stops: calc_root_fraction.c'
                ':173-178), e.g. class %d of cell %d'
                % (os.path.basename(params.path), int(_dead.sum()),
                   int(torch.nonzero(_dead)[0, 1]),
                   int(torch.nonzero(_dead)[0, 0])))
        if not bool(bare_tile.any()):
            bare_tile = None
    elif classes and classes[-1].get('bare'):
        bare_tile = torch.zeros(cv.shape, dtype=torch.bool, device=cv.device)
        bare_tile[:, -1] = True
    elif nveg is not None:
        import numpy as np
        nv = torch.as_tensor(np.asarray(nveg, dtype=np.float64),
                             dtype=dtype, device=cv.device)
        n_cov = covered_file.sum(1).to(dtype)
        k = torch.arange(cv.shape[1], device=cv.device)
        last_cov = torch.where(covered_file, k, torch.full_like(k, -1)
                               ).max(1).values
        one_short = (nv == n_cov - 1.0) & (n_cov > 0)
        if bool(one_short.any()):
            bare_tile = torch.zeros(cv.shape, dtype=torch.bool,
                                    device=cv.device)
            bare_tile[one_short, last_cov[one_short]] = True
            (log or warnings.warn)(
                '  %s: %d cell(s) have Nveg one below their covered classes; '
                'the last covered class runs as VIC\'s bare-soil tile there '
                '(iveg == Nveg, VEG false)'
                % (os.path.basename(params.path), int(one_short.sum())))
        if bool((nv < n_cov - 1.0).any()):
            (log or warnings.warn)(
                '  %s: %d cell(s) have Nveg below their covered classes minus '
                'one; VIC 5 allocates Nveg + 1 tiles (vic_alloc.c:128) and '
                'writes past them there'
                % (os.path.basename(params.path),
                   int((nv < n_cov - 1.0).sum())))
    if (fcanopy is None and str(fcan_src).upper() == 'FROM_DEFAULT'
            and (bool((cv[:, -nbare:] > 0.0).any())
                 or (bare_tile is not None
                     and bool((bare_tile & (cv > 0.0)).any())))):
        # vic_init.c:262-273 (see above): 1.0, and MIN_FCANOPY for the last
        # class, every month. Built too where a bare-soil tile has cover,
        # whose evaporation the water-balance path forms on the fractional
        # canopy's bare-soil branch.
        fcanopy = torch.ones_like(lai)
        # VIC-WUR 3.0 vic_init.c:280-292: the last NBARE classes (NBARE is 1
        # but under the vicwur3 core)
        fcanopy[:, -nbare:, :] = MIN_FCANOPY

    lai = fill(lai, 0.0)
    raw_monthly = None
    if keep_raw:
        # The monthly LAI and fcanopy as vic_force.c:241-262 copies them into
        # veg_hist, before the :419-439 leafless rule below: a *_SRC
        # FROM_VEGHIST history replaces one of the two per step, and the rule
        # then runs on the mix (torchvic.io.veghist).
        fc_raw = (fcanopy.clone() if fcanopy is not None
                  else torch.ones_like(lai))
        if fcanopy is None and str(fcan_src).upper() == 'FROM_DEFAULT':
            fc_raw[:, -nbare:, :] = MIN_FCANOPY
        raw_monthly = (lai.clone(), fill(fc_raw, 1.0))
    if core_version.at_least('5.1.0') and not core_version.vicwur3():
        # VIC 5.1.0 vic_force.c:417-440 (image) and :243-252 (classic), every
        # step and every tile the cell maps, GH#823:
        #     if (fcanopy < MIN_FCANOPY || LAI == 0) { fcanopy = 0; LAI = 0; }
        # A class in a month with no leaves has no canopy at all -- VEG is
        # false (calc_surf_energy_bal.c 5.1.0:145-155), the whole tile is
        # ARNO bare soil (func_surf_energy_bal.c 5.1.0:730-761) -- where
        # 5.0.1 ran it as a closed canopy of zero leaf area that evaporated
        # nothing from its soil. FROM_DEFAULT's MIN_FCANOPY for the last
        # class (vic_init.c 5.1.0:255-268) goes to 0 too where its LAI is 0.
        # India's class 0 is leafless all year (9.5% of 13 cells of basin
        # 37). The monthly values are the ones vic_force copies to every
        # sub-step (:240-262), so the rule is applied to them once here.
        fc_all = (fcanopy if fcanopy is not None
                  else torch.ones_like(lai))
        if fcanopy is None and str(fcan_src).upper() == 'FROM_DEFAULT':
            fc_all = fc_all.clone()
            fc_all[:, -1, :] = MIN_FCANOPY
        off = (fc_all < MIN_FCANOPY) | (lai == 0.0)
        if bool((off & (cv > 0.0).unsqueeze(-1)).any()):
            fcanopy = torch.where(off, torch.zeros_like(fc_all), fc_all)
            lai = torch.where(off, torch.zeros_like(lai), lai)
    if core_version.at_least('5.1.0'):
        # (VIC-WUR 3.0.0 has no leafless rule above -- vic_force.c:428-441
        # floors fcanopy at MIN_FCANOPY as 5.0.1 did -- and divides by the sum
        # always, vic_init.c:1288-1307: x / 1 is x.)
        # vic_init.c 5.1.0:1255-1278: the tile fractions are divided by
        # their sum whenever it is not 1 to within AREA_SUM_ERROR_THRESH
        # (1e-20, vic_driver_shared_image.h:18), i.e. whenever it is not 1
        # exactly; 5.0.1 accepted any sum within 0.001 (vic_init.c
        # 5.0.1:1231) and left it. Summed in class order as the C does
        # (:1207-1256); an absent class adds an exact 0. The image driver
        # only: the classic driver's read_vegparam.c is unchanged here, and
        # its reader has no dataset (`ds` None).
        if (getattr(params, 'has_dataset', False)
                or getattr(params, 'ds', None) is not None):
            cv_sum = cv[:, 0]
            for j in range(1, cv.shape[1]):
                cv_sum = cv_sum + cv[:, j]
            renorm = (cv_sum != 1.0) & (cv_sum > 0.0)
            safe = torch.where(renorm, cv_sum, torch.ones_like(cv_sum))
            cv = torch.where(renorm.unsqueeze(1), cv / safe.unsqueeze(1), cv)

    mosaic = VegMosaic(cv=cv, lai=lai, rmin=fill(rmin, 100.0),
                     rgl=fill(rgl, 100.0), rarc=fill(rarc, 50.0),
                     root_frac=root, dtype=dtype, device=device,
                     # The aerodynamic and radiative properties the file has
                     # carried all along and nothing read.
                     veg_rough=monthly('veg_rough', 0.03),
                     displacement=monthly('displacement', 0.0),
                     albedo=_bare_albedo(monthly('albedo', 0.20), bare_tile),
                     wind_h=fill(scalar('wind_h', 10.0), 10.0),
                     rad_atten=fill(scalar('rad_atten', 0.5), 0.5),
                     overstory=fill(scalar('overstory', 0.0), 0.0),
                     # veg library defaults where a file carries neither
                     wind_atten=fill(scalar('wind_atten', 0.5), 0.5),
                     trunk_ratio=fill(scalar('trunk_ratio', 0.2), 0.2),
                     fcanopy=(None if fcanopy is None
                              else fill(fcanopy, 1.0)),
                     bare_tile=bare_tile,
                     sigma_slope=blow.get('sigma_slope'),
                     lag_one=blow.get('lag_one'),
                     fetch=blow.get('fetch'),
                     max_snow_albedo=msa,
                     carbon=(carb or None))
    if raw_monthly is not None:
        mosaic.raw_monthly = raw_monthly
    return mosaic


def _bare_albedo(albedo, bare_tile):
    """vic_run.c:231-236: the bare-soil tile's albedo is ALBEDO_BARE_SOIL,
    whatever its class carries."""
    if albedo is None or bare_tile is None:
        return albedo
    return torch.where(bare_tile.unsqueeze(-1),
                       torch.full_like(albedo, ALBEDO_BARE_SOIL), albedo)


def build_atmosphere(fields, batch):
    """The `atm` dict `basin.simulate` consumes, from measured forcing.

    Every quantity the file supplies is used as given. Only air density,
    latent heat and the vapour-pressure deficit are derived, and each is a
    thermodynamic identity rather than an estimate.
    """
    missing = [k for k in ('tair', 'prcp', 'vp', 'shortwave', 'longwave')
               if k not in fields]
    if missing:
        extra = ''
        if 'tmax' in fields and 'tmin' in fields:
            # A daily file carries the extremes and little else. VIC 5's own
            # drivers stop here too; what they cannot do is say where the
            # missing fields come from, because VIC 5 deleted the code that
            # made them.
            extra = ('. This looks like a daily forcing (it has tmax and '
                     'tmin): disaggregate it first with '
                     'torchvic.forcing.daily.expand_forcing, which is the '
                     'MetSim-based MTCLIM chain and returns the same layout on '
                     'the model\'s own step')
        raise ValueError('the forcing is missing %s, which the energy '
                         'balance reads every step%s'
                         % (', '.join(missing), extra))
    tair = fields['tair']
    if 'press' in fields:
        press = fields['press']
        press_source = 'forcing file'
    else:
        press = atmos.air_pressure(tair,
                                   batch.elevation.unsqueeze(0).expand_as(tair))
        press_source = 'derived from elevation'
    vp = fields['vp']
    es = snow.svp(tair)
    # A sub-daily run has no daily maximum or minimum, and VIC does not use
    # one: tmax/tmin are the classic driver's disaggregation inputs, not image
    # forcing. Where they are absent the air temperature stands for both, so
    # nothing downstream has to branch on the step length.
    atm = dict(
        tair=tair,
        tmax=fields.get('tmax', tair), tmin=fields.get('tmin', tair),
        prcp=fields['prcp'], press=press,
        # vic_force.c 5.1.0:401-405 (image; classic :120-124): where the
        # forcing's vp exceeds svp(Tair) the deficit is 0 and vp itself is
        # set to svp(Tair). Every reader of force->vp (the canopy and pack
        # vapour fluxes, canopy_evap's vp) then sees the saturated value.
        eact_air=torch.where(es - vp < 0.0, es, vp),
        air_dens=atmos.air_density(tair, press),
        lv=atmos.latent_heat_of_vaporization(tair),
        vpd=torch.clamp(es - vp, min=0.0),
        shortwave=fields['shortwave'],
        longwave=fields['longwave'],
    )
    # Wind was read, unit-corrected and aggregated, and then never put here,
    # so the model fell back to one constant of 1.5 m/s for the whole domain
    # and the whole year while this module's docstring claimed nothing was
    # estimated that the forcing measures. It is measured; use it.
    if 'wind' in fields:
        atm['wind'] = fields['wind']
        wind_source = 'forcing file'
    else:
        wind_source = 'absent from the forcing; model default'
    atm['_sources'] = dict(press=press_source, longwave='forcing file',
                           shortwave='forcing file, 24-hour mean',
                           wind=wind_source)
    return atm
