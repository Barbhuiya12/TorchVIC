"""Convert VIC *classic*-driver parameter files into VIC 5 *image*-driver
inputs: a domain file and a parameter file.

    from torchvic.io.classic_to_image import convert
    convert(soil, vegparam, veglib, snowband, out_domain='domain.nc',
            out_params='params.nc', grid='forcing.nc', snow_bands=5,
            root_zones=3)

or `torchvic convert-classic` on the command line.

**Reading is not done here.** The four ASCII files are read by
`torchvic.io.classic` -- `ClassicDomain` and `ClassicParams`, which
`tests/test_classic_equivalence.py` holds to VIC's classic driver -- so the
numbers written are the numbers `vic_classic` would run with: layer depths
rounded to the millimetre (read_soilparam.c:252-256), MISSING residual
moisture replaced by VIC's default, root fractions and cover normalised as
read_vegparam.c does, snow-band fractions divided by their sum, the
vegetation library's LAI / fcanopy / albedo replaced per tile where
LAI_SRC / FCAN_SRC / ALB_SRC say FROM_VEGPARAM. This module only lays those
arrays onto a grid and writes them under the names, dimension order, dtypes,
fill values and units of VIC 5's image driver (vic_init.c in
drivers/shared_image), which are also those of UW-Hydro's sample file
`Stehekin_test_params_20160327.nc`.

**The bare-soil class.** The classic driver invents a bare-soil class after
the library's last (read_veglib.c:210-229); the image driver expects it as
the parameter file's last `veg_class` (vic_init.c: "assuming bare soil is the
last type"). It is written as the classic driver holds it: rarc 100, rmin 0,
albedo 0.2, LAI 0, fcanopy MIN_FCANOPY, roughness the soil file's `rough`,
displacement rough * 0.667 / 0.123 (read_soilparam.c:489-494), reference
height the global WIND_H. UW-Hydro's image sample stores roughness 0,
displacement 0 and wind_h 2 m for that class instead; `bare_soil=` can ask
for those values, and the two choices differ in the run wherever a cell has
bare soil (vic_run.c:166-192).

**Nveg** is the number of non-bare classes with cover > 0; VIC stops unless
the count of classes with cover > 0, bare soil included, is Nveg or Nveg + 1
(vic_init.c:1155-1166).

**Area** is computed as the classic driver computes it
(`classic.cell_area_vic`, compute_cell_area.c) unless `area='sphere'`, the
exact spherical band area on VIC's earth radius. UW-Hydro's sample domain
used `cdo gridarea` (6371 km), 1.9e-4 relative from the first and 6.9e-5
from the second.
"""
import os
import tempfile

import numpy as np

from .classic import (ClassicDomain, ClassicOptions, ClassicParams,
                      cell_area_vic, MONTHS)
from .global_param import GlobalParam, _announcer

#: netCDF's default fill values (netcdf.h NC_FILL_DOUBLE, NC_FILL_INT),
#: which UW-Hydro's sample parameter file uses as `_FillValue`.
FILL_DOUBLE = 9.969209968386869e36
FILL_INT = -2147483647
CONST_REARTH = 6.37122e6          # vic_physical_constants.h

# name: (units, description) -- the sample file's attributes, which tonic
# wrote from VIC's documentation.
_SOIL_ATTRS = {
    'run_cell': ('N/A', '1 = Run Grid Cell, 0 = Do Not Run'),
    'gridcell': ('N/A', 'Grid cell number'),
    'lats': ('degrees', 'Latitude of grid cell'),
    'lons': ('degrees', 'Longitude of grid cell'),
    'infilt': ('mm/day', 'Variable infiltration curve parameter (binfilt)'),
    'Ds': ('fraction', 'Fraction of Dsmax where non-linear baseflow begins'),
    'Dsmax': ('mm/day', 'Maximum velocity of baseflow'),
    'Ws': ('fraction', 'Fraction of maximum soil moisture where non-linear '
           'baseflow occurs'),
    'c': ('N/A', 'Exponent used in baseflow curve, normally set to 2'),
    'expt': ('N/A', 'Exponent n (=3+2/lambda) in Campbells eqn for hydraulic '
             'conductivity, HBH 5.6 (where lambda = soil pore size '
             'distribution parameter).  Values should be > 3.0.'),
    'Ksat': ('mm/day', 'Saturated hydrologic conductivity'),
    'phi_s': ('mm/mm', 'Soil moisture diffusion parameter'),
    'init_moist': ('mm', 'Initial layer moisture content'),
    'elev': ('m', 'Average elevation of grid cell'),
    'depth': ('m', 'Thickness of each soil moisture layer'),
    'avg_T': ('C', 'Average soil temperature, used as the bottom boundary '
              'for soil heat flux solutions'),
    'dp': ('m', 'Soil thermal damping depth (depth at which soil '
           'temperature remains constant through the year, ~4 m)'),
    'bubble': ('cm', 'Bubbling pressure of soil. Values should be > 0.0'),
    'quartz': ('fraction', 'Quartz content of soil'),
    'bulk_density': ('kg/m3', 'Bulk density of soil layer'),
    'soil_density': ('kg/m3', 'Soil particle density, normally 2685 kg/m3'),
    'organic': ('fraction', 'Fraction of soil layer that is organic'),
    'bulk_density_org': ('kg/m3', 'Bulk density of organic portion of soil'),
    'soil_density_org': ('kg/m3', 'Soil particle density of organic portion '
                         'of soil, normally 1300 kg/m3'),
    'off_gmt': ('hours', 'Time zone offset from GMT. This parameter '
                'determines how VIC interprets sub-daily time steps relative '
                'to the model start date and time.'),
    'Wcr_FRACT': ('fraction', 'Fractional soil moisture content at the '
                  'critical point (~70%% of field capacity) (fraction of '
                  'maximum moisture)'),
    'Wpwp_FRACT': ('fraction', 'Fractional soil moisture content at the '
                   'wilting point (fraction of maximum moisture)'),
    'rough': ('m', 'Surface roughness of bare soil'),
    'snow_rough': ('m', 'Surface roughness of snowpack'),
    'annual_prec': ('mm', 'Average annual precipitation.'),
    'resid_moist': ('fraction', 'Soil moisture layer residual moisture.'),
    'fs_active': ('binary', 'If set to 1, then frozen soil algorithm is '
                  'activated for the grid cell. A 0 indicates that frozen '
                  'soils are not computed even if soil temperatures fall '
                  'below 0C.'),
    'max_snow_distrib_slope': ('m', 'Maximum slope of the snow depth '
                               'distribution (SPATIAL_SNOW)'),
    'frost_slope': ('C', 'Slope of the uniform distribution of soil '
                    'temperature (SPATIAL_FROST)'),
    'July_Tavg': ('C', 'Average July air temperature (JULY_TAVG_SUPPLIED)'),
    'cellnum': ('N/A', 'Grid cell number (should match numbers assigned in '
                'soil parameter file)'),
    'AreaFract': ('fraction', 'Fraction of grid cell covered by each '
                  'elevation band. Sum of the fractions must equal 1.'),
    'elevation': ('m', 'Mean (or median) elevation of elevation band. This '
                  'is used to compute the change in air temperature from '
                  'the grid cell mean elevation.'),
    'Pfactor': ('fraction', 'Fraction of cell precipitation that falls on '
                'each elevation band. Total must equal 1. To ignore effects '
                'of elevation on precipitation, set these fractions equal to '
                'the area fractions.'),
}
_VEG_ATTRS = {
    'Nveg': ('N/A', 'Number of vegetation tiles in the grid cell'),
    'Cv': ('fraction', 'Fraction of grid cell covered by vegetation tile'),
    'root_depth': ('m', 'Root zone thickness (sum of depths is total depth '
                   'of root penetration)'),
    'root_fract': ('fraction', 'Fraction of root in the current root zone'),
    'LAI': ('m2/m2', 'Leaf Area Index, one per month'),
    'fcanopy': ('fraction', 'Fractional area covered by plant canopy, one '
                'per month'),
    'overstory': ('N/A', 'Flag to indicate whether or not the current '
                  'vegetation type has an overstory (1 for overstory present '
                  '[e.g. trees], 0 for overstory not present [e.g. grass])'),
    'rarc': ('s/m', 'Architectural resistance of vegetation type (~2 s/m)'),
    'rmin': ('s/m', 'Minimum stomatal resistance of vegetation type '
             '(~100 s/m)'),
    'wind_h': ('m', 'Height at which wind speed is measured.'),
    'RGL': ('W/m^2.', 'Minimum incoming shortwave radiation at which there '
            'will be transpiration. For trees this is about 30 W/m^2, for '
            'crops about 100 W/m^2.'),
    'rad_atten': ('fraction', 'Radiation attenuation factor. Normally set to '
                  '0.5, though may need to be adjusted for high latitudes.'),
    'wind_atten': ('fraction', 'Wind speed attenuation through the '
                   'overstory. The default value has been 0.5.'),
    'trunk_ratio': ('fraction', 'Ratio of total tree height that is trunk '
                    '(no branches). The default value has been 0.2.'),
    'albedo': ('fraction', 'Shortwave albedo for vegetation type'),
    'veg_rough': ('m', 'Vegetation roughness length (typically 0.123 * '
                  'vegetation height)'),
    'displacement': ('m', 'Vegetation displacement height (typically 0.67 * '
                     'vegetation height)'),
    'sigma_slope': ('N/A', 'Standard deviation of terrain slopes within the '
                    'vegetation tile (BLOWING)'),
    'lag_one': ('N/A', 'Lag-one autocorrelation of terrain slope within the '
                'vegetation tile (BLOWING)'),
    'fetch': ('m', 'Average fetch length within the vegetation tile '
              '(BLOWING)'),
    'Ctype': ('N/A', 'Photosynthetic pathway: 0 = C3, 1 = C4'),
    'MaxCarboxRate': ('mol CO2/m2s', 'Maximum carboxylation rate at 25 C'),
    'MaxiE_or_CO2Spec': ('mol CO2/m2s', 'Maximum electron transport rate at '
                         '25 C (C3) or CO2 specificity at 25 C (C4)'),
    'LUE': ('mol CO2/mol photons', 'Light-use efficiency'),
    'Nscale': ('N/A', '1 = nitrogen scaling of photosynthesis, 0 = none'),
    'Wnpp_inhib': ('fraction', 'Moisture level in top soil layer above which '
                   'net primary productivity is inhibited'),
    'NPPfactor_sat': ('fraction', 'Net primary productivity factor when the '
                      'top soil layer is saturated'),
}
#: Variables written as NC_INT; everything else per cell is NC_DOUBLE.
_INT_VARS = {'run_cell', 'gridcell', 'fs_active', 'Nveg', 'overstory',
             'Ctype', 'Nscale', 'mask'}

SOIL_ORDER = (('infilt', 0), ('Ds', 0), ('Dsmax', 0), ('Ws', 0), ('c', 0),
              ('expt', 1), ('Ksat', 1), ('phi_s', 1), ('init_moist', 1),
              ('elev', 0), ('depth', 1), ('avg_T', 0), ('dp', 0),
              ('bubble', 1), ('quartz', 1), ('bulk_density', 1),
              ('soil_density', 1), ('organic', 1), ('bulk_density_org', 1),
              ('soil_density_org', 1), ('off_gmt', 0), ('Wcr_FRACT', 1),
              ('Wpwp_FRACT', 1), ('rough', 0), ('snow_rough', 0),
              ('annual_prec', 0), ('resid_moist', 1), ('fs_active', 0),
              ('max_snow_distrib_slope', 0), ('frost_slope', 0),
              ('July_Tavg', 0))
#: Class-level fields as (name, extra axis): None, 'month' or 'root_zone'.
VEG_ORDER = (('Cv', None), ('root_depth', 'root_zone'),
             ('root_fract', 'root_zone'), ('LAI', 'month'),
             ('overstory', None), ('rarc', None), ('rmin', None),
             ('wind_h', None), ('RGL', None), ('rad_atten', None),
             ('wind_atten', None), ('trunk_ratio', None),
             ('albedo', 'month'), ('veg_rough', 'month'),
             ('displacement', 'month'), ('fcanopy', 'month'),
             ('sigma_slope', None), ('lag_one', None), ('fetch', None),
             ('Ctype', None), ('MaxCarboxRate', None),
             ('MaxiE_or_CO2Spec', None), ('LUE', None), ('Nscale', None),
             ('Wnpp_inhib', None), ('NPPfactor_sat', None))

#: What UW-Hydro's image sample stores for the bare-soil class where the
#: classic driver derives values (see the module docstring).
BARE_SOIL_SAMPLE = {'veg_rough': 0.0, 'displacement': 0.0, 'wind_h': 2.0}


# ---------------------------------------------------------------------------
# Options: the classic global file's, or the same keys given directly.
# ---------------------------------------------------------------------------
_FLAG_KEYS = {'veglib_fcan': 'VEGLIB_FCAN', 'veglib_photo': 'VEGLIB_PHOTO',
              'vegparam_lai': 'VEGPARAM_LAI', 'vegparam_fcan': 'VEGPARAM_FCAN',
              'vegparam_alb': 'VEGPARAM_ALB', 'blowing': 'BLOWING',
              'spatial_snow': 'SPATIAL_SNOW', 'spatial_frost': 'SPATIAL_FROST',
              'july_tavg_supplied': 'JULY_TAVG_SUPPLIED',
              'organic_fract': 'ORGANIC_FRACT', 'full_energy': 'FULL_ENERGY',
              'frozen_soil': 'FROZEN_SOIL'}
_VALUE_KEYS = {'nlayer': 'NLAYER', 'root_zones': 'ROOT_ZONES',
               'lai_src': 'LAI_SRC', 'fcan_src': 'FCAN_SRC',
               'alb_src': 'ALB_SRC', 'wind_h': 'WIND_H',
               'baseflow': 'BASEFLOW'}


def classic_options(soil=None, vegparam=None, veglib=None, snowband=None,
                    snow_bands=None, global_param=None, sample_data=None,
                    **opts):
    """A `ClassicOptions` for the conversion.

    With `global_param` the classic global file decides every option, as it
    would for `vic_classic`, and the file arguments only override its SOIL,
    VEGPARAM, VEGLIB and SNOW_BAND paths. Without it the options are the
    keyword arguments (`nlayer`, `root_zones`, `veglib_fcan`, `lai_src`, ...;
    see `_FLAG_KEYS` and `_VALUE_KEYS`), written as a classic global file and
    read by the same `GlobalParam` / `ClassicOptions` the classic reader
    uses, so VIC's defaults and VIC's refusals apply unchanged.
    """
    unknown = set(opts) - set(_FLAG_KEYS) - set(_VALUE_KEYS)
    if unknown:
        raise TypeError('unknown option(s): %s' % ', '.join(sorted(unknown)))
    if global_param is not None:
        if opts:
            raise ValueError('options come from %s; give either a global '
                             'file or options (%s), not both'
                             % (global_param, ', '.join(sorted(opts))))
        o = ClassicOptions(GlobalParam(global_param, sample_data=sample_data))
        if snow_bands is not None and snow_bands != o.snow_bands:
            raise ValueError('%s says SNOW_BAND %d, not %d'
                             % (global_param, o.snow_bands, snow_bands))
    else:
        if soil is None or vegparam is None or veglib is None:
            raise ValueError('soil, vegparam and veglib are needed when no '
                             'classic global file is given')
        nb = int(snow_bands or 1)
        if nb > 1 and snowband is None:
            raise ValueError('snow_bands = %d needs the snow band file' % nb)
        lines = ['SOIL %s' % os.path.abspath(soil),
                 'VEGPARAM %s' % os.path.abspath(vegparam),
                 'VEGLIB %s' % os.path.abspath(veglib),
                 'SNOW_BAND %d%s' % (nb, (' ' + os.path.abspath(snowband))
                                     if nb > 1 else '')]
        for k, key in _FLAG_KEYS.items():
            if k in opts:
                lines.append('%s %s' % (key, 'TRUE' if opts[k] else 'FALSE'))
        opts.setdefault('root_zones', 3)
        for k, key in _VALUE_KEYS.items():
            if opts.get(k) is not None:
                lines.append('%s %s' % (key, opts[k]))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'global.classic.txt')
            with open(p, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            o = ClassicOptions(GlobalParam(p))
    if soil is not None:
        o.soil_file = os.path.abspath(soil)
    if vegparam is not None:
        o.vegparam_file = os.path.abspath(vegparam)
    if veglib is not None:
        o.veglib_file = os.path.abspath(veglib)
    if snowband is not None:
        o.snowband_file = os.path.abspath(snowband)
    return o


# ---------------------------------------------------------------------------
# The grid.
# ---------------------------------------------------------------------------
def read_grid(path, lat='lat', lon='lon'):
    """1-D latitude and longitude axes from a netCDF file or a zarr store
    (a domain file, a forcing file, anything on the target grid)."""
    import xarray as xr
    if os.path.isdir(path) and (path.rstrip('/').endswith('.zarr') or
                                os.path.exists(os.path.join(path, '.zgroup'))
                                or os.path.exists(os.path.join(path,
                                                               'zarr.json'))):
        ds = xr.open_zarr(path)
    else:
        from .ncshare import read
        ds = read(path)
    try:
        la, lo = np.asarray(ds[lat].values, float), np.asarray(ds[lon].values,
                                                               float)
    finally:
        ds.close()
    if la.ndim != 1 or lo.ndim != 1:
        raise ValueError('%s: %s and %s must be 1-D axes (got %dD, %dD)'
                         % (path, lat, lon, la.ndim, lo.ndim))
    return la, lo


def grid_from_cells(lat, lon, resolution=None):
    """The regular grid spanning the cells, ascending, at `resolution`
    degrees (default: the smallest spacing between the cells' latitudes and
    longitudes)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if resolution is None:
        d = np.concatenate([np.diff(np.unique(lat)), np.diff(np.unique(lon))])
        d = d[d > 1e-9]
        if d.size == 0:
            raise ValueError('one cell: give the resolution')
        resolution = float(d.min())
    ny = int(round((lat.max() - lat.min()) / resolution)) + 1
    nx = int(round((lon.max() - lon.min()) / resolution)) + 1
    return (lat.min() + resolution * np.arange(ny),
            lon.min() + resolution * np.arange(nx))


def _axis_step(axis, what):
    d = np.diff(axis)
    if d.size == 0:
        raise ValueError('%s axis has one point; give the resolution' % what)
    if not np.allclose(d, d[0], rtol=0, atol=abs(d[0]) * 1e-6):
        raise ValueError('%s axis is not regular (spacing %g..%g)'
                         % (what, d.min(), d.max()))
    return float(d[0])


def locate(axis, values, what, outside='error'):
    """Index of each value on a regular axis; every value must be a grid
    point to within a thousandth of the spacing. A value beyond the axis is
    an error, or with outside='drop' gets index -1."""
    step = _axis_step(axis, what)
    values = np.asarray(values, float)
    idx = np.rint((values - axis[0]) / step).astype(int)
    off = (idx < 0) | (idx >= axis.size)
    if off.any() and outside != 'drop':
        raise ValueError('%d cell(s) have %s outside the grid (%g..%g), first '
                         '%g' % (off.sum(), what, axis.min(), axis.max(),
                                 values[off][0]))
    idx[off] = -1
    miss = ~off & (np.abs(axis[idx] - values) > abs(step) * 1e-3)
    if miss.any():
        raise ValueError('%d cell(s) have %s off the grid points, first %g '
                         '(nearest %g)' % (miss.sum(), what,
                                           values[miss][0],
                                           axis[idx][miss][0]))
    return idx


def cell_area(lat, lon, resolution, method='vic'):
    """Cell area, m2, on the grid's 2-D lat/lon: 'vic' is compute_cell_area.c
    (the classic driver), 'sphere' the exact band area on CONST_REARTH."""
    if method == 'vic':
        return cell_area_vic(lat, lon, resolution)
    if method == 'sphere':
        r = np.deg2rad
        h = resolution / 2.0
        return (CONST_REARTH ** 2 * r(resolution)
                * np.abs(np.sin(r(lat + h)) - np.sin(r(lat - h))))
    raise ValueError("area method %r: 'vic' or 'sphere'" % method)


# ---------------------------------------------------------------------------
# Conversion.
# ---------------------------------------------------------------------------
class Converted(object):
    """The classic files read, with every image variable per active cell.

    `cell[name]` is [cell], [layer|band, cell] or [class, (month|zone,)
    cell]; `iy`, `ix` place the cells on the grid (`lat`, `lon` axes)."""

    def __init__(self, opts, grid=None, resolution=None, bare_soil=None,
                 outside='error', log=None):
        say = _announcer(log)
        self.opts = opts
        if grid is not None and resolution is None:
            resolution = abs(_axis_step(np.asarray(grid[0], float),
                                        'latitude'))
        if resolution is not None and not opts.resolution:
            # The classic driver's RESOLUTION, so ClassicDomain computes
            # areas rather than announcing that it cannot.
            opts.resolution = float(resolution)
        dom = ClassicDomain(opts.soil_file, opts, log=log)
        par = ClassicParams(dom, opts, log=log)
        self.domain, self.params = dom, par
        if grid is None:
            self.lat, self.lon = grid_from_cells(dom.lat, dom.lon, resolution)
        else:
            self.lat, self.lon = (np.asarray(grid[0], float),
                                  np.asarray(grid[1], float))
        iy = locate(self.lat, dom.lat, 'latitude', outside)
        ix = locate(self.lon, dom.lon, 'longitude', outside)
        #: Positions, among the soil file's run cells, of the cells written.
        self.keep = np.flatnonzero((iy >= 0) & (ix >= 0))
        #: Grid cell numbers of run cells beyond the grid (outside='drop').
        self.dropped = dom.gridcell[(iy < 0) | (ix < 0)]
        if self.dropped.size:
            say('  %d run cell(s) lie outside the grid (lat %g..%g, lon '
                '%g..%g) and are not written: gridcell %s'
                % (self.dropped.size, self.lat.min(), self.lat.max(),
                   self.lon.min(), self.lon.max(),
                   ' '.join(str(g) for g in self.dropped[:30])
                   + (' ...' if self.dropped.size > 30 else '')))
        self.iy, self.ix = iy[self.keep], ix[self.keep]
        flat = self.iy * self.lon.size + self.ix
        if np.unique(flat).size != flat.size:
            raise ValueError('two soil-file cells fall on one grid cell')
        self.resolution = (abs(_axis_step(self.lat, 'latitude'))
                           if resolution is None else float(resolution))
        n = dom.n_active

        c = {}
        c['run_cell'] = np.ones(n, np.int32)
        c['gridcell'] = dom.gridcell.astype(np.int32)
        c['lats'], c['lons'] = dom.lat, dom.lon
        soil = dict(par.soil)
        soil.update(par.extra)
        soil['off_gmt'] = par.off_gmt
        for k, _ in SOIL_ORDER:
            if k in soil:
                c[k] = np.asarray(soil[k])
        c['fs_active'] = c['fs_active'].astype(np.int32)
        c['cellnum'] = dom.gridcell.astype(np.float64)
        c.update({k: np.asarray(v) for k, v in par.bands.items()})

        v = dict(par.veg)
        if 'fcanopy' not in v:
            # VIC's FROM_DEFAULT values (1, bare soil MIN_FCANOPY), so the
            # file is complete whichever FCAN_SRC the image global sets.
            v['fcanopy'] = par.fcanopy_default
        if 'Ctype' in v:
            # read_veglib.c keeps one of MaxETransport (C3) and
            # CO2Specificity (C4) and zeroes the other; the image file has
            # one variable for the two.
            v['MaxiE_or_CO2Spec'] = np.where(v['Ctype'] == 0,
                                             v.pop('MaxETransport'),
                                             v.pop('CO2Specificity'))
            v['LUE'] = v.pop('LightUseEff')
            v['Nscale'] = v.pop('NscaleFlag')
        for k in ('overstory', 'Ctype', 'Nscale'):
            if k in v:
                v[k] = np.rint(v[k]).astype(np.int32)
        nc = par.n_veg
        if bare_soil:
            for k, x in bare_soil.items():
                if k not in v:
                    raise ValueError('bare_soil: %r is not a class field' % k)
                v[k][nc - 1] = x
            say('  bare-soil class (%d): %s set as given, not as the classic '
                'driver derives them' % (nc, ', '.join(
                    '%s %g' % kv for kv in sorted(bare_soil.items()))))
        # vic_init.c:1155-1166: Nveg counts the vegetated classes present.
        c['Nveg'] = (v['Cv'][:nc - 1] > 0).sum(0).astype(np.int32)
        for k, _ in VEG_ORDER:
            if k in v:
                c[k] = np.asarray(v[k])
        self.cell = {k: a[..., self.keep] for k, a in c.items()}
        self.n_cells = int(self.keep.size)
        say('  %d active cells on a %d x %d grid'
            % (self.n_cells, self.lat.size, self.lon.size))
        self.n_veg = nc
        self.veg_class = np.asarray(par.veg_class_ids, np.int32)
        self.veg_descr = list(par.veg_descr)
        self.n_layers = par.n_layers
        self.n_bands = opts.snow_bands
        self.n_root = opts.root_zones

    @property
    def mask(self):
        m = np.zeros((self.lat.size, self.lon.size), np.int32)
        m[self.iy, self.ix] = 1
        return m

    def gridded(self, name, fill=None):
        """`cell[name]` on the grid, `fill` elsewhere."""
        a = self.cell[name]
        if fill is None:
            fill = FILL_INT if a.dtype.kind == 'i' else FILL_DOUBLE
        out = np.full(a.shape[:-1] + (self.lat.size, self.lon.size), fill,
                      dtype=a.dtype)
        out[..., self.iy, self.ix] = a
        return out


def _nc(path, mode='w'):
    import netCDF4
    if os.path.exists(path):
        os.remove(path)
    return netCDF4.Dataset(path, mode, format='NETCDF4')


def _coords(ds, lat, lon):
    ds.createDimension('lat', lat.size)
    ds.createDimension('lon', lon.size)
    for name, ax, units, longn, std, axis in (
            ('lat', lat, 'degrees_north', 'latitude of grid cell center',
             'latitude', 'Y'),
            ('lon', lon, 'degrees_east', 'longitude of grid cell center',
             'longitude', 'X')):
        v = ds.createVariable(name, 'f8', (name,))
        v.setncatts({'standard_name': std, 'long_name': longn,
                     'units': units, 'axis': axis})
        v[:] = ax


def write_domain(cv, path, area='vic', frac=None, compress=4):
    """The image driver's domain file: lat, lon, mask, area, frac, elev.

    `frac` is a [lat, lon] array on the grid, or None for 1 on every active
    cell. mask is NC_INT as get_global_domain.c:46-49 demands."""
    lat2, lon2 = np.meshgrid(cv.lat, cv.lon, indexing='ij')
    a = cell_area(lat2, lon2, cv.resolution, area)
    mask = cv.mask
    if frac is None:
        frac = mask.astype(np.float64)
    else:
        frac = np.where(mask == 1, np.asarray(frac, float), 0.0)
    kw = dict(zlib=bool(compress), complevel=compress or None)
    ds = _nc(path)
    try:
        _coords(ds, cv.lat, cv.lon)
        v = ds.createVariable('mask', 'i4', ('lat', 'lon'), **kw)
        v.setncatts({'long_name': 'domain mask', 'comment': '0 value '
                     'indicates cell is not active', 'note': 'unitless'})
        v[:] = mask
        v = ds.createVariable('area', 'f8', ('lat', 'lon'), **kw)
        v.setncatts({'standard_name': 'area', 'long_name': 'area of grid '
                     'cell', 'units': 'm2',
                     'note': {'vic': 'compute_cell_area.c (VIC classic '
                              'driver)', 'sphere': 'spherical band area, '
                              'R = 6.37122e6 m'}[area]})
        v[:] = a
        v = ds.createVariable('frac', 'f8', ('lat', 'lon'), **kw)
        v.setncatts({'long_name': 'fraction of grid cell that is active',
                     'note': 'unitless'})
        v[:] = frac
        v = ds.createVariable('elev', 'f8', ('lat', 'lon'),
                              fill_value=FILL_DOUBLE, **kw)
        v.setncatts({'long_name': 'mean elevation of grid cell',
                     'units': 'm'})
        v[:] = cv.gridded('elev')
        ds.setncatts({'title': 'VIC 5 image driver domain',
                      'source': 'torchvic.io.classic_to_image from %s'
                      % os.path.basename(cv.opts.soil_file)})
    finally:
        ds.close()
    return path


def write_params(cv, path, compress=4):
    """The image driver's parameter file, laid out as UW-Hydro's sample
    `Stehekin_test_params_20160327.nc`: dims lat, lon, nlayer, snow_band,
    veg_class, root_zone, month; NC_DOUBLE / NC_INT with netCDF's default
    fill values on inactive cells."""
    import netCDF4  # noqa: F401  (the writer)
    ds = _nc(path)
    ny, nx = cv.lat.size, cv.lon.size

    def chunks(dims):
        # VIC reads one (lat, lon) slab at a time (get_scatter_nc_field_*).
        return [1] * (len(dims) - 2) + [ny, nx]

    def put(name, dims, attrs):
        a = cv.cell[name]
        is_int = name in _INT_VARS
        kw = dict(zlib=bool(compress), complevel=compress or None,
                  chunksizes=chunks(dims),
                  fill_value=FILL_INT if is_int else FILL_DOUBLE)
        v = ds.createVariable(name, 'i4' if is_int else 'f8', dims, **kw)
        v.setncatts(attrs)
        g = cv.gridded(name)
        if g.dtype.kind == 'i' and not is_int:
            g = g.astype(np.float64)
        v[:] = g

    try:
        _coords(ds, cv.lat, cv.lon)
        ds.createDimension('nlayer', cv.n_layers)
        ds.createDimension('snow_band', cv.n_bands)
        ds.createDimension('veg_class', cv.n_veg)
        ds.createDimension('root_zone', cv.n_root)
        ds.createDimension('month', MONTHS)

        m = ds.createVariable('mask', 'i4', ('lat', 'lon'),
                              fill_value=FILL_INT)
        m.setncatts({'long_name': 'domain mask', 'comment': '0 value '
                     'indicates cell is not active', 'note': 'unitless'})
        m[:] = np.where(cv.mask == 1, 1, FILL_INT)
        v = ds.createVariable('layer', 'i4', ('nlayer',))
        v.long_name = 'soil layer'
        v[:] = np.arange(1, cv.n_layers + 1)

        def soil_put(name, layered):
            units, descr = _SOIL_ATTRS[name]
            dims = (('nlayer',) if layered else ()) + ('lat', 'lon')
            put(name, dims, {'units': units, 'description': descr,
                             'long_name': name})

        for k in ('run_cell', 'gridcell', 'lats', 'lons'):
            soil_put(k, False)
        for k, layered in SOIL_ORDER:
            if k in cv.cell:
                soil_put(k, bool(layered))

        v = ds.createVariable('snow_band', 'i4', ('snow_band',))
        v.long_name = 'snow band'
        v[:] = np.arange(1, cv.n_bands + 1)
        soil_put('cellnum', False)
        for k in ('AreaFract', 'elevation', 'Pfactor'):
            units, descr = _SOIL_ATTRS[k]
            put(k, ('snow_band', 'lat', 'lon'),
                {'units': units, 'description': descr, 'long_name': k})

        v = ds.createVariable('veg_class', 'i4', ('veg_class',))
        v.long_name = 'Vegetation Class'
        v[:] = cv.veg_class
        v = ds.createVariable('veg_descr', str, ('veg_class',))
        v.long_name = 'Vegetation Class Description'
        for j, s in enumerate(cv.veg_descr):
            v[j] = s
        v = ds.createVariable('root_zone', 'i4', ('root_zone',))
        v.long_name = 'root zone'
        v[:] = np.arange(1, cv.n_root + 1)
        v = ds.createVariable('month', 'i4', ('month',))
        v.long_name = 'month of year'
        v[:] = np.arange(1, MONTHS + 1)

        units, descr = _VEG_ATTRS['Nveg']
        put('Nveg', ('lat', 'lon'), {'long_name': 'Nveg', 'units': units,
                                     'description': descr})
        for k, extra in VEG_ORDER:
            if k not in cv.cell:
                continue
            units, descr = _VEG_ATTRS[k]
            dims = ('veg_class',) + ((extra,) if extra else ()) + ('lat',
                                                                    'lon')
            put(k, dims, {'long_name': k, 'units': units,
                          'description': descr})
        o = cv.opts
        ds.setncatts({
            'description': 'VIC parameter file',
            'source': 'torchvic.io.classic_to_image',
            'classic_files': ' '.join(os.path.basename(p) for p in (
                o.soil_file, o.vegparam_file, o.veglib_file,
                o.snowband_file or '') if p),
            'classic_options': 'NLAYER %d ROOT_ZONES %d SNOW_BAND %d '
            'VEGLIB_FCAN %s VEGLIB_PHOTO %s VEGPARAM_LAI %s VEGPARAM_FCAN %s '
            'VEGPARAM_ALB %s LAI_SRC %s FCAN_SRC %s ALB_SRC %s WIND_H %g'
            % (cv.n_layers, o.root_zones, o.snow_bands, o.veglib_fcan,
               o.veglib_photo, o.vegparam_lai, o.vegparam_fcan,
               o.vegparam_alb, o.lai_src, o.fcan_src, o.alb_src, o.wind_h)})
    finally:
        ds.close()
    return path


def convert(soil=None, vegparam=None, veglib=None, snowband=None,
            out_domain=None, out_params=None, grid=None, resolution=None,
            area='vic', frac=None, bare_soil=None, outside='error',
            global_param=None, sample_data=None, snow_bands=None,
            compress=4, log=None, **opts):
    """Classic files to an image domain and parameter file.

    grid: None (the cells' own extent at `resolution`), a (lat, lon) pair of
    1-D axes, or the path of a netCDF file / zarr store with 1-D lat and lon.
    frac: None (1 on active cells), a [lat, lon] array, or a domain file whose
    `frac` is taken. outside: 'error' (default) when a run cell is beyond the
    grid, or 'drop' to leave such cells out, listed in `Converted.dropped`. Other keywords as `classic_options`. Returns the
    `Converted` arrays."""
    o = classic_options(soil, vegparam, veglib, snowband, snow_bands,
                        global_param, sample_data, **opts)
    if isinstance(grid, str):
        grid = read_grid(grid)
    cv = Converted(o, grid=grid, resolution=resolution, bare_soil=bare_soil,
                   outside=outside, log=log)
    if isinstance(frac, str):
        from .ncshare import read
        with read(frac) as d:
            frac = np.nan_to_num(np.asarray(d['frac'].values, float))
    if out_domain:
        write_domain(cv, out_domain, area=area, frac=frac, compress=compress)
    if out_params:
        write_params(cv, out_params, compress=compress)
    return cv
