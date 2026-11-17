"""Read what VIC 5's *classic* driver reads: the ASCII soil, vegetation
library, vegetation parameter and snow-band files, and one ASCII forcing file
per cell.

The point is the one `vic_image` makes for the NetCDF set-up, applied to the
older one: a VIC dataset gives the same answer whichever driver reads it, and
TorchVIC has to match both. VIC's own two drivers were measured agreeing on
Stehekin, so "the same as VIC" has one referent; this
module lets the classic files reach `image_run.build_soil`, `build_mosaic`,
`build_atmosphere` and `basin.simulate` **unchanged**, by producing objects
with the same attributes, the same array layouts and the same units as
`ImageDomain`, `ImageParams` and `ImageForcing`.

The layouts are therefore not chosen here; they are the image reader's:

    soil[k]        [cell]  or  [layer, cell]
    veg[k]         [class, cell], [class, month, cell], [class, zone, cell]
    bands[k]       [band, cell]
    forcing[k]     [time, cell], pressure and vapour pressure in Pa

and the class axis is the library's order with VIC's implicit bare-soil tile
appended last, which is also how UW-Hydro's image parameter files are laid
out (Stehekin's `veg_class` is 1..12 with 12 = "Bare Soil").

**VIC's C is the authority for every column.** The classic files have no
header that names their fields; the column order is fixed by the read order
in `drivers/classic/src/read_soilparam.c`, `read_veglib.c`, `read_vegparam.c`
and `read_snowband.c`, and it changes with global-file options --
ORGANIC_FRACT, SPATIAL_SNOW, SPATIAL_FROST and JULY_TAVG_SUPPLIED each add
columns to the soil file, VEGLIB_FCAN adds twelve to the library, VEGPARAM_LAI
/ VEGPARAM_FCAN / VEGPARAM_ALB each add a line per tile, BLOWING adds three
fields to the tile line. Every one of those is read in the place the C reads
it, and each function below cites the lines it follows.

What VIC computes *on read* is reproduced where the image reader reproduces
it, and only there. `ImageParams` stores the file's own values and derives
porosity, maximum moisture and the critical/wilting moistures on demand
(`read_soilparam.c:607-610, 645-646`); `ClassicParams` inherits those methods
untouched, so the two cannot drift. The organic-fraction mixing of the densities
is applied to both paths by `case.load` (`apply_density_options`); a read-time
transform the image reader does not apply is refused here by name rather than
applied on one path and not the other, because a classic file and an image
file of the same basin would then read differently. The
NIJSSEN2001-to-ARNO baseflow conversion is the other way round: both
drivers' files reach `image_run.build_soil`, which applies it there
(`read_soilparam.c:693-707`), so there is nothing to refuse. Where the two
VIC *drivers* themselves differ
on read, the difference is reproduced on the classic side and named in the
docstring: the bare-soil tile's roughness (`read_soilparam.c:489-494`) and
the normalisation of fractional cover (`read_vegparam.c:628-645`).

Never a silent default. Every choice made on the user's behalf -- a unit
converted, a cell not found in the snow-band file, a trailing soil column the
options say not to read, an inactive cell skipped -- goes through `log`, or
`warnings.warn` when no log is given, exactly as `vic_image` does.
"""
import datetime as dt
import math
import os
import re

import numpy as np

from .global_param import GlobalParam, _announcer
from .forcing import ImageForcing
from .params import ImageParams

# vic/vic_run/include/vic_def.h and vic_physical_constants.h
MISSING = -99999.0          # vic_def.h:54
NODATA_VH = -1.0            # vic_def.h:57, "missing value for veg_hist inputs"
MINSOILDEPTH = 0.001        # vic_def.h:72, m
MIN_FCANOPY = 0.0001        # vic_def.h:73
MM_PER_M = 1000.0
PA_PER_KPA = 1000.0         # vic_physical_constants.h:46
CONST_REARTH = 6.37122e6    # vic_physical_constants.h:77, m
MONTHS = 12
SEC_PER_DAY = 86400

# vic/drivers/shared_all/src/initialize_parameters.c -- the defaults VIC gives
# the bare-soil tile it invents for every cell (read_veglib.c:210-229).
SOIL_RARC = 100.0           # :58
SOIL_RESID_MOIST = 0.0      # :59
ALBEDO_BARE_SOIL = 0.2      # :48

#: The forcing types the classic driver knows (get_force_type.c:61-124). SKIP
#: is a real type: a column that is present in the file and read into
#: nowhere, which is how Stehekin's air-density column is declared.
FORCE_TYPES = ('AIR_TEMP', 'ALBEDO', 'CATM', 'CHANNEL_IN', 'FCANOPY', 'FDIR',
               'LAI_IN', 'LWDOWN', 'PAR', 'PREC', 'PRESSURE', 'VP', 'SWDOWN',
               'WIND', 'SKIP',
               # VIC 4.2.d's names (get_force_type.c 4.2.d), which 5.1.0
               # replaced (SHORTWAVE -> SWDOWN, LONGWAVE -> LWDOWN) or
               # dropped (DENSITY, a column VIC 5 only skips); accepted so
               # OUTPUT_FORCE reads a 4.2.d forcing as it was written
               'SHORTWAVE', 'LONGWAVE', 'DENSITY', 'TMAX', 'TMIN')
#: Per-tile forcing (one value per vegetation class per record) that
#: read_atmos_data.c:229-238 reads N_ELEM values for. Not carried here: they
#: would need a [time, class, cell] layout the image path has no equivalent
#: for, and a file declaring them is refused rather than misread.
VEG_HIST_TYPES = ('ALBEDO', 'LAI_IN', 'FCANOPY')


def _resolve(gp, value):
    """A path taken from a global-file line, expanded as `GlobalParam.path_of`
    expands the one-token keys.

    SNOW_BAND carries its file as the *second* token (`SNOW_BAND 5 path`,
    get_global_param.c:529-531), which `path_of` cannot reach, so the same
    expansion is applied here to an explicit value: ${VIC_SAMPLE_DATA} from
    the caller, then environment variables, then the global file's directory
    for a relative path.
    """
    if value is None:
        return None
    v = value
    if gp.sample_data and '${VIC_SAMPLE_DATA}' in v:
        v = v.replace('${VIC_SAMPLE_DATA}', os.path.abspath(gp.sample_data))
    v = os.path.expandvars(os.path.expanduser(v))
    return v if os.path.isabs(v) else os.path.join(gp.root, v)


def _open_text(path, what):
    """A text file VIC would open, or the error VIC would give.

    `vic_image._open` names near-miss filenames when a NetCDF is missing,
    because UW-Hydro's own sample global files misname their files. The same
    courtesy for ASCII: FindleyLake's classic global says `params/` where the
    directory shipped is `parameters/`.
    """
    if os.path.exists(path):
        return open(path)
    import difflib
    d = os.path.dirname(path) or '.'
    near = []
    parent = os.path.dirname(d)
    if os.path.isdir(d):
        near = difflib.get_close_matches(os.path.basename(path),
                                         os.listdir(d), n=3, cutoff=0.5)
    elif os.path.isdir(parent):
        # The directory itself is misnamed; say which sibling it probably is.
        near = ['%s/' % n for n in difflib.get_close_matches(
            os.path.basename(d), os.listdir(parent), n=3, cutoff=0.5)]
    raise IOError('%s %s does not exist%s' % (
        what, path, ('; the directory holds %s -- the global file may name '
                     'it wrongly' % ', '.join(near)) if near else ''))


def force_columns(global_path):
    """The forcing file's columns, in order, from the FORCE_TYPE lines.

    The classic driver counts FORCE_TYPE lines after FORCING1
    (`count_force_vars`) and assigns each to the next column
    (get_force_type.c:131, `FORCE_INDEX[file_num][field] = type`), so the
    order of the lines *is* the order of the columns. `GlobalParam` cannot
    return that order: a one-token line (`FORCE_TYPE PREC`, Stehekin) lands
    in `raw['FORCE_TYPE']`, a two-token line (`FORCE_TYPE PREC prec`,
    FindleyLake) in the `force_type` dict, and a repeated SKIP in the second
    form would collapse onto one key. So the file is walked once more here,
    for that one key, with the same comment stripping.

    For ASCII forcing the tokens after the type name are ignored by VIC
    (get_force_type.c:136-138 reads them as the binary SIGNED flag and
    multiplier) and are ignored here. Only FORCING1 is read; a FORCING2
    section is refused by `ClassicOptions`.
    """
    cols, in_first = [], False
    for line in open(global_path):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = re.split(r'\s+', line)
        key = parts[0].upper()
        if key == 'FORCING1':
            in_first = True
        elif key == 'FORCING2':
            in_first = False
        elif key == 'FORCE_TYPE' and in_first and len(parts) > 1:
            name = parts[1].upper()
            if name not in FORCE_TYPES:
                raise ValueError('%s: FORCE_TYPE %r is not a forcing type the '
                                 'classic driver knows (get_force_type.c); '
                                 'VIC would stop here too'
                                 % (os.path.basename(global_path), parts[1]))
            cols.append(name)
    return cols


class ClassicOptions(object):
    """The global-file options that decide how the classic files are laid out.

    Everything here is read off a `GlobalParam` -- the *same* class that reads
    an image global file, which already carries SOIL, VEGLIB, VEGPARAM and the
    two-token SNOW_BAND in `raw` -- so there is one parser for both drivers
    and this class only interprets. Defaults are VIC's own
    (`initialize_options.c`, `initialize_global.c`), cited per field.

    Options that would change how a file is *read* and are not implemented
    here are refused by name, in `refused`, rather than parsed past: a soil
    file written with ORGANIC_FRACT TRUE has 3*NLAYER more columns than one
    without (read by `soil_columns`).
    """

    def __init__(self, gp):
        self.gp = gp
        self.refused = []
        g = gp

        self.nlayer = g.get_int('NLAYER', 3)          # initialize_options.c:75
        self.nodes = g.get_int('NODES', 3)            # :76
        self.root_zones = g.get_int('ROOT_ZONES', 0)  # :77
        if self.root_zones <= 0:
            # get_global_param.c:1037-1040 -- VIC stops.
            raise ValueError('ROOT_ZONES must be a positive integer in the '
                             'global file; the vegetation parameter file '
                             'cannot be read without it')
        sb = g.raw.get('SNOW_BAND', [[]])[0]
        self.snow_bands = int(sb[0]) if sb else 1      # :78
        self.snowband_file = (_resolve(g, sb[1]) if len(sb) > 1 else None)
        if self.snow_bands > 1 and not self.snowband_file:
            raise ValueError('SNOW_BAND %d without a file: the classic driver '
                             'reads the band file named on the same line '
                             '(get_global_param.c:529-531)' % self.snow_bands)

        self.full_energy = g.get_bool('FULL_ENERGY')
        self.frozen_soil = g.get_bool('FROZEN_SOIL')
        self.organic_fract = g.get_bool('ORGANIC_FRACT')        # :86
        self.spatial_snow = g.get_bool('SPATIAL_SNOW')          # :70
        # options.SHARE_LAYER_MOIST (initialize_options.c:67) defaults TRUE,
        # so an absent key is TRUE and only an explicit FALSE turns the
        # sharing off. canopy_evap.c:287-362's CASE 1 and CASE 2 are both
        # implemented; before this the flag was not read and a file asking
        # for CASE 2 alone got CASE 1 anyway.
        _share = g.get('SHARE_LAYER_MOIST')
        self.share_layer_moist = (True if _share is None
                                  else g.get_bool('SHARE_LAYER_MOIST'))
        self.spatial_frost = g.get_bool('SPATIAL_FROST')        # :69
        self.july_tavg_supplied = g.get_bool('JULY_TAVG_SUPPLIED')  # :84
        self.blowing = g.get_bool('BLOWING')                    # :44
        self.veglib_fcan = g.get_bool('VEGLIB_FCAN')            # :87
        self.veglib_photo = g.get_bool('VEGLIB_PHOTO')          # :88
        self.vegparam_lai = g.get_bool('VEGPARAM_LAI')          # :91
        self.vegparam_fcan = g.get_bool('VEGPARAM_FCAN')        # :90
        self.vegparam_alb = g.get_bool('VEGPARAM_ALB')          # :89
        self.lai_src = (g.get('LAI_SRC') or 'FROM_VEGLIB').upper()   # :85
        self.fcan_src = (g.get('FCAN_SRC') or 'FROM_DEFAULT').upper()  # :82
        self.alb_src = (g.get('ALB_SRC') or 'FROM_VEGLIB').upper()   # :80
        self.baseflow = (g.get('BASEFLOW') or 'ARNO').upper()
        self.equal_area = g.get_bool('EQUAL_AREA')              # :55
        res = g.get('RESOLUTION')
        self.resolution = float(res) if res is not None else 0.0  # initialize_global.c:57
        self.grid_decimal = g.get_int('GRID_DECIMAL', 2)        # :83
        self.wind_h = float(g.get('WIND_H') or 10.0)            # initialize_global.c:58
        self.force_format = (g.get('FORCE_FORMAT') or '').upper()
        self.force_steps_per_day = g.get_int('FORCE_STEPS_PER_DAY', 0)
        self.snow_steps_per_day = g.get_int('SNOW_STEPS_PER_DAY', 0)
        self.model_steps_per_day = g.get_int('MODEL_STEPS_PER_DAY', 0)
        fy, fm, fd = (g.get_int('FORCEYEAR'), g.get_int('FORCEMONTH', 1),
                      g.get_int('FORCEDAY', 1))
        self.force_start = (dt.datetime(fy, fm, fd)
                            + dt.timedelta(seconds=g.get_int('FORCESEC', 0))
                            if fy else None)
        self.force_columns = force_columns(g.path)

        self.soil_file = g.path_of('SOIL', g.sample_data)
        self.veglib_file = g.path_of('VEGLIB', g.sample_data)
        self.vegparam_file = g.path_of('VEGPARAM', g.sample_data)
        self.forcing_prefix = g.path_of('FORCING1', g.sample_data)

        # -- what this reader will not read, and why -------------------------
        if self.force_format == 'BINARY':
            self.refused.append(
                'FORCE_FORMAT BINARY: the short-int layout with per-type '
                'SIGNED/multiplier and optional 0xFFFF header '
                '(read_atmos_data.c:95-201) is not implemented; only ASCII')
        # ORGANIC_FRACT TRUE: the soil file carries 3*NLAYER more columns
        # (read here, `soil_columns`) and VIC mixes bulk and soil density
        # with the organic fraction on read (read_soilparam.c:364-432,
        # 596-603); `case.load` applies that mixing to every path with
        # ImageParams.apply_density_options (the CRB soils).
        # BASEFLOW NIJSSEN2001 needs no refusal: the d1..d4 the soil
        # file carries are converted to ARNO's Ds, Dsmax, Ws, c in
        # `image_run.build_soil` (read_soilparam.c:693-707), which the
        # classic parameters reach like the image ones.
        # COMPUTE_TREELINE <FALSE | veg class> (get_global_param.c:189-197):
        # anything but FALSE turns it on and is read with atoi as the
        # default vegetation above the treeline (TRUE therefore reads as
        # class 0); -1, the initialize_options.c default, is bare soil.
        self.compute_treeline, self.treeline_veg = False, -1
        for rest in g.raw.get('COMPUTE_TREELINE', []):
            tok = rest[0] if rest else ''
            if tok.upper() == 'FALSE':
                self.compute_treeline = False
            else:
                self.compute_treeline = True
                m = re.match(r'\s*[+-]?\d+', tok)
                self.treeline_veg = int(m.group(0)) if m else 0
        if self.compute_treeline:
            # get_global_param.c:1299-1303 and :1327-1330: both drivers stop.
            if not self.july_tavg_supplied:
                self.refused.append(
                    'COMPUTE_TREELINE with JULY_TAVG_SUPPLIED FALSE: VIC '
                    'stops (get_global_param.c:1299-1303); set both, with '
                    'the July average temperature as the soil file\'s last '
                    'column')
            if g.get('LAKES') and str(g.get('LAKES')).upper() != 'FALSE':
                self.refused.append(
                    'LAKES with COMPUTE_TREELINE: incompatible options in '
                    'VIC (get_global_param.c:1327-1330)')
        # VEGLIB_PHOTO's seven library columns are read (read_veglib.c:
        # 163-190, `n_photo` below) and the carbon cycle runs from them
        # (basin.simulate(carbon=True)). CARBON also needs CATM, PAR and
        # FDIR, which VIC 5 demands as forcing columns and no public dataset
        # carries; torchvic.forcing.daily derives them as VIC 4.2.d did.
        self.carbon = g.get_bool('CARBON')
        # options.Ncanopy -- the number of equal-leaf-area layers the canopy
        # is split into for photosynthesis (initialize_options.c:73).
        self.canopy_layers = g.get_int('CANOPY_LAYERS') or 3
        # RC_MODE RC_PHOTO: the canopy resistance from photosynthetic
        # demand (canopy_evap.c:296-307); basin.simulate takes it as
        # carbon_options['rc_mode']. VIC resets it to RC_JARVIS without
        # CARBON (get_global_param.c:1140-1143).
        self.rc_mode = (str(g.get('RC_MODE') or 'RC_JARVIS').upper()
                        if self.carbon else 'RC_JARVIS')
        for key, src in (('LAI_SRC', self.lai_src), ('FCAN_SRC', self.fcan_src),
                         ('ALB_SRC', self.alb_src)):
            if src == 'FROM_VEGHIST':
                self.refused.append(
                    '%s FROM_VEGHIST: per-tile forcing columns (ALBEDO, LAI_IN, '
                    'FCANOPY; read_atmos_data.c:229-238) have no [time, class, '
                    'cell] home on the image path' % key)
        hist = [c for c in self.force_columns if c in VEG_HIST_TYPES]
        if hist:
            self.refused.append('FORCE_TYPE %s: per-tile forcing is not read'
                                % ', '.join(hist))
        if 'FORCING2' in g.raw:
            self.refused.append('FORCING2: only the first forcing file is '
                                'read; a second would be silently dropped')
        if self.lai_src == 'FROM_VEGPARAM' and not self.vegparam_lai:
            # get_global_param.c:1051-1064 -- VIC stops.
            raise ValueError('LAI_SRC FROM_VEGPARAM requires VEGPARAM_LAI '
                             'TRUE; VIC refuses this combination')
        daily_forcing = 'TMAX' in self.force_columns and \
            'TMIN' in self.force_columns
        if self.force_steps_per_day and self.snow_steps_per_day and \
                self.force_steps_per_day != self.snow_steps_per_day and \
                not daily_forcing:
            # get_global_param.c:986-988 -- VIC stops.
            raise ValueError('FORCE_STEPS_PER_DAY (%d) must match '
                             'SNOW_STEPS_PER_DAY (%d)'
                             % (self.force_steps_per_day,
                                self.snow_steps_per_day))

    def require_supported(self):
        """Raise with every refusal at once, not the first."""
        if self.refused:
            raise ValueError('%s asks for what vic_classic does not read:\n  '
                             % os.path.basename(self.gp.path)
                             + '\n  '.join(self.refused))

    def summary(self):
        lines = ['classic driver options from %s' % self.gp.path,
                 '  soil        %s' % self.soil_file,
                 '  veglib      %s' % self.veglib_file,
                 '  vegparam    %s (ROOT_ZONES %d, LAI %s, FCAN %s, ALB %s)'
                 % (self.vegparam_file, self.root_zones, self.lai_src,
                    self.fcan_src, self.alb_src),
                 '  snow bands  %d%s' % (self.snow_bands,
                                         ' from %s' % self.snowband_file
                                         if self.snowband_file else ''),
                 '  forcing     %s<lat>_<lon> (%d decimals), %s, %d steps/day '
                 'from %s, columns %s'
                 % (self.forcing_prefix, self.grid_decimal,
                    self.force_format or 'FORMAT UNSET',
                    self.force_steps_per_day, self.force_start,
                    ' '.join(self.force_columns)),
                 '  soil columns depend on: ORGANIC_FRACT %s, SPATIAL_SNOW %s, '
                 'SPATIAL_FROST %s, JULY_TAVG_SUPPLIED %s'
                 % (self.organic_fract, self.spatial_snow,
                    self.spatial_frost, self.july_tavg_supplied)]
        if self.refused:
            lines.append('  REFUSED:')
            lines.extend('    ' + r for r in self.refused)
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# The soil parameter file.
# ---------------------------------------------------------------------------
def soil_columns(opts):
    """The soil file's columns, in the order read_soilparam.c reads them.

    Each entry is (name, count). `count` is 1 or NLAYER. This is the C's read
    sequence (lines cited beside each), not a documented format: the user
    guide and the C have disagreed before, and the C is what runs.
    """
    n = opts.nlayer
    cols = [('run_cell', 1), ('gridcell', 1), ('lat', 1), ('lon', 1),   # :71-110
            ('infilt', 1), ('Ds', 1), ('Dsmax', 1), ('Ws', 1), ('c', 1),  # :113-167
            ('expt', n), ('Ksat', n), ('phi_s', n), ('init_moist', n),  # :170-227
            ('elev', 1), ('depth', n), ('avg_T', 1), ('dp', 1),         # :230-284
            ('bubble', n), ('quartz', n),                               # :287-322
            ('bulk_density', n), ('soil_density', n)]                   # :325-362
    if opts.organic_fract:                                              # :364-432
        cols += [('organic', n), ('bulk_density_org', n),
                 ('soil_density_org', n)]
    cols += [('off_gmt', 1), ('Wcr_FRACT', n), ('Wpwp_FRACT', n),       # :442-475
             ('rough', 1), ('snow_rough', 1), ('annual_prec', 1),       # :478-514
             ('resid_moist', n), ('fs_active', 1)]                      # :517-539
    if opts.spatial_snow:                                               # :542-552
        cols.append(('max_snow_distrib_slope', 1))
    if opts.spatial_frost:                                              # :558-568
        cols.append(('frost_slope', 1))
    if opts.july_tavg_supplied:                                         # :575-586
        cols.append(('July_Tavg', 1))
    return cols


def read_soil_file(path, opts, log=None):
    """Every row of the soil file as {column: array}, plus the checks VIC
    makes on each cell.

    One row per cell; the first column is the run flag (read_soilparam.c:
    71-77), and a cell whose flag is 0 is read and not run. Rows are kept in
    file order and *all* rows are returned -- which are active is the
    domain's decision, made the way `ImageDomain` makes it from `run_cell`.

    Tokens beyond the last column the options ask for are dropped, because
    the C tokenises one line and stops asking (Stehekin's file carries a 60th
    column, the July mean temperature, under JULY_TAVG_SUPPLIED FALSE). Too
    few tokens is the C's `log_err("Can't find values for ...")`, raised with
    the same column name.
    """
    say = _announcer(log)
    cols = soil_columns(opts)
    need = sum(c for _, c in cols)
    rows = []
    for ln, line in enumerate(_open_text(path, 'soil parameter file'), 1):
        s = line.split('#', 1)[0].strip()
        if not s:
            continue
        toks = s.split()
        if len(toks) < need:
            got = 0
            for name, c in cols:
                if got + c > len(toks):
                    break
                got += c
            raise ValueError("%s line %d: can't find values for %s -- the "
                             'row has %d columns and the options ask for %d '
                             '(read_soilparam.c stops here too)'
                             % (os.path.basename(path), ln, name, len(toks),
                                need))
        rows.append([float(t) for t in toks[:need]])
        if len(toks) > need and ln == 1:
            say('  %s: %d trailing column(s) per row are not read -- with '
                'ORGANIC_FRACT %s, SPATIAL_SNOW %s, SPATIAL_FROST %s, '
                'JULY_TAVG_SUPPLIED %s the classic driver reads %d columns '
                'and ignores the rest of the line'
                % (os.path.basename(path), len(toks) - need,
                   opts.organic_fract, opts.spatial_snow, opts.spatial_frost,
                   opts.july_tavg_supplied, need))
    if not rows:
        raise ValueError('%s holds no cells' % path)
    a = np.asarray(rows, dtype=np.float64)          # [cell, column]
    out, i = {}, 0
    for name, c in cols:
        out[name] = a[:, i] if c == 1 else a[:, i:i + c].T.copy()  # [layer, cell]
        i += c
    _check_soil(out, opts, os.path.basename(path), say)
    return out


def _check_soil(s, opts, base, say):
    """The per-cell validation read_soilparam.c makes, as errors where VIC
    `log_err`s and as announcements where it silently adjusts."""
    def bad(mask, msg):
        m = np.asarray(mask)
        if m.any():
            cells = np.asarray(s['gridcell'])[np.argwhere(m)[:, -1]]
            raise ValueError('%s: %s (cell %d%s; read_soilparam.c)'
                             % (base, msg, int(cells[0]),
                                ' and %d more' % (cells.size - 1)
                                if cells.size > 1 else ''))
    bad(s['infilt'] <= 0, 'b_infilt <= 0; must be positive (:121-124)')
    bad(s['expt'] < 3.0, 'expt < 3.0 in some layer (:180-183)')
    bad(s['init_moist'] < 0, 'negative initial moisture (:223-226)')
    bad(s['bulk_density'] <= 0, 'mineral bulk density <= 0 (:335-338)')
    bad(s['soil_density'] <= 0, 'mineral soil density <= 0 (:352-355)')
    bad(s['bulk_density'] >= s['soil_density'],
        'bulk density >= soil density (:356-361)')
    if opts.full_energy or opts.frozen_soil:
        bad(s['bubble'] < 0, 'bubbling pressure < 0 with FULL_ENERGY or '
                             'FROZEN_SOIL (:297-302)')
    if opts.full_energy:
        bad((s['avg_T'] > 100) | (s['avg_T'] < -50),
            'average soil temperature not in -50..100 C with FULL_ENERGY '
            '(:268-273)')
        bad((s['quartz'] > 1) | (s['quartz'] < 0),
            'quartz content not a fraction with FULL_ENERGY (:316-321)')
    # :252-256 -- depths are rounded to the nearest millimetre before use.
    d = np.round(s['depth'] * MM_PER_M) / MM_PER_M
    if not np.array_equal(d, s['depth']):
        say('  %s: layer depths rounded to the nearest mm as VIC does '
            '(read_soilparam.c:252-256)' % base)
        s['depth'] = d
    bad(s['depth'] < MINSOILDEPTH, 'layer depth < %g m (:617-621)' % MINSOILDEPTH)
    if s['depth'].shape[0] >= 2:
        bad(s['depth'][0] > s['depth'][1], 'layer 1 deeper than layer 2 '
                                            '(:623-627)')
    # :604-606 -- a MISSING residual moisture takes VIC's default.
    miss = s['resid_moist'] == MISSING
    if miss.any():
        say('  %s: resid_moist is MISSING in %d layer value(s); using VIC\'s '
            'default SOIL_RESID_MOIST = %g (read_soilparam.c:604-606)'
            % (base, int(miss.sum()), SOIL_RESID_MOIST))
        s['resid_moist'] = np.where(miss, SOIL_RESID_MOIST, s['resid_moist'])
    # :644-664 -- the wilting point must sit between residual and critical.
    porosity = 1.0 - s['bulk_density'] / s['soil_density']
    max_moist = s['depth'] * porosity * MM_PER_M
    wcr, wpwp = s['Wcr_FRACT'] * max_moist, s['Wpwp_FRACT'] * max_moist
    bad(wpwp > wcr, 'Wpwp_FRACT > Wcr_FRACT (:647-653)')
    bad(wpwp < s['resid_moist'] * s['depth'] * MM_PER_M,
        'wilting point below residual moisture (:654-663)')


def cell_area_vic(lat, lon, resolution, equal_area=False):
    """Cell area in m2 exactly as compute_cell_area.c does it.

    EQUAL_AREA: `resolution` is a side in km and the area is its square
    (:47-48). Otherwise the cell is `resolution` degrees on a side and the
    area is summed over ten latitude strips, each the great-circle width at
    the strip's southern edge times one tenth of the cell's height
    (:51-67), with `get_dist` the spherical law of cosines on CONST_REARTH
    (get_dist.c:49-59). Reproduced rather than replaced by the closed-form
    band area `ImageBatch` uses, because the two differ at the 1e-5 level
    and this module's job is to give the classic driver's number.
    """
    if equal_area:
        return np.full_like(np.asarray(lat, dtype=np.float64),
                            resolution * 1000.0 * 1000.0)

    def get_dist(lat1, lon1, lat2, lon2):
        d = 2.0 * math.pi / 360.0
        t1, p1, t2, p2 = d * lon1, d * lat1, d * lon2, d * lat2
        temp = (np.cos(p1) * np.cos(t1) * np.cos(p2) * np.cos(t2)
                + np.cos(p1) * np.sin(t1) * np.cos(p2) * np.sin(t2)
                + np.sin(p1) * np.sin(p2))
        return CONST_REARTH * np.arccos(np.minimum(temp, 1.0))

    lat, lon = np.abs(np.asarray(lat, dtype=np.float64)), \
        np.abs(np.asarray(lon, dtype=np.float64))
    start_lat = lat - resolution / 2.0
    right, left = lon + resolution / 2.0, lon - resolution / 2.0
    delta = get_dist(lat, lon, lat + resolution / 10.0, lon)
    area = np.zeros_like(lat)
    for _ in range(10):
        area = area + get_dist(start_lat, left, start_lat, right) * delta
        start_lat = start_lat + resolution / 10.0
    return area


class ClassicDomain(object):
    """The classic driver's domain: the rows of the soil file whose run flag
    is set, with their positions and (when the global file says how) areas.

    There is no domain file in classic mode. `ImageDomain` gives `ImageBatch`
    five things -- `n_active`, `lat`, `lon`, `area`, `frac` -- and this gives
    the same five, from the soil file. `frac` is None: the classic driver has
    no notion of a partially-covered cell.

    **Area.** The classic driver computes it in `compute_cell_area.c` from
    RESOLUTION (and EQUAL_AREA), and with RESOLUTION absent the C's default of
    0 gives every cell an area of 0 -- VIC only uses it for lakes and for
    writing OUT_DISCHARGE, so a water-balance run never notices. An area of
    0 would make `ImageBatch`'s area weights 0/0, so here RESOLUTION absent
    gives `area = None`, and `ImageBatch` falls back to its spherical band
    formula on `basin.EARTH_RADIUS_M` (VIC's CONST_REARTH, 6.37122e6 m) with
    the grid spacing inferred from the latitudes. The fallback is announced.
    """

    def __init__(self, soil_path, opts, log=None):
        say = _announcer(log)
        self.path = soil_path
        self.opts = opts
        self.table = read_soil_file(soil_path, opts, log=log)
        run = self.table['run_cell']
        active = run == 1
        n_off = int((~active).sum())
        if n_off:
            say('  %s: %d of %d cells have run_cell 0 and are not run '
                '(read_soilparam.c:72-77)' % (os.path.basename(soil_path),
                                              n_off, run.size))
        self.rows = np.flatnonzero(active)
        self.n_active = int(self.rows.size)
        if self.n_active == 0:
            raise ValueError('%s: no cell has run_cell = 1' % soil_path)
        self.run_cell = run
        self.selected_on = 'run_cell'
        self.gridcell = self.table['gridcell'][self.rows].astype(np.int64)
        self.lat = self.table['lat'][self.rows]
        self.lon = self.table['lon'][self.rows]
        self.frac = None
        if opts.resolution > 0:
            self.area = cell_area_vic(self.lat, self.lon, opts.resolution,
                                      opts.equal_area)
            self.area_source = ('EQUAL_AREA %g km' if opts.equal_area
                                else 'RESOLUTION %g deg') % opts.resolution
        else:
            self.area = None
            self.area_source = 'none'
            say('  %s: the global file gives no RESOLUTION, so the classic '
                'driver would carry cell_area = 0 (compute_cell_area.c with '
                'global_param.resolution = 0); cell areas will come from '
                "ImageBatch's spherical formula on the inferred grid spacing"
                % os.path.basename(soil_path))

    def take(self, a):
        """A [..., all rows] array reduced to [..., active cell]."""
        return np.asarray(a)[..., self.rows]

    def __repr__(self):
        return ('<ClassicDomain %d active of %d cells, lat %.4f..%.4f, '
                'area %s>' % (self.n_active, self.run_cell.size,
                              float(self.lat.min()), float(self.lat.max()),
                              self.area_source))


# ---------------------------------------------------------------------------
# The vegetation library and the per-cell vegetation parameter file.
# ---------------------------------------------------------------------------
def read_veglib(path, opts, log=None):
    """The vegetation library as an ordered list of class records.

    read_veglib.c:52-60 counts a record for every line whose first character
    is a digit, then reads each with fscanf in this order (:71-161):

        class overstory rarc rmin LAI[12] [fcanopy[12] if VEGLIB_FCAN]
        albedo[12] roughness[12] displacement[12] wind_h RGL rad_atten
        wind_atten trunk_ratio [carbon columns if VEGLIB_PHOTO]

    and `fgets` the rest of the line as a comment (:196), which is where the
    class name lives. The bare-soil record VIC appends (:210-229) is appended
    here too, with the same values; its roughness and displacement are
    MISSING until the soil file supplies `rough` for the cell
    (read_soilparam.c:489-494), which `ClassicParams` does.
    """
    say = _announcer(log)
    base = os.path.basename(path)
    n_fixed = 4 + MONTHS * (4 + (1 if opts.veglib_fcan else 0)) + 5
    # read_veglib.c:162-192, the seven fields VEGLIB_PHOTO adds after
    # trunk_ratio: the photosynthetic pathway, the maximum carboxylation
    # rate, one of the electron transport rate (C3) or the CO2
    # specificity (C4), the light-use efficiency, the nitrogen-scaling
    # flag, and the two that set how a saturated top soil layer inhibits
    # production.
    n_photo = 7 if opts.veglib_photo else 0
    n_fixed += n_photo
    classes = []
    for ln, line in enumerate(_open_text(path, 'vegetation library'), 1):
        if not line or not line[0].isdigit():         # :56, :69
            continue
        toks = line.split()
        if len(toks) < n_fixed:
            raise ValueError('%s line %d: %d fields where the options '
                             '(VEGLIB_FCAN %s) need %d (read_veglib.c:204-207 '
                             '"make sure the file has the right number of '
                             'columns")' % (base, ln, len(toks),
                                            opts.veglib_fcan, n_fixed))
        v = [float(t) for t in toks[:n_fixed]]
        i = 0

        def take(n):
            nonlocal i
            out = v[i:i + n]
            i += n
            return out
        rec = {'veg_class': int(v[0])}
        i = 1
        rec['overstory'] = 0.0 if take(1)[0] == 0 else 1.0    # :72-78
        rec['rarc'] = take(1)[0]
        rec['rmin'] = take(1)[0]
        rec['LAI'] = take(MONTHS)
        rec['fcanopy'] = [1.0] * MONTHS                       # :92-95
        if opts.veglib_fcan:                                  # :96-109
            f = take(MONTHS)
            if opts.fcan_src != 'FROM_DEFAULT':
                if any(x < 0 or x > 1 for x in f):
                    raise ValueError('%s class %d: fcanopy outside 0..1 '
                                     '(read_veglib.c:101-106)'
                                     % (base, rec['veg_class']))
                rec['fcanopy'] = f
        rec['albedo'] = take(MONTHS)                          # :110-116
        if any(x < 0 or x > 1 for x in rec['albedo']):
            raise ValueError('%s class %d: albedo outside 0..1 '
                             '(read_veglib.c:112-115)' % (base, rec['veg_class']))
        rec['veg_rough'] = take(MONTHS)                       # :117-119
        rec['displacement'] = take(MONTHS)                    # :122-132
        for m in range(MONTHS):
            if rec['LAI'][m] > 0 and rec['displacement'][m] <= 0:
                raise ValueError('%s class %d month %d: leaves (LAI %g) but '
                                 'no displacement (read_veglib.c:127-131)'
                                 % (base, rec['veg_class'], m + 1,
                                    rec['LAI'][m]))
        rec['wind_h'] = take(1)[0]                            # :133-139
        if rec['overstory'] and rec['wind_h'] < max(rec['displacement']):
            raise ValueError('%s class %d: wind_h %g below the maximum '
                             'displacement %g for an overstory class '
                             '(read_veglib.c:134-139)'
                             % (base, rec['veg_class'], rec['wind_h'],
                                max(rec['displacement'])))
        rec['RGL'] = take(1)[0]                               # :140-149
        if rec['RGL'] < 0:
            raise ValueError('%s class %d: RGL < 0 (read_veglib.c:143-149)'
                             % (base, rec['veg_class']))
        rec['rad_atten'] = take(1)[0]                         # :150-157
        if not 0 <= rec['rad_atten'] <= 1:
            raise ValueError('%s class %d: rad_atten outside 0..1 '
                             '(read_veglib.c:152-157)' % (base, rec['veg_class']))
        rec['wind_atten'] = take(1)[0]
        rec['trunk_ratio'] = take(1)[0]
        # :162-192. The third field is MaxETransport for a C3 class and
        # CO2Specificity for a C4 one, and the C zeroes whichever it did
        # not read -- one column, two meanings, decided by the first.
        if opts.veglib_photo:
            ctype = take(1)[0]
            if ctype not in (0.0, 1.0):
                raise ValueError('%s class %d: Ctype %g is neither 0 (C3) '
                                 'nor 1 (C4) (read_veglib.c:163-174); the '
                                 'strings "C3" and "C4" VIC still accepts '
                                 'are deprecated and are not read here'
                                 % (base, rec['veg_class'], ctype))
            rec['Ctype'] = ctype
            rec['MaxCarboxRate'] = take(1)[0]
            third = take(1)[0]
            rec['MaxETransport'] = third if ctype == 0.0 else 0.0
            rec['CO2Specificity'] = 0.0 if ctype == 0.0 else third
            rec['LightUseEff'] = take(1)[0]
            rec['NscaleFlag'] = take(1)[0]
            rec['Wnpp_inhib'] = take(1)[0]
            rec['NPPfactor_sat'] = take(1)[0]
        rec['descr'] = ' '.join(toks[n_fixed:])
        classes.append(rec)
    if not classes:
        raise ValueError('%s: no class records (a record is a line starting '
                         'with a digit, read_veglib.c:56)' % path)
    ids = [c['veg_class'] for c in classes]
    if len(set(ids)) != len(ids):
        raise ValueError('%s: repeated class id; read_vegparam.c:459-463 '
                         'would silently take the last' % base)
    # :210-229 -- bare soil, class id one past the count, always last.
    classes.append({
        'veg_class': len(classes) + 1, 'overstory': 0.0, 'rarc': SOIL_RARC,
        'rmin': 0.0, 'LAI': [0.0] * MONTHS, 'fcanopy': [MIN_FCANOPY] * MONTHS,
        'albedo': [ALBEDO_BARE_SOIL] * MONTHS,
        'veg_rough': [MISSING] * MONTHS, 'displacement': [MISSING] * MONTHS,
        'wind_h': opts.wind_h, 'RGL': 0.0, 'rad_atten': 0.0,
        # :231-236, the defaults the C gives the bare-soil record.
        'Ctype': 0.0, 'MaxCarboxRate': 0.0, 'MaxETransport': 0.0,
        'CO2Specificity': 0.0, 'LightUseEff': 1.0, 'NscaleFlag': 0.0,
        'Wnpp_inhib': 1.0, 'NPPfactor_sat': 1.0,
        'wind_atten': 0.0, 'trunk_ratio': 0.0, 'descr': 'Bare Soil',
        'bare': True})
    say('  %s: %d classes plus VIC\'s bare-soil tile (read_veglib.c:210-229)'
        % (base, len(classes) - 1))
    return classes


def read_vegparam(path, opts, classes, log=None):
    """Per-cell tiles: {gridcell: [tile, ...]}, each tile as VIC holds it.

    read_vegparam.c is line-based: a `gridcell Nveg` header (:328), then per
    tile a root-zone line (:382-417)

        class Cv (zone_depth zone_fract) x ROOT_ZONES [sigma_slope lag_one
        fetch if BLOWING]

    followed by an LAI line if VEGPARAM_LAI, an fcanopy line if
    VEGPARAM_FCAN and an albedo line if VEGPARAM_ALB, each of twelve values
    (:489-619). Which of those lines is *used* is a separate switch: the LAI
    line is applied only under LAI_SRC FROM_VEGPARAM (:516), otherwise the
    library's LAI stands; likewise FCAN_SRC and ALB_SRC. Stehekin's global
    sets VEGPARAM_LAI TRUE and LAI_SRC FROM_VEGPARAM, so its leaf area is per
    cell and the library's is not used, which is also what its image
    parameter file carries. A NODATA_VH (-1) month keeps the library value
    (:519-520).

    The tile's class is resolved to the library *index* (:458-471), an
    unknown class is VIC's error, root fractions not summing to one are
    normalised with a warning (:433-442), and the cover sum is normalised
    when above 1 or in (0.99, 1) (:628-645) -- the image driver does neither
    (vic_init.c:1231 only asserts the sum), which is a real difference
    between VIC's drivers at the 1e-6 level on Stehekin and is reproduced,
    not smoothed over. What remains after the sum is the bare-soil tile's
    cover (:717-723).
    """
    say = _announcer(log)
    base = os.path.basename(path)
    by_id = {c['veg_class']: k for k, c in enumerate(classes)
             if not c.get('bare')}
    n_fields = 2 + 2 * opts.root_zones + (3 if opts.blowing else 0)  # :410-413
    lines = [l.split('#', 1)[0].rstrip('\n') for l in
             _open_text(path, 'vegetation parameter file')]
    lines = [l for l in lines if l.strip()]
    cells, pos, warned = {}, 0, {'root': 0, 'cv': 0}
    while pos < len(lines):
        head = lines[pos].split()
        pos += 1
        if len(head) < 2:
            raise ValueError('%s: expected "gridcell Nveg" at %r'
                             % (base, lines[pos - 1][:60]))
        gridcell, nveg = int(head[0]), int(head[1])
        if nveg < 0:
            raise ValueError('%s cell %d: %d tiles (read_vegparam.c:330-333)'
                             % (base, gridcell, nveg))
        tiles, cv_sum = [], 0.0
        for _ in range(nveg):
            if pos >= len(lines):
                raise ValueError('%s cell %d: unexpected end of file '
                                 '(read_vegparam.c:382-385)' % (base, gridcell))
            f = lines[pos].split()
            pos += 1
            if len(f) != n_fields:
                raise ValueError('%s cell %d: expecting %d fields but found '
                                 '%d in veg line %r (read_vegparam.c:414-417)'
                                 % (base, gridcell, n_fields, len(f),
                                    lines[pos - 1][:80]))
            cid, cv = int(f[0]), float(f[1])
            zd = [float(f[2 + 2 * j]) for j in range(opts.root_zones)]
            zf = [float(f[3 + 2 * j]) for j in range(opts.root_zones)]
            if sum(zd) <= 0:
                raise ValueError('%s cell %d: root zone depths sum to %g '
                                 '(read_vegparam.c:430-432)' % (base, gridcell,
                                                                sum(zd)))
            s = sum(zf)
            if s != 1.0:                                       # :433-442
                warned['root'] += 1
                zf = [x / s for x in zf]
            if cid not in by_id:
                raise ValueError('%s cell %d: vegetation class %d is not in '
                                 'the library (read_vegparam.c:464-468)'
                                 % (base, gridcell, cid))
            k = by_id[cid]
            lib = classes[k]
            tile = {'class_index': k, 'veg_class': cid, 'Cv': cv,
                    'root_depth': zd, 'root_fract': zf,
                    # :479-487 -- the library's monthly fields, to be
                    # overridden below by whichever lines the options apply.
                    'LAI': list(lib['LAI']), 'fcanopy': list(lib['fcanopy']),
                    'albedo': list(lib['albedo'])}
            if opts.blowing:                                   # :444-456
                j = 2 * opts.root_zones
                tile['sigma_slope'], tile['lag_one'], tile['fetch'] = (
                    float(f[2 + j]), float(f[3 + j]), float(f[4 + j]))
            for flag, src, key, want in (
                    (opts.vegparam_lai, opts.lai_src, 'LAI', 'FROM_VEGPARAM'),
                    (opts.vegparam_fcan, opts.fcan_src, 'fcanopy',
                     'FROM_VEGPARAM'),
                    (opts.vegparam_alb, opts.alb_src, 'albedo',
                     'FROM_VEGPARAM')):
                if not flag:
                    continue
                if pos >= len(lines):
                    raise ValueError('%s cell %d: unexpected end of file '
                                     'reading %s' % (base, gridcell, key))
                vals = lines[pos].split()
                pos += 1
                if len(vals) != MONTHS:                       # :510-514
                    raise ValueError('%s cell %d: expecting %d %s values but '
                                     'found %d' % (base, gridcell, MONTHS,
                                                   key, len(vals)))
                if src == want:                                # :516-535
                    for m in range(MONTHS):
                        x = float(vals[m])
                        if x != NODATA_VH:
                            tile[key][m] = x
            if lib['overstory'] and any(x == 0 for x in tile['LAI']):
                # :522-530 -- an overstory class must have leaves all year.
                raise ValueError('%s cell %d class %d: overstory class with '
                                 'LAI 0 in some month (read_vegparam.c:'
                                 '522-530)' % (base, gridcell, cid))
            cv_sum += cv
            tiles.append(tile)
        if cv_sum > 1.0 or 0.99 < cv_sum < 1.0:                # :628-645
            warned['cv'] += 1
            for t in tiles:
                t['Cv'] = t['Cv'] / cv_sum
            cv_sum = 1.0
        bare = 1.0 - cv_sum                                    # :717-723
        cells[gridcell] = {'tiles': tiles, 'bare_cv': max(bare, 0.0)}
    if warned['root']:
        say('  %s: root zone fractions did not sum to 1 in %d tile(s); '
            'normalised as read_vegparam.c:433-442 does'
            % (base, warned['root']))
    if warned['cv']:
        say('  %s: cover summed above 1 or between 0.99 and 1 in %d cell(s); '
            'rescaled to 1 as read_vegparam.c:628-645 does (the image driver '
            'does not, vic_init.c:1231)' % (base, warned['cv']))
    return cells


def read_snowband(path, opts, gridcells, elevation, log=None):
    """AreaFract, BandElev and Pfactor per cell, [band, cell].

    read_snowband.c: the file is one line per cell -- `gridcell` then
    SNOW_BAND area fractions, SNOW_BAND elevations, SNOW_BAND precipitation
    fractions (:836-902). A cell not in the file gets one band at the cell
    elevation with a warning (:828-834), which is the default read_soilparam.c
    :838-844 set up: AreaFract [1, 0, ...], BandElev all `elevation`, Pfactor
    all 1.

    What is returned is what the file says, as `ImageParams.bands` is what
    the parameter file says. VIC goes on to divide each Pfactor by its
    AreaFract (:903-910) and to lapse a Tfactor per band (:873-877); those are
    derived quantities the image path computes for itself when it needs them
    and are not stored on either reader. The one read-time adjustment kept is
    the sum normalisation (:847-853, :895-902), because it is announced and
    changes the numbers.
    """
    say = _announcer(log)
    base = os.path.basename(path)
    nb = opts.snow_bands
    rows = {}
    for ln, line in enumerate(_open_text(path, 'snow band file'), 1):
        s = line.split('#', 1)[0].strip()
        if not s:
            continue
        t = s.split()
        if len(t) < 1 + 3 * nb:
            raise ValueError('%s line %d: %d fields, SNOW_BAND %d needs %d'
                             % (base, ln, len(t), nb, 1 + 3 * nb))
        rows[int(t[0])] = [float(x) for x in t[1:1 + 3 * nb]]
    n = len(gridcells)
    area = np.zeros((nb, n)); elev = np.zeros((nb, n)); pf = np.zeros((nb, n))
    missing, renorm = [], 0
    for i, g in enumerate(gridcells):
        r = rows.get(int(g))
        if r is None:
            missing.append(int(g))
            area[0, i], elev[:, i], pf[:, i] = 1.0, elevation[i], 1.0
            continue
        a, e, p = (np.asarray(r[:nb]), np.asarray(r[nb:2 * nb]),
                   np.asarray(r[2 * nb:3 * nb]))
        if (a < 0).any() or (e < 0).any() or (p < 0).any():
            raise ValueError('%s cell %d: negative band value '
                             '(read_snowband.c:840-843, 859-862, 883-886)'
                             % (base, g))
        if ((p > 0) & (a == 0)).any():
            raise ValueError('%s cell %d: precipitation fraction > 0 where '
                             'the area fraction is 0 (read_snowband.c:887-891)'
                             % (base, g))
        if a.sum() != 1.0:
            renorm += 1
            a = a / a.sum()
        if p.sum() != 1.0:
            renorm += 1
            p = p / p.sum()
        area[:, i], elev[:, i], pf[:, i] = a, e, p
    if missing:
        say('  %s: %d cell(s) not in the snow band file (%s%s); one band at '
            'the cell elevation, as read_snowband.c:828-834 warns and does'
            % (base, len(missing), ', '.join(str(m) for m in missing[:5]),
               ', ...' if len(missing) > 5 else ''))
    if renorm:
        say('  %s: %d band fraction row(s) did not sum to 1 and were divided '
            'by their sum (read_snowband.c:847-853, 895-902)' % (base, renorm))
    return {'AreaFract': area, 'elevation': elev, 'Pfactor': pf}


class ClassicParams(ImageParams):
    """`ImageParams` built from the classic files.

    The inheritance is the guarantee, not a convenience: `porosity`,
    `max_moist`, `critical_moisture` and `utc_offset_hours` are the image
    reader's own methods running on arrays of the same shape, so the derived
    soil quantities cannot differ between the two paths by construction. The
    constructor is replaced wholesale because the parent's opens a NetCDF.

    Two departures from a literal transcription of the image file are
    deliberate, and both follow VIC's classic C rather than the image file:

    * **The bare-soil tile's roughness.** The image driver reads `veg_rough`
      and `displacement` for every class from the file (vic_init.c:236-258),
      and UW-Hydro's Stehekin file stores 0 for bare soil. The classic
      driver overwrites the bare-soil class from the soil file's `rough`:
      roughness = rough, displacement = rough * 0.667 / 0.123
      (read_soilparam.c:489-494). Two Stehekin cells carry 1.8% bare soil, so
      this is a difference between VIC's drivers that a flux can see, and the
      classic reader gives the classic driver's number.
    * **Cover normalisation** (read_vegparam.c:628-645), see `read_vegparam`.

    Absent tiles -- a class the library has and the cell does not -- get
    cover 0 and zero leaf area and roots, exactly as the image file encodes
    them; the class-level scalars (rmin, rarc, RGL, albedo, roughness,
    displacement, overstory, wind_h, attenuations, trunk ratio) are the
    library's for every class, which is also what the image file carries.
    """

    def __init__(self, domain, opts, veglib_path=None, vegparam_path=None,
                 snowband_path=None, log=None):
        say = _announcer(log)
        self.path = domain.path
        self.domain = domain
        self.opts = opts
        self.ds = None
        veglib_path = veglib_path or opts.veglib_file
        vegparam_path = vegparam_path or opts.vegparam_file
        snowband_path = snowband_path or opts.snowband_file
        t = domain.table
        n = domain.n_active

        # -- soil: the image reader's keys, the image reader's layouts -------
        # Conditional columns (frost_slope, read_soilparam.c:558-568 only
        # under SPATIAL_FROST) are absent from the table without the
        # option, as a variable the image file does not carry is absent
        # there; with the option they join `soil` as on the image path.
        self.soil = {k: domain.take(t[k])
                     for k in self.SOIL_2D + self.SOIL_3D if k in t}
        #: Columns the classic file carries that ImageParams does not keep on
        #: `soil` -- kept apart so `soil` has the same key set on both paths.
        self.extra = {k: domain.take(t[k]) for k in t
                      if k not in self.soil and k not in
                      ('run_cell', 'gridcell', 'lat', 'lon', 'off_gmt')}
        #: Hours from GMT as a float per cell. (xarray decodes the image
        #: file's `off_gmt`, whose units attribute says "hours", into a
        #: timedelta; the number is the same, the dtype is not.)
        self.off_gmt = domain.take(t['off_gmt'])
        #: The soil file's last column under JULY_TAVG_SUPPLIED
        #: (read_soilparam.c:574-586), or None.
        self.july_tavg = (np.asarray(domain.take(t['July_Tavg']),
                                     dtype=np.float64)
                          if 'July_Tavg' in t else None)
        self.gridcell = domain.gridcell
        self.n_layers = int(self.soil['depth'].shape[0])
        self.n_root = opts.root_zones
        self.n_bands = opts.snow_bands

        # -- vegetation --------------------------------------------------------
        classes = read_veglib(veglib_path, opts, log=log)
        self.veg_classes = classes
        self.veg_class_ids = [c['veg_class'] for c in classes]
        self.veg_descr = [c['descr'] for c in classes]
        nc = len(classes)
        self.n_veg = nc
        cells = read_vegparam(vegparam_path, opts, classes, log=log)
        missing = [int(g) for g in self.gridcell if int(g) not in cells]
        if missing:
            # read_vegparam.c:342-344 -- VIC stops.
            raise ValueError('%s: grid cell(s) %s not found (read_vegparam.c:'
                             '342-344)' % (os.path.basename(vegparam_path),
                                           ', '.join(str(m) for m in
                                                     missing[:8])))
        unused = len(cells) - (n - 0)
        if unused > 0:
            say('  %s describes %d cell(s) the soil file does not run'
                % (os.path.basename(vegparam_path), unused))

        v = {}
        v['Cv'] = np.zeros((nc, n))
        # read_vegparam.c:444-456, the three fields BLOWING adds to each
        # tile line. They are per tile and per cell, not monthly.
        if opts.blowing:
            for k in ('sigma_slope', 'lag_one', 'fetch'):
                v[k] = np.zeros((nc, n))
        for k in ('LAI', 'albedo', 'veg_rough', 'displacement', 'fcanopy'):
            v[k] = np.zeros((nc, MONTHS, n))
        for k in ('root_depth', 'root_fract'):
            v[k] = np.zeros((nc, opts.root_zones, n))
        scalars = ['rmin', 'rarc', 'RGL', 'overstory', 'trunk_ratio',
                   'wind_h', 'wind_atten', 'rad_atten']
        # The carbon library fields are per class like the rest, and are
        # on `veg` only when the file supplied them -- the same rule
        # fcanopy follows, so a run without VEGLIB_PHOTO has no key here
        # rather than a column of VIC's defaults pretending to be data.
        if opts.veglib_photo:
            scalars += ['Ctype', 'MaxCarboxRate', 'MaxETransport',
                        'CO2Specificity', 'LightUseEff', 'NscaleFlag',
                        'Wnpp_inhib', 'NPPfactor_sat']
        for k in scalars:
            v[k] = np.array([[c[k]] * n for c in classes], dtype=np.float64)
        # Class-level monthly fields are the library's for every class...
        for j, c in enumerate(classes):
            for k in ('albedo', 'veg_rough', 'displacement', 'fcanopy'):
                v[k][j] = np.asarray(c[k])[:, None]
            # ... and LAI too, where LAI_SRC says the library is the source;
            # under FROM_VEGPARAM an absent tile has no leaf area to state.
            if opts.lai_src == 'FROM_VEGLIB' and not c.get('bare'):
                v['LAI'][j] = np.asarray(c['LAI'])[:, None]
        # read_soilparam.c:489-494 -- bare soil's aerodynamics from `rough`.
        rough = self.soil['rough']
        v['veg_rough'][nc - 1] = rough[None, :]
        v['displacement'][nc - 1] = (rough * 0.667 / 0.123)[None, :]
        for i, g in enumerate(self.gridcell):
            cell = cells[int(g)]
            for tile in cell['tiles']:
                j = tile['class_index']
                v['Cv'][j, i] = tile['Cv']
                if opts.blowing:
                    for k in ('sigma_slope', 'lag_one', 'fetch'):
                        v[k][j, i] = tile[k]
                v['LAI'][j, :, i] = tile['LAI']
                v['albedo'][j, :, i] = tile['albedo']
                v['fcanopy'][j, :, i] = tile['fcanopy']
                v['root_depth'][j, :, i] = tile['root_depth']
                v['root_fract'][j, :, i] = tile['root_fract']
            v['Cv'][nc - 1, i] = cell['bare_cv']
        # fcanopy is on `veg` only when a file supplied it, as it is on the
        # image reader (`VEG_VARS` picks what the file has). VIC's default of
        # 1.0 for every class and MIN_FCANOPY for bare soil is what the array
        # holds otherwise, and nothing downstream reads it yet.
        if not (opts.veglib_fcan or opts.vegparam_fcan):
            self.fcanopy_default = v.pop('fcanopy')
        self.veg = v
        self.cells = cells

        # -- snow bands ------------------------------------------------------
        if opts.snow_bands > 1:
            self.bands = read_snowband(snowband_path, opts, self.gridcell,
                                       self.soil['elev'], log=log)
        else:
            # read_soilparam.c:838-844 -- one band at the cell elevation.
            self.bands = {'AreaFract': np.ones((1, n)),
                          'elevation': self.soil['elev'][None, :].copy(),
                          'Pfactor': np.ones((1, n))}
        bad = (self.veg['Cv'].sum(0) - 1.0)
        if np.abs(bad).max() > 1e-9:
            raise AssertionError('cover does not sum to 1 after VIC\'s rules: '
                                 'max |sum - 1| = %g' % np.abs(bad).max())

    def summary(self):
        return ('<ClassicParams %d cells, %d layers, %d veg classes (incl. '
                'bare soil), %d snow bands, %d root zones | soil %s | veg %s>'
                % (self.domain.n_active, self.n_layers, self.n_veg,
                   self.n_bands, self.n_root,
                   ','.join(sorted(self.soil)), ','.join(sorted(self.veg))))


# ---------------------------------------------------------------------------
# Forcing: one ASCII file per cell.
# ---------------------------------------------------------------------------
class ClassicForcing(object):
    """The classic driver's forcing, read for the domain's cells into the
    layout and units `ImageForcing.read` produces.

    One file per cell, named `<FORCING1><lat>_<lon>` with both coordinates
    printed to GRID_DECIMAL places (make_in_and_outfiles.c:47-58: the format
    is built as "%.<GRID_DECIMAL>f" and the two are joined with "_"). One
    record per line, one column per FORCE_TYPE line in the global file, in
    that order (read_atmos_data.c:226-243). There is no time axis in the file:
    the first record is FORCEYEAR/FORCEMONTH/FORCEDAY/FORCESEC and each is
    86400/FORCE_STEPS_PER_DAY seconds later.

    **Units.** The file holds precipitation in mm per step, temperature in C,
    radiation in W/m2, wind in m/s and -- the two that matter -- pressure and
    vapour pressure in kPa. VIC's classic driver multiplies those two by
    PA_PER_KPA on the way in (vic_force.c:136-138) and this does the same, so
    `fields['press']` and `fields['vp']` are in Pa here exactly as
    `ImageForcing.CONVERT` leaves them; without it every energy balance
    downstream would be off by a thousand. The conversion is announced, as
    the image reader announces its own.

    SKIP columns are read and dropped. Per-tile forcing columns (ALBEDO,
    LAI_IN, FCANOPY) are refused by `ClassicOptions`.
    """

    def __init__(self, prefix, domain, columns, steps_per_day, start,
                 grid_decimal=4, fmt='ASCII'):
        if fmt and fmt.upper() != 'ASCII':
            raise ValueError('FORCE_FORMAT %s is not read; only ASCII' % fmt)
        if not columns:
            raise ValueError('no FORCE_TYPE columns declared for %s' % prefix)
        if not steps_per_day:
            raise ValueError('FORCE_STEPS_PER_DAY is required to place the '
                             'records in time (get_global_param.c:981-985)')
        if start is None:
            raise ValueError('FORCEYEAR/FORCEMONTH/FORCEDAY are required; the '
                             'classic forcing file carries no time axis')
        self.prefix, self.domain = prefix, domain
        self.columns = list(columns)
        self.steps_per_day = int(steps_per_day)
        self.step_seconds = SEC_PER_DAY // self.steps_per_day
        self.start = start
        self.grid_decimal = int(grid_decimal)
        fmt = '%%.%df' % self.grid_decimal
        self.files = [prefix + (fmt % lat) + '_' + (fmt % lon)
                      for lat, lon in zip(domain.lat, domain.lon)]
        absent = [f for f in self.files if not os.path.exists(f)]
        if absent:
            raise IOError('%d of %d forcing files are missing, first %s; the '
                          'name is FORCING1 + lat + "_" + lon to GRID_DECIMAL '
                          '= %d places (make_in_and_outfiles.c:47-58)'
                          % (len(absent), len(self.files), absent[0],
                             self.grid_decimal))

    def read(self, dtype=None, device=None, log=None, model_start=None,
             model_end=None):
        """{name: tensor[time, cell]} in canonical units, and the time axis.

        `model_start`/`model_end` (dates or datetimes) select the model
        period the way read_atmos_data.c:57-72 does: the number of records
        to skip must be whole, and the file must hold enough records to reach
        the end (:246-255). Without them every record is returned.
        """
        import torch
        dtype = dtype or torch.float64
        say = _announcer(log)
        want = [(i, c) for i, c in enumerate(self.columns) if c != 'SKIP']
        per_cell, nrec = [], None
        for f in self.files:
            a = np.loadtxt(f, dtype=np.float64, ndmin=2)
            if a.shape[1] != len(self.columns):
                raise ValueError('%s has %d columns and the global file '
                                 'declares %d FORCE_TYPE lines (%s); fscanf '
                                 'would read across lines here without '
                                 'complaint' % (os.path.basename(f), a.shape[1],
                                                len(self.columns),
                                                ' '.join(self.columns)))
            if nrec is None:
                nrec = a.shape[0]
            elif a.shape[0] != nrec:
                raise ValueError('%s has %d records where %s has %d'
                                 % (os.path.basename(f), a.shape[0],
                                    os.path.basename(self.files[0]), nrec))
            per_cell.append(a)
        stack = np.stack(per_cell, axis=-1)          # [time, column, cell]
        skip = 0
        step = dt.timedelta(seconds=self.step_seconds)
        if model_start is not None:
            ms = (model_start if isinstance(model_start, dt.datetime)
                  else dt.datetime.combine(model_start, dt.time()))
            off = (ms - self.start).total_seconds()
            if off < 0 or off % self.step_seconds:
                raise ValueError('model start %s is not on a forcing record '
                                 '(first record %s, %d s apart; '
                                 'read_atmos_data.c:63-72)'
                                 % (ms, self.start, self.step_seconds))
            skip = int(off // self.step_seconds)
        n = nrec - skip
        if model_end is not None:
            me = (model_end if isinstance(model_end, dt.datetime)
                  else dt.datetime.combine(model_end, dt.time()))
            # VIC's ENDDAY is inclusive: nrecs runs to the end of that day.
            end = me + dt.timedelta(days=1) if not isinstance(
                model_end, dt.datetime) else me
            n = int((end - (self.start + skip * step)).total_seconds()
                    // self.step_seconds)
            if skip + n > nrec:
                raise ValueError('not enough records in the forcing (%d) to '
                                 'reach %s from %s at %d steps/day '
                                 '(read_atmos_data.c:246-255)'
                                 % (nrec, model_end, self.start,
                                    self.steps_per_day))
        stack = stack[skip:skip + n]
        t0 = np.datetime64(self.start + skip * step, 'ns')
        stamps = t0 + np.arange(n) * np.timedelta64(self.step_seconds, 's')
        stamps = stamps.astype('datetime64[ns]')
        fields = {}
        for i, c in want:
            canon = ImageForcing.CANONICAL.get(c, c.lower())
            v = stack[:, i, :]
            if c in ('PRESSURE', 'VP'):
                v = v * PA_PER_KPA
                say('  %s: %s is kPa in the classic forcing; multiplied by '
                    '%g to Pa as vic_force.c:136-138 does'
                    % (os.path.basename(self.prefix) or 'forcing', c,
                       PA_PER_KPA))
            fields[canon] = torch.as_tensor(np.ascontiguousarray(v),
                                            dtype=dtype, device=device)
        say('  %d forcing file(s), %d records each at %d steps/day from %s'
            % (len(self.files), n, self.steps_per_day, str(stamps[0])[:19]))
        return fields, stamps

    def __repr__(self):
        return ('<ClassicForcing %d file(s), %d cells, %d steps/day from %s>'
                % (len(self.files), self.domain.n_active, self.steps_per_day,
                   self.start))


# ---------------------------------------------------------------------------
# Putting a classic set-up beside an image one.
# ---------------------------------------------------------------------------
def match_cells(classic, image, tol=1e-6):
    """Align two domains on latitude and longitude.

    Returns (i_classic, i_image, dropped_classic, dropped_image): index
    arrays that put the matched cells in the classic file's order, and the
    positions each side has that the other does not. The two formats are
    matched on coordinates and not on `gridcell`, because the image file's
    `gridcell` variable is optional and a classic cell number is the file
    author's, not VIC's.

    The classic and image Stehekin samples turn out to hold the same 16
    cells; nothing is dropped there. A soil file with a cell outside the
    image mask, or an image mask with a cell the soil file lacks, is reported
    from here rather than silently intersected by whoever calls.
    """
    clat, clon = np.asarray(classic.lat), np.asarray(classic.lon)
    ilat, ilon = np.asarray(image.lat), np.asarray(image.lon)
    ic, ii = [], []
    for k in range(clat.size):
        d = np.abs(ilat - clat[k]) + np.abs(ilon - clon[k])
        j = int(np.argmin(d)) if d.size else -1
        if j >= 0 and d[j] <= tol and j not in ii:
            ic.append(k)
            ii.append(j)
    dropped_c = sorted(set(range(clat.size)) - set(ic))
    dropped_i = sorted(set(range(ilat.size)) - set(ii))
    return (np.asarray(ic, dtype=np.int64), np.asarray(ii, dtype=np.int64),
            dropped_c, dropped_i)


def open_classic(global_path, sample_data=None, log=None, strict=True):
    """Everything the classic global file names, read: (opts, domain, params,
    forcing). `strict` raises on any refused option; otherwise the refusals
    stay on `opts.refused` for the caller to look at."""
    gp = global_path if isinstance(global_path, GlobalParam) else \
        GlobalParam(global_path, sample_data=sample_data)
    opts = ClassicOptions(gp)
    if strict:
        opts.require_supported()
    domain = ClassicDomain(opts.soil_file, opts, log=log)
    params = ClassicParams(domain, opts, log=log)
    forcing = ClassicForcing(opts.forcing_prefix, domain, opts.force_columns,
                             opts.force_steps_per_day, opts.force_start,
                             grid_decimal=opts.grid_decimal,
                             fmt=opts.force_format)
    return opts, domain, params, forcing
