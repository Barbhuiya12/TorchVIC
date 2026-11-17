#!/usr/bin/env python3
"""Read VIC 5's own global-file parser and write down every option it knows.

`torchvic.io` could carry two
hand-written sets -- the options it recognised and the options it knew it did
not implement -- and anything outside both was reported as unsupported. A
hand-written list has exactly one failure mode and it is the one that matters:
it drifts. VIC gains an option, nobody updates the set, and a global file that
turns on a physics we do not have reads as if it had turned on nothing.

So the list is not written here. It is *read from VIC*, out of the same
`get_global_param.c` the C model compiles, by matching the comparison VIC uses
to dispatch on a key:

    else if (strcasecmp("QUICK_FLUX", optstr) == 0) {
        sscanf(cmdstr, "%*s %s", flgstr);
        options.QUICK_FLUX = str_to_bool(flgstr);
    }

Three things come out of each branch, and all three are facts about VIC rather
than opinions about us:

* **The key**, and which drivers dispatch on it. The image driver is the one
  we ingest, but a user arriving from the classic driver will hand us a
  classic global file, and saying "that key is real but belongs to the other
  driver" is a better answer than "unrecognised".
* **Whether VIC refuses it.** `ATMOS_STEPS_PER_DAY` and `OUTPUT_FORCE` exist
  in the image driver only to call `log_err` and stop; `ARNO_PARAMS` and
  `NIJSSEN2001_BASEFLOW` exist only to tell you their replacement. A branch
  whose every path ends in `log_err` is a refusal, and a key VIC itself
  refuses is one we should refuse too rather than quietly accept.
* **Whether the value is a boolean.** A branch that calls `str_to_bool` takes
  TRUE/FALSE, and that decides how an unimplemented option is announced:
  `FROZEN_SOIL FALSE` asks for nothing and should stay quiet, while
  `FROZEN_SOIL TRUE` asks for a physics we do not have and must be said out
  loud. Taking this from the C rather than from a second hand-written set is
  the whole point.

What cannot be extracted is what TorchVIC does with each key. That is a
judgement about our own code and it lives in STATUS below, one line per
option, deliberately conservative: a key is `implemented` only when changing
its value changes the run in the way VIC would; `ignored` when we parse it and
it genuinely costs nothing to ignore -- output formatting, log destinations;
and `unsupported` for everything else, including options we half-implement
(one of VIC's two baseflow formulations, one of its two snow densities).
Parsing a key is not implementing it, and this file must never say otherwise.

The drift guard is tests/test_vic_options.py: it re-runs the
extraction and fails if VIC's source carries a key STATUS does not classify.

    tools/extract_vic_options.py               # write torchvic/data/
    tools/extract_vic_options.py --check       # report drift, write nothing

The VIC source read is pristine VIC 5.1.0 (whose global-file keys are the
superset of 5.0.1's): $TORCHVIC_VIC_SOURCE, else
$TORCHVIC_REFERENCE_DIR/VIC-5.1.0/vic.
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The pristine VIC tree read: $TORCHVIC_VIC_SOURCE (a `vic/` directory), else
#: $TORCHVIC_REFERENCE_DIR/VIC-5.1.0/vic.
VIC_ROOT = os.path.dirname(os.environ.get('TORCHVIC_VIC_SOURCE') or os.path.join(
    os.environ.get('TORCHVIC_REFERENCE_DIR',
                   os.path.join(REPO, 'reference')),
    'VIC-5.1.0', 'vic').rstrip(os.sep))
#: Drivers whose global-file parsers are read. `image` is the one TorchVIC
#: ingests; the other two are here so a key from a classic or CESM global file
#: is reported as belonging to another driver rather than as a typo.
DRIVERS = ('image', 'classic', 'cesm')
SOURCE = os.path.join('vic', 'drivers', '%s', 'src', 'get_global_param.c')
VERSION_H = os.path.join(VIC_ROOT, 'vic', 'drivers', 'shared_all', 'include',
                         'vic_version.h')
#: The image driver takes the second token of DOMAIN_TYPE and FORCE_TYPE lines
#: from separate files, so those vocabularies are extracted from where they
#: are actually parsed.
SUB_SOURCES = {
    'DOMAIN_TYPE': os.path.join('vic', 'drivers', 'shared_image', 'src',
                                'get_global_domain.c'),
    'FORCE_TYPE': os.path.join('vic', 'drivers', 'shared_image', 'src',
                               'set_force_type.c'),
}
JSON_OUT = os.path.join(REPO, 'torchvic', 'data', 'vic5_global_options.json')

# VIC dispatches on `optstr`, the first token of the line. Comparisons against
# `flgstr` in the same file are the option's *value* (DENS_SNTHRM, FROM_
# VEGPARAM) and must not be collected as keys, which is why `optstr` is part
# of the pattern rather than being matched loosely.
_KEY = re.compile(r'strcasecmp\(\s*"([A-Za-z0-9_]+)"\s*,\s*optstr\s*\)')
# Anything that records the value somewhere VIC will later read. A branch with
# none of these and a log_err is a refusal, not a parse.
_STORES = re.compile(r'\b(options|global_param|filenames|param_set|param)\s*\.'
                     r'|\bget_domain_type\b|\bset_force_type\b'
                     r'|\bparse_output_info\b|\bget_force_type\b')


def vic_version():
    """VIC's own version, so the inventory says what it was read from."""
    try:
        src = open(VERSION_H).read()
    except IOError:                                          # pragma: no cover
        return 'unknown'
    m = re.search(r'#define\s+SHORT_VERSION\s+"([^"]+)"', src)
    return m.group(1) if m else 'unknown'


def branches(src):
    """Every `strcasecmp(KEY, optstr)` branch, as (keys, body).

    Two shapes have to survive. The ordinary one is one key per branch. The
    other is VIC's alias form, where two keys share a body:

        else if ((strcasecmp("NOFLUX", optstr) == 0) ||
                 (strcasecmp("NO_FLUX", optstr) == 0)) {

    Both spellings mean the same option and must get the same classification,
    so keys are accumulated until a body actually appears -- the text between
    two alias comparisons is `) == 0) || (` and contains no opening brace,
    which is how the two cases are told apart without parsing C.
    """
    hits = list(_KEY.finditer(src))
    pending, out = [], []
    for i, m in enumerate(hits):
        pending.append(m.group(1))
        end = hits[i + 1].start() if i + 1 < len(hits) else len(src)
        chunk = src[m.end():end]
        brace = chunk.find('{')
        if brace < 0:
            continue                      # an alias; its body is the next one
        # The body runs to the brace that closes it. Taking it to the next
        # branch instead would swallow the `else if` that follows and make
        # every branch look as though it stored something.
        body, depth = [], 0
        for ch in chunk[brace:]:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    break
            body.append(ch)
        out.append((tuple(pending), ''.join(body)))
        pending = []
    return out


def read_driver(driver):
    """{key: {'rejected', 'boolean', 'noop'}} for one driver's parser."""
    path = os.path.join(VIC_ROOT, SOURCE % driver)
    src = open(path).read()
    found = {}
    for keys, body in branches(src):
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', body)      # drop the strings
        code = re.sub(r'//[^\n]*|/\*.*?\*/', '', code, flags=re.S)
        fact = {
            # A branch that only ever calls log_err refuses the key. VIC uses
            # this both for options belonging to another driver and for
            # options it has renamed, and in either case accepting the file
            # silently would be worse than saying so.
            'rejected': 'log_err' in code and not _STORES.search(code),
            'boolean': 'str_to_bool' in code,
            # `; // do nothing` -- recognised here, parsed somewhere else.
            'noop': not _STORES.search(code) and 'log_err' not in code,
        }
        for k in keys:
            found[k] = dict(fact)
    return found


def sub_options():
    """The second-token vocabularies: DOMAIN_TYPE names and FORCE_TYPE names.

    These are not global-file keys and are not classified, but they are the
    other half of what an image global file can say, and `vic_image` matches
    both by name. Recording them lets the drift guard see a forcing variable
    VIC learns to read that we do not.
    """
    out = {}
    for key, rel in SUB_SOURCES.items():
        path = os.path.join(VIC_ROOT, rel)
        if os.path.exists(path):
            out[key] = sorted(set(_KEY.findall(open(path).read())))
    return out


# ---------------------------------------------------------------------------
# What TorchVIC does with each option.
#
# This is the one hand-maintained thing here and it is a statement about our
# code, not about VIC's. The rule for `implemented` is that setting the option
# changes the run the way VIC would change it; everything weaker is `ignored`
# or `unsupported`. Where we implement one of VIC's several choices the status
# is `unsupported` with the choice named, because a file selecting the other
# one would otherwise get our one silently.
#
# `ignored` is reserved for keys whose effect is on something TorchVIC does not
# do at all -- writing VIC's history files, choosing a log directory. Ignoring
# those costs nothing. Ignoring a physics option costs the physics.
# ---------------------------------------------------------------------------
IMPLEMENTED, IGNORED, UNSUPPORTED = 'implemented', 'ignored', 'unsupported'

STATUS = {
    # -- the six keys a TorchVIC image run actually rides on ----------------
    'DOMAIN': (IMPLEMENTED, 'the domain file is opened and its mask selects '
                            'the active cells (vic_image.ImageDomain)'),
    'DOMAIN_TYPE': (IMPLEMENTED, 'every domain variable name is taken from '
                                 'the file, not assumed'),
    'PARAMETERS': (IMPLEMENTED, 'soil, vegetation and band fields are read '
                                'from it (vic_image.ImageParams)'),
    'FORCING1': (IMPLEMENTED, 'the forcing prefix is globbed and every year '
                              'read (vic_image.ImageForcing)'),
    'FORCE_TYPE': (IMPLEMENTED, 'the NetCDF variable holding each forcing is '
                                'taken from the file'),
    'RUNOFF_STEPS_PER_DAY': (IMPLEMENTED, 'passed to column.SoilParams and '
                                          'sub-steps the runoff solution'),

    # -- time ---------------------------------------------------------------
    'MODEL_STEPS_PER_DAY': (
        IGNORED, 'the model step (torchvic.case.load): the forcing is '
                 'aggregated, or disaggregated by MTCLIM, onto it'),
    'SNOW_STEPS_PER_DAY': (
        IGNORED, 'read and reported; the snow model steps with the model '
                 'step, not at this rate'),
    'STARTYEAR': (IGNORED, 'parsed and exposed as GlobalParam.start_date; the '
                           'run covers whatever the forcing files hold'),
    'STARTMONTH': (IGNORED, 'see STARTYEAR'),
    'STARTDAY': (IGNORED, 'see STARTYEAR'),
    'STARTSEC': (IGNORED, 'sub-daily start offset; the model steps daily'),
    'ENDYEAR': (IGNORED, 'parsed and exposed as GlobalParam.end_date; the run '
                         'is not clipped to it'),
    'ENDMONTH': (IGNORED, 'see ENDYEAR'),
    'ENDDAY': (IGNORED, 'see ENDYEAR'),
    'NRECS': (IGNORED, 'the record count is taken from the forcing time axis'),
    'CALENDAR': (IGNORED, 'dates come from the forcing file time axis, which '
                          'carries its own calendar attribute; '
                          'torchvic.io.history writes it as the history files\' '
                          'time calendar'),
    'OUT_TIME_UNITS': (IMPLEMENTED, 'the time units of the history files '
                                    'torchvic.io.history writes (default DAYS, '
                                    'since 0001-01-01 as VIC)'),

    # -- files --------------------------------------------------------------
    'FORCING2': (IMPLEMENTED, 'the vegetation history file (io.veghist): '
                              'LAI, fcanopy and albedo for the *_SRC set to '
                              'FROM_VEGHIST; VIC reads no other variable '
                              'from it'),
    'CONSTANTS': (IMPLEMENTED, 'read by GlobalParam.constants(): '
                               'VEG_LAI_WATER_FACTOR is applied '
                               '(p[\'lai_water_factor\']); any other '
                               'constant in the file is refused by name, not '
                               'ignored'),
    'ROUT_PARAM': (UNSUPPORTED, 'routing is a unit hydrograph over one outlet '
                                '(torchvic.model.routing), not RVIC\'s parameter '
                                'file'),
    'INIT_STATE': (UNSUPPORTED, 'no state is restored; every run starts from '
                                'torchvic.model.basin.initial_states'),
    'STATENAME': (UNSUPPORTED, 'no state file is written'),
    'STATEYEAR': (UNSUPPORTED, 'no state file is written'),
    'STATEMONTH': (UNSUPPORTED, 'no state file is written'),
    'STATEDAY': (UNSUPPORTED, 'no state file is written'),
    'STATESEC': (UNSUPPORTED, 'no state file is written'),
    'STATE_FORMAT': (UNSUPPORTED, 'no state file is written'),
    'RESULT_DIR': (IMPLEMENTED, 'where torchvic.io.history writes the OUTFILE '
                                'streams (torchvic run; '
                                '--result-dir overrides)'),
    'LOG_DIR': (IGNORED, 'logging goes to the caller\'s log function'),

    # -- history output: torchvic.io.history, VIC's parse_output_info rules ------
    # basin.simulate itself still returns every output at the model step;
    # these decide what the history files hold, as VIC's do.
    'OUTFILE': (IMPLEMENTED, 'one history stream per OUTFILE '
                             '(torchvic.io.history.parse_streams), written by '
                             'torchvic run / Case.write_history'),
    'OUTVAR': (IMPLEMENTED, 'the stream\'s variables, their AGG_TYPE (5th '
                            'token; VIC\'s per-variable default otherwise) '
                            'and OUT_TYPE; a variable TorchVIC does not '
                            'compute is named and left out'),
    'AGGFREQ': (IMPLEMENTED, 'aggregation period, VIC\'s alarm: NSTEPS, '
                             'NSECONDS..NDAYS, NMONTHS, NYEARS, DATE, NEVER; '
                             'END and an absent AGGFREQ, which VIC 5 writes '
                             'nothing for, are one whole-run record and '
                             'every step'),
    'HISTFREQ': (IMPLEMENTED, 'history file splitting, VIC\'s write alarm'),
    'COMPRESS': (IMPLEMENTED, 'zlib level of netCDF-4 history files'),
    'OUT_FORMAT': (IMPLEMENTED, 'netCDF format of history files '
                                '(NETCDF4_CLASSIC, the default, is written '
                                'as netCDF-3 classic, as VIC\'s get_nc_mode '
                                'does)'),
    'OUTPUT_STEPS_PER_DAY': (IGNORED, 'history output step (CESM driver)'),
    # These seven are output keys, but VIC 5 deprecated them and stops on
    # them, so ignoring one would let a file through that VIC would refuse.
    'ALMA_OUTPUT': (UNSUPPORTED, 'deprecated in VIC 5; the classic driver '
                                 'refuses it'),
    'BINARY_OUTPUT': (UNSUPPORTED, 'deprecated in VIC 5; the classic driver '
                                   'refuses it'),
    'PRT_HEADER': (UNSUPPORTED, 'deprecated in VIC 5; the classic driver '
                                'refuses it'),
    'PRT_SNOW_BAND': (UNSUPPORTED, 'deprecated in VIC 5; the classic driver '
                                   'refuses it'),
    'MOISTFRACT': (UNSUPPORTED, 'deprecated in VIC 5; replaced by the '
                                'OUT_SOIL_*_FRAC output variables'),
    'OUT_DT': (UNSUPPORTED, 'deprecated in VIC 5; replaced by AGGFREQ'),
    'SKIPYEAR': (UNSUPPORTED, 'deprecated in the classic driver; the CESM '
                              'driver parses and drops it'),
    'GRID_DECIMAL': (IMPLEMENTED, 'decimal places in the classic forcing filename (make_in_and_outfiles.c:47-58), honoured by vic_classic'),
    'CONTINUEONERROR': (IGNORED, 'error policy; TorchVIC raises'),

    # -- energy balance and soil thermodynamics ------------------------------
    'FULL_ENERGY': (
        IMPLEMENTED,
        'the surface energy balance in torchvic.core.energy: one root find per '
        'tile per step, for the configuration with no snow on the surface, '
        'no overstory, on the QUICK_FLUX path and on the finite-difference '
        'profile; the runner reads the flag from the '
        'global file. Snow-coupled and overstory energy terms are not '
        'solved yet, and are said so rather than run on the wrong branch'),
    'CLOSE_ENERGY': (IMPLEMENTED, 'the canopy air temperature is solved '
                                  'and the ground/canopy balances iterate '
                                  'to closure (basin._ce_step)'),
    'FROZEN_SOIL': (IMPLEMENTED, 'the QUICK_FLUX path: the layer ice the '
                                 'unfrozen-water curve leaves, that ice in '
                                 'the column and in the thermal properties, '
                                 'and the latent heat of fusion in the '
                                 'surface balance; and the full node profile '
                                 '(QUICK_FLUX FALSE, which is what VIC '
                                 'switches to by default under FROZEN_SOIL), '
                                 'solved in torchvic.core.soiltemp'),
    'QUICK_FLUX': (IMPLEMENTED, 'TRUE is the two-node flux; FALSE solves '
                                'the finite-difference profile at every '
                                'trial surface temperature '
                                '(torchvic.core.soiltemp). Note '
                                'VIC sets it FALSE whenever FROZEN_SOIL '
                                'is TRUE (get_global_param.c:123-125); a '
                                'file that wants both must set QUICK_FLUX '
                                'TRUE after FROZEN_SOIL'),
    'QUICK_SOLVE': (IMPLEMENTED, 'bracketing on the reduced column, '
                                 're-solved on the full one where the '
                                 'surface crosses freezing'),
    'IMPLICIT': (IMPLEMENTED, 'the Newton solve of the finite-difference '
                              'profile, falling back to the sweep where it '
                              'fails (torchvic.core.soiltemp); '
                              'torchvic.core.implicit is the gradients '
                              'machinery it is not built on'),
    'EXP_TRANS': (IMPLEMENTED,
                  'the exponential node grid of the finite-difference '
                  'profile (torchvic.core.soiltemp); on the QUICK_FLUX '
                  'two-node path there is no grid for the transform to '
                  'space, and the flag is announced rather than read'),
    'NOFLUX': (IMPLEMENTED, 'zero-flux bottom boundary of the '
                            'finite-difference profile'),
    'NO_FLUX': (IMPLEMENTED, 'alias of NOFLUX'),
    'GRND_FLUX_TYPE': (IMPLEMENTED, 'GF_410 runs as the two-term Liang flux '
                                    'and GF_406 as the surface term alone, '
                                    'on both thermal paths (energy._residual, '
                                    'soiltemp.profile_ground_terms, '
                                    'energy.water_balance_ground_flux)'),
    'NODES': (IMPLEMENTED, 'the soil thermal node count of the '
                           'finite-difference profile; with QUICK_FLUX VIC '
                           'resets it to 3'),
    'TFALLBACK': (IMPLEMENTED,
                  'TRUE (default): a temperature solve with no root keeps '
                  'the previous temperature and is counted; FALSE stops at '
                  'the first such solve (surface, snow pack, canopy '
                  'foliage, canopy air), naming step, cell and solve '
                  '(core/tfallback.py)'),

    # -- snow ---------------------------------------------------------------
    'SNOW_BAND': (UNSUPPORTED, 'AreaFract, Pfactor and band elevation are '
                               'read from the parameter file but the run uses '
                               'one band per cell'),
    'SNOW_DENSITY': (UNSUPPORTED, 'only VIC\'s default DENS_BRAS is '
                                  'implemented (snow.snow_density_bras); a '
                                  'file asking for DENS_SNTHRM would get '
                                  'DENS_BRAS'),
    'MAX_SNOW_ALBEDO': (UNSUPPORTED, 'per-cell maximum snow albedo from the '
                                     'parameter file; TorchVIC uses a single '
                                     'new_snow_albedo run parameter'),
    'SPATIAL_SNOW': (IMPLEMENTED, 'calc_snow_coverage.c and '
                                  'advected_sensible_heat.c, classic driver '
                                  'only -- the soil file supplies '
                                  'max_snow_distrib_slope and VIC 5.0.1 '
                                  'never reads it in image mode'),
    'BLOWING': (IMPLEMENTED, 'CalcBlowingSnow.c in full: the saltation '
                             'shear root, the occurrence probability and '
                             'threshold, the saltation and suspension '
                             'layers with Romberg integration over '
                             'height, and the log-normal wind '
                             'distribution. Needs the vegetation file\'s '
                             'sigma_slope, lag_one and fetch'),
    'BLOWING_VAR_THRESHOLD': (IMPLEMENTED,
        'the variable threshold shear of Li and Pomeroy (1997), '
        'CalcBlowingSnow.c:645-650; the constant one is there too '),
    'BLOWING_CALC_PROB': (IMPLEMENTED,
        'the probability of blowing snow occurring, '
        'CalcBlowingSnow.c:582-616; off gives probability 1 '),
    'BLOWING_SIMPLE': (IMPLEMENTED,
        'the SBSM single-expression sublimation, '
        'CalcBlowingSnow.c:729-741 '),
    'BLOWING_FETCH': (IMPLEMENTED,
        'the fetch correction to the saltation transport, '
        'CalcBlowingSnow.c:750-752 and :787-789 '),
    'BLOWING_SPATIAL_WIND': (IMPLEMENTED,
        'the log-normal wind distribution cut into BLOWING_NUMINCS '
        'increments, CalcBlowingSnow.c:123-180 '),
    'COMPUTE_TREELINE': (IMPLEMENTED, 'the classic driver\'s treeline '
                         '(compute_treeline.c, put_data.c); VIC\'s image '
                         'driver stops on it'),
    'JULY_TAVG_SUPPLIED': (IMPLEMENTED, 'the cell\'s July air temperature the '
                           'treeline reads (read_soilparam.c:574-586)'),
    'MAX_SNOW_TEMP': (UNSUPPORTED, 'the rain/snow partition uses the '
                                   't_max_snow run parameter, not this key '
                                   '(classic driver)'),
    'MIN_RAIN_TEMP': (UNSUPPORTED, 'the rain/snow partition uses the '
                                   't_min_rain run parameter, not this key '
                                   '(classic driver)'),

    # -- soil and runoff -----------------------------------------------------
    'BASEFLOW': (IMPLEMENTED, 'ARNO runs as the column (soil.arno_baseflow); '
                              'NIJSSEN2001 reaches it as ARNO\'s Ds, Dsmax, '
                              'Ws, c (soil.nijssen2001_to_arno)'),
    'SPATIAL_FROST': (UNSUPPORTED, 'sub-grid frost distribution is not '
                                   'modelled; ice is identically zero'),
    'ORGANIC_FRACT': (UNSUPPORTED, 'organic soil fractions are not read from '
                                   'the parameter file'),
    'BULK_DENSITY_COMB': (UNSUPPORTED, 'bulk density is read as the mineral '
                                       'value; the combined-with-organic '
                                       'convention is not honoured'),
    'SHARE_LAYER_MOIST': (IMPLEMENTED,
                          'both cases of canopy_evap.c:287-362 -- CASE 1, '
                          'where a layer above its critical moisture makes '
                          'up the shortfall of a rooted layer below it, and '
                          'CASE 2, where each layer transpires on its own '
                          'resistance. vic_classic.py reads the flag and '
                          'basin.simulate(share_layer_moist=...) selects the '
                          'case, with VIC\'s default of TRUE when the key '
                          'is absent'),
    'EQUAL_AREA': (UNSUPPORTED, 'cell areas come from the domain file '
                                '(classic driver)'),
    'RESOLUTION': (UNSUPPORTED, 'grid resolution comes from the domain file '
                                '(classic driver)'),

    # -- vegetation and canopy ------------------------------------------------
    'LAI_SRC': (UNSUPPORTED, 'leaf area is always taken from the parameter '
                             'file\'s LAI; the climatology and default '
                             'sources are not implemented'),
    'FCAN_SRC': (UNSUPPORTED, 'canopy fraction is not used by the mosaic'),
    'ALB_SRC': (UNSUPPORTED, 'the parameter file\'s vegetation albedo is read '
                             'but the surface albedo is not driven from it'),
    'VEGCOVER_SRC': (UNSUPPORTED, 'earlier spelling of FCAN_SRC (classic '
                                  'driver)'),
    'VEGPARAM_LAI': (IMPLEMENTED, 'LAI lines in the classic vegetation file, read by vic_classic'),
    'VEGPARAM_ALB': (IMPLEMENTED, 'albedo lines in the classic vegetation file, read by vic_classic'),
    'VEGPARAM_FCAN': (IMPLEMENTED, 'fcanopy lines in the classic vegetation file, read by vic_classic'),
    'VEGPARAM_VEGCOVER': (UNSUPPORTED, 'classic-driver vegetation library '
                                       'layout'),
    'VEGLIB_PHOTO': (IMPLEMENTED,
                     'the seven photosynthesis columns read_veglib.c:162-192 '
                     'adds -- Ctype, MaxCarboxRate, MaxETransport, '
                     'CO2Specificity, NscaleFlag, Wnpp_inhib, NPPfactor_sat '
                     '-- are read by both the classic library reader and the '
                     'image parameter reader, and are what CARBON runs on'),
    'VEGLIB_FCAN': (IMPLEMENTED, 'fcanopy column in the classic library, read by vic_classic'),
    'VEGLIB_VEGCOVER': (UNSUPPORTED, 'classic-driver vegetation library '
                                     'layout'),
    'ROOT_ZONES': (IMPLEMENTED, 'root zone count for the classic vegetation file, honoured by vic_classic'),
    'RC_MODE': (IMPLEMENTED, 'RC_JARVIS (default) and RC_PHOTO, the canopy '
                             'resistance from photosynthetic demand '
                             '(canopy_evap.c:296-307, penman.c calc_rc_ps); '
                             'RC_PHOTO needs CARBON, as VIC resets it '
                             'without. Verified against VIC on Walnut Gulch '
                             '(verification/matrix.py section 14)'),
    'CANOPY_LAYERS': (IMPLEMENTED, 'the canopy is split into Ncanopy layers '
                                   'of equal leaf area for the carbon step, '
                                   'as read_vegparam.c:124-128 splits it, '
                                   'with the absorbed PAR and the nitrogen '
                                   'scaling computed per layer (faparl.c, '
                                   'calc_Nscale_factors.c). Three by '
                                   'default, as VIC has it; '
                                   'basin.simulate takes '
                                   'carbon_options={\'ncanopy\': n}'),
    'CARBON': (IMPLEMENTED, 'canopy_assimilation and photosynth verified '
                            'against VIC\'s own compiled functions to '
                            '1e-12 (torchvic.core.carbon), '
                            'called from the step loop at the point '
                            'surface_fluxes.c:732 calls them -- after the '
                            'water and energy are settled, from the canopy '
                            'resistance the transpiration already computed. '
                            'With VIC\'s default RC_MODE of RC_JARVIS the '
                            'carbon step only reads that resistance, so '
                            'every water and energy row is bitwise '
                            'unchanged with CARBON on '
                            '(tests/test_carbon_run.py). The CATM, '
                            'PAR and FDIR forcings VIC 5 demands and no '
                            'sample data has are derived as '
                            'initialize_atmos.c:1588-1700 derives them '
                            '(torchvic.forcing.daily, carbon=True). VIC 5\'s '
                            'photosynth.c uses a per-kilomole gas constant '
                            'in nine per-mole formulas, and '
                            'compute_soil_resp.c divides two of three pool '
                            'turnover times by hours where dt is seconds; '
                            'both are reproduced by default and corrected '
                            'on request (basin.simulate carbon_options). '
                            'The three soil pools need the surface energy '
                            'balance for their node temperatures and a soil '
                            'file with `bubble` for the lumped water table; '
                            'without either the vegetation fluxes run and '
                            'the pool rows are absent rather than zero'),
    'WIND_H': (
        UNSUPPORTED,
        'the wind measurement height is hard-wired at basin.MEAS_HEIGHT = 2 m '
        'while a real global file says 10 m, so the aerodynamic resistance '
        'is computed at the wrong height'),

    # -- lakes ---------------------------------------------------------------
    # The physics is complete and verified; what is missing is the wiring
    # into basin.simulate, which carries one soil column per cell and not
    # the lake-plus-wetland pair the option splits a cell into. LAKES
    # therefore stays UNSUPPORTED so a global file asking for it is told
    # so, rather than accepted and silently run without a lake.
    'LAKES': (UNSUPPORTED,
              'all of lakes.eb.c, ice_melt.c, water_energy_balance.c, '
              'water_under_ice.c and IceEnergyBalance.c is implemented and '
              'verified against VIC\'s own compiled functions (torchvic.core.lake): '
              'the thermal profile and its convective '
              'mixing, the open water and under-ice energy balances, the '
              'ice and its snowpack, the basin geometry, the water balance '
              'and the area the lake trades with the wetland. '
              'torchvic.core.lake.lake_step runs one lake tile a step, and a '
              '720-step run tracks VIC to 4e-07 on the lake volume. What is '
              'NOT here is the coupling into basin.simulate, which carries '
              'one soil column per grid cell rather than the lake-plus-'
              'wetland pair this option splits a cell into, so a global '
              'file cannot yet reach it'),
    'LAKE_PROFILE': (UNSUPPORTED,
                     'read_lakeparam and compute_lake_params are '
                     'implemented for both the given and the generated '
                     'depth-area curve and are exact against VIC '
                     '(torchvic.core.lake.read_lakeparam); the option is '
                     'unreachable only because LAKES is'),
    'LAKE_NODES': (UNSUPPORTED,
                   'the node grid is implemented (MAX_LAKE_NODES = 20, '
                   'compute_derived_lake_dimensions); the option is '
                   'unreachable only because LAKES is'),

    # -- forcing handling ----------------------------------------------------
    'CORRPREC': (UNSUPPORTED, 'gauge undercatch correction is not applied'),
    'AERO_RESIST_CANSNOW': (UNSUPPORTED, 'the aerodynamic resistance of a '
                                         'snow-covered canopy follows one '
                                         'formulation and the option does not '
                                         'select it'),
    'FORCE_DT': (UNSUPPORTED, 'forcing step (classic driver); the step comes '
                              'from the forcing time axis'),
    'FORCE_STEPS_PER_DAY': (IMPLEMENTED, 'sets the classic forcing time axis in vic_classic.ClassicForcing'),
    'FORCE_FORMAT': (IMPLEMENTED, 'ASCII is read by vic_classic.ClassicForcing; BINARY is refused by name'),
    'FORCE_ENDIAN': (UNSUPPORTED, 'binary forcing byte order (classic '
                                  'driver)'),
    'FORCEYEAR': (IMPLEMENTED, 'classic forcing start, honoured by vic_classic.ClassicForcing'),
    'FORCEMONTH': (IMPLEMENTED, 'see FORCEYEAR'),
    'FORCEDAY': (IMPLEMENTED, 'see FORCEYEAR'),
    'FORCESEC': (IMPLEMENTED, 'see FORCEYEAR'),
    'ALBEDO': (UNSUPPORTED, 'CESM-driver forcing declaration'),
    'LAI': (UNSUPPORTED, 'CESM-driver forcing declaration'),
    'FCANOPY': (UNSUPPORTED, 'CESM-driver forcing declaration'),

    # -- classic-driver file layout ------------------------------------------
    'SOIL': (IMPLEMENTED, 'the classic ASCII soil file, read by vic_classic.ClassicDomain with the column order generated from read_soilparam.c'),
    'VEGPARAM': (IMPLEMENTED, 'the classic per-cell vegetation file, read by vic_classic per read_vegparam.c'),
    'VEGLIB': (IMPLEMENTED, 'the classic vegetation library, read by vic_classic per read_veglib.c including the bare-soil tile'),
    'ARC_SOIL': (UNSUPPORTED, 'ARC/INFO soil files (classic driver)'),
    'NLAYER': (UNSUPPORTED, 'the layer count comes from the parameter file\'s '
                            'nlayer dimension, not from the global file'),
    'SNOW_STEP': (UNSUPPORTED, 'renamed SNOW_STEPS_PER_DAY in VIC 5'),
    'TIME_STEP': (UNSUPPORTED, 'renamed MODEL_STEPS_PER_DAY in VIC 5'),
    'BINARY_STATE_FILE': (UNSUPPORTED, 'renamed STATE_FORMAT in VIC 5'),
    'STATENAME_CESM': (UNSUPPORTED, 'no state file is written'),

    # -- keys VIC itself refuses ---------------------------------------------
    # Nothing stays refused. Each of
    # these is a key some VIC driver or release stops on and TorchVIC runs
    # (ACCEPTED_AFTER_VIC_REFUSAL below says which).
    'ATMOS_STEPS_PER_DAY': (IGNORED, 'the classic driver\'s key; validated '
                            'as get_global_param.c:811-856 does, no effect on '
                            'results (it only sets atmos_dt, which is logged)'),
    'OUTPUT_FORCE': (IMPLEMENTED, 'VIC 4.2.d writes its forcing files '
                     '(write_forcing_file.c); so does TorchVIC, in the same '
                     'files'),
    'ARNO_PARAMS': (IMPLEMENTED, 'read as BASEFLOW NIJSSEN2001 (TRUE) or ARNO '
                    '(FALSE), the replacement VIC\'s own error text names'),
    'NIJSSEN2001_BASEFLOW': (IMPLEMENTED, 'read as BASEFLOW NIJSSEN2001 (TRUE) '
                             'or ARNO (FALSE), the replacement VIC\'s own '
                             'error text names'),
}

#: Keys a VIC driver or release stops on that TorchVIC nevertheless runs
#: (nothing stays refused). `inconsistencies`
#: exempts them; torchvic/io/options.py says what each does.
ACCEPTED_AFTER_VIC_REFUSAL = frozenset((
    'ATMOS_STEPS_PER_DAY', 'OUTPUT_FORCE', 'ARNO_PARAMS',
    'NIJSSEN2001_BASEFLOW', 'COMPUTE_TREELINE'))

# The image driver's dispositions, value by value, are torchvic/io/options.py
# (docs/OPTIONS.md), which torchvic.case.load enforces: an option is run as
# VIC runs it, read to no effect, or refused by name. The entries below
# replace the ones above that it made untrue. A status here is still the
# whole key's: `unsupported` where TorchVIC runs some of VIC's values and
# refuses the others by name.
_SEE = ' (image driver: torchvic/io/options.py, docs/OPTIONS.md)'
_WIN = ('the simulation window: torchvic.case.window runs the forcing '
        'records from STARTYEAR..STARTSEC to the end of ENDYEAR/ENDMONTH/'
        'ENDDAY, or NRECS model steps, and refuses a forcing that does not '
        'cover it')
STATUS.update({
    'MODEL_STEPS_PER_DAY': (
        IMPLEMENTED, 'the model step: the forcing\'s own, or 1 (the day\'s '
                     'records averaged, vic_force.c:404-412); other steps '
                     'are refused' + _SEE),
    'SNOW_STEPS_PER_DAY': (
        UNSUPPORTED, 'must be the forcing\'s step; a daily model step whose '
                     'forcing can snow is refused, since VIC sub-steps the '
                     'snow at this rate and TorchVIC does not' + _SEE),
    'STARTYEAR': (IMPLEMENTED, _WIN), 'STARTMONTH': (IMPLEMENTED, _WIN),
    'STARTDAY': (IMPLEMENTED, _WIN), 'STARTSEC': (IMPLEMENTED, _WIN),
    'ENDYEAR': (IMPLEMENTED, _WIN), 'ENDMONTH': (IMPLEMENTED, _WIN),
    'ENDDAY': (IMPLEMENTED, _WIN), 'NRECS': (IMPLEMENTED, _WIN),
    'CALENDAR': (IMPLEMENTED, 'the dates are the forcing file\'s; a '
                              'CALENDAR that disagrees with it is refused'),
    'WIND_H': (IMPLEMENTED, 'the wind measurement height, resolved by '
                            'torchvic.case.resolve_wind_h'),
    'AERO_RESIST_CANSNOW': (UNSUPPORTED, 'AR_406_FULL and AR_406 are run as '
                            'VIC runs them (basin.simulate('
                            'aero_resist_cansnow=...)); AR_410 and AR_406_LS '
                            'are refused' + _SEE),
    'QUICK_FLUX': (IMPLEMENTED, 'TRUE is the two-node flux; FALSE solves '
                                'the finite-difference profile at every '
                                'trial surface temperature' + _SEE),
    'IMPLICIT': (IMPLEMENTED, 'the solver of the finite-difference profile; '
                              'TRUE with QUICK_FLUX is refused, as VIC '
                              'refuses it' + _SEE),
    'EXP_TRANS': (IMPLEMENTED, 'the node grid of the finite-difference '
                               'profile; TRUE with QUICK_FLUX is refused, '
                               'as VIC refuses it' + _SEE),
    'SNOW_BAND': (UNSUPPORTED, 'TRUE runs only where every cell has one band '
                               'with area; more bands are refused (one '
                               'column per cell)' + _SEE),
    'SPATIAL_SNOW': (UNSUPPORTED, 'calc_snow_coverage.c is implemented and '
                                  'runs from the classic soil file; the '
                                  'image driver refuses TRUE, not verified '
                                  'against VIC' + _SEE),
    'BLOWING': (IMPLEMENTED, 'CalcBlowingSnow.c (torchvic.core.blowing); '
                             'sigma_slope, lag_one and fetch are read from '
                             'the classic vegetation file and the image '
                             'parameter file'),
    'LAI_SRC': (IMPLEMENTED, 'FROM_VEGLIB and FROM_VEGPARAM read the '
                             'parameter file\'s LAI, FROM_VEGHIST the '
                             'per-step LAI of FORCING2 (io.veghist)' + _SEE),
    'ALB_SRC': (IMPLEMENTED, 'FROM_VEGLIB and FROM_VEGPARAM read the '
                             'parameter file\'s albedo, FROM_VEGHIST the '
                             'per-step albedo of FORCING2 (io.veghist)'
                             + _SEE),
    'FCAN_SRC': (IMPLEMENTED, 'FROM_DEFAULT, FROM_VEGLIB and FROM_VEGPARAM as '
                              'VIC (a file without fcanopy refused), '
                              'FROM_VEGHIST the per-step fcanopy of FORCING2 '
                              '(io.veghist)' + _SEE),
    'JULY_TAVG_SUPPLIED': (IMPLEMENTED, 'read with COMPUTE_TREELINE, which '
                                        'is the only reader of July_Tavg'
                           + _SEE),
    'NOFLUX': (IMPLEMENTED, 'the bottom boundary of the finite-difference '
                            'profile; not read on the QUICK_FLUX path' + _SEE),
    'NO_FLUX': (IMPLEMENTED, 'alias of NOFLUX' + _SEE),
    'NODES': (IMPLEMENTED, 'the thermal node count off the QUICK_FLUX path; '
                           'with QUICK_FLUX VIC resets it to 3'),
    'QUICK_SOLVE': (IMPLEMENTED, 'bracketing on the reduced column off the '
                                 'QUICK_FLUX path; with QUICK_FLUX it finds '
                                 'the same root' + _SEE),
})


def inventory():
    """The whole extraction: {key: fact} plus what TorchVIC does with each.

    Returns the dict that is written to torchvic/data/ and read back by
    `torchvic.io`, and a list of keys VIC knows that STATUS does not
    classify -- empty in a healthy tree, and the thing the drift guard tests.
    """
    per_driver = {d: read_driver(d) for d in DRIVERS}
    keys = sorted(set().union(*[set(v) for v in per_driver.values()]))
    options, undeclared = {}, []
    for k in keys:
        drivers = [d for d in DRIVERS if k in per_driver[d]]
        rejected = [d for d in drivers if per_driver[d][k]['rejected']]
        entry = {
            'drivers': drivers,
            'rejected_by': rejected,
            # An option is boolean if any driver reads it with str_to_bool.
            'boolean': any(per_driver[d][k]['boolean'] for d in drivers),
            'parsed_elsewhere': all(per_driver[d][k]['noop'] for d in drivers),
        }
        if k in STATUS:
            entry['status'], entry['note'] = STATUS[k]
        else:
            entry['status'], entry['note'] = 'unclassified', ''
            undeclared.append(k)
        options[k] = entry
    inv = {
        'generated_by': 'tools/extract_vic_options.py',
        'vic_version': vic_version(),
        'vic_source': [SOURCE % d for d in DRIVERS],
        'drivers': list(DRIVERS),
        'options': options,
        'sub_options': sub_options(),
    }
    return inv, undeclared


def inconsistencies(inv):
    """Classifications that contradict what was read out of VIC.

    One rule, and it is worth having because it is the mistake this table
    invites: a key VIC itself refuses cannot be `implemented` or `ignored`
    here. VIC would stop on the file. Treating such a key as harmless would
    let TorchVIC accept a global file that VIC rejects, which is the opposite
    of native ingest.
    """
    bad = []
    for k, e in sorted(inv['options'].items()):
        if k in ACCEPTED_AFTER_VIC_REFUSAL:
            continue
        if e['rejected_by'] and e['status'] in (IMPLEMENTED, IGNORED):
            bad.append('%s is %s here but refused by VIC\'s %s driver'
                       % (k, e['status'], '/'.join(e['rejected_by'])))
    return bad


def counts(inv, driver='image'):
    """How many image-driver options fall in each status."""
    out = {}
    for k, e in inv['options'].items():
        if driver in e['drivers']:
            out[e['status']] = out.get(e['status'], 0) + 1
    return out


def write_doc(inv, path):
    """The generated table, as Markdown."""
    opts = inv['options']
    img = [k for k in sorted(opts) if 'image' in opts[k]['drivers']]
    other = [k for k in sorted(opts) if 'image' not in opts[k]['drivers']]
    c = counts(inv)
    L = []
    L.append('# Every VIC 5 global-file option, and its status in TorchVIC')
    L.append('')
    L.append('**Generated by `tools/extract_vic_options.py` from VIC %s\'s '
             'own source. Do not hand-edit.** Re-run the script after '
             'changing VIC, and `tests/test_vic_options.py` fails '
             'if VIC carries an option this table does not classify.'
             % inv['vic_version'])
    L.append('')
    L.append('The keys are read from the `strcasecmp(KEY, optstr)` dispatch '
             'in each driver\'s `get_global_param.c`, so this list is VIC\'s '
             'list rather than a transcription of it. The *status* column is '
             'the one thing here that is a judgement about TorchVIC:')
    L.append('')
    L.append('| status | meaning |')
    L.append('|---|---|')
    L.append('| `implemented` | setting the option changes the run the way '
             'VIC would change it |')
    L.append('| `ignored` | parsed, and its effect is on something TorchVIC '
             'does not do at all (history files, log destinations) |')
    L.append('| `unsupported` | not honoured. Announced in '
             '`GlobalParam.unsupported` so no file turns on a physics we do '
             'not have without saying so |')
    L.append('')
    L.append('Parsing a key is never on its own grounds for `implemented`. '
             'Where VIC offers a choice and TorchVIC implements one of them '
             '-- ARNO but not NIJSSEN2001 baseflow, DENS_BRAS but not '
             'DENS_SNTHRM -- the status is `unsupported` and the note names '
             'which one you get.')
    L.append('')
    L.append('## Image driver: %d options' % len(img))
    L.append('')
    L.append('%d implemented, %d ignored, %d unsupported.'
             % (c.get('implemented', 0), c.get('ignored', 0),
                c.get('unsupported', 0)
                + c.get('unclassified', 0)))
    L.append('')
    L.append('| option | status | value | VIC | note |')
    L.append('|---|---|---|---|---|')
    for k in img:
        e = opts[k]
        vic = ('**refused by VIC**' if 'image' in e['rejected_by']
               else ('elsewhere' if e['parsed_elsewhere'] else 'parsed'))
        L.append('| `%s` | %s | %s | %s | %s |'
                 % (k, e['status'], 'bool' if e['boolean'] else 'value',
                    vic, e['note']))
    L.append('')
    L.append('## Other drivers: %d options the image driver does not take'
             % len(other))
    L.append('')
    L.append('A user arriving from the classic or CESM driver will hand us '
             'one of these. TorchVIC reads the image driver\'s global file, so '
             'they are listed to be named rather than to be implemented.')
    L.append('')
    L.append('| option | status | driver(s) | note |')
    L.append('|---|---|---|---|')
    for k in other:
        e = opts[k]
        L.append('| `%s` | %s | %s | %s |'
                 % (k, e['status'], ', '.join(e['drivers']), e['note']))
    L.append('')
    sub = inv.get('sub_options', {})
    if sub:
        L.append('## Second tokens')
        L.append('')
        L.append('`DOMAIN_TYPE` and `FORCE_TYPE` lines carry a second keyword '
                 'that VIC parses in `shared_image`, not in '
                 '`get_global_param.c`. They are extracted from where they '
                 'are parsed.')
        L.append('')
        for key in sorted(sub):
            L.append('- **%s**: %s' % (key, ', '.join('`%s`' % s
                                                      for s in sub[key])))
        L.append('')
    open(path, 'w').write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true',
                    help='report unclassified options and write nothing')
    ap.add_argument('--json', default=JSON_OUT)
    ap.add_argument('--doc', default=None,
                    help='also write the table as Markdown to this path')
    args = ap.parse_args()

    inv, undeclared = inventory()
    c = counts(inv)
    print('VIC %s: %d global-file options across %s'
          % (inv['vic_version'], len(inv['options']), ', '.join(DRIVERS)))
    img = [k for k, e in inv['options'].items() if 'image' in e['drivers']]
    print('  image driver: %d options -- %d implemented, %d ignored, '
          '%d unsupported'
          % (len(img), c.get('implemented', 0), c.get('ignored', 0),
             c.get('unsupported', 0)))
    rej = sorted(k for k, e in inv['options'].items() if e['rejected_by'])
    print('  refused by VIC itself: %s' % ', '.join(rej))
    for line in inconsistencies(inv):
        print('  INCONSISTENT: %s' % line)
    if undeclared:
        print('\n%d option(s) VIC knows and this script does not classify:'
              % len(undeclared))
        for k in undeclared:
            print('    %s  (%s)'
                  % (k, ', '.join(inv['options'][k]['drivers'])))
        print('Add each to STATUS in %s with an honest status.'
              % os.path.relpath(__file__, REPO))
    if args.check:
        return 1 if undeclared else 0

    for p in (args.json, args.doc):
        if not p:
            continue
        d = os.path.dirname(p)
        if d and not os.path.isdir(d):
            os.makedirs(d)
    with open(args.json, 'w') as f:
        json.dump(inv, f, indent=2, sort_keys=True)
        f.write('\n')
    print('\nwrote %s' % os.path.relpath(args.json, REPO))
    if args.doc:
        write_doc(inv, args.doc)
        print('wrote %s' % os.path.relpath(args.doc, REPO))
    return 1 if undeclared else 0


if __name__ == '__main__':
    sys.exit(main())
