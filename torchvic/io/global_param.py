"""The VIC 5 global parameter file, and the inventory of VIC's options.

A VIC user with a working image-driver set-up points this model at the
global file and it runs, with no conversion step and no second copy of the
data; domain.py, params.py and forcing.py read the files it names.

The schema is **declared, not assumed**. `get_global_domain.c` takes every
domain variable name from the global parameter file, so this reader parses
that file and honours what it says:

    DOMAIN_TYPE  LAT  lat    DOMAIN_TYPE  MASK  mask
    DOMAIN_TYPE  LON  lon    DOMAIN_TYPE  AREA  area
    DOMAIN_TYPE  YDIM lat    DOMAIN_TYPE  FRAC  frac
    DOMAIN_TYPE  XDIM lon

Unknown keys are recorded rather than rejected. A real global file carries
options this model does not implement -- lakes, carbon, blowing snow -- and
refusing the file because of them would turn a readable set-up into an
unreadable one for no benefit. `GlobalParam.unsupported` lists what was seen
and ignored, so nothing is silently assumed to be off.

**Which keys exist is VIC's answer, not ours.** The three sets below used to
be written out by hand, and a hand-written set of VIC's options drifts the
moment VIC changes: the key stops being unrecognised, starts being *known*,
and a file that switched on a physics we do not have reads as though it had
switched on nothing. So they are loaded from the package data file
`data/vic5_global_options.json`, which `tools/extract_vic_options.py`
generates by parsing the
`strcasecmp(KEY, optstr)` dispatch in VIC's own `get_global_param.c`. The
statuses in that inventory are ours -- what TorchVIC does with each key -- but
the list of keys, and whether each takes TRUE/FALSE, come out of the C.

The hand-written sets survive as a fallback, because the package must still
read a global file when installed without that file. They are the smaller,
older answer; the inventory is the complete one, and
`tests/test_vic_options.py` fails if VIC grows an option neither
knows about.
"""
import json
import os
import re
import warnings


def _announcer(log):
    """Where a reader's announcements go: the caller's `log`, or a warning.

    Never nowhere. Every branch of this module that decides something on the
    user's behalf -- a unit overridden, a rate rescaled, a field the file
    supplied and kept -- reports through here, so that a caller who passes no
    log still sees it on stderr rather than not at all.
    """
    if log is not None:
        return log

    def _warn(message):
        warnings.warn(message, stacklevel=3)
    return _warn


DOMAIN_KEYS = {'LAT': 'lat_var', 'LON': 'lon_var', 'MASK': 'mask_var',
               'AREA': 'area_var', 'FRAC': 'frac_var',
               'YDIM': 'y_dim', 'XDIM': 'x_dim'}

#: What tools/extract_vic_options.py read out of VIC, shipped as package data.
OPTIONS_INVENTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
    'vic5_global_options.json')

# The fallback. This is what the reader knew before the inventory existed, and
# it is deliberately left as it was: an installation without the data file reads
# global files exactly as it used to rather than failing or, worse, treating
# every key as unrecognised.
_FALLBACK_KNOWN = {
    'FULL_ENERGY', 'FROZEN_SOIL', 'NLAYER', 'NODES', 'SNOW_BAND',
    'MODEL_STEPS_PER_DAY', 'SNOW_STEPS_PER_DAY', 'RUNOFF_STEPS_PER_DAY',
    'STARTYEAR', 'STARTMONTH', 'STARTDAY', 'STARTSEC', 'ENDYEAR', 'ENDMONTH',
    'ENDDAY', 'NRECS', 'CALENDAR', 'DOMAIN', 'DOMAIN_TYPE', 'PARAMETERS',
    'FORCING1', 'FORCING2', 'FORCE_TYPE', 'WIND_H', 'RESULT_DIR',
    'LOG_DIR', 'LOG_LVL', 'OUTFILE', 'OUTVAR', 'AOUT_STEP', 'OUT_FORMAT',
}
_FALLBACK_NOT_IMPLEMENTED = {
    'LAKES', 'LAKE_PROFILE', 'CARBON', 'RC_MODE', 'BLOWING',
    'BLOWING_VAR_THRESHOLD', 'BLOWING_CALC_PROB', 'BLOWING_SIMPLE',
    'BLOWING_FETCH', 'BLOWING_SPATIAL_WIND',
    # SPATIAL_SNOW is implemented (torchvic.core.snow.snow_coverage):
    # the classic soil column and the image parameter variable both
    # carry max_snow_distrib_slope.
    'SPATIAL_FROST', 'IMPLICIT', 'EXP_TRANS', 'QUICK_FLUX',
    'QUICK_SOLVE', 'NOFLUX', 'GRND_FLUX_TYPE', 'CLOSE_ENERGY',
}


def load_inventory(path=None):
    """The extracted VIC option inventory, or None if it was not shipped."""
    path = path or OPTIONS_INVENTORY
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            inv = json.load(f)
    except (ValueError, IOError):                            # pragma: no cover
        # A truncated or half-written inventory must not take the reader down
        # with it; the fallback sets still parse a global file.
        warnings.warn('%s could not be read; falling back to the built-in '
                      'option list' % path)
        return None
    return inv if isinstance(inv.get('options'), dict) else None


def _option_sets(inv):
    """The two sets `_parse` dispatches on, derived from the inventory.

    `KNOWN_OPTIONS` is what TorchVIC either honours or can ignore without
    consequence; `NOT_IMPLEMENTED` is the subset of the unsupported options
    that VIC reads with `str_to_bool`. The split is not cosmetic -- it decides
    *when* an option is announced. `FROZEN_SOIL FALSE` asks for nothing and
    saying so on every read would bury the lines that matter, while
    `FROZEN_SOIL TRUE` asks for a physics that is not here and has to be said.
    A non-boolean unsupported key such as `LAI_SRC` has no off position, so it
    is announced whenever it appears.

    Anything in neither set -- an option VIC gained after the inventory was
    generated, or a typo -- still lands in `unsupported`. That is the property
    worth keeping: the reader has no silent third category.
    """
    if inv is None:
        return set(_FALLBACK_KNOWN), set(_FALLBACK_NOT_IMPLEMENTED), {}
    known, boolean_unsupported, notes = set(), set(), {}
    for key, e in inv['options'].items():
        notes[key] = e.get('note', '')
        if e.get('status') in ('implemented', 'ignored'):
            known.add(key)
        elif e.get('boolean'):
            boolean_unsupported.add(key)
    return known, boolean_unsupported, notes


_INVENTORY = load_inventory()
KNOWN_OPTIONS, NOT_IMPLEMENTED, OPTION_NOTES = _option_sets(_INVENTORY)
#: Every key in the inventory, whatever its status: used to tell "VIC has this
#: option and we do not implement it" apart from "VIC has never heard of this".
VIC_OPTIONS = set(_INVENTORY['options']) if _INVENTORY else \
    set(_FALLBACK_KNOWN) | set(_FALLBACK_NOT_IMPLEMENTED)


#: The 4.2.d baseflow switches and the BASEFLOW value each value stands for,
#: as VIC's own error text gives it: ARNO_PARAMS TRUE -> BASEFLOW NIJSSEN2001,
#: FALSE -> BASEFLOW ARNO; NIJSSEN2001_BASEFLOW the same.
BASEFLOW_SPELLINGS = {'ARNO_PARAMS': {True: 'NIJSSEN2001', False: 'ARNO'},
                      'NIJSSEN2001_BASEFLOW': {True: 'NIJSSEN2001',
                                               False: 'ARNO'}}

#: VIC constants (CONSTANTS file keys) TorchVIC applies -> the basin.simulate
#: parameter that carries each one.
CONSTANTS_HONOURED = {'VEG_LAI_WATER_FACTOR': 'lai_water_factor'}


class GlobalParam(object):
    """A parsed VIC global parameter file."""

    def __init__(self, path, sample_data=None):
        self.path = path
        # VIC's own sample global files reference ${VIC_SAMPLE_DATA}; passing
        # it here avoids requiring the caller to set an environment variable
        # just to read a file.
        self.sample_data = sample_data
        self.root = os.path.dirname(os.path.abspath(path))
        self.raw = {}                 # key -> [value, ...] in file order
        # Every line as (KEY, [tokens]) in file order. `raw` groups by key
        # and so loses which OUTVAR follows which OUTFILE; VIC's
        # parse_output_info reads the output section in order, and so does
        # torchvic.io.history.parse_streams, from this.
        self.lines = []
        self.domain_vars = {'lat_var': 'lat', 'lon_var': 'lon',
                            'mask_var': 'mask', 'area_var': 'area',
                            'frac_var': 'frac', 'y_dim': 'lat',
                            'x_dim': 'lon'}
        self.force_type = {}          # VIC name -> NetCDF variable name
        # FORCE_TYPE lines that follow FORCING2 describe the second file
        # (get_global_param.c:334-352, file_num = 1): VIC only counts them
        # (count_force_vars, forcing_utils.c:108-148) to know the file is
        # in use, and reads nothing from it but LAI, fcanopy and albedo.
        self.force2_type = {}
        self._file_num = 0
        self.unsupported = []
        # A key VIC itself does not recognise is a different problem from one
        # TorchVIC has not implemented -- the first is usually a misspelling
        # and VIC would warn about it too -- so the two are kept apart even
        # though both are reported. `unknown` is a subset of `unsupported`.
        self.unknown = []
        #: 'OLD value -> BASEFLOW NEW' for each 4.2.d spelling read as
        #: BASEFLOW (see BASEFLOW_SPELLINGS); torchvic.io.options announces.
        self.translated = []
        self._parse()

    def _parse(self):
        for line in open(self.path):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = re.split(r'\s+', line)
            key, rest = parts[0].upper(), parts[1:]
            if key == 'DOMAIN_TYPE' and len(rest) >= 2:
                field = DOMAIN_KEYS.get(rest[0].upper())
                if field:
                    self.domain_vars[field] = rest[1]
                continue
            if key == 'FORCING1':
                self._file_num = 0
            elif key == 'FORCING2':
                self._file_num = 1
            if key == 'FORCE_TYPE' and len(rest) >= 2:
                if self._file_num == 1:
                    self.force2_type[rest[0].upper()] = rest[1]
                else:
                    self.force_type[rest[0].upper()] = rest[1]
                continue
            if key in BASEFLOW_SPELLINGS:
                # VIC 4.2.d's own spellings. Every VIC release stops on them
                # and names the replacement (4.2.d get_global_param.c:548-565,
                # 5.1.0 image :369-392, classic :576-603); TorchVIC
                # reads them as that replacement,
                # so the line becomes "BASEFLOW NIJSSEN2001|ARNO" in place.
                on = bool(rest) and str(rest[0]).upper() == 'TRUE'
                new = BASEFLOW_SPELLINGS[key][on]
                self.translated.append('%s %s -> BASEFLOW %s'
                                       % (key, rest[0] if rest else '', new))
                key, rest = 'BASEFLOW', [new]
            self.raw.setdefault(key, []).append(rest)
            self.lines.append((key, rest))
            if key in NOT_IMPLEMENTED and rest and \
                    str(rest[0]).upper() in ('TRUE', '1'):
                self.unsupported.append('%s %s' % (key, rest[0]))
            elif key not in KNOWN_OPTIONS and key not in NOT_IMPLEMENTED:
                self.unsupported.append(key)
                if key not in VIC_OPTIONS:
                    self.unknown.append(key)

    # -- accessors --------------------------------------------------------
    def constants(self):
        """{basin.simulate parameter name: value} from the CONSTANTS file.

        VIC reads `KEY value` lines (get_parameters.c). TorchVIC applies
        the constants in CONSTANTS_HONOURED; any other key in the file is
        refused by name, because a constant VIC would apply and this model
        would not is a silent difference between the two runs."""
        path = self.get('CONSTANTS')
        if path is None:
            return {}
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(self.root, path)
        out, other = {}, []
        for line in open(path):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = re.split(r'\s+', line)
            key = parts[0].upper()
            if key in CONSTANTS_HONOURED and len(parts) > 1:
                out[CONSTANTS_HONOURED[key]] = float(parts[1])
            else:
                other.append(key)
        if other:
            raise ValueError(
                'CONSTANTS file %s sets %s, which TorchVIC does not apply '
                '(it applies %s); remove them or run VIC itself'
                % (path, ', '.join(other), ', '.join(sorted(
                    CONSTANTS_HONOURED))))
        return out

    def get(self, key, default=None, index=0):
        v = self.raw.get(key.upper())
        if not v or not v[0]:
            return default
        return v[0][index]

    def get_int(self, key, default=None):
        """An integer option, or `default` if absent or not a number.

        Some keys are numeric in the classic driver and boolean in the image
        driver -- SNOW_BAND is a band count in one and TRUE/FALSE in the other,
        with the count coming from the parameter file. Returning the default
        rather than raising keeps one reader working for both.
        """
        v = self.get(key)
        if v is None:
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def get_bool(self, key, default=False):
        v = self.get(key)
        return default if v is None else str(v).upper() in ('TRUE', '1')

    def path_of(self, key, sample_data=None):
        """A filename from the global file, expanded and resolved.

        Real global files use environment variables in paths -- VIC's own
        sample set uses ${VIC_SAMPLE_DATA} -- so expansion happens before the
        relative/absolute test, or an expanded absolute path gets joined onto
        the global file's directory and becomes nonsense.
        """
        v = self.get(key)
        if v is None:
            return None
        if sample_data and '${VIC_SAMPLE_DATA}' in v:
            # A sample-data root given by the caller is theirs, not the global
            # file's, so a relative one resolves against the working directory.
            # Joining it onto the global file's directory instead produced
            # .../parameters/data/.../parameters/domain.nc -- a path that
            # exists nowhere.
            v = v.replace('${VIC_SAMPLE_DATA}', os.path.abspath(sample_data))
        v = os.path.expandvars(os.path.expanduser(v))
        return v if os.path.isabs(v) else os.path.join(self.root, v)

    @property
    def domain_file(self):
        return self.path_of('DOMAIN', self.sample_data)

    @property
    def params_file(self):
        return self.path_of('PARAMETERS', self.sample_data)

    @property
    def forcing_prefix(self):
        return self.path_of('FORCING1', self.sample_data)

    @property
    def forcing2_prefix(self):
        """The FORCING2 prefix, or None: absent, or FALSE (VIC turns that
        into MISSING, get_global_param.c:336-338)."""
        v = self.get('FORCING2')
        if v is None or str(v).upper() == 'FALSE':
            return None
        return self.path_of('FORCING2', self.sample_data)

    @property
    def start_date(self):
        import datetime as dt
        y, m, d = (self.get_int('STARTYEAR'), self.get_int('STARTMONTH'),
                   self.get_int('STARTDAY'))
        return dt.date(y, m, d) if None not in (y, m, d) else None

    @property
    def end_date(self):
        import datetime as dt
        y, m, d = (self.get_int('ENDYEAR'), self.get_int('ENDMONTH'),
                   self.get_int('ENDDAY'))
        return dt.date(y, m, d) if None not in (y, m, d) else None

    def summary(self):
        lines = ['global parameter file: %s' % self.path,
                 '  domain      %s' % self.domain_file,
                 '  parameters  %s' % self.params_file,
                 '  forcing     %s' % self.forcing_prefix,
                 '  period      %s .. %s' % (self.start_date, self.end_date),
                 '  steps/day   %s' % self.get_int('MODEL_STEPS_PER_DAY'),
                 '  layers %s, nodes %s, snow bands %s'
                 % (self.get_int('NLAYER'), self.get_int('NODES'),
                    self.get_int('SNOW_BAND',
                                 'from params file'
                                 if self.get_bool('SNOW_BAND') else 1)),
                 '  domain vars %s' % self.domain_vars]
        if self.force_type:
            lines.append('  forcing vars %s' % self.force_type)
        if self.unsupported:
            seen = sorted(set(self.unsupported))
            lines.append('  present in the file and NOT implemented here: %s'
                         % ', '.join(seen))
            # A bare list of keys says what was skipped but not what skipping
            # it costs, and "AERO_RESIST_CANSNOW" means nothing to a reader
            # deciding whether the run is usable. The inventory carries a
            # reason for every option, so print it.
            for item in seen:
                note = OPTION_NOTES.get(item.split()[0])
                if note:
                    lines.append('    %-24s %s' % (item.split()[0], note))
        if self.unknown:
            lines.append('  not a VIC 5 option at all (check the spelling, or '
                         're-run tools/extract_vic_options.py if VIC has '
                         'moved): %s' % ', '.join(sorted(set(self.unknown))))
        return '\n'.join(lines)

    def __repr__(self):
        return '<GlobalParam %s>' % os.path.basename(self.path)
