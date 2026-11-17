"""The image driver's forcing files, and their aggregation to days.
"""
import os
import re

from .global_param import _announcer


class ImageForcing(object):
    """VIC's forcing files, read for the active cells only.

    VIC writes one file per year named `<prefix><year>.nc`, so a decade is ten
    files and a continental run is ten large ones. Three things decide whether
    this is fast, and all three are about what is *not* read:

    * **Only the active cells.** A basin is a few hundred columns out of a
      continental grid, and the selection happens on read rather than after.
    * **Chunked along time.** The model steps through time, so a chunk
      spanning every cell for a block of days is one read per block instead of
      one per cell.
    * **Transferred once.** Forcing does not change during training, so it
      belongs on the device for the whole run rather than being copied each
      epoch.

    Variable names come from the global file's FORCE_TYPE declarations, so a
    file naming precipitation `prcp` and one naming it `pr` both load without
    being rewritten first.
    """

    #: Canonical units this model works in, by field.
    UNITS = {'tair': 'C', 'tmax': 'C', 'tmin': 'C', 'press': 'Pa',
             'vp': 'Pa', 'shortwave': 'W/m2', 'longwave': 'W/m2',
             'wind': 'm/s', 'prcp': 'mm', 'qair': 'kg/kg', 'rh': '%'}
    #: Multipliers onto the canonical unit, keyed by what the file declares.
    #: Driven by the file's own `units` attribute rather than assumed: VIC's
    #: image forcing states pressure and vapour pressure in kPa while this
    #: model works in Pa, and taking the numbers as given would be a silent
    #: factor of a thousand in every energy balance.
    CONVERT = {
        ('press', 'kpa'): 1000.0, ('press', 'pa'): 1.0,
        ('vp', 'kpa'): 1000.0, ('vp', 'pa'): 1.0,
        ('tair', 'c'): 1.0, ('tmax', 'c'): 1.0, ('tmin', 'c'): 1.0,
        ('shortwave', 'w/m2'): 1.0, ('longwave', 'w/m2'): 1.0,
        ('wind', 'm/s'): 1.0,
        ('prcp', 'mm/step'): 1.0, ('prcp', 'mm'): 1.0,
        ('qair', 'kg/kg'): 1.0, ('rh', '%'): 1.0, ('rh', 'percent'): 1.0,
    }
    # Kelvin is deliberately absent. CONVERT is multiplicative and K -> C is
    # an offset, so a 'k': 1.0 entry would pass 273 K through as 273 C. A file
    # in Kelvin must raise until an offset path exists.
    #
    # `mm/day` is deliberately absent too, and its absence is the point. It
    # used to sit here with a multiplier of 1.0, which made a *rate* read as
    # an accumulation: `aggregate_to_daily` sums precipitation, so eight
    # three-hourly records each labelled 0.5 mm/day became 4.0 mm on the day,
    # eight times the rain the file describes, with nothing said. The CF
    # spelling `mm d-1` was never in the table and raised instead, so the same
    # file was readable or not depending on how its author wrote the unit.
    # Rates now go through RATE_SECONDS below, which multiplies by the length
    # of the record rather than by one.

    #: Denominators of the rate units precipitation is written in, in seconds.
    #: A rate is turned into the accumulation over one forcing record --
    #: value * (record length / denominator) -- so that what is summed to a
    #: day is a depth, not a rate. 1 kg/m2 of water is 1 mm, which is why the
    #: CF flux spelling belongs on the same path.
    RATE_SECONDS = {'s': 1.0, 'sec': 1.0, 'secs': 1.0, 'second': 1.0,
                    'h': 3600.0, 'hr': 3600.0, 'hour': 3600.0,
                    'd': 86400.0, 'day': 86400.0}
    #: Numerators that are a water depth: both are millimetres of water.
    RATE_DEPTH = ('mm', 'kg/m2')

    #: Unit strings that are *wrong in the file* and cannot be taken at face
    #: value. UW-Hydro's own Walnut Gulch sample forcing labels wind
    #: ``mm s-1`` while the numbers (0.3-8.7, mean 2.8) are plainly m/s -- as
    #: mm/s the basin would be windless all year. Reading the label literally
    #: would be a silent factor of a thousand, and refusing the file would
    #: make the official sample data unreadable, so the override is declared
    #: here by name and announced on every read.
    SUSPECT_UNITS = {
        ('wind', 'mm/s'): ('m/s', 'wind of a few mm/s is not a physical wind '
                                  'speed; the values are m/s'),
    }

    @staticmethod
    def normalise_units(units):
        """Fold the CF spelling of a unit onto the spelling CONVERT uses.

        NetCDF files write exponents as ``w m-2``, ``m s-1``,
        ``mm timestep-1``; this model's table is keyed on ``w/m2``, ``m/s``,
        ``mm/step``. They are the same unit, and a reader that rejected one
        spelling of it would reject the official VIC sample data. Only the
        notation is normalised -- nothing is rescaled here, and an unknown
        unit still falls through to the error in `read`.
        """
        u = str(units).strip().lower()
        if not u:
            return u
        u = u.replace('timestep', 'step').replace('time step', 'step')
        u = re.sub(r'\s*\*\s*', ' ', u)
        # "w m-2" -> "w/m2", "m s-1" -> "m/s", "kg kg-1" -> "kg/kg"
        parts = u.split()
        head, denom = parts[0], []
        for tok in parts[1:]:
            m = re.fullmatch(r'([a-z%]+)-(\d+)', tok)
            if m:
                denom.append(m.group(1) + (m.group(2) if m.group(2) != '1'
                                           else ''))
            else:
                denom.append(tok)
        u = '/'.join([head] + denom) if denom else head
        u = u.replace(' ', '')

        # Celsius, in the spellings CF and udunits actually use. `degC` is
        # udunits' canonical form and what xarray writes by default, so a
        # perfectly ordinary CF forcing file was refused for saying
        # `degC` where this table says `c`. Same argument as the exponent
        # folding above: one unit, several notations.
        #
        # Kelvin is NOT folded, and must not be. CONVERT is multiplicative
        # and K -> C is an offset, so any Kelvin spelling has to keep
        # falling through to the error in `read` rather than be quietly
        # accepted as Celsius -- see the note on CONVERT.
        if u in ('degc', 'degreec', 'degreesc', 'degree_c', 'degrees_c',
                 'celsius', 'degree_celsius', 'degrees_celsius', 'deg_c'):
            return 'c'
        return u

    #: VIC FORCE_TYPE name -> the name this model uses internally.
    CANONICAL = {'AIR_TEMP': 'tair', 'PREC': 'prcp', 'PRESSURE': 'press',
                 'SWDOWN': 'shortwave', 'LWDOWN': 'longwave', 'VP': 'vp',
                 'WIND': 'wind', 'QAIR': 'qair', 'REL_HUMID': 'rh',
                 'CATM': 'Catm', 'PAR': 'par', 'FDIR': 'fdir',
                 'TMAX': 'tmax', 'TMIN': 'tmin',
                 # VIC 4.2.d's names (get_force_type.c), read for
                 # OUTPUT_FORCE and for 4.2.d forcing files
                 'SHORTWAVE': 'shortwave', 'LONGWAVE': 'longwave',
                 'DENSITY': 'density'}

    #: Every FORCE_TYPE name VIC 5 defines (set_force_type.c), plus the two
    #: this project adds for a daily forcing. A name outside this set is
    #: refused rather than lowercased into a field nothing consumes.
    #:
    #: The fallback used to be `k.lower()`, which turned a wrong name into
    #: a plausible one in silence: a global file saying SHORTWAVE instead
    #: of VIC's SWDOWN produced a field called `shortwave` that the MTCLIM
    #: chain then ignored, and the run estimated the radiation it had been
    #: given. That is worth +33% on evaporation
    #: and said nothing. VIC itself calls
    #: log_err on an unrecognised type (set_force_type.c:120-127).
    VIC_FORCE_TYPES = frozenset({
        'AIR_TEMP', 'ALBEDO', 'CATM', 'CHANNEL_IN', 'FCANOPY', 'FDIR',
        'LAI_IN', 'LWDOWN', 'PAR', 'PREC', 'PRESSURE', 'QAIR', 'REL_HUMID',
        'SWDOWN', 'VP', 'WIND', 'SKIP',
        # not VIC 5's -- it dropped MTCLIM -- but this project reads a
        # daily forcing, and TMAX/TMIN are how VIC 4.2.d named these.
        'TMAX', 'TMIN', 'SHORTWAVE', 'LONGWAVE', 'DENSITY'})

    #: FORCE_TYPE for the project's own daily ERA5 store
    #: (era5_isc_daily_1940_2026.zarr), used when a zarr store carries those
    #: variable names and the global file declares no FORCE_TYPE. The names
    #: and units are the store's (sp and vp in kPa, tasmax/tasmin in degC);
    #: the conversions are the NetCDF path's.
    ERA5_DAILY_FORCE_TYPE = {
        'TMAX': 'tasmax', 'TMIN': 'tasmin', 'PREC': 'prcp', 'VP': 'vp',
        'PRESSURE': 'sp', 'WIND': 'wind', 'SWDOWN': 'swdown',
        'LWDOWN': 'lwdown'}

    @staticmethod
    def is_zarr(path):
        """True for a Zarr store path (a directory ending .zarr)."""
        return str(path).rstrip('/').endswith('.zarr')

    @staticmethod
    def open_store(path, **kw):
        """One forcing file or store as an xarray Dataset, lazily.

        NetCDF through xr.open_dataset; Zarr through xr.open_zarr with no
        dask chunks, so indexing reads only the blocks it touches (what
        read_steps and the prefetch thread rely on) and decoding -- fill
        value to NaN, scale/offset, datetime -- is the NetCDF path's.
        """
        import xarray as xr
        if ImageForcing.is_zarr(path):
            try:
                import zarr  # noqa: F401
            except ImportError:
                raise ImportError(
                    'reading the Zarr forcing %s needs the zarr package '
                    '(pip install "torchvic[zarr]")' % path)
            return xr.open_zarr(path, chunks=kw.pop('chunks', None))
        from .ncshare import open_shared
        return open_shared(path, **kw)

    def __init__(self, prefix, domain, force_type=None, years=None,
                 chunk_days=None):
        import glob as _g
        self.prefix = prefix
        self.domain = domain
        self.force_type = force_type or {}
        self.years = [int(y) for y in years] if years else None
        # FORCING1 convention: a path ending .zarr is ONE store holding the
        # whole record (years= then selects along time); otherwise the
        # prefix is globbed for VIC's per-year NetCDF (prefix*.nc), and
        # failing that for per-year stores (prefix*.zarr).
        if self.is_zarr(prefix):
            if not os.path.isdir(prefix):
                raise IOError('no forcing store %s' % prefix)
            files, self._pattern = [prefix.rstrip('/')], prefix
        else:
            self._pattern = prefix + '*.nc'
            files = sorted(_g.glob(self._pattern))
            if not files:
                zp = prefix + '*.zarr'
                files = sorted(_g.glob(zp))
                if files:
                    self._pattern = zp
        if not files:
            raise IOError('no forcing files matching %s' % self._pattern)
        self.single_store = len(files) == 1 and self.is_zarr(files[0])
        if years and not self.single_store:
            files = [f for f in files
                     if any(str(y) in os.path.basename(f) for y in years)]
        self.files = files
        self.chunk_days = chunk_days
        if not self.force_type and self.is_zarr(files[0]):
            ds = self.open_store(files[0])
            if set(self.ERA5_DAILY_FORCE_TYPE.values()) <= set(ds.data_vars):
                self.force_type = dict(self.ERA5_DAILY_FORCE_TYPE)
            ds.close()

    def _open(self, f):
        """`f` opened, restricted to `years` when it is a single store, and
        checked against the domain's coordinates when it is Zarr."""
        import numpy as np
        kw = {'chunks': {'time': self.chunk_days * 24}} if \
            self.chunk_days else {}
        ds = self.open_store(f, **kw)
        if self.single_store and self.years:
            yr = np.asarray(ds['time'].dt.year.values)
            ds = ds.isel(time=np.nonzero(np.isin(yr, self.years))[0])
            if ds.sizes['time'] == 0:
                raise ValueError('%s has no record in years %s'
                                 % (f, self.years))
        if self.is_zarr(f):
            self._check_grid(ds, os.path.basename(f))
        return ds

    def _check_grid(self, ds, base):
        """A store's own lat/lon must be the domain's at the active cells.

        NetCDF forcing is trusted to share the domain's grid (it is VIC's own
        contract); a Zarr store may be a different window of a larger grid
        (the ERA5 store is wider than some basins), and indexing it with the
        domain's rows/cols would then read the wrong cells without a word.
        """
        import numpy as np
        dlat, dlon = getattr(self.domain, 'lat', None), \
            getattr(self.domain, 'lon', None)
        if dlat is None or 'lat' not in ds.coords or 'lon' not in ds.coords:
            return
        la, lo = np.asarray(ds['lat'].values), np.asarray(ds['lon'].values)
        if la.ndim != 1 or lo.ndim != 1:
            return
        ny, nx = getattr(self.domain, 'ny', None), getattr(self.domain, 'nx',
                                                            None)
        if (ny, nx) != (None, None) and (la.size, lo.size) != (ny, nx):
            raise ValueError(
                '%s is a %dx%d grid and the domain is %dx%d; the store must '
                'share the domain grid' % (base, la.size, lo.size, ny, nx))
        r, c = self.domain.rows, self.domain.cols
        if not (np.allclose(la[r], dlat, atol=1e-4)
                and np.allclose(lo[c], dlon, atol=1e-4)):
            raise ValueError(
                '%s: lat/lon at the active cells differ from the domain '
                "file's; the store is on another grid" % base)

    @staticmethod
    def step_seconds(times):
        """The length of one forcing record, in seconds, or None.

        A rate cannot be turned into a depth without it, and an irregular time
        axis has no single answer, so "None" is a real result and the caller
        refuses rather than picking the first interval.
        """
        import numpy as np
        t = np.asarray(times)
        if t.size < 2:
            return None
        try:
            d = np.diff(t.astype('datetime64[s]')).astype('int64')
        except (TypeError, ValueError):
            # cftime objects under a non-standard calendar: they subtract to
            # datetime.timedelta rather than to numpy's.
            d = np.array([(b - a).total_seconds() for a, b in zip(t[:-1],
                                                                  t[1:])])
        return float(d[0]) if np.all(d == d[0]) else None

    def _scale(self, canon, declared, units, step_s, where, say):
        """The multiplier taking `units` onto this model's canonical unit.

        Every branch here exists because of a way a forcing file has been seen
        to be read wrongly without complaint:

        * **No `units` attribute at all.** This used to skip the whole check
          -- `if units and ...` -- so a pressure file with the attribute
          deleted read 85.0 and went into the energy balance as 85 Pa instead
          of 85 kPa. That is precisely the thousand-fold the check was written
          to catch, arriving through the check's own front door. An absent
          unit therefore raises. Announcing and continuing was the other
          candidate and was rejected: the announcement scrolls past in a long
          log and the wrong number still reaches the physics, whereas a file
          whose units are known is one attribute away from being readable.
        * **A field with no canonical unit** -- CHANNEL_IN, or any FORCE_TYPE
          this model does not convert. It cannot be checked, so it is said
          out loud and passed through unconverted; refusing it would make a
          global file unreadable over a variable nothing here consumes.
        * **A rate where an accumulation is expected.** See RATE_SECONDS.
        """
        if canon not in self.UNITS:
            say('  %s: no canonical unit for %r, so %r is passed through '
                'unconverted and unchecked' % (where, canon, declared or
                                               'no units attribute'))
            return 1.0
        if (canon, units) in self.SUSPECT_UNITS:
            units, why = self.SUSPECT_UNITS[(canon, units)]
            say('  %s declares %s in %r; reading it as %s -- %s'
                % (where, canon, declared, units, why))
        if not units:
            raise ValueError(
                '%s gives %s no units attribute and this model works in %s. '
                'A missing unit is not a matching unit: read as it stands, a '
                'pressure in kPa enters the energy balance as Pa, a thousand '
                'times too small, and nothing in the output says so. Add the '
                'units attribute to the file, or add the conversion to '
                'ImageForcing.CONVERT' % (where, canon, self.UNITS[canon]))
        head, _, denom = units.rpartition('/')
        if canon == 'prcp' and head in self.RATE_DEPTH and \
                denom in self.RATE_SECONDS:
            # A rate: what the record holds is a depth per unit time, and what
            # aggregate_to_daily sums must be the depth over the record.
            if step_s is None:
                raise ValueError(
                    '%s states %s as the rate %r, and turning a rate into the '
                    'depth over one record needs the length of a record, '
                    'which this file does not give (one time step, or an '
                    'irregular time axis). State it as an accumulation (%s) '
                    'instead' % (where, canon, declared, 'mm timestep-1'))
            per_step = step_s / self.RATE_SECONDS[denom]
            say('  %s states %s as the rate %r; multiplying by the %.4g s '
                'record to get %s per step (x%.6g) -- summing a rate to a '
                'day would have been %.6gx the precipitation the file '
                'describes' % (where, canon, declared, step_s,
                               self.UNITS[canon], per_step,
                               1.0 / per_step if per_step else float('nan')))
            return per_step
        key = (canon, units)
        if key not in self.CONVERT:
            raise ValueError(
                '%s declares units %r for %s (read as %r) and this '
                'model works in %s; add the conversion to '
                'ImageForcing.CONVERT rather than letting the '
                'numbers through unchanged'
                % (where, declared, canon, units, self.UNITS[canon]))
        return self.CONVERT[key]

    def _wanted(self):
        """{NetCDF variable: canonical field} from FORCE_TYPE, refusing a
        type VIC does not define."""
        # VIC declares which NetCDF variable holds which forcing. Invert it so
        # the file can be read by the name it actually uses.
        unknown = sorted(set(self.force_type) - self.VIC_FORCE_TYPES)
        if unknown:
            raise ValueError(
                'FORCE_TYPE %s is not a VIC forcing type; VIC calls log_err '
                'on an unrecognised one (set_force_type.c:120-127) and this '
                'used to lowercase it into a field name nothing reads. Did '
                'you mean %s? Accepted: %s'
                % (', '.join(unknown),
                   {'SHORTWAVE': 'SWDOWN', 'LONGWAVE': 'LWDOWN',
                    'TEMP': 'AIR_TEMP', 'RAIN': 'PREC',
                    'PRESS': 'PRESSURE'}.get(unknown[0], 'one of the below'),
                   ' '.join(sorted(self.VIC_FORCE_TYPES))))
        wanted = {v: self.CANONICAL.get(k, k.lower())
                  for k, v in self.force_type.items()}
        return wanted

    def read(self, dtype=None, device=None, log=None):
        """Load every file, concatenated in time, as {name: [time, cell]}."""
        import numpy as np
        import torch
        dtype = dtype or torch.float64
        say = _announcer(log)
        wanted = self._wanted()
        out, times, found = {}, [], set()
        for f in self.files:
            ds = self._open(f)
            base = os.path.basename(f)
            names = wanted or {k: k for k in ds.data_vars
                               if 'time' in ds[k].dims}
            t = np.asarray(ds['time'].values)
            step_s = self.step_seconds(t)
            for var, canon in sorted(names.items()):
                if var not in ds:
                    # A FORCE_TYPE naming a variable the file does not have
                    # used to be skipped in silence, which is how a forcing
                    # goes missing: the run completes, the field is simply
                    # never there, and whatever consumes it falls back to a
                    # default. Say it here; if no file has it, refuse below.
                    say('  %s: FORCE_TYPE names %r for %s and the file has '
                        'no such variable' % (base, var, canon))
                    continue
                found.add(var)
                # (…, y, x) in that order, or the cells taken are not the
                # cells meant -- the same check ImageDomain.take makes.
                self.domain.check_dims(ds[var], what='%s in %s' % (var, base))
                v = ds[var].values[..., self.domain.rows, self.domain.cols]
                declared = ds[var].attrs.get('units', '')
                units = self.normalise_units(declared)
                scale = self._scale(canon, declared, units, step_s, base, say)
                # VIC reads every forcing into a double and scales it there
                # (vic_force.c 5.1.0:151, :397-399): kPa -> Pa on a float32
                # array rounded vp to float32 *after* the scaling, so a
                # forcing saturated by MTCLIM (vp = svp(Tair) to float
                # precision) gave svp - vp <= 0 here, a deficit of 0 and no
                # canopy sublimation, where VIC's +1.7e-6 Pa kept both.
                v = np.asarray(v, dtype=np.float64)
                if scale != 1.0:
                    v = v * scale
                out.setdefault(canon, []).append(v)
            times.append((base, t))
            if log:
                log('  %s  %d steps' % (base, ds.sizes.get('time', 0)))
        missing = sorted(set(wanted) - found)
        if missing:
            raise ValueError(
                'FORCE_TYPE declares %s and no file matching %s contains '
                '%s; the run would proceed with the field absent'
                % (', '.join(repr(m) for m in missing), self._pattern,
                   'them' if len(missing) > 1 else 'it'))
        stamps = self._checked_times(times)
        fields = {k: torch.as_tensor(np.concatenate(v, axis=0), dtype=dtype,
                                     device=device)
                  for k, v in out.items()}
        return fields, stamps

    def plan(self, log=None):
        """Open every file and settle what `read` would read -- the time
        axis, which variable is which field, each one's unit scale -- without
        reading a field. Returns the time axis; `read_steps` then reads any
        span of it. The checks and messages are `read`'s.
        """
        import numpy as np
        say = _announcer(log)
        wanted = self._wanted()
        files, times, found = [], [], set()
        for f in self.files:
            ds = self._open(f)
            base = os.path.basename(f)
            names = wanted or {k: k for k in ds.data_vars
                               if 'time' in ds[k].dims}
            t = np.asarray(ds['time'].values)
            step_s = self.step_seconds(t)
            take = []
            for var, canon in sorted(names.items()):
                if var not in ds:
                    say('  %s: FORCE_TYPE names %r for %s and the file has '
                        'no such variable' % (base, var, canon))
                    continue
                found.add(var)
                self.domain.check_dims(ds[var], what='%s in %s' % (var, base))
                declared = ds[var].attrs.get('units', '')
                units = self.normalise_units(declared)
                take.append((var, canon, self._scale(canon, declared, units,
                                                     step_s, base, say)))
            files.append((ds, len(t), take))
            times.append((base, t))
            if log:
                log('  %s  %d steps' % (base, ds.sizes.get('time', 0)))
        missing = sorted(set(wanted) - found)
        if missing:
            raise ValueError(
                'FORCE_TYPE declares %s and no file matching %s contains '
                '%s; the run would proceed with the field absent'
                % (', '.join(repr(m) for m in missing), self._pattern,
                   'them' if len(missing) > 1 else 'it'))
        stamps = self._checked_times(times)
        self._plan = files
        return stamps

    def read_steps(self, start, stop):
        """Steps [start, stop) of `plan`'s time axis, active cells only, as
        {field: numpy [step, cell]} as float64 (VIC reads every forcing into
        a double and scales it there, as `read` does).

        The values are `read`'s bit for bit: the same decoded variable, the
        same cells, the same unit scale applied in the same dtype; `read`
        then casts to its dtype, which the caller does (exactly) after.
        Only the span is read, so a run can read its forcing a window at a
        time, on a background thread, while the previous window computes.
        """
        import numpy as np
        if getattr(self, '_plan', None) is None:
            raise RuntimeError('ImageForcing.read_steps needs plan() first')
        parts, off = {}, 0
        for ds, n, take in self._plan:
            a, b = max(start - off, 0), min(stop - off, n)
            off += n
            if a >= b:
                continue
            for var, canon, scale in take:
                v = np.asarray(ds[var][a:b].values[..., self.domain.rows,
                                                   self.domain.cols],
                               dtype=np.float64)
                if scale != 1.0:
                    v = v * scale
                parts.setdefault(canon, []).append(v)
        return {k: (v[0] if len(v) == 1 else np.concatenate(v, axis=0))
                for k, v in parts.items()}

    def close(self):
        """Close the files `plan` opened."""
        for ds, _, _ in getattr(self, '_plan', None) or ():
            ds.close()
        self._plan = None

    @staticmethod
    def _checked_times(per_file):
        """One time axis from many files, refusing a repeated timestamp.

        The file list is a glob, so a second copy of a year -- a re-download
        under a new name, a `.2008.nc` beside a `.2008.v2.nc` -- is read as
        extra records rather than as the same records twice. Downstream that
        is invisible in the wrong direction: `aggregate_to_daily` folds the
        repeated dates together with `np.unique`, so the day count stays 366
        and the means stay right while every *summed* field doubles. Measured
        on Walnut Gulch: precipitation 414.09 -> 828.17 mm with air
        temperature unmoved at 17.197 C, and no warning anywhere.
        """
        import numpy as np
        all_t = np.concatenate([t for _, t in per_file])
        uniq, counts = np.unique(all_t, return_counts=True)
        if uniq.size == all_t.size:
            return all_t
        dup = uniq[counts > 1]
        holders = [name for name, t in per_file if np.isin(t, dup).any()]
        raise ValueError(
            '%d timestamp(s) appear more than once across the forcing files '
            '(first %s, in %s). Two files covering the same time are read as '
            'twice the weather: every accumulated field doubles while the '
            'means and the day count stay right. Remove the duplicate or '
            'narrow the file list with years='
            % (int(dup.size), str(dup[0])[:19], ', '.join(holders)))

    def __repr__(self):
        return ('<ImageForcing %d file(s), %d active cells>'
                % (len(self.files), self.domain.n_active))


# ---------------------------------------------------------------------------
# Calibrating directly from VIC image-driver files.
#
# A VIC user with a working image-driver set-up should be able to hand this
# their global parameter file and run, with the physics inheriting the
# parameters VIC would have used.
#
# Observations are separate and always will be: VIC's input files describe the
# land surface and the weather, not the streamflow anyone measured. They are
# supplied alongside.
# ---------------------------------------------------------------------------
def local_days(times, utc_offset):
    """The calendar day each timestamp falls in, `utc_offset` hours from GMT.

    Forcing timestamps are UTC; the day a daily model integrates is the local
    one. VIC carries the difference per cell as `off_gmt` -- -7 at Walnut
    Gulch -- and binning on the UTC day instead silently redraws every day
    boundary by that much. The annual means are untouched, which is why a
    water balance cannot see it, but the days themselves move: measured
    against local binning at Walnut Gulch, mean |difference| per day was
    1.91 C in tmax (11.24 C at worst), 13.9 W/m2 in shortwave (133.7 at
    worst) and 0.589 mm in precipitation (18.19 mm at worst). Comparing those
    days against tower observations, which are recorded on local days, then
    compares two different days: shifting the simulation by one day *raised*
    the correlation at both Walnut Gulch towers (WHS 0.829 -> 0.836, WKG
    0.798 -> 0.824), which is the misalignment showing through.
    """
    import numpy as np
    t = np.asarray(times)
    if utc_offset:
        secs = int(round(float(utc_offset) * 3600.0))
        try:
            t = t.astype('datetime64[s]') + np.timedelta64(secs, 's')
        except (TypeError, ValueError):
            # cftime, under a calendar numpy has no dtype for.
            import datetime as _dt
            d = _dt.timedelta(seconds=secs)
            t = np.array([x + d for x in t], dtype=object)
    return np.array([str(x)[:10] for x in t])


def _ordered_index_sum(n, idx, v):
    """zeros(n, ...).index_add(0, idx, v), the sums formed in the order of
    `idx` on every device (core.ordered): on a GPU index_add's atomics
    would add a day's MTCLIM hours in any order."""
    import torch
    from ..core import ordered
    out = torch.zeros((n,) + tuple(v.shape[1:]), dtype=v.dtype,
                      device=v.device)
    return ordered.index_add_(out, 0, idx, v)


def aggregate_to_daily(fields, times, how=None, utc_offset=None, log=None):
    """Sub-daily forcing to daily, by the rule each variable deserves.

    VIC image forcing is commonly hourly. This model steps daily, and the
    aggregation is not one rule: precipitation accumulates, temperature gives
    a daily maximum and minimum as well as a mean, and fluxes average. Summing
    a temperature or averaging a precipitation would both be wrong and neither
    would look obviously wrong in the output.

    `utc_offset` is hours from GMT -- the parameter file's `off_gmt`, which
    `ImageParams.utc_offset_hours()` returns -- and decides where the day
    boundaries fall. It has no default value because there is no defensible
    one: 0 is a claim about the domain, not an absence of a claim. Omitting it
    bins on the UTC day as this function always did, and says so, so that a
    caller who never passes it still gets a run and still gets told which day
    it is on.
    """
    import numpy as np
    import torch
    say = _announcer(log)
    if utc_offset is None:
        say('  aggregate_to_daily: binning on the UTC calendar day because no '
            'utc_offset was given; pass ImageParams.utc_offset_hours() to bin '
            'on the local day the forcing was measured in')
        utc_offset = 0.0
    days = local_days(times, utc_offset)
    uniq, inverse = np.unique(days, return_inverse=True)
    idx = torch.as_tensor(inverse, dtype=torch.long,
                          device=next(iter(fields.values())).device)
    nd = len(uniq)
    rule = dict(prcp='sum', tair='mean', shortwave='mean', longwave='mean',
                press='mean', vp='mean', wind='mean', tmax='max', tmin='min')
    rule.update(how or {})
    out = {}
    for k, v in fields.items():
        r = rule.get(k, 'mean')
        if r in ('max', 'min'):
            fill = -1e30 if r == 'max' else 1e30
            out[k] = torch.full((nd,) + v.shape[1:], fill, dtype=v.dtype,
                                device=v.device).index_reduce(
                0, idx, v, 'amax' if r == 'max' else 'amin',
                include_self=False)
            continue
        acc = _ordered_index_sum(nd, idx, v)
        if r == 'sum':
            out[k] = acc
        else:
            cnt = torch.zeros(nd, dtype=v.dtype, device=v.device).index_add(
                0, idx, torch.ones_like(v[:, :1]).squeeze(-1))
            out[k] = acc / cnt.unsqueeze(-1).clamp(min=1)
    if 'tair' in fields:
        # Daily extremes, which a daily mean cannot recover and which the
        # rain-snow partition and the vegetation both need. Derived only where
        # the file does not supply them: this used to overwrite whatever the
        # forcing declared, so a file whose own TMAX said 99 everywhere came
        # back as tair's daily range and its field was gone without trace --
        # 7.0 C on Walnut Gulch. The file's TMAX is
        # the observation; tair's daily range is a reconstruction of it, and
        # the reconstruction does not get to replace the observation silently.
        for k, red in (('tmax', 'amax'), ('tmin', 'amin')):
            if k in fields:
                say('  aggregate_to_daily: %s comes from the forcing file; it '
                    'is aggregated by "%s" and NOT recomputed from tair'
                    % (k, rule.get(k)))
                continue
            big = torch.full((nd,) + fields['tair'].shape[1:], -1e30,
                             dtype=fields['tair'].dtype,
                             device=fields['tair'].device)
            out[k] = big.index_reduce(0, idx, fields['tair'], red,
                                      include_self=False)
    import datetime as _dt
    dates = [_dt.date(*map(int, d.split('-'))) for d in uniq]
    return out, dates


class WindowPrefetch(object):
    """Forcing windows read ahead on a background thread.

        pre = WindowPrefetch(forcing, [(a0, a1), (a1, a2), ...], fields,
                             device, dtype)
        for f in pre:          # {field: [step, cell]} on `device`, `dtype`
            ...

    `forcing` is an ImageForcing after `plan()`. While the caller computes
    window k, a thread reads window k+1 (`read_steps`), stages it in pinned
    memory and copies it to the device on a side CUDA stream, in the file's
    dtype, casting to `dtype` there; the caller's stream waits on that copy
    only when it takes the window. The values are `read`'s: the cast from
    the file's float32 to float64 is exact wherever it happens.

    At most `depth` windows wait on the device. `wait_s` is the time the
    caller spent waiting for a window (the I/O not hidden by compute).
    """

    def __init__(self, forcing, spans, fields, device, dtype, depth=2):
        import queue
        import threading
        import torch
        self.forcing, self.spans = forcing, list(spans)
        self.fields, self.dtype = tuple(fields), dtype
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.wait_s = 0.0
        self._q = queue.Queue(maxsize=max(int(depth), 1))
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name='forcing-prefetch')
        self._t.start()

    def _run(self):
        import torch
        try:
            stream = (torch.cuda.Stream(device=self.device) if self.cuda
                      else None)
            for a, b in self.spans:
                if self._stop:
                    return
                blk = self.forcing.read_steps(a, b)
                out, ev = {}, None
                if self.cuda:
                    with torch.cuda.stream(stream):
                        for k in self.fields:
                            h = torch.from_numpy(blk[k]).pin_memory()
                            out[k] = h.to(self.device, non_blocking=True
                                          ).to(self.dtype)
                        ev = torch.cuda.Event()
                        ev.record(stream)
                else:
                    for k in self.fields:
                        out[k] = torch.from_numpy(blk[k]).to(
                            self.device, self.dtype)
                del blk
                self._q.put((out, ev, None))
        except BaseException as e:      # handed to the caller
            self._q.put((None, None, e))

    def __iter__(self):
        import time
        import torch
        for _ in self.spans:
            t = time.time()
            out, ev, err = self._q.get()
            self.wait_s += time.time() - t
            if err is not None:
                raise err
            if ev is not None:
                cur = torch.cuda.current_stream(self.device)
                cur.wait_event(ev)
                for v in out.values():
                    v.record_stream(cur)
            yield out

    def close(self):
        self._stop = True
        while self._t.is_alive():
            try:
                self._q.get(timeout=0.1)
            except Exception:
                pass
