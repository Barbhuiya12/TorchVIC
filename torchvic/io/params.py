"""The image driver's parameter file: soil, vegetation and snow-band
fields on the domain's active cells.
"""
import os

from .domain import _open, _run_cell_of, active_cells

#: bump when what ImageParams derives from the file changes (the cache of
#: the reduced parameters, see _cache_path, is then made again)
PARAMS_CACHE_VERSION = 4
_LAZY = object()    # ImageParams.ds not opened yet
_CACHE_FIELDS = ('off_gmt', 'nveg', 'july_tavg', 'bubble')
_CACHE_DICTS = ('soil', 'veg', 'bands')
_CACHE_INTS = ('n_layers', 'n_veg', 'n_bands', 'n_root')


def _file_checksum(path):
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(8 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _save_arrays(path, arrays):
    """{name: ndarray} as a zip of .npy members, deflated at level 1 (the
    parameters are mostly constant: 1.3 GB shrinks to a few hundred MB)."""
    import zipfile
    import numpy as np
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=1) as zf:
        for k, v in arrays.items():
            v = np.asarray(v)
            with zf.open(k.replace('/', '|') + '.npy', 'w',
                         force_zip64=True) as fh:
                np.lib.format.write_array(fh, v, allow_pickle=False)


def _load_arrays(path, threads=8):
    """What _save_arrays wrote, the members inflated by a pool of threads
    (zlib and the copy release the GIL)."""
    import concurrent.futures as cf
    import zipfile
    import numpy as np
    with zipfile.ZipFile(path) as zf:
        names = [i.filename for i in zf.infolist()]
    local = __import__('threading').local()

    def one(name):
        zf = getattr(local, 'zf', None)
        if zf is None:
            zf = local.zf = zipfile.ZipFile(path)
        with zf.open(name) as fh:
            return name, np.lib.format.read_array(fh, allow_pickle=False)
    with cf.ThreadPoolExecutor(max(1, min(threads, len(names)))) as ex:
        got = dict(ex.map(one, names))
    return {k[:-4].replace('|', '/'): v for k, v in got.items()}


def _cache_path(path, domain, cls):
    """Where the parameters of `path` reduced to `domain`'s active cells are
    kept: a binary next to the input (.torchvic_cache/), named by the
    checksum of the file's contents, of the active cells and of the variable
    lists read, so a changed file, domain or code never finds it. None where
    caching is off (TORCHVIC_PARAMS_CACHE=0) or the file is not a path."""
    import hashlib
    import numpy as np
    if (os.environ.get('TORCHVIC_PARAMS_CACHE', '1') == '0'
            or not isinstance(path, (str, os.PathLike))
            or not os.path.isfile(path)):
        return None
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((PARAMS_CACHE_VERSION, cls.SOIL_2D, cls.SOIL_3D,
                   cls.VEG_VARS, int(domain.ny), int(domain.nx),
                   int(domain.n_active), domain.run_cell is None)).encode())
    h.update(np.ascontiguousarray(domain.rows, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(domain.cols, dtype=np.int64).tobytes())
    h.update(_file_checksum(path).encode())
    d = os.path.join(os.path.dirname(os.path.abspath(path)),
                     '.torchvic_cache')
    return os.path.join(d, '%s.%s.zip' % (os.path.basename(path),
                                          h.hexdigest()))


def params_cache_valid(path, domain):
    """The reduced-parameter cache of `path` for `domain` exists."""
    try:
        cp = _cache_path(path, domain, ImageParams)
    except Exception:
        return False
    return cp is not None and os.path.isfile(cp)


class ImageParams(object):
    """VIC's parameter file, reduced to the active cells of a domain."""

    SOIL_2D = ('infilt', 'Ds', 'Dsmax', 'Ws', 'c', 'elev', 'annual_prec',
               'avg_T', 'dp', 'rough', 'snow_rough', 'frost_slope',
               # vic_init.c 5.1.0:684-689, the per-cell frozen-soil switch
               # (frozen_soil.c:39, :384, func_surf_energy_bal.c:535 run
               # the frozen path only where it is set).
               'fs_active',
               # vic_init.c 5.1.0:691-700, read under SPATIAL_SNOW only.
               'max_snow_distrib_slope')
    SOIL_3D = ('expt', 'Ksat', 'depth', 'init_moist', 'bulk_density',
               'soil_density', 'quartz', 'Wcr_FRACT', 'Wpwp_FRACT',
               'resid_moist',
               # vic_init.c 5.1.0:572-617, read under ORGANIC_FRACT and
               # BULK_DENSITY_COMB only.
               'organic', 'bulk_density_org', 'soil_density_org',
               'bulk_density_comb')
    #: In image mode VIC's vegetation *library* lives in the parameter file
    #: alongside the per-cell cover, so the canopy scalars -- minimum stomatal
    #: resistance, architectural resistance, the radiation threshold for
    #: transpiration -- are read here rather than from a separate library.
    #: Omitting them does not fail: the mosaic falls back to defaults and the
    #: run completes with every class sharing one resistance, which is a
    #: silently wrong canopy rather than a missing one.
    VEG_VARS = ('Cv', 'LAI', 'albedo', 'root_depth', 'root_fract',
                'veg_rough', 'displacement', 'fcanopy',
                'rmin', 'rarc', 'RGL', 'overstory', 'trunk_ratio',
                'wind_h', 'wind_atten', 'rad_atten',
                # vic_init.c 5.1.0:186-196, read under MAX_SNOW_ALBEDO only.
                'max_snow_albedo',
                # read_veglib.c:162-192, the seven fields VEGLIB_PHOTO
                # adds, plus the light-use efficiency. Picked up only
                # when the file has them, like everything else here.
                'Ctype', 'MaxCarboxRate', 'MaxETransport',
                'CO2Specificity', 'LightUseEff', 'NscaleFlag',
                'Wnpp_inhib', 'NPPfactor_sat',
                # vic_init.c 5.1.0:1281-1330, BLOWING's three per-class
                # fields; build_mosaic hands them to the blowing snow.
                'sigma_slope', 'lag_one', 'fetch')

    @property
    def ds(self):
        """The file as an in-memory dataset (opened at first use: a run
        that took the parameters from the cache never needs it)."""
        if self.__dict__.get('_ds', _LAZY) is _LAZY:
            self._ds = _open(self.path)
        return self._ds

    @ds.setter
    def ds(self, value):
        self._ds = value

    #: the image reader always has a dataset (the classic reader's is None)
    has_dataset = True

    def _from_cache(self, cp):
        import numpy as np
        try:
            got = _load_arrays(cp)
            vals = {}
            for k in _CACHE_FIELDS:
                vals[k] = got['f/' + k] if 'f/' + k in got else None
            for d in _CACHE_DICTS:
                pre = 'd/%s/' % d
                vals[d] = {k[len(pre):]: v for k, v in got.items()
                           if k.startswith(pre)}
            for k in _CACHE_INTS:
                vals[k] = int(got['i/' + k])
        except Exception:
            return False            # unreadable or stale: read the file
        for k, v in vals.items():
            setattr(self, k, v)
        return True

    def _to_cache(self, cp):
        import numpy as np
        z = {}
        for k in _CACHE_FIELDS:
            v = getattr(self, k)
            if v is not None:
                z['f/' + k] = np.asarray(v)
        for d in _CACHE_DICTS:
            for k, v in getattr(self, d).items():
                z['d/%s/%s' % (d, k)] = np.asarray(v)
        for k in _CACHE_INTS:
            z['i/' + k] = np.asarray(getattr(self, k))
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        tmp = '%s.%d.tmp' % (cp, os.getpid())
        try:
            _save_arrays(tmp, z)
            os.replace(tmp, cp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def __init__(self, path, domain):
        import numpy as np
        self.path = path
        self.domain = domain
        self._ds = _LAZY
        cp = None
        try:
            cp = _cache_path(path, domain, type(self))
        except Exception:
            cp = None
        if cp is not None and os.path.isfile(cp) and self._from_cache(cp):
            return
        ds = self._ds = _open(path)
        # The domain may have been built before this file was opened -- the
        # constructor takes only a domain path -- in which case its active set
        # came from the mask and VIC's came from run_cell here. Where the two
        # agree nothing is said and nothing changes; where they disagree the
        # basin is a different basin, so this is the last moment at which the
        # substitution can be caught, and it is caught rather than absorbed.
        run = _run_cell_of(ds)
        if run is not None and domain.run_cell is None:
            active = active_cells(run, domain.mask)   # raises VIC's log_err
            if int(active.sum()) != domain.n_active or \
                    not np.all(active[domain.rows, domain.cols]):
                raise ValueError(
                    '%s selects %d cells with run_cell and the domain mask '
                    'selects %d; VIC runs the run_cell set '
                    '(get_global_domain.c:70-78). Build the domain as '
                    'ImageDomain(domain_file, names, params=%r) so the active '
                    'set is the one VIC would use'
                    % (os.path.basename(path), int(active.sum()),
                       domain.n_active, os.path.basename(path)))
        #: Hours from GMT, per active cell. VIC carries it as a soil parameter
        #: and the image driver uses it to place the solar day; here it is
        #: what lets `aggregate_to_daily` bin on the *local* day instead of
        #: the UTC one. Kept off `soil` deliberately: it is metadata about the
        #: clock, not a soil property anything downstream integrates.
        self.off_gmt = domain.take(ds['off_gmt']) if 'off_gmt' in ds else None
        # bubbling pressure (cm), read here and cached so that build_soil need
        # not open the file for it; None where the file has none
        self.bubble = domain.take(ds['bubble']) if 'bubble' in ds else None
        #: The file's `Nveg` per active cell, or None. vic_alloc.c:128 sizes
        #: the cell's tiles as Nveg + 1 and vic_run treats tile index Nveg as
        #: bare soil (VEG false), so where Nveg is one less than the number
        #: of covered classes the last covered class runs with no
        #: vegetation at all; image_run.build_mosaic reads it for that.
        self.nveg = domain.take(ds['Nveg']) if 'Nveg' in ds else None
        self.soil = {k: domain.take(ds[k]) for k in self.SOIL_2D if k in ds}
        self.soil.update({k: domain.take(ds[k])
                          for k in self.SOIL_3D if k in ds})
        #: The cell's average July air temperature (read_soilparam.c:574-586,
        #: JULY_TAVG_SUPPLIED), which COMPUTE_TREELINE reads; None when the
        #: file has no `July_Tavg`. -999 turns the treeline off for a cell
        #: (classic vic_force.c:292-296).
        self.july_tavg = (np.asarray(domain.take(ds['July_Tavg']),
                                     dtype=np.float64)
                          if 'July_Tavg' in ds else None)
        self.veg = {k: domain.take(ds[k])
                    for k in self.VEG_VARS if k in ds}
        if 'MaxiE_or_CO2Spec' in ds and 'Ctype' in ds:
            # The image driver's names for the carbon library
            # (shared_image/src/vic_init.c 5.1.0:285-375). Read as the C
            # reads them: the one MaxiE_or_CO2Spec field is assigned to
            # MaxCarboxRate for a C3 class (overwriting what the
            # MaxCarboxRate field gave) and to CO2Specificity for a C4 one,
            # which zeroes MaxCarboxRate; MaxETransport is never assigned,
            # so it stays at the calloc's zero.
            import numpy as np
            ct = self.veg['Ctype']
            e = domain.take(ds['MaxiE_or_CO2Spec'])
            c3 = np.asarray(ct) == 0
            self.veg['MaxCarboxRate'] = np.where(c3, e, 0.0)
            self.veg['CO2Specificity'] = np.where(c3, 0.0, e)
            self.veg['MaxETransport'] = np.zeros_like(
                np.asarray(e, dtype=np.float64))
            for src, dst in (('LUE', 'LightUseEff'), ('Nscale', 'NscaleFlag')):
                if src in ds:
                    self.veg[dst] = domain.take(ds[src])
        self.bands = {k: domain.take(ds[k])
                      for k in ('AreaFract', 'Pfactor', 'elevation')
                      if k in ds}
        self.n_layers = int(ds.sizes.get('nlayer', 3))
        self.n_veg = int(ds.sizes.get('veg_class', 0))
        self.n_bands = int(ds.sizes.get('snow_band', 1))
        self.n_root = int(ds.sizes.get('root_zone', 0))
        if cp is not None:
            try:
                self._to_cache(cp)
            except OSError:
                pass            # a read-only input directory: no cache

    def utc_offset_hours(self):
        """The domain's one UTC offset, for daily binning, or None.

        Daily aggregation puts every cell of the basin on one calendar day, so
        it needs one number; a domain spanning two time zones has no such
        number and gets told so rather than handed the first cell's. None
        means the parameter file carries no `off_gmt` -- also an answer the
        caller has to see, since the alternative is binning on UTC while
        believing it is local.
        """
        import numpy as np
        if self.off_gmt is None:
            return None
        u = np.unique(np.asarray(self.off_gmt))
        if u.size != 1:
            raise ValueError(
                'off_gmt takes %d different values across the active cells '
                '(%s); one daily binning offset cannot represent a domain '
                'spanning time zones -- aggregate per zone or step sub-daily'
                % (u.size, ', '.join('%g' % x for x in u[:6])))
        return float(u[0])

    def apply_density_options(self, organic_fract=False,
                              bulk_density_comb=False):
        """The soil densities as vic_init.c 5.1.0:554-627 and :735-737 set
        them under ORGANIC_FRACT and BULK_DENSITY_COMB.

        The file's `bulk_density` and `soil_density` are always the mineral
        soil's (bulk_dens_min, soil_dens_min: :554-567). The soil's own
        bulk density is then the file's `bulk_density_comb` (BULK_DENSITY_COMB)
        or (1 - organic) * mineral + organic * bulk_density_org (:621-625),
        and its particle density (1 - organic) * mineral + organic *
        soil_density_org (:735-737); without ORGANIC_FRACT the organic
        fraction and both organic densities are zero (init_library.c:51-66).
        `soil['bulk_density']` and `soil['soil_density']` become those, so
        the porosity and every moisture bound read from them are VIC's; the
        mineral pair is kept as `bulk_dens_min` and `soil_dens_min`, with
        `organic`, for the thermal properties (soil_conduction.c:17-99).
        Neither option set: nothing is touched. Raises ValueError naming a
        variable the file lacks."""
        import numpy as np
        if not (organic_fract or bulk_density_comb):
            return
        s = self.soil
        need = []
        if organic_fract:
            need += ['organic', 'soil_density_org']
            if not bulk_density_comb:
                need.append('bulk_density_org')
        if bulk_density_comb:
            need.append('bulk_density_comb')
        miss = [k for k in need if k not in s]
        if miss:
            raise ValueError(
                'ORGANIC_FRACT %s, BULK_DENSITY_COMB %s: the parameter file '
                'has no %s, which VIC reads (vic_init.c:572-617)'
                % ('TRUE' if organic_fract else 'FALSE',
                   'TRUE' if bulk_density_comb else 'FALSE',
                   ', '.join(miss)))
        f64 = lambda k: np.asarray(s[k], np.float64)
        bmin, smin = f64('bulk_density'), f64('soil_density')
        if organic_fract:
            org, sorg = f64('organic'), f64('soil_density_org')
        else:
            org = np.zeros_like(bmin)
            sorg = np.zeros_like(bmin)
        if bulk_density_comb:
            bulk = f64('bulk_density_comb')
        else:
            borg = (f64('bulk_density_org') if organic_fract
                    else np.zeros_like(bmin))
            bulk = (1 - org) * bmin + org * borg
        s['bulk_dens_min'], s['soil_dens_min'], s['organic'] = bmin, smin, org
        s['bulk_density'] = bulk
        s['soil_density'] = (1 - org) * smin + org * sorg

    # -- derived quantities VIC computes on read --------------------------
    def porosity(self):
        """1 - bulk_density / soil_density.  read_soilparam.c."""
        return 1.0 - self.soil['bulk_density'] / self.soil['soil_density']

    def max_moist(self):
        """depth * porosity * 1000, mm.  read_soilparam.c line 606."""
        return self.soil['depth'] * self.porosity() * 1000.0

    def critical_moisture(self):
        """Wcr and Wpwp as absolute moisture, mm, not fractions."""
        mm = self.max_moist()
        return (self.soil['Wcr_FRACT'] * mm, self.soil['Wpwp_FRACT'] * mm)

    def summary(self):
        return ('<ImageParams %d cells, %d layers, %d veg classes, '
                '%d snow bands, %d root zones | soil %s | veg %s>'
                % (self.domain.n_active, self.n_layers, self.n_veg,
                   self.n_bands, self.n_root,
                   ','.join(sorted(self.soil)), ','.join(sorted(self.veg))))

    def __repr__(self):
        return self.summary()
