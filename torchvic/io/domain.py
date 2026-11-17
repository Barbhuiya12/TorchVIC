"""The image driver's domain file: the grid, its mask and the active
cells (VIC 5's get_global_domain.c).
"""
import os

# ---------------------------------------------------------------------------
# Domain and parameters.
#
# VIC's parameter file carries more than this model was assembling by hand:
# per-cell fractional cover, per-cell monthly leaf area index, and per-cell
# root zones. Reading it removes the approximation an earlier version made, where
# root depths and leaf area were borrowed from one cell of a sample file
# because MODIS supplies cover and not rooting depth. Here they are per cell,
# from the same file VIC itself reads.
# ---------------------------------------------------------------------------
def _open(path):
    try:
        import xarray as xr
    except ImportError:                                  # pragma: no cover
        raise ImportError('xarray is required to read VIC image-driver files; '
                          'pip install xarray netCDF4')
    if hasattr(path, 'data_vars'):
        # Already open -- typically a whole-domain parameter file loaded
        # into RAM once, so that many cell batches are read from memory
        # rather than each re-reading gigabytes of netCDF from disk.
        return path
    if not os.path.exists(path):
        # A global file can name a file that is not there -- UW-Hydro's own
        # Walnut Gulch sample says `parameters.WalnutGulch...nc` while the
        # file shipped beside it is `params.WalnutGulch...nc`. Naming the
        # candidates turns a bare "no such file" into a one-line fix, without
        # guessing which of them was meant.
        import difflib
        d = os.path.dirname(path) or '.'
        near = []
        if os.path.isdir(d):
            near = difflib.get_close_matches(os.path.basename(path),
                                             os.listdir(d), n=3, cutoff=0.5)
            hint = ('; the directory holds %s -- the global file may name it '
                    'wrongly' % ', '.join(near)) if near else ''
        else:
            # The directory itself is wrong, so there are no files to
            # suggest. UW-Hydro's FindleyLake global file says `params/`
            # where the published directory is `parameters/` -- the Walnut
            # Gulch filename typo one level up. Look for a sibling directory
            # instead, or the message is a bare "no such file" pointing at a
            # directory that never existed.
            parent = os.path.dirname(d) or '.'
            sibs = (difflib.get_close_matches(os.path.basename(d),
                                              os.listdir(parent), n=2,
                                              cutoff=0.5)
                    if os.path.isdir(parent) else [])
            hint = ('; the directory %s does not exist, but %s does -- the '
                    'global file may name the directory wrongly'
                    % (os.path.basename(d), ' or '.join(sibs))) if sibs else \
                   '; and neither does its directory %s' % d
        raise IOError('%s does not exist%s' % (path, hint))
    from .ncshare import read
    return read(path)


def _run_cell_of(ds):
    """`run_cell` from an open parameter file, or None if it has none.

    VIC insists the variable be `NC_INT` (get_global_domain.c:57-60) but is
    otherwise indifferent to how it was written, and UW-Hydro's own Walnut
    Gulch parameter file stores it as a float. Refusing that file would refuse
    the official sample data over a storage type, so the value is compared
    rather than the dtype.
    """
    import numpy as np
    if 'run_cell' not in ds:
        return None
    return np.asarray(ds['run_cell'].values)


def active_cells(run, mask):
    """VIC's active set: `run_cell` from the parameter file, bounded by `mask`.

    get_global_domain.c:53-78 is unambiguous about which of the two decides.
    `run_cell` comes from the *parameter* file and is counted into
    `ncells_active`; the domain file's `mask` only bounds it, and a cell with
    `run_cell == 1 && mask != 1` is a `log_err` -- VIC stops rather than
    resolving the disagreement. So the same disagreement is raised here.

    Selecting on the mask instead happens to give the same answer wherever the
    two coincide, which is every cell of Walnut Gulch, and gives a different
    basin with a perfectly closed water balance wherever they do not. That is
    the shape of defect this reader exists to refuse.
    """
    import numpy as np
    run, mask = np.asarray(run), np.asarray(mask)
    if run.shape != mask.shape:
        raise ValueError('run_cell is %s and the domain mask is %s; the '
                         'parameter file and the domain file describe '
                         'different grids' % (run.shape, mask.shape))
    active = np.isclose(run, 1)
    bad = np.argwhere(active & ~np.isclose(mask, 1))
    if bad.size:
        raise ValueError(
            'run_cell = 1 should only appear within the mask of the domain '
            "file (VIC's own check, get_global_domain.c:64-69); %d cell(s) "
            'have run_cell = 1 where mask != 1, first at (y=%d, x=%d)'
            % (bad.shape[0], bad[0][0], bad[0][1]))
    return active


class ImageDomain(object):
    """VIC's domain file: which cells run, how large they are, and where.

    Only active cells are carried forward. A continental domain is mostly
    inactive, and the difference between selecting on read and selecting later
    is the difference between a basin-sized problem and a continental one.

    *Which* cells are active is the parameter file's answer, not the domain
    file's: pass `params` and the active set comes from `run_cell` as VIC
    takes it. Without it the mask is used and the substitution is announced,
    because a reader that quietly swapped one field for the other would give a
    different basin mean -- measured at up to 3.3% on a domain where they
    differ -- with nothing in the output to show for it.
    """

    def __init__(self, path, names=None, params=None):
        import numpy as np
        n = dict({'lat_var': 'lat', 'lon_var': 'lon', 'mask_var': 'mask',
                  'area_var': 'area', 'frac_var': 'frac',
                  'y_dim': 'lat', 'x_dim': 'lon'}, **(names or {}))
        self.path = path
        self.names = n
        ds = _open(path)
        self.ds = ds
        self.y_dim, self.x_dim = n['y_dim'], n['x_dim']
        self.check_dims(ds[n['mask_var']],
                        what='%s in %s' % (n['mask_var'],
                                           os.path.basename(path)))
        mask = np.asarray(ds[n['mask_var']].values)
        self.mask = mask
        self.ny, self.nx = mask.shape
        self.run_cell = None
        self.selected_on = 'mask'
        if params is not None:
            pds = params if hasattr(params, 'data_vars') else _open(params)
            run = _run_cell_of(pds)
            if run is None:
                raise ValueError(
                    '%s has no run_cell variable; VIC reads the active set '
                    'from the parameter file (get_global_domain.c:53-61) and '
                    'a parameter file without it is not a VIC parameter file'
                    % getattr(params, 'encoding', {}).get('source', params))
            self.check_dims(pds['run_cell'], what='run_cell')
            self.run_cell = run
            active = active_cells(run, mask)
            self.selected_on = 'run_cell'
        else:
            active = mask > 0
        rc = np.argwhere(active)
        self.rows, self.cols = rc[:, 0], rc[:, 1]
        self.n_active = int(rc.shape[0])
        lat = np.asarray(ds[n['lat_var']].values)
        lon = np.asarray(ds[n['lon_var']].values)
        # A domain may give 1-D coordinate axes or 2-D coordinate fields.
        self.lat = (lat[self.rows] if lat.ndim == 1
                    else lat[self.rows, self.cols])
        self.lon = (lon[self.cols] if lon.ndim == 1
                    else lon[self.rows, self.cols])
        self.area = (self.take(ds[n['area_var']])
                     if n['area_var'] in ds else None)
        self.frac = (self.take(ds[n['frac_var']])
                     if n['frac_var'] in ds else None)
        # Routing to one gauge: each cell's flow length to the outlet, m,
        # written into the domain file from a D8 network. VIC 5's C ignores
        # extra domain variables;
        # without it the basin is routed as one lumped unit hydrograph.
        self.flow_distance = (self.take(ds['flow_distance'])
                              if 'flow_distance' in ds else None)

    def check_dims(self, da, what=None):
        """Refuse an array whose last two axes are not (y_dim, x_dim).

        `take` indexes `v[..., rows, cols]`, which is only the right cell if
        the trailing axes really are latitude then longitude. The global file
        declares which dimensions those are -- `DOMAIN_TYPE YDIM`/`XDIM` --
        and until now this reader stored the two names and never looked at
        them again. On a square grid a transposed file then reads plausible
        wrong numbers rather than failing: elevations of [0, 600, 1200] came
        back as [0, 900, 300], and with `nlayer` equal to `nx` a 3-D variable
        was misread the same silent way. The non-square case raised
        IndexError, which is luck, not a check.

        So the declared names are compared against the variable's own dims.
        VIC makes the same demand from the other side --
        compare_ncdomain_with_global_domain() stops the run when the parameter
        file's grid does not match the domain file's.
        """
        dims = getattr(da, 'dims', None)
        name = what or getattr(da, 'name', None) or '<array>'
        if dims is None:
            return
        dims = tuple(str(d) for d in dims)
        if dims[-2:] != (str(self.y_dim), str(self.x_dim)):
            raise ValueError(
                "%s has dimensions %s; this domain's cells are indexed on "
                '(%s, %s) in that order, as the global file declares with '
                'DOMAIN_TYPE YDIM/XDIM. Transpose the variable or correct '
                'the declaration -- on a square grid indexing it as it '
                'stands returns the wrong cells without failing'
                % (name, dims, self.y_dim, self.x_dim))

    def take(self, da):
        """A (…, y, x) array reduced to (…, active cell)."""
        import numpy as np
        self.check_dims(da)
        v = np.asarray(getattr(da, 'values', da))
        if v.shape[-2:] != (self.ny, self.nx):
            raise ValueError(
                '%s has grid shape %s and the domain is %dx%d'
                % (getattr(da, 'name', None) or '<array>', v.shape[-2:],
                   self.ny, self.nx))
        return v[..., self.rows, self.cols]

    def __repr__(self):
        return ('<ImageDomain %dx%d grid, %d active cells on %s, '
                'lat %.4f..%.4f>'
                % (self.ny, self.nx, self.n_active, self.selected_on,
                   float(self.lat.min()), float(self.lat.max())))
