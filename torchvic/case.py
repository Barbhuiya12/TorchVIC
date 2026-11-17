"""Load a VIC 5 image-driver case into everything `basin.simulate` needs,
from the global parameter file the user already has.

The sequence -- global file, domain, parameters, forcing, the band-weighted
elevation VIC substitutes, the atmosphere, the soil, the mosaic, the wind
height, the initial moisture and the physics options -- has to be identical
everywhere or two callers silently run two different models, so it lives
here and nowhere else.

Two typos in UW-Hydro's own sample global files are handled by name and
announced rather than worked around silently: Walnut Gulch writes
`parameters.` for a file called `params.`, FindleyLake writes `params/`
for a directory called `parameters/`.
"""
import os

import numpy as np
import torch

from .model import basin
from .core import atmos
from .core import snowbands
from .core import tfallback as _tfb
from .core import treeline
from .core import version as core_version
from .core.snow import svp as _svp
from .io import build as image_run
from .io.global_param import GlobalParam
from .io.domain import ImageDomain
from .io.forcing import ImageForcing, aggregate_to_daily, local_days
from .io.params import ImageParams
from .io import options as _opt
from .io import veghist

def sample_data_dir():
    """Where ${VIC_SAMPLE_DATA} in a global file points unless the caller
    says otherwise: $TORCHVIC_SAMPLE_DATA, else $VIC_SAMPLE_DATA (UW-Hydro's
    own name for it), else None, and the variable is then expanded from the
    environment as written."""
    return (os.environ.get('TORCHVIC_SAMPLE_DATA')
            or os.environ.get('VIC_SAMPLE_DATA') or None)

#: VIC's own defaults for what the parameter file does not carry: the rain
#: and snow partition, the snow albedo decay, the routing unit hydrograph and
#: neutral (identity) multipliers on the parameter file's values.
DEFAULT_PARAMS = dict(
    pcf=1.0, t_max_snow=0.5, t_min_rain=-0.5, rs=120.0, et_beta=1.0,
    new_snow_albedo=0.85, alb_accum_a=0.94, alb_accum_b=0.58,
    alb_thaw_a=0.82, alb_thaw_b=0.46, uh_shape=2.5, uh_scale=2.0,
    lai_scale=1.0, rmin_scale=1.0, w_crit=0.7, w_wilt=0.3,
    route_velocity=1.5, route_diffusion=800.0, lai_water_factor=0.1,
    bottom_depth_scale=1.0, top_depth_scale=1.0, wcr_scale=1.0,
    wpwp_scale=1.0)


def default_params(dtype=torch.float64, device=None, **overrides):
    """DEFAULT_PARAMS as tensors, with any of them overridden."""
    out = {k: torch.as_tensor(float(overrides.get(k, v)), dtype=dtype,
                              device=device)
           for k, v in DEFAULT_PARAMS.items()}
    for k, v in overrides.items():
        if k not in out:
            out[k] = torch.as_tensor(v, dtype=dtype, device=device)
    return out


def _rep(x, n):
    """Repeat a tensor n times along its leading (cell) axis, member-major:
    row m*n_cells + c is member m's cell c."""
    if isinstance(x, torch.Tensor):
        return x.repeat((n,) + (1,) * (x.dim() - 1)) if x.dim() else x
    if isinstance(x, dict):
        return {k: _rep(v, n) for k, v in x.items()}
    return x


def _fix_path(path, log):
    """The sample files' two path typos, by name, only if the path is wrong."""
    if os.path.exists(path):
        return path
    for a, b in (('/params/', '/parameters/'), ('parameters.', 'params.')):
        if a in path and os.path.exists(path.replace(a, b)):
            log('%s does not exist; using %s (the global file names it '
                'wrongly)' % (path, path.replace(a, b)))
            return path.replace(a, b)
    return path


def requested_outvars(gp):
    """The OUT_* names global file `gp`'s OUTFILE streams ask for
    (basin.simulate's `out_vars`), or None -- no extra row -- when it has no
    streams, or is a VIC 4.2.d file, whose names are not these. Shared by
    Case and the coupling's stream driver."""
    try:
        from .io import history
        if history.convention_of(gp) == 'vic4.2d':
            return None
        st = history.parse_streams(gp, log=lambda *a: None)
        return frozenset(v.name for s_ in st for v in s_.vars) or None
    except Exception:                                       # noqa: BLE001
        return None


class Case(object):
    """A loaded case. `kwargs()` is what `basin.simulate` takes."""

    def __init__(self, gp, dom, par, batch, atm, soil, mosaic, dates,
                 model_steps, wind_h, init_moist, full_energy, dtype, device,
                 n_cells=None, options=None, core=None):
        self.gp, self.dom, self.par = gp, dom, par
        self.batch, self.atm, self.soil, self.mosaic = batch, atm, soil, mosaic
        self.dates, self.model_steps = dates, model_steps
        self.wind_h, self.init_moist = wind_h, init_moist
        self.full_energy = full_energy
        self.dtype, self.device = dtype, device
        # The global file's physics switches, as basin.simulate's keyword
        # arguments (image_options), and the VIC core the case runs.
        self.options = dict(options or {})
        self.core = core
        # An ensemble has more cells than the domain does; a plain case has
        # exactly the domain's active cells.
        self._n_cells = int(dom.n_active if n_cells is None else n_cells)
        # The global file's CONSTANTS (VEG_LAI_WATER_FACTOR), as parameter
        # defaults: VIC applies them to every run of the file, so does this.
        self.constants = gp.constants() if gp is not None and \
            hasattr(gp, 'constants') else {}
        # Sub-daily records behind a daily step, [day, snow_step, cell],
        # when a snow day has to be sub-stepped. None on every other run.
        self.subdaily = None
        self.snow_steps = None
        # The snow-band expansion (snowbands.expand), or None. Set by load
        # when SNOW_BAND TRUE meets a cell with more than one band with
        # area; the batch, atmosphere, soil, mosaic and initial moisture
        # then hold one pseudo-cell per live (cell, band), and run
        # contracts every output back to cells. None -- no expansion, the
        # one-column run -- everywhere else, so single-band answers cannot
        # move.
        self.snowbands = None
        # The cell-level forcing behind the expansion, for the history's
        # forcing echoes (contract_bands; put_data.c reports force->x, not
        # band means).
        self._cell_echo = None
        # Set by load under COMPUTE_TREELINE: the columns' tile cover then
        # differs between the bands of a cell, and the tile-normalised
        # outputs contract with it (contract_bands).
        self.treeline = False

    @property
    def n_steps(self):
        return len(self.dates)

    @property
    def n_cells(self):
        return self._n_cells

    def params(self, **overrides):
        return default_params(self.dtype, self.device,
                              **dict(self.constants, **overrides))

    def kwargs(self, p=None, log_every=0, **extra):
        """Arguments for basin.simulate, with `p` defaulting to the neutral
        parameters. Anything in `extra` is passed through."""
        kw = dict(batch=self.batch, atm=self.atm,
                  p=self.params() if p is None else p,
                  soil_params=self.soil, mosaic=self.mosaic,
                  wind_h=self.wind_h, init_moist=self.init_moist,
                  steps_per_day=self.model_steps,
                  full_energy=self.full_energy, log_every=log_every)
        if self.subdaily is not None:
            kw['subdaily'] = self.subdaily
            kw['snow_steps'] = self.snow_steps
        if self.snowbands is not None:
            # generate_default_state.c:106, via simulate's band_tfactor.
            kw['band_tfactor'] = torch.as_tensor(
                np.asarray(self.snowbands['tfactor']), dtype=self.dtype,
                device=self.device)
        kw.update(self.options)
        kw.update(extra)
        if 'out_vars' not in kw:
            kw['out_vars'] = self.requested_outvars()
        if 'initial_state' not in kw:
            st = self._init_state()
            if st is not None:
                kw['initial_state'] = st
        return kw

    def requested_outvars(self):
        """The OUT_* names the global file's OUTFILE streams ask for (what
        basin.simulate's `out_vars` takes), or None -- no extra row -- when
        it has no streams, or is a VIC 4.2.d file, whose names are not these.
        """
        gp = self.gp
        if gp is None:
            return None
        cached = getattr(self, '_requested_outvars_cache', 0)
        if cached != 0:
            return cached
        names = requested_outvars(gp)
        self._requested_outvars_cache = names
        return names

    def _init_state(self):
        """INIT_STATE: the global file's VIC state file, read once."""
        gp = self.gp
        path = gp.get('INIT_STATE') if gp is not None else None
        if not path or str(path).upper() == 'FALSE':
            return None
        if getattr(self, '_init_state_cache', None) is None:
            if not os.path.isfile(str(path)):
                # VIC stops at the open (vic_restore.c 5.1.0:46-50)
                raise ValueError(
                    'INIT_STATE %s: the global file\'s state file does not '
                    'exist; VIC stops opening it (vic_restore.c:46-50)'
                    % path)
            from .io import state_file
            self._init_state_cache = state_file.initial_state_from_file(
                path, self)
            state_file.check_start(self, self._init_state_cache)
        return self._init_state_cache

    def run(self, p=None, contract=True, **extra):
        """basin.simulate over the whole case. The case's VIC core, when
        it was given one, is selected first (core_version.set_core). A
        snow-band case runs its pseudo-cells and returns cell outputs,
        contracted (contract_bands) unless `contract` is False (the
        coupling connector contracts the columns itself)."""
        if self.core is not None:
            core_version.set_core(self.core)
        try:
            out = basin.simulate(**self.kwargs(p=p, **extra))
        except _tfb.TemperatureSolveError as e:
            raise _tfb.TemperatureSolveError(
                e.step, e.cell, e.tile, e.kinds, self._where(e)) from None
        if self.snowbands is not None and contract:
            out = self.contract_bands(out)
        elif self.snowbands is None:
            out = self._band_axis(out)
        return out

    def _band_axis(self, out):
        """The OUT_*_BAND rows (`band_*`) of a run without snow bands, with
        the band axis VIC writes them with: one band, so [step, cell, 1]."""
        for k in [k for k, v in out.items() if k.startswith('band_')
                  and torch.is_tensor(v) and v.dim() == 2]:
            out[k] = out[k].unsqueeze(-1)
        return out

    def _where(self, e):
        """' (date, lat/lon)' of the cell and step a solve failed at."""
        ex = self.snowbands
        c = int(ex['cell'][e.cell]) if ex is not None else e.cell
        n = int(getattr(self.dom, 'n_active', 0)) or None
        c = c % n if n else c
        what = []
        try:
            what.append(str(self.dates[e.step]))
        except Exception:                                   # noqa: BLE001
            pass
        try:
            what.append('lat %.4f lon %.4f' % (float(self.dom.lat[c]),
                                               float(self.dom.lon[c])))
        except Exception:                                   # noqa: BLE001
            pass
        return ' (%s; active cell %d)' % (', '.join(what), c)

    #: basin.simulate outputs that are not per-column: the discharge is a
    #: basin mean already (each pseudo-cell routed its share of the cell's
    #: area), and the continuation state stays in column space so a split
    #: run's second half meets the columns the first half exported.
    _BAND_CONTRACT_SKIP = frozenset(('discharge_mm', 'state', 'state_aux'))

    #: Outputs VIC averages over the snow-covered fraction rather than
    #: the cell (put_data.c:433-437). Each band reports its own snowy
    #: mean, so the contraction re-weights by the band's snowy cover.
    #: Outputs VIC normalises by the cover of the bare or the vegetated
    #: tiles (put_data.c:424-429), and which tiles.
    _BAND_COVER_MEAN = {'eb_baresoilt': 'bare', 'eb_vegt': 'veg'}

    _BAND_SNOW_MEAN = frozenset(('salbedo', 'snow_surf_temp',
                                 'snow_pack_temp'))

    def contract_bands(self, out):
        """`out` (a snow-band run over pseudo-cells) back on cells.

        Every history is area-weighted over its cell's bands, which is
        vic_run.c:325-330 and put_data.c's `Cv * AreaFract[band]`, with
        the per-tile `Cv` already applied upstream by the mosaic. Three
        outputs are not cell means in VIC and are contracted as VIC
        forms them: the pack albedo and temperatures over the
        snow-covered fraction (put_data.c:433-437), the aerodynamic
        resistance as the inverse of the mean conductance
        (surface_fluxes.c:1023-1025), and the radiative temperature as
        the fourth root of the mean fourth power (put_data.c:456). The
        forcing echoes (air_temp, pressure, ...) are the cell's own
        forcing instead: VIC reports force->x there, not a band mean
        (put_data.c:115-132).
        """
        ex = self.snowbands
        m, nt = len(ex['cell']), len(self.dates)

        def _hist(v):
            return (torch.is_tensor(v) and v.dim() >= 2
                    and v.shape[0] == nt and v.shape[1] == m)

        res = {}
        for k, v in out.items():
            if k in self._BAND_CONTRACT_SKIP or not torch.is_tensor(v):
                res[k] = v
            elif k in self._BAND_SNOW_MEAN and _hist(v) \
                    and _hist(out.get('snow_cover')):
                # Each band's snowy mean, re-weighted by its snowy cover
                # (snow_cover carries the same mask, basin.py:2686-2692).
                cov = out['snow_cover']
                num = snowbands.contract(v * cov, ex, axis=1)
                den = snowbands.contract(cov, ex, axis=1)
                res[k] = torch.where(
                    den > 0.0,
                    num / torch.where(den > 0.0, den,
                                      torch.ones_like(den)), num)
            elif k == 'eb_aero_resist' and _hist(v) \
                    and _hist(out.get('eb_aero_cond')):
                # The inverse of the mean conductance, as basin.py:4494
                # forms it per column.
                res[k] = 1.0 / torch.clamp(
                    snowbands.contract(out['eb_aero_cond'], ex, axis=1),
                    min=1e-30)
            elif k == 'eb_rad_temp' and _hist(v):
                # The fourth root of the mean fourth power, as
                # basin.py:3719-3720 forms it per column.
                res[k] = snowbands.contract(v * v * v * v, ex, axis=1) ** 0.25
            elif k in self._BAND_COVER_MEAN and self.treeline and _hist(v):
                # put_data.c:424-429: OUT_BARESOILT and OUT_VEGT are sums
                # of Cv * AreaFract * TreeAdjust over the bare or the
                # vegetated tiles, divided by that sum. Per column the model
                # has the ratio; the cell's is the ratio of the sums, and
                # the bands differ in cover once a treeline drops tiles.
                cvb = self.mosaic.cv
                bare = self.mosaic.bare_tile
                bare = (torch.zeros_like(cvb, dtype=torch.bool)
                        if bare is None else bare.expand_as(cvb))
                sel = bare if self._BAND_COVER_MEAN[k] == 'bare' else ~bare
                wcol = (cvb * sel.to(cvb.dtype)).sum(1)
                num = snowbands.contract(v * wcol, ex, axis=1)
                den = snowbands.contract(wcol.expand_as(v), ex, axis=1)
                res[k] = torch.where(
                    den > 0.0, num / torch.where(den > 0.0, den,
                                                 torch.ones_like(den)),
                    torch.zeros_like(num))
            elif k.startswith('band_') and _hist(v):
                # put_data.c's OUT_*_BAND: each pseudo-cell is one band of
                # its cell and reports its own Cv-weighted sum, so the
                # columns are placed on the band axis, not area-weighted
                res[k] = snowbands.to_bands(v, ex)
            elif _hist(v):
                res[k] = snowbands.contract(v, ex, axis=1)
            elif v.dim() >= 1 and v.shape[0] == m and not (
                    v.dim() >= 2 and v.shape[0] == nt):
                # A per-column end state ([column] or [column, ...]); a
                # 1-D tensor of length m is read as one (basin.simulate
                # returns no 1-D time series to confuse it with).
                res[k] = snowbands.contract(v, ex, axis=0)
            else:
                res[k] = v
        # put_data.c 5.1.0:471-492: OUT_AERO_RESIST1 and _RESIST2 are the
        # inverse of the cell's (area-summed) conductance, HUGE_RESIST at or
        # below DBL_EPSILON -- not the area mean of each band's resistance
        for rk_, ck_ in (('aero_resist1', 'aero_cond1'),
                         ('aero_resist2', 'aero_cond2')):
            if (rk_ in res and ck_ in res and torch.is_tensor(res[ck_])
                    and res[ck_].shape == res[rk_].shape):
                c_ = res[ck_]
                big_ = c_ > 2.220446049250313e-16
                res[rk_] = torch.where(
                    big_, 1.0 / torch.where(big_, c_, torch.ones_like(c_)),
                    torch.full_like(c_, 1.0e20))
        echo = self._cell_echo or {}
        if 'air_temp' in res and 'tair' in echo:        # basin.py:5199-5212
            res['air_temp'] = echo['tair']
        if 'pressure' in res and 'press' in echo:
            res['pressure'] = echo['press'] / 1000.0
        if 'vp' in res and 'eact_air' in echo:
            res['vp'] = echo['eact_air'] / 1000.0
        if 'vpd' in res and 'vpd' in echo:
            res['vpd'] = echo['vpd'] / 1000.0
        if 'rel_humid' in res and 'eact_air' in echo and 'vpd' in echo:
            res['rel_humid'] = 100.0 * echo['eact_air'] / (
                echo['eact_air'] + echo['vpd'])
        if 'density' in res and 'air_dens' in echo:
            res['density'] = echo['air_dens']
        if 'swdown' in res and 'shortwave' in echo:
            res['swdown'] = echo['shortwave']
        if 'lwdown' in res and 'longwave' in echo:
            res['lwdown'] = echo['longwave']
        if 'wind' in res and 'wind' in echo:
            res['wind'] = echo['wind']
        return res

    def aggregate(self, out, freq='NDAYS 1', how=None, extremes='vic',
                  log=None, convention='vic5'):
        """`out` (one run of this case) aggregated as VIC 5 aggregates its
        history: fluxes summed, states at the period's end, the rest
        averaged. Returns (dict, period start times, dropped tail); see
        torchvic.io.history.aggregate_output. convention='vic4.2d' aggregates
        as VIC 4.2.d's put_data.c does (the times are then 4.2.d's stamps,
        each period's last step)."""
        from .io import history
        return history.aggregate_output(out, self.dates, self.model_steps,
                                        freq=freq, how=how,
                                        extremes=extremes, log=log,
                                        convention=convention)

    def write_history(self, out, result_dir=None, extremes='vic',
                      streams=None, log=None, convention=None,
                      output_global=None, carry=False):
        """The global file's OUTFILE streams for `out`, aggregated over
        each stream's AGGFREQ and written as VIC 5's image driver writes
        them (torchvic.io.history.write_history). Returns the paths.

        convention='vic4.2d' -- the default when the global file, or
        `output_global`, is a VIC 4.2.d one -- writes VIC 4.2.d's per-cell
        ASCII/BINARY files instead (torchvic.io.history.write_history_42d)."""
        from .io import history
        return history.write_history(out, self, result_dir=result_dir,
                                     extremes=extremes, streams=streams,
                                     log=log, convention=convention,
                                     output_global=output_global,
                                     carry=carry)

    def ensemble(self, n_members):
        """The same case with every cell repeated `n_members` times, so one
        run evaluates that many parameter sets at once.

        Layout is member-major: row `m * n_cells + c` is member m's cell c,
        which is what `member_params` builds against and `split_members`
        reads back. Each member routes to its own outlet, so member m's
        discharge is column m.

        This is how the GPU is made to pay. One cell of FindleyLake is a few
        hundred doubles a step: the arithmetic is nothing and the time is all
        in launching the kernels, which is why the CPU beats the GPU there by
        3x (measured, 49 against 151 ms/step). The launches do not multiply
        with the members, so many members cost about what one does and the
        per-member cost falls by that factor.
        """
        import copy
        n = int(n_members)
        if n < 1:
            raise ValueError('n_members must be at least 1')
        if n == 1:
            return self
        nc = self.n_cells
        b = copy.copy(self.batch)
        # The batch rows are pseudo-cells on a snow-band case, cells
        # otherwise; the ensemble repeats rows, not cells.
        ncols = int(b.n_cells)
        b.n_cells = ncols * n
        for name in ('latitude', 'longitude', 'elevation', 'cell_area',
                     'sw_potential', 'flow_distance'):
            v = getattr(b, name, None)
            if isinstance(v, torch.Tensor):
                setattr(b, name, _rep(v, n))
        if isinstance(getattr(b, 'forcing', None), dict):
            b.forcing = {k: (v.repeat(1, n) if v.dim() == 2 else _rep(v, n))
                         for k, v in b.forcing.items()}
        out0 = int(getattr(b, 'n_outlets', 1))
        s2o = b.source2outlet
        member = torch.arange(n, device=s2o.device).repeat_interleave(ncols)
        b.source2outlet = s2o.repeat(n) + member * out0
        b.n_outlets = out0 * n
        atm = {}
        for k, v in self.atm.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                atm[k] = v.repeat((1, n) + (1,) * (v.dim() - 2))
            else:
                atm[k] = _rep(v, n)
        from .core import column
        soil = column.tile_params(self.soil, n)
        mosaic = None if self.mosaic is None else self.mosaic.tile(n)
        ens = Case(self.gp, self.dom, self.par, b, atm, soil, mosaic,
                   self.dates, self.model_steps, self.wind_h,
                   _rep(self.init_moist, n), self.full_energy, self.dtype,
                   self.device, n_cells=nc * n, options=self.options,
                   core=self.core)
        if self.snowbands is not None:
            # Member-major, like everything else here: member k's
            # pseudo-cell j is the ensemble's row k*m + j, over its own
            # member's cell k*nc + ex['cell'][j].
            ex = self.snowbands
            ens.snowbands = dict(
                cell=(np.asarray(ex['cell'])[None, :]
                      + (np.arange(n) * nc)[:, None]).reshape(-1),
                band=np.tile(np.asarray(ex['band']), n),
                tfactor=np.tile(np.asarray(ex['tfactor']), n),
                pfactor=np.tile(np.asarray(ex['pfactor']), n),
                weight=np.tile(np.asarray(ex['weight']), n),
                n_cell=nc * n,
                mean_elev=np.tile(np.asarray(ex['mean_elev']), n))
            ens._cell_echo = {
                k: (v.repeat((1, n) + (1,) * (v.dim() - 2))
                    if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
                for k, v in (self._cell_echo or {}).items()}
        return ens

    #: per-cell soil parameters `member_params` can vary by member. They
    #: are read from `soil_params` (not from `params()`) by basin.simulate.
    SOIL_MEMBER_PARAMS = ('b_infilt', 'Ds', 'Dsmax', 'Ws', 'c')

    def member_params(self, values, n_members=None):
        """Parameters for an ensemble, from per-member values.

        `values` maps a parameter name to a sequence of `n_members` numbers
        (or a tensor of that length); anything not named keeps its default.
        A parameter that varies by cell as well is given as a
        [n_members, n_cells] tensor and is flattened the same way.

        Two kinds of name are accepted. The names `case.params()` lists
        (rmin_scale, ...) are returned in the result, which is what
        `basin.simulate` takes as `p`. The soil parameters in
        `SOIL_MEMBER_PARAMS` (b_infilt, Ds, Dsmax, Ws, c) live in the soil
        tensors, not in `p`: they are written into this (ensemble) case's
        own copy of them, so `case.run(p=p)` and `case.kwargs(p=p)` pick
        them up; call this on the ensemble case, and note that a later
        call replaces the values of an earlier one. They are the values the
        model uses: in the ARNO form (with BASEFLOW NIJSSEN2001 the file's
        values are converted at load, so give the converted ones). Any
        other name raises a ValueError. Gradients flow back to whatever tensor was passed in.
        """
        p = self.params()
        soil_names = [k for k in values if k not in p
                      and k in self.SOIL_MEMBER_PARAMS]
        bad = [k for k in values if k not in p and k not in soil_names]
        if bad:
            raise ValueError(
                'member_params: unknown parameter(s) %s; allowed: %s (p), '
                '%s (soil)' % (sorted(bad), sorted(p),
                               list(self.SOIL_MEMBER_PARAMS)))
        n = n_members
        for v in values.values():
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            n = int(t.shape[0]) if n is None else n
        if n is None:
            raise ValueError('no members to build parameters for')
        nc = self.n_cells // n                 # cells per member

        def flat(k, v, per_member):
            t = (v if isinstance(v, torch.Tensor)
                 else torch.as_tensor(v, dtype=self.dtype, device=self.device))
            t = t.to(dtype=self.dtype, device=self.device)
            if t.dim() == 1:
                if t.shape[0] != n:
                    raise ValueError('%s: %d values for %d members'
                                     % (k, t.shape[0], n))
                return t.repeat_interleave(per_member)
            if t.dim() == 2:
                return t.reshape(-1)
            raise ValueError('%s: expected [members] or [members, cells]' % k)

        for k, v in values.items():
            if k in p:
                p[k] = flat(k, v, nc)
        if soil_names:
            import copy
            soil = copy.copy(self.soil)        # never touch a shared tensor set
            for k in soil_names:
                cur = getattr(soil, k)
                rows = int(cur.shape[0])
                new = flat(k, values[k], rows // n)
                if new.shape != cur.shape:
                    raise ValueError('%s: gives %s values, the soil has %s'
                                     % (k, tuple(new.shape), tuple(cur.shape)))
                setattr(soil, k, new)
            self.soil = soil
        return p

    def split_members(self, x, n_members):
        """A model output over the ensemble's cells, reshaped to expose the
        member axis: [time, member, cell] or [member, cell]."""
        n = int(n_members)
        nc = self.n_cells // n
        if x.dim() == 1 and x.shape[0] == n * nc:
            return x.reshape(n, nc)
        if x.dim() >= 2 and x.shape[1] == n * nc:
            return x.reshape((x.shape[0], n, nc) + tuple(x.shape[2:]))
        raise ValueError('no ensemble axis of %d in shape %s'
                         % (n * nc, tuple(x.shape)))

    def __repr__(self):
        on = [k for k in ('frozen_soil', 'blowing', 'carbon')
              if self.options.get(k)]
        return ('<Case %d cells, %d steps at %d/day, %s, energy %s%s%s>'
                % (self.n_cells, self.n_steps, self.model_steps,
                   self.device, 'on' if self.full_energy else 'off',
                   ''.join(', ' + k for k in on),
                   ', core %s' % self.core if self.core else ''))


#: VIC 5's default for the global file's WIND_H, m (initialize_global.c:58,
#: global_param.wind_h = 10.0). In the IMAGE driver that value reaches only
#: the lake model (vic_run.c:468) and the precipitation gauge correction
#: (vic_run.c:121). The land surface's reference height -- and with it the
#: snow surface's aerodynamic resistance, CalcAerodynamic.c:96-98 -- is
#: each vegetation class's own `wind_h` from the parameter file
#: (vic_init.c:153-160 reads it into veg_lib, vic_run.c:219 takes it,
#: :259-264 makes it ref_height). UW-Hydro's Stehekin image global says as
#: much beside its commented-out WIND_H. basin.simulate's `wind_h`, which
#: this module fills from WIND_H, is ONE height for the snow surface of
#: every no-overstory tile, so where a file's classes differ no single
#: value reproduces VIC; `resolve_wind_h` decides what to do then.
VIC_DEFAULT_WIND_H = 10.0

#: The warmest air temperature at which basin.simulate can make snow: the
#: top of the range t_max_snow is allowed to take.
#: Precipitation is partitioned on the step's air temperature
#: (basin.partition_precipitation) and a pack forms only from snowfall, so
#: with every step at or above this the snow surface's height is never
#: read.
SNOW_TAIR_GUARD = 3.0


def surface_reference_heights(cv, wind_h, overstory=None, displacement=None,
                              veg_rough=None, soil_rough=None):
    """The reference heights, m, VIC 5's image driver gives the snow surface
    of every no-overstory tile present (cover > 0): a sorted list of the
    distinct values, or None when the file has no wind_h.

    vic_run.c:259-264: the class's wind_h where its displacement is below
    it, else displacement + wind_h + roughness (the soil's where the class
    has none, :245-247), month by month. Overstory tiles take the canopy
    profile's snow entries instead (CalcAerodynamic.c:50-112), which
    basin.simulate computes per tile from the same wind_h.

    cv, wind_h, overstory [veg, cell]; displacement, veg_rough
    [veg, month, cell]; soil_rough [cell].
    """
    if wind_h is None or cv is None:
        return None
    cv = np.asarray(cv, np.float64)
    wh = np.broadcast_to(np.asarray(wind_h, np.float64), cv.shape)
    ov = (np.zeros(cv.shape) if overstory is None else
          np.broadcast_to(np.asarray(overstory, np.float64), cv.shape))
    present = np.isfinite(cv) & (cv > 0) & ~(ov > 0)
    if displacement is None:
        h = wh[present]
    else:
        d = np.asarray(displacement, np.float64)
        w3 = np.broadcast_to(wh[:, None, :], d.shape)
        z0 = (np.zeros(d.shape) if veg_rough is None
              else np.asarray(veg_rough, np.float64))
        if soil_rough is not None:
            z0 = np.where(z0 > 0, z0, np.broadcast_to(
                np.asarray(soil_rough, np.float64), d.shape[-1:]))
        with np.errstate(invalid='ignore'):
            h3 = np.where(d < w3, w3, d + w3 + z0)
        h = h3[np.broadcast_to(present[:, None, :], d.shape)]
    h = h[np.isfinite(h)]
    return sorted(set(np.round(h, 6).tolist()))


def params_reference_heights(par):
    """surface_reference_heights of an ImageParams."""
    v = par.veg
    return surface_reference_heights(
        v.get('Cv'), v.get('wind_h'), v.get('overstory'),
        v.get('displacement'), v.get('veg_rough'), par.soil.get('rough'))


def resolve_wind_h(gp, heights, min_tair=None, mosaic=True, log=None):
    """(value, source, info): basin.simulate's `wind_h` for a case.

    1. WIND_H in the global file: that value, as always. When the
       parameter file's heights differ, the log says VIC 5's image driver
       would use those for the land surface.
    2. Absent, and the parameter file gives every no-overstory tile one
       reference height: that height, which is exactly VIC's for those
       tiles (Walnut Gulch MOD_IGBP: 5 m on every class).
    3. Absent, the heights differ (India and Stehekin: 2, 3 and 5 m), and
       no step of the case is below SNOW_TAIR_GUARD: the value is never
       read -- measured on basin 37, 2001-2003, 2 m and 10 m gave identical
       ET, runoff, baseflow, SWE and soil moisture -- so VIC's 10 m default
       is recorded as inert.
    4. Otherwise refused: snow is possible and no single height is VIC's.
       Declare WIND_H. The loader used to fall back to 10 m here, and before
       that to 2 m; which of the two the snow surface got depended only on
       the code's age.

    `heights` is surface_reference_heights' list (None: no wind_h in the
    file). `min_tair` is the lowest air temperature the case steps on, C.
    With no mosaic (`no_veg`) the height also sets the bare surface's
    evaporation resistance, so it is never inert.
    """
    say = log or (lambda *a: None)
    info = dict(vic_surface_heights=heights,
                min_air_temp=(None if min_tair is None
                              else round(float(min_tair), 3)),
                snow_guard=SNOW_TAIR_GUARD, inert=False)
    w = gp.get('WIND_H')
    if w is not None:
        if heights and sorted(set(heights)) != [float(w)]:
            say('WIND_H %g m from the global file; VIC 5\'s image driver would '
                'give the land surface the parameter file\'s heights %s '
                '(vic_run.c:219, 259-264) and use WIND_H only for lakes and '
                'precipitation correction' % (float(w), heights))
        return float(w), 'global file', info
    if heights and len(heights) == 1:
        say('WIND_H absent: %g m, the one reference height the parameter '
            'file gives every no-overstory tile (vic_run.c:219, 259-264)'
            % heights[0])
        return float(heights[0]), ('parameter file: the one reference height '
                                   'of every no-overstory tile present '
                                   '(vic_run.c:219, 259-264)'), info
    cold = min_tair is None or float(min_tair) < SNOW_TAIR_GUARD
    if mosaic and not cold:
        info['inert'] = True
        say('WIND_H absent and the parameter file\'s heights differ (%s); '
            'the case\'s air is never below %g C (min %.2f C), so no snow '
            'forms and the height is not read: VIC\'s default %g m recorded'
            % (heights, SNOW_TAIR_GUARD, float(min_tair), VIC_DEFAULT_WIND_H))
        return VIC_DEFAULT_WIND_H, ('inert: no snow possible (air never '
                                    'below %g C); VIC\'s default, '
                                    'initialize_global.c:58'
                                    % SNOW_TAIR_GUARD), info
    raise ValueError(
        'WIND_H is not in the global file, and no single wind height is '
        'VIC\'s here: VIC 5\'s image driver gives each no-overstory tile its '
        'class\'s own reference height from the parameter file (%s m; '
        'vic_run.c:219, 259-264), while basin.simulate gives the snow surface '
        'one. %s. Declare WIND_H (the forcing wind\'s measurement height) in '
        'the global file.'
        % (heights if heights else 'the file has no wind_h',
           ('the case has no vegetation mosaic, so the height sets every '
            'cell\'s evaporation resistance') if not mosaic else
           ('Snow is possible: the air reaches %.2f C, below %g C, the top '
            'of t_max_snow\'s range' % (float(min_tair), SNOW_TAIR_GUARD)
            if min_tair is not None else 'The air temperature is unknown')))

#: Days of daily forcing read on each side of a window before MTCLIM runs
#: over it. MTCLIM smooths the daily temperature range with a 30-day
#: trailing mean (mtclim.trailing_mean) and, without a supplied vapour pressure,
#: a 90-day precipitation total (mtclim.srad_humidity); a window with no
#: margin had shortwave off by up to 10.8 W/m2 in its first 30 days
#: (50 India cells, 2003, against a 200-day margin).
MTCLIM_MARGIN_DAYS = 90


def daily_by_off_gmt(fields, times, off_gmt, log=None, with_subdaily=False):
    """Sub-daily forcing to the LOCAL day of each cell, grouped by off_gmt.

    `aggregate_to_daily` bins every cell on one clock, and this function used
    to pass it none, which is the UTC day. The parameter file's off_gmt says
    where each cell's day starts, and one domain can hold several: basin 37
    has 8 distinct values, India 212. A record at UTC time t goes to local
    day floor((t + off_gmt) / 1 day), and on a grid of step s two offsets
    put every record on the same day exactly when floor(off_gmt / s) is the
    same, so that is the grouping: India's 4.41-6.52 h are three groups for
    an hourly forcing, not 212, and each group is binned with its first
    cell's own off_gmt, which is the binning every member would get.

    Only COMPLETE local days are returned: a day is kept when every group
    has all of its records in it (86400 s / the step). A forcing file runs
    on UTC days, so on a local clock its first and last days are partial:
    Walnut Gulch (-7 h, 3-hourly) has 3 of 8 records on 2007-12-31 and 5 of
    8 on 2008-12-31. Averaging them gave a 367-day record whose day 0 was
    one evening -- tair 0.69 C against 8.00 C the next day -- and moved a
    positional study observation by a day (review probe wg_edge.py);
    such days are never part of a window. Here
    they are dropped, and each dropped day is logged with its record
    count. With one group the kept days are
    aggregate_to_daily(utc_offset=off_gmt)'s own values on those days.
    """
    say = log if log is not None else (lambda *a: None)
    step_s = ImageForcing.step_seconds(times) or 3600.0
    per = int(round(86400.0 / step_s))
    off = np.asarray(off_gmt, dtype=np.float64)
    # the record grid's phase within the step: records at phase + k * step
    try:
        t0 = np.asarray(times)[:1].astype('datetime64[s]').astype(np.int64)
        phase = float(t0[0] % int(step_s)) if t0.size else 0.0
    except (TypeError, ValueError):
        # cftime under a calendar numpy has no dtype for
        x = np.asarray(times).flat[0]
        phase = float((x.hour * 3600 + x.minute * 60 + x.second)
                      % int(step_s))
    key = np.floor((off * 3600.0 + phase) / step_s + 1e-9).astype(np.int64)
    groups = np.unique(key)
    dev = next(iter(fields.values())).device
    # a record count per day beside the fields, summed as aggregate_by_off_gmt
    # counts coverage; the fields' own values do not depend on it
    ones = torch.ones(len(times), 1, dtype=torch.float64, device=dev)
    if 'tair' in fields and 'vp' in fields:
        # vic_force.c:395-414 pulls each sub-daily vp back to svp(Tair)
        # BEFORE the NR average (:452), so the daily vp is the mean of the
        # clamped values. Averaging the raw vp and clamping the mean
        # instead differs wherever a sub-step is near saturation: with
        # MTCLIM-saturated forcing three cold sub-steps clamp ~1e-5 Pa and
        # day-0 cell26 vp was off by 4.2e-9 kPa. The clamp is idempotent,
        # so the sub-daily pack below -- re-clamped per step by
        # build_atmosphere -- is unchanged.
        _es = _svp(fields['tair'])
        fields = dict(fields, vp=torch.where(_es - fields['vp'] < 0.0, _es,
                                             fields['vp']))
    parts, counts, complete = [], [], None
    # The records each group averaged, so a snow day can be replayed at
    # hidx 0..NF-1 (surface_fluxes.c:258-269) on the same days.
    raw_groups = []
    for g in groups:
        idx = np.nonzero(key == g)[0]
        if groups.size == 1:
            f = dict(fields)
        else:
            ti = torch.as_tensor(idx, device=dev)
            f = {k: v.index_select(1, ti) for k, v in fields.items()}
        f['_n'] = ones
        if with_subdaily:
            raw_groups.append((idx, f, float(off[idx[0]])))
        day, dates = aggregate_to_daily(f, times, how={'_n': 'sum'},
                                        utc_offset=float(off[idx[0]]),
                                        log=lambda *a: None)
        n_rec = dict(zip(dates, np.rint(day.pop('_n')[:, 0].cpu().numpy())
                         .astype(np.int64)))
        full = {d for d, c in n_rec.items() if c == per}
        complete = full if complete is None else complete & full
        parts.append((idx, day, dates))
        counts.append((float(off[idx[0]]), n_rec))
    keep = sorted(complete)
    dropped = sorted(set(d for _, n_rec in counts for d in n_rec)
                     - set(keep))
    say('daily binning on the local day: %d off_gmt group(s) (%s h), %d '
        'complete days %s .. %s'
        % (groups.size, ', '.join('%+g' % o for o, _ in counts), len(keep),
           keep[0] if keep else '-', keep[-1] if keep else '-'))
    for d in dropped:
        say('  dropped %s: not a complete local day (%s)'
            % (d, '; '.join('%d of %d records at off_gmt %+g h'
                            % (n_rec.get(d, 0), per, o)
                            for o, n_rec in counts
                            if n_rec.get(d, 0) != per)))
    if not keep:
        raise ValueError('no complete local day in the forcing window: every '
                         'day lacks records in some off_gmt group')
    n = off.size
    out = {}
    for idx, day, dates in parts:
        pos = torch.as_tensor([dates.index(d) for d in keep],
                              device=next(iter(day.values())).device)
        if groups.size == 1:
            out = {k: v.index_select(0, pos) for k, v in day.items()}
            continue
        ti = torch.as_tensor(idx, device=pos.device)
        for k, v in day.items():
            if k not in out:
                out[k] = torch.zeros((len(keep), n) + tuple(v.shape[2:]),
                                     dtype=v.dtype, device=v.device)
            out[k][:, ti] = v.index_select(0, pos)
    if not with_subdaily:
        return out, keep
    # Same days, in time order within the day: force->air_temp[hidx].
    sub = {}
    dev = next(iter(fields.values())).device
    for idx, f, off_h in raw_groups:
        loc = np.asarray(local_days(times, off_h))
        stacks = {}
        for d in keep:
            sel_np = np.flatnonzero(loc == str(d)[:10])
            if sel_np.size != per:
                raise ValueError('sub-daily pack: %s has %d records at '
                                 'off_gmt %+g h, and the forcing step has %d'
                                 % (d, int(sel_np.size), off_h, per))
            sel = torch.tensor(sel_np, dtype=torch.long, device=dev)
            for k, v in f.items():
                if k == '_n' or not torch.is_tensor(v):
                    continue
                stacks.setdefault(k, []).append(v.index_select(0, sel))
        packed = {k: torch.stack(v, 0) for k, v in stacks.items()}
        if groups.size == 1:
            return out, keep, packed
        ti = torch.tensor(idx, dtype=torch.long, device=dev)
        for k, v in packed.items():
            if k not in sub:
                sub[k] = torch.zeros((len(keep), per, n) + tuple(v.shape[3:]),
                                     dtype=v.dtype, device=v.device)
            sub[k][:, :, ti] = v
    return out, keep, sub


def image_options(gp, log=None, no_energy=False):
    """basin.simulate's physics switches from a VIC 5 global file:
    {frozen_soil, blowing, blowing_options, carbon, carbon_options,
    share_layer_moist, aero_resist_cansnow, grnd_flux_type, close_energy,
    spatial_snow, snow_density, tfallback, n_frost, quick_flux, implicit,
    exp_trans, noflux, nnodes, quick_solve}.

    Each takes VIC's default where the file is silent
    (initialize_options.c:23-70). Every option of the file is first
    checked against torchvic.io.options.IMAGE_OPTIONS: a value TorchVIC
    does not run as VIC does raises ValueError naming every such option
    and value at once; an option read to no effect, and a key that is not
    an image-driver option, are announced through `log`. The checks that
    need the parameter and forcing files are load's (check_inputs)."""
    say = log or (lambda *a: None)
    refused, announced = _opt.check(gp, no_energy=no_energy)
    if refused:
        raise ValueError('%s selects what TorchVIC does not run as VIC does '
                         '(docs/OPTIONS.md):\n  %s'
                         % (gp.path, '\n  '.join(refused)))
    blow_opt = {}
    for key, name in (('BLOWING_VAR_THRESHOLD', 'var_threshold'),
                      ('BLOWING_CALC_PROB', 'calc_prob'),
                      ('BLOWING_SIMPLE', 'simple'),
                      ('BLOWING_FETCH', 'fetch'),
                      ('BLOWING_SPATIAL_WIND', 'spatial_wind')):
        if gp.get(key) is not None:
            blow_opt[name] = bool(gp.get_bool(key))
    share = gp.get('SHARE_LAYER_MOIST')
    carbon = bool(gp.get_bool('CARBON'))
    carbon_opt = {}
    if gp.get_int('CANOPY_LAYERS'):
        carbon_opt['ncanopy'] = gp.get_int('CANOPY_LAYERS')
    if carbon and str(gp.get('RC_MODE') or '').upper() == 'RC_PHOTO':
        # get_global_param.c:849-853 resets it without CARBON
        carbon_opt['rc_mode'] = 'RC_PHOTO'
    _th = _opt.thermal(gp)
    opts = dict(frozen_soil=bool(_th['FROZEN_SOIL']),
                blowing=bool(gp.get_bool('BLOWING')),
                blowing_options=blow_opt, carbon=carbon,
                carbon_options=carbon_opt,
                share_layer_moist=(True if share is None
                                   else bool(gp.get_bool('SHARE_LAYER_MOIST'))),
                aero_resist_cansnow=str(gp.get('AERO_RESIST_CANSNOW')
                                        or 'AR_406_FULL').upper(),
                grnd_flux_type=str(gp.get('GRND_FLUX_TYPE')
                                   or 'GF_410').upper(),
                n_frost=_opt.spatial_frost_n(gp),
                close_energy=bool(gp.get_bool('CLOSE_ENERGY')),
                spatial_snow=bool(gp.get_bool('SPATIAL_SNOW')),
                snow_density=str(gp.get('SNOW_DENSITY')
                                 or 'DENS_BRAS').upper(),
                # initialize_options.c:51
                tfallback=bool(gp.get_bool('TFALLBACK', True)),
                quick_flux=bool(_th['QUICK_FLUX']),
                implicit=bool(_th['IMPLICIT']),
                exp_trans=bool(_th['EXP_TRANS']),
                noflux=bool(_th['NOFLUX']),
                nnodes=(3 if _th['QUICK_FLUX'] else _th['NODES']),
                quick_solve=bool(_th['QUICK_SOLVE']))
    if gp.get_bool('CORRPREC'):
        # vic_run.c:106-114 takes the global file's WIND_H (10 m by default,
        # initialize_global.c:58) and param.GAUGE_HEIGHT (1 m)
        opts['corrprec'] = dict(
            wind_h=float(gp.get('WIND_H') or VIC_DEFAULT_WIND_H),
            gauge_height=1.0)
    if not opts['tfallback']:
        say('options: TFALLBACK FALSE (stops at the first failed '
            'temperature solve, as VIC does)')
    say('options: FROZEN_SOIL %s, BLOWING %s, SPATIAL_SNOW %s, CARBON %s, '
        'SHARE_LAYER_MOIST %s'
        % tuple('TRUE' if opts[k] else 'FALSE' for k in
                ('frozen_soil', 'blowing', 'spatial_snow', 'carbon',
                 'share_layer_moist'))
        + ', AERO_RESIST_CANSNOW %s, GRND_FLUX_TYPE %s, CLOSE_ENERGY %s'
        % (opts['aero_resist_cansnow'], opts['grnd_flux_type'],
           'TRUE' if opts['close_energy'] else 'FALSE')
        + (', SPATIAL_FROST %d areas' % opts['n_frost']
           if opts['n_frost'] > 1 else '')
        + ', SNOW_DENSITY %s' % opts['snow_density']
        + ('' if opts['quick_flux']
           else ', QUICK_FLUX FALSE (IMPLICIT %s, EXP_TRANS %s, NOFLUX %s, '
                'NODES %d%s)' % ('TRUE' if opts['implicit'] else 'FALSE',
                                 'TRUE' if opts['exp_trans'] else 'FALSE',
                                 'TRUE' if opts['noflux'] else 'FALSE',
                                 opts['nnodes'],
                                 ', QUICK_SOLVE TRUE'
                                 if opts['quick_solve'] else '')))
    for line in announced:
        say('  ' + line)
    return opts


def _classic_global_as_image(gp, copts):
    """A classic global file's SNOW_BAND "<n> <file>" as the TRUE/FALSE the
    image options table knows; the band count and file are the classic
    reader's (ClassicOptions), already read."""
    val = ['TRUE' if copts.snow_bands > 1 else 'FALSE']
    gp.raw['SNOW_BAND'] = [val]
    gp.lines = [(k, val if k == 'SNOW_BAND' else r) for k, r in gp.lines]


def check_inputs(gp, par):
    """The option checks that need the parameter file: a list of refusals.

    SNOW_BAND TRUE runs every band with area as its own column
    (core/snowbands.py) and is refused only where the file cannot run as
    VIC runs it: no AreaFract, more than one band with area but no
    Pfactor or band elevation, or a layout VIC halts on (vic_init.c:1016,
    :1049, :1058-1059). One band with area needs no expansion: VIC
    normalises its Pfactor to exactly 1 (:1071-1080) and lapses from its
    own elevation (Tfactor 0), so the one-column run already is the
    band's. FCAN_SRC FROM_VEGLIB/FROM_VEGPARAM needs the file's fcanopy
    (vic_init.c:270-281 stops without it); BLOWING TRUE needs
    sigma_slope, lag_one and fetch, within VIC's bounds on every tile
    with cover (vic_init.c:1281-1330); SPATIAL_SNOW TRUE needs
    max_snow_distrib_slope, non-negative on every active cell
    (vic_init.c:691-700 reads it, :799-802 stops on a negative)."""
    out = []
    if _opt.treeline(gp)[0] and getattr(par, 'july_tavg', None) is None:
        out.append('COMPUTE_TREELINE: the parameter file has no July_Tavg '
                   '(the soil file\'s last column under JULY_TAVG_SUPPLIED, '
                   'read_soilparam.c:574-586), which the treeline lapses')
    if gp.get_bool('SNOW_BAND'):
        bands = par.bands
        if 'AreaFract' not in bands:
            out.append('SNOW_BAND TRUE: the parameter file has no AreaFract, '
                       'which VIC reads for it (vic_init.c:957-1000)')
        else:
            af = np.asarray(bands['AreaFract'], np.float64)
            n = (af > 0).sum(0)
            if (n > 1).any():
                multi = ('SNOW_BAND TRUE: %d of %d cells have more than one '
                         'elevation band with area (up to %d)'
                         % (int((n > 1).sum()), n.size, int(n.max())))
                if 'Pfactor' not in bands:
                    out.append(multi + ', and the parameter file has no '
                               'Pfactor, which VIC reads for every band '
                               '(vic_init.c:984-992)')
                elif 'elevation' not in bands:
                    out.append(multi + ', and the parameter file has no band '
                               'elevation, which VIC lapses the bands from '
                               '(vic_init.c:974-983)')
                else:
                    try:
                        snowbands.band_factors(bands['AreaFract'],
                                               bands['elevation'],
                                               bands['Pfactor'])
                    except ValueError as e:
                        out.append('SNOW_BAND TRUE: %s' % e)
    fcan = (gp.get('FCAN_SRC') or 'FROM_DEFAULT').upper()
    if fcan in ('FROM_VEGLIB', 'FROM_VEGPARAM') and 'fcanopy' not in par.veg:
        out.append('FCAN_SRC %s: the parameter file has no fcanopy, and VIC '
                   'stops reading it (vic_init.c:270-281). FCAN_SRC '
                   'FROM_DEFAULT runs fcanopy 1' % fcan)
    if gp.get_bool('SPATIAL_FROST'):
        if 'frost_slope' not in par.soil:
            out.append('SPATIAL_FROST TRUE: the parameter file has no '
                       'frost_slope, which VIC reads for it '
                       '(vic_init.c:703-710)')
        else:
            bad = np.asarray(par.soil['frost_slope'], np.float64) < 0.0
            if bad.any():
                out.append('SPATIAL_FROST TRUE: frost_slope must be >= 0 '
                           'and is negative on %d cell(s) (VIC stops, '
                           'vic_init.c:808-811)' % int(bad.sum()))
    if gp.get_bool('BLOWING'):
        miss = [k for k in ('sigma_slope', 'lag_one', 'fetch')
                if k not in par.veg]
        if miss:
            out.append('BLOWING TRUE: the parameter file has no %s, which VIC '
                       'reads for it (vic_init.c:1281-1330)'
                       % ', '.join(miss))
        else:
            cv = np.asarray(par.veg.get('Cv'), np.float64)
            on = np.isfinite(cv) & (cv > 0)
            for k, lo in (('sigma_slope', 0.0), ('lag_one', 0.0),
                          ('fetch', 1.0)):
                v = np.asarray(par.veg[k], np.float64)
                bad = on & ~(v > lo)
                if bad.any():
                    out.append('BLOWING TRUE: %s must be > %g on every tile '
                               'with cover and is not on %d (VIC stops, '
                               'vic_init.c:1281-1330)'
                               % (k, lo, int(bad.sum())))
    if gp.get_bool('SPATIAL_SNOW'):
        if 'max_snow_distrib_slope' not in par.soil:
            out.append('SPATIAL_SNOW TRUE: the parameter file has no '
                       'max_snow_distrib_slope, which VIC reads for it '
                       '(vic_init.c:691-700)')
        else:
            v = np.asarray(par.soil['max_snow_distrib_slope'], np.float64)
            bad = (v < 0.0) | ~np.isfinite(v)
            if bad.any():
                out.append('SPATIAL_SNOW TRUE: max_snow_distrib_slope must '
                           'be >= 0 on every active cell and is not on %d '
                           '(VIC stops, vic_init.c:799-802)'
                           % int(bad.sum()))
    return out


def _stamp(times, when):
    """`when` (a datetime) in the type of the forcing's time stamps."""
    t0 = np.asarray(times).flat[0]
    if isinstance(t0, np.datetime64):
        return np.datetime64(when)
    return type(t0)(when.year, when.month, when.day, when.hour,
                    when.minute, when.second, calendar=t0.calendar)


def _forcing_calendar(times):
    t0 = np.asarray(times).flat[0]
    if isinstance(t0, np.datetime64):
        return 'STANDARD'
    return str(getattr(t0, 'calendar', 'standard')).upper()


_CAL_FAMILY = {'STANDARD': 'G', 'GREGORIAN': 'G', 'PROLEPTIC_GREGORIAN': 'G',
               'NOLEAP': 'N', '365_DAY': 'N', 'ALL_LEAP': 'L', '366_DAY': 'L',
               '360_DAY': '360', 'JULIAN': 'J'}


def window(gp, times, model_steps_per_day):
    """The forcing records VIC runs: [a, b) from the global file's
    STARTYEAR..STARTSEC to the end of ENDYEAR/ENDMONTH/ENDDAY, or NRECS
    model steps (get_global_param.c:686-768). Raises ValueError where VIC
    stops: a start or end not given, or a forcing that does not cover the
    window."""
    import datetime as dt
    sy, sm, sd = (gp.get_int('STARTYEAR'), gp.get_int('STARTMONTH'),
                  gp.get_int('STARTDAY'))
    if not (sy and sm and sd):
        raise ValueError('%s: STARTYEAR, STARTMONTH and STARTDAY define the '
                         'simulation start, and VIC stops without them '
                         '(get_global_param.c:686-702)' % gp.path)
    ssec = 0 if model_steps_per_day == 1 else (gp.get_int('STARTSEC') or 0)
    start = dt.datetime(sy, sm, sd) + dt.timedelta(seconds=ssec)
    nrecs = gp.get_int('NRECS') or 0
    if nrecs > 0:
        end = start + dt.timedelta(seconds=nrecs * 86400.0
                                   / model_steps_per_day)
        what = 'NRECS %d' % nrecs
    else:
        ey, em, ed = (gp.get_int('ENDYEAR'), gp.get_int('ENDMONTH'),
                      gp.get_int('ENDDAY'))
        if not (ey and em and ed):
            raise ValueError('%s: give ENDYEAR, ENDMONTH and ENDDAY, or NRECS; '
                             'VIC stops without them '
                             '(get_global_param.c:718-723)' % gp.path)
        end = dt.datetime(ey, em, ed) + dt.timedelta(days=1)
        what = 'end of %04d-%02d-%02d' % (ey, em, ed)
    if end <= start:
        raise ValueError('%s: the end (%s) is not after the start (%s)'
                         % (gp.path, end, start))
    t = np.asarray(times)
    step = ImageForcing.step_seconds(times) or 86400.0
    a = int(np.searchsorted(t, _stamp(times, start), side='left'))
    b = int(np.searchsorted(t, _stamp(times, end), side='left'))
    last_end = t[-1] + (np.timedelta64(int(step), 's')
                        if isinstance(t[-1], np.datetime64)
                        else dt.timedelta(seconds=step))
    if not (t[0] <= _stamp(times, start)) or last_end < _stamp(times, end) \
            or b <= a:
        raise ValueError('%s: the simulation window %s .. %s (%s) is not '
                         'covered by the forcing, which runs %s .. %s; VIC '
                         'stops on this too'
                         % (gp.path, start, end, what, str(t[0])[:19],
                            str(last_end)[:19]))
    return a, b


def _snow_possible(fields, a=0, b=None, max_snow_temp=0.5):
    """VIC's will_it_snow over records [a, b): a record colder than
    MAX_SNOW_TEMP with precipitation (forcing_utils.c:86-100)."""
    tair, prcp = fields['tair'][a:b], fields['prcp'][a:b]
    return bool(((tair < max_snow_temp) & (prcp > 0)).any())


def _gather_soil(sp, idx):
    """A SoilParams over pseudo-cells: every per-cell tensor gathered by
    `idx` (column.tile_params repeats instead; the layouts differ)."""
    def g(v):
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return v.index_select(0, idx)
        if isinstance(v, dict):
            return {k: g(x) for k, x in v.items()}
        return v
    out = object.__new__(type(sp))
    for name in type(sp).__slots__:
        if hasattr(sp, name):
            setattr(out, name, g(getattr(sp, name)))
    return out


def snow_expansion(par, is_classic=False):
    """The snow-band expansion (snowbands.expand) of `par`'s bands, or None
    when no cell has more than one live band (then the run is the
    one-column run bit for bit). The caller has checked SNOW_BAND and the
    band arrays; check_inputs refused the layouts VIC halts on."""
    live = (np.asarray(par.bands['AreaFract']) > 0).sum(0)
    if not (live > 1).any():
        return None
    ex = snowbands.expand(
        par.bands['AreaFract'], par.bands['elevation'],
        par.bands['Pfactor'],
        cell_elev=par.soil['elev'] if is_classic else None)
    ex['n_band'] = int(np.asarray(par.bands['AreaFract']).shape[0])
    return ex


def _expand_bands(batch, atm, soil, mosaic, init_moist, subdaily, ex,
                  dtype, device):
    """(batch, atm, soil, mosaic, init_moist, subdaily, cell_echo), each
    over the expansion's pseudo-cells.

    VIC runs one column per (cell, band) with area, shifting only the air
    temperature and the precipitation per band (surface_fluxes.c:358-359:
    Tair = air_temp + Tfactor[band], step_prec = prec * Pfactor[band])
    and weighting every output by Cv * AreaFract (vic_run.c:326-330,
    put_data.c). The expansion runs those columns through the unchanged
    physics: the atmosphere's tair is shifted by Tfactor and its prcp
    scaled by Pfactor per pseudo-cell, and Case.contract_bands weights
    the outputs back. Everything else the column sees -- the soil, the
    mosaic, the pressure, density, vapour deficit and radiation -- is the
    cell's, as VIC's is (force->x is per cell, and vic_force.c computes
    the density from the unlapsed air temperature). The one derived
    quantity that follows the band is the latent heat: VIC computes Le
    from the lapsed Tair in every band (solve_snow.c:130,
    penman.c:190), so lv is recomputed from the shifted tair.

    The pseudo-cells share the cell's elevation, VIC's band mean
    (vic_init.c:1020-1034), and split its area by AreaFract, so the
    routing -- a linear convolution followed by an area-weighted basin
    mean -- delivers the contracted runoff without a separate step.
    """
    soil2, mosaic2, init2 = expand_bands_static(soil, mosaic, init_moist,
                                                ex, device)
    b, atm2, sub2, cell_echo = expand_bands_forcing(batch, atm, subdaily, ex,
                                                    dtype, device)
    return b, atm2, soil2, mosaic2, init2, sub2, cell_echo


def expand_bands_static(soil, mosaic, init_moist, ex, device):
    """(soil, mosaic, init_moist) over the expansion's pseudo-cells; the
    part of _expand_bands that does not change from window to window."""
    idx = torch.as_tensor(np.asarray(ex['cell']), dtype=torch.long,
                          device=device)
    return (_gather_soil(soil, idx),
            None if mosaic is None else mosaic.select(idx),
            init_moist.index_select(0, idx))


def expand_bands_forcing(batch, atm, subdaily, ex, dtype, device):
    """(batch, atm, subdaily, cell_echo) over the expansion's pseudo-cells;
    the part of _expand_bands that follows the forcing window."""
    import copy
    idx = torch.as_tensor(np.asarray(ex['cell']), dtype=torch.long,
                          device=device)
    n = int(ex['n_cell'])
    tf = torch.as_tensor(np.asarray(ex['tfactor'], dtype=np.float64),
                         dtype=dtype, device=device)
    pf = torch.as_tensor(np.asarray(ex['pfactor'], dtype=np.float64),
                         dtype=dtype, device=device)
    wt = torch.as_tensor(np.asarray(ex['weight'], dtype=np.float64),
                         dtype=dtype, device=device)
    b = copy.copy(batch)
    b.n_cells = len(ex['cell'])
    b.latitude = batch.latitude.index_select(0, idx)
    b.longitude = batch.longitude.index_select(0, idx)
    b.elevation = torch.as_tensor(np.asarray(ex['mean_elev'],
                                             dtype=np.float64),
                                  dtype=dtype,
                                  device=device).index_select(0, idx)
    b.cell_area = batch.cell_area.index_select(0, idx) * wt
    if getattr(batch, 'flow_distance', None) is not None:
        b.flow_distance = batch.flow_distance.index_select(0, idx)
    b.source2outlet = batch.source2outlet.index_select(0, idx)
    b.cell_index = torch.arange(b.n_cells, device=device)
    # The cell-level forcing, kept for the history's forcing echoes
    # (contract_bands; put_data.c:115-132 reports force->x).
    cell_echo = {k: atm[k] for k in ('tair', 'press', 'eact_air', 'vpd',
                                    'air_dens', 'shortwave', 'longwave',
                                    'wind') if k in atm}
    atm2 = {k: (v.index_select(1, idx)
                if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == n
                else v)
            for k, v in atm.items()}
    atm2['tair'] = atm2['tair'] + tf
    atm2['prcp'] = atm2['prcp'] * pf
    atm2['lv'] = atmos.latent_heat_of_vaporization(atm2['tair'])
    for k in ('tmax', 'tmin'):
        v = atm2.get(k)
        if torch.is_tensor(v) and v.shape == atm2['tair'].shape:
            atm2[k] = v + tf
    if subdaily is None:
        sub2 = None
    else:
        sub2 = {k: (v.index_select(2, idx)
                    if torch.is_tensor(v) and v.dim() == 3 and v.shape[2] == n
                    else v)
                for k, v in subdaily.items()}
        sub2['tair'] = sub2['tair'] + tf
        sub2['prcp'] = sub2['prcp'] * pf
        if 'lv' in sub2:
            sub2['lv'] = atmos.latent_heat_of_vaporization(sub2['tair'])
        for k in ('tmax', 'tmin'):
            v = sub2.get(k)
            if torch.is_tensor(v) and v.shape == sub2['tair'].shape:
                sub2[k] = v + tf
    return b, atm2, sub2, cell_echo


class ForceCase(object):
    """A case under OUTPUT_FORCE TRUE: the forcing as VIC 4.2.d's
    initialize_atmos leaves it, ready for write_forcing_file.c's files.

    There is nothing to simulate -- VIC 4.2.d skips the model (vicNl.c:178-376)
    and writes `atmos` -- so `run` says so. `fields` is {name: [step, cell]} at
    the snow step (SNOW_STEPS_PER_DAY a day), over the global file's window."""

    def __init__(self, gp, dom, par, fields, times, model_steps, snow_steps,
                 elevation, dtype, device):
        self.gp, self.dom, self.par = gp, dom, par
        self.fields, self.times = fields, times
        self.model_steps, self.snow_steps = model_steps, snow_steps
        self.elevation, self.dtype, self.device = elevation, dtype, device
        import pandas as _pd
        self.dates = [_pd.Timestamp(x).to_pydatetime() for x in times]
        self.n_steps = len(self.dates)
        self.n_cells = int(dom.n_active)

    def run(self, *a, **k):
        raise ValueError('OUTPUT_FORCE TRUE: VIC 4.2.d writes its forcing and '
                         'skips the model (vicNl.c:178-376); so does '
                         'TorchVIC -- use write_forcing()')

    def spec(self, dt_hours=None, log=None):
        """The output spec (history.Spec42d) of the global file."""
        from .io import history
        return history.parse_output_42d(
            self.gp, dt=dt_hours or 24.0 / self.model_steps, log=log,
            skip_vic5_streams=True)

    def write_forcing(self, result_dir=None, log=None):
        """Write VIC 4.2.d's forcing files; returns the paths."""
        from .io import history
        gp = self.gp
        dt_h = 24 // self.model_steps
        spec = self.spec(dt_hours=dt_h, log=log)
        for f in spec.files:
            f.out_dt = f.out_dt or dt_h

        def num(key, default):
            v = gp.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default
        plapse = gp.get('PLAPSE')
        data = history.force_data_42d(
            self.fields, elevation=self.elevation, dt_hours=dt_h,
            max_snow_temp=num('MAX_SNOW_TEMP', 0.5),
            min_rain_temp=num('MIN_RAIN_TEMP', -0.5),
            min_wind_speed=num('MIN_WIND_SPEED', 0.1),
            plapse=True if plapse is None else gp.get_bool('PLAPSE'),
            alma=spec.alma)
        rd = history.resolve_result_dir(gp, result_dir, log=log)
        nrecs = self.n_steps // self.snow_steps * 0 + \
            self.n_steps // (self.snow_steps // self.model_steps)
        return history.write_forcing_42d(
            data, self.dates, nrecs, np.asarray(self.dom.lat),
            np.asarray(self.dom.lon), spec, rd, dt_hours=dt_h, log=log)

    def __repr__(self):
        return ('<ForceCase %d cells, %d forcing steps at %d/day (model step '
                '%d/day)>' % (self.n_cells, self.n_steps, self.snow_steps,
                              self.model_steps))


def _f32(x):
    """x as the float32 VIC 4.2.d stores it, back in double."""
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def _load_force(gp, dom, par, fields, times, say, dtype, device, is_classic,
                margin_days):
    """OUTPUT_FORCE TRUE: initialize_atmos.c's forcing, not the model.

    A daily forcing is expanded to the snow step with MTCLIM
    (torchvic.forcing.daily, the MetSim-based chain) over `margin_days` of the
    file either side of the window, as the model path does, and cut back; a
    sub-daily forcing is taken as it is (the snow step must be the forcing's
    step). The window is the global file's."""
    mspd = gp.get_int('MODEL_STEPS_PER_DAY')
    if not mspd:
        th = gp.get_int('TIME_STEP')
        mspd = 24 // th if th else None
    if not mspd:
        raise ValueError('%s: MODEL_STEPS_PER_DAY (or TIME_STEP) is not given'
                         % gp.path)
    is_daily = ('tair' not in fields and 'tmax' in fields
                and 'tmin' in fields)
    sspd = gp.get_int('SNOW_STEPS_PER_DAY')
    if not sspd:
        sh = gp.get_int('SNOW_STEP')
        sspd = 24 // sh if sh else mspd
    spd_f = (None if is_daily else max(int(round(
        86400.0 / (ImageForcing.step_seconds(times) or 86400.0))), 1))
    if spd_f is not None and spd_f != sspd:
        raise ValueError(
            'OUTPUT_FORCE: the forcing has %d records a day and the snow '
            'step is %d a day; VIC reads the forcing at the snow step '
            '(get_global_param.c:590-593)' % (spd_f, sspd))
    w0, w1 = window(gp, times, mspd)
    elev = _f32(image_run.band_mean_elevation(
        par, snow_band=gp.get_bool('SNOW_BAND'), classic=is_classic))
    if is_daily:
        from .forcing import daily as daily_forcing
        # VIC 4.2.d reads the forcing from the model's start to its end and
        # MTCLIM sees nothing outside (read_atmos_data.c skips to the start),
        # so, unlike the model path, no margin of the file is borrowed.
        a0, b0 = w0, w1
        fl = {k: v[a0:b0] for k, v in fields.items()}
        tl = np.asarray(times)[a0:b0]
        say('daily forcing: MTCLIM to %d steps/day (initialize_atmos.c)'
            % sspd)
        # VIC 4.2.d places the solar day by the soil file's off_gmt:
        # time_zone_lng = off_gmt * 360 / 24, held as a float
        # (read_soilparam.c:934, initialize_atmos.c:218). Cells differ in
        # their offset from it, which MTCLIM takes one at a time.
        off = (np.asarray(par.off_gmt, dtype=np.float64)
               if par.off_gmt is not None else None)
        # lat, lng and elevation are floats in VIC 4.2.d's soil_con
        # (vicNl_def.h:995-999), read by initialize_atmos as doubles
        lat_ = _f32(dom.lat)
        lon_ = _f32(dom.lon)
        elv_ = _f32(par.soil['elev'])
        parts, times2 = [], None
        for i in range(dom.n_active):
            tz = (None if off is None else
                  float(np.float32(off[i] * 360. / 24.)))
            fi = {k: v[:, i:i + 1] for k, v in fl.items()}
            f2, t2 = daily_forcing.expand_forcing(
                fi, tl, sspd, lat=lat_[i:i + 1], lng=lon_[i:i + 1],
                elev=elv_[i:i + 1], time_zone_lng=tz,
                log=(say if i == 0 else (lambda *a: None)), carbon=True)
            parts.append(f2)
            times2 = t2
        fields2 = {k: torch.cat([p_[k] for p_ in parts], 1)
                   for k in parts[0]
                   if torch.is_tensor(parts[0][k])}
        lead = (w0 - a0) * sspd
        n = (w1 - w0) * sspd
        fields = {k: v[lead:lead + n] for k, v in fields2.items()
                  if torch.is_tensor(v) and v.shape[0] == len(times2)}
        times = np.asarray(times2)[lead:lead + n]
    else:
        fields = {k: v[w0:w1] for k, v in fields.items()}
        times = np.asarray(times)[w0:w1]
    _force_extras(fields, times, gp, dom, par, sspd, say)
    return ForceCase(gp, dom, par, fields, times, mspd, sspd, elev, dtype,
                     device)


def _force_extras(fields, times, gp, dom, par, sspd, say):
    """The variables initialize_atmos.c adds that write_forcing_file.c can
    write: the cosine of the solar zenith angle (:1588-1607, at the step's
    whole start hour -- dmy.hour is an int), PAR = 0.45 * SWDOWN (:1630-1638),
    CATM = 383 ppm as a mixing ratio (:1688-1695), and -- from MTCLIM --
    the direct fraction and cloud fraction (:1612-1623, :1359-1371). Put
    into `fields` as coszen, par, catm, fdir, tskc, replacing the VIC 5
    convention of coszen the daily-forcing chain returns."""
    import datetime as _dt
    from .io import history
    n, ncell = len(times), dom.n_active
    lat = _f32(dom.lat)
    lng = _f32(dom.lon)
    off = (np.asarray(par.off_gmt, dtype=np.float64)
           if par.off_gmt is not None else None)
    tz = (lng.copy() if off is None
          else np.float32(off * 360. / 24.).astype(np.float64))
    hoff = (tz - lng) * 24 / 360
    hoi = np.where(hoff < 0, np.trunc(hoff - 0.5), np.trunc(hoff + 0.5))
    ts = [_dt.datetime.utcfromtimestamp(
        (np.datetime64(t, 's') - np.datetime64('1970-01-01T00:00:00'))
        .astype(np.int64)) for t in np.asarray(times)]
    doy = np.array([t.timetuple().tm_yday for t in ts], dtype=np.float64)
    snow_h = 24.0 / sspd
    starthour = float(ts[0].hour)
    # hour = rec*dt + j*SNOW_STEP + starthour - hour_offset_int  (+24 if < 0),
    # counted from the run's start (initialize_atmos.c:1597-1600)
    h = (np.arange(n, dtype=np.float64)[:, None] * snow_h + starthour
         - hoi[None, :])
    h = h + np.where(starthour - hoi < 0, 24.0, 0.0)[None, :]
    h = np.trunc(h + 0.5 * snow_h)              # dmy_tmp.hour is an int
    fields['coszen'] = torch.as_tensor(
        history.coszen_42d(lat[None, :], lng[None, :], tz[None, :],
                           doy[:, None], h),
        dtype=fields['tair'].dtype if 'tair' in fields else torch.float64)
    ref = fields['shortwave']
    fields['par'] = 0.45 * ref
    fields['catm'] = torch.full_like(ref, 383 * 1e-6)
    if 'fdir' in fields and 'tskc' in fields:
        return
    # Sub-daily forcing: MTCLIM sees its daily extremes, vapour pressure
    # and radiation (initialize_atmos.c:243-251, :869-956)
    from .forcing import daily as daily_forcing
    spd = sspd
    if n % spd:
        say('OUTPUT_FORCE: %d steps is not a whole number of days; FDIR and '
            'TSKC are not derived' % n)
        return
    nd = n // spd

    def per_day(x, f):
        a = x.reshape(nd, spd, ncell)
        return f(a, 1)
    tair = fields['tair']
    days = dict(tmax=per_day(tair, lambda a, d: a.amax(d)),
                tmin=per_day(tair, lambda a, d: a.amin(d)),
                prcp=per_day(fields['prcp'], lambda a, d: a.sum(d)),
                vp=per_day(fields['vp'], lambda a, d: a.mean(d)),
                shortwave=per_day(fields['shortwave'],
                                  lambda a, d: a.mean(d)))
    start = ts[0].date()
    fd, tk = [], []
    for i in range(ncell):
        di = {k: v[:, i:i + 1].T.contiguous() for k, v in days.items()}
        o = daily_forcing.daily_to_subdaily(
            di, start, lat[i:i + 1], lng[i:i + 1],
            float(_f32(par.soil['elev'])[i]), steps_per_day=spd,
            time_zone_lng=float(tz[i]), carbon=True, dtype=tair.dtype,
            device=tair.device)
        fd.append(o['fdir'].T)
        tk.append(o['tskc'].T)
    fields['fdir'] = torch.cat(fd, 1)
    fields['tskc'] = torch.cat(tk, 1)


def load(global_file, device=None, dtype=torch.float64, steps=0, daily=False,
         no_veg=False, no_energy=False, sample_data=None, log=None,
         first_step=0, margin_days=MTCLIM_MARGIN_DAYS, daily_day='forcing',
         core=None, band_tfactor=None, one_band_cells=None):
    """Read a global file and everything it points at.

    `one_band_cells` ([cell] bool, default None: off) runs those cells on
    one snow band, VIC-CropSyst's ONE_SNOWBAND_FOR_CROP (vicNl.c:447-455,
    `snowbands.one_band_for_crop`); the other cells are as the file has
    them. Off, the run is the file's bit for bit.

    `band_tfactor` ({band: [cell] values}, default None: the file's) sets
    soil_con Tfactor of those bands after vic_init formed it, as VIC-WUR's
    crop plugin does with the crop file's Tfactor on band 0
    (VIC-WUR 3.0 crops/src/crop_init.c:157-163, in plugin_init: after
    compute_treeline, before the default state); the bands' air
    temperature and the initial soil temperatures follow it.

    `core` ('5.1.0', the default, '5.0.1' or 'vicwur3') selects the VIC
    release whose physics the case is built and run with
    (core_version.set_core); None keeps the process's current core (5.1.0
    unless set_core changed it). The global file's FROZEN_SOIL, BLOWING (and its
    five sub-options), CARBON, CANOPY_LAYERS and SHARE_LAYER_MOIST become
    the case's `options` (image_options), which Case.kwargs passes to
    basin.simulate.

    The global file decides the run (torchvic.io.options): every option is
    run as VIC runs it, read to no effect and announced, or refused with a
    ValueError naming it. The model step is MODEL_STEPS_PER_DAY and the
    simulation window STARTYEAR..ENDDAY (or NRECS), as VIC's.

    `steps` truncates the window to that many forcing records (0 = all),
    `first_step` skips that many from its front -- together they select a
    window for a quick measurement or a spin-up. `daily` overrides the
    global file's model step with a daily one (MODEL_STEPS_PER_DAY 1 needs
    no flag): the forcing is aggregated to one record a day (means,
    precipitation summed), which is VIC 5's MODEL_STEPS_PER_DAY 1
    -- the image driver aggregates the same way, vic_force.c:404-412 --
    except that VIC sub-steps a tile with snow and this does not, so
    a window whose forcing can snow is refused. `no_veg` drops
    the mosaic, `no_energy` forces the water-balance path even where the
    global file says FULL_ENERGY TRUE.

    A DAILY forcing window is expanded by MTCLIM with `margin_days` of the
    file on either side (clipped to the file) and then cut back, so the
    window's first days see the same 30-day temperature-range history as
    they would inside a whole-record run. 0 reproduces the old, marginless
    window.

    `daily_day` is the day a SUB-DAILY forcing is binned on for `daily`:
    'forcing' (default) is VIC 5's own daily step, the forcing clock's day
    -- the image driver averages the model step's NF records as the file
    holds them (vic_force.c:404-412) and never reads off_gmt
    (init_library.c:47 sets time_zone_lng 0); 'off_gmt' is each cell's
    local day from the parameter file's off_gmt (daily_by_off_gmt), which
    puts the model day on an observation's local day.
    They are different days: Walnut Gulch (-7 h) Jun-Aug 2008 on the
    off_gmt day is 1.7% more ET and 4.3% less runoff than VIC 5.0.1's
    daily run, on the forcing day within 0.1%
    (tests/test_wb_vs_vic.py). Either way only COMPLETE days are
    returned, each dropped edge day logged.
    """
    say = log if log is not None else (lambda *a: None)
    if core is not None:
        core_version.set_core(core)
        say('VIC core %s' % core_version.CORE)
    gp = GlobalParam(global_file,
                     sample_data=sample_data_dir() if sample_data is None
                     else sample_data)
    steps_per_day = gp.get_int('MODEL_STEPS_PER_DAY')
    if not steps_per_day or steps_per_day < 1:
        raise ValueError('%s: MODEL_STEPS_PER_DAY is not given; VIC stops '
                         'without it (get_global_param.c:550-553)'
                         % global_file)
    runoff_steps = gp.get_int('RUNOFF_STEPS_PER_DAY')
    say('global file: MODEL_STEPS_PER_DAY %d, RUNOFF_STEPS_PER_DAY %s'
        % (steps_per_day, runoff_steps))
    # A classic-driver global file (SOIL, no DOMAIN) is read through
    # torchvic.io.classic into the objects the image reader produces, and
    # runs the same code from here on.
    is_classic = gp.get('SOIL') is not None and gp.get('DOMAIN') is None
    if is_classic:
        from .io import classic as _classic
        copts, dom, par, fc = _classic.open_classic(gp, log=say)
        _classic_global_as_image(gp, copts)
        say('classic driver files: SNOW_BAND %d' % copts.snow_bands)
    options = image_options(gp, log=say, no_energy=no_energy)

    if not is_classic:
        dom = ImageDomain(_fix_path(gp.domain_file, say),
                          names=gp.domain_vars)
        par = ImageParams(_fix_path(gp.params_file, say), dom)
        fc = ImageForcing(gp.forcing_prefix, dom, force_type=gp.force_type)
    say(str(dom)); say(str(par))
    try:
        par.apply_density_options(bool(gp.get_bool('ORGANIC_FRACT')),
                                  bool(gp.get_bool('BULK_DENSITY_COMB')))
    except ValueError as e:
        raise ValueError('%s: %s' % (global_file, e))
    refused = check_inputs(gp, par)

    fields, times = fc.read(dtype=dtype, device=device, log=say)
    if gp.get('OUTPUT_FORCE') is not None and \
            str(gp.get('OUTPUT_FORCE')).upper() == 'TRUE':
        return _load_force(gp, dom, par, fields, times, say, dtype, device,
                           is_classic, margin_days)
    is_daily_forcing = ('tair' not in fields and 'tmax' in fields
                        and 'tmin' in fields)
    # The model step is the global file's (MODEL_STEPS_PER_DAY); `daily`
    # only overrides it. A sub-daily forcing runs at its own step or, at
    # MODEL_STEPS_PER_DAY 1, averaged to days as vic_force.c:404-412
    # averages it; VIC's other aggregations (24 records onto 8 steps) are
    # refused. SNOW_STEPS_PER_DAY must be the forcing's step, and the model
    # step's where that is sub-daily (get_global_param.c:590-593).
    spd_f = (None if is_daily_forcing else max(int(round(
        86400.0 / (ImageForcing.step_seconds(times) or 86400.0))), 1))
    snow_spd = gp.get_int('SNOW_STEPS_PER_DAY')
    if daily and steps_per_day != 1:
        say('--daily: a daily model step, overriding the global file\'s '
            'MODEL_STEPS_PER_DAY %d' % steps_per_day)
    if steps_per_day == 1 and not daily:
        say('MODEL_STEPS_PER_DAY 1: a daily model step')
        daily = True
    if spd_f is not None:
        if not daily and steps_per_day != spd_f:
            refused.append(
                'MODEL_STEPS_PER_DAY %d with a forcing of %d records a day: '
                'VIC averages the forcing onto the model step; TorchVIC does '
                'that for a daily step (MODEL_STEPS_PER_DAY 1) only'
                % (steps_per_day, spd_f))
        if snow_spd and snow_spd != spd_f:
            refused.append('SNOW_STEPS_PER_DAY %d with a forcing of %d '
                           'records a day: VIC stops on this (the forcing '
                           'is read at the snow step)' % (snow_spd, spd_f))
    if not daily and snow_spd and snow_spd != steps_per_day and \
            not is_daily_forcing:
        refused.append('SNOW_STEPS_PER_DAY %d with MODEL_STEPS_PER_DAY %d: '
                       'VIC stops on this (get_global_param.c:590-593)'
                       % (snow_spd, steps_per_day))
    cal = gp.get('CALENDAR')
    if cal is not None:
        fcal = _forcing_calendar(times)
        if _CAL_FAMILY.get(str(cal).upper(), cal) != \
                _CAL_FAMILY.get(fcal, fcal):
            refused.append('CALENDAR %s with a forcing on the %s calendar: '
                           'TorchVIC steps on the forcing\'s dates'
                           % (cal, fcal.lower()))
    # The simulation window, from the global file; first_step and steps
    # select within it.
    try:
        w0, w1 = window(gp, times, 1 if daily else steps_per_day)
    except ValueError as e:
        refused.append(str(e))
        w0, w1 = 0, len(times)
    if refused:
        raise ValueError('%s selects what TorchVIC does not run as VIC does '
                         '(docs/OPTIONS.md):\n  %s'
                         % (gp.path, '\n  '.join(refused)))
    if (w0, w1) != (0, len(times)):
        say('simulation window from the global file: forcing records '
            '%d..%d (%s .. %s)' % (w0, w1, str(np.asarray(times)[w0])[:19],
                                   str(np.asarray(times)[w1 - 1])[:19]))
    lead, n_win = 0, None
    win_days = None
    a = w0 + first_step
    b = w1 if steps <= 0 else min(a + steps, w1)
    if a >= b:
        raise ValueError('first_step %d leaves no record of the window '
                         '(%d records)' % (first_step, w1 - w0))
    # surface_fluxes.c:258-269: a day with a pack, canopy snow, or
    # snowflag is integrated at SNOW_STEPS_PER_DAY inside the daily step.
    # The records are kept below and basin.simulate sub-steps those days.
    # A window with nothing below 0.5 C and wet stays the daily mean.
    snow_sub = bool(daily and not is_daily_forcing and snow_spd
                    and snow_spd > 1 and _snow_possible(fields, a, b))
    if snow_sub:
        say('daily model step, snow possible: sub-stepping snow days at '
            'SNOW_STEPS_PER_DAY %d (surface_fluxes.c:258-269)' % snow_spd)
    if (a, b) != (0, len(times)):
        a0, b0 = a, b
        if is_daily_forcing and margin_days:
            a0 = max(a - int(margin_days), 0)
            b0 = min(b + int(margin_days), len(times))
            lead, n_win = a - a0, b - a
        elif daily and not is_daily_forcing:
            # A sub-daily window binned to days: the window names the
            # forcing-clock days its records fall on, and each of those is
            # returned as a COMPLETE day, so the records an off_gmt day
            # takes from its neighbours (|off_gmt| < 24 h) are read too.
            # Cut at the window's own records instead, an off_gmt window
            # loses its first and last day: Walnut Gulch Jun-Aug 2008
            # (-7 h, 3-hourly, 736 records) gave 91 days, not 92.
            spd_f = max(int(round(86400.0 / (ImageForcing.step_seconds(times)
                                              or 86400.0))), 1)
            a0 = max(a - spd_f, 0)
            b0 = min(b + spd_f, len(times))
            win_days = (np.datetime64(times[a], 'D'),
                        np.datetime64(times[b - 1], 'D'))
        fields = {k: v[a0:b0] for k, v in fields.items()}
        times = times[a0:b0]
        say('forcing window: steps %d..%d of the file%s'
            % (a, b, (' (read %d..%d: the MTCLIM margin)' % (a0, b0))
               if (a0, b0) != (a, b) else ''))

    # CARBON's forcings (CATM, PAR, FDIR and the solar zenith) from a daily
    # file are derived by the MTCLIM chain
    # (torchvic.forcing.daily); a sub-daily file has to supply them,
    # and basin.simulate refuses a carbon run without them, by name.
    _carbon_kw = dict(carbon=True) if options['carbon'] else {}

    def _cut(flds, stamps, per_day):
        """The window without its MTCLIM margin."""
        if n_win is None:
            return flds, stamps
        s0, s1 = lead * per_day, (lead + n_win) * per_day
        return {k: v[s0:s1] for k, v in flds.items()}, stamps[s0:s1]

    sub_raw = None
    vic_daily_nr = False
    if daily and 'tair' not in fields and 'tmax' in fields \
            and 'tmin' in fields:
        # A DAILY forcing for a daily model step: VIC 4.2.d's water-balance
        # mode runs MTCLIM to the sub-daily curve (temperature, solar
        # geometry, longwave) and the daily step reads that day's means --
        # the same chain as below, then the day's average. Skipping MTCLIM
        # would leave the model with no tair, shortwave or longwave at all.
        from .forcing import daily as daily_forcing
        spd_ = max(steps_per_day, 24)
        # Cells in blocks (every cell is computed alone, so the block size
        # changes the memory only): MTCLIM's solar geometry holds 366 x 2880
        # values per cell, 950 GB at the 112,825 cells of ISC.
        _lat, _lng = np.asarray(dom.lat), np.asarray(dom.lon)
        _elev = np.asarray(par.soil['elev'])
        _blk = daily_forcing.auto_cell_chunk(len(times), device, dtype)
        _parts, _stamps = [], times
        _kw = dict(_carbon_kw)
        if _blk < len(_lat):
            # the default annual precipitation is the call's own mean: keep
            # it the whole domain's, not each block's
            _kw['annual_prcp_mm'] = float(
                fields['prcp'].transpose(0, 1).mean(-1).mean()) * 365.25
        for _c0 in range(0, len(_lat), _blk):
            _c1 = min(_c0 + _blk, len(_lat))
            _say = say if _c0 == 0 else (lambda *x: None)
            _f, _t = daily_forcing.expand_forcing(
                {k: v[:, _c0:_c1] for k, v in fields.items()}, _stamps, spd_,
                lat=_lat[_c0:_c1], lng=_lng[_c0:_c1], elev=_elev[_c0:_c1],
                log=_say, **_kw)
            # On the MTCLIM clock: expand_forcing's hours are the cell's
            # local solar hours (time_zone_lng defaults to the cell), so the
            # day is the daily record's own day and no off_gmt shift applies.
            _d, dates = aggregate_to_daily(_f, _t, utc_offset=0.0, log=_say)
            _parts.append(_d)
            del _f
        times = _t
        fields = (_parts[0] if len(_parts) == 1 else
                  {k: torch.cat([p[k] for p in _parts], 1)
                   for k in _parts[0]})
        del _parts
        fields, dates = _cut(fields, dates, 1)
        model_steps = 1
        say('daily forcing: MTCLIM to %d/day, then daily means for a daily '
            'model step (VIC 4.2.d water-balance mode): %d days'
            % (spd_, len(dates)))
    elif daily:
        if daily_day == 'off_gmt':
            off = par.off_gmt
            if off is None:
                say('the parameter file has no off_gmt: binning on the '
                    'forcing clock\'s day')
                off = np.zeros(dom.n_active)
        elif daily_day == 'forcing':
            say('daily binning on the forcing clock\'s day, as VIC 5\'s '
                'daily step (vic_force.c:404-412)')
            off = np.zeros(dom.n_active)
        else:
            raise ValueError("daily_day is 'forcing' or 'off_gmt', not %r"
                             % (daily_day,))
        # complete local days only; partial edge days are dropped and logged
        vic_daily_nr = True
        if snow_sub:
            fields, dates, sub_raw = daily_by_off_gmt(
                fields, times, off, log=say, with_subdaily=True)
        else:
            fields, dates = daily_by_off_gmt(fields, times, off, log=say)
        if win_days is not None:
            # the window's days (the read took one day either side)
            d_a, d_b = (x.astype(object) for x in win_days)
            pos = [i for i, d in enumerate(dates) if d_a <= d <= d_b]
            got = [dates[i] for i in pos]
            if len(got) != (d_b - d_a).days + 1:
                say('  %d of the window\'s %d days are complete local days '
                    '(the forcing file ends)'
                    % (len(got), (d_b - d_a).days + 1))
            ti = torch.as_tensor(pos, device=next(iter(fields.values()))
                                 .device)
            fields = {k: v.index_select(0, ti) for k, v in fields.items()}
            dates = got
            if sub_raw is not None:
                sub_raw = {k: v.index_select(0, ti)
                           for k, v in sub_raw.items()}
        model_steps = 1
        say('forcing aggregated to %d days for a daily model step (VIC '
            'MODEL_STEPS_PER_DAY 1%s)'
            % (len(dates),
               ('; snow days at SNOW_STEPS_PER_DAY %d' % snow_spd)
               if snow_sub else ''))
    else:
        import pandas as _pd
        dates = [_pd.Timestamp(x).to_pydatetime() for x in times]
        model_steps = steps_per_day
    say('%d steps at %d/day (%s .. %s)'
        % (len(dates), model_steps, dates[0], dates[-1]))

    # A daily forcing, expanded before the atmosphere is built.
    #
    # VIC 5 cannot read one at all -- it dropped MTCLIM, and
    # get_global_param.c refuses a forcing whose step does not match the
    # model's -- so a global file naming FORCE_STEPS_PER_DAY 1 against a
    # sub-daily MODEL_STEPS_PER_DAY is a file only this project can run.
    # `daily_forcing.expand_forcing` is the MetSim-based MTCLIM chain and
    # returns the same [time, cell] layout on the model's own step, so
    # nothing after this point learns that the forcing was daily.
    #
    # Detected by what the file contains rather than by the declared
    # FORCE_STEPS_PER_DAY: a file with tmax and tmin and no tair is daily
    # whatever it says, and that is the condition build_atmosphere raises
    # on a few lines below.
    force_spd = gp.get_int('FORCE_STEPS_PER_DAY') or 0
    if (not daily and model_steps > 1 and 'tair' not in fields
            and 'tmax' in fields and 'tmin' in fields):
        from .forcing import daily as daily_forcing
        say('daily forcing (FORCE_STEPS_PER_DAY %d): MTCLIM 4.3 expands it '
            'to %d steps/day' % (force_spd, model_steps))
        # `times`, not `dates`: expand_forcing reads numpy datetime64
        # stamps (it calls .astype on them), while `dates` above has
        # already been converted to python datetimes for the run.
        import pandas as _pd2
        fields, times = daily_forcing.expand_forcing(
            fields, times, model_steps,
            lat=np.asarray(dom.lat), lng=np.asarray(dom.lon),
            elev=np.asarray(par.soil['elev']), log=say, **_carbon_kw)
        fields, times = _cut(fields, times, model_steps)
        dates = [_pd2.Timestamp(x).to_pydatetime() for x in times]
        say('  -> %d sub-daily steps (%s .. %s)'
            % (len(dates), dates[0], dates[-1]))

    # SNOW_BAND FALSE (VIC's default) keeps the file's elev
    # (vic_init.c 5.1.0:957-962); TRUE takes the band mean.
    elev = image_run.band_mean_elevation(
        par, snow_band=gp.get_bool('SNOW_BAND'), classic=is_classic)
    if not np.allclose(elev, np.asarray(par.soil['elev']), atol=1.0):
        say('elevation: the band-weighted mean VIC uses (%.1f m) rather than '
            'the file\'s elev (%.1f m)'
            % (float(elev.mean()), float(np.asarray(par.soil['elev']).mean())))
    batch = image_run.ImageBatch(dom, dates, dtype=dtype, device=device,
                                 elevation=elev)
    atm = image_run.build_atmosphere(fields, batch)
    if vic_daily_nr:
        # vic_force.c:453: vpd[NR] = svp(air_temp[NR]) - vp[NR], with NO
        # clamp -- only the sub-daily vpd[j] are clamped (:402-405). On a
        # saturated day the mean of the clamped vp can exceed svp of the
        # mean temperature and VIC's vpd[NR] goes negative; clamping it to
        # 0 here would differ. build_atmosphere's own clamp is kept for
        # its other (sub-daily) callers, where NF == 1 makes the two the
        # same value.
        atm['vpd'] = _svp(fields['tair']) - atm['eact_air']
    subdaily = None
    if sub_raw is not None:
        # The same atmosphere the daily means were built with, on each
        # snow step: vpd, density and latent heat are the sub-step's own
        # (vic_force.c:395-414), not the day's.
        nd, nf, ncell = sub_raw['tair'].shape
        flat = {k: v.reshape(nd * nf, *v.shape[2:])
                for k, v in sub_raw.items()
                if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == nd
                and v.shape[1] == nf}
        built = image_run.build_atmosphere(flat, batch)
        subdaily = {}
        for k, v in built.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == nd * nf:
                subdaily[k] = v.reshape(nd, nf, *v.shape[1:])
        say('snow sub-steps kept beside the daily means: %d per day '
            '(vic_force.c:443-458)' % nf)
    for k, v in atm.pop('_sources', {}).items():
        say('  %-10s %s' % (k, v))
    if options['carbon']:
        for k in ('par', 'Catm', 'fdir', 'coszen'):
            if k in fields:
                atm[k] = fields[k]
        if 'coszen' not in atm and all(k in atm for k in ('par', 'Catm',
                                                          'fdir')):
            # vic_force.c:196-205: computed, never read; soil_con->
            # time_zone_lng is 0 in the image driver (init_library.c:27),
            # and the second is the one at the start of the step.
            from .core import atmos as _atmos
            doy = torch.tensor([d.timetuple().tm_yday for d in dates],
                               dtype=dtype, device=device)
            sec = torch.tensor([d.hour * 3600 + d.minute * 60 + d.second
                                for d in dates], dtype=dtype, device=device)
            lat_ = torch.as_tensor(np.asarray(dom.lat), dtype=dtype,
                                   device=device).reshape(1, -1)
            lon_ = torch.as_tensor(np.asarray(dom.lon), dtype=dtype,
                                   device=device).reshape(1, -1)
            atm['coszen'] = _atmos.coszen(lat_, lon_, torch.zeros_like(lon_),
                                          doy.reshape(-1, 1),
                                          sec.reshape(-1, 1)).contiguous()
            say('coszen computed at the start of each step '
                '(vic_force.c:196-205, time_zone_lng 0)')
    soil = image_run.build_soil(par, dtype, device,
                                runoff_steps_per_day=runoff_steps,
                                model_steps_per_day=model_steps,
                                baseflow=(gp.get('BASEFLOW') or 'ARNO'))
    # FCAN_SRC decides whether the file's monthly fcanopy is read
    # (vic_init.c:263-281 in the image driver). It was never passed here, so
    # every image case ran FROM_DEFAULT (fcanopy 1) whatever its global file
    # said; the classic reader always honoured it (vic_classic.py:236).
    fcan_src = (gp.get('FCAN_SRC') or 'FROM_DEFAULT').upper()
    tl_on, tl_veg = _opt.treeline(gp)
    if tl_on and gp.get_bool('SNOW_BAND') and par.n_bands > 1:
        # read_vegparam.c:371-433: the tile above-treeline bands can stand on
        treeline.add_above_treeline_tile(par, tl_veg, log=say)
        if getattr(par, 'fcanopy_from_default', False) and \
                fcan_src == 'FROM_DEFAULT':
            fcan_src = 'FROM_VEGLIB'
    hist_vars = veghist.wanted(gp)
    mosaic = None if no_veg else image_run.build_mosaic(
        par, dtype, device, log=say, fcan_src=fcan_src,
        keep_raw=bool(hist_vars),
        max_snow_albedo=bool(gp.get_bool('MAX_SNOW_ALBEDO')),
        **({'nbare': int(gp.get_int('NBARE') or 1)}
           if core_version.vicwur3() else {}))
    if hist_vars and mosaic is not None:
        # LAI_SRC / FCAN_SRC / ALB_SRC FROM_VEGHIST: FORCING2's per-step
        # values replace the parameter file's monthly ones
        # (vic_force.c:264-363). check_inputs has refused the files that
        # would not open.
        years = sorted({d.year for d in dates})
        hist, hstamps = veghist.read(gp.forcing2_prefix, dom, hist_vars,
                                     years, dtype=dtype, device=device,
                                     log=say)
        veghist.apply(mosaic, hist, hstamps, dates, daily, log=say)
    if mosaic is not None:
        say('vegetation: %d tiles per cell, mean cover %.3f (FCAN_SRC %s)'
            % (mosaic.n_tiles, float(mosaic.cv.sum(1).mean()), fcan_src))
    wind_h, wind_src, _ = resolve_wind_h(
        gp, params_reference_heights(par), float(atm['tair'].min()),
        mosaic=mosaic is not None, log=say)
    say('wind height %g m (%s)' % (wind_h, wind_src))
    init_moist = torch.as_tensor(par.soil['init_moist'], dtype=dtype,
                                 device=device).T.contiguous()
    full_energy = bool(gp.get_bool('FULL_ENERGY') and mosaic is not None
                       and not no_energy)
    say('surface energy balance: %s (FULL_ENERGY %s in the global file)'
        % ('ON' if full_energy else 'off', gp.get('FULL_ENERGY')))
    snow_ex, cell_echo = None, None
    if (one_band_cells is not None and gp.get_bool('SNOW_BAND')
            and all(k in par.bands for k in
                    ('AreaFract', 'elevation', 'Pfactor'))):
        # vicNl.c:447-455: a cell with a crop runs on one band
        par.bands = dict(par.bands)
        (par.bands['AreaFract'], par.bands['elevation'],
         par.bands['Pfactor']) = snowbands.one_band_for_crop(
            par.bands['AreaFract'], par.bands['elevation'],
            par.bands['Pfactor'], one_band_cells, par.soil['elev'])
    if gp.get_bool('SNOW_BAND') and all(k in par.bands for k in
                                        ('AreaFract', 'elevation', 'Pfactor')):
        # check_inputs refused the layouts VIC halts on, so expand()
        # cannot raise here. A domain with one band a cell is not
        # expanded at all, and its run is the one-column run bit for bit.
        snow_ex = snow_expansion(par, is_classic)
        if snow_ex is not None:
            if band_tfactor:
                # VIC-WUR 3.0 crop_init.c:157-163 (the treeline, vic_init,
                # saw the file's Tfactor)
                snow_ex['tfactor_file'] = np.array(snow_ex['tfactor'],
                                                   dtype=np.float64)
                tf_ = np.array(snow_ex['tfactor'], dtype=np.float64)
                cl_ = np.asarray(snow_ex['cell'])
                bd_ = np.asarray(snow_ex['band'])
                for b_, v_ in band_tfactor.items():
                    v_ = np.asarray(v_, dtype=np.float64).reshape(-1)
                    on_ = bd_ == int(b_)
                    tf_[on_] = v_[cl_[on_]]
                snow_ex['tfactor'] = tf_
                say('band Tfactor of band(s) %s set by the caller '
                    '(VIC-WUR 3.0 crop_init.c:157-163)' % sorted(band_tfactor))
            say(snowbands.summary(snow_ex))
            (batch, atm, soil, mosaic, init_moist, subdaily,
             cell_echo) = _expand_bands(batch, atm, soil, mosaic, init_moist,
                                        subdaily, snow_ex, dtype, device)
    if band_tfactor and snow_ex is None and any(
            np.any(np.asarray(v_, dtype=np.float64) != 0.0)
            for v_ in band_tfactor.values()):
        raise ValueError('band_tfactor: a non-zero Tfactor on a run without '
                         'snow bands (one column a cell) is not supported')
    if tl_on and mosaic is not None:
        # compute_treeline.c / put_data.c: outputs of the bands above the
        # treeline drop their overstory tiles and re-weight the rest.
        if snow_ex is not None:
            col_cell = np.asarray(snow_ex['cell'])
            col_tf = np.asarray(snow_ex.get('tfactor_file',
                                            snow_ex['tfactor']),
                                dtype=np.float64)
        else:
            col_cell = np.arange(dom.n_active)
            col_tf = np.zeros(dom.n_active)
        above = treeline.above_treeline(par.july_tavg[col_cell], col_tf)
        ov = (mosaic.overstory if mosaic.overstory is not None
              else torch.zeros_like(mosaic.cv))
        mosaic.cv = treeline.adjust_cover(mosaic.cv, ov.cpu().numpy(), above)
        say('COMPUTE_TREELINE: %d of %d columns are above the treeline '
            '(July_Tavg + Tfactor <= %g C)'
            % (int(above.sum()), above.size, treeline.TREELINE_TEMP))
    if gp.get_bool('LAKES'):
        # vic_init.c 5.1.0:1336-1550 and vic_run.c 5.1.0:132-466
        # (torchvic.model.lakecell)
        from .model import lakecell
        if snow_ex is not None:
            raise ValueError('LAKES TRUE with more than one snow band is not '
                             'run: the lake tile is solved on band 0 only '
                             '(vic_run.c 5.1.0:155) and this port has not '
                             'built that')
        if not options.get('quick_flux', True):
            # the lake coupling (model/lakecell.py) was verified on the
            # two-node ground flux only; VIC runs the lake's land tiles on
            # the finite-difference profile too (vic_run.c 5.1.0:132-466)
            raise ValueError('LAKES TRUE with QUICK_FLUX FALSE is not run: '
                             'the lake tile is verified with QUICK_FLUX '
                             'TRUE only')
        if mosaic is None or not full_energy:
            raise ValueError('LAKES TRUE needs FULL_ENERGY TRUE and the '
                             'vegetation mosaic (get_global_param.c '
                             '5.1.0:980)')
        lakes = lakecell.read_image_lakes(
            par.ds, dom, mosaic, soil, bool(gp.get_bool('LAKE_PROFILE')),
            cell_area=batch.cell_area.detach().cpu().numpy(),
            core=core or core_version.CORE, dtype=dtype, device=device,
            log=say)
        if lakes is not None:
            _wh = gp.get('WIND_H')
            lakes.wind_h = float(_wh) if _wh is not None else 10.0
            options = dict(options, lakes=lakes)
    loaded = Case(gp, dom, par, batch, atm, soil, mosaic, dates, model_steps,
                  wind_h, init_moist, full_energy, dtype, device,
                  options=options, core=core)
    loaded.treeline = bool(tl_on and mosaic is not None)
    if snow_ex is not None:
        loaded.snowbands = snow_ex
        loaded._cell_echo = cell_echo
    if subdaily is not None:
        loaded.subdaily = subdaily
        loaded.snow_steps = int(subdaily['tair'].shape[1])
    return loaded
