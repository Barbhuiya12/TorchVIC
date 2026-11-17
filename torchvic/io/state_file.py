"""VIC 5.1.0 image-driver state files (restart files).

The layout, variable names, dimensions and units are those of
`vic_store.c` / `initialize_state_file` (drivers/shared_image/src) and
`state_metadata.c`; `read_state_file` is `vic_restore.c`. A state file holds
the model at the end of one step, per (veg_class, snow_band, cell):

    STATE_SOIL_MOISTURE    (veg_class, snow_band, nlayer, y, x)        mm
    STATE_SOIL_ICE         (veg_class, snow_band, nlayer, frost_area, y, x)
    STATE_CANOPY_WATER     (veg_class, snow_band, y, x)                mm
    STATE_SNOW_*           (veg_class, snow_band, y, x)  age, melt flag,
                           coverage, water equivalent, surface and pack
                           temperature and water, density, cold content
                           and the canopy's snow
    STATE_SOIL_NODE_TEMP   (veg_class, snow_band, soil_node, y, x)     C
    STATE_FOLIAGE_TEMPERATURE, STATE_ENERGY_LONGUNDEROUT,
    STATE_ENERGY_SNOW_FLUX (veg_class, snow_band, y, x)
    STATE_AVG_ALBEDO       (y, x)

and the coordinate variables, dz_node and node_depth. Lakes and the carbon
cycle add variables to VIC's file; they are refused by name.
"""
import datetime as dt
import os

import numpy as np

def _nodes(case):
    """VIC's Nnode: NODES, which QUICK_FLUX resets to 3 (get_global_param.c
    5.1.0:925-931); the case holds the effective value."""
    return int(case.options.get('nnodes', 3))


def _nc_lock():
    """The lock xarray's netCDF4 backend holds around every read. The
    forcing reader runs in a background thread (io/forcing.py) and HDF5 is
    not thread-safe, so this module's own netCDF4 calls take the same one."""
    from xarray.backends.locks import HDF5_LOCK, NETCDFC_LOCK, combine_locks
    return combine_locks([NETCDFC_LOCK, HDF5_LOCK])


FILL_D = 9.969209968386869e+36        # NC_FILL_DOUBLE
FILL_I = -2147483647                  # NC_FILL_INT

#: (name, nc type, dims after the leading ones, long_name, standard_name,
#:  units, description): state_metadata.c, in the enum order of
#: vic_driver_shared_all.h that initialize_state_file defines them in.
_VARS = (
    ('STATE_SOIL_MOISTURE', 'd', 'veg band layer', 'soil_moisture',
     'soil_layer_moisture', 'mm',
     'soil total moisture contents including ice for each soil layer'),
    ('STATE_SOIL_ICE', 'd', 'veg band layer frost', 'soil_ice',
     'soil_moisture_ice_depth', 'mm',
     'soil ice content for each soil layer'),
    ('STATE_CANOPY_WATER', 'd', 'veg band', 'canopy_water',
     'water_in_canopy', 'mm',
     'amount of water stored in the vegetation canopy'),
    ('STATE_SNOW_AGE', 'i', 'veg band', 'snow_age',
     'age_since_last_new_snow', 'model_time_step',
     'number of model time steps since the last new snow'),
    ('STATE_SNOW_MELT_STATE', 'i', 'veg band', 'snow_melt_state',
     'snow_melting_phase', '1 melting, 0 not melting',
     'flag to indicate whether snowpack is in accumulation or melting phase'),
    ('STATE_SNOW_COVERAGE', 'd', 'veg band', 'snow_coverage',
     'snow_coverage_fraction', '1',
     'fraction of grid cell area covered by snow'),
    ('STATE_SNOW_WATER_EQUIVALENT', 'd', 'veg band', 'snow_water_equivalent',
     'snow_water_equivalent', 'm', 'snow water equivalent'),
    ('STATE_SNOW_SURF_TEMP', 'd', 'veg band', 'snow_surf_temp',
     'snow_surface_temperature', 'C', 'snow surface layer temperature'),
    ('STATE_SNOW_SURF_WATER', 'd', 'veg band', 'snow_surf_water',
     'snow_surface_liquid_water', 'm',
     'liquid water content of the snow surface layer'),
    ('STATE_SNOW_PACK_TEMP', 'd', 'veg band', 'snow_pack_temp',
     'snow_pack_temperature', 'C', 'snow pack layer temperature'),
    ('STATE_SNOW_PACK_WATER', 'd', 'veg band', 'snow_pack_water',
     'snow_pack_liquid_water', 'm',
     'liquid water content of the snow pack layer'),
    ('STATE_SNOW_DENSITY', 'd', 'veg band', 'snow_density',
     'snowpack_density', 'kg m-3', 'snowpack density'),
    ('STATE_SNOW_COLD_CONTENT', 'd', 'veg band', 'snow_cold_content',
     'snowpack_cold_content', 'J m-2', 'snowpack cold content'),
    ('STATE_SNOW_CANOPY', 'd', 'veg band', 'snow_canopy',
     'snow_water_equivalent_intercepted_in_canopy', 'm',
     'snow interception storage in canopy'),
    ('STATE_SOIL_NODE_TEMP', 'd', 'veg band node', 'soil_node_temp',
     'soil_node_temperature', 'C',
     'soil temperature of each soil thermal node'),
    ('STATE_FOLIAGE_TEMPERATURE', 'd', 'veg band', 'foliage_temperature',
     'foliage_temperature', 'C', 'overstory vegetaion temperature'),
    ('STATE_ENERGY_LONGUNDEROUT', 'd', 'veg band', 'energy_longunderout',
     'longwave_out_from_understory', 'W m-2',
     'outgoing longwave flux from understory vegetation'),
    ('STATE_ENERGY_SNOW_FLUX', 'd', 'veg band', 'energy_snow_flux',
     'snowpack_thermal_flux', 'W m-2', 'thermal flux through snowpack'),
    ('STATE_AVG_ALBEDO', 'd', '', 'state_avg_albedo',
     'state_gridcell_avg_albedo', 'fraction', 'gridcell averaged albedo'),
)
VARIABLES = tuple(v[0] for v in _VARS)

#: Added by CARBON TRUE (vic_store.c:182-300; state_metadata.c:62-106),
#: all (veg_class, snow_band, y, x) in g m-2.
_VARS_CARBON = (
    ('STATE_ANNUALNPP', 'd', 'veg band', 'annualnpp',
     'running_total_annual_NPP', 'g m-2', 'running total annual NPP'),
    ('STATE_ANNUALNPPPREV', 'd', 'veg band', 'annualnppprev',
     'previous_year_total_annual_NPP', 'g m-2',
     'total annual NPP from previous year'),
    ('STATE_CLITTER', 'd', 'veg band', 'clitter', 'carbon_in_litter_pool',
     'g m-2', 'carbon storage in litter pool'),
    ('STATE_CINTER', 'd', 'veg band', 'cinter', 'carbon_in_intermediate_pool',
     'g m-2', 'carbon storage in intermediate pool'),
    ('STATE_CSLOW', 'd', 'veg band', 'cslow', 'carbon_in_slow_pool', 'g m-2',
     'carbon storage in slow pool'),
)
#: carbon_state key -> the file's variable
_CARBON_KEYS = (('annual_npp', 'STATE_ANNUALNPP'),
                ('annual_npp_prev', 'STATE_ANNUALNPPPREV'),
                ('c_litter', 'STATE_CLITTER'), ('c_inter', 'STATE_CINTER'),
                ('c_slow', 'STATE_CSLOW'))


#: TorchVIC-only variables (VIC ignores them): the SPATIAL_SNOW stores
#: vic_restore.c does not read, so that a TorchVIC restart from its own
#: file continues the run bit for bit; a file without them (VIC's) starts
#: them from VIC's initialisation (initialize_snow.c), as VIC does.
_SP_EXTRA = (('store_swq', 'TV_SP_STORE_SWQ', 'd'),
             ('store_coverage', 'TV_SP_STORE_COVERAGE', 'd'),
             ('max_snow_depth', 'TV_SP_MAX_SNOW_DEPTH', 'd'),
             ('snow_distrib_slope', 'TV_SP_SNOW_DISTRIB_SLOPE', 'd'),
             ('store_snow', 'TV_SP_STORE_SNOW', 'i'),
             # energy->advected_sensible, which the next step's folded
             # thin pack reports (surface_fluxes.c 5.1.0:236, :871)
             ('adv_sens', 'TV_SP_ADV_SENS', 'd'))


#: Added by LAKES TRUE (vic_store.c lake section; state_metadata.c), as VIC
#: 5.1.0's own file names and describes them.
_VARS_LAKE = (
    ('STATE_LAKE_SOIL_MOISTURE', 'd', 'layer', 'lake_soil_moisture',
     'lake_soil_moisture', 'mm',
     'soil moisture below lake'),
    ('STATE_LAKE_SOIL_ICE', 'd', 'layer frost', 'lake_soil_ice',
     'lake_soil_ice_content', 'mm',
     'soil ice content below lake'),
    ('STATE_LAKE_SNOW_AGE', 'i', '', 'lake_snow_age',
     'lake_age_since_last_new_snow', 'model_time_step',
     'number of model time steps since the last new snow on lake ice'),
    ('STATE_LAKE_SNOW_MELT_STATE', 'i', '', 'lake_snow_melt_state',
     'lake_snow_melting_phase', '1 melting, 0 not melting',
     'flag to indicate whether snowpack is in accumulation or melting phase on lake ice'),
    ('STATE_LAKE_SNOW_COVERAGE', 'd', '', 'lake_snow_coverage',
     'lake_snow_coverage_fraction', '1',
     'fraction of grid cell area covered by snow on lake ice'),
    ('STATE_LAKE_SNOW_WATER_EQUIVALENT', 'd', '', 'lake_snow_water_equivalent',
     'lake_snow_water_equivalent', 'm',
     'lake snow water equivalent on lake ice'),
    ('STATE_LAKE_SNOW_SURF_TEMP', 'd', '', 'lake_snow_surf_temp',
     'lake_snow_surface_temperature', 'C',
     'snow surface layer temperature on lake ice'),
    ('STATE_LAKE_SNOW_SURF_WATER', 'd', '', 'lake_snow_surf_water',
     'lake_snow_surface_temperature', 'm',
     'liquid water content of the snow surface layer on lake ice'),
    ('STATE_LAKE_SNOW_PACK_TEMP', 'd', '', 'lake_snow_pack_temp',
     'lake_snow_pack_temperature', 'C',
     'snow pack layer temperature on lake ice'),
    ('STATE_LAKE_SNOW_PACK_WATER', 'd', '', 'lake_snow_pack_water',
     'lake_snow_surface_liquid_water', 'm',
     'liquid water content of the snow surface layer on lake ice'),
    ('STATE_LAKE_SNOW_DENSITY', 'd', '', 'lake_snow_density',
     'lake_snowpack_density', 'kg m-3',
     'snowpack density on lake ice'),
    ('STATE_LAKE_SNOW_COLD_CONTENT', 'd', '', 'lake_snow_cold_content',
     'lake_snowpack_cold_content', 'J m-2',
     'snowpack cold content on lake ice'),
    ('STATE_LAKE_SNOW_CANOPY', 'd', '', 'lake_snow_canopy',
     'lake_snow_water_equivalent_intercepted_in_canopy', 'm',
     'snow interception storage in canopy on lake ice'),
    ('STATE_LAKE_SOIL_NODE_TEMP', 'd', 'node', 'lake_soil_node_temp',
     'lake_soil_node_temperature', 'C',
     'soil temperature of each soil thermal node below lake'),
    ('STATE_LAKE_ACTIVE_LAYERS', 'i', '', 'lake_active_layers',
     'lake_active_layers', '-',
     'number of nodes whose corresponding layers currently contain water'),
    ('STATE_LAKE_LAYER_DZ', 'd', '', 'lake_layer_dz',
     'lake_thickness_layer_below_surface', 'm',
     'vertical thickness of all horizontal lake water layers below the surface layer'),
    ('STATE_LAKE_SURF_LAYER_DZ', 'd', '', 'lake_surf_layer_dz',
     'lake_thickness_surface_layer', 'm',
     'vertical thickness of surface water layer in lake'),
    ('STATE_LAKE_DEPTH', 'd', '', 'lake_depth',
     'lake_liquid_water_depth', 'm',
     'distance from surface to deepest point in lake'),
    ('STATE_LAKE_LAYER_SURF_AREA', 'd', 'lake', 'lake_layer_surf_area',
     'lake_node_surface_area', 'm2',
     'surface area of liquid water in lake at each node'),
    ('STATE_LAKE_SURF_AREA', 'd', '', 'lake_surf_area',
     'lake_surface_area', 'm2',
     'surface area of liquid plus ice water on lake surface'),
    ('STATE_LAKE_VOLUME', 'd', '', 'lake_volume',
     'lake_volume', 'm3',
     'lake total volume including liquid water equivalent of lake ice'),
    ('STATE_LAKE_LAYER_TEMP', 'd', 'lake', 'lake_layer_temp',
     'lake_layer_temp', 'C',
     'lake water temperature at each node'),
    ('STATE_LAKE_AVERAGE_TEMP', 'd', '', 'lake_average_temp',
     'lake_average_temperature', 'C',
     'average liquid water temperature of entire lake'),
    ('STATE_LAKE_ICE_AREA', 'd', '', 'lake_ice_area',
     'lake_ice_coverage', 'm2',
     'area of ice coverage on lake at beginning of time step'),
    ('STATE_LAKE_ICE_AREA_NEW', 'd', '', 'lake_ice_area_new',
     'lake_ice_area_new', 'm2',
     'area of ice coverage on lake at end of time step'),
    ('STATE_LAKE_ICE_WATER_EQUIVALENT', 'd', '', 'lake_ice_water_equivalent',
     'lake_ice_water_equivalent', 'm3',
     'liquid water equivalent volume of lake ice'),
    ('STATE_LAKE_ICE_HEIGHT', 'd', '', 'lake_ice_height',
     'lake_ice_height_thickest', 'm',
     'lake ice height at ghickest point'),
    ('STATE_LAKE_ICE_TEMP', 'd', '', 'lake_ice_temp',
     'lake_ice_temperature', 'C',
     'lake ice temperature'),
    ('STATE_LAKE_ICE_SWE', 'd', '', 'lake_ice_swe',
     'lake_snow_water_equivalent', 'm',
     'liquid water equivalent depth of lake snow'),
    ('STATE_LAKE_ICE_SNOW_SURF_TEMP', 'd', '', 'lake_ice_snow_surf_temp',
     'lake_snow_surface_temperature', 'C',
     'temperature of snow surface layer of lake snow'),
    ('STATE_LAKE_ICE_SNOW_PACK_TEMP', 'd', '', 'lake_ice_snow_pack_temp',
     'lake_ice_snow_pack_temperature', 'C',
     'temperature of snow pack layer of lake snow'),
    ('STATE_LAKE_ICE_SNOW_COLD_CONTENT', 'd', '', 'lake_ice_snow_cold_content',
     'lake_ice_snow_cold_content', 'J m-2',
     'snowpack cold content of snow lake'),
    ('STATE_LAKE_ICE_SNOW_SURF_WATER', 'd', '', 'lake_ice_snow_surf_water',
     'lake_ice_snow_surface_liquid_water', 'm',
     'liquid water content of surface snow layer of lake snow'),
    ('STATE_LAKE_ICE_SNOW_PACK_WATER', 'd', '', 'lake_ice_snow_pack_water',
     'lake_ice_snow_pack_liquid_water', 'm',
     'liquid water content of pack snow layer of lake snow'),
    ('STATE_LAKE_ICE_SNOW_ALBEDO', 'd', '', 'lake_ice_snow_albedo',
     'lake_ice_snow_albedo', '1',
     'albedo of lake snow'),
    ('STATE_LAKE_ICE_SNOW_DEPTH', 'd', '', 'lake_ice_snow_depth',
     'lake_ice_snow_depth', 'm',
     'depth of snow on lake ice'),
)


def variables_for(case):
    """The STATE_* variables VIC defines for this run's options."""
    v = list(VARIABLES)
    if case.options.get('carbon'):
        v[3:3] = [x[0] for x in _VARS_CARBON]
    if case.options.get('lakes') is not None:
        v[-1:-1] = [x[0] for x in _VARS_LAKE]
    return tuple(v)

_DIM_NAMES = {'veg': 'veg_class', 'band': 'snow_band', 'layer': 'nlayer',
              'frost': 'frost_area', 'node': 'soil_node'}

_REFERENCES = (
    'Primary Historical Reference for VIC: Liang, X., D. P. Lettenmaier, '
    'E. F. Wood, and S. J. Burges, 1994: A Simple hydrologically Based Model '
    'of Land Surface Water and Energy Fluxes for GSMs, J. Geophys. Res., '
    '99(D7), 14,415-14,428.')


def state_file_name(statename, when):
    """vic_store.c: `<STATENAME>.<YYYYMMDD>_<SSSSS>.nc`."""
    return '%s.%04d%02d%02d_%05d.nc' % (
        statename, when.year, when.month, when.day,
        when.hour * 3600 + when.minute * 60 + when.second)


def _days(when):
    """date2num for TIME_UNITS_DAYS from 0001-01-01 (proleptic Gregorian)."""
    d = when - dt.datetime(1, 1, 1)
    return d.days + d.seconds / 86400.0


def refuse_unsupported(case):
    """The configurations whose VIC state this module does not cover."""
    o = case.options
    why = []
    if why:
        raise ValueError('a VIC state file (STATENAME / INIT_STATE) is not '
                         'covered for: ' + '; '.join(why))


def colmap(case):
    """The run's columns as (cell, band) of the file: one column per active
    cell without snow bands, one per (cell, band with area) with them
    (core/snowbands.py). Returns dict(cell, band, n_band, n_cell)."""
    ex = case.snowbands
    n = int(case.dom.n_active)
    if ex is None:
        return dict(cell=np.arange(n), band=np.zeros(n, dtype=int),
                    n_band=1, n_cell=n)
    return dict(cell=np.asarray(ex['cell']), band=np.asarray(ex['band']),
                n_band=int(ex['n_band']), n_cell=n)


def _grid(case):
    dom = case.dom
    return (np.asarray(dom.rows), np.asarray(dom.cols), dom.ny, dom.nx)


def _node_grid(case, nnode):
    """dz_node and node_depth, [node, cell]: vic_init.c:833-945."""
    import torch
    from ..core import soiltemp
    th = case.soil.thermal
    if case.options.get('quick_flux', True):
        # vic_init.c:836-846: three nodes, the rest stay 0
        d0, dp = th['depth'][:, 0], th['dp']
        zsum = torch.zeros(d0.shape[0], nnode, dtype=d0.dtype)
        dz = torch.zeros_like(zsum)
        dz[:, 0], dz[:, 1] = d0, d0
        dz[:, 2] = 2.0 * (dp - 1.5 * d0)
        zsum[:, 1], zsum[:, 2] = d0, dp
    else:
        zsum, dz = soiltemp.node_grid(
            th['depth'], th['dp'], nnode,
            bool(case.options.get('exp_trans', False)))
    cm = colmap(case)
    first = np.array([np.nonzero(cm['cell'] == c)[0][0]
                      for c in range(cm['n_cell'])])
    return (dz.detach().cpu().double().numpy()[first].T,
            zsum.detach().cpu().double().numpy()[first].T)


def collect(case, out):
    """The state-file arrays, each [cell, tile(, k)] numpy, from a run's
    out (simulate(return_state=True)) of `case`."""
    import torch
    refuse_unsupported(case)
    st = out['state']
    aux = out.get('state_aux') or {}
    nn_gp = _nodes(case)
    for k in ('snow_flux', 'long_under_out', 'cold_content'):
        if k not in aux:
            raise ValueError('a VIC state file needs the last step\'s %s, '
                             'which this run path (graph or lane mode) does '
                             'not keep' % k)
    N = lambda t: t.detach().cpu().double().numpy()      # noqa: E731
    cv = N(case.mosaic.cv)
    nc, nt = cv.shape
    live = cv > 0.0
    s = st['snow']
    moist = N(st['moist'])
    d = {'moist': moist}
    ice = N(s['ice'])
    d['soil_ice'] = (np.zeros(moist.shape + (1,)) if st['soil_ice'] is None
                     else N(st['soil_ice']).reshape(moist.shape + (-1,)))
    d['wdew'] = N(st['wdew'])
    d['last_snow'] = np.rint(N(st['last_snow'])).astype(np.int64)
    d['melting'] = N(st['melting'].to(torch.float64)).astype(np.int64)
    # calc_surf_energy_bal.c:690-696 (no SPATIAL_SNOW): 1 with a pack
    d['coverage'] = (ice > 0.0).astype(float)
    if st['spatial_snow'] is not None:
        # SPATIAL_SNOW: the fraction the pack covers (calc_snow_coverage.c);
        # VIC's file holds no other of the six stores
        d['coverage'] = N(st['spatial_snow']['coverage'])
        for k, _, t in _SP_EXTRA:
            if k not in st['spatial_snow']:
                continue
            v = N(st['spatial_snow'][k].to(torch.float64))
            d['sp_' + k] = np.rint(v).astype(np.int64) if t == 'i' else v
    # snow->swq is the pack's total (snow_melt.c:454): the ice and the two
    # layers' liquid
    d['swe'] = ice + N(s['surf_water']) + N(s['pack_water'])
    d['surf_temp'] = N(s['surf_temp'])
    d['surf_water'] = N(s['surf_water'])
    d['pack_temp'] = N(s['pack_temp'])
    d['pack_water'] = N(s['pack_water'])
    d['density'] = N(s['density'])
    cs = st['canopy']
    d['snow_canopy'] = (N(cs['int_snow']) if cs is not None
                        else np.zeros_like(ice))
    if st['t_node'] is not None:
        d['t_node'] = N(st['t_node'])
    else:
        # FULL_ENERGY FALSE: Tsurf and T1 are carried; the third node is
        # calc_surf_energy_bal.c:574-578's deep-node formula
        th = case.soil.thermal
        t0, t1 = N(st['wb_t']['t0']), N(st['wb_t']['t1'])
        d1 = N(th['depth'])[:, 0:1]
        dp = N(th['dp'])[:, None]
        avg = N(th['avg_temp'])[:, None]
        t2 = avg + (t1 - avg) * np.exp(-(dp - d1) / dp)
        d['t_node'] = np.stack([t0, t1, t2], -1)
    if d['t_node'].shape[-1] < nn_gp:
        # QUICK_FLUX: VIC allocates NODES nodes, the cold start sets every
        # one to avg_temp (generate_default_state.c:95-99) and the run
        # updates three (calc_surf_energy_bal.c:574-578)
        fill = (N(case.soil.thermal['avg_temp']) if case.full_energy
                or case.options.get('frozen_soil') else np.zeros(nc))
        pad = np.repeat(fill.reshape(nc, 1, 1), nt, 1).repeat(
            nn_gp - d['t_node'].shape[-1], 2)
        d['t_node'] = np.concatenate([d['t_node'], pad], -1)
    # the foliage temperature: the canopy's where there is one, else the
    # air temperature of the step the state ends (solve_snow.c:222,232,498,
    # 514); the bare-soil tile keeps what the C left it (see basin.py)
    # (the contracted forcing echoes of a snow-band run cover the whole
    # case, not a window: count the steps in a flux)
    nstep = int(out['swe'].shape[0])
    if 'tair_t' in aux:
        tfol = N(aux['tair_t']).copy()
    else:
        tfol = np.broadcast_to(N(case.atm['tair'])[nstep - 1].reshape(nc, 1),
                               (nc, nt)).copy()
    if cs is not None:
        # only an overstory tile keeps the canopy's own Tfoliage
        # (snow_intercept.c); every other tile's is the air temperature
        ov = case.mosaic.overstory
        om = np.zeros((nc, nt), bool) if ov is None else N(ov) > 0.5
        tfol = np.where(om, N(cs['t_foliage']), tfol)
    bare = case.mosaic.bare_tile
    if bare is not None:
        bm = N(bare).astype(bool) & live
        tfol[bm] = N(aux['tfoliage_bare'])[bm]
    d['tfoliage'] = tfol
    d['cold_content'] = N(aux['cold_content'])
    d['long_under_out'] = N(aux['long_under_out'])
    d['snow_flux'] = N(aux['snow_flux'])
    # calc_gridcell_avg_albedo.c: 1 - swnet/shortwave, else the cover-
    # weighted albedo (OUT_ALBEDO, put_data.c:886-889), per cell
    cm = colmap(case)
    def per_cell(x):
        x = N(x)[nstep - 1]
        if case.snowbands is not None and x.shape[-1] == len(cm['cell']) \
                and x.shape[-1] != cm['n_cell']:
            from ..core import snowbands as _sb
            x = _sb.contract(x, case.snowbands, axis=0)
        return x
    sw = per_cell(out['swdown'])
    # the water-balance path does not form NetShortAtmos: its shortwave
    # balance is the albedo's, so the cell value is the albedo itself
    # (the water-balance rows of the same name, torchvic.model.wbenergy, are
    # reporting rows and leave the state file as it was: eb_energy_error
    # exists only on the energy path)
    swnet = (per_cell(out['eb_net_short'])
             if 'eb_net_short' in out and 'eb_energy_error' in out
             else None)
    alb = per_cell(out['albedo'])
    # vic_run.c:181,358 passes `overstory`, the local the tile loop last set:
    # the last tile with cover (bare soil in the image driver, whose class has
    # no overstory), not the cell's. Without it every tile reports its
    # ground albedo (calc_gridcell_avg_albedo.c:60-75).
    if 'alb_under' in aux:
        first = np.array([np.nonzero(cm['cell'] == c)[0][0]
                          for c in range(cm['n_cell'])])
        ov = case.mosaic.overstory
        last = np.array([np.nonzero(row)[0][-1] if row.any() else 0
                         for row in (cv > 0.0)[first]])
        ov_last = (np.zeros(cm['n_cell'], bool) if ov is None else
                   N(ov)[first][np.arange(cm['n_cell']), last] > 0.5)
        under_col = (N(aux['alb_under']) * cv).sum(1)
        if case.snowbands is not None:
            from ..core import snowbands as _sb
            under = _sb.contract(under_col, case.snowbands, axis=0)
        else:
            under = under_col
        alb = np.where(ov_last, alb, under)
    with np.errstate(divide='ignore', invalid='ignore'):
        d['avg_albedo'] = (np.where(sw > 0.0, 1.0 - swnet / sw, alb)
                           if swnet is not None else alb)
    if st['carbon'] is not None:
        c = st['carbon']
        for k, _ in _CARBON_KEYS:
            d[k] = N(c[k])
        # vic_run.c:301-306 turns the year over after the last step of 31
        # December, so VIC's file holds the turned-over pair; the loop
        # does it before the next year's first step
        if _ends_year(case, st):
            d['annual_npp_prev'] = d['annual_npp'].copy()
            d['annual_npp'] = np.zeros_like(d['annual_npp'])
    sz = st.get('snow_zero')
    if sz:
        # the snow-free fast path's cached zero stand-in: not VIC's, but
        # the first snow-free step after a restart must find it as the
        # single run does
        # nested dicts (snow_step's 'energy' terms, read at basin.py's
        # `se = new_state['energy']`) flattened with dotted keys
        flat = {}

        def _flat(pre, x):
            for k, v in x.items():
                if isinstance(v, dict):
                    _flat(pre + k + '.', v)
                elif torch.is_tensor(v):
                    flat[pre + k] = v
        _flat('state.', sz['state'])
        flat['melt'] = sz['melt']
        d['snow_zero'] = {k: (N(v.to(torch.float64)),
                              'b' if v.dtype == torch.bool else 'd')
                          for k, v in flat.items()}
    lk = st.get('lake')
    if lk is not None:
        d['lake_cells'] = N(case.options['lakes'].cells.double()).astype(int)
        d['lake_tv'] = {k: N(v.to(torch.float64)) for k, v in lk.items()}
        d['lake_tv_kinds'] = {k: str(v.dtype).replace('torch.', '')
                              for k, v in lk.items()}
        d['lake'] = _lake_vic_vars(d['lake_tv'])
    d['cv'] = cv
    rp = st.get('routing_prefix')
    d['routing_tail'] = (None if rp is None or rp.shape[0] == 0
                         else N(rp))
    return d


def _lake_vic_vars(L_):
    """TorchVIC's lake state (core.lake) as VIC's STATE_LAKE_* arrays,
    [lake cell] / [node, lake cell] (vic_store.c lake section)."""
    n = 20                                           # Nlakenode
    B = L_['sarea'].shape[-1]
    z = np.zeros(B)
    nd = lambda x: np.asarray(x)[:n]                 # noqa: E731
    return {
        'STATE_LAKE_SOIL_MOISTURE': None,            # never set: the fill
        'STATE_LAKE_SOIL_ICE': z,
        'STATE_LAKE_SNOW_AGE': L_['last_snow'],
        'STATE_LAKE_SNOW_MELT_STATE': L_['MELTING'],
        'STATE_LAKE_SNOW_COVERAGE': L_['coverage'],
        'STATE_LAKE_SNOW_WATER_EQUIVALENT': L_['swq'],
        'STATE_LAKE_SNOW_SURF_TEMP': L_['surf_temp'],
        'STATE_LAKE_SNOW_SURF_WATER': L_['lake_surf_water'],
        'STATE_LAKE_SNOW_PACK_TEMP': L_['pack_temp'],
        'STATE_LAKE_SNOW_PACK_WATER': L_['lake_pack_water'],
        'STATE_LAKE_SNOW_DENSITY': z,
        'STATE_LAKE_SNOW_COLD_CONTENT': L_['coldcontent'],
        'STATE_LAKE_SNOW_CANOPY': L_['snow_canopy'],
        'STATE_LAKE_SOIL_NODE_TEMP': z,
        'STATE_LAKE_ACTIVE_LAYERS': L_['activenod'],
        'STATE_LAKE_LAYER_DZ': L_['dz'],
        'STATE_LAKE_SURF_LAYER_DZ': L_['surfdz'],
        'STATE_LAKE_DEPTH': L_['ldepth'],
        'STATE_LAKE_LAYER_SURF_AREA': nd(L_['surface']),
        'STATE_LAKE_SURF_AREA': L_['sarea'],
        'STATE_LAKE_VOLUME': L_['volume'],
        'STATE_LAKE_LAYER_TEMP': nd(L_['temp']),
        'STATE_LAKE_AVERAGE_TEMP': L_['tempavg'],
        'STATE_LAKE_ICE_AREA': L_['areai'],
        'STATE_LAKE_ICE_AREA_NEW': L_['new_ice_area'],
        'STATE_LAKE_ICE_WATER_EQUIVALENT': L_['ice_water_eq'],
        'STATE_LAKE_ICE_HEIGHT': L_['hice'],
        'STATE_LAKE_ICE_TEMP': L_['tempi'],
        'STATE_LAKE_ICE_SWE': L_['swe'],
        'STATE_LAKE_ICE_SNOW_SURF_TEMP': L_['surf_temp'],
        'STATE_LAKE_ICE_SNOW_PACK_TEMP': L_['pack_temp'],
        'STATE_LAKE_ICE_SNOW_COLD_CONTENT': z,
        'STATE_LAKE_ICE_SNOW_SURF_WATER': L_['surf_water'],
        'STATE_LAKE_ICE_SNOW_PACK_WATER': L_['pack_water'],
        'STATE_LAKE_ICE_SNOW_ALBEDO': L_['SAlbedo'],
        'STATE_LAKE_ICE_SNOW_DEPTH': L_['sdepth'],
    }


def _ends_year(case, st):
    """The state's last step is the last of 31 December."""
    step = dt.timedelta(seconds=86400.0 / st['steps_per_day'])
    nxt = st['last_date'] + step
    return nxt.month == 1 and nxt.day == 1 and nxt.hour == 0 and \
        nxt.minute == 0 and nxt.second == 0


def write_state_file(path, case, out, when=None):
    """Write the VIC state file of a run's end (or `when`, the end of the
    run's last step, for the record). `out` is the dict of
    simulate(..., return_state=True)."""
    import netCDF4 as nc4
    d = collect(case, out)
    rows, cols, ny, nx = _grid(case)
    cv = d['cv']
    ncol, ntile = cv.shape
    ncell = int(case.dom.n_active)
    nlayer = d['moist'].shape[-1]
    nnode = d['t_node'].shape[-1]
    nfrost = d['soil_ice'].shape[-1]
    nband = colmap(case)['n_band']
    if when is None:
        st = out['state']
        step = 86400.0 / st['steps_per_day']
        when = st['last_date'] + dt.timedelta(seconds=step)
    dom = case.dom
    names = dom.names
    live = cv > 0.0                                     # vidx >= 0
    coords = (np.asarray(dom.ds[names['lat_var']].values, dtype=float),
              np.asarray(dom.ds[names['lon_var']].values, dtype=float))
    with _nc_lock():
        _write(nc4, path, case, out, d, when, rows, cols, ny, nx, cv, live,
               ncell, ntile, nlayer, nnode, nfrost, nband, dom, names, coords)
    return path


def _write(nc4, path, case, out, d, when, rows, cols, ny, nx, cv, live,
           ncell, ntile, nlayer, nnode, nfrost, nband, dom, names, coords):
    try:
        ds = nc4.Dataset(path, 'w', format='NETCDF4')
    except OSError as e:
        # vic_store.c: nc_create fails -> log_err (VIC stops here too)
        raise ValueError('STATENAME: cannot create the state file %s (%s); '
                         'VIC stops on this too (vic_store.c)'
                         % (path, e))
    try:
        ds.set_fill_on()
        ds.createDimension('time', None)
        tail = d['routing_tail']
        if tail is not None:
            # TorchVIC only: the unit-hydrograph tail of its own discharge,
            # which VIC's file cannot carry (a restart without it starts
            # the routing empty). VIC ignores variables it does not read.
            ds.createDimension('tv_route_step', tail.shape[0])
        lake_on = d.get('lake') is not None
        if lake_on:
            ds.createDimension('lake_node', 20)
        ds.createDimension(dom.x_dim, nx)
        ds.createDimension(dom.y_dim, ny)
        sizes = {'veg': ntile, 'band': nband, 'layer': nlayer,
                 'frost': nfrost, 'node': nnode}
        for k, n in sizes.items():
            ds.createDimension(_DIM_NAMES[k], n)
        # global attributes (set_global_nc_attributes)
        ds.title = 'VIC State File'
        ds.source = 'VIC Image Driver state layout, written by TorchVIC'
        ds.history = 'Created by TorchVIC on %s\n' % dt.datetime.now(
        ).strftime('%a %b %d %H:%M:%S %Y')
        ds.references = _REFERENCES
        ds.comment = ('Output from the Variable Infiltration Capacity (VIC) '
                      'Macroscale Hydrologic Model')
        ds.Conventions = 'CF-1.6'
        ds.VIC_Model_Version = '5.1.0'
        ds.VIC_Driver = 'Image'
        t = ds.createVariable('time', 'f8', ('time',))
        t.standard_name = 'time'
        t.units = 'days since 0001-01-01 00:00:00'
        t.calendar = 'proleptic_gregorian'
        lat_a, lon_a = coords
        two_d = lon_a.ndim == 2
        lon_dims = (dom.y_dim, dom.x_dim) if two_d else (dom.x_dim,)
        lat_dims = (dom.y_dim, dom.x_dim) if two_d else (dom.y_dim,)
        v = ds.createVariable(names['lon_var'], 'f8', lon_dims)
        v.long_name, v.units, v.standard_name = (
            'longitude', 'degrees_east', 'longitude')
        v = ds.createVariable(names['lat_var'], 'f8', lat_dims)
        v.long_name, v.units, v.standard_name = (
            'latitude', 'degrees_north', 'latitude')
        for nm, dim, std in (('veg_class', 'veg', 'vegetation_class_number'),
                             ('snow_band', 'band',
                              'snow_elevation_band_number'),
                             ('layer', 'layer', 'soil_layer_number'),
                             ('frost_area', 'frost', 'frost_area_number')):
            v = ds.createVariable(nm, 'i4', (_DIM_NAMES[dim],))
            v.long_name, v.standard_name = nm, std
        if lake_on:
            v = ds.createVariable('lake_node', 'i4', ('lake_node',))
            v.long_name, v.standard_name = 'lake_node', 'lake_node_number'
        for nm, std in (('dz_node', 'soil_thermal_node_spacing'),
                        ('node_depth', 'soil_thermal_node_depth')):
            v = ds.createVariable(nm, 'f8', ('soil_node', dom.y_dim,
                                             dom.x_dim))
            v.long_name, v.standard_name, v.units = nm, std, 'm'
        carbon_on = d.get('annual_npp') is not None
        _dims_all = dict(_DIM_NAMES, lake='lake_node')
        for (name, typ, dims, ln, sn, un, desc) in (
                _VARS[:-1] + (_VARS_CARBON if carbon_on else ())
                + (_VARS_LAKE if lake_on else ()) + _VARS[-1:]):
            dn = tuple(_dims_all[k] for k in dims.split())
            full = dn + (dom.y_dim, dom.x_dim)
            if typ == 'd':
                v = ds.createVariable(name, 'f8', full, fill_value=FILL_D)
            else:
                v = ds.createVariable(name, 'i4', full, fill_value=FILL_I)
            v.long_name, v.standard_name, v.units, v.description = (
                ln, sn, un, desc)
        for k, nm, ty in _SP_EXTRA:
            if 'sp_' + k in d:
                v = ds.createVariable(
                    nm, 'f8' if ty == 'd' else 'i4',
                    ('veg_class', 'snow_band', dom.y_dim, dom.x_dim),
                    fill_value=FILL_D if ty == 'd' else FILL_I)
                v.long_name = nm.lower()
                v.description = ('TorchVIC only: SPATIAL_SNOW store %s that '
                                 'VIC does not write (not read by VIC)' % k)
        if tail is not None:
            v = ds.createVariable('TV_ROUTING_TAIL', 'f8',
                                  ('tv_route_step', 'snow_band', dom.y_dim,
                                   dom.x_dim), fill_value=FILL_D)
            v.long_name = 'tv_routing_tail'
            v.units = 'mm'
            v.description = ('TorchVIC only: runoff + baseflow of the last '
                             'steps, the tail its unit hydrograph still '
                             'convolves (not read by VIC)')
        for k, (arr, kind) in (d.get('snow_zero') or {}).items():
            v = ds.createVariable(
                'TV_SNOWZERO_' + k.replace('.', '_').upper(), 'f8',
                ('veg_class', 'snow_band', dom.y_dim, dom.x_dim),
                fill_value=FILL_D)
            v.key = k
            v.kind = kind
        if lake_on:
            ds.createDimension('tv_lake_cell', len(d['lake_cells']))
            for k, arr in d['lake_tv'].items():
                dn = ('tv_lake_cell',)
                if arr.ndim == 2:
                    dname = 'tv_lake_n%d' % arr.shape[0]
                    if dname not in ds.dimensions:
                        ds.createDimension(dname, arr.shape[0])
                    dn = (dname,) + dn
                v = ds.createVariable('TV_LAKE_' + k.upper(), 'f8', dn)
                v.key = k
                v.description = ('TorchVIC only: the lake state %s that '
                                 'round-trips the run exactly (not read by '
                                 'VIC); kind %s' % (k, d['lake_tv_kinds'][k]))
        # data
        t[0] = _days(when)
        ds[names['lon_var']][:] = lon_a
        ds[names['lat_var']][:] = lat_a
        ds['veg_class'][:] = np.arange(1, ntile + 1)
        ds['snow_band'][:] = np.arange(nband)
        ds['layer'][:] = np.arange(nlayer)
        ds['frost_area'][:] = np.arange(nfrost)
        dz, zs = _node_grid(case, nnode)
        for nm, arr in (('dz_node', dz), ('node_depth', zs)):
            full = np.full((nnode, ny, nx), FILL_D)
            full[:, rows, cols] = arr
            ds[nm][:] = full

        cm = colmap(case)
        ccell, cband = cm['cell'], cm['band']
        # a tile a cell has (vidx >= 0) exists in every band: bands without
        # area were never run and hold the zeros VIC allocated
        cell_live = np.zeros((ncell, ntile), bool)
        np.logical_or.at(cell_live, ccell, live)

        def write(name, a, fill):
            """a [column, tile, *k] -> the file's [tile, band, *k, y, x];
            tiles a cell does not have (vidx < 0) keep the fill value."""
            a = np.asarray(a)
            k = a.shape[2:]
            tgt = np.full((ntile, nband) + k + (ny, nx), fill,
                          dtype=np.int64 if a.dtype.kind == 'i' else float)
            lv = cell_live.T.reshape((ntile, 1) + (1,) * len(k) + (ncell,))
            zero = np.zeros((), dtype=tgt.dtype)
            tgt[..., rows, cols] = np.where(lv, zero, tgt[..., rows, cols])
            for b in range(nband):
                sel = np.nonzero(cband == b)[0]
                if not len(sel):
                    continue
                mv = np.moveaxis(a[sel], 0, -1)          # [tile, *k, nsel]
                lvb = live[sel].T.reshape((ntile,) + (1,) * len(k)
                                          + (len(sel),))
                view = tgt[:, b]
                r_, c_ = rows[ccell[sel]], cols[ccell[sel]]
                view[..., r_, c_] = np.where(lvb, mv, fill)
            ds[name][:] = tgt
        write('STATE_SOIL_MOISTURE', d['moist'], FILL_D)
        write('STATE_SOIL_ICE', d['soil_ice'], FILL_D)
        write('STATE_CANOPY_WATER', d['wdew'], FILL_D)
        for k, nm, ty in _SP_EXTRA:
            if 'sp_' + k in d:
                write(nm, d['sp_' + k], FILL_D if ty == 'd' else FILL_I)
        if carbon_on:
            for k, nm in _CARBON_KEYS:
                write(nm, d[k], FILL_D)
        write('STATE_SNOW_AGE', d['last_snow'], FILL_I)
        write('STATE_SNOW_MELT_STATE', d['melting'], FILL_I)
        write('STATE_SNOW_COVERAGE', d['coverage'], FILL_D)
        write('STATE_SNOW_WATER_EQUIVALENT', d['swe'], FILL_D)
        write('STATE_SNOW_SURF_TEMP', d['surf_temp'], FILL_D)
        write('STATE_SNOW_SURF_WATER', d['surf_water'], FILL_D)
        write('STATE_SNOW_PACK_TEMP', d['pack_temp'], FILL_D)
        write('STATE_SNOW_PACK_WATER', d['pack_water'], FILL_D)
        write('STATE_SNOW_DENSITY', d['density'], FILL_D)
        write('STATE_SNOW_COLD_CONTENT', d['cold_content'], FILL_D)
        write('STATE_SNOW_CANOPY', d['snow_canopy'], FILL_D)
        write('STATE_SOIL_NODE_TEMP', d['t_node'], FILL_D)
        write('STATE_FOLIAGE_TEMPERATURE', d['tfoliage'], FILL_D)
        write('STATE_ENERGY_LONGUNDEROUT', d['long_under_out'], FILL_D)
        write('STATE_ENERGY_SNOW_FLUX', d['snow_flux'], FILL_D)
        if tail is not None:
            full = np.full((tail.shape[0], nband, ny, nx), FILL_D)
            for b in range(nband):
                sel = np.nonzero(cband == b)[0]
                if len(sel):
                    full[:, b][..., rows[ccell[sel]], cols[ccell[sel]]] = \
                        tail[:, sel]
            ds['TV_ROUTING_TAIL'][:] = full
        for k, (arr, kind) in (d.get('snow_zero') or {}).items():
            write('TV_SNOWZERO_' + k.replace('.', '_').upper(), arr, FILL_D)
        if lake_on:
            ds['lake_node'][:] = np.arange(20)
            lr, lc = rows[d['lake_cells']], cols[d['lake_cells']]
            for (name, typ, dims, ln, sn, un, desc) in _VARS_LAKE:
                v = d['lake'][name]
                dims_ = dims.split()
                shape = tuple({'layer': nlayer, 'frost': nfrost,
                               'node': nnode, 'lake': 20}[k]
                              for k in dims_) + (ny, nx)
                fill = FILL_I if typ == 'i' else FILL_D
                full = np.full(shape, fill, dtype=np.int64 if typ == 'i'
                               else float)
                if v is None:                     # the fill (never set)
                    pass
                else:
                    # cells without a lake hold the initialised zeros
                    full[...] = 0
                    vv = np.asarray(v).astype(full.dtype)
                    if vv.ndim == 1 and len(shape) > 2:
                        vv = np.broadcast_to(
                            vv, shape[:-2] + (len(lr),))
                    full[..., lr, lc] = vv
                ds[name][:] = full
            for k, arr in d['lake_tv'].items():
                ds['TV_LAKE_' + k.upper()][:] = arr
        alb = np.full((ny, nx), FILL_D)
        alb[rows, cols] = d['avg_albedo']
        ds['STATE_AVG_ALBEDO'][:] = alb
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# STATENAME / STATEYEAR / STATEMONTH / STATEDAY / STATESEC: when to save
# ---------------------------------------------------------------------------
def save_plan(case):
    """None when the global file asks for no state file; else
    (n_steps, when, path): the state is the model after `n_steps` steps, at
    `when` (check_save_state_flag.c: the end of the step equals the
    STATE date), written to `<STATENAME>.<YYYYMMDD>_<SSSSS>.nc`.
    Raises ValueError where VIC's get_global_param.c:875-909 stops."""
    gp = case.gp
    name = gp.get('STATENAME') if gp is not None else None
    if not name:
        return None
    y, m, d = (gp.get_int('STATEYEAR') or 0, gp.get_int('STATEMONTH') or 0,
               gp.get_int('STATEDAY') or 0)
    sec = gp.get_int('STATESEC') or 0
    if not (y and m and d):
        raise ValueError(
            'Incomplete specification of the date to save state for state '
            'file (%s): STATEYEAR, STATEMONTH and STATEDAY must be set '
            '(get_global_param.c:882-889)' % name)
    try:
        when = dt.datetime(y, m, d) + dt.timedelta(seconds=sec)
    except ValueError:
        raise ValueError('Unusual specification of the date to save state '
                         'for state file (%s): %04d-%02d-%02d-%05d '
                         '(get_global_param.c:893-908)' % (name, y, m, d, sec))
    if gp.get_bool('STATENAME_CESM'):
        raise ValueError('STATENAME_CESM is the CESM driver\'s file name '
                         'and is not covered')
    step = dt.timedelta(seconds=86400.0 / case.model_steps)
    ends = [_as_datetime(x) + step for x in case.dates]
    if when not in ends:
        return (None, when, state_file_name(name, when))
    return (ends.index(when) + 1, when, state_file_name(name, when))


def _as_datetime(x):
    if isinstance(x, dt.datetime):
        return x
    if hasattr(x, 'to_pydatetime'):
        return x.to_pydatetime()
    return dt.datetime.fromisoformat(str(np.datetime64(x, 's')))


def save_state(case, plan, out=None, log=print, p=None):
    """Write the planned state. `out` is the finished run's output when the
    run was made with return_state=True and ended at the state's step;
    otherwise the first `n` steps are run for it (the run itself is not
    touched). Returns the path, or None when the date is never a step end."""
    import torch
    n, when, path = plan
    if n is None:
        log('STATENAME: %s is not the end of any step of this run; VIC '
            'writes no state file then (check_save_state_flag.c)' % when)
        return None
    refuse_unsupported(case)
    if out is None or n != case.n_steps:
        from ..model import basin
        from ..model.state import time_window
        from ..core import version as core_version
        if case.core is not None:
            core_version.set_core(case.core)
        kw = time_window(case.kwargs(p=p), 0, n)
        with torch.inference_mode():
            out = basin.simulate(**kw, return_state=True)
        if case.snowbands is not None:
            out = case.contract_bands(out)
    write_state_file(path, case, out, when=when)
    log('wrote state %s' % path)
    return path


# ---------------------------------------------------------------------------
# INIT_STATE: vic_restore.c
# ---------------------------------------------------------------------------
def _check_dims(ds, case, nlayer, nfrost, nnode, refs):
    """check_init_state_file (vic_restore.c:885-1100): dimension lengths,
    then longitudes and latitudes to 1e-4."""
    dom = case.dom
    ntile = int(case.mosaic.cv.shape[1])
    want = ((dom.x_dim, dom.nx, 'grid columns'),
            (dom.y_dim, dom.ny, 'grid rows'),
            ('veg_class', ntile, 'veg classes'),
            ('snow_band', colmap(case)['n_band'], 'snow bands'),
            ('nlayer', nlayer, 'soil layers'),
            ('frost_area', nfrost, 'frost areas'),
            ('soil_node', nnode, 'soil nodes'))
    for name, n, what in want:
        if name not in ds.dimensions or len(ds.dimensions[name]) != n:
            got = len(ds.dimensions[name]) if name in ds.dimensions else None
            raise ValueError('Number of %s in state file (%s) does not '
                             'match the parameter file (%d)' % (what, got, n))
    names = dom.names
    for key, ref in (('lon_var', refs[0]), ('lat_var', refs[1])):
        nm = names[key]
        if nm not in ds.variables:
            raise ValueError('Unable to find variable "%s" in the state file'
                             % nm)
        got = np.asarray(ds[nm][:], dtype=float)
        if got.shape != np.asarray(ref).shape or \
                not np.allclose(got, ref, rtol=0.0, atol=1e-4):
            raise ValueError('%s in the state file do not match the '
                             'parameter file' % ('Longitudes' if 'lon' in key
                                                 else 'Latitudes'))


def read_state_file(path, case):
    """vic_restore.c: the arrays of a VIC state file for `case`'s active
    cells, each [cell, tile(, k)] as float64/int64 numpy, with the tiles a
    cell does not have (cover 0) set to None-equivalent zeros. Raises
    ValueError where VIC's check_init_state_file stops, and for a file that
    lacks a tile the model has."""
    import netCDF4 as nc4
    refuse_unsupported(case)
    cv = case.mosaic.cv.detach().cpu().double().numpy()
    live = cv > 0.0
    rows, cols, ny, nx = _grid(case)
    names = case.dom.names
    refs = (np.asarray(case.dom.ds[names['lon_var']].values),
            np.asarray(case.dom.ds[names['lat_var']].values))
    with _nc_lock():
        return _read(nc4, path, case, live, rows, cols, refs)


def _columns(x, cm, rows, cols):
    """A file array [tile, band, *k, y, x] as the run's columns
    [column, tile, *k]."""
    ncol = len(cm['cell'])
    res = None
    for b in range(x.shape[1]):
        sel = np.nonzero(cm['band'] == b)[0]
        if not len(sel):
            continue
        v = np.moveaxis(x[:, b][..., rows[cm['cell'][sel]],
                                cols[cm['cell'][sel]]], -1, 0)
        if res is None:
            res = np.zeros((ncol,) + v.shape[1:])
        res[sel] = v
    return res


def _read(nc4, path, case, live, rows, cols, refs):
    cm = colmap(case)
    ds = nc4.Dataset(path)
    try:
        nlayer = int(case.soil.max_moist.shape[-1])
        nfrost = int(case.options.get('n_frost', 1))
        nnode = _nodes(case)
        _check_dims(ds, case, nlayer, nfrost, nnode, refs)
        names = variables_for(case)
        missing = [v for v in names if v not in ds.variables]
        if missing:
            raise ValueError('state file %s lacks %s' % (path, missing))
        out = {}

        def get(name):
            x = np.ma.filled(ds[name][:].astype(float), np.nan)
            return x
        lake_names = [v[0] for v in _VARS_LAKE]
        for name in names:
            if name in lake_names:
                continue
            x = get(name)
            if name == 'STATE_AVG_ALBEDO':
                out[name] = x[rows, cols]
                continue
            a = _columns(x, cm, rows, cols)               # [col, tile, *k]
            lv = live.reshape(live.shape + (1,) * (a.ndim - 2))
            bad = lv & ~np.isfinite(a)
            if bad.any():
                c, m = np.argwhere(bad.reshape(bad.shape[0], bad.shape[1],
                                               -1).any(-1))[0]
                raise ValueError(
                    '%s has no value for veg tile %d of active cell %d, '
                    'which the parameter file covers' % (name, m + 1, c))
            out[name] = np.where(lv, a, 0.0)
        for k, nm, t in _SP_EXTRA:
            if nm in ds.variables:
                x = np.ma.filled(ds[nm][:].astype(float), np.nan)
                a = _columns(x, cm, rows, cols)
                out['sp_' + k] = np.where(live, a, 0.0)
        sz = {}
        for v in ds.variables:
            if v.startswith('TV_SNOWZERO_'):
                x = np.ma.filled(ds[v][:].astype(float), np.nan)
                sz[ds[v].key] = (np.where(live, _columns(x, cm, rows, cols),
                                          0.0), ds[v].kind)
        out['snow_zero'] = sz or None
        if case.options.get('lakes') is not None:
            cells = np.asarray(case.options['lakes'].cells.cpu())
            lr, lc = rows[cells], cols[cells]
            lv = {}
            for name in lake_names:
                if name == 'STATE_LAKE_SOIL_MOISTURE':
                    # VIC reads it as it is: the fill, never written
                    x = np.ma.getdata(ds[name][:]).astype(float)
                else:
                    x = np.ma.filled(ds[name][:].astype(float), np.nan)
                lv[name] = x[..., lr, lc]
            out['lake_vic'] = lv
            tv = {}
            for v in ds.variables:
                if v.startswith('TV_LAKE_'):
                    kind = ds[v].description.rsplit('kind ', 1)[1]
                    tv[ds[v].key] = (
                        np.ma.filled(ds[v][:].astype(float), np.nan), kind)
            out['lake_tv'] = tv or None
        out['time'] = float(ds['time'][0])
        if 'TV_ROUTING_TAIL' in ds.variables:
            x = np.ma.filled(ds['TV_ROUTING_TAIL'][:].astype(float), np.nan)
            if x.ndim == 3:                     # a file written before bands
                x = x[:, None]
            tl = np.zeros((x.shape[0], len(cm['cell'])))
            for b in range(x.shape[1]):
                sel = np.nonzero(cm['band'] == b)[0]
                if len(sel):
                    tl[:, sel] = x[:, b][..., rows[cm['cell'][sel]],
                                         cols[cm['cell'][sel]]]
            out['routing_tail'] = tl
    finally:
        ds.close()
    return out


def initial_state_from_file(path, case, device=None, dtype=None):
    """`initial_state` for basin.simulate: the file's fields, which the run
    lays over its own cold start (the fields a VIC state file does not hold
    take what VIC's allocation gives them: zeros)."""
    import torch
    f = read_state_file(path, case)
    dtype = dtype or case.dtype
    device = device or case.device
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype,   # noqa: E731
                                  device=device)
    sw, pw = T(f['STATE_SNOW_SURF_WATER']), T(f['STATE_SNOW_PACK_WATER'])
    cv = case.mosaic.cv
    live = cv > 0.0
    bare = case.mosaic.bare_tile
    tf = f['STATE_FOLIAGE_TEMPERATURE']
    tail = f.get('routing_tail')
    st = dict(
        source='vic_file', path=path, time=f['time'],
        routing_prefix=None if tail is None else T(tail),
        moist=T(f['STATE_SOIL_MOISTURE']),
        soil_ice=T(f['STATE_SOIL_ICE']),
        wdew=T(f['STATE_CANOPY_WATER']),
        last_snow=T(f['STATE_SNOW_AGE']),
        melting=T(f['STATE_SNOW_MELT_STATE']) > 0.5,
        coverage=T(f['STATE_SNOW_COVERAGE']),
        swq=T(f['STATE_SNOW_WATER_EQUIVALENT']),
        surf_water=sw, pack_water=pw,
        surf_temp=T(f['STATE_SNOW_SURF_TEMP']),
        pack_temp=T(f['STATE_SNOW_PACK_TEMP']),
        density=T(f['STATE_SNOW_DENSITY']),
        cold_content=T(f['STATE_SNOW_COLD_CONTENT']),
        snow_canopy=T(f['STATE_SNOW_CANOPY']),
        t_node=T(f['STATE_SOIL_NODE_TEMP']),
        t_foliage=T(tf),
        long_under_out=T(f['STATE_ENERGY_LONGUNDEROUT']),
        snow_flux=T(f['STATE_ENERGY_SNOW_FLUX']),
        avg_albedo=T(f['STATE_AVG_ALBEDO']),
        live=live)
    if any('sp_' + k in f for k, _, _ in _SP_EXTRA):
        # TorchVIC's own file (one written before adv_sens lacks it, and
        # that store then starts at zero, as VIC's restart starts it)
        st['sp_extra'] = {k: (T(f['sp_' + k]) > 0.5) if t == 'i'
                          else T(f['sp_' + k]) for k, _, t in _SP_EXTRA
                          if 'sp_' + k in f}
    if f.get('snow_zero'):
        z = {}
        for k, (arr, kind) in f['snow_zero'].items():
            tt = T(arr)
            z[k] = tt > 0.5 if kind == 'b' else tt
        zs = {}
        for k, v in z.items():
            if k.startswith('state.'):
                *path, leaf = k[6:].split('.')
                node = zs
                for p_ in path:
                    node = node.setdefault(p_, {})
                node[leaf] = v
        st['snow_zero_extra'] = dict(state=zs, melt=z['melt'])
    if f.get('lake_vic') is not None:
        st['lake_vic'] = f['lake_vic']
        if f.get('lake_tv'):
            # TorchVIC's own file: the whole lake state, exactly; keys are
            # lower-cased on the way, the lake's names are mixed case
            ex = {}
            for key, (arr, kind) in f['lake_tv'].items():
                tt = torch.as_tensor(arr, device=device)
                dt_ = getattr(torch, kind)
                ex[key] = (tt > 0.5) if dt_ == torch.bool else tt.to(dt_)
            st['lake'] = ex
    if case.options.get('carbon'):
        st['carbon'] = {k: T(f[nm]) for k, nm in _CARBON_KEYS}
    return st


def check_start(case, st):
    """VIC does not compare the state's time with the run's start
    (vic_restore.c reads the file as it is); a mismatch is announced."""
    import warnings
    t0 = _days(_as_datetime(case.dates[0]))
    if abs(t0 - st['time']) > 1e-9:
        warnings.warn(
            'INIT_STATE %s is the model at %.6f days since 0001-01-01 and '
            'the run starts at %.6f: VIC starts from it all the same '
            '(vic_restore.c does not compare the two)' % (
                st['path'], st['time'], t0))
