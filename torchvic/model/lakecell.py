"""The lake and wetland tile inside basin.simulate's step loop (LAKES TRUE).

VIC runs a lake as one vegetation tile of a cell (`lake_idx`) split in two:
the lake, a fraction `lakefrac = sarea / basin[0]` of the tile, and the
wetland, the rest. Each step (vic_run.c 5.1.0:132-466):

  1. the wetland runs the ordinary land physics at Cv * (1 - lakefrac)
     (:136-160); every other tile runs as without lakes
  2. the wetland's runoff and baseflow, and `rpercent` of every other
     tile's, are taken off the land and handed to the lake in m3
     (:368-415)
  3. solve_lake, then water_balance (:416-466), which moves the
     lake/wetland boundary and restates the wetland's stores over its new
     area (lakes.eb.c:1790-2292)

and put_data.c 5.1.0:148-400 reports the cell as the land tiles, the
wetland at Cv * (1 - Clake) and the lake at Cv * Clake, Clake the END of
step fraction, plus the OUT_LAKE_* variables.

`LakeCells` holds the lake state and does steps 1-3 around basin's own
land step: `before(t)` sets the wetland's cover, `after(...)` runs the lake,
writes the wetland's new stores back into the land state and returns the
additive corrections to the cell's rows. The physics is all in
torchvic.core.lake; this is the bookkeeping that ties it to the tiles.

One limitation, carried from core.lake.solve_lake: every lake cell in a run
must have the same number of active lake nodes each step (lakes of one
depth class). A run whose lakes diverge raises rather than solving some of
them on the wrong grid.
"""
import numpy as np
import torch

from ..core import lake as L
from ..core.snow import CONST_LATICE

MM_PER_M = 1000.0
CM_PER_M = 100.0
CONST_TKFRZ = 273.15

#: VIC's OUT_LAKE_* names -> keys of LakeCells' per-step rows.
OUT_LAKE = {
    'OUT_LAKE_AREA_FRAC': 'lake_area_frac', 'OUT_LAKE_DEPTH': 'lake_depth',
    'OUT_LAKE_ICE': 'lake_ice', 'OUT_LAKE_ICE_FRACT': 'lake_ice_fract',
    'OUT_LAKE_ICE_HEIGHT': 'lake_ice_height',
    'OUT_LAKE_ICE_TEMP': 'lake_ice_temp', 'OUT_LAKE_MOIST': 'lake_moist',
    'OUT_LAKE_SURF_AREA': 'lake_surf_area', 'OUT_LAKE_SURF_TEMP':
    'lake_surf_temp', 'OUT_LAKE_SWE': 'lake_swe',
    'OUT_LAKE_SWE_V': 'lake_swe_v', 'OUT_LAKE_VOLUME': 'lake_volume',
    'OUT_LAKE_BF_IN': 'lake_bf_in', 'OUT_LAKE_BF_IN_V': 'lake_bf_in_v',
    'OUT_LAKE_BF_OUT': 'lake_bf_out', 'OUT_LAKE_BF_OUT_V': 'lake_bf_out_v',
    'OUT_LAKE_CHAN_IN': 'lake_chan_in', 'OUT_LAKE_CHAN_IN_V':
    'lake_chan_in_v', 'OUT_LAKE_CHAN_OUT': 'lake_chan_out',
    'OUT_LAKE_CHAN_OUT_V': 'lake_chan_out_v', 'OUT_LAKE_DSTOR':
    'lake_dstor', 'OUT_LAKE_DSTOR_V': 'lake_dstor_v',
    'OUT_LAKE_DSWE': 'lake_dswe', 'OUT_LAKE_DSWE_V': 'lake_dswe_v',
    'OUT_LAKE_EVAP': 'lake_evap', 'OUT_LAKE_EVAP_V': 'lake_evap_v',
    'OUT_LAKE_PREC_V': 'lake_prec_v', 'OUT_LAKE_RCHRG': 'lake_rchrg',
    'OUT_LAKE_RCHRG_V': 'lake_rchrg_v', 'OUT_LAKE_RO_IN': 'lake_ro_in',
    'OUT_LAKE_RO_IN_V': 'lake_ro_in_v', 'OUT_LAKE_VAPFLX': 'lake_vapflx',
    'OUT_LAKE_VAPFLX_V': 'lake_vapflx_v', 'OUT_SURFSTOR': 'surfstor'}


def _active_columns(cv_row):
    """The mosaic columns of a cell's tiles in VIC's veg_con order: the
    classes with cover, in class order (vic_init.c 5.1.0:150-196)."""
    return [j for j, c in enumerate(cv_row) if c > 0.0]


def read_image_lakes(ds, domain, mosaic, soil_params, lake_profile,
                     cell_area, core='5.1.0', dtype=torch.float64,
                     device=None, log=None):
    """The image parameter file's lake fields, checked as vic_init.c
    5.1.0:1336-1550 checks them, as a LakeCells (or None: no cell has a
    lake).

    LAKE_PROFILE FALSE: VIC 5.1.0 reads basin_depth and basin_area with a
    two-dimensional start and count (vic_init.c:1463-1476) whatever their
    shape. On a parameter file whose basin variables carry the lake_node
    dimension -- UW-Hydro's FindleyLake is one -- that reads the wrong
    slab: z[0] comes back 0 and VIC stops at :1489 ("mindepth ... must be
    <= max depth of 0"). Reproduced: such a file is refused the same way.
    """
    import numpy as np
    say = log or (lambda *a: None)
    take = domain.take
    if 'lake_idx' not in ds:
        raise ValueError('LAKES TRUE needs lake_idx in the parameter file '
                         '(vic_init.c 5.1.0:1339)')
    idx = np.asarray(take(ds['lake_idx'])).astype(int)
    cv = mosaic.cv.detach().cpu().numpy()
    ncell = cv.shape[0]
    nveg = [len(_active_columns(cv[i])) for i in range(ncell)]
    for i in range(ncell):
        if not (-1 <= idx[i] < nveg[i]):
            raise ValueError('cell %d lake_idx is %d but we must have -1 <= '
                             'lake_idx < Nveg (%d) (vic_init.c 5.1.0:1346)'
                             % (i, idx[i], nveg[i]))
    cells = [i for i in range(ncell) if idx[i] >= 0]
    if not cells:
        say('LAKES TRUE: no cell has a lake (lake_idx -1 everywhere)')
        return None
    f2 = {k: np.asarray(take(ds[k])) for k in ('numnod', 'mindepth', 'wfrac',
                                                'depth_in', 'rpercent')}
    bd, ba = ds['basin_depth'], ds['basin_area']
    if not lake_profile and 'lake_node' in bd.dims:
        raise ValueError(
            'LAKE_PROFILE FALSE with basin_depth on the lake_node dimension: '
            'VIC 5.1.0 reads it with a 2-D start (vic_init.c:1463-1476), '
            'gets a max depth of 0 and stops (vic_init.c:1489, "lake basin '
            'mindepth is %g but must be <= max depth of 0.000000")'
            % float(f2['mindepth'][cells[0]]))
    if lake_profile:
        z_all = np.stack([np.asarray(take(bd.isel(lake_node=j)))
                          for j in range(bd.sizes['lake_node'])], 0)
        c_all = np.stack([np.asarray(take(ba.isel(lake_node=j)))
                          for j in range(ba.sizes['lake_node'])], 0)
    else:
        z_all = np.asarray(take(bd))[None]
        c_all = np.asarray(take(ba))[None]
    numnod = sorted(set(int(f2['numnod'][i]) for i in cells))
    if len(numnod) != 1:
        raise ValueError('LAKES TRUE: the lake cells have %s basin nodes; '
                         'this port solves lakes of one node count per run'
                         % numnod)
    nn = numnod[0]
    if not (0 < nn < L.MAX_LAKE_NODES):
        raise ValueError('lake numnod %d outside 0 < numnod < %d '
                         '(vic_init.c 5.1.0:1363)' % (nn, L.MAX_LAKE_NODES))
    cell_area = np.asarray(cell_area)
    cols, cons, states = [], [], []
    t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    for i in cells:
        col = _active_columns(cv[i])[idx[i]]
        cvl = float(cv[i, col])
        z = np.zeros(nn + 1)
        cl = np.zeros(nn + 1)
        n_read = nn if lake_profile else 1
        z[:n_read] = z_all[:n_read, i]
        cl[:n_read] = c_all[:n_read, i]
        mind = float(f2['mindepth'][i])
        # vic_init.c 5.1.0:1481-1530
        if not z[0] > 0:
            raise ValueError('cell %d lake basin max depth is %f but must be '
                             '> 0' % (i, z[0]))
        if not mind <= z[0]:
            raise ValueError('cell %d lake basin mindepth is %f but must be '
                             '<= max depth of %f' % (i, mind, z[0]))
        if not (0 < cl[0] <= 1):
            raise ValueError('cell %d lake basin max area fraction is %f but '
                             'must be > 0 and <= 1' % (i, cl[0]))
        if abs(1 - cl[0] / cvl) > 0.01:
            raise ValueError('cell %d lake basin max area fraction is %f but '
                             'must equal the area fraction of the lake tile '
                             '(%f)' % (i, cl[0], cvl))
        cl[0] = cvl
        if lake_profile:
            for j in range(1, nn):
                if not (0 < z[j] < z[j - 1]):
                    raise ValueError('cell %d lake basin node %d depth %f must '
                                     'be > 0 and < node %d\'s %f'
                                     % (i, j, z[j], j - 1, z[j - 1]))
                if not (0 < cl[j] < cl[j - 1]):
                    raise ValueError('cell %d lake basin node %d area fraction '
                                     '%f must be > 0 and < node %d\'s %f'
                                     % (i, j, cl[j], j - 1, cl[j - 1]))
        con = L.compute_lake_params(t(z), t(cl), nn, t(float(cell_area[i])),
                                    t(mind), float(f2['rpercent'][i]),
                                    lake_profile)
        con['depth_in'] = float(f2['depth_in'][i])
        con['wfrac'] = float(f2['wfrac'][i])
        cons.append(con)
        cols.append(col)
        states.append(L.initial_lake_state(
            con, float(soil_params.thermal['avg_temp'][i]), dtype=dtype,
            device=device, vic_core=core))
    say('LAKES TRUE: %d lake cell(s), %d basin nodes, lake tile column(s) %s'
        ', initial depth %s m (VIC %s)'
        % (len(cells), nn, sorted(set(cols)),
           ', '.join('%g' % float(s['ldepth'][0]) for s in states), core))
    return LakeCells(cells, cols, cons, states, cell_area, core,
                     dtype=dtype, device=device)


def _stack_dicts(ds_, axis=-1):
    out = {}
    for k in ds_[0]:
        v = [d[k] for d in ds_]
        if torch.is_tensor(v[0]):
            out[k] = torch.cat([x.reshape(*x.shape[:-1], -1)
                                if x.dim() else x.reshape(1) for x in v],
                               dim=-1) if v[0].dim() else torch.stack(v)
        else:
            out[k] = v[0]
    return out


class LakeCells(object):
    """The lakes of a run, batched over the cells that have one."""

    def __init__(self, cells, cols, cons, states, cell_area, core,
                 dtype=torch.float64, device=None):
        self.core = core
        self.dtype, self.device = dtype, device
        self.cells = torch.as_tensor(cells, dtype=torch.long, device=device)
        self.cols = torch.as_tensor(cols, dtype=torch.long, device=device)
        con = {}
        for k in ('z', 'basin', 'Cl'):
            con[k] = torch.stack([c[k] for c in cons], -1)
        for k in ('maxdepth', 'maxvolume', 'mindepth', 'minvolume',
                  'basin0'):
            con[k] = torch.stack([torch.as_tensor(c[k], dtype=dtype,
                                                  device=device).reshape(())
                                  for c in cons])
        for k in ('rpercent', 'wfrac', 'depth_in'):
            con[k] = torch.as_tensor([float(c[k]) for c in cons],
                                     dtype=dtype, device=device)
        con['numnod'] = cons[0]['numnod']
        self.con = con
        self.cell_area = torch.as_tensor(
            [float(cell_area[i]) for i in cells], dtype=dtype, device=device)
        st = {}
        for k in states[0]:
            v = [s[k] for s in states]
            st[k] = torch.cat(v, dim=-1)
        self.state = st
        self.rows = {k: [] for k in OUT_LAKE.values()}
        self.cv0 = None
        self.initial_volume = st['volume'].clone()
        self._state0 = {k: v.clone() for k, v in st.items()}
        #: the global file's WIND_H, which solve_lake takes as gp->wind_h
        #: (vic_run.c 5.1.0:446-452); set by the loader
        self.wind_h = 10.0

    # -------------------------------------------------------------------
    def reset(self):
        """The cold start again: a LakeCells is the run's own state, and a
        second run on the same case must not begin where the first ended."""
        self.state = {k: v.clone() for k, v in self._state0.items()}
        self.initial_volume = self.state['volume'].clone()
        self.cv0 = None
        self.rows = {k: [] for k in OUT_LAKE.values()}

    def export_state(self):
        return {k: v.clone() for k, v in self.state.items()}

    def set_state(self, state):
        """Continue from `state` (export_state's, or state_from_vic's)."""
        self.state = {k: v.clone() for k, v in state.items()}
        # the opening storage the water error is formed against is the
        # restored lake's (initialize_save_data runs put_data on it)
        self.initial_volume = self.state['volume'].clone()

    def state_from_vic(self, f):
        """The lake state vic_restore.c leaves: the file's fields over the
        cold start, then compute_derived_lake_dimensions
        (vic_populate_model_state.c 5.1.0:47-51). `f` maps VIC's lake
        variable names to arrays [lake cell] / [node, lake cell]."""
        st = {k: v.clone() for k, v in self._state0.items()}
        t = lambda x: torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n = L.MAX_LAKE_NODES
        st['ldepth'] = t(f['STATE_LAKE_DEPTH'])
        st['temp'] = t(f['STATE_LAKE_LAYER_TEMP'])[:n]
        for k, nm in (('tempavg', 'STATE_LAKE_AVERAGE_TEMP'),
                      ('areai', 'STATE_LAKE_ICE_AREA'),
                      ('new_ice_area', 'STATE_LAKE_ICE_AREA_NEW'),
                      ('ice_water_eq', 'STATE_LAKE_ICE_WATER_EQUIVALENT'),
                      ('hice', 'STATE_LAKE_ICE_HEIGHT'),
                      ('tempi', 'STATE_LAKE_ICE_TEMP'),
                      ('swe', 'STATE_LAKE_ICE_SWE'),
                      ('surf_temp', 'STATE_LAKE_ICE_SNOW_SURF_TEMP'),
                      ('pack_temp', 'STATE_LAKE_ICE_SNOW_PACK_TEMP'),
                      ('surf_water', 'STATE_LAKE_ICE_SNOW_SURF_WATER'),
                      ('pack_water', 'STATE_LAKE_ICE_SNOW_PACK_WATER'),
                      ('SAlbedo', 'STATE_LAKE_ICE_SNOW_ALBEDO'),
                      ('sdepth', 'STATE_LAKE_ICE_SNOW_DEPTH'),
                      # lake_var.snow, the lake's own snow struct
                      ('swq', 'STATE_LAKE_SNOW_WATER_EQUIVALENT'),
                      ('coverage', 'STATE_LAKE_SNOW_COVERAGE'),
                      ('coldcontent', 'STATE_LAKE_SNOW_COLD_CONTENT'),
                      ('snow_canopy', 'STATE_LAKE_SNOW_CANOPY'),
                      ('lake_surf_water', 'STATE_LAKE_SNOW_SURF_WATER'),
                      ('lake_pack_water', 'STATE_LAKE_SNOW_PACK_WATER')):
            st[k] = t(f[nm])
        st['last_snow'] = torch.as_tensor(f['STATE_LAKE_SNOW_AGE'],
                                          dtype=torch.long, device=self.device)
        st['MELTING'] = torch.as_tensor(f['STATE_LAKE_SNOW_MELT_STATE'],
                                        device=self.device) > 0.5
        # vic_restore.c 5.1.0:526-536 reads STATE_LAKE_SOIL_MOISTURE into
        # lake_var.soil.layer[j].moist, but :666-675 then reads
        # STATE_LAKE_SOIL_NODE_TEMP over the same field (layer[j].moist,
        # not energy.T[j]) for every node j < Nnode >= Nlayer. VIC writes
        # lake_var.energy.T there (vic_store.c:953-966), zeros, so a VIC
        # warm start reports the lake's soil moisture as the node
        # temperatures in the file. Reproduced as VIC runs it.
        if 'STATE_LAKE_SOIL_NODE_TEMP' in f:
            sm = np.array(f['STATE_LAKE_SOIL_MOISTURE'], dtype=float,
                          copy=True)                  # [layer, B], raw
            nt = np.asarray(f['STATE_LAKE_SOIL_NODE_TEMP'], dtype=float)
            nl = sm.shape[0]
            k_ = min(nl, nt.shape[0])
            sm[:k_] = nt[:k_]
            st['soil_moist'] = t(sm)
        c = self.con
        (st['surfdz'], st['dz'], st['activenod'], st['surface'],
         st['sarea'], st['volume']) = L.derived_lake_dimensions(
            st['ldepth'], st['ice_water_eq'], c['z'], c['basin'],
            c['maxdepth'], c['maxvolume'])
        return st

    # -------------------------------------------------------------------
    def lakefrac(self):
        return self.state['sarea'] / self.con['basin0']

    def before(self, mosaic):
        """vic_run.c 5.1.0:136-160: the wetland runs at Cv * (1 -
        lakefrac), lakefrac from the area the lake ended last step with."""
        if self.cv0 is None:
            self.cv0 = mosaic.cv.clone()
            self.cv_lake = self.cv0[self.cells, self.cols]
        self.frac0 = self.lakefrac()
        cv = self.cv0.clone()
        cv[self.cells, self.cols] = self.cv_lake * (1.0 - self.frac0)
        mosaic.cv = cv
        return cv

    def restore(self, mosaic):
        if self.cv0 is not None:
            mosaic.cv = self.cv0

    # -------------------------------------------------------------------
    def after(self, t, tap, land, atm, p, soil_params, wind_h, day_in_year,
              steps_per_day, mosaic, month):
        """The lake step after the land step. `tap` carries the land step's
        per-tile values, `land` the wetland's state (moist [cell, tile,
        layer], wdew, snow_state, canopy_state); returns (land, patches),
        `land` with the wetland's stores restated over its new area and
        `patches` {row: [cell] or [cell, layer] additive correction}."""
        c, k = self.cells, self.cols
        dt = 86400.0 / steps_per_day
        cvl = self.cv_lake
        lakefrac = self.frac0
        cv = mosaic.cv
        nc = cv.shape[0]
        zero = torch.zeros_like(lakefrac)
        ro_t, bf_t = tap['ro'], tap['bf']                 # [nc, ntl]
        # vic_run.c 5.1.0:368-405 -- the wetland's runoff and baseflow, and
        # the rest of the cell's, over the cell
        is_wet = torch.zeros_like(cv, dtype=torch.bool)
        is_wet[c, k] = True
        w_up = torch.where(is_wet, torch.zeros_like(cv), cv)
        sum_ro = (ro_t * w_up).sum(1)[c]
        sum_bf = (bf_t * w_up).sum(1)[c]
        cv_wet = cv[c, k]
        wet_ro = ro_t[c, k] * cv_wet
        wet_bf = bf_t[c, k] * cv_wet
        rp = self.con['rpercent']
        f = dict(prec=atm['prcp'][t][c], tair=atm['tair'][t][c],
                 wind=atm['wind'][t][c],
                 vp=atm['eact_air'][t][c] / 1000.0,
                 vpd=atm['vpd'][t][c] / 1000.0,
                 pressure=atm['press'][t][c] / 1000.0,
                 shortin=atm['shortwave'][t][c],
                 longin=atm['longwave'][t][c],
                 air_density=atm['air_dens'][t][c],
                 sum_runoff=sum_ro, sum_baseflow=sum_bf,
                 wetland_runoff=wet_ro, wetland_baseflow=wet_bf,
                 t_max_snow=p['t_max_snow'], t_min_rain=p['t_min_rain'])
        sp = soil_params
        th = sp.thermal
        mm = sp.max_moist[c].T                          # [layer, B]
        soil = dict(cell_area=self.cell_area, lat=tap['lat'][c],
                    snow_rough=th['snow_rough'][c],
                    max_moist=mm,
                    Dsmax_step=sp.Dsmax[c] / steps_per_day,
                    Ds=sp.Ds[c], Ws=sp.Ws[c], c=sp.c[c],
                    resid_moist_bot=sp.resid_moist[c][..., -1])
        moist = land['moist']
        snow_state = land['snow_state']
        swq_t = (snow_state['ice'] + snow_state['surf_water']
                 + snow_state['pack_water'])
        int_snow = (land['canopy_state']['int_snow']
                    if land.get('canopy_state') is not None
                    else torch.zeros_like(cv))
        ice_bot = (zero if land.get('soil_ice') is None
                   else land['soil_ice'][c, k][..., -1].reshape(-1))
        wet_moist = moist[c, k].T.clone()               # [layer, B]
        wetland = dict(
            moist=wet_moist, wdew=land['wdew'][c, k],
            snow_canopy=int_snow[c, k], swq=swq_t[c, k], ice_bot=ice_bot,
            baseflow_m3=zero,
            snow_depth=snow_state['depth'][c, k],
            snow_pack_water=snow_state['pack_water'][c, k],
            snow_surf_water=snow_state['surf_water'][c, k],
            evap_total=zero, canopyevap=zero, canopy_vapor_flux=zero,
            vapor_flux=zero,
            soil_T0=tap['t_node'][c, k][..., 0] if 't_node' in tap else zero)
        lake_old = dict(self.state)
        # lake_var.soil.layer[].moist: initialize_lake.c:196-197 sets it to
        # porosity * depth (max_moist) and nothing in the run changes it
        # but a new lake's initialize_lake (lakes.eb.c:2242). A state read
        # from VIC's file carries what vic_restore.c left there instead
        # (state_from_vic); a cold start keeps no store and reports
        # max_moist.
        lake_mm = lake_old.pop('soil_moist', None)
        new_lake, new_wet, diag = L.lake_step(
            lake_old, wetland, soil, self.con, f, dt,
            torch.full_like(lakefrac, float(wind_h)), day_in_year)
        newfrac = diag['newfraction']
        energy = diag['energy']
        if lake_mm is None:
            lake_mm = mm
        else:
            fresh = (lakefrac <= 0.0) & (newfrac > 0.0)
            lake_mm = torch.where(fresh, mm, lake_mm)
            new_lake = dict(new_lake, soil_moist=lake_mm)
        self.state = new_lake

        # ---- the wetland's stores, restated (lakes.eb.c:2179-2195) ------
        swet = new_wet['swq'] / torch.where(swq_t[c, k] > 0, swq_t[c, k],
                                            torch.ones_like(newfrac))
        scale_snow = torch.where(swq_t[c, k] > 0, swet,
                                 torch.ones_like(swet))
        moist = moist.clone()
        moist[c, k] = new_wet['moist'].T
        wdew = land['wdew'].clone()
        wdew[c, k] = new_wet['wdew']
        snow_new = {}
        for key, v in snow_state.items():
            if key in ('ice', 'surf_water', 'pack_water', 'depth'):
                v = v.clone()
                v[c, k] = v[c, k] * scale_snow
            snow_new[key] = v
        land_out = dict(moist=moist, wdew=wdew, snow_state=snow_new)
        if land.get('canopy_state') is not None:
            cs = dict(land['canopy_state'])
            isn = cs['int_snow'].clone()
            isn[c, k] = new_wet['snow_canopy']
            cs['int_snow'] = isn
            land_out['canopy_state'] = cs

        # ---- the cell's rows (put_data.c 5.1.0:148-400) ------------------
        af = cvl * newfrac                         # the lake's AreaFactor
        w_old = cvl * (1.0 - lakefrac)             # the wetland's, as run
        w_new = cvl * (1.0 - newfrac)              # and as reported
        dw = w_new - w_old
        P = {}

        def add(name, val_lake):
            full = torch.zeros(nc, *val_lake.shape[1:], dtype=self.dtype,
                               device=cv.device)
            full[c] = val_lake
            P[name] = P.get(name, 0) + full

        ls = diag['lake_soil']                     # mm over the lake
        # runoff: rpercent of the land's, all the wetland's, to the lake
        add('runoff', -(rp * sum_ro + wet_ro) + ls['runoff'] * af)
        add('baseflow', -(rp * sum_bf + wet_bf) + ls['baseflow'] * af
            + new_wet['baseflow_m3'] * w_new)
        add('inflow', ls['inflow'] * af)
        fr = torch.where(newfrac > 0, lakefrac / torch.clamp(
            newfrac, min=L.DBL_EPSILON), zero)    # rescale_snow_energy_fluxes
        vap = new_lake['snow_vapor_flux'] * fr      # m over the lake
        add('et', (ls['evap'] + vap * MM_PER_M) * af)
        add('sublimation', vap * MM_PER_M * af)
        add('melt', new_lake['snow_melt'] * fr * af)
        add('snow_cover', new_lake['coverage'] * af)
        swq_l = new_lake['swq']
        add('swe', swq_l * MM_PER_M * af
            + (new_wet['swq'] * w_new - swq_t[c, k] * w_old) * MM_PER_M)
        add('snow_depth', new_lake['snow_depth'] * CM_PER_M * af
            + (new_wet['snow_depth'] * w_new
               - snow_state['depth'][c, k] * w_old) * CM_PER_M)
        add('snow_canopy', (new_wet['snow_canopy'] * w_new
                            - int_snow[c, k] * w_old) * MM_PER_M)
        add('wdew', new_wet['wdew'] * w_new - land['wdew'][c, k] * w_old)
        add('soil_liq', (lake_mm * af + new_wet['moist'] * w_new
                         - wet_moist * w_old).T)
        add('asat', af)
        add('soil_wet', af)
        root = mosaic.root_frac[c, k] if mosaic.root_frac.dim() == 3 \
            else mosaic.root_frac[k]
        rz = lambda m_: (m_.T * (root > 0).to(m_.dtype)).sum(-1)
        rm_new = rz(new_wet['moist'])
        add('rootmoist', rm_new * af + rm_new * w_new
            - rz(wet_moist) * w_old)
        # LAI: the lake reports veg_var[0][0]'s, the cell's first tile
        first = torch.as_tensor([_active_columns(self.cv0[int(i)].tolist())[0]
                                 for i in c], device=cv.device)
        lai_t = tap['lai']
        add('lai', lai_t[c, first] * af
            + lai_t[c, k] * (w_new - w_old))
        # energy (collect_eb_terms with the lake's energy, put_data.c
        # 5.1.0:248-301: ground flux, storage and fusion are the wetland's)
        en = energy
        if tap.get('eb') is not None:
            eb = tap['eb']
            fw = (1.0 - lakefrac) / torch.clamp(1.0 - newfrac,
                                               min=L.DBL_EPSILON)
            add('eb_net_short', en['NetShortAtmos'] * fr * af)
            add('eb_net_long', en['NetLongAtmos'] * fr * af)
            add('eb_net_rad', (en['NetShortAtmos'] + en['NetLongAtmos'])
                * fr * af)
            add('eb_latent', -en['AtmosLatent'] * fr * af)
            add('eb_sensible', -en['AtmosSensible'] * fr * af)
            add('eb_grnd_flux', eb['grnd_flux'][c, k] * fw * af)
            add('eb_delta_h', eb['delta_h'][c, k] * fw * af)
            add('eb_t_surf', en['Tsurf'] * af)
            tk = en['Tsurf'] + CONST_TKFRZ
            P['_rad4'] = torch.zeros(nc, dtype=self.dtype, device=cv.device)
            P['_rad4'][c] = tk ** 4 * af
            # OUT_SOIL_TEMP: the lake's soil layers stay at the 0 C
            # initialize_lake gave them (initialize_lake.c 5.1.0:200-204);
            # nothing in vic_run updates them, so they add nothing.
            add('eb_deltacc', en['deltaCC'] * fr * af)
            add('eb_advection', en['advection'] * fr * af)
            add('eb_snow_flux', en['snow_flux'] * fr * af)
            add('eb_rfrz_energy', en['refreeze_energy'] * fr * af)
            add('albedo', en['AlbedoUnder'] * af)
            add('pet', tap['pet'][c, k] * af)
            ra = new_lake['aero_resist']
            add('_aero_cond', torch.where(
                ra > L.DBL_EPSILON, af / torch.where(
                    ra > L.DBL_EPSILON, ra, torch.ones_like(ra)),
                torch.full_like(ra, 1e20)))
        # OUT_IN_LONG: the lake's LongUnderIn is never set (0 from
        # initialize_lake.c 5.1.0:150), and the wetland's is rescaled from
        # 1 - lakefrac to 1 - newfrac (lakes.eb.c:2566), so the cell's
        # incoming longwave loses the lake's START-of-step share
        add('in_long', -f['longin'] * cvl * lakefrac)
        # The wetland's per-area terms that water_balance does not rescale
        # (lakes.eb.c:2421-2600 lists what it does) are reported over
        # Cv * (1 - Clake) with Clake the END-of-step fraction, while the
        # land step weighted them with the start's: the difference is
        # cv * (lakefrac - newfrac) of the wetland's value.
        for key, src in (('asat', 'asat'), ('soil_wet', 'soil_wet'),
                         ('albedo', 'alb'), ('snow_cover', 'snow_cover'),
                         ('pet', 'pet')):
            x = tap.get(src)
            if x is not None:
                add(key, x[c, k] * dw)
        if tap.get('eb') is not None:
            add('eb_t_surf', tap['eb']['t_surf'][c, k] * dw)
            add('eb_layer_temp', tap['eb']['layer_temp'][c, k]
                * dw.unsqueeze(-1))
            P['_rad4'][c] = P['_rad4'][c] + tap['t_rad_k'][c, k] ** 4 * dw
        if tap.get('ra') is not None:
            add('_aero_cond', dw / torch.clamp(tap['ra'][c, k], min=1e-12))
        # OUT_REFREEZE is put_data.c 5.1.0:509-510's RFRZ_ENERGY / Lf * dt,
        # the lake's share included
        add('refreeze', en['refreeze_energy'] * fr * af / CONST_LATICE * dt)
        # Ratio rows. put_data.c 5.1.0:183-201 counts the wetland tile's
        # whole Cv into cv_snow and cv_veg (IsWet sets ThisAreaFract 1, and
        # no lakefactor there) while its terms carry Cv * (1 - Clake), and
        # the lake's snow terms join the numerator without joining cv_snow.
        wet_full = torch.zeros_like(cv)
        wet_full[c, k] = cvl
        cv_rep = torch.where(is_wet, torch.zeros_like(cv), cv)
        w_den = cv_rep + wet_full                  # cv_snow / cv_veg weights
        # The rows are re-weighted from the ones the land step formed: the
        # land step's ratio is num/den over the weights it ran with
        # (`cv`), so num = row * den, and the corrections are the
        # wetland's denominator weight (w_old -> Cv) and the lake's
        # numerator term. The wetland's numerator weight moves from w_old
        # to w_new, a change of cv * (newfrac - lakefrac), ~1e-9 of the
        # tile; that is dropped.
        sr = tap.get('snow_ratio')
        if sr is not None:
            hp = (snow_new['ice'] + snow_new['surf_water']
                  + snow_new['pack_water']) > 0.0
            lake_snow = swq_l > 0.0
            lake_vals = dict(salbedo=zero,
                             snow_surf_temp=new_lake['surf_temp'],
                             snow_pack_temp=new_lake['pack_temp'])
            den_old = (hp.to(cv.dtype) * cv).sum(1)
            den_new = (hp.to(cv.dtype) * w_den).sum(1)
            hpw = hp[c, k]
            for key, x in sr.items():
                lk_ = torch.zeros(nc, dtype=self.dtype, device=cv.device)
                lk_[c] = (torch.where(lake_snow, lake_vals[key] * af, zero)
                          + torch.where(hpw, x[c, k] * dw, zero))
                P['%' + key] = (den_old, den_new, lk_)
        if tap.get('veg_tile') is not None:
            vt = tap['veg_tile']
            lk_ = torch.zeros(nc, dtype=self.dtype, device=cv.device)
            lk_[c] = torch.where(vt[c, k], tap['t_veg'][c, k] * dw, zero)
            vt = vt.to(cv.dtype)
            P['%vegt'] = ((vt * cv).sum(1), (vt * w_den).sum(1), lk_)
        # OUT_BARESOILT: the lake adds its Tsurf over its AreaFactor and
        # is not counted in cv_baresoil (put_data.c 5.1.0:286-301,
        # :424-426, collect_eb_terms :829-834)
        add('_baresoilt', en['Tsurf'] * af)
        self.last = dict(diag=diag, af=af, newfrac=newfrac,
                         lakefrac=lakefrac)
        self._record(new_lake, diag, f, af, newfrac)
        return land_out, P

    def _record(self, lk, diag, f, af, newfrac):
        """put_data.c 5.1.0:305-400, the OUT_LAKE_* variables."""
        R = self.rows
        ca = self.cell_area
        zero = torch.zeros_like(newfrac)
        nia = lk['new_ice_area']
        has_ice = nia > 0.0
        safe = lambda x: torch.where(x > 0, x, torch.ones_like(x))
        R['lake_ice'].append(torch.where(
            has_ice, lk['ice_water_eq'] / safe(nia) * L.CONST_RHOICE
            / L.CONST_RHOFW, zero))
        R['lake_ice_temp'].append(torch.where(has_ice, lk['tempi'], zero))
        R['lake_ice_height'].append(torch.where(has_ice, lk['hice'], zero))
        # put_data.c 5.1.0:317-318 divides by areai, the ice area at the
        # START of the step, which is 0 on the step the ice first forms:
        # 0/0, and VIC writes NaN. Reproduced.
        R['lake_swe'].append(torch.where(has_ice, lk['swe'] / lk['areai'],
                                         zero))
        R['lake_swe_v'].append(torch.where(has_ice, lk['swe'], zero))
        R['lake_dswe_v'].append(lk['swe'] - lk['swe_save'])
        R['lake_dswe'].append((lk['swe'] - lk['swe_save']) * MM_PER_M / ca)
        R['lake_area_frac'].append(af)
        R['lake_depth'].append(lk['ldepth'])
        R['lake_surf_area'].append(lk['sarea'])
        R['lake_ice_fract'].append(torch.where(lk['sarea'] > 0,
                                               nia / safe(lk['sarea']), zero))
        R['lake_volume'].append(lk['volume'])
        R['lake_dstor_v'].append(lk['volume'] - lk['volume_save'])
        R['lake_dstor'].append((lk['volume'] - lk['volume_save']) * MM_PER_M
                               / ca)
        R['lake_surf_temp'].append(lk['temp'][0])
        mo = torch.where(lk['sarea'] > 0, lk['volume'] / ca * MM_PER_M, zero)
        R['lake_moist'].append(mo)
        R['surfstor'].append(mo)
        for key, v in (('bf_in', lk['baseflow_in']),
                       ('bf_out', lk['baseflow_out']),
                       ('chan_in', lk['channel_in']),
                       ('chan_out', lk['runoff_out']),
                       ('evap', lk['evapw']), ('rchrg', lk['recharge']),
                       ('ro_in', lk['runoff_in']),
                       ('vapflx', lk['vapor_flux'])):
            R['lake_%s_v' % key].append(v)
            R['lake_' + key].append(v * MM_PER_M / ca)
        R['lake_prec_v'].append(lk['prec'])

    def outputs(self, nc, nt):
        """The OUT_LAKE_* rows as [time, cell] (zero where a cell has no
        lake, as put_data.c leaves them)."""
        out = {}
        for key, rows in self.rows.items():
            if not rows:
                continue
            x = torch.zeros(len(rows), nc, dtype=self.dtype,
                            device=self.cells.device)
            x[:, self.cells] = torch.stack(rows)
            out[key] = x
        return out
