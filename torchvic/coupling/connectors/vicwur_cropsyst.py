# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_cropsyst: the CropSyst crop engine as a member of the VIC-WUR phase
group (docs/connectors/cropsyst.md).

The crop backend 'cropsyst' of the `crop` connector is an ordinary runtime
connector: it steps once per runtime day and reads VIC's layer moisture with
lag 1, which inside a VIC window (up to 31 days per call) it cannot do. This
member runs the same engine (torchvic.crop.cropsyst.engine, E10) inside the
group's phases instead, so it steps once per model day *between* the days of
a window, as VIC-CropSyst's land unit does:

  P3 `pass3`  the day's `process_day` (VIC_CropSyst/full_energy.c:628-650:
              after the day's surface fluxes), on the end-of-day layer
              moisture with the irrigation of the day already in it (the
              irrigation member's P3 runs before this one): E1/E2 uptake,
              growth, canopy, phenology, harvest. Its uptake per layer, its
              LAI and its signals to irrigation are what VIC's next step
              reads (VIC_soil.cpp:231-232: `layer[i].VCS.transpiration_daily`
              is set at process_day and read by runoff.c:370-374 on the next
              day), so the one-day lag of the reference is kept exactly.
  O  `put_data`  the day's per-tile record as series (lai, gai, biomass,
              yield, root depth, stage, uptake, ...), grid 'crop_tiles'.

VIC side (`core_hooks`, the hooks of torchvic.vicwur.cs_core, each
optional):
  transp   `transp_override`: the per-layer uptake [column, tile, layer] mm
           replaces VIC's transpiration on the crop tiles (P1)
  lai      `veg_override`: the crop's LAI on the crop tiles (P6; Wdmax
           follows LAI as VIC forms it)
  pet_short  VIC's PET_SHORT reference ET of the crop tile is the engine's
           reference ET of the day unless the weather arrays hold et_ref
  wpwp_floor  the Wpwp floor on the crop tiles (P3)
  vcs_drain  VIC-CropSyst's drainage law (core hook `vcs_drain`, needs
           the soil from `saxton_soil_params(vcs_drainage=True)`); the tiles
           of `irrigated` redistribute to Field_Capacity (runoff.c:604-606)
  canopy_evap  P7, CROPSYST_HANDLE_CANOPY_EVAP (core hook `canopy_crop`):
           VIC does not evaporate the canopy store of the crop tiles; the
           engine is handed it (VIC_land_unit_simulation.cpp:353-360), it
           serves the day's potential transpiration first (transpiration.cpp:
           65-78), and what it used leaves the store at the next VIC step as
           that step's canopy evaporation (:435-446)
  soil_evap  (with canopy_evap) VIC-CropSyst's soil evaporation of the crop
           tiles (canopy_evap.c:760-831; core hook `crop_soil_evap`): the
           soil under the gaps of the crop's green canopy evaporates
           min(moisture above half Wpwp, ARNO potential, PET_SHORT
           (1 - total cover)); the cover is the crop's of the day before
The irrigation member reads `signals()` in its P1 (`crop_fn: 'group'` in its
cropsyst block): root depth, water stress index and activity of the crop of
the day before (the lag of the backend's `lag: 1` link).

Options
    cropsyst   the `cropsyst` block of the crop connector (engine.build_run:
               db_dir, tiles + tile_cell, soil ...)
    tile       [T] VIC tile (vegetation class - 1) of each CropSyst tile; the
               tile's cell is cropsyst['tile_cell']
    band       [T] snow band of each CropSyst tile (default 0: every tile on
               the first band, as the CRB cases with one band); with bands a
               VIC tile of a cell has one CropSyst tile per band it should be
               driven on (the bands share the cell's weather and rotation,
               each has its own soil moisture, uptake and irrigation signals)
    weather    dict tmax, tmin [deg C], srad [W m-2], rhmin [%] ([n_day,
               n_cell]), optional et_ref, year, doy; None: from the VIC case
               (daily forcing: tmax, tmin, shortwave, eact_air)
    lat, lon   [n_cell] degrees; None: the case's
    n_cells    number of cells (default: from tile_cell)
    hooks      dict(transp=True, lai=True, wpwp_floor=False,
               vcs_drain=False, canopy_evap=False, soil_evap=False): which
               hooks the member sets on VIC's core
    irrigated  [T] bool: the CropSyst tiles of an irrigated field (the
               reference's `irrigated_field`, surface_fluxes.c:1109-1129);
               default none; read by the vcs_drain hook
    et_ref     'vic' (PET_SHORT of the crop tile, default) or 'weather'

State (checkpoint): the engine state tree and the last day's feedback
([T] / [T, L] tensors the hooks and the irrigation signals read); a restart
equals the continuous run bitwise. Ledger: the crop moves no water itself (its
uptake is VIC's transpiration, its irrigation the irrigation member's): zero
row. No host synchronisation in a step (the day index is a device scalar);
the engine and its day tables are built once, at initialize when `weather`,
`lat` and `lon` are given, else at the first P3 from the attached case.
"""
import numpy as np
import torch

from ..connector import Balance, ComputeError
from ..spec import CouplingError, GridSpec, VarSpec
from ...crop.cropsyst import engine as cse
from .vicwur_common import GRID, WurConnector

TILE_GRID = 'crop_tiles'
MIN_FCANOPY = 1e-4


class VicWurCropSystConnector(WurConnector):
    name = 'vicwur_cropsyst'
    version = '1'
    kind = 'cropsyst'
    role = 'crop'
    flags = dict(cropsyst=True)

    def __init__(self, cropsyst=None, tile=None, weather=None, lat=None,
                 lon=None, n_cells=None, hooks=None, irrigated=None, et_ref='vic',
                 group=None, band=None, **config):
        super().__init__(group=group, **config)
        if cropsyst is None or tile is None:
            raise CouplingError('%s: give cropsyst and tile' % self.name)
        self.cfg = dict(cropsyst)
        if 'tiles' not in self.cfg or 'tile_cell' not in self.cfg:
            raise CouplingError('%s: cropsyst needs tiles and tile_cell '
                                '(the crop tiles and their cells)'
                                % self.name)
        self.tcell = np.asarray(self.cfg['tile_cell'], dtype=np.int64)
        self.vt = np.asarray(tile, dtype=np.int64).reshape(-1)
        self.T = len(self.cfg['tiles'])
        if self.tcell.shape != (self.T,) or self.vt.shape != (self.T,):
            raise CouplingError('%s: tile and tile_cell are one entry per '
                                'CropSyst tile (%d)' % (self.name, self.T))
        self.N = int(n_cells if n_cells is not None else self.tcell.max() + 1)
        if (self.tcell < 0).any() or (self.tcell >= self.N).any():
            raise CouplingError('%s: tile_cell outside the %d cells'
                                % (self.name, self.N))
        self.vb = (np.zeros(self.T, dtype=np.int64) if band is None else
                   np.asarray(band, dtype=np.int64).reshape(-1))
        if self.vb.shape != (self.T,) or (self.vb < 0).any():
            raise CouplingError('%s: band is one non-negative entry per '
                                'CropSyst tile (%d)' % (self.name, self.T))
        if len({(int(c), int(t), int(b)) for c, t, b in
                zip(self.tcell, self.vt, self.vb)}) != self.T:
            raise CouplingError('%s: two CropSyst tiles on one VIC tile and '
                                'band of a cell' % self.name)
        h = dict(transp=True, lai=True, wpwp_floor=False, vcs_drain=False,
                 canopy_evap=False, soil_evap=False)
        h.update(hooks or {})
        bad = set(h) - {'transp', 'lai', 'wpwp_floor', 'vcs_drain',
                        'canopy_evap', 'soil_evap'}
        if bad:
            raise CouplingError('%s: unknown hook(s) %s' % (self.name,
                                                            sorted(bad)))
        self.hooks = h
        self.irr_cfg = (None if irrigated is None else
                        np.asarray(irrigated, dtype=bool).reshape(-1))
        if self.irr_cfg is not None and self.irr_cfg.shape != (self.T,):
            raise CouplingError('%s: irrigated is one flag per CropSyst '
                                'tile (%d)' % (self.name, self.T))
        if et_ref not in ('vic', 'weather'):
            raise CouplingError("%s: et_ref is 'vic' or 'weather'"
                                % self.name)
        self.et_src = et_ref
        self.w_cfg = None if weather is None else dict(weather)
        self.lat_cfg, self.lon_cfg = lat, lon
        self._check_config()
        self.run_ = None
        self.st = None
        self.fb = None                   # the last day's feedback
        self._blob = None                # a loaded state before the build
        self._cols_cache = None
        self._smap = None
        v = lambda n, u, desc, dims=('cell',): VarSpec(
            n, u, dims, kind='state', grid=TILE_GRID, description=desc)
        self.outputs = (
            v('lai', '1', 'leaf area index after the day (0 without a crop)'),
            v('gai', '1', 'green area index of the day record'),
            v('green_cover', '1', 'green canopy solar interception'),
            v('root_depth', 'm', 'rooting depth of the day record'),
            v('harvest', '1', '1 on a harvest day'),
            v('sown', '1', '1 on a sowing day'),
            v('stage', '1', 'normal growth stage code'),
            v('biomass', 'kg m-2', 'vital canopy biomass'),
            v('yield', 'kg ha-1', 'yield on the harvest day'),
            v('crop_active', '1', '1 while a crop stands on the tile'),
            v('water_stress', '1', 'actual / limited potential transpiration'),
            v('act_transp', 'mm', 'uptake of the day (mm per day)'),
            v('et_max', 'mm', 'crop ET max (mm per day)'),
            v('uptake', 'mm', 'uptake per layer (mm per day)',
              ('cell', 'layer')),
            v('canopy_take', 'mm', 'canopy store the crop evaporated on the '
              'day (canopy_evap hook; 0 without it)'))
        self.owned_state = ('%s.state' % self.name,)

    # -- declarations ----------------------------------------------------------
    def grids(self):
        return [GridSpec(GRID, self.N, kind='cells'),
                GridSpec(TILE_GRID, self.T, kind='tiles',
                         description='CropSyst rotation tiles')]

    # -- set-up ----------------------------------------------------------------
    def _init(self, context):
        self._ix = None
        self._maybe_build()

    def _vic_case(self):
        vic = self.group.vic
        d = getattr(vic, 'driver', None) if vic is not None else None
        return getattr(d, 'case', None)

    def _weather(self):
        """Daily weather tables of the run from the attached case's forcing
        (the same construction as the crop connector's)."""
        from ...crop.cropsyst import growth as gr
        case = self._vic_case()
        if case is None:
            raise ComputeError('%s: no weather given and no VIC case '
                               'attached' % self.name)
        if int(case.model_steps) != 1:
            raise ComputeError('%s: the weather of a sub-daily case is not '
                               'built here; give weather=' % self.name)
        a = case.atm
        f = lambda k: torch.as_tensor(a[k]).to(torch.float64)
        G = self.group
        rows = (lambda x: G.cell_rows(x.T).T) if G.layout()['bands'] \
            is not None else (lambda x: x)
        tmax, tmin = rows(f('tmax')), rows(f('tmin'))
        sw, ea = rows(f('shortwave')), rows(f('eact_air')) / 1000.0
        rh = (100.0 * ea / gr.saturation_vp_kpa(tmax)).clamp(0.0, 100.0)
        dts = case.dates
        year = [d.year for d in dts]
        doy = [d.timetuple().tm_yday for d in dts]
        w = dict(tmax=tmax, tmin=tmin, srad=sw, rhmin=rh)
        return w, year, doy

    def _maybe_build(self):
        if self.run_ is not None:
            return
        if self.w_cfg is None and self._vic_case() is None:
            return                       # built at the first P3
        self._build()

    def _build(self):
        dev = self.dev
        cfg = dict(self.cfg)
        if self.w_cfg is not None:
            wx = dict(self.w_cfg)
            year, doy = wx.pop('year'), wx.pop('doy')
        else:
            wx, year, doy = self._weather()
        case = self._vic_case()
        lat, lon = self.lat_cfg, self.lon_cfg
        lon = cfg.pop('lon', lon)
        if lat is None and case is not None:
            lat = case.dom.lat
        if lon is None and case is not None:
            lon = case.dom.lon
        if lat is None or lon is None:
            raise CouplingError('%s: no latitude or longitude (give lat= '
                                'and lon=)' % self.name)
        lat = np.asarray(lat, dtype=np.float64).reshape(-1)
        lon = np.asarray(lon, dtype=np.float64).reshape(-1)
        self.run_ = cse.build_run(cfg, wx, lat, lon, year, doy, self.N, dev)
        self.n_day = self.run_.tables.n_day
        self.L = self.run_.n_layer
        self.st = self.run_.state
        self.cell_t = torch.as_tensor(self.tcell, device=dev)
        self.vt_t = torch.as_tensor(self.vt, device=dev)
        self.vb_t = torch.as_tensor(self.vb, device=dev)
        if self._blob is not None:
            b, self._blob = self._blob, None
            self._load(b)

    # -- the day -----------------------------------------------------------------
    def pass3(self):
        v = self.view
        if v is None:
            raise ComputeError('%s: no step view' % self.name)
        if self.run_ is None:
            self._build()
        step = self.group.step
        if step >= self.n_day:
            raise ComputeError('%s: step %d beyond the %d days of weather'
                               % (self.name, step, self.n_day))
        moist = v.moist[self.cell_t, self.vt_t, self.vb_t]        # [T, L]
        ice = (None if v.ice is None else
               v.ice[self.cell_t, self.vt_t, self.vb_t])
        et = None
        if self.run_.tables.et_ref is None:
            et = self._pet(v)
        icpt = self._wdew(v) / 1000.0 if self.hooks['canopy_evap'] else None
        new, o = self.run_.step(self.st, moist, ice, et, None, icpt)
        self.st = new
        f64 = lambda x: x.to(torch.float64)
        fb = {'uptake': o.uptake * 1e3, 'lai': o.lai,
              'root_depth': o.root_depth, 'water_stress': o.water_stress,
              'present': f64(o.present), 'gai': o.gai, 'cover_total': o.cover_total,
              'green_cover': o.green_cover, 'biomass': o.biomass,
              'yield': o.yield_ * 1e4, 'harvest': f64(o.harvest),
              'sown': f64(o.sown), 'stage': f64(o.stage),
              'et_max': o.et_max * 1e3, 'act_transp': o.act_transp * 1e3,
              'lwp': o.lwp}
        if icpt is not None:
            fb['canopy_take'] = o.icpt_used * 1e3
        self.fb = {k: x.clone() for k, x in fb.items()}
        self._cols_cache = None
        self.flag(~torch.stack([torch.isfinite(self.fb[k]).all() for k in (
            'lai', 'biomass', 'root_depth', 'act_transp', 'water_stress',
            'lwp')]).all())

    def _pet(self, v):
        """The reference ET of the day [T] m: VIC's PET_SHORT of the crop's
        tile (the step just run), mm per day."""
        tf = v.tile_fluxes
        if tf is None or tf.get('pet_short') is None:
            raise ComputeError('%s: no PET_SHORT of VIC (the core hook '
                               'pet_short and tile_fluxes)' % self.name)
        x = self.group.from_columns(tf['pet_short']).to(self.dev,
                                                        torch.float64)
        return x[self.cell_t, self.vt_t, self.vb_t] / 1000.0

    def _wdew(self, v):
        """VIC's canopy store of the crop's tile after the step just run
        [T] mm."""
        tf = v.tile_fluxes
        if tf is None or tf.get('wdew') is None:
            raise ComputeError('%s: no canopy store of VIC (the core hook '
                               'canopy_crop and tile_fluxes)' % self.name)
        x = self.group.from_columns(tf['wdew']).to(self.dev, torch.float64)
        return x[self.cell_t, self.vt_t, self.vb_t]

    # -- the irrigation member's view of the crop ----------------------------------
    def signals_for(self, irr_tile, n_cell):
        """The crop signals of irrigation classes on VIC tiles `irr_tile`
        (0-based, [NI] host) of every cell: dict(root_depth_mm, wsi, active)
        of the day before (zeros and inactive before the first day).
        root_depth_mm and wsi are [n_cell, NI], or [n_cell, NI, B] when a
        tile is driven on several bands (B = the case's band count); active
        is [n_cell, NI]: the crop's presence is the cell's (the bands share
        the rotation and the weather)."""
        key = (tuple(int(t) for t in irr_tile), int(n_cell))
        nb = int(self.group.layout()['area_fract'].shape[1])
        banded = int(self.vb.max(initial=0)) > 0
        if self._smap is None or self._smap[0] != key:
            kmap = np.zeros((n_cell, len(irr_tile), nb), dtype=np.int64)
            has = np.zeros((n_cell, len(irr_tile), nb), dtype=bool)
            allk = np.arange(self.T)
            for i, t in enumerate(irr_tile):
                m = self.vt == int(t)
                kmap[self.tcell[m], i, self.vb[m]] = allk[m]
                has[self.tcell[m], i, self.vb[m]] = True
            self._smap = (key, torch.as_tensor(kmap, device=self.dev),
                          torch.as_tensor(has, device=self.dev))
        _, kmap, has = self._smap
        z = torch.zeros(kmap.shape, dtype=torch.float64, device=self.dev)
        if self.fb is None:
            out = dict(root_depth_mm=z, wsi=z.clone(), active=torch.zeros_like(has[..., 0]))
        else:
            g = lambda k: self.fb[k][kmap]
            pres = (g('present') > 0.5) & has
            out = dict(root_depth_mm=torch.where(has, g('root_depth') * 1e3, z),
                       wsi=torch.where(pres, 1.0 - g('water_stress'), z),
                       active=pres.any(-1))
        if not banded:
            out['root_depth_mm'] = out['root_depth_mm'][..., 0]
            out['wsi'] = out['wsi'][..., 0]
        return out

    # -- the VIC side ----------------------------------------------------------------
    def _dense(self, x, fill=0.0):
        """[T(, L)] on the crop tiles -> the case's columns [col, tile(,
        L)]."""
        G = self.group
        L = G.layout()
        out = torch.full((L['n_cell'], L['n_tile'], L['area_fract'].shape[1])
                         + tuple(x.shape[1:]), fill, dtype=x.dtype,
                         device=x.device)
        out[self.cell_t, self.vt_t, self.vb_t] = x
        return G.to_columns(out)

    def _cols(self):
        if self._cols_cache is None:
            fb = self.fb
            if fb is None:
                z = torch.zeros(self.T, dtype=torch.float64, device=self.dev)
                zl = torch.zeros(self.T, self.L, dtype=torch.float64,
                                 device=self.dev)
                fb = dict(uptake=zl, lai=z, canopy_take=z, green_cover=z,
                          cover_total=z, present=z)
            take = fb.get('canopy_take')
            self._cols_cache = dict(transp=self._dense(fb['uptake']),
                                    cover_green=self._dense(torch.where(
                                        fb['present'] > 0.5, fb['green_cover'],
                                        torch.zeros_like(fb['green_cover']))),
                                    cover_total=self._dense(fb['cover_total']),
                                    take=(None if take is None
                                          else self._dense(take)),
                                    lai=self._dense(fb['lai']),
            fcanopy=self._dense(torch.where(
                fb['lai'] > 0, torch.ones_like(fb['lai']),
                torch.full_like(fb['lai'], MIN_FCANOPY))))
        return self._cols_cache

    def lai_prev_columns(self):
        """(LAI [col, tile], mask [col, tile]): the crop tiles' LAI of the
        last day (the veg_var LAI the land-use member's lu_apply finds in F),
        or None before the first day. The land-use member reads it from the
        group's crop role (WOFOST's member has the same call)."""
        if self.fb is None or not self.hooks['lai']:
            return None
        if getattr(self, '_mask_cols', None) is None:
            self._mask_cols = self._dense(torch.ones(
                self.T, dtype=torch.bool, device=self.dev), False)
        return self._cols()['lai'], self._mask_cols

    def core_hooks(self, core):
        G = self.group
        if self.run_ is None:
            self._maybe_build()
        lay = G.layout()
        mask = self._dense(torch.ones(self.T, dtype=torch.bool,
                                      device=self.dev), False)
        if self.hooks['transp']:
            core.transp = lambda t: self._cols()['transp']
            core.transp_mask = mask
        if self.hooks['lai']:
            core.veg['lai'] = lambda t: self._cols()['lai']
            case = self._vic_case()
            if case is not None and getattr(case.mosaic, 'fcanopy',
                                            None) is not None:
                # a fallow tile (LAI 0) is bare soil: VIC-WUR 3.0 floors its
                # cover at MIN_FCANOPY (vic_force.c:428-441), (1 - fcanopy)
                # of it evaporates as ARNO bare soil
                core.veg['fcanopy'] = lambda t: self._cols()['fcanopy']
            core.veg_mask = mask
        if self.et_src == 'vic' and (self.w_cfg is None
                                     or 'et_ref' not in self.w_cfg):
            core.tile_fluxes = True
            core.pet_short = True
        if self.hooks['wpwp_floor']:
            core.wpwp_floor = mask
        if self.hooks['canopy_evap']:
            core.tile_fluxes = True
            core.canopy_crop = dict(mask=mask, take=lambda t: self._cols()
                                    ['take'])
        if self.hooks['soil_evap']:
            if not self.hooks['canopy_evap']:
                raise CouplingError('%s: soil_evap goes with canopy_evap'
                                    % self.name)
            core.tile_fluxes = True
            core.pet_short = True
            core.crop_soil_evap = dict(
                mask=mask, cover_green=lambda t: self._cols()['cover_green'],
                cover_total=lambda t: self._cols()['cover_total'])
        if self.hooks['vcs_drain']:
            irr = (None if self.irr_cfg is None else self._dense(
                torch.as_tensor(self.irr_cfg, device=self.dev), False))
            core.vcs_drain = {} if irr is None else dict(irrigated=irr)
        del lay

    # -- O ---------------------------------------------------------------------
    def put_data(self):
        fb = self.fb
        if fb is None:
            return
        exp = dict(lai=fb['lai'], gai=fb['gai'],
                   green_cover=fb['green_cover'],
                   root_depth=fb['root_depth'], harvest=fb['harvest'],
                   sown=fb['sown'], stage=fb['stage'],
                   biomass=fb['biomass'], crop_active=fb['present'],
                   water_stress=fb['water_stress'],
                   act_transp=fb['act_transp'], et_max=fb['et_max'],
                   uptake=fb['uptake'])
        exp['canopy_take'] = fb.get('canopy_take', torch.zeros_like(fb['lai']))
        exp['yield'] = fb['yield']
        exp = {k: x.clone() for k, x in exp.items()}
        self._pending = (Balance(n_terms=1, scale=self.zero()), exp,
                         {k: x.unsqueeze(0) for k, x in exp.items()})

    # -- state -----------------------------------------------------------------
    def _state(self):
        if self.st is None:
            return {} if self._blob is None else dict(self._blob)
        d = {'s%d' % i: t.clone() for i, t in enumerate(cse.flatten(self.st))}
        if self.fb is not None:
            d.update({'fb_' + k: x.clone() for k, x in self.fb.items()})
        return d

    def _load(self, blob):
        blob = dict(blob or {})
        # the last day's feedback is read at once (the irrigation member's
        # P1 of the restart day runs before the engine is built)
        fb = {k[3:]: v.to(self.dev).clone() for k, v in blob.items()
              if k.startswith('fb_')}
        self.fb = fb or None
        self._cols_cache = None
        if not blob:
            if self.run_ is not None:
                self.st = self.run_.state
            return
        if self.run_ is None:
            self._blob = blob              # the engine state: at the build
            return
        n = sum(1 for k in blob if k.startswith('s') and k[1:].isdigit())
        self.st = cse.unflatten(self.run_.state, [
            blob['s%d' % i].to(self.dev).clone() for i in range(n)])

    def _w3_reset(self, step):
        pass
