# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_crop: VIC-WUR crops, WOFOST-NPK (C12), as a connector.

Kernel torchvic.crop.wofost_npk (`WofostNPK`); C paths VIC-WUR 3.0
plugins/crops/src and plugins/wofost/src. Daily only, like the C.

  F  `pre_force`: crop_forcing (FORCE_TSUM / FORCE_FERT records, when the
     options ask for them: `forced`)
  U  `pre_vic`: crop_update_step_vars (crop_update_step_vars.c:4-256,
     torchvic.vicwur.crop_feedback): the crops' LAI, cover,
     albedo, height, resistances, roots and layer Wcr of the crop tiles,
     handed to VIC through the veg / wcr hooks and to the irrigation
     member (its Wcr and roots) -- then wofost_update_step_vars
     (wofost_update_step_vars.c:4-71)
  P1 `pass1`: crop_adjust_requirement (crop_adjust_requirement.c:7-97):
     the irrigation flag cleared where the tile's crops neither cultivate
     nor are to be sown within the offset
  P3 irr_wofost's water (irr_use.c:531-644): the irrigation member calls
     `wofost_water` (IRRTAB x Cc x 10 per irrigated tile and band)
  P3 `pass3`: crop_run (crop_run.c:181-332): the day's met (air
     temperature, shortwave, CO2, VIC's potential evaporation of the
     crop's tile and band), the tile's water stress (VIC's stored
     cell->water_stress, overwritten on the irrigated tiles by
     irr_run_shortage just before, crop_run.c:108-148), and one WOFOST day
     of every unit
  O  `put_data`: crop_put_rate_data (crop_output.c:1003-1281): OUT_CROP_EVAP,
     _EVAP_BARE, _EVAP_CANOP, _TRANSP_VEG from VIC's per-tile fluxes of the
     step, and the WOFOST rows

VIC side (`core_hooks`): the stored water stress (the run's vicwur_core
`water_stress`, carried across windows through VicConnector), the per-tile
fluxes (`tile_fluxes`: pot_evap, transp, esoil, canopyevap), and with
`feedback` the crop tiles' vegetation (`veg`, `veg_mask`) and layer Wcr
(`wcr`). Land use reads the crop tiles' LAI in force (`lai_prev_columns`).

Options
    tmpl, setup      CropTemplates and UnitSetup (torchvic.crop.wofost_npk.
                     params: from_wofost_text, setup_from_vicwur)
    wofost_options   the WOFOST_* switches (PIRR PFERT ... FORCE_FERT)
    dates            the run's days [(year, month, day, day_in_year)]
    tile             [N, NC] TorchVIC tile (vegetation class - 1) of each
                     crop class
    air_temp, shortwave, co2, pot_evap
                     Series specs [N] (pot_evap [N, B, NC]); default: the
                     attached case's air temperature and shortwave, the
                     vicwur_co2 member's CO2, potential evaporation 0
                     (TorchVIC's water-balance path forms no per-tile PET;
                     WOFOST-NPK only stores it, crop_run.c:66-69)
    forced           callable(step) -> dict(tsum1, tsum2, dvs, N, P, K)
                     (FORCE_TSUM / FORCE_FERT)
    soil             dict of the cells' soil_con values the feedback reads:
                     depth0 [N] m, wpwp, wfc, wcr [N, L] mm, rough [N] m
                     (VIC-WUR 3.0 vic_init.c:636-795); None: no feedback (VIC
                     keeps its own vegetation)
    dist_season      WOFOST_DIST_SEASON (the sowing day crop_adjust_
                     requirement reads, VIC-WUR 3.0 wofost_init.c:147-168)
    calendar         the run's CALENDAR (leap years of crop_adjust_
                     requirement)
    veg_map          [N, NC] crop_con_map veg_class (0-based, -1 none),
                     every class (VIC-WUR 3.0 crop_init.c:57-73), and
                     tile_present [N, T] (veg_con_map vidx != NODATA_VEG):
                     the tiles crop_update_step_vars clears and transforms
                     (:42-52, :154-164); default: the tiles of the active
                     classes
    dtype            'float32' (production, the C's float with c_promote)
                     or 'float64'; exact_sums

State (checkpoint): every unit state (the WofostNPK dict), the last U's
feedback (P1-P3 of that day read its Wcr and roots) and the crop tiles' LAI
of the last U (what land use reads at the next step). Ledger: the
crop moves no water itself (irr_wofost's water is the irrigation member's);
zero row.
"""
import numpy as np
import torch

from ..connector import Balance, ComputeError
from ..spec import CouplingError, VarSpec
from ...vicwur import crop_feedback as CF
from .vicwur_common import GRID, Series, WurConnector, as_f64, as_i64

_DT = {'float32': torch.float32, 'float64': torch.float64}


class VicWurCropConnector(WurConnector):
    name = 'vicwur_crop'
    version = '1'
    kind = 'wofost'
    role = 'crop'
    flags = dict(wofost=True)

    def __init__(self, tmpl=None, setup=None, wofost_options=None,
                 dates=None, tile=None, air_temp=None, shortwave=None,
                 co2=None, pot_evap=None, forced=None, dtype='float32',
                 c_promote=True, exact_sums=False, soil=None,
                 dist_season=False, calendar='proleptic_gregorian',
                 veg_map=None, tile_present=None, chunk_lanes=None,
                 snapshot='auto', graph=True, leaf_ring='auto',
                 lane_state='auto', group=None, **config):
        super().__init__(group=group, **config)
        #: slots of the WOFOST leaf-class ring (WofostNPK ring): an int
        #: starts there on the CUDA lanes where the leaf sums run on the
        #: live classes (leaf_kernels.supported for every chunk) and grows
        #: before a unit's classes could outgrow it; 'auto' or None: the
        #: whole row of K classes (on ISC a unit's live classes reach 284
        #: of 336 within the year, and each growth recompiles the fused
        #: day)
        self.leaf_ring = leaf_ring
        self._ringq = None
        #: the units' fields on the lanes only (WofostNPK lane_state):
        #: 'auto' where the leaf ring is on (the lanes' day runs in one
        #: chunk); False: on every (cell, band, crop) unit
        self.lane_state_on = lane_state
        #: crop units per WOFOST chunk: crop_run and its outputs run a
        #: chunk of cells at a time and write the state in place
        #: (WofostNPK.crop_run_inplace), so a day's temporaries are a
        #: chunk's; 0 runs every cell at once; default 2^19 replayed from
        #: CUDA graphs (`graph`, CUDA only: WofostNPK.day_graphed), 2^20
        #: without
        self.graph = bool(graph)
        self.chunk_lanes = (int(chunk_lanes) if chunk_lanes is not None
                            else (1 << 19 if graph else 1 << 20))
        self._ver, self._snap = 0, None
        #: where a snapshot of the crop state lives: 'device', 'host'
        #: (pinned memory, copied synchronously: a rollback is rare, and
        #: the runtime's window and the step each hold one) or 'auto'
        #: (host when the state is larger than 1 GiB)
        if snapshot not in ('auto', 'device', 'host'):
            raise CouplingError("%s: snapshot must be 'auto', 'device' or "
                                "'host'" % self.name)
        self.snapshot_on = snapshot
        if tmpl is None or setup is None or dates is None or tile is None:
            raise CouplingError('%s: give tmpl, setup, dates and tile'
                                % self.name)
        if dtype not in _DT:
            raise CouplingError("%s: dtype must be 'float32' or 'float64'"
                                % self.name)
        self.tmpl, self.su = tmpl, setup
        self.opts = dict(wofost_options or {})
        if (self.opts.get('FORCE_TSUM') or self.opts.get('FORCE_FERT')) \
                and forced is None:
            raise CouplingError('%s: FORCE_TSUM / FORCE_FERT need forced'
                                % self.name)
        self.forced_fn = forced
        self.dates = [tuple(int(v) for v in d) for d in dates]
        if any(len(d) != 4 for d in self.dates):
            raise CouplingError('%s: dates are (year, month, day, '
                                'day_in_year)' % self.name)
        self.tile_cfg = tile
        self.N = setup.N
        tl = torch.as_tensor(tile)
        if tuple(tl.shape) != (setup.N, setup.NC):
            raise CouplingError('%s: tile must be [N, NC]' % self.name)
        self.series = {k: (None if v is None else Series(v, k))
                       for k, v in (('air_temp', air_temp),
                                    ('shortwave', shortwave), ('co2', co2),
                                    ('pot_evap', pot_evap))}
        # the crop model's dtype (WurConnector.initialize sets self.dtype,
        # the plugins' float64, after this)
        self.crop_dtype = _DT[dtype]
        self.c_promote, self.exact_sums = bool(c_promote), bool(exact_sums)
        if soil is not None:
            miss = [k for k in ('depth0', 'wpwp', 'wfc', 'wcr', 'rough')
                    if k not in soil]
            if miss:
                raise CouplingError('%s: soil misses %s' % (self.name, miss))
        self.soil_cfg = soil
        self.veg_map, self.tile_present = veg_map, tile_present
        self.dist_season = bool(dist_season)
        self.calendar = str(calendar).lower()
        self._check_config()
        self.outputs = (VarSpec('crop_lai', '1', ('cell', 'crop'),
                                kind='state', grid=GRID,
                                description='OUT_CROP_LAI'),)
        self.owned_state = ('%s.units' % self.name,)
        #: crop_update_step_vars on (VIC's crop tiles follow the crops)
        self.feedback = soil is not None
        self.s = None
        self._fb = None
        self._cols = None
        self._lai_cur = self._lai_prev = None

    def _init(self, context):
        from ...crop.wofost_npk.kernel import WofostNPK
        from ...crop.wofost_npk.params import Calendar
        d = self.dev
        self.m = WofostNPK(self.tmpl, self.su, self.opts,
                           dtype=self.crop_dtype,
                           c_promote=self.c_promote, device=d,
                           exact_sums=self.exact_sums)
        self.m.ring = self._ring0()
        self.m.lane_state = (self.lane_state_on is True or (
            self.lane_state_on == 'auto' and self._ring_supported()))
        self.cal = Calendar(self.dates, self.tmpl, device=d,
                            dtype=self.crop_dtype)
        self.tile = as_i64(self.tile_cfg, d).clamp(min=0)
        for v in self.series.values():
            if v is not None:
                v.to(d)
        if self.s is None:
            self.s = self.m.init_state()
            self._changed()
        self._forced = None
        su = self.su
        self.active = torch.as_tensor(np.asarray(su.active), device=d)
        self.af = torch.as_tensor(np.asarray(su.area_fract),
                                  dtype=torch.float64, device=d)
        self.cc = torch.as_tensor(np.asarray(su.cc), dtype=torch.float64,
                                  device=d)                    # [N,NC,12]
        names = ('Fcanopy MaxHeight Albedo TrunkRatio WindAttenuation '
                 'RadiationAttenuation RGL MaxArchitecturalResistance '
                 'MinStomatalResistance CropGroupNumber').split()
        self.prm = {k: torch.tensor(np.asarray(self.tmpl.scalars[k],
                                               np.float64),
                                    dtype=torch.float32, device=d)
                    for k in names}
        self.soil = None if self.soil_cfg is None else {
            k: as_f64(v, d) for k, v in self.soil_cfg.items()}
        # per step (host, no sync in the step): day of year, leap year,
        # List(Irrigation) of every class (List.c:9-24, the first matching
        # date)
        leap = []
        for (y, mo, dd, doy) in self.dates:
            if self.calendar in ('noleap', '365_day'):
                lp = 0
            elif self.calendar in ('all_leap', '366_day'):
                lp = 1
            elif self.calendar == 'julian':
                lp = int(y % 4 == 0)
            else:
                lp = int((y % 4 == 0 and y % 100 != 0) or y % 400 == 0)
            leap.append(lp)
        self.doy_t = torch.tensor([x[3] for x in self.dates],
                                  dtype=torch.long, device=d)
        self.leap_t = torch.tensor(leap, dtype=torch.long, device=d)
        irr = np.zeros((len(self.dates), self.su.NC), np.float32)
        tab = self.tmpl.mng_tables.get('Irrigation') or []
        for k, (ms, ds, am) in enumerate(tab):
            for t, (y, mo, dd, doy) in enumerate(self.dates):
                for m_, d_, v_ in zip(ms, ds, am):
                    if int(m_) == mo and int(d_) == dd:
                        irr[t, k] = np.float32(v_)
                        break
        self.irr_list = torch.as_tensor(irr, device=d)
        self.start_doy = torch.as_tensor(np.asarray(su.start_doy),
                                         dtype=torch.long, device=d)
        # linked at initialize: a restart's first phases are P1-O of the
        # day before its window, which read the crop tiles' Wcr and roots
        self._linked = False
        self._link()

    # -- inputs ----------------------------------------------------------------
    def _cell(self, name, step):
        sr = self.series[name]
        if sr is not None:
            return sr.get(step).reshape(-1).expand(self.N).to(torch.float64)
        if name == 'co2':
            c = self.group.roles.get('co2')
            if c is None or c.ppm() is None:
                raise ComputeError('%s: no CO2 (give co2, or run with '
                                   'vicwur_co2)' % self.name)
            return c.ppm()
        x = self.group.met({'air_temp': 'tair'}.get(name, name), step)
        if x is None:
            raise ComputeError('%s: no %s (give it, or attach a case)'
                               % (self.name, name))
        return torch.as_tensor(x, dtype=torch.float64, device=self.dev)

    def _tile_cb(self, x):
        """[cell, tile, band] -> the crop's tile [N, B, NC]."""
        n, t, b = x.shape
        idx = self.tile[:, None, :].expand(n, b, self.tile.shape[1])
        return x.permute(0, 2, 1).gather(2, idx)

    def _stress(self, v):
        """cell->water_stress of each crop's tile and band, [N, B, NC]: VIC's
        stored value, the irrigation shortage's on the irrigated tiles."""
        G = self.group
        if v.water_stress is None:
            raise ComputeError('%s: the step view carries no water stress '
                               "(VIC's vicwur_core water_stress)" % self.name)
        ws = G.from_columns(v.water_stress).to(self.dev, torch.float64)
        irr = G.roles.get('irrigation')
        cs = None if irr is None else irr.cell_stress()
        if cs is not None:
            from ...vicwur import irrigation as IR
            act = IR.active_mask(irr._cvday, v.af) & irr.act.unsqueeze(-1)
            for i in range(irr.NI):
                t = int(irr.vc[i]) - 1
                ws[:, t] = torch.where(act[:, i], cs[:, i], ws[:, t])
        return self._tile_cb(ws)

    # -- hooks ---------------------------------------------------------------
    def pre_force(self):
        if self.forced_fn is not None:
            self._forced = self.forced_fn(self.group.step)

    def pre_vic(self):
        self._link()
        if self.feedback:
            self._update_vic()
        self.s = self.m.update_step_vars(self.s, self.cal, self.group.step,
                                         forced=self._forced)
        self._changed()

    def pass1(self):
        """crop_adjust_requirement (crop_adjust_requirement.c:7-97), after
        irr_run_requirement set the flags (VIC-WUR 3.0
        plugin_simulation.c:103-107)."""
        irr = self.group.roles.get('irrigation')
        if irr is None or irr.st is None:
            return
        if not self.dist_season:
            raise ComputeError('%s: crop_adjust_requirement reads the sowing '
                               'day_in_year, which C sets with '
                               'WOFOST_DIST_SEASON only (strpdmy leaves it '
                               'unset)' % self.name)
        step = self.group.step
        irr.st['flag'] = CF.adjust_requirement(
            irr.st['flag'], irr._cvday, self.view.af.to(torch.float64),
            irr.act, irr.tix, irr.omax, self.active, self.tile,
            self._sfull(('cultivating',))['cultivating'], self.start_doy,
            self.doy_t[step],
            self.leap_t[step])

    # -- the feedback --------------------------------------------------------
    def _link(self):
        """Hand the irrigation member the crop tiles' Wcr and roots and the
        irr_wofost water (once)."""
        if self._linked:
            return
        irr = self.group.roles.get('irrigation')
        if irr is not None:
            irr.wofost_water = self.wofost_water
            if self.feedback:
                irr.wcr_fn = self._irr_wcr
                irr.root_fn = self._irr_root
        self._linked = True

    def _tiles(self):
        """crop_tiles [N, T] (the tiles a crop class of the cell maps to)
        and their list (host), from the VIC layout."""
        if getattr(self, '_ct', None) is None:
            T = self.group.layout()['n_tile']
            if self.veg_map is not None:
                tl = np.asarray(self.veg_map).astype(np.int64)
                ok = tl >= 0
            else:
                tl = np.asarray(torch.as_tensor(self.tile_cfg).cpu())
                ok = np.asarray(self.su.active).astype(bool)
            pres = (None if self.tile_present is None
                    else np.asarray(self.tile_present).astype(bool))
            ct = np.zeros((tl.shape[0], T), bool)
            for n in range(tl.shape[0]):
                for c in range(tl.shape[1]):
                    v = int(tl[n, c])
                    if ok[n, c] and (pres is None or (v < pres.shape[1]
                                                      and pres[n, v])):
                        ct[n, v] = True
            lst = sorted({int(v) for v in np.nonzero(ct)[1]})
            # the tiles crop_update_step_vars writes: an active class's
            # (the Set sums) and the crop tiles (Transform)
            cls_tiles = {int(v) for v in np.asarray(
                torch.as_tensor(self.tile_cfg).clamp(min=0).cpu())[
                    np.asarray(self.su.active).astype(bool)]}
            self._fbcols = tuple(sorted(cls_tiles | set(lst)))
            ct = torch.as_tensor(ct, device=self.dev)
            self._ct = (ct, lst, T)
        return self._ct

    def _update_vic(self):
        """crop_update_step_vars of every cell (U), on the crop state after
        crop_run of the day before."""
        if self.soil is None:
            raise ComputeError('%s: the feedback needs soil' % self.name)
        ct, lst, T = self._tiles()
        mon = self.dates[self.group.step][1]
        ccm = self.cc[:, :, mon - 1]
        if self.graph and torch.device(self.dev).type == 'cuda':
            fb = self._fb_graphed(ccm, ct, lst, T)
        else:
            fb = CF.update_step_vars(self._sfull(self.FB_KEYS), self.prm,
                                     ccm, self.af, self.active, self.tile,
                                     ct, lst, self.soil, T,
                                     cols=self._fbcols)
        self._fb = fb
        self._cols = None
        self._lai_prev = self._lai_cur
        self._lai_cur = torch.where(ct, fb['lai'], torch.zeros_like(
            fb['lai']))

    def _fb_graphed(self, ccm, ct, lst, T):
        """crop_update_step_vars replayed from a CUDA graph (some 10^3
        small ops; the same kernels on the same data, so the same bits).
        The state is written in place by crop_run, so its memory is the
        graph's input; the month's Cc is copied into the graph's buffer;
        the outputs are copied out (the feedback is kept by reference).
        A new state layout captures again."""
        key = (tuple(v.data_ptr() for v in self.s.values()),
               tuple(ccm.shape), tuple(lst), T)
        G = getattr(self, '_ugraph', None)
        if G is not None and G[0] == key:
            G[1].copy_(ccm)
            G[2].replay()
            return {k: v.clone() for k, v in G[3].items()}
        self.m.own(self.s)
        key = (tuple(v.data_ptr() for v in self.s.values()),
               tuple(ccm.shape), tuple(lst), T)
        # the eager call reads the graph's buffer too: the fused region's
        # guards (strides) then hold in the capture
        buf = ccm.clone()
        fb = CF.update_step_vars(self._sfull(self.FB_KEYS), self.prm, buf,
                                 self.af, self.active, self.tile, ct, lst,
                                 self.soil, T, cols=self._fbcols)
        self._ugraph = None
        torch.cuda.synchronize(self.dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = CF.update_step_vars(self._sfull(self.FB_KEYS), self.prm,
                                      buf, self.af, self.active, self.tile,
                                      ct, lst, self.soil, T,
                                      cols=self._fbcols)
        self._ugraph = (key, buf, g, out)
        return fb

    def _columns(self):
        """The feedback on VIC's columns (built once per U)."""
        if self._cols is None:
            G, fb = self.group, self._fb
            lay = G.layout()
            cell = G._idx('cell', self.dev)
            ct = self._tiles()[0]
            c = {k: fb[k].index_select(0, cell) for k in CF.HIST + CF.LIB}
            c['root_frac'] = G.to_columns(fb['root'])
            mask = ct.index_select(0, cell)
            base = getattr(self, '_wcr_base', None)
            if base is None:
                w = torch.as_tensor(G.vic.driver.soil().w_crit_mm,
                                    dtype=torch.float64, device=self.dev)
                base = w[:, None, :].expand(lay['n_col'], lay['n_tile'],
                                            w.shape[-1])
                self._wcr_base = base
            c['wcr'] = torch.where(mask[..., None], G.to_columns(fb['wcr']),
                                   base)
            self._cols = c
        return self._cols

    def _vic_field(self, k):
        return lambda t: self._columns()[k]

    def _irr_wcr(self):
        """The layer Wcr of each irrigated tile and band [N, NI, B, L]: the
        crop's on a crop tile (crop_update_step_vars.c:143-146, 190-191,
        220-221), the soil's elsewhere."""
        return self._irr_sel('wcr')

    def _irr_root(self):
        """veg_var root of each irrigated tile and band [N, NI, B, L]."""
        return self._irr_sel('root')

    def _irr_sel(self, k):
        irr = self.group.roles['irrigation']
        x = self._fb[k].index_select(1, irr.tix)            # [N,NI,B,L]
        own = (irr.wcr[:, None, None, :] if k == 'wcr'
               else irr.root[:, :, None, :]).expand_as(x)
        ct = self._tiles()[0].index_select(1, irr.tix)
        return torch.where(ct[:, :, None, None], x, own)

    def wofost_water(self):
        """irr_wofost's water of the step [N, NI, B] (irr_use.c:568-603)."""
        irr = self.group.roles['irrigation']
        step = self.group.step
        mon = self.dates[step][1]
        w, bad = CF.wofost_water(self.irr_list[step], self.cc[:, :, mon - 1],
                                 self.active, self.tile, irr.tix, irr.act,
                                 self.af, self.af.shape[1])
        self.flag(bad)
        return w

    def lai_prev_columns(self):
        """(LAI [col, T], mask [col, T]): the crop tiles' LAI VIC's veg_var
        holds when lu_apply runs in F (the last step's update_step_vars),
        or None."""
        if not self.feedback or self._lai_prev is None:
            return None
        G = self.group
        cell = G._idx('cell', self.dev)
        return (self._lai_prev.index_select(0, cell),
                self._tiles()[0].index_select(0, cell))

    def pass3(self):
        v, step = self.view, self.group.step
        if v is None:
            raise ComputeError('%s: no step view' % self.name)
        if step >= self.cal.T:
            raise ComputeError('%s: step %d beyond the dates given'
                               % (self.name, step))
        pe = self.series['pot_evap']
        tf = v.tile_fluxes
        if pe is not None:
            pot = pe.get(step).expand(self.m.shape)
        elif tf is not None and tf.get('pet') is not None:
            # cell_data pot_evap of the crop's tile and band (crop_run.c
            # :68-71); a band without area has none (0, initialize_soil.c)
            pot = self._tile_cb(self.group.from_columns(
                tf['pet']).to(self.dev, torch.float64))
        else:
            pot = torch.zeros(self.m.shape, dtype=torch.float64,
                              device=self.dev)
        cv = v.cv.to(self.dev, torch.float64).gather(1, self.tile)
        self._ring_guard()
        self.m.own(self.s)
        self._changed()
        x = (self._cell('air_temp', step), self._cell('shortwave', step),
             self._cell('co2', step), pot, self._stress(v))
        if self.graph and torch.device(self.dev).type == 'cuda':
            self._rates = self.m.day_graphed(self.s, self.cal, step, *x,
                                             cv=cv, cells=self._cells())
        else:
            self.s = self.m.crop_run_inplace(self.s, self.cal, step, *x,
                                             cv=cv, cells=self._cells())
            self._rates = self.m.rate_outputs_cells(self.s, self.cal, step,
                                                    cells=self._cells())
        self.flag(self.s['err'] != 0)

    #: the crop state fields crop_update_step_vars reads
    FB_KEYS = ('st_LAI', 'st_Development', 'st_RootDepth',
               'met_PotEvaptrans')

    def _sfull(self, keys):
        """Fields of the crop state on [N, B, NC] units (a lane state's
        expanded, WofostNPK.full_state)."""
        return self.m.full_state(self.s, keys)

    def _ring_supported(self):
        """The lanes' leaf sums run on the live classes (every chunk)."""
        from ...crop.wofost_npk import leaf_kernels as LK
        m = self.m
        if not m.lanes or m.exact_sums:
            return False
        lp = m._lptr
        rows = [lp[c1] - lp[c0] for c0, c1 in m.chunks(self._cells())]
        return all(LK.supported(m.K, r) for r in rows if r)

    def _ring0(self):
        """The leaf-class ring of a new state (see leaf_ring)."""
        m = self.m
        if self.leaf_ring in (None, 'auto') or not self._ring_supported():
            return m.K
        return max(2, min(m.K, int(self.leaf_ring)))

    def _ring_guard(self):
        """Grow the leaf-class ring before a unit's classes could fill it.
        A unit gains at most one class a day, so the largest t - h read
        a days ago bounds today's by + a; it is read without a host sync
        (a pinned copy, taken up the next day) and the ring grows (a
        sync, the classes moved to their new slots) when the bound
        reaches R - 1. ERR_RING still flags an overflow."""
        s, K = self.s, self.m.K
        R = s['lw'].shape[-1]
        if R >= K or s['lw'].dim() != 2:
            return
        q = self._ringq
        if q is not None and q[0].query():
            self._wmax, self._wage = int(q[1]), 0
            self._ringq = None
        if not hasattr(self, '_wmax'):
            self._wmax, self._wage = int((s['lt'] - s['lh']).max()), 0
        self._wage += 1
        if self._wmax + self._wage + 1 >= R:
            w = int((s['lt'] - s['lh']).max())
            R2 = R
            while R2 <= w + 32:
                R2 *= 2
            self._ring_grow(min(K, R2))
            self._wmax, self._wage = w, 1
        if self._ringq is None and torch.device(self.dev).type == 'cuda':
            buf = getattr(self, '_ringbuf', None)
            if buf is None:
                buf = self._ringbuf = torch.zeros((), dtype=torch.int32,
                                                  pin_memory=True)
            buf.copy_((s['lt'] - s['lh']).amax(), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            self._ringq = (ev, buf)

    def _ring_grow(self, R2):
        """The leaf classes moved to a ring of R2 slots."""
        m, s = self.m, self.s
        lane = m.lane
        h = m.lane_values(s, 'lh').long()
        t = m.lane_values(s, 'lt').long()
        R = s['lw'].shape[-1]
        U = h.shape[0]
        j = torch.arange(R2, device=lane.device, dtype=torch.int32)
        new = {n: torch.zeros(U, R2, dtype=s[n].dtype, device=lane.device)
               for n in ('lw', 'la', 'lage')}
        step = 1 << 18                  # rows at a time: small temporaries
        for u0 in range(0, U, step):
            hh = h[u0:u0 + step, None].to(torch.int32)
            k = hh + (j - hh) % R2
            live = k < t[u0:u0 + step, None]
            src = (k % R).long()
            for n, y in new.items():
                x = s[n][u0:u0 + step].gather(1, src)
                y[u0:u0 + step] = torch.where(live, x, torch.zeros_like(x))
        s.update(new)
        self._changed()

    def _cells(self):
        N, B, NC = self.m.shape
        return (max(1, self.chunk_lanes // (B * NC)) if self.chunk_lanes
                else None)

    def _changed(self):
        """The state moved on: a cached snapshot is stale."""
        self._ver += 1
        self._snap = None

    def _message(self):
        return ('%s: a WOFOST unit failed (state err bits: crop cover or Cv '
                '<= 0 while growing, every leaf class dead, |lat| > 90)'
                % self.name)

    def put_data(self):
        r = getattr(self, '_rates', None)
        if r is None:
            return
        self._rates = None
        tf = None if self.view is None else self.view.tile_fluxes
        if tf is not None and tf.get('transp') is not None:
            # crop_output.c:1003-1032: VIC's fluxes of the step on the
            # crop's tile, x AreaFract x Cc x growing
            G = self.group
            fc = lambda x: G.from_columns(x).to(self.dev, torch.float64)
            mon = self.dates[self.group.step][1]
            r = dict(r)
            r.update(CF.evap_outputs(
                fc(tf['esoil']), fc(tf['transp']), fc(tf['canop']),
                self.cc[:, :, mon - 1], self.af,
                self._sfull(('growing',))['growing'],
                self.active, self.tile))
        exp = dict(crop_lai=r['OUT_CROP_LAI'].clone())
        ser = {k: x.clone().unsqueeze(0) for k, x in r.items()}
        self._pending = (Balance(n_terms=1, scale=self.zero()), exp, ser)

    # -- the VIC side --------------------------------------------------------
    def core_hooks(self, core):
        core.water_stress = True
        core.tile_fluxes = True
        vic = self.group.vic
        last = None if vic is None else vic._wur_last
        if last is not None and last.get('water_stress_layer') is not None:
            core.stress_init = last['water_stress_layer']
        if self.feedback:
            # crop_update_step_vars writes veg_hist[NR], veg_lib, veg_var
            # root and layer Wcr of the crop tiles in U; update_step_vars
            # copies veg_hist into veg_var in V (VIC-WUR 3.0
            # update_step_vars.c:56-64): the veg and wcr hooks, read before
            # every step
            mos = vic.driver.mosaic()
            if not getattr(self, '_ov_ok', False) and \
                    mos.overstory is not None and bool(
                        (mos.overstory.to(torch.bool).cpu()
                         & self._tiles()[0].cpu().index_select(
                             0, self.group._idx('cell', 'cpu'))).any()):
                raise ComputeError('%s: a crop tile with an overstory (C '
                                   'sets veg_lib overstory false, '
                                   'crop_update_step_vars.c:53); not '
                                   'carried by the veg hook' % self.name)
            self._ov_ok = True             # a static check, once
            core.veg.update({k: self._vic_field(k) for k in
                             CF.HIST + CF.LIB + ('root_frac',)})
            core.veg_mask = self._tiles()[0].index_select(
                0, self.group._idx('cell', self.dev))
            core.wcr = self._vic_field('wcr')

    # -- w3_restart ----------------------------------------------------------
    def _w3_reset(self, step):
        """No crop or WOFOST state is in VIC-WUR 3.0's state file
        (plugin_store, plugin_simulation.c:224-237): a new run's units are
        crop_init / wofost_init's (`init_state`), and VIC's veg_var holds no
        crop LAI, roots or Wcr until the first crop_update_step_vars."""
        self.s = self.m.init_state()
        self._changed()
        self._fb = None
        self._cols = None
        self._lai_cur = self._lai_prev = None
        self._rates = None

    # -- state ---------------------------------------------------------------
    def _state(self):
        # crop_run writes the state in place: a snapshot is a copy. The
        # snapshots taken at one point of the run (the runtime's window,
        # the VIC window's group snapshot, a checkpoint) share one copy:
        # it is kept until the state moves on (`_changed`)
        if self.s is None:
            return {}
        if self._snap is not None and self._snap[0] == self._ver:
            return dict(self._snap[1])
        host = self.snapshot_on == 'host' or (
            self.snapshot_on == 'auto' and sum(
                x.numel() * x.element_size() for x in self.s.values())
            > 1 << 30)
        d = {k: self._copy(v, host) for k, v in self.s.items()}
        if self._lai_cur is not None:
            d['__lai_cur'] = self._lai_cur
        if self._fb is not None:
            # the last U's feedback: P1-P3 of that day (at the next
            # window's start) read the crop tiles' Wcr and roots (built
            # anew every U, never written in place: kept by reference)
            d.update({'__fb_' + k: v for k, v in self._fb.items()})
        self._snap = (self._ver, d)
        return dict(d)

    @staticmethod
    def _copy(v, host):
        if not host or v.device.type == 'cpu':
            return v.clone()
        h = torch.empty(v.shape, dtype=v.dtype, device='cpu',
                        pin_memory=True)
        h.copy_(v)
        return h

    def _load(self, blob):
        blob = dict(blob or {})
        lc = blob.pop('__lai_cur', None)
        fb = {k[5:]: blob.pop(k) for k in list(blob) if k.startswith('__fb_')}
        # a copy the state owns (crop_run writes it in place); into the
        # live tensors where they match (a rollback on a large domain then
        # needs no second copy of the state on the device)
        own = lambda v: (v.to(self.dev) if v.device.type == 'cpu'
                         and torch.device(self.dev).type != 'cpu'
                         else v.to(self.dev).clone())
        cur = self.s or {}

        def into(k, v):
            x = cur.get(k)
            if torch.is_tensor(x) and x.shape == v.shape and \
                    x.dtype == v.dtype and x.is_contiguous():
                x.copy_(v)
                return x
            return own(v)
        self.s = {k: into(k, v) for k, v in blob.items()} if blob else None
        self._changed()
        self._lai_cur = None if lc is None else lc.to(self.dev)
        self._lai_prev = None
        self._fb = {k: v.to(self.dev) for k, v in fb.items()} \
            if fb else None
        self._cols = None
        self._rates = None
