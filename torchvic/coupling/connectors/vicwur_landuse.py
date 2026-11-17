# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_landuse: VIC-WUR land use (C11) as a connector.

Kernels torchvic.vicwur.landuse; C paths VIC-WUR 3.0 plugins/landuse/src.

  F  `pre_force`: lu_forcing (lu_force.c:62-95): on the run's first step
     and on every new CV record the forced Cv per vegetation class is held
     and lu_apply is due
  F  `at_state` (the group's state phase, run by VIC's vicwur_core
     state_fn at the start of the step): lu_apply (lu_apply.c:995-1140) on
     VIC's carried tile states -- map_cv_forcing, cv_update, the paddy /
     Pbare pass, the water pass with its post-fixes, energy of the snow
     pack, the irrigation leftover of the changed irrigated tiles to the
     river ring -- and the new Cv, which VIC uses from this step on (the
     core's `cv`) and the step view and the other members read
     (`cv_columns`)
  O  `put_data`: OUT_CV per class (lu_output.c) and the ledger

TorchVIC's tile axis is the vegetation class (tile k = class k + 1; VIC's
own bare tile, if a cell has one, is tile NVEGTYPES); the C's compact tiles
are the active ones in class order, so the kernel's sums over tiles visit
the same values in the same order. A class with no tile in the parameter
file is not a tile here either (forced Cv > 0 there is the C's log_err).

VIC states (basin's names, [column, tile, ...], to [cell, tile, band]):
moist; wdew; snow.swq_c (the C swq, m), snow.pack_water, snow.surf_water
(snow.ice = swq - surf_water - pack_water), snow.surf_temp,
snow.pack_temp, snow.cold_content, snow.density, snow.depth, last_snow,
melting; canopy.int_snow (snow_canopy); the thermal nodes 0 and 1 of the
water-balance path (wb_t0, wb_t1: the T[0], T[1] the ground flux of the
next step starts from, calc_surf_energy_bal.c:574-578), with their heat
capacities from the moisture (distribute_node_moisture_properties, and the
exact mode's Cs[2] overflow: lu_heat_capacity_fn); node 2 is not carried
(its temperature reaches no water flux) and is not written. Not reproduced:
the carbon pools (CARBON is not wired with the hooks). FROZEN_SOIL is
refused by the hooks.

Options
    cv        Series spec of the forced Cv [N, K] per vegetation class (K =
              NVEGTYPES), e.g. {'forcing': <reader with get(step), reads(step)>}
    soil      dict: max_moist [N, L] mm (the C's), lat [N]; thermal (the
              node and layer statics of the energy step, vicwur_setup.
              thermal_statics; without it the node temperatures are left
              per tile)
    pbare     PBARE (1-based class) or None; NVEGTYPES skips the paddy pass
    mode      'exact' (the C, quirks included) or 'fixed'
    max_surface_swe, dens_max    snow constants (0.125 m, 400 kg m-3)
    air_temp  Series spec [N] (C) for the new-snow density; default the
              attached case's own air temperature

State (checkpoint): the Cv in force, the held forced Cv, the due flag.

Ledger (m3 per day): what the leftover hand-over moves: I_ext = leftover
taken from the irrigation store (+ water the exact mode's unweighted
per-band hand-over creates, detail `exact_created`), Q_out = water put
into the river ring (the router books it as ring_in). The VIC-side terms
of the exact mode (`lost`, `orphan`, `subthreshold`) change VIC's states
between two steps and show in VIC's own water error, as in C; they are in
`detail`.
"""
import numpy as np
import torch

from ..connector import Balance, ComputeError
from ..spec import CouplingError, VarSpec
from ...vicwur import landuse as LU
from .vicwur_common import GRID, STEP_S, Series, WurConnector, as_f64

_SNOW = (('pack_water', 'snow.pack_water'), ('surf_water', 'snow.surf_water'),
         ('surf_temp', 'snow.surf_temp'), ('pack_temp', 'snow.pack_temp'),
         ('coldcontent', 'snow.cold_content'), ('density', 'snow.density'),
         ('depth', 'snow.depth'))


class VicWurLanduseConnector(WurConnector):
    name = 'vicwur_landuse'
    version = '1'
    kind = 'landuse'
    role = 'landuse'
    flags = dict(force_landuse=True)

    def __init__(self, cv=None, soil=None, pbare=None, mode='fixed',
                 max_surface_swe=0.125, dens_max=400.0, air_temp=None,
                 group=None, **config):
        super().__init__(group=group, **config)
        if cv is None or soil is None:
            raise CouplingError('%s: give cv and soil' % self.name)
        miss = [k for k in ('max_moist', 'lat') if k not in soil]
        if miss:
            raise CouplingError('%s: soil lacks %s' % (self.name, miss))
        if mode not in ('exact', 'fixed'):
            raise CouplingError("%s: mode must be 'exact' or 'fixed'"
                                % self.name)
        self.f = Series(cv, 'cv')
        self.soil_cfg = soil
        self.pbare = None if pbare is None else int(pbare)
        self.mode = mode
        self.msw, self.dens_max = float(max_surface_swe), float(dens_max)
        self.air = None if air_temp is None else Series(air_temp, 'air_temp')
        self.N = int(np.asarray(soil['lat']).shape[0])
        self._check_config()
        self.outputs = (VarSpec('cv', '1', ('cell', 'tile'), kind='state',
                                grid=GRID, description='OUT_CV'),)
        self.owned_state = ('%s.cv' % self.name,)
        self.cv = None
        self._forced = None
        self._due = False
        #: the run's first step (C's current == 0): 0, or the restart step
        #: of a w3_restart
        self._first = 0
        #: exact mode: (mask [cell, tile], {state name: [col, tile, ...]})
        #: of the active tiles the last record closed (Cv <= 0), see _thaw
        self._frozen = None

    def _init(self, context):
        d = self.dev
        self.f.to(d)
        if self.air is not None:
            self.air.to(d)
        self.mm = as_f64(self.soil_cfg['max_moist'], d)
        self.lat = as_f64(self.soil_cfg['lat'], d)
        if self.mm.shape[0] != self.N:
            raise CouplingError('%s: soil.max_moist must be [N, L]'
                                % self.name)
        th = self.soil_cfg.get('thermal')
        self.thermal = None
        if th is not None:
            self.thermal = {k: (as_f64(v, d) if k != 'fs_active' else
                                torch.as_tensor(v, device=d).bool())
                            for k, v in th.items()}
            self.cs_fn = LU.lu_heat_capacity_fn(self.thermal, self.mode)
        self._st = None

    # -- statics from the attached VIC ----------------------------------------
    def _statics(self):
        if self._st is not None:
            return self._st
        G = self.group
        if G.vic is None:
            raise ComputeError('%s: land use needs VIC in the group (it edits '
                               "VIC's tile states)" % self.name)
        L = G.layout()
        if L['n_cell'] != self.N:
            raise ComputeError('%s: %d cells, VIC has %d' % (
                self.name, self.N, L['n_cell']))
        d = self.dev
        mos = G.vic.driver.mosaic()
        cv0 = G.cell_rows(mos.cv).to(d, torch.float64)
        nt = cv0.shape[1]
        k = torch.arange(nt, device=d)
        active = cv0 > 0                                    # nv_active
        S = dict(cv0=cv0, active=active, nt=nt)
        S['over'] = (G.cell_rows(torch.as_tensor(mos.overstory)).to(d)
                     .bool() if getattr(mos, 'overstory', None) is not None
                     else torch.zeros_like(active))
        S['k'] = k
        irr = G.roles.get('irrigation')
        tc = torch.full((self.N, nt), -1, dtype=torch.long, device=d)
        S['tile_k'] = k
        S['tc'] = tc
        S['irr'] = irr
        paddy = torch.zeros_like(active)
        if irr is not None:
            for i in range(irr.NI):
                if bool(irr.irr_np['paddy'].reshape(-1)[i]):
                    t = int(irr.vc[i]) - 1
                    paddy[:, t] = irr.act[:, i].to(d)
        S['paddy'] = paddy & active
        pb = torch.zeros_like(active)
        if self.pbare is not None and self.pbare - 1 < nt:
            pb[:, self.pbare - 1] = True
        S['pbare'] = pb & active
        tf = (L['bands'] or {}).get('tfactor')
        if tf is not None:
            tfc = torch.zeros(self.N, L['area_fract'].shape[1],
                              dtype=torch.float64, device=d)
            tfc[L['cell'].to(d), L['band'].to(d)] = torch.as_tensor(
                np.asarray(tf), dtype=torch.float64, device=d)
        else:
            tfc = torch.zeros_like(L['area_fract'].to(d))
        S['tfactor'] = tfc
        S['af'] = L['area_fract'].to(d)
        self._st = S
        return S

    def cv_columns(self):
        """The Cv in force on VIC's columns [col, tile] (None before the
        first record is applied: VIC's own)."""
        if self.cv is None:
            return None
        dev = self.group.vic.driver.mosaic().cv.device
        # the same tensor while the Cv is the same (lu_apply replaces it):
        # VIC's hook then neither gathers nor checks it again
        c = getattr(self, '_cv_cols', None)
        if c is not None and c[0] is self.cv \
                and c[2].device == torch.device(dev):
            return c[2]
        x = self.cv.to(dev).index_select(0, self.group._idx('cell', dev))
        self._cv_cols = (self.cv, None, x)
        return x

    def wants_state(self):
        return True

    # -- hooks ---------------------------------------------------------------
    def pre_force(self):
        st = self.group.step
        if self.f.reads(st) or st == self._first:
            x = self.f.get(st)
            if x.dim() != 2 or x.shape[0] != self.N:
                raise ComputeError('%s: the forced Cv must be [N, K]'
                                   % self.name)
            self._forced = x
            self._due = True

    def _air(self, step):
        if self.air is not None:
            return self.air.get(step).reshape(-1).expand(self.N)
        a = self.group.met('tair', step)
        if a is None:
            raise ComputeError('%s: no air temperature (give air_temp)'
                               % self.name)
        return torch.as_tensor(a, dtype=torch.float64, device=self.dev)

    def at_state(self, states):
        """lu_apply on VIC's carried states at the step start."""
        if not self._due:
            return
        self._due = False
        G, S, d = self.group, self._statics(), self.dev
        step = G.step
        self._thaw(states)
        cv_old = self.cv if self.cv is not None else S['cv0']
        K = self._forced.shape[1]
        tile_class = torch.where(S['active'] & (S['k'] < K),
                                 S['k'].expand(self.N, -1),
                                 torch.full_like(S['tc'], -1))
        cvf, fatal_map = LU.map_cv_forcing(self._forced, tile_class)
        cvn, dcv, _, fatal_cv, _ = LU.cv_update(cvf, cv_old, S['active'])
        bare = S['active'] & (S['k'] == K)
        # VIC's states on [cell, tile, band]
        fc = lambda x, fill=0.0: G.from_columns(x, fill).to(d)
        get = states.get
        names = set(states.names())
        st = {}
        st['moist'] = fc(get('moist')).to(torch.float64)
        ice = get('soil_ice')
        st['ice'] = (torch.zeros_like(st['moist']) if ice is None else
                     fc(ice.reshape(get('moist').shape)).to(torch.float64)
                     ).unsqueeze(-1)
        st['wdew'] = fc(get('wdew')).to(torch.float64)
        snow = 'snow.swq_c' in names
        zero = torch.zeros_like(st['wdew'])
        st['swq'] = fc(get('snow.swq_c')).double() if snow else zero
        for k, n in _SNOW:
            st[k] = fc(get(n)).double() if n in names else zero
        st['snow_canopy'] = (fc(get('canopy.int_snow')).double()
                             if 'canopy.int_snow' in names else zero)
        st['last_snow'] = (fc(get('last_snow')) if 'last_snow' in names
                           else zero)
        st['melting'] = (fc(get('melting'), False).bool()
                         if 'melting' in names else zero.bool())
        energy = self.thermal is not None and {'wb_t0', 'wb_t1'} <= names
        if energy:
            t0 = fc(get('wb_t0')).double()
            t1 = fc(get('wb_t1')).double()
            # node 2 is not carried: its value reaches neither node 0 nor 1
            st['t_node'] = torch.stack([t0, t1, torch.zeros_like(t0)], -1)
            dz_node = self.thermal['dz_node']
            cs_fn = self.cs_fn
        else:
            st['t_node'] = torch.zeros_like(st['wdew']).unsqueeze(-1)
            dz_node = torch.zeros(self.N, 1, dtype=torch.float64, device=d)
            cs_fn = lambda m, i, t: torch.zeros_like(t)
        irr = S['irr']
        route = irr is not None and G.router is not None
        r = G.router
        if route:
            irr._ensure(S['af'].shape[1])
            st['leftover'] = irr.st['leftover']
            phys = r.logical(r._bins)
            st['ring'] = r.ring.index_select(0, phys).index_select(
                1, r.pos).T.contiguous()
            itile = torch.where(irr.act, irr.tix.expand(self.N, -1),
                                torch.full_like(irr.act, -1,
                                                dtype=torch.long))
        dates = G.vic.driver.dates
        mos = G.vic.driver.mosaic()
        # veg_var LAI as lu_apply finds it in F: 0 before the first
        # update_step_vars (initialize_veg.c:46), then the last step's
        # (update_step_vars.c:61 runs in V, after F)
        if step == self._first:
            lai = torch.zeros_like(mos.cv)
        else:
            lai = mos.lai
            lai = lai[:, :, dates[step - 1].month - 1] if lai.dim() == 3 \
                else lai
            crop = G.roles.get('crop')
            cl = None if crop is None else crop.lai_prev_columns()
            if cl is not None:
                # the crop tiles' veg_var LAI is crop_update_step_vars'
                # (VIC-WUR 3.0 crop_update_step_vars.c:106, copied in V by
                # update_step_vars.c:61) of the step before
                lai = torch.where(cl[1], cl[0].to(lai.dtype), lai)
        doy = dates[step].timetuple().tm_yday
        P = dict(area_fract=S['af'], max_moist=self.mm, overstory=S['over'],
                 active=S['active'], bare=bare, paddy=S['paddy'],
                 pbare=S['pbare'], lai=fc(lai).double(),
                 dz_node=dz_node,
                 max_surface_swe=self.msw, doy=doy, lat=self.lat,
                 new_density=LU.band_new_density(self._air(step),
                                                 S['tfactor'],
                                                 dens_max=self.dens_max))
        paddy_x = irr is not None and self.pbare is not None and \
            self.pbare != K
        if paddy_x:
            order = torch.full((self.N, irr.NI + 1), -1, dtype=torch.long,
                               device=d)
            for i in range(irr.NI):
                if bool(irr.irr_np['paddy'].reshape(-1)[i]):
                    order[:, i] = torch.where(irr.act[:, i], irr.tix[i],
                                              torch.full_like(order[:, i],
                                                              -1))
            pbi = torch.where(S['pbare'].any(1),
                              S['pbare'].long().argmax(1),
                              torch.full((self.N,), -1, dtype=torch.long,
                                         device=d))
            order[:, irr.NI] = pbi
            P['paddy_order'] = order
        if route:
            P.update(irr_tile=itile, area=r.to_cells(r.area), dt=STEP_S,
                     rps=r.nsub, uh_length=r.L)
        st1, terms = LU.apply(st, cv_old, cvn, P, mode=self.mode,
                              paddy_exchange=paddy_x, route_leftover=route,
                              cs_fn=cs_fn, full_energy=False)
        self.flag(fatal_map | fatal_cv | terms['fatal'])
        # back into VIC (columns)
        tc = lambda x: G.to_columns(x)
        mo = get('moist')
        states.set('moist', tc(st1['moist']).to(mo.dtype))
        states.set('wdew', tc(st1['wdew']).to(get('wdew').dtype))
        if snow:
            sw, pw = st1['surf_water'], st1['pack_water']
            states.set('snow.swq_c', tc(st1['swq']))
            states.set('snow.ice', tc(st1['swq'] - sw - pw))
            for k, n in _SNOW:
                if n in names:
                    states.set(n, tc(st1[k]))
        if 'melting' in names:
            states.set('melting', tc(st1['melting']))
        if energy:
            states.set('wb_t0', tc(st1['t_node'][..., 0]))
            states.set('wb_t1', tc(st1['t_node'][..., 1]))
        if 'canopy.int_snow' in names:
            states.set('canopy.int_snow', tc(st1['snow_canopy']))
        if route:
            irr.st['leftover'] = st1['leftover']
            cells = r.pos
            m0 = G.ring_mass(cells)
            new = st1['ring'].T                     # [R, N] logical, cells
            r.ring.index_put_((phys[:, None].expand(-1, self.N),
                               cells[None, :].expand(r.R, -1)), new)
            r.discharge.index_copy_(0, cells, terms['discharge'])
            r.stream.index_copy_(0, cells, terms['stream'])
            G.book_ring(cells, m0)
            a = r.to_cells(r.area) / 1000.
            rem = (terms['leftover_removed'] * a).sum()
            put = ((terms['leftover_routed'] + terms['leftover_clip'])
                   * a).sum()
            self.acc('removed', rem)
            self.acc('to_ring', put)
        af = S['af']
        ca = (r.to_cells(r.area) if r is not None else None)
        for k in ('lost', 'orphan', 'subthreshold'):
            v = (terms[k] * af).sum(1)
            self.acc('vic_' + k, (v * ca / 1000.).sum() if ca is not None
                     else v.sum())
        self.cv = cvn
        self._freeze(states, S['active'] & (cvn <= 0))
        self.acc('applied', 1.0)

    # -- tiles without cover (exact) -------------------------------------------
    def _freeze(self, states, off):
        """exact: keep the states of the active tiles this record closed
        (Cv <= 0). VIC-WUR's vic_run solves a tile only while its Cv > 0
        (VIC-WUR 3.0 vic_run.c:137, :388; put_data.c:183): a closed tile
        keeps the states the record left until a later record gives it
        cover again. TorchVIC's step runs every tile of the mosaic (weight
        0 in every cell sum), so the states it carries for a closed tile
        drift. They reach a flux only when a later record re-opens the tile
        with a gain below MINCOVERAGECHANGE (lu_apply keeps the tile's own
        states then, VIC-WUR 3.0 landuse.h:31): in exact mode _thaw puts
        the kept states back before that record. Mode fixed has no
        threshold (a re-opened tile takes the pool's states), so it needs
        none of this."""
        if self.mode != 'exact':
            return
        snap = {}
        for n in states.names():
            x = states.get(n)
            if torch.is_tensor(x) and x.dim() >= 2:
                snap[n] = x.clone()
        self._frozen = (off.clone(), snap)

    def _thaw(self, states):
        """exact: the kept states of the tiles closed since the last record
        back into VIC's columns (see _freeze)."""
        if self.mode != 'exact' or self._frozen is None:
            return
        off, snap = self._frozen
        for n, x in snap.items():
            cur = states.get(n)
            if cur is None or tuple(cur.shape) != tuple(x.shape):
                continue
            m = off.to(cur.device).index_select(
                0, self.group._idx('cell', cur.device))
            if tuple(m.shape) != tuple(cur.shape[:2]):
                continue
            m = m.reshape(tuple(m.shape) + (1,) * (cur.dim() - 2))
            states.set(n, torch.where(m, x.to(cur.device, cur.dtype), cur))

    def put_data(self):
        n = self.pop('applied')
        rem, put = self.pop('removed'), self.pop('to_ring')
        det = {k: self.pop('vic_' + k) for k in ('lost', 'orphan',
                                                   'subthreshold')}
        z = self.zero()
        rem = z + rem
        put = z + put
        created = put - rem
        bal = Balance(I_ext=rem + created, Q_out=put,
                      n_terms=max(self.N * 8, 1), scale=rem.abs() + put.abs(),
                      detail=dict(leftover_removed=rem, to_ring=put,
                                  exact_created=created,
                                  lu_apply_calls=z + n,
                                  **{'vic_' + k: v for k, v in det.items()}))
        cv = self.cv if self.cv is not None else self._statics()['cv0']
        exp = dict(cv=cv.clone())
        self._pending = (bal, exp, {k: x.clone().unsqueeze(0)
                                    for k, x in exp.items()})

    # -- the VIC side --------------------------------------------------------
    def core_hooks(self, core):
        core.cv = lambda t: self.cv_columns()

    # -- w3_restart ----------------------------------------------------------
    def _w3_reset(self, step):
        """Cv is not in VIC-WUR 3.0's state file: a new run's veg_con Cv is the
        parameter file's (vic_init.c:1136-1155, 1285-1305), the held
        record is gone and the first step (current == 0) reads the CV
        record and runs lu_apply from the parameter Cv (plugin_forcing_
        utils.c:174-185, lu_force.c:69-94), with veg_var LAI 0
        (initialize_veg.c:46; update_step_vars runs after plugin_force)."""
        self.cv = None
        self._forced = None
        self._due = False
        self._first = step

    # -- state ---------------------------------------------------------------
    def _state(self):
        fz = self._frozen
        return dict(cv=None if self.cv is None else self.cv.clone(),
                    forced=None if self._forced is None
                    else self._forced.clone(), due=self._due,
                    # replaced whole at a record, never edited in place
                    frozen=fz)

    def _load(self, blob):
        c, f = blob.get('cv'), blob.get('forced')
        self.cv = None if c is None else c.to(self.dev).clone()
        self._forced = None if f is None else f.to(self.dev).clone()
        self._due = bool(blob.get('due', False))
        fz = blob.get('frozen')
        self._frozen = None if fz is None else (fz[0].to(self.dev), fz[1])

