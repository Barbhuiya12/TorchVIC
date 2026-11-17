# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR's hooks in vic_run (C13); see docs/connectors/vicwur.md.

VIC-WUR 3.0.0 (<data root>/build/vicwur3/src_tree/vic) changes vic_run in
two kinds of place, and TorchVIC keeps the two apart. File names with line
numbers in this docstring refer to that source.

* **The VIC-WUR 3.0 core** -- what VIC-WUR 3.0's vic_run does differently
  from VIC 5.1.0 with every plugin off. It is the VIC core `vicwur3`
  (torchvic.core.version), selected only by an explicit
  `set_core('vicwur3')` / `case.load(..., core='vicwur3')`; each difference
  is switched where it lives:
    calc_root_fraction.c:137    `zone++`: a zone reaching past a layer
                                loses its lower part (core.vegetation.
                                root_fractions_w3)
    vic_force.c:428-441         fcanopy floored at MIN_FCANOPY, no
                                5.1.0 leafless rule (io.build)
    runoff.c:88-89, arno_evap.c:172-175
                                residual moisture formed as
                                (resid * depth) * 1000 (io.build)
    canopy_evap.c:341-372       SHARE_LAYER_MOIST: `spare_transp +=`
                                (every dry layer's shortfall, not the
                                last one's) and the `root_sum > 0` guard
                                (core.vegetation.transpiration,
                                share_fix)
    vic_alloc.c:128-129, vic_init.c:1130,1145-1191, get_global_param.c:76-77
                                NBARE: every tile at an index >= Nveg is
                                bare (io.build, `nbare`)
  G0a/G0b (plugins off) through the vicwur3 core equal VIC-WUR 3.0's outputs
  to the precision TorchVIC's 5.1.0 core has against VIC 5.1.0
  (tests/test_vicwur_core.py).

* **The hooks** -- the per-tile fields VIC-WUR 3.0's plugins write into
  all_vars between two vic_run calls, which vic_run then reads. They are
  this module's `VicwurCore`, handed to basin.simulate as `vicwur_core=`;
  None (the default) leaves every code path of the model as it was.
    ksat      layer[l].Ksat, read only in the drainage: runoff.c:156-157
              `Ksat[l] = layer[l].Ksat / runoff_steps_per_day`
              (vic_init.c:1634-1635 starts it at soil_con->Ksat;
              irr_run.c:62-67, :118, :153-156 write it for paddy)
    wcr       layer[l].Wcr, read by the transpiration:
              calc_surf_energy_bal.c:154-160, func_surf_energy_bal.c:342-352,
              snow_intercept.c:82-86 hand canopy_evap the layer values
              (vic_init.c:1636-1637 starts them at soil_con->Wcr;
              crop_update_step_vars.c:67,143-146,190-191,220-221 write them)
    veg       veg_var (LAI, fcanopy, albedo, displacement, roughness: from
              veg_hist[NR], update_step_vars.c:56-64), veg_lib (rmin,
              RGL, rarc, rad_atten, wind_atten, trunk_ratio, wind_h:
              vic_run.c:181-211,246, surface_fluxes.c:789-798,
              canopy_evap.c:95-101,300-305,411-416) and veg_var->root
              (vic_run.c:325,351; vic_init.c:1638-1639;
              crop_update_step_vars.c:128-135,183-184,213); per step,
              where `veg_mask` is set, the parameter file's value elsewhere
    co2       canopy resistance factor 1 / (1 + b_co2 (CO2/CO2REF - 1)) in
              calc_rc (penman.c:83-87) for the transpiration
              (canopy_evap.c:300-304, :411-416), with b_co2 per tile
              (vic_init.c:185-199) and Catm per step (vic_force.c:261-268;
              co2_update_step_vars.c:14-17), torchvic.vicwur.co2.
              compute_pot_evap takes it too (compute_pot_evap.c:70-71,
              surface_fluxes.c:789-798); TorchVIC forms OUT_PET on the
              energy-balance path only, where the hooks are not wired
    water_stress
              layer[l].water_stress = gsm_inv, stored by every transpiration
              call (canopy_evap.c:292-296 CASE 1, :403-405 CASE 2) and
              kept otherwise (a tile under snow, a leafless or bare tile:
              func_surf_energy_bal.c:756-762), and cell->water_stress =
              sum_l layer[l].water_stress * root[l] after every surface_fluxes
              (surface_fluxes.c:1046-1049); both start at 0
              (initialize_soil.c:62,66). Returned as out['vicwur'].

    states, cv
              `state_fn` reads and writes the carried tile states between
              two steps (TileStates), and `cv` sets the tile Cv from a
              step on: what lu_apply does to all_vars and veg_con
              (lu_apply.c:995-1140), for the land-use component.

Shapes are the run's columns: [col, tile, ...], a snow-band run's pseudo
cells included (`band_columns` maps a [cell, tile, band, ...] array onto
them). Every hook is a device tensor (or a callable of the step) read by the
step loop; nothing is read back to the host. `step_fn(t, core)`, when given,
is called before every model step and may replace any of the fields (a
replay or a connector writing the next day's values).

The hooks run on the water-balance path (FULL_ENERGY FALSE): on the padded
tile grid (CPU, or CUDA with TORCHVIC_LANES=0), and on a CUDA device on the
covered tiles (basin's LANE_COMPACT) and inside its CUDA graphs, where each
hook value is gathered to the lanes once (a hook handing the same tensor
again is not gathered again) into a buffer the graph reads, and the
outputs go back to the columns where they are read; bit for bit the
padded run's on the covered tiles. A Cv hook may not cover a tile the
call's lanes (cv > 0 at the call's start) do not hold: refused. basin.
simulate refuses the hooks where they are not wired (the energy-balance
path, sub-daily snow steps, lakes, CLOSE_ENERGY) rather than ignore them.
"""
import torch

from . import co2 as _co2

__all__ = ['VicwurCore', 'TileStates', 'TABLE_FIELDS', 'TILE_FIELDS',
           'band_columns', 'bands_from_columns']

#: veg_hist fields, the mosaic's [col, tile, 12 or step] tables
#: (VIC-WUR 3.0 update_step_vars.c:56-64)
TABLE_FIELDS = ('lai', 'fcanopy', 'albedo', 'displacement', 'veg_rough')
#: veg_lib fields and the root fractions, the mosaic's [col, tile(, layer)]
#: attributes (VIC-WUR 3.0 crop_update_step_vars.c:112-135)
TILE_FIELDS = ('rmin', 'rgl', 'rarc', 'rad_atten', 'wind_atten',
               'trunk_ratio', 'wind_h', 'root_frac')


def band_columns(x, snowbands, tile_axis=True):
    """A per-band array [cell, tile, band, ...] (or [cell, band, ...] with
    `tile_axis=False`) on a snow-band run's columns, [col, tile, ...]:
    column p is (snowbands['cell'][p], snowbands['band'][p])
    (core.snowbands.expand). `snowbands` None: the run has one column per
    cell and the band axis must be 1."""
    if snowbands is None:
        return x[:, :, 0] if tile_axis else x[:, 0]
    dev = x.device
    c = torch.as_tensor(snowbands['cell'], dtype=torch.long, device=dev)
    b = torch.as_tensor(snowbands['band'], dtype=torch.long, device=dev)
    if tile_axis:
        return x[c, :, b]
    return x[c, b]


class VicwurCore(object):
    """The hooks of one run (see the module docstring). Every field is
    optional; a field left None is VIC 5's value, read where VIC 5 reads it.

        ksat          [col, tile, layer] mm/day, or callable(t)
        wcr           [col, tile, layer] mm, or callable(t)
        veg           {field: value} with field in TABLE_FIELDS ([col, tile])
                      or TILE_FIELDS ([col, tile], root_frac [col, tile,
                      layer]); each a tensor or callable(t)
        veg_mask      [col, tile] bool, where `veg` applies (None: all)
        b_co2         [col, tile] (0 = no CO2 effect)
        catm          Catm (mixing ratio) per column: [col], [step, col] or
                      callable(t); None = CANOPY_CO2REF * 1e-6 (VIC-WUR 3.0
                      vic_force.c:261-268)
        co2ref        CANOPY_CO2REF, ppm (VIC-WUR 3.0
                      initialize_parameters.c:74)
        water_stress  True: carry and return the stored stresses
        stress_history  True (default): out['vicwur']['water_stress_layer']
                      holds every step's layer stress [step, col, tile,
                      layer]; False: only the end value
                      (water_stress_layer_end; what the coupling reads --
                      on a basin the history is GBs a window)
        stress_init   [col, tile, layer], the stored layer stress at the
                      start (None: 0, VIC-WUR 3.0 initialize_soil.c:62)
        diagnostics   True: also return each step's per-tile, per-layer
                      transpiration and per-tile bare-soil evaporation
        throughfall   'fixed' (the vicwur3 default through the group):
                      water-conserving, a leafless vegetated tile passes
                      all its rain. True or 'exact': VIC-WUR 3.0's
                      throughfall of a vegetated tile that is not VEG
                      (LAI 0) but keeps fcanopy > 0 (the MIN_FCANOPY floor,
                      vic_force.c:428-441): (1 - fcanopy) rain + fcanopy x
                      veg_var->throughfall (func_surf_energy_bal.c:785-786),
                      the latter the step's copy zeroed at
                      surface_fluxes.c:267, so the fcanopy share of the rain
                      is lost (a VIC-WUR 3.0 water leak; the R1 runs meet it
                      on the crop tiles between seasons)
        tile_fluxes   True: keep the last step's per-tile fluxes the crop
                      plugin reads (cell_data pot_evap, layer transp,
                      layer[0].esoil, veg_var canopyevap: VIC-WUR 3.0
                      crop_run.c:68-71, crop_output.c:1003-1031); the day
                      view's day_fluxes() and out['vicwur']['tile_fluxes_end']
                      hand them on (tf_pet, tf_transp, tf_esoil, tf_canop)
        step_fn       callable(t, core), called before every model step
        state_fn      callable(t, states), called before every model step
                      after step_fn: `states` (TileStates) reads and
                      writes the run's carried tile states (moisture, ice,
                      canopy store, snow, temperatures), as a plugin edits
                      all_vars between two vic_run calls
        cv            [col, tile] or callable(t) (None for a step: the
                      Cv in force): veg_con Cv from step t on, the tile
                      weights of every cell sum (VIC-WUR 3.0
                      lu_apply.c:995-1029)
        gwm           None, 'exact' or 'fixed': VIC-WUR 3.0's GWM soil
                      (runoff_GWM, core.column.step_gwm)
        transp        [col, tile, layer] mm per step, or callable(t): the
                      per-layer transpiration that replaces VIC's on the
                      tiles of `transp_mask` (VIC-CropSyst: CropSyst's
                      uptake, VIC_soil.cpp:231-232, layer[i].VCS.
                      transpiration_daily, read by runoff.c:370-374; the
                      caller maps its layers onto VIC's and gives 0 for a
                      layer that takes none). None: VIC 5's
        transp_mask   [col, tile] bool, where `transp` applies (None: all)
        wpwp_floor    VIC-CropSyst's minimum liquid moisture (runoff.c:
                      301-309): the residual in layer 0, Wpwp below. True
                      (every tile) or a [col, tile] bool mask; None: VIC 5's
                      residual everywhere
        vcs_drain     VIC-CropSyst's drainage law (runoff.c:575-621): True
                      (every tile) or a dict(irrigated=[col, tile] bool, optional): the 4.x
                      power law with the caller's expt (3 + 2 b_campbell,
                      cs_core.saxton_soil_params(vcs_drainage=True)), capped
                      at the water above Field_Capacity, none below it, and
                      redistributed to Field_Capacity on the irrigated
                      tiles. The Field_Capacity is the soil parameters'
                      `vcs_fc_mm` (saxton_soil_params with vcs_drainage).
                      None: VIC 5's
        canopy_crop   VIC-CropSyst's CROPSYST_HANDLE_CANOPY_EVAP (P7;
                      canopy_evap.c:609-611, VIC_land_unit_simulation.cpp:
                      353-360, 435-446): dict(mask=[col, tile] bool,
                      take=[col, tile] mm or callable(t) or None). On the
                      masked tiles VIC does not evaporate the canopy store
                      (canopyevap = 0); `take` is what the crop evaporated
                      of the store it was handed after the previous day
                      (the member reads the store as tile_fluxes['wdew']):
                      it leaves the store at the start of the step and is
                      that step's canopy evaporation (the reference books it
                      on the crop's own day: one day earlier). Needs
                      tile_fluxes for the read. None: VIC 5's
        crop_soil_evap  VIC-CropSyst's soil evaporation of the crop tiles
                      (canopy_evap.c:760-831, VIC_soil_hydrology.c:394-411):
                      dict(mask=[col, tile] bool, cover_green=, cover_total=
                      [col, tile] or callable(t)): on the masked tiles the
                      bare-soil evaporation is E = max(0, min((1 - green)
                      (moist_0 - 0.5 Wpwp_0), min((1 - green) ARNO(rc = 0,
                      rarc = 100), PET_SHORT (1 - total)))) instead of
                      ARNO's (1 - fcanopy) share. Needs tile_fluxes and
                      pet_short (the step's PET_SHORT). None: VIC 5's
        pet_short     True: tile_fluxes also holds tf_pet_short, PET_SHORT
                      with rc = 70 s/m and rarc = 0 (compute_pot_evap.c:
                      70-75) on the PET_SHORT reference crop (ref_veg_*
                      [2], global.c:46-58). Needs tile_fluxes
    """

    def __init__(self, ksat=None, wcr=None, veg=None, veg_mask=None,
                 b_co2=None, catm=None, co2ref=_co2.CANOPY_CO2REF,
                 water_stress=False, stress_init=None, diagnostics=False,
                 step_fn=None, state_fn=None, cv=None, tile_fluxes=False,
                 throughfall=False, gwm=None, transp=None,
                 transp_mask=None, wpwp_floor=None, pet_short=False,
                 vcs_drain=None, canopy_crop=None, crop_soil_evap=None):
        if gwm not in (None, 'exact', 'fixed'):
            raise ValueError("vicwur_core gwm: None, 'exact' or 'fixed'")
        #: GWM TRUE (VIC-WUR 3.0 surface_fluxes.c:1086-1093): runoff_GWM
        #: instead of runoff (core.column.step_gwm); 'fixed' does not create
        #: water in the Ksat cut. out['gwrecharge'] / ['recharge'] are
        #: OUT_GWRECHARGE / OUT_RECHARGE, in the water balance's outflow
        #: (VIC-WUR 3.0 put_data.c:537-538)
        self.gwm = gwm
        if wpwp_floor is not None and gwm is not None:
            raise ValueError('vicwur_core wpwp_floor is wired on the VIC '
                             'runoff only (not gwm)')
        if vcs_drain is not None and vcs_drain is not False \
                and gwm is not None:
            raise ValueError('vicwur_core vcs_drain is wired on the VIC '
                             'runoff only (not gwm)')
        if vcs_drain is True:
            vcs_drain = {}
        elif vcs_drain is False:
            vcs_drain = None
        if vcs_drain is not None:
            bad = set(vcs_drain) - {'irrigated'}
            if bad:
                raise ValueError('vicwur_core vcs_drain: unknown key(s) %s '
                                 "(known: irrigated)" % sorted(bad))
        self.vcs_drain = vcs_drain
        if canopy_crop is not None and canopy_crop is not False:
            canopy_crop = dict(canopy_crop)
            if canopy_crop.get('mask') is None \
                    or set(canopy_crop) - {'mask', 'take'}:
                raise ValueError('vicwur_core canopy_crop: dict(mask=, '
                                 'take=)')
        else:
            canopy_crop = None
        self.canopy_crop = canopy_crop
        if crop_soil_evap is not None and crop_soil_evap is not False:
            crop_soil_evap = dict(crop_soil_evap)
            if set(crop_soil_evap) != {'mask', 'cover_green', 'cover_total'}:
                raise ValueError('vicwur_core crop_soil_evap: dict(mask=, '
                                 'cover_green=, cover_total=)')
            if not (tile_fluxes and pet_short):
                raise ValueError('vicwur_core crop_soil_evap needs '
                                 'tile_fluxes and pet_short')
        else:
            crop_soil_evap = None
        self.crop_soil_evap = crop_soil_evap
        if pet_short and not tile_fluxes:
            raise ValueError('vicwur_core pet_short needs tile_fluxes')
        if transp_mask is not None and transp is None:
            raise ValueError('vicwur_core transp_mask needs transp')
        self.transp, self.transp_mask = transp, transp_mask
        self.wpwp_floor = (None if wpwp_floor is None
                           or wpwp_floor is False else wpwp_floor)
        self.pet_short = bool(pet_short)
        self.ksat, self.wcr = ksat, wcr
        self.veg = dict(veg or {})
        bad = [k for k in self.veg if k not in TABLE_FIELDS + TILE_FIELDS]
        if bad:
            raise ValueError('vicwur_core veg: unknown field(s) %s (known: %s)'
                             % (', '.join(bad),
                                ', '.join(TABLE_FIELDS + TILE_FIELDS)))
        self.veg_mask = veg_mask
        self.b_co2, self.catm, self.co2ref = b_co2, catm, float(co2ref)
        self.water_stress = bool(water_stress)
        self.stress_history = True
        self.stress_init = stress_init
        self.diagnostics = bool(diagnostics)
        self.tile_fluxes = bool(tile_fluxes)
        # True/'exact': VIC-WUR 3.0's leak; 'fixed': water-conserving (the
        # cover share of the rain is not lost); False/None: off
        if throughfall in (True, 'exact'):
            throughfall = 'exact'
        elif throughfall in (False, None):
            throughfall = False
        elif throughfall != 'fixed':
            raise ValueError("vicwur_core throughfall: False, 'exact' or "
                             "'fixed'")
        self.throughfall = throughfall
        self.step_fn = step_fn
        self.state_fn = state_fn
        self.cv = cv

    # -- per step ---------------------------------------------------------
    @staticmethod
    def _at(v, t):
        return v(t) if callable(v) else v

    def cv_at(self, t):
        return self._at(self.cv, t)

    def ksat_at(self, t):
        return self._at(self.ksat, t)

    def wcr_at(self, t):
        return self._at(self.wcr, t)

    def transp_at(self, t):
        return self._at(self.transp, t)

    def canopy_take_at(self, t):
        """The [col, tile] mm the crop takes from the store at step t (or
        None)."""
        return (None if self.canopy_crop is None
                else self._at(self.canopy_crop.get('take'), t))

    def co2factor_at(self, t, like):
        """The [col, tile] factor of step t, or None when no b_co2 is
        given (calc_rc then runs VIC 5's expression)."""
        if self.b_co2 is None:
            return None
        c = self._at(self.catm, t)
        if c is None:
            ppm = torch.full_like(like[:, 0], self.co2ref
                                  * _co2.PPM_TO_MIXRATIO)
        else:
            c = torch.as_tensor(c, dtype=like.dtype, device=like.device)
            ppm = c[t] if c.dim() == 2 else c
        # VIC-WUR 3.0 canopy_evap.c:302 / surface_fluxes.c:793 hand calc_rc
        # Catm / PPM_to_MIXRATIO
        ppm = _co2.ppm_from_catm(ppm)
        return _co2.co2factor(self.b_co2, ppm.unsqueeze(-1), self.co2ref)

    def active(self):
        """True when any hook would change what the step computes."""
        return (self.ksat is not None or self.wcr is not None
                or bool(self.veg) or self.b_co2 is not None
                or self.water_stress or self.diagnostics
                or self.tile_fluxes or self.throughfall == 'exact'
                or self.step_fn is not None
                or self.state_fn is not None or self.cv is not None
                or self.gwm is not None or self.transp is not None
                or self.wpwp_floor is not None or self.pet_short
                or self.vcs_drain is not None
                or self.canopy_crop is not None
                or self.crop_soil_evap is not None)

    def __repr__(self):
        on = [k for k in ('ksat', 'wcr', 'b_co2') if getattr(self, k)
              is not None]
        on += ['veg:' + k for k in self.veg]
        if self.water_stress:
            on.append('water_stress')
        if self.tile_fluxes:
            on.append('tile_fluxes')
        if self.gwm is not None:
            on.append('gwm:' + self.gwm)
        on += [k for k in ('transp', 'wpwp_floor')
               if getattr(self, k) is not None]
        if self.pet_short:
            on.append('pet_short')
        if self.vcs_drain is not None:
            on.append('vcs_drain')
        if self.canopy_crop is not None:
            on.append('canopy_crop')
        if self.crop_soil_evap is not None:
            on.append('crop_soil_evap')
        return 'VicwurCore(%s)' % ', '.join(on or ['no hooks'])


class TileStates(object):
    """The carried tile states of a run between two steps (state_fn).

        states.names()          what this run carries: 'moist' [col, tile,
                                layer] mm (total, ice included), 'wdew'
                                [col, tile] mm, 'soil_ice', 'last_snow',
                                'melting', 'wb_t0'/'wb_t1' (the water-balance
                                path's two soil temperatures), 't_node',
                                'snow.<key>' (basin's snow store: ice,
                                surf_water, pack_water, surf_temp, pack_temp,
                                depth, density, cold_content, ...),
                                'canopy.<key>' (an overstory's canopy snow)
        states.get(name)        the tensor the next step starts from (do not
                                modify it in place)
        states.set(name, x)     replace it; same shape, cast to its dtype
                                and device

    Columns are the run's (a snow-band run's pseudo cells; band_columns and
    bands_from_columns convert)."""

    def __init__(self, get, set_, names):
        self._get, self._set, self._names = get, set_, names

    def names(self):
        return self._names()

    def get(self, name):
        return self._get(name)

    def set(self, name, value):
        self._set(name, value)


def bands_from_columns(x, snowbands, n_band, fill=0.0):
    """The inverse of band_columns: [col, tile, ...] -> [cell, tile,
    band, ...], `fill` where a band has no area (no column)."""
    if snowbands is None:
        return x.unsqueeze(2)
    c = torch.as_tensor(snowbands['cell'], dtype=torch.long, device=x.device)
    b = torch.as_tensor(snowbands['band'], dtype=torch.long, device=x.device)
    n = int(snowbands['n_cell'])
    out = torch.full((n, x.shape[1], n_band) + tuple(x.shape[2:]), fill,
                     dtype=x.dtype, device=x.device)
    out[c, :, b] = x
    return out


class _VegOverride(object):
    """Writes the step's override into the mosaic and puts the file's
    values back where the mask is off (VIC-WUR 3.0 update_step_vars.c:56-64
    copies veg_hist[NR] every step; a plugin writes its tiles after that).

    `to_lanes` (basin.simulate on the covered tiles): callable(x, tail)
    taking a [col, tile, *tail] value (or one that broadcasts to it) to the
    lanes, [lane, 1, *tail]. The mosaic is then the lanes' and every field
    is written in place into a private copy, so that a CUDA graph reading
    the field's memory reads each step's value."""

    def __init__(self, mosaic, fields, to_lanes=None):
        self.mosaic = mosaic
        self.orig = {}
        self.to_lanes = to_lanes
        for k in fields:
            v = getattr(mosaic, k, None)
            if v is None:
                raise ValueError('vicwur_core veg %r: the case has no %s '
                                 '(the parameter file does not carry it)'
                                 % (k, k))
            self.orig[k] = v
            if k in TABLE_FIELDS or to_lanes is not None:
                # a private table (field), written in place every step
                setattr(mosaic, k, v.clone())

    def tables(self):
        return [k for k in self.orig if k in TABLE_FIELDS]

    def apply(self, core, t, idx):
        m = self.mosaic
        mask = core.veg_mask
        tl = self.to_lanes
        if mask is not None and tl is not None:
            mask = tl(torch.as_tensor(mask), (), key='veg_mask')
        for k, v0 in self.orig.items():
            new = core._at(core.veg[k], t)
            if k in TABLE_FIELDS:
                base = v0[:, :, idx]
                new = torch.as_tensor(new, dtype=base.dtype,
                                      device=base.device)
                if tl is not None:
                    new = tl(new, (), key='veg.' + k)
                new = new.expand_as(base)
                val = new if mask is None else torch.where(mask, new, base)
                getattr(m, k)[:, :, idx] = val
            else:
                new = torch.as_tensor(new, dtype=v0.dtype, device=v0.device)
                if tl is not None:
                    new = tl(new, tuple(v0.shape[2:]), key='veg.' + k)
                new = new.expand_as(v0)
                if mask is None:
                    val = new
                else:
                    mk = mask if v0.dim() == mask.dim() else mask.unsqueeze(-1)
                    val = torch.where(mk, new, v0)
                if tl is not None:
                    getattr(m, k).copy_(val)
                else:
                    setattr(m, k, val)

    def restore(self):
        for k, v0 in self.orig.items():
            setattr(self.mosaic, k, v0)


#: VIC-WUR plugin keys of a global file; an image run of the VIC core
#: alone reads past them (the plugins are TorchVIC components)
_PLUGIN_KEYS = ('ROUTING', 'ROUT_STEPS_PER_DAY', 'ROUTING_PARAMETERS',
                'DECOMPOSITION', 'DAMS', 'DAMS_PARAMETERS', 'WATERUSE',
                'WATERUSE_PARAMETERS', 'LOCAL_WITHDRAWAL',
                'REMOTE_WITHDRAWAL', 'COMPENSATION_WITHDRAWAL',
                'NONRENEWABLE_WITHDRAWAL', 'NONRENEWABLE_RUNOFF',
                'PLUGIN_FORCE_TYPE', 'IRRIGATION', 'IRRIGATION_PARAMETERS',
                'EFFICIENT_IRRIGATION', 'OFFSET_IRRIGATION',
                'POTENTIAL_IRRIGATION', 'PBARE', 'EFR', 'LANDUSE_FORCE',
                'WOFOST', 'WOFOST_STEPS_PER_DAY', 'CROP_PARAMETERS',
                'WOFOST_TEXT_PARAMETERS', 'WOFOST_CALC_MINERALIZATION',
                'WOFOST_DIST_SEASON', 'WOFOST_DIST_TSUM',
                'WOFOST_DIST_FERT', 'WOFOST_CONTINUE', 'BCO2_SRC', 'GWM',
                'CONSTANTS')


def image_global(src, dst):
    """A VIC-WUR global file as a VIC 5 image global file for case.load.

    VIC-WUR 3.0 names a path per forcing variable (`FORCE_TYPE AIR_TEMP tair
    <prefix>`, set_force_type.c) where VIC 5 has `FORCING1 <prefix>` and
    `FORCE_TYPE AIR_TEMP tair`; every variable of the R1 runs is in one
    file set, so the first prefix becomes FORCING1 (a second prefix is
    refused). The plugin keys (`_PLUGIN_KEYS`) are dropped: their
    components are TorchVIC's own, coupled to the VIC core through the
    hooks. NBARE is kept (the vicwur3 core reads it). Returns `dst`."""
    import re
    out, prefix = [], None
    for ln in open(src).read().splitlines():
        w = ln.split()
        if not w or w[0].startswith('#'):
            out.append(ln)
            continue
        m = re.match(r'FORCE_TYPE\s+(\S+)\s+(\S+)\s+(\S+)\s*$', ln)
        if m:
            if prefix is None:
                prefix = m.group(3)
                out.append('FORCING1 %s' % prefix)
            elif m.group(3) != prefix:
                raise ValueError('%s: forcing variables in more than one file '
                                 'set (%s, %s)' % (src, prefix, m.group(3)))
            out.append('FORCE_TYPE %s %s' % (m.group(1), m.group(2)))
            continue
        if w[0].upper() in _PLUGIN_KEYS:
            continue
        out.append(ln)
    with open(dst, 'w') as f:
        f.write('\n'.join(out) + '\n')
    return dst
