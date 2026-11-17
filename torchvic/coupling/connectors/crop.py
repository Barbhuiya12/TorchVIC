# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""CropConnector: WOFOST 7.2 (torchvic.crop) coupled to VIC in 'external'
water mode.

VIC owns the soil water. Each day it supplies WOFOST's transpiration
reduction RFTRA = TRA / TRAMX, actual over potential *crop* transpiration
(wofost72.py:269-311: WOFOST keeps its own TRAMX from the reference
evaporation and the crop's LAI and scales assimilation and leaf death with
RFTRA). Without oxygen stress WOFOST 7.2's RFTRA is its water-stress factor

    RFWS = (W - Wwp) / (Wcr - Wwp)  over the root zone [0, RD], in [0, 1]

and with `vic.soil_moisture` linked that is what the connector forms, from
VIC's layer moisture (mm) and VIC's critical and wilting points (Wcr_FRACT,
Wpwp_FRACT x max_moist, the same points VIC's own transpiration stress uses),
each layer counted with the part of its thickness above the crop's rooting
depth RD (`rftra_soil`). The older ratio vic.transp / vic.pet (`rftra_of`,
`rftra_from: transp_pet`) is NOT that: OUT_PET is the cell's potential
evaporation with an unstressed canopy and a bare-soil part with no surface
resistance (compute_pot_evap.c, fcanopy Epot_veg + (1 - fcanopy) Epot_soil,
summed over all tiles by Cv), while OUT_TRANSP_VEG is only the dry canopy's
transpiration at VIC's library LAI. With wet soil the ratio is the
canopy-cover and dry-fraction share of the potential (0.1-0.3 in a monsoon),
not 1, so it starves the crop; it is kept only for comparison. WOFOST returns, per cell: LAI, rooting depth (m),
development stage, total weight of storage organs TWSO (the yield, kg ha-1),
above-ground biomass, `crop_active` (1 while the crop stands: the irrigation
calendar the WaterUseConnector reads) and, with a `height_table` (AFGEN of
DVS, WOFOST 7.2 itself has no height), the crop height (m).

Weather: from the VIC case's forcing (min/max air temperature, mean
shortwave, rainfall) summed to the day; the reference evaporation E0 = ES0 =
ET0 is VIC's potential evaporation (OUT_PET) of the day, cm.

Feeding LAI back into VIC without touching VIC's equations
-----------------------------------------------------------
VIC reads a per-step LAI only from the vegetation-history input (FORCING2
with LAI_SRC FROM_VEGHIST, torchvic/io/veghist.py; VIC 5.1.0
vic_force.c:264-363). That path is read once, when the case is loaded:
`veghist.apply` builds the mosaic's [cell, tile, step] LAI from the files.
There is no in-run input for LAI in the VicConnector (its inputs are
`precip_add` and `soil_add` only), and writing into the mosaic between
windows would bypass the input path. So the feedback is implemented as the
existing input path, used in two passes:

    pass 1: VIC + crop (this connector)          -> `lai` series
    write_veghist(case, lai, crop_tiles, prefix)    -> <prefix><year>.nc
    pass 2: VIC with FORCING2 <prefix>, LAI_SRC FROM_VEGHIST

VIC then runs with the crop's LAI and its own equations. The two-pass
loop can be repeated to a fixed point (the crop's RFTRA of pass 2 differs
a little); a same-run feedback needs an LAI input on the VicConnector, a
change to `torchvic/coupling/vic.py`, not made here.

Backend 'cropsyst' (VIC-CropSyst port, mode A)
---------------------------------------------
`backend='cropsyst'` runs the CropSyst engine of torchvic.crop.cropsyst on rotation tiles
(one tile per cycle year of each cell's rotation) in place of WOFOST; the default stays
WOFOST and its code path is unchanged. VIC keeps its transpiration and soil water (mode A):
CropSyst takes VIC's layer moisture, computes root uptake and water stress (E1/E2, diagnostic
only) and its own growth, and exports LAI, root depth, stage, biomass, yield and the per-layer
uptake on the grid 'crop_tiles'. The day order is torchvic.crop.cropsyst.engine (E10). The block
`cropsyst={...}` is documented at engine.build_run; `weather` carries the daily arrays, or the
VIC case's forcing is used. Link `vic.soil_moisture` with `lag: 1` (the reference reads the
layer moisture of the start of the day) and `vic.pet` for the reference ET (PET_SHORT).
`nitrogen=True` runs crop and soil nitrogen (torchvic.crop.cropsyst.n_engine): it needs
the soil's bulk_density, wcr, wpwp, quartz and clay in the block and the layer temperature each
day (input `soil_temperature`, VIC layer T; `initial_soil_temperature` covers the first day of a
lagged link); it exports the N stress factor and the soil NO3-N / NH4-N per layer (the NH4 of the
reference build turns NaN where residue N mineralizes; see n_engine). Ledger: like WOFOST, the crop holds no water
of its own in mode A (no balance terms). Restart: the state tensors are saved as they are, so
a restarted run continues bit for bit.

The engine's leaf cohorts are updated in place by Wofost72, so `compute`
swaps references: the accepted engine keeps a clone of the three cohort
arrays, the proposal holds the advanced ones, `commit` swaps them in.
"""
import numpy as np
import torch

from ..connector import ComputeError, Connector, DeviceFlag, Proposal
from ..spec import CouplingError, GridSpec, VarSpec
from ...crop import CropParams, Drivers, Wofost72
from ...crop.cropsyst import engine as cse
from . import common

GRID = 'vic_cells'
TILE_GRID = 'crop_tiles'
DAY = 86400.0
STAGE_NAMES = ('emerging', 'vegetative', 'reproductive', 'mature')


class _Weather(object):
    """Daily WOFOST drivers from the VIC case's forcing."""

    def __init__(self, driver, lat, device, dtype):
        self.d, self.dev, self.dt = driver, device, dtype
        self.spd = int(driver.steps_per_day)
        self.lat = lat
        self.n_days = driver.n_steps // self.spd

    def atm(self, day):
        s0, s1 = day * self.spd, (day + 1) * self.spd
        d = self.d
        if getattr(d, 'case', None) is not None:
            a = d.case.atm
            return {k: a[k][s0:s1] for k in ('tair', 'shortwave', 'prcp')}
        w = d.window(s0, s1)['atm']
        return {k: w[k] for k in ('tair', 'shortwave', 'prcp')}

    def doy(self, day):
        return float(self.d.dates[day * self.spd].timetuple().tm_yday)

    def drivers(self, day, pet_mm):
        a = self.atm(day)
        tair = a['tair'].to(self.dev, self.dt)
        et0 = pet_mm.to(self.dev, self.dt) / 10.0          # cm d-1
        n = tair.shape[1]
        return Drivers(
            tmin=tair.min(0).values, tmax=tair.max(0).values,
            irrad=a['shortwave'].to(self.dev, self.dt).mean(0) * DAY,
            rain=a['prcp'].to(self.dev, self.dt).sum(0) / 10.0,
            e0=et0, es0=et0, et0=et0, lat=self.lat,
            doy=torch.full((n,), self.doy(day), dtype=self.dt,
                           device=self.dev))


class _ArrayWeather(object):
    """Daily drivers given as arrays: {tmin, tmax, irrad (J m-2 d-1), rain
    (cm d-1), et0 (cm d-1; e0, es0 default to it), doy}, each [day, cell] or
    [day, 1] -- the way torchvic.crop.simulate takes them. VIC's potential
    evaporation is not used when 'et0' is given."""

    def __init__(self, arrays, lat, n, device, dtype):
        self.a = {k: torch.as_tensor(v, dtype=dtype, device=device)
                  for k, v in arrays.items()}
        self.lat, self.n = lat, n
        self.n_days = int(self.a['tmin'].shape[0])
        self.has_et0 = 'et0' in self.a

    def drivers(self, day, pet_mm):
        g = lambda k: self.a[k][day].expand(self.n)
        et0 = g('et0') if self.has_et0 else pet_mm / 10.0
        e0 = g('e0') if 'e0' in self.a else et0
        es0 = g('es0') if 'es0' in self.a else et0
        return Drivers(tmin=g('tmin'), tmax=g('tmax'), irrad=g('irrad'),
                       rain=g('rain'), e0=e0, es0=es0, et0=et0,
                       lat=self.lat, doy=g('doy'))


def _snapshot(e):
    return (dict(e.s), e.r, e.t, e.done, e.done_day)


def _restore(e, snap):
    e.s, e.r, e.t, e.done, e.done_day = snap


class CropConnector(Connector):
    name = 'crop'
    version = '1'
    backends = ('cpu', 'cuda')
    graph_safe = False
    checkpoint = 'full'
    validation = ('WOFOST 7.2 against PCSE 6.0.13 (tests/test_crop_pcse.py); '
                  'coupled run equal to the standalone engine with the same '
                  'RFTRA series, bit for bit (tests/test_conn_agri.py)')
    owned_state = ('crop',)
    interval_s = DAY

    def __init__(self, driver=None, vic=None, global_param=None, params=None,
                 start_day=0, height_table=None, lat=None, max_days=None,
                 end_type='maturity', start_type='emergence', weather=None,
                 harvest_idx=None, max_duration=None, n_cells=None,
                 params_file=None, backend='wofost', cropsyst=None,
                 start_doy=None, rftra_from='auto', soil=None, **config):
        super().__init__(**config)
        if backend not in ('wofost', 'cropsyst'):
            raise CouplingError("crop: backend must be 'wofost' or 'cropsyst', "
                                "not %r" % (backend,))
        self.backend = backend
        self.cs_cfg = dict(cropsyst or {})
        if backend == 'cropsyst':
            cse.check_nitrogen(self.cs_cfg.get('nitrogen', False))
        elif cropsyst:
            raise CouplingError('crop: cropsyst= needs backend=\'cropsyst\'')
        self.cs = None
        self.driver = driver
        self._cfg_vic = dict(vic or {})
        if global_param:
            self._cfg_vic['global_param'] = global_param
        self.params_in = params if params is not None else params_file
        self.start_day = int(start_day)
        # start_doy: sow again on this day of the year every year (a new crop
        # replaces the last one; the first sowing is the first such day on or
        # after start_day). None: one crop, sown once on start_day.
        self.start_doy = None if start_doy is None else int(start_doy)
        self.sow_days = None
        # where RFTRA comes from: 'soil' (vic.soil_moisture, WOFOST's RFWS),
        # 'transp_pet' (vic.transp / vic.pet, see the module docstring),
        # 'auto': a direct rftra input, else soil_moisture when linked, else
        # transp / pet
        if rftra_from not in ('auto', 'soil', 'transp_pet'):
            raise CouplingError("crop: rftra_from must be 'auto', 'soil' or "
                                "'transp_pet', not %r" % (rftra_from,))
        self.rftra_from = rftra_from
        # soil: {depth [cell, layer] m, wcr, wpwp [cell, layer] mm}; default
        # the VIC case's (Wcr_FRACT, Wpwp_FRACT x max_moist)
        self.soil_in = soil
        self.height_table = height_table
        self.lat_in = lat
        self.max_days = max_days
        self.start_type, self.end_type = start_type, end_type
        self.weather_in = weather
        self.n_cells = n_cells
        self.harvest_idx, self.max_duration = harvest_idx, max_duration
        self.engine = None
        self.started = False
        self.day = 0
        outs = [
            VarSpec('lai', '1', ('cell',), kind='state', grid=GRID,
                    description='leaf + stem + pod area index'),
            VarSpec('root_depth', 'm', ('cell',), kind='state', grid=GRID,
                    description='rooting depth (WOFOST RD, cm / 100)'),
            VarSpec('dvs', '1', ('cell',), kind='state', grid=GRID,
                    description='development stage'),
            VarSpec('yield', 'kg ha-1', ('cell',), kind='state', grid=GRID,
                    description='weight of storage organs (TWSO)'),
            VarSpec('crop_active', '1', ('cell',), kind='state', grid=GRID,
                    description='1 while the crop stands (irrigation '
                    'season)'),
            VarSpec('rftra', '1', ('cell',), kind='state', grid=GRID,
                    description='transpiration reduction used today')]
        if height_table:
            outs.append(VarSpec('crop_height', 'm', ('cell',), kind='state',
                                grid=GRID, description='AFGEN(DVS) of the '
                                'configured height_table'))
        self.outputs = tuple(outs)
        self.inputs = (
            VarSpec('transp', 'mm', ('cell',), kind='flux', grid=GRID,
                    description='VIC transpiration over the day'),
            VarSpec('pet', 'mm', ('cell',), kind='flux', grid=GRID,
                    description='VIC potential evaporation over the day'),
            VarSpec('soil_moisture', 'mm', ('cell', 'layer'), kind='state',
                    grid=GRID, description='VIC layer moisture: RFTRA = '
                    'WOFOST RFWS over the root zone (rftra_soil)'),
            VarSpec('rftra', '1', ('cell',), kind='state', grid=GRID,
                    description='transpiration reduction supplied directly '
                    '(replaces the soil water and transp/pet)'))
        if backend == 'cropsyst':
            self._declare_cropsyst()

    def grids(self):
        if self.driver is None:
            g = [GridSpec(GRID, int(self.n_cells or 0), kind='cells')]
        else:
            d = self.driver
            area = None
            if d.area is not None:
                area = torch.as_tensor(d.area * (d.frac if d.frac is not None
                                                 else 1.0),
                                       dtype=torch.float64)
            g = [GridSpec(GRID, d.n_cells, area=area, kind='cells')]
        if self.backend == 'cropsyst':
            g.append(GridSpec(TILE_GRID, self._cs_tiles(), kind='tiles',
                              description='CropSyst rotation tiles'))
        return g

    # -- life cycle ---------------------------------------------------------
    def initialize(self, context):
        if self.backend == 'cropsyst':
            return self._cs_initialize(context)
        if self.driver is None and self.weather_in is None:
            self.driver = common.shared_driver(self._cfg_vic, context)
        dev, dt = context.device, torch.float64
        self.device = dev
        d = self.driver
        n = d.n_cells if d is not None else int(self.n_cells or 0)
        self.n = n
        lat = self.lat_in
        if lat is None and d is not None:
            lat = (d.case.dom.lat if getattr(d, 'case', None) is not None
                   else getattr(getattr(d, 'dom', None), 'lat', None))
        if lat is None:
            raise CouplingError('crop: no latitude (give lat=)')
        lat = torch.as_tensor(np.asarray(lat), dtype=dt, device=dev)
        lat = lat.expand(n).clone() if lat.dim() == 0 else lat
        if self.weather_in is not None:
            self.weather = _ArrayWeather(self.weather_in, lat, n, dev, dt)
        else:
            self.weather = _Weather(d, lat, dev, dt)
        p = self.params_in
        if p is None:
            raise CouplingError('crop: give params (a WOFOST parameter '
                                'dict, a CropParams, or a yaml file)')
        if isinstance(p, str):
            p = self._load_params(p)
        if not isinstance(p, CropParams):
            p = CropParams.from_sets([p], n=n, device=dev, dtype=dt,
                                     start_type=self.start_type,
                                     end_type=self.end_type)
        self.cparams = p
        self.soil = None
        sp = self.soil_in
        if sp is None and d is not None and hasattr(d, 'soil'):
            so = d.soil()
            if getattr(so, 'w_crit_mm', None) is not None and \
                    getattr(so, 'depth', None) is not None:
                sp = dict(depth=so.depth, wcr=so.w_crit_mm,
                          wpwp=so.w_wilt_mm)
        if sp is not None:
            self.soil = {k: torch.as_tensor(sp[k], dtype=dt, device=dev)
                         for k in ('depth', 'wcr', 'wpwp')}
            if self.soil['depth'].shape != self.soil['wcr'].shape or \
                    self.soil['depth'].shape[0] != n:
                raise CouplingError('crop: soil depth, wcr, wpwp must be '
                                    '[cell, layer]')
        days = self.weather.n_days
        self.engine = Wofost72(
            p, mode='external', max_days=int(self.max_days or days),
            harvest_idx=self.harvest_idx, max_duration=self.max_duration)
        self.htab = None
        if self.height_table:
            from ...crop.afgen import Afgen
            self.htab = Afgen.from_lists([list(self.height_table)], dev,
                                         dt).expand(torch.zeros(
                                             n, dtype=torch.long, device=dev))
        self.started = False
        self.day = 0
        self.sow_days = None
        if self.start_doy is not None:
            if self.weather_in is not None:
                doy = self.weather.a['doy'].reshape(self.weather.n_days, -1)
                doys = [int(round(float(x))) for x in doy[:, 0].tolist()]
            else:
                spd = int(d.steps_per_day)
                doys = [d.dates[k * spd].timetuple().tm_yday
                        for k in range(days)]
            self.sow_days = set(k for k in range(self.start_day, days)
                                if doys[k] == self.start_doy)
            if not self.sow_days:
                raise CouplingError('crop: start_doy %d never occurs on or '
                                    'after start_day %d in the run'
                                    % (self.start_doy, self.start_day))

    def _new_engine(self):
        return Wofost72(
            self.cparams, mode='external',
            max_days=int(self.max_days or self.weather.n_days),
            harvest_idx=self.harvest_idx, max_duration=self.max_duration)

    @staticmethod
    def _load_params(path):
        import json
        if path.endswith('.json'):
            with open(path) as fh:
                return json.load(fh)
        import yaml
        with open(path) as fh:
            doc = yaml.safe_load(fh)
        if 'CropParameters' in doc:
            raise CouplingError('crop: params file %s is a WOFOST crop '
                                'library; name the variety with '
                                'torchvic.crop.load_wofost_crop_yaml and '
                                'pass the dict' % path)
        return doc

    # -- RFTRA from VIC ---------------------------------------------------------
    @staticmethod
    def rftra_of(transp, pet):
        """actual / potential transpiration, in [0, 1]; 1 where there is
        no demand."""
        pet = torch.as_tensor(pet)
        ok = pet > 1e-12
        r = torch.where(ok, torch.as_tensor(transp) / pet.clamp(min=1e-12),
                        torch.ones_like(pet))
        return r.clamp(0.0, 1.0)

    @staticmethod
    def rftra_soil(moist, depth, wcr, wpwp, rd):
        """WOFOST 7.2's water-stress factor RFWS = TRA / TRAMX (no oxygen
        stress) over the root zone: (W - Wwp) / (Wcr - Wwp) in [0, 1], the
        moisture W and the critical and wilting points Wcr, Wwp [cell,
        layer] mm summed over the layers, each layer weighted by the part of
        its thickness `depth` [cell, layer] (m) above the rooting depth `rd`
        [cell] (m). 1 where the root zone has no range (Wcr = Wwp)."""
        moist, depth = torch.as_tensor(moist), torch.as_tensor(depth)
        top = torch.cumsum(depth, -1) - depth
        rd = torch.as_tensor(rd, dtype=depth.dtype, device=depth.device)
        f = ((rd.unsqueeze(-1) - top) / depth).clamp(0.0, 1.0)
        num = ((moist - wpwp) * f).sum(-1)
        den = ((wcr - wpwp) * f).sum(-1)
        ok = den > 0
        r = torch.where(ok, num / torch.where(ok, den, torch.ones_like(den)),
                        torch.ones_like(den))
        return r.clamp(0.0, 1.0)

    def _view(self, e):
        o = e.output()
        s = e.s
        active = (s['stage'] != 3) & ~e.done
        ex = dict(lai=o['lai'], root_depth=o['rd'] / 100.0, dvs=o['dvs'],
                  yield_=o['twso'], crop_active=active.to(torch.float64),
                  rftra=o['rftra'])
        if self.htab is not None:
            ex['crop_height'] = self.htab(o['dvs'])
        return o, ex

    # -- backend 'cropsyst' (mode A) ----------------------------------------
    def _declare_cropsyst(self):
        T = TILE_GRID
        v = lambda n, u, desc, dims=('cell',): VarSpec(
            n, u, dims, kind='state', grid=T, description=desc)
        self.outputs = (
            v('lai', '1', 'leaf area index (0 without a crop)'),
            v('gai', '1', 'green area index of the day record'),
            v('green_cover', '1', 'green canopy solar interception'),
            v('root_depth', 'm', 'rooting depth of the day record'),
            v('harvest', '1', '1 on a harvest day'),
            v('sown', '1', '1 on a sowing day'),
            v('stage', '1', 'normal growth stage code'),
            v('biomass', 'kg m-2', 'vital canopy biomass'),
            v('yield', 'kg ha-1', 'yield on the harvest day (fruit dry '
              'biomass for orchard crops)'),
            v('crop_active', '1', '1 while a crop stands on the tile'),
            v('water_stress', '1', 'actual / limited potential transpiration'),
            v('act_transp', 'mm', 'uptake diagnosed from the VIC layer '
              'moisture (mm per day)'),
            v('et_max', 'mm', 'crop ET max (mm per day)'),
            v('leaf_water_pot', 'J kg-1', 'leaf water potential'),
            v('uptake', 'mm', 'uptake per VIC layer (mm per day)',
              ('cell', 'layer')),
            v('fruit_biomass', 'kg m-2', 'fruit dry biomass of an orchard '
              'crop (0 for other crops)'),
            v('chill_remaining', '1', 'chill units an orchard crop still '
              'needs (0 without a chill requirement)'),
            v('clip_margin', '1', 'closest keep/drop decision of the day\'s '
              'canopy removals by clipping (-1: none; a diagnostic)'),
            v('orchard_interception', '1', 'solid canopy interception of the '
              'orchard canopy architecture (0 without it)'),
            v('canopy_width', 'm', 'orchard canopy width across the row (0 '
              'without canopy architecture)'),
            v('canopy_breadth', 'm', 'orchard canopy breadth along the row'),
            v('canopy_height', 'm', 'orchard canopy height of the '
              'architecture'))
        if self.cs_cfg.get('nitrogen', False):
            self.outputs = self.outputs + (
                v('n_stress', '1', 'leaf N stress factor (1: none)'),
                v('soil_no3_n', 'kg m-2', 'soil NO3-N per layer', ('cell', 'layer')),
                v('soil_nh4_n', 'kg m-2', 'soil NH4-N per layer (NaN where the '
                  'reference turns it NaN)', ('cell', 'layer')),
                v('n_overflow', '1', '1 if a day had more events than the '
                  'slots or a residue list filled up'))
        i = lambda n, u, kind, desc, dims=('cell',): VarSpec(
            n, u, dims, kind=kind, grid=GRID, description=desc)
        self.inputs = (
            i('soil_moisture', 'mm', 'state', 'VIC layer moisture, liquid + '
              'ice, at the start of the day (link with lag 1)',
              ('cell', 'layer')),
            i('soil_ice', 'mm', 'state', 'VIC layer ice (optional)',
              ('cell', 'layer')),
            i('soil_temperature', 'C', 'state', 'VIC layer temperature at '
              'the start of the day (nitrogen only)', ('cell', 'layer')),
            i('pet', 'mm', 'flux', 'VIC potential evaporation of the day: '
              'the reference ET when the weather arrays hold no et_ref'))

    def _cs_tiles(self):
        if self.cs is not None:
            return self.cs.n_tile
        c = self.cs_cfg
        if 'tiles' in c:
            return len(c['tiles'])
        if 'rotations' in c and 'db_dir' in c:
            from ...crop.cropsyst import calendars as cal
            rs = cal.load_rotations(c['db_dir'], sorted(set(c['rotations'])),
                                    'cpu')
            return int(sum(int(rs.years[rs.index_of(k)])
                           for k in c['rotations']))
        return 0

    def _cs_weather_from_driver(self, d, lat_cell):
        """Daily tmax, tmin (C), shortwave (W m-2), min RH (%) per cell for
        the whole run from the VIC case's forcing, and the year / day of
        year of each day. RHmin = actual vapour pressure of the day over the
        saturation pressure at Tmax (Murray, as the reference)."""
        from ...crop.cropsyst import growth as gr
        w = _Weather(d, lat_cell, self.device, torch.float64)
        spd = w.spd
        keys = ('tair', 'shortwave', 'eact_air')
        tmax, tmin, sw, rh, year, doy = [], [], [], [], [], []
        for day in range(w.n_days):
            s0, s1 = day * spd, (day + 1) * spd
            if getattr(d, 'case', None) is not None:
                a = {k: d.case.atm[k][s0:s1] for k in keys}
            else:
                a = d.window(s0, s1)['atm']
            sl = lambda k, a=a: a[k]
            tair = sl('tair').to(self.device, torch.float64)
            tx = tair.max(0).values
            tmax.append(tx)
            tmin.append(tair.min(0).values)
            sw.append(sl('shortwave').to(self.device, torch.float64).mean(0))
            ea = sl('eact_air').to(self.device, torch.float64).mean(0) / 1000.0
            rh.append((100.0 * ea / gr.saturation_vp_kpa(tx)).clamp(0.0, 100.0))
            dt = d.dates[s0]
            year.append(dt.year)
            doy.append(dt.timetuple().tm_yday)
        st = lambda x: torch.stack(x, 0)
        return (dict(tmax=st(tmax), tmin=st(tmin), srad=st(sw), rhmin=st(rh)),
                year, doy)

    def _cs_initialize(self, context):
        cfg = dict(self.cs_cfg)
        cse.check_nitrogen(cfg.get('nitrogen', False))
        dev = context.device
        self.device = dev
        if self.driver is None and self.weather_in is None and \
                self.n_cells is None:
            self.driver = common.shared_driver(self._cfg_vic, context)
        d = self.driver
        n = d.n_cells if d is not None else int(self.n_cells or 0)
        self.n = n
        lat = self.lat_in
        if lat is None and d is not None:
            lat = (d.case.dom.lat if getattr(d, 'case', None) is not None
                   else getattr(getattr(d, 'dom', None), 'lat', None))
        if lat is None:
            raise CouplingError('crop: no latitude (give lat=)')
        lat = np.asarray(lat, dtype=np.float64).reshape(-1)
        lon = cfg.pop('lon', None)
        if lon is None and d is not None:
            lon = (d.case.dom.lon if getattr(d, 'case', None) is not None
                   else getattr(getattr(d, 'dom', None), 'lon', None))
        if lon is None:
            raise CouplingError('crop: no longitude (give cropsyst lon=)')
        lon = np.asarray(lon, dtype=np.float64).reshape(-1)
        if lat.shape[0] == 1:
            lat = np.repeat(lat, n)
        if lon.shape[0] == 1:
            lon = np.repeat(lon, n)
        wx = dict(self.weather_in or {})
        year, doy = wx.pop('year', None), wx.pop('doy', None)
        if 'tmax' not in wx:
            if d is None:
                raise CouplingError('crop: no weather arrays and no VIC '
                                    'case to take the forcing from')
            wx, year, doy = self._cs_weather_from_driver(d, torch.as_tensor(
                lat, dtype=torch.float64, device=dev))
        if year is None or doy is None:
            raise CouplingError("crop: the weather arrays need 'year' and "
                                "'doy' (one per day)")
        self.cs = cse.build_run(cfg, wx, lat, lon, year, doy, n, dev)
        self.day = 0
        self.started = False

    def _cs_compute(self, inputs, context, dt):
        run = self.cs
        if run is None:
            raise ComputeError('crop: not initialised')
        day = int(round(context.t / DAY))
        if day != self.day:
            raise ComputeError('crop: asked for day %d, the accepted state '
                               'is at day %d' % (day, self.day))
        if day >= run.tables.n_day:
            raise ComputeError('crop: day %d is past the %d days of weather'
                               % (day, run.tables.n_day))
        dev, tc = self.device, run.tile_cell
        tile = lambda x: None if x is None else torch.as_tensor(
            x, dtype=torch.float64, device=dev)[tc]
        moist, ice = tile(inputs.get('soil_moisture')), \
            tile(inputs.get('soil_ice'))
        if moist is None and run.tables.act_frac is None:
            if day > 0:
                raise ComputeError('crop: needs vic.soil_moisture as an '
                                   'input')
            # a lagged link has no value on the first day: the profile starts
            # at `initial_moisture` (a fraction of the maximum moisture)
            moist = run.S.max_moist * float(
                self.cs_cfg.get('initial_moisture', 0.5))
        et = None
        if run.tables.et_ref is None:
            pet = inputs.get('pet')
            if pet is None:
                raise ComputeError('crop: needs vic.pet (or et_ref in the '
                                   'weather arrays)')
            et = tile(pet) / 1000.0
        soil_t = None
        if run.nitrogen and run.tables.moist is None:
            soil_t = tile(inputs.get('soil_temperature'))
            if soil_t is None:
                t0 = self.cs_cfg.get('initial_soil_temperature')
                if day > 0 or t0 is None:
                    raise ComputeError('crop: nitrogen needs vic soil_temperature '
                                       '(layer T) as an input')
                soil_t = torch.full_like(run.S.thick, float(t0))
            if ice is None:
                ice = torch.zeros_like(moist)
        new, o = run.step(run.state, moist, ice, et, soil_t)
        f64 = lambda x: x.to(torch.float64)
        ex = dict(lai=o.lai, gai=o.gai, root_depth=o.root_depth,
                  green_cover=o.green_cover, harvest=f64(o.harvest),
                  sown=f64(o.sown),
                  stage=f64(o.stage), biomass=o.biomass,
                  crop_active=f64(o.present),
                  water_stress=o.water_stress, act_transp=o.act_transp * 1e3,
                  et_max=o.et_max * 1e3, leaf_water_pot=o.lwp,
                  uptake=o.uptake * 1e3)
        ex['yield'] = o.yield_ * 1e4
        zt = torch.zeros_like(o.gai)
        ex['fruit_biomass'] = zt if o.orc is None else o.orc.fruit_bm
        ex['chill_remaining'] = zt if o.orc is None else o.orc.chill_remaining
        ex['clip_margin'] = zt - 1.0 if o.clip is None else torch.where(
            torch.isinf(o.clip.margin), torch.full_like(o.clip.margin, -1.0), o.clip.margin)
        arch = o.orc is not None and o.orc.interception is not None
        ex['orchard_interception'] = o.orc.interception if arch else zt
        ex['canopy_width'] = o.orc.width if arch else zt
        ex['canopy_breadth'] = o.orc.breadth if arch else zt
        ex['canopy_height'] = o.orc.height if arch else zt
        if o.n is not None:
            ex.update(n_stress=o.n.n_stress, soil_no3_n=o.n.no3_n,
                      soil_nh4_n=o.n.nh4_n, n_overflow=f64(o.n.overflow))
        ex = {k: v.clone() for k, v in ex.items()}
        return Proposal(connector=self.name, t0=context.t, t1=context.t + dt,
                        state=new, exports=ex, backend=str(dev),
                        series={k: v.unsqueeze(0) for k, v in ex.items()},
                        info=dict(day=day, started=True),
                        flags=[])

    def _cs_validate(self, proposal, context):
        ex = proposal.exports
        ok = torch.stack([torch.isfinite(ex[k]).all() for k in
                          ('lai', 'biomass', 'root_depth', 'act_transp',
                           'water_stress', 'leaf_water_pot')]).all()
        flags = [DeviceFlag('crop: non-finite CropSyst output', ~ok)]
        if 'n_overflow' in ex:
            flags.append(DeviceFlag('crop: nitrogen event slots or residue '
                                    'lists overflowed', (ex['n_overflow'] > 0).any()))
        return flags

    def compute(self, inputs, context, dt):
        if self.backend == 'cropsyst':
            return self._cs_compute(inputs, context, dt)
        e = self.engine
        if e is None:
            raise ComputeError('crop: not initialised')
        day = int(round(context.t / DAY))
        if day != self.day:
            raise ComputeError('crop: asked for day %d, the accepted state '
                               'is at day %d' % (day, self.day))
        n = self.n
        dev = self.device
        resow = self.sow_days is not None and day in self.sow_days
        if day < self.start_day or (self.sow_days is not None
                                    and not self.started and not resow):
            z = torch.zeros(n, dtype=torch.float64, device=dev)
            ex = dict(lai=z, root_depth=z, dvs=z, crop_active=z,
                      rftra=z + 1.0)
            ex['yield'] = z
            if self.htab is not None:
                ex['crop_height'] = z
            return Proposal(connector=self.name, t0=context.t,
                            t1=context.t + dt, state=None, exports=ex,
                            backend=str(dev), info=dict(day=day, snap=None),
                            series={k: v.unsqueeze(0)
                                    for k, v in ex.items()})
        r_in = inputs.get('rftra')
        if r_in is not None:
            rftra = torch.as_tensor(r_in, dtype=torch.float64, device=dev)
        else:
            sm = inputs.get('soil_moisture')
            how = self.rftra_from
            if how == 'auto':
                how = 'soil' if sm is not None else 'transp_pet'
            if how == 'soil':
                if sm is None or self.soil is None:
                    raise ComputeError('crop: RFTRA from the soil water needs '
                                       'vic.soil_moisture as input and the '
                                       "VIC case's soil (or soil=)")
                # the rooting depth the day starts from (RDI for a new crop)
                rd = (e.s['rd'] if self.started and not resow
                      else self.cparams.RDI) / 100.0
                rftra = self.rftra_soil(
                    sm.to(dev, torch.float64), self.soil['depth'],
                    self.soil['wcr'], self.soil['wpwp'], rd.to(dev))
            else:
                tr, pet = inputs.get('transp'), inputs.get('pet')
                if tr is None or pet is None:
                    raise ComputeError('crop: needs vic.soil_moisture, or '
                                       'vic.transp and vic.pet, or rftra as '
                                       'inputs')
                rftra = self.rftra_of(tr.to(dev), pet.to(dev))
        pet_in = inputs.get('pet')
        pet = torch.zeros(n, dtype=torch.float64, device=dev) \
            if pet_in is None else pet_in.to(dev)
        drv = self.weather.drivers(day, pet)
        pre = None
        new_e = None
        if resow:
            # a new crop: a fresh engine; the accepted one is not touched
            new_e = e = self._new_engine()
        elif self.started:
            pre = _snapshot(e)
            s = dict(pre[0])
            for k in ('lv', 'sla', 'lvage'):
                s[k] = e.s[k].clone()          # the accepted copy
            pre = (s,) + pre[1:]
        try:
            if new_e is not None or not self.started:
                e.start(drv, rftra)
            else:
                e.advance(drv, rftra)
            post = _snapshot(e)
            o, ex = self._view(e)
            ex['yield'] = ex.pop('yield_')
            ex = {k: v.clone() for k, v in ex.items()}
            series = dict(
                lai=o['lai'], dvs=o['dvs'], tagp=o['tagp'], twso=o['twso'],
                rd=o['rd'], rftra=o['rftra'], tra=o['tra'],
                et0=drv.et0, crop_active=ex['crop_active'])
            series = {k: v.clone().unsqueeze(0) for k, v in series.items()}
        finally:
            if pre is not None:
                _restore(e, pre)
        return Proposal(connector=self.name, t0=context.t, t1=context.t + dt,
                        state=post, exports=ex, series=series,
                        backend=str(dev),
                        info=dict(day=day, started=True, new_engine=new_e))

    def validate(self, proposal, context):
        if self.backend == 'cropsyst':
            return self._cs_validate(proposal, context)
        st = proposal.state
        if st is None:
            return []
        s = st[0]
        bad = [DeviceFlag('crop: non-finite %s' % k,
                          ~torch.isfinite(s[k]).all())
               for k in ('lai', 'tagp', 'twso', 'rd')]
        return bad + [DeviceFlag('crop: negative LAI',
                                 (s['lai'] < -1e-9).any())]

    def commit(self, proposal, context):
        if self.backend == 'cropsyst':
            self.cs.state = proposal.state
            self.day = proposal.info['day'] + 1
            return
        if proposal.state is not None:
            if proposal.info.get('new_engine') is not None:
                self.engine = proposal.info['new_engine']
            _restore(self.engine, proposal.state)
            self.started = True
        self.day = proposal.info['day'] + 1

    # -- restart ------------------------------------------------------------
    def save_state(self):
        if self.backend == 'cropsyst':
            return dict(backend='cropsyst', day=self.day, started=self.day > 0,
                        flat=[t.detach().cpu().clone()
                              for t in cse.flatten(self.cs.state)])
        if not self.started:
            return dict(day=self.day, started=False)
        e = self.engine
        cpu = lambda x: x.cpu() if torch.is_tensor(x) else x
        return dict(day=self.day, started=True,
                    s={k: cpu(v) for k, v in e.s.items()},
                    r={k: cpu(v) for k, v in e.r.items()}, t=e.t,
                    done=e.done.cpu(), done_day=e.done_day.cpu())

    def load_state(self, blob):
        if self.backend == 'cropsyst':
            if blob.get('backend') != 'cropsyst':
                raise CouplingError('crop: the checkpoint is not a cropsyst '
                                    'backend state')
            self.day = int(blob['day'])
            self.cs.state = cse.unflatten(
                self.cs.state, [t.to(self.device) for t in blob['flat']])
            return
        self.day = int(blob['day'])
        self.started = bool(blob['started'])
        if not self.started:
            return
        e, dev = self.engine, self.device
        to = lambda x: x.to(dev) if torch.is_tensor(x) else x
        e.s = {k: to(v) for k, v in blob['s'].items()}
        e.r = {k: to(v) for k, v in blob['r'].items()}
        e.t = int(blob['t'])
        e.done = blob['done'].to(dev)
        e.done_day = blob['done_day'].to(dev)


def write_veghist(case, lai_days, crop_tiles, prefix, steps_per_day=None):
    """Write the vegetation-history files VIC reads with `FORCING2 <prefix>`
    and `LAI_SRC FROM_VEGHIST` (io/veghist.py): `<prefix><year>.nc` with
    LAI(time, veg_class, lat, lon), one record per model step.

    `lai_days` [day, cell] is the crop's LAI (m2 m-2, a CropConnector series
    `lai`); `crop_tiles` the mosaic tiles that carry the crop. Every other
    covered tile keeps the parameter file's monthly LAI of that step's month
    (FROM_VEGHIST replaces the LAI of all covered tiles).
    Returns the list of files written."""
    import xarray as xr
    dom, mosaic = case.dom, case.mosaic
    spd = int(steps_per_day or case.model_steps)
    lai_days = np.asarray(lai_days.cpu() if torch.is_tensor(lai_days)
                          else lai_days, dtype=np.float64)
    base = mosaic.lai.cpu().numpy()                     # cell, tile, month
    dates = list(case.dates)
    nt = len(dates)
    if lai_days.shape[0] * spd < nt:
        raise CouplingError('write_veghist: %d days of LAI for %d steps'
                            % (lai_days.shape[0], nt))
    months = np.array([d.month - 1 for d in dates])
    tiles = list(crop_tiles)
    rec = base[:, :, months]                            # cell, tile, step
    day_of = np.arange(nt) // spd
    for t in tiles:
        rec[:, t, :] = lai_days[day_of].T
    out = np.zeros((nt, base.shape[1], dom.ny, dom.nx), dtype=np.float64)
    out[:, :, dom.rows, dom.cols] = np.transpose(rec, (2, 1, 0))
    files = []
    for y in sorted({d.year for d in dates}):
        idx = [i for i, d in enumerate(dates) if d.year == y]
        ds = xr.Dataset(
            {'LAI': ((('time', 'veg_class', dom.y_dim, dom.x_dim)),
                     out[idx])},
            coords={'time': [np.datetime64(dates[i].strftime(
                '%Y-%m-%dT%H:%M:%S')) for i in idx]})
        path = '%s%04d.nc' % (prefix, y)
        ds.to_netcdf(path)
        files.append(path)
    return files
