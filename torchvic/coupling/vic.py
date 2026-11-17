"""VicConnector: the TorchVIC land surface model behind the Connector API.

VIC's physics is not touched. The connector runs `basin.simulate` -- through
`Case.run` for a loaded case, so the case's VIC core, its snow-band
contraction and its error reporting are the ones `torchvic run` uses -- over
one exchange interval at a time, starting each interval from the state the
previous one exported (`return_state` / `initial_state`; tests/
test_state_carry.py shows a run cut this way is the single run bit for bit
on the CPU). The interval is

    window_days: N    N days of model steps (31: the windows the ISC
                      benchmark runs, benchmarks/run_isc_torchvic.py)
    window_days: 0    the whole simulation in one call (Case.run itself)
    exchange: day     one day of model steps (a streamed case on a GPU runs
                      the days of a 62-day block in one simulate call and
                      hands them out one at a time, coupling/vic_session;
                      the same bits as a call a day, `day_session: false`)
    exchange: step    one model step (per-step coupling; eager on a GPU,
                      CUDA graphs need more than one step per call)

A loaded case (case.load, not the forcing stream) holds the run's forcing on
its device; `forcing_on: host` keeps it in pinned host memory instead and
copies each window's to the device when the window runs (same values; the
device then holds one window of forcing whatever the run's length).

Where external water can enter without changing VIC's equations
(docs/CONNECTOR_API.md lists these and what would need a physics change):

  precip_add  [cell] or [step, cell], mm over the interval: added to the
              forcing precipitation before the interval runs. It is then
              VIC's precipitation in every respect: partitioned into rain
              and snow by air temperature, intercepted by the canopy,
              infiltrated or run off. Refused with a precipitation
              correction (pcf != 1, CORRPREC TRUE), which would scale it.
  soil_add    [cell, layer] or [cell, tile, layer], mm: added to (or, when
              negative, taken from) the soil moisture of the state the
              interval starts from, at the interval boundary. Validated
              against the layer's capacity and residual moisture; the
              report totals VIC's first DEL terms are formed against are
              moved by the same amount, so VIC's own water error stays
              VIC's. Works on the padded [cell, tile] state and on the
              GPU's lane-compacted one (covered tiles only). Refused with
              FROZEN_SOIL (ice shares the layer) or snow bands (the state
 is per band).
  soil_add_<src>  (soil_add_sources: [src, ...]) one more soil_add input per
              named producer, e.g. irrigation and floodplain infiltration
              on the same cells. The interval's addition is the sum of
              `soil_add` and every `soil_add_<src>` (one injection, one
              capacity check: the waters arrive together at the interval
              start). Signed as soil_add (a negative part, e.g. a rounding
              -1e-18 of irrigation's leftover, takes water out under the
              residual check). What does not fit (cap mode) is split back
              to the producers in proportion to what each one added to the
              cell (positive parts only; no priority): `soil_add_unmet_<src>`, and
              `soil_add_unmet` for the plain `soil_add`. The parts add up
              to the cell's unmet (the largest share takes the rounding
              remainder), and no producer gets back more than it added, so
              no water is handed back twice (a flux has one consumer).
              Not with day_coupled.
  lai         [cell, tile] (or [cell, tile, step]): the leaf area index of
              the next interval in place of the vegetation history's
              (LAI_SRC FROM_VEGHIST only: VIC already reads LAI per step
              there), with VIC's leafless rule (vic_force.c 5.1.0:419-439).

Exports are only quantities VIC computes, summed over the interval (fluxes,
mm) or at its end (states).
"""
import os

import numpy as np
import torch

from .connector import (Balance, ComputeError, Connector, DeviceFlag,
                        Proposal)
from .spec import CouplingError, GridSpec, VarSpec
from ..model import state as model_state

GRID = 'vic_cells'

#: simulate outputs that are the run's end state, not a series over steps
FINAL_KEYS = frozenset(('soil_moisture', 'soil_moisture_tile',
                        'canopy_storage', 'state', 'state_aux'))

#: (export, kind, units, dims, description, how to form it from `out`)
_FLUX = [
    ('runoff', 'surface runoff (OUT_RUNOFF)', ('runoff',)),
    ('baseflow', 'baseflow: drainage out of the bottom soil layer '
     '(OUT_BASEFLOW); the recharge candidate for a groundwater model',
     ('baseflow',)),
    ('total_runoff', 'runoff + baseflow, what a routing model takes',
     ('runoff', 'baseflow')),
    ('et', 'total evaporation incl. sublimation (OUT_EVAP)', ('et',)),
    ('transp', 'transpiration (OUT_TRANSP_VEG)', ('transp',)),
    ('evap_canop', 'canopy evaporation (OUT_EVAP_CANOP)', ('evap_canop',)),
    ('evap_bare', 'bare soil evaporation (OUT_EVAP_BARE)', ('evap_bare',)),
    ('sub_snow', 'snow sublimation (OUT_SUB_SNOW)', ('sub_snow',)),
    ('sub_canop', 'canopy snow sublimation (OUT_SUB_CANOP)', ('sub_canop',)),
    ('rainf', 'rainfall (OUT_RAINF)', ('rainf',)),
    ('snowf', 'snowfall (OUT_SNOWF)', ('snowf',)),
    ('snow_melt', 'snow melt (OUT_SNOW_MELT)', ('snow_melt',)),
    ('pet', 'potential evapotranspiration (OUT_PET)', ('pet',)),
]
_STATE = [
    ('swe', 'mm', ('cell',), 'snow water equivalent at the interval end '
     '(OUT_SWE)'),
    ('soil_moisture', 'mm', ('cell', 'layer'), 'layer moisture, liquid + '
     'ice, at the interval end (OUT_SOIL_MOIST)'),
    ('wdew', 'mm', ('cell',), 'canopy liquid water at the interval end '
     '(OUT_WDEW)'),
    ('snow_canopy', 'mm', ('cell',), 'canopy snow at the interval end '
     '(OUT_SNOW_CANOPY)'),
]

OUTPUTS = tuple(
    [VarSpec(n, 'mm', ('cell',), kind='flux', grid=GRID, description=d)
     for n, d, _ in _FLUX]
    + [VarSpec(n, u, dims, kind='state', grid=GRID, description=d)
       for n, u, dims, d in _STATE])
OUTPUTS = OUTPUTS + (
    VarSpec('total_runoff_daily', 'mm', ('day', 'cell'), kind='flux',
            grid=GRID, description='runoff + baseflow of each model day of '
            'the interval, [day, cell]; the exchange buffer hands a consumer '
            'the days its interval covers (a router gets VIC\'s own daily '
            'series, not the window total spread evenly)'),
    VarSpec('soil_add_unmet', 'mm', ('cell',), kind='flux', grid=GRID,
            description='the part of soil_add that did not fit under the '
                        'layer capacity (soil_add_mode: cap), handed back'),)
INPUTS_LAI = VarSpec(
    'lai', '1', ('cell', 'tile'), kind='state', grid=GRID,
    description='leaf area index per tile for the next interval, in place '
                'of the vegetation history\'s (LAI_SRC FROM_VEGHIST only), '
                'e.g. from a crop model')
INPUTS = (
    VarSpec('precip_add', 'mm', ('cell',), kind='flux', grid=GRID,
            description='water added to the forcing precipitation over '
                        'the interval (e.g. sprinkler irrigation)'),
    VarSpec('soil_add', 'mm', ('cell', 'layer'), kind='flux', grid=GRID,
            description='water added to (negative: taken from) the soil '
                        'layers at the interval start'),
)


# ---------------------------------------------------------------------------
# Drivers: where a window's simulate arguments come from
# ---------------------------------------------------------------------------
class CaseDriver(object):
    """Windows of a loaded torchvic Case (case.load). A window covering the
    whole case is Case.run itself."""

    def __init__(self, case, p=None, forcing_on='device'):
        self.case = case
        self.p = p
        self.dates = case.dates
        self.n_steps = case.n_steps
        #: 'host': the run's forcing is held in pinned host memory and each
        #: window's is copied to the device when it runs (the device holds
        #: one window, not the run); the values are the same
        self.host_forcing = (forcing_on == 'host'
                             and torch.device(case.device).type == 'cuda')
        if self.host_forcing:
            nt = case.n_steps
            pin = lambda d: {k: (v.cpu().pin_memory()
                                 if torch.is_tensor(v) and v.dim() >= 1
                                 and v.shape[0] == nt and v.is_cuda else v)
                             for k, v in d.items()}
            case.atm = pin(case.atm)
            if getattr(case, '_cell_echo', None):
                case._cell_echo = pin(case._cell_echo)
            if isinstance(getattr(case, 'subdaily', None), dict):
                case.subdaily = pin(case.subdaily)
        self.steps_per_day = int(case.model_steps)
        dom = case.dom
        n = case.n_cells
        self.area = np.asarray(dom.area, dtype=np.float64)[:n] \
            if getattr(dom, 'area', None) is not None else None
        self.frac = np.asarray(dom.frac, dtype=np.float64)[:n] \
            if getattr(dom, 'frac', None) is not None else None
        self.banded = case.snowbands is not None
        self.lakes = case.options.get('lakes') is not None
        self.device, self.dtype = case.device, case.dtype

    @property
    def n_cells(self):
        return self.case.n_cells

    def soil(self):
        return self.case.soil

    def mosaic(self):
        return self.case.mosaic

    def precip_scaled(self):
        """True when the case scales its precipitation (pcf != 1 or
        CORRPREC), which would scale injected water too."""
        p = self.p if self.p is not None else self.case.params()
        pcf = p.get('pcf') if isinstance(p, dict) else None
        scaled = pcf is not None and bool(torch.any(torch.as_tensor(pcf)
                                                    != 1.0))
        return scaled or bool(self.case.options.get('corrprec'))

    def run(self, t0, t1, state, precip_add=None, lai=None, day_hook=None,
            vicwur_core=None):
        nt = self.n_steps
        extra = dict(return_state=True)
        if state is not None:
            extra['initial_state'] = state
        if day_hook is not None:
            extra['day_hook'] = day_hook
        if vicwur_core is not None:
            extra['vicwur_core'] = vicwur_core
        whole = (t0, t1) == (0, nt)
        if not whole or precip_add is not None or lai is not None or \
                day_hook is not None or self.host_forcing:
            base = dict(batch=self.case.batch, atm=self.case.atm,
                        subdaily=self.case.subdaily, mosaic=self.case.mosaic)
            w = base if whole else model_state.time_window(base, t0, t1)
            extra['batch'] = w['batch']
            atm = w['atm']
            moved = False
            if self.host_forcing:
                dev = torch.device(self.case.device)
                to_dev = lambda d: {k: (v.to(dev, non_blocking=True)
                                        if torch.is_tensor(v)
                                        and v.device.type == 'cpu' else v)
                                    for k, v in d.items()}
                atm = to_dev(atm)
                if isinstance(w.get('subdaily'), dict):
                    w = dict(w, subdaily=to_dev(w['subdaily']))
                moved = True
            if precip_add is not None:
                atm = dict(atm)
                atm['prcp'] = atm['prcp'] + precip_add.to(atm['prcp'].dtype)
            elif day_hook is not None and not moved:
                # the hook adds to the window's own precipitation buffer
                atm = dict(atm)
                atm['prcp'] = atm['prcp'].clone()
            extra['atm'] = atm
            if w.get('subdaily') is not None:
                extra['subdaily'] = w['subdaily']
            if w.get('mosaic') is not None:
                extra['mosaic'] = (w['mosaic'] if lai is None else
                                   with_lai(w['mosaic'], lai, t1 - t0))
        # a banded case's columns are contracted by the connector, for a
        # run that is one window as for any other (contract_bands only
        # recognises a whole run's histories: a double contraction)
        return self.case.run(p=self.p, contract=False, **extra)


def with_lai(mosaic, lai, n):
    """`mosaic` with its per-step LAI (a vegetation history, LAI_SRC
    FROM_VEGHIST) replaced by `lai` ([cell, tile] for every step of the
    interval, or [cell, tile, step]) on the covered tiles, and VIC's
    leafless rule applied to it as vic_force.c 5.1.0:419-439 applies it to
    a history record (LAI 0 or fcanopy below MIN_FCANOPY: both 0)."""
    import copy
    from ..io import build as _build
    if not getattr(mosaic, 'step_axis', False):
        raise ComputeError(
            'vic: an lai input needs the vegetation history path (LAI_SRC '
            'FROM_VEGHIST with FORCING2), where VIC reads LAI per step; '
            'this case takes LAI from the parameter file\'s months')
    m = copy.copy(mosaic)
    old = mosaic.lai
    new = torch.as_tensor(lai, dtype=old.dtype, device=old.device)
    if new.dim() == 2:
        new = new.unsqueeze(-1).expand(-1, -1, old.shape[2])
    if tuple(new.shape) != tuple(old.shape):
        raise ComputeError('vic: lai is %s, the interval needs %s'
                           % (tuple(new.shape), tuple(old.shape)))
    if bool((new < 0).any()) or not bool(torch.isfinite(new).all()):
        raise ComputeError('vic: lai must be finite and >= 0')
    covered = (mosaic.cv > 0.0).unsqueeze(-1)
    lai_ = torch.where(covered, new, old)
    fc = mosaic.fcanopy
    off = (fc < _build.MIN_FCANOPY) | (lai_ == 0.0)
    m.lai = torch.where(off, torch.zeros_like(lai_), lai_)
    m.fcanopy = torch.where(off, torch.zeros_like(fc), fc)
    del n
    return m


class KwargsDriver(object):
    """Windows supplied by the caller: `static` is simulate's keyword
    arguments that do not change between windows (p, soil_params, mosaic,
    wind_h, init_moist, steps_per_day, full_energy and the physics options)
    and `window(t0, t1)` returns {'batch', 'atm'[, 'subdaily']} for model
    steps [t0, t1). This is how a pipeline with its own forcing reader (the
    ISC benchmark's prefetched windows) drives the connector."""

    def __init__(self, static, window, dates, steps_per_day, area=None,
                 frac=None, n_cells=None):
        self.static = dict(static)
        self.window = window
        self.dates = list(dates)
        self.n_steps = len(self.dates)
        self.steps_per_day = int(steps_per_day)
        self.area = None if area is None else np.asarray(area, np.float64)
        self.frac = None if frac is None else np.asarray(frac, np.float64)
        self.banded = False
        self.lakes = self.static.get('lakes') is not None
        self._n = n_cells
        sp = self.static.get('soil_params')
        im = self.static.get('init_moist')
        self.device = (im.device if torch.is_tensor(im)
                       else getattr(sp, 'device', None))
        self.dtype = (im.dtype if torch.is_tensor(im)
                      else getattr(sp, 'dtype', torch.float64))
        self._cache = None

    @property
    def n_cells(self):
        if self._n is not None:
            return int(self._n)
        return int(self.area.shape[0]) if self.area is not None else 0

    def soil(self):
        return self.static.get('soil_params')

    def mosaic(self):
        return self.static.get('mosaic')

    def precip_scaled(self):
        p = self.static.get('p') or {}
        pcf = p.get('pcf') if isinstance(p, dict) else None
        scaled = pcf is not None and bool(torch.any(torch.as_tensor(pcf)
                                                    != 1.0))
        return scaled or bool(self.static.get('corrprec'))

    def run(self, t0, t1, state, precip_add=None, lai=None, day_hook=None,
            vicwur_core=None, **sim):
        """`sim`: more of basin.simulate's keywords (day_outputs,
        tail_from: a day session, coupling/vic_session.py). A precip_add
        with fewer steps than the window is added to its first steps."""
        from ..model import basin
        if lai is not None:
            raise ComputeError('vic: an lai input is supported on a loaded '
                               'case (CaseDriver) only')
        if self._cache is None or self._cache[0] != (t0, t1):
            # one call per window; a retried window reuses its forcing
            self._cache = ((t0, t1), self.window(t0, t1))
        w = dict(self._cache[1])
        if precip_add is not None:
            w['atm'] = dict(w['atm'])
            k = int(precip_add.shape[0])
            if k < int(w['atm']['prcp'].shape[0]):
                p_ = w['atm']['prcp'].clone()
                p_[:k] = p_[:k] + precip_add
                w['atm']['prcp'] = p_
            else:
                w['atm']['prcp'] = w['atm']['prcp'] + precip_add
        elif day_hook is not None:
            w['atm'] = dict(w['atm'])
            w['atm']['prcp'] = w['atm']['prcp'].clone()
        kw = dict(day_hook=day_hook) if day_hook is not None else {}
        if vicwur_core is not None:
            kw['vicwur_core'] = vicwur_core
        kw.update(sim)
        return basin.simulate(**self.static, **w, initial_state=state,
                              return_state=True, **kw)


# ---------------------------------------------------------------------------
# The connector
# ---------------------------------------------------------------------------
class VicConnector(Connector):
    name = 'vic'
    outputs = OUTPUTS
    owned_state = ('moist', 'snow', 'wdew', 'canopy', 't_node', 'soil_ice',
                   'carbon', 'routing_prefix', 'lake')
    imported_fluxes = ('precip_add', 'soil_add')
    exported_fluxes = tuple(n for n, _, _ in _FLUX) + (
        'total_runoff_daily', 'soil_add_unmet')
    #: VIC captures its own CUDA graphs inside a window (basin.GRAPHS); the
    #: connector's compute is one simulate call and is not itself captured
    graph_safe = False
    backends = ('cpu', 'cuda')
    checkpoint = 'full'
    validation = ('VIC 5.0.1 / 5.1.0 physics as verified for torchvic run '
                  '(docs/verification.md); the connector adds no physics')

    inputs = INPUTS + (INPUTS_LAI,)

    def __init__(self, driver=None, window_days=31, exchange=None,
                 ledger=True, exports=(), soil_add_mode='refuse',
                 day_coupled=(), soil_add_sources=(), soil_add_bands=False,
                 soil_add_frozen=False, soil_add_floor='resid',
                 phase_group=None, gwrecharge_view=False, **config):
        super().__init__(**config)
        from .. import __version__
        self.version = __version__
        self.driver = driver
        self.window_days = int(window_days or 0)
        self.exchange = exchange or ('window' if self.window_days
                                     else 'whole')
        if self.exchange not in ('window', 'whole', 'day', 'step'):
            raise CouplingError("vic: exchange must be window, whole, day or "
                                "step, not %r" % (self.exchange,))
        self.ledger_on = bool(ledger)
        if soil_add_mode not in ('refuse', 'cap', 'over'):
            raise CouplingError("vic: soil_add_mode must be refuse, cap or "
                                "over")
        self.soil_add_mode = soil_add_mode
        # VIC-WUR's soil edits (docs/connectors/vicwur.md); all off by
        # default, and with them off every soil_add path below is the one
        # before they existed.
        #   soil_add_bands   a 4-d [cell, tile, band, layer] soil_add and
        #                    the snow-band case (the state is per band)
        #   soil_add_frozen  FROZEN_SOIL: the addition is to the total
        #                    moisture, ice untouched (irr_use.c:265)
        #   soil_add_floor   where a negative addition stops: 'resid' the
        #                    residual moisture (refused below it, the
        #                    default), 'zero' VIC-WUR's pumping clip
        #                    (wu_run_local.c:417-419), 'liquid' the layer's
        #                    ice (only liquid is taken)
        if soil_add_floor not in ('resid', 'zero', 'liquid'):
            raise CouplingError("vic: soil_add_floor must be resid, zero or "
                                "liquid")
        self.soil_add_bands = bool(soil_add_bands)
        self.soil_add_frozen = bool(soil_add_frozen)
        self.soil_add_floor = soil_add_floor
        #   phase_group      VIC-WUR's plugins (a vicwur.order.
        #                    PhaseScheduler) run between the window's days
        #                    in C order (docs/connectors/vicwur.md);
        #                    their soil edits are soil_adds, so it needs a
        #                    pumping floor ('zero' or 'liquid')
        if phase_group is not None and soil_add_floor == 'resid':
            raise CouplingError("vic: phase_group needs soil_add_floor zero "
                                "or liquid (VIC-WUR's pumping clip)")
        self.phase_group = phase_group
        #   gwrecharge_view  the day's OUT_GWRECHARGE (vicwur_core gwm) on
        #                    the phase group's view, as runoff/baseflow
        #                    (the VIC-WUR 3.0 MODFLOW recharge; off: unchanged)
        self.gwrecharge_view = bool(gwrecharge_view)
        #: modules the runtime runs between this connector's days, inside
        #: its window (DayCoupler, basin.simulate's day_hook)
        self.day_coupled = tuple(day_coupled or ())
        self.coupler = None
        self._init_soil_add_sources(soil_add_sources)
        self.wanted = set(exports)
        self.state = None
        self.t_step = 0
        #: VIC-WUR: the last day's runoff/baseflow (and stored water
        #: stress) of the window committed last, for the plugins that run
        #: at the next window's start (docs/connectors/vicwur.md)
        self._wur_last = None
        self.storage_end = None        # [cell] mm, the last committed end
        #: the day exchange as one simulate call a block of days
        #: (vic_session; `day_session: false` or TORCHVIC_DAY_SESSION=0:
        #: one call a day). Bit-identical either way.
        self._session = None
        self._sess_off = not (
            bool(self.config.get('day_session', True))
            and os.environ.get('TORCHVIC_DAY_SESSION', '1') != '0')
        self.session_days = max(int(self.config.get('session_days', 62)), 1)
        #: set by the runtime: VIC is the run's only module (nothing feeds
        #: it, no other module's code runs beside it), so a session may run
        #: ahead of the days handed out (vic_session)
        self.alone = False
        self.last_backend = ''
        self.windows = 0

    def _init_soil_add_sources(self, sources):
        src = [str(x) for x in (sources or ())]
        if len(set(src)) != len(src) or not all(
                x.isidentifier() for x in src):
            raise CouplingError('vic: soil_add_sources must be distinct '
                                'names, not %r' % (sources,))
        if src and self.day_coupled:
            raise CouplingError('vic: soil_add_sources with day_coupled is '
                                'not supported')
        self.soil_add_sources = tuple(src)
        if not src:
            return
        self.inputs = self.inputs + tuple(
            VarSpec('soil_add_' + x, 'mm', ('cell', 'layer'), kind='flux',
                    grid=GRID, description='soil_add from %s (added to '
                    'soil_add, see soil_add_sources)' % x) for x in src)
        self.outputs = self.outputs + tuple(
            VarSpec('soil_add_unmet_' + x, 'mm', ('cell',), kind='flux',
                    grid=GRID, description='the part of soil_add_%s that '
                    'did not fit (its share of the cell\'s unmet)' % x)
            for x in src)
        self.imported_fluxes = tuple(self.imported_fluxes) + tuple(
            'soil_add_' + x for x in src)
        self.exported_fluxes = tuple(self.exported_fluxes) + tuple(
            'soil_add_unmet_' + x for x in src)

    def _gather_soil_add(self, inputs):
        """(the interval's soil_add, [(unmet export, cell total added)])
        -- the parts are None without soil_add_sources."""
        sa = inputs.get('soil_add')
        if not self.soil_add_sources:
            return sa, None
        d = self.driver
        parts, tot = [], None
        for nm, un in [('soil_add', 'soil_add_unmet')] + [
                ('soil_add_' + x, 'soil_add_unmet_' + x)
                for x in self.soil_add_sources]:
            v = inputs.get(nm)
            if v is None:
                continue
            # signed, as the plain soil_add: a (rounding-size) negative
            # part takes water out, under the same residual-moisture check
            v = torch.as_tensor(v, dtype=d.dtype, device=d.device)
            ct = v if v.dim() == 1 else (
                v.sum(-1) if v.dim() == 2 else
                self._cell_layers(v, None).sum(-1))
            parts.append((un, ct))
            if tot is None:
                tot = v
            else:
                if tot.dim() != v.dim():
                    a, b = (tot, v) if tot.dim() < v.dim() else (v, tot)
                    if a.dim() == 1:
                        a = torch.cat([a.unsqueeze(-1), a.new_zeros(
                            a.shape[0], b.shape[-1] - 1)], -1)
                    if b.dim() == 3:
                        a = a.unsqueeze(1).expand_as(b)
                    tot, v = a, b
                tot = tot + v
        return tot, parts

    def _split_unmet(self, unmet, parts):
        """{export: [cell] mm}: the cell's unmet in proportion to what
        each producer added; the largest share takes the remainder so the
        parts add up to the unmet."""
        n = self.driver.n_cells
        if not parts:
            return {}
        if unmet is None:
            z = torch.zeros(n, dtype=self.driver.dtype,
                            device=self.driver.device)
            return {un: z for un, _ in parts}
        # shares of what each producer added: a negative part (a removal)
        # gets no unmet back, so no share exceeds its producer's addition
        A = torch.stack([ct for _, ct in parts]).clamp(min=0)
        tot = A.sum(0)
        sh = torch.where(tot > 0, A / torch.where(tot > 0, tot,
                                                   torch.ones_like(tot)),
                         torch.zeros_like(A))
        U = sh * unmet.unsqueeze(0)
        k = torch.argmax(sh, 0)
        ar = torch.arange(n, device=U.device)
        U[k, ar] = 0.0
        U[k, ar] = unmet - U.sum(0)
        return {un: U[i] for i, (un, _) in enumerate(parts)}

    # -- declarations -------------------------------------------------------
    def _need_driver(self):
        if self.driver is None:
            raise CouplingError('vic: no case loaded (initialize first)')
        return self.driver

    def grids(self):
        if self.driver is None:
            return [GridSpec(GRID, 0, kind='cells')]
        d = self.driver
        area = None
        if d.area is not None:
            area = d.area * (d.frac if d.frac is not None else 1.0)
            area = torch.as_tensor(area, dtype=torch.float64)
        return [GridSpec(GRID, d.n_cells, area=area, kind='cells',
                         description='VIC active cells; area = domain '
                                     'area x frac (m2)')]

    @property
    def model_step_s(self):
        return 86400.0 / self._need_driver().steps_per_day

    def time_span(self):
        d = self._need_driver()
        return (0.0, d.n_steps * self.model_step_s)

    def window_steps(self):
        d = self._need_driver()
        if self.exchange == 'whole':
            return d.n_steps
        if self.exchange == 'step':
            return 1
        if self.exchange == 'day':
            return d.steps_per_day
        return self.window_days * d.steps_per_day

    # -- life cycle ---------------------------------------------------------
    def initialize(self, context):
        if isinstance(self.phase_group, str):
            # the VIC-WUR connectors' group, by name (they are constructed
            # before any module initializes)
            from .connectors.vicwur_common import get_group
            self.phase_group = get_group(self.phase_group)
        if self.driver is None:
            self.driver = load_driver(self.config, context)
        attach = getattr(self.phase_group, 'attach_vic', None)
        if attach is not None:
            # the VIC-WUR group reads the case's layout (bands, tiles, soil)
            attach(self)
        self.interval_s = self.window_steps() * self.model_step_s
        if hasattr(self.driver, 'aggregate'):
            # a streamed case: the history is aggregated window by window
            # at commit (vic_stream); the per-step series is not kept
            self.keep_outputs = 'none'
        if self.ledger_on and self.driver.lakes:
            context.say('vic: LAKES TRUE -- the lake store is not in the '
                        'ledger yet; the VIC balance check is off for this '
                        'run')
            self.ledger_on = False
        if self.ledger_on and self.driver.area is None:
            context.say('vic: the domain has no cell areas; the ledger checks '
                        'the per-cell balance only')

    def _steps(self, context, dt):
        h = self.model_step_s
        s0 = context.t / h
        s1 = (context.t + dt) / h
        if abs(s0 - round(s0)) > 1e-9 or abs(s1 - round(s1)) > 1e-9:
            raise ComputeError('vic: interval [%g, %g) s is not on the '
                               'model step (%g s)' % (context.t,
                                                      context.t + dt, h))
        s0, s1 = int(round(s0)), min(int(round(s1)), self.driver.n_steps)
        if s0 != self.t_step:
            raise ComputeError('vic: asked for steps from %d, the accepted '
                               'state is at step %d' % (s0, self.t_step))
        return s0, s1

    def compute(self, inputs, context, dt):
        self._sa_flags = []
        if getattr(self, 'coupler', None) is not None:
            self._end_session()
            return self._compute_day_coupled(inputs, context, dt)
        if self._session_on(inputs) and not context.redo:
            # (a redone step takes the one-day call, which checks the
            # graph path's assumptions itself)
            prop = self._compute_session(inputs, context, dt)
            if prop is not None:
                return prop
        self._end_session()
        d = self._need_driver()
        s0, s1 = self._steps(context, dt)
        n = s1 - s0
        state = self.state
        inj_s = None
        inj_p = self._precip_in(inputs, n)
        unmet = None
        sa, parts = self._gather_soil_add(inputs)
        if sa is not None and state is None and not bool(
                torch.as_tensor(sa).any()):
            sa = None           # nothing to add before the first interval
        if sa is not None:
            state, inj_s, unmet = self._soil_add(state, sa)
        lai = inputs.get('lai')
        try:
            out = d.run(s0, s1, state, precip_add=inj_p, lai=lai)
        except ComputeError:
            raise
        except Exception as e:                           # noqa: BLE001
            raise ComputeError('vic: steps %d..%d failed: %s: %s'
                               % (s0, s1, type(e).__name__, e)) from e
        return self._proposal(out, context, dt, (s0, s1), inj_p, inj_s,
                              unmet, n, parts)

    def _precip_in(self, inputs, n):
        """The interval's precip_add as [step, cell] mm (None without
        one), checked."""
        if inputs.get('precip_add') is None:
            return None
        d = self.driver
        if d.precip_scaled():
            raise ComputeError(
                'vic: precip_add with a precipitation correction (pcf '
                '!= 1 or CORRPREC TRUE) would be scaled by it; refused')
        a = torch.as_tensor(inputs['precip_add'], dtype=d.dtype,
                            device=d.device)
        if a.dim() == 1:
            # an interval total, spread evenly over its steps
            a = (a / n).expand(n, -1)
        elif a.shape[0] != n:
            raise ComputeError('vic: precip_add has %d steps, the '
                               'interval %d' % (a.shape[0], n))
        self._refuse_if((a < 0).any(),
                        'vic: precip_add must not be negative')
        return a

    # -- the day exchange as one call a block (vic_session) -----------------
    def _session_on(self, inputs):
        """A day session runs this interval: the day exchange of a
        streamed case on a GPU (the path whose per-call set-up and graph
        capture it saves), nothing the hook path does not take."""
        d = self.driver
        if self._sess_off or self.exchange != 'day' or d is None \
                or self.phase_group is not None \
                or not hasattr(d, 'aggregate') or d.banded or d.lakes \
                or inputs.get('lai') is not None:
            return False
        dev = getattr(d, 'device', None)
        return dev is not None and torch.device(dev).type == 'cuda'

    def _end_session(self):
        S = getattr(self, '_session', None)
        self._session = None
        if S is not None:
            S.close()

    def _compute_session(self, inputs, context, dt):
        """compute() through a DaySession; None when the plain path must
        run this interval (sessions are then off for the run)."""
        from .vic_session import DaySession
        d = self.driver
        s0, s1 = self._steps(context, dt)
        n = s1 - s0
        spd = d.steps_per_day
        if n != spd:
            return None
        state = self.state
        inj_p = self._precip_in(inputs, n)
        sa, parts = self._gather_soil_add(inputs)
        if sa is not None and state is None and not bool(
                torch.as_tensor(sa).any()):
            sa = None           # nothing to add before the first interval
        S = self._session
        if S is not None and not (S.open and S.next_step == s0
                                  and S.last_state is state
                                  and self.t_step == s0):
            # not the session's next day from the state it handed out
            self._end_session()
            S = None
        inj_s = unmet = None
        try:
            if S is None:
                if sa is not None:
                    state, inj_s, unmet = self._soil_add(state, sa)
                b1 = min(s0 + self.session_days * spd, d.n_steps)
                S = self._session = DaySession(
                    d, s0, b1, state, precip_add=inj_p,
                    ahead=2 if self.alone else 0, defer_day_check=True)
                out = S.first()
            else:
                view, ret = S.view, None
                if view is None and (inj_p is not None or sa is not None):
                    raise CouplingError('vic: water for a day of a session '
                                        'that runs ahead')
                if inj_p is not None:
                    t_ = s0 - S.s0
                    view.prcp[t_:t_ + spd] += inj_p
                if sa is not None:
                    # _soil_add on the state the day starts from: the
                    # loop's own moisture (the state handed out holds a
                    # copy of it)
                    self._soil_add_ok(state)
                    lanes = view.lanes
                    new, inj_s, unmet = self._add_soil_cols(
                        view.moist, sa, lanes,
                        ice=view.soil_ice if self.soil_add_floor == 'liquid'
                        else None)
                    ret = (new, inj_s)
                out = S.next(ret)
        except ComputeError:
            self._end_session()
            raise
        except Exception as e:                           # noqa: BLE001
            # the session path failed where the one-day call may not:
            # this run goes on without sessions (nothing was committed)
            self._end_session()
            self._sess_off = True
            context.say('vic: day sessions off for this run (%s: %s); '
                        'one call a day' % (type(e).__name__, e))
            return None
        S.last_state = out.get('state')
        day_flag = out.pop('_day_flag', None)
        dev = getattr(d, 'device', None)
        backend = '%s%s' % (dev, ' cuda-graphs' if S.graphs else ' eager')
        if not S.alive and S.next_step - S.s0 == spd and not S.graphs:
            backend = None      # a one-day block: GRAPH_STATS is the call's
        prop = self._proposal(out, context, dt, (s0, s1), inj_p, inj_s,
                              unmet, n, parts, backend=backend)
        if day_flag is not None:
            # an assumption of the graph path failed in the session by this
            # day (simulate's day check, read here with the step's other
            # checks): the step is run again, VIC as a one-day call, which
            # checks and redoes the day itself
            prop.flags.append(DeviceFlag(
                'vic: a CUDA-graph assumption failed in the day session',
                (day_flag != 0).any(), retry=self._end_session))
        return prop

    def _proposal(self, out, context, dt, steps, inj_p, inj_s, unmet, n,
                  parts=None, backend=None):
        d = self.driver
        from ..model import basin
        dev = getattr(d, 'device', None)
        graphs = basin.GRAPH_STATS.get('graphs', 0) if isinstance(
            basin.GRAPH_STATS.get('graphs', 0), int) else 0
        if backend is None:
            backend = ('%s%s' % (dev if dev is not None else 'cpu',
                                 ' cuda-graphs' if graphs else ' eager'))
        new_state = out.pop('state')
        prop = Proposal(connector=self.name, t0=context.t,
                        t1=context.t + dt, state=new_state, series=out,
                        backend=backend,
                        info=dict(steps=steps))
        fl = getattr(self, '_sa_flags', None)
        if fl:
            prop.flags.extend(fl)
            self._sa_flags = []
        if self.ledger_on:
            prop.balance = self._balance(out, inj_p, inj_s, n, unmet)
            rep = new_state.get('report') if isinstance(new_state, dict) \
                else None
            if rep is not None and not d.banded and 'storage' in rep:
                # the storage VIC carries at the end, and the opening it
                # implies through its own DEL terms: the next interval must
                # open where this one closed (plus any soil_add)
                end = rep['storage']
                prop.info['storage_end'] = end
                prop.info['storage_open'] = end - (
                    out['delsoilmoist'] + out['delswe']
                    + out['delintercept']).sum(0)
                prop.info['storage_scale'] = (
                    end.abs() + (out['delsoilmoist'].abs()
                                 + out['delswe'].abs()
                                 + out['delintercept'].abs()).sum(0))
        if self.wanted:
            un_names = {'soil_add_unmet'} | {
                'soil_add_unmet_' + x for x in self.soil_add_sources}
            prop.exports = self.exports_of(
                out, self.wanted - un_names, new_state)
            if parts is not None:
                sp = self._split_unmet(unmet, parts)
                z = None
                for nm in self.wanted & un_names:
                    if nm not in sp and z is None:
                        z = torch.zeros(d.n_cells, dtype=d.dtype,
                                        device=d.device)
                    prop.exports[nm] = sp.get(nm, z)
            elif 'soil_add_unmet' in self.wanted:
                prop.exports['soil_add_unmet'] = (
                    unmet if unmet is not None else
                    torch.zeros(d.n_cells, dtype=d.dtype, device=d.device))
        prop.info['soil_add'] = inj_s
        return prop

    def _compute_day_coupled(self, inputs, context, dt):
        """One window with the day-coupled connectors run between its
        days (the runtime's DayCoupler) through basin.simulate's day_hook:
        the window stays one call -- its CUDA graphs captured once -- and
        each day's additions are the ones a one-day window would get."""
        d = self._need_driver()
        cp = self.coupler
        s0, s1 = self._steps(context, dt)
        n = s1 - s0
        spd = d.steps_per_day
        if n % spd:
            raise ComputeError('vic: a day-coupled interval must be whole '
                               'days (%d steps at %d a day)' % (n, spd))
        state = self.state
        nc = d.n_cells
        acc = dict(p=torch.zeros(n, nc, dtype=d.dtype, device=d.device),
                   s=None, u=None, any_p=False)

        def precip(day, pa):
            if pa is None:
                return None
            if d.precip_scaled():
                raise ComputeError(
                    'vic: precip_add with a precipitation correction (pcf '
                    '!= 1 or CORRPREC TRUE) would be scaled by it; refused')
            pa = torch.as_tensor(pa, dtype=d.dtype, device=d.device)
            if bool((pa < 0).any()):
                raise ComputeError('vic: precip_add must not be negative')
            x = (pa / spd).expand(spd, -1)
            acc['p'][day * spd:(day + 1) * spd] = x
            acc['any_p'] = True
            return x

        def soil(applied, unmet):
            acc['s'] = applied if acc['s'] is None else acc['s'] + applied
            acc['u'] = unmet if acc['u'] is None else acc['u'] + unmet

        # day 0: the window's first day, from the state it starts on
        lanes0 = (self._lanes(d.mosaic(), state)
                  if state is not None and 'lanes' in state else None)
        grp = self.phase_group
        if grp is not None:
            # the window's rejection restores the plugins with VIC
            gsnap = grp.save_state()

            def _undo():
                grp.covered = False
                grp.load_state(gsnap)
            cp.undo.append(_undo)
            # inside the window this snapshot covers every day: the group
            # keeps no per-day copy of its states
            grp.covered = hasattr(grp, 'covered')
            # P1-O of the day before the window (its last day, which
            # ran in the previous window) unless the run starts here, then
            # F, U of day 0
            if state is not None:
                state, ga, gu = self._group_state(state, context.t, lanes0,
                                                  post=self.t_step > 0)
                if ga is not None:
                    soil(ga, gu)
            else:
                # F, U of the run's first step see the cold start (EFR's
                # start-of-step moisture, efr_force.c:194)
                grp.between(self._cold_view(), None, context.t, post=False)
        vin = cp.day(0, None if state is None else
                     self._day_moist(state['moist'], lanes0),
                     soil=None if state is None else self._soil_view(
                         state['moist'], state.get('soil_ice'), lanes0))
        precip(0, vin.get('precip_add'))
        sa = vin.get('soil_add')
        if sa is not None and state is None and not bool(
                torch.as_tensor(sa).any()):
            sa = None
        if sa is not None:
            state, applied, unmet = self._soil_add(state, sa)
            soil(applied, unmet)

        def hook(t_, view):
            day = t_ // spd
            moist_, cols_ = view.moist, None
            ice_ = view.soil_ice if self.soil_add_floor == 'liquid' else None
            if grp is not None:
                # P1-O of day-1 on its end state, then F, U of day
                t1_ = context.t + day * 86400.0
                sv_ = self._soil_view(moist_, view.soil_ice, view.lanes)
                wv_ = self._wur_view(sv_)
                if getattr(grp, 'wants_fluxes', False):
                    self._set_fluxes(wv_, view.day_fluxes())
                delta_ = grp.between(wv_, t1_ - 86400.0, t1_)
                self._hook_ok()
                moist_, a_, u_ = self._add_soil_cols(moist_, delta_,
                                                     view.lanes, ice=ice_)
                soil(self._cols_to_cells(a_), self._cols_to_cells(u_))
                cols_ = a_
            vin_ = cp.day(day, self._day_moist(moist_, view.lanes),
                          soil=self._soil_view(moist_, view.soil_ice,
                                               view.lanes))
            x = precip(day, vin_.get('precip_add'))
            if x is not None:
                view.prcp[t_:t_ + spd] += x
            sa_ = vin_.get('soil_add')
            if sa_ is None:
                return None if cols_ is None else (moist_, cols_)
            self._hook_ok()
            # basin's day_hook takes the addition on its own cells (the
            # band columns when banded); the ledger takes it on cells
            new_, applied_, unmet_ = self._add_soil_cols(
                moist_, sa_, view.lanes, ice=ice_)
            soil(self._cols_to_cells(applied_), self._cols_to_cells(unmet_))
            return new_, (applied_ if cols_ is None else cols_ + applied_)

        kw = {}
        core = (grp.vic_core() if grp is not None
                and hasattr(grp, 'vic_core') else None)
        if core is not None:
            # VIC-WUR's vic_run hooks (vicwur.core.VicwurCore): paddy Ksat,
            # Catm/b_co2, land use's tile states and Cv, stored stress
            kw['vicwur_core'] = core
        try:
            out = d.run(s0, s1, state,
                        precip_add=acc['p'] if acc['any_p'] else None,
                        lai=inputs.get('lai'), day_hook=hook, **kw)
        except ComputeError:
            raise
        except Exception as e:                           # noqa: BLE001
            raise ComputeError('vic: steps %d..%d failed: %s: %s'
                               % (s0, s1, type(e).__name__, e)) from e
        finally:
            if grp is not None and hasattr(grp, 'covered'):
                grp.covered = False
        last = None
        if getattr(grp, 'wants_fluxes', False):
            # the window's last day, read by the plugins at the next
            # window's start (P1-O of that day run there)
            last = dict(runoff=out['runoff'][-spd:].sum(0),
                        baseflow=out['baseflow'][-spd:].sum(0))
            if self.gwrecharge_view:
                last['gwrecharge'] = out['gwrecharge'][-spd:].sum(0)
            vw = out.get('vicwur') or {}
            ws = vw.get('water_stress')
            if ws is not None:
                last['water_stress'] = ws[-1]
            if vw.get('water_stress_layer_end') is not None:
                # the stored layer stress the next window starts from
                last['water_stress_layer'] = vw['water_stress_layer_end']
            # the last step's per-tile fluxes (vicwur_core tile_fluxes)
            last.update(vw.get('tile_fluxes_end') or {})

        prop = self._proposal(out, context, dt, (s0, s1),
                              acc['p'] if acc['any_p'] else None, acc['s'],
                              acc['u'], n)
        prop.info['wur_last'] = last
        return prop

    def _set_fluxes(self, wv, fl):
        """The day's OUT_RUNOFF, OUT_BASEFLOW on the cells (a banded case's
        columns area-weighted) and the stored water stress (columns) onto
        the plugins' view."""
        if wv is None or fl is None:
            return
        wv.runoff = self._banded_cells(fl['runoff'])
        wv.baseflow = self._banded_cells(fl['baseflow'])
        if self.gwrecharge_view:
            wv.gwrecharge = self._banded_cells(fl['gwrecharge'])
        wv.water_stress = fl.get('water_stress')
        # the per-tile fluxes of the step just run, [column, tile(, layer)]
        # (vicwur_core tile_fluxes: pet, transp, esoil, canop)
        tf = {k[3:]: v for k, v in fl.items() if k.startswith('tf_')}
        wv.tile_fluxes = tf or None

    def _cold_view(self):
        """The plugins' view of the cold start (generate_default_state.c
        :75-84: every tile at the cell's init_moist, clipped at max_moist),
        for F, U of the run's first step; None without tiles."""
        from ..model import basin
        d = self.driver
        mos = d.mosaic()
        if mos is None or getattr(mos, 'cv', None) is None:
            return None
        sp = d.soil()
        im = (d.case.init_moist if isinstance(d, CaseDriver)
              else d.static.get('init_moist'))
        n = mos.cv.shape[0]
        _, m = basin.initial_states(n, sp, d.dtype, d.device, init_moist=im)
        m = m.unsqueeze(1).expand(n, mos.cv.shape[1], m.shape[-1]).clone()
        return self._wur_view(self._soil_view(m, None, None))

    def _hook_ok(self):
        d = self.driver
        opts = (d.case.options if isinstance(d, CaseDriver) else d.static)
        if d.banded and not self.soil_add_bands:
            raise ComputeError('vic: soil_add with snow bands (the state is '
                               'per band) is not supported')
        if opts.get('frozen_soil') and not self.soil_add_frozen:
            raise ComputeError('vic: soil_add with FROZEN_SOIL (the layer '
                               'holds ice) is not supported')

    def _soil_add(self, state, add):
        """soil_add at the interval start: the state with the addition,
        the cell total applied and the cell total unmet (mm)."""
        d = self.driver
        self._soil_add_ok(state)
        lanes = self._lanes(d.mosaic(), state) if 'lanes' in state else None
        new, applied, unmet = self._add_soil_cols(
            state['moist'], add, lanes,
            ice=state.get('soil_ice') if self.soil_add_floor == 'liquid'
            else None)
        st = dict(state)
        st['moist'] = new
        rep = dict(st['report'])
        # the report is per column; the cell total only without bands
        rep['total_moist'] = rep['total_moist'] + applied
        rep['storage'] = rep['storage'] + applied
        st['report'] = rep
        if self._band_ex() is not None:
            applied, unmet = self._to_cells(applied), self._to_cells(unmet)
        return st, applied, unmet

    def _band_ex(self):
        """The snow-band expansion when soil_add_bands is on and the case
        is banded, else None."""
        if not self.soil_add_bands:
            return None
        return getattr(getattr(self.driver, 'case', None), 'snowbands', None)

    def _band_idx(self, device):
        """(cell[column], band[column], n_band, AreaFract[column], n_cell)
        of the band expansion on `device`, built once."""
        ex = getattr(self.driver.case, 'snowbands', None)
        c = getattr(self, '_band_cache', None)
        if c is None or c[0] is not ex or c[1] != device:
            cell = torch.as_tensor(np.asarray(ex['cell']), dtype=torch.long,
                                   device=device)
            band = torch.as_tensor(np.asarray(ex['band']), dtype=torch.long,
                                   device=device)
            w = torch.as_tensor(np.asarray(ex['weight']),
                                dtype=self.driver.dtype, device=device)
            nb = int(np.asarray(ex['band']).max()) + 1
            c = (ex, device, (cell, band, nb, w, int(ex['n_cell'])))
            self._band_cache = c
        return c[2]

    def _to_cells(self, x, axis=0):
        """[column] -> [cell] on `axis`: the bands' area-weighted sum
        (snowbands.contract; vic_run.c:349-422)."""
        cell, _, _, w, n = self._band_idx(x.device)
        x = x.movedim(axis, 0)
        out = x.new_zeros((n,) + tuple(x.shape[1:]))
        # a cell's bands in band order on a GPU too (core.ordered: no
        # atomics, the same sums every run)
        from ..core import ordered
        ordered.index_add_(out, 0, cell, x * w.reshape((-1,) + (1,) * (
            x.dim() - 1)).to(x.dtype))
        return out.movedim(0, axis)

    def _banded_cells(self, x, axis=0):
        """A banded case's per-column result on cells (the ledger and the
        exports are on the domain's cells); unchanged without bands."""
        if not getattr(self.driver, 'banded', False):
            return x
        return self._to_cells(x, axis)

    def _cols_to_cells(self, x):
        return x if self._band_ex() is None else self._to_cells(x)

    def _day_moist(self, moist, lanes):
        """The [cell, layer] soil moisture the day-coupled connectors
        read (on cells, also with bands when soil_add_bands is on)."""
        return self._cols_to_cells(self._cell_layers(moist, lanes))

    def _soil_view(self, moist, ice, lanes):
        """The read-only per-tile/band soil state (Context.soil)."""
        from ..vicwur.view import SoilView
        mos = self.driver.mosaic()
        if mos is None or getattr(mos, 'cv', None) is None:
            return None                   # no tiles: no per-tile view
        cv = mos.cv
        cvf = getattr(self.phase_group, 'cv_columns', None)
        if cvf is not None:
            # VIC-WUR 3.0 land use: the tile Cv in force (lu_apply.c:995-1029)
            c_ = cvf()
            if c_ is not None:
                cv = c_
        if self._band_ex() is None:
            return SoilView(moist, ice, lanes, cv)
        cell, band, nb, w, n = self._band_idx(moist.device)
        return SoilView(moist, ice, lanes, cv, (cell, band, nb), n, w)

    def _wur_view(self, sv):
        """The phase group's writable StepView over `sv`."""
        mx = self.driver.soil().max_moist
        mx = torch.as_tensor(mx, dtype=self.driver.dtype,
                             device=self.driver.device)
        if self._band_ex() is not None:
            cell = self._band_idx(mx.device)[0]
            mxc = mx.new_zeros((sv.n_cell,) + tuple(mx.shape[1:]))
            mxc[cell] = mx                  # a cell's bands share it
            mx = mxc
        return sv.step_view(mx, getattr(self.phase_group, 'wfc', None))

    def _group_state(self, state, t0, lanes, post=True):
        """At a window's start: the phase group between the previous day
        and day 0 on `state`; returns the state with its edits."""
        sv = self._soil_view(state['moist'], state.get('soil_ice'), lanes)
        wv = self._wur_view(sv)
        if post and self._wur_last is not None:
            self._set_fluxes(wv, self._wur_last)
        delta = self.phase_group.between(wv, t0 - 86400.0, t0, post=post)
        if not post:
            return state, None, None
        self._hook_ok()
        return self._soil_add(state, delta)

    def _to_columns(self, add):
        """A per-cell soil_add on the banded case's columns (pseudo-cells):
        [cell, tile, band, layer] picks each column's band; a 1-3-d
        addition is the same mm on every band."""
        ex = self._band_ex()
        add = torch.as_tensor(add, dtype=self.driver.dtype,
                              device=self.driver.device)
        if ex is None:
            if add.dim() == 4:
                if add.shape[2] != 1:
                    raise ComputeError('vic: a [cell, tile, band, layer] '
                                       'soil_add needs a snow-band case')
                return add[:, :, 0]
            return add
        c, b = self._band_idx(add.device)[:2]
        if add.dim() == 4:
            return add[c, :, b]
        return add[c]

    def _soil_add_ok(self, state):
        if self.driver.banded and not self.soil_add_bands:
            raise ComputeError('vic: soil_add with snow bands (the state is '
                               'per band) is not supported')
        if state is None:
            raise ComputeError('vic: soil_add before the first interval '
                               '(no state to add to yet); give it from the '
                               'second interval on')
        if state.get('soil_ice') is not None and not self.soil_add_frozen:
            raise ComputeError('vic: soil_add with FROZEN_SOIL (the layer '
                               'holds ice) is not supported')

    def _cell_layers(self, x, lanes):
        """[cell, layer] cover-weighted sums of a per-tile [cell, tile,
        layer] (or per-lane [lane, 1, layer]) quantity, summed as the padded
        grid sums it: over tiles, uncovered tiles zero."""
        mosaic = self.driver.mosaic()
        if x.dim() == 2:
            return x
        cv = mosaic.cv.to(x.dtype)
        nl = x.shape[-1]
        if lanes is not None:
            lc, lt = lanes.lane_cell, lanes.lane_tile
            pad = x.new_zeros(cv.shape[0], cv.shape[1], nl)
            pad[lc, lt] = x.reshape(x.shape[0], nl) * cv[lc, lt].unsqueeze(-1)
            return pad.sum(1)
        return (x * cv.unsqueeze(-1)).sum(1)

    def _add_soil(self, moist, add, lanes, ice=None):
        """_add_soil_cols with the cell totals on cells (not columns)."""
        new, applied, unmet = self._add_soil_cols(moist, add, lanes, ice)
        if self._band_ex() is not None:
            applied, unmet = self._to_cells(applied), self._to_cells(unmet)
        return new, applied, unmet

    def _add_soil_cols(self, moist, add, lanes, ice=None):
        """(new moisture, applied, unmet): `add` ([cell] into layer 0,
        [cell, layer] or [cell, tile, layer], mm; with soil_add_bands also
        [cell, tile, band, layer]) on `moist` in the loop's layout (padded
        or lanes), checked against the layer's capacity and residual
        moisture as soil_add_mode says. The totals are per column (a
        column is a cell without bands)."""
        d = self.driver
        if self.soil_add_bands or (torch.is_tensor(add) and add.dim() == 4):
            add = self._to_columns(add)
        add = torch.as_tensor(add, dtype=moist.dtype, device=moist.device)
        soil = d.soil()
        nl = moist.shape[-1]
        if add.dim() == 1:
            # [cell]: into the top layer, as VIC-WUR's irrigation
            # (irr_use.c:265 adds to layer 0)
            add = torch.cat([add.unsqueeze(-1), add.new_zeros(
                add.shape[0], nl - 1)], -1)
        mx, rs = soil.max_moist, getattr(soil, 'resid_moist', None)
        if lanes is not None:
            # the GPU's compact layout: one row per covered (cell, tile)
            # (torchvic.model.lanes); the same additions at the lanes
            lc, lt = lanes.lane_cell, lanes.lane_tile
            add_t = add[lc] if add.dim() == 2 else add[lc, lt]
            mx_, rs_ = mx[lc], (None if rs is None else rs[lc])
            if moist.dim() == 3:
                # each lane is a "cell" of one tile: [lane, 1, layer]
                add_t, mx_ = add_t.unsqueeze(1), mx_.unsqueeze(1)
                rs_ = None if rs_ is None else rs_.unsqueeze(1)
        else:
            if moist.dim() == 3 and add.dim() == 2:
                add_t = add.unsqueeze(1).expand_as(moist)
            else:
                add_t = add
            mx_ = mx.unsqueeze(1) if moist.dim() == 3 else mx
            rs_ = None if rs is None else (rs.unsqueeze(1) if moist.dim() == 3
                                           else rs)
        new = moist + add_t
        floor = self.soil_add_floor
        if floor != 'resid':
            # VIC-WUR's pumping: a negative addition stops at 0 (or at the
            # layer's ice, liquid only); what could not be taken is unmet
            # (negative). Additions >= 0 are untouched.
            if floor == 'zero' or ice is None:
                lo = torch.zeros_like(moist)
            else:
                lo = self._ice_eff(ice, moist, lanes)
            new = torch.where(add_t < 0,
                              torch.maximum(new, torch.minimum(lo, moist)),
                              new)
        if self.soil_add_mode == 'cap':
            # what does not fit stays unmet and is handed back
            new = torch.minimum(new, torch.maximum(mx_, moist))
        elif self.soil_add_mode == 'over':
            # VIC-WUR 3.0's capillary-rise state edit checks the Cv-weighted
            # cell total only (vic_runner.py update_statefile) and leaves a
            # tile above max_moist; the next vic_run's runoff() moves the
            # excess down and out (runoff.c:242-268). Nothing to refuse.
            pass
        else:
            self._refuse_if(
                (new > mx_ + 1e-9).any(),
                'vic: soil_add would fill a layer above its capacity '
                '(max_moist); refused (soil_add_mode: cap takes what fits '
                'and reports the rest)')
        if rs_ is not None and floor == 'resid':
            low = (add_t < 0) & (new < rs_ - 1e-9)
            self._refuse_if(low.any(), 'vic: soil_add would take a layer '
                            'below its residual moisture; refused')
        applied = self._cell_layers(new - moist, lanes).sum(-1)
        unmet = self._cell_layers(add_t, lanes).sum(-1) - applied
        return new, applied, unmet

    def _refuse_if(self, bad, message):
        """A soil_add check: on the CPU it raises now (ComputeError); on a
        GPU it is a device flag on the interval's proposal, read with the
        step's other checks (the step is rejected all the same, without a
        host read of its own)."""
        if bad.device.type == 'cpu' or not hasattr(self, '_sa_flags'):
            if bool(bad):
                raise ComputeError(message)
            return
        self._sa_flags.append(DeviceFlag(message, bad))

    def _ice_eff(self, ice, moist, lanes):
        """The layer's ice as the surface sees it: the frost areas'
        weighted sum (basin._ice_eff), in `moist`'s layout."""
        if ice.dim() == moist.dim() + 1:
            from ..core import frozen
            w = torch.as_tensor(frozen.frost_fractions(ice.shape[-1]),
                                dtype=ice.dtype, device=ice.device)
            ice = (ice * w).sum(-1)
        return ice.reshape(moist.shape)

    def _lanes(self, mosaic, state):
        from ..model import lanes as _lanes_mod
        ln = getattr(self, '_lanes_cache', None)
        if ln is None:
            ln = self._lanes_cache = _lanes_mod.Lanes(mosaic.cv)
        if ln.n_lanes != int(state['lanes']['n_lanes']):
            raise ComputeError('vic: the state is on %d lanes, the mosaic '
                               'has %d covered tiles'
                               % (state['lanes']['n_lanes'], ln.n_lanes))
        return ln

    def _balance(self, out, inj_p, inj_s, n, unmet=None):
        """The interval's VIC balance (see ledger): per cell in mm and as
        domain volumes in m3."""
        if getattr(self.driver, 'banded', False):
            # the band columns' series, area-weighted onto the cells
            ncol = out['runoff'].shape[1]
            out = {k: (self._to_cells(v, 1) if torch.is_tensor(v)
                       and v.dim() >= 2 and v.shape[1] == ncol else v)
                   for k, v in out.items()}
        P = (out['rainf'] + out['snowf'])
        Q = out['runoff'] + out['baseflow']
        if 'gwrecharge' in out:
            # VIC-WUR 3.0 GWM: OUT_GWRECHARGE leaves the column
            # (put_data.c:537)
            Q = Q + out['gwrecharge']
        dS_t = out['delsoilmoist'] + out['delswe'] + out['delintercept']
        we = out['water_error']
        terms = torch.stack([P.sum(0), out['et'].sum(0), Q.sum(0),
                             dS_t.sum(0), we.sum(0)])
        scale = (P.abs().sum(0) + out['et'].abs().sum(0) + Q.abs().sum(0)
                 + dS_t.abs().sum(0) + we.abs().sum(0))
        # the DEL terms and the water error are differences of storage
        # totals: their rounding is relative to the storage, not to the
        # (small) change, so the storage enters the scale too
        stor = None
        for k in ('soil_liq', 'soil_ice'):
            if k in out:
                v = out[k].sum(-1) if out[k].dim() == 3 else out[k]
                stor = v.abs() if stor is None else stor + v.abs()
        for k in ('swe', 'wdew', 'snow_canopy'):
            if k in out:
                stor = out[k].abs() if stor is None else stor + out[k].abs()
        if stor is not None:
            scale = scale + 4.0 * stor.sum(0)
        Pc, ETc, Qc, dSc, Wc = terms
        Ic = torch.zeros_like(Pc)
        if inj_p is not None:
            ip = inj_p.sum(0)
            Pc = Pc - ip
            Ic = Ic + ip
            scale = scale + inj_p.abs().sum(0)
        if inj_s is not None:
            Ic = Ic + inj_s
            dSc = dSc + inj_s
            scale = scale + 2 * inj_s.abs()
        if unmet is not None:
            # delivered but not taken: in I_ext and handed back in Q_out
            Ic = Ic + unmet
            Qc = Qc + unmet
            scale = scale + 2 * unmet.abs()
        resid = Pc + Ic - ETc - Qc - dSc - Wc
        eps = float(torch.finfo(P.dtype).eps)
        per_cell = dict(residual=resid, scale=scale, n_terms=8 * n + 8,
                        eps=eps)
        d = self.driver
        if d.area is not None:
            vc = getattr(self, '_vol_cache', None)
            if vc is None or vc[0] != P.device:
                # the cells' m3 per mm, copied to the device once
                vc = self._vol_cache = (P.device, torch.as_tensor(
                    d.area * (d.frac if d.frac is not None else 1.0),
                    dtype=torch.float64, device=P.device) / 1000.0)
            vol = vc[1]
            M = torch.stack([Pc, Ic, ETc, Qc, dSc, Wc, scale]).to(
                torch.float64)
            # stays on the device; the ledger reads it at its check point
            v = (M * vol).sum(1)
            b = Balance(P=v[0], I_ext=v[1], ET=v[2], Q_out=v[3], dS=v[4],
                        closure=v[5], n_terms=(8 * n + 8) * d.n_cells,
                        eps=eps, scale=v[6], per_cell=per_cell)
        else:
            b = Balance(n_terms=1, eps=eps, scale=0.0, per_cell=per_cell)
        return b

    def validate(self, proposal, context):
        # device flags, read on the host with the step's ledger terms (one
        # copy a step, runtime._raise_flags), not one sync a check
        problems = []
        st = proposal.state
        m = st.get('moist') if isinstance(st, dict) else None
        if m is not None:
            problems.append(DeviceFlag(
                'vic: the soil moisture of the proposed state is not finite',
                ~torch.isfinite(m).all()))
        info = proposal.info
        if self.storage_end is not None and 'storage_open' in info:
            # continuity: this interval opens on the storage the last one
            # committed (VIC's report totals, carried in its state)
            want = self.storage_end
            if info.get('soil_add') is not None:
                want = want + info['soil_add']
            gap = (info['storage_open'] - want).abs()
            eps = float(torch.finfo(gap.dtype).eps)
            tol = 4 * eps * (info['steps'][1] - info['steps'][0] + 2) * (
                info['storage_scale'] + want.abs())
            i = torch.argmax(gap - tol).reshape(1)
            problems.append(DeviceFlag(
                'vic: the interval opens on %.9g mm at cell %d, the previous '
                'one closed on %.9g mm: the state was not carried',
                ~(gap <= tol).all(),
                # (index_select: indexing with a device tensor reads it on
                # the host)
                values=(info['storage_open'].index_select(0, i)[0], i[0],
                        want.index_select(0, i)[0])))
        return problems

    #: what the runtime's output manager keeps of this connector's series
    keep_outputs = 'memory'

    def commit(self, proposal, context):
        if hasattr(self.driver, 'aggregate') and proposal.series is not None:
            self.driver.aggregate(proposal.series, *proposal.info['steps'])
        self.state = proposal.state
        self.t_step = proposal.info['steps'][1]
        self.last_backend = proposal.backend
        if 'wur_last' in proposal.info:
            self._wur_last = proposal.info['wur_last']
        self.windows += 1
        if 'storage_end' in proposal.info:
            self.storage_end = proposal.info['storage_end']

    def exports_of(self, out, names, state=None):
        """{name: tensor} of the declared outputs `names` from a window's
        simulate output."""
        ex = {}
        flux = {n: srcs for n, _, srcs in _FLUX}
        for nm in names:
            if nm == 'total_runoff_daily':
                spd = int(self.driver.steps_per_day)
                nt = out['runoff'].shape[0]
                ex[nm] = torch.stack([
                    out['runoff'][i:i + spd].sum(0)
                    + out['baseflow'][i:i + spd].sum(0)
                    for i in range(0, nt, spd)])
            elif nm in flux:
                v = None
                for s in flux[nm]:
                    if s not in out:
                        raise ComputeError('vic: this run computes no %r '
                                           '(needed for export %r)'
                                           % (s, nm))
                    v = out[s].sum(0) if v is None else v + out[s].sum(0)
                ex[nm] = v
            elif nm == 'soil_moisture':
                # the end state's layer moisture, cover-weighted per cell
                # (the same sum a day-coupled window gives its days)
                st = state if state is not None else {}
                if 'moist' in st:
                    ln = (self._lanes(self.driver.mosaic(), st)
                          if 'lanes' in st else None)
                    ex[nm] = self._cell_layers(st['moist'], ln)
                else:
                    ex[nm] = (out['soil_liq'] + out['soil_ice'])[-1]
            elif nm in ('swe', 'wdew', 'snow_canopy'):
                if nm not in out:
                    raise ComputeError('vic: this run computes no %r' % nm)
                ex[nm] = out[nm][-1]
            else:
                raise ComputeError('vic: %r is not an output of VIC' % nm)
            ex[nm] = self._banded_cells(
                ex[nm], 1 if nm == 'total_runoff_daily' else 0)
        return ex

    def finalize(self, context):
        self._end_session()
        if self.phase_group is not None and self.state is not None \
                and self.t_step > 0:
            # C runs plugin_run after the run's last vic_run too: P1-O of
            # the last day, on its end state (its soil edits have no step
            # left to enter, so they are dropped)
            st = self.state
            lanes = (self._lanes(self.driver.mosaic(), st)
                     if 'lanes' in st else None)
            sv = self._soil_view(st['moist'], st.get('soil_ice'), lanes)
            wv = self._wur_view(sv)
            if self._wur_last is not None:
                self._set_fluxes(wv, self._wur_last)
            self.phase_group.between(
                wv, self.t_step * self.model_step_s - 86400.0, None,
                pre=False)
        close = getattr(self.driver, 'close', None)
        if close is not None:
            close()

    def write_history(self, result_dir, log=None):
        """A streamed case's history streams (vic_stream); [] otherwise."""
        w = getattr(self.driver, 'write_history', None)
        return [] if w is None else w(result_dir, log=log)

    def save_state(self):
        st = None
        if self.state is not None:
            st = model_state.map_state(self.state, lambda t: t.cpu())
        se = None if self.storage_end is None else self.storage_end.cpu()
        wl = None if self._wur_last is None else {
            k: v.cpu() for k, v in self._wur_last.items()}
        return dict(state=st, t_step=self.t_step, windows=self.windows,
                    storage_end=se, wur_last=wl)

    def w3_restart(self, context, t):
        """checkpoint.w3_restart: VIC's state file holds the VIC state
        that the checkpoint holds, but not VIC-WUR's stored layer water
        stress (initialize_soil.c:62-66 sets it to 0 in vic_init on every
        start): with a VIC-WUR phase group it is zeroed when the group
        enters the restart step (after P1-O of the day before, which read
        the day's stress)."""
        grp = self.phase_group
        if isinstance(grp, str):
            from .connectors.vicwur_common import get_group
            grp = get_group(grp)
        pend = getattr(grp, 'w3_pending', None)
        if pend is None:
            return
        step = int(round(float(t) / 86400.0))

        def reset():
            wl = self._wur_last
            if wl is not None and wl.get('water_stress_layer') is not None:
                wl = dict(wl)
                wl['water_stress_layer'] = torch.zeros_like(
                    wl['water_stress_layer'])
                self._wur_last = wl
        pend(step, reset)

    def snapshot(self):
        """What restore() needs to come back here (runtime._window): the
        accepted state by reference -- a commit replaces it, nothing writes
        into it -- and the history's length."""
        h = getattr(self.driver, 'hist_mark', None)
        return dict(state=self.state, t_step=self.t_step,
                    windows=self.windows, storage_end=self.storage_end,
                    wur_last=self._wur_last,
                    hist=h() if h is not None else None)

    def restore(self, snap):
        self._end_session()
        self.state = snap['state']
        self.t_step = snap['t_step']
        self.windows = snap['windows']
        self.storage_end = snap['storage_end']
        self._wur_last = snap['wur_last']
        if snap['hist'] is not None:
            self.driver.hist_reset(snap['hist'])

    def load_state(self, blob):
        self._end_session()
        dev = getattr(self.driver, 'device', None)
        st = blob['state']
        if st is not None and dev is not None:
            st = model_state.map_state(st, lambda t: t.to(dev))
        self.state = st
        self.t_step = int(blob['t_step'])
        self.windows = int(blob.get('windows', 0))
        se = blob.get('storage_end')
        self.storage_end = (None if se is None else
                            se.to(dev) if dev is not None else se)
        wl = blob.get('wur_last')
        self._wur_last = None if wl is None else {
            k: (v.to(dev) if dev is not None else v) for k, v in wl.items()}


def load_driver(cfg, context):
    """A CaseDriver from a connector's configuration block (global_param
    and case.load's keyword arguments)."""
    from .. import case as _case, config as _config
    gp = cfg.get('global_param')
    if not gp:
        raise CouplingError('vic: global_param is required')
    dtype = context.dtype
    if isinstance(dtype, str):
        dtype = _config.DTYPES[dtype]
    # a large forcing is read a window at a time (vic_stream), as the
    # production pipeline reads it; everything else is case.load
    from .vic_stream import open_stream
    drv = open_stream(
        gp, context.device, dtype, core=cfg.get('core'),
        no_energy=bool(cfg.get('no_energy', False)),
        sample_data=cfg.get('sample_data'),
        stream=cfg.get('stream', 'auto'),
        daily=bool(cfg.get('daily', False)),
        steps=int(cfg.get('steps') or 0),
        first_step=int(cfg.get('first_step') or 0),
        feed=cfg.get('forcing_feed'), prefetch=int(cfg.get('prefetch', 2)),
        write_procs=int(cfg.get('write_procs', 4)),
        snow_bands=cfg.get('snow_bands', 'expand'),
        log=context.log or (lambda *a: None))
    btf = cfg.get('band_tfactor')
    if drv is not None:
        if btf:
            raise CouplingError('vic: band_tfactor needs case.load (the '
                                'snow bands expanded to columns), not the '
                                'forcing stream')
        return drv
    kw = dict(device=context.device, dtype=dtype,
              core=cfg.get('core'), steps=int(cfg.get('steps') or 0),
              first_step=int(cfg.get('first_step') or 0),
              daily=bool(cfg.get('daily', False)),
              no_energy=bool(cfg.get('no_energy', False)),
              sample_data=cfg.get('sample_data'),
              log=context.log or (lambda *a: None))
    if btf:
        # VIC-WUR 3.0's crop file Tfactor on band 0 (crop_init.c:157-163)
        kw['band_tfactor'] = {int(k): v for k, v in btf.items()}
    case = _case.load(gp, **kw)
    if isinstance(case, _case.ForceCase):
        raise CouplingError('vic: %s is an OUTPUT_FORCE run (it writes '
                            'forcing, no model to couple)' % gp)
    return CaseDriver(case, forcing_on=cfg.get('forcing_on', 'device'))

