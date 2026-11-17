"""The VIC-WUR phase scheduler (docs/connectors/vicwur.md).

One model step runs its plugin calls in the fixed order of VIC-WUR's C
driver, not in the order of the connector graph. `plan(options)` is that
order as data (every entry cites its C line, paths under
`<data root>/build/vicwur3/src_tree/vic/`); `PhaseScheduler` runs it:

    F  forcing          plugin_force            plugins/general/src/plugin_simulation.c:33-58
    U  step vars        plugin_update_step_vars plugin_simulation.c:63-85
    V  vic_run          update_step_vars, vic_run, put_data
                                                drivers/shared_image/src/vic_image_run.c:59-79
    P1 pass 1 (cells)   plugin_run, first loop  plugin_simulation.c:102-119
    P2 sweep (routing)  plugin_run, routing     plugin_simulation.c:121-145
    P3 pass 3 (cells)   plugin_run, last loop   plugin_simulation.c:148-165
    O  put_data         plugin_put_data_all     vic_image_run.c:82; plugin_simulation.c:172-218

and the step itself is vic_force, vic_image_run, vic_write_output
(drivers/image/src/vic_image.c:150-161).

A component is any object with some of the hooks named in `HOOKS`
(`pre_force`, `pre_vic`, `pass1`, `sweep_slot(level, slot, cells)`,
`pass3`, `put_data`; plus the named sub-calls of `plan`: `pass1_demand`,
`pass3_nonrenew`, `pass3_withdrawn`, `random_run`), optionally
`set_time(t)` (the phase group's clock, s) and `set_view(view)` (the
step's StepView, VIC-WUR's all_vars) and, to take part in a transaction,
`save_state()` and `load_state(blob)` (the Connector contract). Components are registered by
kind (`KINDS`); a kind absent from the options is never called, as a plugin
switched off in C is not.

Transaction. `step()` needs a registered 'vic' component with
save_state/load_state and snapshots every registered component (VIC
included) and the step-local view before F. When all phases return, the
view's accumulated delta is handed to `commit` (the soil_add of the next
vic_run) and the snapshots are dropped. Any exception restores every
component, VIC too, and the view: the step commits all or nothing.

Day-coupled windows. Inside a VIC window VIC runs its days in one call and
the scheduler runs between them (`between`, called from VicConnector's
day hook): at the boundary after day d, P1-O of day d on the end-of-day-d
state, then F, U of day d+1; each group is told its clock (`set_time`). The
window's own rollback (DayCoupler) restores the scheduler with VIC. No host
sync here: the scheduler holds only Python control flow.
"""
from collections import namedtuple
from dataclasses import dataclass

PHASES = ('F', 'U', 'V', 'P1', 'P2', 'P3', 'O')

#: the hook a phase calls on a component
HOOKS = {'F': 'pre_force', 'U': 'pre_vic', 'P1': 'pass1', 'P3': 'pass3',
         'O': 'put_data'}

#: sweep slots of P2, in order (plugin_simulation.c:127-139)
SLOTS = ('local_dam', 'route', 'river_exchange', 'wu_local', 'global_dam',
         'wu_remote')

BASIN, FILE, RANDOM = 'basin', 'file', 'random'

Call = namedtuple('Call', 'phase kind hook cite')


@dataclass(frozen=True)
class Options(object):
    """plugin_options, the flags the order reads (the C name in caps)."""
    force_co2: bool = False        # FORCE_CO2
    routing: bool = False          # ROUTING
    force_routing: bool = False    # FORCE_ROUTING
    force_landuse: bool = False    # FORCE_LANDUSE
    efr: bool = False              # EFR
    wateruse: bool = False         # WATERUSE
    wofost: bool = False           # WOFOST
    cropsyst: bool = False         # CropSyst crop day in P3 (no C name)
    irrigation: bool = False       # IRRIGATION
    dams: bool = False             # DAMS
    local_with: bool = False       # LOCAL_WITH
    remote_with: bool = False      # REMOTE_WITH
    nonrenew_with: bool = False    # NONRENEW_WITH
    potential_irrigation: bool = False   # POTENTIAL_IRRIGATION
    # VIC-WUR 3.0 MODFLOW coupling (GWM, plugins/modflow)
    groundwater: bool = False
    #: mode fixed of the groundwater connector: the river-aquifer exchange
    #: is taken from / given to the river's ring in the sweep, after its
    #: routing (no C line: a TorchVIC slot, bounded by the water there)
    river_exchange: bool = False
    decomposition: str = BASIN     # DECOMPOSITION


def plan(o):
    """The step's plugin calls for options `o`, in VIC-WUR's order, as
    [Call(phase, kind, hook, cite)]. P2's sweep entries carry the slot as
    hook ('sweep:<slot>'); the per-level loop wraps them in `run`."""
    P = 'plugin_simulation.c'
    c = []

    def add(ph, kind, hook, cite):
        c.append(Call(ph, kind, hook, cite))

    # F: plugin_force
    if o.force_co2:
        add('F', 'co2', 'pre_force', P + ':39-41')
    if o.routing and o.force_routing:
        add('F', 'routing', 'pre_force', P + ':42-44')
    if o.force_landuse:
        add('F', 'landuse', 'pre_force', P + ':45-47')
    if o.efr:
        add('F', 'efr', 'pre_force', P + ':48-50')
    if o.wateruse:
        add('F', 'wateruse', 'pre_force', P + ':51-53')
    if o.wofost:
        add('F', 'wofost', 'pre_force', P + ':54-56')
    # U: plugin_update_step_vars (per cell: co2, crop then wofost, irrigation)
    if o.force_co2:
        add('U', 'co2', 'pre_vic', P + ':74-76')
    if o.wofost:
        add('U', 'wofost', 'pre_vic', P + ':77-80')
    if o.irrigation:
        add('U', 'irrigation', 'pre_vic', P + ':81-83')
    # V: the VIC step, run by the caller
    add('V', 'vic', 'run', 'vic_image_run.c:59-79')
    # P1: parallel per cell
    if o.irrigation:
        add('P1', 'irrigation', 'pass1', P + ':103-105 irr_run_requirement')
        if o.wofost:
            add('P1', 'wofost', 'pass1', P + ':105-107 crop_adjust_requirement')
        if o.wateruse:
            add('P1', 'irrigation', 'pass1_demand', P + ':108-110 irr_set_demand')
    if o.wateruse:
        add('P1', 'wateruse', 'pass1', P + ':112-114 wu_run')
    if o.routing or (o.wateruse and o.nonrenew_with):
        add('P1', 'routing', 'pass1', P + ':115-118 rout_run')
    # P2: serial over routing_order (here level by level)
    if o.routing:
        if o.decomposition in (BASIN, FILE):
            if o.dams:
                add('P2', 'dams', 'sweep:local_dam', P + ':127-129')
            add('P2', 'routing', 'sweep:route', P + ':130')
            if o.river_exchange:
                add('P2', 'groundwater', 'sweep:river_exchange',
                    'TorchVIC fixed mode (after :130, before :131)')
            if o.wateruse and o.local_with:
                add('P2', 'wateruse', 'sweep:wu_local', P + ':131-133')
            if o.dams:
                add('P2', 'dams', 'sweep:global_dam', P + ':134-136')
            if o.wateruse and o.remote_with:
                add('P2', 'wateruse', 'sweep:wu_remote', P + ':137-139')
        elif o.decomposition == RANDOM:
            # rout_random_run only: no dams, no local or remote water use
            add('P2', 'routing', 'random_run', P + ':142-144')
        else:
            raise ValueError('unknown DECOMPOSITION %r' % (o.decomposition,))
    # P3: parallel per cell
    if o.wateruse and o.nonrenew_with:
        add('P3', 'wateruse', 'pass3_nonrenew', P + ':150-152 wu_nonrenew')
    if o.irrigation:
        if o.potential_irrigation or o.wateruse or o.wofost:
            add('P3', 'irrigation', 'pass3_withdrawn',
                P + ':154-158 irr_get_withdrawn')
        add('P3', 'irrigation', 'pass3', P + ':159 irr_run_shortage')
    if o.wofost:
        add('P3', 'wofost', 'pass3', P + ':162-164 crop_run')
    if o.cropsyst:
        # VIC-CropSyst runs the land unit's process_day at the end of the
        # day, after the day's surface fluxes and the irrigation
        # (VIC_CropSyst/full_energy.c:628-650); here that is P3, in the
        # slot of crop_run, on the moisture after irrigation. Its uptake
        # and canopy reach VIC at the next step (VIC_soil.cpp:231-232).
        add('P3', 'cropsyst', 'pass3', P + ':162-164 crop_run slot')
    # O: plugin_put_data per cell, then plugin_store_error
    if o.routing or (o.wateruse and o.nonrenew_with):
        add('O', 'routing', 'put_data', P + ':194-197')
    if o.force_landuse:
        add('O', 'landuse', 'put_data', P + ':198-200')
    if o.efr:
        add('O', 'efr', 'put_data', P + ':201-203')
    if o.dams:
        add('O', 'dams', 'put_data', P + ':204-206')
    if o.wateruse:
        add('O', 'wateruse', 'put_data', P + ':207-209')
    if o.irrigation:
        add('O', 'irrigation', 'put_data', P + ':210-212')
    if o.wofost:
        add('O', 'wofost', 'put_data', P + ':213-215')
    if o.cropsyst:
        add('O', 'cropsyst', 'put_data', P + ':213-215 slot')
    add('O', 'balance', 'put_data', P + ':217 plugin_store_error')
    # VIC-WUR 3.0 MODFLOW: after the period's last step VIC-WUR has written
    # its outputs and state; MODFLOW runs between the monthly VIC runs
    # (plugins/modflow/vic_global_online_full.py:55-84)
    if o.groundwater:
        add('O', 'groundwater', 'put_data',
            'plugins/modflow/vic_global_online_full.py:55-84')
    return c


class PhaseScheduler(object):
    """Runs one model step in VIC-WUR's order over registered components.

    `levels` is the P2 order as a list of cell-index tensors (the routing
    levels; one cell per level is the serial routing_order).
    """

    def __init__(self, options, levels=()):
        self.o = options
        self.levels = list(levels)
        self.plan = plan(options)
        self.kinds = {}
        self.trace = None
        #: set by the caller while an outer snapshot (a VIC window's)
        #: restores every component and the view when anything raises:
        #: `between` then keeps no per-day snapshot of its own
        self.covered = False

    def register(self, kind, component):
        if kind in self.kinds:
            raise ValueError('component kind %r registered twice' % kind)
        self.kinds[kind] = component

    def _hook(self, call):
        comp = self.kinds.get(call.kind)
        fn = getattr(comp, call.hook.split(':')[-1], None) \
            if comp is not None else None
        return comp, fn

    def _call(self, call, *a):
        comp, fn = self._hook(call)
        if self.trace is not None:
            self.trace.append((call.phase, call.kind, call.hook) + (
                () if not a else tuple(a)))
        if fn is not None:
            return fn(*a)

    def run(self, vic_run, view=None):
        """The step on the staged state (no commit): F, U, V, P1, P2, P3, O.
        `vic_run()` is the VIC step (V)."""
        by = {}
        for c in self.plan:
            by.setdefault(c.phase, []).append(c)
        for ph in PHASES:
            calls = by.get(ph, [])
            if ph == 'V':
                if self.trace is not None:
                    self.trace.append(('V', 'vic', 'run'))
                vic_run()
            elif ph == 'P2':
                self._sweep(calls)
            else:
                for c in calls:
                    self._call(c)

    def _sweep(self, calls):
        slots = [c for c in calls if c.hook.startswith('sweep:')]
        if not slots:
            for c in calls:
                self._call(c)
            return
        comps = []
        for c in slots:
            comp = self.kinds.get(c.kind)
            if comp is not None and comp not in comps:
                comps.append(comp)
        for comp in comps:
            fn = getattr(comp, 'sweep_begin', None)
            if fn is not None:
                fn()
        key = self._graph_key(comps)
        if key is not None and self._sweep_graph is not None \
                and self._sweep_graph[0] == key:
            self._sweep_graph[1].replay()
        else:
            self._sweep_levels(slots)
            if key is not None:
                self._capture_sweep(key, slots)
        for comp in comps:
            fn = getattr(comp, 'sweep_end', None)
            if fn is not None:
                fn()

    def _sweep_levels(self, slots):
        # level by level, the slots in C order within a level
        # (plugin_simulation.c:124-140: one cell does all its slots before
        # the next cell starts; a level is a set of independent cells)
        for lv, cells in enumerate(self.levels):
            for c in slots:
                comp = self.kinds.get(c.kind)
                fn = getattr(comp, 'sweep_slot', None)
                slot = c.hook.split(':')[1]
                if self.trace is not None:
                    self.trace.append(('P2', c.kind, slot, lv))
                if fn is not None:
                    fn(lv, slot, cells)

    #: replay P2 from a CUDA graph when every slot component allows it
    #: (`graph_key`): the sweep's kernels and arguments are the eager
    #: sweep's, so its result is the same; none of its Python bookkeeping
    #: runs inside (`sweep_begin`, `sweep_end` run around it)
    SWEEP_GRAPH = True
    _sweep_graph = None

    def _graph_key(self, comps):
        from . import fused
        if not (self.SWEEP_GRAPH and fused.ENABLED and fused.GRAPH) or \
                self.trace is not None or not comps:
            return None
        key = []
        for comp in comps:
            fn = getattr(comp, 'graph_key', None)
            k = fn() if fn is not None else None
            if k is None:
                return None
            key.append(k)
        return (len(self.levels),) + tuple(key)

    def _capture_sweep(self, key, slots):
        import torch
        self._sweep_graph = None
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._sweep_levels(slots)
        self._sweep_graph = (key, g)

    # -- transactions ------------------------------------------------------
    def save_state(self):
        """Every registered component's state (those with save_state)."""
        return {k: c.save_state() for k, c in self.kinds.items()
                if hasattr(c, 'save_state')}

    def load_state(self, blob):
        for k, b in blob.items():
            self.kinds[k].load_state(b)

    def _set_time(self, t):
        if t is None:
            return
        if self.trace is not None:
            self.trace.append(('T', t))
        for c in self.kinds.values():
            fn = getattr(c, 'set_time', None)
            if fn is not None:
                fn(t)

    def _set_view(self, view):
        """Hand the step's writable view to the components that take it
        (`set_view`): the plugins' all_vars."""
        for c in self.kinds.values():
            fn = getattr(c, 'set_view', None)
            if fn is not None:
                fn(view)

    def _phases(self, phases):
        by = {}
        for c in self.plan:
            by.setdefault(c.phase, []).append(c)
        for ph in phases:
            calls = by.get(ph, [])
            if ph == 'P2':
                self._sweep(calls)
            else:
                for c in calls:
                    self._call(c)

    def _vic(self):
        v = self.kinds.get('vic')
        if v is None or not hasattr(v, 'save_state') \
                or not hasattr(v, 'load_state'):
            raise ValueError("step() needs a 'vic' component with "
                             "save_state/load_state (it is rolled back "
                             "with the others)")
        return v

    def step(self, vic_run=None, view=None, commit=None):
        """One whole transactional step F, U, V, P1, P2, P3, O. Needs a
        registered 'vic' component with save_state/load_state; `vic_run`
        defaults to its `run`. Snapshots every registered component (VIC
        included) and the view, runs the step, then calls `commit(view)`
        (default view.commit()) and drops the snapshots; on any exception
        restores them all (edits and base) and re-raises."""
        v = self._vic()
        if vic_run is None:
            vic_run = v.run
        snaps = self.save_state()
        vsnap = view.save_state() if view is not None else None
        try:
            self._set_view(view)
            self.run(vic_run, view)
            if commit is not None:
                commit(view)
            elif view is not None:
                view.commit()
        except BaseException:
            self.load_state(snaps)
            if view is not None:
                view.load_state(vsnap)
            raise

    def between(self, view, t_post=None, t_pre=None, post=True, pre=True):
        """The plugins between two VIC steps, for the day-coupled window
        (VicConnector's day hook; VIC itself has run already). With `view`
        the state at the end of day d: P1, P2, P3, O of day d (clock
        `t_post`), then F, U of day d+1 (clock `t_pre`); `post=False` at
        the run's first step (C runs no plugin_run before the first
        vic_run), `pre=False` after the run's last. Transactional like
        `step` (VIC is not touched here; the window's own rollback covers
        it). Returns the view's committed delta [cell, tile, band, layer],
        the soil_add of day d+1's first step. With `covered` (a VIC
        window's snapshot of the whole group, restored when the window is
        rolled back) no per-day snapshot is taken: a failure here fails
        the window, whose rollback restores the group."""
        if self.covered:
            return self._between(view, t_post, t_pre, post, pre)
        snaps = {k: b for k, b in self.save_state().items() if k != 'vic'}
        vsnap = view.save_state() if view is not None else None
        try:
            return self._between(view, t_post, t_pre, post, pre)
        except BaseException:
            self.load_state(snaps)
            if view is not None:
                view.load_state(vsnap)
            raise

    def _between(self, view, t_post, t_pre, post, pre):
        self._set_view(view)
        if post:
            self._set_time(t_post)
            self._phases(('P1', 'P2', 'P3', 'O'))
        if pre:
            self._set_time(t_pre)
            self._phases(('F', 'U'))
        return None if view is None else view.commit()
