"""Carrying basin.simulate's loop state from one run into the next:
export_state and restore_state, time_window, and warm_up.
"""
import torch


# ---------------------------------------------------------------------------
# Carrying the state across runs: warm-up, then a scored window.
#
# A window's first months are only right if the soil, the pack and the canopy
# entered them the way a long run would have left them. `warm_up` runs the
# spin-up under torch.no_grad() and the window starts from the state it left.
#
# VIC 5 carries a run into the next with its state file
# (write_model_state.c, read_initial_model_state.c). What is carried here is
# the port's own loop state, which is more than VIC's file: it also holds the
# totals put_data.c keeps in save_data for the next step's DEL* terms and
# water error, and the routing tail RVIC keeps in its ring buffer.
# ---------------------------------------------------------------------------

#: Bumped whenever the set of carried quantities changes, so a state from an
#: older version is refused instead of silently missing a new store.
STATE_VERSION = 3

#: The loop's carried quantities.
#:   moist            soil moisture per tile and layer, mm ([cell, layer]
#:                    without a mosaic)
#:   snow             the pack per tile: ice, surf_water, pack_water (m),
#:                    surf_temp, pack_temp (C), depth (m), density (kg/m3),
#:                    cold_content (snow->coldcontent, J/m2; version 2)
#:   last_snow        snow age in steps since the last snowfall
#:   melting          VIC's MELTING flag (bool)
#:   spatial_snow     SPATIAL_SNOW's six stores, None with the option off
#:   wdew             canopy liquid water per tile, mm
#:   canopy           canopy snow per tile (int_snow, t_foliage,
#:                    temp_int_storage) where a covered overstory exists
#:   ground_long_out  the ground's emission the next canopy step reads
#:   t_node           the soil thermal nodes per tile, 3 or NODES (energy)
#:   soil_ice         FROZEN_SOIL's layer ice, mm
#:   wb_t             water-balance mode's T[0] and quick-flux T1 per tile
#:   carbon           the three soil carbon pools and the annual NPP pair
#:   snow_zero        the snow-free fast path's zero stand-in for snow_step:
#:                    the next run then takes the same path on its first
#:                    snow-free step as the single run does
#:   ce_seed          CLOSE_ENERGY's per-tile (snow_flux, grnd_flux, deltaH,
#:                    fusion) in VIC's internal signs (version 3)
#: Lakes are not part of basin.simulate (torchvic.core.lake runs on its own), so
#: there is no lake state here.
CARRIED = ('moist', 'snow', 'last_snow', 'melting', 'spatial_snow', 'wdew',
           'canopy', 'ground_long_out', 't_node', 'soil_ice', 'wb_t',
           'carbon', 'snow_zero', 'ce_seed')


def _copy_tree(x):
    """Dicts copied (the loop assigns into them), tensors shared."""
    if isinstance(x, dict):
        return {k: _copy_tree(v) for k, v in x.items()}
    return x


def map_state(state, fn):
    """`state` with `fn` applied to every tensor in it, e.g.
    `map_state(s, lambda t: t.to('cuda:1'))` to move it."""
    if torch.is_tensor(state):
        return fn(state)
    if isinstance(state, dict):
        return {k: map_state(v, fn) for k, v in state.items()}
    return state


def detach_state(state):
    """The same state cut from the graph that produced it: a run started
    from it has no gradient back into the run that left it."""
    return map_state(state, lambda t: t.detach())


def export_state(carry, report, routing_prefix, steps_per_day, last_date,
                 last_yday):
    """What simulate(return_state=True) returns as out['state']."""
    missing = set(CARRIED) - set(carry)
    if missing:
        raise ValueError('export_state: %s not given' % sorted(missing))
    st = {k: _copy_tree(carry[k]) for k in CARRIED}
    st.update(version=STATE_VERSION, steps_per_day=int(steps_per_day),
              last_date=last_date, last_yday=int(last_yday),
              report=dict(report), routing_prefix=routing_prefix)
    return st


def _match(name, given, cold):
    """`given` from a state, checked against what this run's cold start
    built in its place."""
    if given is None and cold is None:
        return None
    if given is None or cold is None:
        raise ValueError(
            'initial_state[%r]: %s. The state was exported from a run with '
            'a different configuration (energy balance, overstory, '
            'SPATIAL_SNOW, FROZEN_SOIL, CARBON or mosaic) than this one.'
            % (name, 'this run carries it and the state has none'
               if given is None else 'the state carries it and this run '
               'has no use for it'))
    if isinstance(cold, dict):
        if not isinstance(given, dict) or set(given) != set(cold):
            raise ValueError(
                'initial_state[%r]: keys %s, this run carries %s'
                % (name, sorted(given) if isinstance(given, dict)
                   else type(given).__name__, sorted(cold)))
        return {k: _match(name + '.' + k, given[k], cold[k]) for k in cold}
    if not torch.is_tensor(given):
        raise ValueError('initial_state[%r] is a %s, not a tensor'
                         % (name, type(given).__name__))
    if (tuple(given.shape) != tuple(cold.shape) or given.dtype != cold.dtype
            or given.device != cold.device):
        raise ValueError(
            'initial_state[%r] is %s %s on %s; this run needs %s %s on %s '
            '(map_state moves a state between devices)'
            % (name, tuple(given.shape), given.dtype, given.device,
               tuple(cold.shape), cold.dtype, cold.device))
    return given


def restore_state(state, cold, steps_per_day, frozen_soil=False, n_frost=1):
    """The carried quantities for simulate's warm start: `state` checked
    against `cold` (the same names, as this run's cold start built them),
    with its dicts copied so the loop never writes into the caller's
    state."""
    if not isinstance(state, dict) or state.get('version') != STATE_VERSION:
        raise ValueError('initial_state is not a basin.simulate state of '
                         'version %d (got version %r)'
                         % (STATE_VERSION, state.get('version')
                            if isinstance(state, dict) else None))
    # Snow age is counted in steps (solve_snow.c:311-327) and so is the
    # routing tail, so a state belongs to runs at its own step.
    if int(state['steps_per_day']) != int(steps_per_day):
        raise ValueError('initial_state was exported at %d steps a day and '
                         'this run takes %d: the snow age and the routing '
                         'tail are counted in steps'
                         % (state['steps_per_day'], steps_per_day))
    out = {}
    for k in CARRIED:
        g = state.get(k)
        if k == 'ground_long_out':
            # None until an energy-balance step with a canopy has run.
            if g is not None:
                _match(k, g, None if cold['canopy'] is None
                       else cold['canopy']['int_snow'])
            out[k] = g
        elif k == 'soil_ice':
            # None until the first FROZEN_SOIL step has run (with more
            # than one frost area the cold start already holds the
            # initial ice, [..., n_frost]).
            if g is not None:
                _want = cold['moist'] if frozen_soil else None
                if _want is not None and n_frost > 1:
                    _want = _want.unsqueeze(-1).expand(*_want.shape, n_frost)
                _match(k, g, _want)
            out[k] = g
        elif k == 'snow_zero':
            # Empty until the first snow-free step has run.
            g = {} if g is None else g
            if g and (tuple(g['melt'].shape)
                      != tuple(cold['snow']['ice'].shape)):
                raise ValueError(
                    "initial_state['snow_zero'] is for packs of shape %s; "
                    'this run carries %s' % (tuple(g['melt'].shape),
                                             tuple(cold['snow']['ice'].shape)))
            out[k] = _copy_tree(g)
        else:
            out[k] = _copy_tree(_match(k, g, cold[k]))
    out['report'] = dict(state['report'])
    out['last_yday'] = int(state['last_yday'])
    return out


def _batch_window(batch, t0, t1):
    """`batch` over steps [t0, t1) only: everything simulate reads per step
    is sliced, the per-cell attributes are shared."""
    import copy
    nd = len(batch.dates)
    new = copy.copy(batch)
    new.dates = batch.dates[t0:t1]
    for name in ('day_of_year', 'sw_potential', 'discharge_mm', 'et',
                 'et_mask'):
        v = getattr(batch, name, None)
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == nd:
            setattr(new, name, v[t0:t1])
    if isinstance(getattr(batch, 'forcing', None), dict):
        new.forcing = {k: v[t0:t1] for k, v in batch.forcing.items()}
    if hasattr(batch, 'n_days'):
        new.n_days = len(new.dates)
    if hasattr(batch, 't0'):
        new.t0 = batch.t0 + t0
        new.t1 = new.t0 + len(new.dates)
    return new


#: the VegMosaic fields that are [cell, tile, step] under a vegetation
#: history (basin._STEP_FIELDS, torchvic.io.veghist.apply)
STEP_FIELDS = ('lai', 'fcanopy', 'albedo', 'displacement', 'veg_rough')


def time_window(kwargs, t0, t1=None):
    """simulate's keyword arguments with `batch` and `atm` cut to model
    steps [t0, t1) (t1 None: to the end)."""
    kw = dict(kwargs)
    nt = int(kw['atm']['tair'].shape[0])
    t1 = nt if t1 is None else int(t1)
    if not 0 <= t0 < t1 <= nt:
        raise ValueError('time_window [%d, %d) is outside the run\'s %d '
                         'steps' % (t0, t1, nt))
    kw['batch'] = _batch_window(kw['batch'], t0, t1)
    kw['atm'] = {k: (v[t0:t1] if torch.is_tensor(v) and v.dim() >= 1
                     and v.shape[0] == nt else v)
                 for k, v in kw['atm'].items()}
    m = kw.get('mosaic')
    if m is not None and getattr(m, 'step_axis', False):
        # a vegetation history (FORCING2) is one value per model step on
        # the mosaic's last axis, indexed by the window's own step: the
        # window takes its steps of it
        import copy
        m = copy.copy(m)
        for name in STEP_FIELDS:
            v = getattr(m, name, None)
            if torch.is_tensor(v) and v.dim() == 3 and v.shape[2] == nt:
                setattr(m, name, v[:, :, t0:t1])
        kw['mosaic'] = m
    sd = kw.get('subdaily')
    if isinstance(sd, dict):
        # the snow sub-steps behind a daily step, [day, snow step, cell]
        kw['subdaily'] = {k: (v[t0:t1] if torch.is_tensor(v) and v.dim() >= 1
                              and v.shape[0] == nt else v)
                          for k, v in sd.items()}
    return kw


def warm_up(kwargs, ndays=None, n_steps=None, exact=False, **extra):
    """Spin the model up over the first `ndays` (or `n_steps`) of the run
    `kwargs` describes -- simulate's keyword arguments, e.g. Case.kwargs()
    -- and return (state, window_kwargs): the state it left, detached, and
    the arguments for the rest of the run started from it.

        state, win = basin.warm_up(c.kwargs(p=p), ndays=180)
        out = basin.simulate(**win)       # the graph starts at day 181

    By default the warm-up runs under torch.no_grad(), which is the point:
    no graph is built for steps nothing is learned from. That forward is
    not the grad-mode forward to the last bit wherever an energy balance is
    solved: implicit.root_with_grad (implicit.py:318) returns the bracketed
    root itself under no_grad and the root plus one Newton correction with
    grad on, the same number to solver tolerance. Measured on the synthetic
    snow case of tests/test_state_carry.py, the window's outputs
    then differ from the single grad-mode run by at most 1e-12 mm (SWE
    9.6e-13, runoff 1.3e-13). The water-balance path solves no energy
    balance and stays bit-identical.

    `exact=True` runs the warm-up with grad on and detaches its state
    afterwards: the window is then the single run's bit for bit, and the
    warm-up's graph is built and freed, so peak memory is the larger of the
    two graphs rather than their sum. Either way the window's gradient no
    longer includes how the parameters shaped the warm-up.
    """
    if (ndays is None) == (n_steps is None):
        raise ValueError('give exactly one of ndays and n_steps')
    n = (int(n_steps) if n_steps is not None
         else int(ndays) * int(kwargs.get('steps_per_day', 1)))
    head = time_window(kwargs, 0, n)
    head.pop('return_state', None)
    from .basin import simulate
    with torch.set_grad_enabled(bool(exact)):
        out = simulate(**head, return_state=True, **extra)
    state = detach_state(out['state'])
    del out
    win = time_window(kwargs, n)
    win['initial_state'] = state
    # The state carries the routing tail; a prefix given for the warm-up's
    # own start would be the wrong history for the window.
    win.pop('routing_prefix', None)
    return state, win


# ---------------------------------------------------------------------------
# Starting from a VIC state file (INIT_STATE, vic_restore.c).
# ---------------------------------------------------------------------------
def from_vic_file(fs, cold, cv, frozen_soil=False, n_frost=1, any_over=False):
    """The carried quantities for simulate's warm start from a VIC image
    state file (`fs`, torchvic.io.state_file.initial_state_from_file),
    laid over this run's own cold start `cold` (the same names
    restore_state takes): what the file holds replaces it, tile by tile
    where the cell has the tile; everything vic_restore.c does not read
    keeps its cold value, which for the stores VIC allocates with calloc
    (canopy temp_int_storage, the carried advection and refreeze, the
    closure seeds) is zero in both. Returns restore_state's dict, with the
    report rows the first DEL* terms are formed against (initialize_save_
    data forms them from the restored state) and `vic_aux`, the file-only
    stores torchvic.io.state_file writes back."""
    live = fs['live']
    nc = cv.shape[0]
    why = []
    if why:
        raise ValueError('INIT_STATE is not covered for: ' + '; '.join(why))

    def lay(file_t, cold_t):
        m = live.reshape(live.shape + (1,) * (file_t.dim() - 2))
        if tuple(file_t.shape) != tuple(cold_t.shape):
            raise ValueError('the state file holds %s where this run needs '
                             '%s' % (tuple(file_t.shape),
                                     tuple(cold_t.shape)))
        return torch.where(m, file_t.to(cold_t.dtype), cold_t)

    out = {}
    out['moist'] = lay(fs['moist'], cold['moist'])
    sn = cold['snow']
    swq, sw, pw = fs['swq'], fs['surf_water'], fs['pack_water']
    dens = fs['density']
    snow = dict(
        ice=lay(swq - sw - pw, sn['ice']),
        surf_water=lay(sw, sn['surf_water']),
        pack_water=lay(pw, sn['pack_water']),
        surf_temp=lay(fs['surf_temp'], sn['surf_temp']),
        pack_temp=lay(fs['pack_temp'], sn['pack_temp']),
        density=lay(dens, sn['density']),
        cold_content=lay(fs['cold_content'], sn['cold_content']))
    # compute_derived_state_vars.c:78-86
    if 'depth' in sn:
        snow['depth'] = lay(torch.where(
            dens > 0.0, 1000.0 * swq / torch.where(dens > 0.0, dens,
                                                   torch.ones_like(dens)),
            sn['depth']), sn['depth'])
    out['snow'] = snow
    out['last_snow'] = lay(fs['last_snow'], cold['last_snow'])
    out['melting'] = torch.where(live, fs['melting'], cold['melting'])
    if cold['spatial_snow'] is not None:
        # vic_restore.c reads snow->coverage only; the other five stores
        # (store_swq, store_coverage, max_snow_depth, snow_distrib_slope,
        # store_snow) keep what the cold start gives them, as VIC's
        # allocation does
        sp = dict(cold['spatial_snow'])
        sp['coverage'] = lay(fs['coverage'], sp['coverage'])
        for k, v in (fs.get('sp_extra') or {}).items():
            # TorchVIC's own file: the stores VIC's file lacks
            sp[k] = torch.where(live, v, sp[k])
        out['spatial_snow'] = sp
    else:
        out['spatial_snow'] = None
    out['wdew'] = lay(fs['wdew'], cold['wdew'])
    if cold['canopy'] is not None:
        cs = dict(cold['canopy'])
        cs['int_snow'] = lay(fs['snow_canopy'], cs['int_snow'])
        cs['t_foliage'] = lay(fs['t_foliage'], cs['t_foliage'])
        out['canopy'] = cs
    else:
        out['canopy'] = None
    out['ground_long_out'] = (lay(fs['long_under_out'],
                                  cold['ground_long_out'])
                              if cold['ground_long_out'] is not None
                              else None)
    if cold['t_node'] is not None:
        # QUICK_FLUX: the file holds NODES nodes of which three are used
        fs = dict(fs, t_node=fs['t_node'][..., :cold['t_node'].shape[-1]])
    out['t_node'] = (lay(fs['t_node'], cold['t_node'])
                     if cold['t_node'] is not None else None)
    if frozen_soil:
        ice = fs['soil_ice']
        if n_frost == 1:
            ice = ice[..., 0]
        # the cold start holds no ice until the first FROZEN_SOIL step has
        # run (basin.simulate); the file's is the state either way
        out['soil_ice'] = lay(ice, cold['soil_ice'] if cold['soil_ice']
                              is not None else torch.zeros_like(ice))
    else:
        out['soil_ice'] = None
    if cold['wb_t'] is not None:
        # FULL_ENERGY FALSE: T[0] and T[1] are the surface and the second
        # node (calc_surf_energy_bal.c:574-578)
        tn = fs['t_node']
        out['wb_t'] = dict(t0=lay(tn[..., 0], cold['wb_t']['t0']),
                           t1=lay(tn[..., 1], cold['wb_t']['t1']))
    else:
        out['wb_t'] = None
    if cold['carbon'] is not None:
        out['carbon'] = {k: lay(fs['carbon'][k], v)
                         for k, v in cold['carbon'].items()}
    else:
        out['carbon'] = None
    out['snow_zero'] = _copy_tree(fs.get('snow_zero_extra')
                                  or cold['snow_zero'] or {})
    if cold['ce_seed'] is not None:
        ce = torch.zeros_like(cold['ce_seed'])
        ce[..., 0] = torch.where(live, fs['snow_flux'], ce[..., 0])
        out['ce_seed'] = ce
    else:
        out['ce_seed'] = None
    # the report rows (basin.simulate's `opening`): cell totals of the
    # restored stores, formed as the steps form theirs
    cvl = cv.unsqueeze(-1)
    tm = (out['moist'] * cvl).sum(1).sum(-1)
    swe = ((snow['ice'] + snow['surf_water'] + snow['pack_water']) * cv
           ).sum(1) * 1000.0
    can = (((out['canopy']['int_snow'] * cv).sum(1) * 1000.0)
           if out['canopy'] is not None else torch.zeros_like(tm))
    wd = (out['wdew'] * cv).sum(1)
    out['report'] = dict(total_moist=tm, swe_canopy=swe + can, wdew=wd,
                         storage=tm + swe + wd + can)
    # the file already holds the year turnover of a 31 December state
    out['last_yday'] = 1
    out['vic_aux'] = dict(
        snow_flux=fs['snow_flux'], long_under_out=fs['long_under_out'],
        cold_content=fs['cold_content'], tfoliage_bare=fs['t_foliage'])
    return out
