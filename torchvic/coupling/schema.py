"""The coupled-run configuration file (YAML or JSON) and its validation.

    version: 1
    run:
      device: auto            # auto | cpu | cuda | cuda:N
      dtype: float64          # float64 | float32
      backend_fallback: error # error | cpu  (a connector without GPU code)
      duration_days: 0        # only when no connector defines the run's span
      keep_outputs_on: device # device | host | none: where the kept
                              # output series live (host: copied off the GPU
                              # at each commit, so a long run's history does
                              # not fill device memory; none: not kept, and
                              # a day-coupled module's daily series dropped
                              # as soon as it is committed)
    modules:
      vic:
        connector: vic        # registry name (default: the module's name)
        enabled: true
        global_param: path/to/global_param.txt
        core: 5.1.0
        window_days: 31       # 0: the whole run in one call
        exchange: window      # window | whole | day | step
        stream: auto          # auto | true | false: read the forcing a
                              # window at a time (coupling/vic_stream.py)
      routing:
        enabled: false        # listed, not run
    connections:              # producer output -> consumer input
      - from: vic.total_runoff
        to: routing.inflow
        lag: 0                # consumer intervals of delay (>= 0)
        mapping: cells_to_reaches
        iterate: {max_iter: 10, tol: 1.0e-9}   # only inside a lag-0 cycle
    mappings:
      cells_to_reaches:
        source: vic_cells
        target: reaches
        pairs: [[0, 0, 1.0], [1, 0, 0.5], [1, 1, 0.5]]  # source, target,
                                                       # fraction of source
        # or file: weights.csv (columns source,target,fraction)
    ledger:
      enabled: true
      factor: 4               # tolerance = factor * eps * n * sum|terms|
      on_violation: reject    # reject | warn
      check_every: 1          # 1: check before every commit; N: batch the
                              # checks of N steps (no rollback, fewer syncs)
      speculate: auto         # steps checked together, rolled back and run
                              # again step by step when a check fails (the
                              # results of check_every 1); auto: 31 days on a
                              # GPU, off on the CPU; 0: off
    checkpoint:
      path: null              # where save_checkpoint writes by default
      every: 0                # write one every N steps (0: never)
      restart_from: null      # start from this checkpoint
      w3_restart: false       # VIC-WUR exact: on restart keep only what
                              # VIC-WUR 3.0's state file holds, reset the
                              # rest as a new run started there
    output:
      dir: results            # history, ledger.csv, ledger.json
      history: true           # the VIC global file's OUTFILE streams
      ledger: true

${VAR} in a string is replaced from the environment (an unset variable is an
error); relative paths are taken from the configuration file's directory.
`validate()` returns every problem found, not only the first.
"""
import json
import os
import re

from .spec import CouplingError


class ConfigError(CouplingError):
    def __init__(self, problems, path=None):
        self.problems = list(problems)
        head = 'invalid coupling configuration%s' % (
            ' %s' % path if path else '')
        super().__init__(head + ':\n  - ' + '\n  - '.join(self.problems))


TOP = {'version', 'run', 'modules', 'connections', 'mappings', 'ledger',
       'checkpoint', 'output'}
RUN = {'device', 'dtype', 'backend_fallback', 'duration_days', 'log',
       'keep_outputs_on'}
LEDGER = {'enabled', 'factor', 'on_violation', 'check_every', 'speculate'}
CHECKPOINT = {'path', 'every', 'restart_from', 'w3_restart'}
OUTPUT = {'dir', 'history', 'ledger'}
CONNECTION = {'from', 'to', 'lag', 'mapping', 'iterate'}
MAPPING = {'source', 'target', 'pairs', 'file', 'rest_to'}
#: keys whose string values are paths, resolved against the file's directory
PATH_KEYS = {'global_param', 'dir', 'path', 'restart_from', 'file',
             'fdir', 'domain', 'gauges', 'dams', 'params_file'}

_VAR = re.compile(r'\$\{([^}]+)\}')


def _expand(x, problems, where):
    if isinstance(x, str):
        def rep(m):
            v = os.environ.get(m.group(1))
            if v is None:
                problems.append('%s: environment variable %s is not set'
                                % (where, m.group(1)))
                return m.group(0)
            return v
        return _VAR.sub(rep, x)
    if isinstance(x, dict):
        return {k: _expand(v, problems, '%s.%s' % (where, k))
                for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v, problems, '%s[%d]' % (where, i))
                for i, v in enumerate(x)]
    return x


def _paths(x, base):
    if isinstance(x, dict):
        return {k: (os.path.normpath(os.path.join(base, v))
                    if k in PATH_KEYS and isinstance(v, str) and v
                    and not os.path.isabs(v) else _paths(v, base))
                for k, v in x.items()}
    if isinstance(x, list):
        return [_paths(v, base) for v in x]
    return x


def read(path):
    with open(path) as fh:
        text = fh.read()
    if path.endswith('.json'):
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def normalise(raw, base_dir='.'):
    """(config with defaults filled, env vars expanded and paths resolved,
    [problems])."""
    problems = []
    if not isinstance(raw, dict):
        return {}, ['the file is not a mapping of sections']
    cfg = _expand(raw, problems, 'config')
    cfg = _paths(cfg, base_dir)
    problems += validate(cfg)
    full = dict(version=cfg.get('version', 1))
    full['run'] = dict(dict(device='auto', dtype='float64',
                            backend_fallback='error', duration_days=0,
                            log=True), **(cfg.get('run') or {}))
    full['modules'] = {k: dict(v or {}) for k, v in
                       (cfg.get('modules') or {}).items()}
    for k, v in full['modules'].items():
        v.setdefault('connector', k)
        v.setdefault('enabled', True)
    full['connections'] = [dict(dict(lag=0, mapping=None, iterate=None), **c)
                           for c in (cfg.get('connections') or [])
                           if isinstance(c, dict)]
    full['mappings'] = dict(cfg.get('mappings') or {})
    full['ledger'] = dict(dict(enabled=True, factor=4.0,
                               on_violation='reject'),
                          **(cfg.get('ledger') or {}))
    full['checkpoint'] = dict(dict(path=None, every=0, restart_from=None,
                                   w3_restart=False),
                              **(cfg.get('checkpoint') or {}))
    full['output'] = dict(dict(dir=None, history=True, ledger=True),
                          **(cfg.get('output') or {}))
    return full, problems


def _keys(d, allowed, where, problems):
    if d is None:
        return
    if not isinstance(d, dict):
        problems.append('%s must be a mapping' % where)
        return
    for k in d:
        if k not in allowed:
            problems.append('%s: unknown key %r (allowed: %s)'
                            % (where, k, ', '.join(sorted(allowed))))


def validate(cfg):
    """Every structural problem of a configuration (before any connector is
    built): unknown keys, wrong types, values out of range, connections
    naming modules that are not there."""
    from . import registry
    p = []
    _keys(cfg, TOP, 'config', p)
    if cfg.get('version', 1) != 1:
        p.append('version: only version 1 is understood, not %r'
                 % cfg.get('version'))
    run = cfg.get('run') or {}
    _keys(run, RUN, 'run', p)
    if isinstance(run, dict):
        dev = str(run.get('device', 'auto'))
        if not (dev in ('auto', 'cpu', 'cuda') or re.match(r'^cuda:\d+$',
                                                           dev)):
            p.append('run.device: %r is not auto, cpu, cuda or cuda:N' % dev)
        if run.get('dtype', 'float64') not in ('float64', 'float32'):
            p.append('run.dtype: %r is not float64 or float32'
                     % run.get('dtype'))
        if run.get('backend_fallback', 'error') not in ('error', 'cpu'):
            p.append('run.backend_fallback must be error or cpu')
        if run.get('keep_outputs_on', 'device') not in ('device', 'host',
                                                         'none'):
            p.append('run.keep_outputs_on must be device, host or none')
        dd = run.get('duration_days', 0)
        if not isinstance(dd, (int, float)) or dd < 0:
            p.append('run.duration_days must be a number >= 0')
    mods = cfg.get('modules')
    if not isinstance(mods, dict) or not mods:
        p.append('modules: at least one module is required')
        mods = {}
    for name, m in mods.items():
        if m is None:
            m = {}
        if not isinstance(m, dict):
            p.append('modules.%s must be a mapping' % name)
            continue
        if not isinstance(m.get('enabled', True), bool):
            p.append('modules.%s.enabled must be true or false' % name)
        conn = m.get('connector', name)
        try:
            e = registry.get(conn)
        except CouplingError as err:
            p.append('modules.%s: %s' % (name, err))
            continue
        if m.get('enabled', True) and e.factory is None:
            p.append('modules.%s: connector %r is %s: %s (set enabled: '
                     'false to keep it listed)' % (name, conn, e.status,
                                                   e.reason))
        if conn == 'vic' and m.get('enabled', True):
            gp = m.get('global_param')
            if not gp:
                p.append('modules.%s.global_param is required' % name)
            elif not os.path.isfile(gp):
                p.append('modules.%s.global_param: %s does not exist'
                         % (name, gp))
            wd = m.get('window_days', 31)
            if not isinstance(wd, int) or wd < 0:
                p.append('modules.%s.window_days must be an integer >= 0'
                         % name)
            ex = m.get('exchange')
            if ex is not None and ex not in ('window', 'whole', 'day',
                                             'step'):
                p.append('modules.%s.exchange must be window, whole, day or '
                         'step' % name)
            sm = m.get('stream', 'auto')
            if sm not in (True, False, 'auto'):
                p.append('modules.%s.stream must be auto, true or false'
                         % name)
    enabled = {k for k, m in mods.items()
               if isinstance(m, dict) and m.get('enabled', True)
               or m is None}
    maps = cfg.get('mappings') or {}
    if not isinstance(maps, dict):
        p.append('mappings must be a mapping')
        maps = {}
    for mn, m in maps.items():
        _keys(m, MAPPING, 'mappings.%s' % mn, p)
        if isinstance(m, dict):
            for k in ('source', 'target'):
                if not m.get(k):
                    p.append('mappings.%s.%s is required' % (mn, k))
            if ('pairs' in m) == ('file' in m):
                p.append('mappings.%s: give exactly one of pairs and file'
                         % mn)
            if 'file' in m and not os.path.isfile(str(m['file'])):
                p.append('mappings.%s.file: %s does not exist'
                         % (mn, m['file']))
    conns = cfg.get('connections') or []
    if not isinstance(conns, list):
        p.append('connections must be a list')
        conns = []
    for i, c in enumerate(conns):
        w = 'connections[%d]' % i
        if not isinstance(c, dict):
            p.append('%s must be a mapping' % w)
            continue
        _keys(c, CONNECTION, w, p)
        for side in ('from', 'to'):
            v = c.get(side)
            if not isinstance(v, str) or v.count('.') != 1:
                p.append('%s.%s must be "module.variable", not %r'
                         % (w, side, v))
                continue
            mod = v.split('.')[0]
            if mod not in mods:
                p.append('%s.%s: no module %r' % (w, side, mod))
            elif mod not in enabled:
                p.append('%s.%s: module %r is disabled' % (w, side, mod))
        lag = c.get('lag', 0)
        if not isinstance(lag, int) or lag < 0:
            p.append('%s.lag must be an integer >= 0' % w)
        if c.get('mapping') is not None and c['mapping'] not in maps:
            p.append('%s.mapping: no mapping %r' % (w, c['mapping']))
        it = c.get('iterate')
        if it is not None:
            if not isinstance(it, dict) or set(it) - {'max_iter', 'tol'}:
                p.append('%s.iterate must be {max_iter: N, tol: x}' % w)
    led = cfg.get('ledger') or {}
    _keys(led, LEDGER, 'ledger', p)
    if isinstance(led, dict):
        if led.get('on_violation', 'reject') not in ('reject', 'warn'):
            p.append('ledger.on_violation must be reject or warn')
        ce = led.get('check_every', 1)
        if not isinstance(ce, int) or ce < 1:
            p.append('ledger.check_every must be an integer >= 1')
        sp = led.get('speculate', 'auto')
        if sp != 'auto' and (not isinstance(sp, int) or isinstance(sp, bool)
                             or sp < 0):
            p.append('ledger.speculate must be auto or an integer >= 0')
        f = led.get('factor', 4.0)
        if not isinstance(f, (int, float)) or f <= 0:
            p.append('ledger.factor must be a positive number')
    ck = cfg.get('checkpoint') or {}
    _keys(ck, CHECKPOINT, 'checkpoint', p)
    if isinstance(ck, dict):
        ev = ck.get('every', 0)
        if not isinstance(ev, int) or ev < 0:
            p.append('checkpoint.every must be an integer >= 0')
        if ck.get('restart_from') and not os.path.isfile(ck['restart_from']):
            p.append('checkpoint.restart_from: %s does not exist'
                     % ck['restart_from'])
        if ev and not ck.get('path'):
            p.append('checkpoint.every needs checkpoint.path')
        w3 = ck.get('w3_restart', False)
        if not isinstance(w3, bool):
            p.append('checkpoint.w3_restart must be true or false')
        elif w3 and not ck.get('restart_from'):
            p.append('checkpoint.w3_restart needs checkpoint.restart_from')
    _keys(cfg.get('output') or {}, OUTPUT, 'output', p)
    return p


def load(path):
    """The normalised configuration of `path`; ConfigError listing every
    problem when it is not valid."""
    if not os.path.isfile(path):
        raise ConfigError(['%s does not exist' % path], path)
    try:
        raw = read(path)
    except Exception as e:                               # noqa: BLE001
        raise ConfigError(['cannot parse: %s' % e], path)
    cfg, problems = normalise(raw, os.path.dirname(os.path.abspath(path)))
    if problems:
        raise ConfigError(problems, path)
    cfg['_path'] = os.path.abspath(path)
    return cfg


def from_dict(raw, base_dir='.'):
    cfg, problems = normalise(raw, base_dir)
    if problems:
        raise ConfigError(problems)
    return cfg
