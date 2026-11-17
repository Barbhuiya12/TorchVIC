"""The coupled runtime: one configuration, the connectors it names, and the
transactional step that advances them together.

    from torchvic import TorchVIC
    m = TorchVIC.from_config('configs/vic_only.yaml')
    m.initialize()
    m.run()                       # or m.step() interval by interval
    m.save_checkpoint('ck.pt')
    m.finalize()                  # history, ledger.csv, run_report.json
    out = m.outputs('vic')        # VIC's per-step outputs, as torchvic.run

A step (one base tick of the TimeManager) is a transaction:

  1. read accepted   every due connector receives its inputs from the
                     accepted exports of its producers (lag >= 1) or from
                     the proposals of producers earlier in the order (lag 0)
  2. compute         each returns a Proposal; the accepted state is not
                     touched
  3. validate        the connector's own checks, then the ledger: each
                     module's balance and the coupled domain's
  4. commit          all of them, or -- if anything raised, a check failed
                     or the balance did not close -- none: every connector,
                     buffer, ledger row and the clock stay where they were,
                     and StepRejected says why

Commit swaps references (the proposal's device tensors become the accepted
state); nothing is copied. Exchanged fields stay on the device; the ledger
brings its sums to the host once per check (ledger.check_every: 1 checks
before every commit; N > 1 batches the checks of N steps, which then detect
a violation at the check point but cannot roll back what was committed).
"""
import json
import os
import time

import torch

from . import registry, schema
from .connector import Context, ComputeError, problems_of, flags_of
from .hostfetch import Fetch, host
from .graph import DependencyGraph
from .ledger import Ledger, gather, resolve
from .managers import (BackendManager, CheckpointManager, ExchangeBuffer,
                       Mapping, OutputManager, SpatialMapper, StateManager,
                       TimeManager, config_digest)
from .spec import CouplingError, unit_dimension, unit_factor


def _device_save_state(c):
    """c.save_state() with every tensor it copies to the host copied on
    its device instead (Tensor.cpu -> clone while it runs): a snapshot that
    costs no host synchronisation. load_state takes it as it takes the
    host copy. Falls back to the plain save_state when that fails."""
    real = torch.Tensor.cpu

    def cpu(self, *a, **k):
        if self.device.type == 'cpu':
            return real(self, *a, **k)
        return self.clone()
    torch.Tensor.cpu = cpu
    try:
        return c.save_state()
    except Exception:                                   # noqa: BLE001
        torch.Tensor.cpu = real
        return c.save_state()
    finally:
        torch.Tensor.cpu = real


class StepRejected(CouplingError):
    """A step was not committed; `problems` lists why. Nothing changed."""

    def __init__(self, t, problems):
        self.t = t
        self.problems = list(problems)
        super().__init__('step at t = %g s rejected:\n  - %s'
                         % (t, '\n  - '.join(self.problems)))


def _resolve_device(dev):
    from ..config import resolve_device
    return resolve_device(dev or 'auto')


class Runtime(object):
    """`config`: a normalised configuration (schema.load / from_dict).
    `connectors`: {module name: Connector} to use instead of building them
    from the registry (an embedding application's own instances)."""

    def __init__(self, config, connectors=None, log=None):
        self.cfg = config
        run = config['run']
        self.log = log if log is not None else (
            print if run.get('log', True) else (lambda *a: None))
        self.device = torch.device(_resolve_device(run.get('device')))
        self.dtype = {'float64': torch.float64,
                      'float32': torch.float32}[run.get('dtype', 'float64')]
        self.given = dict(connectors or {})
        self.connectors = {}
        self.disabled = {}
        self.initialized = False
        self.finalized = False
        self.tick = 0
        led = config['ledger']
        self.ledger_on = bool(led.get('enabled', True))
        self.ledger = Ledger(factor=led.get('factor', 4.0),
                             on_violation=led.get('on_violation', 'reject'))
        self.check_every = int(led.get('check_every', 1) or 1)
        self._deferred = []
        #: the deferred checks of a speculative window (None outside one)
        self._spec = None
        self.spec_replays = 0
        self._spec_k = 1 << 30
        self.state = StateManager()
        self.backends = BackendManager(self.device,
                                       run.get('backend_fallback', 'error'))
        self.outputs_mgr = OutputManager(
            host=run.get('keep_outputs_on', 'device') == 'host',
            keep=run.get('keep_outputs_on', 'device') != 'none')
        self.checkpoints = CheckpointManager(config_digest(config))
        self.buffers = {}
        self.inner, self.inner_of, self._couplers = {}, {}, []
        self.timing = dict(compute_s=0.0, validate_s=0.0, ledger_s=0.0,
                           commit_s=0.0, steps=0)
        self._build()

    # -- construction (declarations only; nothing is loaded) ----------------
    def _build(self):
        for name, m in self.cfg['modules'].items():
            opts = {k: v for k, v in m.items()
                    if k not in ('connector', 'enabled')}
            if not m.get('enabled', True):
                e = registry.get(m.get('connector', name))
                self.disabled[name] = 'disabled in the configuration%s' % (
                    '' if e.factory else ' (%s)' % e.reason)
                continue
            if name in self.given:
                c = self.given[name]
            else:
                c = registry.get(m['connector']).make(**opts)
            c.name = name
            self.connectors[name] = c
        for k, why in self.disabled.items():
            self.log('module %s: %s' % (k, why))
        declared = {k: tuple(c.dependencies)
                    for k, c in self.connectors.items()}
        self.graph = DependencyGraph(list(self.connectors),
                                     self.cfg['connections'], declared)
        seen = {}
        water = {'depth', 'volume'}
        for c in self.cfg['connections']:
            p, var = c['from'].split('.')
            q, inp = c['to'].split('.')
            spec_o = self.connectors[p].info().output(var)
            spec_i = self.connectors[q].info().input(inp)
            if spec_o.kind != spec_i.kind:
                raise CouplingError('%s -> %s: a %s cannot feed a %s'
                                    % (c['from'], c['to'], spec_o.kind,
                                       spec_i.kind))
            if spec_o.kind == 'flux':
                # the same water cannot be delivered twice
                if c['from'] in seen:
                    raise CouplingError(
                        'flux %s feeds both %s and %s: water would be '
                        'delivered twice' % (c['from'], seen[c['from']],
                                             c['to']))
                seen[c['from']] = c['to']
            if not ({unit_dimension(spec_o.units),
                     unit_dimension(spec_i.units)} <= water):
                try:
                    unit_factor(spec_o.units, spec_i.units)
                except CouplingError as e:
                    raise CouplingError('%s -> %s: %s'
                                        % (c['from'], c['to'], e))
            if c.get('mapping') is None and spec_o.grid != spec_i.grid:
                raise CouplingError(
                    '%s -> %s: grid mismatch (%r -> %r) and no mapping'
                    % (c['from'], c['to'], spec_o.grid, spec_i.grid))
        for name, c in self.connectors.items():
            self.state.register(name, c.owned_state)
        self.wanted = {}
        for c in self.cfg['connections']:
            p, var = c['from'].split('.')
            self.wanted.setdefault(p, set()).add(var)

    def describe(self):
        """The plan, as text: modules, order, connections."""
        lines = ['modules (in order): %s' % ', '.join(self.graph.order)]
        for k, why in self.disabled.items():
            lines.append('  %s: not run (%s)' % (k, why))
        for k in self.graph.order:
            i = self.connectors[k].info()
            lines.append('  %s: %s v%s; backends %s; validation: %s'
                         % (k, type(self.connectors[k]).__name__, i.version,
                            '/'.join(i.backends), i.validation))
        for c in self.cfg['connections']:
            lines.append('  %s -> %s (lag %d%s)'
                         % (c['from'], c['to'], c.get('lag', 0),
                            ', mapping %s' % c['mapping']
                            if c.get('mapping') else ''))
        for comp, it in self.graph.iterative.items():
            lines.append('  iterate %s: %s' % (list(comp), it))
        return '\n'.join(lines)

    # -- life cycle --------------------------------------------------------
    def _context(self, name, t=0):
        return Context(t=float(t), t_end=float(getattr(self, 't_end', 0)),
                       step_index=self.tick,
                       device=self.backends.chosen.get(name, self.device),
                       dtype=self.dtype,
                       config=self.cfg['modules'].get(name, {}),
                       run=self.cfg['run'], log=self.log,
                       base_dir=os.path.dirname(self.cfg.get('_path', '.')),
                       redo=getattr(self, '_redo', False))

    def initialize(self):
        if self.initialized:
            return self
        if self.device.type == 'cuda':
            # every module may capture a CUDA graph (crop, groundwater, the
            # phase scheduler, reservoirs ...) inside torch.inference_mode,
            # not only the VIC Segmenter: the generator's graph state is made
            # outside it first (found by the ISC campaign: WOFOST's capture
            # poisoned the state before the Segmenter's own call)
            from ..core import graphsafe
            graphsafe.ensure_generator_graph_state()
        for name in self.graph.order:
            c = self.connectors[name]
            self.backends.choose(name, tuple(c.backends))
            if hasattr(c, 'peers'):
                c.peers = self.connectors     # connectors that share a step
            if getattr(c, 'wanted', None) is not None:
                c.wanted = set(c.wanted) | self.wanted.get(name, set())
            c.initialize(self._context(name))
            self.outputs_mgr.keep(name, getattr(c, 'keep_outputs', 'memory'))
            if hasattr(c, 'alone'):
                c.alone = len(self.connectors) == 1
        grids = {}
        for name, c in self.connectors.items():
            for g in c.grids():
                if g.name in grids and grids[g.name].n != g.n:
                    raise CouplingError('grid %r is declared with %d and %d '
                                        'elements' % (g.name, grids[g.name].n,
                                                      g.n))
                grids[g.name] = g
        maps = {mn: Mapping.from_config(mn, m, grids)
                for mn, m in self.cfg['mappings'].items()}
        self.mapper = SpatialMapper(grids, maps)
        for c in self.cfg['connections']:
            p, var = c['from'].split('.')
            q, inp = c['to'].split('.')
            so = self.connectors[p].info().output(var)
            si = self.connectors[q].info().input(inp)
            self.mapper.check(so.grid, si.grid, so.units, si.units,
                              c.get('mapping'), '%s -> %s' % (c['from'],
                                                              c['to']))
            if so.kind == 'flux':
                if self.ledger_on and not self.mapper.can_account(so.grid,
                                                                  so.units):
                    raise CouplingError(
                        '%s: %s on grid %r cannot be turned into a volume '
                        '(no areas), so the ledger cannot account it'
                        % (c['from'], so.units, so.grid))
                self.buffers[(c['from'], c['to'])] = ExchangeBuffer(
                    (c['from'], c['to']), profile=bool(
                        so.dims and so.dims[0] == 'day'))
        spans = {k: c.time_span() for k, c in self.connectors.items()}
        spans = {k: v for k, v in spans.items() if v is not None}
        if spans:
            ends = {k: v[1] for k, v in spans.items()}
            if len(set(round(x) for x in ends.values())) > 1:
                raise CouplingError('the connectors disagree on the run\'s '
                                    'length: %s' % ends)
            self.t_end = next(iter(ends.values()))
        else:
            self.t_end = float(self.cfg['run'].get('duration_days', 0)) \
                * 86400.0
        self._setup_day_coupling()
        intervals = {k: (float(c.interval_s) or self.t_end)
                     for k, c in self.connectors.items()
                     if k not in self.inner_of}
        self.time = TimeManager(intervals, self.t_end)
        for c in self.cfg['connections']:
            p = c['from'].split('.')[0]
            q = c['to'].split('.')[0]
            if p in self.inner_of or q in self.inner_of:
                continue
            so = self.connectors[p].info().output(c['from'].split('.')[1])
            if so.kind == 'flux' and c.get('lag', 0) == 0 and \
                    self.time.intervals[p] < self.time.intervals[q]:
                raise CouplingError(
                    '%s -> %s: the producer\'s interval (%d s) is shorter '
                    'than the consumer\'s (%d s), so at lag 0 the consumer '
                    'would need water not yet computed; give the connection '
                    'lag: 1' % (c['from'], c['to'], self.time.intervals[p],
                                self.time.intervals[q]))
        self.initialized = True
        ck = self.cfg['checkpoint'].get('restart_from')
        if ck:
            self.load_checkpoint(ck)
            if self.cfg['checkpoint'].get('w3_restart'):
                self.w3_restart()
        self.log('coupled run: %s; %d s in base steps of %d s'
                 % (', '.join('%s every %d s' % kv
                              for kv in self.time.intervals.items()),
                    self.time.t_end, self.time.base))
        return self

    def _setup_day_coupling(self):
        """Connectors a window connector runs between its days
        (`day_coupled: [names]` on it, e.g. VIC's 31-day windows with a
        daily irrigation model): checked here, scheduled by DayCoupler."""
        self.inner = {}
        self.inner_of = {}
        for name, c in self.connectors.items():
            names = list(getattr(c, 'day_coupled', ()) or ())
            if not names:
                continue
            for n in names:
                if n not in self.connectors:
                    raise CouplingError('%s.day_coupled names %r, which is '
                                        'not an enabled module' % (name, n))
                if n in self.inner_of:
                    raise CouplingError('%r is day-coupled to two modules'
                                        % n)
                iv = float(self.connectors[n].interval_s)
                if iv != 86400.0:
                    raise CouplingError('%s is day-coupled to %s but steps '
                                        'every %g s, not a day' % (n, name,
                                                                   iv))
                self.inner_of[n] = name
            self.inner[name] = [m for m in self.graph.order if m in names]
        for c in self.cfg['connections']:
            p, var = c['from'].split('.')
            q, inp = c['to'].split('.')
            lag = int(c.get('lag', 0))
            if q in self.inner_of and p == self.inner_of[q]:
                if var != 'soil_moisture' or lag != 1:
                    raise CouplingError(
                        '%s -> %s: a day-coupled module reads its window '
                        'module\'s soil_moisture at lag 1 (the state the day '
                        'starts from) only' % (c['from'], c['to']))
            elif p in self.inner_of and q == self.inner_of[p]:
                if lag != 0 or inp not in ('precip_add', 'soil_add'):
                    raise CouplingError(
                        '%s -> %s: a day-coupled module feeds precip_add or '
                        'soil_add at lag 0' % (c['from'], c['to']))
                self.buffers.pop((c['from'], c['to']), None)
            elif p in self.inner_of or q in self.inner_of:
                other = q if p in self.inner_of else p
                if self.inner_of.get(other) != self.inner_of.get(
                        p if other == q else q) or lag != 0:
                    raise CouplingError(
                        '%s -> %s: a day-coupled module exchanges with its '
                        'window module or with other modules day-coupled to '
                        'it (lag 0) only' % (c['from'], c['to']))
                self.buffers.pop((c['from'], c['to']), None)

    # -- the transaction -------------------------------------------------------
    def _inputs(self, name, t, dt, proposals, staged):
        """({input: value}, [delivered transfers]) for `name` over
        [t, t+dt)."""
        inputs = {}
        delivered = []
        for c in self.graph.upstream(name):
            p, var = c['from'].split('.')
            if self.inner_of.get(p) == name:
                continue            # delivered between the days (DayCoupler)
            inp = c['to'].split('.')[1]
            so = self.connectors[p].info().output(var)
            si = self.connectors[name].info().input(inp)
            lag = int(c.get('lag', 0))
            if so.kind == 'state':
                if lag == 0 and p in proposals:
                    v = proposals[p].exports.get(var)
                else:
                    v = self.state.get(p, var)
                if v is None:
                    continue
                v = self.mapper.apply(v, so.grid, si.grid, so.units,
                                      si.units, c.get('mapping'))
            else:
                key = (c['from'], c['to'])
                buf = staged.get(key)
                if buf is None:
                    buf = staged[key] = self.buffers[key].copy()
                    if p in proposals and var in proposals[p].exports:
                        pr = proposals[p]
                        buf.push(pr.t0, pr.t1, pr.exports[var])
                        buf.pushed = True
                a, b = t - lag * dt, t + dt - lag * dt
                if b <= 0:
                    continue
                cov = buf.covered_until()
                if (cov is None or cov < b - 1e-6) and lag == 0 and \
                        p not in proposals and self._same_cycle(p, name):
                    # first round of an iterated cycle: no guess yet
                    continue
                if cov is None or cov < b - 1e-6:
                    raise ComputeError(
                        '%s -> %s: water for [%g, %g) s is not computed yet '
                        '(the producer has reached %s s)'
                        % (c['from'], c['to'], a, b, cov))
                raw = buf.take(max(a, 0.0), b)
                if raw is None:
                    continue
                v = self.mapper.apply(raw, so.grid, si.grid, so.units,
                                      si.units, c.get('mapping'))
                if self.ledger_on:
                    tr = self.ledger.transfer(
                        p, name, var, 0.0, 'm3', si.grid, a, b,
                        internal=True)
                    tr['amount'] = self.mapper.volume(v, si.grid,
                                                      si.units).sum()
                    tr['leg'] = 'delivered'
                    delivered.append(tr)
            inputs[inp] = v if inp not in inputs else inputs[inp] + v
        return inputs, delivered

    def _same_cycle(self, a, b):
        return any(a in comp and b in comp for comp in self.graph.iterative)

    def _compute(self, name, t, proposals, staged, transfers):
        c = self.connectors[name]
        dt = self.time.dt(name, t)
        ctx = self._context(name, t)
        inputs, delivered = self._inputs(name, t, dt, proposals, staged)
        c.prepare_step(ctx, dt)
        if name in self.inner or getattr(c, 'phase_group', None) is not None:
            # also a window module with VIC-WUR's phase group and no
            # day-coupled module: the coupler carries its rollback
            cp = DayCoupler(self, name, t)
            self._couplers.append(cp)
            c.coupler = cp
            try:
                prop = c.compute(inputs, ctx, dt)
            finally:
                c.coupler = None
        else:
            prop = c.compute(inputs, ctx, dt)
        if prop is None:
            raise ComputeError('%s.compute returned nothing' % name)
        transfers.extend(delivered)
        return prop, ctx

    def step(self):
        """Advance one base tick. Returns False when the run is over;
        raises StepRejected (nothing committed) when the step fails."""
        if not self.initialized:
            self.initialize()
        if self.time.done:
            return False
        t = self.time.t
        due = set(self.time.due(t))
        for attempt in (0, 1):
            self._couplers = []
            # on the redo every connector takes its synchronous path
            # (Context.redo)
            self._redo = attempt == 1
            try:
                return self._step(t, due)
            except StepRejected as e:
                for cp in self._couplers:
                    cp.rollback()
                if attempt == 0 and getattr(e, 'retry', False):
                    # a connector's optimistic (no-host-read) result turned
                    # out unusable: the step committed nothing (no proposal,
                    # no ledger row, couplers rolled back), so run it again
                    self.log('step at t = %g s redone: %s'
                             % (t, '; '.join(e.problems)))
                    continue
                raise
            finally:
                self._couplers = []
                self._redo = False

    def _step(self, t, due):
        proposals, contexts, staged, transfers = {}, {}, {}, []
        problems = []
        t0 = time.time()
        try:
            for group in self.graph.groups:
                todo = [m for m in group if m in due]
                if not todo:
                    continue
                it = self.graph.iterative.get(tuple(group))
                if it is None:
                    for m in todo:
                        proposals[m], contexts[m] = self._compute(
                            m, t, proposals, staged, transfers)
                else:
                    self._iterate(todo, it, t, proposals, contexts, staged,
                                  transfers)
        except Exception as e:                         # noqa: BLE001
            raise StepRejected(t, ['%s: %s' % (type(e).__name__, e)]) \
                from e
        t1 = time.time()
        dflags = []
        for m, prop in proposals.items():
            try:
                res = self.connectors[m].validate(prop, contexts[m])
                problems += problems_of(res)
                dflags += flags_of(res) + list(prop.flags)
            except Exception as e:                     # noqa: BLE001
                problems.append('%s.validate raised %s: %s'
                                % (m, type(e).__name__, e))
        for cp in self._couplers:
            dflags += cp.flags
        t2 = time.time()
        rows, crow, gat = [], None, None
        spec = self._spec is not None
        defer = (self.check_every > 1 and self.ledger_on) or spec
        F = None if defer else Fetch()
        retry = False
        if defer and not problems and not spec:
            # a flag that asks for a redo cannot wait for the window (the
            # step would be committed): those are read now, in one copy
            rf = [f for f in dflags if f.retry is not None]
            if rf:
                dflags = [f for f in dflags if f.retry is None]
                problems, retry = self._raise_flags(rf, Fetch(), problems)
        if self.ledger_on and not problems:
            # what each producer sent into its flux connections this tick
            for (src, dst), buf in self.buffers.items():
                p, var = src.split('.')
                if p in proposals and var in proposals[p].exports:
                    so = self.connectors[p].info().output(var)
                    pr = proposals[p]
                    tr = self.ledger.transfer(p, dst.split('.')[0], var, 0.0,
                                              'm3', so.grid, pr.t0, pr.t1,
                                              internal=True)
                    tr['amount'] = self.mapper.volume(
                        pr.exports[var], so.grid, so.units).sum()
                    tr['leg'] = 'sent'
                    transfers.append(tr)
                    self._push(staged, (src, dst), buf, pr, var)
            pend = [(m, p.t0, p.t1, p.balance) for m, p in proposals.items()
                    if p.balance is not None]
            for cp in self._couplers:
                pend += cp.pend
                transfers += cp.transfers
            if spec:
                self._spec.append((pend, transfers, staged,
                                   dict(self.buffers), dflags))
            elif not defer:
                gat = self._gather(pend, transfers, staged, self.buffers, F)
            else:
                self._deferred.append((pend, transfers, staged,
                                       dict(self.buffers), dflags))
        elif not self.ledger_on:
            for (src, dst), buf in self.buffers.items():
                p, var = src.split('.')
                if p in proposals and var in proposals[p].exports:
                    self._push(staged, (src, dst), buf, proposals[p], var)
            if spec and not problems:
                self._spec.append((None, None, None, None, dflags))
        if not defer:
            # the step's only host copy: the connectors' device flags and
            # every ledger term together
            problems, retry = self._raise_flags(dflags, F, problems)
            if gat is not None and not problems:
                rows, pl, crow = self._finish(gat)
                problems += pl
        t3 = time.time()
        if problems:
            if self.ledger.on_violation == 'warn' and all(
                    'water balance' in x for x in problems):
                self.ledger.warnings.extend(problems)
                for x in problems:
                    self.log('ledger warning: %s' % x)
            else:
                e = StepRejected(t, problems)
                e.retry = retry
                raise e
        # ---- commit: all or nothing (every check is over) ----
        for m, prop in proposals.items():
            self.connectors[m].commit(prop, contexts[m])
            self.state.accept(m, prop.exports)
            self.backends.record(m, prop.backend or str(
                self.backends.chosen.get(m)))
            self.outputs_mgr.commit(m, prop)
        for k, b in staged.items():
            self.buffers[k] = b
        for cp in self._couplers:
            cp.commit_outputs()
        if not defer and self.ledger_on:
            self.ledger.accept(rows, crow, transfers)
        self.time.t = self.time.next_t()
        self.tick += 1
        if defer and not spec and (self.tick % self.check_every == 0
                                   or self.time.done):
            self.flush_ledger()
        t4 = time.time()
        tm = self.timing
        tm['compute_s'] += t1 - t0
        tm['validate_s'] += t2 - t1
        tm['ledger_s'] += t3 - t2
        tm['commit_s'] += t4 - t3
        tm['steps'] += 1
        every = int(self.cfg['checkpoint'].get('every') or 0)
        if every and self.tick % every == 0 and not spec:
            # (a speculative window ends here and saves it once checked)
            self.save_checkpoint(self.cfg['checkpoint']['path'])
        return True

    @staticmethod
    def _push(staged, key, buf, pr, var):
        """The producer's proposal into the staged copy of a buffer, once."""
        b = staged.get(key)
        if b is None:
            b = staged[key] = buf.copy()
        if not getattr(b, 'pushed', False):
            b.push(pr.t0, pr.t1, pr.exports[var])
            b.pushed = True

    @staticmethod
    def _raise_flags(dflags, F, problems):
        """Run the Fetch `F` with the device flags queued on it; returns
        (problems + the raised flags' texts, retry). retry: every raised
        flag asks for a redo and nothing else failed; their callbacks are
        called."""
        fh = [(f, F.add(f.vector())) for f in dflags]
        F.run()
        raised = [(f, f.text(h.value())) for f, h in fh]
        raised = [(f, x) for f, x in raised if x is not None]
        retry = bool(raised) and not problems \
            and all(f.retry is not None for f, x in raised)
        if retry:
            for f, x in raised:
                f.retry()
        return problems + [x for f, x in raised], retry

    def _buffer_volume_dev(self, bufs):
        """m3 still in the buffers, a device scalar (no host copy)."""
        tot = 0.0
        for (src, dst), b in bufs.items():
            p, var = src.split('.')
            so = self.connectors[p].info().output(var)
            tot = tot + b.content(lambda x, g=so.grid, u=so.units:
                                  self.mapper.volume(x, g, u))
        return tot

    def _buffer_volume(self, bufs):
        from .hostfetch import host
        return host(self._buffer_volume_dev(bufs))

    def _gather(self, pend, transfers, staged, buffers_before, F):
        """Queue everything one ledger check needs on the host (module
        terms, transfer amounts, the buffers' contents before and after)
        on the Fetch `F`; `_finish` completes the check after F.run()."""
        mods = [(m, a, b, gather(bal, F, self.ledger.factor))
                for m, a, b, bal in pend]
        trs = [(tr, F.add(tr['amount'])) for tr in transfers
               if torch.is_tensor(tr['amount'])]
        vol = None
        if buffers_before or len(mods) > 1:
            merged = dict(buffers_before)
            merged.update(staged)
            vol = (F.add(self._buffer_volume_dev(buffers_before)),
                   F.add(self._buffer_volume_dev(merged)))
        return mods, trs, vol, transfers

    def _finish(self, g):
        mods, trs, vol, transfers = g
        rows, problems = [], []
        for m, a, b, gg in mods:
            row, pr = self.ledger.check_module(m, a, b, None,
                                               resolved=resolve(gg))
            rows.append(row)
            problems += pr
        for tr, h in trs:
            tr['amount'] = h.value()
        crow = None
        if vol is not None:
            before, after = vol[0].value(), vol[1].value()
            # delivered water is measured at the receiver after mapping and
            # dS_buffers from the buffers' own contents before it: a mapping
            # that loses or creates water does not close
            t0 = min((r['t0'] for r in rows), default=0.0)
            t1 = max((r['t1'] for r in rows), default=0.0)
            crow, pr = self.ledger.check_coupled(
                t0, t1, rows, transfers, after - before,
                buffer_magnitude=before + after)
            problems += pr
        return rows, problems, crow

    def flush_ledger(self):
        """Run the deferred checks (ledger.check_every > 1): the ledger of
        every deferred step and the connectors' device flags, all brought
        to the host in ONE copy."""
        if not self._deferred:
            return
        F = Fetch()
        gs = [(self._gather(pend, transfers, staged, before, F),
               [(f, F.add(f.vector())) for f in dflags])
              for pend, transfers, staged, before, dflags in self._deferred]
        F.run()
        problems, hard = [], []
        for g, fh in gs:
            hard += [t for t in (f.text(h.value()) for f, h in fh)
                     if t is not None]
            rows, pr, crow = self._finish(g)
            problems += pr
            self.ledger.accept(rows, crow, g[3])
        self._deferred = []
        if (problems and self.ledger.on_violation == 'reject') or hard:
            raise CouplingError('deferred ledger check (already committed, '
                                'not rolled back):\n  - '
                                + '\n  - '.join(hard + problems))
        self.ledger.warnings.extend(problems)

    def _iterate(self, group, it, t, proposals, contexts, staged,
                 transfers):
        """A lag-0 cycle: compute its members in order until their exports
        change by at most `tol` (max abs) between rounds."""
        prev = None
        for _ in range(int(it['max_iter'])):
            local_tr = []
            for key in list(staged):
                if key[1].split('.')[0] in group or \
                        key[0].split('.')[0] in group:
                    del staged[key]
            for m in group:
                proposals[m], contexts[m] = self._compute(
                    m, t, proposals, staged, local_tr)
            cur = {m: {n: v for n, v in proposals[m].exports.items()
                       if torch.is_tensor(v)} for m in group}
            if prev is not None:
                ds = [(cur[m][n] - prev[m][n]).abs().max()
                      for m in cur for n in cur[m] if n in prev[m]]
                # one host copy for all members' differences
                diff = max(host(torch.stack(ds)), default=0.0) if ds else 0.0
                if diff <= float(it['tol']):
                    transfers.extend(local_tr)
                    return
            prev = cur
        raise ComputeError('iteration over %s did not converge in %d rounds'
                           % (group, it['max_iter']))

    def run(self, until=None):
        """Step to the end (or to `until` seconds)."""
        if not self.initialized:
            self.initialize()
        while not self.time.done and (until is None or self.time.t < until):
            if self._spec_ticks() > 1:
                self._window(until)
            else:
                self.step()
        return self

    # -- speculative windows (ledger.speculate) -----------------------------
    def _spec_ticks(self):
        """Steps a speculative window holds (<= 1: every step checked on
        its own). ledger.speculate: N steps; 'auto' (the default): 31 days
        of base ticks on a GPU, off on the CPU (where a check costs no
        device synchronisation) and with check_every > 1."""
        sp = self.cfg['ledger'].get('speculate', 'auto')
        if self.check_every > 1 or sp in (None, False, 0, 1):
            return 1
        if sp == 'auto':
            if self.device.type != 'cuda':
                return 1
            return max(int(round(31 * 86400.0 / self.time.base)), 1)
        return int(sp)

    def _window(self, until):
        """Up to _spec_ticks() steps with every check deferred -- the
        ledger, every device flag, also those that ask for a redo -- and
        read at the end in ONE host copy. When all pass, the window stands
        as committed (the same rows, in the same order, as checking each
        step). When one fails -- or a step raises -- everything goes back
        to the window's start (connectors, buffers, ledger, clock, kept
        outputs) and the steps run again checked one at a time, so the
        failure is met, redone or rejected exactly as without the window:
        a run's results do not depend on it."""
        K = min(self._spec_ticks(), self._spec_k)
        every = int(self.cfg['checkpoint'].get('every') or 0)
        snap = self._snapshot()
        n, bad, err = 0, None, None
        self._spec = []
        self.outputs_mgr.hold = []
        try:
            while n < K and not self.time.done and (
                    until is None or self.time.t < until):
                n += 1
                self.step()
                if every and self.tick % every == 0:
                    break
        except Exception as e:                          # noqa: BLE001
            # step n raised (and committed nothing): the steps before it
            # are checked now; when they pass, the run is where a checked
            # run is when step n raises, and so is the error
            err = e
        try:
            bad = self._spec_check()
        except Exception:                               # noqa: BLE001
            bad = n
        finally:
            self._spec = None
            held, self.outputs_mgr.hold = self.outputs_mgr.hold, None
        if bad is None:
            self.outputs_mgr.release(held)
            if err is not None:
                self._spec_k = max(self._spec_k // 4, 1)
                raise err
            if every and self.tick % every == 0:
                self.save_checkpoint(self.cfg['checkpoint']['path'])
            # a clean window: the next may be longer again
            self._spec_k = min(self._spec_k * 2, self._spec_ticks())
            return
        # back to the window's start, and its steps up to the first one
        # that failed again checked one at a time (which meets the failure
        # -- redone or rejected -- exactly as a checked run does); the
        # windows after it are shorter for a while
        self._restore(snap)
        self.spec_replays += 1
        self._spec_k = max(self._spec_k // 4, 1)
        for _ in range(bad):
            if self.time.done:
                break
            self.step()

    def _spec_check(self):
        """The window's deferred checks in one host copy: None when the
        window stands (its ledger rows then accepted), else the number of
        its steps up to the first that failed."""
        F = Fetch()
        gs = []
        for pend, transfers, staged, before, dflags in self._spec:
            g = (self._gather(pend, transfers, staged, before, F)
                 if pend is not None else None)
            gs.append((g, [(f, F.add(f.vector())) for f in dflags]))
        F.run()
        done = []
        for j, (g, fh) in enumerate(gs):
            if any(f.text(h.value()) is not None for f, h in fh):
                return j + 1
            if g is None:
                continue
            rows, pr, crow = self._finish(g)
            if pr and not (self.ledger.on_violation == 'warn' and all(
                    'water balance' in x for x in pr)):
                return j + 1
            done.append((rows, pr, crow, g[3]))
        for rows, pr, crow, trs in done:
            if pr:
                self.ledger.warnings.extend(pr)
                for x in pr:
                    self.log('ledger warning: %s' % x)
            self.ledger.accept(rows, crow, trs)
        return None

    def _snapshot(self):
        """What _restore needs to put the run back where it is: each
        connector's snapshot() (by reference, where the connector offers
        one) or its save_state() with the tensors copied on the device
        (no host copy), the buffers, the ledger's lengths, the accepted
        exports, the clock and the kept outputs' lengths."""
        conn = {}
        for k, c in self.connectors.items():
            f = getattr(c, 'snapshot', None)
            conn[k] = ('snap', f()) if f is not None else (
                'state', _device_save_state(c))
        L = self.ledger
        return dict(
            t=self.time.t, tick=self.tick, conn=conn,
            bufs={k: b.copy() for k, b in self.buffers.items()},
            led=(len(L.balances), len(L.coupled), len(L.transfers),
                 len(L.warnings)),
            accepted={k: dict(d) for k, d in self.state.accepted.items()},
            ran={k: dict(d) for k, d in self.backends.ran.items()},
            parts={k: len(v) for k, v in self.outputs_mgr.parts.items()})

    def _restore(self, s):
        for k, c in self.connectors.items():
            kind, blob = s['conn'][k]
            if kind == 'snap':
                c.restore(blob)
            else:
                c.load_state(blob)
        self.buffers = {k: b.copy() for k, b in s['bufs'].items()}
        L = self.ledger
        nb, nc, nt, nw = s['led']
        del L.balances[nb:]
        del L.coupled[nc:]
        del L.transfers[nt:]
        del L.warnings[nw:]
        self.state.accepted = {k: dict(d) for k, d in s['accepted'].items()}
        self.backends.ran = {k: dict(d) for k, d in s['ran'].items()}
        parts = self.outputs_mgr.parts
        for k in list(parts):
            if k not in s['parts']:
                del parts[k]
            else:
                del parts[k][s['parts'][k]:]
        self.time.t = s['t']
        self.tick = s['tick']

    # -- checkpoints --------------------------------------------------------
    def save_checkpoint(self, path=None):
        path = path or self.cfg['checkpoint'].get('path')
        if not path:
            raise CouplingError('save_checkpoint: no path (give one or set '
                                'checkpoint.path)')
        if self._deferred:
            self.flush_ledger()
        return self.checkpoints.save(path, self.time.t, self.tick,
                                     self.connectors, self.buffers,
                                     self.ledger, self.state)

    def load_checkpoint(self, path):
        if not self.initialized:
            self.initialize()
        blob = self.checkpoints.load(path)
        if set(blob['connectors']) != set(self.connectors):
            raise CouplingError('%s holds modules %s, this run has %s'
                                % (path, sorted(blob['connectors']),
                                   sorted(self.connectors)))
        for k, c in self.connectors.items():
            c.load_state(blob['connectors'][k])
        for k, entries in blob['buffers'].items():
            key = tuple(k.split('|'))
            p = key[0].split('.')[0]
            self.buffers[key].load(entries, self.backends.chosen.get(p))
        self.ledger.load(blob['ledger'])
        self.state.accepted = {
            k: {n: (v.to(self.backends.chosen.get(k, self.device))
                    if torch.is_tensor(v) else v) for n, v in d.items()}
            for k, d in blob.get('accepted', {}).items()}
        self.time.t = int(blob['t'])
        self.tick = int(blob['tick'])
        self.log('restarted from %s at t = %d s' % (path, self.time.t))
        return self

    def w3_restart(self):
        """`checkpoint.w3_restart`: after the load, every connector with a
        `w3_restart(context, t)` hook keeps what VIC-WUR 3.0.0's state file
        holds and resets the rest to what a new run started at t has
        (docs/connectors/vicwur.md); the others keep their
        checkpoint whole."""
        t = self.time.t
        done = []
        for name in self.graph.order:
            fn = getattr(self.connectors[name], 'w3_restart', None)
            if fn is not None:
                fn(self._context(name, t), t)
                done.append(name)
        self.log('w3_restart at t = %d s: %s' % (t, ', '.join(done)
                                                  or 'no connector'))
        return self

    # -- the end --------------------------------------------------------------
    def finalize(self):
        if self.finalized:
            return self
        if self._deferred:
            self.flush_ledger()
        for name in self.graph.order:
            self.connectors[name].finalize(self._context(name, self.time.t))
        self.finalized = True
        return self

    def report(self):
        return dict(t=self.time.t, t_end=self.time.t_end, tick=self.tick,
                    backends=self.backends.report(), timing=dict(self.timing),
                    ledger={m: self.ledger.totals(m)
                            for m in self.connectors},
                    coupled_rows=len(self.ledger.coupled),
                    warnings=list(self.ledger.warnings))


class DayCoupler(object):
    """Runs the modules day-coupled to a window connector (VIC) between
    the window's days, from inside its one simulate call (basin's
    day_hook). Each day: the day-coupled modules compute in order from the
    window module's state at the day's start and from each other, validate
    and commit at once (the next day starts from them); their water for the
    window module comes back as its inputs for that day. Their states are
    saved before the window and restored if the step is rejected, so the
    transaction still commits all or nothing."""

    def __init__(self, rt, outer, t0):
        self.rt, self.outer, self.t0 = rt, outer, float(t0)
        self.flags = []
        self.names = rt.inner.get(outer, [])
        self.snap = {n: rt.connectors[n].save_state() for n in self.names}
        self.pend, self.transfers, self.done = [], [], []
        #: callables the window module registers to undo its own staged
        #: work (VIC-WUR's phase group) when the step is rejected
        self.undo = []

    def day(self, d, soil_moisture, soil=None):
        rt = self.rt
        t = self.t0 + d * 86400.0
        props = {}
        for name in self.names:
            c = rt.connectors[name]
            inputs = {}
            for conn in rt.graph.upstream(name):
                p, var = conn['from'].split('.')
                inp = conn['to'].split('.')[1]
                so = rt.connectors[p].info().output(var)
                si = c.info().input(inp)
                if p == self.outer:
                    v = soil_moisture
                elif p in props:
                    v = props[p].exports.get(var)
                else:
                    v = rt.state.get(p, var)
                if v is None:
                    continue
                v = rt.mapper.apply(v, so.grid, si.grid, so.units, si.units,
                                    conn.get('mapping'))
                inputs[inp] = v if inp not in inputs else inputs[inp] + v
            ctx = rt._context(name, t)
            if soil is not None:
                ctx.soil = soil
            c.prepare_step(ctx, 86400.0)
            prop = c.compute(inputs, ctx, 86400.0)
            res = c.validate(prop, ctx)
            pr = problems_of(res)
            self.flags += flags_of(res) + list(prop.flags)
            if pr:
                raise ComputeError('; '.join(pr))
            c.commit(prop, ctx)
            if rt.outputs_mgr.drops(name):
                # nobody reads the day's series: not held to the window end
                prop.series = None
            props[name] = prop
            self.done.append((name, prop))
            if prop.balance is not None:
                self.pend.append((name, prop.t0, prop.t1, prop.balance))
        vin = {}
        for conn in rt.graph.upstream(self.outer):
            p, var = conn['from'].split('.')
            if p not in props:
                continue
            inp = conn['to'].split('.')[1]
            so = rt.connectors[p].info().output(var)
            si = rt.connectors[self.outer].info().input(inp)
            v = props[p].exports.get(var)
            if v is None:
                continue
            v = rt.mapper.apply(v, so.grid, si.grid, so.units, si.units,
                                conn.get('mapping'))
            vin[inp] = v if inp not in vin else vin[inp] + v
            if rt.ledger_on:
                vol = rt.mapper.volume(v, si.grid, si.units).sum()
                for leg in ('sent', 'delivered'):
                    tr = rt.ledger.transfer(p, self.outer, var, 0.0, 'm3',
                                            si.grid, t, t + 86400.0,
                                            internal=True)
                    tr['amount'] = vol
                    tr['leg'] = leg
                    self.transfers.append(tr)
        return vin

    def rollback(self):
        for n, blob in self.snap.items():
            self.rt.connectors[n].load_state(blob)
        for fn in reversed(self.undo):
            fn()
        self.undo = []

    def commit_outputs(self):
        rt = self.rt
        for name, prop in self.done:
            rt.state.accept(name, prop.exports)
            rt.backends.record(name, prop.backend or str(
                rt.backends.chosen.get(name)))
            rt.outputs_mgr.commit(name, prop)


class TorchVIC(object):
    """The application: a Runtime from a configuration file, with VIC's
    outputs written as `torchvic run` writes them."""

    def __init__(self, config, connectors=None, log=None):
        self.cfg = config
        self.runtime = Runtime(config, connectors=connectors, log=log)

    @classmethod
    def from_config(cls, path, log=None, connectors=None, **run_overrides):
        cfg = schema.load(path)
        for k, v in run_overrides.items():
            if v is not None:
                cfg['run'][k] = v
        return cls(cfg, connectors=connectors, log=log)

    @classmethod
    def from_dict(cls, raw, base_dir='.', connectors=None, log=None):
        return cls(schema.from_dict(raw, base_dir), connectors=connectors,
                   log=log)

    def initialize(self):
        self.runtime.initialize()
        return self

    def step(self):
        return self.runtime.step()

    def run(self, until=None):
        self.runtime.run(until)
        return self

    def save_checkpoint(self, path=None):
        return self.runtime.save_checkpoint(path)

    def load_checkpoint(self, path):
        self.runtime.load_checkpoint(path)
        return self

    @property
    def ledger(self):
        return self.runtime.ledger

    def connector(self, name):
        return self.runtime.connectors[name]

    def outputs(self, name='vic'):
        """The connector's committed per-step outputs joined over its
        intervals (for VIC: what Case.run(return_state=True) returns)."""
        from .vic import FINAL_KEYS, VicConnector
        c = self.runtime.connectors[name]
        fk = FINAL_KEYS if isinstance(c, VicConnector) else ()
        out = self.runtime.outputs_mgr.series(name, fk)
        if isinstance(c, VicConnector) and c.state is not None:
            out['state'] = c.state
        return out

    def finalize(self, write=True):
        """Finish the connectors; with `write`, VIC's history streams and
        the ledger go to output.dir. Returns the paths written."""
        self.runtime.finalize()
        paths = []
        if not write:
            return paths
        o = self.cfg['output']
        from .vic import VicConnector, CaseDriver
        for name, c in self.runtime.connectors.items():
            if isinstance(c, VicConnector) and o.get('history', True) and \
                    hasattr(c.driver, 'write_history'):
                # a streamed case: its streams were aggregated per window
                from ..io import history
                rd = history.resolve_result_dir(c.driver.gp, o.get('dir'),
                                                log=self.runtime.log)
                paths += c.write_history(rd, log=self.runtime.log)
                continue
            if isinstance(c, VicConnector) and o.get('history', True) and \
                    isinstance(c.driver, CaseDriver):
                case = c.driver.case
                from ..io import history
                streams = history.parse_streams(case.gp,
                                                log=lambda *a: None)
                if streams:
                    rd = history.resolve_result_dir(case.gp, o.get('dir'),
                                                    log=self.runtime.log)
                    out = self.outputs(name)
                    out.pop('state', None)
                    paths += case.write_history(out, result_dir=rd,
                                                streams=streams,
                                                log=self.runtime.log)
        if o.get('ledger', True) and o.get('dir') and \
                self.runtime.ledger_on:
            os.makedirs(o['dir'], exist_ok=True)
            paths.append(self.runtime.ledger.write_csv(
                os.path.join(o['dir'], 'ledger.csv')))
            p = os.path.join(o['dir'], 'run_report.json')
            with open(p, 'w') as fh:
                json.dump(self.runtime.report(), fh, indent=1, default=str)
            paths.append(p)
        return paths

    def report(self):
        return self.runtime.report()


def validate_config(path, deep=False, log=None):
    """[problems] of a configuration: the schema, the registry, the
    dependency graph and the units always; with `deep`, the connectors are
    initialised too (data read), which checks grids, mappings and spans."""
    try:
        cfg = schema.load(path)
    except schema.ConfigError as e:
        return e.problems
    try:
        rt = Runtime(cfg, log=log or (lambda *a: None))
        if deep:
            rt.initialize()
    except CouplingError as e:
        return [str(e)]
    return []
