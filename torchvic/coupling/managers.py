"""The runtime's managers: time, state ownership, spatial mapping, exchange
buffers, backends, outputs and checkpoints. Each is small; the Runtime
(runtime.py) composes them.
"""
import csv
import datetime as _dt
import hashlib
import json
import math
import os

import torch

from .spec import CouplingError, unit_dimension, unit_factor


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
class TimeManager(object):
    """Multi-rate clock in whole seconds from the run's start. Each
    connector has an exchange interval; the runtime advances by the greatest
    common divisor of them (the base tick), and a connector is due at a tick
    on which one of its intervals starts. The last interval of a connector
    is cut at the run's end."""

    def __init__(self, intervals, t_end):
        self.intervals = {}
        for k, v in intervals.items():
            iv = int(round(v))
            if iv <= 0 or abs(iv - v) > 1e-6:
                raise CouplingError('%s: exchange interval %r s is not a '
                                    'positive whole number of seconds'
                                    % (k, v))
            self.intervals[k] = iv
        self.t_end = int(round(t_end))
        if self.t_end <= 0:
            raise CouplingError('the run has no length (t_end = %r s)'
                                % t_end)
        g = 0
        for v in self.intervals.values():
            g = math.gcd(g, v)
        self.base = g or self.t_end
        self.t = 0

    def due(self, t=None):
        t = self.t if t is None else t
        return [k for k, v in self.intervals.items() if t % v == 0]

    def dt(self, name, t=None):
        t = self.t if t is None else t
        return min(self.intervals[name], self.t_end - t)

    def next_t(self):
        return min(self.t + self.base, self.t_end)

    @property
    def done(self):
        return self.t >= self.t_end


# ---------------------------------------------------------------------------
# State ownership
# ---------------------------------------------------------------------------
class StateManager(object):
    """Who owns which store, and each connector's last accepted exports.
    Two connectors may not own the same store: a store has one writer."""

    def __init__(self):
        self.owner = {}
        self.accepted = {}

    def register(self, connector, names, shared=()):
        for n in names:
            if n in shared:
                continue
            if n in self.owner and self.owner[n] != connector:
                raise CouplingError(
                    'state %r is owned by %s and %s: a store has one '
                    'writer (declare it in one connector, exchange it as a '
                    'flux)' % (n, self.owner[n], connector))
            self.owner[n] = connector

    def accept(self, connector, exports):
        if exports:
            d = self.accepted.setdefault(connector, {})
            d.update(exports)

    def get(self, connector, name):
        return self.accepted.get(connector, {}).get(name)


# ---------------------------------------------------------------------------
# Spatial mapping
# ---------------------------------------------------------------------------
class Mapping(object):
    """A conservative mapping from the elements of one grid to another:
    `pairs` (source, target, fraction): the fraction of source element's
    water that goes to target element. Each source element's fractions sum
    to 1 (checked), so the total volume is preserved exactly up to rounding.
    Depths are converted with the grids' areas (m2):

        volume_t = sum_s f_st * depth_s * area_s
        depth_t  = volume_t / area_t
    """

    def __init__(self, name, source, target, pairs, n_source, n_target):
        self.name, self.source, self.target = name, source, target
        self.n_source, self.n_target = int(n_source), int(n_target)
        s = torch.tensor([int(p[0]) for p in pairs], dtype=torch.long)
        t = torch.tensor([int(p[1]) for p in pairs], dtype=torch.long)
        f = torch.tensor([float(p[2]) for p in pairs], dtype=torch.float64)
        if len(pairs) == 0:
            raise CouplingError('mapping %s has no pairs' % name)
        if int(s.min()) < 0 or int(s.max()) >= self.n_source:
            raise CouplingError('mapping %s: source index outside 0..%d'
                                % (name, self.n_source - 1))
        if int(t.min()) < 0 or int(t.max()) >= self.n_target:
            raise CouplingError('mapping %s: target index outside 0..%d'
                                % (name, self.n_target - 1))
        if bool((f < 0).any()):
            raise CouplingError('mapping %s: negative fraction' % name)
        tot = torch.zeros(self.n_source, dtype=torch.float64).index_add_(
            0, s, f)
        bad = (tot - 1.0).abs() > 1e-12
        if bool(bad.any()):
            i = int(bad.nonzero()[0, 0])
            raise CouplingError(
                'mapping %s is not conservative: the fractions of source '
                'element %d sum to %.15g, not 1' % (name, i, float(tot[i])))
        self.s, self.t, self.f = s, t, f

    @classmethod
    def from_config(cls, name, m, grids):
        src, dst = m['source'], m['target']
        for g in (src, dst):
            if g not in grids:
                raise CouplingError('mapping %s: no grid %r (grids: %s)'
                                    % (name, g, ', '.join(sorted(grids))))
        if 'pairs' in m:
            pairs = m['pairs']
        else:
            with open(m['file']) as fh:
                rows = list(csv.DictReader(fh))
            pairs = [(r['source'], r['target'], r['fraction'])
                     for r in rows]
        if m.get('rest_to') is not None:
            # every source element the pairs leave out goes whole to one
            # target element, so the mapping stays conservative: a sparse
            # link on a big grid, e.g. routing.withdrawn on all river sites
            # of which only the flood's entry sites ever carry water
            k = int(m['rest_to'])
            seen = set(int(p[0]) for p in pairs)
            pairs = list(pairs) + [(i, k, 1.0) for i in range(grids[src].n)
                                   if i not in seen]
        return cls(name, src, dst, pairs, grids[src].n, grids[dst].n)

    def volumes(self, vol_s):
        out = torch.zeros(self.n_target, dtype=vol_s.dtype,
                          device=vol_s.device)
        # the index/weight tensors live on the device after the first call
        # (no host-to-device copy, hence no stream sync, per exchange)
        c = self.__dict__.setdefault('_dev', {})
        k = (vol_s.device, vol_s.dtype)
        if k not in c:
            c[k] = (self.s.to(vol_s.device), self.t.to(vol_s.device),
                    self.f.to(vol_s.device, vol_s.dtype))
        s_, t_, f_ = c[k]
        return out.index_add_(0, t_, vol_s[s_] * f_)


class SpatialMapper(object):
    """Moves an exchanged amount from the producer's grid and units to the
    consumer's. Same grid: units only (depth <-> volume with the grid's
    area). Different grids: a declared Mapping, else a grid mismatch."""

    def __init__(self, grids, mappings=None):
        self.grids = dict(grids)
        self.mappings = dict(mappings or {})

    def check(self, src_grid, dst_grid, src_units, dst_units, mapping=None,
              where=''):
        """Raise CouplingError for a connection that cannot be made."""
        ds, dd = unit_dimension(src_units), unit_dimension(dst_units)
        water = {'depth', 'volume'}
        if src_units != dst_units and not ({ds, dd} <= water):
            unit_factor(src_units, dst_units)       # raises on mismatch
        if src_grid != dst_grid:
            if mapping is None:
                raise CouplingError(
                    '%sgrid mismatch: %s lives on %r, %s on %r, and the '
                    'connection names no mapping' % (
                        where + ': ' if where else '', 'the output',
                        src_grid, 'the input', dst_grid))
            m = self.mappings.get(mapping)
            if m is None:
                raise CouplingError('%sno mapping %r' % (
                    where + ': ' if where else '', mapping))
            if (m.source, m.target) != (src_grid, dst_grid):
                raise CouplingError(
                    '%sgrid mismatch: mapping %s goes %s -> %s, the '
                    'connection %s -> %s' % (
                        where + ': ' if where else '', mapping, m.source,
                        m.target, src_grid, dst_grid))
        if ds != dd and {ds, dd} <= water:
            for g in {src_grid, dst_grid}:
                if self.grids.get(g) is None or self.grids[g].area is None:
                    raise CouplingError(
                        '%s%s -> %s needs the areas of grid %r, which it '
                        'does not declare' % (where + ': ' if where else '',
                                              src_units, dst_units, g))
        if src_grid == dst_grid and src_grid in self.grids and \
                dst_grid in self.grids and \
                self.grids[src_grid].n != self.grids[dst_grid].n:
            raise CouplingError('%sgrid %r has two sizes' % (
                where + ': ' if where else '', src_grid))

    def volume(self, x, grid, units):
        """`x` in `units` on `grid` as m3 per element (water quantities
        only)."""
        dim = unit_dimension(units)
        if dim == 'volume':
            return x.to(torch.float64) * unit_factor(units, 'm3')
        if dim == 'depth':
            a = self.grids[grid].area_tensor(device=x.device)
            return x.to(torch.float64) * unit_factor(units, 'm') * a
        raise CouplingError('%s is not a water amount; it cannot be '
                            'accounted' % units)

    def can_account(self, grid, units):
        dim = unit_dimension(units)
        if dim == 'volume':
            return True
        return dim == 'depth' and self.grids.get(grid) is not None and \
            self.grids[grid].area is not None

    def apply(self, x, src_grid, dst_grid, src_units, dst_units,
              mapping=None, dtype=None):
        dim_s, dim_d = unit_dimension(src_units), unit_dimension(dst_units)
        if src_grid == dst_grid and dim_s == dim_d:
            y = x * unit_factor(src_units, dst_units) \
                if src_units != dst_units else x
            return y if dtype is None else y.to(dtype)
        if dim_s in ('depth', 'volume') and dim_d in ('depth', 'volume'):
            vol = self.volume(x, src_grid, src_units)
            if src_grid != dst_grid:
                vol = self.mappings[mapping].volumes(vol)
            if dim_d == 'volume':
                y = vol * unit_factor('m3', dst_units)
            else:
                a = self.grids[dst_grid].area_tensor(device=vol.device)
                y = vol / a * unit_factor('m', dst_units)
            return y if dtype is None else y.to(dtype)
        if src_grid != dst_grid:
            # a state on another grid: area-weighted mean over the sources
            m = self.mappings[mapping]
            a = self.grids[src_grid].area_tensor(device=x.device)
            num = m.volumes(x.to(torch.float64) * a)
            den = m.volumes(a)
            y = num / torch.where(den > 0, den, torch.ones_like(den))
            y = y * unit_factor(src_units, dst_units)
            return y if dtype is None else y.to(dtype)
        return x * unit_factor(src_units, dst_units)


# ---------------------------------------------------------------------------
# Exchange buffers (flux connections)
# ---------------------------------------------------------------------------
def _f64(x):
    return x.to(torch.float64) if torch.is_tensor(x) else float(x)


class ExchangeBuffer(object):
    """The amounts one flux connection has received from its producer and
    not yet delivered, as [t0, t1, amount, delivered-until] entries; an
    amount is taken uniformly in time (a consumer interval that covers part
    of a producer interval gets that part's share). Water in a buffer is in
    transit between two modules and is a store of the coupled domain."""

    def __init__(self, key, profile=False):
        self.key = key
        self.entries = []
        #: the amounts are [row, ...] with equal rows spread over the
        #: entry's interval (VIC's daily series): a consumer interval takes
        #: the rows it covers, not a uniform share of the total
        self.profile = bool(profile)

    @staticmethod
    def _rows(amt, t0, t1, lo, hi):
        """Sum of the rows of `amt` over [lo, hi) within [t0, t1): a row
        wholly inside counts as it is, a row cut by the bounds in
        proportion."""
        n = amt.shape[0]
        h = (t1 - t0) / n
        out = None
        for i in range(n):
            a, b = t0 + i * h, t0 + (i + 1) * h
            o = min(b, hi) - max(a, lo)
            if o <= 1e-9 * h:
                continue
            r = amt[i] if o >= h * (1 - 1e-9) else amt[i] * (o / h)
            out = r if out is None else out + r
        return out

    def copy(self):
        b = ExchangeBuffer(self.key, self.profile)
        b.entries = [list(e) for e in self.entries]
        return b

    def push(self, t0, t1, amount):
        self.entries.append([float(t0), float(t1), amount, float(t0)])

    def covered_until(self):
        return max((e[1] for e in self.entries), default=None)

    def take(self, a, b):
        """The amount over [a, b) (summed over entries, prorated), and
        marks it delivered. Needs [a, b) inside what was pushed."""
        total = None
        for e in self.entries:
            t0, t1, amt, done = e
            lo, hi = max(a, done, t0), min(b, t1)
            if hi <= lo:
                continue
            part = (self._rows(amt, t0, t1, lo, hi) if self.profile
                    else amt * ((hi - lo) / (t1 - t0)))
            if part is None:
                continue
            total = part if total is None else total + part
            e[3] = max(done, hi)
        self.entries = [e for e in self.entries if e[3] < e[1]]
        return total

    def content(self, to_volume):
        """m3 still in the buffer (to_volume: amount -> m3 per element); a
        device scalar when the amounts are on the device (no host copy)."""
        v = 0.0
        for t0, t1, amt, done in self.entries:
            if self.profile:
                rest = self._rows(amt, t0, t1, done, t1)
                if rest is not None:
                    v = v + _f64(to_volume(rest).sum())
                continue
            frac = (t1 - done) / (t1 - t0)
            # float64 as main's float(): a float32 sum is widened before
            # the fraction and the accumulation
            v = v + _f64(to_volume(amt).sum()) * frac
        return v

    def state(self):
        return [[e[0], e[1], e[2].cpu() if torch.is_tensor(e[2]) else e[2],
                 e[3]] for e in self.entries]

    def load(self, entries, device=None):
        self.entries = [[e[0], e[1], e[2].to(device) if device is not None
                         and torch.is_tensor(e[2]) else e[2], e[3]]
                        for e in entries]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class BackendManager(object):
    """Chooses each connector's device from the run's and the connector's
    declared backends, and records which backend actually ran each
    interval."""

    def __init__(self, device, fallback='error'):
        self.device = torch.device(device)
        self.fallback = fallback
        self.chosen = {}
        self.ran = {}

    def choose(self, name, backends):
        want = self.device.type
        if want in backends:
            self.chosen[name] = self.device
        elif self.fallback == 'cpu' and 'cpu' in backends:
            self.chosen[name] = torch.device('cpu')
        else:
            raise CouplingError(
                '%s runs on %s only; the run asks for %s (set '
                'run.backend_fallback: cpu to run it on the CPU)'
                % (name, ', '.join(backends), want))
        return self.chosen[name]

    def record(self, name, backend):
        d = self.ran.setdefault(name, {})
        d[backend] = d.get(backend, 0) + 1

    def report(self):
        return {k: dict(device=str(self.chosen.get(k)),
                        ran=dict(self.ran.get(k, {})))
                for k in self.chosen}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def concat_series(parts, final_keys=()):
    """Window outputs ({name: [steps, ...] tensor}) joined along time;
    `final_keys` and anything that is not per step are taken from the last
    window (the run's end state); dicts of counts are summed."""
    if not parts:
        return {}
    if len(parts) == 1:
        return dict(parts[0])
    out = {}
    for k, v in parts[-1].items():
        if k in final_keys:
            out[k] = v
        elif torch.is_tensor(v) and v.dim() >= 1 and all(
                torch.is_tensor(p.get(k)) for p in parts):
            n = [int(p[k].shape[0]) for p in parts]
            out[k] = torch.cat([p[k] for p in parts], 0) if all(
                int(p[k].shape[1:] == v.shape[1:]) for p in parts) else v
            del n
        elif isinstance(v, dict) and v and all(
                isinstance(x, (int, float)) for x in v.values()):
            s = {}
            for p in parts:
                for kk, vv in (p.get(k) or {}).items():
                    s[kk] = s.get(kk, 0) + vv
            out[k] = s
        else:
            out[k] = v
    return out


def _to_host(x):
    if torch.is_tensor(x):
        return x.cpu() if x.device.type != 'cpu' else x
    if isinstance(x, dict):
        return {k: _to_host(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_host(v) for v in x]
    if isinstance(x, tuple):
        v = [_to_host(u) for u in x]
        return type(x)(*v) if hasattr(x, '_fields') else tuple(v)
    return x


class OutputManager(object):
    """Receives every committed proposal. Per connector: 'memory' keeps the
    per-step series for `series()`; 'none' keeps nothing; a callable sink
    (name, proposal) is called as well (e.g. a pipeline's own history
    aggregation)."""

    def __init__(self, host=False, keep=True):
        self.mode = {}
        self.parts = {}
        self.sinks = []
        #: keep the series on the host (run.keep_outputs_on: host)
        self.host = bool(host)
        #: False: keep no series (run.keep_outputs_on: none)
        self.keep_any = bool(keep)
        #: a speculative window (runtime._window) holds the sinks' calls
        #: until its checks pass; None: they are called at commit
        self.hold = None

    def keep(self, name, mode='memory'):
        self.mode[name] = mode if self.keep_any else 'none'

    def drops(self, name):
        """Nothing reads `name`'s series: not kept and no sink."""
        return self.mode.get(name, 'none') != 'memory' and not self.sinks

    def add_sink(self, fn):
        self.sinks.append(fn)

    def commit(self, name, proposal):
        if self.mode.get(name, 'none') == 'memory' and \
                proposal.series is not None:
            ser = proposal.series
            if self.host:
                ser = _to_host(ser)
            self.parts.setdefault(name, []).append(ser)
        if self.hold is not None:
            if self.sinks:
                self.hold.append((name, proposal))
            return
        for fn in self.sinks:
            fn(name, proposal)

    def release(self, held):
        """The held commits to the sinks, in order."""
        for name, proposal in held or ():
            for fn in self.sinks:
                fn(name, proposal)

    def series(self, name, final_keys=()):
        return concat_series(self.parts.get(name, []), final_keys)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
FORMAT = 'torchvic-coupling-checkpoint'


#: settings that change how fast a run goes, not what it computes: a
#: checkpoint is taken by a run that differs in them
SPEED_ONLY = {'ledger': ('speculate',),
              'modules': ('day_session', 'session_days')}


def config_digest(cfg):
    keep = {k: v for k, v in cfg.items() if not k.startswith('_')
            and k not in ('checkpoint', 'output')}
    if isinstance(keep.get('ledger'), dict):
        keep['ledger'] = {k: v for k, v in keep['ledger'].items()
                          if k not in SPEED_ONLY['ledger']}
    if isinstance(keep.get('modules'), dict):
        keep['modules'] = {
            m: ({k: v for k, v in b.items()
                 if k not in SPEED_ONLY['modules']}
                if isinstance(b, dict) else b)
            for m, b in keep['modules'].items()}
    return hashlib.sha256(json.dumps(keep, sort_keys=True,
                                     default=str).encode()).hexdigest()


class CheckpointManager(object):
    """All module states, the clock, the exchange buffers and the ledger in
    one file (torch.save). A restart refuses a checkpoint of another
    configuration (other modules, connections or settings)."""

    def __init__(self, digest):
        self.digest = digest

    def save(self, path, t, tick, connectors, buffers, ledger, state_mgr):
        blob = dict(format=FORMAT, version=1, digest=self.digest,
                    written=_dt.datetime.now().isoformat(timespec='seconds'),
                    t=int(t), tick=int(tick),
                    connectors={k: c.save_state()
                                for k, c in connectors.items()},
                    buffers={'|'.join(k): b.state()
                             for k, b in buffers.items()},
                    ledger=ledger.state(),
                    accepted={k: {n: (v.cpu() if torch.is_tensor(v) else v)
                                  for n, v in d.items()}
                              for k, d in state_mgr.accepted.items()})
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        tmp = path + '.part'
        torch.save(blob, tmp)
        os.replace(tmp, path)
        return path

    def load(self, path):
        blob = torch.load(path, map_location='cpu', weights_only=False)
        if blob.get('format') != FORMAT:
            raise CouplingError('%s is not a coupling checkpoint' % path)
        if blob['digest'] != self.digest:
            raise CouplingError('%s was written by another configuration '
                                '(modules, connections or settings differ)'
                                % path)
        return blob
