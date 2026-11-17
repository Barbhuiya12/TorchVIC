"""Forcing read once and handed to every shard.

A sharded run used to have each process read every window of the whole
grid (the files are float32 [time, y, x]) and keep its own cells: N
processes read and decoded N times the data, and the reader thread of each
fought its compute thread for the GIL (4 GPUs: 13-15 s of waiting per
process, against 6.6 s for one). Here one reader process per forcing field
(the files are uncompressed, so a field is a copy) reads each window once,
cuts it into the shards' cells -- float64 and the unit scale applied, as
`ImageForcing.read_steps` gives it -- and writes it into a slot of shared
memory; every shard takes its cells from the slot and copies them to its
device on a side stream. The values are `read_steps`' bit for bit.

Slots form a ring of `slots` windows. A reader waits for every shard to
have released a slot before it overwrites it; a shard waits for every field
of its cells to be in.
"""
import os
import time

import numpy as np


class Feed(object):
    """Shared state of one feed; created by the parent, passed to the
    reader and shard processes at spawn."""

    def __init__(self, ctx, global_file, spans, fields, rows, cols,
                 directory, slots=3):
        self.global_file = global_file
        self.spans = [tuple(map(int, s)) for s in spans]
        self.fields = list(fields)
        self.rows, self.cols = rows, cols              # per shard
        self.n = len(rows)
        self.slots = int(slots)
        self.tmax = max(b - a for a, b in self.spans)
        self.dir = directory
        # one file per (slot, shard, field) in `directory` (tmpfs): created
        # here, sized, never written until a reader fills it -- nothing is
        # touched up front
        for s in range(self.slots):
            for i in range(self.n):
                for k in range(len(self.fields)):
                    with open(self.path(s, i, k), 'wb') as fh:
                        fh.truncate(self.tmax * len(rows[i]) * 8)
        self.ready = [[ctx.Semaphore(0) for _ in range(self.n)]
                      for _ in range(self.slots)]
        self.free = [[ctx.Semaphore(0) for _ in self.fields]
                     for _ in range(self.slots)]
        self.abort = ctx.Event()
        self._maps = {}

    def path(self, s, i, k):
        return os.path.join(self.dir, 's%d_c%d_f%d.f64' % (s, i, k))

    def view(self, s, i, k):
        """The slot's [tmax, cells of shard i] float64 array (a memory map
        of the file, opened once per process)."""
        key = (s, i, k)
        m = self._maps.get(key)
        if m is None:
            m = self._maps[key] = np.memmap(
                self.path(s, i, k), dtype=np.float64, mode='r+',
                shape=(self.tmax, len(self.rows[i])))
        return m

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_maps'] = {}
        return d


def _wait(sem, feed, who):
    while not sem.acquire(timeout=1.0):
        if feed.abort.is_set():
            raise RuntimeError('forcing feed aborted (%s)' % who)


def reader(feed, k):
    """Process target: field `k` of every window into the slots."""
    try:
        from ..io.domain import ImageDomain
        from ..io.forcing import ImageForcing
        from ..io.global_param import GlobalParam
        gp = GlobalParam(feed.global_file)
        dom = ImageDomain(gp.domain_file, names=gp.domain_vars)
        fc = ImageForcing(gp.forcing_prefix, dom, force_type=gp.force_type)
        fc.plan()
        canon = feed.fields[k]
        for w, (a0, b0) in enumerate(feed.spans):
            s = w % feed.slots
            if w >= feed.slots:
                for _ in range(feed.n):
                    _wait(feed.free[s][k], feed, 'reader %s' % canon)
            views = [feed.view(s, i, k) for i in range(feed.n)]
            off = 0
            for ds, n, take in fc._plan:
                a, b = max(a0 - off, 0), min(b0 - off, n)
                lo = max(off, a0) - a0                    # offset in window
                off += n
                if a >= b:
                    continue
                for var, can, scale in take:
                    if can != canon:
                        continue
                    arr = ds[var][a:b].values
                    for i in range(feed.n):
                        v = np.asarray(arr[..., feed.rows[i], feed.cols[i]],
                                       dtype=np.float64)
                        if scale != 1.0:
                            v = v * scale
                        views[i][lo:lo + (b - a)] = v
                    del arr
            for i in range(feed.n):
                feed.ready[s][i].release()
        fc.close()
    except BaseException:                                # noqa: BLE001
        feed.abort.set()
        raise


class SharedWindows(object):
    """What WindowPrefetch is, for shard `i` of a Feed: iterating gives
    {field: [step, cell]} on the device in `dtype`; `wait_s` is the time the
    caller waited."""

    def __init__(self, feed, i, device, dtype, depth=2):
        import queue
        import threading
        import torch
        self.feed, self.i = feed, i
        self.device, self.dtype = torch.device(device), dtype
        self.cuda = self.device.type == 'cuda'
        self.wait_s = 0.0
        self._q = queue.Queue(maxsize=max(int(depth), 1))
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name='forcing-feed')
        self._t.start()

    def _run(self):
        import torch
        f, i = self.feed, self.i
        try:
            stream = (torch.cuda.Stream(device=self.device) if self.cuda
                      else None)
            for w, (a, b) in enumerate(f.spans):
                if self._stop:
                    return
                s, nt = w % f.slots, b - a
                for _ in f.fields:
                    _wait(f.ready[s][i], f, 'shard %d' % i)
                out, ev = {}, None
                if self.cuda:
                    with torch.cuda.stream(stream):
                        for k, name in enumerate(f.fields):
                            out[name] = torch.from_numpy(
                                f.view(s, i, k)[:nt]).to(
                                self.device, non_blocking=True
                            ).to(self.dtype)
                        ev = torch.cuda.Event()
                        ev.record(stream)
                    ev.synchronize()       # the slot may be rewritten now
                else:
                    for k, name in enumerate(f.fields):
                        out[name] = torch.from_numpy(
                            f.view(s, i, k)[:nt]).to(
                            self.device, self.dtype).clone()
                for k in range(len(f.fields)):
                    f.free[s][k].release()
                self._q.put((out, ev, None))
        except BaseException as e:      # handed to the caller
            f.abort.set()
            self._q.put((None, None, e))

    def __iter__(self):
        import torch
        for _ in self.feed.spans:
            t = time.time()
            out, ev, err = self._q.get()
            self.wait_s += time.time() - t
            if err is not None:
                raise err
            if ev is not None:
                cur = torch.cuda.current_stream(self.device)
                cur.wait_event(ev)
                for v in out.values():
                    v.record_stream(cur)
            yield out

    def close(self):
        import queue
        self._stop = True
        while self._t.is_alive():
            try:
                self._q.get(timeout=0.1)
            except queue.Empty:
                pass
