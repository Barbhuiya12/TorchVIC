"""History files written while the model computes.

`AsyncStreamWriter` writes the OUTFILE streams of a run in a forked
process: each window's aggregated records are handed over as they are
made, and a file (one HISTFREQ period of a stream) is created at its first
record, appended to window by window and closed at its last. When the last
window is done only that window's slice is left to write. The files are
the ones `history.write_record` writes -- the same code (history._GroupFile)
puts the same values in the same variables; only the order in which the
records arrive differs.

Create the writer before the run starts any thread or reads the forcing
(it forks); tensors travel through shared memory.
"""
import collections
import traceback

import torch

from . import history


class _Sources(dict):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def __missing__(self, k):
        return self.tag


def _worker(ctxs, q, res):
    files, paths, err = {}, [], None
    try:
        while True:
            item = q.get()
            if item is None:
                break
            if err is not None:
                continue
            try:
                prefix, r0, data = item
                ctx = ctxs[prefix]
                arrays = {k: v.numpy() for k, v in data.items()}
                n = len(next(iter(arrays.values())))
                for gn, g in enumerate(ctx.groups):
                    if g[-1] < r0 or g[0] >= r0 + n:
                        continue
                    f = files.get((prefix, gn))
                    if f is None:
                        f = files[(prefix, gn)] = history._GroupFile(ctx, g)
                    f.append(arrays, r0)
                    if r0 + n > g[-1]:
                        paths.append(f.close())
                        del files[(prefix, gn)]
                del data, arrays
            except BaseException:                    # noqa: BLE001
                err = traceback.format_exc()
    finally:
        for f in files.values():
            f.close()
    res.put((paths, err))


class AsyncStreamWriter(object):
    def __init__(self, streams, dates, spd, domain, gp, out,
                 source='run_isc_torchvic', depth=3):
        import torch.multiprocessing as mp
        dt = 86400 // spd
        times = history.as_datetimes(dates)
        self.ctxs = {}
        self.next = collections.Counter()
        for s in streams:
            recs, tail = history.windows(dates, dt, s.agg)
            if not recs:
                continue
            ctx = history._FileCtx(s, s.agg, recs, times, dt, domain, out, gp)
            ctx.sources = _Sources(source)
            self.ctxs[s.prefix] = ctx
        mpc = mp.get_context('fork')
        self.q = mpc.Queue(depth)
        self.res = mpc.Queue()
        self.proc = mpc.Process(target=_worker,
                                args=(self.ctxs, self.q, self.res),
                                daemon=True)
        self.proc.start()

    def _put(self, item):
        import queue
        while True:
            try:
                self.q.put(item, timeout=5)
                return
            except queue.Full:
                if not self.proc.is_alive():
                    raise RuntimeError('the history writer process died')

    def add(self, prefix, data):
        """The next records of stream `prefix`: {variable: [record, cell,
        (elem)] CPU tensor}."""
        if prefix not in self.ctxs or not data:
            return
        data = {k: v.contiguous() for k, v in data.items()}
        n = len(next(iter(data.values())))
        self._put((prefix, self.next[prefix], data))
        self.next[prefix] += n

    def close(self):
        """Wait for the files; returns their paths."""
        self._put(None)
        paths, err = self.res.get()
        self.proc.join()
        if err is not None:
            raise RuntimeError('history writer failed:\n' + err)
        return paths
