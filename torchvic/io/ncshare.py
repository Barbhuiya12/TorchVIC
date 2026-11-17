"""Keeping NetCDF handles from corrupting one another.

netCDF-C opens every file with HDF5's H5F_CLOSE_STRONG, and HDF5 keeps one
shared file structure per file on disk. Two xarray datasets over the same
path therefore share that structure, and closing either one -- explicitly, by
the garbage collector, or by xarray's LRU file cache -- destroys the objects
the other still holds: its reads return garbage and the next nc_open of the
file segfaults inside libhdf5 (reproduced with plain xarray: open A, open B,
close B, read A, open C; this killed the full test suite). Two rules keep a
process clear of it:

* `read(path)` -- parameter, domain and other small input files -- opens,
  loads into memory and closes at once, so nothing stays open to be damaged
  and no Case holds a handle.
* `open_shared(path)` -- forcing files read lazily during a run -- keeps one
  handle per file; every caller gets a shallow copy of the same dataset, and
  the handle closes when the last copy is closed or collected.
"""
import contextlib
import os
import threading
import weakref

_LOCK = threading.RLock()
_OPEN = {}            # (realpath, mtime_ns, size, kwargs) -> [Dataset, users]
_MEM = {}             # (realpath, mtime_ns, size, '[]') -> in-memory Dataset


def _key(path, kw):
    st = os.stat(path)
    return (os.path.realpath(path), st.st_mtime_ns, st.st_size,
            repr(sorted(kw.items())))


def _drop(key):
    with _LOCK:
        ent = _OPEN.get(key)
        if ent is None:
            return
        ent[1] -= 1
        if ent[1] <= 0:
            del _OPEN[key]
            ent[0].close()


def _copy(key):
    ent = _OPEN[key]
    ent[1] += 1
    ds = ent[0].copy(deep=False)
    fin = weakref.finalize(ds, _drop, key)
    ds.set_close(fin)
    return ds


def open_shared(path, **kw):
    import xarray as xr
    path = os.fspath(path)
    key = _key(path, kw)
    with _LOCK:
        if key not in _OPEN:
            _OPEN[key] = [xr.open_dataset(path, **kw), 0]
        return _copy(key)


def read(path, **kw):
    """The file as an in-memory dataset; no handle is left open."""
    import xarray as xr
    path = os.fspath(path)
    with _LOCK:
        key = _key(path, kw)
        if key in _MEM:
            return _MEM[key].copy(deep=False)
        if key in _OPEN:
            with _copy(key) as ds:
                return ds.load()
        with xr.open_dataset(path, **kw) as ds:
            return ds.load()


@contextlib.contextmanager
def preloaded(path, ds):
    """Inside the block `read(path)` returns a shallow copy of `ds`, the
    file already read into memory (parallel.preread)."""
    path = os.fspath(path)
    key = _key(path, {})
    with _LOCK:
        _MEM[key] = ds
    try:
        yield ds
    finally:
        with _LOCK:
            _MEM.pop(key, None)
