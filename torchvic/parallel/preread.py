"""The parameter file read by several processes at once.

The image driver's parameter file is zlib-compressed, one 2-D slab per
chunk, and HDF5 inflates one chunk at a time per process: on the ISC domain
(2.7 GB inflated) a serial read costs about 7 s and the run cannot start
before. `preloaded` inflates the large per-cell variables in a pool of
forked processes, each writing its variable straight into a buffer shared
with the parent (an anonymous shared mapping made before the fork, so no
copy goes through a pipe), and assembles the file in memory as
`io.ncshare.read` returns it: the same variables in the same order with the
same attributes and encodings and the same decoded values. Inside the block
`io.ncshare.read` of that file answers from this dataset, so ImageParams
(and anything else reading the file) takes its variables from memory instead
of inflating the whole file again. A variable the pool could not read is read
by the parent, so the values are always the numbers HDF5 decodes.

Every handle follows io.ncshare's rule (one shared handle per file per
process, closed as soon as the read is done); the parent's is closed before
the pool forks.
"""
import contextlib
import mmap
import os

import numpy as np

_BUF = {}           # name -> (mmap, dtype, shape), inherited by the pool


def _read(args):
    path, name = args
    from ..io.ncshare import open_shared
    buf, dtype, shape = _BUF[name]
    with open_shared(path) as ds:
        v = np.asarray(ds[name].values)
    if v.dtype != dtype or v.shape != shape:
        return name, False
    np.frombuffer(buf, dtype=dtype).reshape(shape)[...] = v
    return name, True


def wanted_names(ds):
    """The variables ImageParams takes with `ImageDomain.take`."""
    from ..io.params import ImageParams as P
    names = list(P.SOIL_2D + P.SOIL_3D + P.VEG_VARS)
    names += ['off_gmt', 'Nveg', 'July_Tavg', 'MaxiE_or_CO2Spec', 'LUE',
              'Nscale', 'AreaFract', 'Pfactor', 'elevation']
    out = []
    for k in dict.fromkeys(names):
        if k in ds:
            out.append(k)
    out.sort(key=lambda k: -int(np.prod(ds[k].shape)))
    return out


def read_full(path, procs=8):
    """`io.ncshare.read(path)`, the large variables inflated by `procs`
    forked processes."""
    import concurrent.futures as cf
    import multiprocessing as mp
    from ..io.ncshare import open_shared
    with open_shared(path) as ds:
        order = list(ds.data_vars)
        names = [k for k in wanted_names(ds)
                 if k in ds.data_vars and ds[k].ndim >= 2 and
                 ds[k].dtype.kind in 'biuf']
        rest = ds.drop_vars(names).load()
        lazy = {k: ds[k].variable for k in names}
        meta = {k: (np.dtype(v.dtype), tuple(v.shape))
                for k, v in lazy.items()}
    procs = max(1, min(int(procs), len(names)))
    _BUF.clear()
    for k, (dt, sh) in meta.items():
        _BUF[k] = (mmap.mmap(-1, max(1, dt.itemsize * int(np.prod(sh)))),
                   dt, sh)
    try:
        ok = {}
        if procs > 1:
            with cf.ProcessPoolExecutor(
                    procs, mp_context=mp.get_context('fork')) as ex:
                ok = dict(ex.map(_read, [(path, k) for k in names]))
        out = {}
        for k in names:
            buf, dt, sh = _BUF[k]
            if ok.get(k):
                data = np.frombuffer(buf, dtype=dt).reshape(sh)
            else:
                with open_shared(path) as d:
                    data = np.asarray(d[k].values)
            out[k] = lazy[k].copy(deep=False, data=data)
    finally:
        _BUF.clear()
    full = rest.assign(out)[order]
    full.attrs = dict(rest.attrs)
    full.encoding = dict(rest.encoding)
    return full


@contextlib.contextmanager
def preloaded(path, domain=None, procs=None):
    """Inside the block `io.ncshare.read(path)` answers from the file read
    into memory by `read_full` (TORCHVIC_PREREAD=0 or 1 turns the pool off).
    `domain`, if given, lets a valid cache of the reduced parameters
    (io.params) stand in for the read."""
    from ..io import ncshare
    procs = int(os.environ.get('TORCHVIC_PREREAD', 8 if procs is None
                               else procs))
    if procs <= 1 or not isinstance(path, str) or not os.path.exists(path):
        yield None
        return
    if domain is not None:
        # the reduced parameters are cached next to the file (io.params):
        # nothing to inflate when ImageParams will find them
        from ..io.params import params_cache_valid
        if params_cache_valid(path, domain):
            yield None
            return
    ds = read_full(path, procs)
    with ncshare.preloaded(path, ds):
        yield ds
