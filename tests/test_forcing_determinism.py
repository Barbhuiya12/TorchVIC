# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The forcing's daily aggregation is the same every run on a GPU, and the
CPU's: io.forcing._ordered_index_sum (aggregate_to_daily's sums) against
the CPU's index_add, and two default-mode ISC runs (no deterministic
algorithms asked for) bit for bit -- they used to differ in the last bits
of the MTCLIM daily means (GPU index_add's atomics), and the whole coupled
run with them."""
import os
import sys

import numpy as np
import pytest
import torch

import _paths  # noqa: F401
from torchvic.io import forcing as F

HERE = os.path.dirname(os.path.abspath(__file__))
cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason='needs CUDA')


@pytest.mark.gpu
@cuda
def test_ordered_index_sum_is_the_cpus_on_the_gpu():
    g = torch.Generator().manual_seed(5)
    n, c = 24 * 40 + 7, 333
    v = torch.randn(n, c, generator=g, dtype=torch.float64) * 1e3
    # bins in time order with a ragged first and last day, then shuffled
    idx = torch.arange(n) // 24
    perm = torch.randperm(n, generator=g)
    for ix, x in ((idx, v), (idx[perm], v[perm])):
        nb = int(ix.max()) + 1
        ref = torch.zeros(nb, c, dtype=torch.float64).index_add(0, ix, x)
        assert torch.equal(F._ordered_index_sum(nb, ix, x), ref)
        a = F._ordered_index_sum(nb, ix.cuda(), x.cuda()).cpu()
        b = F._ordered_index_sum(nb, ix.cuda(), x.cuda()).cpu()
        assert torch.equal(a, ref) and torch.equal(b, ref)


@pytest.mark.gpu
@cuda
def test_aggregate_to_daily_gpu_is_cpu():
    import datetime
    g = torch.Generator().manual_seed(9)
    T, c = 24 * 10, 50
    t0 = datetime.datetime(2005, 1, 1)
    times = np.array([t0 + datetime.timedelta(hours=h) for h in range(T)],
                     dtype='datetime64[s]')
    f = {k: torch.rand(T, c, generator=g, dtype=torch.float64) * 30
         for k in ('prcp', 'tair', 'shortwave', 'vp')}
    a, _ = F.aggregate_to_daily(f, times, utc_offset=5.5)
    b, _ = F.aggregate_to_daily({k: v.cuda() for k, v in f.items()},
                                times, utc_offset=5.5)
    for k in a:
        assert torch.equal(a[k], b[k].cpu()), k


ISC = _paths.data('vicwur/isc/smoke')


@pytest.mark.gpu
@pytest.mark.slow
@cuda
def test_isc_two_default_runs_are_bitwise():
    """ISC (112,825 cells, every VIC-WUR plugin but WOFOST), 10 days, built
    and run twice in default mode: every connector's state and every
    output series bit for bit."""
    gp = os.path.join(ISC, 'global_isc_nowofost_2005.txt')
    if not os.path.exists(gp):
        pytest.skip('ISC inputs absent: ' + ISC)
    assert not torch.are_deterministic_algorithms_enabled()
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools',
                                    'vicwur', 'isc'))
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools',
                                    'vicwur'))
    import isc_run as I
    import l3_runs as L
    from torch.utils._pytree import tree_map
    static = dict(np.load(os.path.join(ISC, 'gw_static.npz')))
    old = I.KEEP
    I.KEEP = 'host'
    runs = []
    try:
        for _ in range(2):
            m, case = I.build(gp, 10, 'cuda', 'fixed', 10, static)
            del case
            with torch.inference_mode():
                m.initialize()
                m.run()
                st = tree_map(lambda x: x.cpu() if torch.is_tensor(x) else x,
                              {k: c.save_state() for k, c in
                               m.runtime.connectors.items()})
                ser = {n: {k: v.cpu() for k, v in m.outputs(n).items()
                           if torch.is_tensor(v)}
                       for n in m.runtime.connectors}
            runs.append((st, ser))
            del m
            torch.cuda.empty_cache()
    finally:
        I.KEEP = old
    (s0, o0), (s1, o1) = runs
    assert L._diff(s0, s1) is None, L._diff(s0, s1)
    assert L._diff(o0, o1) is None, L._diff(o0, o1)
    assert sum(len(v) for v in o0.values()) > 10


@pytest.mark.gpu
@cuda
@pytest.mark.parametrize('fan', [3, 40, 5000])
def test_ordered_index_add_is_the_cpus(fan):
    """core.ordered.index_add_ on a GPU: the CPU's index_add_ bit for bit
    (into a non-zero start, on dim 0 and 1), passes or segment sums."""
    from torchvic.core import ordered
    g = torch.Generator().manual_seed(fan)
    n = 6 * fan + 3
    idx = torch.randint(0, 7, (n,), generator=g)
    for dim in (0, 1):
        shp = (n, 5) if dim == 0 else (4, n)
        oshp = (9, 5) if dim == 0 else (4, 9)
        src = torch.randn(*shp, generator=g, dtype=torch.float64)
        out = torch.randn(*oshp, generator=g, dtype=torch.float64)
        ref = out.clone().index_add_(dim, idx, src)
        got = ordered.index_add_(out.cuda(), dim, idx.cuda(), src.cuda())
        assert torch.equal(got.cpu(), ref), (fan, dim)
        got = ordered.index_add(out.cuda(), dim, idx.cuda(), src.cuda())
        assert torch.equal(got.cpu(), ref), (fan, dim)
