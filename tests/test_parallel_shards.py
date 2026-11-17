"""torchvic run --gpus N (torchvic/parallel): the cells split over N
processes give the single run's history and state files, value for value.
CPU processes stand in for the GPUs (--device cpu); the code path is the
same. Stehekin, 16 cells, 10 days.

The core is passed explicitly. On the 5.0.1 core the files are equal bit for
bit. On the 5.1.0 core they agree to an absolute 1e-12 (measured: at most
1.1e-13 for 3 shards, bit-identical for 2): this is rounding, not a physics
difference. The 5.1.0 code path depends on the batch shape, so the number of
cells in a shard changes the order of a floating-point sum."""
import glob
import os
import sys
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

import _paths                                               # noqa: E402

sys.path.insert(0, os.path.join(_paths.REPO, 'verification'))
matrix = pytest.importorskip('matrix')
from torchvic import cli                                    # noqa: E402
from torchvic.parallel import shard                         # noqa: E402
from torchvic.parallel.run import run_sharded               # noqa: E402

if not os.path.exists(matrix.CASES['stehe']['domain']):
    pytest.skip('Stehekin sample data is absent', allow_module_level=True)


def _global(tmp, state):
    c = dict(matrix.CASES['stehe'])
    c['start'], c['end'] = (1949, 1, 1), (1949, 1, 10)
    matrix.CASES['_t_par'] = c
    r = dict(matrix.row('stehe', 'EB'), case='_t_par')
    res = os.path.join(str(tmp), 'res')
    os.makedirs(res, exist_ok=True)
    txt = matrix.global_text(
        r, res, res, ['OUT_PREC', 'OUT_RUNOFF', 'OUT_EVAP', 'OUT_SWE',
                      'OUT_SOIL_MOIST', 'OUT_SOIL_TEMP'],
        extra_lines=['STATENAME %s' % state, 'STATEYEAR 1949',
                     'STATEMONTH 1', 'STATEDAY 6', 'STATESEC 0',
                     'WIND_H 10'])
    g = os.path.join(str(tmp), 'global.txt')
    open(g, 'w').write(txt)
    return g


CORES = ('5.0.1', '5.1.0')
#: absolute tolerance of the sharded files against the single run, per core
ATOL = {'5.0.1': 0.0, '5.1.0': 1e-12}


def _run(g, out, gpus=None, core='5.0.1'):
    argv = ['run', g, '--device', 'cpu', '--out', out, '--quiet',
            '--core', core]
    if gpus:
        argv += ['--gpus', str(gpus)]
    assert cli.main(argv) == 0


def _same(a, b, atol=0.0):
    n = 0
    fa, fb = sorted(glob.glob(a)), sorted(glob.glob(b))
    assert fa and [os.path.basename(x) for x in fa] == \
        [os.path.basename(x) for x in fb]
    for x, y in zip(fa, fb):
        with netCDF4.Dataset(x) as da, netCDF4.Dataset(y) as db:
            assert list(da.variables) == list(db.variables)
            for k in da.variables:
                da[k].set_auto_maskandscale(False)
                db[k].set_auto_maskandscale(False)
                u, v = np.asarray(da[k][:]), np.asarray(db[k][:])
                if atol and u.dtype.kind == 'f':
                    # VIC's fill value (9.97e36) and NaN must sit in the same
                    # places; the rest agrees to `atol`
                    ok = np.isfinite(u) & (np.abs(u) < 1e30)
                    assert np.array_equal(ok, np.isfinite(v) & (np.abs(v) < 1e30)), k
                    assert np.array_equal(u[~ok], v[~ok], equal_nan=True), k
                    assert not ok.any() or np.abs(u[ok] - v[ok]).max() <= atol, k
                else:
                    assert np.array_equal(u, v, equal_nan=u.dtype.kind == 'f'), k
                n += 1
    assert n


@pytest.fixture(scope='module', params=CORES)
def runs(request, tmp_path_factory):
    core = request.param
    tmp = tmp_path_factory.mktemp('par' + core.replace('.', ''))
    out = {'core': core}
    for tag, gpus in (('one', None), ('two', 2), ('three', 3)):
        t = tmp / tag
        t.mkdir()
        (t / 'state').mkdir()
        _run(_global(t, str(t / 'state' / 'st')), str(t / 'out'), gpus, core)
        out[tag] = t
    return out


@pytest.mark.parametrize('tag', ['two', 'three'])
def test_history_is_the_single_runs_value_for_value(runs, tag):
    _same(str(runs['one'] / 'out' / '*.nc'), str(runs[tag] / 'out' / '*.nc'),
          ATOL[runs['core']])


@pytest.mark.parametrize('tag', ['two', 'three'])
def test_state_file_is_the_single_runs_value_for_value(runs, tag):
    _same(str(runs['one'] / 'state' / '*.nc'),
          str(runs[tag] / 'state' / '*.nc'), ATOL[runs['core']])


def test_shards_partition_the_cells_and_merge_back_in_order():
    import torch
    n, k = 11, 3
    idx = [shard.shard_index(n, k, i) for i in range(k)]
    assert sorted(np.concatenate(idx)) == list(range(n))
    x = torch.arange(n * 2, dtype=torch.float64).reshape(1, n, 2)
    merged = shard.merge_data([{'v': x[:, ix]} for ix in idx], idx)
    assert torch.equal(merged['v'], x)
    with pytest.raises(ValueError):
        shard.shard_index(2, 3, 0)


def test_contiguous_ranges_are_balanced_by_weight():
    w = np.array([2, 2, 2, 2, 8, 2, 2, 2, 2.])
    r = shard.shard_ranges(w, 3)
    assert r[0][0] == 0 and r[-1][1] == len(w)
    assert all(a < b for a, b in r)


def test_study_files_are_refused():
    with pytest.raises(SystemExit):
        run_sharded(SimpleNamespace(input='s.yaml', gpus=2))
