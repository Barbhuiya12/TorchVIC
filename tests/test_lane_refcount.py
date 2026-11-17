"""No run may drop a reference it never took (the bool_dealloc abort).

torch 2.6's get_eval_frame_callback returns a borrowed reference; every
torch.compiler.set_stance, which the captured LANE_COMPACT step enters for
each compiled region, then lowered the count of False (dynamo's callback
there). CPython 3.11's bools are not immortal: a calibration process
aborted with "Fatal Python error: bool_dealloc" after a few thousand runs.
core.fusion replaces the getter (fusion._own_eval_frame_callback).
"""
import gc
import os
import sys

import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def test_eval_frame_getter_keeps_the_count():
    from torchvic.core import fusion                       # noqa: F401
    try:
        from torch._C._dynamo import eval_frame as ef
    except ImportError:
        pytest.skip('no torch._C._dynamo')
    for value in (None, False):
        prev = ef.set_eval_frame(value)
        try:
            n0 = sys.getrefcount(value)
            for _ in range(20):
                ef.get_eval_frame_callback()
            n1 = sys.getrefcount(value)
        finally:
            ef.set_eval_frame(prev)
        assert n1 == n0, (value, n1 - n0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
@pytest.mark.parametrize('lanes', [False, True], ids=['padded', 'lanes'])
def test_fifty_runs_keep_the_bool_counts(tmp_path, lanes):
    import test_lanes_graphs as lg
    from torchvic.model import basin, state as model_state
    if not os.path.exists(os.path.join(lg.forest.BOX, 'params.nc')):
        pytest.skip('the ISC forest box is not present')
    c = lg._case(tmp_path, torch.device('cuda'))
    kw = model_state.time_window(c.kwargs(), 0, 8 * 2)
    old = basin.LANE_COMPACT
    basin.LANE_COMPACT = lanes
    try:
        def run():
            with torch.no_grad():
                o = basin.simulate(**kw, return_state=True)
            torch.cuda.synchronize()
            return o
        for _ in range(2):                  # compile, capture, caches
            o = run()
        on_lanes = o['state'].get('lanes') is not None
        del o
        gc.collect()                        # cyclic garbage is not a leak
        t0, f0 = sys.getrefcount(True), sys.getrefcount(False)
        for _ in range(50):
            run()
        gc.collect()
        t1, f1 = sys.getrefcount(True), sys.getrefcount(False)
    finally:
        basin.LANE_COMPACT = old
    assert on_lanes == lanes
    assert (f1 - f0, t1 - t0) == (0, 0)
