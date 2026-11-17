"""The CUDA generator's graph state is made outside inference mode
(graphsafe.ensure_generator_graph_state): a capture inside
torch.inference_mode, as basin.simulate makes, no longer makes a later
capture outside it fail with "Inplace update to inference tensor outside
InferenceMode" (marker gpu; each case in a fresh process, since the state is
made once per process)."""
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='no CUDA device')

_CODE = """
import torch
{pre}
def capture():
    x = torch.zeros(4, device='cuda')
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        x.add_(1)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x.add_(1)
    g.replay(); torch.cuda.synchronize()
    return float(x[0])
with torch.inference_mode():
    {first}
print(capture())
"""


def _run(first, pre=''):
    return subprocess.run(
        [sys.executable, '-c', _CODE.format(pre=pre, first=first)],
        capture_output=True, text=True)


@pytest.mark.gpu
@needs_cuda
def test_a_capture_inside_inference_mode_poisons_a_later_one_without_the_fix():
    r = _run('capture()')
    # torch alone: the later capture outside inference mode fails (the bug
    # the fix works around); a torch that has fixed it passes, and so be it
    assert r.returncode == 0 or 'inference tensor' in r.stderr


@pytest.mark.gpu
@needs_cuda
def test_capture_outside_inference_mode_works_after_a_segmenter_inside_it():
    r = _run('graphsafe.Segmenter(None); capture()',
             pre='from torchvic.core import graphsafe')
    assert r.returncode == 0, r.stderr[-600:]


def test_runtime_initialize_makes_the_generator_state_first():
    """Runtime.initialize makes the generator's graph state before any
    connector initialises: a module's own capture (crop, groundwater, phase
    scheduler) inside inference mode must not poison the Segmenter's later
    one (ISC WOFOST run on an H200). Source-level guard; the GPU behaviour is
    the ISC campaign's S8n step."""
    import inspect
    from torchvic.coupling import runtime as R
    src = inspect.getsource(R.Runtime.initialize)
    assert src.index('ensure_generator_graph_state') < src.index(
        'c.initialize(')
