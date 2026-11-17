"""pytest -p tests.fast_router_plugin: run the router suites on FastStepRouter."""
from torchvic.model import river_network as rn
from torchvic.model import routing_fast

_orig = rn.build_step_router


def _fast(*a, **k):
    r, info = _orig(*a, **k)
    return routing_fast.FastStepRouter.from_router(r), info


def pytest_configure(config):
    rn.build_step_router = _fast
