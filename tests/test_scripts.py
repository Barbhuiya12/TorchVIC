"""The standalone verification scripts (conftest.SCRIPTS), each run in its
own process and required to exit 0.

They are scripts rather than pytest modules because each prints its own
comparison table against VIC's compiled code and some select a default
torch device at import, which must not leak into the rest of the session.
"""
import os
import subprocess
import sys

import pytest

import _paths
from conftest import ORACLES, SCRIPTS

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.reference
@pytest.mark.parametrize('script', SCRIPTS)
def test_script(script):
    if not any(os.path.exists(_paths.oracle(n)) for n in ORACLES):
        pytest.skip('no oracle program built in %s' % _paths.ORACLE_DIR)
    r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       cwd=HERE, capture_output=True, text=True,
                       timeout=3600)
    assert r.returncode == 0, (r.stdout[-4000:] + r.stderr[-4000:])
