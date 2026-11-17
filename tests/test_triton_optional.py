# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The Triton kernel modules import without Triton.

CPU-only torch wheels do not ship Triton. The modules that define Triton
kernels used to `import triton` unconditionally, and the coupled system's
connectors import them, so `import torchvic.coupling.connectors` failed on such
an install. They now fall back to torchvic.core.triton_stub, whose kernels raise
one clear error when launched. With Triton installed they use the real modules.
"""
import importlib
import os
import subprocess
import sys
import textwrap

import pytest

import _paths

MODULES = ('torchvic.model.routing_kernels', 'torchvic.model.routing_cascade_kernels',
           'torchvic.vicwur.dams_kernels', 'torchvic.vicwur.efr_kernels',
           'torchvic.vicwur.river_exchange_kernels', 'torchvic.vicwur.wateruse_kernels',
           'torchvic.flood.triton_step', 'torchvic.flood.multigpu')


def test_connectors_and_kernel_modules_import_without_triton():
    code = textwrap.dedent("""
        import sys
        # what a CPU-only torch install looks like: there is no triton package
        for name in ('triton', 'triton.language', 'triton.language.extra',
                     'triton.language.extra.cuda'):
            sys.modules[name] = None
        import importlib
        import torchvic.coupling.connectors
        for name in %r:
            importlib.import_module(name)
        from torchvic.core import triton_stub
        from torchvic.model import routing_kernels as K, fastlaunch
        assert K.triton is triton_stub.triton and K.tl is triton_stub.tl
        # launching a kernel, or inspecting one the way the launch helpers do,
        # stops with one clear error
        for call in (lambda: K.gather_kernel[(1,)](None, 1, None, None, None, BF=32),
                     lambda: K.gather_kernel(None),
                     lambda: fastlaunch.launch(K.gather_kernel, (1,), (None, 1), {})):
            try:
                call()
            except RuntimeError as e:
                assert 'needs Triton' in str(e) and 'gather_kernel' in str(e), e
            else:
                raise SystemExit('a kernel launched without Triton')
        # calling into triton or triton.language other than at import time raises too
        for call in (lambda: triton_stub.triton.cdiv(10, 3), lambda: triton_stub.tl.float64(1)):
            try:
                call()
            except RuntimeError as e:
                assert 'needs Triton' in str(e), e
            else:
                raise SystemExit('a Triton call returned without Triton')
        print('ok')
    """ % (MODULES,))
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([_paths.REPO] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env['CUDA_VISIBLE_DEVICES'] = ''
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                       env=env, cwd=_paths.REPO)
    assert r.returncode == 0 and r.stdout.strip().endswith('ok'), r.stdout[-600:] + r.stderr[-1800:]


def test_with_triton_installed_the_kernel_modules_use_the_real_triton():
    triton = pytest.importorskip('triton')
    tl = pytest.importorskip('triton.language')
    for name in MODULES:
        m = importlib.import_module(name)
        assert m.triton is triton, name
        assert m.tl is tl, name
