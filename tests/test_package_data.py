# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The wheel carries every data file the package reads.

An editable install reads the source tree, so a data file that is missing from
`[tool.setuptools.package-data]` in pyproject.toml goes unnoticed until
someone installs the wheel. Every non-Python file under torchvic/ must match a
package-data pattern, and the one the nitrogen code reads at run time
(crop/cropsyst/sdr_field_ops.txt) must load.
"""
import fnmatch
import os

import pytest

import _paths

PKG = os.path.join(_paths.REPO, 'torchvic')
PYPROJECT = os.path.join(_paths.REPO, 'pyproject.toml')


def _patterns():
    if not os.path.exists(PYPROJECT):
        pytest.skip('pyproject.toml is not next to the package')
    try:
        import tomllib
    except ImportError:                      # Python 3.10
        pytest.skip('tomllib needs Python 3.11')
    with open(PYPROJECT, 'rb') as f:
        cfg = tomllib.load(f)
    return cfg['tool']['setuptools']['package-data']['torchvic']


def test_every_data_file_of_the_package_is_listed_in_package_data():
    patterns = _patterns()
    unlisted = []
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for name in files:
            if name.endswith(('.py', '.pyc')):
                continue
            rel = os.path.relpath(os.path.join(root, name), PKG)
            rel = rel.replace(os.sep, '/')
            if not any(fnmatch.fnmatch(rel, p) for p in patterns):
                unlisted.append(rel)
    assert not unlisted, (
        'data files under torchvic/ that a wheel would leave out (add them '
        'to [tool.setuptools.package-data]): %s' % sorted(unlisted))


def test_the_nrcs_field_operations_table_loads():
    from torchvic.crop.cropsyst import soil_n
    ops = soil_n.sdr_operations()
    assert len(ops) > 100
    first = next(iter(ops.values()))
    assert {'number', 'depth_m', 'mixing', 'rating', 'effect', 'flatten',
            'burial'} <= set(first)
