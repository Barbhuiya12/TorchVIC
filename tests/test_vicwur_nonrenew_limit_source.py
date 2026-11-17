# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""NONRENEWABLE_LIMIT is read where VIC-WUR 3.0 reads it: the CONSTANTS file only.

VIC-WUR 3.0 (wu_get_global_param.c:158-174) takes NREN_LIM from the parameter
(CONSTANTS) file; in the global file the key is an unrecognized option, so a
VIC-WUR 3.0 run with `NONRENEWABLE_LIMIT 10` in its global file pumps without a cap.
TorchVIC read it from the global file too, which capped the non-renewable
withdrawal at 10 mm per cell where VIC-WUR 3.0 had none (day 1 of the
Punjab sub-domain, 4.5e3 vs 5.8e4 mm summed over the cells)."""
import warnings

import pytest

from torchvic.coupling.connectors import vicwur_setup as WS


def test_global_file_limit_is_ignored_with_a_warning():
    g = {'NONRENEWABLE_LIMIT': [['10']]}
    with pytest.warns(UserWarning, match='CONSTANTS'):
        assert WS.nonrenew_limit(g) is None


def test_constants_file_limit_is_read(tmp_path):
    c = tmp_path / 'constants.txt'
    c.write_text('# W3 parameters\nNONRENEWABLE_LIMIT 12.5  # mm\n')
    g = {'CONSTANTS': [[str(c)]]}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert WS.nonrenew_limit(g) == 12.5


def test_both_places_take_the_constants_value(tmp_path):
    c = tmp_path / 'constants.txt'
    c.write_text('NONRENEWABLE_LIMIT 7\n')
    g = {'CONSTANTS': [[str(c)]], 'NONRENEWABLE_LIMIT': [['10']]}
    with pytest.warns(UserWarning):
        assert WS.nonrenew_limit(g) == 7.0


def test_no_limit_anywhere_is_unlimited():
    assert WS.nonrenew_limit({}) is None
