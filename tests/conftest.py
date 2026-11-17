"""Test configuration: data locations (see _paths.py) and the oracles.

Before collection, any oracle program missing from TORCHVIC_ORACLE_DIR is
compiled from tests/oracle against the pristine VIC trees in
TORCHVIC_REFERENCE_DIR, so a test that needs one runs rather than skips.
TORCHVIC_BUILD_ORACLES=0 turns that off. A build that fails is reported and
the tests that need the program skip, saying so.
"""
import os
import shutil
import subprocess

import pytest

import _paths

# ${VIC_SAMPLE_DATA} in the sample global files resolves here.
os.environ.setdefault('TORCHVIC_SAMPLE_DATA', _paths.SAMPLE_DATA)

ORACLES = ('ref_soil', 'ref_snow', 'ref_mtclim', 'ref_route', 'ref_cropsyst_uptake', 'ref_cropsyst_irrigation',
           'ref_cropsyst_phenology', 'ref_vicwur_route')

#: Standalone verification scripts: each has a main() returning 0 on
#: success, and some select a default torch device at import. pytest does
#: not import them; test_scripts.py runs each in its own process.
SCRIPTS = ('test_atmos.py', 'test_frozen_solve.py', 'test_implicit.py',
           'test_routing.py', 'test_routing_vs_rvic.py', 'test_snow_step.py',
           'test_soil_vs_vic.py', 'test_vegetation.py', 'test_vic_column.py',
           'test_vic_frozen.py', 'test_vic_snow.py')
collect_ignore = list(SCRIPTS)


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'slow: takes more than ~20 s on a CPU; left out of CI')
    config.addinivalue_line(
        'markers', 'reference: needs the pristine VIC trees, built oracles '
        'or sample data (skips when absent)')
    config.addinivalue_line('markers', 'gpu: needs a CUDA device')
    config.addinivalue_line(
        'markers', "vic_core(name): run the module's fixtures and tests on "
        "this VIC core ('5.0.1', '5.1.0', 'vicwur3'); without it a test "
        'runs on the default core (5.1.0)')

    if os.environ.get('TORCHVIC_BUILD_ORACLES', '1') == '0':
        return
    if not shutil.which(os.environ.get('CC', 'cc')):
        return
    for name in ORACLES:
        if os.path.exists(_paths.oracle(name)):
            continue
        script = os.path.join(_paths.ORACLE_SRC,
                              'build_%s.sh' % name)
        r = subprocess.run(['bash', script], capture_output=True, text=True)
        if r.returncode != 0:
            print('torchvic tests: could not build %s: %s'
                  % (name, (r.stderr or r.stdout).strip()[-500:]))


#: Tests that took 10 s or more on a CPU (durations of 2026-10-10). CI
#: leaves them out (-m "not slow"); a plain `pytest tests` runs them.
SLOW = (
    'test_full_energy_isc_vs_vic.py::test_igp_wilting_clamp_on_the_energy_path',
    'test_full_energy_isc_vs_vic.py::test_forest_snow_free_energy_balance_is_vics',
    'test_frozen_soil.py::test_frost_runs_close_and_carry_their_ice',
    'test_frozen_soil.py::test_ice_forms_and_stays_between_zero_and_the_moisture',
    'test_frozen_soil.py::test_the_option_is_inert_when_off',
    'test_frozen_soil.py::test_it_runs_in_water_balance_mode',
    'test_snow_energy_terms.py::test_gangotri_snow_rows_are_vic51s',
    'test_snow_energy_terms.py::test_frozen_soil_under_and_after_snow_is_vics',
    'test_snow_energy_terms.py::test_stehekin_full_energy_snow_rows_are_vic51s',
    'test_spatial_snow.py::test_spatial_snow_off_changes_nothing',
    'test_spatial_snow.py::test_the_cover_fraction_is_a_fraction_and_actually_goes_partial',
    'test_state_carry.py::',
    'test_tfallback.py::test_unsolvable_step_stops_at_vics_step_and_cell_and_true_counts_it',
    'test_refused_zero.py::test_the_treeline_changes_what_vic_classic_writes',
    'test_refused_zero.py::test_treeline_off_for_a_cell_with_july_tavg_minus_999_is_the_base_run',
)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if any(p in item.nodeid for p in SLOW):
            item.add_marker(pytest.mark.slow)


def pytest_report_header(config):
    return ['torchvic reference: %s' % _paths.REFERENCE,
            'torchvic sample data: %s' % _paths.SAMPLE_DATA,
            'torchvic samples: %s' % _paths.SAMPLES,
            'torchvic oracles: %s' % _paths.ORACLE_DIR]


import pytest                                               # noqa: E402
import torch                                                # noqa: E402


from torchvic.core import version as _core_version         # noqa: E402

#: The core a fresh process starts with (torchvic/core/version.py).
_DEFAULT_CORE = _core_version.CORE


@pytest.fixture(scope='module', autouse=True)
def _module_vic_core(request):
    """`pytestmark = pytest.mark.vic_core('5.0.1')` runs the module's own
    fixtures (which may build a case) on that core as well as its tests."""
    m = request.node.get_closest_marker('vic_core')
    if m is not None:
        _core_version.set_core(m.args[0])
    yield


@pytest.fixture(autouse=True)
def _restore_core(request):
    """Loading a case selects its core for the whole process
    (core_version.set_core); put the default back after every test so a
    test that ran a 5.0.1 case does not change the next one's physics.
    The default, not the value at setup: a module-scoped fixture that
    loads a case is set up before this one. A test or module marked
    vic_core(...) starts on that core."""
    m = request.node.get_closest_marker('vic_core')
    if m is not None:
        _core_version.set_core(m.args[0])
    yield
    _core_version.CORE = _DEFAULT_CORE


#: Test modules may call torch.set_default_device / set_default_dtype at
#: import (the old standalone scripts did). That must not reach other
#: modules: each module's import is bracketed, the process default is put
#: back, and the module's own default is applied only to its own tests.
_BASE = {}
_MODULE_DEFAULTS = {}


def _current_defaults():
    return torch.empty(0).device, torch.get_default_dtype()


def _apply_defaults(dev, dtype):
    torch.set_default_device(dev)
    torch.set_default_dtype(dtype)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    is_mod = isinstance(collector, pytest.Module)
    if is_mod:
        _BASE.setdefault('d', _current_defaults())
        _apply_defaults(*_BASE['d'])
    yield
    if is_mod:
        _MODULE_DEFAULTS[str(collector.path)] = _current_defaults()
        _apply_defaults(*_BASE['d'])


@pytest.fixture(autouse=True)
def _restore_torch_defaults(request):
    """Run each test with its module's import-time default device/dtype
    (none set = the process's own), and restore the process default after."""
    base = _BASE.get('d') or _current_defaults()
    mod = getattr(request.node, 'path', None)
    _apply_defaults(*_MODULE_DEFAULTS.get(str(mod), base))
    yield
    _apply_defaults(*base)
