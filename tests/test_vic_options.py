"""The drift guard: VIC may not gain a global-file option we do not classify.

`torchvic.io` decides what a global
file is asking for by consulting a list of VIC's options, and the whole value
of that list is that it matches VIC. A list that has fallen behind does not
announce itself -- the key it has never heard of simply stops being reported,
and a run that switched on lakes or blowing snow looks exactly like a run that
did not.

So these tests re-run the extraction against pristine VIC 5.1.0's C source
(TORCHVIC_REFERENCE_DIR) on every invocation and fail when it disagrees with
what is recorded. What would have
been seen had the step been wrong: `test_every_vic_option_is_classified` names
the key VIC has and we have not, rather than leaving it to be discovered when
somebody's basin comes out dry.

The three statuses are load-bearing and the test treats them as such:
`implemented` is a claim that the option changes the run, so the test checks
that the keys claiming it are the ones the image reader genuinely consumes,
not merely the ones it parses.
"""
import json
import os
import sys

import pytest

import _paths

HERE = _paths.REPO
sys.path.insert(0, _paths.TOOLS)
os.environ.setdefault('TORCHVIC_REFERENCE_DIR', _paths.REFERENCE)

C_SOURCE = os.path.join(_paths.VIC51_TREE, 'drivers', 'image', 'src',
                        'get_global_param.c')
INVENTORY = os.path.join(HERE, 'torchvic', 'data',
                         'vic5_global_options.json')

extract = pytest.importorskip('extract_vic_options')

pytestmark = pytest.mark.skipif(
    not os.path.exists(C_SOURCE),
    reason='pristine VIC 5.1.0 source not present; the drift guard needs it')


@pytest.fixture(scope='module')
def extracted():
    """The inventory rebuilt from VIC's source right now, not from disk."""
    inv, undeclared = extract.inventory()
    return inv, undeclared


def test_every_vic_option_is_classified(extracted):
    # The guard itself. A key VIC dispatches on and STATUS does not mention is
    # a key TorchVIC would treat as a typo rather than as an unimplemented
    # physics, which is the failure this guard exists to prevent.
    inv, undeclared = extracted
    assert undeclared == [], (
        'VIC recognises %d global-file option(s) that '
        'tools/extract_vic_options.py does not classify: %s. Add each to '
        'STATUS with an honest status and re-run the script.'
        % (len(undeclared), ', '.join(undeclared)))


def test_classification_does_not_contradict_vic(extracted):
    # A key VIC refuses cannot be harmless here: VIC would stop on the file,
    # so accepting it would make TorchVIC read a set-up VIC will not run.
    inv, _ = extracted
    assert extract.inconsistencies(inv) == []


def test_the_extraction_finds_what_a_hand_read_finds(extracted):
    # Guards the parser rather than the classification. These are branches
    # with shapes that broke an earlier version of it: an alias pair sharing
    # one body, a key whose only statement is a log_err, and a key recognised
    # here but parsed in shared_image/parse_output_info.c.
    inv, _ = extracted
    o = inv['options']
    assert o['NOFLUX']['status'] == o['NO_FLUX']['status']
    assert o['ATMOS_STEPS_PER_DAY']['rejected_by'] == ['image']
    assert o['OUTPUT_FORCE']['rejected_by'] == ['image']
    assert o['OUTVAR']['parsed_elsewhere'] is True
    assert o['FULL_ENERGY']['boolean'] is True
    assert o['BASEFLOW']['boolean'] is False
    # `flgstr` comparisons are option *values*, not keys, and must not be
    # collected: DENS_SNTHRM is a value of SNOW_DENSITY.
    assert 'DENS_SNTHRM' not in o
    assert 'FROM_VEGPARAM' not in o


def test_the_shipped_inventory_matches_the_source(extracted):
    # The inventory is checked in, so it can go stale while the source moves.
    # Re-running the script is the fix and the message says so.
    inv, _ = extracted
    if not os.path.exists(INVENTORY):
        pytest.skip('torchvic/data/vic5_global_options.json not generated')
    on_disk = json.load(open(INVENTORY))
    assert on_disk['options'] == inv['options'], (
        'torchvic/data/vic5_global_options.json disagrees with VIC; re-run '
        'tools/extract_vic_options.py')


def test_the_reader_uses_the_inventory_when_it_is_present():
    from torchvic.io import global_param as vic_image
    if not os.path.exists(INVENTORY):
        pytest.skip('torchvic/data/vic5_global_options.json not generated')
    # Every key the reader treats as known or as a known-unimplemented boolean
    # must be one VIC actually has. The old hand-written set carried LOG_LVL
    # (a compile-time macro) and AOUT_STEP (a VIC 4 key), neither of which VIC
    # 5 dispatches on; loading from the inventory is what removed them.
    assert vic_image.KNOWN_OPTIONS <= vic_image.VIC_OPTIONS
    assert vic_image.NOT_IMPLEMENTED <= vic_image.VIC_OPTIONS
    assert 'LOG_LVL' not in vic_image.VIC_OPTIONS
    assert 'AOUT_STEP' not in vic_image.VIC_OPTIONS


def test_the_reader_still_works_without_the_evidence_file(tmp_path):
    # The package is importable and usable when the inventory is missing,
    # which is the whole reason the hand-written sets are kept.
    from torchvic.io import global_param as vic_image
    missing = str(tmp_path / 'not-generated.json')
    assert vic_image.load_inventory(missing) is None
    known, boolean, notes = vic_image._option_sets(None)
    assert known == vic_image._FALLBACK_KNOWN
    assert boolean == vic_image._FALLBACK_NOT_IMPLEMENTED
    assert notes == {}


def test_an_option_is_never_implemented_merely_because_it_is_parsed(extracted):
    # The honesty check. Everything claiming `implemented` must be a key the
    # image reader or its runner actually consumes, by name, in our source.
    inv, _ = extracted
    claimed = sorted(k for k, e in inv['options'].items()
                     if e['status'] == 'implemented')
    sources = []
    # Both readers. The classic driver's keys -- SOIL, VEGLIB, VEGPARAM, the
    # FORCE* family -- are consumed by name in vic_classic.py and nowhere
    # else, and a list that omitted it would refuse them as unread.
    # torchvic/io/history.py reads the output section (OUTFILE, AGGFREQ, ...).
    for rel in ('torchvic/io/global_param.py', 'torchvic/io/domain.py',
                'torchvic/io/params.py', 'torchvic/io/forcing.py',
                'torchvic/io/classic.py', 'torchvic/io/build.py',
                'torchvic/case.py', 'torchvic/io/history.py'):
        sources.append(open(os.path.join(HERE, rel)).read())
    blob = '\n'.join(sources)
    for key in claimed:
        if inv['options'][key].get('note', '').startswith('alias of '):
            # An alias is consumed where its canonical key is: NO_FLUX is
            # folded into NOFLUX by options.thermal, which case.image_options
            # reads; the extraction test pins alias pairs to equal status.
            continue
        assert "'%s'" % key in blob, (
            '%s is classified implemented but no TorchVIC source reads it by '
            'name' % key)
    # And the count is small on purpose. If this rises, it is because more
    # options were genuinely honoured, and the number belongs in the commit
    # that honoured them.
    # Thirty-four, from nine: the classic reader (vic_classic.py) honours
    # SOIL, VEGLIB, VEGPARAM, the FORCE* family, GRID_DECIMAL, ROOT_ZONES and
    # the VEGPARAM_*/VEGLIB_FCAN layout flags by name. SPATIAL_SNOW joined
    # them when calc_snow_coverage.c and advected_sensible_heat.c were
    # implemented -- vic_classic.py reads the flag, the soil reader adds the
    # max_snow_distrib_slope column, and basin.simulate runs the physics.
    # FROZEN_SOIL joined on the same terms: the reader honours the flag
    # and basin.simulate runs the quick-flux frozen soil.
    # QUICK_FLUX, IMPLICIT, EXP_TRANS, NODES, NOFLUX, NO_FLUX and
    # QUICK_SOLVE joined with the finite-difference profile: case.
    # image_options reads all seven from the global file (NO_FLUX folded
    # into NOFLUX by options.thermal) and basin.simulate solves the
    # profile at every trial surface temperature (torchvic.core.soiltemp),
    # verified against VIC's own outputs on himal_EB_510_3h_QUICK_FLUX=FALSE.
    # BLOWING and its five sub-options came in with torchvic.core.blowing:
    # case.image_options reads all six from the global file and passes
    # them to basin.simulate, which refuses a run without the vegetation
    # file's sigma_slope, lag_one and fetch.
    # CARBON and CANOPY_LAYERS joined when the carbon cycle reached the step
    # loop: vic_classic.py reads both by name, the library's seven
    # VEGLIB_PHOTO columns are parsed, basin.simulate(carbon=True) runs
    # canopy_assimilation at the point surface_fluxes.c:732 does, and
    # daily_forcing derives the CATM/PAR/FDIR/coszen VIC 5 demands as
    # forcing columns. RC_MODE RC_PHOTO joined with the transpiration:
    # canopy_evap.c's calc_rc_ps replaces Jarvis's resistance in both cases.
    # SHARE_LAYER_MOIST joined with VEGLIB_PHOTO. Both cases of
    # canopy_evap.c had been transcribed and tested for a while, but no
    # reader consumed the flag, so a file asking for CASE 2 alone got
    # CASE 1 anyway -- implemented and unreachable, which is worse than
    # unimplemented because nothing says so. vic_classic.py now reads it
    # (TRUE when the key is absent, as initialize_options.c:67 sets it) and
    # basin.simulate threads it to vegetation.transpiration.
    # VEGLIB_PHOTO joined last: vic_classic.py reads the flag by name
    # (`self.veglib_photo`) and read_veglib widens each library line by the
    # seven columns read_veglib.c:162-192 adds when it is on, choosing
    # between MaxETransport and CO2Specificity on Ctype exactly as the C
    # does; the image parameter reader picks the same eight variables up
    # from the parameter file. It had been classified unsupported with the
    # note "photosynthesis parameters are not read", which stopped being
    # true when CARBON did, and an inventory that says something false
    # about a file people hand us is the failure mode this whole table
    # exists to prevent.
    # CONSTANTS: GlobalParam.constants() reads the file, applies
    # VEG_LAI_WATER_FACTOR and refuses any other key by name, so a constants
    # file is never half-applied in silence.
    # The history output section joined with torchvic.io.history: OUTFILE,
    # OUTVAR, AGGFREQ, HISTFREQ, COMPRESS and OUT_FORMAT are read by name
    # in parse_streams, OUT_TIME_UNITS and RESULT_DIR in write_record and
    # write_history, and `torchvic run` writes the streams.
    # Aggregating VIC's own per-step history reproduces VIC's aggregated
    # history to the bit for every AGG_TYPE under seven AGGFREQs, and the
    # written file is VIC's (tests/test_output_aggregation.py).
    # basin.simulate still returns every output at the model step: these
    # decide what the history files hold, which is all they decide in VIC.
    # Nothing stays refused: ARNO_PARAMS and NIJSSEN2001_BASEFLOW
    # (read as BASEFLOW in GlobalParam._parse), COMPUTE_TREELINE and
    # JULY_TAVG_SUPPLIED (the classic driver's treeline, core/treeline.py)
    # and OUTPUT_FORCE (VIC 4.2.d's forcing files, history.write_forcing_42d)
    # joined; ATMOS_STEPS_PER_DAY is `ignored`: validated, no effect.
    # Each was confirmed read by name before this list was extended; the
    # per-key check above is what actually enforces that, this list is the
    # human's sign-off.
    # The image driver's options table (torchvic/io/options.py) added the
    # model step, the simulation window (START*/END*/NRECS), CALENDAR,
    # and WIND_H, each read by name in torchvic/case.py
    # and enforced by case.load; SPATIAL_SNOW left, because the image driver
    # refuses it until it is verified against VIC's image driver.
    # BASEFLOW joined when NIJSSEN2001 did: case.load reads the flag by
    # name and build_soil converts the file's d1..d4 to ARNO's Ds, Dsmax,
    # Ws, c (vic_init.c:815-829; read_soilparam.c:693-707) before the
    # column runs, for the classic parameters as for the image ones.
    # GRND_FLUX_TYPE joined when GF_406 did: case.image_options reads the
    # flag by name (GF_410 when the file is silent, as
    # initialize_options.c:39 sets it) and basin.simulate threads it to
    # the surface energy balance, where GF_406 is the surface term alone
    # and GF_410 the two-term Liang flux (func_surf_energy_bal.c:385-403).
    # CLOSE_ENERGY joined when the iteration did: case.image_options
    # reads the flag by name (FALSE when the file is silent, as
    # initialize_options.c:31 leaves it) and basin.simulate iterates the
    # ground/canopy balances to closure with the solved canopy air
    # temperature (calc_atmos_energy_bal.c:73-102,
    # surface_fluxes.c:442-720).
    assert claimed == [
        'AGGFREQ',
        'ALB_SRC',
        'ARNO_PARAMS',
        'BASEFLOW',
        'BLOWING',
        'BLOWING_CALC_PROB',
        'BLOWING_FETCH',
        'BLOWING_SIMPLE',
        'BLOWING_SPATIAL_WIND',
        'BLOWING_VAR_THRESHOLD',
        'CALENDAR',
        'CANOPY_LAYERS',
        'CARBON',
        'CLOSE_ENERGY',
        'COMPRESS',
        'COMPUTE_TREELINE',
        'CONSTANTS',
        'DOMAIN',
        'DOMAIN_TYPE',
        'ENDDAY',
        'ENDMONTH',
        'ENDYEAR',
        'EXP_TRANS',
        'FCAN_SRC',
        'FORCEDAY',
        'FORCEMONTH',
        'FORCESEC',
        'FORCEYEAR',
        'FORCE_FORMAT',
        'FORCE_STEPS_PER_DAY',
        'FORCE_TYPE',
        'FORCING1',
        'FORCING2',
        'FROZEN_SOIL',
        'FULL_ENERGY',
        'GRID_DECIMAL',
        'GRND_FLUX_TYPE',
        'HISTFREQ',
        'IMPLICIT',
        'JULY_TAVG_SUPPLIED',
        'LAI_SRC',
        'MODEL_STEPS_PER_DAY',
        'NIJSSEN2001_BASEFLOW',
        'NODES',
        'NOFLUX',
        'NO_FLUX',
        'NRECS',
        'OUTFILE',
        'OUTPUT_FORCE',
        'OUTVAR',
        'OUT_FORMAT',
        'OUT_TIME_UNITS',
        'PARAMETERS',
        'QUICK_FLUX',
        'QUICK_SOLVE',
        'RC_MODE',
        'RESULT_DIR',
        'ROOT_ZONES',
        'RUNOFF_STEPS_PER_DAY',
        'SHARE_LAYER_MOIST',
        'SOIL',
        'STARTDAY',
        'STARTMONTH',
        'STARTSEC',
        'STARTYEAR',
        'TFALLBACK',  # case.image_options reads it; core/tfallback.py
        'VEGLIB',
        'VEGLIB_FCAN',
        'VEGLIB_PHOTO',
        'VEGPARAM',
        'VEGPARAM_ALB',
        'VEGPARAM_FCAN',
        'VEGPARAM_LAI',
        'WIND_H']


def test_unsupported_options_are_announced_not_dropped(tmp_path):
    # The property that must survive every change to how the sets are built:
    # a global file that turns on something we do not have says so.
    from torchvic.io.global_param import GlobalParam
    p = tmp_path / 'global.txt'
    p.write_text('\n'.join([
        '# a global file exercising each announcement rule',
        'MODEL_STEPS_PER_DAY 24',
        'LAKES        TRUE',        # unsupported boolean, on
        'BLOWING      FALSE',       # unsupported boolean, off
        'INIT_STATE   state.nc',  # unsupported, no off position
        'NOT_A_VIC_OPTION  1',      # VIC has never heard of this
    ]) + '\n')
    g = GlobalParam(str(p))
    assert 'LAKES TRUE' in g.unsupported
    assert 'INIT_STATE' in g.unsupported
    assert 'NOT_A_VIC_OPTION' in g.unsupported
    assert g.unknown == ['NOT_A_VIC_OPTION']
    # An option explicitly switched off asks for nothing, and reporting it
    # would bury the lines that matter.
    assert not any(s.startswith('BLOWING') for s in g.unsupported)
