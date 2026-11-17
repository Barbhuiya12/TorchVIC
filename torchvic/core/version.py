"""Which VIC release's core physics TorchVIC reproduces.

    from torchvic import core_version
    core_version.set_core('5.0.1')

'5.1.0' (the default since 0.4.8) is the official VIC 5.1.0 (December 2021), which
VIC-WUR is built on; tests/test_vic51_vs_vic.py runs the 5.1.0 core against it.
'5.0.1' (the default before 0.4.8) is VIC 5.0.1, verified against its image
driver (tests/test_wb_vs_vic.py); `set_core('5.0.1')`, `torchvic run --core
5.0.1` or `core: 5.0.1` in a study file selects it. Each difference between the
two is switched where it lives, with the release's own reference.
'vicwur3' is the vic_run of VIC-WUR 3.0.0 (<data root>/build/vicwur3/src_tree):
the 5.1.0 core (at_least('5.1.0') is true) with the places where VIC-WUR 3.0's
VIC differs from the 5.1.0 release, each behind `vicwur3()` where it lives
(docs/connectors/vicwur.md; tests/test_vicwur_core.py runs it against
VIC-WUR 3.0's G0a/G0b). Only an explicit set_core('vicwur3') selects it.
"""

CORES = ('5.0.1', '5.1.0', 'vicwur3')
CORE = '5.1.0'


def set_core(version):
    """Select the core physics for every later run in this process."""
    global CORE
    version = str(version)
    if version not in CORES:
        raise ValueError('VIC core %r: one of %s' % (version, ', '.join(CORES)))
    CORE = version
    return CORE


def at_least(version):
    """True when the selected core is `version` or later."""
    return CORES.index(CORE) >= CORES.index(str(version))


def vicwur3():
    """True when the selected core is VIC-WUR 3.0.0's vic_run."""
    return CORE == 'vicwur3'
