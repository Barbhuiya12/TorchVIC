"""Where the tests find the reference VIC builds, the sample data and the
oracle binaries. Each location can be moved with an environment variable:

    TORCHVIC_REFERENCE_DIR  pristine VIC trees as released, with their built
                            drivers: VIC-VIC.5.0.1 and VIC-5.1.0
                            (vic/drivers/image/vic_image.exe), VIC-VIC.4.2.d,
                            plus run/global_WG_full.txt and mtclim5
    TORCHVIC_SAMPLE_DATA    UW-Hydro's VIC_sample_data (image/WalnutGulch,
                            image/FindleyLake, classic/..., obs/...)
    TORCHVIC_SAMPLES        VIC's own samples/data (image/ and classic/
                            Stehekin)
    TORCHVIC_ORACLE_DIR     the compiled tests/oracle programs (default
                            build/oracle; conftest.py builds missing ones)
    TORCHVIC_DATA_ROOT      optional large inputs and reference runs that are
                            not distributed (CropSyst and VIC-WUR reference
                            builds, ISC domain files, ...); tests that need
                            them skip when it is unset. TORCHVIC_ALL_DATA is
                            read as a fallback.

A test whose data is absent skips and says which.
"""
import atexit
import os
import re
import shutil
import tempfile

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
TOOLS = os.path.join(REPO, 'tools')
ORACLE_SRC = os.path.join(TESTS, 'oracle')

#: Defaults are inside the repository and are not tracked (.gitignore);
#: tools/get_test_data.sh fills them.
REFERENCE = os.environ.get('TORCHVIC_REFERENCE_DIR',
                           os.path.join(REPO, 'reference'))
SAMPLE_DATA = os.environ.get('TORCHVIC_SAMPLE_DATA',
                             os.path.join(REPO, 'data', 'VIC_sample_data'))
SAMPLES = os.environ.get('TORCHVIC_SAMPLES',
                         os.path.join(REPO, 'data', 'samples'))
ORACLE_DIR = os.environ.get('TORCHVIC_ORACLE_DIR',
                            os.path.join(REPO, 'build', 'oracle'))

#: Root of the optional large data sets (see above); the default is an
#: untracked directory inside the repository, normally absent.
DATA_ROOT = (os.environ.get('TORCHVIC_DATA_ROOT')
             or os.environ.get('TORCHVIC_ALL_DATA')
             or os.path.join(REPO, 'data', 'extra'))


def data(*parts):
    """A path below DATA_ROOT."""
    return os.path.join(DATA_ROOT, *parts)


def env_path(names, rest):
    """The first of the environment variables `names` that is set, else
    DATA_ROOT/rest. The TORCHVIC_* name comes first, the older name after it."""
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return data(rest)


VIC501_TREE = os.path.join(REFERENCE, 'VIC-VIC.5.0.1', 'vic')
VIC51_TREE = os.path.join(REFERENCE, 'VIC-5.1.0', 'vic')
VIC42D_SRC = os.path.join(REFERENCE, 'VIC-VIC.4.2.d', 'src')
VIC501_IMAGE = os.path.join(VIC501_TREE, 'drivers', 'image', 'vic_image.exe')
VIC51_IMAGE = os.path.join(VIC51_TREE, 'drivers', 'image', 'vic_image.exe')
#: VIC's own system-test global files (identical in 5.1.0 and the tree these
#: tests were written against); they locate their data through
#: $test_data_dir, which is SAMPLES.
VIC_SYSTEM_TESTS = os.path.join(REFERENCE, 'VIC-5.1.0', 'tests', 'system')
STEHE_IMAGE_GLOBAL = os.path.join(VIC_SYSTEM_TESTS, 'global.image.STEHE.txt')
STEHE_CLASSIC_GLOBAL = os.path.join(VIC_SYSTEM_TESTS,
                                    'global.classic.STEHE.txt')


def oracle(name):
    """The path of a compiled oracle (tests/oracle/build_ref_<x>.sh)."""
    return os.path.join(ORACLE_DIR, name)


_TMP = []


def _tmpdir():
    if not _TMP:
        d = tempfile.mkdtemp(prefix='torchvic_tests_')
        _TMP.append(d)
        atexit.register(shutil.rmtree, d, True)
    return _TMP[0]


_SAMPLE_PATH = re.compile(r'\S*/VIC_sample_data(?=/)')


def reference_global(name):
    """REFERENCE/run/<name>, a global file written for the reference runs,
    with every path into VIC_sample_data moved to SAMPLE_DATA and RESULT_DIR
    and LOG_DIR moved into a scratch directory, so a run never writes into
    the reference tree. Returns the rewritten copy's path (or the original
    path, which does not exist, when the file is absent)."""
    src = os.path.join(REFERENCE, 'run', name)
    if not os.path.exists(src):
        return src
    tmp = os.path.join(_tmpdir(), 'run')
    os.makedirs(os.path.join(tmp, 'results'), exist_ok=True)
    os.makedirs(os.path.join(tmp, 'logs'), exist_ok=True)
    lines = []
    for ln in open(src).read().splitlines():
        key = ln.split()[0] if ln.split() else ''
        if key == 'RESULT_DIR':
            ln = 'RESULT_DIR %s/results/' % tmp
        elif key == 'LOG_DIR':
            ln = 'LOG_DIR %s/logs/' % tmp
        else:
            ln = _SAMPLE_PATH.sub(SAMPLE_DATA, ln)
        lines.append(ln)
    out = os.path.join(tmp, name)
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out


#: Walnut Gulch, 2008, the global file the VIC 5 reference runs were made
#: with (full energy balance, AGGFREQ NDAYS 1).
WG_FULL = reference_global('global_WG_full.txt')
WG_DOMAIN = os.path.join(SAMPLE_DATA, 'image', 'WalnutGulch', 'parameters',
                         'domain.WalnutGulch.0.0625_deg.nc')


def relocated_global(path, set_options=None):
    """`path`, a global file whose absolute paths may name the machine it
    was written on, as a scratch copy in which every absolute path that does
    not exist here, but passes through a directory named like the global
    file's own, points into the global file's directory instead; RESULT_DIR
    and LOG_DIR go to scratch. `set_options` {KEY: value} replaces those
    keys' lines. Returns `path` itself when it is absent."""
    if not os.path.exists(path):
        return path
    set_options = dict(set_options or {})
    here = os.path.dirname(os.path.abspath(path))
    mark = os.sep + os.path.basename(here) + os.sep
    tmp = os.path.join(_tmpdir(), 'relocated', os.path.basename(here))
    os.makedirs(os.path.join(tmp, 'out'), exist_ok=True)
    lines = []
    for ln in open(path).read().splitlines():
        key = ln.split()[0] if ln.split() else ''
        if key in ('RESULT_DIR', 'LOG_DIR'):
            ln = '%s %s/out/' % (key, tmp)
        elif key == 'STATENAME' and key not in set_options:
            # VIC's own test runner fills $state_dir; here it is scratch
            ln = '%s %s/out/states' % (key, tmp)
        elif key in set_options:
            ln = '%s %s' % (key, set_options[key])
        else:
            parts = ln.split()
            for tok in parts[1:]:
                if (tok.startswith(os.sep) and not os.path.exists(tok)
                        and mark in tok):
                    new = os.path.join(here, tok.split(mark, 1)[1])
                    ln = ln.replace(tok, new)
        lines.append(ln)
    out = os.path.join(tmp, os.path.basename(path))
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return out


#: A 100-cell VIC 5 image case (basin 37, India), used where a test needs a
#: real multi-cell domain: $TORCHVIC_B37_GLOBAL, relocated (relocated_global).
#: The file declares QUICK_FLUX FALSE and the classic driver's SNOW_BAND 5;
#: what the tests run on this case is the QUICK_FLUX path at the file's
#: elev (SNOW_BAND 5 is refused), so the copy says that.
B37_GLOBAL = relocated_global(os.environ.get(
    'TORCHVIC_B37_GLOBAL',
    os.path.join(REPO, 'data', 'India_basin37_Nandgaon', 'global_param.txt')),
    set_options={'QUICK_FLUX': 'TRUE', 'SNOW_BAND': 'FALSE'})
