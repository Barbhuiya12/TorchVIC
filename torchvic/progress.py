"""Progress bars for long runs, and nothing else.

The forward pass is a Python loop over time steps, and a long run over a
large domain says nothing for minutes or hours without one; the only way to
tell a slow run from a hung one was to go and look at the CPU time of the
process.

`bar()` wraps an iterable in a `tqdm` bar when progress is asked for and
`tqdm` is installed, and in a do-nothing stand-in otherwise. Three rules it
keeps, each for a reason:

* **Off unless asked for.** `basin.simulate` is called by hundreds of tests;
  a bar per call would bury their output. Library functions take
  `progress=False` and only the scripts turn it on.
* **Observation only.** A bar changes no value. `tests/test_progress.py`
  asserts the model's outputs are bit-identical with the bar on and off.
* **Missing `tqdm` is not an error.** The python 3.11 environment does not
  ship it; a run there must still run, silently rather than not at all.

When stderr is not a terminal -- a run under `nohup` writing to a log file,
which is how long runs are usually launched -- the bar redraws
only every thirty seconds and uses ASCII, so the log stays readable with
`tail` rather than filling with a redraw per step.
"""
import sys
import warnings

_WARNED = []


class _Null(object):
    """The interface of a tqdm bar, doing nothing."""

    def __init__(self, iterable=None, **kw):
        self._it = iterable

    def __iter__(self):
        return iter(self._it if self._it is not None else ())

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def update(self, n=1):
        pass

    def set_postfix(self, *a, **kw):
        pass

    def set_postfix_str(self, *a, **kw):
        pass

    def set_description(self, *a, **kw):
        pass

    def set_description_str(self, *a, **kw):
        pass

    def refresh(self):
        pass

    def close(self):
        pass

    @staticmethod
    def write(msg, file=None):
        print(msg, file=file or sys.stdout)


def _tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        if not _WARNED:
            _WARNED.append(True)
            warnings.warn('progress was asked for but tqdm is not installed '
                          '(pip install tqdm); running without a bar')
        return None


def bar(iterable=None, total=None, desc=None, enable=False, unit='it',
        leave=True, position=None, **kw):
    """A tqdm bar over `iterable` if `enable` and tqdm exists, else a no-op.

    Extra keyword arguments go to tqdm unchanged.
    """
    if not enable:
        return _Null(iterable)
    tqdm = _tqdm()
    if tqdm is None:
        return _Null(iterable)
    tty = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
    opts = dict(total=total, desc=desc, unit=unit, leave=leave,
                dynamic_ncols=tty, ascii=not tty,
                mininterval=0.5 if tty else 30.0,
                maxinterval=10.0 if tty else 120.0,
                smoothing=0.1, file=sys.stderr)
    if position is not None:
        opts['position'] = position
    opts.update(kw)
    return tqdm(iterable, **opts)


def write(msg, enable=False):
    """Print `msg` without breaking an active bar."""
    tqdm = _tqdm() if enable else None
    if tqdm is None:
        print(msg)
    else:
        tqdm.write(msg)
