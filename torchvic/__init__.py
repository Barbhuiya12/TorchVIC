"""TorchVIC: the VIC 5 land surface model (5.0.1 and 5.1.0 physics) in
PyTorch, run forward on a GPU or a CPU from native VIC 5 image-driver inputs.

    import torchvic
    case = torchvic.load_case('global_param.txt', device='cuda')
    out = torchvic.run(case)
    torchvic.write_history(out, case, result_dir='results')

`load_case` reads the global parameter file and everything it names (domain,
parameters, daily or sub-daily forcing), `run` executes the time loop
(torchvic.model.basin.simulate) without building an autograd graph, and
`write_history` writes the global file's OUTFILE streams as VIC 5's image
driver writes them. The command line does the same: `torchvic run`.
"""

__version__ = "0.4.9"

__all__ = ['__version__', 'default_device', 'load_case', 'load_study', 'run',
           'write_history', 'set_core', 'TorchVIC']


def __getattr__(name):
    # the coupled runtime, imported on first use (torchvic.coupling)
    if name == 'TorchVIC':
        from .coupling import TorchVIC
        return TorchVIC
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def default_device():
    """'cuda' when a GPU is visible, else 'cpu'."""
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def set_core(version):
    """Select the VIC core physics ('5.1.0', the default, '5.0.1' or 'vicwur3') for
    later runs."""
    from .core import version as core_version
    return core_version.set_core(version)


def load_case(global_param, **kwargs):
    """A torchvic.case.Case from a VIC 5 image-driver global parameter file;
    keyword arguments are torchvic.case.load's (device, dtype, core, steps,
    first_step, daily, no_energy, log, ...)."""
    from . import case
    return case.load(global_param, **kwargs)


def load_study(path, **kwargs):
    """A torchvic.config.Study from a study file (YAML or JSON)."""
    from . import config
    return config.load(path, **kwargs)


def run(case, **kwargs):
    """Run `case` forward and return its outputs ({name: tensor}).

    Runs under torch.inference_mode(): no graph is recorded, and tensors
    created inside it cannot later enter an autograd computation. Keyword
    arguments are passed to torchvic.model.basin.simulate."""
    import torch
    with torch.inference_mode():
        return case.run(**kwargs)


def write_history(out, case, result_dir=None, **kwargs):
    """Write the global file's OUTFILE streams for the run `out` of `case`
    (torchvic.io.history.write_history); returns the paths written."""
    return case.write_history(out, result_dir=result_dir, **kwargs)


# any CUDA graph capture first makes the generator state outside inference mode
from .core import graphsafe as _graphsafe  # noqa: E402,F401
