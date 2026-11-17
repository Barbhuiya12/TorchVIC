"""Progress bars observe the run and change nothing.

`torchvic.progress` puts a tqdm bar over the time steps of `basin.simulate`.
A bar is supposed to be pure observation, and in a differentiable model that
is a claim about the GRADIENT as well as the value -- a wrapper that touched
a tensor, or reordered an operation, could leave the forward answer intact
and move the derivative. So both are asserted bit-identical, on and off.
"""
import io
import os

import pytest
import torch

from torchvic.model import basin
from torchvic.io import build as image_run
from torchvic import progress
from torchvic.io.global_param import GlobalParam
from torchvic.io.domain import ImageDomain
from torchvic.io.forcing import ImageForcing
from torchvic.io.params import ImageParams

import _paths                                           # noqa: E402
HERE = _paths.REPO
SD = _paths.SAMPLE_DATA
WG = os.path.join(SD, 'image', 'WalnutGulch', 'parameters')
GLOBAL = os.path.join(WG, 'global_param.WalnutGulch.L2015.txt')

#: The CPU, explicitly: bit-identity is a CPU statement in this project.
DEV = torch.device('cpu')
DT = torch.float64

needs_data = pytest.mark.skipif(not os.path.exists(GLOBAL),
                                reason='VIC sample data not present')


def _case(days=1):
    import pandas as pd
    gp = GlobalParam(GLOBAL, sample_data=SD)
    dom = ImageDomain(gp.domain_file, names=gp.domain_vars)
    par = ImageParams(
        os.path.join(WG, 'params.WalnutGulch.L2015.0.0625_deg.nc'), dom)
    steps = gp.get_int('MODEL_STEPS_PER_DAY')
    fc = ImageForcing(gp.forcing_prefix, dom, force_type=gp.force_type)
    fields, times = fc.read(dtype=DT, device=DEV)
    n = days * steps
    fields = {k: v[:n] for k, v in fields.items()}
    dates = [pd.Timestamp(x).to_pydatetime() for x in times[:n]]
    batch = image_run.ImageBatch(dom, dates, dtype=DT, device=DEV,
                                 elevation=image_run.band_mean_elevation(par))
    atm = image_run.build_atmosphere(fields, batch)
    atm.pop('_sources')
    sp = image_run.build_soil(
        par, DT, DEV,
        runoff_steps_per_day=gp.get_int('RUNOFF_STEPS_PER_DAY'),
        model_steps_per_day=steps)
    mosaic = image_run.build_mosaic(par, DT, DEV)
    init = torch.as_tensor(par.soil['init_moist'], dtype=DT,
                           device=DEV).T.contiguous()
    return dict(batch=batch, atm=atm, soil_params=sp, mosaic=mosaic,
                wind_h=float(gp.get('WIND_H')), init_moist=init,
                steps_per_day=steps)


def _params(pcf=1.0, grad=False):
    t = lambda v, g=False: torch.tensor(v, dtype=DT, device=DEV,
                                        requires_grad=g)
    return dict(pcf=t(pcf, grad), t_max_snow=t(0.5), t_min_rain=t(-0.5),
                rs=t(120.0), et_beta=t(1.0), new_snow_albedo=t(0.85),
                alb_accum_a=t(0.94), alb_accum_b=t(0.58),
                alb_thaw_a=t(0.82), alb_thaw_b=t(0.46), uh_shape=t(2.5),
                uh_scale=t(2.0), lai_scale=t(1.0), rmin_scale=t(1.0),
                w_crit=t(0.7), w_wilt=t(0.3))


# --- the bar changes nothing --------------------------------------------

@needs_data
def test_outputs_are_bit_identical_with_the_bar_on_and_off():
    kw = _case()
    with torch.no_grad():
        off = basin.simulate(p=_params(), progress=False, **kw)
        on = basin.simulate(p=_params(), progress=True, **kw)
    assert set(off) == set(on)
    for k in off:
        if torch.is_tensor(off[k]):
            assert torch.equal(off[k], on[k]), k


@needs_data
def test_the_gradient_is_bit_identical_with_the_bar_on_and_off():
    """The claim that matters in a differentiable model."""
    kw = _case()
    grads = []
    for prog in (False, True):
        p = _params(pcf=1.1, grad=True)
        out = basin.simulate(p=p, progress=prog, **kw)
        (g,) = torch.autograd.grad(out['runoff'].sum() + out['et'].sum(),
                                   [p['pcf']])
        grads.append(g)
    assert torch.isfinite(grads[0]).all()
    assert torch.equal(grads[0], grads[1]), (
        'bar on/off moved the gradient: %r vs %r' % (grads[0], grads[1]))


@needs_data
def test_the_bar_is_off_unless_asked_for(capsys):
    """Hundreds of tests call simulate; none of them should draw a bar."""
    kw = _case()
    with torch.no_grad():
        basin.simulate(p=_params(), **kw)
    err = capsys.readouterr().err
    assert 'forward' not in err and 'step/s' not in err, err[:200]


# --- the stand-in -------------------------------------------------------

def test_disabled_bar_is_a_transparent_iterator():
    b = progress.bar(range(5), total=5, enable=False)
    assert list(b) == [0, 1, 2, 3, 4]
    b.set_postfix(loss=1.0)
    b.set_description('x')
    b.update(2)
    b.close()


def test_missing_tqdm_degrades_to_the_stand_in(monkeypatch):
    """The python 3.11 environment does not ship tqdm; a run there must
    still run."""
    import builtins
    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == 'tqdm' or name.startswith('tqdm.'):
            raise ImportError('no tqdm here')
        return real(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', fake)
    monkeypatch.setattr(progress, '_WARNED', [True])     # no warning noise
    b = progress.bar(range(3), total=3, enable=True)
    assert isinstance(b, progress._Null)
    assert list(b) == [0, 1, 2]


def test_enabled_bar_counts_to_the_total():
    pytest.importorskip('tqdm')
    buf = io.StringIO()
    b = progress.bar(range(7), total=7, desc='t', enable=True, file=buf,
                     mininterval=0)
    assert list(b) == list(range(7))
    assert '7/7' in buf.getvalue()
