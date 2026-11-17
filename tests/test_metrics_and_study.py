"""The metrics a run is reported with, and the study file that drives it."""
import os

import pytest
import torch

from torchvic import case, config, metrics  # noqa: E402

import _paths                                           # noqa: E402
WG = _paths.WG_FULL
D = torch.float64
CPU = torch.device('cpu')


def _series(n=120, seed=0):
    g = torch.Generator(device=CPU).manual_seed(seed)
    return 1.0 + 4.0 * torch.rand(n, generator=g, dtype=D, device=CPU)


@pytest.mark.parametrize('name', sorted(metrics.METRICS))
def test_every_metric_is_perfect_on_itself_and_worse_on_noise(name):
    obs = _series()
    g = torch.Generator(device=CPU).manual_seed(1)
    noisy = obs + 0.5 * torch.randn(obs.shape, generator=g, dtype=D, device=CPU)
    perfect = float(metrics.score(name, obs.clone(), obs))
    assert abs(perfect - metrics.PERFECT[name]) < 1e-9, \
        '%s is %g on an exact match, not %g' % (name, perfect,
                                                metrics.PERFECT[name])
    # and worse on noise, in the direction the metric points
    worse = float(metrics.score(name, noisy, obs))
    orient = metrics.ORIENTATION[name]
    if orient > 0:
        assert worse < perfect - 1e-6
    elif orient < 0:
        assert worse > perfect + 1e-6
    else:
        assert abs(worse) > 1e-6


def test_kge_components_are_the_published_decomposition():
    """Gupta et al. (2009): KGE = 1 - sqrt((r-1)^2+(a-1)^2+(b-1)^2)."""
    obs = _series()
    sim = 1.3 * obs + 0.4
    k, comp = metrics.kge(sim, obs, components=True)
    r, a, b = float(comp['r']), float(comp['alpha']), float(comp['beta'])
    assert abs(r - 1.0) < 1e-9                       # an affine map: r = 1
    assert abs(a - 1.3) < 1e-9                       # sd ratio is the slope
    expect = 1.0 - ((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2) ** 0.5
    assert abs(float(k) - expect) < 1e-12
    # the 2012 variant differs when there is a bias in the mean
    assert abs(float(metrics.kge(sim, obs, variant='2012')) - float(k)) > 1e-6


def test_missing_observations_are_ignored_and_gradients_stay_finite():
    obs = _series()
    holed = obs.clone()
    holed[::7] = float('nan')
    sim = (obs * 1.1).requires_grad_(True)
    for name in sorted(metrics.METRICS):
        v = metrics.score(name, sim, holed)
        assert torch.isfinite(v).all(), name
    metrics.kge(sim, holed).backward()
    assert torch.isfinite(sim.grad).all()
    # dropping the same points by hand gives the same score
    keep = torch.isfinite(holed)
    assert abs(float(metrics.nse(sim.detach()[keep], obs[keep]))
               - float(metrics.nse(sim.detach(), holed))) < 1e-10


def _study_file(tmp_path):
    if not os.path.exists(WG):
        pytest.skip('the Walnut Gulch global file is not present')
    import numpy as np
    c = case.load(WG, device=CPU, steps=48)
    out = c.run()
    for name in ('runoff', 'baseflow'):
        np.savetxt(str(tmp_path / ('obs_%s.txt' % name)),
                   out[name].detach().mean(-1).cpu().numpy())
    text = """
case:
  global_param: %s
  device: cpu
  steps: 48
observations:
  runoff: {file: obs_runoff.txt}
  baseflow: {file: obs_baseflow.txt}
report:
  metrics: [nse, kge, rmse]
""" % os.path.abspath(WG)
    p = tmp_path / 'study.yaml'
    p.write_text(text)
    return str(p), out


def test_one_file_carries_the_data_and_the_observations(tmp_path):
    path, out = _study_file(tmp_path)
    st = config.load(path)
    assert st.case.n_steps == 48
    assert set(st.observations) == {'runoff', 'baseflow'}
    assert st.metrics == ['nse', 'kge', 'rmse']
    # observations made from the same run score as perfect
    rep = st.report(st.case.run())
    for var in ('runoff', 'baseflow'):
        assert abs(rep[var]['nse'] - 1.0) < 1e-9
        assert abs(rep[var]['rmse']) < 1e-9
    # and a different run does not
    other = dict(out)
    other['runoff'] = out['runoff'] * 1.5
    assert config.load(path).report(other)['runoff']['nse'] < 1.0 - 1e-6


def test_the_study_file_overrides_reach_the_case(tmp_path):
    path, _ = _study_file(tmp_path)
    st = config.load(path, dtype='float32', core='5.0.1')
    assert st.case.dtype == torch.float32
    assert st.case.atm['tair'].dtype == torch.float32
    assert st.case.core == '5.0.1'
    with pytest.raises(KeyError, match='metric'):
        bad = tmp_path / 'bad.yaml'
        bad.write_text(open(path).read().replace('[nse, kge, rmse]',
                                                 '[nash]'))
        config.load(str(bad))
