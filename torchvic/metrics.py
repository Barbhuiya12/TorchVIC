"""Goodness-of-fit metrics for reporting a run against observations.

Every metric here takes `sim` and `obs` of the same shape, reduces along
`dim` (time by default), ignores any position where `obs` is not finite, and
returns a tensor with `dim` removed -- so a [time, n] simulation against a
[time, 1] observation gives one number per column.

The metrics themselves:

* **KGE**, Gupta et al. (2009), *Decomposition of the mean squared error and
  NSE performance criteria*, J. Hydrol. 377:80-91 --
  ``KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)`` with `r` the
  correlation, `alpha = sd(sim)/sd(obs)` the variability ratio and
  `beta = mean(sim)/mean(obs)` the bias ratio. 1 is perfect. Unlike NSE it
  does not reward a simulation for merely tracking the mean, and its three
  components can be read separately, which is why it is the modern default.
* **KGE'**, Kling et al. (2012), J. Hydrol. 424:264-277 -- the same with
  `gamma = CV(sim)/CV(obs)` in place of `alpha`, so the variability term is
  not confounded by a bias in the mean.
* **NSE**, Nash and Sutcliffe (1970) -- 1 minus the ratio of the squared
  error to the variance of the observations. 0 means no better than the
  observed mean.
* **log-NSE** -- NSE on log-transformed values, the usual way to weight low
  flows; `eps` is added before the log and defaults to a hundredth of the
  mean observation, as recommended by Pushpalatha et al. (2012),
  J. Hydrol. 420:171-182.
* **MSE, RMSE, MAE, PBIAS, r** -- the plain ones, with PBIAS in per cent and
  positive meaning the simulation is too high.

`ORIENTATION` says which way is better: +1 larger, -1 smaller, 0 nearest
zero.
"""
import math

import torch


def _masked(sim, obs, dim):
    """(sim, obs, mask, n) with non-finite observations zeroed out."""
    mask = torch.isfinite(obs)
    if not mask.all():
        sim = torch.where(mask, sim, torch.zeros_like(sim))
        obs = torch.where(mask, obs, torch.zeros_like(obs))
    m = mask.to(sim.dtype)
    n = m.sum(dim).clamp(min=1.0)
    return sim, obs, m, n


def _moments(sim, obs, dim, eps=1e-12):
    sim, obs, m, n = _masked(sim, obs, dim)
    mu_s = sim.sum(dim) / n
    mu_o = obs.sum(dim) / n
    ds = (sim - mu_s.unsqueeze(dim)) * m
    do = (obs - mu_o.unsqueeze(dim)) * m
    var_s = (ds * ds).sum(dim) / n
    var_o = (do * do).sum(dim) / n
    cov = (ds * do).sum(dim) / n
    return mu_s, mu_o, var_s, var_o, cov, m, n


def pearson_r(sim, obs, dim=0, eps=1e-12):
    """Pearson correlation, -1 to 1."""
    _, _, var_s, var_o, cov, _, _ = _moments(sim, obs, dim)
    return cov / (torch.sqrt(var_s.clamp(min=eps) * var_o.clamp(min=eps)))


def kge(sim, obs, dim=0, variant='2009', eps=1e-12, components=False):
    """Kling-Gupta efficiency. `variant` is '2009' (Gupta) or '2012' (Kling)."""
    mu_s, mu_o, var_s, var_o, cov, _, _ = _moments(sim, obs, dim)
    sd_s, sd_o = torch.sqrt(var_s.clamp(min=eps)), torch.sqrt(var_o.clamp(min=eps))
    r = cov / (sd_s * sd_o)
    beta = mu_s / torch.where(mu_o.abs() > eps, mu_o,
                              torch.full_like(mu_o, eps))
    if variant == '2012':
        # gamma = CV(sim)/CV(obs): the variability ratio with the bias taken out
        cv_s = sd_s / torch.where(mu_s.abs() > eps, mu_s,
                                  torch.full_like(mu_s, eps))
        cv_o = sd_o / torch.where(mu_o.abs() > eps, mu_o,
                                  torch.full_like(mu_o, eps))
        var_term = cv_s / torch.where(cv_o.abs() > eps, cv_o,
                                      torch.full_like(cv_o, eps))
    elif variant == '2009':
        var_term = sd_s / sd_o
    else:
        raise ValueError("kge variant is '2009' or '2012', not %r" % variant)
    ed = torch.sqrt(((r - 1.0) ** 2 + (var_term - 1.0) ** 2
                     + (beta - 1.0) ** 2).clamp(min=0.0))
    out = 1.0 - ed
    if components:
        return out, dict(r=r, alpha=var_term, beta=beta)
    return out


def kge_2012(sim, obs, dim=0):
    return kge(sim, obs, dim=dim, variant='2012')


def nse(sim, obs, dim=0, eps=1e-12):
    """Nash-Sutcliffe efficiency: 1 perfect, 0 no better than the mean."""
    sim, obs, m, n = _masked(sim, obs, dim)
    mu_o = obs.sum(dim) / n
    num = (((sim - obs) * m) ** 2).sum(dim)
    den = (((obs - mu_o.unsqueeze(dim)) * m) ** 2).sum(dim)
    return 1.0 - num / den.clamp(min=eps)


def log_nse(sim, obs, dim=0, eps=None, floor=1e-6):
    """NSE on log-transformed values, for low flows.

    `eps` defaults to a hundredth of the mean observation (Pushpalatha et
    al. 2012); both series are floored before the log so a zero or a small
    negative simulation cannot produce an infinite gradient.
    """
    _, obs_m, _, n = _masked(sim, obs, dim)
    if eps is None:
        eps = (obs_m.sum(dim) / n).abs().mean() * 0.01
        eps = torch.clamp(eps, min=floor)
    ls = torch.log(torch.clamp(sim + eps, min=floor))
    lo = torch.log(torch.clamp(obs + eps, min=floor))
    lo = torch.where(torch.isfinite(obs), lo, obs)      # keep the mask
    return nse(ls, lo, dim=dim)


def mse(sim, obs, dim=0):
    sim, obs, m, n = _masked(sim, obs, dim)
    return (((sim - obs) * m) ** 2).sum(dim) / n


def rmse(sim, obs, dim=0):
    return torch.sqrt(mse(sim, obs, dim=dim).clamp(min=0.0))


def mae(sim, obs, dim=0):
    sim, obs, m, n = _masked(sim, obs, dim)
    return (((sim - obs) * m).abs()).sum(dim) / n


def pbias(sim, obs, dim=0, eps=1e-12):
    """Per cent bias; positive means the simulation is too high."""
    sim, obs, m, n = _masked(sim, obs, dim)
    s, o = sim.sum(dim), obs.sum(dim)
    return 100.0 * (s - o) / torch.where(o.abs() > eps, o,
                                         torch.full_like(o, eps))


#: name -> (function, +1 if larger is better, -1 if smaller is better)
METRICS = {
    'kge': (kge, +1),
    'kge2012': (kge_2012, +1),
    'nse': (nse, +1),
    'lognse': (log_nse, +1),
    'r': (pearson_r, +1),
    'mse': (mse, -1),
    'rmse': (rmse, -1),
    'mae': (mae, -1),
    'pbias': (pbias, 0),          # best at zero, either sign is bad
}

#: What a perfect score is, for reporting.
PERFECT = {'kge': 1.0, 'kge2012': 1.0, 'nse': 1.0, 'lognse': 1.0, 'r': 1.0,
           'mse': 0.0, 'rmse': 0.0, 'mae': 0.0, 'pbias': 0.0}

ORIENTATION = {k: v[1] for k, v in METRICS.items()}


def metric(name):
    """The metric function by name."""
    try:
        return METRICS[name][0]
    except KeyError:
        raise KeyError('unknown metric %r; known: %s'
                       % (name, ', '.join(sorted(METRICS))))


def score(name, sim, obs, dim=0, **kw):
    """The metric `name` of `sim` against `obs`."""
    return metric(name)(sim, obs, dim=dim, **kw)
