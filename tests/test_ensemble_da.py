"""Ensemble data assimilation: EnKF, ETKF, LETKF, constraints, ledger."""
import os
import types

import pytest
import torch

import _paths                                           # noqa: E402
from torchvic import case                               # noqa: E402
from torchvic.ensemble import (Assimilator, Ensemble,   # noqa: E402
                               ForcingPerturbation, StateView, gaspari_cohn)

WG = _paths.WG_FULL
CPU = torch.device('cpu')
F64 = torch.float64
CUDA = torch.cuda.is_available()


# ---- a synthetic ensemble (no model run) ----------------------------------
def _fake(n, nc, nl=3, maxm=1e9, lat=None):
    ens = types.SimpleNamespace(n=n, nc=nc)
    ens.case = types.SimpleNamespace(
        mosaic=None,
        soil=types.SimpleNamespace(
            max_moist=torch.full((n * nc, nl), maxm, dtype=F64),
            resid_moist=torch.zeros(n * nc, nl, dtype=F64)),
        batch=types.SimpleNamespace(
            cell_area=torch.full((nc,), 1e6, dtype=F64),
            latitude=torch.linspace(30, 31, nc, dtype=F64) if lat is None
            else lat,
            longitude=torch.zeros(nc, dtype=F64)))
    return ens


def _state(moist, swe_m=None):
    r = moist.shape[0]
    z = torch.zeros(r, dtype=F64)
    ice = z.clone() if swe_m is None else swe_m.clone()
    snow = dict(ice=ice, pack_water=z.clone(), surf_water=z.clone(),
                swq_c=ice.clone(), depth=2 * ice, cold_content=-5 * ice)
    return dict(moist=moist.clone(), snow=snow)


def _linear_gaussian(n=4000, nc=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = torch.tensor([[3, 0, 0], [2, 2, 0], [1, 1, 2.]], dtype=F64)
    z = torch.randn(n * nc, 3, generator=g, dtype=F64)
    return torch.tensor([50., 60., 70.], dtype=F64) + z @ L.T


@pytest.mark.parametrize('method,rad', [('enkf', None), ('etkf', None),
                                        ('letkf', 1e6), ('enkf', 1e6)])
def test_analysis_matches_the_kalman_filter_of_the_sample(method, rad):
    n, nc = 4000, 5
    ens = _fake(n, nc)
    st = _state(_linear_gaussian(n, nc))
    A = Assimilator(ens, layers=(0, 1, 2), method=method, radius_km=rad)
    X0 = A.matrix(st)
    sig = 1.0
    obs = A.obs_soil_moisture(torch.arange(nc), torch.full((nc,), 58., dtype=F64),
                              sig, layers=[0, 1], scale=0.5)
    new, _ = A.analysis(st, obs)
    Xa = A.matrix(new)
    for c in range(nc):
        x = X0[:, c]
        mu, P = x.mean(0), torch.cov(x.T)
        H = torch.tensor([.5, .5, 0.], dtype=F64)
        s = H @ P @ H + sig ** 2
        K = P @ H / s
        mean_a = mu + K * (58. - H @ mu)
        Pa = P - torch.outer(K, H @ P)
        mtol = 0.15           # the other cells' observations add sample noise
        assert float((Xa[:, c].mean(0) - mean_a).abs().max()) < mtol
        # the stochastic filter carries sampling noise of ~sqrt(2/N) in the
        # covariance; the transform filters reproduce it much closer
        tol = 0.25 if method == 'enkf' else 0.06
        assert float((torch.cov(Xa[:, c].T) - Pa).abs().max()) < tol


def test_etkf_and_letkf_with_wide_radius_agree():
    n, nc = 24, 6
    ens = _fake(n, nc)
    st = _state(_linear_gaussian(n, nc, seed=1))
    out = []
    for method, rad in (('etkf', None), ('letkf', 1e7)):
        A = Assimilator(ens, layers=(0, 1, 2), method=method, radius_km=rad)
        obs = A.obs_soil_moisture(torch.arange(nc), torch.full((nc,), 55., dtype=F64),
                                  1.0, layers=[0])
        out.append(A.matrix(A.analysis(st, obs)[0]))
    # the ETKF assimilates all six observations in one global transform and
    # the LETKF once per cell with the same observations (all weights ~1),
    # so they agree to the weights of the Gaspari-Cohn taper at 1e7 km
    assert float((out[0] - out[1]).abs().max()) < 1e-3


def test_letkf_localisation_leaves_distant_cells_alone():
    n, nc = 20, 8
    lat = torch.tensor([30., 30.1, 30.2, 30.3, 40., 40.1, 40.2, 40.3], dtype=F64)
    ens = _fake(n, nc, lat=lat)
    st = _state(_linear_gaussian(n, nc, seed=2))
    A = Assimilator(ens, layers=(0, 1, 2), method='letkf', radius_km=200.0)
    obs = A.obs_soil_moisture(torch.tensor([0, 1]), torch.tensor([40., 40.], dtype=F64),
                              0.5, layers=[0])
    X0 = A.matrix(st)
    Xa = A.matrix(A.analysis(st, obs)[0])
    assert torch.equal(Xa[:, 4:], X0[:, 4:])            # 1100 km away
    assert float((Xa[:, :4] - X0[:, :4]).abs().max()) > 0.1


def test_gaspari_cohn_shape():
    d = torch.tensor([0., 50., 100., 150., 200., 300.], dtype=F64)
    g = gaspari_cohn(d, 200.0)
    assert float(g[0]) == 1.0 and float(g[4]) == 0.0 and float(g[5]) == 0.0
    assert bool((g[1:5] < g[:4]).all())
    assert abs(float(g[2]) - 5.0 / 24.0) < 1e-12        # d = radius / 2


def test_constraints_and_ledger_account_for_every_millimetre():
    n, nc = 30, 4
    ens = _fake(n, nc, maxm=80.0)
    g = torch.Generator().manual_seed(3)
    moist = 40.0 + 30.0 * torch.rand(n * nc, 3, generator=g, dtype=F64)
    swe = 0.05 * torch.rand(n * nc, generator=g, dtype=F64)       # m
    st = _state(moist, swe)
    A = Assimilator(ens, layers=(0, 1, 2), swe=True, method='etkf')
    obs = [A.obs_soil_moisture(torch.arange(nc), torch.full((nc,), 95., dtype=F64),
                               0.5, layers=[0]),
           A.obs_swe(torch.arange(nc), torch.zeros(nc, dtype=F64), 2.0)]
    new, info = A.analysis(st, obs, time='t0')
    v0, v1 = StateView(ens, st), StateView(ens, new)
    assert float(new['moist'].max()) <= 80.0 + 1e-12
    assert float(new['moist'].min()) >= 0.0
    e = A.ledger.entries[0]
    after = v1.moist_cell().sum(1) + v1.swe_cell()
    before = v0.moist_cell().sum(1) + v0.swe_cell()
    got = (after - before).reshape(n, nc)
    assert float((got - e['realised']).abs().max()) < 1e-10
    assert float(A.ledger.clipped_mm(0).abs().max()) > 0      # clipping bit
    assert float(new['snow']['ice'].min()) >= 0
    # density of the pack (depth/ice) is kept where there is a pack
    m = new['snow']['ice'] > 1e-9
    assert torch.allclose(new['snow']['depth'][m] / new['snow']['ice'][m],
                          torch.full_like(new['snow']['ice'][m], 2.0))
    row = A.ledger.rows()[0]
    assert row['term'] == 'da_increment' and row['clipped_mm_abs'] > 0
    # member volumes: sum over cells of realised mm * area
    assert abs(A.ledger.member_m3(0).mean().item() - row['external_m3']) < 1e-6
    # the input state is untouched
    assert torch.equal(st['moist'], moist)


def test_no_pack_is_created_where_a_member_has_none():
    n, nc = 10, 2
    ens = _fake(n, nc)
    st = _state(_linear_gaussian(n, nc, seed=5), torch.zeros(n * nc, dtype=F64))
    A = Assimilator(ens, layers=(0,), swe=True, method='etkf')
    obs = A.obs_swe(torch.arange(nc), torch.full((nc,), 30., dtype=F64), 1.0)
    new, _ = A.analysis(st, obs)
    assert float(new['snow']['ice'].abs().max()) == 0.0


# ---- the model: a twin experiment ------------------------------------------
def _wg(device=CPU, steps=96, first=196 * 8):
    if not os.path.exists(WG):
        pytest.skip('the Walnut Gulch global file is not present')
    return case.load(WG, device=device, steps=steps, first_step=first)


def _mm(ens, st):
    return StateView(ens, st).moist_cell().reshape(ens.n, ens.nc, -1)


def _twin(device, steps, n=8, window=8, seed=0):
    c = _wg(device, steps)
    truth = Ensemble(c, 1, forcing=ForcingPerturbation(
        prcp_sigma=0.4, seed=999, phi=0.9))
    fp = ForcingPerturbation(prcp_sigma=0.4, seed=1, phi=0.9)
    ol, da = Ensemble(c, n, forcing=fp), Ensemble(c, n, forcing=fp)
    A = Assimilator(da, layers=(0, 1, 2), method='enkf', inflation=1.1,
                    seed=seed)
    g = torch.Generator().manual_seed(5)
    st_t = st_o = st_d = None
    err_o, err_d, ext = [], [], []
    for w in range(0, c.n_steps, window):
        st_t = truth.run(w, w + window, st_t, True)['state']
        st_o = ol.run(w, w + window, st_o, True)['state']
        st_d = da.run(w, w + window, st_d, True)['state']
        yt = _mm(truth, st_t)[0]
        obs = []
        for ly in (0, 1):
            noise = 0.5 * torch.randn(c.n_cells, generator=g, dtype=F64)
            obs.append(A.obs_soil_moisture(
                torch.arange(c.n_cells), yt[:, ly] + noise.to(yt.device), 0.5,
                layers=[ly]))
        before = _mm(da, st_d).sum(-1)
        st_d, info = A.analysis(st_d, obs, time=w)
        after = _mm(da, st_d).sum(-1)
        e = A.ledger.entries[-1]
        assert float((after - before - e['d_moist'].sum(-1)).abs().max()) < 1e-9
        err_o.append(float((_mm(ol, st_o).mean(0) - yt).pow(2).mean().sqrt()))
        err_d.append(float((_mm(da, st_d).mean(0) - yt).pow(2).mean().sqrt()))
        ext.append(info['external_m3'])
    return err_o, err_d, A


def test_twin_experiment_analysis_beats_open_loop_cpu():
    # days 196-208 of 2008, the Walnut Gulch monsoon onset
    eo, ed, A = _twin(CPU, steps=96)
    assert sum(ed) < 0.5 * sum(eo), (sum(eo), sum(ed))
    assert sum(1 for a, b in zip(eo, ed) if b <= a) >= 0.75 * len(eo)
    assert len(A.ledger.entries) == len(eo)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason='needs CUDA')
def test_twin_experiment_on_gpu_lanes_state():
    eo, ed, A = _twin(torch.device('cuda'), steps=96)
    assert sum(ed) < 0.5 * sum(eo), (sum(eo), sum(ed))
