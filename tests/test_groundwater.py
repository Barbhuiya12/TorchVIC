"""Groundwater engine: analytic solutions, conservation, dry cells, restart,
device and CUDA-graph agreement (no MODFLOW 6 needed)."""
import numpy as np
import pytest
import torch

from torchvic.groundwater import Grid, GroundwaterModel, \
    GroundwaterNotConverged

TIGHT = dict(outer_dvclose=1e-10, inner_dvclose=1e-11, inner_rclose=1e-8)


def _line(ncol=21, dx=100.0, **kw):
    g = Grid.dis(1, 1, ncol, dx, 50.0, 10.0, [-90.0])
    return g, GroundwaterModel(g, k11=5.0, strt=0.0, **TIGHT, **kw)


def test_confined_linear_and_parabolic_steady():
    # CHD at both ends; without recharge the head is linear, with uniform
    # recharge it is the parabola h = hl + (hr-hl) x/L + R x (L-x) / (2 K b),
    # which the central difference reproduces exactly at cell centres.
    for rch in (0.0, 1e-3):
        g, m = _line()
        m.set_chd(([0, 0], [0, 0], [0, 20]), [10.0, 4.0])
        m.set_recharge(rch)
        m.step(1.0, steady=True)
        x = np.arange(21) * 100.0
        L = x[-1]
        T = 5.0 * 100.0
        h = 10.0 + (4.0 - 10.0) * x / L + rch * x * (L - x) / (2 * T)
        assert np.max(np.abs(m.h[0, 0].numpy() - h)) < 1e-8


def test_unconfined_dupuit_steady():
    # Dupuit: h^2 = hl^2 + (hr^2-hl^2) x/L + R x (L-x)/K, discretely only
    # approximate (harmonic transmissivity); check to 1 cm on a fine line.
    g = Grid.dis(1, 1, 201, 10.0, 1.0, 100.0, [0.0])
    m = GroundwaterModel(g, k11=10.0, icelltype=1, strt=20.0, **TIGHT)
    m.set_chd(([0, 0], [0, 0], [0, 200]), [20.0, 15.0])
    m.set_recharge(1e-3)
    m.step(1.0, steady=True)
    x = np.arange(201) * 10.0
    L = x[-1]
    h = np.sqrt(20.0 ** 2 + (15.0 ** 2 - 20.0 ** 2) * x / L
                + 1e-3 * x * (L - x) / 10.0)
    assert np.max(np.abs(m.h[0, 0].numpy() - h)) < 1e-2


def _closed(**kw):
    g = Grid.dis(2, 12, 15, 250.0, 200.0, 40.0, [10.0, -30.0])
    m = GroundwaterModel(g, k11=[8.0, 3.0], k33=[0.8, 0.3], icelltype=[1, 0],
                         ss=1e-5, sy=0.15, strt=25.0, variablecv=True,
                         **TIGHT, **kw)
    m.set_recharge(2e-3)
    m.set_wells(([1, 0], [5, 8], [6, 3]), [-4e3, -1e3], auto_flow_reduce=0.1)
    m.set_rivers(([0] * 3, [2, 3, 4], [10, 10, 10]), 24.0, 2e3, 20.0)
    return g, m


def test_conservation_closed_domain():
    g, m = _closed()
    area = torch.as_tensor(g.area)
    for _ in range(5):
        h0 = m.h.clone()
        r = m.step(1.0)
        rates = r.rates()
        net_bnd = sum(a - b for k, (a, b) in rates.items()
                      if not k.startswith('STO'))
        sto = sum(a - b for k, (a, b) in rates.items() if k.startswith('STO'))
        assert abs(net_bnd + sto) < 1e-9 * rates['RCH'][0]
        # storage rates agree with the head change: Sy in layer 1 (water
        # table inside the cell), Ss*b in confined layer 2
        dsy = -(m.h[0] - h0[0]) * 0.15 * area
        dss = -(m.h[1] - h0[1]) * 1e-5 * m.thk[1] * area
        assert torch.allclose(r.flows['STO-SY'][0], dsy, rtol=1e-12,
                              atol=1e-9)
        assert torch.allclose(r.flows['STO-SS'][1], dss, rtol=1e-12,
                              atol=1e-9)
        assert abs(r.discrepancy()[1]) < 1e-8


def test_dry_cell_and_auto_flow_reduce():
    g = Grid.dis(1, 3, 3, 100.0, 100.0, 10.0, [0.0])
    m = GroundwaterModel(g, k11=1e-3, icelltype=1, sy=0.1, strt=1.0, **TIGHT)
    m.set_wells(([0], [1], [1]), [-500.0], auto_flow_reduce=0.1)
    for _ in range(10):
        r = m.step(1.0)
    q = r.rates()['WEL'][1]
    assert 0.0 < q < 500.0                   # reduced, not cut off
    m2 = GroundwaterModel(g, k11=1e-3, icelltype=1, sy=0.1, strt=1.0, **TIGHT)
    m2.set_wells(([0], [1], [1]), [-2000.0])
    m2.step(1.0)
    assert int(m2.ibound[0, 1, 1]) == 0      # went dry, stays dry
    r = m2.step(1.0)
    assert r.rates()['WEL'] == (0.0, 0.0)
    assert float(m2.h[0, 1, 1]) == -1e30


def test_restart_bit_identical(tmp_path):
    g, a = _closed()
    for _ in range(6):
        a.step(1.0)
    g, b = _closed()
    for _ in range(3):
        b.step(1.0)
    b.save(tmp_path / 'gw.pt')
    g, c = _closed()
    c.load(tmp_path / 'gw.pt')
    for _ in range(3):
        c.step(1.0)
    assert torch.equal(a.h, c.h) and torch.equal(a.ibound, c.ibound)


def test_nonconvergence_is_transactional():
    g, m = _closed(outer_maximum=1)
    h0 = m.h.clone()
    with pytest.raises(GroundwaterNotConverged):
        m.step(1.0)
    assert torch.equal(m.h, h0) and m.kstp == 0


def test_preconditioners_agree():
    g, a = _closed(precond='jacobi')
    g, b = _closed(precond='rbgs')
    for _ in range(3):
        a.step(1.0)
        b.step(1.0)
    assert float((a.h - b.h).abs().max()) < 1e-8


def test_latlon_geometry_and_disu():
    lat = 20.0 + 0.0625 * np.arange(4)
    lon = 75.0 + 0.0625 * np.arange(5)
    g = Grid.latlon(lat, lon, 100.0, np.full((4, 5), 0.0))
    # spherical area equals the sum of a fine-latitude integral
    R = 6.37122e6
    d = np.deg2rad(0.0625)
    assert np.allclose(g.area[:, 0], R * R * d * (
        np.sin(np.deg2rad(lat) + d / 2) - np.sin(np.deg2rad(lat) - d / 2)))
    u = g.disu_arrays()
    assert u['iac'].sum() == u['ja'].size
    # every connection appears in both directions with the same width
    start = np.r_[0, np.cumsum(u['iac'])[:-1]]
    pairs = {}
    for n in range(len(u['iac'])):
        for p in range(start[n] + 1, start[n] + u['iac'][n]):
            pairs[(n, u['ja'][p])] = u['hwva'][p]
    for (n, m), w in pairs.items():
        assert pairs[(m, n)] == w


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_gpu_matches_cpu_and_graph_matches_eager():
    out = {}
    for dev, graph in (('cpu', False), ('cuda', False), ('cuda', True)):
        g, m = _closed(device=dev, cuda_graph=graph)
        for _ in range(4):
            m.step(1.0)
        out[(dev, graph)] = m.h.cpu()
    assert float((out[('cpu', False)] - out[('cuda', False)]).abs().max()) \
        < 1e-9
    assert torch.equal(out[('cuda', False)], out[('cuda', True)])
