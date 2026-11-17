"""The energy path's per-tile evaporation (VegMosaic.evap_per_tile) is the
water-balance path's (VegMosaic.step, per tile) at the same inputs, under
both cores, with a fractional canopy, a bare-soil tile and a layer below
its wilting point: func_surf_energy_bal.c 5.0.1:756-805 / 5.1.0:722-761
and canopy_evap.c 5.1.0:478-508 are one rule on both paths."""
import pytest
import torch

from torchvic.core import vegetation as V
from torchvic.core import version as core_version

D = torch.float64


@pytest.fixture(params=['5.0.1', '5.1.0'])
def core(request):
    old = core_version.CORE
    core_version.set_core(request.param)
    yield request.param
    core_version.set_core(old)


def _mosaic():
    n, nt = 2, 3
    T = lambda v, *s: torch.full(s, float(v), dtype=D)          # noqa: E731
    fc = torch.tensor([[0.4, 1.0, 1e-4], [0.75, 0.2, 1e-4]],
                      dtype=D).unsqueeze(-1).expand(n, nt, 12).clone()
    bare = torch.tensor([[False, False, True], [False, False, True]])
    return V.VegMosaic(
        cv=torch.tensor([[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]], dtype=D),
        lai=T(2.5, n, nt, 12), rmin=T(100.0, n, nt), rgl=T(100.0, n, nt),
        rarc=T(60.0, n, nt),
        root_frac=torch.tensor([0.5, 0.3, 0.2], dtype=D).expand(n, nt, 3),
        fcanopy=fc, rad_atten=T(0.5, n, nt), bare_tile=bare,
        dtype=D, device=torch.device('cpu'))


def test_energy_and_water_paths_agree_per_tile(core):
    mos = _mosaic()
    n, nt = 2, 3
    T = lambda v, *s: torch.full(s, float(v), dtype=D)          # noqa: E731
    # layer 0 of cell 1 sits just above Wpwp, so the clamp binds there
    moist = torch.tensor([[[30.0, 120.0, 200.0]] * nt,
                          [[15.02, 120.0, 200.0]] * nt], dtype=D)
    wcr = torch.tensor([[40.0, 90.0, 120.0]] * n, dtype=D)
    wpwp = torch.tensor([[15.0, 30.0, 40.0]] * n, dtype=D)
    wdew = torch.tensor([[0.1, 0.2, 0.0], [0.05, 0.0, 0.0]], dtype=D)
    rain = T(0.3, n)
    ns, tair, vpd = T(450.0, n, nt), T(28.0, n), T(1500.0, n)
    ra, rad, elev = T(40.0, n, nt), T(380.0, n, nt), T(100.0, n)
    bs = dict(moist=moist[:, :, 0], max_moist=T(45.0, n, nt),
              b_infilt=T(0.2, n, nt), depth=T(0.1, n, nt),
              resid_frac=T(0.02, n, nt),
              resid_mm=(T(2.0, n, nt) if core == '5.1.0' else None),
              ra=T(55.0, n, nt))
    dt = 10800.0
    e = mos.evap_per_tile(0, wdew, rain, moist, wcr, wpwp, ns, tair, vpd,
                          ra, rad, elev, dt=dt, bare_soil=bs,
                          wilting_clamp=True)
    w = mos.step(0, wdew, rain, moist, wcr, wpwp, ns, tair, vpd, ra, rad,
                 elev, dt=dt, per_tile=True, bare_soil=bs,
                 wilting_clamp=True, exact_wetness=True)
    for k, (a, b) in enumerate(zip(e, w)):
        assert torch.allclose(a, b, rtol=1e-13, atol=1e-15), (k, a, b)
    # the bare soil between the plants and on the bare tile is formed
    assert float(e[4][:, 0].min()) > 0.0 and float(e[4][:, 2].min()) > 0.0
    # nothing transpires on the bare-soil tile
    assert float(e[0][:, 2].abs().max()) == 0.0
    # cell 1's dry layer gives up no more than it holds above Wpwp
    assert float(e[0][1, :, 0].max()) <= 0.02 + 1e-15


def test_a_daily_energy_step_stores_rain_on_an_empty_canopy(core):
    """canopy_evap.c:78-111 on the energy path (func_surf_energy_bal.c
    :757 via evap_per_tile). An empty canopy has wetness 0, so a daily
    step evaporates none of the rain that lands and keeps it as Wdew.
    The floored wetness evaporated the rain instead: igp_EB_510_D,
    2004-01-01, cell 0, VIC Wdew 0.000794 mm and EVAP_CANOP 0."""
    mos = _mosaic()
    n, nt = 2, 3
    T = lambda v, *s: torch.full(s, float(v), dtype=D)          # noqa: E731
    moist = T(80.0, n, nt, 3)
    wcr = torch.tensor([[40.0, 90.0, 120.0]] * n, dtype=D)
    wpwp = torch.tensor([[15.0, 30.0, 40.0]] * n, dtype=D)
    wdew = torch.zeros(n, nt, dtype=D)
    rain = torch.tensor([0.0007942765951156616, 0.2], dtype=D)
    ns, tair, vpd = T(200.0, n, nt), T(15.0, n), T(800.0, n)
    ra, rad, elev = T(80.0, n, nt), T(40.0, n, nt), T(100.0, n)
    e = mos.evap_per_tile(
        0, wdew, rain, moist, wcr, wpwp, ns, tair, vpd, ra, rad, elev,
        dt=V.CONST_CDAY, wilting_clamp=True)
    # canopy_evap.c:78-86: wetness of an empty store is 0, so none of the
    # rain evaporates. It is held (func_surf_energy_bal.c scales Wdew by
    # fcanopy on the way out).
    assert torch.equal(e[1][:, :2], torch.zeros_like(e[1][:, :2]))
    assert float(e[3][:, :2].min()) > 0.0
    w = mos.step(
        0, wdew, rain, moist, wcr, wpwp, ns, tair, vpd, ra, rad, elev,
        dt=V.CONST_CDAY, per_tile=True, wilting_clamp=True,
        exact_wetness=True)
    # the vegetated tiles follow the same interception as the water path
    assert torch.equal(e[1][:, :2], w[1][:, :2])
    assert torch.equal(e[3][:, :2], w[3][:, :2])
