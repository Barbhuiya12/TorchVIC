"""snow_intercept.c assembled: canopy_step on random canopies -- finite, no
fallbacks on physical inputs, the canopy's mass closed with the vapour
counted, and every branch reached (snow melting at 0 C, snow frozen, no
snow)."""

import torch

from torchvic.core import canopy as C  # noqa: E402

D = torch.float64
CPU = torch.device('cpu')


def test_canopy_step_is_finite_closed_and_reaches_every_branch():
    g = torch.Generator(device=CPU).manual_seed(3)
    n = 500
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g, dtype=D, device=CPU)
    coin = lambda p: torch.rand(n, generator=g, dtype=D, device=CPU) < p
    z = torch.zeros(n, dtype=D, device=CPU)
    state = dict(int_snow=torch.where(coin(0.35), z, r(0.0, 0.015)),
                 int_rain=torch.where(coin(0.4), z, r(0.0, 0.0006)),
                 t_foliage=r(-15.0, 3.0),
                 temp_int_storage=torch.where(coin(0.6), z, r(0.0, 0.004)))
    lai = r(1.0, 6.0)
    f = dict(rainfall=torch.where(coin(0.5), z, r(0.0, 0.003)),
             snowfall=torch.where(coin(0.5), z, r(0.0, 0.004)),
             lai=lai, max_int=0.0001 * lai, wind1=r(0.2, 5.0),
             ra0=r(30.0, 120.0), ra1=r(10.0, 40.0),
             air_dens=r(1.0, 1.3), eact_air=r(100.0, 900.0),
             press=r(70000.0, 101300.0), lv=torch.full((n,), 2.5e6, dtype=D, device=CPU),
             t_canopy=r(-12.0, 6.0), vpd=r(0.0, 400.0),
             long_over_in=r(180.0, 340.0), long_under_out=r(250.0, 360.0),
             short_over_in=r(0.0, 500.0), bare_albedo=r(0.1, 0.2),
             dt=torch.full((n,), 3600.0, dtype=D, device=CPU),
             aero_406=torch.ones(n, dtype=torch.bool, device=CPU),
             aero_406_full=torch.zeros(n, dtype=torch.bool, device=CPU))
    evap_of = lambda t, nr, ns: torch.clamp(nr, min=0.0) / (2.5e6 * 1000.0) * 0.2
    ns, out = C.canopy_step(state, f, evap_of)
    for k, v in list(ns.items()) + list(out.items()):
        assert torch.isfinite(v.to(D)).all(), k
    assert int(out['fallback'].sum()) == 0, '%d canopies fell back' % int(out['fallback'].sum())
    before = state['int_snow'] + state['int_rain'] + f['rainfall'] + f['snowfall']
    after = ns['int_snow'] + ns['int_rain'] + out['rain_through'] + out['snow_through'] + out['vapor_flux']
    # the frozen branch's refreeze converts with the C's own unit quirk
    # (snow_intercept.c:505-508) and does not conserve; every other branch must
    has_snow = (ns['int_snow'] > 0) | (f['snowfall'] > 0) | (state['int_snow'] > 0)
    frozen = has_snow & (ns['t_foliage'] < 0.0)
    err = (after - before).abs()
    assert float(err[~frozen].max()) < 1e-12, float(err[~frozen].max())
    melting = has_snow & (ns['t_foliage'] == 0.0)
    assert int(melting.sum()) > 20 and int(frozen.sum()) > 20 and int((~has_snow).sum()) > 20, \
        (int(melting.sum()), int(frozen.sum()), int((~has_snow).sum()))
    assert float(ns['int_snow'].min()) >= 0.0 and float(ns['int_rain'].min()) >= 0.0
