"""What `basin.simulate` actually receives, not what the reader produced.

Every defect the ingest audit found lived in the gap between those two. The
existing image tests are all assertions about the reader -- step counts,
array shapes, Cv sums, unit strings -- and not one of them runs the model or
asks what reached it. These do.

Each test here corresponds to a defect that was live and silent: a canopy
where every class shared one resistance, root zones used as soil layers, an
initial soil moisture the parameter file never asked for, and a measured wind
field read, unit-corrected, aggregated and then dropped.
"""
import os

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('xarray')

from torchvic.model import basin  # noqa: E402
from torchvic.io import build as image_run
from torchvic.core import vegetation
from torchvic.core.vegetation import root_fractions  # noqa: E402
from torchvic.io.global_param import GlobalParam
from torchvic.io.domain import ImageDomain
from torchvic.io.params import ImageParams

import _paths                                           # noqa: E402
HERE = _paths.REPO
SD = _paths.SAMPLE_DATA
WG = os.path.join(SD, 'image', 'WalnutGulch', 'parameters')
GLOBAL = os.path.join(WG, 'global_param.WalnutGulch.L2015.txt')

pytestmark = [pytest.mark.vic_core('5.0.1'), pytest.mark.skipif(
    not os.path.exists(GLOBAL),
    reason='VIC sample data not present (docs/downloads/data.md)')]

# Ten of the standalone scripts in this directory call
# `torch.set_default_device` at module scope, and pytest imports every
# test_*.py when it collects, so by the time these run the process default may
# already be CUDA. Asking for 'cpu' here then mixes devices inside torch's own
# factory functions. Take the live default instead of fighting it: what these
# tests check is physics, and it is the same on either device.
DEVICE = torch.empty(0).device


@pytest.fixture(scope='module')
def parts():
    gp = GlobalParam(GLOBAL, sample_data=SD)
    dom = ImageDomain(gp.domain_file, names=gp.domain_vars)
    par = ImageParams(
        os.path.join(WG, 'params.WalnutGulch.L2015.0.0625_deg.nc'), dom)
    return gp, dom, par


# -- root zones are not soil layers ------------------------------------------

# calc_root_fraction.c, run by hand for Walnut Gulch's zones (0.1/1.0/0.5 m)
# against its soil layers (0.1/0.3/1.0 m). The zone fractions are NOT the
# layer fractions, and the difference is where transpiration is drawn from.
VIC_ROOTS = [
    ([0.1, 1.0, 0.5], [0.10, 0.70, 0.20], [0.10, 0.210, 0.690]),
    ([0.1, 1.0, 0.5], [0.10, 0.65, 0.25], [0.10, 0.195, 0.705]),
    ([0.1, 1.0, 0.5], [0.05, 0.70, 0.25], [0.05, 0.210, 0.740]),
]


@pytest.mark.parametrize('depth,frac,expect', VIC_ROOTS)
def test_root_fractions_match_vic(depth, frac, expect):
    got = root_fractions(torch.tensor(depth, dtype=torch.float64, device=DEVICE),
                         torch.tensor(frac, dtype=torch.float64, device=DEVICE),
                         torch.tensor([0.1, 0.3, 1.0], dtype=torch.float64, device=DEVICE))
    assert torch.allclose(got, torch.tensor(expect, dtype=torch.float64, device=DEVICE),
                          atol=1e-12)


def test_roots_below_the_column_go_to_the_bottom_layer():
    # Zones reach 1.6 m, the column stops at 1.4 m. VIC caps the layer index
    # so the remainder lands in the last layer; renormalising instead would
    # spread it over every layer and move uptake upward.
    got = root_fractions(torch.tensor([0.1, 1.0, 0.5], dtype=torch.float64, device=DEVICE),
                         torch.tensor([0.1, 0.7, 0.2], dtype=torch.float64, device=DEVICE),
                         torch.tensor([0.1, 0.3, 1.0], dtype=torch.float64, device=DEVICE))
    assert float(got[-1]) == pytest.approx(0.69, abs=1e-12)
    assert float(got.sum()) == pytest.approx(1.0, abs=1e-12)


def test_the_mosaic_does_not_use_zone_fractions_as_layer_fractions(parts):
    _, _, par = parts
    m = image_run.build_mosaic(par, torch.float64, DEVICE)
    import numpy as np
    zone = np.asarray(par.veg['root_fract'])          # [tile, zone, cell]
    grass = int(np.argmax(np.asarray(par.veg['Cv']).reshape(
        zone.shape[0], -1).mean(1)))
    used = m.root_frac[0, grass].cpu().numpy()
    given = zone[grass, :, 0, 0] if zone.ndim == 4 else zone[grass, :, 0]
    assert not np.allclose(used, given, atol=1e-6), (
        'root zone fractions are reaching the soil column unchanged')


# -- moisture stress is applied once, through the resistance -----------------

def test_transpiration_is_not_stressed_twice(parts):
    """canopy_evap.c CASE 2 has no gsm_inv on the penman result.

    The defect multiplied by gsm a second time, so the ratio against VIC was
    exactly gsm: right at saturation, and 30 per cent of VIC at 30 per cent
    wetness. Comparing a wet layer with a half-dry one catches it; comparing
    only saturated conditions never would.
    """
    from torchvic.core.vegetation import transpiration
    kw = dict(dtype=torch.float64, device=DEVICE)
    lai = torch.tensor([1.0], **kw)
    args = dict(lai=lai, rmin=torch.tensor([50.0], **kw),
                rgl=torch.tensor([100.0], **kw),
                net_short=torch.tensor([250.0], **kw),
                tair=torch.tensor([20.0], **kw),
                vpd=torch.tensor([1500.0], **kw),
                ra=torch.tensor([40.0], **kw),
                rarc=torch.tensor([25.0], **kw),
                rad=torch.tensor([250.0], **kw),
                elevation=torch.tensor([1300.0], **kw))
    w_crit = torch.tensor([50.0], **kw)
    w_wilt = torch.tensor([20.0], **kw)
    root = torch.tensor([1.0], **kw)
    for wet in (1.0, 0.6, 0.3):
        avail = w_wilt + wet * (w_crit - w_wilt)
        ref = transpiration(avail, w_crit, w_wilt, root, **args)
        # A single-tile, single-layer mosaic must reproduce the reference.
        m = image_run.VegMosaic(
            cv=torch.ones(1, 1, **kw), lai=lai.reshape(1, 1, 1).expand(1, 1, 12),
            rmin=args['rmin'].reshape(1, 1), rgl=args['rgl'].reshape(1, 1),
            rarc=args['rarc'].reshape(1, 1), root_frac=root.reshape(1, 1, 1),
            dtype=torch.float64, device=DEVICE)
        got = m._transpiration(
            0, avail, w_crit, w_wilt, args['net_short'], args['tair'],
            args['vpd'], args['ra'], args['rad'], args['elevation'])
        assert float(got.reshape(-1)[0]) == pytest.approx(float(ref.reshape(-1)[0]),
                                                          rel=1e-9)


def test_a_layer_at_wilting_point_transpires_nothing(parts):
    _, _, par = parts
    m = image_run.build_mosaic(par, torch.float64, DEVICE)
    n = m.cv.shape[0]
    w_wilt = torch.full((n, 3), 20.0, dtype=torch.float64, device=DEVICE)
    w_crit = torch.full((n, 3), 50.0, dtype=torch.float64, device=DEVICE)
    one = torch.ones(n, dtype=torch.float64, device=DEVICE)
    got = m._transpiration(0, w_wilt.clone(), w_crit, w_wilt, 250.0 * one,
                           20.0 * one, 1500.0 * one, 40.0 * one, 250.0 * one,
                           1300.0 * one)
    assert float(got.abs().max()) == 0.0


# -- the initial state is the file's ----------------------------------------

def test_initial_moisture_comes_from_the_parameter_file(parts):
    _, _, par = parts
    sp = image_run.build_soil(par, torch.float64, DEVICE)
    im = torch.as_tensor(par.soil['init_moist'],
                         dtype=torch.float64, device=DEVICE).T.contiguous()
    _, moist = basin.initial_states(sp.max_moist.shape[0], sp,
                                    torch.float64, DEVICE, init_moist=im)
    assert torch.allclose(moist, torch.minimum(im, sp.max_moist))
    # And it must not be the old constant, or the test proves nothing.
    assert not torch.allclose(moist, 0.7 * sp.max_moist, atol=1.0)


def test_initial_moisture_is_clipped_at_capacity(parts):
    _, _, par = parts
    sp = image_run.build_soil(par, torch.float64, DEVICE)
    huge = sp.max_moist * 10.0
    _, moist = basin.initial_states(sp.max_moist.shape[0], sp,
                                    torch.float64, DEVICE, init_moist=huge)
    assert torch.allclose(moist, sp.max_moist)


# -- the forcing's wind reaches the model ------------------------------------

def test_wind_reaches_the_atmosphere(parts):
    _, dom, par = parts
    fields = {k: torch.ones(3, dom.n_active, dtype=torch.float64, device=DEVICE)
              for k in ('tair', 'tmax', 'tmin', 'prcp', 'press', 'vp',
                        'shortwave', 'longwave')}
    fields['wind'] = torch.full((3, dom.n_active), 4.25, dtype=torch.float64, device=DEVICE)
    b = image_run.ImageBatch(dom, [None, None, None], dtype=torch.float64,
                             device='cpu', elevation=par.soil['elev'])
    atm = image_run.build_atmosphere(fields, b)
    assert 'wind' in atm, 'the forcing measures wind and the model never sees it'
    assert float(atm['wind'].mean()) == pytest.approx(4.25)
    assert atm['_sources']['wind'] == 'forcing file'


def test_the_wind_height_is_the_global_files(parts):
    gp, _, _ = parts
    assert float(gp.get('WIND_H')) == 10.0
    # A resistance built at 2 m is 1.91x too small at this site.
    import math
    z0 = basin.SNOW_Z0
    assert (math.log(10.0 / z0) ** 2) / (math.log(2.0 / z0) ** 2) == \
        pytest.approx(1.913, abs=5e-3)


def test_snow_forcing_wind_is_not_frozen_at_import():
    """The default was bound in the signature, so the constant was dead.

    `wind=DEFAULT_WIND` in a def statement captures 1.5 at import; changing
    the module constant afterwards changed nothing, and the atmosphere's own
    wind could not override it either.
    """
    assert basin.snow_forcing_at.__defaults__[0] is None


# -- what the loop computed and used to drop ---------------------------------
#
# basin.simulate now returns the quantities VIC writes to its history file
# that the model had all along -- the rain/snow partition, the pack's own
# diagnostics, the per-step stores and their changes, a per-cell water error,
# the forcing echoes. None of them is new physics, so the tests here are
# identities the bookkeeping must satisfy rather than agreement with VIC:
# if the partition does not sum to the precipitation, or the per-step deltas
# do not sum to the end-minus-start of the store, the *recording* is wrong,
# and a scoreboard row built from it would be wrong by the same amount.

def _phase0_params(dtype, device):
    """The runner's defaults, VIC's own values, with the neutral multipliers."""
    def t(v):
        return torch.tensor(v, dtype=dtype, device=device)
    return dict(pcf=t(1.0), t_max_snow=t(0.5), t_min_rain=t(-0.5),
                rs=t(120.0), et_beta=t(1.0),
                new_snow_albedo=t(0.85), alb_accum_a=t(0.94),
                alb_accum_b=t(0.58), alb_thaw_a=t(0.82), alb_thaw_b=t(0.46),
                uh_shape=t(2.5), uh_scale=t(2.0),
                lai_scale=t(1.0), rmin_scale=t(1.0),
                w_crit=t(0.7), w_wilt=t(0.3))


@pytest.fixture(scope='module')
def phase0_run(parts):
    """Three days of Walnut Gulch at the global file's own step, through the
    mosaic and the water-balance path. Short because every identity below
    holds per step, so a year would prove nothing three days do not."""
    import pandas as pd
    from torchvic.io.forcing import ImageForcing
    gp, dom, par = parts
    dtype = torch.float64
    steps_per_day = gp.get_int('MODEL_STEPS_PER_DAY')
    fc = ImageForcing(gp.forcing_prefix, dom, force_type=gp.force_type)
    fields, times = fc.read(dtype=dtype, device=DEVICE)
    n = 3 * steps_per_day
    fields = {k: v[:n] for k, v in fields.items()}
    dates = [pd.Timestamp(x).to_pydatetime() for x in times[:n]]
    batch = image_run.ImageBatch(dom, dates, dtype=dtype, device=DEVICE,
                                 elevation=image_run.band_mean_elevation(par))
    atm = image_run.build_atmosphere(fields, batch)
    atm.pop('_sources')
    sp = image_run.build_soil(
        par, dtype, DEVICE,
        runoff_steps_per_day=gp.get_int('RUNOFF_STEPS_PER_DAY'),
        model_steps_per_day=steps_per_day)
    mosaic = image_run.build_mosaic(par, dtype, DEVICE)
    p = _phase0_params(dtype, DEVICE)
    init_moist = torch.as_tensor(par.soil['init_moist'], dtype=dtype,
                                 device=DEVICE).T.contiguous()
    out = basin.simulate(batch, atm, p, sp, mosaic=mosaic,
                         wind_h=float(gp.get('WIND_H')),
                         init_moist=init_moist, steps_per_day=steps_per_day,
                         full_energy=False)
    moist0 = torch.minimum(init_moist.expand_as(sp.max_moist), sp.max_moist)
    return dict(out=out, atm=atm, p=p, sp=sp, moist0=moist0, n=n)


PHASE0_KEYS = ('rainf', 'snowf', 'snow_melt', 'inflow', 'wdew', 'snow_cover',
               'salbedo', 'snow_surf_temp', 'snow_pack_temp', 'asat',
               'soil_wet', 'rootmoist', 'zwt', 'soil_liq', 'soil_ice',
               'delsoilmoist', 'delswe', 'delintercept', 'water_error',
               'air_temp', 'swdown', 'lwdown', 'pressure', 'vp', 'vpd',
               'wind', 'rel_humid', 'density', 'lai', 'fcanopy', 'surf_cond')


def test_phase0_outputs_are_present_finite_and_per_step(phase0_run):
    out, n = phase0_run['out'], phase0_run['n']
    for k in PHASE0_KEYS:
        assert k in out, '%s is not returned' % k
        assert out[k].shape[0] == n, '%s is not a per-step series' % k
        assert torch.isfinite(out[k]).all(), '%s has a non-finite value' % k
    assert out['soil_liq'].dim() == 3 and out['soil_ice'].dim() == 3


def test_rain_plus_snow_is_the_precipitation(phase0_run):
    out, atm, p = phase0_run['out'], phase0_run['atm'], phase0_run['p']
    prcp = atm['prcp'] * p['pcf']
    assert torch.allclose(out['rainf'] + out['snowf'], prcp, atol=1e-12)
    assert (out['rainf'] >= 0).all() and (out['snowf'] >= 0).all()


def test_delsoilmoist_sums_to_end_minus_start(phase0_run):
    out, moist0 = phase0_run['out'], phase0_run['moist0']
    change = out['soil_moisture'].sum(-1) - moist0.sum(-1)
    assert torch.allclose(out['delsoilmoist'].sum(0), change, atol=1e-10)
    # And the per-step series ends where the end state says it does.
    assert torch.equal(out['soil_liq'][-1], out['soil_moisture'])


def test_delswe_and_delintercept_sum_to_the_end_stores(phase0_run):
    out = phase0_run['out']
    # Both stores open empty: no pack, a dry canopy.
    assert torch.allclose(out['delswe'].sum(0), out['swe'][-1], atol=1e-12)
    assert torch.allclose(out['delintercept'].sum(0), out['canopy_storage'],
                          atol=1e-12)
    assert torch.equal(out['wdew'][-1], out['canopy_storage'])


def test_water_error_is_zero_per_cell_per_step(phase0_run):
    out = phase0_run['out']
    # put_data.c:531-544 per cell. A residual here that the runner's basin
    # mean does not show would be a cell losing what another gains.
    assert float(out['water_error'].abs().max()) < 1e-9


def test_snow_cover_is_the_pack_indicator(phase0_run):
    out = phase0_run['out']
    pack = out['swe'] > 0.0
    assert torch.equal(out['snow_cover'] > 0.5, pack)
    assert ((out['snow_cover'] == 0.0) | (out['snow_cover'] == 1.0)).all()
    # put_data.c:727-731: albedo and pack temperatures only where a pack is.
    for k in ('salbedo', 'snow_surf_temp', 'snow_pack_temp'):
        assert float(out[k][~pack].abs().max()) == 0.0 if (~pack).any() \
            else True
    # Melt is pack outflow: it can only leave a step that began with a pack
    # or received snow during it (solve_snow.c:125 leaves it zero otherwise).
    had = torch.cat([torch.zeros_like(pack[:1]), pack[:-1]]) \
        | (out['snowf'] > 0.0)
    assert (out['snow_melt'] >= 0.0).all()
    if (~had).any():
        assert float(out['snow_melt'][~had].abs().max()) == 0.0


def test_the_zeros_and_ones_are_vics_own(phase0_run):
    out = phase0_run['out']
    # FROZEN_SOIL off: no ice. OUT_SURF_COND: never assigned in VIC 5.0.1.
    # FCAN_SRC FROM_DEFAULT: fcanopy is 1.0 (vic_init.c:267).
    assert float(out['soil_ice'].abs().max()) == 0.0
    assert float(out['surf_cond'].abs().max()) == 0.0
    assert torch.equal(out['fcanopy'], torch.ones_like(out['fcanopy']))


def test_forcing_echoes_are_in_vics_output_units(phase0_run):
    out, atm = phase0_run['out'], phase0_run['atm']
    # put_data.c:137,151-152 divide by PA_PER_KPA; :141 is a percentage.
    assert torch.allclose(out['pressure'] * 1000.0, atm['press'])
    assert torch.allclose(out['vp'] * 1000.0, atm['eact_air'])
    assert torch.allclose(out['vpd'] * 1000.0, atm['vpd'])
    assert (out['rel_humid'] >= 0.0).all() and (out['rel_humid'] <= 100.0).all()
    assert torch.equal(out['air_temp'], atm['tair'])
    assert torch.equal(out['wind'], atm['wind'])
    assert torch.equal(out['lwdown'], atm['longwave'])


def test_root_zone_moisture_is_bounded_by_the_column(phase0_run):
    out = phase0_run['out']
    total = out['soil_liq'].sum(-1)
    assert (out['rootmoist'] >= 0.0).all()
    assert (out['rootmoist'] <= total + 1e-9).all()
    # Every tile at Walnut Gulch roots into the top layer at least, so the
    # root-zone moisture is never less than layer one's.
    assert (out['rootmoist'] >= out['soil_liq'][..., 0] - 1e-9).all()
    assert ((out['asat'] >= 0.0) & (out['asat'] <= 1.0)).all()
    assert ((out['soil_wet'] >= -1.0) & (out['soil_wet'] <= 1.0)).all()


def test_full_energy_refuses_an_overstory_tile_by_name():
    """The energy balance is the no-overstory branch. A forest tile solved on
    it would come out as grass with every balance closing -- the exact
    silence this project exists to remove. It must refuse, and say which
    cells and which classes.
    """
    from torchvic.core.vegetation import VegMosaic
    kw = dict(dtype=torch.float64, device=DEVICE)
    m = VegMosaic(cv=torch.tensor([[0.2, 0.8]], **kw),
                  lai=torch.full((1, 2, 12), 1.0, **kw),
                  rmin=torch.full((1, 2), 100.0, **kw),
                  rgl=torch.full((1, 2), 100.0, **kw),
                  rarc=torch.full((1, 2), 50.0, **kw),
                  root_frac=torch.tensor([[[0.3, 0.3, 0.4]] * 2], **kw),
                  dtype=torch.float64, device=DEVICE,
                  overstory=torch.tensor([[0.0, 1.0]], **kw))
    n = 1
    nt = 2
    atm = {k: torch.full((nt, n), 1.0, **kw) for k in
           ('tair', 'tmax', 'tmin', 'prcp', 'press', 'vp', 'shortwave',
            'longwave', 'wind', 'air_dens', 'lv', 'vpd', 'eact_air')}
    class B:  # the little a simulate call needs from a batch
        dates = [__import__('datetime').datetime(2008, 1, 1, h) for h in (0, 3)]
        elevation = torch.tensor([1000.0], **kw)
        cell_area = torch.tensor([1.0e6], **kw)
        source2outlet = torch.zeros(1, dtype=torch.long, device=DEVICE)
        n_outlets = 1
    from torchvic.core import column
    one = torch.ones(n, **kw)
    sp = column.SoilParams(b_infilt=0.2 * one, Ds=0.02 * one, Dsmax=10 * one,
                           Ws=0.8 * one, c=2 * one,
                           expt=torch.stack([4 * one] * 3, -1),
                           Ksat=torch.stack([250 * one] * 3, -1),
                           max_moist=torch.stack([100 * one, 300 * one, 600 * one], -1),
                           resid_moist=torch.stack([10 * one] * 3, -1),
                           runoff_steps_per_day=24, model_steps_per_day=8)
    sp.thermal = dict(avg_temp=10 * one, dp=4 * one,
                      depth=torch.tensor([[0.1, 0.3, 1.0]], **kw),
                      bulk_density=torch.full((1, 3), 1500.0, **kw),
                      soil_density=torch.full((1, 3), 2650.0, **kw),
                      quartz=torch.full((1, 3), 0.4, **kw), rough=0.01 * one)
    p = {k: torch.tensor(v, **kw) for k, v in dict(
        pcf=1., t_max_snow=.5, t_min_rain=-.5, rs=120., et_beta=1.,
        new_snow_albedo=.85, alb_accum_a=.94, alb_accum_b=.58, alb_thaw_a=.82,
        alb_thaw_b=.46, uh_shape=2.5, uh_scale=2., lai_scale=1., rmin_scale=1.,
        w_crit=.7, w_wilt=.3).items()}
    with pytest.raises(NotImplementedError) as e:
        basin.simulate(B(), atm, p, sp, mosaic=m, steps_per_day=8,
                       full_energy=True, allow_overstory=False)
    msg = str(e.value)
    assert 'overstory' in msg and '[1]' in msg and '--no-energy' in msg
    # and the water-balance path still runs the same mosaic
    out = basin.simulate(B(), atm, p, sp, mosaic=m, steps_per_day=8,
                         full_energy=False)
    assert bool(torch.isfinite(out['et']).all())


def test_albedo_is_returned_and_is_the_cover_weighted_class_albedo(parts):
    _, dom, par = parts
    m = image_run.build_mosaic(par, torch.float64, DEVICE)
    expect = (m.albedo[:, :, 0] * m.cv).sum(1)
    assert bool(torch.isfinite(expect).all())
    assert float(expect.min()) > 0.1 and float(expect.max()) < 0.3   # a real land albedo, not 0.20 everywhere
    assert float((expect - 0.20).abs().max()) > 1e-3, 'looks like the old fixed LAND_ALBEDO'


def test_reported_albedo_is_the_snow_albedo_under_a_pack(parts):
    """VIC's OUT_ALBEDO is AlbedoUnder: snow albedo where there is a pack,
    the tile albedo where there is not. A snow-free class albedo reported
    under 500 mm of SWE read 0.12 against VIC's 0.42 on FindleyLake.
    """
    _, dom, par = parts
    kw = dict(dtype=torch.float64, device=DEVICE)
    n = dom.n_active
    nt = 16
    # cold and snowing every step: a pack forms on step 1 and stays
    atm = {k: torch.full((nt, n), v, **kw) for k, v in dict(
        tair=-10.0, tmax=-8.0, tmin=-12.0, prcp=5.0, press=85000.0, vp=200.0,
        shortwave=50.0, longwave=200.0, wind=2.0, air_dens=1.1, lv=2.5e6,
        vpd=50.0, eact_air=200.0).items()}
    b = image_run.ImageBatch(dom, [__import__('datetime').datetime(2008, 1, 1) + __import__('datetime').timedelta(hours=3 * i) for i in range(nt)],
                             dtype=torch.float64, device=DEVICE, elevation=par.soil['elev'])
    sp = image_run.build_soil(par, torch.float64, DEVICE, 24, 8)
    m = image_run.build_mosaic(par, torch.float64, DEVICE)
    p = {k: torch.tensor(v, **kw) for k, v in dict(
        pcf=1., t_max_snow=.5, t_min_rain=-.5, rs=120., et_beta=1.,
        new_snow_albedo=.85, alb_accum_a=.94, alb_accum_b=.58, alb_thaw_a=.82,
        alb_thaw_b=.46, uh_shape=2.5, uh_scale=2., lai_scale=1., rmin_scale=1.,
        w_crit=.7, w_wilt=.3).items()}
    out = basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=8)
    assert float(out['swe'][-1].min()) > 10.0, 'no pack formed; test is not exercising the snow branch'
    bare = (m.albedo[:, :, 0] * m.cv).sum(1)
    late = out['albedo'][-1]
    assert float(late.min()) > 0.6, 'under a pack the reported albedo must be the snow albedo, got %.3f' % float(late.min())
    assert float((late - bare).abs().min()) > 0.3


def test_pack_depth_is_carried_and_consistent_with_its_density(parts):
    """solve_snow.c:358-372: a pack that exists has a density and a depth
    of CONST_RHOFW * swq / density; :471-472: one that does not has neither.
    OUT_SNOW_DEPTH (put_data.c:724) is that depth in cm at end of step.
    """
    _, dom, par = parts
    kw = dict(dtype=torch.float64, device=DEVICE)
    n = dom.n_active
    nt = 16
    atm = {k: torch.full((nt, n), v, **kw) for k, v in dict(
        tair=-10.0, tmax=-8.0, tmin=-12.0, prcp=5.0, press=85000.0, vp=200.0,
        shortwave=50.0, longwave=200.0, wind=2.0, air_dens=1.1, lv=2.5e6,
        vpd=50.0, eact_air=200.0).items()}
    b = image_run.ImageBatch(dom, [__import__('datetime').datetime(2008, 1, 1) + __import__('datetime').timedelta(hours=3 * i) for i in range(nt)],
                             dtype=torch.float64, device=DEVICE, elevation=par.soil['elev'])
    sp = image_run.build_soil(par, torch.float64, DEVICE, 24, 8)
    m = image_run.build_mosaic(par, torch.float64, DEVICE)
    p = {k: torch.tensor(v, **kw) for k, v in dict(
        pcf=1., t_max_snow=.5, t_min_rain=-.5, rs=120., et_beta=1.,
        new_snow_albedo=.85, alb_accum_a=.94, alb_accum_b=.58, alb_thaw_a=.82,
        alb_thaw_b=.46, uh_shape=2.5, uh_scale=2., lai_scale=1., rmin_scale=1.,
        w_crit=.7, w_wilt=.3).items()}
    out = basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=8)
    assert 'snow_depth' in out, 'simulate returned no snow_depth'
    swe = out['swe']                     # mm
    depth_cm = out['snow_depth']         # cm
    assert float(swe[-1].min()) > 10.0, 'no pack formed; the test is inert'
    # a pack has a depth, and no pack has none
    assert float(depth_cm[-1].min()) > 0.0
    assert float(depth_cm[0].abs().max()) == 0.0 or float(swe[0].min()) > 0.0
    # the implied density is a snow density: new snow at -10 C is ~50-150
    # kg/m3 and compaction over two days cannot take it past ice
    rho = (swe[-1] / 1000.0) / (depth_cm[-1] / 100.0) * 1000.0   # kg/m3
    assert float(rho.min()) > 30.0 and float(rho.max()) < 917.0, \
        'implied density %.1f..%.1f kg/m3' % (float(rho.min()), float(rho.max()))
    # depth grows with the pack while it snows every step
    assert float((depth_cm[-1] - depth_cm[4]).min()) > 0.0


def test_the_pack_sees_the_top_soil_node_as_its_ground_temperature(parts, monkeypatch):
    """surface_fluxes.c:389: Tgrnd = energy->T[0]. With the energy block on,
    the snow forcing's t_grnd is the cover-weighted top soil node of the
    previous step, not the 0 C placeholder. With it off VIC still carries
    T[0]: water balance sets Tsurf = Tair (calc_surf_energy_bal.c:542-545)
    and stores it as T[0] (:593-602, frozen_soil.c:66-69), from nodes that
    start at 0 C in this mode (generate_default_state.c:110-114). So the
    pack sees 0 C on the first step and the previous step's air temperature
    after it. This used to assert 0 C throughout; with T[0] carried and the
    rest of VIC's water-balance mode in place, Walnut Gulch 2008 3-hourly
    water balance puts snow sublimation at 1.2494 mm against VIC's 1.2494
    (it was 1.2670) and every water flux within 0.0001% of VIC
    (tests/test_wb_vs_vic.py).
    """
    _, dom, par = parts
    kw = dict(dtype=torch.float64, device=DEVICE)
    n = dom.n_active
    nt = 8
    atm = {k: torch.full((nt, n), v, **kw) for k, v in dict(
        tair=-4.0, tmax=-2.0, tmin=-6.0, prcp=3.0, press=85000.0, vp=300.0,
        shortwave=80.0, longwave=230.0, wind=2.0, air_dens=1.1, lv=2.5e6,
        vpd=50.0, eact_air=300.0).items()}
    b = image_run.ImageBatch(dom, [__import__('datetime').datetime(2008, 1, 1) + __import__('datetime').timedelta(hours=3 * i) for i in range(nt)],
                             dtype=torch.float64, device=DEVICE, elevation=par.soil['elev'])
    sp = image_run.build_soil(par, torch.float64, DEVICE, 24, 8)
    m = image_run.build_mosaic(par, torch.float64, DEVICE)
    p = {k: torch.tensor(v, **kw) for k, v in dict(
        pcf=1., t_max_snow=.5, t_min_rain=-.5, rs=120., et_beta=1.,
        new_snow_albedo=.85, alb_accum_a=.94, alb_accum_b=.58, alb_thaw_a=.82,
        alb_thaw_b=.46, uh_shape=2.5, uh_scale=2., lai_scale=1., rmin_scale=1.,
        w_crit=.7, w_wilt=.3).items()}
    seen = []
    real = basin.snow_forcing_at

    def spy(*a, **k):
        f = real(*a, **k)
        seen.append(f['t_grnd'].detach().clone())
        return f
    monkeypatch.setattr(basin, 'snow_forcing_at', spy)
    basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=8, full_energy=True)
    with_eb = torch.stack(seen[:nt])
    seen.clear()
    basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=8, full_energy=False)
    without = torch.stack(seen[:nt])
    assert float(without[0].abs().max()) == 0.0, \
        'without the energy block the first step must see the 0 C start'
    assert bool((without[1:] == -4.0).all()), \
        'without the energy block t_grnd must be the previous step\'s Tair'
    # the first step sees the initial node (the soil's average temperature,
    # non-zero), later steps the solved node
    assert float(with_eb[0].abs().min()) > 0.0, 'with the energy block t_grnd must be the top soil node'
    assert float((with_eb[1:] - with_eb[0]).abs().max()) > 0.0, 'the node must evolve'


def test_each_tile_keeps_its_own_soil_column(parts):
    """vic_run.c keeps a soil column per vegetation tile. After a warm dry
    spell the tiles' columns differ -- a grass tile with rmin 50 draws its
    column down faster than a shrub tile with rmin 135 -- while the cell's
    reported moisture is their cover-weighted sum and the per-cell water
    balance still closes.
    """
    _, dom, par = parts
    kw = dict(dtype=torch.float64, device=DEVICE)
    n = dom.n_active
    nt = 64
    atm = {k: torch.full((nt, n), v, **kw) for k, v in dict(
        tair=28.0, tmax=34.0, tmin=20.0, prcp=0.0, press=85000.0, vp=900.0,
        shortwave=600.0, longwave=380.0, wind=2.5, air_dens=1.05, lv=2.43e6,
        vpd=2500.0, eact_air=900.0).items()}
    b = image_run.ImageBatch(dom, [__import__('datetime').datetime(2008, 6, 1) + __import__('datetime').timedelta(hours=3 * i) for i in range(nt)],
                             dtype=torch.float64, device=DEVICE, elevation=par.soil['elev'])
    sp = image_run.build_soil(par, torch.float64, DEVICE, 24, 8)
    m = image_run.build_mosaic(par, torch.float64, DEVICE)
    p = {k: torch.tensor(v, **kw) for k, v in dict(
        pcf=1., t_max_snow=.5, t_min_rain=-.5, rs=120., et_beta=1.,
        new_snow_albedo=.85, alb_accum_a=.94, alb_accum_b=.58, alb_thaw_a=.82,
        alb_thaw_b=.46, uh_shape=2.5, uh_scale=2., lai_scale=1., rmin_scale=1.,
        w_crit=.7, w_wilt=.3).items()}
    out = basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=8)
    tile = out['soil_moisture_tile']                       # [cell, tile, layer]
    assert tile.shape == (n, m.n_tiles, 3)
    # the covered tiles' columns differ from one another after eight days
    covered = m.cv > 0.0
    spread = torch.zeros(n, **kw)
    for c in range(n):
        cols = tile[c][covered[c]].sum(-1)                 # column totals
        spread[c] = cols.max() - cols.min()
    assert float(spread.max()) > 0.5, 'tiles share one column: spread %.3f mm' % float(spread.max())
    # the cell's reported moisture is the cover-weighted sum
    weighted = (tile * m.cv.unsqueeze(-1)).sum(1)
    assert float((weighted - out['soil_moisture']).abs().max()) < 1e-9
    # and the per-cell water error stays closed with tiles weighted
    assert float(out['water_error'].abs().max()) < 1e-9, float(out['water_error'].abs().max())


def test_mosaic_carries_the_overstory_wind_parameters(parts):
    """CalcAerodynamic.c's overstory branch needs wind_atten and trunk_ratio
    per tile; the parameter file carries both and the mosaic must hand
    them on, shaped like the other per-tile scalars.
    """
    _, dom, par = parts
    m = image_run.build_mosaic(par, torch.float64, DEVICE)
    assert m.wind_atten is not None and m.trunk_ratio is not None
    assert m.wind_atten.shape == m.cv.shape and m.trunk_ratio.shape == m.cv.shape
    covered = m.cv > 0
    assert float(m.wind_atten[covered].min()) > 0.0
    assert 0.0 < float(m.trunk_ratio[covered].min()) <= 1.0
    k = min(3, m.cv.shape[0])
    sub = m.select(torch.arange(k, device=DEVICE))
    assert sub.wind_atten.shape[0] == k and sub.trunk_ratio.shape[0] == k


def test_a_covered_overstory_tile_takes_the_overstory_resistances(parts, monkeypatch):
    """With allow_overstory, a covered overstory tile is solved with
    CalcAerodynamic.c's overstory Ra[0]/U[0] and the understory references,
    canopy evaporation takes Ra[1], and the run still closes. Walnut Gulch
    has no covered overstory class, so one tile is flipped.
    """
    _, dom, par = parts
    kw = dict(dtype=torch.float64, device=DEVICE)
    n = dom.n_active
    nt = 8
    atm = {k: torch.full((nt, n), v, **kw) for k, v in dict(
        tair=12.0, tmax=18.0, tmin=6.0, prcp=1.0, press=85000.0, vp=600.0,
        shortwave=300.0, longwave=300.0, wind=2.5, air_dens=1.1, lv=2.45e6,
        vpd=800.0, eact_air=600.0).items()}
    b = image_run.ImageBatch(dom, [__import__('datetime').datetime(2008, 4, 1) + __import__('datetime').timedelta(hours=3 * i) for i in range(nt)],
                             dtype=torch.float64, device=DEVICE, elevation=par.soil['elev'])
    sp = image_run.build_soil(par, torch.float64, DEVICE, 24, 8)
    m = image_run.build_mosaic(par, torch.float64, DEVICE)
    # the most covered tile of every cell becomes an overstory of 6 m
    k = int(m.cv.sum(0).argmax())
    m.overstory[:, k] = 1.0
    m.displacement[:, k, :] = 4.0
    m.veg_rough[:, k, :] = 0.5
    p = {kk: torch.tensor(v, **kw) for kk, v in dict(
        pcf=1., t_max_snow=.5, t_min_rain=-.5, rs=120., et_beta=1.,
        new_snow_albedo=.85, alb_accum_a=.94, alb_accum_b=.58, alb_thaw_a=.82,
        alb_thaw_b=.46, uh_shape=2.5, uh_scale=2., lai_scale=1., rmin_scale=1.,
        w_crit=.7, w_wilt=.3).items()}
    calls = []
    real = vegetation.aerodynamic_resistance_overstory

    def spy(*a, **kw_):
        r = real(*a, **kw_)
        calls.append(r)
        return r
    monkeypatch.setattr(vegetation, 'aerodynamic_resistance_overstory', spy)
    with pytest.raises(NotImplementedError):
        basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=8, full_energy=True,
                       allow_overstory=False)
    out = basin.simulate(b, atm, p, sp, mosaic=m, steps_per_day=8,
                         full_energy=True)
    assert len(calls) == nt, 'the overstory branch must run once per step'
    r = calls[-1]
    assert torch.isfinite(r['ra'][:, k]).all() and float(r['ra'][:, k, 1].min()) > 0.0
    assert float(out['water_error'].abs().max()) < 1e-9
    for key in ('eb_latent', 'eb_sensible', 'eb_t_surf', 'eb_vegt', 'et'):
        assert torch.isfinite(out[key]).all(), key
    # Tfoliage = Tair on the overstory tile pulls OUT_VEGT toward the air
    assert float((out['eb_vegt'][-1] - 12.0).abs().max()) < float((out['eb_t_surf'][-1] - 12.0).abs().max()) + 1e-9
