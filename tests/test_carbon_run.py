"""CARBON wired into the step loop, on Walnut Gulch.

`tests/test_carbon.py` audits the carbon routines one at a time
against VIC's own compiled code. This checks the thing that audit cannot:
that the step loop calls them at the right point, with the right inputs,
and -- the claim that makes the wiring safe -- that turning CARBON on does
not move a single drop of water or joule of energy.

That claim is not a hope. VIC 5's default RC_MODE is RC_JARVIS
(initialize_options.c:66), so with CARBON on the canopy resistance is still
Jarvis's empirical function of light, temperature and vapour pressure
deficit; the carbon step *reads* it (canopy_evap.c:296-312) and writes back
only into carbon variables. A hydrology row that moved would mean the
wiring had reached somewhere it should not.

The sample data has no VEGLIB_PHOTO columns -- almost no public dataset
does, which is most of why CARBON is hard to run at all -- so the library
values here are Knorr's own for a C3 canopy and the four forcing columns
come from `torchvic.forcing.daily`, exactly as VIC 4.2.d derived them.
"""
import datetime
import os

import pytest
import torch

from torchvic.model import basin
from torchvic.io import build as image_run
from torchvic.forcing import daily as daily_forcing
from torchvic.io.global_param import GlobalParam
from torchvic.io.domain import ImageDomain
from torchvic.io.params import ImageParams

import _paths                                           # noqa: E402
HERE = _paths.REPO
SD = _paths.SAMPLE_DATA
WG = os.path.join(SD, 'image', 'WalnutGulch', 'parameters')
GLOBAL = os.path.join(WG, 'global_param.WalnutGulch.L2015.txt')

pytestmark = pytest.mark.skipif(
    not os.path.exists(GLOBAL),
    reason='VIC sample data not present (docs/downloads/data.md)')

#: The live default, for the reason test_image_physics.py gives: ten modules
#: in this directory set it at import time and pytest imports them all.
DEVICE = torch.empty(0).device
DT = torch.float64

#: read_veglib.c:162-192. A C3 canopy with Knorr's own capacities; Ctype 0
#: is C3 (vic_def.h PHOTO_C3).
CARBON_LIB = dict(Ctype=0.0, MaxCarboxRate=3.0e-5, MaxETransport=6.0e-5,
                  CO2Specificity=0.0, LightUseEff=0.5, NscaleFlag=1.0,
                  Wnpp_inhib=0.5, NPPfactor_sat=0.5)

#: Rows that must not move when CARBON is switched on.
HYDROLOGY = ('runoff', 'baseflow', 'discharge_mm', 'et', 'transp',
             'evap_canop', 'swe', 'sub_snow', 'soil_moisture',
             'canopy_storage')


@pytest.fixture(scope='module')
def wg():
    """Three days of Walnut Gulch with the carbon library and forcings
    bolted on."""
    import pandas as pd
    from torchvic.io.forcing import ImageForcing

    gp = GlobalParam(GLOBAL, sample_data=SD)
    dom = ImageDomain(gp.domain_file, names=gp.domain_vars)
    par = ImageParams(
        os.path.join(WG, 'params.WalnutGulch.L2015.0.0625_deg.nc'), dom)
    steps = gp.get_int('MODEL_STEPS_PER_DAY')
    fc = ImageForcing(gp.forcing_prefix, dom, force_type=gp.force_type)
    fields, times = fc.read(dtype=DT, device=DEVICE)
    n = 3 * steps
    fields = {k: v[:n] for k, v in fields.items()}
    dates = [pd.Timestamp(x).to_pydatetime() for x in times[:n]]
    batch = image_run.ImageBatch(dom, dates, dtype=DT, device=DEVICE,
                                 elevation=image_run.band_mean_elevation(par))
    atm = image_run.build_atmosphere(fields, batch)
    atm.pop('_sources')

    # The four columns VIC 5 demands and no sample dataset carries, derived
    # as initialize_atmos.c:1588-1700 derives them. `daily_to_subdaily` wants
    # a daily record; the forcing here is sub-daily, so its carbon block is
    # used directly on the days this run covers.
    lat = torch.as_tensor(dom.lat, dtype=DT, device=DEVICE).reshape(-1)
    lon = torch.as_tensor(dom.lon, dtype=DT, device=DEVICE).reshape(-1)
    doy, sec = [], []
    for d in dates:
        doy.append(d.timetuple().tm_yday)
        sec.append(d.hour * 3600 + d.minute * 60)
    from torchvic.core import atmos as _atmos
    doy_t = torch.tensor(doy, dtype=DT, device=DEVICE).reshape(1, -1)
    sec_t = torch.tensor(sec, dtype=DT, device=DEVICE).reshape(1, -1)
    cz = _atmos.coszen(lat.reshape(-1, 1), lon.reshape(-1, 1),
                       lon.reshape(-1, 1), doy_t, sec_t).transpose(0, 1)
    atm['coszen'] = cz.contiguous()
    atm['par'] = daily_forcing.SW2PAR * atm['shortwave']
    atm['Catm'] = torch.full_like(atm['shortwave'],
                                  daily_forcing.CATM_CURRENT_PPM * 1e-6)
    # MTCLIM's direct fraction is a daily number; without one, VIC 5 would
    # have read it from the file. A constant stands in here -- what is under
    # test is the wiring, and test_carbon.py audits the value's own maths.
    atm['fdir'] = torch.full_like(atm['shortwave'], 0.6)

    sp = image_run.build_soil(
        par, DT, DEVICE,
        runoff_steps_per_day=gp.get_int('RUNOFF_STEPS_PER_DAY'),
        model_steps_per_day=steps)
    mosaic = image_run.build_mosaic(par, DT, DEVICE, log=lambda s: None)
    mosaic.carbon = {k: torch.full_like(mosaic.cv, v)
                     for k, v in CARBON_LIB.items()}

    def t(v):
        return torch.tensor(v, dtype=DT, device=DEVICE)
    p = dict(pcf=t(1.0), t_max_snow=t(0.5), t_min_rain=t(-0.5), rs=t(120.0),
             et_beta=t(1.0), new_snow_albedo=t(0.85), alb_accum_a=t(0.94),
             alb_accum_b=t(0.58), alb_thaw_a=t(0.82), alb_thaw_b=t(0.46),
             uh_shape=t(2.5), uh_scale=t(2.0), lai_scale=t(1.0),
             rmin_scale=t(1.0), w_crit=t(0.7), w_wilt=t(0.3))
    init_moist = torch.as_tensor(par.soil['init_moist'], dtype=DT,
                                 device=DEVICE).T.contiguous()

    def run(carbon, carbon_options=None, full_energy=False):
        return basin.simulate(batch, atm, p, sp, mosaic=mosaic,
                              wind_h=float(gp.get('WIND_H')),
                              init_moist=init_moist, steps_per_day=steps,
                              full_energy=full_energy, carbon=carbon,
                              carbon_options=carbon_options)

    return dict(run=run, atm=atm, mosaic=mosaic, n=n, steps=steps)


@pytest.fixture(scope='module')
def both(wg):
    return wg['run'](False), wg['run'](True), wg


# ---------------------------------------------------------------------------
# The claim that makes the wiring safe
# ---------------------------------------------------------------------------
def test_carbon_moves_no_water_and_no_energy(both):
    """Bit for bit, every hydrology row is the same with CARBON on.

    RC_MODE is RC_JARVIS, so the canopy resistance the carbon step reads is
    the one the transpiration already used; nothing flows the other way.
    """
    off, on, _wg = both
    checked = 0
    for k in sorted(off):
        if k.startswith('carbon_'):
            continue
        a, b = off[k], on[k]
        assert torch.equal(a, b), k
        checked += 1
    assert checked >= len(HYDROLOGY)
    for k in HYDROLOGY:
        assert k in off, k


def test_carbon_rows_appear_only_when_asked(both):
    off, on, _wg = both
    assert not [k for k in off if k.startswith('carbon_')]
    for k in ('GPP', 'NPP', 'Raut', 'Rdark', 'Rphoto', 'Rmaint', 'Rgrowth',
              'aPAR', 'Ci'):
        assert 'carbon_' + k in on, k


# ---------------------------------------------------------------------------
# The fluxes themselves
# ---------------------------------------------------------------------------
def test_net_primary_production_is_gross_minus_respiration(both):
    """canopy_assimilation.c:180-188 -- NPP = GPP - Raut, and every scaling
    the step loop applies afterwards multiplies both."""
    _off, on, _wg = both
    gpp, raut, npp = (on['carbon_GPP'], on['carbon_Raut'], on['carbon_NPP'])
    assert float((npp - (gpp - raut)).abs().max()) < 1e-12 * max(
        1.0, float(gpp.abs().max()))


def test_autotrophic_respiration_is_its_four_parts(both):
    """:176-179 -- Raut = Rdark + Rphoto + Rmaint + Rgrowth."""
    _off, on, _wg = both
    parts = sum(on['carbon_' + k] for k in ('Rmaint', 'Rgrowth'))
    assert float(on['carbon_Raut'].min()) >= -1e-30
    assert float((on['carbon_Raut'] - parts).abs().max()) >= 0.0
    # Rmaint is built from Rdark and Rphoto, so the identity that holds at
    # the top level is Raut = Rmaint + Rgrowth.
    assert float((on['carbon_Raut'] - parts).abs().max()) < 1e-12 * max(
        1.0, float(on['carbon_Raut'].abs().max()))


def test_nothing_is_assimilated_in_the_dark(both):
    """photosynth.c gates on PAR, and the step loop gates on snow, leaf area
    and the dry fraction besides. A night-time GPP would mean one of those
    gates is on the wrong side."""
    _off, on, wg = both
    night = wg['atm']['shortwave'] <= 0.0
    assert bool(night.any()), 'no night in this window to test'
    assert float(on['carbon_GPP'][night.any(-1)].abs().max()) == 0.0
    assert float(on['carbon_aPAR'][night.any(-1)].abs().max()) == 0.0


def test_something_is_assimilated_in_the_day(both):
    """The other half: a lit, leafy, snow-free tile must fix carbon."""
    _off, on, wg = both
    day = (wg['atm']['shortwave'] > 200.0).any(-1)
    assert bool(day.any())
    assert float(on['carbon_GPP'][day].max()) > 0.0
    assert float(on['carbon_aPAR'][day].max()) > 0.0
    # Order of magnitude: a semi-arid shrubland fixes well under
    # 1e-5 gC/m2/s (about 1 gC/m2/day) at midday.
    assert float(on['carbon_GPP'].max()) < 1e-4


def test_the_carbon_step_needs_its_forcings_and_says_so(wg):
    """VIC 5 reads CATM, FDIR and PAR from the file and stops without them.
    Failing here with the name of the function that builds them beats a
    KeyError four frames down."""
    atm = dict(wg['atm'])
    atm.pop('Catm')
    with pytest.raises(ValueError, match='Catm'):
        basin.simulate.__wrapped__ if False else wg['run']
        # Re-run with the field removed.
        import torchvic.model.basin as _b
        _b.simulate(None, atm, {}, None, mosaic=wg['mosaic'], carbon=True)


def test_the_carbon_step_needs_the_library(wg):
    mos = wg['mosaic']
    saved, mos.carbon = mos.carbon, None
    try:
        with pytest.raises(ValueError, match='VEGLIB_PHOTO'):
            import torchvic.model.basin as _b
            _b.simulate(None, wg['atm'], {}, None, mosaic=mos, carbon=True)
    finally:
        mos.carbon = saved


def test_unknown_carbon_options_are_refused(wg):
    import torchvic.model.basin as _b
    with pytest.raises(ValueError, match='unknown carbon_options'):
        _b.simulate(None, wg['atm'], {}, None, mosaic=wg['mosaic'],
                    carbon=True, carbon_options={'Ncanopy': 3})


# ---------------------------------------------------------------------------
# VIC 5's gas constant
# ---------------------------------------------------------------------------
def test_vic5_gas_constant_is_the_default_and_the_correction_is_offered(wg):
    """photosynth.c uses CONST_RGAS -- J/(kmol K) -- in nine expressions VIC
    4.2.d wrote with Rgas = 8.3143 J/(mol K), and the activation energies are
    stated per mole. This model reproduces VIC 5 by default, because that is
    its contract, and offers the per-mole arithmetic as an option.

    The correction must raise gross primary production: VIC 5's supply term
    r0 is a thousand times too large, so its stomata deliver a thousandth of
    the CO2 they should and assimilation is starved everywhere.
    """
    vic5 = wg['run'](True)
    per_mol = wg['run'](True, {'gas_constant': 'per_mol'})
    assert float(vic5['carbon_GPP'].max()) > 0.0
    assert float(per_mol['carbon_GPP'].max()) > \
        10.0 * float(vic5['carbon_GPP'].max())
    # And still no effect on the water: the resistance the carbon step reads
    # is the hydrology's, whichever constant the photosynthesis uses.
    off = wg['run'](False)
    for k in HYDROLOGY:
        assert torch.equal(off[k], per_mol[k]), k


def test_the_temperature_response_is_what_distinguishes_them():
    """The clearest single consequence: with the per-kilomole constant the
    exponent of every Arrhenius factor is a thousandth of what it should be,
    so Vcmax at 5 C is 0.998 of its 25 C value instead of 0.183 -- the
    temperature dependence of photosynthesis is switched off.
    """
    import math
    from torchvic.core import carbon
    t0, t1 = 5.0 + 273.15 - 298.16, 298.16
    t = 5.0 + 273.15
    vic5 = math.exp(carbon.PHOTO_EV * (t0 / t1) / (carbon.CONST_RGAS * t))
    vic4 = math.exp(carbon.PHOTO_EV * (t0 / t1)
                    / (carbon.PHOTO_RGAS_PER_MOL * t))
    assert abs(vic5 - 0.998303) < 1e-5
    assert abs(vic4 - 0.183007) < 1e-5


def test_an_unknown_gas_constant_is_refused(wg):
    import torchvic.model.basin as _b
    with pytest.raises(ValueError, match='gas_constant'):
        _b.simulate(None, wg['atm'], {}, None, mosaic=wg['mosaic'],
                    carbon=True, carbon_options={'gas_constant': 'per_kmol'})


# ---------------------------------------------------------------------------
# The soil pools
# ---------------------------------------------------------------------------
SOIL_POOLS = ('CLitter', 'CInter', 'CSlow')
SOIL_FLUXES = ('RhLitter', 'RhInter', 'RhSlow', 'RhLitter2Atm', 'RhTot',
               'Litterfall', 'NEE')


@pytest.fixture(scope='module')
def energy_run(wg):
    """The same three days through the surface energy balance, which is what
    gives the soil thermal node temperatures the pools are evaluated at."""
    return wg['run'](False, full_energy=True), wg['run'](True,
                                                         full_energy=True)


def test_soil_pools_need_the_energy_balance_and_say_so(wg):
    """soil_carbon_balance.c:84-89 reads energy->T. Without the energy
    balance there are no node temperatures, so the pools are absent -- and
    absent, not silently zero, because zero is a number someone would plot.
    """
    on = wg['run'](True)                        # water balance only
    for k in SOIL_POOLS + SOIL_FLUXES:
        assert 'carbon_' + k not in on, k
    assert 'carbon_GPP' in on                   # the vegetation side still runs


def test_soil_pools_appear_with_the_energy_balance(energy_run):
    _off, on = energy_run
    for k in SOIL_POOLS + SOIL_FLUXES:
        assert 'carbon_' + k in on, k
        assert torch.isfinite(on['carbon_' + k]).all(), k


def test_the_soil_pools_move_no_water_either(energy_run):
    off, on = energy_run
    for k in sorted(off):
        if k.startswith('carbon_'):
            continue
        assert torch.equal(off[k], on[k]), k


def test_the_pools_start_empty_and_respiration_draws_on_them(energy_run):
    """initialize_soil.c:46-48 -- all three start at zero, and with no
    previous year there is no litterfall, so the only thing that can happen
    to them is respiration. Every pool must therefore be zero or negative,
    and RhTot must be zero while they are empty: a pool that grew from
    nothing would mean litterfall arrived a year early.
    """
    _off, on = energy_run
    for k in SOIL_POOLS:
        assert float(on['carbon_' + k].max()) <= 1e-30, k
    assert float(on['carbon_Litterfall'].abs().max()) == 0.0
    assert float(on['carbon_RhTot'].abs().max()) == 0.0


def test_net_ecosystem_exchange_is_production_minus_respiration(energy_run):
    """put_data.c:525 -- OUT_NEE is OUT_NPP minus OUT_RHET."""
    _off, on = energy_run
    assert float((on['carbon_NEE']
                  - (on['carbon_NPP'] - on['carbon_RhTot'])).abs().max()) \
        < 1e-30


def test_respiration_splits_the_way_vic_splits_it(energy_run):
    """soil_carbon_balance.c:164-166 -- the litter's respiration goes to the
    air in the fraction SRESP_FAIR and to the intermediate and slow pools in
    the rest, and RhTot is the part that reaches the atmosphere."""
    from torchvic.core import carbon
    _off, on = energy_run
    assert float((on['carbon_RhLitter2Atm']
                  - carbon.SRESP_FAIR * on['carbon_RhLitter']).abs().max()) \
        < 1e-30
    assert float((on['carbon_RhTot']
                  - (on['carbon_RhLitter2Atm'] + on['carbon_RhInter']
                     + on['carbon_RhSlow'])).abs().max()) < 1e-30


# ---------------------------------------------------------------------------
# The pools, seeded as a state file seeds them
# ---------------------------------------------------------------------------
SEEDED = dict(initial_pools=(500.0, 2000.0, 8000.0), annual_npp_prev=120.0)


def test_seeded_pools_respire_and_receive_litterfall(wg):
    """VIC reads the pools and last year's production from its state file.
    Given both, the litter must gain litterfall and lose respiration, and the
    other two must gain what the litter passes on and lose their own.
    """
    on = wg['run'](True, SEEDED, full_energy=True)
    lit = on['carbon_CLitter']
    assert float(on['carbon_Litterfall'].min()) > 0.0
    assert float(on['carbon_RhLitter'].min()) > 0.0
    # soil_carbon_balance.c:173 -- litter in, respiration out.
    step = lit[1:] - lit[:-1]
    want = (on['carbon_Litterfall'] - on['carbon_RhLitter'])[1:]
    assert float((step - want).abs().max()) < 1e-9


def test_the_intermediate_pool_receives_what_the_litter_does_not_exhale(wg):
    """:174-176 -- the fraction of litter respiration that does not reach
    the air is split between the intermediate and slow pools."""
    from torchvic.core import carbon
    on = wg['run'](True, SEEDED, full_energy=True)
    passed = (1.0 - carbon.SRESP_FAIR) * on['carbon_RhLitter']
    inter = on['carbon_CInter']
    step = inter[1:] - inter[:-1]
    want = (passed * carbon.SRESP_FINTER - on['carbon_RhInter'])[1:]
    assert float((step - want).abs().max()) < 1e-9


def test_vic5_empties_the_slow_pools_3600_times_too_fast(wg):
    """compute_soil_resp.c:180-192 divides the litter pool's turnover time by
    SEC_PER_DAY/dt and the other two by HOURS_PER_DAY/dt, with dt in seconds.
    VIC 4.2.d had dt in hours and wrote 365.25*24/dt on all three lines.

    Reproduced by default, corrected on request. The two must differ by the
    ratio of the two constants, and the litter pool -- whose line is right --
    must not move at all.
    """
    from torchvic.core import carbon
    a = wg['run'](True, SEEDED, full_energy=True)
    b = wg['run'](True, dict(SEEDED, soil_respiration='per_second'),
                  full_energy=True)
    assert float((a['carbon_RhLitter'] - b['carbon_RhLitter']).abs().max()) \
        < 1e-30
    ratio = carbon.SEC_PER_DAY / carbon.HOURS_PER_DAY
    # First step only: afterwards the pools have diverged, so the ratio of
    # the fluxes is no longer the ratio of the denominators.
    for k in ('RhInter', 'RhSlow'):
        got = float(a['carbon_' + k][0].mean() / b['carbon_' + k][0].mean())
        assert abs(got - ratio) < 1e-6 * ratio, (k, got)
    # And the consequence, plainly: a third of the intermediate pool in three
    # days against a thirty-three year turnover time.
    assert float(a['carbon_CInter'][-1].mean()) < \
        0.75 * float(a['carbon_CInter'][0].mean())
    assert float(b['carbon_CInter'][-1].mean()) > \
        0.999 * float(b['carbon_CInter'][0].mean())


def test_an_unknown_soil_respiration_choice_is_refused(wg):
    import torchvic.model.basin as _b
    with pytest.raises(ValueError, match='soil_respiration'):
        _b.simulate(None, wg['atm'], {}, None, mosaic=wg['mosaic'],
                    carbon=True, carbon_options={'soil_respiration': 'hours'})


# ---------------------------------------------------------------------------
# RC_MODE = RC_PHOTO: the resistance the transpiration uses
# ---------------------------------------------------------------------------
def test_rc_photo_needs_carbon(wg):
    """get_global_param.c:849-853 puts RC_MODE back to RC_JARVIS without
    CARBON; a caller who names RC_PHOTO there is told, not overruled."""
    with pytest.raises(ValueError, match='RC_PHOTO needs CARBON'):
        wg['run'](False, carbon_options={'rc_mode': 'RC_PHOTO'})
    with pytest.raises(ValueError, match='rc_mode'):
        wg['run'](True, carbon_options={'rc_mode': 'RC_NONE'})


def test_rc_jarvis_by_name_is_the_default(both, wg):
    off, on, _ = both
    named = wg['run'](True, carbon_options={'rc_mode': 'RC_JARVIS'})
    for k in on:
        assert torch.equal(on[k], named[k]), k


@pytest.mark.parametrize('full_energy', [False, True])
def test_rc_photo_moves_the_transpiration(wg, full_energy):
    """Photosynthetic resistance is not Jarvis's, so the water leaves
    differently; the carbon rows stay finite and the water closes."""
    jar = wg['run'](True, full_energy=full_energy)
    pho = wg['run'](True, full_energy=full_energy,
                    carbon_options={'rc_mode': 'RC_PHOTO'})
    assert not torch.equal(jar['transp'], pho['transp'])
    for k in ('transp', 'et', 'carbon_GPP', 'carbon_NPP', 'carbon_Ci'):
        assert bool(torch.isfinite(pho[k]).all()), k
    assert float(pho['carbon_GPP'].max()) > 0.0
    # a day with light: the demand-driven resistance still transpires
    assert float(pho['transp'].sum()) > 0.0


def test_rc_photo_transpiration_hook_replaces_calc_rc():
    """vegetation.transpiration with a constant-resistance hook is Penman at
    that resistance, in both CASEs, and reports the rs it aggregated
    (canopy_evap.c:296-307, :419-460)."""
    from torchvic.core import vegetation as V
    shp = (2, 3)
    z = lambda x: torch.full(shp, x, dtype=DT)          # noqa: E731
    am = torch.tensor([[[60.0, 70.0, 80.0]] * 3] * 2, dtype=DT)[:, :1]
    am = am.expand(2, 3, 3).contiguous()
    wc = torch.tensor([50.0, 60.0, 70.0], dtype=DT).expand(2, 3, 3)
    wp = torch.tensor([20.0, 30.0, 40.0], dtype=DT).expand(2, 3, 3)
    root = torch.tensor([0.3, 0.4, 0.3], dtype=DT).expand(2, 3, 3)
    sink = {}

    def hook(g):
        return (torch.full_like(g, 250.0),
                torch.full((3,) + tuple(g.shape), 750.0, dtype=DT))
    for share in (False, True):
        out, rc = V.transpiration(am, wc, wp, root, z(2.0), z(0.), z(0.),
                                  z(100.0), z(25.0), z(1500.0), z(40.0),
                                  z(200.0), z(1000.0), z(300.0),
                                  share_layer_moist=share, return_rc=True,
                                  rc_ps=hook, rs_sink=sink)
        assert sink['rs'].shape == (3,) + shp
        # every layer is above its wilting point (CASE 2 conducts in all
        # three, 1/rc = 3/250 -> 83.3 s/m) or CASE 1 takes the hook at 1
        want_rc = 250.0 / 3.0 if not share else 250.0
        assert float((rc - want_rc).abs().max()) < 1e-9
        assert float((sink['rs'] - 750.0 / 3.0 if not share
                      else sink['rs'] - 750.0).abs().max()) < 1e-9
        assert torch.isfinite(out).all()
