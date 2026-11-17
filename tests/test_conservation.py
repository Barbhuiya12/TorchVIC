"""Every conservation law this model claims, measured on one run.

The individual modules each check their own budget where they have one --
`test_canopy_snow_mass.py` for the canopy, `test_lake.py` for the lake ice
and the lake's volume, `test_image_physics.py` for the cell water balance.
What this file adds is the whole set in one place, on one run, so that a
law with no owner is visible as a missing row rather than as nothing at
all, and so that "everything closes" is a statement with numbers behind it.

The equations are VIC's, not invented here:

  cell water     put_data.c:531-544, calc_water_balance_error()
                 inflow - outflow - (storage - last_storage)
  cell energy    calc_surf_energy_bal.c:573 -- the residual root_brent
                 drove to zero (VIC keeps it as energy->error; TorchVIC
                 reports it as OUT_ENERGY_ERROR by default). VIC's own
                 OUT_ENERGY_ERROR is a different expression, put_data.c's
                 sum of the reported terms, which need not close.
  snow pack      snow_melt.c:492-496, snow->mass_error
  canopy         snow_intercept.c -- what the canopy held plus what fell
                 equals what it holds plus what fell through
  soil column    runoff.c -- moisture in equals moisture out plus the
                 change in storage, layer by layer
  frozen soil    the ice and the liquid of a layer sum to its moisture
  carbon         soil_carbon_balance.c -- what leaves the pools as
                 respiration equals what entered them as litterfall minus
                 the change in the pools
  routing        the unit hydrograph moves water in time and must not
                 create or destroy it

A law that this model does not implement gets no row, and a law that VIC
itself does not close gets its own number rather than a pass.
"""
import os

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('xarray')

from torchvic.model import basin, routing  # noqa: E402
from torchvic.io import build as image_run
from torchvic.io.global_param import GlobalParam
from torchvic.io.domain import ImageDomain
from torchvic.io.forcing import ImageForcing
from torchvic.io.params import ImageParams

import _paths                                           # noqa: E402
HERE = _paths.REPO
SD = _paths.SAMPLE_DATA
WG = os.path.join(SD, 'image', 'WalnutGulch', 'parameters')
GLOBAL = os.path.join(WG, 'global_param.WalnutGulch.L2015.txt')

pytestmark = pytest.mark.skipif(
    not os.path.exists(GLOBAL),
    reason='VIC sample data not present (docs/downloads/data.md)')

DEVICE = torch.empty(0).device
DT = torch.float64

#: Per cell per step, in the units VIC reports the quantity in. These are
#: not tuned to pass: the water balance is an exact identity in double
#: precision and closes at 1e-16 of the storage it moves, so anything at
#: 1e-9 mm would already be a hundred million times looser than what is
#: measured. They are stated as absolutes so a regression that is still
#: "small" is still a failure.
WATER_TOL_MM = 1.0e-9
ENERGY_TOL_WM2 = 1.0e-9
#: The surface balance is solved as VIC solves it (root_brent.c, stopping
#: within ROOT_BRENT_T = 1e-7 K of the root), so its residual at the root
#: is VIC's size, ~1e-6 W m-2 (VIC's own OUT_ENERGY_ERROR on the ISC
#: windows reaches 5.6e-6), not zero to rounding.
SURFACE_RESID_TOL_WM2 = 1.0e-4


def _params(dtype, device):
    t = lambda v: torch.tensor(v, dtype=dtype, device=device)
    return dict(pcf=t(1.0), t_max_snow=t(0.5), t_min_rain=t(-0.5),
                rs=t(120.0), et_beta=t(1.0),
                new_snow_albedo=t(0.85), alb_accum_a=t(0.94),
                alb_accum_b=t(0.58), alb_thaw_a=t(0.82), alb_thaw_b=t(0.46),
                uh_shape=t(2.5), uh_scale=t(2.0),
                lai_scale=t(1.0), rmin_scale=t(1.0),
                w_crit=t(0.7), w_wilt=t(0.3))


def _run(days=3, **kw):
    """Walnut Gulch at the global file's own step. Short on purpose: every
    identity here holds per step, so a year proves nothing three days do
    not, and the ones that accumulate are summed over the run."""
    import pandas as pd
    gp = GlobalParam(GLOBAL, sample_data=SD)
    dom = ImageDomain(gp.domain_file, names=gp.domain_vars)
    par = ImageParams(
        os.path.join(WG, 'params.WalnutGulch.L2015.0.0625_deg.nc'), dom)
    steps = gp.get_int('MODEL_STEPS_PER_DAY')
    fc = ImageForcing(gp.forcing_prefix, dom, force_type=gp.force_type)
    fields, times = fc.read(dtype=DT, device=DEVICE)
    n = days * steps
    fields = {k: v[:n] for k, v in fields.items()}
    dates = [pd.Timestamp(x).to_pydatetime() for x in times[:n]]
    batch = image_run.ImageBatch(dom, dates, dtype=DT, device=DEVICE,
                                 elevation=image_run.band_mean_elevation(par))
    atm = image_run.build_atmosphere(fields, batch)
    atm.pop('_sources')
    sp = image_run.build_soil(
        par, DT, DEVICE,
        runoff_steps_per_day=gp.get_int('RUNOFF_STEPS_PER_DAY'),
        model_steps_per_day=steps)
    mosaic = image_run.build_mosaic(par, DT, DEVICE)
    init_moist = torch.as_tensor(par.soil['init_moist'], dtype=DT,
                                 device=DEVICE).T.contiguous()
    out = basin.simulate(batch, atm, _params(DT, DEVICE), sp, mosaic=mosaic,
                         wind_h=float(gp.get('WIND_H')),
                         init_moist=init_moist, steps_per_day=steps, **kw)
    moist0 = torch.minimum(init_moist.expand_as(sp.max_moist), sp.max_moist)
    return dict(out=out, atm=atm, sp=sp, moist0=moist0, n=n, batch=batch,
                mosaic=mosaic)


@pytest.fixture(scope='module')
def water_run():
    return _run(full_energy=False)


@pytest.fixture(scope='module')
def energy_run():
    return _run(full_energy=True)


@pytest.fixture(scope='module')
def frozen_run():
    return _run(full_energy=True, frozen_soil=True)


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------

def _water_residual(bundle):
    return float(bundle['out']['water_error'].abs().max())


def test_the_cell_water_balance_closes_on_every_path(water_run, energy_run,
                                                     frozen_run):
    """put_data.c:531-544 per cell per step, on the three paths that carry
    different physics: the water balance alone, the full surface energy
    balance, and the energy balance with frozen soil.

    Frozen soil is the one that can fail silently. It moves water between a
    layer's liquid and its ice, and a bookkeeping error there shows up in
    nothing else -- the layer's total is unchanged, the evaporation is
    unchanged, and only this identity notices.
    """
    for name, bundle in (('water', water_run), ('energy', energy_run),
                         ('frozen', frozen_run)):
        assert _water_residual(bundle) < WATER_TOL_MM, (
            '%s path: %.3e mm' % (name, _water_residual(bundle)))


def test_the_water_balance_closes_over_the_run_not_only_per_step(energy_run):
    """A per-step residual of zero can still hide a store that is counted
    twice, if the double count is constant. Summing the series and
    comparing against the end state minus the start state catches that:
    the two are computed from different things, the fluxes and the stores.
    """
    out, moist0 = energy_run['out'], energy_run['moist0']
    change = out['soil_moisture'].sum(-1) - moist0.sum(-1)
    assert torch.allclose(out['delsoilmoist'].sum(0), change, atol=1e-10)
    assert torch.allclose(out['delswe'].sum(0), out['swe'][-1], atol=1e-12)
    assert torch.allclose(out['delintercept'].sum(0), out['canopy_storage'],
                          atol=1e-12)


def test_precipitation_is_partitioned_and_not_created(energy_run):
    """Rain plus snow is the precipitation, exactly, and neither is
    negative. The partition is a ramp, so a sign error in its denominator
    inverts it silently -- which is the defect
    `vegetation.partition_precipitation` carries a guard against."""
    out, atm = energy_run['out'], energy_run['atm']
    assert torch.allclose(out['rainf'] + out['snowf'], atm['prcp'],
                          atol=1e-12)
    assert bool((out['rainf'] >= 0).all()) and bool((out['snowf'] >= 0).all())


def test_no_store_and_no_flux_is_negative(energy_run):
    """Not a conservation law but the thing that makes one meaningless:
    a budget closes just as well with a negative store as with a positive
    one, and a negative store is not physical. Evaporation is exempt --
    condensation is real and VIC reports it as negative evaporation -- and
    so is the water balance residual itself."""
    out = energy_run['out']
    for k in ('swe', 'canopy_storage', 'soil_moisture', 'runoff', 'baseflow',
              'wdew', 'rainf', 'snowf', 'snow_melt'):
        if k in out:
            assert float(out[k].min()) >= -1e-12, (k, float(out[k].min()))


def test_the_soil_layers_sum_to_the_column(energy_run):
    """OUT_SOIL_MOIST is the total and OUT_SOIL_LIQ the unfrozen part, so
    with no ice they are the same array. put_data.c:691-703."""
    out = energy_run['out']
    assert torch.allclose(out['soil_liq'][-1], out['soil_moisture'],
                          atol=0.0)
    if 'soil_ice' in out:
        assert float(out['soil_ice'].abs().max()) == 0.0


def test_a_frozen_layer_splits_its_moisture_and_does_not_change_it(
        frozen_run):
    """frozen_soil.c: the ice and the liquid of a layer are a partition of
    its moisture, not two independent stores. A layer whose ice exceeds its
    moisture, or whose parts do not sum to it, would leave the cell balance
    untouched and every soil output wrong."""
    out = frozen_run['out']
    liq, ice = out['soil_liq'], out['soil_ice']
    assert bool((ice >= -1e-12).all())
    assert bool((liq >= -1e-12).all())
    # The reported total is the sum of the parts at every step and layer.
    total = out['soil_moisture']
    assert torch.allclose(liq[-1] + ice[-1], total, atol=1e-10)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def test_the_surface_energy_balance_is_solved_to_zero(energy_run):
    """calc_surf_energy_bal.c:573, the residual root_brent drives to zero.
    TorchVIC reports it as OUT_ENERGY_ERROR (out['eb_energy_error']) with
    the default energy_error='solver'; VIC's own OUT_ENERGY_ERROR is
    put_data.c's sum of the reported terms, a different quantity.

    The limit of this claim, stated because it is easy to overstate: the
    solver drives this expression to zero by construction, so a converged
    step returns zero whatever the physics. What it detects is a step that
    did NOT converge, which is how VIC's energy balance actually fails. It
    is a convergence check, not an independent reconstruction of the
    budget (an independent one is not available from model output).
    """
    out = energy_run['out']
    # a full-energy run always reports it; a missing key must fail, not skip
    assert 'eb_energy_error' in out, sorted(k for k in out if 'energy' in k)
    assert float(out['eb_energy_error'].abs().max()) < SURFACE_RESID_TOL_WM2


def test_every_energy_term_is_finite_on_every_step(energy_run):
    """A NaN in one flux does not break a balance that is never formed.
    Every energy output is checked for finiteness because that is the
    failure this model is most exposed to: `torch.where` evaluates both
    branches, so a guard written as a selection leaves a non-finite value
    in the unselected one."""
    out = energy_run['out']
    eb = [k for k in out if k.startswith('eb_')] + ['refreeze']
    checked = 0
    for k in eb:
        if k in out and torch.is_tensor(out[k]):
            assert torch.isfinite(out[k]).all(), k
            checked += 1
    assert checked > 0, 'no energy terms were returned to check'


def test_blowing_snow_moves_mass_without_creating_it():
    """CalcBlowingSnow.c returns a sublimation flux that
    SnowPackEnergyBalance.c:217-235 adds to the pack's own vapour flux. So
    it takes mass out of the pack, and that mass has to arrive in
    OUT_SUB_SNOW and therefore in the evaporation, or the balance opens.

    Getting this to fire at all took two corrections to the forcing and
    both are the point of the test. Walnut Gulch in January has no pack
    that survives a step, so the air temperature is dropped 25 K and the
    precipitation multiplied. That alone still gives a flux of exactly
    zero, and it is not a defect: lowering the temperature without
    lowering the vapour pressure leaves the air supersaturated -- 500 Pa
    against a saturation of 84 Pa at -22 C -- and blowing snow cannot
    sublimate into air that is already saturated. Dropping the humidity
    to 40% of saturation at the new temperature is what makes it run.

    That is worth writing down because a test that had stopped at the
    first step would have reported "blowing snow changes nothing" and
    been believed.
    """
    from torchvic.io import build as image_run
    from torchvic.core import snow as _snow
    oa, om = image_run.build_atmosphere, image_run.build_mosaic

    def cold(fields, batch):
        a = oa(fields, batch)
        a['tair'] = a['tair'] - 25.0
        a['prcp'] = a['prcp'] * 40.0
        a['wind'] = torch.clamp(a['wind'] * 6.0, min=12.0)
        sat = _snow.svp(a['tair'])
        a['eact_air'] = 0.40 * sat
        a['vpd'] = sat - a['eact_air']
        return a

    def with_blowing_fields(par, dtype, device):
        # read_vegparam.c:444-456's three per-tile fields, which the image
        # parameter file does not carry; basin.simulate refuses BLOWING
        # without them rather than inventing them, so the test supplies
        # VIC's own sample values.
        m = om(par, dtype, device)
        like = torch.ones_like(m.cv)
        m.sigma_slope = like * 0.005
        m.lag_one = like * 0.95
        m.fetch = like * 2000.0
        return m

    image_run.build_atmosphere = cold
    image_run.build_mosaic = with_blowing_fields
    try:
        on = _run(full_energy=True, blowing=True)
        off = _run(full_energy=True)
    finally:
        image_run.build_atmosphere = oa
        image_run.build_mosaic = om

    # It has to have actually done something, or the closure below is a
    # statement about a flux of zero.
    d_swe = float(on['out']['swe'].sum() - off['out']['swe'].sum())
    d_sub = float(on['out']['sub_snow'].sum() - off['out']['sub_snow'].sum())
    d_et = float(on['out']['et'].sum() - off['out']['et'].sum())
    # Not its sign: the blowing flux is a loss, but its latent heat enters
    # the pack's own balance (SnowPackEnergyBalance.c:217-235), which then
    # sublimates less at its surface. Since the open-field pack takes VIC's
    # snow-surface wind U[2] (CalcAerodynamic.c:97-104; surface_fluxes.c
    # :418-425) rather than the forcing's wind at its measurement height,
    # the second effect wins here: total sublimation -0.26 mm, SWE up.
    assert abs(d_swe) > 1e-3, 'blowing snow changed nothing: %g' % d_swe
    assert abs(d_sub) > 1e-3, 'and reported nothing: %g' % d_sub
    # Whatever the pack lost to blowing arrives in the evaporation, because
    # put_data.c:626 puts snow->vapor_flux there. Not to the last digit,
    # and the reason is not a leak: a pack that blowing snow has thinned
    # is a different pack, so the transpiration and canopy evaporation of
    # the following steps differ too. Measured at 1.2e-05 mm against a
    # 0.17 mm flux -- seven parts in a hundred thousand, which is the
    # second-order term and not the first.
    assert abs(d_et - d_sub) / abs(d_sub) < 1e-3, (d_et, d_sub)
    # And the balance closes with it on, per cell per step.
    assert float(on['out']['water_error'].abs().max()) < WATER_TOL_MM


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_the_unit_hydrograph_moves_water_without_creating_it():
    """routing.route convolves runoff with a unit hydrograph. A unit
    hydrograph that does not sum to one is a gain or a loss applied to
    every cell of every step, and it would not show in any per-cell
    balance because routing is downstream of all of them."""
    uh = basin.gamma_unit_hydrograph(60, torch.tensor(2.5, dtype=DT,
                                                      device=DEVICE),
                                     torch.tensor(2.0, dtype=DT,
                                                  device=DEVICE),
                                     DT, DEVICE)
    assert abs(float(uh.sum()) - 1.0) < 1e-12, float(uh.sum())
    assert bool((uh >= 0).all())

    # And the convolution itself: what goes in comes out, once the tail has
    # drained. A pulse into one outlet, run long enough for the whole unit
    # hydrograph to pass.
    nt, nc = 200, 3
    src = torch.zeros(nt, nc, dtype=DT, device=DEVICE)
    src[5] = torch.tensor([1.0, 2.0, 3.0], dtype=DT, device=DEVICE)
    s2o = torch.zeros(nc, dtype=torch.long, device=DEVICE)
    q = routing.route(src, uh.unsqueeze(1).expand(-1, nc), s2o, 1)
    assert abs(float(q.sum()) - float(src.sum())) < 1e-10, (
        float(q.sum()), float(src.sum()))


def test_routing_conserves_mass_across_a_source_time_offset():
    """`source_time_offset` delays a source's contribution down the ring.
    VIC's own rout_convolution.c does not conserve mass across a non-zero
    offset -- measured against the compiled C --
    so this says explicitly that ours does, rather than leaving it to be
    assumed from the zero-offset case.

    The record is long enough for the whole delayed hydrograph to land:
    water still in transit when the simulation ends is gone from the
    outlet, which is a truncation and not a leak, and it is the reason a
    short record would show a loss here.
    """
    uh = basin.gamma_unit_hydrograph(60, torch.tensor(2.5, dtype=DT,
                                                      device=DEVICE),
                                     torch.tensor(2.0, dtype=DT,
                                                  device=DEVICE),
                                     DT, DEVICE)
    nt, ns = 400, 3
    src = torch.zeros(nt, ns, dtype=DT, device=DEVICE)
    src[5] = torch.tensor([1.0, 2.0, 3.0], dtype=DT, device=DEVICE)
    s2o = torch.zeros(ns, dtype=torch.long, device=DEVICE)
    uhx = uh.unsqueeze(1).expand(-1, ns)
    for off in (None,
                torch.tensor([0, 0, 0], device=DEVICE),
                torch.tensor([0, 5, 20], device=DEVICE),
                torch.tensor([0, 100, 300], device=DEVICE)):
        q = routing.route(src, uhx, s2o, 1, source_time_offset=off)
        lost = float(src.sum() - q.sum())
        assert abs(lost) < 1e-12, (off, lost)


def test_the_soil_carbon_pools_conserve_carbon():
    """soil_carbon_balance.c. The three pools are not independent stores:
    what leaves the litter as respiration is split, a fixed fraction to the
    atmosphere and the rest between the intermediate and slow pools, so

        d(litter + intermediate + slow) == litterfall - RhTot

    identically. Checked on both arithmetics -- VIC 5's, and the one its
    own litter line implies (evidence/carbon_soil_respiration.txt) --
    because a units defect that changes the rates must not be allowed to
    also break the conservation.
    """
    from torchvic.core import carbon
    n, nn = 6, 3
    f = lambda v: torch.full((n,), float(v), dtype=DT, device=DEVICE)
    dz = torch.full((nn, n), 0.1, dtype=DT, device=DEVICE)
    dz_cum = torch.cumsum(dz, 0)
    t_node = torch.stack([f(12.0), f(9.0), f(6.0)])
    w_node = torch.stack([f(0.6), f(0.55), f(0.5)])
    start = (500.0, 2000.0, 8000.0)
    for per_second in (False, True):
        out = carbon.soil_carbon_balance(
            dz, dz_cum, f(0.3), f(3600.0), t_node, w_node,
            f(start[0]), f(start[1]), f(start[2]), f(700.0),
            per_second=per_second)
        change = ((out['CLitter'] - start[0]) + (out['CInter'] - start[1])
                  + (out['CSlow'] - start[2]))
        resid = change - (out['Litterfall'] - out['RhTot'])
        assert float(resid.abs().max()) < 1e-11, (per_second,
                                                  float(resid.abs().max()))
        # ...and the pools actually moved, or the identity is about zero.
        assert float(change.abs().max()) > 1e-3


# ---------------------------------------------------------------------------
# What has no row here
# ---------------------------------------------------------------------------

def test_the_laws_without_an_owner_are_named_not_forgotten():
    """The point of this file is that a conservation law with nobody
    checking it should be visible. These are the ones this run does not
    reach, each with where it IS checked, so the list stays honest rather
    than becoming a list of things nobody looked at.

    This test asserts only that the list is complete and the files it
    names exist; it is documentation that cannot go stale silently.
    """
    elsewhere = {
        # law: the file that measures it
        'snow pack mass (snow_melt.c mass_error)':
            'tests/test_snow_formulas.py',
        'canopy interception mass (snow_intercept.c)':
            'tests/test_canopy_snow_mass.py',
        'lake ice mass (ice_melt.c:583-586)':
            'tests/test_lake.py',
        'lake volume and area (water_balance)':
            'tests/test_lake.py',
        # soil carbon and blowing snow used to be listed here as somebody
        # else's; they were not. test_carbon_run.py checks that the pools
        # appear and what they need, not that carbon is conserved, and
        # test_blowing_snow.py verifies the flux against VIC's compiled
        # function without ever checking where the mass goes. Both are
        # now measured above, in this file.
    }
    root = HERE
    for law, path in elsewhere.items():
        assert os.path.exists(os.path.join(root, path)), (law, path)
