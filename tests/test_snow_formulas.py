#!/usr/bin/env python3
"""torchvic.core.snow against *pristine* VIC 5.0.1, formula by formula.

tests/test_vic_snow.py already checks this module against
`ref_soil`, which links the `vic/` directory in this repository. That
directory is not upstream: it carries our own modifications, and at least one
of them -- `param.SNOW_NEW_SNOW_DENS_MAX`, a 400 kg/m3 ceiling on new snow
density that VIC 5.0.1 does not have anywhere -- had been mirrored into
torchvic/core/snow.py and was passing that check. Verified against the wrong
authority, a divergence looks exactly like agreement.

So the oracle here is `ref_snow`, built by tests/oracle/build_ref_snow.sh
against reference/VIC-VIC.5.0.1/. Where that binary is present the reference
numbers are taken from it directly; where it is not -- no compiler, no
upstream tree -- the same numbers are used from the table below, which was
recorded from it. `test_the_recorded_reference_still_matches_vic` is the test
that keeps the table honest, and it skips rather than passes when the binary
is missing, because a skipped check must never read as a green one.

Every case here is a case torchvic.core.snow got wrong before the audit of
2026-09; the comment on each says what it used to return.

Run: pytest tests/test_snow_formulas.py
"""
import os
import subprocess
import sys

import pytest
import torch


from torchvic.core import snow  # noqa: E402

import _paths                                           # noqa: E402

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')
REF = _paths.oracle('ref_snow')

# Double precision throughout: the comparison is against C doubles and the
# tolerances below are tighter than float32 can represent.
DTYPE = torch.float64

# The surface temperature comes from a root find, so it -- and everything
# downstream of it, which by the end of the step is every store -- agrees only
# to the solver's tolerance. VIC's own is param.ROOT_BRENT_T = 1e-7 K and
# `torchvic.core.implicit` uses the same; measured, that lands the temperatures
# within 1e-10 K and the stores within 1e-13 m. 1e-9 relative is therefore
# loose enough for the solver and still a thousand times tighter than the
# smallest defect this file was written for -- the 0.0155 K pack temperature
# of case 5, which is 1.1e-3 relative.
RTOL = 1e-9
T_ATOL = 1e-6
# An absolute floor as well, because a relative tolerance says nothing about a
# quantity whose true value is near zero -- case 6 sublimates 0.028 mm and the
# solver moves the last 1e-10 mm of it. 1e-11 m is 1e-8 mm; the smallest
# defect this file exists for is 0.028 mm, six orders of magnitude above it.
ATOL = 1e-11


# ---------------------------------------------------------------------------
# The seven whole-step cases of ref_snow.c's `melt` mode, and what VIC 5.0.1
# makes of them. Regenerate with:
#
#     tests/oracle/build_ref_snow.sh && ref_snow melt
#
# Inputs are (swq, pack_water, surf_water, surf_temp, pack_temp, depth,
# density, snowfall_mm, rainfall_mm, t_air, t_grnd, wind, net_short, long_in,
# air_dens, eact, lv, press, vpd, ra, z, z0, dt); outputs are (melt_mm, swq,
# pack_water, surf_water, surf_temp, pack_temp, sub_snow, depth, density).
MELT_CASES = [
    # 0: deep cold pack, dry windy air. Sublimates 0.219 mm in three hours
    #    and melts nothing. We used to return swq unchanged at 0.300 and no
    #    sublimation at all, and a surface temperature of -7.9166.
    (dict(swq=0.30, pack_water=0.0, surf_water=0.0, surf_temp=-5.0,
          pack_temp=-6.0, depth=1.0, density=300.0, snowfall=0.0, rain=0.0,
          t_air=-8.0, t_grnd=-2.0, wind=3.0, net_short=40.0, long_in=220.0,
          air_dens=1.30, eact=180.0, lv=2.5e6, press=88000.0, vpd=120.0,
          ra=60.0, z=2.0, z0=0.010, dt=10800.0),
     dict(melt=0.0, swq=0.29978129638575202, pack_water=0.0, surf_water=0.0,
          surf_temp=-7.9132459411491594, pack_temp=-6.0000000000000009,
          sublimation=0.00021870361424795981, depth=0.99497161413094326,
          density=301.29633059692424)),
    # 1: shallow pack under warm air. Deposition adds 0.263 mm, which melts
    #    out in the same step; we used to report 11.863 mm of melt.
    (dict(swq=0.05, pack_water=0.0, surf_water=0.0, surf_temp=0.0,
          pack_temp=0.0, depth=0.2, density=250.0, snowfall=0.0, rain=0.0,
          t_air=6.0, t_grnd=1.0, wind=4.0, net_short=200.0, long_in=330.0,
          air_dens=1.20, eact=800.0, lv=2.5e6, press=97000.0, vpd=150.0,
          ra=45.0, z=2.0, z0=0.010, dt=10800.0),
     dict(melt=12.125433101625417, swq=0.038137092735116312, pack_water=0.0,
          surf_water=0.0012896601408010349, surf_temp=0.0, pack_temp=0.0,
          sublimation=-0.00026252583674172175, depth=0.15195963332758089,
          density=250.96857566709051)),
    # 2: 12 mm of snow onto an existing cold pack.
    (dict(swq=0.20, pack_water=0.0, surf_water=0.0, surf_temp=-3.0,
          pack_temp=-4.0, depth=0.7, density=285.0, snowfall=12.0, rain=0.0,
          t_air=-4.0, t_grnd=-2.0, wind=2.0, net_short=30.0, long_in=240.0,
          air_dens=1.29, eact=300.0, lv=2.5e6, press=90000.0, vpd=60.0,
          ra=70.0, z=2.0, z0=0.010, dt=10800.0),
     dict(melt=0.0, swq=0.2118587822265853, pack_water=0.0, surf_water=0.0,
          surf_temp=-4.6034343401127247, pack_temp=-4.0823865617730215,
          sublimation=0.00014121777341471214, depth=0.74304855816501292,
          density=285.12104612621647)),
    # 3: rain on a ripe pack. The pack density update must use the water
    #    equivalent from before the rain arrived, not after.
    (dict(swq=0.10, pack_water=0.005, surf_water=0.003, surf_temp=0.0,
          pack_temp=0.0, depth=0.35, density=300.0, snowfall=0.0, rain=5.0,
          t_air=3.0, t_grnd=0.5, wind=2.5, net_short=100.0, long_in=310.0,
          air_dens=1.24, eact=700.0, lv=2.5e6, press=95000.0, vpd=80.0,
          ra=55.0, z=2.0, z0=0.010, dt=10800.0),
     dict(melt=15.883516177837352, swq=0.089214126992291171, pack_water=0.0,
          surf_water=0.0030169028451499436, surf_temp=0.0, pack_temp=0.0,
          sublimation=-9.7643170128528087e-05, depth=0.31111096549167117,
          density=286.75982812531095)),
    # 4: a thin pack that melts out completely. 4.222 mm leaves, not 4.000:
    #    the difference is the 0.222 mm the air deposited on it.
    (dict(swq=0.004, pack_water=0.0, surf_water=0.0, surf_temp=0.0,
          pack_temp=0.0, depth=0.02, density=200.0, snowfall=0.0, rain=0.0,
          t_air=10.0, t_grnd=4.0, wind=3.0, net_short=250.0, long_in=340.0,
          air_dens=1.20, eact=900.0, lv=2.5e6, press=97000.0, vpd=200.0,
          ra=40.0, z=2.0, z0=0.010, dt=10800.0),
     dict(melt=4.2222882045586534, swq=0.0, pack_water=0.0, surf_water=0.0,
          surf_temp=0.0, pack_temp=0.0,
          sublimation=-0.00022228820455865355, depth=0.0, density=0.0)),
    # 5: very cold and calm, with 2 mm of liquid in the surface layer. The
    #    surface solves below freezing, so that liquid freezes into the ice.
    #    We used to leave it liquid and report a pack temperature of -14.000.
    (dict(swq=0.50, pack_water=0.0, surf_water=0.002, surf_temp=-12.0,
          pack_temp=-14.0, depth=1.6, density=310.0, snowfall=0.0, rain=0.0,
          t_air=-20.0, t_grnd=-6.0, wind=1.0, net_short=10.0, long_in=150.0,
          air_dens=1.40, eact=60.0, lv=2.5e6, press=80000.0, vpd=40.0,
          ra=120.0, z=2.0, z0=0.010, dt=10800.0),
     dict(melt=0.0, swq=0.49984525773239025, pack_water=0.0, surf_water=0.0,
          surf_temp=-17.157260412194908, pack_temp=-14.015542304640631,
          sublimation=0.0001547422676097515, depth=1.595293837036289,
          density=313.32488481306655)),
    # 6: 8 mm of snow at -6 C onto bare ground. The change in cold content is
    #    measured from the temperature the surface had *before* the new snow
    #    was mixed in; measuring it after gave -12.279 C here.
    (dict(swq=0.0, pack_water=0.0, surf_water=0.0, surf_temp=0.0,
          pack_temp=0.0, depth=0.0, density=0.0, snowfall=8.0, rain=0.0,
          t_air=-6.0, t_grnd=-3.0, wind=2.0, net_short=20.0, long_in=200.0,
          air_dens=1.31, eact=150.0, lv=2.5e6, press=88000.0, vpd=90.0,
          ra=70.0, z=2.0, z0=0.010, dt=10800.0),
     dict(melt=0.0, swq=0.0079717103350624118, pack_water=0.0,
          surf_water=0.0, surf_temp=-11.510751877509858, pack_temp=0.0,
          sublimation=2.8289664937588687e-05, depth=0.083426519937183655,
          density=95.553672154426962)),
]

# new_snow_density() at two air temperatures warm enough to pass the ceiling
# that VIC 5.0.1 does not impose. From `ref_snow dens`.
NEW_SNOW_DENSITY = {20.0: (512.40000000000009, 115764.15134728422),
                    30.0: (789.60000000000002, 5497145.7261182452)}


def _t(v):
    return torch.tensor([v], dtype=DTYPE, device='cpu')


def _step(case):
    """Run one MELT_CASES input through snow_step, as VIC's caller would.

    Stores are metres and snowfall and rainfall arrive in millimetres, which
    is the convention snow_melt.c:103-104 converts at its own front door. The
    rain goes into the surface liquid store before the call because
    snow_melt.c:168 puts it there before the balance is formed, and
    `snow_step` documents that it expects it already done.
    """
    ice = case['swq'] - case['pack_water'] - case['surf_water']
    state = dict(ice=_t(ice), depth=_t(case['depth']),
                 surf_water=_t(case['surf_water'] + case['rain'] / 1000.0),
                 pack_water=_t(case['pack_water']),
                 surf_temp=_t(case['surf_temp']),
                 pack_temp=_t(case['pack_temp']))
    forcing = dict(snowfall=_t(case['snowfall'] / 1000.0),
                   rain=_t(case['rain'] / 1000.0),
                   air_temp=_t(case['t_air']), t_grnd=_t(case['t_grnd']),
                   wind=_t(case['wind']), net_short=_t(case['net_short']),
                   long_snow_in=_t(case['long_in']),
                   air_dens=_t(case['air_dens']), eact_air=_t(case['eact']),
                   lv=_t(case['lv']), press=_t(case['press']),
                   vpd=_t(case['vpd']), dt=_t(case['dt']), z=_t(case['z']),
                   z0=_t(case['z0']), ra=_t(case['ra']),
                   snow_depth=_t(case['depth']),
                   snow_density=_t(case['density']))
    out, melt = snow.snow_step(state, forcing)
    got = {k: float(out[k][0]) for k in
           ('swq', 'pack_water', 'surf_water', 'surf_temp', 'pack_temp',
            'sublimation', 'depth', 'density')}
    got['melt'] = float(melt[0]) * 1000.0
    return got


def _close(got, want, name):
    # Temperatures are compared with an absolute floor as well, because a
    # pack at exactly 0 C has no scale for a relative one.
    tol = T_ATOL if 'temp' in name else ATOL
    assert abs(got - want) <= tol + RTOL * abs(want), (
        '%s: got %.17g, VIC 5.0.1 gives %.17g' % (name, got, want))


# ---------------------------------------------------------------------------
# The defects, one test each.

def test_new_snow_density_has_no_upper_limit():
    """snow_utility.c:207-239 caps nothing; our 400 kg/m3 ceiling was 5.1's.

    The ceiling is `param.SNOW_NEW_SNOW_DENS_MAX`, which exists in the `vic/`
    tree in this repository and not in VIC 5.0.1. Under VIC's own
    precipitation partition it never binds -- snow stops falling at +0.5 C,
    where Bras gives 158 kg/m3 -- so this is a disagreement that costs
    nothing today and would cost 49% to anything that widened the partition.
    """
    for temp, (bras, snthrm) in NEW_SNOW_DENSITY.items():
        got = float(snow.new_snow_density(_t(temp), method='bras')[0])
        _close(got, bras, 'bras at %g C' % temp)
        got = float(snow.new_snow_density(_t(temp), method='snthrm')[0])
        _close(got, snthrm, 'snthrm at %g C' % temp)

    # The ceiling is still reachable, for anyone who wants VIC 5.1's answer,
    # but only by asking for it by name.
    assert float(snow.new_snow_density(_t(30.0), dens_max=400.0)[0]) == 400.0


def test_the_energy_balance_reports_the_vapour_flux():
    """SnowPackEnergyBalance.c:233 writes it; discarding it loses OUT_SUB_SNOW.

    The flux is in metres of water per step with the C's sign, negative out
    of the pack. This case is the cold dry windy one, where it is -0.4315 mm
    over three hours.
    """
    st = dict(t_air=_t(-8.0), t_grnd=_t(-2.0), wind=_t(3.0),
              net_short=_t(40.0), long_snow_in=_t(220.0), air_dens=_t(1.30),
              eact_air=_t(180.0), lv=_t(2.5e6), press=_t(88000.0),
              vpd=_t(120.0), rain=_t(0.0), dt=_t(10800.0), z=_t(2.0),
              z0=_t(0.010), ra=_t(60.0), snow_depth=_t(0.30),
              snow_density=_t(250.0), swe_surface=_t(0.05),
              old_t_surf=_t(-4.0), surface_liquid_water=_t(0.0),
              blowing_mass_flux=_t(0.0))
    returned = snow.snow_pack_energy_balance(_t(-5.0), st,
                                             return_refreeze=True)
    assert len(returned) == 3, (
        'the balance must hand back the vapour flux, not only the residual '
        'and the refreeze energy')
    residual, refreeze, vapor_flux = returned
    _close(float(residual[0]), -205.55556253188175, 'residual')
    _close(float(refreeze[0]), 0.0, 'refreeze energy')
    _close(float(vapor_flux[0]), -0.00043150135035055997, 'vapour flux')


def test_the_pack_sublimates():
    """snow_melt.c:361-374 takes the vapour flux off the surface ice.

    A cold pack under dry windy air loses mass without melting a drop. VIC
    reports 1.249 mm/yr of it at Walnut Gulch; this module reported none,
    because the flux never left the energy balance.
    """
    got = _step(MELT_CASES[0][0])
    want = MELT_CASES[0][1]
    _close(got['sublimation'], want['sublimation'], 'sublimation')
    _close(got['swq'], want['swq'], 'swq')
    assert got['sublimation'] > 0.0, 'a drying pack must lose mass, not gain'


def test_the_vapour_flux_reaches_the_melt():
    """snow_melt.c:217-228 adjusts the surface liquid before the melt routes.

    Deposition onto a melting pack is water that leaves with the melt in the
    same step, so dropping it understates the outflow -- by 2.2% in case 1
    and 5.3% in case 4, where the whole pack melts out.
    """
    for i in (1, 3, 4):
        case, want = MELT_CASES[i]
        got = _step(case)
        _close(got['melt'], want['melt'], 'case %d melt' % i)
        _close(got['sublimation'], want['sublimation'],
               'case %d sublimation' % i)


def test_surface_liquid_freezes_when_the_surface_solves_below_zero():
    """snow_melt.c:355-357: all of it, into the ice, at once.

    Leaving it liquid loses the phase change and hands the layer rebalance
    the wrong water equivalent, which is why the pack temperature moves too.
    """
    case, want = MELT_CASES[5]
    got = _step(case)
    assert got['surf_water'] == 0.0, (
        'a surface layer at %.3f C cannot hold %.6f m of liquid'
        % (got['surf_temp'], got['surf_water']))
    _close(got['pack_temp'], want['pack_temp'], 'pack temperature')
    _close(got['swq'], want['swq'], 'swq')


def test_the_old_surface_temperature_predates_the_snowfall():
    """snow_melt.c:107 reads OldTSurf before line 132-164 overwrites it.

    The change-in-cold-content term is measured against the temperature the
    surface layer had at the start of the step. Measuring it against the
    post-mixing temperature instead makes it the new snow's own temperature,
    and the solved surface comes out 0.77 K too cold here.
    """
    case, want = MELT_CASES[6]
    got = _step(case)
    _close(got['surf_temp'], want['surf_temp'], 'surface temperature')
    assert abs(got['surf_temp'] - (-12.2791983106)) > 0.1, (
        'this is the value the pre-audit code returned; seeing it again '
        'means OldTSurf is being taken after the snowfall is mixed in')


def test_the_pack_is_compacted_and_its_depth_recomputed():
    """solve_snow.c:358-372, which snow_melt() leaves to its caller.

    Depth and density feed the ground conduction term of the next step's
    balance. `snow_step` used to return neither, so a caller had nothing to
    carry and the term stayed at whatever constant it was handed.
    """
    for i, (case, want) in enumerate(MELT_CASES):
        got = _step(case)
        _close(got['density'], want['density'], 'case %d density' % i)
        _close(got['depth'], want['depth'], 'case %d depth' % i)


def test_the_ground_flux_has_no_threshold_of_its_own():
    """SnowPackEnergyBalance.c:252 tests `SnowDepth > 0.` and nothing else.

    A 1e-8 m guard sat here, which zeroed a conduction term VIC computes.
    At 1e-9 m of snow the term is large, and the point is that VIC keeps it.
    """
    st = dict(t_air=_t(-5.0), t_grnd=_t(0.0), wind=_t(2.0),
              net_short=_t(50.0), long_snow_in=_t(250.0), air_dens=_t(1.25),
              eact_air=_t(200.0), lv=_t(2.5e6), press=_t(90000.0),
              vpd=_t(100.0), rain=_t(0.0), dt=_t(3600.0), z=_t(2.0),
              z0=_t(0.010), ra=_t(60.0), snow_depth=_t(1.0e-9),
              snow_density=_t(250.0), swe_surface=_t(1.0e-7),
              old_t_surf=_t(-5.0), surface_liquid_water=_t(0.0),
              blowing_mass_flux=_t(0.0))
    with_snow = float(snow.snow_pack_energy_balance(_t(-5.0), st)[0])
    st_bare = dict(st, snow_depth=_t(0.0))
    without = float(snow.snow_pack_energy_balance(_t(-5.0), st_bare)[0])
    # SNOW_CONDUCT * 250^2 * (0 - -5) / 1e-9 / 3600, counted twice as VIC
    # counts it -- once as the flux and once through the cold content.
    expect = 2.0 * snow.SNOW_CONDUCT * 250.0 * 250.0 * 5.0 / 1.0e-9 / 3600.0
    assert abs((with_snow - without) - expect) <= 1e-6 * abs(expect), (
        'the conduction term is being zeroed below a depth VIC still uses')


# ---------------------------------------------------------------------------
# The whole step, and the table's own provenance.

@pytest.mark.parametrize('index', range(len(MELT_CASES)))
def test_the_whole_step_matches_vic(index):
    """Every state variable of snow_melt(), not only the ones we plot."""
    case, want = MELT_CASES[index]
    got = _step(case)
    for name in ('melt', 'swq', 'pack_water', 'surf_water', 'surf_temp',
                 'pack_temp', 'sublimation', 'depth', 'density'):
        _close(got[name], want[name], 'case %d %s' % (index, name))


def test_the_recorded_reference_still_matches_vic():
    """Re-read the oracle, so the table above cannot drift away from it.

    Skipped, loudly, when ref_snow has not been built: a table checked
    only against itself is not a check, and pretending otherwise is how the
    400 kg/m3 ceiling survived.
    """
    if not os.path.exists(REF):
        pytest.skip('ref_snow missing; '
                    'run tests/oracle/build_ref_snow.sh to check the table '
                    'against VIC 5.0.1 itself')
    out = subprocess.run([REF, 'melt'], capture_output=True, text=True,
                         check=True).stdout
    rows = [line for line in out.splitlines()
            if line and not line.startswith('#') and not line[0].isalpha()]
    assert len(rows) == len(MELT_CASES)
    names = ('melt', 'swq', 'pack_water', 'surf_water', 'surf_temp',
             'pack_temp', 'sublimation')
    for row, (_, want) in zip(rows, MELT_CASES):
        fields = [float(x) for x in row.split('|')[1:]]
        for j, name in enumerate(names):
            assert fields[j] == want[name], (
                '%s in the recorded table is %.17g; VIC 5.0.1 now says %.17g'
                % (name, want[name], fields[j]))
        assert fields[10] == want['depth']
        assert fields[11] == want['density']


def test_the_step_still_carries_gradients():
    """The fixes must not have severed the graph they were added to.

    `sublimation` in particular is a new output and is the one a calibration
    against OUT_SUB_SNOW would score, so it has to be differentiable in the
    forcing it comes from.
    """
    case = dict(MELT_CASES[0][0])
    eact = _t(case['eact']).requires_grad_(True)
    state = dict(ice=_t(case['swq']), surf_water=_t(0.0), pack_water=_t(0.0),
                 surf_temp=_t(case['surf_temp']),
                 pack_temp=_t(case['pack_temp']))
    forcing = dict(snowfall=_t(0.0), rain=_t(0.0), air_temp=_t(case['t_air']),
                   t_grnd=_t(case['t_grnd']), wind=_t(case['wind']),
                   net_short=_t(case['net_short']),
                   long_snow_in=_t(case['long_in']),
                   air_dens=_t(case['air_dens']), eact_air=eact,
                   lv=_t(case['lv']), press=_t(case['press']),
                   vpd=_t(case['vpd']), dt=_t(case['dt']), z=_t(case['z']),
                   z0=_t(case['z0']), ra=_t(case['ra']),
                   snow_depth=_t(case['depth']),
                   snow_density=_t(case['density']))
    out, _ = snow.snow_step(state, forcing)
    out['sublimation'].sum().backward()
    assert eact.grad is not None and torch.isfinite(eact.grad).all()
    # Wetter air sublimates less, so the derivative has a sign and it is this
    # one. A zero would mean the flux had been detached rather than computed.
    assert float(eact.grad[0]) < 0.0


def test_mass_closes_once_the_sublimation_is_counted():
    """snow_melt.c:493 is the balance, and the vapour flux is a term in it.

        MassBalanceError = (InitialSwq - swq) + (RainFall + SnowFall)
                           - melt + vapor_flux

    Before the audit the pack exchanged no vapour, so `snowfall - melt` alone
    closed the books and a test could check it that way. It cannot any more:
    tests/test_snow_step.py still balances without the term and now
    reports a residual of 0.297 mm over 48 cold steps and 0.744 mm over 200
    warm ones -- which is not a leak, it is exactly the mass that went to or
    came from the air. This is that check written with the term included.
    """
    n = 3

    def const(v):
        return torch.full((n,), v, dtype=DTYPE, device='cpu')

    def forcing(air_temp, net_short, long_in, snowfall):
        return dict(t_grnd=const(0.0), wind=const(2.0),
                    net_short=const(net_short), long_snow_in=const(long_in),
                    air_dens=const(1.2), eact_air=const(400.0),
                    lv=const(2.5e6), press=const(101300.0), vpd=const(200.0),
                    rain=const(0.0), dt=const(3600.0), z=const(2.0),
                    z0=const(0.01), ra=const(60.0), snow_depth=const(0.4),
                    snow_density=const(250.0), air_temp=const(air_temp),
                    snowfall=const(snowfall))

    for steps, air_temp, net_short, long_in, snowfall in (
            (48, -6.0, 40.0, 250.0, 0.001),
            (200, 8.0, 250.0, 330.0, 0.0)):
        state = dict(ice=const(0.10), surf_water=const(0.0),
                     pack_water=const(0.0), surf_temp=const(-5.0),
                     pack_temp=const(-3.0))
        store0 = state['ice'] + state['surf_water'] + state['pack_water']
        melt_total = const(0.0)
        snow_total = const(0.0)
        sublimated = const(0.0)
        for _ in range(steps):
            f = forcing(air_temp, net_short, long_in, snowfall)
            state, melt = snow.snow_step(state, f)
            melt_total = melt_total + melt
            snow_total = snow_total + f['snowfall']
            sublimated = sublimated + state['sublimation']
        store1 = state['ice'] + state['surf_water'] + state['pack_water']
        residual = (store1 - store0) - (snow_total - melt_total) + sublimated
        assert float(residual.abs().max()) < 1e-14, (
            '%d steps at %g C leak %.3g m of water'
            % (steps, air_temp, float(residual.abs().max())))
        # And the omitted term really is the size of the discrepancy, so that
        # this test would fail if sublimation were quietly zeroed again.
        assert float(sublimated.abs().max()) > 1e-5


def test_the_pack_keeps_its_ice_when_drawn_down_through_the_layer_maximum():
    """snow_melt.c:454-485. `Ice` is the total taken before the layers are
    rebalanced, and snow->swq is built from it; in the Ice <= MAX branch the
    C discards its local PackSwq without folding it into SurfaceSwq, which
    costs nothing there because the layers are re-derived from Ice on the
    next call (:114-120).

    Summing the two layers *after* that branch instead of keeping Ice drops
    the pack's ice: a pack of exactly SNOW_MAX_SURFACE_SWE plus a little
    surface liquid, cold and under a strong vapour flux, refreezes the liquid
    into a few hundredths of a millimetre of pack, sublimates the surface
    below the maximum some steps later, and at that step lost the whole pack
    layer -- 3.84e-5 m here, and the 1.5e-2 mm of 1977-01-24 at FindleyLake
    that was the decade's entire water-balance residual.
    """
    n = 1

    def const(v):
        return torch.full((n,), v, dtype=DTYPE, device='cpu')

    def forcing():
        return dict(t_grnd=const(0.0), wind=const(6.0), net_short=const(0.0),
                    long_snow_in=const(180.0), air_dens=const(1.2),
                    eact_air=const(50.0), lv=const(2.5e6),
                    press=const(85000.0), vpd=const(400.0), rain=const(0.0),
                    dt=const(3600.0), z=const(10.0), z0=const(0.03),
                    ra=const(30.0), snow_depth=const(0.4),
                    snow_density=const(250.0), air_temp=const(-10.0),
                    snowfall=const(0.0))

    state = dict(ice=const(snow.SNOW_MAX_SURFACE_SWE), surf_water=const(1e-3),
                 pack_water=const(0.0), surf_temp=const(-6.0),
                 pack_temp=const(-4.0))
    crossed = False
    for k in range(24):
        pack_before = float(state.get('pack_swq', const(0.0)))
        store0 = state['ice'] + state['surf_water'] + state['pack_water']
        state, melt = snow.snow_step(state, forcing())
        store1 = state['ice'] + state['surf_water'] + state['pack_water']
        residual = float(((store0 - store1) - melt - state['sublimation']).abs().max())
        assert residual < 1e-14, (
            'step %d: %.3g m of ice left the store unreported (pack before '
            'the step %.3g m)' % (k, residual, pack_before))
        if pack_before > 0.0 and float(state['pack_swq']) == 0.0:
            crossed = True
    # The regime must actually have been exercised: the pack existed and was
    # drawn to nothing through the layer maximum within the 24 steps.
    assert crossed, 'the pack never crossed the layer maximum -- test is inert'


def test_returned_energy_terms_close_at_the_solved_temperature():
    """snow_step returns the balance's terms at the branch it took. On the
    frozen branch the root satisfies rest + refreeze = 0; on the melting
    branch refreeze is -rest by construction (SnowPackEnergyBalance.c). Either
    way net_rad + sensible + latent + latent_sub + advection + grnd_flux -
    delta_cc + refreeze_energy is zero to the solver's tolerance, unless the
    pack melted completely, when refreeze_energy is rescaled to the melt
    available (snow_melt.c:256-258) and the sum is no longer zero.
    """
    n = 1

    def const(v):
        return torch.full((n,), v, dtype=DTYPE, device='cpu')

    def forcing(air, sw, li, eact):
        return dict(t_grnd=const(0.0), wind=const(3.0), net_short=const(sw),
                    long_snow_in=const(li), air_dens=const(1.2),
                    eact_air=const(eact), lv=const(2.5e6),
                    press=const(85000.0), vpd=const(200.0), rain=const(0.0),
                    dt=const(3600.0), z=const(10.0), z0=const(0.03),
                    ra=const(40.0), snow_depth=const(0.4),
                    snow_density=const(250.0), air_temp=const(air),
                    snowfall=const(0.0))

    # A cold pack under warm air refreezes its liquid before it melts, so
    # the melting case starts isothermal at 0 C with no cold content to
    # absorb the surplus.
    for air, sw, li, eact, t0 in ((-8.0, 0.0, 200.0, 150.0, -3.0),   # frozen
                                  (6.0, 400.0, 350.0, 800.0, 0.0)):   # melting
        state = dict(ice=const(0.3), surf_water=const(1e-3),
                     pack_water=const(0.0), surf_temp=const(t0),
                     pack_temp=const(t0), depth=const(1.2),
                     density=const(250.0))
        state, melt = snow.snow_step(state, forcing(air, sw, li, eact))
        e = state['energy']
        assert float(state['ice']) > 0.0, 'pack melted completely; case inert'
        total = (e['net_rad'] + e['sensible'] + e['latent'] + e['latent_sub']
                 + e['advection'] + e['grnd_flux'] - e['delta_cc']
                 + e['refreeze_energy'])
        assert float(total.abs().max()) < 1e-6, (
            'at %g C the returned terms leave %.3e W/m2 unaccounted'
            % (air, float(total.abs().max())))
    # The melting case must actually have melted. `melt` is the pack's
    # outflow (snow_melt.c:441-447), and a 0.3 m pack holds
    # SNOW_LIQUID_WATER_CAPACITY * swq = 10.5 mm of liquid before it releases
    # any, so an hour's melt stays inside; the ice going down is the signal.
    assert float(state['ice']) < 0.3, 'the 0 C pack under 400 W/m2 did not melt'
