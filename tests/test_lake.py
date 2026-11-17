"""torchvic.core.lake against pristine VIC 5.0.1.

LAKES is the last of VIC's five big physics switches this model did not
have. The C is lakes.eb.c, 2,683 lines; this covers the closed-form layer it
is built on -- the state functions of water, the drag and turbulent fluxes
at the surface, and the three albedos -- each against upstream's own
compiled function through `ref_snow lake`.

Nothing here iterates or holds state, so every function is exact against
VIC rather than close to it, and the tests say so with tolerances that
would fail on a transcription slip rather than tolerances chosen to pass.
"""
import pathlib
import subprocess

import pytest
import torch

from torchvic.core import lake

import _paths                                           # noqa: E402

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')
ROOT = pathlib.Path(_paths.REPO)
REF = pathlib.Path(_paths.oracle('ref_snow'))
DT = torch.float64
#: Bit-exactness against VIC's compiled output is a CPU statement: CUDA's
#: `pow` differs from glibc's by about one unit in the last place, so any
#: chain that reaches `pow` -- the stability correction, the albedo aging
#: curve, the Brooks-Corey exponents -- lands a ulp away. That is measured
#: and written down in docs/differences.md, and
#: tests/test_snow_formulas.py pins its own oracle comparisons to
#: the CPU for the same reason. The comparisons below assert exact
#: agreement and so run here; `test_the_lake_agrees_between_devices` is
#: what keeps the GPU honest, by measuring the gap rather than tolerating
#: it silently.
DEV = torch.device('cpu')
#: the live default, for the one test that compares the two
LIVE = torch.empty(0).device


@pytest.fixture(scope='module')
def oracle():
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake'], capture_output=True, text=True,
                         check=True).stdout
    section, rows = None, {}
    for line in out.splitlines():
        if line.startswith('# '):
            if '|' not in line:
                section = line[2:].strip().split(',')[0]
            continue
        rows.setdefault(section, []).append(
            [float(x) for x in line.replace('|', ',').split(',')])
    return rows


def _col(rows, i):
    return torch.tensor([r[i] for r in rows], dtype=DT, device=DEV)


def test_water_density_is_the_difference_from_fresh_water(oracle):
    """lakes.eb.c:791-796 returns `rhow - CONST_RHOFW`, not rhow. A caller
    who took it for a density would have lakes a thousand times too light,
    so this pins the offset as well as the polynomial."""
    r = oracle['calc_density']
    t = _col(r, 1)
    got = lake.calc_density(t)
    assert float((got - _col(r, 3)).abs().max()) < 1e-14
    # Fresh water is densest near 4 C, which the polynomial must reproduce
    # or the lake would never overturn: a column that gets denser all the way
    # down to freezing has no spring or autumn mixing.
    d = lake.calc_density(torch.tensor([0.0, 4.0, 8.0, 25.0], dtype=DT,
                                       device=DEV))
    assert float(d[1]) > float(d[0]) and float(d[1]) > float(d[2])
    assert float(d[3]) < float(d[2])           # and it falls away with warmth
    assert abs(float(d[1])) < 0.05             # within 0.05 kg/m3 of 1000


def test_specific_heat_matches(oracle):
    r = oracle['calc_density']
    got = lake.specheat(_col(r, 1))
    ref = _col(r, 4)
    assert float(((got - ref).abs() / ref).max()) < 1e-15


def test_freezing_point_matches(oracle):
    """rhoinit, lakes.eb.c:1255-1278. Salinity is zero in VIC, so only the
    pressure term survives; the answer is a few thousandths of a degree."""
    r = oracle['calc_density']
    got = lake.freezing_point(_col(r, 2) / 1000.0)
    assert float((got - _col(r, 5)).abs().max()) < 1e-17
    assert float(got.max()) < 0.0            # pressure only depresses it


def test_drag_coefficient_matches_across_stability(oracle):
    """lkdrag over both Richardson branches and both roughnesses."""
    r = oracle['lkdrag']
    got = lake.lkdrag(_col(r, 1), _col(r, 2), _col(r, 3), _col(r, 4),
                      _col(r, 5))
    ref = _col(r, 6)
    # 1e-11 rather than machine epsilon: the coefficient is a log and a
    # square root of the C library's, and torch's differ from glibc's in the
    # last couple of bits. Still six orders tighter than a transcription slip.
    assert float(((got - ref).abs() / ref).max()) < 1e-14
    # Both branches were exercised: stable and unstable both appear.
    stable = (_col(r, 1) / _col(r, 2)) > 1.0
    assert bool(stable.any()) and bool((~stable).any())


def test_drag_never_falls_below_its_floor():
    """:1239-1246 -- the larger of a quarter of the neutral coefficient and
    6e-4. A calm, strongly stable surface would otherwise decouple from the
    atmosphere entirely."""
    n = 50
    ts = torch.full((n,), 260.0, dtype=DT, device=DEV)
    ta = torch.linspace(261.0, 320.0, n, dtype=DT, device=DEV)
    wind = torch.full((n,), 0.01, dtype=DT, device=DEV)
    rough = torch.full((n,), lake.LAKE_ZWATER, dtype=DT, device=DEV)
    z = torch.full((n,), 10.0, dtype=DT, device=DEV)
    cdrn = (lake.CONST_KARMAN / torch.log(z / rough)) ** 2
    got = lake.lkdrag(ts, ta, wind, rough, z)
    assert float((got - torch.clamp(0.25 * cdrn, min=6.0e-4)).min()) >= -1e-15


def test_turbulent_fluxes_match_over_water_and_over_ice(oracle):
    """latsens, lakes.eb.c:533-614. The saturation vapour pressure takes one
    curve over open water and another over ice, and the sample covers both."""
    r = oracle['latsens']
    evap, qsen = lake.latsens(*[_col(r, i) for i in range(1, 10)])
    for got, col in ((evap, 10), (qsen, 11)):
        ref = _col(r, col)
        assert float(((got - ref).abs()
                      / ref.abs().clamp(min=1e-12)).max()) < 1e-14
    assert bool((_col(r, 3) > 0.0).any()) and bool((_col(r, 3) == 0.0).any())


def test_a_frozen_lake_evaporates_less_than_an_open_one():
    """The two vapour-pressure curves are not a formality: ice holds its
    water. Same temperature, same air, one with a cover and one without."""
    one = lambda v: torch.tensor([v], dtype=DT, device=DEV)
    args = dict(t_surf_k=one(272.0), t_cutoff_k=one(lake.CONST_TKFRZ),
                tair=one(-6.0), wind=one(4.0), pressure=one(95000.0),
                vp=one(200.0), air_density=one(1.25), wind_h=one(10.0))
    open_e, _ = lake.latsens(hice=one(0.0), **args)
    iced_e, _ = lake.latsens(hice=one(0.3), **args)
    assert float(open_e) > float(iced_e)


def test_lake_albedos_match_including_the_integer_snow_age(oracle):
    """alblake, lakes.eb.c:621-706. The snow age is an integer count of
    steps and MELTING is a flag, so both are exact or wrong."""
    r = oracle['alblake']
    c = lambda i: _col(r, i)
    sa, albs, albi, albw, ls, me = lake.alblake(
        c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9) > 0.5,
        c(10).to(torch.long), c(11))
    for name, got, col in (('snowalbedo', sa, 12), ('albs', albs, 13),
                           ('albi', albi, 14), ('albw', albw, 15)):
        assert float((got - c(col)).abs().max()) < 1e-15, name
    assert float((ls - c(16)).abs().max()) == 0.0
    assert int((me.to(DT) - c(17)).abs().sum()) == 0


def test_the_lake_snowfall_threshold_is_in_millimetres():
    """lakes.eb.c:160 divides solve_lake's `snowfall` by MM_PER_M to get
    metres, so what alblake compares against SNOW_TRACESNOW is millimetres.
    The rest of this model works in metres -- torchvic.core.snow carries the same
    VIC constant as 3e-5 for that reason -- and passing metres here would
    put every snowfall below the threshold and leave the albedo clock
    running all winter, which is a mistake this project has already made
    once in the snow module.
    """
    from torchvic.core import snow
    assert lake.SNOW_TRACESNOW == 0.03
    assert snow.TRACE_SNOW_M == 3.0e-5
    assert abs(lake.SNOW_TRACESNOW - snow.TRACE_SNOW_M * 1000.0) < 1e-15
    # 0.1 mm of snow resets the age; 0.01 mm does not.
    one = lambda v: torch.tensor([v], dtype=DT, device=DEV)
    for fall, want in ((0.1, 1.0), (0.01, 6.0)):
        *_, ls, _m = lake.alblake(
            one(0.0), one(-5.0), one(0.6), one(fall), one(-1.0e5),
            one(3600.0), one(5.0), one(0.05), one(False) > 0.5,
            torch.tensor([15], device=DEV), one(47.0))
        assert float(ls) == want, fall


def test_every_lake_function_carries_a_gradient():
    """These sit under a calibration loop like everything else here."""
    t = torch.linspace(-2.0, 25.0, 12, dtype=DT, device=DEV,
                       requires_grad=True)
    (lake.calc_density(t).sum() + lake.specheat(t).sum()).backward()
    assert torch.isfinite(t.grad).all() and float(t.grad.abs().min()) > 0.0

    w = torch.linspace(0.1, 12.0, 12, dtype=DT, device=DEV,
                       requires_grad=True)
    ts = torch.full((12,), 275.0, dtype=DT, device=DEV)
    ta = torch.linspace(268.0, 290.0, 12, dtype=DT, device=DEV)
    rough = torch.full((12,), lake.LAKE_ZWATER, dtype=DT, device=DEV)
    z = torch.full((12,), 10.0, dtype=DT, device=DEV)
    lake.lkdrag(ts, ta, w, rough, z).sum().backward()
    assert torch.isfinite(w.grad).all()
    # Both Richardson branches are in that sweep, including rib exactly at
    # the sqrt's zero, which is where an unguarded transcription gives NaN.
    assert float(w.grad.abs().max()) > 0.0


# ---------------------------------------------------------------------------
# The thermal profile's helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def oracle2():
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake2'], capture_output=True, text=True,
                         check=True).stdout
    section, rows = None, {}
    for line in out.splitlines():
        if line.startswith('# '):
            if '|' not in line:
                section = line[2:].strip()
            continue
        rows.setdefault(section, []).append(
            [float(x) for x in line.replace('|', ',').split(',')])
    return rows


def _nodes(values):
    """A [node] list into a [node, 1] tensor -- one column of lake."""
    return torch.tensor([[v] for v in values], dtype=DT, device=DEV)


def test_tridiagonal_solve_matches(oracle2):
    """lakes.eb.c:1716-1758. This is the inner solve of the lake's
    temperature equation, so it has to be the C's arithmetic and not just
    the C's answer: the forward sweeps stop one node short and the last row
    is eliminated on its own, which is the same in exact arithmetic and not
    in floating point.
    """
    worst = 0.0
    for r in oracle2['tridia']:
        n, o = int(r[1]), 2
        a, o = _nodes(r[o:o + n]), o + n
        b, o = _nodes(r[o:o + n]), o + n
        c, o = _nodes(r[o:o + n]), o + n
        y, o = _nodes(r[o:o + n]), o + n
        ref = _nodes(r[o:o + n])
        got = lake.tridia(a, b, c, y)
        worst = max(worst, float(((got - ref).abs() / ref.abs()).max()))
    assert worst < 1e-14, worst


def test_the_tridiagonal_solve_actually_solves_the_system():
    """Independent of the oracle: A x must equal y."""
    n = 8
    g = torch.Generator(device=DEV).manual_seed(4)
    a = -0.3 - torch.rand(n, 3, generator=g, dtype=DT, device=DEV)
    c = -0.3 - torch.rand(n, 3, generator=g, dtype=DT, device=DEV)
    b = 3.0 + torch.rand(n, 3, generator=g, dtype=DT, device=DEV)
    y = torch.randn(n, 3, generator=g, dtype=DT, device=DEV)
    x = lake.tridia(a, b, c, y)
    resid = b * x
    resid[1:] = resid[1:] + a[1:] * x[:-1]
    resid[:-1] = resid[:-1] + c[:-1] * x[1:]
    assert float((resid - y).abs().max()) < 1e-12


def test_eddy_diffusivity_matches(oracle2):
    """lakes.eb.c:802-926, both the ice branch (molecular only) and the
    wind-stirred one."""
    worst, froze, stirred = 0.0, 0, 0
    for r in oracle2['eddy']:
        ff, n = int(r[1]), int(r[4])
        got = lake.eddy(torch.tensor([ff], device=DEV),
                        torch.tensor([r[2]], dtype=DT, device=DEV),
                        _nodes(r[7:7 + n]),
                        torch.tensor([r[3]], dtype=DT, device=DEV),
                        n, r[5], r[6])
        ref = _nodes(r[7 + n:7 + 2 * n])
        worst = max(worst, float(((got - ref).abs() / ref.abs()).max()))
        froze += ff != 1
        stirred += ff == 1
    assert worst < 1e-14, worst
    assert froze and stirred


def test_under_ice_the_column_only_diffuses_molecularly():
    """:838-843 -- no wind reaches the water, so the profile collapses to
    LAKE_DM everywhere. A stratified column with a gale over it must still
    give the same number as a still one."""
    rho = lake.calc_density(torch.tensor([[18.0], [12.0], [6.0], [2.0]],
                                         dtype=DT, device=DEV))
    lat = torch.tensor([47.0], dtype=DT, device=DEV)
    for wind in (0.5, 20.0):
        de = lake.eddy(torch.tensor([0], device=DEV),
                       torch.tensor([wind], dtype=DT, device=DEV),
                       rho, lat, 4, 0.5, 0.2)
        assert float((de - lake.LAKE_DM).abs().max()) == 0.0


def test_column_average_matches(oracle2):
    """colavg, lakes.eb.c:717-772 -- the water and ice columns weighted by
    their masses, and the density that follows."""
    wt, wd = 0.0, 0.0
    for r in oracle2['colavg']:
        n = int(r[2])
        f, d = lake.colavg(_nodes(r[5:5 + n]), _nodes(r[5 + n:5 + 2 * n]),
                           torch.tensor([r[1]], dtype=DT, device=DEV),
                           n, r[3], r[4])
        wt = max(wt, float((f - _nodes(r[5 + 2 * n:5 + 3 * n])).abs().max()))
        wd = max(wd, float((d - _nodes(r[5 + 3 * n:5 + 4 * n])).abs().max()))
    assert wt < 1e-13 and wd < 1e-14


def test_the_column_average_is_bounded_by_its_two_columns():
    """A weighted mean of the water and ice temperatures cannot leave the
    interval they span, whatever the ice fraction."""
    tw = torch.tensor([[15.0], [9.0], [4.0]], dtype=DT, device=DEV)
    ti = torch.tensor([[0.5], [0.2], [0.0]], dtype=DT, device=DEV)
    for frac in (0.0, 0.3, 0.7, 1.0):
        f, _d = lake.colavg(tw, ti, torch.tensor([frac], dtype=DT,
                                                 device=DEV), 3, 0.5, 0.2)
        assert bool((f <= torch.maximum(tw, ti) + 1e-12).all())
        assert bool((f >= torch.minimum(tw, ti) - 1e-12).all())


def test_column_heat_content_matches(oracle2):
    """energycalc, lakes.eb.c:1765-1806. The deepest layer takes half of its
    upper boundary's area alone (:1788-1793), because the lake closes to a
    point below it; every other layer takes the mean of the two boundaries
    that bound it."""
    worst = 0.0
    for r in oracle2['energycalc']:
        n, o = int(r[1]), 4
        ft, o = _nodes(r[o:o + n]), o + n
        cp, o = _nodes(r[o:o + n]), o + n
        de, o = _nodes(r[o:o + n]), o + n
        su, o = _nodes(r[o:o + n + 1]), o + n + 1
        got = lake.energycalc(ft, n, r[2], r[3], su, cp, de)
        ref = torch.tensor([r[o]], dtype=DT, device=DEV)
        worst = max(worst, float(((got - ref).abs() / ref.abs()).max()))
    assert worst < 1e-14, worst


def test_the_thermal_helpers_carry_gradients():
    rho = lake.calc_density(torch.tensor([[18.0], [12.0], [6.0], [2.0]],
                                         dtype=DT, device=DEV))
    wind = torch.tensor([4.0], dtype=DT, device=DEV, requires_grad=True)
    lat = torch.tensor([47.0], dtype=DT, device=DEV)
    lake.eddy(torch.tensor([1], device=DEV), wind, rho, lat, 4, 0.5,
              0.2).sum().backward()
    assert torch.isfinite(wind.grad).all() and float(wind.grad.abs()) > 0.0

    n = 5
    b = torch.full((n, 1), 3.0, dtype=DT, device=DEV, requires_grad=True)
    a = torch.full((n, 1), -0.4, dtype=DT, device=DEV)
    c = torch.full((n, 1), -0.5, dtype=DT, device=DEV)
    y = torch.ones(n, 1, dtype=DT, device=DEV)
    lake.tridia(a, b, c, y).sum().backward()
    assert torch.isfinite(b.grad).all() and float(b.grad.abs().max()) > 0.0


# ---------------------------------------------------------------------------
# The temperature solve and the convective mixing
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def oracle3():
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake3'], capture_output=True, text=True,
                         check=True).stdout
    section, rows = None, {}
    for line in out.splitlines():
        if line.startswith('# '):
            if '|' not in line:
                section = line[2:].strip()
            continue
        rows.setdefault(section, []).append(
            [float(x) for x in line.replace('|', ',').split(',')])
    return rows


def test_temperature_solve_matches(oracle3):
    """temp_area, lakes.eb.c:1294-1560 -- Crank-Nicolson diffusion on a
    column whose area shrinks with depth, forced by a two-band shortwave.
    Ten columns of three to seven nodes."""
    wt = we = wj = 0.0
    one = lambda v: torch.tensor([v], dtype=DT, device=DEV)
    for r in oracle3['temp_area']:
        n, o = int(r[1]), 8
        t, o = _nodes(r[o:o + n]), o + n
        rho, o = _nodes(r[o:o + n]), o + n
        de, o = _nodes(r[o:o + n]), o + n
        cp, o = _nodes(r[o:o + n]), o + n
        su, o = _nodes(r[o:o + n + 1]), o + n + 1
        ref_t, o = _nodes(r[o:o + n]), o + n
        tn, eob, jn = lake.temp_area(one(r[2]), one(r[3]), one(r[4]), t, rho,
                                     de, r[5], su, n, r[6], r[7], cp)
        wt = max(wt, float(((tn - ref_t).abs()
                            / ref_t.abs().clamp(min=1e-9)).max()))
        we = max(we, float((eob - one(r[o])).abs() / abs(r[o])))
        wj = max(wj, float((jn - one(r[o + 1])).abs() / abs(r[o + 1])))
    assert wt < 1e-14 and we < 1e-14 and wj < 1e-14


def test_an_unforced_lake_only_diffuses_molecularly():
    """The solve's own consistency check. With no sun and no surface flux,
    the only thing left is molecular diffusion: LAKE_DM is 1.4e-7 m2/s, so
    across a 4 C/m gradient one second moves the column by a few millionths
    of a degree and nothing else happens.""" 
    n = 5
    t = torch.tensor([[14.0], [12.0], [10.0], [8.0], [6.0]], dtype=DT,
                     device=DEV)
    rho = lake.calc_density(t)
    cp = lake.specheat(t)
    de = torch.full_like(t, lake.LAKE_DM)
    su = torch.tensor([[1e6]] * (n + 1), dtype=DT, device=DEV)
    zero = torch.zeros(1, dtype=DT, device=DEV)
    tn, eob, _j = lake.temp_area(zero, zero, zero, t, rho, de, 1.0, su, n,
                                 0.5, 0.2, cp)
    moved = float((tn - t).abs().max())
    assert 0.0 < moved < 1e-5, moved
    assert float(eob.abs()) == 0.0
    # And it moves heat from the warm top downward, not the other way.
    assert float(tn[0]) < float(t[0]) and float(tn[-1]) > float(t[-1])


def test_convective_mixing_matches_including_the_restart(oracle3):
    """tracer_mixer, lakes.eb.c:1567-1692. The sample covers a stable
    column, one inverted at the top, one inverted in the middle, and ones
    that make a new inversion when mixed -- which is the `mixprev = 0;
    k = -1` restart at :1674-1678, the part that is hard to vectorise.

    `mixdepth` is an integer VIC carries forward, so it is exact or wrong.
    """
    wt, bad = 0.0, 0
    for r in oracle3['tracer_mixer']:
        n, o = int(r[1]), 5
        t, o = _nodes(r[o:o + n]), o + n
        cp, o = _nodes(r[o:o + n]), o + n
        su, o = _nodes(r[o:o + n + 1]), o + n + 1
        ref_t = _nodes(r[o:o + n])
        got, _rho, md = lake.tracer_mixer(
            t, torch.tensor([int(r[4])], device=DEV), su, n, r[2], r[3], cp)
        wt = max(wt, float((got - ref_t).abs().max()))
        bad += int(md) != int(r[o + n])
    assert wt < 1e-13, wt
    assert bad == 0


def test_mixing_leaves_a_stable_column():
    """Whatever it does, it must stop: no layer may be denser than the one
    below it when it is finished. That is the criterion the scan itself
    uses, so a column that fails this has been left mid-adjustment."""
    g = torch.Generator(device=DEV).manual_seed(11)
    n, ncol = 9, 24
    t = 2.0 + 18.0 * torch.rand(n, ncol, generator=g, dtype=DT, device=DEV)
    cp = lake.specheat(t)
    su = torch.linspace(1e6, 2e5, n + 1, dtype=DT,
                        device=DEV).unsqueeze(-1).expand(n + 1, ncol)
    md = torch.zeros(ncol, dtype=torch.long, device=DEV)
    got, rho, _md = lake.tracer_mixer(t, md, su, n, 0.5, 0.2, cp)
    assert float((rho[:-1] - rho[1:]).max()) <= 1e-12


def test_mixing_conserves_heat_within_each_mixed_block():
    """A column mixed to a single temperature must hold the heat it held:
    the mass-weighted mean is the point. Checked on a fully inverted column,
    which mixes end to end."""
    n = 6
    t = torch.linspace(2.0, 20.0, n, dtype=DT,
                       device=DEV).unsqueeze(-1)     # coldest at the top
    cp = lake.specheat(t)
    su = torch.full((n + 1, 1), 1.0e6, dtype=DT, device=DEV)
    dz, surfdz = 0.5, 0.5
    rho0 = lake.calc_density(t)
    z = torch.full_like(t, dz)
    hc0 = z * (1.0e3 + rho0) * cp * su[:n]
    before = float((t * hc0).sum())
    got, rho1, _md = lake.tracer_mixer(
        t, torch.zeros(1, dtype=torch.long, device=DEV), su, n, dz, surfdz,
        cp)
    # Fully mixed, so one temperature throughout.
    assert float((got - got[0]).abs().max()) < 1e-12
    hc1 = z * (1.0e3 + rho1) * lake.specheat(got) * su[:n]
    after = float((got * hc1).sum())
    # Within a per cent: the heat capacity and density are re-evaluated at
    # the mixed temperature, which VIC does too, so this is not an identity.
    assert abs(after - before) / abs(before) < 0.01


# ---------------------------------------------------------------------------
# Ice
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def oracle4():
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake4'], capture_output=True, text=True,
                         check=True).stdout
    section, rows = None, {}
    for line in out.splitlines():
        if line.startswith('# '):
            if '|' not in line:
                section = line[2:].strip()
            continue
        rows.setdefault(section, []).append(
            [float(x) for x in line.replace('|', ',').split(',')])
    return rows


def _one(v):
    return torch.tensor([v], dtype=DT, device=DEV)


def test_new_ice_formation_matches(oracle4):
    """iceform, lakes.eb.c:936-1038. Every node below freezing gives up the
    heat that would take it back, and the heat becomes ice -- limited by the
    water there is. The sample reaches all three of the C's branches at
    :1002-1022: enough water, some, and none."""
    worst = {k: 0.0 for k in ('T', 'qf', 'aa', 'h', 'eq')}
    for r in oracle4['iceform']:
        n, o = int(r[1]), 8
        t, o = _nodes(r[o:o + n]), o + n
        cp, o = _nodes(r[o:o + n]), o + n
        su, o = _nodes(r[o:o + n + 1]), o + n + 1
        ref_t, o = _nodes(r[o:o + n]), o + n
        got_t, qf, aa, h, eq = lake.iceform(
            t, _one(r[2]), _one(r[3]), n, r[4], r[5], r[6], cp, su, _one(r[7]))
        rel = lambda g, v: float(((g - _one(v)).abs()
                                  / max(abs(v), 1e-12)).max())
        worst['T'] = max(worst['T'], float((got_t - ref_t).abs().max()))
        worst['qf'] = max(worst['qf'], rel(qf, r[o]))
        worst['aa'] = max(worst['aa'], rel(aa, r[o + 1]))
        worst['h'] = max(worst['h'], rel(h, r[o + 2]))
        worst['eq'] = max(worst['eq'], rel(eq, r[o + 3]))
    for k, v in worst.items():
        assert v < 1e-14, (k, v)


def test_new_ice_spreads_before_it_thickens():
    """:1027-1037 -- ice covers the open water at LAKE_FRACMIN thickness
    first and only thickens once there is nowhere left to spread."""
    n = 4

    def run(area_m2, supercool):
        t = torch.full((n, 1), -supercool, dtype=DT, device=DEV)
        cp = lake.specheat(t)
        su = torch.full((n + 1, 1), area_m2, dtype=DT, device=DEV)
        return lake.iceform(t, _one(0.0), _one(0.0), n, 3600.0, 0.5, 0.2,
                            cp, su, _one(1.0e9))

    # Barely supercooled over a wide lake: the ice spreads at LAKE_FRACMIN
    # and covers only part of the open water.
    _t, _q, aa, h, _e = run(1.0e6, 0.01)
    assert abs(float(h) - lake.LAKE_FRACMIN) < 1e-15
    assert float(aa) < 1.0e6
    # Strongly supercooled over the same lake: there is nowhere left to
    # spread, so the cover thickens instead and takes the whole surface.
    _t, _q, aa, h, _e = run(1.0e6, 5.0)
    assert float(h) > lake.LAKE_FRACMIN
    assert abs(float(aa) - 1.0e6) < 1e-6


def test_shortwave_through_snow_and_ice_matches(oracle4):
    """icerad, lakes.eb.c:1043-1106, Patterson and Hamblin equations 7 and 8."""
    r = oracle4['icerad']
    c = lambda i: torch.tensor([x[i] for x in r], dtype=DT, device=DEV)
    ac, swnet, under = lake.icerad(c(1), c(2), c(3))
    for got, col in ((ac, 4), (swnet, 5), (under, 6)):
        ref = c(col)
        assert float(((got - ref).abs()
                      / ref.abs().clamp(min=1e-12)).max()) < 1e-14


def test_snow_shuts_the_light_off_faster_than_ice():
    """LAKE_LAMSSW is 6 /m and LAKE_LAMISW is 1.5 /m, so a few centimetres
    of snow do what tens of centimetres of ice do. If these were ever
    swapped the lake would go on absorbing light under a snowdrift."""
    # `SW_under_ice` is the shortwave ABSORBED in the cover, so a bigger
    # number means less light reaches the water. Same thickness of each.
    sw = _one(400.0)
    _a, _s, absorbed_ice = lake.icerad(sw, _one(0.05), _one(0.0))
    _a, _s, absorbed_snow = lake.icerad(sw, _one(0.0), _one(0.05))
    assert float(absorbed_snow) > float(absorbed_ice)
    # And the combined conductivity is the series resistance, written over a
    # common denominator; zero thickness means zero resistance.
    a0, _s, _u = lake.icerad(sw, _one(0.0), _one(0.0))
    assert float(a0) == 0.0


def test_ice_growth_at_the_base_matches(oracle4):
    """lakeice, lakes.eb.c:1111-1192, with both the freezing branch and its
    water limit, and the melting branch that VIC does not limit."""
    r = oracle4['lakeice']
    c = lambda i: torch.tensor([x[i] for x in r], dtype=DT, device=DEV)
    qf, em, eq, _dib = lake.lakeice(c(1), c(2), r[0][3], c(4), c(5), c(6),
                                    c(7), c(8), c(9))
    for got, col in ((qf, 10), (em, 11), (eq, 12)):
        ref = c(col)
        assert float(((got - ref).abs()
                      / ref.abs().clamp(min=1e-12)).max()) < 1e-14


def test_freezing_cannot_use_water_that_is_not_there():
    """:1170-1183 -- a lake whose water is already all ice freezes no more,
    however much heat is taken out of it."""
    # Positive energy_ice_melt_bot is freezing (lakes.eb.c:1146), so the
    # heat flux into the ice has to exceed what the water supplies.
    args = dict(sw_ice=_one(0.0), fracice=_one(1.0), dt=3600.0,
                snowflux=_one(200.0), qw=_one(0.0), sw_absorbed=_one(0.0),
                sarea=_one(1.0e5))
    # All the water is already ice.
    _q, _e, eq, dibot = lake.lakeice(ice_water_eq=_one(500.0),
                                     volume=_one(500.0), **args)
    assert float(eq) == 500.0 and float(dibot) == 0.0
    # Some water left: it freezes exactly what is left and no more.
    _q, _e, eq, _d = lake.lakeice(ice_water_eq=_one(400.0),
                                  volume=_one(500.0), **args)
    assert float(eq) <= 500.0 + 1e-9


def test_the_ice_routines_carry_gradients():
    hi = torch.linspace(0.02, 1.0, 10, dtype=DT, device=DEV,
                        requires_grad=True)
    hs = torch.full((10,), 0.04, dtype=DT, device=DEV)
    _a, swnet, under = lake.icerad(torch.full((10,), 300.0, dtype=DT,
                                              device=DEV), hi, hs)
    (swnet.sum() + under.sum()).backward()
    assert torch.isfinite(hi.grad).all() and float(hi.grad.abs().min()) > 0.0


# ---------------------------------------------------------------------------
# The basin: stage, area and volume
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def hypsometry():
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake5'], capture_output=True, text=True,
                         check=True).stdout
    cases, queries = {}, {}
    for line in out.splitlines():
        if line.startswith('#'):
            continue
        if line.startswith('q,'):
            p = line.replace('|', ',').split(',')[1:]
            queries.setdefault(int(p[0]), []).append(
                [float(x) for x in p[1:]])
        else:
            p = [float(x) for x in line.replace('|', ',').split(',')]
            cases[int(p[0])] = p
    return cases, queries


def _basin(p):
    n = int(p[1])
    z = _nodes(p[4:4 + n + 1])
    basin = _nodes(p[4 + n + 1:4 + 2 * (n + 1)])
    return z, basin, _one(p[2]), _one(p[3])       # z, basin, maxdepth, maxvol


def test_basin_geometry_matches(hypsometry):
    """get_sarea, get_volume and get_depth (lake_utils.c:30-192) over six
    basins of four to seven levels, at twenty-one depths each including ones
    past the basin's top."""
    cases, queries = hypsometry
    ws = wv = wd = 0.0
    for c, p in sorted(cases.items()):
        z, basin, maxd, maxv = _basin(p)
        for q in queries[c]:
            d = _one(q[0])
            sa = lake.get_sarea(z, basin, d)
            vo = lake.get_volume(z, basin, d, maxv)
            bk = lake.get_depth(z, basin, vo, maxv, maxd)
            ws = max(ws, abs(float(sa) - q[2]) / max(abs(q[2]), 1.0))
            wv = max(wv, abs(float(vo) - q[3]) / max(abs(q[3]), 1.0))
            wd = max(wd, abs(float(bk) - q[4]) / max(abs(q[4]), 1e-9))
    assert ws < 1e-14 and wv < 1e-14 and wd < 1e-14


def test_depth_and_volume_invert_one_another(hypsometry):
    """Independent of the oracle. These two are used in both directions
    every step -- the water balance adds water and asks for the new stage,
    the energy balance asks for the area at that stage -- so a pair that did
    not invert would leak water with every step."""
    cases, _q = hypsometry
    for _c, p in sorted(cases.items()):
        z, basin, maxd, maxv = _basin(p)
        d = torch.linspace(0.0, float(maxd), 25, dtype=DT,
                           device=DEV).unsqueeze(-1)
        v = lake.get_volume(z, basin, d, maxv)
        back = lake.get_depth(z, basin, v, maxv, maxd)
        assert float((back - d).abs().max()) < 1e-9


def test_a_lake_above_its_banks_stands_on_its_own_area(hypsometry):
    """lake_utils.c:129-132 -- past the basin's capacity the extra water
    spreads over the full surface and nothing else. This is what lets a
    lake flood."""
    cases, _q = hypsometry
    for _c, p in sorted(cases.items()):
        z, basin, maxd, maxv = _basin(p)
        extra = 1.0e5
        d = lake.get_depth(z, basin, maxv + extra, maxv, maxd)
        assert abs(float(d) - (float(maxd) + extra / float(basin[0]))) < 1e-9
        # And its surface area is the whole basin.
        assert float(lake.get_sarea(z, basin, d)) == float(basin[0])


def test_above_the_basin_vics_own_pair_does_not_invert(hypsometry):
    """And this reproduces that rather than repairing it.

    get_volume sets `*volume = maxvolume` at lake_utils.c:84-87 when the
    depth is over the top level and then falls into the accumulation loop
    without returning, so every level is added on top of the cap: a lake one
    level over its banks reports twice its capacity. VIC signals it with a
    status of 1 and leaves the caller to notice; the callers in lakes.eb.c
    are written around it, so quietly fixing it here would make this model
    disagree with VIC everywhere a lake floods.
    """
    cases, _q = hypsometry
    _c, p = sorted(cases.items())[0]
    z, basin, maxd, maxv = _basin(p)
    over = _one(float(maxd) * 1.05)
    v = lake.get_volume(z, basin, over, maxv)
    # Twice the capacity, not a little over it.
    assert float(v) > 1.9 * float(maxv)
    back = lake.get_depth(z, basin, v, maxv, maxd)
    assert float(back) > 1.4 * float(over)
    # Inside the basin the same pair is exact.
    inside = _one(float(maxd) * 0.95)
    assert abs(float(lake.get_depth(
        z, basin, lake.get_volume(z, basin, inside, maxv), maxv, maxd))
        - float(inside)) < 1e-9


def test_area_and_volume_rise_with_depth(hypsometry):
    """A basin that narrows downward cannot hold less water when it is
    fuller, nor present less surface."""
    cases, _q = hypsometry
    for _c, p in sorted(cases.items()):
        z, basin, maxd, maxv = _basin(p)
        d = torch.linspace(1e-6, float(maxd), 40, dtype=DT,
                           device=DEV).unsqueeze(-1)
        a = lake.get_sarea(z, basin, d)
        v = lake.get_volume(z, basin, d, maxv)
        assert bool((a[1:] >= a[:-1] - 1e-9).all())
        assert bool((v[1:] >= v[:-1] - 1e-9).all())


def test_the_basin_relation_carries_a_gradient(hypsometry):
    """Lake depth is a state variable a calibration loop passes through."""
    cases, _q = hypsometry
    _c, p = sorted(cases.items())[0]
    z, basin, maxd, maxv = _basin(p)
    d = torch.linspace(0.2, float(maxd) * 0.9, 10, dtype=DT, device=DEV
                       ).unsqueeze(-1).requires_grad_(True)
    (lake.get_sarea(z, basin, d).sum()
     + lake.get_volume(z, basin, d, maxv).sum()).backward()
    assert torch.isfinite(d.grad).all()
    assert float(d.grad.abs().min()) > 0.0


# ---------------------------------------------------------------------------
# The node grid
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def derived():
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake6'], capture_output=True, text=True,
                         check=True).stdout
    cases, queries = {}, {}
    for line in out.splitlines():
        if line.startswith('#'):
            continue
        if line.startswith('d,'):
            p = line.replace('|', ',').split(',')[1:]
            queries.setdefault(int(p[0]), []).append(
                [float(x) for x in p[1:]])
        else:
            p = [float(x) for x in line.replace('|', ',').split(',')]
            cases[int(p[0])] = p
    return cases, queries


def test_node_grid_matches(derived):
    """compute_derived_lake_dimensions.c:34-104 over five basins and
    twenty-three depths each -- dry, one layer, two, and many, including the
    cap at MAX_LAKE_NODES.

    `activenod` is an integer the whole solve is sized by, so it is exact or
    wrong.
    """
    cases, queries = derived
    worst = {k: 0.0 for k in ('surfdz', 'dz', 'sarea', 'volume', 'surface')}
    nodes_wrong, n = 0, 0
    for c, p in sorted(cases.items()):
        nn = int(p[1])
        z = _nodes(p[4:4 + nn + 1])
        basin = _nodes(p[4 + nn + 1:4 + 2 * (nn + 1)])
        maxd, maxv = _one(p[2]), _one(p[3])
        for q in queries[c]:
            n += 1
            sd, dz, an, su, sa, vo = lake.derived_lake_dimensions(
                _one(q[0]), _one(q[1]), z, basin, maxd, maxv)
            worst['surfdz'] = max(worst['surfdz'], abs(float(sd) - q[2]))
            worst['dz'] = max(worst['dz'], abs(float(dz) - q[3]))
            nodes_wrong += int(an) != int(q[4])
            worst['sarea'] = max(worst['sarea'],
                                 abs(float(sa) - q[5]) / max(abs(q[5]), 1.0))
            worst['volume'] = max(worst['volume'],
                                  abs(float(vo) - q[6]) / max(abs(q[6]), 1.0))
            ref = _nodes(q[7:7 + lake.MAX_LAKE_NODES + 1])
            rel = (su - ref).abs() / ref.abs().clamp(min=1.0)
            worst['surface'] = max(worst['surface'], float(rel.max()))
    assert nodes_wrong == 0
    for k, v in worst.items():
        assert v < 1e-14, (k, v)
    assert n > 100


def test_the_layer_rule_is_vics(derived):
    """:45-72 -- at most LAKE_MAX_SURFACE for the surface layer, then equal
    layers below it; too shallow for two and it splits in half; shallower
    still and it is one layer; dry and it has none."""
    cases, _q = derived
    _c, p = sorted(cases.items())[0]
    nn = int(p[1])
    z = _nodes(p[4:4 + nn + 1])
    basin = _nodes(p[4 + nn + 1:4 + 2 * (nn + 1)])
    maxd, maxv = _one(p[2]), _one(p[3])
    run = lambda d: lake.derived_lake_dimensions(_one(d), _one(0.0), z, basin,
                                                 maxd, maxv)
    sd, dz, an, _s, _a, _v = run(0.0)
    assert int(an) == 0 and float(sd) == 0.0 and float(dz) == 0.0
    sd, dz, an, _s, _a, _v = run(lake.LAKE_MAX_SURFACE * 0.5)
    assert int(an) == 1 and abs(float(sd) - lake.LAKE_MAX_SURFACE * 0.5) < 1e-15
    sd, dz, an, _s, _a, _v = run(lake.LAKE_MAX_SURFACE * 1.5)
    assert int(an) == 2 and abs(float(sd) - float(dz)) < 1e-15
    sd, dz, an, _s, _a, _v = run(10.0)
    assert int(an) == int(10.0 / lake.LAKE_MAX_SURFACE)
    assert abs(float(sd) - lake.LAKE_MAX_SURFACE) < 1e-15
    # And the layers add up to the lake.
    assert abs(float(sd) + float(dz) * (int(an) - 1) - 10.0) < 1e-12


def test_the_node_count_is_capped(derived):
    """:53-56 -- a very deep lake is still solved on MAX_LAKE_NODES layers,
    which are then thicker than LAKE_MAX_SURFACE apart from the first."""
    cases, _q = derived
    _c, p = sorted(cases.items())[0]
    nn = int(p[1])
    z = _nodes(p[4:4 + nn + 1])
    basin = _nodes(p[4 + nn + 1:4 + 2 * (nn + 1)])
    deep = 500.0
    _sd, dz, an, _s, _a, _v = lake.derived_lake_dimensions(
        _one(deep), _one(0.0), z, basin, _one(p[2]), _one(p[3]))
    assert int(an) == lake.MAX_LAKE_NODES
    assert float(dz) > lake.LAKE_MAX_SURFACE


def test_inactive_nodes_are_zero_not_stale(derived):
    """`activenod` changes from step to step, and a batched model cannot
    change an array's length, so `surface` is built to MAX_LAKE_NODES + 1
    with the inactive entries zero. A caller that sums without masking gets
    zeros rather than the previous step's areas, which is the safe
    direction and is why this is pinned."""
    cases, _q = derived
    _c, p = sorted(cases.items())[0]
    nn = int(p[1])
    z = _nodes(p[4:4 + nn + 1])
    basin = _nodes(p[4 + nn + 1:4 + 2 * (nn + 1)])
    _sd, _dz, an, su, _a, _v = lake.derived_lake_dimensions(
        _one(1.0), _one(0.0), z, basin, _one(p[2]), _one(p[3]))
    k = int(an)
    assert float(su[k + 1:].abs().max()) == 0.0
    # Node `activenod` itself sits at zero depth -- the bottom of the lake,
    # where the basin closes to a point -- so its area is legitimately zero.
    # Everything above it holds water.
    assert float(su[:k].min()) > 0.0
    assert float(su[k]) == 0.0


# ---------------------------------------------------------------------------
# The lake's own water balance
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def outflow():
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake7'], capture_output=True, text=True,
                         check=True).stdout
    cases, queries = {}, {}
    for line in out.splitlines():
        if line.startswith('#'):
            continue
        if line.startswith('o,'):
            p = line.replace('|', ',').split(',')[1:]
            queries.setdefault(int(p[0]), []).append(
                [float(x) for x in p[1:]])
        else:
            p = [float(x) for x in line.replace('|', ',').split(',')]
            cases[int(p[0])] = p
    return cases, queries


def test_lake_outflow_matches(outflow):
    """lakes.eb.c:1873-1897 and :2010-2118 -- evaporation clipped to the
    liquid water there is, ARNO baseflow through the wetland's bottom layer
    over the lake's area, and runoff over a broad-crested weir.

    The oracle calls water_balance() itself with lakefrac at 1, so the
    wetland recharge between those two sections is a no-op and this
    isolates them.
    """
    cases, queries = outflow
    names = ('volume', 'ldepth', 'sarea', 'baseflow_out', 'runoff_out',
             'evapw')
    worst = {n: 0.0 for n in names}
    n_q = 0
    for c, p in sorted(cases.items()):
        nn = int(p[1])
        z = _nodes(p[4:4 + nn + 1])
        basin = _nodes(p[4 + nn + 1:4 + 2 * (nn + 1)])
        maxv = _one(p[3])
        for q in queries[c]:
            n_q += 1
            got = lake.lake_outflow(
                *[_one(q[i]) for i in range(6)], z, basin, maxv,
                *[_one(q[i]) for i in (6, 7, 8)], q[9],
                *[_one(q[i]) for i in (10, 11, 12, 13, 14, 15, 16, 17)])
            for i, name in enumerate(names):
                ref = q[18 + i]
                worst[name] = max(worst[name],
                                  abs(float(got[i]) - ref) / max(abs(ref), 1.0))
    assert n_q > 50
    for name, v in worst.items():
        assert v < 1e-14, (name, v)


def test_evaporation_cannot_exceed_the_liquid_water(outflow):
    """:1876-1891 -- the lake may not evaporate its ice, nor water it does
    not have. Ask for far more than there is and it takes exactly what is
    available and stops at the ice."""
    cases, _q = outflow
    _c, p = sorted(cases.items())[0]
    nn = int(p[1])
    z = _nodes(p[4:4 + nn + 1])
    basin = _nodes(p[4 + nn + 1:4 + 2 * (nn + 1)])
    maxv = _one(p[3])
    vol, ice = _one(float(maxv) * 0.4), _one(float(maxv) * 0.1)
    vol2, _d, _s, _b, _r, evapw = lake.lake_outflow(
        vol, ice, _one(1.0e12), _one(0.0), _one(0.0), _one(0.0), z, basin,
        maxv, _one(0.2), _one(0.0), _one(0.02), 3600.0, _one(2.5),
        _one(0.02), _one(0.9), _one(2.0), _one(400.0), _one(20.0),
        _one(0.0), _one(0.0))
    assert abs(float(evapw) - (float(vol) - float(ice))) < 1e-6
    assert float(vol2) <= float(ice) + 1e-6


def test_a_lake_below_its_outflow_depth_does_not_run_off(outflow):
    """:2065-2067 -- no water leaves over the weir until the lake is deeper
    than `mindepth`. A lake that drained anyway would never hold anything."""
    cases, _q = outflow
    _c, p = sorted(cases.items())[0]
    nn = int(p[1])
    z = _nodes(p[4:4 + nn + 1])
    basin = _nodes(p[4 + nn + 1:4 + 2 * (nn + 1)])
    maxv = _one(p[3])
    # Deep enough that mindepth is far above the water.
    _v, ldepth, _s, _b, runoff, _e = lake.lake_outflow(
        _one(float(maxv) * 0.02), _one(0.0), _one(0.0), _one(0.0), _one(0.0),
        _one(0.0), z, basin, maxv, _one(float(p[2]) * 0.9), _one(0.0),
        _one(0.02), 3600.0, _one(0.0), _one(0.02), _one(0.9), _one(2.0),
        _one(400.0), _one(20.0), _one(0.0), _one(0.0))
    assert float(ldepth) < float(p[2]) * 0.9
    assert float(runoff) == 0.0


def test_the_lake_water_balance_carries_a_gradient(outflow):
    """The lake's stage is a state variable, and the parameters that drain
    it -- wfrac, the ARNO coefficients -- are calibration targets."""
    cases, _q = outflow
    _c, p = sorted(cases.items())[0]
    nn = int(p[1])
    z = _nodes(p[4:4 + nn + 1])
    basin = _nodes(p[4 + nn + 1:4 + 2 * (nn + 1)])
    maxv = _one(p[3])
    # Small enough that the weir, not the water available, sets the
    # runoff -- otherwise the answer is the lake's volume and wfrac does
    # not enter it at all.
    wfrac = _one(0.001).requires_grad_(True)
    _v, _d, _s, _b, runoff, _e = lake.lake_outflow(
        _one(float(maxv) * 0.6), _one(0.0), _one(0.0), _one(0.0), _one(0.0),
        _one(0.0), z, basin, maxv, _one(0.2), _one(0.0), wfrac, 3600.0,
        _one(2.5), _one(0.02), _one(0.9), _one(2.0), _one(400.0), _one(20.0),
        _one(0.0), _one(0.0))
    runoff.sum().backward()
    assert torch.isfinite(wfrac.grad).all() and float(wfrac.grad) > 0.0


# ---------------------------------------------------------------------------
# Storage moving between the lake and the wetland beside it
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def advect():
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake8'], capture_output=True, text=True,
                         check=True).stdout
    section, rows = None, {}
    for line in out.splitlines():
        if line.startswith('# '):
            if '|' not in line:
                section = line[2:].strip()
            continue
        rows.setdefault(section, []).append(line.split('|'))
    return rows


def _f(part):
    return [x for x in part.replace(' ', '').split(',') if x != '']


def test_wetland_moisture_carries_across_an_area_change(advect):
    """advect_soil_veg_storage, lakes.eb.c:2313-2432. Eighteen cases where
    the lake grows, shrinks and stays put, with recharge large enough to
    overflow some columns so the downward-then-upward cascade and the
    baseflow spill are all reached."""
    wm = ww = wb = 0.0
    for a, b, c in advect['advect_soil_veg']:
        a, b, c = _f(a), _f(b), _f(c)
        lf, mn, nf, b0, bf = [_one(float(x)) for x in a[1:6]]
        n = 3
        moist = _nodes([float(x) for x in b[0:n]])
        dm = _nodes([float(x) for x in b[n:2 * n]])
        mm = _nodes([float(x) for x in b[2 * n:3 * n]])
        wdew = _one(float(b[3 * n]))
        gm, gw, gb = lake.advect_soil_veg_storage(lf, mn, nf, dm, moist, mm,
                                                  wdew, b0, bf)
        wm = max(wm, float((gm - _nodes([float(x) for x in c[0:n]])).abs().max()))
        ww = max(ww, abs(float(gw) - float(c[n])))
        wb = max(wb, abs(float(gb) - float(c[n + 1])))
    assert wm == 0.0 and ww == 0.0 and wb == 0.0


def test_recharge_that_will_not_fit_leaves_as_baseflow():
    """:2367-2371 -- the column is filled downward, then upward, and only
    then does what is left become baseflow. A version that spilled straight
    out would lose water the soil could have held."""
    n = 3
    mm = torch.tensor([[135.0], [225.0], [315.0]], dtype=DT, device=DEV)
    # Already full, and handed more.
    moist = mm.clone()
    dm = torch.tensor([[50.0], [0.0], [0.0]], dtype=DT, device=DEV)
    out, _w, bf = lake.advect_soil_veg_storage(
        _one(0.2), _one(0.2), _one(0.2), dm, moist, mm, _one(1.0),
        _one(1.0e6), _one(0.0))
    assert float((out - mm).abs().max()) < 1e-9      # still just full
    assert float(bf) > 0.0                           # and the rest left
    # Not full: the same water is absorbed and nothing leaves.
    moist2 = mm * 0.5
    out2, _w, bf2 = lake.advect_soil_veg_storage(
        _one(0.2), _one(0.2), _one(0.2), dm.clone(), moist2, mm, _one(1.0),
        _one(1.0e6), _one(0.0))
    assert float(bf2) == 0.0
    assert float(out2.sum()) > float(moist2.sum())


def test_flux_rescaling_matches(advect):
    """rescale_soil_veg_fluxes, rescale_snow_storage and
    rescale_snow_energy_fluxes (lakes.eb.c:2441-2640): a flux per unit area
    restated over a new area, and zeroed where the tile did not exist."""
    worst = 0.0
    for a, b, c in advect['rescale']:
        a, b, c = _f(a), _f(b), _f(c)
        of, nf = _one(float(a[1])), _one(float(a[2]))
        ev, bf, ro, swq, melt, grnd = [_one(float(x)) for x in b]
        got = (list(lake.rescale_fluxes(of, nf, ev, bf, ro))
               + list(lake.rescale_snow_storage(of, nf, swq))
               + list(lake.rescale_fluxes(of, nf, melt, grnd)))
        for i, g in enumerate(got):
            ref = float(c[i])
            worst = max(worst, abs(float(g) - ref) / max(abs(ref), 1.0))
    assert worst == 0.0


def test_a_tile_that_did_not_exist_has_no_fluxes_to_rescale():
    """:2465-2476 -- zero, not a division by an area of zero."""
    f = _one(5.0)
    out, = lake.rescale_fluxes(_one(0.0), _one(0.3), f)
    assert float(out) == 0.0
    # And a tile that has just vanished divides by DBL_EPSILON, not zero.
    out, = lake.rescale_fluxes(_one(0.3), _one(0.0), f)
    assert torch.isfinite(out).all()


def test_wetland_snow_follows_the_ground_it_was_lying_on(advect):
    """advect_snow_storage, lakes.eb.c:2484-2517. Unlike a flux, snow is
    scaled by the area it was *on*: the pre-flood area when the lake grew,
    the starting area when it receded. Snow in the strip the lake took goes
    with the lake."""
    worst = 0.0
    for a, b, c in advect['advect_snow']:
        a, b, c = _f(a), _f(b), _f(c)
        lf, mn, nf = [_one(float(x)) for x in a[1:4]]
        swq, dep = _one(float(b[0])), _one(float(b[1]))
        got = lake.advect_snow_storage(lf, mn, nf, swq, dep)
        for i, g in enumerate(got):
            ref = float(c[i])
            worst = max(worst, abs(float(g) - ref) / max(abs(ref), 1e-9))
    assert worst == 0.0


def test_soil_carbon_follows_the_area_that_grew(advect):
    """advect_carbon_storage, lakes.eb.c:2647-2683. The tile that gained
    area takes a mass-weighted share of the other's carbon; the one that
    lost area keeps its concentration. The two branches are not symmetric
    in the C and are not made so here."""
    worst = 0.0
    for a, b, c in advect['advect_carbon']:
        a, b, c = _f(a), _f(b), _f(c)
        lf, nf = _one(float(a[1])), _one(float(a[2]))
        lp, cp = _one(float(b[0])), _one(float(b[1]))
        gl, gc = lake.advect_carbon_storage(lf, nf, lp, cp)
        worst = max(worst, abs(float(gl) - float(c[0])) / max(abs(float(c[0])), 1.0))
        worst = max(worst, abs(float(gc) - float(c[1])) / max(abs(float(c[1])), 1.0))
    assert worst == 0.0


def test_carbon_is_diluted_not_created_when_the_lake_grows():
    """A lake that spreads over carbon-rich wetland ends up with a
    concentration between the two, never outside them."""
    lf, nf = _one(0.3), _one(0.5)
    lp, cp = _one(100.0), _one(400.0)
    gl, gc = lake.advect_carbon_storage(lf, nf, lp, cp)
    assert 100.0 < float(gl) < 400.0
    assert float(gc) == 400.0          # the wetland kept its concentration


# ---------------------------------------------------------------------------
# The two fixed-point energy balances
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def energy():
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake9'], capture_output=True, text=True,
                         check=True).stdout
    section, rows = None, {}
    for line in out.splitlines():
        if line.startswith('# '):
            if '|' not in line:
                section = line[2:].strip()
            continue
        rows.setdefault(section, []).append(line.split('|'))
    return rows


WEB_OUT = ('evapw', 'Qh', 'Qle', 'LWnet', 'deltaH', 'energy_ice_formation',
           'new_ice_area', 'new_ice_height', 'new_ice_water_eq',
           'energy_out_bottom')


def test_open_water_energy_balance_matches(energy):
    """water_energy_balance.c:33-205 -- a fixed point on the lake's surface
    temperature, with ice forming inside the loop whenever the new surface
    falls below freezing.

    Fourteen columns of three to seven nodes, spanning air temperatures from
    -12 to +12 C, so both the freezing and non-freezing branches run.
    """
    wt = 0.0
    worst = {k: 0.0 for k in WEB_OUT}
    for a, b, c in energy['web']:
        a, b, c = _f(a), _f(b), _f(c)
        n = int(a[1])
        dt, dz, sdz = float(a[2]), float(a[3]), float(a[4])
        args = [_one(float(x)) for x in a[5:17]]
        lat, tcut, tair, wind, press, vp, rhoa, lwin, swin, wh, fp, lv = args
        T = _nodes([float(x) for x in b[0:n]])
        rho = _nodes([float(x) for x in b[n:2 * n]])
        cp = _nodes([float(x) for x in b[2 * n:3 * n]])
        su = _nodes([float(x) for x in b[3 * n:4 * n + 1]])
        r = lake.water_energy_balance(su, None, dt, dz, sdz, lat, tcut, tair,
                                      wind, press, vp, rhoa, lwin, swin, wh,
                                      T, rho, fp, cp, lv, n)
        wt = max(wt, float((r['T']
                            - _nodes([float(x) for x in c[0:n]])).abs().max()))
        for i, k in enumerate(WEB_OUT):
            ref = float(c[n + i])
            worst[k] = max(worst[k],
                           abs(float(r[k]) - ref) / max(abs(ref), 1.0))
    assert wt < 1e-13
    for k, v in worst.items():
        assert v < 1e-13, (k, v)


def test_the_ice_outputs_are_untouched_when_no_ice_forms(energy):
    """water_energy_balance.c:148-160 -- iceform is called only when the new
    surface temperature went below freezing. On a step where it did not, the
    C leaves its four outputs at whatever the caller set, and solve_lake:102
    sets them to zero each step. Returning them regardless would put ice on
    a warm lake, which is how this was caught."""
    n = 3
    T = torch.full((n, 1), 12.0, dtype=DT, device=DEV)   # far above freezing
    rho = lake.calc_density(T)
    cp = lake.specheat(T)
    su = torch.full((n + 1, 1), 1.0e6, dtype=DT, device=DEV)
    r = lake.water_energy_balance(
        su, None, 3600.0, 0.5, 0.2, _one(45.0), _one(0.0), _one(15.0),
        _one(2.0), _one(95000.0), _one(800.0), _one(1.2), _one(320.0),
        _one(400.0), _one(10.0), T, rho, _one(0.0), cp, _one(1.0e6), n)
    for k in ('energy_ice_formation', 'new_ice_area', 'new_ice_height',
              'new_ice_water_eq'):
        assert float(r[k]) == 0.0, k


def test_water_under_ice_matches(energy):
    """water_under_ice.c:31-142 -- a fixed point on the heat the water gives
    up to the ice, with the column mixed inside the loop."""
    wt = 0.0
    worst = {'qw': 0.0, 'deltaH': 0.0, 'energy_out_bottom': 0.0}
    md_wrong = 0
    for a, b, c in energy['wui']:
        a, b, c = _f(a), _f(b), _f(c)
        ff, n = int(a[1]), int(a[2])
        dt, dz, sdz = float(a[3]), float(a[4]), float(a[5])
        lat, tcut, wind, sw = [_one(float(x)) for x in a[6:10]]
        md = torch.tensor([int(a[10])], device=DEV)
        hice, sdep = _one(float(a[11])), _one(float(a[12]))
        T = _nodes([float(x) for x in b[0:n]])
        rho = _nodes([float(x) for x in b[n:2 * n]])
        cp = _nodes([float(x) for x in b[2 * n:3 * n]])
        su = _nodes([float(x) for x in b[3 * n:4 * n + 1]])
        r = lake.water_under_ice(torch.tensor([ff], device=DEV), sw, wind, T,
                                 rho, lat, n, dz, sdz, tcut, su, cp, md,
                                 hice, sdep, dt)
        wt = max(wt, float((r['Ti']
                            - _nodes([float(x) for x in c[0:n]])).abs().max()))
        for i, k in enumerate(('qw', 'deltaH', 'energy_out_bottom')):
            ref = float(c[n + i])
            worst[k] = max(worst[k],
                           abs(float(r[k]) - ref) / max(abs(ref), 1.0))
        md_wrong += int(r['mixdepth']) != int(c[n + 3])
    assert wt < 1e-13
    for k, v in worst.items():
        assert v < 1e-13, (k, v)
    assert md_wrong == 0


def test_the_mixing_depth_under_ice_is_discarded_by_vic():
    """water_under_ice.c:45 takes `int mixdepth` by value and passes its
    address to tracer_mixer, so the depth the convective adjustment finds is
    written into a local copy and thrown away. It comes back here unchanged
    for the same reason: the caller must see what VIC's caller sees."""
    n = 4
    T = torch.tensor([[0.5], [2.0], [3.5], [4.0]], dtype=DT, device=DEV)
    rho = lake.calc_density(T)
    cp = lake.specheat(T)
    su = torch.full((n + 1, 1), 1.0e6, dtype=DT, device=DEV)
    md_in = torch.tensor([0], device=DEV)
    r = lake.water_under_ice(torch.tensor([1], device=DEV), _one(20.0),
                             _one(3.0), T, rho, _one(50.0), n, 0.5, 0.2,
                             _one(0.0), su, cp, md_in, _one(0.3), _one(0.05),
                             3600.0)
    assert int(r['mixdepth']) == int(md_in)
    # And the column really was mixed, so the discard is a loss of
    # information and not a no-op.
    assert float((r['Ti'] - T).abs().max()) > 0.0


# ---------------------------------------------------------------------------
# The surface energy balance over ice
# ---------------------------------------------------------------------------
def test_ice_energy_balance_matches():
    """IceEnergyBalance.c:34-214, forty cases through the C's own
    CalcIcePackEnergyBalance wrapper -- so the variable-argument list is
    assembled exactly as root_brent assembles it.

    The sample includes windless steps (where the surface decouples and the
    resistance goes to HUGE_RESIST) and steps at exactly 0 C with energy to
    spare (where the surplus melts rather than warming a surface that
    cannot get warmer), because those are the two branches.
    """
    if not REF.exists():
        pytest.skip('ref_snow missing')
    out = subprocess.run([str(REF), 'lake10'], capture_output=True,
                         text=True, check=True).stdout
    rows = [l.split('|') for l in out.splitlines() if not l.startswith('#')]
    keys = ('rest', 'ra_used', 'long_out', 'sensible', 'latent',
            'latent_sub', 'vapor_flux', 'surface_flux', 'advected', 'qf',
            'refreeze_energy')
    worst = {k: 0.0 for k in keys}
    calm = melting = 0
    for a, b in rows:
        a, b = [float(x) for x in _f(a)], [float(x) for x in _f(b)]
        st = dict(dt=_one(a[2]), ra=_one(a[3]), z=_one(a[4]), z0=_one(a[5]),
                  wind=_one(a[6]), short_rad=_one(a[7]),
                  long_rad_in=_one(a[8]), air_dens=_one(a[9]),
                  lv=_one(a[10]), t_air=_one(a[11]), press=_one(a[12]),
                  vpd=_one(a[13]), eact_air=_one(a[14]), rain=_one(a[15]),
                  surface_liquid_water=_one(a[16]),
                  blowing_flux=_one(a[17]), t_freeze=_one(a[18]),
                  avg_cond=_one(a[19]), sw_conducted=_one(a[20]))
        r, terms = lake.ice_energy_balance(_one(a[1]), st)
        got = dict(terms)
        got['rest'] = r
        for i, k in enumerate(keys):
            ref = b[i]
            worst[k] = max(worst[k],
                           abs(float(got[k]) - ref) / max(abs(ref), 1.0))
        calm += a[6] == 0.0
        melting += a[1] == 0.0
    for k, v in worst.items():
        assert v == 0.0, (k, v)
    assert calm > 0 and melting > 0


def test_the_ice_balance_has_no_cold_content_term():
    """IceEnergyBalance.c:185-186 -- the lake's ice has no pack to cool, so
    the deltaCC that appears in the snow pack's balance is absent here. It
    is absent rather than zero, and the conduction through the ice takes its
    place: with a thicker cover the same surface loses less to the water."""
    st = dict(dt=_one(3600.0), ra=_one(40.0), z=_one(10.0),
              z0=_one(lake.LAKE_ZSNOW), wind=_one(3.0), short_rad=_one(60.0),
              long_rad_in=_one(220.0), air_dens=_one(1.25),
              lv=_one(2.5e6), t_air=_one(-5.0), press=_one(95000.0),
              vpd=_one(100.0), eact_air=_one(200.0), rain=_one(0.0),
              surface_liquid_water=_one(0.0), blowing_flux=_one(0.0),
              t_freeze=_one(0.0), avg_cond=_one(0.05),
              sw_conducted=_one(2.0))
    _r, thin = lake.ice_energy_balance(_one(-3.0), st)
    st['avg_cond'] = _one(0.5)          # ten times the resistance
    _r, thick = lake.ice_energy_balance(_one(-3.0), st)
    assert abs(float(thick['qf'])) < abs(float(thin['qf']))
    assert 'deltaCC' not in thin


def test_the_ice_balance_carries_a_gradient():
    ts = torch.linspace(-10.0, -0.5, 12, dtype=DT, device=DEV,
                        requires_grad=True)
    one12 = lambda v: torch.full((12,), v, dtype=DT, device=DEV)
    st = dict(dt=one12(3600.0), ra=one12(40.0), z=one12(10.0),
              z0=one12(lake.LAKE_ZSNOW), wind=one12(3.0),
              short_rad=one12(60.0), long_rad_in=one12(220.0),
              air_dens=one12(1.25), lv=one12(2.5e6), t_air=one12(-5.0),
              press=one12(95000.0), vpd=one12(100.0), eact_air=one12(200.0),
              rain=one12(0.0), surface_liquid_water=one12(1e-3),
              blowing_flux=one12(0.0), t_freeze=one12(0.0),
              avg_cond=one12(0.05), sw_conducted=one12(2.0))
    r, _t = lake.ice_energy_balance(ts, st)
    r.sum().backward()
    assert torch.isfinite(ts.grad).all()
    assert float(ts.grad.abs().min()) > 0.0


# ---------------------------------------------------------------------------
# ice_melt.c -- the whole snow-and-ice pack step on a lake
# ---------------------------------------------------------------------------

_ICEMELT_OUT = (
    ('swq', 1), ('surf_water', 2), ('pack_water', 3), ('surf_temp', 4),
    ('pack_temp', 5), ('coldcontent', 6), ('vapor_flux', 7),
    ('surface_flux', 8), ('mass_error', 9), ('coverage', 10),
    ('ice_water_eq', 11), ('volume', 12), ('melt_mm', 13),
    ('advection', 14), ('deltaCC', 15), ('SnowFlux', 16), ('latent', 17),
    ('sensible', 18), ('Qnet', 19), ('refreeze_energy', 20), ('LWnet', 21))


@pytest.fixture(scope='module')
def icemelt():
    """`ref_snow lake11`: forty-four whole ice_melt() steps."""
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake11'], capture_output=True,
                         text=True, check=True).stdout
    rows = [l.split('|') for l in out.splitlines() if not l.startswith('#')]
    a = torch.tensor([[float(x) for x in r[0].strip().rstrip(',').split(',')]
                      for r in rows], dtype=DT, device=DEV)
    b = torch.tensor([[float(x) for x in r[1].strip().split(',')]
                      for r in rows], dtype=DT, device=DEV)
    c = lambda i: a[:, i].clone()
    state = dict(swq=c(1), surf_water=c(2), pack_water=c(3), surf_temp=c(4),
                 pack_temp=c(5), ice_water_eq=c(6), areai=c(7), hice=c(8),
                 volume=c(9))
    f = dict(dt=c(10), rainfall=c(11), snowfall=c(12), air_temp=c(13),
             wind=c(14), net_short=c(15), long_rad_in=c(16), air_dens=c(17),
             lv=c(18), press=c(19), vpd=c(20), eact_air=c(21), ra=c(22),
             z=c(23), z0=c(24), t_cutoff=c(25))
    new, diag = lake.ice_melt(state, f)
    got = dict(new)
    got.update(melt_mm=diag['melt'], advection=diag['advection'],
               deltaCC=diag['deltaCC'], SnowFlux=diag['snow_flux'],
               latent=diag['latent'], sensible=diag['sensible'],
               Qnet=diag['qnet'], refreeze_energy=diag['refreeze_energy'],
               LWnet=diag['lwnet'])
    return dict(ref=b, got=got, diag=diag, state=state, f=f)


def _icemelt_worst(bundle, mask):
    worst = {}
    for name, i in _ICEMELT_OUT:
        ref = bundle['ref'][:, i][mask]
        g = bundle['got'][name][mask]
        rel = (g - ref).abs() / ref.abs().clamp(min=1.0)
        worst[name] = float(rel.max())
    return worst


def test_ice_melt_takes_every_branch_the_c_has(icemelt):
    """The sample has to reach all three paths or the numbers below say
    nothing about the two it missed: the melting branch (the balance at 0 C
    returns exactly zero and the surplus melts), the sub-freezing branch
    (a root find, then all the surface liquid freezes), and the thin-pack
    branch (ice_melt.c:347 -- a surface layer at or under
    SNOW_MIN_SWQ_EB_THRES is not solved here at all and gets the 999 flag).
    """
    d = icemelt['diag']
    assert int(d['melting'].sum()) >= 8
    assert int((~d['melting'] & ~d['thin']).sum()) >= 8
    assert int(d['thin'].sum()) >= 2
    # Every case the C solved without falling back, so a disagreement below
    # is arithmetic and not VIC declining to answer.
    assert int(icemelt['ref'][:, 0].abs().sum()) == 0     # status
    assert int(icemelt['ref'][:, 22].sum()) == 0          # surf_temp_fbflag
    assert int(d['bracket_failed'].sum()) == 0
    # And the melt reaches the lake's own ice, not just the snow on it.
    assert float(d['ice_melt'].max()) > 0.0


def test_ice_melt_is_bit_exact_where_no_root_is_found(icemelt):
    """The melting and thin-pack branches do no iteration at all, so every
    one of the twenty-one outputs must match VIC's double exactly -- the
    two layer water equivalents and temperatures, both cold contents, the
    three sublimation components, the lake's ice volume and the lake's
    volume, the outflow, and all eight saved energy terms.

    This is the test that caught two association slips. `PackSwq +=
    SurfaceSwq - MAX` is `pack_swq + (surface_swq - MAX)`, not
    `(pack_swq + surface_swq) - MAX`, and `SurfaceSwq -= SurfaceSwq - MAX`
    is not an assignment of MAX. Either one written the natural way leaves
    the layer temperatures one ulp out, which is invisible in a plot and
    exactly what a bit-exact comparison is for.
    """
    d = icemelt['diag']
    mask = d['melting'] | d['thin']
    assert int(mask.sum()) >= 20
    worst = _icemelt_worst(icemelt, mask)
    assert max(worst.values()) == 0.0, worst


def test_ice_melt_matches_to_solver_tolerance_where_it_iterates(icemelt):
    """On the sub-freezing branch the surface temperature comes from a root
    find rather than from unrolling VIC's Brent solver, so it agrees to the
    solver's own tolerance -- param.ROOT_BRENT_T = 1e-7 in temperature --
    and everything downstream inherits that. The water stores that do not
    depend on the temperature stay exact.
    """
    d = icemelt['diag']
    mask = (~d['melting']) & (~d['thin'])
    worst = _icemelt_worst(icemelt, mask)
    assert worst['surf_temp'] < 1e-7
    for k in ('SnowFlux', 'latent', 'sensible', 'LWnet', 'coldcontent'):
        assert worst[k] < 1e-6, (k, worst[k])
    for k in ('surf_water', 'pack_water', 'ice_water_eq', 'volume',
              'melt_mm', 'advection', 'deltaCC', 'refreeze_energy',
              'coverage'):
        assert worst[k] == 0.0, (k, worst[k])
    assert worst['swq'] < 1e-11


def test_ice_melt_closes_its_own_mass_balance(icemelt):
    """ice_melt.c:583-586 computes its own residual and stores it in
    snow->mass_error. On every case that solves its own surface it is at
    the rounding limit, and so must ours be -- a port that lost a
    millimetre somewhere would show up in this term and nowhere else,
    because the outputs it is built from would each still look plausible.

    On the thin-pack cases it does NOT close, in VIC as much as here, and
    by an amount that is exactly the vapour flux: :415 skips the block that
    applies `snow->vapor_flux` to the ice, while :583-586 counts it in the
    residual regardless. That is not a leak in the model -- a pack that
    thin is handed to calc_surf_energy_bal, which applies the flux there --
    but it does mean ice_melt's own mass balance is not a closure test on
    those steps, and a port that quietly made it close would be the one
    that had diverged.
    """
    d = icemelt['diag']
    solved = ~d['thin']
    assert float(icemelt['got']['mass_error'][solved].abs().max()) < 1e-15
    assert float(icemelt['ref'][:, 9][solved].abs().max()) < 1e-15
    thin = d['thin']
    assert int(thin.sum()) >= 2
    # snow->vapor_flux is negated at :592, so the residual is its negative.
    left_over = (icemelt['got']['mass_error'][thin]
                 + icemelt['got']['vapor_flux'][thin])
    assert float(left_over.abs().max()) < 1e-17
    assert float(icemelt['got']['vapor_flux'][thin].abs().max()) > 1e-5


def test_the_root_bracket_widens_the_way_root_brent_widens_it():
    """root_brent.c:168-179. ice_melt.c:348 opens with
    [surf_temp - 5, surf_temp + 5], which frequently does not contain the
    root: a surface sitting at 0 C over a cold clear night solves to
    -11.6 C. VIC widens by ROOT_BRENT_TSTEP = 10 C on BOTH sides, up to
    ROOT_BRENT_MAXTRIES = 5 times.

    Handing a bracketed solver the interval as written instead returns a
    bound rather than a root, and that is not a small error: on the first
    case of the oracle it put the sensible heat at -344.8 W/m2 against
    VIC's -77.3 and left a residual of -320.8 W/m2 where VIC's is 2e-7.
    """
    n = 6
    x0 = torch.linspace(-2.0, 3.0, n, dtype=DT, device=DEV)
    # A residual whose root sits well outside [x0-5, x0+5] for some columns.
    roots = torch.tensor([-1.0, -7.0, -14.0, 20.0, -40.0, 3.0], dtype=DT,
                         device=DEV)
    fn = lambda t, st: st['r'] - t
    lo, hi, failed = lake._widen_bracket(fn, x0 - lake.SNOW_DT,
                                         x0 + lake.SNOW_DT, dict(r=roots))
    assert not bool(failed.any())
    assert bool(((lo < roots) & (hi > roots)).all())
    # Only the columns that needed it were widened, and each by a whole
    # number of ROOT_BRENT_TSTEP steps.
    tries = (hi - (x0 + lake.SNOW_DT)) / lake.ROOT_BRENT_TSTEP
    assert bool((tries == tries.round()).all())
    assert float(tries[0]) == 0.0 and float(tries[4]) == 4.0
    # And it gives up after MAXTRIES rather than widening forever.
    far = torch.tensor([200.0], dtype=DT, device=DEV)
    _lo, _hi, failed = lake._widen_bracket(
        fn, far - lake.SNOW_DT, far + lake.SNOW_DT,
        dict(r=torch.zeros(1, dtype=DT, device=DEV)))
    assert bool(failed.all())


def test_ice_melt_carries_a_gradient_to_the_forcing(icemelt):
    """The whole step, differentiated. The surface temperature is a root, so
    the gradient comes through the implicit function theorem; everything
    else is a transcription and differentiates directly. What this is really
    checking is that none of the masked branches leaked a NaN: `torch.where`
    evaluates both sides, and a division by a layer that is not there would
    poison the gradient of a column that never touched it."""
    st = {k: v.clone() for k, v in icemelt['state'].items()}
    f = {k: v.clone() for k, v in icemelt['f'].items()}
    f['air_temp'].requires_grad_(True)
    f['net_short'].requires_grad_(True)
    new, diag = lake.ice_melt(st, f)
    (new['swq'].sum() + new['ice_water_eq'].sum()
     + diag['melt'].sum()).backward()
    assert torch.isfinite(f['air_temp'].grad).all()
    assert torch.isfinite(f['net_short'].grad).all()
    assert float(f['air_temp'].grad.abs().sum()) > 0.0
    assert float(f['net_short'].grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# lakes.eb.c:33-532 -- solve_lake, the whole lake step
# ---------------------------------------------------------------------------

_SOLVE_OUT = (
    'volume', 'ice_water_eq', 'areai', 'new_ice_area', 'hice', 'SAlbedo',
    'tempavg', 'tempi', 'swe', 'sdepth', 'snowmlt', 'evapw', 'vapor_flux',
    'pack_water', 'surf_water', 'ice_throughfall', 'aero_resist', 'snow_swq',
    'snow_surf_water', 'snow_pack_water', 'snow_surf_temp', 'snow_pack_temp',
    'snow_coldcontent', 'snow_depth', 'snow_coverage', 'snow_vapor_flux',
    'snow_blowing_flux', 'snow_surface_flux', 'snow_melt', 'last_snow',
    'MELTING', 'AlbedoLake', 'AtmosLatent', 'AtmosSensible', 'NetLongAtmos',
    'NetShortAtmos', 'deltaH', 'grnd_flux', 'refreeze_energy', 'advection',
    'deltaCC', 'snow_flux', 'Tsurf', 'error')


def _solve_inputs(a, mask=None):
    """Rebuild solve_lake's arguments from one oracle row block."""
    if mask is not None:
        a = a[mask]
    c = lambda i: a[:, i].clone()
    lk = dict(activenod=c(1).long(), dz=c(2), surfdz=c(3), sarea=c(4),
              volume=c(5), ice_water_eq=c(6), areai=c(7), hice=c(8),
              SAlbedo=c(9), swq=c(10), surf_water=c(11), pack_water=c(12),
              surf_temp=c(13), pack_temp=c(14), coldcontent=c(15),
              last_snow=c(16).long(), MELTING=c(17).bool(), swe=c(18),
              surface=a[:, 35:56].T.clone(), temp=a[:, 56:76].T.clone())
    f = dict(snowfall=c(19), rainfall=c(20), tair=c(21), wind=c(22),
             vp=c(23), shortin=c(24), longin=c(25), vpd=c(26),
             pressure=c(27), air_density=c(28))
    soil = dict(lat=c(33), snow_rough=c(34))
    return lk, f, soil, c(29), c(30), c(31), c(32)


def _solve_flat(new, en):
    return dict(
        volume=new['volume'], ice_water_eq=new['ice_water_eq'],
        areai=new['areai'], new_ice_area=new['new_ice_area'],
        hice=new['hice'], SAlbedo=new['SAlbedo'], tempavg=new['tempavg'],
        tempi=new['tempi'], swe=new['swe'], sdepth=new['sdepth'],
        snowmlt=new['snowmlt'], evapw=new['evapw'],
        vapor_flux=new['vapor_flux'], pack_water=new['lake_pack_water'],
        surf_water=new['lake_surf_water'],
        ice_throughfall=new['ice_throughfall'],
        aero_resist=new['aero_resist'], snow_swq=new['swq'],
        snow_surf_water=new['surf_water'], snow_pack_water=new['pack_water'],
        snow_surf_temp=new['surf_temp'], snow_pack_temp=new['pack_temp'],
        snow_coldcontent=new['coldcontent'], snow_depth=new['snow_depth'],
        snow_coverage=new['coverage'],
        snow_vapor_flux=new['snow_vapor_flux'],
        snow_blowing_flux=new['blowing_flux'],
        snow_surface_flux=new['surface_flux'], snow_melt=new['snow_melt'],
        last_snow=new['last_snow'].to(DT), MELTING=new['MELTING'].to(DT),
        AlbedoLake=en['AlbedoLake'], AtmosLatent=en['AtmosLatent'],
        AtmosSensible=en['AtmosSensible'], NetLongAtmos=en['NetLongAtmos'],
        NetShortAtmos=en['NetShortAtmos'], deltaH=en['deltaH'],
        grnd_flux=en['grnd_flux'], refreeze_energy=en['refreeze_energy'],
        advection=en['advection'], deltaCC=en['deltaCC'],
        snow_flux=en['snow_flux'], Tsurf=en['Tsurf'], error=en['error'])


@pytest.fixture(scope='module')
def solvelake():
    """`ref_snow lake12`: forty whole solve_lake() steps.

    The batch is split by `activenod` because solve_lake needs one node
    count across a call -- see its docstring and the test below -- and the
    results are reassembled in the oracle's row order.
    """
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake12'], capture_output=True,
                         text=True, check=True).stdout
    rows = [l.split('|') for l in out.splitlines() if not l.startswith('#')]
    a = torch.tensor([[float(x) for x in r[0].strip().rstrip(',').split(',')]
                      for r in rows], dtype=DT, device=DEV)
    b = torch.tensor([[float(x) for x in r[1].strip().split(',')]
                      for r in rows], dtype=DT, device=DEV)
    n = a.shape[0]
    got = {k: torch.zeros(n, dtype=DT, device=DEV) for k in _SOLVE_OUT}
    temps = torch.zeros(20, n, dtype=DT, device=DEV)
    nod = a[:, 1].long()
    for nv in sorted(set(nod.tolist())):
        m = nod == nv
        new, en = lake.solve_lake(*_solve_inputs(a, m))
        flat = _solve_flat(new, en)
        for k in _SOLVE_OUT:
            got[k][m] = flat[k].to(DT)
        temps[:new['temp'].shape[0], m] = new['temp']
    return dict(a=a, ref=b, got=got, temps=temps, nod=nod)


def _solve_worst(bundle, keys):
    worst = {}
    for k in keys:
        i = 1 + _SOLVE_OUT.index(k)
        ref = bundle['ref'][:, i]
        rel = (bundle['got'][k] - ref).abs() / ref.abs().clamp(min=1.0)
        worst[k] = float(rel.max())
    return worst


def test_solve_lake_reaches_open_water_partial_ice_and_full_ice(solvelake):
    """The sample has to cover the branches or the agreement below is only
    about the ones it happened to take: an ice-free lake (section 6's else,
    where whatever ice there was is melted outright), partial cover (both
    the open-water and the ice balances run and their fluxes are area
    weighted), complete cover (section 4 is skipped entirely), and a cover
    below LAKE_FRACLIM, which VIC treats as none at all."""
    frac = solvelake['a'][:, 32]
    assert int((frac == 0.0).sum()) >= 4
    assert int(((frac > lake.LAKE_FRACLIM) & (frac < 1.0)).sum()) >= 8
    assert int((frac >= 1.0).sum()) >= 4
    assert int(((frac > 0.0) & (frac < lake.LAKE_FRACLIM)).sum()) >= 1
    # A one-layer lake and a twenty-layer one, because the node functions
    # special-case both ends.
    assert int(solvelake['nod'].min()) == 1
    assert int(solvelake['nod'].max()) >= 15
    # Every case ran to completion in VIC.
    assert float(solvelake['ref'][:, 0].abs().max()) == 0.0
    # And new ice actually formed in the open fraction somewhere.
    i = 1 + _SOLVE_OUT.index('new_ice_area')
    assert float(solvelake['ref'][:, i].max()) > 0.0


def test_solve_lake_is_exact_where_no_root_is_found(solvelake):
    """Everything that does not descend from ice_melt's surface temperature
    matches VIC's double exactly: the lake's volume and ice geometry, the
    precipitation split, the albedos, and the terms the ice branch leaves
    at zero."""
    exact = ('volume', 'areai', 'hice', 'SAlbedo', 'pack_water',
             'ice_throughfall', 'NetShortAtmos', 'AlbedoLake', 'advection',
             'deltaCC', 'snow_flux', 'snow_melt', 'last_snow', 'MELTING',
             'snow_pack_water')
    worst = _solve_worst(solvelake, exact)
    assert max(worst.values()) == 0.0, worst


def test_solve_lake_matches_the_water_column_to_rounding(solvelake):
    """The temperature profile, the column average and the water fluxes go
    through the open-water fixed point and the convective mixing, neither of
    which iterates on a root, so they come back at the rounding limit."""
    ref = solvelake['ref'][:, 45:65].T
    rel = (solvelake['temps'] - ref).abs() / ref.abs().clamp(min=1.0)
    assert float(rel.max()) < 1e-12
    worst = _solve_worst(solvelake, ('tempavg', 'evapw', 'surf_water',
                                     'snow_surf_water', 'snow_coldcontent'))
    assert max(worst.values()) < 1e-12, worst


def test_solve_lake_matches_the_ice_terms_to_solver_tolerance(solvelake):
    """Everything downstream of ice_melt's surface temperature inherits the
    root find's tolerance, which is param.ROOT_BRENT_T = 1e-7 in degrees.
    `error` is the whole-lake energy closure, a sum of eight terms each
    carrying that, so it is compared absolutely."""
    keys = ('ice_water_eq', 'new_ice_area', 'tempi', 'swe', 'sdepth',
            'snowmlt', 'vapor_flux', 'aero_resist', 'snow_swq',
            'snow_surf_temp', 'snow_pack_temp', 'snow_depth',
            'snow_coverage', 'snow_vapor_flux', 'snow_surface_flux',
            'AtmosLatent', 'AtmosSensible', 'NetLongAtmos', 'deltaH',
            'grnd_flux', 'refreeze_energy', 'Tsurf', 'error')
    worst = _solve_worst(solvelake, keys)
    assert max(worst.values()) < 1e-6, worst
    # VIC's own closure is near zero on most of these cases, and ours has to
    # be near VIC's rather than near zero -- the two are different claims.
    i = 1 + _SOLVE_OUT.index('error')
    ref = solvelake['ref'][:, i]
    assert float((solvelake['got']['error'] - ref).abs().max()) < 1e-6


def test_solve_lake_gives_a_column_the_same_answer_alone_as_in_a_batch(
        solvelake):
    """The property that a batched port has to have and that comparing a
    whole batch against a whole oracle will not check: each column's answer
    must not depend on what else is in the batch.

    This is not hypothetical. `tracer_mixer` advanced its stable depth
    inside an `if not unstable.any()` guard, so in a batch where ANY column
    had an inversion at node k, every stable column skipped the update and
    later mixed from the surface instead of from its own stable depth. Case
    20 of this oracle came out with a uniform 3.94 C column against VIC's
    3.76 C surface over a mixed bottom -- and it was right when run alone,
    which is exactly why a single-column test could not see it.
    """
    a = solvelake['a']
    nod = solvelake['nod']
    nv = int(torch.mode(nod).values)
    idx = (nod == nv).nonzero().flatten()
    assert idx.numel() >= 3
    batch, en = lake.solve_lake(*_solve_inputs(a, nod == nv))
    flat_batch = _solve_flat(batch, en)
    for pos, row in enumerate(idx.tolist()):
        mask = torch.zeros(a.shape[0], dtype=torch.bool, device=DEV)
        mask[row] = True
        one, en1 = lake.solve_lake(*_solve_inputs(a, mask))
        flat_one = _solve_flat(one, en1)
        for k in _SOLVE_OUT:
            assert float(flat_batch[k][pos] - flat_one[k][0]) == 0.0, (row, k)
        assert float((batch['temp'][:, pos] - one['temp'][:, 0]).abs().max()
                     ) == 0.0, row


def test_the_mixer_advances_every_stable_column_not_only_all_of_them():
    """The regression for the bug above, stated at the level of the function
    that had it.

    Two columns, both below 4 C so density rises with temperature. The
    first is stable down to node 2 and inverted at node 3; the second is
    inverted at node 2. VIC mixes the first from its own stable depth --
    nodes 3 and 4 only -- and leaves nodes 0 to 2 alone. With the bug the
    first column's stable depth stopped advancing at the node where the
    SECOND column was unstable, so it mixed 2 through 4 and node 2 moved.
    """
    t = torch.tensor([[1.0, 1.0], [1.5, 1.5], [2.0, 3.0], [3.5, 2.0],
                      [2.5, 2.5]], dtype=DT, device=DEV)
    surface = torch.full((6, 2), 1.0e5, dtype=DT, device=DEV)
    cp = lake.specheat(t)
    md = torch.zeros(2, dtype=torch.long, device=DEV)
    both, _r, _m = lake.tracer_mixer(t, md, surface, 5, 0.6, 0.6, cp)
    alone, _r, _m = lake.tracer_mixer(
        t[:, :1], md[:1], surface[:, :1], 5, 0.6, 0.6, cp[:, :1])
    assert float((both[:, 0] - alone[:, 0]).abs().max()) == 0.0
    # Nodes 0-2 of the first column are untouched and nodes 3-4 are mixed,
    # so the test fails both if the column mixes too much and if it mixes
    # nothing at all.
    assert float((both[:3, 0] - t[:3, 0]).abs().max()) == 0.0
    assert float((both[3:, 0] - t[3:, 0]).abs().min()) > 0.0
    assert float((both[:, 1] - t[:, 1]).abs().max()) > 0.0


def test_solve_lake_refuses_a_batch_of_mixed_node_counts(solvelake):
    """Stated as an error rather than solved silently. The node functions
    take the node count as a Python integer because the tridiagonal system's
    last row is indexed by it; two depths in one call would need two
    systems. A caller gets told, not given the wrong depth for half its
    columns."""
    a = solvelake['a']
    nod = solvelake['nod']
    mixed = torch.zeros(a.shape[0], dtype=torch.bool, device=DEV)
    mixed[int((nod == nod.min()).nonzero()[0])] = True
    mixed[int((nod == nod.max()).nonzero()[0])] = True
    with pytest.raises(ValueError, match='one active node count'):
        lake.solve_lake(*_solve_inputs(a, mixed))


def test_solve_lake_carries_a_gradient_to_the_forcing(solvelake):
    """The whole step differentiated, on the deepest sample column so the
    node loop, the open-water fixed point, the mixing and the ice pack are
    all in the graph."""
    a = solvelake['a']
    nod = solvelake['nod']
    m = nod == int(nod.max())
    lk, f, soil, dt, wind_h, doy, frac = _solve_inputs(a, m)
    f['tair'].requires_grad_(True)
    f['shortin'].requires_grad_(True)
    new, en = lake.solve_lake(lk, f, soil, dt, wind_h, doy, frac)
    (new['volume'].sum() + new['ice_water_eq'].sum()
     + new['temp'].sum() + en['error'].sum()).backward()
    assert torch.isfinite(f['tair'].grad).all()
    assert torch.isfinite(f['shortin'].grad).all()
    assert float(f['tair'].grad.abs().sum()) > 0.0
    assert float(f['shortin'].grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# lakes.eb.c:1808-2300 -- water_balance, the lake's water and its area
# ---------------------------------------------------------------------------

_WB_IN = (
    'case activenod dz surfdz sarea volume ice_water_eq new_ice_area evapw '
    'ice_throughfall snowmlt runoff_in baseflow_in channel_in lakefrac '
    'isave_n dt Dsmax Ds Ws cexp m0 m1 m2 mm0 mm1 mm2 resid_bot ice_bot '
    'Wdew snow_canopy wet_swq wet_depth wet_packw wet_surfw wet_baseflow '
    'wet_runoff wet_inflow wet_evap0 wet_evap1 wet_evap2 canopyevap '
    'throughfall lake_swq lake_depth lake_packw lake_surfw lake_canopy '
    'basin0 maxvolume maxdepth mindepth minvolume wfrac').split()
_WB_OUT = (
    'status volume ldepth sarea activenod dz surfdz evapw recharge '
    'baseflow_out runoff_out lk_swq lk_depth lk_packw lk_surfw lk_canopy '
    'lk_coverage lk_soil_runoff lk_soil_baseflow lk_soil_inflow '
    'lk_soil_evap0 wm0 wm1 wm2 Wdew wet_swq wet_depth wet_packw wet_surfw '
    'snow_canopy wet_baseflow wet_runoff wet_inflow wet_evap0 wet_evap1 '
    'wet_evap2 canopyevap throughfall').split()
_WB_NUMNOD = 10
#: the oracle's input row: scalars, surface[0..20], temp[0..19], then the
#: basin profile z[0..10] and basin[0..10]
_WB_Z0 = len(_WB_IN) + 21 + 20


def _wb_call(a, soil_t0=3.0):
    """Rebuild water_balance's arguments from one block of oracle rows."""
    i = {k: n for n, k in enumerate(_WB_IN)}
    s = len(_WB_IN)
    c = lambda k: a[:, i[k]].clone()
    n = a.shape[0]
    lk = dict(volume=c('volume'), ice_water_eq=c('ice_water_eq'),
              new_ice_area=c('new_ice_area'), evapw=c('evapw'),
              ice_throughfall=c('ice_throughfall'), snowmlt=c('snowmlt'),
              runoff_in=c('runoff_in'), baseflow_in=c('baseflow_in'),
              channel_in=c('channel_in'), activenod=c('activenod').long(),
              temp=a[:, s + 21:s + 41].T.clone(), sarea=c('sarea'),
              swq=c('lake_swq'), snow_depth=c('lake_depth'),
              pack_water=c('lake_packw'), surf_water=c('lake_surfw'),
              snow_canopy=c('lake_canopy'),
              coverage=torch.zeros(n, dtype=DT, device=DEV))
    wet = dict(moist=torch.stack([c('m0'), c('m1'), c('m2')], 0),
               wdew=c('Wdew'), snow_canopy=c('snow_canopy'), swq=c('wet_swq'),
               snow_depth=c('wet_depth'), snow_pack_water=c('wet_packw'),
               snow_surf_water=c('wet_surfw'), ice_bot=c('ice_bot'),
               baseflow_m3=torch.zeros(n, dtype=DT, device=DEV),
               evap_total=c('wet_evap0') + c('wet_evap1') + c('wet_evap2'),
               canopyevap=c('canopyevap'),
               canopy_vapor_flux=torch.zeros(n, dtype=DT, device=DEV),
               vapor_flux=torch.zeros(n, dtype=DT, device=DEV),
               soil_T0=torch.full((n,), soil_t0, dtype=DT, device=DEV),
               fluxes=dict(evap0=c('wet_evap0'), evap1=c('wet_evap1'),
                           evap2=c('wet_evap2'),
                           baseflow=c('wet_baseflow'), runoff=c('wet_runoff'),
                           inflow=c('wet_inflow'), canopyevap=c('canopyevap'),
                           throughfall=c('throughfall')))
    soil = dict(max_moist=torch.stack([c('mm0'), c('mm1'), c('mm2')], 0),
                resid_moist_bot=c('resid_bot'), Dsmax_step=c('Dsmax') / 24.0,
                Ds=c('Ds'), Ws=c('Ws'), c=c('cexp'))
    lc = dict(z=a[:, _WB_Z0:_WB_Z0 + _WB_NUMNOD + 1].T.clone(),
              basin=a[:, _WB_Z0 + _WB_NUMNOD + 1:
                      _WB_Z0 + 2 * _WB_NUMNOD + 2].T.clone(),
              basin0=c('basin0'), maxvolume=c('maxvolume'),
              maxdepth=c('maxdepth'), mindepth=c('mindepth'),
              minvolume=c('minvolume'), wfrac=c('wfrac'))
    isave = int(a[0, i['isave_n']])
    return lake.water_balance(lk, wet, soil, lc, c('dt'), c('lakefrac'),
                              isave)


def _wb_flat(nl, nw, d):
    return dict(
        volume=nl['volume'], ldepth=nl['ldepth'], sarea=nl['sarea'],
        activenod=nl['activenod'].to(DT), dz=nl['dz'], surfdz=nl['surfdz'],
        evapw=nl['evapw'], recharge=nl['recharge'],
        baseflow_out=nl['baseflow_out'], runoff_out=nl['runoff_out'],
        lk_swq=nl['swq'], lk_depth=nl['snow_depth'],
        lk_packw=nl['pack_water'], lk_surfw=nl['surf_water'],
        lk_canopy=nl['snow_canopy'], lk_coverage=nl['coverage'],
        lk_soil_runoff=d['lake_soil']['runoff'],
        lk_soil_baseflow=d['lake_soil']['baseflow'],
        lk_soil_inflow=d['lake_soil']['inflow'],
        lk_soil_evap0=d['lake_soil']['evap'],
        wm0=nw['moist'][0], wm1=nw['moist'][1], wm2=nw['moist'][2],
        Wdew=nw['wdew'], wet_swq=nw['swq'], wet_depth=nw['snow_depth'],
        wet_packw=nw['snow_pack_water'], wet_surfw=nw['snow_surf_water'],
        snow_canopy=nw['snow_canopy'], wet_baseflow=nw['baseflow'],
        wet_runoff=nw['runoff'], wet_inflow=nw['inflow'],
        wet_evap0=nw['evap0'], wet_evap1=nw['evap1'], wet_evap2=nw['evap2'],
        canopyevap=nw['canopyevap'], throughfall=nw['throughfall'])


@pytest.fixture(scope='module')
def waterbal():
    """`ref_snow lake13`: thirty-six whole water_balance() steps."""
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake13'], capture_output=True,
                         text=True, check=True).stdout
    rows = [l.split('|') for l in out.splitlines() if not l.startswith('#')]
    a = torch.tensor([[float(x) for x in r[0].strip().rstrip(',').split(',')]
                      for r in rows], dtype=DT, device=DEV)
    b = torch.tensor([[float(x) for x in r[1].strip().split(',')]
                      for r in rows], dtype=DT, device=DEV)
    n = a.shape[0]
    keys = _WB_OUT[1:]
    got = {k: torch.zeros(n, dtype=DT, device=DEV) for k in keys}
    temps = torch.zeros(20, n, dtype=DT, device=DEV)
    nod = a[:, _WB_IN.index('activenod')].long()
    for nv in sorted(set(nod.tolist())):
        m = nod == nv
        nl, nw, d = _wb_call(a[m])
        flat = _wb_flat(nl, nw, d)
        for k in keys:
            got[k][m] = flat[k].to(DT)
        temps[:nl['temp'].shape[0], m] = nl['temp']
    return dict(a=a, ref=b, got=got, temps=temps, nod=nod)


def _wb_worst(bundle, keys):
    worst = {}
    for k in keys:
        ref = bundle['ref'][:, _WB_OUT.index(k)]
        rel = (bundle['got'][k] - ref).abs() / ref.abs().clamp(min=1.0)
        worst[k] = float(rel.max())
    return worst


def test_water_balance_reaches_every_area_change_the_c_handles(waterbal):
    """A lake that grows into the wetland, one that recedes, one created
    where there was none at the start of the step, and a cell that was all
    lake and now has wetland again. Plus the node count changing, which is
    the path that carries the temperature profile across a regrid."""
    a = waterbal['a']
    lf = a[:, _WB_IN.index('lakefrac')]
    i = _WB_OUT.index
    newfrac = waterbal['ref'][:, i('sarea')] / a[:, _WB_IN.index('basin0')]
    assert int(((newfrac > lf) & (lf > 0.0)).sum()) >= 4   # grew
    assert int((newfrac < lf).sum()) >= 4                  # receded
    assert int(((lf == 0.0) & (newfrac > 0.0)).sum()) >= 4  # created
    assert int((lf >= 1.0).sum()) >= 4                     # was all lake
    # The recharge goes both ways: positive when the soil takes more than
    # the snow it swallowed was worth, negative otherwise.
    assert float(waterbal['ref'][:, i('recharge')].max()) > 0.0
    # And the node count changes with a real lake present, so the regrid is
    # exercised rather than skipped.
    old = a[:, _WB_IN.index('isave_n')].long()
    new = waterbal['ref'][:, i('activenod')].long()
    assert int(((old != new) & (lf > 0.0)).sum()) >= 10
    assert float(waterbal['ref'][:, 0].abs().max()) == 0.0


def test_water_balance_matches_vic_to_rounding(waterbal):
    """Nothing here iterates, so every one of the thirty-seven outputs --
    the lake's volume, depth, area and node grid; the recharge it traded
    with the wetland; both outflows; both tiles' snow, moisture and canopy
    storages; and every rescaled flux -- comes back at the rounding limit.
    """
    worst = _wb_worst(waterbal, _WB_OUT[1:])
    assert max(worst.values()) < 1e-14, worst
    t0 = len(_WB_OUT) + 21          # after the surface[0..20] block
    ref = waterbal['ref'][:, t0:t0 + 20].T
    rel = (waterbal['temps'] - ref).abs() / ref.abs().clamp(min=1.0)
    assert float(rel.max()) == 0.0


def test_a_brand_new_lake_reports_no_outflow_however_much_water_left_it(
        waterbal):
    """lakes.eb.c:2288-2291 assigns the lake's four soil fluxes, and :2262
    then calls initialize_lake, which sets them back to zero
    (initialize_lake.c:202-222). So a lake that did not exist at the start
    of the step reports no runoff, no baseflow and no evaporation over its
    own area whatever `runoff_out` says -- and `runoff_out` itself, which
    is `essential`, survives. This is VIC's, it is reproduced, and it is
    stated here because a caller that sums the per-area fluxes will lose
    that water and a caller that sums the volumes will not."""
    a = waterbal['a']
    lf = a[:, _WB_IN.index('lakefrac')]
    fresh = lf == 0.0
    i = _WB_OUT.index
    assert int(fresh.sum()) >= 4
    for k in ('lk_soil_runoff', 'lk_soil_baseflow', 'lk_soil_inflow',
              'lk_soil_evap0', 'lk_swq', 'lk_depth', 'lk_packw',
              'lk_surfw', 'lk_coverage'):
        assert float(waterbal['ref'][:, i(k)][fresh].abs().max()) == 0.0
        assert float(waterbal['got'][k][fresh].abs().max()) == 0.0
    # ...while the volume that actually left is still there.
    drained = waterbal['ref'][:, i('runoff_out')][fresh]
    assert float(drained.max()) > 0.0
    assert float((waterbal['got']['runoff_out'][fresh]
                  - drained).abs().max()) == 0.0


def test_the_node_regrid_reads_one_past_the_old_profile():
    """lakes.eb.c:2156-2163 walks a cursor down the old profile in steps of
    1/activenod and never resets it between nodes, so after
    activenod * isave_n steps it has advanced by exactly isave_n and the
    last sample reads temp[isave_n] -- one node past the old profile. That
    entry is whatever the array held from an earlier, deeper step.

    This is reproduced rather than clamped, and the test says so by showing
    that the stale entry changes the answer: two profiles identical over
    the old nodes but differing beyond them give different results.
    """
    a = torch.zeros(20, 1, dtype=DT, device=DEV)
    a[:4, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DT, device=DEV)
    b = a.clone()
    b[4, 0] = 99.0          # one past isave_n = 4, a stale deeper node
    ra = lake.redistribute_lake_nodes(a, 2, 4)
    rb = lake.redistribute_lake_nodes(b, 2, 4)
    assert float((ra - rb).abs().max()) > 0.0
    # The first new node averages the first isave_n samples of the walk,
    # which at step 1/2 land on nodes 0, 1, 1, 2 -- not on 0, 0, 1, 1.
    assert float(ra[0, 0]) == pytest.approx((1.0 + 2.0 + 2.0 + 3.0) / 4.0)
    assert float(ra[1, 0]) == pytest.approx((3.0 + 4.0 + 4.0 + 0.0) / 4.0)
    assert float(rb[1, 0]) == pytest.approx((3.0 + 4.0 + 4.0 + 99.0) / 4.0)
    # And no regrid at all when the node count did not change.
    same = lake.redistribute_lake_nodes(a, 4, 4)
    assert float((same - a).abs().max()) == 0.0


def test_water_balance_carries_a_gradient(waterbal):
    a = waterbal['a']
    nod = waterbal['nod']
    m = nod == int(torch.mode(nod).values)
    sub = a[m].clone()
    sub.requires_grad_(True)
    nl, nw, d = _wb_call(sub)
    (nl['volume'].sum() + nl['temp'].sum() + nw['moist'].sum()
     + nl['recharge'].sum()).backward()
    assert torch.isfinite(sub.grad).all()
    assert float(sub.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# The lake parameter file and the state a run starts from
# ---------------------------------------------------------------------------

FINDLEY_LAKE_PARAM = (pathlib.Path(_paths.SAMPLE_DATA) / 'classic'
                      / 'FindleyLake' / 'parameters'
                      / 'lake_param.FindleyLake')


def test_compute_lake_params_matches_vic_in_both_profile_modes():
    """compute_lake_params.c:34-90 through `ref_snow lake14`: twelve
    basins whose depth-area curve VIC generates and twelve whose curve the
    file gives. Every field of lake_con -- the two curves, the areas, the
    capacity and the volume at the minimum depth -- exactly."""
    if not REF.exists():
        pytest.skip('ref_snow missing')
    out = subprocess.run([str(REF), 'lake14'], capture_output=True,
                         text=True, check=True).stdout
    rows = [l.split('|') for l in out.splitlines() if not l.startswith('#')]
    worst = 0.0
    seen = [0, 0]
    for ra, rb in rows:
        a = [float(x) for x in ra.strip().rstrip(',').split(',')]
        b = [float(x) for x in rb.strip().split(',')]
        profile = bool(int(a[1]))
        numnod = int(a[2])
        seen[profile] += 1
        z = torch.tensor(a[6:27], dtype=DT, device=DEV)
        cl = torch.tensor(a[27:48], dtype=DT, device=DEV)
        con = lake.compute_lake_params(
            z[:numnod + 1], cl[:numnod + 1], numnod,
            torch.tensor(a[3], dtype=DT, device=DEV),
            torch.tensor(a[4], dtype=DT, device=DEV), a[5], profile)
        scalars = (con['maxdepth'], con['maxvolume'], con['minvolume'],
                   torch.tensor(con['bpercent'], dtype=DT, device=DEV),
                   con['basin0'])
        for i, v in enumerate(scalars):
            worst = max(worst, abs(float(v) - b[i]) / max(abs(b[i]), 1.0))
        for off, arr in ((5, con['z']), (26, con['Cl']), (47, con['basin'])):
            for j in range(numnod + 1):
                ref = b[off + j]
                worst = max(worst,
                            abs(float(arr[j]) - ref) / max(abs(ref), 1.0))
    assert seen[0] >= 8 and seen[1] >= 8
    assert worst == 0.0, worst


def test_lake_beta_is_one_thousandth_not_one_half():
    """param.LAKE_BETA is 0.001 (initialize_parameters.c:78). The name and
    the formula `(z/maxdepth)**BETA * radius` both suggest a square root,
    and a generated basin at 0.5 would be a shallow bowl. At 0.001 the
    walls are all but vertical: a tenth of the way up from the bottom the
    basin already covers 99.8% of its surface area. The two give depths
    that differ by metres, so this is pinned by value and by consequence,
    not by name."""
    assert lake.LAKE_BETA == 0.001
    z = torch.tensor([10.0] + [0.0] * 10, dtype=DT, device=DEV)
    cl = torch.zeros(11, dtype=DT, device=DEV)
    cl[0] = 0.1
    con = lake.compute_lake_params(
        z, cl, 10, torch.tensor(1.0e7, dtype=DT, device=DEV),
        torch.tensor(0.0, dtype=DT, device=DEV), 1.0, False)
    # node 9 sits at a tenth of the maximum depth
    assert float(con['z'][9]) == pytest.approx(1.0)
    assert float(con['basin'][9] / con['basin'][0]) > 0.99
    # ...and the basin is therefore nearly a cylinder rather than a bowl:
    # 94.8% of the cylinder's volume, the missing twentieth being the
    # deepest level, which closes to a point however steep the walls are.
    cylinder = float(con['basin'][0]) * 10.0
    assert 0.94 < float(con['maxvolume']) / cylinder < 0.96
    # A square-root profile would be barely half of it.


def test_the_findley_lake_parameter_file_reads_as_vic_reads_it():
    """The shipped sample, which is the only lake parameter file this
    project has: LAKE_PROFILE TRUE, ten nodes, a basin 24.4 m deep
    covering 8.85% of the cell."""
    if not FINDLEY_LAKE_PARAM.exists():
        pytest.skip('FindleyLake sample data missing')
    con, idx = lake.read_lakeparam(str(FINDLEY_LAKE_PARAM), 13099, 0.0885,
                                   1.0e7, True, dtype=DT, device=DEV)
    assert idx == 0
    assert con['numnod'] == 10
    assert float(con['maxdepth']) == pytest.approx(24.4)
    assert float(con['mindepth']) == pytest.approx(23.4)
    assert con['depth_in'] == pytest.approx(23.4)
    assert con['wfrac'] == pytest.approx(0.001671)
    assert con['rpercent'] == pytest.approx(1.0)
    assert float(con['basin0']) == pytest.approx(885000.0)
    # The curve closes to a point, deepest last.
    assert float(con['z'][10]) == 0.0 and float(con['Cl'][10]) == 0.0
    assert float(con['z'][1]) == pytest.approx(23.4)
    zs = con['z'][:11]
    assert bool((zs[:-1] > zs[1:]).all())
    # Reading and inverting the curve agree: the capacity stands at the
    # basin's own depth.
    d = lake.get_depth(con['z'].unsqueeze(-1), con['basin'].unsqueeze(-1),
                       con['maxvolume'].reshape(1),
                       con['maxvolume'].reshape(1),
                       con['maxdepth'].reshape(1))
    assert float(d) == pytest.approx(24.4, abs=1e-9)


def test_vic_discards_the_initial_lake_depth_the_file_gives():
    """generate_default_state.c's `generate_default_lake_state` takes
    `all_vars_struct *` and then copies the whole lake struct into a local
    -- `lake_var_struct lake; lake = all_vars->lake_var;` -- writes
    `lake.ldepth = lake_con.depth_in` and the initial temperatures into the
    copy, and never writes it back. The lake therefore starts at the depth
    `initialize_lake` left, which is zero.

    Measured against pristine VIC 5.0.1 run on the shipped FindleyLake
    cell, whose parameter file asks for 23.4 m in a basin 24.4 m deep:
    OUT_LAKE_DEPTH is 1.9938 m at the end of the first day and does not
    reach 23.4 m until 1970-05-20, the 140th daily record of the run. `initial_lake_state`
    reproduces that by default and takes `honour_depth_in=True` for a run
    that starts where the file says -- which is then not comparable with
    VIC's.
    """
    if not FINDLEY_LAKE_PARAM.exists():
        pytest.skip('FindleyLake sample data missing')
    con, _idx = lake.read_lakeparam(str(FINDLEY_LAKE_PARAM), 13099, 0.0885,
                                    1.0e7, True, dtype=DT, device=DEV)
    vic = lake.initial_lake_state(con, 8.0, dtype=DT, device=DEV)
    assert float(vic['ldepth']) == 0.0
    assert float(vic['volume']) == 0.0
    assert int(vic['activenod']) == 0
    assert float(vic['temp'].abs().max()) == 0.0
    asked = lake.initial_lake_state(con, 8.0, dtype=DT, device=DEV,
                                    honour_depth_in=True)
    assert float(asked['ldepth']) == pytest.approx(23.4)
    assert float(asked['volume']) == pytest.approx(float(con['minvolume']))
    assert int(asked['activenod']) > 0
    assert float(asked['temp'][0]) == pytest.approx(8.0)


def test_the_reader_refuses_a_lake_that_does_not_match_its_tile():
    """read_lakeparam.c:141-150 requires the area fraction at the top of
    the basin to equal the vegetation tile's own to within 1%, and then
    replaces it with the tile's outright -- so a file that is within
    tolerance but not equal does not shift the basin."""
    if not FINDLEY_LAKE_PARAM.exists():
        pytest.skip('FindleyLake sample data missing')
    with pytest.raises(ValueError, match='requires them equal'):
        lake.read_lakeparam(str(FINDLEY_LAKE_PARAM), 13099, 0.5, 1.0e7,
                            True, dtype=DT, device=DEV)
    # Within 1%, and the tile's fraction is what is used.
    con, _ = lake.read_lakeparam(str(FINDLEY_LAKE_PARAM), 13099, 0.0884,
                                 1.0e7, True, dtype=DT, device=DEV)
    assert float(con['Cl'][0]) == pytest.approx(0.0884)
    assert float(con['basin0']) == pytest.approx(884000.0)
    with pytest.raises(ValueError, match='not in'):
        lake.read_lakeparam(str(FINDLEY_LAKE_PARAM), 99999, 0.0885, 1.0e7,
                            True, dtype=DT, device=DEV)


# ---------------------------------------------------------------------------
# vic_run.c:155-482 -- a lake driven step after step
# ---------------------------------------------------------------------------

_RUN_OUT = ('status volume ldepth sarea activenod areai new_ice_area hice '
            'ice_water_eq swq surf_temp tempavg tempi evapw recharge '
            'baseflow_out runoff_out SAlbedo coverage error fraci lakefrac '
            'wm0 wm1 wm2 Wdew wet_swq').split()
#: the FindleyLake basin, the one lake profile VIC ships
_FINDLEY_Z = [24.4, 23.4, 20.8, 18.2, 15.6, 13.0, 10.4, 7.8, 5.2, 2.6, 0.0]
_FINDLEY_CL = [0.0885, 0.0884, 0.0698, 0.0535, 0.0393, 0.0273, 0.0175,
               0.0098, 0.0044, 0.0011, 0.0]


@pytest.fixture(scope='module')
def lakerun():
    """`ref_snow lake15`: one lake driven for 720 hourly steps.

    Every single-step test above compares one call. This compares a run:
    the state each function leaves for the next, the ice area carried from
    the end of one step to the start of the next, and the node grid
    re-derived as the lake fills and drains. A transcription that is right
    step by step and wrong about what is carried between them passes every
    other test in this file and fails this one.
    """
    if not REF.exists():
        pytest.skip('ref_snow missing; run '
                    'tests/oracle/build_ref_snow.sh')
    out = subprocess.run([str(REF), 'lake15'], capture_output=True,
                         text=True, check=True).stdout
    rows = [l.split('|') for l in out.splitlines() if not l.startswith('#')]
    a = [[float(x) for x in r[0].strip().split(',')] for r in rows]
    b = [[float(x) for x in r[1].strip().split(',')] for r in rows]

    one = lambda v: torch.tensor([float(v)], dtype=DT, device=DEV)
    cell_area = one(1.0e7)
    con = lake.compute_lake_params(
        torch.tensor(_FINDLEY_Z, dtype=DT, device=DEV).unsqueeze(-1),
        torch.tensor(_FINDLEY_CL, dtype=DT, device=DEV).unsqueeze(-1), 10,
        cell_area, one(2.0), 1.0, True)
    con['wfrac'] = one(0.001671)
    soil = dict(lat=one(47.32), snow_rough=one(0.001), cell_area=cell_area,
                max_moist=torch.tensor([[120.], [180.], [240.]], dtype=DT,
                                       device=DEV),
                resid_moist_bot=one(0.02 * 0.9 * 1000.0),
                Dsmax_step=one(12.0 / 24.0), Ds=one(0.02), Ws=one(0.8),
                c=one(2.0))
    surfdz, dz, activenod, surface, sarea, volume = \
        lake.derived_lake_dimensions(one(6.0), one(0.0), con['z'],
                                     con['basin'], con['maxdepth'],
                                     con['maxvolume'])
    temp = torch.zeros(20, 1, dtype=DT, device=DEV)
    temp[:int(activenod)] = 6.0
    lk = dict(ldepth=one(6.0), surfdz=surfdz, dz=dz, activenod=activenod,
              surface=surface, sarea=sarea, volume=volume, temp=temp,
              ice_water_eq=one(0), areai=one(0), new_ice_area=one(0),
              hice=one(0), tempi=one(0), tempavg=one(0), swe=one(0),
              sdepth=one(0), SAlbedo=one(0), coldcontent=one(0),
              surf_temp=one(0), pack_temp=one(0), surf_water=one(0),
              pack_water=one(0), swq=one(0), snow_depth=one(0),
              snow_canopy=one(0), coverage=one(0), aero_resist=one(0),
              last_snow=torch.zeros(1, dtype=torch.long, device=DEV),
              MELTING=torch.zeros(1, dtype=torch.bool, device=DEV),
              evapw=one(0), ice_throughfall=one(0), snowmlt=one(0),
              recharge=one(0), vapor_flux=one(0), snow_vapor_flux=one(0),
              blowing_flux=one(0), surface_flux=one(0), snow_melt=one(0),
              baseflow_out=one(0), runoff_out=one(0))
    wet = dict(moist=torch.tensor([[60.], [90.], [120.]], dtype=DT,
                                  device=DEV),
               wdew=one(0), snow_canopy=one(0), swq=one(0),
               snow_depth=one(0), snow_pack_water=one(0),
               snow_surf_water=one(0), ice_bot=one(0), baseflow_m3=one(0),
               evap_total=one(0), canopyevap=one(0),
               canopy_vapor_flux=one(0), vapor_flux=one(0), soil_T0=one(4.0),
               fluxes=dict(baseflow=one(0), runoff=one(0), inflow=one(0),
                           evap0=one(0), evap1=one(0), evap2=one(0),
                           canopyevap=one(0), throughfall=one(0)))

    rel = {k: 0.0 for k in _RUN_OUT[1:]}
    absd = {k: 0.0 for k in _RUN_OUT[1:]}
    for row_a, row_b in zip(a, b):
        f = dict(tair=one(row_a[1]), wind=one(row_a[2]), vp=one(row_a[3]),
                 vpd=one(row_a[4]), shortin=one(row_a[5]),
                 longin=one(row_a[6]), air_density=one(row_a[7]),
                 pressure=one(row_a[8]), prec=one(row_a[9]),
                 sum_runoff=one(row_a[10]), sum_baseflow=one(row_a[11]),
                 wetland_runoff=one(0), wetland_baseflow=one(0),
                 channel_in=one(0), t_max_snow=0.5, t_min_rain=-0.5)
        lk, wet, d = lake.lake_step(lk, wet, soil, con, f, one(3600.0),
                                    one(10.0), one(row_a[12]))
        got = dict(volume=lk['volume'], ldepth=lk['ldepth'],
                   sarea=lk['sarea'], activenod=lk['activenod'].to(DT),
                   areai=lk['areai'], new_ice_area=lk['new_ice_area'],
                   hice=lk['hice'], ice_water_eq=lk['ice_water_eq'],
                   swq=lk['swq'], surf_temp=lk['surf_temp'],
                   tempavg=lk['tempavg'], tempi=lk['tempi'],
                   evapw=lk['evapw'], recharge=lk['recharge'],
                   baseflow_out=lk['baseflow_out'],
                   runoff_out=lk['runoff_out'], SAlbedo=lk['SAlbedo'],
                   coverage=lk['coverage'], error=d['energy']['error'],
                   fraci=d['fraci'], lakefrac=d['lakefrac'],
                   wm0=wet['moist'][0], wm1=wet['moist'][1],
                   wm2=wet['moist'][2], Wdew=wet['wdew'], wet_swq=wet['swq'])
        for i, k in enumerate(_RUN_OUT[1:], start=1):
            ref = row_b[i]
            diff = abs(float(got[k]) - ref)
            rel[k] = max(rel[k], diff / max(abs(ref), 1.0))
            absd[k] = max(absd[k], diff)
    return dict(ref=b, rel=rel, absd=absd)


def test_the_run_exercises_ice_snow_and_a_changing_node_grid(lakerun):
    """Thirty days of a lake cooling well below freezing and thawing again.
    The comparison is only worth as much as what it reaches, so this says
    what it reached."""
    b = lakerun['ref']
    i = _RUN_OUT.index
    assert len(b) == 720
    assert sum(1 for r in b if r[i('hice')] > 0.0) > 300      # ice
    assert sum(1 for r in b if r[i('swq')] > 0.0) > 200       # snow on it
    assert len(set(int(r[i('activenod')]) for r in b)) >= 3   # regrids
    assert max(r[i('fraci')] for r in b) > 0.5                # mostly frozen
    assert min(r[i('fraci')] for r in b) == 0.0               # and open
    assert max(r[i('SAlbedo')] for r in b) > 0.8              # fresh snow
    assert all(r[0] == 0.0 for r in b)                        # VIC never errored


def test_a_lake_run_tracks_vic_for_seven_hundred_and_twenty_steps(lakerun):
    """The state each function leaves for the next, over a run rather than
    a step. The node count is exact at every one of the 720 steps -- an
    integer, so a single step's disagreement would show -- and the water
    stores drift only by the surface temperature root find's tolerance
    compounding: 4.0e-07 relative on the lake's volume after thirty days.

    The energy closure residual is the one term compared absolutely,
    because it is a difference of eight large fluxes and inherits their
    tolerance amplified: 9.5 W/m2 against VIC's own residual, which reaches
    4570 W/m2 on this run when the lake's area changes under it.
    """
    rel, absd = lakerun['rel'], lakerun['absd']
    assert rel['activenod'] == 0.0
    assert rel['SAlbedo'] == 0.0
    for k in ('volume', 'ldepth', 'sarea', 'hice', 'swq', 'baseflow_out',
              'runoff_out', 'lakefrac', 'wm0', 'wm1', 'wm2'):
        assert rel[k] < 1e-6, (k, rel[k])
    for k in ('areai', 'new_ice_area', 'ice_water_eq', 'recharge',
              'coverage', 'fraci', 'evapw', 'tempavg'):
        assert rel[k] < 1e-4, (k, rel[k])
    assert absd['surf_temp'] < 1e-4      # C
    assert absd['tempavg'] < 1e-4        # C
    assert absd['error'] < 20.0          # W/m2


def test_the_lake_agrees_between_devices(solvelake):
    """The claim the CPU-pinned comparisons above leave open.

    Every oracle comparison in this file runs on the CPU, because that is
    where "bit for bit against VIC" is true. This runs the same whole lake
    step on the live device as well and measures the gap, so the GPU is
    covered and the size of the difference is a number rather than an
    assumption. A ulp is 2.2e-16 relative; anything larger than a few of
    them is a real divergence and not `pow`.
    """
    if LIVE.type == 'cpu':
        pytest.skip('no second device to compare against')
    a = solvelake['a']
    nod = solvelake['nod']
    m = nod == int(torch.mode(nod).values)
    args_cpu = _solve_inputs(a, m)
    on_cpu, en_cpu = lake.solve_lake(*args_cpu)

    def to_live(x):
        if torch.is_tensor(x):
            return x.to(LIVE)
        if isinstance(x, dict):
            return {k: to_live(v) for k, v in x.items()}
        return x

    args_live = tuple(to_live(x) for x in args_cpu)
    on_live, en_live = lake.solve_lake(*args_live)
    worst = 0.0
    for k in ('volume', 'ice_water_eq', 'sarea', 'hice', 'swq', 'temp'):
        ref = on_cpu[k]
        got = on_live[k].to('cpu')
        worst = max(worst, float((got - ref).abs().max()
                                 / ref.abs().clamp(min=1.0).max()))
    for k in ('AtmosLatent', 'AtmosSensible', 'NetLongAtmos', 'deltaH'):
        ref, got = en_cpu[k], en_live[k].to('cpu')
        worst = max(worst, float((got - ref).abs().max()
                                 / ref.abs().clamp(min=1.0).max()))
    # A few ulp, not a tolerance chosen to pass: 2.2e-16 is one.
    assert worst < 1e-14, worst
