"""LAKES TRUE: the lake and wetland tile in the step loop
(torchvic.model.lakecell, vic_run.c 5.1.0:132-466, put_data.c 5.1.0:148-400).

The numbers are pristine VIC 5.1.0's (vic_image.exe) on UW-Hydro's Findley
Lake, image driver, FULL_ENERGY, LAKE_PROFILE TRUE, hourly from 1970-01-01:
the end of the first and of the second day. The whole year is the matrix row
findley_EB_510_1h_LAKES=TRUE_LAKE_PROFILE=TRUE (verification/matrix.py).
"""
import os

import pytest
import torch

from _paths import SAMPLE_DATA

FINDLEY = os.path.join(SAMPLE_DATA, 'image', 'FindleyLake')

#: VIC 5.1.0, steps 23 and 47 (1970-01-01 23:00 and 01-02 23:00).
VIC510 = {
    'OUT_LAKE_DEPTH': (23.4070425368087, 23.396217862887973),
    'OUT_LAKE_VOLUME': (895913.3205291509, 896812.6639677935),
    'OUT_LAKE_ICE_FRACT': (0.013272442527407046, 0.20155622076676866),
    'OUT_LAKE_ICE_HEIGHT': (0.1, 0.10785104068029266),
    'OUT_LAKE_SURF_TEMP': (-0.0065357568059566905, -0.006504043130524887),
    'OUT_LAKE_BF_OUT_V': (4.751538402696301, 4.7500653448964965),
    'OUT_LAKE_EVAP_V': (2.3097652248327143, 4.223310249210761),
    'OUT_LAKE_BF_IN_V': (48.998463489025326, 48.99953422310873),
    'OUT_LAKE_AREA_FRAC': (0.08840070425368088, 0.08837301141665738),
    'OUT_RUNOFF': (0.005346846733310209, 0.0),
    'OUT_BASEFLOW': (0.0036833631028653497, 0.0036822211975941835),
    'OUT_EVAP': (0.0019578786509595486, 0.0063763226176081385),
    'OUT_SWE': (0.0, 0.12890652293732416),
    'OUT_LATENT': (1.3677417812736277, 2.501554190594618),
    'OUT_SENSIBLE': (0.41602766741161346, 3.6894879060763595),
    'OUT_SURF_TEMP': (-5.530356460144688, -5.556543472657074),
    'OUT_IN_LONG': (207.39175517338927, 219.24606656090666),
    'OUT_SNOW_SURF_TEMP': (0.0, -5.664598030703949),
    'OUT_VEGT': (-5.112308949544141, -8.589221978454338),
    'OUT_BARESOILT': (-0.0005704570538078733, -0.07883349741303411),
}


def _global(tmp_path, profile='TRUE', lakes='TRUE'):
    p = os.path.join(FINDLEY, 'parameters')
    lines = [
        'MODEL_STEPS_PER_DAY 24', 'SNOW_STEPS_PER_DAY 24',
        'RUNOFF_STEPS_PER_DAY 24', 'STARTYEAR 1970', 'STARTMONTH 1',
        'STARTDAY 1', 'ENDYEAR 1970', 'ENDMONTH 1', 'ENDDAY 2',
        'CALENDAR PROLEPTIC_GREGORIAN', 'FULL_ENERGY TRUE',
        'FROZEN_SOIL FALSE', 'NODES 3',
        'DOMAIN %s' % os.path.join(p, 'domain.FindleyLake.nc'),
        'DOMAIN_TYPE LAT lat', 'DOMAIN_TYPE LON lon',
        'DOMAIN_TYPE MASK mask', 'DOMAIN_TYPE AREA area',
        'DOMAIN_TYPE FRAC frac', 'DOMAIN_TYPE YDIM lat',
        'DOMAIN_TYPE XDIM lon',
        'FORCING1 %s' % os.path.join(
            FINDLEY, 'forcings', 'livneh_NAmerExt_15Oct2014.FindleyLake.'),
        'FORCE_TYPE AIR_TEMP temp', 'FORCE_TYPE PREC prec',
        'FORCE_TYPE PRESSURE air_pressure', 'FORCE_TYPE SWDOWN shortwave',
        'FORCE_TYPE LWDOWN longwave', 'FORCE_TYPE VP vapor_pressure',
        'FORCE_TYPE WIND wind', 'WIND_H 10.0',
        'PARAMETERS %s' % os.path.join(p, 'params.FindleyLake.nc'),
        'ALB_SRC FROM_VEGPARAM', 'FCAN_SRC FROM_VEGPARAM',
        'LAI_SRC FROM_VEGPARAM', 'SNOW_BAND FALSE',
        'LAKES %s' % lakes, 'LAKE_PROFILE %s' % profile,
        'RESULT_DIR %s/' % tmp_path, 'LOG_DIR %s/' % tmp_path,
        'OUTFILE fluxes', 'AGGFREQ NSTEPS 1', 'OUTVAR OUT_PREC']
    g = tmp_path / 'global.txt'
    g.write_text('\n'.join(lines) + '\n')
    return str(g)


def _need_data():
    if not os.path.isdir(FINDLEY):
        pytest.skip('FindleyLake sample data missing (%s)' % FINDLEY)


@pytest.fixture(scope='module')
def run48(tmp_path_factory):
    _need_data()
    from torchvic import case
    from torchvic.io.history import outvar_series
    c = case.load(_global(tmp_path_factory.mktemp('lake')), core='5.1.0')
    assert c.n_steps == 48
    out = c.run()
    return c, {v: outvar_series(out, v)[0] for v in VIC510}


def test_findley_two_days_match_vic_510(run48):
    """Every reported lake, water and energy term at the end of each day
    agrees with pristine VIC 5.1.0 to 1e-6 of its size."""
    _, rows = run48
    bad = []
    for v, (a23, a47) in VIC510.items():
        x = rows[v]
        assert x is not None, v
        for t, ref in ((23, a23), (47, a47)):
            got = float(x[t].reshape(-1)[0])
            if abs(got - ref) > 1e-6 * (abs(ref) + 1e-3):
                bad.append('%s step %d: %r vs VIC %r' % (v, t, got, ref))
    assert not bad, '\n'.join(bad)


def test_the_510_lake_starts_full_and_at_0_C(run48):
    """generate_default_lake_state.c 5.1.0:31-35 keeps depth_in; the
    temperature loop runs over the activenod initialize_lake left (0)."""
    from torchvic.core import lake as L
    c, _ = run48
    con = c.options['lakes'].con
    one = {k: (v[..., 0] if torch.is_tensor(v) and v.dim() else v)
           for k, v in con.items()}
    one['depth_in'] = float(con['depth_in'][0])
    s510 = L.initial_lake_state(one, 5.0, vic_core='5.1.0')
    assert float(s510['ldepth']) == pytest.approx(23.4)
    assert int(s510['activenod']) == L.MAX_LAKE_NODES
    assert float(s510['temp'].abs().max()) == 0.0
    s501 = L.initial_lake_state(one, 5.0)
    assert float(s501['ldepth']) == 0.0


def test_the_lake_tile_and_its_basin(run48):
    c, _ = run48
    lk = c.options['lakes']
    assert lk.cols.tolist() == [0]
    cv = float(c.mosaic.cv[0, 0])
    area = float(c.batch.cell_area[0])
    assert float(lk.con['basin0'][0]) == pytest.approx(cv * area, rel=1e-12)
    assert float(lk.con['maxdepth'][0]) == pytest.approx(24.4)
    # the wetland's cover is restored after the run
    assert float(c.mosaic.cv[0, 0]) == 0.0885


def test_lake_profile_false_is_refused_as_vic_stops(tmp_path):
    """VIC 5.1.0 reads a basin_depth with a lake_node dimension with a 2-D
    start under LAKE_PROFILE FALSE, gets 0 and stops (vic_init.c:1489)."""
    _need_data()
    from torchvic import case
    with pytest.raises(ValueError, match='vic_init.c:1489'):
        case.load(_global(tmp_path, profile='FALSE'), core='5.1.0', steps=2)


def test_lakes_need_the_energy_balance(tmp_path):
    _need_data()
    from torchvic import case
    with pytest.raises(ValueError, match='FULL_ENERGY'):
        case.load(_global(tmp_path), core='5.1.0', steps=2, no_energy=True)
