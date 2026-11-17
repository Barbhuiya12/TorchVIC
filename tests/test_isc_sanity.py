"""Physical sanity checker (tools/isc_campaign/sanity.py): unit tests on
synthetic series, and a 31-day ISC smoke of the full VIC-WUR stack that fails
on any severity-high finding not listed in tools/isc_campaign/known_issues.json.

The smoke (slow, gpu) exists because synthetic tests missed the first real
ISC run's groundwater-river bug (MODFLOW baseflow ~100,000 mm/yr in the
Himalaya, oscillating and negative discharge).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import _paths

TOOLS = os.path.join(_paths.TOOLS, 'isc_campaign')
sys.path.insert(0, TOOLS)
import sanity as SA                                          # noqa: E402

ISC = _paths.data('vicwur/isc/smoke')
PY = os.environ.get('TORCHVIC_PYTHON', sys.executable)


def _base(T=60, N=50, seed=0):
    r = np.random.RandomState(seed)
    P = r.gamma(0.5, 6.0, (T, N)).astype(np.float32)
    return {'vic.rainf': P, 'vic.snowf': np.zeros_like(P),
            'vic.et': (0.3 * P).astype(np.float32),
            'vic.runoff': (0.2 * P).astype(np.float32),
            'vic.baseflow': np.full((T, N), 0.2, np.float32),
            'vic.swe': np.zeros((T, N), np.float32),
            'vic.air_temp': np.full((T, N), 25.0, np.float32),
            'vic.swdown': np.full((T, N), 200.0, np.float32),
            'vic.soil_moist': (300 + np.cumsum(0.3 * P - 0.3 * P.mean(), 0)
                               ).astype(np.float32),
            'vicwur_routing.discharge': np.tile(
                (100 + 20 * np.sin(np.arange(T) / 9.0))[:, None],
                (1, N)).astype(np.float32)}


def _keys(F, sev=None):
    return {(f['module'], f['variable'], f['check']) for f in F
            if sev is None or f['severity'] == sev}


def test_clean_series_has_no_findings():
    assert SA.check(_base()) == []


def test_nan_and_negative_discharge():
    S = _base()
    S['vicwur_routing.discharge'][5, 3] = np.nan
    S['vicwur_routing.discharge'][10:20, 7] = -50.0
    k = _keys(SA.check(S), 'high')
    assert ('vicwur_routing', 'discharge', 'nonfinite') in k
    assert ('vicwur_routing', 'discharge', 'negative') in k


def test_oscillating_discharge_is_flagged():
    S = _base()
    q = S['vicwur_routing.discharge']
    q[::2, :] *= 3.0                       # sawtooth in every cell
    assert ('vicwur_routing', 'discharge', 'oscillation') in _keys(
        SA.check(S), 'high')


def test_runaway_groundwater_baseflow(monkeypatch):
    """Every cell of a chain gives 5 m3/s to its river: far more than falls
    on the catchment. The per-cell excess alone is only noted (a river cell
    collects its neighbourhood); the catchment budget is the high finding."""
    S = _base()
    area = np.full(50, 6.0e6)
    S['groundwater.baseflow'] = np.full((60, 50), 5.0, np.float32)  # m3/s
    k = _keys(SA.check(S, area_m2=area))
    assert ('groundwater', 'baseflow', 'baseflow_high') in k
    monkeypatch.setattr(SA, 'BASIN_KM2', 100.0)     # 300 km2 in all
    down = np.minimum(np.arange(50) + 1, 49)
    F = SA.check(S, area_m2=area, down=down)
    assert ('groundwater', 'baseflow', 'baseflow_budget') in _keys(F, 'high')
    # a negligible exchange: no finding
    S['groundwater.baseflow'][:] = 1e-4
    assert ('groundwater', 'baseflow', 'baseflow_budget') not in _keys(
        SA.check(S, area_m2=area, down=down))


def test_recharge_bound_includes_capillary_rise():
    """Capillary rise puts aquifer water into the soil, so recharge may
    exceed P + irrigation by it (the bound of et_budget and runoff_ratio)."""
    S = _base()
    P = S['vic.rainf']
    area = np.full(50, 6.0e6)
    S['groundwater.capillary_rise'] = np.full((60, 50), 0.05, np.float32)
    cap = 0.05 * 86400.0 / 6.0e6 * 1000.0 * 60       # mm over the window
    S['vic.gwrecharge'] = (P + 0.5 * cap / 60.0).astype(np.float32)
    assert ('vic', 'gwrecharge', 'baseflow_high') not in _keys(
        SA.check(S, area_m2=area))
    S['vic.gwrecharge'] = (P + 2.0 * cap / 60.0).astype(np.float32)
    assert ('vic', 'gwrecharge', 'baseflow_high') in _keys(
        SA.check(S, area_m2=area))


def test_wet_cell_recharge_above_absolute_bound_is_not_flagged():
    """Gross bottom-layer drainage of a 6000 mm/yr cell exceeds BOUND_MM
    (3000) without exceeding the rain that fell on it."""
    S = _base()
    S['vic.rainf'] = np.full((365, 50), 6000.0 / 365, np.float32)
    S['vic.gwrecharge'] = np.full((365, 50), 4000.0 / 365, np.float32)
    assert ('vic', 'gwrecharge', 'baseflow_high') not in _keys(SA.check(S))
    S['vic.gwrecharge'] = np.full((365, 50), 7000.0 / 365, np.float32)
    assert ('vic', 'gwrecharge', 'baseflow_high') in _keys(SA.check(S))


def test_et_above_inflow_and_snow_deposition():
    S = _base()
    S['vic.et'][:, :5] += 6.0              # 5 cells evaporate far more than falls
    k = _keys(SA.check(S))
    assert ('vic', 'et', 'et_budget') in k
    S = _base()
    S['vic.swe'][:] = 20.0
    S['vic.et'][3, 0] = -0.3               # deposition on snow: legitimate
    assert ('vic', 'et', 'negative') not in _keys(SA.check(S))


def test_ledger_rows():
    S = _base()
    led = dict(balances=[dict(module='vicwur_dams', ok=False, residual=5.0,
                              scale=10.0, t0=0)], coupled=[])
    f = SA.check(S, led)
    assert ('vicwur_dams', 'water balance', 'ledger') in _keys(f, 'high')


@pytest.mark.slow
@pytest.mark.gpu
def test_isc_31_day_smoke_has_no_high_findings(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('needs a CUDA device')
    if not os.path.exists(ISC + '/global_isc_2005.txt'):
        pytest.skip('ISC VIC-WUR smoke inputs are not present')
    env = dict(os.environ)
    env['TORCHVIC_ISC_OUT'] = str(tmp_path)
    r = subprocess.run(
        [PY, os.path.join(TOOLS, 'run_step.py'), 'S7', '--days', '31',
         '--start', '2005-01-01', '--no-spun', '--tag', 'smoke31', '--out-root',
         str(tmp_path)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    d = str(tmp_path / 'smoke31')
    S, led = SA.load_run(d)
    area = SA.domain_area() if next(iter(S.values())).shape[1] == 112825 \
        else None
    F = SA.check(S, led, area)
    SA.write(F, d)
    known = json.load(open(os.path.join(TOOLS, 'known_issues.json')))['known']
    high = [f for f in F if f['severity'] == 'high'
            and '/'.join((f['module'], f['variable'], f['check']))
            not in known]
    assert not high, '\n'.join(
        '%s %s %s: %d cells, first %s: %s' % (
            f['module'], f['variable'], f['check'], f['cells'],
            f['first_day'], f['detail']) for f in high)
