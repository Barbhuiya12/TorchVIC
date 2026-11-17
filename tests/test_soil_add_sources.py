"""Two soil_add producers on one VIC (VicConnector soil_add_sources):
irrigation and floodplain infiltration add to the same cells.

  * VIC adds their sum once (one capacity check) and hands the unmet back
    split in proportion to what each producer added: the parts add up to
    VIC's unmet and no producer gets back more than it added;
  * a flux cannot feed two consumers (VIC's soil_add_unmet to irrigation
    and flood would hand the same water back twice): refused;
  * VIC (Stehekin) + irrigation + flood infiltration: every module and
    coupled ledger row closes and the combined balance closes at rounding.
"""
import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from torchvic.coupling import TorchVIC, CouplingError     # noqa: E402
from torchvic.coupling.vic import VicConnector            # noqa: E402

DAY = 86400.0


def _fake(n):
    v = VicConnector(soil_add_sources=['irrigation', 'flood'])
    v.driver = types.SimpleNamespace(n_cells=n, dtype=torch.float64,
                                     device=torch.device('cpu'))
    return v


def test_declarations():
    v = VicConnector(soil_add_sources=['irrigation', 'flood'])
    ins = {s.name for s in v.inputs}
    outs = {s.name for s in v.outputs}
    assert {'soil_add', 'soil_add_irrigation', 'soil_add_flood'} <= ins
    assert {'soil_add_unmet', 'soil_add_unmet_irrigation',
            'soil_add_unmet_flood'} <= outs
    assert 'soil_add_flood' in v.imported_fluxes
    assert 'soil_add_unmet_flood' in v.exported_fluxes
    # without sources nothing changes
    w = VicConnector()
    assert w.inputs == VicConnector.inputs
    assert w.outputs == VicConnector.outputs
    with pytest.raises(CouplingError):
        VicConnector(soil_add_sources=['a', 'a'])
    with pytest.raises(CouplingError):
        VicConnector(soil_add_sources=['a'], day_coupled=['x'])


def test_unmet_split_in_proportion():
    v = _fake(5)
    # irrigation [cell], flood [cell, layer]
    a = torch.tensor([3.0, 0.0, 1.0, 0.0, 2.0], dtype=torch.float64)
    b = torch.tensor([[1.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 0.0],
                      [0.1, 0.2]], dtype=torch.float64)
    sa, parts = v._gather_soil_add({'soil_add_irrigation': a,
                                    'soil_add_flood': b})
    assert torch.equal(sa, torch.stack([a, a * 0], -1) + b)
    unmet = torch.tensor([2.0, 0.0, 0.4, 0.0, 1.0 / 3.0],
                         dtype=torch.float64)
    sp = v._split_unmet(unmet, parts)
    ui, uf = sp['soil_add_unmet_irrigation'], sp['soil_add_unmet_flood']
    assert torch.allclose(ui + uf, unmet, rtol=0, atol=1e-15)
    assert torch.allclose(ui, torch.tensor([1.5, 0, 0.1, 0, 2 / 2.3 / 3],
                                       dtype=torch.float64),
                          rtol=1e-14, atol=0)
    assert bool((ui >= 0).all()) and bool((uf >= 0).all())
    assert bool((ui <= a).all()) and bool((uf <= b.sum(-1)).all())
    # a negative part (irrigation's rounding-size leftover deficit) is
    # added as is and gets no unmet back; the positive part takes it all
    neg = torch.tensor([-1e-18, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    sa, parts = v._gather_soil_add({'soil_add_irrigation': neg,
                                    'soil_add_flood': b})
    assert torch.equal(sa, torch.stack([neg, neg * 0], -1) + b)
    sp = v._split_unmet(unmet, parts)
    assert torch.equal(sp['soil_add_unmet_irrigation'],
                       torch.zeros(5, dtype=torch.float64))
    assert torch.equal(sp['soil_add_unmet_flood'], unmet)


def test_unmet_cannot_feed_two_consumers():
    """The reason for soil_add_sources: one unmet flux, two producers."""
    import _paths
    stehe = _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                    set_options={'SNOW_BAND': 'FALSE'})
    cfg = dict(
        run=dict(device='cpu', log=False),
        modules=dict(vic=dict(global_param=stehe, exchange='day',
                              soil_add_mode='cap'),
                     irrigation=dict(connector='irrigation',
                                     vic=dict(global_param=stehe),
                                     irrigated_fraction=0.5),
                     irr2=dict(connector='irrigation',
                               vic=dict(global_param=stehe),
                               irrigated_fraction=0.5)),
        connections=[
            {'from': 'vic.soil_add_unmet', 'to': 'irrigation.soil_unmet'},
            {'from': 'vic.soil_add_unmet', 'to': 'irr2.soil_unmet'}])
    if not os.path.exists(stehe):
        pytest.skip("VIC's Stehekin sample is not present")
    os.environ['test_data_dir'] = _paths.SAMPLES
    with pytest.raises(CouplingError, match='twice'):
        TorchVIC.from_dict(cfg).initialize()


@pytest.mark.reference
def test_irrigation_and_floodplain_share_vic_soil(monkeypatch):
    import _paths
    from torchvic import case as _case
    from torchvic.coupling.vic import CaseDriver
    from torchvic.coupling.connectors.wateruse import WaterUseConnector
    stehe = _paths.relocated_global(_paths.STEHE_IMAGE_GLOBAL,
                                    set_options={'SNOW_BAND': 'FALSE'})
    if not (os.path.exists(stehe) and os.path.isdir(
            os.path.join(_paths.SAMPLES, 'image', 'Stehekin'))):
        pytest.skip("VIC's Stehekin sample is not present")
    monkeypatch.setenv('test_data_dir', _paths.SAMPLES)
    drv = CaseDriver(_case.load(stehe, device='cpu', log=lambda *a: None))
    nv = 16
    n = 16
    pairs = []
    for r in range(n):
        for c in range(n):
            pairs.append([(r // 4) * 4 + (c // 4), r * n + c, 1.0 / 16])
    yy, xx = np.mgrid[0:n, 0:n]
    dem = 10.0 + 1e-4 * np.hypot(xx - n / 2, yy - n / 2) * 1000.0
    h0 = np.where(np.hypot(xx - n / 2, yy - n / 2) < 6, 3.0, 0.0)
    ident = [[k, k, 1.0] for k in range(nv)]
    seen = []
    conns = dict(
        vic=VicConnector(driver=drv, exchange='day', soil_add_mode='cap',
                         soil_add_sources=['irrigation', 'flood']),
        irrigation=WaterUseConnector(
            driver=drv, wcr_fract=0.95, wfc_fract=1.0,
            irrigated_fraction=0.5, efficiency=0.7, conveyance_loss=0.1))
    cfg = dict(
        run=dict(device='cpu', log=False),
        modules=dict(
            vic=dict(connector='vic', global_param=stehe),
            irrigation=dict(connector='irrigation'),
            flood=dict(connector='flood', dem=dem, dx=1000.0,
                       interval_s=DAY, max_dt=300.0, backend='engine',
                       initial_depth=h0, infiltration_mm_day=1000.0,
                       open_water_evap_mm_day=3.0, vic_map=pairs)),
        connections=[
            {'from': 'vic.soil_moisture', 'to': 'irrigation.soil_moisture'},
            {'from': 'vic.soil_add_unmet_irrigation',
             'to': 'irrigation.soil_unmet'},
            {'from': 'irrigation.soil_add', 'to': 'vic.soil_add_irrigation',
             'lag': 1},
            {'from': 'vic.total_runoff', 'to': 'flood.runoff',
             'mapping': 'v2f'},
            {'from': 'flood.infiltration', 'to': 'vic.soil_add_flood',
             'lag': 1, 'mapping': 'fv2v'},
            {'from': 'vic.soil_add_unmet_flood',
             'to': 'flood.infiltration_unmet', 'mapping': 'v2fv'}],
        mappings=dict(
            v2f=dict(source='vic_cells', target='flood_cells', pairs=pairs),
            fv2v=dict(source='flood_vic', target='vic_cells', pairs=ident),
            v2fv=dict(source='vic_cells', target='flood_vic', pairs=ident)),
        ledger=dict(enabled=True, on_violation='reject'))
    vic = conns['vic']
    orig = vic._split_unmet

    def spy(unmet, parts):
        sp = orig(unmet, parts)
        if unmet is not None:
            tot = sum(sp.values())
            seen.append((float(unmet.sum()), float(tot.sum()),
                         float((tot - unmet).abs().max()),
                         {k: float(v.sum()) for k, v in sp.items()}))
            for un, ct in parts:
                assert bool((sp[un] <= ct * (1 + 1e-12) + 1e-300).all())
        return sp
    vic._split_unmet = spy
    with torch.inference_mode():
        m = TorchVIC.from_dict(cfg, connectors=conns).initialize()
        m.run(until=20 * DAY)
    assert all(r['ok'] for r in m.ledger.balances)
    assert all(r['ok'] for r in m.ledger.coupled)
    fl = m.connector('flood')
    assert fl.totals['infiltration'] > 0 and fl.totals['unmet'] > 0
    assert seen and max(s[2] for s in seen) <= 1e-12 * max(
        s[0] for s in seen)
    # both producers added water on the same days
    assert any(s[3].get('soil_add_unmet_flood', 0) > 0 for s in seen)
    from test_coupling_flood_2way import combined_residual
    R, P = combined_residual(m)
    assert abs(R) <= 1e-12 * (abs(P) + fl.totals['stored_start']), (R, P)


def test_mapping_rest_to_is_conservative():
    """rest_to: the source elements the pairs leave out go whole to one
    target element (the full system's site_pt / v2fv: only the flood's
    entry sites and box VIC cells carry water, the rest maps to element 0)."""
    from types import SimpleNamespace
    from torchvic.coupling.managers import Mapping
    grids = dict(big=SimpleNamespace(n=6), small=SimpleNamespace(n=2))
    m = Mapping.from_config('m', dict(source='big', target='small',
                                      pairs=[[2, 1, 0.5], [2, 0, 0.5],
                                             [4, 1, 1.0]], rest_to=0), grids)
    v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    out = m.volumes(v)
    assert float(out.sum()) == float(v.sum())
    assert out.tolist() == [1.0 + 2.0 + 1.5 + 4.0 + 6.0, 1.5 + 5.0]
    with pytest.raises(CouplingError, match='not conservative'):
        Mapping.from_config('m', dict(source='big', target='small',
                                      pairs=[[2, 1, 1.0]]), grids)


def test_segment_sum_fixed_order():
    """The flood's per-VIC-cell infiltration and per-hydraulic-cell unmet
    are summed in a fixed order (SegmentSum), not with atomics: equal to
    index_add_ up to rounding, and the same on every call."""
    from torchvic.coupling.connectors.flood import SegmentSum
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, 7, (5000,), generator=g)
    x = torch.rand(5000, generator=g, dtype=torch.float64)
    ref = torch.zeros(9, dtype=torch.float64).index_add_(0, idx, x)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    s = SegmentSum(idx, 9, dev)
    a = s(x.to(dev)).cpu()
    assert torch.allclose(a, ref, rtol=1e-13, atol=0)
    assert a[7:].eq(0).all()
    for _ in range(5):
        assert torch.equal(s(x.to(dev)).cpu(), a)
