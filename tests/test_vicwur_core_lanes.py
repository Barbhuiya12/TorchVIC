# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR's vic_run hooks (torchvic.vicwur.core) on the covered tiles
(basin LANE_COMPACT) and inside the CUDA graphs: every hook on -- a layer
Ksat and Wcr changing by the step, the vegetation override under a mask,
b_co2 with a Catm series, the stored stress from a start value, the
diagnostics, the tile fluxes, VIC-WUR 3.0's throughfall, a step_fn, a state_fn
editing the carried states, a Cv changing twice, GWM -- the padded run,
the lanes run eagerly and the lanes run on CUDA graphs give the same
outputs bit for bit, and the hooks' per-tile rows on the covered tiles.
"""
import os

import pytest
import torch

import _paths  # noqa: F401
from torchvic.core import version as core_version
from torchvic.model import basin
from torchvic.vicwur import core as VC

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available(), reason='needs CUDA')]

D = torch.float64
VER = _paths.env_path(('TORCHVIC_VICWUR_VERIFICATION', 'VICWUR_VERIFICATION'), 'verification')
G0C = os.path.join(VER, 'vicwur_19', 'gruns', 'R1', 'G0c')
G11 = _paths.env_path(('TORCHVIC_VICWUR_4_GRUNS', 'VICWUR_4_GRUNS'), 'verification/vicwur_4/gruns/R1') + '/G11'


@pytest.fixture(autouse=True)
def _core():
    old = core_version.CORE
    yield
    core_version.set_core(old)
    torch.set_default_device('cpu')


def _case(rd, tmp_path, steps):
    from torchvic import case
    if not os.path.exists(os.path.join(rd, 'global_W3.txt')):
        pytest.skip('VIC-WUR run absent: ' + rd)
    gf = VC.image_global(os.path.join(rd, 'global_W3.txt'),
                         str(tmp_path / 'global.txt'))
    torch.set_default_device('cuda')
    try:
        return case.load(gf, device=torch.device('cuda'), core='vicwur3',
                         steps=steps)
    finally:
        torch.set_default_device('cpu')


def _hooks(c, gwm=None):
    m, sp = c.mosaic, c.soil
    nc, nt = m.cv.shape
    T = len(c.dates)
    dev = m.cv.device
    g = torch.Generator(device='cpu').manual_seed(11)
    rnd = lambda *s: torch.rand(*s, generator=g, dtype=D,
                                 device='cpu').to(dev)
    mask = rnd(nc, nt) < 0.5
    months = [d.month - 1 for d in c.dates]
    lai = m.lai.clone() * (0.5 + rnd(nc, nt, 1))
    ks1 = sp.Ksat.unsqueeze(1).expand(nc, nt, 3) * (0.2 + rnd(nc, nt, 3))
    wc0 = sp.w_crit_mm.unsqueeze(1).expand(nc, nt, 3).clone()
    root = rnd(nc, nt, 3) + 0.1
    root = root / root.sum(-1, keepdim=True)
    catm = (380.0 + 40 * rnd(T, nc)) * 1e-6
    covered = m.cv > 0
    cv_a = torch.where(covered, m.cv * (0.5 + rnd(nc, nt)), m.cv)
    cv_a = cv_a / cv_a.sum(1, keepdim=True)
    cv_b = cv_a.clone()
    # a covered tile of every other cell loses its cover
    k_ = covered.long().argmax(1)
    two = covered.sum(1) > 1
    cv_b[torch.arange(nc, device=dev)[two], k_[two]] = 0.0
    cv_b = cv_b / cv_b.sum(1, keepdim=True)
    h = VC.VicwurCore(
        ksat=ks1, wcr=lambda t: wc0 * (0.8 + 0.1 * (t % 3)),
        veg={'lai': lambda t: lai[:, :, 0] * (1 + 0.01 * months[t]),
             'fcanopy': m.fcanopy[:, :, 0].clone() * 0.9,
             'rmin': m.rmin * 0.7, 'root_frac': root},
        veg_mask=mask, b_co2=0.5 * rnd(nc, nt), catm=catm,
        water_stress=True, stress_init=0.5 * rnd(nc, nt, 3),
        diagnostics=True, tile_fluxes=True, throughfall='exact',
        cv=lambda t: None if t < 9 else (cv_a if t < 23 else cv_b),
        gwm=gwm)

    def step_fn(t, core):
        if t == 15:
            core.ksat = ks1 * 1.5

    def state_fn(t, states):
        if t in (4, 30):
            x = states.get('moist')
            states.set('moist', torch.where(covered.unsqueeze(-1),
                                            x * 0.97, x))
            w = states.get('wdew')
            states.set('wdew', w * 0.5)
    h.step_fn, h.state_fn = step_fn, state_fn
    return h, covered


def _run(c, monkeypatch, mode, gwm):
    monkeypatch.setattr(basin, 'VW_LANES', mode != 'padded')
    monkeypatch.setattr(basin, 'GRAPHS', mode.startswith('graphs'))
    h, cov = _hooks(c, gwm)
    basin.GRAPH_STATS.clear()
    # inference mode (as the coupling runs): tensors without a version
    # counter, the hooks' values then gathered every step
    with (torch.inference_mode() if mode.endswith('inf')
          else torch.no_grad()):
        o = c.run(p=c.params(), vicwur_core=h, contract=False)
    torch.cuda.synchronize()
    return o, cov, dict(basin.GRAPH_STATS)


@pytest.mark.parametrize('rd,gwm', [(G0C, None), (G11, 'fixed')])
def test_hooks_padded_lanes_graphs_bitwise(tmp_path, monkeypatch, rd, gwm):
    c = _case(rd, tmp_path, 40)
    runs = {m: _run(c, monkeypatch, m, gwm)
            for m in ('padded', 'lanes', 'graphs', 'graphs_inf')}
    assert runs['graphs'][2].get('graphs', 0) >= 1, runs['graphs'][2]
    assert runs['graphs'][2]['eager_steps'] < 10, runs['graphs'][2]
    o0, cov, _ = runs['padded']
    assert 'cv_end' in o0['vicwur']
    for m in ('lanes', 'graphs', 'graphs_inf'):
        o1 = runs[m][0]
        # a per-tile output (soil_moisture_tile) on the covered tiles: the
        # lanes hold no uncovered tile
        on = lambda x: x[cov] if tuple(x.shape[:2]) == tuple(cov.shape) \
            else x
        bad = [k for k, v in o0.items() if torch.is_tensor(v)
               and not torch.equal(on(v), on(o1[k]))]
        assert not bad, (m, bad)
        for k, v in o0['vicwur'].items():
            w = o1['vicwur'][k]
            if isinstance(v, dict):
                pairs = [(k + '.' + j, v[j], w[j]) for j in v]
            else:
                pairs = [(k, v, w)]
            for n, a, b in pairs:
                assert a.shape == b.shape, (m, n)
                if n == 'cv_end':
                    assert torch.equal(a, b), (m, n)
                    continue
                # per-tile rows: the covered tiles (lanes) bit for bit
                sel = (lambda x: x[:, cov]) if a.dim() >= 3 and \
                    a.shape[1:3] == cov.shape else (lambda x: x[cov])
                assert torch.equal(sel(a), sel(b)), (m, n)
    # the case keeps its file's Cv
    assert not torch.equal(c.mosaic.cv, o0['vicwur']['cv_end'])
