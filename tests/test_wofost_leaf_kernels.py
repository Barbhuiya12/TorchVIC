# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""WOFOST-NPK leaf sums on a ring of leaf classes (crop/wofost_npk/
leaf_kernels.py) against eager's float64 sum and cumsum over the [rows, K]
row of classes, bit for bit: values over many binades, random windows,
rings of R <= K slots, scan chunks of 32 and 1024."""
import pytest

torch = pytest.importorskip('torch')

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(),
                                 reason='needs CUDA')]


def _case(rows, K, R, seed):
    from torchvic.crop.wofost_npk import leaf_kernels as LK
    g = torch.Generator(device='cpu').manual_seed(seed)
    dev = 'cuda'
    h = torch.randint(0, K, (rows,), generator=g)
    n = torch.randint(0, R + 1, (rows,), generator=g)
    t = torch.minimum(h + n, torch.full_like(h, K))
    e = torch.randint(-30, 8, (rows, K), generator=g).double()
    v = (torch.rand(rows, K, generator=g, dtype=torch.float64) * 2 ** e)
    v = torch.where(torch.rand(rows, K, generator=g) < 0.05,
                    torch.zeros_like(v), v).float()
    la = (torch.rand(rows, K, generator=g) * 0.01).float()
    k = torch.arange(K)
    live = (k >= h[:, None]) & (k < t[:, None])
    # the ring: class k in slot k mod R (the live ones)
    j = torch.arange(R)
    kk = h[:, None] + (j - h[:, None]) % R
    ok = kk < t[:, None]
    src = kk.clamp(max=K - 1)
    ring = torch.where(ok, v.gather(1, src), torch.zeros(()))
    ringa = torch.where(ok, la.gather(1, src), torch.zeros(()))
    v, la, live, h, t = (x.to(dev) for x in (v, la, live, h, t))
    ring, ringa = ring.to(dev).contiguous(), ringa.to(dev).contiguous()
    return LK, v, la, live, h, t, ring, ringa


@pytest.mark.parametrize('rows,K,R', [(4096, 336, 64), (300000, 336, 96),
                                      (1000, 132, 132), (20000, 200, 50),
                                      (16, 336, 336)])
def test_ring_sum_is_eager_sum(rows, K, R):
    LK, v, la, live, h, t, ring, ringa = _case(rows, K, R, rows + K)
    z = torch.zeros((), dtype=torch.float64, device='cuda')
    ref = torch.where(live, v.double(), z).sum(-1)
    got = LK.ring_sum(ring, h, t, K)
    assert torch.equal(ref, got)
    ref = torch.where(live, (v * la).double(), z).sum(-1)
    got = LK.ring_sum(ring, h, t, K, b=ringa)
    assert torch.equal(ref, got)


@pytest.mark.parametrize('rows,K,R', [(4096, 336, 64), (600000, 336, 128),
                                      (20000, 200, 50), (1000, 132, 132)])
def test_ring_scan_is_eager_cumsum(rows, K, R):
    LK, v, la, live, h, t, ring, ringa = _case(rows, K, R, 7 * rows + K)
    z = torch.zeros((), dtype=torch.float64, device='cuda')
    cs = torch.cumsum(torch.where(live, v.double(), z), -1)
    # death: around a live prefix sum, sometimes exactly on it
    pick = (h + (t - h) // 2).clamp(max=K - 1)
    d = cs.gather(1, pick[:, None])[:, 0]
    d = torch.where(torch.arange(rows, device='cuda') % 3 == 0, d,
                    d * 1.0000001)
    m = (live & (d[:, None] > cs)).sum(-1)
    h1 = h + m
    csm = cs.gather(1, (h1 - 1).clamp(0, K - 1)[:, None])[:, 0]
    gm, gc = LK.ring_scan(ring, h, t, d, LK.scan_chunk(K, rows))
    assert torch.equal(m, gm)
    sel = m > 0
    assert torch.equal(csm[sel], gc[sel])
