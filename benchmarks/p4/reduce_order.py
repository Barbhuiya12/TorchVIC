"""Does an explicit addition tree reproduce eager CUDA's float64 sum over
dim 1 of a [N, R] or [N, R, L] tensor bit for bit? (core/xsum.py)

Inputs are spread over many binades with random exact zeros, so a
different association order shows up as a different last bit."""
import sys

import torch

sys.path.insert(0, __file__.rsplit('/benchmarks/', 1)[0])
from torchvic.core import xsum  # noqa: E402

dev = torch.device('cuda')
torch.manual_seed(0)
bad = 0
tot = 0


def data(shape, pz):
    m = torch.randn(shape, dtype=torch.float64, device=dev)
    e = torch.randint(-30, 30, shape, device=dev).double()
    x = m * torch.exp2(e)
    z = torch.rand(shape, device=dev) < pz
    x = torch.where(z, torch.zeros_like(x), x)
    neg0 = torch.rand(shape, device=dev) < 0.02
    return torch.where(neg0, torch.full_like(x, -0.0), x)


for R in list(range(1, 41)) + [64, 100, 128]:
    for L in (None, 1, 2, 3, 5, 10, 25):
        for N in (1, 7, 1000, 112680):
            if N * R * (L or 1) > 3e7:
                continue
            for pz in (0.0, 0.5, 0.85):
                shape = (N, R) if L is None else (N, R, L)
                x = data(shape, pz)
                ref = x.sum(1)
                got = xsum.sum1(x)
                tot += 1
                same = torch.equal(ref.view(-1).view(torch.int64),
                                   got.reshape(-1).view(torch.int64))
                if not same:
                    bad += 1
                    if bad < 20:
                        d = (ref - got).abs().max().item()
                        print('DIFF R=%d L=%s N=%d pz=%.2f max %.3e'
                              % (R, L, N, pz, d), flush=True)
print('%s: %d of %d cases differ' % (torch.cuda.get_device_name(), bad, tot))

# compiled (fusion setup): the same tree inside a region
from torchvic.core import fusion, graphsafe  # noqa: E402
fusion.setup()
f = torch.compile(xsum.sum1, dynamic=False)
bad = tot = 0
for R in (3, 5, 11, 17, 18, 19, 33):
    for L in (None, 3, 10):
        x = data((112680, R) if L is None else (112680, R, L), 0.6)
        tot += 1
        if not torch.equal(x.sum(1).view(-1).view(torch.int64),
                           f(x).reshape(-1).view(torch.int64)):
            bad += 1
            print('COMPILED DIFF R=%d L=%s' % (R, L))
print('compiled: %d of %d differ' % (bad, tot))
