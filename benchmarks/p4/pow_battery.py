"""Eager CUDA float64 pow against the fused region's pow (core.fusion
setup: libdevice.pow / rsqrt / 1/(x*x)), bit for bit, over bases spanning
the range with zeros, negatives, subnormals, inf and nan."""
import sys
import torch
from torchvic.core import fusion
fusion.setup()
torch.manual_seed(0)
dev = 'cuda'
n = 1 << 21
mag = torch.exp2(torch.randint(-1070, 1020, (n,), device=dev).double()) * (1 + torch.rand(n, device=dev, dtype=torch.float64))
sgn = torch.where(torch.rand(n, device=dev) < 0.2, -1.0, 1.0).double()
x = mag * sgn
special = torch.tensor([0.0, -0.0, 1.0, -1.0, float('inf'), -float('inf'), float('nan'), 5e-324, 2.2250738585072014e-308, 1e308], dtype=torch.float64, device=dev)
x[:special.numel()] = special
y = torch.rand(n, device=dev, dtype=torch.float64) * 40 - 20
y[::7] = torch.round(y[::7])          # integer exponents
y[::11] = 0.5
xs = torch.rand(n, device=dev, dtype=torch.float64) * 4 - 1   # moderate bases
EXPS = [0.25, 1.69, 4, 5, 10, -3, -0.5, -2, 2.5, 1 / 3, 1e-3, 100, 0, 7, -4, 0.75, 1.5, -1.5, 3.0, 2.0, -1.0, 0.5, 1.0, 17, 0.1]
cases = {}
for e in EXPS:
    cases['x**%r' % e] = (lambda a, e=e: a ** e, x)
    cases['xs**%r' % e] = (lambda a, e=e: a ** e, xs)
cases['pow(x,y)'] = (lambda a: torch.pow(a, y), x)
cases['pow(|x|,y)'] = (lambda a: torch.pow(a.abs(), y), x)
cases['pow(xs,y)'] = (lambda a: torch.pow(a, y), xs)
for c in (10.0, 2.0, 2.718281828459045, 0.5, -2.0, 1.5):
    cases['%r**y' % c] = (lambda a, c=c: c ** a, y)
cases['chain'] = (lambda a: (a.abs() + 1.0) ** 1.69 * 3.0 + a ** 4, xs)
bad = []
for k, (f, a) in cases.items():
    e = f(a)
    g = torch.compile(f, dynamic=False)(a)
    torch._dynamo.reset()
    same = bool(((e.view(torch.int64) == g.view(torch.int64)) | (torch.isnan(e) & torch.isnan(g))).all())
    if not same:
        nd = int(((e != g) & ~(torch.isnan(e) & torch.isnan(g))).sum())
        bad.append(k)
        print('%-16s DIFF n=%d' % (k, nd), flush=True)
print(torch.cuda.get_device_name(), 'cases', len(cases), 'bad:', bad)
