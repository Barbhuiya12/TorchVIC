"""FastStepRouter vs StepRouter on the ISC network: 731 daily steps with 2000
withdrawal/injection sites, every output and state tensor compared bit for
bit; then timing in a CUDA graph.  Real ISC gauge list (nearest cell)."""
import argparse, csv, hashlib, time
import netCDF4, numpy as np, torch
from torchvic.model import river_network as rn
from torchvic.model import routing_fast

ap = argparse.ArgumentParser()
ap.add_argument('--domain', required=True); ap.add_argument('--fdir', required=True)
ap.add_argument('--gauges', required=True); ap.add_argument('--sites', type=int, default=2000)
ap.add_argument('--steps', type=int, default=731)
a = ap.parse_args()
d = netCDF4.Dataset(a.domain)
lat, lon = d['lat'][:].data, d['lon'][:].data
mask = np.asarray(d['mask'][:]) > 0
f, v = a.fdir.split(':')
net, cells = rn.d8_downstream(np.asarray(netCDF4.Dataset(f)[v][:]), mask, lat, lon)
area = np.asarray(d['area'][:], dtype=np.float64)[cells[:, 0], cells[:, 1]]
cl, cc = lat[cells[:, 0]], lon[cells[:, 1]]
gl = []
for row in csv.DictReader(open(a.gauges)):
    la, lo = float(row['lat']), float(row['lng'])
    gl.append(int(np.argmin((cl - la) ** 2 + ((cc - lo) * np.cos(np.radians(la))) ** 2)))
gc = np.array(sorted(set(gl)))
sites = np.random.default_rng(0).choice(net.n, a.sites, replace=False)
mk = lambda: rn.build_step_router(net, gc, area, 86400.0, site_cells=sites, device='cuda')[0]
std = mk(); fast = routing_fast.FastStepRouter.from_router(mk())
g = torch.Generator(device='cuda').manual_seed(0)
n = net.n
hs, hf = hashlib.sha256(), hashlib.sha256()
bad = 0
for t in range(a.steps):
    ro = torch.rand(n, generator=g, device='cuda', dtype=torch.float64) * 5
    wd = torch.rand(a.sites, generator=g, device='cuda', dtype=torch.float64) * 4e5 * (t % 5 != 0)
    ij = torch.rand(a.sites, generator=g, device='cuda', dtype=torch.float64) * 1e5
    qa, ga = std.step(ro, withdraw=wd, inject=ij)
    qb, gb = fast.step(ro, withdraw=wd, inject=ij)
    hs.update(qa.cpu().numpy().tobytes() + ga.cpu().numpy().tobytes())
    hf.update(qb.cpu().numpy().tobytes() + gb.cpu().numpy().tobytes())
for k in std._STATE:
    if not torch.equal(getattr(std, k), getattr(fast, k)):
        bad += 1; print('STATE DIFFERS', k)
print('outputs sha256 std  ', hs.hexdigest()); print('outputs sha256 fast ', hf.hexdigest())
print('BYTE-IDENTICAL' if hs.hexdigest() == hf.hexdigest() and not bad else 'DIFFERENT')
print('ledger std %.2e fast %.2e' % (std.balance().abs().max().item(), fast.balance().abs().max().item()))
wd = torch.full((a.sites,), 2e5, device='cuda', dtype=torch.float64)
ro = torch.rand(n, generator=g, device='cuda', dtype=torch.float64) * 5
for name, r in (('StepRouter', std), ('FastStepRouter', fast)):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        r.step(ro, withdraw=wd)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        r.step(ro, withdraw=wd)
    for _ in range(5): gr.replay()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(300): gr.replay()
    torch.cuda.synchronize()
    print('%s CUDA graph, %d sites withdraw: %.3f ms/step' % (name, a.sites, (time.time() - t0) / 300 * 1e3))
