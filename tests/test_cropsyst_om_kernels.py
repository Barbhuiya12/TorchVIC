"""The residue pool kernels of the nitrogen day (torchvic.crop.cropsyst.om_kernels) against the
torch functions they replace (organic_matter.start_day, organic_matter.tillage), bit for bit on
every tensor (the free slots included), on random pool lists with holes, near-matching pools
(tolerance edges), empty and full lists and overflow.
"""
import pytest

torch = pytest.importorskip("torch")

from torchvic.crop.cropsyst import organic_matter as om  # noqa: E402

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")]


def _pools(T, N, surface, g, dev):
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    ri = lambda lo, hi: torch.randint(lo, hi, (T, N), generator=g)
    n = torch.randint(0, N + 1, (T,), generator=g)
    n = torch.where(r(T) < 0.3, torch.full_like(n, N), n)
    active = (torch.arange(N) < n[:, None]) & (r(T, N) > 0.25)
    pick = lambda vals: torch.tensor(vals, dtype=torch.float64)[torch.randint(0, len(vals), (T, N), generator=g)]
    mass = torch.exp(-20 * r(T, N)) * pick([1.0, 1.0, 0.0, 1e-8, -0.0])
    pos = ri(1, 3) if surface else torch.where(r(T, N) < 0.9, 0, ri(1, 3))
    layer = torch.zeros(T, N, dtype=torch.long) if surface else torch.where(r(T, N) < 0.3, 2, ri(1, 4))
    p = om.Pools(active, torch.where(r(T, N) < 0.85, 0, 1), pos, layer, ri(0, 4), ri(1, 4), mass,
                 pick([0.46, 0.4, 0.46000001]), pick([50.0, 30.5, 80.0, 50.0000001]), pick([0.231, 0.1155, 0.0231]),
                 pick([0.000963, 0.0, 1e-10]) if surface else pick([0.0, 0.0, 1e-7]), pick([2.0, 4.0, 8.0]), n)
    return om.Pools(*[x.to(dev) for x in p])


def _flat(r):
    return list(r.sur) + list(r.sub) + [r.flat_contact, r.overflow]


def _bits_equal(a, b):
    if a.dtype.is_floating_point:
        return bool((((a == b) & (torch.signbit(a) == torch.signbit(b))) | (torch.isnan(a) & torch.isnan(b))).all())
    return bool((a == b).all())


def _clone(r):
    return om.Residues(om.Pools(*[x.clone() for x in r.sur]), om.Pools(*[x.clone() for x in r.sub]),
                       r.flat_contact.clone(), r.overflow.clone())


@pytest.mark.parametrize("T,S,B,seed", [(3000, 96, 192, 1), (3000, 8, 16, 2), (2000, 5, 9, 4)])
def test_kernels_equal_the_torch_functions_bit_for_bit(T, S, B, seed):
    from torchvic.crop.cropsyst import om_kernels as omk
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    r = om.Residues(_pools(T, S, True, g, dev), _pools(T, B, False, g, dev),
                    torch.rand(T, generator=g, dtype=torch.float64).to(dev), (torch.rand(T, generator=g) < 0.1).to(dev))
    a = om.start_day(r)
    b = omk.start_day(r)
    assert all(_bits_equal(x, y) for x, y in zip(_flat(a), _flat(b)))
    th = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64, device=dev).expand(T, 3).contiguous()
    merged = appended = 0
    for rep in range(3):
        mask = (torch.rand(T, generator=g) < 0.6).to(dev)
        depth = torch.tensor([0.0, 0.05, 0.1, 0.2, 0.5, 1.5], dtype=torch.float64)[
            torch.randint(0, 6, (T,), generator=g)].to(dev)
        eff = torch.randint(0, 5, (T,), generator=g).to(dev)
        fr = torch.rand(T, generator=g, dtype=torch.float64).to(dev)
        br = torch.rand(T, generator=g, dtype=torch.float64).to(dev)
        a2 = om.tillage(a, mask, th, depth, eff, fr, br)
        b2 = omk.tillage_(_clone(a), mask, th, depth, eff, fr, br)
        assert all(_bits_equal(x, y) for x, y in zip(_flat(a2), _flat(b2))), rep
        merged += int(((a2.sub.mass != a.sub.mass) & a.sub.active).any(-1).sum())
        appended += int((a2.sub.n > a.sub.n).sum())
        a = a2
    assert merged > 100 and appended > 100
    if S < 96:
        assert bool(a.overflow.any())


@pytest.mark.parametrize("T,S,B,seed", [(3000, 96, 192, 5), (3000, 8, 16, 6)])
def test_harvest_residue_kernel_equals_add_surface_and_add_subsurface(T, S, B, seed):
    from torchvic.crop.cropsyst import om_kernels as omk
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    rr = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64).to(dev)
    r = om.start_day(om.Residues(_pools(T, S, True, g, dev), _pools(T, B, False, g, dev), rr(T),
                                 (torch.rand(T, generator=g) < 0.1).to(dev)))
    P = om.residue_params({}, device=dev)
    P = om.ResidueParams(*[x.expand((T,) + x.shape).contiguous() for x in P])
    hv = rr(T) < 0.7
    chaff = torch.where(rr(T) < 0.8, rr(T), -rr(T))
    stub = torch.where(rr(T) < 0.8, rr(T) * 1e-6, rr(T))
    roots = torch.where(rr(T, 3) < 0.7, rr(T, 3) * 1e-3, rr(T, 3) * 1e-6)
    conc = torch.where(rr(T) < 0.5, torch.zeros_like(chaff), rr(T) * 0.02)
    rconc = torch.where(rr(T) < 0.5, torch.zeros_like(chaff), rr(T) * 0.02)
    merges = 0
    for rep in range(3):          # the same harvest again: the root pools now merge
        mf, ms, mr = hv & (chaff > 1e-7), hv & (stub > 1e-7), hv & (roots.sum(-1) > 1e-7)
        a = om.add_surface(r, mf, om.FLAT, chaff, conc, P)
        a = om.add_surface(a, ms, om.STUBBLE, stub, conc, P)
        a = om.add_subsurface(a, mr, roots, rconc, P)
        b = omk.harvest_residue_(_clone(r), mf, chaff, ms, stub, mr, roots, conc, rconc, P)
        assert all(_bits_equal(x, y) for x, y in zip(_flat(a), _flat(b))), rep
        merges += int(((a.sub.mass != r.sub.mass) & r.sub.active).any(-1).sum())
        r = a
    assert merges > 1000


@pytest.mark.parametrize("hum,seed", [(float("nan"), 7), (0.3, 8), (0.0, 9)])
def test_process_kernel_equals_process_with_list_order_sums(hum, seed, monkeypatch):
    """om_kernels.process_ against organic_matter.process whose per-layer sums are taken slot by
    slot (the CPU's scatter_add order; torch's CUDA scatter_add order changes from run to run)."""
    from torchvic.crop.cropsyst import om_kernels as omk, soil_n as sn
    dev = torch.device("cuda")

    def seq_scatter(v, layer1, L):
        out = torch.zeros(v.shape[:-1] + (L,), dtype=v.dtype, device=v.device)
        idx = (layer1 - 1).clamp(0, L - 1)
        for j in range(v.shape[-1]):
            out = out.scatter_add(-1, idx[..., j:j + 1], v[..., j:j + 1])
        return out
    monkeypatch.setattr(om, "_scatter_layers", seq_scatter)
    T = 2000
    g = torch.Generator().manual_seed(seed)
    rr = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64).to(dev)
    r = om.start_day(om.Residues(_pools(T, 96, True, g, dev), _pools(T, 192, False, g, dev), rr(T),
                                 (torch.rand(T, generator=g) < 0.1).to(dev)))
    full = lambda v: torch.full((T, 3), v, dtype=torch.float64, device=dev)
    lay = sn.soil_layers(torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64, device=dev).expand(T, 3), full(1300.0),
                         full(100.0), full(60.0), full(30.0))
    env = sn.abiotic_update(0.1 + 0.3 * rr(T, 3), full(0.0), 30 * rr(T, 3) - 5, lay)
    s = sn.SoilN(rr(T, 3) * 1e-3, torch.where(rr(T, 3) < 0.3, full(0.0), rr(T, 3) * 1e-4))
    a_res, a_n, a_dec = om.process(r, env, s, humification=hum)
    b_res, b_n, b_dec = omk.process_(_clone(r), env, s, humification=hum)
    assert all(_bits_equal(x, y) for x, y in zip(_flat(a_res), _flat(b_res)))
    for x, y in ((a_n.no3, b_n.no3), (a_n.nh4, b_n.nh4), (a_dec.co2, b_dec.co2),
                 (a_dec.mineralized, b_dec.mineralized), (a_dec.demand, b_dec.demand)):
        assert _bits_equal(x, y)
    assert int((a_dec.co2 != 0).any(-1).sum()) > 1000
