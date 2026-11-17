"""What a VIC warm start leaves behind, reproduced (vic_restore.c and the
paths a restart first reaches).

- A thin pack folded into the surface solve under a partial cover
  (SPATIAL_SNOW; func_surf_energy_bal.c 5.1.0:617-636, :770-781;
  calc_surf_energy_bal.c 5.1.0:211-218, :598-601). A VIC restart drops the
  spatial-snow stores, so the first melt after it opens a partial cover a
  continuous run would not have had.
- The lake's soil moisture: vic_restore.c 5.1.0:666-675 reads
  STATE_LAKE_SOIL_NODE_TEMP into lake_var.soil.layer[j].moist.
- Node arrays whose length equals the tile count (NODES 12 on a 12-tile
  cell): soiltemp._bc must not mistake [cell, node] for [cell, tile].
"""
import os

import numpy as np
import pytest
import torch

from torchvic.core import energy, soiltemp
from torchvic.core import snow as S

D = torch.float64


def _thin_residual(cov, ts=-3.0):
    n = len(cov)
    f = lambda v: torch.full((n,), float(v), dtype=D)        # noqa: E731
    c = torch.tensor(cov, dtype=D)
    pack = dict(thin=torch.ones(n, dtype=torch.bool), coverage=c,
                swq=f(4e-4), surf_water=f(1.4e-5), old_t_surf=f(0.0),
                density=f(132.0), net_short_snow=f(29.0), eact_air=f(380.0),
                press=f(69600.0), vpd=f(410.0), kappa_snow=f(16.0),
                t_snow_surf=f(0.0), melt_energy=f(0.0), advection=f(0.02))
    nsb = 141.7 * (1.0 - c) * (1.0 - 0.332)                 # NetShortBare
    evap_of = lambda nbr, ra, ns=None, **kw: torch.zeros_like(nbr)  # noqa
    resid, terms = energy._residual(
        f(ts), f(3.7), f(-11.8), f(-9.0), f(1.6), f(0.258), f(0.5),
        f(1.15), f(0.4), f(2.45e6), f(1.13e6), f(4.0), 10800.0, f(1.0),
        nsb, f(231.8), f(124.6), f(2.0), f(0.0), f(0.001), f(2.5),
        f(0.95), f(2.5e6), evap_of, pack=pack, bare_frac=1.0 - c)
    return resid, terms, c, nsb


def test_a_thin_pack_under_partial_cover_carries_the_bare_part():
    """INCLUDE_SNOW with cover c: NetLongBare = (1-c) L - (1-c) Lo,
    NetLongSnow = c L - c Lo; NetBareRad keeps NetShortBare; the pack's
    latent_sub, deltaCC and refreeze are weighted by c."""
    _, full, _, _ = _thin_residual([1.0, 1.0])
    _, t, c, nsb = _thin_residual([1.0, 0.47])
    lo = S.calc_outgoing_longwave(torch.full((2,), -3.0, dtype=D)
                                  + energy.CONST_TKFRZ, energy.EMISS_GRND)
    want = (((1.0 - c) * 231.8 - (1.0 - c) * lo)
            + (c * 231.8 - c * lo))
    assert torch.equal(t['net_long'], want)
    nlb = (1.0 - c) * 231.8 - (1.0 - c) * lo
    assert torch.equal(t['net_bare_rad'],
                       (((nsb + nlb) + t['grnd']) + t['delta_h'])
                       + t['fusion'])
    for k in ('latent_sub', 'delta_cc', 'refreeze_energy'):
        assert torch.equal(t[k], full[k] * c), k
    # vapour flux is the unweighted mass flux (func_surf_energy_bal.c:785)
    assert torch.equal(t['vapor_flux'], full['vapor_flux'])
    # at full cover nothing of the bare part is left
    assert full['net_bare_rad'][0] == (((0.0 + 0.0) + full['grnd'][0])
                                       + full['delta_h'][0]) + \
        full['fusion'][0]


def test_node_axis_equal_to_the_tile_axis_is_broadcast_per_cell():
    """NODES 12 on cells with 12 tiles crashed (index dimension mismatch):
    a [cell, node] grid passed through as if it were [cell, tile]."""
    nc, nt, nn = 2, 4, 4
    zsum = torch.tensor([[0.0, 0.1, 0.4, 2.0], [0.0, 0.2, 0.6, 3.0]],
                        dtype=D)
    depth = torch.tensor([[0.1, 0.5, 1.4], [0.2, 0.6, 2.2]], dtype=D)
    temp = torch.linspace(-5, 3, nc * nt * nn, dtype=D).reshape(nc, nt, nn)
    fr = torch.ones(nc, nt, 1, dtype=D)
    fs = torch.zeros(nc, nt, dtype=D)
    got = soiltemp.frost_temperature_and_depth(zsum, temp, depth, fr, fs,
                                               nn, 3)
    ref = soiltemp.frost_temperature_and_depth(
        zsum.unsqueeze(1).expand(nc, nt, nn).contiguous(), temp,
        depth.unsqueeze(1).expand(nc, nt, 3).contiguous(), fr, fs, nn, 3)
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


def test_the_lake_soil_moisture_is_what_vic_restore_leaves(tmp_path):
    """vic_restore.c 5.1.0:526-536 reads the never-written
    STATE_LAKE_SOIL_MOISTURE (the fill) and :666-675 overwrites it with
    STATE_LAKE_SOIL_NODE_TEMP, Nnode >= Nlayer values."""
    from test_lakecell import _global, _need_data
    _need_data()
    from torchvic import case
    from torchvic.io import state_file as SF
    c = case.load(_global(tmp_path), core='5.1.0')
    lakes = c.options['lakes']
    class _Cold(dict):
        def __missing__(self, k):                  # stores the run adds
            return np.zeros(self['sarea'].shape[-1])
    st0 = _Cold({k: v.double().cpu().numpy()
                 for k, v in lakes._state0.items()})
    f = SF._lake_vic_vars(st0)
    nb = st0['sarea'].shape[-1]
    f['STATE_LAKE_SOIL_MOISTURE'] = np.full((3, nb), 9.969209968386869e36)
    nt = np.arange(3 * nb, dtype=float).reshape(3, nb) - 1.5
    f['STATE_LAKE_SOIL_NODE_TEMP'] = nt
    st = lakes.state_from_vic(f)
    assert np.array_equal(st['soil_moist'].cpu().numpy(), nt)
    del c
