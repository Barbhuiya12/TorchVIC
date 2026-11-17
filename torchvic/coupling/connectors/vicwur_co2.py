# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_co2: VIC-WUR CO2 forcing (C10) as a connector.

Kernels torchvic.vicwur.co2; C paths VIC-WUR 3.0 plugins/co2/src and
drivers/image/src:

  F  `pre_force`: co2_forcing (co2_forcing.c:30-41): the held record, ppm
  U  `pre_vic`: co2_update_step_vars (co2_update_step_vars.c:14-17):
     Catm[i < NF] = CO2 * 1e-6, Catm[NR] = average
  V  the canopy resistance factor 1 / (1 + b_co2 (CO2/CO2REF - 1)) in
     calc_rc (penman.c:83-87) with b_co2 of each tile's class (vic_init.c:
     185-199, BCO2_SRC FROM_VEGPARAM): the run's vicwur_core `b_co2` and
     `catm` (`core_hooks`; Catm is read before every step, so the value of
     the day's U phase is the one vic_run sees)
  OUT_CATM (Catm[NR] / 1e-6, put_data.c:157) per cell is the row of the
     day whose U phase formed it (C has no CO2 put_data; the plan has no O
     call for it)

The crop component reads `ppm()` (co2_force.CO2, crop_run.c:62-64).

Options
    co2      a Series spec of CO2 (ppm) per cell [N]: {'forcing':
             Co2Forcing / PluginForcing}, {'series': ...} or a constant
    b_co2    [nveg, N] per vegetation class (params.read_bco2), or None
             (no CO2 effect in the canopy: VIC's own calc_rc)
    co2ref   CANOPY_CO2REF (366 ppm)

State (checkpoint): the held CO2 and Catm. Ledger: no water; zero row.
"""
import torch

from ..connector import Balance, ComputeError
from ..spec import CouplingError, VarSpec
from ...vicwur import co2 as CO2
from .vicwur_common import GRID, Series, WurConnector, as_f64


class VicWurCo2Connector(WurConnector):
    name = 'vicwur_co2'
    version = '1'
    kind = 'co2'
    role = 'co2'
    flags = dict(force_co2=True)

    def __init__(self, co2=None, b_co2=None, co2ref=CO2.CANOPY_CO2REF,
                 n_cells=None, group=None, **config):
        super().__init__(group=group, **config)
        if co2 is None:
            raise CouplingError('%s: give co2 (ppm)' % self.name)
        self.f = Series(co2, 'co2')
        self.b_cfg = b_co2
        self.co2ref = float(co2ref)
        if not self.co2ref > 0:
            raise CouplingError('%s: co2ref must be > 0' % self.name)
        if b_co2 is not None and torch.as_tensor(b_co2).dim() != 2:
            raise CouplingError('%s: b_co2 must be [nveg, N]' % self.name)
        self.n_cfg = n_cells
        self._check_config()
        self.outputs = (VarSpec('catm', 'ppm', ('cell',), kind='state',
                                grid=GRID, description='OUT_CATM'),)
        self.owned_state = ('%s.co2' % self.name,)
        self.co2 = self.catm = None

    def _init(self, context):
        self.f.to(self.dev)
        self.b = None if self.b_cfg is None else as_f64(self.b_cfg, self.dev)
        if self.N == 0:
            self.N = (int(self.n_cfg) if self.n_cfg is not None else
                      self.b.shape[1] if self.b is not None else 0)

    # -- hooks ---------------------------------------------------------------
    def pre_force(self):
        x = self.f.get(self.group.step)
        n = self.N or x.numel()
        self.co2 = x.reshape(-1).expand(n).contiguous() if x.numel() == 1 \
            else x.reshape(-1)
        self.N = self.co2.numel()

    def pre_vic(self):
        if self.co2 is None:
            raise ComputeError('%s: U before F' % self.name)
        self.catm = CO2.catm_from_co2(self.co2, 1)
        # C has no CO2 put_data (OUT_CATM is VIC's own, put_data.c:157):
        # the day's row is the Catm vic_run reads, booked here
        self.put_data()

    def ppm(self):
        """co2_force.CO2 [N] (ppm), the crop's CO2."""
        return self.co2

    def put_data(self):
        z = self.zero()
        exp = dict(catm=CO2.ppm_from_catm(self.catm[:, -1]).clone())
        self._pending = (Balance(n_terms=1, scale=z), exp,
                         {k: v.clone().unsqueeze(0) for k, v in exp.items()})

    # -- the VIC side --------------------------------------------------------
    def core_hooks(self, core):
        if self.b is None:
            return
        G = self.group
        if G.vic is None:
            raise ComputeError('%s: b_co2 needs VIC in the group' % self.name)
        L = G.layout()
        nt = L['n_tile']
        nveg = self.b.shape[0]
        # tile k is vegetation class k; a bare tile beyond the classes gets
        # b_co2 = 0 (factor 1, vic_init.c:185-199)
        bc = torch.zeros(self.b.shape[1], nt, dtype=torch.float64,
                         device=self.dev)
        k = min(nt, nveg)
        bc[:, :k] = self.b[:k].T
        vdev = G.vic.driver.mosaic().cv.device
        cell = G._idx('cell', self.dev)
        core.b_co2 = bc.index_select(0, cell).to(vdev)
        core.co2ref = self.co2ref
        core.catm = lambda t: self.catm[:, 0].index_select(0, cell).to(vdev)

    # -- state ---------------------------------------------------------------
    def _state(self):
        return dict(co2=None if self.co2 is None else self.co2.clone(),
                    catm=None if self.catm is None else self.catm.clone())

    def _load(self, blob):
        c, a = blob.get('co2'), blob.get('catm')
        self.co2 = None if c is None else c.to(self.dev).clone()
        self.catm = None if a is None else a.to(self.dev).clone()
