# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""vicwur_efr: VIC-WUR environmental flow requirements (C9) as a connector.

F `pre_force`: efr_forcing + efr_derived_forcing on the START-of-step bottom
layer (the view at the day boundary): efr_moist [cell, tile, band] and the
held forcing efr_discharge (m3 s-1) and efr_baseflow (mm per model step).
With a land-use member in the group the start-of-step moisture is the one
after lu_apply (lu_force runs first in F, plugin_simulation.c:45-50), which
only exists inside VIC's step: the derivation then runs in `at_state` (the
group's state phase, VicConnector's vicwur_core), on the carried tile
states and the Cv in force.
Local and remote water use read `moist` and `q`; the dams read `q`.
O `put_data`: the OUT_EFR_* values.

Options
    discharge, baseflow   Series specs ([cell], cell order)
    soil                  dict of [cell] tensors for the bottom layer:
                          resid (resid_moist), depth (m), max_moist (mm),
                          dsmax (mm per day), ds, ws, c; optional valid
                          [cell, tile]
    exact                 True keeps the C cell-0 parameter quirk

State (checkpoint): moist, q, baseflow (held for P2 of the day the window
ends on), frac.

Ledger: EFR moves no water; its row is zero by construction, with the
OUT_EFR values in `series`.
"""
import torch

from ..connector import Balance, ComputeError
from ..spec import CouplingError, VarSpec
from ...vicwur import efr as E
from .vicwur_common import (GRID, STEP_S, Series, WurConnector, as_f64, put)

_SOIL = ('resid', 'depth', 'max_moist', 'dsmax', 'ds', 'ws', 'c')


class VicWurEfrConnector(WurConnector):
    name = 'vicwur_efr'
    version = '1'
    kind = 'efr'
    role = 'efr'
    flags = dict(efr=True)

    def __init__(self, discharge=None, baseflow=None, soil=None, exact=False,
                 gchunk=None, group=None, **config):
        super().__init__(group=group, **config)
        if discharge is None or baseflow is None or soil is None:
            raise CouplingError('%s: give discharge, baseflow and soil'
                                % self.name)
        miss = [k for k in _SOIL if k not in soil]
        if miss:
            raise CouplingError('%s: soil lacks %s' % (self.name, miss))
        self.q_f = Series(discharge, 'discharge')
        self.b_f = Series(baseflow, 'baseflow')
        self.soil_cfg, self.exact = soil, bool(exact)
        self.gchunk = gchunk
        self.N = int(torch.as_tensor(soil['ds']).numel())
        self._check_config()
        self.outputs = (VarSpec('efr_discharge', 'm3 s-1', ('cell',),
                                kind='state', grid=GRID),)
        self.owned_state = ('%s.moist' % self.name,)
        self.moist = self.q = self.bf = self.frac = None

    def _init(self, context):
        G, d = self.group, self.dev
        self.soil = {k: as_f64(self.soil_cfg[k], d) for k in _SOIL}
        for k, v in self.soil.items():
            if v.shape != (self.N,):
                raise CouplingError('%s: soil.%s must be [N]'
                                    % (self.name, k))
        sv = self.soil_cfg.get('valid')
        self.valid = None if sv is None else torch.as_tensor(sv, device=d)
        self.q_f.to(d)
        self.b_f.to(d)
        self.q = torch.zeros(self.N, dtype=torch.float64, device=d)
        self.bf = torch.zeros_like(self.q)
        self.frac = torch.zeros_like(self.q)
        self.moist = None
        self._q_sweep = None
        self._due = False

    def wants_state(self):
        """With land use, efr_forcing reads the moisture lu_apply left."""
        return self.group.roles.get('landuse') is not None

    def q_sweep(self):
        """efr_discharge in the router's sweep order."""
        return self.group.router.to_sweep(self.q)

    # -- hooks ---------------------------------------------------------------
    def pre_force(self):
        if self.wants_state():
            self._due = True              # at_state, after lu_apply
            return
        v = self.view
        if v is not None:
            m = v.moist[..., -1]
            ice = (v.ice[..., -1] if v.ice is not None
                   else torch.zeros_like(m)).unsqueeze(-1)
            cv, af = v.cv, v.af
        else:
            raise ComputeError('%s: no soil state at the start of the run '
                               '(the driver gives no view)' % self.name)
        self._derive(m, ice, cv, af)

    def at_state(self, states):
        """efr_forcing on the carried tile states at the step start (after
        land use), [column, tile, layer] -> [cell, tile, band]."""
        if not self._due:
            return
        self._due = False
        G = self.group
        m = G.from_columns(states.get('moist')[..., -1:])[..., 0]
        si = states.get('soil_ice')
        ice = (torch.zeros_like(m) if si is None else
               G.from_columns(si.reshape(states.get('moist').shape)
                              [..., -1:])[..., 0])
        lu = G.roles.get('landuse')
        cv = (lu.cv if lu is not None and lu.cv is not None
              else G.cell_rows(G.vic.driver.mosaic().cv))
        af = G.layout()['area_fract']
        d = self.dev
        self._derive(m.to(d, torch.float64), ice.unsqueeze(-1).to(
            d, torch.float64), cv.to(d, torch.float64), af.to(d))

    def _derive(self, m, ice, cv, af):
        s = self.soil
        q = self.q_f.get(self.group.step).expand(self.N).contiguous()
        b = self.b_f.get(self.group.step).expand(self.N).contiguous()
        st = E.efr_step(m, ice, torch.ones_like(m[:, 0, 0]).unsqueeze(1),
                        s['resid'], s['depth'], s['max_moist'], s['dsmax'],
                        s['ds'], s['ws'], s['c'], cv, af, b, q,
                        steps_per_day=1, valid=self.valid, bug=self.exact,
                        gchunk=self.gchunk)
        self.moist, self.frac = st.efr_moist, st.frac
        # q into the held tensor: the CUDA sweep graph reads its memory
        q = getattr(self, 'q', None)
        if torch.is_tensor(q) and q.shape == st.discharge.shape and \
                q.dtype == st.discharge.dtype and q.is_contiguous():
            q.copy_(st.discharge)
        else:
            self.q = st.discharge.clone()
        self.bf = st.baseflow
        self._cell_moist = st.cell_moist

    def put_data(self):
        z = self.zero()
        exp = dict(efr_discharge=self.q.clone())
        ser = dict(efr_discharge=self.q.clone().unsqueeze(0),
                   efr_baseflow=self.bf.clone().unsqueeze(0),
                   efr_moist=self._cell_moist.clone().unsqueeze(0))
        self._pending = (Balance(n_terms=1, scale=z), exp, ser)

    # -- state ---------------------------------------------------------------
    def _state(self):
        return dict(moist=None if self.moist is None else self.moist.clone(),
                    q=self.q.clone(), bf=self.bf.clone(),
                    frac=self.frac.clone(),
                    cell_moist=getattr(self, '_cell_moist', None),
                    due=self._due)

    def _load(self, blob):
        d = self.dev
        self.moist = None if blob['moist'] is None else blob['moist'].to(d)
        self.q, self.bf = blob['q'].to(d).clone(), blob['bf'].to(d).clone()
        self.frac = blob['frac'].to(d).clone()
        cm = blob.get('cell_moist')
        if cm is not None:
            self._cell_moist = cm.to(d).clone()
        self._due = bool(blob.get('due', False))
