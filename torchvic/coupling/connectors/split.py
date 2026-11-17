# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""SplitConnector: partitions a land-surface runoff between two consumers.

Where a flood model owns a part of the domain (a hydraulic box of VIC
cells), the runoff of those cells goes to the flood model and the rest to
the river router: each cell's water goes whole to one of them, so no water
is delivered twice and the coupled ledger (sent = delivered) closes.

    input   runoff   flux mm   on the VIC cell grid (vic.total_runoff)
    output  routed   flux mm   on the VIC cell grid (the box cells zero)
    output  box      flux mm   on the grid `box_cells` (the box's cells in
                               the order of `cells`)

Options: `cells` (a list of VIC cell indices, or a csv file with a column
`cell`) and `domain` (the domain file with lat, lon, mask, area, frac, the
areas of the cells: a depth is accounted as a volume).
"""
import numpy as np
import torch

from ..connector import Balance, ComputeError, Connector, Proposal
from ..spec import CouplingError, GridSpec, VarSpec

GRID = 'vic_cells'
BOX = 'box_cells'


def read_cells(cells, base_dir='.'):
    """[int] from a list or a csv file with a `cell` column."""
    import csv
    import os
    if isinstance(cells, (list, tuple)):
        return [int(c) for c in cells]
    p = str(cells)
    if not os.path.isabs(p):
        p = os.path.join(base_dir, p)
    return [int(r['cell']) for r in csv.DictReader(open(p))]


class SplitConnector(Connector):
    name = 'split'
    version = '1'
    backends = ('cpu', 'cuda')
    checkpoint = 'full'
    imported_fluxes = ('runoff',)
    exported_fluxes = ('routed', 'box')
    validation = ('a partition: routed + box = runoff, bit for bit '
                  '(tests/test_combos_glue.py)')
    interval_s = 86400.0

    def __init__(self, cells=None, domain=None, step_s=86400.0, area=None,
                 **config):
        super().__init__(**config)
        if cells is None or (domain is None and area is None):
            raise CouplingError('split: give cells and domain (or area)')
        self.cells_cfg, self.domain, self.area_cfg = cells, domain, area
        self.interval_s = float(step_s)
        self.cells = self.area = self.box_area = None
        self.inputs = (VarSpec('runoff', 'mm', ('cell',), kind='flux',
                               grid=GRID),)
        self.outputs = (VarSpec('routed', 'mm', ('cell',), kind='flux',
                                grid=GRID),
                        VarSpec('box', 'mm', ('cell',), kind='flux',
                                grid=BOX))

    def initialize(self, context):
        cells = read_cells(self.cells_cfg, context.base_dir)
        if self.area_cfg is not None:
            a = np.asarray(self.area_cfg, dtype=np.float64)
        else:
            import netCDF4
            d = netCDF4.Dataset(self.domain)
            mask = np.asarray(d['mask'][:]) > 0
            a = np.asarray(d['area'][:], dtype=np.float64)
            if 'frac' in d.variables:
                a = a * np.asarray(d['frac'][:], dtype=np.float64)
            a = a[mask]
        if min(cells) < 0 or max(cells) >= len(a) or \
                len(set(cells)) != len(cells):
            raise CouplingError('split: cells must be distinct indices of '
                                'the %d domain cells' % len(a))
        dev = context.device
        self.cells = torch.as_tensor(cells, dtype=torch.long, device=dev)
        self.area = torch.as_tensor(a, dtype=torch.float64, device=dev)
        self.box_area = self.area[self.cells]
        self.inbox = torch.zeros(len(a), dtype=torch.bool, device=dev)
        self.inbox[self.cells] = True

    def grids(self):
        if self.cells is None:
            return [GridSpec(BOX, 0, kind='cells')]
        return [GridSpec(BOX, int(self.cells.numel()),
                         area=self.box_area.cpu(), kind='cells',
                         description='the VIC cells handed to the flood '
                                     'model')]

    def compute(self, inputs, context, dt):
        x = inputs.get('runoff')
        if x is None:
            raise ComputeError('split: no runoff input')
        x = torch.as_tensor(x, device=self.cells.device)
        box = x[self.cells].clone()
        routed = torch.where(self.inbox, torch.zeros_like(x), x)
        v = x.to(torch.float64) * self.area / 1000.0
        vin = v.sum()
        vout = (routed.to(torch.float64) * self.area / 1000.0).sum() \
            + (box.to(torch.float64) * self.box_area / 1000.0).sum()
        return Proposal(
            self.name, context.t, context.t + dt,
            exports=dict(routed=routed, box=box),
            balance=Balance(I_ext=vin, Q_out=vout, n_terms=x.numel(),
                            scale=vin.abs() + vout.abs()),
            backend=str(x.device))

    def commit(self, proposal, context):
        pass
