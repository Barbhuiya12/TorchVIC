"""The step-local writable view of VIC's tile state.

Inside one VIC-WUR step the plugins read and edit the VIC soil state after
vic_run (end-of-step moisture plus the edits so far): C5 pumps the bottom
layer, C8 irrigates layer 0 and then reads the pumped layer for its
shortage and stress. `StepView` is that state as a tensor view
[cell, tile, band, layer] over the connector's moisture and ice, with the
edits held as one delta. Nothing reaches VIC until `commit()`, which returns
the delta (the soil_add of the next vic_run, VicConnector.soil_add_bands);
`discard()` drops it (a rejected step). Everything is a masked tensor op on
the state's device: no host sync, no data-dependent branch.

Rules, each from VIC-WUR's C (src_tree/vic/plugins/):
  * irrigation (`add`): nothing when received <= 0; at most the room up to
    max_moist[layer] (or Wfc[layer] for EFFICIENT_IRRIGATION on a non-paddy
    tile), room floored at 0; applied + leftover = received; the water
    goes into the total moisture, ice untouched (irrigation/src/irr_use.c:
    245-269).
  * pumping (`withdraw`): moist -= amount, clipped at 0, on tiles with
    Cv > 0 and bands with AreaFract > 0 only (wateruse/src/wu_run_local.c:
    397-419). The C then computes the layer's ice and caps that local at
    the moisture (:421-433) but never stores it: the ice store is not
    changed. `floor='liquid'` is TorchVIC's stricter option (stop at the
    ice, take liquid only), off by default.
  * a column is counted at cell level with weight Cv * AreaFract[band]
    (vic_run.c:349-422; wu_run_local.c:410-413).
"""
import torch


class StepView(object):
    """moist, ice, max_moist, wfc: [cell, tile, band, layer] (ice may also
    carry a trailing frost axis; max_moist/wfc may be [cell, layer]);
    cv [cell, tile]; area_fract [cell, band]."""

    def __init__(self, moist, max_moist, cv, area_fract, ice=None, wfc=None,
                 frost_fract=None):
        self.base = moist
        self.max_moist = self._wide(max_moist)
        self.wfc = None if wfc is None else self._wide(wfc)
        self.cv = cv
        self.af = area_fract
        # live columns: Cv > 0 and AreaFract > 0 (wu_run_local.c:401,406)
        self.live = ((cv > 0).unsqueeze(2)
                     & (area_fract > 0).unsqueeze(1)).unsqueeze(-1)
        self.weight = (cv.unsqueeze(2) * area_fract.unsqueeze(1)
                       ).unsqueeze(-1)
        if ice is not None and ice.dim() == moist.dim() + 1:
            # VIC's frost areas (vic_init.c:711-725; the two end areas
            # halved), unless the caller gives soil_con.frost_fract
            if frost_fract is None:
                from ..core import frozen
                frost_fract = frozen.frost_fractions(ice.shape[-1])
            ice = (ice * torch.as_tensor(frost_fract, dtype=ice.dtype,
                                         device=ice.device)).sum(-1)
        self.ice = ice
        self.delta = torch.zeros_like(moist)
        #: the step's VIC OUT_RUNOFF and OUT_BASEFLOW [cell] mm per step,
        #: set by the driver before the plugins run (rout_run, the local
        #: dams); None when the driver gives none
        self.runoff = None
        self.baseflow = None
        #: VIC's stored cell->water_stress per tile of the step just run
        #: ([column, tile], vicwur_core water_stress; None otherwise): the
        #: crop component reads it (crop_run.c:108-148)
        self.water_stress = None
        #: the per-tile fluxes of the step just run ({'pet', 'esoil',
        #: 'canop': [column, tile], 'transp': [column, tile, layer]},
        #: vicwur_core tile_fluxes; None otherwise): what the crop reads
        self.tile_fluxes = None
        #: OUT_GWRECHARGE [cell] mm per step (GWM, VIC-WUR 3.0
        #: runoff.c:709-737), the recharge of the VIC-WUR 3.0 MODFLOW
        #: coupling; VicConnector sets it with `gwrecharge_view` (else None)
        self.gwrecharge = None

    def _wide(self, x):
        x = torch.as_tensor(x, dtype=self.base.dtype, device=self.base.device)
        if x.dim() == 2:                        # [cell, layer]
            x = x[:, None, None, :]
        return x

    @property
    def moist(self):
        """End-of-step moisture plus the edits so far."""
        return self.base + self.delta

    @property
    def liquid(self):
        m = self.moist
        return m if self.ice is None else m - self.ice

    def add(self, layer, received, efficient=None):
        """Irrigation (irr_use.c:245-269) on `layer`: `received` [cell,
        tile, band] mm (broadcast ok). `efficient` is a bool [cell, tile]
        (EFFICIENT_IRRIGATION and not paddy): the room is up to Wfc.
        Returns (applied, leftover), both [cell, tile, band]."""
        m = self.moist[..., layer]
        cap = self.max_moist[..., layer]
        if efficient is not None:
            if self.wfc is None:
                raise ValueError('add(efficient=...) needs the view built '
                                 'with wfc')
            cap = torch.where(efficient.unsqueeze(2), self.wfc[..., layer],
                              cap)
        recv = torch.as_tensor(received, dtype=m.dtype, device=m.device
                               ).expand_as(m)
        recv = torch.where(self.live[..., 0], recv, torch.zeros_like(recv))
        room = (cap - m).clamp_min(0.)
        applied = torch.where(recv > 0, torch.minimum(recv, room),
                              torch.zeros_like(recv))
        self.delta[..., layer] += applied
        return applied, recv.clamp_min(0.) - applied

    def withdraw(self, layer, amount, floor='liquid'):
        """Pumping (wu_run_local.c:410-419) from `layer`: `amount` [cell,
        tile, band] mm >= 0 taken from the total moisture and clipped at 0
        ('zero', VIC-WUR's C) or at the ice ('liquid', default: the C
        computes the ice but never stores it, wu_run_local.c:421-433). Returns (taken, unmet)
        [cell, tile, band], taken >= 0."""
        m = self.moist[..., layer]
        # a pump only takes: wu_run_local.c:397 acts on withdrawn_gw > 0
        amt = torch.as_tensor(amount, dtype=m.dtype, device=m.device
                              ).expand_as(m).clamp_min(0.)
        amt = torch.where(self.live[..., 0], amt, torch.zeros_like(amt))
        lo = torch.zeros_like(m)
        if floor == 'liquid' and self.ice is not None:
            lo = torch.minimum(self.ice[..., layer], m)
        elif floor not in ('zero', 'liquid'):
            raise ValueError("floor must be 'zero' or 'liquid'")
        taken = torch.minimum(amt, (m - lo).clamp_min(0.))
        self.delta[..., layer] -= taken
        return taken, amt - taken

    # -- cell subsets (the level sweep touches a few cells at a time) -------
    def layer_at(self, layer, cells):
        """End-of-step moisture of `layer` for `cells` (a long tensor),
        edits included: [n, tile, band]."""
        return (self.base.index_select(0, cells)[..., layer]
                + self.delta.index_select(0, cells)[..., layer])

    def ice_at(self, layer, cells):
        """The layer's ice for `cells` [n, tile, band] (zeros without
        FROZEN_SOIL)."""
        if self.ice is None:
            m = self.base.index_select(0, cells)[..., layer]
            return torch.zeros_like(m)
        return self.ice.index_select(0, cells)[..., layer]

    def withdraw_cells(self, layer, cells, amount, floor='liquid'):
        """`withdraw` (same floors, default 'liquid') on `cells` only:
        `amount` [n, tile, band] mm. Returns `taken` [n, tile, band], >= 0.
        The cells must be distinct. No host sync."""
        if floor not in ('zero', 'liquid'):
            raise ValueError("floor must be 'zero' or 'liquid'")
        m = self.layer_at(layer, cells)
        live = self.live.index_select(0, cells)[..., 0]
        amt = torch.as_tensor(amount, dtype=m.dtype, device=m.device
                              ).clamp_min(0.)
        amt = torch.where(live, amt, torch.zeros_like(amt))
        lo = torch.zeros_like(m)
        if floor == 'liquid' and self.ice is not None:
            lo = torch.minimum(self.ice_at(layer, cells), m)
        taken = torch.minimum(amt, (m - lo).clamp_min(0.))
        self.delta[..., layer].index_add_(0, cells, -taken)
        return taken

    def weight_at(self, cells):
        """Cv * AreaFract of `cells` [n, tile, band]."""
        return self.weight.index_select(0, cells)[..., 0]

    def cell_total(self, delta=None):
        """The edits as a cell total, mm: sum over tiles and bands of
        Cv * AreaFract * delta, over layers."""
        d = self.delta if delta is None else delta
        return (d * self.weight).sum((1, 2, 3))

    def commit(self):
        """The accumulated edits [cell, tile, band, layer] (the next
        vic_run's soil_add); the view starts again from zero."""
        d = self.delta
        self.delta = torch.zeros_like(d)
        self.base = self.base + d
        return d

    def save_state(self):
        return (self.base.clone(), self.delta.clone())

    def load_state(self, blob):
        self.base, self.delta = blob[0].clone(), blob[1].clone()

    def discard(self):
        """Drop the edits (a rejected step)."""
        self.delta = torch.zeros_like(self.delta)


class SoilView(object):
    """VIC's soil state between two days, read-only, per tile and band.

    Built by VicConnector from the loop's own layout (padded [column, tile,
    layer] or the GPU lanes [lane, 1, layer]; a column is a cell, or a
    (cell, band) pseudo-cell of the snow-band expansion). Nothing is
    computed until a property is read, and every property is a new tensor:
    writing to it never reaches VIC (edits go through StepView and
    soil_add). The day-coupled connectors see it as `Context.soil`.

      moist, ice   [cell, tile, band, layer] mm (ice frost-area weighted;
                   None without FROZEN_SOIL)
      cv           [cell, tile]
      area_fract   [cell, band]
    """

    def __init__(self, moist, ice, lanes, cv, band_index=None, n_cell=None,
                 band_weight=None):
        self._m, self._i, self._lanes = moist, ice, lanes
        self._cv = cv                    # [column, tile] (mosaic)
        self._bi = band_index            # (cell[col], band[col]) or None
        self._bw = band_weight           # AreaFract per column
        self.n_cell = n_cell if n_cell is not None else cv.shape[0]
        self.n_band = (1 if band_index is None
                       else int(band_index[2]))

    def _cols(self, x):
        """loop layout -> [column, tile, layer]"""
        ln = self._lanes
        if ln is None:
            return x.clone() if x.dim() == 3 else x.unsqueeze(1).clone()
        nl = x.shape[-1]
        pad = x.new_zeros(self._cv.shape[0], self._cv.shape[1], nl)
        pad[ln.lane_cell, ln.lane_tile] = x.reshape(-1, nl)
        return pad

    def _cells(self, x):
        """[column, tile, layer] -> [cell, tile, band, layer]"""
        if self._bi is None:
            return x.unsqueeze(2)
        c, b = self._bi[0], self._bi[1]
        out = x.new_zeros(self.n_cell, x.shape[1], self.n_band, x.shape[2])
        out[c, :, b] = x
        return out

    @property
    def moist(self):
        return self._cells(self._cols(self._m))

    @property
    def ice(self):
        if self._i is None:
            return None
        ice = self._i
        if ice.dim() == self._m.dim() + 1:
            from ..core import frozen
            w = torch.as_tensor(frozen.frost_fractions(ice.shape[-1]),
                                dtype=ice.dtype, device=ice.device)
            ice = (ice * w).sum(-1)
        return self._cells(self._cols(ice.reshape(self._m.shape)))

    @property
    def cv(self):
        if self._bi is None:
            return self._cv.clone()
        out = self._cv.new_zeros(self.n_cell, self._cv.shape[1])
        out[self._bi[0]] = self._cv          # equal over a cell's bands
        return out

    @property
    def area_fract(self):
        if self._bi is None:
            return self._cv.new_ones(self.n_cell, 1)
        out = self._cv.new_zeros(self.n_cell, self.n_band)
        out[self._bi[0], self._bi[1]] = self._bw
        return out

    def step_view(self, max_moist, wfc=None):
        """A StepView over this state (the plugins' writable view)."""
        return StepView(self.moist, max_moist, self.cv, self.area_fract,
                        ice=self.ice, wfc=wfc)
