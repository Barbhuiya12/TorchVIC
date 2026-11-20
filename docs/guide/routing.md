# Route runoff to gauges

VIC produces runoff and baseflow per cell; discharge at a gauge is those
fluxes routed down the river network. TorchVIC routes on the device,
following RVIC's conventions: every (cell, gauge) pair gets a unit
hydrograph from the channel distance between them (Lohmann et al. 1996),
and the gauge's discharge is the sum over its pairs of the cell's runoff
convolved with that hydrograph, weighted by cell area. Gauges nest: a
gauge downstream of another contains it. The convolution is one
depthwise `conv1d` over all pairs.

## What you need

* a **D8 flow-direction grid** on the domain (`[lat, lon]`, ESRI codes
  1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE; 0 a sink). The reader
  assumes the grid's row index increases with latitude, the layout of the
  image domain files and of the maps `tools/` writes; a map from a
  raster with north at row 0 must be flipped first. Sources and tools:
  [Your own domain](../tutorials/own-domain.md), step 7.
* the **gauge cells**: active-cell indices (snap each gauge to the cell
  whose grid catchment best matches its catchment polygon;
  `tools/gauge_review.py` does that by intersection over union).
* the run's **runoff plus baseflow** per cell and step, in mm, from the
  history files or from `out`.

## In Python

```python
import numpy as np, torch, xarray as xr
from torchvic.model import river_network as rn

d = xr.open_dataset('domain.nc')
lat, lon = d.lat.values, d.lon.values
mask = d.mask.values > 0
area = d.area.values.astype('f8')
fdir = xr.open_dataset('fdir.nc')['fdir'].values               # ESRI D8 codes on the domain grid

net, cells = rn.d8_downstream(fdir, mask, lat, lon)             # the network; cells [n, 2] = (row, col) per cell
print(net.n, 'cells, longest path %.0f km' % (net.dist_to_mouth.max() / 1e3))

gauge_cells = np.array([1234, 5678])                            # indices into `cells`
pairs = rn.build_pairs(net, gauge_cells)                        # every (cell, gauge) pair with its channel distance

dev = 'cuda'
dt = 86400.0                                                    # the runoff's step, s (daily history here)
cell_uh = torch.ones(1, dtype=torch.float64, device=dev)        # within-cell response (RVIC's default: none)
uh = rn.pair_unit_hydrographs(torch.as_tensor(pairs.distance, device=dev),
                              velocity=1.5, diffusion=800.0,    # VIC's routing defaults, m/s and m2/s
                              cell_uh=cell_uh, dt=dt)           # [n_uh, pair], columns sum to 1

# runoff [time, cell] in mm per step, cells in the order of `cells`
ds = xr.open_mfdataset('results/fluxes.*.nc')
r = (ds.OUT_RUNOFF + ds.OUT_BASEFLOW).values[:, cells[:, 0], cells[:, 1]]
r = torch.as_tensor(np.nan_to_num(r), device=dev)
q = rn.route_network(r, pairs, uh, torch.as_tensor(area[cells[:, 0], cells[:, 1]], device=dev), dt)
print(q.shape)                                                  # [time, gauge], m3/s
```

`velocity` and `diffusion` may be per pair (`[n_pair]` tensors). Gauges whose catchment is smaller than a few cells are
added with `rn.add_subcell_gauges(pairs, [(cell_indices, fractions), ...])`:
their discharge is the area-weighted sum of the covered cells' runoff,
routed with the cell hydrograph only.

`tools/route_isc.py` is this sequence as a command for the ISC case: it
reads the history files of a run directory, snaps the gauges, routes, and
writes `discharge [time, gauge]` as NetCDF and CSV.

## Single outlet, RVIC style

`torchvic.model.routing.route(runoff, unit_hydrograph, source2outlet,
n_outlets, ...)` is RVIC's convolution for a prepared source list, and
`route_reference` the transcription of RVIC's ring-buffer loop it is
tested against. `routing.lohmann_unit_hydrographs(distance, velocity,
diffusion, cell_uh, dt)` builds each cell's hydrograph to one outlet.
The convolution is linear and written in torch, so gradients flow to the
runoff and to the hydrographs.

## Inside the model

`basin.simulate` also carries a lumped per-case outlet
(`out['discharge_mm']`), used by the study report and by ensembles, where
each member routes to its own outlet.

## Next step

[Observations and ensembles](study.md) scores the discharge against gauge data. Routing inside a coupled run: [River routing connector](../connectors/river.md).
