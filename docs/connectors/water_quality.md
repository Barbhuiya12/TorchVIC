# Water quality and stream temperature

The `torchvic.quality` module transports constituents and heat on the river routing network.
The class `WaterQuality` routes every constituent, with first-order decay, and the heat of the
flow with the same linear operator as the water on the stepping router (`routing.StepRouter`),
and reports concentrations and stream temperature at the router's nodes. Point sources,
injections and withdrawals act at the router's sites, and a per-node ledger checks the
balance of water, mass and heat. The code is in `torchvic/quality/` (`transport.py`,
`heat.py`, `batch.py`), the tests are in `tests/test_quality.py`, and `tools/bench_wq_isc.py`
is a timing script. In the coupled system the module is the `quality` connector (see
[River routing](river.md); example configuration `configs/vic_quality_routing.yaml`). The
implementation is PyTorch tensor code written from the published equations listed under
Sources; no third-party source is copied.

## Constituent transport

`routing.StepRouter` routes water one step at a time with linear unit hydrographs, following
the conventions of RVIC, the routing model of the VIC project: every (cell, node) pair has a
unit hydrograph from the channel distance (Lohmann et al. 1996, 1998), the runoff of the cell
is added to a ring buffer weighted by it, and the head of the ring is the discharge. A
constituent mass flux is routed with the same linear operator, so mixing at confluences is
automatic (masses add) and the concentration at a node is

```text
C = M_flux / Q
```

where `M_flux` is the constituent flux and `Q` the discharge at the node. With the load in g,
`C` is in g m-3 (= mg L-1). `C = 0` where `Q <= qmin`, a valid-flow mask (`qmin` is 1e-9 m3 s-1
by default). A cell load `L_c` (argument `loads`, mass per step) enters pair (cell, node) as
`L_c * f_p / dt`, with `f_p` the area fraction of the pair.

### First-order decay

First-order decay, `dC/dt = -k C`, is exact along a path: mass with travel time `t` arrives
multiplied by `exp(-k t)`. The kernel of a constituent is

```text
UH_k[l] = UH[l] exp(-k l dt)
```

The rate `k` (s-1, argument `constituents={name: k}`) is a scalar or has one value per cell;
the rate of the emitting cell holds along its whole path. Point sources and injections enter
through the channel kernels of the sites, which are decayed in the same way. The mass decayed
on the way into each node is tracked and returned by `decayed(name)`. A conservative tracer
has `k = 0`.

### Point sources, injections and withdrawals

Point sources, injections and withdrawals act at the router's sites. The sites are visited
level by level, from upstream to downstream, in the order of `StepRouter.sweep`; the level of a
site is the length of the longest chain of sites above it, so a site sees the water that
reaches it after all the sites above it. Per site and step:

* a withdrawal of volume `v` (argument `withdraw`, m3) from the volume `V` passing the site
  removes the fraction `v / V` of every constituent and of the heat passing (mass taken =
  concentration x volume);
* a point source (argument `point`) injects the mass `m`, and injected water of volume `v_i`
  (argument `inject`) with concentration `C_i` (argument `inject_conc`) injects the mass
  `v_i C_i`;
* the water withdrawal is `take = min(request, cap)`, applied by the router, where `cap` is the
  largest volume whose removal leaves no ring slot of the water router negative. Every
  constituent and heat ring slot then loses the same fraction of its own content as the water in
  that slot (the change of the water in the slot divided by the water in it, a number in
  [-1, 0]). A withdrawal therefore never changes a concentration or temperature, even in a
  drained reach, so none can overshoot;
* all channels are stacked and updated in one batched operation per level, on the nodes that
  the level reaches.

## Stream temperature

Heat `Q T` (`T >= 0 C`, the freezing floor) is advected like a constituent. The surface heat
exchange is linearised about the equilibrium temperature `Te` (Edinger et al. 1968; Brady et al.
1969):

```text
dT/dt = -(K / (rho cp h)) (T - Te)
Te    = Td + Rs (1 - albedo)(1 - shade) / K
K     = 4.5 + 0.05 Ta + (beta + 0.47) f(W)
beta  = 0.35 + 0.015 Tm + 0.0012 Tm^2,    f(W) = 9.2 + 0.46 W^2,   Tm = (Td + Ta)/2
```

`Ta` is the air temperature (C), `Rs` the incoming shortwave radiation (W m-2), `W` the wind
speed (m s-1), `h` the mean depth (m) and `K` the surface exchange coefficient (W m-2 K-1), in
which the water temperature of the original formulation is replaced by the air temperature (no
iteration). `albedo` is the albedo of the water (default 0.06) and `shade` the fraction of the
shortwave blocked by riparian cover (default 0). `Td` is the dew point (C), obtained from the
vapour pressure `e` (kPa) by the Magnus form:

```text
Td = 237.3 g / (17.27 - g),    g = ln(e / 0.6108)
```

The volumetric heat capacity of water is `rho cp = 4.1813e6 J m-3 K-1` (`rho` = 1000 kg m-3,
`cp` = 4181.3 J kg-1 K-1). `Te` and the runoff temperature are floored at 0 C. Air temperature,
vapour pressure, wind speed and shortwave radiation (`t_air`, `vp`, `wind`, `sw` in `met`) are
given per cell and step; they are variables of the VIC forcing. The `quality` connector takes
them from input connections or from constants set under `met` in the configuration.

For constant `K` and `Te`, the solution along a path of travel time `t` is

```text
T(t) = Te + (T0 - Te) exp(-kT t),    kT = K_ref / (rho cp h)
```

with `T0` the temperature of the runoff (argument `t_runoff`, by default the air temperature)
and `K_ref` a constant reference exchange coefficient that replaces `K` in the rate. The
parameters `depth` (the mean depth `h`), `k_ref` (`K_ref`, 30 W m-2 K-1 by default), `albedo`
and `shade` are the keys of the `temperature` dictionary given to `WaterQuality`.

The solution is linear in `(T0, Te)`, so two routers carry it: the heat of the runoff, with the
kernel `UH exp(-kT l dt)`, and the heat taken from the atmosphere, with the kernel
`UH (1 - exp(-kT l dt))`. The temperature at a node is `T = (H1 + H2) / Q`, with `H1` and `H2`
the heat fluxes that arrive through the two routers. The heat gained from the atmosphere at each
node is returned by `atmosphere_heat()`.

### Comparison with RBM and limits of this version

RBM (Yearsley 2009; 2012) solves the full surface energy budget (net shortwave, longwave, latent
and sensible heat) with a semi-Lagrangian scheme. The equilibrium-temperature form used here
takes the same forcing variables and is a linearisation of that budget. Limits of this version:

* the relaxation rate uses a constant `K_ref` per cell; the weather dependence of `K` enters
  only through `Te`;
* `Te` is frozen at the time and cell of emission;
* the depth `h` is a constant per cell;
* riparian shading is a constant fraction (`shade`); there are no shading dynamics;
* groundwater and tributary inflows carry no heat beyond the runoff temperature;
* the module is not calibrated against observed water quality. It contains no loading or
  forcing data: loads, forcing and the runoff temperature are inputs, and the example
  configuration sets them to constants.

## Ledger

`WaterQuality.ledger()` returns, per node and for each router (water, every constituent and
heat), the cumulative residual

```text
in_runoff + in_hooks - out - stored
```

in m3 for water, in mass for a constituent and in m3 K for heat. It is zero to round-off.
`in_runoff` is the input from the cells credited to the node, `in_hooks` the net input of
injections and withdrawals credited to the node, `out` the flux read at the node times the
step (summed over the steps), and `stored` the content of the ring. Decayed mass and heat
taken from the atmosphere are the sink and the source outside the ring; they are reported by
`decayed(name)` and `atmosphere_heat()`.

## Fast router

On CUDA the quality engine runs on `routing_fast.FastStepRouter`
(`torchvic/model/routing_fast.py`, `routing_kernels.py`), a `StepRouter` whose hot paths are
fused Triton kernels; on the CPU it runs on `StepRouter`. `routing_fast.build_fast_step_router`
returns a `FastStepRouter` on CUDA, and the `routing` connector builds one on CUDA by default.
`WaterQuality` selects its channel implementation to match the router it is given.

For each level of sites, the fast router uses

* one kernel for the cap, the take or put and the fill of the tree-sum input;
* cuBLAS for the tree product;
* a gather kernel for the level sums, which keeps torch's own `sum(1)` because its addition
  order depends on the shape;
* one kernel for the ring update with the ledger credit.

The kernels of the water router perform the same floating-point operations in the same order
as the torch operations they replace. Division by `dt` is a multiplication by `1/dt`, as torch
divides by a scalar, and `a + b*c` is not fused into a multiply-add, as in torch.

### Pure-lag channels

Channels with one decay or relaxation rate per constituent or for heat (`batch.py`) share the
water kernel. Node sums are linear, so

```text
t_c[node, l] = d[l] sum_p x_c[p] UH[p, l]
```

with `d[l]` the lag factor of the channel and `x_c[p]` its input at pair `p`. All channels are
multiplied against the kernel in one batched product and the lag factors are applied
afterwards. Per-cell rates use one transformed kernel per channel instead. The node sums of
the batched channels are reduced by a separate kernel (`treered`) whose summation order
differs from torch's `sum(1)`, so they are not required to match the plain-torch channels to
the last bit.

## Tests

`tests/test_quality.py` (see [Testing](../development/testing.md)) checks, on the
CPU and, when CUDA is available, on the GPU: first-order decay on one reach
against the analytic solution (decayed plus delivered equals loaded); a
conservative tracer on a routed network (`C = c` for a uniform runoff
concentration, tracer mass equal to `c` times the water volume); decay with scalar
and per-cell rates; that a withdrawal removes the same fraction of the load as of
the water and leaves the concentration unchanged, also in a reach that is nearly
drained; withdrawals, injections and point sources on a network with closed
ledgers; the equilibrium temperature and the relaxation of the temperature to it;
the heat ledger; and that a step replayed from a CUDA graph agrees with the eager
step. The tests of the `quality` connector are in `tests/test_coupling_routing.py`
and are listed on the [River routing](river.md) page.

`tests/test_routing_fast.py` (CUDA only) checks that `FastStepRouter` and
`StepRouter`, stepped with random runoff, withdrawals and injections, give
identical discharge, delivered withdrawals and state tensors. The script
`tools/compare_fast_router_isc.py` makes the same comparison on a network built
from a domain file, a flow-direction grid and a gauge list, and times both
routers in a CUDA graph.

## Sources

* Edinger, J.E., Duttweiler, D.W., Geyer, J.C. (1968) The response of water temperatures
  to meteorological conditions. Water Resour. Res. 4:1137-1143.
* Brady, D.K., Graves, W.L., Geyer, J.C. (1969) Surface heat exchange at power plant
  cooling lakes. Edison Electric Institute Publ. 69-901.
* Yearsley, J.R. (2009) A semi-Lagrangian water temperature model for advection-dominated
  river systems. Water Resour. Res. 45:W12421. Yearsley (2012) HESS 16:4151-4164.
* Lohmann, D., Nolte-Holube, R., Raschke, E. (1996) Tellus 48A:708-721; Lohmann et al.
  (1998) Hydrol. Sci. J. 43:131-141 (routing used by the router).
* RVIC, the routing model of the UW-Hydro VIC project, <https://github.com/UW-Hydro/RVIC>
  (conventions followed by the routing; see also [References](../about/references.md)).
* Chapra, S.C. (1997) Surface Water-Quality Modeling. McGraw-Hill (first-order decay,
  mass balance at junctions).
