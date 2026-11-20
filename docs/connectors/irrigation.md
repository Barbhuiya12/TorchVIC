# Irrigation and human water use connector

This page covers the lumped irrigation connector, which works from the
cell-mean soil moisture. The per-tile irrigation of VIC-WUR
(`vicwur_irrigation`) is described in [VIC-WUR](vicwur.md).

## What it does

The irrigation connector (`WaterUseConnector`, registry name `irrigation`,
`torchvic/coupling/connectors/wateruse.py`) adds irrigation and other human
water use to a coupled run. Its engine, `torchvic.wateruse.WaterUseEngine`,
implements the equations of VIC-WUR (Droppers et al., 2020; see
[References](../about/references.md)) as independent PyTorch code. No VIC-WUR
source file is copied. VIC-WUR itself is GPL-2.0 code, available at
<https://github.com/wur-wsg/VIC>. Each exchange interval (one day by default)
the connector computes demand, allocates it to sources, delivers the irrigation
water to VIC layer 0 and returns the losses; the steps are under
[Technical details](#technical-details). The example configuration is
`configs/vic_irrigation.yaml`.

## When to use it

* Use it when irrigation and other sector withdrawals are needed on top of VIC
  and the cell-mean soil moisture is enough to decide the demand.
* Use it with the [crop connector](crop.md) when the growing season and the
  rooting depth should come from a crop model.
* Do not use it when you need per-tile irrigation, paddy-rice saturated
  conductivity, remote withdrawals or non-renewable groundwater; use the
  VIC-WUR connectors ([VIC-WUR](vicwur.md)).
* Do not use it when withdrawals must come from the routed river and from dams;
  see [River routing](river.md) and the slots named under
  [Sources](#sources).

## Inputs

Connection inputs, received from other modules:

| input | file or variable | unit | meaning |
|---|---|---|---|
| `soil_moisture` | `vic.soil_moisture` (cell, layer) | mm | VIC layer moisture at the end of the interval; the demand is computed from it |
| `soil_unmet` | `vic.soil_add_unmet` (cell) | mm | what VIC could not add under the layer capacity (`soil_add_mode: cap`); returned to the leftover store |
| `season` | `crop.crop_active` (cell), optional | 1 | 1 where the irrigation season is open; otherwise the calendar window is used |
| `root_depth` | `crop.root_depth` (cell), optional | m | rooting depth from the crop connector; layers whose top lies above it are rooted |
| `gw_pumped` | `groundwater.pumping` (cell), only with `groundwater: true`, `lag: 1` | m3 | water the wells delivered |

Configuration inputs:

| input | file or variable | unit | meaning |
|---|---|---|---|
| irrigated tiles | `irrigated_tiles` | none | tiles of the cell that are irrigated |
| irrigated fraction | `irrigated_fraction` (file and variable) | none | irrigated fraction of the cell; demand arises there |
| season window | `season` with `start_doy`, `end_doy` | day of year | calendar growing season |
| critical point | `Wcr_FRACT` in the parameter file, or `wcr_fract` | none | critical point Wcr of the rooted layers |
| field capacity | `wfc_fract` (default Wcr / 0.7) | none | field capacity Wfc of the rooted layers |
| other sectors | `other` with `demand_mm_per_day` | mm per day | demand of municipal, energy, manufacturing and livestock use |
| consumption fraction | `consumption` (per other sector) | none | share of the withdrawal that is consumed |
| conveyance loss | `conveyance_loss` | none | conveyance loss |
| groundwater fraction | `gw_fraction` | none | share of the demand taken from groundwater |
| source stores | `river_store_m3`, `res_store_m3`, `gw_store_m3` | m3 | water held by each source (unlimited by default) |
| interval | `interval_days` | day | exchange interval (one day by default) |
| allocation | `allocation` | none | `priority` or `proportional` |
| fallback | `fallback` | none | on by default; groundwater and surface water serve each other's shortfall |
| loss return | `loss_return` | none | `river` (default) or `soil` |

## Config block

```yaml
modules:
  vic:
    connector: vic
    global_param: path/to/global_param.txt
    core: 5.1.0
    snow_bands: mean
    window_days: 1
    exchange: day
    soil_add_mode: cap
  irrigation:
    connector: irrigation
    vic: {global_param: path/to/global_param.txt, core: 5.1.0, snow_bands: mean}
    irrigated_fraction: {file: path/to/irrigation.nc, var: irr_frac}
    gw_fraction: {file: path/to/irrigation.nc, var: gw_frac}
    efficiency: 0.7
    conveyance_loss: 0.1
    allocation: priority
    loss_return: river
connections:
- {from: vic.soil_moisture, to: irrigation.soil_moisture}
- {from: vic.soil_add_unmet, to: irrigation.soil_unmet}
- {from: irrigation.soil_add, to: vic.soil_add, lag: 1}
```

Add `groundwater: true` and the groundwater connections described under
[Sources](#sources) for pumping. For the season and rooting depth from the crop
connector, add `crop.crop_active` to `irrigation.season` and `crop.root_depth`
to `irrigation.root_depth`.

## Outputs

| output | unit | meaning |
|---|---|---|
| `soil_add` | mm | irrigation delivered to VIC layer 0, mm over the cell |
| `demand` | m3 | water demand of all sectors this interval (diagnostic) |
| `unmet_demand` | m3 | demand no source could supply (diagnostic) |
| `gw_request` | m3 | groundwater withdrawn this interval, asked of the wells (only with `groundwater: true`) |
| `gw_payable` | m3 | groundwater taken and not yet delivered by the wells (diagnostic, only with `groundwater: true`) |

The connector also writes its ledger terms per interval (see
[Ledger](#ledger)).

## Technical details

### Demand

VIC's soil moisture at the end of the interval (`vic.soil_moisture`, cell mean)
is compared with the critical point Wcr and the field capacity Wfc over the
rooted layers. Demand arises in the irrigated fraction of the cell
(`irrigated_tiles` or `irrigated_fraction`) inside the growing season. The
season is a day-of-year window (`season` with `start_doy` and `end_doy`), or it
comes from the [crop connector](crop.md): `crop.crop_active` linked to
`irrigation.season`. The rooted layers are those with roots in the irrigated
tiles; `crop.root_depth` linked to `irrigation.root_depth` adds the layers
whose top lies above the crop's rooting depth. Wcr is read from the parameter
file (`Wcr_FRACT`) or set with `wcr_fract`; Wfc is Wcr / 0.7 or set with
`wfc_fract`. The other sectors (municipal, energy, manufacturing, livestock)
are given through `other` as demand arrays (`demand_mm_per_day`), each with a
consumption fraction, a conveyance loss and a groundwater fraction
(`consumption`, `conveyance_loss`, `gw_fraction`).

### Allocation

The demand is split between groundwater, reservoir and river and served by
sector in priority order (municipal, energy, manufacturing, livestock,
irrigation), never above what each source offers. What groundwater cannot serve
may be taken from surface water, and the other way round (`fallback`, on by
default). With `allocation: proportional` a shortfall is shared in proportion
to demand instead. Unmet demand is reported (`unmet_demand`).

### Delivery

The consumed irrigation water goes into VIC layer 0. `irrigation.soil_add` (mm
over the cell) is connected to `vic.soil_add` with `lag: 1`, so the water
computed for one day is added to VIC at the start of the next, as VIC-WUR adds
irrigation after `vic_run`. VIC is set to `exchange: day` and
`soil_add_mode: cap` (see [Connector API](../CONNECTOR_API.md)). The water is
added to the soil moisture state, not to precipitation, so it does not pass
interception or infiltration partitioning. What layer 0 cannot hold stays in a
leftover store. What VIC reports it could not take comes back as
`vic.soil_add_unmet`, linked to `irrigation.soil_unmet`, and is offered again.

### Returns

The conveyance and field losses of irrigation water go back to the river
(`loss_return: river`, the default) or to the soil (`loss_return: soil`). The
return flows of the other sectors go to the river, and their consumption leaves
as evapotranspiration.

### Sources

The river, reservoir and groundwater sources are stores held inside the
connector (`river_store_m3`, `res_store_m3`, `gw_store_m3`; unlimited by
default). The engine never takes more than a store holds, and returns to the
river are added to the river store. The river and reservoir stores are not
connected to the routing and reservoir connectors. For withdrawals from the
routed river and from dams, see the `withdrawal_request` and `withdrawn` slots
of the routing connector ([River routing](river.md)).

With `groundwater: true` the groundwater share is requested from the
[groundwater connector](groundwater.md) (`gw_request`) and paid for by what its
wells deliver one day later (`gw_pumped`, `lag: 1`).
`configs/vic_groundwater_irrigation.yaml` is an example; it gives the aquifer
properties one value for the whole domain, not regional data.

### Ledger

The connector keeps its water balance per interval, in m3 over the domain:

| term | meaning |
|---|---|
| `I_ext` | withdrawals from the sources plus the water VIC hands back (`soil_unmet`) |
| `Q_out` | `soil_add` delivered to VIC plus returns to the river |
| `ET` | consumption of the other sectors |
| `dS` | change of the leftover store |

With `groundwater: true` the groundwater share enters `I_ext` when the wells
deliver it, and the water requested but not yet delivered (`gw_payable`) enters
`dS`. The balance closes in every interval and over the run. In every interval
`validate` checks that no source is asked for more than it holds, that no more
than the demand is delivered and that the withdrawals are finite (with
`groundwater: true`, also that the wells did not deliver more than was asked).
The ledger is described in [Water accounting](../WATER_ACCOUNTING.md).

### State and restart

The connector checkpoints the leftover store, the water-use ledger, the source
stores and the groundwater payable. The runtime writes and reads checkpoints
(`checkpoint:` in the [configuration file](../CONFIGURATION.md)). A run
restarted from a checkpoint equals the continuous run.

### Data sources

Published datasets for India:

* irrigated area by crop and month: MIRCA-OS (Kebede et al., 2024) and
  MIRCA2000 (Portmann et al., 2010);
* sector withdrawals: Huang et al. (2018), Zenodo
  [10.5281/zenodo.1209296](https://doi.org/10.5281/zenodo.1209296);
* irrigation efficiency: Frenken and Gillet (2012);
* consumption fractions: Shiklomanov (2000) and Goldstein and Smith (2002), as
  used in Droppers et al. (2020).

Check each dataset's licence before use. For the code licence see
[License and credits](../downloads/code.md#license-and-credits) and
`THIRD_PARTY_NOTICES.md`.

## Limitations

Limits that apply to the whole coupled system are in
[Limits of the coupled system](../coupled/overview.md#limits).

* The demand uses the cell-mean soil moisture, because VIC exports no per-tile
  moisture. The addition goes to layer 0 of every tile of the cell (mm over the
  cell), so a tile that is wetter than the cell mean can reach the layer
  capacity first; `soil_unmet` returns what VIC cannot add.
* No FROZEN_SOIL or snow bands: these are the limits of the VIC connector's
  `soil_add` input, which this connector uses ([Connector
  API](../CONNECTOR_API.md)).
* Paddy-rice saturated conductivity, remote withdrawals and non-renewable
  groundwater are not part of the lumped connector; they are in the VIC-WUR
  connectors ([VIC-WUR](vicwur.md)).
* The VIC-WUR paper and code differ on where the conveyance and field losses of
  irrigation water go: the paper returns them to the soil, the code through the
  river. The connector follows the code by default; `loss_return: soil`
  follows the paper.

## Verification status

Checked in the test suite against the VIC-WUR equations (engine) and VIC's
Stehekin sample (connector); not part of the VIC verification on the
Verification page. A test whose sample data is absent is skipped
([Testing](../development/testing.md)).

* `tests/test_wateruse.py` tests the engine: the demand flags, priority and
  proportional allocation, source limits, conservation with both `loss_return`
  options, checkpoint and restart, and withdrawal and injection on the step
  router.
* `tests/test_conn_agri.py` tests the connector on VIC's Stehekin sample:
  ledger closure and the engine's cumulative identities (demand = withdrawn +
  unmet), source limits, that the water irrigation sends is the water VIC
  receives, the return of `soil_unmet` to the leftover store, VIC-only cells
  unchanged, and restart equal to the continuous run. The crop connector's
  tests are in the same file.


## Next step

[Groundwater connector](groundwater.md)
