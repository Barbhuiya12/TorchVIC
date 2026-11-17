# Irrigation and human water use connector

The irrigation connector (`WaterUseConnector`, registry name `irrigation`,
`torchvic/coupling/connectors/wateruse.py`) adds irrigation and other human
water use to a VIC run. Its engine, `torchvic.wateruse.WaterUseEngine`,
implements the equations of VIC-WUR (Droppers et al., 2020; see
[References](../about/references.md)) as independent PyTorch code. No VIC-WUR
source file is copied. VIC-WUR itself is GPL-2.0 code, available at
<https://github.com/wur-wsg/VIC>. This page covers the lumped connector, which
works from the cell-mean soil moisture; the per-tile irrigation of VIC-WUR
(`vicwur_irrigation`) is described in [VIC-WUR](vicwur.md). The example
configuration is `configs/vic_irrigation.yaml`.

## What it does

Each exchange interval (one day by default, `interval_days`) the connector runs
four steps.

1. **Demand.** VIC's soil moisture at the end of the interval
   (`vic.soil_moisture`, cell mean) is compared with the critical point Wcr and
   the field capacity Wfc over the rooted layers. Demand arises in the
   irrigated fraction of the cell (`irrigated_tiles` or `irrigated_fraction`)
   inside the growing season. The season is a day-of-year window (`season`
   with `start_doy` and `end_doy`), or it comes from the
   [crop connector](crop.md): `crop.crop_active` linked to `irrigation.season`.
   The rooted layers are those with roots in the irrigated tiles;
   `crop.root_depth` linked to `irrigation.root_depth` adds the layers whose
   top lies above the crop's rooting depth. Wcr is read from the parameter
   file (`Wcr_FRACT`) or set with `wcr_fract`; Wfc is Wcr / 0.7 or set with
   `wfc_fract`. The other sectors (municipal, energy, manufacturing,
   livestock) are given through `other` as demand arrays
   (`demand_mm_per_day`), each with a consumption fraction, a conveyance loss
   and a groundwater fraction (`consumption`, `conveyance_loss`,
   `gw_fraction`).
2. **Allocation.** The demand is split between groundwater, reservoir and
   river and served by sector in priority order (municipal, energy,
   manufacturing, livestock, irrigation), never above what each source offers.
   What groundwater cannot serve may be taken from surface water, and the other
   way round (`fallback`, on by default). With `allocation: proportional` a
   shortfall is shared in proportion to demand instead. Unmet demand is
   reported (`unmet_demand`).
3. **Delivery.** The consumed irrigation water goes into VIC layer 0.
   `irrigation.soil_add` (mm over the cell) is connected to `vic.soil_add` with
   `lag: 1`, so the water computed for one day is added to VIC at the start of
   the next, as VIC-WUR adds irrigation after `vic_run`. VIC is set to
   `exchange: day` and `soil_add_mode: cap` (see [Connector
   API](../CONNECTOR_API.md)). The water is added to the soil moisture state,
   not to precipitation, so it does not pass interception or infiltration
   partitioning. What layer 0 cannot hold stays in a leftover store. What VIC
   reports it could not take comes back as `vic.soil_add_unmet`, linked to
   `irrigation.soil_unmet`, and is offered again.
4. **Returns.** The conveyance and field losses of irrigation water go back to
   the river (`loss_return: river`, the default) or to the soil
   (`loss_return: soil`). The return flows of the other sectors go to the
   river, and their consumption leaves as evapotranspiration.

## Sources

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

## Ledger

The connector keeps its water balance per interval, in m3 over the domain:

| term | meaning |
|---|---|
| `I_ext` | withdrawals from the sources plus the water VIC hands back (`soil_unmet`) |
| `Q_out` | `soil_add` delivered to VIC plus returns to the river |
| `ET` | consumption of the other sectors |
| `dS` | change of the leftover store |

With `groundwater: true` the groundwater share enters `I_ext` when the wells
deliver it, and the water requested but not yet delivered (`gw_payable`)
enters `dS`. The balance closes in every interval and over the run. In every
interval `validate` checks that no source is asked for more than it holds, that
no more than the demand is delivered and that the withdrawals are finite (with
`groundwater: true`, also that the wells did not deliver more than was asked).
The ledger is described in [Water accounting](../WATER_ACCOUNTING.md).

## State and restart

The connector checkpoints the leftover store, the water-use ledger, the source
stores and the groundwater payable. The runtime writes and reads checkpoints
(`checkpoint:` in the [configuration file](../CONFIGURATION.md)). A run
restarted from a checkpoint equals the continuous run.

## Limits

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

## Data sources

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

## Tests

* `tests/test_wateruse.py` tests the engine: the demand flags, priority and
  proportional allocation, source limits, conservation with both `loss_return`
  options, checkpoint and restart, and withdrawal and injection on the step
  router.
* `tests/test_conn_agri.py` tests the connector on VIC's Stehekin sample:
  ledger closure and the engine's cumulative identities (demand = withdrawn +
  unmet), source limits, that the water irrigation sends is the water VIC
  receives, the return of `soil_unmet` to the leftover store, VIC-only cells
  unchanged, and restart equal to the continuous run. The crop connector's
  tests are in the same file. A test whose sample data is absent is skipped
  ([Testing](../development/testing.md)).
