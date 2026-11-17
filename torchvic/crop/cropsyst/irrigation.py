"""E12 and E13 of the VIC-CropSyst port (docs/connectors/cropsyst.md):
the irrigation trigger and amount, the irrigation loss partition (table default and the mechanistic
Ed / Ro option), and the technology and nozzle table. Pure kernels with no state: the caller (the
M4 scheme on VIC-WUR C8) owns the season, the crop and the soil layers and passes tensors in.

Sources (VIC_CropSyst = <data root>/static/upstream/cropsyst/VIC-CropSyst-Package/VIC_CropSyst):
  trigger and deficit      crop/VIC_soil_hydrology.c:44-204  need_irrigation
  closest layer            crop/VIC_soil_hydrology.c:872-890 closest_layer_at
  season gate, observation depth, MAD, gross amount, table partition
                           canopy_evap.c:319-349 (season, observation depth), :407-442, :482-497 (gross), :560-562, :632-641
  technology tables        agronomic/irrigation_lib.c:55-113 (nozzle), :153-290 (efficiency, capacity, MAD)
  Ed, eq. 2-3              crop/VIC_soil_hydrology.c:541-642 evaporation_from_irrigation_systems,
                           calc_wetted_percentage :775-824
  Ro, Philip, eq. 4-5      calc_irrigation_runoff :655-772,
                           calc_runoff_from_center_pivot :827-853, calc_infiltration_capacity :855-862

Re-derived from the shipped C++ (Academic Free License; no code text is copied). The nozzle
table holds the shipped numbers. Layout: leading dims are batch ([cell, crop] or any), the layer axis is
last. There is no Python loop over cells, crops or layers, no host sync and no data-dependent control
flow: every C++ branch is a `torch.where` and the 4-segment centre-pivot runoff is unrolled, so the
functions are CUDA-graph capturable. NaN handling follows the C comparison rules (std::max(a, b) is
`a < b ? b : a`, std::min(a, b) is `b < a ? b : a`).

Which parts are called and which are transcribed (oracle tests/oracle/ref_cropsyst_irrigation.cpp): need_irrigation,
closest_layer_at, calc_irrigation_runoff, the Philip pieces, calc_wetted_percentage and the library are CALLED in the
compiled reference; Ed and the Saxton block are the shipped source cut out and compiled. The gross-amount formula
(canopy_evap.c:489-497), the Ed/Ro/Dp table partition (canopy_evap.c:560-562, :632-641) and the observation depth
(canopy_evap.c:328-349) live inside the 1600-line canopy_evap() and cannot be called alone: the harness TRANSCRIBES
those lines, so for them the oracle test checks the port against a transcription of the cited lines, not a call.

Build facts the port relies on (checked against the compiled reference, <data root>/build/vic_cropsyst, whose
Makefile defines IRRIGATION_FROM_DEMAND, FULL_IRRIGATION, USE_IRRIGATION_PARAM_FILE,
IRRIG_PARAMETERFILE_DECIDE_IRRIGATION, POSTPONE_FIRST_IRRIGATION_PERENNIAL_CROPS, MAD_FOR_NETIRRIGATION_DEMAND
and leaves MECHANISTIC_IRRIGATION, MECHANISTIC_RUNOFF and USE_CROPSYST_CROP_IRRIGATION off):
- Table mode is the build default: Ed, Ro and Dp are fractions of the gross amount.
  Mechanistic Ed and Ro are an option; the reference build does not contain Ed (its source sits behind
  MECHANISTIC_IRRIGATION), so the oracle harness compiles that source range itself.
- Deficit-irrigation patterns (`amount_of_deficit_irrigation`, `read_irrig_pattern`, the proration in
  need_irrigation, canopy_evap.c:506-511 and calc_irrigation_runoff) exist only in the build without
  FULL_IRRIGATION. They are ported as an option (`pattern_amount`, `PatternTable`, the `deficit_amount_mm`
  argument of `need_irrigation` and `gross_irrigation`, `proration` of `irrigation_runoff`); off, the
  full-irrigation path is untouched. Their oracle reference is the shipped source of that path compiled into the
  harness with FULL_IRRIGATION undefined. Canopy interception of irrigation water
  (ENABLE_IRRIGATION_CANOPY_INTERCEPTION) is off, so the canopy takes none and irrig_over_soil is the
  amount after Ed.
- Hard-coded constants are parameters with the shipped values as defaults: the
  March-October window (`SeasonWindow`), the sprinkler/surface irrigation hours and area factor and the
  /10 and /100 divisors of Ed (`EdParams`), the postponed first-irrigation MAD, the 0.3 mm floor of the
  observation depth, BASIN_PRORATION (`proration`).
- The reference's quirks are kept for parity and named where they occur: sin() of an angle made with
  the truncated constant 0.0174532925 (not pi/180); the centre-pivot irrigation time divided by 3600
  twice; the first soil layer left out of the depletion sums; the deepest irrigated layer being the
  second-last (`Nlayer - 2`).
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import torch

IRRIGATION_TYPE_ACCOUNTS = 50                   # irrigation_lib.h; the table fills 49 rows, the last is zero
SEC_PER_DAY = 86400.0                           # canopy_evap.c:420-421
GRAVITY = 9.81                                  # vicNl_def.h:248 (VIC_CROPSYST_VERSION 3 build, VCS_V5 off)
DEG_TO_RAD_TRUNC = 0.0174532925                 # VIC_soil_hydrology.c:589 (not pi/180; kept for parity)
DEEPEST_LAYER_OFFSET = 2                        # need_irrigation: deepest irrigated layer = Nlayer - 2 (:85)
ROOT_ZERO_TOL = 0.00001                         # need_irrigation :87-92 (is_approximately tolerance)
MIN_OBS_DEPTH_MM = 0.3                          # canopy_evap.c:347
POSTPONED_MAD = 0.5                             # canopy_evap.c:438
PERFECT_DRIP_CODES = (63, 64)                   # IrrigTP_Sub_surf_drip_perfect[_eliminate_top]
PERFECT_DRIP_TOP_FRACTION = 0.2                 # canopy_evap.c:442 (code 64)
WATER_STRESS_TOLERANCE = 1.0                    # irrigation_lib.h:21 WATER_STRESS_TOLERANCE_FOR_IRRIGATION

# General_Irrigation_Type (irrigation_lib.h:84-98)
NO_GENERAL, CENTER_PIVOT, BIG_GUN, SOLID_SET, WHEEL_LINE, DRIP, FLOOD, RILL, FURROW, SPRINKLER, \
    MICRO_SPRINKLER = range(11)
# Depletion_observation_depth_mode (CropSyst/source/mgmt_types.h:38)
CONSTANT_FIXED_DEPTH, FRACTION_OF_ROOT_DEPTH, SOIL_PROFILE_DEPTH = range(3)

# Facility columns of the nozzle table (Irrigation_Facility_Charactor, irrigation_lib.h:68-82)
FAC_COLS = ("Runoff", "D", "h_op", "H", "cd", "y0_spk", "x_dm_spk", "Q", "q_nuzzle", "v", "A_Ap",
            "tirrig", "actual_droplet_size")
# Efficiency columns (Irrigation_Efficiency, irrigation_lib.h:25-60), set by management factor
EFF_COLS = ("time_irrig", "uniformity", "presumed_efficiency", "table_efficiency", "table_runoff_loss",
            "table_evap_loss", "table_deep_percolation_loss", "maximum_capacity", "max_allowable_depletion")

# (irrigation type code, 13 facility values) in table order, irrigation_lib.c:59-113
NOZZLE_TABLE = (
    (1, (0.03, 2.78, 4.1, 41.0, 0.9, 1.8, 11.7, 0.61, 23.0, 25.5, 0.1, 24.0, 1.16)),  # IrrigTP_CP_impact_14VH_RainBird
    (2, (0.03, 1.59, 1.7, 17.0, 0.9, 1.1, 7.8, 0.13, 15.0, 16.4, 0.1, 24.0, 1.42)),  # IrrigTP_CP_impact_M20VH_PM_RainBird
    (3, (0.03, 6.35, 5.5, 55.0, 0.9, 3.0, 19.8, 3.75, 27.0, 29.6, 0.1, 24.0, 1.68)),  # IrrigTP_CP_impact_65PJ_RainBird
    (4, (0.03, 3.18, 5.5, 55.0, 0.9, 2.7, 13.3, 0.92, 27.0, 29.6, 0.1, 24.0, 0.99)),  # IrrigTP_CP_impact_30FH_30FWH_RainBird
    (5, (0.03, 7.14, 3.5, 35.0, 0.9, 4.0, 14.0, 3.7, 10.0, 23.6, 0.1, 24.0, 2.74)),  # IrrigTP_CP_impact_L36H_L36AH_RainBird_1
    (6, (0.03, 3.97, 1.4, 14.0, 0.9, 4.0, 13.1, 0.95, 10.0, 14.9, 0.1, 24.0, 2.6845)),  # IrrigTP_CP_impact_L36H_L36AH_RainBird_2
    (7, (0.03, 8.7, 1.7, 17.0, 0.9, 4.3, 18.6, 3.88, 27.0, 16.4, 0.1, 24.0, 3.4)),  # IrrigTP_CP_impact_85EHD_RainBird_1
    (8, (0.03, 12.7, 3.5, 35.0, 0.9, 4.3, 26.6, 11.0, 27.0, 23.6, 0.1, 24.0, 2.17)),  # IrrigTP_CP_impact_85EHD_RainBird_2
    (9, (0.03, 17.5, 6.9, 69.0, 0.9, 4.3, 35.4, 29.0, 27.0, 33.1, 0.1, 24.0, 1.91)),  # IrrigTP_CP_impact_85EHD_RainBird_3
    (10, (0.03, 8.7, 1.7, 17.0, 0.9, 3.0, 15.8, 3.88, 17.0, 16.4, 0.1, 24.0, 2.52)),  # IrrigTP_CP_impact_85EHD_LA_RainBird_1
    (11, (0.03, 12.7, 3.5, 35.0, 0.9, 3.0, 24.2, 11.0, 17.0, 23.6, 0.1, 24.0, 2.17)),  # IrrigTP_CP_impact_85EHD_LA_RainBird_2
    (12, (0.03, 17.5, 6.9, 69.0, 0.9, 3.0, 32.9, 29.0, 17.0, 33.1, 0.1, 24.0, 1.91)),  # IrrigTP_CP_impact_85EHD_LA_RainBird_3
    (21, (0.03, 2.44064516129032, 1.0, 10.0, 0.9, 0.9, 10.4, 3.63, 5.0, 12.6, 0.1, 24.0, 1.89)),  # IrrigTP_CP_spray_S3000_Nelson
    (22, (0.03, 7.32193548387097, 1.0, 10.0, 0.9, 1.8, 14.3, 32.74, 5.0, 12.6, 0.1, 24.0, 3.05726451612903)),  # IrrigTP_CP_spray_O3000_Nelson
    (23, (0.03, 6.50838709677419, 2.04, 20.4, 0.9, 3.7, 22.6, 36.9, 5.0, 18.0, 0.1, 24.0, 2.06569259962049)),  # IrrigTP_CP_spray_R3000_Nelson_1
    (24, (0.03, 7.32193548387097, 1.7, 17.0, 0.9, 1.8, 17.7, 42.24, 5.0, 16.4, 0.1, 24.0, 2.33368500948767)),  # IrrigTP_CP_spray_R3000_Nelson_2
    (25, (0.03, 2.03387096774194, 0.69, 6.9, 0.9, 1.8, 9.1, 2.04, 5.0, 10.5, 0.1, 24.0, 2.00743338008415)),  # IrrigTP_CP_spray_A3000_Nelson_1
    (26, (0.03, 8.53548387096774, 0.69, 6.9, 0.9, 2.7, 16.5, 37.13, 5.0, 10.5, 0.1, 24.0, 2.27)),  # IrrigTP_CP_spray_A3000_Nelson_2
    (30, (0.03, 17.5, 6.9, 69.0, 0.9, 4.3, 35.4, 29.0, 27.0, 33.1, 0.1, 24.0, 1.91)),  # DEFAULT_CENTER_PIVOT
    (31, (0.05, 10.2, 3.0, 30.0, 0.9, 2.0, 47.0, 6.6, 24.0, 21.8, 0.2, 24.0, 2.116)),  # IrrigTP_Big_Gun_75TR_Nelson_1
    (32, (0.05, 14.0, 1.75, 17.5, 0.9, 2.0, 44.0, 9.5, 24.0, 16.7, 0.2, 24.0, 3.22)),  # IrrigTP_Big_Gun_75TR_Nelson_2
    (33, (0.05, 20.3, 6.0, 60.0, 0.9, 2.0, 87.0, 37.2, 24.0, 30.9, 0.2, 24.0, 2.112)),  # IrrigTP_Big_Gun_75TR_Nelson_3
    (34, (0.05, 17.8, 3.5, 35.0, 0.9, 2.0, 76.0, 23.0, 24.0, 23.6, 0.2, 24.0, 2.52)),  # IrrigTP_Big_Gun_150TB_Nelson_1
    (35, (0.05, 30.5, 7.0, 70.0, 0.9, 2.0, 128.0, 96.6, 24.0, 33.4, 0.2, 24.0, 2.35)),  # IrrigTP_Big_Gun_150TB_Nelson_2
    (36, (0.05, 35.6, 9.0, 90.0, 0.9, 2.0, 149.0, 148.0, 24.0, 37.8, 0.2, 24.0, 2.25)),  # IrrigTP_Big_Gun_150TB_Nelson_3
    (37, (0.05, 26.7, 4.0, 40.0, 0.9, 2.0, 104.0, 55.7, 27.0, 25.2, 0.2, 24.0, 2.902)),  # IrrigTP_Big_Gun_200TB_Nelson_1
    (38, (0.05, 40.6, 9.0, 90.0, 0.9, 2.0, 170.0, 195.0, 27.0, 37.8, 0.2, 24.0, 2.38)),  # IrrigTP_Big_Gun_200TB_Nelson_2
    (39, (0.05, 48.3, 9.0, 90.0, 0.9, 2.0, 188.0, 274.0, 27.0, 37.8, 0.2, 24.0, 2.588)),  # IrrigTP_Big_Gun_200TB_Nelson_3
    (40, (0.05, 30.5, 7.0, 70.0, 0.9, 2.0, 128.0, 96.6, 24.0, 33.4, 0.2, 24.0, 2.35)),  # DEFAULT_BIG_GUN
    (41, (0.05, 10.1693548387097, 2.75, 27.5, 0.9, 0.501515, 5.5, 110.0, 9.0, 20.9, 0.1, 24.0, 2.19)),  # IrrigTP_Solid_set_R5_POP_UP_Nelson
    (42, (0.05, 2.03387096774194, 2.75, 27.5, 0.9, 1.001515, 10.7, 254.0, 25.0, 20.9, 0.1, 24.0, 1.47750146627566)),  # IrrigTP_Solid_set_R2000WF_6_Nelson_1
    (43, (0.05, 3.0508064516129, 2.75, 27.5, 0.9, 1.001515, 11.9, 574.0, 25.0, 20.9, 0.1, 24.0, 1.57)),  # IrrigTP_Solid_set_R2000WF_6_Nelson_2
    (44, (0.05, 4.06774193548387, 4.5, 45.0, 0.9, 1.001515, 13.7, 1295.0, 25.0, 26.7, 0.1, 24.0, 1.52)),  # IrrigTP_Solid_set_R2000WF_6_Nelson_3
    (45, (0.05, 4.8, 4.5, 45.0, 0.9, 1.001515, 15.2, 1862.0, 25.0, 26.7, 0.1, 24.0, 1.556)),  # IrrigTP_Solid_set_R33_Nelson
    (50, (0.05, 4.06774193548387, 4.5, 45.0, 0.9, 1.001515, 13.7, 1295.0, 25.0, 26.7, 0.1, 24.0, 1.52)),  # DEFAULT_SOLID_SET
    (51, (0.05, 2.03387096774194, 2.75, 27.5, 0.9, 1.001515, 10.7, 254.0, 25.0, 20.9, 0.1, 24.0, 1.48)),  # IrrigTP_Moving_wheel_R2000WF_6_Nelson_1
    (52, (0.05, 3.0508064516129, 2.75, 27.5, 0.9, 1.001515, 11.9, 574.0, 25.0, 20.9, 0.1, 24.0, 1.57)),  # IrrigTP_Moving_wheel_R2000WF_6_Nelson_2
    (53, (0.05, 4.06774193548387, 4.5, 45.0, 0.9, 1.001515, 13.7, 1295.0, 25.0, 26.7, 0.1, 24.0, 1.52)),  # IrrigTP_Moving_wheel_R2000WF_6_Nelson_3
    (60, (0.05, 3.0508064516129, 2.75, 27.5, 0.9, 1.001515, 11.9, 574.0, 25.0, 20.9, 0.1, 24.0, 1.57)),  # DEFAULT_MOVING_WHEEL
    (61, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 5.0, 0.0)),  # IrrigTP_drip_0_0
    (62, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 5.0, 0.0)),  # IrrigTP_Sub_surf_drip_0_0
    (63, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 5.0, 0.0)),  # IrrigTP_Sub_surf_drip_perfect
    (64, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 5.0, 0.0)),  # IrrigTP_Sub_surf_drip_perfect_eliminate_top
    (70, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 5.0, 0.0)),  # DEFAULT_DRIP
    (71, (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 5.0, 0.0)),  # IrrigTP_flood_0_0
    (72, (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 5.0, 0.0)),  # IrrigTP_rill_0_0
    (73, (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 5.0, 0.0)),  # IrrigTP_furrow_0_0
    (74, (0.05, 4.06774193548387, 4.5, 45.0, 0.9, 1.001515, 13.7, 1295.0, 25.0, 26.7, 0.1, 24.0, 1.52)),  # DEFAULT_SPRINKLER
    (75, (0.05, 4.06774193548387, 4.5, 45.0, 0.9, 1.001515, 13.7, 1295.0, 25.0, 26.7, 0.1, 24.0, 1.52)),  # DEFAULT_MICRO_SPRINKLER
)


def _cmax(a, b):
    """std::max(a, b) = (a < b) ? b : a."""
    return torch.where(a < b, b, a)


def _cmin(a, b):
    """std::min(a, b) = (b < a) ? b : a."""
    return torch.where(b < a, b, a)


def _near(x, target, tol):
    """CORN::is_approximately(x, target, tol): |target - x| <= tol (false for NaN)."""
    return (target - x).abs() <= tol


def general_type(code: int) -> int:
    """identify_general_irrigation_type (irrigation_lib.c:119-141); host-side, used to build the table."""
    if 1 <= code <= 30:
        return CENTER_PIVOT
    if 31 <= code <= 40:
        return BIG_GUN
    if 41 <= code <= 50:
        return SOLID_SET
    if 51 <= code <= 60:
        return WHEEL_LINE
    if 61 <= code <= 70:
        return DRIP
    return {71: FLOOD, 72: RILL, 73: FURROW, 74: SPRINKLER, 75: MICRO_SPRINKLER}.get(code, NO_GENERAL)


def efficiency_row(code: int, management: int = 1) -> Tuple[float, ...]:
    """The EFF_COLS values of one technology for a management factor (1 fair, 2 poor, other good),
    set_irrigation_efficiency_of_irrigation_library, irrigation_lib.c:153-290. Rows of the reference table
    that no branch writes (code 0, the padding row) stay zero, as in the static C array."""
    g = general_type(code)
    m = management

    def pick(fair, poor, good):
        return fair if m == 1 else (poor if m == 2 else good)

    perfect = code in PERFECT_DRIP_CODES
    if g in (FLOOD, RILL, FURROW):
        return (0.0, 0.5, pick(0.70, 0.50, 0.80), 0.5, 0.15, 0.0, 0.35, 250.0, 0.5)
    if g == DRIP and not perfect:
        return (20.0, 0.95, pick(0.90, 0.70, 0.95), 0.97, 0.0, 0.03, 0.0, 30.0, 0.1)
    if perfect:
        return (20.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 30.0, 0.1)
    if g == WHEEL_LINE:
        return (0.0, 0.95, pick(0.8, 0.65, 0.85), 0.7, 0.0, 0.3, 0.0, 150.0, 0.5)
    if g == SOLID_SET:
        return (0.0, 0.95, pick(0.8, 0.65, 0.85), 0.8, 0.0, 0.2, 0.0, 100.0, 0.3)
    if g == SPRINKLER:
        return (0.0, 0.95, pick(0.8, 0.65, 0.85), 0.8, 0.0, 0.2, 0.0, 250.0, 0.5)
    if g == MICRO_SPRINKLER:
        return (0.0, 0.95, pick(0.8, 0.65, 0.85), 0.8, 0.0, 0.2, 0.0, 50.0, 0.2)
    if g == BIG_GUN:
        return (0.0, 0.95, pick(0.75, 0.6, 0.8), 0.55, 0.05, 0.4, 0.0, 500.0, 0.5)
    if g == CENTER_PIVOT:
        return (0.0, 0.92, pick(0.85, 0.75, 0.90), 0.85, 0.03, 0.12, 0.0, 30.0, 0.1)
    return (0.0,) * len(EFF_COLS)


class IrrigationLibrary(NamedTuple):
    """The technology table as tensors, one row per slot of Irrigation_library (50 rows)."""
    code: torch.Tensor           # [50] long; row 49 is the zero padding row
    general: torch.Tensor        # [50] long, General_Irrigation_Type
    facility: torch.Tensor       # [50, 13]  FAC_COLS
    eff: torch.Tensor            # [50, 9]   EFF_COLS


def build_library(management: int = 1, device="cpu", dtype=torch.float64) -> IrrigationLibrary:
    """Nozzle table plus the efficiency, capacity and MAD tables for `management` (the build default
    is IRRIGATION_MANAGEMENT_QUALITY = 1)."""
    n = IRRIGATION_TYPE_ACCOUNTS
    codes = [c for c, _ in NOZZLE_TABLE] + [0] * (n - len(NOZZLE_TABLE))
    fac = [list(v) for _, v in NOZZLE_TABLE] + [[0.0] * len(FAC_COLS)] * (n - len(NOZZLE_TABLE))
    eff = [list(efficiency_row(c, management)) if c else [0.0] * len(EFF_COLS) for c in codes]
    gen = [general_type(c) if c else NO_GENERAL for c in codes]
    return IrrigationLibrary(torch.tensor(codes, dtype=torch.long, device=device),
                             torch.tensor(gen, dtype=torch.long, device=device),
                             torch.tensor(fac, dtype=dtype, device=device),
                             torch.tensor(eff, dtype=dtype, device=device))


def lookup(lib: IrrigationLibrary, code: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """get_irrig_lib_index: row of each technology code, and whether it exists. Any batch shape. The
    padding row (code 0) never matches, as the reference returns -1 for code 0 (and for any code not in
    the table, where it exits the program; here the caller masks with `found`)."""
    eq = (lib.code[None, :] == code.reshape(-1, 1)) & (lib.code[None, :] != 0)
    found = eq.any(dim=1)
    idx = eq.to(torch.int32).argmax(dim=1)
    return idx.reshape(code.shape), found.reshape(code.shape)


def _row(table: torch.Tensor, idx: torch.Tensor, col: int) -> torch.Tensor:
    return table[idx, col]


class TechRow(NamedTuple):
    """Table values gathered for a batch of technologies (each shaped like `code`)."""
    idx: torch.Tensor
    found: torch.Tensor
    general: torch.Tensor
    fac: torch.Tensor            # [..., 13]
    eff: torch.Tensor            # [..., 9]

    def f(self, name: str) -> torch.Tensor:
        return self.fac[..., FAC_COLS.index(name)]

    def e(self, name: str) -> torch.Tensor:
        return self.eff[..., EFF_COLS.index(name)]


def tech_row(lib: IrrigationLibrary, code: torch.Tensor) -> TechRow:
    idx, found = lookup(lib, code)
    return TechRow(idx, found, lib.general[idx], lib.facility[idx], lib.eff[idx])


def fill_to_fc_flag(code: torch.Tensor) -> torch.Tensor:
    """canopy_evap.c:427-428: the perfect sub-surface drips fill the root zone to field capacity."""
    return (code == PERFECT_DRIP_CODES[0]) | (code == PERFECT_DRIP_CODES[1])


def top_layer_fraction(code: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """canopy_evap.c:442: 0.2 for code 64 (the top layer only goes to 20 % of FC), else 1."""
    one = torch.ones(code.shape, dtype=dtype, device=code.device)
    return torch.where(code == PERFECT_DRIP_CODES[1], one * PERFECT_DRIP_TOP_FRACTION, one)


# ----------------------------------------------------------------------------- season and depth

class SeasonWindow(NamedTuple):
    """The hard-coded March-October irrigation window of canopy_evap.c:325 (made a
    parameter; it would shut out the rabi season in India)."""
    first_month: int = 3
    last_month: int = 10


def irrigation_season(month: torch.Tensor, has_irrigation_params: torch.Tensor,
                      manual_target_m: torch.Tensor, do_irrigate: torch.Tensor,
                      window: SeasonWindow = SeasonWindow(), daily_step: bool = True) -> torch.Tensor:
    """canopy_evap.c:319-328, :407: is an irrigation decision taken today. `has_irrigation_params` is
    the crop having an irrigation record; a manual event target > 1e-6 m also opens the season; the
    month window then closes it; `do_irrigate` is options.VCS.do_irrigate_crop; only the daily step
    irrigates (the reference tests delta_t == 86400)."""
    season = has_irrigation_params | (manual_target_m > 1.0e-6)
    season = season & (month >= window.first_month) & (month <= window.last_month)
    return season & do_irrigate & daily_step


def observation_depth_mm(mode: torch.Tensor, const_depth_m: torch.Tensor, root_depth_mm: torch.Tensor,
                         root_fraction: torch.Tensor, total_thickness_mm: torch.Tensor) -> torch.Tensor:
    """Depth over which depletion is observed, canopy_evap.c:328-349: a fixed depth, a fraction of the
    root depth or the whole profile; floored at 0.3 mm (the reference compares millimetres with 0.3)."""
    d = const_depth_m * 1000.0
    d = torch.where(mode == FRACTION_OF_ROOT_DEPTH, root_depth_mm * root_fraction, d)
    d = torch.where(mode == SOIL_PROFILE_DEPTH, total_thickness_mm, d)
    return torch.where(d < MIN_OBS_DEPTH_MM, torch.full_like(d, MIN_OBS_DEPTH_MM), d)


def allowable_depletion(table_mad: torch.Tensor, first_irrigation_postponed: torch.Tensor,
                        postponed_mad: float = POSTPONED_MAD) -> torch.Tensor:
    """canopy_evap.c:431-439: the technology's MAD, except that a perennial crop (alfalfa, pasture) that has
    not yet been irrigated this year waits until depletion reaches 0.5. The caller builds the mask from the
    crop codes (7701, 8205, 8704 in the reference) and `this_year_start_irrigated`."""
    return torch.where(first_irrigation_postponed, torch.full_like(table_mad, postponed_mad), table_mad)


# ----------------------------------------------------------------------------- E12 trigger

def closest_layer_at(depth_m: torch.Tensor, given_depth_m: torch.Tensor) -> torch.Tensor:
    """closest_layer_at (VIC_soil_hydrology.c): 0-based layer whose lower boundary is nearest the depth. Walk down
    to the first layer whose cumulative thickness reaches the depth (or the last layer); step back one layer if
    more than half of that layer lies below the depth. depth_m [..., L], given_depth_m [...]."""
    L = depth_m.shape[-1]
    cum = torch.cumsum(depth_m, dim=-1)
    reached = cum >= given_depth_m[..., None]
    done = reached | (torch.arange(L, device=depth_m.device) == L - 1)
    layer = done.to(torch.int32).argmax(dim=-1)
    gi = layer[..., None]
    cum_at = cum.gather(-1, gi)[..., 0]
    thick_at = depth_m.gather(-1, gi)[..., 0]
    hit = reached.gather(-1, gi)[..., 0]
    back = hit & ((cum_at - given_depth_m) > thick_at / 2.0) & (layer > 0)
    return layer - back.to(layer.dtype)


class Decision(NamedTuple):
    irrigate: torch.Tensor           # bool [...]
    demand_mm: torch.Tensor          # deficit to field capacity down to the recharge layer, 0 if no irrigation
    depletion: torch.Tensor          # 1 - available / maximum available over the observation layers
    real_added_mm: torch.Tensor      # water put into the soil by the fill option (0 otherwise)
    moist_mm: torch.Tensor           # [..., L] layer moisture after the fill option
    recharge_layer: torch.Tensor
    observe_layer: torch.Tensor


def need_irrigation(mad: torch.Tensor, observe_depth_mm_: torch.Tensor, root_depth_mm: torch.Tensor,
                    depth_m: torch.Tensor, field_capacity_mm: torch.Tensor, wpwp_mm: torch.Tensor,
                    moist_mm: torch.Tensor, water_stress_index: torch.Tensor,
                    water_stress_tolerance=WATER_STRESS_TOLERANCE,
                    fill_to_fc: Optional[torch.Tensor] = None,
                    max_capacity_mm: Optional[torch.Tensor] = None,
                    top_layer_to_fc_fraction: Optional[torch.Tensor] = None,
                    deficit_amount_mm: Optional[torch.Tensor] = None) -> Decision:
    """E12 trigger and deficit, need_irrigation (VIC_soil_hydrology.c:44-204, full-irrigation build).

    Irrigate when (depletion > MAD or water stress index > tolerance) and the deficit to field capacity is
    positive. The deficit is summed from layer 0 to the recharge layer (the root-zone layer, but at least the
    observation layer and at most `Nlayer - 2`); depletion uses layers 1..observation layer, because the top
    evaporation layer is excluded (:111). Layer fields are [..., L], the rest [...]. With `fill_to_fc` (the
    perfect sub-surface drips) the decision also fills layers 0..recharge up to field capacity, top layer to
    `top_layer_to_fc_fraction` of it, from a budget of `max_capacity_mm`, in layer order; the new layer moisture
    and the water added are returned. Without it `moist_mm` is returned unchanged.

    Deficit-irrigation option (the build without FULL_IRRIGATION, VIC_soil_hydrology.c:170-176): pass
    `deficit_amount_mm` from `pattern_amount`. Where it is >= 0 the pattern overrides the decision: irrigate
    exactly when it is > 0, with that amount as the demand; where it is -1 (crop not in the pattern table) the
    full-irrigation decision stands."""
    L = depth_m.shape[-1]
    deepest = L - DEEPEST_LAYER_OFFSET
    idx = torch.arange(L, device=depth_m.device)
    rec0 = torch.where(_near(root_depth_mm, 0.0, ROOT_ZERO_TOL), torch.full(root_depth_mm.shape, deepest, dtype=torch.long, device=depth_m.device),
                       closest_layer_at(depth_m, root_depth_mm * 0.001) + 1)
    obs0 = torch.where(_near(observe_depth_mm_, 0.0, ROOT_ZERO_TOL),
                       torch.full(observe_depth_mm_.shape, deepest, dtype=torch.long, device=depth_m.device),
                       closest_layer_at(depth_m, observe_depth_mm_ * 0.001) + 1)
    rec = torch.clamp(rec0, max=deepest)
    obs = torch.clamp(obs0, max=deepest)
    rec = torch.where(rec < obs, obs, rec)
    in_rec = idx <= rec[..., None]
    in_obs = idx <= obs[..., None]
    zero = torch.zeros_like(field_capacity_mm)
    deficit = torch.where(in_rec, _cmax(field_capacity_mm - moist_mm, zero), zero).sum(-1)
    count = in_obs & (idx != 0)
    max_avail = torch.where(count, _cmax(field_capacity_mm - wpwp_mm, zero), zero).sum(-1)
    avail_l = _cmin(moist_mm - wpwp_mm, field_capacity_mm - wpwp_mm)
    act_avail = torch.where(count, _cmax(avail_l, zero), zero).sum(-1)
    depletion = 1.0 - act_avail / max_avail
    irrigate = ((depletion > mad) | (water_stress_index > water_stress_tolerance)) & (deficit > 0.0)
    demand = torch.where(irrigate, deficit, torch.zeros_like(deficit))
    real_added = torch.zeros_like(deficit)
    moist_new = moist_mm
    if fill_to_fc is not None:
        cap = max_capacity_mm if max_capacity_mm is not None else torch.zeros_like(deficit)
        topf = top_layer_to_fc_fraction if top_layer_to_fc_fraction is not None else torch.ones_like(deficit)
        to = torch.where(idx == 0, topf[..., None] * field_capacity_mm, field_capacity_mm)
        need = torch.where(in_rec & (moist_mm < to), to - moist_mm, zero)
        before = torch.cumsum(need, dim=-1) - need
        add = _cmin(need, _cmax(cap[..., None] - before, zero))
        do = (irrigate & fill_to_fc)[..., None]
        add = torch.where(do, add, zero)
        moist_new = moist_mm + add
        real_added = add.sum(-1)
    if deficit_amount_mm is not None:
        has = deficit_amount_mm >= 0.0
        irrigate = torch.where(has, deficit_amount_mm > 0.0, irrigate)
        demand = torch.where(has, torch.where(irrigate, deficit_amount_mm, torch.zeros_like(demand)), demand)
    return Decision(irrigate, demand, depletion, real_added, moist_new, rec, obs)


# ----------------------------------------------------------------------------- E12 amount, table losses

class Partition(NamedTuple):
    gross_mm: torch.Tensor       # total_irrig: water taken from the source
    evap_mm: torch.Tensor        # Ed, lost to the atmosphere
    after_evap_mm: torch.Tensor  # irrig_aft_evap = irrig_over_soil (no canopy interception, see module notes)
    runoff_mm: torch.Tensor      # Ro, to the surface-runoff ring
    deep_perc_mm: torch.Tensor   # Dp
    net_mm: torch.Tensor         # irrig_for_pot_infiltration: what enters the soil


def gross_irrigation(demand_mm: torch.Tensor, tech: TechRow, delta_t_sec=SEC_PER_DAY,
                     mechanistic: bool = False, filled_mm: Optional[torch.Tensor] = None,
                     code: Optional[torch.Tensor] = None,
                     deficit_amount_mm: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Gross amount = demand / efficiency, capped at the technology capacity (canopy_evap.c:420-421, :482-497).
    Table mode divides by the table efficiency; mechanistic mode by uniformity x presumed efficiency. The
    deficit option replaces the amount by the pattern amount where it is >= 0 (:506-511), after the capacity cap
    and not for the perfect drips. The perfect sub-surface drips (codes 63, 64) take exactly the water filled into the soil (`filled_mm`, from
    need_irrigation with fill_to_fc, :493-494). Capacity is mm/day times delta_t / 86400."""
    cap = tech.e("maximum_capacity") * delta_t_sec / SEC_PER_DAY
    eff = tech.e("uniformity") * tech.e("presumed_efficiency") if mechanistic else tech.e("table_efficiency")
    total = demand_mm / eff
    total = torch.where(total > cap, cap, total)
    if deficit_amount_mm is not None:
        total = torch.where(deficit_amount_mm >= 0.0, deficit_amount_mm, total)
    if filled_mm is not None and code is not None:
        total = torch.where(fill_to_fc_flag(code), filled_mm, total)
    return total


def table_partition(gross_mm: torch.Tensor, tech: TechRow) -> Partition:
    """Table default: Ed, Ro and Dp are table fractions of the gross amount (canopy_evap.c:560-562, :632-641).
    Dp is capped at gross - Ro. The net water for infiltration is gross - Ed - Ro - Dp (Dp stays in the profile
    in the shipped `180524LML` choice, so the caller decides where it goes)."""
    ed = tech.e("table_evap_loss") * gross_mm
    aft = gross_mm - ed
    ro = gross_mm * tech.e("table_runoff_loss")
    dp = gross_mm * tech.e("table_deep_percolation_loss")
    dp = torch.where(dp > (gross_mm - ro), gross_mm - ro, dp)
    return Partition(gross_mm, ed, aft, ro, dp, aft - ro - dp)


# ----------------------------------------------------------------------------- E13 Ed

class EdParams(NamedTuple):
    """The constants hard-coded in evaporation_from_irrigation_systems."""
    sprinkler_hours: float = 3.0         # big gun, solid set, wheel line: time_of_irrig = 3.0 (:624)
    sprinkler_area_fraction: float = 1.0 # Ap = 1.0 for those (:625)
    surface_hours: float = 2.0           # flood, rill, furrow: time_of_irrig = 2.0 (:631)
    hours_divisor: float = 10.0          # "time_of_irrig / 10.0" (:627, :632, :636)
    amount_divisor: float = 100.0        # "irrigation_amount / 100.0" (:622, :626)
    drip_wetted_min: float = 0.2         # must_be_between(wetted, 0.2, 0.9) (:634-635)
    drip_wetted_max: float = 0.9
    t_exponent: float = 1.5707           # K_t (:615)
    d_exponent: float = -0.52044         # k_D (:616)


class WettedParams(NamedTuple):
    """The constants hard-coded in calc_wetted_percentage (:775-824)."""
    drip_hours: float = 7.0
    row_spacing_annual_m: float = 1.0
    col_spacing_annual_m: float = 1.0
    emitters_annual: float = 1.0
    row_spacing_perennial_m: float = 2.7
    col_spacing_perennial_m: float = 2.5
    emitters_perennial: float = 3.0
    pi_perennial: float = 3.14           # the literal 3.14 of the perennial branch (:814)
    furrow_fraction: float = 0.5


def wetted_percentage(tech: TechRow, amount_mm: torch.Tensor, ksat_mm_day: torch.Tensor, wpwp0_mm: torch.Tensor,
                      fc0_mm: torch.Tensor, bulk_density0: torch.Tensor, perennial: torch.Tensor,
                      p: WettedParams = WettedParams()) -> torch.Tensor:
    """Fraction of the area wetted (calc_wetted_percentage): 0 for sprinklers, a Malek-and-Peters wetted
    diameter for drip, 1 (0.5 for furrow) for surface systems. Technologies that are none of these (the
    generic sprinkler and micro-sprinkler types) fall into the surface branch, as in the reference."""
    g = tech.general
    def pick(a, b):                      # python scalars in torch.where would come back float32
        return torch.where(perennial, torch.full_like(amount_mm, a), torch.full_like(amount_mm, b))
    area = pick(p.row_spacing_perennial_m * p.col_spacing_perennial_m,
                p.row_spacing_annual_m * p.col_spacing_annual_m)
    nemit = pick(p.emitters_perennial, p.emitters_annual)
    row = pick(p.row_spacing_perennial_m, p.row_spacing_annual_m)
    col = pick(p.col_spacing_perennial_m, p.col_spacing_annual_m)
    vol = area * amount_mm
    discharge = vol / p.drip_hours / nemit
    ks = ksat_mm_day / 24.0 / 10.0
    d_teta = (wpwp0_mm / 100.0 + fc0_mm / 100.0) / 2.0
    pb = bulk_density0 / 1000.0
    d_mp = (torch.pow(discharge, 0.543) * torch.pow(ks, 0.772) * torch.pow(torch.full_like(ks, p.drip_hours), 0.419)
            * torch.pow(d_teta, -0.687) * torch.pow(pb, 0.305))
    w_per = nemit * torch.pow(d_mp / 100.0, 2.0) * p.pi_perennial / row / col
    w_ann = nemit * d_mp / 100.0 / row / col
    w = torch.where(perennial, w_per, w_ann)
    w = torch.where(w > 1.0, torch.ones_like(w), w)
    surface = torch.where(g == FURROW, torch.full_like(w, p.furrow_fraction), torch.ones_like(w))
    sprink = (g == CENTER_PIVOT) | (g == BIG_GUN) | (g == SOLID_SET) | (g == WHEEL_LINE)
    out = torch.where(g == DRIP, w, surface)
    return torch.where(sprink, torch.zeros_like(out), out)


def droplet_evaporation(tech: TechRow, amount_mm: torch.Tensor, et0_mm: torch.Tensor,
                        et0_open_water_mm: torch.Tensor, wetted_pct: torch.Tensor,
                        p: EdParams = EdParams()) -> torch.Tensor:
    """E13 mechanistic Ed (eq. 2-3), evaporation_from_irrigation_systems (VIC_soil_hydrology.c:542-640).

    Sprinklers (centre pivot, big gun, solid set, wheel line): Ed = Ap ET0 D_term t_term amount / 100, with
    t_term the droplet flight time to the (crop) height raised to K_t and D_term a droplet-size term
    (pressure head over 10 x droplet size) ** k_D; big gun, solid set and wheel line use a 3 h irrigation time
    and Ap = 1, which add `* time / 10`. Surface systems: ET0_open_water x wetted x 2 / 10. Drip: ET0_open_water x
    wetted (clipped to 0.2-0.9) x tirrig / 10. The generic sprinkler and micro-sprinkler types give 0. The
    nozzle angle uses the truncated 0.0174532925 of the reference."""
    g = tech.general
    h = tech.f("h_op") * 100.0
    teta = tech.f("q_nuzzle") * DEG_TO_RAD_TRUNC
    y0 = tech.f("y0_spk")
    v0 = tech.f("v")
    ap = tech.f("A_Ap")
    t_term = torch.pow(v0 * torch.sin(teta) / GRAVITY
                       + torch.pow(torch.pow(v0, 2.0) * torch.pow(torch.sin(teta), 2.0) * GRAVITY * y0, 0.5) / GRAVITY,
                       p.t_exponent)
    d_term = torch.pow(h / (tech.f("actual_droplet_size") * 10.0), p.d_exponent)
    ed_cp = ap * et0_mm * d_term * t_term * amount_mm / p.amount_divisor
    ed_gun = (p.sprinkler_area_fraction * et0_mm * d_term * t_term * amount_mm / p.amount_divisor
              * p.sprinkler_hours / p.hours_divisor)
    ed_surf = et0_open_water_mm * wetted_pct * p.surface_hours / p.hours_divisor
    w_drip = torch.where(wetted_pct < p.drip_wetted_min, torch.full_like(wetted_pct, p.drip_wetted_min), wetted_pct)
    w_drip = torch.where(p.drip_wetted_max < w_drip, torch.full_like(w_drip, p.drip_wetted_max), w_drip)
    ed_drip = et0_open_water_mm * w_drip * tech.f("tirrig") / p.hours_divisor
    gun_like = (g == BIG_GUN) | (g == SOLID_SET) | (g == WHEEL_LINE)
    surface = (g == FLOOD) | (g == RILL) | (g == FURROW)
    ed = torch.zeros_like(ed_cp)
    ed = torch.where(g == CENTER_PIVOT, ed_cp, ed)
    ed = torch.where(gun_like, ed_gun, ed)
    ed = torch.where(surface, ed_surf, ed)
    return torch.where(g == DRIP, ed_drip, ed)


# ----------------------------------------------------------------------------- E13 Ro

class RoParams(NamedTuple):
    """Constants of calc_irrigation_runoff / calc_runoff_from_center_pivot."""
    pivot_length_m: float = 400.0        # calc_runoff_from_center_pivot :831
    pivot_segments: int = 4              # :832 (unrolled)
    pivot_lag_hours: float = 0.5         # :833
    pivot_rotation_factor: float = 350.0 # "350 is the length of irrigation system?" (:747)
    typical_evap_loss: float = 0.05      # assumed 5 % evaporative loss in the typical amount (:714)
    mad_half: float = 0.5                # first layer dries sooner (:718)
    max_loss_factor: float = 1.2         # :757
    min_loss_factor: float = 0.2         # :761


PI = 3.14159265358979323846              # CONST_PI


def infiltration_capacity_mm(s_philip: torch.Tensor, time_hr: torch.Tensor, ks_cm_hr: torch.Tensor) -> torch.Tensor:
    """calc_infiltration_capacity: Philip, 10 (S sqrt(t) + Ks t) [mm], t in hours, Ks in cm/h."""
    return (s_philip * torch.pow(time_hr, 0.5) + ks_cm_hr * time_hr) * 10.0


def center_pivot_runoff(amount_mm: torch.Tensor, sprinkler_diameter_m: torch.Tensor, ks_cm_hr: torch.Tensor,
                        s_philip: torch.Tensor, rotation_hr: torch.Tensor, p: RoParams = RoParams()) -> torch.Tensor:
    """calc_runoff_from_center_pivot (VIC_soil_hydrology.c:827-853): the pivot is cut into 4 rings; each ring's
    irrigation time follows from its speed, its runoff is amount minus Philip infiltration, and the rings are
    weighted by area. The reference divides the irrigation time by 3600 twice (it is 3600 times too short);
    kept for parity."""
    total_area = PI * p.pivot_length_m ** 2.0
    seg = p.pivot_length_m / float(p.pivot_segments)
    out = torch.zeros_like(amount_mm)
    for i in range(p.pivot_segments):                     # unrolled at trace time: fixed count, no tensor control flow
        r = i * seg + seg / 2.0
        perimeter = 2.0 * PI * r
        speed = perimeter / rotation_hr / 3600.0
        t = sprinkler_diameter_m / speed / 3600.0
        inf = infiltration_capacity_mm(s_philip, t + p.pivot_lag_hours, ks_cm_hr)
        seg_runoff = _cmax(amount_mm - inf, torch.zeros_like(amount_mm))
        out = out + seg_runoff * (PI * r ** 2.0) / total_area
    return out


def irrigation_runoff(tech: TechRow, amount_mm: torch.Tensor, top_moist_mm: torch.Tensor, mad: torch.Tensor,
                      ksat0_mm_day: torch.Tensor, s_max: torch.Tensor, porosity0: torch.Tensor,
                      wpwp0_mm: torch.Tensor, fc0_mm: torch.Tensor,
                      proration: float = 1.0, p: RoParams = RoParams(), mode: str = 'fixed',
                      top_thickness_mm=None) -> torch.Tensor:
    """E13 mechanistic Ro (eq. 4-5), calc_irrigation_runoff (VIC_soil_hydrology.c:655-772): surface runoff as the
    part of the amount the top layer cannot take at its Philip infiltration capacity, over the irrigation time
    of the technology, kept between 20 % of the presumed loss and 120 % of the total efficiency loss.
    `amount_mm` is the water reaching the surface (gross minus Ed). Drip, and a zero amount, give 0.
    `proration` is BASIN_PRORATION (1 in the full-irrigation build).

    NaN: where (wpwp + (FC - wpwp) MAD / 2) / 100 exceeds the porosity (the mm water content is divided by 100
    as if the top layer were 100 mm thick, so a thicker layer or a low porosity does it) the square root is of a
    negative number and Ro is NaN, in the reference too (identical mask, tested). The reference has no guard and
    its caller adds Ro straight into irrig_for_pot_infiltration (canopy_evap.c:632-644); the option is off in the
    shipped build (MECHANISTIC_RUNOFF FALSE, user_def.h:144), so no shipped run sees it. A coupled use of this
    option must mask non-finite Ro (the M4 scheme does that); this kernel keeps the reference's NaN for parity
    in mode 'exact'.

    mode 'fixed' (default): the volumetric water contents use the real top-layer thickness
    (`top_thickness_mm`, required), so the porosity test is physical and Ro is finite. Mode 'exact' divides by
    100 as the reference does (`top_thickness_mm` unused)."""
    if mode not in ('exact', 'fixed'):
        raise ValueError("mode must be 'exact' or 'fixed'")
    if mode == 'fixed':
        if top_thickness_mm is None:
            raise ValueError("mode 'fixed' needs top_thickness_mm (use mode='exact' for the reference's /100)")
        div = top_thickness_mm
    else:
        div = 100.0
    g = tech.general
    ks = ksat0_mm_day / 24.0 / 10.0
    wet = porosity0 > top_moist_mm / div
    s_philip = torch.where(wet, torch.pow(2.0 * (porosity0 - top_moist_mm / div) * ks * s_max, 0.5),
                           torch.zeros_like(ks))
    cap = tech.e("maximum_capacity")
    unif = tech.e("uniformity")
    pres = tech.e("presumed_efficiency")
    typical_amount = cap * proration / unif / 10.0 - p.typical_evap_loss * (cap * proration / 10.0)
    typical_wc = wpwp0_mm + (fc0_mm - wpwp0_mm) * mad * p.mad_half
    typical_s = torch.pow(2.0 * (porosity0 - typical_wc / div) * ks * s_max, 0.5)
    typical_time = torch.pow((-typical_s + torch.pow(torch.pow(typical_s, 2.0) + 4.0 * ks * typical_amount, 0.5))
                             / (2.0 * ks), 2.0)
    surface = (g == FLOOD) | (g == RILL) | (g == FURROW)
    sprinkler = (g == BIG_GUN) | (g == SOLID_SET) | (g == WHEEL_LINE)
    x = tech.f("x_dm_spk")
    single = tech.f("Q") / (PI * torch.pow(x / 2.0, 2.0)) * 1000.0
    overlap = typical_amount * 10.0 / pres / typical_time / single
    app_rate = single * overlap
    irrig_time = torch.zeros_like(amount_mm)
    irrig_time = torch.where(surface, typical_time, irrig_time)
    irrig_time = torch.where(sprinkler, amount_mm / app_rate + 0.0, irrig_time)
    rotation = typical_time * p.pivot_rotation_factor * PI * 2.0 / x
    ro_cp = center_pivot_runoff(amount_mm, x, ks, s_philip, rotation, p)
    ro_other = amount_mm - infiltration_capacity_mm(s_philip, irrig_time, ks)
    ro = torch.where(g == CENTER_PIVOT, ro_cp, ro_other)
    max_loss = p.max_loss_factor * (1.0 - unif * pres) * amount_mm * proration
    min_loss = (1.0 - pres) * amount_mm * proration * p.min_loss_factor
    ro = _cmin(_cmax(ro, min_loss), max_loss)
    return torch.where((g == DRIP) | _near(amount_mm, 0.0, 0.0000001), torch.zeros_like(ro), ro)


def mechanistic_partition(gross_mm: torch.Tensor, tech: TechRow, et0_mm: torch.Tensor, et0_open_water_mm: torch.Tensor,
                          top_moist_mm: torch.Tensor, mad: torch.Tensor, ksat0_mm_day: torch.Tensor,
                          s_max: torch.Tensor, porosity0: torch.Tensor, wpwp0_mm: torch.Tensor, fc0_mm: torch.Tensor,
                          bulk_density0: torch.Tensor, perennial: torch.Tensor, delta_t_sec=SEC_PER_DAY,
                          ed: EdParams = EdParams(), wet: WettedParams = WettedParams(),
                          ro: RoParams = RoParams(), proration: float = 1.0, mode: str = 'fixed',
                          top_thickness_mm=None) -> Partition:
    """The mechanistic option of the loss partition (MECHANISTIC_IRRIGATION with MECHANISTIC_RUNOFF): Ed from
    `droplet_evaporation` (scaled by delta_t / 86400, canopy_evap.c:545), Ro from `irrigation_runoff` on the
    water left after Ed. Deep percolation is 0 in this option (the reference leaves it to the soil)."""
    wp = wetted_percentage(tech, gross_mm, ksat0_mm_day, wpwp0_mm, fc0_mm, bulk_density0, perennial, wet)
    e = droplet_evaporation(tech, gross_mm, et0_mm, et0_open_water_mm, wp, ed) * (delta_t_sec / SEC_PER_DAY)
    aft = gross_mm - e
    r = torch.where(gross_mm > 0.0,
                    irrigation_runoff(tech, aft, top_moist_mm, mad, ksat0_mm_day, s_max, porosity0, wpwp0_mm, fc0_mm,
                                      proration, ro, mode, top_thickness_mm), torch.zeros_like(aft))
    dp = torch.zeros_like(aft)
    return Partition(gross_mm, e, aft, r, dp, aft - r - dp)


# ----------------------------------------------------------------------------- deficit-irrigation patterns

class PatternTable(NamedTuple):
    """Irrigation-pattern records (rotation code, simulation day index) -> (amount, proration rate), sorted by
    key, as `read_irrig_pattern` fills girrig_patt (read_croplib.c:179-225)."""
    rotation: torch.Tensor       # [n] long
    rec: torch.Tensor            # [n] long, days since the first simulation day
    amount_mm: torch.Tensor      # [n] full irrigation demand
    proration: torch.Tensor      # [n] 0-1, or -1 = use the basin-wide rate
    key: torch.Tensor            # [n] sorted
    rotations: torch.Tensor      # [m] sorted unique rotation codes


_REC_OFFSET = 1 << 22
_ROT_SHIFT = 1 << 24


def _pattern_key(rotation, rec):
    return rotation.long() * _ROT_SHIFT + (rec.long() + _REC_OFFSET)


def build_pattern_table(rotation, rec, amount_mm, proration, device="cpu", dtype=torch.float64) -> PatternTable:
    """Build the table from records; a repeated (rotation, rec) keeps the last, as the C map assignment does."""
    rot = torch.as_tensor(rotation, dtype=torch.long)
    rc = torch.as_tensor(rec, dtype=torch.long)
    am = torch.as_tensor(amount_mm, dtype=dtype)
    pr = torch.as_tensor(proration, dtype=dtype)
    key = _pattern_key(rot, rc)
    # last record wins: stable sort, then keep the last of each run
    order = torch.argsort(key, stable=True)
    key, rot, rc, am, pr = key[order], rot[order], rc[order], am[order], pr[order]
    last = torch.ones_like(key, dtype=torch.bool)
    if key.numel() > 1:
        last[:-1] = key[:-1] != key[1:]
    key, rot, rc, am, pr = key[last], rot[last], rc[last], am[last], pr[last]
    return PatternTable(rot.to(device), rc.to(device), am.to(device), pr.to(device), key.to(device),
                        torch.unique(rot).to(device))


def read_irrig_pattern(path: str, first_year: int, first_month: int, first_day: int,
                       device="cpu", dtype=torch.float64) -> PatternTable:
    """Read an irrigation pattern file (read_croplib.c:179-225): each line `year doy rotation amount [proration]`;
    4 columns mean proration -1 (basin-wide rate). The record index is the unsigned day count from the first
    simulation day (days_between(..., false), so a date before the start gives the mirrored positive index). Blank lines are skipped (the reference re-stores the previous record for them, which changes nothing)."""
    import datetime as _dt
    first = _dt.date(first_year, first_month, first_day)
    rot, rec, am, pr = [], [], [], []
    with open(path, "r", encoding="latin-1") as fh:
        for ln, line in enumerate(fh, 1):
            w = line.split()
            if not w:
                continue
            if len(w) not in (4, 5):
                raise ValueError(f"{path}:{ln}: wrong irrigation pattern file format")
            day = _dt.date(int(w[0]), 1, 1) + _dt.timedelta(days=int(w[1]) - 1)
            rec.append(abs((day - first).days))      # days_between(first, false) is unsigned: a record before the start maps onto the same index after it
            rot.append(int(w[2]))
            am.append(float(w[3]))
            pr.append(float(w[4]) if len(w) == 5 else -1.0)
    return build_pattern_table(rot, rec, am, pr, device, dtype)


def pattern_amount(table: PatternTable, rotation_code: torch.Tensor, rec: torch.Tensor,
                   basin_proration: float = 1.0) -> torch.Tensor:
    """amount_of_deficit_irrigation (VIC_soil_hydrology.c:190-204): the prescribed amount for a rotation on a day;
    -1 if the rotation has no pattern (the caller then irrigates fully), 0 if it has a pattern but no record for
    the day, else amount x proration (the basin-wide rate where the record's own proration is < 0).
    `basin_proration` is global_param.VCS.basin_wide_proration (BASIN_PRORATION, default 1)."""
    shape = rotation_code.shape
    rot = rotation_code.reshape(-1).long()
    rc = rec.reshape(-1).long()
    n = int(table.key.numel())
    if n == 0:
        return torch.full(shape, -1.0, dtype=table.amount_mm.dtype, device=table.amount_mm.device)
    k = _pattern_key(rot, rc)
    pos = torch.searchsorted(table.key, k).clamp(max=n - 1)
    hit = table.key[pos] == k
    rp = torch.searchsorted(table.rotations, rot).clamp(max=int(table.rotations.numel()) - 1)
    rot_in = table.rotations[rp] == rot
    pr = table.proration[pos]
    amt = table.amount_mm[pos] * torch.where(pr >= 0.0, pr, torch.full_like(pr, basin_proration))
    out = torch.where(rot_in, torch.where(hit, amt, torch.zeros_like(amt)), torch.full_like(amt, -1.0))
    return out.reshape(shape)
