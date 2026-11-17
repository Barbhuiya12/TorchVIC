"""M5 rotation and sowing calendar tables (docs/connectors/cropsyst.md).

VIC-CropSyst expands every rotation code R with Y years into Y tiles with runtime code R*100 + phase
(read_vegparam.c:477-510, MULTIPLIER_FOR_SOWING_CODE=100; call_crop_model.c:528 DecomposeVegclassCode).
Tile `phase` is in rotation cycle year (year_index + phase) % Y. Checked against the reference runs
(328617 rot 3003, 308200 rot 3002: Crop_name changes in vic_crop_daily.csv per tile).

Everything here is built once at setup from the rotation, crop and management files. The per-year
tables are plain tensors [tile, year, slot]; the daily engine advances by indexing them, nothing is parsed
and no Python runs per step.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from .readers import (IniTable, InputError, build_table, crop_table, mgt_table, read_rot,
                      split_date32)
from .schema import ROT_SCHEMA

# read_vegparam.c / vicNl_def.h:215 (a parameter here, not a literal in the code below)
SOWING_CODE_MULTIPLIER = 100
# vicNl_def.h:210-216: codes above VEG_CLASS_CODE_START_SPECIFIC_SOWING carry the cycle phase
START_SPECIFIC_SOWING = (3000 + 1) * SOWING_CODE_MULTIPLIER
NONIRRIGATED_OFFSET = 10000      # VIC_land_unit_simulation.cpp:795, 804 (non-irrigated crop codes)
NO_SOW = -1  # sentinel in the integer tables


def decompose_code(code: int):
    """DecomposeVegclassCode (call_crop_model.c:526-538) and the land unit's rotation code
    (VIC_land_unit_simulation.cpp:792-795: codes >= VEG_CLASS_CODE_START_SPECIFIC_SOWING carry a
    phase, codes > 10000 after that are non-irrigated): (rotation or crop code, cycle phase)."""
    code = int(code)
    if code >= START_SPECIFIC_SOWING:
        base, phase = divmod(code, SOWING_CODE_MULTIPLIER)
    else:
        base, phase = code, 0
    if base > NONIRRIGATED_OFFSET:
        base -= NONIRRIGATED_OFFSET
    return base, phase


def established_files(db_dir: str, code: int):
    """A code without a rotation file is an established crop (orchard, VIC_land_unit_simulation.cpp:
    800-840): Crop/Code/<code>.CS_crop and the optional Management/Code/<code>.mgt. Returns
    (crop path, management path or None), or None when there is no such crop file."""
    crop = os.path.join(db_dir, "Crop", "Code", f"{code}.CS_crop")
    if not os.path.exists(crop):
        return None
    mgt = os.path.join(db_dir, "Management", "Code", f"{code}.mgt")
    return crop, (mgt if os.path.exists(mgt) else None)


def is_rotation(db_dir: str, code: int) -> bool:
    return os.path.exists(os.path.join(db_dir, "Rotation", "Code", f"{code}.CS_rotation"))


@dataclass
class RotationSet:
    """Rotations (one per code) with their crop and management tables.

    rot: IniTable of the .rot files (params['sowing.*'] are [n_rot, S]); crops, mgts: IniTables of the unique
    .crp/.mgt files they reference. Index tensors are [n_rot, S] with NO_SOW where the slot does not exist.
    """
    codes: List[int]
    rot: IniTable
    crops: IniTable
    mgts: IniTable
    years: torch.Tensor          # [n_rot] cycle length in years
    n_sow: torch.Tensor          # [n_rot]
    sow_valid: torch.Tensor      # [n_rot, S] bool
    sow_year: torch.Tensor       # [n_rot, S] cycle year of the sowing (date32 // 1000)
    sow_doy: torch.Tensor        # [n_rot, S] day of year of the sowing (date32 % 1000)
    crop_idx: torch.Tensor       # [n_rot, S] row in `crops`
    mgt_idx: torch.Tensor        # [n_rot, S] row in `mgts`
    sync_code: torch.Tensor      # [n_rot, S] code into rot.vocab['sowing.event_synchronization']
    once: torch.Tensor = None    # [n_rot] bool, established crop: sown once, on the first day

    def index_of(self, code: int) -> int:
        return self.codes.index(code)


def _resolve(rot_path: str, rel: str) -> str:
    base = os.path.dirname(os.path.realpath(rot_path))
    p = os.path.normpath(os.path.join(base, rel.replace("\\", "/")))
    if not os.path.exists(p):  # the shipped tree mixes Crop/Name and Crop/name
        d, f = os.path.split(p)
        up = os.path.dirname(d)
        for alt in os.listdir(up) if os.path.isdir(up) else []:
            if alt.lower() == os.path.basename(d).lower() and os.path.exists(os.path.join(up, alt, f)):
                return os.path.join(up, alt, f)
        raise InputError(f"{rot_path}: referenced file {rel!r} not found")
    return p


def _established_rot(code, crop, mgt, start_doy):
    """A one-sowing rotation standing for an established crop: sown on the simulation start day
    (land_unit_sim.cpp:1026-1029, start_date)."""
    from .readers import IniFile
    f = IniFile(path=f"<established {code}>")
    f.sections["rotation"] = [{"years": 1, "sowing:count": 1}]
    f.sections["sowing"] = [{"event_date": int(start_doy), "crop": crop, "management": mgt,
                             "event_synchronization": "actual_date"}]
    f.repeated.add("sowing")
    return f


def load_rotations(db_dir: str, codes: Sequence[int], device="cpu",
                   dtype=torch.float64, start_doy: int = 1) -> RotationSet:
    """Load rotation `codes` from <db_dir>/Rotation/Code/<code>.CS_rotation (the shipped symlinks).
    A code without a rotation file but with Crop/Code/<code>.CS_crop is an established crop
    (orchard): one tile sown once, on the first simulation day (day of year `start_doy`)."""
    paths, files, once = [], [], []
    for c in codes:
        p = os.path.join(db_dir, "Rotation", "Code", f"{c}.CS_rotation")
        est = None if os.path.exists(p) else established_files(db_dir, c)
        if est is None:
            paths.append(p)
            files.append(read_rot(p))
            once.append(False)
            continue
        if est[1] is None:
            raise InputError(f"cropsyst: established crop {c} has no Management/Code/{c}.mgt (a crop "
                             "without management is not ported)")
        paths.append(os.path.join(db_dir, "Crop", "Code", f"{c}.CS_crop"))
        files.append(_established_rot(c, os.path.realpath(est[0]), os.path.realpath(est[1]), start_doy))
        once.append(True)
    rot = build_table(files, ROT_SCHEMA, device, dtype)
    crop_paths, mgt_paths = [], []
    crop_rows, mgt_rows = [], []
    for p, f in zip(paths, files):
        cr, mr = [], []
        for sw in f.sections["sowing"]:
            for rel, lst, rows in ((sw["crop"], crop_paths, cr), (sw["management"], mgt_paths, mr)):
                full = rel if os.path.isabs(rel) else os.path.realpath(_resolve(p, rel))
                if full not in lst:
                    lst.append(full)
                rows.append(lst.index(full))
        crop_rows.append(cr)
        mgt_rows.append(mr)
    S = rot.params["sowing.event_date"].shape[1]
    n = len(codes)

    def pad(rows):
        t = torch.full((n, S), NO_SOW, dtype=torch.long)
        for i, r in enumerate(rows):
            t[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return t.to(device)

    valid = rot.valid["sowing"]
    date = rot.params["sowing.event_date"]
    years = torch.tensor([f.get("rotation", "years") for f in files], dtype=torch.long, device=device)
    n_sow = valid.sum(dim=1)
    sow_year = torch.where(valid, date // 1000, torch.full_like(date, NO_SOW))
    sow_doy = torch.where(valid, date % 1000, torch.full_like(date, NO_SOW))
    if bool((sow_year >= years[:, None]).logical_and(valid).any()):
        raise InputError("a sowing year offset is outside its rotation cycle")
    return RotationSet(list(codes), rot, crop_table(crop_paths, device, dtype),
                       mgt_table(mgt_paths, device, dtype), years, n_sow, valid, sow_year, sow_doy,
                       pad(crop_rows), pad(mgt_rows), rot.params["sowing.event_synchronization"],
                       torch.tensor(once, dtype=torch.bool, device=device))


@dataclass
class SowingCalendar:
    """Per-tile, per-year sowing table. tile = one VIC-CropSyst expanded tile (cell, rotation, phase).

    sow_doy, crop_idx, mgt_idx: [n_tile, n_year, S] long, NO_SOW where nothing is sown in that year/slot.
    cycle_year: [n_tile, n_year] the rotation cycle year the tile is in. runtime_code: [n_tile] R*100+phase.
    """
    sow_doy: torch.Tensor
    crop_idx: torch.Tensor
    mgt_idx: torch.Tensor
    cycle_year: torch.Tensor
    runtime_code: torch.Tensor
    first_year: int

    def at_year(self, year: int):
        """Day-step access: the year's slice (an index, no parsing)."""
        i = year - self.first_year
        return self.sow_doy[:, i], self.crop_idx[:, i], self.mgt_idx[:, i]


def build_calendar(rs: RotationSet, tile_rot: torch.Tensor, tile_phase: torch.Tensor,
                   first_year: int, n_years: int) -> SowingCalendar:
    """Vectorised over tiles and years. tile_rot [n_tile]: row in `rs`; tile_phase [n_tile]: expansion phase."""
    dev = rs.years.device
    tile_rot, tile_phase = tile_rot.to(dev), tile_phase.to(dev)
    Y = rs.years[tile_rot]
    if bool((tile_phase >= Y).any()):
        raise InputError("tile phase exceeds the rotation length")
    yidx = torch.arange(n_years, device=dev)
    cyc = (yidx[None, :] + tile_phase[:, None]) % Y[:, None]                      # [T, n_year]
    sy = rs.sow_year[tile_rot]                                                    # [T, S]
    hit = rs.sow_valid[tile_rot][:, None, :] & (sy[:, None, :] == cyc[:, :, None])  # [T, n_year, S]
    if rs.once is not None:                                                     # established crops
        hit = hit & (~rs.once[tile_rot][:, None, None] | (yidx[None, :, None] == 0))
    fill = torch.full_like(hit, NO_SOW, dtype=torch.long)
    doy = torch.where(hit, rs.sow_doy[tile_rot][:, None, :].expand_as(fill), fill)
    cr = torch.where(hit, rs.crop_idx[tile_rot][:, None, :].expand_as(fill), fill)
    mg = torch.where(hit, rs.mgt_idx[tile_rot][:, None, :].expand_as(fill), fill)
    code = torch.tensor(rs.codes, dtype=torch.long, device=dev)[tile_rot] * SOWING_CODE_MULTIPLIER + tile_phase
    return SowingCalendar(doy, cr, mg, cyc, code, first_year)


def expand_tiles(rs: RotationSet, cell_rot_codes: Sequence[int]):
    """One tile per cycle year for each cell's rotation code (read_vegparam.c expansion).

    Returns (tile_cell [n_tile], tile_rot [n_tile], tile_phase [n_tile]) as long tensors.
    """
    cells, rots, phases = [], [], []
    for c, code in enumerate(cell_rot_codes):
        r = rs.index_of(code)
        for ph in range(int(rs.years[r])):
            cells.append(c), rots.append(r), phases.append(ph)
    t = lambda x: torch.tensor(x, dtype=torch.long, device=rs.years.device)
    return t(cells), t(rots), t(phases)
