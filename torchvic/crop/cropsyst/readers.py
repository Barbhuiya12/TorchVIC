"""Readers for the VIC-CropSyst inputs: .crp (crop), .mgt (management), .rot (rotation),
IRRIGPARAM (irrigation_parameter.txt), CROPCODELIB (cropcodelib.txt) and the VEGPARAM tile codes.

See docs/connectors/cropsyst.md (M5). Reference: <data root>/build/vic_cropsyst (VIC-CropSyst V3).
IRRIGPARAM follows VIC_CropSyst/read_irrigparam.c (cell, ntiles, then per tile: crop_code
[optional 0/1 full-irrigation flag] type); CROPCODELIB follows read_veglib.c:185 (read_cropcodelib).

Rules:
- Parsing is plain Python and runs once, at setup. Nothing here is called per step.
- An unknown section or key is an error (the schemas in schema.py list every known key).
- Outputs are torch tensors on the requested device: [n_file] for scalar keys, [n_file, K] with a
  validity mask for repeated sections ([adjustment:N], [tillage:N], [sowing:N] ...).
- Constants are arguments (dtype, missing-value fill), not literals in the data path.
"""
from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .schema import CRP_SCHEMA, MGT_SCHEMA, ROT_SCHEMA

_SEC = re.compile(r"^\[([^\]]+)\]\s*$")
_NUM = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
_INT = re.compile(r"^\s*([+-]?\d+)")
_IDX = re.compile(r"^(.*):(\d+)$")

# Missing-value fill for tensors (a parameter, not a magic number in the data path).
MISSING_FLOAT = float("nan")
MISSING_INT = -1


class InputError(ValueError):
    """Unknown key/section, malformed line or inconsistent count in a CropSyst input file."""


# ----------------------------------------------------------------------------- generic INI

@dataclass
class IniFile:
    """One parsed file: sections[name] = list of {key: python value}; repeated sections keep order."""
    path: str
    sections: "OrderedDict[str, List[dict]]" = field(default_factory=OrderedDict)
    repeated: set = field(default_factory=set)

    def get(self, section: str, key: str, index: int = 0):
        return self.sections[section][index][key]


def _convert(kind: str, raw: str, where: str):
    raw = raw.strip()
    if kind == "t":
        return raw
    if kind == "e":
        return raw.split()[0] if raw else ""
    if kind == "B":                     # lenient bool (CORN expect_bool): true/1 are true, else false
        return raw.split()[0].lower() in ("true", "1") if raw else False
    if kind == "b":
        if raw not in ("true", "false"):
            raise InputError(f"{where}: expected true/false, got {raw!r}")
        return raw == "true"
    m = (_INT if kind == "i" else _NUM).match(raw)
    if not m:
        raise InputError(f"{where}: expected a number, got {raw!r}")
    return int(m.group(1)) if kind == "i" else float(m.group(1))


def parse_ini(path: str, schema: dict) -> IniFile:
    """Strict parse of a CropSyst INI file against `schema` {section: {key: type}}."""
    with open(path, "r", encoding="latin-1") as fh:
        lines = fh.read().replace("\r", "").split("\n")
    out = IniFile(path=path)
    cur: Optional[dict] = None
    cur_name = None
    seen_idx: Dict[str, int] = {}
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f"{path}:{n}"
        m = _SEC.match(line)
        if m:
            head = m.group(1)
            mi = _IDX.match(head)
            name, idx = (mi.group(1), int(mi.group(2))) if mi else (head, None)
            if name not in schema:
                raise InputError(f"{where}: unknown section [{head}]")
            if idx is None:
                if name in out.sections:
                    raise InputError(f"{where}: duplicate section [{name}]")
            else:
                if idx != seen_idx.get(name, 0) + 1:
                    raise InputError(f"{where}: section [{head}] out of order")
                seen_idx[name] = idx
                out.repeated.add(name)
            cur = {}
            cur_name = name
            out.sections.setdefault(name, []).append(cur)
            continue
        if cur is None or "=" not in line:
            raise InputError(f"{where}: cannot parse line {line!r}")
        key, val = line.split("=", 1)
        kind = schema[cur_name].get(key)
        if kind is None:
            raise InputError(f"{where}: unknown key {key!r} in section [{cur_name}]")
        if key in cur:
            raise InputError(f"{where}: duplicate key {key!r} in [{cur_name}]")
        cur[key] = _convert(kind, val, where)
    # the "<name>:count" keys must agree with the number of [<name>:N] sections
    for sec, items in out.sections.items():
        for key, val in items[0].items():
            if key.endswith(":count"):
                base = key[: -len(":count")]
                have = len(out.sections.get(base, [])) if base in out.repeated else 0
                if have != val:
                    raise InputError(f"{path}: [{sec}] {key}={val} but {have} [{base}:N] sections")
    return out


@dataclass
class IniTable:
    """Batched tensors for a list of files of one kind.

    params[f"{section}.{key}"]: [n] (single section) or [n, K] (repeated section, K = max count).
    Floats fill missing entries with `MISSING_FLOAT`, ints/enums with `MISSING_INT`, bools False.
    valid[section]: [n] / [n, K] bool, True where that section instance exists in the file.
    vocab[f"{section}.{key}"]: enum code -> string (codes are indices into the list).
    text[f"{section}.{key}"]: host-side strings (descriptions, paths) per file, never on device.
    """
    paths: List[str]
    params: Dict[str, torch.Tensor]
    valid: Dict[str, torch.Tensor]
    vocab: Dict[str, List[str]]
    text: Dict[str, list]

    @property
    def n(self) -> int:
        return len(self.paths)

    def code(self, name: str, word: str) -> int:
        return self.vocab[name].index(word)


def build_table(files: Sequence[IniFile], schema: dict, device="cpu",
                dtype=torch.float64) -> IniTable:
    n = len(files)
    params, valid, vocab, text = {}, {}, {}, {}
    sections = OrderedDict()
    for f in files:
        for s in f.sections:
            sections.setdefault(s, False)
            if s in f.repeated:
                sections[s] = True
    for s, repeated in sections.items():
        K = max(len(f.sections.get(s, [])) for f in files) if repeated else 1
        shape = (n, K) if repeated else (n,)
        v = torch.zeros(shape, dtype=torch.bool)
        for i, f in enumerate(files):
            for j in range(len(f.sections.get(s, []))):
                if repeated:
                    v[i, j] = True
                else:
                    v[i] = True
        valid[s] = v.to(device)
        for key, kind in schema[s].items():
            name = f"{s}.{key}"
            vals = [[(it.get(key) if key in it else None) for it in f.sections.get(s, [])]
                    for f in files]
            if kind == "t":
                text[name] = vals if repeated else [r[0] if r else None for r in vals]
                continue
            if all(x is None for r in vals for x in r):
                continue  # key never present in this batch
            if kind == "e":
                words = sorted({x for r in vals for x in r if x is not None})
                vocab[name] = words
                conv = {w: i for i, w in enumerate(words)}
                fill, dt = MISSING_INT, torch.long
                cvt = lambda x: fill if x is None else conv[x]
            elif kind == "i":
                fill, dt, cvt = MISSING_INT, torch.long, lambda x: MISSING_INT if x is None else x
            elif kind in ("b", "B"):
                fill, dt, cvt = False, torch.bool, lambda x: False if x is None else x
            else:
                fill, dt, cvt = MISSING_FLOAT, dtype, lambda x: MISSING_FLOAT if x is None else x
            t = torch.full(shape, fill, dtype=dt)
            for i, r in enumerate(vals):
                for j, x in enumerate(r):
                    if repeated:
                        t[i, j] = cvt(x)
                    else:
                        t[i] = cvt(x)
            params[name] = t.to(device)
    return IniTable([f.path for f in files], params, valid, vocab, text)


# ----------------------------------------------------------------------------- .crp / .mgt / .rot

def read_crp(path: str) -> IniFile:
    return parse_ini(path, CRP_SCHEMA)


def read_mgt(path: str) -> IniFile:
    return parse_ini(path, MGT_SCHEMA)


def read_rot(path: str) -> IniFile:
    return parse_ini(path, ROT_SCHEMA)


def crop_table(paths: Sequence[str], device="cpu", dtype=torch.float64) -> IniTable:
    """Batched .crp parameters over crops: params['transpiration.max_water_uptake'] is [n_crop]."""
    return build_table([read_crp(p) for p in paths], CRP_SCHEMA, device, dtype)


def mgt_table(paths: Sequence[str], device="cpu", dtype=torch.float64) -> IniTable:
    """Batched .mgt operations: params['harvest.event_offset'] is [n_mgt, K_harvest] with valid['harvest']."""
    return build_table([read_mgt(p) for p in paths], MGT_SCHEMA, device, dtype)


def rot_table(paths: Sequence[str], device="cpu", dtype=torch.float64) -> IniTable:
    """Batched .rot sowing events: params['sowing.event_date'] is [n_rot, K_sowing]."""
    return build_table([read_rot(p) for p in paths], ROT_SCHEMA, device, dtype)


def split_date32(value: int) -> Tuple[int, int]:
    """CropSyst date32 (year*1000 + day of year; the text after the number is only a label) -> (year offset, doy)."""
    return value // 1000, value % 1000


# ----------------------------------------------------------------------------- IRRIGPARAM

# irrigation_lib.h:95-163 (enum Irrigation_Type) and read_irrigparam.c get_irrigation_type_from_str
IRRIGATION_TYPE_CODES = OrderedDict([
    ("NO_IRRIGATION", 0),
    ("IrrigTP_CP_impact_14VH_RainBird", 1), ("IrrigTP_CP_impact_M20VH_PM_RainBird", 2),
    ("IrrigTP_CP_impact_65PJ_RainBird", 3), ("IrrigTP_CP_impact_30FH_30FWH_RainBird", 4),
    ("IrrigTP_CP_impact_L36H_L36AH_RainBird_1", 5), ("IrrigTP_CP_impact_L36H_L36AH_RainBird_2", 6),
    ("IrrigTP_CP_impact_85EHD_RainBird_1", 7), ("IrrigTP_CP_impact_85EHD_RainBird_2", 8),
    ("IrrigTP_CP_impact_85EHD_RainBird_3", 9), ("IrrigTP_CP_impact_85EHD_LA_RainBird_1", 10),
    ("IrrigTP_CP_impact_85EHD_LA_RainBird_2", 11), ("IrrigTP_CP_impact_85EHD_LA_RainBird_3", 12),
    ("IrrigTP_CP_spray_S3000_Nelson", 21), ("IrrigTP_CP_spray_O3000_Nelson", 22),
    ("IrrigTP_CP_spray_R3000_Nelson_1", 23), ("IrrigTP_CP_spray_R3000_Nelson_2", 24),
    ("IrrigTP_CP_spray_A3000_Nelson_1", 25), ("IrrigTP_CP_spray_A3000_Nelson_2", 26),
    ("DEFAULT_CENTER_PIVOT", 30),
    ("IrrigTP_Big_Gun_75TR_Nelson_1", 31), ("IrrigTP_Big_Gun_75TR_Nelson_2", 32),
    ("IrrigTP_Big_Gun_75TR_Nelson_3", 33), ("IrrigTP_Big_Gun_150TB_Nelson_1", 34),
    ("IrrigTP_Big_Gun_150TB_Nelson_2", 35), ("IrrigTP_Big_Gun_150TB_Nelson_3", 36),
    ("IrrigTP_Big_Gun_200TB_Nelson_1", 37), ("IrrigTP_Big_Gun_200TB_Nelson_2", 38),
    ("IrrigTP_Big_Gun_200TB_Nelson_3", 39), ("DEFAULT_BIG_GUN", 40),
    ("IrrigTP_Solid_set_R5_POP_UP_Nelson", 41), ("IrrigTP_Solid_set_R2000WF_6_Nelson_1", 42),
    ("IrrigTP_Solid_set_R2000WF_6_Nelson_2", 43), ("IrrigTP_Solid_set_R2000WF_6_Nelson_3", 44),
    ("IrrigTP_Solid_set_R33_Nelson", 45), ("DEFAULT_SOLID_SET", 50),
    ("IrrigTP_Moving_wheel_R2000WF_6_Nelson_1", 51), ("IrrigTP_Moving_wheel_R2000WF_6_Nelson_2", 52),
    ("IrrigTP_Moving_wheel_R2000WF_6_Nelson_3", 53), ("DEFAULT_MOVING_WHEEL", 60),
    ("IrrigTP_drip_0_0", 61), ("IrrigTP_Sub_surf_drip_0_0", 62), ("IrrigTP_Sub_surf_drip_perfect", 63),
    ("IrrigTP_Sub_surf_drip_perfect_eliminate_top", 64), ("DEFAULT_DRIP", 70),
    ("IrrigTP_flood_0_0", 71), ("DEFAULT_FLOOD", 71), ("IrrigTP_rill_0_0", 72), ("DEFAULT_RILL", 72),
    ("IrrigTP_furrow_0_0", 73), ("DEFAULT_FURROW", 73), ("DEFAULT_SPRINKLER", 74),
    ("DEFAULT_MICRO_SPRINKLER", 75),
])
# short names the shipped file uses (read_irrigparam.c:70-80)
IRRIGATION_TYPE_ALIASES = {
    "CENTER_PIVOT": "DEFAULT_CENTER_PIVOT", "SPRINKLER": "DEFAULT_SPRINKLER",
    "BIG_GUN": "DEFAULT_BIG_GUN", "SOLID_SET": "DEFAULT_SOLID_SET",
    "MOVING_WHEEL": "DEFAULT_MOVING_WHEEL", "WHEEL_LINE": "DEFAULT_MOVING_WHEEL",
    "DRIP": "DEFAULT_DRIP", "FLOOD": "DEFAULT_FLOOD", "RILL": "DEFAULT_RILL",
    "FURROW": "DEFAULT_FURROW",
}


def irrigation_type_code(name: str) -> int:
    name = IRRIGATION_TYPE_ALIASES.get(name, name)
    if name not in IRRIGATION_TYPE_CODES:
        raise InputError(f"unknown irrigation type {name!r}")
    return IRRIGATION_TYPE_CODES[name]


@dataclass
class IrrigParam:
    """Irrigation technology per (cell, crop code), in CSR form.

    cell_id [n_cell] (file order); offset [n_cell+1]; crop_code, full_irrigation (bool),
    fraction_full [n_entry] float, type_code [n_entry] long (irrigation_lib.h codes).
    A cell or crop code with no record means no irrigation (read_irrigparam.c).
    """
    cell_id: torch.Tensor
    offset: torch.Tensor
    crop_code: torch.Tensor
    full_irrigation: torch.Tensor
    fraction_full: torch.Tensor
    type_code: torch.Tensor

    def lookup(self, cells: torch.Tensor, crop_codes: torch.Tensor) -> torch.Tensor:
        """Technology code for (cell, crop_code) pairs (same shape, any device); 0 = none. Vectorised."""
        dev = self.cell_id.device
        cells, crop_codes = cells.to(dev), crop_codes.to(dev)
        order = torch.argsort(self.cell_id)
        sorted_ids = self.cell_id[order]
        pos = torch.searchsorted(sorted_ids, cells.reshape(-1)).clamp(max=len(sorted_ids) - 1)
        hit = sorted_ids[pos] == cells.reshape(-1)
        row = order[pos]
        lo, hi = self.offset[row], self.offset[row + 1]
        maxn = int((self.offset[1:] - self.offset[:-1]).max())
        j = lo[:, None] + torch.arange(maxn, device=dev)[None, :]
        inr = (j < hi[:, None]) & hit[:, None]
        jj = j.clamp(max=len(self.crop_code) - 1)
        match = inr & (self.crop_code[jj] == crop_codes.reshape(-1)[:, None])
        first = match.to(torch.int8).argmax(dim=1)
        found = match.any(dim=1)
        pick = jj.gather(1, first[:, None])[:, 0]
        out = torch.where(found, self.type_code[pick], torch.zeros_like(pick))
        return out.reshape(cells.shape)


def read_irrigparam(path: str, device="cpu", dtype=torch.float64) -> IrrigParam:
    cell_id, offset = [], [0]
    code, full, frac, typ = [], [], [], []
    with open(path, "r", encoding="latin-1") as fh:
        lines = [ln for ln in fh.read().replace("\r", "").split("\n")]
    i, n = 0, len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        where = f"{path}:{i + 1}"
        head = lines[i].split()
        if len(head) != 2:
            raise InputError(f"{where}: expected '<cell> <ntiles>', got {lines[i]!r}")
        cell, nt = int(head[0]), int(head[1])
        if nt < 0:
            raise InputError(f"{where}: negative tile count")
        if i + nt >= n:
            raise InputError(f"{where}: unexpected end of file for cell {cell}")
        for k in range(nt):
            tok = lines[i + 1 + k].split()
            if len(tok) not in (2, 3):
                raise InputError(f"{path}:{i + 2 + k}: expected '<code> [0|1] <type>'")
            code.append(int(tok[0]))
            if len(tok) == 3:
                if tok[1] not in ("0", "1"):
                    raise InputError(f"{path}:{i + 2 + k}: full-irrigation flag must be 0 or 1")
                full.append(tok[1] == "1")
                frac.append(float(tok[1]))
            else:
                full.append(True)
                frac.append(1.0)
            typ.append(irrigation_type_code(tok[-1]))
        cell_id.append(cell)
        offset.append(len(code))
        i += nt + 1
    t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
    return IrrigParam(t(cell_id, torch.long), t(offset, torch.long), t(code, torch.long),
                      t(full, torch.bool), t(frac, dtype), t(typ, torch.long))


# ----------------------------------------------------------------------------- CROPCODELIB

@dataclass
class CropCodeLib:
    """cropcodelib.txt: rows (code, name); several names may share a code. codes [n] long."""
    codes: torch.Tensor
    names: List[str]

    def names_for(self, code: int) -> List[str]:
        return [nm for c, nm in zip(self.codes.tolist(), self.names) if c == code]


def read_cropcodelib(path: str, device="cpu") -> CropCodeLib:
    codes, names = [], []
    with open(path, "r", encoding="latin-1") as fh:
        for n, line in enumerate(fh.read().replace("\r", "").split("\n"), 1):
            if not line.strip() or line.startswith("#"):
                continue  # read_veglib.c:191 counts only lines that start with a digit
            m = re.match(r"^(\d+)[ \t]+(.+?)\s*$", line)
            if not m:
                raise InputError(f"{path}:{n}: expected '<code> <name>', got {line!r}")
            codes.append(int(m.group(1)))
            names.append(m.group(2))
    return CropCodeLib(torch.tensor(codes, dtype=torch.long, device=device), names)


# ----------------------------------------------------------------------------- VEGPARAM tile codes

def read_vegparam_codes(path: str) -> Tuple[List[int], List[List[int]]]:
    """VEGPARAM '<cell> <ntiles>' then ntiles rows '<veg code> <Cv> <root params...>': returns cells and the codes."""
    cells, codes = [], []
    with open(path, "r", encoding="latin-1") as fh:
        lines = [ln for ln in fh.read().replace("\r", "").split("\n") if ln.strip()]
    i = 0
    while i < len(lines):
        head = lines[i].split()
        if len(head) != 2:
            raise InputError(f"{path}: expected '<cell> <ntiles>', got {lines[i]!r}")
        nt = int(head[1])
        cells.append(int(head[0]))
        codes.append([int(lines[i + 1 + k].split()[0]) for k in range(nt)])
        i += nt + 1
    return cells, codes
