"""Reader for the dump of the reference N-on runs.

The dump binary <data root>/build/vic_cropsyst/crb_N/vic_cropsyst_i6bdump.exe (sources in
crb_N/dump_patch/; dump build only: the NH4 layer array is zero-initialised and print hooks are
added; the crop csv of the three replay cases equals the original reference byte for byte) writes
VIC_output/i6b_dump.txt, one record per tile and day at each step of the land-unit day:

  H  once per land unit: denitrification scale, rate_max, alpha, (unused), pH(1), transformation
     layers; per layer: thickness, bulk density g/cm3, sand %, clay %, saturation, field capacity,
     wilting point, NH4 pH function, depth
  P  VIC_land_unit_process_day entry: per layer VIC moist [mm], ice, T; N
  S  after the organic-matter start_day: N, residue pools
  Q  an operation: type code, NRCS description [I: NH4 source, method, volatilization
     calculated, volatilization %, NO3 kg/ha, NH4 kg/ha, injection depth cm, N_soil_critical,
     sampling depth]
  O  after the operations: N, residue pools, X sand/clay, D disturbance factor/rating
  M  after the organic-matter process_day: N, A per layer: CO2-C loss, residue N mineralization,
     plant available water, OM moisture function, decomposition temperature function (daily
     mean), nitrification temperature function (hour 0), immobilization demand
  N  crop, before Crop_nitrogen_V5::update_limited_growth: N, V, W and per layer root lengths,
     root fractions, PAW
  n  crop, after the limited potential transpiration: N, V
  C  after the crop's process_day: N
  T  after the chemical transformations: N, X denitrified/nitrified N per layer
  B  after the residue balance: N, R flat and stubble contact fractions
  F  distribute_any_crop_biomass_fate: N, pools, Y
  E  after end_day: N, X, D, C (crop N), all pools (older format; the residue pools are kept)
N = NO3-N and NH4-N per layer (elemental kg/m2). Hook pools: type, position, layer, cycling,
abiotic-environment layer, mass, C fraction, C/N, decomposition constant, detrition constant,
contact fraction, area per mass, potential and actual C decomposition (-1 when none).
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict

L = 17
V_NAMES = ["attainable", "canopy_n", "root_n", "canopy_conc", "root_conc", "store", "limited_growth",
           "limited_pot", "stress", "stress32", "uptake", "fixation", "n_max", "n_crit", "n_min",
           "max_uptake_per_length", "canopy_biomass_vital", "tue_co2_adjusted"]


def _f(x):
    return float.fromhex(x)


def _take_n(tok, i):
    assert tok[i] == "N", tok[i]
    v = [_f(x) for x in tok[i + 1:i + 1 + 2 * L]]
    return (v[0::2], v[1::2]), i + 1 + 2 * L


def _take_pools(tok, i):
    n = int(tok[i + 1])
    j = i + 2
    pools = []
    for _ in range(n):
        a = tok[j:j + 14]
        pools.append(dict(type=int(a[0]), position=int(a[1]), layer=int(a[2]), cycling=int(a[3]), env=int(a[4]),
                          mass=_f(a[5]), cfrac=_f(a[6]), cn=_f(a[7]), k=_f(a[8]), kdet=_f(a[9]),
                          contact=_f(a[10]), apm=_f(a[11]), pot=_f(a[12]), act=_f(a[13])))
        j += 14
    return pools, j


def read_dump(path):
    """Returns (header by tile, records): records[(year, doy, tile)][tag] = dict (Q: list)."""
    head = {}
    rec = defaultdict(dict)
    for line in open(path):
        if line.startswith("Q "):
            q = line.split('"')
            a = q[0].split()
            key = (int(a[1]), int(a[2]), int(a[3]))
            item = dict(type=int(a[4]), desc=q[1])
            rest = q[2].split()
            if rest and rest[0] == "I":
                v = rest[1:]
                item.update(nh4_source=int(v[0]), nh4_method=int(v[1]), vol_calculated=int(v[2]),
                            volatilization=_f(v[3]), no3_kg_ha=_f(v[4]), nh4_kg_ha=_f(v[5]),
                            injection_depth_cm=_f(v[6]))
            rec[key].setdefault("Q", []).append(item)
            continue
        tok = line.split()
        tag = tok[0]
        key = (int(tok[1]), int(tok[2]), int(tok[3]))
        if tag == "H":
            v = tok[4:]
            h = dict(scale=_f(v[0]), rate_max=_f(v[1]), alpha=_f(v[2]), ph=_f(v[4]), trans_layers=int(v[5]))
            per = [_f(x) for x in v[6:]]
            for k, nm in enumerate(["thickness", "bulk_density", "sand", "clay", "sat", "fc", "pwp",
                                    "ph_function", "depth"]):
                h[nm] = per[k::9]
            head[key[2]] = h
            continue
        d = {}
        i = 4
        if tag == "P":
            v = [_f(x) for x in tok[4:4 + 3 * L]]
            d.update(moist=v[0::3], ice=v[1::3], T=v[2::3])
            i = 4 + 3 * L
        if tag == "E":
            (d["no3"], d["nh4"]), i = _take_n(tok, i)
            i += 1 + 2 * L                     # X
            i += 1 + L                         # D
            if tok[i] == "C0":
                i += 1
            else:
                d["crop"] = [_f(x) for x in tok[i + 1:i + 17]]
                i += 17
            n = int(tok[i + 1])
            j = i + 2
            pools = []
            for _ in range(n):
                a = tok[j:j + 14]
                if int(a[0]) <= 1:
                    pools.append(dict(type=int(a[0]), position=int(a[1]), layer=int(a[2]), cycling=int(a[3]),
                                      mass=_f(a[4]), cfrac=_f(a[5]), cn=_f(a[6]), k=_f(a[7]), kdet=_f(a[8]),
                                      contact=_f(a[9]), apm=_f(a[10])))
                j += 14
            d["pools"] = pools
            rec[key]["E"] = d
            continue
        if i < len(tok) and tok[i] == "N":
            (d["no3"], d["nh4"]), i = _take_n(tok, i)
        if i < len(tok) and tok[i] == "O":
            d["pools"], i = _take_pools(tok, i)
        while i < len(tok):
            t = tok[i]
            if t in ("X", "D"):
                v = [_f(x) for x in tok[i + 1:i + 1 + 2 * L]]
                d["x" if t == "X" else "dist"] = (v[0::2], v[1::2])
                i += 1 + 2 * L
            elif t == "A":
                v = [_f(x) for x in tok[i + 1:i + 1 + 7 * L]]
                for k, nm in enumerate(["co2", "mineralized", "paw", "moisture", "tf_decomp", "tf_nitr", "demand"]):
                    d[nm] = v[k::7]
                i += 1 + 7 * L
            elif t == "R":
                d["flat_contact"], d["stubble_contact"] = _f(tok[i + 1]), _f(tok[i + 2])
                i += 3
            elif t == "Y0":
                i += 1
            elif t == "Y":
                v = [_f(x) for x in tok[i + 1:i + 9 + L]]
                d.update(chaff=v[0], stubble=v[1], yield_=v[2], useful=v[3], disposal=v[4], uncut=v[5],
                         residue_conc=v[6], root_conc=v[7], roots_dead=v[8:8 + L])
                i += 9 + L
            elif t == "V":
                d.update(dict(zip(V_NAMES, [_f(x) for x in tok[i + 1:i + 19]])))
                i += 19
            elif t == "W":
                v = tok[i + 1:i + 11]
                d.update(raf=_f(v[0]), root_biomass=_f(v[1]), root_growth=_f(v[2]), root_length=_f(v[3]),
                         max_root_density=_f(v[4]), total_root_lengths=_f(v[5]), stage=int(v[6]),
                         gdds=_f(v[7]), peak_lai=_f(v[8]), gai_vital=_f(v[9]))
                per = [_f(x) for x in tok[i + 11:i + 11 + 3 * L]]
                d.update(root_lengths=per[0::3], root_fract=per[1::3], paw_n=per[2::3])
                i += 11 + 3 * L
            else:
                raise ValueError(f"unexpected token {t!r} in {tag} record")
        rec[key][tag] = d
    return head, rec


def dump_path(runs, case):
    f = glob.glob(os.path.join(runs, f"Liu_Linux_{case}_*", "VIC_output", "i6b_dump.txt"))
    return f[0] if f else None
