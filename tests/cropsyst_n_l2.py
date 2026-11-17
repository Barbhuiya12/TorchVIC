"""Season driver for the replay of the N-on CRB cases, day by day.

Drivers taken from the reference (as in the replay of tests/test_cropsyst_growth.py): the crop csv
weather, the csv water stress index (act = (1 - WSI) * limited potential transpiration), the
leaf water potential of the dump binary (i6_dump.txt), the VIC layer moisture, ice and temperature
(i6b_dump.txt P records) and the operations performed (Q records). Everything else is the port:
phenology (phenology.py), growth/canopy/roots/yield (growth.py, canopy.py, root.py, yield_.py),
crop N (nitrogen.py), soil mineral N (soil_n.py), residues (organic_matter.py), operations and
disturbance (n_management.py).

The day follows the land unit (CS/land_unit_sim.cpp:1281-2038): crop start_day and the first
half of the crop process (daily.potential, which does not depend on soil N); organic-matter
start_day; the operations; plant available water; organic-matter process_day; the crop's N
update, transpiration bookkeeping and actual growth (skipped on the harvest day: the harvest
operation terminates the crop before its process_day); hourly transformations; residue balance;
soil disturbance; end_day with the harvest residue.
"""
from __future__ import annotations

import csv
import glob
import os
import _paths  # noqa: E402
import re
from collections import defaultdict

import torch

from cropsyst_n_dump import read_dump, dump_path, L

from torchvic.crop.cropsyst import daily as dl
from torchvic.crop.cropsyst import phenology as ph
from torchvic.crop.cropsyst import canopy as cn
from torchvic.crop.cropsyst import yield_ as yd
from torchvic.crop.cropsyst import uptake as up
from torchvic.crop.cropsyst import readers as rdr
from torchvic.crop.cropsyst import soil_n as sn
from torchvic.crop.cropsyst import organic_matter as om
from torchvic.crop.cropsyst import nitrogen as nt
from torchvic.crop.cropsyst import n_management as nm
from torchvic.crop.cropsyst import n_engine as ne

SIM = _paths.data('build/vic_cropsyst/VIC-CropSyst_v3_CRB/Simulation')
N_RUNS = _paths.env_path(('TORCHVIC_CROPSYST_N_RUNS',), 'build/vic_cropsyst/crb_N')
OP_SOWING, OP_HARVEST, OP_TILLAGE, OP_INORGANIC_N = 1, 2, 3, 6
OP_AUTO_N_APPL, OP_SOIL_N_SAMPLING = 14, 15
COLS = ("GAI", "Biomass_kg_m2", "Root_depth_mm", "Green_Canopy_Cover", "Yield_kg_m2", "crop_evapotransp_max_mm",
        "Act_Transp_mm", "Accum_DD")


def _ini(path):
    secs = defaultdict(list)
    cur = None
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"^\[([^\]]+)\]", line.strip())
        if m:
            cur = {}
            secs[m.group(1)].append(cur)
            continue
        if cur is not None and "=" in line:
            k, v = line.split("=", 1)
            cur[k.strip()] = v.strip()
    return secs


def _num(v, default=0.0):
    try:
        return float(v.split()[0])
    except (AttributeError, IndexError, ValueError):
        return default


def rotation_managements(case):
    """crop name -> management ini (rotation file sowing entries)."""
    rot = re.search(r"_rot_(\d+)", glob.glob(f"{SIM}/Scenarios/Liu_Linux_{case}_*")[0]).group(1)
    secs = _ini(f"{SIM}/Database/Rotation/Code/{rot}.CS_rotation")
    out = {}
    for k, v in secs.items():
        if not k.startswith("sowing:"):
            continue
        for s in v:
            crop = os.path.splitext(os.path.basename(s["crop"]))[0]
            mgt = os.path.normpath(os.path.join(f"{SIM}/Database/Rotation/Code", s["management"]))
            if not os.path.exists(mgt):
                mgt = os.path.join(f"{SIM}/Database/Management/Name", os.path.basename(s["management"]))
            out[crop] = _ini(mgt)
    return out


def _saxton(quartz, clay, depth):
    """VIC soil hydraulics with CLAY_INPUT (read_soilparam.c:735-781), as tests/test_cropsyst_growth."""
    q = torch.tensor(quartz, dtype=torch.float64) * 100.0
    c = torch.tensor(clay, dtype=torch.float64) * 100.0
    por = 0.332 - 0.0007251 * q + torch.log10(c) * 0.1276
    b = -(-3.14 - 0.00222 * c ** 2 - 0.00003484 * q ** 2 * c)
    A = 100.0 * torch.exp(-4.396 - 0.0715 * c - 0.000488 * q ** 2 - 0.00004285 * q ** 2 * c)
    ae = -A * torch.pow(por, -b)
    return dict(b=b, ae=ae)


def _where_tree(mask, a, b):
    """Select between two NamedTuple trees per tile."""
    if isinstance(a, tuple):
        return type(a)(*[_where_tree(mask, x, y) for x, y in zip(a, b)])
    return torch.where(mask.reshape(mask.shape + (1,) * (a.dim() - mask.dim())), a, b)


def run_case(case, trace=None):
    """Replay a case; returns dict column -> list of (day, tile, ref, got) and N diagnostics."""
    f = glob.glob(f"{SIM}/Scenarios/Liu_Linux_{case}_*/VIC_output/vic_crop_daily*")[0]
    rows = list(csv.reader(open(f)))
    H = {h: i for i, h in enumerate(rows[0])}
    data = defaultdict(list)
    for r in rows[1:]:
        data[r[H["CroppingSyst_code"]]].append(r)
    tiles = sorted(data)
    Tn = len(tiles)
    nd = len(data[tiles[0]])
    col = lambda name, fn=float: [[fn(data[k][d][H[name]]) for k in tiles] for d in range(nd)]
    crop, lab = col("Crop_name", str), col("Grow_Stage", str)
    tmax, tmin, wsi, rhmin = col("Tmax"), col("Tmin"), col("water_stress_index"), col("RHUM_min_%")
    sw, etref, doy, year = col("SWRAD_w_m2"), col("CropSyst_refET_mm"), col("DOY", int), col("Year", int)
    refs = {c: col(c) for c in COLS}
    lat, lon = float(rows[1][H["lat"]]), float(rows[1][H["lon"]])
    names = sorted({r[H["Crop_name"]] for r in rows[1:]} - {"NOCROP"})
    tab = rdr.crop_table([f"{SIM}/Database/Crop/name/{c}.crp" for c in names])
    PPc, GPc, TPc = ph.phenology_params(tab), dl.growth_params(tab), up.transpiration_params(tab)
    NPc, RPc = nt.n_params_from_table(tab), om.residue_params_from_table(tab)
    mgts = rotation_managements(case)
    # dumps
    dpath = dump_path(N_RUNS, case)
    i6 = {}
    for line in open(os.path.join(os.path.dirname(dpath), "i6_dump.txt")):
        a = line.split()
        i6[(int(a[0]), int(a[1]), a[2])] = [float.fromhex(x) for x in a[3:]]
    head, rec = read_dump(dpath)
    tens = lambda v: torch.tensor(v, dtype=torch.float64)
    lay = sn.SoilLayers(*[tens([head[int(t)][k] for t in tiles]) for k in
                          ("thickness", "bulk_density", "sat", "fc", "pwp", "ph_function")])
    thickT = lay.thickness
    soil = open(glob.glob(f"{SIM}/Database/Soil/soil_param_id_{case}_*")[0]).read().split()
    sx = _saxton([float(x) for x in soil[114:131]], [float(x) for x in soil[273:290]], None)
    ae, bb = sx["ae"].expand(Tn, L), sx["b"].expand(Tn, L)
    real_wilt = torch.zeros(Tn, L, dtype=torch.float64)
    # states
    cid = torch.zeros(Tn, dtype=torch.long)
    s = g = None
    n = nt.initial_state((Tn,))
    st = nm.SoilState(sn.initial_state((Tn,), L), om.initial_state((Tn,), S=96, B=192),
                      tens([head[int(t)]["sand"] for t in tiles]), tens([head[int(t)]["clay"] for t in tiles]),
                      nm.initial_disturbance((Tn,), L))
    auto_total = torch.zeros(Tn, dtype=torch.float64)
    sdr = sn.sdr_operations()
    pairs = {c: [] for c in COLS}
    ndiag = []
    for d in range(nd):
        keys = [(year[d][k], doy[d][k], int(tiles[k])) for k in range(Tn)]
        sow = [crop[d][k] != "NOCROP" and (d == 0 or crop[d - 1][k] == "NOCROP") for k in range(Tn)]
        harvest = [crop[d][k] != "NOCROP" and "harvest" in lab[d][k] for k in range(Tn)]
        for k in range(Tn):
            if sow[k]:
                cid[k] = names.index(crop[d][k])
        PP, GP, TP = ph.gather_params(PPc, cid), dl.gather_params(GPc, cid), up.TranspirationParams(*[x[cid] for x in TPc])
        NP, RP = nt.gather_params(NPc, cid), om.gather_residue_params(RPc, cid)
        sowm, harv = torch.tensor(sow), torch.tensor(harvest)
        tx, tn = tens(tmax[d]), tens(tmin[d])
        tn_tom = tens(tmin[d + 1] if d + 1 < nd else tmin[d])
        if s is None:
            s, g = ph.initial_state(PP), dl.initial_state((Tn,), L, GP.gsp_stage.shape[-1], 400)
        s = ph.sow(s, PP, sowm, d)
        g = dl.sow(g, sowm, tx, tn_tom)
        n = nt.sow(n, sowm, NP)
        # Soil_base::reinitialize_for_crop (soil_base.cpp:194-220) at sowing
        rw = lay.sat * torch.pow(TP.lwp_wilt_point[:, None] / ae, -1.0 / bb)
        real_wilt = torch.where(sowm[:, None], rw, real_wilt)
        vmax = dl.gr.vpd_max_from_rhmin(tx, tens(rhmin[d]))
        sr = yd.sunrise_hour(lat, lon, doy[d][0]) + torch.zeros(Tn, dtype=torch.float64)
        g_before = g
        s, g = dl.potential(s, g, PP, GP, d, tx, tn, tn_tom, tens(sw[d]) * 0.0864, vmax,
                            tens(etref[max(d - 1, 0)]) / 1000.0, 370.0, 0.1, 0.065, 0.002, thickT, sr)
        # drivers of the day
        P = [rec[key].get("P") for key in keys]
        moist = tens([p["moist"] if p else [0.0] * L for p in P])
        ice = tens([p["ice"] if p else [0.0] * L for p in P])
        soil_t = tens([p["T"] if p else [0.0] * L for p in P])
        liquid = moist / (thickT * 1000.0)
        # organic matter start_day
        st = st._replace(residues=om.start_day(st.residues))
        # operations
        harvest_fate = None
        for k in range(Tn):
            for q in rec[keys[k]].get("Q", []):
                mk = torch.zeros(Tn, dtype=torch.bool)
                mk[k] = True
                if q["type"] in (OP_SOWING, OP_TILLAGE, OP_INORGANIC_N) and q["desc"]:
                    if q["type"] == OP_SOWING and not sow[k]:
                        continue                    # postponed sowing attempt: no field operation
                    if q["desc"] in sdr:
                        op = dict(sdr[q["desc"]], name=q["desc"])
                        ox = (1.5, 0.5) if q["type"] == OP_TILLAGE else None
                        st = nm.field_operation(st, mk, lay, op, ox)
                if q["type"] == OP_INORGANIC_N:
                    fr = sn.fract_inorganic_nh4_to_soil(q["nh4_source"], q["nh4_method"], q["vol_calculated"],
                                                         q["volatilization"], 20.0, 7.0)
                    st = st._replace(n=sn.fertilizer(st.n, mk, lay, tens(q["no3_kg_ha"]), tens(q["nh4_kg_ha"]),
                                                     tens(fr), tens(q["injection_depth_cm"])))
                if q["type"] in (OP_SOIL_N_SAMPLING, OP_AUTO_N_APPL):
                    cname = next((crop[dd][k] for dd in range(d, nd) if crop[dd][k] != "NOCROP"), "NOCROP")
                if q["type"] == OP_SOIL_N_SAMPLING:
                    m = mgts.get(cname)    # the management of the crop the sampling precedes
                    an = (m or {}).get("automatic_nitrogen", [{}])[0]
                    smp = (m or {}).get("soil_N_sampling", [{}])[0]
                    if an.get("automatic_N_mode") in ("balance_based", "local_calibration"):
                        depth = _num(smp.get("sampling_depth"), 0.0)
                        last = int(sn.layer_at_depth(thickT[k:k + 1], tens([depth]))[0]) if depth != 0.0 else L
                        m3 = 0.0
                        m4 = 0.0
                        for l in range(last):
                            m3 += float(st.n.no3[k, l])
                            m4 += float(st.n.nh4[k, l])
                        auto_total[k] = nm.auto_n_total(m3 * sn.NO3_TO_N + m4 * sn.NH4_TO_N,
                                                        _num(an.get("target_yield_N_uptake")),
                                                        _num(an.get("estimated_mineralization")),
                                                        _num(an.get("fertilizer_use_efficiency"), 100.0),
                                                        an.get("automatic_N_mode"),
                                                        _num(an.get("critical_soil_N_for_no_response")))
                if q["type"] == OP_AUTO_N_APPL:
                    ap = mgts[cname]["auto_N_application:1"][0]
                    amt = nm.auto_n_split(float(auto_total[k]), _num(ap.get("percent"), 100.0),
                                          _num(ap.get("NO3_N")), _num(ap.get("NH4_N")))
                    if amt is not None:
                        st = st._replace(n=sn.auto_n_application(st.n, mk, (tens(amt[0]), tens(amt[1]))))
        # plant available water (VIC_soil.cpp:129-143, real wilt point)
        paw = (liquid - real_wilt) / (lay.fc - real_wilt)
        paw = sn._macro_between(paw, torch.zeros_like(paw), torch.ones_like(paw))
        env = sn.abiotic_update(liquid, ice / (thickT * 1000.0), soil_t, lay)
        res, sN, dec = om.process(st.residues, env, st.n)
        st = st._replace(residues=res, n=sN)
        # crop: N update, transpiration, actual growth (not on the harvest day)
        crop_day = ~harv
        n2, sN, n_lim, n_pot = nt.crop_potential(n, NP, PP, GP, s, g_before, g, lay, st.n, paw, soil_t[:, 1])
        ce = s.present & g.canopy.exists & crop_day
        n_sd = nt.crop_start_day(n, NP, PP, GP, s, g_before, g, lay)      # the harvest day's start_day
        n = nt.NState(*[torch.where(crop_day, a, torch.where(harv, b, c)) for a, b, c in zip(n2, n_sd, n)])
        st = st._replace(n=sn.SoilN(torch.where(crop_day[:, None], sN.no3, st.n.no3),
                                    torch.where(crop_day[:, None], sN.nh4, st.n.nh4)))
        g = g._replace(limited_pot=torch.where(ce, n_pot, g.limited_pot))
        n = nt.NState(*[torch.where(crop_day, a, b) for a, b in zip(nt.crop_after_actual(n, s, g), n)])
        ws_ref = 1.0 - tens(wsi[d])
        act = ws_ref * g.limited_pot
        lw = tens([i6.get((keys[k][0], keys[k][1], tiles[k]), [0.0])[0] for k in range(Tn)])
        g_act = dl.actual(s, g, PP, GP, act, lw, sr, n_limited_growth=n_lim, n_leaf_stress=nt.leaf_stress_factor(n, NP))
        g = _where_tree(crop_day, g_act, g)
        # transformations, residue balance, disturbance
        sN, den, nit = sn.transformations(st.n, lay, env, soil_t, dec.co2)
        st = st._replace(n=sN, residues=om.balance(st.residues))
        aw = sn._macro_between((liquid - lay.pwp) / (lay.fc - lay.pwp), torch.zeros_like(liquid), torch.ones_like(liquid))
        st = st._replace(disturbance=nm.process_disturbance(st.disturbance, aw, st.clay))
        # harvest: yield, fate and residue (perform_harvest_or_clipping, distribute_any_crop_biomass_fate)
        bfh, yl, hi = dl.harvest(g, GP)
        if harv.any():
            res = st.residues
            conc = nt.residue_concentration(n, NP)
            pct = torch.stack([tens([_num(mgts.get(crop[d][k], {}).get("harvest:1", [{}])[0].get(key), 0.0)
                                     for k in range(Tn)])
                               for key in ("use", "graze", "return_as_manure", "dispose", "residue", "dead", "live")], -1)
            chaff, stubble = ne.harvest_fate(g.canopy, yl, pct, torch.ones_like(harv))
            roots = g.root.fract * g.root.biomass[:, None]
            res = om.add_surface(res, harv & (chaff > 0.0000001), om.FLAT, chaff, conc, RP)
            res = om.add_surface(res, harv & (stubble > 0.0000001), om.STUBBLE, stubble, conc, RP)
            res = om.add_subsurface(res, harv & (roots.sum(-1) > 0.0000001), roots, n.root_conc, RP)
            st = st._replace(residues=res)
            # start_fallow: Soil_base::reinitialize_for_crop(-1500) (land_unit_sim.cpp:2147-2160)
            rw = lay.sat * torch.pow(torch.full_like(ae, -1500.0) / ae, -1.0 / bb)
            real_wilt = torch.where(harv[:, None], rw, real_wilt)
        # compare
        got = {"GAI": cn.gai_vital(g.canopy), "Biomass_kg_m2": cn.biomass_vital(g.canopy),
               "Root_depth_mm": g.root.depth * 1000.0, "Green_Canopy_Cover": cn.interception_green(g.canopy, GP.kc),
               "Yield_kg_m2": torch.where(harv, yl, torch.zeros_like(yl)),
               "crop_evapotransp_max_mm": g.et_max * 1000.0, "Act_Transp_mm": act * 1000.0, "Accum_DD": s.gdds}
        for k in range(Tn):
            if crop[d][k] == "NOCROP":
                continue
            for c in COLS:
                if c == "Yield_kg_m2" and not harvest[k]:
                    continue
                if harvest[k] and c in ("GAI", "Biomass_kg_m2", "Green_Canopy_Cover", "Act_Transp_mm"):
                    continue
                ref = refs[c][d][k]
                if c == "Act_Transp_mm":     # the dump's full precision (the csv WSI driver has 8 digits)
                    full = i6.get((keys[k][0], keys[k][1], tiles[k]))
                    ref = full[4] * 1000.0 if full else ref
                pairs[c].append((d, tiles[k], ref, float(got[c][k])))
            e = rec[keys[k]].get("E")
            if e is not None:
                ndiag.append((d, tiles[k], e["no3"], e["nh4"], sn.no3_n(st.n)[k].tolist(), sn.nh4_n(st.n)[k].tolist()))
        if trace is not None:
            trace.append(dict(d=d, keys=keys, n=n, st=st, g=g))
        s = ph.terminate(s, harv)
    return pairs, ndiag
