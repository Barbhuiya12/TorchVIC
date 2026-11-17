"""E10 of the VIC-CropSyst port: the daily call order of one crop tile, the run state
and its restart, for the crop connector's backend 'cropsyst' (mode A: VIC keeps its water balance,
CropSyst diagnoses uptake and stress).

The order of one day, from Crop_complete (crop_cropsyst.cpp) and the land unit
(VIC_land_unit_simulation.cpp), is

  1. sowing events of the day          new Crop_complete on the tile (plant/start, :619-652):
                                       phenology, canopy, roots, yield, leaf water potential
  2. ET max, start_day, process_day    daily.potential: ET max on yesterday's canopy (the
     up to the transpiration call      reference ET of the PREVIOUS day, land_unit_sim.cpp:1301),
                                       temperatures, phenology, thermal-time events, CO2, roots,
                                       canopy, attainable growth, limited potential transpiration
  3. E1/E2 root water uptake           Crop_common::process_transpiration_m (crop_common.cpp:124-200)
                                       on the layer moisture VIC passes in (state at the start of
                                       the day, as the reference reads it); a replay run replaces
                                       it by the water stress factor and the leaf water potential
                                       of a reference run
  4. actual growth, canopy, end_day    daily.actual
  5. harvest events                    synchronised to a growth stage plus an offset
                                       (event_scheduler_crop.cpp): yield from the biomass at
                                       maturity, then the crop is terminated when the event says so
                                       (biomass fate harvest_once with terminate: the canopy leaves)

Everything is a masked tensor operation over the tile axis [T] (and [T, layer]); the day counter is
a device tensor and the day's inputs are rows of device tables picked with index_select, so a day
step has no host synchronisation and no data-dependent control flow: it is capturable in a CUDA
graph and traces under torch.compile. The state is a tree of tensors; `flatten`/`unflatten`
turn it into a list for the restart, which writes and reads the bytes unchanged (bit-identical
continuation).

Nitrogen: a run built with nitrogen=True carries the nitrogen static data and state
(Static.n, CropSystState.n) and its day is n_engine.day_step_n: crop N, soil mineral N, residues,
and the management events that act on them; the nitrogen-off day below is unchanged.

Orchards and perennial crops: when a crop of the run is a fruit tree or perennial,
Static.orc / CropSystState.orc carry orchard.py's parameters and state, and both day steps call its
hooks: day_begin (the land unit's hourly temperatures, start of an established crop), after_start
(fruit events, the harvest event, the inactive period: chill, bud break, restart, dormancy) between
the crop's start_day and process_day, and after_actual. A tile code without a rotation file is an
established crop (Crop/Code/<code>.CS_crop, sown on the first day).
"""
from __future__ import annotations

import os
from typing import NamedTuple, Optional

import torch

from . import calendars as cal
from . import canopy as cn
from . import daily as dl
from . import phenology as ph
from . import uptake as up
from . import yield_ as yd
from .readers import InputError

PSI_AT_FC = -33.0                 # soil_con.VCS.water_pot_at_FC (read_soilparam.c:779) [J/kg]
DEFAULT_CO2 = 370.0               # ppm
# delta, gamma, ra enter only the CO2 transpiration factor; unless the weather carries them
# (day_tables) these fixed values are used, as in the replay of the reference runs.
DEFAULT_DELTA, DEFAULT_GAMMA, DEFAULT_RA = 0.1, 0.065, 0.002


class NotYetImplemented(InputError, NotImplementedError):
    """An option of the CropSyst port that is planned but not built."""


def check_nitrogen(nitrogen):
    """nitrogen must be a bool: True runs crop and soil nitrogen (n_engine), False without."""
    if not isinstance(nitrogen, (bool, int)) or nitrogen not in (0, 1):
        raise InputError(f"cropsyst: nitrogen must be True or False, not {nitrogen!r}")
    return bool(nitrogen)


# ----------------------------------------------------------------------------- static data

class Static(NamedTuple):
    """Run constants on the device: per-crop tables [n_crop], harvest events per management
    [n_mgt, H], soil per tile [T, L]."""
    pp: ph.PhenologyParams
    gp: dl.GrowthParams
    tp: up.TranspirationParams
    h_stage: torch.Tensor            # [n_mgt, H] long, growth stage of the event (-1: no event)
    h_offset: torch.Tensor           # [n_mgt, H] long days after the stage
    h_terminate: torch.Tensor        # [n_mgt, H] bool
    thick: torch.Tensor              # [T, L] m
    max_moist: torch.Tensor          # [T, L] mm
    air_entry: torch.Tensor          # [T, L] J/kg
    campbell_b: torch.Tensor         # [T, L]
    co2_on: torch.Tensor             # 0-d bool
    n: Optional[object] = None       # n_engine.NStatic with nitrogen on
    orc: Optional[object] = None     # orchard.OrchardStatic when a crop is perennial or a fruit tree
    h_min_lai: Optional[torch.Tensor] = None   # [n_mgt, H] harvest event constraints
    h_min_bm: Optional[torch.Tensor] = None    # [n_mgt, H] kg/m2
    clip: Optional[object] = None    # clipping.ClipStatic when a management clips


class DayInputs(NamedTuple):
    """One day of drivers, all [T] on the device (rows of DayTables, or built by a coupled run)."""
    tmax: torch.Tensor               # deg C
    tmin: torch.Tensor
    tmin_tom: torch.Tensor           # tomorrow's Tmin (hourly temperature estimator)
    srad: torch.Tensor               # MJ m-2 d-1
    rhmin: torch.Tensor              # %
    et_ref: torch.Tensor             # m d-1, today's reference ET (used by the NEXT day's ET max)
    co2: torch.Tensor                # ppm
    delta: torch.Tensor
    gamma: torch.Tensor
    ra: torch.Tensor
    sunrise: torch.Tensor            # hour
    sow_crop: torch.Tensor           # long, crop row sown today (-1: none)
    sow_mgt: torch.Tensor            # long, management row of that sowing
    sow_mode: torch.Tensor           # long, 0: sow as soon as the 15-day mean temperature allows
    sow_thr: torch.Tensor            # deg C, that mode's required mean temperature
    warm15: torch.Tensor             # deg C, mean daily mean temperature of today .. today + 14
    act_frac: Optional[torch.Tensor] = None     # replay: 1 - WSI of the reference (act / limited pot)
    lwp: Optional[torch.Tensor] = None          # replay: leaf water potential of the reference
    moist: Optional[torch.Tensor] = None        # replay: [T, L] layer moisture mm (liquid + ice)
    ice: Optional[torch.Tensor] = None          # replay: [T, L] layer ice mm
    soil_t: Optional[torch.Tensor] = None       # replay: [T, L] layer temperature deg C
    doy: Optional[torch.Tensor] = None          # 0-d long day of year (orchard crops)


class DayTables(NamedTuple):
    """The same fields as DayInputs, stacked over the days: [n_day, T]."""
    tmax: torch.Tensor
    tmin: torch.Tensor
    tmin_tom: torch.Tensor
    srad: torch.Tensor
    rhmin: torch.Tensor
    et_ref: Optional[torch.Tensor]
    co2: torch.Tensor
    delta: torch.Tensor
    gamma: torch.Tensor
    ra: torch.Tensor
    sunrise: torch.Tensor
    sow_crop: torch.Tensor
    sow_mgt: torch.Tensor
    sow_mode: torch.Tensor
    sow_thr: torch.Tensor
    warm15: torch.Tensor
    act_frac: Optional[torch.Tensor] = None
    lwp: Optional[torch.Tensor] = None
    moist: Optional[torch.Tensor] = None
    ice: Optional[torch.Tensor] = None
    soil_t: Optional[torch.Tensor] = None
    doy: Optional[torch.Tensor] = None          # [n_day] long

    @property
    def n_day(self):
        return int(self.tmax.shape[0])

    def at(self, day: torch.Tensor, et_ref=None) -> DayInputs:
        """Row `day` (0-d long tensor on the device): index_select, no host copy."""
        i = day.reshape(1)
        row = lambda t: None if t is None else torch.index_select(t, 0, i)[0]
        vals = {f: row(getattr(self, f)) for f in DayInputs._fields}
        if et_ref is not None:
            vals["et_ref"] = et_ref
        return DayInputs(**vals)


class CropSystState(NamedTuple):
    day: torch.Tensor                # 0-d long, the day the next step runs
    phen: ph.PhenologyState
    grow: dl.GrowthState
    cid: torch.Tensor                # [T] long crop row of the tile's current crop
    mid: torch.Tensor                # [T] long management row
    lwp: torch.Tensor                # [T] leaf water potential carried over (E1)
    et_prev: torch.Tensor            # [T] yesterday's reference ET [m], NaN before the first day
    pend_crop: torch.Tensor          # [T] long, a scheduled sowing waiting for its conditions (-1: none)
    pend_mgt: torch.Tensor
    pend_mode: torch.Tensor
    pend_thr: torch.Tensor
    n: Optional[object] = None       # n_engine.NEState with nitrogen on
    orc: Optional[object] = None     # orchard.OrchardState with orchard crops
    clip: Optional[object] = None    # clipping.ClipState when a management clips


class DayOutput(NamedTuple):
    """What a day reports, [T] unless noted. Canopy and root fields are those of the day record
    (before a harvest removes the crop); `lai` and `present` are the state after the day."""
    gai: torch.Tensor
    biomass: torch.Tensor            # kg m-2 vital canopy biomass
    root_depth: torch.Tensor         # m (day record; 0 on a tile without a crop)
    green_cover: torch.Tensor
    et_max: torch.Tensor             # m d-1
    limited_pot: torch.Tensor        # m d-1
    act_transp: torch.Tensor         # m d-1
    water_stress: torch.Tensor       # act / limited pot (ws; WSI = 1 - ws)
    lwp: torch.Tensor                # J/kg
    uptake: torch.Tensor             # [T, L] m d-1 per layer (CropSyst layer l+1 = VIC layer l)
    yield_: torch.Tensor             # kg m-2 on the harvest day, else 0
    hi: torch.Tensor
    harvest: torch.Tensor            # bool
    sown: torch.Tensor               # bool
    stage: torch.Tensor              # long
    lai: torch.Tensor                # state after the day
    present: torch.Tensor            # bool, crop stands after the day
    active_record: torch.Tensor      # bool, a crop was on the tile during the day
    n: Optional[object] = None       # n_engine.NDayOutput with nitrogen on
    orc: Optional[object] = None     # orchard.OrchardDayOutput with orchard crops
    icpt_used: Optional[torch.Tensor] = None   # m: stored interception the day evaporated (day_step icpt_m)
    clip: Optional[object] = None    # clipping.ClipDayOutput when a management clips
    cover_total: Optional[torch.Tensor] = None   # total canopy cover fraction after the day (0 without canopy)


def _take(t, idx):
    return t[idx]


def day_step(st: CropSystState, S: Static, d: DayInputs, moist_mm: Optional[torch.Tensor] = None,
             ice_mm: Optional[torch.Tensor] = None, soil_t: Optional[torch.Tensor] = None,
             icpt_m: Optional[torch.Tensor] = None):
    """One day, in the order of the module docstring. Returns (state, DayOutput).
    `icpt_m` [T] m: VIC's canopy store handed to the crop (CROPSYST_HANDLE_CANOPY_EVAP,
    VIC_land_unit_simulation.cpp:353-360); it serves the day's potential transpiration first
    (transpiration.cpp:65-78, 105-106) and `DayOutput.icpt_used` is what it evaporated."""
    if S.n is not None:
        from . import n_engine as ne
        if d.moist is not None:                     # replay drivers of a reference run
            moist_mm, ice_mm, soil_t = d.moist, d.ice, d.soil_t
        return ne.day_step_n(st, S, d, moist_mm, ice_mm, soil_t, icpt_m)
    dev = st.lwp.device
    # sowing events: scheduled on their date; a computed sowing (mode min_temp_req) is postponed
    # day by day until the mean temperature of the next 15 days reaches the threshold
    # (Sowing_operation::good_planting_conditions, rot_param.cpp:255-344; perform_op.cpp:530-565)
    sched = d.sow_crop >= 0
    pc = torch.where(sched, d.sow_crop, st.pend_crop)
    pm = torch.where(sched, d.sow_mgt, st.pend_mgt)
    pmode = torch.where(sched, d.sow_mode, st.pend_mode)
    pthr = torch.where(sched, d.sow_thr, st.pend_thr)
    sow = (pc >= 0) & ((pmode != 0) | (d.warm15 >= pthr))
    cid = torch.where(sow, pc, st.cid)
    mid = torch.where(sow, pm, st.mid)
    neg = torch.full_like(pc, -1)
    pc = torch.where(sow, neg, pc)
    PP = ph.gather_params(S.pp, cid)
    GP = dl.gather_params(S.gp, cid)
    TP = up.TranspirationParams(*[_take(x, cid) for x in S.tp])
    clip = S.clip
    if clip is not None:                            # clipping (clipping.py)
        from . import clipping as cl
        csx = cl.sow(st.clip, clip, mid, sow)
        PP = cl.adjust_params(PP, csx)
    # 1. sowing
    phen = ph.sow(st.phen, PP, sow, st.day)
    grow = dl.sow(st.grow, sow, d.tmax, d.tmin_tom)
    lwp = torch.where(sow, torch.zeros_like(st.lwp), st.lwp)
    orc = S.orc
    if orc is not None:                             # perennial and fruit crops (orchard.py)
        from . import orchard as oc
        OP = oc.gather(orc.OP, cid)
        o, phen, grow = oc.day_begin(st.orc, OP, phen, grow, PP, GP, sow, st.day, d.doy, orc.south,
                                     d.tmax, d.tmin, d.tmin_tom, S.thick, AS=orc.arch)
    # 2. potential: yesterday's ET reference (today's on the first day)
    et_used = torch.where(torch.isnan(st.et_prev), d.et_ref, st.et_prev)
    vpd_max = dl.gr.vpd_max_from_rhmin(d.tmax, d.rhmin)
    phn, g1, ctx = dl.potential_start(phen, grow, PP, GP, st.day, d.tmax, d.tmin, d.tmin_tom, d.srad,
                                      vpd_max, et_used, d.co2, d.delta, d.gamma, d.ra, S.thick,
                                      co2_on=S.co2_on)
    ccif = 1.0
    if orc is not None:
        phn, g1, o, restarted, _dormant, hv_ev, ccif = oc.after_start(
            phn, g1, o, OP, PP, GP, st.day, d.doy, d.tmin, (d.tmax + d.tmin) / 2.0, orc.south, S.thick,
            _take(S.h_stage, mid), _take(S.h_offset, mid), _take(S.h_min_lai, mid), _take(S.h_min_bm, mid),
            AS=orc.arch, srad=d.srad)
        lwp = torch.where(restarted, torch.zeros_like(lwp), lwp)
    ckill = None
    if clip is not None:                            # the day's performable operations (clipping.py)
        csx, phn, g1, _, cday = cl.morning(csx, clip, cid, mid, phn, g1, PP, GP, st.day,
                                           fruit=None if orc is None else OP.is_fruit, tavg=(d.tmax + d.tmin) / 2.0)
        lwp = torch.where(cday.lwp_reset, torch.zeros_like(lwp), lwp)
        ckill = cday.kill
        if orc is not None:
            o = oc._rep(o, em_count=o.em_count + cday.lwp_reset.to(torch.long))
            o = oc.harvest(o, cday.fruit_harvest)
            hv_ev = hv_ev | cday.fruit_harvest
    g = dl.potential_process(phn, g1, PP, GP, ctx, S.thick, ccif)
    g_proc = g
    # 3. uptake
    icpt_used = None
    if d.act_frac is not None:                      # replay of a reference run's water stress
        act = d.act_frac * g.limited_pot
        lw = d.lwp
        uptake = torch.zeros_like(S.thick)
        lwp_next = lw
    else:
        if moist_mm is None:
            raise ValueError("day_step: layer moisture needed (or a replay act_frac/lwp)")
        ice = torch.zeros_like(moist_mm) if ice_mm is None else ice_mm
        psi = up.soil_water_potential(moist_mm, ice, S.thick, S.max_moist, S.air_entry,
                                      S.campbell_b)
        r = up.root_uptake(g.limited_pot, cn.interception_green(g.canopy, GP.kc),
                           torch.zeros_like(lwp) if icpt_m is None else icpt_m, g.root.fract, psi,
                           torch.full_like(psi, PSI_AT_FC), TP, lwp,
                           co2_transp_factor=g.transp_factor, active=g.canopy.exists & phn.present)
        act = r.act_transp_m                        # dl.actual clamps a negative uptake to 0
        lw = r.leaf_water_pot
        uptake = r.uptake_m
        lwp_next = r.leaf_water_pot
        if icpt_m is not None:
            icpt_used = icpt_m - r.interception_m
    # 4. actual growth and end of day
    if orc is None:
        g = dl.actual(phn, g, PP, GP, act, lw, d.sunrise)
    else:
        g, o = dl.actual(phn, g, PP, GP, act, lw, d.sunrise, orchard=(OP, o))
    clip_out = None
    if clip is not None:
        # a terminating event ends the crop before its process_day (land_unit_sim.cpp:1830-1831)
        g = cl._where_tree(ckill, dl._rep(g1, et_max=g_proc.et_max), g)
        act = torch.where(ckill, torch.zeros_like(act), act)
        csx, phn, g, _, aclip, afate, ia2 = cl.evening(csx, clip, cid, mid, phn, g, PP, GP, st.day,
                                                       tavg=(d.tmax + d.tmin) / 2.0, mask=~ckill)
        lwp_next = torch.where(ia2, torch.zeros_like(lwp_next), lwp_next)
        if orc is not None:
            o = oc._rep(o, em_count=o.em_count + ia2.to(torch.long))
        clip_out = cl.day_output(cday, afate, aclip)
    # 5. harvest events
    bfh, yl, hi = dl.harvest(g, GP)
    hs = _take(S.h_stage, mid)
    ho = _take(S.h_offset, mid)
    ht = _take(S.h_terminate, mid)
    sd = phn.sync_day.gather(-1, hs.clamp(min=0))
    fire = phn.present[..., None] & (hs >= 0) & (sd >= 0) & (st.day == sd + ho)
    if orc is not None:                             # fruit trees: harvested in after_start
        fire = fire & ~OP.is_fruit[..., None]
    harvest = fire.any(-1)
    terminate = (fire & ht).any(-1)
    rec_present = phn.present
    root_rec = g.root.depth
    orc_out = None
    if orc is not None:
        phn, g, o = oc.after_actual(phn, g, o, OP, st.day, S.thick)
        hv = hv_ev
        yl = torch.where(hv, o.latest_yield, yl)
        hi = torch.where(hv, torch.zeros_like(hi), hi)
        harvest = harvest | hv
        orc_out = oc.day_output(o, phn, hv, restarted)
    if clip is not None:
        cut = (clip_out.event & ~cday.fruit_harvest) | clip_out.auto
        yl = torch.where(harvest, yl, torch.zeros_like(yl)) + clip_out.harvested
        hi = torch.where(harvest, hi, torch.zeros_like(hi))
        harvest = harvest | cut
        terminate = terminate | ckill
        csx = cl.end_crop(csx, terminate)
    phn_out = ph.terminate(phn, terminate)
    c = g.canopy
    out = DayOutput(
        gai=cn.gai_vital(c), biomass=cn.biomass_vital(c), root_depth=torch.where(
            rec_present, root_rec, torch.zeros_like(root_rec)),
        green_cover=cn.interception_green(c, GP.kc),
        et_max=g.et_max, limited_pot=g.limited_pot, act_transp=act, water_stress=g.ws, lwp=lw,
        uptake=uptake, yield_=torch.where(harvest, yl, torch.zeros_like(yl)),
        hi=torch.where(harvest, hi, torch.zeros_like(hi)), harvest=harvest, sown=sow,
        stage=phn.stage,
        lai=torch.where(phn_out.present & c.exists, cn.lai(c), torch.zeros_like(g.ws)),
        present=phn_out.present, active_record=rec_present, orc=orc_out, clip=clip_out, icpt_used=icpt_used,
        cover_total=torch.where(phn_out.present & c.exists, c.cover_total, torch.zeros_like(c.cover_total)))
    new = CropSystState(st.day + 1, phn_out, g, cid, mid, lwp_next, d.et_ref, pc, pm, pmode, pthr,
                        None, o if orc is not None else None, csx if clip is not None else None)
    return new, out


# ----------------------------------------------------------------------------- state as a list

def flatten(x):
    """The tensors of a state tree, in a fixed order."""
    out = []
    for v in x:
        if v is None:                    # an absent part (nitrogen off)
            continue
        out += flatten(v) if isinstance(v, tuple) else [v]
    return out


def unflatten(template, flat):
    it = iter(flat)

    def build(t):
        if t is None:
            return None
        if isinstance(t, tuple):
            vals = [build(v) for v in t]
            return type(t)(*vals) if hasattr(t, "_fields") else tuple(vals)
        return next(it)
    out = build(template)
    if next(it, None) is not None:
        raise ValueError("unflatten: more tensors than the state holds")
    return out


def initial_state(S: Static, T, n_layer, P=400, dtype=torch.float64, device=None) -> CropSystState:
    dev = device if device is not None else S.thick.device
    cid = torch.zeros(T, dtype=torch.long, device=dev)
    PP = ph.gather_params(S.pp, cid)
    G = S.gp.gsp_stage.shape[-1]
    z = torch.zeros(T, dtype=dtype, device=dev)
    nst = None
    if S.n is not None:
        from . import n_engine as ne
        nst = ne.initial_state(S.n, T, dtype, dev)
    ost = None
    if S.orc is not None:
        from . import orchard as oc
        ost = oc.initial_state(T, dtype, dev)
        if S.orc.arch is not None:                  # canopy architecture (orchard_arch.py)
            ost = ost._replace(arch=oc.oa.initial_state(T, dtype, dev))
    cst = None
    if S.clip is not None:
        from . import clipping as cl
        cst = cl.initial_state(S.clip, T, dtype, dev)
    return CropSystState(torch.zeros((), dtype=torch.long, device=dev), ph.initial_state(PP),
                         dl.initial_state((T,), n_layer, G, P, dtype=dtype, device=dev), cid,
                         cid.clone(), z.clone(), torch.full((T,), float("nan"), dtype=dtype,
                                                            device=dev),
                         cid - 1, cid - 1, cid - 1, z.clone(), nst, ost, cst)


# ----------------------------------------------------------------------------- soil

def saxton_soil(quartz, clay, thickness_m):
    """Soil hydraulics the CRB reference derives from texture with CLAY_INPUT
    (read_soilparam.c:735-781; P5/E14): Campbell b, air entry and maximum moisture [mm] from the
    quartz and clay fractions [..., L] and the layer thickness [m]. Porosity from the Saxton
    pedotransfer; bulk density 2650 (1 - porosity)."""
    q = torch.as_tensor(quartz, dtype=torch.float64) * 100.0
    c = torch.as_tensor(clay, dtype=torch.float64) * 100.0
    depth = torch.as_tensor(thickness_m, dtype=torch.float64)
    por = 0.332 - 0.0007251 * q + torch.log10(c) * 0.1276
    b = -(-3.14 - 0.00222 * c ** 2 - 0.00003484 * q ** 2 * c)
    A = 100.0 * torch.exp(-4.396 - 0.0715 * c - 0.000488 * q ** 2 - 0.00004285 * q ** 2 * c)
    ae = -A * torch.pow(por, -b)
    bulk = 2650.0 * (1.0 - por)
    max_moist = depth * (1.0 - bulk / 2650.0) * 1000.0
    return dict(campbell_b=b, air_entry=ae, max_moist=max_moist)


# ----------------------------------------------------------------------------- tables

def _harvest_tables(mgts, device, dtype=torch.float64):
    """[n_mgt, H] harvest events: growth stage, offset, terminate, and the constraints of the land
    unit's harvest (min_LAI, min_biomass kg/m2, perform_op.cpp:1140-1150; default 0)."""
    P, V = mgts.params, mgts.vocab
    n = mgts.n
    key = "harvest.event_offset"
    if key not in P or P[key].shape[1] == 0:
        z = torch.zeros((n, 1), dtype=torch.long, device=device)
        zf = torch.zeros((n, 1), dtype=dtype, device=device)
        return z - 1, z, torch.zeros((n, 1), dtype=torch.bool, device=device), zf, zf
    valid = mgts.valid["harvest"].cpu()
    H = P[key].shape[1]
    stage = torch.full((n, H), -1, dtype=torch.long)
    off = P[key].cpu().long().clone()
    term = P["harvest.terminate_crop"].cpu().clone() if "harvest.terminate_crop" in P \
        else torch.ones((n, H), dtype=torch.bool)
    from .clipping import is_classic_harvest
    for i in range(n):                                  # setup-time parsing on the host
        for j in range(H):
            if not bool(valid[i, j]):
                continue
            if "harvest.enabled" in P and not bool(P["harvest.enabled"][i, j]):
                continue
            if not is_classic_harvest(mgts, i, j):
                continue                                # the clipping path (clipping.py)
            sync = V["harvest.event_synchronization"][int(P["harvest.event_synchronization"][i, j])]
            if sync == "after_normal_crop_growth_stage":
                word = V["harvest.event_phenologic"][int(P["harvest.event_phenologic"][i, j])]
                stage[i, j] = dl.STAGE_WORDS[word]
            elif sync == "after_fruit_tree_growth_stage":   # fruit_tree_event_sequence (growth_stages.c)
                from .orchard import FRUIT_STAGE_LABELS
                k = "harvest.event_phenologic(fruit)"
                stage[i, j] = FRUIT_STAGE_LABELS.index(V[k][int(P[k][i, j])])
            else:
                raise NotYetImplemented(
                    f"cropsyst: harvest event synchronised {sync!r} (management row {i}) is not "
                    "ported (crops harvest after a normal or fruit tree growth stage)")

    def con(k, scale):
        t = P.get(f"harvest.{k}")
        if t is None:
            return torch.zeros((n, H), dtype=dtype)
        return torch.nan_to_num(t.cpu().to(dtype), nan=0.0) * scale
    return (stage.to(device), off.to(device), term.to(device), con("min_LAI", 1.0).to(device),
            (con("min_biomass", 1.0) / 10000.0).to(device))


def sowing_day_tables(calendar: cal.SowingCalendar, year, doy, device, slot_mode, slot_thr):
    """[n_day, T] crop row, management row, sowing mode (0: min_temp_req, 1: fixed date) and
    required temperature of the sowing scheduled on each run day, from the calendar and the
    (year, day of year) of every day. slot_mode, slot_thr [T, S]: the mode and avg_temp of
    each tile's sowing slots. Built once; the day step only indexes it."""
    year = torch.as_tensor(year, dtype=torch.long, device=device)
    doy = torch.as_tensor(doy, dtype=torch.long, device=device)
    yi = (year - calendar.first_year).clamp(0, calendar.sow_doy.shape[1] - 1)
    inside = (year - calendar.first_year >= 0) & (year - calendar.first_year < calendar.sow_doy.shape[1])
    sd = calendar.sow_doy.to(device)[:, yi, :]                      # [T, n_day, S]
    hit = (sd == doy[None, :, None]) & inside[None, :, None]
    neg = torch.full_like(sd, -1)
    crop = torch.where(hit, calendar.crop_idx.to(device)[:, yi, :], neg)
    mgt = torch.where(hit, calendar.mgt_idx.to(device)[:, yi, :], neg)
    mode = torch.where(hit, slot_mode.to(device)[:, None, :].expand_as(sd), torch.ones_like(sd))
    thr = torch.where(hit, slot_thr.to(device)[:, None, :].expand_as(sd),
                      torch.zeros(sd.shape, dtype=slot_thr.dtype, device=device))
    f = lambda t: t.amax(-1).T.contiguous()
    return f(crop), f(mgt), mode.amin(-1).T.contiguous(), thr.amax(-1).T.contiguous()


def _warm_window(tmax, tmin, window=15):
    """[n_day, T] mean over today .. today + window - 1 of the daily mean temperature (the days
    that exist, at the end of the data)."""
    n = tmax.shape[0]
    t = (tmax + tmin) / 2.0
    pad = torch.cat([t, torch.zeros((window - 1,) + tuple(t.shape[1:]), dtype=t.dtype,
                                    device=t.device)], 0)
    sums = pad.unfold(0, window, 1).sum(-1)
    cnt = (n - torch.arange(n, device=t.device)).clamp(max=window).to(t.dtype)
    return sums / cnt.reshape((-1,) + (1,) * (t.dim() - 1))


def _slot_modes(rs: cal.RotationSet, tile_rot):
    """[T, S] sowing mode code (0 min_temp_req, 1 fixed date) and avg_temp of each tile's slots.
    The plant-available-water and precipitation-delay conditions are not evaluated; a rotation
    that sets them to anything but the permissive CRB values is refused."""
    P, V = rs.rot.params, rs.rot.vocab
    S = P["sowing.event_date"].shape[1]
    mode = torch.ones_like(P["sowing.event_date"])
    thr = torch.zeros(mode.shape, dtype=torch.float64, device=mode.device)
    if "sowing.mode" in P:
        raw = P["sowing.mode"]
        words = V["sowing.mode"]
        for i, w in enumerate(words):
            if w != "min_temp_req":
                raise NotYetImplemented(f"cropsyst: sowing mode {w!r} is not ported (only fixed "
                                        "date and min_temp_req)")
        comp = (raw >= 0) & rs.sow_valid
        mode = torch.where(comp, torch.zeros_like(mode), mode)
        avg = torch.nan_to_num(P["sowing.avg_temp"].to(torch.float64), nan=0.0)
        thr = torch.where(comp, avg, thr)
        bad = comp & ((torch.nan_to_num(P["sowing.min_PAW"], nan=0.0) > 0.0)
                      | (torch.nan_to_num(P["sowing.max_PAW"], nan=1.0) < 1.0)
                      | (torch.where(P["sowing.delay_after_precip"] < 0,
                                     torch.zeros_like(P["sowing.delay_after_precip"]),
                                     P["sowing.delay_after_precip"]) > 0))
        if bool(bad.any()):
            raise NotYetImplemented("cropsyst: a computed sowing with a plant-available-water or "
                                    "precipitation-delay condition is not ported (CRB rotations "
                                    "use the permissive values 0 / 1 / 0)")
    tr = tile_rot.to(mode.device)
    return mode[tr], thr[tr]


def build_static(rs: cal.RotationSet, thick, max_moist, air_entry, campbell_b, device,
                 co2_on=True, dtype=torch.float64, tile_lat=None, year=None, doy=None,
                 clip_mode="fixed") -> Static:
    """Static tables from the rotation set (its crop and management tables) and the per-tile soil
    [T, L]."""
    pp = ph.phenology_params(rs.crops, device=device)
    gp = dl.growth_params(rs.crops, device=device)
    tp = up.TranspirationParams(*[x.to(device) for x in up.transpiration_params(rs.crops)])
    hs, ho, ht, hl, hb = _harvest_tables(rs.mgts, device, dtype)
    t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    orc = None
    from . import orchard as oc
    OP = oc.orchard_params(rs.crops, device=device, dtype=dtype)
    oc.check_supported(rs.crops, OP, pp)
    arch = rs.crops.params.get("morphology.canopy_architecture")
    if arch is not None and bool((arch == 1).any()):    # canopy architecture (orchard_arch.py)
        OP = OP._replace(arch=oc.oa.arch_params(rs.crops, pp, device=device, dtype=dtype))
    if bool((OP.perennial | OP.is_fruit).any()):
        T = int(torch.as_tensor(thick).shape[0])
        lat_t = torch.zeros(T, dtype=dtype) if tile_lat is None else torch.as_tensor(tile_lat, dtype=dtype)
        orc = oc.OrchardStatic(OP, hl, hb, (lat_t < 0.0).to(device))
    from . import clipping as cl
    clip = cl.build_static(rs, [] if year is None else list(year), [] if doy is None else list(doy), device, dtype,
                           clip_mode)
    return Static(pp, gp, tp, hs, ho, ht, t(thick), t(max_moist), t(air_entry), t(campbell_b),
                  torch.as_tensor(bool(co2_on), device=device), None, orc, hl, hb, clip)


def day_tables(weather: dict, sow, tile_cell, lat, lon, doy, device,
               dtype=torch.float64, replay=None) -> DayTables:
    """DayTables [n_day, T] from daily weather arrays and the sowing tables.

    sow: (crop, mgt, mode, thr) day tables of sowing_day_tables.
    weather: tmax, tmin [deg C], srad [W m-2 daily mean], rhmin [%], optional et_ref [mm d-1]
    (else supplied by the run each day), co2 [ppm], delta, gamma, ra; each [n_day, n_cell] or
    [n_day] (all cells). `tile_cell` [T] maps tiles to cells. lat, lon [n_cell] degrees; doy
    [n_day]. replay: dict act_frac, lwp [n_day, T]."""
    tc = torch.as_tensor(tile_cell, dtype=torch.long, device=device)

    def tab(name, default=None):
        v = weather.get(name)
        if v is None:
            if default is None:
                return None
            v = default
        v = torch.as_tensor(v, dtype=dtype, device=device)
        n_day = sow[0].shape[0]
        if v.dim() == 0:
            return v.expand(n_day, tc.shape[0]).contiguous()
        if v.dim() == 1:
            return v[:, None].expand(n_day, tc.shape[0]).contiguous()
        return v[:, tc].contiguous()

    tmax, tmin = tab("tmax"), tab("tmin")
    tmin_tom = torch.cat([tmin[1:], tmin[-1:]], 0)
    et = tab("et_ref")
    lat = torch.as_tensor(lat, dtype=torch.float64).reshape(-1)
    lon = torch.as_tensor(lon, dtype=torch.float64).reshape(-1)
    d = torch.as_tensor(doy, dtype=torch.float64).reshape(-1, 1)
    sunrise = yd.sunrise_hour(lat[None, :], lon[None, :], d).to(device=device, dtype=dtype)
    sunrise = sunrise.expand(d.shape[0], lat.shape[0])[:, tc].contiguous()
    return DayTables(
        tmax=tmax, tmin=tmin, tmin_tom=tmin_tom, srad=tab("srad") * 0.0864, rhmin=tab("rhmin"),
        et_ref=None if et is None else et / 1000.0, co2=tab("co2", DEFAULT_CO2),
        delta=tab("delta", DEFAULT_DELTA), gamma=tab("gamma", DEFAULT_GAMMA),
        ra=tab("ra", DEFAULT_RA), sunrise=sunrise, sow_crop=sow[0], sow_mgt=sow[1],
        sow_mode=sow[2], sow_thr=sow[3], warm15=_warm_window(tmax, tmin),
        act_frac=None if replay is None else torch.as_tensor(replay["act_frac"], dtype=dtype,
                                                              device=device),
        lwp=None if replay is None else torch.as_tensor(replay["lwp"], dtype=dtype, device=device),
        doy=torch.as_tensor(doy, dtype=torch.long, device=device).reshape(-1),
        **{k: (None if replay is None or replay.get(k) is None else
               torch.as_tensor(replay[k], dtype=dtype, device=device)) for k in ("moist", "ice", "soil_t")})


class CropSystRun(object):
    """Everything a crop backend run holds: static data, the day tables, the state, and the day
    step. The state is replaced, never changed in place, so a proposal can be computed without
    touching the accepted one."""

    def __init__(self, S: Static, tables: DayTables, tile_cell, n_layer, device, P=400,
                 dtype=torch.float64):
        self.S, self.tables, self.device = S, tables, device
        self.tile_cell = torch.as_tensor(tile_cell, dtype=torch.long, device=device)
        self.n_tile = int(self.tile_cell.shape[0])
        self.n_layer = n_layer
        self.state = initial_state(S, self.n_tile, n_layer, P, dtype, device)

    @property
    def nitrogen(self):
        return self.S.n is not None

    def step(self, state=None, moist_mm=None, ice_mm=None, et_ref=None, soil_t=None, icpt_m=None):
        st = self.state if state is None else state
        d = self.tables.at(st.day, et_ref)
        if d.et_ref is None:
            raise ValueError("the day tables hold no et_ref; pass et_ref= each day")
        return day_step(st, self.S, d, moist_mm, ice_mm, soil_t, icpt_m)


# ----------------------------------------------------------------------------- setup from a config

def _per_tile(x, tile_cell, dtype, device):
    t = torch.as_tensor(x, dtype=dtype)
    tc = torch.as_tensor(tile_cell, dtype=torch.long)
    if t.dim() == 1:
        return t.expand(tc.shape[0], t.shape[0]).clone().to(device)
    return t[tc].to(device)


def build_run(cfg: dict, weather: dict, lat, lon, year, doy, n_cells, device,
              dtype=torch.float64, P=400) -> CropSystRun:
    """A CropSystRun from the `cropsyst` block of the crop connector.

    cfg keys:
      nitrogen      True: crop and soil nitrogen (n_engine; needs soil bulk_density, wcr,
                    wpwp, quartz, clay and each day the layer temperature), default False
      db_dir        <CropSyst Database> with Rotation/Code/<code>.CS_rotation
      rotations     [n_cell] rotation code of each cell; every cell gets one tile per cycle year
        or tiles    [T] runtime codes (rotation * 100 + phase) with tile_cell [T]
      first_year    calendar year of the first rotation year (default: year[0])
      soil          {thickness [L] or [n_cell, L] (m), and either quartz, clay (fractions) or
                     max_moist (mm), air_entry (J/kg), campbell_b}
      orchard_interception_model  the scenario's [submodels] orchard_interception_model for crops with
                    canopy_architecture=true: none, Kemanian-Stockle or Oyarzun-Stockle (default,
                    model_options.cpp:132)
      slope_percent, slope_aspect  terrain of those models (scalar or [n_cell]; default 0 % and 180
                    degrees, 0 north, 90 east; the VIC build's slope is 0)
      initial_moisture  fraction of the maximum moisture the profile holds on day 0 when no layer
                    moisture is linked yet (a lagged link; default 0.5)
      co2_response  CO2 transpiration/TUE response on (default True)
      clip_mode     'fixed' (default): a canopy portion a clipping may not remove (GAI to retain)
                    stays on the plant; 'exact': the reference pops it and loses it (clipping.py)
      replay        {act_frac, lwp} [n_day, T]: a reference run's water stress and leaf water
                    potential in place of E1 (validation); with nitrogen also moist, ice,
                    soil_t [n_day, T, L], the reference's layer moisture, ice (mm) and temperature
      soil (nitrogen) bulk_density (kg/m3) or bulk_density_g_cm3, wcr and wpwp (mm) or
                    field_capacity and wilting_point (volumetric), optional saturation: VIC's soil
    weather: see day_tables; year, doy: [n_day]."""
    nitrogen = check_nitrogen(cfg.get("nitrogen", False))
    unknown = set(cfg) - {"nitrogen", "db_dir", "rotations", "tiles", "tile_cell", "first_year",
                          "soil", "co2_response", "replay", "lon", "n_portions",
                          "initial_moisture", "initial_soil_temperature",
                          "orchard_interception_model", "slope_percent", "slope_aspect", "clip_mode"}
    if unknown:
        raise InputError(f"cropsyst: unknown option(s) {sorted(unknown)}")
    if "db_dir" not in cfg:
        raise InputError("cropsyst: give db_dir (the CropSyst Database with Rotation/Code)")
    if ("rotations" in cfg) == ("tiles" in cfg):
        raise InputError("cropsyst: give either rotations (one code per cell) or tiles with "
                         "tile_cell")
    year = torch.as_tensor(year, dtype=torch.long)
    doy_t = torch.as_tensor(doy, dtype=torch.long)
    first_year = int(cfg.get("first_year", int(year[0])))
    n_years = int(year.max()) - first_year + 1
    start_doy = int(doy_t[0])
    if "rotations" in cfg:
        codes = [cal.decompose_code(int(c))[0] for c in cfg["rotations"]]
        if len(codes) != n_cells:
            raise InputError(f"cropsyst: rotations has {len(codes)} codes for {n_cells} cells")
        uniq = sorted(set(codes))
        rs = cal.load_rotations(cfg["db_dir"], uniq, device, dtype, start_doy=start_doy)
        tile_cell, tile_rot, tile_phase = cal.expand_tiles(rs, codes)
        # expand_tiles indexes rs by the code's position in rs.codes
    else:
        dec = [cal.decompose_code(int(c)) for c in cfg["tiles"]]
        uniq = sorted({b for b, _ in dec})
        rs = cal.load_rotations(cfg["db_dir"], uniq, device, dtype, start_doy=start_doy)
        tile_cell = torch.as_tensor(cfg["tile_cell"], dtype=torch.long, device=device)
        tile_rot = torch.tensor([rs.index_of(b) for b, _ in dec], dtype=torch.long, device=device)
        tile_phase = torch.tensor([ph_ for _, ph_ in dec], dtype=torch.long, device=device)
    if rs.once is not None and bool(rs.once[tile_rot].any()) and first_year != int(year[0]):
        raise InputError("cropsyst: established crops are sown on the first simulation day; "
                         "first_year must be the first year of the run")
    calendar = cal.build_calendar(rs, tile_rot, tile_phase, first_year, n_years)
    slot_mode, slot_thr = _slot_modes(rs, tile_rot)
    sow = sowing_day_tables(calendar, year, doy_t, device, slot_mode, slot_thr)
    soil = dict(cfg.get("soil") or {})
    if "thickness" not in soil:
        raise InputError("cropsyst: soil needs thickness (m per layer)")
    tcell = tile_cell.cpu()
    thick = _per_tile(soil["thickness"], tcell, dtype, device)
    if "quartz" in soil:
        sx = saxton_soil(_per_tile(soil["quartz"], tcell, dtype, "cpu"),
                         _per_tile(soil["clay"], tcell, dtype, "cpu"), thick.cpu())
        sx = {k: v.to(device) for k, v in sx.items()}
    else:
        sx = {k: _per_tile(soil[k], tcell, dtype, device)
              for k in ("max_moist", "air_entry", "campbell_b")}
        if bool((sx["air_entry"] > 0).any()):
            raise InputError("cropsyst: soil air_entry is the air-entry potential in J/kg and must be "
                             "negative (a positive one gives a positive matric potential)")
    lat_c = torch.as_tensor(lat, dtype=torch.float64).reshape(-1)
    tile_lat = lat_c[tcell.clamp(max=lat_c.shape[0] - 1)] if lat_c.numel() else None
    S = build_static(rs, thick, sx["max_moist"], sx["air_entry"], sx["campbell_b"], device,
                     cfg.get("co2_response", True), dtype, tile_lat=tile_lat, year=year.tolist(),
                     doy=doy_t.tolist(), clip_mode=cfg.get("clip_mode", "fixed"))
    if nitrogen:
        from . import n_engine as ne
        nsoil = {k: _per_tile(soil[k], tcell, dtype, device) for k in
                 ("bulk_density", "bulk_density_g_cm3", "wcr", "wpwp", "field_capacity", "wilting_point",
                  "saturation", "quartz", "clay") if k in soil}
        S = S._replace(n=ne.build_n_static(rs, nsoil, thick, sx["max_moist"], sow[1], year.tolist(),
                                           doy_t.tolist(), device, dtype))
    lonv = cfg.get("lon", lon)
    tables = day_tables(weather, sow, tile_cell, lat, lonv, doy_t, device, dtype,
                        cfg.get("replay"))
    if S.orc is not None and S.orc.OP.arch is not None:
        from . import orchard_arch as oa
        # the land unit's site for the orchard interception models (orchard_arch.py)
        lon_c = torch.as_tensor(lonv, dtype=torch.float64).reshape(-1)
        cl = tcell.clamp(max=lon_c.shape[0] - 1)
        per_cell = lambda v: (torch.as_tensor(v, dtype=torch.float64).reshape(-1)[cl]
                              if torch.as_tensor(v).numel() > 1 else v)
        AS = oa.arch_static(cfg.get("orchard_interception_model", oa.DEFAULT_MODEL), tile_lat,
                               lon_c[cl], per_cell(cfg.get("slope_percent", 0.0)),
                               per_cell(cfg.get("slope_aspect", 180.0)), tables.srad[0].cpu(),
                               int(doy_t[0]), S.orc.OP.arch, device, dtype)
        S = S._replace(orc=S.orc._replace(arch=AS))
    return CropSystRun(S, tables, tile_cell, int(thick.shape[-1]), device,
                       int(cfg.get("n_portions", P)), dtype)
