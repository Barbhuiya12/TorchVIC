# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR water balance and per-cell outputs (C16): pure functions.

Self-contained: only torch.  Every function takes tensors (any leading cell
shape, float64) and returns tensors.  No host sync; the only Python loops run
over the 5 water-use sectors and the soil layers, to keep the C summation
order.  The calendar part of aggregation (which step belongs to which output
period) is computed once on the host (`agg_periods`) and passed as tensors.

C references are VIC-WUR 3.0 (<data root>/build/vicwur3/src_tree/vic):
  plugins/general/src/plugin_store_error.c:33-148  plugin_store_error
      :53-70 inflow, :72-100 outflow, :102-121 storage, :123-129 error and
      stored storage, :131-147 correction of VIC's save_data
  drivers/shared_all/src/put_data.c:533-554   VIC OUT_WATER_ERROR
      (inflow/outflow/storage, calc_water_balance_error :1087-1099)
  drivers/shared_image/src/vic_image_run.c:76-82   put_data, then
      plugin_put_data_all (plugin_simulation.c:172-217; store_error last :217)
  drivers/shared_image/src/vic_init_output.c:68-70  plugin_put_data at init
      (so the first step's previous storage is the initial plugin storage)
  plugins/dams/src/dam_run.c:87-90   local dam rewrites out_data runoff
      after put_data (local_dam_runoff_original undoes it)
  drivers/shared_all/src/agg_data.c:47-103   aggregation of a stream
  drivers/shared_all/src/vic_history.c:233-332   default type (AVG; SUM for
      BASEFLOW/GWRECHARGE...), then plugin_output.c:204-237 overrides:
      plugin_output.c:209-210 ROUTING_ERROR SUM,
      rout_output.c:105-112, dam_output.c:272-294, efr_output.c:82-89,
      wu_output.c:312-327, irr_output.c:102-111, lu_output.c:95-96,
      crop_output.c:687-726
Where the design doc (docs/connectors/vicwur.md) and the C differ the C
is followed: the options gate each term, GWM=true drops WI_NREN from the
outflow and keeps the groundwater withdrawal in the storage correction.
"""
import torch

MM_PER_M = 1000.
M3_PER_HM3 = 1000000.
WU_NSECTORS = 5


# ------------------------------------------------------------ OUT_ROUTING_ERROR
def routing_balance(o, area, dt, prev_storage, **flags):
    """plugin_store_error.c:33-129 -> (OUT_ROUTING_ERROR, storage) [mm].
    Flags as for `routing_terms`.  See there for `o`, `area` and `dt`.
    `prev_storage` is plugin_save_data.total_moist_storage; the returned
    storage replaces it."""
    inflow, outflow, storage = routing_terms(o, area, dt, **flags)
    return inflow - outflow - (storage - prev_storage), storage


def routing_terms(o, area, dt, *, routing=True, dams=False,
                  irrigation=False, wateruse=False, nonrenew_with=False,
                  gwm=False):
    """plugin_store_error.c:53-121 -> (inflow, outflow, storage) [mm].

    `o` maps C output names (without OUT_) to tensors over cells, the
    sector ones with a trailing sector axis of 5: STREAM_RUNOFF,
    STREAM_INFLOW, DISCHARGE, STREAM_MOIST, NONREN_DEFICIT (m3/s and mm),
    LDAM_INFLOW, LDAM_RELEASE, GDAM_STORAGE, LDAM_STORAGE (hm3), RECEIVED,
    RETURNED, APPLIED, LEFTOVER (mm), WI_SURF_SECT, WI_DAM_SECT,
    WI_REM_SECT, WI_NREN_SECT (mm).  Only the ones the options use are read.
    `area` is m2 and `dt` the model step in s.  The C operation order
    (left to right: x * dt / area * MM_PER_M) is kept.
    """
    inflow = torch.zeros_like(area)
    if routing:
        inflow = inflow + (o['STREAM_RUNOFF'] + o['STREAM_INFLOW']) * dt \
            / area * MM_PER_M
    if dams:
        inflow = inflow + o['LDAM_INFLOW'] * M3_PER_HM3 / area * MM_PER_M
    if irrigation:
        inflow = inflow + o['RECEIVED']
    if wateruse:
        inflow = inflow + o['RETURNED']

    outflow = torch.zeros_like(area)
    if routing:
        outflow = outflow + o['DISCHARGE'] * dt / area * MM_PER_M
    if dams:
        outflow = outflow + o['LDAM_RELEASE'] * M3_PER_HM3 / area * MM_PER_M
    if irrigation:
        outflow = outflow + o['APPLIED']
    if wateruse:
        for i in range(WU_NSECTORS):
            s = o['WI_SURF_SECT'][..., i] + o['WI_DAM_SECT'][..., i] \
                + o['WI_REM_SECT'][..., i]
            if not gwm:
                s = s + o['WI_NREN_SECT'][..., i]
            outflow = outflow + s

    storage = torch.zeros_like(area)
    if routing or (wateruse and nonrenew_with):
        if gwm:
            storage = storage + o['STREAM_MOIST']
        else:
            storage = storage + (o['STREAM_MOIST'] - o['NONREN_DEFICIT'])
    if dams:
        storage = storage + (o['GDAM_STORAGE'] + o['LDAM_STORAGE']) \
            * M3_PER_HM3 / area * MM_PER_M
    if irrigation:
        storage = storage + o['LEFTOVER']

    return inflow, outflow, storage


# ----------------------------------------------- VIC OUT_WATER_ERROR (+ fix)
def vic_water_error(prec, lake_chan_in, evap, runoff, gwrecharge, baseflow,
                    soil_liq, soil_ice, swe, snow_canopy, wdew, surfstor,
                    prev_storage):
    """put_data.c:533-554 -> (OUT_WATER_ERROR, storage) [mm].  soil_liq and
    soil_ice have the layer axis last (summed layer by layer, in order)."""
    inflow = prec + lake_chan_in
    outflow = evap + runoff + gwrecharge + baseflow
    storage = torch.zeros_like(prec)
    for l in range(soil_liq.shape[-1]):
        storage = storage + (soil_liq[..., l] + soil_ice[..., l])
    storage = storage + swe + snow_canopy + wdew + surfstor
    return inflow - outflow - (storage - prev_storage), storage


def local_dam_runoff_original(runoff_out, ldam_inflow, ldam_release,
                              inflow_frac, area):
    """The cell runoff VIC saw in put_data (and in OUT_WATER_ERROR) from the
    OUT_RUNOFF a run writes.  An operating local dam rewrites out_data runoff
    AFTER put_data (dam_run.c:87-90: runoff *= 1 - frac; runoff += release
    hm3 -> mm), so the history holds the post-dam value while OUT_WATER_ERROR
    used the original.  A dam that is not operating (no registered inflow and
    no release, e.g. before its start year) leaves the runoff alone.
    `ldam_inflow` and `ldam_release` are hm3, `inflow_frac` 0 in cells
    without a local dam."""
    rel_mm = ldam_release * M3_PER_HM3 * MM_PER_M / area
    orig = (runoff_out - rel_mm) / (1. - inflow_frac)
    return torch.where((ldam_inflow > 0) | (ldam_release > 0), orig,
                       runoff_out)


def water_error_correction(save_moist_storage, save_soil_moist, applied,
                           wi_gw_sect, *, irrigation, wateruse, gwm=False):
    """plugin_store_error.c:131-147: the adaptation of VIC's save_data after
    the plugins moved water (irrigation applied into the soil, groundwater
    withdrawn out of it) so that VIC's OUT_WATER_ERROR of the NEXT step does
    not see them.  -> (total_moist_storage, total_soil_moist), new tensors."""
    sm, ss = save_moist_storage, save_soil_moist
    if irrigation:
        sm = sm + applied
        ss = ss + applied
    if wateruse and not gwm:
        for i in range(WU_NSECTORS):
            ss = ss - wi_gw_sect[..., i]
            sm = sm - wi_gw_sect[..., i]
    return sm, ss


# ------------------------------------------------- aggregation types (C)
SUM, AVG, END, MAX, MIN, BEG = 'sum', 'avg', 'end', 'max', 'min', 'beg'

_AGG = {}
for _n in ('ROUTING_ERROR',):                                    # :209-210
    _AGG[_n] = SUM
for _n in ('STREAM_MOIST', 'NONREN_DEFICIT'):                    # rout :105-107
    _AGG[_n] = END
for _n in ('STREAM_RUNOFF', 'STREAM_INFLOW', 'DISCHARGE'):       # rout :109-112
    _AGG[_n] = AVG
for _n in ('LDAM_STORAGE', 'GDAM_STORAGE', 'GDAM_OP_STORAGE',
           'LDAM_OP_STORAGE'):                                   # dam :272-276
    _AGG[_n] = END
for _p in ('LDAM', 'GDAM'):                                      # dam :278-294
    for _n in ('HIST_INFLOW', 'HIST_DEMAND', 'HIST_EFR', 'OP_RELEASE',
               'INFLOW', 'DEMAND', 'EFR', 'RELEASE'):
        _AGG[_p + '_' + _n] = SUM
_AGG['EFR_MOIST'] = END                                          # efr :82-83
_AGG['EFR_BASEFLOW'] = SUM                                       # efr :85-86
_AGG['EFR_DISCHARGE'] = AVG                                      # efr :88-89
for _n in ('DE_GW_SECT', 'DE_SURF_SECT', 'DE_NREN_SECT', 'DE_REM_SECT',
           'DE_TREM_SECT', 'WI_GW_SECT', 'WI_SURF_SECT', 'WI_DAM_SECT',
           'WI_REM_SECT', 'WI_TREM_SECT', 'WI_NREN_SECT', 'DEMAND',
           'WITHDRAWN', 'RETURNED', 'CONSUMED'):                 # wu :312-327
    _AGG[_n] = SUM
for _n in ('REQUIREMENT', 'SHORTAGE'):                           # irr :102-104
    _AGG[_n] = AVG
for _n in ('RECEIVED', 'APPLIED'):                               # irr :106-108
    _AGG[_n] = SUM
_AGG['LEFTOVER'] = END                                           # irr :110-111
_AGG['CV'] = AVG                                                 # lu :95-96
for _n in ('CULTIVATE GROW WSTRESS NSTRESS TMAXSTRESS TMINSTRESS NNI PNI KNI '
           'NPKI EVAP EVAP_BARE EVAP_CANOP TRANSP_VEG DVS WLV WST WSO WRT '
           'WDLV WDST WDRT NLV NST NSO NRT NSOIL PSOIL KSOIL NUPT PUPT KUPT '
           'NDEM PDEM KDEM ROOTDEPTH').split():                  # crop :687-723
    _AGG['CROP_' + _n] = SUM
_AGG['CROP_LAI'] = MAX                                           # crop :725-726
# VIC core variables: default AVG except these (vic_history.c:238-326)
_CORE_END = ('''
    ASAT LAKE_AREA_FRAC LAKE_DEPTH LAKE_ICE LAKE_ICE_FRACT
    LAKE_ICE_HEIGHT LAKE_MOIST LAKE_SURF_AREA LAKE_SWE LAKE_SWE_V
    LAKE_VOLUME ROOTMOIST SMFROZFRAC SMLIQFRAC SNOW_CANOPY SNOW_COVER
    SNOW_DEPTH SOIL_ICE SOIL_LIQ SOIL_MOIST SOIL_EFF_SAT SOIL_WET
    SURFSTOR SURF_FROST_FRAC SWE WDEW ZWT ZWT_LUMPED SNOW_CANOPY_BAND
    SNOW_COVER_BAND SNOW_DEPTH_BAND SWE_BAND''').split()
_CORE_SUM = ('''
    GWRECHARGE RECHARGE BASEFLOW DELINTERCEPT DELSOILMOIST DELSWE
    DELSURFSTOR EVAP EVAP_BARE EVAP_CANOP INFLOW LAKE_BF_IN
    LAKE_BF_IN_V LAKE_BF_OUT LAKE_BF_OUT_V LAKE_CHAN_IN LAKE_CHAN_IN_V
    LAKE_CHAN_OUT LAKE_CHAN_OUT_V LAKE_DSTOR LAKE_DSTOR_V LAKE_DSWE
    LAKE_DSWE_V LAKE_EVAP LAKE_EVAP_V LAKE_PREC_V LAKE_RCHRG
    LAKE_RCHRG_V LAKE_RO_IN LAKE_RO_IN_V LAKE_VAPFLX LAKE_VAPFLX_V PET
    PREC RAINF REFREEZE RUNOFF SNOWF SUB_BLOWING SUB_CANOP SUB_SNOW
    SUB_SURFACE TRANSP_VEG DELTACC_BAND SNOW_MELT SNOWT_FBFLAG
    SOILT_FBFLAG SURFT_FBFLAG TCAN_FBFLAG TFOL_FBFLAG''').split()


def agg_type(name):
    """C aggregation type of an OUTVAR (name with or without 'OUT_')."""
    if name.startswith('OUT_'):
        name = name[4:]
    if name in _AGG:
        return _AGG[name]
    if name in _CORE_END:
        return END
    if name in _CORE_SUM:
        return SUM
    return AVG


def _closes(day, freq, n, first):
    """True when `day` is the last day of an output period of `freq`/`n`
    (the next calendar day starts a new period)."""
    import calendar
    y, m, d = day
    if d != calendar.monthrange(y, m)[1]:
        return False
    k = (y - first[0]) * 12 + (m - first[1])
    if freq == 'months':
        return (k + 1) % n == 0
    return m == 12 and ((k // 12) + 1) % n == 0


def agg_periods(dmy, freq, n, device=None):
    """Host-side calendar of one output stream (set once; indexes the steps).

    dmy: list of (year, month, day) per step; freq 'steps'/'days' (n steps or
    days; one step per day here) or 'months'/'years' (n calendar months or
    years counted from the first step; the period closes on the last day of
    its last month, as the C alarm; at n > 1 the period must be full).  Returns (seg [T] long, last [P] long, count [P] long):
    the period of every step, the step that closes each period, and its
    length.  A trailing incomplete period is dropped as in C (never written):
    its steps get seg = -1."""
    T = len(dmy)
    seg = []
    if freq in ('steps', 'days'):
        seg = [t // n for t in range(T)]
    else:
        y0, m0, _ = dmy[0]
        for (y, m, d) in dmy:
            k = (y - y0) * 12 + (m - m0)
            seg.append(k // n if freq == 'months' else (k // 12) // n)
    # a period is complete if its closing step exists: for days when it has
    # n steps, for months/years when the next step opens a new period
    P = max(seg) + 1
    count = [0] * P
    last = [0] * P
    for t, s in enumerate(seg):
        count[s] += 1
        last[s] = t
    complete = []
    for s in range(P):
        if freq in ('steps', 'days'):
            ok = count[s] == n
        else:
            ok = _closes(dmy[last[s]], freq, n, dmy[0])
        complete.append(ok)
    remap, k = {}, 0
    for s in range(P):
        if complete[s]:
            remap[s] = k
            k += 1
    seg2 = [remap.get(s, -1) for s in seg]
    last2 = [last[s] for s in range(P) if complete[s]]
    count2 = [count[s] for s in range(P) if complete[s]]
    kw = dict(device=device, dtype=torch.long)
    return (torch.tensor(seg2, **kw), torch.tensor(last2, **kw),
            torch.tensor(count2, **kw))


def aggregate(x, kind, seg, last, count):
    """agg_data.c:47-103 over a whole run: x [T, ...] -> [P, ...].

    SUM/AVG add the steps in order from 0 (AVG then divides by the period's
    step count), MAX/MIN start from 0 as the C aggdata does (so MAX never
    reports below 0 and MIN never above 0), END takes the closing step, BEG
    the first.  Steps with seg < 0 (an incomplete last period) are dropped.
    Pure device ops; the segment tensors come from `agg_periods`."""
    P = last.shape[0]
    keep = seg >= 0
    # an incomplete tail is excluded by routing it to a dummy segment P
    sg = torch.where(keep, seg, torch.full_like(seg, P))
    shape = (P + 1,) + tuple(x.shape[1:])
    if kind in (SUM, AVG):
        out = torch.zeros(shape, dtype=x.dtype, device=x.device)
        out.index_add_(0, sg, x)
        out = out[:P]
        if kind == AVG:
            c = count.to(x.dtype).reshape((P,) + (1,) * (x.dim() - 1))
            out = out / c
        return out
    if kind in (MAX, MIN):
        out = torch.zeros(shape, dtype=x.dtype, device=x.device)
        idx = sg.reshape((-1,) + (1,) * (x.dim() - 1)).expand_as(x)
        out.scatter_reduce_(0, idx, x, 'amax' if kind == MAX else 'amin',
                            include_self=True)
        return out[:P]
    if kind == END:
        return x[last]
    if kind == BEG:
        return x[last - count + 1]
    raise ValueError(kind)


def aggregate_named(name, x, seg, last, count):
    """aggregate() with the C type of OUTVAR `name`."""
    return aggregate(x, agg_type(name), seg, last, count)
