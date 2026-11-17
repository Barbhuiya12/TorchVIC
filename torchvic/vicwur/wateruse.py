"""VIC-WUR water-use core (C4) and local water use (C5): pure tensor kernels.

Every function takes tensors and returns tensors; batched over cells
(leading dim C) and sectors (S=5, enum order MUN, ENE, MAN, LIV, IRR =
priority order).  No loop over cells and no host sync.  The only loops are
over the (5) sectors, where the C code is sequential in sector priority.

C references are VIC-WUR 3.0 plugins/wateruse/src
(<data root>/build/vicwur3/src_tree/vic):
  wu_run.c:77-111           demand
  wu_run_local.c:33-54      cell demand totals
  wu_run_local.c:95-135     groundwater availability
  wu_run_local.c:139-153    river availability (whole ring, EFR)
  wu_run_local.c:155-170    local dam availability
  wu_run_local.c:194-249    pro-rata division over sectors
  wu_run_local.c:271-345    withdrawal, consumption, returns
  wu_run_local.c:397-436    groundwater pumping
  wu_run_local.c:439-469    deficit payback from returns
  wu_run_local.c:543-562    local dam withdrawal
  wu_nonrenewable.c:54-65   unmet demand
The ring edit (wu_run_local.c:471-541) belongs to the routing level kernel
and is not here.

The C macro min(a, b) is `(a < b) ? a : b`: NaN in `a` gives `b`.  `cmin`
reproduces it.  Inactive sectors (sidx == NODATA_WU) are skipped by the C
loops; here `active` [C, S] bool masks them and their outputs stay 0.
VIC-WUR has no reservoir-first rule: river and local dams form one pool and
are shared pro rata.
"""
import torch

MUN, ENE, MAN, LIV, IRR = range(5)
NSECT = 5
MM_PER_M = 1000.0
M3_PER_HM3 = 1e6


def cmin(a, b):
    """C `min` macro: (a < b) ? a : b (NaN in a gives b)."""
    return torch.where(a < b, a, b)


def _act(active, like):
    if active is None:
        return torch.ones_like(like, dtype=torch.bool)
    return active


def demand(dem, gw_frac, active=None, gwm=False):
    """Sector demand split (wu_run.c:94-109).  dem, gw_frac [C, S] (mm).

    Returns (demand_gw, demand_surf); inactive sectors give 0.  gwm=True is
    the VIC-WUR 3.0 GWM branch: demand_gw = 0, demand_surf = demand.
    """
    act = _act(active, dem)
    zero = torch.zeros_like(dem)
    if gwm:
        dgw, dsf = zero, dem
    else:
        dgw, dsf = dem * gw_frac, dem * (1 - gw_frac)
    return torch.where(act, dgw, zero), torch.where(act, dsf, zero)


def cell_demand(dgw, dsf, active=None):
    """Cell totals over active sectors, in sector order (wu_run_local.c:45-53)."""
    act = _act(active, dgw)
    zero = torch.zeros_like(dgw)
    tg = torch.zeros_like(dgw[..., 0])
    ts = torch.zeros_like(tg)
    for s in range(dgw.shape[-1]):
        tg = tg + torch.where(act[..., s], dgw[..., s], zero[..., s])
        ts = ts + torch.where(act[..., s], dsf[..., s], zero[..., s])
    return tg, ts


def availability_gw(moist, ice, frost_fract, resid_mm, area_fract, cv,
                    efr_moist=None):
    """Groundwater availability of the bottom layer (wu_run_local.c:95-135).

    moist [C,V,B] bottom-layer moisture (mm); ice [C,V,B,F]; frost_fract
    [C,F] or [F]; resid_mm [C] = resid_moist[L] * depth[L] * 1000;
    area_fract [C,B]; cv [C,V]; efr_moist [C,V,B] or None (EFR off).
    Returns (av_gw [C,V,B], A_gw [C]).  Tiles with Cv <= 0 or AreaFract <= 0
    are skipped by C and stay 0.  The caller passes bottom-layer state only;
    the GWM=TRUE skip of this block is the caller's (A_gw = 0).
    """
    ff = frost_fract if frost_fract.dim() > 1 else frost_fract.expand(
        moist.shape[0], -1)
    av = moist
    if efr_moist is not None:
        av = av - efr_moist
    av = av - resid_mm[:, None, None]
    # ice = sum_f ice[f] * frost_fract[f], summed in frost order
    tot = torch.zeros_like(moist)
    for f in range(ice.shape[-1]):
        tot = tot + ice[..., f] * ff[:, None, None, f]
    av = av - tot
    av = av * area_fract[:, None, :] * cv[:, :, None]
    av = torch.where(av < 0, torch.zeros_like(av), av)
    valid = (cv[:, :, None] > 0) & (area_fract[:, None, :] > 0)
    av = torch.where(valid, av, torch.zeros_like(av))
    return av, av.flatten(1).sum(1)


def availability_surf(dt_discharge, dt, area, efr_q=None, discharge=None):
    """River availability: the WHOLE ring (wu_run_local.c:139-153).

    dt_discharge [C,K] (m3/s); dt [s]; area [C] (m2).  With EFR:
    frac = min(efr_q/discharge, 1) by the C macro, so discharge = 0 gives 1.
    Returns A_sf [C] (mm), clipped at 0.
    """
    a = (dt_discharge * dt / area[:, None] * MM_PER_M).sum(1)
    if efr_q is not None:
        r = efr_q / discharge
        frac = torch.where(r < 1, r, torch.ones_like(r))
        a = a * (1 - frac)
    return torch.where(a < 0, torch.zeros_like(a), a)


def availability_dam(storage, active, is_local, area):
    """Local dam availability (wu_run_local.c:155-170).

    storage [C,D] (hm3); active, is_local [C,D] bool; area [C] (m2).
    Returns (av_dam [C,D] mm, A_dam [C]).
    """
    m = active & is_local
    av = storage * M3_PER_HM3 / area[:, None] * MM_PER_M
    av = torch.where(m, av, torch.zeros_like(av))
    a = av.sum(1)
    return av, torch.where(a < 0, torch.zeros_like(a), a)


def division(dgw, dsf, a_gw, a_sf, a_dam, tot_dgw, tot_dsf, active=None):
    """Pro-rata division over sectors (wu_run_local.c:196-248).

    dgw, dsf [C,S]; a_gw, a_sf, a_dam, tot_dgw, tot_dsf [C].
    Groundwater is sequential by sector priority.  Surface: when the sector
    demand reaches river+dam, take both pools whole, else take
    demand*pool/total from each (proportional, not river-first).
    Returns avail_gw, avail_surf, avail_dam [C,S] (0 where unassigned).
    """
    act = _act(active, dgw)
    zeros = torch.zeros_like(dgw[..., 0])
    gw_t, sf_t, dm_t = a_gw, a_sf, a_dam
    ogw, osf, odm = [], [], []
    gate_g = tot_dgw > 0
    gate_s = tot_dsf > 0
    for s in range(dgw.shape[-1]):
        ok = act[..., s]
        d_g, d_s = dgw[..., s], dsf[..., s]
        # groundwater
        take = d_g >= gw_t
        v = torch.where(take, gw_t, d_g)
        m = ok & gate_g
        ogw.append(torch.where(m, v, zeros))
        gw_t = torch.where(m, torch.where(take, zeros, gw_t - d_g), gw_t)
        # surface water and dam
        tot = sf_t + dm_t
        full = d_s >= tot
        # tot == 0 with demand < 0 gives 0/0 = NaN, as in C (negative guard)
        ps = d_s * (sf_t / tot)
        pd = d_s * (dm_t / tot)
        m = ok & gate_s
        osf.append(torch.where(m, torch.where(full, sf_t, ps), zeros))
        odm.append(torch.where(m, torch.where(full, dm_t, pd), zeros))
        sf_t = torch.where(m, torch.where(full, zeros, sf_t - ps), sf_t)
        dm_t = torch.where(m, torch.where(full, zeros, dm_t - pd), dm_t)
    return torch.stack(ogw, -1), torch.stack(osf, -1), torch.stack(odm, -1)


def use(dgw, dsf, avail_gw, avail_surf, avail_dam, cons_frac,
        returned_in=None, consumed_in=None, active=None):
    """Withdrawal, consumption and returns (wu_run_local.c:271-345).

    All [C,S] except nothing else.  returned_in/consumed_in are values
    already in wu_var (C adds with +=; the irrigation sector may have set
    them); default 0.  Returns a dict: withdrawn_gw/surf/dam, returned,
    consumed [C,S] and the cell sums cell_gw/cell_surf/cell_dam/cell_returned
    [C], summed in sector order as C does.  The 0/0 -> NaN -> macro -> 1 path
    (zero river or zero dam share) is reproduced and harmless.
    """
    act = _act(active, dgw)
    z = torch.zeros_like(dgw)
    one = torch.ones_like(dgw)
    ret = z if returned_in is None else returned_in
    con = z if consumed_in is None else consumed_in
    # groundwater
    pos = avail_gw > 0
    f = dgw / avail_gw
    f = torch.where(f < 1, f, one)
    wgw = torch.where(pos, avail_gw * f, z)
    ret = ret + wgw * (1 - cons_frac)
    con = con + wgw * cons_frac
    # surface and dams
    tot = avail_surf + avail_dam
    pos = tot > 0
    f = dsf * (avail_surf / tot) / avail_surf
    f = torch.where(f < 1, f, one)
    wsf = torch.where(pos, avail_surf * f, z)
    f = dsf * (avail_dam / tot) / avail_dam
    f = torch.where(f < 1, f, one)
    wdm = torch.where(pos, avail_dam * f, z)
    ret = ret + wsf * (1 - cons_frac)
    ret = ret + wdm * (1 - cons_frac)
    con = con + wsf * cons_frac
    con = con + wdm * cons_frac
    wgw, wsf, wdm = [torch.where(act, x, z) for x in (wgw, wsf, wdm)]
    ret = torch.where(act, ret, z if returned_in is None else returned_in)
    con = torch.where(act, con, z if consumed_in is None else consumed_in)
    cg = torch.zeros_like(wgw[..., 0])
    cs, cd, cr = cg, cg, cg
    for s in range(wgw.shape[-1]):
        m = act[..., s]
        cg = cg + torch.where(m, wgw[..., s], 0.0)
        cs = cs + torch.where(m, wsf[..., s], 0.0)
        cd = cd + torch.where(m, wdm[..., s], 0.0)
        cr = cr + torch.where(m, ret[..., s], 0.0)
    return dict(withdrawn_gw=wgw, withdrawn_surf=wsf, withdrawn_dam=wdm,
                returned=ret, consumed=con, cell_gw=cg, cell_surf=cs,
                cell_dam=cd, cell_returned=cr)


def unmet(dgw, dsf, wgw, wsf, wdm, wtremote=None):
    """Unmet demand per sector = non-renewable demand (wu_nonrenewable.c:55-64):
    max(0, demand_surf + demand_gw - withdrawn_surf - withdrawn_gw -
    withdrawn_dam - withdrawn_tremote)."""
    w = wsf + wgw + wdm
    if wtremote is not None:
        w = w + wtremote
    d = (dsf + dgw) - w
    return torch.where(d < 0, torch.zeros_like(d), d)


def pump_depth(av_gw, a_gw, w_gw_cell, area_fract, cv):
    """What `pump` takes from every tile and band, mm: (d [C,V,B], valid
    [C,V,B]); d = W/(AF Cv) av/A_gw where the pump runs (wu_run_local.c:
    398-419), 0 elsewhere."""
    af = area_fract[:, None, :]
    c = cv[:, :, None]
    valid = (c > 0) & (af > 0) & (w_gw_cell > 0)[:, None, None]
    den = torch.where(valid, af * c, torch.ones_like(af * c))
    d = w_gw_cell[:, None, None] / den * av_gw / a_gw[:, None, None]
    return torch.where(valid, d, torch.zeros_like(d)), valid


def pump(moist, av_gw, a_gw, w_gw_cell, area_fract, cv):
    """Groundwater pumping from the bottom layer (wu_run_local.c:398-419).

    moist, av_gw [C,V,B]; a_gw, w_gw_cell [C]; returns new moist, clipped at
    0.  Applied only when w_gw_cell > 0 and the tile has Cv > 0 and
    AreaFract > 0.  The ice clamp that follows in C (:421-433) is dead code.
    """
    d, valid = pump_depth(av_gw, a_gw, w_gw_cell, area_fract, cv)
    new = moist - d
    new = torch.where(new < 0, torch.zeros_like(new), new)
    return torch.where(valid, new, moist)


def payback(deficit, returned_sector, returned_cell, active=None):
    """Deficit payback from returns (wu_run_local.c:439-469).

    deficit [C] (non-renewable deficit, mm); returned_sector [C,S] (wu_var
    returned, NOT reduced); returned_cell [C] (cell sum).  Gated on
    returned_cell > 0.  Per sector in order x = min(deficit, returned[s]);
    deficit -= x when deficit > 0 (clipped at 0); returned_cell -= x (clipped
    at 0).  Returns (deficit, returned_cell) after the loop.
    """
    act = _act(active, returned_sector)
    gate = returned_cell > 0
    d, r = deficit, returned_cell
    zero = torch.zeros_like(d)
    for s in range(returned_sector.shape[-1]):
        m = gate & act[..., s]
        x = cmin(d, returned_sector[..., s])
        nd = torch.where(d > 0, d - x, d)
        nd = torch.where(nd < 0, zero, nd)
        nr = r - x
        nr = torch.where(nr < 0, zero, nr)
        d = torch.where(m, nd, d)
        r = torch.where(m, nr, r)
    return d, r


def dam_withdraw(storage, av_dam, a_dam, w_dam_cell, active, is_local, area):
    """Local dam storage after withdrawal (wu_run_local.c:544-561), hm3.

    storage, av_dam [C,D]; a_dam, w_dam_cell, area [C]; clipped at 0.
    """
    valid = active & is_local & (w_dam_cell > 0)[:, None]
    safe = torch.where(a_dam == 0, torch.ones_like(a_dam), a_dam)
    d = (w_dam_cell[:, None] * (av_dam / safe[:, None]) / MM_PER_M
         * area[:, None] / M3_PER_HM3)
    new = storage - d
    new = torch.where(new < 0, torch.zeros_like(new), new)
    return torch.where(valid, new, storage)


def wu_local_step(dem, gw_frac, cons_frac, moist, ice, frost_fract, resid_mm,
                  area_fract, cv, dt_discharge, dt, area, dam_storage=None,
                  dam_active=None, dam_local=None, efr_moist=None, efr_q=None,
                  discharge=None, active=None, deficit=None, returned_in=None,
                  consumed_in=None, gwm=False):
    """Whole C5 chain of wu_run_local.c up to (not including) the ring edit.

    Returns a dict of all intermediate and updated fields.
    """
    act = _act(active, dem)
    dgw, dsf = demand(dem, gw_frac, act, gwm)
    tg, ts = cell_demand(dgw, dsf, act)
    zc = torch.zeros_like(tg)
    if gwm:
        av_gw, a_gw = torch.zeros_like(moist), zc
    else:
        av_gw, a_gw = availability_gw(moist, ice, frost_fract, resid_mm,
                                      area_fract, cv, efr_moist)
    a_sf = availability_surf(dt_discharge, dt, area, efr_q, discharge)
    if dam_storage is not None:
        av_dam, a_dam = availability_dam(dam_storage, dam_active, dam_local,
                                         area)
    else:
        av_dam, a_dam = None, zc
    ag, asf, adm = division(dgw, dsf, a_gw, a_sf, a_dam, tg, ts, act)
    u = use(dgw, dsf, ag, asf, adm, cons_frac, returned_in, consumed_in, act)
    out = dict(demand_gw=dgw, demand_surf=dsf, A_gw=a_gw, A_surf=a_sf,
               A_dam=a_dam, av_gw=av_gw, av_dam=av_dam, available_gw=ag,
               available_surf=asf, available_dam=adm, **u)
    out['unmet'] = unmet(dgw, dsf, u['withdrawn_gw'], u['withdrawn_surf'],
                         u['withdrawn_dam'])
    out['moist'] = moist if gwm else pump(moist, av_gw, a_gw, u['cell_gw'],
                                          area_fract, cv)
    r = u['cell_returned']
    if deficit is not None:
        out['deficit'] = deficit
        if not gwm:
            out['deficit'], r = payback(deficit, u['returned'], r, act)
    out['returned_to_river'] = r
    if dam_storage is not None:
        out['dam_storage'] = dam_withdraw(dam_storage, av_dam, a_dam,
                                          u['cell_dam'], dam_active,
                                          dam_local, area)
    return out
