# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR non-renewable groundwater withdrawal (C7): pure tensor kernels.

Demand that local water use (C5) and remote water use (C6) leave unmet is
served from a virtual non-renewable aquifer whose depletion is tracked as a
deficit [mm]. The deficit is repaid from the next steps' baseflow (and
runoff), and from the return flow of the non-renewable withdrawal itself.
See docs/connectors/vicwur.md.

Every function takes tensors and returns tensors; batched over cells (leading
dim C) and sectors (S=5, enum order MUN, ENE, MAN, LIV, IRR = priority
order).  No loop over cells and no host sync.  The only loop is over the five
sectors, where the C code is sequential in sector priority.

C references are VIC-WUR 3.0 (<data root>/build/vicwur3/src_tree/vic/plugins):
  wateruse/src/wu_nonrenewable.c:33-67    calculate_demand_nonrenew
  wateruse/src/wu_nonrenewable.c:72-85    calculate_availability_nonrenew
  wateruse/src/wu_nonrenewable.c:90-130   calculate_division_nonrenew
  wateruse/src/wu_nonrenewable.c:135-181  calculate_use_nonrenew
  wateruse/src/wu_nonrenewable.c:186-306  calculate_hydrology_nonrenew
  wateruse/src/wu_nonrenewable.c:311-354  check_water_use_balance_nonrenew
  wateruse/src/wu_nonrenewable.c:359-407  wu_nonrenew
  routing/src/rout_run.c:49-82            payback from baseflow and runoff
  wateruse/src/wu_run.c:67-69             per-step reset of the three fields

Quirks reproduced (mode `exact`; none of them is a mass-creating bug, so
`fixed` is identical):
  * ENERGY is skipped in the demand (:45-47) but not in the division and use
    loops (:106-180); its demand is 0, so it receives 0.
  * `availability = NREN_LIM - deficit` caps the stock, not the rate
    (:79-84); NREN_LIM defaults to Inf (plugin_init_library.c:115).
  * The division is always sequential, never proportional (:104-129), and is
    gated on the cell total demand_nonrenew > 0 (:113).
  * The immediate payback reads the deficit AFTER this step's withdrawal was
    added (:211-217), sector by sector (:228), so a withdrawal only grows the
    deficit by its consumed part.  It is gated on the cell `returned` > 0 and
    reduces `returned` by the sector's repaid amount, floored at 0 (:247-250).
  * What is left goes to the FUTURE ring bins only (:258-289), so it reaches
    the downstream cell at t+1 at the earliest.
  * options.GWM (VIC-WUR 3.0 GWM) books no deficit and repays nothing
    (:211,220; rout_run.c:49-51,56).

The C macro `min(a, b)` is `(a < b) ? a : b` (vic_def.h:91): NaN in `a` gives
`b`.  `cmin` is wateruse.cmin.
"""
import torch

from .wateruse import ENE, NSECT, MM_PER_M, cmin

__all__ = ['demand', 'availability', 'division', 'use', 'book_and_repay',
           'ring_return', 'payback_flow', 'runoff_to_discharge',
           'balance_violation', 'nonrenew_step', 'conservation_residual']


def _act(active, like):
    if active is None:
        return torch.ones_like(like, dtype=torch.bool)
    return active


def _nz(x):
    """C `if (x < 0) x = 0` (NaN stays NaN)."""
    return torch.where(x < 0, torch.zeros_like(x), x)


def demand(dgw, dsf, wgw, wsf, wdm, wtremote=None, active=None):
    """Unmet demand per sector and the cell total (wu_nonrenewable.c:44-66).

    All [C,S] (mm).  d = (demand_surf + demand_gw) - (withdrawn_surf +
    withdrawn_gw + withdrawn_dam + withdrawn_tremote), floored at 0.  ENERGY
    and inactive sectors stay 0 and do not enter the total.  The total is
    summed in sector order, as C does.  Returns (d [C,S], total [C]).
    """
    act = _act(active, dgw)
    w = wsf + wgw + wdm
    if wtremote is not None:
        w = w + wtremote
    d = _nz((dsf + dgw) - w)
    z = torch.zeros_like(d)
    m = act.clone()
    m[..., ENE] = False
    d = torch.where(m, d, z)
    tot = torch.zeros_like(d[..., 0])
    for s in range(d.shape[-1]):
        tot = tot + d[..., s]
    return d, tot


def availability(deficit, nren_lim):
    """Cell availability max(0, NREN_LIM - deficit) (:79-84).  deficit [C]
    (mm); nren_lim scalar or [C] (mm), default Inf."""
    a = nren_lim - deficit
    return _nz(a)


def division(d, avail, d_cell, active=None):
    """Sequential division over sectors (:104-129).

    d [C,S]; avail, d_cell [C].  Sector s takes min(d_s, what is left) when
    the cell total demand is positive; the loop covers ENERGY too (d = 0).
    Returns available_nonrenew [C,S].
    """
    act = _act(active, d)
    z = torch.zeros_like(avail)
    tmp = avail
    gate = d_cell > 0
    out = []
    for s in range(d.shape[-1]):
        m = act[..., s] & gate
        take = d[..., s] >= tmp
        out.append(torch.where(m, torch.where(take, tmp, d[..., s]), z))
        nt = torch.where(take, z, tmp - d[..., s])
        tmp = torch.where(m, nt, tmp)
    return torch.stack(out, -1)


def use(d, avail, cons_frac, returned_in=None, consumed_in=None, active=None):
    """Withdrawal, consumption and returns (:150-180).

    d, avail, cons_frac [C,S].  withdrawn = avail * min(d / avail, 1) where
    avail > 0, else 0; returned += W (1 - cf); consumed += W cf, on top of
    returned_in / consumed_in (wu_var values from the other sources; default
    0).  Returns dict: withdrawn [C,S], returned, consumed [C,S], cell_returned
    [C] (this withdrawal's return only, as C's local `returned`),
    cell_withdrawn [C]; the cell sums run in sector order.
    """
    act = _act(active, d)
    z = torch.zeros_like(d)
    one = torch.ones_like(d)
    pos = avail > 0
    safe = torch.where(pos, avail, one)
    f = d / safe
    f = torch.where(f < 1, f, one)               # C macro min(frac, 1)
    w = torch.where(pos, avail * f, z)
    w = torch.where(act, w, z)
    ret_s = w * (1 - cons_frac)
    con_s = w * cons_frac
    ret0 = z if returned_in is None else returned_in
    con0 = z if consumed_in is None else consumed_in
    ret = torch.where(act, ret0 + ret_s, ret0)
    con = torch.where(act, con0 + con_s, con0)
    cr = torch.zeros_like(w[..., 0])
    cw = torch.zeros_like(cr)
    for s in range(w.shape[-1]):
        cr = cr + torch.where(act[..., s], ret_s[..., s], 0.0)
        cw = cw + torch.where(act[..., s], w[..., s], 0.0)
    return dict(withdrawn=w, returned=ret, consumed=con, cell_returned=cr,
                cell_withdrawn=cw)


def book_and_repay(deficit, w_cell, w, cons_frac, returned, active=None,
                   gwm=False):
    """Deficit booking and immediate payback (:210-252).

    deficit [C]; w_cell [C] = sum of withdrawn_nonrenew; w, cons_frac [C,S];
    returned [C] = the cell's non-renewable return (use()['cell_returned']).
    Returns (deficit, returned) where `returned` is what is left for the
    river after the payback.  gwm=True books nothing and repays nothing.
    """
    act = _act(active, w)
    z = torch.zeros_like(deficit)
    if gwm:
        return deficit, returned
    d = torch.where(w_cell > 0, _nz(deficit + w_cell), deficit)
    r = returned
    gate = returned > 0
    for s in range(w.shape[-1]):
        m = gate & act[..., s]
        avail_tmp = d
        rt = cmin(avail_tmp, w[..., s] * (1 - cons_frac[..., s]))
        nd = torch.where(avail_tmp > 0, _nz(d - rt), d)
        nr = _nz(r - rt)
        d = torch.where(m, nd, d)
        r = torch.where(m, nr, r)
    return d, r


def ring_return(dt_discharge, returned, area, dt, n_sub, uh_length,
                discharge=None, stream=None):
    """Surface return into the FUTURE ring bins (:254-305).

    dt_discharge [C,K] with K = uh_length + n_sub - 1 (m3 s-1); returned [C]
    (mm); area [C] (m2); dt model step (s); n_sub = ROUT_STEPS_PER_DAY /
    MODEL_STEPS_PER_DAY.  Bins n_sub .. K-1 receive returned/1000 * area / dt
    in proportion to their content (equal shares when they are all empty),
    clipped at 0; discharge = sum of bins < n_sub, stream = sum of the rest.
    Only cells with returned > 0 change.  discharge and stream are the sums
    of the new ring; cells whose gate is closed keep the discharge and stream
    passed in (the ring is untouched there), or the sums of the unchanged ring
    when none are passed.  Returns (dt_discharge, discharge, stream).
    """
    k = dt_discharge.shape[-1]
    if k != uh_length + n_sub - 1:
        raise ValueError('ring length %d != UH_LENGTH + n_sub - 1 = %d'
                         % (k, uh_length + n_sub - 1))
    gate = returned > 0
    fut = dt_discharge[..., n_sub:]
    avail = fut.sum(-1)
    rs = returned / MM_PER_M * area / dt
    pos = avail > 0
    safe = torch.where(pos, avail, torch.ones_like(avail))
    add = torch.where(pos[:, None], rs[:, None] * (fut / safe[:, None]),
                      (rs / (uh_length - 1))[:, None] + torch.zeros_like(fut))
    new = _nz(fut + add)
    new = torch.where(gate[:, None], new, fut)
    out = torch.cat([dt_discharge[..., :n_sub], new], -1)
    d_new = out[..., :n_sub].sum(-1)
    s_new = new.sum(-1)
    if discharge is not None:
        d_new = torch.where(gate, d_new, discharge)
    if stream is not None:
        s_new = torch.where(gate, s_new, stream)
    return out, d_new, s_new


def payback_flow(deficit, baseflow, runoff, nonrenew_runoff=False,
                 gwm=False):
    """Repay the deficit from the VIC baseflow, then runoff (rout_run.c:49-76).

    deficit, baseflow, runoff [C] (mm).  GWM: baseflow is dropped (:49-51)
    and nothing is repaid (:56).  Baseflow first: if it exceeds the deficit
    the deficit is cleared and the excess flows on, else the baseflow is
    consumed whole.  With NONRENEWABLE_RUNOFF the same is done with runoff on
    the updated deficit.  Returns (deficit, baseflow, runoff).
    """
    bf = torch.zeros_like(baseflow) if gwm else baseflow
    if gwm:
        return deficit, bf, runoff
    gate = deficit > 0
    z = torch.zeros_like(deficit)
    big = bf > deficit
    d1 = torch.where(big, z, deficit - bf)
    b1 = torch.where(big, bf - deficit, z)
    d, b, r = (torch.where(gate, d1, deficit), torch.where(gate, b1, bf),
               runoff)
    if nonrenew_runoff:
        bigr = r > d1
        d2 = torch.where(bigr, z, d1 - r)
        r2 = torch.where(bigr, r - d1, z)
        d = torch.where(gate, d2, deficit)
        r = torch.where(gate, r2, runoff)
    return d, b, r


def runoff_to_discharge(runoff, baseflow, area, dt):
    """rout_var.runoff = (in_runoff + in_baseflow) * area / (dt * 1000)
    [m3 s-1] (rout_run.c:78-81)."""
    return (runoff + baseflow) * area / (dt * MM_PER_M)


def balance_violation(d, avail, w, d_cell, avail_cell, w_cell, thresh=1e-10,
                      active=None):
    """The two conditions of check_water_use_balance_nonrenew (:324-353) as a
    bool [C]: any sector with withdrawn - demand > thresh or withdrawn -
    available > thresh, or the cell with thresh * NWUTYPES.  C stops the run
    (log_err); a device-side flag lets the caller assert once per window."""
    act = _act(active, d)
    s = ((w - d > thresh) | (w - avail > thresh)) & act
    n = d.shape[-1]
    c = (w_cell - d_cell > thresh * n) | (w_cell - avail_cell > thresh * n)
    return s.any(-1) | c


def conservation_residual(deficit0, deficit1, w_cell, paid_back):
    """Identity of plugin_store_error.c:96-100 for the virtual aquifer:
    (deficit1 - deficit0) - (sum W - sum payback); 0 up to rounding.
    paid_back [C] is what the deficit lost to baseflow, runoff and the
    immediate return payback."""
    return (deficit1 - deficit0) - (w_cell - paid_back)


def nonrenew_step(dgw, dsf, wgw, wsf, wdm, cons_frac, deficit, nren_lim=None,
                  active=None, wtremote=None, returned_in=None,
                  consumed_in=None, gwm=False, dt_discharge=None, area=None,
                  dt=None, n_sub=1, uh_length=None, discharge=None,
                  stream=None):
    """The whole wu_nonrenew(iCell) of wu_nonrenewable.c:359-407 (P3).

    Inputs dgw, dsf, wgw, wsf, wdm [C,S] are wu_var after the local and
    remote passes; deficit [C] is the deficit after this step's payback from
    baseflow (payback_flow).  nren_lim defaults to Inf.  The ring arguments
    are optional; without them the surface return is only reported.

    Returns a dict: demand_nonrenew, available_nonrenew, withdrawn_nonrenew
    [C,S] (OUT_DE_NREN_SECT, -, OUT_WI_NREN_SECT), returned, consumed [C,S]
    (wu_var, now including the non-renewable part), deficit [C]
    (OUT_NONREN_DEFICIT at the end of the step), returned_to_river [C] (mm),
    cell_withdrawn, cell_demand, cell_available [C], violation [C] bool and,
    with a ring, dt_discharge [C,K], discharge, stream.
    """
    act = _act(active, dgw)
    if nren_lim is None:
        nren_lim = float('inf')
    d, dcell = demand(dgw, dsf, wgw, wsf, wdm, wtremote, act)
    a_cell = availability(deficit, nren_lim)
    av = division(d, a_cell, dcell, act)
    u = use(d, av, cons_frac, returned_in, consumed_in, act)
    dfc, r = book_and_repay(deficit, u['cell_withdrawn'], u['withdrawn'],
                            cons_frac, u['cell_returned'], act, gwm)
    out = dict(demand_nonrenew=d, available_nonrenew=av,
               withdrawn_nonrenew=u['withdrawn'], returned=u['returned'],
               consumed=u['consumed'], deficit=dfc, returned_to_river=r,
               cell_withdrawn=u['cell_withdrawn'], cell_demand=dcell,
               cell_available=a_cell)
    out['violation'] = balance_violation(d, av, u['withdrawn'], dcell,
                                         a_cell, u['cell_withdrawn'],
                                         active=act)
    if dt_discharge is not None:
        ring, dis, st = ring_return(dt_discharge, r, area, dt, n_sub,
                                    uh_length, discharge, stream)
        out.update(dt_discharge=ring, discharge=dis, stream=st)
    return out
