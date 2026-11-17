# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR environmental flow requirements (C9): pure tensor kernels.

Batched over cells (C), tiles (V), snow bands (B) and frost areas (F); no
loop over cells and no host sync.  The only Python loops run over the tiles
and bands (to keep the C summation order of `calculated_baseflow`) and over
the frost areas.  The 100-point search grid is evaluated for all grid points
at once ([C,G]).

C references are VIC-WUR 3.0 plugins/efr
(<data root>/build/vicwur3/src_tree/vic/plugins/efr; byte-identical in the
VIC-WUR git master):
  efr_force.c:64-69    last-layer constants res, max_moist, dsmax
  efr_force.c:71-110   grid search over frac = 1.0, 0.99, ... for the fraction
                       of the liquid water whose trial baseflow stays below
                       the required baseflow
  efr_force.c:72-103   trial baseflow, summed over ALL tiles (Cv = 0 is not
                       skipped) and bands with weight Cv * AreaFract
  efr_force.c:106-109  first frac with calculated_baseflow < req, then
                       frac = min(1, frac + 0.01)
  efr_force.c:112-125  efr_moist = liq * frac + ice
  efr_force.c:194      the whole of it runs at the START of the step (F phase)
  efr_output.c:33-128  OUT_EFR_DISCHARGE (AVG), OUT_EFR_BASEFLOW (SUM),
                       OUT_EFR_MOIST (END): sum over tiles/bands of
                       moist * Cv * AreaFract (efr_put_data, :107-120)
Consumers (the kernels are in wateruse.py, this module only gives them the
terms):
  wu_run_local.c:110-111  av_gw -= efr_moist[tile][band]
  wu_run_local.c:147-149  A_sf *= 1 - min(efr_q / discharge, 1)
  wu_remote.c:229-231     A_remote *= 1 - min(efr_q / discharge, 1)
  dam_register.c:241      dam efr = efr_q * dt * inflow_frac / 1e6 (hm3)

Two C quirks are kept in `exact` mode:
  * the non-linear baseflow term uses `soil_con->Ws`, `->Ds` and `->c`, that
    is cell 0 of the local domain, not cell i (efr_force.c:94-98).  `bug`
    selects element 0 of Ds, Ws, c along the cell axis.
  * efr_moist = liq * frac + ice keeps the ice, which water use subtracts
    again (wu_run_local.c:111,121-126).  This is a consumer matter and is
    not changed here.
The forcing baseflow `req` is mm per MODEL STEP (the paper says m3/s).

The C macro `min(a, b)` is `(a < b) ? a : b` (vic_def.h:91).
"""
import collections

import torch

MM_PER_M = 1000.0
FRAC_STEP = 0.01        # efr_force.c:71 and :107

_GRID = {}


def search_grid(device='cpu', dtype=torch.float64):
    """The frac sequence of `for (frac = 1.0; frac >= 0.0; frac -= 0.01)`.

    Returns (grid [G], last): the G values that are evaluated (frac_0 = 1,
    frac_{k+1} = frac_k - 0.01 in double, sequential; G = 100) and the value
    the loop variable holds when it falls through (about -7.5e-16).  A
    constant of the loop, built once on the host from Python floats.
    """
    key = (str(device), dtype)
    if key not in _GRID:
        vals, f = [], 1.0
        while f >= 0.0:
            vals.append(f)
            f -= FRAC_STEP
        _GRID[key] = (torch.tensor(vals, dtype=dtype, device=device),
                      torch.tensor(f, dtype=dtype, device=device))
    return _GRID[key]


def cmin(a, b):
    """C macro min(a, b) = (a < b) ? a : b."""
    return torch.where(a < b, a, b)


def ice_sum(ice, frost_fract):
    """sum_f ice[f] * frost_fract[f], in frost order (efr_force.c:80-84).

    ice [C,V,B,F]; frost_fract [C,F] -> [C,V,B].
    """
    tot = torch.zeros_like(ice[..., 0])
    for f in range(ice.shape[-1]):
        tot = tot + ice[..., f] * frost_fract[:, None, None, f]
    return tot


def trial_baseflow(liq, frac, res, max_moist, dsmax, ds, ws, c, cv,
                   area_fract, valid=None, bug=False):
    """calculated_baseflow(frac) of efr_force.c:72-103.

    liq [C,V,B] liquid water of the last layer; frac [G] grid; res,
    max_moist, dsmax, ds, ws, c [C]; cv [C,V]; area_fract [C,B]; valid
    [C,V] bool = tile index < nv_active (padded tiles are not visited; Cv = 0
    tiles are).  bug=True: the non-linear term reads cell 0 of ds, ws, c.
    Returns [C,G].  Sequential over tiles then bands, as the C loops.
    """
    nc, nv, nb = liq.shape
    g = frac[None, :]
    ds0, ws0, c0 = ((ds[:1], ws[:1], c[:1]) if bug else (ds, ws, c))
    ds0, ws0, c0 = ds0[:, None], ws0[:, None], c0[:, None]
    res_, mm_, dm_ = res[:, None], max_moist[:, None], dsmax[:, None]
    ds_, ws_ = ds[:, None], ws[:, None]
    tot = torch.zeros(nc, frac.shape[0], dtype=liq.dtype, device=liq.device)
    for j in range(nv):
        for k in range(nb):
            rel = (liq[:, j, k, None] * g - res_) / (mm_ - res_)
            bf = rel * dm_ * ds_ / ws_
            nl = dm_ * (1 - ds0 / ws0) * torch.pow(
                (rel - ws0) / (1 - ws0), c0)
            bf = bf + torch.where(rel > ws0, nl, torch.zeros_like(nl))
            bf = torch.where(bf < 0, torch.zeros_like(bf), bf)
            term = bf * cv[:, j, None] * area_fract[:, None, k]
            if valid is not None:
                term = torch.where(valid[:, j, None], term,
                                   torch.zeros_like(term))
            tot = tot + term
    return tot


def select_frac(calc, req, frac_grid, frac_last):
    """The break rule of efr_force.c:106-109 on [C,G] trial baseflows.

    First grid point with calc < req gives min(1, frac + 0.01); none gives
    the fall-through value of the loop variable (frac_last).  Returns
    (frac [C], k [C] first-violation index, G if none).
    """
    viol = calc < req[:, None]
    any_ = viol.any(1)
    k = torch.argmax(viol.to(torch.int8), 1)
    f = frac_grid[k]
    chosen = cmin(torch.ones_like(f), f + FRAC_STEP)
    frac = torch.where(any_, chosen, frac_last.expand_as(f))
    k = torch.where(any_, k, torch.full_like(k, frac_grid.shape[0]))
    return frac, k


EfrState = collections.namedtuple(
    'EfrState', 'efr_moist frac cell_moist discharge baseflow')


def efr_step(moist, ice, frost_fract, resid, depth, max_moist, dsmax_year,
             ds, ws, c, cv, area_fract, req_baseflow, req_discharge,
             steps_per_day=1, valid=None, bug=False, gchunk=None):
    """efr_derived_forcing + the outputs of efr_put_data (efr_force.c:34-126,
    efr_output.c:107-126) for every cell.

    moist [C,V,B] last-layer moisture (mm, ice included) at the START of the
    step; ice [C,V,B,F]; frost_fract [C,F]; resid, depth (m), max_moist
    (mm), dsmax_year (the soil file Dsmax, mm/day), ds, ws, c [C]; cv
    [C,V]; area_fract [C,B]; req_baseflow [C] (mm per model step);
    req_discharge [C] (m3/s); valid [C,V] (tile < nv_active).
    gchunk splits the 100 grid points into chunks to bound the [C,G,...]
    memory; the result does not depend on it.
    Returns EfrState: efr_moist [C,V,B] (mm), frac [C], cell_moist [C]
    (OUT_EFR_MOIST), discharge [C] (OUT_EFR_DISCHARGE) and baseflow [C]
    (OUT_EFR_BASEFLOW).
    """
    grid, last = search_grid(moist.device, moist.dtype)
    ice_t = ice_sum(ice, frost_fract)
    liq = moist - ice_t
    res = resid * depth * MM_PER_M
    dsmax = dsmax_year / steps_per_day
    from . import fused
    if fused.ENABLED and fused.EFR and moist.is_cuda and \
            moist.dtype == torch.float64:
        from . import efr_kernels as EK
        sl = torch.stack([torch.full_like(last, FRAC_STEP), last])
        frac = EK.search(liq, grid, sl, res, max_moist, dsmax, ds, ws, c,
                         cv, area_fract, req_baseflow, valid, bug)
        efr_moist = liq * frac[:, None, None] + ice_t
        return EfrState(efr_moist, frac, cell_moisture(
            efr_moist, cv, area_fract, valid), req_discharge, req_baseflow)
    parts = []
    step = grid.shape[0] if gchunk is None else gchunk
    for g0 in range(0, grid.shape[0], step):
        parts.append(trial_baseflow(liq, grid[g0:g0 + step], res, max_moist,
                                    dsmax, ds, ws, c, cv, area_fract, valid,
                                    bug))
    calc = torch.cat(parts, 1)
    frac, _ = select_frac(calc, req_baseflow, grid, last)
    efr_moist = liq * frac[:, None, None] + ice_t
    return EfrState(efr_moist, frac, cell_moisture(efr_moist, cv, area_fract,
                                                   valid),
                    req_discharge, req_baseflow)


def cell_moisture(efr_moist, cv, area_fract, valid=None):
    """OUT_EFR_MOIST: sum over tiles and bands of moist * Cv * AreaFract,
    tile-major (efr_output.c:111-120)."""
    tot = torch.zeros_like(efr_moist[:, 0, 0])
    for j in range(efr_moist.shape[1]):
        for k in range(efr_moist.shape[2]):
            t = efr_moist[:, j, k] * cv[:, j] * area_fract[:, k]
            if valid is not None:
                t = torch.where(valid[:, j], t, torch.zeros_like(t))
            tot = tot + t
    return tot


def surface_scale(efr_discharge, ring_discharge):
    """1 - min(efr_q / discharge, 1), the factor on A_sf and A_remote
    (wu_run_local.c:147-149, wu_remote.c:229-231).  discharge = 0 gives
    inf (or NaN) -> min gives 1 -> scale 0.  rout_var.discharge is the
    cell's current-step discharge, not the ring."""
    r = efr_discharge / ring_discharge
    return 1 - cmin(r, torch.ones_like(r))


def dam_efr(efr_discharge, dt, inflow_frac):
    """Dam EFR volume in hm3 (dam_register.c:241), the term the dams take."""
    return efr_discharge * dt * inflow_frac / 1e6
